/* Modernizr 2.6.2 (Custom Build) | MIT & BSD
 * Build: http://modernizr.com/download/#-inlinesvg-svg-svgclippaths-touch-shiv-mq-cssclasses-teststyles-prefixes-ie8compat-load
 */
;
window.Modernizr = function(a, b, c) {
    function y(a) {
        j.cssText = a
    }
    function z(a, b) {
        return y(m.join(a + ";") + (b || ""))
    }
    function A(a, b) {
        return typeof a === b
    }
    function B(a, b) {
        return !!~("" + a).indexOf(b)
    }
    function C(a, b, d) {
        for (var e in a) {
            var f = b[a[e]];
            if (f !== c)
                return d===!1 ? a[e] : A(f, "function") ? f.bind(d || b) : f
        }
        return !1
    }
    var d = "2.6.2", e = {}, f=!0, g = b.documentElement, h = "modernizr", i = b.createElement(h), j = i.style, k, l = {}.toString, m = " -webkit- -moz- -o- -ms- ".split(" "), n = {
        svg: "http://www.w3.org/2000/svg"
    }, o = {}, p = {}, q = {}, r = [], s = r.slice, t, u = function(a, c, d, e) {
        var f, i, j, k, l = b.createElement("div"), m = b.body, n = m || b.createElement("body");
        if (parseInt(d, 10))
            while (d--)
                j = b.createElement("div"), j.id = e ? e[d] : h + (d + 1), l.appendChild(j);
        return f = ["&#173;", '<style id="s', h, '">', a, "</style>"].join(""), l.id = h, (m ? l : n).innerHTML += f, n.appendChild(l), m || (n.style.background = "", n.style.overflow = "hidden", k = g.style.overflow, g.style.overflow = "hidden", g.appendChild(n)), i = c(l, a), m ? l.parentNode.removeChild(l) : (n.parentNode.removeChild(n), g.style.overflow = k), !!i
    }, v = function(b) {
        var c = a.matchMedia || a.msMatchMedia;
        if (c)
            return c(b).matches;
        var d;
        return u("@media " + b + " { #" + h + " { position: absolute; } }", function(b) {
            d = (a.getComputedStyle ? getComputedStyle(b, null) : b.currentStyle)["position"] == "absolute"
        }), d
    }, w = {}.hasOwnProperty, x;
    !A(w, "undefined")&&!A(w.call, "undefined") ? x = function(a, b) {
        return w.call(a, b)
    } : x = function(a, b) {
        return b in a && A(a.constructor.prototype[b], "undefined")
    }, Function.prototype.bind || (Function.prototype.bind = function(b) {
        var c = this;
        if (typeof c != "function")
            throw new TypeError;
        var d = s.call(arguments, 1), e = function() {
            if (this instanceof e) {
                var a = function() {};
                a.prototype = c.prototype;
                var f = new a, g = c.apply(f, d.concat(s.call(arguments)));
                return Object(g) === g ? g : f
            }
            return c.apply(b, d.concat(s.call(arguments)))
        };
        return e
    }), o.touch = function() {
        var c;
        return "ontouchstart"in a || a.DocumentTouch && b instanceof DocumentTouch ? c=!0 : u(["@media (", m.join("touch-enabled),("), h, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(a) {
            c = a.offsetTop === 9
        }), c
    }, o.svg = function() {
        return !!b.createElementNS&&!!b.createElementNS(n.svg, "svg").createSVGRect
    }, o.inlinesvg = function() {
        var a = b.createElement("div");
        return a.innerHTML = "<svg/>", (a.firstChild && a.firstChild.namespaceURI) == n.svg
    }, o.svgclippaths = function() {
        return !!b.createElementNS && /SVGClipPath/.test(l.call(b.createElementNS(n.svg, "clipPath")))
    };
    for (var D in o)
        x(o, D) && (t = D.toLowerCase(), e[t] = o[D](), r.push((e[t] ? "" : "no-") + t));
    return e.addTest = function(a, b) {
        if (typeof a == "object")
            for (var d in a)
                x(a, d) && e.addTest(d, a[d]);
        else {
            a = a.toLowerCase();
            if (e[a] !== c)
                return e;
            b = typeof b == "function" ? b() : b, typeof f != "undefined" && f && (g.className += " " + (b ? "" : "no-") + a), e[a] = b
        }
        return e
    }, y(""), i = k = null, function(a, b) {
        function k(a, b) {
            var c = a.createElement("p"), d = a.getElementsByTagName("head")[0] || a.documentElement;
            return c.innerHTML = "x<style>" + b + "</style>", d.insertBefore(c.lastChild, d.firstChild)
        }
        function l() {
            var a = r.elements;
            return typeof a == "string" ? a.split(" ") : a
        }
        function m(a) {
            var b = i[a[g]];
            return b || (b = {}, h++, a[g] = h, i[h] = b), b
        }
        function n(a, c, f) {
            c || (c = b);
            if (j)
                return c.createElement(a);
            f || (f = m(c));
            var g;
            return f.cache[a] ? g = f.cache[a].cloneNode() : e.test(a) ? g = (f.cache[a] = f.createElem(a)).cloneNode() : g = f.createElem(a), g.canHaveChildren&&!d.test(a) ? f.frag.appendChild(g) : g
        }
        function o(a, c) {
            a || (a = b);
            if (j)
                return a.createDocumentFragment();
            c = c || m(a);
            var d = c.frag.cloneNode(), e = 0, f = l(), g = f.length;
            for (; e < g; e++)
                d.createElement(f[e]);
            return d
        }
        function p(a, b) {
            b.cache || (b.cache = {}, b.createElem = a.createElement, b.createFrag = a.createDocumentFragment, b.frag = b.createFrag()), a.createElement = function(c) {
                return r.shivMethods ? n(c, a, b) : b.createElem(c)
            }, a.createDocumentFragment = Function("h,f", "return function(){var n=f.cloneNode(),c=n.createElement;h.shivMethods&&(" + l().join().replace(/\w+/g, function(a) {
                return b.createElem(a), b.frag.createElement(a), 'c("' + a + '")'
            }) + ");return n}")(r, b.frag)
        }
        function q(a) {
            a || (a = b);
            var c = m(a);
            return r.shivCSS&&!f&&!c.hasCSS && (c.hasCSS=!!k(a, "article,aside,figcaption,figure,footer,header,hgroup,nav,section{display:block}mark{background:#FF0;color:#000}")), j || p(a, c), a
        }
        var c = a.html5 || {}, d = /^<|^(?:button|map|select|textarea|object|iframe|option|optgroup)$/i, e = /^(?:a|b|code|div|fieldset|h1|h2|h3|h4|h5|h6|i|label|li|ol|p|q|span|strong|style|table|tbody|td|th|tr|ul)$/i, f, g = "_html5shiv", h = 0, i = {}, j;
        (function() {
            try {
                var a = b.createElement("a");
                a.innerHTML = "<xyz></xyz>", f = "hidden"in a, j = a.childNodes.length == 1 || function() {
                    b.createElement("a");
                    var a = b.createDocumentFragment();
                    return typeof a.cloneNode == "undefined" || typeof a.createDocumentFragment == "undefined" || typeof a.createElement == "undefined"
                }()
            } catch (c) {
                f=!0, j=!0
            }
        })();
        var r = {
            elements: c.elements || "abbr article aside audio bdi canvas data datalist details figcaption figure footer header hgroup mark meter nav output progress section summary time video",
            shivCSS: c.shivCSS!==!1,
            supportsUnknownElements: j,
            shivMethods: c.shivMethods!==!1,
            type: "default",
            shivDocument: q,
            createElement: n,
            createDocumentFragment: o
        };
        a.html5 = r, q(b)
    }(this, b), e._version = d, e._prefixes = m, e.mq = v, e.testStyles = u, g.className = g.className.replace(/(^|\s)no-js(\s|$)/, "$1$2") + (f ? " js " + r.join(" ") : ""), e
}(this, this.document), function(a, b, c) {
    function d(a) {
        return "[object Function]" == o.call(a)
    }
    function e(a) {
        return "string" == typeof a
    }
    function f() {}
    function g(a) {
        return !a || "loaded" == a || "complete" == a || "uninitialized" == a
    }
    function h() {
        var a = p.shift();
        q = 1, a ? a.t ? m(function() {
            ("c" == a.t ? B.injectCss : B.injectJs)(a.s, 0, a.a, a.x, a.e, 1)
        }, 0) : (a(), h()) : q = 0
    }
    function i(a, c, d, e, f, i, j) {
        function k(b) {
            if (!o && g(l.readyState) && (u.r = o = 1, !q && h(), l.onload = l.onreadystatechange = null, b)) {
                "img" != a && m(function() {
                    t.removeChild(l)
                }, 50);
                for (var d in y[c])
                    y[c].hasOwnProperty(d) && y[c][d].onload()
            }
        }
        var j = j || B.errorTimeout, l = b.createElement(a), o = 0, r = 0, u = {
            t: d,
            s: c,
            e: f,
            a: i,
            x: j
        };
        1 === y[c] && (r = 1, y[c] = []), "object" == a ? l.data = c : (l.src = c, l.type = a), l.width = l.height = "0", l.onerror = l.onload = l.onreadystatechange = function() {
            k.call(this, r)
        }, p.splice(e, 0, u), "img" != a && (r || 2 === y[c] ? (t.insertBefore(l, s ? null : n), m(k, j)) : y[c].push(l))
    }
    function j(a, b, c, d, f) {
        return q = 0, b = b || "j", e(a) ? i("c" == b ? v : u, a, b, this.i++, c, d, f) : (p.splice(this.i++, 0, a), 1 == p.length && h()), this
    }
    function k() {
        var a = B;
        return a.loader = {
            load: j,
            i: 0
        }, a
    }
    var l = b.documentElement, m = a.setTimeout, n = b.getElementsByTagName("script")[0], o = {}.toString, p = [], q = 0, r = "MozAppearance"in l.style, s = r&&!!b.createRange().compareNode, t = s ? l: n.parentNode, l = a.opera && "[object Opera]" == o.call(a.opera), l=!!b.attachEvent&&!l, u = r ? "object" : l ? "script" : "img", v = l ? "script" : u, w = Array.isArray || function(a) {
        return "[object Array]" == o.call(a)
    }, x = [], y = {}, z = {
        timeout: function(a, b) {
            return b.length && (a.timeout = b[0]), a
        }
    }, A, B;
    B = function(a) {
        function b(a) {
            var a = a.split("!"), b = x.length, c = a.pop(), d = a.length, c = {
                url: c,
                origUrl: c,
                prefixes: a
            }, e, f, g;
            for (f = 0; f < d; f++)
                g = a[f].split("="), (e = z[g.shift()]) && (c = e(c, g));
            for (f = 0; f < b; f++)
                c = x[f](c);
            return c
        }
        function g(a, e, f, g, h) {
            var i = b(a), j = i.autoCallback;
            i.url.split(".").pop().split("?").shift(), i.bypass || (e && (e = d(e) ? e : e[a] || e[g] || e[a.split("/").pop().split("?")[0]]), i.instead ? i.instead(a, e, f, g, h) : (y[i.url] ? i.noexec=!0 : y[i.url] = 1, f.load(i.url, i.forceCSS ||!i.forceJS && "css" == i.url.split(".").pop().split("?").shift() ? "c" : c, i.noexec, i.attrs, i.timeout), (d(e) || d(j)) && f.load(function() {
                k(), e && e(i.origUrl, h, g), j && j(i.origUrl, h, g), y[i.url] = 2
            })))
        }
        function h(a, b) {
            function c(a, c) {
                if (a) {
                    if (e(a))
                        c || (j = function() {
                            var a = [].slice.call(arguments);
                            k.apply(this, a), l()
                        }), g(a, j, b, 0, h);
                    else if (Object(a) === a)
                        for (n in m = function() {
                            var b = 0, c;
                            for (c in a)
                                a.hasOwnProperty(c) && b++;
                                return b
                            }(), a)
                                a.hasOwnProperty(n) && (!c&&!--m && (d(j) ? j = function() {
                                    var a = [].slice.call(arguments);
                                    k.apply(this, a), l()
                                } : j[n] = function(a) {
                                    return function() {
                                        var b = [].slice.call(arguments);
                                        a && a.apply(this, b), l()
                                    }
                                }(k[n])), g(a[n], j, b, n, h))
                } else 
                    !c && l()
            }
            var h=!!a.test, i = a.load || a.both, j = a.callback || f, k = j, l = a.complete || f, m, n;
            c(h ? a.yep : a.nope, !!i), i && c(i)
        }
        var i, j, l = this.yepnope.loader;
        if (e(a))
            g(a, 0, l, 0);
        else if (w(a))
            for (i = 0; i < a.length; i++)
                j = a[i], e(j) ? g(j, 0, l, 0) : w(j) ? B(j) : Object(j) === j && h(j, l);
        else 
            Object(a) === a && h(a, l)
    }, B.addPrefix = function(a, b) {
        z[a] = b
    }, B.addFilter = function(a) {
        x.push(a)
    }, B.errorTimeout = 1e4, null == b.readyState && b.addEventListener && (b.readyState = "loading", b.addEventListener("DOMContentLoaded", A = function() {
        b.removeEventListener("DOMContentLoaded", A, 0), b.readyState = "complete"
    }, 0)), a.yepnope = k(), a.yepnope.executeStack = h, a.yepnope.injectJs = function(a, c, d, e, i, j) {
        var k = b.createElement("script"), l, o, e = e || B.errorTimeout;
        k.src = a;
        for (o in d)
            k.setAttribute(o, d[o]);
        c = j ? h : c || f, k.onreadystatechange = k.onload = function() {
            !l && g(k.readyState) && (l = 1, c(), k.onload = k.onreadystatechange = null)
        }, m(function() {
            l || (l = 1, c(1))
        }, e), i ? k.onload() : n.parentNode.insertBefore(k, n)
    }, a.yepnope.injectCss = function(a, c, d, e, g, i) {
        var e = b.createElement("link"), j, c = i ? h: c || f;
        e.href = a, e.rel = "stylesheet", e.type = "text/css";
        for (j in d)
            e.setAttribute(j, d[j]);
        g || (n.parentNode.insertBefore(e, n), m(c, 0))
    }
}(this, document), Modernizr.load = function() {
    yepnope.apply(window, [].slice.call(arguments, 0))
}, Modernizr.addTest("ie8compat", function() {
    return !window.addEventListener && document.documentMode && document.documentMode === 7
});


/*
 * Foundation Responsive Library
 * http://foundation.zurb.com
 * Copyright 2013, ZURB
 * Free to use under the MIT license.
 * http://www.opensource.org/licenses/mit-license.php
*/
/*jslint unparam: true, browser: true, indent: 2 */
// Accommodate running jQuery or Zepto in noConflict() mode by
// using an anonymous function to redefine the $ shorthand name.
// See http://docs.jquery.com/Using_jQuery_with_Other_Libraries
// and http://zeptojs.com/
var libFuncName = null;
if (typeof jQuery == "undefined" && typeof Zepto == "undefined" && typeof $ == "function")
    libFuncName = $;
else if (typeof jQuery == "function")
    libFuncName = jQuery;
else {
    if (typeof Zepto != "function")
        throw new TypeError;
    libFuncName = Zepto
}(function(e) {
    (function() {
        Array.prototype.filter || (Array.prototype.filter = function(e) {
            "use strict";
            if (this == null)
                throw new TypeError;
            var t = Object(this), n = t.length>>>0;
            if (typeof e != "function")
                try {
                    throw new TypeError
            } catch (r) {
                return 
            }
            var i = [], s = arguments[1];
            for (var o = 0; o < n; o++)
                if (o in t) {
                    var u = t[o];
                    e && e.call(s, u, o, t) && i.push(u)
                }
            return i
        }, Function.prototype.bind || (Function.prototype.bind = function(e) {
            if (typeof this != "function")
                throw new TypeError("Function.prototype.bind - what is trying to be bound is not callable");
            var t = Array.prototype.slice.call(arguments, 1), n = this, r = function() {}, i = function() {
                return n.apply(this instanceof r && e ? this : e, t.concat(Array.prototype.slice.call(arguments)))
            };
            return r.prototype = this.prototype, i.prototype = new r, i
        })), e.fn.stop = e.fn.stop || function() {
            return this
        }
    })(), function(t, n, r) {
        "use strict";
        t.Foundation = {
            name: "Foundation",
            version: "4.1.0",
            cache: {},
            init: function(t, n, r, i, s, o) {
                var u, a = [t, r, i, s], f = [], o = o ||!1;
                o && (this.nc = o), this.rtl = /rtl/i.test(e("html").attr("dir")), this.scope = t || this.scope;
                if (n && typeof n == "string") {
                    if (/off/i.test(n))
                        return this.off();
                    u = n.split(" ");
                    if (u.length > 0)
                        for (var l = u.length - 1; l >= 0; l--)
                            f.push(this.init_lib(u[l], a))
                } else 
                    for (var c in this.libs)
                        f.push(this.init_lib(c, a));
                return typeof n == "function" && a.unshift(n), this.response_obj(f, a)
            },
            response_obj: function(e, t) {
                for (var n = 0, r = t.length; n < r; n++)
                    if (typeof t[n] == "function")
                        return t[n]({
                            errors: e.filter(function(e) {
                                if (typeof e == "string")
                                    return e
                                })
                            });
                return e
            },
            init_lib: function(e, t) {
                return this.trap(function() {
                    if (this.libs.hasOwnProperty(e))
                        return this.patch(this.libs[e]), this.libs[e].init.apply(this.libs[e], t)
                }.bind(this), e)
            },
            trap: function(e, t) {
                if (!this.nc)
                    try {
                        return e()
                } catch (n) {
                    return this.error({
                        name: t,
                        message: "could not be initialized",
                        more: n.name + " " + n.message
                    })
                }
                return e()
            },
            patch: function(e) {
                this.fix_outer(e), e.scope = this.scope, e.rtl = this.rtl
            },
            inherit: function(e, t) {
                var n = t.split(" ");
                for (var r = n.length - 1; r >= 0; r--)
                    this.lib_methods.hasOwnProperty(n[r]) && (this.libs[e.name][n[r]] = this.lib_methods[n[r]])
            },
            random_str: function(e) {
                var t = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghiklmnopqrstuvwxyz".split("");
                e || (e = Math.floor(Math.random() * t.length));
                var n = "";
                for (var r = 0; r < e; r++)
                    n += t[Math.floor(Math.random() * t.length)];
                return n
            },
            libs: {},
            lib_methods: {
                set_data: function(e, t) {
                    var n = [this.name, + (new Date), Foundation.random_str(5)].join("-");
                    return Foundation.cache[n] = t, e.attr("data-" + this.name + "-id", n), t
                },
                get_data: function(e) {
                    return Foundation.cache[e.attr("data-" + this.name + "-id")]
                },
                remove_data: function(t) {
                    t ? (delete Foundation.cache[t.attr("data-" + this.name + "-id")], t.attr("data-" + this.name + "-id", "")) : e("[data-" + this.name + "-id]").each(function() {
                        delete Foundation.cache[e(this).attr("data-" + this.name + "-id")], e(this).attr("data-" + this.name + "-id", "")
                    })
                },
                throttle: function(e, t) {
                    var n = null;
                    return function() {
                        var r = this, i = arguments;
                        clearTimeout(n), n = setTimeout(function() {
                            e.apply(r, i)
                        }, t)
                    }
                },
                data_options: function(t) {
                    function u(e) {
                        return !isNaN(e - 0) && e !== null && e !== "" && e!==!1 && e!==!0
                    }
                    function a(t) {
                        return typeof t == "string" ? e.trim(t) : t
                    }
                    var n = {}, r, i, s = (t.attr("data-options") || ":").split(";"), o = s.length;
                    for (r = o - 1; r >= 0; r--)
                        i = s[r].split(":"), /true/i.test(i[1]) && (i[1]=!0), /false/i.test(i[1]) && (i[1]=!1), u(i[1]) && (i[1] = parseInt(i[1], 10)), i.length === 2 && i[0].length > 0 && (n[a(i[0])] = a(i[1]));
                    return n
                },
                delay: function(e, t) {
                    return setTimeout(e, t)
                },
                scrollTo: function(n, r, i) {
                    if (i < 0)
                        return;
                    var s = r - e(t).scrollTop(), o = s / i * 10;
                    this.scrollToTimerCache = setTimeout(function() {
                        isNaN(parseInt(o, 10)) || (t.scrollTo(0, e(t).scrollTop() + o), this.scrollTo(n, r, i - 10))
                    }.bind(this), 10)
                },
                scrollLeft: function(e) {
                    if (!e.length)
                        return;
                    return "scrollLeft"in e[0] ? e[0].scrollLeft : e[0].pageXOffset
                },
                empty: function(e) {
                    if (e.length && e.length > 0)
                        return !1;
                    if (e.length && e.length === 0)
                        return !0;
                    for (var t in e)
                        if (hasOwnProperty.call(e, t))
                            return !1;
                    return !0
                }
            },
            fix_outer: function(e) {
                e.outerHeight = function(e, t) {
                    return typeof Zepto == "function" ? e.height() : typeof t != "undefined" ? e.outerHeight(t) : e.outerHeight()
                }, e.outerWidth = function(e) {
                    return typeof Zepto == "function" ? e.width() : typeof bool != "undefined" ? e.outerWidth(bool) : e.outerWidth()
                }
            },
            error: function(e) {
                return e.name + " " + e.message + "; " + e.more
            },
            off: function() {
                return e(this.scope).off(".fndtn"), e(t).off(".fndtn"), !0
            },
            zj: function() {
                try {
                    return Zepto
                } catch (e) {
                    return jQuery
                }
            }()
        }, e.fn.foundation = function() {
            var e = Array.prototype.slice.call(arguments, 0);
            return this.each(function() {
                return Foundation.init.apply(Foundation, [this].concat(e)), this
            })
        }
    }(this, this.document)
})(libFuncName), function(e, t, n, r) {
    "use strict";
    Foundation.libs.clearing = {
        name: "clearing",
        version: "4.1.2",
        settings: {
            templates: {
                viewing: '<a href="#" class="clearing-close">&times;</a><div class="visible-img" style="display: none"><img src="//:0"><p class="clearing-caption"></p><a href="#" class="clearing-main-prev"><span></span></a><a href="#" class="clearing-main-next"><span></span></a></div>'
            },
            close_selectors: ".clearing-close",
            init: !1,
            locked: !1
        },
        init: function(t, n, r) {
            var i = this;
            return Foundation.inherit(this, "set_data get_data remove_data throttle data_options"), typeof n == "object" && (r = e.extend(!0, this.settings, n)), typeof n != "string" ? (e(this.scope).find("ul[data-clearing]").each(function() {
                var t = e(this), n = n || {}, r = t.find("li"), s = i.get_data(t);
                !s && r.length > 0 && (n.$parent = t.parent(), i.set_data(t, e.extend({}, i.settings, n, i.data_options(t))), i.assemble(t.find("li")), i.settings.init || i.events().swipe_events())
            }), this.settings.init) : this[n].call(this, r)
        },
        events: function() {
            var n = this;
            return e(this.scope).on("click.fndtn.clearing", "ul[data-clearing] li", function(t, r, i) {
                var r = r || e(this), i = i || r, s = r.next("li"), o = n.get_data(r.parent()), u = e(t.target);
                t.preventDefault(), o || n.init(), i.hasClass("visible") && r[0] === i[0] && s.length > 0 && n.is_open(r) && (i = s, u = i.find("img")), n.open(u, r, i), n.update_paddles(i)
            }).on("click.fndtn.clearing", ".clearing-main-next", function(e) {
                this.nav(e, "next")
            }.bind(this)).on("click.fndtn.clearing", ".clearing-main-prev", function(e) {
                this.nav(e, "prev")
            }.bind(this)).on("click.fndtn.clearing", this.settings.close_selectors, function(e) {
                Foundation.libs.clearing.close(e, this)
            }).on("keydown.fndtn.clearing", function(e) {
                this.keydown(e)
            }.bind(this)), e(t).on("resize.fndtn.clearing", function(e) {
                this.resize()
            }.bind(this)), this.settings.init=!0, this
        },
        swipe_events: function() {
            var t = this;
            e(this.scope).on("touchstart.fndtn.clearing", ".visible-img", function(t) {
                t.touches || (t = t.originalEvent);
                var n = {
                    start_page_x: t.touches[0].pageX,
                    start_page_y: t.touches[0].pageY,
                    start_time: (new Date).getTime(),
                    delta_x: 0,
                    is_scrolling: r
                };
                e(this).data("swipe-transition", n), t.stopPropagation()
            }).on("touchmove.fndtn.clearing", ".visible-img", function(n) {
                n.touches || (n = n.originalEvent);
                if (n.touches.length > 1 || n.scale && n.scale !== 1)
                    return;
                var r = e(this).data("swipe-transition");
                typeof r == "undefined" && (r = {}), r.delta_x = n.touches[0].pageX - r.start_page_x, typeof r.is_scrolling == "undefined" && (r.is_scrolling=!!(r.is_scrolling || Math.abs(r.delta_x) < Math.abs(n.touches[0].pageY - r.start_page_y)));
                if (!r.is_scrolling&&!r.active) {
                    n.preventDefault();
                    var i = r.delta_x < 0 ? "next": "prev";
                    r.active=!0, t.nav(n, i)
                }
            }).on("touchend.fndtn.clearing", ".visible-img", function(t) {
                e(this).data("swipe-transition", {}), t.stopPropagation()
            })
        },
        assemble: function(t) {
            var n = t.parent();
            n.after('<div id="foundationClearingHolder"></div>');
            var r = e("#foundationClearingHolder"), i = this.get_data(n), s = n.detach(), o = {
                grid: '<div class="carousel">' + this.outerHTML(s[0]) + "</div>",
                viewing: i.templates.viewing
            }, u = '<div class="clearing-assembled"><div>' + o.viewing + o.grid + "</div></div>";
            return r.after(u).remove()
        },
        open: function(e, t, n) {
            var r = n.closest(".clearing-assembled"), i = r.find("div").first(), s = i.find(".visible-img"), o = s.find("img").not(e);
            this.locked() || (o.attr("src", this.load(e)).css("visibility", "hidden"), this.loaded(o, function() {
                o.css("visibility", "visible"), r.addClass("clearing-blackout"), i.addClass("clearing-container"), s.show(), this.fix_height(n).caption(s.find(".clearing-caption"), e).center(o).shift(t, n, function() {
                    n.siblings().removeClass("visible"), n.addClass("visible")
                })
            }.bind(this)))
        },
        close: function(t, n) {
            t.preventDefault();
            var r = function(e) {
                return /blackout/.test(e.selector) ? e : e.closest(".clearing-blackout")
            }(e(n)), i, s;
            return n === t.target && r && (i = r.find("div").first(), s = i.find(".visible-img"), this.settings.prev_index = 0, r.find("ul[data-clearing]").attr("style", "").closest(".clearing-blackout").removeClass("clearing-blackout"), i.removeClass("clearing-container"), s.hide()), !1
        },
        is_open: function(e) {
            return e.parent().attr("style").length > 0
        },
        keydown: function(t) {
            var n = e(".clearing-blackout").find("ul[data-clearing]");
            t.which === 39 && this.go(n, "next"), t.which === 37 && this.go(n, "prev"), t.which === 27 && e("a.clearing-close").trigger("click")
        },
        nav: function(t, n) {
            var r = e(".clearing-blackout").find("ul[data-clearing]");
            t.preventDefault(), this.go(r, n)
        },
        resize: function() {
            var t = e(".clearing-blackout .visible-img").find("img");
            t.length && this.center(t)
        },
        fix_height: function(t) {
            var n = t.parent().children(), r = this;
            return n.each(function() {
                var t = e(this), n = t.find("img");
                t.height() > r.outerHeight(n) && t.addClass("fix-height")
            }).closest("ul").width(n.length * 100 + "%"), this
        },
        update_paddles: function(e) {
            var t = e.closest(".carousel").siblings(".visible-img");
            e.next().length > 0 ? t.find(".clearing-main-next").removeClass("disabled") : t.find(".clearing-main-next").addClass("disabled"), e.prev().length > 0 ? t.find(".clearing-main-prev").removeClass("disabled") : t.find(".clearing-main-prev").addClass("disabled")
        },
        center: function(e) {
            return this.rtl ? e.css({
                marginRight: - (this.outerWidth(e) / 2),
                marginTop: - (this.outerHeight(e) / 2)
            }) : e.css({
                marginLeft: - (this.outerWidth(e) / 2),
                marginTop: - (this.outerHeight(e) / 2)
            }), this
        },
        load: function(e) {
            var t = e.parent().attr("href");
            return this.preload(e), t ? t : e.attr("src")
        },
        preload: function(e) {
            this.img(e.closest("li").next()).img(e.closest("li").prev())
        },
        loaded: function(e, t) {
            function n() {
                t()
            }
            function r() {
                this.one("load", n);
                if (/MSIE (\d+\.\d+);/.test(navigator.userAgent)) {
                    var e = this.attr("src"), t = e.match(/\?/) ? "&": "?";
                    t += "random=" + (new Date).getTime(), this.attr("src", e + t)
                }
            }
            if (!e.attr("src")) {
                n();
                return 
            }
            e[0].complete || e[0].readyState === 4 ? n() : r.call(e)
        },
        img: function(e) {
            if (e.length) {
                var t = new Image, n = e.find("a");
                n.length ? t.src = n.attr("href") : t.src = e.find("img").attr("src")
            }
            return this
        },
        caption: function(e, t) {
            var n = t.data("caption");
            return n ? e.text(n).show() : e.text("").hide(), this
        },
        go: function(e, t) {
            var n = e.find(".visible"), r = n[t]();
            r.length && r.find("img").trigger("click", [n, r])
        },
        shift: function(e, t, n) {
            var r = t.parent(), i = this.settings.prev_index || t.index(), s = this.direction(r, e, t), o = parseInt(r.css("left"), 10), u = this.outerWidth(t), a;
            t.index() !== i&&!/skip/.test(s) ? /left/.test(s) ? (this.lock(), r.animate({
                left: o + u
            }, 300, this.unlock())) : /right/.test(s) && (this.lock(), r.animate({
                left: o - u
            }, 300, this.unlock())) : /skip/.test(s) && (a = t.index() - this.settings.up_count, this.lock(), a > 0 ? r.animate({
                left: - (a * u)
            }, 300, this.unlock()) : r.animate({
                left: 0
            }, 300, this.unlock())), n()
        },
        direction: function(t, n, r) {
            var i = t.find("li"), s = this.outerWidth(i) + this.outerWidth(i) / 4, o = Math.floor(this.outerWidth(e(".clearing-container")) / s) - 1, u = i.index(r), a;
            return this.settings.up_count = o, this.adjacent(this.settings.prev_index, u) ? u > o && u > this.settings.prev_index ? a = "right" : u > o - 1 && u <= this.settings.prev_index ? a = "left" : a=!1 : a = "skip", this.settings.prev_index = u, a
        },
        adjacent: function(e, t) {
            for (var n = t + 1; n >= t - 1; n--)
                if (n === e)
                    return !0;
            return !1
        },
        lock: function() {
            this.settings.locked=!0
        },
        unlock: function() {
            this.settings.locked=!1
        },
        locked: function() {
            return this.settings.locked
        },
        outerHTML: function(e) {
            return e.outerHTML || (new XMLSerializer).serializeToString(e)
        },
        off: function() {
            e(this.scope).off(".fndtn.clearing"), e(t).off(".fndtn.clearing"), this.remove_data(), this.settings.init=!1
        },
        reflow: function() {
            this.init()
        }
    }
}(Foundation.zj, this, this.document), function(e, t, n, r) {
    "use strict";
    Foundation.libs.topbar = {
        name: "topbar",
        version: "4.1.2",
        settings: {
            index: 0,
            stickyClass: "sticky",
            custom_back_text: !0,
            back_text: "Back",
            init: !1
        },
        init: function(n, r, i) {
            var s = this;
            return typeof r == "object" && e.extend(!0, this.settings, r), typeof r != "string" ? (e(".top-bar").each(function() {
                s.settings.$w = e(t), s.settings.$topbar = e(this), s.settings.$section = s.settings.$topbar.find("section"), s.settings.$titlebar = s.settings.$topbar.children("ul").first(), s.settings.$topbar.data("index", 0);
                var n = e("<div class='top-bar-js-breakpoint'/>").insertAfter(s.settings.$topbar);
                s.settings.breakPoint = n.width(), n.remove(), s.assemble(), s.settings.$topbar.parent().hasClass("fixed") && e("body").css("padding-top", s.outerHeight(s.settings.$topbar))
            }), s.settings.init || this.events(), this.settings.init) : this[r].call(this, i)
        },
        events: function() {
            var n = this, r = this.outerHeight(e(".top-bar"));
            e(this.scope).on("click.fndtn.topbar", ".top-bar .toggle-topbar", function(i) {
                var s = e(this).closest(".top-bar"), o = s.find("section, .section"), u = s.children("ul").first();
                s.data("height") || n.largestUL(), i.preventDefault(), n.breakpoint() && s.toggleClass("expanded").css("min-height", ""), s.hasClass("expanded") ? s.parent().hasClass("fixed") && (s.parent().removeClass("fixed"), s.addClass("fixed"), e("body").css("padding-top", "0"), t.scrollTo(0, 0)) : (n.rtl ? (o.css({
                    right: "0%"
                }), o.find(">.name").css({
                    right: "100%"
                })) : (o.css({
                    left: "0%"
                }), o.find(">.name").css({
                    left: "100%"
                })), o.find("li.moved").removeClass("moved"), s.data("index", 0), s.hasClass("fixed") && (s.parent().addClass("fixed"), s.removeClass("fixed"), e("body").css("padding-top", r)))
            }).on("click.fndtn.topbar", ".top-bar .has-dropdown>a", function(t) {
                var r = e(this).closest(".top-bar"), i = r.find("section, .section"), s = r.children("ul").first();
                (Modernizr.touch || n.breakpoint()) && t.preventDefault();
                if (n.breakpoint()) {
                    var o = e(this), u = o.closest("li");
                    r.data("index", r.data("index") + 1), u.addClass("moved"), n.rtl ? (i.css({
                        right: - (100 * r.data("index")) + "%"
                    }), i.find(">.name").css({
                        right: 100 * r.data("index") + "%"
                    })) : (i.css({
                        left: - (100 * r.data("index")) + "%"
                    }), i.find(">.name").css({
                        left: 100 * r.data("index") + "%"
                    })), o.siblings("ul").height(r.data("height") + n.outerHeight(s, !0)), r.css("min-height", r.data("height") + n.outerHeight(s, !0) * 2)
                }
            }), e(t).on("resize.fndtn.topbar", function() {
                n.breakpoint() || e(".top-bar").css("min-height", "").removeClass("expanded")
            }.bind(this)), e(this.scope).on("click.fndtn", ".top-bar .has-dropdown .back", function(t) {
                t.preventDefault();
                var r = e(this), i = r.closest(".top-bar"), s = i.find("section, .section"), o = r.closest("li.moved"), u = o.parent();
                i.data("index", i.data("index") - 1), n.rtl ? (s.css({
                    right: - (100 * i.data("index")) + "%"
                }), s.find(">.name").css({
                    right: 100 * i.data("index") + "%"
                })) : (s.css({
                    left: - (100 * i.data("index")) + "%"
                }), s.find(">.name").css({
                    left: 100 * i.data("index") + "%"
                })), i.data("index") === 0 && i.css("min-height", 0), setTimeout(function() {
                    o.removeClass("moved")
                }, 300)
            })
        },
        breakpoint: function() {
            return e(t).width() <= this.settings.breakPoint || e("html").hasClass("lt-ie9")
        },
        assemble: function() {
            var t = this;
            this.settings.$section.detach(), this.settings.$section.find(".has-dropdown>a").each(function() {
                var n = e(this), r = n.siblings(".dropdown"), i = e('<li class="title back js-generated"><h5><a href="#"></a></h5></li>');
                t.settings.custom_back_text == 1 ? i.find("h5>a").html("&laquo; " + t.settings.back_text) : i.find("h5>a").html("&laquo; " + n.html()), r.prepend(i)
            }), this.settings.$section.appendTo(this.settings.$topbar), this.sticky()
        },
        largestUL: function() {
            var t = this.settings.$topbar.find("section ul ul"), n = t.first(), r = 0, i = this;
            t.each(function() {
                e(this).children("li").length > n.children("li").length && (n = e(this))
            }), n.children("li").each(function() {
                r += i.outerHeight(e(this), !0)
            }), this.settings.$topbar.data("height", r)
        },
        sticky: function() {
            var n = "." + this.settings.stickyClass;
            if (e(n).length > 0) {
                var r = e(n).length ? e(n).offset().top: 0, i = e(t), s = this.outerHeight(e(".top-bar"));
                i.scroll(function() {
                    i.scrollTop() >= r ? (e(n).addClass("fixed"), e("body").css("padding-top", s)) : i.scrollTop() < r && (e(n).removeClass("fixed"), e("body").css("padding-top", "0"))
                })
            }
        },
        off: function() {
            e(this.scope).off(".fndtn.topbar"), e(t).off(".fndtn.topbar")
        }
    }
}(Foundation.zj, this, this.document);


/**
* hoverIntent r6 // 2011.02.26 // jQuery 1.5.1+
* <http://cherne.net/brian/resources/jquery.hoverIntent.html>
*
* @param  f  onMouseOver function || An object with configuration options
* @param  g  onMouseOut function  || Nothing (use configuration options object)
* @author    Brian Cherne brian(at)cherne(dot)net
*/
(function($) {
    $.fn.hoverIntent = function(f, g) {
        var cfg = {
            sensitivity: 7,
            interval: 100,
            timeout: 0
        };
        cfg = $.extend(cfg, g ? {
            over: f,
            out: g
        } : f);
        var cX, cY, pX, pY;
        var track = function(ev) {
            cX = ev.pageX;
            cY = ev.pageY
        };
        var compare = function(ev, ob) {
            ob.hoverIntent_t = clearTimeout(ob.hoverIntent_t);
            if ((Math.abs(pX - cX) + Math.abs(pY - cY)) < cfg.sensitivity) {
                $(ob).unbind("mousemove", track);
                ob.hoverIntent_s = 1;
                return cfg.over.apply(ob, [ev])
            } else {
                pX = cX;
                pY = cY;
                ob.hoverIntent_t = setTimeout(function() {
                    compare(ev, ob)
                }, cfg.interval)
            }
        };
        var delay = function(ev, ob) {
            ob.hoverIntent_t = clearTimeout(ob.hoverIntent_t);
            ob.hoverIntent_s = 0;
            return cfg.out.apply(ob, [ev])
        };
        var handleHover = function(e) {
            var ev = jQuery.extend({}, e);
            var ob = this;
            if (ob.hoverIntent_t) {
                ob.hoverIntent_t = clearTimeout(ob.hoverIntent_t)
            }
            if (e.type == "mouseenter") {
                pX = ev.pageX;
                pY = ev.pageY;
                $(ob).bind("mousemove", track);
                if (ob.hoverIntent_s != 1) {
                    ob.hoverIntent_t = setTimeout(function() {
                        compare(ev, ob)
                    }, cfg.interval)
                }
            } else {
                $(ob).unbind("mousemove", track);
                if (ob.hoverIntent_s == 1) {
                    ob.hoverIntent_t = setTimeout(function() {
                        delay(ev, ob)
                    }, cfg.timeout)
                }
            }
        };
        return this.bind('mouseenter', handleHover).bind('mouseleave', handleHover)
    }
})(jQuery);

/**
 * BxSlider v4.1 - Fully loaded, responsive content slider
 * http://bxslider.com
 *
 * Copyright 2012, Steven Wanderski - http://stevenwanderski.com - http://bxcreative.com
 * Written while drinking Belgian ales and listening to jazz
 *
 * Released under the WTFPL license - http://sam.zoy.org/wtfpl/
 */
(function(e) {
    var t = {}, n = {
        mode: "horizontal",
        slideSelector: "",
        infiniteLoop: !0,
        hideControlOnEnd: !1,
        speed: 500,
        easing: null,
        slideMargin: 0,
        startSlide: 0,
        randomStart: !1,
        captions: !1,
        ticker: !1,
        tickerHover: !1,
        adaptiveHeight: !1,
        adaptiveHeightSpeed: 500,
        video: !1,
        useCSS: !0,
        preloadImages: "visible",
        touchEnabled: !0,
        swipeThreshold: 50,
        oneToOneTouch: !0,
        preventDefaultSwipeX: !0,
        preventDefaultSwipeY: !1,
        pager: !0,
        pagerType: "full",
        pagerShortSeparator: " / ",
        pagerSelector: null,
        buildPager: null,
        pagerCustom: null,
        controls: !0,
        nextText: "Next",
        prevText: "Prev",
        nextSelector: null,
        prevSelector: null,
        autoControls: !1,
        startText: "Start",
        stopText: "Stop",
        autoControlsCombine: !1,
        autoControlsSelector: null,
        auto: !1,
        pause: 4e3,
        autoStart: !0,
        autoDirection: "next",
        autoHover: !1,
        autoDelay: 0,
        minSlides: 1,
        maxSlides: 1,
        moveSlides: 0,
        slideWidth: 0,
        onSliderLoad: function() {},
        onSlideBefore: function() {},
        onSlideAfter: function() {},
        onSlideNext: function() {},
        onSlidePrev: function() {}
    };
    e.fn.bxSlider = function(s) {
        if (0 != this.length) {
            if (this.length > 1)
                return this.each(function() {
                    e(this).bxSlider(s)
                }), this;
            var o = {}, r = this;
            t.el = this;
            var a = e(window).width(), l = e(window).height(), d = function() {
                o.settings = e.extend({}, n, s), o.settings.slideWidth = parseInt(o.settings.slideWidth), o.children = r.children(o.settings.slideSelector), o.children.length < o.settings.minSlides && (o.settings.minSlides = o.children.length), o.children.length < o.settings.maxSlides && (o.settings.maxSlides = o.children.length), o.settings.randomStart && (o.settings.startSlide = Math.floor(Math.random() * o.children.length)), o.active = {
                    index: o.settings.startSlide
                }, o.carousel = o.settings.minSlides > 1 || o.settings.maxSlides > 1, o.carousel && (o.settings.preloadImages = "all"), o.minThreshold = o.settings.minSlides * o.settings.slideWidth + (o.settings.minSlides - 1) * o.settings.slideMargin, o.maxThreshold = o.settings.maxSlides * o.settings.slideWidth + (o.settings.maxSlides - 1) * o.settings.slideMargin, o.working=!1, o.controls = {}, o.interval = null, o.animProp = "vertical" == o.settings.mode ? "top" : "left", o.usingCSS = o.settings.useCSS && "fade" != o.settings.mode && function() {
                    var e = document.createElement("div"), t = ["WebkitPerspective", "MozPerspective", "OPerspective", "msPerspective"];
                    for (var i in t)
                        if (void 0 !== e.style[t[i]])
                            return o.cssPrefix = t[i].replace("Perspective", "").toLowerCase(), o.animProp = "-" + o.cssPrefix + "-transform", !0;
                    return !1
                }(), "vertical" == o.settings.mode && (o.settings.maxSlides = o.settings.minSlides), c()
            }, c = function() {
                if (r.wrap('<div class="bx-wrapper"><div class="bx-viewport"></div></div>'), o.viewport = r.parent(), o.loader = e('<div class="bx-loading" />'), o.viewport.prepend(o.loader), r.css({
                    width: "horizontal" == o.settings.mode ? 215 * o.children.length + "%": "auto",
                    position: "relative"
                }), o.usingCSS && o.settings.easing ? r.css("-" + o.cssPrefix + "-transition-timing-function", o.settings.easing) : o.settings.easing || (o.settings.easing = "swing"), v(), o.viewport.css({
                    width: "100%",
                    overflow: "hidden",
                    position: "relative"
                }), o.viewport.parent().css({
                    maxWidth: u()
                }), o.children.css({
                    "float": "horizontal" == o.settings.mode ? "left": "none",
                    listStyle: "none",
                    position: "relative"
                }), o.children.width(p()), "horizontal" == o.settings.mode && o.settings.slideMargin > 0 && o.children.css("marginRight", o.settings.slideMargin), "vertical" == o.settings.mode && o.settings.slideMargin > 0 && o.children.css("marginBottom", o.settings.slideMargin), "fade" == o.settings.mode && (o.children.css({
                    position: "absolute",
                    zIndex: 0,
                    display: "none"
                }), o.children.eq(o.settings.startSlide).css({
                    zIndex: 50,
                    display: "block"
                })), o.controls.el = e('<div class="bx-controls" />'), o.settings.captions && E(), o.settings.infiniteLoop && "fade" != o.settings.mode&&!o.settings.ticker) {
                    var t = "vertical" == o.settings.mode ? o.settings.minSlides: o.settings.maxSlides, i = o.children.slice(0, t).clone().addClass("bx-clone"), n = o.children.slice( - t).clone().addClass("bx-clone");
                    r.append(i).prepend(n)
                }
                o.active.last = o.settings.startSlide == f() - 1, o.settings.video && r.fitVids();
                var s = o.children.eq(o.settings.startSlide);
                "all" == o.settings.preloadImages && (s = r.children()), o.settings.ticker || (o.settings.pager && w(), o.settings.controls && T(), o.settings.auto && o.settings.autoControls && C(), (o.settings.controls || o.settings.autoControls || o.settings.pager) && o.viewport.after(o.controls.el)), s.imagesLoaded(g)
            }, g = function() {
                o.loader.remove(), m(), "vertical" == o.settings.mode && (o.settings.adaptiveHeight=!0), o.viewport.height(h()), r.redrawSlider(), o.settings.onSliderLoad(o.active.index), o.initialized=!0, e(window).bind("resize", X), o.settings.auto && o.settings.autoStart && L(), o.settings.ticker && W(), o.settings.pager && M(o.settings.startSlide), o.settings.controls && D(), o.settings.touchEnabled&&!o.settings.ticker && O()
            }, h = function() {
                var t = 0, n = e();
                if ("vertical" == o.settings.mode || o.settings.adaptiveHeight)
                    if (o.carousel) {
                        var s = 1 == o.settings.moveSlides ? o.active.index: o.active.index * x();
                        for (n = o.children.eq(s), i = 1; o.settings.maxSlides - 1 >= i; i++)
                            n = s + i >= o.children.length ? n.add(o.children.eq(i - 1)) : n.add(o.children.eq(s + i))
                    } else 
                        n = o.children.eq(o.active.index);
                else 
                    n = o.children;
                return "vertical" == o.settings.mode ? (n.each(function() {
                    t += e(this).outerHeight()
                }), o.settings.slideMargin > 0 && (t += o.settings.slideMargin * (o.settings.minSlides - 1))) : t = Math.max.apply(Math, n.map(function() {
                    return e(this).outerHeight(!1)
                }).get()), t
            }, u = function() {
                var e = "100%";
                return o.settings.slideWidth > 0 && (e = "horizontal" == o.settings.mode ? o.settings.maxSlides * o.settings.slideWidth + (o.settings.maxSlides - 1) * o.settings.slideMargin : o.settings.slideWidth), e
            }, p = function() {
                var e = o.settings.slideWidth, t = o.viewport.width();
                return 0 == o.settings.slideWidth || o.settings.slideWidth > t&&!o.carousel || "vertical" == o.settings.mode ? e = t : o.settings.maxSlides > 1 && "horizontal" == o.settings.mode && (t > o.maxThreshold || o.minThreshold > t && (e = (t - o.settings.slideMargin * (o.settings.minSlides - 1)) / o.settings.minSlides)), e
            }, v = function() {
                var e = 1;
                if ("horizontal" == o.settings.mode && o.settings.slideWidth > 0)
                    if (o.viewport.width() < o.minThreshold)
                        e = o.settings.minSlides;
                    else if (o.viewport.width() > o.maxThreshold)
                        e = o.settings.maxSlides;
                    else {
                        var t = o.children.first().width();
                        e = Math.floor(o.viewport.width() / t)
                    } else 
                        "vertical" == o.settings.mode && (e = o.settings.minSlides);
                return e
            }, f = function() {
                var e = 0;
                if (o.settings.moveSlides > 0)
                    if (o.settings.infiniteLoop)
                        e = o.children.length / x();
                    else 
                        for (var t = 0, i = 0; o.children.length > t;)
                            ++e, t = i + v(), i += o.settings.moveSlides <= v() ? o.settings.moveSlides : v();
                else 
                    e = Math.ceil(o.children.length / v());
                return e
            }, x = function() {
                return o.settings.moveSlides > 0 && o.settings.moveSlides <= v() ? o.settings.moveSlides : v()
            }, m = function() {
                if (o.children.length > o.settings.maxSlides && o.active.last&&!o.settings.infiniteLoop) {
                    if ("horizontal" == o.settings.mode) {
                        var e = o.children.last(), t = e.position();
                        S( - (t.left - (o.viewport.width() - e.width())), "reset", 0)
                    } else if ("vertical" == o.settings.mode) {
                        var i = o.children.length - o.settings.minSlides, t = o.children.eq(i).position();
                        S( - t.top, "reset", 0)
                    }
                } else {
                    var t = o.children.eq(o.active.index * x()).position();
                    o.active.index == f() - 1 && (o.active.last=!0), void 0 != t && ("horizontal" == o.settings.mode ? S( - t.left, "reset", 0) : "vertical" == o.settings.mode && S( - t.top, "reset", 0))
                }
            }, S = function(e, t, i, n) {
                if (o.usingCSS) {
                    var s = "vertical" == o.settings.mode ? "translate3d(0, " + e + "px, 0)": "translate3d(" + e + "px, 0, 0)";
                    r.css("-" + o.cssPrefix + "-transition-duration", i / 1e3 + "s"), "slide" == t ? (r.css(o.animProp, s), r.bind("transitionend webkitTransitionEnd oTransitionEnd MSTransitionEnd", function() {
                        r.unbind("transitionend webkitTransitionEnd oTransitionEnd MSTransitionEnd"), I()
                    })) : "reset" == t ? r.css(o.animProp, s) : "ticker" == t && (r.css("-" + o.cssPrefix + "-transition-timing-function", "linear"), r.css(o.animProp, s), r.bind("transitionend webkitTransitionEnd oTransitionEnd MSTransitionEnd", function() {
                        r.unbind("transitionend webkitTransitionEnd oTransitionEnd MSTransitionEnd"), S(n.resetValue, "reset", 0), H()
                    }))
                } else {
                    var a = {};
                    a[o.animProp] = e, "slide" == t ? r.animate(a, i, o.settings.easing, function() {
                        I()
                    }) : "reset" == t ? r.css(o.animProp, e) : "ticker" == t && r.animate(a, speed, "linear", function() {
                        S(n.resetValue, "reset", 0), H()
                    })
                }
            }, b = function() {
                var t = "";
                pagerQty = f();
                for (var i = 0; pagerQty > i; i++) {
                    var n = "";
                    o.settings.buildPager && e.isFunction(o.settings.buildPager) ? (n = o.settings.buildPager(i), o.pagerEl.addClass("bx-custom-pager")) : (n = i + 1, o.pagerEl.addClass("bx-default-pager")), t += '<div class="bx-pager-item"><a href="" data-slide-index="' + i + '" class="bx-pager-link">' + n + "</a></div>"
                }
                o.pagerEl.html(t)
            }, w = function() {
                o.settings.pagerCustom ? o.pagerEl = e(o.settings.pagerCustom) : (o.pagerEl = e('<div class="bx-pager" />'), o.settings.pagerSelector ? e(o.settings.pagerSelector).html(o.pagerEl) : o.controls.el.addClass("bx-has-pager").append(o.pagerEl), b()), o.pagerEl.delegate("a", "click", z)
            }, T = function() {
                o.controls.next = e('<a class="bx-next" href="">' + o.settings.nextText + "</a>"), o.controls.prev = e('<a class="bx-prev" href="">' + o.settings.prevText + "</a>"), o.controls.next.bind("click", A), o.controls.prev.bind("click", P), o.settings.nextSelector && e(o.settings.nextSelector).append(o.controls.next), o.settings.prevSelector && e(o.settings.prevSelector).append(o.controls.prev), o.settings.nextSelector || o.settings.prevSelector || (o.controls.directionEl = e('<div class="bx-controls-direction" />'), o.controls.directionEl.append(o.controls.prev).append(o.controls.next), o.controls.el.addClass("bx-has-controls-direction").append(o.controls.directionEl))
            }, C = function() {
                o.controls.start = e('<div class="bx-controls-auto-item"><a class="bx-start" href="">' + o.settings.startText + "</a></div>"), o.controls.stop = e('<div class="bx-controls-auto-item"><a class="bx-stop" href="">' + o.settings.stopText + "</a></div>"), o.controls.autoEl = e('<div class="bx-controls-auto" />'), o.controls.autoEl.delegate(".bx-start", "click", y), o.controls.autoEl.delegate(".bx-stop", "click", k), o.settings.autoControlsCombine ? o.controls.autoEl.append(o.controls.start) : o.controls.autoEl.append(o.controls.start).append(o.controls.stop), o.settings.autoControlsSelector ? e(o.settings.autoControlsSelector).html(o.controls.autoEl) : o.controls.el.addClass("bx-has-controls-auto").append(o.controls.autoEl), q(o.settings.autoStart ? "stop" : "start")
            }, E = function() {
                o.children.each(function() {
                    var t = e(this).find("img:first").attr("title");
                    void 0 != t && e(this).append('<div class="bx-caption"><span>' + t + "</span></div>")
                })
            }, A = function(e) {
                o.settings.auto && r.stopAuto(), r.goToNextSlide(), e.preventDefault()
            }, P = function(e) {
                o.settings.auto && r.stopAuto(), r.goToPrevSlide(), e.preventDefault()
            }, y = function(e) {
                r.startAuto(), e.preventDefault()
            }, k = function(e) {
                r.stopAuto(), e.preventDefault()
            }, z = function(t) {
                o.settings.auto && r.stopAuto();
                var i = e(t.currentTarget), n = parseInt(i.attr("data-slide-index"));
                n != o.active.index && r.goToSlide(n), t.preventDefault()
            }, M = function(t) {
                return "short" == o.settings.pagerType ? (o.pagerEl.html(t + 1 + o.settings.pagerShortSeparator + o.children.length), void 0) : (o.pagerEl.find("a").removeClass("active"), o.pagerEl.each(function(i, n) {
                    e(n).find("a").eq(t).addClass("active")
                }), void 0)
            }, I = function() {
                if (o.settings.infiniteLoop) {
                    var e = "";
                    0 == o.active.index ? e = o.children.eq(0).position() : o.active.index == f() - 1 && o.carousel ? e = o.children.eq((f() - 1) * x()).position() : o.active.index == o.children.length - 1 && (e = o.children.eq(o.children.length - 1).position()), "horizontal" == o.settings.mode ? S( - e.left, "reset", 0) : "vertical" == o.settings.mode && S( - e.top, "reset", 0)
                }
                o.working=!1, o.settings.onSlideAfter(o.children.eq(o.active.index), o.oldIndex, o.active.index)
            }, q = function(e) {
                o.settings.autoControlsCombine ? o.controls.autoEl.html(o.controls[e]) : (o.controls.autoEl.find("a").removeClass("active"), o.controls.autoEl.find("a:not(.bx-" + e + ")").addClass("active"))
            }, D = function() {
                !o.settings.infiniteLoop && o.settings.hideControlOnEnd ? 0 == o.active.index ? (o.controls.prev.addClass("disabled"), o.controls.next.removeClass("disabled")) : o.active.index == f() - 1 ? (o.controls.next.addClass("disabled"), o.controls.prev.removeClass("disabled")) : (o.controls.prev.removeClass("disabled"), o.controls.next.removeClass("disabled")) : 1 == f() && (o.controls.prev.addClass("disabled"), o.controls.next.addClass("disabled"))
            }, L = function() {
                o.settings.autoDelay > 0 ? setTimeout(r.startAuto, o.settings.autoDelay) : r.startAuto(), o.settings.autoHover && r.hover(function() {
                    o.interval && (r.stopAuto(!0), o.autoPaused=!0)
                }, function() {
                    o.autoPaused && (r.startAuto(!0), o.autoPaused = null)
                })
            }, W = function() {
                var t = 0;
                if ("next" == o.settings.autoDirection)
                    r.append(o.children.clone().addClass("bx-clone"));
                else {
                    r.prepend(o.children.clone().addClass("bx-clone"));
                    var i = o.children.first().position();
                    t = "horizontal" == o.settings.mode?-i.left : - i.top
                }
                S(t, "reset", 0), o.settings.pager=!1, o.settings.controls=!1, o.settings.autoControls=!1, o.settings.tickerHover&&!o.usingCSS && o.viewport.hover(function() {
                    r.stop()
                }, function() {
                    var t = 0;
                    o.children.each(function() {
                        t += "horizontal" == o.settings.mode ? e(this).outerWidth(!0) : e(this).outerHeight(!0)
                    });
                    var i = o.settings.speed / t, n = "horizontal" == o.settings.mode ? "left": "top", s = i * (t - Math.abs(parseInt(r.css(n))));
                    H(s)
                }), H()
            }, H = function(e) {
                speed = e ? e : o.settings.speed;
                var t = {
                    left: 0,
                    top: 0
                }, i = {
                    left: 0,
                    top: 0
                };
                "next" == o.settings.autoDirection ? t = r.find(".bx-clone").first().position() : i = o.children.first().position();
                var n = "horizontal" == o.settings.mode?-t.left : - t.top, s = "horizontal" == o.settings.mode?-i.left : - i.top, a = {
                    resetValue: s
                };
                S(n, "ticker", speed, a)
            }, O = function() {
                o.touch = {
                    start: {
                        x: 0,
                        y: 0
                    },
                    end: {
                        x: 0,
                        y: 0
                    }
                }, o.viewport.bind("touchstart", N)
            }, N = function(e) {
                if (o.working)
                    e.preventDefault();
                else {
                    o.touch.originalPos = r.position();
                    var t = e.originalEvent;
                    o.touch.start.x = t.changedTouches[0].pageX, o.touch.start.y = t.changedTouches[0].pageY, o.viewport.bind("touchmove", B), o.viewport.bind("touchend", Q)
                }
            }, B = function(e) {
                var t = e.originalEvent, i = Math.abs(t.changedTouches[0].pageX - o.touch.start.x), n = Math.abs(t.changedTouches[0].pageY - o.touch.start.y);
                if (3 * i > n && o.settings.preventDefaultSwipeX ? e.preventDefault() : 3 * n > i && o.settings.preventDefaultSwipeY && e.preventDefault(), "fade" != o.settings.mode && o.settings.oneToOneTouch) {
                    var s = 0;
                    if ("horizontal" == o.settings.mode) {
                        var r = t.changedTouches[0].pageX - o.touch.start.x;
                        s = o.touch.originalPos.left + r
                    } else {
                        var r = t.changedTouches[0].pageY - o.touch.start.y;
                        s = o.touch.originalPos.top + r
                    }
                    S(s, "reset", 0)
                }
            }, Q = function(e) {
                o.viewport.unbind("touchmove", B);
                var t = e.originalEvent, i = 0;
                if (o.touch.end.x = t.changedTouches[0].pageX, o.touch.end.y = t.changedTouches[0].pageY, "fade" == o.settings.mode) {
                    var n = Math.abs(o.touch.start.x - o.touch.end.x);
                    n >= o.settings.swipeThreshold && (o.touch.start.x > o.touch.end.x ? r.goToNextSlide() : r.goToPrevSlide(), r.stopAuto())
                } else {
                    var n = 0;
                    "horizontal" == o.settings.mode ? (n = o.touch.end.x - o.touch.start.x, i = o.touch.originalPos.left) : (n = o.touch.end.y - o.touch.start.y, i = o.touch.originalPos.top), !o.settings.infiniteLoop && (0 == o.active.index && n > 0 || o.active.last && 0 > n) ? S(i, "reset", 200) : Math.abs(n) >= o.settings.swipeThreshold ? (0 > n ? r.goToNextSlide() : r.goToPrevSlide(), r.stopAuto()) : S(i, "reset", 200)
                }
                o.viewport.unbind("touchend", Q)
            }, X = function() {
                var t = e(window).width(), i = e(window).height();
                (a != t || l != i) && (a = t, l = i, r.redrawSlider())
            };
            return r.goToSlide = function(t, i) {
                if (!o.working && o.active.index != t)
                    if (o.working=!0, o.oldIndex = o.active.index, o.active.index = 0 > t ? f() - 1 : t >= f() ? 0 : t, o.settings.onSlideBefore(o.children.eq(o.active.index), o.oldIndex, o.active.index), "next" == i ? o.settings.onSlideNext(o.children.eq(o.active.index), o.oldIndex, o.active.index) : "prev" == i && o.settings.onSlidePrev(o.children.eq(o.active.index), o.oldIndex, o.active.index), o.active.last = o.active.index >= f() - 1, o.settings.pager && M(o.active.index), o.settings.controls && D(), "fade" == o.settings.mode)
                        o.settings.adaptiveHeight && o.viewport.height() != h() && o.viewport.animate({
                            height: h()
                        }, o.settings.adaptiveHeightSpeed), o.children.filter(":visible").fadeOut(o.settings.speed).css({
                            zIndex: 0
                        }), o.children.eq(o.active.index).css("zIndex", 51).fadeIn(o.settings.speed, function() {
                            e(this).css("zIndex", 50), I()
                        });
                    else {
                        o.settings.adaptiveHeight && o.viewport.height() != h() && o.viewport.animate({
                            height: h()
                        }, o.settings.adaptiveHeightSpeed);
                        var n = 0, s = {
                            left: 0,
                            top: 0
                        };
                        if (!o.settings.infiniteLoop && o.carousel && o.active.last)
                            if ("horizontal" == o.settings.mode) {
                                var a = o.children.eq(o.children.length - 1);
                                s = a.position(), n = o.viewport.width() - a.width()
                            } else {
                                var l = o.children.length - o.settings.minSlides;
                                s = o.children.eq(l).position()
                            } else if (o.carousel && o.active.last && "prev" == i) {
                                var d = 1 == o.settings.moveSlides ? o.settings.maxSlides - x(): (f() - 1) * x() - (o.children.length - o.settings.maxSlides), a = r.children(".bx-clone").eq(d);
                                s = a.position()
                            } else if ("next" == i && 0 == o.active.index)
                                s = r.find(".bx-clone").eq(o.settings.maxSlides).position(), o.active.last=!1;
                            else if (t >= 0) {
                                var c = t * x();
                                s = o.children.eq(c).position()
                            }
                            var g = "horizontal" == o.settings.mode?-(s.left - n) : - s.top;
                            S(g, "slide", o.settings.speed)
                        }
                    }, r.goToNextSlide = function() {
                if (o.settings.infiniteLoop ||!o.active.last) {
                    var e = parseInt(o.active.index) + 1;
                    r.goToSlide(e, "next")
                }
            }, r.goToPrevSlide = function() {
                if (o.settings.infiniteLoop || 0 != o.active.index) {
                    var e = parseInt(o.active.index) - 1;
                    r.goToSlide(e, "prev")
                }
            }, r.startAuto = function(e) {
                o.interval || (o.interval = setInterval(function() {
                    "next" == o.settings.autoDirection ? r.goToNextSlide() : r.goToPrevSlide()
                }, o.settings.pause), o.settings.autoControls && 1 != e && q("stop"))
            }, r.stopAuto = function(e) {
                o.interval && (clearInterval(o.interval), o.interval = null, o.settings.autoControls && 1 != e && q("start"))
            }, r.getCurrentSlide = function() {
                return o.active.index
            }, r.getSlideCount = function() {
                return o.children.length
            }, r.redrawSlider = function() {
                o.children.add(r.find(".bx-clone")).width(p()), o.viewport.css("height", h()), o.settings.ticker || m(), o.active.last && (o.active.index = f() - 1), o.active.index >= f() && (o.active.last=!0), o.settings.pager&&!o.settings.pagerCustom && (b(), M(o.active.index))
            }, r.destroySlider = function() {
                o.initialized && (o.initialized=!1, e(".bx-clone", this).remove(), o.children.removeAttr("style"), this.removeAttr("style").unwrap().unwrap(), o.controls.el && o.controls.el.remove(), o.controls.next && o.controls.next.remove(), o.controls.prev && o.controls.prev.remove(), o.pagerEl && o.pagerEl.remove(), e(".bx-caption", this).remove(), o.controls.autoEl && o.controls.autoEl.remove(), clearInterval(o.interval), e(window).unbind("resize", X))
            }, r.reloadSlider = function(e) {
                void 0 != e && (s = e), r.destroySlider(), d()
            }, d(), this
        }
    }
})(jQuery), function(e, t) {
    var i = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
    e.fn.imagesLoaded = function(n) {
        function s() {
            var t = e(g), i = e(h);
            a && (h.length ? a.reject(d, t, i) : a.resolve(d)), e.isFunction(n) && n.call(r, d, t, i)
        }
        function o(t, n) {
            t.src === i||-1 !== e.inArray(t, c) || (c.push(t), n ? h.push(t) : g.push(t), e.data(t, "imagesLoaded", {
                isBroken: n,
                src: t.src
            }), l && a.notifyWith(e(t), [n, d, e(g), e(h)]), d.length === c.length && (setTimeout(s), d.unbind(".imagesLoaded")))
        }
        var r = this, a = e.isFunction(e.Deferred) ? e.Deferred(): 0, l = e.isFunction(a.notify), d = r.find("img").add(r.filter("img")), c = [], g = [], h = [];
        return e.isPlainObject(n) && e.each(n, function(e, t) {
            "callback" === e ? n = t : a && a[e](t)
        }), d.length ? d.bind("load.imagesLoaded error.imagesLoaded", function(e) {
            o(e.target, "error" === e.type)
        }).each(function(n, s) {
            var r = s.src, a = e.data(s, "imagesLoaded");
            a && a.src === r ? o(s, a.isBroken) : s.complete && s.naturalWidth !== t ? o(s, 0 === s.naturalWidth || 0 === s.naturalHeight) : (s.readyState || s.complete) && (s.src = i, s.src = r)
        }) : s(), a ? a.promise(r) : r
    }
}(jQuery);


jQuery(document).ready(function($) {

    //----------------------------------------------------------------------------
    // Top Nav - functionality to make Zurb Top Bar work with Wordpress
    //----------------------------------------------------------------------------


    $('#menu-top-menu').addClass('right');
    // add classes to activate dropdowns
    $('#menu-top-menu .sub-menu').parent().addClass('has-dropdown');
    $('#menu-top-menu .sub-menu').addClass('dropdown');
    // inject back button for menu in mobile
    $('#menu-top-menu .has-dropdown .dropdown').prepend('<li class="notch"><span></span></li><li class="title back js-generated"><h5><a href="#">Back</a></h5></li>');

    // clone menu and append to subnav to generate a sub navigation
    $('#menu-top-menu > li').each(function() {
        the_html = $(this).children('ul').html();
        if (the_html != null) {
            $('.subnav').append('<ul>' + the_html + '</ul>');
        }
    });
    // hide all subnavs unless it is the current page's parent sub nav
    $('.subnav ul .current-menu-item').parent().css("display", "block");

    // $('#menu-top-menu .sub-menu.dropdown li.current-menu-item').addClass('active');

    //----------------------------------------------------------------------------
    // Miscellaneous
    //----------------------------------------------------------------------------

    // add intro call to the first paragraph
    $('.page-content.need-intro p:nth-of-type(1)').addClass('intro');

    // Silver Button Span Append
    $('.silbtn').each(function() {
        link_text = $(this).html();
        $(this).empty();
        $(this).append('<span>' + link_text + '</span>');
    });

    // Project Stories List Toggle
    $('#ps .ps-title .see').click(function() {
        $('#ps .ps-drop').slideToggle();
        return false;
    });

    // Taking Action Toggle
    $('.ta-button').click(function() {
        $('.ta-info').slideToggle();
        return false;
    });

    // Enactus Worldwide Button Toggle
    $('.enworld-toggle').click(function() {
        $('#world-drop').fadeToggle();
        return false;
    });


    //----------------------------------------------------------------------------
    // Home Page Toggles
    //----------------------------------------------------------------------------

    // Educators
    $('.toggle-panel.educators > h3 ').click(function() {
        $('.toggle-panel.educators p').slideToggle();
        return false;
    });

    // Educators
    $('.toggle-panel.executives > h3 ').click(function() {
        $('.toggle-panel.executives p').slideToggle();
        return false;
    });

    // Educators
    $('.toggle-panel.students > h3 ').click(function() {
        $('.toggle-panel.students p').slideToggle();
        return false;
    });

    // Educators
    $('.toggle-panel.project > h3 ').click(function() {
        $('.toggle-panel.project p').slideToggle();
        return false;
    });

    //----------------------------------------------------------------------------
    // Blog
    //----------------------------------------------------------------------------

    // Blog Sidebar RSS
    $('#blog-sidebar .container-rss input').val('Subscribe');
    $('#blog-sidebar .container-rss input').focus(function() {
        if ($(this).val() == 'Subscribe') {
            $(this).val('').removeClass('default');
        }
    });
    $('#blog-sidebar .container-rss input').blur(function() {
        if ($(this).val() == '') {
            $(this).val('Subscribe').addClass('default');
        }
    });

    // Blog Search
    $('#blog-sidebar .container-search input').val('Search');
    $('#blog-sidebar .container-search input').focus(function() {
        if ($(this).val() == 'Search') {
            $(this).val('').removeClass('default');
        }
    });
    $('#blog-sidebar .container-search input').blur(function() {
        if ($(this).val() == '') {
            $(this).val('Search').addClass('default');
        }
    });

    //----------------------------------------------------------------------------
    // Where We Work
    //----------------------------------------------------------------------------

    // Where Maps Show/Hide
    $('#where-map .map-wrap').hide();
    $('#where-map .map-wrap.op-map').show().addClass('current');
    $('#where-map .buttons-wrap a').click(function() {
        $('#where-map .buttons-wrap a').removeClass('current');
        which = $(this).addClass('current').attr('id');
        $('#where-map .map-wrap').fadeOut().removeClass('current');
        $('#where-map .' + which).fadeIn().addClass('current');
        return false;
    });

    // Where Map Country Hovers
    $('#where-map  .pos').hoverIntent(function() {
        $(this).children('div').fadeIn('fast');
    }, function() {
        $(this).children('div').fadeOut('fast');
    });

    //----------------------------------------------------------------------------
    // Universities
    //----------------------------------------------------------------------------

    // Country Toggles
    $(".uni-container h3, .uni-container .uni-list .close").click(function() {
        $(this).parents(".uni-container").toggleClass('expanded').find('.uni-list').slideToggle(250);
    })
    //----------------------------------------------------------------------------
    // Country Detail
    //----------------------------------------------------------------------------

    // Active Teams Toggle on Mobile only
    if (screen.width < 500) {
        $('.country-content ul.active').addClass('expand-mobile');
        $('.country-content .ads').addClass('expand-mobile');
    }

    $('.country-content .active-teams').click(function() {
        $(this).toggleClass('minimize');
        $('.country-content .active.expand-mobile').slideToggle();
        return false;
    });

    $('.country-content .active-board').click(function() {
        $(this).toggleClass('minimize');
        $('.country-content .ads.expand-mobile').slideToggle();
        return false;
    });

    //----------------------------------------------------------------------------
    // World Cup
    //----------------------------------------------------------------------------

    //Carousel Hover
    $('#wc-footer .slide').hoverIntent(function() {
        $(this).children('.img').children('.blue-overlay').fadeOut();
    }, function() {
        $(this).children('.img').children('.blue-overlay').fadeIn();
    });

    //World Cup Detail Sub Nav
    $('#wc-subnav a').click(function() {
        $('#wc-subnav li a.cur').removeClass('cur');
        $('#wc-subpages > li').hide();
        $('#t_' + $(this).addClass('cur').attr('href').substr(1)).show();
        return false;
    });
    if ($('#wc-subnav > li a.cur').length == 0)
        $('#wc-subnav > li:first-child a').click();

    //----------------------------------------------------------------------------
    // Footer
    //----------------------------------------------------------------------------

    // Footer Section Toggles
    if (screen.width < 500) {
        $('.footer-sections li').addClass('expand');
    }

    $('.footer-sections .ft-involved.expand h3 ').click(function() {
        $(this).toggleClass('minimize');
        $('.footer-sections .ft-involved.expand .ft-container').slideToggle();
        return false;
    });

    $('.footer-sections .ft-connected.expand h3 ').click(function() {
        $(this).toggleClass('minimize');
        $('.footer-sections .ft-connected.expand .ft-container').slideToggle();
        return false;
    });

    $('.footer-sections .ft-contact.expand h3 ').click(function() {
        $(this).toggleClass('minimize');
        $('.footer-sections .ft-contact.expand .ft-container').slideToggle();
        return false;
    });

    //----------------------------------------------------------------------------
    // Smooth Scroll
    //----------------------------------------------------------------------------

    function goSmoothScroll (id) {
        $('html, body').animate({
            scrollTop: ($(id).offset().top - 0)
        }, 'slow');
    };

    $('.smoothscroll').click(function() {
        goSmoothScroll($(this).attr('href'));
        return false;
    });


    //----------------------------------------------------------------------------
    // National Competition Social Media Stuff
    //----------------------------------------------------------------------------

    // Twitter
    window.cuttertTweetWrap = 1;
    setTimeout("changeTweet()", 1000);
    changeTweet = function() {
        var queue = [],
        $wrappers = $(".twitter-wrap"),
        timer = null,
        reset = function (el) {
            if (el) {
                queue.push(el);
            }
            timer = window.setTimeout( fire, 1000 );
        },
        rotate = function () {
            var $wrap = $(this);


            if ($wrap.find('li').length) {
                $wrap.find('li').last().fadeOut('slow', function() {
                    $(this).prependTo($wrap).show();
                    reset($wrap);
                });
            } else {
                reset();
            }
        },
        fire = function () {
            window.clearTimeout(timer);
            timer = null;
            if ($(queue[0]).hasClass('paused')) {
                reset(queue.shift());
            } else {
                rotate.apply(queue.shift());
            }
        };

        timer = window.setTimeout( fire, 1000 );

        $wrappers.each(function (i, el) {
            queue.push($(el));
        });

        $wrappers.hover(function () {
            $(this).addClass('paused');
        }, function () {
            $(this).removeClass('paused');
        });
    }

    // Instagram
    window.currentInstagramGroup = 1;
    window.nextInstagramGroup = 2;
    setTimeout("rotateInstagram()", 6000);
    rotateInstagram = function() {
        if ( $(".instagram-group-" + window.nextInstagramGroup).hasClass('paused') || $(".instagram-group-" + window.currentInstagramGroup).hasClass('paused') ) {
            setTimeout("rotateInstagram()", 7000);
        } else {
            $(".instagram-group-" + window.currentInstagramGroup + " li.img:gt(" + ($(".instagram-group-" + window.currentInstagramGroup + " li").length - 9) + ")").prependTo(".instagram-group-" + window.currentInstagramGroup);
            $(".instagram-group-" + window.nextInstagramGroup).fadeOut('slow', function() {
                $(".instagram-group-" + window.nextInstagramGroup).prependTo('.instagram-container').show();
                if (window.currentInstagramGroup == 1) {
                    window.currentInstagramGroup = 2;
                    window.nextInstagramGroup = 1;
                } else {
                    window.currentInstagramGroup = 1;
                    window.nextInstagramGroup = 2;
                }
                setTimeout("rotateInstagram()", 7000);
            })
        }
    };

    $(".instagram-group-" + window.nextInstagramGroup + ", .instagram-group-" + window.currentInstagramGroup).hover(function() {
        $(this).addClass('paused');
    }, function() {
        $(this).removeClass('paused');
    });

    // Instagram Hovers
    $('.instagram-small li a, .instagram-large li a').hoverIntent(function() {
        $(this).children('span').fadeIn('fast');
    }, function() {
        $(this).children('span').fadeOut('fast');
    });


    //----------------------------------------------------------------------------
    // World Cup 2014
    //----------------------------------------------------------------------------

    wistiaCupInvite = Wistia.embed("wb86jkgs6z", {
        videoFoam: true
    });

    wistiaCupInvite.bind("end", function() {
        $('body').removeClass('is-wc14-invite-video-active');
    });

    $('.wc14-v-btn.toggle').click(function() {
        wistiaCupInvite.play();
        $('body').addClass('is-wc14-invite-video-active');
    });

    $('.wc14-v-btn.close').click(function() {
        wistiaCupInvite.pause().time(0);
        $('body').removeClass('is-wc14-invite-video-active');
    });

});
