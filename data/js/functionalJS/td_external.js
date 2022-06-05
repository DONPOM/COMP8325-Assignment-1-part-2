//@prepros-append td_style_customizer.js


/* Modernizr 2.6.2 (Custom Build) | MIT & BSD
 * Build: http://modernizr.com/download/#-csstransforms-csstransforms3d-csstransitions-svg-touch-shiv-mq-cssclasses-prefixed-teststyles-testprop-testallprops-prefixes-domprefixes-load
 */
;
window.Modernizr = function(a, b, c) {
    function B(a) {
        j.cssText = a
    }
    function C(a, b) {
        return B(m.join(a + ";") + (b || ""))
    }
    function D(a, b) {
        return typeof a === b
    }
    function E(a, b) {
        return !!~("" + a).indexOf(b)
    }
    function F(a, b) {
        for (var d in a) {
            var e = a[d];
            if (!E(e, "-") && j[e] !== c)
                return b == "pfx" ? e : !0
        }
        return !1
    }
    function G(a, b, d) {
        for (var e in a) {
            var f = b[a[e]];
            if (f !== c)
                return d===!1 ? a[e] : D(f, "function") ? f.bind(d || b) : f
        }
        return !1
    }
    function H(a, b, c) {
        var d = a.charAt(0).toUpperCase() + a.slice(1), e = (a + " " + o.join(d + " ") + d).split(" ");
        return D(b, "string") || D(b, "undefined") ? F(e, b) : (e = (a + " " + p.join(d + " ") + d).split(" "), G(e, b, c))
    }
    var d = "2.6.2", e = {}, f=!0, g = b.documentElement, h = "modernizr", i = b.createElement(h), j = i.style, k, l = {}.toString, m = " -webkit- -moz- -o- -ms- ".split(" "), n = "Webkit Moz O ms", o = n.split(" "), p = n.toLowerCase().split(" "), q = {
        svg: "http://www.w3.org/2000/svg"
    }, r = {}, s = {}, t = {}, u = [], v = u.slice, w, x = function(a, c, d, e) {
        var f, i, j, k, l = b.createElement("div"), m = b.body, n = m || b.createElement("body");
        if (parseInt(d, 10))
            while (d--)
                j = b.createElement("div"), j.id = e ? e[d] : h + (d + 1), l.appendChild(j);
        return f = ["&#173;", '<style id="s', h, '">', a, "</style>"].join(""), l.id = h, (m ? l : n).innerHTML += f, n.appendChild(l), m || (n.style.background = "", n.style.overflow = "hidden", k = g.style.overflow, g.style.overflow = "hidden", g.appendChild(n)), i = c(l, a), m ? l.parentNode.removeChild(l) : (n.parentNode.removeChild(n), g.style.overflow = k), !!i
    }, y = function(b) {
        var c = a.matchMedia || a.msMatchMedia;
        if (c)
            return c(b).matches;
        var d;
        return x("@media " + b + " { #" + h + " { position: absolute; } }", function(b) {
            d = (a.getComputedStyle ? getComputedStyle(b, null) : b.currentStyle)["position"] == "absolute"
        }), d
    }, z = {}.hasOwnProperty, A;
    !D(z, "undefined")&&!D(z.call, "undefined") ? A = function(a, b) {
        return z.call(a, b)
    } : A = function(a, b) {
        return b in a && D(a.constructor.prototype[b], "undefined")
    }, Function.prototype.bind || (Function.prototype.bind = function(b) {
        var c = this;
        if (typeof c != "function")
            throw new TypeError;
        var d = v.call(arguments, 1), e = function() {
            if (this instanceof e) {
                var a = function() {};
                a.prototype = c.prototype;
                var f = new a, g = c.apply(f, d.concat(v.call(arguments)));
                return Object(g) === g ? g : f
            }
            return c.apply(b, d.concat(v.call(arguments)))
        };
        return e
    }), r.touch = function() {
        var c;
        return "ontouchstart"in a || a.DocumentTouch && b instanceof DocumentTouch ? c=!0 : x(["@media (", m.join("touch-enabled),("), h, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(a) {
            c = a.offsetTop === 9
        }), c
    }, r.csstransforms = function() {
        return !!H("transform")
    }, r.csstransforms3d = function() {
        var a=!!H("perspective");
        return a && "webkitPerspective"in g.style && x("@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}", function(b, c) {
            a = b.offsetLeft === 9 && b.offsetHeight === 3
        }), a
    }, r.csstransitions = function() {
        return H("transition")
    }, r.svg = function() {
        return !!b.createElementNS&&!!b.createElementNS(q.svg, "svg").createSVGRect
    };
    for (var I in r)
        A(r, I) && (w = I.toLowerCase(), e[w] = r[I](), u.push((e[w] ? "" : "no-") + w));
    return e.addTest = function(a, b) {
        if (typeof a == "object")
            for (var d in a)
                A(a, d) && e.addTest(d, a[d]);
        else {
            a = a.toLowerCase();
            if (e[a] !== c)
                return e;
            b = typeof b == "function" ? b() : b, typeof f != "undefined" && f && (g.className += " " + (b ? "" : "no-") + a), e[a] = b
        }
        return e
    }, B(""), i = k = null, function(a, b) {
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
    }(this, b), e._version = d, e._prefixes = m, e._domPrefixes = p, e._cssomPrefixes = o, e.mq = y, e.testProp = function(a) {
        return F([a])
    }, e.testAllProps = H, e.testStyles = x, e.prefixed = function(a, b, c) {
        return b ? H(a, b, c) : H(a, "pfx")
    }, g.className = g.className.replace(/(^|\s)no-js(\s|$)/, "$1$2") + (f ? " js " + u.join(" ") : ""), e
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
};




// jQuery Easing v1.3 - http://gsgd.co.uk/sandbox/jquery/easing/
jQuery.easing.jswing = jQuery.easing.swing;
jQuery.extend(jQuery.easing, {
    def: "easeOutQuad",
    swing: function(e, a, c, b, d) {
        return jQuery.easing[jQuery.easing.def](e, a, c, b, d)
    },
    easeInQuad: function(e, a, c, b, d) {
        return b * (a/=d) * a + c
    },
    easeOutQuad: function(e, a, c, b, d) {
        return - b * (a/=d) * (a - 2) + c
    },
    easeInOutQuad: function(e, a, c, b, d) {
        return 1 > (a/=d / 2) ? b / 2 * a * a + c : - b / 2 * (--a * (a - 2) - 1) + c
    },
    easeInCubic: function(e, a, c, b, d) {
        return b * (a/=d) * a * a + c
    },
    easeOutCubic: function(e, a, c, b, d) {
        return b * ((a = a / d - 1) * a * a + 1) + c
    },
    easeInOutCubic: function(e, a, c, b, d) {
        return 1 > (a/=d / 2) ? b / 2 * a * a * a + c :
        b / 2 * ((a -= 2) * a * a + 2) + c
    },
    easeInQuart: function(e, a, c, b, d) {
        return b * (a/=d) * a * a * a + c
    },
    easeOutQuart: function(e, a, c, b, d) {
        return - b * ((a = a / d - 1) * a * a * a - 1) + c
    },
    easeInOutQuart: function(e, a, c, b, d) {
        return 1 > (a/=d / 2) ? b / 2 * a * a * a * a + c : - b / 2 * ((a -= 2) * a * a * a - 2) + c
    },
    easeInQuint: function(e, a, c, b, d) {
        return b * (a/=d) * a * a * a * a + c
    },
    easeOutQuint: function(e, a, c, b, d) {
        return b * ((a = a / d - 1) * a * a * a * a + 1) + c
    },
    easeInOutQuint: function(e, a, c, b, d) {
        return 1 > (a/=d / 2) ? b / 2 * a * a * a * a * a + c : b / 2 * ((a -= 2) * a * a * a * a + 2) + c
    },
    easeInSine: function(e, a, c, b, d) {
        return - b * Math.cos(a /
        d * (Math.PI / 2)) + b + c
    },
    easeOutSine: function(e, a, c, b, d) {
        return b * Math.sin(a / d * (Math.PI / 2)) + c
    },
    easeInOutSine: function(e, a, c, b, d) {
        return - b / 2 * (Math.cos(Math.PI * a / d) - 1) + c
    },
    easeInExpo: function(e, a, c, b, d) {
        return 0 == a ? c : b * Math.pow(2, 10 * (a / d - 1)) + c
    },
    easeOutExpo: function(e, a, c, b, d) {
        return a == d ? c + b : b * ( - Math.pow(2, - 10 * a / d) + 1) + c
    },
    easeInOutExpo: function(e, a, c, b, d) {
        return 0 == a ? c : a == d ? c + b : 1 > (a/=d / 2) ? b / 2 * Math.pow(2, 10 * (a - 1)) + c : b / 2 * ( - Math.pow(2, - 10*--a) + 2) + c
    },
    easeInCirc: function(e, a, c, b, d) {
        return - b * (Math.sqrt(1 - (a/=d) *
        a) - 1) + c
    },
    easeOutCirc: function(e, a, c, b, d) {
        return b * Math.sqrt(1 - (a = a / d - 1) * a) + c
    },
    easeInOutCirc: function(e, a, c, b, d) {
        return 1 > (a/=d / 2)?-b / 2 * (Math.sqrt(1 - a * a) - 1) + c : b / 2 * (Math.sqrt(1 - (a -= 2) * a) + 1) + c
    },
    easeInElastic: function(e, a, c, b, d) {
        e = 1.70158;
        var f = 0, g = b;
        if (0 == a)
            return c;
        if (1 == (a/=d))
            return c + b;
        f || (f = 0.3 * d);
        g < Math.abs(b) ? (g = b, e = f / 4) : e = f / (2 * Math.PI) * Math.asin(b / g);
        return - (g * Math.pow(2, 10 * (a -= 1)) * Math.sin((a * d - e) * 2 * Math.PI / f)) + c
    },
    easeOutElastic: function(e, a, c, b, d) {
        e = 1.70158;
        var f = 0, g = b;
        if (0 == a)
            return c;
        if (1 ==
        (a/=d))
            return c + b;
        f || (f = 0.3 * d);
        g < Math.abs(b) ? (g = b, e = f / 4) : e = f / (2 * Math.PI) * Math.asin(b / g);
        return g * Math.pow(2, - 10 * a) * Math.sin((a * d - e) * 2 * Math.PI / f) + b + c
    },
    easeInOutElastic: function(e, a, c, b, d) {
        e = 1.70158;
        var f = 0, g = b;
        if (0 == a)
            return c;
        if (2 == (a/=d / 2))
            return c + b;
        f || (f = d * 0.3 * 1.5);
        g < Math.abs(b) ? (g = b, e = f / 4) : e = f / (2 * Math.PI) * Math.asin(b / g);
        return 1 > a?-0.5 * g * Math.pow(2, 10 * (a -= 1)) * Math.sin((a * d - e) * 2 * Math.PI / f) + c : 0.5 * g * Math.pow(2, - 10 * (a -= 1)) * Math.sin((a * d - e) * 2 * Math.PI / f) + b + c
    },
    easeInBack: function(e, a, c, b, d, f) {
        void 0 ==
        f && (f = 1.70158);
        return b * (a/=d) * a * ((f + 1) * a - f) + c
    },
    easeOutBack: function(e, a, c, b, d, f) {
        void 0 == f && (f = 1.70158);
        return b * ((a = a / d - 1) * a * ((f + 1) * a + f) + 1) + c
    },
    easeInOutBack: function(e, a, c, b, d, f) {
        void 0 == f && (f = 1.70158);
        return 1 > (a/=d / 2) ? b / 2 * a * a * (((f*=1.525) + 1) * a - f) + c : b / 2 * ((a -= 2) * a * (((f*=1.525) + 1) * a + f) + 2) + c
    },
    easeInBounce: function(e, a, c, b, d) {
        return b - jQuery.easing.easeOutBounce(e, d - a, 0, b, d) + c
    },
    easeOutBounce: function(e, a, c, b, d) {
        return (a/=d) < 1 / 2.75 ? b * 7.5625 * a * a + c : a < 2 / 2.75 ? b * (7.5625 * (a -= 1.5 / 2.75) * a + 0.75) + c : a < 2.5 / 2.75 ?
        b * (7.5625 * (a -= 2.25 / 2.75) * a + 0.9375) + c : b * (7.5625 * (a -= 2.625 / 2.75) * a + 0.984375) + c
    },
    easeInOutBounce: function(e, a, c, b, d) {
        return a < d / 2 ? 0.5 * jQuery.easing.easeInBounce(e, 2 * a, 0, b, d) + c : 0.5 * jQuery.easing.easeOutBounce(e, 2 * a - d, 0, b, d) + 0.5 * b + c
    }
});



/*
 *  v0.2b - jQuery plugin
 * Copyright (c) 2008 Joel Birch
 * Dual licensed under the MIT and GPL licenses:
 * 	http://www.opensource.org/licenses/mit-license.php
 * 	http://www.gnu.org/licenses/gpl.html
 */
(function(a) {
    a.fn.supersubs = function(h) {
        var f = a.extend({}, a.fn.supersubs.defaults, h);
        return this.each(function() {
            var c = a(this), d = a.meta ? a.extend({}, f, c.data()): f, h = a('<li id="menu-fontsize">&#8212;</li>').css({
                padding: 0,
                position: "absolute",
                top: "-999em",
                width: "auto"
            }).appendTo(c).width();
            a("#menu-fontsize").remove();
            $ULs = c.find("ul");
            $ULs.each(function(e) {
                e = $ULs.eq(e);
                var g = e.children(), c = g.children("a"), f = g.css("white-space", "nowrap").css("float"), b = e.add(g).add(c).css({
                    "float": "none",
                    width: "auto"
                }).end().end()[0].clientWidth /
                h, b = b + d.extraWidth;
                b > d.maxWidth ? b = d.maxWidth : b < d.minWidth && (b = d.minWidth);
                b += "em";
                e.css("width", b);
                g.css({
                    "float": f,
                    width: "100%",
                    "white-space": "normal"
                }).each(function() {
                    var c = a(">ul", this), d = void 0 !== c.css("left") ? "left" : "right";
                    c.css(d, b)
                })
            })
        })
    };
    a.fn.supersubs.defaults = {
        minWidth: 9,
        maxWidth: 25,
        extraWidth: 0
    }
})(jQuery);



/**
 * hoverIntent is similar to jQuery's built-in "hover" method except that
 * instead of firing the handlerIn function immediately, hoverIntent checks
 * to see if the user's mouse has slowed down (beneath the sensitivity
 * threshold) before firing the event. The handlerOut function is only
 * called after a matching handlerIn.
 *
 * hoverIntent r7 // 2013.03.11 // jQuery 1.9.1+
 * http://cherne.net/brian/resources/jquery.hoverIntent.html
 */
(function(e) {
    e.fn.hoverIntent = function(a, f, n) {
        var b = {
            interval: 100,
            sensitivity: 7,
            timeout: 0
        }, b = "object" === typeof a ? e.extend(b, a): e.isFunction(f) ? e.extend(b, {
            over: a,
            out: f,
            selector: n
        }): e.extend(b, {
            over: a,
            out: a,
            selector: f
        }), g, h, j, k, l = function(b) {
            g = b.pageX;
            h = b.pageY
        }, m = function(a, d) {
            d.hoverIntent_t = clearTimeout(d.hoverIntent_t);
            if (Math.abs(j - g) + Math.abs(k - h) < b.sensitivity)
                return e(d).off("mousemove.hoverIntent", l), d.hoverIntent_s = 1, b.over.apply(d, [a]);
            j = g;
            k = h;
            d.hoverIntent_t = setTimeout(function() {
                m(a, d)
            },
            b.interval)
        };
        a = function(a) {
            var d = jQuery.extend({}, a), c = this;
            c.hoverIntent_t && (c.hoverIntent_t = clearTimeout(c.hoverIntent_t));
            "mouseenter" == a.type ? (j = d.pageX, k = d.pageY, e(c).on("mousemove.hoverIntent", l), 1 != c.hoverIntent_s && (c.hoverIntent_t = setTimeout(function() {
                m(d, c)
            }, b.interval))) : (e(c).off("mousemove.hoverIntent", l), 1 == c.hoverIntent_s && (c.hoverIntent_t = setTimeout(function() {
                c.hoverIntent_t = clearTimeout(c.hoverIntent_t);
                c.hoverIntent_s = 0;
                b.out.apply(c, [d])
            }, b.timeout)))
        };
        return this.on({
            "mouseenter.hoverIntent": a,
            "mouseleave.hoverIntent": a
        }, b.selector)
    }
})(jQuery);

/*
 * Superfish v1.5.13 - jQuery menu widget
 * Copyright (c) 2013 Joel Birch
 *
 * Dual licensed under the MIT and GPL licenses:
 * 	http://www.opensource.org/licenses/mit-license.php
 * 	http://www.gnu.org/licenses/gpl.html
 *
 */
(function(c) {
    c.fn.superfish = function(d) {
        var f = c.fn.superfish, a = f.c, g = c('<span class="' + a.arrowClass + '"> &#187;</span>'), h = function() {
            var e = c(this), b = l(e);
            clearTimeout(b.sfTimer);
            e.showSuperfishUl().siblings().hideSuperfishUl()
        }, j = function(e) {
            var b = c(this), d = l(b), a = f.op, g = function() {
                a.retainPath =- 1 < c.inArray(b[0], a.$path);
                b.hideSuperfishUl();
                a.$path.length && 1 > b.parents("li." + a.hoverClass).length && (a.onIdle.call(), c.proxy(h, a.$path, e)())
            };
            "click" === e.type ? g() : (clearTimeout(d.sfTimer), d.sfTimer = setTimeout(g,
            a.delay))
        }, l = function(e) {
            e.hasClass(a.menuClass) && c.error("Superfish requires you to update to a version of hoverIntent that supports event-delegation, such as this one: https://github.com/joeldbirch/onHoverIntent");
            e = e.closest("." + a.menuClass)[0];
            f.op = f.o[e.serial];
            return e
        }, n = function(e) {
            var b = c(this), a = b.siblings("ul");
            if (0 < a.length&&!a.is(":visible") && (b.data("follow", !1), "MSPointerDown" === e.type))
                return b.trigger("focus"), !1
        }, p = function(a) {
            var b = c(this), d = b.siblings("ul"), g=!1 === b.data("follow") ?
            !1 : !0;
            if (d.length && (f.op.useClick ||!g))
                a.preventDefault(), d.is(":visible") ? f.op.useClick && g && c.proxy(j, b.parent("li"), a)() : c.proxy(h, b.parent("li"))()
        };
        return this.addClass(a.menuClass).each(function() {
            var e = this.serial = f.o.length, b = c.extend({}, f.defaults, d), k = c(this), l = k.find("li:has(ul)");
            b.$path = k.find("li." + b.pathClass).slice(0, b.pathLevels).each(function() {
                c(this).addClass(b.hoverClass + " " + a.bcClass).filter("li:has(ul)").removeClass(b.pathClass)
            });
            f.o[e] = f.op = b;
            b.autoArrows && l.children("a").each(function() {
                c(this).addClass(a.anchorClass).append(g.clone())
            });
            k.css("ms-touch-action", "none");
            if (!f.op.useClick)
                if (c.fn.hoverIntent&&!f.op.disableHI)
                    k.hoverIntent(h, j, "li:has(ul)");
                else 
                    k.on("mouseenter", "li:has(ul)", h).on("mouseleave", "li:has(ul)", j);
            e = "MSPointerDown";
            navigator.userAgent.toLowerCase().match(/(iphone|ipod|ipad)/) || (e += " touchstart");
            k.on("focusin", "li", h).on("focusout", "li", j).on("click", "a", p).on(e, "a", n);
            l.not("." + a.bcClass).children("ul").show().hide();
            b.onInit.call(this)
        })
    };
    var g = c.fn.superfish;
    g.o = [];
    g.op = {};
    g.c = {
        bcClass: "sf-breadcrumb",
        menuClass: "sf-js-enabled",
        anchorClass: "sf-with-ul",
        arrowClass: "sf-sub-indicator"
    };
    g.defaults = {
        hoverClass: "sfHover",
        pathClass: "overideThisToUse",
        pathLevels: 1,
        delay: 800,
        animation: {
            opacity: "show"
        },
        animationOut: {
            opacity: "hide"
        },
        speed: "normal",
        speedOut: "fast",
        autoArrows: !0,
        disableHI: !1,
        useClick: !1,
        onInit: function() {},
        onBeforeShow: function() {},
        onShow: function() {},
        onHide: function() {},
        onIdle: function() {}
    };
    c.fn.extend({
        hideSuperfishUl: function() {
            var d = g.op, f = this, a=!0 === d.retainPath ? d.$path : "";
            d.retainPath =
            !1;
            c("li." + d.hoverClass, this).add(this).not(a).children("ul").stop(!0, !0).animate(d.animationOut, d.speedOut, function() {
                $ul = c(this);
                $ul.parent().removeClass(d.hoverClass);
                d.onHide.call($ul);
                g.op.useClick && f.children("a").data("follow", !1)
            });
            return this
        },
        showSuperfishUl: function() {
            var d = g.op, f = this, a = this.children("ul");
            f.addClass(d.hoverClass);
            d.onBeforeShow.call(a);
            a.stop(!0, !0).animate(d.animation, d.speed, function() {
                d.onShow.call(a);
                f.children("a").data("follow", !0)
            });
            var m = c(window).width(), h = this.children("ul").first();
            if (0 < h.length) {
                var j = this.children("ul").first().offset().left + h.width();
                j > m && (h.parent().parent().hasClass("sf-menu") ? h.css("left", "-" + (j - m) + "px") : h.addClass("reversed").css("left", "-" + (h.width() + 0) + "px"))
            }
            return this
        }
    })
})(jQuery);


/* ==========================================================
 * bootstrap-affix.js v2.3.1
 * http://twitter.github.com/bootstrap/javascript.html#affix
 * ==========================================================*/
!function(a) {
    var h = function(c, b) {
        this.options = a.extend({}, a.fn.affix.defaults, b);
        this.$window = a(window).on("scroll.affix.data-api", a.proxy(this.checkPosition, this)).on("click.affix.data-api", a.proxy(function() {
            setTimeout(a.proxy(this.checkPosition, this), 1)
        }, this));
        this.$element = a(c);
        this.checkPosition()
    };
    h.prototype.checkPosition = function() {
        if (this.$element.is(":visible")) {
            var c = a(document).height(), b = this.$window.scrollTop(), e = this.$element.offset(), d = this.options.offset, f = d.bottom, g = d.top;
            "object" !=
            typeof d && (f = g = d);
            "function" == typeof g && (g = d.top());
            "function" == typeof f && (f = d.bottom());
            c = null != this.unpin && b + this.unpin <= e.top?!1 : null != f && e.top + this.$element.height() >= c - f ? "bottom" : null != g && b <= g ? "top" : !1;
            this.affixed !== c && (this.affixed = c, this.unpin = "bottom" == c ? e.top - b : null, this.$element.removeClass("affix affix-top affix-bottom").addClass("affix" + (c ? "-" + c : "")))
        }
    };
    var j = a.fn.affix;
    a.fn.affix = function(c) {
        return this.each(function() {
            var b = a(this), e = b.data("affix"), d = "object" == typeof c && c;
            e || b.data("affix",
            e = new h(this, d));
            if ("string" == typeof c)
                e[c]()
        })
    };
    a.fn.affix.Constructor = h;
    a.fn.affix.defaults = {
        offset: 0
    };
    a.fn.affix.noConflict = function() {
        a.fn.affix = j;
        return this
    };
    a(window).on("load", function() {
        a('[data-spy="affix"]').each(function() {
            var c = a(this), b = c.data();
            b.offset = b.offset || {};
            b.offsetBottom && (b.offset.bottom = b.offsetBottom);
            b.offsetTop && (b.offset.top = b.offsetTop);
            c.affix(b)
        })
    })
}(window.jQuery);


/* ============================================================
 * bootstrap-dropdown.js v2.3.1
 * http://twitter.github.com/bootstrap/javascript.html#dropdowns
 * ============================================================ */
!function(d) {
    function h() {
        d(f).each(function() {
            g(d(this)).removeClass("open")
        })
    }
    function g(b) {
        var a = b.attr("data-target");
        a || (a = (a = b.attr("href")) && /#/.test(a) && a.replace(/.*(?=#[^\s]*$)/, ""));
        a = a && d(a);
        if (!a ||!a.length)
            a = b.parent();
        return a
    }
    var f = "[data-toggle=dropdown]", e = function(b) {
        var a = d(b).on("click.dropdown.data-api", this.toggle);
        d("html").on("click.dropdown.data-api", function() {
            a.parent().removeClass("open")
        })
    };
    e.prototype = {
        constructor: e,
        toggle: function() {
            var b = d(this), a, c;
            if (!b.is(".disabled, :disabled"))
                return a =
                g(b), c = a.hasClass("open"), h(), c || a.toggleClass("open"), b.focus(), !1
        },
        keydown: function(b) {
            var a, c, e;
            if (/(38|40|27)/.test(b.keyCode) && (a = d(this), b.preventDefault(), b.stopPropagation(), !a.is(".disabled, :disabled"))) {
                c = g(a);
                e = c.hasClass("open");
                if (!e || e && 27 == b.keyCode)
                    return 27 == b.which && c.find(f).focus(), a.click();
                a = d("[role=menu] li:not(.divider):visible a", c);
                a.length && (c = a.index(a.filter(":focus")), 38 == b.keyCode && 0 < c && c--, 40 == b.keyCode && c < a.length - 1 && c++, ~c || (c = 0), a.eq(c).focus())
            }
        }
    };
    var j = d.fn.dropdown;
    d.fn.dropdown = function(b) {
        return this.each(function() {
            var a = d(this), c = a.data("dropdown");
            c || a.data("dropdown", c = new e(this));
            "string" == typeof b && c[b].call(a)
        })
    };
    d.fn.dropdown.Constructor = e;
    d.fn.dropdown.noConflict = function() {
        d.fn.dropdown = j;
        return this
    };
    d(document).on("click.dropdown.data-api", h).on("click.dropdown.data-api", ".dropdown form", function(b) {
        b.stopPropagation()
    }).on("click.dropdown.data-api", f, e.prototype.toggle).on("keydown.dropdown.data-api", f + ", [role=menu]", e.prototype.keydown)
}(window.jQuery);



// Generated by CoffeeScript 1.4.0
/*
 jQuery Waypoints - v2.0.2
 Copyright (c) 2011-2013 Caleb Troughton
 Dual licensed under the MIT license and GPL license.
 https://github.com/imakewebthings/jquery-waypoints/blob/master/licenses.txt
 */
(function() {
    var t = [].indexOf || function(t) {
        for (var e = 0, n = this.length; e < n; e++) {
            if (e in this && this[e] === t)
                return e
        }
        return - 1
    }, e = [].slice;
    (function(t, e) {
        if (typeof define === "function" && define.amd) {
            return define("waypoints", ["jquery"], function(n) {
                return e(n, t)
            })
        } else {
            return e(t.jQuery, t)
        }
    })(this, function(n, r) {
        var i, o, l, s, f, u, a, c, h, d, p, y, v, w, g, m;
        i = n(r);
        c = t.call(r, "ontouchstart") >= 0;
        s = {
            horizontal: {},
            vertical: {}
        };
        f = 1;
        a = {};
        u = "waypoints-context-id";
        p = "resize.waypoints";
        y = "scroll.waypoints";
        v = 1;
        w = "waypoints-waypoint-ids";
        g = "waypoint";
        m = "waypoints";
        o = function() {
            function t(t) {
                var e = this;
                this.$element = t;
                this.element = t[0];
                this.didResize = false;
                this.didScroll = false;
                this.id = "context" + f++;
                this.oldScroll = {
                    x: t.scrollLeft(),
                    y: t.scrollTop()
                };
                this.waypoints = {
                    horizontal: {},
                    vertical: {}
                };
                t.data(u, this.id);
                a[this.id] = this;
                t.bind(y, function() {
                    var t;
                    if (!(e.didScroll || c)) {
                        e.didScroll = true;
                        t = function() {
                            e.doScroll();
                            return e.didScroll = false
                        };
                        return r.setTimeout(t, n[m].settings.scrollThrottle)
                    }
                });
                t.bind(p, function() {
                    var t;
                    if (!e.didResize) {
                        e.didResize = true;
                        t = function() {
                            n[m]("refresh");
                            return e.didResize = false
                        };
                        return r.setTimeout(t, n[m].settings.resizeThrottle)
                    }
                })
            }
            t.prototype.doScroll = function() {
                var t, e = this;
                t = {
                    horizontal: {
                        newScroll: this.$element.scrollLeft(),
                        oldScroll: this.oldScroll.x,
                        forward: "right",
                        backward: "left"
                    },
                    vertical: {
                        newScroll: this.$element.scrollTop(),
                        oldScroll: this.oldScroll.y,
                        forward: "down",
                        backward: "up"
                    }
                };
                if (c && (!t.vertical.oldScroll ||!t.vertical.newScroll)) {
                    n[m]("refresh")
                }
                n.each(t, function(t, r) {
                    var i, o, l;
                    l = [];
                    o = r.newScroll > r.oldScroll;
                    i = o ? r.forward : r.backward;
                    n.each(e.waypoints[t], function(t, e) {
                        var n, i;
                        if (r.oldScroll < (n = e.offset) && n <= r.newScroll) {
                            return l.push(e)
                        } else if (r.newScroll < (i = e.offset) && i <= r.oldScroll) {
                            return l.push(e)
                        }
                    });
                    l.sort(function(t, e) {
                        return t.offset - e.offset
                    });
                    if (!o) {
                        l.reverse()
                    }
                    return n.each(l, function(t, e) {
                        if (e.options.continuous || t === l.length - 1) {
                            return e.trigger([i])
                        }
                    })
                });
                return this.oldScroll = {
                    x: t.horizontal.newScroll,
                    y: t.vertical.newScroll
                }
            };
            t.prototype.refresh = function() {
                var t, e, r, i = this;
                r = n.isWindow(this.element);
                e = this.$element.offset();
                this.doScroll();
                t = {
                    horizontal: {
                        contextOffset: r ? 0: e.left,
                        contextScroll: r ? 0: this.oldScroll.x,
                        contextDimension: this.$element.width(),
                        oldScroll: this.oldScroll.x,
                        forward: "right",
                        backward: "left",
                        offsetProp: "left"
                    },
                    vertical: {
                        contextOffset: r ? 0: e.top,
                        contextScroll: r ? 0: this.oldScroll.y,
                        contextDimension: r ? n[m]("viewportHeight"): this.$element.height(),
                        oldScroll: this.oldScroll.y,
                        forward: "down",
                        backward: "up",
                        offsetProp: "top"
                    }
                };
                return n.each(t, function(t, e) {
                    return n.each(i.waypoints[t], function(t, r) {
                        var i, o, l, s, f;
                        i = r.options.offset;
                        l = r.offset;
                        o = n.isWindow(r.element) ? 0 : r.$element.offset()[e.offsetProp];
                        if (n.isFunction(i)) {
                            i = i.apply(r.element)
                        } else if (typeof i === "string") {
                            i = parseFloat(i);
                            if (r.options.offset.indexOf("%")>-1) {
                                i = Math.ceil(e.contextDimension * i / 100)
                            }
                        }
                        r.offset = o - e.contextOffset + e.contextScroll - i;
                        if (r.options.onlyOnScroll && l != null ||!r.enabled) {
                            return 
                        }
                        if (l !== null && l < (s = e.oldScroll) && s <= r.offset) {
                            return r.trigger([e.backward])
                        } else if (l !== null && l > (f = e.oldScroll) && f >= r.offset) {
                            return r.trigger([e.forward])
                        } else if (l === null && e.oldScroll >= r.offset) {
                            return r.trigger([e.forward])
                        }
                    })
                })
            };
            t.prototype.checkEmpty = function() {
                if (n.isEmptyObject(this.waypoints.horizontal) && n.isEmptyObject(this.waypoints.vertical)) {
                    this.$element.unbind([p, y].join(" "));
                    return delete a[this.id]
                }
            };
            return t
        }();
        l = function() {
            function t(t, e, r) {
                var i, o;
                r = n.extend({}, n.fn[g].defaults, r);
                if (r.offset === "bottom-in-view") {
                    r.offset = function() {
                        var t;
                        t = n[m]("viewportHeight");
                        if (!n.isWindow(e.element)) {
                            t = e.$element.height()
                        }
                        return t - n(this).outerHeight()
                    }
                }
                this.$element = t;
                this.element = t[0];
                this.axis = r.horizontal ? "horizontal" : "vertical";
                this.callback = r.handler;
                this.context = e;
                this.enabled = r.enabled;
                this.id = "waypoints" + v++;
                this.offset = null;
                this.options = r;
                e.waypoints[this.axis][this.id] = this;
                s[this.axis][this.id] = this;
                i = (o = t.data(w)) != null ? o : [];
                i.push(this.id);
                t.data(w, i)
            }
            t.prototype.trigger = function(t) {
                if (!this.enabled) {
                    return 
                }
                if (this.callback != null) {
                    this.callback.apply(this.element, t)
                }
                if (this.options.triggerOnce) {
                    return this.destroy()
                }
            };
            t.prototype.disable = function() {
                return this.enabled = false
            };
            t.prototype.enable = function() {
                this.context.refresh();
                return this.enabled = true
            };
            t.prototype.destroy = function() {
                delete s[this.axis][this.id];
                delete this.context.waypoints[this.axis][this.id];
                return this.context.checkEmpty()
            };
            t.getWaypointsByElement = function(t) {
                var e, r;
                r = n(t).data(w);
                if (!r) {
                    return []
                }
                e = n.extend({}, s.horizontal, s.vertical);
                return n.map(r, function(t) {
                    return e[t]
                })
            };
            return t
        }();
        d = {
            init: function(t, e) {
                var r;
                if (e == null) {
                    e = {}
                }
                if ((r = e.handler) == null) {
                    e.handler = t
                }
                this.each(function() {
                    var t, r, i, s;
                    t = n(this);
                    i = (s = e.context) != null ? s : n.fn[g].defaults.context;
                    if (!n.isWindow(i)) {
                        i = t.closest(i)
                    }
                    i = n(i);
                    r = a[i.data(u)];
                    if (!r) {
                        r = new o(i)
                    }
                    return new l(t, r, e)
                });
                n[m]("refresh");
                return this
            },
            disable: function() {
                return d._invoke(this, "disable")
            },
            enable: function() {
                return d._invoke(this, "enable")
            },
            destroy: function() {
                return d._invoke(this, "destroy")
            },
            prev: function(t, e) {
                return d._traverse.call(this, t, e, function(t, e, n) {
                    if (e > 0) {
                        return t.push(n[e - 1])
                    }
                })
            },
            next: function(t, e) {
                return d._traverse.call(this, t, e, function(t, e, n) {
                    if (e < n.length - 1) {
                        return t.push(n[e + 1])
                    }
                })
            },
            _traverse: function(t, e, i) {
                var o, l;
                if (t == null) {
                    t = "vertical"
                }
                if (e == null) {
                    e = r
                }
                l = h.aggregate(e);
                o = [];
                this.each(function() {
                    var e;
                    e = n.inArray(this, l[t]);
                    return i(o, e, l[t])
                });
                return this.pushStack(o)
            },
            _invoke: function(t, e) {
                t.each(function() {
                    var t;
                    t = l.getWaypointsByElement(this);
                    return n.each(t, function(t, n) {
                        n[e]();
                        return true
                    })
                });
                return this
            }
        };
        n.fn[g] = function() {
            var t, r;
            r = arguments[0], t = 2 <= arguments.length ? e.call(arguments, 1) : [];
            if (d[r]) {
                return d[r].apply(this, t)
            } else if (n.isFunction(r)) {
                return d.init.apply(this, arguments)
            } else if (n.isPlainObject(r)) {
                return d.init.apply(this, [null, r])
            } else if (!r) {
                return n.error("jQuery Waypoints needs a callback function or handler option.")
            } else {
                return n.error("The " + r + " method does not exist in jQuery Waypoints.")
            }
        };
        n.fn[g].defaults = {
            context: r,
            continuous: true,
            enabled: true,
            horizontal: false,
            offset: 0,
            triggerOnce: false
        };
        h = {
            refresh: function() {
                return n.each(a, function(t, e) {
                    return e.refresh()
                })
            },
            viewportHeight: function() {
                var t;
                return (t = r.innerHeight) != null ? t : i.height()
            },
            aggregate: function(t) {
                var e, r, i;
                e = s;
                if (t) {
                    e = (i = a[n(t).data(u)]) != null ? i.waypoints : void 0
                }
                if (!e) {
                    return []
                }
                r = {
                    horizontal: [],
                    vertical: []
                };
                n.each(r, function(t, i) {
                    n.each(e[t], function(t, e) {
                        return i.push(e)
                    });
                    i.sort(function(t, e) {
                        return t.offset - e.offset
                    });
                    r[t] = n.map(i, function(t) {
                        return t.element
                    });
                    return r[t] = n.unique(r[t])
                });
                return r
            },
            above: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "vertical", function(t, e) {
                    return e.offset <= t.oldScroll.y
                })
            },
            below: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "vertical", function(t, e) {
                    return e.offset > t.oldScroll.y
                })
            },
            left: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "horizontal", function(t, e) {
                    return e.offset <= t.oldScroll.x
                })
            },
            right: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "horizontal", function(t, e) {
                    return e.offset > t.oldScroll.x
                })
            },
            enable: function() {
                return h._invoke("enable")
            },
            disable: function() {
                return h._invoke("disable")
            },
            destroy: function() {
                return h._invoke("destroy")
            },
            extendFn: function(t, e) {
                return d[t] = e
            },
            _invoke: function(t) {
                var e;
                e = n.extend({}, s.vertical, s.horizontal);
                return n.each(e, function(e, n) {
                    n[t]();
                    return true
                })
            },
            _filter: function(t, e, r) {
                var i, o;
                i = a[n(t).data(u)];
                if (!i) {
                    return []
                }
                o = [];
                n.each(i.waypoints[e], function(t, e) {
                    if (r(i, e)) {
                        return o.push(e)
                    }
                });
                o.sort(function(t, e) {
                    return t.offset - e.offset
                });
                return n.map(o, function(t) {
                    return t.element
                })
            }
        };
        n[m] = function() {
            var t, n;
            n = arguments[0], t = 2 <= arguments.length ? e.call(arguments, 1) : [];
            if (h[n]) {
                return h[n].apply(null, t)
            } else {
                return h.aggregate.call(null, n)
            }
        };
        n[m].settings = {
            resizeThrottle: 100,
            scrollThrottle: 30
        };
        return i.load(function() {
            return n[m]("refresh")
        })
    })
}).call(this);


/*
 * jQuery FlexSlider v2.1
 * Copyright 2012 WooThemes
 * Contributing Author: Tyler Smith
 */
;
(function(d) {
    d.flexslider = function(i, k) {
        var a = d(i), c = d.extend({}, d.flexslider.defaults, k), e = c.namespace, p = "ontouchstart"in window || window.DocumentTouch && document instanceof DocumentTouch, t = p ? "touchend": "click", l = "vertical" === c.direction, m = c.reverse, h = 0 < c.itemWidth, r = "fade" === c.animation, s = "" !== c.asNavFor, f = {};
        d.data(i, "flexslider", a);
        f = {
            init: function() {
                a.animating=!1;
                a.currentSlide = c.startAt;
                a.animatingTo = a.currentSlide;
                a.atEnd = 0 === a.currentSlide || a.currentSlide === a.last;
                a.containerSelector = c.selector.substr(0,
                c.selector.search(" "));
                a.slides = d(c.selector, a);
                a.container = d(a.containerSelector, a);
                a.count = a.slides.length;
                a.syncExists = 0 < d(c.sync).length;
                "slide" === c.animation && (c.animation = "swing");
                a.prop = l ? "top" : "marginLeft";
                a.args = {};
                a.manualPause=!1;
                var b = a, g;
                if (g=!c.video)
                    if (g=!r)
                        if (g = c.useCSS)
                            a: {
                                g = document.createElement("div");
                                var n = ["perspectiveProperty", "WebkitPerspective", "MozPerspective", "OPerspective", "msPerspective"], e;
                                for (e in n)
                                    if (void 0 !== g.style[n[e]]) {
                                        a.pfx = n[e].replace("Perspective", "").toLowerCase();
                                        a.prop = "-" + a.pfx + "-transform";
                                        g=!0;
                                        break a
                                    }
                                    g=!1
                                }
                b.transitions = g;
                "" !== c.controlsContainer && (a.controlsContainer = 0 < d(c.controlsContainer).length && d(c.controlsContainer));
                "" !== c.manualControls && (a.manualControls = 0 < d(c.manualControls).length && d(c.manualControls));
                c.randomize && (a.slides.sort(function() {
                    return Math.round(Math.random()) - 0.5
                }), a.container.empty().append(a.slides));
                a.doMath();
                s && f.asNav.setup();
                a.setup("init");
                c.controlNav && f.controlNav.setup();
                c.directionNav && f.directionNav.setup();
                c.keyboard &&
                (1 === d(a.containerSelector).length || c.multipleKeyboard) && d(document).bind("keyup", function(b) {
                    b = b.keyCode;
                    if (!a.animating && (39 === b || 37 === b))
                        b = 39 === b ? a.getTarget("next") : 37 === b ? a.getTarget("prev") : !1, a.flexAnimate(b, c.pauseOnAction)
                });
                c.mousewheel && a.bind("mousewheel", function(b, g) {
                    b.preventDefault();
                    var d = 0 > g ? a.getTarget("next"): a.getTarget("prev");
                    a.flexAnimate(d, c.pauseOnAction)
                });
                c.pausePlay && f.pausePlay.setup();
                c.slideshow && (c.pauseOnHover && a.hover(function() {
                    !a.manualPlay&&!a.manualPause && a.pause()
                },
                function() {
                    !a.manualPause&&!a.manualPlay && a.play()
                }), 0 < c.initDelay ? setTimeout(a.play, c.initDelay) : a.play());
                p && c.touch && f.touch();
                (!r || r && c.smoothHeight) && d(window).bind("resize focus", f.resize);
                setTimeout(function() {
                    c.start(a)
                }, 200)
            },
            asNav: {
                setup: function() {
                    a.asNav=!0;
                    a.animatingTo = Math.floor(a.currentSlide / a.move);
                    a.currentItem = a.currentSlide;
                    a.slides.removeClass(e + "active-slide").eq(a.currentItem).addClass(e + "active-slide");
                    a.slides.click(function(b) {
                        b.preventDefault();
                        var b = d(this), g = b.index();
                        !d(c.asNavFor).data("flexslider").animating&&!b.hasClass("active") && (a.direction = a.currentItem < g ? "next" : "prev", a.flexAnimate(g, c.pauseOnAction, !1, !0, !0))
                    })
                }
            },
            controlNav: {
                setup: function() {
                    a.manualControls ? f.controlNav.setupManual() : f.controlNav.setupPaging()
                },
                setupPaging: function() {
                    var b = 1, g;
                    a.controlNavScaffold = d('<ol class="' + e + "control-nav " + e + ("thumbnails" === c.controlNav ? "control-thumbs" : "control-paging") + '"></ol>');
                    if (1 < a.pagingCount)
                        for (var n = 0; n < a.pagingCount; n++)
                            g = "thumbnails" === c.controlNav ?
                            '<img src="' + a.slides.eq(n).attr("data-thumb") + '"/>' : "<a>" + b + "</a>", a.controlNavScaffold.append("<li>" + g + "</li>"), b++;
                    a.controlsContainer ? d(a.controlsContainer).append(a.controlNavScaffold) : a.append(a.controlNavScaffold);
                    f.controlNav.set();
                    f.controlNav.active();
                    a.controlNavScaffold.delegate("a, img", t, function(b) {
                        b.preventDefault();
                        var b = d(this), g = a.controlNav.index(b);
                        b.hasClass(e + "active") || (a.direction = g > a.currentSlide ? "next" : "prev", a.flexAnimate(g, c.pauseOnAction))
                    });
                    p && a.controlNavScaffold.delegate("a",
                    "click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                setupManual: function() {
                    a.controlNav = a.manualControls;
                    f.controlNav.active();
                    a.controlNav.live(t, function(b) {
                        b.preventDefault();
                        var b = d(this), g = a.controlNav.index(b);
                        b.hasClass(e + "active") || (g > a.currentSlide ? a.direction = "next" : a.direction = "prev", a.flexAnimate(g, c.pauseOnAction))
                    });
                    p && a.controlNav.live("click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                set: function() {
                    a.controlNav = d("." + e + "control-nav li " + ("thumbnails" === c.controlNav ? "img" : "a"),
                    a.controlsContainer ? a.controlsContainer : a)
                },
                active: function() {
                    a.controlNav.removeClass(e + "active").eq(a.animatingTo).addClass(e + "active")
                },
                update: function(b, c) {
                    1 < a.pagingCount && "add" === b ? a.controlNavScaffold.append(d("<li><a>" + a.count + "</a></li>")) : 1 === a.pagingCount ? a.controlNavScaffold.find("li").remove() : a.controlNav.eq(c).closest("li").remove();
                    f.controlNav.set();
                    1 < a.pagingCount && a.pagingCount !== a.controlNav.length ? a.update(c, b) : f.controlNav.active()
                }
            },
            directionNav: {
                setup: function() {
                    var b = d('<ul class="' +
                    e + 'direction-nav"><li><a class="' + e + 'prev" href="#">' + c.prevText + '</a></li><li><a class="' + e + 'next" href="#">' + c.nextText + "</a></li></ul>");
                    a.controlsContainer ? (d(a.controlsContainer).append(b), a.directionNav = d("." + e + "direction-nav li a", a.controlsContainer)) : (a.append(b), a.directionNav = d("." + e + "direction-nav li a", a));
                    f.directionNav.update();
                    a.directionNav.bind(t, function(b) {
                        b.preventDefault();
                        b = d(this).hasClass(e + "next") ? a.getTarget("next") : a.getTarget("prev");
                        a.flexAnimate(b, c.pauseOnAction)
                    });
                    p && a.directionNav.bind("click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                update: function() {
                    var b = e + "disabled";
                    1 === a.pagingCount ? a.directionNav.addClass(b) : c.animationLoop ? a.directionNav.removeClass(b) : 0 === a.animatingTo ? a.directionNav.removeClass(b).filter("." + e + "prev").addClass(b) : a.animatingTo === a.last ? a.directionNav.removeClass(b).filter("." + e + "next").addClass(b) : a.directionNav.removeClass(b)
                }
            },
            pausePlay: {
                setup: function() {
                    var b = d('<div class="' + e + 'pauseplay"><a></a></div>');
                    a.controlsContainer ?
                    (a.controlsContainer.append(b), a.pausePlay = d("." + e + "pauseplay a", a.controlsContainer)) : (a.append(b), a.pausePlay = d("." + e + "pauseplay a", a));
                    f.pausePlay.update(c.slideshow ? e + "pause" : e + "play");
                    a.pausePlay.bind(t, function(b) {
                        b.preventDefault();
                        d(this).hasClass(e + "pause") ? (a.manualPause=!0, a.manualPlay=!1, a.pause()) : (a.manualPause=!1, a.manualPlay=!0, a.play())
                    });
                    p && a.pausePlay.bind("click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                update: function(b) {
                    "play" === b ? a.pausePlay.removeClass(e + "pause").addClass(e +
                    "play").text(c.playText) : a.pausePlay.removeClass(e + "play").addClass(e + "pause").text(c.pauseText)
                }
            },
            touch: function() {
                function b(b) {
                    j = l ? d - b.touches[0].pageY : d - b.touches[0].pageX;
                    p = l ? Math.abs(j) < Math.abs(b.touches[0].pageX - e) : Math.abs(j) < Math.abs(b.touches[0].pageY - e);
                    if (!p || 500 < Number(new Date) - k)
                        b.preventDefault(), !r && a.transitions && (c.animationLoop || (j/=0 === a.currentSlide && 0 > j || a.currentSlide === a.last && 0 < j ? Math.abs(j) / q + 2 : 1), a.setProps(f + j, "setTouch"))
                }
                function g() {
                    i.removeEventListener("touchmove",
                    b, !1);
                    if (a.animatingTo === a.currentSlide&&!p && null !== j) {
                        var h = m?-j : j, l = 0 < h ? a.getTarget("next") : a.getTarget("prev");
                        a.canAdvance(l) && (550 > Number(new Date) - k && 50 < Math.abs(h) || Math.abs(h) > q / 2) ? a.flexAnimate(l, c.pauseOnAction) : r || a.flexAnimate(a.currentSlide, c.pauseOnAction, !0)
                    }
                    i.removeEventListener("touchend", g, !1);
                    f = j = e = d = null
                }
                var d, e, f, q, j, k, p=!1;
                i.addEventListener("touchstart", function(j) {
                    a.animating ? j.preventDefault() : 1 === j.touches.length && (a.pause(), q = l ? a.h : a.w, k = Number(new Date), f = h && m && a.animatingTo ===
                    a.last ? 0 : h && m ? a.limit - (a.itemW + c.itemMargin) * a.move * a.animatingTo : h && a.currentSlide === a.last ? a.limit : h ? (a.itemW + c.itemMargin) * a.move * a.currentSlide : m ? (a.last - a.currentSlide + a.cloneOffset) * q : (a.currentSlide + a.cloneOffset) * q, d = l ? j.touches[0].pageY : j.touches[0].pageX, e = l ? j.touches[0].pageX : j.touches[0].pageY, i.addEventListener("touchmove", b, !1), i.addEventListener("touchend", g, !1))
                }, !1)
            },
            resize: function() {
                !a.animating && a.is(":visible") && (h || a.doMath(), r ? f.smoothHeight() : h ? (a.slides.width(a.computedW),
                a.update(a.pagingCount), a.setProps()) : l ? (a.viewport.height(a.h), a.setProps(a.h, "setTotal")) : (c.smoothHeight && f.smoothHeight(), a.newSlides.width(a.computedW), a.setProps(a.computedW, "setTotal")))
            },
            smoothHeight: function(b) {
                if (!l || r) {
                    var c = r ? a: a.viewport;
                    b ? c.animate({
                        height: a.slides.eq(a.animatingTo).height()
                    }, b) : c.height(a.slides.eq(a.animatingTo).height())
                }
            },
            sync: function(b) {
                var g = d(c.sync).data("flexslider"), e = a.animatingTo;
                switch (b) {
                case "animate":
                    g.flexAnimate(e, c.pauseOnAction, !1, !0);
                    break;
                case "play":
                    !g.playing &&
                    !g.asNav && g.play();
                    break;
                case "pause":
                    g.pause()
                }
            }
        };
        a.flexAnimate = function(b, g, n, i, k) {
            s && 1 === a.pagingCount && (a.direction = a.currentItem < b ? "next" : "prev");
            if (!a.animating && (a.canAdvance(b, k) || n) && a.is(":visible")) {
                if (s && i)
                    if (n = d(c.asNavFor).data("flexslider"), a.atEnd = 0 === b || b === a.count - 1, n.flexAnimate(b, !0, !1, !0, k), a.direction = a.currentItem < b ? "next" : "prev", n.direction = a.direction, Math.ceil((b + 1) / a.visible) - 1 !== a.currentSlide && 0 !== b)
                        a.currentItem = b, a.slides.removeClass(e + "active-slide").eq(b).addClass(e +
                        "active-slide"), b = Math.floor(b / a.visible);
                    else 
                        return a.currentItem = b, a.slides.removeClass(e + "active-slide").eq(b).addClass(e + "active-slide"), !1;
                a.animating=!0;
                a.animatingTo = b;
                c.before(a);
                g && a.pause();
                a.syncExists&&!k && f.sync("animate");
                c.controlNav && f.controlNav.active();
                h || a.slides.removeClass(e + "active-slide").eq(b).addClass(e + "active-slide");
                a.atEnd = 0 === b || b === a.last;
                c.directionNav && f.directionNav.update();
                b === a.last && (c.end(a), c.animationLoop || a.pause());
                if (r)
                    p ? (a.slides.eq(a.currentSlide).css({
                        opacity: 0,
                        zIndex: 1
                    }), a.slides.eq(b).css({
                        opacity: 1,
                        zIndex: 2
                    }), a.slides.unbind("webkitTransitionEnd transitionend"), a.slides.eq(a.currentSlide).bind("webkitTransitionEnd transitionend", function() {
                        c.after(a)
                    }), a.animating=!1, a.currentSlide = a.animatingTo) : (a.slides.eq(a.currentSlide).fadeOut(c.animationSpeed, c.easing), a.slides.eq(b).fadeIn(c.animationSpeed, c.easing, a.wrapup));
                else {
                    var q = l ? a.slides.filter(":first").height(): a.computedW;
                    h ? (b = c.itemWidth > a.w ? 2 * c.itemMargin : c.itemMargin, b = (a.itemW + b) * a.move * a.animatingTo,
                    b = b > a.limit && 1 !== a.visible ? a.limit : b) : b = 0 === a.currentSlide && b === a.count - 1 && c.animationLoop && "next" !== a.direction ? m ? (a.count + a.cloneOffset) * q : 0 : a.currentSlide === a.last && 0 === b && c.animationLoop && "prev" !== a.direction ? m ? 0 : (a.count + 1) * q : m ? (a.count - 1 - b + a.cloneOffset) * q : (b + a.cloneOffset) * q;
                    a.setProps(b, "", c.animationSpeed);
                    if (a.transitions) {
                        if (!c.animationLoop ||!a.atEnd)
                            a.animating=!1, a.currentSlide = a.animatingTo;
                        a.container.unbind("webkitTransitionEnd transitionend");
                        a.container.bind("webkitTransitionEnd transitionend",
                        function() {
                            a.wrapup(q)
                        })
                    } else 
                        a.container.animate(a.args, c.animationSpeed, c.easing, function() {
                            a.wrapup(q)
                        })
                    }
                c.smoothHeight && f.smoothHeight(c.animationSpeed)
            }
        };
        a.wrapup = function(b) {
            !r&&!h && (0 === a.currentSlide && a.animatingTo === a.last && c.animationLoop ? a.setProps(b, "jumpEnd") : a.currentSlide === a.last && (0 === a.animatingTo && c.animationLoop) && a.setProps(b, "jumpStart"));
            a.animating=!1;
            a.currentSlide = a.animatingTo;
            c.after(a)
        };
        a.animateSlides = function() {
            a.animating || a.flexAnimate(a.getTarget("next"))
        };
        a.pause =
        function() {
            clearInterval(a.animatedSlides);
            a.playing=!1;
            c.pausePlay && f.pausePlay.update("play");
            a.syncExists && f.sync("pause")
        };
        a.play = function() {
            a.animatedSlides = setInterval(a.animateSlides, c.slideshowSpeed);
            a.playing=!0;
            c.pausePlay && f.pausePlay.update("pause");
            a.syncExists && f.sync("play")
        };
        a.canAdvance = function(b, g) {
            var d = s ? a.pagingCount - 1: a.last;
            return g?!0 : s && a.currentItem === a.count - 1 && 0 === b && "prev" === a.direction?!0 : s && 0 === a.currentItem && b === a.pagingCount - 1 && "next" !== a.direction?!1 : b === a.currentSlide &&
            !s?!1 : c.animationLoop?!0 : a.atEnd && 0 === a.currentSlide && b === d && "next" !== a.direction?!1 : a.atEnd && a.currentSlide === d && 0 === b && "next" === a.direction?!1 : !0
        };
        a.getTarget = function(b) {
            a.direction = b;
            return "next" === b ? a.currentSlide === a.last ? 0 : a.currentSlide + 1 : 0 === a.currentSlide ? a.last : a.currentSlide - 1
        };
        a.setProps = function(b, g, d) {
            var e, f = b ? b: (a.itemW + c.itemMargin) * a.move * a.animatingTo;
            e =- 1 * function() {
                if (h)
                    return "setTouch" === g ? b : m && a.animatingTo === a.last ? 0 : m ? a.limit - (a.itemW + c.itemMargin) * a.move * a.animatingTo : a.animatingTo ===
                    a.last ? a.limit : f;
                switch (g) {
                case "setTotal":
                    return m ? (a.count - 1 - a.currentSlide + a.cloneOffset) * b : (a.currentSlide + a.cloneOffset) * b;
                case "setTouch":
                    return b;
                case "jumpEnd":
                    return m ? b : a.count * b;
                case "jumpStart":
                    return m ? a.count * b : b;
                default:
                    return b
                }
            }() + "px";
            a.transitions && (e = l ? "translate3d(0," + e + ",0)" : "translate3d(" + e + ",0,0)", d = void 0 !== d ? d / 1E3 + "s" : "0s", a.container.css("-" + a.pfx + "-transition-duration", d));
            a.args[a.prop] = e;
            (a.transitions || void 0 === d) && a.container.css(a.args)
        };
        a.setup = function(b) {
            if (r)
                a.slides.css({
                    width: "100%",
                    "float": "left",
                    marginRight: "-100%",
                    position: "relative"
                }), "init" === b && (p ? a.slides.css({
                    opacity: 0,
                    display: "block",
                    webkitTransition: "opacity " + c.animationSpeed / 1E3 + "s ease",
                    zIndex: 1
                }).eq(a.currentSlide).css({
                    opacity: 1,
                    zIndex: 2
                }) : a.slides.eq(a.currentSlide).fadeIn(c.animationSpeed, c.easing)), c.smoothHeight && f.smoothHeight();
            else {
                var g, n;
                "init" === b && (a.viewport = d('<div class="' + e + 'viewport"></div>').css({
                    overflow: "hidden",
                    position: "relative"
                }).appendTo(a).append(a.container), a.cloneCount = 0, a.cloneOffset =
                0, m && (n = d.makeArray(a.slides).reverse(), a.slides = d(n), a.container.empty().append(a.slides)));
                c.animationLoop&&!h && (a.cloneCount = 2, a.cloneOffset = 1, "init" !== b && a.container.find(".clone").remove(), a.container.append(a.slides.first().clone().addClass("clone")).prepend(a.slides.last().clone().addClass("clone")));
                a.newSlides = d(c.selector, a);
                g = m ? a.count - 1 - a.currentSlide + a.cloneOffset : a.currentSlide + a.cloneOffset;
                l&&!h ? (a.container.height(200 * (a.count + a.cloneCount) + "%").css("position", "absolute").width("100%"),
                setTimeout(function() {
                    a.newSlides.css({
                        display: "block"
                    });
                    a.doMath();
                    a.viewport.height(a.h);
                    a.setProps(g * a.h, "init")
                }, "init" === b ? 100 : 0)) : (a.container.width(200 * (a.count + a.cloneCount) + "%"), a.setProps(g * a.computedW, "init"), setTimeout(function() {
                    a.doMath();
                    a.newSlides.css({
                        width: a.computedW,
                        "float": "left",
                        display: "block"
                    });
                    c.smoothHeight && f.smoothHeight()
                }, "init" === b ? 100 : 0))
            }
            h || a.slides.removeClass(e + "active-slide").eq(a.currentSlide).addClass(e + "active-slide")
        };
        a.doMath = function() {
            var b = a.slides.first(),
            d = c.itemMargin, e = c.minItems, f = c.maxItems;
            a.w = a.width();
            a.h = b.height();
            a.boxPadding = b.outerWidth() - b.width();
            h ? (a.itemT = c.itemWidth + d, a.minW = e ? e * a.itemT : a.w, a.maxW = f ? f * a.itemT : a.w, a.itemW = a.minW > a.w ? (a.w - d * e) / e : a.maxW < a.w ? (a.w - d * f) / f : c.itemWidth > a.w ? a.w : c.itemWidth, a.visible = Math.floor(a.w / (a.itemW + d)), a.move = 0 < c.move && c.move < a.visible ? c.move : a.visible, a.pagingCount = Math.ceil((a.count - a.visible) / a.move + 1), a.last = a.pagingCount - 1, a.limit = 1 === a.pagingCount ? 0 : c.itemWidth > a.w ? (a.itemW + 2 * d) * a.count - a.w -
            d : (a.itemW + d) * a.count - a.w - d) : (a.itemW = a.w, a.pagingCount = a.count, a.last = a.count - 1);
            a.computedW = a.itemW - a.boxPadding
        };
        a.update = function(b, d) {
            a.doMath();
            h || (b < a.currentSlide ? a.currentSlide += 1 : b <= a.currentSlide && 0 !== b && (a.currentSlide -= 1), a.animatingTo = a.currentSlide);
            if (c.controlNav&&!a.manualControls)
                if ("add" === d&&!h || a.pagingCount > a.controlNav.length)
                    f.controlNav.update("add");
                else if ("remove" === d&&!h || a.pagingCount < a.controlNav.length)
                    h && a.currentSlide > a.last && (a.currentSlide -= 1, a.animatingTo -= 1),
                    f.controlNav.update("remove", a.last);
            c.directionNav && f.directionNav.update()
        };
        a.addSlide = function(b, e) {
            var f = d(b);
            a.count += 1;
            a.last = a.count - 1;
            l && m ? void 0 !== e ? a.slides.eq(a.count - e).after(f) : a.container.prepend(f) : void 0 !== e ? a.slides.eq(e).before(f) : a.container.append(f);
            a.update(e, "add");
            a.slides = d(c.selector + ":not(.clone)", a);
            a.setup();
            c.added(a)
        };
        a.removeSlide = function(b) {
            var e = isNaN(b) ? a.slides.index(d(b)): b;
            a.count -= 1;
            a.last = a.count - 1;
            isNaN(b) ? d(b, a.slides).remove() : l && m ? a.slides.eq(a.last).remove() :
            a.slides.eq(b).remove();
            a.doMath();
            a.update(e, "remove");
            a.slides = d(c.selector + ":not(.clone)", a);
            a.setup();
            c.removed(a)
        };
        f.init()
    };
    d.flexslider.defaults = {
        namespace: "flex-",
        selector: ".slides > li",
        animation: "fade",
        easing: "swing",
        direction: "horizontal",
        reverse: !1,
        animationLoop: !0,
        smoothHeight: !1,
        startAt: 0,
        slideshow: !0,
        slideshowSpeed: 7E3,
        animationSpeed: 600,
        initDelay: 0,
        randomize: !1,
        pauseOnAction: !0,
        pauseOnHover: !1,
        useCSS: !0,
        touch: !0,
        video: !1,
        controlNav: !0,
        directionNav: !0,
        prevText: "Previous",
        nextText: "Next",
        keyboard: !0,
        multipleKeyboard: !1,
        mousewheel: !1,
        pausePlay: !1,
        pauseText: "Pause",
        playText: "Play",
        controlsContainer: "",
        manualControls: "",
        sync: "",
        asNavFor: "",
        itemWidth: 0,
        itemMargin: 0,
        minItems: 0,
        maxItems: 0,
        move: 0,
        start: function() {},
        before: function() {},
        after: function() {},
        end: function() {},
        added: function() {},
        removed: function() {}
    };
    d.fn.flexslider = function(i) {
        void 0 === i && (i = {});
        if ("object" === typeof i)
            return this.each(function() {
                var a = d(this), c = a.find(i.selector ? i.selector : ".slides > li");
                1 === c.length ? (c.fadeIn(400),
                i.start && i.start(a)) : void 0 == a.data("flexslider") && new d.flexslider(this, i)
            });
        var k = d(this).data("flexslider");
        switch (i) {
        case "play":
            k.play();
            break;
        case "pause":
            k.pause();
            break;
        case "next":
            k.flexAnimate(k.getTarget("next"), !0);
            break;
        case "prev":
        case "previous":
            k.flexAnimate(k.getTarget("prev"), !0);
            break;
        default:
            "number" === typeof i && k.flexAnimate(i, !0)
        }
    }
})(jQuery);



/*
 * iosSlider - http://iosscripts.com/iosslider/
 */

(function(b) {
    var na = 0, W = 0, da = 0, L = 0, za = "ontouchstart"in window, Ba = "onorientationchange"in window, ea=!1, aa=!1, X=!1, oa=!1, ha=!1, ia = "pointer", sa = "pointer", ja = [], T = [], ta = [], $ = [], B = [], ba = [], x = [], m = [], s = [], ua = [], fa = [], e = {
        showScrollbar: function(a, e) {
            a.scrollbarHide && b("." + e).css({
                opacity: a.scrollbarOpacity,
                filter: "alpha(opacity:" + 100 * a.scrollbarOpacity + ")"
            })
        },
        hideScrollbar: function(b, f, c, g, h, d, m, s, B, x) {
            if (b.scrollbar && b.scrollbarHide)
                for (var t = c; t < c + 25; t++)
                    f[f.length] = e.hideScrollbarIntervalTimer(10 * t, g[c], (c + 24 - t) / 24, h, d, m, s, B, x, b)
        },
        hideScrollbarInterval: function(a, f, c, g, h, d, m, B, x) {
            L =- 1 * a / s[B] * (h - d - m - g);
            e.setSliderOffset("." + c, L);
            b("." + c).css({
                opacity: x.scrollbarOpacity * f,
                filter: "alpha(opacity:" + 100 * x.scrollbarOpacity * f + ")"
            })
        },
        slowScrollHorizontalInterval: function(a, f, c, g, h, d, K, O, N, J, t, y, z, w, v, q, G, p, n) {
            if (n.infiniteSlider) {
                if (c<=-1 * s[q]) {
                    var r = b(a).width();
                    if (c<=-1 * ua[q]) {
                        var u =- 1 * t[0];
                        b(f).each(function(c) {
                            e.setSliderOffset(b(f)[c], u + G);
                            c < y.length && (y[c] =- 1 * u);
                            u += v[c]
                        });
                        c +=- 1 * y[0];
                        m[q] =- 1 * y[0] + G;
                        s[q] = m[q] + r - d;
                        x[q] = 0
                    } else {
                        var k = 0, C = e.getSliderOffset(b(f[0]), "x");
                        b(f).each(function(c) {
                            e.getSliderOffset(this, "x") < C && (C = e.getSliderOffset(this, "x"), k = c)
                        });
                        z = m[q] + r;
                        e.setSliderOffset(b(f)[k], z);
                        m[q] =- 1 * y[1] + G;
                        s[q] = m[q] + r - d;
                        y.splice(0, 1);
                        y.splice(y.length, 0, - 1 * z + G);
                        x[q]++
                    }
                }
                if (c>=-1 * m[q] || 0 <= c) {
                    r = b(a).width();
                    if (0 <= c)
                        for (u =- 1 * t[0], b(f).each(function(c) {
                            e.setSliderOffset(b(f)[c], u + G);
                            c < y.length && (y[c] =- 1 * u);
                            u += v[c]
                        }), c-=-1 * y[0], m[q] =- 1 * y[0] + G, s[q] = m[q] + r - d, x[q] = w; 0<-1 * y[0] - r + G;) {
                            var A = 0, I = e.getSliderOffset(b(f[0]), "x");
                            b(f).each(function(c) {
                                e.getSliderOffset(this, "x") > I && (I = e.getSliderOffset(this, "x"), A = c)
                            });
                            z = m[q] - v[A];
                            e.setSliderOffset(b(f)[A], z);
                            y.splice(0, 0, - 1 * z + G);
                            y.splice(y.length - 1, 1);
                            m[q] =- 1 * y[0] + G;
                            s[q] = m[q] + r - d;
                            x[q]--;
                            B[q]++
                        }
                    0 > c && (A = 0, I = e.getSliderOffset(b(f[0]), "x"), b(f).each(function(c) {
                        e.getSliderOffset(this, "x") > I && (I = e.getSliderOffset(this, "x"), A = c)
                    }), z = m[q] - v[A], e.setSliderOffset(b(f)[A], z), y.splice(0, 0, - 1 * z + G), y.splice(y.length - 1, 1), m[q] =- 1 * y[0] + G, s[q] = m[q] + r - d, x[q]--)
                }
            }
            t=!1;
            d = e.calcActiveOffset(n, c, y, d, x[q], w, J, q);
            z = (d + x[q] + w)%w;
            n.infiniteSlider ? z != ba[q] && (t=!0) : d != B[q] && (t=!0);
            if (t && (w = new e.args("change", n, a, b(a).children(":eq(" + z + ")"), z, p), b(a).parent().data("args", w), "" != n.onSlideChange))
                n.onSlideChange(w);
            B[q] = d;
            ba[q] = z;
            c = Math.floor(c);
            e.setSliderOffset(a, c);
            n.scrollbar && (L = Math.floor(( - 1 * c - m[q] + G) / (s[q] - m[q] + G) * (K - O - h)), a = h - N, c>=-1 * m[q] + G ? (a = h - N - - 1 * L, e.setSliderOffset(b("." + g), 0)) : (c<=-1 * s[q] + 1 && (a = K - O - N - L), e.setSliderOffset(b("." + g), L)), b("." + g).css({
                width: a + "px"
            }))
        },
        slowScrollHorizontal: function(a, f, c, g, h, d, K, O, N, J, t, y, z, w, v, q, G, p, n, r, u) {
            var k = e.getSliderOffset(a, "x");
            d = [];
            var C = 0, A = 25 / 1024 * O;
            frictionCoefficient = u.frictionCoefficient;
            elasticFrictionCoefficient = u.elasticFrictionCoefficient;
            snapFrictionCoefficient = u.snapFrictionCoefficient;
            h > u.snapVelocityThreshold && u.snapToChildren&&!n ? C = 1 : h<-1 * u.snapVelocityThreshold && (u.snapToChildren&&!n) && (C =- 1);
            h<-1 * A ? h =- 1 * A : h > A && (h = A);
            b(a)[0] !== b(p)[0] && (C*=-1, h*=-2);
            p = x[v];
            if (u.infiniteSlider)
                var I = m[v], l = s[v];
            n = [];
            for (var A = [], E = 0; E < z.length; E++)
                n[E] = z[E], E < f.length && (A[E] = e.getSliderOffset(b(f[E]), "x"));
            for (; 1 < h||-1 > h;) {
                h*=frictionCoefficient;
                k += h;
                (k>-1 * m[v] || k<-1 * s[v])&&!u.infiniteSlider && (h*=elasticFrictionCoefficient, k += h);
                if (u.infiniteSlider) {
                    if (k<=-1 * l) {
                        for (var l = b(a).width(), L = 0, Q = A[0], E = 0; E < A.length; E++)
                            A[E] < Q && (Q = A[E], L = E);
                        E = I + l;
                        A[L] = E;
                        I =- 1 * n[1] + r;
                        l = I + l - O;
                        n.splice(0, 1);
                        n.splice(n.length, 0, - 1 * E + r);
                        p++
                    }
                    if (k>=-1 * I) {
                        l = b(a).width();
                        L = 0;
                        Q = A[0];
                        for (E = 0; E < A.length; E++)
                            A[E] > Q && (Q = A[E], L = E);
                        E = I - w[L];
                        A[L] = E;
                        n.splice(0, 0, - 1 * E + r);
                        n.splice(n.length - 1, 1);
                        I =- 1 * n[0] + r;
                        l = I + l - O;
                        p--
                    }
                }
                d[d.length] = k
            }
            A=!1;
            h = e.calcActiveOffset(u, k, n, O, p, G, B[v], v);
            I = (h + p + G)%G;
            u.snapToChildren && (u.infiniteSlider ? I != ba[v] && (A=!0) : h != B[v] && (A=!0), 0 > C&&!A ? (h++, h >= z.length&&!u.infinteSlider && (h = z.length - 1)) : 0 < C&&!A && (h--, 0 > h&&!u.infinteSlider && (h = 0)));
            if (u.snapToChildren || (k>-1 * m[v] || k<-1 * s[v])&&!u.infiniteSlider) {
                (k>-1 * m[v] || k<-1 * s[v])&&!u.infiniteSlider ? d.splice(0, d.length) : (d.splice(0.1 * d.length, d.length), k = 0 < d.length ? d[d.length - 1] : k);
                for (; k < n[h] - 0.5 || k > n[h] + 0.5;)
                    k = (k - n[h]) * snapFrictionCoefficient + n[h], d[d.length] = k;
                d[d.length] = n[h]
            }
            C = 1;
            0 != d.length%2 && (C = 0);
            for (k = 0; k < c.length; k++)
                clearTimeout(c[k]);
            p = (h + p + G)%G;
            I = 0;
            for (k = C; k < d.length; k += 2)
                if (k == C || 1 < Math.abs(d[k] - I) || k >= d.length - 2)
                    I = d[k], c[c.length] = e.slowScrollHorizontalIntervalTimer(10 * k, a, f, d[k], g, K, O, N, J, t, h, y, z, q, G, w, v, r, p, u);
            I = (h + x[v] + G)%G;
            "" != u.onSlideComplete && 1 < d.length && (c[c.length] = e.onSlideCompleteTimer(10 * (k + 1), u, a, b(a).children(":eq(" + I + ")"), p, v));
            $[v] = c;
            e.hideScrollbar(u, c, k, d, g, K, O, J, t, v)
        },
        onSlideComplete: function(a, f, c, g, h) {
            c = new e.args("complete", a, b(f), c, g, g);
            b(f).parent().data("args", c);
            if ("" != a.onSlideComplete)
                a.onSlideComplete(c)
        },
        getSliderOffset: function(a, e) {
            var c = 0;
            e = "x" == e ? 4 : 5;
            if (!ea || aa || X)
                c = parseInt(b(a).css("left"), 10);
            else {
                for (var c = ["-webkit-transform", "-moz-transform", "transform"], g, h = 0; h < c.length; h++)
                    if (void 0 != b(a).css(c[h]) && 0 < b(a).css(c[h]).length) {
                    g = b(a).css(c[h]).split(",");
                    break
                }
                c = void 0 == g[e] ? 0 : parseInt(g[e], 10)
            }
            return c
        },
        setSliderOffset: function(a, e) {
            !ea || aa || X ? b(a).css({
                left: e + "px"
            }) : b(a).css({
                webkitTransform: "matrix(1,0,0,1," + e + ",0)",
                MozTransform: "matrix(1,0,0,1," + e + ",0)",
                transform: "matrix(1,0,0,1," + e + ",0)"
            })
        },
        setBrowserInfo: function() {
            null != navigator.userAgent.match("WebKit") ? (ia = "-webkit-grab", sa = "-webkit-grabbing") : null != navigator.userAgent.match("Gecko") ? (ha=!0, ia = "move", sa = "-moz-grabbing") : null != navigator.userAgent.match("MSIE 7") ? oa = aa=!0 : null != navigator.userAgent.match("MSIE 8") ? oa = X=!0 : null != navigator.userAgent.match("MSIE 9") && (oa=!0)
        },
        has3DTransform: function() {
            var a = !1, e = b("<div />").css({
                webkitTransform: "matrix(1,1,1,1,1,1)",
                MozTransform: "matrix(1,1,1,1,1,1)",
                transform: "matrix(1,1,1,1,1,1)"
            });
            "" == e.attr("style") ? a=!1 : ha&&!za && 21 <= parseInt(navigator.userAgent.split("/")[3], 10) ? a=!1 : void 0 != e.attr("style") && (a=!0);
            return a
        },
        getSlideNumber: function(b, e, c) {
            return (b - x[e] + c)%c
        },
        calcActiveOffset: function(b, e, c, g, h, d, m, s) {
            h=!1;
            b = [];
            var x;
            e > c[0] && (x = 0);
            e < c[c.length - 1] && (x = d - 1);
            for (d = 0; d < c.length; d++)
                c[d] <= e && c[d] > e - g && (h || c[d] == e || (b[b.length] = c[d - 1]), b[b.length] = c[d], h=!0);
            0 == b.length && (b[0] = c[c.length - 1]);
            for (d = h = 0; d < b.length; d++)
                m = Math.abs(e - b[d]), m < g && (h = b[d], g = m);
            for (d = 0; d < c.length; d++)
                h == c[d] && (x = d);
            return x
        },
        changeSlide: function(a, f, c, g, h, d, m, s, N, J, t, y, z, w, v, q, G, p) {
            e.autoSlidePause(w);
            for (var n = 0; n < g.length; n++)
                clearTimeout(g[n]);
            var r = Math.ceil(p.autoSlideTransTimer / 10) + 1, u = e.getSliderOffset(f, "x"), k = y[a], k = k - u, C = a - (B[w] + x[w] + q)%q;
            if (p.infiniteSlider) {
                a = (a - x[w] + 2 * q)%q;
                n=!1;
                0 == a && 2 == q && (a = q, y[a] = y[a - 1] - b(c).eq(0).outerWidth(!0), n=!0);
                var k = y[a], k = k - u, A = [y[a] - b(f).width(), y[a] + b(f).width()];
                n && y.splice(y.length - 1, 1);
                for (n = 0; n < A.length; n++)
                    Math.abs(A[n] - u) < Math.abs(k) && (k = A[n] - u);
                0 > k&&-1 == C ? k += b(f).width() : 0 < k && 1 == C && (k -= b(f).width())
            }
            C = [];
            e.showScrollbar(p, h);
            for (n = 0; n <= r; n++)
                A = n, A/=r, A--, A = u + k * (Math.pow(A, 5) + 1), C[C.length] = A;
            r = (a + x[w] + q)%q;
            for (n = u = 0; n < C.length; n++) {
                if (0 == n || 1 < Math.abs(C[n] - u) || n >= C.length - 2)
                    u = C[n], g[n] = e.slowScrollHorizontalIntervalTimer(10 * (n + 1), f, c, C[n], h, d, m, s, N, J, a, t, y, v, q, z, w, G, r, p);
                0 == n && "" != p.onSlideStart && (k = (B[w] + x[w] + q)%q, p.onSlideStart(new e.args("start", p, f, b(f).children(":eq(" + k + ")"), k, a)))
            }
            u=!1;
            p.infiniteSlider ? r != ba[w] && (u=!0) : a != B[w] && (u=!0);
            u && "" != p.onSlideComplete && (g[g.length] = e.onSlideCompleteTimer(10 * (n + 1), p, f, b(f).children(":eq(" + r + ")"), r, w));
            $[w] = g;
            e.hideScrollbar(p, g, n, C, h, d, m, N, J, w);
            e.autoSlide(f, c, g, h, d, m, s, N, J, t, y, z, w, v, q, G, p)
        },
        autoSlide: function(b, f, c, g, h, d, m, s, N, J, t, y, z, w, v, q, G) {
            if (!T[z].autoSlide)
                return !1;
            e.autoSlidePause(z);
            ja[z] = setTimeout(function() {
                !G.infiniteSlider && B[z] > t.length - 1 && (B[z] -= v);
                e.changeSlide((B[z] + x[z] + t.length + 1)%t.length, b, f, c, g, h, d, m, s, N, J, t, y, z, w, v, q, G);
                e.autoSlide(b, f, c, g, h, d, m, s, N, J, t, y, z, w, v, q, G)
            }, G.autoSlideTimer + G.autoSlideTransTimer)
        },
        autoSlidePause: function(b) {
            clearTimeout(ja[b])
        },
        isUnselectable: function(a, e) {
            return "" != e.unselectableSelector && 1 == b(a).closest(e.unselectableSelector).size()?!0 : !1
        },
        slowScrollHorizontalIntervalTimer: function(b, f, c, g, h, d, m, s, x, B, t, y, z, w, v, q, G, p, n, r) {
            return setTimeout(function() {
                e.slowScrollHorizontalInterval(f, c, g, h, d, m, s, x, B, t, y, z, w, v, q, G, p, n, r)
            }, b)
        },
        onSlideCompleteTimer: function(b, f, c, g, h, d) {
            return setTimeout(function() {
                e.onSlideComplete(f, c, g, h, d)
            }, b)
        },
        hideScrollbarIntervalTimer: function(b, f, c, g, h, d, m, s, x, B) {
            return setTimeout(function() {
                e.hideScrollbarInterval(f, c, g, h, d, m, s, x, B)
            }, b)
        },
        args: function(a, f, c, g, h, d) {
            this.prevSlideNumber = void 0 == b(c).parent().data("args") ? void 0 : b(c).parent().data("args").prevSlideNumber;
            this.prevSlideObject = void 0 == b(c).parent().data("args") ? void 0 : b(c).parent().data("args").prevSlideObject;
            this.targetSlideNumber = d + 1;
            this.targetSlideObject = b(c).children(":eq(" + d + ")");
            this.slideChanged=!1;
            "load" == a ? this.targetSlideObject = this.targetSlideNumber = void 0 : "start" == a ? this.targetSlideObject = this.targetSlideNumber = void 0 : "change" == a ? (this.slideChanged=!0, this.prevSlideNumber = void 0 == b(c).parent().data("args") ? f.startAtSlide : b(c).parent().data("args").currentSlideNumber, this.prevSlideObject = b(c).children(":eq(" + this.prevSlideNumber + ")")) : "complete" == a && (this.slideChanged = b(c).parent().data("args").slideChanged);
            this.settings = f;
            this.data = b(c).parent().data("iosslider");
            this.sliderObject = c;
            this.sliderContainerObject = b(c).parent();
            this.currentSlideObject = g;
            this.currentSlideNumber = h + 1;
            this.currentSliderOffset =- 1 * e.getSliderOffset(c, "x")
        },
        preventDrag: function(b) {
            b.preventDefault()
        },
        preventClick: function(b) {
            b.stopImmediatePropagation();
            return !1
        },
        enableClick: function() {
            return !0
        }
    };
    e.setBrowserInfo();
    var V = {
        init: function(a, f) {
            ea = e.has3DTransform();
            var c = b.extend(!0, {
                elasticPullResistance: 0.6,
                frictionCoefficient: 0.92,
                elasticFrictionCoefficient: 0.6,
                snapFrictionCoefficient: 0.92,
                snapToChildren: !1,
                snapSlideCenter: !1,
                startAtSlide: 1,
                scrollbar: !1,
                scrollbarDrag: !1,
                scrollbarHide: !0,
                scrollbarLocation: "top",
                scrollbarContainer: "",
                scrollbarOpacity: 0.4,
                scrollbarHeight: "4px",
                scrollbarBorder: "0",
                scrollbarMargin: "5px",
                scrollbarBackground: "#000",
                scrollbarBorderRadius: "100px",
                scrollbarShadow: "0 0 0 #000",
                scrollbarElasticPullResistance: 0.9,
                desktopClickDrag: !1,
                keyboardControls: !1,
                tabToAdvance: !1,
                responsiveSlideContainer: !0,
                responsiveSlides: !0,
                navSlideSelector: "",
                navPrevSelector: "",
                navNextSelector: "",
                autoSlideToggleSelector: "",
                autoSlide: !1,
                autoSlideTimer: 5E3,
                autoSlideTransTimer: 750,
                autoSlideHoverPause: !0,
                infiniteSlider: !1,
                snapVelocityThreshold: 5,
                slideStartVelocityThreshold: 0,
                horizontalSlideLockThreshold: 5,
                verticalSlideLockThreshold: 3,
                stageCSS: {
                    position: "relative",
                    top: "0",
                    left: "0",
                    overflow: "hidden",
                    zIndex: 1
                },
                unselectableSelector: "",
                onSliderLoaded: "",
                onSliderUpdate: "",
                onSliderResize: "",
                onSlideStart: "",
                onSlideChange: "",
                onSlideComplete: ""
            }, a);
            void 0 == f && (f = this);
            return b(f).each(function(a) {
                function f() {
                    e.autoSlidePause(d);
                    va = b(F).find("a");
                    ja = b(F).find("[onclick]");
                    pa = b(F).find("*");
                    b(n).css("width", "");
                    b(n).css("height", "");
                    b(F).css("width", "");
                    D = b(F).children().not("script").get();
                    ga = [];
                    M = [];
                    c.responsiveSlides && b(D).css("width", "");
                    s[d] = 0;
                    l = [];
                    q = b(n).parent().width();
                    r = b(n).outerWidth(!0);
                    c.responsiveSlideContainer && (r = b(n).outerWidth(!0) > q ? q : b(n).outerWidth(!0));
                    b(n).css({
                        position: c.stageCSS.position,
                        top: c.stageCSS.top,
                        left: c.stageCSS.left,
                        overflow: c.stageCSS.overflow,
                        zIndex: c.stageCSS.zIndex,
                        webkitPerspective: 1E3,
                        webkitBackfaceVisibility: "hidden",
                        msTouchAction: "pan-y",
                        width: r
                    });
                    b(c.unselectableSelector).css({
                        cursor: "default"
                    });
                    for (var a = 0; a < D.length; a++) {
                        ga[a] = b(D[a]).width();
                        M[a] = b(D[a]).outerWidth(!0);
                        var h = M[a];
                        c.responsiveSlides && (M[a] > r ? (h = r +- 1 * (M[a] - ga[a]), ga[a] = h, M[a] = r) : h = ga[a], b(D[a]).css({
                            width: h
                        }));
                        b(D[a]).css({
                            webkitBackfaceVisibility: "hidden",
                            overflow: "hidden",
                            position: "absolute"
                        });
                        l[a] =- 1 * s[d];
                        s[d] = s[d] + h + (M[a] - ga[a])
                    }
                    c.snapSlideCenter && (p = 0.5 * (r - M[0]), c.responsiveSlides && M[0] > r && (p = 0));
                    ua[d] = 2 * s[d];
                    for (a = 0; a < D.length; a++)
                        e.setSliderOffset(b(D[a]), - 1 * l[a] + s[d] + p), l[a] -= s[d];
                    if (!c.infiniteSlider&&!c.snapSlideCenter) {
                        for (a = 0; a < l.length&&!(l[a]<=-1 * (2 * s[d] - r)); a++)
                            ha = a;
                        l.splice(ha + 1, l.length);
                        l[l.length] =- 1 * (2 * s[d] - r)
                    }
                    for (a = 0; a < l.length; a++)
                        E[a] = l[a];
                    I && (c.startAtSlide = T[d].startAtSlide > l.length ? l.length : T[d].startAtSlide, c.infiniteSlider ? (c.startAtSlide = (T[d].startAtSlide - 1 + H)%H, B[d] = T[d].startAtSlide) : (c.startAtSlide = 0 > T[d].startAtSlide - 1 ? l.length - 1 : T[d].startAtSlide, B[d] = T[d].startAtSlide - 1), ba[d] = B[d]);
                    m[d] = s[d] + p;
                    b(F).css({
                        position: "relative",
                        cursor: ia,
                        webkitPerspective: "0",
                        webkitBackfaceVisibility: "hidden",
                        width: s[d] + "px"
                    });
                    R = s[d];
                    s[d] = 2 * s[d] - r + 2 * p;
                    (Y = R < r || 0 == r?!0 : !1) && b(F).css({
                        cursor: "default"
                    });
                    G = b(n).parent().outerHeight(!0);
                    u = b(n).height();
                    c.responsiveSlideContainer && (u = u > G ? G : u);
                    b(n).css({
                        height: u
                    });
                    e.setSliderOffset(F, l[B[d]]);
                    if (c.infiniteSlider&&!Y) {
                        a = e.getSliderOffset(b(F), "x");
                        for (h =- 1 * ((x[d] + H)%H); 0 > h;) {
                            var g = 0, A = e.getSliderOffset(b(D[0]), "x");
                            b(D).each(function(b) {
                                e.getSliderOffset(this, "x") < A && (A = e.getSliderOffset(this, "x"), g = b)
                            });
                            var L = m[d] + R;
                            e.setSliderOffset(b(D)[g], L);
                            m[d] =- 1 * l[1] + p;
                            s[d] = m[d] + R - r;
                            l.splice(0, 1);
                            l.splice(l.length, 0, - 1 * L + p);
                            h++
                        }
                        for (; 0<-1 * l[0] - R + p && c.snapSlideCenter && I;) {
                            var N = 0, O = e.getSliderOffset(b(D[0]), "x");
                            b(D).each(function(b) {
                                e.getSliderOffset(this, "x") > O && (O = e.getSliderOffset(this, "x"), N = b)
                            });
                            L = m[d] - M[N];
                            e.setSliderOffset(b(D)[N], L);
                            l.splice(0, 0, - 1 * L + p);
                            l.splice(l.length - 1, 1);
                            m[d] = - 1 * l[0] + p;
                            s[d] = m[d] + R - r;
                            x[d]--;
                            B[d]++
                        }
                        for (; a<=-1 * s[d];)
                            g = 0, A = e.getSliderOffset(b(D[0]), "x"), b(D).each(function(b) {
                                e.getSliderOffset(this, "x") < A && (A = e.getSliderOffset(this, "x"), g = b)
                            }), L = m[d] + R, e.setSliderOffset(b(D)[g], L), m[d] =- 1 * l[1] + p, s[d] = m[d] + R - r, l.splice(0, 1), l.splice(l.length, 0, - 1 * L + p), x[d]++, B[d]--
                    }
                    e.setSliderOffset(F, l[B[d]]);
                    c.desktopClickDrag || b(F).css({
                        cursor: "default"
                    });
                    c.scrollbar && (b("." + J).css({
                        margin: c.scrollbarMargin,
                        overflow: "hidden",
                        display: "none"
                    }), b("." + J + " ." + t).css({
                        border: c.scrollbarBorder
                    }), k = parseInt(b("." + J).css("marginLeft")) + parseInt(b("." + J).css("marginRight")), C = parseInt(b("." + J + " ." + t).css("borderLeftWidth"), 10) + parseInt(b("." + J + " ." + t).css("borderRightWidth"), 10), w = "" != c.scrollbarContainer ? b(c.scrollbarContainer).width() : r, v = r / R * (w - k), c.scrollbarHide || (V = c.scrollbarOpacity), b("." + J).css({
                        position: "absolute",
                        left: 0,
                        width: w - k + "px",
                        margin: c.scrollbarMargin
                    }), "top" == c.scrollbarLocation ? b("." + J).css("top", "0") : b("." + J).css("bottom", "0"), b("." + J + " ." + t).css({
                        borderRadius: c.scrollbarBorderRadius,
                        background: c.scrollbarBackground,
                        height: c.scrollbarHeight,
                        width: v - C + "px",
                        minWidth: c.scrollbarHeight,
                        border: c.scrollbarBorder,
                        webkitPerspective: 1E3,
                        webkitBackfaceVisibility: "hidden",
                        position: "relative",
                        opacity: V,
                        filter: "alpha(opacity:" + 100 * V + ")",
                        boxShadow: c.scrollbarShadow
                    }), e.setSliderOffset(b("." + J + " ." + t), Math.floor(( - 1 * l[B[d]] - m[d] + p) / (s[d] - m[d] + p) * (w - k - v))), b("." + J).css({
                        display: "block"
                    }), y = b("." + J + " ." + t), z = b("." + J));
                    c.scrollbarDrag&&!Y && b("." + J + " ." + t).css({
                        cursor: ia
                    });
                    c.infiniteSlider && (S = (s[d] + r) / 3);
                    "" != c.navSlideSelector && b(c.navSlideSelector).each(function(a) {
                        b(this).css({
                            cursor: "pointer"
                        });
                        b(this).unbind(P).bind(P, function(f) {
                            "touchstart" == f.type ? b(this).unbind("click.iosSliderEvent") : b(this).unbind("touchstart.iosSliderEvent");
                            P = f.type + ".iosSliderEvent";
                            e.changeSlide(a, F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c)
                        })
                    });
                    "" != c.navPrevSelector && (b(c.navPrevSelector).css({
                        cursor: "pointer"
                    }), b(c.navPrevSelector).unbind(P).bind(P, function(a) {
                        "touchstart" == a.type ? b(this).unbind("click.iosSliderEvent") : b(this).unbind("touchstart.iosSliderEvent");
                        P = a.type + ".iosSliderEvent";
                        a = (B[d] + x[d] + H)%H;
                        (0 < a || c.infiniteSlider) && e.changeSlide(a - 1, F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c)
                    }));
                    "" != c.navNextSelector && (b(c.navNextSelector).css({
                        cursor: "pointer"
                    }), b(c.navNextSelector).unbind(P).bind(P, function(a) {
                        "touchstart" == a.type ? b(this).unbind("click.iosSliderEvent") : b(this).unbind("touchstart.iosSliderEvent");
                        P = a.type + ".iosSliderEvent";
                        a = (B[d] + x[d] + H)%H;
                        (a < l.length - 1 || c.infiniteSlider) && e.changeSlide(a + 1, F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c)
                    }));
                    c.autoSlide&&!Y && "" != c.autoSlideToggleSelector && (b(c.autoSlideToggleSelector).css({
                        cursor: "pointer"
                    }), b(c.autoSlideToggleSelector).unbind(P).bind(P, function(a) {
                        "touchstart" == a.type ? b(this).unbind("click.iosSliderEvent") : b(this).unbind("touchstart.iosSliderEvent");
                        P = a.type + ".iosSliderEvent";
                        wa ? (e.autoSlide(F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c), wa=!1, b(c.autoSlideToggleSelector).removeClass("on")) : (e.autoSlidePause(d), wa=!0, b(c.autoSlideToggleSelector).addClass("on"))
                    }));
                    e.autoSlide(F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c);
                    b(n).bind("mouseleave.iosSliderEvent", function() {
                        e.autoSlide(F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c)
                    });
                    b(n).bind("touchend.iosSliderEvent", function() {
                        e.autoSlide(F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c)
                    });
                    c.autoSlidePauseHover && b(n).bind("mouseenter.iosSliderEvent", function() {
                        e.autoSlidePause(d)
                    });
                    b(n).data("iosslider", {
                        obj: Aa,
                        settings: c,
                        scrollerNode: F,
                        slideNodes: D,
                        numberOfSlides: H,
                        centeredSlideOffset: p,
                        sliderNumber: d,
                        originalOffsets: E,
                        childrenOffsets: l,
                        sliderMax: s[d],
                        scrollbarClass: t,
                        scrollbarWidth: v,
                        scrollbarStageWidth: w,
                        stageWidth: r,
                        scrollMargin: k,
                        scrollBorder: C,
                        infiniteSliderOffset: x[d],
                        infiniteSliderWidth: S,
                        slideNodeOuterWidths: M
                    });
                    I=!1;
                    return !0
                }
                na++;
                var d = na, K = [];
                T[d] = c;
                m[d] = 0;
                s[d] = 0;
                var O = [0, 0], N = [0, 0], J = "scrollbarBlock" + na, t = "scrollbar" + na, y, z, w, v, q, G, p = 0, n = b(this), r, u, k, C, A, I=!0;
                a =- 1;
                var l, E = [], V = 0, Q = 0, ea = 0, F = b(this).children(":first-child"), D, ga, M, H = b(F).children().not("script").size(), U=!1, ha = 0, xa=!1, qa = void 0, S;
                x[d] = 0;
                var Y=!1, wa=!1;
                ta[d]=!1;
                var Z, ra=!1, ka=!1, P = "touchstart.iosSliderEvent click.iosSliderEvent", R, va, ja, pa;
                fa[d]=!1;
                $[d] = [];
                c.scrollbarDrag && (c.scrollbar=!0, c.scrollbarHide=!1);
                var Aa = b(this);
                if (void 0 != Aa.data("iosslider")
                    )return !0;
                b(this).find("img").bind("dragstart.iosSliderEvent", function(b) {
                    b.preventDefault()
                });
                c.infiniteSlider && (c.scrollbar=!1);
                c.infiniteSlider && 1 == H && (c.infiniteSlider=!1);
                c.scrollbar && ("" != c.scrollbarContainer ? b(c.scrollbarContainer).append("<div class = '" + J + "'><div class = '" + t + "'></div></div>") : b(F).parent().append("<div class = '" + J + "'><div class = '" + t + "'></div></div>"));
                if (!f())
                    return !0;
                b(this).find("a").bind("mousedown", e.preventDrag);
                b(this).find("[onclick]").bind("click", e.preventDrag).each(function() {
                    b(this).data("onclick", this.onclick)
                });
                a = e.calcActiveOffset(c, e.getSliderOffset(b(F), "x"), l, r, x[d], H, void 0, d);
                a = (a + x[d] + H)%H;
                a = new e.args("load", c, F, b(F).children(":eq(" + a + ")"), a, a);
                b(n).data("args", a);
                if ("" != c.onSliderLoaded)
                    c.onSliderLoaded(a);
                if (T[d].responsiveSlides || T[d].responsiveSlideContainer)
                    a = Ba ? "orientationchange" : "resize", b(window).bind(a + ".iosSliderEvent-" + d, function() {
                        if (!f())
                            return !0;
                            var a = b(n).data("args");
                            if ("" != c.onSliderResize)
                                c.onSliderResize(a)
                            });
                !c.keyboardControls&&!c.tabToAdvance || Y || b(document).bind("keydown.iosSliderEvent", function(b) {
                    aa || X || (b = b.originalEvent);
                    if (37 == b.keyCode && c.keyboardControls)
                        b.preventDefault(), b = (B[d] + x[d] + H)%H, (0 < b || c.infiniteSlider) && e.changeSlide(b - 1, F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c);
                    else if (39 == b.keyCode && c.keyboardControls || 9 == b.keyCode && c.tabToAdvance)
                        b.preventDefault(), b = (B[d] + x[d] + H)%H, (b < l.length - 1 || c.infiniteSlider) && e.changeSlide(b + 1, F, D, K, t, v, r, w, k, C, E, l, M, d, S, H, p, c)
                });
                if (za || c.desktopClickDrag) {
                    var ca=!1, la = b(F), ma = b(F), ya=!1;
                    c.scrollbarDrag && (la = la.add(y), ma = ma.add(z));
                    b(la).bind("mousedown.iosSliderEvent touchstart.iosSliderEvent", function(a) {
                        if (ca)
                            return !0;
                        ca=!0;
                        "touchstart" == a.type ? b(ma).unbind("mousedown.iosSliderEvent") : b(ma).unbind("touchstart.iosSliderEvent");
                        if (fa[d] || Y || (ya = e.isUnselectable(a.target, c)))
                            return U = ca=!1, !0;
                        Z = b(this)[0] === b(y)[0] ? y : F;
                        aa || X || (a = a.originalEvent);
                        e.autoSlidePause(d);
                        pa.unbind(".disableClick");
                        if ("touchstart" == a.type)
                            eventX = a.touches[0].pageX, eventY = a.touches[0].pageY;
                        else {
                            if (window.getSelection)
                                window.getSelection().empty ? window.getSelection().empty() : window.getSelection().removeAllRanges && window.getSelection().removeAllRanges();
                            else if (document.selection)
                                if (X)
                                    try {
                                        document.selection.empty()
                            } catch (f) {} else 
                                document.selection.empty();
                            eventX = a.pageX;
                            eventY = a.pageY;
                            xa=!0;
                            qa = F;
                            b(this).css({
                                cursor: sa
                            })
                        }
                        O = [0, 0];
                        N = [0, 0];
                        W = 0;
                        U=!1;
                        for (a = 0; a < K.length; a++)
                            clearTimeout(K[a]);
                        a = e.getSliderOffset(F, "x");
                        a>-1 * m[d] + p + R ? (a =- 1 * m[d] + p + R, e.setSliderOffset(b("." + t), a), b("." + t).css({
                            width: v - C + "px"
                        })) : a<-1 * s[d] && (e.setSliderOffset(b("." + t), w - k - v), b("." + t).css({
                            width: v - C + "px"
                        }));
                        a = b(this)[0] === b(y)[0] ? m[d] : 0;
                        Q =- 1 * (e.getSliderOffset(this, "x") - eventX - a);
                        e.getSliderOffset(this, "y");
                        O[1] = eventX;
                        N[1] = eventY;
                        ka=!1
                    });
                    b(ma).bind("touchmove.iosSliderEvent mousemove.iosSliderEvent", function(a) {
                        aa || X || (a = a.originalEvent);
                        if (fa[d] || Y || ya)
                            return !0;
                        var f = 0;
                        if ("touchmove" == a.type)
                            eventX = a.touches[0].pageX, eventY = a.touches[0].pageY;
                        else {
                            if (window.getSelection)
                                window.getSelection().empty || window.getSelection().removeAllRanges && window.getSelection().removeAllRanges();
                            else if (document.selection)
                                if (X)
                                    try {
                                        document.selection.empty()
                            } catch (h) {} else 
                                document.selection.empty();
                            eventX = a.pageX;
                            eventY = a.pageY;
                            if (!xa ||!oa && ("undefined" != typeof a.webkitMovementX || "undefined" != typeof a.webkitMovementY) && 0 === a.webkitMovementY && 0 === a.webkitMovementX)
                                return !0
                        }
                        O[0] = O[1];
                        O[1] = eventX;
                        W = (O[1] - O[0]) / 2;
                        N[0] = N[1];
                        N[1] = eventY;
                        da = (N[1] - N[0]) / 2;
                        if (!U) {
                            var g = (B[d] + x[d] + H)%H, g = new e.args("start", c, F, b(F).children(":eq(" + g + ")"), g, void 0);
                            b(n).data("args", g);
                            if ("" != c.onSlideStart)
                                c.onSlideStart(g)
                        }(da > c.verticalSlideLockThreshold || da<-1 * c.verticalSlideLockThreshold) && ("touchmove" == a.type&&!U) && (ra=!0);
                        (W > c.horizontalSlideLockThreshold || W<-1 * c.horizontalSlideLockThreshold) && "touchmove" == a.type && a.preventDefault();
                        if (W > c.slideStartVelocityThreshold || W<-1 * c.slideStartVelocityThreshold)
                            U=!0;
                        if (U&&!ra) {
                            var g = e.getSliderOffset(F, "x"), q = b(Z)[0] === b(y)[0] ? m[d]: p, u = b(Z)[0] === b(y)[0] ? (m[d] - s[d] - p) / (w - k - v): 1, z = b(Z)[0] === b(y)[0] ? c.scrollbarElasticPullResistance: c.elasticPullResistance, G = c.snapSlideCenter && b(Z)[0] === b(y)[0] ? 0: p, J = c.snapSlideCenter && b(Z)[0] === b(y)[0] ? p: 0;
                            "touchmove" == a.type && (ea != a.touches.length && (Q =- 1 * g + eventX), ea = a.touches.length);
                            if (c.infiniteSlider) {
                                if (g<=-1 * s[d]) {
                                    var I = b(F).width();
                                    if (g<=-1 * ua[d]) {
                                        var K =- 1 * E[0];
                                        b(D).each(function(a) {
                                            e.setSliderOffset(b(D)[a], K + p);
                                            a < l.length && (l[a] =- 1 * K);
                                            K += M[a]
                                        });
                                        Q-=-1 * l[0];
                                        m[d] =- 1 * l[0] + p;
                                        s[d] = m[d] + I - r;
                                        x[d] = 0
                                    } else {
                                        var S = 0, T = e.getSliderOffset(b(D[0]), "x");
                                        b(D).each(function(b) {
                                            e.getSliderOffset(this, "x") < T && (T = e.getSliderOffset(this, "x"), S = b)
                                        });
                                        z = m[d] + I;
                                        e.setSliderOffset(b(D)[S], z);
                                        m[d] =- 1 * l[1] + p;
                                        s[d] = m[d] + I - r;
                                        l.splice(0, 1);
                                        l.splice(l.length, 0, - 1 * z + p);
                                        x[d]++
                                    }
                                }
                                if (g>=-1 * m[d] || 0 <= g)
                                    if (I = b(F).width(), 0 <= g)
                                        for (K =- 1 * E[0], b(D).each(function(a) {
                                            e.setSliderOffset(b(D)[a], K + p);
                                            a < l.length && (l[a] =- 1 * K);
                                            K += M[a]
                                        }), Q +=- 1 * l[0], m[d] =- 1 * l[0] + p, s[d] = m[d] + I - r, x[d] = H; 0<-1 * l[0] - I + p;) {
                                            var P = 0, R = e.getSliderOffset(b(D[0]), "x");
                                            b(D).each(function(b) {
                                                e.getSliderOffset(this, "x") > R && (R = e.getSliderOffset(this, "x"), P = b)
                                            });
                                            z = m[d] - M[P];
                                            e.setSliderOffset(b(D)[P], z);
                                            l.splice(0, 0, - 1 * z + p);
                                            l.splice(l.length - 1, 1);
                                            m[d] =- 1 * l[0] + p;
                                            s[d] = m[d] + I - r;
                                            x[d]--;
                                            B[d]++
                                        } else 
                                            P = 0, R = e.getSliderOffset(b(D[0]), "x"), b(D).each(function(b) {
                                                e.getSliderOffset(this, "x") > R && (R = e.getSliderOffset(this, "x"), P = b)
                                            }), z = m[d] - M[P], e.setSliderOffset(b(D)[P], z), l.splice(0, 0, - 1 * z + p), l.splice(l.length - 1, 1), m[d] =- 1 * l[0] + p, s[d] = m[d] + I - r, x[d]--
                            } else 
                                I = b(F).width(), g>-1 * m[d] + p && (f =- 1 * (m[d] +- 1 * (Q - q - eventX + G) * u - q) * z / u), g<-1 * s[d] && (f =- 1 * (s[d] + J +- 1 * (Q - q - eventX) * u - q) * z / u);
                            e.setSliderOffset(F, - 1 * (Q - q - eventX - f) * u - q + J);
                            c.scrollbar && (e.showScrollbar(c, t), L = Math.floor((Q - eventX - f - m[d] + G) / (s[d] - m[d] + p) * (w - k - v) * u), g = v, 0 >= L ? (g = v - C - - 1 * L, e.setSliderOffset(b("." + t), 0), b("." + t).css({
                                width: g + "px"
                            })) : L >= w - k - C - v ? (g = w - k - C - L, e.setSliderOffset(b("." + t), L), b("." + t).css({
                                width: g + "px"
                            })) : e.setSliderOffset(b("." + t), L));
                            "touchmove" == a.type && (A = a.touches[0].pageX);
                            a=!1;
                            f = e.calcActiveOffset(c, - 1 * (Q - eventX - f), l, r, x[d], H, void 0, d);
                            g = (f + x[d] + H)%H;
                            c.infiniteSlider ? g != ba[d] && (a=!0) : f != B[d] && (a=!0);
                            if (a && (B[d] = f, ba[d] = g, ka=!0, g = new e.args("change", c, F, b(F).children(":eq(" + g + ")"), g, g), b(n).data("args", g), "" != c.onSlideChange))
                                c.onSlideChange(g)
                        }
                        ca=!1
                    });
                    a = b(window);
                    if (X || aa)
                        a = b(document);
                    b(la).bind("touchend.iosSliderEvent", function(b) {
                        b = b.originalEvent;
                        if (fa[d] || Y || ya)
                            return !0;
                        if (0 != b.touches.length)
                            for (var a = 0; a < b.touches.length; a++)
                                b.touches[a].pageX == A && e.slowScrollHorizontal(F, D, K, t, W, da, v, r, w, k, C, E, l, M, d, S, H, Z, ka, p, c);
                        else 
                            e.slowScrollHorizontal(F, D, K, t, W, da, v, r, w, k, C, E, l, M, d, S, H, Z, ka, p, c);
                        ca = ra=!1
                    });
                    b(a).bind("mouseup.iosSliderEvent-" + d, function(a) {
                        U ? va.unbind("click.disableClick").bind("click.disableClick", e.preventClick) : va.unbind("click.disableClick").bind("click.disableClick", e.enableClick);
                        ja.each(function() {
                            this.onclick = function(a) {
                                if (U)
                                    return !1;
                                b(this).data("onclick").call(this, a || window.event)
                            }
                        });
                        1.8 <= parseFloat(b().jquery) ? pa.each(function() {
                            var a = b._data(this, "events");
                            if (void 0 != a && void 0 != a.click && "iosSliderEvent" != a.click[0].namespace) {
                                if (!U)
                                    return !1;
                                b(this).one("click.disableClick", e.preventClick);
                                var a = b._data(this, "events").click, c = a.pop();
                                a.splice(0, 0, c)
                            }
                        }) : 1.6 <= parseFloat(b().jquery) && pa.each(function() {
                            var a = b(this).data("events");
                            if (void 0 != a && void 0 != a.click && "iosSliderEvent" != a.click[0].namespace) {
                                if (!U)
                                    return !1;
                                b(this).one("click.disableClick", e.preventClick);
                                var a = b(this).data("events").click, c = a.pop();
                                a.splice(0, 0, c)
                            }
                        });
                        if (!ta[d]) {
                            if (Y)
                                return !0;
                            b(la).css({
                                cursor: ia
                            });
                            xa=!1;
                            if (void 0 == qa)
                                return !0;
                            e.slowScrollHorizontal(qa, D, K, t, W, da, v, r, w, k, C, E, l, M, d, S, H, Z, ka, p, c);
                            qa = void 0
                        }
                        ca = ra=!1
                    })
                }
            })
        },
        destroy: function(a, f) {
            void 0 == f && (f = this);
            return b(f).each(function() {
                var c = b(this), f = c.data("iosslider");
                if (void 0 == f)
                    return !1;
                void 0 == a && (a=!0);
                e.autoSlidePause(f.sliderNumber);
                ta[f.sliderNumber]=!0;
                b(window).unbind(".iosSliderEvent-" + f.sliderNumber);
                b(document).unbind(".iosSliderEvent-" + f.sliderNumber);
                b(document).unbind("keydown.iosSliderEvent");
                b(this).unbind(".iosSliderEvent");
                b(this).children(":first-child").unbind(".iosSliderEvent");
                b(this).children(":first-child").children().unbind(".iosSliderEvent");
                a && (b(this).attr("style", ""), b(this).children(":first-child").attr("style", ""), b(this).children(":first-child").children().attr("style", ""), b(f.settings.navSlideSelector).attr("style", ""), b(f.settings.navPrevSelector).attr("style", ""), b(f.settings.navNextSelector).attr("style", ""), b(f.settings.autoSlideToggleSelector).attr("style", ""), b(f.settings.unselectableSelector).attr("style", ""));
                f.settings.scrollbar && b(".scrollbarBlock" + f.sliderNumber).remove();
                for (var f = $[f.sliderNumber], h = 0; h < f.length; h++)
                    clearTimeout(f[h]);
                c.removeData("iosslider");
                c.removeData("args")
            })
        },
        update: function(a) {
            void 0 == a && (a = this);
            return b(a).each(function() {
                var a = b(this), c = a.data("iosslider");
                if (void 0 == c)
                    return !1;
                c.settings.startAtSlide = a.data("args").currentSlideNumber;
                V.destroy(!1, this);
                1 != c.numberOfSlides && c.settings.infiniteSlider && (c.settings.startAtSlide = (B[c.sliderNumber] + 1 + x[c.sliderNumber] + c.numberOfSlides)%c.numberOfSlides);
                V.init(c.settings, this);
                a = new e.args("update", c.settings, c.scrollerNode, b(c.scrollerNode).children(":eq(" + (c.settings.startAtSlide - 1) + ")"), c.settings.startAtSlide - 1, c.settings.startAtSlide - 1);
                b(c.stageNode).data("args", a);
                if ("" != c.settings.onSliderUpdate)
                    c.settings.onSliderUpdate(a)
            })
        },
        addSlide: function(a, e) {
            return this.each(function() {
                var c = b(this), g = c.data("iosslider");
                if (void 0 == g)
                    return !1;
                0 == b(g.scrollerNode).children().size() ? (b(g.scrollerNode).append(a), c.data("args").currentSlideNumber = 1) : g.settings.infiniteSlider ? (1 == e ? b(g.scrollerNode).children(":eq(0)").before(a) : b(g.scrollerNode).children(":eq(" + (e - 2) + ")").after(a), - 1 > x[g.sliderNumber] && B[g.sliderNumber]--, c.data("args").currentSlideNumber >= e && B[g.sliderNumber]++) : (e <= g.numberOfSlides ? b(g.scrollerNode).children(":eq(" + (e - 1) + ")").before(a) : b(g.scrollerNode).children(":eq(" + (e - 2) + ")").after(a), c.data("args").currentSlideNumber >= e && c.data("args").currentSlideNumber++);
                c.data("iosslider").numberOfSlides++;
                V.update(this)
            })
        },
        removeSlide: function(a) {
            return this.each(function() {
                var e = b(this).data("iosslider");
                if (void 0 == e)
                    return !1;
                b(e.scrollerNode).children(":eq(" + (a - 1) + ")").remove();
                B[e.sliderNumber] > a - 1 && B[e.sliderNumber]--;
                V.update(this)
            })
        },
        goToSlide: function(a, f) {
            void 0 == f && (f = this);
            return b(f).each(function() {
                var c = b(this).data("iosslider");
                if (void 0 == c)
                    return !1;
                a = a > c.childrenOffsets.length ? c.childrenOffsets.length - 1 : a - 1;
                e.changeSlide(a, b(c.scrollerNode), b(c.slideNodes), $[c.sliderNumber], c.scrollbarClass, c.scrollbarWidth, c.stageWidth, c.scrollbarStageWidth, c.scrollMargin, c.scrollBorder, c.originalOffsets, c.childrenOffsets, c.slideNodeOuterWidths, c.sliderNumber, c.infiniteSliderWidth, c.numberOfSlides, c.centeredSlideOffset, c.settings);
                B[c.sliderNumber] = a
            })
        },
        prevSlide: function() {
            return this.each(function() {
                var a = b(this).data("iosslider");
                if (void 0 == a)
                    return !1;
                var f = (B[a.sliderNumber] + x[a.sliderNumber] + a.numberOfSlides)%a.numberOfSlides;
                (0 < f || a.settings.infiniteSlider) && e.changeSlide(f - 1, b(a.scrollerNode), b(a.slideNodes), $[a.sliderNumber], a.scrollbarClass, a.scrollbarWidth, a.stageWidth, a.scrollbarStageWidth, a.scrollMargin, a.scrollBorder, a.originalOffsets, a.childrenOffsets, a.slideNodeOuterWidths, a.sliderNumber, a.infiniteSliderWidth, a.numberOfSlides, a.centeredSlideOffset, a.settings);
                B[a.sliderNumber] = f
            })
        },
        nextSlide: function() {
            return this.each(function() {
                var a = b(this).data("iosslider");
                if (void 0 == a)
                    return !1;
                var f = (B[a.sliderNumber] + x[a.sliderNumber] + a.numberOfSlides)%a.numberOfSlides;
                (f < a.childrenOffsets.length - 1 || a.settings.infiniteSlider) && e.changeSlide(f + 1, b(a.scrollerNode), b(a.slideNodes), $[a.sliderNumber], a.scrollbarClass, a.scrollbarWidth, a.stageWidth, a.scrollbarStageWidth, a.scrollMargin, a.scrollBorder, a.originalOffsets, a.childrenOffsets, a.slideNodeOuterWidths, a.sliderNumber, a.infiniteSliderWidth, a.numberOfSlides, a.centeredSlideOffset, a.settings);
                B[a.sliderNumber] = f
            })
        },
        lock: function() {
            return this.each(function() {
                var a = b(this).data("iosslider");
                if (void 0 == a)
                    return !1;
                fa[a.sliderNumber]=!0
            })
        },
        unlock: function() {
            return this.each(function() {
                var a = b(this).data("iosslider");
                if (void 0 == a)
                    return !1;
                fa[a.sliderNumber]=!1
            })
        },
        getData: function() {
            return this.each(function() {
                var a = b(this).data("iosslider");
                return void 0 == a?!1 : a
            })
        },
        autoSlidePause: function() {
            return this.each(function() {
                var a = b(this).data("iosslider");
                if (void 0 == a)
                    return !1;
                T[a.sliderNumber].autoSlide=!1;
                e.autoSlidePause(a.sliderNumber);
                return a
            })
        },
        autoSlidePlay: function() {
            return this.each(function() {
                var a = b(this).data("iosslider");
                if (void 0 == a)
                    return !1;
                T[a.sliderNumber].autoSlide=!0;
                e.autoSlide(b(a.scrollerNode), b(a.slideNodes), $[a.sliderNumber], a.scrollbarClass, a.scrollbarWidth, a.stageWidth, a.scrollbarStageWidth, a.scrollMargin, a.scrollBorder, a.originalOffsets, a.childrenOffsets, a.slideNodeOuterWidths, a.sliderNumber, a.infiniteSliderWidth, a.numberOfSlides, a.centeredSlideOffset, a.settings);
                return a
            })
        }
    };
    b.fn.iosSlider = function(a) {
        if (V[a])
            return V[a].apply(this, Array.prototype.slice.call(arguments, 1));
        if ("object" !== typeof a && a)
            b.error("invalid method call!");
        else 
            return V.init.apply(this, arguments)
    }
})(jQuery);


/*!
 jQuery Colorbox v1.4.17 - 2013-05-23
 (c) 2013 Jack Moore - jacklmoore.com/colorbox
 license: http://www.opensource.org/licenses/mit-license.php
 */
(function(t, e, i) {
    function o(i, o, n) {
        var r = e.createElement(i);
        return o && (r.id = te + o), n && (r.style.cssText = n), t(r)
    }
    function n() {
        return i.innerHeight ? i.innerHeight : t(i).height()
    }
    function r(t) {
        var e = E.length, i = (j + t)%e;
        return 0 > i ? e + i : i
    }
    function h(t, e) {
        return Math.round((/%/.test(t) ? ("x" === e ? H.width() : n()) / 100 : 1) * parseInt(t, 10))
    }
    function l(t, e) {
        return t.photo || t.photoRegex.test(e)
    }
    function s(t, e) {
        return t.retinaUrl && i.devicePixelRatio > 1 ? e.replace(t.photoRegex, t.retinaSuffix) : e
    }
    function a(t) {
        "contains"in x[0]&&!x[0].contains(t.target) && (t.stopPropagation(), x.focus())
    }
    function d() {
        var e, i = t.data(A, Z);
        null == i ? (_ = t.extend({}, Y), console && console.log && console.log("Error: cboxElement missing settings object")) : _ = t.extend({}, i);
        for (e in _)
            t.isFunction(_[e]) && "on" !== e.slice(0, 2) && (_[e] = _[e].call(A));
        _.rel = _.rel || A.rel || t(A).data("rel") || "nofollow", _.href = _.href || t(A).attr("href"), _.title = _.title || A.title, "string" == typeof _.href && (_.href = t.trim(_.href))
    }
    function c(i, o) {
        t(e).trigger(i), se.trigger(i), t.isFunction(o) && o.call(A)
    }
    function u() {
        var t, e, i, o, n, r = te + "Slideshow_", h = "click." + te;
        _.slideshow && E[1] ? (e = function() {
            clearTimeout(t)
        }, i = function() {
            (_.loop || E[j + 1]) && (t = setTimeout(J.next, _.slideshowSpeed))
        }, o = function() {
            M.html(_.slideshowStop).unbind(h).one(h, n), se.bind(ne, i).bind(oe, e).bind(re, n), x.removeClass(r + "off").addClass(r + "on")
        }, n = function() {
            e(), se.unbind(ne, i).unbind(oe, e).unbind(re, n), M.html(_.slideshowStart).unbind(h).one(h, function() {
                J.next(), o()
            }), x.removeClass(r + "on").addClass(r + "off")
        }, _.slideshowAuto ? o() : n()) : x.removeClass(r + "off " + r + "on")
    }
    function f(i) {
        G || (A = i, d(), E = t(A), j = 0, "nofollow" !== _.rel && (E = t("." + ee).filter(function() {
            var e, i = t.data(this, Z);
            return i && (e = t(this).data("rel") || i.rel || this.rel), e === _.rel
        }), j = E.index(A), - 1 === j && (E = E.add(A), j = E.length - 1)), g.css({
            opacity: parseFloat(_.opacity),
            cursor: _.overlayClose ? "pointer": "auto",
            visibility: "visible"
        }).show(), V && x.add(g).removeClass(V), _.className && x.add(g).addClass(_.className), V = _.className, K.html(_.close).show(), $ || ($ = q=!0, x.css({
            visibility: "hidden",
            display: "block"
        }), W = o(ae, "LoadedContent", "width:0; height:0; overflow:hidden").appendTo(v), D = b.height() + k.height() + v.outerHeight(!0) - v.height(), B = C.width() + T.width() + v.outerWidth(!0) - v.width(), N = W.outerHeight(!0), z = W.outerWidth(!0), _.w = h(_.initialWidth, "x"), _.h = h(_.initialHeight, "y"), J.position(), u(), c(ie, _.onOpen), O.add(S).hide(), x.focus(), _.trapFocus && e.addEventListener && (e.addEventListener("focus", a, !0), se.one(he, function() {
            e.removeEventListener("focus", a, !0)
        })), _.returnFocus && se.one(he, function() {
            t(A).focus()
        })), w())
    }
    function p() {
        !x && e.body && (X=!1, H = t(i), x = o(ae).attr({
            id: Z,
            "class": t.support.opacity===!1 ? te + "IE": "",
            role: "dialog",
            tabindex: "-1"
        }).hide(), g = o(ae, "Overlay").hide(), L = o(ae, "LoadingOverlay").add(o(ae, "LoadingGraphic")), y = o(ae, "Wrapper"), v = o(ae, "Content").append(S = o(ae, "Title"), I = o(ae, "Current"), P = t('<button type="button"/>').attr({
            id: te + "Previous"
        }), R = t('<button type="button"/>').attr({
            id: te + "Next"
        }), M = o("button", "Slideshow"), L, K = t('<button type="button"/>').attr({
            id: te + "Close"
        })), y.append(o(ae).append(o(ae, "TopLeft"), b = o(ae, "TopCenter"), o(ae, "TopRight")), o(ae, !1, "clear:left").append(C = o(ae, "MiddleLeft"), v, T = o(ae, "MiddleRight")), o(ae, !1, "clear:left").append(o(ae, "BottomLeft"), k = o(ae, "BottomCenter"), o(ae, "BottomRight"))).find("div div").css({
            "float": "left"
        }), F = o(ae, !1, "position:absolute; width:9999px; visibility:hidden; display:none"), O = R.add(P).add(I).add(M), t(e.body).append(g, x.append(y, F)))
    }
    function m() {
        function i(t) {
            t.which > 1 || t.shiftKey || t.altKey || t.metaKey || t.control || (t.preventDefault(), f(this))
        }
        return x ? (X || (X=!0, R.click(function() {
            J.next()
        }), P.click(function() {
            J.prev()
        }), K.click(function() {
            J.close()
        }), g.click(function() {
            _.overlayClose && J.close()
        }), t(e).bind("keydown." + te, function(t) {
            var e = t.keyCode;
            $ && _.escKey && 27 === e && (t.preventDefault(), J.close()), $ && _.arrowKey && E[1]&&!t.altKey && (37 === e ? (t.preventDefault(), P.click()) : 39 === e && (t.preventDefault(), R.click()))
        }), t.isFunction(t.fn.on) ? t(e).on("click." + te, "." + ee, i) : t("." + ee).live("click." + te, i)), !0) : !1
    }
    function w() {
        var n, r, a, u = J.prep, f=++de;
        q=!0, U=!1, A = E[j], d(), c(le), c(oe, _.onLoad), _.h = _.height ? h(_.height, "y") - N - D : _.innerHeight && h(_.innerHeight, "y"), _.w = _.width ? h(_.width, "x") - z - B : _.innerWidth && h(_.innerWidth, "x"), _.mw = _.w, _.mh = _.h, _.maxWidth && (_.mw = h(_.maxWidth, "x") - z - B, _.mw = _.w && _.w < _.mw ? _.w : _.mw), _.maxHeight && (_.mh = h(_.maxHeight, "y") - N - D, _.mh = _.h && _.h < _.mh ? _.h : _.mh), n = _.href, Q = setTimeout(function() {
            L.show()
        }, 100), _.inline ? (a = o(ae).hide().insertBefore(t(n)[0]), se.one(le, function() {
            a.replaceWith(W.children())
        }), u(t(n))) : _.iframe ? u(" ") : _.html ? u(_.html) : l(_, n) ? (n = s(_, n), U = e.createElement("img"), t(U).addClass(te + "Photo").bind("error", function() {
            _.title=!1, u(o(ae, "Error").html(_.imgError))
        }).one("load", function() {
            var e;
            f === de && (U.alt = t(A).attr("alt") || t(A).attr("data-alt") || "", _.retinaImage && i.devicePixelRatio > 1 && (U.height = U.height / i.devicePixelRatio, U.width = U.width / i.devicePixelRatio), _.scalePhotos && (r = function() {
                U.height -= U.height * e, U.width -= U.width * e
            }, _.mw && U.width > _.mw && (e = (U.width - _.mw) / U.width, r()), _.mh && U.height > _.mh && (e = (U.height - _.mh) / U.height, r())), _.h && (U.style.marginTop = Math.max(_.mh - U.height, 0) / 2 + "px"), E[1] && (_.loop || E[j + 1]) && (U.style.cursor = "pointer", U.onclick = function() {
                J.next()
            }), U.style.width = U.width + "px", U.style.height = U.height + "px", setTimeout(function() {
                u(U)
            }, 1))
        }), setTimeout(function() {
            U.src = n
        }, 1)) : n && F.load(n, _.data, function(e, i) {
            f === de && u("error" === i ? o(ae, "Error").html(_.xhrError) : t(this).contents())
        })
    }
    var g, x, y, v, b, C, T, k, E, H, W, F, L, S, I, M, R, P, K, O, _, D, B, N, z, A, j, U, $, q, G, Q, J, V, X, Y = {
        transition: "elastic",
        speed: 300,
        fadeOut: 300,
        width: !1,
        initialWidth: "600",
        innerWidth: !1,
        maxWidth: !1,
        height: !1,
        initialHeight: "450",
        innerHeight: !1,
        maxHeight: !1,
        scalePhotos: !0,
        scrolling: !0,
        inline: !1,
        html: !1,
        iframe: !1,
        fastIframe: !0,
        photo: !1,
        href: !1,
        title: !1,
        rel: !1,
        opacity: .9,
        preloading: !0,
        className: !1,
        retinaImage: !1,
        retinaUrl: !1,
        retinaSuffix: "@2x.$1",
        current: "image {current} of {total}",
        previous: "previous",
        next: "next",
        close: "close",
        xhrError: "This content failed to load.",
        imgError: "This image failed to load.",
        open: !1,
        returnFocus: !0,
        trapFocus: !0,
        reposition: !0,
        loop: !0,
        slideshow: !1,
        slideshowAuto: !0,
        slideshowSpeed: 2500,
        slideshowStart: "start slideshow",
        slideshowStop: "stop slideshow",
        photoRegex: /\.(gif|png|jp(e|g|eg)|bmp|ico|webp)((#|\?).*)?$/i,
        onOpen: !1,
        onLoad: !1,
        onComplete: !1,
        onCleanup: !1,
        onClosed: !1,
        overlayClose: !0,
        escKey: !0,
        arrowKey: !0,
        top: !1,
        bottom: !1,
        left: !1,
        right: !1,
        fixed: !1,
        data: void 0
    }, Z = "colorbox", te = "cbox", ee = te + "Element", ie = te + "_open", oe = te + "_load", ne = te + "_complete", re = te + "_cleanup", he = te + "_closed", le = te + "_purge", se = t("<a/>"), ae = "div", de = 0;
    t.colorbox || (t(p), J = t.fn[Z] = t[Z] = function(e, i) {
        var o = this;
        if (e = e || {}, p(), m()) {
            if (t.isFunction(o))
                o = t("<a/>"), e.open=!0;
            else if (!o[0])
                return o;
            i && (e.onComplete = i), o.each(function() {
                t.data(this, Z, t.extend({}, t.data(this, Z) || Y, e))
            }).addClass(ee), (t.isFunction(e.open) && e.open.call(o) || e.open) && f(o[0])
        }
        return o
    }, J.position = function(t, e) {
        function i(t) {
            b[0].style.width = k[0].style.width = v[0].style.width = parseInt(t.style.width, 10) - B + "px", v[0].style.height = C[0].style.height = T[0].style.height = parseInt(t.style.height, 10) - D + "px"
        }
        var o, r, l, s = 0, a = 0, d = x.offset();
        H.unbind("resize." + te), x.css({
            top: - 9e4,
            left: - 9e4
        }), r = H.scrollTop(), l = H.scrollLeft(), _.fixed ? (d.top -= r, d.left -= l, x.css({
            position: "fixed"
        })) : (s = r, a = l, x.css({
            position: "absolute"
        })), a += _.right!==!1 ? Math.max(H.width() - _.w - z - B - h(_.right, "x"), 0) : _.left!==!1 ? h(_.left, "x") : Math.round(Math.max(H.width() - _.w - z - B, 0) / 2), s += _.bottom!==!1 ? Math.max(n() - _.h - N - D - h(_.bottom, "y"), 0) : _.top!==!1 ? h(_.top, "y") : Math.round(Math.max(n() - _.h - N - D, 0) / 2), x.css({
            top: d.top,
            left: d.left,
            visibility: "visible"
        }), t = x.width() === _.w + z && x.height() === _.h + N ? 0 : t || 0, y[0].style.width = y[0].style.height = "9999px", o = {
            width: _.w + z + B,
            height: _.h + N + D,
            top: s,
            left: a
        }, 0 === t && x.css(o), x.dequeue().animate(o, {
            duration: t,
            complete: function() {
                i(this), q=!1, y[0].style.width = _.w + z + B + "px", y[0].style.height = _.h + N + D + "px", _.reposition && setTimeout(function() {
                    H.bind("resize." + te, J.position)
                }, 1), e && e()
            },
            step: function() {
                i(this)
            }
        })
    }, J.resize = function(t) {
        $ && (t = t || {}, t.width && (_.w = h(t.width, "x") - z - B), t.innerWidth && (_.w = h(t.innerWidth, "x")), W.css({
            width: _.w
        }), t.height && (_.h = h(t.height, "y") - N - D), t.innerHeight && (_.h = h(t.innerHeight, "y")), t.innerHeight || t.height || (W.css({
            height: "auto"
        }), _.h = W.height()), W.css({
            height: _.h
        }), J.position("none" === _.transition ? 0 : _.speed))
    }, J.prep = function(e) {
        function i() {
            return _.w = _.w || W.width(), _.w = _.mw && _.mw < _.w ? _.mw : _.w, _.w
        }
        function n() {
            return _.h = _.h || W.height(), _.h = _.mh && _.mh < _.h ? _.mh : _.h, _.h
        }
        if ($) {
            var h, a = "none" === _.transition ? 0: _.speed;
            W.empty().remove(), W = o(ae, "LoadedContent").append(e), W.hide().appendTo(F.show()).css({
                width: i(),
                overflow: _.scrolling ? "auto": "hidden"
            }).css({
                height: n()
            }).prependTo(v), F.hide(), t(U).css({
                "float": "none"
            }), h = function() {
                function e() {
                    t.support.opacity===!1 && x[0].style.removeAttribute("filter")
                }
                var i, n, h = E.length, d = "frameBorder", u = "allowTransparency";
                $ && (n = function() {
                    clearTimeout(Q), L.hide(), c(ne, _.onComplete)
                }, S.html(_.title).add(W).show(), h > 1 ? ("string" == typeof _.current && I.html(_.current.replace("{current}", j + 1).replace("{total}", h)).show(), R[_.loop || h - 1 > j ? "show": "hide"]().html(_.next), P[_.loop || j ? "show": "hide"]().html(_.previous), _.slideshow && M.show(), _.preloading && t.each([r( - 1), r(1)], function() {
                    var e, i, o = E[this], n = t.data(o, Z);
                    n && n.href ? (e = n.href, t.isFunction(e) && (e = e.call(o))) : e = t(o).attr("href"), e && l(n, e) && (e = s(n, e), i = new Image, i.src = e)
                })) : O.hide(), _.iframe ? (i = o("iframe")[0], d in i && (i[d] = 0), u in i && (i[u] = "true"), _.scrolling || (i.scrolling = "no"), t(i).attr({
                    src: _.href,
                    name: (new Date).getTime(),
                    "class": te + "Iframe",
                    allowFullScreen: !0,
                    webkitAllowFullScreen: !0,
                    mozallowfullscreen: !0
                }).one("load", n).appendTo(W), se.one(le, function() {
                    i.src = "//about:blank"
                }), _.fastIframe && t(i).trigger("load")) : n(), "fade" === _.transition ? x.fadeTo(a, 1, e) : e())
            }, "fade" === _.transition ? x.fadeTo(a, 0, function() {
                J.position(0, h)
            }) : J.position(a, h)
        }
    }, J.next = function() {
        !q && E[1] && (_.loop || E[j + 1]) && (j = r(1), f(E[j]))
    }, J.prev = function() {
        !q && E[1] && (_.loop || j) && (j = r( - 1), f(E[j]))
    }, J.close = function() {
        $&&!G && (G=!0, $=!1, c(re, _.onCleanup), H.unbind("." + te), g.fadeTo(_.fadeOut || 0, 0), x.stop().fadeTo(_.fadeOut || 0, 0, function() {
            x.add(g).css({
                opacity: 1,
                cursor: "auto"
            }).hide(), c(le), W.empty().remove(), setTimeout(function() {
                G=!1, c(he, _.onClosed)
            }, 1)
        }))
    }, J.remove = function() {
        x && (x.stop(), t.colorbox.close(), x.stop().remove(), g.remove(), G=!1, x = null, t("." + ee).removeData(Z).removeClass(ee), t(e).unbind("click." + te))
    }, J.element = function() {
        return t(A)
    }, J.settings = Y)
})(jQuery, document, window);



/*! Magnific Popup - v0.9.9 - 2013-12-27
 * http://dimsemenov.com/plugins/magnific-popup/
 * Copyright (c) 2013 Dmitry Semenov; */
(function(e) {
    var t, n, i, o, r, a, s, l = "Close", c = "BeforeClose", d = "AfterClose", u = "BeforeAppend", p = "MarkupParse", f = "Open", m = "Change", g = "mfp", h = "." + g, v = "mfp-ready", C = "mfp-removing", y = "mfp-prevent-close", w = function() {}, b=!!window.jQuery, I = e(window), x = function(e, n) {
        t.ev.on(g + e + h, n)
    }, k = function(t, n, i, o) {
        var r = document.createElement("div");
        return r.className = "mfp-" + t, i && (r.innerHTML = i), o ? n && n.appendChild(r) : (r = e(r), n && r.appendTo(n)), r
    }, T = function(n, i) {
        t.ev.triggerHandler(g + n, i), t.st.callbacks && (n = n.charAt(0).toLowerCase() + n.slice(1), t.st.callbacks[n] && t.st.callbacks[n].apply(t, e.isArray(i) ? i : [i]))
    }, E = function(n) {
        return n === s && t.currTemplate.closeBtn || (t.currTemplate.closeBtn = e(t.st.closeMarkup.replace("%title%", t.st.tClose)), s = n), t.currTemplate.closeBtn
    }, _ = function() {
        e.magnificPopup.instance || (t = new w, t.init(), e.magnificPopup.instance = t)
    }, S = function() {
        var e = document.createElement("p").style, t = ["ms", "O", "Moz", "Webkit"];
        if (void 0 !== e.transition)
            return !0;
        for (; t.length;)
            if (t.pop() + "Transition"in e)
                return !0;
        return !1
    };
    w.prototype = {
        constructor: w,
        init: function() {
            var n = navigator.appVersion;
            t.isIE7 =- 1 !== n.indexOf("MSIE 7."), t.isIE8 =- 1 !== n.indexOf("MSIE 8."), t.isLowIE = t.isIE7 || t.isIE8, t.isAndroid = /android/gi.test(n), t.isIOS = /iphone|ipad|ipod/gi.test(n), t.supportsTransition = S(), t.probablyMobile = t.isAndroid || t.isIOS || /(Opera Mini)|Kindle|webOS|BlackBerry|(Opera Mobi)|(Windows Phone)|IEMobile/i.test(navigator.userAgent), o = e(document), t.popupsCache = {}
        },
        open: function(n) {
            i || (i = e(document.body));
            var r;
            if (n.isObj===!1) {
                t.items = n.items.toArray(), t.index = 0;
                var s, l = n.items;
                for (r = 0; l.length > r; r++)
                    if (s = l[r], s.parsed && (s = s.el[0]), s === n.el[0]) {
                        t.index = r;
                        break
                    }
            } else 
                t.items = e.isArray(n.items) ? n.items : [n.items], t.index = n.index || 0;
            if (t.isOpen)
                return t.updateItemHTML(), void 0;
            t.types = [], a = "", t.ev = n.mainEl && n.mainEl.length ? n.mainEl.eq(0) : o, n.key ? (t.popupsCache[n.key] || (t.popupsCache[n.key] = {}), t.currTemplate = t.popupsCache[n.key]) : t.currTemplate = {}, t.st = e.extend(!0, {}, e.magnificPopup.defaults, n), t.fixedContentPos = "auto" === t.st.fixedContentPos?!t.probablyMobile : t.st.fixedContentPos, t.st.modal && (t.st.closeOnContentClick=!1, t.st.closeOnBgClick=!1, t.st.showCloseBtn=!1, t.st.enableEscapeKey=!1), t.bgOverlay || (t.bgOverlay = k("bg").on("click" + h, function() {
                t.close()
            }), t.wrap = k("wrap").attr("tabindex", - 1).on("click" + h, function(e) {
                t._checkIfClose(e.target) && t.close()
            }), t.container = k("container", t.wrap)), t.contentContainer = k("content"), t.st.preloader && (t.preloader = k("preloader", t.container, t.st.tLoading));
            var c = e.magnificPopup.modules;
            for (r = 0; c.length > r; r++) {
                var d = c[r];
                d = d.charAt(0).toUpperCase() + d.slice(1), t["init" + d].call(t)
            }
            T("BeforeOpen"), t.st.showCloseBtn && (t.st.closeBtnInside ? (x(p, function(e, t, n, i) {
                n.close_replaceWith = E(i.type)
            }), a += " mfp-close-btn-in") : t.wrap.append(E())), t.st.alignTop && (a += " mfp-align-top"), t.fixedContentPos ? t.wrap.css({
                overflow: t.st.overflowY,
                overflowX: "hidden",
                overflowY: t.st.overflowY
            }) : t.wrap.css({
                top: I.scrollTop(),
                position: "absolute"
            }), (t.st.fixedBgPos===!1 || "auto" === t.st.fixedBgPos&&!t.fixedContentPos) && t.bgOverlay.css({
                height: o.height(),
                position: "absolute"
            }), t.st.enableEscapeKey && o.on("keyup" + h, function(e) {
                27 === e.keyCode && t.close()
            }), I.on("resize" + h, function() {
                t.updateSize()
            }), t.st.closeOnContentClick || (a += " mfp-auto-cursor"), a && t.wrap.addClass(a);
            var u = t.wH = I.height(), m = {};
            if (t.fixedContentPos && t._hasScrollBar(u)) {
                var g = t._getScrollbarSize();
                g && (m.marginRight = g)
            }
            t.fixedContentPos && (t.isIE7 ? e("body, html").css("overflow", "hidden") : m.overflow = "hidden");
            var C = t.st.mainClass;
            return t.isIE7 && (C += " mfp-ie7"), C && t._addClassToMFP(C), t.updateItemHTML(), T("BuildControls"), e("html").css(m), t.bgOverlay.add(t.wrap).prependTo(t.st.prependTo || i), t._lastFocusedEl = document.activeElement, setTimeout(function() {
                t.content ? (t._addClassToMFP(v), t._setFocus()) : t.bgOverlay.addClass(v), o.on("focusin" + h, t._onFocusIn)
            }, 16), t.isOpen=!0, t.updateSize(u), T(f), n
        },
        close: function() {
            t.isOpen && (T(c), t.isOpen=!1, t.st.removalDelay&&!t.isLowIE && t.supportsTransition ? (t._addClassToMFP(C), setTimeout(function() {
                t._close()
            }, t.st.removalDelay)) : t._close())
        },
        _close: function() {
            T(l);
            var n = C + " " + v + " ";
            if (t.bgOverlay.detach(), t.wrap.detach(), t.container.empty(), t.st.mainClass && (n += t.st.mainClass + " "), t._removeClassFromMFP(n), t.fixedContentPos) {
                var i = {
                    marginRight: ""
                };
                t.isIE7 ? e("body, html").css("overflow", "") : i.overflow = "", e("html").css(i)
            }
            o.off("keyup" + h + " focusin" + h), t.ev.off(h), t.wrap.attr("class", "mfp-wrap").removeAttr("style"), t.bgOverlay.attr("class", "mfp-bg"), t.container.attr("class", "mfp-container"), !t.st.showCloseBtn || t.st.closeBtnInside && t.currTemplate[t.currItem.type]!==!0 || t.currTemplate.closeBtn && t.currTemplate.closeBtn.detach(), t._lastFocusedEl && e(t._lastFocusedEl).focus(), t.currItem = null, t.content = null, t.currTemplate = null, t.prevHeight = 0, T(d)
        },
        updateSize: function(e) {
            if (t.isIOS) {
                var n = document.documentElement.clientWidth / window.innerWidth, i = window.innerHeight * n;
                t.wrap.css("height", i), t.wH = i
            } else 
                t.wH = e || I.height();
            t.fixedContentPos || t.wrap.css("height", t.wH), T("Resize")
        },
        updateItemHTML: function() {
            var n = t.items[t.index];
            t.contentContainer.detach(), t.content && t.content.detach(), n.parsed || (n = t.parseEl(t.index));
            var i = n.type;
            if (T("BeforeChange", [t.currItem ? t.currItem.type: "", i]), t.currItem = n, !t.currTemplate[i]) {
                var o = t.st[i] ? t.st[i].markup: !1;
                T("FirstMarkupParse", o), t.currTemplate[i] = o ? e(o) : !0
            }
            r && r !== n.type && t.container.removeClass("mfp-" + r + "-holder");
            var a = t["get" + i.charAt(0).toUpperCase() + i.slice(1)](n, t.currTemplate[i]);
            t.appendContent(a, i), n.preloaded=!0, T(m, n), r = n.type, t.container.prepend(t.contentContainer), T("AfterChange")
        },
        appendContent: function(e, n) {
            t.content = e, e ? t.st.showCloseBtn && t.st.closeBtnInside && t.currTemplate[n]===!0 ? t.content.find(".mfp-close").length || t.content.append(E()) : t.content = e : t.content = "", T(u), t.container.addClass("mfp-" + n + "-holder"), t.contentContainer.append(t.content)
        },
        parseEl: function(n) {
            var i, o = t.items[n];
            if (o.tagName ? o = {
                el: e(o)
            } : (i = o.type, o = {
                data: o,
                src: o.src
            }), o.el) {
                for (var r = t.types, a = 0; r.length > a; a++)
                    if (o.el.hasClass("mfp-" + r[a])) {
                        i = r[a];
                        break
                    }
                o.src = o.el.attr("data-mfp-src"), o.src || (o.src = o.el.attr("href"))
            }
            return o.type = i || t.st.type || "inline", o.index = n, o.parsed=!0, t.items[n] = o, T("ElementParse", o), t.items[n]
        },
        addGroup: function(e, n) {
            var i = function(i) {
                i.mfpEl = this, t._openClick(i, e, n)
            };
            n || (n = {});
            var o = "click.magnificPopup";
            n.mainEl = e, n.items ? (n.isObj=!0, e.off(o).on(o, i)) : (n.isObj=!1, n.delegate ? e.off(o).on(o, n.delegate, i) : (n.items = e, e.off(o).on(o, i)))
        },
        _openClick: function(n, i, o) {
            var r = void 0 !== o.midClick ? o.midClick : e.magnificPopup.defaults.midClick;
            if (r || 2 !== n.which&&!n.ctrlKey&&!n.metaKey) {
                var a = void 0 !== o.disableOn ? o.disableOn : e.magnificPopup.defaults.disableOn;
                if (a)
                    if (e.isFunction(a)) {
                        if (!a.call(t))
                            return !0
                    } else if (a > I.width())
                        return !0;
                n.type && (n.preventDefault(), t.isOpen && n.stopPropagation()), o.el = e(n.mfpEl), o.delegate && (o.items = i.find(o.delegate)), t.open(o)
            }
        },
        updateStatus: function(e, i) {
            if (t.preloader) {
                n !== e && t.container.removeClass("mfp-s-" + n), i || "loading" !== e || (i = t.st.tLoading);
                var o = {
                    status: e,
                    text: i
                };
                T("UpdateStatus", o), e = o.status, i = o.text, t.preloader.html(i), t.preloader.find("a").on("click", function(e) {
                    e.stopImmediatePropagation()
                }), t.container.addClass("mfp-s-" + e), n = e
            }
        },
        _checkIfClose: function(n) {
            if (!e(n).hasClass(y)) {
                var i = t.st.closeOnContentClick, o = t.st.closeOnBgClick;
                if (i && o)
                    return !0;
                if (!t.content || e(n).hasClass("mfp-close") || t.preloader && n === t.preloader[0])
                    return !0;
                if (n === t.content[0] || e.contains(t.content[0], n)) {
                    if (i)
                        return !0
                } else if (o && e.contains(document, n))
                    return !0;
                return !1
            }
        },
        _addClassToMFP: function(e) {
            t.bgOverlay.addClass(e), t.wrap.addClass(e)
        },
        _removeClassFromMFP: function(e) {
            this.bgOverlay.removeClass(e), t.wrap.removeClass(e)
        },
        _hasScrollBar: function(e) {
            return (t.isIE7 ? o.height() : document.body.scrollHeight) > (e || I.height())
        },
        _setFocus: function() {
            (t.st.focus ? t.content.find(t.st.focus).eq(0) : t.wrap).focus()
        },
        _onFocusIn: function(n) {
            return n.target === t.wrap[0] || e.contains(t.wrap[0], n.target) ? void 0 : (t._setFocus(), !1)
        },
        _parseMarkup: function(t, n, i) {
            var o;
            i.data && (n = e.extend(i.data, n)), T(p, [t, n, i]), e.each(n, function(e, n) {
                if (void 0 === n || n===!1)
                    return !0;
                if (o = e.split("_"), o.length > 1) {
                    var i = t.find(h + "-" + o[0]);
                    if (i.length > 0) {
                        var r = o[1];
                        "replaceWith" === r ? i[0] !== n[0] && i.replaceWith(n) : "img" === r ? i.is("img") ? i.attr("src", n) : i.replaceWith('<img src="' + n + '" class="' + i.attr("class") + '" />') : i.attr(o[1], n)
                    }
                } else 
                    t.find(h + "-" + e).html(n)
            })
        },
        _getScrollbarSize: function() {
            if (void 0 === t.scrollbarSize) {
                var e = document.createElement("div");
                e.id = "mfp-sbm", e.style.cssText = "width: 99px; height: 99px; overflow: scroll; position: absolute; top: -9999px;", document.body.appendChild(e), t.scrollbarSize = e.offsetWidth - e.clientWidth, document.body.removeChild(e)
            }
            return t.scrollbarSize
        }
    }, e.magnificPopup = {
        instance: null,
        proto: w.prototype,
        modules: [],
        open: function(t, n) {
            return _(), t = t ? e.extend(!0, {}, t) : {}, t.isObj=!0, t.index = n || 0, this.instance.open(t)
        },
        close: function() {
            return e.magnificPopup.instance && e.magnificPopup.instance.close()
        },
        registerModule: function(t, n) {
            n.options && (e.magnificPopup.defaults[t] = n.options), e.extend(this.proto, n.proto), this.modules.push(t)
        },
        defaults: {
            disableOn: 0,
            key: null,
            midClick: !1,
            mainClass: "",
            preloader: !0,
            focus: "",
            closeOnContentClick: !1,
            closeOnBgClick: !0,
            closeBtnInside: !0,
            showCloseBtn: !0,
            enableEscapeKey: !0,
            modal: !1,
            alignTop: !1,
            removalDelay: 0,
            prependTo: null,
            fixedContentPos: "auto",
            fixedBgPos: "auto",
            overflowY: "auto",
            closeMarkup: '<button title="%title%" type="button" class="mfp-close">&times;</button>',
            tClose: "Close (Esc)",
            tLoading: "Loading..."
        }
    }, e.fn.magnificPopup = function(n) {
        _();
        var i = e(this);
        if ("string" == typeof n)
            if ("open" === n) {
                var o, r = b ? i.data("magnificPopup"): i[0].magnificPopup, a = parseInt(arguments[1], 10) || 0;
                r.items ? o = r.items[a] : (o = i, r.delegate && (o = o.find(r.delegate)), o = o.eq(a)), t._openClick({
                    mfpEl: o
                }, i, r)
            } else 
                t.isOpen && t[n].apply(t, Array.prototype.slice.call(arguments, 1));
        else 
            n = e.extend(!0, {}, n), b ? i.data("magnificPopup", n) : i[0].magnificPopup = n, t.addGroup(i, n);
        return i
    };
    var P, O, z, M = "inline", B = function() {
        z && (O.after(z.addClass(P)).detach(), z = null)
    };
    e.magnificPopup.registerModule(M, {
        options: {
            hiddenClass: "hide",
            markup: "",
            tNotFound: "Content not found"
        },
        proto: {
            initInline: function() {
                t.types.push(M), x(l + "." + M, function() {
                    B()
                })
            },
            getInline: function(n, i) {
                if (B(), n.src) {
                    var o = t.st.inline, r = e(n.src);
                    if (r.length) {
                        var a = r[0].parentNode;
                        a && a.tagName && (O || (P = o.hiddenClass, O = k(P), P = "mfp-" + P), z = r.after(O).detach().removeClass(P)), t.updateStatus("ready")
                    } else 
                        t.updateStatus("error", o.tNotFound), r = e("<div>");
                    return n.inlineElement = r, r
                }
                return t.updateStatus("ready"), t._parseMarkup(i, {}, n), i
            }
        }
    });
    var F, H = "ajax", L = function() {
        F && i.removeClass(F)
    }, A = function() {
        L(), t.req && t.req.abort()
    };
    e.magnificPopup.registerModule(H, {
        options: {
            settings: null,
            cursor: "mfp-ajax-cur",
            tError: '<a href="%url%">The content</a> could not be loaded.'
        },
        proto: {
            initAjax: function() {
                t.types.push(H), F = t.st.ajax.cursor, x(l + "." + H, A), x("BeforeChange." + H, A)
            },
            getAjax: function(n) {
                F && i.addClass(F), t.updateStatus("loading");
                var o = e.extend({
                    url: n.src,
                    success: function(i, o, r) {
                        var a = {
                            data: i,
                            xhr: r
                        };
                        T("ParseAjax", a), t.appendContent(e(a.data), H), n.finished=!0, L(), t._setFocus(), setTimeout(function() {
                            t.wrap.addClass(v)
                        }, 16), t.updateStatus("ready"), T("AjaxContentAdded")
                    },
                    error: function() {
                        L(), n.finished = n.loadError=!0, t.updateStatus("error", t.st.ajax.tError.replace("%url%", n.src))
                    }
                }, t.st.ajax.settings);
                return t.req = e.ajax(o), ""
            }
        }
    });
    var j, N = function(n) {
        if (n.data && void 0 !== n.data.title)
            return n.data.title;
        var i = t.st.image.titleSrc;
        if (i) {
            if (e.isFunction(i))
                return i.call(t, n);
            if (n.el)
                return n.el.attr(i) || ""
        }
        return ""
    };
    e.magnificPopup.registerModule("image", {
        options: {
            markup: '<div class="mfp-figure"><div class="mfp-close"></div><figure><div class="mfp-img"></div><figcaption><div class="mfp-bottom-bar"><div class="mfp-title"></div><div class="mfp-counter"></div></div></figcaption></figure></div>',
            cursor: "mfp-zoom-out-cur",
            titleSrc: "title",
            verticalFit: !0,
            tError: '<a href="%url%">The image</a> could not be loaded.'
        },
        proto: {
            initImage: function() {
                var e = t.st.image, n = ".image";
                t.types.push("image"), x(f + n, function() {
                    "image" === t.currItem.type && e.cursor && i.addClass(e.cursor)
                }), x(l + n, function() {
                    e.cursor && i.removeClass(e.cursor), I.off("resize" + h)
                }), x("Resize" + n, t.resizeImage), t.isLowIE && x("AfterChange", t.resizeImage)
            },
            resizeImage: function() {
                var e = t.currItem;
                if (e && e.img && t.st.image.verticalFit) {
                    var n = 0;
                    t.isLowIE && (n = parseInt(e.img.css("padding-top"), 10) + parseInt(e.img.css("padding-bottom"), 10)), e.img.css("max-height", t.wH - n)
                }
            },
            _onImageHasSize: function(e) {
                e.img && (e.hasSize=!0, j && clearInterval(j), e.isCheckingImgSize=!1, T("ImageHasSize", e), e.imgHidden && (t.content && t.content.removeClass("mfp-loading"), e.imgHidden=!1))
            },
            findImageSize: function(e) {
                var n = 0, i = e.img[0], o = function(r) {
                    j && clearInterval(j), j = setInterval(function() {
                        return i.naturalWidth > 0 ? (t._onImageHasSize(e), void 0) : (n > 200 && clearInterval(j), n++, 3 === n ? o(10) : 40 === n ? o(50) : 100 === n && o(500), void 0)
                    }, r)
                };
                o(1)
            },
            getImage: function(n, i) {
                var o = 0, r = function() {
                    n && (n.img[0].complete ? (n.img.off(".mfploader"), n === t.currItem && (t._onImageHasSize(n), t.updateStatus("ready")), n.hasSize=!0, n.loaded=!0, T("ImageLoadComplete")) : (o++, 200 > o ? setTimeout(r, 100) : a()))
                }, a = function() {
                    n && (n.img.off(".mfploader"), n === t.currItem && (t._onImageHasSize(n), t.updateStatus("error", s.tError.replace("%url%", n.src))), n.hasSize=!0, n.loaded=!0, n.loadError=!0)
                }, s = t.st.image, l = i.find(".mfp-img");
                if (l.length) {
                    var c = document.createElement("img");
                    c.className = "mfp-img", n.img = e(c).on("load.mfploader", r).on("error.mfploader", a), c.src = n.src, l.is("img") && (n.img = n.img.clone()), c = n.img[0], c.naturalWidth > 0 ? n.hasSize=!0 : c.width || (n.hasSize=!1)
                }
                return t._parseMarkup(i, {
                    title: N(n),
                    img_replaceWith: n.img
                }, n), t.resizeImage(), n.hasSize ? (j && clearInterval(j), n.loadError ? (i.addClass("mfp-loading"), t.updateStatus("error", s.tError.replace("%url%", n.src))) : (i.removeClass("mfp-loading"), t.updateStatus("ready")), i) : (t.updateStatus("loading"), n.loading=!0, n.hasSize || (n.imgHidden=!0, i.addClass("mfp-loading"), t.findImageSize(n)), i)
            }
        }
    });
    var W, R = function() {
        return void 0 === W && (W = void 0 !== document.createElement("p").style.MozTransform), W
    };
    e.magnificPopup.registerModule("zoom", {
        options: {
            enabled: !1,
            easing: "ease-in-out",
            duration: 300,
            opener: function(e) {
                return e.is("img") ? e : e.find("img")
            }
        },
        proto: {
            initZoom: function() {
                var e, n = t.st.zoom, i = ".zoom";
                if (n.enabled && t.supportsTransition) {
                    var o, r, a = n.duration, s = function(e) {
                        var t = e.clone().removeAttr("style").removeAttr("class").addClass("mfp-animated-image"), i = "all " + n.duration / 1e3 + "s " + n.easing, o = {
                            position: "fixed",
                            zIndex: 9999,
                            left: 0,
                            top: 0,
                            "-webkit-backface-visibility": "hidden"
                        }, r = "transition";
                        return o["-webkit-" + r] = o["-moz-" + r] = o["-o-" + r] = o[r] = i, t.css(o), t
                    }, d = function() {
                        t.content.css("visibility", "visible")
                    };
                    x("BuildControls" + i, function() {
                        if (t._allowZoom()) {
                            if (clearTimeout(o), t.content.css("visibility", "hidden"), e = t._getItemToZoom(), !e)
                                return d(), void 0;
                            r = s(e), r.css(t._getOffset()), t.wrap.append(r), o = setTimeout(function() {
                                r.css(t._getOffset(!0)), o = setTimeout(function() {
                                    d(), setTimeout(function() {
                                        r.remove(), e = r = null, T("ZoomAnimationEnded")
                                    }, 16)
                                }, a)
                            }, 16)
                        }
                    }), x(c + i, function() {
                        if (t._allowZoom()) {
                            if (clearTimeout(o), t.st.removalDelay = a, !e) {
                                if (e = t._getItemToZoom(), !e)
                                    return;
                                r = s(e)
                            }
                            r.css(t._getOffset(!0)), t.wrap.append(r), t.content.css("visibility", "hidden"), setTimeout(function() {
                                r.css(t._getOffset())
                            }, 16)
                        }
                    }), x(l + i, function() {
                        t._allowZoom() && (d(), r && r.remove(), e = null)
                    })
                }
            },
            _allowZoom: function() {
                return "image" === t.currItem.type
            },
            _getItemToZoom: function() {
                return t.currItem.hasSize ? t.currItem.img : !1
            },
            _getOffset: function(n) {
                var i;
                i = n ? t.currItem.img : t.st.zoom.opener(t.currItem.el || t.currItem);
                var o = i.offset(), r = parseInt(i.css("padding-top"), 10), a = parseInt(i.css("padding-bottom"), 10);
                o.top -= e(window).scrollTop() - r;
                var s = {
                    width: i.width(),
                    height: (b ? i.innerHeight() : i[0].offsetHeight) - a - r
                };
                return R() ? s["-moz-transform"] = s.transform = "translate(" + o.left + "px," + o.top + "px)" : (s.left = o.left, s.top = o.top), s
            }
        }
    });
    var Z = "iframe", q = "//about:blank", D = function(e) {
        if (t.currTemplate[Z]) {
            var n = t.currTemplate[Z].find("iframe");
            n.length && (e || (n[0].src = q), t.isIE8 && n.css("display", e ? "block" : "none"))
        }
    };
    e.magnificPopup.registerModule(Z, {
        options: {
            markup: '<div class="mfp-iframe-scaler"><div class="mfp-close"></div><iframe class="mfp-iframe" src="//about:blank" frameborder="0" allowfullscreen></iframe></div>',
            srcAction: "iframe_src",
            patterns: {
                youtube: {
                    index: "youtube.com",
                    id: "v=",
                    src: "//www.youtube.com/embed/%id%?autoplay=1"
                },
                vimeo: {
                    index: "vimeo.com/",
                    id: "/",
                    src: "//player.vimeo.com/video/%id%?autoplay=1"
                },
                gmaps: {
                    index: "//maps.google.",
                    src: "%id%&output=embed"
                }
            }
        },
        proto: {
            initIframe: function() {
                t.types.push(Z), x("BeforeChange", function(e, t, n) {
                    t !== n && (t === Z ? D() : n === Z && D(!0))
                }), x(l + "." + Z, function() {
                    D()
                })
            },
            getIframe: function(n, i) {
                var o = n.src, r = t.st.iframe;
                e.each(r.patterns, function() {
                    return o.indexOf(this.index)>-1 ? (this.id && (o = "string" == typeof this.id ? o.substr(o.lastIndexOf(this.id) + this.id.length, o.length) : this.id.call(this, o)), o = this.src.replace("%id%", o), !1) : void 0
                });
                var a = {};
                return r.srcAction && (a[r.srcAction] = o), t._parseMarkup(i, a, n), t.updateStatus("ready"), i
            }
        }
    });
    var K = function(e) {
        var n = t.items.length;
        return e > n - 1 ? e - n : 0 > e ? n + e : e
    }, Y = function(e, t, n) {
        return e.replace(/%curr%/gi, t + 1).replace(/%total%/gi, n)
    };
    e.magnificPopup.registerModule("gallery", {
        options: {
            enabled: !1,
            arrowMarkup: '<button title="%title%" type="button" class="mfp-arrow mfp-arrow-%dir%"></button>',
            preload: [0, 2],
            navigateByImgClick: !0,
            arrows: !0,
            tPrev: "Previous (Left arrow key)",
            tNext: "Next (Right arrow key)",
            tCounter: "%curr% of %total%"
        },
        proto: {
            initGallery: function() {
                var n = t.st.gallery, i = ".mfp-gallery", r = Boolean(e.fn.mfpFastClick);
                return t.direction=!0, n && n.enabled ? (a += " mfp-gallery", x(f + i, function() {
                    n.navigateByImgClick && t.wrap.on("click" + i, ".mfp-img", function() {
                        return t.items.length > 1 ? (t.next(), !1) : void 0
                    }), o.on("keydown" + i, function(e) {
                        37 === e.keyCode ? t.prev() : 39 === e.keyCode && t.next()
                    })
                }), x("UpdateStatus" + i, function(e, n) {
                    n.text && (n.text = Y(n.text, t.currItem.index, t.items.length))
                }), x(p + i, function(e, i, o, r) {
                    var a = t.items.length;
                    o.counter = a > 1 ? Y(n.tCounter, r.index, a) : ""
                }), x("BuildControls" + i, function() {
                    if (t.items.length > 1 && n.arrows&&!t.arrowLeft) {
                        var i = n.arrowMarkup, o = t.arrowLeft = e(i.replace(/%title%/gi, n.tPrev).replace(/%dir%/gi, "left")).addClass(y), a = t.arrowRight = e(i.replace(/%title%/gi, n.tNext).replace(/%dir%/gi, "right")).addClass(y), s = r ? "mfpFastClick": "click";
                        o[s](function() {
                            t.prev()
                        }), a[s](function() {
                            t.next()
                        }), t.isIE7 && (k("b", o[0], !1, !0), k("a", o[0], !1, !0), k("b", a[0], !1, !0), k("a", a[0], !1, !0)), t.container.append(o.add(a))
                    }
                }), x(m + i, function() {
                    t._preloadTimeout && clearTimeout(t._preloadTimeout), t._preloadTimeout = setTimeout(function() {
                        t.preloadNearbyImages(), t._preloadTimeout = null
                    }, 16)
                }), x(l + i, function() {
                    o.off(i), t.wrap.off("click" + i), t.arrowLeft && r && t.arrowLeft.add(t.arrowRight).destroyMfpFastClick(), t.arrowRight = t.arrowLeft = null
                }), void 0) : !1
            },
            next: function() {
                t.direction=!0, t.index = K(t.index + 1), t.updateItemHTML()
            },
            prev: function() {
                t.direction=!1, t.index = K(t.index - 1), t.updateItemHTML()
            },
            goTo: function(e) {
                t.direction = e >= t.index, t.index = e, t.updateItemHTML()
            },
            preloadNearbyImages: function() {
                var e, n = t.st.gallery.preload, i = Math.min(n[0], t.items.length), o = Math.min(n[1], t.items.length);
                for (e = 1; (t.direction ? o : i) >= e; e++)
                    t._preloadItem(t.index + e);
                for (e = 1; (t.direction ? i : o) >= e; e++)
                    t._preloadItem(t.index - e)
            },
            _preloadItem: function(n) {
                if (n = K(n), !t.items[n].preloaded) {
                    var i = t.items[n];
                    i.parsed || (i = t.parseEl(n)), T("LazyLoad", i), "image" === i.type && (i.img = e('<img class="mfp-img" />').on("load.mfploader", function() {
                        i.hasSize=!0
                    }).on("error.mfploader", function() {
                        i.hasSize=!0, i.loadError=!0, T("LazyLoadError", i)
                    }).attr("src", i.src)), i.preloaded=!0
                }
            }
        }
    });
    var U = "retina";
    e.magnificPopup.registerModule(U, {
        options: {
            replaceSrc: function(e) {
                return e.src.replace(/\.\w+$/, function(e) {
                    return "@2x" + e
                })
            },
            ratio: 1
        },
        proto: {
            initRetina: function() {
                if (window.devicePixelRatio > 1) {
                    var e = t.st.retina, n = e.ratio;
                    n = isNaN(n) ? n() : n, n > 1 && (x("ImageHasSize." + U, function(e, t) {
                        t.img.css({
                            "max-width": t.img[0].naturalWidth / n,
                            width: "100%"
                        })
                    }), x("ElementParse." + U, function(t, i) {
                        i.src = e.replaceSrc(i, n)
                    }))
                }
            }
        }
    }), function() {
        var t = 1e3, n = "ontouchstart"in window, i = function() {
            I.off("touchmove" + r + " touchend" + r)
        }, o = "mfpFastClick", r = "." + o;
        e.fn.mfpFastClick = function(o) {
            return e(this).each(function() {
                var a, s = e(this);
                if (n) {
                    var l, c, d, u, p, f;
                    s.on("touchstart" + r, function(e) {
                        u=!1, f = 1, p = e.originalEvent ? e.originalEvent.touches[0] : e.touches[0], c = p.clientX, d = p.clientY, I.on("touchmove" + r, function(e) {
                            p = e.originalEvent ? e.originalEvent.touches : e.touches, f = p.length, p = p[0], (Math.abs(p.clientX - c) > 10 || Math.abs(p.clientY - d) > 10) && (u=!0, i())
                        }).on("touchend" + r, function(e) {
                            i(), u || f > 1 || (a=!0, e.preventDefault(), clearTimeout(l), l = setTimeout(function() {
                                a=!1
                            }, t), o())
                        })
                    })
                }
                s.on("click" + r, function() {
                    a || o()
                })
            })
        }, e.fn.destroyMfpFastClick = function() {
            e(this).off("touchstart" + r + " click" + r), n && I.off("touchmove" + r + " touchend" + r)
        }
    }(), _()
})(window.jQuery || window.Zepto);


/**
 * placeholder
 */
(function(q, f, d) {
    function r(b) {
        var a = {}, c = /^jQuery\d+$/;
        d.each(b.attributes, function(b, d) {
            d.specified&&!c.test(d.name) && (a[d.name] = d.value)
        });
        return a
    }
    function g(b, a) {
        var c = d(this);
        if (this.value == c.attr("placeholder") && c.hasClass("placeholder"))
            if (c.data("placeholder-password")) {
                c = c.hide().next().show().attr("id", c.removeAttr("id").data("placeholder-id"));
                if (!0 === b)
                    return c[0].value = a;
                    c.focus()
            } else 
                this.value = "", c.removeClass("placeholder"), this == m() && this.select()
    }
    function k() {
        var b, a = d(this), c =
        this.id;
        if ("" == this.value) {
            if ("password" == this.type) {
                if (!a.data("placeholder-textinput")) {
                    try {
                        b = a.clone().attr({
                            type: "text"
                        })
                    } catch (e) {
                        b = d("<input>").attr(d.extend(r(this), {
                            type: "text"
                        }))
                    }
                    b.removeAttr("name").data({
                        "placeholder-password": a,
                        "placeholder-id": c
                    }).bind("focus.placeholder", g);
                    a.data({
                        "placeholder-textinput": b,
                        "placeholder-id": c
                    }).before(b)
                }
                a = a.removeAttr("id").hide().prev().attr("id", c).show()
            }
            a.addClass("placeholder");
            a[0].value = a.attr("placeholder")
        } else 
            a.removeClass("placeholder")
    }
    function m() {
        try {
            return f.activeElement
        } catch (b) {}
    }
    var h = "placeholder"in f.createElement("input"), l = "placeholder"in f.createElement("textarea"), e = d.fn, n = d.valHooks, p = d.propHooks;
    h && l ? (e = e.placeholder = function() {
        return this
    }, e.input = e.textarea=!0) : (e = e.placeholder = function() {
        this.filter((h ? "textarea" : ":input") + "[placeholder]").not(".placeholder").bind({
            "focus.placeholder": g,
            "blur.placeholder": k
        }).data("placeholder-enabled", !0).trigger("blur.placeholder");
        return this
    }, e.input = h, e.textarea = l, e = {
        get: function(b) {
            var a =
            d(b), c = a.data("placeholder-password");
            return c ? c[0].value : a.data("placeholder-enabled") && a.hasClass("placeholder") ? "" : b.value
        },
        set: function(b, a) {
            var c = d(b), e = c.data("placeholder-password");
            if (e)
                return e[0].value = a;
            if (!c.data("placeholder-enabled"))
                return b.value = a;
            "" == a ? (b.value = a, b != m() && k.call(b)) : c.hasClass("placeholder") ? g.call(b, !0, a) || (b.value = a) : b.value = a;
            return c
        }
    }, h || (n.input = e, p.value = e), l || (n.textarea = e, p.value = e), d(function() {
        d(f).delegate("form", "submit.placeholder", function() {
            var b =
            d(".placeholder", this).each(g);
            setTimeout(function() {
                b.each(k)
            }, 10)
        })
    }), d(q).bind("beforeunload.placeholder", function() {
        d(".placeholder").each(function() {
            this.value = ""
        })
    }))
})(this, document, jQuery);





//fgnass.github.com/spin.js#v2.0.1
!function(a, b) {
    "object" == typeof exports ? module.exports = b() : "function" == typeof define && define.amd ? define(b) : a.Spinner = b()
}(this, function() {
    "use strict";
    function a(a, b) {
        var c, d = document.createElement(a || "div");
        for (c in b)
            d[c] = b[c];
        return d
    }
    function b(a) {
        for (var b = 1, c = arguments.length; c > b; b++)
            a.appendChild(arguments[b]);
        return a
    }
    function c(a, b, c, d) {
        var e = ["opacity", b, ~~(100 * a), c, d].join("-"), f = .01 + c / d * 100, g = Math.max(1 - (1 - a) / b * (100 - f), a), h = j.substring(0, j.indexOf("Animation")).toLowerCase(), i = h && "-" + h + "-" || "";
        return l[e] || (m.insertRule("@" + i + "keyframes " + e + "{0%{opacity:" + g + "}" + f + "%{opacity:" + a + "}" + (f + .01) + "%{opacity:1}" + (f + b)%100 + "%{opacity:" + a + "}100%{opacity:" + g + "}}", m.cssRules.length), l[e] = 1), e
    }
    function d(a, b) {
        var c, d, e = a.style;
        for (b = b.charAt(0).toUpperCase() + b.slice(1), d = 0; d < k.length; d++)
            if (c = k[d] + b, void 0 !== e[c])
                return c;
        return void 0 !== e[b] ? b : void 0
    }
    function e(a, b) {
        for (var c in b)
            a.style[d(a, c) || c] = b[c];
        return a
    }
    function f(a) {
        for (var b = 1; b < arguments.length; b++) {
            var c = arguments[b];
            for (var d in c)
                void 0 === a[d] && (a[d] = c[d])
        }
        return a
    }
    function g(a, b) {
        return "string" == typeof a ? a : a[b%a.length]
    }
    function h(a) {
        this.opts = f(a || {}, h.defaults, n)
    }
    function i() {
        function c(b, c) {
            return a("<" + b + ' xmlns="urn:schemas-microsoft.com:vml" class="spin-vml">', c)
        }
        m.addRule(".spin-vml", "behavior:url(#default#VML)"), h.prototype.lines = function(a, d) {
            function f() {
                return e(c("group", {
                    coordsize: k + " " + k,
                    coordorigin: - j + " " +- j
                }), {
                    width: k,
                    height: k
                })
            }
            function h(a, h, i) {
                b(m, b(e(f(), {
                    rotation: 360 / d.lines * a + "deg",
                    left: ~~h
                }), b(e(c("roundrect", {
                    arcsize: d.corners
                }), {
                    width: j,
                    height: d.width,
                    left: d.radius,
                    top: - d.width>>1,
                    filter: i
                }), c("fill", {
                    color: g(d.color, a),
                    opacity: d.opacity
                }), c("stroke", {
                    opacity: 0
                }))))
            }
            var i, j = d.length + d.width, k = 2 * j, l = 2*-(d.width + d.length) + "px", m = e(f(), {
                position: "absolute",
                top: l,
                left: l
            });
            if (d.shadow)
                for (i = 1; i <= d.lines; i++)
                    h(i, - 2, "progid:DXImageTransform.Microsoft.Blur(pixelradius=2,makeshadow=1,shadowopacity=.3)");
            for (i = 1; i <= d.lines; i++)
                h(i);
            return b(a, m)
        }, h.prototype.opacity = function(a, b, c, d) {
            var e = a.firstChild;
            d = d.shadow && d.lines || 0, e && b + d < e.childNodes.length && (e = e.childNodes[b + d], e = e && e.firstChild, e = e && e.firstChild, e && (e.opacity = c))
        }
    }
    var j, k = ["webkit", "Moz", "ms", "O"], l = {}, m = function() {
        var c = a("style", {
            type: "text/css"
        });
        return b(document.getElementsByTagName("head")[0], c), c.sheet || c.styleSheet
    }(), n = {
        lines: 12,
        length: 7,
        width: 5,
        radius: 10,
        rotate: 0,
        corners: 1,
        color: "#000",
        direction: 1,
        speed: 1,
        trail: 100,
        opacity: .25,
        fps: 20,
        zIndex: 2e9,
        className: "spinner",
        top: "50%",
        left: "50%",
        position: "absolute"
    };
    h.defaults = {}, f(h.prototype, {
        spin: function(b) {
            this.stop();
            {
                var c = this, d = c.opts, f = c.el = e(a(0, {
                    className: d.className
                }), {
                    position: d.position,
                    width: 0,
                    zIndex: d.zIndex
                });
                d.radius + d.length + d.width
            }
            if (e(f, {
                left: d.left,
                top: d.top
            }), b && b.insertBefore(f, b.firstChild || null), f.setAttribute("role", "progressbar"), c.lines(f, c.opts), !j) {
                var g, h = 0, i = (d.lines - 1) * (1 - d.direction) / 2, k = d.fps, l = k / d.speed, m = (1 - d.opacity) / (l * d.trail / 100), n = l / d.lines;
                !function o() {
                    h++;
                    for (var a = 0; a < d.lines; a++)
                        g = Math.max(1 - (h + (d.lines - a) * n)%l * m, d.opacity), c.opacity(f, a * d.direction + i, g, d);
                    c.timeout = c.el && setTimeout(o, ~~(1e3 / k))
                }()
            }
            return c
        },
        stop: function() {
            var a = this.el;
            return a && (clearTimeout(this.timeout), a.parentNode && a.parentNode.removeChild(a), this.el = void 0), this
        },
        lines: function(d, f) {
            function h(b, c) {
                return e(a(), {
                    position: "absolute",
                    width: f.length + f.width + "px",
                    height: f.width + "px",
                    background: b,
                    boxShadow: c,
                    transformOrigin: "left",
                    transform: "rotate("+~~(360 / f.lines * k + f.rotate) + "deg) translate(" + f.radius + "px,0)",
                    borderRadius: (f.corners * f.width>>1) + "px"
                })
            }
            for (var i, k = 0, l = (f.lines - 1) * (1 - f.direction) / 2; k < f.lines; k++)
                i = e(a(), {
                    position: "absolute",
                    top: 1+~(f.width / 2) + "px",
                    transform: f.hwaccel ? "translate3d(0,0,0)": "",
                    opacity: f.opacity,
                    animation: j && c(f.opacity, f.trail, l + k * f.direction, f.lines) + " " + 1 / f.speed + "s linear infinite"
                }), f.shadow && b(i, e(h("#000", "0 0 4px #000"), {
                    top: "2px"
                })), b(d, b(i, h(g(f.color, k), "0 0 1px rgba(0,0,0,.1)")));
            return d
        },
        opacity: function(a, b, c) {
            b < a.childNodes.length && (a.childNodes[b].style.opacity = c)
        }
    });
    var o = e(a("group"), {
        behavior: "url(#default#VML)"
    });
    return !d(o, "transform") && o.adj ? i() : j = d(o, "animation"), h
});




// http://paulirish.com/2011/requestanimationframe-for-smart-animating/
// http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating

// requestAnimationFrame polyfill by Erik Möller. fixes from Paul Irish and Tino Zijdel

// MIT license

(function() {
    var lastTime = 0;
    var vendors = ['ms', 'moz', 'webkit', 'o'];
    for (var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
        window.requestAnimationFrame = window[vendors[x] + 'RequestAnimationFrame'];
        window.cancelAnimationFrame = window[vendors[x] + 'CancelAnimationFrame']
        || window[vendors[x] + 'CancelRequestAnimationFrame'];
    }

    if (!window.requestAnimationFrame)
        window.requestAnimationFrame = function(callback, element) {
            var currTime = new Date().getTime();
            var timeToCall = Math.max(0, 16 - (currTime - lastTime));
            var id = window.setTimeout(function() {
                callback(currTime + timeToCall);
            },
            timeToCall);
            lastTime = currTime + timeToCall;
            return id;
        };

    if (!window.cancelAnimationFrame)
        window.cancelAnimationFrame = function(id) {
            clearTimeout(id);
        };
}());







function td_smooth_scroll() {


    // SmoothScroll v1.2.1
    // Licensed under the terms of the MIT license.

    // People involved
    //  - Balazs Galambosi (maintainer)
    //  - Patrick Brunner  (original idea)
    //  - Michael Herf     (Pulse Algorithm)

    // Scroll Variables (tweakable)
    var defaultOptions = {

        // Scrolling Core
        frameRate : 150,
        // [Hz]
        animationTime : 600,
        // [px]
        stepSize : 120,
        // [px]

        // Pulse (less tweakable)
        // ratio of "tail" to "acceleration"
        pulseAlgorithm : true,
        pulseScale : 8,
        pulseNormalize : 1,

        // Acceleration
        accelerationDelta : 20,
        // 20
        accelerationMax : 1,
        // 1

        // Keyboard Settings
        keyboardSupport : true,
        // option
        arrowScroll : 50,
        // [px]

        // Other
        touchpadSupport : true,
        fixedBackground : true,
        excluded : ""
    };

    var options = defaultOptions;


    // Other Variables
    var isExcluded = false;
    var isFrame = false;
    var direction = {
        x: 0,
        y: 0 
    };
    var initDone = false;
    var root = document.documentElement;
    var activeElement;
    var observer;
    var deltaBuffer = [ 120, 120, 120 ];

    var key = {
        left: 37,
        up: 38,
        right: 39,
        down: 40,
        spacebar: 32,
        pageup: 33,
        pagedown: 34,
        end: 35,
        home: 36 
    };


    /***********************************************
         * SETTINGS
         ***********************************************/
    if (typeof(chrome) !== 'undefined' && typeof(chrome.storage) !== 'undefined') {
        chrome.storage.sync.get(defaultOptions, function (syncedOptions) {

            options = syncedOptions;

            // it seems that sometimes settings come late
            // and we need to test again for excluded pages
            initTest();
        });
    }

    /***********************************************
         * INITIALIZE
         ***********************************************/

    /**
         * Tests if smooth scrolling is allowed. Shuts down everything if not.
         */
    function initTest() {

        var disableKeyboard = false;

        // disable keys for google reader (spacebar conflict)
        if (document.URL.indexOf("google.com/reader/view") > - 1) {
            disableKeyboard = true;
        }

        // disable everything if the page is blacklisted
        if (options.excluded) {
            var domains = options.excluded.split(/[,\n] ?/);
            domains.push("mail.google.com"); // exclude Gmail for now
            for (var i = domains.length; i--;) {
                if (document.URL.indexOf(domains[i]) > - 1) {
                    observer && observer.disconnect();
                    removeEvent("mousewheel", wheel);
                    disableKeyboard = true;
                    isExcluded = true;
                    break;
                }
            }
        }

        // disable keyboard support if anything above requested it
        if (disableKeyboard) {
            removeEvent("keydown", keydown);
        }

        if (options.keyboardSupport && !disableKeyboard) {
            addEvent("keydown", keydown);
        }
    }

    /**
         * Sets up scrolls array, determines if frames are involved.
         */
    function init() {

        if (!document.body) 
            return;

        var body = document.body;
        var html = document.documentElement;
        var windowHeight = window.innerHeight;
        var scrollHeight = body.scrollHeight;

        // check compat mode for root element
        root = (document.compatMode.indexOf('CSS') >= 0) ? html : body;
        activeElement = body;

        initTest();
        initDone = true;

        // Checks if this script is running in a frame
        if (top != self) {
            isFrame = true;
        }

        /**
                 * This fixes a bug where the areas left and right to
                 * the content does not trigger the onmousewheel event
                 * on some pages. e.g.: html, body { height: 100% }
                 */
        else if (scrollHeight > windowHeight &&
        (body.offsetHeight <= windowHeight ||
        html.offsetHeight <= windowHeight)) {

            // DOMChange (throttle): fix height
            var pending = false;
            var refresh = function () {
                if (!pending && html.scrollHeight != document.height) {
                    pending = true; // add a new pending action
                    setTimeout(function () {
                        html.style.height = document.height + 'px';
                        pending = false;
                    }, 500); // act rarely to stay fast
                }
            };
            html.style.height = 'auto';
            setTimeout(refresh, 10);

            var config = {
                attributes: true,
                childList: true,
                characterData: false
            };

            observer = new MutationObserver(refresh);
            observer.observe(body, config);

            // clearfix
            if (root.offsetHeight <= windowHeight) {
                var underlay = document.createElement("div");
                underlay.style.clear = "both";
                body.appendChild(underlay);
            }
        }

        // gmail performance fix
        if (document.URL.indexOf("mail.google.com") > - 1) {
            var s = document.createElement("style");
            s.innerHTML = ".iu { visibility: hidden }";
            (document.getElementsByTagName("head")[0] || html).appendChild(s);
        }
        // facebook better home timeline performance
        // all the HTML resized images make rendering CPU intensive
        else if (document.URL.indexOf("www.facebook.com") > - 1) {
            var home_stream = document.getElementById("home_stream");
            home_stream && (home_stream.style.webkitTransform = "translateZ(0)");
        }
        // disable fixed background
        if (!options.fixedBackground && !isExcluded) {
            body.style.backgroundAttachment = "scroll";
            html.style.backgroundAttachment = "scroll";
        }
    }


    /************************************************
         * SCROLLING
         ************************************************/

    var que = [];
    var pending = false;
    var lastScroll = + new Date;

    /**
         * Pushes scroll actions to the scrolling queue.
         */
    function scrollArray(elem, left, top, delay) {

        delay || (delay = 1000);
        directionCheck(left, top);

        if (options.accelerationMax != 1) {
            var now = + new Date;
            var elapsed = now - lastScroll;
            if (elapsed < options.accelerationDelta) {
                var factor = (1 + (30 / elapsed)) / 2;
                if (factor > 1) {
                    factor = Math.min(factor, options.accelerationMax);
                    left *= factor;
                    top *= factor;
                }
            }
            lastScroll = + new Date;
        }

        // push a scroll command
        que.push({
            x: left,
            y: top,
            lastX: (left < 0) ? 0.99 : - 0.99,
            lastY: (top < 0) ? 0.99 : - 0.99,
            start: + new Date
        });

        // don't act if there's a pending queue
        if (pending) {
            return;
        }

        var scrollWindow = (elem === document.body);

        var step = function (time) {

            var now = + new Date;
            var scrollX = 0;
            var scrollY = 0;

            for (var i = 0; i < que.length; i++) {

                var item = que[i];
                var elapsed = now - item.start;
                var finished = (elapsed >= options.animationTime);

                // scroll position: [0, 1]
                var position = (finished) ? 1 : elapsed / options.animationTime;

                // easing [optional]
                if (options.pulseAlgorithm) {
                    position = pulse(position);
                }

                // only need the difference
                var x = (item.x * position - item.lastX) >> 0;
                var y = (item.y * position - item.lastY) >> 0;

                // add this to the total scrolling
                scrollX += x;
                scrollY += y;

                // update last values
                item.lastX += x;
                item.lastY += y;

                // delete and step back if it's over
                if (finished) {
                    que.splice(i, 1);
                    i--;
                }
            }

            // scroll left and top
            if (scrollWindow) {
                window.scrollBy(scrollX, scrollY);
            } else {
                if (scrollX) 
                    elem.scrollLeft += scrollX;
                if (scrollY) 
                    elem.scrollTop += scrollY;
            }

            // clean up if there's nothing left to do
            if (!left && !top) {
                que = [];
            }

            if (que.length) {
                requestFrame(step, elem, (delay / options.frameRate + 1));
            } else {
                pending = false;
            }
        };

        // start a new queue of actions
        requestFrame(step, elem, 0);
        pending = true;
    }


    /***********************************************
         * EVENTS
         ***********************************************/

    /**
         * Mouse wheel handler.
         * @param {Object} event
         */
    function wheel(event) {

        if (!initDone) {
            init();
        }

        var target = event.target;
        var overflowing = overflowingAncestor(target);

        // use default if there's no overflowing
        // element or default action is prevented
        if (!overflowing || event.defaultPrevented ||
        isNodeName(activeElement, "embed") ||
        (isNodeName(target, "embed") && /\.pdf/i.test(target.src))) {
            return true;
        }

        var deltaX = event.wheelDeltaX || 0;
        var deltaY = event.wheelDeltaY || 0;

        // use wheelDelta if deltaX/Y is not available
        if (!deltaX && !deltaY) {
            deltaY = event.wheelDelta || 0;
        }

        // check if it's a touchpad scroll that should be ignored
        if (!options.touchpadSupport && isTouchpad(deltaY)) {
            return true;
        }

        // scale by step size
        // delta is 120 most of the time
        // synaptics seems to send 1 sometimes
        if (Math.abs(deltaX) > 1.2) {
            deltaX *= options.stepSize / 120;
        }
        if (Math.abs(deltaY) > 1.2) {
            deltaY *= options.stepSize / 120;
        }

        scrollArray(overflowing, - deltaX, - deltaY);
        event.preventDefault();
    }

    /**
         * Keydown event handler.
         * @param {Object} event
         */
    function keydown(event) {

        var target = event.target;
        var modifier = event.ctrlKey || event.altKey || event.metaKey ||
        (event.shiftKey && event.keyCode !== key.spacebar);

        // do nothing if user is editing text
        // or using a modifier key (except shift)
        // or in a dropdown
        if ( /input|textarea|select|embed/i.test(target.nodeName) ||
        target.isContentEditable ||
        event.defaultPrevented ||
        modifier ) {
            return true;
        }
        // spacebar should trigger button press
        if (isNodeName(target, "button") &&
        event.keyCode === key.spacebar) {
            return true;
        }

        var shift, x = 0, y = 0;
        var elem = overflowingAncestor(activeElement);
        var clientHeight = elem.clientHeight;

        if (elem == document.body) {
            clientHeight = window.innerHeight;
        }

        switch (event.keyCode) {
        case key.up:
            y = - options.arrowScroll;
            break;
        case key.down:
            y = options.arrowScroll;
            break;
        case key.spacebar:
            // (+ shift)
            shift = event.shiftKey ? 1 : - 1;
            y = - shift * clientHeight * 0.9;
            break;
        case key.pageup:
            y = - clientHeight * 0.9;
            break;
        case key.pagedown:
            y = clientHeight * 0.9;
            break;
        case key.home:
            y = - elem.scrollTop;
            break;
        case key.end:
            var damt = elem.scrollHeight - elem.scrollTop - clientHeight;
            y = (damt > 0) ? damt + 10 : 0;
            break;
        case key.left:
            x = - options.arrowScroll;
            break;
        case key.right:
            x = options.arrowScroll;
            break;
        default:
            return true; // a key we don't care about
        }

        scrollArray(elem, x, y);
        event.preventDefault();
    }

    /**
         * Mousedown event only for updating activeElement
         */
    function mousedown(event) {
        activeElement = event.target;
    }


    /***********************************************
         * OVERFLOW
         ***********************************************/

    var cache = {}; // cleared out every once in while
    setInterval(function () {
        cache = {};
    }, 10 * 1000);

    var uniqueID = (function () {
        var i = 0;
        return function (el) {
            return el.uniqueID || (el.uniqueID = i++);
        };
    })();

    function setCache(elems, overflowing) {
        for (var i = elems.length; i--;)
            cache[uniqueID(elems[i])] = overflowing;
        return overflowing;
    }

    function overflowingAncestor(el) {
        var elems = [];
        var rootScrollHeight = root.scrollHeight;
        do {
            var cached = cache[uniqueID(el)];
            if (cached) {
                return setCache(elems, cached);
            }
            elems.push(el);
            if (rootScrollHeight === el.scrollHeight) {
                if (!isFrame || root.clientHeight + 10 < rootScrollHeight) {
                    return setCache(elems, document.body); // scrolling root in WebKit
                }
            } else if (el.clientHeight + 10 < el.scrollHeight) {
                overflow = getComputedStyle(el, "").getPropertyValue("overflow-y");
                if (overflow === "scroll" || overflow === "auto") {
                    return setCache(elems, el);
                }
            }
        }
        while (el = el.parentNode);
    }


    /***********************************************
         * HELPERS
         ***********************************************/

    function addEvent(type, fn, bubble) {
        window.addEventListener(type, fn, (bubble || false));
    }

    function removeEvent(type, fn, bubble) {
        window.removeEventListener(type, fn, (bubble || false));
    }

    function isNodeName(el, tag) {
        return (el.nodeName || "").toLowerCase() === tag.toLowerCase();
    }

    function directionCheck(x, y) {
        x = (x > 0) ? 1 : - 1;
        y = (y > 0) ? 1 : - 1;
        if (direction.x !== x || direction.y !== y) {
            direction.x = x;
            direction.y = y;
            que = [];
            lastScroll = 0;
        }
    }

    var deltaBufferTimer;

    function isTouchpad(deltaY) {
        if (!deltaY) 
            return;
        deltaY = Math.abs(deltaY)
        deltaBuffer.push(deltaY);
        deltaBuffer.shift();
        clearTimeout(deltaBufferTimer);
        deltaBufferTimer = setTimeout(function () {
            chrome.storage.local.set({
                deltaBuffer: deltaBuffer 
            });
        }, 1000);
        var allEquals = (deltaBuffer[0] == deltaBuffer[1] &&
        deltaBuffer[1] == deltaBuffer[2]);
        var allDivisable = (isDivisible(deltaBuffer[0], 120) &&
        isDivisible(deltaBuffer[1], 120) &&
        isDivisible(deltaBuffer[2], 120));
        return !(allEquals || allDivisable);
    }

    function isDivisible(n, divisor) {
        return (Math.floor(n / divisor) == n / divisor);
    }

    if (typeof(chrome) !== 'undefined' && typeof(chrome.storage) !== 'undefined') {
        chrome.storage.local.get('deltaBuffer', function (stored) {
            if (stored.deltaBuffer) {
                deltaBuffer = stored.deltaBuffer;
            }
        });
    }

    var requestFrame = (function () {
        return window.requestAnimationFrame ||
        window.webkitRequestAnimationFrame ||
        function (callback, element, delay) {
            window.setTimeout(callback, delay || (1000 / 60));
        };
    })();

    var MutationObserver = window.MutationObserver || window.WebKitMutationObserver;


    /***********************************************
         * PULSE
         ***********************************************/

    /**
         * Viscous fluid with a pulse for part and decay for the rest.
         * - Applies a fixed force over an interval (a damped acceleration), and
         * - Lets the exponential bleed away the velocity over a longer interval
         * - Michael Herf, http://stereopsis.com/stopping/
         */
    function pulse_(x) {
        var val, start, expx;
        // test
        x = x * options.pulseScale;
        if (x < 1) {
            // acceleartion
            val = x - (1 - Math.exp( - x));
        } else {
            // tail
            // the previous animation ended here:
            start = Math.exp( - 1);
            // simple viscous drag
            x -= 1;
            expx = 1 - Math.exp( - x);
            val = start + (expx * (1 - start));
        }
        return val * options.pulseNormalize;
    }

    function pulse(x) {
        if (x >= 1) 
            return 1;
        if (x <= 0) 
            return 0;

        if (options.pulseNormalize == 1) {
            options.pulseNormalize /= pulse_(1);
        }
        return pulse_(x);
    }

    addEvent("mousedown", mousedown);
    addEvent("mousewheel", wheel);
    addEvent("load", init);
}
//end smooth scroll


