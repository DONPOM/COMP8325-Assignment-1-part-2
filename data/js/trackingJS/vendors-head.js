Array.prototype.indexOf || (Array.prototype.indexOf = function(a) {
    "use strict";
    if (null == this)
        throw new TypeError;
    var b = Object(this), c = b.length>>>0;
    if (0 === c)
        return - 1;
    var d = 0;
    if (arguments.length > 1 && (d = Number(arguments[1]), d != d ? d = 0 : 0 != d && 1 / 0 != d && d!=-1 / 0 && (d = (d > 0||-1) * Math.floor(Math.abs(d)))), d >= c)
        return - 1;
    for (var e = d >= 0 ? d : Math.max(c - Math.abs(d), 0); c > e; e++)
        if (e in b && b[e] === a)
            return e;
    return - 1
}), Array.prototype.forEach || (Array.prototype.forEach = function(a, b) {
    for (var c = 0, d = this.length; d > c; ++c)
        a.call(b, this[c], c, this)
}), Date.prototype.toISOString ||!function() {
    function a(a) {
        var b = String(a);
        return 1 === b.length && (b = "0" + b), b
    }
    Date.prototype.toISOString = function() {
        return this.getUTCFullYear() + "-" + a(this.getUTCMonth() + 1) + "-" + a(this.getUTCDate()) + "T" + a(this.getUTCHours()) + ":" + a(this.getUTCMinutes()) + ":" + a(this.getUTCSeconds()) + "." + String((this.getUTCMilliseconds() / 1e3).toFixed(3)).slice(2, 5) + "Z"
    }
}(), Date.now || (Date.now = function() {
    return + new Date
}), window.console || (window.console = {}), window.console&&!window.console.log && (window.console.log = function() {}), window.Modernizr = function(a, b, c) {
    function d(a) {
        t.cssText = a
    }
    function e(a, b) {
        return d(w.join(a + ";") + (b || ""))
    }
    function f(a, b) {
        return typeof a === b
    }
    function g(a, b) {
        return !!~("" + a).indexOf(b)
    }
    function h(a, b) {
        for (var d in a) {
            var e = a[d];
            if (!g(e, "-") && t[e] !== c)
                return "pfx" == b ? e : !0
        }
        return !1
    }
    function i(a, b, d) {
        for (var e in a) {
            var g = b[a[e]];
            if (g !== c)
                return d===!1 ? a[e] : f(g, "function") ? g.bind(d || b) : g
        }
        return !1
    }
    function j(a, b, c) {
        var d = a.charAt(0).toUpperCase() + a.slice(1), e = (a + " " + y.join(d + " ") + d).split(" ");
        return f(b, "string") || f(b, "undefined") ? h(e, b) : (e = (a + " " + z.join(d + " ") + d).split(" "), i(e, b, c))
    }
    function k() {
        o.input = function(c) {
            for (var d = 0, e = c.length; e > d; d++)
                C[c[d]] = c[d]in u;
            return C.list && (C.list=!!b.createElement("datalist")&&!!a.HTMLDataListElement), C
        }("autocomplete autofocus list placeholder max min multiple pattern required step".split(" ")), o.inputtypes = function(a) {
            for (var d, e, f, g = 0, h = a.length; h > g; g++)
                u.setAttribute("type", e = a[g]), d = "text" !== u.type, d && (u.value = v, u.style.cssText = "position:absolute;visibility:hidden;", /^range$/.test(e) && u.style.WebkitAppearance !== c ? (q.appendChild(u), f = b.defaultView, d = f.getComputedStyle && "textfield" !== f.getComputedStyle(u, null).WebkitAppearance && 0 !== u.offsetHeight, q.removeChild(u)) : /^(search|tel)$/.test(e) || (d = /^(url|email)$/.test(e) ? u.checkValidity && u.checkValidity()===!1 : u.value != v)), B[a[g]]=!!d;
            return B
        }("search tel url email datetime date month week time datetime-local number range color".split(" "))
    }
    var l, m, n = "2.8.2", o = {}, p=!0, q = b.documentElement, r = "modernizr", s = b.createElement(r), t = s.style, u = b.createElement("input"), v = ":)", w = ({}.toString, " -webkit- -moz- -o- -ms- ".split(" ")), x = "Webkit Moz O ms", y = x.split(" "), z = x.toLowerCase().split(" "), A = {}, B = {}, C = {}, D = [], E = D.slice, F = function(a, c, d, e) {
        var f, g, h, i, j = b.createElement("div"), k = b.body, l = k || b.createElement("body");
        if (parseInt(d, 10))
            for (; d--;)
                h = b.createElement("div"), h.id = e ? e[d] : r + (d + 1), j.appendChild(h);
        return f = ["&#173;", '<style id="s', r, '">', a, "</style>"].join(""), j.id = r, (k ? j : l).innerHTML += f, l.appendChild(j), k || (l.style.background = "", l.style.overflow = "hidden", i = q.style.overflow, q.style.overflow = "hidden", q.appendChild(l)), g = c(j, a), k ? j.parentNode.removeChild(j) : (l.parentNode.removeChild(l), q.style.overflow = i), !!g
    }, G = {}.hasOwnProperty;
    m = f(G, "undefined") || f(G.call, "undefined") ? function(a, b) {
        return b in a && f(a.constructor.prototype[b], "undefined")
    } : function(a, b) {
        return G.call(a, b)
    }, Function.prototype.bind || (Function.prototype.bind = function(a) {
        var b = this;
        if ("function" != typeof b)
            throw new TypeError;
        var c = E.call(arguments, 1), d = function() {
            if (this instanceof d) {
                var e = function() {};
                e.prototype = b.prototype;
                var f = new e, g = b.apply(f, c.concat(E.call(arguments)));
                return Object(g) === g ? g : f
            }
            return b.apply(a, c.concat(E.call(arguments)))
        };
        return d
    }), A.flexbox = function() {
        return j("flexWrap")
    }, A.canvas = function() {
        var a = b.createElement("canvas");
        return !!a.getContext&&!!a.getContext("2d")
    }, A.touch = function() {
        var c;
        return "ontouchstart"in a || a.DocumentTouch && b instanceof DocumentTouch ? c=!0 : F(["@media (", w.join("touch-enabled),("), r, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(a) {
            c = 9 === a.offsetTop
        }), c
    }, A.history = function() {
        return !!a.history&&!!history.pushState
    }, A.borderradius = function() {
        return j("borderRadius")
    }, A.opacity = function() {
        return e("opacity:.55"), /^0.55$/.test(t.opacity)
    }, A.cssanimations = function() {
        return j("animationName")
    }, A.cssgradients = function() {
        var a = "background-image:", b = "gradient(linear,left top,right bottom,from(#9f9),to(white));", c = "linear-gradient(left top,#9f9, white);";
        return d((a + "-webkit- ".split(" ").join(b + a) + w.join(c + a)).slice(0, - a.length)), g(t.backgroundImage, "gradient")
    }, A.csstransforms = function() {
        return !!j("transform")
    }, A.csstransforms3d = function() {
        var a=!!j("perspective");
        return a && "webkitPerspective"in q.style && F("@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}", function(b) {
            a = 9 === b.offsetLeft && 3 === b.offsetHeight
        }), a
    }, A.csstransitions = function() {
        return j("transition")
    }, A.fontface = function() {
        var a;
        return F('@font-face {font-family:"font";src:url("https://")}', function(c, d) {
            var e = b.getElementById("smodernizr"), f = e.sheet || e.styleSheet, g = f ? f.cssRules && f.cssRules[0] ? f.cssRules[0].cssText: f.cssText || "": "";
            a = /src/i.test(g) && 0 === g.indexOf(d.split(" ")[0])
        }), a
    }, A.generatedcontent = function() {
        var a;
        return F(["#", r, "{font:0/0 a}#", r, ':after{content:"', v, '";visibility:hidden;font:3px/1 a}'].join(""), function(b) {
            a = b.offsetHeight >= 3
        }), a
    }, A.localstorage = function() {
        try {
            return localStorage.setItem(r, r), localStorage.removeItem(r), !0
        } catch (a) {
            return !1
        }
    }, A.sessionstorage = function() {
        try {
            return sessionStorage.setItem(r, r), sessionStorage.removeItem(r), !0
        } catch (a) {
            return !1
        }
    };
    for (var H in A)
        m(A, H) && (l = H.toLowerCase(), o[l] = A[H](), D.push((o[l] ? "" : "no-") + l));
    return o.input || k(), o.addTest = function(a, b) {
        if ("object" == typeof a)
            for (var d in a)
                m(a, d) && o.addTest(d, a[d]);
        else {
            if (a = a.toLowerCase(), o[a] !== c)
                return o;
            b = "function" == typeof b ? b() : b, "undefined" != typeof p && p && (q.className += " " + (b ? "" : "no-") + a), o[a] = b
        }
        return o
    }, d(""), s = u = null, function(a, b) {
        function c(a, b) {
            var c = a.createElement("p"), d = a.getElementsByTagName("head")[0] || a.documentElement;
            return c.innerHTML = "x<style>" + b + "</style>", d.insertBefore(c.lastChild, d.firstChild)
        }
        function d() {
            var a = s.elements;
            return "string" == typeof a ? a.split(" ") : a
        }
        function e(a) {
            var b = r[a[p]];
            return b || (b = {}, q++, a[p] = q, r[q] = b), b
        }
        function f(a, c, d) {
            if (c || (c = b), k)
                return c.createElement(a);
            d || (d = e(c));
            var f;
            return f = d.cache[a] ? d.cache[a].cloneNode() : o.test(a) ? (d.cache[a] = d.createElem(a)).cloneNode() : d.createElem(a), !f.canHaveChildren || n.test(a) || f.tagUrn ? f : d.frag.appendChild(f)
        }
        function g(a, c) {
            if (a || (a = b), k)
                return a.createDocumentFragment();
            c = c || e(a);
            for (var f = c.frag.cloneNode(), g = 0, h = d(), i = h.length; i > g; g++)
                f.createElement(h[g]);
            return f
        }
        function h(a, b) {
            b.cache || (b.cache = {}, b.createElem = a.createElement, b.createFrag = a.createDocumentFragment, b.frag = b.createFrag()), a.createElement = function(c) {
                return s.shivMethods ? f(c, a, b) : b.createElem(c)
            }, a.createDocumentFragment = Function("h,f", "return function(){var n=f.cloneNode(),c=n.createElement;h.shivMethods&&(" + d().join().replace(/[\w\-]+/g, function(a) {
                return b.createElem(a), b.frag.createElement(a), 'c("' + a + '")'
            }) + ");return n}")(s, b.frag)
        }
        function i(a) {
            a || (a = b);
            var d = e(a);
            return s.shivCSS&&!j&&!d.hasCSS && (d.hasCSS=!!c(a, "article,aside,dialog,figcaption,figure,footer,header,hgroup,main,nav,section{display:block}mark{background:#FF0;color:#000}template{display:none}")), k || h(a, d), a
        }
        var j, k, l = "3.7.0", m = a.html5 || {}, n = /^<|^(?:button|map|select|textarea|object|iframe|option|optgroup)$/i, o = /^(?:a|b|code|div|fieldset|h1|h2|h3|h4|h5|h6|i|label|li|ol|p|q|span|strong|style|table|tbody|td|th|tr|ul)$/i, p = "_html5shiv", q = 0, r = {};
        !function() {
            try {
                var a = b.createElement("a");
                a.innerHTML = "<xyz></xyz>", j = "hidden"in a, k = 1 == a.childNodes.length || function() {
                    b.createElement("a");
                    var a = b.createDocumentFragment();
                    return "undefined" == typeof a.cloneNode || "undefined" == typeof a.createDocumentFragment || "undefined" == typeof a.createElement
                }()
            } catch (c) {
                j=!0, k=!0
            }
        }();
        var s = {
            elements: m.elements || "abbr article aside audio bdi canvas data datalist details dialog figcaption figure footer header hgroup main mark meter nav output progress section summary template time video",
            version: l,
            shivCSS: m.shivCSS!==!1,
            supportsUnknownElements: k,
            shivMethods: m.shivMethods!==!1,
            type: "default",
            shivDocument: i,
            createElement: f,
            createDocumentFragment: g
        };
        a.html5 = s, i(b)
    }(this, b), o._version = n, o._prefixes = w, o._domPrefixes = z, o._cssomPrefixes = y, o.testProp = function(a) {
        return h([a])
    }, o.testAllProps = j, o.testStyles = F, o.prefixed = function(a, b, c) {
        return b ? j(a, b, c) : j(a, "pfx")
    }, q.className = q.className.replace(/(^|\s)no-js(\s|$)/, "$1$2") + (p ? " js " + D.join(" ") : ""), o
}(this, this.document), Modernizr.addTest("display-table", function() {
    var a, b = window.document, c = b.documentElement, d = b.createElement("div"), e = b.createElement("div"), f = b.createElement("div");
    return d.style.cssText = "display: table", e.style.cssText = f.style.cssText = "display: table-cell; padding: 10px", d.appendChild(e), d.appendChild(f), c.insertBefore(d, c.firstChild), a = e.offsetLeft < f.offsetLeft, c.removeChild(d), a
}), Modernizr.addTest("csschecked", function() {
    return Modernizr.testStyles("#modernizr input {margin-left:0px;} #modernizr input:checked {margin-left: 20px;}", function(a) {
        var b = document.createElement("input");
        return b.type = "checkbox", b.checked = "checked", a.appendChild(b), a.lastChild.offsetLeft >= 20
    })
});
