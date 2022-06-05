// dom-homepage 9.0.0
!function(a, b) {
    function c(a, b) {
        "use strict";
        function e(a, b) {
            return function() {
                return a.apply(b, arguments)
            }
        }
        var f;
        if (b = b || {}, this.trackingClick=!1, this.trackingClickStart = 0, this.targetElement = null, this.touchStartX = 0, this.touchStartY = 0, this.lastTouchIdentifier = 0, this.touchBoundary = b.touchBoundary || 10, this.layer = a, this.tapDelay = b.tapDelay || 200, !c.notNeeded(a)) {
            for (var g = ["onMouse", "onClick", "onTouchStart", "onTouchMove", "onTouchEnd", "onTouchCancel"], h = this, i = 0, j = g.length; j > i; i++)
                h[g[i]] = e(h[g[i]], h);
            d && (a.addEventListener("mouseover", this.onMouse, !0), a.addEventListener("mousedown", this.onMouse, !0), a.addEventListener("mouseup", this.onMouse, !0)), a.addEventListener("click", this.onClick, !0), a.addEventListener("touchstart", this.onTouchStart, !1), a.addEventListener("touchmove", this.onTouchMove, !1), a.addEventListener("touchend", this.onTouchEnd, !1), a.addEventListener("touchcancel", this.onTouchCancel, !1), Event.prototype.stopImmediatePropagation || (a.removeEventListener = function(b, c, d) {
                var e = Node.prototype.removeEventListener;
                "click" === b ? e.call(a, b, c.hijacked || c, d) : e.call(a, b, c, d)
            }, a.addEventListener = function(b, c, d) {
                var e = Node.prototype.addEventListener;
                "click" === b ? e.call(a, b, c.hijacked || (c.hijacked = function(a) {
                    a.propagationStopped || c(a)
                }), d) : e.call(a, b, c, d)
            }), "function" == typeof a.onclick && (f = a.onclick, a.addEventListener("click", function(a) {
                f(a)
            }, !1), a.onclick = null)
        }
    }
    b["dom-homepage"] = a, window.Modernizr = function(a, b, c) {
        function d(a) {
            t.cssText = a
        }
        function e(a, b) {
            return d(x.join(a + ";") + (b || ""))
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
            var d = a.charAt(0).toUpperCase() + a.slice(1), e = (a + " " + z.join(d + " ") + d).split(" ");
            return f(b, "string") || f(b, "undefined") ? h(e, b) : (e = (a + " " + A.join(d + " ") + d).split(" "), i(e, b, c))
        }
        function k() {
            o.input = function(c) {
                for (var d = 0, e = c.length; e > d; d++)
                    E[c[d]]=!!(c[d]in u);
                return E.list && (E.list=!(!b.createElement("datalist") ||!a.HTMLDataListElement)), E
            }("autocomplete autofocus list placeholder max min multiple pattern required step".split(" ")), o.inputtypes = function(a) {
                for (var d, e, f, g = 0, h = a.length; h > g; g++)
                    u.setAttribute("type", e = a[g]), d = "text" !== u.type, d && (u.value = v, u.style.cssText = "position:absolute;visibility:hidden;", /^range$/.test(e) && u.style.WebkitAppearance !== c ? (q.appendChild(u), f = b.defaultView, d = f.getComputedStyle && "textfield" !== f.getComputedStyle(u, null).WebkitAppearance && 0 !== u.offsetHeight, q.removeChild(u)) : /^(search|tel)$/.test(e) || (d = /^(url|email)$/.test(e) ? u.checkValidity && u.checkValidity()===!1 : u.value != v)), D[a[g]]=!!d;
                return D
            }("search tel url email datetime date month week time datetime-local number range color".split(" "))
        }
        var l, m, n = "2.8.2", o = {}, p=!0, q = b.documentElement, r = "modernizr", s = b.createElement(r), t = s.style, u = b.createElement("input"), v = ":)", w = {}.toString, x = " -webkit- -moz- -o- -ms- ".split(" "), y = "Webkit Moz O ms", z = y.split(" "), A = y.toLowerCase().split(" "), B = {
            svg: "http://www.w3.org/2000/svg"
        }, C = {}, D = {}, E = {}, F = [], G = F.slice, H = function(a, c, d, e) {
            var f, g, h, i, j = b.createElement("div"), k = b.body, l = k || b.createElement("body");
            if (parseInt(d, 10))
                for (; d--;)
                    h = b.createElement("div"), h.id = e ? e[d] : r + (d + 1), j.appendChild(h);
            return f = ["&#173;", '<style id="s', r, '">', a, "</style>"].join(""), j.id = r, (k ? j : l).innerHTML += f, l.appendChild(j), k || (l.style.background = "", l.style.overflow = "hidden", i = q.style.overflow, q.style.overflow = "hidden", q.appendChild(l)), g = c(j, a), k ? j.parentNode.removeChild(j) : (l.parentNode.removeChild(l), q.style.overflow = i), !!g
        }, I = function(b) {
            var c = a.matchMedia || a.msMatchMedia;
            if (c)
                return c(b) && c(b).matches ||!1;
            var d;
            return H("@media " + b + " { #" + r + " { position: absolute; } }", function(b) {
                d = "absolute" == (a.getComputedStyle ? getComputedStyle(b, null) : b.currentStyle).position
            }), d
        }, J = function() {
            function a(a, e) {
                e = e || b.createElement(d[a] || "div"), a = "on" + a;
                var g = a in e;
                return g || (e.setAttribute || (e = b.createElement("div")), e.setAttribute && e.removeAttribute && (e.setAttribute(a, ""), g = f(e[a], "function"), f(e[a], "undefined") || (e[a] = c), e.removeAttribute(a))), e = null, g
            }
            var d = {
                select: "input",
                change: "input",
                submit: "form",
                reset: "form",
                error: "img",
                load: "img",
                abort: "img"
            };
            return a
        }(), K = {}.hasOwnProperty;
        m = f(K, "undefined") || f(K.call, "undefined") ? function(a, b) {
            return b in a && f(a.constructor.prototype[b], "undefined")
        } : function(a, b) {
            return K.call(a, b)
        }, Function.prototype.bind || (Function.prototype.bind = function(a) {
            var b = this;
            if ("function" != typeof b)
                throw new TypeError;
            var c = G.call(arguments, 1), d = function() {
                if (this instanceof d) {
                    var e = function() {};
                    e.prototype = b.prototype;
                    var f = new e, g = b.apply(f, c.concat(G.call(arguments)));
                    return Object(g) === g ? g : f
                }
                return b.apply(a, c.concat(G.call(arguments)))
            };
            return d
        }), C.flexbox = function() {
            return j("flexWrap")
        }, C.flexboxlegacy = function() {
            return j("boxDirection")
        }, C.canvas = function() {
            var a = b.createElement("canvas");
            return !(!a.getContext ||!a.getContext("2d"))
        }, C.canvastext = function() {
            return !(!o.canvas ||!f(b.createElement("canvas").getContext("2d").fillText, "function"))
        }, C.webgl = function() {
            return !!a.WebGLRenderingContext
        }, C.touch = function() {
            var c;
            return "ontouchstart"in a || a.DocumentTouch && b instanceof DocumentTouch ? c=!0 : H(["@media (", x.join("touch-enabled),("), r, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(a) {
                c = 9 === a.offsetTop
            }), c
        }, C.geolocation = function() {
            return "geolocation"in navigator
        }, C.postmessage = function() {
            return !!a.postMessage
        }, C.websqldatabase = function() {
            return !!a.openDatabase
        }, C.indexedDB = function() {
            return !!j("indexedDB", a)
        }, C.hashchange = function() {
            return J("hashchange", a) && (b.documentMode === c || b.documentMode > 7)
        }, C.history = function() {
            return !(!a.history ||!history.pushState)
        }, C.draganddrop = function() {
            var a = b.createElement("div");
            return "draggable"in a || "ondragstart"in a && "ondrop"in a
        }, C.websockets = function() {
            return "WebSocket"in a || "MozWebSocket"in a
        }, C.rgba = function() {
            return d("background-color:rgba(150,255,150,.5)"), g(t.backgroundColor, "rgba")
        }, C.hsla = function() {
            return d("background-color:hsla(120,40%,100%,.5)"), g(t.backgroundColor, "rgba") || g(t.backgroundColor, "hsla")
        }, C.multiplebgs = function() {
            return d("background:url(https://),url(https://),red url(https://)"), /(url\s*\(.*?){3}/.test(t.background)
        }, C.backgroundsize = function() {
            return j("backgroundSize")
        }, C.borderimage = function() {
            return j("borderImage")
        }, C.borderradius = function() {
            return j("borderRadius")
        }, C.boxshadow = function() {
            return j("boxShadow")
        }, C.textshadow = function() {
            return "" === b.createElement("div").style.textShadow
        }, C.opacity = function() {
            return e("opacity:.55"), /^0.55$/.test(t.opacity)
        }, C.cssanimations = function() {
            return j("animationName")
        }, C.csscolumns = function() {
            return j("columnCount")
        }, C.cssgradients = function() {
            var a = "background-image:", b = "gradient(linear,left top,right bottom,from(#9f9),to(white));", c = "linear-gradient(left top,#9f9, white);";
            return d((a + "-webkit- ".split(" ").join(b + a) + x.join(c + a)).slice(0, - a.length)), g(t.backgroundImage, "gradient")
        }, C.cssreflections = function() {
            return j("boxReflect")
        }, C.csstransforms = function() {
            return !!j("transform")
        }, C.csstransforms3d = function() {
            var a=!!j("perspective");
            return a && "webkitPerspective"in q.style && H("@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}", function(b) {
                a = 9 === b.offsetLeft && 3 === b.offsetHeight
            }), a
        }, C.csstransitions = function() {
            return j("transition")
        }, C.fontface = function() {
            var a;
            return H('@font-face {font-family:"font";src:url("https://")}', function(c, d) {
                var e = b.getElementById("smodernizr"), f = e.sheet || e.styleSheet, g = f ? f.cssRules && f.cssRules[0] ? f.cssRules[0].cssText: f.cssText || "": "";
                a = /src/i.test(g) && 0 === g.indexOf(d.split(" ")[0])
            }), a
        }, C.generatedcontent = function() {
            var a;
            return H(["#", r, "{font:0/0 a}#", r, ':after{content:"', v, '";visibility:hidden;font:3px/1 a}'].join(""), function(b) {
                a = b.offsetHeight >= 3
            }), a
        }, C.video = function() {
            var a = b.createElement("video"), c=!1;
            try {
                (c=!!a.canPlayType) && (c = new Boolean(c), c.ogg = a.canPlayType('video/ogg; codecs="theora"').replace(/^no$/, ""), c.h264 = a.canPlayType('video/mp4; codecs="avc1.42E01E"').replace(/^no$/, ""), c.webm = a.canPlayType('video/webm; codecs="vp8, vorbis"').replace(/^no$/, ""))
            } catch (d) {}
            return c
        }, C.audio = function() {
            var a = b.createElement("audio"), c=!1;
            try {
                (c=!!a.canPlayType) && (c = new Boolean(c), c.ogg = a.canPlayType('audio/ogg; codecs="vorbis"').replace(/^no$/, ""), c.mp3 = a.canPlayType("audio/mpeg;").replace(/^no$/, ""), c.wav = a.canPlayType('audio/wav; codecs="1"').replace(/^no$/, ""), c.m4a = (a.canPlayType("audio/x-m4a;") || a.canPlayType("audio/aac;")).replace(/^no$/, ""))
            } catch (d) {}
            return c
        }, C.localstorage = function() {
            try {
                return localStorage.setItem(r, r), localStorage.removeItem(r), !0
            } catch (a) {
                return !1
            }
        }, C.sessionstorage = function() {
            try {
                return sessionStorage.setItem(r, r), sessionStorage.removeItem(r), !0
            } catch (a) {
                return !1
            }
        }, C.webworkers = function() {
            return !!a.Worker
        }, C.applicationcache = function() {
            return !!a.applicationCache
        }, C.svg = function() {
            return !!b.createElementNS&&!!b.createElementNS(B.svg, "svg").createSVGRect
        }, C.inlinesvg = function() {
            var a = b.createElement("div");
            return a.innerHTML = "<svg/>", (a.firstChild && a.firstChild.namespaceURI) == B.svg
        }, C.smil = function() {
            return !!b.createElementNS && /SVGAnimate/.test(w.call(b.createElementNS(B.svg, "animate")))
        }, C.svgclippaths = function() {
            return !!b.createElementNS && /SVGClipPath/.test(w.call(b.createElementNS(B.svg, "clipPath")))
        };
        for (var L in C)
            m(C, L) && (l = L.toLowerCase(), o[l] = C[L](), F.push((o[l] ? "" : "no-") + l));
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
                return !s.shivCSS || j || d.hasCSS || (d.hasCSS=!!c(a, "article,aside,dialog,figcaption,figure,footer,header,hgroup,main,nav,section{display:block}mark{background:#FF0;color:#000}template{display:none}")), k || h(a, d), a
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
        }(this, b), o._version = n, o._prefixes = x, o._domPrefixes = A, o._cssomPrefixes = z, o.mq = I, o.hasEvent = J, o.testProp = function(a) {
            return h([a])
        }, o.testAllProps = j, o.testStyles = H, o.prefixed = function(a, b, c) {
            return b ? j(a, b, c) : j(a, "pfx")
        }, q.className = q.className.replace(/(^|\s)no-js(\s|$)/, "$1$2") + (p ? " js " + F.join(" ") : ""), o
    }(this, this.document), function(a, b) {
        "object" == typeof module && "object" == typeof module.exports ? module.exports = a.document ? b(a, !0) : function(a) {
            if (!a.document)
                throw new Error("jQuery requires a window with a document");
            return b(a)
        } : b(a)
    }("undefined" != typeof window ? window : this, function(a, b) {
        function c(a) {
            var b = a.length, c = _.type(a);
            return "function" === c || _.isWindow(a)?!1 : 1 === a.nodeType && b?!0 : "array" === c || 0 === b || "number" == typeof b && b > 0 && b - 1 in a
        }
        function d(a, b, c) {
            if (_.isFunction(b))
                return _.grep(a, function(a, d) {
                    return !!b.call(a, d, a) !== c
                });
            if (b.nodeType)
                return _.grep(a, function(a) {
                    return a === b !== c
                });
            if ("string" == typeof b) {
                if (hb.test(b))
                    return _.filter(b, a, c);
                b = _.filter(b, a)
            }
            return _.grep(a, function(a) {
                return U.call(b, a) >= 0 !== c
            })
        }
        function e(a, b) {
            for (; (a = a[b]) && 1 !== a.nodeType;);
            return a
        }
        function f(a) {
            var b = ob[a] = {};
            return _.each(a.match(nb) || [], function(a, c) {
                b[c]=!0
            }), b
        }
        function g() {
            Z.removeEventListener("DOMContentLoaded", g, !1), a.removeEventListener("load", g, !1), _.ready()
        }
        function h() {
            Object.defineProperty(this.cache = {}, 0, {
                get: function() {
                    return {}
                }
            }), this.expando = _.expando + Math.random()
        }
        function i(a, b, c) {
            var d;
            if (void 0 === c && 1 === a.nodeType)
                if (d = "data-" + b.replace(ub, "-$1").toLowerCase(), c = a.getAttribute(d), "string" == typeof c) {
                    try {
                        c = "true" === c?!0 : "false" === c?!1 : "null" === c ? null : + c + "" === c?+c : tb.test(c) ? _.parseJSON(c) : c
                    } catch (e) {}
                    sb.set(a, b, c)
                } else 
                    c = void 0;
            return c
        }
        function j() {
            return !0
        }
        function k() {
            return !1
        }
        function l() {
            try {
                return Z.activeElement
            } catch (a) {}
        }
        function m(a, b) {
            return _.nodeName(a, "table") && _.nodeName(11 !== b.nodeType ? b : b.firstChild, "tr") ? a.getElementsByTagName("tbody")[0] || a.appendChild(a.ownerDocument.createElement("tbody")) : a
        }
        function n(a) {
            return a.type = (null !== a.getAttribute("type")) + "/" + a.type, a
        }
        function o(a) {
            var b = Kb.exec(a.type);
            return b ? a.type = b[1] : a.removeAttribute("type"), a
        }
        function p(a, b) {
            for (var c = 0, d = a.length; d > c; c++)
                rb.set(a[c], "globalEval", !b || rb.get(b[c], "globalEval"))
        }
        function q(a, b) {
            var c, d, e, f, g, h, i, j;
            if (1 === b.nodeType) {
                if (rb.hasData(a) && (f = rb.access(a), g = rb.set(b, f), j = f.events)) {
                    delete g.handle, g.events = {};
                    for (e in j)
                        for (c = 0, d = j[e].length; d > c; c++)
                            _.event.add(b, e, j[e][c])
                        }
                sb.hasData(a) && (h = sb.access(a), i = _.extend({}, h), sb.set(b, i))
            }
        }
        function r(a, b) {
            var c = a.getElementsByTagName ? a.getElementsByTagName(b || "*"): a.querySelectorAll ? a.querySelectorAll(b || "*"): [];
            return void 0 === b || b && _.nodeName(a, b) ? _.merge([a], c) : c
        }
        function s(a, b) {
            var c = b.nodeName.toLowerCase();
            "input" === c && yb.test(a.type) ? b.checked = a.checked : ("input" === c || "textarea" === c) && (b.defaultValue = a.defaultValue)
        }
        function t(b, c) {
            var d, e = _(c.createElement(b)).appendTo(c.body), f = a.getDefaultComputedStyle && (d = a.getDefaultComputedStyle(e[0])) ? d.display: _.css(e[0], "display");
            return e.detach(), f
        }
        function u(a) {
            var b = Z, c = Ob[a];
            return c || (c = t(a, b), "none" !== c && c || (Nb = (Nb || _("<iframe frameborder='0' width='0' height='0'/>")).appendTo(b.documentElement), b = Nb[0].contentDocument, b.write(), b.close(), c = t(a, b), Nb.detach()), Ob[a] = c), c
        }
        function v(a, b, c) {
            var d, e, f, g, h = a.style;
            return c = c || Rb(a), c && (g = c.getPropertyValue(b) || c[b]), c && ("" !== g || _.contains(a.ownerDocument, a) || (g = _.style(a, b)), Qb.test(g) && Pb.test(b) && (d = h.width, e = h.minWidth, f = h.maxWidth, h.minWidth = h.maxWidth = h.width = g, g = c.width, h.width = d, h.minWidth = e, h.maxWidth = f)), void 0 !== g ? g + "" : g
        }
        function w(a, b) {
            return {
                get: function() {
                    return a() ? void delete this.get : (this.get = b).apply(this, arguments)
                }
            }
        }
        function x(a, b) {
            if (b in a)
                return b;
            for (var c = b[0].toUpperCase() + b.slice(1), d = b, e = Xb.length; e--;)
                if (b = Xb[e] + c, b in a)
                    return b;
            return d
        }
        function y(a, b, c) {
            var d = Tb.exec(b);
            return d ? Math.max(0, d[1] - (c || 0)) + (d[2] || "px") : b
        }
        function z(a, b, c, d, e) {
            for (var f = c === (d ? "border" : "content") ? 4 : "width" === b ? 1 : 0, g = 0; 4 > f; f += 2)
                "margin" === c && (g += _.css(a, c + wb[f], !0, e)), d ? ("content" === c && (g -= _.css(a, "padding" + wb[f], !0, e)), "margin" !== c && (g -= _.css(a, "border" + wb[f] + "Width", !0, e))) : (g += _.css(a, "padding" + wb[f], !0, e), "padding" !== c && (g += _.css(a, "border" + wb[f] + "Width", !0, e)));
            return g
        }
        function A(a, b, c) {
            var d=!0, e = "width" === b ? a.offsetWidth : a.offsetHeight, f = Rb(a), g = "border-box" === _.css(a, "boxSizing", !1, f);
            if (0 >= e || null == e) {
                if (e = v(a, b, f), (0 > e || null == e) && (e = a.style[b]), Qb.test(e))
                    return e;
                d = g && (Y.boxSizingReliable() || e === a.style[b]), e = parseFloat(e) || 0
            }
            return e + z(a, b, c || (g ? "border" : "content"), d, f) + "px"
        }
        function B(a, b) {
            for (var c, d, e, f = [], g = 0, h = a.length; h > g; g++)
                d = a[g], d.style && (f[g] = rb.get(d, "olddisplay"), c = d.style.display, b ? (f[g] || "none" !== c || (d.style.display = ""), "" === d.style.display && xb(d) && (f[g] = rb.access(d, "olddisplay", u(d.nodeName)))) : (e = xb(d), "none" === c && e || rb.set(d, "olddisplay", e ? c : _.css(d, "display"))));
            for (g = 0; h > g; g++)
                d = a[g], d.style && (b && "none" !== d.style.display && "" !== d.style.display || (d.style.display = b ? f[g] || "" : "none"));
            return a
        }
        function C(a, b, c, d, e) {
            return new C.prototype.init(a, b, c, d, e)
        }
        function D() {
            return setTimeout(function() {
                Yb = void 0
            }), Yb = _.now()
        }
        function E(a, b) {
            var c, d = 0, e = {
                height: a
            };
            for (b = b ? 1 : 0; 4 > d; d += 2 - b)
                c = wb[d], e["margin" + c] = e["padding" + c] = a;
            return b && (e.opacity = e.width = a), e
        }
        function F(a, b, c) {
            for (var d, e = (cc[b] || []).concat(cc["*"]), f = 0, g = e.length; g > f; f++)
                if (d = e[f].call(c, b, a))
                    return d
        }
        function G(a, b, c) {
            var d, e, f, g, h, i, j, k, l = this, m = {}, n = a.style, o = a.nodeType && xb(a), p = rb.get(a, "fxshow");
            c.queue || (h = _._queueHooks(a, "fx"), null == h.unqueued && (h.unqueued = 0, i = h.empty.fire, h.empty.fire = function() {
                h.unqueued || i()
            }), h.unqueued++, l.always(function() {
                l.always(function() {
                    h.unqueued--, _.queue(a, "fx").length || h.empty.fire()
                })
            })), 1 === a.nodeType && ("height"in b || "width"in b) && (c.overflow = [n.overflow, n.overflowX, n.overflowY], j = _.css(a, "display"), k = "none" === j ? rb.get(a, "olddisplay") || u(a.nodeName) : j, "inline" === k && "none" === _.css(a, "float") && (n.display = "inline-block")), c.overflow && (n.overflow = "hidden", l.always(function() {
                n.overflow = c.overflow[0], n.overflowX = c.overflow[1], n.overflowY = c.overflow[2]
            }));
            for (d in b)
                if (e = b[d], $b.exec(e)) {
                    if (delete b[d], f = f || "toggle" === e, e === (o ? "hide" : "show")) {
                        if ("show" !== e ||!p || void 0 === p[d])
                            continue;
                            o=!0
                    }
                    m[d] = p && p[d] || _.style(a, d)
                } else 
                    j = void 0;
            if (_.isEmptyObject(m))
                "inline" === ("none" === j ? u(a.nodeName) : j) && (n.display = j);
            else {
                p ? "hidden"in p && (o = p.hidden) : p = rb.access(a, "fxshow", {}), f && (p.hidden=!o), o ? _(a).show() : l.done(function() {
                    _(a).hide()
                }), l.done(function() {
                    var b;
                    rb.remove(a, "fxshow");
                    for (b in m)
                        _.style(a, b, m[b])
                });
                for (d in m)
                    g = F(o ? p[d] : 0, d, l), d in p || (p[d] = g.start, o && (g.end = g.start, g.start = "width" === d || "height" === d ? 1 : 0))
            }
        }
        function H(a, b) {
            var c, d, e, f, g;
            for (c in a)
                if (d = _.camelCase(c), e = b[d], f = a[c], _.isArray(f) && (e = f[1], f = a[c] = f[0]), c !== d && (a[d] = f, delete a[c]), g = _.cssHooks[d], g && "expand"in g) {
                    f = g.expand(f), delete a[d];
                    for (c in f)
                        c in a || (a[c] = f[c], b[c] = e)
                } else 
                    b[d] = e
        }
        function I(a, b, c) {
            var d, e, f = 0, g = bc.length, h = _.Deferred().always(function() {
                delete i.elem
            }), i = function() {
                if (e)
                    return !1;
                for (var b = Yb || D(), c = Math.max(0, j.startTime + j.duration - b), d = c / j.duration || 0, f = 1 - d, g = 0, i = j.tweens.length; i > g; g++)
                    j.tweens[g].run(f);
                return h.notifyWith(a, [j, f, c]), 1 > f && i ? c : (h.resolveWith(a, [j]), !1)
            }, j = h.promise({
                elem: a,
                props: _.extend({}, b),
                opts: _.extend(!0, {
                    specialEasing: {}
                }, c),
                originalProperties: b,
                originalOptions: c,
                startTime: Yb || D(),
                duration: c.duration,
                tweens: [],
                createTween: function(b, c) {
                    var d = _.Tween(a, j.opts, b, c, j.opts.specialEasing[b] || j.opts.easing);
                    return j.tweens.push(d), d
                },
                stop: function(b) {
                    var c = 0, d = b ? j.tweens.length: 0;
                    if (e)
                        return this;
                    for (e=!0; d > c; c++)
                        j.tweens[c].run(1);
                    return b ? h.resolveWith(a, [j, b]) : h.rejectWith(a, [j, b]), this
                }
            }), k = j.props;
            for (H(k, j.opts.specialEasing);
            g > f;
            f++)if (d = bc[f].call(j, a, k, j.opts))
                return d;
            return _.map(k, F, j), _.isFunction(j.opts.start) && j.opts.start.call(a, j), _.fx.timer(_.extend(i, {
                elem: a,
                anim: j,
                queue: j.opts.queue
            })), j.progress(j.opts.progress).done(j.opts.done, j.opts.complete).fail(j.opts.fail).always(j.opts.always)
        }
        function J(a) {
            return function(b, c) {
                "string" != typeof b && (c = b, b = "*");
                var d, e = 0, f = b.toLowerCase().match(nb) || [];
                if (_.isFunction(c))
                    for (; d = f[e++];)
                        "+" === d[0] ? (d = d.slice(1) || "*", (a[d] = a[d] || []).unshift(c)) : (a[d] = a[d] || []).push(c)
            }
        }
        function K(a, b, c, d) {
            function e(h) {
                var i;
                return f[h]=!0, _.each(a[h] || [], function(a, h) {
                    var j = h(b, c, d);
                    return "string" != typeof j || g || f[j] ? g?!(i = j) : void 0 : (b.dataTypes.unshift(j), e(j), !1)
                }), i
            }
            var f = {}, g = a === vc;
            return e(b.dataTypes[0]) ||!f["*"] && e("*")
        }
        function L(a, b) {
            var c, d, e = _.ajaxSettings.flatOptions || {};
            for (c in b)
                void 0 !== b[c] && ((e[c] ? a : d || (d = {}))[c] = b[c]);
            return d && _.extend(!0, a, d), a
        }
        function M(a, b, c) {
            for (var d, e, f, g, h = a.contents, i = a.dataTypes; "*" === i[0];)
                i.shift(), void 0 === d && (d = a.mimeType || b.getResponseHeader("Content-Type"));
            if (d)
                for (e in h)
                    if (h[e] && h[e].test(d)) {
                        i.unshift(e);
                        break
                    }
            if (i[0]in c)
                f = i[0];
            else {
                for (e in c) {
                    if (!i[0] || a.converters[e + " " + i[0]]) {
                        f = e;
                        break
                    }
                    g || (g = e)
                }
                f = f || g
            }
            return f ? (f !== i[0] && i.unshift(f), c[f]) : void 0
        }
        function N(a, b, c, d) {
            var e, f, g, h, i, j = {}, k = a.dataTypes.slice();
            if (k[1])
                for (g in a.converters)
                    j[g.toLowerCase()] = a.converters[g];
            for (f = k.shift(); f;)
                if (a.responseFields[f] && (c[a.responseFields[f]] = b), !i && d && a.dataFilter && (b = a.dataFilter(b, a.dataType)), i = f, f = k.shift())
                    if ("*" === f)
                        f = i;
                    else if ("*" !== i && i !== f) {
                        if (g = j[i + " " + f] || j["* " + f], !g)
                            for (e in j)
                                if (h = e.split(" "), h[1] === f && (g = j[i + " " + h[0]] || j["* " + h[0]])) {
                                    g===!0 ? g = j[e] : j[e]!==!0 && (f = h[0], k.unshift(h[1]));
                                    break
                                }
                                if (g!==!0)
                                    if (g && a["throws"])
                                        b = g(b);
                                    else 
                                        try {
                                            b = g(b)
                                        } catch (l) {
                                            return {
                                                state: "parsererror",
                                                error: g ? l: "No conversion from " + i + " to " + f
                                            }
                                        }
                                    }
            return {
                state: "success",
                data: b
            }
        }
        function O(a, b, c, d) {
            var e;
            if (_.isArray(b))
                _.each(b, function(b, e) {
                    c || zc.test(a) ? d(a, e) : O(a + "[" + ("object" == typeof e ? b : "") + "]", e, c, d)
                });
            else if (c || "object" !== _.type(b))
                d(a, b);
            else 
                for (e in b)
                    O(a + "[" + e + "]", b[e], c, d)
                }
        function P(a) {
            return _.isWindow(a) ? a : 9 === a.nodeType && a.defaultView
        }
        var Q = [], R = Q.slice, S = Q.concat, T = Q.push, U = Q.indexOf, V = {}, W = V.toString, X = V.hasOwnProperty, Y = {}, Z = a.document, $ = "2.1.1", _ = function(a, b) {
            return new _.fn.init(a, b)
        }, ab = /^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, bb = /^-ms-/, cb = /-([\da-z])/gi, db = function(a, b) {
            return b.toUpperCase()
        };
        _.fn = _.prototype = {
            jquery: $,
            constructor: _,
            selector: "",
            length: 0,
            toArray: function() {
                return R.call(this)
            },
            get: function(a) {
                return null != a ? 0 > a ? this[a + this.length] : this[a] : R.call(this)
            },
            pushStack: function(a) {
                var b = _.merge(this.constructor(), a);
                return b.prevObject = this, b.context = this.context, b
            },
            each: function(a, b) {
                return _.each(this, a, b)
            },
            map: function(a) {
                return this.pushStack(_.map(this, function(b, c) {
                    return a.call(b, c, b)
                }))
            },
            slice: function() {
                return this.pushStack(R.apply(this, arguments))
            },
            first: function() {
                return this.eq(0)
            },
            last: function() {
                return this.eq( - 1)
            },
            eq: function(a) {
                var b = this.length, c =+ a + (0 > a ? b : 0);
                return this.pushStack(c >= 0 && b > c ? [this[c]] : [])
            },
            end: function() {
                return this.prevObject || this.constructor(null)
            },
            push: T,
            sort: Q.sort,
            splice: Q.splice
        }, _.extend = _.fn.extend = function() {
            var a, b, c, d, e, f, g = arguments[0] || {}, h = 1, i = arguments.length, j=!1;
            for ("boolean" == typeof g && (j = g, g = arguments[h] || {}, h++), "object" == typeof g || _.isFunction(g) || (g = {}), h === i && (g = this, h--); i > h; h++)
                if (null != (a = arguments[h]))
                    for (b in a)
                        c = g[b], d = a[b], g !== d && (j && d && (_.isPlainObject(d) || (e = _.isArray(d))) ? (e ? (e=!1, f = c && _.isArray(c) ? c : []) : f = c && _.isPlainObject(c) ? c : {}, g[b] = _.extend(j, f, d)) : void 0 !== d && (g[b] = d));
            return g
        }, _.extend({
            expando: "jQuery" + ($ + Math.random()).replace(/\D/g, ""),
            isReady: !0,
            error: function(a) {
                throw new Error(a)
            },
            noop: function() {},
            isFunction: function(a) {
                return "function" === _.type(a)
            },
            isArray: Array.isArray,
            isWindow: function(a) {
                return null != a && a === a.window
            },
            isNumeric: function(a) {
                return !_.isArray(a) && a - parseFloat(a) >= 0
            },
            isPlainObject: function(a) {
                return "object" !== _.type(a) || a.nodeType || _.isWindow(a)?!1 : a.constructor&&!X.call(a.constructor.prototype, "isPrototypeOf")?!1 : !0
            },
            isEmptyObject: function(a) {
                var b;
                for (b in a)
                    return !1;
                return !0
            },
            type: function(a) {
                return null == a ? a + "" : "object" == typeof a || "function" == typeof a ? V[W.call(a)] || "object" : typeof a
            },
            globalEval: function(a) {
                var b, c = eval;
                a = _.trim(a), a && (1 === a.indexOf("use strict") ? (b = Z.createElement("script"), b.text = a, Z.head.appendChild(b).parentNode.removeChild(b)) : c(a))
            },
            camelCase: function(a) {
                return a.replace(bb, "ms-").replace(cb, db)
            },
            nodeName: function(a, b) {
                return a.nodeName && a.nodeName.toLowerCase() === b.toLowerCase()
            },
            each: function(a, b, d) {
                var e, f = 0, g = a.length, h = c(a);
                if (d) {
                    if (h)
                        for (; g > f && (e = b.apply(a[f], d), e!==!1); f++);
                    else 
                        for (f in a)
                            if (e = b.apply(a[f], d), e===!1)
                                break
                } else if (h)
                    for (; g > f && (e = b.call(a[f], f, a[f]), e!==!1); f++);
                else 
                    for (f in a)
                        if (e = b.call(a[f], f, a[f]), e===!1)
                            break;
                return a
            },
            trim: function(a) {
                return null == a ? "" : (a + "").replace(ab, "")
            },
            makeArray: function(a, b) {
                var d = b || [];
                return null != a && (c(Object(a)) ? _.merge(d, "string" == typeof a ? [a] : a) : T.call(d, a)), d
            },
            inArray: function(a, b, c) {
                return null == b?-1 : U.call(b, a, c)
            },
            merge: function(a, b) {
                for (var c =+ b.length, d = 0, e = a.length; c > d; d++)
                    a[e++] = b[d];
                return a.length = e, a
            },
            grep: function(a, b, c) {
                for (var d, e = [], f = 0, g = a.length, h=!c; g > f; f++)
                    d=!b(a[f], f), d !== h && e.push(a[f]);
                return e
            },
            map: function(a, b, d) {
                var e, f = 0, g = a.length, h = c(a), i = [];
                if (h)
                    for (; g > f; f++)
                        e = b(a[f], f, d), null != e && i.push(e);
                else 
                    for (f in a)
                        e = b(a[f], f, d), null != e && i.push(e);
                return S.apply([], i)
            },
            guid: 1,
            proxy: function(a, b) {
                var c, d, e;
                return "string" == typeof b && (c = a[b], b = a, a = c), _.isFunction(a) ? (d = R.call(arguments, 2), e = function() {
                    return a.apply(b || this, d.concat(R.call(arguments)))
                }, e.guid = a.guid = a.guid || _.guid++, e) : void 0
            },
            now: Date.now,
            support: Y
        }), _.each("Boolean Number String Function Array Date RegExp Object Error".split(" "), function(a, b) {
            V["[object " + b + "]"] = b.toLowerCase()
        });
        var eb = function(a) {
            function b(a, b, c, d) {
                var e, f, g, h, i, j, l, n, o, p;
                if ((b ? b.ownerDocument || b : O) !== G && F(b), b = b || G, c = c || [], !a || "string" != typeof a)
                    return c;
                if (1 !== (h = b.nodeType) && 9 !== h)
                    return [];
                if (I&&!d) {
                    if (e = sb.exec(a))
                        if (g = e[1]) {
                            if (9 === h) {
                                if (f = b.getElementById(g), !f ||!f.parentNode)
                                    return c;
                                    if (f.id === g)
                                        return c.push(f), c
                            } else if (b.ownerDocument && (f = b.ownerDocument.getElementById(g)) && M(b, f) && f.id === g)
                                return c.push(f), c
                        } else {
                            if (e[2])
                                return _.apply(c, b.getElementsByTagName(a)), c;
                                if ((g = e[3]) && v.getElementsByClassName && b.getElementsByClassName)
                                    return _.apply(c, b.getElementsByClassName(g)), c
                        }
                    if (v.qsa && (!J ||!J.test(a))) {
                        if (n = l = N, o = b, p = 9 === h && a, 1 === h && "object" !== b.nodeName.toLowerCase()) {
                            for (j = z(a), (l = b.getAttribute("id")) ? n = l.replace(ub, "\\$&") : b.setAttribute("id", n), n = "[id='" + n + "'] ", i = j.length; i--;)
                                j[i] = n + m(j[i]);
                            o = tb.test(a) && k(b.parentNode) || b, p = j.join(",")
                        }
                        if (p)
                            try {
                                return _.apply(c, o.querySelectorAll(p)), c
                        } catch (q) {} finally {
                            l || b.removeAttribute("id")
                        }
                    }
                }
                return B(a.replace(ib, "$1"), b, c, d)
            }
            function c() {
                function a(c, d) {
                    return b.push(c + " ") > w.cacheLength && delete a[b.shift()], a[c + " "] = d
                }
                var b = [];
                return a
            }
            function d(a) {
                return a[N]=!0, a
            }
            function e(a) {
                var b = G.createElement("div");
                try {
                    return !!a(b)
                } catch (c) {
                    return !1
                } finally {
                    b.parentNode && b.parentNode.removeChild(b), b = null
                }
            }
            function f(a, b) {
                for (var c = a.split("|"), d = a.length; d--;)
                    w.attrHandle[c[d]] = b
            }
            function g(a, b) {
                var c = b && a, d = c && 1 === a.nodeType && 1 === b.nodeType && (~b.sourceIndex || W) - (~a.sourceIndex || W);
                if (d)
                    return d;
                if (c)
                    for (; c = c.nextSibling;)
                        if (c === b)
                            return - 1;
                return a ? 1 : - 1
            }
            function h(a) {
                return function(b) {
                    var c = b.nodeName.toLowerCase();
                    return "input" === c && b.type === a
                }
            }
            function i(a) {
                return function(b) {
                    var c = b.nodeName.toLowerCase();
                    return ("input" === c || "button" === c) && b.type === a
                }
            }
            function j(a) {
                return d(function(b) {
                    return b =+ b, d(function(c, d) {
                        for (var e, f = a([], c.length, b), g = f.length; g--;)
                            c[e = f[g]] && (c[e]=!(d[e] = c[e]))
                    })
                })
            }
            function k(a) {
                return a && typeof a.getElementsByTagName !== V && a
            }
            function l() {}
            function m(a) {
                for (var b = 0, c = a.length, d = ""; c > b; b++)
                    d += a[b].value;
                return d
            }
            function n(a, b, c) {
                var d = b.dir, e = c && "parentNode" === d, f = Q++;
                return b.first ? function(b, c, f) {
                    for (; b = b[d];)
                        if (1 === b.nodeType || e)
                            return a(b, c, f)
                } : function(b, c, g) {
                    var h, i, j = [P, f];
                    if (g) {
                        for (; b = b[d];)
                            if ((1 === b.nodeType || e) && a(b, c, g))
                                return !0
                    } else 
                        for (; b = b[d];)
                            if (1 === b.nodeType || e) {
                                if (i = b[N] || (b[N] = {}), (h = i[d]) && h[0] === P && h[1] === f)
                                    return j[2] = h[2];
                                    if (i[d] = j, j[2] = a(b, c, g))
                                        return !0
                            }
                }
            }
            function o(a) {
                return a.length > 1 ? function(b, c, d) {
                    for (var e = a.length; e--;)
                        if (!a[e](b, c, d))
                            return !1;
                    return !0
                } : a[0]
            }
            function p(a, c, d) {
                for (var e = 0, f = c.length; f > e; e++)
                    b(a, c[e], d);
                return d
            }
            function q(a, b, c, d, e) {
                for (var f, g = [], h = 0, i = a.length, j = null != b; i > h; h++)(f = a[h]) 
                    && (!c || c(f, d, e)) && (g.push(f), j && b.push(h));
                return g
            }
            function r(a, b, c, e, f, g) {
                return e&&!e[N] && (e = r(e)), f&&!f[N] && (f = r(f, g)), d(function(d, g, h, i) {
                    var j, k, l, m = [], n = [], o = g.length, r = d || p(b || "*", h.nodeType ? [h] : h, []), s=!a ||!d && b ? r : q(r, m, a, h, i), t = c ? f || (d ? a : o || e) ? [] : g : s;
                    if (c && c(s, t, h, i), e)
                        for (j = q(t, n), e(j, [], h, i), k = j.length; k--;)(l = j[k]) 
                            && (t[n[k]]=!(s[n[k]] = l));
                    if (d) {
                        if (f || a) {
                            if (f) {
                                for (j = [], k = t.length; k--;)(l = t[k]) 
                                    && j.push(s[k] = l);
                                f(null, t = [], j, i)
                            }
                            for (k = t.length; k--;)(l = t[k]) 
                                && (j = f ? bb.call(d, l) : m[k])>-1 && (d[j]=!(g[j] = l))
                            }
                    } else 
                        t = q(t === g ? t.splice(o, t.length) : t), f ? f(null, g, t, i) : _.apply(g, t)
                })
            }
            function s(a) {
                for (var b, c, d, e = a.length, f = w.relative[a[0].type], g = f || w.relative[" "], h = f ? 1 : 0, i = n(function(a) {
                    return a === b
                }, g, !0), j = n(function(a) {
                    return bb.call(b, a)>-1
                }, g, !0), k = [function(a, c, d) {
                    return !f && (d || c !== C) || ((b = c).nodeType ? i(a, c, d) : j(a, c, d))
                }
                ]; e > h; h++)
                    if (c = w.relative[a[h].type])
                        k = [n(o(k), c)];
                    else {
                        if (c = w.filter[a[h].type].apply(null, a[h].matches), c[N]) {
                            for (d=++h; e > d&&!w.relative[a[d].type]; d++);
                            return r(h > 1 && o(k), h > 1 && m(a.slice(0, h - 1).concat({
                                value: " " === a[h - 2].type ? "*": ""
                            })).replace(ib, "$1"), c, d > h && s(a.slice(h, d)), e > d && s(a = a.slice(d)), e > d && m(a))
                        }
                        k.push(c)
                    }
                return o(k)
            }
            function t(a, c) {
                var e = c.length > 0, f = a.length > 0, g = function(d, g, h, i, j) {
                    var k, l, m, n = 0, o = "0", p = d && [], r = [], s = C, t = d || f && w.find.TAG("*", j), u = P += null == s ? 1: Math.random() || .1, v = t.length;
                    for (j && (C = g !== G && g); o !== v && null != (k = t[o]); o++) {
                        if (f && k) {
                            for (l = 0; m = a[l++];)
                                if (m(k, g, h)) {
                                    i.push(k);
                                    break
                                }
                            j && (P = u)
                        }
                        e && ((k=!m && k) && n--, d && p.push(k))
                    }
                    if (n += o, e && o !== n) {
                        for (l = 0; m = c[l++];)
                            m(p, r, g, h);
                        if (d) {
                            if (n > 0)
                                for (; o--;)
                                    p[o] || r[o] || (r[o] = Z.call(i));
                            r = q(r)
                        }
                        _.apply(i, r), j&&!d && r.length > 0 && n + c.length > 1 && b.uniqueSort(i)
                    }
                    return j && (P = u, C = s), p
                };
                return e ? d(g) : g
            }
            var u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N = "sizzle" +- new Date, O = a.document, P = 0, Q = 0, R = c(), S = c(), T = c(), U = function(a, b) {
                return a === b && (E=!0), 0
            }, V = "undefined", W = 1<<31, X = {}.hasOwnProperty, Y = [], Z = Y.pop, $ = Y.push, _ = Y.push, ab = Y.slice, bb = Y.indexOf || function(a) {
                for (var b = 0, c = this.length; c > b; b++)
                    if (this[b] === a)
                        return b;
                return - 1
            }, cb = "checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped", db = "[\\x20\\t\\r\\n\\f]", eb = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+", fb = eb.replace("w", "w#"), gb = "\\[" + db + "*(" + eb + ")(?:" + db + "*([*^$|!~]?=)" + db + "*(?:'((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\"|(" + fb + "))|)" + db + "*\\]", hb = ":(" + eb + ")(?:\\((('((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\")|((?:\\\\.|[^\\\\()[\\]]|" + gb + ")*)|.*)\\)|)", ib = new RegExp("^" + db + "+|((?:^|[^\\\\])(?:\\\\.)*)" + db + "+$", "g"), jb = new RegExp("^" + db + "*," + db + "*"), kb = new RegExp("^" + db + "*([>+~]|" + db + ")" + db + "*"), lb = new RegExp("=" + db + "*([^\\]'\"]*?)" + db + "*\\]", "g"), mb = new RegExp(hb), nb = new RegExp("^" + fb + "$"), ob = {
                ID: new RegExp("^#(" + eb + ")"),
                CLASS: new RegExp("^\\.(" + eb + ")"),
                TAG: new RegExp("^(" + eb.replace("w", "w*") + ")"),
                ATTR: new RegExp("^" + gb),
                PSEUDO: new RegExp("^" + hb),
                CHILD: new RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\(" + db + "*(even|odd|(([+-]|)(\\d*)n|)" + db + "*(?:([+-]|)" + db + "*(\\d+)|))" + db + "*\\)|)", "i"),
                bool: new RegExp("^(?:" + cb + ")$", "i"),
                needsContext: new RegExp("^" + db + "*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\(" + db + "*((?:-\\d)?\\d*)" + db + "*\\)|)(?=[^-]|$)", "i")
            }, pb = /^(?:input|select|textarea|button)$/i, qb = /^h\d$/i, rb = /^[^{]+\{\s*\[native \w/, sb = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/, tb = /[+~]/, ub = /'|\\/g, vb = new RegExp("\\\\([\\da-f]{1,6}" + db + "?|(" + db + ")|.)", "ig"), wb = function(a, b, c) {
                var d = "0x" + b - 65536;
                return d !== d || c ? b : 0 > d ? String.fromCharCode(d + 65536) : String.fromCharCode(d>>10 | 55296, 1023 & d | 56320)
            };
            try {
                _.apply(Y = ab.call(O.childNodes), O.childNodes), Y[O.childNodes.length].nodeType
            } catch (xb) {
                _ = {
                    apply: Y.length ? function(a, b) {
                        $.apply(a, ab.call(b))
                    }
                    : function(a, b) {
                        for (var c = a.length, d = 0; a[c++] = b[d++];);
                        a.length = c - 1
                    }
                }
            }
            v = b.support = {}, y = b.isXML = function(a) {
                var b = a && (a.ownerDocument || a).documentElement;
                return b ? "HTML" !== b.nodeName : !1
            }, F = b.setDocument = function(a) {
                var b, c = a ? a.ownerDocument || a: O, d = c.defaultView;
                return c !== G && 9 === c.nodeType && c.documentElement ? (G = c, H = c.documentElement, I=!y(c), d && d !== d.top && (d.addEventListener ? d.addEventListener("unload", function() {
                    F()
                }, !1) : d.attachEvent && d.attachEvent("onunload", function() {
                    F()
                })), v.attributes = e(function(a) {
                    return a.className = "i", !a.getAttribute("className")
                }), v.getElementsByTagName = e(function(a) {
                    return a.appendChild(c.createComment("")), !a.getElementsByTagName("*").length
                }), v.getElementsByClassName = rb.test(c.getElementsByClassName) && e(function(a) {
                    return a.innerHTML = "<div class='a'></div><div class='a i'></div>", a.firstChild.className = "i", 2 === a.getElementsByClassName("i").length
                }), v.getById = e(function(a) {
                    return H.appendChild(a).id = N, !c.getElementsByName ||!c.getElementsByName(N).length
                }), v.getById ? (w.find.ID = function(a, b) {
                    if (typeof b.getElementById !== V && I) {
                        var c = b.getElementById(a);
                        return c && c.parentNode ? [c] : []
                    }
                }, w.filter.ID = function(a) {
                    var b = a.replace(vb, wb);
                    return function(a) {
                        return a.getAttribute("id") === b
                    }
                }) : (delete w.find.ID, w.filter.ID = function(a) {
                    var b = a.replace(vb, wb);
                    return function(a) {
                        var c = typeof a.getAttributeNode !== V && a.getAttributeNode("id");
                        return c && c.value === b
                    }
                }), w.find.TAG = v.getElementsByTagName ? function(a, b) {
                    return typeof b.getElementsByTagName !== V ? b.getElementsByTagName(a) : void 0
                } : function(a, b) {
                    var c, d = [], e = 0, f = b.getElementsByTagName(a);
                    if ("*" === a) {
                        for (; c = f[e++];)
                            1 === c.nodeType && d.push(c);
                        return d
                    }
                    return f
                }, w.find.CLASS = v.getElementsByClassName && function(a, b) {
                    return typeof b.getElementsByClassName !== V && I ? b.getElementsByClassName(a) : void 0
                }, K = [], J = [], (v.qsa = rb.test(c.querySelectorAll)) && (e(function(a) {
                    a.innerHTML = "<select msallowclip=''><option selected=''></option></select>", a.querySelectorAll("[msallowclip^='']").length && J.push("[*^$]=" + db + "*(?:''|\"\")"), a.querySelectorAll("[selected]").length || J.push("\\[" + db + "*(?:value|" + cb + ")"), a.querySelectorAll(":checked").length || J.push(":checked")
                }), e(function(a) {
                    var b = c.createElement("input");
                    b.setAttribute("type", "hidden"), a.appendChild(b).setAttribute("name", "D"), a.querySelectorAll("[name=d]").length && J.push("name" + db + "*[*^$|!~]?="), a.querySelectorAll(":enabled").length || J.push(":enabled", ":disabled"), a.querySelectorAll("*,:x"), J.push(",.*:")
                })), (v.matchesSelector = rb.test(L = H.matches || H.webkitMatchesSelector || H.mozMatchesSelector || H.oMatchesSelector || H.msMatchesSelector)) && e(function(a) {
                    v.disconnectedMatch = L.call(a, "div"), L.call(a, "[s!='']:x"), K.push("!=", hb)
                }), J = J.length && new RegExp(J.join("|")), K = K.length && new RegExp(K.join("|")), b = rb.test(H.compareDocumentPosition), M = b || rb.test(H.contains) ? function(a, b) {
                    var c = 9 === a.nodeType ? a.documentElement: a, d = b && b.parentNode;
                    return a === d ||!(!d || 1 !== d.nodeType ||!(c.contains ? c.contains(d) : a.compareDocumentPosition && 16 & a.compareDocumentPosition(d)))
                } : function(a, b) {
                    if (b)
                        for (; b = b.parentNode;)
                            if (b === a)
                                return !0;
                    return !1
                }, U = b ? function(a, b) {
                    if (a === b)
                        return E=!0, 0;
                    var d=!a.compareDocumentPosition-!b.compareDocumentPosition;
                    return d ? d : (d = (a.ownerDocument || a) === (b.ownerDocument || b) ? a.compareDocumentPosition(b) : 1, 1 & d ||!v.sortDetached && b.compareDocumentPosition(a) === d ? a === c || a.ownerDocument === O && M(O, a)?-1 : b === c || b.ownerDocument === O && M(O, b) ? 1 : D ? bb.call(D, a) - bb.call(D, b) : 0 : 4 & d?-1 : 1)
                } : function(a, b) {
                    if (a === b)
                        return E=!0, 0;
                    var d, e = 0, f = a.parentNode, h = b.parentNode, i = [a], j = [b];
                    if (!f ||!h)
                        return a === c?-1 : b === c ? 1 : f?-1 : h ? 1 : D ? bb.call(D, a) - bb.call(D, b) : 0;
                    if (f === h)
                        return g(a, b);
                    for (d = a; d = d.parentNode;)
                        i.unshift(d);
                    for (d = b; d = d.parentNode;)
                        j.unshift(d);
                    for (; i[e] === j[e];)
                        e++;
                    return e ? g(i[e], j[e]) : i[e] === O?-1 : j[e] === O ? 1 : 0
                }, c) : G
            }, b.matches = function(a, c) {
                return b(a, null, null, c)
            }, b.matchesSelector = function(a, c) {
                if ((a.ownerDocument || a) !== G && F(a), c = c.replace(lb, "='$1']"), !(!v.matchesSelector ||!I || K && K.test(c) || J && J.test(c)))
                    try {
                        var d = L.call(a, c);
                        if (d || v.disconnectedMatch || a.document && 11 !== a.document.nodeType)
                            return d
                } catch (e) {}
                return b(c, G, null, [a]).length > 0
            }, b.contains = function(a, b) {
                return (a.ownerDocument || a) !== G && F(a), M(a, b)
            }, b.attr = function(a, b) {
                (a.ownerDocument || a) !== G && F(a);
                var c = w.attrHandle[b.toLowerCase()], d = c && X.call(w.attrHandle, b.toLowerCase()) ? c(a, b, !I): void 0;
                return void 0 !== d ? d : v.attributes ||!I ? a.getAttribute(b) : (d = a.getAttributeNode(b)) && d.specified ? d.value : null
            }, b.error = function(a) {
                throw new Error("Syntax error, unrecognized expression: " + a)
            }, b.uniqueSort = function(a) {
                var b, c = [], d = 0, e = 0;
                if (E=!v.detectDuplicates, D=!v.sortStable && a.slice(0), a.sort(U), E) {
                    for (; b = a[e++];)
                        b === a[e] && (d = c.push(e));
                    for (; d--;)
                        a.splice(c[d], 1)
                }
                return D = null, a
            }, x = b.getText = function(a) {
                var b, c = "", d = 0, e = a.nodeType;
                if (e) {
                    if (1 === e || 9 === e || 11 === e) {
                        if ("string" == typeof a.textContent)
                            return a.textContent;
                        for (a = a.firstChild; a; a = a.nextSibling)
                            c += x(a)
                        } else if (3 === e || 4 === e)
                        return a.nodeValue
                } else 
                    for (; b = a[d++];)
                        c += x(b);
                return c
            }, w = b.selectors = {
                cacheLength: 50,
                createPseudo: d,
                match: ob,
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
                    ATTR: function(a) {
                        return a[1] = a[1].replace(vb, wb), a[3] = (a[3] || a[4] || a[5] || "").replace(vb, wb), "~=" === a[2] && (a[3] = " " + a[3] + " "), a.slice(0, 4)
                    },
                    CHILD: function(a) {
                        return a[1] = a[1].toLowerCase(), "nth" === a[1].slice(0, 3) ? (a[3] || b.error(a[0]), a[4] =+ (a[4] ? a[5] + (a[6] || 1) : 2 * ("even" === a[3] || "odd" === a[3])), a[5] =+ (a[7] + a[8] || "odd" === a[3])) : a[3] && b.error(a[0]), a
                    },
                    PSEUDO: function(a) {
                        var b, c=!a[6] && a[2];
                        return ob.CHILD.test(a[0]) ? null : (a[3] ? a[2] = a[4] || a[5] || "" : c && mb.test(c) && (b = z(c, !0)) && (b = c.indexOf(")", c.length - b) - c.length) && (a[0] = a[0].slice(0, b), a[2] = c.slice(0, b)), a.slice(0, 3))
                    }
                },
                filter: {
                    TAG: function(a) {
                        var b = a.replace(vb, wb).toLowerCase();
                        return "*" === a ? function() {
                            return !0
                        } : function(a) {
                            return a.nodeName && a.nodeName.toLowerCase() === b
                        }
                    },
                    CLASS: function(a) {
                        var b = R[a + " "];
                        return b || (b = new RegExp("(^|" + db + ")" + a + "(" + db + "|$)")) && R(a, function(a) {
                            return b.test("string" == typeof a.className && a.className || typeof a.getAttribute !== V && a.getAttribute("class") || "")
                        })
                    },
                    ATTR: function(a, c, d) {
                        return function(e) {
                            var f = b.attr(e, a);
                            return null == f ? "!=" === c : c ? (f += "", "=" === c ? f === d : "!=" === c ? f !== d : "^=" === c ? d && 0 === f.indexOf(d) : "*=" === c ? d && f.indexOf(d)>-1 : "$=" === c ? d && f.slice( - d.length) === d : "~=" === c ? (" " + f + " ").indexOf(d)>-1 : "|=" === c ? f === d || f.slice(0, d.length + 1) === d + "-" : !1) : !0
                        }
                    },
                    CHILD: function(a, b, c, d, e) {
                        var f = "nth" !== a.slice(0, 3), g = "last" !== a.slice( - 4), h = "of-type" === b;
                        return 1 === d && 0 === e ? function(a) {
                            return !!a.parentNode
                        } : function(b, c, i) {
                            var j, k, l, m, n, o, p = f !== g ? "nextSibling": "previousSibling", q = b.parentNode, r = h && b.nodeName.toLowerCase(), s=!i&&!h;
                            if (q) {
                                if (f) {
                                    for (; p;) {
                                        for (l = b; l = l[p];)
                                            if (h ? l.nodeName.toLowerCase() === r : 1 === l.nodeType)
                                                return !1;
                                        o = p = "only" === a&&!o && "nextSibling"
                                    }
                                    return !0
                                }
                                if (o = [g ? q.firstChild: q.lastChild], g && s) {
                                    for (k = q[N] || (q[N] = {}), j = k[a] || [], n = j[0] === P && j[1], m = j[0] === P && j[2], l = n && q.childNodes[n]; l=++n && l && l[p] || (m = n = 0) 
                                        || o.pop();
                                    )if (1 === l.nodeType&&++m && l === b) {
                                        k[a] = [P, n, m];
                                        break
                                    }
                                } else if (s && (j = (b[N] || (b[N] = {}))[a]) && j[0] === P)
                                    m = j[1];
                                else 
                                    for (; (l=++n && l && l[p] || (m = n = 0) || o.pop()) 
                                        && ((h ? l.nodeName.toLowerCase() !== r : 1 !== l.nodeType)||!++m || (s && ((l[N] || (l[N] = {}))[a] = [P, m]), l !== b));
                                );
                                return m -= e, m === d || m%d === 0 && m / d >= 0
                            }
                        }
                    },
                    PSEUDO: function(a, c) {
                        var e, f = w.pseudos[a] || w.setFilters[a.toLowerCase()] || b.error("unsupported pseudo: " + a);
                        return f[N] ? f(c) : f.length > 1 ? (e = [a, a, "", c], w.setFilters.hasOwnProperty(a.toLowerCase()) ? d(function(a, b) {
                            for (var d, e = f(a, c), g = e.length; g--;)
                                d = bb.call(a, e[g]), a[d]=!(b[d] = e[g])
                        }) : function(a) {
                            return f(a, 0, e)
                        }) : f
                    }
                },
                pseudos: {
                    not: d(function(a) {
                        var b = [], c = [], e = A(a.replace(ib, "$1"));
                        return e[N] ? d(function(a, b, c, d) {
                            for (var f, g = e(a, null, d, []), h = a.length; h--;)(f = g[h]) 
                                && (a[h]=!(b[h] = f))
                        }) : function(a, d, f) {
                            return b[0] = a, e(b, null, f, c), !c.pop()
                        }
                    }),
                    has: d(function(a) {
                        return function(c) {
                            return b(a, c).length > 0
                        }
                    }),
                    contains: d(function(a) {
                        return function(b) {
                            return (b.textContent || b.innerText || x(b)).indexOf(a)>-1
                        }
                    }),
                    lang: d(function(a) {
                        return nb.test(a || "") || b.error("unsupported lang: " + a), a = a.replace(vb, wb).toLowerCase(), function(b) {
                            var c;
                            do 
                                if (c = I ? b.lang : b.getAttribute("xml:lang") || b.getAttribute("lang"))
                                    return c = c.toLowerCase(), c === a || 0 === c.indexOf(a + "-");
                            while ((b = b.parentNode) && 1 === b.nodeType);
                            return !1
                        }
                    }),
                    target: function(b) {
                        var c = a.location && a.location.hash;
                        return c && c.slice(1) === b.id
                    },
                    root: function(a) {
                        return a === H
                    },
                    focus: function(a) {
                        return a === G.activeElement && (!G.hasFocus || G.hasFocus())&&!!(a.type || a.href||~a.tabIndex)
                    },
                    enabled: function(a) {
                        return a.disabled===!1
                    },
                    disabled: function(a) {
                        return a.disabled===!0
                    },
                    checked: function(a) {
                        var b = a.nodeName.toLowerCase();
                        return "input" === b&&!!a.checked || "option" === b&&!!a.selected
                    },
                    selected: function(a) {
                        return a.parentNode && a.parentNode.selectedIndex, a.selected===!0
                    },
                    empty: function(a) {
                        for (a = a.firstChild; a; a = a.nextSibling)
                            if (a.nodeType < 6)
                                return !1;
                        return !0
                    },
                    parent: function(a) {
                        return !w.pseudos.empty(a)
                    },
                    header: function(a) {
                        return qb.test(a.nodeName)
                    },
                    input: function(a) {
                        return pb.test(a.nodeName)
                    },
                    button: function(a) {
                        var b = a.nodeName.toLowerCase();
                        return "input" === b && "button" === a.type || "button" === b
                    },
                    text: function(a) {
                        var b;
                        return "input" === a.nodeName.toLowerCase() && "text" === a.type && (null == (b = a.getAttribute("type")) || "text" === b.toLowerCase())
                    },
                    first: j(function() {
                        return [0]
                    }),
                    last: j(function(a, b) {
                        return [b - 1]
                    }),
                    eq: j(function(a, b, c) {
                        return [0 > c ? c + b: c]
                    }),
                    even: j(function(a, b) {
                        for (var c = 0; b > c; c += 2)
                            a.push(c);
                        return a
                    }),
                    odd: j(function(a, b) {
                        for (var c = 1; b > c; c += 2)
                            a.push(c);
                        return a
                    }),
                    lt: j(function(a, b, c) {
                        for (var d = 0 > c ? c + b : c; --d >= 0;)
                            a.push(d);
                        return a
                    }),
                    gt: j(function(a, b, c) {
                        for (var d = 0 > c ? c + b : c; ++d < b;)
                            a.push(d);
                        return a
                    })
                }
            }, w.pseudos.nth = w.pseudos.eq;
            for (u in{
                radio: !0,
                checkbox: !0,
                file: !0,
                password: !0,
                image: !0
            })
                w.pseudos[u] = h(u);
            for (u in{
                submit: !0,
                reset: !0
            })
                w.pseudos[u] = i(u);
            return l.prototype = w.filters = w.pseudos, w.setFilters = new l, z = b.tokenize = function(a, c) {
                var d, e, f, g, h, i, j, k = S[a + " "];
                if (k)
                    return c ? 0 : k.slice(0);
                for (h = a, i = [], j = w.preFilter; h;) {
                    (!d || (e = jb.exec(h))) && (e && (h = h.slice(e[0].length) || h), i.push(f = [])), d=!1, (e = kb.exec(h)) && (d = e.shift(), f.push({
                        value: d,
                        type: e[0].replace(ib, " ")
                    }), h = h.slice(d.length));
                    for (g in w.filter)
                        !(e = ob[g].exec(h)) || j[g]&&!(e = j[g](e)) || (d = e.shift(), f.push({
                            value: d,
                            type: g,
                            matches: e
                        }), h = h.slice(d.length));
                    if (!d)
                        break
                }
                return c ? h.length : h ? b.error(a) : S(a, i).slice(0)
            }, A = b.compile = function(a, b) {
                var c, d = [], e = [], f = T[a + " "];
                if (!f) {
                    for (b || (b = z(a)), c = b.length; c--;)
                        f = s(b[c]), f[N] ? d.push(f) : e.push(f);
                    f = T(a, t(e, d)), f.selector = a
                }
                return f
            }, B = b.select = function(a, b, c, d) {
                var e, f, g, h, i, j = "function" == typeof a && a, l=!d && z(a = j.selector || a);
                if (c = c || [], 1 === l.length) {
                    if (f = l[0] = l[0].slice(0), f.length > 2 && "ID" === (g = f[0]).type && v.getById && 9 === b.nodeType && I && w.relative[f[1].type]) {
                        if (b = (w.find.ID(g.matches[0].replace(vb, wb), b) || [])[0], !b)
                            return c;
                        j && (b = b.parentNode), a = a.slice(f.shift().value.length)
                    }
                    for (e = ob.needsContext.test(a) ? 0 : f.length; e--&&(g = f[e], !w.relative[h = g.type]);)
                        if ((i = w.find[h]) && (d = i(g.matches[0].replace(vb, wb), tb.test(f[0].type) && k(b.parentNode) || b))) {
                            if (f.splice(e, 1), a = d.length && m(f), !a)
                                return _.apply(c, d), c;
                                break
                        }
                }
                return (j || A(a, l))(d, b, !I, c, tb.test(a) && k(b.parentNode) || b), c
            }, v.sortStable = N.split("").sort(U).join("") === N, v.detectDuplicates=!!E, F(), v.sortDetached = e(function(a) {
                return 1 & a.compareDocumentPosition(G.createElement("div"))
            }), e(function(a) {
                return a.innerHTML = "<a href='#'></a>", "#" === a.firstChild.getAttribute("href")
            }) || f("type|href|height|width", function(a, b, c) {
                return c ? void 0 : a.getAttribute(b, "type" === b.toLowerCase() ? 1 : 2)
            }), v.attributes && e(function(a) {
                return a.innerHTML = "<input/>", a.firstChild.setAttribute("value", ""), "" === a.firstChild.getAttribute("value")
            }) || f("value", function(a, b, c) {
                return c || "input" !== a.nodeName.toLowerCase() ? void 0 : a.defaultValue
            }), e(function(a) {
                return null == a.getAttribute("disabled")
            }) || f(cb, function(a, b, c) {
                var d;
                return c ? void 0 : a[b]===!0 ? b.toLowerCase() : (d = a.getAttributeNode(b)) && d.specified ? d.value : null
            }), b
        }(a);
        _.find = eb, _.expr = eb.selectors, _.expr[":"] = _.expr.pseudos, _.unique = eb.uniqueSort, _.text = eb.getText, _.isXMLDoc = eb.isXML, _.contains = eb.contains;
        var fb = _.expr.match.needsContext, gb = /^<(\w+)\s*\/?>(?:<\/\1>|)$/, hb = /^.[^:#\[\.,]*$/;
        _.filter = function(a, b, c) {
            var d = b[0];
            return c && (a = ":not(" + a + ")"), 1 === b.length && 1 === d.nodeType ? _.find.matchesSelector(d, a) ? [d] : [] : _.find.matches(a, _.grep(b, function(a) {
                return 1 === a.nodeType
            }))
        }, _.fn.extend({
            find: function(a) {
                var b, c = this.length, d = [], e = this;
                if ("string" != typeof a)
                    return this.pushStack(_(a).filter(function() {
                        for (b = 0; c > b; b++)
                            if (_.contains(e[b], this))
                                return !0
                            }));
                for (b = 0; c > b; b++)
                    _.find(a, e[b], d);
                return d = this.pushStack(c > 1 ? _.unique(d) : d), d.selector = this.selector ? this.selector + " " + a : a, d
            },
            filter: function(a) {
                return this.pushStack(d(this, a || [], !1))
            },
            not: function(a) {
                return this.pushStack(d(this, a || [], !0))
            },
            is: function(a) {
                return !!d(this, "string" == typeof a && fb.test(a) ? _(a) : a || [], !1).length
            }
        });
        var ib, jb = /^(?:\s*(<[\w\W]+>)[^>]*|#([\w-]*))$/, kb = _.fn.init = function(a, b) {
            var c, d;
            if (!a)
                return this;
            if ("string" == typeof a) {
                if (c = "<" === a[0] && ">" === a[a.length - 1] && a.length >= 3 ? [null, a, null] : jb.exec(a), !c ||!c[1] && b)
                    return !b || b.jquery ? (b || ib).find(a) : this.constructor(b).find(a);
                if (c[1]) {
                    if (b = b instanceof _ ? b[0] : b, _.merge(this, _.parseHTML(c[1], b && b.nodeType ? b.ownerDocument || b : Z, !0)), gb.test(c[1]) && _.isPlainObject(b))
                        for (c in b)
                            _.isFunction(this[c]) ? this[c](b[c]) : this.attr(c, b[c]);
                    return this
                }
                return d = Z.getElementById(c[2]), d && d.parentNode && (this.length = 1, this[0] = d), this.context = Z, this.selector = a, this
            }
            return a.nodeType ? (this.context = this[0] = a, this.length = 1, this) : _.isFunction(a) ? "undefined" != typeof ib.ready ? ib.ready(a) : a(_) : (void 0 !== a.selector && (this.selector = a.selector, this.context = a.context), _.makeArray(a, this))
        };
        kb.prototype = _.fn, ib = _(Z);
        var lb = /^(?:parents|prev(?:Until|All))/, mb = {
            children: !0,
            contents: !0,
            next: !0,
            prev: !0
        };
        _.extend({
            dir: function(a, b, c) {
                for (var d = [], e = void 0 !== c; (a = a[b]) && 9 !== a.nodeType;)
                    if (1 === a.nodeType) {
                        if (e && _(a).is(c))
                            break;
                            d.push(a)
                    }
                return d
            },
            sibling: function(a, b) {
                for (var c = []; a; a = a.nextSibling)
                    1 === a.nodeType && a !== b && c.push(a);
                return c
            }
        }), _.fn.extend({
            has: function(a) {
                var b = _(a, this), c = b.length;
                return this.filter(function() {
                    for (var a = 0; c > a; a++)
                        if (_.contains(this, b[a]))
                            return !0
                })
            },
            closest: function(a, b) {
                for (var c, d = 0, e = this.length, f = [], g = fb.test(a) || "string" != typeof a ? _(a, b || this.context) : 0; e > d; d++)
                    for (c = this[d]; c && c !== b; c = c.parentNode)
                        if (c.nodeType < 11 && (g ? g.index(c)>-1 : 1 === c.nodeType && _.find.matchesSelector(c, a))) {
                            f.push(c);
                            break
                        }
                return this.pushStack(f.length > 1 ? _.unique(f) : f)
            },
            index: function(a) {
                return a ? "string" == typeof a ? U.call(_(a), this[0]) : U.call(this, a.jquery ? a[0] : a) : this[0] && this[0].parentNode ? this.first().prevAll().length : - 1
            },
            add: function(a, b) {
                return this.pushStack(_.unique(_.merge(this.get(), _(a, b))))
            },
            addBack: function(a) {
                return this.add(null == a ? this.prevObject : this.prevObject.filter(a))
            }
        }), _.each({
            parent: function(a) {
                var b = a.parentNode;
                return b && 11 !== b.nodeType ? b : null
            },
            parents: function(a) {
                return _.dir(a, "parentNode")
            },
            parentsUntil: function(a, b, c) {
                return _.dir(a, "parentNode", c)
            },
            next: function(a) {
                return e(a, "nextSibling")
            },
            prev: function(a) {
                return e(a, "previousSibling")
            },
            nextAll: function(a) {
                return _.dir(a, "nextSibling")
            },
            prevAll: function(a) {
                return _.dir(a, "previousSibling")
            },
            nextUntil: function(a, b, c) {
                return _.dir(a, "nextSibling", c)
            },
            prevUntil: function(a, b, c) {
                return _.dir(a, "previousSibling", c)
            },
            siblings: function(a) {
                return _.sibling((a.parentNode || {}).firstChild, a)
            },
            children: function(a) {
                return _.sibling(a.firstChild)
            },
            contents: function(a) {
                return a.contentDocument || _.merge([], a.childNodes)
            }
        }, function(a, b) {
            _.fn[a] = function(c, d) {
                var e = _.map(this, b, c);
                return "Until" !== a.slice( - 5) && (d = c), d && "string" == typeof d && (e = _.filter(d, e)), this.length > 1 && (mb[a] || _.unique(e), lb.test(a) && e.reverse()), this.pushStack(e)
            }
        });
        var nb = /\S+/g, ob = {};
        _.Callbacks = function(a) {
            a = "string" == typeof a ? ob[a] || f(a) : _.extend({}, a);
            var b, c, d, e, g, h, i = [], j=!a.once && [], k = function(f) {
                for (b = a.memory && f, c=!0, h = e || 0, e = 0, g = i.length, d=!0; i && g > h; h++)
                    if (i[h].apply(f[0], f[1])===!1 && a.stopOnFalse) {
                        b=!1;
                        break
                    }
                d=!1, i && (j ? j.length && k(j.shift()) : b ? i = [] : l.disable())
            }, l = {
                add: function() {
                    if (i) {
                        var c = i.length;
                        !function f(b) {
                            _.each(b, function(b, c) {
                                var d = _.type(c);
                                "function" === d ? a.unique && l.has(c) || i.push(c) : c && c.length && "string" !== d && f(c)
                            })
                        }(arguments), d ? g = i.length : b && (e = c, k(b))
                    }
                    return this
                },
                remove: function() {
                    return i && _.each(arguments, function(a, b) {
                        for (var c; (c = _.inArray(b, i, c))>-1;)
                            i.splice(c, 1), d && (g >= c && g--, h >= c && h--)
                    }), this
                },
                has: function(a) {
                    return a ? _.inArray(a, i)>-1 : !(!i ||!i.length)
                },
                empty: function() {
                    return i = [], g = 0, this
                },
                disable: function() {
                    return i = j = b = void 0, this
                },
                disabled: function() {
                    return !i
                },
                lock: function() {
                    return j = void 0, b || l.disable(), this
                },
                locked: function() {
                    return !j
                },
                fireWith: function(a, b) {
                    return !i || c&&!j || (b = b || [], b = [a, b.slice ? b.slice(): b], d ? j.push(b) : k(b)), this
                },
                fire: function() {
                    return l.fireWith(this, arguments), this
                },
                fired: function() {
                    return !!c
                }
            };
            return l
        }, _.extend({
            Deferred: function(a) {
                var b = [["resolve", "done", _.Callbacks("once memory"), "resolved"], ["reject", "fail", _.Callbacks("once memory"), "rejected"], ["notify", "progress", _.Callbacks("memory")]], c = "pending", d = {
                    state: function() {
                        return c
                    },
                    always: function() {
                        return e.done(arguments).fail(arguments), this
                    },
                    then: function() {
                        var a = arguments;
                        return _.Deferred(function(c) {
                            _.each(b, function(b, f) {
                                var g = _.isFunction(a[b]) && a[b];
                                e[f[1]](function() {
                                    var a = g && g.apply(this, arguments);
                                    a && _.isFunction(a.promise) ? a.promise().done(c.resolve).fail(c.reject).progress(c.notify) : c[f[0] + "With"](this === d ? c.promise() : this, g ? [a] : arguments)
                                })
                            }), a = null
                        }).promise()
                    },
                    promise: function(a) {
                        return null != a ? _.extend(a, d) : d
                    }
                }, e = {};
                return d.pipe = d.then, _.each(b, function(a, f) {
                    var g = f[2], h = f[3];
                    d[f[1]] = g.add, h && g.add(function() {
                        c = h
                    }, b[1^a][2].disable, b[2][2].lock), e[f[0]] = function() {
                        return e[f[0] + "With"](this === e ? d : this, arguments), this
                    }, e[f[0] + "With"] = g.fireWith
                }), d.promise(e), a && a.call(e, e), e
            },
            when: function(a) {
                var b, c, d, e = 0, f = R.call(arguments), g = f.length, h = 1 !== g || a && _.isFunction(a.promise) ? g: 0, i = 1 === h ? a: _.Deferred(), j = function(a, c, d) {
                    return function(e) {
                        c[a] = this, d[a] = arguments.length > 1 ? R.call(arguments) : e, d === b ? i.notifyWith(c, d) : --h || i.resolveWith(c, d)
                    }
                };
                if (g > 1)
                    for (b = new Array(g), c = new Array(g), d = new Array(g); g > e; e++)
                        f[e] && _.isFunction(f[e].promise) ? f[e].promise().done(j(e, d, f)).fail(i.reject).progress(j(e, c, b)) : --h;
                return h || i.resolveWith(d, f), i.promise()
            }
        });
        var pb;
        _.fn.ready = function(a) {
            return _.ready.promise().done(a), this
        }, _.extend({
            isReady: !1,
            readyWait: 1,
            holdReady: function(a) {
                a ? _.readyWait++ : _.ready(!0)
            },
            ready: function(a) {
                (a===!0?--_.readyWait : _.isReady) || (_.isReady=!0, a!==!0&&--_.readyWait > 0 || (pb.resolveWith(Z, [_]), _.fn.triggerHandler && (_(Z).triggerHandler("ready"), _(Z).off("ready"))))
            }
        }), _.ready.promise = function(b) {
            return pb || (pb = _.Deferred(), "complete" === Z.readyState ? setTimeout(_.ready) : (Z.addEventListener("DOMContentLoaded", g, !1), a.addEventListener("load", g, !1))), pb.promise(b)
        }, _.ready.promise();
        var qb = _.access = function(a, b, c, d, e, f, g) {
            var h = 0, i = a.length, j = null == c;
            if ("object" === _.type(c)) {
                e=!0;
                for (h in c)
                    _.access(a, b, h, c[h], !0, f, g)
            } else if (void 0 !== d && (e=!0, _.isFunction(d) || (g=!0), j && (g ? (b.call(a, d), b = null) : (j = b, b = function(a, b, c) {
                return j.call(_(a), c)
            })), b))for (; i > h; h++)
                b(a[h], c, g ? d : d.call(a[h], h, b(a[h], c)));
            return e ? a : j ? b.call(a) : i ? b(a[0], c) : f
        };
        _.acceptData = function(a) {
            return 1 === a.nodeType || 9 === a.nodeType||!+a.nodeType
        }, h.uid = 1, h.accepts = _.acceptData, h.prototype = {
            key: function(a) {
                if (!h.accepts(a))
                    return 0;
                var b = {}, c = a[this.expando];
                if (!c) {
                    c = h.uid++;
                    try {
                        b[this.expando] = {
                            value: c
                        }, Object.defineProperties(a, b)
                    } catch (d) {
                        b[this.expando] = c, _.extend(a, b)
                    }
                }
                return this.cache[c] || (this.cache[c] = {}), c
            },
            set: function(a, b, c) {
                var d, e = this.key(a), f = this.cache[e];
                if ("string" == typeof b)
                    f[b] = c;
                else if (_.isEmptyObject(f))
                    _.extend(this.cache[e], b);
                else 
                    for (d in b)
                        f[d] = b[d];
                return f
            },
            get: function(a, b) {
                var c = this.cache[this.key(a)];
                return void 0 === b ? c : c[b]
            },
            access: function(a, b, c) {
                var d;
                return void 0 === b || b && "string" == typeof b && void 0 === c ? (d = this.get(a, b), void 0 !== d ? d : this.get(a, _.camelCase(b))) : (this.set(a, b, c), void 0 !== c ? c : b)
            },
            remove: function(a, b) {
                var c, d, e, f = this.key(a), g = this.cache[f];
                if (void 0 === b)
                    this.cache[f] = {};
                else {
                    _.isArray(b) ? d = b.concat(b.map(_.camelCase)) : (e = _.camelCase(b), b in g ? d = [b, e] : (d = e, d = d in g ? [d] : d.match(nb) || [])), c = d.length;
                    for (; c--;)
                        delete g[d[c]]
                }
            },
            hasData: function(a) {
                return !_.isEmptyObject(this.cache[a[this.expando]] || {})
            },
            discard: function(a) {
                a[this.expando] && delete this.cache[a[this.expando]]
            }
        };
        var rb = new h, sb = new h, tb = /^(?:\{[\w\W]*\}|\[[\w\W]*\])$/, ub = /([A-Z])/g;
        _.extend({
            hasData: function(a) {
                return sb.hasData(a) || rb.hasData(a)
            },
            data: function(a, b, c) {
                return sb.access(a, b, c)
            },
            removeData: function(a, b) {
                sb.remove(a, b)
            },
            _data: function(a, b, c) {
                return rb.access(a, b, c)
            },
            _removeData: function(a, b) {
                rb.remove(a, b)
            }
        }), _.fn.extend({
            data: function(a, b) {
                var c, d, e, f = this[0], g = f && f.attributes;
                if (void 0 === a) {
                    if (this.length && (e = sb.get(f), 1 === f.nodeType&&!rb.get(f, "hasDataAttrs"))) {
                        for (c = g.length; c--;)
                            g[c] && (d = g[c].name, 0 === d.indexOf("data-") && (d = _.camelCase(d.slice(5)), i(f, d, e[d])));
                        rb.set(f, "hasDataAttrs", !0)
                    }
                    return e
                }
                return "object" == typeof a ? this.each(function() {
                    sb.set(this, a)
                }) : qb(this, function(b) {
                    var c, d = _.camelCase(a);
                    if (f && void 0 === b) {
                        if (c = sb.get(f, a), void 0 !== c)
                            return c;
                        if (c = sb.get(f, d), void 0 !== c)
                            return c;
                        if (c = i(f, d, void 0), void 0 !== c)
                            return c
                    } else 
                        this.each(function() {
                            var c = sb.get(this, d);
                            sb.set(this, d, b), - 1 !== a.indexOf("-") && void 0 !== c && sb.set(this, a, b)
                        })
                }, null, b, arguments.length > 1, null, !0)
            },
            removeData: function(a) {
                return this.each(function() {
                    sb.remove(this, a)
                })
            }
        }), _.extend({
            queue: function(a, b, c) {
                var d;
                return a ? (b = (b || "fx") + "queue", d = rb.get(a, b), c && (!d || _.isArray(c) ? d = rb.access(a, b, _.makeArray(c)) : d.push(c)), d || []) : void 0
            },
            dequeue: function(a, b) {
                b = b || "fx";
                var c = _.queue(a, b), d = c.length, e = c.shift(), f = _._queueHooks(a, b), g = function() {
                    _.dequeue(a, b)
                };
                "inprogress" === e && (e = c.shift(), d--), e && ("fx" === b && c.unshift("inprogress"), delete f.stop, e.call(a, g, f)), !d && f && f.empty.fire()
            },
            _queueHooks: function(a, b) {
                var c = b + "queueHooks";
                return rb.get(a, c) || rb.access(a, c, {
                    empty: _.Callbacks("once memory").add(function() {
                        rb.remove(a, [b + "queue", c])
                    })
                })
            }
        }), _.fn.extend({
            queue: function(a, b) {
                var c = 2;
                return "string" != typeof a && (b = a, a = "fx", c--), arguments.length < c ? _.queue(this[0], a) : void 0 === b ? this : this.each(function() {
                    var c = _.queue(this, a, b);
                    _._queueHooks(this, a), "fx" === a && "inprogress" !== c[0] && _.dequeue(this, a)
                })
            },
            dequeue: function(a) {
                return this.each(function() {
                    _.dequeue(this, a)
                })
            },
            clearQueue: function(a) {
                return this.queue(a || "fx", [])
            },
            promise: function(a, b) {
                var c, d = 1, e = _.Deferred(), f = this, g = this.length, h = function() {
                    --d || e.resolveWith(f, [f])
                };
                for ("string" != typeof a && (b = a, a = void 0), a = a || "fx"; g--;)
                    c = rb.get(f[g], a + "queueHooks"), c && c.empty && (d++, c.empty.add(h));
                return h(), e.promise(b)
            }
        });
        var vb = /[+-]?(?:\d*\.|)\d+(?:[eE][+-]?\d+|)/.source, wb = ["Top", "Right", "Bottom", "Left"], xb = function(a, b) {
            return a = b || a, "none" === _.css(a, "display") ||!_.contains(a.ownerDocument, a)
        }, yb = /^(?:checkbox|radio)$/i;
        !function() {
            var a = Z.createDocumentFragment(), b = a.appendChild(Z.createElement("div")), c = Z.createElement("input");
            c.setAttribute("type", "radio"), c.setAttribute("checked", "checked"), c.setAttribute("name", "t"), b.appendChild(c), Y.checkClone = b.cloneNode(!0).cloneNode(!0).lastChild.checked, b.innerHTML = "<textarea>x</textarea>", Y.noCloneChecked=!!b.cloneNode(!0).lastChild.defaultValue
        }();
        var zb = "undefined";
        Y.focusinBubbles = "onfocusin"in a;
        var Ab = /^key/, Bb = /^(?:mouse|pointer|contextmenu)|click/, Cb = /^(?:focusinfocus|focusoutblur)$/, Db = /^([^.]*)(?:\.(.+)|)$/;
        _.event = {
            global: {},
            add: function(a, b, c, d, e) {
                var f, g, h, i, j, k, l, m, n, o, p, q = rb.get(a);
                if (q)
                    for (c.handler && (f = c, c = f.handler, e = f.selector), c.guid || (c.guid = _.guid++), (i = q.events) || (i = q.events = {}), (g = q.handle) || (g = q.handle = function(b) {
                        return typeof _ !== zb && _.event.triggered !== b.type ? _.event.dispatch.apply(a, arguments) : void 0
                    }), b = (b || "").match(nb) || [""], j = b.length; j--;)
                        h = Db.exec(b[j]) || [], n = p = h[1], o = (h[2] || "").split(".").sort(), n && (l = _.event.special[n] || {}, n = (e ? l.delegateType : l.bindType) || n, l = _.event.special[n] || {}, k = _.extend({
                            type: n,
                            origType: p,
                            data: d,
                            handler: c,
                            guid: c.guid,
                            selector: e,
                            needsContext: e && _.expr.match.needsContext.test(e),
                            namespace: o.join(".")
                        }, f), (m = i[n]) || (m = i[n] = [], m.delegateCount = 0, l.setup && l.setup.call(a, d, o, g)!==!1 || a.addEventListener && a.addEventListener(n, g, !1)), l.add && (l.add.call(a, k), k.handler.guid || (k.handler.guid = c.guid)), e ? m.splice(m.delegateCount++, 0, k) : m.push(k), _.event.global[n]=!0)
            },
            remove: function(a, b, c, d, e) {
                var f, g, h, i, j, k, l, m, n, o, p, q = rb.hasData(a) && rb.get(a);
                if (q && (i = q.events)) {
                    for (b = (b || "").match(nb) || [""], j = b.length; j--;)
                        if (h = Db.exec(b[j]) || [], n = p = h[1], o = (h[2] || "").split(".").sort(), n) {
                            for (l = _.event.special[n] || {}, n = (d ? l.delegateType : l.bindType) || n, m = i[n] || [], h = h[2] && new RegExp("(^|\\.)" + o.join("\\.(?:.*\\.|)") + "(\\.|$)"), g = f = m.length; f--;)
                                k = m[f], !e && p !== k.origType || c && c.guid !== k.guid || h&&!h.test(k.namespace) || d && d !== k.selector && ("**" !== d ||!k.selector) || (m.splice(f, 1), k.selector && m.delegateCount--, l.remove && l.remove.call(a, k));
                                g&&!m.length && (l.teardown && l.teardown.call(a, o, q.handle)!==!1 || _.removeEvent(a, n, q.handle), delete i[n])
                        } else 
                            for (n in i)
                                _.event.remove(a, n + b[j], c, d, !0);
                    _.isEmptyObject(i) && (delete q.handle, rb.remove(a, "events"))
                }
            },
            trigger: function(b, c, d, e) {
                var f, g, h, i, j, k, l, m = [d || Z], n = X.call(b, "type") ? b.type: b, o = X.call(b, "namespace") ? b.namespace.split("."): [];
                if (g = h = d = d || Z, 3 !== d.nodeType && 8 !== d.nodeType&&!Cb.test(n + _.event.triggered) && (n.indexOf(".") >= 0 && (o = n.split("."), n = o.shift(), o.sort()), j = n.indexOf(":") < 0 && "on" + n, b = b[_.expando] ? b : new _.Event(n, "object" == typeof b && b), b.isTrigger = e ? 2 : 3, b.namespace = o.join("."), b.namespace_re = b.namespace ? new RegExp("(^|\\.)" + o.join("\\.(?:.*\\.|)") + "(\\.|$)") : null, b.result = void 0, b.target || (b.target = d), c = null == c ? [b] : _.makeArray(c, [b]), l = _.event.special[n] || {}, e ||!l.trigger || l.trigger.apply(d, c)!==!1)
                    ) {
                    if (!e&&!l.noBubble&&!_.isWindow(d)) {
                        for (i = l.delegateType || n, Cb.test(i + n) || (g = g.parentNode); g; g = g.parentNode)
                            m.push(g), h = g;
                        h === (d.ownerDocument || Z) && m.push(h.defaultView || h.parentWindow || a)
                    }
                    for (f = 0; (g = m[f++])&&!b.isPropagationStopped();)
                        b.type = f > 1 ? i : l.bindType || n, k = (rb.get(g, "events") || {})[b.type] && rb.get(g, "handle"), k && k.apply(g, c), k = j && g[j], k && k.apply && _.acceptData(g) && (b.result = k.apply(g, c), b.result===!1 && b.preventDefault());
                    return b.type = n, e || b.isDefaultPrevented() || l._default && l._default.apply(m.pop(), c)!==!1 ||!_.acceptData(d) || j && _.isFunction(d[n])&&!_.isWindow(d) && (h = d[j], h && (d[j] = null), _.event.triggered = n, d[n](), _.event.triggered = void 0, h && (d[j] = h)), b.result
                }
            },
            dispatch: function(a) {
                a = _.event.fix(a);
                var b, c, d, e, f, g = [], h = R.call(arguments), i = (rb.get(this, "events") || {})[a.type] || [], j = _.event.special[a.type] || {};
                if (h[0] = a, a.delegateTarget = this, !j.preDispatch || j.preDispatch.call(this, a)!==!1) {
                    for (g = _.event.handlers.call(this, a, i), b = 0; (e = g[b++])&&!a.isPropagationStopped();)
                        for (a.currentTarget = e.elem, c = 0; (f = e.handlers[c++])&&!a.isImmediatePropagationStopped();)(!a.namespace_re || a.namespace_re.test(f.namespace)
                            ) && (a.handleObj = f, a.data = f.data, d = ((_.event.special[f.origType] || {}).handle || f.handler).apply(e.elem, h), void 0 !== d && (a.result = d)===!1 && (a.preventDefault(), a.stopPropagation()));
                    return j.postDispatch && j.postDispatch.call(this, a), a.result
                }
            },
            handlers: function(a, b) {
                var c, d, e, f, g = [], h = b.delegateCount, i = a.target;
                if (h && i.nodeType && (!a.button || "click" !== a.type))
                    for (; i !== this; i = i.parentNode || this)
                        if (i.disabled!==!0 || "click" !== a.type) {
                            for (d = [], c = 0; h > c; c++)
                                f = b[c], e = f.selector + " ", void 0 === d[e] && (d[e] = f.needsContext ? _(e, this).index(i) >= 0 : _.find(e, this, null, [i]).length), d[e] && d.push(f);
                                d.length && g.push({
                                    elem: i,
                                    handlers: d
                                })
                        }
                return h < b.length && g.push({
                    elem: this,
                    handlers: b.slice(h)
                }), g
            },
            props: "altKey bubbles cancelable ctrlKey currentTarget eventPhase metaKey relatedTarget shiftKey target timeStamp view which".split(" "),
            fixHooks: {},
            keyHooks: {
                props: "char charCode key keyCode".split(" "),
                filter: function(a, b) {
                    return null == a.which && (a.which = null != b.charCode ? b.charCode : b.keyCode), a
                }
            },
            mouseHooks: {
                props: "button buttons clientX clientY offsetX offsetY pageX pageY screenX screenY toElement".split(" "),
                filter: function(a, b) {
                    var c, d, e, f = b.button;
                    return null == a.pageX && null != b.clientX && (c = a.target.ownerDocument || Z, d = c.documentElement, e = c.body, a.pageX = b.clientX + (d && d.scrollLeft || e && e.scrollLeft || 0) - (d && d.clientLeft || e && e.clientLeft || 0), a.pageY = b.clientY + (d && d.scrollTop || e && e.scrollTop || 0) - (d && d.clientTop || e && e.clientTop || 0)), a.which || void 0 === f || (a.which = 1 & f ? 1 : 2 & f ? 3 : 4 & f ? 2 : 0), a
                }
            },
            fix: function(a) {
                if (a[_.expando])
                    return a;
                var b, c, d, e = a.type, f = a, g = this.fixHooks[e];
                for (g || (this.fixHooks[e] = g = Bb.test(e) ? this.mouseHooks : Ab.test(e) ? this.keyHooks : {}), d = g.props ? this.props.concat(g.props) : this.props, a = new _.Event(f), b = d.length; b--;)
                    c = d[b], a[c] = f[c];
                return a.target || (a.target = Z), 3 === a.target.nodeType && (a.target = a.target.parentNode), g.filter ? g.filter(a, f) : a
            },
            special: {
                load: {
                    noBubble: !0
                },
                focus: {
                    trigger: function() {
                        return this !== l() && this.focus ? (this.focus(), !1) : void 0
                    },
                    delegateType: "focusin"
                },
                blur: {
                    trigger: function() {
                        return this === l() && this.blur ? (this.blur(), !1) : void 0
                    },
                    delegateType: "focusout"
                },
                click: {
                    trigger: function() {
                        return "checkbox" === this.type && this.click && _.nodeName(this, "input") ? (this.click(), !1) : void 0
                    },
                    _default: function(a) {
                        return _.nodeName(a.target, "a")
                    }
                },
                beforeunload: {
                    postDispatch: function(a) {
                        void 0 !== a.result && a.originalEvent && (a.originalEvent.returnValue = a.result)
                    }
                }
            },
            simulate: function(a, b, c, d) {
                var e = _.extend(new _.Event, c, {
                    type: a,
                    isSimulated: !0,
                    originalEvent: {}
                });
                d ? _.event.trigger(e, null, b) : _.event.dispatch.call(b, e), e.isDefaultPrevented() && c.preventDefault()
            }
        }, _.removeEvent = function(a, b, c) {
            a.removeEventListener && a.removeEventListener(b, c, !1)
        }, _.Event = function(a, b) {
            return this instanceof _.Event ? (a && a.type ? (this.originalEvent = a, this.type = a.type, this.isDefaultPrevented = a.defaultPrevented || void 0 === a.defaultPrevented && a.returnValue===!1 ? j : k) : this.type = a, b && _.extend(this, b), this.timeStamp = a && a.timeStamp || _.now(), void(this[_.expando]=!0)) : new _.Event(a, b)
        }, _.Event.prototype = {
            isDefaultPrevented: k,
            isPropagationStopped: k,
            isImmediatePropagationStopped: k,
            preventDefault: function() {
                var a = this.originalEvent;
                this.isDefaultPrevented = j, a && a.preventDefault && a.preventDefault()
            },
            stopPropagation: function() {
                var a = this.originalEvent;
                this.isPropagationStopped = j, a && a.stopPropagation && a.stopPropagation()
            },
            stopImmediatePropagation: function() {
                var a = this.originalEvent;
                this.isImmediatePropagationStopped = j, a && a.stopImmediatePropagation && a.stopImmediatePropagation(), this.stopPropagation()
            }
        }, _.each({
            mouseenter: "mouseover",
            mouseleave: "mouseout",
            pointerenter: "pointerover",
            pointerleave: "pointerout"
        }, function(a, b) {
            _.event.special[a] = {
                delegateType: b,
                bindType: b,
                handle: function(a) {
                    var c, d = this, e = a.relatedTarget, f = a.handleObj;
                    return (!e || e !== d&&!_.contains(d, e)) && (a.type = f.origType, c = f.handler.apply(this, arguments), a.type = b), c
                }
            }
        }), Y.focusinBubbles || _.each({
            focus: "focusin",
            blur: "focusout"
        }, function(a, b) {
            var c = function(a) {
                _.event.simulate(b, a.target, _.event.fix(a), !0)
            };
            _.event.special[b] = {
                setup: function() {
                    var d = this.ownerDocument || this, e = rb.access(d, b);
                    e || d.addEventListener(a, c, !0), rb.access(d, b, (e || 0) + 1)
                },
                teardown: function() {
                    var d = this.ownerDocument || this, e = rb.access(d, b) - 1;
                    e ? rb.access(d, b, e) : (d.removeEventListener(a, c, !0), rb.remove(d, b))
                }
            }
        }), _.fn.extend({
            on: function(a, b, c, d, e) {
                var f, g;
                if ("object" == typeof a) {
                    "string" != typeof b && (c = c || b, b = void 0);
                    for (g in a)
                        this.on(g, b, c, a[g], e);
                    return this
                }
                if (null == c && null == d ? (d = b, c = b = void 0) : null == d && ("string" == typeof b ? (d = c, c = void 0) : (d = c, c = b, b = void 0)), d===!1)
                    d = k;
                else if (!d)
                    return this;
                return 1 === e && (f = d, d = function(a) {
                    return _().off(a), f.apply(this, arguments)
                }, d.guid = f.guid || (f.guid = _.guid++)), this.each(function() {
                    _.event.add(this, a, d, c, b)
                })
            },
            one: function(a, b, c, d) {
                return this.on(a, b, c, d, 1)
            },
            off: function(a, b, c) {
                var d, e;
                if (a && a.preventDefault && a.handleObj)
                    return d = a.handleObj, _(a.delegateTarget).off(d.namespace ? d.origType + "." + d.namespace : d.origType, d.selector, d.handler), this;
                if ("object" == typeof a) {
                    for (e in a)
                        this.off(e, b, a[e]);
                    return this
                }
                return (b===!1 || "function" == typeof b) && (c = b, b = void 0), c===!1 && (c = k), this.each(function() {
                    _.event.remove(this, a, c, b)
                })
            },
            trigger: function(a, b) {
                return this.each(function() {
                    _.event.trigger(a, b, this)
                })
            },
            triggerHandler: function(a, b) {
                var c = this[0];
                return c ? _.event.trigger(a, b, c, !0) : void 0
            }
        });
        var Eb = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi, Fb = /<([\w:]+)/, Gb = /<|&#?\w+;/, Hb = /<(?:script|style|link)/i, Ib = /checked\s*(?:[^=]|=\s*.checked.)/i, Jb = /^$|\/(?:java|ecma)script/i, Kb = /^true\/(.*)/, Lb = /^\s*<!(?:\[CDATA\[|--)|(?:\]\]|--)>\s*$/g, Mb = {
            option: [1, "<select multiple='multiple'>", "</select>"],
            thead: [1, "<table>", "</table>"],
            col: [2, "<table><colgroup>", "</colgroup></table>"],
            tr: [2, "<table><tbody>", "</tbody></table>"],
            td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
            _default: [0, "", ""]
        };
        Mb.optgroup = Mb.option, Mb.tbody = Mb.tfoot = Mb.colgroup = Mb.caption = Mb.thead, Mb.th = Mb.td, _.extend({
            clone: function(a, b, c) {
                var d, e, f, g, h = a.cloneNode(!0), i = _.contains(a.ownerDocument, a);
                if (!(Y.noCloneChecked || 1 !== a.nodeType && 11 !== a.nodeType || _.isXMLDoc(a)))
                    for (g = r(h), f = r(a), d = 0, e = f.length; e > d; d++)
                        s(f[d], g[d]);
                if (b)
                    if (c)
                        for (f = f || r(a), g = g || r(h), d = 0, e = f.length; e > d; d++)
                            q(f[d], g[d]);
                    else 
                        q(a, h);
                return g = r(h, "script"), g.length > 0 && p(g, !i && r(a, "script")), h
            },
            buildFragment: function(a, b, c, d) {
                for (var e, f, g, h, i, j, k = b.createDocumentFragment(), l = [], m = 0, n = a.length; n > m; m++)
                    if (e = a[m], e || 0 === e)
                        if ("object" === _.type(e))
                            _.merge(l, e.nodeType ? [e] : e);
                        else if (Gb.test(e)) {
                            for (f = f || k.appendChild(b.createElement("div")), g = (Fb.exec(e) || ["", ""])[1].toLowerCase(), h = Mb[g] || Mb._default, f.innerHTML = h[1] + e.replace(Eb, "<$1></$2>") + h[2], j = h[0]; j--;)
                                f = f.lastChild;
                                _.merge(l, f.childNodes), f = k.firstChild, f.textContent = ""
                        } else 
                            l.push(b.createTextNode(e));
                for (k.textContent = "", m = 0; e = l[m++];)
                    if ((!d||-1 === _.inArray(e, d)) && (i = _.contains(e.ownerDocument, e), f = r(k.appendChild(e), "script"), i && p(f), c))
                        for (j = 0; e = f[j++];)
                            Jb.test(e.type || "") && c.push(e);
                return k
            },
            cleanData: function(a) {
                for (var b, c, d, e, f = _.event.special, g = 0; void 0 !== (c = a[g]); g++) {
                    if (_.acceptData(c) && (e = c[rb.expando], e && (b = rb.cache[e]))) {
                        if (b.events)
                            for (d in b.events)
                                f[d] ? _.event.remove(c, d) : _.removeEvent(c, d, b.handle);
                        rb.cache[e] && delete rb.cache[e]
                    }
                    delete sb.cache[c[sb.expando]]
                }
            }
        }), _.fn.extend({
            text: function(a) {
                return qb(this, function(a) {
                    return void 0 === a ? _.text(this) : this.empty().each(function() {
                        (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) && (this.textContent = a)
                    })
                }, null, a, arguments.length)
            },
            append: function() {
                return this.domManip(arguments, function(a) {
                    if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                        var b = m(this, a);
                        b.appendChild(a)
                    }
                })
            },
            prepend: function() {
                return this.domManip(arguments, function(a) {
                    if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                        var b = m(this, a);
                        b.insertBefore(a, b.firstChild)
                    }
                })
            },
            before: function() {
                return this.domManip(arguments, function(a) {
                    this.parentNode && this.parentNode.insertBefore(a, this)
                })
            },
            after: function() {
                return this.domManip(arguments, function(a) {
                    this.parentNode && this.parentNode.insertBefore(a, this.nextSibling)
                })
            },
            remove: function(a, b) {
                for (var c, d = a ? _.filter(a, this) : this, e = 0; null != (c = d[e]); e++)
                    b || 1 !== c.nodeType || _.cleanData(r(c)), c.parentNode && (b && _.contains(c.ownerDocument, c) && p(r(c, "script")), c.parentNode.removeChild(c));
                return this
            },
            empty: function() {
                for (var a, b = 0; null != (a = this[b]); b++)
                    1 === a.nodeType && (_.cleanData(r(a, !1)), a.textContent = "");
                return this
            },
            clone: function(a, b) {
                return a = null == a?!1 : a, b = null == b ? a : b, this.map(function() {
                    return _.clone(this, a, b)
                })
            },
            html: function(a) {
                return qb(this, function(a) {
                    var b = this[0] || {}, c = 0, d = this.length;
                    if (void 0 === a && 1 === b.nodeType)
                        return b.innerHTML;
                    if ("string" == typeof a&&!Hb.test(a)&&!Mb[(Fb.exec(a) || ["", ""])[1].toLowerCase()]) {
                        a = a.replace(Eb, "<$1></$2>");
                        try {
                            for (; d > c; c++)
                                b = this[c] || {}, 1 === b.nodeType && (_.cleanData(r(b, !1)), b.innerHTML = a);
                            b = 0
                        } catch (e) {}
                    }
                    b && this.empty().append(a)
                }, null, a, arguments.length)
            },
            replaceWith: function() {
                var a = arguments[0];
                return this.domManip(arguments, function(b) {
                    a = this.parentNode, _.cleanData(r(this)), a && a.replaceChild(b, this)
                }), a && (a.length || a.nodeType) ? this : this.remove()
            },
            detach: function(a) {
                return this.remove(a, !0)
            },
            domManip: function(a, b) {
                a = S.apply([], a);
                var c, d, e, f, g, h, i = 0, j = this.length, k = this, l = j - 1, m = a[0], p = _.isFunction(m);
                if (p || j > 1 && "string" == typeof m&&!Y.checkClone && Ib.test(m))
                    return this.each(function(c) {
                        var d = k.eq(c);
                        p && (a[0] = m.call(this, c, d.html())), d.domManip(a, b)
                    });
                if (j && (c = _.buildFragment(a, this[0].ownerDocument, !1, this), d = c.firstChild, 1 === c.childNodes.length && (c = d), d)) {
                    for (e = _.map(r(c, "script"), n), f = e.length; j > i; i++)
                        g = c, i !== l && (g = _.clone(g, !0, !0), f && _.merge(e, r(g, "script"))), b.call(this[i], g, i);
                    if (f)
                        for (h = e[e.length - 1].ownerDocument, _.map(e, o), i = 0; f > i; i++)
                            g = e[i], Jb.test(g.type || "")&&!rb.access(g, "globalEval") && _.contains(h, g) && (g.src ? _._evalUrl && _._evalUrl(g.src) : _.globalEval(g.textContent.replace(Lb, "")))
                }
                return this
            }
        }), _.each({
            appendTo: "append",
            prependTo: "prepend",
            insertBefore: "before",
            insertAfter: "after",
            replaceAll: "replaceWith"
        }, function(a, b) {
            _.fn[a] = function(a) {
                for (var c, d = [], e = _(a), f = e.length - 1, g = 0; f >= g; g++)
                    c = g === f ? this : this.clone(!0), _(e[g])[b](c), T.apply(d, c.get());
                return this.pushStack(d)
            }
        });
        var Nb, Ob = {}, Pb = /^margin/, Qb = new RegExp("^(" + vb + ")(?!px)[a-z%]+$", "i"), Rb = function(a) {
            return a.ownerDocument.defaultView.getComputedStyle(a, null)
        };
        !function() {
            function b() {
                g.style.cssText = "-webkit-box-sizing:border-box;-moz-box-sizing:border-box;box-sizing:border-box;display:block;margin-top:1%;top:1%;border:1px;padding:1px;width:4px;position:absolute", g.innerHTML = "", e.appendChild(f);
                var b = a.getComputedStyle(g, null);
                c = "1%" !== b.top, d = "4px" === b.width, e.removeChild(f)
            }
            var c, d, e = Z.documentElement, f = Z.createElement("div"), g = Z.createElement("div");
            g.style && (g.style.backgroundClip = "content-box", g.cloneNode(!0).style.backgroundClip = "", Y.clearCloneStyle = "content-box" === g.style.backgroundClip, f.style.cssText = "border:0;width:0;height:0;top:0;left:-9999px;margin-top:1px;position:absolute", f.appendChild(g), a.getComputedStyle && _.extend(Y, {
                pixelPosition: function() {
                    return b(), c
                },
                boxSizingReliable: function() {
                    return null == d && b(), d
                },
                reliableMarginRight: function() {
                    var b, c = g.appendChild(Z.createElement("div"));
                    return c.style.cssText = g.style.cssText = "-webkit-box-sizing:content-box;-moz-box-sizing:content-box;box-sizing:content-box;display:block;margin:0;border:0;padding:0", c.style.marginRight = c.style.width = "0", g.style.width = "1px", e.appendChild(f), b=!parseFloat(a.getComputedStyle(c, null).marginRight), e.removeChild(f), b
                }
            }))
        }(), _.swap = function(a, b, c, d) {
            var e, f, g = {};
            for (f in b)
                g[f] = a.style[f], a.style[f] = b[f];
            e = c.apply(a, d || []);
            for (f in b)
                a.style[f] = g[f];
            return e
        };
        var Sb = /^(none|table(?!-c[ea]).+)/, Tb = new RegExp("^(" + vb + ")(.*)$", "i"), Ub = new RegExp("^([+-])=(" + vb + ")", "i"), Vb = {
            position: "absolute",
            visibility: "hidden",
            display: "block"
        }, Wb = {
            letterSpacing: "0",
            fontWeight: "400"
        }, Xb = ["Webkit", "O", "Moz", "ms"];
        _.extend({
            cssHooks: {
                opacity: {
                    get: function(a, b) {
                        if (b) {
                            var c = v(a, "opacity");
                            return "" === c ? "1" : c
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
            style: function(a, b, c, d) {
                if (a && 3 !== a.nodeType && 8 !== a.nodeType && a.style) {
                    var e, f, g, h = _.camelCase(b), i = a.style;
                    return b = _.cssProps[h] || (_.cssProps[h] = x(i, h)), g = _.cssHooks[b] || _.cssHooks[h], void 0 === c ? g && "get"in g && void 0 !== (e = g.get(a, !1, d)) ? e : i[b] : (f = typeof c, "string" === f && (e = Ub.exec(c)) && (c = (e[1] + 1) * e[2] + parseFloat(_.css(a, b)), f = "number"), null != c && c === c && ("number" !== f || _.cssNumber[h] || (c += "px"), Y.clearCloneStyle || "" !== c || 0 !== b.indexOf("background") || (i[b] = "inherit"), g && "set"in g && void 0 === (c = g.set(a, c, d)) || (i[b] = c)), void 0)
                }
            },
            css: function(a, b, c, d) {
                var e, f, g, h = _.camelCase(b);
                return b = _.cssProps[h] || (_.cssProps[h] = x(a.style, h)), g = _.cssHooks[b] || _.cssHooks[h], g && "get"in g && (e = g.get(a, !0, c)), void 0 === e && (e = v(a, b, d)), "normal" === e && b in Wb && (e = Wb[b]), "" === c || c ? (f = parseFloat(e), c===!0 || _.isNumeric(f) ? f || 0 : e) : e
            }
        }), _.each(["height", "width"], function(a, b) {
            _.cssHooks[b] = {
                get: function(a, c, d) {
                    return c ? Sb.test(_.css(a, "display")) && 0 === a.offsetWidth ? _.swap(a, Vb, function() {
                        return A(a, b, d)
                    }) : A(a, b, d) : void 0
                },
                set: function(a, c, d) {
                    var e = d && Rb(a);
                    return y(a, c, d ? z(a, b, d, "border-box" === _.css(a, "boxSizing", !1, e), e) : 0)
                }
            }
        }), _.cssHooks.marginRight = w(Y.reliableMarginRight, function(a, b) {
            return b ? _.swap(a, {
                display: "inline-block"
            }, v, [a, "marginRight"]) : void 0
        }), _.each({
            margin: "",
            padding: "",
            border: "Width"
        }, function(a, b) {
            _.cssHooks[a + b] = {
                expand: function(c) {
                    for (var d = 0, e = {}, f = "string" == typeof c ? c.split(" ") : [c]; 4 > d; d++)
                        e[a + wb[d] + b] = f[d] || f[d - 2] || f[0];
                    return e
                }
            }, Pb.test(a) || (_.cssHooks[a + b].set = y)
        }), _.fn.extend({
            css: function(a, b) {
                return qb(this, function(a, b, c) {
                    var d, e, f = {}, g = 0;
                    if (_.isArray(b)) {
                        for (d = Rb(a), e = b.length; e > g; g++)
                            f[b[g]] = _.css(a, b[g], !1, d);
                        return f
                    }
                    return void 0 !== c ? _.style(a, b, c) : _.css(a, b)
                }, a, b, arguments.length > 1)
            },
            show: function() {
                return B(this, !0)
            },
            hide: function() {
                return B(this)
            },
            toggle: function(a) {
                return "boolean" == typeof a ? a ? this.show() : this.hide() : this.each(function() {
                    xb(this) ? _(this).show() : _(this).hide()
                })
            }
        }), _.Tween = C, C.prototype = {
            constructor: C,
            init: function(a, b, c, d, e, f) {
                this.elem = a, this.prop = c, this.easing = e || "swing", this.options = b, this.start = this.now = this.cur(), this.end = d, this.unit = f || (_.cssNumber[c] ? "" : "px")
            },
            cur: function() {
                var a = C.propHooks[this.prop];
                return a && a.get ? a.get(this) : C.propHooks._default.get(this)
            },
            run: function(a) {
                var b, c = C.propHooks[this.prop];
                return this.pos = b = this.options.duration ? _.easing[this.easing](a, this.options.duration * a, 0, 1, this.options.duration) : a, this.now = (this.end - this.start) * b + this.start, this.options.step && this.options.step.call(this.elem, this.now, this), c && c.set ? c.set(this) : C.propHooks._default.set(this), this
            }
        }, C.prototype.init.prototype = C.prototype, C.propHooks = {
            _default: {
                get: function(a) {
                    var b;
                    return null == a.elem[a.prop] || a.elem.style && null != a.elem.style[a.prop] ? (b = _.css(a.elem, a.prop, ""), b && "auto" !== b ? b : 0) : a.elem[a.prop]
                },
                set: function(a) {
                    _.fx.step[a.prop] ? _.fx.step[a.prop](a) : a.elem.style && (null != a.elem.style[_.cssProps[a.prop]] || _.cssHooks[a.prop]) ? _.style(a.elem, a.prop, a.now + a.unit) : a.elem[a.prop] = a.now
                }
            }
        }, C.propHooks.scrollTop = C.propHooks.scrollLeft = {
            set: function(a) {
                a.elem.nodeType && a.elem.parentNode && (a.elem[a.prop] = a.now)
            }
        }, _.easing = {
            linear: function(a) {
                return a
            },
            swing: function(a) {
                return .5 - Math.cos(a * Math.PI) / 2
            }
        }, _.fx = C.prototype.init, _.fx.step = {};
        var Yb, Zb, $b = /^(?:toggle|show|hide)$/, _b = new RegExp("^(?:([+-])=|)(" + vb + ")([a-z%]*)$", "i"), ac = /queueHooks$/, bc = [G], cc = {
            "*": [function(a, b) {
                var c = this.createTween(a, b), d = c.cur(), e = _b.exec(b), f = e && e[3] || (_.cssNumber[a] ? "" : "px"), g = (_.cssNumber[a] || "px" !== f&&+d) && _b.exec(_.css(c.elem, a)), h = 1, i = 20;
                if (g && g[3] !== f) {
                    f = f || g[3], e = e || [], g =+ d || 1;
                    do 
                        h = h || ".5", g/=h, _.style(c.elem, a, g + f);
                    while (h !== (h = c.cur() / d) && 1 !== h&&--i)
                    }
                return e && (g = c.start =+ g||+d || 0, c.unit = f, c.end = e[1] ? g + (e[1] + 1) * e[2] : + e[2]), c
            }
            ]
        };
        _.Animation = _.extend(I, {
            tweener: function(a, b) {
                _.isFunction(a) ? (b = a, a = ["*"]): a = a.split(" ");
                for (var c,
                d = 0,
                e = a.length;
                e > d;
                d++)c = a[d],
                cc[c] = cc[c] || [],
                cc[c].unshift(b)
            }, prefilter : function(a, b) {
                b ? bc.unshift(a) : bc.push(a)
            }
        }), _.speed = function(a, b, c) {
            var d = a && "object" == typeof a ? _.extend({}, a): {
                complete: c ||!c && b || _.isFunction(a) && a,
                duration: a,
                easing: c && b || b&&!_.isFunction(b) && b
            };
            return d.duration = _.fx.off ? 0 : "number" == typeof d.duration ? d.duration : d.duration in _.fx.speeds ? _.fx.speeds[d.duration] : _.fx.speeds._default, (null == d.queue || d.queue===!0) && (d.queue = "fx"), d.old = d.complete, d.complete = function() {
                _.isFunction(d.old) && d.old.call(this), d.queue && _.dequeue(this, d.queue)
            }, d
        }, _.fn.extend({
            fadeTo: function(a, b, c, d) {
                return this.filter(xb).css("opacity", 0).show().end().animate({
                    opacity: b
                }, a, c, d)
            },
            animate: function(a, b, c, d) {
                var e = _.isEmptyObject(a), f = _.speed(b, c, d), g = function() {
                    var b = I(this, _.extend({}, a), f);
                    (e || rb.get(this, "finish")) && b.stop(!0)
                };
                return g.finish = g, e || f.queue===!1 ? this.each(g) : this.queue(f.queue, g)
            },
            stop: function(a, b, c) {
                var d = function(a) {
                    var b = a.stop;
                    delete a.stop, b(c)
                };
                return "string" != typeof a && (c = b, b = a, a = void 0), b && a!==!1 && this.queue(a || "fx", []), this.each(function() {
                    var b=!0, e = null != a && a + "queueHooks", f = _.timers, g = rb.get(this);
                    if (e)
                        g[e] && g[e].stop && d(g[e]);
                    else 
                        for (e in g)
                            g[e] && g[e].stop && ac.test(e) && d(g[e]);
                    for (e = f.length; e--;)
                        f[e].elem !== this || null != a && f[e].queue !== a || (f[e].anim.stop(c), b=!1, f.splice(e, 1));
                    (b ||!c) && _.dequeue(this, a)
                })
            },
            finish: function(a) {
                return a!==!1 && (a = a || "fx"), this.each(function() {
                    var b, c = rb.get(this), d = c[a + "queue"], e = c[a + "queueHooks"], f = _.timers, g = d ? d.length: 0;
                    for (c.finish=!0, _.queue(this, a, []), e && e.stop && e.stop.call(this, !0), b = f.length; b--;)
                        f[b].elem === this && f[b].queue === a && (f[b].anim.stop(!0), f.splice(b, 1));
                    for (b = 0; g > b; b++)
                        d[b] && d[b].finish && d[b].finish.call(this);
                    delete c.finish
                })
            }
        }), _.each(["toggle", "show", "hide"], function(a, b) {
            var c = _.fn[b];
            _.fn[b] = function(a, d, e) {
                return null == a || "boolean" == typeof a ? c.apply(this, arguments) : this.animate(E(b, !0), a, d, e)
            }
        }), _.each({
            slideDown: E("show"),
            slideUp: E("hide"),
            slideToggle: E("toggle"),
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
            _.fn[a] = function(a, c, d) {
                return this.animate(b, a, c, d)
            }
        }), _.timers = [], _.fx.tick = function() {
            var a, b = 0, c = _.timers;
            for (Yb = _.now(); b < c.length; b++)
                a = c[b], a() || c[b] !== a || c.splice(b--, 1);
            c.length || _.fx.stop(), Yb = void 0
        }, _.fx.timer = function(a) {
            _.timers.push(a), a() ? _.fx.start() : _.timers.pop()
        }, _.fx.interval = 13, _.fx.start = function() {
            Zb || (Zb = setInterval(_.fx.tick, _.fx.interval))
        }, _.fx.stop = function() {
            clearInterval(Zb), Zb = null
        }, _.fx.speeds = {
            slow: 600,
            fast: 200,
            _default: 400
        }, _.fn.delay = function(a, b) {
            return a = _.fx ? _.fx.speeds[a] || a : a, b = b || "fx", this.queue(b, function(b, c) {
                var d = setTimeout(b, a);
                c.stop = function() {
                    clearTimeout(d)
                }
            })
        }, function() {
            var a = Z.createElement("input"), b = Z.createElement("select"), c = b.appendChild(Z.createElement("option"));
            a.type = "checkbox", Y.checkOn = "" !== a.value, Y.optSelected = c.selected, b.disabled=!0, Y.optDisabled=!c.disabled, a = Z.createElement("input"), a.value = "t", a.type = "radio", Y.radioValue = "t" === a.value
        }();
        var dc, ec, fc = _.expr.attrHandle;
        _.fn.extend({
            attr: function(a, b) {
                return qb(this, _.attr, a, b, arguments.length > 1)
            },
            removeAttr: function(a) {
                return this.each(function() {
                    _.removeAttr(this, a)
                })
            }
        }), _.extend({
            attr: function(a, b, c) {
                var d, e, f = a.nodeType;
                if (a && 3 !== f && 8 !== f && 2 !== f)
                    return typeof a.getAttribute === zb ? _.prop(a, b, c) : (1 === f && _.isXMLDoc(a) || (b = b.toLowerCase(), d = _.attrHooks[b] || (_.expr.match.bool.test(b) ? ec : dc)), void 0 === c ? d && "get"in d && null !== (e = d.get(a, b)) ? e : (e = _.find.attr(a, b), null == e ? void 0 : e) : null !== c ? d && "set"in d && void 0 !== (e = d.set(a, c, b)) ? e : (a.setAttribute(b, c + ""), c) : void _.removeAttr(a, b))
            },
            removeAttr: function(a, b) {
                var c, d, e = 0, f = b && b.match(nb);
                if (f && 1 === a.nodeType)
                    for (; c = f[e++];)
                        d = _.propFix[c] || c, _.expr.match.bool.test(c) && (a[d]=!1), a.removeAttribute(c)
            },
            attrHooks: {
                type: {
                    set: function(a, b) {
                        if (!Y.radioValue && "radio" === b && _.nodeName(a, "input")) {
                            var c = a.value;
                            return a.setAttribute("type", b), c && (a.value = c), b
                        }
                    }
                }
            }
        }), ec = {
            set: function(a, b, c) {
                return b===!1 ? _.removeAttr(a, c) : a.setAttribute(c, c), c
            }
        }, _.each(_.expr.match.bool.source.match(/\w+/g), function(a, b) {
            var c = fc[b] || _.find.attr;
            fc[b] = function(a, b, d) {
                var e, f;
                return d || (f = fc[b], fc[b] = e, e = null != c(a, b, d) ? b.toLowerCase() : null, fc[b] = f), e
            }
        });
        var gc = /^(?:input|select|textarea|button)$/i;
        _.fn.extend({
            prop: function(a, b) {
                return qb(this, _.prop, a, b, arguments.length > 1)
            },
            removeProp: function(a) {
                return this.each(function() {
                    delete this[_.propFix[a] || a]
                })
            }
        }), _.extend({
            propFix: {
                "for": "htmlFor",
                "class": "className"
            },
            prop: function(a, b, c) {
                var d, e, f, g = a.nodeType;
                if (a && 3 !== g && 8 !== g && 2 !== g)
                    return f = 1 !== g ||!_.isXMLDoc(a), f && (b = _.propFix[b] || b, e = _.propHooks[b]), void 0 !== c ? e && "set"in e && void 0 !== (d = e.set(a, c, b)) ? d : a[b] = c : e && "get"in e && null !== (d = e.get(a, b)) ? d : a[b]
            },
            propHooks: {
                tabIndex: {
                    get: function(a) {
                        return a.hasAttribute("tabindex") || gc.test(a.nodeName) || a.href ? a.tabIndex : - 1
                    }
                }
            }
        }), Y.optSelected || (_.propHooks.selected = {
            get: function(a) {
                var b = a.parentNode;
                return b && b.parentNode && b.parentNode.selectedIndex, null
            }
        }), _.each(["tabIndex", "readOnly", "maxLength", "cellSpacing", "cellPadding", "rowSpan", "colSpan", "useMap", "frameBorder", "contentEditable"], function() {
            _.propFix[this.toLowerCase()] = this
        });
        var hc = /[\t\r\n\f]/g;
        _.fn.extend({
            addClass: function(a) {
                var b, c, d, e, f, g, h = "string" == typeof a && a, i = 0, j = this.length;
                if (_.isFunction(a))
                    return this.each(function(b) {
                        _(this).addClass(a.call(this, b, this.className))
                    });
                if (h)
                    for (b = (a || "").match(nb) || []; j > i; i++)
                        if (c = this[i], d = 1 === c.nodeType && (c.className ? (" " + c.className + " ").replace(hc, " ") : " ")) {
                            for (f = 0; e = b[f++];)
                                d.indexOf(" " + e + " ") < 0 && (d += e + " ");
                                g = _.trim(d), c.className !== g && (c.className = g)
                        }
                return this
            },
            removeClass: function(a) {
                var b, c, d, e, f, g, h = 0 === arguments.length || "string" == typeof a && a, i = 0, j = this.length;
                if (_.isFunction(a))
                    return this.each(function(b) {
                        _(this).removeClass(a.call(this, b, this.className))
                    });
                if (h)
                    for (b = (a || "").match(nb) || []; j > i; i++)
                        if (c = this[i], d = 1 === c.nodeType && (c.className ? (" " + c.className + " ").replace(hc, " ") : "")) {
                            for (f = 0; e = b[f++];)
                                for (; d.indexOf(" " + e + " ") >= 0;)
                                    d = d.replace(" " + e + " ", " ");
                                    g = a ? _.trim(d) : "", c.className !== g && (c.className = g)
                        }
                return this
            },
            toggleClass: function(a, b) {
                var c = typeof a;
                return "boolean" == typeof b && "string" === c ? b ? this.addClass(a) : this.removeClass(a) : this.each(_.isFunction(a) ? function(c) {
                    _(this).toggleClass(a.call(this, c, this.className, b), b)
                } : function() {
                    if ("string" === c)
                        for (var b, d = 0, e = _(this), f = a.match(nb) || []; b = f[d++];)
                            e.hasClass(b) ? e.removeClass(b) : e.addClass(b);
                    else (c === zb || "boolean" === c) 
                        && (this.className && rb.set(this, "__className__", this.className), this.className = this.className || a===!1 ? "" : rb.get(this, "__className__") || "")
                })
            },
            hasClass: function(a) {
                for (var b = " " + a + " ", c = 0, d = this.length; d > c; c++)
                    if (1 === this[c].nodeType && (" " + this[c].className + " ").replace(hc, " ").indexOf(b) >= 0)
                        return !0;
                return !1
            }
        });
        var ic = /\r/g;
        _.fn.extend({
            val: function(a) {
                var b, c, d, e = this[0];
                {
                    if (arguments.length)
                        return d = _.isFunction(a), this.each(function(c) {
                            var e;
                            1 === this.nodeType && (e = d ? a.call(this, c, _(this).val()) : a, null == e ? e = "" : "number" == typeof e ? e += "" : _.isArray(e) && (e = _.map(e, function(a) {
                                return null == a ? "" : a + ""
                            })), b = _.valHooks[this.type] || _.valHooks[this.nodeName.toLowerCase()], b && "set"in b && void 0 !== b.set(this, e, "value") || (this.value = e))
                        });
                    if (e)
                        return b = _.valHooks[e.type] || _.valHooks[e.nodeName.toLowerCase()], b && "get"in b && void 0 !== (c = b.get(e, "value")) ? c : (c = e.value, "string" == typeof c ? c.replace(ic, "") : null == c ? "" : c)
                }
            }
        }), _.extend({
            valHooks: {
                option: {
                    get: function(a) {
                        var b = _.find.attr(a, "value");
                        return null != b ? b : _.trim(_.text(a))
                    }
                },
                select: {
                    get: function(a) {
                        for (var b, c, d = a.options, e = a.selectedIndex, f = "select-one" === a.type || 0 > e, g = f ? null : [], h = f ? e + 1 : d.length, i = 0 > e ? h : f ? e : 0; h > i; i++)
                            if (c = d[i], !(!c.selected && i !== e || (Y.optDisabled ? c.disabled : null !== c.getAttribute("disabled")) || c.parentNode.disabled && _.nodeName(c.parentNode, "optgroup"))) {
                                if (b = _(c).val(), f)
                                    return b;
                                    g.push(b)
                            }
                        return g
                    },
                    set: function(a, b) {
                        for (var c, d, e = a.options, f = _.makeArray(b), g = e.length; g--;)
                            d = e[g], (d.selected = _.inArray(d.value, f) >= 0) && (c=!0);
                        return c || (a.selectedIndex =- 1), f
                    }
                }
            }
        }), _.each(["radio", "checkbox"], function() {
            _.valHooks[this] = {
                set: function(a, b) {
                    return _.isArray(b) ? a.checked = _.inArray(_(a).val(), b) >= 0 : void 0
                }
            }, Y.checkOn || (_.valHooks[this].get = function(a) {
                return null === a.getAttribute("value") ? "on" : a.value
            })
        }), _.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error contextmenu".split(" "), function(a, b) {
            _.fn[b] = function(a, c) {
                return arguments.length > 0 ? this.on(b, null, a, c) : this.trigger(b)
            }
        }), _.fn.extend({
            hover: function(a, b) {
                return this.mouseenter(a).mouseleave(b || a)
            },
            bind: function(a, b, c) {
                return this.on(a, null, b, c)
            },
            unbind: function(a, b) {
                return this.off(a, null, b)
            },
            delegate: function(a, b, c, d) {
                return this.on(b, a, c, d)
            },
            undelegate: function(a, b, c) {
                return 1 === arguments.length ? this.off(a, "**") : this.off(b, a || "**", c)
            }
        });
        var jc = _.now(), kc = /\?/;
        _.parseJSON = function(a) {
            return JSON.parse(a + "")
        }, _.parseXML = function(a) {
            var b, c;
            if (!a || "string" != typeof a)
                return null;
            try {
                c = new DOMParser, b = c.parseFromString(a, "text/xml")
            } catch (d) {
                b = void 0
            }
            return (!b || b.getElementsByTagName("parsererror").length) && _.error("Invalid XML: " + a), b
        };
        var lc, mc, nc = /#.*$/, oc = /([?&])_=[^&]*/, pc = /^(.*?):[ \t]*([^\r\n]*)$/gm, qc = /^(?:about|app|app-storage|.+-extension|file|res|widget):$/, rc = /^(?:GET|HEAD)$/, sc = /^\/\//, tc = /^([\w.+-]+:)(?:\/\/(?:[^\/?#]*@|)([^\/?#:]*)(?::(\d+)|)|)/, uc = {}, vc = {}, wc = "*/".concat("*");
        try {
            mc = location.href
        } catch (xc) {
            mc = Z.createElement("a"), mc.href = "", mc = mc.href
        }
        lc = tc.exec(mc.toLowerCase()) || [], _.extend({
            active: 0,
            lastModified: {},
            etag: {},
            ajaxSettings: {
                url: mc,
                type: "GET",
                isLocal: qc.test(lc[1]),
                global: !0,
                processData: !0,
                async: !0,
                contentType: "application/x-www-form-urlencoded; charset=UTF-8",
                accepts: {
                    "*": wc,
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
                    "text json": _.parseJSON,
                    "text xml": _.parseXML
                },
                flatOptions: {
                    url: !0,
                    context: !0
                }
            },
            ajaxSetup: function(a, b) {
                return b ? L(L(a, _.ajaxSettings), b) : L(_.ajaxSettings, a)
            },
            ajaxPrefilter: J(uc),
            ajaxTransport: J(vc),
            ajax: function(a, b) {
                function c(a, b, c, g) {
                    var i, k, r, s, u, w = b;
                    2 !== t && (t = 2, h && clearTimeout(h), d = void 0, f = g || "", v.readyState = a > 0 ? 4 : 0, i = a >= 200 && 300 > a || 304 === a, c && (s = M(l, v, c)), s = N(l, s, v, i), i ? (l.ifModified && (u = v.getResponseHeader("Last-Modified"), u && (_.lastModified[e] = u), u = v.getResponseHeader("etag"), u && (_.etag[e] = u)), 204 === a || "HEAD" === l.type ? w = "nocontent" : 304 === a ? w = "notmodified" : (w = s.state, k = s.data, r = s.error, i=!r)) : (r = w, (a ||!w) && (w = "error", 0 > a && (a = 0))), v.status = a, v.statusText = (b || w) + "", i ? o.resolveWith(m, [k, w, v]) : o.rejectWith(m, [v, w, r]), v.statusCode(q), q = void 0, j && n.trigger(i ? "ajaxSuccess" : "ajaxError", [v, l, i ? k: r]), p.fireWith(m, [v, w]), j && (n.trigger("ajaxComplete", [v, l]), --_.active || _.event.trigger("ajaxStop")))
                }
                "object" == typeof a && (b = a, a = void 0), b = b || {};
                var d, e, f, g, h, i, j, k, l = _.ajaxSetup({}, b), m = l.context || l, n = l.context && (m.nodeType || m.jquery) ? _(m): _.event, o = _.Deferred(), p = _.Callbacks("once memory"), q = l.statusCode || {}, r = {}, s = {}, t = 0, u = "canceled", v = {
                    readyState: 0,
                    getResponseHeader: function(a) {
                        var b;
                        if (2 === t) {
                            if (!g)
                                for (g = {}; b = pc.exec(f);)
                                    g[b[1].toLowerCase()] = b[2];
                            b = g[a.toLowerCase()]
                        }
                        return null == b ? null : b
                    },
                    getAllResponseHeaders: function() {
                        return 2 === t ? f : null
                    },
                    setRequestHeader: function(a, b) {
                        var c = a.toLowerCase();
                        return t || (a = s[c] = s[c] || a, r[a] = b), this
                    },
                    overrideMimeType: function(a) {
                        return t || (l.mimeType = a), this
                    },
                    statusCode: function(a) {
                        var b;
                        if (a)
                            if (2 > t)
                                for (b in a)
                                    q[b] = [q[b], a[b]];
                            else 
                                v.always(a[v.status]);
                        return this
                    },
                    abort: function(a) {
                        var b = a || u;
                        return d && d.abort(b), c(0, b), this
                    }
                };
                if (o.promise(v).complete = p.add, v.success = v.done, v.error = v.fail, l.url = ((a || l.url || mc) + "").replace(nc, "").replace(sc, lc[1] + "//"), l.type = b.method || b.type || l.method || l.type, l.dataTypes = _.trim(l.dataType || "*").toLowerCase().match(nb) || [""], null == l.crossDomain && (i = tc.exec(l.url.toLowerCase()), l.crossDomain=!(!i || i[1] === lc[1] && i[2] === lc[2] && (i[3] || ("http:" === i[1] ? "80" : "443")) === (lc[3] || ("http:" === lc[1] ? "80" : "443")))), l.data && l.processData && "string" != typeof l.data && (l.data = _.param(l.data, l.traditional)), K(uc, l, b, v), 2 === t)
                    return v;
                j = l.global, j && 0 === _.active++&&_.event.trigger("ajaxStart"), l.type = l.type.toUpperCase(), l.hasContent=!rc.test(l.type), e = l.url, l.hasContent || (l.data && (e = l.url += (kc.test(e) ? "&" : "?") + l.data, delete l.data), l.cache===!1 && (l.url = oc.test(e) ? e.replace(oc, "$1_=" + jc++) : e + (kc.test(e) ? "&" : "?") + "_=" + jc++)), l.ifModified && (_.lastModified[e] && v.setRequestHeader("If-Modified-Since", _.lastModified[e]), _.etag[e] && v.setRequestHeader("If-None-Match", _.etag[e])), (l.data && l.hasContent && l.contentType!==!1 || b.contentType) && v.setRequestHeader("Content-Type", l.contentType), v.setRequestHeader("Accept", l.dataTypes[0] && l.accepts[l.dataTypes[0]] ? l.accepts[l.dataTypes[0]] + ("*" !== l.dataTypes[0] ? ", " + wc + "; q=0.01" : "") : l.accepts["*"]);
                for (k in l.headers)
                    v.setRequestHeader(k, l.headers[k]);
                if (l.beforeSend && (l.beforeSend.call(m, v, l)===!1 || 2 === t))
                    return v.abort();
                u = "abort";
                for (k in{
                    success: 1,
                    error: 1,
                    complete: 1
                })
                    v[k](l[k]);
                if (d = K(vc, l, b, v)) {
                    v.readyState = 1, j && n.trigger("ajaxSend", [v, l]), l.async && l.timeout > 0 && (h = setTimeout(function() {
                        v.abort("timeout")
                    }, l.timeout));
                    try {
                        t = 1, d.send(r, c)
                    } catch (w) {
                        if (!(2 > t))
                            throw w;
                        c( - 1, w)
                    }
                } else 
                    c( - 1, "No Transport");
                return v
            },
            getJSON: function(a, b, c) {
                return _.get(a, b, c, "json")
            },
            getScript: function(a, b) {
                return _.get(a, void 0, b, "script")
            }
        }), _.each(["get", "post"], function(a, b) {
            _[b] = function(a, c, d, e) {
                return _.isFunction(c) && (e = e || d, d = c, c = void 0), _.ajax({
                    url: a,
                    type: b,
                    dataType: e,
                    data: c,
                    success: d
                })
            }
        }), _.each(["ajaxStart", "ajaxStop", "ajaxComplete", "ajaxError", "ajaxSuccess", "ajaxSend"], function(a, b) {
            _.fn[b] = function(a) {
                return this.on(b, a)
            }
        }), _._evalUrl = function(a) {
            return _.ajax({
                url: a,
                type: "GET",
                dataType: "script",
                async: !1,
                global: !1,
                "throws": !0
            })
        }, _.fn.extend({
            wrapAll: function(a) {
                var b;
                return _.isFunction(a) ? this.each(function(b) {
                    _(this).wrapAll(a.call(this, b))
                }) : (this[0] && (b = _(a, this[0].ownerDocument).eq(0).clone(!0), this[0].parentNode && b.insertBefore(this[0]), b.map(function() {
                    for (var a = this; a.firstElementChild;)
                        a = a.firstElementChild;
                    return a
                }).append(this)), this)
            },
            wrapInner: function(a) {
                return this.each(_.isFunction(a) ? function(b) {
                    _(this).wrapInner(a.call(this, b))
                } : function() {
                    var b = _(this), c = b.contents();
                    c.length ? c.wrapAll(a) : b.append(a)
                })
            },
            wrap: function(a) {
                var b = _.isFunction(a);
                return this.each(function(c) {
                    _(this).wrapAll(b ? a.call(this, c) : a)
                })
            },
            unwrap: function() {
                return this.parent().each(function() {
                    _.nodeName(this, "body") || _(this).replaceWith(this.childNodes)
                }).end()
            }
        }), _.expr.filters.hidden = function(a) {
            return a.offsetWidth <= 0 && a.offsetHeight <= 0
        }, _.expr.filters.visible = function(a) {
            return !_.expr.filters.hidden(a)
        };
        var yc = /%20/g, zc = /\[\]$/, Ac = /\r?\n/g, Bc = /^(?:submit|button|image|reset|file)$/i, Cc = /^(?:input|select|textarea|keygen)/i;
        _.param = function(a, b) {
            var c, d = [], e = function(a, b) {
                b = _.isFunction(b) ? b() : null == b ? "" : b, d[d.length] = encodeURIComponent(a) + "=" + encodeURIComponent(b)
            };
            if (void 0 === b && (b = _.ajaxSettings && _.ajaxSettings.traditional), _.isArray(a) || a.jquery&&!_.isPlainObject(a)
                )_.each(a, function() {
                e(this.name, this.value)
            });
            else 
                for (c in a)
                    O(c, a[c], b, e);
            return d.join("&").replace(yc, "+")
        }, _.fn.extend({
            serialize: function() {
                return _.param(this.serializeArray())
            },
            serializeArray: function() {
                return this.map(function() {
                    var a = _.prop(this, "elements");
                    return a ? _.makeArray(a) : this
                }).filter(function() {
                    var a = this.type;
                    return this.name&&!_(this).is(":disabled") && Cc.test(this.nodeName)&&!Bc.test(a) && (this.checked ||!yb.test(a))
                }).map(function(a, b) {
                    var c = _(this).val();
                    return null == c ? null : _.isArray(c) ? _.map(c, function(a) {
                        return {
                            name: b.name,
                            value: a.replace(Ac, "\r\n")
                        }
                    }) : {
                        name: b.name,
                        value: c.replace(Ac, "\r\n")
                    }
                }).get()
            }
        }), _.ajaxSettings.xhr = function() {
            try {
                return new XMLHttpRequest
            } catch (a) {}
        };
        var Dc = 0, Ec = {}, Fc = {
            0: 200,
            1223: 204
        }, Gc = _.ajaxSettings.xhr();
        a.ActiveXObject && _(a).on("unload", function() {
            for (var a in Ec)
                Ec[a]()
        }), Y.cors=!!Gc && "withCredentials"in Gc, Y.ajax = Gc=!!Gc, _.ajaxTransport(function(a) {
            var b;
            return Y.cors || Gc&&!a.crossDomain ? {
                send: function(c, d) {
                    var e, f = a.xhr(), g=++Dc;
                    if (f.open(a.type, a.url, a.async, a.username, a.password), a.xhrFields)
                        for (e in a.xhrFields)
                            f[e] = a.xhrFields[e];
                    a.mimeType && f.overrideMimeType && f.overrideMimeType(a.mimeType), a.crossDomain || c["X-Requested-With"] || (c["X-Requested-With"] = "XMLHttpRequest");
                    for (e in c)
                        f.setRequestHeader(e, c[e]);
                    b = function(a) {
                        return function() {
                            b && (delete Ec[g], b = f.onload = f.onerror = null, "abort" === a ? f.abort() : "error" === a ? d(f.status, f.statusText) : d(Fc[f.status] || f.status, f.statusText, "string" == typeof f.responseText ? {
                                text: f.responseText
                            } : void 0, f.getAllResponseHeaders()))
                        }
                    }, f.onload = b(), f.onerror = b("error"), b = Ec[g] = b("abort");
                    try {
                        f.send(a.hasContent && a.data || null)
                    } catch (h) {
                        if (b)
                            throw h
                    }
                },
                abort: function() {
                    b && b()
                }
            } : void 0
        }), _.ajaxSetup({
            accepts: {
                script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
            },
            contents: {
                script: /(?:java|ecma)script/
            },
            converters: {
                "text script": function(a) {
                    return _.globalEval(a), a
                }
            }
        }), _.ajaxPrefilter("script", function(a) {
            void 0 === a.cache && (a.cache=!1), a.crossDomain && (a.type = "GET")
        }), _.ajaxTransport("script", function(a) {
            if (a.crossDomain) {
                var b, c;
                return {
                    send: function(d, e) {
                        b = _("<script>").prop({
                            async: !0,
                            charset: a.scriptCharset,
                            src: a.url
                        }).on("load error", c = function(a) {
                            b.remove(), c = null, a && e("error" === a.type ? 404 : 200, a.type)
                        }), Z.head.appendChild(b[0])
                    },
                    abort: function() {
                        c && c()
                    }
                }
            }
        });
        var Hc = [], Ic = /(=)\?(?=&|$)|\?\?/;
        _.ajaxSetup({
            jsonp: "callback",
            jsonpCallback: function() {
                var a = Hc.pop() || _.expando + "_" + jc++;
                return this[a]=!0, a
            }
        }), _.ajaxPrefilter("json jsonp", function(b, c, d) {
            var e, f, g, h = b.jsonp!==!1 && (Ic.test(b.url) ? "url" : "string" == typeof b.data&&!(b.contentType || "").indexOf("application/x-www-form-urlencoded") && Ic.test(b.data) && "data");
            return h || "jsonp" === b.dataTypes[0] ? (e = b.jsonpCallback = _.isFunction(b.jsonpCallback) ? b.jsonpCallback() : b.jsonpCallback, h ? b[h] = b[h].replace(Ic, "$1" + e) : b.jsonp!==!1 && (b.url += (kc.test(b.url) ? "&" : "?") + b.jsonp + "=" + e), b.converters["script json"] = function() {
                return g || _.error(e + " was not called"), g[0]
            }, b.dataTypes[0] = "json", f = a[e], a[e] = function() {
                g = arguments
            }, d.always(function() {
                a[e] = f, b[e] && (b.jsonpCallback = c.jsonpCallback, Hc.push(e)), g && _.isFunction(f) && f(g[0]), g = f = void 0
            }), "script") : void 0
        }), _.parseHTML = function(a, b, c) {
            if (!a || "string" != typeof a)
                return null;
            "boolean" == typeof b && (c = b, b=!1), b = b || Z;
            var d = gb.exec(a), e=!c && [];
            return d ? [b.createElement(d[1])] : (d = _.buildFragment([a], b, e), e && e.length && _(e).remove(), _.merge([], d.childNodes))
        };
        var Jc = _.fn.load;
        _.fn.load = function(a, b, c) {
            if ("string" != typeof a && Jc)
                return Jc.apply(this, arguments);
            var d, e, f, g = this, h = a.indexOf(" ");
            return h >= 0 && (d = _.trim(a.slice(h)), a = a.slice(0, h)), _.isFunction(b) ? (c = b, b = void 0) : b && "object" == typeof b && (e = "POST"), g.length > 0 && _.ajax({
                url: a,
                type: e,
                dataType: "html",
                data: b
            }).done(function(a) {
                f = arguments, g.html(d ? _("<div>").append(_.parseHTML(a)).find(d) : a)
            }).complete(c && function(a, b) {
                g.each(c, f || [a.responseText, b, a])
            }), this
        }, _.expr.filters.animated = function(a) {
            return _.grep(_.timers, function(b) {
                return a === b.elem
            }).length
        };
        var Kc = a.document.documentElement;
        _.offset = {
            setOffset: function(a, b, c) {
                var d, e, f, g, h, i, j, k = _.css(a, "position"), l = _(a), m = {};
                "static" === k && (a.style.position = "relative"), h = l.offset(), f = _.css(a, "top"), i = _.css(a, "left"), j = ("absolute" === k || "fixed" === k) && (f + i).indexOf("auto")>-1, j ? (d = l.position(), g = d.top, e = d.left) : (g = parseFloat(f) || 0, e = parseFloat(i) || 0), _.isFunction(b) && (b = b.call(a, c, h)), null != b.top && (m.top = b.top - h.top + g), null != b.left && (m.left = b.left - h.left + e), "using"in b ? b.using.call(a, m) : l.css(m)
            }
        }, _.fn.extend({
            offset: function(a) {
                if (arguments.length)
                    return void 0 === a ? this : this.each(function(b) {
                    _.offset.setOffset(this, a, b)
                });
                var b, c, d = this[0], e = {
                    top: 0,
                    left: 0
                }, f = d && d.ownerDocument;
                if (f)
                    return b = f.documentElement, _.contains(b, d) ? (typeof d.getBoundingClientRect !== zb && (e = d.getBoundingClientRect()), c = P(f), {
                        top: e.top + c.pageYOffset - b.clientTop,
                        left: e.left + c.pageXOffset - b.clientLeft
                    }) : e
            },
            position: function() {
                if (this[0]) {
                    var a, b, c = this[0], d = {
                        top: 0,
                        left: 0
                    };
                    return "fixed" === _.css(c, "position") ? b = c.getBoundingClientRect() : (a = this.offsetParent(), b = this.offset(), _.nodeName(a[0], "html") || (d = a.offset()), d.top += _.css(a[0], "borderTopWidth", !0), d.left += _.css(a[0], "borderLeftWidth", !0)), {
                        top: b.top - d.top - _.css(c, "marginTop", !0),
                        left: b.left - d.left - _.css(c, "marginLeft", !0)
                    }
                }
            },
            offsetParent: function() {
                return this.map(function() {
                    for (var a = this.offsetParent || Kc; a&&!_.nodeName(a, "html") && "static" === _.css(a, "position");)
                        a = a.offsetParent;
                    return a || Kc
                })
            }
        }), _.each({
            scrollLeft: "pageXOffset",
            scrollTop: "pageYOffset"
        }, function(b, c) {
            var d = "pageYOffset" === c;
            _.fn[b] = function(e) {
                return qb(this, function(b, e, f) {
                    var g = P(b);
                    return void 0 === f ? g ? g[c] : b[e] : void(g ? g.scrollTo(d ? a.pageXOffset : f, d ? f : a.pageYOffset) : b[e] = f)
                }, b, e, arguments.length, null)
            }
        }), _.each(["top", "left"], function(a, b) {
            _.cssHooks[b] = w(Y.pixelPosition, function(a, c) {
                return c ? (c = v(a, b), Qb.test(c) ? _(a).position()[b] + "px" : c) : void 0
            })
        }), _.each({
            Height: "height",
            Width: "width"
        }, function(a, b) {
            _.each({
                padding: "inner" + a,
                content: b,
                "": "outer" + a
            }, function(c, d) {
                _.fn[d] = function(d, e) {
                    var f = arguments.length && (c || "boolean" != typeof d), g = c || (d===!0 || e===!0 ? "margin" : "border");
                    return qb(this, function(b, c, d) {
                        var e;
                        return _.isWindow(b) ? b.document.documentElement["client" + a] : 9 === b.nodeType ? (e = b.documentElement, Math.max(b.body["scroll" + a], e["scroll" + a], b.body["offset" + a], e["offset" + a], e["client" + a])) : void 0 === d ? _.css(b, c, g) : _.style(b, c, d, g)
                    }, b, f ? d : void 0, f, null)
                }
            })
        }), _.fn.size = function() {
            return this.length
        }, _.fn.andSelf = _.fn.addBack, "function" == typeof define && define.amd && define("jquery", [], function() {
            return _
        });
        var Lc = a.jQuery, Mc = a.$;
        return _.noConflict = function(b) {
            return a.$ === _ && (a.$ = Mc), b && a.jQuery === _ && (a.jQuery = Lc), _
        }, typeof b === zb && (a.jQuery = a.$ = _), _
    }), function(a) {
        "function" == typeof define && define.amd ? define(["jquery"], a) : a(jQuery)
    }(function(a) {
        function b(a) {
            return h.raw ? a : encodeURIComponent(a)
        }
        function c(a) {
            return h.raw ? a : decodeURIComponent(a)
        }
        function d(a) {
            return b(h.json ? JSON.stringify(a) : String(a))
        }
        function e(a) {
            0 === a.indexOf('"') && (a = a.slice(1, - 1).replace(/\\"/g, '"').replace(/\\\\/g, "\\"));
            try {
                a = decodeURIComponent(a.replace(g, " "))
            } catch (b) {
                return 
            }
            try {
                return h.json ? JSON.parse(a) : a
            } catch (b) {}
        }
        function f(b, c) {
            var d = h.raw ? b: e(b);
            return a.isFunction(c) ? c(d) : d
        }
        var g = /\+/g, h = a.cookie = function(e, g, i) {
            if (void 0 !== g&&!a.isFunction(g)
                ) {
                if (i = a.extend({}, h.defaults, i), "number" == typeof i.expires) {
                    var j = i.expires, k = i.expires = new Date;
                    k.setDate(k.getDate() + j)
                }
                return document.cookie = [b(e), "=", d(g), i.expires ? "; expires=" + i.expires.toUTCString(): "", i.path ? "; path=" + i.path: "", i.domain ? "; domain=" + i.domain: "", i.secure ? "; secure": ""].join("")
            }
            for (var l = e ? void 0 : {}, m = document.cookie ? document.cookie.split("; ") : [], n = 0, o = m.length; o > n; n++) {
                var p = m[n].split("="), q = c(p.shift()), r = p.join("=");
                if (e && e === q) {
                    l = f(r, g);
                    break
                }
                e || void 0 === (r = f(r)) || (l[q] = r)
            }
            return l
        };
        h.defaults = {}, a.removeCookie = function(b, c) {
            return void 0 !== a.cookie(b) ? (a.cookie(b, "", a.extend({}, c, {
                expires: - 1
            })), !0) : !1
        }
    }), User = function(a) {
        this.parser = a || EntitlementParser, this.profileData = null, this.profileDataSet=!1, this.organizationsData = null, this.entitledDesktopAppsData = null, this.sessionStatus = void 0, this.desktopAppsEntitlement = void 0, this.servicesEntitlement = void 0, this.orgRole = void 0, this.entitledDesktopApps = void 0, this.avatarUrl = void 0, this.setProfileData = function(a) {
            this.profileData = a, this.profileDataSet=!0, this.sessionStatus = void 0, this.desktopAppsEntitlement = void 0, this.servicesEntitlement = void 0
        }, this.setAvatarUrl = function(a) {
            this.avatarUrl = a
        }, this.setOrganizationsData = function(a) {
            this.organizationsData = a, this.orgRole = void 0
        }, this.setEntitledDesktopAppsData = function(a) {
            this.entitledDesktopAppsData = a, this.entitledDesktopApps = void 0
        }, this.setPaidEntitlementOverride = function() {
            this.desktopAppsEntitlement = EntitlementParser.DESKTOP_APPS_ENTITLEMENT.PAID, this.servicesEntitlement = EntitlementParser.SERVICES_ENTITLEMENT.PAID
        }, this.setNoEntitlements = function() {
            this.desktopAppsEntitlement = EntitlementParser.DESKTOP_APPS_ENTITLEMENT.NONE, this.servicesEntitlement = EntitlementParser.SERVICES_ENTITLEMENT.NONE, this.orgRole = EntitlementParser.ROLE.NONE, this.entitledDesktopApps = []
        }, this.isUserSignedIn = function() {
            return this.getSessionStatus() == EntitlementParser.SESSION_STATE.SIGNED_IN
        }, this.getSessionStatus = function() {
            return this.sessionStatus ? this.sessionStatus : this.profileData ? (this.sessionStatus = this.parser.parseProfileStatus(this.profileData), this.sessionStatus) : this.profileDataSet ? EntitlementParser.SESSION_STATE.SIGNED_OUT : EntitlementParser.SESSION_STATE.UNKNOWN
        }, this.getAvatarUrl = function() {
            return this.avatarUrl
        }, this.getDesktopAppsEntitlement = function() {
            return this.desktopAppsEntitlement ? this.desktopAppsEntitlement : this.profileData ? (this.desktopAppsEntitlement = this.parser.parseDesktopAppsEntitlement(this.profileData), this.desktopAppsEntitlement) : EntitlementParser.DESKTOP_APPS_ENTITLEMENT.UNKNOWN
        }, this.getServicesEntitlement = function() {
            return this.servicesEntitlement ? this.servicesEntitlement : this.profileData ? (this.servicesEntitlement = this.parser.parseServicesEntitlement(this.profileData), this.servicesEntitlement) : EntitlementParser.SERVICES_ENTITLEMENT.UNKNOWN
        }, this.getOrgRole = function() {
            return this.orgRole ? this.orgRole : this.organizationsData ? (this.orgRole = this.parser.parseOrgRole(this.organizationsData), this.orgRole) : EntitlementParser.ROLE.UNKNOWN
        }, this.getEntitledDesktopApps = function() {
            return this.entitledDesktopApps ? this.entitledDesktopApps : this.entitledDesktopAppsData ? (this.entitledDesktopApps = this.parser.parseEntitledDesktopApps(this.entitledDesktopAppsData), this.entitledDesktopApps) : []
        }, this.isEntitledToDesktopApp = function(a) {
            if (!a)
                return !1;
            var b = this.getSpecialCaseAppEntitlements(), c = this.getEntitledDesktopApps();
            return c.concat(b).indexOf(a)>-1
        }, this.saveMemento = function() {
            return {
                sessionStatus: this.getSessionStatus(),
                desktopAppEntitlement: this.getDesktopAppsEntitlement(),
                servicesEntitlement: this.getServicesEntitlement(),
                orgRole: this.getOrgRole(),
                avatarUrl: this.getAvatarUrl(),
                entitledDesktopApps: this.getEntitledDesktopApps()
            }
        }, this.restoreMemento = function(a) {
            this.sessionStatus = a.sessionStatus, this.desktopAppsEntitlement = a.desktopAppsEntitlement, this.servicesEntitlement = a.servicesEntitlement, this.avatarUrl = a.avatarUrl, this.orgRole = a.orgRole, this.entitledDesktopApps = a.entitledDesktopApps
        }, this.updateLegacyAccountInfo = function(a) {
            a.profile = this.profileData, a.avatar = this.avatarUrl, a.status = this.legacyStatus(), a.entitlement = this.legacyEntitlement(), a.role = this.legacyRole(), a.entitledDesktopApps = this.getEntitledDesktopApps()
        }, this.legacyStatus = function() {
            switch (this.getSessionStatus()) {
            case EntitlementParser.SESSION_STATE.UNKNOWN:
                return "UNKNOWN";
            case EntitlementParser.SESSION_STATE.SIGNED_IN:
                return "SIGNED_IN";
            case EntitlementParser.SESSION_STATE.SIGNED_OUT:
                return "SIGNED_OUT"
            }
        }, this.legacyEntitlement = function() {
            var a = this.getDesktopAppsEntitlement(), b = this.getServicesEntitlement();
            if (a == EntitlementParser.DESKTOP_APPS_ENTITLEMENT.UNKNOWN)
                return "UNKNOWN";
            if (b == EntitlementParser.SERVICES_ENTITLEMENT.UNKNOWN)
                return "UNKNOWN";
            var c = a == EntitlementParser.DESKTOP_APPS_ENTITLEMENT.PAID, d = a == EntitlementParser.DESKTOP_APPS_ENTITLEMENT.FREE, e = b == EntitlementParser.SERVICES_ENTITLEMENT.PAID, f = b == EntitlementParser.SERVICES_ENTITLEMENT.FREE, g = b == EntitlementParser.SERVICES_ENTITLEMENT.NONE;
            return c && g ? "CC_PAID_NO_SERVICES" : c || e ? "CC_PAID" : d || f ? "CC_FREE" : "NONE"
        }, this.legacyRole = function() {
            switch (this.getOrgRole()) {
            case EntitlementParser.ROLE.UNKNOWN:
                return "UNKNOWN";
            case EntitlementParser.ROLE.NONE:
                return "NONE";
            case EntitlementParser.ROLE.TEAM_ADMIN:
                return "TEAM_ADMIN"
            }
        }, this.getSpecialCaseAppEntitlements = function() {
            var a = ["ExtendScriptToolkit", "GamingSdk", "PreludeLiveLogger"];
            return this.getServicesEntitlement() == EntitlementParser.SERVICES_ENTITLEMENT.PAID && (a.push("PhoneGapBuild"), a.push("StoryPlus")), a
        }
    }, EntitlementParser = {
        DESKTOP_APPS_ENTITLEMENT: {
            UNKNOWN: "DESKTOP_APPS_UNKNOWN",
            NONE: "DESKTOP_APPS_NONE",
            FREE: "DESKTOP_APPS_FREE",
            PAID: "DESKTOP_APPS_PAID"
        },
        SERVICES_ENTITLEMENT: {
            UNKNOWN: "SERVICES_UNKNOWN",
            NONE: "SERVICES_NONE",
            FREE: "SERVICES_FREE",
            PAID: "SERVICES_PAID"
        },
        SESSION_STATE: {
            UNKNOWN: "SESSION_UNKNOWN",
            SIGNED_IN: "SESSION_SIGNED_IN",
            SIGNED_OUT: "SESSION_SIGNED_OUT"
        },
        ROLE: {
            UNKNOWN: "ROLE_UNKNOWN",
            NONE: "ROLE_NONE",
            TEAM_ADMIN: "ROLE_TEAM_ADMIN"
        },
        parseProfileStatus: function(a) {
            return a ? a.userId ? this.SESSION_STATE.SIGNED_IN : this.SESSION_STATE.SIGNED_OUT : this.SESSION_STATE.UNKNOWN
        },
        parseDesktopAppsEntitlement: function(a) {
            var b = {
                CS_LVL_1: this.DESKTOP_APPS_ENTITLEMENT.FREE,
                CS_LVL_2: this.DESKTOP_APPS_ENTITLEMENT.PAID
            }, c = {
                cce_complete: this.DESKTOP_APPS_ENTITLEMENT.PAID,
                cce_select: this.DESKTOP_APPS_ENTITLEMENT.PAID,
                cce_single_app: this.DESKTOP_APPS_ENTITLEMENT.PAID
            };
            if (!a)
                return this.DESKTOP_APPS_ENTITLEMENT.UNKNOWN;
            var d = a.serviceAccounts, e = a.projectedProductContext;
            if (e.length > 0)
                for (var f = 0; f < e.length; f++) {
                    var g = e[f].prodCtx, h = g.serviceCode;
                    if ("ACTIVE" === g.statusCode && c[h])
                        return c[h]
                }
            if (d)
                for (var f = 0; f < d.length; f++) {
                    var i = d[f];
                    if ("creative_cloud" === i.serviceCode && "ACTIVE" === i.serviceStatus)
                        return b[i.serviceLevel]
                }
            return this.DESKTOP_APPS_ENTITLEMENT.NONE
        },
        parseServicesEntitlement: function(a) {
            var b = {
                CS_LVL_1: this.SERVICES_ENTITLEMENT.FREE,
                CS_LVL_2: this.SERVICES_ENTITLEMENT.PAID
            };
            if (!a)
                return this.SERVICES_ENTITLEMENT.UNKNOWN;
            var c = a.serviceAccounts;
            if (c)
                for (var d = 0; d < c.length; d++) {
                    var e = c[d];
                    if ("creative_cloud" === e.serviceCode && "ACTIVE" === e.serviceStatus)
                        return b[e.serviceLevel]
                }
            return this.SERVICES_ENTITLEMENT.NONE
        },
        parseOrgRole: function(a) {
            if (!a)
                return this.ROLE.UNKNOWN;
            for (var b, c = a.orgs, d = 0; d < c.length; d++) {
                b = c[d].groups || [];
                for (var e, f = 0; f < b.length; f++)
                    if (e = b[f], "GRP_ADMIN" === e.role)
                        return this.ROLE.TEAM_ADMIN
            }
            return this.ROLE.NONE
        },
        parseEntitledDesktopApps: function(a) {
            if (!a)
                return [];
            for (var b, c = [], d = a.getElementsByTagName("leid"), e = 0; e < d.length; e++) {
                b = d[e].getAttribute("name");
                var f = /(.*{})?(.*)-(.*)-(.*)-GM/g.exec(b);
                if (f) {
                    var g = f[1], h = f[2], i = f[3];
                    g ? c.push(h) : "CS7" == i ? c.push(h) : "7" == i && "ExMan" == h ? c.push(h) : "CS6" == i && "CreativeCloud" == h ? c.push(h) : "CS6" == i && "Fireworks" == h && c.push(h)
                }
            }
            return c
        }
    }, function(a) {
        "use strict";
        function b() {
            $.removeCookie("account-info-cache", {
                path: "/"
            })
        }
        function c() {
            var a = [];
            a.push(h()), a.push(f()), a.push(g()), $.when.apply($, a).then(d, e)
        }
        function d() {
            n.resolve()
        }
        function e(a) {
            n.reject(a)
        }
        function f() {
            var a = $.Deferred();
            return i.on("gnav:avatar", function b(c) {
                i.off("gnav:avatar", b), l.setAvatarUrl(c), n.notify(k.AVATAR_LOADED), a.resolve()
            }), a.promise()
        }
        function g() {
            var b = $.Deferred();
            return $.ajax({
                url: "https://adobeid-na1" + a.adobe.dom.globalNavVars.ImsJS + ".services.adobe.com/ims/organizations",
                data: {
                    client_id: a.adobeid.client_id,
                    bearer_token: a.adobeIMS.getAccessToken()
                },
                dataType: "jsonp",
                success: function(a) {
                    return a.error ? b.reject(a.error) : (l.setOrganizationsData(a), n.notify(k.ORGS_LOADED), void b.resolve())
                },
                error: function(a, c, d) {
                    b.reject({
                        error: c,
                        error_description: d
                    })
                }
            }), b.promise()
        }
        function h() {
            var b = $.Deferred(), c = a.location.hostname.indexOf("www.adobe.com")>-1 ? "https://lm.licenses.adobe.com": "https://na2m-stg2.licenses.adobe.com", d = "<Request><pguid>" + m.profile.userId + "</pguid><AccessToken>" + a.adobeIMS.getAccessToken() + "</AccessToken></Request>";
            return $.ajax({
                url: c + "/aents/aents/v1/entitlements",
                type: "POST",
                data: d,
                contentType: "application/xml",
                success: function(a) {
                    l.setEntitledDesktopAppsData(a), n.notify(k.ENTITLED_DESKTOP_APPS_LOADED), b.resolve()
                },
                error: function(a, c, d) {
                    b.reject({
                        error: c,
                        error_description: d
                    })
                }
            }), b.promise()
        }
        var i = a.adobe.events, j = a.adobe.account = {}, k = (j.Entitlement = {
            UNKNOWN: "UNKNOWN",
            NONE: "NONE",
            CC_FREE: "CC_FREE",
            CC_PAID: "CC_PAID",
            CC_PAID_NO_SERVICES: "CC_PAID_NO_SERVICES"
        }, j.Status = {
            UNKNOWN: "UNKNOWN",
            SIGNED_IN: "SIGNED_IN",
            SIGNED_OUT: "SIGNED_OUT"
        }, j.Role = {
            UNKNOWN: "UNKNOWN",
            NONE: "NONE",
            TEAM_ADMIN: "TEAM_ADMIN"
        }, j.LoadingProgress = {
            CACHE_LOADED: "CACHE_LOADED",
            PROFILE_LOADED: "PROFILE_LOADED",
            ENTITLEMENTS_LOADED: "ENTITLEMENTS_LOADED",
            ENTITLED_DESKTOP_APPS_LOADED: "ENTITLED_DESKTOP_APPS_LOADED",
            AVATAR_LOADED: "AVATAR_LOADED",
            ORGS_LOADED: "ORGS_LOADED"
        }), l = new User, m = j.info = {};
        l.updateLegacyAccountInfo(m);
        var n = $.Deferred();
        n.promise(m);
        var o = $.cookie("account-info-cache");
        if (o) {
            var p = JSON.parse(o);
            l.restoreMemento(p), $.cookie("recent-purchases") && l.setPaidEntitlementOverride(), n.notify(k.CACHE_LOADED)
        }
        i.on("gnav:signOut", b), i.on("ims:onProfile", function q(a) {
            i.off("ims:onProfile", q), l.setProfileData(a), n.notify(k.PROFILE_LOADED), l.isUserSignedIn() ? ($.cookie("recent-purchases") && l.setPaidEntitlementOverride(), n.notify(k.ENTITLEMENTS_LOADED), c()) : (l.setNoEntitlements(), n.notify(k.ENTITLEMENTS_LOADED), n.notify(k.ENTITLED_DESKTOP_APPS_LOADED), n.notify(k.ORGS_LOADED), n.resolve())
        }), m.progress(function(a) {
            var b = $("body"), c = a === j.LoadingProgress.CACHE_LOADED, d = function(a) {
                var b = new RegExp("\\b" + a + "\\S+", "g");
                return function(a, c) {
                    return (c.match(b) || []).join(" ")
                }
            };
            if (l.updateLegacyAccountInfo(j.info), c || a === j.LoadingProgress.PROFILE_LOADED) {
                var e = m.status || j.Status.UNKNOWN;
                b.removeClass(d("account-status--")), b.addClass("account-status--" + e.toLowerCase())
            }
            if (c || a === j.LoadingProgress.ENTITLEMENTS_LOADED) {
                var f = m.entitlement || j.Entitlement.UNKNOWN;
                b.removeClass(d("account-entitlement--")), b.addClass("account-entitlement--" + f.toLowerCase())
            }
            if (c || a === j.LoadingProgress.ENTITLED_DESKTOP_APPS_LOADED) {
                var g = $(".dom-stateful-download-link");
                g.length > 0 && $.each(g, function(a, b) {
                    var c = $(b).data("leid-product-name");
                    l.isEntitledToDesktopApp(c) && ("cc-desktopapps" === $("body").data("dom-page") ? $(b).parent().parent().addClass("account-entitled-product") : $(b).addClass("account-entitled-product"))
                })
            }
            if (c || a === j.LoadingProgress.ORGS_LOADED) {
                var h = m.role || j.Role.NONE;
                b.removeClass(d("account-role--")), $("body").addClass("account-role--" + h.toLowerCase())
            }
        }), m.done(function() {
            if (l.isUserSignedIn()) {
                var a = l.saveMemento();
                $.cookie("account-info-cache", JSON.stringify(a), {
                    path: "/"
                })
            } else 
                b()
        })
    }(window, document), !function() {
        "use strict";
        function a(a, b) {
            a.s = null !== b && void 0 !== b ? "string" == typeof b ? b : b.toString() : b, a.orig = b, null !== b && void 0 !== b ? a.__defineGetter__ ? a.__defineGetter__("length", function() {
                return a.s.length
            }) : a.length = b.length : a.length =- 1
        }
        function b(b) {
            a(this, b)
        }
        function c() {
            for (var a in l)
                !function(a) {
                    var b = l[a];
                    k.hasOwnProperty(a) || (m.push(a), k[a] = function() {
                        return String.prototype.s = this, b.apply(this, arguments)
                    })
                }(a)
        }
        function d() {
            for (var a = 0; a < m.length; ++a)
                delete String.prototype[m[a]];
            m.length = 0
        }
        function e() {
            for (var a = f(), b = {}, c = 0; c < a.length; ++c) {
                var d = a[c], e = k[d];
                try {
                    var g = typeof e.apply("teststring", []);
                    b[d] = g
                } catch (h) {}
            }
            return b
        }
        function f() {
            var a = [];
            if (Object.getOwnPropertyNames)
                return a = Object.getOwnPropertyNames(k), a.splice(a.indexOf("valueOf"), 1), a.splice(a.indexOf("toString"), 1), a;
            var b = {};
            for (var c in String.prototype)
                b[c] = c;
            for (var c in Object.prototype)
                delete b[c];
            for (var c in b)
                a.push(c);
            return a
        }
        function g(a) {
            return new b(a)
        }
        function h(a, b) {
            var c, d = [];
            for (c = 0; c < a.length; c++)
                d.push(a[c]), b && b.call(a, a[c], c);
            return d
        }
        var i = "1.8.0", j = {}, k = String.prototype, l = b.prototype = {
            between: function(a, b) {
                var c = this.s, d = c.indexOf(a), e = c.indexOf(b, d + a.length);
                return new this.constructor( - 1 == e && null != b ? "" : - 1 == e && null == b ? c.substring(d + a.length) : c.slice(d + a.length, e))
            },
            camelize: function() {
                var a = this.trim().s.replace(/(\-|_|\s)+(.)?/g, function(a, b, c) {
                    return c ? c.toUpperCase() : ""
                });
                return new this.constructor(a)
            },
            capitalize: function() {
                return new this.constructor(this.s.substr(0, 1).toUpperCase() + this.s.substring(1).toLowerCase())
            },
            charAt: function(a) {
                return this.s.charAt(a)
            },
            chompLeft: function(a) {
                var b = this.s;
                return 0 === b.indexOf(a) ? (b = b.slice(a.length), new this.constructor(b)) : this
            },
            chompRight: function(a) {
                if (this.endsWith(a)) {
                    var b = this.s;
                    return b = b.slice(0, b.length - a.length), new this.constructor(b)
                }
                return this
            },
            collapseWhitespace: function() {
                var a = this.s.replace(/[\s\xa0]+/g, " ").replace(/^\s+|\s+$/g, "");
                return new this.constructor(a)
            },
            contains: function(a) {
                return this.s.indexOf(a) >= 0
            },
            count: function(a) {
                for (var b = 0, c = this.s.indexOf(a); c >= 0;)
                    b += 1, c = this.s.indexOf(a, c + 1);
                return b
            },
            dasherize: function() {
                var a = this.trim().s.replace(/[_\s]+/g, "-").replace(/([A-Z])/g, "-$1").replace(/-+/g, "-").toLowerCase();
                return new this.constructor(a)
            },
            decodeHtmlEntities: function() {
                var a = this.s;
                return a = a.replace(/&#(\d+);?/g, function(a, b) {
                    return String.fromCharCode(b)
                }).replace(/&#[xX]([A-Fa-f0-9]+);?/g, function(a, b) {
                    return String.fromCharCode(parseInt(b, 16))
                }).replace(/&([^;\W]+;?)/g, function(a, b) {
                    var c = b.replace(/;$/, ""), d = j[b] || b.match(/;$/) && j[c];
                    return "number" == typeof d ? String.fromCharCode(d) : "string" == typeof d ? d : a
                }), new this.constructor(a)
            },
            endsWith: function(a) {
                var b = this.s.length - a.length;
                return b >= 0 && this.s.indexOf(a, b) === b
            },
            escapeHTML: function() {
                return new this.constructor(this.s.replace(/[&<>"']/g, function(a) {
                    return "&" + q[a] + ";"
                }))
            },
            ensureLeft: function(a) {
                var b = this.s;
                return 0 === b.indexOf(a) ? this : new this.constructor(a + b)
            },
            ensureRight: function(a) {
                var b = this.s;
                return this.endsWith(a) ? this : new this.constructor(b + a)
            },
            humanize: function() {
                if (null === this.s || void 0 === this.s)
                    return new this.constructor("");
                var a = this.underscore().replace(/_id$/, "").replace(/_/g, " ").trim().capitalize();
                return new this.constructor(a)
            },
            isAlpha: function() {
                return !/[^a-z\xC0-\xFF]/.test(this.s.toLowerCase())
            },
            isAlphaNumeric: function() {
                return !/[^0-9a-z\xC0-\xFF]/.test(this.s.toLowerCase())
            },
            isEmpty: function() {
                return null === this.s || void 0 === this.s?!0 : /^[\s\xa0]*$/.test(this.s)
            },
            isLower: function() {
                return this.isAlpha() && this.s.toLowerCase() === this.s
            },
            isNumeric: function() {
                return !/[^0-9]/.test(this.s)
            },
            isUpper: function() {
                return this.isAlpha() && this.s.toUpperCase() === this.s
            },
            left: function(a) {
                if (a >= 0) {
                    var b = this.s.substr(0, a);
                    return new this.constructor(b)
                }
                return this.right( - a)
            },
            lines: function() {
                return this.replaceAll("\r\n", "\n").s.split("\n")
            },
            pad: function(a, b) {
                if (null == b && (b = " "), this.s.length >= a)
                    return new this.constructor(this.s);
                a -= this.s.length;
                var c = Array(Math.ceil(a / 2) + 1).join(b), d = Array(Math.floor(a / 2) + 1).join(b);
                return new this.constructor(c + this.s + d)
            },
            padLeft: function(a, b) {
                return null == b && (b = " "), new this.constructor(this.s.length >= a ? this.s : Array(a - this.s.length + 1).join(b) + this.s)
            },
            padRight: function(a, b) {
                return null == b && (b = " "), new this.constructor(this.s.length >= a ? this.s : this.s + Array(a - this.s.length + 1).join(b))
            },
            parseCSV: function(a, b, c, d) {
                a = a || ",", c = c || "\\", "undefined" == typeof b && (b = '"');
                var e = 0, f = [], g = [], h = this.s.length, i=!1, j = this, k = function(a) {
                    return j.s.charAt(a)
                };
                if ("undefined" != typeof d)
                    var l = [];
                for (b || (i=!0); h > e;) {
                    var m = k(e);
                    switch (m) {
                    case c:
                        if (i && (c !== b || k(e + 1) === b)) {
                            e += 1, f.push(k(e));
                            break
                        }
                        if (c !== b)
                            break;
                    case b:
                        i=!i;
                        break;
                    case a:
                        i && b ? f.push(m) : (g.push(f.join("")), f.length = 0);
                        break;
                    case d:
                        i ? f.push(m) : l && (g.push(f.join("")), l.push(g), g = [], f.length = 0);
                        break;
                    default:
                        i && f.push(m)
                    }
                    e += 1
                }
                return g.push(f.join("")), l ? (l.push(g), l) : g
            },
            replaceAll: function(a, b) {
                var c = this.s.split(a).join(b);
                return new this.constructor(c)
            },
            right: function(a) {
                if (a >= 0) {
                    var b = this.s.substr(this.s.length - a, a);
                    return new this.constructor(b)
                }
                return this.left( - a)
            },
            setValue: function(b) {
                return a(this, b), this
            },
            slugify: function() {
                var a = new b(this.s.replace(/[^\w\s-]/g, "").toLowerCase()).dasherize().s;
                return "-" === a.charAt(0) && (a = a.substr(1)), new this.constructor(a)
            },
            startsWith: function(a) {
                return 0 === this.s.lastIndexOf(a, 0)
            },
            stripPunctuation: function() {
                return new this.constructor(this.s.replace(/[^\w\s]|_/g, "").replace(/\s+/g, " "))
            },
            stripTags: function() {
                var a = this.s, b = arguments.length > 0 ? arguments: [""];
                return h(b, function(b) {
                    a = a.replace(RegExp("</?" + b + "[^<>]*>", "gi"), "")
                }), new this.constructor(a)
            },
            template: function(a, b, c) {
                var d = this.s, b = b || g.TMPL_OPEN, c = c || g.TMPL_CLOSE, e = b.replace(/[-[\]()*\s]/g, "\\$&").replace(/\$/g, "\\$"), f = c.replace(/[-[\]()*\s]/g, "\\$&").replace(/\$/g, "\\$"), h = new RegExp(e + "(.+?)" + f, "g"), i = d.match(h) || [];
                return i.forEach(function(e) {
                    var f = e.substring(b.length, e.length - c.length);
                    "undefined" != typeof a[f] && (d = d.replace(e, a[f]))
                }), new this.constructor(d)
            },
            times: function(a) {
                return new this.constructor(new Array(a + 1).join(this.s))
            },
            toBoolean: function() {
                if ("string" == typeof this.orig) {
                    var a = this.s.toLowerCase();
                    return "true" === a || "yes" === a || "on" === a
                }
                return this.orig===!0 || 1 === this.orig
            },
            toFloat: function(a) {
                var b = parseFloat(this.s);
                return a ? parseFloat(b.toFixed(a)) : b
            },
            toInt: function() {
                return /^\s*-?0x/i.test(this.s) ? parseInt(this.s, 16) : parseInt(this.s, 10)
            },
            trim: function() {
                var a;
                return a = "undefined" == typeof k.trim ? this.s.replace(/(^\s*|\s*$)/g, "") : this.s.trim(), new this.constructor(a)
            },
            trimLeft: function() {
                var a;
                return a = k.trimLeft ? this.s.trimLeft() : this.s.replace(/(^\s*)/g, ""), new this.constructor(a)
            },
            trimRight: function() {
                var a;
                return a = k.trimRight ? this.s.trimRight() : this.s.replace(/\s+$/, ""), new this.constructor(a)
            },
            truncate: function(a, c) {
                var d = this.s;
                if (a=~~a, c = c || "...", d.length <= a)
                    return new this.constructor(d);
                var e = function(a) {
                    return a.toUpperCase() !== a.toLowerCase() ? "A" : " "
                }, f = d.slice(0, a + 1).replace(/.(?=\W*\w*$)/g, e);
                return f = f.slice(f.length - 2).match(/\w\w/) ? f.replace(/\s*\S+$/, "") : new b(f.slice(0, f.length - 1)).trimRight().s, new b((f + c).length > d.length ? d : d.slice(0, f.length) + c)
            },
            toCSV: function() {
                function a(a) {
                    return null !== a && "" !== a
                }
                var c = ",", d = '"', e = "\\", f=!0, g=!1, h = [];
                if ("object" == typeof arguments[0] ? (c = arguments[0].delimiter || c, c = arguments[0].separator || c, d = arguments[0].qualifier || d, f=!!arguments[0].encloseNumbers, e = arguments[0].escape || e, g=!!arguments[0].keys) : "string" == typeof arguments[0] && (c = arguments[0]), "string" == typeof arguments[1] && (d = arguments[1]), null === arguments[1] && (d = null), this.orig instanceof Array)
                    h = this.orig;
                else 
                    for (var i in this.orig)
                        this.orig.hasOwnProperty(i) && h.push(g ? i : this.orig[i]);
                for (var j = e + d, k = [], l = 0; l < h.length; ++l) {
                    var m = a(d);
                    if ("number" == typeof h[l] && (m&=f), m && k.push(d), null !== h[l] && void 0 !== h[l]) {
                        var n = new b(h[l]).replaceAll(d, j).s;
                        k.push(n)
                    } else 
                        k.push("");
                    m && k.push(d), c && k.push(c)
                }
                return k.length = k.length - 1, new this.constructor(k.join(""))
            },
            toString: function() {
                return this.s
            },
            underscore: function() {
                var a = this.trim().s.replace(/([a-z\d])([A-Z]+)/g, "$1_$2").replace(/[-\s]+/g, "_").toLowerCase();
                return new b(this.s.charAt(0)).isUpper() && (a = "_" + a), new this.constructor(a)
            },
            unescapeHTML: function() {
                return new this.constructor(this.s.replace(/\&([^;]+);/g, function(a, b) {
                    var c;
                    return b in p ? p[b] : (c = b.match(/^#x([\da-fA-F]+)$/)) ? String.fromCharCode(parseInt(c[1], 16)) : (c = b.match(/^#(\d+)$/)) ? String.fromCharCode(~~c[1]) : a
                }))
            },
            valueOf: function() {
                return this.s.valueOf()
            }
        }, m = [], n = e();
        for (var o in n)
            !function(a) {
                var b = k[a];
                "function" == typeof b && (l[a] || (l[a] = "string" === n[a] ? function() {
                    return new this.constructor(b.apply(this, arguments))
                } : b))
            }(o);
        l.repeat = l.times, l.include = l.contains, l.toInteger = l.toInt, l.toBool = l.toBoolean, l.decodeHTMLEntities = l.decodeHtmlEntities, l.constructor = b, g.extendPrototype = c, g.restorePrototype = d, g.VERSION = i, g.TMPL_OPEN = "{{", g.TMPL_CLOSE = "}}", g.ENTITIES = j, "undefined" != typeof module && "undefined" != typeof module.exports ? module.exports = g : "function" == typeof define && define.amd ? define([], function() {
            return g
        }) : window.S = g;
        var p = {
            lt: "<",
            gt: ">",
            quot: '"',
            apos: "'",
            amp: "&"
        }, q = {};
        for (var r in p)
            q[p[r]] = r;
        j = {
            amp: "&",
            gt: ">",
            lt: "<",
            quot: '"',
            apos: "'",
            AElig: 198,
            Aacute: 193,
            Acirc: 194,
            Agrave: 192,
            Aring: 197,
            Atilde: 195,
            Auml: 196,
            Ccedil: 199,
            ETH: 208,
            Eacute: 201,
            Ecirc: 202,
            Egrave: 200,
            Euml: 203,
            Iacute: 205,
            Icirc: 206,
            Igrave: 204,
            Iuml: 207,
            Ntilde: 209,
            Oacute: 211,
            Ocirc: 212,
            Ograve: 210,
            Oslash: 216,
            Otilde: 213,
            Ouml: 214,
            THORN: 222,
            Uacute: 218,
            Ucirc: 219,
            Ugrave: 217,
            Uuml: 220,
            Yacute: 221,
            aacute: 225,
            acirc: 226,
            aelig: 230,
            agrave: 224,
            aring: 229,
            atilde: 227,
            auml: 228,
            ccedil: 231,
            eacute: 233,
            ecirc: 234,
            egrave: 232,
            eth: 240,
            euml: 235,
            iacute: 237,
            icirc: 238,
            igrave: 236,
            iuml: 239,
            ntilde: 241,
            oacute: 243,
            ocirc: 244,
            ograve: 242,
            oslash: 248,
            otilde: 245,
            ouml: 246,
            szlig: 223,
            thorn: 254,
            uacute: 250,
            ucirc: 251,
            ugrave: 249,
            uuml: 252,
            yacute: 253,
            yuml: 255,
            copy: 169,
            reg: 174,
            nbsp: 160,
            iexcl: 161,
            cent: 162,
            pound: 163,
            curren: 164,
            yen: 165,
            brvbar: 166,
            sect: 167,
            uml: 168,
            ordf: 170,
            laquo: 171,
            not: 172,
            shy: 173,
            macr: 175,
            deg: 176,
            plusmn: 177,
            sup1: 185,
            sup2: 178,
            sup3: 179,
            acute: 180,
            micro: 181,
            para: 182,
            middot: 183,
            cedil: 184,
            ordm: 186,
            raquo: 187,
            frac14: 188,
            frac12: 189,
            frac34: 190,
            iquest: 191,
            times: 215,
            divide: 247,
            "OElig;": 338,
            "oelig;": 339,
            "Scaron;": 352,
            "scaron;": 353,
            "Yuml;": 376,
            "fnof;": 402,
            "circ;": 710,
            "tilde;": 732,
            "Alpha;": 913,
            "Beta;": 914,
            "Gamma;": 915,
            "Delta;": 916,
            "Epsilon;": 917,
            "Zeta;": 918,
            "Eta;": 919,
            "Theta;": 920,
            "Iota;": 921,
            "Kappa;": 922,
            "Lambda;": 923,
            "Mu;": 924,
            "Nu;": 925,
            "Xi;": 926,
            "Omicron;": 927,
            "Pi;": 928,
            "Rho;": 929,
            "Sigma;": 931,
            "Tau;": 932,
            "Upsilon;": 933,
            "Phi;": 934,
            "Chi;": 935,
            "Psi;": 936,
            "Omega;": 937,
            "alpha;": 945,
            "beta;": 946,
            "gamma;": 947,
            "delta;": 948,
            "epsilon;": 949,
            "zeta;": 950,
            "eta;": 951,
            "theta;": 952,
            "iota;": 953,
            "kappa;": 954,
            "lambda;": 955,
            "mu;": 956,
            "nu;": 957,
            "xi;": 958,
            "omicron;": 959,
            "pi;": 960,
            "rho;": 961,
            "sigmaf;": 962,
            "sigma;": 963,
            "tau;": 964,
            "upsilon;": 965,
            "phi;": 966,
            "chi;": 967,
            "psi;": 968,
            "omega;": 969,
            "thetasym;": 977,
            "upsih;": 978,
            "piv;": 982,
            "ensp;": 8194,
            "emsp;": 8195,
            "thinsp;": 8201,
            "zwnj;": 8204,
            "zwj;": 8205,
            "lrm;": 8206,
            "rlm;": 8207,
            "ndash;": 8211,
            "mdash;": 8212,
            "lsquo;": 8216,
            "rsquo;": 8217,
            "sbquo;": 8218,
            "ldquo;": 8220,
            "rdquo;": 8221,
            "bdquo;": 8222,
            "dagger;": 8224,
            "Dagger;": 8225,
            "bull;": 8226,
            "hellip;": 8230,
            "permil;": 8240,
            "prime;": 8242,
            "Prime;": 8243,
            "lsaquo;": 8249,
            "rsaquo;": 8250,
            "oline;": 8254,
            "frasl;": 8260,
            "euro;": 8364,
            "image;": 8465,
            "weierp;": 8472,
            "real;": 8476,
            "trade;": 8482,
            "alefsym;": 8501,
            "larr;": 8592,
            "uarr;": 8593,
            "rarr;": 8594,
            "darr;": 8595,
            "harr;": 8596,
            "crarr;": 8629,
            "lArr;": 8656,
            "uArr;": 8657,
            "rArr;": 8658,
            "dArr;": 8659,
            "hArr;": 8660,
            "forall;": 8704,
            "part;": 8706,
            "exist;": 8707,
            "empty;": 8709,
            "nabla;": 8711,
            "isin;": 8712,
            "notin;": 8713,
            "ni;": 8715,
            "prod;": 8719,
            "sum;": 8721,
            "minus;": 8722,
            "lowast;": 8727,
            "radic;": 8730,
            "prop;": 8733,
            "infin;": 8734,
            "ang;": 8736,
            "and;": 8743,
            "or;": 8744,
            "cap;": 8745,
            "cup;": 8746,
            "int;": 8747,
            "there4;": 8756,
            "sim;": 8764,
            "cong;": 8773,
            "asymp;": 8776,
            "ne;": 8800,
            "equiv;": 8801,
            "le;": 8804,
            "ge;": 8805,
            "sub;": 8834,
            "sup;": 8835,
            "nsub;": 8836,
            "sube;": 8838,
            "supe;": 8839,
            "oplus;": 8853,
            "otimes;": 8855,
            "perp;": 8869,
            "sdot;": 8901,
            "lceil;": 8968,
            "rceil;": 8969,
            "lfloor;": 8970,
            "rfloor;": 8971,
            "lang;": 9001,
            "rang;": 9002,
            "loz;": 9674,
            "spades;": 9824,
            "clubs;": 9827,
            "hearts;": 9829,
            "diams;": 9830
        }
    }.call(this);
    var d = navigator.userAgent.indexOf("Android") > 0, e = /iP(ad|hone|od)/.test(navigator.userAgent), f = e && /OS 4_\d(_\d)?/.test(navigator.userAgent), g = e && /OS ([6-9]|\d{2})_\d/.test(navigator.userAgent);
    c.prototype.needsClick = function(a) {
        "use strict";
        switch (a.nodeName.toLowerCase()) {
        case"button":
        case"select":
        case"textarea":
            if (a.disabled)
                return !0;
            break;
        case"input":
            if (e && "file" === a.type || a.disabled)
                return !0;
            break;
        case"label":
        case"video":
            return !0
        }
        return /\bneedsclick\b/.test(a.className)
    }, c.prototype.needsFocus = function(a) {
        "use strict";
        switch (a.nodeName.toLowerCase()) {
        case"textarea":
            return !0;
        case"select":
            return !d;
        case"input":
            switch (a.type) {
            case"button":
            case"checkbox":
            case"file":
            case"image":
            case"radio":
            case"submit":
                return !1
            }
            return !a.disabled&&!a.readOnly;
        default:
            return /\bneedsfocus\b/.test(a.className)
        }
    }, c.prototype.sendClick = function(a, b) {
        "use strict";
        var c, d;
        document.activeElement && document.activeElement !== a && document.activeElement.blur(), d = b.changedTouches[0], c = document.createEvent("MouseEvents"), c.initMouseEvent(this.determineEventType(a), !0, !0, window, 1, d.screenX, d.screenY, d.clientX, d.clientY, !1, !1, !1, !1, 0, null), c.forwardedTouchEvent=!0, a.dispatchEvent(c)
    }, c.prototype.determineEventType = function(a) {
        "use strict";
        return d && "select" === a.tagName.toLowerCase() ? "mousedown" : "click"
    }, c.prototype.focus = function(a) {
        "use strict";
        var b;
        e && a.setSelectionRange && 0 !== a.type.indexOf("date") && "time" !== a.type ? (b = a.value.length, a.setSelectionRange(b, b)) : a.focus()
    }, c.prototype.updateScrollParent = function(a) {
        "use strict";
        var b, c;
        if (b = a.fastClickScrollParent, !b ||!b.contains(a)) {
            c = a;
            do {
                if (c.scrollHeight > c.offsetHeight) {
                    b = c, a.fastClickScrollParent = c;
                    break
                }
                c = c.parentElement
            }
            while (c)
            }
        b && (b.fastClickLastScrollTop = b.scrollTop)
    }, c.prototype.getTargetElementFromEventTarget = function(a) {
        "use strict";
        return a.nodeType === Node.TEXT_NODE ? a.parentNode : a
    }, c.prototype.onTouchStart = function(a) {
        "use strict";
        var b, c, d;
        if (a.targetTouches.length > 1)
            return !0;
        if (b = this.getTargetElementFromEventTarget(a.target), c = a.targetTouches[0], e) {
            if (d = window.getSelection(), d.rangeCount&&!d.isCollapsed)
                return !0;
            if (!f) {
                if (c.identifier === this.lastTouchIdentifier)
                    return a.preventDefault(), !1;
                this.lastTouchIdentifier = c.identifier, this.updateScrollParent(b)
            }
        }
        return this.trackingClick=!0, this.trackingClickStart = a.timeStamp, this.targetElement = b, this.touchStartX = c.pageX, this.touchStartY = c.pageY, a.timeStamp - this.lastClickTime < this.tapDelay && a.preventDefault(), !0
    }, c.prototype.touchHasMoved = function(a) {
        "use strict";
        var b = a.changedTouches[0], c = this.touchBoundary;
        return Math.abs(b.pageX - this.touchStartX) > c || Math.abs(b.pageY - this.touchStartY) > c?!0 : !1
    }, c.prototype.onTouchMove = function(a) {
        "use strict";
        return this.trackingClick ? ((this.targetElement !== this.getTargetElementFromEventTarget(a.target) || this.touchHasMoved(a)) && (this.trackingClick=!1, this.targetElement = null), !0) : !0
    }, c.prototype.findControl = function(a) {
        "use strict";
        return void 0 !== a.control ? a.control : a.htmlFor ? document.getElementById(a.htmlFor) : a.querySelector("button, input:not([type=hidden]), keygen, meter, output, progress, select, textarea")
    }, c.prototype.onTouchEnd = function(a) {
        "use strict";
        var b, c, h, i, j, k = this.targetElement;
        if (!this.trackingClick)
            return !0;
        if (a.timeStamp - this.lastClickTime < this.tapDelay)
            return this.cancelNextClick=!0, !0;
        if (this.cancelNextClick=!1, this.lastClickTime = a.timeStamp, c = this.trackingClickStart, this.trackingClick=!1, this.trackingClickStart = 0, g && (j = a.changedTouches[0], k = document.elementFromPoint(j.pageX - window.pageXOffset, j.pageY - window.pageYOffset) || k, k.fastClickScrollParent = this.targetElement.fastClickScrollParent), h = k.tagName.toLowerCase(), "label" === h) {
            if (b = this.findControl(k)) {
                if (this.focus(k), d)
                    return !1;
                k = b
            }
        } else if (this.needsFocus(k))
            return a.timeStamp - c > 100 || e && window.top !== window && "input" === h ? (this.targetElement = null, !1) : (this.focus(k), this.sendClick(k, a), e && "select" === h || (this.targetElement = null, a.preventDefault()), !1);
        return e&&!f && (i = k.fastClickScrollParent, i && i.fastClickLastScrollTop !== i.scrollTop)?!0 : (this.needsClick(k) || (a.preventDefault(), this.sendClick(k, a)), !1)
    }, c.prototype.onTouchCancel = function() {
        "use strict";
        this.trackingClick=!1, this.targetElement = null
    }, c.prototype.onMouse = function(a) {
        "use strict";
        return this.targetElement ? a.forwardedTouchEvent?!0 : a.cancelable && (!this.needsClick(this.targetElement) || this.cancelNextClick) ? (a.stopImmediatePropagation ? a.stopImmediatePropagation() : a.propagationStopped=!0, a.stopPropagation(), a.preventDefault(), !1) : !0 : !0
    }, c.prototype.onClick = function(a) {
        "use strict";
        var b;
        return this.trackingClick ? (this.targetElement = null, this.trackingClick=!1, !0) : "submit" === a.target.type && 0 === a.detail?!0 : (b = this.onMouse(a), b || (this.targetElement = null), b)
    }, c.prototype.destroy = function() {
        "use strict";
        var a = this.layer;
        d && (a.removeEventListener("mouseover", this.onMouse, !0), a.removeEventListener("mousedown", this.onMouse, !0), a.removeEventListener("mouseup", this.onMouse, !0)), a.removeEventListener("click", this.onClick, !0), a.removeEventListener("touchstart", this.onTouchStart, !1), a.removeEventListener("touchmove", this.onTouchMove, !1), a.removeEventListener("touchend", this.onTouchEnd, !1), a.removeEventListener("touchcancel", this.onTouchCancel, !1)
    }, c.notNeeded = function(a) {
        "use strict";
        var b, c;
        if ("undefined" == typeof window.ontouchstart)
            return !0;
        if (c =+ (/Chrome\/([0-9]+)/.exec(navigator.userAgent) || [, 0])[1]) {
            if (!d)
                return !0;
            if (b = document.querySelector("meta[name=viewport]")) {
                if ( - 1 !== b.content.indexOf("user-scalable=no"))
                    return !0;
                if (c > 31 && document.documentElement.scrollWidth <= window.outerWidth)
                    return !0
            }
        }
        return "none" === a.style.msTouchAction?!0 : !1
    }, c.attach = function(a, b) {
        "use strict";
        return new c(a, b)
    }, "function" == typeof define && "object" == typeof define.amd && define.amd ? define(function() {
        "use strict";
        return c
    }) : "undefined" != typeof module && module.exports ? (module.exports = c.attach, module.exports.FastClick = c) : window.FastClick = c, function(a, b, d, e) {
        "use strict";
        function f(a) {
            return ("string" == typeof a || a instanceof String) && (a = a.replace(/^['\\/"]+|(;\s?})+|['\\/"]+$/g, "")), a
        }
        var g = function(b) {
            for (var c = b.length, d = a("head"); c--;)
                0 === d.has("." + b[c]).length && d.append('<meta class="' + b[c] + '" />')
        };
        g(["foundation-mq-small", "foundation-mq-medium", "foundation-mq-large", "foundation-mq-xlarge", "foundation-mq-xxlarge", "foundation-data-attribute-namespace"]), a(function() {
            "undefined" != typeof c && "undefined" != typeof d.body && c.attach(d.body)
        });
        var h = function(b, c) {
            if ("string" == typeof b) {
                if (c) {
                    var e;
                    if (c.jquery) {
                        if (e = c[0], !e)
                            return c
                    } else 
                        e = c;
                    return a(e.querySelectorAll(b))
                }
                return a(d.querySelectorAll(b))
            }
            return a(b, c)
        }, i = function(a) {
            var b = [];
            return a || b.push("data"), this.namespace.length > 0 && b.push(this.namespace), b.push(this.name), b.join("-")
        }, j = function(a) {
            for (var b = a.split("-"), c = b.length, d = []; c--;)
                0 !== c ? d.push(b[c]) : this.namespace.length > 0 ? d.push(this.namespace, b[c]) : d.push(b[c]);
            return d.reverse().join("-")
        }, k = function(b, c) {
            var d = this, e=!h(this).data(this.attr_name(!0));
            return "string" == typeof b ? this[b].call(this, c) : void(h(this.scope).is("[" + this.attr_name() + "]") ? (h(this.scope).data(this.attr_name(!0) + "-init", a.extend({}, this.settings, c || b, this.data_options(h(this.scope)))), e && this.events(this.scope)) : h("[" + this.attr_name() + "]", this.scope).each(function() {
                var e=!h(this).data(d.attr_name(!0) + "-init");
                h(this).data(d.attr_name(!0) + "-init", a.extend({}, d.settings, c || b, d.data_options(h(this)))), e && d.events(this)
            }))
        }, l = function(a, b) {
            function c() {
                b(a[0])
            }
            function d() {
                if (this.one("load", c), /MSIE (\d+\.\d+);/.test(navigator.userAgent)) {
                    var a = this.attr("src"), b = a.match(/\?/) ? "&": "?";
                    b += "random=" + (new Date).getTime(), this.attr("src", a + b)
                }
            }
            return a.attr("src") ? void(a[0].complete || 4 === a[0].readyState ? c() : d.call(a)) : void c()
        };
        b.matchMedia = b.matchMedia || function(a) {
            var b, c = a.documentElement, d = c.firstElementChild || c.firstChild, e = a.createElement("body"), f = a.createElement("div");
            return f.id = "mq-test-1", f.style.cssText = "position:absolute;top:-100em", e.style.background = "none", e.appendChild(f), function(a) {
                return f.innerHTML = '&shy;<style media="' + a + '"> #mq-test-1 { width: 42px; }</style>', c.insertBefore(e, d), b = 42 === f.offsetWidth, c.removeChild(e), {
                    matches: b,
                    media: a
                }
            }
        }(d), function() {
            function a() {
                c && (f(a), h && jQuery.fx.tick())
            }
            for (var c, d = 0, e = ["webkit", "moz"], f = b.requestAnimationFrame, g = b.cancelAnimationFrame, h = "undefined" != typeof jQuery.fx; d < e.length&&!f; d++)
                f = b[e[d] + "RequestAnimationFrame"], g = g || b[e[d] + "CancelAnimationFrame"] || b[e[d] + "CancelRequestAnimationFrame"];
            f ? (b.requestAnimationFrame = f, b.cancelAnimationFrame = g, h && (jQuery.fx.timer = function(b) {
                b() && jQuery.timers.push(b)&&!c && (c=!0, a())
            }, jQuery.fx.stop = function() {
                c=!1
            })) : (b.requestAnimationFrame = function(a) {
                var c = (new Date).getTime(), e = Math.max(0, 16 - (c - d)), f = b.setTimeout(function() {
                    a(c + e)
                }, e);
                return d = c + e, f
            }, b.cancelAnimationFrame = function(a) {
                clearTimeout(a)
            })
        }(jQuery), b.Foundation = {
            name: "Foundation",
            version: "5.2.2",
            media_queries: {
                small: h(".foundation-mq-small").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
                medium: h(".foundation-mq-medium").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
                large: h(".foundation-mq-large").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
                xlarge: h(".foundation-mq-xlarge").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
                xxlarge: h(".foundation-mq-xxlarge").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, "")
            },
            stylesheet: a("<style></style>").appendTo("head")[0].sheet,
            global: {
                namespace: e
            },
            init: function(a, b, c, d, e) {
                var f = [a, c, d, e], g = [];
                if (this.rtl = /rtl/i.test(h("html").attr("dir")), this.scope = a || this.scope, this.set_namespace(), b && "string" == typeof b&&!/reflow/i.test(b))
                    this.libs.hasOwnProperty(b) && g.push(this.init_lib(b, f));
                else 
                    for (var i in this.libs)
                        g.push(this.init_lib(i, b));
                return a
            },
            init_lib: function(b, c) {
                return this.libs.hasOwnProperty(b) ? (this.patch(this.libs[b]), c && c.hasOwnProperty(b) ? ("undefined" != typeof this.libs[b].settings ? a.extend(!0, this.libs[b].settings, c[b]) : "undefined" != typeof this.libs[b].defaults && a.extend(!0, this.libs[b].defaults, c[b]), this.libs[b].init.apply(this.libs[b], [this.scope, c[b]])) : (c = c instanceof Array ? c : new Array(c), this.libs[b].init.apply(this.libs[b], c))) : function() {}
            },
            patch: function(a) {
                a.scope = this.scope, a.namespace = this.global.namespace, a.rtl = this.rtl, a.data_options = this.utils.data_options, a.attr_name = i, a.add_namespace = j, a.bindings = k, a.S = this.utils.S
            },
            inherit: function(a, b) {
                for (var c = b.split(" "), d = c.length; d--;)
                    this.utils.hasOwnProperty(c[d]) && (a[c[d]] = this.utils[c[d]])
            },
            set_namespace: function() {
                var b = this.global.namespace === e ? a(".foundation-data-attribute-namespace").css("font-family"): this.global.namespace;
                this.global.namespace = b === e || /false/i.test(b) ? "" : b
            },
            libs: {},
            utils: {
                S: h,
                throttle: function(a, b) {
                    var c = null;
                    return function() {
                        var d = this, e = arguments;
                        null == c && (c = setTimeout(function() {
                            a.apply(d, e), c = null
                        }, b))
                    }
                },
                debounce: function(a, b, c) {
                    var d, e;
                    return function() {
                        var f = this, g = arguments, h = function() {
                            d = null, c || (e = a.apply(f, g))
                        }, i = c&&!d;
                        return clearTimeout(d), d = setTimeout(h, b), i && (e = a.apply(f, g)), e
                    }
                },
                data_options: function(b) {
                    function c(a) {
                        return !isNaN(a - 0) && null !== a && "" !== a && a!==!1 && a!==!0
                    }
                    function d(b) {
                        return "string" == typeof b ? a.trim(b) : b
                    }
                    var e, f, g, h = {}, i = function(a) {
                        var b = Foundation.global.namespace;
                        return a.data(b.length > 0 ? b + "-options" : "options")
                    }, j = i(b);
                    if ("object" == typeof j)
                        return j;
                    for (g = (j || ":").split(";"), e = g.length; e--;)
                        f = g[e].split(":"), /true/i.test(f[1]) && (f[1]=!0), /false/i.test(f[1]) && (f[1]=!1), c(f[1]) && (f[1] =- 1 === f[1].indexOf(".") ? parseInt(f[1], 10) : parseFloat(f[1])), 2 === f.length && f[0].length > 0 && (h[d(f[0])] = d(f[1]));
                    return h
                },
                register_media: function(b, c) {
                    Foundation.media_queries[b] === e && (a("head").append('<meta class="' + c + '">'), Foundation.media_queries[b] = f(a("." + c).css("font-family")))
                },
                add_custom_rule: function(a, b) {
                    if (b === e)
                        Foundation.stylesheet.insertRule(a, Foundation.stylesheet.cssRules.length);
                    else {
                        var c = Foundation.media_queries[b];
                        c !== e && Foundation.stylesheet.insertRule("@media " + Foundation.media_queries[b] + "{ " + a + " }")
                    }
                },
                image_loaded: function(a, b) {
                    var c = this, d = a.length;
                    0 === d && b(a), a.each(function() {
                        l(c.S(this), function() {
                            d -= 1, 0 === d && b(a)
                        })
                    })
                },
                random_str: function() {
                    return this.fidx || (this.fidx = 0), this.prefix = this.prefix || [this.name || "F", ( + new Date).toString(36)].join("-"), this.prefix + (this.fidx++).toString(36)
                }
            }
        }, a.fn.foundation = function() {
            var a = Array.prototype.slice.call(arguments, 0);
            return this.each(function() {
                return Foundation.init.apply(Foundation, [this].concat(a)), this
            })
        }
    }(jQuery, this, this.document), function(a, b, c, d) {
        "use strict";
        function e(a) {
            var b = /fade/i.test(a), c = /pop/i.test(a);
            return {
                animate: b || c,
                pop: c,
                fade: b
            }
        }
        Foundation.libs.reveal = {
            name: "reveal",
            version: "5.2.2",
            locked: !1,
            settings: {
                animation: "fadeAndPop",
                animation_speed: 250,
                close_on_background_click: !0,
                close_on_esc: !0,
                dismiss_modal_class: "close-reveal-modal",
                bg_class: "reveal-modal-bg",
                open: function() {},
                opened: function() {},
                close: function() {},
                closed: function() {},
                bg: a(".reveal-modal-bg"),
                css: {
                    open: {
                        opacity: 0,
                        visibility: "visible",
                        display: "block"
                    },
                    close: {
                        opacity: 1,
                        visibility: "hidden",
                        display: "none"
                    }
                }
            },
            init: function(b, c, d) {
                a.extend(!0, this.settings, c, d), this.bindings(c, d)
            },
            events: function() {
                var a = this, b = a.S;
                return b(this.scope).off(".reveal").on("click.fndtn.reveal", "[" + this.add_namespace("data-reveal-id") + "]", function(c) {
                    if (c.preventDefault(), !a.locked) {
                        var d = b(this), e = d.data(a.data_attr("reveal-ajax"));
                        if (a.locked=!0, "undefined" == typeof e)
                            a.open.call(a, d);
                        else {
                            var f = e===!0 ? d.attr("href"): e;
                            a.open.call(a, d, {
                                url: f
                            })
                        }
                    }
                }), b(c).on("touchend.fndtn.reveal click.fndtn.reveal", this.close_targets(), function(c) {
                    if (c.preventDefault(), !a.locked) {
                        var d = b("[" + a.attr_name() + "].open").data(a.attr_name(!0) + "-init"), e = b(c.target)[0] === b("." + d.bg_class)[0];
                        if (e) {
                            if (!d.close_on_background_click)
                                return;
                            c.stopPropagation()
                        }
                        a.locked=!0, a.close.call(a, e ? b("[" + a.attr_name() + "].open") : b(this).closest("[" + a.attr_name() + "]"))
                    }
                }), b("[" + a.attr_name() + "]", this.scope).length > 0 ? b(this.scope).on("open.fndtn.reveal", this.settings.open).on("opened.fndtn.reveal", this.settings.opened).on("opened.fndtn.reveal", this.open_video).on("close.fndtn.reveal", this.settings.close).on("closed.fndtn.reveal", this.settings.closed).on("closed.fndtn.reveal", this.close_video) : b(this.scope).on("open.fndtn.reveal", "[" + a.attr_name() + "]", this.settings.open).on("opened.fndtn.reveal", "[" + a.attr_name() + "]", this.settings.opened).on("opened.fndtn.reveal", "[" + a.attr_name() + "]", this.open_video).on("close.fndtn.reveal", "[" + a.attr_name() + "]", this.settings.close).on("closed.fndtn.reveal", "[" + a.attr_name() + "]", this.settings.closed).on("closed.fndtn.reveal", "[" + a.attr_name() + "]", this.close_video), !0
            },
            key_up_on: function() {
                var a = this;
                return a.S("body").off("keyup.fndtn.reveal").on("keyup.fndtn.reveal", function(b) {
                    var c = a.S("[" + a.attr_name() + "].open"), d = c.data(a.attr_name(!0) + "-init");
                    d && 27 === b.which && d.close_on_esc&&!a.locked && a.close.call(a, c)
                }), !0
            },
            key_up_off: function() {
                return this.S("body").off("keyup.fndtn.reveal"), !0
            },
            open: function(b, c) {
                var d = this;
                if (b)
                    if ("undefined" != typeof b.selector)
                        var e = d.S("#" + b.data(d.data_attr("reveal-id")));
                    else {
                        var e = d.S(this.scope);
                        c = b
                    } else 
                        var e = d.S(this.scope);
                var f = e.data(d.attr_name(!0) + "-init");
                if (!e.hasClass("open")) {
                    var g = d.S("[" + d.attr_name() + "].open");
                    if ("undefined" == typeof e.data("css-top") && e.data("css-top", parseInt(e.css("top"), 10)).data("offset", this.cache_offset(e)), this.key_up_on(e), e.trigger("open"), g.length < 1 && this.toggle_bg(e), "string" == typeof c && (c = {
                        url: c
                    }), "undefined" != typeof c && c.url) {
                        var h = "undefined" != typeof c.success ? c.success: null;
                        a.extend(c, {
                            success: function(b, c, i) {
                                a.isFunction(h) && h(b, c, i), e.html(b), d.S(e).foundation("section", "reflow"), g.length > 0 && d.hide(g, f.css.close), d.show(e, f.css.open)
                            }
                        }), a.ajax(c)
                    } else 
                        g.length > 0 && this.hide(g, f.css.close), this.show(e, f.css.open)
                }
            },
            close: function(a) {
                var a = a && a.length ? a: this.S(this.scope), b = this.S("[" + this.attr_name() + "].open"), c = a.data(this.attr_name(!0) + "-init");
                b.length > 0 && (this.locked=!0, this.key_up_off(a), a.trigger("close"), this.toggle_bg(a), this.hide(b, c.css.close, c))
            },
            close_targets: function() {
                var a = "." + this.settings.dismiss_modal_class;
                return this.settings.close_on_background_click ? a + ", ." + this.settings.bg_class : a
            },
            toggle_bg: function(b) {
                b.data(this.attr_name(!0));
                0 === this.S("." + this.settings.bg_class).length && (this.settings.bg = a("<div />", {
                    "class": this.settings.bg_class
                }).appendTo("body").hide()), this.settings.bg.filter(":visible").length > 0 ? this.hide(this.settings.bg) : this.show(this.settings.bg)
            },
            show: function(c, d) {
                if (d) {
                    var f = c.data(this.attr_name(!0) + "-init");
                    if (0 === c.parent("body").length) {
                        var g = c.wrap('<div style="display: none;" />').parent(), h = this.settings.rootElement || "body";
                        c.on("closed.fndtn.reveal.wrapped", function() {
                            c.detach().appendTo(g), c.unwrap().unbind("closed.fndtn.reveal.wrapped")
                        }), c.detach().appendTo(h)
                    }
                    var i = e(f.animation);
                    if (i.animate || (this.locked=!1), i.pop) {
                        d.top = a(b).scrollTop() - c.data("offset") + "px";
                        var j = {
                            top: a(b).scrollTop() + c.data("css-top") + "px",
                            opacity: 1
                        };
                        return setTimeout(function() {
                            return c.css(d).animate(j, f.animation_speed, "linear", function() {
                                this.locked=!1, c.trigger("opened")
                            }.bind(this)).addClass("open")
                        }.bind(this), f.animation_speed / 2)
                    }
                    if (i.fade) {
                        d.top = a(b).scrollTop() + c.data("css-top") + "px";
                        var j = {
                            opacity: 1
                        };
                        return setTimeout(function() {
                            return c.css(d).animate(j, f.animation_speed, "linear", function() {
                                this.locked=!1, c.trigger("opened")
                            }.bind(this)).addClass("open")
                        }.bind(this), f.animation_speed / 2)
                    }
                    return c.css(d).show().css({
                        opacity: 1
                    }).addClass("open").trigger("opened")
                }
                var f = this.settings;
                return e(f.animation).fade ? c.fadeIn(f.animation_speed / 2) : (this.locked=!1, c.show())
            },
            hide: function(c, d) {
                if (d) {
                    var f = c.data(this.attr_name(!0) + "-init"), g = e(f.animation);
                    if (g.animate || (this.locked=!1), g.pop) {
                        var h = {
                            top: - a(b).scrollTop() - c.data("offset") + "px",
                            opacity: 0
                        };
                        return setTimeout(function() {
                            return c.animate(h, f.animation_speed, "linear", function() {
                                this.locked=!1, c.css(d).trigger("closed")
                            }.bind(this)).removeClass("open")
                        }.bind(this), f.animation_speed / 2)
                    }
                    if (g.fade) {
                        var h = {
                            opacity: 0
                        };
                        return setTimeout(function() {
                            return c.animate(h, f.animation_speed, "linear", function() {
                                this.locked=!1, c.css(d).trigger("closed")
                            }.bind(this)).removeClass("open")
                        }.bind(this), f.animation_speed / 2)
                    }
                    return c.hide().css(d).removeClass("open").trigger("closed")
                }
                var f = this.settings;
                return e(f.animation).fade ? c.fadeOut(f.animation_speed / 2) : c.hide()
            },
            close_video: function(b) {
                var c = a(".flex-video", b.target), d = a("iframe", c);
                d.length > 0 && (d.attr("data-src", d[0].src), d.attr("src", "about:blank"), c.hide())
            },
            open_video: function(b) {
                var c = a(".flex-video", b.target), e = c.find("iframe");
                if (e.length > 0) {
                    var f = e.attr("data-src");
                    if ("string" == typeof f)
                        e[0].src = e.attr("data-src");
                    else {
                        var g = e[0].src;
                        e[0].src = d, e[0].src = g
                    }
                    c.show()
                }
            },
            data_attr: function(a) {
                return this.namespace.length > 0 ? this.namespace + "-" + a : a
            },
            cache_offset: function(a) {
                var b = a.show().height() + parseInt(a.css("top"), 10);
                return a.hide(), b
            },
            off: function() {
                a(this.scope).off(".fndtn.reveal")
            },
            reflow: function() {}
        }
    }(jQuery, this, this.document), function(a, b, c) {
        "use strict";
        var d = function() {}, e = function(d, e) {
            if (d.hasClass(e.slides_container_class))
                return this;
            var h, i, j, k, l, m = this, n = d, o = 0, p = n.find("." + e.active_slide_class).length > 0;
            m.cache = {}, m.slides = function() {
                return n.children(e.slide_selector)
            }, p || m.slides().first().addClass(e.active_slide_class), m.update_slide_number = function(b) {
                e.slide_number && (i.find("span:first").text(parseInt(b) + 1), i.find("span:last").text(m.slides().length)), e.bullets && (j.children().removeClass(e.bullets_active_class), a(j.children().get(b)).addClass(e.bullets_active_class))
            }, m.update_active_link = function(b) {
                var c = a('[data-orbit-link="' + m.slides().eq(b).attr("data-orbit-slide") + '"]');
                c.siblings().removeClass(e.bullets_active_class), c.addClass(e.bullets_active_class)
            }, m.build_markup = function() {
                n.wrap('<div class="' + e.container_class + '"></div>'), h = n.parent(), n.addClass(e.slides_container_class), n.addClass(e.animation), e.stack_on_small && h.addClass(e.stack_on_small_class), e.navigation_arrows && (h.append(a('<a href="#"><span></span></a>').addClass(e.prev_class)), h.append(a('<a href="#"><span></span></a>').addClass(e.next_class))), e.timer && (k = a("<div>").addClass(e.timer_container_class), k.append("<span>"), e.timer_show_progress_bar && k.append(a("<div>").addClass(e.timer_progress_class)), k.addClass(e.timer_paused_class), h.append(k)), e.slide_number && (i = a("<div>").addClass(e.slide_number_class), i.append("<span></span> " + e.slide_number_text + " <span></span>"), h.append(i)), e.bullets && (j = a("<ol>").addClass(e.bullets_container_class), h.append(j), j.wrap('<div class="orbit-bullets-container"></div>'), m.slides().each(function(b) {
                    var c = a("<li>").attr("data-orbit-slide", b);
                    j.append(c)
                }))
            }, m._prepare_direction = function(b) {
                var c = "next";
                o >= b && (c = "prev"), "slide" === e.animation && setTimeout(function() {
                    n.removeClass("swipe-prev swipe-next"), "next" === c ? n.addClass("swipe-next") : "prev" === c && n.addClass("swipe-prev")
                }, 0);
                var d = m.slides();
                if (b >= d.length) {
                    if (!e.circular)
                        return !1;
                    b = 0
                } else if (0 > b) {
                    if (!e.circular)
                        return !1;
                    b = d.length - 1
                }
                var f = a(d.get(o)), g = a(d.get(b));
                return [c, f, g, b]
            }, m._goto = function(a, b) {
                if (null === a)
                    return !1;
                if (m.cache.animating)
                    return !1;
                if (a === o)
                    return !1;
                "object" == typeof m.cache.timer && m.cache.timer.restart();
                var c = m.slides();
                m.cache.animating=!0;
                var d = m._prepare_direction(a), f = d[0], g = d[1], h = d[2], a = d[3];
                if (d===!1)
                    return !1;
                n.trigger("before-slide-change.fndtn.orbit"), e.before_slide_change(), o = a, g.css("transitionDuration", e.animation_speed + "ms"), h.css("transitionDuration", e.animation_speed + "ms");
                var i = function() {
                    var d = function() {
                        b===!0 && m.cache.timer.restart(), m.update_slide_number(o), h.addClass(e.active_slide_class), m.update_active_link(a), n.trigger("after-slide-change.fndtn.orbit", [{
                            slide_number: o,
                            total_slides: c.length
                        }
                        ]), e.after_slide_change(o, c.length), setTimeout(function() {
                            m.cache.animating=!1
                        }, 100)
                    };
                    n.height() != h.height() && e.variable_height ? n.animate({
                        height: h.height()
                    }, 250, "linear", d) : d()
                };
                if (1 === c.length)
                    return i(), !1;
                var j = function() {
                    "next" === f && l.next(g, h, i), "prev" === f && l.prev(g, h, i)
                };
                h.height() > n.height() && e.variable_height ? n.animate({
                    height: h.height()
                }, 250, "linear", j) : j()
            }, m.next = function(a) {
                a.stopImmediatePropagation(), a.preventDefault(), m._prepare_direction(o + 1), setTimeout(function() {
                    m._goto(o + 1)
                }, 100)
            }, m.prev = function(a) {
                a.stopImmediatePropagation(), a.preventDefault(), m._prepare_direction(o - 1), setTimeout(function() {
                    m._goto(o - 1)
                }, 100)
            }, m.link_custom = function(b) {
                b.preventDefault();
                var c = a(this).attr("data-orbit-link");
                if ("string" == typeof c && "" != (c = a.trim(c))) {
                    var d = h.find("[data-orbit-slide=" + c + "]");
                    - 1 != d.index() && setTimeout(function() {
                        m._goto(d.index())
                    }, 100)
                }
            }, m.link_bullet = function() {
                var b = a(this).attr("data-orbit-slide");
                if ("string" == typeof b && "" != (b = a.trim(b)))
                    if (isNaN(parseInt(b))) {
                        var c = h.find("[data-orbit-slide=" + b + "]");
                        - 1 != c.index() && setTimeout(function() {
                            m._goto(c.index() + 1)
                        }, 100)
                    } else 
                        setTimeout(function() {
                            m._goto(parseInt(b))
                        }, 100)
            }, m.timer_callback = function() {
                m._goto(o + 1, !0)
            }, m.compute_dimensions = function() {
                var b = a(m.slides().get(o)), c = b.height();
                e.variable_height || m.slides().each(function() {
                    a(this).height() > c && (c = a(this).height())
                }), n.height(c)
            }, m.create_timer = function() {
                var a = new f(h.find("." + e.timer_container_class), e, m.timer_callback);
                return a
            }, m.stop_timer = function() {
                "object" == typeof m.cache.timer && m.cache.timer.stop()
            }, m.toggle_timer = function() {
                var a = h.find("." + e.timer_container_class);
                a.hasClass(e.timer_paused_class) ? ("undefined" == typeof m.cache.timer && (m.cache.timer = m.create_timer()), m.cache.timer.start()) : "object" == typeof m.cache.timer && m.cache.timer.stop()
            }, m.init = function() {
                if (m.build_markup(), e.timer && (m.cache.timer = m.create_timer(), Foundation.utils.image_loaded(this.slides().children("img"), m.cache.timer.start)), l = new g(e, n), p) {
                    var d = n.find("." + e.active_slide_class), f = e.animation_speed;
                    e.animation_speed = 1, d.removeClass("active"), m._goto(d.index()), e.animation_speed = f
                }
                h.on("click", "." + e.next_class, m.next), h.on("click", "." + e.prev_class, m.prev), e.next_on_click && h.on("click", "[data-orbit-slide]", m.link_bullet), h.on("click", m.toggle_timer), e.swipe && n.on("touchstart.fndtn.orbit", function(a) {
                    m.cache.animating || (a.touches || (a = a.originalEvent), a.preventDefault(), a.stopPropagation(), m.cache.start_page_x = a.touches[0].pageX, m.cache.start_page_y = a.touches[0].pageY, m.cache.start_time = (new Date).getTime(), m.cache.delta_x = 0, m.cache.is_scrolling = null, m.cache.direction = null, m.stop_timer())
                }).on("touchmove.fndtn.orbit", function(a) {
                    Math.abs(m.cache.delta_x) > 5 && (a.preventDefault(), a.stopPropagation()), m.cache.animating || requestAnimationFrame(function() {
                        if (a.touches || (a = a.originalEvent), !(a.touches.length > 1 || a.scale && 1 !== a.scale || (m.cache.delta_x = a.touches[0].pageX - m.cache.start_page_x, null === m.cache.is_scrolling && (m.cache.is_scrolling=!!(m.cache.is_scrolling || Math.abs(m.cache.delta_x) < Math.abs(a.touches[0].pageY - m.cache.start_page_y))), m.cache.is_scrolling))) {
                            var b = m.cache.delta_x < 0 ? o + 1: o - 1;
                            if (m.cache.direction !== b) {
                                var c = m._prepare_direction(b);
                                m.cache.direction = b, m.cache.dir = c[0], m.cache.current = c[1], m.cache.next = c[2]
                            }
                            if ("slide" === e.animation) {
                                var d, f;
                                d = m.cache.delta_x / h.width() * 100, f = d >= 0?-(100 - d) : 100 + d, m.cache.current.css("transform", "translate3d(" + d + "%,0,0)"), m.cache.next.css("transform", "translate3d(" + f + "%,0,0)")
                            }
                        }
                    })
                }).on("touchend.fndtn.orbit", function(a) {
                    m.cache.animating || (a.preventDefault(), a.stopPropagation(), setTimeout(function() {
                        m._goto(m.cache.direction)
                    }, 50))
                }), h.on("mouseenter.fndtn.orbit", function() {
                    e.timer && e.pause_on_hover && m.stop_timer()
                }).on("mouseleave.fndtn.orbit", function() {
                    e.timer && e.resume_on_mouseout && m.cache.timer.start()
                }), a(c).on("click", "[data-orbit-link]", m.link_custom), a(b).on("load resize", m.compute_dimensions);
                var i = this.slides().find("img");
                Foundation.utils.image_loaded(i, m.compute_dimensions), Foundation.utils.image_loaded(i, function() {
                    h.prev("." + e.preloader_class).css("display", "none"), m.update_slide_number(o), m.update_active_link(o), n.trigger("ready.fndtn.orbit")
                })
            }, m.init()
        }, f = function(a, b, c) {
            var d, e, f = this, g = b.timer_speed, h = a.find("." + b.timer_progress_class), i = h && "none" != h.css("display"), j =- 1;
            this.update_progress = function(a) {
                var b = h.clone();
                b.attr("style", ""), b.css("width", a + "%"), h.replaceWith(b), h = b
            }, this.restart = function() {
                clearTimeout(e), a.addClass(b.timer_paused_class), j =- 1, i && f.update_progress(0), f.start()
            }, this.start = function() {
                return a.hasClass(b.timer_paused_class) ? (j =- 1 === j ? g : j, a.removeClass(b.timer_paused_class), i && (d = (new Date).getTime(), h.animate({
                    width: "100%"
                }, j, "linear")), e = setTimeout(function() {
                    f.restart(), c()
                }, j), void a.trigger("timer-started.fndtn.orbit")) : !0
            }, this.stop = function() {
                if (a.hasClass(b.timer_paused_class))
                    return !0;
                if (clearTimeout(e), a.addClass(b.timer_paused_class), i) {
                    var c = (new Date).getTime();
                    j -= c - d;
                    var h = 100 - j / g * 100;
                    f.update_progress(h)
                }
                a.trigger("timer-stopped.fndtn.orbit")
            }
        }, g = function(a, b) {
            var c = "webkitTransitionEnd otransitionend oTransitionEnd msTransitionEnd transitionend";
            this.next = function(d, e, f) {
                Modernizr.csstransitions ? e.on(c, function() {
                    e.unbind(c), d.removeClass("active animate-out"), e.removeClass("animate-in"), b.children().css({
                        transform: "",
                        "-ms-transform": "",
                        "-webkit-transition-duration": "",
                        "-moz-transition-duration": "",
                        "-o-transition-duration": "",
                        "transition-duration": ""
                    }), f()
                }) : setTimeout(function() {
                    d.removeClass("active animate-out"), e.removeClass("animate-in"), b.children().css({
                        transform: "",
                        "-ms-transform": "",
                        "-webkit-transition-duration": "",
                        "-moz-transition-duration": "",
                        "-o-transition-duration": "",
                        "transition-duration": ""
                    }), f()
                }, a.animation_speed), b.children().css({
                    transform: "",
                    "-ms-transform": "",
                    "-webkit-transition-duration": "",
                    "-moz-transition-duration": "",
                    "-o-transition-duration": "",
                    "transition-duration": ""
                }), d.addClass("animate-out"), e.addClass("animate-in")
            }, this.prev = function(d, e, f) {
                Modernizr.csstransitions ? e.on(c, function() {
                    e.unbind(c), d.removeClass("active animate-out"), e.removeClass("animate-in"), b.children().css({
                        transform: "",
                        "-ms-transform": "",
                        "-webkit-transition-duration": "",
                        "-moz-transition-duration": "",
                        "-o-transition-duration": "",
                        "transition-duration": ""
                    }), f()
                }) : setTimeout(function() {
                    d.removeClass("active animate-out"), e.removeClass("animate-in"), b.children().css({
                        transform: "",
                        "-ms-transform": "",
                        "-webkit-transition-duration": "",
                        "-moz-transition-duration": "",
                        "-o-transition-duration": "",
                        "transition-duration": ""
                    }), f()
                }, a.animation_speed), b.children().css({
                    transform: "",
                    "-ms-transform": "",
                    "-webkit-transition-duration": "",
                    "-moz-transition-duration": "",
                    "-o-transition-duration": "",
                    "transition-duration": ""
                }), d.addClass("animate-out"), e.addClass("animate-in")
            }
        };
        Foundation.libs = Foundation.libs || {}, Foundation.libs.orbit = {
            name: "orbit",
            version: "5.2.2",
            settings: {
                animation: "slide",
                timer_speed: 1e4,
                pause_on_hover: !0,
                resume_on_mouseout: !1,
                next_on_click: !0,
                animation_speed: 500,
                stack_on_small: !1,
                navigation_arrows: !0,
                slide_number: !0,
                slide_number_text: "of",
                container_class: "orbit-container",
                stack_on_small_class: "orbit-stack-on-small",
                next_class: "orbit-next",
                prev_class: "orbit-prev",
                timer_container_class: "orbit-timer",
                timer_paused_class: "paused",
                timer_progress_class: "orbit-progress",
                timer_show_progress_bar: !0,
                slides_container_class: "orbit-slides-container",
                preloader_class: "preloader",
                slide_selector: "*",
                bullets_container_class: "orbit-bullets",
                bullets_active_class: "active",
                slide_number_class: "orbit-slide-number",
                caption_class: "orbit-caption",
                active_slide_class: "active",
                orbit_transition_class: "orbit-transitioning",
                bullets: !0,
                circular: !0,
                timer: !0,
                variable_height: !1,
                swipe: !0,
                before_slide_change: d,
                after_slide_change: d
            },
            init: function(a, b, c) {
                this.bindings(b, c)
            },
            events: function(a) {
                var b = new e(this.S(a), this.S(a).data("orbit-init"));
                this.S(a).data(self.name + "-instance", b)
            },
            reflow: function() {
                var a = this;
                if (a.S(a.scope).is("[data-orbit]")) {
                    var b = a.S(a.scope), c = b.data(a.name + "-instance");
                    c.compute_dimensions()
                } else 
                    a.S("[data-orbit]", a.scope).each(function(b, c) {
                        var d = a.S(c), e = (a.data_options(d), d.data(a.name + "-instance"));
                        e.compute_dimensions()
                    })
            }
        }
    }(jQuery, this, this.document), function(a) {
        a.fn.flowtype = function(b) {
            var c = a.extend({
                maximum: 9999,
                minimum: 1,
                maxFont: 9999,
                minFont: 1,
                fontRatio: 35
            }, b), d = function(b) {
                var d = a(b), e = d.width(), f = e > c.maximum ? c.maximum: e < c.minimum ? c.minimum: e, g = f / c.fontRatio, h = g > c.maxFont ? c.maxFont: g < c.minFont ? c.minFont: g;
                d.css("font-size", h + "px")
            };
            return this.each(function() {
                var b = this;
                a(window).resize(function() {
                    d(b)
                }), d(this)
            })
        }
    }(jQuery), function(a, b) {
        "use strict";
        var c = function(a, b, c) {
            var d;
            return function() {
                function e() {
                    c || a.apply(f, g), d = null
                }
                var f = this, g = arguments;
                d ? clearTimeout(d) : c && a.apply(f, g), d = setTimeout(e, b || 100)
            }
        };
        jQuery.fn[b] = function(a) {
            return a ? this.bind("resize", c(a)) : this.trigger(b)
        }
    }(jQuery, "smartresize"), function(a) {
        "use strict";
        function b() {
            this.reset()
        }
        function c() {
            a(".balance-text").balanceText()
        }
        var d = document.documentElement.style, e = d.textWrap || d.WebkitTextWrap || d.MozTextWrap || d.MsTextWrap || d.OTextWrap;
        b.prototype.reset = function() {
            this.index = 0, this.width = 0
        };
        var f = function(a) {
            return Boolean(a.match(/^\s$/))
        }, g = function(b) {
            b.find('br[data-owner="balance-text"]').replaceWith(" ");
            var c = b.find('span[data-owner="balance-text"]');
            if (c.length > 0) {
                var d = "";
                c.each(function() {
                    d += a(this).text(), a(this).remove()
                }), b.html(d)
            }
        }, h = function(a) {
            return d = a.get(0).currentStyle || window.getComputedStyle(a.get(0), null), "justify" === d.textAlign
        }, i = function(b, c, d) {
            c = a.trim(c);
            var e = c.split(" ").length;
            if (c += " ", 2 > e)
                return c;
            var f = a("<span></span>").html(c);
            b.append(f);
            var g = f.width();
            f.remove();
            var h = Math.floor((d - g) / (e - 1));
            return f.css("word-spacing", h + "px").attr("data-owner", "balance-text"), a("<div></div>").append(f).html()
        }, j = function(a, b) {
            return 0 === b || b === a.length || f(a.charAt(b - 1))&&!f(a.charAt(b))
        }, k = function(a, b, c, d, e, f, g) {
            for (var h; ;) {
                for (; !j(b, f);)
                    f += e;
                if (a.text(b.substr(0, f)), h = a.width(), 0 > e ? d >= h || 0 >= h || 0 === f : h >= d || h >= c || f === b.length)
                    break;
                f += e
            }
            g.index = f, g.width = h
        };
        a.fn.balanceText = function() {
            return e ? this : this.each(function() {
                var c = a(this), d = 5e3;
                g(c);
                var e = "";
                c.attr("style") && c.attr("style").indexOf("line-height") >= 0 && (e = c.css("line-height")), c.css("line-height", "normal");
                var f = c.width(), j = c.height(), l = c.css("white-space"), m = c.css("float"), n = c.css("display"), o = c.css("position");
                c.css({
                    "white-space": "nowrap",
                    "float": "none",
                    display: "inline",
                    position: "static"
                });
                var p = c.width(), q = c.height(), r = "pre-wrap" === l ? 0: q / 4;
                if (f > 0 && p > f && d > p) {
                    for (var s = c.text(), t = "", u = "", v = h(c), w = Math.round(j / q), x = w; x > 1;) {
                        var y = Math.round((p + r) / x - r), z = Math.round((s.length + 1) / x) - 1, A = new b;
                        k(c, s, f, y, - 1, z, A);
                        var B = new b;
                        z = A.index, k(c, s, f, y, 1, z, B), A.reset(), z = B.index, k(c, s, f, y, - 1, z, A);
                        var C;
                        C = 0 === A.index ? B.index : f < B.width || A.index === B.index ? A.index : Math.abs(y - A.width) < Math.abs(B.width - y) ? A.index : B.index, u = s.substr(0, C), v ? t += i(c, u, f) : (t += u.replace(/\s+$/, ""), t += '<br data-owner="balance-text" />'), s = s.substr(C), x--, c.text(s), p = c.width()
                    }
                    c.html(v ? t + i(c, s, f) : t + s)
                }
                c.css({
                    position: o,
                    display: n,
                    "float": m,
                    "white-space": l,
                    "line-height": e
                })
            })
        }, a(window).ready(c), a(window).smartresize(c)
    }(jQuery), function(a) {
        "use strict";
        function b() {
            function a(a) {
                for (var c, d = window.document.querySelectorAll(e.overlays), f = 0; c = d[f]; f++)
                    c.style.opacity = a ? 1 : 0;
                a ? window.removeEventListener("keydown", b) : window.addEventListener("keydown", b)
            }
            function b(a) {
                27 === a.keyCode && f.parentNode.querySelector(e.media).pause()
            }
            function d(b) {
                document.webkitIsFullScreen && b.webkitExitFullscreen && b.webkitExitFullscreen(), f.checked=!1, a(!0)
            }
            var e = c, f = window.document.querySelector(e.initiate);
            f && f.addEventListener("change", function(b) {
                var c = b.target.parentNode, d = c.querySelector(e.media);
                d && (this.checked ? (d.msZoom=!0, d.play(), a(!1)) : (d.pause(), a(!0)))
            });
            var g = window.document.querySelector(e.media);
            g && (g.addEventListener("pause", d), g.addEventListener("ended", d))
        }
        var c = {
            initiate: "input[name=hero-video--visible]",
            media: ".hero video",
            overlays: "header"
        };
        a.hero = b
    }(window), function(a) {
        "use strict";
        var b = $("body").data("dom-page");
        if ("lobby" === b) {
            var c = 2e3, d = $(".dom-lobby-content"), e = function(a) {
                return d.hasClass("state--" + a)
            }, f = function(a) {
                d.removeClass("state--loading"), d.removeClass("state--dashboard"), d.removeClass("state--lobby"), d.addClass("state--" + a)
            }, g = setTimeout(function() {
                e("dashboard") || f("lobby")
            }, c), h = $.map($(".lobby-panels [data-dom-lobby-view]"), function(a) {
                return $(a).data("dom-lobby-view")
            }), i = h[0], j = function(b, c) {
                - 1 !== h.indexOf(b) && (d.find(".lobby-panels li").removeClass("active"), d.find('.lobby-panels li[data-dom-lobby-view="' + b + '"]').addClass("active"), d.find(".tab-bar li").removeClass("active"), d.find('.tab-bar li[data-dom-lobby-view="' + b + '"]').addClass("active"), c && (a.location.hash = b))
            };
            d.on("click", ".tab-bar li", function() {
                var a = $(this).data("dom-lobby-view");
                j(a, !0)
            });
            var k = function() {
                var b = a.location.hash && a.location.hash.substr(1) || i;
                - 1 !== h.indexOf(b) && j(b, !1)
            };
            $(a).on("hashchange", k), k();
            var l = a.adobe.account, m = l.info, n = l.Entitlement, o = [n.CC_FREE, n.CC_PAID, n.CC_PAID_NO_SERVICES];
            m.progress(function(b) {
                switch (b) {
                case l.LoadingProgress.ENTITLEMENTS_LOADED:
                    if (m.status === l.Status.SIGNED_IN) {
                        if ( - 1 === o.indexOf(m.entitlement))
                            return void f("lobby");
                        var c = $(".dom-lobby-profile-sign-out");
                        a.adobeGlobalNav.setSections(["sitemap", "search", "logo"]), $(".dom-lobby-profile-first-name").text(m.profile.first_name), $(".dom-lobby-profile-last-name").text(m.profile.last_name), f("dashboard"), c.click(function() {
                            a.adobeIMS.signOut()
                        })
                    }
                    break;
                case l.LoadingProgress.AVATAR_LOADED:
                    $(".dom-lobby-profile-avatar").attr("src", m.avatar)
                }
            }), m.done(function() {
                m.status === l.Status.SIGNED_OUT && f("lobby"), clearTimeout(g)
            }), m.fail(function(a) {})
        }
    }(window, document), function(a) {
        "use strict";
        function b(b) {
            var c = b ? b.target: a, e = $(c).scrollTop();
            e > 200 && 0 == d ? ($(".header-container").addClass("slideInDown header-container--fixed"), $(".product-header-title").addClass("product-header-title--innav"), $(".product-title-text").addClass("product-title-text--innav"), $(".product-title-logo").addClass("product-title-logo--innav"), d=!0) : 200 >= e && d && ($(".header-container").removeClass("slideInDown header-container--fixed"), $(".product-header-title").removeClass("product-header-title--innav"), $(".product-title-text").removeClass("product-title-text--innav"), $(".product-title-logo").removeClass("product-title-logo--innav"), d=!1)
        }
        var c = $("body").data("dom-page");
        if ("creativecloud" === c) {
            var d=!1;
            $(a).scroll(b), b()
        }
    }(window, document), function(a, b, c) {
        function d(b) {
            var d = b ? b.target: a, e = c(d).scrollTop();
            e > 200 && 0 == f ? (c(".header-container").addClass("slideInDown header-container--fixed"), c(".product-header-title").addClass("product-header-title--innav"), c(".product-title-text").addClass("product-title-text--innav"), c(".product-title-logo").addClass("product-title-logo--innav"), c(".innav-item > .dom-link--caret").addClass("innav-link--right"), c(".innav-item:last-child").css({
                display: "inline-block"
            }), f=!0) : 200 >= e && f && (c(".header-container").removeClass("slideInDown header-container--fixed"), c(".product-header-title").removeClass("product-header-title--innav"), c(".product-title-text").removeClass("product-title-text--innav"), c(".product-title-logo").removeClass("product-title-logo--innav"), c(".innav-item > .dom-link--caret").removeClass("innav-link--right"), c(".innav-item:last-child").css({
                display: "none"
            }), f=!1)
        }
        var e = c("body").data("dom-page");
        if (e.indexOf("products-")>-1 || e.indexOf("cc-video-segment")>-1) {
            var f=!1;
            c(a).scroll(d), d(), hero(), c(function(b) {
                b(a).on("beforeunload", function() {
                    b("[name=tab--visible]").blur()
                })
            }), function() {
                for (var d, e = a.location.search.substring(1).split("&"), f = 0; f < e.length; f++) {
                    var g = e[f];
                    g.match(/feature/) && (d = g.split("=")[1])
                }
                if (d) {
                    var h = c("#tabbed-feature-" + d);
                    if (h.prop("checked", !0), h.length) {
                        var i = c(".tabbed-features").offset().top;
                        i += 20, c(b).ready(function() {
                            c("body, html").animate({
                                scrollTop: i
                            }, 1e3)
                        })
                    }
                }
            }()
        }
    }(window, document, jQuery), function() {
        "use strict";
        $("#jottouch-referral-link").on("click", function(a) {
            a.preventDefault(), $("#buy-jottouch").submit()
        })
    }(window, document), function(a, b) {
        "use strict";
        function c(a) {
            if (a.preventDefault(), a.stopPropagation(), i !== a.currentTarget.id) {
                i = a.currentTarget.id;
                var b = f(i), c = g(i);
                h(c), $(b).fadeIn(900), n.fadeIn(1e3)
            }
        }
        function d(a) {
            a.preventDefault(), a.stopPropagation(), e()
        }
        function e() {
            i = null, m.hide(), n.hide(), k.attr("start", 1), l.removeClass("cc-faq-question-item--hidden")
        }
        function f(a) {
            return "#" + a.replace(q, o)
        }
        function g(a) {
            return "#" + a.replace(q, p)
        }
        function h(a) {
            l.each(function(b, c) {
                var d = "#" + c.id;
                d !== a ? $(d).addClass("cc-faq-question-item--hidden") : k.attr("start", b + 1)
            })
        }
        var i, j = $("#cc-segment-faq"), k = $(".cc-faq-questions"), l = $(".cc-faq-question-item"), m = $(".cc-faq-answer-item"), n = $(".cc-faq-answers-back-link"), o = "cc-faq-answer-", p = "cc-faq-question-", q = "cc-faq-question-link-";
        $("[id^=" + q + "]").on("click", c), $(".cc-faq-answers-back-link").on("click", d), $(b).on("closed", "[data-reveal]", function(a) {
            j && a.target === j[0] && e()
        })
    }(window, document), function(a, b) {
        "use strict";
        var c;
        $("*[data-video-url]").click(function(a) {
            var b = $(a.currentTarget).data("video-url");
            c = $("#video-modal"), c.foundation("reveal", "open", {
                bg_class: "reveal-modal-bg"
            }), c.click(function(a) {
                ("video-modal" === a.target.id || "video-close-button" === a.target.id) && c.foundation("reveal", "close")
            }), $("#video-container").html('<iframe id="video-iframe" class="dom-cc-video-modal-iframe" title="Adobe Video Publishing Cloud Player" src="' + b + '" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen scrolling="no"></iframe>')
        }), $(b).on("open", "[data-reveal]", function(a) {
            c && a.target === c[0] && $("body").css("overflow", "hidden")
        }), $(b).on("close", "[data-reveal]", function(a) {
            c && a.target === c[0] && ($("#video-container").html(), $("body").css("overflow", "auto"), c = null)
        })
    }(window, document), $.getURLParameter = function(a) {
        return decodeURIComponent((location.search.match(RegExp("[?&]" + a + "=([^&]*)")) || [, ""])[1])
    };
    var h = window.adobe || (window.adobe = {});
    "undefined" == typeof h.fn && (h.fn = {}), h.fn = {
        checkInternational: function() {
            if ($("#regions-visible").prop("checked", ""), $.cookie("eu-banner-shown") ? $(".dom-content--eu-banner").hide() : $(".dom-content--eu-banner-button").click(function() {
                $.cookie("eu-banner-shown", "true", {
                    domain: "adobe.com",
                    expires: 365,
                    path: "/"
                })
            }), $.cookie("international") || $.cookie("georouting_presented")) {
                var a = $(".dom-footer--region "), b = $(".dom-footer--changeregion");
                $.cookie("international") ? (a.show(), b.hide()) : (a.hide(), b.show())
            } else 
                h.fn.initGeorouting();
            this.observeRegionPanel()
        },
        observeRegionPanel: function() {
            $(".dom-footer--region-panel dd a").click(function(a) {
                a.preventDefault();
                var b = $(this).attr("href").replace(/\//g, "");
                h.fn.changeRegionFooter(b)
            }), $("#region-selector").click(function(a) {
                a.preventDefault();
                var b = $("#regions-visible").is(":checked") ? "": "checked";
                $("#regions-visible").prop("checked", b)
            })
        },
        changeRegionFooter: function(a) {
            S(a).startsWith("be_") ? ($.cookie("international", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }), $.cookie("storeregion", "be", {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            })) : S(a).startsWith("ca") ? ($.cookie("international", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }), $.cookie("storeregion", "ca", {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            })) : S(a).startsWith("eeur") ? ($.cookie("international", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }), $.cookie("storeregion", "eu", {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            })) : S(a).startsWith("hk_") ? ($.cookie("international", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }), $.cookie("storeregion", "cn", {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            })) : S(a).startsWith("lu_") ? ($.cookie("international", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }), $.cookie("storeregion", "lu", {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            })) : S(a).startsWith("uk") ? ($.cookie("international", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }), $.cookie("storeregion", "gb", {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            })) : ($.cookie("international", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }), $.cookie("storeregion", a, {
                domain: "adobe.com",
                expires: 365,
                path: "/"
            }));
            var b = window.location.pathname.replace(/\/special\/dom\/tests\/homepage\/v10\/pages/, "") + window.location.search;
            S(b).startsWith("/content/dotcom/") ? b = b.replace("/content/dotcom/", "/") : S(b).startsWith("/content/help/") && (b = b.replace("/content/help/", "/")), S(b).startsWith("/en/") && (b = b.replace("/en/", "/"));
            var c = ["africa", "ap", "at", "au", "be_en", "be_fr", "be_nl", "bg", "br", "ca", "ca_fr", "ch_de", "ch_fr", "ch_it", "cn", "cz", "de", "dk", "eeurope", "ee", "en", "en-US", "es", "fi", "fr", "hk_en", "hk_zh", "hr", "hu", "ie", "il_en", "il_he", "in", "it", "jp", "kr", "la", "lt", "lu_de", "lu_en", "lu_fr", "lv", "mena_ar", "mena_en", "mena_fr", "mx", "nl", "no", "nz", "pl", "pt", "ro", "rs", "ru", "si", "se", "sea", "sk", "tr", "tw", "ua", "uk"];
            $.each(c, function() {
                return S(b).startsWith("/" + this + "/") ? (b = b.replace("/" + this + "/", "/"), !1) : void 0
            }), newURL = "us" != a ? "/" + a + b : b, $(".dom-footer--region-panel-container").is(":visible") && $(".dom-footer--region-panel-container").toggle(), window.location.replace(newURL)
        },
        initGeorouting: function() {
            $("#Georouting").length&&!$.cookie("georouting_presented") && $(window).width() > 750 && this.getGeoLocation()
        },
        getGeoLocation: function() {
            var a = "http:" === window.location.protocol ? "http://geo.adobe.com/": "https://geo2.adobe.com/", b = a + "json/?callback=adobe.fn.setGeoLocation", c = "&";
            $.getURLParameter("testCountry") && (c += "country=" + $.getURLParameter("testCountry")), $.ajax({
                url: b + c,
                dataType: "jsonp",
                crossDomain: !0
            })
        },
        setGeoLocation: function(a) {
            var b = $("footer").attr("data-language"), c = a.country;
            if ("en" === b && "US" !== c) {
                var d = "/etc/pagetables/georouting_ip/us_" + h.fn.geoMapTable[c] + ".modal.html";
                $.ajax({
                    url: d,
                    success: function(a) {
                        var b = $(a), b = b.splice(0, 4);
                        h.fn.revealGeoLocationModal(b)
                    }
                })
            }
        },
        revealGeoLocationModal: function(a) {
            var b = $("#dom-georouting-modal"), c = $("#dom-georouting-modal-content");
            c.append(a), c.find("script").remove(), changeRegion = h.fn.changeRegionFooter, changeGeoRoutingPanel = function() {
                var a = $(".dom-footer--region-panel").clone(), b = $("#dom-georouting-modal"), c = $("#dom-georouting-modal-content");
                c.html(a), b.addClass("dom-georouting-modal--regionpanel"), h.fn.observeRegionPanel()
            }, b.foundation("reveal", {
                animation: "fadeAndPop",
                animation_speed: 250,
                close_on_background_click: !0,
                close_on_esc: !0,
                dismiss_modal_class: "close-reveal-modal",
                bg_class: "reveal-modal-bg",
                bg: $(".reveal-modal-bg"),
                css: {
                    open: {
                        opacity: 0,
                        visibility: "visible",
                        display: "block"
                    },
                    close: {
                        opacity: 1,
                        visibility: "hidden",
                        display: "none"
                    }
                }
            }), b.foundation("reveal", "open"), $.cookie("georouting_presented", "true", {
                domain: "adobe.com",
                expires: 30,
                path: "/"
            })
        },
        closeGeoLocationModal: function() {
            var a = $("#dom-georouting-modal");
            a.foundation("reveal", "close")
        }
    }, h.fn.checkInternational(), h.fn.geoMapTable = {
        AF: "afghanistan",
        AX: "alandislands",
        AL: "albania",
        DZ: "algeria",
        AS: "americansamoa",
        AD: "andorra",
        AO: "angola",
        AI: "anguilla",
        AQ: "antarctica",
        AG: "antiguaandbarbuda",
        AR: "argentina",
        AM: "armenia",
        AW: "aruba",
        AU: "australia",
        AT: "austria",
        AZ: "azerbaijan",
        BS: "bahamas",
        BH: "bahrain",
        BD: "bangladesh",
        BB: "barbados",
        BY: "belarus",
        BE: "belgium",
        BZ: "belize",
        BJ: "benin",
        BM: "bermuda",
        BT: "bhutan",
        BO: "bolivia",
        BA: "bosniaandherzegowina",
        BW: "botswana",
        BV: "bouvetisland",
        BR: "brazil",
        IO: "britishindianoceanterritory",
        VG: "britishvirginislands",
        BN: "bruneidarussalam",
        BG: "bulgaria",
        BF: "burkinafaso",
        BI: "burundi",
        KH: "cambodia",
        CM: "cameroon",
        CA: "canada",
        CV: "capeverde",
        KY: "caymanislands",
        CF: "centralafricanrepublic",
        TD: "chad",
        CL: "chile",
        CN: "china",
        CX: "christmasisland",
        CC: "cocoskeelingislands",
        CO: "colombia",
        KM: "comoros",
        CG: "congo",
        CD: "congothedemocraticrepof",
        CK: "cookislands",
        CR: "costarica",
        CI: "cotedivoire",
        HR: "croatialocalnamehrvatska",
        CU: "cuba",
        CY: "cyprus",
        CZ: "czechrepublic",
        DK: "denmark",
        DJ: "djibouti",
        DM: "dominica",
        DO: "dominicanrepublic",
        EC: "ecuador",
        EG: "egypt",
        SV: "elsalvador",
        GQ: "equatorialguinea",
        ER: "eritrea",
        EE: "estonia",
        ET: "ethiopia",
        FK: "falklandislandsmalvinas",
        FO: "faroeislands",
        FJ: "fiji",
        FI: "finland",
        FR: "france",
        GF: "frenchguiana",
        PF: "frenchpolynesia",
        TF: "frenchsouthernterritories",
        GA: "gabon",
        GM: "gambia",
        GE: "georgia",
        DE: "germany",
        GH: "ghana",
        GI: "gibraltar",
        GR: "greece",
        GL: "greenland",
        GD: "grenada",
        GP: "guadeloupe",
        GU: "guam",
        GT: "guatemala",
        GG: "guernsey",
        GN: "guinea",
        GW: "guineabissau",
        GY: "guyana",
        HT: "haiti",
        HM: "heardandmcdonaldislands",
        VA: "holyseevaticancitystate",
        HN: "honduras",
        HK: "hongkong",
        HU: "hungary",
        IS: "iceland",
        IN: "india",
        ID: "indonesia",
        IR: "iranislamicrepublicof",
        IQ: "iraq",
        IE: "ireland",
        IM: "isleofman",
        IL: "israel",
        IT: "italy",
        JM: "jamaica",
        JP: "japan",
        JE: "jersey",
        JO: "jordan",
        KZ: "kazakhstan",
        KE: "kenya",
        KI: "kiribati",
        KP: "koreanorth",
        KR: "koreasouth",
        KW: "kuwait",
        KG: "kyrgyzstan",
        LA: "laopeoplesdemocraticrepublic",
        LV: "latvia",
        LB: "lebanon",
        LS: "lesotho",
        LR: "liberia",
        LY: "libyanarabjamahiriya",
        LI: "liechtenstein",
        LT: "lithuania",
        LU: "luxembourg",
        MO: "macau",
        MK: "macedoniathefrmyugoslavrepof",
        MG: "madagascar",
        MW: "malawi",
        MY: "malaysia",
        MV: "maldives",
        ML: "mali",
        MT: "malta",
        MH: "marshallislands",
        MQ: "martinique",
        MR: "mauritania",
        MU: "mauritius",
        YT: "mayotte",
        MX: "mexico",
        FM: "micronesiafederatedstatesof",
        MD: "moldovarepublicof",
        MC: "monaco",
        MN: "mongolia",
        ME: "montenegro",
        MS: "montserrat",
        MA: "morocco",
        MZ: "mozambique",
        MM: "myanmar",
        NA: "namibia",
        NR: "nauru",
        NP: "nepal",
        NL: "netherlands",
        AN: "netherlandsantilles",
        NC: "newcaledonia",
        NZ: "newzealand",
        NI: "nicaragua",
        NE: "niger",
        NG: "nigeria",
        NU: "niue",
        NF: "norfolkisland",
        MP: "northernmarianaislands",
        NO: "norway",
        OM: "oman",
        PK: "pakistan",
        PW: "palau",
        PS: "palestinianterritories",
        PA: "panama",
        PG: "papuanewguinea",
        PY: "paraguay",
        PE: "peru",
        PH: "philippines",
        PN: "pitcairn",
        PL: "poland",
        PT: "portugal",
        PR: "puertorico",
        QA: "qatar",
        RE: "reunion",
        RO: "romania",
        RU: "russianfederation",
        RW: "rwanda",
        KN: "saintkittsandnevis",
        LC: "saintlucia",
        VC: "saintvincentandthegrenadines",
        WS: "samoa",
        SM: "sanmarino",
        ST: "saotomeandprincipe",
        SA: "saudiarabia",
        SN: "senegal",
        RS: "serbia",
        SC: "seychelles",
        SL: "sierraleone",
        SG: "singapore",
        SK: "slovakiaslovakrepublic",
        SI: "slovenia",
        SB: "solomonislands",
        SO: "somalia",
        ZA: "southafrica",
        GS: "southgeorgiasouthsandwichisl",
        ES: "spain",
        LK: "srilanka",
        SH: "sthelena",
        PM: "stpierreandmiquelon",
        SD: "sudan",
        SR: "suriname",
        SJ: "svalbardandjanmayenislands",
        SZ: "swaziland",
        SE: "sweden",
        CH: "switzerland",
        SY: "syrianarabrepublic",
        TW: "taiwanprovinceofchina",
        TJ: "tajikistan",
        TZ: "tanzaniaunitedrepublicof",
        TH: "thailand",
        TL: "timorleste",
        TG: "togo",
        TK: "tokelau",
        TO: "tonga",
        TT: "trinidadandtobago",
        TN: "tunisia",
        TR: "turkey",
        TM: "turkmenistan",
        TC: "turksandcaicosislands",
        TV: "tuvalu",
        UG: "uganda",
        UA: "ukraine",
        AE: "unitedarabemirates",
        GB: "unitedkingdom",
        US: "unitedstates",
        UY: "uruguay",
        UM: "usminoroutlyingislands",
        VI: "usvirginislands",
        UZ: "uzbekistan",
        VU: "vanuatu",
        VE: "venezuela",
        VN: "vietnam",
        WF: "wallisandfutunaislands",
        EH: "westernsahara",
        YE: "yemen",
        ZM: "zambia",
        ZW: "zimbabwe"
    }, h.fn.systemMessage = {
        validateSystemMessageContent: function() {
            var a = $(".dom-content--system-message-content");
            a && a.html().trim().length > 0 && $(".dom-content--system-message").show()
        }
    }, h.fn.systemMessage.validateSystemMessageContent()
}({}, function() {
    return this
}());
