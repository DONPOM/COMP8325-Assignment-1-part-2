/* printio_widget@v2.8.1m,  licensed. http://print.io */
!function(a) {
    var b = function() {
        var b = {
            v: "1",
            hasOpenChannel: !1,
            opts: {},
            lastOpts: {},
            fallbackCountry: "",
            hadConfig: !1,
            fns: {},
            hasLocalStorage: !1,
            hadHistWire: !1,
            pHist: [],
            pHistPointer: 0
        }, c = this, d = function(a) {
            return "pio." + b.v + "." + a
        }, e = {
            debug: !1,
            embedInElement: !1,
            images: [],
            items: [],
            canUseUpload: !0,
            skipProductDetails: !1,
            canBuyOtherProducts: !0,
            canUseMap: !0,
            alwaysUseSingleTemplate: !1,
            hideComingSoonProducts: !1,
            goTo: "",
            goToCustomize: !1,
            orderId: null,
            usePushState: !0,
            fadeBackground: !0,
            countryCode: "",
            currencyCode: "",
            languageCode: "",
            preLoad: {
                on: !1
            },
            fns: {
                onCartChange: function() {},
                onClose: function() {},
                onOpen: function() {},
                onEvent: function() {}
            },
            style: {
                showHeader: !0,
                cssUrl: ""
            },
            iframeStyles: {},
            vars: {},
            recipeId: "",
            url: "http://staging.widget.print.io/widget/",
            helpUrl: "",
            productsUrl: "",
            confirmationUrl: "",
            supportEmail: "",
            coupon: null
        }, f = function(a) {
            return void 0 === a
        }, g = Array.isArray || function(a) {
            return "[object Array]" === Object.prototype.toString.call(a)
        }, h = function(a, c) {
            var d = b;
            return d.fns.hasOwnProperty(a) ? d.fns[a].apply(d.fns[a], c) : void 0
        }, i = function(a) {
            a = a.replace(/[\[]/, "\\[").replace(/[\]]/, "\\]");
            var b = new RegExp("[\\?&]" + a + "=([^&#]*)"), c = b.exec(location.search);
            return null == c ? "" : decodeURIComponent(c[1].replace(/\+/g, " "))
        }, j = function() {
            return a && a.console ? function() {
                console.log.apply(console, arguments)
            } : function() {}
        }, k = function(a, b) {
            var c = {};
            if (!b)
                return a;
            for (var d in a)
                c[d] = b.hasOwnProperty(d) ? g(a[d]) ? b[d].slice(0) : b[d] : g(a[d]) ? a[d].slice(0) : a[d];
            return c
        }, l = function(b) {
            var c = a.location, d = c.protocol + "//" + c.hostname, e = c.pathname, f = c.port;
            return f && 80 != f && (d += ":" + f), d += "/", b ? e.lastIndexOf("/") < 1 ? d : d + e.substr(1, e.lastIndexOf("/") + 1) : d
        }, m = function() {
            var c = b.lastOpts, d = document.createElement("iframe"), e = d.style, g = c.style, h = c.url, i = c.languageCode, j = c.recipeId, k = c.countryCode, l = b.fallbackCountry, m = c.preLoad.on;
            if (i = i || "en", "/" != h[h.length - 1] && (h += "/"), "https://api.print.io/widget/" === h && (h = "https://widget.print.io/widget/"), (k || l) && i ? (d.src += h + (k || l).toLowerCase() + "/" + i.toLowerCase() + "/" + j, c.debug && (d.src += "?debug=true")) : (d.src = h + j, (k || i || c.debug || l) && (d.src += "?", i && (d.src += "language=" + i + "&"), k ? d.src += "country=" + k + "&" : l && (d.src += "country=" + l + "&"), c.debug && (d.src += "debug=true&"), d.src = d.src.substr(0, d.src.length - 1))), d.id = "__pio", e.width = "98%", e.maxWidth = "960px", e.height = "700px", !g || f(g.showHeader) || g.showHeader || (e.height = "650px"), d.frameBorder = 0, d.setAttribute("seamless", "seamless"), d.marginHeight = 0, d.marginWidth = 0, d.hspace = 0, d.vspace = 0, e.margin = "auto", e.boxSizing = "border-box", e.borderRadius = "5px", e.overflow = "none", e.zIndex = 999999, c.embedInElement || (e.position = "absolute", e.left = 0, e.right = 0, e.top = g && g.top ? g.top : (void 0 !== a.pageYOffset ? a.pageYOffset : (document.documentElement || document.body.parentNode || document.body).scrollTop) + 20 + "px"), c.iframeStyles)for (var n in c.iframeStyles)
                e[n] = c.iframeStyles[n];
            return m && (d.id = "__piop", e.width = "0px", e.height = "0px"), d
        }, n = function() {
            return document.getElementById("__pio")
        }, o = function(a) {
            var b = n(), c = document.getElementById("__pio_f"), d = a || h("onClose");
            a && (d=!0), (void 0 === d || d===!0) && (b && b.parentNode.removeChild(b), c && c.parentNode.removeChild(c));
            var e = document.getElementById("__piop");
            e && e.parentNode.removeChild(e)
        }, p = function() {
            return !!b.lastOpts.embedInElement&&!b.lastOpts.preLoad.on
        }, q = function(a) {
            p() ? b.lastOpts.embedInElement.appendChild(a) : document.getElementsByTagName("body")[0].appendChild(a)
        }, r = function() {
            var a = b.lastOpts, c = document.createElement("div"), d = c.style;
            !a.fadeBackground || p() || a.preLoad.on || document.getElementById("__pio_f") || (d.position = "absolute", d.top = "0px", d.left = "0px", d.width = "100%", d.height = Math.max(document.documentElement.scrollHeight, document.height || document.documentElement.clientWidth) + "px", d.background = "rgba(0,0,0,.7)", d.zIndex = 999998, c.id = "__pio_f", document.getElementsByTagName("body")[0].appendChild(c), c.addEventListener("click", function() {
                o()
            }))
        }, s = function(c) {
            if ("close" === c.cmd)
                o();
            else if ("hi" === c.cmd)
                w("set-options", b.lastOpts);
            else if ("onCartChange" === c.cmd)
                B(c.data);
            else if ("paypal" === c.cmd) {
                var d = document.createElement("div"), e = a.location.toString();
                b.lastOpts.confirmationUrl && (e = b.lastOpts.confirmationUrl), d.innerHTML = c.data.replace("!RRR!", e), document.body.appendChild(d);
                var f = document.getElementsByClassName("js-paypal-form")[0];
                f.submit()
            } else if ("analytics-addTransaction" === c.cmd)
                a.ga && ga("ecommerce:addTransaction", c.data);
            else if ("analytics-addItem" === c.cmd)
                a.ga && ga("ecommerce:addItem", c.data);
            else if ("analytics-send" === c.cmd)
                a.ga && ga("ecommerce:send");
            else if ("analytics-event" === c.cmd)
                a.ga && ga("send", c.data);
            else if ("analytics-pageview" === c.cmd)
                a.ga && ga("send", "pageview", c.data);
            else if ("open-page" === c.cmd)
                c.data && a.open(c.data, "_blank");
            else if ("redirect-to-page" === c.cmd)
                c.data && a.location.replace(c.data);
            else if ("help-page" === c.cmd)
                b.lastOpts.helpUrl && a.open(b.lastOpts.helpUrl, "_blank");
            else if ("section-init" === c.cmd) {
                if (b.hadHistWire && c.data.tplName) {
                    if (!b.justPopped) {
                        var g = c.data.tplName;
                        a.history.pushState({
                            n: g
                        }, "", "")
                    }
                    b.justPopped=!1
                }
            } else if ("order-confirmation" === c.cmd) {
                var i = b.lastOpts.confirmationUrl;
                i && (i += i.indexOf("?") > 0 ? "&orderid=" + c.data : "?orderid=" + c.data, a.location.replace(i))
            } else 
                "should-preload" === c.cmd && document.getElementById("__piop") && w("do-preload", {});
            h("onEvent", [c.cmd, c.data])
        }, t = function(a) {
            var b = a.data || a.originalEvent.data;
            if (b && b.data && b.data.cmd && (b.hasOwnProperty("PIOF") || b.hasOwnProperty("PIOP"))) {
                var c = b.data;
                s(c)
            }
        };
        this._m = function(a) {
            s(a)
        };
        var u=!1, v = function() {
            b.hasLocalStorage = y();
            var d = z("countryCode");
            if (d ? b.fallbackCountry = d : E(), !u) {
                var e = a.location.toString(), f = ( - 1 !== e.indexOf("paypalreturn")&&-1 !== e.indexOf("orderid"), i("orderid"));
                f && c.clearCart(), u=!0
            }
        }, w = function(a, b) {
            var c = document.getElementById("__pio") || document.getElementById("__piop");
            c.contentWindow.postMessage({
                PIOP: !0,
                cmd: a,
                data: b
            }, "*")
        }, x = function() {
            b.hasOpenChannel || (a.addEventListener("message", t, !1), b.hasOpenChannel=!0)
        }, y = function() {
            try {
                return "localStorage"in a && null !== a.localStorage
            } catch (b) {
                return !1
            }
        }, z = function(a) {
            var c, e = b.fns.getVal;
            return e ? c = e(d(a)) : b.hasLocalStorage && (c = localStorage[d(a)]), c ? JSON.parse(c) : void 0
        }, A = function(a, c) {
            var e = b.fns.setVal;
            return e ? e(d(a), c ? JSON.stringify(c) : null) : void(b.hasLocalStorage && (localStorage[d(a)] = c ? JSON.stringify(c) : null))
        }, B = function(a) {
            var c = "cart";
            a ? A(c, a) : A(c, null), b.fns && b.fns.onCartChange && b.fns.onCartChange(a)
        }, C = function() {
            var a = z("cart");
            if (a) {
                var b = new Date, c = new Date(b - 864e5);
                if (Date.parse(a.timeStamp) > c)
                    return a
            }
        }, D = function() {
            b.lastOpts.usePushState && (history.pushState({
                to: "start"
            }, null), b.hadHistWire || (a.addEventListener("popstate", function(a) {
                var d = a.state;
                d && d.n && (b.justPopped=!0, j()("PIO: POP! " + d.n), c.msg("goto", d.n))
            }), b.hadHistWire=!0))
        }, E = function() {
            var b = a.document;
            b.addEventListener("DOMContentLoaded", function() {
                var a, c = "pio-js-country", d = b.getElementsByTagName("script")[0];
                b.getElementById(c) || (a = b.createElement("script"), a.id = c, a.async=!0, a.src = "//printio-geo.appspot.com/ip/jsonp?callback=PIO._c", d ? d.parentNode.insertBefore(a, d) : b.head.appendChild(a))
            })
        }, F = function() {
            var a = C();
            return a && a.items ? a.items : []
        }, G = function() {
            var a = b.opts.style, c=!1;
            b.opts = k(b.opts, {
                images: [],
                items: [],
                goTo: "",
                goToCustomize: !1,
                orderId: null
            }), a && a.cssUrl && 0 !== a.cssUrl.indexOf("http") && 0 !== a.cssUrl.indexOf("//") && (0 !== a.cssUrl.indexOf("/") && (c=!0), a.cssUrl = l(c) + a.cssUrl)
        }, H = function() {
            b.opts.preLoad.on && setTimeout(function() {
                c.open({
                    fromPreLoad: !0
                })
            }, 1e3)
        };
        this.config = function(a, c) {
            v();
            var d = k(e, a || {});
            b.opts = d, b.fns = d.fns, b.opts.fns && delete b.opts.fns, c && c(b), G(), b.hadConfig=!0, H()
        }, this.open = function(a, c) {
            var d = a;
            b.lastOpts.preLoad && d.preLoad && b.lastOpts.preLoad.hadPreLoad && (d.preLoad.on=!1), d&&!d.fromPreLoad && (d.preLoad && d.preLoad.on || b.opts.preLoad && b.opts.preLoad.on) && (d.preLoad = {
                on: !1
            }, b.opts.preLoad.on=!1), b.lastOpts = {}, b.lastOpts = b.hadConfig ? k(k(e, b.opts), {}) : k(e, {}), b.lastOpts = k(b.lastOpts, d || {}), d && d.fns && (b.fns = b.hadConfig ? k(b.fns, d.fns) : k(e.fns, d.fns), delete b.lastOpts.fns), v(), F().forEach(function(a) {
                b.lastOpts.items.push(a)
            }), o(!0), x(), D(), r(), q(m()), b.lastOpts.fns && delete b.lastOpts.fns, b.lastOpts.embedInElement && (b.lastOpts.embedInElement = null), b.lastOpts.preLoad.on && (b.lastOpts.preLoad.hadPreLoad=!0, b.lastOpts.preLoad.on=!1), b.opts.preLoad && (b.opts.preLoad.on=!1), c && c(b), b.fns.onOpen && b.fns.onOpen()
        }, this.getOptions = function() {
            return b.lastOpts
        }, this.close = function() {
            s({
                cmd: "close",
                data: {}
            })
        }, this.msg = function(a, b) {
            b = b || 1, w(a, b)
        }, this.showCart = function(a) {
            n() ? w("goto", "tpl-cart") : (a = a || {}, a.goTo = "tpl-cart", c.open(a))
        }, this.clearCart = function() {
            B(null)
        }, this.getCart = C, this.getNumItems = function() {
            v();
            var a = F();
            return a.length ? a.reduce(function(a, b) {
                return a += b.quantity
            }, 0) : 0
        }, this._c = function(a) {
            b.fallbackCountry = a, A("countryCode", a)
        }, v()
    };
    a.PIO = new b, a.PIO.ctor = b
}(window);
