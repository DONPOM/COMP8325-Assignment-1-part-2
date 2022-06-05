(function() {
    function f(a) {
        function b(a, b) {
            var c = new XMLHttpRequest;
            if (!("withCredentials"in c))
                throw Error();
            c.open("GET", a, !0);
            c.onreadystatechange = function() {
                4 === c.readyState && (200 <= c.status && 300 > c.status ? b.b(JSON.parse(c.responseText)) : b.error())
            };
            c.send()
        }
        function c(a, b) {
            function c() {
                try {
                    delete window[e], g.url = null, d.removeChild(g)
                } catch (a) {}
                clearTimeout(k)
            }
            var d = document.getElementsByTagName("head")[0], g = document.createElement("script"), e = "jcb" + Math.round(1E20 * Math.random());
            a += (0 < a.indexOf("?") ? "&" : "?") +
            "callback=" + e;
            d.appendChild(g);
            g.src = a;
            var k = setTimeout(function() {
                b.error();
                c()
            }, 1E4);
            window[e] = function(a) {
                b.b(a);
                c()
            }
        }
        var d = {
            b: function() {},
            error: function() {}
        };
        try {
            b(a, d)
        } catch (g) {
            c(a, d)
        }
        var e = {
            b: function(a) {
                d.b = a;
                return e
            },
            error: function(a) {
                d.error = a;
                return e
            }
        };
        return e
    };
    function h(a) {
        if (!a || "object" !== typeof a || "undefined" !== typeof window && a === window)
            return !1;
        var b = a.length;
        return b && 1 === a.nodeType?!0 : "number" === typeof b && (0 === b || b - 1 in a)
    }
    function l(a, b) {
        if (a)
            if ("[object Array]" !== Object.prototype.toString.apply(a) && (a instanceof Function || "number" !== typeof a.length))
                for (var c in a)
                    a.hasOwnProperty(c) && b(a[c], c, a);
            else if (a.forEach)
                a.forEach(b);
            else {
                c = a.length;
                for (var d = 0; d < c; d++)
                    b(a[d], d, a)
            }
    };
    function p(a) {
        var b = document, c = b.createElement("div"), d = b.createDocumentFragment();
        for (c.innerHTML = a; b = c.firstChild;)
            d.appendChild(b);
        return d
    }
    function q(a) {
        var b = document.createElement("div");
        b.innerHTML = a;
        return b.textContent || b.innerText || ""
    }
    function r(a, b, c) {
        b.addEventListener ? b.addEventListener(a, c, !1) : b.attachEvent && b.attachEvent("on" + a, c)
    }
    function s(a, b, c) {
        b.removeEventListener ? b.removeEventListener(a, c, !1) : b.detachEvent && b.detachEvent("on" + a, c)
    };
    function u(a) {
        var b, c = [];
        l(a, function(a, g) {
            b = encodeURIComponent(g + "") + "=";
            null !== a && "undefined" !== typeof a && (b += encodeURIComponent(a + ""));
            c.push(b)
        });
        c.sort();
        return c.join("&")
    };
    function v(a) {
        a.tmpl || (a.tmpl = new Function("obj", "var s=" + q + ", f=" + w() + ";return ['" + a.innerHTML.replace(/[\r\n]/g, " ").replace(/([\\'])/g, "\\$1").replace(/\$\{(\s*[a-zA-Z0-9_.]+\s*)}/g, "',s('$1' in obj ? obj['$1'] : ''),'").replace(/\$\{(\s*[a-zA-Z0-9_.]+\s*(?:\|\s*([a-zA-Z0-9_]+(?:\(\s*(?:(?:"[^"]*"|\\'[^']*\\'|[\d.]+)(?:\s*,\s*(?:"[^"]*"|\\'[^']*\\'|[\d.]+))*)?\s*\))?))+\s*)}/g, function(a, c) {
            for (var d = c.split("|").reverse(), g = d.pop(), g = "'" + g + "' in obj ? obj['" + g + "'] : ''"; d.length;) {
                var e = d.pop(),
                e = e.replace(/\\'/g, "'"), m = e.substr(0, (e.indexOf("(") + 1 || e.length + 1) - 1), e = e.substring(e.indexOf("(") + 1, e.lastIndexOf(")")).split(","), e = 1 === e.length && "" === e[0].split(" ").join("") ? []: e;
                e.unshift(g);
                g = "f['" + m + "'](" + e.join(",") + ")"
            }
            return "',s(" + g + "),'"
        }) + "'].join('')"));
        return a.tmpl
    }
    function w() {
        if (!y.e) {
            var a = "{";
            l(y, function(b, c) {
                "function" === typeof b && (a += "'" + c + "':" + b + ",")
            });
            y.e = a.substr(0, a.length - 1) + "}"
        }
        return y.e
    }
    var y = {
        format_duration: function(a) {
            var b = Math.floor(a / 3600), c = Math.floor(a / 60)%60;
            a = Math.floor(a)%60;
            return (b ? [b, (10 > c ? "0" : "") + c, (10 > a ? "0" : "") + a] : [c, (10 > a ? "0" : "") + a]).join(":")
        },
        truncate: function(a, b) {
            return a.substr(b)
        },
        replace: function(a, b, c, d) {
            return a.replace(new RegExp(b, d), c)
        }
    };
    function z() {
        var a =+ new Date;
        this.get = function() {
            return (new Date - a) / 1E3
        }
    };
    var A = {
        a: {},
        c: {
            insert: [],
            click: [],
            empty: []
        },
        g: {
            visitor_id: "i",
            placement: "p",
            positions: "n",
            format: "f",
            categories: "c",
            thumbnail_density: "td",
            thumbnail_width: "tw",
            thumbnail_height: "th",
            thumbnail_crop: "tc",
            thumbnail_format: "tf"
        },
        f: function(a, b) {
            void 0 === A.a[a] && (A.a[a] = b)
        },
        set: function(a, b) {
            return A.a[a] = b
        },
        get: function(a, b) {
            return void 0 !== A.a[a] ? A.a[a] : b
        },
        bind: function(a, b) {
            A.c[a] && A.c[a].push(b)
        },
        d: function(a, b) {
            l(A.c[a], function(a) {
                a.apply(null, b)
            })
        },
        refresh: function() {
            l(Array.prototype.slice.call(document.body.getElementsByTagName("script")),
            function(a) {
                if ("text/pxl-tmpl" === a.getAttribute("type") && a.parentNode) {
                    var b = {}, c = new z;
                    l(Array.prototype.slice.call(a.attributes), function(a) {
                        var c = a.name;
                        0 === c.indexOf("data-") && (c = c.substr(5), c = c.split("-").join("_"), c = A.g[c] || c, b[c] = a.value)
                    });
                    b.r = Math.random().toString(36).substr(2, 5);
                    A.get("debug") && (b.debug = A.get("debug"));
                    !b.td && 1 < window.devicePixelRatio && (b.td = window.devicePixelRatio);
                    !b.tf && A.get("use-webp") && (b.tf = "webp");
                    a.setAttribute("type", "t");
                    f(A.get("base_url") + "?" + u(b)).b(function(b) {
                        b =
                        b.units;
                        h(b) && 0 != b.length || A.d("empty");
                        if (h(b)) {
                            var g = "", e = "http" + ("https:" === window.location.protocol ? "s" : "") + "://";
                            l(b, function(b) {
                                "video_id"in b && (b.target_url = [e, "www.dailymotion.com/video/", b.video_id, "?", u(b.parameters)].join(""));
                                g += "<div>" + v(a)(b) + "</div>"
                            });
                            a.parentNode.insertBefore(p("<div>" + g + "</div>"), a.nextSibling);
                            var m = a.nextSibling.childNodes;
                            a.parentNode.removeChild(a);
                            l(b, function(a, b) {
                                var d = m[b], g = a.track = function(b, c) {
                                    if ("tracker_domain"in a && "tracker_id"in a) {
                                        var d = e + a.tracker_domain +
                                        "/ev/" + a.tracker_id + "/" + b;
                                        c && (d += (0 < d.indexOf("?") ? "&" : "?") + u(c));
                                        (new Image).src = d
                                    }
                                };
                                if ("target_url"in a)
                                    for (var x = a.target_url, k = d.getElementsByTagName("A"), n = 0, D = k.length; n < D; n++) {
                                        var t = k[n];
                                        t.href === x && r("click", t, function() {
                                            g("click");
                                            A.d("click", [a, d])
                                        })
                                    }
                                a.track("impression", {
                                    t: c.get()
                                });
                                A.d("insert", [a, d])
                            })
                        }
                    })
                }
            })
        }
    };
    var B = {
        lossy: "UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA",
        lossless: "UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==",
        alpha: "UklGRkoAAABXRUJQVlA4WAoAAAAQAAAAAAAAAAAAQUxQSAwAAAARBxAR/Q9ERP8DAABWUDggGAAAABQBAJ0BKgEAAQAAAP4AAA3AAP7mtQAAAA==",
        animation: "UklGRlIAAABXRUJQVlA4WAoAAAASAAAAAAAAAAAAQU5JTQYAAAD/////AABBTk1GJgAAAAAAAAAAAAAAAAAAAGQAAABWUDhMDQAAAC8AAAAQBxAREYiI/gcA"
    };
    for (var C = window.PXLObject || "pxl", E = (window[C] || {}).q || [], F = window[C] = function() {
        var a = Array.prototype.slice.call(arguments), b = a.shift();
        (b = A[b]) && b.apply(A, a)
    }, G; G = E.shift();)
        F.apply(null, G);
    A.f("base_url", "http://ad.pxlad.io/ad");
    if (0 <= document.location.search.indexOf("pxldebug")) {
        var H = document.location.search.match(/pxldebug(?:=([a-z]+))?/);
        A.set("debug", H[1] || "always")
    }(function(a, b) {
        var c = new Image;
        c.onload = function() {
            b(a, 0 < c.width && 0 < c.height)
        };
        c.onerror = function() {
            b(a, !1)
        };
        c.src = "data:image/webp;base64," + B[a]
    })("lossy", function(a, b) {
        A.f("use-webp", b)
    });
    (function(a) {
        if ("complete" === document.readyState)
            a();
        else {
            var b = function() {
                s("DOMContentLoaded", document, b);
                s("load", window, b);
                a()
            };
            r("DOMContentLoaded", document, b);
            r("load", window, b)
        }
    })(A.refresh);
})();

