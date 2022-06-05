(function() {
    /*

    SPF
    (c) 2012-2014 Google, Inc.
    License: MIT
    */
    function l(a, b, c) {
        var d = Array.prototype.slice.call(arguments, 2);
        return function() {
            var c = d.slice();
            c.push.apply(c, arguments);
            return a.apply(b, c)
        }
    }
    function aa(a, b) {
        if (a) {
            var c = Array.prototype.slice.call(arguments, 1);
            try {
                return a.apply(null, c)
            } catch (d) {
                return d
            }
        }
    }
    function m(a, b) {
        if (document.createEvent) {
            var c = document.createEvent("CustomEvent");
            c.initCustomEvent(a, !0, !0, b);
            return document.dispatchEvent(c)
        }
        return !0
    }
    function n() {
        return (new Date).getTime()
    }
    function ba() {};
    function ca(a) {
        return a.classList ? a.classList : a.className && a.className.match(/\S+/g) || []
    }
    function p(a, b) {
        if (a.classList)
            return a.classList.contains(b);
        for (var c = ca(a), d = 0, e = c.length; d < e; d++)
            if (c[d] == b)
                return !0;
        return !1
    }
    function da(a, b) {
        a.classList ? a.classList.add(b) : p(a, b) || (a.className += " " + b)
    }
    function ea(a, b) {
        if (a.classList)
            a.classList.remove(b);
        else {
            for (var c = ca(a), d = [], e = 0, f = c.length; e < f; e++)
                c[e] != b && d.push(c[e]);
            a.className = d.join(" ")
        }
    };
    function r(a, b) {
        return s[a] = b
    }
    var s = window._spf_state || {};
    window._spf_state = s;
    function t(a, b) {
        if (a.forEach)
            a.forEach(b, void 0);
        else 
            for (var c = 0, d = a.length; c < d; c++)
                c in a && b.call(void 0, a[c], c, a)
    }
    function fa(a, b) {
        if (a.every)
            return a.every(b, void 0);
        for (var c = 0, d = a.length; c < d; c++)
            if (c in a&&!b.call(void 0, a[c], c, a))
                return !1;
        return !0
    }
    function ga(a, b) {
        if (a.map)
            return a.map(b, void 0);
        var c = [];
        c.length = a.length;
        t(a, function(a, e, f) {
            c[e] = b.call(void 0, a, e, f)
        });
        return c
    }
    function u(a) {
        return v(a) ? a : [a]
    }
    function v(a) {
        return "[object Array]" == Object.prototype.toString.call(a)
    };
    var ha = {
        "animation-class": "spf-animate",
        "animation-duration": 425,
        "cache-lifetime": 6E5,
        "cache-max": 50,
        "cache-unified": !1,
        "link-class": "spf-link",
        "nolink-class": "spf-nolink",
        "navigate-limit": 20,
        "navigate-lifetime": 864E5,
        "reload-identifier": null,
        "request-timeout": 0,
        "url-identifier": "?spf=__type__"
    }, w = {};
    "config"in s || r("config", w);
    w = s.config;
    function ia(a, b) {
        var c = b || document;
        return c.querySelectorAll ? c.querySelectorAll(a) : []
    }
    function ja(a) {
        var b, c = a.parentNode;
        if (c && 11 != c.nodeType)
            if (a.removeNode)
                a.removeNode(!1);
            else {
                for (; b = a.firstChild;)
                    c.insertBefore(b, a);
                    c.removeChild(a)
            }
    }
    function ka(a, b, c) {
        for (; a;) {
            if (b(a))
                return a;
            if (c && a == c)
                break;
            a = a.parentNode
        }
        return null
    }
    function la(a, b, c) {
        a = a || "";
        b = b || document;
        var d = b.createElement("iframe");
        d.id = a;
        d.src = 'javascript:""';
        d.style.display = "none";
        c && (d.onload = l(c, null, d));
        b.body.appendChild(d);
        return d
    };
    function ma(a, b, c, d) {
        var e = d || {}, f=!1, h = 0, k, g = new XMLHttpRequest;
        g.open(a, b, !0);
        g.timing = {};
        var q = g.abort;
        g.abort = function() {
            clearTimeout(k);
            g.onreadystatechange = null;
            q.call(g)
        };
        g.onreadystatechange = function() {
            var a = g.timing;
            if (2 == g.readyState) {
                a.responseStart = a.responseStart || n();
                f =- 1 < (g.getResponseHeader("Transfer-Encoding") || "").toLowerCase().indexOf("chunked");
                if (!f) {
                    var a = g.getResponseHeader("X-Firefox-Spdy"), c = window.chrome && chrome.loadTimes && chrome.loadTimes(), c = c && c.wasFetchedViaSpdy;
                    f=!(!a &&
                    !c)
                }
                e.B && e.B(g)
            } else 
                3 == g.readyState ? f && e.m && (a = g.responseText.substring(h), h = g.responseText.length, e.m(g, a)) : 4 == g.readyState && (a.responseEnd = a.responseEnd || n(), window.performance && window.performance.getEntriesByName && (g.resourceTiming = window.performance.getEntriesByName(b)[0]), f && e.m && g.responseText.length > h && (a = g.responseText.substring(h), h = g.responseText.length, e.m(g, a)), clearTimeout(k), e.A && e.A(g))
        };
        a = "POST" == a;
        if (e.headers)
            for (var C in e.headers)
                g.setRequestHeader(C, e.headers[C]), "content-type" ==
                C.toLowerCase() && (a=!1);
        a && g.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
        0 < e.D && (k = setTimeout(function() {
            g.abort();
            e.C && e.C(g)
        }, e.D));
        g.timing.fetchStart = n();
        g.send(c);
        return g
    };
    function x(a, b) {
        var c = a.length - b.length;
        return 0 <= c && a.indexOf(b, c) == c
    }
    var y = String.prototype.trim ? function(a) {
        return a.trim()
    }
    : function(a) {
        return a.replace(/^\s+|\s+$/g, "")
    };
    function z(a) {
        a = a.split("#");
        var b = 1 == a.length;
        return [a[0], b ? "": "#", b ? "": a.slice(1).join("#")]
    };
    function na(a) {
        var b = A();
        a in b && delete b[a]
    }
    function oa() {
        var a = A(), b;
        for (b in a)
            pa(a[b]) || delete a[b];
        a = A();
        b = parseInt(w["cache-max"], 10);
        b = isNaN(b) ? Infinity : b;
        b = Object.keys(a).length - b;
        if (!(0 >= b))
            for (var c = 0; c < b; c++) {
                var d = Infinity, e, f;
                for (f in a)
                    a[f].count < d && (e = f, d = a[f].count);
                    delete a[e]
            }
    }
    function pa(a) {
        if (!(a && "data"in a))
            return !1;
        var b = a.life, b = isNaN(b) ? Infinity: b;
        return n() - a.time < b
    }
    function qa(a) {
        var b = parseInt(s["cache-counter"], 10) || 0;
        b++;
        r("cache-counter", b);
        a.count = b
    }
    function A(a) {
        return !a && "cache-storage"in s ? s["cache-storage"] : r("cache-storage", a || {})
    };
    function ra(a, b, c, d) {
        var e = window.history.state;
        d && e && (b = b || e);
        sa(!0, a, b, c)
    }
    function sa(a, b, c, d) {
        if (b || c) {
            b = b || window.location.href;
            c = c || {};
            var e = n();
            r("history-timestamp", e);
            c["spf-timestamp"] = e;
            if (a)
                ta(c, b);
            else if (a = ua().contentWindow.history.pushState, "function" == typeof a)
                a.call(window.history, c, "", b);
            else 
                throw Error("history.pushState is not a function.");
            r("history-url", b);
            d && (d = s["history-callback"]) && d(b, c)
        }
    }
    function va(a) {
        var b = window.location.href;
        if (s["history-ignore-pop"])
            r("history-ignore-pop", !1);
        else if (a.state) {
            a = a.state;
            var c = a["spf-timestamp"];
            b == s["history-url"] ? (r("history-timestamp", c), ta(a, b)) : (a["spf-back"] = c < parseInt(s["history-timestamp"], 10), a["spf-current"] = s["history-url"], r("history-timestamp", c), r("history-url", b), (c = s["history-callback"]) && c(b, a))
        }
    }
    function ta(a, b) {
        var c = ua().contentWindow.history.replaceState;
        if ("function" == typeof c)
            c.call(window.history, a, "", b);
        else 
            throw Error("history.replaceState is not a function");
    }
    function ua() {
        var a = document.getElementById("history-iframe");
        a || (a = la("history-iframe"));
        return a
    };
    function wa(a, b) {
        a && b && (a in B || (B[a] = []), B[a].push(b))
    }
    function xa(a, b) {
        a in B && b && fa(B[a], function(a, d, e) {
            return a == b ? (e[d] = null, !1) : !0
        })
    }
    function ya(a) {
        a in B && t(B[a], function(a, c, d) {
            d[c] = null;
            a && a()
        })
    }
    var B = {};
    "ps-s"in s || r("ps-s", B);
    B = s["ps-s"];
    function D(a, b, c) {
        var d = E[a];
        return a && b ? (d || (d = E[a] = {
            items: [],
            l: 0,
            i: 0,
            p: 1
        }), d.items.push({
            F: b,
            s: c || 0
        })) : d && d.items.length || 0
    }
    function F(a, b) {
        var c = E[a];
        if (c) {
            var d=!!c.l||!!c.i;
            0 < c.p && (b ||!d) && za(a, b)
        }
    }
    function G(a) {
        (a = E[a]) && a.p--
    }
    function H(a, b) {
        var c = E[a];
        c && (c.p++, F(a, b))
    }
    function Aa(a) {
        var b = E[a];
        b && (Ba(b), delete E[a])
    }
    function Ca(a) {
        var b = parseInt(s.uid, 10) || 0;
        b++;
        return a["spf-key"] || (a["spf-key"] = "" + r("uid", b))
    }
    function za(a, b) {
        var c = E[a];
        if (c && (Ba(c), 0 < c.p && c.items.length)) {
            var d = c.items[0];
            if (d) {
                var e = l(function(a, b) {
                    b();
                    a()
                }, null, l(za, null, a, b));
                b ? (c.items.shift(), e(d.F)) : Da(c, d, e)
            }
        }
    }
    function Da(a, b, c) {
        b.s ? (c = l(c, null, ba), a.i = setTimeout(c, b.s), b.s = 0) : (a.items.shift(), c = l(c, null, b.F), (b = w["advanced-task-scheduler"]) ? a.l = b.K(c) : a.i = setTimeout(c, 0))
    }
    function Ba(a) {
        if (a.l) {
            var b = w["advanced-task-scheduler"];
            b && b.L(a.l);
            a.l = 0
        }
        a.i && (clearTimeout(a.i), a.i = 0)
    }
    var E = {};
    function I(a) {
        var b = document.createElement("a");
        b.href = a;
        b.href = b.href;
        a = {
            href: b.href,
            protocol: b.protocol,
            host: b.host,
            hostname: b.hostname,
            port: b.port,
            pathname: b.pathname,
            search: b.search,
            hash: b.hash,
            J: b.J,
            I: b.I
        };
        a.origin = a.protocol + "//" + a.host;
        a.pathname && "/" == a.pathname[0] || (a.pathname = "/" + a.pathname);
        return a
    }
    function J(a, b) {
        var c = I(a);
        return b ? c.href : z(c.href)[0]
    }
    function Ea(a, b) {
        var c = z(a);
        a = c[0];
        for (var d = 0; d < b.length; d++)
            a = a.replace(new RegExp("([?&])" + b[d] + "(?:=[^&]*)?(?:(?=[&])|$)", "g"), function(a, b) {
                return "?" == b ? b : ""
            });
        x(a, "?") && (a = a.slice(0, - 1));
        return a + c[1] + c[2]
    };
    function Fa(a, b, c, d) {
        var e = "js" == a;
        if (v(b)) {
            var f = b;
            setTimeout(function() {
                throw Error(a + ' load for "' + c + '" called with too many urls ' + f + "; pass only one string");
            }, 0);
            b = b[0]
        }
        b = K(a, b);
        var h = d;
        if (c) {
            var k = Ga(a, b), g = L(a, c);
            !k && g && b != g && (m(e ? "spfjsbeforeunload" : "spfcssbeforeunload", {
                name: c,
                urls: [g],
                url: g
            }), Ha(a, c), h = function() {
                Ia(a, c, g);
                d && d()
            })
        }
        e = c || "^" + b;
        Ja(a, e, b);
        e = M(a, e);
        wa(e, h);
        h = l(Ka, null, a);
        N(a, b) ? h() : (b = La(a, b, h)) && c && b.setAttribute("name", c)
    }
    function Ma(a, b) {
        var c = L(a, b);
        Ha(a, b);
        Ia(a, b, c)
    }
    function Ia(a, b, c) {
        c && (m("js" == a ? "spfjsunload" : "spfcssunload", {
            name: b,
            urls: [c],
            url: c
        }), Na(a, c))
    }
    function Ka(a) {
        var b = M(a, ""), c;
        for (c in B)
            0 == c.indexOf(b) && fa(c.substring(b.length).split("|"), l(Oa, null, a)) && ya(c)
    }
    function La(a, b, c, d, e) {
        function f() {
            N(a, b, e) && Pa(2, a, b, e);
            h && g && g.parentNode && k == document && g.parentNode.removeChild(g);
            c && setTimeout(c, 0);
            return null
        }
        var h = "js" == a;
        b = K(a, b);
        Pa(1, a, b, e);
        var k = d || document, g = k.createElement(h ? "script" : "link");
        if (!b)
            return f();
        d = O(b);
        g.className = M(a, d);
        "onload"in g ? g.onerror = g.onload = f : g.onreadystatechange = function() {
            /^c|loade/.test(g.readyState) && f()
        };
        d = k.getElementsByTagName("head")[0];
        h ? (g.async=!0, g.src = b, d.insertBefore(g, d.firstChild)) : (g.rel = "stylesheet", g.href =
        b, d.appendChild(g));
        return g
    }
    function Na(a, b) {
        b = K(a, b);
        var c = O(b), c = M(a, c), c = ia("." + c, void 0);
        t(c, function(a) {
            a && a.parentNode && a.parentNode.removeChild(a)
        });
        Qa(a, b)
    }
    function Ra(a) {
        var b = "js" == a, c = [];
        t(ia(b ? "script[src]" : 'link[rel~="stylesheet"]'), function(d) {
            var e = b ? d.src: d.href, e = K(a, e);
            if (!N(a, e)) {
                Pa(2, a, e);
                var f = O(e), f = M(a, f);
                da(d, f);
                (f = d.getAttribute("name")) && Ja(a, f, e);
                c.push(d)
            }
        })
    }
    function Sa(a, b) {
        if (b && (b = K(a, b), !N(a, b))) {
            var c = O(b), d = M(a, c), e = M(a, "prefetch"), c = document.getElementById(e);
            if (!c)
                c = la(e, null, function(a) {
                    a.title = e;
                    F(e, !0)
                });
            else if (c.contentWindow.document.getElementById(d))
                return;
            d = l(Ta, null, c, a, b, d, e);
            c.title ? d() : D(e, d)
        }
    }
    function Ta(a, b, c, d, e) {
        var f = "css" == b;
        a = a.contentWindow.document;
        "js" == b ? (b = a.createElement("object"), Ua ? a.createElement("script").src = c : b.data = c, b.id = d, a.body.appendChild(b)) : f ? (b = La(b, c, null, a, e), b.id = d) : (b = a.createElement("img"), b.src = c, b.id = d, a.body.appendChild(b))
    }
    function Va(a, b, c) {
        for (var d = "js" == a, e = L(a, c), f = b.replace(/\s/g, ""), f = f || "", h = 0, k = 0, g = f.length; k < g; ++k)
            h = 31 * h + f.charCodeAt(k), h%=4294967296;
        f = "hash-" + h;
        Ja(a, c, f);
        !Ga(a, f) && (b = Wa(a, b)) && (Pa(2, a, f), b&&!d && (d = O(f), d = M(a, d), b.className = d, b.setAttribute("name", c)), (e = e && e[0]) && Na(a, e))
    }
    function Wa(a, b) {
        b = y(b);
        if (!b)
            return null;
        var c = document.getElementsByTagName("head")[0] || document.body, d;
        "js" == a ? (d = document.createElement("script"), d.text = b, c.appendChild(d), c.removeChild(d)) : (d = document.createElement("style"), c.appendChild(d), "styleSheet"in d ? d.styleSheet.cssText = b : d.appendChild(document.createTextNode(b)));
        return d
    }
    function K(a, b) {
        var c = "rsrc-p-" + a;
        if (b) {
            var d = b.indexOf("//");
            if (0 > d) {
                if (0 == b.lastIndexOf("hash-", 0))
                    return b;
                c = s[c] || "";
                if ("[object String]" == Object.prototype.toString.call(c))
                    b = c + b;
                else 
                    for (var e in c)
                        b = b.replace(e, c[e]);
                b = 0 > b.indexOf("." + a) ? b + "." + a : b;
                b = J(b)
            } else 
                0 == d && (b = J(b))
        }
        return b
    }
    function M(a, b, c) {
        return a + "-" + b + (c ? "-" + c : "")
    }
    function O(a) {
        return a ? String(a).replace(/[^\w]/g, "") : ""
    }
    function Pa(a, b, c, d) {
        b = M(b, c, d);
        P[b] = a
    }
    function N(a, b, c) {
        a = M(a, b, c);
        return P[a]
    }
    function Qa(a, b) {
        var c = M(a, b);
        delete P[c]
    }
    function Ga(a, b) {
        var c = N(a, b);
        return "" == b || 2 == c
    }
    function Ja(a, b, c) {
        a = M(a, b);
        Q[a] = c
    }
    function L(a, b) {
        var c = M(a, b), c = Q[c];
        v(c) && (c = c[0]);
        return c
    }
    function Ha(a, b) {
        var c = M(a, b);
        delete Q[c]
    }
    function Oa(a, b) {
        var c = L(a, b);
        return "" == b || void 0 != c && Ga(a, c)
    }
    var P = {}, Q = {}, Ua =- 1 != navigator.userAgent.indexOf(" Trident/");
    "rsrc-s"in s || r("rsrc-s", P);
    P = s["rsrc-s"];
    "rsrc-u"in s || r("rsrc-u", Q);
    Q = s["rsrc-u"];
    function Xa(a) {
        a = u(a);
        t(a, function(a) {
            Sa("img", a)
        })
    };
    function Ya(a, b, c) {
        Fa("js", a, b, c)
    }
    function Za(a) {
        Ma("js", a)
    }
    function $a(a, b) {
        La("js", a, b)
    }
    function ab(a) {
        a = u(a);
        t(a, function(a) {
            Sa("js", a)
        })
    }
    function bb(a, b, c) {
        a = u(a);
        var d = [];
        t(a, function(a) {
            a && void 0 == L("js", a) && d.push(a)
        });
        var e=!d.length;
        if (b) {
            var f = fa(a, l(Oa, null, "js"));
            e && f ? b() : (a = M("js", a.sort().join("|")), wa(a, b))
        }
        c&&!e && c(d)
    }
    function cb(a, b) {
        a = u(a);
        t(a, function(a) {
            if (a) {
                var b = R[a] || a;
                v(b) && (b = b[0]);
                var b = K("js", b), e = L("js", a);
                e && b != e && db(a)
            }
        });
        bb(a, b, eb)
    }
    function eb(a) {
        t(a, function(a) {
            function c() {
                Ya(e, a)
            }
            var d = S[a], e = R[a] || a;
            v(e) && (e = e[0]);
            d ? cb(d, c) : c()
        })
    }
    function db(a) {
        a = u(a);
        t(a, function(a) {
            var c = [], d;
            for (d in S) {
                var e = S[d], e = u(e);
                t(e, function(e) {
                    e == a && c.push(d)
                })
            }
            t(c, function(a) {
                db(a)
            });
            Za(a)
        })
    }
    function fb(a, b) {
        Va("js", a, b)
    }
    function gb(a) {
        Wa("js", a)
    }
    var S = {};
    "js-d"in s || r("js-d", S);
    var S = s["js-d"], R = {};
    "js-u"in s || r("js-u", R);
    R = s["js-u"];
    function hb(a, b, c) {
        Fa("css", a, b, c)
    }
    function ib(a, b) {
        La("css", a, b)
    }
    function jb(a) {
        a = u(a);
        t(a, function(a) {
            Sa("css", a)
        })
    };
    function kb(a, b, c) {
        if (b) {
            var d = [], e;
            b = 0;
            c && (a += "\r\n");
            var f = a.indexOf("[\r\n", b);
            for ( - 1 < f && (b = f + 3); - 1 < (f = a.indexOf(",\r\n", b));)
                e = y(a.substring(b, f)), b = f + 3, e && d.push(JSON.parse(e));
            f = a.indexOf("]\r\n", b);
            - 1 < f && (e = y(a.substring(b, f)), b = f + 3, e && d.push(JSON.parse(e)));
            e = "";
            a.length > b && (e = a.substring(b), c && x(e, "\r\n") && (e = e.substring(0, e.length - 2)));
            return {
                k: d,
                a: e
            }
        }
        a = JSON.parse(a);
        "number" == typeof a.length ? d = a : d = [a];
        return {
            k: d,
            a: ""
        }
    }
    function T(a, b, c, d, e) {
        var f = "process " + J(a), h=!w["experimental-process-async"], k;
        k = 0;
        b.timing || (b.timing = {});
        b.title && (document.title = b.title);
        d && b.url && J(b.url) != J(window.location.href) && ra(b.url + window.location.hash, null, !1, !0);
        b.head && (k = l(function(a, b) {
            var c = U(a);
            lb(c);
            mb(c);
            G(f);
            nb(c, function() {
                b.spfProcessHead = n();
                H(f, h)
            })
        }, null, b.head, b.timing), k = D(f, k));
        b.attr && (k = l(function(a, b) {
            for (var c in a) {
                var d = document.getElementById(c);
                if (d) {
                    var e = a[c], f = void 0;
                    for (f in e) {
                        var g = e[f];
                        "class" == f ? d.className =
                        g : "style" == f ? d.style.cssText = g : d.setAttribute(f, g)
                    }
                }
            }
            b.spfProcessAttr = n()
        }, null, b.attr, b.timing), k = D(f, k));
        d = b.body || {};
        var g = k, q;
        for (q in d)
            k = l(function(a, b) {
                var c = document.getElementById(a);
                if (c) {
                    var d = U(b), g = w["animation-class"];
                    if (ob && p(c, g)) {
                        G(f);
                        var k = Ca(c);
                        F(k, !0);
                        g = {
                            r: d,
                            reverse: !!e,
                            f: null,
                            h: null,
                            g: c,
                            G: g + "-old",
                            H: g + "-new",
                            w: e ? g + "-reverse-start": g + "-forward-start",
                            v: e ? g + "-reverse-end": g + "-forward-end"
                        };
                        c = l(function(a) {
                            mb(a.r);
                            da(a.g, a.w);
                            a.f = document.createElement("div");
                            a.f.className = a.G;
                            var b =
                            a.g, c = a.f;
                            if (c) {
                                for (var d; d = b.firstChild;)
                                    c.appendChild(d);
                                    b.appendChild(c)
                            }
                            a.h = document.createElement("div");
                            a.h.className = a.H;
                            a.h.innerHTML = a.r.j;
                            a.reverse ? (b = a.f, b.parentNode.insertBefore(a.h, b)) : (b = a.f, b.parentNode.insertBefore(a.h, b.nextSibling))
                        }, null, g);
                        D(k, c, 0);
                        c = l(function(a) {
                            ea(a.g, a.w);
                            da(a.g, a.v)
                        }, null, g);
                        D(k, c, 0);
                        c = l(function(a) {
                            a.g.removeChild(a.f);
                            ea(a.g, a.v);
                            ja(a.h);
                            G(k);
                            nb(a.r, function() {
                                H(k)
                            })
                        }, null, g);
                        D(k, c, parseInt(w["animation-duration"], 10));
                        c = l(function(a, b) {
                            H(b)
                        }, null, g, f);
                        D(k, c);
                        F(k)
                    } else {
                        mb(d);
                        var q = function() {
                            G(f);
                            nb(d, function() {
                                H(f, h)
                            })
                        };
                        (g = w["experimental-html-handler"]) ? (G(f), g(d.j, c, function() {
                            q();
                            H(f, h)
                        })) : (c.innerHTML = d.j, q())
                    }
                }
            }, null, q, d[q], b.timing), k = D(f, k);
        q = k - g;
        b.foot ? (k = l(function(a, b, c) {
            c && (b.spfProcessBody = n());
            a = U(a);
            mb(a);
            G(f);
            nb(a, function() {
                b.spfProcessFoot = n();
                H(f, h)
            })
        }, null, b.foot, b.timing, q), k = D(f, k)) : q && (k = l(function(a) {
            a.spfProcessBody = n()
        }, null, b.timing), k = D(f, k));
        c && (k = D(f, l(c, null, a, b)));
        F(f, h)
    }
    function pb(a, b, c) {
        var d = "preprocess " + J(a), e;
        b.head && (e = l(function(a) {
            a = U(a);
            lb(a);
            qb(a);
            rb(a)
        }, null, b.head), D(d, e));
        var f = b.body || {}, h;
        for (h in f)
            f[h] && (e = l(function(a, b) {
                var c = U(b);
                qb(c);
                rb(c)
            }, null, h, f[h]), D(d, e));
        b.foot && (e = l(function(a) {
            a = U(a);
            qb(a);
            rb(a)
        }, null, b.foot), D(d, e));
        c && D(d, l(c, null, a, b));
        F(d)
    }
    function U(a) {
        var b = new sb;
        if (!a)
            return b;
        if ("[object String]" != Object.prototype.toString.call(a))
            return a.scripts && t(a.scripts, function(a) {
                b.d.push({
                    url: a.url || "",
                    text: a.text || "",
                    name: a.name || "",
                    async: a.async ||!1
                })
            }), a.styles && t(a.styles, function(a) {
                b.c.push({
                    url: a.url || "",
                    text: a.text || "",
                    name: a.name || ""
                })
            }), a.links && w["experimental-preconnect"] && t(a.links, function(a) {
                "spf-preconnect" == a.rel && b.o.push({
                    url: a.url || "",
                    rel: a.rel || ""
                })
            }), b.j = a.html || "", b;
        a = a.replace(tb, function(a, d, e, f) {
            return "script" ==
            d ? (a = (a = e.match(ub)) ? a[1] : "", d = (d = e.match(vb)) ? d[1] : "", e = wb.test(e), b.d.push({
                url: d,
                text: f,
                name: a,
                async: e
            }), "") : "style" == d ? (a = (a = e.match(ub)) ? a[1] : "", b.c.push({
                url: "",
                text: f,
                name: a
            }), "") : a
        });
        a = a.replace(xb, function(a, d) {
            var e = d.match(yb), e = e ? e[1]: "";
            if ("stylesheet" == e) {
                var e = (e = d.match(ub)) ? e[1]: "", f = d.match(zb), f = f ? f[1]: "";
                b.c.push({
                    url: f,
                    text: "",
                    name: e
                });
                return ""
            }
            return "spf-preconnect" == e && w["experimental-preconnect"] ? (f = (f = d.match(zb)) ? f[1] : "", b.o.push({
                url: f,
                rel: e
            }), "") : a
        });
        b.j = a;
        return b
    }
    function nb(a, b) {
        if (0 >= a.d.length)
            b && b();
        else {
            var c =- 1, d = function() {
                c++;
                if (c < a.d.length) {
                    var e = a.d[c], f = function() {};
                    e.url ? f = e.name ? l(Ya, null, e.url, e.name) : l($a, null, e.url) : e.text && (f = e.name ? l(fb, null, e.text, e.name) : l(gb, null, e.text));
                    e.url&&!e.async ? f(d) : (f(), d())
                } else 
                    b && b()
            };
            d()
        }
    }
    function rb(a) {
        0 >= a.d.length || (a = ga(a.d, function(a) {
            return a.url
        }), ab(a))
    }
    function mb(a) {
        if (!(0 >= a.c.length))
            for (var b = 0, c = a.c.length; b < c; b++) {
                var d = a.c[b];
                d.url ? d.name ? hb(d.url, d.name) : ib(d.url) : d.text && (d.name ? Va("css", d.text, d.name) : Wa("css", d.text))
            }
    }
    function qb(a) {
        0 >= a.c.length || (a = ga(a.c, function(a) {
            return a.url
        }), jb(a))
    }
    function lb(a) {
        0 >= a.o.length || (a = ga(a.o, function(a) {
            return "spf-preconnect" == a.rel ? a.url : ""
        }), Xa(a))
    }
    function sb() {
        this.j = "";
        this.d = [];
        this.c = [];
        this.o = []
    }
    var ob = function() {
        var a = document.createElement("div");
        if ("transition"in a.style)
            return !0;
        for (var b = ["webkit", "Moz", "Ms", "O", "Khtml"], c = 0, d = b.length; c < d; c++)
            if (b[c] + "Transition"in a.style)
                return !0;
        return !1
    }(), xb = /\x3clink([\s\S]*?)\x3e/ig, tb = /\x3c(script|style)([\s\S]*?)\x3e([\s\S]*?)\x3c\/\1\x3e/ig, wb = /(?:\s|^)async(?:\s|=|$)/i, zb = /(?:\s|^)href\s*=\s*["']?([^\s"']+)/i, ub = /(?:\s|^)name\s*=\s*["']?([^\s"']+)/i, yb = /(?:\s|^)rel\s*=\s*["']?([^\s"']+)/i, vb = /(?:\s|^)src\s*=\s*["']?([^\s"']+)/i;
    function Ab(a, b) {
        var c = b || {};
        c.method = ((c.method || "GET") + "").toUpperCase();
        c.type = c.type || "request";
        var d = a, e = w["url-identifier"] || "";
        if (e) {
            var e = e.replace("__type__", c.type || ""), f = z(d), d = f[0];
            0 == e.lastIndexOf("?", 0)&&-1 != d.indexOf("?") && (e = e.replace("?", "&"));
            d += e + f[1] + f[2]
        }
        d = J(d);
        e = {};
        e.spfUrl = d;
        e.startTime = n();
        e.fetchStart = e.startTime;
        n: {
            var h = Bb(a, c.current, null, c.type, !1), k = c.current, f = [];
            k && (f.push(h + " previous " + k), f.push(h + " previous " + I(k).pathname));
            f.push(h);
            h = 0;
            for (k = f.length; h < k; h++) {
                var g;
                r:
                {
                    g = f[h];
                    var q = A();
                    if (g in q) {
                        q = q[g];
                        if (pa(q)) {
                            qa(q);
                            g = q.data;
                            break r
                        }
                        na(g)
                    }
                    g = void 0
                }
                if (g) {
                    f = {
                        key: f[h],
                        response: g.response,
                        type: g.type
                    };
                    break n
                }
            }
            f = null
        }
        e.spfPrefetched=!!f && "prefetch" == f.type;
        e.spfCached=!!f;
        if (f)
            return setTimeout(l(Cb, null, a, c, e, f.key, f.response), 0), null;
        f = {};
        void 0 != c.u && (f["X-SPF-Referer"] = c.u);
        c.current && (f["X-SPF-Previous"] = c.current);
        if (h = w["advanced-header-identifier"])
            f["X-SPF-Request"] = h.replace("__type__", c.type), f.Accept = "application/json";
        h = {
            t: !1,
            a: "",
            complete: []
        };
        e =
        l(Db, null, a, c, e, h);
        e = {
            headers: f,
            D: w["request-timeout"],
            B: l(Eb, null, a, h),
            m: l(Fb, null, a, c, h),
            A: e,
            C: e
        };
        return "POST" == c.method ? ma("POST", d, c.q, e) : ma("GET", d, null, e)
    }
    function Cb(a, b, c, d, e) {
        var f=!1;
        c.responseStart = c.responseEnd = n();
        b.type && 0 == b.type.lastIndexOf("navigate", 0) && (c.navigationStart = c.startTime, w["cache-unified"] || (na(d), f=!0));
        if (b.b && "multipart" == e.type) {
            d = e.parts;
            for (var h = 0; h < d.length; h++)
                b.b(a, d[h])
        }
        Gb(a, b, c, e, f)
    }
    function Eb(a, b, c) {
        a = c.getResponseHeader("X-SPF-Response-Type") || "";
        b.t =- 1 != a.toLowerCase().indexOf("multipart")
    }
    function Fb(a, b, c, d, e, f) {
        if (c.t) {
            e = c.a + e;
            var h;
            try {
                h = kb(e, !0, f)
            } catch (k) {
                d.abort();
                b.e && b.e(a, k);
                return 
            }
            if (b.b)
                for (d = 0; d < h.k.length; d++)
                    b.b(a, h.k[d]);
            c.complete = c.complete.concat(h.k);
            c.a = h.a
        }
    }
    function Db(a, b, c, d, e) {
        if (e.timing)
            for (var f in e.timing)
                c[f] = e.timing[f];
        if (e.resourceTiming)
            if ("load" == b.type)
                for (var h in e.resourceTiming)
                    c[h] = e.resourceTiming[h];
            else if (window.performance && window.performance.timing) {
                f = window.performance.timing.navigationStart;
                for (var k in e.resourceTiming)
                    h = e.resourceTiming[k], void 0 !== h && (x(k, "Start") || x(k, "End") || "startTime" == k) && (c[k] = f + Math.round(h))
            }
        "load" != b.type && (c.navigationStart = c.startTime);
        d.complete.length && (d.a = y(d.a), d.a && Fb(a, b, d, e, "", !0));
        var g;
        try {
            g = kb(e.responseText).k
        } catch (q) {
            b.e && b.e(a, q);
            return 
        }
        if (b.b && 1 < g.length)
            for (d = d.complete.length; d < g.length; d++)
                b.b(a, g[d]);
        if (1 < g.length) {
            var C;
            d = 0;
            for (e = g.length; d < e; d++)
                k = g[d], k.cacheType && (C = k.cacheType);
            g = {
                parts: g,
                type: "multipart"
            };
            C && (g.cacheType = C)
        } else 
            g = 1 == g.length ? g[0] : {};
        Gb(a, b, c, g, !0)
    }
    function Gb(a, b, c, d, e) {
        if (e && "POST" != b.method && (e = Bb(a, b.current, d.cacheType, b.type, !0))) {
            d.cacheKey = e;
            var f = {
                response: d,
                type: b.type || ""
            }, h = parseInt(w["cache-lifetime"], 10), k = parseInt(w["cache-max"], 10);
            0 >= h || 0 >= k || (k = A(), f = {
                data: f,
                life: h,
                time: n(),
                count: 0
            }, qa(f), k[e] = f, setTimeout(oa, 1E3))
        }
        d.timing = c;
        b.n && b.n(a, d)
    }
    function Bb(a, b, c, d, e) {
        a = J(a);
        var f;
        w["cache-unified"] ? f = a : "navigate-back" == d || "navigate-forward" == d ? f = "history " + a : "navigate" == d ? f = (e ? "history " : "prefetch ") + a : "prefetch" == d && (f = e ? "prefetch " + a : "");
        b && "url" == c ? f += " previous " + b : b && "path" == c && (f += " previous " + I(b).pathname);
        return f || ""
    };
    function Hb(a) {
        return ka(a, function(a) {
            return p(a, w["link-class"])
        })
    }
    function Ib(a) {
        return ka(a, function(a) {
            return p(a, w["nolink-class"])
        })
    }
    function Jb(a, b) {
        return ka(a, function(a) {
            return a.href && "img" != a.tagName.toLowerCase()
        }, b)
    }
    function Kb(a) {
        if (a.metaKey || a.altKey || a.ctrlKey || a.shiftKey || 0 < a.button)
            return null;
        var b = Hb(a.target);
        return !b || w["nolink-class"] && Ib(a.target) ? null : (a = Jb(a.target, b)) ? a.href : null
    }
    function Lb(a) {
        return I(a).origin != I(window.location.href).origin?!1 : !0
    }
    function Mb() {
        if (!s["nav-init"])
            return !1;
        var a = parseInt(s["nav-counter"], 10) || 0;
        a++;
        var b = parseInt(w["navigate-limit"], 10), b = isNaN(b) ? Infinity: b;
        if (a > b)
            return !1;
        a = parseInt(s["nav-init-time"], 10);
        a--;
        a = n() - a;
        b = parseInt(w["navigate-lifetime"], 10);
        b = isNaN(b) ? Infinity : b;
        return a > b?!1 : !0
    }
    function Nb(a) {
        if (!a.defaultPrevented) {
            var b = Kb(a);
            b && Lb(b) && Mb() && m("spfclick", {
                url: b,
                target: a.target
            }) && (Ob(b), a.preventDefault())
        }
    }
    function Pb(a) {
        var b = Kb(a);
        b && setTimeout(function() {
            Qb(b, void 0)
        }, 0)
    }
    function Rb(a, b) {
        var c=!(!b ||!b["spf-back"]), d = b && b["spf-referer"], e = b && b["spf-current"], f = w["reload-identifier"];
        f && (a = Ea(a, [f]));
        Lb(a) ? Mb() ? m("spfhistory", {
            url: a,
            referer: d,
            previous: e
        }) && Ob(a, null, e, d, !0, c) : V(a, "1") : V(a, "9")
    }
    function Ob(a, b, c, d, e, f) {
        W();
        b = b || {};
        d = void 0 == d ? window.location.href : d;
        c = e ? c : window.location.href;
        var h;
        n: {
            if ( - 1 != a.indexOf("#") && (h = J(a), z(c || window.location.href)[0] == h)) {
                h=!1;
                break n
            }
            h=!0
        }
        if (h)
            if (Sb(a, d, c, b)) {
                r("nav-counter", (parseInt(s["nav-counter"], 10) || 0) + 1);
                Tb(a);
                h = J(a);
                var k = "preprocess " + J(h), g;
                for (g in E)
                    k != g && 0 == g.lastIndexOf("preprocess", 0) && Aa(g);
                    g = Ub()[h];
                    r("nav-request", g);
                    r("nav-promote", null);
                    r("nav-promote-time", null);
                    g && 4 != g.readyState ? (c=!!e, f = "preprocess " + J(a), e = "promote " +
                    J(a), r("nav-promote", a), r("nav-promote-time", n()), Aa(f), F(e, !0), c || Vb(a, d, l(X, null, b))) : (e=!!e, g=!!f, f = l(X, null, b), h = l(Wb, null, b, g), k = l(Xb, null, b, g, ""), w["advanced-navigate-persist-timing"] || Yb(), b = Ab(a, {
                        method: b.method,
                        b: h,
                        e: f,
                        n: k,
                        q: b.postData,
                        type: "navigate" + (e ? g ? "-back" : "-forward" : ""),
                        current: c,
                        u: d
                    }), r("nav-request", b), e || Vb(a, d, f))
            } else 
                V(a, "2");
        else 
            Zb(a), e || Vb(a, d, l(X, null, b))
    }
    function Zb(a) {
        a = z(a);
        a[1] && (a[2] ? (a = document.getElementById(a[2])) && a.scrollIntoView() : window.scroll(0, 0))
    }
    function Vb(a, b, c) {
        try {
            b = {
                "spf-referer": b
            }, J(a, !0) == window.location.href ? ra(a, null, !1, !0) : sa(!1, a, b, void 0)
        } catch (d) {
            W(), c(a, d)
        }
    }
    function X(a, b, c) {
        r("nav-request", null);
        $b(b, c, a) && V(b, "10", c)
    }
    function Wb(a, b, c, d) {
        if (ac(c, d, a))
            if (d.reload)
                V(c, "5");
            else if (d.redirect)
                bc(a, d.redirect);
            else 
                try {
                    T(c, d, function() {
                        cc(c, d, a)
                    }, !0, b)
        } catch (e) {
            X(a, c, e)
        } else 
            V(c, "3")
    }
    function Xb(a, b, c, d, e) {
        r("nav-request", null);
        s["nav-promote"] == c && (c = e.timing || {}, c.navigationStart = s["nav-promote-time"], c.spfPrefetched=!0);
        c = "multipart" == e.type;
        if (!c) {
            if (!dc(d, e, a)) {
                V(d, "4");
                return 
            }
            if (e.reload) {
                V(d, "5");
                return 
            }
            if (e.redirect) {
                bc(a, e.redirect);
                return 
            }
        }
        try {
            T(d, c ? {} : e, function() {
                Zb(d);
                ec(d, e, a)
            }, !0, b)
        } catch (f) {
            X(a, d, f)
        }
    }
    function bc(a, b) {
        try {
            b += window.location.hash, ra(b, null, !0, !0)
        } catch (c) {
            W(), X(a, b, c)
        }
    }
    function W() {
        var a = s["nav-request"];
        a && (a.abort(), r("nav-request", null))
    }
    function Y(a, b) {
        var c;
        a && (c = Array.prototype.slice.call(arguments), c[0] = a, c = aa.apply(null, c));
        return !1 !== c
    }
    function V(a, b, c) {
        c = c ? c.message : "";
        W();
        Tb();
        var d = b;
        c && (d += " Message: " + c);
        m("spfreload", {
            url: a,
            reason: d
        });
        w["experimental-remove-history"] && window.location.href == a && (r("history-ignore-pop", !0), window.history.back());
        setTimeout(function() {
            var c = w["reload-identifier"];
            if (c) {
                var d = {};
                d[c] = encodeURIComponent(b);
                var c = a, h = z(c), c = h[0], k =- 1 != c.indexOf("?") ? "&" : "?", g;
                for (g in d)
                    c += k + g, d[g] && (c += "=" + d[g]), k = "&";
                a = c + h[1] + h[2]
            }
            window.location.href = a
        }, 0)
    }
    function fc(a, b, c) {
        b = b || {};
        c = c || a;
        Sb(a, void 0, void 0, b, !0) && Ab(a, {
            method: b.method,
            b: l(gc, null, !1, b, c),
            e: l(hc, null, !1, b, c),
            n: l(ic, null, !1, b, c),
            q: b.postData,
            type: "load"
        })
    }
    function Qb(a, b, c) {
        b = b || {};
        c = c || a;
        var d = window.location.href;
        Sb(a, void 0, void 0, b, !0) && (b = Ab(a, {
            method: b.method,
            b: l(gc, null, !0, b, c),
            e: l(hc, null, !0, b, c),
            n: l(ic, null, !0, b, c),
            q: b.postData,
            type: "prefetch",
            current: d
        }), a = J(a), Ub()[a] = b)
    }
    function hc(a, b, c, d, e) {
        a && jc(d);
        a && s["nav-promote"] == c ? X(b, d, e) : $b(d, e, b, !0)
    }
    function gc(a, b, c, d, e) {
        if (ac(d, e, b, !0)) {
            if (e.reload) {
                if (!a)
                    return;
                if (s["nav-promote"] == c) {
                    V(d, "5");
                    return 
                }
            }
            if (e.redirect)
                kc(a, b, c, e.redirect);
            else {
                if (a) {
                    var f = l(Wb, null, b, !1, d, e), h = "promote " + J(c);
                    D(h, f);
                    if (s["nav-promote"] == c) {
                        F(h, !0);
                        return 
                    }
                }(a ? pb : T)(d, e, function() {
                    cc(d, e, b, !0)
                })
            }
        }
    }
    function ic(a, b, c, d, e) {
        var f = "multipart" == e.type;
        if (!f) {
            if (!dc(d, e, b, !0)) {
                V(d, "4");
                return 
            }
            if (e.reload) {
                if (!a)
                    return;
                if (s["nav-promote"] == c) {
                    V(d, "5");
                    return 
                }
            }
            if (e.redirect) {
                kc(a, b, c, e.redirect);
                return 
            }
        }
        var h = "promote " + J(c);
        if (a) {
            jc(d);
            if (s["nav-promote"] == c) {
                D(h, l(Xb, null, b, !1, c, d, e));
                F(h, !0);
                return 
            }
            Aa(h)
        }
        h = a ? pb : T;
        try {
            h(d, f ? {} : e, function() {
                ec(d, e, b, !0)
            })
        } catch (k) {
            hc(a, b, c, d, k)
        }
    }
    function kc(a, b, c, d) {
        a = a ? Qb : fc;
        var e = {};
        t("onError onRequest onPartProcess onPartDone onProcess onDone".split(" "), function(a) {
            e[a] = b[a]
        });
        a(d, e, c)
    }
    function $b(a, b, c, d) {
        a = {
            url: a,
            err: b
        };
        (c = Y((c || {}).onError, a))&&!d && (c = m("spferror", a));
        return c
    }
    function Sb(a, b, c, d, e) {
        a = {
            url: a,
            referer: b,
            previous: c
        };
        (d = Y((d || {}).onRequest, a))&&!e && (d = m("spfrequest", a));
        return d
    }
    function ac(a, b, c, d) {
        a = {
            url: a,
            part: b
        };
        (c = Y((c || {}).onPartProcess, a))&&!d && (c = m("spfpartprocess", a));
        return c
    }
    function cc(a, b, c, d) {
        a = {
            url: a,
            part: b
        };
        Y((c || {}).onPartDone, a)&&!d && m("spfpartdone", a)
    }
    function dc(a, b, c, d) {
        a = {
            url: a,
            response: b
        };
        (c = Y((c || {}).onProcess, a))&&!d && (c = m("spfprocess", a));
        return c
    }
    function ec(a, b, c, d) {
        a = {
            url: a,
            response: b
        };
        Y((c || {}).onDone, a)&&!d && m("spfdone", a)
    }
    function jc(a) {
        a = J(a);
        var b = Ub(), c = b[a];
        c && c.abort();
        delete b[a]
    }
    function Tb(a) {
        var b = Ub();
        a = a && J(a);
        for (var c in b)
            a != c && jc(c)
    }
    var Yb, lc = window.performance && (window.performance.clearResourceTimings || window.performance.webkitClearResourceTimings || window.performance.mozClearResourceTimings || window.performance.msClearResourceTimings || window.performance.oClearResourceTimings);
    Yb = lc ? l(lc, window.performance) : ba;
    function Ub() {
        return "nav-prefetches"in s ? s["nav-prefetches"] : r("nav-prefetches", {})
    };
    function Z() {
        Ra("js");
        Ra("css");
        "complete" == document.readyState && (document.removeEventListener ? document.removeEventListener("DOMContentLoaded", Z, !1) : document.detachEvent && document.detachEvent("onreadystatechange", Z))
    }
    document.addEventListener ? document.addEventListener("DOMContentLoaded", Z, !1) : document.attachEvent && document.attachEvent("onreadystatechange", Z);
    Z();
    var mc = {
        init: function(a) {
            var b=!("function" != typeof window.history.pushState&&!ua().contentWindow.history.pushState);
            a = a || {};
            for (var c in ha)
                w[c] = c in a ? a[c] : ha[c];
            for (c in a)
                c in ha || (w[c] = a[c]);
            if (b) {
                c = $b;
                if (!s["history-init"] && window.addEventListener) {
                    a = window.location.href;
                    window.addEventListener("popstate", va, !1);
                    r("history-init", !0);
                    r("history-callback", Rb);
                    r("history-error-callback", c);
                    r("history-listener", va);
                    r("history-url", a);
                    r("history-timestamp", n());
                    var d = {
                        "spf-referer": document.referrer
                    };
                    try {
                        ra(a, d)
                    } catch (e) {
                        c && c(a, e)
                    }
                }
                !s["nav-init"] && document.addEventListener && (document.addEventListener("click", Nb, !1), !w["experimental-prefetch-mousedown"] || "ontouchstart"in window || 0 < window.navigator.maxTouchPoints || 0 < window.navigator.msMaxTouchPoints || (document.addEventListener("mousedown", Pb, !1), r("prefetch-listener", Pb)), r("nav-init", !0), r("nav-init-time", n()), r("nav-counter", 0), r("nav-listener", Nb))
            }
            return b
        },
        dispose: function() {
            "undefined" != typeof History && History.prototype.pushState && (W(), s["nav-init"] &&
            (document.removeEventListener && (document.removeEventListener("click", s["nav-listener"], !1), w["experimental-prefetch-mousedown"] && document.removeEventListener("mousedown", s["prefetch-listener"], !1)), r("nav-init", !1), r("nav-init-time", null), r("nav-counter", null), r("nav-listener", null)), s["history-init"] && (window.removeEventListener && window.removeEventListener("popstate", s["history-listener"], !1), r("history-init", !1), r("history-callback", null), r("history-error-callback", null), r("history-listener", null),
            r("history-url", null), r("history-timestamp", 0)));
            for (var a in w)
                delete w[a]
        },
        navigate: function(a, b) {
            a && (Lb(a) ? Mb() ? Ob(a, b) : V(a, "1") : V(a, "9"))
        },
        load: function(a, b) {
            fc(a, b)
        },
        prefetch: function(a, b) {
            Qb(a, b)
        },
        process: function(a, b) {
            function c(a, c, d, e) {
                a == c && b && b(e)
            }
            var d = window.location.href;
            if ("multipart" == a.type)
                for (var e = a.parts, f = 0; f < e.length; f++) {
                    var h = l(c, null, f, e.length - 1);
                    T(d, e[f], h)
                } else 
                    h = l(c, null, 0, 0), T(d, a, h)
        }
    }, nc = {
        cache: {
            remove: na,
            clear: function() {
                A({})
            }
        },
        script: {
            load: Ya,
            get: $a,
            ready: bb,
            done: function(a) {
                Ja("js",
                a, "");
                Ka("js")
            },
            require: cb,
            declare: function(a, b) {
                if (a) {
                    for (var c in a)
                        S[c] = a[c];
                    if (b)
                        for (c in b)
                            R[c] = b[c]
                }
            },
            path: function(a) {
                r("rsrc-p-js", a)
            },
            unload: Za,
            ignore: function(a, b) {
                a = u(a);
                var c = M("js", a.sort().join("|"));
                xa(c, b)
            },
            unrequire: db,
            prefetch: ab
        },
        style: {
            load: hb,
            get: ib,
            unload: function(a) {
                Ma("css", a)
            },
            path: function(a) {
                r("rsrc-p-css", a)
            },
            prefetch: jb
        }
    }, oc = this.spf = this.spf || {}, pc;
    for (pc in mc)
        oc[pc] = mc[pc];
    for (var $ in nc)
        for (var qc in nc[$])
            oc[$] = oc[$] || {}, oc[$][qc] = nc[$][qc];
    m("spfready");
})();

