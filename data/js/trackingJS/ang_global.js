// Springboard v3.17
window.springBoard = window.springBoard || {};
window.springBoard.urlParams;
window.springBoard.createCookie = function(e, t, n) {
    if (n) {
        var r = new Date;
        r.setTime(r.getTime() + n * 24 * 60 * 60 * 1e3);
        var i = "; expires=" + r.toGMTString()
    } else 
        var i = "";
    var s = document.location.host.indexOf("www.") == 0 ? document.location.host.substr(3, document.location.host.length): document.location.host;
    document.cookie = escape(e) + "=" + escape(t) + i + ";domain=" + s + ";path=/" + (document.location.protocol == "https:" ? "; secure" : "");
    return null
};
window.springBoard.readCookie = function(e) {
    var t = escape(e) + "=";
    var n = document.cookie.split(";");
    for (var r = 0; r < n.length; r++) {
        var i = n[r];
        while (i.charAt(0) == " ")
            i = i.substring(1, i.length);
        if (i.indexOf(t) == 0)
            return unescape(i.substring(t.length, i.length))
    }
    return null
};
window.springBoard.eraseCookie = function(e) {
    window.springBoard.createCookie(e, "", - 1);
    return null
};
window.springBoard.addCss = function(e) {
    head = document.getElementsByTagName("head")[0], style = document.createElement("style");
    style.type = "text/css";
    if (style.styleSheet) {
        style.styleSheet.cssText = e
    } else {
        style.appendChild(document.createTextNode(e))
    }
    head.appendChild(style)
};
window.springBoard.addJS = function(e) {
    head = document.getElementsByTagName("head")[0], script = document.createElement("script");
    script.type = "text/javascript";
    script.src = e;
    head.appendChild(script)
};
window.onpopstate = function() {
    var e, t = /\+/g, n = /([^&=]+)=?([^&]*)/g, r = function(e) {
        return decodeURIComponent(e.replace(t, " "))
    }, i = window.location.search.substring(1);
    window.springBoard.urlParams = {};
    while (e = n.exec(i))
        window.springBoard.urlParams[r(e[1])] = r(e[2])
};
window.onpopstate();
if (window.springBoard.urlParams["cfQA"]) {
    cfQA = window.springBoard.urlParams["cfQA"].toLowerCase();
    if (cfQA == "true") {
        window.springBoard.createCookie("cfQA", "true")
    }
    if (cfQA == "false") {
        window.springBoard.eraseCookie("cfQA")
    }
}
window.springBoard.postOptimizelyActions = function() {
    if (typeof window["optimizely"] != "undefined" && typeof window["optimizely"].data != "undefined") {
        window.springBoard.postGAActions();
        if (window.springBoard.readCookie("cfQA") == "true") {
            document.title = "[QA] " + document.title
        }
        try {
            var e = window["optimizely"].data.state.activeExperiments;
            window.springBoard.experimentData = "";
            for (var t = 0; t < e.length; t++) {
                window.springBoard.experimentData = window.springBoard.experimentData + window["optimizely"].data.experiments[e[t]].name + " | " + e[t] + " | " + window["optimizely"].data.state.variationNamesMap[e[t]] + "\n";
                var n = {};
                n["cfTest-" + e[t]] = window["optimizely"].data.experiments[e[t]].name;
                n["cfVariation-" + e[t]] = window["optimizely"].data.state.variationNamesMap[e[t]];
                var r = r || [];
                r.push(["set", n])
            }
        } catch (i) {}
    } else {
        setTimeout(function() {
            window.springBoard.postOptimizelyActions()
        }, 500)
    }
};
window.springBoard.postGAActions = function() {
    if (typeof window.optimizely != "undefined" && typeof window.ga != "undefined" && typeof window.optimizely.activeExperiments != "undefined" && typeof window.ga.getAll != "undefined") {
        var e = window.optimizely.activeExperiments;
        var t = window.ga.getAll()[0].get("name") == "t0" ? "": window.ga.getAll()[0].get("name") + ".";
        for (var n = e.length - 1; n >= 0; n--) {
            window.ga(t + "send", {
                hitType: "event",
                eventCategory: "Optimizely",
                eventAction: window.optimizely.data.experiments[e[n]].name,
                eventLabel: window.optimizely.data.experiments[e[n]].name + "-" + window.optimizely.variationNamesMap[e[n]],
                nonInteraction: 1
            })
        }
    } else {
        setTimeout(function() {
            window.springBoard.postGAActions()
        }, 500)
    }
};
window.springBoard.postOptimizelyActions();
window.cFact = window.springBoard;


// IN QA 
if (springBoard.readCookie("cfQA") == "true") {}

// Bugsnag
!function(a, b) {
    function c(a, b) {
        try {
            if ("function" != typeof a)
                return a;
            if (!a.bugsnag) {
                var c = e();
                a.bugsnag = function(d) {
                    if (b && b.eventHandler && (u = d), v = c, !y) {
                        var e = a.apply(this, arguments);
                        return v = null, e
                    }
                    try {
                        return a.apply(this, arguments)
                    } catch (f) {
                        throw l("autoNotify", !0) && (x.notifyException(f, null, null, "error"), s()), f
                    } finally {
                        v = null
                    }
                }, a.bugsnag.bugsnag = a.bugsnag
            }
            return a.bugsnag
        } catch (d) {
            return a
        }
    }
    function d() {
        B=!1
    }
    function e() {
        var a = document.currentScript || v;
        if (!a && B) {
            var b = document.scripts || document.getElementsByTagName("script");
            a = b[b.length - 1]
        }
        return a
    }
    function f(a) {
        var b = e();
        b && (a.script = {
            src: b.src,
            content: l("inlineScript", !0) ? b.innerHTML: ""
        })
    }
    function g(b) {
        var c = a.console;
        void 0 !== c && void 0 !== c.log && c.log("[Bugsnag] " + b)
    }
    function h(a, b, c) {
        if (c >= 5)
            return encodeURIComponent(b) + "=[RECURSIVE]";
        c = c + 1 || 1;
        var d = [];
        for (var e in a)
            if (a.hasOwnProperty(e) && null != e && null != a[e]) {
                var f = b ? b + "[" + e + "]": e, g = a[e];
                d.push("object" == typeof g ? h(g, f, c) : encodeURIComponent(f) + "=" + encodeURIComponent(g))
            }
        return d.join("&")
    }
    function i(a, b) {
        if (null == b)
            return a;
        a = a || {};
        for (var c in b)
            if (b.hasOwnProperty(c))
                try {
                    a[c] = b[c].constructor === Object ? i(a[c], b[c]) : b[c]
        } catch (d) {
            a[c] = b[c]
        }
        return a
    }
    function j(a, b) {
        a += "?" + h(b) + "&ct=img&cb=" + (new Date).getTime();
        var c = new Image;
        c.src = a
    }
    function k(a) {
        for (var b = {}, c = /^data\-([\w\-]+)$/, d = a.attributes, e = 0; e < d.length; e++) {
            var f = d[e];
            if (c.test(f.nodeName)) {
                var g = f.nodeName.match(c)[1];
                b[g] = f.value || f.nodeValue
            }
        }
        return b
    }
    function l(a, b) {
        C = C || k(J);
        var c = void 0 !== x[a] ? x[a] : C[a.toLowerCase()];
        return "false" === c && (c=!1), void 0 !== c ? c : b
    }
    function m(a) {
        return null != a && a.match(D)?!0 : (g("Invalid API key '" + a + "'"), !1)
    }
    function n(b, c) {
        var d = l("apiKey");
        if (m(d) && A) {
            A -= 1;
            var e = l("releaseStage"), f = l("notifyReleaseStages");
            if (f) {
                for (var h=!1, k = 0; k < f.length; k++)
                    if (e === f[k]) {
                        h=!0;
                        break
                    }
                if (!h)
                    return 
            }
            var n = [b.name, b.message, b.stacktrace].join("|");
            if (n !== w) {
                w = n, u && (c = c || {}, c["Last Event"] = q(u));
                var o = {
                    notifierVersion: H,
                    apiKey: d,
                    projectRoot: l("projectRoot") || a.location.protocol + "//" + a.location.host,
                    context: l("context") || a.location.pathname,
                    userId: l("userId"),
                    user: l("user"),
                    metaData: i(i({}, l("metaData")), c),
                    releaseStage: e,
                    appVersion: l("appVersion"),
                    url: a.location.href,
                    userAgent: navigator.userAgent,
                    language: navigator.language || navigator.userLanguage,
                    severity: b.severity,
                    name: b.name,
                    message: b.message,
                    stacktrace: b.stacktrace,
                    file: b.file,
                    lineNumber: b.lineNumber,
                    columnNumber: b.columnNumber,
                    payloadVersion: "2"
                }, p = x.beforeNotify;
                if ("function" == typeof p) {
                    var r = p(o, o.metaData);
                    if (r===!1)
                        return 
                }
                return 0 === o.lineNumber && /Script error\.?/.test(o.message) ? g("Ignoring cross-domain script error. See https://bugsnag.com/docs/notifiers/js/cors") : (j(l("endpoint") || G, o), void 0)
            }
        }
    }
    function o() {
        var a, b, c = 10, d = "[anonymous]";
        try {
            throw new Error("")
        } catch (e) {
            a = "<generated>\n", b = p(e)
        }
        if (!b) {
            a = "<generated-ie>\n";
            var f = [];
            try {
                for (var h = arguments.callee.caller.caller; h && f.length < c;) {
                    var i = E.test(h.toString()) ? RegExp.$1 || d: d;
                    f.push(i), h = h.caller
                }
            } catch (j) {
                g(j)
            }
            b = f.join("\n")
        }
        return a + b
    }
    function p(a) {
        return a.stack || a.backtrace || a.stacktrace
    }
    function q(a) {
        var b = {
            millisecondsAgo: new Date - a.timeStamp,
            type: a.type,
            which: a.which,
            target: r(a.target)
        };
        return b
    }
    function r(a) {
        if (a) {
            var b = a.attributes;
            if (b) {
                for (var c = "<" + a.nodeName.toLowerCase(), d = 0; d < b.length; d++)
                    b[d].value && "null" != b[d].value.toString() && (c += " " + b[d].name + '="' + b[d].value + '"');
                return c + ">"
            }
            return a.nodeName
        }
    }
    function s() {
        z += 1, a.setTimeout(function() {
            z -= 1
        })
    }
    function t(a, b, c) {
        var d = a[b], e = c(d);
        a[b] = e
    }
    var u, v, w, x = {}, y=!0, z = 0, A = 10;
    x.noConflict = function() {
        return a.Bugsnag = b, x
    }, x.refresh = function() {
        A = 10
    }, x.notifyException = function(a, b, c, d) {
        b && "string" != typeof b && (c = b, b = void 0), c || (c = {}), f(c), n({
            name: b || a.name,
            message: a.message || a.description,
            stacktrace: p(a) || o(),
            file: a.fileName || a.sourceURL,
            lineNumber: a.lineNumber || a.line,
            columnNumber: a.columnNumber ? a.columnNumber + 1: void 0,
            severity: d || "warning"
        }, c)
    }, x.notify = function(b, c, d, e) {
        n({
            name: b,
            message: c,
            stacktrace: o(),
            file: a.location.toString(),
            lineNumber: 1,
            severity: e || "warning"
        }, d)
    };
    var B = "complete" !== document.readyState;
    document.addEventListener ? (document.addEventListener("DOMContentLoaded", d, !0), a.addEventListener("load", d, !0)) : a.attachEvent("onload", d);
    var C, D = /^[0-9a-f]{32}$/i, E = /function\s*([\w\-$]+)?\s*\(/i, F = "https://notify.bugsnag.com/", G = F + "js", H = "2.4.5", I = document.getElementsByTagName("script"), J = I[I.length - 1];
    if (a.atob) {
        if (a.ErrorEvent)
            try {
                0 === new a.ErrorEvent("test").colno && (y=!1)
            } catch (K) {}
    } else 
        y=!1;
    if (l("autoNotify", !0)) {
        t(a, "onerror", function(b) {
            return function(c, d, e, g, h) {
                var i = l("autoNotify", !0), j = {};
                !g && a.event && (g = a.event.errorCharacter), f(j), v = null, i&&!z && n({
                    name: h && h.name || "window.onerror",
                    message: c,
                    file: d,
                    lineNumber: e,
                    columnNumber: g,
                    stacktrace: h && p(h) || o(),
                    severity: "error"
                }, j), b && b(c, d, e, g, h)
            }
        });
        var L = function(a) {
            return function(b, d) {
                if ("function" == typeof b) {
                    b = c(b);
                    var e = Array.prototype.slice.call(arguments, 2);
                    return a(function() {
                        b.apply(this, e)
                    }, d)
                }
                return a(b, d)
            }
        };
        t(a, "setTimeout", L), t(a, "setInterval", L), a.requestAnimationFrame && t(a, "requestAnimationFrame", function(a) {
            return function(b) {
                return a(c(b))
            }
        }), a.setImmediate && t(a, "setImmediate", function(a) {
            return function() {
                var b = Array.prototype.slice.call(arguments);
                return b[0] = c(b[0]), a.apply(this, b)
            }
        }), "EventTarget Window Node ApplicationCache AudioTrackList ChannelMergerNode CryptoOperation EventSource FileReader HTMLUnknownElement IDBDatabase IDBRequest IDBTransaction KeyOperation MediaController MessagePort ModalWindow Notification SVGElementInstance Screen TextTrack TextTrackCue TextTrackList WebSocket WebSocketWorker Worker XMLHttpRequest XMLHttpRequestEventTarget XMLHttpRequestUpload".replace(/\w+/g, function(b) {
            var d = a[b] && a[b].prototype;
            d && d.hasOwnProperty && d.hasOwnProperty("addEventListener") && (t(d, "addEventListener", function(a) {
                return function(b, d, e, f) {
                    return d && d.handleEvent && (d.handleEvent = c(d.handleEvent, {
                        eventHandler: !0
                    })), a.call(this, b, c(d, {
                        eventHandler: !0
                    }), e, f)
                }
            }), t(d, "removeEventListener", function(a) {
                return function(b, d, e, f) {
                    return a.call(this, b, d, e, f), a.call(this, b, c(d), e, f)
                }
            }))
        })
    }
    a.Bugsnag = x, "function" == typeof define && define.amd ? define([], function() {
        return x
    }) : "object" == typeof module && "object" == typeof module.exports && (module.exports = x)
}(window, window.Bugsnag);
Bugsnag.apiKey = 'b3795aa0b5add7ba29b34e9e3219535a';
Bugsnag.autoNotify = false;

// Optimizely
cFact.addJS('//cdn.optimizely.com/js/201285641.js');

// Hotjar
(function(f, b, g) {
    var d = g.prototype.open, a = g.prototype.send, c;
    f.hj = f.hj || function() {
        (f.hj.q = f.hj.q || []).push(arguments)
    };
    f._hjSettings = {
        hjid: 237
    };
    if (b.addEventListener) {
        b.addEventListener("DOMContentLoaded", function() {
            f.hj.documentHtml = b.documentElement.outerHTML
        })
    }
    c = b.createElement("script");
    c.async = 1;
    c.src = "//static.hotjar.com/insights.js";
    b.getElementsByTagName("head")[0].appendChild(c);
    f.hj.xo = g.prototype.open;
    f.hj.xs = g.prototype.send;
    if (!f._hjPlayback && b.addEventListener) {
        f.hj.xo = g.prototype.open;
        f.hj.xs = g.prototype.send;
        g.prototype.open = function(l, j, m, h, k) {
            this._u = j;
            f.hj.xo.call(this, l, j, m, h, k)
        };
        g.prototype.send = function(e) {
            var j = this, i = j._u.indexOf("insights.hotjar.com")===-1;
            if (i) {
                function h() {
                    if (j.readyState === 4) {
                        f.hj("_xhr", j._u, j.status, j.response)
                    }
                }
                this.addEventListener("readystatechange", h, false)
            }
            f.hj.xs.call(this, e)
        }
    }
})(window, document, window.XMLHttpRequest);

// Heap
window.heap = window.heap || [], heap.load = function(t, e) {
    window.heap.appid = t, window.heap.config = e;
    var a = document.createElement("script");
    a.type = "text/javascript", a.async=!0, a.src = ("https:" === document.location.protocol ? "https:" : "http:") + "//cdn.heapanalytics.com/js/heap.js";
    var n = document.getElementsByTagName("script")[0];
    n.parentNode.insertBefore(a, n);
    for (var o = function(t) {
        return function() {
            heap.push([t].concat(Array.prototype.slice.call(arguments, 0)))
        }
    }, p = ["identify", "track"], c = 0; c < p.length; c++)
        heap[p[c]] = o(p[c])
};
heap.load("243575103");

// MAIN FUNCTION
springBoard.main = function() {}
