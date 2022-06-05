(function(f, h) {
    var c;
    c = function(a) {
        return new e(a)
    };
    c.isEmptyObject = function(a) {
        return Object.keys(a).length?!1 : !0
    };
    c.isFunction = function(a) {
        return "function" === typeof a
    };
    c.isWindow = function(a) {
        return a === window
    };
    c.isDocument = function(a) {
        return a === window.document
    };
    c.noop = function() {};
    c.each = function(a, b) {
        var d, g;
        if ("object" === typeof a&&!Array.isArray(a))
            if ((g = a[Object.keys(a)[0]]) && void 0 !== g.nodeName)
                for (d in a)
                    a.hasOwnProperty(d) && "length" !== d && b(d, a[d], a);
            else 
                for (d in a)
                    a.hasOwnProperty(d) && b(d,
                    a[d], a);
        else 
            a.forEach(function(a, d, g) {
                b(d, a, g)
            })
    };
    c.trim = function(a) {
        return a.trim()
    };
    c.inArray = function(a, b) {
        return b.indexOf(a)
    };
    c.stringify = function(a, b, d) {
        var g;
        g = Array.prototype.toJSON;
        delete Array.prototype.toJSON;
        a = JSON.stringify(a, b, d);
        g && (Array.prototype.toJSON = g);
        return a
    };
    c.ajax = function(a) {
        var b = new XMLHttpRequest;
        a.type = a.type || "GET";
        b.open(a.type, a.url, !0);
        "POST" === a.type && b.setRequestHeader("Content-Type", (a.contentType ? a.contentType : "application/x-www-form-urlencoded") + "; charset=UTF-8");
        b.onload = function() {
            200 <= b.status && 400 > b.status ? c.isFunction(a.success) && a.success(b.responseText && JSON.parse(b.responseText), b) : c.isFunction(a.error) && a.error(b)
        };
        b.onerror = function() {
            c.isFunction(a.error) && a.error(b)
        };
        c.isFunction(a.requestAnnotator) && a.requestAnnotator(b);
        "POST" === a.type && a.data ? b.send(a.data) : b.send()
    };
    c.get = function(a, b) {
        var d = new XMLHttpRequest;
        d.open("GET", a, !0);
        d.onload = function() {
            200 <= d.status && 400 > d.status && b && b(d.responseText)
        };
        d.send()
    };
    c.eventHandlers = {};
    c.selector = "";
    var e = function(a) {
        var b;
        c.selector = a;
        if (c.isWindow(a))
            this[0] = window, this.length = 1;
        else if (c.isDocument(a))
            this[0] = h, this.length = 1;
        else if ("object" === typeof a)
            this[0] = a, this.length = 1;
        else if ("string" === typeof a && "<" === a.charAt(0) && ">" === a.charAt(a.length - 1) && 3 <= a.length)
            b = h.createElement("div"), b.innerHTML = a, this[0] = b.childNodes[0], this.length = 1;
        else if (a) {
            a = h.querySelectorAll(a);
            for (b = 0; b < a.length; b++)
                this[b] = a[b];
            this.length = a.length
        }
        return this
    };
    e.prototype.val = function(a) {
        "undefined" !== typeof a &&
        0 < this.length && (this[0].value = a);
        if (void 0 !== this[0])
            return this[0] ? this[0].value : ""
    };
    e.prototype.each = function(a, b) {
        Array.prototype.forEach.call(this, function(a, g, c) {
            b(g, a, c)
        })
    };
    e.prototype.append = function(a) {
        var b;
        "object" === typeof a ? "body" === c.selector ? h.body.appendChild(a.get(0)) : this[0].appendChild(a.get(0)) : "body" === c.selector ? (b = h.createElement("div"), b.innerHTML = a, h.body.appendChild(b)) : (b = h.createElement("div"), b.innerHTML = a, this[0].appendChild(b))
    };
    e.prototype.hasClass = function(a) {
        return this[0].classList ?
        this[0].classList.contains(a) : RegExp("(^| )" + a + "( |$)", "gi").test(this[0].className)
    };
    e.prototype.addClass = function(a) {
        var b;
        for (b = 0; b < this.length; b++)
            this[b].classList ? this[b].classList.add(a) : this[b].className += " " + a;
        return this
    };
    e.prototype.removeClass = function(a) {
        var b;
        for (b = 0; b < this.length; b++)
            this[b].classList ? this[b].classList.remove(a) : this[b].className = this[b].className.replace(RegExp("(^|\\b)" + a.split(" ").join("|") + "(\\b|$)", "gi"), " ");
        return this
    };
    e.prototype.is = function(a) {
        var b;
        a: {
            b =
            this[0];
            var d = b.matchesSelector || b.msMatchesSelector || b.mozMatchesSelector || b.webkitMatchesSelector || b.oMatchesSelector;
            if (d)
                b = d.call(b, a);
            else {
                a = b.parentNode.querySelectorAll(a);
                for (d = a.length; d--;)
                    if (a[d] === b) {
                        b=!0;
                        break a
                    }
                b=!1
            }
        }
        return b
    };
    e.prototype.remove = function() {
        var a;
        for (a = 0; a < this.length; a++)
            this[a].parentNode.removeChild(this[a])
    };
    e.prototype.click = function(a) {
        for (i = 0; i < this.length; i += 1)
            event = h.createEvent("HTMLEvents"), event.initEvent("click", !0, !1), this[i].dispatchEvent(event), a && a()
    };
    e.prototype.trigger = function(a) {
        var b, d = a.split(" "), g;
        for (a = 0; a < this.length; a += 1)
            for (b = 0; b < d.length; b += 1)
                g = h.createEvent("HTMLEvents"), g.initEvent(d[b], !0, !1), this[a].dispatchEvent(g)
    };
    e.prototype.on = function(a, b, d) {
        var g, l = a.split(" "), m, e, f, n, p;
        if (c.isDocument(this[0]) && "string" === typeof b)
            for (a = 0; a < l.length; a += 1)
                "string" === typeof b ? ("boolean" === typeof d&&!1 === d && (d = function(a) {
                    a.preventDefault();
                    return !1
                }), m = b + "." + l[a], "array" !== typeof c.eventHandlers[m] && (c.eventHandlers[m] = []), c.eventHandlers[m].push(d),
                h.addEventListener(l[a].split(".")[0], function(a) {
                    if (e = h.querySelectorAll(b)) {
                        f = a.target;
                        for (n =- 1; f&&-1 === (n = Array.prototype.indexOf.call(e, f));)
                            f = f.parentElement;
                            - 1 < n && d.call(f, a)
                    }
                })) : ("boolean" === typeof b&&!1 === b && (b = function(a) {
                    a.preventDefault();
                    return !1
                }), "array" !== typeof c.eventHandlers.document && (c.eventHandlers.document = []), c.eventHandlers.document.push(b), this[0].addEventListener(l[a].split(".")[0], b, !1));
        else if (c.isDocument(this[0]))
            for (a = 0; a < l.length; a += 1)
                "boolean" === typeof b&&!1 === b &&
                (b = function(a) {
                    a.preventDefault();
                    return !1
                }), m = "document." + l[a], "array" !== typeof c.eventHandlers[m] && (c.eventHandlers[m] = []), c.eventHandlers[m].push(b), h.addEventListener(l[a].split(".")[0], b, !1);
        else if (c.isWindow(this[0]))
            for (a = 0; a < l.length; a += 1)
                "boolean" === typeof b&&!1 === b && (b = function(a) {
                    a.preventDefault();
                    return !1
                }), m = "window." + l[a], "array" !== typeof c.eventHandlers[m] && (c.eventHandlers[m] = []), c.eventHandlers[m].push(b), window.addEventListener(l[a].split(".")[0], b, !1);
        else 
            for (g = 0; g < this.length; g +=
            1)
                for (a = 0; a < l.length; a += 1)
                    "object" === typeof b ? (p = b, b = function(a) {
                        a.data = p;
                        d.call(this[g], a)
                    }) : "boolean" === typeof b&&!1 === b && (b = function(a) {
                        a.preventDefault();
                        return !1
                    }), m = c.selector + "." + l[a], "array" !== typeof c.eventHandlers[m] && (c.eventHandlers[m] = []), c.eventHandlers[m].push(b), this[g].addEventListener(l[a].split(".")[0], b, !1);
        return this
    };
    e.prototype.off = function(a, b, d) {
        var g, l, m = a.split(" "), e, f;
        for (a = 0; a < this.length; a += 1)
            for (g = 0; g < m.length; g += 1)
                if (c.isDocument(this[a]) && "string" === typeof b) {
                    if (l =
                    h.querySelectorAll(b)) {
                        e = event.target;
                        for (f =- 1; e&&-1 === (f = Array.prototype.indexOf.call(l, e));)
                            e = e.parentElement;
                            if ( - 1 < f)
                                if ("undefined" === typeof b)
                                    for (l = 0; l < c.eventHandlers[d + "." + m[g]].length; l++)
                                        e.removeEventListener(m[g].split(".")[0], c.eventHandlers[d + "." + m[g]][l], !1);
                                else 
                                    e.removeEventListener(m[g].split(".")[0], d, !1)
                                }
                } else if (c.isDocument(this[a]))
                    if ("undefined" === typeof b)
                        for (l = 0; l < c.eventHandlers["document." + m[g]].length; l++)
                            h.removeEventListener(m[g].split(".")[0], c.eventHandlers["document." +
                            m[g]][l], !1);
                    else 
                        h.removeEventListener(m[g].split(".")[0], b, !1);
        else if (c.isWindow(this[a]))
            if ("undefined" === typeof b)
                for (l = 0; l < c.eventHandlers["window." + m[g]].length; l++)
                    window.removeEventListener(m[g].split(".")[0], c.eventHandlers["window." + m[g]][l], !1);
            else 
                window.removeEventListener(m[g].split(".")[0], b, !1);
        else if ("undefined" === typeof b)
            for (l = 0; l < c.eventHandlers[c.selector + "." + m[g]].length; l++)
                this[a].removeEventListener(m[g].split(".")[0], c.eventHandlers[c.selector + "." + m[g]][l], !1);
        else 
            this[a].removeEventListener(m[g].split(".")[0],
            b, !1);
        return this
    };
    e.prototype.scrollTop = function() {
        return window.document.body.scrollTop || window.document.documentElement.scrollTop
    };
    e.prototype.height = function() {
        var a;
        return c.isWindow(this[0]) ? h.documentElement.clientHeight : 9 === this[0].nodeType ? (a = this[0].documentElement, Math.max(this[0].body.scrollHeight, a.scrollHeight, this[0].body.offsetHeight, a.offsetHeight, a.clientHeight)) : Math.max(this[0].scrollHeight, this[0].offsetHeight)
    };
    e.prototype.width = function() {
        var a;
        return c.isWindow(this[0]) ? h.documentElement.clientWidth :
        9 === this[0].nodeType ? (a = this[0].documentElement, Math.max(this[0].body.scrollWidth, a.scrollWidth, this[0].body.offsetWidth, a.offsetWidth, a.clientWidth)) : Math.max(this[0].scrollWidth, this[0].offsetWidth)
    };
    e.prototype.outerHeight = function() {
        return this[0].offsetHeight
    };
    e.prototype.offset = function() {
        var a = (this[0] && this[0].ownerDocument).documentElement;
        return {
            top: this[0].getBoundingClientRect().top + window.pageYOffset - a.clientTop,
            left: this[0].getBoundingClientRect().left + window.pageXOffset - a.clientLeft
        }
    };
    e.prototype.attr = function(a, b) {
        var d;
        if (b) {
            for (d = 0; d < this.length; d += 1)
                this[d].setAttribute(a, b);
            return this
        }
        if (null !== this[0].getAttribute(a))
            return this[0].getAttribute(a)
    };
    e.prototype.ready = function(a) {
        c.isDocument(this[0]) && ("interactive" === h.readyState || "complete" === h.readyState || "loaded" === h.readyState ? a() : h.addEventListener("DOMContentLoaded", a, !1))
    };
    e.prototype.parent = function() {
        return c(this[0].parentNode)
    };
    e.prototype.get = function(a) {
        return this[a]
    };
    e.prototype.show = function() {
        var a;
        for (a =
        0; a < this.length; a += 1)
            this[a].style.display = "";
        return this
    };
    e.prototype.hide = function() {
        var a;
        for (a = 0; a < this.length; a += 1)
            this[a].style.display = "none";
        return this
    };
    e.prototype.focus = function() {
        var a;
        for (a = 0; a < this.length; a += 1)
            this[a].focus();
        return this
    };
    e.prototype.blur = function() {
        var a;
        for (a = 0; a < this.length; a += 1)
            this[a].blur();
        return this
    };
    e.prototype.clone = function() {
        return this[0].cloneNode(!0)
    };
    e.prototype.removeAttr = function(a) {
        var b;
        for (b = 0; b < this.length; b += 1)
            this[b].removeAttribute(a);
        return this
    };
    e.prototype.find = function(a) {
        a = this[0].querySelectorAll(a);
        var b = c(), d;
        for (d = 0; d < a.length; d++)
            b[d] = a[d];
        b.length = a.length;
        return b
    };
    e.prototype.is = function(a) {
        var b, d=!1;
        if (!a)
            return !1;
        if ("object" === typeof a)
            return c(this[0]).get(0) === a.get(0);
        if ("string" === typeof a) {
            if (":visible" === a)
                return !(0 >= this[0].offsetWidth && 0 >= this[0].offsetHeight);
            if (":hidden" === a)
                return 0 >= this[0].offsetWidth && 0 >= this[0].offsetHeight;
            if ( - 1 < a.indexOf("[")) {
                if (b = /([A-Za-z]+)\[([A-Za-z-]+)\=([A-Za-z]+)\]/g.exec(a), b.length)
                    return c.each(c(this[0]).get(0).attributes,
                    function(a, c) {
                        c.name === b[2] && c.value === b[3] && (d=!0)
                    }), c(this[0]).get(0).nodeName.toLowerCase() === b[1] && d
            } else 
                return c(this[0]).get(0).nodeName.toLowerCase() === a
        }
    };
    e.prototype.css = function(a, b) {
        var d, g;
        for (g = 0; g < this.length; g += 1)
            if ("object" === typeof a)
                for (d in a)
                    this[g].style[d] = a[d];
            else if ("number" === typeof b || "string" === typeof b)
                this[g].style[a] = b;
            else 
                return getComputedStyle(this[g])[a];
        return this
    };
    e.prototype.animate = function(a, b) {
        var d, g = this;
        "undefined" === typeof b && (b = 400);
        for (d = 0; d < g.length; d +=
        1)
            c.each(a, function(a, c) {
                function e(a, d) {
                    a.style[d[0].attribute] = d[0].value;
                    d.shift();
                    d.length ? z = setTimeout(function() {
                        e(a, d)
                    }, 10) : clearTimeout(z)
                }
                c = c.toString();
                var f = parseFloat(getComputedStyle(g[d])[a]) || 0, h = getComputedStyle(g[d])[a].replace(/[0-9.-]/g, ""), p = parseFloat(c), r = c.replace(/[0-9.-]/g, ""), h = h || r, u = p - f, r = parseFloat(b / 10), u = u / r, s = [], x, z;
                for (x = 0; x < r; x += 1)
                    f += u, s.push({
                        attribute: a,
                        value: h ? parseInt(f) + h: parseFloat(f).toFixed(1)
                    });
                    s.pop();
                    s.push({
                        attribute: a,
                        value: p + h
                    });
                    s.length && e(g[d], s)
                });
        return this
    };
    e.prototype.filter = function(a) {
        return Array.prototype.filter.call(h.querySelectorAll(c.selector), function(b, d) {
            a(d, b)
        })
    };
    f.hj = f.hj || {};
    f.hj.hq = f.hj.hq || c
})(this, document);
(function() {
    (function(f) {
        window.onerror = function(h, c, e, a, b) {
            return c && ( - 1 !== c.indexOf("insights.js") || /hotjar-\d+\.js/.test(c)) ? (b ? console.error("--- HOTJAR ERROR ---\nMESSAGE\n" + h + "\n\nERROR TRACEBACK\n" + b.stack) : console.error("--- HOTJAR ERROR ---\nMESSAGE\n" + h), !0) : f ? f.apply(this, arguments) : !1
        }
    })(window.onerror);
    window.hj = window.hj || function() {
        (window.hj.q = window.hj.q || []).push(arguments)
    };
    window.hj.q = window.hj.q || [];
    window._hjSettings = window._hjSettings || {};
    hj.host = _hjSettings.host || "insights.hotjar.com";
    hj.hostname = hj.host.split(":")[0];
    hj.secure = "https:" == location.protocol;
    hj.port = hj.host.split(":")[1] || (hj.secure ? "443" : "80");
    hj.apiUrlBase = location.protocol + "//" + hj.host + "/api/v1";
    hj.dcServer = null;
    hj.includedInSample=!1;
    hj.isPlayback = Boolean(window._hjPlayback);
    hj.isPreview = Boolean(_hjSettings.preview);
    hj.noDataPush = hj.isPreview || hj.isPlayback;
    hj.pageId = null;
    hj.pageVisitId = null;
    hj.settings = null;
    hj.startTime = (new Date).getTime();
    hj.userId = null;
    hj.cookie = function() {
        return {
            get: function(f) {
                return (f =
                RegExp("(?:^|; )" + f + "=([^;]*)").exec(document.cookie)) ? f[1] : null
            },
            set: function(f, h, c) {
                var e = new Date;
                f = f + "=" + h + "; path=/; ";
                c || (e.setTime(e.getTime() + 31536E6), f += "expires=" + e.toUTCString());
                document.cookie = f
            }
        }
    }();
    hj.dataPusher = function() {
        function f(d, c, m) {
            m ? g[d] = c : (g[d] = g[d] || [], g[d].push(c));
            hj.dataPusher._pushDataDirty=!0;
            !a&&!b && (b=!0, hj.event.listen(["gotDCServer"], function() {
                s = new WebSocket((hj.secure ? "wss://" : "ws://") + hj.dcServer + "/api/v1/client/userdata");
                s.onopen = function() {
                    a=!0;
                    b=!1;
                    h()
                }
            }))
        }
        function h() {
            var b = hj.hq.isEmptyObject(g), c = hj.isPreview || hj.noDataPush;
            !b && (!c&&!q && d && a) && (p && (r = (new Date).getTime()), (new Date).getTime() - r > t ? q=!0 : a && (clearInterval(m), m = setInterval(h, l), g.page_visit_id = hj.pageVisitId, n ? (g = hj.userId + "\n" + hj.hq.stringify(g), n=!1) : g = hj.hq.stringify(g), hj.log.debug("Pushing data: " + g), s.send(g), g = {}))
        }
        function c() {
            a && (h(), s.close())
        }
        var e = {}, a=!1, b=!1, d=!1, g = {}, l = 5E3, m = void 0, t = 12E4, q=!1, n=!0, p=!0, r = void 0, u = "hidden", s = null;
        e.init = function() {
            hj.hq(document).ready(function() {
                function a(d) {
                    var b =
                    {
                        focus: "visible",
                        focusin: "visible",
                        pageshow: "visible",
                        blur: "hidden",
                        focusout: "hidden",
                        pagehide: "hidden"
                    };
                    d = d || window.event;
                    p = d.type in b ? "visible" === b[d.type] : !this[u]
                }
                "hidden"in document ? document.addEventListener("visibilitychange", a) : (u = "mozHidden")in document ? document.addEventListener("mozvisibilitychange", a) : (u = "webkitHidden")in document ? document.addEventListener("webkitvisibilitychange", a) : (u = "msHidden")in document ? document.addEventListener("msvisibilitychange", a) : "onfocusin"in document ? document.onfocusin =
                document.onfocusout = a : window.onpageshow = window.onpagehide = window.onfocus = window.onblur = a;
                void 0 !== document[u] && a({
                    type: document[u] ? "blur": "focus"
                })
            });
            window.addEventListener("unload", c, !1)
        };
        e.pushImmediately = function(a, b, g, c) {
            !0 === c && (d=!0);
            null !== a && f(a, b, g);
            h()
        };
        e.pushLater = function(a, d, b) {
            f(a, d, b);
            "undefined" === typeof m && (m = setInterval(h, l))
        };
        return e
    }();
    hj.event = function() {
        function f() {
            var a, b, d, g;
            for (b in e)
                if (e.hasOwnProperty(b))
                    for (a = c.length - 1; 0 <= a; a -= 1)
                        g = c[a], d = g.eventIds.indexOf(b), - 1 !==
                        d && (g.eventIds.splice(d, 1), 0 === g.eventIds.length && (g.callback(), c.splice(a, 1)))
                }
        var h = {}, c = [], e = {};
        h.listen = function(a, b) {
            c.unshift({
                eventIds: a,
                callback: b
            });
            f()
        };
        h.signal = function(a) {
            e[a]=!0;
            f()
        };
        return h
    }();
    hj.log = function() {
        return {
            init: function() {
                "undefined" === typeof window.console && (window.console = {
                    debug: function() {},
                    trace: function() {},
                    log: function() {},
                    info: function() {},
                    warn: function() {},
                    error: function() {}
                })
            },
            debug: function(f) {
                _hjSettings.hjdebug && console.log(f)
            },
            info: function(f) {
                _hjSettings.hjdebug &&
                console.log("%c" + f, "color: #006EFF")
            },
            warn: function(f) {
                console.log("%c" + f, "color: #E8910C")
            },
            error: function(f) {
                console.error("Hotjar error: " + f)
            }
        }
    }();
    hj.loader = function() {
        var f = {}, h = [];
        f.getModules = function() {
            return h
        };
        f.registerModule = function(c, e) {
            h.push({
                name: c,
                module: e
            })
        };
        f.loadScripts = function(c, e) {
            function a(a) {
                b += 1;
                hj.log.debug("Script loaded: " + b + " (" + a + ")");
                b === c.length && e()
            }
            var b = 0, d = {}, g;
            0 === c.length && e();
            for (g = 0; g < c.length; g++)
                d[g] = document.createElement("script"), d[g].src = c[g], d[g].onload =
                function(b) {
                    return function() {
                        a(d[b].src)
                    }
                }(g), d[g].onreadystatechange = function(b) {
                    return function() {
                        if ("complete" === this.readyState || "loaded" === this.readyState)
                            hj.log.debug(this.readyState + ": " + d[b].src + " (IE onreadystatechange)"), d[b].onreadystatechange = null, a(d[b].src)
                        }
                    }(g), document.getElementsByTagName("head")[0].appendChild(d[g])
        };
        f.loadSettings = function(c) {
            hj.isPreview || hj.isPlayback ? c() : "undefined" !== typeof window.hjSiteSettings ? c(window.hjSiteSettings) : hj.request.get(hj.apiUrlBase + "/client/sites/" +
            _hjSettings.hjid, c)
        };
        return f
    }();
    hj.targeting = function() {
        return {
            checkIfWidgetInCookie: function(f, h) {
                var c = hj.cookie.get(f), e, c = c ? c.split(","): [];
                for (e = 0; e < c.length; e++)
                    if (h.toString() === c[e])
                        return !0;
                return !1
            },
            ruleMatches: function(f) {
                var h = window.location.pathname, c = window.location.href, e, a, b;
                for (b = 0; b < f.length; b += 1) {
                    e = f[b];
                    switch (e.match_operation) {
                    case "exact":
                        a = h === e.pattern;
                        break;
                    case "starts_with":
                        a = 0 === h.indexOf(e.pattern);
                        break;
                    case "ends_with":
                        a = h.lastIndexOf(e.pattern) + e.length === h.length;
                        break;
                    case "contains":
                        a =- 1 !== h.indexOf(e.pattern);
                        break;
                    case "regex":
                        a = RegExp(e.pattern).test(c)
                    }
                    if (a)
                        return hj.log.debug("PATH MATCH: " + e.component + "|" + e.match_operation + "|" + e.pattern), !0
                }
                return !1
            }
        }
    }();
    hj.request = function() {
        return {
            get: function(f, h, c) {
                h = h || hj.hq.noop;
                if ("XDomainRequest"in window && null !== window.XDomainRequest && 10 > hj.utils.ieVersion()) {
                    var e = new XDomainRequest;
                    e._hjDontCapture=!1 === c;
                    e.open("get", f);
                    e.send();
                    e.onload = function() {
                        h(e.responseText && JSON.parse(e.responseText))
                    }
                } else 
                    hj.hq.ajax({
                        url: f,
                        success: h,
                        requestAnnotator: function(a) {
                            a._hjDontCapture=!1 === c
                        }
                    })
            },
            post: function(f, h, c, e) {
                c = c || hj.hq.noop;
                if ("XDomainRequest"in window && null !== window.XDomainRequest && 10 > hj.utils.ieVersion()) {
                    var a = new XDomainRequest;
                    a._hjDontCapture=!1 === e;
                    a.open("post", f);
                    a.send(hj.hq.stringify(h));
                    a.onload = function() {
                        c(a.responseText && JSON.parse(a.responseText))
                    }
                } else 
                    c = c || hj.hq.noop, hj.hq.ajax({
                        url: f,
                        type: "POST",
                        data: hj.hq.stringify(h),
                        contentType: "application/json",
                        success: c,
                        requestAnnotator: function(a) {
                            a._hjDontCapture =
                            !1 === e
                        }
                    })
            },
            sendPageView: function(f, h) {
                var c = hj.ui.getWindowSize();
                hj.noDataPush || hj.request.post(hj.apiUrlBase + "/client/sites/" + _hjSettings.hjid + "/visit-data", {
                    user_id: hj.userId,
                    path: f,
                    window_width: c.width,
                    window_height: c.height,
                    included_in_sample: hj.includedInSample
                }, function(c, a) {
                    !0 === h && (hj.pageId = c.page_id, hj.pageVisitId = c.page_visit_id);
                    var b, d;
                    a && (b = sessionStorage.getItem("_hjDcServer"), d = a.getResponseHeader("X-DC-Server"), hj.dcServer = b || d, !b && d && sessionStorage.setItem("_hjDcServer", hj.dcServer),
                    hj.event.signal("gotDCServer"));
                    hj.event.signal("gotPageVisitId")
                })
            },
            sendXhrResponse: function(f) {
                "undefined" !== typeof hj.dataPusher && (f = {
                    url: f._url,
                    response: f.response,
                    status: f.status
                }, hj.log.debug("Sending XHR response: " + hj.hq.stringify(f)), hj.dataPusher.pushImmediately("xhr", {
                    time: (new Date).getTime() - hj.startTime,
                    xhr: f
                }, !1))
            }
        }
    }();
    hj.utils = function() {
        self.ieVersion = function(f) {
            f = f || navigator.userAgent;
            var h;
            for (h = 5; 11 > h; h++)
                if (0 < f.indexOf("MSIE " + h))
                    return h;
            return - 1 !== f.indexOf("Trident/") ? 11 :
            "notIE"
        };
        self.uuid4 = function() {
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(f) {
                var h = 16 * Math.random() | 0;
                return ("x" == f ? h : h & 3 | 8).toString(16)
            })
        };
        return self
    }();
    hj.widget = function() {
        function f(a, d) {
            d(a)
        }
        function h(a, d) {
            setTimeout(function() {
                d(a)
            }, 1E3 * a.display_delay)
        }
        function c(a, d, b) {
            function g() {
                d(a);
                c.off("mousemove." + b);
                c.off("mouseout." + b)
            }
            var c = hj.hq(document), e = hj.hq(window), f = [];
            c.on("mousemove." + b, function(a) {
                f.push({
                    x: a.clientX,
                    y: a.clientY
                });
                2 < f.length && f.shift()
            });
            c.on("mouseout." + b, function(a) {
                if (!a.relatedTarget || a.relatedTarget !== this&&!(this.compareDocumentPosition(a.relatedTarget) & Node.DOCUMENT_POSITION_CONTAINED_BY)) {
                    var d = f[1];
                    a = f[0];
                    "undefined" !== typeof d&&!(d.y >= a.y) && (a.x === d.x && g(), d = (a.y - d.y) / (a.x - d.x), a = a.y - d * a.x, a =- a / d, 0 < a && a < e.width() && g())
                }
            })
        }
        function e(a, d, b) {
            var g = hj.hq(document), c = hj.hq(window);
            c.on("scroll." + b, function() {
                var e = g.height();
                0.5 <= (g.scrollTop() + hj.ui.getWindowSize().height) / e && (c.off("scroll." + b), d(a))
            })
        }
        function a(d) {
            for (var g in d) {
                var c;
                d.hasOwnProperty(g) && (c = d[g], "object" === typeof c ? c instanceof hj.widget.TrustedString ? d[g] = c.value : a(c) : "string" === typeof c && (d[g] = b(c)))
            }
        }
        function b(a) {
            var d = {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
                "/": "&#x2F;"
            };
            return String(a).replace(/[&<>"'\/]/g, function(a) {
                return d[a]
            })
        }
        var d = {}, g = "collapsed", l = {};
        d.ctrl = void 0;
        d.model = {};
        d.modelChangeCallbacks = [];
        d.applyMobileClasses = function() {
            hj.isPreview || hj.widget.isNarrowScreen() ? hj.widget.ctrl.addClass("_hj-f5b2a1eb-9b07_widget_centered") :
            hj.widget.ctrl.removeClass("_hj-f5b2a1eb-9b07_widget_centered");
            hj.widget.isShortScreen() ? hj.widget.ctrl.addClass("_hj-f5b2a1eb-9b07_widget_short") : hj.widget.ctrl.removeClass("_hj-f5b2a1eb-9b07_widget_short")
        };
        d.callAccordingToCondition = function(a, d, b) {
            var g = {
                immediate: f,
                delay: h,
                abandon: c,
                scroll: e
            }
            [a.display_condition];
            hj.log.debug("Calling widget: " + a.display_condition);
            if (g)
                g(a, b, d);
            else 
                throw Error('Unhandled display condition: "' + a.display_condition + '"');
        };
        d.changeState = function(a, d) {
            d = d || a.data.state;
            if ("open" === d || "collapsed" === d)
                g = d;
            hj.widget.ctrl.removeClass("_hj-f5b2a1eb-9b07_widget_open").removeClass("_hj-f5b2a1eb-9b07_widget_collapsed").removeClass("_hj-f5b2a1eb-9b07_widget_thankyou").removeClass("_hj-f5b2a1eb-9b07_widget_hidden").addClass("_hj-f5b2a1eb-9b07_widget_" + d)
        };
        d.openWidget = function() {
            var a = hj.isPreview ? 0: 300;
            hj.widget.ctrl.removeClass("_hj-f5b2a1eb-9b07_widget_collapsed").addClass("_hj-f5b2a1eb-9b07_widget_open").animate({
                bottom: "0"
            }, a);
            g = "open"
        };
        d.collapseWidget = function() {
            var a =
            hj.isPreview ? 0: 450;
            hj.widget.ctrl.removeClass("_hj-f5b2a1eb-9b07_widget_open").addClass("_hj-f5b2a1eb-9b07_widget_collapsed").animate({
                bottom: "0"
            }, a);
            g = "collapsed"
        };
        d.toggleWidget = function() {
            hj.widget.ctrl.hasClass("_hj-f5b2a1eb-9b07_widget_hidden") ? hj.widget.changeState(null, g) : (hj.widget.changeState(null, "hidden"), hj.widget.setMinimized())
        };
        d.changeColorLuminance = function(a, d) {
            a = (new String(a)).replace(/[^0-9a-f]/gi, "");
            6 > a.length && (a = a[0] + a[0] + a[1] + a[1] + a[2] + a[2]);
            d = d || 0;
            var b = "#", g, c;
            for (c = 0; 3 >
            c; c++)
                g = parseInt(a.substr(2 * c, 2), 16), g = Math.round(Math.min(Math.max(0, g + 255 * d), 255)).toString(16), b += ("00" + g).substr(g.length);
            return b
        };
        d.dataBind = function() {
            hj.hq.each(hj.widget.ctrl.find("[data-bind]"), function(a, b) {
                var g = hj.hq(b);
                g.on("keyup change", function() {
                    var a = g.attr("name").split("_hj-f5b2a1eb-9b07_")[1];
                    d.model[a] = g.val();
                    hj.hq.each(d.modelChangeCallbacks, function(a, d) {
                        d()
                    })
                })
            })
        };
        d.disableSubmit = function() {
            hj.widget.ctrl.find("#_hj-f5b2a1eb-9b07_action_submit").addClass("_hj-f5b2a1eb-9b07_btn_disabled")
        };
        d.dismissWidget = function() {
            hj.widget.setDone();
            hj.widget.ctrl.hide()
        };
        d.init = function() {
            hj.widget.ctrl.find("._hj-f5b2a1eb-9b07_action_toggle_widget").on("click", hj.widget.toggleWidget);
            hj.widget.ctrl.find("._hj-f5b2a1eb-9b07_action_open_widget").on("click", {
                state: "open"
            }, hj.widget.changeState);
            hj.widget.ctrl.find("._hj-f5b2a1eb-9b07_action_dismiss_widget").on("click", hj.widget.dismissWidget)
        };
        d.enableSubmit = function() {
            hj.widget.ctrl.find("#_hj-f5b2a1eb-9b07_action_submit").removeClass("_hj-f5b2a1eb-9b07_btn_disabled")
        };
        d.isNarrowScreen = function() {
            return 768 > hj.hq(window).width()
        };
        d.isShortScreen = function() {
            return 400 > hj.hq(window).height()
        };
        d.selectAnswer = function(a) {
            var b = hj.hq(a), g = b.attr("data-key"), b = b.attr("data-value");
            d.model[g] = b;
            hj.hq.each(d.modelChangeCallbacks, function(a, d) {
                d()
            });
            hj.widget.ctrl.find("._hj-f5b2a1eb-9b07_button_radio").removeClass("_hj-f5b2a1eb-9b07_button_radio_active");
            hj.hq(a).addClass("_hj-f5b2a1eb-9b07_button_radio_active")
        };
        d.sendPageViewIfNotDone = function(a) {
            var d = hj.ui.getWindowSize();
            "undefined" !== typeof hj.pageVisitId ? a() : hj.request.post(hj.apiUrlBase + "/client/sites/" + _hjSettings.hjid + "/visit-data", {
                user_id: hj.userId,
                path: location.pathname,
                window_width: d.width,
                window_height: d.height,
                included_in_sample: hj.includedInSample
            }, function(d) {
                hj.pageVisitId = d.page_visit_id;
                hj.pageId = d.page_id;
                a()
            })
        };
        d.renderTemplate = function(d, b) {
            var g = l[d];
            g || (g = "var p=[],print=function(){p.push.apply(p,arguments);};with(obj){p.push('" + d.replace(/[\r\t\n]/g, " ").replace(/'(?=[^%]*%>)/g, "\t").split("'").join("\\'").split("\t").join("'").replace(/<%=(.+?)%>/g,
            "',$1,'").split("<%").join("');").split("%>").join("p.push('") + "');}return p.join('');", g = new Function("obj", g), l[d] = g);
            a(b);
            return g(b)
        };
        d.TrustedString = function(a) {
            this.value = a
        };
        d.commonCSS = '            <style type="text/css">                /*reset css*/                ._hj-f5b2a1eb-9b07_widget, ._hj-f5b2a1eb-9b07_widget * {line-height: normal; font-family: Arial, sans-serif, Tahoma !important;}                ._hj-f5b2a1eb-9b07_widget div, ._hj-f5b2a1eb-9b07_widget span, ._hj-f5b2a1eb-9b07_widget p, ._hj-f5b2a1eb-9b07_widget a, ._hj-f5b2a1eb-9b07_widget img, ._hj-f5b2a1eb-9b07_widget strong, ._hj-f5b2a1eb-9b07_widget form, ._hj-f5b2a1eb-9b07_widget label {border:0; outline:0; font-size:100%; vertical-align:baseline; background:transparent; margin:0; padding:0;}                ._hj-f5b2a1eb-9b07_widget ol {list-style:decimal;}                ._hj-f5b2a1eb-9b07_widget ul {list-style:disc;}                ._hj-f5b2a1eb-9b07_widget ul ul {list-style:circle;}                ._hj-f5b2a1eb-9b07_widget hr {display:block; height:1px; border:0; border-top:1px solid #ccc; margin:1em 0; padding:0;}                ._hj-f5b2a1eb-9b07_widget input[type=submit], ._hj-f5b2a1eb-9b07_widget input[type=button], ._hj-f5b2a1eb-9b07_widget button {margin:0; padding:0;}                ._hj-f5b2a1eb-9b07_widget input, ._hj-f5b2a1eb-9b07_widget select, ._hj-f5b2a1eb-9b07_widget a img {vertical-align:middle;}                /*generic css*/                ._hj-f5b2a1eb-9b07_widget {font-size:13px; position: fixed; z-index: 2147483647; bottom: -400px; right: 100px; width: 300px; -webkit-border-radius: 5px 5px 0 0; -moz-border-radius: 5px 5px 0 0; border-radius: 5px 5px 0 0;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_position_left {right: auto; left: 100px;}                 ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_rounded_corners {-webkit-border-radius: 4px; -moz-border-radius: 4px; border-radius: 4px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_shadow {-webkit-box-shadow: 0 2px 3px 0 rgba(0, 0, 0, 0.15); -moz-box-shadow: 0 2px 3px 0 rgba(0, 0, 0, 0.15); box-shadow: 0 2px 3px 0 rgba(0, 0, 0, 0.15);}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_transition {-webkit-transition: all 0.2s ease-in-out; -moz-transition: all 0.2s ease-in-out; -o-transition: all 0.2s ease-in-out; -ms-transition: all 0.2s ease-in-out; transition: all 0.2s ease-in-out;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_pull_left {float: left !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_pull_right {float: right !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_clear_both {clear: both !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_hidden {display: none !important;}                                /*common css*/                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_centered {right:50%; margin-right: -150px; left: auto;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_state {display: none;}                                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_icon {background-repeat: no-repeat; width: 16px; height: 16px; display: -moz-inline-stack; display: inline-block; zoom: 1; *display: inline; vertical-align: top;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_open_close {text-align: center; position: absolute; top: -18px; right: 20px; width: 40px; height: 18px; padding-top: 2px; cursor: pointer; -webkit-border-radius: 5px 5px 0 0; -moz-border-radius: 5px 5px 0 0; border-radius: 5px 5px 0 0;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_open_close ._hj-f5b2a1eb-9b07_widget_icon {background-position: -32px 0;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_hidden_handle {display:none; height: 4px; cursor:pointer; -webkit-border-radius: 5px 5px 0 0; -moz-border-radius: 5px 5px 0 0; border-radius: 5px 5px 0 0;}                                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_title {font-weight: bold; text-align: center; padding: 12px; margin: 0; line-height: 17px; min-height: 17px; word-break: break-word; word-wrap: break-word; cursor: pointer;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_initiator {display: none; padding: 0 12px 12px 12px; text-align: center;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_initiator button {padding-left: 20px; padding-right: 20px;}                                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_primary {cursor: pointer; text-decoration: none; font-weight: bold; padding: 7px 10px; border: 0; outline: 0; display: -moz-inline-stack; display: inline-block; zoom: 1; *display: inline; vertical-align: top;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_primary {background-color: #00C764; color: white}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_primary:hover, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_primary:focus, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_primary:active {background-color: #00a251;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled:hover, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled:focus, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled:active {cursor: default; -webkit-box-shadow: none; -moz-box-shadow: none; box-shadow: none;}                                /*content*/                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content {padding:0 12px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content_overflow {max-height: 280px; overflow-y: auto; overflow-x: hidden;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_short ._hj-f5b2a1eb-9b07_widget_content {padding:0 11px 0 12px; max-height: 120px; overflow-y: auto; overflow-x: hidden;}                                /*open ended*/                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_input_field {font-family: Arial, sans-serif, Tahoma; font-size: 14px; color: #333; padding: 6px; height: 30px; width:100%; margin: 0 0 12px 0; background: white; border: 1px solid <%= widgetStyle.footerBorderColor %> !important; -webkit-box-sizing: border-box; -moz-box-sizing: border-box; box-sizing: border-box;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_input_field:focus {border: 1px solid #00a251;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content textarea._hj-f5b2a1eb-9b07_input_field {resize: none; height: 100px;}                                /*close ended*/                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_button_radio {position: relative; min-height: 45px; text-align:left !important; height:auto !important; height: 45px; -webkit-box-sizing: border-box; -moz-box-sizing: border-box; box-sizing: border-box;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_button_radio span._hj-f5b2a1eb-9b07_widget_icon {-webkit-border-radius: 30px; -moz-border-radius: 30px; border-radius: 30px; border: 2px solid #AAA; width: 22px; height: 22px; display: block; position: absolute; left: 12px; top: 50%; margin-top: -12px; background-position: -64px -100px; -webkit-box-sizing: content-box; -moz-box-sizing: content-box; box-sizing: content-box; *}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_button_radio span._hj-f5b2a1eb-9b07_radio_text {text-align:left !important; padding: 14px 20px 14px 50px; position: relative; display: block;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_button_radio_full {margin-left:-12px; margin-right: -12px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_button_radio._hj-f5b2a1eb-9b07_button_radio_active, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_button_radio._hj-f5b2a1eb-9b07_button_radio_active:hover {background-color: <%= widgetStyle.alternateColor %> !important; color: white !important; cursor: default;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_button_radio._hj-f5b2a1eb-9b07_button_radio_active span {border-color: white; background-position: -97px 4px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_double_control {margin: 0 0 12px 0;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_double_control ._hj-f5b2a1eb-9b07_double_first {width:49%; float:left; margin-bottom: 0}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_double_control ._hj-f5b2a1eb-9b07_double_second {width:49%; float:left; margin-bottom: 0; margin-left:2%;}                                /* footer*/                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left {padding: 21px 0 0 12px; font-size: 11px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left a, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left a:hover, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left a:focus, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left a:active {text-decoration: underline;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left span {background-position: -16px 0; margin-right: 4px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_right {padding: 12px 12px 12px 0;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_right button {font-size: 13px; padding-right: 5px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_right button span {background-position: -64px 0; margin-left: 8px;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_right button._hj-f5b2a1eb-9b07_btn_disabled span {background-position: -48px 0;}                                /*state: hidden*/                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_hidden ._hj-f5b2a1eb-9b07_widget_open_close ._hj-f5b2a1eb-9b07_widget_icon {background-position: 0 0;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_hidden ._hj-f5b2a1eb-9b07_widget_title {display: none;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_hidden ._hj-f5b2a1eb-9b07_widget_hidden_handle {display: block;}                                /*state: collapsed */                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_collapsed ._hj-f5b2a1eb-9b07_widget_initiator {display: block;}                                /*state: open*/                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_open ._hj-f5b2a1eb-9b07_widget_state_open {display: block;}                                /*state: thankyou*/                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_thankyou ._hj-f5b2a1eb-9b07_widget_state_thankyou {display: block;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_thankyou ._hj-f5b2a1eb-9b07_widget_open_close {display: none;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_thankyou ._hj-f5b2a1eb-9b07_widget_title {display: none;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_widget_thankyou ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_right ._hj-f5b2a1eb-9b07_btn span {background-position: -80px 0;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_thankyou_message {text-align: center; padding: 20px; margin: 0;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_thankyou_message button {margin-top: 12px; padding: 7px 20px;}                                /* theme css */                ._hj-f5b2a1eb-9b07_widget {background: <%= widgetStyle.primaryColor %> !important; color: <%= widgetStyle.fontColor %> !important;}                ._hj-f5b2a1eb-9b07_widget p {color: <%= widgetStyle.fontColor %> !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_icon {background-image: url(//<%= hjHost %>/static/client/_hj-f5b2a1eb-9b07_widget_icons_light.png) !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_open_close {background: <%= widgetStyle.primaryColor %> !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_hidden_handle {background: <%= widgetStyle.primaryColor %> !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn {background-color: <%= widgetStyle.secondaryColor %> !important; color: <%= widgetStyle.fontColor %> !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn:hover, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn:focus,  ._hj-f5b2a1eb-9b07_btn:active {background-color: #666 !important;}                                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_button_radio {border-bottom: 1px solid <%= widgetStyle.primaryColor %> !important; border-top: 1px solid <%= widgetStyle.alternateColor %> !important; background: <%= widgetStyle.secondaryColor %> !important; cursor: pointer !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_button_radio_last {border-bottom:0 !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_button_radio:hover {background: <%= widgetStyle.alternateColor %> !important; color: <%= widgetStyle.fontColorNegative %> }                                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer {border-top: 1px solid <%= widgetStyle.footerBorderColor %> !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left a, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_footer ._hj-f5b2a1eb-9b07_pull_left a:hover {color: <%= widgetStyle.footerTextColor %> !important;}                ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled:hover, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled:focus, ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_btn_disabled:active {color: <%= widgetStyle.primaryColor %> !important; background-color: <%= widgetStyle.secondaryColor %> !important;}                                /*light theme css*/                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_skin_light {-webkit-box-shadow: 0 0 7px 0 rgba(0, 0, 0, 0.3) !important; -moz-box-shadow: 0 0 7px 0 rgba(0, 0, 0, 0.3) !important; box-shadow: 0 0 7px 0 rgba(0, 0, 0, 0.3) !important;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_skin_light ._hj-f5b2a1eb-9b07_widget_icon {background-image: url(//<%= hjHost %>/static/client/_hj-f5b2a1eb-9b07_widget_icons_light.png) !important;}                                                /*dark theme css*/                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_skin_dark {-webkit-box-shadow: 0 0 7px 0 rgba(0, 0, 0, 0.6) !important; -moz-box-shadow: 0 0 7px 0 rgba(0, 0, 0, 0.6) !important; box-shadow: 0 0 7px 0 rgba(0, 0, 0, 0.6) !important;}                ._hj-f5b2a1eb-9b07_widget._hj-f5b2a1eb-9b07_skin_dark ._hj-f5b2a1eb-9b07_widget_icon {background-image: url(//<%= hjHost %>/static/client/_hj-f5b2a1eb-9b07_widget_icons_dark.png) !important;}                            </style>';
        return d
    }();
    hj.ui = function() {
        return {
            getWindowSize: function() {
                return {
                    width: window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth,
                    height: window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight
                }
            }
        }
    }()
})();
(function() {
    function f() {
        return {
            run: function() {
                hj.includedInSample && hj.request.sendPageView(location.pathname, !0)
            }
        }
    }
    !0 !== hj.isPlayback && hj.loader.registerModule("VisitData", f())
})();
(function() {
    function f() {
        function f() {
            b.push = function() {
                var a;
                for (a = 0; a < arguments.length; a += 1)
                    this[this.length] = arguments[a];
                c();
                return this.length
            }
        }
        function c() {
            var d = Array.prototype.slice.call(b.shift()), e = a[d[0]], f = d.slice(1);
            hj.log.debug("Processing command: " + d.join(", "));
            hj.hq.isFunction(e) ? e(f) : hj.log.debug('Unknown command: "' + d[0] + '"');
            0 < b.length && c()
        }
        var e = {}, a = {}, b = window.hj.q, d=!1;
        a.vpv = function(a) {
            a = a[0];
            hj.includedInSample && (hj.log.debug('Sending virtual page view for URL "' + a + '"'),
            hj.request.sendPageView(a, !1))
        };
        a._xhr = function(a) {
            d || (hj.settings.record && hj.request.sendXhrResponse({
                _url: a[0],
                status: a[1],
                response: a[2]
            }), d=!0)
        };
        e.run = function() {
            f();
            0 < b.length && c()
        };
        return e
    }
    !0 !== hj.isPlayback && hj.loader.registerModule("Command", f())
})();
(function() {
    function f() {
        function f(a) {
            var d, b, g, v;
            b = a.attr("id");
            var e = a.attr("name");
            v = function(a, b) {
                g = hj.hq(b);
                for (d = 0; d < g.length; d++)
                    if (g[d] === a[0])
                        return d
            };
            if (b)
                return H.test(b) ? '0:[id="' + b + '"]' : "0:#" + b;
            if (e)
                return b = '*[name="' + e + '"]', v(a, b) + ":" + b;
            b = c(a);
            return v(a, b) + ":" + b
        }
        function c(a, d) {
            var b, g, v, e;
            "undefined" === typeof d && (d = "");
            e = d;
            if (a.is("html"))
                return "html" + d;
            b = a.get(0).nodeName.toLowerCase();
            g = a.attr("id");
            v = a.attr("class");
            if (g)
                return H.test(g) ? b + '[id="' + g + '"]' + d : b + "#" + g + d;
            v && (b += "." +
            hj.hq.trim(v).split(/[\s\n]+/).join("."));
            g = c;
            v = a.parent();
            var f;
            try {
                hj.hq(b + d), f=!0
            } catch (l) {
                f=!1
            }
            return g(v, f ? ">" + b + d : e)
        }
        function e() {
            return parseInt(1E3 * (hj.hq(window).scrollTop() + hj.ui.getWindowSize().height) / hj.hq(document).height(), 10)
        }
        function a() {
            var a = document.doctype, d = (null === document.doctype ? "" : "<!DOCTYPE " + a.name + (a.publicId ? ' PUBLIC "' + a.publicId + '"' : "") + (!a.publicId && a.systemId ? " SYSTEM" : "") + (a.systemId ? ' "' + a.systemId + '"' : "") + ">\n") + (window.hj.documentHtml || document.documentElement.outerHTML),
            b = q(t(d)), g = hj.apiUrlBase + "/sites/" + _hjSettings.hjid + "/pages/" + hj.pageId;
            hj.request.get(g + "/content-id/" + b, function(a) {
                a.exists ? (hj.pageContentId = a.page_content_id, hj.event.signal("gotPageContentId")) : hj.request.post(g + "/content", {
                    content: d,
                    content_md5: b
                }, function(a) {
                    hj.pageContentId = a.page_content_id;
                    hj.event.signal("gotPageContentId")
                })
            })
        }
        function b(a, b) {
            var c = a[0], k = a[1], e = a[2], f = a[3], c = g(c, k, e, f, b[0], 7, - 680876936), f = g(f, c, k, e, b[1], 12, - 389564586), e = g(e, f, c, k, b[2], 17, 606105819), k = g(k, e, f, c, b[3],
            22, - 1044525330), c = g(c, k, e, f, b[4], 7, - 176418897), f = g(f, c, k, e, b[5], 12, 1200080426), e = g(e, f, c, k, b[6], 17, - 1473231341), k = g(k, e, f, c, b[7], 22, - 45705983), c = g(c, k, e, f, b[8], 7, 1770035416), f = g(f, c, k, e, b[9], 12, - 1958414417), e = g(e, f, c, k, b[10], 17, - 42063), k = g(k, e, f, c, b[11], 22, - 1990404162), c = g(c, k, e, f, b[12], 7, 1804603682), f = g(f, c, k, e, b[13], 12, - 40341101), e = g(e, f, c, k, b[14], 17, - 1502002290), k = g(k, e, f, c, b[15], 22, 1236535329), c = l(c, k, e, f, b[1], 5, - 165796510), f = l(f, c, k, e, b[6], 9, - 1069501632), e = l(e, f, c, k, b[11], 14, 643717713), k = l(k,
            e, f, c, b[0], 20, - 373897302), c = l(c, k, e, f, b[5], 5, - 701558691), f = l(f, c, k, e, b[10], 9, 38016083), e = l(e, f, c, k, b[15], 14, - 660478335), k = l(k, e, f, c, b[4], 20, - 405537848), c = l(c, k, e, f, b[9], 5, 568446438), f = l(f, c, k, e, b[14], 9, - 1019803690), e = l(e, f, c, k, b[3], 14, - 187363961), k = l(k, e, f, c, b[8], 20, 1163531501), c = l(c, k, e, f, b[13], 5, - 1444681467), f = l(f, c, k, e, b[2], 9, - 51403784), e = l(e, f, c, k, b[7], 14, 1735328473), k = l(k, e, f, c, b[12], 20, - 1926607734), c = d(k^e^f, c, k, b[5], 4, - 378558), f = d(c^k^e, f, c, b[8], 11, - 2022574463), e = d(f^c^k, e, f, b[11], 16, 1839030562),
            k = d(e^f^c, k, e, b[14], 23, - 35309556), c = d(k^e^f, c, k, b[1], 4, - 1530992060), f = d(c^k^e, f, c, b[4], 11, 1272893353), e = d(f^c^k, e, f, b[7], 16, - 155497632), k = d(e^f^c, k, e, b[10], 23, - 1094730640), c = d(k^e^f, c, k, b[13], 4, 681279174), f = d(c^k^e, f, c, b[0], 11, - 358537222), e = d(f^c^k, e, f, b[3], 16, - 722521979), k = d(e^f^c, k, e, b[6], 23, 76029189), c = d(k^e^f, c, k, b[9], 4, - 640364487), f = d(c^k^e, f, c, b[12], 11, - 421815835), e = d(f^c^k, e, f, b[15], 16, 530742520), k = d(e^f^c, k, e, b[2], 23, - 995338651), c = m(c, k, e, f, b[0], 6, - 198630844), f = m(f, c, k, e, b[7], 10, 1126891415),
            e = m(e, f, c, k, b[14], 15, - 1416354905), k = m(k, e, f, c, b[5], 21, - 57434055), c = m(c, k, e, f, b[12], 6, 1700485571), f = m(f, c, k, e, b[3], 10, - 1894986606), e = m(e, f, c, k, b[10], 15, - 1051523), k = m(k, e, f, c, b[1], 21, - 2054922799), c = m(c, k, e, f, b[8], 6, 1873313359), f = m(f, c, k, e, b[15], 10, - 30611744), e = m(e, f, c, k, b[6], 15, - 1560198380), k = m(k, e, f, c, b[13], 21, 1309151649), c = m(c, k, e, f, b[4], 6, - 145523070), f = m(f, c, k, e, b[11], 10, - 1120210379), e = m(e, f, c, k, b[2], 15, 718787259), k = m(k, e, f, c, b[9], 21, - 343485551);
            a[0] = n(c, a[0]);
            a[1] = n(k, a[1]);
            a[2] = n(e, a[2]);
            a[3] =
            n(f, a[3])
        }
        function d(a, b, d, c, e, g) {
            b = n(n(b, a), n(c, g));
            return n(b<<e | b>>>32 - e, d)
        }
        function g(a, b, c, e, g, f, l) {
            return d(b & c|~b & e, a, b, g, f, l)
        }
        function l(a, b, c, e, g, f, l) {
            return d(b & e | c&~e, a, b, g, f, l)
        }
        function m(a, b, c, e, g, f, l) {
            return d(c^(b|~e), a, b, g, f, l)
        }
        function t(a) {
            var d = a.length, c = [1732584193, - 271733879, - 1732584194, 271733878], e;
            for (e = 64; e <= a.length; e += 64) {
                for (var g = c, f = a.substring(e - 64, e), l = [], h = void 0, h = 0; 64 > h; h += 4)
                    l[h>>2] = f.charCodeAt(h) + (f.charCodeAt(h + 1)<<8) + (f.charCodeAt(h + 2)<<16) + (f.charCodeAt(h + 3)<<
                    24);
                b(g, l)
            }
            a = a.substring(e - 64);
            g = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            for (e = 0; e < a.length; e++)
                g[e>>2]|=a.charCodeAt(e)<<(e%4<<3);
            g[e>>2]|=128<<(e%4<<3);
            if (55 < e) {
                b(c, g);
                for (e = 0; 16 > e; e++)
                    g[e] = 0
            }
            g[14] = 8 * d;
            b(c, g);
            return c
        }
        function q(a) {
            for (var b = 0; b < a.length; b++) {
                for (var d = a, c = b, e = a[b], g = "", f = 0; 4 > f; f++)
                    g += I[e>>8 * f + 4 & 15] + I[e>>8 * f & 15];
                d[c] = g
            }
            return a.join("")
        }
        function n(a, b) {
            return a + b & 4294967295
        }
        function p() {
            (function(a) {
                var b = a.prototype.open, d = a.prototype.send;
                a.prototype.open = function(a, d, c, e, g) {
                    this._url =
                    d;
                    b.call(this, a, d, c, e, g)
                };
                a.prototype.send = function(a) {
                    function b() {
                        4 === c.readyState && hj.request.sendXhrResponse(c)
                    }
                    var c = this;
                    !0 !== c._hjDontCapture && (this.addEventListener ? this.addEventListener("readystatechange", b, !1) : this.attachEvent("onreadystatechange", b));
                    d.call(this, a)
                }
            })(XMLHttpRequest)
        }
        function r() {
            var a = e();
            a > B && (B = a, hj.dataPusher.pushLater("scroll_reach", {
                max_bottom: B
            }, !0))
        }
        function u() {
            C && (hj.dataPusher.pushLater("scroll", {
                time: (new Date).getTime() - hj.startTime,
                y: parseInt(hj.hq(window).scrollTop(),
                10)
            }), C=!1)
        }
        function s() {
            parseInt(hj.hq(window).scrollTop(), 10);
            C=!0
        }
        function x(a) {
            var b = f(hj.hq(a.target)), d = [], c;
            "target"in a && ("pageX"in a && "pageY"in a && void 0 !== b) && (c = hj.hq(a.target).offset(), d.left = a.pageX - c.left, d.top = a.pageY - c.top, a = {
                offset_x: d.left,
                offset_y: d.top,
                selector: b
            }, hj.log.debug("[CLICK] x|y offset:" + a.offset_x + "|" + a.offset_y + " selector:" + a.selector), hj.settings.record && (a.time = (new Date).getTime() - hj.startTime), hj.dataPusher.pushImmediately("mouse_click", a, !0))
        }
        function z() {
            D &&
            (hj.settings.record && hj.dataPusher.pushLater("mouse_move", {
                time: (new Date).getTime() - hj.startTime,
                x: E,
                y: F
            }), y && hj.dataPusher.pushLater("relative_mouse_move", {
                offset_x: J,
                offset_y: K,
                selector: L
            }), D=!1)
        }
        function N(a) {
            var b, d;
            E = a.clientX;
            F = a.clientY;
            b = hj.hq(document.elementFromPoint(E, F));
            if (d = b.offset())
                J = a.pageX - parseInt(d.left, 10), K = a.pageY - parseInt(d.top, 10), L = f(b), D=!0
        }
        function O(a) {
            a = hj.hq(a.target);
            var b = a.val();
            w|=a.is("input[type=password]");
            w|=!isNaN(parseInt(b, 10)) && 12 < b.length;
            w|="undefined" !==
            typeof a.attr("data-hj-masked");
            A.push({
                time: (new Date).getTime() - hj.startTime,
                selector: f(a),
                text: b
            });
            G=!0
        }
        function P() {
            G && (w && hj.hq.each(A, function(a, b) {
                b.text = Array(b.text.length + 1).join("*")
            }), hj.dataPusher.pushImmediately("key_press", A, !0), w = G=!1, A = [])
        }
        var M = {}, G=!1, w=!1, A = [], C=!1, B = 0, E = 0, F, D=!1, J = 0, K = 0, L = null, y, H = /(^\d+|\.|\[|\])/, I = "0123456789abcdef".split("");
        "5d41402abc4b2a76b9719d911017c592" != q(t("hello")) && (n = function(a, b) {
            var d = (a & 65535) + (b & 65535);
            return (a>>16) + (b>>16) + (d>>16)<<16 | d & 65535
        });
        M.run = function() {
            var b = {};
            !hj.isPreview && (hj.includedInSample&&!(10 > hj.utils.ieVersion())) && (hj.hq.each(hj.settings.heatmaps || [], function(a, b) {
                if (hj.targeting.ruleMatches([b.regex ? {
                    match_operation: "regex",
                    pattern: b.regex
                }
                : {
                    match_operation: "exact",
                    pattern: b.path
                }
                ]))
                    return y = b.id, !1
            }), hj.settings.record && p(), hj.hq(document).ready(function() {
                var d = ["gotPageVisitId"];
                b.record = hj.settings.record;
                if (y || hj.settings.record)
                    hj.hq(document).on("mousemove", N), hj.hq(document).on("mousedown", x), setInterval(z,
                    100);
                y && (b.heatmap_id = y, hj.hq(window).on("scroll resize", r), hj.hq(document).ready(function() {
                    b.max_bottom = e();
                    hj.event.signal("gotMaxBottom")
                }), d.push("gotMaxBottom"));
                hj.settings.record && (hj.hq(window).on("scroll", s), hj.hq("input").on("keyup", O), hj.hq("input").on("blur", P), setInterval(u, 100), hj.event.listen(["gotPageVisitId"], a), hj.event.listen(["gotPageContentId"], function() {
                    b.page_content_id = hj.pageContentId
                }), d.push("gotPageContentId"));
                (y || hj.settings.record) && hj.event.listen(d, function() {
                    hj.dataPusher.pushImmediately("helo",
                    b, !0, !0);
                    hj.dataPusher.pushImmediately(null)
                })
            }))
        };
        return M
    }
    !0 !== hj.isPlayback && hj.loader.registerModule("BehaviorData", f())
})();
(function() {
    function f() {
        function f(a) {
            hj.widget.ctrl = hj.hq("#_hj-f5b2a1eb-9b07_poll");
            hj.widget.submitResponse = function() {
                var b;
                hj.hq("#_hj-f5b2a1eb-9b07_action_submit").hasClass("_hj-f5b2a1eb-9b07_btn_disabled") || ("single-close-ended" === a.type ? b = hj.widget.model.response : (b = hj.widget.ctrl.find("input[name=_hj-f5b2a1eb-9b07_answer]").val(), void 0 === b && (b = hj.widget.ctrl.find("textarea[name=_hj-f5b2a1eb-9b07_answer]").val() || "")), b && (!hj.isPreview&&!hj.noDataPush && hj.widget.sendPageViewIfNotDone(function() {
                    hj.request.post(hj.apiUrlBase +
                    "/client/sites/" + _hjSettings.hjid + "/polls/" + a.id, {
                        action: "poll_response",
                        response_text: b,
                        response_path: location.pathname,
                        page_visit_id: hj.pageVisitId
                    });
                    hj.widget.setDone()
                }), hj.widget.changeState(!1, "thankyou")))
            };
            hj.widget.setDone = function() {
                if (!hj.isPreview) {
                    var b = hj.cookie.get("_hjDonePolls"), b = b ? b.split(","): [];
                    - 1 === hj.hq.inArray(a.id.toString(), b) && (b.push(a.id), hj.cookie.set("_hjDonePolls", b.join(",")))
                }
            };
            hj.widget.setMinimized = function() {
                if (!hj.isPreview) {
                    var b = hj.cookie.get("_hjMinimizedPolls"),
                    b = b ? b.split(","): [];
                    - 1 === hj.hq.inArray(a.id.toString(), b) && (b.push(a.id), hj.cookie.set("_hjMinimizedPolls", b.join(",")))
                }
            };
            hj.widget.ctrl.find("._hj-f5b2a1eb-9b07_button_radio").on("click", function() {
                hj.widget.selectAnswer(this);
                hj.widget.enableSubmit()
            });
            hj.widget.ctrl.find("._hj-f5b2a1eb-9b07_input_field").on("keyup", function() {
                0 < hj.hq(this).val().length ? hj.widget.enableSubmit() : hj.widget.disableSubmit()
            });
            hj.widget.ctrl.find("#_hj-f5b2a1eb-9b07_action_submit").on("click", function() {
                hj.widget.submitResponse()
            });
            hj.hq(window).on("resize", function() {
                hj.widget.applyMobileClasses()
            });
            hj.widget.applyMobileClasses();
            hj.isPreview ||!hj.widget.isNarrowScreen() ? hj.widget.openWidget() : hj.widget.collapseWidget();
            !hj.isPreview && hj.targeting.checkIfWidgetInCookie("_hjMinimizedPolls", a.id) && hj.widget.changeState(null, "hidden")
        }
        function c(a) {
            hj.log.debug("-- RENDERING POLL WIDGET --");
            var c = hj.widget.changeColorLuminance(a.background, - 0.1), e = hj.widget.changeColorLuminance(a.background, 0.1), m = hj.widget.changeColorLuminance(a.background,
            - 0.2), t = hj.widget.changeColorLuminance(a.background, 0.2), q = hj.widget.changeColorLuminance(a.background, - 0.6), n = hj.widget.changeColorLuminance(a.background, 0.6), p = {
                id: a.id,
                type: a.type,
                question: a.content.question,
                thankyou: a.content.thankyou,
                answers: a.content.answers
            };
            a = hj.widget.renderTemplate(b, {
                apiUrlBase: new hj.widget.TrustedString(hj.apiUrlBase),
                hjHost: new hj.widget.TrustedString(hj.host),
                hjid: _hjSettings.hjid,
                preview: hj.isPreview ||!1,
                noDataPush: hj.noDataPush,
                poll: p,
                widgetStyle: {
                    position: a.position,
                    skin: a.skin,
                    primaryColor: a.background,
                    secondaryColor: "light" === a.skin ? c: e,
                    alternateColor: "light" === a.skin ? m: t,
                    footerTextColor: "light" === a.skin ? q: n,
                    footerBorderColor: "light" === a.skin ? m: c,
                    fontColor: "light" === a.skin ? "#111": "#FFF",
                    fontColorNegative: "light" === a.skin ? "#FFF": "#111"
                }
            });
            hj.hq("#_hj_poll_container").remove();
            hj.hq("body").append(a);
            f(p);
            hj.widget.init();
            hj.log.debug("Done")
        }
        function e(a) {
            var b = void 0;
            if (a)
                return a;
            hj.hq.each(hj.settings.polls || [], function(a, d) {
                if (!hj.targeting.checkIfWidgetInCookie("_hjDonePolls",
                d.id) && hj.targeting.ruleMatches(d.targeting))
                    return b = d, !1
            });
            return b
        }
        var a = {}, b = ['<div id="_hj_poll_container">', hj.widget.commonCSS, '<div id="_hj-f5b2a1eb-9b07_poll" class="_hj-f5b2a1eb-9b07_widget _hj-f5b2a1eb-9b07_skin_<%= widgetStyle.skin %> _hj-f5b2a1eb-9b07_position_<%= widgetStyle.position %>">                        <a class="_hj-f5b2a1eb-9b07_widget_open_close _hj-f5b2a1eb-9b07_action_toggle_widget"><span class="_hj-f5b2a1eb-9b07_widget_icon"></span></a>                        <div class="_hj-f5b2a1eb-9b07_widget_hidden_handle _hj-f5b2a1eb-9b07_action_toggle_widget"></div>                        <p class="_hj-f5b2a1eb-9b07_widget_title _hj-f5b2a1eb-9b07_action_open_widget"><%= poll.question %></p>                        <div class="_hj-f5b2a1eb-9b07_widget_initiator">                            <button type="button" class="_hj-f5b2a1eb-9b07_btn_primary _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow _hj-f5b2a1eb-9b07_action_open_widget">Reply</button>                        </div>                        <div class="_hj-f5b2a1eb-9b07_widget_state _hj-f5b2a1eb-9b07_widget_state_open">                            <% if (poll.type === "single-close-ended") { %>                                <div class="_hj-f5b2a1eb-9b07_widget_content<% if (poll.answers.length >= 5) { %> _hj-f5b2a1eb-9b07_widget_content_overflow<% } %>">                                    <% for (var i = 0; i < poll.answers.length; i++) { %>                                    <div class="_hj-f5b2a1eb-9b07_button_radio _hj-f5b2a1eb-9b07_button_radio_full <%if (i+1 === poll.answers.length) { %>_hj-f5b2a1eb-9b07_button_radio_last<% } %>" data-key="response" data-value="<%= poll.answers[i].text %>">                                        <% if (poll.answers.length > 1) { %>                                            <span class="_hj-f5b2a1eb-9b07_widget_icon"></span>                                            <span class="_hj-f5b2a1eb-9b07_radio_text"><%= poll.answers[i].text %></span>                                        <% } %>                                    </div>                                    <% } %>                                </div>                            <% } else if (poll.type === "single-open-ended-single-line") { %>                                <div class="_hj-f5b2a1eb-9b07_widget_content">                                    <input class="_hj-f5b2a1eb-9b07_input_field _hj-f5b2a1eb-9b07_rounded_corners" type="text" name="_hj-f5b2a1eb-9b07_answer" placeholder="Please type here..." />                                </div>                            <% } else if (poll.type === "single-open-ended-multiple-line") { %>                                <div class="_hj-f5b2a1eb-9b07_widget_content">                                    <textarea class="_hj-f5b2a1eb-9b07_input_field _hj-f5b2a1eb-9b07_rounded_corners" name="_hj-f5b2a1eb-9b07_answer" rows="3" placeholder="Please type here..."></textarea>                                </div>                            <% } %>                            <div class="_hj-f5b2a1eb-9b07_widget_footer">                                <div class="_hj-f5b2a1eb-9b07_pull_left">                                    <span class="_hj-f5b2a1eb-9b07_widget_icon"></span>powered by <a href="http://www.hotjar.com/?utm_source=client&utm_medium=poll&utm_campaign=insights" target="_blank">Hotjar</a>                                </div>                                <div class="_hj-f5b2a1eb-9b07_pull_right">                                    <button type="button" id="_hj-f5b2a1eb-9b07_action_submit" class="_hj-f5b2a1eb-9b07_btn_primary _hj-f5b2a1eb-9b07_btn_disabled _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow">Send <span class="_hj-f5b2a1eb-9b07_widget_icon"></span></button>                                </div>                                <div class="_hj-f5b2a1eb-9b07_clear_both"></div>                            </div>                        </div>                        <div class="_hj-f5b2a1eb-9b07_widget_state _hj-f5b2a1eb-9b07_widget_state_thankyou">                            <p class="_hj-f5b2a1eb-9b07_thankyou_message">                                <%= poll.thankyou %><br />                                <button type="button" class="_hj-f5b2a1eb-9b07_btn_primary _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow _hj-f5b2a1eb-9b07_action_dismiss_widget">Close</button>                            </p>                            <div class="_hj-f5b2a1eb-9b07_widget_footer">                                <div class="_hj-f5b2a1eb-9b07_pull_left">                                    <span class="_hj-f5b2a1eb-9b07_widget_icon"></span>powered by <a href="http://www.hotjar.com/?utm_source=client&utm_medium=poll&utm_campaign=insights" target="_blank">Hotjar</a>                                </div>                                <div class="_hj-f5b2a1eb-9b07_pull_right">                                    <button type="button" class="_hj-f5b2a1eb-9b07_btn _hj-f5b2a1eb-9b07_btn_disabled _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow">Sent <span class="_hj-f5b2a1eb-9b07_widget_icon"></span></button>                                </div>                                <div class="_hj-f5b2a1eb-9b07_clear_both"></div>                            </div>                        </div>                    </div></div>'].join("");
        a.getWidgetImportance = function(a) {
            return (a = e(a)) ? a.created_epoch_time : 0
        };
        a.run = function(a, b) {
            var f = e(b);
            f && a && (f.display_condition = hj.isPreview ? "immediate" : f.display_condition, hj.widget.callAccordingToCondition(f, "poll", c))
        };
        hj.isPreview && (window._hjPollReload = function(b) {
            a.run(!0, b)
        });
        return a
    }
    !0 !== hj.isPlayback && hj.loader.registerModule("Polls", f())
})();
(function() {
    function f() {
        function f(a) {
            function b() {
                return Boolean(hj.widget.model.name && hj.widget.model.age && hj.widget.model.sex && hj.widget.model.city && hj.widget.model.email && hj.widget.model.phone)
            }
            hj.widget.ctrl = hj.hq("#_hj-f5b2a1eb-9b07_testers");
            hj.widget.modelChangeCallbacks.push(function() {
                b() ? hj.widget.enableSubmit() : hj.widget.disableSubmit()
            });
            hj.widget.submitResponse = function() {
                b() && (!hj.isPreview&&!hj.noDataPush && hj.widget.sendPageViewIfNotDone(function() {
                    hj.request.post(hj.apiUrlBase + "/client/sites/" +
                    _hjSettings.hjid + "/testers/" + a.id, {
                        action: "testers_widget_response",
                        response: hj.widget.model,
                        response_path: location.pathname,
                        page_visit_id: hj.pageVisitId
                    });
                    hj.widget.setDone()
                }), hj.widget.changeState(!1, "thankyou"))
            };
            hj.widget.setDone = function() {
                if (!hj.isPreview) {
                    var b = hj.cookie.get("_hjDoneTestersWidgets"), b = b ? b.split(","): [];
                    - 1 === hj.hq.inArray(a.id.toString(), b) && (b.push(a.id), hj.cookie.set("_hjDoneTestersWidgets", b.join(",")))
                }
            };
            hj.widget.setMinimized = function() {
                if (!hj.isPreview) {
                    var b = hj.cookie.get("_hjMinimizedTestersWidgets"),
                    b = b ? b.split(","): [];
                    - 1 === hj.hq.inArray(a.id.toString(), b) && (b.push(a.id), hj.cookie.set("_hjMinimizedTestersWidgets", b.join(",")))
                }
            };
            hj.widget.ctrl.find("._hj-f5b2a1eb-9b07_button_radio").on("click", function() {
                hj.widget.selectAnswer(this)
            });
            hj.widget.ctrl.find("#_hj-f5b2a1eb-9b07_action_submit").on("click", function() {
                hj.widget.submitResponse()
            });
            hj.hq(window).on("resize", function() {
                hj.widget.applyMobileClasses()
            });
            hj.widget.applyMobileClasses();
            hj.widget.collapseWidget();
            hj.widget.dataBind();
            !hj.isPreview &&
            hj.targeting.checkIfWidgetInCookie("_hjMinimizedTestersWidgets", a.id) && hj.widget.changeState(null, "hidden")
        }
        function c(a) {
            hj.log.debug("-- RENDERING TESTERS WIDGET --");
            var c = hj.widget.changeColorLuminance(a.background, - 0.1), e = hj.widget.changeColorLuminance(a.background, 0.1), m = hj.widget.changeColorLuminance(a.background, - 0.2), t = hj.widget.changeColorLuminance(a.background, 0.2), q = hj.widget.changeColorLuminance(a.background, - 0.6), n = hj.widget.changeColorLuminance(a.background, 0.6), p = {
                id: a.id,
                invitation: a.content.invitation,
                description: a.content.description,
                thankyou: a.content.thankyou
            }, c = hj.widget.renderTemplate(b, {
                apiUrlBase: new hj.widget.TrustedString(hj.apiUrlBase),
                hjHost: new hj.widget.TrustedString(hj.host),
                hjid: _hjSettings.hjid,
                preview: hj.isPreview ||!1,
                noDataPush: hj.noDataPush,
                testersWidget: p,
                widgetStyle: {
                    position: a.position,
                    skin: a.skin,
                    primaryColor: a.background,
                    secondaryColor: "light" === a.skin ? c: e,
                    alternateColor: "light" === a.skin ? m: t,
                    footerTextColor: "light" === a.skin ? q: n,
                    footerBorderColor: "light" === a.skin ? m: c,
                    fontColor: "light" ===
                    a.skin ? "#111": "#FFF",
                    fontColorNegative: "light" === a.skin ? "#FFF": "#111"
                }
            });
            hj.hq("#_hj_testers_container").remove();
            hj.log.debug("Done");
            hj.hq("body").append(c);
            f(a);
            hj.widget.init();
            "abandon" === a.display_condition && hj.hq("#_hj-f5b2a1eb-9b07_testers").removeClass("_hj-f5b2a1eb-9b07_widget_collapsed").addClass("_hj-f5b2a1eb-9b07_widget_open")
        }
        function e(a) {
            var b = void 0;
            if (a)
                return a;
            hj.hq.each(hj.settings.testers_widgets || [], function(a, c) {
                if (!hj.targeting.checkIfWidgetInCookie("_hjDoneTestersWidgets",
                c.id) && hj.targeting.ruleMatches(c.targeting))
                    return b = c, !1
            });
            return b
        }
        var a = {}, b = ['<div id="_hj_testers_container">', hj.widget.commonCSS, '<style type="text/css">                        ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_widget_description {padding:0; margin: 0 0 12px 0; text-align: center;}                        ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_input_field {margin-bottom: 6px;}                        ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_double_control {margin-bottom: 6px;}                        ._hj-f5b2a1eb-9b07_widget ._hj-f5b2a1eb-9b07_widget_content ._hj-f5b2a1eb-9b07_double_control ._hj-f5b2a1eb-9b07_input_field  {margin-bottom: 0;}                    </style><div id="_hj-f5b2a1eb-9b07_testers" class="_hj-f5b2a1eb-9b07_widget _hj-f5b2a1eb-9b07_skin_<%= widgetStyle.skin %> _hj-f5b2a1eb-9b07_position_<%= widgetStyle.position %>">                        <a class="_hj-f5b2a1eb-9b07_widget_open_close _hj-f5b2a1eb-9b07_action_toggle_widget"><span class="_hj-f5b2a1eb-9b07_widget_icon"></span></a>                        <div class="_hj-f5b2a1eb-9b07_widget_hidden_handle _hj-f5b2a1eb-9b07_action_toggle_widget"></div>                        <p class="_hj-f5b2a1eb-9b07_widget_title _hj-f5b2a1eb-9b07_action_open_widget"><%= testersWidget.invitation %></p>                        <div class="_hj-f5b2a1eb-9b07_widget_initiator">                            <button type="button" class="_hj-f5b2a1eb-9b07_btn_primary _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow _hj-f5b2a1eb-9b07_action_open_widget">Sign me up!</button>                        </div>                        <div class="_hj-f5b2a1eb-9b07_widget_state _hj-f5b2a1eb-9b07_widget_state_open">                            <div class="_hj-f5b2a1eb-9b07_widget_content _hj-f5b2a1eb-9b07_widget_content_overflow">                                <p class="_hj-f5b2a1eb-9b07_widget_description"><%= testersWidget.description %></p>                                                                <input type="text" name="_hj-f5b2a1eb-9b07_name" class="_hj-f5b2a1eb-9b07_input_field _hj-f5b2a1eb-9b07_rounded_corners" placeholder="Full name" data-bind />                                <div class="_hj-f5b2a1eb-9b07_double_control">                                     <input type="text" name="_hj-f5b2a1eb-9b07_age" class="_hj-f5b2a1eb-9b07_input_field _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_double_first" placeholder="Age" data-bind />                                    <input type="text" name="_hj-f5b2a1eb-9b07_city" class="_hj-f5b2a1eb-9b07_input_field _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_double_second" placeholder="City" data-bind />                                    <div class="_hj-f5b2a1eb-9b07_clear_both"></div>                                </div>                                <input type="text" name="_hj-f5b2a1eb-9b07_email" class="_hj-f5b2a1eb-9b07_input_field _hj-f5b2a1eb-9b07_rounded_corners" placeholder="Email" data-bind />                                <input type="text" name="_hj-f5b2a1eb-9b07_phone" class="_hj-f5b2a1eb-9b07_input_field _hj-f5b2a1eb-9b07_rounded_corners" placeholder="Phone number" data-bind />                                <div class="_hj-f5b2a1eb-9b07_double_control">                                     <div class="_hj-f5b2a1eb-9b07_button_radio _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_double_first" data-key="sex" data-value="male">                                            <span class="_hj-f5b2a1eb-9b07_widget_icon"></span>                                            <span class="_hj-f5b2a1eb-9b07_radio_text">Male</span>                                    </div>                                    <div class="_hj-f5b2a1eb-9b07_button_radio _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_double_second" data-key="sex" data-value="female">                                            <span class="_hj-f5b2a1eb-9b07_widget_icon"></span>                                            <span class="_hj-f5b2a1eb-9b07_radio_text">Female</span>                                    </div>                                    <div class="_hj-f5b2a1eb-9b07_clear_both"></div>                                </div>                            </div>                            <div class="_hj-f5b2a1eb-9b07_widget_footer">                                <div class="_hj-f5b2a1eb-9b07_pull_left">                                    <span class="_hj-f5b2a1eb-9b07_widget_icon"></span>powered by <a href="http://www.hotjar.com/?utm_source=client&utm_medium=recruit&utm_campaign=insights" target="_blank">Hotjar</a>                                </div>                                <div class="_hj-f5b2a1eb-9b07_pull_right">                                    <button type="button" id="_hj-f5b2a1eb-9b07_action_submit" class="_hj-f5b2a1eb-9b07_btn_primary _hj-f5b2a1eb-9b07_btn_disabled _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow">Send <span class="_hj-f5b2a1eb-9b07_widget_icon"></span></button>                                </div>                                <div class="_hj-f5b2a1eb-9b07_clear_both"></div>                            </div>                        </div>                        <div class="_hj-f5b2a1eb-9b07_widget_state _hj-f5b2a1eb-9b07_widget_state_thankyou">                            <p class="_hj-f5b2a1eb-9b07_thankyou_message">                                <%= testersWidget.thankyou %><br />                                <button type="button" class="_hj-f5b2a1eb-9b07_btn_primary _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow _hj-f5b2a1eb-9b07_action_dismiss_widget">Close</button>                            </p>                            <div class="_hj-f5b2a1eb-9b07_widget_footer">                                <div class="_hj-f5b2a1eb-9b07_pull_left">                                    <span class="_hj-f5b2a1eb-9b07_widget_icon"></span>powered by <a href="http://www.hotjar.com/?utm_source=client&utm_medium=recruit&utm_campaign=insights" target="_blank">Hotjar</a>                                </div>                                <div class="_hj-f5b2a1eb-9b07_pull_right">                                    <button type="button" class="_hj-f5b2a1eb-9b07_btn _hj-f5b2a1eb-9b07_btn_disabled _hj-f5b2a1eb-9b07_rounded_corners _hj-f5b2a1eb-9b07_transition _hj-f5b2a1eb-9b07_shadow">Sent <span class="_hj-f5b2a1eb-9b07_widget_icon"></span></button>                                </div>                                <div class="_hj-f5b2a1eb-9b07_clear_both"></div>                            </div>                        </div>                    </div></div>'].join("");
        a.getWidgetImportance = function(a) {
            return (a = e(a)) ? a.created_epoch_time : 0
        };
        a.run = function(a, b) {
            var f = e(b);
            f && a && (f.display_condition = hj.isPreview ? "immediate" : f.display_condition, hj.widget.callAccordingToCondition(f, "testersWidget", c))
        };
        hj.isPreview && (window._hjTestersWidgetReload = function(b) {
            a.run(!0, b)
        });
        return a
    }
    !0 !== hj.isPlayback && hj.loader.registerModule("Testers", f())
})();
(function() {
    function f() {
        var f, c;
        function e(b) {
            hj.hq(document).ready(function() {
                p = document.body;
                r = document.documentElement;
                f = Math.max(p.scrollWidth, p.offsetWidth, r.clientWidth, r.scrollWidth, r.offsetWidth);
                c = Math.max(p.scrollHeight, p.offsetHeight, r.clientHeight, r.scrollHeight, r.offsetHeight);
                a(b)
            })
        }
        function a(a) {
            a = a.data;
            "string" !== typeof a && hj.hq.each(a, function(a, c) {
                hj.hq.each(c, function(a, c) {
                    switch (c.type) {
                    case "key_press":
                        var e = d(c.selector);
                        "undefined" !== typeof e && (e = hj.hq(e), e.val(c.text));
                        break;
                    case "mouse_move":
                        992 > f || g({
                            x: c.x + (document.body.scrollLeft || window.pageXOffset),
                            y: c.y + (document.body.scrollTop || window.pageYOffset)
                        });
                        break;
                    case "mouse_click":
                        var e = d(c.selector), l;
                        if ("undefined" !== typeof e) {
                            l = hj.hq(e);
                            l.is("select") && l.is(q) ? (q.blur(), q = null) : l.is("select") ? (b(e), q = l) : null !== q && (l.is("option") && q.val(l.val()), q.blur(), q = null);
                            var s = hj.hq(e).offset();
                            l = Math.round(c.offset_x + s.left - 15) + "px";
                            s = Math.round(c.offset_y + s.top - 15) + "px";
                            l = hj.hq('<div style="border: 0 solid transparent; box-sizing: border-box; background: #DF4C2B; height: 30px; width: 30px; position: absolute; z-index: 2147483647; -webkit-border-radius: 40px; -moz-border-radius: 40px; border-radius: 40px; opacity: .4; background-clip: padding-box; left: ' +
                            l + "; top: " + s + ';"></div>');
                            hj.hq(e).is(":visible") && (hj.hq("body").append(l), l.animate({
                                borderWidth: "9px",
                                opacity: "1"
                            }, 500));
                            c.isLast || (l = document.createEvent("MouseEvents"), l.initMouseEvent("click", !0, !0, window, 1, 0, 0, 0, 0, !1, !1, !1, !1, 0, null), e.dispatchEvent(l))
                        }
                        break;
                    case "scroll":
                        window.scrollTo(0, c.y);
                        m && t && g({
                            x: m.x,
                            y: m.y + (c.y - t.y)
                        });
                        t = {
                            x: 0,
                            y: c.y
                        };
                        break;
                    case "xhr":
                        n.push(c)
                    }
                })
            })
        }
        function b(a) {
            var b = hj.hq(a), c = b.offset(), d, e;
            d = hj.hq(b.clone());
            d.attr("size", Math.min(20, b.find("option").length)).removeAttr("id");
            d.css({
                position: "absolute",
                left: c.left + "px",
                top: c.top + b.outerHeight() + "px",
                width: b.width(),
                height: "auto",
                zIndex: "1000000"
            });
            hj.hq("body").append(d);
            e = function() {
                a.removeEventListener("blur", e, !1);
                d.remove()
            };
            a.addEventListener("blur", e, !1);
            a.focus()
        }
        function d(a) {
            var b = parseInt(a.split(":")[0]);
            a = a.split(":")[1];
            return document.querySelectorAll(a)[b]
        }
        function g(a) {
            a = {
                x: a.x,
                y: a.y
            };
            if (null === m) {
                var b;
                hj.hq("body").append('<canvas id="canvasTracerLine" style="position: absolute; top:0; left: 0; z-index: 2147483640;"></canvas>');
                b = document.getElementById("canvasTracerLine");
                b.width = f;
                b.height = c
            } else 
                b = document.getElementById("canvasTracerLine").getContext("2d"), b.beginPath(), b.moveTo(m.x, m.y), b.lineTo(a.x, a.y), b.lineWidth = 3, b.strokeStyle = "rgba(223,76,43,.4)", b.stroke();
            m = a
        }
        var l = {}, m = null, t = null, q = null, n = [], p, r;
        l.run = function() {
            window.addEventListener ? window.addEventListener("message", e, !1) : window.attachEvent("onmessage", e);
            parent.playbackModuleLoaded=!0;
            hj.hq(document).off("submit", "form");
            hj.hq(document).on("submit", "form",
            !1);
            hj.hq("form").off("submit");
            hj.hq("form").on("submit", !1)
        };
        return l
    }
    !0 === hj.isPlayback && hj.loader.registerModule("Playback", f())
})();
(function() {
    hj._init = {
        _determineIncludedInSample: function() {
            var f = hj.cookie.get("_hjIncludedInSample");
            f ? hj.includedInSample = "1" === f : (hj.includedInSample = Math.random() <= (hj.settings.r || 1), hj.cookie.set("_hjIncludedInSample", hj.includedInSample ? "1" : "0", !0))
        },
        _determineUserId: function() {
            hj.userId = hj.cookie.get("_hjUserId");
            hj.userId || (hj.userId = hj.utils.uuid4(), hj.cookie.set("_hjUserId", hj.userId))
        },
        run: function() {
            9 > hj.utils.ieVersion() ? hj.log.info("IE < 9 is not supported") : hj.loader.loadSettings(function(f) {
                function h() {
                    var b =
                    [];
                    10 > hj.utils.ieVersion() && b.push("//cdn.jsdelivr.net/placeholders/3.0.2/placeholders.min.js");
                    hj.loader.loadScripts(b, function() {
                        hj.hq.each(hj.loader.getModules(), function(b, d) {
                            hj.hq.isFunction(d.module.getWidgetImportance) && (a = d.module.getWidgetImportance(), a > e && (e = a, c = d.module))
                        });
                        hj.hq.each(hj.loader.getModules(), function(a, b) {
                            hj.log.info("Running module " + b.name);
                            b.module.run(void 0 !== c && c === b.module)
                        })
                    })
                }
                var c = void 0, e =- 1, a, b;
                hj.settings = f || {};
                hj.log.info("Site settings: " + hj.hq.stringify(hj.settings,
                null, "\t"));
                - 1 === navigator.userAgent.indexOf("Hotjar") && (hj._init._determineIncludedInSample(), hj._init._determineUserId(), hj.dataPusher.init(), hj.isPlayback ? b = setInterval(function() {
                    null !== document.body && (clearInterval(b), h())
                }, 200) : h())
            })
        }
    };
    hj._init.run()
})();

