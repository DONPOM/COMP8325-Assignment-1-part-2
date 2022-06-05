// All code and conventions are protected by copyright
!function(e, a, n) {
    function i() {
        y.addEventHandler(e, "orientationchange", i.orientationChange)
    }
    function o() {
        this.rules = y.filter(y.rules, function(e) {
            return "elementexists" === e.event
        })
    }
    function s() {
        this.rules = y.filter(y.rules, function(e) {
            return "videoplayed" === e.event.substring(0, 11)
        }), this.eventHandler = y.bind(this.onUpdateTime, this)
    }
    function r() {
        var e = this.eventRegex = /^hover\(([0-9]+)\)$/, t = this.rules = [];
        y.each(y.rules, function(a) {
            var n = a.event.match(e);
            n && t.push([Number(a.event.match(e)[1]), a.selector])
        })
    }
    function c(t) {
        y.domReady(y.bind(function() {
            this.twttr = t || e.twttr, this.initialize()
        }, this))
    }
    function l(e) {
        this.delay = 250, this.FB = e, y.domReady(y.bind(function() {
            y.poll(y.bind(this.initialize, this), this.delay, 8)
        }, this))
    }
    function d(t) {
        t = t || y.rules, this.rules = y.filter(t, function(e) {
            return "inview" === e.event
        }), this.elements = [], this.eventHandler = y.bind(this.track, this), y.addEventHandler(e, "scroll", this.eventHandler), y.addEventHandler(e, "load", this.eventHandler)
    }
    function f(e) {
        y.BaseTool.call(this, e), this.name = e.name || "VisitorID", this.initialize()
    }
    function u(e) {
        y.BaseTool.call(this, e)
    }
    function h(e) {
        y.BaseTool.call(this, e)
    }
    function b(e) {
        y.BaseTool.call(this, e), this.varBindings = {}, this.events = [], this.products = [], this.customSetupFuns = []
    }
    function p() {
        y.BaseTool.call(this), this.asyncScriptCallbackQueue = [], this.argsForBlockingScripts = []
    }
    function g(e) {
        y.BaseTool.call(this, e), this.styleElements = {}, this.targetPageParamsStore = {}
    }
    function m(e) {
        y.BaseTool.call(this, e), this.name = e.name || "Basic"
    }
    var v = Object.prototype.toString, _ = e._satellite && e._satellite.override, y = {
        $data: function(e, t, a) {
            var i = "__satellite__", o = y.dataCache, s = e[i];
            s || (s = e[i] = y.uuid++);
            var r = o[s];
            return r || (r = o[s] = {}), a === n ? r[t] : void(r[t] = a)
        },
        uuid: 1,
        dataCache: {},
        keys: function(e) {
            var t = [];
            for (var a in e)
                t.push(a);
            return t
        },
        values: function(e) {
            var t = [];
            for (var a in e)
                t.push(e[a]);
            return t
        },
        isArray: Array.isArray || function(e) {
            return "[object Array]" === v.apply(e)
        },
        isObject: function(e) {
            return null != e&&!y.isArray(e) && "object" == typeof e
        },
        isString: function(e) {
            return "string" == typeof e
        },
        isNumber: function(e) {
            return "[object Number]" === v.apply(e)&&!y.isNaN(e)
        },
        isNaN: function(e) {
            return e !== e
        },
        isRegex: function(e) {
            return e instanceof RegExp
        },
        isLinkTag: function(e) {
            return !(!e ||!e.nodeName || "a" !== e.nodeName.toLowerCase())
        },
        each: function(e, t, a) {
            for (var n = 0, i = e.length; i > n; n++)
                t.call(a, e[n], n, e)
        },
        map: function(e, t, a) {
            for (var n = [], i = 0, o = e.length; o > i; i++)
                n.push(t.call(a, e[i], i, e));
            return n
        },
        filter: function(e, t, a) {
            for (var n = [], i = 0, o = e.length; o > i; i++) {
                var s = e[i];
                t.call(a, s, i, e) && n.push(s)
            }
            return n
        },
        any: function(e, t, a) {
            for (var n = 0, i = e.length; i > n; n++) {
                var o = e[n];
                if (t.call(a, o, n, e))
                    return !0
            }
            return !1
        },
        every: function(e, t, a) {
            for (var n=!0, i = 0, o = e.length; o > i; i++) {
                var s = e[i];
                n = n && t.call(a, s, i, e)
            }
            return n
        },
        contains: function(e, t) {
            return - 1 !== y.indexOf(e, t)
        },
        indexOf: function(e, t) {
            if (e.indexOf)
                return e.indexOf(t);
            for (var a = e.length; a--;)
                if (t === e[a])
                    return a;
            return - 1
        },
        find: function(e, t, a) {
            for (var n = 0, i = e.length; i > n; n++) {
                var o = e[n];
                if (t.call(a, o, n, e))
                    return o
            }
            return null
        },
        textMatch: function(e, t) {
            return "string" == typeof t ? e === t : t instanceof RegExp ? t.test(e) : !1
        },
        stringify: function(e, t) {
            if (t = t || [], y.isObject(e)) {
                if (y.contains(t, e))
                    return "<Cycle>";
                t.push(e)
            }
            if (y.isArray(e))
                return "[" + y.map(e, function(e) {
                    return y.stringify(e, t)
                }).join(",") + "]";
            if (y.isString(e))
                return '"' + String(e) + '"';
            if (y.isObject(e)) {
                var a = [];
                for (var n in e)
                    a.push(n + ": " + y.stringify(e[n], t));
                return "{" + a.join(", ") + "}"
            }
            return String(e)
        },
        trim: function(e) {
            return null == e ? null : e.trim ? e.trim() : e.replace(/^ */, "").replace(/ *$/, "")
        },
        bind: function(e, t) {
            return function() {
                return e.apply(t, arguments)
            }
        },
        throttle: function(e, t) {
            var a = null;
            return function() {
                var n = this, i = arguments;
                clearTimeout(a), a = setTimeout(function() {
                    e.apply(n, i)
                }, t)
            }
        },
        domReady: function(e) {
            function t(e) {
                for (u = 1; e = i.shift();)
                    e()
            }
            var n, i = [], o=!1, s = a, r = s.documentElement, c = r.doScroll, l = "DOMContentLoaded", d = "addEventListener", f = "onreadystatechange", u = /^loade|^c/.test(s.readyState);
            return s[d] && s[d](l, n = function() {
                s.removeEventListener(l, n, o), t()
            }, o), c && s.attachEvent(f, n = function() {
                /^c/.test(s.readyState) && (s.detachEvent(f, n), t())
            }), e = c ? function(t) {
                self != top ? u ? t() : i.push(t) : function() {
                    try {
                        r.doScroll("left")
                    } catch (a) {
                        return setTimeout(function() {
                            e(t)
                        }, 50)
                    }
                    t()
                }()
            } : function(e) {
                u ? e() : i.push(e)
            }
        }(),
        loadScript: function(e, t) {
            var n = a.createElement("script");
            y.scriptOnLoad(e, n, t), n.src = e, a.getElementsByTagName("head")[0].appendChild(n)
        },
        scriptOnLoad: function(e, t, a) {
            function n(e) {
                e && y.logError(e), a && a(e)
            }
            "onload"in t ? (t.onload = function() {
                n()
            }, t.onerror = function() {
                n(new Error("Failed to load script " + e))
            }) : "readyState"in t && (t.onreadystatechange = function() {
                var e = t.readyState;
                ("loaded" === e || "complete" === e) && (t.onreadystatechange = null, n())
            })
        },
        loadScriptOnce: function(e, t) {
            y.loadedScriptRegistry[e] || y.loadScript(e, function(a) {
                a || (y.loadedScriptRegistry[e]=!0), t && t(a)
            })
        },
        loadedScriptRegistry: {},
        loadScriptSync: function(e) {
            return y.domReadyFired ? void y.notify('Cannot load sync the "' + e + '" script after DOM Ready.', 1) : void a.write('<script src="' + e + '"></script>')
        },
        pushAsyncScript: function(e) {
            y.tools["default"].pushAsyncScript(e)
        },
        pushBlockingScript: function(e) {
            y.tools["default"].pushBlockingScript(e)
        },
        addEventHandler: e.addEventListener ? function(e, t, a) {
            e.addEventListener(t, a, !1)
        }
        : function(e, t, a) {
            e.attachEvent("on" + t, a)
        },
        preventDefault: e.addEventListener ? function(e) {
            e.preventDefault()
        }
        : function(e) {
            e.returnValue=!1
        },
        stopPropagation: function(e) {
            e.cancelBubble=!0, e.stopPropagation && e.stopPropagation()
        },
        containsElement: function(e, t) {
            return e.contains ? e.contains(t) : !!(16 & e.compareDocumentPosition(t))
        },
        matchesCss: function(t) {
            function n(e, t) {
                var a = t.tagName;
                return a ? e.toLowerCase() === a.toLowerCase() : !1
            }
            var i = t.matchesSelector || t.mozMatchesSelector || t.webkitMatchesSelector || t.oMatchesSelector || t.msMatchesSelector;
            return i ? function(t, n) {
                if (n === a || n === e)
                    return !1;
                try {
                    return i.call(n, t)
                } catch (o) {
                    return !1
                }
            } : t.querySelectorAll ? function(e, t) {
                var a = t.parentNode;
                if (!a)
                    return !1;
                if (e.match(/^[a-z]+$/i))
                    return n(e, t);
                try {
                    for (var i = t.parentNode.querySelectorAll(e), o = i.length; o--;)
                        if (i[o] === t)
                            return !0
                } catch (s) {}
                return !1
            } : function(e, t) {
                if (e.match(/^[a-z]+$/i))
                    return n(e, t);
                try {
                    return y.Sizzle.matches(e, [t]).length > 0
                } catch (a) {
                    return !1
                }
            }
        }(a.documentElement),
        cssQuery: function(e) {
            return e.querySelectorAll ? function(t, a) {
                var n;
                try {
                    n = e.querySelectorAll(t)
                } catch (i) {
                    n = []
                }
                a(n)
            } : function(e, t) {
                if (y.Sizzle) {
                    var a;
                    try {
                        a = y.Sizzle(e)
                    } catch (n) {
                        a = []
                    }
                    t(a)
                } else 
                    y.sizzleQueue.push([e, t])
            }
        }(a),
        hasAttr: function(e, t) {
            return e.hasAttribute ? e.hasAttribute(t) : e[t] !== n
        },
        inherit: function(e, t) {
            var a = function() {};
            a.prototype = t.prototype, e.prototype = new a, e.prototype.constructor = e
        },
        extend: function(e, t) {
            for (var a in t)
                t.hasOwnProperty(a) && (e[a] = t[a])
        },
        toArray: function() {
            try {
                var e = Array.prototype.slice;
                return e.call(a.documentElement.childNodes, 0)[0].nodeType, function(t) {
                    return e.call(t, 0)
                }
            } catch (t) {
                return function(e) {
                    for (var t = [], a = 0, n = e.length; n > a; a++)
                        t.push(e[a]);
                    return t
                }
            }
        }(),
        equalsIgnoreCase: function(e, t) {
            return null == e ? null == t : null == t?!1 : String(e).toLowerCase() === String(t).toLowerCase()
        },
        poll: function(e, t, a) {
            function n() {
                y.isNumber(a) && i++>=a || e() || setTimeout(n, t)
            }
            var i = 0;
            t = t || 1e3, n()
        },
        escapeForHtml: function(e) {
            return e ? e.replace(/\&/g, "&amp;").replace(/\</g, "&lt;").replace(/\>/g, "&gt;").replace(/\"/g, "&quot;").replace(/\'/g, "&#x27;").replace(/\//g, "&#x2F;") : e
        }
    };
    y.availableTools = {}, y.availableEventEmitters = [], y.fireOnceEvents = ["condition", "elementexists"], y.initEventEmitters = function() {
        y.eventEmitters = y.map(y.availableEventEmitters, function(e) {
            return new e
        })
    }, y.eventEmitterBackgroundTasks = function() {
        y.each(y.eventEmitters, function(e) {
            "backgroundTasks"in e && e.backgroundTasks()
        })
    }, y.initTools = function(e) {
        var t = {
            "default": new p
        }, a = y.settings.euCookieName || "sat_track";
        for (var n in e) {
            var i, o, s;
            if (i = e[n], i.euCookie) {
                var r = "true" !== y.readCookie(a);
                if (r)
                    continue
            }
            if (o = y.availableTools[i.engine], !o) {
                var c = [];
                for (var l in y.availableTools)
                    c.push(l);
                throw new Error("No tool engine named " + i.engine + ", available: " + c.join(",") + ".")
            }
            s = new o(i), s.id = n, t[n] = s
        }
        return t
    }, y.preprocessArguments = function(e, t, a, n) {
        function i(e) {
            return n && y.isString(e) ? e.toLowerCase() : e
        }
        function o(e) {
            var n = {};
            for (var r in e)
                if (e.hasOwnProperty(r)) {
                    var c = e[r];
                    n[r] = y.isObject(c) ? o(c) : y.isArray(c) ? s(c) : i(y.replace(c, t, a))
                }
            return n
        }
        function s(e) {
            for (var n = [], s = 0, r = e.length; r > s; s++) {
                var c = e[s];
                y.isString(c) ? c = i(y.replace(c, t, a)) : c && c.constructor === Object && (c = o(c)), n.push(c)
            }
            return n
        }
        return e ? s(e, n) : e
    }, y.execute = function(e, t, a, n) {
        function i(i) {
            var o = n[i || "default"];
            if (o)
                try {
                    o.triggerCommand(e, t, a)
            } catch (s) {
                y.logError(s)
            }
        }
        if (!_satellite.settings.hideActivity)
            if (n = n || y.tools, e.engine) {
                var o = e.engine;
                for (var s in n) {
                    var r = n[s];
                    r.settings && r.settings.engine === o && i(s)
                }
            } else 
                e.tool instanceof Array ? y.each(e.tool, function(e) {
                    i(e)
                }) : i(e.tool)
        }, y.notify = e.console ? function(e, t) {
        if (y.settings.notifications)
            switch (t) {
            case 1:
            case 2:
            case 3:
                console.log("SATELLITE: " + e);
                break;
            case 4:
                console.warn("SATELLITE: " + e);
                break;
            case 5:
                console.error("SATELLITE: " + e);
                break;
            default:
                console.log("SATELLITE: Notify called with incorrect priority.")
            }
    } : function() {}, y.cleanText = function(e) {
        return null == e ? null : y.trim(e).replace(/\s{2,}/g, " ").replace(/[^\000-\177]*/g, "")
    }, y.text = function(e) {
        return e.textContent || e.innerText
    }, y.specialProperties = {
        text: y.text,
        cleanText: function(e) {
            return y.cleanText(y.text(e))
        }
    }, y.getObjectProperty = function(e, t, a) {
        for (var i, o = t.split("."), s = e, r = y.specialProperties, c = 0, l = o.length; l > c; c++) {
            if (null == s)
                return n;
            var d = o[c];
            if (a && "@" === d.charAt(0)) {
                var f = d.slice(1);
                s = r[f](s)
            } else if (s.getAttribute && (i = d.match(/^getAttribute\((.+)\)$/))) {
                var u = i[1];
                s = s.getAttribute(u)
            } else 
                s = s[d]
        }
        return s
    }, y.getToolsByType = function(e) {
        if (!e)
            throw new Error("Tool type is missing");
        var a = [];
        for (t in y.tools) {
            var n = y.tools[t];
            n.settings && n.settings.engine === e && a.push(n)
        }
        return a
    }, y.setVar = function() {
        var e = y.data.customVars;
        if (null == e && (y.data.customVars = {}, e = y.data.customVars), "string" == typeof arguments[0]) {
            var t = arguments[0];
            e[t] = arguments[1]
        } else if (arguments[0]) {
            var a = arguments[0];
            for (var n in a)
                e[n] = a[n]
        }
    }, y.dataElementSafe = function(e, t) {
        if (arguments.length > 2) {
            var a = arguments[2];
            "pageview" === t ? y.dataElementSafe.pageviewCache[e] = a : "session" === t ? y.setCookie("_sdsat_" + e, a) : "visitor" === t && y.setCookie("_sdsat_" + e, a, 730)
        } else {
            if ("pageview" === t)
                return y.dataElementSafe.pageviewCache[e];
            if ("session" === t || "visitor" === t)
                return y.readCookie("_sdsat_" + e)
        }
    }, y.dataElementSafe.pageviewCache = {}, y.realGetDataElement = function(t) {
        var a;
        return t.selector ? y.hasSelector && y.cssQuery(t.selector, function(e) {
            if (e.length > 0) {
                var n = e[0];
                "text" === t.property ? a = n.innerText || n.textContent : y.hasAttr(n, t.property) && (a = n[t.property] || n.getAttribute(t.property))
            }
        }) : t.queryParam ? a = t.ignoreCase ? y.getQueryParamCaseInsensitive(t.queryParam) : y.getQueryParam(t.queryParam) : t.cookie ? a = y.readCookie(t.cookie) : t.jsVariable ? a = y.getObjectProperty(e, t.jsVariable) : t.customJS && (a = t.customJS()), a
    }, y.getDataElement = function(e, t, a) {
        a = a || y.dataElements[e];
        var i = y.realGetDataElement(a);
        return i === n && a.storeLength ? i = y.dataElementSafe(e, a.storeLength) : i !== n && a.storeLength && y.dataElementSafe(e, a.storeLength, i), i !== n || t || (i = a["default"] || ""), a.forceLowerCase && (i = i.toLowerCase()), i
    }, y.getVar = function(t, i, o) {
        var s, r, c = y.data.customVars, l = o ? o.target || o.srcElement: null, d = {
            URI: y.data.URI,
            uri: y.data.URI,
            protocol: a.location.protocol,
            hostname: a.location.hostname
        };
        if (y.dataElements && t in y.dataElements)
            return y.getDataElement(t);
        if (r = d[t], r === n)
            if ("this." === t.substring(0, 5))
                t = t.slice(5), r = y.getObjectProperty(i, t, !0);
            else if ("event." === t.substring(0, 6))
                t = t.slice(6), r = y.getObjectProperty(o, t);
            else if ("target." === t.substring(0, 7))
                t = t.slice(7), r = y.getObjectProperty(l, t);
            else if ("window." === t.substring(0, 7))
                t = t.slice(7), r = y.getObjectProperty(e, t);
            else if ("param." === t.substring(0, 6))
                t = t.slice(6), r = y.getQueryParam(t);
            else if (s = t.match(/^rand([0-9]+)$/)) {
                var f = Number(s[1]), u = (Math.random() * (Math.pow(10, f) - 1)).toFixed(0);
                r = Array(f - u.length + 1).join("0") + u
            } else 
                r = y.getObjectProperty(c, t);
        return r
    }, y.getVars = function(e, t, a) {
        var n = {};
        return y.each(e, function(e) {
            n[e] = y.getVar(e, t, a)
        }), n
    }, y.replace = function(e, t, a) {
        return "string" != typeof e ? e : e.replace(/%(.*?)%/g, function(e, n) {
            var i = y.getVar(n, t, a);
            return null == i ? e : i
        })
    }, y.searchVariables = function(e, t, a) {
        if (!e || 0 === e.length)
            return "";
        for (var n = [], i = 0, o = e.length; o > i; i++) {
            var s = e[i], r = y.getVar(s, t, a);
            n.push(s + "=" + escape(r))
        }
        return "?" + n.join("&")
    }, y.fireRule = function(e, t, a) {
        var n = e.trigger;
        if (n) {
            for (var i = 0, o = n.length; o > i; i++) {
                var s = n[i];
                y.execute(s, t, a)
            }
            y.contains(y.fireOnceEvents, e.event) && (e.expired=!0)
        }
    }, y.isLinked = function(e) {
        for (var t = e; t; t = t.parentNode)
            if (y.isLinkTag(t))
                return !0;
        return !1
    }, y.firePageLoadEvent = function(e) {
        for (var t = a.location, n = {
            type: e,
            target: t
        }, i = y.pageLoadRules, o = i.length; o--;) {
            var s = i[o];
            y.ruleMatches(s, n, t) && (y.notify('Rule "' + s.name + '" fired.', 1), y.fireRule(s, t, n), i.splice(o, 1))
        }
        for (var r in y.tools) {
            var c = y.tools[r];
            c.endPLPhase && c.endPLPhase(e)
        }
    }, y.track = function(e) {
        e = e.replace(/^\s*/, "").replace(/\s*$/, "");
        for (var t = 0; t < y.directCallRules.length; t++) {
            var a = y.directCallRules[t];
            if (a.name === e)
                return y.notify('Direct call Rule "' + e + '" fired.', 1), void y.fireRule(a, location, {
                type: e
            })
        }
        y.notify('Direct call Rule "' + e + '" not found.', 1)
    }, y.basePath = function() {
        return y.data.host ? ("https:" === a.location.protocol ? "https://" + y.data.host.https : "http://" + y.data.host.http) + "/" : this.settings.basePath
    }, y.setLocation = function(t) {
        e.location = t
    }, y.parseQueryParams = function(e) {
        var t = function(e) {
            var t = e;
            try {
                t = decodeURIComponent(e)
            } catch (a) {}
            return t
        };
        if ("" === e || y.isString(e)===!1)
            return {};
        0 === e.indexOf("?") && (e = e.substring(1));
        var a = {}, n = e.split("&");
        return y.each(n, function(e) {
            e = e.split("="), e[1] && (a[t(e[0])] = y.escapeForHtml(t(e[1])))
        }), a
    }, y.getCaseSensitivityQueryParamsMap = function(e) {
        var t = y.parseQueryParams(e), a = {};
        for (var n in t)
            t.hasOwnProperty(n) && (a[n.toLowerCase()] = t[n]);
        return {
            normal: t,
            caseInsensitive: a
        }
    }, y.QueryParams = y.getCaseSensitivityQueryParamsMap(e.location.search), y.getQueryParam = function(e) {
        return y.QueryParams.normal[e]
    }, y.getQueryParamCaseInsensitive = function(e) {
        return y.QueryParams.caseInsensitive[e.toLowerCase()]
    }, y.encodeObjectToURI = function(e) {
        if (y.isObject(e)===!1)
            return "";
        var t = [];
        for (var a in e)
            e.hasOwnProperty(a) && t.push(encodeURIComponent(a) + "=" + encodeURIComponent(e[a]));
        return t.join("&")
    }, y.readCookie = function(e) {
        for (var t = e + "=", i = a.cookie.split(";"), o = 0; o < i.length; o++) {
            for (var s = i[o]; " " == s.charAt(0);)
                s = s.substring(1, s.length);
            if (0 === s.indexOf(t))
                return s.substring(t.length, s.length)
        }
        return n
    }, y.setCookie = function(e, t, n) {
        var i;
        if (n) {
            var o = new Date;
            o.setTime(o.getTime() + 24 * n * 60 * 60 * 1e3), i = "; expires=" + o.toGMTString()
        } else 
            i = "";
        a.cookie = e + "=" + t + i + "; path=/"
    }, y.removeCookie = function(e) {
        y.setCookie(e, "", - 1)
    }, y.getElementProperty = function(e, t) {
        if ("@" === t.charAt(0)) {
            var a = y.specialProperties[t.substring(1)];
            if (a)
                return a(e)
        }
        return "innerText" === t ? y.text(e) : t in e ? e[t] : e.getAttribute ? e.getAttribute(t) : n
    }, y.propertiesMatch = function(e, t) {
        if (e)
            for (var a in e) {
                var n = e[a], i = y.getElementProperty(t, a);
                if ("string" == typeof n && n !== i)
                    return !1;
                    if (n instanceof RegExp&&!n.test(i))
                        return !1
            }
        return !0
    }, y.ruleMatches = function(e, t, n, i) {
        var o = (a.location, e.condition), s = e.conditions, r = e.property, c = t.type, l = e.value, d = t.target || t.srcElement, f = n === d;
        if (e.event !== c)
            return !1;
        if (e.isDefault && i > 0)
            return !1;
        if (e.expired)
            return !1;
        if ("inview" === c && t.inviewDelay !== e.inviewDelay)
            return !1;
        if (!f && (e.bubbleFireIfParent===!1 || 0 !== i && e.bubbleFireIfChildFired===!1))
            return !1;
        if (e.selector&&!y.matchesCss(e.selector, n))
            return !1;
        if (!y.propertiesMatch(r, n))
            return !1;
        if (null != l)
            if ("string" == typeof l) {
                if (l !== n.value)
                    return !1
            } else if (!l.test(n.value))
                return !1;
        if (o)
            try {
                if (!o.call(n, t, d))
                    return y.notify('Condition for rule "' + e.name + '" not met.', 1), !1
        } catch (u) {
            return y.notify('Condition for rule "' + e.name + '" not met. Error: ' + u.message, 1), !1
        }
        if (s) {
            var h = y.find(s, function(a) {
                try {
                    return !a.call(n, t, d)
                } catch (i) {
                    return y.notify('Condition for rule "' + e.name + '" not met. Error: ' + i.message, 1), !0
                }
            });
            if (h)
                return y.notify("Condition " + h.toString() + ' for rule "' + e.name + '" not met.', 1), !1
        }
        return !0
    }, y.evtHandlers = {}, y.bindEvent = function(e, t) {
        var a = y.evtHandlers;
        a[e] || (a[e] = []), a[e].push(t)
    }, y.whenEvent = y.bindEvent, y.unbindEvent = function(e, t) {
        var a = y.evtHandlers;
        if (a[e]) {
            var n = y.indexOf(a[e], t);
            a[e].splice(n, 1)
        }
    }, y.bindEventOnce = function(e, t) {
        var a = function() {
            y.unbindEvent(e, a), t.apply(null, arguments)
        };
        y.bindEvent(e, a)
    }, y.isVMLPoisoned = function(e) {
        if (!e)
            return !1;
        try {
            e.nodeName
        } catch (t) {
            if ("Attribute only valid on v:image" === t.message)
                return !0
        }
        return !1
    }, y.handleEvent = function(e) {
        if (!y.$data(e, "eventProcessed")) {
            var t = e.type.toLowerCase(), a = e.target || e.srcElement, n = 0, i = y.rules, o = (y.tools, y.evtHandlers[e.type]);
            if (y.isVMLPoisoned(a))
                return void y.notify("detected " + t + " on poisoned VML element, skipping.", 1);
            o && y.each(o, function(t) {
                t(e)
            });
            var s = a && a.nodeName;
            s ? y.notify("detected " + t + " on " + a.nodeName, 1) : y.notify("detected " + t, 1);
            for (var r = a; r; r = r.parentNode) {
                var c=!1;
                if (y.each(i, function(t) {
                    y.ruleMatches(t, e, r, n) && (y.notify('Rule "' + t.name + '" fired.', 1), y.fireRule(t, r, e), n++, t.bubbleStop && (c=!0))
                }), c)
                    break
            }
            y.$data(e, "eventProcessed", !0)
        }
    }, y.onEvent = a.querySelectorAll ? function(e) {
        y.handleEvent(e)
    } : function() {
        var e = [], t = function(t) {
            t.selector ? e.push(t) : y.handleEvent(t)
        };
        return t.pendingEvents = e, t
    }(), y.fireEvent = function(e, t) {
        y.onEvent({
            type: e,
            target: t
        })
    }, y.registerEvents = function(e, t) {
        for (var a = t.length - 1; a >= 0; a--) {
            var n = t[a];
            y.$data(e, n + ".tracked") || (y.addEventHandler(e, n, y.onEvent), y.$data(e, n + ".tracked", !0))
        }
    }, y.registerEventsForTags = function(e, t) {
        for (var n = e.length - 1; n >= 0; n--)
            for (var i = e[n], o = a.getElementsByTagName(i), s = o.length - 1; s >= 0; s--)
                y.registerEvents(o[s], t)
    }, y.setListeners = function() {
        y.registerEvents(a, ["click", "submit"])
    }, y.setFormListeners = function() {
        y.registerEventsForTags(["input", "select", "textarea", "button"], ["select", "change", "focus", "blur", "keypress"])
    }, y.setVideoListeners = function() {
        y.registerEventsForTags(["video"], ["play", "pause", "ended", "volumechange", "stalled", "timeupdate", "loadeddata"])
    }, y.readStoredSetting = function(t) {
        return e.localStorage ? (t = "sdsat_" + t, e.localStorage.getItem(t)) : null
    }, y.loadStoredSettings = function() {
        var e = y.readStoredSetting("debug"), t = y.readStoredSetting("hide_activity");
        e && (y.settings.notifications = "true" === e), t && (y.settings.hideActivity = "true" === t)
    }, y.isRuleActive = function(e, t) {
        function a(e, t) {
            return t = i(t, {
                hour: e[h](),
                minute: e[b]()
            }), Math.floor(Math.abs((e.getTime() - t.getTime()) / 864e5))
        }
        function n(e, t) {
            function a(e) {
                return 12 * e[f]() + e[u]()
            }
            return Math.abs(a(e) - a(t))
        }
        function i(e, t) {
            var a = new Date(e.getTime());
            for (var n in t) {
                var i = t[n];
                switch (n) {
                case"hour":
                    a[p](i);
                    break;
                case"minute":
                    a[g](i);
                    break;
                case"date":
                    a[m](i)
                }
            }
            return a
        }
        function o(e, t) {
            var a = e[h](), n = e[b](), i = t[h](), o = t[b]();
            return 60 * a + n > 60 * i + o
        }
        function s(e, t) {
            var a = e[h](), n = e[b](), i = t[h](), o = t[b]();
            return 60 * i + o > 60 * a + n
        }
        var r = e.schedule;
        if (!r)
            return !0;
        var c = r.utc, l = c ? "getUTCDate": "getDate", d = c ? "getUTCDay": "getDay", f = c ? "getUTCFullYear": "getFullYear", u = c ? "getUTCMonth": "getMonth", h = c ? "getUTCHours": "getHours", b = c ? "getUTCMinutes": "getMinutes", p = c ? "setUTCHours": "setHours", g = c ? "setUTCMinutes": "setMinutes", m = c ? "setUTCDate": "setDate";
        if (t = t || new Date, r.repeat) {
            if (o(r.start, t))
                return !1;
            if (s(r.end, t))
                return !1;
            if (t < r.start)
                return !1;
            if (r.endRepeat && t >= r.endRepeat)
                return !1;
            if ("daily" === r.repeat) {
                if (r.repeatEvery) {
                    var v = a(r.start, t);
                    if (v%r.repeatEvery !== 0)
                        return !1
                }
            } else if ("weekly" === r.repeat) {
                if (r.days) {
                    if (!y.contains(r.days, t[d]()))
                        return !1
                } else if (r.start[d]() !== t[d]())
                    return !1;
                if (r.repeatEvery) {
                    var _ = a(r.start, t);
                    if (_%(7 * r.repeatEvery) !== 0)
                        return !1
                }
            } else if ("monthly" === r.repeat) {
                if (r.repeatEvery) {
                    var k = n(r.start, t);
                    if (k%r.repeatEvery !== 0)
                        return !1
                }
                if (r.nthWeek && r.mthDay) {
                    if (r.mthDay !== t[d]())
                        return !1;
                    var w = Math.floor((t[l]() - t[d]() + 1) / 7);
                    if (r.nthWeek !== w)
                        return !1
                } else if (r.start[l]() !== t[l]())
                    return !1
            } else if ("yearly" === r.repeat) {
                if (r.start[u]() !== t[u]())
                    return !1;
                if (r.start[l]() !== t[l]())
                    return !1;
                if (r.repeatEvery) {
                    var _ = Math.abs(r.start[f]() - t[f]());
                    if (_%r.repeatEvery !== 0)
                        return !1
                }
            }
        } else {
            if (r.start > t)
                return !1;
            if (r.end < t)
                return !1
        }
        return !0
    }, y.isOutboundLink = function(e) {
        if (!e.getAttribute("href"))
            return !1;
        var t = e.hostname, a = (e.href, e.protocol);
        if ("http:" !== a && "https:" !== a)
            return !1;
        var n = y.any(y.settings.domainList, function(e) {
            return y.isSubdomainOf(t, e)
        });
        return n?!1 : t !== location.hostname
    }, y.isLinkerLink = function(e) {
        return e.getAttribute && e.getAttribute("href") ? y.hasMultipleDomains() && e.hostname != location.hostname&&!e.href.match(/^javascript/i)&&!y.isOutboundLink(e) : !1
    }, y.isSubdomainOf = function(e, t) {
        if (e === t)
            return !0;
        var a = e.length - t.length;
        return a > 0 ? y.equalsIgnoreCase(e.substring(a), t) : !1
    }, y.getVisitorId = function() {
        var e = y.getToolsByType("visitor_id");
        return 0 === e.length ? null : e[0].getInstance()
    }, y.filterRules = function() {
        function e(e) {
            return y.ruleInScope(e, t) && y.isRuleActive(e)?!0 : !1
        }
        var t = {
            hostname: location.hostname,
            protocol: location.protocol,
            URI: y.data.URI
        };
        y.rules = y.filter(y.rules, e), y.pageLoadRules = y.filter(y.pageLoadRules, e)
    }, y.ruleInScope = function(e, t) {
        function a(e, t) {
            function a(e) {
                return t.match(e)
            }
            var i = e.include, o = e.exclude;
            if (i && n(i, t))
                return !0;
            if (o) {
                if (y.isString(o) && o === t)
                    return !0;
                if (y.isArray(o) && y.any(o, a))
                    return !0;
                if (y.isRegex(o) && a(o))
                    return !0
            }
            return !1
        }
        function n(e, t) {
            function a(e) {
                return t.match(e)
            }
            return y.isString(e) && e !== t?!0 : y.isArray(e)&&!y.any(e, a)?!0 : y.isRegex(e)&&!a(e)?!0 : !1
        }
        var i = e.scope;
        if (!i)
            return !0;
        var o = i.URI, s = i.subdomains, r = i.domains, c = i.protocols;
        return o && a(o, t.URI)?!1 : s && a(s, t.hostname)?!1 : r && n(r, t.hostname)?!1 : c && n(c, t.protocol)?!1 : !0
    }, y.backgroundTasks = function() {
        + new Date;
        y.setFormListeners(), y.setVideoListeners(), y.loadStoredSettings(), y.registerNewElementsForDynamicRules(), y.eventEmitterBackgroundTasks();
        + new Date
    }, y.registerNewElementsForDynamicRules = function() {
        function e(t, a) {
            var n = e.cache[t];
            return n ? a(n) : void y.cssQuery(t, function(n) {
                e.cache[t] = n, a(n)
            })
        }
        e.cache = {}, y.each(y.dynamicRules, function(t) {
            e(t.selector, function(e) {
                y.each(e, function(e) {
                    y.$data(e, "dynamicRules.seen") || (y.$data(e, "dynamicRules.seen", !0), y.propertiesMatch(t.property, e) && y.registerEvents(e, [t.event]))
                })
            })
        })
    }, y.ensureCSSSelector = function() {
        return a.querySelectorAll ? void(y.hasSelector=!0) : (y.loadingSizzle=!0, y.sizzleQueue = [], void y.loadScript(y.basePath() + "selector.js", function() {
            if (!y.Sizzle)
                return void y.logError(new Error("Failed to load selector.js"));
            var e = y.onEvent.pendingEvents;
            y.each(e, function(e) {
                y.handleEvent(e)
            }, this), y.onEvent = y.handleEvent, y.hasSelector=!0, delete y.loadingSizzle, y.each(y.sizzleQueue, function(e) {
                y.cssQuery(e[0], e[1])
            }), delete y.sizzleQueue
        }))
    }, y.errors = [], y.logError = function(e) {
        y.errors.push(e), y.notify(e.name + " - " + e.message, 5)
    }, y.pageBottom = function() {
        y.pageBottomFired=!0, y.firePageLoadEvent("pagebottom")
    }, y.stagingLibraryOverride = function() {
        var e = "true" === y.readStoredSetting("stagingLibrary");
        if (e) {
            for (var t, n, i, o = a.getElementsByTagName("script"), s = /^(.*)satelliteLib-(.*)\.js$/, r = /^(.*)satelliteLib-(.*)-staging\.js$/, c = 0, l = o.length; l > c && (i = o[c].getAttribute("src"), !i || (t || (t = i.match(s)), n || (n = i.match(r)), !n)); c++);
            if (t&&!n) {
                var d = t[1] + "satelliteLib-" + t[2] + "-staging.js";
                return a.write('<script src="' + d + '"></script>'), !0
            }
        }
        return !1
    }, y.checkAsyncInclude = function() {
        e.satellite_asyncLoad && y.notify('You may be using the async installation of Satellite. In-page HTML and the "pagebottom" event will not work. Please update your Satellite installation for these features.', 5)
    }, y.hasMultipleDomains = function() {
        return y.settings.domainList.length > 1
    }, y.handleOverrides = function() {
        if (_)
            for (var e in _)
                _.hasOwnProperty(e) && (y.data[e] = _[e])
    }, y.privacyManagerParams = function() {
        var e = {};
        y.extend(e, y.settings.privacyManagement);
        var t = [];
        for (var a in y.tools) {
            var n = y.tools[a], i = n.settings;
            i && "sc" === i.engine && t.push(n)
        }
        var o = y.filter(y.map(t, function(e) {
            return e.getTrackingServer()
        }), function(e) {
            return null != e
        });
        e.adobeAnalyticsTrackingServers = o;
        for (var s = ["bannerText", "headline", "introductoryText", "customCSS"], r = 0; r < s.length; r++) {
            var c = s[r], l = e[c];
            if (l)
                if ("text" === l.type)
                    e[c] = l.value;
                else {
                    if ("data" !== l.type)
                        throw new Error("Invalid type: " + l.type);
                        e[c] = y.getVar(l.value)
                    }
            }
        return e
    }, y.prepareLoadPrivacyManager = function() {
        function t(e) {
            function t() {
                o++, o === a.length && (n(), clearTimeout(s), e())
            }
            function n() {
                y.each(a, function(e) {
                    y.unbindEvent(e.id + ".load", t)
                })
            }
            function i() {
                n(), e()
            }
            var o = 0;
            y.each(a, function(e) {
                y.bindEvent(e.id + ".load", t)
            });
            var s = setTimeout(i, 5e3)
        }
        var a = y.filter(y.values(y.tools), function(e) {
            return e.settings && "sc" === e.settings.engine
        });
        0 === a.length ? y.addEventHandler(e, "load", y.loadPrivacyManager) : t(y.loadPrivacyManager)
    }, y.loadPrivacyManager = function() {
        var e = y.basePath() + "privacy_manager.js";
        y.loadScript(e, function() {
            var e = y.privacyManager;
            e.configure(y.privacyManagerParams()), e.openIfRequired()
        })
    }, y.init = function(t) {
        if (!y.stagingLibraryOverride()) {
            y.configurationSettings = t;
            var a = t.tools;
            delete t.tools;
            for (var i in t)
                y[i] = t[i];
            y.data.customVars === n && (y.data.customVars = {}), y.data.queryParams = y.QueryParams.normal, y.handleOverrides(), y.detectBrowserInfo(), y.trackVisitorInfo && y.trackVisitorInfo(), y.loadStoredSettings(), y.checkAsyncInclude(), y.ensureCSSSelector(), y.filterRules(), y.dynamicRules = y.filter(y.rules, function(e) {
                return e.eventHandlerOnElement
            }), y.tools = y.initTools(a), y.initEventEmitters(), y.firePageLoadEvent("aftertoolinit"), y.settings.forceLowerCase && (y.data.URI = y.data.URI.toLowerCase()), y.settings.privacyManagement && y.prepareLoadPrivacyManager(), y.hasSelector && y.domReady(y.eventEmitterBackgroundTasks), y.setListeners(), y.domReady(function() {
                y.poll(function() {
                    y.backgroundTasks()
                }, y.settings.recheckEvery || 3e3)
            }), y.domReady(function() {
                y.domReadyFired=!0, y.pageBottomFired || y.pageBottom(), y.firePageLoadEvent("domready")
            }), y.addEventHandler(e, "load", function() {
                y.firePageLoadEvent("windowload")
            }), y.firePageLoadEvent("pagetop")
        }
    }, y.pageLoadPhases = ["aftertoolinit", "pagetop", "pagebottom", "domready", "windowload"], y.loadEventBefore = function(e, t) {
        return y.indexOf(y.pageLoadPhases, e) <= y.indexOf(y.pageLoadPhases, t)
    }, y.flushPendingCalls = function(e) {
        e.pending && (y.each(e.pending, function(t) {
            var a = t[0], n = t[1], i = t[2], o = t[3];
            a in e ? e[a].apply(e, [n, i].concat(o)) : e.emit ? e.emit(a, n, i, o) : y.notify("Failed to trigger " + a + " for tool " + e.id, 1)
        }), delete e.pending)
    }, y.setDebug = function(t) {
        e.localStorage && e.localStorage.setItem("sdsat_debug", t)
    }, y.detectBrowserInfo = function() {
        function e(e) {
            return function(t) {
                for (var a in e) {
                    var n = e[a], i = n.test(t);
                    if (i)
                        return a
                }
                return "Unknown"
            }
        }
        var t = e({
            OmniWeb: /OmniWeb/,
            "Opera Mini": /Opera Mini/,
            "Opera Mobile": /Opera Mobi/,
            Opera: /Opera/,
            "Mobile Safari": /Mobile(\/[0-9A-z]+)? Safari/,
            Chrome: /Chrome/,
            Firefox: /Firefox/,
            "IE Mobile": /IEMobile/,
            IE: /MSIE|Trident/,
            Safari: /Safari/
        }), a = e({
            iOS: /iPhone|iPad|iPod/,
            Blackberry: /BlackBerry/,
            "Symbian OS": /SymbOS/,
            Maemo: /Maemo/,
            Android: /Android [0-9\.]+;/,
            Linux: / Linux /,
            Unix: /FreeBSD|OpenBSD|CrOS/,
            Windows: /[\( ]Windows /,
            MacOS: /Macintosh;/
        }), n = e({
            iPhone: /iPhone/,
            iPad: /iPad/,
            iPod: /iPod/,
            Nokia: /SymbOS|Maemo/,
            "Windows Phone": /IEMobile/,
            Blackberry: /BlackBerry/,
            Android: /Android [0-9\.]+;/,
            Desktop: /.*/
        }), i = navigator.userAgent;
        y.browserInfo = {
            browser: t(i),
            os: a(i),
            deviceType: n(i)
        }
    }, y.isHttps = function() {
        return "https:" == a.location.protocol
    }, y.BaseTool = function(e) {
        this.settings = e || {}, this.forceLowerCase = y.settings.forceLowerCase, "forceLowerCase"in this.settings && (this.forceLowerCase = this.settings.forceLowerCase)
    }, y.BaseTool.prototype = {
        triggerCommand: function(e, t, a) {
            var n = this.settings || {};
            if (this.initialize && this.isQueueAvailable() && this.isQueueable(e) && a && y.loadEventBefore(a.type, n.loadOn))
                return void this.queueCommand(e, t, a);
            var i = y.preprocessArguments(e.arguments, t, a, this.forceLowerCase), o = e.command, s = this["$" + o];
            s ? s.apply(this, [t, a].concat(i)) : this.$missing$ ? this.$missing$(o, t, a, i) : y.notify("Failed to trigger " + o + " for tool " + this.id, 1)
        },
        endPLPhase: function() {},
        isQueueable: function(e) {
            return "cancelToolInit" !== e.command
        },
        isQueueAvailable: function() {
            return !this.initialized&&!this.initializing
        },
        flushQueue: function() {
            this.pending && (y.each(this.pending, function(e) {
                this.triggerCommand.apply(this, e)
            }, this), this.pending = [])
        },
        queueCommand: function(e, t, a) {
            this.pending || (this.pending = []), this.pending.push([e, t, a])
        },
        $cancelToolInit: function() {
            this._cancelToolInit=!0
        }
    }, e._satellite = y, i.orientationChange = function(t) {
        var a = 0 === e.orientation ? "portrait": "landscape";
        t.orientation = a, y.onEvent(t)
    }, y.availableEventEmitters.push(i), o.prototype.backgroundTasks = function() {
        y.each(this.rules, function(e) {
            y.cssQuery(e.selector, function(e) {
                if (e.length > 0) {
                    var t = e[0];
                    if (y.$data(t, "elementexists.seen"))
                        return;
                    y.$data(t, "elementexists.seen", !0), y.onEvent({
                        type: "elementexists",
                        target: t
                    })
                }
            })
        })
    }, y.availableEventEmitters.push(o), s.prototype = {
        backgroundTasks: function() {
            var e = this.eventHandler;
            y.each(this.rules, function(t) {
                y.cssQuery(t.selector || "video", function(t) {
                    y.each(t, function(t) {
                        y.$data(t, "videoplayed.tracked") || (y.addEventHandler(t, "timeupdate", y.throttle(e, 100)), y.$data(t, "videoplayed.tracked", !0))
                    })
                })
            })
        },
        evalRule: function(e, t) {
            var a = t.event, n = e.seekable, i = n.start(0), o = n.end(0), s = e.currentTime, r = t.event.match(/^videoplayed\(([0-9]+)([s%])\)$/);
            if (r) {
                var c = r[2], l = Number(r[1]), d = "%" === c ? function() {
                    return 100 * (s - i) / (o - i) >= l
                }
                : function() {
                    return s - i >= l
                };
                !y.$data(e, a) && d() && (y.$data(e, a, !0), y.onEvent({
                    type: a,
                    target: e
                }))
            }
        },
        onUpdateTime: function(e) {
            var t = this.rules, a = e.target;
            if (a.seekable && 0 !== a.seekable.length)
                for (var n = 0, i = t.length; i > n; n++)
                    this.evalRule(a, t[n])
        }
    }, y.availableEventEmitters.push(s), r.prototype = {
        backgroundTasks: function() {
            var e = this;
            y.each(this.rules, function(t) {
                var a = t[1], n = t[0];
                y.cssQuery(a, function(t) {
                    y.each(t, function(t) {
                        e.trackElement(t, n)
                    })
                })
            }, this)
        },
        trackElement: function(e, t) {
            var a = this, n = y.$data(e, "hover.delays");
            n ? y.contains(n, t) || n.push(t) : (y.addEventHandler(e, "mouseover", function(t) {
                a.onMouseOver(t, e)
            }), y.addEventHandler(e, "mouseout", function(t) {
                a.onMouseOut(t, e)
            }), y.$data(e, "hover.delays", [t]))
        },
        onMouseOver: function(e, t) {
            var a = e.target || e.srcElement, n = e.relatedTarget || e.fromElement, i = (t === a || y.containsElement(t, a))&&!y.containsElement(t, n);
            i && this.onMouseEnter(t)
        },
        onMouseEnter: function(e) {
            var t = y.$data(e, "hover.delays"), a = y.map(t, function(t) {
                return setTimeout(function() {
                    y.onEvent({
                        type: "hover(" + t + ")",
                        target: e
                    })
                }, t)
            });
            y.$data(e, "hover.delayTimers", a)
        },
        onMouseOut: function(e, t) {
            var a = e.target || e.srcElement, n = e.relatedTarget || e.toElement, i = (t === a || y.containsElement(t, a))&&!y.containsElement(t, n);
            i && this.onMouseLeave(t)
        },
        onMouseLeave: function(e) {
            var t = y.$data(e, "hover.delayTimers");
            t && y.each(t, function(e) {
                clearTimeout(e)
            })
        }
    }, y.availableEventEmitters.push(r), c.prototype = {
        initialize: function() {
            var e = this.twttr;
            e && "function" == typeof e.ready && e.ready(y.bind(this.bind, this))
        },
        bind: function() {
            this.twttr.events.bind("tweet", function(e) {
                e && (y.notify("tracking a tweet button", 1), y.onEvent({
                    type: "twitter.tweet",
                    target: a
                }))
            })
        }
    }, y.availableEventEmitters.push(c), l.prototype = {
        initialize: function() {
            return this.FB = this.FB || e.FB, this.FB && this.FB.Event && this.FB.Event.subscribe ? (this.bind(), !0) : void 0
        },
        bind: function() {
            this.FB.Event.subscribe("edge.create", function() {
                y.notify("tracking a facebook like", 1), y.onEvent({
                    type: "facebook.like",
                    target: a
                })
            }), this.FB.Event.subscribe("edge.remove", function() {
                y.notify("tracking a facebook unlike", 1), y.onEvent({
                    type: "facebook.unlike",
                    target: a
                })
            }), this.FB.Event.subscribe("message.send", function() {
                y.notify("tracking a facebook share", 1), y.onEvent({
                    type: "facebook.send",
                    target: a
                })
            })
        }
    }, y.availableEventEmitters.push(l), d.offset = function(t) {
        var n;
        try {
            n = t.getBoundingClientRect()
        } catch (i) {}
        var o = a, s = o.documentElement, r = o.body, c = e, l = s.clientTop || r.clientTop || 0, d = s.clientLeft || r.clientLeft || 0, f = c.pageYOffset || s.scrollTop || r.scrollTop, u = c.pageXOffset || s.scrollLeft || r.scrollLeft, h = n.top + f - l, b = n.left + u - d;
        return {
            top: h,
            left: b
        }
    }, d.getViewportHeight = function() {
        var t = e.innerHeight, n = a.compatMode;
        return n && (t = "CSS1Compat" == n ? a.documentElement.clientHeight : a.body.clientHeight), t
    }, d.getScrollTop = function() {
        return a.documentElement.scrollTop ? a.documentElement.scrollTop : a.body.scrollTop
    }, d.prototype = {
        backgroundTasks: function() {
            var e = this.elements;
            y.each(this.rules, function(t) {
                y.cssQuery(t.selector, function(a) {
                    var n = 0;
                    y.each(a, function(t) {
                        y.contains(e, t) || (e.push(t), n++)
                    }), n && y.notify(t.selector + " added " + n + " elements.", 1)
                })
            }), this.track()
        },
        elementIsInView: function(e) {
            var t = d.getViewportHeight(), a = d.getScrollTop(), n = d.offset(e).top, i = e.offsetHeight;
            return !(a > n + i || n > a + t)
        },
        checkInView: function(e, t) {
            var a = y.$data(e, "inview");
            if (this.elementIsInView(e)) {
                a || y.$data(e, "inview", !0);
                var n = this;
                this.processRules(e, function(a, i, o) {
                    if (t ||!a.inviewDelay)
                        y.$data(e, i, !0), y.onEvent({
                            type: "inview",
                            target: e,
                            inviewDelay: a.inviewDelay
                        });
                    else if (a.inviewDelay) {
                        var s = y.$data(e, o);
                        s && clearTimeout(s), s = setTimeout(function() {
                            n.checkInView(e, !0)
                        }, a.inviewDelay), y.$data(e, o, s)
                    }
                })
            } else 
                a && y.$data(e, "inview", !1), this.processRules(e, function(t, a, n) {
                    var i = y.$data(e, n);
                    i && clearTimeout(i)
                })
        },
        track: function() {
            y.each(this.elements, function(e) {
                this.checkInView(e)
            }, this)
        },
        processRules: function(e, t) {
            y.each(this.rules, function(a, n) {
                var i = a.inviewDelay ? "viewed_" + a.inviewDelay: "viewed", o = "inview_timeout_id_" + n;
                y.$data(e, i) || y.matchesCss(a.selector, e) && t(a, i, o)
            })
        }
    }, y.availableEventEmitters.push(d), y.extend(f.prototype, {
        initialize: function() {
            var e, t = this.settings;
            y.notify("Visitor ID: Initializing tool", 1), e = this.createInstance(t.mcOrgId, t.namespace, t.initVars), null !== e && (t.customerIDs && this.applyCustomerIDs(e, t.customerIDs), t.autoRequest && e.getMarketingCloudVisitorID(), this.instance = e)
        },
        createInstance: function(e, t, a) {
            if (!y.isString(e))
                return y.notify('Visitor ID: Cannot create instance using mcOrgId: "' + e + '"', 4), null;
            y.notify('Visitor ID: Create instance using mcOrgId: "' + e + '"', 1);
            var n = new Visitor(e, t);
            return this.applyInitVars(n, a), n
        },
        getInstance: function() {
            return this.instance
        },
        applyInitVars: function(e, t) {
            y.isObject(t)!==!1 && (t = this.parseValues(t), y.extend(e, t), y.notify("Visitor ID: Set variables: " + y.stringify(t), 1))
        },
        applyCustomerIDs: function(e, t) {
            y.isObject(t)!==!1 && (t = this.parseValues(t), e.setCustomerIDs(t), y.notify("Visitor ID: Set Customer IDs: " + y.stringify(t), 1))
        },
        parseValues: function(e) {
            var t = {};
            for (var a in e)
                e.hasOwnProperty(a) && (t[a] = y.replace(e[a]));
            return t
        }
    }), y.availableTools.visitor_id = f, y.inherit(u, y.BaseTool), y.extend(u.prototype, {
        name: "GA",
        initialize: function() {
            var t = this.settings, a = e._gaq, n = t.initCommands || [], i = t.customInit;
            if (a || (_gaq = []), this.isSuppressed())
                y.notify("GA: page code not loaded(suppressed).", 1);
            else {
                if (!a&&!u.scriptLoaded) {
                    var o = y.isHttps(), s = (o ? "https://ssl" : "http://www") + ".google-analytics.com/ga.js";
                    t.url && (s = o ? t.url.https : t.url.http), y.loadScript(s), u.scriptLoaded=!0, y.notify("GA: page code loaded.", 1)
                }
                {
                    var r = (t.domain, t.trackerName), c = k.allowLinker(), l = t.account;
                    y.settings.domainList || []
                }
                _gaq.push([this.cmd("setAccount"), l]), c && _gaq.push([this.cmd("setAllowLinker"), c]), _gaq.push([this.cmd("setDomainName"), k.cookieDomain()]), y.each(n, function(e) {
                    var t = [this.cmd(e[0])].concat(e.slice(1));
                    _gaq.push(t)
                }, this), i && (this.suppressInitialPageView=!1 === i(_gaq, r)), t.pageName && this.$overrideInitialPageView(null, null, t.pageName)
            }
            this.initialized=!0, y.fireEvent(this.id + ".configure", _gaq, r)
        },
        isSuppressed: function() {
            return this._cancelToolInit || this.settings.initTool===!1
        },
        tracker: function() {
            return this.settings.trackerName
        },
        cmd: function(e) {
            var t = this.tracker();
            return t ? t + "._" + e : "_" + e
        },
        $overrideInitialPageView: function(e, t, a) {
            this.urlOverride = a
        },
        trackInitialPageView: function() {
            if (!this.isSuppressed()&&!this.suppressInitialPageView)
                if (this.urlOverride) {
                    var e = y.preprocessArguments([this.urlOverride], location, null, this.forceLowerCase);
                    this.$missing$("trackPageview", null, null, e)
                } else 
                    this.$missing$("trackPageview")
        },
        endPLPhase: function(e) {
            var t = this.settings.loadOn;
            e === t && (y.notify("GA: Initializing at " + e, 1), this.initialize(), this.flushQueue(), this.trackInitialPageView())
        },
        call: function(e, t, a, n) {
            if (!this._cancelToolInit) {
                var i = (this.settings, this.tracker()), o = this.cmd(e), n = n ? [o].concat(n): [o];
                _gaq.push(n), i ? y.notify("GA: sent command " + e + " to tracker " + i + (n.length > 1 ? " with parameters [" + n.slice(1).join(", ") + "]" : "") + ".", 1) : y.notify("GA: sent command " + e + (n.length > 1 ? " with parameters [" + n.slice(1).join(", ") + "]" : "") + ".", 1)
            }
        },
        $missing$: function(e, t, a, n) {
            this.call(e, t, a, n)
        },
        $postTransaction: function(t, a, n) {
            var i = y.data.customVars.transaction = e[n];
            this.call("addTrans", t, a, [i.orderID, i.affiliation, i.total, i.tax, i.shipping, i.city, i.state, i.country]), y.each(i.items, function(e) {
                this.call("addItem", t, a, [e.orderID, e.sku, e.product, e.category, e.unitPrice, e.quantity])
            }, this), this.call("trackTrans", t, a)
        },
        delayLink: function(e, t) {
            var a = this;
            if (k.allowLinker() && e.hostname.match(this.settings.linkerDomains)&&!y.isSubdomainOf(e.hostname, location.hostname)) {
                y.preventDefault(t);
                var n = y.settings.linkDelay || 100;
                setTimeout(function() {
                    a.call("link", e, t, [e.href])
                }, n)
            }
        },
        popupLink: function(t, a) {
            if (e._gat) {
                y.preventDefault(a);
                var n = this.settings.account, i = e._gat._createTracker(n), o = i._getLinkerUrl(t.href);
                e.open(o)
            }
        },
        $link: function(e, t) {
            "_blank" === e.getAttribute("target") ? this.popupLink(e, t) : this.delayLink(e, t)
        },
        $trackEvent: function(e, t) {
            var a = Array.prototype.slice.call(arguments, 2);
            if (a.length >= 4 && null != a[3]) {
                var n = parseInt(a[3], 10);
                y.isNaN(n) && (n = 1), a[3] = n
            }
            this.call("trackEvent", e, t, a)
        }
    }), y.availableTools.ga = u, y.inherit(h, y.BaseTool), y.extend(h.prototype, {
        name: "GAUniversal",
        isPageCodeLoadSuppressed: function() {
            return this.settings.initTool===!1 || this._cancelToolInit===!0
        },
        isCallSuppressed: function() {
            return this._cancelToolInit===!0
        },
        endPLPhase: function(e) {
            var t = this.settings, a = t.loadOn;
            if (this.isPageCodeLoadSuppressed())
                return void y.notify("GAU: Page code not loaded(suppressed).", 1);
            if (e === a) {
                if (y.notify("GAU: Initializing at " + e, 1), this.initialize(), this.flushQueue(), this.suppressInitialPageView)
                    return;
                this.call("send", "pageview")
            }
        },
        initialize: function() {
            var t = "ga";
            e[t] = e[t] || this.createGAObject(), e.GoogleAnalyticsObject = t, y.notify("GAU: Page code loaded.", 1), y.loadScriptOnce(this.getToolUrl());
            var a = this.settings;
            if (k.allowLinker() && a.allowLinker!==!1 ? this.createAccountForLinker() : this.createAccount(), this.executeInitCommands(), a.customInit) {
                var n = a.customInit, i = n(e[t], this.getTrackerName());
                i===!1 && (this.suppressInitialPageView=!0)
            }
            this.initialized=!0
        },
        getTrackerName: function() {
            return this.settings.trackerSettings.name || ""
        },
        createGAObject: function() {
            var e = function() {
                e.q.push(arguments)
            };
            return e.q = [], e.l = 1 * new Date, e
        },
        create: function(e) {
            var t = this.settings.trackerSettings;
            t.cookieDomain || (t.cookieDomain = k.cookieDomain()), y.extend(t, e || {}), this.call("create", t)
        },
        createAccount: function() {
            this.create()
        },
        createAccountForLinker: function() {
            var e = {};
            k.allowLinker() && (e.allowLinker=!0), this.create(e), this.call("require", "linker"), this.call("linker:autoLink", this.autoLinkDomains(), !1, !0)
        },
        autoLinkDomains: function() {
            var e = location.hostname;
            return y.filter(y.settings.domainList, function(t) {
                return t !== e
            })
        },
        executeInitCommands: function() {
            var e = this.settings;
            e.initCommands && y.each(e.initCommands, function(e) {
                this.call.apply(this, e)
            }, this)
        },
        call: function() {
            return "function" != typeof ga ? void y.notify("GA Universal function not found!", 4) : void(this.isCallSuppressed() || (arguments[0] = this.cmd(arguments[0]), this.log(y.toArray(arguments)), ga.apply(e, arguments)))
        },
        $missing$: function(e, t, a, n) {
            n = n || [], n = [e].concat(n), this.call.apply(this, n)
        },
        getToolUrl: function() {
            var e = this.settings, t = y.isHttps();
            return e.url ? t ? e.url.https : e.url.http : (t ? "https://ssl" : "http://www") + ".google-analytics.com/analytics.js"
        },
        cmd: function(e) {
            var t = ["send", "set", "get"], a = this.getTrackerName();
            return a&&-1 !== y.indexOf(t, e) ? a + "." + e : e
        },
        log: function(e) {
            var t = e[0], a = this.getTrackerName() || "default", n = "GA Universal: sent command " + t + " to tracker " + a;
            if (e.length > 1) {
                {
                    y.stringify(e.slice(1))
                }
                n += " with parameters " + y.stringify(e.slice(1))
            }
            n += ".", y.notify(n, 1)
        }
    }), y.availableTools.ga_universal = h;
    var k = {
        allowLinker: function() {
            return y.hasMultipleDomains()
        },
        cookieDomain: function() {
            var t = y.settings.domainList, a = y.find(t, function(t) {
                var a = e.location.hostname;
                return y.equalsIgnoreCase(a.slice(a.length - t.length), t)
            }), n = a ? "." + a: "auto";
            return n
        }
    };
    y.inherit(b, y.BaseTool), y.extend(b.prototype, {
        name: "SC",
        initialize: function(t) {
            if (!this._cancelToolInit)
                if (this.settings.initVars = this.substituteVariables(this.settings.initVars, {
                    type: t
                }), this.settings.initTool!==!1) {
                    var a = this.settings.sCodeURL || y.basePath() + "s_code.js";
                    "object" == typeof a && (a = "https:" === e.location.protocol ? a.https : a.http), a.match(/^https?:/) || (a = y.basePath() + a), this.settings.initVars && this.$setVars(null, null, this.settings.initVars), y.loadScript(a, y.bind(this.onSCodeLoaded, this)), this.initializing=!0
                } else 
                    this.initializing=!0, this.pollForSC()
        },
        onSCodeLoaded: function() {
            this.initialized=!0, this.initializing=!1, y.notify("Adobe Analytics: loaded.", 1), y.fireEvent(this.id + ".load", this.getS()), this.flushQueueExceptTrackLink(), this.sendBeacon(), this.flushQueue()
        },
        pollForSC: function() {
            y.poll(y.bind(function() {
                return "function" == typeof e.s_gi ? (this.initialized=!0, this.initializing=!1, y.notify("Adobe Analytics: loaded (manual).", 1), y.fireEvent(this.id + ".load", this.getS()), this.flushQueue(), !0) : void 0
            }, this))
        },
        flushQueueExceptTrackLink: function() {
            if (this.pending) {
                for (var e = [], t = 0; t < this.pending.length; t++) {
                    var a = this.pending[t], n = a[0];
                    "trackLink" === n.command ? e.push(a) : this.triggerCommand.apply(this, a)
                }
                this.pending = e
            }
        },
        isQueueAvailable: function() {
            return !this.initialized
        },
        substituteVariables: function(e, t) {
            var a = {};
            for (var n in e) {
                var i = e[n];
                a[n] = y.replace(i, location, t)
            }
            return a
        },
        endPLPhase: function(e) {
            var t = this.settings.loadOn;
            e === t && this.initialize(e)
        },
        $setVars: function(e, t, a) {
            for (var n in a) {
                var i = a[n];
                "function" == typeof i && (i = i()), this.varBindings[n] = i
            }
            y.notify("Adobe Analytics: set variables.", 2)
        },
        $customSetup: function(e, t, a) {
            this.customSetupFuns.push(function(n) {
                a.call(e, t, n)
            })
        },
        getAccount: function(t) {
            return e.s_account ? e.s_account : t && this.settings.accountByHost ? this.settings.accountByHost[t] || this.settings.account : this.settings.account
        },
        isValidSCInstance: function(e) {
            return !!e && "function" == typeof e.t && "function" == typeof e.tl
        },
        getS: function(t, a) {
            var n = a && a.hostname || e.location.hostname, i = this.concatWithToolVarBindings(a && a.setVars || this.varBindings), o = a && a.addEvent || this.events, s = this.getAccount(n), r = e.s_gi;
            if (!r)
                return null;
            if (this.isValidSCInstance(t) || (t = null), !s&&!t)
                return y.notify("Adobe Analytics: tracker not initialized because account was not found", 1), null;
            var t = t || r(s), c = "D" + y.appVersion;
            "undefined" != typeof t.tagContainerMarker ? t.tagContainerMarker = c : "string" == typeof t.version && t.version.substring(t.version.length - 5) !== "-" + c && (t.version += "-" + c), t.sa && this.settings.skipSetAccount!==!0 && this.settings.initTool!==!1 && t.sa(this.settings.account), this.applyVarBindingsOnTracker(t, i), o.length > 0 && (t.events = o.join(","));
            var l = y.getVisitorId();
            return l && (t.visitor = y.getVisitorId()), t
        },
        concatWithToolVarBindings: function(e) {
            var t = this.settings.initVars || {};
            return y.map(["trackingServer", "trackingServerSecure"], function(a) {
                t[a]&&!e[a] && (e[a] = t[a])
            }), e
        },
        applyVarBindingsOnTracker: function(e, t) {
            for (var a in t)
                e[a] = t[a]
        },
        clearVarBindings: function() {
            this.varBindings = {}
        },
        clearCustomSetup: function() {
            this.customSetupFuns = []
        },
        sendBeacon: function() {
            var t = this.getS(e[this.settings.renameS || "s"]);
            return t ? this.settings.customInit && this.settings.customInit(t)===!1 ? void y.notify("Adobe Analytics: custom init suppressed beacon", 1) : (this.settings.executeCustomPageCodeFirst && this.applyVarBindingsOnTracker(t, this.varBindings), this.executeCustomSetupFuns(t), t.t(), this.clearVarBindings(), this.clearCustomSetup(), void y.notify("Adobe Analytics: tracked page view", 1)) : void y.notify("Adobe Analytics: page code not loaded", 1)
        },
        executeCustomSetupFuns: function(t) {
            y.each(this.customSetupFuns, function(a) {
                a.call(e, t)
            })
        },
        $trackLink: function(e, t, a) {
            a = a || {};
            var n = a.type, i = a.linkName;
            !i && e && e.nodeName && "a" === e.nodeName.toLowerCase() && (i = e.innerHTML), i || (i = "link clicked");
            var o = a && a.setVars, s = a && a.addEvent || [], r = this.getS(null, {
                setVars: o,
                addEvent: s
            });
            if (!r)
                return void y.notify("Adobe Analytics: page code not loaded", 1);
            var c = r.linkTrackVars, l = r.linkTrackEvents, d = this.definedVarNames(o);
            a && a.customSetup && a.customSetup.call(e, t, r), s.length > 0 && d.push("events"), r.products && d.push("products"), d = this.mergeTrackLinkVars(r.linkTrackVars, d), s = this.mergeTrackLinkVars(r.linkTrackEvents, s), r.linkTrackVars = this.getCustomLinkVarsList(d);
            var f = y.map(s, function(e) {
                return e.split(":")[0]
            });
            r.linkTrackEvents = this.getCustomLinkVarsList(f), r.tl(!0, n || "o", i), y.notify(["Adobe Analytics: tracked link ", "using: linkTrackVars=", y.stringify(r.linkTrackVars), "; linkTrackEvents=", y.stringify(r.linkTrackEvents)].join(""), 1), r.linkTrackVars = c, r.linkTrackEvents = l
        },
        mergeTrackLinkVars: function(e, t) {
            return e && (t = e.split(",").concat(t)), t
        },
        getCustomLinkVarsList: function(e) {
            var t = y.indexOf(e, "None");
            return t>-1 && e.length > 1 && e.splice(t, 1), e.join(",")
        },
        definedVarNames: function(e) {
            e = e || this.varBindings;
            var t = [];
            for (var a in e)
                /^(eVar[0 - 9] + ) | (prop[0 - 9] + )$ / .test(a) && t.push(a);
            return t
        },
        getTrackingServer: function() {
            var t = this, a = t.getS();
            if (a && a.trackingServer)
                return a.trackingServer;
            var n = t.getAccount(e.location.hostname);
            if (!n)
                return null;
            var i, o, s, r = "", c = a && a.dc;
            return i = n, o = i.indexOf(","), o >= 0 && (i = i.gb(0, o)), i = i.replace(/[^A-Za-z0-9]/g, ""), r || (r = "2o7.net"), c = c ? ("" + c).toLowerCase() : "d1", "2o7.net" == r && ("d1" == c ? c = "112" : "d2" == c && (c = "122"), s = ""), o = i + "." + c + "." + s + r
        },
        $trackPageView: function(e, t, a) {
            var n = a && a.setVars, i = a && a.addEvent || [], o = this.getS(null, {
                setVars: n,
                addEvent: i
            });
            return o ? (o.linkTrackVars = "", o.linkTrackEvents = "", this.executeCustomSetupFuns(o), a && a.customSetup && a.customSetup.call(e, t, o), o.t(), this.clearVarBindings(), this.clearCustomSetup(), void y.notify("Adobe Analytics: tracked page view", 1)) : void y.notify("Adobe Analytics: page code not loaded", 1)
        },
        $postTransaction: function(t, a, n) {
            var i = y.data.transaction = e[n], o = this.varBindings, s = this.settings.fieldVarMapping;
            if (y.each(i.items, function(e) {
                this.products.push(e)
            }, this), o.products = y.map(this.products, function(e) {
                var t = [];
                if (s && s.item)
                    for (var a in s.item) {
                        var n = s.item[a];
                        t.push(n + "=" + e[a]), "event" === n.substring(0, 5) && this.events.push(n)
                    }
                var i = ["", e.product, e.quantity, e.unitPrice * e.quantity];
                return t.length > 0 && i.push(t.join("|")), i.join(";")
            }, this).join(","), s && s.transaction) {
                var r = [];
                for (var c in s.transaction) {
                    var n = s.transaction[c];
                    r.push(n + "=" + i[c]), "event" === n.substring(0, 5) && this.events.push(n)
                }
                o.products.length > 0 && (o.products += ","), o.products += ";;;;" + r.join("|")
            }
        },
        $addEvent: function() {
            for (var e = 2, t = arguments.length; t > e; e++)
                this.events.push(arguments[e])
        },
        $addProduct: function() {
            for (var e = 2, t = arguments.length; t > e; e++)
                this.products.push(arguments[e])
        }
    }), y.availableTools.sc = b, y.inherit(p, y.BaseTool), y.extend(p.prototype, {
        name: "Default",
        $loadIframe: function(t, a, n) {
            var i = n.pages, o = n.loadOn, s = y.bind(function() {
                y.each(i, function(e) {
                    this.loadIframe(t, a, e)
                }, this)
            }, this);
            o || s(), "domready" === o && y.domReady(s), "load" === o && y.addEventHandler(e, "load", s)
        },
        loadIframe: function(e, t, n) {
            var i = a.createElement("iframe");
            i.style.display = "none";
            var o = y.data.host, s = n.data, r = this.scriptURL(n.src), c = y.searchVariables(s, e, t);
            o && (r = y.basePath() + r), r += c, i.src = r;
            var l = a.getElementsByTagName("body")[0];
            l ? l.appendChild(i) : y.domReady(function() {
                a.getElementsByTagName("body")[0].appendChild(i)
            })
        },
        scriptURL: function(e) {
            var t = y.settings.scriptDir || "";
            return t + e
        },
        $loadScript: function(t, a, n) {
            var i = n.scripts, o = n.sequential, s = n.loadOn, r = y.bind(function() {
                o ? this.loadScripts(t, a, i) : y.each(i, function(e) {
                    this.loadScripts(t, a, [e])
                }, this)
            }, this);
            s ? "domready" === s ? y.domReady(r) : "load" === s && y.addEventHandler(e, "load", r) : r()
        },
        loadScripts: function(e, t, a) {
            function n() {
                if (o.length > 0 && i) {
                    var c = o.shift();
                    c.call(e, t, s)
                }
                var l = a.shift();
                if (l) {
                    var d = y.data.host, f = r.scriptURL(l.src);
                    d && (f = y.basePath() + f), y.loadScript(f, n), i = l
                }
            }
            try {
                var i, a = a.slice(0), o = this.asyncScriptCallbackQueue, s = t.target || t.srcElement, r = this
            } catch (c) {
                console.error("scripts is", y.stringify(a))
            }
            n()
        },
        $loadBlockingScript: function(e, t, a) {
            var n = a.scripts, i = (a.loadOn, y.bind(function() {
                y.each(n, function(a) {
                    this.loadBlockingScript(e, t, a)
                }, this)
            }, this));
            i()
        },
        loadBlockingScript: function(e, t, a) {
            var n = this.scriptURL(a.src), i = y.data.host, o = t.target || t.srcElement;
            i && (n = y.basePath() + n), this.argsForBlockingScripts.push([e, t, o]), y.loadScriptSync(n)
        },
        pushAsyncScript: function(e) {
            this.asyncScriptCallbackQueue.push(e)
        },
        pushBlockingScript: function(e) {
            var t = this.argsForBlockingScripts.shift(), a = t[0];
            e.apply(a, t.slice(1))
        },
        $writeHTML: function(e, t) {
            if (y.domReadyFired)
                return void y.notify("Command writeHTML failed. You may be using an async installation of Satellite.", 1);
            if ("pagebottom" !== t.type && "pagetop" !== t.type)
                return void y.notify("You can only use writeHTML on the `pagetop` and `pagebottom` events.", 1);
            for (var n = 2, i = arguments.length; i > n; n++) {
                var o = arguments[n].html;
                o = y.replace(o, e, t), a.write(o)
            }
        },
        linkNeedsDelayActivate: function(t, a) {
            a = a || e;
            var n = t.tagName, i = t.getAttribute("target"), o = t.getAttribute("href");
            return n && "a" !== n.toLowerCase()?!1 : o ? i ? "_blank" === i?!1 : "_top" === i ? a.top === a : "_parent" === i?!1 : "_self" === i?!0 : a.name ? i === a.name : !0 : !0 : !1
        },
        $delayActivateLink: function(e, t) {
            if (this.linkNeedsDelayActivate(e)) {
                y.preventDefault(t);
                var a = y.settings.linkDelay || 100;
                setTimeout(function() {
                    y.setLocation(e.href)
                }, a)
            }
        },
        isQueueable: function(e) {
            return "writeHTML" !== e.command
        }
    }), y.availableTools["default"] = p, y.inherit(g, y.BaseTool), y.extend(g.prototype, {
        name: "tnt",
        endPLPhase: function(e) {
            "aftertoolinit" === e && this.initialize()
        },
        initialize: function() {
            y.notify("Test & Target: Initializing", 1), this.initializeTargetPageParams(), this.load()
        },
        initializeTargetPageParams: function() {
            e.targetPageParams && this.updateTargetPageParams(this.parseTargetPageParamsResult(e.targetPageParams())), this.updateTargetPageParams(this.settings.pageParams), this.setTargetPageParamsFunction()
        },
        load: function() {
            var e = this.getMboxURL(this.settings.mboxURL);
            this.settings.initTool!==!1 ? this.settings.loadSync ? (y.loadScriptSync(e), this.onScriptLoaded()) : (y.loadScript(e, y.bind(this.onScriptLoaded, this)), this.initializing=!0) : this.initialized=!0
        },
        getMboxURL: function(t) {
            var a = t;
            return y.isObject(t) && (a = "https:" === e.location.protocol ? t.https : t.http), a.match(/^https?:/) ? a : y.basePath() + a
        },
        onScriptLoaded: function() {
            y.notify("Test & Target: loaded.", 1), this.flushQueue(), this.initialized=!0, this.initializing=!1
        },
        $addMbox: function(e, t, a) {
            var n = a.mboxGoesAround, i = n + "{visibility: hidden;}", o = this.appendStyle(i);
            n in this.styleElements || (this.styleElements[n] = o), this.initialized ? this.$addMBoxStep2(null, null, a) : this.initializing && this.queueCommand({
                command: "addMBoxStep2",
                arguments: [a]
            }, e, t)
        },
        $addMBoxStep2: function(t, n, i) {
            var o = this.generateID(), s = this;
            y.addEventHandler(e, "load", y.bind(function() {
                y.cssQuery(i.mboxGoesAround, function(t) {
                    var n = t[0];
                    if (n) {
                        var r = a.createElement("div");
                        r.id = o, n.parentNode.replaceChild(r, n), r.appendChild(n), e.mboxDefine(o, i.mboxName);
                        var c = [i.mboxName];
                        i.arguments && (c = c.concat(i.arguments)), e.mboxUpdate.apply(null, c), s.reappearWhenCallComesBack(n, o, i.timeout, i)
                    }
                })
            }, this)), this.lastMboxID = o
        },
        $addTargetPageParams: function(e, t, a) {
            this.updateTargetPageParams(a)
        },
        generateID: function() {
            var e = "_sdsat_mbox_" + String(Math.random()).substring(2) + "_";
            return e
        },
        appendStyle: function(e) {
            var t = a.getElementsByTagName("head")[0], n = a.createElement("style");
            return n.type = "text/css", n.styleSheet ? n.styleSheet.cssText = e : n.appendChild(a.createTextNode(e)), t.appendChild(n), n
        },
        reappearWhenCallComesBack: function(e, t, a, n) {
            function i() {
                var e = o.styleElements[n.mboxGoesAround];
                e && (e.parentNode.removeChild(e), delete o.styleElements[n.mboxGoesAround])
            }
            var o = this;
            y.cssQuery('script[src*="omtrdc.net"]', function(e) {
                var t = e[0];
                if (t) {
                    y.scriptOnLoad(t.src, t, function() {
                        y.notify("Test & Target: request complete", 1), i(), clearTimeout(n)
                    });
                    var n = setTimeout(function() {
                        y.notify("Test & Target: bailing after " + a + "ms", 1), i()
                    }, a)
                } else 
                    y.notify("Test & Target: failed to find T&T ajax call, bailing", 1), i()
            })
        },
        updateTargetPageParams: function(e) {
            var t = {};
            for (var a in e)
                e.hasOwnProperty(a) && (t[y.replace(a)] = y.replace(e[a]));
            y.extend(this.targetPageParamsStore, t)
        },
        getTargetPageParams: function() {
            return this.targetPageParamsStore
        },
        setTargetPageParamsFunction: function() {
            e.targetPageParams = y.bind(this.getTargetPageParams, this)
        },
        parseTargetPageParamsResult: function(e) {
            var t = e;
            return y.isArray(e) && (e = e.join("&")), y.isString(e) && (t = y.parseQueryParams(e)), t
        }
    }), y.availableTools.tnt = g, y.inherit(m, y.BaseTool), y.extend(m.prototype, {
        initialize: function() {
            var e = this.settings;
            if (this.settings.initTool!==!1) {
                var t = e.url;
                t = "string" == typeof t ? y.basePath() + t : y.isHttps() ? t.https : t.http, y.loadScript(t, y.bind(this.onLoad, this)), this.initializing=!0
            } else 
                this.initialized=!0
        },
        isQueueAvailable: function() {
            return !this.initialized
        },
        onLoad: function() {
            this.initialized=!0, this.initializing=!1, this.settings.initialBeacon && this.settings.initialBeacon(), this.flushQueue()
        },
        endPLPhase: function(e) {
            var t = this.settings.loadOn;
            e === t && (y.notify(this.name + ": Initializing at " + e, 1), this.initialize())
        },
        $fire: function(e, t, a) {
            return this.initializing ? void this.queueCommand({
                command : "fire", arguments : [a]
            },
            e,
            t): void a.call(this.settings,
            e,
            t)
        }
    }), y.availableTools.am = m, y.availableTools.adlens = m, y.ecommerce = {
        addItem: function() {
            var e = [].slice.call(arguments);
            y.onEvent({
                type: "ecommerce.additem",
                target: e
            })
        },
        addTrans: function() {
            var e = [].slice.call(arguments);
            y.data.saleData.sale = {
                orderId: e[0],
                revenue: e[2]
            }, y.onEvent({
                type: "ecommerce.addtrans",
                target: e
            })
        },
        trackTrans: function() {
            y.onEvent({
                type: "ecommerce.tracktrans",
                target: []
            })
        }
    }, _satellite.init({
        tools: {
            "896d299d710b5a7ec5a0bddeef3248a094c91d33": {
                engine: "am",
                loadOn: "pagetop",
                name: "AudienceManager",
                initTool: !0,
                url: "659ec8ada5450db95675e43beaaae92399591a11/dil-contents-896d299d710b5a7ec5a0bddeef3248a094c91d33.js"
            },
            ba1811d4769e557169de925c6cbb9b6b: {
                engine: "sc",
                loadOn: "pagebottom",
                account: "adbadobenonacdcprod",
                euCookie: !1,
                sCodeURL: "659ec8ada5450db95675e43beaaae92399591a11/s-code-contents-b723f21249faac1929a77742fdda1049c35c4018.js",
                renameS: "s_adbadobenonacdc",
                initVars: {
                    trackInlineStats: !0,
                    trackDownloadLinks: !0,
                    trackExternalLinks: !0,
                    linkLeaveQueryString: !1,
                    dynamicVariablePrefix: "D="
                },
                skipSetAccount: !0,
                customInit: function() {
                    var e = _adobe_getProducts();
                    e && (s_adbadobenonacdc.prop2 = e, s_adbadobenonacdc.products = ";" + e, s_adbadobenonacdc.events = _satellite._apl(s_adbadobenonacdc.events, "prodView,event3", ",", 2)), $("li[data-dom-lobby-view=marketing-cloud]:eq(1)").click(function() {
                        - 1 == $("li[data-dom-lobby-view=marketing-cloud]:eq(1)").index(".active") && (s_adbadobenonacdc.pageName = "adobe.com:HomePage:onClick_MCFooterButton", s_adbadobenonacdc.t())
                    }), $("li[data-dom-lobby-view=creative-cloud]:eq(1)").click(function() {
                        - 1 == $("li[data-dom-lobby-view=creative-cloud]:eq(1)").index(".active") && (s_adbadobenonacdc.pageName = "adobe.com:HomePage:onClick_CCFooterButton", s_adbadobenonacdc.t())
                    }), $("li[data-dom-lobby-view=acrobat]:eq(1)").click(function() {
                        - 1 == $("li[data-dom-lobby-view=acrobat]:eq(1)").index(".active") && (s_adbadobenonacdc.pageName = "adobe.com:HomePage:onClick_DSFooterButton", s_adbadobenonacdc.t())
                    })
                }
            }
        },
        pageLoadRules: [{
            name: "adbreimaginetest - Page Load",
            trigger: [{
                tool: ["ba1811d4769e557169de925c6cbb9b6b"],
                command: "setVars",
                arguments: [{
                    eVar16: "D=c12",
                    eVar17: "%adbadobenonacdc_eVar17_Campaign_Stacking%",
                    eVar18: "%adbadobenonacdc_eVar18_New_Repeat_Visitor%",
                    eVar22: "%adbadobenonacdc_eVar22_Time_Parting_Day%",
                    eVar28: "%adbadobenonacdc_eVar28_Page_URL%",
                    eVar36: "%adbadobenonacdc_eVar36_Affiliate_ID%",
                    eVar4: "%adbadobenonacdc_eVar4_OnSite_Campaigns_Dig_Mkt%",
                    eVar43: "%adbadobenonacdc_eVar43_Cross_Channel_Targeting%",
                    eVar5: "%adbadobenonacdc_eVar5_OnSite_Campagins_Dig_Media_promoid%",
                    eVar6: "%adbadobenonacdc_eVar6_External_Campaigns_365_expire%",
                    eVar7: "%adbadobenonacdc_eVar7_Content_Dig_Mkt%",
                    prop1: "%adbadobenonacdc_prop1_Page_Type_Template_Name%",
                    prop12: "%adbadobenonacdc_prop12_Previous_Page_Value%",
                    prop13: "100",
                    prop25: "%isCookie_Renga%",
                    prop3: "%adbadobenonacdc_prop3_Site_Subdomain%",
                    prop4: "%adbadobenonacdc_prop4_Language_Locale%",
                    prop5: "%adbadobenonacdc_prop5_Page_Name_Granular%",
                    server: "%adbadobenonacdc_server%",
                    pageName: "%adbadobenonacdc_pageName_Page_Name%",
                    channel: "%adbadobenonacdc_channel_Site_Section%",
                    campaign: "D=v6"
                }
                ]
            }, {
                tool: ["ba1811d4769e557169de925c6cbb9b6b"],
                command: "customSetup",
                arguments: [function() {
                    var e = a.title;
                    - 1 != e.indexOf("page cannot be found") && (s_adobe.pageType = "errorPage");
                    var t = _adobe_getProducts();
                    t && (s_adbadobenonacdc.prop2 = t, s_adbadobenonacdc.products = ";" + t, s_adbadobenonacdc.events = _satellite._apl(s_adbadobenonacdc.events, "prodView,event3", ",", 2))
                }
                ]
            }, {
                command: "writeHTML",
                arguments: [{
                    html: "<script type=\"text/javascript\">\n   \"use strict\";\n    \n    function _adobe_getProducts () {\n        \"use strict\";\n        \n        var pathname = window.location.pathname,\n            products = '',\n            categoryPortion = '',\n            productPortion = '',\n            pathnameSplit = [],\n            countryUrls = {\n                'africa': 1,\n                'at': 1,\n                'au': 1,\n                'be_en': 1,\n                'be_fr': 1,\n                'be_nl': 1,\n                'bg': 1,\n                'br': 1,\n                'ca': 1,\n                'ca_fr': 1,\n                'ch_de': 1,\n                'ch_fr': 1,\n                'ch_it': 1,\n                'cin': 1,\n                'cn': 1,\n                'cz': 1,\n                'de': 1,\n                'dk': 1,\n                'ee': 1,\n                'eeurope': 1,\n                'es': 1,\n                'fi': 1,\n                'fr': 1,\n                'hk_en': 1,\n                'hk_zh': 1,\n                'hr': 1,\n                'hu': 1,\n                'ie': 1,\n                'il_en': 1,\n                'il_he': 1,\n                'in': 1,\n                'it': 1,\n                'jp': 1,\n                'kr': 1,\n                'la': 1,\n                'lt': 1,\n                'lu_de': 1,\n                'lu_en': 1,\n                'lu_fr': 1,\n                'lv': 1,\n                'mena_ar': 1,\n                'mena_en': 1,\n                'mena_fr': 1,\n                'mx': 1,\n                'nl': 1,\n                'no': 1,\n                'nz': 1,\n                'pl': 1,\n                'pt': 1,\n                'ro': 1,\n                'rs': 1,\n                'ru': 1,\n                'se': 1,\n                'sea': 1,\n                'si': 1,\n                'sk': 1,\n                'tr': 1,\n                'tw': 1,\n                'ua': 1,\n                'uk': 1,\n                'us': 1\n            };\n        \n        //------------------------------------------------------------------------------\n        // Pathname\n        //------------------------------------------------------------------------------\n        // remove language_locale from path\n        // we do it this way to ensure that it only occurs in the \n        // beginning of the string\n        pathnameSplit = pathname.split('/');\n        if (pathnameSplit.length >= 2 && countryUrls[pathnameSplit[1]]) {\n            pathnameSplit.splice(1, 1);\n        }\n        \n        if (pathnameSplit.length >= 2) {\n            categoryPortion = pathnameSplit[1].toLowerCase();\n            // adobe.com/products/*\n            if (categoryPortion === 'products') {\n                if (pathnameSplit.length >= 3) {\n                    productPortion = pathnameSplit[2].toLowerCase();\n                    // photoshop-lightroom first so that we don't accidentally \n                    // assume photoshop\n                    if (productPortion.indexOf('photoshop-lightroom') !== -1) {\n                        products = 'PhotoshopLightroom';\n                    } else if (productPortion.indexOf('photoshop') !== -1) {\n                        products = 'Photoshop';\n                    } else if (productPortion.indexOf('illustrator') !== -1) {\n                        products = 'Illustrator';\n                    } else if (productPortion.indexOf('indesign') !== -1) {\n                        products = 'InDesign';\n                    } else if (productPortion.indexOf('premiere') !== -1) {\n                        products = 'PremierePro';\n                    } else if (productPortion.indexOf('aftereffects') !== -1) {\n                        products = 'AfterEffects';\n                    } else if (productPortion.indexOf('dreamweaver') !== -1) {\n                        products = 'Dreamweaver';\n                    } else if (productPortion.indexOf('muse') !== -1) {\n                        products = 'Muse';\n                    } else if (productPortion.indexOf('flash') !== -1) {\n                        products = 'FlashP';\n                    }\n                }\n            // adobe.com/creativecloud.html\n            } else if (categoryPortion.indexOf('creativecloud') !== -1) {\n                products = 'CreativeCloud';\n            }\n        }\n        \n        return products;\n    }\n</script>"
                }
                ]
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud.html",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: Photoshop CC Landing Page\nURL of the webpage where the tag is expected to be placed: https://www.adobe.com/products/photoshop.html?promoid=KLXLS\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 07/29/2014\n-->\n<script type="text/javascript">\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=creat746;ord=1;num=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=creat746;ord=1;num=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none">"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud.html")
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/buy/students.html",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: CCM Student Landing Page\nURL of the webpage where the tag is expected to be placed: http://www.adobe.com/creativecloud/buy/students.html\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 07/11/2014\n-->\n<script type="text/javascript">\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=CCMSt0;ord=1;num=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=CCMSt0;ord=1;num=1?" width="1" height="1" frameborder="0" style="display:none"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud/buy/students.html")
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/students/make-it-with-creative-cloud.html",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: CCM Student - Homepage\nURL of the webpage where the tag is expected to be placed: http://http://www.adobe.com/creativecloud/students/make-it-with-creative-cloud.html\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 08/12/2014\n-->\n<script type="text/javascript">\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=CCMSt001;ord=1;num=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=CCMSt001;ord=1;num=1?" width="1" height="1" frameborder="0" style="display:none"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud/students/make-it-with-creative-cloud.html")
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /products/photoshop.html",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: Photoshop CC Landing Page\nURL of the webpage where the tag is expected to be placed: https://www.adobe.com/products/photoshop.html?promoid=KLXLS\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 07/11/2014\n-->\n<script type="text/javascript">\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=Photo0;ord=1;num=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=Photo0;ord=1;num=1?" width="1" height="1" frameborder="0" style="display:none"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/products/photoshop.html")
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /products/photoshop/features.html",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: Photoshop CC - Whats New - Features LP\nURL of the webpage where the tag is expected to be placed: https://www.adobe.com/products/photoshop/features.html\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 07/11/2014\n-->\n<script type="text/javascript">\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=Whats0;ord=1;num=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=cs;cat=Whats0;ord=1;num=1?" width="1" height="1" frameborder="0" style="display:none"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/products/photoshop/features.html")
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - Retargeting Instrumentatlity Control Group",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: Persistent RT Group_Control\nURL of the webpage where the tag is expected to be placed: http://tbd.com\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 08/11/2014\n-->\n<script type="text/javascript">\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=rt;cat=Persi00;ord=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=rt;cat=Persi00;ord=1?" width="1" height="1" frameborder="0" style="display:none"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return !("www.adobe.com" !== e.location.hostname && "www.stage.adobe.com" !== e.location.hostname||-1 === e.location.pathname.indexOf("/products/")&&-1 === e.location.pathname.indexOf("/creativecloud/"))
            }, function() {
                return _satellite.textMatch(_satellite.getVar("dcm_rti_group"), "control")
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - Retargeting Instrumentatlity Test Group",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: Persistent RT Group_Test\nURL of the webpage where the tag is expected to be placed: http://tbd.com\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 08/11/2014\n-->\n<script type="text/javascript">\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=rt;cat=Persi0;ord=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=rt;cat=Persi0;ord=1?" width="1" height="1" frameborder="0" style="display:none"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return !("www.adobe.com" !== e.location.hostname && "www.stage.adobe.com" !== e.location.hostname||-1 === e.location.pathname.indexOf("/products/")&&-1 === e.location.pathname.indexOf("/creativecloud/"))
            }, function() {
                return _satellite.textMatch(_satellite.getVar("dcm_rti_group"), "test")
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - DoubleClick - Floodlight Tags - /",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<!--\nStart of DoubleClick Floodlight Tag: Please do not remove\nActivity name of this tag: Adobe Homepage\nURL of the webpage where the tag is expected to be placed: http://www.adobe.com\nThis tag must be placed between the <body> and </body> tags, as close as possible to the opening tag.\nCreation Date: 06/26/2014\n-->\n<script type="text/javascript">\n_satellite.notify(\'s_fid: \' + _satellite.readCookie(\'s_fid\'));\nvar axel = Math.random() + "";\nvar a = axel * 10000000000000;\ndocument.write(\'<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=adobe994;cat=adobe008;u17=\' + _satellite.readCookie(\'s_fid\') + \';ord=\' + (new Date()).getTime() + \';num=\' + a + \'?" width="1" height="1" frameborder="0" style="display:none"></iframe>\');\n</script>\n<noscript>\n<iframe src="https://1295336.fls.doubleclick.net/activityi;src=1295336;type=adobe994;cat=adobe008;u17=[s_fid];ord=1;num=1?" width="1" height="1" frameborder="0" style="display:none"></iframe>\n</noscript>\n<!-- End of DoubleClick Floodlight Tag: Please do not remove -->'
                }
                ]
            }
            ],
            conditions: [function() {
                return !("www.adobe.com" !== e.location.hostname && "www.stage.adobe.com" !== e.location.hostname || "/" !== e.location.pathname && "" !== e.location.pathname && "/africa" !== e.location.pathname && "/ap" !== e.location.pathname && "/at" !== e.location.pathname && "/au" !== e.location.pathname && "/be_en" !== e.location.pathname && "/be_fr" !== e.location.pathname && "/be_nl" !== e.location.pathname && "/bg" !== e.location.pathname && "/br" !== e.location.pathname && "/ca" !== e.location.pathname && "/ca_fr" !== e.location.pathname && "/ch_de" !== e.location.pathname && "/ch_fr" !== e.location.pathname && "/ch_it" !== e.location.pathname && "/cis" !== e.location.pathname && "/cn" !== e.location.pathname && "/cz" !== e.location.pathname && "/de" !== e.location.pathname && "/dk" !== e.location.pathname && "/ee" !== e.location.pathname && "/eeurope" !== e.location.pathname && "/es" !== e.location.pathname && "/fi" !== e.location.pathname && "/fr" !== e.location.pathname && "/hk_en" !== e.location.pathname && "/hk_zh" !== e.location.pathname && "/hr" !== e.location.pathname && "/hu" !== e.location.pathname && "/ie" !== e.location.pathname && "/il_en" !== e.location.pathname && "/il_he" !== e.location.pathname && "/in" !== e.location.pathname && "/it" !== e.location.pathname && "/jp" !== e.location.pathname && "/kr" !== e.location.pathname && "/la" !== e.location.pathname && "/lt" !== e.location.pathname && "/lu_de" !== e.location.pathname && "/lu_en" !== e.location.pathname && "/lu_fr" !== e.location.pathname && "/lv" !== e.location.pathname && "/mena_ar" !== e.location.pathname && "/mena_en" !== e.location.pathname && "/mena_fr" !== e.location.pathname && "/mx" !== e.location.pathname && "/nl" !== e.location.pathname && "/no" !== e.location.pathname && "/nz" !== e.location.pathname && "/pl" !== e.location.pathname && "/pt" !== e.location.pathname && "/ro" !== e.location.pathname && "/rs" !== e.location.pathname && "/ru" !== e.location.pathname && "/se" !== e.location.pathname && "/sea" !== e.location.pathname && "/si" !== e.location.pathname && "/sk" !== e.location.pathname && "/tr" !== e.location.pathname && "/tw" !== e.location.pathname && "/ua" !== e.location.pathname && "/uk" !== e.location.pathname && "/us" !== e.location.pathname)
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - Forsee",
            trigger: [{
                engine: "sc",
                command: "setVars",
                arguments: [{
                    hier1: ""
                }
                ]
            }, {
                command: "loadScript",
                arguments: [{
                    sequential: !1,
                    scripts: [{
                        src: "satellite-537523ee30f0f630d80005af.js",
                        data: ['16).toString(16);7<b.length&&(b=b.substr(b.length-7));return b+"-"+a.length+d.substr(d.length-6)+"-xxxx-xxxx-xxxxx".replace(/[xy]/g,function(a){var b=16*Math.random()|0;return("x"==a?b:b&3|8).toString(16)})};n.Da=function(){return 0+100*Math.random()};n.mf=function(a,b){var d=l.document.createElement("a");\nd.href=l.location.href;var c=d.hostname,f=d.protocol;d.href=a;var m=d.hostname||c,h=0==d.protocol.indexOf("http")?d.protocol:f;d.href=b;f=0==d.protocol.indexOf("http")?d.protocol:f;return m.toLowerCase()==(d.hostname||c).toLowerCase()&&h.toLowerCase()==f.toLowerCase()};n.O=function(a,b,d){var e="";if(a)for(var f in a)e+=(0!=e.length?"&":"")+(b?b+"["+f+"]":f)+"="+(d?a[f]:c.bb(a[f]));return e};n.hash=function(a){a=a.split("_");return 3*a[0]+1357+""+(9*a[1]+58)};n.qe=function(a){var b=0,d="";if(0==a.length)return b;\nfor(w=0;w<a.length;w++)d=a.charCodeAt(w),b=(b<<5)-b+d,b&=b;return b};n.Vc=function(a){a=a.replace(/[\\[]/,"\\\\[").replace(/[\\]]/,"\\\\]");a=RegExp("[\\\\?&+]"+a+"=([^&#]*)").exec(c.L());return null==a?!1:a[1]};n.oa=function(a,b,d){return a[b]||a[d]};n.Ob=function(a){a=a.replace(/[^0-9]/g,"");return 10==a.length||"1"==a[0]&&11==a.length};n.Nb=function(a){return null!=a.match(/^([a-zA-Z0-9_\\.\\-\\+])+\\@(([a-zA-Z0-9\\-])+\\.)+([a-zA-Z0-9]{2,6})+$/)};for(var ea={},Y=["onload","onerror","onabort"],w=0;w<Y.length;w++)ea[Y[w]]=\nfunction(){this.lb(0==arguments.callee.re?1:0);this.pb=!1},ea[Y[w]].re=w;u.M=function(a,b){this.options=c.A({},a);this.pb=!1;this.event=b;this.ic=0;return this};u.M.prototype.lb=function(a,b){if(this.pb)switch(this.pb=!1,this.status=a,a){case 1:(this.options.onSuccess||c.X)(b);break;case 0:this.event?this.of():(this.options.onFailure||c.X)(b);break;case -1:(this.options.onError||c.X)(b)}};u.M.prototype.of=function(){if(3>this.ic)this.uc();else this.onFailure()};u.M.prototype.xc=function(a,b){this.pb=\n!0;var d=n.O(c.A(a,{uid:c.now()})),d=c.za()+"//"+this.options.host+this.options.path+this.options.url+"?"+d;b=c.A({},ea,b);for(var e=new Image,f=0;f<Y.length;f++){var m=Y[f];e[m]=function(){var a=arguments.callee;a.Ja.onload=a.Ja.onerror=a.Ja.onabort=null;a.ie.call(a.self,a.Ja);a.Ja=null};e[m].ie=b[m];e[m].Ja=e;e[m].self=this}e.src=d};u.M.prototype.send=function(a){this.uf=a;this.uc()};u.M.prototype.Ba=function(){var a=c.A(this.options.mb,{protocol:c.za()});this.xc(a,{onload:function(a){this.options.Y&&\na.width!=this.options.Y?this.lb(0,a.width):this.lb(1,a.width)},onerror:function(){this.lb(-1)}})};u.M.prototype.uc=function(){var a;this.ic++;a=c.A({event:this.event,ver:this.ic},this.uf,a);this.xc(a)};c.aa.wd={};var s=c.aa.wd;s.ca=function(a,b){var d,e,f;c.k(a.length)||(a=[a]);d=0;for(e=a.length;d<e;d++){f=a[d];var m=f.className||"";RegExp("\\\\b"+b+"\\\\b").test(m)||(f.className=(""==m?"":m+" ")+b)}};s.fa=function(a,b){var d,e,f;c.k(a.length)||(a=[a]);d=0;for(e=a.length;d<e;d++)f=a[d],f.className&&\n(f.className=f.className.replace(RegExp("(\\\\s|^)"+b+"(\\\\s|$)")," ").replace(/^\\s+|\\s+$/g,""))};s.be=function(a,b){if(a){c.k(a.length)||(a=[a]);for(var d=0;d<a.length;d++)for(var e in b)e&&(-1=="zIndex".indexOf(e)&&("number"==typeof b[e]&&"opacity"!=e)&&(b[e]+="px"),a[d].style[e]=b[e])}return a};s.Af=function(a,b){if(a){c.k(a.length)||(a=[a]);for(var d=0;d<a.length;d++)for(var e in b)a[d].setAttribute(e,b[e])}return a};var N=s.be;s.outerHTML=function(a){if(c.k(a.outerHTML))return a.outerHTML;var b=\n{TEXTAREA:!0},d={HR:!0,BR:!0,IMG:!0,INPUT:!0},e=[],f="",m=a.nodeName;switch(a.nodeType){case 1:f=f+"<"+m.toLowerCase();if(b[m])switch(m){case "TEXTAREA":for(b=0;b<a.attributes.length;b++)if("value"!=a.attributes[b].nodeName.toLowerCase())f+=" "+a.attributes[b].nodeName.toUpperCase()+\'="\'+a.attributes[b].nodeValue+\'"\';else var h=a.attributes[b].nodeValue;f+=">";f+=h;f+="</"+m+">"}else{for(b=a.attributes.length-1;0<=b;b--)h=a.attributes[b].nodeName.toLowerCase(),-1<"style,class,id".indexOf(h.toLowerCase())&&\n(f+=" "+h+\'="\'+a.attributes[b].nodeValue+\'"\');f+=">";d[m]||(f+=a.innerHTML,f+="</"+m.toLowerCase()+">")}break;case 3:f+=a.nodeValue;break;case 8:f+="\\x3c!--"+a.nodeValue+"--\\x3e"}e.push(f);return e.join("")};c.stringify=function(a,b,d){var e;l.Prototype&&(e=Array.prototype.toJSON,delete Array.prototype.toJSON);if(l.JSON&&"function"===typeof l.JSON.stringify)a=l.JSON.stringify(a,b,d);else{var f;V=B="";if("number"===typeof d)for(f=0;f<d;f+=1)V+=" ";else"string"===typeof d&&(V=d);if((J=b)&&"function"!==\ntypeof b&&("object"!==typeof b||"number"!==typeof b.length))throw Error("_4c.stringify");a=$("",{"":a})}c.k(e)&&(Array.prototype.toJSON=e);return a};c.parse=function(a){if(l.JSON&&c.N(l.JSON.parse))return l.JSON.parse(a);a=String(a);fa.lastIndex=0;fa.test(a)&&(a=a.replace(fa,function(a){return"\\\\u"+("0000"+a.charCodeAt(0).toString(16)).slice(-4)}));if(/^[\\],:{}\\s]*$/.test(a.replace(/\\\\(?:["\\\\\\/bfnrt]|u[0-9a-fA-F]{4})/g,"@").replace(/"[^"\\\\\\n\\r]*"|true|false|null|-?\\d+(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?/g,\n"]").replace(/(?:^|:|,)(?:\\s*\\[)+/g,"")))return(new Function("return "+a))();throw new SyntaxError("_4c.parse");};var fa=/[\\u0000\\u00ad\\u0600-\\u0604\\u070f\\u17b4\\u17b5\\u200c-\\u200f\\u2028-\\u202f\\u2060-\\u206f\\ufeff\\ufff0-\\uffff]/g,da=/[\\\\\\"\\x00-\\x1f\\x7f-\\x9f\\u00ad\\u0600-\\u0604\\u070f\\u17b4\\u17b5\\u200c-\\u200f\\u2028-\\u202f\\u2060-\\u206f\\ufeff\\ufff0-\\uffff]/g,B,V,ra={"\\b":"\\\\b","\\t":"\\\\t","\\n":"\\\\n","\\f":"\\\\f","\\r":"\\\\r",\'"\':\'\\\\"\',"\\\\":"\\\\\\\\"},J;n.D={};n.D.Ga={};n.D.tb=function(a,b,d,e){var f=n.D.Ga;if(a){f[b]||\n(f[b]=[]);f[b].push({Gc:a,ab:d});if("unload"==b){if(c.k(c.Va)){c.Va.push(d);return}c.Va=[]}"propertychange"!=b&&a.addEventListener?a.addEventListener(b,d,!e):a.attachEvent&&a.attachEvent("on"+b,d)}};n.D.wf=function(a,b,d,c,f){var m=n.D;if(f){if(a.getAttribute("_fsr"+b))return!1;a.setAttribute("_fsr"+b,"true")}else if(f=m.Ga[b])for(m=f.length-1;0<=m;m--)if(f[m].Gc==a&&(c||f[m].ab==d))return!1;n.D.tb(a,b,d)};n.D.xf=function(a,b,d){n.D.tb(a,b,d,!0)};n.D.pc=function(a,b,d,c){try{"propertychange"!=b&&\na.removeEventListener?a.removeEventListener(b,d,!!c):a.detachEvent&&a.detachEvent("on"+b,d)}catch(f){}};var z=n.D.tb,Q=n.D.pc;n.D.yd=function(){for(var a=c.Va.length-1;0<=a;a--)try{c.Va[a].call()}catch(b){}c.fc(c.Va);n.D.Ad();c.Gb()};z(l,"unload",n.D.yd);n.D.Ad=function(){if(c){var a=n.D,b;for(b in a.Ga){for(var d=a.Ga[b],e={};e=d.pop();)a.pc(e.Gc,b,e.ab),c.fc(e);delete a.Ga[b]}}};n.D.ub=function(){this.Ta=[];this.ee=!1};n.D.ub.prototype.Sa=function(a){this.Ta[this.Ta.length]={Je:!1,ab:a}};n.D.ub.prototype.G=\nfunction(){this.ee=!0;for(var a=0;a<this.Ta.length;a++){var b=this.Ta[a];b.ab.apply(this,arguments);b.Je&&(this.Ta.splice(a,1),a--)}};var F=n.D.ub;s.oc={Gd:{}};try{Array.prototype.slice.call(document.getElementsByTagName("html")),makeArray=function(a){return Array.prototype.slice.call(a)}}catch(wa){}var y=s.oc.Rf=function(a,b,d){b=b||v;d=!c.k(c.f)||!c.f.Q.rf||d;if(b.querySelectorAll&&!(c.la&&8>=r.q.ia&&-1<a.indexOf("nth")))return ka(b.querySelectorAll(a));if(!d&&l.$&&!l.Prototype)return l.$(a,b);\na=a.split(",");d=[];for(var e=a.length-1;0<=e;e--){var f=a[e].replace(/^\\s\\s*/,"").replace(/\\s\\s*$/,"").replace(/\\*=/g,"=").replace(/\\>/g," > ").replace(/\\s+/g," ");if(-1<f.indexOf(" ")){for(var f=f.split(" "),m=[b],h=!1,g=0;g<f.length;g++)if(">"==f[g])h=!0;else{for(var k=[],n=m.length-1;0<=n;n--)k=k.concat(ja(f[g],m[n],h));m=k;h=!1}d=d.concat(c.de(m))}else d=d.concat(ja(f,b))}return d};c.aa.f={};var h=c.aa.f;h.Mc=function(a,b){for(var d=a.name,c=[a.site,a.section,b,g.g("q"),g.g("l")],f=0;f<c.length;f++)d+=\nc[f]?"-"+c[f]:"";return d};h.Be=function(a,b){function d(b){"ok"===b&&c.surveydefs&&(c.A(p,c.properties),k.Fa=k.surveydefs=c.surveydefs,a())}var e=k.definition||"foresee-surveydef.js";b?setTimeout(function(){d("ok")},100):c.kb(n.oa(k.site,"js_files","files")+e,"script",d)};h.log=function(a,b){if(p.events.enabled){var d=g.g(),e=d.get("sd");c.k(e)||(e=d.get("cd"));c.k(e)||(e=0);var e=k.Fa[e],f=new Date;(new u.M((new h.V(p)).event(),"logit")).send({cid:k.id,rid:d.get("rid")||"",cat:e.name,sec:e.section||\n"",type:d.get("q")||"",site:k.site.name||"",lang:d.get("l")||(c.$S?c.$S.locale:""),msg:a,param:b,tms:f.getTime(),tmz:6E4*f.getTimezoneOffset()})}};u.C=function(a,b){var d={method:"POST",url:c.L(),data:{},contentType:"application/x-www-form-urlencoded",Y:c.X,pa:c.X};this.ud=this.hc=!1;var e=n.oa;if(l.Worker&&!b){var f=n.mf,h=e(k.site,"js_files","files");if(f(h,l.location.href))this.Ld(h+(k.worker||"foresee-worker.js"));else{var e=e(k.site,"html_files","files"),g=document.createElement("a");g.href=\ne;(this.Kb=g.protocol+"//"+g.hostname)&&f(e,h)&&(this.Kd(e+"iframe_proxier.html"),h!=e&&this.Pd(h+"foresee_worker.js"))}}this.options=c.A(d,a)};u.C.prototype.send=function(a,b){var d=c.A(this.options,a);!l.XDomainRequest||"IE"==r.q.name&&10<=r.q.ia?this.ud&&!b?this.vc(d):this.hc&&!b?this.Sd(d):l.XMLHttpRequest&&this.Ud(d):this.Td(d)};u.C.prototype.Gb=function(){this.sb&&this.sb.terminate();this.ka&&(this.ka.parentNode.removeChild(ifr),ifr=null);c.fc(this.options)};u.C.isSupported=function(){return c.la&&\n10>r.q.ia&&"https"!=c.L().substring(0,5)&&l==l.top?!1:!0};u.C.hb=function(a){a.call(u.C)};u.C.prototype.Kd=function(a){this.ka=document.createElement("iframe");this.ka.src=a;this.ka.onload=u.C.Hd(this);this.ka.style.display="none";document.body.appendChild(this.ka);this.Ua=0;this.Oa={};this.ud=!0;z(l,"message",function(a){return function(d){u.C.rc(a,d)}}(this))};u.C.prototype.Ld=function(a){try{this.sb=new Worker(a),this.hc=!0}catch(b){}this.hc&&(this.Ua=0,this.Oa={},this.sb.onmessage=function(a){return function(b){u.C.rc(a,\nb)}}(this))};u.C.rc=function(a,b){var d=a.Oa[b.data.i];switch(b.data.status){case 200:d.Y&&d.Y.call(d,b.data.rt);break;case -1:c.r.da();break;default:d.pa&&d.pa.call(d,b.data.rt)}delete a.Oa[b.data.i]};u.C.Hd=function(a){return function(){a.se=!0;if(a.ua)for(var b=0;b<a.ua.length;b++)a.vc(a.ua[b]);a.ua=null}};u.C.prototype.Ud=function(a){var b=new l.XMLHttpRequest,d=c.k(a.bc)&&!0==a.bc?a.data:n.O(a.data,null,!1);try{b.open(a.method,a.url,!0)}catch(e){c.r.da();return}b.setRequestHeader("Accept","*/*");\nb.setRequestHeader("Content-Type",a.contentType);b.onreadystatechange=function(a,b){return function(){4==b.readyState&&200==b.status?a.Y&&a.Y.apply(a,[b.responseText]):4==b.readyState&&200!=b.status&&a.pa&&a.pa.apply(a,[b.responseText])}}(a,b);b.send(d)};u.C.prototype.Sd=function(a){a=c.A(this.options,a);this.Oa[++this.Ua]=a;this.sb.postMessage(u.C.tc(a,this.Ua))};u.C.prototype.vc=function(a){var b=c.A(this.options,a);this.se?(this.Oa[++this.Ua]=b,this.ka.contentWindow.postMessage(u.C.tc(b,this.Ua),\nthis.Kb)):this.ua?this.ua[this.ua.length]=a:this.ua=[a]};u.C.prototype.Td=function(a){var b=c.k(a.bc)&&!0==a.bc?a.data:n.O(a.data,null,!1),d=new l.XDomainRequest;d.onerror=a.pa;d.ontimeout=a.pa;d.onprogress=c.X;d.onload=function(a,b){return function(){b.Y(a.responseText);b=a=null}}(d,a);d.timeout=3E4;try{d.open("post",a.url)}catch(e){c.r.da();return}d.send(b)};u.C.prototype.Pd=function(a){var b={m:"worker_url"};b.u=a;this.ka.contentWindow.postMessage(b,this.Kb)};u.C.tc=function(a,b){var d={i:b},c=\n["method","url","data","contentType"],f;for(f in c)d[c[f]]=a[c[f]];return{m:"CORS",d:d}};c.aa.Bd={};var g=c.aa.Bd;g.qa=function(a){return a+(k.site.cookie?"."+k.site.cookie:"")};g.g=function(a,b){var d=g.qa("fsr.s"),d=g.Za(d,g.cb(d));return a?c.k(b)?d.set(a,b):d.get(a):d};g.cb=function(a){var b;b="window"==k.storageOption&&g.Ha.isSupported()?function(){var a=arguments.callee;return new g.Ha(a.Uc,a.Jc||{})}:function(){var a=arguments.callee;return new g.l(a.Uc,c.A({path:"/",domain:a.Sb.site.domain,\nsecure:a.Sb.site.secure,encode:a.Sb.encode},a.Jc||{}))};b.Uc=a;b.Sb=k;b.Jc=void 0;return b};var la={};g.Za=function(a,b){var d=la[a];return null!=d?d:d=la[a]=new b};var ma={IE:6.9,Safari:2,Firefox:1.4,Opera:1E3},na={Android:1.9,Winphone:7.4};n.vd=function(){function a(){c.Qa(function(a,b,d){return function(){a.jb=b();a.xe=d();a.Mb=!0;a.Cb.G()}}(g,e,b))}function b(){var a=!0;g.S&&(g.jb=e(),"Android"==g.B.name&&(2.2>g.B.version?a=!1:3>g.B.version&&g.jb&&(a=!1)));return a}function d(){g.q.name=p.name;\ng.q.version=p.version;g.q.ia="IE"!=g.q.name?g.q.version:6<g.q.version&&10>g.q.version?h("Trident")||7!=g.q.version?h("Trident/5.0")&&9>=g.q.version?9:h("Trident/4.0")&&9>g.q.version?8:g.q.version:7:g.q.version;g.B.name=f(g.S);var a=g.B,b;g.S?(b=k.match(/Android[\\/\\s](\\d+\\.?\\d+)/)||k.match(/Windows Phone OS[\\/\\s](\\d+\\.?\\d+)/)||k.match(/Windows Phone[\\/\\s](\\d+\\.?\\d+)/),b=null==b?1:b[1]):b=1;a.version=b}function e(){if("Winphone"!=g.B.name){var a=y("head meta[name=viewport],head meta[name=VIEWPORT],head meta[name=Viewport]")||\n[];c.I(a)||(a=[a]);if(0<a.length)for(var b=0;b<a.length;b++){var d=function(a,b){return a.match(RegExp("[\\\\w\\\\W]*"+b+"[\\\\s]*=[\\\\s]*([^\\\\s,;]*)[\\\\w\\\\W]*","i"))},e=d(a[b].content,"user-scalable"),f=d(a[b].content,"initial-scale"),d=d(a[b].content,"maximum-scale");if(e&&1<e.length&&(0<="iphone,ipad,ipod".indexOf(g.B.name.toLowerCase())&&"0"==e[1].toLowerCase()||0<="android".indexOf(g.B.name.toLowerCase())&&"no"==e[1].toLowerCase()))return!1;if(f&&d)return!("1.0"==f[1]&&"1.0"==d[1])}return!0}return!1}\nfunction f(a){if(a)return h("iPod")?"iPod":h("iPad")?"iPad":h("iPhone")?"iPhone":(h("blackberry")||h("playbook")||h("BB10"))&&h("applewebkit")?"Blackberry":h("Windows Phone")?"Winphone":h("Kindle")||h("Silk")?"Kindle":h("BNTV")||h("Nook")?"Nook":h("Android")?"Android":void 0!=l.orientation?"Mobile":"Other";if(h("Windows"))return"Windows";if(h("OS X"))return"Mac";if(h("Linux"))return"Linux";if(h("Mac"))return"Mac"}function h(a){return-1<k.toLowerCase().indexOf(a.toLowerCase())}var g=this;g.B={name:"",\nversion:0};g.q={name:"",version:0,ia:0};g.Ab="";g.S=!1;g.La=!1;g.xe=!0;g.jb=!0;g.Mb=!1;g.Cb=new F;g.Rc=!1;g.jd=c.za()+"//device.4seeresults.com/detect?accessToken=";var k=g.Ab=l.navigator.userAgent;g.S=/iphone|ipad|ipod|android|kindle|silk|bntv|nook|blackberry|playbook|mini|windows\\sce|windows\\sphone|palm|bb10/i.test(k);g.ye=/Windows Phone/i.test(k);g.S&&(/iphone|ipad|ipod/i.test(k)&&(g.Rc=!0),/ipad|silk|kindle|playbook|nook|bntv/i.test(k)&&(g.La=!0));var p=function(a){var b="Unknown",d;null!=(d=\na.match(/Opera[\\/\\s](\\d+\\.\\d+)/))?b="Opera":null!=(d=a.match(/MSIE (\\d+\\.\\d+)/))?b="IE":null!=(d=a.match(/Navigator[\\/\\s](\\d+\\.\\d+)/))?b="Netscape":null!=(d=a.match(/Chrome[\\/\\s](\\d+\\.\\d+)/))?b="Chrome":null!=(d=a.match(/Safari[\\/\\s](\\d+\\.?\\d+)/))?b="Safari":null!=(d=a.match(/Firefox[\\/\\s](\\d+\\.\\d+)/))&&(b="Firefox");return{name:b,version:null!=d?parseFloat(d[1]):void 0}}(k);if(g.S)if(g.Rc||""==g.jd||g.La||g.ye)d(),a(),a();else{var r=function(b){b=c.parse(b);g.q.name=b.browser.name;g.q.version=g.q.ia=\nb.browser.version;g.B.name=b.os.name;g.B.version=b.os.version;g.S=b.isMobile;g.La=b.isTablet;a()},s;if(l.sessionStorage){var C=l.sessionStorage;s=C.getItem("FSR_BROWSER")}s?r(s):(s={method:"GET",url:g.jd+n.qe(function(){var a=new Date,b=(a.getMonth()+1).toString(),d=a.getDate().toString();return a.getFullYear().toString()+(b[1]?b:"0"+b[0])+(d[1]?d:"0"+d[0])}()+"ForeSee"+(l.location.origin||"null"))+"&ua="+k,type:"*/*",contentType:"application/x-www-form-urlencoded",Y:function(a){C&&C.setItem("FSR_BROWSER",\na);r(a)},pa:function(){d();a();a()}},(new u.C(s,!0)).send())}else d(),g.Mb=!0,g.Cb.G()};var r=new n.vd;s.ha={};s.ha.nc=function(a){var b=0,d=0,c=a.document,f=c.documentElement;"number"==typeof a.innerWidth?(b=a.innerWidth,d=a.innerHeight):f&&(f.clientWidth||f.clientHeight)?(b=f.clientWidth,d=f.clientHeight):c.body&&(c.body.clientWidth||c.body.clientHeight)&&(b=c.body.clientWidth,d=c.body.clientHeight);return{w:b,h:d}};s.ha.xd=function(a){return r.S?{w:a.innerWidth,h:a.innerHeight}:s.ha.nc(a)};s.ha.mc=\nfunction(a){var b=0,d=0,c=a.document,f=c.documentElement;"number"==typeof a.pageYOffset?(d=a.pageYOffset,b=a.pageXOffset):c.body&&(c.body.scrollLeft||c.body.scrollTop)?(d=c.body.scrollTop,b=c.body.scrollLeft):f&&(f.scrollLeft||f.scrollTop)&&(d=f.scrollTop,b=f.scrollLeft);return{x:b,y:d}};s.ha.zf=function(a,b,d){a.scrollTo(b,d);window.document.body.scrollTop=d;window.document.body.scrollLeft=b};h.vb={};h.vb.Ra=function(a,b){if(a){var d=g.g("m");if(d&&(d=(new Date).getTime()-d,d<1E3*b)){var c=function(){var a=\n(new h.V(p)).Ge();a.mb={rid:k.rid,cid:k.id};(new u.M(a)).Ba()};c();var f=setInterval(c,1E3*a);setTimeout(function(){clearInterval(f)},1E3*b-d)}}};h.V=function(a){a=a&&a.survey||{};this.sc={name:a.host||"survey.foreseeresults.com"};this.Id={name:a.events_host||"events.foreseeresults.com"};this.qc={name:".4seeresults.com"};this.wc={name:"i.4see.mobi"};this.Qd=a.protocol||c.za()};h.V.prototype.kf=function(){return{host:this.sc.name,path:"/survey",url:"/display",protocol:this.Qd}};h.V.prototype.He=function(){return{host:this.wc.name,\npath:"/e",url:"/initialize"}};h.V.prototype.Ge=function(){return{host:this.wc.name,path:"/e",url:"/recordHeartbeat"}};h.V.prototype.F=function(){return{host:"controller"+this.qc.name,path:"/fsrSurvey",url:"/OTCImg",Y:3}};h.V.prototype.event=function(){return{host:this.Id.name,path:"/rec",url:"/process"}};h.V.prototype.domain=function(){return{host:this.sc.name,path:"/survey",url:"/FSRImg",Y:3}};h.V.prototype.df=function(){return{host:"replaycontroller"+this.qc.name,path:"/images",enabled:!0}};h.U=\nfunction(a,b){this.options=a;this.Z=b;this.Z.invite=c.A({position:{pin:{left:!1,right:!1,top:!1,bottom:!1},offset:{left:"0px",top:"0px"}}},this.Z.invite);this.zb=new F;this.Fb=new F;this.fd=new F};h.U.prototype.show=function(a,b,d){this.Tb=b;this.Wd=d;this.Fc=this.yc=!1;this.Zb=!0;var e=r.S;b=a[0].mobileExitDialog;if(0==this.Tb&&(c.I(this.Z.invite.dialogs)&&1<this.Z.invite.dialogs.length&&(this.Zb=!1),s.ca(l.document.documentElement,"fsrInvitePresent"),e)){s.ca(l.document.documentElement,"fsrM");\ns.ca(l.document.documentElement,"fsrOnExit");-1<"Winphone".indexOf(r.B.name)&&s.ca(l.document.documentElement,"fsrWinPhone");var f="Android"==r.B.name&&3>r.B.version;f&&s.ca(l.document.documentElement,"fsrMobileCompat");this.Na=y(\'meta[name="viewport"]\',l.document.head);if(!this.Na.length||r.jb)f?(this.K=H(\'<meta name="viewport" content="width=device-width, user-scalable=no, target-densityDpi=high-dpi" />\'),v.head.appendChild(this.K)):-1<"iPod,iPad,iPhone".indexOf(r.B.name)?(this.K=H(\'<meta name="viewport" content="user-scalable=0"/>\'),\nv.head.appendChild(this.K)):-1<"Android".indexOf(r.B.name)&&(this.K=H("<meta content=\'width=device-width; initial-scale=1.0; maximum-scale=1.0;minimum-scale=1.0; user-scalable=no;\' name=\'viewport\' />"),v.head.appendChild(this.K));z(v,"touchstart",c.X)}var m=this.Z.invite,p=n.oa(k.site,"image_files","files"),G=g.g("l"),f=this.Ma=H(\'<div id="fsrOverlay" class="fsrC" style="font-size:12px"><div class="fsrFloatingContainer"><div class="fsrFloatingMid"><div class="fsrInvite"></div></div></div></div>\');\nm.hideOnClick&&z(f,"click",function(a){return function(b){"fsrOverlay"==(b.originalTarget||b.target||b.srcElement).id&&(b&&b.preventDefault?b.preventDefault():l.event&&l.event.returnValue&&(l.eventReturnValue=!1),a.ja())}}(this));var M=y(".fsrFloatingContainer",f)[0],U=y(".fsrInvite",f)[0],u=H(\'<div class="fsrDialogs"></div>\');U.appendChild(u);a=h.U.tf(a,d,G);d=m.siteLogo?m.siteLogo:"";"object"===typeof d&&(d=d.hasOwnProperty(G)?d[G]:d.base);m=m.siteLogoAlt?m.siteLogoAlt:"";for(G=0;G<a.length;G++){var C=\na[G],A=\'<div class="fsrLogos">\',K=G==a.length-1,D="";0==G&&(A+=""!=d?\'<img class="fsrSiteLogo" alt="\'+m+\'" src="$SITEFILES$SLOGO">\':\'<img class="fsrSiteLogo" alt="" src="">\');K&&(A+=\'<img class="fsrCorpLogo" alt="Foresee" src="$SITEFILESfsrlogo.gif">\');var A=A+"</div>",q=\'<p class="fsrSubBlurb">$FNOTICE</p>\';C.noticeAboutSurvey||(q="");var w="";b&&(w=\'<input type="hidden" id="mobileOnExitSupport" value="\'+b.support+\'"/><div class="fsrMobileExitErrorFieldRequired fsrMobileExitError hideField">\'+b.fieldRequiredErrorText+\n\'</div><div class="fsrMobileExitErrorInvalidFormat fsrMobileExitError hideField">\'+b.invalidFormatErrorText+\'</div><input type="email" class="fsrEmailOrNumber" id="mobileOnExitInput" placeholder="\'+b.inputMessage+\'">\');var t=C.quizContent,x="";1<a.length&&(x+=" fsrMultiDialog",G<a.length-1&&(x+=" fsrDSpacer"));A=H((\'<div class="fsrDialog \'+x+\'" style="margin-left: 0px;">\'+A+\'<p class="fsrHeading">$FHEAD</p><p class="fsrBlurb">$FBLURB</p>\'+q+w+"</div>").replace(/\\$SITEFILES/gi,p).replace(/\\$SLOGO/gi,\nd).replace(/\\$FHEAD/gi,C.headline).replace(/\\$FBLURB/gi,C.blurb).replace(/\\$FNOTICE/gi,C.noticeAboutSurvey));if(t){q=H(\'<div class="fsrQuiz"></div>\');q.appendChild(H(\'<p class="fsrQuizQuestion">\'+t.question+"</p>"));for(D=0;D<t.answers.length;D++){var w=t.answers[D],x=function(){return function(a){a=(a.originalTarget||a.target||a.srcElement).parentNode.parentNode.parentNode;N(y(".fsrQuiz",a),{display:"none"});N(y(".fsrSubBlurb",a),{display:"block"});N(y(".fsrB",a),{display:"block"})}},F=function(a,\nb,d){return function(c){c=(c.originalTarget||c.target||c.srcElement).parentNode.parentNode.parentNode;c.innerHTML=(\'<div class="fsrDialog" style="margin-left: 0px;"><div class="fsrLogos"><img class="fsrCorpLogo" alt="ForeSee" src="$SITEFILESfsrlogo.gif"></div><p class="fsrHeading fsrCTitle">\'+b.cancelTitle+\'</p><p class="fsrBlurb">\'+b.cancelText+\'</p><div class="fsrB" style="display: block;"><a class="declineButton fsrDb">\'+d+"</a></div></div>").replace(/\\$SITEFILES/gi,p);pa(y(".declineButton",c),\nfunction(a){return function(){a.ja()}}(a));c=null}},B=H(\'<p class="fsrAnswer" id="fsrAns\'+G+"_"+D+\'"><input name="fsrQuiz\'+G+\'" type="radio" id="fsrA\'+G+"_"+D+\'"><label for="fsrA\'+G+"_"+D+\'">\'+w.answer+"</label></p>");q.appendChild(B);w.proceedWithSurvey?z(B,"click",x()):z(B,"click",F(this,w,C.closeInviteButtonText))}w=B=null;D="display:none;";A.appendChild(q)}C.attribution&&(t=H(\'<p class="fsrAttribution">$FATTR</p>\'.replace(/\\$FATTR/gi,C.attribution)),A.appendChild(t));t=H((\'<div class="fsrB" style="\'+\nD+\'"><div class="fsrAcceptButtonContainer"><a tabindex="2" class="fsrAcceptButton" href="javascript:void(0)">$ABTN</a></div><div class="fsrDeclineButtonContainer"><a tabindex="1" class="fsrDeclineButton" href="javascript:void(0)">$FDECL</a></div></div>\').replace(/\\$ABTN/gi,C.acceptButton).replace(/\\$FDECL/gi,C.declineButton));A.appendChild(t);K&&(A.appendChild(H(\'<div class="fsrFooter"><a class="fsrTE" target="_blank" title="Validate TRUSTe privacy certification" href="http://privacy-policy.truste.com/click-with-confidence/ctv/en/www.foreseeresults.com/seal_m"><img class="fsrTruste" alt="TRUSTe verified" src="$SITEFILEStruste.png"></a></div>\'.replace(/\\$SITEFILES/gi,\np))),U.appendChild(H(\'<a title="$CCLOSE" class="fsrCloseBtn" href="#">&#215;</a>\'.replace(/\\$CCLOSE/gi,C.closeInviteButtonText))),z(y(".fsrCloseBtn",U)[0],"click",function(a){return function(b){b&&b.preventDefault?b.preventDefault():l.event&&l.event.returnValue&&(l.eventReturnValue=!1);a.ja()}}(this)));u.appendChild(A);var E=C.locale;z(y(".fsrAcceptButton",A)[0],"click",function(a,b){return function(){g.g("l",b);a.ma(b)}}(this,E));z(y(".fsrDeclineButton",A)[0],"click",function(a,b){return function(){a.ja(b)}}(this,\nE));if(1<a.length&&K){K=function(a){return y(".fsrB",a)[0].offsetTop};t=y(".fsrDialog");D=t[0];for(q=0;q<t.length-1;q++)K(D)<K(t[q+1])&&(D=t[q+1]);for(q=0;q<t.length;q++)t[q]!=D&&(w=K(D)-K(t[q]),x=y(".fsrHeading",t[q])[0],"IE"==r.q.name&&9>r.q.ia?x.style.cssText="padding-top: "+w.toString()+"px":N(x,{"padding-top":w}))}if(b){var I=n;this.Fe=function(a,b,d,c){var e=!1,f=y(".fsrAcceptButton")[0];a&&(I.Nb(a)||I.Ob(a))&&(I.Nb(a)?f.innerHTML=d:I.Ob(a)&&(f.innerHTML=c),e=!0);e||(f.innerHTML=b)};var J=y(".fsrEmailOrNumber",\nA)[0],P=function(a){return function(){var b=a.getBoundingClientRect();l.scrollTo(0,b.top+s.ha.mc(l).y-(s.ha.nc(l).h-b.height)/2)}}(J);z(J,"focus",function(a){return function(){a.Xb=!0;s.fa(y(".fsrMobileExitError"),"showField");s.ca(y(".fsrMobileExitError"),"hideField");"Android"==r.B.name&&setTimeout(P,500)}}(this,E));z(J,"blur",function(a){return function(){a.Xb=!1;setTimeout(Q,1)}}(this));z(J,"keyup",function(a,b,d,c){return function(e){a.Fe(this.value,b,d,c);13==(e.ze?e.keyCode:e.which)&&(J.blur(),\ng.g("l",E),a.ma(E))}}(this,C.acceptButton,b.emailMeButtonText,b.textMeButtonText))}v.body.appendChild(f);c.la&&"CSS1Compat"!=l.document.compatMode&&(f.className="fsrC ie6");e||(this.Jb=function(a){return function(b){27==(b.ze?b.keyCode:b.which)&&a.ja()}}(this),z(v,"keyup",this.Jb));s.fa(l.document.documentElement,"fsrWider");var R={width:M.offsetWidth,height:M.offsetHeight,gd:M.offsetWidth/M.offsetHeight};s.ca(l.document.documentElement,"fsrWider");var S={width:M.offsetWidth,height:M.offsetHeight,\ngd:M.offsetWidth/M.offsetHeight};s.fa(l.document.documentElement,"fsrWider");var Q=this.Ca=function(a,b){return function(){setTimeout(function(){if(!a.Xb){var d=s.ha,c=d.xd(l),d=d.mc(l),f=1,f=0.98;r.La&&(f=0.55);c.aw=c.w*f;c.ah=c.h*f;winratio=c.aw/c.ah;f=R;c.w>c.h?(s.ca(l.document.documentElement,"fsrWider"),f=S):s.fa(l.document.documentElement,"fsrWider");f=f.gd>winratio?c.aw/f.width:c.ah/f.height;f=Math.max(Math.min(12*f,e?84:12),e?3:7);N(b,{visibility:"visible",display:"block",width:c.w+"px",height:c.h+\n"px",top:d.y+"px",left:d.x+"px",fontSize:f+"px"});N(M,{marginTop:(b.offsetHeight-M.offsetHeight)/2+"px"})}},150)}}(this,f);this.Xb=!1;Q();z(l,"resize",this.Ca);z(l,"scroll",this.Ca);if("Android"==r.B.name||"Winphone"==r.B.name){var O=!1;this.Yb=function(a){O=!0;-1<a.target.className.indexOf("fsr")&&(O=!1)};z(f,"mousedown",this.Yb,!0);this.Wb=function(a){if(O)return a.preventDefault(),a.stopPropagation(),!1};z(v,"click",this.Wb)}}};h.U.tf=function(a,b,d){for(var e=[],f=0;f<a.length;f++){var g=a[f],\nh=!1;b&&(g.locale&&b!=g.locale)&&(h=!0);h||((h=g.locales)&&h[d]&&(g=c.A(g,h[d]),c.k(g.locale)||(g.locale=d)),g.skipThisInvite||e.push(g))}return e};h.U.prototype.sf=function(a,b){this.Nc(".mobileExitErrorFieldRequired");this.Nc(".mobileExitErrorInvalidFormat");if(""===a)return this.ld(".fsrMobileExitErrorFieldRequired"),!1;var d=n.Nb(a),c=n.Ob(a);(d="b"==b?d||c:"e"==b?d:"s"==b?c:!1)||this.ld(".fsrMobileExitErrorInvalidFormat");return d};h.U.prototype.Nc=function(a){s.fa(y(a),"showField");s.ca(y(a),\n"hideField")};h.U.prototype.ld=function(a){s.fa(y(a),"hideField");s.ca(y(a),"showField")};h.U.prototype.ma=function(a){this.Ca();c.xa("mobileOnExitInput")?this.sf(c.trim(c.xa("mobileOnExitInput").value),c.trim(c.xa("mobileOnExitSupport").value))&&this.zb.G(a,this.Tb):this.zb.G(a,this.Tb)};h.U.prototype.ja=function(a){this.Zb=!0;this.Fb.G(a)};h.U.prototype.ob=function(a){this.fd.G(a)};h.U.prototype.fb=function(){if(this.Zb&&(s.fa(l.document.documentElement,"fsrInvitePresent"),r.S)){for(var a=["fsrM",\n"fsrMobileCompat","fsrWinPhone","fsrOnExit"],b=0;b<a.length;b++)s.fa(l.document.documentElement,a[b]);if(this.Na&&this.Na.length&&c.k(this.K)&&this.K.parentNode)for(this.K.parentNode.removeChild(this.K),a=0;a<this.Na.length;a++)v.head.appendChild(this.Na[a]);else c.k(this.K)&&this.K.parentNode&&(this.K.parentNode.removeChild(this.K),this.K="Android"==r.B?H(\'<meta name="viewport" content="user-scalable=yes;"/>\'):H(\'<meta name="viewport" content="user-scalable=1;"/>\'),v.head.appendChild(this.K));Q(v,\n"touchstart",c.X)}s.fa(l.document.documentElement,"fsrWider");this.Jb&&Q(l.document,"keyup",this.Jb,!0);this.Ca&&(Q(l,"resize",this.Ca,!0),Q(l,"scroll",this.Ca,!0));this.Ma&&this.Ma.parentNode&&this.Ma.parentNode.removeChild(this.Ma);this.Yb&&Q(this.Ma,"mousedown",this.Yb,!0);this.Wb&&Q(v,"click",this.Wb,!0)};x={width:"1",height:"1",id:"_"+(""+Math.random()).slice(9),allowfullscreen:!0,allowscriptaccess:"always",quality:"high",version:[3,0],Ie:null,he:null,jc:!1,Zd:!1};l.attachEvent&&l.attachEvent("onunload",\nfunction(){__flash_unloadHandler=L();__flash_savedUnloadHandler=L()});var aa=c.A(c.Cf,{Bf:x,ne:function(){var a,b;try{b=navigator.plugins["Shockwave Flash"].description.slice(16)}catch(d){try{b=(a=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"))&&a.GetVariable("$version")}catch(c){try{b=(a=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"))&&a.GetVariable("$version")}catch(f){}}}return(b=/(\\d+)[^\\d]+(\\d+)[^\\d]*(\\d*)/.exec(b))?[b[1],b[3]]:[0,0]},Bb:function(a){if(null===a||void 0===a)return null;\nvar b=typeof a;"object"==b&&a.push&&(b="array");switch(b){case "string":return a=a.replace(RegExp(\'(["\\\\\\\\])\',"g"),"\\\\$1"),a=a.replace(/^\\s?(\\d+\\.?\\d*)']
                    }
                    ]
                }
                ]
            }
            ],
            conditions: [function() {
                return - 1 !== ["Desktop"].indexOf(_satellite.browserInfo.deviceType)
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - PS Streaming Trial",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: "<script type=\"text/javascript\">\n    $(\"body\").bind(\"mf2-ready\", function() {\n        var mf2;\n        try {\n            mf2 = document.querySelector('#embed-container #mf2-events').jsMF2;\n        } catch (e) {\n            console.log('SATELLITE ---- error in loading mf2');\n            return;\n        }\n        var originalPN = s_adbadobenonacdc.pageName || _satellite.getVar(\"adbadobenonacdc_pageName_Page_Name\");\n        // window.beforeunload\n        window.addEventListener('beforeunload', function(event) {\n            console.log('SATELLITE ---- beforeunload');\n            console.log('SATELLITE ---- ' + Math.round((mf2.getSessionInfo().allowedTime - mf2.getSessionInfo().timeRemaining) / 60));\n\n            if (typeof window._trackedPlayerClose !== 'boolean' && !window._trackedPlayerClose) {\n                s_adbadobenonacdc.prop42 = \"window_close\";\n                s_adbadobenonacdc.prop43 = \"\" + Math.round((mf2.getSessionInfo().allowedTime - mf2.getSessionInfo().timeRemaining) / 60);\n                s_adbadobenonacdc.events = _satellite._apl(s_adbadobenonacdc.events, 'prodView', ',', 2);\n                s_adbadobenonacdc.products = _satellite._apl(s_adbadobenonacdc.products, ';Photoshop', ',', 2);\n                s_adbadobenonacdc.pageName = originalPN + ':onClose_Player';\n                s_adbadobenonacdc.t_s(); // track synchronously\n                \n                console.log('SATELLITE ---- shoulda tracked');\n    \n                window._trackedPlayerClose = true;\n\n                return;\n            }\n        });\n        //buttons tracking after trial experience\n        function trackMf2CloseEvents(eventName) {\n            var totalTime = \"\" + Math.round((mf2.getSessionInfo().allowedTime - mf2.getSessionInfo().timeRemaining) / 60);\n            s_adbadobenonacdc.prop42 = eventName;\n            s_adbadobenonacdc.prop43 = totalTime;\n            s_adbadobenonacdc.events = _satellite._apl(s_adbadobenonacdc.events, 'prodView', ',', 2);\n            s_adbadobenonacdc.products = _satellite._apl(s_adbadobenonacdc.products, ';Photoshop', ',', 2);\n            s_adbadobenonacdc.pageName = originalPN + ':onClose_Player';\n            s_adbadobenonacdc.t();\n        }\n\n        function nextStep() {\n            setTimeout(function() {\n                $(\"#nextstep-join-now\").on(\"click\", function() {\n                    trackMf2CloseEvents(\"html_join\");\n                });\n                $(\"#nextstep-download-trial\").on(\"click\", function() {\n                    trackMf2CloseEvents(\"html_download\");\n                });\n            }, 500);\n        };\n        //on click of close inside window\n        mf2.bind('closed', function(info) {\n            console.log('SATELLITE ---- CLOSED');\n            console.log('SATELLITE ---- ' + Math.round((mf2.getSessionInfo().allowedTime - mf2.getSessionInfo().timeRemaining) / 60));\n            if (typeof window._trackedPlayerClose !== 'boolean' && !window._trackedPlayerClose) {\n\n                //console.log('SATELLITE ---- shoulda tracked');\n                window._trackedPlayerClose = true;\n                nextStep();\n            }\n        });\n        //learning panel tracking\n      function trackOpenURLEvnt(eventName, timeUsed) {\n            s_adbadobenonacdc.prop42 = eventName;\n            s_adbadobenonacdc.prop43 = timeUsed;\n            s_adbadobenonacdc.events = _satellite._apl(s_adbadobenonacdc.events, 'prodView', ',', 2);\n            s_adbadobenonacdc.products = _satellite._apl(s_adbadobenonacdc.products, ';Photoshop', ',', 2);\n            s_adbadobenonacdc.pageName = originalPN + ':onClose_Player';\n            s_adbadobenonacdc.t();\n        }\n        mf2.bind('onOpenURL', function(info) {\n            if (typeof window._trackedPlayerClose !== 'boolean' && !window._trackedPlayerClose) {\n                var jnApiURL = \"https://mfdemo.editinps.com/api/join\";\n                var dwnldApiURL = \"https://mfdemo.editinps.com/api/download\";\n                var timeUsed = \"\" + Math.round((mf2.getSessionInfo().allowedTime - mf2.getSessionInfo().timeRemaining) / 60);\n                window._trackedPlayerClose = true;\n                if (info == jnApiURL) {\n                    trackOpenURLEvnt(\"panel_join\", timeUsed);\n                } else if (info == dwnldApiURL) {\n                    trackOpenURLEvnt(\"panel_download\", timeUsed);\n                }\n            }\n        });\n        //Error based tracking\n        mf2.bind('error', function(err) {\n            trackMf2CloseEvents(\"mf2_error_download\");\n        });\n    });\n</script>"
                }
                ]
            }
            ],
            conditions: [function() {
                return !("www.adobe.com" !== e.location.hostname && "www.stage.adobe.com" !== e.location.hostname||-1 === e.location.pathname.indexOf("/special/tnt/cs0103")&&-1 === e.location.pathname.indexOf("/special/tnt/streaming")&&-1 === e.location.pathname.indexOf("/special/tnt/streaming_stage"))
            }
            ],
            event: "pagebottom"
        }, {
            name: "~0 - SAMCAP - set TID cookie",
            trigger: [{
                command: "loadScript",
                arguments: [{
                    sequential: !1,
                    scripts: [{
                        src: "satellite-5388d296cb3b815c6300036e.js",
                        data: []
                    }
                    ]
                }
                ]
            }
            ],
            event: "pagetop"
        }, {
            name: "~3 - initTools",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: "<script type=\"text/javascript\">\n\n// adbreimaginetest analytics tool\nif (_satellite.getVar('useTool_adbreimaginetest')) {\n    _satellite.notify('Using tool adbreimaginetest', 1);\n} else {\n    _satellite.tools['ba1811d4769e557169de925c6cbb9b6b'].settings.initTool = false;\n    _satellite.notify('Not using tool adbreimaginetest', 1);\n}\n\n</script>"
                }
                ]
            }
            ],
            event: "pagetop"
        }, {
            name: "~5 - AppMeasurement Plugins",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<script type="text/javascript">\n/*! dtm-adobe-analytics-plugins 2014-11-13 11:04:37 */\n!function(a){"use strict";var b="1.0";a._adobeAnalytics_plugin_getTrackingServer=function(a){return a?(a._plugin_getTrackingServer=function(){var a,b=this,c=b.trackingServer,d="2o7.net",e=b.dc,f=b.visitorNamespace;return c?b.trackingServerSecure&&b.ssl&&(c=b.trackingServerSecure):(f||(f=b.account,a=f.indexOf(","),a>=0&&(f=f.substring(0,a)),f=f.replace(/[^A-Za-z0-9]/g,"")),e=e?(""+e).toLowerCase():"d1","d1"==e?e="112":"d2"==e&&(e="122"),c=f+"."+e+"."+d),c},a._plugin_getTrackingServer.version=b,!0):!1},a._adobeAnalytics_plugin_getTrackingServer.version=b}(_satellite),function(a){"use strict";var b="1.1",c={};a._adobeAnalytics_plugin_combinedCookies=function(a){if(!a)return!1;if(c[a._in])return!1;c[a._in]=!0;var d=null,e=null;return d=a.escape?function(b){return a.escape(b)}:a.ape?function(b){return a.ape(b)}:encodeURIComponent,e=a.unescape?function(b){return a.unescape(b)}:a.epa?function(b){return a.epa(b)}:decodeURIComponent,a.c_rr=a.c_r,a.c_wr=a.c_w,a.c_rspers=function(){var a,b,c=this,d=c.c_rr("s_pers"),e=(new Date).getTime(),f="",g=[],h="",i="";if(!d)return i;for(g=d.split(";"),a=0,b=g.length;b>a;a++)h=g[a],f=h.match(/\\|([0-9]+)$/),f&&parseInt(f[1],10)>=e&&(i+=h+";");return i},a.c_r=a.cookieRead=function(a){var b=this,c=b.c_rr(a),f=d(a),g=" "+f+"=",h="",i=b.c_rspers(),j=-1!==i.indexOf(g),k=b.c_rr("s_sess"),l=-1!==k.indexOf(g),m=-1,n=-1;return c?c:(j?h=i:l&&(h=k),m=h.indexOf(g),-1===m?c="":(n=h.indexOf("|",m),-1===n&&(n=h.indexOf(";",m)),-1===n&&(n=h.length),c=e(h.substring(m+2+f.length,n))),c)},a.c_w=a.cookieWrite=function(a,b,c){var e=this,f=new Date,g=(new Date).getTime(),h=0,i=d(a),j=" "+i+"=",k="s_pers",l=e.c_rspers(),m=new Date,n=l.indexOf(j),o=-1!==n,p=l,q=!1,r=0,s=0,t="s_sess",u=e.c_rr(t),v=u.indexOf(j),w=-1!==v,x=u,y=!1,z="";if(e.c_rr(a)&&(f.setTime(g-6e4),e.c_wr(a,"",f)),o&&(p=l.substring(0,n)+l.substring(l.indexOf(";",n)+1),q=!0),w&&(x=u.substring(0,v)+u.substring(u.indexOf(";",v)+1),y=!0),c?(1==c&&(c=new Date,h=c.getYear(),c.setYear(h+5+(1900>h?1900:0))),c.getTime()>g&&(p+=j+d(b)+"|"+c.getTime()+";",q=!0)):(x+=j+d(b)+";",y=!0),x=x.replace(/%00/g,""),p=p.replace(/%00/g,""),y&&e.c_wr(t,x,0),q){for(z=p;z&&-1!==z.indexOf(";");)s=parseInt(z.substring(z.indexOf("|")+1,z.indexOf(";")),10),z=z.substring(z.indexOf(";")+1),s>r&&(r=s);m.setTime(r),e.c_wr(k,p,m)}return b===e.c_r(a)},a.c_r.version=b,a.c_w.version=b,!0},a._adobeAnalytics_plugin_combinedCookies.version=b}(_satellite),function(a){"use strict";var b="1.0";a._adobeAnalytics_plugin_socialAuthors=function(a){return a?(a._plugin_socialAuthors=a.socialAuthors=function(){var a,b=this,c=!0;return a=b.referrer?b.referrer:document.referrer,(0===a.indexOf("http://t.co/")||0===a.indexOf("https://t.co/")||-1!==a.indexOf("pinterest.com/pin")||-1!==a.indexOf("tumblr.com")||-1!==a.indexOf("youtube.com"))&&(b.Integrate.add("SocialAuthor"),b.Integrate.SocialAuthor.get("http://sa-services.social.omniture.com/author/name?var=[VAR]&callback=s_c_il["+b._in+"].socialAuthorSearch&rs="+encodeURIComponent(b.account)+"&q="+encodeURIComponent(a)),b.Integrate.SocialAuthor.delay(),b.Integrate.SocialAuthor.setVars=function(a){a.contextData["a.socialauthor"]=a.user}),c},a._plugin_socialAuthorSearch=a.socialAuthorSearch=function(a){var b=this,c=!0;return b.user="undefined"==typeof a||"undefined"==typeof a.author?"Not Found":a.author,b.Integrate.SocialAuthor.ready(),c},a._plugin_socialAuthors.version=b,a._plugin_socialAuthorSearch.version=b,!0):!1},a._adobeAnalytics_plugin_socialAuthors.version=b}(_satellite),function(a){"use strict";var b="1.0";a._adobeAnalytics_plugin_clickPast=function(a){if(!a)return!1;var c={};return a._plugin_clickPast=a.clickPast=function(a,b,d,e,f){var g,h=this,i=-1;return c[b]||(c[b]=!0,e||(e="s_cpc"),f||(f=","),g=h.events?h.events+f:"",a?(h.events=g+b,h.c_w(e,1,0),i=1):h.c_r(e)>=1&&(h.events=g+d,h.c_w(e,0,0),i=0)),i},a._plugin_clickPast.version=b,!0},a._adobeAnalytics_plugin_clickPast.version=b}(_satellite),function(a){"use strict";var b="2.0";a._adobeAnalytics_plugin_trackState=function(a){return a?(a._plugin_trackState=a.trackState=function(a,b){var c=this,d={};d.pageName=a,b&&(d.contextData=b),c.t(d)},a._plugin_trackState.version=a.trackState.version=b,!0):!1},a._adobeAnalytics_plugin_trackState.version=b}(_satellite),function(a){"use strict";var b="2.0";a._adobeAnalytics_plugin_trackAction=function(a){return a?(a._plugin_trackAction=a.trackAction=function(a,b){var c=this,d="",e=[],f="";b||(b={}),f=c.linkTrackVars,b["a.action"]||(b["a.action"]=a),e.push("contextData.a.action");for(d in b)b.hasOwnProperty(d)&&e.push("contextData."+d);c.linkTrackVars=e.join(","),c.tl(!0,"o",a,{contextData:b}),c.linkTrackVars=f},a._plugin_trackAction.version=a.trackAction.version=b,!0):!1},a._adobeAnalytics_plugin_trackAction.version=b}(_satellite),function(a){"use strict";var b="1.4";a._adobeAnalytics_plugin_demandbase=function(a){return a?(a._plugin_demandbase=function(a){function b(){var b=window[a.var_name],d="",e="",f="",g="",h="",i="",j="";if("done"!=c.c_r(a.var_name)){if("object"==typeof b){d=b.demandbase_sid,h=a.data_mapping.sid,h&&d&&(c[h]=d,j+=(j?",":"")+h,"events"===h&&(c.linkTrackEvents=c[h])),e=b.industry,f=b.sub_industry,h=a.data_mapping.industry_complete,h&&e&&f&&(c[h]=e+"::"+f,j+=(j?",":"")+h,"events"===h&&(c.linkTrackEvents=c[h]));for(g in b)b.hasOwnProperty(g)&&(i=b[g],h=a.data_mapping[g],h&&"string"==typeof i&&i&&(c[h]=i,j+=(j?",":"")+h,"events"===h&&(c.linkTrackEvents=c[h])));j&&(c.linkTrackVars=j),c.tl(!0,"o",a.link_name)}c.c_w(a.var_name,"done",0)}}var c=this,d="",e=null;"done"!=c.c_r(a.var_name)&&(d=("https:"===window.location.protocol?"https://":"http://")+"api.demandbase.com/api/v2/ip.js?key="+a.key+"&var="+a.var_name+"&rnd="+Math.floor(10001*Math.random()),e=document.createElement("SCRIPT"),e.type="text/javascript",e.src=d,"onload"in e?(e.onload=function(){b()},e.onerror=function(){}):"readyState"in e&&(e.onreadystatechange=function(){var a=e.readyState;("loaded"===a||"complete"===a)&&(e.onreadystatechange=null,b())}),document.getElementsByTagName("HEAD")[0].appendChild(e))},a._plugin_demandbase.version=b,!0):!1},a._adobeAnalytics_plugin_demandbase.version=b}(_satellite),function(a){"use strict";var b="1.4";a._adobeAnalytics_plugin_demandbaseDataConnector=function(a,c){return a&&c?(c.contextName=c.contextName||"s_dmdbase",c.contextNameCustom=c.contextNameCustom||"s_dmdbase_custom",c.apiBase=c.apiBase||"//api.demandbase.com/api/v2/ip.json",c.var_name=c.var_name||"s_demandbase_v2.2",c.link_name=c.link_name||"Demandbase Event",c.delim=c.delim||":",c.nonOrgMatchLabel=c.nonOrgMatchLabel||"[n/a]",a._plugin_demandbaseDataConnector=function(){var a=this,b="_plugin_demandbaseDataConnector_"+Math.floor(1e7*Math.random()),d=function(){window[b]=void 0;try{delete window[b]}catch(a){}},e=null,f=null,g="",h="",i=function(a,b){var d,e,f,g,h={city:"registry_city",state:"registry_state",zip:"registry_zip_code",country:"registry_country_code",latitude:"registry_latitude",longitude:"registry_longitude"},i=[];for(d=0,e=b.length;e>d;d++)f=b[d],f.max=f.max||20,f.id&&(a[f.id]||a[h[f.id]]?(g=a[f.id]||a[h[f.id]],g=g.toString(),g=g.replace(c.delim," "),i.push(g.substring(0,f.max))):i.push(a[f.id]===!1?"false":c.nonOrgMatchLabel));return i.join(c.delim)},j=function(b){var d,e,f,g,h,j={contextData:{}},k=0,l=a.linkTrackVars;if("done"==a.c_r(c.var_name))return!1;if(h=a.c_w(c.var_name,"done"),"object"!=typeof b)return!1;for(d in b)b.hasOwnProperty(d)&&k++;if(0===k)return!1;for(d in b)if(b.hasOwnProperty(d)&&"object"==typeof b[d]&&null!==b[d]&&b.hasOwnProperty(d)){for(e in b[d])b[d].hasOwnProperty(e)&&(b[d+"_"+e]=b[d][e]);delete b[d]}c.dimensionArray&&(f=i(b,c.dimensionArray),j.contextData[c.contextName]=f,h=a.c_w(c.contextName,f),a.linkTrackVars+=",contextData."+c.contextName),c.dimensionArrayCustom&&(g=i(b,c.dimensionArrayCustom),j.contextData[c.contextNameCustom]=g,h=a.c_w(c.contextNameCustom,g),a.linkTrackVars+=",contextData."+c.contextNameCustom),a.tl(!0,"o",c.link_name,j),a.linkTrackVars=l};return"done"==a.c_r(c.var_name)?(g=a.c_r(c.contextName),h=a.c_r(c.contextNameCustom),g&&(a.contextData[c.contextName]=g),h&&(a.contextData[c.contextNameCustom]=h),g):(f=document.getElementsByTagName("HEAD"),e=document.createElement("SCRIPT"),e.type="text/javascript",e.src=("https:"===window.location.protocol?"https:":"http:")+c.apiBase+"?key="+c.key+"&callback="+b+"&rnd="+Math.floor(10001*Math.random()),f=f&&f[0]?f[0]:document.body,f&&(window[b]=function(a){j(a),d()},f.appendChild(e)),!0)},a._plugin_demandbaseDataConnector.version=b,!0):!1},a._adobeAnalytics_plugin_demandbaseDataConnector.version=b}(_satellite),function(a){"use strict";var b="1.0";a._adobeAnalytics_plugin_trackTNT=function(a){return a?(a._plugin_trackTNT=function(a,b,c){var d=this,e="s_tnt",f="s_tntref",g="",h=!1;return a=a?a:e,b=b?b:e,c=c?c:!0,""!==d.Util.getQueryParam(f)?d.referrer=d.Util.getQueryParam(f):""!==d.c_r(f)?(d.referrer=d.c_r(f),document.cookie=f+"=;path=/;expires=Thu, 01-Jan-1970 00:00:01 GMT;"):(-1!=document.cookie.indexOf(f)&&""===d.c_r(f)||-1!=window.location.search.indexOf(f+"=")&&""===d.Util.getQueryParam(f))&&(d.referrer="Typed/Bookmarked",document.cookie=f+"=;path=/;expires=Thu, 01-Jan-1970 00:00:01 GMT;"),""!==d.Util.getQueryParam(b)?h=d.Util.getQueryParam(b):d.c_r(b)?(h=d.c_r(b),document.cookie=b+"=;path=/;expires=Thu, 01-Jan-1970 00:00:01 GMT;"):""===d.c_r(b)&&""===d.Util.getQueryParam(b)&&(h=""),h&&(g+=h+","),void 0!==window[a]&&(g+=window[a]),c&&(window[a]=""),g},a._plugin_trackTNT.version=b,!0):!1},a._adobeAnalytics_plugin_trackTNT.version=b}(_satellite),function(a){"use strict";var b="1.3";a._adobeAnalytics_plugin_trackSynchronous=function(c){if(!c)return!1;if("undefined"==typeof c._plugin_getTrackingServer&&("undefined"==typeof a._adobeAnalytics_plugin_getTrackingServer||!a._adobeAnalytics_plugin_getTrackingServer(c)))return!1;var d="",e=document.createElement("SCRIPT"),f=document.getElementsByTagName("HEAD"),g="_plugin_trackSynchronous_"+Math.floor(1e7*Math.random()),h=function(){window[g]=void 0;try{delete window[g]}catch(a){}},i=c.ssl?"https://":"http://",j=c._plugin_getTrackingServer();return f=f&&f[0]?f[0]:document.body,f&&(window[g]=function(a){a&&a.id&&(d=a.id),h()},e.type="text/javascript",e.src=i+j+"/id?callback="+g,f.firstChild?f.insertBefore(e,f.firstChild):f.appendChild(e)),c._plugin_trackSynchronous=c.t_s=function(a,b){var c,e,f=this,g=null,h=new Date,i=Math.floor(1e13*Math.random()),j="s"+Math.floor(h.getTime()/108e5)%10+i,k=h.getYear(),l=h.getDate()+"/"+h.getMonth()+"/"+(1900>k?k+1900:k)+" "+h.getHours()+":"+h.getMinutes()+":"+h.getSeconds()+" "+h.getDay()+" "+h.getTimezoneOffset(),m="t="+f.escape(l),n=window,o=null,p=function(a,b){var c=f.account,d=f.mobile?"5.":"",e=f.AudienceManagement&&f.AudienceManagement.isReady(),g=b?b:e?"10":"1",h=f.version,i=f.mb?"T":"",j=f.tagContainerMarker?"-"+f.tagContainerMarker:"";return"/b/ss/"+c+"/"+d+g+"/JS-"+h+i+j+"/"+a},q=function(a,b){var c=f.AudienceManagement&&f.AudienceManagement.isReady(),e=c?"callback=s_c_il["+f._in+"].AudienceManagement.passData&":"",g=f.ssl?"https://":"http://",h=f._plugin_getTrackingServer(),i=p(a,"10"),j=p(a),k="?AQB=1&ndh=1&aid="+d+"&"+b+"&AQE=1",l="?AQB=1&ndh=1&"+e+b+"&AQE=1",m=g+h+i+k,n=g+h+j+l,o=null;if(f.ab&&(m=m.substring(0,2047)),f.fb(m),"undefined"!=typeof XMLHttpRequest&&(o=new XMLHttpRequest)&&"withCredentials"in o){f.wa=f.r();try{if(o.open("GET",m,!1),o.send(null),200==o.status&&""!==o.responseText)return c&&f.Q&&f.AudienceManagement.passData(f.P(o.responseText)),f.Q?f.P(o.responseText):o.responseText;f.Sa(n),f.Y()}catch(q){f.Sa(n),f.Y()}}return null};try{for(c=n.parent,e=n.location;c&&c.location&&e&&""+c.location!=""+e&&n.location&&""+c.location!=""+n.location&&c.location.host==e.host;)n=c,c=n.parent}catch(r){}if(f.visitor&&(f.visitor.getAuthState&&(f.authState=f.visitor.getAuthState()),!f.supplementalDataID&&f.visitor.getSupplementalDataID&&(f.supplementalDataID=f.visitor.getSupplementalDataID("AppMeasurement:"+f._in,f.expectSupplementalData?!1:!0))),f.q("_s"),!f.B("track",arguments)){if(!f.Ga(a)){if(b&&f.K(b),a&&(o={},f.Aa(o,0),f.K(a)),f.bb()&&(f.analyticsVisitorID||f.marketingCloudVisitorID||(f.fid=f.Wa()),f.hb(),f.usePlugins&&f.doPlugins&&f.doPlugins(f),f.account)){if(!f.abort){f.trackOffline&&!f.timestamp&&(f.timestamp=Math.floor(h.getTime()/1e3));var s=window.location;f.pageURL||(f.pageURL=s.href?s.href:s),f.referrer||f.Ba||(f.referrer=n.document.referrer,f.Ba=1),f.referrer=f.Ua(f.referrer),f.q("_g")}f.Ya()&&!f.abort&&(f.Za(),m+=f.Xa(),g=q(j,m),f.q("_t"),f.referrer="")}a&&f.K(o,1)}f.abort=f.supplementalDataID=f.timestamp=f.pageURLRest=f.linkObject=f.clickObject=f.linkURL=f.linkName=f.linkType=window.vb=f.pe=f.pev1=f.pev2=f.pev3=f.g=0}return g},c._plugin_trackSynchronous.version=b,!0},a._adobeAnalytics_plugin_trackSynchronous.version=b}(_satellite),function(a){"use strict";var b="1.1";a._adobeAnalytics_plugin_spiderWeb=function(a,c){if(!a)return!1;c||(c={});var d,e,f={a:"a",abbr:"ab",acronym:"ac",address:"ad",altglyph:"ag",altglyphdef:"agd",altglyphitem:"agi",animate:"an",animatecolor:"anc",animatemotion:"anm",animatetransform:"ant",applet:"ap",area:"r",article:"c",aside:"A",audio:"au",b:"b",base:"bs",basefont:"ba",bdi:"bd",bdo:"bo",bgsound:"bg",big:"bi",blockquote:"bl",br:"br",button:"B",canvas:"ca",caption:"cp",center:"ce",cite:"C",circle:"ci",clippath:"ch",code:"co",col:"cl",colgroup:"cg","color-profile":"cf",content:"ct",cursor:"cu",data:"da",datalist:"dl",dd:"y",decorator:"dc",defs:"dE",del:"de",desc:"ds",details:"dt",dfn:"df",dialog:"di",dir:"dr",div:"d",dl:"D",dt:"z",element:"el",ellipse:"eL",em:"E",embed:"eb",feblend:"feb",fecolormatrix:"fcm",fecomponenttransfer:"fct",fecomposite:"fc",feconvolvematrix:"fcx",fediffuselighting:"fdl",fedisplacementmap:"fdm",fedistantlight:"fd",feflood:"ff",fefunca:"ffa",fefuncb:"ffb",fefuncg:"ffg",fefuncr:"ffr",fegaussianblur:"fgb",feimage:"fei",femerge:"fem",femergenode:"fen",femorphology:"fef",feoffset:"fe",fepointlight:"fpl",fespecularlighting:"fsl",fespotlight:"fSl",fetile:"fet",feturbulence:"ftb",fieldset:"J",figcaption:"fg",figure:"fi",filter:"fl",font:"fo","font-face":"ffc","font-face-format":"fff","font-face-name":"ffn","font-face-src":"ffs","font-face-uri":"ffu",footer:"f",foreignobject:"fro",form:"F",frame:"fr",frameset:"ft",g:"g",glyph:"gl",glyphref:"gr",h1:"ha",h2:"hb",h3:"hc",h4:"hd",h5:"he",h6:"hf",header:"h",hgroup:"hg",hkern:"hk",hr:"hr",i:"i",iframe:"if",image:"im",img:"I",input:"j",ins:"in",isindex:"is",kbd:"kb",keygen:"ke",label:"e",legend:"L",li:"l",line:"li",lineargradient:"lg",link:"lk",listing:"ls",main:"m",map:"M",mark:"mr",marker:"mk",marquee:"mq",mask:"ms",menu:"me",menuitem:"mi",metadata:"md",meter:"mt","missing-glyph":"mg",mpath:"mp",nav:"n",nobr:"no",noframes:"nf",noscript:"ns",object:"ob",ol:"o",optgroup:"op",option:"O",output:"ou",p:"p",param:"pa",path:"ph",pattern:"pe",picture:"pi",plaintext:"pt",polygon:"po",polyline:"pl",pre:"pr",progress:"pg",q:"q",radialgradient:"rg",rect:"re",rp:"rp",rt:"rt",ruby:"ru",s:"s",samp:"sa",script:"sc",section:"S",select:"N",set:"se",shadow:"sh",small:"sm",source:"so",spacer:"sp",span:"P",stop:"sT",strike:"st",strong:"sr",style:"St",sub:"sB",summary:"su",sup:"sP",svg:"G","switch":"sw",symbol:"sy",table:"t",tbody:"v",td:"T",template:"te",text:"tx",textarea:"k",textpath:"tp",tfoot:"x",th:"H",thead:"w",time:"ti",title:"Ti",tr:"R",tref:"tr",track:"tc",tspan:"ts",tt:"tt",u:"u",ul:"U",use:"us","var":"va",video:"vi",view:"vw",vkern:"vk",wbr:"wb",xmp:"xm"},g={},h={1:"a",2:"b",3:"c",4:"d",5:"e",6:"f",7:"g",8:"h",9:"i",0:"j"},i={a:"1",b:"2",c:"3",d:"4",e:"5",f:"6",g:"7",h:"8",i:"9",j:"0"};for(e in f)f.hasOwnProperty(e)&&(d=f[e],g[d]=e);return a._plugin_spiderWeb_webs=[],a._plugin_spiderWeb=function(a){var b,c,d=this;if(!a||"[object Array]"!==Object.prototype.toString.call(a))return!1;for(b=0,c=a.length;c>b;b++)d._plugin_spiderWeb_addWeb(a[b])},a._plugin_spiderWeb_addWeb=function(a){var b=this;return a&&"object"==typeof a?("undefined"==typeof a.callback,a.rootElement||(a.rootElement=document.body),a.pageName||(a.pageName=b.pageName?b.pageName:""),a?(b._plugin_spiderWeb_webs.push(a),!0):!1):!1},a._plugin_spiderWeb_getUniqueIdForElement=function(a,b){var c,d,e=null,g=0,i=null,j="",k="",l=!1;for(b=b||document.body,e=a.parentNode;e!==document;){if(e===b){l=!0;break}e=e.parentNode}if(!l)return"";for(;a!==b;){for(j=a.tagName.toLowerCase(),e=a.parentNode,g=0,c=0,d=e.children.length;d>c&&(i=e.children[c],j==i.tagName.toLowerCase()&&g++,i!==a);c++);if(f[j]?j=f[j]:(j=j.replace(/[0-9]/g,function(a){return"^"+h[a]}),j="{"+j+"}"),!g)throw new Error("An error occurred in generating uniqueId for element: "+j);k=j+g+k,a=e}return k},a._plugin_spiderWeb_getElementByUniqueId=function(a,b){var c,d,e,f,h=[],j="",k="",l=0,m=[],n=null,o=0,p=b||document.body;if("string"==typeof a){for(a=a.replace(/([0-9]+)/g,function(a,b){return b+" "}),h=a.split(" "),c=0,d=h.length;d>c;c++)if(j=h[c],""!==j){for(k=j.replace(/[0-9]+/g,""),l=j.replace(/[^0-9]+/g,""),g[k]?k=g[k]:(k=k.replace(/\\^([a-j])/g,function(a,b){return i[b]}),-1!==k.indexOf("{")&&(k=k.replace("{","")),-1!==k.indexOf("}")&&(k=k.replace("}",""))),m=p.children,o=0,e=0,f=m.length;f>e;e++)if(n=m[e],k==n.tagName.toLowerCase()&&o++,o==l){p=n;break}if(!p)return null}return p}},a._plugin_spiderWeb_getLinkElement=function(a,b){var c=null,d="",e=null,f=!1;for(b=b||document.body,c=a.parentNode;c!==document;){if(c===b){f=!0;break}c=c.parentNode}if(!f)return"";for(;a!==b;){if(d=a.tagName.toLowerCase(),c=a.parentNode,("a"===d||"area"===d)&&a.href){e=a;break}a=c}return e},a._plugin_spiderWeb_isExitLink=function(a){var b,c,d=this,e=null,f=0,g=0,h=!1;if(d.linkInternalFilters||(d.linkInternalFilters=window.location.hostname),d.linkExternalFilters?(e=d.linkExternalFilters.toLowerCase().split(","),f=1):d.linkInternalFilters&&(e=d.linkInternalFilters.toLowerCase().split(",")),e){for(b=0;b<e.length;b++)c=e[b],a.indexOf(c)>=0&&(g=1);g?f&&(h=!0):f||(h=!0)}return h},a._plugin_spiderWeb_shouldNavigate=function(a){return a.href&&a.href?0===a.href.indexOf("about:")?!1:0===a.href.indexOf("opera:")?!1:0===a.href.indexOf("javascript:")?!1:(!a.target||"_self"==a.target||"_top"==a.target||"_parent"==a.target||window.name&&a.target==window.name)&&a.hash&&(a.href===window.location.href||a.protocol===window.location.protocol&&a.host===window.location.host&&a.pathname===window.location.pathname&&a.search===window.location.search)?!1:!0:!1},a._plugin_spiderWeb_spider=function(b){var c,d,e=(b.type.toLowerCase(),b.target||b.srcElement),f=null,g=null,h="",i="",j={location:{href:window.location.href,protocol:window.location.protocol,host:window.location.host,hostname:window.location.hostname.toLowerCase(),port:window.location.port,pathname:window.location.pathname,search:window.location.search,hash:window.location.hash},id:"",id1:"",id2:"",id3:"",id4:"",id5:"",pageName:"",destination:{href:"",protocol:"",host:"",hostname:"",port:"",pathname:"",search:"",hash:""},element:{nodeName:e.nodeName,tagName:e.tagName,target:e.target},linkElement:null,isExitLink:!1,shouldNavigate:!1};for(c=0,d=a._plugin_spiderWeb_webs.length;d>c;c++)f=a._plugin_spiderWeb_webs[c],h=a._plugin_spiderWeb_getUniqueIdForElement(e,f.rootElement),j.id=h,i=e.getAttribute("id"),j.id1=i?i:"",f.pageName?j.pageName=f.pageName:a.pageName&&(j.pageName=a.pageName),g=a._plugin_spiderWeb_getLinkElement(e,f.rootElement),g&&(j.linkElement=g,j.destination={href:g.href,protocol:g.protocol,host:g.host,hostname:g.hostname.toLowerCase(),port:g.port,pathname:g.pathname,search:g.search,hash:g.hash},j.isExitLink=a._plugin_spiderWeb_isExitLink(g.href),j.shouldNavigate=a._plugin_spiderWeb_shouldNavigate(g)),f.callback&&"function"==typeof f.callback&&f.callback(j,b)},document.addEventListener?document.addEventListener("click",a._plugin_spiderWeb_spider,!0):document.attachEvent&&document.attachEvent("onClick",a._plugin_spiderWeb_spider),a._plugin_spiderWeb.version=b,a._plugin_spiderWeb_addWeb.version=b,a._plugin_spiderWeb_getUniqueIdForElement.version=b,a._plugin_spiderWeb_getElementByUniqueId.version=b,!0},a._adobeAnalytics_plugin_spiderWeb.version=b}(_satellite),function(a){"use strict";var b="1.0";a._adobeAnalytics_plugin_getLoadTime=function(c){return c?(c._plugin_getLoadTime=c.getLoadTime=function(b,c,d){var e,f=this;try{if(f.alreadySetLoadTime){for(var g=f.events.split(","),h="",i=0;i<g.length;i++)-1==g[i].indexOf(c+"=")&&g[i]!=d&&(h=h+g[i]+",");return void(f.events=h.substring(0,h.length-1))}if("header"==b&&"undefined"!=typeof s_preLoad)e=(new Date).getTime()-s_preLoad;else{if("browserapi"!=b||"undefined"==typeof performance)return;try{e=(new Date).getTime()-performance.timing.requestStart}catch(j){return}}if(e/1e3>500)return;if(0>e)return}catch(j){return}f.events=a._apl(f.events,c+"="+Math.round(e/1e3)+","+d,",",2);var k,l,m,n=navigator.appVersion,o=navigator.userAgent,p=navigator.appName,q=""+parseFloat(n),r=parseInt(n,10);return-1!=(l=o.indexOf("Opera"))||-1!=(l=o.indexOf("OPR/"))?(p="Opera",q=o.substring(-1!=o.indexOf("OPR/")?l+4:l+6),-1!=(l=o.indexOf("Version"))&&(q=o.substring(l+8))):-1!=(l=o.indexOf("MSIE"))?(p="Microsoft Internet Explorer",q=o.substring(l+5)):-1!=(l=o.indexOf("Chrome"))?(p="Chrome",q=o.substring(l+7)):-1!=(l=o.indexOf("Safari"))?(p="Safari",q=o.substring(l+7),-1!=(l=o.indexOf("Version"))&&(q=o.substring(l+8))):-1!=(l=o.indexOf("Firefox"))?(p="Firefox",q=o.substring(l+8)):(k=o.lastIndexOf(" ")+1)<(l=o.lastIndexOf("/"))?(p=o.substring(k,l),q=o.substring(l+1),p.toLowerCase()==p.toUpperCase()&&(p=navigator.appName)):(p="Other Unknown Browser",q=""),-1!=(m=q.indexOf(";"))&&(q=q.substring(0,m)),-1!=(m=q.indexOf(" "))&&(q=q.substring(0,m)),r=parseInt(""+q,10),isNaN(r)&&(q=""+parseFloat(navigator.appVersion),r=parseInt(navigator.appVersion,10)),f.alreadySetLoadTime=!0,p+" "+r},c._plugin_getLoadTime.version=b,!0):!1},a._adobeAnalytics_plugin_getLoadTime.version=b}(_satellite);\n</script>'
                }
                ]
            }
            ],
            event: "pagetop"
        }, {
            name: "~5 - AppMeasurement_Module_Integrate.js - Shared AppMeasurement Integrate Module Library",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<script type="text/javascript">\nfunction AppMeasurement_Module_Integrate(s){var m=this;m.s=s;var w=window;if(!w.s_c_in)w.s_c_il=[],w.s_c_in=0;m._il=w.s_c_il;m._in=w.s_c_in;m._il[m._in]=m;w.s_c_in++;m._c="s_m";m.list=[];m.add=function(c,b){var a;b||(b="s_Integrate_"+c);w[b]||(w[b]={});a=m[c]=w[b];a.a=c;a.e=m;a._c=0;a._d=0;a.disable==void 0&&(a.disable=0);a.get=function(b,c){var d=document,f=d.getElementsByTagName("HEAD"),g;if(!a.disable&&(c||(v="s_"+m._in+"_Integrate_"+a.a+"_get_"+a._c),a._c++,a.VAR=v,a.CALLBACK="s_c_il["+m._in+\n"]."+a.a+".callback",a.delay(),f=f&&f.length>0?f[0]:d.body))try{g=d.createElement("SCRIPT");g.type="text/javascript";g.setAttribute("async","async");g.src=m.c(a,b);if(b.indexOf("[CALLBACK]")<0)g.onload=g.onreadystatechange=function(){a.callback(w[v])};f.firstChild?f.insertBefore(g,f.firstChild):f.appendChild(g)}catch(s){}};a.callback=function(b){var m;if(b)for(m in b)Object.prototype[m]||(a[m]=b[m]);a.ready()};a.beacon=function(b){var c="s_i_"+m._in+"_Integrate_"+a.a+"_"+a._c;if(!a.disable)a._c++,\nc=w[c]=new Image,c.src=m.c(a,b)};a.script=function(b){a.get(b,1)};a.delay=function(){a._d++};a.ready=function(){a._d--;a.disable||s.delayReady()};m.list.push(c)};m._g=function(c){var b,a=(c?"use":"set")+"Vars";for(c=0;c<m.list.length;c++)if((b=m[m.list[c]])&&!b.disable&&b[a])try{b[a](s,b)}catch(w){}};m._t=function(){m._g(1)};m._d=function(){var c,b;for(c=0;c<m.list.length;c++)if((b=m[m.list[c]])&&!b.disable&&b._d>0)return 1;return 0};m.c=function(m,b){var a,w,e,d;b.toLowerCase().substring(0,4)!="http"&&\n(b="http://"+b);s.ssl&&(b=s.replace(b,"http:","https:"));m.RAND=Math.floor(Math.random()*1E13);for(a=0;a>=0;)a=b.indexOf("[",a),a>=0&&(w=b.indexOf("]",a),w>a&&(e=b.substring(a+1,w),e.length>2&&e.substring(0,2)=="s."?(d=s[e.substring(2)])||(d=""):(d=""+m[e],d!=m[e]&&parseFloat(d)!=m[e]&&(e=0)),e&&(b=b.substring(0,a)+encodeURIComponent(d)+b.substring(w+1)),a=w));return b}}\n</script>'
                }
                ]
            }
            ],
            event: "pagetop"
        }, {
            name: "~5 - AppMeasurement_Module_media.js - Shared AppMeasurement Media Module Library",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<script type="text/javascript">\nvar j=null;function E(){return function(){}}\nfunction AppMeasurement_Module_Media(s){var m=this;m.s=s;var w=window;if(!w.s_c_in)w.s_c_il=[],w.s_c_in=0;m._il=w.s_c_il;m._in=w.s_c_in;m._il[m._in]=m;w.s_c_in++;m._c="s_m";m.list=[];m.open=function(e,g,d,l){var c={},a=new Date,b="",h;g||(g=-1);if(e&&d){if(!m.list)m.list={};m.list[e]&&m.close(e);if(l&&l.id)b=l.id;if(b)for(h in m.list)!Object.prototype[h]&&m.list[h]&&m.list[h].Rf==b&&m.close(m.list[h].name);c.name=e;c.length=g;c.xc=0;c.U=0;c.playerName=m.playerName?m.playerName:d;c.Rf=b;c.ce=0;c.aa=\n0;c.timestamp=Math.floor(a.getTime()/1E3);c.za=0;c.wc=c.timestamp;c.T=-1;c.Dc="";c.ha=-1;c.Ic=0;c.Qd={};c.Mc=0;c.Ja=0;c.V="";c.Ob=0;c.Yd=0;c.Cc=0;c.Kc=0;c.xa=!1;c.Jb="";c.yc="";c.zc=0;c.sc=!1;c.na="";c.complete=0;c.Af=0;c.Hb=0;c.Ib=0;m.list[e]=c;c.Ud=!1}};m.openAd=function(e,g,d,l,c,a,b,h){var f={};m.open(e,g,d,h);if(f=m.list[e])f.xa=!0,f.Jb=l,f.yc=c,f.zc=a,f.na=b};m.Oe=function(e){var g=m.list[e];m.list[e]=0;g&&g.monitor&&clearTimeout(g.monitor.O)};m.close=function(e){m.ja(e,0,-1)};m.play=function(e,\ng,d,l){var c=m.ja(e,1,g,d,l);if(c&&!c.monitor)c.monitor={},c.monitor.update=function(){c.za==1&&m.ja(c.name,3,-1);c.monitor.O=setTimeout(c.monitor.update,1E3)},c.monitor.update()};m.click=function(e,g){m.ja(e,7,g)};m.complete=function(e,g){m.ja(e,5,g)};m.stop=function(e,g){m.ja(e,2,g)};m.track=function(e){m.ja(e,4,-1)};m.yf=function(e,g){var d="a.media.",l=e.linkTrackVars,c=e.linkTrackEvents,a="m_i",b,h=e.contextData,f;if(g.xa){d+="ad.";if(g.Jb)h["a.media.name"]=g.Jb,h[d+"pod"]=g.yc,h[d+"podPosition"]=\ng.zc;if(!g.Mc)h[d+"CPM"]=g.na}if(g.sc)h[d+"clicked"]=!0,g.sc=!1;h["a.contentType"]="video"+(g.xa?"Ad":"");h["a.media.channel"]=m.channel;h[d+"name"]=g.name;h[d+"playerName"]=g.playerName;if(g.length>0)h[d+"length"]=g.length;h[d+"timePlayed"]=Math.floor(g.aa);Math.floor(g.aa)>0&&(h[d+"timePlayed"]=Math.floor(g.aa));if(!g.Mc)h[d+"view"]=!0,a="m_s",m.Heartbeat&&m.Heartbeat.enabled&&(a=g.xa?m.__primetime?"mspa_s":"msa_s":m.__primetime?"msp_s":"ms_s"),g.Mc=1;if(g.V){h[d+"segmentNum"]=g.Ja;h[d+"segment"]=\ng.V;if(g.Ob>0)h[d+"segmentLength"]=g.Ob;g.Cc&&g.aa>0&&(h[d+"segmentView"]=!0)}if(!g.Af&&g.complete)h[d+"complete"]=!0,g.gg=1;if(g.Hb>0)h[d+"milestone"]=g.Hb;if(g.Ib>0)h[d+"offsetMilestone"]=g.Ib;if(l)for(f in h)Object.prototype[f]||(l+=",contextData."+f);b=h["a.contentType"];e.pe=a;e.pev3=b;var B,C;if(m.contextDataMapping){if(!e.events2)e.events2="";l&&(l+=",events");for(f in m.contextDataMapping)if(!Object.prototype[f]){a=f.length>d.length&&f.substring(0,d.length)==d?f.substring(d.length):"";b=m.contextDataMapping[f];\nif(typeof b=="string"){B=b.split(",");for(C=0;C<B.length;C++)b=B[C],f=="a.contentType"?(l&&(l+=","+b),e[b]=h[f]):a=="view"||a=="segmentView"||a=="clicked"||a=="complete"||a=="timePlayed"||a=="CPM"?(c&&(c+=","+b),a=="timePlayed"||a=="CPM"?h[f]&&(e.events2+=(e.events2?",":"")+b+"="+h[f]):h[f]&&(e.events2+=(e.events2?",":"")+b)):a=="segment"&&h[f+"Num"]?(l&&(l+=","+b),e[b]=h[f+"Num"]+":"+h[f]):(l&&(l+=","+b),e[b]=h[f])}else if(a=="milestones"||a=="offsetMilestones")f=f.substring(0,f.length-1),h[f]&&\nm.contextDataMapping[f+"s"][h[f]]&&(c&&(c+=","+m.contextDataMapping[f+"s"][h[f]]),e.events2+=(e.events2?",":"")+m.contextDataMapping[f+"s"][h[f]]);h[f]&&(h[f]=0);a=="segment"&&h[f+"Num"]&&(h[f+"Num"]=0)}}e.linkTrackVars=l;e.linkTrackEvents=c};m.ja=function(e,g,d,l,c){var a={},b=(new Date).getTime()/1E3,h,f,B=m.trackVars,C=m.trackEvents,F=m.trackSeconds,n=m.trackMilestones,q=m.trackOffsetMilestones,v=m.segmentByMilestones,p=m.segmentByOffsetMilestones,r,t,y=1,k={},G;if(!m.channel)m.channel=m.s.w.location.hostname;\nif(a=e&&m.list&&m.list[e]?m.list[e]:0){if(a.xa)F=m.adTrackSeconds,n=m.adTrackMilestones,q=m.adTrackOffsetMilestones,v=m.adSegmentByMilestones,p=m.adSegmentByOffsetMilestones;d<0&&(d=a.za==1&&a.wc>0?b-a.wc+a.T:a.T);a.length>0&&(d=d<a.length?d:a.length);d<0&&(d=0);a.xc=d;if(a.length>0)a.U=a.xc/a.length*100,a.U=a.U>100?100:a.U;if(a.T<0)a.T=d;G=a.Ic;k.name=e;k.ad=a.xa;k.length=a.length;k.openTime=new Date;k.openTime.setTime(a.timestamp*1E3);k.offset=a.xc;k.percent=a.U;k.playerName=a.playerName;k.mediaEvent=\na.ha<0?"OPEN":g==1?"PLAY":g==2?"STOP":g==3?"MONITOR":g==4?"TRACK":g==5?"COMPLETE":g==7?"CLICK":"CLOSE";if(g>2||g!=a.za&&(g!=2||a.za==1)){if(!c)l=a.Ja,c=a.V;if(g){if(g==1)a.T=d;if((g<=3||g>=5)&&a.ha>=0)if(y=!1,B=C="None",a.ha!=d){f=a.ha;if(f>d)f=a.T,f>d&&(f=d);r=n?n.split(","):0;if(a.length>0&&r&&d>=f)for(t=0;t<r.length;t++)if((h=r[t]?parseFloat(""+r[t]):0)&&f/a.length*100<h&&a.U>=h)y=!0,t=r.length,k.mediaEvent="MILESTONE",a.Hb=k.milestone=h;if((r=q?q.split(","):0)&&d>=f)for(t=0;t<r.length;t++)if((h=\nr[t]?parseFloat(""+r[t]):0)&&f<h&&d>=h)y=!0,t=r.length,k.mediaEvent="OFFSET_MILESTONE",a.Ib=k.offsetMilestone=h}if(a.Yd||!c){if(v&&n&&a.length>0){if(r=n.split(",")){r.push("100");for(t=f=0;t<r.length;t++)if(h=r[t]?parseFloat(""+r[t]):0){if(a.U<h)l=t+1,c="M:"+f+"-"+h,t=r.length;f=h}}}else if(p&&q&&(r=q.split(","))){r.push(""+(a.length>0?a.length:"E"));for(t=f=0;t<r.length;t++)if((h=r[t]?parseFloat(""+r[t]):0)||r[t]=="E"){if(d<h||r[t]=="E")l=t+1,c="O:"+f+"-"+h,t=r.length;f=h}}if(c)a.Yd=!0}if((c||a.V)&&\nc!=a.V){a.Kc=!0;if(!a.V)a.Ja=l,a.V=c;a.ha>=0&&(y=!0)}if((g>=2||a.U>=100)&&a.T<d)a.ce+=d-a.T,a.aa+=d-a.T;if(g<=2||g==3&&!a.za)a.Dc+=(g==1||g==3?"S":"E")+Math.floor(d),a.za=g==3?1:g;if(!y&&a.ha>=0&&g<=3&&(F=F?F:0)&&a.aa>=F)y=!0,k.mediaEvent="SECONDS";a.wc=b;a.T=d}if(!g||g<=3&&a.U>=100)a.za!=2&&(a.Dc+="E"+Math.floor(d)),g=0,B=C="None",k.mediaEvent="CLOSE";if(g==7)y=k.clicked=a.sc=!0;if(g==5||m.completeByCloseOffset&&(!g||a.U>=100)&&a.length>0&&d>=a.length-m.completeCloseOffsetThreshold)y=k.complete=\na.complete=!0;b=k.mediaEvent;b=="MILESTONE"?b+="_"+k.milestone:b=="OFFSET_MILESTONE"&&(b+="_"+k.offsetMilestone);a.Qd[b]?k.eventFirstTime=!1:(k.eventFirstTime=!0,a.Qd[b]=1);k.event=k.mediaEvent;k.timePlayed=a.ce;k.segmentNum=a.Ja;k.segment=a.V;k.segmentLength=a.Ob;m.monitor&&g!=4&&m.monitor(m.s,k);if(m.Heartbeat&&m.Heartbeat.enabled){k=[];k.push(a.name);if(!a.Ud)a.Ud=!0,k.push(a.length),k.push(a.playerName),a.xa?(k.push(a.Jb),k.push(a.yc),k.push(a.zc),k.push(a.na),m.Heartbeat.callMethodWhenReady("openAd",\nk)):m.Heartbeat.callMethodWhenReady("open",k),k=[],k.push(a.name);g==0?m.Heartbeat.callMethodWhenReady("close",k):(k.push(d),g==1?(k.push(a.Ja),k.push(a.V),k.push(a.Ob),m.Heartbeat.callMethodWhenReady("play",k)):g==2?m.Heartbeat.callMethodWhenReady("stop",k):g==3?m.Heartbeat.callMethodWhenReady("monitor",k):g==5?m.Heartbeat.callMethodWhenReady("complete",k):g==7&&m.Heartbeat.callMethodWhenReady("click",k));a.ha>=0&&(y=!1)}g==0&&m.Oe(e);if(y&&a.Ic==G){e={};e.contextData={};e.linkTrackVars=B;e.linkTrackEvents=\nC;if(!e.linkTrackVars)e.linkTrackVars="";if(!e.linkTrackEvents)e.linkTrackEvents="";m.yf(e,a);e.linkTrackVars||(e["!linkTrackVars"]=1);e.linkTrackEvents||(e["!linkTrackEvents"]=1);m.s.track(e);if(a.Kc)a.Ja=l,a.V=c,a.Cc=!0,a.Kc=!1;else if(a.aa>0)a.Cc=!1;a.Dc="";a.Hb=a.Ib=0;a.aa-=Math.floor(a.aa);a.ha=d;a.Ic++}}}return a};m.vf=function(e,g,d,l,c){var a=0;if(e&&(!m.autoTrackMediaLengthRequired||g&&g>0)){if(!m.list||!m.list[e]){if(d==1||d==3)m.open(e,g,"HTML5 Video",c),a=1}else a=1;a&&m.ja(e,d,l,-1,0)}};\nm.attach=function(e){var g,d,l;if(e&&e.tagName&&e.tagName.toUpperCase()=="VIDEO"){if(!m.hb)m.hb=function(c,a,b){var h,f;if(m.autoTrack){h=c.currentSrc;(f=c.duration)||(f=-1);if(b<0)b=c.currentTime;m.vf(h,f,a,b,c)}};g=function(){m.hb(e,1,-1)};d=function(){m.hb(e,1,-1)};m.ra(e,"play",g);m.ra(e,"pause",d);m.ra(e,"seeking",d);m.ra(e,"seeked",g);m.ra(e,"ended",function(){m.hb(e,0,-1)});m.ra(e,"timeupdate",g);l=function(){!e.paused&&!e.ended&&!e.seeking&&m.hb(e,3,-1);setTimeout(l,1E3)};l()}};m.ra=function(m,\ng,d){m.attachEvent?m.attachEvent("on"+g,d):m.addEventListener&&m.addEventListener(g,d,!1)};if(m.completeByCloseOffset==void 0)m.completeByCloseOffset=1;if(m.completeCloseOffsetThreshold==void 0)m.completeCloseOffsetThreshold=1;var D=new function(m){this.Je=function(g){this.s=g;this.enabled=!1;this.v=new this.Ka.Of.ne(g)};this.open=function(g,d,m){this.v.open(g,d,m)};this.openAd=function(g,d,m,c,a,b,h){this.v.openAd(g,d,m,c,a,b,h)};this.close=function(g){this.v.close(g)};this.play=function(g,d,m,c,\na){this.v.play(g,d,m,c,a)};this.monitor=function(g,m){this.v.monitor(g,m)};this.stop=function(g,m){this.v.stop(g,m)};this.click=function(g,m){this.v.click(g,m)};this.complete=function(g,m){this.v.complete(g,m)};this.setup=function(g){this.v.Wf(g)};this.bufferStart=function(){this.v.xf()};this.bitrateChange=function(g){this.v.wf(g)};this.updateQoSInfo=function(g,m,e){this.v.bg(g,m,e)};this.adBreakStart=function(m){this.v.sf(m)};this.adBreakEnd=function(){this.v.rf()};this.trackError=function(m,d,e){this.v.$f(m,\nd,e)};this.sessionComplete=function(){this.v.Uf()};this.__setPsdkVersion=function(m){this.v.Ke(m)};(function(m){if(typeof d==="undefined")var d={};if(typeof e==="undefined")var e={};e.event||(e.event={});e.a||(e.a={});e.H||(e.H={});e.bb||(e.bb={});e.M||(e.M={});(function(c){c.extend=function(a,b){function c(){this.constructor=a}for(var f in b)b.hasOwnProperty(f)&&(a[f]=b[f]);c.prototype=b.prototype;a.prototype=new c;a.r=b.prototype;return a}})(d);(function(c){c.Q=function(a,b){var c=[],f;for(f in b)b.hasOwnProperty(f)&&\ntypeof b[f]==="function"&&c.push(f);for(f=0;f<c.length;f++){var B=c[f];a.prototype[B]=b[B]}}})(d);(function(c){c.Md={Pd:function(){this.ea&&this.ea.apply(this,arguments);this.ea=j}}})(d);(function(c){c.Oa=!1;c.P={N:function(a){this.Ua=!0;this.Ab=a},jg:function(){this.Ua=!1},log:function(a){c.Oa&&this.Ua&&window.console&&window.console.log&&window.console.log(this.Ab+a)},info:function(a){c.Oa&&this.Ua&&window.console&&window.console.info&&window.console.info(this.Ab+a)},warn:function(a){c.Oa&&this.Ua&&\nwindow.console&&window.console.warn&&window.console.warn(this.Ab+a)},error:function(a){if(c.Oa&&this.Ua&&window.console&&window.console.error)throw a=this.Ab+a,window.console.error(a),Error(a);}}})(d);(function(c){function a(a,c){this.type=a;this.data=c}a.wb="success";a.Xb="error";c.S=a})(d);(function(c){function a(){this.F={}}a.prototype.addEventListener=function(a,c,f){a&&c&&(this.F[a]=this.F[a]||[],this.F[a].push({zf:c,Ld:f||window}))};a.prototype.dispatchEvent=function(a){if(a.type)for(var c in this.F)if(this.F.hasOwnProperty(c)&&\na.type===c){var f=this.F[c];for(c=0;c<f.length;c++)f[c].zf.call(f[c].Ld,a);break}};a.prototype.eb=function(a){if(a){var c,f;for(f in this.F)if(this.F.hasOwnProperty(f)){for(c=this.F[f].length-1;c>=0;c--)this.F[f][c].Ld===a&&this.F[f].splice(c,1);this.F[f].length||delete this.F[f]}}else this.F={}};c.kd=a})(d);(function(c){function a(){if(!a.prototype.Ta)a.prototype.Ta=new b;return a.prototype.Ta}var b=c.kd;c.ca=a})(d);(function(c){function a(){}function b(){b.r.constructor.call(this)}var h=c.S,f=c.kd;\na.ld="GET";c.extend(b,f);b.prototype.Cf=function(a){a.I=new window.XMLHttpRequest;if(!("withCredentials"in a.I)&&(a.I=typeof window.XDomainRequest!=="undefined"?new window.XDomainRequest:j,a.I)){var f=this;a.I.onreadystatechange=function(){if(a.I.readyState===4){var m={};m[b.pd]=a.I.status;a.I.status>=200&&a.I.status<400?(m[b.od]=a.I.responseText,m[b.qb]=f,f.dispatchEvent(new c.S(h.wb,m))):f.dispatchEvent(new c.S(h.Xb,m))}}}};b.Eb=j;b.prototype.Mf=function(a){if(!b.Eb)b.Eb=new Image,b.Eb.alt="";b.Eb.src=\na.url;a={};a[b.pd]=200;a[b.od]="";a[b.qb]=this;this.dispatchEvent(new c.S(h.wb,a))};b.prototype.close=function(){this.eb()};b.prototype.load=function(a){a&&a.method&&a.url&&(this.Cf(a),a.I?(a.I.open(a.method,a.url,!0),a.I.send()):this.Mf(a))};b.pd="status";b.od="response";b.qb="instance";c.Ie=a;c.He=function(a,b){this.url=a||j;this.method=b;this.I=j};c.Ge=b})(d);(function(c,a){function b(){}b.Fa="report";b.qa="what";b.Ga="reset";b.Sb="account";b.cc="sc_tracking_server";b.xb="tracking_server";b.lb=\n"check_status_server";b.rb="job_id";b.Pa="publisher";b.fc="stream_type";b.$b="ovp";b.ec="sdk";b.bd="channel";b.nb="debug_tracking";b.yb="track_local";b.Ha="visitor_id";b.Aa="analytics_visitor_id";b.Da="marketing_cloud_visitor_id";b.i="name";b.Ca="length";b.Ea="player_name";b.X="timer_interval";b.rd="tracking_interval";b.cd="check_status_interval";b.gc="track_external_errors";b.ac="parent_name";b.nd="parent_pod";b.bc="parent_pod_position";b.ub="parent_pod_offset";b.na="parent_pod_cpm";b.B="offset";\nb.vb="source";b.Yb="error_id";b.kb="bitrate";b.Zb="fps";b.Vb="dropped_frames";a.event.ba=b})(d,e);(function(c,a){function b(a,f){b.r.constructor.call(this,a,f)}c.extend(b,c.S);b.La="api_destroy";b.Tb="api_config";b.Uc="api_open_main";b.Tc="api_open_ad";b.Qc="api_close";b.Vc="api_play";b.Sc="api_monitor";b.Yc="api_stop";b.Pc="api_click";b.Rc="api_complete";b.Zc="api_track_error";b.Wc="api_qos_info";b.Nc="api_bitrate_change";b.Oc="api_buffer_start";b.Ub="api_pod_offset";b.Xc="api_session_complete";\na.event.Ma=b})(d,e);(function(c,a){function b(a,f){b.r.constructor.call(this,a,f)}c.extend(b,c.S);b.Ba="context_data_available";a.event.dd=b})(d,e);(function(c,a){function b(a,f){b.r.constructor.call(this,a,f)}c.extend(b,c.S);b.oa="data_request";b.mb="data_response";b.ya={Qa:"tracking_timer_interval",md:"main_video_publisher"};a.event.Wb=b})(d,e);(function(c,a){function b(a,f){b.r.constructor.call(this,a,f)}c.extend(b,c.S);b.sb="network_check_status_complete";a.event.tb=b})(d,e);(function(c,a){function b(a,\nf){b.r.constructor.call(this,a,f)}c.extend(b,c.S);b.CLOCK_TRACKING_TICK="CLOCK_TRACKING_TICK";b.CLOCK_TRACKING_ENABLE="CLOCK_TRACKING_ENABLE";b.CLOCK_TRACKING_DISABLE="CLOCK_TRACKING_DISABLE";b.CLOCK_CHECK_STATUS_TICK="CLOCK_CHECK_STATUS_TICK";b.CLOCK_CHECK_STATUS_ENABLE="CLOCK_CHECK_STATUS_ENABLE";b.CLOCK_CHECK_STATUS_DISABLE="CLOCK_CHECK_STATUS_DISABLE";a.event.Na=b})(d,e);(function(c,a){function b(a,b){this.value=a;this.hint=b}function h(a){this.Bc=a;this.data={}}b.pa="short";h.prototype.c=function(a,\nb,c){var h=this;return function(){arguments.length&&(h[a]=arguments[0],h.Pb(b,arguments[0],c));return h[a]}};h.prototype.Pb=function(a,c,h){this.data[a]=new b(c,h)};a.a.R=h;a.a.ed=b})(d,e);(function(c,a){function b(a,c){b.r.constructor.call(this,a);this.cg=this.c("_year","year",h.pa);this.Qf=this.c("_month","month",h.pa);this.Df=this.c("_day","day",h.pa);this.Kf=this.c("_hour","hour",h.pa);this.Pf=this.c("_minute","minute",h.pa);this.Sf=this.c("_second","second",h.pa);this.cg(c.getUTCFullYear());\nthis.Qf(c.getUTCMonth()+1);this.Df(c.getUTCDate());this.Kf(c.getUTCHours());this.Pf(c.getUTCMinutes());this.Sf(c.getUTCSeconds())}var h=a.a.ed;c.extend(b,a.a.R);a.a.he=b})(d,e);(function(c,a){function b(){b.r.constructor.call(this,"asset");this.Fb=this.c("_cpm","cpm",j);this.L=this.c("_adId","ad_id",j);this.Nb=this.c("_resolver","resolver",j);this.Kb=this.c("_podId","pod_id",j);this.Lb=this.c("_podPosition","pod_position",j);this.Mb=this.c("_podSecond","pod_second",j);this.length=this.c("_length",\n"length",j);this.Fb("");this.L("");this.Nb("");this.Kb("");this.Lb("");this.Mb(0);this.length(0);if(arguments.length&&arguments[0]instanceof b){var a=arguments[0];this.Fb(a.Fb());this.L(a.L());this.Nb(a.Nb());this.Kb(a.Kb());this.Lb(a.Lb());this.Mb(a.Mb());this.length(a.length())}}c.extend(b,a.a.R);a.a.$c=b})(d,e);(function(c,a){function b(){b.r.constructor.call(this,"asset");this.type=this.c("_type","type",j);this.k=this.c("_videoId","video_id",j);this.K=this.c("_publisher","publisher",j);this.q=\nthis.c("_adData","ad_data",j);this.duration=this.c("_duration","duration",j);this.type("");this.k("");this.K("");this.q(j);this.duration(0);if(arguments.length&&arguments[0]instanceof b){var a=arguments[0];this.type(a.type());this.k(a.k());this.K(a.K());this.duration(a.duration());(a=a.q())&&this.q(new h(a))}}var h=a.a.$c;c.extend(b,a.a.R);b.sd="vod";b.Ce="live";b.Be="linear";b.Ra="ad";a.a.jb=b})(d,e);(function(c,a){function b(a){b.r.constructor.call(this,"event");this.pf=a;this.type=this.c("_type",\n"type",j);this.count=this.c("_count","count",j);this.Gc=this.c("_totalCount","total_count",j);this.duration=this.c("_duration","duration",j);this.Hc=this.c("_totalDuration","total_duration",j);this.ka=this.c("_playhead","playhead",j);this.id=this.c("_id","id",j);this.source=this.c("_source","source",j);this.Ac=this.c("_prevTs","prev_ts",j);this.qf=function(){var a=this.pf*1E3;this.pc=new Date(Math.floor(this.oc/a)*a);this.Pb("ts_as_date",new h(this.Bc,this.pc),j)};this.Rb=function(){if(arguments.length)this.oc=\narguments[0],this.Pb("ts",this.oc,j),this.qf();return this.oc};this.ag=function(){if(arguments.length)this.pc=arguments[0],this.Pb("ts_as_date",new h(this.Bc,this.pc),j)};this.type("");this.count(0);this.Gc(0);this.duration(0);this.Hc(0);this.ka(0);this.id("");this.source("");this.Ac(-1);this.Rb((new Date).getTime())}var h=a.a.he;c.extend(b,a.a.R);b.le="load";b.me="unload";b.ob="start";b.jd="play";b.hd="pause";b.je="buffer";b.ie="bitrate_change";b.ke="error";b.fd="active";b.gd="complete";a.a.pb=b})(d,\ne);(function(c,a){function b(){b.r.constructor.call(this,"stream");this.qc=this.c("_bitrate","bitrate",j);this.Rd=this.c("_fps","fps",j);this.Od=this.c("_droppedFrames","dropped_frames",j);this.qc(0);this.Rd(0);this.Od(0)}c.extend(b,a.a.R);a.a.re=b})(d,e);(function(c,a){function b(){b.r.constructor.call(this,"sc");this.Xd=this.c("_reportSuiteId","rsid",j);this.trackingServer=this.c("_trackingServer","tracking_server",j);this.Xd("");this.trackingServer("")}c.extend(b,a.a.R);a.a.Ae=b})(d,e);(function(c,\na){function b(){b.r.constructor.call(this,"sp");this.ia=this.c("_ovp","ovp",j);this.la=this.c("_sdk","sdk",j);this.channel=this.c("_channel","channel",j);this.playerName=this.c("_playerName","player_name",j);this.ia("unknown");this.la("unknown");this.channel("unknown");this.playerName("")}c.extend(b,a.a.R);a.a.xe=b})(d,e);(function(c,a){function b(){b.r.constructor.call(this,"event");this.Ec=this.c("_sessionId","sid",j);this.Ec("")}c.extend(b,a.a.R);a.a.ye=b})(d,e);(function(c,a){function b(){b.r.constructor.call(this,\n"stream");this.rc=this.c("_cdn","cdn",j);this.name=this.c("_name","name",j);this.rc("");this.name("");if(arguments.length&&arguments[0]instanceof b){var a=arguments[0];this.rc(a.rc());this.name(a.name())}}c.extend(b,a.a.R);a.a.qd=b})(d,e);(function(c,a){function b(){b.r.constructor.call(this,"user");this.uc=this.c("_device","device",j);this.country=this.c("_country","country",j);this.city=this.c("_city","city",j);this.latitude=this.c("_latitude","latitude",j);this.longitude=this.c("_longitude","longitude",\nj);this.ib=this.c("_visitorId","id",j);this.$a=this.c("_analyticsVisitorId","aid",j);this.ab=this.c("_marketingCloudVisitorId","mid",j);this.uc("");this.country("");this.city("");this.latitude("");this.longitude("");this.ib("");this.$a("");this.ab("");if(arguments.length&&arguments[0]instanceof b){var a=arguments[0];this.uc(a.uc());this.country(a.country());this.city(a.city());this.latitude(a.latitude());this.longitude(a.longitude());this.ib(a.ib());this.$a(a.$a());this.ab(a.ab())}}c.extend(b,a.a.R);\na.a.vd=b})(d,e);(function(c,a){a.a.ue=function(a,c,f,m,e){this.ga=a;this.g=c;this.Lc=f;this.Fc=m;this.cb=e}})(d,e);(function(c,a){var b=a.a.pb;a.a.te=function(a,c,m){this.Xf=a;this.Tf=c;this.Vf=m;this.G=[];this.Za=function(a){this.G.push(a)};this.lg=function(){return this.G};this.Gf=function(){if(this.G.length)for(var a=this.G.length-1;a>=0;a--)this.G[a].ga.type()===b.hd&&this.G.splice(a,1)}}})(d,e);(function(c,a){function b(){}b.prototype.$d=E();b.prototype.ae=E();b.prototype.W=E();b.prototype.Zd=\nE();b.prototype.be=E();a.a.we=b})(d,e);(function(c,a){function b(){this.N("[media-fork::QuerystringSerializer] > ");this.da=function(a){return a?a+"&":""};this.Gd=function(a){a&&a.length>0&&(a=a.substring(0,a.length-1));return a};this.mf=function(a){var b=[],c;for(c in a.data)if(a.data.hasOwnProperty(c)){var f=a.data[c],p=f.value;f=f.hint;var m=j,h=a.Bc;p===j||typeof p==="undefined"||(typeof p==="number"?m=this.Zd(c,p,h,f):typeof p==="string"?m=this.be(c,p,h,f):p instanceof e?m=this.W(p):this.warn("#_processDao() > Unable to serialize DAO. Field: "+\nc+". Value: "+p+"."),m&&b.push(m))}return b}}var m=c.Q,f=c.P,e=a.a.R,g=a.a.ed;c.extend(b,a.a.we);m(b,f);b.prototype.$d=function(a){for(var b=[],c=a.G,f=0;f<c.length;f++){var p=this.ae(c[f])+"&";p+=this.da(this.W(a.Xf));p+=this.da(this.W(a.Tf));p+=this.da(this.W(a.Vf));p=this.Gd(p);b.push(p)}return b};b.prototype.ae=function(a){var b=this.da(this.W(a.ga));b+=this.da(this.W(a.g));b+=this.da(this.W(a.Lc));b+=this.da(this.W(a.Fc));b+=this.da(this.W(a.cb));return b=this.Gd(b)};b.prototype.W=function(a){a=\nthis.mf(a);for(var b="",c=0;c<a.length;c++)b+=c==a.length-1?a[c]:a[c]+"&";return b};b.prototype.Zd=function(a,b,c,f){var p="l";if(b!=j&&b!==void 0&&!isNaN(b))return f&&typeof f==="string"&&f===g.pa&&(p="h"),p+":"+c+":"+a+"="+b;return j};b.prototype.be=function(a,b,c){if(b)return"s:"+c+":"+a+"="+window.encodeURIComponent(b);return j};a.a.se=b})(d,e);(function(c,a){function b(a){this.Qb=0;this.O=a;this.Gb=!1}function m(){if(m.prototype.Ta)return m.prototype.Ta;var a=this;this.N("[media-fork::TimerManager] > ");\nthis.Ad=0;this.fa={};this.ua=function(){this.log("#_onApiDestroy()");clearInterval(this.yd);n().eb(this)};this.jf=function(){this.log("#_onTick() > ------------------- ("+this.Ad+")");this.Ad++;for(var a in this.fa)if(this.fa.hasOwnProperty(a)){var b=this.fa[a];if(b.Gb&&(b.Qb++,b.Qb%b.O===0)){var c={};c[d.X]=b.O;n().dispatchEvent(new g(g[a],c))}}};n().addEventListener(e.La,this.ua,this);this.yd=setInterval(function(){a.jf()},q*1E3);this.Lf=function(a){return(a=this.fa[a])&&a.Gb};this.Kd=function(a,\nc){this.fa[a]=new b(c)};this.Ef=function(a){delete this.fa[a]};this.Yf=function(a,b){this.log("#startTimer(name="+a+", reset="+b+")");var c=this.fa[a];if(c&&(c.Gb=!0,b))this.log("Resseting timer: "+a),c.Qb=0};this.Zf=function(a,b){this.log("#startTimer(name="+a+", reset="+b+")");var c=this.fa[a];if(c&&(c.Gb=!1,b))this.log("Resseting timer: "+a),c.Qb=0};m.prototype.Ta=this}var f=c.Q,e=a.event.Ma,g=a.event.Na,d=a.event.ba,n=c.ca,q=1;f(m,c.P);new m;a.M.Ee=m})(d,e);(function(c,a){function b(a,b,c,m){this.N("[media-fork::Timer] > ");\nthis.O=m;this.ma=a;this.Hf=b;this.Ff=c;g().Kd(this.ma,this.O);this.ua=function(){this.Nd()};this.lf=function(a){a=a.data;var b=!1;a&&a.hasOwnProperty(d.Ga)&&(b=a[d.Ga]);this.start(b)};this.kf=function(a){a=a.data;var b=!1;a&&a.hasOwnProperty(d.Ga)&&(b=a[d.Ga]);this.stop(b)};f().addEventListener(e.La,this.ua,this);f().addEventListener(this.Hf,this.lf,this);f().addEventListener(this.Ff,this.kf,this)}var m=c.Q,f=c.ca,e=a.event.Ma,g=a.M.Ee,d=a.event.ba;m(b,c.P);b.prototype.start=function(a){this.log("#start("+\nthis.ma+")");g().Yf(this.ma,a)};b.prototype.stop=function(a){this.log("#stop("+this.ma+")");g().Zf(this.ma,a)};b.prototype.Nd=function(){f().eb(this);g().Ef(this.ma)};b.prototype.setInterval=function(a){var b=g().Lf(this.ma);this.stop(!0);this.O=a;g().Kd(this.ma,this.O);b&&this.start(!0)};a.M.ud=b})(d,e);(function(c,a){function b(){this.N("[media-fork::TrackingTimer] > ");b.r.constructor.call(this,n.CLOCK_TRACKING_TICK,n.CLOCK_TRACKING_ENABLE,n.CLOCK_TRACKING_DISABLE,v);this.kc=function(a){a=a.data[q.rd];\nthis.log("#_onCheckStatusComplete(interval="+a+")");a?a===this.O?this.log("#_onCheckStatusComplete() > Interval value not changed."):(this.log("#_onCheckStatusComplete() > Interval changed to: "+a),this.setInterval(a)):(this.warn("#_onCheckStatusComplete() > Invalid interval value."),this.setInterval(v))};this.lc=function(a){a=a.data[q.qa];this.log("#_onDataRequest(what="+a+")");switch(a){case g.ya.Qa:a={},a[q.X]=this.O,e().dispatchEvent(new g(g.mb,a))}};e().addEventListener(d.sb,this.kc,this);e().addEventListener(g.oa,\nthis.lc,this)}var m=c.Q,f=c.P,e=c.ca,g=a.event.Wb,d=a.event.tb,n=a.event.Na,q=a.event.ba,v=10;c.extend(b,a.M.ud);m(b,f);a.M.Fe=b})(d,e);(function(c,a){function b(){this.N("[media-fork::CheckStatusTimer] > ");b.r.constructor.call(this,v.CLOCK_CHECK_STATUS_TICK,v.CLOCK_CHECK_STATUS_ENABLE,v.CLOCK_CHECK_STATUS_DISABLE,f);var a=this;setTimeout(function(){a.Qe()},200);this.Qe=function(){this.log("#_initialCheck()");var a={};a[q.X]=this.O;d().dispatchEvent(new v(v.CLOCK_CHECK_STATUS_TICK,a))};this.kc=function(a){a=\na.data[q.cd];this.log("#_onCheckStatusComplete(interval="+a+")");a?a===this.O?this.log("#_onCheckStatusComplete() > Interval value not changed."):a>m?(this.warn("#_onCheckStatusComplete() > Interval value too large: "+a),this.setInterval(m)):(this.log("#_onCheckStatusComplete() > Interval changed to: "+a),this.setInterval(a)):(this.warn("#_onCheckStatusComplete() > Invalid interval value."),this.setInterval(f))};d().addEventListener(n.sb,this.kc,this)}var m=600,f=60,e=c.Q,g=c.P,d=c.ca,n=a.event.tb,\nq=a.event.ba,v=a.event.Na;c.extend(b,a.M.ud);e(b,g);a.M.de=b})(d,e);(function(c,a){var b=a.M.de,m=a.M.Fe;a.M.ee=function(){this.dg=new b;this.fg=new m}})(d,e);(function(c,a){function b(a){this.N("[media-fork::SettingsParser] > ");this.Bd=a;this.log("#SettingsParser(data="+a+")")}var m=c.Q,f=c.ca,e=a.event.ba,g=a.event.tb;m(b,c.P);b.prototype.parse=function(){var a,b,c,m;if(this.Bd){window.DOMParser?m=(new window.DOMParser).parseFromString(this.Bd,"text/xml"):(m=new window.ActiveXObject("Microsoft.XMLDOM"),\nm.async=!1,m.loadXML(this.data));var p;(p=parseInt(m.getElementsByTagName("trackingInterval")[0].childNodes[0].nodeValue,10))&&(a=p);(p=parseInt(m.getElementsByTagName("setupCheckInterval")[0].childNodes[0].nodeValue,10))&&(b=p);(p=parseInt(m.getElementsByTagName("trackExternalErrors")[0].childNodes[0].nodeValue,10))&&(c=p===1);m={};m[e.rd]=a;m[e.cd]=b;m[e.gc]=c;this.log("#parse() > Obtained configuration settings: "+JSON.stringify(m));f().dispatchEvent(new g(g.sb,m))}else this.warn("#SettingsParser() > No data available for parsing.")};\na.bb.ze=b})(d,e);(function(c,a){function b(a){this.N("[media-fork::Network] > ");this.ta=this.Hd=this.Cd=!1;this.of=a;this.Ed=this.xd=this.Id=j;this.jc=function(a){a=a.data;this.log("#_onApiConfig(sb_server="+a[p.xb]+", check_status_server="+a[p.lb]+", job_id="+a[p.rb]+", debug_tracking="+a[p.nb]+", track_local="+a[p.yb]+")");this.Id=a[p.xb];this.xd=a[p.lb];this.Ed=a[p.rb];this.Cd=a[p.nb];this.Hd=a[p.yb];this.ta=!0};this.ua=function(){this.log("#_onApiDestroy()");g().eb(this)};this.gf=function(a){if(this.ta){if(a=\nthis.of.$d(a.data[p.Fa]),!this.Hd)for(var b=0;b<a.length;b++){var c=new l(this.Id+"/?__job_id="+this.Ed+"&"+a[b],n.ld);this.Cd&&window.console&&window.console.info&&window.console.info(c.method+" : "+c.url);(function(a,b){a.addEventListener(e.wb,function(){a.close()},this);a.addEventListener(e.Xb,function(c){b.warn("#_onContextDataAvailable() > Failed to send heartbeat report: "+JSON.stringify(c));a.close()},this);a.load(c)})(new v,this)}}else this.warn("#_onContextDataAvailable() > Unable to send request: not configured.")};\nthis.ef=function(){function a(b){b.data&&(new k(b.data.response)).parse();b.data[v.qb].close()}function b(a){c.warn("_onClockCheckStatusTick() > Failed to obtain the config. settings: "+JSON.stringify(a));a.data[v.qb].close()}if(this.ta){var c=this;this.ea=function(c){if(c=c[p.Pa]){c=c.replace(/[^a-zA-Z0-9]+/,"-").toLocaleLowerCase();c=this.xd+c+".xml?r="+(new Date).getTime();var f=new l(c,n.ld),m=new v;m.addEventListener(e.wb,a,this);m.addEventListener(e.Xb,b,this);this.log("#_onClockCheckStatusTick() > Get new settings from: "+\nc);m.load(f)}else this.warn("#_onClockCheckStatusTick() > Publisher is NULL.")};var f={};f[p.qa]=r.ya.md;g().dispatchEvent(new r(r.oa,f))}else this.warn("#_onClockCheckStatusTick() > Unable to send request: not configured.")};this.mc=function(a){this.Pd(a.data)};g().addEventListener(r.mb,this.mc,this);g().addEventListener(d.Tb,this.jc,this);g().addEventListener(d.La,this.ua,this);g().addEventListener(y.Ba,this.gf,this);g().addEventListener(t.CLOCK_CHECK_STATUS_TICK,this.ef,this)}var m=c.Q,f=c.P,e=\nc.S,g=c.ca,d=a.event.Ma,n=c.Ie,l=c.He,v=c.Ge,p=a.event.ba,r=a.event.Wb,t=a.event.Na,y=a.event.dd,k=a.bb.ze;m(b,c.Md);m(b,f);a.bb.qe=b})(d,e);(function(c,a){function b(){this.N("[media-fork::Counters] > ");this.va={};this.wa={};this.Sd=function(a,b,c){a=b+"."+c+"."+a;this.va[a]||(this.va[a]=0);this.log("#getTotalCount(key="+a+")");return this.va[a]};this.mg=function(a,b,c){a=b+"."+c+"."+a;this.log("#resetTotalCount(key="+a+")");this.va[a]=0};this.Wd=function(a,b,c){a=b+"."+c+"."+a;this.va[a]||(this.va[a]=\n0);this.log("#incrementTotalCount(key="+a+")");this.va[a]++};this.Td=function(a,b,c){a=b+"."+c+"."+a;this.wa[a]||(this.wa[a]=0);this.log("#getTotalDuration(key="+a+")");return this.wa[a]};this.ng=function(a,b,c){a=b+"."+c+"."+a;this.log("#resetTotalDuration(key="+a+")");this.wa[a]=0};this.Vd=function(a,b,c,m){a=b+"."+c+"."+a;this.wa[a]||(this.wa[a]=0);this.log("#increaseTotalDuration(key="+a+", amount="+m+")");this.wa[a]+=m}}var m=c.Q;m(b,c.P);a.H.ge=b})(d,e);(function(c,a){function b(){this.N("[media-fork::History] > ");\nthis.Dd={};this.zd=function(a){var b=a.g;return(b.q()?b.q().L():b.k())+"."+b.type()+"."+a.$};this.gb=function(a){var b=this.zd(a);this.log("#updateWith(key="+b+")");this.Dd[b]=a};this.J=function(a){a=this.zd(a);this.log("#getPreviousItemOfSameTypeWith(key="+a+")");return this.Dd[a]}}var m=c.Q;m(b,c.P);a.H.oe=b})(d,e);(function(c,a){var b=a.a.pb,m=a.a.jb,f=a.a.vd,e=a.a.qd;a.H.td=function(a,c,g,d,v,p){this.timestamp=new Date;this.g=new m(a);this.Lc=new f(c);this.Fc=new e(g);this.$=v;this.cb=d;this.ka=\np;this.A=void 0;this.If=function(){if(this.$===b.fd)return this.g.k();return this.g.type()===m.Ra?this.g.q().L():this.g.k()};this.kg=function(){return 1}}})(d,e);(function(c,a){a.H.De=function(){this.Z=[];this.Jf=function(){return this.Z.slice()};this.tf=function(a){for(var c=-1,m=this.Z.length-1;m>=0;m--){if(a.timestamp>=this.Z[m].timestamp)break;c=m}c>0?this.Z.splice(m,0,a):this.Z.push(a)}}})(d,e);(function(c,a){function b(a){this.N("[media-fork::ReporterHelper] > ");this.j=a;this.Fd=j;this.Ne=\nfunction(a,b,c){c*=1E3;a=a.getTime()===0?b.getTime()-c/2:a.getTime()/2+b.getTime()/2;a=Math.floor(a/c)*c;this.Fd==a&&(a+=c);this.Fd=a;return new Date(a)};this.Sa=function(a,b,c){var m=this.j.ic,f=a.$,g=a.If(),h=a.g.type(),n=f===e.ob?0:a.ka;m.Wd(f,g,h);m.Vd(f,g,h,b);c=new e(c);c.type(f);c.count(1);c.duration(b);c.Gc(m.Sd(f,g,h));c.Hc(m.Td(f,g,h));c.ka(n);c.Rb(a.timestamp.getTime());c.Ac(a.A?a.A.timestamp.getTime():-1);return new d(c,a.g,a.Lc,a.Fc,a.cb)};this.wd=function(a,b,c){if(a.G.length){var m=\nnew g(this.j.e);m.type(this.j.Bb);m.q(j);m=new n(m,this.j.n,this.j.C,this.j.u,e.fd,this.j.p[this.j.e.k()]);m.A=this.j.o.J(m);this.j.o.gb(m);a.Za(this.Sa(m,b*1E3,b));if(c!=j)for(b=0;b<a.G.length;b++)a.G[b].ga.ag(c)}};this.Xa=function(a,b){return b.getTime()-a.getTime()};this.tc=function(a,b,c){var m=new f(this.j.Db,this.j.Ia,this.j.Wa);m.Za(this.Sa(a,0,b));c&&this.wd(m,b,j);return m};this.Jd=function(a,b,c){var m,d,h=new f(this.j.Db,this.j.Ia,this.j.Wa),n=this.j.Z.Jf(),l=[];d=j;var o=0;for(o=m=0;o<\nn.length;o++)m=n[o],m.timestamp>a&&m.timestamp<=b&&l.push(m),m.timestamp<=a&&(d=m);this.log("#createReportForQuantum() > -------------TRACK REPORT----------------");this.log("#createReportForQuantum() > Interval: ["+a.getTime()+" , "+b.getTime()+"]. Tracking interval: "+c);this.log("#createReportForQuantum() > -----------------------------------------");for(o=0;o<n.length;o++)this.log("#createReportForQuantum() > ["+n[o].timestamp.getTime()+"] :"+n[o].$+" | "+n[o].g.type());this.log("#createReportForQuantum() > -----------------------------------------");\nfor(o=0;o<l.length;o++)this.log("#createReportForQuantum() > ["+l[o].timestamp.getTime()+"] :"+l[o].$+" | "+l[o].g.type());this.log("#createReportForQuantum() > -----------------------------------------");if(d){if(d.A)d.A.timestamp=d.timestamp;d.timestamp=new Date(a.getTime()+1);m=d.g.q()?d.g.q().L():d.g.k();d.ka=this.j.p[m]}if(l.length){n=0;d&&(n=d.$===e.ob&&d.g.type()!==g.Ra?this.Xa(d.timestamp,l[0].timestamp):this.Xa(a,l[0].timestamp),h.Za(this.Sa(d,n,c)));for(o=0;o<l.length;o++){var q=l[o];n=\no==l.length-1?this.Xa(q.timestamp,b):this.Xa(q.timestamp,l[o+1].timestamp);var x=!1,u=h.G;for(m=0;m<u.length;m++)if(d=u[m],q.g.type()===d.g.type()&&q.$===d.ga.type()&&(x=q.g.type()===g.Ra?d.g.q().L()===q.g.q().L():d.g.k()===q.g.k()),x){u=d.ga;var z=d.g.type();m=d.g.q()?d.g.q().L():d.g.k();var i=this.j.ic;i.Wd(u.type(),m,z);i.Vd(u.type(),m,z,n);d.cb=q.cb;u.ka(this.j.p[m]);u.duration(u.duration()+n);u.Gc(i.Sd(u.type(),m,z));u.Hc(i.Td(u.type(),m,z));u.Rb(q.timestamp.getTime());break}if(!x)this.log("#createReportForQuantum() > Adding event to report: "+\nq.$),m=q.g.q()?q.g.q().L():q.g.k(),q.ka=this.j.p[m],h.Za(this.Sa(q,n,c))}}else d&&h.Za(this.Sa(d,this.Xa(a,b),c));h.Gf();o=this.Ne(a,b,c);this.wd(h,c,o);this.log("#createReportForQuantum() > Final report ----- START -----");for(o=0;o<h.G.length;o++)d=h.G[o],c=d.ga,m=d.g.q()?d.g.q().L():d.g.k(),this.log("#createReportForQuantum() > Final report ["+c.Rb()+"/"+c.Ac()+"] :"+c.type()+" | type="+d.g.type()+", name="+m+", duration="+c.duration()+", playhead="+c.ka());this.log("#createReportForQuantum() > Final report ----- END -----");\nreturn h}}var m=c.Q,f=a.a.te,e=a.a.pb,g=a.a.jb,d=a.a.ue,n=a.H.td;m(b,c.P);a.H.ve=b})(d,e);(function(c,a){function b(){this.N("[media-fork::Context] > ");this.zb=this.z=!1;this.Bb=j;this.hc=!1;this.l=this.Cb=j;this.f={fb:j,K:j};this.Ya=this.ea=j;this.p={};this.Va=new g(this);this.Z=new e;this.o=new l;this.Wa=new v;this.Db=new p;this.Ia=new t;this.e=new r;this.n=new y;this.C=new k;this.u=new G;this.ic=new w;this.jc=function(a){a=a.data;this.log("#_onApiConfig(account="+a[i.Sb]+", sc_server="+a[i.cc]+\n", sb_server="+a[i.xb]+", check_status_server="+a[i.lb]+", job_id="+a[i.rb]+", publisher="+a[i.Pa]+", ovp="+a[i.$b]+", sdk="+a[i.ec]+", debug_tracking="+a[i.nb]+", track_local="+a[i.yb]+")");this.Db.Xd(a[i.Sb]);this.Db.trackingServer(a[i.cc]);this.f.K=a[i.Pa];this.Ia.ia(a[i.$b]);this.Ia.la(a[i.ec]);this.Ia.channel(a[i.bd]);d().dispatchEvent(new o(o.CLOCK_CHECK_STATUS_ENABLE))};this.ua=function(){this.log("#_onApiDestroy()");d().eb(this)};this.Ye=function(a){a=a.data;this.log("#_onApiOpenMain(name="+\na[i.i]+", length="+a[i.Ca]+", type="+a[i.fc]+", player_name="+a[i.Ea]+", vid="+a[i.Ha]+", aid="+a[i.Aa]+", mid="+a[i.Da]+")");this.nf();this.l=a[i.i];this.p[this.l]=0;this.Ia.playerName(a[i.Ea]);this.n.ib(a[i.Ha]);this.n.$a(a[i.Aa]);this.n.ab(a[i.Da]);this.e.k(this.l);this.e.duration(a[i.Ca]);this.e.type(a[i.fc]);this.Bb=this.e.type();this.C.name(this.l);this.Pe();a={};a[i.Ga]=!0;d().dispatchEvent(new o(o.CLOCK_TRACKING_ENABLE,a));this.ea=function(a){a=a[i.X];var b=new n(this.e,this.n,this.C,this.u,\nA.le,0);b.A=this.o.J(b);this.o.gb(b);a=this.Va.tc(b,a,!0);b={};b[i.Fa]=a;d().dispatchEvent(new z(z.Ba,b))};a={};a[i.qa]=x.ya.Qa;d().dispatchEvent(new x(x.oa,a));a=new n(this.e,this.n,this.C,this.u,A.ob,0);a.A=this.o.J(a);this.Y(a);this.z=!0};this.Xe=function(a){if(this.z){this.info("Call detected: onApiOpenAd().");a=a.data;this.log(this,"#_onApiOpenAd(name="+a[i.i]+", length="+a[i.Ca]+", player_name="+a[i.Ea]+", parent_name="+a[i.ac]+", pod_pos="+a[i.bc]+", pod_offset="+a[i.ub]+", cpm="+a[i.na]+")");\nthis.e.k()||this.e.k(a[i.ac]);this.l=a[i.i];this.p[this.l]=0;var b=new s;b.L(this.l);b.length(a[i.Ca]);b.Nb(a[i.Ea]);b.Fb(a[i.na]);b.Kb(a[i.nd]);b.Mb(this.Cb);b.Lb(a[i.bc]+"");this.e.q(b);this.e.type(r.Ra);a=new n(this.e,this.n,this.C,this.u,A.ob,0);a.A=this.o.J(a);this.Y(a);a=new n(this.e,this.n,this.C,this.u,A.jd,this.p[this.l]);a.A=this.o.J(a);this.Y(a)}else this.warn("#_onApiOpenAd() > No active viewing session.")};this.Ue=function(a){this.z?(a=a.data[i.i],this.log("#_onApiClose(name="+a+")"),\na===this.e.k()?this.Me():this.Le()):this.warn("#_onApiClose() > No active viewing session.")};this.Ze=function(a){if(this.z){if(a=a.data,this.log("#_onApiPlay(name="+a[i.i]+", offset="+a[i.B]+", vid="+a[i.Ha]+", aid="+a[i.Aa]+", mid="+a[i.Da]+")"),!(a[i.i]==this.e.k&&this.zb))this.n.ib(a[i.Ha]),this.n.$a(a[i.Aa]),this.n.ab(a[i.Da]),this.l=a[i.i],this.p[this.l]=Math.floor(a[i.B]),d().dispatchEvent(new o(o.CLOCK_TRACKING_ENABLE)),a=new n(this.e,this.n,this.C,this.u,A.jd,this.p[this.l]),a.A=this.o.J(a),\nthis.Y(a)}else this.warn("#_onApiPlay() > No active viewing session.")};this.cf=function(a){this.z?(a=a.data,this.log("#_onApiStop(name="+a[i.i]+", offset="+a[i.B]+")"),this.l=a[i.i],this.p[this.l]=Math.floor(a[i.B]),a=new n(this.e,this.n,this.C,this.u,A.hd,this.p[this.l]),a.A=this.o.J(a),this.Y(a),d().dispatchEvent(new o(o.CLOCK_TRACKING_DISABLE))):this.warn("#_onApiStop() > No active viewing session.")};this.Te=function(a){this.z?(a=a.data,this.log("#_onApiClick(name="+a[i.i]+", offset="+a[i.B]+\n")")):this.warn("#_onApiClick() > No active viewing session.")};this.Ve=function(a){this.z?(a=a.data,this.log("#_onApiComplete(name="+a[i.i]+", offset="+a[i.B]+")")):this.warn("#_onApiComplete() > No active viewing session.")};this.af=function(a){this.z?(a=a.data,this.log("#_onApiQoSInfo(bitrate="+a[i.i]+", fps="+a[i.Zb]+", dropped_frames="+a[i.Vb]+")"),this.u.qc(a[i.kb]),this.u.Rd(a[i.Zb]),this.u.Od(a[i.Vb])):this.warn("#_onApiQoSInfo() > No active viewing session.")};this.Re=function(a){if(this.z){a=\na.data;this.log("#_onApiBitrateChange(bitrate="+a[i.i]+")");this.u.qc(a[i.kb]);var b=new n(this.e,this.n,this.C,this.u,A.ie,this.p[this.l]);b.A=this.o.J(b);this.o.gb(b);this.ea=function(a){a=this.Va.tc(b,a[i.X],!1);var c={};c[i.Fa]=a;d().dispatchEvent(new z(z.Ba,c))};a={};a[i.qa]=x.ya.Qa;d().dispatchEvent(new x(x.oa,a))}else this.warn("#_onApiBitrateChange() > No active viewing session.")};this.Se=function(){if(this.z){this.log("#_onApiBufferStart()");var a=new n(this.e,this.n,this.C,this.u,A.je,\nthis.p[this.l]);a.A=this.o.J(a);this.Y(a)}else this.warn("#_onApiBufferStart() > No active viewing session.")};this.df=function(a){if(this.z){var b=a.data;this.log("#_onApiTrackError(source="+b[i.vb]+", err_id="+b[i.Yb]+", offset="+b[i.B]+")");if(!(this.hc&&b[i.vb]!==H)){var c=new n(this.e,this.n,this.C,this.u,A.ke,Math.floor(b[i.B]));c.A=this.o.J(c);this.o.gb(c);this.ea=function(a){a=this.Va.tc(c,a[i.X],!1);var m=a.G[0];m.ga.id(b[i.Yb]);m.ga.source(b[i.vb]);m={};m[i.Fa]=a;d().dispatchEvent(new z(z.Ba,\nm))};a={};a[i.qa]=x.ya.Qa;d().dispatchEvent(new x(x.oa,a))}}else this.warn("#_onApiTrackError() > No active viewing session.")};this.$e=function(a){this.z?(this.Cb=Math.floor(a.data[i.ub]),this.log("#_onApiPodOffset(podOffset="+this.Cb+")")):this.warn("#_onApiPodOffset() > No active viewing session.")};this.bf=function(){if(this.z){this.log("#_onApiSessionComplete()");var a=new n(this.e,this.n,this.C,this.u,A.me,0);a.A=this.o.J(a);this.Y(a);this.ea=function(a){var b=new Date;a=this.Va.Jd(this.Ya||\nnew Date(0),b,a[i.X]);var c={};c[i.Fa]=a;d().dispatchEvent(new z(z.Ba,c));this.Ya=b};a={};a[i.qa]=x.ya.Qa;d().dispatchEvent(new x(x.oa,a));a={};a[i.Ga]=!0;d().dispatchEvent(new o(o.CLOCK_TRACKING_DISABLE,a));this.z=!1}else this.warn("#_onApiSessionComplete() > No active viewing session.")};this.We=function(a){if(this.z){var b=a.data;this.log("#_onApiMonitor(name="+b[i.i]+", offset="+b[i.B]+")");this.l=b[i.i];this.p[this.l]=Math.floor(a.data[i.B])}else this.warn("#_onApiMonitor() > No active viewing session.")};\nthis.ff=function(a){if(this.z){this.log("#_onClockTrackingTick(interval="+a.data[i.X]+")");var b=new Date;a=this.Va.Jd(this.Ya||new Date(0),b,a.data[i.X]);var c={};c[i.Fa]=a;d().dispatchEvent(new z(z.Ba,c));this.Ya=b}else this.warn("#_onClockTrackingTick() > No active viewing session.")};this.hf=function(a){this.log("#_onNetworkCheckStatusComplete(track_ext_err="+a.data[i.gc]+")");a=a.data[i.gc];if(a!==j)this.hc=a};this.lc=function(a){a=a.data[i.qa];this.log("#_onDataRequest(what="+a+")");switch(a){case x.ya.md:a=\n{},a[i.Pa]=this.f.K,d().dispatchEvent(new x(x.mb,a))}};this.mc=function(a){this.log("#_onDataResponse()");this.Pd(a.data)};this.nf=function(){this.log("#_resetInternalState()");this.zb=this.z=!1;this.Bb=j;this.hc=!1;this.p={};this.Ya=this.Cb=j;this.ic=new w;this.o=new l;this.Z=new e;this.n=new y;this.C=new k;this.u=new G;this.Wa=new v;this.e=new r;this.e.K(this.f.K);this.e.type(this.f.fb)};this.Pe=function(){this.Wa.Ec(""+(new Date).getTime()+Math.floor(Math.random()*1E9));this.log("#_generateSessionId() > New session id: "+\nthis.Wa.Ec)};this.Y=function(a){this.log("#_placeItemOnTimeline(type="+a.$+")");this.Z.tf(a);this.o.gb(a)};this.Me=function(){if(this.zb)this.warn("#_closeMainVideo() > The main video content was already closed.");else{this.p[this.e.k()]==-1&&(this.p[this.e.k()]=this.e.duration());var a=new n(this.e,this.n,this.C,this.u,A.gd,this.p[this.e.k()]);a.A=this.o.J(a);this.Y(a);this.zb=!0}};this.Le=function(){var a=new n(this.e,this.n,this.C,this.u,A.gd,this.p[this.l]);a.A=this.o.J(a);this.Y(a);this.e.type(this.Bb);\nthis.l=this.e.k();this.e.q(j)};d().addEventListener(u.Tb,this.jc,this);d().addEventListener(u.La,this.ua,this);d().addEventListener(u.Uc,this.Ye,this);d().addEventListener(u.Tc,this.Xe,this);d().addEventListener(u.Qc,this.Ue,this);d().addEventListener(u.Vc,this.Ze,this);d().addEventListener(u.Yc,this.cf,this);d().addEventListener(u.Pc,this.Te,this);d().addEventListener(u.Rc,this.Ve,this);d().addEventListener(u.Wc,this.af,this);d().addEventListener(u.Nc,this.Re,this);d().addEventListener(u.Oc,this.Se,\nthis);d().addEventListener(u.Zc,this.df,this);d().addEventListener(u.Ub,this.$e,this);d().addEventListener(u.Xc,this.bf,this);d().addEventListener(u.Sc,this.We,this);d().addEventListener(o.CLOCK_TRACKING_TICK,this.ff,this);d().addEventListener(D.sb,this.hf,this);d().addEventListener(x.oa,this.lc,this);d().addEventListener(x.mb,this.mc,this)}var m=c.Q,f=c.P,d=c.ca,g=a.H.ve,e=a.H.De,n=a.H.td,l=a.H.oe,v=a.a.ye,p=a.a.Ae,r=a.a.jb,t=a.a.xe,y=a.a.vd,k=a.a.qd,G=a.a.re,w=a.H.ge,o=a.event.Na,D=a.event.tb,x=\na.event.Wb,u=a.event.Ma,z=a.event.dd,i=a.event.ba,A=a.a.pb,s=a.a.$c,H="player";m(b,c.Md);m(b,f);a.H.fe=b})(d,e);(function(c){function a(a){this.N("[media-fork::HeartbeatMediaFork] > ");this.m=a;this.D=function(){var a=this.ta&&(this.m.analyticsVisitorID||this.m.marketingCloudVisitorID||this.m.visitorID);a||this.warn("Unable to track! Is configured: "+this.ta+" analyticsVisitorID: "+this.m.analyticsVisitorID+" marketingCloudVisitorID: "+this.m.marketingCloudVisitorID+" visitorID: "+this.m.visitorID);\nreturn a};this.ta=!1;this.j=new n;this.eg=new l(new v);this.yd=new g;this.nc=j;this.f={trackingServer:j,vc:j,K:j,fb:j,ia:j,la:j,channel:j,debugTracking:!1,Jc:!1}}var b=d.Q,m=d.ca,f=c.event.ba,e=c.event.Ma,g=c.M.ee,l=c.bb.qe,n=c.H.fe,q=c.a.jb,v=c.a.se;b(a,d.P);a.prototype.Wf=function(a){if(a&&a.hasOwnProperty("debugLogging"))d.Oa=a.debugLogging;this.log("#setup(configData={trackingServer: "+a.trackingServer+", jobId: "+a.vc+", streamType: "+a.fb+", publisher: "+a.K+", ovp: "+a.ia+", sdk: "+a.la+", debugLogging: "+\na.ig+"})");this.f.debugTracking=this.m.debugTracking;this.f.Jc=this.m.trackLocal;this.f.channel=this.m.Media.channel;if(a){if(a.hasOwnProperty("trackingServer"))this.f.trackingServer=a.trackingServer;if(a.hasOwnProperty("jobId"))this.f.vc=a.jobId;if(a.hasOwnProperty("publisher"))this.f.K=a.publisher;if(a.hasOwnProperty("ovp"))this.f.ia=a.ovp;if(a.hasOwnProperty("sdk"))this.f.la=a.sdk;if(a.hasOwnProperty("streamType"))this.f.fb=a.streamType===q.sd||a.streamType===q.Ce||a.streamType===q.Be||a.streamType===\nq.Ra?a.streamType:q.sd;if(this.m.Media.__primetime)this.f.ia="primetime";if(this.nc!=j)this.f.la=this.nc;this.log("#setup() > Applying configuration: {account: "+this.m.account+", scTrackingServer: "+this.m.trackingServer+", sbTrackingServer: "+this.f.trackingServer+", jobId: "+this.f.trackingServer+", publisher: "+this.f.K+", ovp: "+this.f.ia+", sdk: "+this.f.la+", channel: "+this.f.channel+", debugTracking: "+this.f.debugTracking+", trackLocal: "+this.f.Jc+"}");a={};a[f.Sb]=this.m.account;a[f.cc]=\nthis.m.trackingServer;a[f.xb]=this.f.trackingServer;a[f.lb]=this.f.trackingServer+"/settings/";a[f.rb]=this.f.vc;a[f.Pa]=this.f.K;a[f.$b]=this.f.ia;a[f.ec]=this.f.la;a[f.bd]=this.f.channel;a[f.nb]=this.f.debugTracking;a[f.yb]=this.f.Jc;m().dispatchEvent(new e(e.Tb,a));this.ta=!0}};a.prototype.open=function(a,b,c){this.log("#open(name="+a+", length="+b+", playerName="+c+")");if(this.D()){var d={};d[f.Ha]=this.m.visitorID;d[f.Aa]=this.m.analyticsVisitorID;d[f.Da]=this.m.Nf;d[f.i]=a;d[f.Ca]=b;d[f.fc]=\nthis.f.fb;d[f.Ea]=c;m().dispatchEvent(new e(e.Uc,d))}};a.prototype.openAd=function(a,b,c,d,g,l,n){this.log("#openAd(name="+a+", length="+b+", playerName="+c+", parentName="+d+", parentPod="+g+", parentPodPosition="+l+", cpm="+n+", )");if(this.D()){var o={};o[f.i]=a;o[f.Ca]=b;o[f.Ea]=c;o[f.ac]=d;o[f.nd]=g;o[f.bc]=l;o[f.na]=n;m().dispatchEvent(new e(e.Tc,o))}};a.prototype.close=function(a){this.log("#close(name="+a+")");if(this.D()){var b={};b[f.i]=a;m().dispatchEvent(new e(e.Qc,b))}};a.prototype.play=\nfunction(a,b,c,d,g){this.log("#play(name="+a+", offset="+b+", segmentNum="+c+", segment="+d+", segmentLength="+g+")");if(this.D())c={},c[f.Ha]=this.m.visitorID,c[f.Aa]=this.m.analyticsVisitorID,c[f.Da]=this.m.Nf,c[f.i]=a,c[f.B]=b,m().dispatchEvent(new e(e.Vc,c))};a.prototype.monitor=function(a,b){this.log("#monitor(name="+a+", offset="+b+")");var c={};c[f.i]=a;c[f.B]=b;m().dispatchEvent(new e(e.Sc,c))};a.prototype.stop=function(a,b){this.log("#stop(name="+a+", offset="+b+")");if(this.D()){var c={};\nc[f.i]=a;c[f.B]=b;m().dispatchEvent(new e(e.Yc,c))}};a.prototype.click=function(a,b){this.log("#click(name="+a+", offset="+b+")");if(this.D()){var c={};c[f.i]=a;c[f.B]=b;m().dispatchEvent(new e(e.Pc,c))}};a.prototype.complete=function(a,b){this.log("#complete(name="+a+", offset="+b+")");if(this.D()){var c={};c[f.i]=a;c[f.B]=b;m().dispatchEvent(new e(e.Rc,c))}};a.prototype.Nd=function(){Logger.hg(this,"#destroy()");m().dispatchEvent(new e(e.La))};a.prototype.$f=function(a,b,c){this.log("#trackError(source="+\na+", errorId="+b+", offset="+c+")");if(this.D()){var d={};d[f.vb]=a;d[f.Yb]=b;d[f.B]=c;m().dispatchEvent(new e(e.Zc,d))}};a.prototype.bg=function(a,b,c){this.log("#updateQoSInfo(bitrate="+a+", fps="+b+", droppedFrames="+c+")");if(this.D()){var d={};d[f.kb]=a;d[f.Zb]=b;d[f.Vb]=c;m().dispatchEvent(new e(e.Wc,d))}};a.prototype.wf=function(a){this.log("#bitrateChange(bitrate="+a+")");if(this.D()){var b={};b[f.kb]=a;m().dispatchEvent(new e(e.Nc,b))}};a.prototype.xf=function(){this.log("#bufferStart()");\nthis.D()&&m().dispatchEvent(new e(e.Oc))};a.prototype.sf=function(a){this.log("#adBreakStart(offset="+a+")");if(this.D()){var b={};b[f.ub]=a;m().dispatchEvent(new e(e.Ub,b))}};a.prototype.rf=function(){this.log("#adBreakEnd()");if(this.D()){var a={};a[f.ub]=j;m().dispatchEvent(new e(e.Ub,a))}};a.prototype.Uf=function(){this.log("#sessionComplete()");this.D()&&m().dispatchEvent(new e(e.Xc))};a.prototype.Ke=function(a){this.log("#__setPsdkVersion(version="+a+")");this.nc=a};c.ne=a})(e);m.Ka||(m.Ka=\n{});m.Ka.Bf||(m.Ka.Bf=d);m.Ka.Of=e})(this);this.Je(m)}(m.s);D.callMethodWhenReady=function(m,g){s.visitor!=j&&(s.isReadyToTrack()?D[m].apply(this,g):s.callbackWhenReadyToTrack(D,D[m],g))};m.Heartbeat=D;m.uf=function(){var e,g;if(m.autoTrack&&(e=m.s.d.getElementsByTagName("VIDEO")))for(g=0;g<e.length;g++)m.attach(e[g])};m.ra(w,"load",m.uf)}\n</script>'
                }
                ]
            }
            ],
            event: "pagetop"
        }, {
            name: "~7 - AppMeasurement.js - Shared AppMeasurement Library",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<script type="text/javascript">\n/*\n ============== DO NOT ALTER ANYTHING BELOW THIS LINE ! ===============\n\n AppMeasurement for JavaScript version: 1.4.1\n Copyright 1996-2013 Adobe, Inc. All Rights Reserved\n More info available at http://www.omniture.com\n*/\nfunction AppMeasurement(){var s=this;s.version="1.4.1";var w=window;if(!w.s_c_in)w.s_c_il=[],w.s_c_in=0;s._il=w.s_c_il;s._in=w.s_c_in;s._il[s._in]=s;w.s_c_in++;s._c="s_c";var k=w.sb;k||(k=null);var m=w,i,o;try{i=m.parent;for(o=m.location;i&&i.location&&o&&""+i.location!=""+o&&m.location&&""+i.location!=""+m.location&&i.location.host==o.host;)m=i,i=m.parent}catch(p){}s.eb=function(s){try{console.log(s)}catch(a){}};s.ta=function(s){return""+parseInt(s)==""+s};s.replace=function(s,a,c){if(!s||s.indexOf(a)<\n0)return s;return s.split(a).join(c)};s.escape=function(b){var a,c;if(!b)return b;b=encodeURIComponent(b);for(a=0;a<7;a++)c="+~!*()\'".substring(a,a+1),b.indexOf(c)>=0&&(b=s.replace(b,c,"%"+c.charCodeAt(0).toString(16).toUpperCase()));return b};s.unescape=function(b){if(!b)return b;b=b.indexOf("+")>=0?s.replace(b,"+"," "):b;try{return decodeURIComponent(b)}catch(a){}return unescape(b)};s.Va=function(){var b=w.location.hostname,a=s.fpCookieDomainPeriods,c;if(!a)a=s.cookieDomainPeriods;if(b&&!s.cookieDomain&&\n!/^[0-9.]+$/.test(b)&&(a=a?parseInt(a):2,a=a>2?a:2,c=b.lastIndexOf("."),c>=0)){for(;c>=0&&a>1;)c=b.lastIndexOf(".",c-1),a--;s.cookieDomain=c>0?b.substring(c):b}return s.cookieDomain};s.c_r=s.cookieRead=function(b){b=s.escape(b);var a=" "+s.d.cookie,c=a.indexOf(" "+b+"="),e=c<0?c:a.indexOf(";",c);b=c<0?"":s.unescape(a.substring(c+2+b.length,e<0?a.length:e));return b!="[[B]]"?b:""};s.c_w=s.cookieWrite=function(b,a,c){var e=s.Va(),d=s.cookieLifetime,f;a=""+a;d=d?(""+d).toUpperCase():"";c&&d!="SESSION"&&\nd!="NONE"&&((f=a!=""?parseInt(d?d:0):-60)?(c=new Date,c.setTime(c.getTime()+f*1E3)):c==1&&(c=new Date,f=c.getYear(),c.setYear(f+5+(f<1900?1900:0))));if(b&&d!="NONE")return s.d.cookie=b+"="+s.escape(a!=""?a:"[[B]]")+"; path=/;"+(c&&d!="SESSION"?" expires="+c.toGMTString()+";":"")+(e?" domain="+e+";":""),s.cookieRead(b)==a;return 0};s.C=[];s.B=function(b,a,c){if(s.ma)return 0;if(!s.maxDelay)s.maxDelay=250;var e=0,d=(new Date).getTime()+s.maxDelay,f=s.d.qb,g=["webkitvisibilitychange","visibilitychange"];\nif(!f)f=s.d.rb;if(f&&f=="prerender"){if(!s.X){s.X=1;for(c=0;c<g.length;c++)s.d.addEventListener(g[c],function(){var a=s.d.qb;if(!a)a=s.d.rb;if(a=="visible")s.X=0,s.delayReady()})}e=1;d=0}else c||s.q("_d")&&(e=1);e&&(s.C.push({m:b,a:a,t:d}),s.X||setTimeout(s.delayReady,s.maxDelay));return e};s.delayReady=function(){var b=(new Date).getTime(),a=0,c;for(s.q("_d")&&(a=1);s.C.length>0;){c=s.C.shift();if(a&&!c.t&&c.t>b){s.C.unshift(c);setTimeout(s.delayReady,parseInt(s.maxDelay/2));break}s.ma=1;s[c.m].apply(s,\nc.a);s.ma=0}};s.setAccount=s.sa=function(b){var a,c;if(!s.B("setAccount",arguments))if(s.account=b,s.allAccounts){a=s.allAccounts.concat(b.split(","));s.allAccounts=[];a.sort();for(c=0;c<a.length;c++)(c==0||a[c-1]!=a[c])&&s.allAccounts.push(a[c])}else s.allAccounts=b.split(",")};s.foreachVar=function(b,a){var c,e,d,f,g="";d=e="";if(s.lightProfileID)c=s.H,(g=s.lightTrackVars)&&(g=","+g+","+s.ba.join(",")+",");else{c=s.c;if(s.pe||s.linkType)if(g=s.linkTrackVars,e=s.linkTrackEvents,s.pe&&(d=s.pe.substring(0,\n1).toUpperCase()+s.pe.substring(1),s[d]))g=s[d].pb,e=s[d].ob;g&&(g=","+g+","+s.z.join(",")+",");e&&g&&(g+=",events,")}a&&(a=","+a+",");for(e=0;e<c.length;e++)d=c[e],(f=s[d])&&(!g||g.indexOf(","+d+",")>=0)&&(!a||a.indexOf(","+d+",")>=0)&&b(d,f)};s.J=function(b,a,c,e,d){var f="",g,j,w,q,i=0;b=="contextData"&&(b="c");if(a){for(g in a)if(!Object.prototype[g]&&(!d||g.substring(0,d.length)==d)&&a[g]&&(!c||c.indexOf(","+(e?e+".":"")+g+",")>=0)){w=!1;if(i)for(j=0;j<i.length;j++)g.substring(0,i[j].length)==\ni[j]&&(w=!0);if(!w&&(f==""&&(f+="&"+b+"."),j=a[g],d&&(g=g.substring(d.length)),g.length>0))if(w=g.indexOf("."),w>0)j=g.substring(0,w),w=(d?d:"")+j+".",i||(i=[]),i.push(w),f+=s.J(j,a,c,e,w);else if(typeof j=="boolean"&&(j=j?"true":"false"),j){if(e=="retrieveLightData"&&d.indexOf(".contextData.")<0)switch(w=g.substring(0,4),q=g.substring(4),g){case "transactionID":g="xact";break;case "channel":g="ch";break;case "campaign":g="v0";break;default:s.ta(q)&&(w=="prop"?g="c"+q:w=="eVar"?g="v"+q:w=="list"?\ng="l"+q:w=="hier"&&(g="h"+q,j=j.substring(0,255)))}f+="&"+s.escape(g)+"="+s.escape(j)}}f!=""&&(f+="&."+b)}return f};s.Xa=function(){var b="",a,c,e,d,f,g,j,w,i="",k="",m=c="";if(s.lightProfileID)a=s.H,(i=s.lightTrackVars)&&(i=","+i+","+s.ba.join(",")+",");else{a=s.c;if(s.pe||s.linkType)if(i=s.linkTrackVars,k=s.linkTrackEvents,s.pe&&(c=s.pe.substring(0,1).toUpperCase()+s.pe.substring(1),s[c]))i=s[c].pb,k=s[c].ob;i&&(i=","+i+","+s.z.join(",")+",");k&&(k=","+k+",",i&&(i+=",events,"));s.events2&&(m+=(m!=\n""?",":"")+s.events2)}s.AudienceManagement&&s.AudienceManagement.isReady()&&(b+=s.J("d",s.AudienceManagement.getEventCallConfigParams()));for(c=0;c<a.length;c++){d=a[c];f=s[d];e=d.substring(0,4);g=d.substring(4);!f&&d=="events"&&m&&(f=m,m="");if(f&&(!i||i.indexOf(","+d+",")>=0)){switch(d){case "supplementalDataID":d="sdid";break;case "timestamp":d="ts";break;case "dynamicVariablePrefix":d="D";break;case "visitorID":d="vid";break;case "marketingCloudVisitorID":d="mid";break;case "analyticsVisitorID":d=\n"aid";break;case "audienceManagerLocationHint":d="aamlh";break;case "audienceManagerBlob":d="aamb";break;case "authState":d="as";break;case "pageURL":d="g";if(f.length>255)s.pageURLRest=f.substring(255),f=f.substring(0,255);break;case "pageURLRest":d="-g";break;case "referrer":d="r";break;case "vmk":case "visitorMigrationKey":d="vmt";break;case "visitorMigrationServer":d="vmf";s.ssl&&s.visitorMigrationServerSecure&&(f="");break;case "visitorMigrationServerSecure":d="vmf";!s.ssl&&s.visitorMigrationServer&&\n(f="");break;case "charSet":d="ce";break;case "visitorNamespace":d="ns";break;case "cookieDomainPeriods":d="cdp";break;case "cookieLifetime":d="cl";break;case "variableProvider":d="vvp";break;case "currencyCode":d="cc";break;case "channel":d="ch";break;case "transactionID":d="xact";break;case "campaign":d="v0";break;case "latitude":d="lat";break;case "longitude":d="lon";break;case "resolution":d="s";break;case "colorDepth":d="c";break;case "javascriptVersion":d="j";break;case "javaEnabled":d="v";\nbreak;case "cookiesEnabled":d="k";break;case "browserWidth":d="bw";break;case "browserHeight":d="bh";break;case "connectionType":d="ct";break;case "homepage":d="hp";break;case "events":m&&(f+=(f!=""?",":"")+m);if(k){g=f.split(",");f="";for(e=0;e<g.length;e++)j=g[e],w=j.indexOf("="),w>=0&&(j=j.substring(0,w)),w=j.indexOf(":"),w>=0&&(j=j.substring(0,w)),k.indexOf(","+j+",")>=0&&(f+=(f?",":"")+g[e])}break;case "events2":f="";break;case "contextData":b+=s.J("c",s[d],i,d);f="";break;case "lightProfileID":d=\n"mtp";break;case "lightStoreForSeconds":d="mtss";s.lightProfileID||(f="");break;case "lightIncrementBy":d="mti";s.lightProfileID||(f="");break;case "retrieveLightProfiles":d="mtsr";break;case "deleteLightProfiles":d="mtsd";break;case "retrieveLightData":s.retrieveLightProfiles&&(b+=s.J("mts",s[d],i,d));f="";break;default:s.ta(g)&&(e=="prop"?d="c"+g:e=="eVar"?d="v"+g:e=="list"?d="l"+g:e=="hier"&&(d="h"+g,f=f.substring(0,255)))}f&&(b+="&"+d+"="+(d.substring(0,3)!="pev"?s.escape(f):f))}d=="pev3"&&s.g&&\n(b+=s.g)}return b};s.u=function(s){var a=s.tagName;if(""+s.wb!="undefined"||""+s.ib!="undefined"&&(""+s.ib).toUpperCase()!="HTML")return"";a=a&&a.toUpperCase?a.toUpperCase():"";a=="SHAPE"&&(a="");a&&((a=="INPUT"||a=="BUTTON")&&s.type&&s.type.toUpperCase?a=s.type.toUpperCase():!a&&s.href&&(a="A"));return a};s.oa=function(s){var a=s.href?s.href:"",c,e,d;c=a.indexOf(":");e=a.indexOf("?");d=a.indexOf("/");if(a&&(c<0||e>=0&&c>e||d>=0&&c>d))e=s.protocol&&s.protocol.length>1?s.protocol:l.protocol?l.protocol:\n"",c=l.pathname.lastIndexOf("/"),a=(e?e+"//":"")+(s.host?s.host:l.host?l.host:"")+(h.substring(0,1)!="/"?l.pathname.substring(0,c<0?0:c)+"/":"")+a;return a};s.D=function(b){var a=s.u(b),c,e,d="",f=0;if(a){c=b.protocol;e=b.onclick;if(b.href&&(a=="A"||a=="AREA")&&(!e||!c||c.toLowerCase().indexOf("javascript")<0))d=s.oa(b);else if(e)d=s.replace(s.replace(s.replace(s.replace(""+e,"\\r",""),"\\n",""),"\\t","")," ",""),f=2;else if(a=="INPUT"||a=="SUBMIT"){if(b.value)d=b.value;else if(b.innerText)d=b.innerText;\nelse if(b.textContent)d=b.textContent;f=3}else if(b.src&&a=="IMAGE")d=b.src;if(d)return{id:d.substring(0,100),type:f}}return 0};s.tb=function(b){for(var a=s.u(b),c=s.D(b);b&&!c&&a!="BODY";)if(b=b.parentElement?b.parentElement:b.parentNode)a=s.u(b),c=s.D(b);if(!c||a=="BODY")b=0;if(b&&(a=b.onclick?""+b.onclick:"",a.indexOf(".tl(")>=0||a.indexOf(".trackLink(")>=0))b=0;return b};s.hb=function(){var b,a,c=s.linkObject,e=s.linkType,d=s.linkURL,f,g;s.ca=1;if(!c)s.ca=0,c=s.clickObject;if(c){b=s.u(c);for(a=\ns.D(c);c&&!a&&b!="BODY";)if(c=c.parentElement?c.parentElement:c.parentNode)b=s.u(c),a=s.D(c);if(!a||b=="BODY")c=0;if(c){var j=c.onclick?""+c.onclick:"";if(j.indexOf(".tl(")>=0||j.indexOf(".trackLink(")>=0)c=0}}else s.ca=1;!d&&c&&(d=s.oa(c));d&&!s.linkLeaveQueryString&&(f=d.indexOf("?"),f>=0&&(d=d.substring(0,f)));if(!e&&d){var i=0,k=0,m;if(s.trackDownloadLinks&&s.linkDownloadFileTypes){j=d.toLowerCase();f=j.indexOf("?");g=j.indexOf("#");f>=0?g>=0&&g<f&&(f=g):f=g;f>=0&&(j=j.substring(0,f));f=s.linkDownloadFileTypes.toLowerCase().split(",");\nfor(g=0;g<f.length;g++)(m=f[g])&&j.substring(j.length-(m.length+1))=="."+m&&(e="d")}if(s.trackExternalLinks&&!e&&(j=d.toLowerCase(),s.ra(j))){if(!s.linkInternalFilters)s.linkInternalFilters=w.location.hostname;f=0;s.linkExternalFilters?(f=s.linkExternalFilters.toLowerCase().split(","),i=1):s.linkInternalFilters&&(f=s.linkInternalFilters.toLowerCase().split(","));if(f){for(g=0;g<f.length;g++)m=f[g],j.indexOf(m)>=0&&(k=1);k?i&&(e="e"):i||(e="e")}}}s.linkObject=c;s.linkURL=d;s.linkType=e;if(s.trackClickMap||\ns.trackInlineStats)if(s.g="",c){e=s.pageName;d=1;c=c.sourceIndex;if(!e)e=s.pageURL,d=0;if(w.s_objectID)a.id=w.s_objectID,c=a.type=1;if(e&&a&&a.id&&b)s.g="&pid="+s.escape(e.substring(0,255))+(d?"&pidt="+d:"")+"&oid="+s.escape(a.id.substring(0,100))+(a.type?"&oidt="+a.type:"")+"&ot="+b+(c?"&oi="+c:"")}};s.Ya=function(){var b=s.ca,a=s.linkType,c=s.linkURL,e=s.linkName;if(a&&(c||e))a=a.toLowerCase(),a!="d"&&a!="e"&&(a="o"),s.pe="lnk_"+a,s.pev1=c?s.escape(c):"",s.pev2=e?s.escape(e):"",b=1;s.abort&&(b=\n0);if(s.trackClickMap||s.trackInlineStats){a={};c=0;var d=s.cookieRead("s_sq"),f=d?d.split("&"):0,g,j,w;d=0;if(f)for(g=0;g<f.length;g++)j=f[g].split("="),e=s.unescape(j[0]).split(","),j=s.unescape(j[1]),a[j]=e;e=s.account.split(",");if(b||s.g){b&&!s.g&&(d=1);for(j in a)if(!Object.prototype[j])for(g=0;g<e.length;g++){d&&(w=a[j].join(","),w==s.account&&(s.g+=(j.charAt(0)!="&"?"&":"")+j,a[j]=[],c=1));for(f=0;f<a[j].length;f++)w=a[j][f],w==e[g]&&(d&&(s.g+="&u="+s.escape(w)+(j.charAt(0)!="&"?"&":"")+j+\n"&u=0"),a[j].splice(f,1),c=1)}b||(c=1);if(c){d="";g=2;!b&&s.g&&(d=s.escape(e.join(","))+"="+s.escape(s.g),g=1);for(j in a)!Object.prototype[j]&&g>0&&a[j].length>0&&(d+=(d?"&":"")+s.escape(a[j].join(","))+"="+s.escape(j),g--);s.cookieWrite("s_sq",d)}}}return b};s.Za=function(){if(!s.nb){var b=new Date,a=m.location,c,e,d=e=c="",f="",g="",w="1.2",i=s.cookieWrite("s_cc","true",0)?"Y":"N",k="",n="";if(b.setUTCDate&&(w="1.3",(0).toPrecision&&(w="1.5",b=[],b.forEach))){w="1.6";e=0;c={};try{e=new Iterator(c),\ne.next&&(w="1.7",b.reduce&&(w="1.8",w.trim&&(w="1.8.1",Date.parse&&(w="1.8.2",Object.create&&(w="1.8.5")))))}catch(o){}}c=screen.width+"x"+screen.height;d=navigator.javaEnabled()?"Y":"N";e=screen.pixelDepth?screen.pixelDepth:screen.colorDepth;f=s.w.innerWidth?s.w.innerWidth:s.d.documentElement.offsetWidth;g=s.w.innerHeight?s.w.innerHeight:s.d.documentElement.offsetHeight;try{s.b.addBehavior("#default#homePage"),k=s.b.ub(a)?"Y":"N"}catch(p){}try{s.b.addBehavior("#default#clientCaps"),n=s.b.connectionType}catch(r){}s.resolution=\nc;s.colorDepth=e;s.javascriptVersion=w;s.javaEnabled=d;s.cookiesEnabled=i;s.browserWidth=f;s.browserHeight=g;s.connectionType=n;s.homepage=k;s.nb=1}};s.I={};s.loadModule=function(b,a){var c=s.I[b];if(!c){c=w["AppMeasurement_Module_"+b]?new w["AppMeasurement_Module_"+b](s):{};s.I[b]=s[b]=c;c.Fa=function(){return c.Ja};c.Ka=function(a){if(c.Ja=a)s[b+"_onLoad"]=a,s.B(b+"_onLoad",[s,c],1)||a(s,c)};try{Object.defineProperty?Object.defineProperty(c,"onLoad",{get:c.Fa,set:c.Ka}):c._olc=1}catch(e){c._olc=\n1}}a&&(s[b+"_onLoad"]=a,s.B(b+"_onLoad",[s,c],1)||a(s,c))};s.q=function(b){var a,c;for(a in s.I)if(!Object.prototype[a]&&(c=s.I[a])){if(c._olc&&c.onLoad)c._olc=0,c.onLoad(s,c);if(c[b]&&c[b]())return 1}return 0};s.bb=function(){var b=Math.floor(Math.random()*1E13),a=s.visitorSampling,c=s.visitorSamplingGroup;c="s_vsn_"+(s.visitorNamespace?s.visitorNamespace:s.account)+(c?"_"+c:"");var e=s.cookieRead(c);if(a){e&&(e=parseInt(e));if(!e){if(!s.cookieWrite(c,b))return 0;e=b}if(e%1E4>v)return 0}return 1};\ns.K=function(b,a){var c,e,d,f,g,w;for(c=0;c<2;c++){e=c>0?s.ia:s.c;for(d=0;d<e.length;d++)if(f=e[d],(g=b[f])||b["!"+f]){if(!a&&(f=="contextData"||f=="retrieveLightData")&&s[f])for(w in s[f])g[w]||(g[w]=s[f][w]);s[f]=g}}};s.Aa=function(b,a){var c,e,d,f;for(c=0;c<2;c++){e=c>0?s.ia:s.c;for(d=0;d<e.length;d++)f=e[d],b[f]=s[f],!a&&!b[f]&&(b["!"+f]=1)}};s.Ua=function(s){var a,c,e,d,f,g=0,w,i="",k="";if(s&&s.length>255&&(a=""+s,c=a.indexOf("?"),c>0&&(w=a.substring(c+1),a=a.substring(0,c),d=a.toLowerCase(),\ne=0,d.substring(0,7)=="http://"?e+=7:d.substring(0,8)=="https://"&&(e+=8),c=d.indexOf("/",e),c>0&&(d=d.substring(e,c),f=a.substring(c),a=a.substring(0,c),d.indexOf("google")>=0?g=",q,ie,start,search_key,word,kw,cd,":d.indexOf("yahoo.co")>=0&&(g=",p,ei,"),g&&w)))){if((s=w.split("&"))&&s.length>1){for(e=0;e<s.length;e++)d=s[e],c=d.indexOf("="),c>0&&g.indexOf(","+d.substring(0,c)+",")>=0?i+=(i?"&":"")+d:k+=(k?"&":"")+d;i&&k?w=i+"&"+k:k=""}c=253-(w.length-k.length)-a.length;s=a+(c>0?f.substring(0,c):\n"")+"?"+w}return s};s.U=!1;s.O=!1;s.Ia=function(b){s.marketingCloudVisitorID=b;s.O=!0;s.k()};s.R=!1;s.L=!1;s.Ca=function(b){s.analyticsVisitorID=b;s.L=!0;s.k()};s.T=!1;s.N=!1;s.Ea=function(b){s.audienceManagerLocationHint=b;s.N=!0;s.k()};s.S=!1;s.M=!1;s.Da=function(b){s.audienceManagerBlob=b;s.M=!0;s.k()};s.isReadyToTrack=function(){var b=!0,a=s.visitor;if(a&&a.isAllowed()){if(!s.U&&!s.marketingCloudVisitorID&&a.getMarketingCloudVisitorID&&(s.U=!0,s.marketingCloudVisitorID=a.getMarketingCloudVisitorID([s,\ns.Ia]),s.marketingCloudVisitorID))s.O=!0;if(!s.R&&!s.analyticsVisitorID&&a.getAnalyticsVisitorID&&(s.R=!0,s.analyticsVisitorID=a.getAnalyticsVisitorID([s,s.Ca]),s.analyticsVisitorID))s.L=!0;if(!s.T&&!s.audienceManagerLocationHint&&a.getAudienceManagerLocationHint&&(s.T=!0,s.audienceManagerLocationHint=a.getAudienceManagerLocationHint([s,s.Ea]),s.audienceManagerLocationHint))s.N=!0;if(!s.S&&!s.audienceManagerBlob&&a.getAudienceManagerBlob&&(s.S=!0,s.audienceManagerBlob=a.getAudienceManagerBlob([s,\ns.Da]),s.audienceManagerBlob))s.M=!0;if(s.U&&!s.O&&!s.marketingCloudVisitorID||s.R&&!s.L&&!s.analyticsVisitorID||s.T&&!s.N&&!s.audienceManagerLocationHint||s.S&&!s.M&&!s.audienceManagerBlob)b=!1}return b};s.j=k;s.l=0;s.callbackWhenReadyToTrack=function(b,a,c){var e;e={};e.Oa=b;e.Na=a;e.La=c;if(s.j==k)s.j=[];s.j.push(e);if(s.l==0)s.l=setInterval(s.k,100)};s.k=function(){var b;if(s.isReadyToTrack()){if(s.l)clearInterval(s.l),s.l=0;if(s.j!=k)for(;s.j.length>0;)b=s.j.shift(),b.Na.apply(b.Oa,b.La)}};s.Ga=\nfunction(b){var a,c,e=k,d=k;if(!s.isReadyToTrack()){a=[];if(b!=k)for(c in e={},b)e[c]=b[c];d={};s.Aa(d,!0);a.push(e);a.push(d);s.callbackWhenReadyToTrack(s,s.track,a);return!0}return!1};s.Wa=function(){var b=s.cookieRead("s_fid"),a="",c="",e;e=8;var d=4;if(!b||b.indexOf("-")<0){for(b=0;b<16;b++)e=Math.floor(Math.random()*e),a+="0123456789ABCDEF".substring(e,e+1),e=Math.floor(Math.random()*d),c+="0123456789ABCDEF".substring(e,e+1),e=d=16;b=a+"-"+c}s.cookieWrite("s_fid",b,1)||(b=0);return b};s.t=s.track=\nfunction(b,a){var c,e=new Date,d="s"+Math.floor(e.getTime()/108E5)%10+Math.floor(Math.random()*1E13),f=e.getYear();f="t="+s.escape(e.getDate()+"/"+e.getMonth()+"/"+(f<1900?f+1900:f)+" "+e.getHours()+":"+e.getMinutes()+":"+e.getSeconds()+" "+e.getDay()+" "+e.getTimezoneOffset());if(s.visitor){if(s.visitor.getAuthState)s.authState=s.visitor.getAuthState();if(!s.supplementalDataID&&s.visitor.getSupplementalDataID)s.supplementalDataID=s.visitor.getSupplementalDataID("AppMeasurement:"+s._in,s.expectSupplementalData?\n!1:!0)}s.q("_s");if(!s.B("track",arguments)){if(!s.Ga(b)){a&&s.K(a);b&&(c={},s.Aa(c,0),s.K(b));if(s.bb()){if(!s.analyticsVisitorID&&!s.marketingCloudVisitorID)s.fid=s.Wa();s.hb();s.usePlugins&&s.doPlugins&&s.doPlugins(s);if(s.account){if(!s.abort){if(s.trackOffline&&!s.timestamp)s.timestamp=Math.floor(e.getTime()/1E3);e=w.location;if(!s.pageURL)s.pageURL=e.href?e.href:e;if(!s.referrer&&!s.Ba)s.referrer=m.document.referrer,s.Ba=1;s.referrer=s.Ua(s.referrer);s.q("_g")}if(s.Ya()&&!s.abort)s.Za(),f+=\ns.Xa(),s.gb(d,f),s.q("_t"),s.referrer=""}}b&&s.K(c,1)}s.abort=s.supplementalDataID=s.timestamp=s.pageURLRest=s.linkObject=s.clickObject=s.linkURL=s.linkName=s.linkType=w.vb=s.pe=s.pev1=s.pev2=s.pev3=s.g=0}};s.tl=s.trackLink=function(b,a,c,e,d){s.linkObject=b;s.linkType=a;s.linkName=c;if(d)s.i=b,s.p=d;return s.track(e)};s.trackLight=function(b,a,c,e){s.lightProfileID=b;s.lightStoreForSeconds=a;s.lightIncrementBy=c;return s.track(e)};s.clearVars=function(){var b,a;for(b=0;b<s.c.length;b++)if(a=s.c[b],\na.substring(0,4)=="prop"||a.substring(0,4)=="eVar"||a.substring(0,4)=="hier"||a.substring(0,4)=="list"||a=="channel"||a=="events"||a=="eventList"||a=="products"||a=="productList"||a=="purchaseID"||a=="transactionID"||a=="state"||a=="zip"||a=="campaign")s[a]=void 0};s.tagContainerMarker="";s.gb=function(b,a){var c,e=s.trackingServer;c="";var d=s.dc,f="sc.",w=s.visitorNamespace;if(e){if(s.trackingServerSecure&&s.ssl)e=s.trackingServerSecure}else{if(!w)w=s.account,e=w.indexOf(","),e>=0&&(w=w.substring(0,\ne)),w=w.replace(/[^A-Za-z0-9]/g,"");c||(c="2o7.net");d=d?(""+d).toLowerCase():"d1";c=="2o7.net"&&(d=="d1"?d="112":d=="d2"&&(d="122"),f="");e=w+"."+d+"."+f+c}c=s.ssl?"https://":"http://";d=s.AudienceManagement&&s.AudienceManagement.isReady();c+=e+"/b/ss/"+s.account+"/"+(s.mobile?"5.":"")+(d?"10":"1")+"/JS-"+s.version+(s.mb?"T":"")+(s.tagContainerMarker?"-"+s.tagContainerMarker:"")+"/"+b+"?AQB=1&ndh=1&pf=1&"+(d?"callback=s_c_il["+s._in+"].AudienceManagement.passData&":"")+a+"&AQE=1";s.Sa(c);s.Y()};\ns.Sa=function(b){s.e||s.$a();s.e.push(b);s.aa=s.r();s.za()};s.$a=function(){s.e=s.cb();if(!s.e)s.e=[]};s.cb=function(){var b,a;if(s.fa()){try{(a=w.localStorage.getItem(s.da()))&&(b=w.JSON.parse(a))}catch(c){}return b}};s.fa=function(){var b=!0;if(!s.trackOffline||!s.offlineFilename||!w.localStorage||!w.JSON)b=!1;return b};s.pa=function(){var b=0;if(s.e)b=s.e.length;s.v&&b++;return b};s.Y=function(){if(!s.v)if(s.qa=k,s.ea)s.aa>s.G&&s.xa(s.e),s.ha(500);else{var b=s.Ma();if(b>0)s.ha(b);else if(b=s.na())s.v=\n1,s.fb(b),s.jb(b)}};s.ha=function(b){if(!s.qa)b||(b=0),s.qa=setTimeout(s.Y,b)};s.Ma=function(){var b;if(!s.trackOffline||s.offlineThrottleDelay<=0)return 0;b=s.r()-s.wa;if(s.offlineThrottleDelay<b)return 0;return s.offlineThrottleDelay-b};s.na=function(){if(s.e.length>0)return s.e.shift()};s.fb=function(b){if(s.debugTracking){var a="AppMeasurement Debug: "+b;b=b.split("&");var c;for(c=0;c<b.length;c++)a+="\\n\\t"+s.unescape(b[c]);s.eb(a)}};s.Ha=function(){return s.marketingCloudVisitorID||s.analyticsVisitorID};\ns.Q=!1;var n;try{n=JSON.parse(\'{"x":"y"}\')}catch(r){n=null}n&&n.x=="y"?(s.Q=!0,s.P=function(s){return JSON.parse(s)}):w.$&&w.$.parseJSON?(s.P=function(s){return w.$.parseJSON(s)},s.Q=!0):s.P=function(){return null};s.jb=function(b){var a,c,e;if(s.Ha()&&b.length>2047&&(typeof XMLHttpRequest!="undefined"&&(a=new XMLHttpRequest,"withCredentials"in a?c=1:a=0),!a&&typeof XDomainRequest!="undefined"&&(a=new XDomainRequest,c=2),a&&s.AudienceManagement&&s.AudienceManagement.isReady()))s.Q?a.ja=!0:a=0;!a&&\ns.ab&&(b=b.substring(0,2047));if(!a&&s.d.createElement&&s.AudienceManagement&&s.AudienceManagement.isReady()&&(a=s.d.createElement("SCRIPT"))&&"async"in a)(e=(e=s.d.getElementsByTagName("HEAD"))&&e[0]?e[0]:s.d.body)?(a.type="text/javascript",a.setAttribute("async","async"),c=3):a=0;if(!a)a=new Image,a.alt="";a.la=function(){try{if(s.ga)clearTimeout(s.ga),s.ga=0;if(a.timeout)clearTimeout(a.timeout),a.timeout=0}catch(b){}};a.onload=a.lb=function(){a.la();s.Ra();s.V();s.v=0;s.Y();if(a.ja){a.ja=!1;try{var b=\ns.P(a.responseText);AudienceManagement.passData(b)}catch(c){}}};a.onabort=a.onerror=a.Ta=function(){a.la();(s.trackOffline||s.ea)&&s.v&&s.e.unshift(s.Qa);s.v=0;s.aa>s.G&&s.xa(s.e);s.V();s.ha(500)};a.onreadystatechange=function(){a.readyState==4&&(a.status==200?a.lb():a.Ta())};s.wa=s.r();if(c==1||c==2){var d=b.indexOf("?");e=b.substring(0,d);d=b.substring(d+1);d=d.replace(/&callback=[a-zA-Z0-9_.\\[\\]]+/,"");c==1?(a.open("POST",e,!0),a.send(d)):c==2&&(a.open("POST",e),a.send(d))}else if(a.src=b,c==3){if(s.ua)try{e.removeChild(s.ua)}catch(f){}e.firstChild?\ne.insertBefore(a,e.firstChild):e.appendChild(a);s.ua=s.Pa}if(a.abort)s.ga=setTimeout(a.abort,5E3);s.Qa=b;s.Pa=w["s_i_"+s.replace(s.account,",","_")]=a;if(s.useForcedLinkTracking&&s.A||s.p){if(!s.forcedLinkTrackingTimeout)s.forcedLinkTrackingTimeout=250;s.W=setTimeout(s.V,s.forcedLinkTrackingTimeout)}};s.Ra=function(){if(s.fa()&&!(s.va>s.G))try{w.localStorage.removeItem(s.da()),s.va=s.r()}catch(b){}};s.xa=function(b){if(s.fa()){s.za();try{w.localStorage.setItem(s.da(),w.JSON.stringify(b)),s.G=s.r()}catch(a){}}};\ns.za=function(){if(s.trackOffline){if(!s.offlineLimit||s.offlineLimit<=0)s.offlineLimit=10;for(;s.e.length>s.offlineLimit;)s.na()}};s.forceOffline=function(){s.ea=!0};s.forceOnline=function(){s.ea=!1};s.da=function(){return s.offlineFilename+"-"+s.visitorNamespace+s.account};s.r=function(){return(new Date).getTime()};s.ra=function(s){s=s.toLowerCase();if(s.indexOf("#")!=0&&s.indexOf("about:")!=0&&s.indexOf("opera:")!=0&&s.indexOf("javascript:")!=0)return!0;return!1};s.setTagContainer=function(b){var a,\nc,e;s.mb=b;for(a=0;a<s._il.length;a++)if((c=s._il[a])&&c._c=="s_l"&&c.tagContainerName==b){s.K(c);if(c.lmq)for(a=0;a<c.lmq.length;a++)e=c.lmq[a],s.loadModule(e.n);if(c.ml)for(e in c.ml)if(s[e])for(a in b=s[e],e=c.ml[e],e)if(!Object.prototype[a]&&(typeof e[a]!="function"||(""+e[a]).indexOf("s_c_il")<0))b[a]=e[a];if(c.mmq)for(a=0;a<c.mmq.length;a++)e=c.mmq[a],s[e.m]&&(b=s[e.m],b[e.f]&&typeof b[e.f]=="function"&&(e.a?b[e.f].apply(b,e.a):b[e.f].apply(b)));if(c.tq)for(a=0;a<c.tq.length;a++)s.track(c.tq[a]);\nc.s=s;break}};s.Util={urlEncode:s.escape,urlDecode:s.unescape,cookieRead:s.cookieRead,cookieWrite:s.cookieWrite,getQueryParam:function(b,a,c){var e;a||(a=s.pageURL?s.pageURL:w.location);c||(c="&");if(b&&a&&(a=""+a,e=a.indexOf("?"),e>=0&&(a=c+a.substring(e+1)+c,e=a.indexOf(c+b+"="),e>=0&&(a=a.substring(e+c.length+b.length+1),e=a.indexOf(c),e>=0&&(a=a.substring(0,e)),a.length>0))))return s.unescape(a);return""}};s.z=["supplementalDataID","timestamp","dynamicVariablePrefix","visitorID","marketingCloudVisitorID",\n"analyticsVisitorID","audienceManagerLocationHint","authState","fid","vmk","visitorMigrationKey","visitorMigrationServer","visitorMigrationServerSecure","charSet","visitorNamespace","cookieDomainPeriods","fpCookieDomainPeriods","cookieLifetime","pageName","pageURL","referrer","contextData","currencyCode","lightProfileID","lightStoreForSeconds","lightIncrementBy","retrieveLightProfiles","deleteLightProfiles","retrieveLightData","pe","pev1","pev2","pev3","pageURLRest"];s.c=s.z.concat(["purchaseID",\n"variableProvider","channel","server","pageType","transactionID","campaign","state","zip","events","events2","products","audienceManagerBlob","tnt"]);s.ba=["timestamp","charSet","visitorNamespace","cookieDomainPeriods","cookieLifetime","contextData","lightProfileID","lightStoreForSeconds","lightIncrementBy"];s.H=s.ba.slice(0);s.ia=["account","allAccounts","debugTracking","visitor","trackOffline","offlineLimit","offlineThrottleDelay","offlineFilename","usePlugins","doPlugins","configURL","visitorSampling",\n"visitorSamplingGroup","linkObject","clickObject","linkURL","linkName","linkType","trackDownloadLinks","trackExternalLinks","trackClickMap","trackInlineStats","linkLeaveQueryString","linkTrackVars","linkTrackEvents","linkDownloadFileTypes","linkExternalFilters","linkInternalFilters","useForcedLinkTracking","forcedLinkTrackingTimeout","trackingServer","trackingServerSecure","ssl","abort","mobile","dc","lightTrackVars","maxDelay","expectSupplementalData","AudienceManagement"];for(i=0;i<=250;i++)i<76&&\n(s.c.push("prop"+i),s.H.push("prop"+i)),s.c.push("eVar"+i),s.H.push("eVar"+i),i<6&&s.c.push("hier"+i),i<4&&s.c.push("list"+i);i=["latitude","longitude","resolution","colorDepth","javascriptVersion","javaEnabled","cookiesEnabled","browserWidth","browserHeight","connectionType","homepage"];s.c=s.c.concat(i);s.z=s.z.concat(i);s.ssl=w.location.protocol.toLowerCase().indexOf("https")>=0;s.charSet="UTF-8";s.contextData={};s.offlineThrottleDelay=0;s.offlineFilename="AppMeasurement.offline";s.wa=0;s.aa=0;\ns.G=0;s.va=0;s.linkDownloadFileTypes="exe,zip,wav,mp3,mov,mpg,avi,wmv,pdf,doc,docx,xls,xlsx,ppt,pptx";s.w=w;s.d=w.document;try{s.ab=navigator.appName=="Microsoft Internet Explorer"}catch(t){}s.V=function(){if(s.W)w.clearTimeout(s.W),s.W=k;s.i&&s.A&&s.i.dispatchEvent(s.A);if(s.p)if(typeof s.p=="function")s.p();else if(s.i&&s.i.href)s.d.location=s.i.href;s.i=s.A=s.p=0};s.ya=function(){s.b=s.d.body;if(s.b)if(s.o=function(b){var a,c,e,d,f;if(!(s.d&&s.d.getElementById("cppXYctnr")||b&&b["s_fe_"+s._in])){if(s.ka)if(s.useForcedLinkTracking)s.b.removeEventListener("click",\ns.o,!1);else{s.b.removeEventListener("click",s.o,!0);s.ka=s.useForcedLinkTracking=0;return}else s.useForcedLinkTracking=0;s.clickObject=b.srcElement?b.srcElement:b.target;try{if(s.clickObject&&(!s.F||s.F!=s.clickObject)&&(s.clickObject.tagName||s.clickObject.parentElement||s.clickObject.parentNode)){var g=s.F=s.clickObject;if(s.Z)clearTimeout(s.Z),s.Z=0;s.Z=setTimeout(function(){if(s.F==g)s.F=0},1E4);e=s.pa();s.track();if(e<s.pa()&&s.useForcedLinkTracking&&b.target){for(d=b.target;d&&d!=s.b&&d.tagName.toUpperCase()!=\n"A"&&d.tagName.toUpperCase()!="AREA";)d=d.parentNode;if(d&&(f=d.href,s.ra(f)||(f=0),c=d.target,b.target.dispatchEvent&&f&&(!c||c=="_self"||c=="_top"||c=="_parent"||w.name&&c==w.name))){try{a=s.d.createEvent("MouseEvents")}catch(i){a=new w.MouseEvent}if(a){try{a.initMouseEvent("click",b.bubbles,b.cancelable,b.view,b.detail,b.screenX,b.screenY,b.clientX,b.clientY,b.ctrlKey,b.altKey,b.shiftKey,b.metaKey,b.button,b.relatedTarget)}catch(k){a=0}if(a)a["s_fe_"+s._in]=a.s_fe=1,b.stopPropagation(),b.kb&&b.kb(),\nb.preventDefault(),s.i=b.target,s.A=a}}}}else s.clickObject=0}catch(m){s.clickObject=0}}},s.b&&s.b.attachEvent)s.b.attachEvent("onclick",s.o);else{if(s.b&&s.b.addEventListener){if(navigator&&(navigator.userAgent.indexOf("WebKit")>=0&&s.d.createEvent||navigator.userAgent.indexOf("Firefox/2")>=0&&w.MouseEvent))s.ka=1,s.useForcedLinkTracking=1,s.b.addEventListener("click",s.o,!0);s.b.addEventListener("click",s.o,!1)}}else setTimeout(s.ya,30)};s.ya()}\nfunction s_gi(s){var w,k=window.s_c_il,m,i,o=s.split(","),p,n,r=0;if(k)for(m=0;!r&&m<k.length;){w=k[m];if(w._c=="s_c"&&(w.account||w.oun))if(w.account&&w.account==s)r=1;else{i=w.account?w.account:w.oun;i=w.allAccounts?w.allAccounts:i.split(",");for(p=0;p<o.length;p++)for(n=0;n<i.length;n++)o[p]==i[n]&&(r=1)}m++}r||(w=new AppMeasurement);w.setAccount?w.setAccount(s):w.sa&&w.sa(s);return w}AppMeasurement.getInstance=s_gi;window.s_objectID||(window.s_objectID=0);\nfunction s_pgicq(){var s=window,w=s.s_giq,k,m,i;if(w)for(k=0;k<w.length;k++)m=w[k],i=s_gi(m.oun),i.setAccount(m.un),i.setTagContainer(m.tagContainerName);s.s_giq=0}s_pgicq();\n</script>'
                }
                ]
            }
            ],
            event: "pagetop"
        }, {
            name: "~9 - DTM Plugins",
            trigger: [{
                command: "writeHTML",
                arguments: [{
                    html: '<script type="text/javascript">\n/*! dtm-plugins 2014-11-13 11:24:29 */\n!function(a){"use strict";a._getDTMWebPropertyName=function(){for(var a,b,c,d=document.getElementsByTagName("script"),e=/^(.*)satelliteLib-(.*)\\.js$/,f=/^(.*)satelliteLib-(.*)-staging\\.js$/,g={"46e65db5bb0c375f8f64619be31cc9b29acf4867":"~ Global Container",d6f79f42e6a413a42094a79ac480aea348f592ac:"ADBADOBE Microsite Shared","6d5cdee78e7fff2a60e0d0464002fbb53b803460":"~ Adobe.com Re-imagined","738a3482bfd34fd3f5b7aa8622233c010f4c8ae1":"~ creative.adobe.com"},h=0,i=d.length;i>h&&(c=d[h].getAttribute("src"),!c||(a||(a=c.match(e)),b||(b=c.match(f)),!b));h++);return a&&!b?g[a[2]]:g[b[2]]},a._getDTMWebPropertyName.version="1.0"}(_satellite),function(a){"use strict";a._getDTMEnvironment=function(){for(var a,b,c,d=document.getElementsByTagName("script"),e=/^(.*)satelliteLib-(.*)\\.js$/,f=/^(.*)satelliteLib-(.*)-staging\\.js$/,g=0,h=d.length;h>g&&(c=d[g].getAttribute("src"),!c||(a||(a=c.match(e)),b||(b=c.match(f)),!b));g++);return a&&!b?"production":"staging"},a._getDTMEnvironment.version="1.0"}(_satellite),function(a){"use strict";a._useStagingLibrary=function(a){a?window.localStorage.setItem("sdsat_stagingLibrary","true"):(window.localStorage.setItem("sdsat_stagingLibrary","false"),window.localStorage.removeItem("sdsat_stagingLibrary"))},a._useStagingLibrary.version="1.0"}(_satellite),function(a){"use strict";var b=0;a._addEventBasedRule=function(c){return"object"!=typeof c||"string"!=typeof c.selector||"string"!=typeof c.event?!1:(b++,"undefined"!=typeof a.rules&&a.rules.push({name:c.name||"dynamically added rule #"+b,selector:c.selector,event:c.event,bubbleFireIfParent:("undefined"!=typeof c.bubbleFireIfParent?c.bubbleFireIfParent:!0)?!0:!1,bubbleFireIfChildFired:("undefined"!=typeof c.bubbleFireIfChildFired?c.bubbleFireIfChildFired:!0)?!0:!1,bubbleStop:("undefined"!=typeof c.bubbleStop?c.bubbleStop:!1)?!0:!1,conditions:c.conditions,trigger:c.trigger}),!0)},a._addEventBasedRule.version="1.0"}(_satellite),function(a){"use strict";var b,c="1.0",d=[],e=1,f=function(a){var b,c,d=!0;for(b=0,c=a.dependencies.length;c>b;b++)d="function"==typeof a.dependencies[b]?d&&a.dependencies[b]():d&&"undefined"!=typeof a.dependencies[b];return d},g=function(){var a,c,h=(new Date).getTime(),i=!1;for(a=d.length-1;a>=0;a--)c=d[a],h<c.config._interval||(f(c)?(c.func(),d.splice(a,1),i=c.config.interval==e):h>c.config._timeout?(d.splice(a,1),i=c.config.interval==e):c.config._interval+=c.config.interval);if(d.length<1)clearInterval(b),b=null;else if(i){for(e=d[0].config.interval,a=d.length-1;a>=0;a--)c=d[a],c.config.interval<e&&(e=c.config.interval);clearInterval(b),b=setInterval(g,e)}};a._runWhenResolved=function(a,c,h){var i,j=(new Date).getTime();h.timeout=h.timeout||3e3,h._timeout=j+h.timeout,h.interval=h.interval||1,h._interval=j+h.interval,i={func:a,dependencies:c,config:h},f(i)?a():(d.unshift(i),b?h.interval<e&&(clearInterval(b),e=h.interval,b=setInterval(g,e)):(e=h.interval,b=setInterval(g,e)))},a._runWhenResolved.version=c}(_satellite),function(a){"use strict";a._appendList=a._apl=function(a,b,c,d){var e,f,g,h,i,j=a&&"string"==typeof a?a.split(c):[],k=b&&"string"==typeof b?b.split(c):[];for(e=0;e<k.length;e++)if(f=k[e],d){for(i=!1,g=0;g<j.length&&(h=j[g],!(i=1==d?f==h:f.toLowerCase()==h.toLowerCase()));g++);i||j.push(f)}else j.push(f);return j.join(c)},a._appendList.version="1.0",a._apl.version="1.0"}(_satellite),function(a){"use strict";var b="";a._getDomain=function(){if(b)return b;var a=window.location.hostname.toLowerCase().split("."),c=[],d="",e=null,f=!1;for(d=a.pop(),c.unshift(d);a.length>0;)if(d=a.pop(),c.unshift(d),e=new Date,e.setTime(e.getTime()+1e3),document.cookie=["sat_domain","=","A","; expires="+e.toUTCString(),"; domain="+c.join("."),"; path=/"].join(""),"A"===this.readCookie("sat_domain")){f=!0,b=c.join(".");break}return f?b:""},a._getDomain.version="1.2"}(_satellite),function(a){"use strict";a._getDomainPeriods=function(){var a=this._getDomain();return a?a.split(".").length:2},a._getDomainPeriods.version="1.0"}(_satellite),function(a){"use strict";a._getAdobeAnalyticsTool=function(b){var c=a.tools,d="";for(d in c)if("sc"==c[d].settings.engine&&c[d].settings.renameS==b)return c[d];return null},a._getAdobeAnalyticsTool.version="1.0"}(_satellite),function(a){"use strict";a._getAdobeAnalyticsAccount=function(b){var c=a._getAdobeAnalyticsTool(b);return c?c.settings.account:""},a._getAdobeAnalyticsAccount.version="1.2"}(_satellite),function(a){"use strict";a._replace=function(a,b,c){return!a||a.indexOf(b)<0?a:a.split(b).join(c)},a._replace.version="1.0"}(_satellite),function(a){"use strict";a._escape=function(b){var c,d,e,f=this===window?a:this,g="+~!*()\'";if(!b)return b;for(c=encodeURIComponent(b),d=0;d<g.length;d++)e=g.substring(d,d+1),c.indexOf(e)>=0&&(c=f._replace(c,e,"%"+e.charCodeAt(0).toString(16).toUpperCase()));return c},a._escape.version="1.0"}(_satellite),function(a){"use strict";a._unescape=function(b){var c,d=this===window?a:this;if(!b)return b;c=b.indexOf("+")>=0?d._replace(b,"+"," "):b;try{return decodeURIComponent(c)}catch(e){}return unescape(c)},a._unescape.version="1.0"}(_satellite),function(a){"use strict";a._getExpirationDate=function(a){var b,c;return a&&a instanceof Date&&!isNaN(a.valueOf())||(a=parseInt(a,10),a?1===a?(a=new Date,b=a.getYear(),a.setYear(b+5+(1900>b?1900:0))):(c=new Date,c.setTime(c.getTime()+a),a=c):a=""),a},a._getExpirationDate.version="1.0"}(_satellite),function(a){"use strict";a._readCookie=function(b){var c=this===window?a:this;b=c._escape(b);var d=" "+document.cookie,e=d.indexOf(" "+b+"="),f=0>e?e:d.indexOf(";",e),g=0>e?"":c._unescape(d.substring(e+2+b.length,0>f?d.length:f));return"[[B]]"!=g?g:""},a._readCookie.version="1.0"}(_satellite),function(a){"use strict";a._setCookie=function(b,c,d,e,f){var g=this===window?a:this;return e=e?e:g._getDomain(),f=f?f:"/",c=""+c,d=g._getExpirationDate(d),d=d?d:"",b?(document.cookie=b+"="+g._escape(""!==c?c:"[[B]]")+"; path="+f+(d?"; expires="+d.toGMTString():"")+(e?"; domain="+e:""),g._readCookie(b)==c):0},a._setCookie.version="1.0"}(_satellite),function(a){"use strict";var b=a._readCookie;a._readCombinedCookie=function(c,d){if(d)return b.call(this,c);var e,f,g,h,i,j=this===window?a:this,k=b.call(j,c),l="",m="",n=(new Date).getTime(),o=null,p=[];if(k)return k;for(l=b.call(j,"s_pers"),l=l?l:"",p=l.split(";"),e=0,i=p.length;i>e;e++)o=p[e].match(/\\|([0-9]+)$/),o&&parseInt(o[1],10)>=n&&(m+=p[e]+";");return l=m,c=j._escape(c),f=l.indexOf(" "+c+"="),l=0>f?b.call(j,"s_sess"):l,f=l.indexOf(" "+c+"="),g=0>f?f:l.indexOf("|",f),h=0>f?f:l.indexOf(";",f),g=g>0?g:h,k=0>f?"":j._unescape(l.substring(f+2+c.length,0>g?l.length:g))},a._readCookie=a._readCombinedCookie,a._readCombinedCookie.version="1.0"}(_satellite),function(a){"use strict";var b=a._setCookie;a._setCombinedCookie=function(c,d,e,f,g,h){if(h)return b.call(this,c,d,e,f,g);var i,j,k,l,m,n,o,p,q=this===window?a:this,r=new Date,s=(new Date).getTime(),t=0,u="s_pers",v="s_sess",w="",x=[],y=null,z=0,A=0;for(r.setTime(r.getTime()-6e4),q._readCookie(c,!0)&&b.call(q,c,"",r),m=q._readCookie("s_pers",!0),m=m?m:"",x=m.split(";"),i=0,j=x.length;j>i;i++)y=x[i].match(/\\|([0-9]+)$/),y&&parseInt(y[1],10)>=s&&(w+=x[i]+";");if(k=w,c=q._escape(c),n=k.indexOf(" "+c+"="),n>-1&&(k=k.substring(0,n)+k.substring(k.indexOf(";",n)+1),z=1),l=q._readCookie(v,!0),n=l.indexOf(" "+c+"="),n>-1&&(l=l.substring(0,n)+l.substring(l.indexOf(";",n)+1),A=1),e=q._getExpirationDate(e),e=e?e:"",r=new Date,e?e.getTime()>r.getTime()&&(k+=" "+c+"="+q._escape(d)+"|"+e.getTime()+";",z=1):(l+=" "+c+"="+q._escape(d)+";",A=1),l=l.replace(/%00/g,""),k=k.replace(/%00/g,""),A&&b.call(q,v,l,0),z){for(o=k;o&&-1!=o.indexOf(";");)p=parseInt(o.substring(o.indexOf("|")+1,o.indexOf(";")),10),o=o.substring(o.indexOf(";")+1),t=p>t?p:t;r.setTime(t),b.call(q,u,k,r)}return d==q._readCombinedCookie(q._unescape(c))},a._setCookie=a._setCombinedCookie,a._setCombinedCookie.version="1.0"}(_satellite),function(a){"use strict";var b="1.0",c=!1,d="",e=function(a,b,c){var e=document.createElement("SCRIPT"),f=document.getElementsByTagName("HEAD"),g="_initBrownies_"+Math.floor(1e7*Math.random()),h=function(){window[g]=void 0;try{delete window[g]}catch(a){}};f=f&&f[0]?f[0]:document.body,f&&(window[g]=function(a){c&&c(a),h()},e.type="text/javascript",e.src="//"+d+"/"+a+"?data="+encodeURIComponent(JSON.stringify(b))+"&callback="+g,f.firstChild?f.insertBefore(e,f.firstChild):f.appendChild(e))},f={},g={aid:1,mid:1,adobeguid:1};a._initBrownies=function(a,b){var h,i,j={};if(d=a,"object"==typeof b&&null!==b)for(h in b)if(b.hasOwnProperty(h)&&g[h]&&(j[h]={},"object"==typeof b[h]&&null!==b[h]))for(i in b[h])b[h].hasOwnProperty(i)&&(j[h][i]=1);e("get",b,function(a){f=a,c=!0})},a._readBrownie=function(a,b,c){var d=new Date;return f&&f[a]&&f[a][b]&&f[a][b][c]&&d.getTime()<f[a][b][c].d+f[a][b][c].e?f[a][b][c].v:void 0},a._setBrownie=a._updateBrownie=function(a,b,c,d,h){var i={},j=new Date;return h=h||864e5,g[a]?(f[a]||(f[a]={}),i[a]={},f[a][b]||(f[a][b]={}),i[a][b]={},f[a][b][c]||(f[a][b][c]={}),i[a][b][c]={},f[a][b][c].v=d,f[a][b][c].d=j.getTime(),f[a][b][c].e=h,i[a][b][c].v=d,i[a][b][c].d=j.getTime(),i[a][b][c].e=h,e("set",i),!0):!1},a._removeBrownie=a._deleteBrownie=function(a,b,c){var d={};return g[a]?(d[a]={},d[a][b]={},d[a][b][c]=!0,e("delete",d),!1):!1},a._initBrownies.version=b,a._readBrownie.version=b,a._setBrownie.version=b,a._updateBrownie.version=b,a._removeBrownie.version=b,a._deleteBrownie.version=b}(_satellite),function(a){"use strict";a._returnValueWhenNew=a._getValOnce=function(b,c,d,e){var f,g=this===window?a:this,h=new Date,i="m"==e?6e4:864e5;return b=b?b:"",c=c?c:"s_gvo",d=d?d:0,f=g._readCookie(c),b&&(h.setTime(h.getTime()+d*i),g._setCookie(c,b,0===d?0:h)),b==f?"":b},a._returnValueWhenNew.version="1.0",a._getValOnce.version="1.0"}(_satellite),function(a){"use strict";a._getVisitStart=function(b){var c=this===window?a:this,d=1,e=new Date;return e.setTime(e.getTime()+18e5),c._readCookie(b)&&(d=0),c._setCookie(b,1,e)||c._setCookie(b,1,0),c._readCookie(b)||(d=0),d},a._getVisitStart.version="1.0"}(_satellite),function(a){"use strict";a._getNewRepeat=function(b,c){var d,e,f=this===window?a:this,g=new Date,h=g.getTime();return b=b?b:30,c=c?c:"s_nr",g.setTime(h+24*b*60*60*1e3),d=f._readCookie(c),0===d.length?(f._setCookie(c,h+"-New",g),"New"):(e=d.split("-"),h-e[0]<18e5&&"New"==e[1]?(f._setCookie(c,h+"-New",g),"New"):(f._setCookie(c,h+"-Repeat",g),"Repeat"))},a._getNewRepeat.version="1.0"}(_satellite),function(a){"use strict";a._getPreviousValue=function(b,c){var d=this===window?a:this,e=new Date,f="";return e.setTime(e.getTime()+18e5),d._readCookie(c)&&(f=d._readCookie(c)),b?d._setCookie(c,b,e):d._setCookie(c,"",e),f},a._getPreviousValue.version="1.0"}(_satellite),function(a){"use strict";var b={2012:"3/11,11/4",2013:"3/10,11/3",2014:"3/9,11/2",2015:"3/8,11/1",2016:"3/13,11/6",2017:"3/12,11/5",2018:"3/11,11/4",2019:"3/10,11/3"};a._getTimeParting=function(a,c){var d;if(d=new Date("1/1/2000"),6!==d.getDay()||0!==d.getMonth())return"Data Not Available";var e,f,g,h,i,j,k,l,m,n=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],o=new Date,p=[];if(c=c?c:0,c=parseFloat(c),b){var q=b[o.getFullYear()].split(/,/);j=new Date(q[0]+"/"+o.getFullYear()),k=new Date(q[1]+"/"+o.getFullYear()),"n"==a&&o>j&&k>o?c+=1:"s"==a&&(o>k||j>o)&&(c+=1)}return o=o.getTime()+6e4*o.getTimezoneOffset(),o=new Date(o+36e5*c),e=o.getHours(),f=o.getMinutes(),f=10>f?"0"+f:f,g=o.getDay(),i="AM",h="Weekday",e>=12&&(i="PM",e-=12),0===e&&(e=12),(6===g||0===g)&&(h="Weekend"),g=n[g],l=e+":"+f+i,m=e+":"+(f>30?"30":"00")+i,p=[l,m,g,h]},a._getTimeParting.version="1.0"}(_satellite),function(a){"use strict";var b,c=!1;a._getPercentPageViewed=function(d,e){var f,g,h=d&&"string"==typeof d&&"-"!==d?d:window.location.href,i=["load","unload","scroll","resize","zoom","keyup","mouseup","touchend","orientationchange","pan"],j=0,k=0,l=0;if(!c){for(b=function(b,c){var d=a,f=e?e:"s_ppv",g=f+"l",h=d._readCookie(b||c?f:g),i=h.indexOf(",")>-1?h.split(",",10):[""],j=i.length,k=0;for(i[0]=d._unescape(i[0]),c=c||b&&b!=i[0]||0,i.length=10,"string"!=typeof i[0]&&(i[0]=""),k=1;10>k;k++)i[k]=!c&&j>k?parseInt(i[k],10)||0:0;return(10>j||"string"!=typeof i[9])&&(i[9]=""),c&&(d._setCookie(g,h),d._setCookie(f,"?")),i},f=function(c,d){var g=a,h=e?e:"s_ppv",i=document.body,l=document.documentElement,m=window.screen||0,n="offsetHeight",o="scrollHeight",p="scrollTop",q="clientWidth",r="clientHeight",s=100,t=Math,u="object",v="number";if(c=c&&typeof c==u?c.type||"":"",c.indexOf("on")||(c=c.substring(2)),j=j||0,k&&!c&&(clearTimeout(k),k=0,2>j&&j++),typeof g==u){var w=t.max(i[o]||l[o],i[n]||l[n],i[r]||l[r]),x=window.innerWidth||l[q]||i[q]||0,y=window.innerHeight||l[r]||i[r]||0,z=m?m.width:0,A=m?m.height:0,B=t.round(s*(window.devicePixelRatio||1))/s,C=(document.pageYOffset||l[p]||i[p]||0)+y,D=w>0&&C>0?t.round(s*C/w):0,E=window.orientation,F=isNaN(E)?y>x?0:90:t.abs(E)%180,G="load"==c||1>j,H=b(d,G),I=function(a,b,c,d){return a=parseInt(typeof H==u&&H.length>a?H[a]:"0",10)||0,b=typeof b!=v?a:b,b=c||b>a?b:a,d?b:b>s?s:0>b?0:b};new RegExp("(iPod|iPad|iPhone)").exec(navigator.userAgent||"")&&F&&(F=z,z=A,A=F),F=F?"P":"L",H[9]=G?"":H[9].substring(0,1),g._setCookie(h,g._escape(d)+","+I(1,D,G)+","+(G||!I(2)?D:I(2))+","+I(3,C,G,1)+","+x+","+y+","+z+","+A+","+B+","+H[9]+(H[9]==F?"":F))}k||"unload"==c||(k=setTimeout(function(a){f(a,d)},333))},g=function(a){f(a,h)},l=0;l<i.length;l++)window.addEventListener?window.addEventListener(i[l],g,!1):window.attachEvent&&window.attachEvent("on"+i[l],g);c=!0,f(null,h)}var m=b();return""===m[0]?"":d&&"-"!=d?m:m[1]},a._getPercentPageViewed.version="1.1"}(_satellite);\n</script>'
                }
                ]
            }
            ],
            event: "pagetop"
        }
        ],
        rules: [{
            name: "adbadobenonacdc - Download Trial Click",
            trigger: [{
                engine: "sc",
                command: "trackPageView",
                arguments: [{
                    setVars: {
                        eVar44: "D=c2",
                        pageName: "adobe.com:Products:OnClick_DownloadTrial_from_ProductOverviewPage"
                    },
                    addEvent: ["event35"]
                }
                ]
            }, {
                command: "delayActivateLink"
            }
            ],
            conditions: [function() {
                return _satellite.getVar("adbadobenonacdc_event35_condition")
            }
            ],
            selector: "a.innav-button.dom-stateful-download-link",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "adbadobenonacdc - HPFooter - Marketing Cloud Click",
            trigger: [{
                engine: "sc",
                command: "trackPageView",
                arguments: [{
                    setVars: {
                        pageName: "adobe.com:HomePage:onClick_MCFooterButton",
                        channel: "%adbadobenonacdc_channel_Site_Section%"
                    }
                }
                ]
            }
            ],
            selector: "li[data-dom-lobby-view=marketing-cloud]:eq(1)",
            event: "click",
            bubbleFireIfParent: !1,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "adbadobenonacdc - Ink and Slide - Buy Button Click",
            trigger: [{
                engine: "sc",
                command: "trackPageView",
                arguments: [{
                    setVars: {
                        pageName: "adobe.com:products:ink-and-slide:onClick_BuyNow"
                    }
                }
                ]
            }
            ],
            selector: "a[data-reveal-id=inkslide-buy-modal]",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "adbadobenonacdc - Ink and Slide - Continue to Adonit.com form submit",
            trigger: [{
                engine: "sc",
                command: "trackPageView",
                arguments: [{
                    setVars: {
                        pageName: "adobe.com:products:ink-and-slide:AdonitModal:onClick_ContinueToAdonit"
                    },
                    customSetup: function(t) {
                        "boolean" == typeof e._trackedButtonClick || e._trackedButtonClick || (e._trackedButtonClick=!1, t.preventDefault(), setTimeout(function() {
                            e._trackedButtonClick=!0, $("#buy-now").submit()
                        }, 500))
                    },
                    addEvent: ["event60"]
                }
                ]
            }
            ],
            conditions: [function() {
                return "boolean" != typeof e._trackedButtonClick
            }
            ],
            selector: "#buy-now",
            event: "submit",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/buy/students.html (bottom of page video)",
            trigger: [{
                command: "loadIframe",
                arguments: [{
                    pages: [{
                        src: "satellite-53e15d31be963cedae000078.html",
                        data: []
                    }
                    ]
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud/buy/students.html")
            }
            ],
            selector: "a.dom-cc-seg-student-video-button",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/buy/students.html (top of page video) ",
            trigger: [{
                command: "loadIframe",
                arguments: [{
                    pages: [{
                        src: "satellite-53e15c9d49e9ca0dfb000734.html",
                        data: []
                    }
                    ]
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud/buy/students.html")
            }
            ],
            selector: "a.dom-cc-seg-hero-video-button",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/students/make-it-with-creative-cloud.html (Animation tab Video)",
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud/students/make-it-with-creative-cloud.html")
            }
            ],
            selector: "a",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            property: {
                "data-video-url": /video.tv.adobe.com\/v\/231\//i,
                className: "section-leaf-button"
            },
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/students/make-it-with-creative-cloud.html (Illustration tab Video)",
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud/students/make-it-with-creative-cloud.html")
            }
            ],
            selector: "a",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            property: {
                "data-video-url": /video.tv.adobe.com\/v\/228\//i,
                className: "section-leaf-button"
            },
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/students/make-it-with-creative-cloud.html (Photography tab Video)",
            selector: "a",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            property: {
                className: "section-leaf-button",
                "data-video-url": /video.tv.adobe.com\/v\/220\//i
            },
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/students/make-it-with-creative-cloud.html (Special Effects tab Video)",
            selector: "a",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            property: {
                className: "section-leaf-button",
                "data-video-url": /video.tv.adobe.com\/v\/236\//i
            },
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /creativecloud/students/make-it-with-creative-cloud.html (top of page video) ",
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/creativecloud/students/make-it-with-creative-cloud.html")
            }
            ],
            selector: "label.hero-leaf-button",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "~0 - DoubleClick - Floodlight Tag - /products/photoshop.html (top of page video)",
            trigger: [{
                command: "loadIframe",
                arguments: [{
                    pages: [{
                        src: "satellite-53e15e38009c09933d00035e.html",
                        data: []
                    }
                    ]
                }
                ]
            }
            ],
            conditions: [function() {
                return ("www.adobe.com" === e.location.hostname || "www.stage.adobe.com" === e.location.hostname)&&-1 !== e.location.pathname.indexOf("/products/photoshop.html")
            }
            ],
            selector: "label.hero-leaf-button",
            event: "click",
            bubbleFireIfParent: !0,
            bubbleFireIfChildFired: !0,
            bubbleStop: !0,
            eventHandlerOnElement: !0
        }, {
            name: "(Rule for AAM integration)",
            event: "ba1811d4769e557169de925c6cbb9b6b.load",
            trigger: [{
                tool: "896d299d710b5a7ec5a0bddeef3248a094c91d33",
                command: "fire",
                arguments: [function() {}
                ]
            }
            ]
        }
        ],
        directCallRules: [],
        settings: {
            trackInternalLinks: !0,
            libraryName: "satelliteLib-6d5cdee78e7fff2a60e0d0464002fbb53b803460",
            isStaging: !1,
            allowGATTcalls: !1,
            downloadExtensions: /\.(?:doc|docx|eps|jpg|png|svg|xls|ppt|pptx|pdf|xlsx|tab|csv|zip|txt|vsd|vxd|xml|js|css|rar|exe|wma|mov|avi|wmv|mp3|wav|m4v)($|\&|\?)/i,
            notifications: !1,
            utilVisible: !1,
            domainList: ["adobe.com"],
            scriptDir: "659ec8ada5450db95675e43beaaae92399591a11/scripts/",
            tagTimeout: 3e3
        },
        data: {
            URI: a.location.pathname + a.location.search,
            browser: {},
            cartItems: [],
            revenue: "",
            host: {
                http: "assets.adobedtm.com",
                https: "assets.adobedtm.com"
            }
        },
        dataElements: {
            adbadobenonacdc_channel_Site_Section: {
                customJS: function() {
                    var t = "", a = e.location.hostname.toLowerCase(), n = e.location.pathname, i = _satellite.getVar("adbadobenonacdc_pageName_Page_Name");
                    return "undefined" != typeof e.s_channel && "Adobe Homepages" == e.s_channel ? t = "Adobe.com Homepage" : - 1 !== n.indexOf("/products/")||-1 !== n.indexOf("/creativecloud/")||-1 !== n.indexOf("/creativecloud.html") ? t = "Products" : _satellite.getVar("isSite_AdobeHelpX") ? t = "Support & Learning" : _satellite.getVar("adbadobenonacdc_server") && "www.adobe.com Static Content pages" == _satellite.getVar("adbadobenonacdc_server")?-1 !== n.indexOf("/products") ? t = "Products" : - 1 !== n.indexOf("/downloads") ? t = "Adobe.com Download" : - 1 !== n.indexOf("/misc/terms.html") ? t = "Adobe.com About Adobe" : - 1 !== n.indexOf("/international/selector") ? t = "Adobe.com International Selector" : - 1 !== n.indexOf("") && (t = "Products") : t =- 1 !== a.indexOf("success.adobe.com") ? "success.adobe.com" : - 1 !== i.indexOf("adobe.com:company")||-1 !== i.indexOf("adobe.com:careers")||-1 !== i.indexOf("adobe.com:investor-relations")||-1 !== i.indexOf("adobe.com:news-room")||-1 !== i.indexOf("adobe.com:corporate-responsibility") ? "company" : - 1 !== i.indexOf("adobe.com:customershowcase") ? "customershowcase" : - 1 !== i.indexOf("adobe.com:privacy") ? "privacy" : a.replace("www.", ""), t
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_eVar17_Campaign_Stacking: {
                queryParam: "chid",
                storeLength: "pageview",
                ignoreCase: 0
            },
            adbadobenonacdc_eVar18_New_Repeat_Visitor: {
                customJS: function() {
                    return _satellite._getNewRepeat(365, "s_nr")
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_eVar22_Time_Parting_Day: {
                customJS: function() {
                    var e = _satellite._getTimeParting("n", "-8");
                    return e[2] + " - " + e[1]
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_eVar28_Page_URL: {
                customJS: function() {
                    var e = location.hostname + ("/" == location.pathname.charAt(0) ? location.pathname : "/" + location.pathname);
                    return e = e.toLowerCase(), - 1 != e.indexOf("?") && (e = e.replace(/\?(.*)/, "")), e
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_eVar36_Affiliate_ID: {
                queryParam: "pid",
                storeLength: "pageview",
                ignoreCase: 0
            },
            adbadobenonacdc_eVar43_Cross_Channel_Targeting: {
                queryParam: "ttsrccat",
                storeLength: "pageview",
                ignoreCase: 0
            },
            adbadobenonacdc_eVar4_OnSite_Campaigns_Dig_Mkt: {
                queryParam: "s_osc",
                storeLength: "pageview",
                ignoreCase: 0
            },
            adbadobenonacdc_eVar5_OnSite_Campagins_Dig_Media_promoid: {
                queryParam: "promoid",
                storeLength: "pageview",
                ignoreCase: 0
            },
            adbadobenonacdc_eVar60_Event_Name: {
                customJS: function() {
                    {
                        var t = "";
                        e.location.hostname.toLowerCase(), e.location.pathname
                    }
                    return "object" == typeof attendeaseEvent && "string" == typeof attendeaseEvent.name && (t = attendeaseEvent.name), t
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_eVar6_External_Campaigns_365_expire: {
                customJS: function() {
                    var e = "";
                    return _satellite.getQueryParam("sdid") ? e = _satellite.getQueryParam("sdid") : _satellite.getQueryParam("trackingid") ? e = _satellite.getQueryParam("trackingid") : _satellite.getQueryParam("s_cid") ? e = _satellite.getQueryParam("s_cid") : _satellite.getQueryParam("s_rtid") && (e = _satellite.getQueryParam("s_rtid")), e = _satellite._getValOnce(e, "s_a_campaign", 0)
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_eVar7_Content_Dig_Mkt: {
                queryParam: "s_iid",
                storeLength: "pageview",
                ignoreCase: 0
            },
            adbadobenonacdc_event35_condition: {
                customJS: function() {
                    var t = e.location.pathname, a=!1, n = "", i = [], o = {
                        africa: 1,
                        at: 1,
                        au: 1,
                        be_en: 1,
                        be_fr: 1,
                        be_nl: 1,
                        bg: 1,
                        br: 1,
                        ca: 1,
                        ca_fr: 1,
                        ch_de: 1,
                        ch_fr: 1,
                        ch_it: 1,
                        cin: 1,
                        cn: 1,
                        cz: 1,
                        de: 1,
                        dk: 1,
                        ee: 1,
                        eeurope: 1,
                        es: 1,
                        fi: 1,
                        fr: 1,
                        hk_en: 1,
                        hk_zh: 1,
                        hr: 1,
                        hu: 1,
                        ie: 1,
                        il_en: 1,
                        il_he: 1,
                        "in": 1,
                        it: 1,
                        jp: 1,
                        kr: 1,
                        la: 1,
                        lt: 1,
                        lu_de: 1,
                        lu_en: 1,
                        lu_fr: 1,
                        lv: 1,
                        mena_ar: 1,
                        mena_en: 1,
                        mena_fr: 1,
                        mx: 1,
                        nl: 1,
                        no: 1,
                        nz: 1,
                        pl: 1,
                        pt: 1,
                        ro: 1,
                        rs: 1,
                        ru: 1,
                        se: 1,
                        sea: 1,
                        si: 1,
                        sk: 1,
                        tr: 1,
                        tw: 1,
                        ua: 1,
                        uk: 1,
                        us: 1
                    };
                    return i = t.split("/"), i.length >= 2 && o[i[1]] && i.splice(1, 1), i.length >= 2 && (n = i[1].toLowerCase(), "products" === n && (a=!0)), a
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_pageName_Page_Name: {
                customJS: function() {
                    var t = 0, n = 0, i = "", o = a.title, s = e.location.pathname, r = e.location.hostname.toLowerCase(), c = "", l = "", d = [], f = "", u = [".aspx", ".php", ".html"], h = {
                        africa: 1,
                        at: 1,
                        au: 1,
                        be_en: 1,
                        be_fr: 1,
                        be_nl: 1,
                        bg: 1,
                        br: 1,
                        ca: 1,
                        ca_fr: 1,
                        ch_de: 1,
                        ch_fr: 1,
                        ch_it: 1,
                        cin: 1,
                        cn: 1,
                        cz: 1,
                        de: 1,
                        dk: 1,
                        ee: 1,
                        eeurope: 1,
                        es: 1,
                        fi: 1,
                        fr: 1,
                        hk_en: 1,
                        hk_zh: 1,
                        hr: 1,
                        hu: 1,
                        ie: 1,
                        il_en: 1,
                        il_he: 1,
                        "in": 1,
                        it: 1,
                        jp: 1,
                        kr: 1,
                        la: 1,
                        lt: 1,
                        lu_de: 1,
                        lu_en: 1,
                        lu_fr: 1,
                        lv: 1,
                        mena_ar: 1,
                        mena_en: 1,
                        mena_fr: 1,
                        mx: 1,
                        nl: 1,
                        no: 1,
                        nz: 1,
                        pl: 1,
                        pt: 1,
                        ro: 1,
                        rs: 1,
                        ru: 1,
                        se: 1,
                        sea: 1,
                        si: 1,
                        sk: 1,
                        tr: 1,
                        tw: 1,
                        ua: 1,
                        uk: 1,
                        us: 1
                    };
                    if ( - 1 !== o.indexOf("page cannot be found"))
                        i = "";
                    else {
                        for (c = s, f = r, f = f.replace("www.", ""), t = 0, n = u.length; n > t; t++)
                            c = c.replace(u[t], "");
                        for (d = c.split("/"), t = d.length - 1; t >= 0; t--)
                            l = d[t], "" === l && d.splice(t, 1);
                        d.length >= 1 && h[d[0]] && d.splice(0, 1), c = d.join(":"), i = f + ("" === c ? "" : ":" + c)
                    }
                    return i
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_prop12_Previous_Page_Value: {
                customJS: function() {
                    return _satellite._getPreviousValue(_satellite.getVar("adbadobenonacdc_pageName_Page_Name"), "gpv")
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_prop13_Percentage_Page_Viewed: {
                customJS: function() {
                    return _satellite._getPercentPageViewed(e.location.hostname + e.location.pathname)[1]
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_prop1_Page_Type_Template_Name: {
                customJS: function() {
                    var e = "";
                    return "object" == typeof SiteCatalyst && "string" == typeof SiteCatalyst.pageType && (e = SiteCatalyst.pageType), e
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_prop3_Site_Subdomain: {
                customJS: function() {
                    return e.location.hostname.toLowerCase()
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_prop4_Language_Locale: {
                customJS: function() {
                    var t = e.location.pathname, a = t.split("/"), n = "en_us", i = {
                        africa: "en_africa",
                        at: "de_at",
                        au: "en_au",
                        be_en: "en_be",
                        be_fr: "fr_be",
                        be_nl: "nl_be",
                        bg: "bg_bg",
                        br: "pt_br",
                        ca: "en_ca",
                        ca_fr: "fr_ca",
                        ch_de: "de_ch",
                        ch_fr: "fr_ch",
                        ch_it: "it_ch",
                        cn: "zh_cn",
                        cz: "cs_cz",
                        de: "de_de",
                        dk: "da_dk",
                        ee: "et_ee",
                        eeurope: "en_eeurope",
                        es: "es_es",
                        fi: "fi_fi",
                        fr: "fr_fr",
                        hk_en: "en_hk",
                        hk_zh: "zh_hk",
                        hr: "hr_hr",
                        hu: "hu_hu",
                        ie: "en_ie",
                        il_en: "en_il",
                        il_he: "he_il",
                        "in": "en_in",
                        it: "it_it",
                        jp: "ja_jp",
                        kr: "ko_kr",
                        la: "es_la",
                        lt: "lt_lt",
                        lu_de: "de_lu",
                        lu_en: "en_lu",
                        lu_fr: "fr_lu",
                        lv: "lv_lv",
                        mena_ar: "ar_mena",
                        mena_en: "en_mena",
                        mena_fr: "fr_mena",
                        mx: "es_mx",
                        nl: "nl_nl",
                        no: "nb_no",
                        nz: "en_nz",
                        pl: "pl_pl",
                        pt: "pt_pt",
                        ro: "ro_ro",
                        rs: "sr_sp",
                        ru: "ru_ru",
                        se: "sv_se",
                        sea: "en_sea",
                        si: "sl_si",
                        sk: "sk_sk",
                        tr: "tr_tr",
                        tw: "zh_tw",
                        ua: "uk_ua",
                        uk: "en_uk",
                        us: "en_us"
                    };
                    return "object" == typeof adobeGlobalNav && "object" == typeof adobeGlobalNav._options && "string" == typeof adobeGlobalNav._options.locale && adobeGlobalNav._options.locale ? n = adobeGlobalNav._options.locale.toLowerCase() : a.length >= 2 && i[a[1]] && (n = i[a[1]]), n
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_prop5_Page_Name_Granular: {
                customJS: function() {
                    var e = _satellite.getVar("adbadobenonacdc_prop4_Language_Locale"), t = _satellite.getVar("adbadobenonacdc_pageName_Page_Name"), a = t;
                    return a = e ? e + ":" + t : "none:" + t
                },
                storeLength: "pageview"
            },
            adbadobenonacdc_server: {
                customJS: function() {
                    var t = "", a = e.location.hostname.toLowerCase();
                    return t =- 1 !== a.indexOf("success.adobe.com") ? "success.adobe.com" : a
                },
                storeLength: "pageview"
            },
            dcm_rti_group: {
                customJS: function() {
                    var e = _satellite._readCookie("dcm_rti_group", !0), t = new Date;
                    return e || (e = Math.random() < .1 ? "control" : "test"), t.setTime(t.getTime() + 63072e6), _satellite._setCookie("dcm_rti_group", e, t, "adobe.com", "/", !0), e
                },
                storeLength: "visitor"
            },
            generic_pageName_Page_Name: {
                customJS: function() {
                    return e.location.host + e.location.pathname.split("/").join(":")
                },
                storeLength: "pageview"
            },
            isCookie_Renga: {
                customJS: function() {
                    var t = e.location.host;
                    return - 1 !== t.indexOf(".adobe.com") && _satellite._readCookie("RengaRMT")&&-1 == t.indexOf("creative.adobe.com") ? "SignedIn" : "NotSignedIn"
                },
                storeLength: "pageview"
            },
            useModule_media: {
                customJS: function() {
                    var t=!1, a = e.location.hostname.toLowerCase();
                    return - 1 !== a.indexOf(".com")&&-1 !== a.indexOf("adobe.com") && ( - 1 !== a.indexOf("tv.adobe.com")||-1 !== a.indexOf("createnow.adobe.com")) && (t=!0), t
                },
                storeLength: "pageview"
            },
            useTool_adbreimaginetest: {
                customJS: function() {
                    return !0
                },
                storeLength: "pageview"
            },
            useTool_audienceManager: {
                customJS: function() {
                    return !0
                },
                storeLength: "pageview"
            }
        },
        appVersion: "4BD",
        buildDate: "2014-11-19 10:02:36 UTC",
        publishDate: "2014-11-19 10:02:26 UTC"
    })
}(window, document);
