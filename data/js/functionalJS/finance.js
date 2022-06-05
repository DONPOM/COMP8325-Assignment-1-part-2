var Perf, globalLeft;
define("jqBehavior", ["jquery", "viewport"], function(n) {
    return function(t, i, r) {
        function u(n) {
            var t = n.length;
            return t > 1 ? function() {
                for (var i = 0; i < t; i++)
                    n[i]()
            } : t ? n[0] : f
        }
        function f() {}
        if (typeof t != "function")
            throw "Behavior constructor must be a function";
        if (i && typeof i != "object")
            throw "Defaults must be an object or null";
        if (r && typeof r != "object")
            throw "Exclude must be an object or null";
        return r = r || {}, function(f, e, o) {
            function c(n) {
                n && (typeof n.setup == "function" && l.push(n.setup), typeof n.teardown == "function" && a.push(n.teardown), typeof n.update == "function" && v.push(n.update))
            }
            var h;
            if (o && typeof o != "object")
                throw "Options must be an object or null";
            var s = n.extend(!0, {}, i, o), l = [], a = [], v = [], y=!0;
            if (r.query) {
                if (typeof f != "string")
                    throw "Selector must be a string";
                c(t(f, s))
            } else 
                h = n(f, e), r.each ? c(t(h, s)) : (y = h.length > 0, h.each(function() {
                    var i = n(this);
                    c(t(i, r.data ? s : n.extend(!0, {}, s, i.data ? i.data() : {})))
                }));
            return y ? {
                setup: u(l),
                teardown: u(a),
                update: u(v)
            } : null
        }
    }
});
define("trackInfo.tokens", {
    spinTimeout: 150,
    browserFilterTable: {
        webkit: "530.0.0",
        mozilla: "1.9.0"
    }
});
define("afire", ["image"], function(n) {
    return function(t) {
        if (t) {
            var i = new n;
            i.onload = i.onerror = function() {
                i.onload = i.onerror = null
            };
            i.src = t.replace(/&amp;/gi, "&")
        }
    }
});
define("track.generic", ["track", "extend", "trackInfo"], function(n, t, i) {
    function r(n) {
        this.defaultOpts = t(!0, {}, u, n);
        this.samplingRate = this.defaultOpts.samplingRate
    }
    var u = {
        base: "",
        samplingRate: 100,
        eventAlias: {
            submit: "click",
            mouseenter: "click",
            mouseleave: "click",
            click_nonnav: "click",
            mouseenter_nav: "click"
        }
    };
    return r.prototype = {
        getEventTrackingUrl: function(t) {
            var u = this.defaultOpts, r, f;
            return (t || (t = (i.event || {}).type), r = u[t], !r && u.eventAlias && (r = u[u.eventAlias[t]]), r) ? (f = u.base + (r.url ? r.url : ""), n.generateUrl(f, u.common, u.commonMap, r.param, r.paramMap)) : ""
        },
        getPageViewTrackingUrl: function() {
            return this.getEventTrackingUrl("impr")
        }
    }, r
});
define("trackInfo", ["trackInfo.tokens", "dom", "pageInstance", "getCookie", "screen", "window", "document", "viewAware", "deviceGroup"], function(n, t, i, r, u, f, e, o, s) {
    function tt(n, t) {
        this.element = n;
        this.visible = t;
        this.visible && (this.compress = function() {
            var n = this.element.getAttribute("data-m"), t = n.replace(/\'/g, '"');
            try {
                return JSON.parse(t)
            } catch (i) {
                return n
            }
        })
    }
    function d() {
        f.innerWidth ? (a = f.innerWidth, v = f.innerHeight) : (a = e.documentElement.clientWidth, v = e.documentElement.clientHeight)
    }
    function g(n) {
        if (n) {
            var i = b(n), r = p(i, "data-aop") || p(i, "data-region"), t;
            return (t = g(i), t && r) ? [t, r].join(h.cmSeparator) : r || t
        }
    }
    function nt(n) {
        if (n) {
            var r = b(n), i = p(r, "data-region"), t;
            return (t = nt(r), t && i) ? [t, i].join(h.cmSeparator) : i || t
        }
    }
    var p = t.attr, b = t.parent, l, k, a, v, c, y = [], w = "", h = {
        notrack: "notrack",
        cmSeparator: ">",
        defaultModule: "body",
        defaultFormHeadline: "[form submit]",
        piitxt: "data-piitxt",
        piiurl: "piiurl",
        wrapperId: "wrapper",
        defaultConnectionType: "LAN",
        smpCookie: "Sample",
        smpExp: 182,
        MUIDCookie: "MUID",
        spinTimeout: n.spinTimeout,
        trackTcm: "tcm",
        trackAop: "aop",
        curAop: "",
        trackRegion: "region",
        curRegion: "",
        event: {},
        sitePage: {},
        userStatic: {},
        oobWaitTime: 150,
        enableOOB: 1,
        bwVerTable: n.browserFilterTable,
        client: {
            clientId: function() {
                return l || l === "" || (l = r(h.MUIDCookie) || h.sitePage.requestId || ""), l
            },
            OTFTelemetry: function(n) {
                require(["binding", "c.deferred"], function() {
                    var t, a, u, o, s, c, l, r, i;
                    for (typeof n == "undefined" ? n = "load" : n == null && (n = ""), y = [], t = e.body.querySelectorAll("[data-m]"), a = f.innerHeight, u = f.innerWidth, o = 0, s = e.getElementById("content"), s && (c = s.getBoundingClientRect(), u = c.right, o = c.left), l = [], r = 0; r < t.length; r++)
                        l.push(h.telemetryTracking.isVisible(t[r], u, o, a));
                    for (i = 0; i < t.length; i++)
                        y.push(new tt(t[i], l[i]));
                    h.telemetryTracking.init(!1, n, h.telemetryTracking.formatModules(), h)
                })
            },
            OTFSpecialSection: function(n) {
                h.telemetryTracking.specialSection(n)
            },
            colorDepth: u.colorDepth,
            connectionType: function() {
                return h.defaultConnectionType
            },
            cookieSupport: function() {
                return e.cookie ? "Y" : "N"
            },
            height: function() {
                return v || d(), v
            },
            pageUrl: function() {
                return f.location.href
            },
            referrer: function() {
                return e.referrerOverride || e.referrer
            },
            screenResolution: function() {
                return [u.width, u.height].join("x")
            },
            width: function() {
                return a || d(), a
            },
            timezone: function() {
                var i = new Date, r = new Date, n, t;
                return r.setMonth(i.getMonth() + 6), n = Math.round(i.getTimezoneOffset() / 60)*-1, t = Math.round(r.getTimezoneOffset() / 60)*-1, n < t ? n : t
            },
            viewType: function() {
                return s.isTmx&&!c && o.listen(function(n) {
                    n & o.views.SIZE1COLUMN ? c = "size1column" : n & o.views.SIZE2COLUMN ? c = "size2column" : n & o.views.SIZE3COLUMN ? c = "size3column" : n & o.views.SIZE4COLUMN && (c = "size4column")
                }), c
            },
            isIE: function() {
                return f.ActiveXObject?!0 : !1
            }
        },
        userDynamic: {
            anid: function() {
                return r("ANON")
            },
            isHomePage: function() {
                var n = e.documentElement, t = 0;
                if (n.addBehavior && (k || n.addBehavior("#default#homePage") && (k = 1)))
                    try {
                        t = n.isHomePage(f.location.href) ? "Y" : "N"
                } catch (i) {}
                return t
            },
            timeStamp: function() {
                return + new Date
            },
            AOP: function() {
                return h.curAop != null && h.curAop == "" && (h.curAop = g(t.getTarget(h.event)) || ""), h.curAop
            },
            Region: function() {
                return h.curRegion != null && h.curRegion == "" && (h.curRegion = nt(t.getTarget(h.event)) || ""), h.curRegion
            },
            slideType: function() {
                for (var n = h.event.target, i = "halfpane"; n && t.name(n) != "SECTION";) {
                    if (t.name(n) == "LI" && t.containsClass(n, i))
                        return i;
                    n = t.parent(n)
                }
                return null
            },
            eventNumber: 0
        },
        telemetryTracking: {
            decorateTelemetryPayload: function(n, t) {
                var r = h.sitePage, u;
                return (w = r.otfURL, !w) ? null : (u = {
                    evt: n,
                    rid: i.getActivityId() || r.requestId,
                    di: r.domainId,
                    clid: h.client.clientId()
                }, t && (u.data = t), u)
            },
            setupParameters: function(n) {
                return this.decorateTelemetryPayload(n)
            },
            init: function(n, t, i) {
                var r = this.setupParameters("impr_update");
                r && (r.mech = t, r.e1 = i, this.sendUpdate(r))
            },
            isVisible: function(n, t, i, r) {
                if (!n)
                    return !1;
                var f=!1, e=!1, o=!1, s=!1, u = n.getBoundingClientRect();
                return f = u.top >= 0 && u.top < r, e = u.left >= i && u.left < t, o = u.right > i && u.right <= t, s = u.height !== 0 || u.width !== 0, s && f && (e || o)
            },
            formatModules: function() {
                var t = [], i, n;
                for (i in y)
                    n = y[i], n && n.compress && t.push(n.compress());
                return {
                    e: t
                }
            },
            specialSection: function(n) {
                var t = this.setupParameters("section");
                t && (t.sn = n, h.telemetryTracking.sendUpdate(t))
            },
            sendUpdate: function(n) {
                var i = JSON.stringify([n]), t;
                t = new XMLHttpRequest;
                t.open("POST", w, !0);
                t.setRequestHeader("Content-type", "application/json; charset=utf-8");
                t.onreadystatechange = function() {};
                t.send(i)
            },
            postTelemetryPayload: function(n, t) {
                var i = h.telemetryTracking.decorateTelemetryPayload(n, t);
                return h.telemetryTracking.sendUpdate(i)
            }
        }
    };
    return h
});
define("c.track.mobi", ["track", "trackInfo", "pageInstance", "screen", "navigator", "dom", "events", "document"], function(n, t, i, r, u, f, e, o) {
    function y(n) {
        var i = l.exec(n), t = i[i.length - 1].split("."), r = t.length;
        return r > 1 && (t = t.slice(r - 2)), t.join(".")
    }
    function p(t, i, r, u, f, e, o, s) {
        if (t) {
            t[h] = i;
            r = r && r.length > 0 ? r : a;
            var c = n.createEvent(null, r, t);
            n.trackEvent(c, t, u, f, e, o, s)
        }
    }
    var a = "click_nonnav", h = "touch_gesture", s=!1, c = 0, v = 1e3, l;
    return n.extend({
        client: {
            scrW: function() {
                return r.width
            },
            scrH: function() {
                return r.height
            },
            orientation: function() {
                return r.width > r.height ? "landscape" : "portrait"
            },
            userAgent: function() {
                return u.userAgent
            },
            hourOfDay: function() {
                return (new Date).getUTCHours()
            },
            linkDomain: function() {
                var n = t.report ? t.report.destinationUrl: null;
                return n && n.length ? y(n) : null
            },
            pageTitle: function() {
                return o.title
            },
            gesture: function() {
                var n = f.getTarget(t.event);
                return n && n[h] ? n[h] : t.event && (t.event.type == "click" || t.event.type == "click_nonnav" || t.event.type == "submit") && s==!0 && t.sitePage.device ? "tap" : ""
            }
        },
        userDynamic: {
            requestId: function() {
                var n = i.getActivityId() || t.sitePage.requestId;
                return n ? n : window.console && console.error("No requestID passed in from the server or generated via guidGen.js, please check the trackInfo.sitePage.requestId or pageInstance.js")
            }
        }
    }), l = /^(\w+:\/\/)?([^:\/]*)/, u.pointerEnabled || u.msPointerEnabled ? e.bind(o, u.pointerEnabled ? "pointerup" : "MSPointerUp", function(n) {
        s = n.pointerType == n.MSPOINTER_TYPE_TOUCH || n.pointerType == "touch"?!0 : !1
    }) : (e.bind(o, "mouseup", function() {
        var n = (new Date).getTime();
        n > c + v && (s=!1)
    }), e.bind(o, "touchend", function() {
        s=!0;
        c = (new Date).getTime()
    })), n.trackGesture = p, 1
});
define("track", ["trackInfo", "extend", "dom", "getCookie", "setCookie", "events", "window", "document", "mediator", "navigation", "headData", "pageInstance", "afire"], function(n, t, i, r, u, f, e, o, s, h, c, l, a) {
    function ot(n) {
        var t, i = 0;
        for (st(), it.incrementEventNumber();
        i < nt.length;
        i++)t = nt[i], t.defaultOpts.isGeneratedEarly ? t.defaultOpts.isGeneratedEarly=!1 : t && t.samplingRate >= y && a(t[n]());
        p.curAop = "";
        p.curRegion = ""
    }
    function st() {
        if ( - 1 == y) {
            var n = p.smpCookie;
            y = parseInt(r(n));
            y = isNaN(y) ? Math.floor(Math.random() * 100) : y%100;
            u(n, y, 182, u.topDomain, "/")
        }
        return y
    }
    function tt(n, t, i) {
        var u = g(n) || [], r = 0, f;
        for (t = t || "alt"; r < u.length; r++)
            if (f = v(u[r], t) || tt(u[r], t, i), f&&!(i == u[r].localName))
                return f
    }
    function ht(n) {
        if (n) {
            var r = d(n), t = v(r, "id"), i;
            return p.wrapperId == t ? void 0 : (i = ht(r), i && t) ? [i, t].join(p.cmSeparator) : t || i
        }
    }
    function ct(n) {
        if (n) {
            var u = d(n), i, t = 0, r = 0;
            if (!v(u, "id"))
                if (t = ct(u), t)
                    t--;
                else 
                    return 0;
            for (i = g(u) || []; r < i.length; r++) {
                if (i[r] == n) {
                    t++;
                    break
                }
                t += lt(i[r])
            }
            return t
        }
    }
    function lt(n) {
        var t = 0, i = 0, r;
        if (n&&!v(n, "id"))
            for (r = g(n) || [], v(n, "href")&&!v(n, p.notrack) && t++; i < r.length; i++)
                t += lt(r[i]);
        return t
    }
    function at(n, t, i) {
        var u, r, e, f;
        if (i && n && t) {
            if (n.tagName == t)
                return n;
            for (u = g(n) || [], r = 0, e = u.length; r < e; r++) {
                if (u[r].tagName == t)
                    return u[r];
                if (f = at(u[r], t, i - 1), f)
                    return f
            }
        }
        return null
    }
    function w(t, r, u, f, e, o, s) {
        var c, l, p, h, a, y;
        (!r && t && (r = t.target), r&&!v(r, n.notrack)) && (n.event = t, r.jquery && (r = r[0]), c = r.href || v(r, "href"), c == "#" && (c = r.href), u = u || v(r, n.piiurl) || c || v(r, "action") || "", f || (l = at(r, "H4", 2), l && (f = i.text(l))), f = f || v(r, n.piitxt) || ("FORM" == rt(r) ? n.defaultFormHeadline : tt(r, "title", "img") || i.text(r) || v(r, "alt") || tt(r, "alt") || ""), f.trim != undefined && (f = f.trim()), e = e || ht(r) || n.defaultModule, o = o || (v(r, "id") ? 1 : ct(r)), p = r.className || v(r, "class"), s = s || (/GT1-(\d+)\b/i.exec(p) ? RegExp.$1 : "") || (/[?&]GT1=(\d+)\b/i.exec(c) ? RegExp.$1 : ""), h = "", t && (t.type === "click" || t.type === k || t.type === "submit") && (a = "data-m", h = v(r, a), h || (y = d(r), h = y ? v(y, a) : ""), h && (h = h.replace(/\'/g, '"'))), n.report = {
            destinationUrl: u,
            headline: f,
            contentModule: e,
            contentElement: o,
            campaignId: s,
            module: h,
            sourceIndex: r.sourceIndex || "",
            nodeName: r.nodeName || ""
        }, ot("getEventTrackingUrl", t?!t.noSpin : 1))
    }
    function wt(n) {
        var i = null, t = new RegExp(pt).exec(n);
        return t && t.length >= 1 && t[1] && (i = t[1]), i
    }
    function bt(n) {
        var i = null, r, t;
        return n && (n.indexOf(et)==-1 ? (r = n.substring(0, n.indexOf("#")).toLowerCase(), t = e.location.href.toLowerCase(), (t == r || t.substring(0, t.indexOf("#")) == r) && (i = k)) : i = wt(n)), i
    }
    function vt(n, t, i) {
        var r = null;
        return b.createEvent ? (r = b.createEvent("Events"), r.initEvent(t, !1, !0, i || e, 0, 0, 0, 0, 0, !1, !1, !1, !1, 0, null)) : b.createEventObject && (r = b.createEventObject(n), r.type = t), r && (r.customTarget = i), r
    }
    function kt(n) {
        n.preventDefault ? n.preventDefault() : n.returnValue=!1
    }
    function yt(n) {
        var t, u, r, f;
        if (2 != n.button) {
            try {
                n.customTarget && n.customTarget.useMap && (n.customTarget = n.customTarget.document.activeElement, n.target = n.customTarget)
            } catch (e) {}
            for (t = i.getTarget(n); t&&!v(t, "href") && t.nodeName != "BUTTON";)
                t = d(t);
            t&&!t.form && (u = n, r = t.href, r && r.length ? (n.type == "click" && (r.indexOf("#") == r.length - 1 || r.indexOf(et) >= 0) && (f = bt(r), f && (u = vt(n, k, t))), u.type == k && kt(n), w(u, t)) : w({
                type: k,
                target: t
            }))
        }
    }
    var v = i.attr, rt = i.name, d = i.parent, g = i.children, nt = [], y =- 1, ut, ft = e, b = o, et = "tevt=", k = "click_nonnav", pt = /#tevt=([A-Za-z0-9]+_[A-Za-z0-9]+)(;*)/g, p, it = {
        onClick: yt,
        trackEvent: w,
        createEvent: vt,
        trackPage: function() {
            delete n.event;
            delete n.userStatic.requestId;
            c && c.clientSettings && c.clientSettings.static_page && (n.sitePage.requestId = l.getActivityId());
            ot("getPageViewTrackingUrl");
            ut ? s.pub("pageView") : ut = 1
        },
        register: function() {
            for (var u = 0, n, f = h.getParamsFromUrl(e.location.href), r, t, i; n = arguments[u++];) {
                if (n.defaultOpts.disableOnAutoRefresh) {
                    for (r = n.defaultOpts.disableOnAutoRefresh.split(";"), t=!1, i = 0; i < r.length; i++)
                        t = t || r[i] == n.defaultOpts.id;
                    if (t && f.AR)
                        continue
                }
                isNaN(n.samplingRate) && (n.samplingRate = 99);
                nt.push(n)
            }
        },
        incrementEventNumber: function() {
            p.userDynamic.eventNumber++
        },
        isSampled: function(n) {
            return !(st() > n)
        },
        generateUrl: function(i, r, u, f, e) {
            var h, c, o, s, l = [];
            f = t({}, r, f);
            e = t(!0, {}, u, e);
            for (h in e)
                if (n[h]) {
                    c = e[h];
                    for (o in c)
                        s = n[h][c[o]], typeof s == "function" && (s = s()), s != null && (f[o] = s)
                }
            for (o in f)
                l.push(encodeURIComponent(o) + "=" + encodeURIComponent(f[o]));
            return i + l.join("&").replace(/%20/g, "+")
        },
        extend: function(i) {
            t(!0, n, i)
        },
        form: function(n) {
            n && n.length || (n = [n]);
            for (var t, i = 0; t = n[i++];)
                "FORM" == rt(t) && f.bind(t, "submit", w)
        }
    };
    return f.bind(b, "click", yt), f.bind(ft, "load", w), f.bind(ft, "unload", w), require(["c.dom"], function() {
        it.form(i.getElementsByTagName("form"))
    }), s.sub("ViewabilityUpdatedEvent", n.client.OTFTelemetry), p = n, it
});
define("binding", ["viewAware", "mediator", "window"], function(n, t, i) {
    function e(t, r, u) {
        this.behaviorName = t;
        this.selector = r;
        this.context = u;
        this.views = n.views.NONE;
        this.timeout = i.setTimeout(function() {
            throw 'binding("' + t + '", "' + r + '") was not followed with an all(), view(), or fallback() call!';
        }, 0)
    }
    function o(t) {
        this.view(n.views.ALL&~this.views, t)
    }
    function s(n, t) {
        for (var u = null, i = 0; i < r.length; i++)
            if (r[i].behaviorName == n && r[i].args[0] == t) {
                u = r[i];
                break
            }
        return u
    }
    function c(n) {
        var t, f, i;
        for (u = n, i = 0; i < r.length; i++)
            t = r[i], f = t.view & u, t.active && (f ? t.update() : (t.teardown(), t.active = 0));
        for (i = 0; i < r.length;) {
            if (t = r[i], f = t.view & u, f&&!t.active&&!h(t)) {
                r.splice(i, 1);
                continue
            }
            ++i
        }
    }
    function h(n) {
        if (!n.setup) {
            t.pub("bind", {
                action: n.behaviorName,
                view: n.view,
                selector: n.args[0]
            });
            var i = n.behavior.apply(null, n.args);
            i && (n.setup = i.setup, n.teardown = i.teardown, n.update = i.update)
        }
        return n.setup ? (n.setup(), n.active = 1, !0) : !1
    }
    var r = [], u, f;
    return n.listen(c), e.prototype = {
        view: function(n, t) {
            var e = this.behaviorName, o = [this.selector, this.context, t], f = {
                active: 0,
                args: o,
                behaviorName: this.behaviorName,
                view: n
            };
            return this.views|=n, i.clearTimeout(this.timeout), require([e], function(t) {
                f.behavior = t;
                var i=!0;
                n & u && (i = h(f));
                i && r.push(f)
            }), this
        },
        fallback: o,
        all: o,
        get: s
    }, f = function(n, t, i) {
        return new e(n, t, i)
    }, f.views = n.views, i.binding = f, i.binding.get = s, f
});
define("bindingAuditor", ["mediator", "window"], function(n, t) {
    function e(n) {
        for (var t = n.length, i = 0, u = [2, 3, 5, 7], r = 0; r < n.length; r++)
            t = (t + n.charCodeAt(r) * u[i])%16777215, i = (i + 1)%u.length;
        return t.toString(16)
    }
    function f(n, t) {
        return n === t ? 0 : n > t ? 1 : - 1
    }
    var i = [], r, u;
    return n.sub("bind", function(n) {
        i.push(n)
    }), r = t.location.search.match(/[?&]bind=([^&#]*)/i), r && require(["c.deferred"], function() {
        setTimeout(function() {
            if (r[1] == "report")
                t.console && console.log(u.getReport());
            else if (r[1] == "save")
                t.localStorage.setItem("bindingAudit", u.getHash());
            else if (r[1] == "check") {
                var n = u.getHash();
                alert((t.localStorage.getItem("bindingAudit") == n ? "Same bindings" : "bindings changed") + "\n" + n)
            }
        }, 100)
    }), u = {
        getReport: function() {
            for (var t = ["Binding Audit", u.getHash()], n = 0; n < i.length; n++)
                t.push(i[n].action + "(" + i[n].view + "): " + i[n].selector);
            return t.join("\n")
        },
        getHash: function() {
            var t = "", n;
            for (i.sort(function(n, t) {
                return f(n.action, t.action) || f(n.selector, t.selector)
            }), n = 0; n < i.length; n++)
                t += i[n].action + i[n].selector;
            return e(t)
        }
    }
});
define("getCookie", function() {
    return function(n) {
        var i = new RegExp("\\b" + n + "\\s*=\\s*([^;]*)", "i"), t = i.exec(document.cookie);
        return t && t.length > 1 ? t[1] : ""
    }
});
define("setCookie", ["location"], function(n) {
    function t(n, t, i, r, u, f) {
        var o, e = [n, "=", t], s;
        - 1 == i ? o = "Fri, 31 Dec 1999 23:59:59 GMT" : i && (s = new Date, s.setTime(s.getTime() + i * 864e5), o = s.toUTCString());
        o && e.push(";expires=", o);
        r && e.push(";domain=", r);
        u && e.push(";path=", u);
        f && e.push(";secure");
        document.cookie = e.join("")
    }
    return t.topDomain = (n.hostname.match(/[^.]+\.[^.\d]+$/) || {})[0] || "", t
});
define("dom", function() {
    var n = document, t = /[\n\t]/g, i = /(^\s+)|(\s+$)/mg;
    return {
        attr: function(n, t) {
            return n && (n.getAttribute ? n.getAttribute(t, 2) : n[t]) || ""
        },
        name: function(n) {
            return n && n.nodeName || ""
        },
        text: function(n) {
            return (n && (n.textContent || n.innerText) || "").replace(i, "")
        },
        children: function(n) {
            return n && n.children || []
        },
        parent: function(n) {
            return n && n.parentNode
        },
        getElementsByTagName: function(t) {
            return n.getElementsByTagName(t)
        },
        create: function(t) {
            return n.createElement(t)
        },
        containsClass: function(n, i) {
            return n && (" " + (n.className || n.getAttribute("class")) + " ").replace(t, " ").indexOf(" " + i + " ")>-1
        },
        getTarget: function(n) {
            return n && (n.customTarget || n.target || n.srcElement) || document
        }
    }
});
define("events", function() {
    function n(n, t, i) {
        n.addEventListener && n.addEventListener(t, i, !1)
    }
    function t(n, t, i) {
        n.removeEventListener && n.removeEventListener(t, i, !1)
    }
    return {
        bind: n,
        unbind: t
    }
});
define("extend", function() {
    function n() {
        var f = arguments, t = f[0] || {}, e = 1, i, r, o, u;
        for ((typeof t == "boolean" || typeof t == "number") && (o=!!t, t = f[1], e = 2); e < f.length; e++) {
            r = f[e];
            for (i in r)
                r[i] !== undefined && (o && typeof r[i] == "object" ? (u = t[i], typeof u != "object" && (u = {}), n(!0, u, r[i]), t[i] = u) : t[i] = r[i])
        }
        return t
    }
    return n
});
define("format", function() {
    function t(t, i, r) {
        for (var u = 0; u < r.length - i; ++u)
            t = t.replace(n[u] || (n[u] = new RegExp("\\{" + u + "\\}", "g")), r[u + i]);
        return r.length ? t : t.toString()
    }
    var n = [];
    return String.prototype.format = function() {
        return t(this, 0, arguments)
    }, function(n) {
        return (typeof n == "function" && (n = n.apply(n, Array.prototype.slice.call(arguments, 1))), n == null) ? "" : (typeof n != "string" && (n = n.toString()), n) ? t(n, 1, arguments) : ""
    }
});
define("refreshDeferredModules", ["jquery", "refreshModules"], function(n, t) {
    function i() {
        return t(n("[data-module-deferred]:not([data-sso-dependent])"))
    }
    return i
});
define("navigation", ["location"], function(n) {
    function i(n, t, i) {
        var s = function(n) {
            return n = n.replace(/\+/g, " "), decodeURIComponent(n)
        }, u = {}, o, e;
        if (n)
            for (n = n.split("#")[0], o = n.split("&"), e = 0; e < o.length; e++) {
                var h = o[e].split("="), r = h[0], f = h[1];
                i && (r = s(r), f && (f = s(f)));
                t ? (u[r] || (u[r] = []), u[r].push(f)) : u[r] = f
            }
        return u
    }
    function r(n) {
        var t = u.exec(n);
        return t ? t[2] : !1
    }
    var u = /[a-z][a-z0-9+\-.]*:\/\/([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\])/i, t = {
        getUrl: function(n) {
            return t.filter ? t.filter(n) : n
        },
        navigate: function(i, r) {
            t.filter && (i = t.filter(i));
            r ? n.replace(i) : n.href = i
        },
        getHostName: r,
        isLocal: function(t) {
            var i = r(t);
            return !i || n.hostname == i
        },
        getParams: i,
        getParamsFromUrl: function(n, t, r) {
            var u = n.split("?")[1];
            return i(u, t, r)
        },
        mergeQueryStringParams: function(n, t) {
            var f, u, e, r;
            if (t) {
                if (f = n.split("?"), f[1]) {
                    u = i(f[1], !1, !0);
                    for (r in t)
                        u[r] = t[r]
                } else 
                    u = t;
                n = f[0];
                e = "?";
                for (r in u)
                    n += u[r] ? e + encodeURIComponent(r) + "=" + encodeURIComponent(u[r]) : e + encodeURIComponent(r), e = "&"
            }
            return n
        },
        filter: null
    };
    return t
});
define("refreshPdpModules", ["jquery", "refreshModules"], function(n, t) {
    function i() {
        var i = n("[data-pdp-dependent]"), r = t(i, {
            pfr: 1
        });
        r.done(function() {
            define("c.pdpready", 1)
        })
    }
    return i
});
define("refreshSigninModule", ["jquery", "refreshModules", "binding"], function(n, t, i) {
    function r() {
        var r = n("[data-sso-dependent]");
        t(r).done(function() {
            i.get("flyout", "#sign-in").teardown();
            i("flyout", "#username").all({
                eventOrigin: "#username>button,#username>figure"
            });
            i.get("marketDropdown", "#site-market").teardown();
            i("marketDropdown", "#site-market").all();
            i("marketDropdown", ".marketswitch").all({
                triggerEvent: "click"
            });
            i("marketDropdown", ".langtoggle").all({
                triggerEvent: "click"
            })
        })
    }
    return r
});
define("pointerEvents", ["navigator"], function() {
    return window.PointerEvent ? {
        enabled: !0,
        down: "pointerdown",
        up: "pointerup",
        cancel: "pointercancel",
        move: "pointermove",
        over: "pointerover",
        out: "pointerout",
        enter: "pointerenter",
        leave: "pointerleave"
    } : window.MSPointerEvent ? {
        enabled: !0,
        down: "MSPointerDown",
        up: "MSPointerUp",
        cancel: "MSPointerCancel",
        move: "MSPointerMove",
        over: "MSPointerOver",
        out: "MSPointerOut",
        enter: "MSPointerOver",
        leave: "MSPointerOut"
    } : {
        enabled: !1,
        down: "mousedown",
        up: "mouseup",
        cancel: "mouseup",
        move: "mousemove",
        over: "mouseover",
        out: "mouseout",
        enter: "mouseenter",
        leave: "mouseleave"
    }
});
define("promise", ["window"], function(n) {
    function t(n, e) {
        function y(n) {
            s == i && (o = n, s = u, v())
        }
        function a(n) {
            s == i && (o = n, s = f, v())
        }
        function p(n) {
            for (var t = 0; t < h.length; t++)
                h[t](n)
        }
        function v() {
            for (var n = 0; n < c.length; n++)
                l(c[n])
        }
        function l(n) {
            switch (s) {
            case i:
                c.push(n);
                break;
            case u:
                n.complete();
                break;
            case f:
                n.error()
            }
        }
        if (typeof n != "function")
            throw "promise must take an init function as the first parameter";
        var s = i, o, c = [], h = [];
        this.addEventListener = function() {
            throw "not implemented";
        };
        this.cancel = function() {
            s == i && (e && (e(), e = null), a(new Error("Cancel")))
        };
        this.dispatchEvent = function() {
            throw "not implemented";
        };
        this.done = function(n, i, r) {
            function u(n) {
                if (t.onerror)
                    t.onerror(n);
                else 
                    throw n;
            }
            l({
                complete: function() {
                    if (n)
                        try {
                            n(o)
                    } catch (t) {
                        u(t)
                    }
                },
                error: function() {
                    if (i)
                        try {
                            i(o)
                    } catch (n) {
                        u(n)
                    } else 
                        u(o)
                }
            });
            r && h.push(r)
        };
        this.removeEventListener = function() {
            throw "not implemented";
        };
        this.then = function(n, i, u) {
            var f, e, s, c = new t(function(n, t, i) {
                f = n;
                e = t;
                s = i
            });
            return l({
                complete: function() {
                    var i, u = 1;
                    if (n)
                        try {
                            i = n(o);
                            typeof i == "undefined" && (i = o)
                    } catch (s) {
                        u = 0;
                        e(r(s))
                    } else 
                        i = o;
                    u && (t.is(i) ? i.then(f, e) : f(i))
                },
                error: function() {
                    var n, u = 0;
                    if (i)
                        try {
                            n = i(o);
                            typeof n == "undefined" ? n = o : u = 1
                    } catch (s) {
                        n = r(s)
                    } else 
                        n = o;
                    u ? t.is(n) ? n.then(f, e) : f(n) : e(n)
                }
            }), u && h.push(u), c
        };
        n(y, a, p)
    }
    function r(n) {
        return n && n.name == e || (n = new Error(n)), n
    }
    var i = 1, u = 2, f = 3, e = "Error";
    return t.any = function(n) {
        function u() {
            r()
        }
        for (var r, f = new t(function(n) {
            r = n
        }), i = 0; i < n.length; i++)
            n[i].then(u, u);
        return f
    }, t.as = function(n) {
        if (n) {
            if (n.name == e)
                return t.wrapError(n);
            if (t.is(n))
                return n
        }
        return t.wrap(n)
    }, t.is = function(n) {
        return n && typeof n.then == "function"
    }, t.join = function(n) {
        function u() {
            --f || r()
        }
        for (var f = n.length, r, e = new t(function(n) {
            r = n
        }), i = 0; i < n.length; i++)
            n[i].then(u, u);
        return e
    }, t.theneach = function() {
        throw "not implemented";
    }, t.timeout = function(i, r) {
        function f() {
            u = 1
        }
        var u;
        return r && r.done(f, f), new t(function(t) {
            function f() {
                u || (r && r.cancel(), t())
            }
            i === 0 && n.setImmediate ? n.setImmediate(f) : n.setTimeout(f, i)
        })
    }, t.wrap = function(n) {
        return new t(function(t) {
            t(n)
        })
    }, t.wrapError = function(n) {
        return new t(function(t, i) {
            i(r(n))
        })
    }, t
});
define("scrollLeft", ["dir.tokens", "device"], function(n, t) {
    var i, r;
    if (!n.ltr) {
        if (r = t.capability("RtlScrollLeftAdjustment"), r == "fromLeft")
            return i = function(n, t) {
                return t ? n.scrollLeft = n.scrollWidth - n.clientWidth - t : n.scrollWidth - n.clientWidth - n.scrollLeft
            }, i.adjustValue = function(n, t) {
                return n.scrollWidth - n.clientWidth - t
            }, i;
        if (r == "negativeValue")
            return i = function(n, t) {
                return t ? n.scrollLeft = t*-1 : Math.abs(n.scrollLeft)
            }, i.adjustValue = function(n, t) {
                return t*-1
            }, i
    }
    return i = function(n, t) {
        return t ? n.scrollLeft = t : n.scrollLeft
    }, i.adjustValue = function(n, t) {
        return t
    }, i
});
define("sso", ["headData"], function(n) {
    return function(t, i) {
        var f = n.sso, r, l;
        if (t = t || function() {}, i = i || window, !f || n.clientSettings && n.clientSettings.static_page) {
            t({
                signedIn: !1,
                closed: !1
            });
            return 
        }
        f = f.replace(/amp;/g, "&");
        r = i.document.createElement("IFRAME");
        r.id = "sso_frame";
        r.style.width = "1px";
        r.style.height = "1px";
        r.style.position = "absolute";
        r.style.visibility = "hidden";
        r.style.left = "0px";
        r.style.bottom = "0px";
        r.src = f;
        i.document.body.appendChild(r);
        var e = null, c = 0, o, u, s, h = function(n) {
            (n.data === "sso_completed" || n.data === "sso_failed") && (e = n.data)
        };
        i.addEventListener ? (u = "message", s = i.addEventListener(u, h, !1), o = i.removeEventListener) : (u = "onmessage", i.attachEvent(u, h), s = h, o = i.detachEvent);
        l = i.setInterval(function() {
            if (e || c >= 5) {
                var f = {
                    signedIn: e === "sso_completed",
                    closed: !!e
                };
                i.clearInterval(l);
                o.call(i, u, s);
                i.document.body.removeChild(r);
                t(f);
                f.signedIn && define("c.sso", n.signedin = 1)
            } else 
                c++
        }, 1e3)
    }
});
require(["logging", "document", "c.dom"], function(n, t) {
    function r(i) {
        var r = t.createElement("iframe");
        r.setAttribute("style", "left:-1000px;top:-1000px;position:absolute;");
        typeof r.onload != "undefined" ? (r.onload = function() {
            i(r)
        }, r = t.body.appendChild(r)) : typeof r.onreadystatechange != "undefined" ? (r.onreadystatechange = function() {
            r.readyState === "complete" && i(r)
        }, r = t.body.appendChild(r)) : n.error("Couldn't find either load or onreadystatechange to check when modernizr iframe is ready.")
    }
    function u(n, t, i) {
        function st(n) {
            g.cssText = n
        }
        function h(n, t) {
            return typeof n === t
        }
        function ht(n, t) {
            return !!~("" + n).indexOf(t)
        }
        function et(n, t) {
            var u, r;
            for (u in n)
                if (r = n[u], !ht(r, "-") && g[r] !== i)
                    return t == "pfx" ? r : !0;
            return !1
        }
        function ct(n, t, r) {
            var f, u;
            for (f in n)
                if (u = t[n[f]], u !== i)
                    return r===!1 ? n[f] : h(u, "function") ? u.bind(r || t) : u;
            return !1
        }
        function o(n, t, i) {
            var r = n.charAt(0).toUpperCase() + n.slice(1), u = (n + " " + rt.join(r + " ") + r).split(" ");
            return h(t, "string") || h(t, "undefined") ? et(u, t) : (u = (n + " " + ut.join(r + " ") + r).split(" "), ct(u, t, i))
        }
        function lt() {
            r.input = function(i) {
                for (var r = 0, f = i.length; r < f; r++)
                    c[i[r]]=!!(i[r]in u);
                return c.list && (c.list=!!(t.createElement("datalist") && n.HTMLDataListElement)), c
            }("autocomplete autofocus list placeholder max min multiple pattern required step".split(" "));
            r.inputtypes = function(n) {
                for (var f = 0, r, o, s, h = n.length; f < h; f++)
                    u.setAttribute("type", o = n[f]), r = u.type !== "text", r && (u.value = nt, u.style.cssText = "position:absolute;visibility:hidden;", /^range$/.test(o) && u.style.WebkitAppearance !== i ? (e.appendChild(u), s = t.defaultView, r = s.getComputedStyle && s.getComputedStyle(u, null).WebkitAppearance !== "textfield" && u.offsetHeight !== 0, e.removeChild(u)) : /^(search|tel)$/.test(o) || (r = /^(url|email)$/.test(o) ? u.checkValidity && u.checkValidity()===!1 : u.value != nt)), ft[n[f]]=!!r;
                return ft
            }("search tel url email datetime date month week time datetime-local number range color".split(" "))
        }
        var r = {}, y=!0, e = t.documentElement, s = "modernizr", d = t.createElement(s), g = d.style, u = t.createElement("input"), nt = ":)", tt = " -webkit- -moz- -o- -ms- ".split(" "), it = "Webkit Moz O ms", rt = it.split(" "), ut = it.toLowerCase().split(" "), f = {}, ft = {}, c = {}, p = [], w = p.slice, l, a = function(n, i, r, u) {
            var l, a, c, v, f = t.createElement("div"), h = t.body, o = h || t.createElement("body");
            if (parseInt(r, 10))
                while (r--)
                    c = t.createElement("div"), c.id = u ? u[r] : s + (r + 1), f.appendChild(c);
            return l = ["&#173;", '<style id="s', s, '">', n, "<\/style>"].join(""), f.id = s, (h ? f : o).innerHTML += l, o.appendChild(f), h || (o.style.background = "", o.style.overflow = "hidden", v = e.style.overflow, e.style.overflow = "hidden", e.appendChild(o)), a = i(f, n), h ? f.parentNode.removeChild(f) : (o.parentNode.removeChild(o), e.style.overflow = v), !!a
        }, ot = function(t) {
            var i = n.matchMedia || n.msMatchMedia, r;
            return i ? i(t).matches : (a("@media " + t + " { #" + s + " { position: absolute; } }", function(t) {
                r = (n.getComputedStyle ? getComputedStyle(t, null) : t.currentStyle).position == "absolute"
            }), r)
        }, b = {}.hasOwnProperty, k, v;
        k = h(b, "undefined") || h(b.call, "undefined") ? function(n, t) {
            return t in n && h(n.constructor.prototype[t], "undefined")
        } : function(n, t) {
            return b.call(n, t)
        };
        Function.prototype.bind || (Function.prototype.bind = function(n) {
            var t = this, i, r;
            if (typeof t != "function")
                throw new TypeError;
            return i = w.call(arguments, 1), r = function() {
                var f, e, u;
                return this instanceof r ? (f = function() {}, f.prototype = t.prototype, e = new f, u = t.apply(e, i.concat(w.call(arguments))), Object(u) === u) ? u : e : t.apply(n, i.concat(w.call(arguments)))
            }, r
        });
        f.flexbox = function() {
            return o("flexWrap")
        };
        f.canvas = function() {
            var n = t.createElement("canvas");
            return !!(n.getContext && n.getContext("2d"))
        };
        f.touch = function() {
            var i;
            return "ontouchstart"in n || n.DocumentTouch && t instanceof DocumentTouch ? i=!0 : a(["@media (", tt.join("touch-enabled),("), s, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(n) {
                i = n.offsetTop === 9
            }), i
        };
        f.geolocation = function() {
            return "geolocation"in navigator
        };
        f.history = function() {
            return !!(n.history && history.pushState)
        };
        f.cssanimations = function() {
            return o("animationName")
        };
        f.csstransforms = function() {
            return !!o("transform")
        };
        f.csstransforms3d = function() {
            var n=!!o("perspective");
            return n && "webkitPerspective"in e.style && a("@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}", function(t) {
                n = t.offsetLeft === 9 && t.offsetHeight === 3
            }), n
        };
        f.csstransitions = function() {
            return o("transition")
        };
        f.video = function() {
            var i = t.createElement("video"), n=!1;
            try {
                n=!!i.canPlayType;
                n && (n = new Boolean(n), n.ogg = i.canPlayType('video/ogg; codecs="theora"').replace(/^no$/, ""), n.h264 = i.canPlayType('video/mp4; codecs="avc1.42E01E"').replace(/^no$/, ""), n.webm = i.canPlayType('video/webm; codecs="vp8, vorbis"').replace(/^no$/, ""))
            } catch (r) {}
            return n
        };
        for (v in f)
            k(f, v) && (l = v.toLowerCase(), r[l] = f[v](), p.push((r[l] ? "" : "no-") + l));
        return r.input || lt(), r.addTest = function(n, t) {
            if (typeof n == "object")
                for (var u in n)
                    k(n, u) && r.addTest(u, n[u]);
            else {
                if (n = n.toLowerCase(), r[n] !== i)
                    return r;
                t = typeof t == "function" ? t() : t;
                typeof y != "undefined" && y && (e.className += " " + (t ? "" : "no-") + n);
                r[n] = t
            }
            return r
        }, st(""), d = u = null, r._version = "2.7.1", r._prefixes = tt, r._domPrefixes = ut, r._cssomPrefixes = rt, r.mq = ot, r.testProp = function(n) {
            return et([n])
        }, r.testAllProps = o, r.testStyles = a, r.prefixed = function(n, t, i) {
            return t ? o(n, t, i) : o(n, "pfx")
        }, e.className = e.className.replace(/(^|\s)no-js(\s|$)/, "$1$2") + (y ? " js " + p.join(" ") : ""), r
    }
    var i;
    r(function(t) {
        if (!t.contentDocument ||!t.contentDocument.documentElement) {
            n.error("Creating an iframe for modernizr failed.");
            return 
        }
        i = u(t, t.contentDocument);
        t.style.display = "none";
        i.transEndEventNames = {
            WebkitTransition: "webkitTransitionEnd",
            MozTransition: "transitionend",
            transition: "transitionend"
        };
        i.animEndEventNames = {
            WebkitTransition: "webkitAnimationEnd",
            MozTransition: "animationend",
            transition: "animationend"
        };
        window.Modernizr = i;
        define("modernizr", i)
    })
}), function(n) {
    function u() {
        return Math.round(r())
    }
    function e(n) {
        var t = {
            hasFired: !1
        }, r = {}, f = n.send || function(i) {
            var r = n.url, f = r.indexOf("?") >= 0 ? "&": "?", u = new Image;
            t.onsent && (u.onload = t.onsent);
            u.src = encodeURI(r + f + "DATA=" + i)
        }, i;
        t.mark = function(n, i) {
            var o = typeof n, e, f;
            if (o === "string")
                for ((i === null || i === undefined) && (i = u()), e = n.split("."), f = r, f[n] = i; e.length;)
                    n = e.shift(), e.length > 0 ? f = f[n] = f[n] || {} : f[n] = i;
            else if (o === "object") {
                i = n;
                for (n in i)
                    i.hasOwnProperty(n) && t.mark(n, i[n])
            }
            return i
        };
        t.fire = function() {
            var i = t.onbeforefire, n;
            i && i();
            this.payload = JSON.stringify(r);
            r = {};
            n = t.onfire;
            n && n();
            t.hasFired=!0
        };
        for (i in n)
            !t.hasOwnProperty(i) && n.hasOwnProperty(i) && (t[i] = n[i]);
        return t
    }
    var i = "now", f = window, t = f.performance, r;
    n.timing = t && t.timing;
    n.navigation = t && t.navigation;
    r = t && t[i] && t[i].bind(t) || Date[i] || function() {
        return (new Date).getTime()
    };
    n.now = u;
    n.Beacon = e
}(Perf || (Perf = {}));
window.pp = new window.Perf.Beacon({}), function(n, t) {
    function r() {
        var r = {}, e = "navigationStart", u = t[e], n, i, f, o, s;
        if (u === null || u === undefined || u < 0)
            return r;
        for (n in t)
            n !== e && (i = t[n], i>-1) && (i >= u && (i -= u), o = (f = n.lastIndexOf("End")) > 0 && 1 || (f = n.lastIndexOf("Start")) > 0 && 0, f!==-1 ? (n = n.substr(0, f), s = r[n] || (r[n] = [ - 1, - 1]), s[o] = i) : r[n] = i);
        return r
    }
    if (t) {
        var i = n.onbeforefire;
        n.onbeforefire = function() {
            i && i();
            n.mark("w3c", r())
        }
    }
}(pp, Perf.timing);
define("w3cTimer", function() {
    return window.pp
});
define("guidGen", function() {
    return function() {
        function t(t) {
            var i = (n + Math.random() * 16)%16 | 0;
            return n = Math.floor(n / 16), (t == "x" ? i : i & 7 | 8).toString(16)
        }
        var n = (new Date).getTime();
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, t)
    }
});
define("pageInstance", ["mediator", "guidGen"], function(n, t) {
    var i, r = "updatePageInstance";
    return n.sub(r, function() {
        var r, u, f, n;
        if (i = t().replace(/-/g, ""), r = document.body.querySelector("#srchfrm"), r)
            for (u = r.querySelectorAll("input[name='refig']"), f = u.length, n = 0; n < f; n++)
                u[n].value = i
    }), {
        getActivityId: function() {
            return i
        },
        eventName: r
    }
});
define("animate", ["jquery", "promise"], function(n, t) {
    function u(n) {
        return new t(function(t) {
            n.one(i.endEvent, t)
        })
    }
    function f(i) {
        if (i.length == 1)
            return u(i);
        var r = [];
        return i.each(function() {
            r.push(u(n(this)))
        }), t.join(r)
    }
    function r(n, u, e) {
        return n.css(e), new t(function(t) {
            if (n.css(u), i.enabled) {
                var o = setTimeout(function() {
                    t("timeout")
                }, r.timeout);
                f(n).then(function() {
                    clearTimeout(o);
                    e.transitionProperty && n.css("transitionProperty", "");
                    e.transitionDuration && n.css("transitionDuration", "0s");
                    t()
                })
            } else 
                t()
        })
    }
    var i = {
        enabled: !1
    };
    return navigator.msPointerEnabled && (i = {
        enabled: !0,
        endEvent: "transitionend",
        timeout: 400
    }), r.timeout = i.timeout, r
});
define("refreshModules", ["jquery", "navigation", "headData", "window", "location", "promise", "requestAnimationFrame", "imgSrc", "mediator"], function(n, t, i, r, u, f, e, o, s) {
    function y() {
        var t, n, i;
        if (r.localStorage && localStorage.deferLoadedItems) {
            for (t = localStorage.deferLoadedItems.split(","), n = 0, i = t.length; n < i; n++)
                t[n] && localStorage.removeItem(t[n]);
            localStorage.removeItem("deferLoadedItems")
        }
    }
    function p(t, i) {
        if (!r.localStorage ||!t ||!i)
            return null;
        var u, f = localStorage.getItem(t);
        try {
            u = JSON.parse(f)
        } catch (e) {
            return null
        }
        return u && u.data && u.lastUpdatedTime && n.now() - u.lastUpdatedTime < i ? u.data : null
    }
    function w(t, i) {
        r.localStorage && (localStorage.deferLoadedItems || (localStorage.deferLoadedItems = ""), localStorage.deferLoadedItems += "," + t, localStorage.setItem(t, JSON.stringify({
            lastUpdatedTime: n.now(),
            data: i
        })))
    }
    function v(t, i, r) {
        var u, f, e;
        i && (u = n("<data>" + n.trim(i) + "<data>"), k(u), f = u.find("[data-module-id]"), t.each(function() {
            var s = n(this), t, i = s.data("moduleId"), u, o;
            t = f.filter("[data-module-id='" + i + "']");
            t.length && (u = n("[data-module-id='" + i + "']"), o = t.html(), o && o != c[i] && (c[i] = o, b(t, u), t.insertAfter(u.first()), u.remove(), e=!0), !r && s.data("cache-duration") && w(i, t[0].outerHTML))
        }))
    }
    function b(t, i) {
        n("img[data-src]", t).each(function() {
            o.force(this, i) || o.go(this)
        })
    }
    function k(t) {
        var i = t.children("link[rel=stylesheet]");
        i.length && i.each(function() {
            var t = this.getAttribute("href");
            l[t] || ((a || (a = n("head"))).append(this), l[t]=!0)
        })
    }
    var h = i.moduleapi, c = {}, l = {}, a;
    return h ? function(e, o) {
        if (i.signedin || y(), !e || e.length === 0)
            return new f(function(n) {
                n()
            });
        e instanceof n || (e = n(e));
        typeof o == "string" && (o = t.getParams(o));
        var k = n.extend(t.getParamsFromUrl(u.href), o), l = n.param(k), w = h.split("?")[0] + (l ? "?" + l : "");
        w += (w.indexOf("?")===-1 ? "?" : "&") + "ou=" + encodeURIComponent(u.href);
        var c = {
            RequestUri: u.href.split("?")[0] + (l ? "?" + l : ""),
            ExperienceId: i.xdid || null
        }, a = {}, b = [];
        return e.each(function() {
            var i = n(this).data("moduleId"), r, u, t;
            if (i) {
                if (r = n(this).data("cache-duration"), u = r ? p(i, r) : null, u) {
                    b.push(u);
                    return 
                }
                if (t = i.split("|"), t.length == 5) {
                    var o = {
                        key: t[3],
                        type: t[4]
                    }, f = {
                        key: t[1],
                        type: t[2],
                        Modules: [o]
                    }, e = f.key + f.type;
                    a[e] ? a[e].Modules.push(o) : a[e] = f;
                    c.Type || (c.Type = t[0])
                }
            }
        }), b.length && v(e, b.join(""), !0), c.Regions = n.map(a, function(n) {
            return n
        }), c.Regions != null && c.Regions.length > 0 ? new f(function(t, i) {
            n.ajax({
                url: w,
                dataType: "html",
                data: JSON.stringify(c),
                method: "POST",
                contentType: "application/json"
            }).done(function(i) {
                v(e, i);
                t(i);
                e.each(function() {
                    var t = n(this).data("moduleId");
                    t && s.pub("moduleRefreshed-" + t)
                })
            }).fail(function() {
                i(arguments)
            })
        }) : (r.console && r.console.log && console.warn("Regions were empty or null"), new f(function(n) {
            n()
        }))
    } : (r.console && console.log("moduleApiEndpoint is missing from header, ajax module update will exit."), n.noop)
});
define("safeCss", ["jquery"], function(n) {
    function r(n) {
        var t = {}, i = "";
        this.css = function(r, u) {
            if (u)
                t[r] = i;
            else if (typeof r == "object")
                for (var f in r)
                    t[f] = i;
            return n.css.apply(n, arguments), this
        };
        this.hide = function() {
            throw "not implemented";
        };
        this.show = function() {
            throw "not implemented";
        };
        this.toggle = function() {
            throw "not implemented";
        };
        this.reset = function() {
            n.css(t);
            t = {}
        }
    }
    var t = 1, i = ".";
    return {
        createGroup: function() {
            function e(t) {
                var h = [], s, e;
                return t.each(function() {
                    var i = n(this), t = i.data(f);
                    t || (t = o++, i.data(f, t));
                    h.push(t)
                }), s = h.join(i), e = u[s], e || (e = new r(t), u[s] = e), e
            }
            var f = "safeCssId" + t++, o = 1, u = {};
            return e.reset = function() {
                for (var n in u)
                    u[n].reset()
            }, e
        }
    }
});
define("scrollManager", ["window", "jquery", "requestAnimationFrame"], function(n, t, i) {
    function r(n) {
        function h() {
            o=!1;
            var n = u.scrollLeft(), i = u.scrollTop();
            t.each(r, function(t, u) {
                u.remove ? delete r[t] : (n != u.left || i != u.top) && (u.handler({
                    left: n,
                    top: i
                }), u.left = n, u.top = i)
            })
        }
        function s() {
            o || (o=!0, i(h))
        }
        var r = {}, u = t(n), e = 0, o=!1, f;
        return {
            register: function(n, t) {
                if (typeof t == "function") {
                    if (!r[n] || r[n].remove) {
                        if (!e)
                            if (f)
                                clearTimeout(f), f = 0;
                            else 
                                u.on("scroll", s);
                        e++
                    }
                    r[n] = {
                        handler: t,
                        top: u.scrollTop(),
                        left: u.scrollLeft(),
                        remove: !1
                    }
                }
            },
            unregister: function(n) {
                r[n] && (r[n].remove=!0, e--, e || (f = setTimeout(function() {
                    u.off("scroll", s);
                    r = {};
                    f = 0
                }, 200)))
            }
        }
    }
    var u = new r(n);
    return u.newManager = function(n) {
        return new r(n)
    }, u
});
define("touchGestures", ["jquery", "modernizr", "window", "deviceGroup"], function(n, t, i, r) {
    function c(n) {
        n.originalEvent && n.originalEvent.preventManipulation && n.originalEvent.preventManipulation();
        n.preventDefault();
        n.stopPropagation()
    }
    function p(n) {
        var t = n.originalEvent, i = t.changedTouches || t.touches;
        return i || (i = [{
            identifier: t.pointerId,
            screenX: t.screenX,
            screenY: t.screenY,
            clientX: t.clientX,
            clientY: t.clientY,
            pageX: t.pageX,
            pageY: t.pageY,
            force: t.pressure,
            target: t.target
        }
        ]), i && i.length ? i : [n]
    }
    function w(n, t) {
        var i = t, r = n.y / n.x;
        return !isNaN(r)&!(n.y < s.minimumDistanceForMove && n.x < s.minimumDistanceForMove) && (i = n.x && Math.abs(Math.atan(r)) < v ? h : y), i
    }
    function l(n, t) {
        return n && t ? {
            x: n.x / t,
            y: n.y / t
        } : {
            x: 0,
            y: 0
        }
    }
    function e(n, t) {
        return n && t ? {
            x: n.x - t.x,
            y: n.y - t.y
        } : {
            x: 0,
            y: 0
        }
    }
    function o(t, i, r) {
        n.each(f[i], function(n, u) {
            t.bind(u, function(n) {
                r(n, i)
            })
        })
    }
    function b(t, i) {
        n.fn[t] = function(n, i) {
            return n ? this.bind(t, i, n) : this.trigger(t, i)
        };
        n.event.special[t] = {
            setup: function(t, r) {
                i(n(this), t, r)
            }
        }
    }
    var s = {
        direction: "all",
        maximumDistanceForClick: 20,
        maximumTimeForClick: 250,
        distanceBeforeDirectionDetection: 10,
        minimumTimeForSwipe: .02,
        maximumTimeForThrow: 1200,
        minimumLastThrowSpeed: 1,
        maximumTimeForSwipe: 400,
        minimumDistanceForSwipe: 30,
        directionChangeBuffer: 5
    }, v = Math.PI / 4, a = n(i), u = {
        eventName: "gesture",
        start: "start",
        move: "move",
        stop: "stop",
        cancel: "cancel",
        tap: "tap",
        swipeUp: "swipeUp",
        swipeRight: "swipeRight",
        swipeDown: "swipeDown",
        swipeLeft: "swipeLeft",
        throwUp: "throwUp",
        throwRight: "throwRight",
        throwDown: "throwDown",
        throwLeft: "throwLeft"
    }, h = "horizontal", y = "vertical", f = {
        pointer: {
            start: ["pointerdown"],
            stop: ["pointerup"],
            move: ["pointermove"],
            cancel: ["pointercancel", "pointerleave"],
            touchSupported: !0,
            type: "pointer"
        },
        mspointer: {
            start: ["MSPointerDown"],
            stop: ["MSPointerUp"],
            move: ["MSPointerMove"],
            cancel: ["MSPointerCancel", "MSPointerLeave"],
            touchSupported: !0,
            type: "mspointer"
        },
        touch: {
            start: ["touchstart", "touchenter"],
            stop: ["touchend", "touchleave"],
            move: ["touchmove"],
            cancel: ["touchcancel"],
            touchSupported: !0,
            type: "touch"
        },
        mouse: {
            start: ["mousedown"],
            stop: ["mouseup"],
            move: ["mousemove"],
            cancel: ["mouseleave"],
            touchSupported: !1,
            type: "mouse"
        }
    };
    return f=!t.touch || r.isPc && typeof InstallTrigger != "undefined" ? navigator.pointerEnabled ? f.pointer : navigator.msPointerEnabled ? f.mspointer : f.mouse : f.touch, b(u.eventName, function(t, i) {
        function it() {
            tt = r = v = k = d = g = 0
        }
        function nt(t, i) {
            if (f.type == "mouse" || (f.type == "pointer" || f.type == "mspointer") && t.originalEvent.pointerType == "mouse")
                b=!0;
            else {
                b=!1;
                var e = p(t);
                (tt || d && r.totalDistance >= y.maximumDistanceForClick) && (tt=!0, c(t));
                (r || i == u.start) && n.each(e, function(n, r) {
                    rt(n, r, t, i)
                })
            }
        }
        function rt(i, f, o, s) {
            var p, rt, ot, st, tt;
            if (ft(f), p = {
                touchType: s,
                coord: {
                    x: f.pageX,
                    y: f.pageY
                },
                screenCoord: {
                    x: f.screenX,
                    y: f.screenY
                },
                scrollCoord: {
                    x: a.scrollLeft(),
                    y: a.scrollTop()
                },
                count: v ? v.count + 1: 0,
                time: (new Date).getTime()
            }, s == u.start && (it(), b=!1, r = n.extend({
                totalDistance: 0
            }, p), v = n.extend({}, p)), p.delta = e(p.coord, v.coord), p.screenDelta = e(p.screenCoord, v.screenCoord), p.duration = p.time - v.time, p.speed = l(p.delta, v.duration), r.delta = e(p.coord, r.coord), r.screenDelta = e(p.screenCoord, r.screenCoord), r.totalDistance += Math.sqrt(Math.pow(p.delta.x, 2) + Math.pow(p.delta.y, 2)), rt = y.direction == h, v && v.speed && (ot = rt ? p.delta.x : p.delta.y, st = rt ? v.delta.x : v.delta.y, ot * st < 0 && (r.movecoord = p.coord, r.movetime = p.time)), r.movecoord && (r.movechange = e(p.coord, r.movecoord), r.moveduration = p.time - r.movetime, r.movespeed = l(r.movechange, r.moveduration)), r.duration = p.time - r.time, r.speed = l(r.delta, r.duration), k = w(r.delta, y.direction), !g && r.totalDistance > y.distanceBeforeDirectionDetection && y.direction != "all"&&!d && (r.direction = k, y.direction != k ? (g=!0, p.touchType = u.cancel) : (c(o), d=!0)), p.touchType == u.stop) {
                var ut = (r.movechange || r.delta).x, et = (r.movechange || r.delta).y, ht = r.moveduration || r.duration, nt = k == h, lt = nt ? Math.abs(r.speed.x): Math.abs(r.speed.y), at = nt ? Math.abs(p.speed.x): Math.abs(p.speed.y), vt = nt ? Math.abs(v.speed.x): Math.abs(v.speed.y), ct = Math.max((at + vt) / 2, lt);
                r.duration < y.maximumTimeForClick && r.totalDistance < y.maximumDistanceForClick ? (b=!0, p.touchType = u.tap) : (p.distance = Math.abs(nt ? ut : et), ct > y.minimumTimeForSwipe && (ht < y.maximumTimeForSwipe ? p.touchType = nt ? ut > 0 ? u.swipeRight : u.swipeLeft : et > 0 ? u.swipeDown : u.swipeUp : ht < y.maximumTimeForThrow && ct > y.minimumLastThrowSpeed && (p.touchType = nt ? ut > 0 ? u.throwRight : u.throwLeft : et > 0 ? u.throwDown : u.throwUp)))
            }(v.touchType != p.touchType || v.coord.x != p.coord.x || v.coord.y != p.coord.y) && (g && p.touchType != u.cancel && p.touchType != u.tap || (tt = n.Event(u.eventName, {
                current: p,
                last: v,
                start: r,
                direction: k,
                originalEvent: o,
                allowClick: b
            }), t.trigger(tt), b = tt.allowClick!==!1, tt.isDefaultPrevented() && o.preventDefault(), tt.isPropagationStopped() && o.stopPropagation(), tt.isImmediatePropagationStopped() && o.stopImmediatePropagation()));
            s == u.stop || s == u.cancel ? it() : v = p
        }
        function ut(n) {
            return b || c(n), b
        }
        function ft(n) {
            n.pageX || n.originalEvent && (n.pageX = n.originalEvent.pageX, n.pageY = n.originalEvent.pageY, n.screenX = n.originalEvent.screenX, n.screenY = n.originalEvent.screenY, n.pointerId = n.originalEvent.pointerId, n.identifier = n.originalEvent.identifier)
        }
        var r, v, k, d, tt, g, b=!0, y = n.extend({}, s, i);
        it();
        t.addClass("unselectable").find("A, [onclick]").not(".mobilead").bind("click", ut);
        o(t, u.start, nt);
        o(t, u.move, nt);
        o(t, u.stop, nt);
        o(t, u.cancel, nt)
    }), {
        types: u,
        settings: f
    }
});
define("touchDataManager", ["jquery"], function(n) {
    function o() {
        var c = .85, l = .1, h, o, u, r, n;
        this.reset = function(t) {
            h = t;
            o =+ new Date;
            u = o;
            r = h;
            n = 0
        };
        this.input = function(t) {
            var e =+ new Date, i = e - u, f;
            i = i > 0 ? i : 1;
            u = e;
            r += n * i;
            f = t - r;
            r += c * f;
            n += l * f / i
        };
        this.getPosition = function() {
            return r
        };
        this.getVelocity = function() {
            return n
        };
        this.getDelta = function() {
            return r - h
        };
        this.getDuration = function() {
            return 220
        };
        this.getTouchDuration = function() {
            return u - o
        };
        this.getDrift = function(t, i) {
            var e = this.getDelta(), l, s;
            e < 0 && (e =- e);
            var h = 1 - e / t, c = .2, r = t * (h > c ? h : c), f = n * 150, a = u - o < 300 && (n > .3 || n<-.3);
            return a ? (l = i, s = l - e, f = (n > 0 ? 1 : - 1) * (s < r ? s : r)) : f > r ? f = r : f<-r && (f =- r), f
        };
        this.getTimingFunction = function() {
            return "cubic-bezier(0," + e(n) + ",.58,1)"
        };
        this.ensureJQueryEase = function() {
            var r = e(n);
            r = Math.round(r / f) * f;
            i[r] || (i[r] = s(r));
            t = i[r]
        };
        this.reset(0)
    }
    function e(n) {
        var t = n > 0 ? n: - n;
        return t * .3
    }
    function s(n) {
        for (var f = [], t, e, o, i, u = r.length; u--;)
            t = r[u], i = 1 - t, o = i * t * t * .58 + t * t * t, e = i * i * t * n + i * t * t + t * t * t, f[u] = {
                x: e,
                t: o
            };
        return f
    }
    for (var i = {}, f = .1, t, r = [], u = 0; u <= 1; u += .01)
        r.push(u);
    return n.easing.cubicBezier = function(n) {
        if (n == 1)
            return 1;
        for (var r, o, u, s, i, f = t.length - 1, e = 0; f--&&!e;)
            i = t[f], r = i.t, r <= n && (u = i.x, i = t[f + 1], o = i.t, s = i.x, e = (n - r) / (o - r) * (s - u) + u);
        return e
    }, n.easing.cubicBezierQuint = function(n) {
        return 1 - Math.pow(1 - n, 5)
    }, o
});
define("carouselManager", function() {
    return function() {
        var n = [], t = 0;
        this.load = function(t) {
            n = t
        };
        this.getItem = function(i) {
            if (!n.length)
                return null;
            var r = (t + i)%n.length;
            return n[r < 0 ? r + n.length: r]
        };
        this.setIndex = function(n) {
            t = n
        };
        this.changeIndex = function(n) {
            t += n
        }
    }
});
globalLeft = "left";
define("dir.tokens", {
    ltr: globalLeft == "left",
    left: "left",
    paddingLeft: "padding-left",
    paddingRight: "padding-right",
    marginLeft: "margin-left",
    marginRight: "margin-right"
});
define("socialPlugins.tokens", {
    twitterLang: "en",
    facebookLang: "en_US",
    loadDelay: "1000"
});
define("truncate.tokens", {
    truncateEllipsis: "…"
});
define("webApp.tokens", {
    appVersion: "2.0.5442.29569"
});
define("sharingToolbar.tokens", {
    shareCountMinimumThreshold: 1
});
define("feedback.tokens", {
    inputBoxError: "Please leave a comment.",
    categoryNotSelectedError: "Please leave a comment or select a problem category."
});
define("flyout", ["jquery", "jqBehavior", "document"], function(n, t, i) {
    function r(t, r) {
        function p(n) {
            var t = f[0], i;
            n.preventDefault();
            t.el.hasClass(t.cls) ? (i = s ? s[0] : !1, r.allowToggleOff && c(n, h), i !== n.target && u.filter(n.target).length === 1 && y(n, e)) : y(n, e)
        }
        function w(i) {
            var f = t.find(r.contentEventOrigin), e;
            f.length || (f = t);
            e = r.skipContentOriginEventTargetCheck?!(n.contains(f[0], i.target)===!0 || u[0] === i.target) : !(f[0] === i.target || n.contains(f[0], i.target)===!0 || u[0] === i.target);
            e && (r.disableEventBubbling && i.preventDefault(), c(i, h))
        }
        function h(t, i) {
            for (var u, r = 0; r < f.length; r++)
                u = f[r], u.el.removeClass(u.cls).attr("aria-hidden", !0);
            a.off(v, w);
            s=!1;
            n.isFunction(i) && i !== e && i()
        }
        function b(n) {
            t.find("." + r.toggleClass).length && c(n, h)
        }
        function e(t, i) {
            var r, u;
            for (s = n(t.target), r = 0; r < f.length; r++)
                u = f[r], u.el.addClass(u.cls).attr("aria-hidden", !1);
            a.on(v, w);
            n.isFunction(i) && i !== e && i()
        }
        var a = n(i), u = t.find(r.eventOrigin), l;
        u.length || (u = n(r.eventOrigin));
        var v = r.outsideEvent, o = r.toggleTargets, s=!1, y = n.isFunction(r.openFlyoutDelegate) ? r.openFlyoutDelegate : e, c = n.isFunction(r.closeFlyoutDelegate) ? r.closeFlyoutDelegate : h, f = [];
        if (typeof o != "object")
            o===!0 && f.push({
                el: t,
                cls: "show"
            });
        else 
            for (l in o)
                f.push({
                    el: t.find(l),
                    cls: o[l]
                });
        return {
            setup: function() {
                u.on(r.triggerEvent, p)
            },
            update: function() {
                b()
            },
            teardown: function() {
                b();
                u.off(r.triggerEvent, p)
            }
        }
    }
    return t(r, {
        allowToggleOff: !0,
        eventOrigin: ".flyout > button",
        toggleTargets: !0,
        triggerEvent: "click",
        outsideEvent: "mousedown touchstart scroll",
        openFlyoutDelegate: !1,
        skipContentOriginEventTargetCheck: !1,
        closeFlyoutDelegate: !1
    })
});
define("feedback", ["jquery", "trackInfo", "feedback.tokens", "flyout", "logging", "jqBehavior", "mediator", "document", "headData"], function(n, t, i, r, u, f, e, o, s) {
    function y(f) {
        function ht() {
            et ? o.body.appendChild(d) : (d = o.createElement("div"), d.setAttribute("class", "layoutoverlay"), o.body.appendChild(d), y.$provideFeedback = f.find(".feedbackheader .selected"), y.$reportProblem = n("#feedback-reportproblem"), y.$submitButton = n("#feedback-submit"), y.$close = n("#feedback-ok"), y.$closeButton = n("#feedback-close"), y.$feedbackArea = n("#feedback-feedbackarea"), y.$feedbackInput = n("#feedback-inputbox"), y.$userAlias = n("#feedback-useralias"), y.$starRating = f.find(".stars > ul > li"), et=!0);
            b=!1;
            y.$selectedCategory = null;
            w.text(st);
            w.removeClass("uploading uploaded");
            rt = ct();
            it=!1
        }
        function kt() {
            if (t.sitePage) {
                var n = t.sitePage;
                p["feedback.id"] = rt;
                p.pageType = n.pageName;
                p.subcvs = n.vertical;
                p.cvs = n.canvas;
                p.contentid = n.entityId;
                p.cu = n.pageUrl;
                p.evt = "feedback";
                p.mkt = n.localeCode;
                p["client.deviceOS"] = n.os;
                p.flightid = n.flightid;
                p.referrer = o.referrer;
                p.signedin = t.userStatic ? t.userStatic.isSignedIn : "";
                p.scr = screen.width && screen.height ? screen.width.toString() + " X " + screen.height.toString() : "";
                p["client.deviceModel"] = s ? s.dg : ""
            }
        }
        function dt() {
            y.$feedbackInput.length && (p.submitValue = y.$feedbackInput[0].value);
            y.$userAlias.length && (p["feedback.source"] = y.$userAlias[0].value)
        }
        function gt() {
            if (p["feedback.area"])
                return k && nt(), !0;
            if (!k && y.$feedbackInput.length) {
                if (y.$feedbackInput[0].value.length > 0)
                    return !0;
                y.$feedbackInput.addClass("error");
                y.$feedbackInput[0].value = b ? i.categoryNotSelectedError : i.inputBoxError;
                y.$feedbackInput.on("focus", nt);
                k=!0
            }
            return !1
        }
        function nt() {
            k && (y.$feedbackInput.removeClass("error").off("focus", nt), k=!1);
            y.$feedbackInput[0].value = ""
        }
        function ct() {
            var n = (new Date).getTime();
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(t) {
                var i = (n + Math.random() * 16)%16 | 0;
                return n = Math.floor(n / 16), (t == "x" ? i : i & 7 | 8).toString(16)
            })
        }
        function lt(n) {
            n.preventDefault();
            g.click()
        }
        function at() {
            var t = g[0], r = t.value ? t.value.replace(/^.*(\\|\/|\:)/, ""): "", i;
            w.text(r || st);
            t.value && t.files && t.files.length && (w.addClass("uploading"), i = new window.FormData, i.append("file", t.files[0]), n.ajax({
                url: h + "?" + n.param({
                    "feedback.id": rt
                }),
                data: i,
                processData: !1,
                contentType: !1,
                type: "POST"
            }).done(function() {
                w.removeClass("uploading");
                w.addClass("uploaded")
            }))
        }
        function ni() {
            kt();
            dt();
            n.ajax({
                type: "POST",
                url: h,
                data: JSON.stringify([p]),
                contentType: "application/json; charset=utf-8",
                dataType: "json"
            }).fail(function(n, t, i) {
                u.error("Error submitting user feedback. ErrorType: {0}. Error: {1}".format(t, i.message))
            })
        }
        function ti(n, t) {
            ht();
            y.$submitButton.on("click", wt);
            y.$closeButton.on("click", tt);
            y.$reportProblem.on("click", bt);
            y.$provideFeedback.on("click", ft);
            y.$starRating.on("click", pt);
            g.on("change", at);
            w.on("click", lt);
            e.pub("feedback.open");
            t(n)
        }
        function vt(n) {
            var t = y.$feedbackArea.find(".feedbackarea > ul > li");
            if (t.length)
                if (n)
                    t.on("click", yt);
                else 
                    t.off("click", yt)
        }
        function yt(t) {
            var i = n(t.currentTarget);
            p["feedback.area"] = i.text();
            p["feedback.type"] = "bug";
            y.$selectedCategory && y.$selectedCategory.removeClass("selectedcategory");
            y.$selectedCategory = i;
            i.addClass("selectedcategory")
        }
        function pt(t) {
            var i = n(t.currentTarget), r = i.index() + 1;
            i.parent().attr("data-selected", r);
            p["feedback.rating"] = r
        }
        function ii() {
            p["feedback.rating"] = 0;
            y.$starRating.parent().attr("data-selected", 0).off("click", pt)
        }
        function tt() {
            e.pub("feedback.close");
            it && (y.$close.off("click", tt), ni());
            f.removeClass("show finalpage");
            o.body.removeChild(d);
            y.$submitButton.off("click", wt);
            y.$closeButton.off("click", tt);
            y.$reportProblem.off("click", bt);
            y.$provideFeedback.off("click", ft);
            g.off("change", at);
            w.off("click", lt);
            nt();
            ii();
            b && ft()
        }
        function wt() {
            if (gt()) {
                f.addClass("finalpage");
                it=!0;
                y.$close.on("click", tt)
            }
        }
        function bt() {
            b || (b=!0, y.$feedbackArea.addClass("showproblemarea"), y.$provideFeedback.removeClass("selected"), y.$reportProblem.addClass("selected"), vt(!0))
        }
        function ft() {
            b && (vt(!1), p["feedback.type"] = "General", p["feedback.area"] = undefined, y.$provideFeedback.addClass("selected"), y.$reportProblem.removeClass("selected"), y.$feedbackArea.removeClass("showproblemarea"), b=!1, y.$selectedCategory && y.$selectedCategory.removeClass("selectedcategory"), y.$selectedCategory = null)
        }
        var et=!1, ot = {
            $provideFeedback: null,
            $reportProblem: null,
            $submitButton: null,
            $close: null,
            $closeButton: null,
            $feedbackInput: null,
            $feedbackArea: null,
            $starRating: null,
            $selectedCategory: null,
            $userAlias: null
        }, b, k, d, it, y = ot, p = {}, rt = ct(), g = f.find(a), w = f.find(v), st = "Upload a Screenshot", ut;
        return p["feedback.rating"] = 0, p["feedback.type"] = "General", ut = r(c, o, {
            eventOrigin: l,
            outsideEvent: "",
            openFlyoutDelegate: ti,
            allowToggleOff: !1
        }), {
            setup: ut.setup,
            teardown: ut.teardown,
            feedback: {
                initialize: ht,
                feedbackVariables: ot
            }
        }
    }
    var h = "http://otf.msn.com/c.gif", c = "#feedback-flyout", l = "#footer_feedback, #usermenu-feedback, #feedback-bottomentry", a = "#feedback-file", v = "#feedback-upload";
    return f(y)
});
define("hover", ["jquery", "jqBehavior", "mediator", "tabKeyPressed", "pointerEvents"], function(n, t, i, r, u) {
    function s(n, t) {
        function b(n) {
            return n && (!n.originalEvent ||!n.originalEvent.pointerType || n.originalEvent.pointerType != n.originalEvent.MSPOINTER_TYPE_TOUCH && n.originalEvent.pointerType != "touch")
        }
        function h(n) {
            f.disabled===!1 && b(n) && (clearTimeout(w), l || (l=!0, y.addClass(t.hoverClass), i.pubChannel("hoverShow", a, n)))
        }
        function s(n) {
            b(n) && l && (w = setTimeout(function() {
                l=!1;
                v && clearTimeout(v);
                y.removeClass(t.hoverClass);
                i.pubChannel("hoverHide", a, n)
            }, it))
        }
        function k(n) {
            t.preventClick && n && (n.preventDefault(), n.stopImmediatePropagation());
            c && y.hasClass(t.hoverClass) ? s({}) : (c=!0, h({}), t.autoHide && t.clickAutoHideTimeout > 0 && (v = setTimeout(s, t.clickAutoHideTimeout, n)))
        }
        function d(n) {
            r() && h(n)
        }
        function g(n) {
            r() && s(n)
        }
        function nt(n) {
            c=!1;
            setTimeout(function() {
                c=!0
            }, 200);
            h(n)
        }
        var tt = t.enableClickEvents, a = o++, v, y = t.hoverClassTarget || n, p = t.clickTarget || n, c=!0, l=!1, w, it = t.hideEventTimeout;
        return n.data(e, a), {
            setup: function() {
                if (u.enabled) {
                    n.on(u.over, h);
                    t.autoHide && n.on(u.out, s)
                } else {
                    n.on("mouseenter", nt);
                    t.autoHide && n.on("mouseleave", s)
                }
                if (tt)
                    p.on("click", k);
                n.on("focusin", d);
                t.autoHide && n.on("focusout", g)
            },
            teardown: function() {
                u.enabled ? (n.off(u.over, h), t.autoHide && n.off(u.out, s)) : (n.off("mouseenter", nt), t.autoHide && n.off("mouseleave", s));
                p.off("click", k);
                n.off("focusin", d);
                t.autoHide && n.off("focusout", g)
            }
        }
    }
    var e = "hoverId", o = 1, f = t(s, {
        autoHide: !0,
        clickAutoHideTimeout: 2500,
        enableClickEvents: !0,
        hideEventTimeout: 220,
        hoverClass: "hover",
        hoverClassTarget: null,
        preventClick: !0
    });
    return f.hoverId = e, f.disabled=!1, f
});
define("marketDropdown", ["jquery", "jqBehavior", "window", "headData", "track"], function(n, t, i, r, u) {
    function e(t, r) {
        function o() {
            t.on(r.triggerEvent, e)
        }
        function e() {
            var i, n, e;
            t[0].tagName == "SELECT" ? (e = t.find("option:selected"), i = e.data("marketSwitchLocale"), n = e.data("marketSwitchUrl")) : (i = r.marketSwitchLocale, n = r.marketSwitchUrl);
            n && i && (u.trackEvent({
                type: "click_nonnav",
                target: t
            }, "", "", n), h(f, n, i))
        }
        function s() {
            t.off(r.triggerEvent, e)
        }
        function h(t, r, u) {
            var f = JSON.stringify({
                Market: u,
                SuppressPrompt: !0
            }), e = [{
                Path: "MSNHomePage/MarketConfiguration",
                Operation: "Update",
                Payload: f
            }
            ];
            n.ajax({
                url: t,
                dataType: "html",
                data: JSON.stringify(e),
                method: "POST",
                contentType: "application/json"
            }).done(function() {
                i.location.href = r
            }).fail(function() {
                i.location.href = r
            })
        }
        return {
            setup: o,
            teardown: s
        }
    }
    var f = r.pdpdeltaupdateapi;
    return t(e, {
        triggerEvent: "change"
    })
});
define("navArrowScroll", ["jquery", "jqBehavior", "modernizr", "device", "window", "dir.tokens", "touchGestures"], function(n, t, i, r, u, f, e) {
    function l(t, i) {
        function yt() {
            it.on("click", d);
            rt.on("click", p);
            ri(r);
            kt();
            y();
            ii()
        }
        function pt() {
            it.off("click", d);
            rt.off("click", p);
            y()
        }
        function wt() {
            w = r.position().left;
            c = 0;
            st(0);
            y()
        }
        function ot(n) {
            ht();
            var t = bt(n);
            st(t)
        }
        function st(n) {
            s ? (n = o.replace("{0}", n), r.css("transform", n)) : r.css(o, n)
        }
        function ht() {
            if (g = h.width(), tt)
                a = r[0].scrollWidth;
            else {
                a = 0;
                for (var n = 0; n < r.length; n++)
                    a += r[n].scrollWidth
            }
            v = r.position().left - w
        }
        function bt(n) {
            var t = 0, i;
            return n == 1 ? c > 0 && (c = c - 1, t = l[c].getAttribute(nt)) : c < l.length - 1 && (t = l[c].getAttribute(nt), c = c + 1), i = f.ltr ? n : - n, ft = v + i * t
        }
        function kt() {
            for (var t, n = 0, i = l.length; n < i; n++)
                t = l.eq(n), l[n].setAttribute(nt, t.outerWidth(!0)), t.hasClass(vt) && (et = n)
        }
        function ct() {
            f.ltr || (w = r.position().left, v = r.position().left - w)
        }
        function y() {
            if (ht(), a > g) {
                tt=!0;
                dt();
                var n = f.ltr ? v: - v, t = parseInt(r.css(f.marginLeft)), i = g - (a + n);
                c == 0 ? (h.addClass(k).removeClass(b), ct()) : i >= t ? h.addClass(b).removeClass(k) : h.addClass(ut)
            } else 
                tt=!1, h.removeClass(ut), gt(), ct()
        }
        function dt() {
            for (var t = r[1].children.length, n = 0; n < t; n++)
                r[1].children[0].id = "mylink_" + n, r[0].appendChild(r[1].children[0])
        }
        function gt() {
            for (var t = r[0].children.length, n = 0; n < t; n++)
                !r[0].children["mylink_" + n] || r[1].appendChild(r[0].children["mylink_" + n])
        }
        function d() {
            ot(1);
            y()
        }
        function p() {
            ot( - 1);
            y()
        }
        function ni() {
            f.ltr ? p() : d()
        }
        function ti() {
            f.ltr ? d() : p()
        }
        function lt() {
            return f.ltr && h.hasClass(k) ||!f.ltr && h.hasClass(b)
        }
        function at() {
            return f.ltr && h.hasClass(b) ||!f.ltr && h.hasClass(k)
        }
        function ii() {
            for (var n = 0; n < et; n++)(f.ltr && lt() ||!f.ltr && at()
                ) && p()
        }
        function ri(n) {
            e.settings.touchSupported && n.gesture(ui, {
                direction: "horizontal"
            })
        }
        function ui(t) {
            if (t.direction)
                switch (t.current.touchType) {
                case e.types.throwRight:
                case e.types.swipeRight:
                    at() && ti();
                    break;
                case e.types.throwLeft:
                case e.types.swipeLeft:
                    lt() && ni();
                    break;
                case e.types.stop:
                case e.types.tap:
                    var i = t.originalEvent.originalEvent, r = i.target, f = document.createEvent("MouseEvent");
                    f.initMouseEvent("click", !1, !1, u, i.detail, i.screenX, i.screenY, i.clientX, i.clientY, i.ctrlKey, i.altKey, i.shiftKey, i.metaKey, i.button, null);
                    r.tagName !== "A" && (r = n(r).parentsUntil("LI", "A")[0]);
                    r && r.dispatchEvent(f)
                }
        }
        var h = t, r = h.find(i.navSelector), l = h.find(i.navItemSelector), it = h.find("button").first(), rt = h.find("button").last(), w = r.position().left, c = 0, g = 0, a = 0, b = "show-left", k = "show-right", ut = "show-left show-right", vt = "current", ft = 0, v = 0, nt = "data-width", et = 0, tt=!1;
        return {
            setup: yt,
            teardown: pt,
            update: wt
        }
    }
    var o = "left", s=!1, c, h;
    return i.csstransforms && r.isCapable("AllowTransform2d") && (o = "translateX({0}px)", s=!0), c = n(u), h = {
        navSelector: "ul",
        navItemSelector: "ul > li"
    }, t(l, h)
});
define("scrollStop", ["jquery", "jqBehavior", "touchGestures"], function(n, t, i) {
    function f(t, f) {
        function o() {
            e===!1 ? e=!0 : clearTimeout(s);
            s = setTimeout(a, l)
        }
        function h() {
            e && o()
        }
        function a() {
            e=!1;
            t.trigger("scrollStop")
        }
        var c = n.extend({}, u, f), l = c.delay, e, s;
        return {
            setup: function() {
                e=!1;
                t.on("scroll", o);
                if (i.settings.touchSupported)
                    t.on(r, h)
            },
            teardown: function() {
                t.off(r, h);
                t.off("scroll", o)
            }
        }
    }
    var u = {
        delay: 100
    }, e = i.settings.start.join(" "), r = i.settings.move.join(" ");
    return t(f, {})
});
define("sharingToolbar", ["sharingToolbar.tokens", "jquery", "jqBehavior", "track", "format", "window"], function(n, t, i, r, u, f) {
    function l(n) {
        n.each(function() {
            var n = t(this);
            n.hasClass(h) || n.find("a.stb-btn").each(function() {
                var n = t(this);
                n.click(function() {
                    return (n.hasClass(e.toolbarClassPrefix + "pinterest") && a(n), n.hasClass(e.toolbarClassPrefix + "email"))?!0 : (s(n.attr("href")), !1)
                })
            })
        })
    }
    function a(n) {
        var f = n.attr("href"), t = y(f), r = v(), u, i;
        r && t && (u = encodeURIComponent(r.split("?")[0]), w(t, u), i = b(t), i && n.attr("href", i))
    }
    function v() {
        var n = null, f = t(".gallery"), e = t("section[itemprop='articleBody']"), i, r, u;
        return f.length ? (i = f.find(".show"), i.length && (r = i.find(".image"), r.length && (n = r[0].src))) : e.length && (u = e.find(".image"), u.length && (n = u[0].src)), n
    }
    function y(n) {
        if (!n)
            return null;
        var t = {};
        return (n.replace(c, function(n, i, r, u) {
            !n || r || u ? i && u && (t[i] = u) : t.baseUrl = n
        }), p(t) !== 4) ? null : t
    }
    function p(n) {
        var t = 0;
        for (var i in n)
            n.hasOwnProperty(i) && t++;
        return t
    }
    function w(n, t) {
        n && n.media && t && (n.media = t)
    }
    function b(n) {
        var i, t;
        if (n && n.baseUrl && n.media && n.url && n.description) {
            i = n.baseUrl + "?";
            for (t in n)
                n.hasOwnProperty(t) && t !== "baseUrl" && (i += t + "=" + n[t] + "&");
            return i.substring(0, i.length - 1)
        }
        return null
    }
    function s(n) {
        f.open(n, null, "toolbar=0,status=0,resizable=1,scrollbars=1")
    }
    var e = {
        toolbarClassPrefix: "stb-"
    }, h = "stb-processed", c = new RegExp("([^?=&]+)(=([^&]*))?", "g"), o = i(l);
    return o.openNewWindow = s, o
});
define("searchBing", ["jquery", "jqBehavior", "track", "device"], function(n, t, i, r) {
    return r.capability("SupportFixedPosition") !== "false" && r.capability("SupportFixedPosition") || n("body>.head").addClass("unfixposition"), t(function(t) {
        function f() {
            var n = t.attr("action"), r = t.attr("target");
            i.trackEvent({
                type: "submit",
                target: t[0]
            }, null, u);
            t.attr("action", u).attr("target", "_blank").submit().attr("action", n).attr("target", r)
        }
        var r = n("button[name='bingSearch']", t), e = n("body>.head"), u = t.data("bing-action-uri");
        return {
            setup: function() {
                r.on("click", f)
            },
            teardown: function() {
                r.off("click", f)
            }
        }
    })
});
define("searchTargetSelf", ["jqBehavior"], function(n) {
    function u(n) {
        return i = n.attr(t), {
            setup: function() {
                n.attr(t, r)
            },
            teardown: function() {
                n.attr(t, i)
            }
        }
    }
    var t = "target", r = "_self", i;
    return n(u)
});
define("socialPlugins", ["jquery", "jqBehavior", "format", "socialPlugins.tokens"], function(n, t, i, r) {
    function e(n, t) {
        function f() {
            if (t.facebookLikeUrl && t.facebookButtonWidth) {
                var f = "//www.facebook.com/plugins/like.php?locale=" + encodeURIComponent(r.facebookLang) + "&href=" + encodeURIComponent(t.facebookLikeUrl) + "&amp;send=false&amp;layout=button_count&amp;width=450&amp;show_faces=false&amp;font&amp;colorscheme=light&amp;action=like&amp;height=21";
                n.find(t.facebookSelector).append(i(u, f, t.facebookButtonWidth))
            }
        }
        function e() {
            var f, e;
            t.twitterFollowScreenName && t.twitterButtonWidth && (f = "//platform.twitter.com/widgets/follow_button.html?show_screen_name=" + t.twitterShowScreenName + "&screen_name=" + encodeURIComponent(t.twitterFollowScreenName) + "&show_count=false&lang=" + encodeURIComponent(r.twitterLang), n.find(t.twitterSelector).append(i(u, f, t.twitterButtonWidth)), t.twitterShowScreenName && (e = t.twitterButtonWidth, typeof InstallTrigger != "undefined" && (e = parseFloat(t.twitterButtonWidth) + .3 + "rem"), n.find(t.twitterBreakingNewsSelector).append(i(u, f, e))))
        }
        window.setTimeout(function() {
            f();
            e()
        }, t.delay)
    }
    var f = {
        facebookSelector: "#fbcount",
        twitterSelector: "#twcount",
        twitterBreakingNewsSelector: "#bnewstwcount",
        delay: r.loadDelay
    }, u = '<iframe src="{0}" scrolling="no" frameborder="0" allowTransparency="true" allowTransparency="true" style="width:{1};"><\/iframe>';
    return t(e, f)
});
define("truncate", ["jquery", "jqBehavior", "mediator", "requestAnimationFrame", "measure", "format", "truncate.tokens"], function(n, t, i, r, u, f, e) {
    function l(t, e) {
        function st() {
            var i, f, e, o, r;
            if (it) {
                for (it=!1, i = l; i--;) {
                    for (f = "", e = 0, r = t[i].firstChild; r != null;)
                        r.nodeType == 1 ? (++e, o = r) : r.nodeType == 3 && (f += r.nodeValue), r = r.nextSibling;
                    e == 1 && n.trim(f) == "" && (p[i] = o)
                }
                for (i = l; i--;)
                    w[i] = u(t[i]), v[i] = (p[i] || t[i]).innerHTML, nt[i] = t[i].title
            } else 
                ot();
            for (i = l; i--;)
                rt(i), b[i] = v[i], y[i]=!0;
            ut()
        }
        function ht() {
            for (var n = l; n--;)
                rt(n), y[n] = t[n].scrollHeight > a[n];
            ut()
        }
        function rt(n) {
            var i = t[n];
            if (i && i.parentNode) {
                var r = w[n], o = i.getAttribute("data-truncate-lines") || e.truncateLines, u = parseFloat(r("paddingTop")), s = parseFloat(r("paddingBottom")), h = r("lineHeight"), f = parseFloat(r("maxHeight"));
                o ? (a[n] = parseFloat(h) * o + s, a[n] > f && (a[n] = f), a[n] += u) : a[n] = (f || i.clientHeight) + u;
                a[n] = a[n] + .5 | 0;
                tt[n] = i.clientWidth;
                g[n] = u + 2 * parseFloat(h) + s + .5 | 0
            } else 
                w[n] = p[n] = null, y[n] = d[n]=!1
        }
        function ut() {
            for (var f=!0, e = 1e3, i, r, u; f&&--e;) {
                for (f=!1, i = l; i--;)
                    if (y[i])
                        if (d[i] = t[i].scrollHeight > a[i], d[i]) {
                            if (k[i]=!0, t[i].scrollHeight < g[i]) {
                                ft(i);
                                continue
                            }
                            r = b[i];
                            r.slice( - s) == o && (r = r.slice(0, - s));
                            u = c.exec(r);
                            u && u[0] != r ? (f=!0, t[i].scrollHeight > a[i] * 2 && (r = r.substr(0, r.length / 2)), b[i] = r.substr(0, r.length - u[0].length) + o) : ft(i)
                        } else 
                            y[i]=!1;
                for (i = l; i--;)
                    d[i] && ((p[i] || t[i]).innerHTML = b[i])
            }
            for (i = l; i--;)
                k[i] && (t[i].title = n.trim(n("<span>" + v[i] + "<\/span>").text()))
        }
        function ft(n) {
            var t = tt[n] - parseInt(w[n]("paddingLeft")) - parseInt(w[n]("paddingRight"));
            b[n] = t > 0 ? f(h, v[n], t + "px") : v[n];
            y[n]=!1
        }
        function et() {
            r(st)
        }
        function ot() {
            for (var n = l; n--;)
                k[n] && ((p[n] || t[n]).innerHTML = v[n], t[n].title = nt[n], k[n]=!1)
        }
        var l = t.length, p = new Array(l), w = new Array(l), g = new Array(l), v = new Array(l), nt = new Array(l), tt = new Array(l), b = new Array(l), a = new Array(l), y = new Array(l), k = new Array(l), d = new Array(l), it=!0;
        return i.sub("truncate", function() {
            r(ht)
        }), {
            setup: et,
            teardown: ot,
            update: et
        }
    }
    var h = '<span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;width:{1}">{0}<\/span>', o = e.truncateEllipsis, s = o.length, c = /(?:\s|[,!\.\?:;])*([\u3000-\u30ff\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[^\s\u3000-\u30ff\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+)(?:\s|[,!\.\?:;])*$/;
    return t(l, null, {
        each: 1
    })
});
define("twitterComments", ["jquery", "jqBehavior", "window"], function(n, t, i) {
    function r(t) {
        function r(t) {
            t.preventDefault();
            i.open(n(this).attr("href"), "Twitter", "status=0,toolbar=0,location=0,resizable=1,scrollbars=1,left=" + (i.screen.width - 250) / 2 + ",top=" + (i.screen.height - 350) / 2 + ",width=500,height=450")
        }
        return {
            setup: function() {
                t.find(".twitteractions > a").on("click", r)
            },
            teardown: function() {
                t.find(".twitteractions > a").off("click")
            }
        }
    }
    return t(r)
});
require(["window", "jquery", "device", "mediator"], function(n, t, i, r) {
    var f, u = t("meta[name=viewport]"), e = i.capability("Viewport_Landscape"), s = u.attr("content"), o = function() {
        n.innerHeight > n.innerWidth ? u.attr("content", s) : u.attr("content", e);
        r.pubChannel("update", "mediaQuery")
    };
    u.length && e && (o(), t(n).resize(function() {
        clearTimeout(f);
        f = setTimeout(o, 50)
    }))
});
require(["jquery", "navigation", "headData", "webApp.tokens"], function(n, t, i, r) {
    var u = /(\?|&)ver=[\d\.]+(&|$)/gi;
    n.ajaxPrefilter(function(n) {
        var f;
        if (n = n || {}, f = n.url, i.ssl===!0 && f.substr(0, 7).toLowerCase() == "http://" && (f = "https://" + f.substr(7)), !n.notVersioning && f && t.isLocal(f)&&!f.match(u)) {
            var e = f.split("#"), o = e[0], s = e[1] ? "#" + e[1]: "";
            n.url = o + (o.indexOf("?")>-1 ? "&" : "?") + "ver=" + r.appVersion + s
        }
    })
});
require(["getCookie", "setCookie", "navigation", "location", "c.onload"], function(n, t, i, r) {
    var f = 365, u = "ocid", e = "hp", o = "homepage";
    (function() {
        var s = i.getParamsFromUrl(r.href), h = s && s[u] ? s[u].toLowerCase(): "";
        h.indexOf(e)==-1 && h.indexOf(o)==-1 || n(u) || t(u, h, f, t.topDomain, "/")
    })()
});
require(["binding", "c.dom"], function(n) {
    require(["c.deferred"], function() {
        require(["jquery", "imgSrc"], function(n, t) {
            n("#username img,#sign-in img").each(function() {
                t.go(this)
            })
        });
        n("twitterComments", "#breakingnews").all();
        n("marketDropdown", "#site-market").all();
        n("marketDropdown", ".marketswitch").all({
            triggerEvent: "click"
        });
        n("marketDropdown", ".langtoggle").all({
            triggerEvent: "click"
        });
        require(["c.onload"], function() {
            n("socialPlugins", "#social-plugins").all();
            n("socialPlugins", "#bnewssocial-plugins").all()
        });
        n("makeHomePage", "#makehomepage").all()
    });
    n("poll", ".pollcontainer").all();
    n("flyout", ".megamenu").all({
        eventOrigin: "#header-common>h1>.vertical",
        skipContentOriginEventTargetCheck: !0
    });
    n("flyout", "#header-common").view(n.views.SIZE12COLUMN, {
        eventOrigin: "#header-search>button",
        outsideEvent: "mousedown touchstart"
    });
    n("flyout", "#header-common").view(n.views.SIZE34COLUMN, {
        eventOrigin: "#q",
        triggerEvent: "focus click",
        allowToggleOff: !1,
        outsideEvent: "mousedown touchstart",
        contentEventOrigin: "#srchfrm"
    });
    n("flyout", "#username").all({
        eventOrigin: "#username>button,#username>figure"
    });
    n("flyout", "#sign-in").all({
        eventOrigin: "#sign-in>a"
    });
    n("flyout", "#language-toggle").view(n.views.SIZE34COLUMN, {
        eventOrigin: "#language-toggle>button"
    });
    n("feedback", "#feedback-flyout").all();
    n("searchBing", "#srchfrm").all();
    n("navArrowScroll", ".navinner").all();
    n("sharingToolbar", ".stb").all();
    n("sharingToolbar", ".pinit-button").all()
});
require(["sso", "headData", "c.deferred"], function(n) {
    n();
    var t = document.getElementsByTagName("head")[0];
    t && t.getAttribute("data-pdp-stale") && define("c.pdp", 1)
});
define("jqueryPlus", ["jquery", "modernizr", "device", "screen", "requestAnimationFrame", "deviceGroup"], function(n, t, i, r, u, f) {
    function b(n) {
        var t = n.match(a);
        return t && t.length > 2 ? t[2].split(",") : []
    }
    function v(n) {
        var t = b(n), i = t.length;
        return parseFloat(i == 6 ? t[4] : i == 16 ? t[12] : 0)
    }
    function y(n) {
        return parseFloat(n.css("font-size")) || 12
    }
    function k() {
        var u, t, i, n, r;
        for (u in o) {
            t = o[u];
            for (i in t)
                i.indexOf("$") != 0 && (n = t[i], n && (r = (new Date).getTime() - n.animation.startTime, r >= n.animation.options.duration ? w(t, n) : d(t, n, r)))
        }
        p()
    }
    function p() {
        s.length ? (l=!0, u(function() {
            k()
        })) : l=!1
    }
    function d(t, i, r) {
        var h = i.animation.options.easing || "swing", e = n.easing[h], o = i.animation.options.step, u, f, s;
        o && (i = o(i));
        u = i.startValue;
        e && (f = i.animation.options.duration, s = e(r / f, r, 0, 1, f), u = i.startValue + s * (i.endValue - i.startValue) || i.startValue);
        t.$element.css(i.name, u)
    }
    function w(t, i, r) {
        var f = i.name, u;
        r || t.$element.css(f, i.endValue);
        u = i.animation;
        delete o[t.$element.uniqueId()];
        delete u.properties[f];
        delete t[i];
        u.properties.length || (u.options.complete && u.options.complete(), n.inArray(u, s) + 1 && s.splice(n.inArray(u, s), 1))
    }
    var h = t.prefixed("transform") || "transform", e, a, c, o, s, l;
    return (n.fn.innerHeight = function() {
        var n = this[0];
        return n && n.innerHeight ? n.innerHeight : this.height()
    }, e = {}, Number.prototype.mod = function(n) {
        return e[this] || (e[this] = {}), e[this][n] || (e[this][n] = (this%n + n)%n), e[this][n]
    }, t.translateXCss = t.csstransforms3d && i.isCapable("AllowTransform3d") ? "translateX3d" : t.csstransforms && i.isCapable("AllowTransform2d") ? "translateX" : "left", a = new RegExp("matrix(3d)?\\(([0-9-., ]*)\\)", "i"), n.cssHooks.translateX = {
        get: function(t) {
            var i = n(t).css(h);
            return v(i)
        },
        set: function(t, i) {
            n(t).css(h, i === "" ? i : "translateX(" + i + ")")
        }
    }, n.fx.step.translateX = function(t) {
        n.cssHooks.translateX.set(t.elem, t.now + t.unit)
    }, n.cssHooks.translateX3d = {
        get: function(t) {
            var i = n(t).css(h);
            return v(i)
        },
        set: function(t, i) {
            n(t).css(h, i ? "translate3d(" + i + ",0,0)" : "")
        }
    }, n.fx.step.translateX3d = function(t) {
        n.cssHooks.translateX3d.set(t.elem, t.now + t.unit)
    }, n.extend(n.easing, {
        easeOutQuad: function(n, t, i, r, u) {
            return - r * (t/=u) * (t - 2) + i
        },
        easeOutCubic: function(n, t, i, r, u) {
            return r * ((t = t / u - 1) * t * t + 1) + i
        },
        easeInCubic: function(n, t, i, r, u) {
            return r * (t/=u) * t * t + i
        }
    }), n.fn.pxToEm = function(n, t) {
        if (!n)
            return 0;
        var i = y(t || this);
        return parseFloat(n) / i + "em"
    }, n.fn.toPx = function(n, t) {
        var i = 0;
        return n && (i = parseFloat(n), n.indexOf("em")!=-1 && (i*=y(t || this))), i
    }, c = 0, n.fn.uniqueId = function() {
        var t = n(this).data("uniqueId");
        return t || (c++, t = c, n(this).data("uniqueId", t)), t
    }, o = {}, s = [], f.isPc) ? n : (n.fn.animate = function(t, i) {
        for (var f, r, a, v, y = (new Date).getTime(), e = 0; e < this.length; e++) {
            var h = n(this[e]), c = h.uniqueId(), u = o[c];
            u || (u = o[c] = {
                $element: h
            });
            i.step && typeof i.step != "function" && (i.step = null);
            f = {
                properties: t,
                options: i,
                startTime: y
            };
            s.push(f);
            for (r in t)
                u[r] && w(u, u[r], !0, f), a = t[r], v = h.css(r), u[r] = {
                    name: r,
                    startValue: v,
                    endValue: a,
                    animation: f
                }
        }
        return l || p(), n(this)
    }, n.fn.stop = function() {
        return n(this)
    }, n)
});
define("touchEvents", ["jquery", "modernizr", "window"], function(n, t, i) {
    function p() {
        return "ontouchstart"in i
    }
    function a(n) {
        return f(n), !1
    }
    function f(n) {
        n.originalEvent && n.originalEvent.preventManipulation && n.originalEvent.preventManipulation();
        n.preventDefault();
        n.stopPropagation()
    }
    function w(n) {
        var t = n.originalEvent.changedTouches || n.originalEvent.touches;
        return t && t.length ? t : [n]
    }
    function b(n) {
        return n.x && Math.abs(Math.atan(n.y / n.x)) < y ? s : l
    }
    function h(n, t) {
        return n && t ? {
            x: n.x / t,
            y: n.y / t
        } : {
            x: 0,
            y: 0
        }
    }
    function e(n, t) {
        return n && t ? {
            x: n.x - t.x,
            y: n.y - t.y
        } : {
            x: 0,
            y: 0
        }
    }
    function o(t, i, r) {
        n.each(u[i], function(n, u) {
            t.bind(u, function(n) {
                r(n, i)
            })
        })
    }
    function k(t, i) {
        n.fn[t] = function(n, i) {
            return n ? this.bind(t, i, n) : this.trigger(t, i)
        };
        n.event.special[t] = {
            setup: function(t, r) {
                i(n(this), t, r)
            }
        }
    }
    var v = {
        direction: "all",
        preventDefaultOnStart: !0,
        maximumDistanceForClick: 20,
        maximumTimeForClick: 300,
        distanceBeforeDirectionDetection: 0,
        maximumTimeForThrow: 1200,
        minimumLastThrowSpeed: 1,
        maximumTimeForSwipe: 400,
        minimumDistanceForSwipe: 30,
        directionChangeBuffer: 5
    }, y = Math.PI / 4, c = n(i), r = {
        touch: "touch",
        start: "start",
        move: "move",
        stop: "stop",
        cancel: "cancel",
        tap: "tap",
        swipeUp: "swipeUp",
        swipeRight: "swipeRight",
        swipeDown: "swipeDown",
        swipeLeft: "swipeLeft",
        throwUp: "throwUp",
        throwRight: "throwRight",
        throwDown: "throwDown",
        throwLeft: "throwLeft"
    }, s = "horizontal", l = "vertical", u = {
        touch: {
            start: ["touchstart", "touchenter"],
            stop: ["touchend", "touchleave"],
            move: ["touchmove"],
            cancel: ["touchcancel"],
            preventDefaultOnStart: !1,
            mimickBrowserScroll: !1
        },
        mouse: {
            start: ["mousedown"],
            stop: ["mouseup"],
            move: ["mousemove"],
            cancel: ["mouseleave"],
            preventDefaultOnStart: !1,
            mimickBrowserScroll: !1
        }
    };
    return n.each({
        touch: t.touch || p(),
        mouse: !0
    }, function(n, t) {
        return t && (u = u[n]), !t
    }), k(r.touch, function(t, y) {
        function ft() {
            ut = p = k = g = it = nt = 0
        }
        function rt(o, a) {
            var v = w(o);
            u.preventDefaultOnStart && f(o);
            (ut || it && p.totalDistance >= d.maximumDistanceForClick) && (ut=!0, f(o));
            (p || a == r.start) && n.each(v, function(v, y) {
                var w, ot, ct, lt, ut;
                if (et(y), w = {
                    touchType: a,
                    coord: {
                        x: y.pageX,
                        y: y.pageY
                    },
                    screenCoord: {
                        x: y.screenX,
                        y: y.screenY
                    },
                    scrollCoord: {
                        x: c.scrollLeft(),
                        y: c.scrollTop()
                    },
                    count: k ? k.count + 1: 0,
                    time: (new Date).getTime()
                }, a == r.start && (ft(), tt=!1, p = n.extend({
                    totalDistance: 0
                }, w), k = n.extend({}, w)), w.delta = e(w.coord, k.coord), w.screenDelta = e(w.screenCoord, k.screenCoord), w.duration = w.time - k.time, w.speed = h(w.delta, k.duration), p.delta = e(w.coord, p.coord), p.screenDelta = e(w.screenCoord, p.screenCoord), p.totalDistance += Math.sqrt(Math.pow(w.delta.x, 2) + Math.pow(w.delta.y, 2)), ot = d.direction == s, k && k.speed && (ct = ot ? w.delta.x : w.delta.y, lt = ot ? k.delta.x : k.delta.y, ct * lt < 0 && (p.movecoord = w.coord, p.movetime = w.time)), p.movecoord && (p.movechange = e(w.coord, p.movecoord), p.moveduration = w.time - p.movetime, p.movespeed = h(p.movechange, p.moveduration)), p.duration = w.time - p.time, p.speed = h(p.delta, p.duration), g = b(p.delta), p.totalDistance > d.distanceBeforeDirectionDetection && d.direction != "all"&&!it&&!nt && (p.direction = g, d.direction != g ? (nt=!0, w.touchType = r.cancel) : p.totalDistance > d.maximumDistanceForClick && (it=!0, f(o))), w.touchType == r.stop) {
                    var st = (p.movechange || p.delta).x, ht = (p.movechange || p.delta).y, at = p.moveduration || p.duration, rt = g == s, yt = rt ? Math.abs(p.speed.x): Math.abs(p.speed.y), pt = rt ? Math.abs(w.speed.x): Math.abs(w.speed.y), wt = rt ? Math.abs(k.speed.x): Math.abs(k.speed.y), vt = Math.max((pt + wt) / 2, yt);
                    p.duration < d.maximumTimeForClick && p.totalDistance < d.maximumDistanceForClick ? (tt=!0, w.touchType = r.tap) : (w.distance = Math.abs(rt ? st : ht), vt > .3 && (at < d.maximumTimeForSwipe ? w.touchType = rt ? st > 0 ? r.swipeRight : r.swipeLeft : ht > 0 ? r.swipeDown : r.swipeUp : at < d.maximumTimeForThrow && vt > d.minimumLastThrowSpeed && (w.touchType = rt ? st > 0 ? r.throwRight : r.throwLeft : ht > 0 ? r.throwDown : r.throwUp)))
                }(k.touchType != w.touchType || k.coord.x != w.coord.x || k.coord.y != w.coord.y) && (nt && w.touchType != r.cancel || (ut = n.Event(r.touch, {
                    current: w,
                    last: k,
                    start: p,
                    direction: g,
                    originalEvent: o
                }), t.trigger(ut), tt = ut.allowClick!==!1, ut.isDefaultPrevented() && o.preventDefault(), ut.isPropagationStopped() && o.stopPropagation(), ut.isImmediatePropagationStopped() && o.stopImmediatePropagation()));
                nt && u.mimickBrowserScroll && g == l && (p.scrollCoord.y -= w.screenDelta.y, n(i).scrollTop(p.scrollCoord.y));
                a == r.stop || a == r.cancel ? ft() : k = w
            })
        }
        function et(n) {
            n.pageX || n.originalEvent && (n.pageX = n.originalEvent.pageX, n.pageY = n.originalEvent.pageY, n.screenX = n.originalEvent.screenX, n.screenY = n.originalEvent.screenY, n.pointerId = n.originalEvent.pointerId, n.identifier = n.originalEvent.identifier)
        }
        var p, k, g, it, ut, nt, tt=!1, d = n.extend({}, v, y);
        ft();
        t.find("*").addBack().each(function() {
            n(this).attr("draggable", !1).attr("selectable", !1).bind("ondragstart", a).bind("drag", a).not(".mobilead").not(".ip,.ip *").bind("click", function(n) {
                return tt || f(n), tt
            });
            this.onselectstart !== undefined ? this.onselectstart = function() {
                return !1
            } : this.style.MozUserSelect !== undefined && (this.style.MozUserSelect = "none")
        });
        o(t, r.start, rt);
        o(t, r.move, rt);
        o(t, r.stop, rt);
        o(t, r.cancel, rt)
    }), r
});
define("navHover", ["jquery", "jqBehavior", "tabKeyPressed", "pointerEvents", "window"], function(n, t, i, r, u) {
    function f(t, f) {
        function p(t) {
            var i = n(this).closest("li");
            (t.originalEvent.pointerType == 2 || t.originalEvent.pointerType == "touch") && i.children().length > 1 && (l=!0, i.hasClass("hover") ? s() : h(i))
        }
        function w() {
            if (i()) {
                var t = n(this).closest("li");
                t.children().length > 1 ? h(t) : s();
                e.length || (e = t);
                o = e
            }
        }
        function b() {
            v = setTimeout(s, 0);
            a=!1
        }
        function k() {
            clearTimeout(v);
            a=!0
        }
        function s() {
            e.removeClass("hover").attr("aria-hidden", !0)
        }
        function h(n) {
            e.removeClass("hover").attr("aria-hidden", !0);
            e = n.addClass("hover").attr("aria-hidden", !1)
        }
        function d(n) {
            l && (n.preventDefault(), l=!1)
        }
        function g() {
            n(this).find("ul").eq(0).attr("aria-hidden", !1)
        }
        function nt() {
            n(this).find("ul").eq(0).attr("aria-hidden", !0)
        }
        function tt(n) {
            if (a) {
                var i=!0, t;
                switch (n.keyCode) {
                case 27:
                    s();
                    t = e;
                    break;
                case 37:
                    t = e.prev();
                    t.length && h(t);
                    break;
                case 38:
                    o.closest("ul").hasClass("inner") ? (t = o.prev(), t.length || (t = o.parent().closest("li"))) : s();
                    break;
                case 39:
                    t = e.next();
                    t.length && h(t);
                    break;
                case 40:
                    o.closest("ul").hasClass("inner") ? t = o.next() : (t = o.find("li").eq(0), h(e));
                    break;
                default:
                    i=!1
                }
                t && t.length && (o = t, o.find("a").eq(0).focus());
                i && n.preventDefault()
            }
        }
        var e = n(), o = n(), l=!1, v, c = t.find("li:has(ul)"), y = n(u), a=!1;
        return t.find("a").filter(function(n) {
            return n
        }).attr("tabIndex", - 1), {
            setup: function() {
                t.on(r.up, f.mainNavSelector, p).on("focus", f.mainNavSelector, w).on("click", f.mainNavSelector, d).on("focusout", b).on("focusin", k);
                c.on(r.over, g).on(r.out, nt);
                y.on("keydown", tt);
                c.children("a").attr("aria-haspopup", !0)
            },
            teardown: function() {
                t.off(r.up, f.mainNavSelector, p).off("focus", f.mainNavSelector, w).off("click", f.mainNavSelector, d).off("focusout", b).off("focusin", k);
                c.off(r.over, g).off(r.out, nt);
                y.off("keydown", tt);
                c.children("a").attr("aria-haspopup", !1)
            }
        }
    }
    return t(f, {
        mainNavSelector: "ul.outer a:not(ul.inner a)"
    })
});
define("tabKeyPressed", ["jquery"], function(n) {
    var t=!1;
    n(document).on("keydown", function(n) {
        n.keyCode == 9 && (t=!0)
    }).on("keyup", function(n) {
        n.keyCode == 9 && (t=!1)
    });
    return function() {
        return t
    }
});
require(["jquery", "binding", "c.dom"], function(n, t) {
    var i = n("html");
    require(["c.deferred"], function() {
        i.addClass("loaded")
    });
    n(".ad a.adchoices").removeClass("adchoices").addClass("adchoicesjs");
    t("searchTargetSelf", "#srchfrm").view(t.views.SIZE12COLUMN);
    t("searchScope", "#search-scope").view(t.views.SIZE3COLUMN | t.views.SIZE4COLUMN);
    t("truncate", ".truncate").all()
});
define("arrowScroll", ["jquery", "jqBehavior", "imgSrc", "window", "remToPixel", "scrollLeft", "dir.tokens"], function(n, t, i, r, u, f, e) {
    function o(t, r) {
        function b() {
            w = r.reduceScrollAmount ? u(r.reduceScrollAmount) : 0;
            g();
            o.on("click", it);
            s.on("click", rt);
            p();
            a.on("scroll", p)
        }
        function k() {
            o.off("click", it);
            s.off("click", rt);
            a.off("scroll", p)
        }
        function ut() {
            k();
            b()
        }
        function p() {
            if (g(), !l) {
                et();
                return 
            }
            v === 0 ? (e.ltr ? tt : nt)() : v > 0 && v < l ? ot() : v === l && (e.ltr ? nt : tt)()
        }
        function d(n) {
            a.animate({
                scrollLeft: f.adjustValue(a[0], ft(n))
            }, {
                queue: !1,
                duration: 220,
                complete: function() {
                    p()
                }
            })
        }
        function g() {
            var n = a[0];
            n && (y = n.clientWidth, v = f(n), l = n.scrollWidth - y - r.scrollOffset, i.loadInViewport && i.loadInViewport(t[0]))
        }
        function ft(n) {
            return e.ltr || (n*=-1), n==-1 ? h > 0 && (h -= y - w + r.scrollOffset, h < 0 && (h = 0)) : n == 1 && h < l && (h += y - w + r.scrollOffset, h > l && (h = l)), h
        }
        function et() {
            o.removeClass(c);
            s.removeClass(c)
        }
        function ot() {
            o.addClass(c);
            s.addClass(c)
        }
        function nt() {
            s.removeClass(c);
            o.addClass(c)
        }
        function tt() {
            o.removeClass(c);
            s.addClass(c)
        }
        function it() {
            d( - 1)
        }
        function rt() {
            d(1)
        }
        function st() {
            o = n('<button class="stripearrow"><\/button>');
            s = n('<button class="stripearrow"><\/button>');
            a.before(o).after(s)
        }
        var o, s, a = r.scrollRegion ? t.find(r.scrollRegion): t.children("div");
        r.addArrowButtons ? st() : (o = t.find("button").first(), s = t.find("button").last());
        var v = 0, l = 0, c = "show", h = 0, y, w;
        return {
            setup: b,
            update: ut,
            teardown: k
        }
    }
    var s = n(r);
    return t(o, {
        addArrowButtons: !1,
        reduceScrollAmountForArrows: !1,
        scrollOffset: 0
    })
});
define("meFacebook", ["jquery", "jqBehavior", "navigation", "meModule", "meModule.Tokens", "window", "headData", "getCookie", "setCookie", "mediator"], function(n, t, i, r, u, f, e, o, s, h) {
    function it(t) {
        var i = {
            notificationCounts: {
                messages: 0,
                notifications: 0
            },
            init: function(t, r) {
                n("head").prepend("<style id='fb-animation-fix'><\/style>");
                f.fbAsyncInit = function() {
                    f.FB.init({
                        appId: t,
                        xfbml: !1,
                        cookie: !0,
                        version: "v2.0"
                    });
                    i.ready()
                }, function(n, t, i) {
                    var u, f = n.getElementsByTagName(t)[0];
                    n.getElementById(i) || (u = n.createElement(t), u.id = i, u.src = "//connect.facebook.net/" + r + "/sdk.js", f.parentNode.insertBefore(u, f))
                }(document, "script", "facebook-jssdk");
                i.bindEvents()
            },
            bindEvents: function() {
                t.on("click", g, function() {
                    t.find(".meflyout").hasClass(r.classNames.signout) && i.login()
                });
                t.on("click", k, function() {
                    i.notificationCounts.messages = 0;
                    i.deferUpdateNotifications()
                });
                t.on("click", d, function() {
                    i.notificationCounts.notifications = 0;
                    i.deferUpdateNotifications()
                })
            },
            ready: function() {
                if (f.FB) {
                    var n = o(p);
                    if (n) {
                        i.logout();
                        s(p, 0, - 1);
                        return 
                    }
                    f.FB.getLoginStatus(function(n) {
                        n && n.status === "connected" && r.deferredFlyoutOnHover(t, "FacebookFlyoutVisible", i.getNews)
                    })
                }
            },
            getNews: function() {
                h.unsub("FacebookFlyoutVisible", i.getNews);
                r.updateFlyout(t, "facebook", {
                    showProgress: !0
                }, !0, y, i.logout);
                i.notificationCounts.messages = 0;
                i.notificationCounts.notifications = 0;
                i.updateNotifications();
                n.ajax({
                    url: a,
                    cache: !1,
                    timeout: tt
                }).fail(function(n, t) {
                    r.logAjaxError(t, "Facebook", "get news");
                    i.updateFlyout({
                        signedIn: !0,
                        error: !0
                    })
                }).done(function(n) {
                    i.updateFlyout({
                        result: n,
                        signedIn: !0,
                        error: !1
                    })
                })
            },
            updateFlyout: function(u) {
                var f = {}, s = u.signedIn, h = u.result, e, o, c;
                i.notificationCounts.notifications = 0;
                i.notificationCounts.messages = 0;
                s ? u.error ? f.showError=!0 : (e = h && h.trim(), e ? (o = n(e), i.notificationCounts.notifications = o.data("notifications-count") || 0, i.notificationCounts.messages = o.data("messages-count") || 0, c = o.find("li"), c.length ? f.content = e : f.showNoContent=!0) : f.showNoContent=!0) : f.signedOutImageUrl = l;
                r.updateFlyout(t, "facebook", f, s, y, i.logout);
                i.updateNotifications()
            },
            deferUpdateNotifications: function() {
                f.setTimeout(function() {
                    i && i.updateNotifications()
                }, 2e3)
            },
            updateNotifications: function() {
                var n = i.notificationCounts.messages, u = i.notificationCounts.notifications, f = n + u, e = f && f > 0, o = [{
                    index: 0,
                    count: n,
                    displayNotification: n > 0
                }, {
                    index: 1,
                    count: u,
                    displayNotification: u > 0
                }
                ];
                r.updateNotifications(t, ".facebook", o, c, {
                    maxNotifications: nt,
                    count: f,
                    updateInTile: !0,
                    updateInTask: !0,
                    displayNotification: e
                })
            },
            login: function() {
                f.FB && f.FB.login(function(n) {
                    n.authResponse && i.getNews()
                }, {
                    scope: b
                })
            },
            logout: function(n) {
                n && n.preventDefault();
                try {
                    f.FB.getLoginStatus(function(n) {
                        n.status === "connected" ? f.FB.logout(function() {
                            i.updateFlyout({
                                signedIn: !1
                            })
                        }) : i.updateFlyout({
                            signedIn: !1
                        })
                    }, !0)
                } catch (t) {
                    i.updateFlyout({
                        signedIn: !1
                    })
                }
            }
        };
        return {
            setup: function() {
                var o, n, u;
                if (e && e.clientSettings && e.clientSettings.fbid) {
                    o = e.clientSettings.fbid;
                    n = "en_US";
                    l || (l = t.find(r.selector.contentDiv).data("signedout-image-url"));
                    try {
                        u = r.info.locale.split("-");
                        n = u[0] + "_" + u[1].toUpperCase()
                    } catch (s) {}
                    i.init(o, n)
                } else 
                    f.console && f.console.warn && f.console.warn("Missing fbid in data-client-settings. Can't initialize the Facebook module for this origin: " + f.location.origin)
            },
            facebook: i
        }
    }
    var w = r.info.origin + "/" + r.info.locale, a = w + "/homepage/Facebook/Data", v = location.href.split("?")[1];
    typeof v != "undefined" && (a += "?" + v);
    var b = "read_stream,manage_notifications,read_mailbox", c = u.tile.facebook[1], k = r.selector.tasks[0], d = r.selector.tasks[1], y = r.selector.tasks[2], g = r.selector.flyoutFooter, nt = c.maxNotifications, tt = c.requestTimeout, l, p = "FBForceSignout";
    return t(it)
});
define("meTwitter", ["jquery", "jqBehavior", "navigation", "meModule", "meModule.Tokens", "getCookie", "window", "track", "mediator"], function(n, t, i, r, u, f, e, o, s) {
    function k(t, i) {
        function c(n, f) {
            var o = n.signedIn && f !== "USER_AUTH_FAILED", s = n.error, e = {}, h;
            o ? (h = u.tile.twitter[1], s ? e.showError=!0 : f && f.trim() ? e.content = f : e.showNoContent=!0) : e.signedOutImageUrl = v;
            r.updateFlyout(t, "twitter", e, o, r.selector.tasks[2], a);
            o ? b() : r.attachSignInPopupWindow(t, l, i.popupWidth, i.popupHeight)
        }
        function a(t) {
            t.preventDefault();
            n.get(p).always(c.bind(null, {
                signedIn: !1,
                error: !1
            }))
        }
        function b() {
            var i, u, f;
            t.find(r.selector.contentItem).off(h).on(h, function(t) {
                var i = n(this).data("landingurl"), r = t.target || t.srcElement;
                i && r && r.tagName.toLowerCase() === "p" && (e.open(i, "_blank"), o.trackEvent({
                    type: "click",
                    target: this
                }))
            });
            t.find(".tweetactions a").off(h).on(h, function(n) {
                var t = n.target || n.srcElement;
                t && t.href && (n.preventDefault(), k(t.href))
            });
            i = t.find(r.selector.contentList);
            u = i.length ? i.data("meurl") : null;
            u && (f = t.find(r.selector.tasks[1]), f.length && (f[0].href = u))
        }
        function k(n) {
            var i = 550, t = 420, r = screen.height, u = screen.width, f = Math.round(u / 2 - i / 2), o = r > t ? Math.round(r / 2 - t / 2): 0;
            e.open(n, "intent", "scrollbars=yes,resizable=yes,toolbar=no,location=yes,width=" + i + ",height=" + t + ",left=" + f + ",top=" + o)
        }
        return e.populateTweets = function() {
            s.unsub("TwitterFlyoutVisible", e.populateTweets);
            r.updateFlyout(t, "twitter", {
                showProgress: !0
            }, !0, r.selector.tasks[1], a);
            n.ajax({
                url: y,
                type: "GET",
                cache: !1,
                timeout: i.timeout
            }).fail(function(n, t) {
                r.logAjaxError(t, "Twitter", "get tweets");
                c({
                    signedIn: !0,
                    error: !0
                })
            }).done(c.bind(null, {
                signedIn: !0,
                error: !1
            }))
        }, {
            setup: function() {
                v || (v = t.find(r.selector.contentDiv).data("signedout-image-url"));
                f(w) ? r.deferredFlyoutOnHover(t, "TwitterFlyoutVisible", e.populateTweets) : r.attachSignInPopupWindow(t, l, i.popupWidth, i.popupHeight)
            }
        }
    }
    var b = {
        popupWidth: 500,
        popupHeight: 500,
        timeout: u.tile.twitter[1].requestTimeout
    }, h = "click", c = r.info.origin + "/" + r.info.locale + "/homepage/", y = c + "Twitter/Tweets", l = c + "Twitter/Authorize", a = location.href.split("?")[1], p, w, v;
    return typeof a != "undefined" && (y += "?" + a, l += "?" + a), p = c + "Twitter/SignOut", w = "TwitterSignedIn", t(k, b)
});
define("meYahooMail", ["jquery", "jqBehavior", "meModule", "meModule.Tokens", "getCookie", "window", "setCookie", "mediator"], function(n, t, i, r, u, f, e, o) {
    function nt(t, c) {
        function nt(r, u) {
            var o = r.signedIn, b = r.error, f = {}, e = "", h, a, w;
            o ? b ? f.showError=!0 : u && u.trim() && (h = n(u.trim()), e = h.attr("data-unre-msg-count"), n("li", h).length ? f.content = u : f.showNoContent=!0) : f.signedOutImageUrl = l;
            i.updateFlyout(t, "yahoomail", f, o, p, g);
            a = e && e.trim() > 0;
            i.updateNotifications(t, ".yahoomail", 0, s, {
                maxNotifications: c.maxNotifications,
                count: e,
                updateInTile: !0,
                updateInTask: !1,
                displayNotification: a
            });
            o || (w = t.find(i.selector.notification), w.text(y), i.attachSignInPopupWindow(t, v, c.popupWidth, c.popupHeight))
        }
        function tt() {
            g();
            e(a, 0, - 1);
            o.unsubChannel("SignOutEvent", h, null)
        }
        function g(i) {
            i && i.preventDefault();
            n.get(b).always(nt.bind(null, {
                signedIn: !1,
                error: !1
            }));
            n("<iframe/>").attr("id", "yahoologout").attr("src", k).attr("style", "display: none").load(function() {
                t.find("#yahoologout").remove()
            }).appendTo(t)
        }
        return f.populateYahooMailInbox = function() {
            o.unsub("YahooFlyoutVisible", f.populateYahooMailInbox);
            i.updateFlyout(t, "yahoomail", {
                showProgress: !0
            }, !0, p, g);
            i.updateNotifications(t, ".yahoomail", 0, s, {
                maxNotifications: c.maxNotifications,
                count: - 1,
                updateInTile: !0,
                updateInTask: !1,
                displayNotification: !1
            });
            n.ajax({
                url: w,
                type: "GET",
                cache: !1,
                timeout: c.timeout
            }).fail(function(n, t) {
                i.logAjaxError(t, "YahooMail", "get email");
                nt({
                    signedIn: !0,
                    error: !0
                })
            }).done(nt.bind(null, {
                signedIn: !0,
                error: !1
            }))
        }, {
            setup: function() {
                var n, s;
                if (l || (l = t.find(i.selector.contentDiv).data("signedout-image-url")), n = u(a), n) {
                    g();
                    e(a, 0, - 1);
                    return 
                }
                u(d) ? i.deferredFlyoutOnHover(t, "YahooFlyoutVisible", f.populateYahooMailInbox) : (s = t.find(i.selector.notification), s.text(y), i.attachSignInPopupWindow(t, v, c.popupWidth, c.popupHeight));
                h = i.getMeModuleId(t);
                h && o.subChannel(r.tile.mailProviderOptions[0], h, tt)
            }
        }
    }
    var c = i.info.origin + "/" + i.info.locale + "/homepage/", w = c + "YahooMail/Emails", v = c + "YahooMail/Authorize", b = c + "YahooMail/SignOut", k = "https://login.yahoo.com/config/login?logout=1", d = "YahooMailSignedIn", s = r.tile.yahoomail[1], y = r.tile.yahoomail[0].notification, p = i.selector.tasks[2], l, a = "YahooForceSignout", h, g = {
        popupWidth: 600,
        popupHeight: 500,
        maxNotifications: s.maxNotifications,
        timeout: s.requestTimeout
    };
    return t(nt, g)
});
define("meGmail", ["jquery", "jqBehavior", "meModule", "meModule.Tokens", "getCookie", "window", "setCookie", "mediator"], function(n, t, i, r, u, f, e, o) {
    function g(t, s) {
        function g(u, f) {
            var c = f.signedIn, k = f.error, e = {}, a = 0, g, o, p, w, b;
            c ? (g = r.tile.gmail[1], k ? e.showError=!0 : (o = u && u.trim(), o ? (p = n(o), a = p.data("unread-count"), w = p.find("li"), w.length ? e.content = o : e.showNoContent=!0) : e.showNoContent=!0)) : e.signedOutImageUrl = l;
            b = a > 0;
            i.updateNotifications(t, ".gmail", 0, h, {
                maxNotifications: s.maxNotifications,
                count: a,
                updateInTile: !0,
                updateInTask: !1,
                displayNotification: b
            });
            i.updateFlyout(t, "gmail", e, c, y, d);
            c || i.attachSignInPopupWindow(t, v, s.popupWidth, s.popupHeight)
        }
        function nt() {
            d();
            e(a, 0, - 1);
            o.unsubChannel("SignOutEvent", c, null)
        }
        function d(i) {
            i && i.preventDefault();
            n.get(w).always(function() {
                g(null, {
                    signedIn: !1,
                    error: !1
                })
            });
            n("<iframe/>").attr("id", "logoutframe").attr("src", b).attr("style", "display: none").load(function() {
                t.find("#logoutframe").remove()
            }).appendTo(t)
        }
        return f.populateGmail = function() {
            o.unsub("GmailFlyoutVisible", f.populateGmail);
            i.updateFlyout(t, "gmail", {
                showProgress: !0
            }, !0, y, d);
            i.updateNotifications(t, ".gmail", 0, h, {
                maxNotifications: s.maxNotifications,
                count: - 1,
                updateInTile: !0,
                updateInTask: !1,
                displayNotification: !1
            });
            n.ajax({
                url: p,
                type: "GET",
                cache: !1,
                timeout: s.timeout
            }).fail(function(n, t) {
                i.logAjaxError(t, "Gmail", "get emails");
                g(!1, {
                    signedIn: !0,
                    error: !0
                })
            }).done(function(n) {
                g(n, {
                    signedIn: !0,
                    error: !1
                })
            })
        }, {
            setup: function() {
                l || (l = t.find(i.selector.contentDiv).data("signedout-image-url"));
                var n = u(a);
                if (n) {
                    d();
                    e(a, 0, - 1);
                    return 
                }
                u(k) ? i.deferredFlyoutOnHover(t, "GmailFlyoutVisible", f.populateGmail) : i.attachSignInPopupWindow(t, v, s.popupWidth, s.popupHeight);
                c = i.getMeModuleId(t);
                c && o.subChannel(r.tile.mailProviderOptions[0], c, nt)
            }
        }
    }
    var s = i.info.origin + "/" + i.info.locale + "/homepage/", v = s + "Gmail/Authorize", p = s + "Gmail/Emails", nt = s + "Gmail/AppId", w = s + "Gmail/SignOut", b = "https://accounts.google.com/logout", k = "GmailSignedIn", h = r.tile.gmail[1], y = i.selector.tasks[2], l, a = "GmailForceSignout", c, d = {
        popupWidth: 600,
        popupHeight: 640,
        maxNotifications: h.maxNotifications,
        timeout: h.timeout
    };
    return t(g, d)
});
define("meOffice", ["jquery", "jqBehavior", "mediator"], function(n, t, i) {
    function r(t) {
        function r() {
            var i = t.find("section li time");
            i.each(function() {
                var t = new Date(n(this).attr("datetime"));
                t && n(this).html(t.toLocaleString())
            })
        }
        return {
            setup: function() {
                var n, u;
                r();
                n = t.find("[data-module-id]");
                n.length && (u = n.data("moduleId"), u && i.sub("moduleRefreshed-" + u, r))
            },
            teardown: function() {
                var u = t.find("[data-module-id]"), n;
                u.length && (n = u.data("moduleId"), n && i.unsub("moduleRefreshed-" + n, r))
            }
        }
    }
    return t(r)
});
define("loadMeImages", ["jquery", "jqBehavior", "imgSrc"], function(n, t, i) {
    function r(t, r) {
        function f() {
            e.each(function() {
                i.go(this)
            })
        }
        var e = n(r.imagesToLoad), o = navigator.pointerEnabled || navigator.msPointerEnabled ? "pointermove MSPointerMove": "mousemove", u = o + " touchstart MSPointerDown pointerdown";
        return {
            setup: function() {
                t.one(u, f)
            },
            teardown: function() {
                t.off(u, f)
            }
        }
    }
    return t(r)
});
define("outlookInbox", ["jquery", "jqBehavior", "outlookModel", "outlookInbox.tokens", "format", "headData", "meModule", "meModule.Tokens"], function(n, t, i, r, u, f, e, o) {
    var h = o.tile.mail[1], s = function(t, r) {
        function c(n) {
            u && a(n);
            l(n)
        }
        function l(n) {
            var i = n.unre;
            e.updateNotifications(t, ".mail", 0, h, {
                count: i,
                updateInTile: !0,
                updateInTask: !1,
                displayNotification: i > 0
            })
        }
        function a(n) {
            var i = n && n.msgs ? v(n.msgs): null;
            e.updateFlyout(t, "mail", {
                content: i
            }, !0)
        }
        function v(t) {
            var i = "", r = o.tile.mail[1].piiText, u = o.tile.mail[1].piiUrl;
            return n.each(t, function(n, t) {
                var f = new Date(t.date), e = "<li " + p(t) + '><a href="' + t.murl + '" target="_blank" data-piitxt="' + r + '" piiurl="' + u + '"><span class="from">' + t.name + "<\/span><p>" + t.subj + '<\/p><time datetime="' + f.toISOString() + '">' + f.toLocaleString() + "<\/time><\/a><\/li>";
                i += e
            }), i && (i = "<ul>" + i + "<\/ul>"), i
        }
        function y() {
            u && e.updateFlyout(t, "mail", {
                showError: !0
            }, !0)
        }
        function p(n) {
            return n.read ? "" : 'class="unread"'
        }
        if (!f.signedin) {
            require(["c.sso"], s.bind(null, t, r));
            return 
        }
        var u = i.isPcDeviceGroup(), w = n(e.selector.contentDiv, t);
        e.updateFlyout(t, "mail", {
            showProgress: !0
        }, !0);
        i.getOutlookData().done(c, y)
    };
    return t(s, {})
});
define("meModule", ["jquery", "meModule.Tokens", "headData", "window", "logging", "perfPing", "mediator"], function(n, t, i, r, u, f, e) {
    function a(n, t, i, r, u, f) {
        var e = n.find(f.tasks[i]), o = e.find("span");
        r ? (e.addClass(c), o.text((t.notifications || l).replace("{0}", u))) : (e.removeClass(c), o.empty())
    }
    function y(n, t, i, r, u) {
        n.find(u.notification).text(i ? (t.notifications || l).replace("{0}", r) : t.notificationEmpty || "")
    }
    function p(n, t, i, u) {
        var f = (h.width - t) / 2, e = (h.height - i - 100) / 2, o = "status=0,toolbar=0,location=0,resizable=1,scrollbars=1,left=" + f + ",top=" + e + ",width=" + t + ",height=" + i;
        u.preventDefault();
        r.open(n, "Login", o)
    }
    function w(t, i) {
        if (i && i.taskLinks && i.taskLinks.trim()) {
            var r = i.taskLinks.split(";");
            r.length && (t.push("<ul>"), n.each(r, function(n, i) {
                var r = i.split("|"), u;
                r[0] && r[1] && (u = r[0], t.push("<li><a target='_blank' data-piitxt='", u, "' href='", r[1], "'>", u, "<span><\/span><\/a><\/li>"))
            }), t.push("<\/ul>"))
        }
    }
    var h = r.screen, o = "click", c = "notifications", s = ".popUpWindow", l = "({0})", v = "{0}+";
    return {
        info: {
            locale: n("html").attr("lang"),
            origin: r.location.protocol + "//" + r.location.host
        },
        logAjaxError: function(n, t, i) {
            n != "abort" && (f.setMarker('meModule service error: "' + t + '" call failed with status of "' + n + '", ' + i), f.getPayLoad(u.error))
        },
        updateFlyout: function(r, u, f, e, h, c) {
            var a = "", v = typeof e != "undefined" ? e: i.signedin, k = t.tile[u][v ? 1: 0], l = {};
            n.extend(!0, l, k, f);
            a = ["<section>"];
            l.headerUrl && a.push("<a href='", l.headerUrl, "' target='_blank'><h3>", l.header, "<\/h3><\/a>");
            a.push("<div>");
            l.showProgress ? a.push("<p class='meloading'>", l.content || "", "<\/p>") : l.showError ? a.push("<p class='meerror'>", l.errorMessage || "", "<\/p>") : l.showNoContent ? a.push("<p class='meerror'>", l.noContent || "", "<\/p>") : v ? a.push(l.content.length ? l.content : "<p class='meerror'>" + l.noContent + "<\/p>") : (l.signedOutImageUrl && a.push("<img src='", l.signedOutImageUrl, "' />"), a.push("<p>", l.content.length ? l.content : l.ErrorMessage, "<\/p>"));
            a.push("<\/div>");
            !v && l.footerText && a.push("<a href='", l.footerUrl, "' class='mefoot'>", l.footerText, "<\/a>");
            a.push("<\/section>");
            w(a, l);
            var y = r.find(this.selector.flyout).html(a.join("")), p = this.classNames.signout, b = this.classNames.signin;
            if (v) {
                if (r.find(this.selector.flyoutFooter).off(s), typeof h == "string" && (h = r.find(h)), h && h.length && c && n.isFunction(c))
                    h.off(o).on(o, c);
                n(y).removeClass(p).addClass(b)
            } else 
                n(y).removeClass(b).addClass(p)
        },
        attachSignInPopupWindow: function(n, t, i, r) {
            n.find(this.selector.flyoutFooter).off(s).on(o + s, p.bind(null, t, i, r))
        },
        selector: {
            flyout: ".meflyout",
            flyouts: ".meflyouts",
            notification: ">a>h3>span",
            flyoutHeader: ".meflyout>section>h3",
            flyoutFooter: ".meflyout>section>.mefoot",
            contentDiv: ".meflyout>section>div",
            contentItem: ".meflyout>section>div p",
            contentList: ".meflyout>section>div ul",
            taskLinks: ".meflyout>ul>li>a",
            tasks: [".meflyout>ul>li>a:eq(0)", ".meflyout>ul>li>a:eq(1)", ".meflyout>ul>li>a:eq(2)", ".meflyout>ul>li>a:eq(3)"]
        },
        classNames: {
            signout: "signout",
            signin: "signin"
        },
        updateNotifications: function(t, i, r, u, f) {
            var s = t.closest(".stripeouter").find(".mestripe " + i), o = this.selector, e = f.count;
            f.maxNotifications && e > f.maxNotifications && (e = (u.notificationsTooMany || v).replace("{0}", f.maxNotifications));
            f.updateInTile && y(s, u, f.displayNotification, e, o);
            f.updateInTask && (n.isNumeric(r) ? a(t, u, r, f.displayNotification, e, o) : n(r).each(function(n, i) {
                a(t, u, i.index, i.displayNotification, i.count, o)
            }))
        },
        ssoAutoRefresh: function() {
            var i = this, r = n(i.selector.flyouts);
            n.each(t.tile, function(n, u) {
                var f, o, s, e;
                u[0].ssoAutoRefresh && (f = r.children("." + n), o = f.find(i.selector.flyout).hasClass(i.classNames.signout), f.length && o && (s = t.tile[n][1], e = f.find("[data-module-id]"), e.html(s.content), i.updateFlyout(f, n, {
                    content: e.length ? e[0].outerHTML: " "
                }, !0, null, null)))
            })
        },
        deferredFlyoutOnHover: function(n, t, i) {
            var r = n.data("deferred");
            r ? e.sub(t, i) : i()
        },
        getMeModuleId: function(t) {
            var i = n(".meflyouts .meflyoutcontainer").index(t);
            return n(".mestripe ul li").eq(i).attr("data-module-id")
        }
    }
});
define("outlookModel", ["jquery", "getCookie", "outlookInbox.tokens", "promise", "headData", "window", "logging"], function(n, t, i, r, u, f, e) {
    function nt() {
        return t(s.canaryCookieName) || ""
    }
    function w() {
        return u.dg ? u.dg.indexOf(s.pcDgIdent)!=-1 : !1
    }
    function tt() {
        return h || (h = setTimeout(it, s.timeoutInMs)), o || (o = new r(k)), o
    }
    function it() {
        f[l] = null;
        o && (o.cancel(), o = null);
        b("outlookModel.js: outlook preview api request timeout")
    }
    function b(n) {
        e.error(n)
    }
    function k(t, i) {
        a && v || (a = t, v = i);
        c = rt();
        n.ajax({
            type: "GET",
            url: c,
            jsonp: "cb",
            jsonpCallback: l,
            contentType: "application/json",
            dataType: "jsonp"
        })
    }
    function rt() {
        var i = w() ? s.numOfMsgForPc: s.numOfMsgForMobile, r = {
            num: i,
            nocache: + new Date,
            mac: f.decodeURIComponent(nt())
        }, t = d;
        return document.location.hostname.indexOf("betaplace") >= 0 && (t = g), c = t + "?" + n.param(r)
    }
    function ut(n) {
        if (!o) {
            y();
            return 
        }
        n && n.err&&!p && n.err.client && n.err.client.invalidmac ? (p=!0, k()) : n && n.err ? (y(), v(n), o = null, b("outlookModel.js: error response from outlook preview mail api \r\n" + JSON.stringify(n.err))) : (y(), a(n), o = null)
    }
    function y() {
        f[l] = null;
        h && clearTimeout(h)
    }
    var s = {
        timeoutInMs: 1e4,
        canaryCookieName: "WLMMAC",
        numOfMsgForPc: 3,
        numOfMsgForMobile: 0,
        pcDgIdent: ".pc",
        globalFnName: "setOutlookData"
    }, o, d = i.outlookPreviewApiUrl, g = i.outlookPreviewApiUrlBetaplace, h, c, p=!1, a, v, l = s.globalFnName + Math.floor(Math.random() * 1e4 + 1) + + new Date;
    return f[l] = ut, {
        isPcDeviceGroup: w,
        getOutlookData: tt
    }
});
define("meModule.Tokens", {
    tile: {
        mail: [{
            header: "Outlook.com",
            content: "Outlook.com is free personal email from Microsoft. Share, organise and stay up to date like never before.",
            footerText: "Sign in",
            footerUrl: "[[signin]]",
            taskLinks: "",
            appendAddMoreTask: !0
        }, {
            header: "Inbox",
            headerUrl: "https://mail.live.com/default.aspx",
            content: "Loading...",
            noContent: "Wow, you have a very clean inbox!",
            errorMessage: "Couldn't connect to Outlook.com",
            taskLinks: "Compose|https://mail.live.com/default.aspx?rru=compose;Calendar|https://calendar.live.com/calendar/calendar.aspx",
            appendAddMoreTask: !0,
            piiText: "Read Outlook Email",
            piiUrl: "http://www.hotmail.msn.com/pii/ReadOutlookEmail/"
        }
        ],
        office: [{
            header: "Office",
            content: "See your recent documents, or start one for free with Office Online.",
            footerText: "Sign in",
            footerUrl: "[[signin]]",
            ssoAutoRefresh: !0,
            taskLinks: "Word Online|https://office.live.com/start/Word.aspx?ui=en%2DUS;Excel Online|https://office.live.com/start/Excel.aspx?ui=en%2DUS;PowerPoint Online|https://office.live.com/start/PowerPoint.aspx?ui=en%2DUS;OneNote Online|https://www.onenote.com/notebooks?WT.mc_id=MSN_OneNote_TopMenu"
        }, {
            header: "Recent documents",
            headerUrl: "https://onedrive.live.com/#qt=mru",
            content: "Loading...",
            noContent: "This folder is empty. Click below to create a new document.",
            errorMessage: "Couldn't connect to Office Online",
            errorFooterText: "Go to Office Online",
            taskLinks: "Word Online|https://office.live.com/start/Word.aspx?ui=en%2DUS;Excel Online|https://office.live.com/start/Excel.aspx?ui=en%2DUS;PowerPoint Online|https://office.live.com/start/PowerPoint.aspx?ui=en%2DUS;OneNote Online|https://www.onenote.com/notebooks?WT.mc_id=MSN_OneNote_TopMenu"
        }
        ],
        onenote: [{
            header: "OneNote",
            content: "Sign in to your Microsoft account to see recent notes.",
            footerText: "Sign in",
            footerUrl: "[[signin]]",
            ssoAutoRefresh: !0,
            taskLinks: "Quick Note|https://www.onenote.com/notebooks?WT.mc_id=MSN_OneNote_QuickNote"
        }, {
            header: "Recent Notebooks",
            headerUrl: "https://www.onenote.com/notebooks?WT.mc_id=MSN_OneNote_Recent",
            content: "Loading...",
            noContent: "This folder is empty. Click below to create new notebook.",
            errorMessage: "Couldn't connect to OneNote Online",
            errorFooterText: "Go to OneNote Online",
            taskLinks: "Quick Note|https://www.onenote.com/notebooks?WT.mc_id=MSN_OneNote_QuickNote"
        }
        ],
        onedrive: [{
            header: "OneDrive",
            content: "Sign in to your Microsoft account to see recent OneDrive contents.",
            footerText: "Sign in",
            footerUrl: "[[signin]]",
            ssoAutoRefresh: !0,
            taskLinks: "Files|https://onedrive.live.com;Photos|https://onedrive.live.com/?qt=allmyphotos;Recent documents|https://onedrive.live.com/?qt=mru;Download the OneDrive app|https://onedrive.live.com/about/en/download/"
        }, {
            header: "OneDrive",
            headerUrl: "https://onedrive.live.com?wt.mc_id=oo_msn_msnhomepage_header",
            content: "Loading...",
            errorMessage: "Couldn't connect to OneDrive",
            errorFooterText: "Go to OneDrive",
            taskLinks: "Files|https://onedrive.live.com;Photos|https://onedrive.live.com/?qt=mru;Recent documents|https://onedrive.live.com;Download the OneDrive app|https://onedrive.live.com/about/en/download/"
        }
        ],
        finance: [{
            header: "",
            content: "",
            footerText: "",
            footerUrl: "",
            taskLinks: ""
        }, {
            header: "",
            headerUrl: "",
            content: "",
            errorMessage: "",
            taskLinks: ""
        }
        ],
        sports: [{
            header: "Scores - Live",
            content: "",
            footerText: "All scores",
            footerUrl: "/en-us/sports",
            taskLinks: ""
        }, {
            header: "Scores - Live",
            headerUrl: "/en-us/sports",
            content: "",
            errorMessage: "",
            taskLinks: ""
        }
        ],
        maps: [{
            header: "Maps",
            headerUrl: "",
            content: "",
            footerText: "",
            footerUrl: "",
            taskLinks: "me_maps_directions|http://www.bing.com/maps/directions;me_maps_traffic|http://www.bing.com/maps/"
        }, {
            header: "Maps",
            headerUrl: "",
            content: "",
            errorMessage: "",
            taskLinks: "me_maps_directions|http://www.bing.com/maps/directions;me_maps_traffic|http://www.bing.com/maps/"
        }
        ],
        skype: [{
            header: "Skype keeps the world talking",
            headerUrl: "",
            content: "Use Skype to call, see, message and share with others—wherever they are.",
            footerText: "Go to Skype.com",
            footerUrl: "http://www.skype.com",
            taskLinks: "me_skype_download|http://www.skype.com/download-skype"
        }, {
            header: "Group video calling",
            headerUrl: "",
            content: "Now it's free to get the whole family together on the same video call.",
            errorMessage: "",
            taskLinks: ""
        }
        ],
        facebook: [{
            header: "Facebook",
            content: "Get your latest feed from Facebook",
            errorMessage: "Couldn’t connect to Facebook",
            footerText: "Sign in",
            footerUrl: "#",
            taskLinks: ""
        }, {
            header: "News feed",
            headerUrl: "http://www.facebook.com",
            content: "Loading...",
            noContent: "Your feed is currently empty",
            errorMessage: "Couldn’t connect to Facebook",
            errorFooterText: "Go to Facebook",
            maxNotifications: "99",
            requestTimeout: "10000",
            taskLinks: "Messages|https://www.facebook.com/messages;Notifications|https://www.facebook.com/notifications;Sign out|#"
        }
        ],
        twitter: [{
            header: "Twitter",
            content: "Get your Twitter Updates",
            footerText: "Sign In",
            footerUrl: "https://twitter.com",
            requestTimeout: "10000",
            taskLinks: ""
        }, {
            header: "Tweets",
            headerUrl: "https://twitter.com",
            content: "Loading...",
            noContent: "Your timeline is currently empty",
            errorMessage: "Couldn't connect to Twitter",
            errorFooterText: "Go to Twitter",
            taskLinks: "Notifications|https://twitter.com/i/notifications;Me|#;Sign out|#"
        }
        ],
        xbox: [{
            header: "Spotlight",
            content: "",
            footerText: "View All",
            footerUrl: "",
            taskLinks: "me_xbox_taskLinks_Explore|http://music.xbox.com/explore;me_xbox_taskLinks_Radio|http://music.xbox.com/radio"
        }, {
            header: "My amazing playlist",
            headerUrl: "https://music.xbox.com",
            content: "",
            errorMessage: "",
            taskLinks: "me_xbox_taskLinks_Explore|http://music.xbox.com/explore;me_xbox_taskLinks_Radio|http://music.xbox.com/radio"
        }
        ],
        stores: [{
            header: "Microsoft Store",
            content: "",
            footerText: "Sign in",
            footerUrl: "",
            taskLinks: "me_stores_taskLinks_store|http://www.microsoftstore.com/;me_stores_taskLinks_support|http://answerdesk.microsoftstore.com/;me_stores_taskLinks_apps|http://windows.microsoft.com/{locale}/windows-8/apps;me_stores_taskLinks_findastore|http://www.microsoftstore.com/Locations"
        }, {
            header: "Featured App",
            headerUrl: "",
            content: "",
            errorMessage: "",
            taskLinks: "me_stores_taskLinks_store|http://www.microsoftstore.com/;me_stores_taskLinks_support|http://answerdesk.microsoftstore.com/;me_stores_taskLinks_apps|http://windows.microsoft.com/{locale}/windows-8/apps;me_stores_taskLinks_findastore|http://www.microsoftstore.com/Locations"
        }
        ],
        yahoomail: [{
            header: "Yahoo Mail",
            content: "Sign in to view your emails",
            footerText: "Sign in",
            footerUrl: "https://mail.yahoo.com",
            taskLinks: "",
            appendAddMoreTask: !0
        }, {
            header: "Inbox",
            headerUrl: "https://mail.yahoo.com",
            content: "Loading...",
            noContent: "There are no emails in your Inbox folder.",
            errorMessage: "Couldn't connect to Yahoo Mail",
            errorFooterText: "Go to Yahoo Mail",
            maxNotifications: "9999",
            requestTimeout: "10000",
            taskLinks: "Compose|https://mrd.mail.yahoo.com/compose;Calendar|https://calendar.yahoo.com/;Sign out|#",
            contentEmpty: "There are no emails in your Inbox folder.",
            appendAddMoreTask: !0
        }
        ],
        gmail: [{
            header: "Gmail",
            content: "Sign in to view your emails",
            footerText: "Sign in",
            footerUrl: "",
            taskLinks: "",
            appendAddMoreTask: !0
        }, {
            header: "Inbox",
            headerUrl: "https://www.gmail.com",
            content: "Loading...",
            noContent: "There are no emails in your Inbox folder.",
            errorMessage: "Couldn't connect to Gmail",
            errorFooterText: "Go to Gmail",
            maxNotifications: "9999",
            requestTimeout: "10000",
            taskLinks: "Compose|https://mail.google.com/mail/?view=cm;Google Calendar|https://www.google.com/calendar;Sign out|#",
            contentEmpty: "There are no emails in your Inbox folder.",
            appendAddMoreTask: !0
        }
        ],
        mailProviderOptions: [{
            mailTileRemovedEvent: "mailTileRemovedEvent"
        }
        ]
    }
});
define("outlookInbox.tokens", {
    outlookPreviewApiUrl: "https://hotmailproxy.msn.com/pm/v1.0/getheaders.aspx",
    outlookPreviewApiUrlBetaplace: "https://hotmailproxy.betaplace.com/pm/v1.0/getheaders.aspx"
});
define("pagingSection", ["jquery", "jqBehavior", "modernizr", "pagingAnimator", "device", "pointerEvents", "touchGestures", "touchDataManager", "requestAnimationFrame", "format", "imgSrc", "paging.tokens", "dir.tokens", "viewAware", "mediator", "window", "c.deferred"], function(n, t, i, r, u, f, e, o, s, h, c, l, a, v, y, p) {
    function st(t, i) {
        function pi() {
            rt = n("<button class='leftarrow'><\/button>");
            ut = n("<button class='rightarrow'><\/button>");
            lt.prepend(rt).append(ut)
        }
        function wi(n) {
            i.contentSelector = n || i.contentSelector;
            f = lt.find(i.contentSelector);
            ht = r(f);
            b && ht.setup();
            ki()
        }
        function bi() {
            v > 1 && di();
            b && ht.teardown();
            y.unsub(tt, nr)
        }
        function rr() {
            c.loadInViewport && s(function() {
                f.attr(c.dataOffsetAttr, Math.round( - u * p * w) + ";0");
                c.loadInViewport(lt[0])
            })
        }
        function ki() {
            s(ur)
        }
        function ur() {
            for (var e, o, n, r = ci, s = f.length, t = 0; t < s; t++)
                if (vt.children().length && (vt.empty(), rt.removeClass(at), ut.removeClass(at), kt(f, 0)), k = vi = ct = yi = st = v = u = 0, pt = wt = yt=!1, ai = vt.offset().left, e = r[0].clientWidth || r.outerWidth(!1), p = e + i.columnGap + i.scrollOffset, p)
                    if (v = Math.ceil(f[t].scrollWidth / p), y.sub(tt, nr), v > 1) {
                        for (b && ht.initialize(v, p, i.columnGap), or(), ni || fr(), ot = [], o = p / 2, n = 0; n < v; n++)
                            ot[n] = p * n + o - 1;
                            gt = f[0].scrollWidth;
                            ui(0);
                            tr()
                    } else 
                        ni && di()
        }
        function fr() {
            er(f);
            rt.on("click", fi);
            ut.on("click", fi);
            y.sub(it, gi);
            ni=!0
        }
        function di() {
            lt.off("gesture");
            rt.off("click", fi);
            ut.off("click", fi);
            y.unsub(it, gi);
            ni=!1
        }
        function gi(n) {
            f[0] === n.elem && (b && (u += n.direction*-1, ui()), si())
        }
        function nr(n) {
            f[0] === n && (bi(), wi())
        }
        function er(n) {
            if (e.settings.touchSupported&&!b && n.data("gesture")!==!0) {
                n.on("gesture", {
                    direction: "horizontal"
                }, cr);
                n.data("gesture", !0)
            }
        }
        function tr() {
            v > 1 && (wt ? ut.addClass(at) : ut.removeClass(at), pt ? rt.addClass(at) : rt.removeClass(at), i.extendedArrows && (wt || pt) && rt.add(ut).text(h(l.pagingArrowText, u + 1, v)))
        }
        function ri(n) {
            bt || s(function() {
                bt=!0;
                var t = hr(n) * w;
                u += n*-1;
                b ? (ht.incrementViewPort(n, d, nt), ui()) : (ht.animationStarting(f), kt(f, t, d, nt));
                window.setTimeout(si, i.autoHideArrowMS)
            })
        }
        function si() {
            s(function() {
                b || ht.animationCompleted(f);
                yt = bt=!1;
                sr();
                tr();
                rr();
                y.pub(et, ii)
            })
        }
        function ui(n) {
            b ? (pt = ht.allowScrollLeft(), wt = ht.allowScrollRight()) : (vi = k, yi = ct, k = n * w, ct = gt + k, ct < 0 && (ct = 0), pt = k < 0, wt = ct > p)
        }
        function or() {
            for (var t, n = 0; n < v; n++)
                t = document.createElement("span"), n == 0 && (t.className = "selected"), vt[0].appendChild(t);
            ei = vt.children();
            dt = ei.first()
        }
        function sr() {
            dt && dt.removeClass("selected");
            dt = n(ei[u]).addClass("selected")
        }
        function hr(n) {
            if (n == 1)
                k < ai && (ti != 0 ? (st = k + ti, ti = 0) : st = k + p, st > 0 && (st = 0));
            else if (n==-1 && k < ct) {
                st = k - p;
                st > ct && (st = ct);
                var t = gt + st + i.columnGap;
                !i.alwaysScrollCompletePage && t < ct && t < p && (st = k - t, ti = t)
            }
            return st
        }
        function ir(t) {
            t&&!i.autoHideArrow ? n(t).removeClass(at) : (rt.removeClass(at), ut.removeClass(at))
        }
        function fi(n) {
            ii = "click";
            typeof n.originalEvent != "undefined" && n.originalEvent && n.originalEvent.pointerType === "touch" && (ii = "tap");
            n.target == ut[0] ? u + 1 < v ? ri( - 1) : ir(n.target) : u > 0 ? ri(1) : ir(n.target);
            n.preventDefault()
        }
        function kt(n, t, i) {
            if (t == undefined) {
                var r = n.css("transform").split(","), u = 4;
                r.length > 6 && (u = 12);
                t = parseFloat(r[u]);
                t = isNaN(t) ? 0 : t
            } else 
                ui(t), i || (i = 0), ht.animate(t, i, nt, g);
            return t
        }
        function hi(n) {
            s(function() {
                var t = 0, o = Math.abs(k), s, r, h;
                if (bt=!0, k >= 0)
                    t = 0, u = 0;
                else if (!i.alwaysScrollCompletePage && ot[ot.length - 1] < Math.abs(k))
                    t =- ot[ot.length - 1] * w, u = v - 1;
                else if (o > p * (v - 1))
                    u = v - 1, t =- p * u * w;
                else {
                    if (n)
                        s=!0, (a.ltr && n == e.types.swipeRight ||!a.ltr && n == e.types.swipeLeft) && (s=!1), s ? u < v && (u = u + 1) : u > 0 && (u = u - 1);
                    else 
                        for (r = 0; r < v; r++)
                            if (o <= ot[r]) {
                                u = r;
                                break
                            } else if (o < p * (r + 1)) {
                                u = r + 1;
                                break
                            }
                    t =- p * u * w;
                    i.alwaysScrollCompletePage || (h = gt - p + i.columnGap, Math.abs(t) > h && (t =- h * w), v - 1 == u && ot[ot.length - 1] != t && (ot.pop(), ot.push(Math.abs(t))))
                }
                ht.animationStarting(f);
                kt(f, t, d);
                window.setTimeout(si, i.autoHideArrowMS)
            })
        }
        function cr(t) {
            if (t.direction) {
                ii = "swipe";
                switch (t.current.touchType) {
                case e.types.move:
                    ft&&!bt && s(function() {
                        yt ? oi.input(t.current.coord.x) : (oi.reset(t.current.coord.x), li = kt(f), yt=!0);
                        kt(f, li + oi.getDelta() | 0)
                    });
                    break;
                case e.types.swipeUp:
                case e.types.swipeDown:
                case e.types.throwUp:
                case e.types.throwDown:
                case e.types.cancel:
                case e.types.stop:
                    hi();
                    break;
                case e.types.swipeRight:
                case e.types.throwRight:
                    yt ? hi(e.types.swipeRight) : pt && ri(1);
                    break;
                case e.types.swipeLeft:
                case e.types.throwLeft:
                    yt ? hi(e.types.swipeLeft) : wt && ri( - 1);
                    break;
                case e.types.tap:
                    var i = t.originalEvent.originalEvent, r = i.target, u = document.createEvent("MouseEvent");
                    u.initMouseEvent("click", !1, !1, window, i.detail, i.screenX, i.screenY, i.clientX, i.clientY, i.ctrlKey, i.altKey, i.shiftKey, i.metaKey, i.button, null);
                    r.tagName !== "A" && (r = n(r).parentsUntil("LI", "A")[0]);
                    r && r.dispatchEvent(u)
                }
            }
        }
        var lt = t, f, rt, ut, ci = i.useViewPortForWidthContainer ? lt: lt.find(i.widthContainerSelector), vt = lt.find("." + i.pagingClass);
        vt.length ? (rt = lt.find(".leftarrow"), ut = lt.find(".rightarrow"), rt.length || pi()) : (vt = n("<div class='" + i.pagingClass + "'><\/div>"), ci.append(vt), pi());
        var ei, dt, oi = new o, yt=!1, bt=!1, li, ai, k, gt, vi, ct, yi, ot, at = "show", st, v, u, pt, wt, p, ni=!1, ti = 0, ht, ii = "";
        return {
            setup: wi,
            teardown: bi,
            update: ki
        }
    }
    var w, ot, ut;
    w = a.ltr ? 1 : - 1;
    var d = "400ms", g, k = "left", nt = "cubicBezierQuint", tt = "PagingContentUpdated", ft = u.isCapable("ShowMoveTouchGestures"), it = "snapCompleted", b = window.navigator && navigator.msManipulationViewsEnabled, rt=!1, et = "ViewabilityUpdatedEvent";
    return i.csstransforms3d && u.isCapable("AllowTransform3d") ? (k = "translate3d", g = k + "({0}px, 0px, 0px)", rt=!0) : i.csstransforms && u.isCapable("AllowTransform2d") && (k = "translateX", g = k + "({0}px)", rt=!0), ot = n(p), ut = {
        autoHideArrow: !1,
        autoHideArrowMS: 250,
        alwaysScrollCompletePage: !1,
        contentSelector: ".sectioncontent",
        columnGap: 23,
        extendedArrows: !0,
        pagingClass: "paging",
        scrollOffset: 0,
        useViewPortForWidthContainer: !1,
        widthContainerSelector: ".full-width"
    }, t(st, ut)
});
define("renderSectionAd", ["jquery", "jqBehavior", "adLoad", "c.dom"], function(n, t, i) {
    function r(n, t) {
        return {
            setup: function() {
                i.loadSectionAd(n[0], t.adItemSelector)
            }
        }
    }
    return t(r, {
        adItemSelector: ".showcasead"
    })
});
define("autoRefresh", ["jquery", "jqBehavior", "window", "headData"], function(n, t, i, r) {
    function u(t, u) {
        function v() {
            n.isNumeric(u.timeout) && (clearTimeout(s), s = i.setTimeout(w, u.timeout))
        }
        function y() {
            n.isNumeric(u.idleTimeout) && (f=!1, clearTimeout(h), h = i.setTimeout(p, u.timeout))
        }
        function p() {
            f || (f=!0);
            c && (i.location = a(i.location.href, l))
        }
        function w() {
            f ? i.location = a(i.location.href, l) : c=!0
        }
        function a(n, t) {
            var i = new RegExp("([?&])" + t + "=.*?(&|$)", "i"), f = n.indexOf("?")!==-1 ? "&": "?", r, u;
            return n.match(i) ? (r = n.match(i)[0].substring(4, 5), u = parseInt(r) + 1, n.replace(i, "$1" + t + "=" + u + "$2")) : n + f + t + "=1"
        }
        var e = {}, o=!1;
        r && r.clientSettings && r.clientSettings.autorefreshsettings && (e = {
            isEnabled: r.clientSettings.autorefreshsettings.is_market_enabled,
            timeout: r.clientSettings.autorefreshsettings.timeout,
            idle: r.clientSettings.autorefreshsettings.idle_enabled,
            idleTimeout: r.clientSettings.autorefreshsettings.idle_timeout
        }, o = r.clientSettings.autorefresh);
        var s, h, f=!0, c=!1, l = "AR";
        return {
            setup: function() {
                if (u = n.extend(!0, {}, u, e), o && u.isEnabled && (n.isNumeric(u.timeout) && (u.timeout = u.timeout * 6e4), v(), u.idle))
                    n(document).on(u.monitorEvents, y)
            }
        }
    }
    return t(u, {
        isEnabled: !1,
        timeout: 1,
        idle: !0,
        idleTimeout: 1,
        monitorEvents: "mousemove keypress"
    })
});
define("makeHomePage", ["jquery", "jqBehavior", "track", "navigation", "makeHomePage.tokens", "window"], function(n, t, i, r, u, f) {
    function e(n, t) {
        function o() {
            function o() {
                var i = t.fbText;
                i ? n.html(i).attr("href", t.fbUrl) : n.closest("li").hide()
            }
            var e = u.targetUrl || t.targetUrl;
            if (e)
                if (r.isHomePage(e))
                    o();
                else 
                    n.on("click", function s(t) {
                        t.preventDefault();
                        t.stopImmediatePropagation();
                        try {
                            r.setHomePage(e);
                            var u = r.isHomePage(e);
                            i.trackEvent({
                                type: "click_nonnav"
                            }, n, "", n.text() + ";dhplink" + (u ? "yes" : "no"));
                            u && (n.off("click", s), o())
                        } catch (h) {
                            f.console && console.error("setHomePage: " + h)
                        }
                    })
        }
        try {
            if (n.html() != u.text)
                return;
            var r = document.body;
            typeof r.isHomePage == "undefined" && (r.style.behavior = 'url("#default#homepage")');
            typeof r.isHomePage != "undefined" && o()
        } catch (e) {
            f.console && console.error("isHomePage: " + e)
        }
    }
    return t(e)
});
define("makeHomePage.tokens", {
    text: "Make MSN my home page",
    targetUrl: "http://www.msn.com/?pc=U142&ocid=U142DHP"
});
define("modal", ["jquery", "modal.tokens"], function(n, t) {
    function s() {
        var n = document.createElement("div");
        return n.setAttribute("id", r), n.setAttribute("title", t.closeButtonTitle), n
    }
    function h(n) {
        var t = document.createElement("section");
        return t.setAttribute("class", o), t.setAttribute("role", "dialog"), t.setAttribute("aria-labelledby", "dialog-title"), t.setAttribute("aria-describedby", "dialog-desc"), n && t.setAttribute("id", n), t
    }
    function a(n) {
        var i = document.createElement("button");
        return i.setAttribute("class", "close"), i.setAttribute("title", t.closeButtonTitle), i.click(function() {
            f.close(n)
        }), i
    }
    var r = "mask", o = "modal", c = "error", i, u, e = 220, l = "<h1>" + t.errorMessageTitle + "<\/h1><p>" + t.errorMessageLabel + "<\/p>", f = {
        open: function(t, v, y, p) {
            var w, b;
            i = i || n("body");
            u || (i.append(s()), u = n("#" + r), u.click(function() {
                f.close(y)
            }));
            w = n("#" + y);
            w.length == 0 && (b = h(), i.append(b), n("." + o + ":not([id])").attr("id", y), w.load(v, function(t, i, r) {
                i === "error" ? (w.append(l).addClass(c), n("p", this).append(r.status + " " + r.statusText)) : typeof p == "function" && p(y);
                w.append(a(y))
            }));
            i.addClass(r);
            n("#" + r + ", #" + y).fadeIn(e);
            n(".webkit > #" + y).show()
        },
        close: function(t) {
            i = i || n("body");
            n("#" + r + ", #" + t).fadeOut(e).hide();
            i.removeClass(r)
        },
        openModalWithContent: function(t, o, c, l) {
            var y, a, v;
            i = i || n("body");
            u || (i.append(s()), u = n("#" + r), u.click(function() {
                l && typeof l == "function" ? l() : f.close(o)
            }));
            y = n("#" + o);
            y.length == 0 && (a = h(o), c && (v = n("#" + c), v && v.appendTo(a)), i.append(a));
            i.addClass(r);
            n("#" + r + ", #" + o).fadeIn(e);
            n(".webkit > #" + o).show()
        }
    };
    return f
});
define("sectionPersonalizationManager", ["jqBehavior", "jquery", "window", "sectionPersonalization", "personalizationModal", "mediator"], function(n, t, i, r, u, f) {
    function e(n, e) {
        function g() {
            var n = t(this).closest(e.sectionsContainerElements), r = n.find(".stripenav").eq(0).find("li:first-child"), f = r.text(), i = c.getSectionInfo(n).sectionId, o = e.sectionRemovedText.replace("%s", f);
            n.append('<span class="undosectionremove" data-undo-id="' + i + '">' + o + '<\/span><button class="undosectionremove undobutton" data-undo-id="' + i + '"> ' + e.sectionUndoText + '<\/button><button class="undosectionremove closebutton" data-undo-id="' + i + '"/> ');
            t('button.undobutton[data-undo-id="' + i + '"]').on("click", function(t) {
                t.preventDefault();
                p(n, i)
            });
            t('button.closebutton[data-undo-id="' + i + '"]').on("click", function(t) {
                t.preventDefault();
                n.remove()
            });
            y();
            n.addClass("personalizationhidden").removeClass("expanded");
            a(n);
            u.removeSection(i);
            c.removeSection(n, function() {}, function() {
                p(n, i)
            })
        }
        function y() {
            t(e.hiddenStripes).remove()
        }
        function p(n, i) {
            t('.undosectionremove[data-undo-id="' + i + '"]').remove();
            n.removeClass("personalizationhidden").addClass("expanded");
            b();
            a(n);
            u.addSection(i)
        }
        function nt() {
            w.call(this, !0)
        }
        function tt() {
            w.call(this, !1)
        }
        function w(n) {
            var o, h;
            y();
            var c=!1, r = t(this).closest(e.sectionsContainerElements), f = t(e.sectionsContainerElements), u = r.prevAll(e.sectionsContainerElements).length;
            n && u > 0 && (o = f.eq(u - 1), u === 1 && (r.find(e.personalizationContextMenu).addClass(e.firstContextMenu), o.find(e.personalizationContextMenu).removeClass(e.firstContextMenu)), u === f.length - 1 && (r.find(e.personalizationContextMenu).removeClass(e.lastContextMenu), o.find(e.personalizationContextMenu).addClass(e.lastContextMenu)), t(i).scrollTop(t(i).scrollTop() - r.offset().top + o.offset().top), o.before(r), c=!0);
            !n && u < f.length - 1 && (h = f.eq(u + 1), u === 0 && (r.find(e.personalizationContextMenu).removeClass(e.firstContextMenu), h.find(e.personalizationContextMenu).addClass(e.firstContextMenu)), u === f.length - 2 && (r.find(e.personalizationContextMenu).addClass(e.lastContextMenu), h.find(e.personalizationContextMenu).removeClass(e.lastContextMenu)), t(i).scrollTop(t(i).scrollTop() + h.offset().top - r.offset().top), h.after(r), c=!0);
            c && b();
            s=!1
        }
        function b() {
            clearTimeout(o);
            o = setTimeout(k, e.reorderTimeout)
        }
        function k() {
            clearTimeout(o);
            o = null;
            var n = t(e.sectionsContainerElements);
            c.showSectionsInOrder(n)
        }
        function it() {
            var n = t(this).closest(e.sectionsContainerElements);
            u.open(n)
        }
        function h() {
            t(e.expandedPersonalizationContextMenu).data(e.contextMenuExpanded, !1).removeClass(e.expandedMenuClass);
            l=!1
        }
        function rt(n) {
            h();
            n.data(e.contextMenuExpanded, !0).addClass(e.expandedMenuClass);
            l=!0
        }
        function ut() {
            var n = t(this).closest(e.personalizationContextMenu);
            n.data(e.contextMenuExpanded) ? h() : rt(n);
            s=!1
        }
        function ft() {
            l && s && h();
            s=!0
        }
        function a() {
            var i = t(e.sectionsElements), n;
            t(e.personalizationContextMenu).removeClass(e.firstContextMenu + " " + e.lastContextMenu + " " + e.onlyContextMenu);
            i.first().find(e.personalizationContextMenu).addClass(e.firstContextMenu);
            i.last().find(e.personalizationContextMenu).addClass(e.lastContextMenu);
            n = t(e.sectionsContainerElements).not(".personalizationhidden");
            n.length === 1 && n.find(e.personalizationContextMenu).addClass(e.onlyContextMenu)
        }
        function v(n) {
            n && d(n);
            a(n);
            t(e.personalizationContextMenuExpander, n).on("click", ut);
            t(i).unload(function() {
                typeof o == "number" && (clearTimeout(o), k())
            });
            t(e.removeSectionButtonClass, n).on("click", g);
            t(e.moveSectionUpButtonClass, n).on("click", nt);
            t(e.moveSectionDownButtonClass, n).on("click", tt);
            t(e.addBelowButtonClass, n).on("click", it)
        }
        function d(n) {
            t(e.personalizationContextMenuExpander, n).off("click");
            t(e.removeSectionButtonClass, n).off("click");
            t(e.moveSectionUpButtonClass, n).off("click");
            t(e.moveSectionDownButtonClass, n).off("click");
            t(e.addBelowButtonClass, n).off("click")
        }
        var c = new r, s=!0, l=!1, o = null;
        t(e.headerPersonalizationButton).on("click", function() {
            u.open()
        });
        return {
            setup: function() {
                t("body").on("click.hideContextMenu", ft);
                f.sub("deferredPageContentLoaded", v);
                u.initPersonalizationModal();
                v()
            },
            teardown: function() {
                t("body").off("click.hideContextMenu");
                f.unsub("deferredPageContentLoaded", v);
                h();
                d()
            }
        }
    }
    return n(e, {
        addBelowButtonClass: ".personalizationcontextmenu .personalizationaddbelow",
        contextMenuExpanded: "menuExpanded",
        expandedMenuClass: "expanded",
        expandedPersonalizationContextMenu: ".personalizationcontextmenu.expanded",
        firstContextMenu: "first",
        lastContextMenu: "last",
        onlyContextMenu: "only",
        moveSectionDownButtonClass: ".personalizationcontextmenu .personalizationmovedown",
        moveSectionUpButtonClass: ".personalizationcontextmenu .personalizationmoveup",
        personalizationContextMenu: ".personalizationcontextmenu",
        personalizationContextMenuExpander: ".personalizationcontextmenu .expandpersonalizationmenu",
        removeSectionButtonClass: ".personalizationcontextmenu .personalizationremove",
        sectionRemovedText: "%s section removed",
        sectionRemoveTimeout: 2500,
        sectionsContainerElements: ".stripecontainer",
        sectionsElements: ".stripeouter:not(.mestripeouter)",
        sectionUndoText: "Undo",
        reorderTimeout: 2e3,
        headerPersonalizationButton: "header ul[role='menu'] li button.personalization",
        hiddenStripes: ".stripecontainer.personalizationhidden"
    })
});
define("personalizationModal", ["jquery", "modal", "sectionPersonalization", "imgSrc", "track"], function(n, t, i, r, u) {
    function tt() {
        p = new i;
        e = n("#" + f.modalId);
        e.remove();
        w();
        e.on("click", f.tileSelector, it).find(k).on("click", function() {
            w();
            t.close(f.modalContainerId)
        });
        v = e.find(d).one("click", l).one("click", l);
        y = e.find(g);
        e.appendTo(n("body"))
    }
    function w() {
        var i = n(f.sectionSelector), t = e.find(f.tileSelector);
        nt = t.length;
        h = 0;
        s = [];
        t.each(function(t, r) {
            var u = n(r), e = u.data(f.dialogSectionIdAttr), c = i.filter("[data-" + f.pageSectionIdAttr + "='" + e + "']");
            !c.length || c.eq(0).hasClass(f.removedSectionClass) ? u.addClass(o) : (u.removeClass(o), h++, s.push(e))
        })
    }
    function it(t) {
        var i, r, f;
        t.preventDefault();
        i = n(this);
        i.hasClass(o) ? (i.removeClass(o), h++) : h > 1 && (i.addClass(o), h--);
        r=!i.hasClass(o);
        f = i.attr("data-stripe-id");
        u.trackEvent({
            type: "click_nonnav",
            target: this
        }, "", "", f + "|add=" + r);
        t.stopPropagation()
    }
    function rt(n) {
        b(n, !1);
        var t = s.indexOf(n);
        t >= 0 && s.splice(t, 1)
    }
    function ut(n) {
        b(n, !0);
        s.push(n)
    }
    function b(n, t) {
        if (e && e.length) {
            var i = e.find("button[data-" + f.dialogSectionIdAttr + "='" + n + "']");
            t ? (i.removeClass(o), h++) : (i.addClass(o), h--)
        }
    }
    function ft() {
        n("img", e).each(function() {
            r.reset(this)
        })
    }
    function et(n) {
        a = n;
        ft();
        t.openModalWithContent(null, f.modalContainerId, f.modalId);
        v.off("click", l).one("click", l).removeClass("disabled");
        y.hide()
    }
    function ot() {
        var r = {}, i = [], t, u;
        return c = [], t = n(f.sectionSelector).map(function(t, i) {
            var u = n(i), e = u.data(f.pageSectionIdAttr);
            return r[e] = t, {
                id: e,
                path: f.pdpPath,
                shown: !u.hasClass(f.removedSectionClass)
            }
        }), e.find(f.tileSelector).each(function(u, e) {
            var l = n(e), s = l.data(f.dialogSectionIdAttr), a=!l.hasClass(o), h = t[r[s]];
            h ? (h.shown = a, h.shown && c.push(s)) : a && (i.push({
                id: s,
                path: f.pdpPath,
                shown: !0
            }), c.push(s))
        }), i.length && (u = t.length, a && (u = r[a.data(f.pageSectionIdAttr)] + 1), i.unshift(u, 0), [].splice.apply(t, i)), t
    }
    function st() {
        var t, n;
        if (c.length !== s.length)
            return !0;
        for (s.sort(), c.sort(), t = c.length, n = 0; n < t; n++)
            if (c[n] !== s[n])
                return !0;
        return !1
    }
    function l(n) {
        v.addClass("disabled");
        var r = ot(), i = st();
        i ? (y.show(), p.updateSectionInfosInOrder(r, function() {
            location.reload(!0)
        })) : t.close(f.modalContainerId);
        u.trackEvent({
            type: "click_nonnav",
            target: this
        }, "", "", "done|dirty=" + i);
        n.stopPropagation()
    }
    var f = {
        pageSectionIdAttr: "section-id",
        dialogSectionIdAttr: "stripe-id",
        pdpPath: "MSNHomePage/Stripes",
        modalId: "addRemoveSectionsModal",
        modalContainerId: "addRemoveSectionsModalContainer",
        removedSectionClass: "personalizationhidden",
        sectionSelector: ".stripecontainer",
        tileSelector: "button[data-stripe-id]"
    }, o = "personalizationPending", k = ".closebutton", d = ".donebutton", g = ".content div.loading", nt, h, a, v, y, e, p, s = [], c = [];
    return {
        initPersonalizationModal: tt,
        removeSection: rt,
        addSection: ut,
        open: et,
        setConfig: function(t) {
            n.extend(!0, f, t)
        }
    }
});
define("sectionPersonalization", ["jquery", "headData"], function(n, t) {
    function e(n, t) {
        return t.pdpPathTemplate.replace("%sectionId", n)
    }
    function f(n, t) {
        var i = n.attr(t.sectionIdAttributeName), r;
        return u[i] || (r = e(i, t), u[i] = {
            sectionId: i,
            pdpPath: r
        }), u[i]
    }
    function o(n, t) {
        return f(n, t).sectionId
    }
    function s(n, t) {
        return f(n, t).pdpPath
    }
    function w(n, t) {
        return {
            Id: n,
            State: t ? "expanded": "hidden"
        }
    }
    function h(n, t, i) {
        return {
            operation: "Add",
            path: t,
            payload: JSON.stringify(w(n, i))
        }
    }
    function c(n, t, i) {
        var r = [];
        return r.push(h(n, t, i)), JSON.stringify(r)
    }
    function b(t) {
        var i = n.map(t, function(n) {
            return h(n.id, n.path, n.shown)
        });
        return JSON.stringify(i)
    }
    function l(n, t, i) {
        return c(o(n, i), s(n, i), t)
    }
    function a(n, t, i) {
        return c(n, e(n, i), t)
    }
    function k(t, i) {
        return n.map(t, function(t) {
            var r = n(t);
            return {
                id: o(r, i),
                path: s(r, i),
                shown: !0
            }
        })
    }
    function r(t, i, r) {
        i = i || n.noop;
        r ? t.then(function(n) {
            i(n)
        }, function(n) {
            r(n)
        }) : t.always(function(n) {
            i(n)
        })
    }
    function v(i, r, u, f, e) {
        if (r >= u)
            f && f();
        else {
            var o = n.ajax({
                url: t.pdpdeltaupdateapi,
                type: "POST",
                data: b(i[r])
            }), s = v.bind(null, i, r + 1, u, f, e);
            o.then(s, function(n) {
                e ? e(n) : f && f(n)
            })
        }
    }
    function d(i, u, f) {
        var e = n.ajax({
            url: t.pdpdeltaupdateapi,
            type: "POST",
            data: l(i, !1, this._config)
        });
        r(e, u, f)
    }
    function g(i, u, f) {
        var e = n.ajax({
            url: t.pdpdeltaupdateapi,
            type: "POST",
            data: l(i, !0, this._config)
        });
        r(e, u, f)
    }
    function nt(i, u, f) {
        var e = n.ajax({
            url: t.pdpdeltaupdateapi,
            type: "POST",
            data: a(i, !1, this._config)
        });
        r(e, u, f)
    }
    function tt(i, u, f) {
        var e = n.ajax({
            url: t.pdpdeltaupdateapi,
            type: "POST",
            data: a(i, !0, this._config)
        });
        r(e, u, f)
    }
    function it(n, t, i) {
        var r = k(n, this._config);
        y(r, t, i)
    }
    function y(n, t, i) {
        for (var u = [], r = 0, f = n.length; r < f; r += 10)
            u.push(n.slice(r, r + 10));
        v(u, 0, u.length, t, i)
    }
    function i(t) {
        var i = {};
        n.extend(!0, i, p, t);
        this._config = i
    }
    var p = {
        sectionIdAttributeName: "data-section-id",
        pdpPathTemplate: "MSNHomePage/Stripes"
    }, u = {};
    return i.prototype.removeSection = d, i.prototype.addSection = g, i.prototype.removeSectionById = nt, i.prototype.addSectionById = tt, i.prototype.showSectionsInOrder = it, i.prototype.updateSectionInfosInOrder = y, i.prototype.getSectionInfo = function(n) {
        return f(n, this._config)
    }, i
});
define("modal.tokens", {
    closeButtonTitle: "Close",
    errorMessageTitle: "Message:",
    errorMessageLabel: "Error loading content"
});
define("paging.tokens", {
    seeAllLink: "See all",
    pagingArrowText: "{0} of {1}"
});
define("pageBindings", ["binding", "c.dom"], function(n) {
    return function(t) {
        var i, r;
        n("truncate", ".stripecontent h4,.stripecontent p,.meflyout>section p,.smalla h4,.today1 h4,.todaymodule .title,.todaystripe .title,.todayprimemobile h4,.popularnowimagewithlinks p,.popularnowimagewithlinks .first .title", t).all();
        i = 20;
        n("arrowScroll", ".stripeouter.mestripeouter", t).all({
            scrollRegion: ".mestripe",
            reduceScrollAmount: i,
            addArrowButtons: !0
        });
        n("arrowScroll", ".todaynavigation .full-width", t).all({
            scrollRegion: ".sectioncontent",
            reduceScrollAmount: i,
            addArrowButtons: !0
        });
        r = n.views.SIZE234COLUMN;
        n("pagingSection", ".stripeouter:not(.mestripeouter)", t).view(r, {
            contentSelector: ".stripe .paging-container",
            scrollOffset: 0,
            alwaysScrollCompletePage: !1,
            columnGap: 15,
            extendedArrows: !0,
            useViewPortForWidthContainer: !1,
            widthContainerSelector: ".paging-container"
        });
        require(["c.deferred"], function() {
            n("sectionPersonalizationManager", "body", t).all();
            n("autoRefresh", "#main", t).all();
            n("outlookInbox", ".meflyouts .mail", t).all();
            n("meTwitter", ".meflyouts .twitter", t).all();
            n("meFacebook", ".meflyouts .facebook", t).all();
            n("meGmail", ".meflyouts .gmail", t).all();
            n("meYahooMail", ".meflyouts .yahoomail", t).all();
            n("loadMeImages", ".mestripe li.maps", t).all({
                imagesToLoad: ".meflyouts .maps img"
            });
            n("loadMeImages", ".mestripe li.xbox", t).all({
                imagesToLoad: ".meflyouts .xbox section li img"
            });
            n("makeHomePage", ".todaynavigation .stripenav h2 a", t).all()
        })
    }
});
require(["refreshPdpModules", "c.sso"], function(n) {
    n()
});
require(["meModule", "c.sso"], function(n) {
    n.ssoAutoRefresh()
});
require(["refreshPdpModules", "c.pdp"], function(n) {
    n()
});
require(["refreshDeferredModules", "binding", "c.onload"], function(n, t) {
    n().done(function() {
        t("meOffice", ".meflyouts .office").all();
        t("meOffice", ".meflyouts .onenote").all()
    })
});
define("pagingAnimator", ["jquery", "modernizr"], function(n, t) {
    var i = t.translateXCss;
    return function(t) {
        function r(n, r, u) {
            var f = {};
            f[i] = n + "px";
            t.animate(f, {
                queue: !1,
                duration: r,
                easing: u
            })
        }
        return {
            initialize: n.noop,
            animationStarting: n.noop,
            animate: r,
            animationCompleted: n.noop
        }
    }
});
define("meStripe", ["jquery", "jqBehavior", "dom", "document", "dir.tokens", "imgSrc", "touchGestures", "scrollStop", "track", "mediator", "viewport", "truncate", "scrollLeft", "requestAnimationFrame"], function(n, t, i, r, u, f, e, o, s, h, c, l, a, v) {
    function y(t) {
        function pt(n) {
            return n.pointerType && n.pointerType !== "mouse" && n.pointerType !== 4
        }
        function ei(i) {
            if (i.type === "click"&&!yt&&!pt(i.originalEvent)) {
                var r = n(this).parent();
                t.addClass(ot);
                bt(y.index(r))
            }
            i.preventDefault();
            i.stopPropagation();
            n(this).blur()
        }
        function oi(t) {
            if (t.type === "touchend" || yt || pt(t.originalEvent))
                return t.preventDefault(), t.stopPropagation(), !1;
            n(this).blur()
        }
        function wt() {
            v(function() {
                t.find(".signout img").each(function() {
                    f.go(this)
                })
            });
            t.off(ft + " " + et, wt)
        }
        function ii(n) {
            y.find(n.target).length || at.find(n.target).length || (p(), dt.off(et, ii))
        }
        function si() {
            var i = n(this).parent();
            if (i.hasClass(tt))
                p();
            else {
                if (ut)
                    return;
                t.addClass(ot);
                bt(y.index(i));
                dt.on(et, ii)
            }
        }
        function ri() {
            ut=!1
        }
        function ct() {
            p();
            b = w.offset();
            it = w.outerWidth();
            b.right = b.left + it;
            b.bottom = b.top + t.outerHeight()
        }
        function ui(i) {
            var h, f, c;
            if (!ut&&!pt(i.originalEvent)) {
                var e = i.originalEvent, o = u.ltr ? e.pageX: window.innerWidth - e.pageX, s = e.pageY;
                if (s >= b.top && s <= b.bottom && o >= b.left && o <= ni) {
                    for (st || (k || (k = setTimeout(hi, 700)), st=!0), h = o + vt - b.left, f = 0, c = rt.length; f < c; f++)
                        if (h < rt[f]) {
                            nt !== f && (n.contains(y[f], r.elementFromPoint(e.clientX, e.clientY)) ? bt(f) : p());
                            break
                        }
                } else 
                    g ? o < g.x1 || o > g.x2 || s < g.y1 || s > g.y2 ? p() : k!==!1 && (clearTimeout(k), k=!1, p()) : st && (t.removeClass(ot), k && (clearTimeout(k), k=!1), st=!1)
            }
        }
        function lt(n) {
            var t = a(w[0]);
            n ? t !== vt && (ut=!0, p()) : p();
            vt = t
        }
        function fi() {
            var t, n, i;
            for (rt = [], t = 0, n = 0, i = y.length; n < i; n++)
                rt.push(t += y[n].offsetWidth);
            ni = rt[n - 1] + b.left
        }
        function hi() {
            k=!1;
            t.addClass(ot)
        }
        function ci() {
            var t = n(this);
            t.data({
                name: i.text(this)
            })
        }
        function li(n) {
            var t = y.eq(n);
            nt == n
        }
        function bt(n) {
            var t, e;
            if (!ut && nt !== n) {
                var r = y.eq(n), i = at.eq(n), f = r.data();
                li(n);
                p();
                d = i.addClass(tt);
                r.addClass(tt);
                ht = setTimeout(function() {
                    var n = f && f.name ? f.name: "meTile";
                    h.pub(n.split(" ")[0] + "FlyoutVisible");
                    s.trackEvent({
                        type: "mouseenter"
                    }, r, "", n)
                }, 250);
                i.attr("data-truncated") !== "true" && i.find("> section ul > li").length && (l(" > section p", i, null).setup(), i.attr("data-truncated", "true"));
                nt = n;
                ai(r, d);
                t = d.offset();
                e = d.outerWidth();
                g = {
                    x1: u.ltr ? t.left: window.innerWidth - t.left - e,
                    y1: t.top,
                    x2: u.ltr ? t.left + e: window.innerWidth - t.left,
                    y2: t.top + d.outerHeight()
                }
            }
        }
        function p() {
            d && (d.removeClass(tt), y.eq(nt).removeClass(tt), d = nt = g=!1, ht && (clearTimeout(ht), ht=!1))
        }
        function ai(n, t) {
            var r = t.outerWidth(), i = u.ltr ? n.position().left: it - n.position().left - n[0].offsetWidth, f = i + r;
            f > it && (i = u.ltr ? n.position().left + n.outerWidth() - r : it - n.position().left - r);
            u.ltr ? t.css({
                left: i
            }) : t.css({
                right: i
            })
        }
        var kt = n(r), dt = n("body"), gt = n(window), w = t.find(".mestripe"), y = w.find("> ul > li"), at = t.find(".meflyouts .meflyout"), tt = "hover", ot = "noflyoutdelay", b, vt = a(w[0]), it = w.outerWidth(), rt = [], ni, d=!1, nt=!1, g=!1, st=!1, k=!1, ut=!1, yt = /(iPad|iPhone|iPod|Android)/g.test(navigator.userAgent), ft = navigator.pointerEnabled || navigator.msPointerEnabled ? "pointermove MSPointerMove" : "mousemove", et = "touchstart MSPointerDown pointerdown", ti = "touchend MSPointerUp pointerup", ht;
        return y.each(ci), {
            setup: function() {
                if (h.sub(c.sizeChangeEventName, ct), !yt)
                    kt.on(ft, ui);
                o(w, null).setup();
                w.on("scroll", lt).on("scrollStop", ri);
                t.on(ft + " " + et, wt);
                gt.on("blur MSOrientationChange orientationchange", p);
                if (lt(), ct(), fi(), e.settings.touchSupported)
                    y.children("a").on(ti, si);
                y.children("a").on("click touchend", function() {
                    (n(this).attr("href") == "#" ? ei : oi).apply(this, arguments)
                })
            },
            update: function() {
                y = w.find("> ul > li");
                at = t.find(".meflyouts .meflyout");
                lt();
                ct();
                fi()
            },
            teardown: function() {
                h.unsub(c.sizeChangeEventName, ct);
                kt.off(ft, ui);
                w.off("scroll", lt).off("scrollStop", ri);
                t.off(ft + " " + et, wt);
                y.children("a").off(ti).off("mouseup").off("click");
                gt.off("blur MSOrientationChange orientationchange", p)
            }
        }
    }
    return t(y)
});
define("keypressCapture", ["jquery", "jqBehavior", "mediator", "perfMarker", "window", "document"], function(n, t, i, r, u, f) {
    function s(t, u) {
        function a() {
            h=!0
        }
        function v() {
            h || n(this).focus()
        }
        function c(t) {
            t.charCode > 32 && t.charCode <= 126 && n(t.target).is(l) && s.focus()
        }
        var s = n(u.focusSelector), l = ":not(input,textarea,select,{selector})".replace("{selector}", u.focusSelector), h=!1, f;
        o.one("focusin mousedown touchstart", a);
        s.one("keypress", v);
        return f = {
            setup: function() {
                t.on("keypress", c);
                e || (e=!0, r("TimeToFirstSearchInteractive", !0))
            },
            teardown: function() {
                t.off("keypress", c)
            }
        }, i.sub("feedback.open", f.teardown), i.sub("feedback.close", f.setup), f
    }
    var o = n(f), e;
    return t(s, {
        focusSelector: ""
    })
});
define("pageBindings.pc", ["binding", "perfMarker", "headData", "c.dom"], function(n, t, i) {
    return n("keypressCapture", "body").all({
        focusSelector: "#q"
    }), function(t) {
        i && i.clientSettings && i.clientSettings.static_page || (n("meStripe", ".mestripeouter", t).all(), require(["c.deferred"], function() {}))
    }
});
define("makeHomepageDialog", ["jquery", "jqBehavior", "window", "track"], function(n, t, i, r) {
    function u(t) {
        function a() {
            u.length && (s.click(c), u.click(f), e.click(f))
        }
        function f(n) {
            var f, u;
            return n.preventDefault(), f = t.find("#makehomepagedialog"), t.css("visibility", "hidden").show(), f.css({
                top: "50%",
                marginTop: f.height()/-2
            }), t.css("visibility", "visible"), u = t.attr(l), u && u != "false" && (i.location = u), r.trackEvent({
                type: h
            }, this), !1
        }
        function c() {
            t.hide();
            u.hide();
            e.hide();
            r.trackEvent({
                type: h
            }, this)
        }
        function v() {
            s.off(o, c);
            u.off(o, f);
            e.off(o, f)
        }
        var u = n(".todaynavigation .makehomepage"), e = n("#user-profile #makehomepage"), s = t.find(".closebutton"), o = "click", h = "click_nonnav", l = "data-installer";
        return {
            setup: a,
            teardown: v
        }
    }
    return t(u)
});
require(["binding", "c.dom"], function(n) {
    n("makeHomepageDialog", "#makehomepagecontainer").all()
});
define("bingAutoSuggest.tokens", {
    resourceJs: "http://www.bing.com/s/as/1436447/en.js",
    helpLinkText: "What's this?",
    helpLinkUrl: "http://help.live.com/help.aspx?project=wl_searchv1&market=en-US&querytype=keyword&query=nsmtseggus",
    market: "en-US",
    popularNowText: "Popular Now",
    enablePopularNow: 1,
    bingHelp: "http://onlinehelp.microsoft.com/en-us/bing/ff808490.aspx",
    disableText: "Turn off search suggestions",
    enableText: "Turn on search suggestions"
});
define("bingAutoSuggest", ["bingAutoSuggest.tokens", "jquery", "jqBehavior", "mediator", "getCookie", "setCookie", "track", "events", "format", "headData", "window", "document"], function(n, t, i, r, u, f, e, o, s, h, c, l) {
    function b(n, i) {
        function k() {
            var n = new RegExp("\\b" + it + "=0\\b", "i");
            return u(tt).match(n) ? 1 : 0
        }
        function lt(n, i, r) {
            if (typeof n != "undefined") {
                var u = n.js;
                u && t.isFunction(i) && t.ajax({
                    url: u,
                    dataType: "script",
                    success: function() {
                        i(r)
                    }
                })
            }
        }
        function at() {
            nt();
            var u = t.extend(!0, {}, {
                f: n.attr("id"),
                i: i.inputId
            }, i.config, {
                sid: w || ""
            });
            u.cb = function(n) {
                if (g=!0, r.pub(y.beforeSubmit, n.value), g) {
                    var u = t(n).parents("form"), i = u[0], f = i && i.action && i.action.indexOf("://") >= 0 ? i.action: "";
                    e.trackEvent({
                        type: "submit",
                        target: i
                    }, null, f);
                    u.submit()
                }
            };
            c._G = {
                Mkt: i.market
            };
            typeof c.sa_autosuggest != "undefined" && (c[v] = new c.sa_autosuggest(u), c[v].init(v), u.lmh || t(".sa_om").hide(), nt(), n[0] && o.unbind(n[0], "submit", e.trackEvent))
        }
        function ot() {
            var r = k(), u;
            return (h.attr(ut, r ? ft : et), typeof c[v] != "undefined") ? (c[v].enable(!k()), nt(), 1) : n[0] && h[0]&&!r ? (i.config.asId && (rt = i.config.asId), n.append(t("<div><\/div>").attr("id", rt)), i.delayBind == 2 ? (d = 1, st()) : ht(p.delay, st), lt(i.resources, at), u = new Image, u.src = i.config.u + "&q=", 1) : 0
        }
        function vt(n, t) {
            var r = t ? "1": "0";
            i.cookieDomain || (i.cookieDomain = f.topDomain);
            f(tt, it + "=" + r, i.cookieExpiry, i.cookieDomain, "/");
            i.delayBind = 2;
            ot();
            n.preventDefault()
        }
        function st() {
            return a(u("_SS"), 5) || f("_SS", "SID=00", 365, i.cookieDomain, "/"), 1
        }
        function ht(n, i) {
            n == 0 ? i() : n == 1 ? t(l).ready(i) : n == 2 && yt(i)
        }
        function yt(n) {
            var t = ".asue";
            h.bind("click" + t, function() {
                n() && h.unbind(t)
            }).bind("keyup" + t, function(i) {
                i.which != 27 && i.which != 9 && n() && h.unbind(t)
            })
        }
        function nt() {
            var n, r, i, f;
            w || (n = a(p.cn, 1) ? p.cn.split("+") : null, n && a(n[0], 1) && (r = u(n[0]), a(r, 4) && (i = r.match(/SID=[\d(A-Z(a-z)]+/), i && a(i[0], 5) && (w = i[0].substr(4)))));
            typeof c[v] != "undefined" && (f = c[v].sid, t.isFunction(f) && f(w), d && (d = 0, l.activeElement == h[0] && h.click()));
            w && t.isFunction(p.onCk) && (p.onCk(), p.onCk = 0)
        }
        function ct(n) {
            var t = k();
            n.text(t ? i.enableText : i.disableText);
            h.attr(ut, t ? ft : et)
        }
        var tt = "SRCHHPGUSR", it = "AS", rt = "sw_as", ut = "autocomplete", ft = "on", et = "off", v = "sa_inst", p = i.sharedCk, pt = new RegExp("^http(s?)://[a-zA-z\\d\\-.]+\\.(" + p.domain + ")"), d = 0, w, b, h, g;
        i.config.u = s(i.config.u, i.formCode);
        i.sharedCk.ru = s(i.sharedCk.ru, i.formCode);
        b = t(i.toggleSelector);
        h = t("#" + i.inputId, n);
        r.sub(y.preventSubmit, function() {
            g=!1
        });
        ct(b);
        b.click(function(n) {
            n.stopImmediatePropagation();
            vt(n, k());
            ct(b)
        });
        ht(i.delayBind, ot)
    }
    function a(n, t) {
        return typeof n == "string" && (!t || n.length >= t)
    }
    var v = h.ssl===!0 ? "https": "http", w = {
        resources: {
            js: v + n.resourceJs.substr(n.resourceJs.indexOf("://"))
        },
        config: {
            l: [n.helpLinkText, n.helpLinkUrl, 1],
            r: "AutoSugShared",
            o: "s+a+p+hs+",
            h: 1,
            k: 0,
            m: 8,
            d: 100,
            u: v + "://api.bing.com/qsonhs.aspx?form={0}",
            mkt: n.market,
            ol: 1,
            tPN: n.popularNowText,
            eLO: 1,
            eHS: 1,
            ePN: n.enablePopularNow,
            nw: "true",
            lh: n.bingHelp,
            lmh: 0
        },
        disableText: n.disableText,
        enableText: n.enableText,
        inputId: "q",
        openNew: "1",
        market: n.market,
        cookieExpiry: 365,
        cookieDomain: "msn.com",
        delayBind: 1,
        sharedCk: {
            delay: 1,
            ru: v + "://" + location.host + "/sck.aspx&form={0}",
            pu: v + "://www.bing.com/sck",
            cn: "_SS",
            domain: "msn.com",
            onCk: function() {}
        },
        toggleSelector: "#asugoff"
    }, y = {
        beforeSubmit: "autosuggestBeforeSubmit",
        preventSubmit: "autosuggestPreventSubmit"
    }, p = i(b, w);
    return p.event = y, p.resize = function(n) {
        c.sa_inst && c.sa_inst.autosuggest.setQuery(n.val())
    }, p
});
require(["binding", "c.deferred"], function(n) {
    n("bingAutoSuggest", "#srchfrm").all()
});
define("autosuggestDestinationUrl", ["jquery", "format", "c.deferred"], function(n, t) {
    var i = {
        weather: "/{7}/we-latlon-{0},{1}?c={2}&st={3}&cn={4}&iso={5}&type={6}&id={8}",
        geolocation: "?lat={0}&long={1}&cty={2}&subdiv={3}&cntreg={4}&cc={5}&type={6}&id={8}",
        weathercity: "/{5}/{2},{3},{4}/we-city-{0},{1}?q={7}&form={9}",
        weatherski: "/{5}/{2},{3},{4}/we-ski-{8}-{0},{1}?q={7}&form={9}"
    }, u = function(n, u) {
        var o = u && u.urltype ? u.urltype.toLowerCase(): "geolocation", s = i[o] ? i[o]: i.geolocation, e = f;
        o == "weather" && (s = n.type == 2 ? i.weatherski : i.weathercity, e = r);
        var c = u.formcode && u.formcode.length ? u.formcode.toUpperCase(): "", h = "", l = n.state && n.name != n.state ? n.name + "-" + n.state: n.name;
        return s && n.name && n.lat && n.lon && (h = t(s, n.lat, n.lon, e(n.name), e(n.state), e(n.countryRegion), e(n.isoCode), n.type, r(l).toLowerCase().replace(/\&/g, ""), n.id != null ? e(n.id) : "", c)), h
    }, f = function(n) {
        return n ? encodeURIComponent(n.replace(/\./g, "").replace(/\//g, "")) : ""
    }, r = function(n) {
        return n ? encodeURIComponent(n.replace(/\./g, "").replace(/\//g, "").replace(/\&/g, "").replace(/\s+/g, "-")) : ""
    };
    return u
});
define("queryAutosuggest", ["jquery", "window", "mediator", "c.deferred"], function(n, t, i) {
    var e = {
        dataset: "w8weather",
        host: "//api.bing.com/qsonhs.aspx",
        market: "en-us",
        count: 5,
        form: "PRWKWB"
    }, r = null, f = null, o = function(t, u) {
        var o = n.extend({}, e, u), s;
        (r && r.readystate != 4 && r.abort(), t && o.dataset && o.host && o.market && o.form && o.count) && (s = {
            form: o.form,
            ds: o.dataset,
            mkt: o.market,
            count: o.count,
            type: "cb",
            cb: "populateAutosuggestResponse",
            q: t
        }, f = t, r = n.ajax({
            url: o.host,
            cache: "true",
            async: "false",
            type: "GET",
            data: s,
            dataType: "jsonp",
            timeout: 1e4
        }).fail(function(n, t, r) {
            t != "parsererror" && t != "abort" && i.pub("locSrchError", r)
        }))
    }, u = function(n, t) {
        var i = t ? t: 2;
        return n && (n = parseFloat(n).toFixed(i)), n || ""
    }, s = function(n, t) {
        var i, r, f, e;
        if (t.type != 1) {
            if (i = "ID_" + t.id, n[i])
                return !1;
            n[i] = t
        }
        return (r = [t.name, t.state, t.countryRegion].join(), n[r])?!1 : (n[r] = t, f = [u(t.lat), u(t.lon)].join(), n[f])?!1 : (n[f] = t, e = [t.name, u(t.lat, 1), u(t.lon, 1)].join(), n[e])?!1 : (n[e] = t, !0)
    };
    return t.populateAutosuggestResponse = t.populateAutosuggestResponse || function(n) {
        var l = [], a = {}, e, o, u, h, c, t, r;
        if (n && n.AS) {
            if (f != n.AS.Query)
                return;
            if (e = n.AS.Results, e && e.length > 0)
                for (o = 0; o < e.length; o++)
                    if (u = e[o], u && u.Suggests && u.Suggests.length > 0)
                        for (h = 0; h < u.Suggests.length; h++)
                            c = u.Suggests[h], c && c.Txt && (t = c.Txt.split("_"), t && t.length > 6 && (r = {}, r.name = t[0] || "", r.state = t[2] || "", r.countryRegion = t[3] || "", r.isoCode = t[4] || "", r.lat = t[5] || "", r.lon = t[6] || "", r.id = t[7] || null, r.type = t[1] == "S" ? 2 : 1, s(a, r) && l.push(r)))
        }
        i.pub("locSrchUpdate", {
            result: l,
            mode: "as",
            status: "success"
        })
    }, o
});
define("weaLocationAutosuggest", ["jquery", "jqBehavior", "queryAutosuggest", "autosuggestDestinationUrl", "mediator", "keyCode", "format", "navigation", "window", "track", "c.deferred"], function(n, t, i, r, u, f, e, o, s, h) {
    function l(t, l) {
        function ft(n) {
            n.data("init") || (rt = n, n.data("init", !0), v = n.closest("form"), v.append(a))
        }
        function ht(n) {
            switch (n.keyCode) {
            case f.Arrow.Down:
            case f.Arrow.Up:
                break;
            default:
                et.call(this, n)
            }
        }
        function ct(t) {
            var i, r, u;
            switch (t.keyCode) {
            case f.Arrow.Down:
                t.preventDefault();
                b && (d ? (i = n("li.selected", a).removeClass("selected"), r = i.next(), r[0] ? r.addClass("selected") : n("li", a).first().addClass("selected")) : it());
                break;
            case f.Arrow.Up:
                t.preventDefault();
                b && (d ? (i = n("li.selected", a).removeClass("selected"), u = i.prev(), u[0] ? u.addClass("selected") : n("li", a).last().addClass("selected")) : it());
                break;
            case f.Space:
            case f.Enter:
                ot(t, n(this));
                break;
            case f.Esc:
                d && (t.preventDefault(), g())
            }
        }
        function lt() {
            var t = n(this);
            tt=!0;
            b > 0 && (t.select(), it());
            u.sub("locSrchUpdate", st);
            u.sub("locSrchError", nt)
        }
        function at() {
            tt=!1;
            d && g();
            u.unsub("locSrchUpdate", st);
            u.unsub("locSrchError", nt)
        }
        function et() {
            var r = n(this), t;
            ft(r);
            t = n.trim(r.val());
            t ? t != w && (w = t, i(t, y)) : k()
        }
        function vt() {
            n(this).val("");
            k()
        }
        function ot(t, i) {
            var b = n(".selected", a), e, r, l, d, p, tt, it, c;
            b.length ? (e = b.text(), w = e, r = y.baseurl + b.data("link"), l = y.method && y.method.length ? y.method.toLowerCase() : "get", l == "custom" ? (i.val(e), w = null, k(), u.pub("myLocationSelected", {
                name: e,
                url: r,
                source: "as",
                event: t
            })) : l == "refresh" && y.selector ? (d = o.getParamsFromUrl(location.href), p = o.getParamsFromUrl(r), p = n.extend(!0, {}, d, p), i.val(e), w = null, k(), require(["refreshModules"], function(n) {
                var t = n(y.selector, p);
                t.done(function() {
                    u.pub("myLocationSelected", {
                        name: e,
                        url: r,
                        source: "as"
                    })
                })
            })) : l == "ajax" && y.selector ? (v.hide(), tt = o.getParamsFromUrl(r + "&pfr=1"), r = r.split("?")[0], ut({
                url: r,
                data: tt,
                method: "get",
                contentType: "application/json",
                refreshElement: y.refreshElement
            }).done(function(t) {
                if (w = null, k(), t.status == "fail")
                    nt();
                else {
                    i.val(e);
                    var r = n(y.selector);
                    r.length && t.length && r.replaceWith(t)
                }
            }).fail(function() {
                w = null;
                k();
                nt()
            })) : (i.val(e), location.href = r), t.preventDefault(), g()) : t.keyCode != f.Space && (w = i.val(), it = "", v.length && (c = v[0].action, c && (it = c && c.indexOf("://") >= 0 ? c : s.location.host + c)), h.trackEvent({
                type: "click",
                target: v
            }, "", "", "srchbtn"), t.stopPropagation(), t.preventDefault(), v.submit(), g())
        }
        function k() {
            a.empty();
            n(".nolocerr,.errmsg", v).hide();
            b = 0
        }
        function yt() {
            b && tt ? it() : g()
        }
        function g() {
            a && (a.hide(), n("li.selected", a).removeClass("selected"), d=!1)
        }
        function it() {
            a && (a.show(), d=!0)
        }
        function pt(t, i) {
            return n("<li>").text(t).data("link", i).on("mousedown click", function(t) {
                var i = n(this), r;
                i.hasClass("selected") || (r = v.find(p), n("li.selected", a).removeClass("selected"), i.addClass("selected"), h.trackEvent({
                    type: "click",
                    target: v
                }, "", "", "autosuggest"), ot(t, r))
            })
        }
        function wt(t, i) {
            var u, f;
            if (t && (u = n(t), u.length)) {
                f = n("<ul id='locList'><\/ul>");
                a.append(f);
                b = u.length;
                u.each(function(n) {
                    var i = u[n].state && u[n].name != u[n].state ? u[n].state + ", ": "", t = u[n].name;
                    t = t ? t + ", " : "";
                    f.append(pt(t + i + u[n].countryRegion, r(u[n], y)))
                });
                return 
            }
            i != "as" && n(".nolocerr", v).each(function() {
                var t = n(this);
                t.text(e(t.data("message"), w)).show()
            });
            b = 0
        }
        function st(n) {
            tt && (k(), n.status == "success" ? wt(n.result, n.mode) : nt(), yt())
        }
        function nt() {
            rt.trigger("clear");
            v.show();
            n(".errmsg", v).show()
        }
        var p = "input[name=q]", rt = n(p, t), y = n.extend({}, c, l, rt.data("auto-suggest-settings")), w, a = n("<div class='locResult' />"), v, b, d, tt, ut = n.ajax;
        return require(["ajaxWithAnimation"], function(n) {
            ut = n
        }), {
            setup: function() {
                t.on("keydown.autosuggest", p, ct).on("keyup.autosuggest", p, ht).on("change.autosuggest", p, et).on("focus.autosuggest", p, lt).on("blur.autosuggest", p, at).on("clear.autosuggest", p, vt);
                ft(n(p, t))
            },
            teardown: function() {
                t.off(".autosuggest", p);
                a && a.empty()
            }
        }
    }
    var c = {
        mode: "AS",
        method: "get",
        baseurl: "/en-us/weather/today",
        refreshElement: "#placesmodule",
        formcode: ""
    };
    return t(l, c)
});
define("searchLocation", ["jquery", "jqBehavior", "mediator"], function(n, t, i) {
    function r(t) {
        function r(i, r) {
            var e = i.data("url"), f;
            r && r.coords && (e += "?lat=" + r.coords.latitude + "&long=" + r.coords.longitude);
            f = n("input[name=q]", t);
            f.val("");
            f.focus();
            u(e);
            f.focus()
        }
        function u(t) {
            n.ajax({
                method: "get",
                url: t,
                contentType: "application/json"
            }).done(function(n) {
                i.pub("locSrchUpdate", n)
            }).fail(function(n, t, r) {
                i.pub("locSrchError", r)
            })
        }
        t.on("click", "div.weaaddlocation .gps", function(t) {
            var i = n(this);
            navigator.geolocation ? navigator.geolocation.getCurrentPosition(function(n) {
                r(i, n)
            }, function() {
                r(i)
            }) : r(n(this));
            t.stopPropagation();
            t.preventDefault()
        });
        t.on("submit", "div.weaaddlocation form", function(i) {
            var r = n("input[name=q]", t), f = r.val(), e;
            r.focus();
            f && f.length && (e = n(this).data("action") + "?q=" + encodeURIComponent(f), u(e));
            r.focus();
            i.stopPropagation();
            i.preventDefault()
        })
    }
    return t(r)
});
define("detectGpsLocation", ["jquery", "jqBehavior", "format", "c.deferred"], function(n, t, i) {
    var r = function(t, r) {
        function f(f) {
            f && f.coords && f.coords.latitude && f.coords.longitude ? location.href = i("{0}?lat={1}&long={2}", n(r.selector, t).data("url"), f.coords.latitude, f.coords.longitude) : u()
        }
        function u() {
            location.href = n(r.selector, t).attr("href")
        }
        t.delegate(r.selector, "click", function(n) {
            navigator.geolocation && (n.preventDefault(), navigator.geolocation.getCurrentPosition(f, u))
        })
    };
    return t(r, {
        selector: ".buttons .gps"
    })
});
define("ajaxWithAnimation", ["jquery", "c.dom"], function(n) {
    var i = "<div class='ajaxspinner'><\/div>", t = function(t, i) {
        var f = i;
        return typeof t == "object" && (f = t), f && f.refreshElement && r(f.refreshElement), n.ajax(t, i).done(function() {
            u(f.refreshElement)
        })
    }, r = function(t) {
        var r = n(t);
        r.length && r.addClass("ajaxloader").prepend(i)
    }, u = function(t) {
        var i = n(t);
        i.length && i.removeClass("ajaxloader").remove(".ajaxspinner")
    };
    return n.extend(n, {
        ajaxWithAnimation: t
    }), t
});
define("todayMiniModule", ["jquery", "jqBehavior", "navigation", "mediator", "refreshModules", "binding", "document", "window"], function(n, t, i, r, u, f, e, o) {
    function v(t) {
        function y() {
            t.on(s + ".editselection", l, function(i) {
                i.preventDefault();
                var u = n(this), f = n(h), r = n(c + " .weapopup", t);
                !f.length && r.length ? (r.detach(), r.appendTo(n("body")), define("todaydropdown_moved", 1)) : (r.length && r.remove(), p());
                tt();
                t.off(s + ".editselection", l);
                setTimeout(function() {
                    u.trigger("click")
                }, 10)
            })
        }
        function b(t) {
            t && t.keyCode == 27 && n(e).trigger("closepopup")
        }
        function g(t, i) {
            p();
            i(t);
            a || require(["todaydropdown_moved"], function() {
                n(h + " input[name=q]").focus()
            });
            n(e).on("keydown", b)
        }
        function nt(t, i) {
            i(t);
            n(e).off("keydown", b)
        }
        function tt() {
            var t = n(".weapopup");
            t && t.length && (t[0].behaviorConstructors = null);
            f("flyout", h).all({
                outsideEvent: "mousedown touchstart closepopup",
                triggerEvent: s,
                eventOrigin: ".weatodaymini>#edithome",
                toggleTargets: {
                    ".weatherdropdown": "show"
                },
                openFlyoutDelegate: g,
                closeFlyoutDelegate: nt
            })
        }
        var w = h + ".setdegree", o, v;
        n(e).delegate(w + " button", s, function(t) {
            t.preventDefault();
            var i = n(h), r = n("button[name=weaDegreeType]", i);
            r.removeClass("selected");
            n(this).addClass("selected");
            o = n(this).attr("value")
        });
        n(e).delegate(w, "submit", function(n) {
            n.preventDefault()
        });
        var k = function(n) {
            n && n.event && n.event.stopPropagation();
            v = n
        }, d = function() {
            var r, f;
            n(".weatherdropdown").removeClass("show");
            (v || o) && (r = v ? v.url + "&sethome=true" : n(c + " a", t).data("loc"), o && (r += "&weadegreetype=" + o), f = r.indexOf("?")!=-1 ? i.getParamsFromUrl(r) : i.getParams(r), u(c, f).done(function() {
                y()
            }))
        }, p = function() {
            var t = n(h), i = n(l), r, u;
            o = v = null;
            r = n("input[name=q]", t);
            r.attr("placeholder", i.data("home")).trigger("clear");
            u = n("button[name=weaDegreeType]", t);
            u.removeClass("selected");
            n("button[value='" + i.data("degreetype") + "']", t).addClass("selected")
        };
        require(["c.pdp"], function() {
            y()
        });
        require(["c.sso"], function() {
            y()
        });
        n(e).delegate(".weatherpopupcontent .donebutton", s, function() {
            d()
        });
        r.sub("myLocationSelected", k);
        n(e).delegate(".weatherpopupbg, .weatherpopupcontent .close", s, function() {
            p();
            n(e).trigger("closepopup")
        });
        y()
    }
    var s = "click", c = ".weatodaymini", l = c + ">#edithome", h = "body > .weapopup ", a = /(iPad|iPhone|iPod|Android)/g.test(o.navigator.userAgent);
    return t(v)
});
require(["jquery", "binding", "c.dom"], function(n, t) {
    require(["todaydropdown_moved"], function() {
        t("weaLocationAutosuggest", "body>.weapopup").all();
        t("searchLocation", "body>.weapopup").all();
        t("detectGpsLocation", "body>.weapopup .buttons").all()
    });
    t("todayMiniModule", "#main").all()
});
define("slideShowNavigation.tokens", {
    arrowLeftText: "Previous Slide",
    arrowRightText: "Next Slide",
    arrowRightNextGalleryText: "Next Gallery",
    playPauseText: "Pause/Rotate"
});
define("slideShowNavigation", ["jquery", "jqBehavior", "mediator", "slideShowNavUtils", "safeCss", "track", "tabKeyPressed", "window", "slideShowNavigation.tokens", "requestAnimationFrame", "dir.tokens"], function(n, t, i, r, u, f, e, o, s, h, c) {
    function tt(t, s) {
        function li(t) {
            var r = parseInt(n(".show").find(".count").html()), f = isNaN(r) ? 1: r, i = o.location.href.toLowerCase(), u = i.indexOf("#image=");
            return u >= 0 && (i = i.substring(0, u)), i + ("#image=" + (f + t))
        }
        function ct(n) {
            (s.autoRotate=!!n) ? ht = setTimeout(kt, s.autoRotateWait) : lt()
        }
        function fi() {
            st && (ni(t).css({
                padding: ""
            }), tt.resize(), yi())
        }
        function kt() {
            if (s.autoRotate) {
                var n = tt.current();
                (n >= oi - 1 || bi(n + 1)) && (ti=!0, i.pubChannel(l.animate, y, 1));
                ht = setTimeout(kt, s.autoRotateWait)
            }
        }
        function bi(n) {
            var t=!0, i = at.eq(n);
            return i.hasClass("hasimage") && (t=!i.find("img[data-src]").is(".loading")), t
        }
        function ai() {
            s.autoRotate && e() && lt()
        }
        function lt() {
            s.autoRotate=!1;
            clearTimeout(ht)
        }
        function vi(n) {
            n ? clearTimeout(ht) : s.autoRotate && (ht = setTimeout(kt, s.autoRotateWait))
        }
        function yi() {
            var n = (rt.height() - ft.height()) / 2;
            ot.css("top", n + "px");
            ft.css("top", n + "px");
            et && et.css("top", n + "px")
        }
        function dt() {
            var n = tt.hasNext(), t = tt.hasPrevious();
            if (!s.carousel&&!s.isThumbnail) {
                ki(n, t);
                return 
            }
            wt && t ? yt || (ot.removeClass(a), yt=!0) : yt && (ot.addClass(a), yt=!1);
            wt && n ? pt || (ft.removeClass(a), pt=!0) : pt && (ft.addClass(a), pt=!1);
            ui && clearTimeout(ui)
        }
        function ki(n, t) {
            n ? (ft.removeClass(a), et&&!et.hasClass(a) && et.addClass(a)) : (ft.addClass(a), et && et.removeClass(a));
            t ? ot.removeClass(a) : ot.addClass(a)
        }
        function pi(n) {
            gt( - 1, n)
        }
        function wi(n) {
            gt(1, n)
        }
        function gt(n, t) {
            var u, r, e;
            (n === 1 && tt.hasNext() || n===-1 && tt.hasPrevious()) && (u = location.href, s.isThumbnail || (r = this, t && (t.preventDefault(), t.stopImmediatePropagation(), r = t.target), e = s.carousel ? "" : li(n), f.trackEvent({
                type: hi,
                target: r
            }, null, e), s.carousel || (f.trackEvent({
                type: "unload",
                noSpin: 1
            }, o), i.pubChannel(l.afterUpdate, y, n))), tt.animate(n), ut.trackNewImpression(u))
        }
        var ni = u.createGroup(), at = t.children().filter(it), ei = n("a", at), rt = t.parent(), oi = at.length, si, vt=!s.autoRotate, ii, ri, yt, pt, wt, ui, bt, ut, tt;
        setTimeout(function() {
            vt=!s.autoRotate
        }, 0);
        var hi = s.carousel ? "click_nonnav": "click", ci=!s.carousel&&!s.isThumbnail, st = oi > 1;
        if (st) {
            var y = b++, ti, ht, ot = n(k).click(function(n) {
                gt( - 1, n)
            }).on(v, function() {
                e() && ut.showArrows()
            }).on(w, function() {
                e() && ut.hideArrows()
            }), ft = n(d).click(function(n) {
                gt(1, n)
            }).on(v, function() {
                e() && ut.showArrows()
            }).on(w, function() {
                e() && ut.hideArrows()
            }), et;
            s.isThumbnail || s.carousel || (ii = t.find("li.relatedgallery"), ri = ii.find("ul").first().attr("data-url"), ii.length && ri && (et = n(g).click(function() {
                f.trackEvent({
                    type: "click",
                    target: this
                }, "", "", "rightarrow-recircle");
                document.location = ri
            })));
            s.isThumbnail ? rt && (bt = rt.parent(), bt && bt.length && bt.append(ot).append(ft)) : (rt.append(ot).append(ft), et && rt.append(et));
            s.autoRotate && n(nt).click(function() {
                ct(!s.autoRotate)
            }).appendTo(rt);
            t.data(p, y);
            si=!0
        } else 
            t.removeClass("loading");
        return ut = {
            slides: function(n, t) {
                var h = location.href, r, u, e;
                t && (r = t.touch_gesture, t.touch_gesture = "swipe", u = s.carousel || s.isThumbnail ? "" : li(n), e = s.isThumbnail ? "click_nonnav" : hi, f.trackEvent({
                    type: e,
                    target: t,
                    noSpin: 1
                }, null, u), t.touch_gesture = r, ci && (f.trackEvent({
                    type: "unload",
                    noSpin: 1
                }, o), i.pubChannel(l.afterUpdate, y, n)));
                i.pubChannel(l.update, y, n);
                s.carousel || dt();
                t && ut.trackNewImpression(h);
                ti || lt();
                ti=!1;
                vt=!1
            },
            hideArrows: function() {
                vt || s.alwaysShowArrows || (wt=!1, h(dt))
            },
            showArrows: function() {
                wt=!0;
                h(dt)
            },
            trackNewImpression: function(n) {
                ci && (document.referrerOverride = n, f.trackPage(), i.pubChannel(l.thirdParty, y))
            }
        }, tt = r(st, s.carousel, s.isThumbnail, t, at, rt, ni, ut), tt.addHoverElements && s.addHoverSelector && tt.addHoverElements(n(s.addHoverSelector)), st && dt(), {
            setup: function() {
                if (!si)
                    return !1;
                if (st) {
                    if (t.addClass("loaded"), ot.show(), ft.show(), s.isThumbnail || yi(), ut.showArrows(), vt || (ui = setTimeout(ut.hideArrows, s.showArrowTime)), i.subChannel(l.realign, y, fi), i.subChannel(l.animate, y, tt.animate), i.subChannel(l.change, y, tt.change), i.subChannel("autoRotate", y, ct), i.subChannel(l.next, y, wi), i.subChannel(l.previous, y, pi), s.autoRotate) {
                        ht = setTimeout(kt, s.autoRotateWait);
                        rt.one("mousedown", lt);
                        ei.one(v, ai)
                    }
                    i.sub("fullscreen", vi);
                    i.sub("allAutoRotate", ct);
                    rt.on("keydown", function(n) {
                        return n.which == 37 ? (tt.animate(c.ltr?-1 : 1), !1) : n.which == 39 ? (tt.animate(c.ltr ? 1 : - 1), !1) : void 0
                    })
                }
                return tt.setup(), !0
            },
            teardown: function() {
                t.removeClass("loaded");
                ni.reset();
                st && (rt.off("keydown"), ut.hideArrows(), ot.hide(), ft.hide(), i.unsubChannel(l.realign, y, fi), i.unsubChannel(l.animate, y, tt.animate), i.unsubChannel(l.change, y, tt.change), i.unsubChannel(l.next, y, wi), i.unsubChannel(l.previous, y, pi), i.unsubChannel("autoRotate", y, ct), rt.off("mousedown", lt), ei.off(v, ai), i.unsub("fullscreen", vi), i.unsub("allAutoRotate", ct));
                tt.teardown()
            },
            update: fi
        }
    }
    function it() {
        return !n(this).hasClass("navtile")
    }
    var l = {
        animate: "slideShowNavAnimate",
        change: "slideShowNavChange",
        update: "slideShowNavUpdate",
        next: "slideShowNavNext",
        previous: "slideShowNavPrevious",
        afterUpdate: "slideShowNavAfterUpdate",
        thirdParty: "slideShowNavThirdParty",
        realign: "slideShowNavRealign",
        exitFullScreen: "exitFullScreen"
    }, p = "slideShowNavId", b = 1, v = "focus", w = "blur", k = '<button class="leftarrow fade" title="' + s.arrowLeftText + '">' + s.arrowLeftText + "<\/button>", d = '<button class="rightarrow fade" title="' + s.arrowRightText + '">' + s.arrowRightText + "<\/button>", g = '<button class="rightarrow fade" title="' + s.arrowRightNextGalleryText + '">' + s.arrowRightNextGalleryText + "<\/button>", nt = '<button class="playpause hide" title="' + s.playPauseText + '">' + s.playPauseText + "<\/button>", a = "fade", y = t(tt, {
        carousel: !0,
        autoRotate: !1,
        autoRotateWait: 5e3,
        alwaysShowArrows: !1,
        showArrowTime: 3500,
        isThumbnail: !1
    });
    return y.event = l, y.id = p, y
});
define("infopaneNavVNext", ["jquery", "jqBehavior", "slideShowNavigation", "mediator", "format", "tabKeyPressed", "document"], function(n, t, i, r, u, f, e) {
    function l() {
        return !n(this).hasClass("navtile")
    }
    function a(t, a) {
        function it(n) {
            var t = (v + n)%d;
            t < 0 && (t += d);
            y.eq(v).removeClass(a.selectedClass);
            k.eq(v).removeClass(a.selectedClass);
            y.eq(t).addClass(a.selectedClass);
            k.eq(t).addClass(a.selectedClass);
            v = t
        }
        function rt(i) {
            w = 0;
            i.keyCode == s && (g = n(i.target).closest(t).length, w += o);
            i.shiftKey && (w += c)
        }
        function ut() {
            f() && (b ? b=!1 : g ? w == o ? r.pubChannel(i.event.change, a.slideShowNavigationId, 1) : w == h && r.pubChannel(i.event.change, a.slideShowNavigationId, - 1) : (b=!0, y.eq(v).find("a").focus()))
        }
        function ft() {
            var i, r;
            for (p = n(u('<div class="slidecount {0}"/>', a.accentColor)), i = 0; i < y.length; i++)
                r = i ? "" : ' class="' + a.selectedClass + '"', p.append(u("<span{0}/>", r));
            k = p.children();
            t.after(p)
        }
        var v = 0, y = t.children().filter(l), d = y.length, p, b=!1, g=!1, k, w = 0, nt = n(e), tt = t.find("a");
        return {
            setup: function() {
                r.subChannel(i.event.update, a.slideShowNavigationId, it);
                p || ft();
                nt.on("keydown", rt);
                tt.on("focusin", ut)
            },
            teardown: function() {
                r.unsubChannel(i.event.update, a.slideShowNavigationId, it);
                nt.off("keydown", rt);
                tt.off("focusin", ut)
            }
        }
    }
    var s = 9, h = 3, c = 2, o = 1;
    return t(a, {
        selectedClass: "selected",
        accentColor: "pink"
    })
});
define("infopaneVNext", ["jquery", "jqBehavior", "safeCss", "slideShowNavigation", "infopaneNavVNext", "mediator", "imgSrc"], function(n, t, i, r, u, f, e) {
    function o(t, o) {
        function tt() {
            var t = a.children().filter(function() {
                return !n(this).hasClass("navtile")
            }), i = new Array(t.length);
            return t.each(function(t) {
                i[t] = n("img", this)
            }), i
        }
        function b(n) {
            var t = (v + n)%w;
            return t < 0 && (t += w), t
        }
        function k(n) {
            var i, t;
            for (v = b(n), i = n < 0?-1 : 1, t = 0; t <= 2; ++t)
                p[b(t * i)].each(function() {
                    e.go(this)
                })
        }
        function d() {
            c.find("." + l).removeClass(l);
            c.find("a[data-tab-id=" + v + "]").addClass(l)
        }
        function g(t) {
            var u = n(t.target).data("tab-id"), i = u - v;
            i != 0 && f.pubChannel(r.event.animate, h, i)
        }
        var l, c;
        o.tabbedInfopane && (l = "selected", c = n(".infopane-tabs li"), n.extend(o, {
            alwaysShowArrows: !1,
            autoRotate: !1,
            showArrows: !1,
            showDotPaginition: !1
        }));
        var nt = i.createGroup(), a = n("ul.swipenav", t), y = r(a, null, o), h = a.data(r.id), p = tt(), w = p.length, v = 0, s=!1;
        return !o.tabbedInfopane && o.showDotPaginition && (s = u(a, 0, n.extend({
            slideShowNavigationId: h
        }, o))), {
            setup: function() {
                if (!y.setup())
                    return t.addClass("invalid"), !1;
                if (f.subChannel(r.event.update, h, k), s && s.setup(), o.tabbedInfopane) {
                    f.subChannel(r.event.update, h, d);
                    c.on("click", g)
                }
                return !0
            },
            teardown: function() {
                nt.reset();
                y.teardown();
                s && s.teardown();
                f.unsubChannel(r.event.update, h, k);
                o.tabbedInfopane && (f.unsubChannel(r.event.update, h, d), c.off("click", g))
            },
            update: function() {
                y.update();
                s && s.update()
            }
        }
    }
    return t(o, {
        autoRotate: !0,
        showDotPaginition: !0
    })
});
require(["binding", "infopaneVNext", "c.deferred"], function(n) {
    n("infopaneVNext", ".ip").all()
});
define("slideShowNavUtils", ["jqueryPlus", "touchDataManager", "carouselManager", "mediator", "modernizr", "touchGestures", "dir.tokens", "requestAnimationFrame"], function(n, t, i, r, u, f, e, o) {
    var s = u.translateXCss, h = e.ltr ? 1: - 1, c = e.ltr ? 1: - 1;
    return function(u, l, a, v, y, p, w, b) {
        function pt(n) {
            (vt = n) ? (w(y).css("left", ""), ni()) : a || (w(y).css({
                top: "",
                width: ""
            }), w(v).css("width", ""))
        }
        function ni() {
            a || (w(v).css("width", 100 * y.length + "%"), w(y).css("width", 100 / y.length + "%"))
        }
        function ti(n) {
            if (u)
                n.on("gesture", {
                    direction: "horizontal"
                }, ii)
        }
        function ii(n) {
            if (ot && n.direction)
                switch (n.current.touchType) {
                case f.types.move:
                    st ? it.input(n.current.coord.x) : (it.reset(n.current.coord.x), wt(), gt = ht(), st=!0);
                    tt(v, gt + it.getDelta() | 0);
                    break;
                case f.types.swipeUp:
                case f.types.swipeDown:
                case f.types.throwUp:
                case f.types.throwDown:
                case f.types.cancel:
                case f.types.stop:
                    st=!1;
                    si(n.target);
                    break;
                case f.types.swipeRight:
                case f.types.swipeLeft:
                case f.types.throwRight:
                case f.types.throwLeft:
                    st=!1;
                    var t = n.current.touchType == f.types.swipeRight || n.current.touchType == f.types.throwRight, i = n.current.touchType == f.types.throwLeft || n.current.touchType == f.types.throwRight, r = i ? ri(): 0;
                    kt(r + (t?-h : h), n.target)
                }
        }
        function wt() {
            et && (v.stop(), et=!1)
        }
        function ei(n) {
            ft || o(function() {
                ft();
                ft = null
            });
            ft = n
        }
        function bt(n) {
            return ct(n) + c * (p.width() - n.width()) / 2
        }
        function oi(n) {
            if (n ||!a) {
                var i, t = k.getItem(0);
                i = a ? ct(t) : bt(t);
                typeof n != "undefined" ? ei(function() {
                    it.ensureJQueryEase();
                    et=!0;
                    var n = {};
                    n[s] = i;
                    v.stop().animate(n, {
                        duration: it.getDuration(),
                        easing: "cubicBezierQuint",
                        step: function(n) {
                            var i = bt(t), r = ht();
                            return (i | 0) != (n.endValue | 0) && (n.startValue = r - t.width(), n.endValue = i), n
                        },
                        complete: function() {
                            et=!1;
                            (ht() | 0) != (bt(t) | 0) && rt()
                        }
                    })
                }) : v.css(s) !== i && tt(v, i)
            }
        }
        function si(n) {
            kt(ri(), n)
        }
        function ri() {
            for (var u, t, i = 0, r = Infinity, f = ht(), n =- yt; n <= dt; n++)
                u = ct(k.getItem(n)), t = Math.abs(u - f), t < r && (r = t, i = n);
            return i
        }
        function ht() {
            return tt(v)
        }
        function tt(n, t) {
            return isNaN(n.leftRelSlider) && (n.leftRelSlider = 0), t == undefined ? t = parseFloat(n.css(s)) : w(n).css(s, t - n.leftRelSlider), t + n.leftRelSlider
        }
        function kt(n, t) {
            var i, r, u;
            (l ? d = (d + n + nt)%nt : (i = d + n, i < 0 ? i = 0 : i >= nt && (i = nt - 1), i != d + n && (n = i - d), d = i), k.changeIndex(n), r = ct(k.getItem(0)), u = tt(v), r != u) && (rt(n), b.slides(n, t))
        }
        function ct(n) {
            return l ? (p.width() - n.outerWidth(!0)) / 2 - tt(n) - .5 | 0 : - n.leftRelSlider
        }
        function ui() {
            rt()
        }
        function rt(n) {
            var u, r, f, t, i;
            if (ot) {
                for (u = n && tt(k.getItem( - n)), wt(), r = k.getItem( - yt).outerWidth(!0) / 2, f = lt(y.eq(0)), t =- yt; t <= dt; t++)
                    i = k.getItem(t), i.leftRelSlider = lt(i) - f, r += i.outerWidth(!0);
                vt || a || w(v).css("width", r);
                oi(n, l && n && tt(k.getItem( - n)) - u)
            }
        }
        function lt(n) {
            return n.offset().left + (e.ltr ? 0 : n.outerWidth(!0))
        }
        var g = p, nt = y.length, k = new i, d = 0, at = n("body"), vt = (at.hasClass("gfp") || at.hasClass("afp")) && at.hasClass("fullscreen"), ut = f.settings.touchSupported, ft, yt = Math.floor((nt - 1) / 2), dt = Math.floor(nt / 2), fi, et, ot, it = new t, st, gt;
        return ut && ti(g), setTimeout(function() {
            vt && ni();
            r.sub("fullscreen", pt)
        }, 100), {
            setup: function() {
                var f, e, t, i;
                for (a || w(v).css("overflow", "hidden"), ut || w(y).css("position", "relative"), f = [], e = lt(y.eq(0)), t = 0; t < nt; ++t)
                    i = y.eq(t), i.leftRelSlider = lt(i) - e, f[t] = i;
                k.load(f);
                g.on("mouseover", b.showArrows);
                g.on("mouseout", b.hideArrows);
                if (!l)
                    n("img", p).on("load", ui);
                r.sub("fullscreen", pt);
                u && (ot=!0, fi = 0, rt())
            },
            teardown: function() {
                ot=!1;
                ut && (wt(), tt(v, 0));
                g.off("mouseover", b.showArrows);
                g.off("mouseout", b.hideArrows);
                g.off("gesture", {
                    direction: "horizontal"
                }, ii);
                l || n("img", p).off("load", ui);
                r.unsub("fullscreen", pt)
            },
            animate: kt,
            change: function(n) {
                k.changeIndex(n);
                d += n;
                b.slides(n)
            },
            resize: rt,
            current: function() {
                return d
            },
            hasNext: function() {
                return l || d < nt - 1
            },
            hasPrevious: function() {
                return l || d > 0
            },
            addHoverElements: function(n) {
                ut && ti(n);
                g = g.add(n)
            }
        }
    }
});
define("financeHomepageAutoSuggest", ["jquery", "jqBehavior", "keyCode", "homepageFinanceIndices", "track"], function(n, t, i, r, u) {
    function f(t, f) {
        function ot() {
            var t = n(".selected", e)[0] || n("li", e)[0];
            t && y(t)
        }
        function st(t) {
            var i = n("input[type='search']", t.target);
            i.val() != null && i.val().trim().length == 0 && t.preventDefault()
        }
        function ht(r) {
            var u, f, o;
            switch (r.keyCode) {
            case i.Arrow.Down:
                s && (h ? (u = n("li.selected", e).removeClass("selected"), f = u.next(), f[0] ? f.addClass("selected") : n("li", e).first().addClass("selected")) : (b(), c()));
                break;
            case i.Arrow.Up:
                s && (h ? (u = n("li.selected", e).removeClass("selected"), o = u.prev(), o[0] ? o.addClass("selected") : n("li", e).last().addClass("selected")) : (b(), c()));
                break;
            case i.Enter:
                y();
                break;
            case i.Esc:
                h && (r.preventDefault(), l());
                c();
                t.blur()
            }
        }
        function ct() {
            k=!0;
            v();
            s > 0 && (t.select(), b(), c())
        }
        function lt() {
            k=!1;
            h && (n(".selected", e).length > 0 ? y() : l());
            c()
        }
        function v() {
            var i = n.trim(t.val());
            i ? i != it && (it = i, dt(i)) : (yt(), pt(), l(), et(), ft(), s = 0)
        }
        function y(i, r) {
            g[a] != undefined && function() {
                var o = i;
                g[a].done(function() {
                    var s = o || n(".selected", e)[0] || n("li", e)[0], i, h;
                    if (s)
                        if (f.isCallBackProvided)
                            i = {}, i.instrumentSymbol = n(s).find("a")[0].getAttribute("instrumentSymbol"), i.instrumentType = n(s).find("a")[0].getAttribute("instrumentSymbolType"), i.instrumentTicker = n(s).find("a")[0].getAttribute("instrumentTicker"), i.instrumentFullname = n(s).find("a")[0].getAttribute("instrumentFullname"), i.instrumentMarket = n(s).find("a")[0].getAttribute("instrumentMarket"), i.KeyUsed = r, f.callback(i);
                        else 
                            return h = n(s).find("a")[0].href, u.trackEvent({
                                type: "submit",
                                target: t
                            }, null, h), window.location.href = h, !1;
                    return l(), !1
                })
            }()
        }
        function at(t) {
            var i = t || n(".selected", o)[0];
            return n(i).href == undefined?!1 : (window.location.href = n(i).href, !1)
        }
        function dt(t) {
            var e = f.count ? f.count: 6, i =+ new Date, u;
            a = i;
            yt();
            u = n.ajax({
                url: f.endpoint,
                dataType: "jsonp",
                cache: !0,
                data: {
                    q: t,
                    locale: r.market.replace("-", ":"),
                    count: e
                },
                jsonpCallback: "cb" + tt++
            });
            u.fail(function() {
                i == a && l()
            }).done(function(n) {
                n && i == a && (pt(), n = n.data, gt(n, t), s = n.length, ti())
            });
            g[i] = u
        }
        function gt(i, u) {
            for (var a, l, h = n("<ul><\/ul>"), c = 0; c < i.length; c++) {
                var f = i[c], s = {
                    label: f.OS001 + "." + f.OS01W + "." + f.OS010 + "." + f.AC040,
                    displaySymbol: f.OS001,
                    companyName: ni(f, r.market),
                    symbolType: f.OS010,
                    exchangeName: f.AC040,
                    value: f.OS001
                }, o = n("<a><\/a>");
                o.attr("href", si(f));
                o.attr("instrumentSymbol", wt(f));
                o.attr("instrumentTicker", s.displaySymbol);
                o.attr("instrumentFullname", s.companyName);
                o.attr("instrumentMarket", f.RT0EC);
                a = p(f.OS010);
                o.attr("instrumentSymbolType", a);
                o.html('<span class="fi-as-sym">' + vt(s.displaySymbol, u) + '<\/span><span class="fi-as-cn">' + vt(s.companyName, u) + '<\/span><span class="fi-as-en">' + s.exchangeName + "<\/span>");
                l = n("<li><\/li>");
                l.append(o);
                h.append(l)
            }
            h.width(t.width);
            e.append(h)
        }
        function ni(n, t) {
            var r = n.OS01W, i = n.OS01V, e = t ? t.split("-")[0]: null, u, f;
            return i && (i = i.toLowerCase(), e !== "en" && (i === t || i === e) && (u = n.OS0LN, f = n.RT0SN, r = u ? u : f ? f : r)), r
        }
        function vt(n, t) {
            var i = t.toLowerCase(), r = t.toUpperCase();
            return t = n.indexOf(i) >= 0 ? i : r, n.replace(t, "<b>" + t + "<\/b>")
        }
        function yt() {
            w && w.val() && w.val("")
        }
        function pt() {
            e || ii();
            e.empty();
            s = 0
        }
        function ti() {
            s ? k ? (b(), c()) : y() : (l(), et(), ft())
        }
        function l() {
            e && (e.hide(), e.addClass("hideautosuggest"), h=!1)
        }
        function b() {
            e && (e.show(), e.removeClass("hideautosuggest"), h=!0)
        }
        function ii() {
            e = n('<div class="' + rt + '">').on("mousedown", function(t) {
                var i, r;
                t.preventDefault();
                i = n(t.target).closest("li");
                i[0] ? (i.hasClass("selected") || (n(".selected", e).removeClass("selected"), i.addClass("selected")), y()) : (r = n(t.target).closest("a"), at(r))
            });
            t.before(e)
        }
        function ri(t) {
            return n.inArray(t, ["ST"])!=-1
        }
        function ui(t) {
            return n.inArray(t, ["FE", "CE"])!=-1
        }
        function fi(t) {
            return n.inArray(t, ["FO", "FC"])!=-1
        }
        function nt(t) {
            return n.inArray(t, ["XI"])!=-1
        }
        function ei(t) {
            return n.inArray(t, ["CUR"])!=-1
        }
        function oi(t) {
            return n.inArray(t, ["COM"])!=-1
        }
        function si(n) {
            var i = wt(n), u = p(n.OS010), t = null;
            return u === "Stock" && (t = r.stockdetailsurl + "/fi-" + i + "?symbol=" + n.OS001 + "&form=" + r.FORMCode), u === "Index" && (t = r.indexdetailsurl + "/fi-" + i + "?symbol=" + n.OS001 + "&form=" + r.FORMCode), u === "Etf" && (t = r.etfdetailsurl + "/fi-" + i + "?symbol=" + n.OS001 + "&form=" + r.FORMCode), u === "Fund" && (t = r.funddetailsurl + "/fi-" + i + "?symbol=" + n.OS001 + "&form=" + r.FORMCode), u === "Currency" && (t = r.currencyConverterUrl + "/fi-" + i + "?symbol=" + n.OS001 + "&form=" + r.FORMCode), u === "Commodity" && (t = r.commoditydetailsurl + "/fi-" + i + "?symbol=" + n.OS001 + "&form=" + r.FORMCode), t
        }
        function p(n) {
            var t = "";
            return n && (t = fi(n) ? "Fund" : ri(n) ? "Stock" : ui(n) ? "Etf" : nt(n) ? "Index" : ei(n) ? "Currency" : oi(n) ? "Commodity" : ""), t
        }
        function wt(n) {
            var t = {}, o, s, f;
            if (n) {
                var i = n.OS010, r = n.OS001, h = n.LS01Z, e = n.RT00E, u = "";
                if (p(i) == "Commodity")
                    return t = n.RT00E + "." + n.RT00T + "." + n.RT00S, bt(t);
                if (p(i) == "Currency")
                    return n.OS001.substring(0, 3) + "-" + n.OS001.substring(3, 6) + "-1.0000";
                if (p(i) == "Fund")
                    return n.SecId;
                nt(i) ? (o = "10", u = o) : (s = "1", u = n.RT00T || s);
                f = n.RT00S ? n.RT00S : r ? r : "";
                nt(i) ? t = hi({
                    E1: e,
                    Sym: f,
                    St: u,
                    E2: e,
                    Eqsm: r
                }) : (t = e + "." + u + "." + f + "." + h, r !== f && (t += "." + r))
            }
            return bt(t)
        }
        function bt(n) {
            return n == null || typeof n != "string" ? "" : (n = n.replace(/\//g, "|SLA|"), n = n.replace(/</g, "|LETHA|"), n = n.replace(/>/g, "|GRETHA|"), n = n.replace(/\*/g, "|ASTER|"), n = n.replace(/%/g, "|PERCENT|"), n = n.replace(/:/g, "|SEPA|"), n = n.replace(/&/g, "|AMP|"), n = n.replace("/\\/g", "|BKSLA|"), encodeURIComponent(n))
        }
        function hi(n) {
            var t = "";
            return n && (t = n.Eqsm && n.Sym && n.Sym !== n.Eqsm ? n.E1 + "." + n.St + "." + n.Sym + "." + n.E2 + "." + n.Eqsm : n.E1 + "." + n.St + "." + n.Sym), t
        }
        function kt(n, t) {
            if (t) {
                var i = n && n.indexOf("?")!==-1, r = i ? "&": "?";
                n += r + t
            }
            return n
        }
        var a, tt, it, e, s, h, k, w, rt, o;
        f.autoSuggestSettings && (f = f.autoSuggestSettings);
        tt = 1;
        rt = f.cssclass || "autosuggest";
        f.endpoint = f.endpoint || "//finance.services.appex.bing.com/Market.svc/MTAutocomplete";
        var d = f.recentQuotesRequired ||!1, ut=!1, g = [];
        n(function() {
            t.data("holder", t.attr("placeholder"));
            var n;
            t.focusin(function() {
                t.attr("placeholder", "");
                n && n.length ? t.val(n) : t.val("")
            });
            t.focusout(function() {
                n = t.val();
                t.attr("placeholder", t.data("holder"))
            })
        });
        o = n('<div class="recent-quotes autosuggest"><\/div>').on("mousedown", function(t) {
            t.preventDefault ? t.preventDefault() : t.returnValue=!1;
            var i = n(t.target).closest("a");
            at(i)
        });
        t.before(o);
        var c = d==!0 ? function() {
            o && (o.hide(), o.addClass("hideautosuggest"), ut=!1)
        }
        : function() {}, ft = d==!0 ? function() {
            o && (o.show(), o.removeClass("hideautosuggest"), ut=!0)
        }
        : function() {}, et = d==!0 ? function() {
            var t = r.recentquotesurl, i = f.stocksInNewsRequired ? kt(t, "backfillwithstocksinnews=true"): t;
            n.ajax(kt(i, "form=RecentQuoteFormCode")).done(function(n) {
                o.empty();
                o.html(n)
            }).fail(function() {})
        }
        : function() {};
        if (f.endpoint)
            return f.codefield && (w = n("#" + f.codefield)), {
                setup: function() {
                    t.on("keydown", ht).on("keyup", v).on("change", v).on("focus", ct).on("blur", lt);
                    t.siblings("span").on("click", ot);
                    t.closest("form").on("submit", st)
                },
                teardown: function() {
                    t.off("keydown", ht).off("keyup", v).off("change", v).off("focus", ct).off("blur", lt);
                    t.closest("form").off("submit", st);
                    t.siblings("button").off("click", ot);
                    e && (e.remove(), e = null)
                }
            }
    }
    return t(f)
});
define("keyCode", function() {
    return {
        Back: 8,
        Tab: 9,
        Enter: 13,
        Shift: 16,
        Ctrl: 17,
        Alt: 18,
        Break: 19,
        CapsLock: 20,
        Esc: 27,
        Space: 32,
        Page: {
            Up: 33,
            Down: 34
        },
        End: 35,
        Home: 36,
        Arrow: {
            Left: 37,
            Up: 38,
            Right: 39,
            Down: 40
        },
        Print: 44,
        Insert: 45,
        Delete: 46,
        Colon2: 59,
        Equals2: 61,
        Equals3: 107,
        Minus2: 109,
        Period: 190,
        Windows: {
            Left: 91,
            Right: 92,
            Opera: 219
        },
        Menu: 93,
        NumPad: {
            0: 96,
            1: 97,
            2: 98,
            3: 99,
            4: 100,
            5: 101,
            6: 102,
            7: 103,
            8: 104,
            9: 105,
            Multiply: 106,
            Plus: 107,
            Minus: 109,
            Dot: 110,
            Divide: 111
        },
        Function: {
            1: 112,
            2: 113,
            3: 114,
            4: 115,
            5: 116,
            6: 117,
            7: 118,
            8: 119,
            9: 120,
            10: 121,
            11: 122,
            12: 123
        },
        Lock: {
            Caps: 20,
            Num: 144,
            Scroll: 145
        },
        Colon: 186,
        Equals: 187,
        Comma: 188,
        Minus: 189,
        Slash: {
            Forward: 191,
            Back: 220
        },
        Tilde: 192,
        Bracket: {
            Open: 219,
            Close: 221
        },
        Quote: 222
    }
});
require(["binding", "c.deferred"], function(n) {
    n("financeHomepageAutoSuggest", "#finance-autosuggest").all({
        recentQuotesRequired: !0,
        stocksInNewsRequired: !0
    })
})
