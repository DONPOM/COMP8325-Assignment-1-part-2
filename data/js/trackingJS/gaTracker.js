!function() {
    var k;
    k = {
        debug: !1,
        chat: {
            cookie: "_jimdoge_chat",
            domain: ".jimdo.com"
        },
        jLang: {
            referrerCookie: "jLangRef",
            cookie: "jLang",
            domain: ".jimdo.com"
        },
        affiliate: {
            endpoint: "//a.jimdo.com/app/web/affiliate/type",
            timeout: 5E3
        },
        overrideReferrer: {
            cookie: "_j_or_ref",
            domain: ".jimdo.com"
        },
        jimdoLocator: {
            LOCAL_HOSTNAMES: "www.jimdo.com de.jimdo.com fr.jimdo.com cn.jimdo.com it.jimdo.com ru.jimdo.com es.jimdo.com jp.jimdo.com nl.jimdo.com br.jimdo.com pl.jimdo.com tr.jimdo.com".split(" "),
            SUPPORT_HOSTNAMES: "support.jimdo.com hilfe.jimdo.com aide.jimdo.com cn-help.jimdo.com aiuto.jimdo.com ru-help.jimdo.com ayuda.jimdo.com jp-help.jimdo.com nl-help.jimdo.com ajuda.jimdo.com pomoc.jimdo.com destek.jimdo.com".split(" "),
            OTHERS: ["a.jimdo.com", "blog.jimdo.com", "partner.jimdo.com", /www\d{1,3}\.jimdo\.com/, "seminarios.jimdo.com", "educacao.jimdo.com"]
        },
        uaTracker: {
            settings: {
                globalVar: "jimBob",
                endpoint: "//www.google-analytics.com/analytics.js",
                globalHostnames: ["a.jimdo.com"],
                debugTrackingId: "UA-47302067-26"
            },
            initDefaults: {
                alwaysSendReferrer: !0,
                cookieDomain: "jimdo.com",
                allowAnchor: !0,
                anonymizeIp: !0,
                siteSpeedSampleRate: 50
            },
            trackers: [{
                settings: {
                    name: "global",
                    trackingId: "UA-47302067-1",
                    cookieName: "_jimBobGlobal"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /.*/
                }
            }, {
                settings: {
                    name: "en",
                    trackingId: "UA-47302067-4",
                    cookieName: "_jimBobEn"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^en/i
                }
            }, {
                settings: {
                    name: "enSupport",
                    trackingId: "UA-47302067-5",
                    cookieName: "_jimBobEnSup"
                },
                matcher: {
                    hostname: "support.jimdo.com",
                    lang: /^en/i
                }
            }, {
                settings: {
                    name: "enBlog",
                    trackingId: "UA-47302067-27",
                    cookieName: "_jimBobEnBlog"
                },
                matcher: {
                    hostname: "blog.jimdo.com",
                    lang: /^en/i
                }
            }, {
                settings: {
                    name: "de",
                    trackingId: "UA-47302067-2",
                    cookieName: "_jimBobDe"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^de/i
                }
            }, {
                settings: {
                    name: "deSupport",
                    trackingId: "UA-47302067-3",
                    cookieName: "_jimBobDeSup"
                },
                matcher: {
                    hostname: "hilfe.jimdo.com",
                    lang: /^de/i
                }
            }, {
                settings: {
                    name: "dePartner",
                    trackingId: "UA-47302067-29",
                    cookieName: "_jimBobDePartner"
                },
                matcher: {
                    hostname: "partner.jimdo.com",
                    lang: /^de/i
                }
            }, {
                settings: {
                    name: "fr",
                    trackingId: "UA-47302067-6",
                    cookieName: "_jimBobFr"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^fr/i
                }
            }, {
                settings: {
                    name: "frSupport",
                    trackingId: "UA-47302067-7",
                    cookieName: "_jimBobFrSup"
                },
                matcher: {
                    hostname: "aide.jimdo.com",
                    lang: /^fr/i
                }
            }, {
                settings: {
                    name: "cn",
                    trackingId: "UA-47302067-8",
                    cookieName: "_jimBobCn"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^zh/i
                }
            }, {
                settings: {
                    name: "cnSupport",
                    trackingId: "UA-47302067-9",
                    cookieName: "_jimBobCnSup"
                },
                matcher: {
                    hostname: "cn-help.jimdo.com",
                    lang: /^zh/i
                }
            }, {
                settings: {
                    name: "it",
                    trackingId: "UA-47302067-10",
                    cookieName: "_jimBobIt"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^it/
                }
            }, {
                settings: {
                    name: "itSupport",
                    trackingId: "UA-47302067-11",
                    cookieName: "_jimBobItSup"
                },
                matcher: {
                    hostname: "aiuto.jimdo.com",
                    lang: /^it/
                }
            }, {
                settings: {
                    name: "ru",
                    trackingId: "UA-47302067-12",
                    cookieName: "_jimBobRu"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^ru/i
                }
            }, {
                settings: {
                    name: "ruSupport",
                    trackingId: "UA-47302067-13",
                    cookieName: "_jimBobRuSup"
                },
                matcher: {
                    hostname: "ru-help.jimdo.com",
                    lang: /^ru/i
                }
            }, {
                settings: {
                    name: "es",
                    trackingId: "UA-47302067-14",
                    cookieName: "_jimBobEs"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^es/i
                }
            }, {
                settings: {
                    name: "esSupport",
                    trackingId: "UA-47302067-15",
                    cookieName: "_jimBobEsSup"
                },
                matcher: {
                    hostname: "ayuda.jimdo.com",
                    lang: /^es/i
                }
            }, {
                settings: {
                    name: "esEducation",
                    trackingId: "UA-47302067-37",
                    cookieName: "_jimBobEsEdu"
                },
                matcher: {
                    hostname: "seminarios.jimdo.com",
                    lang: /^es/i
                }
            }, {
                settings: {
                    name: "jp",
                    trackingId: "UA-47302067-16",
                    cookieName: "_jimBobJp"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^ja/i
                }
            }, {
                settings: {
                    name: "jpSupport",
                    trackingId: "UA-47302067-17",
                    cookieName: "_jimBobJpSup"
                },
                matcher: {
                    hostname: "jp-help.jimdo.com",
                    lang: /^ja/i
                }
            }, {
                settings: {
                    name: "nl",
                    trackingId: "UA-47302067-18",
                    cookieName: "_jimBobNl"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^nl/i
                }
            }, {
                settings: {
                    name: "nlSupport",
                    trackingId: "UA-47302067-19",
                    cookieName: "_jimBobNlSup"
                },
                matcher: {
                    hostname: "nl-help.jimdo.com",
                    lang: /^nl/i
                }
            }, {
                settings: {
                    name: "pt",
                    trackingId: "UA-47302067-20",
                    cookieName: "_jimBobPt"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^pt/i
                }
            }, {
                settings: {
                    name: "ptSupport",
                    trackingId: "UA-47302067-21",
                    cookieName: "_jimBobPtSup"
                },
                matcher: {
                    hostname: "ajuda.jimdo.com",
                    lang: /^pt/i
                }
            }, {
                settings: {
                    name: "ptEducation",
                    trackingId: "UA-47302067-36",
                    cookieName: "_jimBobPtEdu"
                },
                matcher: {
                    hostname: "educacao.jimdo.com",
                    lang: /^pt/i
                }
            }, {
                settings: {
                    name: "pl",
                    trackingId: "UA-47302067-22",
                    cookieName: "_jimBobPl"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^pl/i
                }
            },
            {
                settings: {
                    name: "plSupport",
                    trackingId: "UA-47302067-23",
                    cookieName: "_jimBobPlSup"
                },
                matcher: {
                    hostname: "pomoc.jimdo.com",
                    lang: /^pl/i
                }
            }, {
                settings: {
                    name: "tr",
                    trackingId: "UA-47302067-24",
                    cookieName: "_jimBobTr"
                },
                matcher: {
                    hostname: /.*/,
                    lang: /^tr/i
                }
            }, {
                settings: {
                    name: "trSupport",
                    trackingId: "UA-47302067-25",
                    cookieName: "_jimBobTrSup"
                },
                matcher: {
                    hostname: "destek.jimdo.com",
                    lang: /^tr/i
                }
            }
            ]
        }
    };
    var q, m, s, r, w, x, t, n, y;
    t = function(a, b, d, c) {
        var f;
        null == d && (d=!1);
        null == c && (c = 500);
        f = setInterval(function() {
            if (a())
                return d ||
                clearInterval(f), b()
        }, c);
        return function() {
            return clearInterval(f)
        }
    };
    q = function(a, b, d) {
        var c, f, e, h, g, l, k;
        0 === a.length && d();
        c = a.length;
        h = 0;
        f = function() {
            h += 1;
            if (h === c)
                return d()
        };
        k = [];
        g = 0;
        for (l = a.length; g < l; g++)
            e = a[g], k.push(b(f, e, a));
        return k
    };
    r = function(a, b, d) {
        var c, f;
        null == d && (d=!1);
        for (c in b)
            f = b[c], d ? a[c] = f : null == a[c] && (a[c] = f);
        return a
    };
    m = function(a, b, d) {
        var c, f, e, h, g;
        c = decodeURIComponent;
        f = encodeURIComponent;
        if (null == b)
            return a = c(a), a = document.cookie.match(RegExp("(?:^|; )" + a + "\x3d([^;]+)(?:;|$)")) ||
            ["", ""], c(a[1]);
        d = null != d ? d : {};
        e = d.expires;
        h = d.path;
        c = d.domain;
        g = d.secure;
        null != e && e ===+ e ? (d = new Date, d.setTime( + d + 1E3 * e), e = "; expires\x3d" + d.toUTCString()) : e = "";
        h = "; path\x3d" + (h ? h : "/");
        c = null != c ? "; domain\x3d" + c : "";
        g = null != g ? "; secure" : "";
        return document.cookie = "" + f(a) + "\x3d" + f(b) + e + h + c + g
    };
    n = function(a) {
        var b;
        b = {};
        ("" + String(a).replace(/^[^\?\#$]*(\?|\#|$)/, "").replace("#", "\x26").replace(/^\s*|\s*$/g, "") + "\x26").replace(/([^\&]+)&/g, function(a, c) {
            var f;
            f = c.split("\x3d");
            return b[decodeURIComponent(f[0])] =
            f[1] && decodeURIComponent(f[1])
        });
        return b
    };
    n = function(a, b) {
        null == b && (b = function(a) {
            return a
        });
        return function() {
            var d, c;
            d = Array.prototype.slice.call(arguments);
            c = b(d);
            return a[c] = a.hasOwnProperty(c) ? a[c] : a.apply(a, d)
        }
    }(n);
    x = function(a, b, d, c) {
        var f, e, h, g, l, k, m, n;
        null == c && (c = "callback");
        m = document.createElement("script");
        g = document.getElementsByTagName("script")[0];
        k = "?";
        h = encodeURIComponent;
        e = "cb" + + new Date + (0 | 1E7 * Math.random()).toString(32);
        f=!1;
        window[e] = function() {
            if (!f)
                return f=!0, d.apply(d, [].slice.call(arguments)),
                m.parentNode.removeChild(m), delete window[e]
        };
        for (l in b)
            n = b[l], k += "" + h(l) + "\x3d" + h(n) + "\x26";
        m.src = "" + a + (k + ("" + c + "\x3d" + e));
        g.parentNode.insertBefore(m, g);
        return e
    };
    s = document.location.hash;
    w = function(a) {
        "addEventListener"in window && "onhashchange"in window ? window.addEventListener("hashchange", a, !1) : t(function() {
            var a;
            a = document.location.hash;
            return a !== s ? (s = a, !0) : !1
        }, a, !0);
        return null
    };
    y = function(a) {
        return Object.prototype.toString.call(a)
    };
    (function() {
        function a(a, d, c, f, e) {
            var h;
            this.path = null !=
            a ? a : location.pathname;
            this.search = null != d ? d : location.search;
            this.hash = null != c ? c : location.hash;
            this.title = null != f ? f : document.title;
            this.stateObj = null != e ? e : {};
            this.supportsReplaceState = null != (null != (h = window.history) ? h.replaceState : void 0) && "function" === typeof window.history.replaceState
        }
        a.prototype.getCleanPath = function(a) {
            var d, c, f, e, h, g, l;
            h = [];
            c = [];
            if (null != a)
                for (g = 0, l = a.length; g < l; g++)
                    f = a[g], e = this.search.indexOf("" + f + "\x3d"), 0 < e && (d = this.search.indexOf("\x26", e), - 1 === d && (d = this.search.length),
                    h.push(this.search.substring(e, d))), e = this.hash.indexOf("" + f + "\x3d"), 0 < e && (d = this.hash.indexOf("\x26", e), - 1 === d && (d = this.hash.length), c.push(this.hash.substring(e, d)));
            a = this.path;
            0 < h.length && (a += "?" + h.join("\x26"));
            0 < c.length && (a += "#" + c.join("\x26"));
            return a
        };
        a.prototype.objToParams = function(a) {
            var d, c, f, e;
            f = [];
            d = encodeURIComponent;
            for (c in a)
                e = a[c], f.push("" + d(c) + "\x3d" + d(e));
            return f.join("\x26")
        };
        a.prototype.addParams = function(a, d) {
            var c, f;
            null == d && (d = void 0);
            f = this.getCleanPath(d);
            a = this.objToParams(a);
            c =- 1 < f.indexOf("#") ? "\x26" : "#";
            return this.replaceState(f + c + a)
        };
        a.prototype.replaceState = function(a) {
            if (this.supportsReplaceState)
                return window.history.replaceState(this.stateObj, this.title, a)
        };
        return a
    })();
    var u;
    u = function() {
        function a(a, c, f) {
            this.options = a;
            this.langMatch = null != c ? c : null;
            this.hostnameMatch = null != f ? f : null;
            this.requiredEcommerce = this.initialized=!1;
            this.id = this.options.trackingId
        }
        var b;
        b = function() {};
        a.GA_NAME = "GoogleAnalyticsObject";
        a.GA_NOT_FOUND_ERROR = Error("The Google Analytics Object could not be found in the global Namespace");
        a.getGAObject = function() {
            return window[window[a.GA_NAME]]
        };
        a.prototype.isSupportedLanguage = function(a) {
            return this.matches(this.langMatch, a)
        };
        a.prototype.isSupportedHostname = function(a) {
            return this.matches(this.hostnameMatch, a)
        };
        a.prototype.matches = function(a, c) {
            if (null == c)
                return !1;
            switch (Object.prototype.toString.call(a)) {
            case "[object RegExp]":
                return a.test(c);
            case "[object String]":
                return a === c;
            default:
                return !1
            }
        };
        a.prototype.getGATracker = function(d) {
            var c, f, e;
            null == d && (d = b);
            if (this.initialized)
                return d(null,
                this.rawTracker);
            e = this;
            f = this.options;
            c = a.getGAObject();
            return null != c ? c(function() {
                var c, b;
                c = a.getGAObject();
                b = c.getByName(f.name);
                null == b && (c("create", f), c("" + f.name + ".require", "displayfeatures"), b = c.getByName(f.name));
                e.rawTracker = b;
                e.initialized=!0;
                return d(null, b)
            }) : d(a.GA_NOT_FOUND_ERROR)
        };
        a.prototype.set = function(a, c) {
            null == c && (c = b);
            return this.getGATracker(function(f, e) {
                null == f && e.set(a);
                return c(f)
            })
        };
        a.prototype.get = function(a, c) {
            null == c && (c = b);
            return this.getGATracker(function(f, e) {
                var b;
                null == f && (b = e.get(a));
                return c(f, b)
            })
        };
        a.prototype.send = function(a, c) {
            null == c && (c = b);
            return this.getGATracker(function(f, b) {
                null == f && b.send(a);
                return c(f)
            })
        };
        a.prototype.requireEcommerce = function(d) {
            var c, f;
            null == d && (d = b);
            f = this;
            c = this.options.name;
            return this.requiredEcommerce ? d(null) : this.getGATracker(function(b) {
                var h;
                null == b && (h = a.getGAObject(), h("" + c + ".require", "ecommerce", "ecommerce.js"), f.requiredEcommerce=!0);
                return d(b)
            })
        };
        a.prototype.transaction = function(d, c, f) {
            var e;
            null == f && (f = b);
            e = this.options.name;
            return this.requireEcommerce(function(b) {
                var g, l, k;
                b = a.getGAObject();
                b("" + e + ".ecommerce:addTransaction", d);
                l = 0;
                for (k = c.length; l < k; l++)
                    g = c[l], b("" + e + ".ecommerce:addItem", g);
                b("" + e + ".ecommerce:send");
                return f()
            })
        };
        return a
    }();
    var v;
    v = function() {
        function a(a, c) {
            var f, b, h, g;
            this.trackers = [];
            h = 0;
            for (g = a.length; h < g; h++)
                b = a[h], f = r(b.settings, c), this.trackers.push(new u(f, b.matcher.lang, b.matcher.hostname))
        }
        var b;
        b = function() {};
        a.loadUniversalAnalytics = function(a, c) {
            var f, b;
            null == a && (a = "jimBob");
            null == c &&
            (c = "//www.google-analytics.com/analytics.js");
            window.GoogleAnalyticsObject = a;
            window[a] = window[a] || function() {
                var c;
                return (window[a].q = null != (c = window[a].q) ? c : []).push(arguments)
            };
            window[a].l =+ new Date;
            b = document.createElement("script");
            b.async = 1;
            b.src = c;
            f = document.getElementsByTagName("script")[0];
            return f.parentNode.insertBefore(b, f)
        };
        a.prototype.getTrackers = function(a, c) {
            var f, b, h, g, l;
            f = [];
            l = this.trackers;
            h = 0;
            for (g = l.length; h < g; h++)
                b = l[h], b.isSupportedLanguage(a) && b.isSupportedHostname(c) && f.push(b);
            return f
        };
        a.prototype.setForAllTrackers = function(a, c, f, e) {
            null == e && (e = b);
            return q(this.getTrackers(a, c), function(c, a) {
                return a.set(f, c)
            }, e)
        };
        a.prototype.sendForAllTrackers = function(a, c, f, e) {
            null == e && (e = b);
            return q(this.getTrackers(a, c), function(c, a) {
                return a.send(f, c)
            }, e)
        };
        a.prototype.transactionForAllTrackers = function(a, c, f, e, h) {
            null == h && (h = b);
            return q(this.getTrackers(a, c), function(c, a) {
                return a.transaction(f, e, c)
            }, h)
        };
        return a
    }();
    var z;
    z = function() {
        function a(c, a, b, d, g) {
            this.trackerManager = c;
            this.config =
            a;
            this.lcHostnames = b;
            this.supportHostnames = d;
            this.otherHostnames = g;
            this.whitelistedHostnames = [].concat(this.lcHostnames, this.supportHostnames, this.otherHostnames);
            this.settingsCallbacks = [];
            this.settingsApplied=!1;
            this.trackerSettings = {}
        }
        var b, d;
        b = {
            br: "pt",
            jp: "ja",
            cn: "zh"
        };
        d = function() {};
        a.prototype.setTrackerSetting = function(c) {
            this.settingsApplied=!1;
            return this.trackerSettings = r(this.trackerSettings, c, !0)
        };
        a.prototype.getTrackerSetting = function(c) {
            return this.trackerSettings[c]
        };
        a.prototype.resetTrackerSettings =
        function(c) {
            this.trackerSettings = {};
            c && this.setTrackerSetting(c);
            return this.settingsApplied=!1
        };
        a.prototype.registerSettingsCallback = function(c) {
            return this.settingsCallbacks.push(c)
        };
        a.prototype.applySettingsCallbacks = function(c) {
            var a;
            null == c && (c = d);
            a = this;
            return this.settingsApplied ? c() : q(this.settingsCallbacks, function(c, a) {
                if (1 === a.length)
                    return a(c);
                a();
                return c()
            }, function() {
                a.settingsApplied=!0;
                return c()
            })
        };
        a.prototype.sendToTrackers = function(c, a) {
            var b, h, g;
            null == a && (a = d);
            g = this;
            h = this.getLang();
            b = this.getHostname();
            if (this.hasWhitelistedHostname(b))
                return this.applySettingsCallbacks(function() {
                    g.trackerManager.setForAllTrackers(h, b, g.trackerSettings);
                    g.trackerManager.sendForAllTrackers(h, b, c);
                    return a()
                })
        };
        a.prototype.flushData = function(c) {
            null == c && (c = {});
            c = r(c, {
                hitType: "event",
                eventCategory: "custom pusher",
                eventAction: "flushing some data",
                eventLabel: "(not set)",
                nonInteraction: !0
            });
            return this.sendToTrackers(c)
        };
        a.prototype.sendPage = function(c, a) {
            null == c && (c = this.getFullPath());
            this.resetTrackerSettings({
                page: c
            });
            return this.sendToTrackers({
                hitType: "pageview"
            }, a)
        };
        a.prototype.sendEcommerce = function(c, a, b) {
            var h, g, k;
            null == b && (b = d);
            k = this;
            g = this.getLang();
            h = this.getHostname();
            if (this.hasWhitelistedHostname(h))
                return this.applySettingsCallbacks(function() {
                    k.trackerManager.setForAllTrackers(g, h, k.trackerSettings);
                    k.trackerManager.transactionForAllTrackers(g, h, c, a);
                    return b()
                })
        };
        a.prototype.sendEvent = function(c, a, b, d, g) {
            return this.sendToTrackers({
                hitType: "event",
                eventCategory: c,
                eventAction: a,
                eventLabel: b,
                eventValue: d,
                nonInteraction: !!g
            })
        };
        a.prototype.sendTiming = function(a, b, e, d) {
            return this.sendToTrackers({
                hitType: "timing",
                timingCategory: a,
                timingVar: b,
                timingValue: e,
                timingLabel: d
            })
        };
        a.prototype.sendSocial = function(a, b, e) {
            return this.sendToTrackers({
                hitType: "social",
                socialNetwork: a,
                socialAction: b,
                socialTarget: e
            })
        };
        a.prototype.sendDimension = function(a, b) {
            var e;
            e = {};
            e["dimension" + a] = b;
            return this.flushData(e)
        };
        a.prototype.sendMetric = function(a, b) {
            var e;
            e = {};
            e["metric" + a] =+ b;
            return this.flushData(e)
        };
        a.prototype.extractHostnameFromUrl =
        function(a) {
            a = String(a).replace(/https?\:\/\//, "").replace(/(?:\/|\?|#).*/, "");
            return this.isValidHostname(a) ? a : null
        };
        a.prototype.extractFirstSubdomainFromUrl = function(a) {
            var b;
            a = this.extractHostnameFromUrl(a);
            return null != a && (b = a.split("."), 2 < b.length) ? a.split(".")[0] : null
        };
        a.prototype.getReferrerFromCookie = function() {
            var a, b, e, d, g;
            e = k.jLang.referrerCookie;
            b = k.jLang.domain;
            g = k.overrideReferrer.cookie;
            d = k.overrideReferrer.domain;
            a = m(e);
            "" !== a && (m(e, "", {
                expires: - 1,
                domain: b,
                path: "/"
            }), "" !== m(e) && (a =
            null));
            b = m(g);
            "" !== b && (m(g, "", {
                expires: - 1,
                domain: d,
                path: "/"
            }), "" !== m(g) && (b = null));
            if (this.isValidUrl(b))
                return b;
            if (this.isValidUrl(a))
                return a
        };
        a.prototype.getAffiliateType = function(a, b) {
            var e, d;
            a = a.replace("a", "");
            d=!0;
            x(k.affiliate.endpoint, {
                id: a
            }, function(a) {
                if (d)
                    return clearTimeout(e), b(null, null != a ? a.type : void 0)
            });
            return e = setTimeout(function() {
                d=!1;
                return b(Error("Request Timeout"))
            }, k.affiliate.timeout)
        };
        a.prototype.hasCampaignParams = function(a) {
            var b, e;
            null == a && (a = this.getHref());
            b = n(a);
            e = null != b.utm_source && "" !== b.utm_source;
            a = null != b.utm_campaign && "" !== b.utm_campaign;
            b = null != b.utm_medium && "" !== b.utm_medium;
            return e && a && b
        };
        a.prototype.isValidUrl = function(a) {
            return /^https?:\/\/(?:[^\.\/]+\.){1,}[a-z]{2,}(?:$|\/|\?|#)/i.test(a)
        };
        a.prototype.isValidHostname = function(a) {
            return /^(?:[^\.\/]+\.){1,}[a-z]{2,}$/i.test(a)
        };
        a.prototype.isLoginPage = function(a) {
            null == a && (a = this.getPathname());
            return /^\/app\/auth\/signin(?:\/|$|\?|#)/.test(a)
        };
        a.prototype.isForumsPage = function(a) {
            null == a && (a =
            this.getPathname());
            return /^\/app\/forum(?:\/|$|\?|#)/.test(a)
        };
        a.prototype.isPartnerPage = function(a) {
            null == a && (a = this.getHref());
            return /^(?:https?:\/\/)?partner\.jimdo\.com(?:\/|$|\?|#)/.test(a)
        };
        a.prototype.isCheckoutPage = function(a) {
            null == a && (a = this.getPathname());
            return /^\/app\/billing\/[^\?#]+/.test(a)
        };
        a.prototype.isLandingPage = function(a) {
            null == a && (a = this.getHref());
            return /^(?:https?:\/\/)?www\.jimdo\.com\/l\/[^\?#]+/.test(a)
        };
        a.prototype.isFramebusterPage = function(a) {
            null == a && (a = this.getHref());
            return /^(?:(?:https?:\/\/)?[^\.\/]+\.jimdo\.com)?\/[^\/]+\/framebuster\/?(?:$|\?|#)/.test(a)
        };
        a.prototype.isFreeSignupGoal = function(a) {
            null == a && (a = this.getHref());
            return /^(?:(?:https?:\/\/)?[^\.\/]+\.jimdo\.com)?\/app\/signup\/index\/tracking\/?(?:$|\?|#)/.test(a)
        };
        a.prototype.isBlogPage = function(a) {
            var b;
            null == a && (a = this.getHref());
            b = /^(https?:\/\/)?blog.jimdo.com(?:\/|$|\?|#)/.test(a);
            return this.isLCPage(a) && /^https?:\/\/[^\/]+\/\d{4}\/\d{2}\/\d{2}/.test(a) || b
        };
        a.prototype.isEducationPage = function(a) {
            var b,
            e, d, g;
            null == a && (a = this.getHostname());
            a = this.extractHostnameFromUrl(a);
            g = ["seminarios.jimdo.com", "educacao.jimdo.com"];
            e = 0;
            for (d = g.length; e < d; e++)
                if (b = g[e], b === a)
                    return !0;
            return !1
        };
        a.prototype.isLCPage = function(a) {
            var b, e, d, g;
            null == a && (a = this.getHostname());
            a = this.extractHostnameFromUrl(a);
            g = this.lcHostnames;
            e = 0;
            for (d = g.length; e < d; e++)
                if (b = g[e], b === a)
                    return !0;
            return !1
        };
        a.prototype.isSupportPage = function(a) {
            var b, d, h, g;
            null == a && (a = this.getHostname());
            a = this.extractHostnameFromUrl(a);
            g = this.supportHostnames;
            d = 0;
            for (h = g.length; d < h; d++)
                if (b = g[d], b === a)
                    return !0;
            return !1
        };
        a.prototype.isCmsPage = function(a) {
            null == a && (a = this.getHref());
            return /^(?:https?:\/\/)?www\d{1,3}/.test(a)
        };
        a.prototype.isJimdoPage = function(a) {
            null == a && (a = this.getHref());
            return /^(?:https?:\/\/)?[^\/]*?jimdo\.com(?:#|\?|\/|$)/.test(a)
        };
        a.prototype.isIframed = function() {
            return self !== top
        };
        a.prototype.isGXBOPage = function(a) {
            var b;
            b=!0;
            null != a ? b=!1 : a = this.getHref();
            if (/^(https?:\/\/[^\.]+\.jimdo\.com)?\/app\/cooperation\/.*/.test(a))
                return !0;
            if (!b)
                return !1;
            a = document.getElementById("gaTracker");
            return null != a ? (a = a.getAttribute("data-gxbo"), "true" === String(a)) : !1
        };
        a.prototype.isFinalCheckoutPage = function() {
            var a;
            if (!this.isCheckoutPage())
                return !1;
            a = document.getElementById("gaTracker");
            return null != a ? (a = a.getAttribute("data-final-bill-view"), "true" === String(a)) : !1
        };
        a.prototype.isErrorPage = function() {
            var a;
            a = this.getFromDataObject("responseCode");
            return null != a && 400 <= a
        };
        a.prototype.isBrowserPreview = function() {
            var a;
            a = this.getFromDataObject("hasPrefetchHeader");
            return null != a && a
        };
        a.prototype.isFreeJimdoPage = function(a) {
            return this.isJimdoPage(a)&&!this.hasWhitelistedHostname(a)
        };
        a.prototype.hasWhitelistedHostname = function(a) {
            var b, d, h, g;
            null == a && (a = this.getHostname());
            a = this.extractHostnameFromUrl(a);
            g = this.whitelistedHostnames;
            d = 0;
            for (h = g.length; d < h; d++)
                if (b = g[d], "string" === typeof b) {
                    if (b === a)
                        return !0
                } else if (b.test(a))
                    return !0;
            return !1
        };
        a.prototype.getRawLangFromHTMLElement = function() {
            var a;
            return null != (a = document.getElementsByTagName("html")[0]) ? a.getAttribute("lang") :
            void 0
        };
        a.prototype.getRawLangFromMetaElement = function() {
            var a, b, d, h;
            b = document.getElementsByTagName("meta");
            d = 0;
            for (h = b.length; d < h; d++)
                if (a = b[d], "language" === a.getAttribute("name") || "content-language" === a.getAttribute("http-equiv"))
                    return a.getAttribute("content");
            return null
        };
        a.prototype.getRawLangFromQuery = function(a) {
            var b;
            null == a && (a = this.getFullPath());
            a = n(a);
            return null != (b = null != a ? a.cmsLanguage : void 0) ? b : null != a ? a.locale : void 0
        };
        a.prototype.getRawLangFromData = function() {
            return this.getFromDataObject("cmsLanguage") ||
            this.getFromDataObject("locale")
        };
        a.prototype.getLang = function() {
            var a, d, e, h, g, k;
            a = this.getRawLangFromData();
            d = this.getRawLangFromHTMLElement();
            e = this.getRawLangFromMetaElement();
            h = this.getRawLangFromQuery();
            a = (null != (g = null != (k = null != a ? a : d) ? k : e) ? g : h) || "";
            a = a.substr(0, 2).toLowerCase();
            null != b[a] && (a = b[a]);
            return a || null
        };
        a.prototype.getFromDataObject = function(a) {
            return (window.jimdoData || window.sessionData || {})[a]
        };
        a.prototype.getHostname = function() {
            return location.hostname
        };
        a.prototype.getHref = function() {
            return location.href
        };
        a.prototype.getPathname = function() {
            return location.pathname
        };
        a.prototype.getSearch = function() {
            return location.search
        };
        a.prototype.getHash = function() {
            return location.hash
        };
        a.prototype.getFullPath = function() {
            return this.getPathname() + this.getSearch() + this.getHash()
        };
        a.prototype.getReferrer = function() {
            return document.referrer
        };
        return a
    }();
    var A;
    A = function(a) {
        a.registerSettingsCallback(function() {
            var b;
            b = a.getReferrerFromCookie();
            null != b && a.setTrackerSetting({
                referrer: b
            })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isBlogPage())
                return b = a.getTrackerSetting("page"), a.setTrackerSetting({
                    page: "/blog" + b
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isEducationPage())
                return b = a.getTrackerSetting("page"), a.setTrackerSetting({
                    page: "/education" + b
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isSupportPage())
                return b = a.getTrackerSetting("page"), b = "/support" + b, a.isIframed() && (b = "/cmsView" + b), a.setTrackerSetting({
                    page: b
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isPartnerPage())
                return b = a.getTrackerSetting("page"),
                a.setTrackerSetting({
                    page: "/partner" + b
                })
        });
        a.registerSettingsCallback(function() {
            if (a.isLoginPage())
                return a.setTrackerSetting({
                    page: "/login"
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isCheckoutPage() || a.isForumsPage())
                return b = a.getTrackerSetting("page"), b = b.replace(/^\/app\//, "/"), a.setTrackerSetting({
                    page: b
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isFinalCheckoutPage())
                return b = a.getTrackerSetting("page"), b = b.replace("/bill/", "/finalBill/"), a.setTrackerSetting({
                    page: b
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isLandingPage())
                return b = a.getTrackerSetting("page"), b = b.replace(/^\/l\//, "/landingpage/"), a.setTrackerSetting({
                    page: b
                })
        });
        a.registerSettingsCallback(function() {
            var b, d, c;
            if (a.isErrorPage())
                return d = a.getFromDataObject("responseCode"), c = a.extractFirstSubdomainFromUrl(a.getHostname()), b = a.getTrackerSetting("page"), a.setTrackerSetting({
                    page: "/" + d + "/" + c + b
                })
        });
        a.registerSettingsCallback(function() {
            var b, d;
            if (a.isFreeSignupGoal())
                return d = n(location.search).trial,
                b = "/goal/free-signup", null != d && "" !== d && (b = "/goal/trial-" + d), a.setTrackerSetting({
                    page: b,
                    dimension1: !0
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isFramebusterPage()&&!a.hasCampaignParams())
                return b = a.getTrackerSetting("referrer") || a.getReferrer(), b = a.extractHostnameFromUrl(b) || "(not set)", a.setTrackerSetting({
                    campaignName: "framebuster referrals",
                    campaignSource: b,
                    campaignMedium: "framebuster"
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            b = a.getTrackerSetting("referrer") || a.getReferrer();
            if (a.isCmsPage(b))
                return a.hasCampaignParams() || (b = "cms link", a.isBlogPage() ? b = "cms news" : a.isSupportPage() && a.isIframed() && (b = "cms iframe"), a.setTrackerSetting({
                    campaignName: "cms referral",
                    campaignSource: b,
                    campaignMedium: "cms referral"
                })), a.setTrackerSetting({
                    dimension1: !0
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            b = a.getTrackerSetting("referrer") || a.getReferrer();
            /^https?:\/\/[^\/]*youtube\.com/.test(b) && (b = b.match(/^(https?:\/\/[^\/]*youtube\.com[^\?]*)\?(?:.*&)?(?:v|search_query)=([^&]+)/),
            null != b && 2 < b.length && (b = "" + b[1] + "/" + b[2], a.setTrackerSetting({
                referrer: b
            })))
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isGXBOPage() && (b = a.getTrackerSetting("page"), b = b.replace(/^\/app\//, "/"), a.setTrackerSetting({
                page: "/cooperation" + b
            }), b = a.getTrackerSetting("referrer") || a.getReferrer(), !a.hasWhitelistedHostname(b)))
                return b = a.extractHostnameFromUrl(b) || "(not set)", b = {
                    campaignName: "gxbo cooperation",
                    campaignSource: b,
                    campaignMedium: "gxbo"
                }, a.setTrackerSetting(b)
        });
        a.registerSettingsCallback(function() {
            var b;
            b = a.getTrackerSetting("referrer") || a.getReferrer();
            if (!a.isFramebusterPage() && a.isFreeJimdoPage(b)&&!a.hasCampaignParams())
                return b = a.extractFirstSubdomainFromUrl(b) || "(not set)", a.setTrackerSetting({
                    campaignName: "jimdofree referral",
                    campaignSource: b,
                    campaignMedium: "jimdofree referral"
                })
        });
        a.registerSettingsCallback(function() {
            var b;
            if (a.isBrowserPreview())
                return b = a.getTrackerSetting("page"), a.setTrackerSetting({
                    page: "/x-browser-preview" + b
                })
        });
        return a.registerSettingsCallback(function(b) {
            var d;
            return (d = n(location.href).ref) ? a.getAffiliateType(d, function(c, f) {
                var e, h;
                if (null != c)
                    return b();
                switch ("" + f) {
                case "1":
                    e = "Jimdo Affiliate";
                    break;
                case "2":
                    e = "Jimdo Partner";
                    break;
                default:
                    return b()
                }
                h = a.getTrackerSetting("referrer") || a.getReferrer();
                h = "" + (a.extractHostnameFromUrl(h) || "(direct)") + " - " + d;
                a.setTrackerSetting({
                    campaignName: e,
                    campaignSource: h,
                    campaignMedium: "affiliate link"
                });
                return b()
            }) : b()
        })
    };
    var B;
    B = function(a) {
        (function() {
            var b;
            a.isSupportPage() && (b = n(a.getHash()), w(function() {
                var d,
                c, f;
                d = n(a.getHash());
                if (d.stq !== b.stq || d.stp !== b.stp) {
                    b = d;
                    c = a.getPathname();
                    f = a.getSearch();
                    if ("" === f||-1 === f.indexOf("?"))
                        f = "?" + f;
                    null != d.stq && (f += "\x26stq\x3d" + d.stq);
                    null != d.stp && (f += "\x26stp\x3d" + d.stp);
                    a.sendPage(c + f)
                }
                return null
            }))
        })();
        return function() {
            var b, d, c;
            b = k.chat.cookie;
            d = k.chat.domain;
            "" === m(b) && (c = t(function() {
                return !!window.SnapEngage
            }, function() {
                var c, e, h, g, k;
                if ("function" === typeof(null != (k = window.SnapEngage) ? k.setCallback : void 0))
                    return h = c = e=!1, g = function() {
                        var g;
                        if (e && c)
                            return m(b,
                            "true", {
                                domain: d,
                                path: "/"
                            }), h=!0, g = a.getLang() || "(not set)", g = {
                                campaignName: "jimdo snapengage",
                                campaignMedium: "chat",
                                campaignSource: g
                            }, a.setTrackerSetting(g), a.sendPage()
                        }, SnapEngage.setCallback("ChatMessageSent", function() {
                            e=!0;
                            if (!h)
                                return g()
                            }), SnapEngage.setCallback("ChatMessageReceived", function() {
                                c=!0;
                                if (!h)
                                    return g()
                                })
            }), setTimeout(c, 8E3))
        }()
    };
    var p;
    p = null;
    (function() {
        var a, b, d;
        d = window._jimDoge = window._jimDoge || [];
        v.loadUniversalAnalytics(k.uaTracker.settings.globalVar, k.uaTracker.settings.endpoint);
        b = new v(k.uaTracker.trackers, k.uaTracker.initDefaults);
        p = new z(b, k, k.jimdoLocator.LOCAL_HOSTNAMES, k.jimdoLocator.SUPPORT_HOSTNAMES, k.jimdoLocator.OTHERS);
        a = function(a, b) {
            return p[a].apply(p, b)
        };
        window._jimDoge.push = function(b) {
            var d;
            d = y(b).toLowerCase();
            if ("[object array]" === d) {
                d = b.shift();
                switch (d) {
                case "transaction":
                    a("sendEcommerce", b);
                    break;
                case "event":
                    a("sendEvent", b);
                    break;
                case "social":
                    a("sendSocial", b);
                    break;
                case "timing":
                    a("sendTiming", b);
                    break;
                case "dimension":
                    a("sendDimension", b);
                    break;
                case "metric":
                    a("sendMetric", b);
                    break;
                case "pageview":
                case "page":
                    a("sendPage", b);
                    break;
                default:
                    throw Error("unknown command `" + d + "`");
                }
                return null
            }
            if ("[object function]" === d)
                return u.getGAObject()(function() {
                    return b(p)
                }), null;
            throw Error("unknown first argument type `" + d + "`");
        };
        A(p);
        B(p);
        p.isCmsPage() || d.push(["pageview"]);
        for (b = []; d.length;)
            b.push(d.push(d.shift()));
        return b
    })()
}();
