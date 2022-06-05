document.currentScript && (document.currentScript.onload = null);
(function(m) {
    var k, c = {}, a = function(a) {
        try {
            truste_undefined
        } catch (e) {
            return (a = e.stack) && (a = /:\/\/(([^\/]*)\/[^:]*)/.exec(a.substr(a.lastIndexOf(0 <= a.indexOf("@") ? "@" : 0 <= a.indexOf("(") ? "(" : " at ")))) ? a[1] : window.location.hostname + "/__loadedModule"
        }
    };
    k = /\/([^\/\?#]*(\?[^#]*)?)(#.*)?$/.exec(a())[1];
    c.load = function() {
        var b = m.toString();
        try {
            delete window[k]
        } catch (e) {
            window[k] = void 0
        }
        k = m = c = c.load = a = null;
        return b.substring(b.indexOf("{"))
    };
    window[k] = c
})(function() {
    var m = {}, k = Array.prototype.slice.call(arguments),
    c = {
        defaults: k[0] || {},
        binfo: k[1] || k[0] || {}
    };
    this != window && (this.inner = c);
    c.fake = {
        capabilities: ["getConsent"],
        default_consent: "denied",
        default_source: "implied",
        reportlevel: 5,
        consent: {
            all: {
                value: null,
                type: {}
            }
        },
        domain: window.location.hostname
    };
    c.requestors = {
        loading: []
    };
    c.authorities = [".truste.com", c.fake.domain];
    c.blacklist = [".example-xxx.com"];
    c.fake.consentDecision = null;
    c.fake.capabilities.push("getConsentDecision");
    c.valid_values = {
        consent: {
            denied: 1,
            approved: 2
        },
        source: {
            implied: 1,
            asserted: 2
        },
        type: {
            session: 1048561,
            necessary: 65523,
            limited: 65527,
            host: 65311,
            shared: 65343,
            present: 65407,
            systemic: 61951,
            functional: 62463,
            unique: 63487,
            uuid: 65535,
            user: 8191,
            site: 16383,
            party: 32767,
            "private": 16241,
            security: 16147,
            functionality: 29043,
            preferences: 30579,
            behavioral: 13119,
            tracking: 62335,
            analytic: 30583,
            advertising: 63487,
            requested: 8055,
            required: 16179,
            functionality: 29555,
            targeting: 65535
        }
    };
    c.caddy = null;
    c.isCapable = function(a) {
        for (var b = this.fake.capabilities.length; 0 < b--;)
            if (this.fake.capabilities[b] == a)
                return b + 1;
        return 0
    };
    c.endsWith =
    function(a, b) {
        return null != b && b.replace ? RegExp(".*" + b.replace(/\./g, "\\.") + "$").test(a) : !1
    };
    c._hasLoadedPrefs=!1;
    c.loadOldPrefs = function(a) {
        var b = this.getStorage("PrivacyManagerAPI.preferences");
        if (b)
            for (var e in b)
                a.consent[e] = b[e] || a.consent[e];
        this._hasLoadedPrefs=!0;
        for (var b = 0, d = this.requestors.loading && this.requestors.loading.length || 0; b < d;) {
            var h = this.requestors.loading[b++];
            if (a = this.processMessage(h.apiOb, h)) {
                for (e in a)
                    h.apiOb[e] = a[e];
                delete h.apiOb.loading;
                this.sendPost(h, {
                    PrivacyManagerAPI: h.apiOb
                })
            }
        }
    };
    c.isAuthorized = function(a, b, e, d) {
        d || (d = "");
        d.charAt && (d = d.split(/\s*,\s*/));
        var h, c = 0, g = "." + window.location.hostname;
        b = b || g;
        for (var n = this.blacklist.length; 0 < n--;)
            if (this.endsWith(b, this.blacklist[n]))
                return 0;
        if (this.endsWith(a, e) || this.endsWith(a, b))
            return - 4;
        if (!d)
            return 0;
        for (n = d.length; 0 < n--;) {
            var l = d[n];
            "." != l.charAt(0) && (l = d[n] = "." + l);
            for (var k = this.authorities.length; 0 < k--;)
                if (h = this.authorities[k], this.endsWith(l, h))
                    return k + 1;
            this.endsWith(b, l) ? c = Math.min( - 3, c) : this.endsWith(a, l) ? c = Math.min( - 2,
            c) : this.endsWith(g, l) && (c = Math.min( - 1, c))
        }
        if (c)
            return c;
        if (!a ||!e)
            return 0;
        this.sendEvent("uka", d, 0, e, null, a, b);
        return 0
    };
    c.sendUpdatesTo = function(a, b, e) {
        this.caddy = {
            hold: !0
        };
        var d, c = {
            PrivacyManagerAPI: {
                timestamp: e,
                capabilities: this.fake.capabilities
            }
        }, f = c.PrivacyManagerAPI, g;
        for (g in a)
            if (e = a[g])
                for (var k = e.length; 0 < k--;)
                    if ((d = e[k]) && d.w)
                        if (d.getConsent) {
                            var l = this.apiDo("getConsent", this.authorities[0], d.d, this.authorities[0], d.t);
                            if (d.s != l.source || d.c != l.consent)
                                f.consent = d.c = l.consent, f.source =
                                d.s = l.source, f.self = d.a, f.domain = d.d, f.action = "getConsent", this.sendPost(d.w, c)
                        } else 
                            d.getConsentDecision && b && (f.consent = f.source = null, f.self = d.a, f.action = "getConsentDecision", this.sendPost(d.w, c));
        this.caddy = null
    };
    c.getBType = function(a) {
        var b = 65535;
        if (a) {
            a.charAt ? a = a.split(/\W+/) : 0 < a && (a = [a]);
            for (var e = a.length; 0 < e--;) {
                var d = parseInt(a[e]);
                if (isNaN(d))
                    if (this.valid_values.type[a[e]])
                        b&=this.valid_values.type[a[e]];
                    else 
                        throw Error("invalid type");
                else 
                    b&=d
            }
        }
        return b
    };
    c.getTypePermission = function(a,
    b) {
        if (!b || isNaN(b) || b.length)
            b = this.getBType(b);
        var e = [];
        if (0 != b)
            for (var d in a)
                a[d] && this.valid_values.consent[a[d]] && (e.temp = this.getBType(d), (e.temp | b) == e.temp && "approved" == a[d] || ((e.temp | b) != e.temp || e.temp == b) && "denied" == a[d]) && (e[a[d]] || e.push(a[d]), e[a[d]] = d);
        return e.denied ? "denied" : e.join(",")
    };
    c.updatePreferences = function(a, b, e, d) {
        if (!a)
            return !1;
        "." != a.charAt(0) && (a = "." + a);
        if (b || e) {
            var c = this.getConsentForDomain(a, d) || {
                type: {}
            };
            this.valid_values.consent[b] && (c.value = b);
            if (e)
                for (var f in e)
                    if (this.valid_values.consent[e[f]])
                        isNaN(f) ?
                        this.valid_values.type[f] && (c.type[f] = e[f]) : c.type[f] = e[f];
                    else if (null === e[f] || "null" === e[f])
                        c.type[f] = null, delete c.type[f];
            d.consent[a] = c
        } else if (null === b || "null" === b)
            d.consent[a] = null, delete d.consent[a];
        else 
            return !1;
        this.getStorage("PrivacyManagerAPI.preferences", d.consent);
        return !0
    };
    c.apiDo = function(a, b, e, d, c) {
        if (!a ||!b ||!this.isCapable(a))
            return {
                error: "Call is missing required parameters or not allowed"
            };
        switch (a) {
        case "getConsent":
            d = d || window.location.hostname;
            (e = e || window.location.hostname) &&
            "." != e.charAt(0) && (e = "." + e);
            var f = isNaN(d) ? this.isAuthorized(e, (this.caddy || {}).from, b, d): d;
            if ("all" == e)
                return {
                    error: "Call is not authorized"
                };
            var g = 0;
            try {
                g = this.getBType(c)
            } catch (k) {
                return {
                    error: "Invalid Type parameter"
                }
            }
            c = this.getConsentForDomain(e, this.fake);
            var l = this.getTypePermission(this.fake.consent.all.type, g) || this.fake.consent.all.value;
            c && (l = this.getTypePermission(c.type, g) || c.value || l);
            this.sendEvent(a, f ? f : d, g, b, this.caddy, e);
            c = l ? {
                source: "asserted",
                consent: l
            } : {
                source: this.fake.default_source,
                consent: this.fake.default_consent
            };
            0 < f && (c.origin = window.location.hostname);
            return c;
        default:
            return this.secondaryAction(a, b, e)
        }
    };
    c.processMessage = function(a, b) {
        var e, d;
        if (!a ||!b ||!(d = b.origin || b.domain))
            return null;
        a.capabilities = this.fake.capabilities;
        this.tconsole.log("processing message from " + d);
        b = {
            origin: b.origin,
            domain: b.domain,
            source: b.source
        };
        if ("null" == d || "" == d)
            d = window.location.hostname;
        0 < (e = d.indexOf("://")) && (d = d.substring(e + 3));
        0 < (e = d.indexOf(":")) && (d = d.substring(0, e));
        d = "." + d;
        switch (a.action) {
        case "getConsent":
            var c,
            f, g;
            e = a.type || void 0;
            (c = a.self) && "." != c.charAt(0) && (c = "." + c);
            (g = a.domain) && "." != g.charAt(0) && (g = "." + g);
            f = a.authority;
            if (g&&!this.endsWith(g, d)) {
                if (!f ||!c || (f = 0 >= this.isAuthorized(g, d, c, f)))
                    return {
                        error: "Call is not authorized"
                    }
            } else 
                g || (g = d), f || (f = g);
            this.caddy = {
                from: d
            };
            d = this.apiDo("getConsent", c, g, f, e);
            this.caddy = null;
            d&&!d.error && (this.requestors[g] = this.requestors[g] || [], this.requestors[g].push({
                w: b,
                getConsent: 1,
                t: e,
                a: c,
                d: g,
                s: d.source,
                c: d.consent
            }), d.domain = g, d.self = c);
            return d;
        case "updatePreference":
            if (0 <
            this.isAuthorized(null, null, null, d)) {
                c = a.domain;
                if (!c)
                    return {
                        error: "Required parameter 'domain' not sent"
                    };
                "." != c.charAt(0) && (c = "." + c);
                if (!this.updatePreferences(c, a.value, a.type, this.fake))
                    return {
                        error: "Invalid value for required parameter 'value' sent"
                    };
                g = this.requestors;
                "all" != c && (g = {}, g[c] = this.requestors[c]);
                this.sendUpdatesTo(g, !1, a.timestamp);
                return null
            }
            return {
                error: "Call is not from an authorized Location"
            };
        default:
            return this.secondaryMessageProcessing(a, b, d)
        }
    };
    c.loadConsentDecision = function(a) {
        var b;
        null == a.consentDecision && (b = this.getStorage("truste.eu.cookie.notice_preferences"), b instanceof Object && (b = b.value), b = parseInt(b), isNaN(b) || (a.consentDecision = b + 1));
        b = a.consentDecision;
        if (3 == b || 0 === b)
            a.consent.all.value = "approved";
        else if (2 == b || 1 == b)
            a.consent.all.value = "denied";
        this.adjustTypeValues(a);
        this.binfo && this.binfo.gtm && this.updateGTM && this.updateGTM(a)
    };
    c.adjustTypeValues = function(a) {
        var b = a.consentDecision;
        delete a.consent.all.type.functional;
        delete a.consent.all.type.advertising;
        1 == b ?
        (a.consent.all.type.functional = "denied", a.consent.all.type.advertising = "denied") : 2 == b && (a.consent.all.type.functional = "approved", a.consent.all.type.advertising = "denied")
    };
    m.callApi = function() {
        try {
            return c.caddy = null, c.apiDo.apply(c, arguments)
        } catch (a) {
            try {
                c.sendError(arguments[0], arguments[3], arguments[2], arguments[1], null, !1, {
                    code: 1001,
                    result: a.stack || a.message,
                    context: navigator.platform,
                    version: "1.2"
                })
            } catch (b) {}
            c.tconsole.log(a.stack);
            c.caddy = null;
            return {
                error: "Unknown Error occured"
            }
        }
    };
    c.getConsentFromDomainlist =
    function(a, b, e) {
        if (!e)
            return null;
        var d = null, c;
        for (c in e)
            if (e[c].domains && (e[c].domains[a] || e[c].domains[a.substring(1)])) {
                d = {
                    value: null,
                    type: {}
                };
                e[c].value < b && (d.value = "approved");
                break
            }
        return d
    };
    c.getConsentForDomain = function(a, b) {
        if (!a ||!b)
            return null;
        this.loadConsentDecision(b);
        var e = b.consent[a];
        if (e && e.currentDecision == b.consentDecision)
            return e;
        e = this.getConsentFromDomainlist(a, b.consentDecision, this.getStorage("optout_domains"));
        if (!e || e.currentDecision && e.currentDecision != b.consentDecision)
            return null;
        e.currentDecision = b.consentDecision;
        return b.consent[a] = e
    };
    c._imgrep = [];
    c.sendEvent = function(a, b, e, c, h, f) {
        if (!(this.caddy && this.caddy.hold || this.tconsole.isDebug())) {
            if (this.isCapable(a)) {
                if (!isNaN(b) || this.fake.reportlevel & 4)
                    if (this.caddy) {
                        if (this.fake.reportlevel & 2)
                            return;
                            var g = this.caddy.from;
                            if (this.fake.reportlevel & 16 && f && g && this.endsWith(f, g) || this.fake.reportlevel & 32 && c && g && this.endsWith(g, c))
                                return 
                    } else if (this.fake.reportlevel & 1 || this.fake.reportlevel & 8 && c && f && this.endsWith(f, c))
                        return 
            } else if ("uka" ==
            a && this.fake.reportlevel & 64)
                return;
            null == h && (h = {
                page: window.location.pathname
            });
            f = window.location.hostname;
            b = "?a=" + encodeURIComponent(b) + (e ? "&t=" + encodeURIComponent(e) : "") + "&u=" + encodeURIComponent(c) + (h ? "&n=" + encodeURIComponent(this.cheapJSON(h)) : "");
            this.binfo && (f = this.binfo.domain, e = this.binfo.locale.indexOf("-"), b += "&c=" + encodeURIComponent(this.binfo.locale.slice(0, e)) + "&b=" + encodeURIComponent(this.binfo.behavior) + "&l=" + encodeURIComponent(this.binfo.locale.substr(e + 1)));
            this._imgrep[this._imgrep.push(new Image(1,
            1)) - 1].src = "//trackerapi.truste.com/trackerapi/1.0/log/cma/" + f + "/" + a + b + "&ts=" + (new Date).getTime()
        }
    };
    c.sendError = function(a, b, e, c, h, f, g) {
        this.fake.reportlevel & 128 || ("object" == typeof g && (g = this.cheapJSON(g)), f = window.location.hostname, a = "?a=" + encodeURIComponent(b) + (e ? "&t=" + encodeURIComponent(e) : "") + "&u=" + encodeURIComponent(c) + (h ? "&n=" + encodeURIComponent(h) : "") + "&i=" + encodeURIComponent(a), this.binfo && (f = this.binfo.domain, a += "&c=" + this.binfo.locale.slice(0, 2) + "&b=" + this.binfo.behavior + "&l=" + this.binfo.locale.substr(3)),
        b = new (window.XMLHttpRequest || window.ActiveXObject)("MSXML2.XMLHTTP.3.0"), b.open("POST", "//trackerapi.truste.com/trackerapi/1.0/log/cma/" + f + "/error" + a + "&ts=" + (new Date).getTime(), !0), b.setRequestHeader("Content-type", "application/json"), b.send(g))
    };
    c.secondaryMessageProcessing = function(a, b, c) {
        switch (a.action) {
        case "getConsentDecision":
            if (!a.self)
                return {
                    error: "Missing identity of API caller"
                };
            this.requestors[c] = this.requestors[c] || [];
            this.requestors[c].push({
                w: b,
                getConsentDecision: 1,
                t: null,
                a: a.self
            });
            return {
                consentDecision: null,
                source: null
            };
        case "updateDecision":
            if (0 < this.isAuthorized(null, null, null, c)) {
                b = a.value;
                if (isNaN(b))
                    return {
                        error: "Decision value is invalid"
                    };
                if (b == this.fake.consentDecision)
                    break;
                this.fake.consentDecision = b;
                this.loadConsentDecision(this.fake);
                this.tconsole.log("update decision to :" + b);
                this.sendUpdatesTo(this.requestors, !0, a.timestamp);
                return null
            }
            return {
                error: "Call is not from an authorized Location"
            }
        }
    };
    c.secondaryAction = function(a, b, c) {
        switch (a) {
        case "getConsentDecision":
            this.loadConsentDecision(this.fake);
            c = this.fake.consentDecision || 0;
            var d = null != this.fake.consentDecision ? "asserted": "implied";
            this.sendEvent(a, - 1, 0, b);
            return {
                consentDecision: c,
                source: d
            };
        case "changeReportLevel":
            return this.fake.reportlevel = parseInt(c + "")
        }
    };
    c.handleCMMessage = function(a) {
        if ("preference_manager" == a.source && "submit_preferences" == a.message)
            return a = parseInt("object" == typeof a.data ? a.data.value : a.data), this.tconsole.log("got new values from consent manager: " + a), isNaN(a) ? {} : {
                value: a + 1,
                action: "updateDecision",
                timestamp: 1
            }
    };
    c.handleMessageError = function(a, b) {
        try {
            var c = this.caddy;
            c && (c.data = caddy.data || {
                version: "1.2"
            }, c.data.context = navigator.platform, c.data.code = 1002, c.data.result = a.toString(), sendError(c.action, c.auth, c.type, c.asker, null, !1, c.data))
        } catch (d) {}
        this.caddy = null;
        this.tconsole.log("TRUSTe Consent Manager API unknown error. Returning Error. " + a.toString());
        this.tconsole.log(b);
        this.sendPost(b, {
            PrivacyManagerAPI: {
                error: "An unknown error occurred: " + a.toString()
            }
        });
        if (window.console)
            console.log(a.stack);
        else 
            throw a;
    };
    c.tconsole = {
        isDebug: function() {
            return (window.PrivacyManagerAPI || m).debug || 0 > window.location.hostname.indexOf(".")
        },
        log: function(a) {
            c.tconsole.isDebug() && window.console && window.console.log(a)
        }
    };
    c.parseJSON = function(a) {
        if ("string" != typeof a)
            return a;
        try {
            return window.JSON ? JSON.parse(a) : !/[^,:{}\[\]0-9.\-+Eaeflnr-u \n\r\t]/.test(a.replace(/"(\\.|[^"\\])*"/g, "")) && eval("(" + a + ")")
        } catch (b) {}
        return null
    };
    c.cheapJSON = function(a) {
        return window.truste && truste.util && truste.util.getJSON(a) || window.JSON && JSON.stringify(a) ||
        '{"PrivacyManagerAPI":{"message":"The API needs a JSON parser"}}'
    };
    c.getStorage = function(a, b) {
        try {
            null != b && (b.charAt || (b = this.cheapJSON(b)));
            if (window.localStorage)
                try {
                    if (null == b) {
                        if (b = window.localStorage[a] || window.localStorage.getItem(a))
                            return this.parseJSON(b) || b;
                            b = null
                    } else 
                        b ? window.localStorage.setItem(a, b) : delete window.localStorage[a]
            } catch (c) {
                this.tconsole.log("said was localstorage but wasn't: " + c.stack)
            }
            var d;
            if (null == b) {
                if ((d = RegExp("\\s*" + a.replace(".", "\\.") + "\\s*=\\s*([^,;\\s]*)").exec(document.cookie)) &&
                1 < d.length) {
                    if ((b = decodeURIComponent(d[1])) && window.localStorage)
                        try {
                            window.localStorage.setItem(a, b)
                        } catch (h) {
                        this.tconsole.log("said was localstorage but wasn't: " + h.stack)
                    }
                    return this.parseJSON(b) || b
                }
            } else {
                var f = this.fake.domain || null;
                f && "." != f.slice(0, 1) && (f = "." + f);
                var g = "; expires=" + (b ? (d = new Date) && d.setDate(720) && d.toGMTString() : "Thu, 01 Jan 1970 00:00:01 GMT"), g = g + ("; path=/" + (f ? "; domain=" + f : ""));
                document.cookie = a + "=" + encodeURIComponent(b) + g
            }
        } catch (k) {
            this.tconsole.log("error with getStorage : " +
            k.stack)
        }
        return null
    };
    c.sendPost = function(a, b) {
        if (window.postMessage && a && a.source && b) {
            "object" == typeof b && (b = this.cheapJSON(b));
            var c = a.origin || a.domain;
            "null" != c && c || (c = "*");
            this.tconsole.log("responding to (" + c + ") message : " + b);
            b && a.source.postMessage(b, c)
        }
    };
    c.init = function(a, b, e) {
        if (!this._hasLoadedPrefs) {
            b = b || this.fake;
            try {
                a && "string" == typeof a && (a = this.parseJSON(a));
                if (a)
                    for (var d in b)
                        b[d] = a[d] || b[d];
                e && c.loadOldPrefs && this.loadOldPrefs(b)
            } catch (h) {
                this.tconsole.log(h)
            }
        }
    };
    c.messageListener =
    function(a) {
        var b, e = a.data && c.parseJSON(a.data);
        if (e && (b = e.PrivacyManagerAPI || c.handleCMMessage(e)))
            if (b.capabilities || b.error)
                c.tconsole.log("got my own message, returning"), c.tconsole.log(a);
            else if (b.timestamp && b.action)
                try {
                    c.tconsole.log("GOT VALID MESSAGE: " + a.data);
                    var d = c.processMessage(b, a);
                    if (d) {
                        for (var h in d)
                            b[h] = d[h];
                            e.PrivacyManagerAPI && c.sendPost(a, e)
                        }
                } catch (f) {
            c.handleMessageError(f, a)
        } else 
            c.sendPost(a, '{"PrivacyManagerAPI":{"error":"API Object missing required fields"}}')
    };
    m.init =
    function(a, b) {
        c.init(a, null, b)
    };
    window.PREF_MGR_API_DEBUG = c;
    return m
});

