"function" != typeof DIL && (DIL = function(b, c) {
    var d = [], f, k;
    b !== Object(b) && (b = {});
    var l, m, t, C, u, A, v, s, B, J, K;
    l = b.partner;
    m = b.containerNSID;
    t = b.iframeAttachmentDelay;
    C=!!b.disableDestinationPublishingIframe;
    u = b.iframeAkamaiHTTPS;
    A = b.mappings;
    v = b.uuidCookie;
    s=!0 === b.enableErrorReporting;
    B = b.visitorService;
    J = b.declaredId;
    K=!0 === b.removeFinishedScriptsAndCallbacks;
    var w, L, F, D;
    w=!0 === b.disableScriptAttachment;
    L=!0 === b.disableDefaultRequest;
    F = b.afterResultForDefaultRequest;
    D = b.dpIframeSrc;
    s && DIL.errorModule.activate();
    var M=!0 === window._dil_unit_tests;
    (f = c) && d.push(f + "");
    if (!l || "string" != typeof l)
        return f = "DIL partner is invalid or not specified in initConfig", DIL.errorModule.handleError({
            name: "error",
            message: f,
            filename: "dil.js"
        }), Error(f);
    f = "DIL containerNSID is invalid or not specified in initConfig, setting to default of 0";
    if (m || "number" == typeof m)
        m = parseInt(m, 10), !isNaN(m) && 0 <= m && (f = "");
    f && (m = 0, d.push(f), f = "");
    k = DIL.getDil(l, m);
    if (k instanceof DIL && k.api.getPartner() == l && k.api.getContainerNSID() == m)
        return k;
    if (this instanceof DIL)
        DIL.registerDil(this, l, m);
    else 
        return new DIL(b, "DIL was not instantiated with the 'new' operator, returning a valid instance with partner \x3d " + l + " and containerNSID \x3d " + m);
    var x = {
        IS_HTTPS: "https:" == document.location.protocol,
        POST_MESSAGE_ENABLED: !!window.postMessage,
        COOKIE_MAX_EXPIRATION_DATE: "Tue, 19 Jan 2038 03:14:07 UTC"
    }, G = {
        stuffed: {}
    }, n = {}, p = {
        firingQueue: [],
        fired: [],
        firing: !1,
        sent: [],
        errored: [],
        reservedKeys: {
            sids: !0,
            pdata: !0,
            logdata: !0,
            callback: !0,
            postCallbackFn: !0,
            useImageRequest: !0
        },
        callbackPrefix: "demdexRequestCallback",
        firstRequestHasFired: !1,
        useJSONP: !0,
        abortRequests: !1,
        num_of_jsonp_responses: 0,
        num_of_jsonp_errors: 0,
        num_of_img_responses: 0,
        num_of_img_errors: 0,
        toRemove: [],
        removed: [],
        readyToRemove: !1,
        platformParams: {
            d_nsid: m + "",
            d_rtbd: "json",
            d_jsonv: DIL.jsonVersion + "",
            d_dst: "1"
        },
        nonModStatsParams: {
            d_rtbd: !0,
            d_dst: !0,
            d_cts: !0,
            d_rs: !0
        },
        modStatsParams: null,
        adms: {
            TIME_TO_CATCH_ALL_REQUESTS_RELEASE: 2E3,
            calledBack: !1,
            mid: null,
            noVisitorAPI: !1,
            instance: null,
            releaseType: "no VisitorAPI",
            admsProcessingStarted: !1,
            process: function(a) {
                try {
                    if (!this.admsProcessingStarted) {
                        var e = this, g, h, b, H, f;
                        if ("function" == typeof a && "function" == typeof a.getInstance) {
                            if (B === Object(B) && (g = B.namespace) && "string" == typeof g)
                                h = a.getInstance(g);
                            else {
                                this.releaseType = "no namespace";
                                this.releaseRequests();
                                return 
                            }
                            if (h === Object(h) && "function" == typeof h.isAllowed && "function" == typeof h.getMarketingCloudVisitorID) {
                                if (!h.isAllowed()) {
                                    this.releaseType = "VisitorAPI not allowed";
                                    this.releaseRequests();
                                    return 
                                }
                                this.instance =
                                h;
                                this.admsProcessingStarted=!0;
                                b = function(a) {
                                    "VisitorAPI" != e.releaseType && (e.mid = a, e.releaseType = "VisitorAPI", e.releaseRequests())
                                };
                                if (M && (H = B.server) && "string" == typeof H)
                                    h.server = H;
                                f = h.getMarketingCloudVisitorID(b);
                                if ("string" == typeof f && f.length) {
                                    b(f);
                                    return 
                                }
                                setTimeout(function() {
                                    "VisitorAPI" != e.releaseType && (e.releaseType = "timeout", e.releaseRequests())
                                }, this.TIME_TO_CATCH_ALL_REQUESTS_RELEASE);
                                return 
                            }
                            this.releaseType = "invalid instance"
                        } else 
                            this.noVisitorAPI=!0;
                        this.releaseRequests()
                    }
                } catch (c) {
                    this.releaseRequests()
                }
            },
            releaseRequests: function() {
                this.calledBack=!0;
                p.registerRequest()
            },
            getMarketingCloudVisitorID: function() {
                return this.instance ? this.instance.getMarketingCloudVisitorID() : null
            },
            getMIDQueryString: function() {
                var a = r.isPopulatedString, e = this.getMarketingCloudVisitorID();
                if (!a(this.mid) || this.mid != e)
                    this.mid = e;
                return a(this.mid) ? "d_mid\x3d" + this.mid + "\x26" : ""
            }
        },
        declaredId: {
            declaredId: {
                init: null,
                request: null
            },
            declaredIdCombos: {},
            setDeclaredId: function(a, e) {
                var g = r.isPopulatedString, h = encodeURIComponent;
                if (a === Object(a) && g(e)) {
                    var b = a.dpid, f = a.dpuuid, c = null;
                    if (g(b) && g(f)) {
                        c = h(b) + "$" + h(f);
                        if (!0 === this.declaredIdCombos[c])
                            return "setDeclaredId: combo exists for type '" + e + "'";
                        this.declaredIdCombos[c]=!0;
                        this.declaredId[e] = {
                            dpid: b,
                            dpuuid: f
                        };
                        return "setDeclaredId: succeeded for type '" + e + "'"
                    }
                }
                return "setDeclaredId: failed for type '" + e + "'"
            },
            getDeclaredIdQueryString: function() {
                var a = this.declaredId.request, e = this.declaredId.init, g = "";
                null !== a ? g = "\x26d_dpid\x3d" + a.dpid + "\x26d_dpuuid\x3d" + a.dpuuid : null !== e &&
                (g = "\x26d_dpid\x3d" + e.dpid + "\x26d_dpuuid\x3d" + e.dpuuid);
                return g
            }
        },
        registerRequest: function(a) {
            var e = this.firingQueue;
            a === Object(a) && e.push(a);
            !this.firing && e.length && (this.adms.calledBack ? (a = e.shift(), a.src = a.src.replace(/demdex.net\/event\?d_nsid=/, "demdex.net/event?" + this.adms.getMIDQueryString() + "d_nsid\x3d"), y.fireRequest(a), !this.firstRequestHasFired && "script" == a.tag && (this.firstRequestHasFired=!0)) : this.processVisitorAPI())
        },
        processVisitorAPI: function() {
            this.adms.process(window.Visitor)
        },
        requestRemoval: function(a) {
            if (!K)
                return "removeFinishedScriptsAndCallbacks is not boolean true";
            var e = this.toRemove, g, h;
            a === Object(a) && (g = a.script, h = a.callbackName, (g === Object(g) && "SCRIPT" == g.nodeName || "no script created" == g) && ("string" == typeof h && h.length) && e.push(a));
            if (this.readyToRemove && e.length) {
                h = e.shift();
                g = h.script;
                h = h.callbackName;
                "no script created" != g ? (a = g.src, g.parentNode.removeChild(g)) : a = g;
                window[h] = null;
                try {
                    delete window[h]
                } catch (b) {}
                this.removed.push({
                    scriptSrc: a,
                    callbackName: h
                });
                DIL.variables.scriptsRemoved.push(a);
                DIL.variables.callbacksRemoved.push(h);
                return this.requestRemoval()
            }
            return "requestRemoval() processed"
        }
    };
    k = function() {
        var a = "http://fast.", e = "?d_nsid\x3d" + m + "#" + encodeURIComponent(document.location.href);
        if ("string" === typeof D && D.length)
            return D + e;
        x.IS_HTTPS && (a=!0 === u ? "https://fast." : "https://");
        return a + l + ".demdex.net/dest4.html" + e
    };
    var z = {
        THROTTLE_START: 3E4,
        throttleTimerSet: !1,
        id: "destination_publishing_iframe_" + l + "_" + m,
        url: k(),
        iframe: null,
        iframeHasLoaded: !1,
        sendingMessages: !1,
        messages: [],
        messagesPosted: [],
        messageSendingInterval: x.POST_MESSAGE_ENABLED ? 15: 100,
        jsonProcessed: [],
        attachIframe: function() {
            var a =
            this, e = document.createElement("iframe");
            e.id = this.id;
            e.style.cssText = "display: none; width: 0; height: 0;";
            e.src = this.url;
            q.addListener(e, "load", function() {
                a.iframeHasLoaded=!0;
                a.requestToProcess()
            });
            document.body.appendChild(e);
            this.iframe = e
        },
        requestToProcess: function(a, e) {
            var g = this;
            a&&!r.isEmptyObject(a) && this.process(a, e);
            this.iframeHasLoaded && (this.messages.length&&!this.sendingMessages) && (this.throttleTimerSet || (this.throttleTimerSet=!0, setTimeout(function() {
                g.messageSendingInterval = x.POST_MESSAGE_ENABLED ?
                15 : 150
            }, this.THROTTLE_START)), this.sendingMessages=!0, this.sendMessages())
        },
        process: function(a, e) {
            var g = encodeURIComponent, h, b, f, c, d, l;
            e === Object(e) && (l = q.encodeAndBuildRequest(["", e.dpid || "", e.dpuuid || ""], ","));
            if ((h = a.dests) && h instanceof Array && (b = h.length))
                for (f = 0; f < b; f++)
                    c = h[f], c = [g("dests"), g(c.id || ""), g(c.y || ""), g(c.c || "")], this.addMessage(c.join("|"));
            if ((h = a.ibs) && h instanceof Array && (b = h.length))
                for (f = 0; f < b; f++)
                    c = h[f], c = [g("ibs"), g(c.id || ""), g(c.tag || ""), q.encodeAndBuildRequest(c.url || [],
                    ","), g(c.ttl || ""), "", l], this.addMessage(c.join("|"));
            if ((h = a.dpcalls) && h instanceof Array && (b = h.length))
                for (f = 0; f < b; f++)
                    c = h[f], d = c.callback || {}, d = [d.obj || "", d.fn || "", d.key || "", d.tag || "", d.url || ""], c = [g("dpm"), g(c.id || ""), g(c.tag || ""), q.encodeAndBuildRequest(c.url || [], ","), g(c.ttl || ""), q.encodeAndBuildRequest(d, ","), l], this.addMessage(c.join("|"));
            this.jsonProcessed.push(a)
        },
        addMessage: function(a) {
            var e = encodeURIComponent, e = s ? e("---destpub-debug---"): e("---destpub---");
            this.messages.push(e + a)
        },
        sendMessages: function() {
            var a =
            this, e;
            this.messages.length ? (e = this.messages.shift(), DIL.xd.postMessage(e, this.url, this.iframe.contentWindow), this.messagesPosted.push(e), setTimeout(function() {
                a.sendMessages()
            }, this.messageSendingInterval)) : this.sendingMessages=!1
        }
    }, I = {
        traits: function(a) {
            r.isValidPdata(a) && (n.sids instanceof Array || (n.sids = []), q.extendArray(n.sids, a));
            return this
        },
        pixels: function(a) {
            r.isValidPdata(a) && (n.pdata instanceof Array || (n.pdata = []), q.extendArray(n.pdata, a));
            return this
        },
        logs: function(a) {
            r.isValidLogdata(a) &&
            (n.logdata !== Object(n.logdata) && (n.logdata = {}), q.extendObject(n.logdata, a));
            return this
        },
        customQueryParams: function(a) {
            r.isEmptyObject(a) || q.extendObject(n, a, p.reservedKeys);
            return this
        },
        signals: function(a, e) {
            var g, h = a;
            if (!r.isEmptyObject(h)) {
                if (e && "string" == typeof e)
                    for (g in h = {}, a)
                        a.hasOwnProperty(g) && (h[e + g] = a[g]);
                q.extendObject(n, h, p.reservedKeys)
            }
            return this
        },
        declaredId: function(a) {
            p.declaredId.setDeclaredId(a, "request");
            return this
        },
        result: function(a) {
            "function" == typeof a && (n.callback = a);
            return this
        },
        afterResult: function(a) {
            "function" == typeof a && (n.postCallbackFn = a);
            return this
        },
        useImageRequest: function() {
            n.useImageRequest=!0;
            return this
        },
        clearData: function() {
            n = {};
            return this
        },
        submit: function() {
            y.submitRequest(n);
            n = {};
            return this
        },
        getPartner: function() {
            return l
        },
        getContainerNSID: function() {
            return m
        },
        getEventLog: function() {
            return d
        },
        getState: function() {
            var a = {}, e = {};
            q.extendObject(a, p, {
                callbackPrefix: !0,
                useJSONP: !0,
                registerRequest: !0
            });
            q.extendObject(e, z, {
                attachIframe: !0,
                requestToProcess: !0,
                process: !0,
                sendMessages: !0
            });
            return {
                pendingRequest: n,
                otherRequestInfo: a,
                destinationPublishingInfo: e
            }
        },
        idSync: function(a) {
            if (a !== Object(a) || "string" != typeof a.dpid ||!a.dpid.length)
                return "Error: config or config.dpid is empty";
            if ("string" != typeof a.url ||!a.url.length)
                return "Error: config.url is empty";
            var e = a.url, g = a.minutesToLive, h = encodeURIComponent, b, e = e.replace(/^https:/, "").replace(/^http:/, "");
            if ("undefined" == typeof g)
                g = 20160;
            else if (g = parseInt(g, 10), isNaN(g) || 0 >= g)
                return "Error: config.minutesToLive needs to be a positive number";
            b = q.encodeAndBuildRequest(["", a.dpid, a.dpuuid || ""], ",");
            a = ["ibs", h(a.dpid), "img", h(e), g, "", b];
            z.addMessage(a.join("|"));
            p.firstRequestHasFired && z.requestToProcess();
            return "Successfully queued"
        },
        aamIdSync: function(a) {
            if (a !== Object(a) || "string" != typeof a.dpuuid ||!a.dpuuid.length)
                return "Error: config or config.dpuuid is empty";
            a.url = "//dpm.demdex.net/ibs:dpid\x3d" + a.dpid + "\x26dpuuid\x3d" + a.dpuuid;
            return this.idSync(a)
        },
        passData: function(a) {
            if (r.isEmptyObject(a))
                return "Error: json is empty or not an object";
            y.defaultCallback(a);
            return "json submitted for processing"
        },
        getPlatformParams: function() {
            return p.platformParams
        },
        getEventCallConfigParams: function() {
            var a = p, e = a.modStatsParams, g = a.platformParams, h;
            if (!e) {
                e = {};
                for (h in g)
                    g.hasOwnProperty(h)&&!a.nonModStatsParams[h] && (e[h.replace(/^d_/, "")] = g[h]);
                a.modStatsParams = e
            }
            return e
        }
    }, y = {
        submitRequest: function(a) {
            p.registerRequest(y.createQueuedRequest(a));
            return !0
        },
        createQueuedRequest: function(a) {
            var e = p, g, h = a.callback, b = "img";
            if (!r.isEmptyObject(A)) {
                var c,
                f, d;
                for (c in A)
                    if (A.hasOwnProperty(c) && (f = A[c], !(null == f || "" === f) && c in a&&!(f in a)&&!(f in p.reservedKeys)))
                        d = a[c], null == d || "" === d || (a[f] = d)
            }
            r.isValidPdata(a.sids) || (a.sids = []);
            r.isValidPdata(a.pdata) || (a.pdata = []);
            r.isValidLogdata(a.logdata) || (a.logdata = {});
            a.logdataArray = q.convertObjectToKeyValuePairs(a.logdata, "\x3d", !0);
            a.logdataArray.push("_ts\x3d" + (new Date).getTime());
            "function" != typeof h && (h = this.defaultCallback);
            if (e.useJSONP=!a.useImageRequest || "boolean" != typeof a.useImageRequest)
                b = "script",
                g = e.callbackPrefix + "_" + l + "_" + m + "_" + (new Date).getTime();
            return {
                tag: b,
                src: y.makeRequestSrc(a, g),
                internalCallbackName: g,
                callbackFn: h,
                postCallbackFn: a.postCallbackFn,
                useImageRequest: a.useImageRequest,
                requestData: a
            }
        },
        defaultCallback: function(a, e) {
            var g, h, b, c, f, d, l, k, m;
            if ((g = a.stuff) && g instanceof Array && (h = g.length))
                for (b = 0; b < h; b++)
                    if ((c = g[b]) && c === Object(c)) {
                        f = c.cn;
                        d = c.cv;
                        l = c.ttl;
                        if ("undefined" == typeof l || "" === l)
                            l = Math.floor(q.getMaxCookieExpiresInMinutes() / 60 / 24);
                            k = c.dmn || "." + document.domain.replace(/^www\./,
                            "");
                            m = c.type;
                            if (f && (d || "number" == typeof d))
                                "var" != m && (l = parseInt(l, 10))&&!isNaN(l) && q.setCookie(f, d, 1440 * l, "/", k, !1), G.stuffed[f] = d
                    }
            g = a.uuid;
            if (r.isPopulatedString(g)&&!r.isEmptyObject(v)) {
                h = v.path;
                if ("string" != typeof h ||!h.length)
                    h = "/";
                b = parseInt(v.days, 10);
                isNaN(b) && (b = 100);
                q.setCookie(v.name || "aam_did", g, 1440 * b, h, v.domain || "." + document.domain.replace(/^www\./, ""), !0 === v.secure)
            }
            !C&&!p.abortRequests && z.requestToProcess(a, e)
        },
        makeRequestSrc: function(a, e) {
            a.sids = r.removeEmptyArrayValues(a.sids || []);
            a.pdata = r.removeEmptyArrayValues(a.pdata || []);
            var g = p, h = g.platformParams, b = q.encodeAndBuildRequest(a.sids, ","), c = q.encodeAndBuildRequest(a.pdata, ","), f = (a.logdataArray || []).join("\x26");
            delete a.logdataArray;
            var d = x.IS_HTTPS ? "https://": "http://", k = g.declaredId.getDeclaredIdQueryString(), m;
            m = [];
            var s, n, u, t;
            for (s in a)
                if (!(s in g.reservedKeys) && a.hasOwnProperty(s))
                    if (n = a[s], s = encodeURIComponent(s), n instanceof Array) {
                        u = 0;
                        for (t = n.length; u < t; u++)
                            m.push(s + "\x3d" + encodeURIComponent(n[u]))
                    } else 
                        m.push(s +
                        "\x3d" + encodeURIComponent(n));
            m = m.length ? "\x26" + m.join("\x26") : "";
            return d + l + ".demdex.net/event?d_nsid\x3d" + h.d_nsid + k + (b.length ? "\x26d_sid\x3d" + b : "") + (c.length ? "\x26d_px\x3d" + c : "") + (f.length ? "\x26d_ld\x3d" + encodeURIComponent(f) : "") + m + (g.useJSONP ? "\x26d_rtbd\x3d" + h.d_rtbd + "\x26d_jsonv\x3d" + h.d_jsonv + "\x26d_dst\x3d" + h.d_dst + "\x26d_cb\x3d" + (e || "") : "")
        },
        fireRequest: function(a) {
            if ("img" == a.tag)
                this.fireImage(a);
            else if ("script" == a.tag) {
                var e = p.declaredId, e = e.declaredId.request || e.declaredId.init || {};
                this.fireScript(a, {
                    dpid: e.dpid || "",
                    dpuuid: e.dpuuid || ""
                })
            }
        },
        fireImage: function(a) {
            var e = p, g, b;
            e.abortRequests || (e.firing=!0, g = new Image(0, 0), e.sent.push(a), g.onload = function() {
                e.firing=!1;
                e.fired.push(a);
                e.num_of_img_responses++;
                e.registerRequest()
            }, b = function(g) {
                f = "imgAbortOrErrorHandler received the event of type " + g.type;
                d.push(f);
                e.abortRequests=!0;
                e.firing=!1;
                e.errored.push(a);
                e.num_of_img_errors++;
                e.registerRequest()
            }, g.addEventListener ? (g.addEventListener("error", b, !1), g.addEventListener("abort",
            b, !1)) : g.attachEvent && (g.attachEvent("onerror", b), g.attachEvent("onabort", b)), g.src = a.src)
        },
        fireScript: function(a, e) {
            var g = this, b = p, c, m, k = a.src, s = a.postCallbackFn, n = "function" == typeof s, u = a.internalCallbackName;
            b.abortRequests || (b.firing=!0, window[u] = function(g) {
                try {
                    g !== Object(g) && (g = {});
                    var c = a.callbackFn;
                    b.firing=!1;
                    b.fired.push(a);
                    b.num_of_jsonp_responses++;
                    c(g, e);
                    n && s(g, e)
                } catch (k) {
                    k.message = "DIL jsonp callback caught error with message " + k.message;
                    f = k.message;
                    d.push(f);
                    k.filename = k.filename ||
                    "dil.js";
                    k.partner = l;
                    DIL.errorModule.handleError(k);
                    try {
                        c({
                            error: k.name + "|" + k.message
                        }), n && s({
                            error: k.name + "|" + k.message
                        })
                    } catch (p) {}
                } finally {
                    b.requestRemoval({
                        script: m,
                        callbackName: u
                    }), b.registerRequest()
                }
            }, w ? (b.firing=!1, b.requestRemoval({
                script: "no script created",
                callbackName: u
            })) : (m = document.createElement("script"), m.addEventListener && m.addEventListener("error", function(e) {
                b.requestRemoval({
                    script: m,
                    callbackName: u
                });
                f = "jsonp script tag error listener received the event of type " + e.type + " with src " +
                k;
                g.handleScriptError(f, a)
            }, !1), m.type = "text/javascript", m.src = k, c = DIL.variables.scriptNodeList[0], c.parentNode.insertBefore(m, c)), b.sent.push(a), b.declaredId.declaredId.request = null)
        },
        handleScriptError: function(a, e) {
            var b = p;
            d.push(a);
            b.abortRequests=!0;
            b.firing=!1;
            b.errored.push(e);
            b.num_of_jsonp_errors++;
            b.registerRequest()
        }
    }, r = {
        isValidPdata: function(a) {
            return a instanceof Array && this.removeEmptyArrayValues(a).length?!0 : !1
        },
        isValidLogdata: function(a) {
            return !this.isEmptyObject(a)
        },
        isEmptyObject: function(a) {
            if (a !==
            Object(a))
                return !0;
            for (var e in a)
                if (a.hasOwnProperty(e))
                    return !1;
            return !0
        },
        removeEmptyArrayValues: function(a) {
            for (var e = 0, b = a.length, c, f = [], e = 0; e < b; e++)
                c = a[e], "undefined" != typeof c && null != c && f.push(c);
            return f
        },
        isPopulatedString: function(a) {
            return "string" == typeof a && a.length
        }
    }, q = {
        addListener: function() {
            if (document.addEventListener)
                return function(a, e, b) {
                    a.addEventListener(e, function(a) {
                        "function" == typeof b && b(a)
                    }, !1)
                };
            if (document.attachEvent)
                return function(a, e, b) {
                    a.attachEvent("on" + e, function(a) {
                        "function" ==
                        typeof b && b(a)
                    })
                }
        }(),
        convertObjectToKeyValuePairs: function(a, e, b) {
            var c = [];
            e = e || "\x3d";
            var f, d;
            for (f in a)
                d = a[f], "undefined" != typeof d && null != d && c.push(f + e + (b ? encodeURIComponent(d) : d));
            return c
        },
        encodeAndBuildRequest: function(a, e) {
            return this.map(a, function(a) {
                return encodeURIComponent(a)
            }).join(e)
        },
        map: function(a, e) {
            if (Array.prototype.map)
                return a.map(e);
            if (void 0 === a || null === a)
                throw new TypeError;
            var b = Object(a), c = b.length>>>0;
            if ("function" !== typeof e)
                throw new TypeError;
            for (var f = Array(c), d = 0; d <
            c; d++)
                d in b && (f[d] = e.call(e, b[d], d, b));
            return f
        },
        filter: function(a, b) {
            if (!Array.prototype.filter) {
                if (void 0 === a || null === a)
                    throw new TypeError;
                var c = Object(a), f = c.length>>>0;
                if ("function" !== typeof b)
                    throw new TypeError;
                for (var d = [], l = 0; l < f; l++)
                    if (l in c) {
                        var k = c[l];
                        b.call(b, k, l, c) && d.push(k)
                    }
                return d
            }
            return a.filter(b)
        },
        getCookie: function(a) {
            a += "\x3d";
            var b = document.cookie.split(";"), c, f, d;
            c = 0;
            for (f = b.length; c < f; c++) {
                for (d = b[c]; " " == d.charAt(0);)
                    d = d.substring(1, d.length);
                if (0 == d.indexOf(a))
                    return decodeURIComponent(d.substring(a.length,
                    d.length))
            }
            return null
        },
        setCookie: function(a, b, c, f, d, l) {
            var k = new Date;
            c && (c*=6E4);
            document.cookie = a + "\x3d" + encodeURIComponent(b) + (c ? ";expires\x3d" + (new Date(k.getTime() + c)).toUTCString() : "") + (f ? ";path\x3d" + f : "") + (d ? ";domain\x3d" + d : "") + (l ? ";secure" : "")
        },
        extendArray: function(a, b) {
            return a instanceof Array && b instanceof Array ? (Array.prototype.push.apply(a, b), !0) : !1
        },
        extendObject: function(a, b, c) {
            var f;
            if (a === Object(a) && b === Object(b)) {
                for (f in b)
                    if (b.hasOwnProperty(f) && (r.isEmptyObject(c) ||!(f in c)))
                        a[f] =
                        b[f];
                return !0
            }
            return !1
        },
        getMaxCookieExpiresInMinutes: function() {
            return ((new Date(x.COOKIE_MAX_EXPIRATION_DATE)).getTime() - (new Date).getTime()) / 1E3 / 60
        }
    };
    "error" == l && 0 == m && q.addListener(window, "load", function() {
        DIL.windowLoaded=!0
    });
    var E = function() {
        O();
        !C&&!p.abortRequests && z.attachIframe();
        p.readyToRemove=!0;
        p.requestRemoval()
    }, O = function() {
        C || setTimeout(function() {
            !L && (!p.firstRequestHasFired&&!p.adms.admsProcessingStarted&&!p.adms.calledBack) && ("function" == typeof F ? I.afterResult(F).submit() : I.submit())
        },
        DIL.constants.TIME_TO_DEFAULT_REQUEST)
    }, N = document;
    "error" != l && (DIL.windowLoaded ? E() : "complete" != N.readyState && "loaded" != N.readyState ? q.addListener(window, "load", E) : DIL.isAddedPostWindowLoadWasCalled ? q.addListener(window, "load", E) : (t = "number" == typeof t ? parseInt(t, 10) : 0, 0 > t && (t = 0), setTimeout(E, t || DIL.constants.TIME_TO_CATCH_ALL_DP_IFRAME_ATTACHMENT)));
    p.declaredId.setDeclaredId(J, "init");
    this.api = I;
    this.getStuffedVariable = function(a) {
        var b = G.stuffed[a];
        !b && "number" != typeof b && (b = q.getCookie(a), !b &&
        "number" != typeof b && (b = ""));
        return b
    };
    this.validators = r;
    this.helpers = q;
    this.constants = x;
    this.log = d;
    M && (this.pendingRequest = n, this.requestController = p, this.setDestinationPublishingUrl = k, this.destinationPublishing = z, this.requestProcs = y, this.variables = G)
}, function() {
    var b = document, c;
    null == b.readyState && b.addEventListener && (b.readyState = "loading", b.addEventListener("DOMContentLoaded", c = function() {
        b.removeEventListener("DOMContentLoaded", c, !1);
        b.readyState = "complete"
    }, !1))
}(), DIL.extendStaticPropertiesAndMethods =
function(b) {
    var c;
    if (b === Object(b))
        for (c in b)
            b.hasOwnProperty(c) && (this[c] = b[c])
}, DIL.extendStaticPropertiesAndMethods({
    version: "4.9",
    jsonVersion: 1,
    constants: {
        TIME_TO_DEFAULT_REQUEST: 50,
        TIME_TO_CATCH_ALL_DP_IFRAME_ATTACHMENT: 500
    },
    variables: {
        scriptNodeList: document.getElementsByTagName("script"),
        scriptsRemoved: [],
        callbacksRemoved: []
    },
    windowLoaded: !1,
    dils: {},
    isAddedPostWindowLoadWasCalled: !1,
    isAddedPostWindowLoad: function(b) {
        this.isAddedPostWindowLoadWasCalled=!0;
        this.windowLoaded = "function" == typeof b ?
        !!b() : "boolean" == typeof b ? b : !0
    },
    create: function(b) {
        try {
            return new DIL(b)
        } catch (c) {
            return (new Image(0, 0)).src = "http://error.demdex.net/event?d_nsid\x3d0\x26d_px\x3d14137\x26d_ld\x3dname%3Derror%26filename%3Ddil.js%26partner%3Dno_partner%26message%3DError%2520in%2520attempt%2520to%2520create%2520DIL%2520instance%2520with%2520DIL.create()%26_ts%3D" + (new Date).getTime(), Error("Error in attempt to create DIL instance with DIL.create()")
        }
    },
    registerDil: function(b, c, d) {
        c = c + "$" + d;
        c in this.dils || (this.dils[c] =
        b)
    },
    getDil: function(b, c) {
        var d;
        "string" != typeof b && (b = "");
        c || (c = 0);
        d = b + "$" + c;
        return d in this.dils ? this.dils[d] : Error("The DIL instance with partner \x3d " + b + " and containerNSID \x3d " + c + " was not found")
    },
    dexGetQSVars: function(b, c, d) {
        c = this.getDil(c, d);
        return c instanceof this ? c.getStuffedVariable(b) : ""
    },
    xd: {
        postMessage: function(b, c, d) {
            var f = 1;
            c && (window.postMessage ? d.postMessage(b, c.replace(/([^:]+:\/\/[^\/]+).*/, "$1")) : c && (d.location = c.replace(/#.*$/, "") + "#" + + new Date + f++ + "\x26" + b))
        }
    }
}), DIL.errorModule =
function() {
    var b = DIL.create({
        partner: "error",
        containerNSID: 0,
        disableDestinationPublishingIframe: !0
    }), c = {
        harvestererror: 14138,
        destpuberror: 14139,
        dpmerror: 14140,
        generalerror: 14137,
        error: 14137,
        noerrortypedefined: 15021,
        evalerror: 15016,
        rangeerror: 15017,
        referenceerror: 15018,
        typeerror: 15019,
        urierror: 15020
    }, d=!1;
    return {
        activate: function() {
            d=!0
        },
        handleError: function(f) {
            if (!d)
                return "DIL error module has not been activated";
            f !== Object(f) && (f = {});
            var k = f.name ? (new String(f.name)).toLowerCase(): "", l = [];
            f = {
                name: k,
                filename: f.filename ? f.filename + "": "",
                partner: f.partner ? f.partner + "": "no_partner",
                site: f.site ? f.site + "": document.location.href,
                message: f.message ? f.message + "": ""
            };
            l.push(k in c ? c[k] : c.noerrortypedefined);
            b.api.pixels(l).logs(f).useImageRequest().submit();
            return "DIL error report sent"
        },
        pixelMap: c
    }
}(), DIL.tools = {}, DIL.modules = {
    helpers: {
        handleModuleError: function(b, c, d) {
            var f = "";
            c = c || "Error caught in DIL module/submodule: ";
            b === Object(b) ? f = c + (b.message || "err has no message") : (f = c + "err is not a valid object",
            b = {});
            b.message = f;
            d instanceof DIL && (b.partner = d.api.getPartner());
            DIL.errorModule.handleError(b);
            return this.errorMessage = f
        }
    }
});
DIL.tools.getSearchReferrer = function(b, c) {
    var d = DIL.getDil("error"), f = DIL.tools.decomposeURI(b || document.referrer), k = "", l = "", m = {
        queryParam: "q"
    }, k = d.helpers.filter([c === Object(c) ? c: {}, {
        hostPattern: /aol\./
    }, {
        hostPattern: /ask\./
    }, {
        hostPattern: /bing\./
    }, {
        hostPattern: /google\./
    }, {
        hostPattern: /yahoo\./,
        queryParam: "p"
    }
    ], function(b) {
        return !(!b.hasOwnProperty("hostPattern") ||!f.hostname.match(b.hostPattern))
    }).shift();
    return !k ? {
        valid: !1,
        name: "",
        keywords: ""
    } : {
        valid: !0,
        name: f.hostname,
        keywords: (d.helpers.extendObject(m,
        k), l = m.queryPattern ? (k = ("" + f.search).match(m.queryPattern)) ? k[1] : "" : f.uriParams[m.queryParam], decodeURIComponent(l || "").replace(/\+|%20/g, " "))
    }
};
DIL.tools.decomposeURI = function(b) {
    var c = DIL.getDil("error"), d = document.createElement("a");
    d.href = b || document.referrer;
    return {
        hash: d.hash,
        host: d.host.split(":").shift(),
        hostname: d.hostname,
        href: d.href,
        pathname: d.pathname.replace(/^\//, ""),
        protocol: d.protocol,
        search: d.search,
        uriParams: function(b, d) {
            c.helpers.map(d.split("\x26"), function(c) {
                c = c.split("\x3d");
                b[c.shift()] = c.shift()
            });
            return b
        }({}, d.search.replace(/^(\/|\?)?|\/$/g, ""))
    }
};
DIL.tools.getMetaTags = function() {
    var b = {}, c = document.getElementsByTagName("meta"), d, f, k, l, m;
    d = 0;
    for (k = arguments.length; d < k; d++)
        if (l = arguments[d], null !== l)
            for (f = 0; f < c.length; f++)
                if (m = c[f], m.name == l) {
                    b[l] = m.content;
                    break
                }
    return b
};
DIL.modules.siteCatalyst = {
    dil: null,
    handle: DIL.modules.helpers.handleModuleError,
    init: function(b, c, d) {
        try {
            var f = this, k = {
                name: "DIL Site Catalyst Module Error"
            }, l = function(b) {
                k.message = b;
                DIL.errorModule.handleError(k);
                return b
            };
            this.dil = null;
            if (c instanceof DIL)
                this.dil = c;
            else 
                return l("dilInstance is not a valid instance of DIL");
            k.partner = c.api.getPartner();
            if (b !== Object(b))
                return l("siteCatalystReportingSuite is not an object");
            window.AppMeasurement_Module_DIL = b.m_DIL = function(b) {
                var c = "function" ===
                typeof b.m_i ? b.m_i("DIL"): this;
                if (c !== Object(c))
                    return l("m is not an object");
                c.trackVars = f.constructTrackVars(d);
                c.d = 0;
                c.s = b;
                c._t = function() {
                    var b, c, f = "," + this.trackVars + ",", d = this.s, k, m = [];
                    k = [];
                    var t = {}, w=!1;
                    if (d !== Object(d))
                        return l("Error in m._t function: s is not an object");
                    if (this.d) {
                        if ("function" == typeof d.foreachVar)
                            d.foreachVar(function(b, c) {
                                t[b] = c;
                                w=!0
                            }, this.trackVars);
                        else {
                            if (!(d.va_t instanceof Array))
                                return l("Error in m._t function: s.va_t is not an array");
                            if (d.lightProfileID)(b =
                            d.lightTrackVars) && (b = "," + b + "," + d.vl_mr + ",");
                            else if (d.pe || d.linkType)
                                b = d.linkTrackVars, d.pe && (c = d.pe.substring(0, 1).toUpperCase() + d.pe.substring(1), d[c] && (b = d[c].trackVars)), b && (b = "," + b + "," + d.vl_l + "," + d.vl_l2 + ",");
                            if (b) {
                                c = 0;
                                for (m = b.split(","); c < m.length; c++)
                                    0 <= f.indexOf("," + m[c] + ",") && k.push(m[c]);
                                k.length && (f = "," + k.join(",") + ",")
                            }
                            k = 0;
                            for (c = d.va_t.length; k < c; k++)
                                b = d.va_t[k], 0 <= f.indexOf("," + b + ",") && (null != d[b] && "" !== d[b]) && (t[b] = d[b], w=!0)
                            }
                        w && this.d.api.signals(t, "c_").submit()
                    }
                }
            };
            b.loadModule("DIL");
            b.DIL.d = c;
            return k.message ? k.message : "DIL.modules.siteCatalyst.init() completed with no errors"
        } catch (m) {
            return this.handle(m, "DIL.modules.siteCatalyst.init() caught error with message ", this.dil)
        }
    },
    constructTrackVars: function(b) {
        var c = [], d, f, k, l, m;
        if (b === Object(b)) {
            d = b.names;
            if (d instanceof Array && (k = d.length))
                for (f = 0; f < k; f++)
                    l = d[f], "string" == typeof l && l.length && c.push(l);
            b = b.iteratedNames;
            if (b instanceof Array && (k = b.length))
                for (f = 0; f < k; f++)
                    if (d = b[f], d === Object(d) && (l = d.name, m = parseInt(d.maxIndex,
                    10), "string" == typeof l && l.length&&!isNaN(m) && 0 <= m))
                        for (d = 0; d <= m; d++)
                            c.push(l + d);
            if (c.length)
                return c.join(",")
        }
        return this.constructTrackVars({
            names: "pageName channel campaign products events pe pev1 pev2 pev3".split(" "),
            iteratedNames: [{
                name: "prop",
                maxIndex: 75
            }, {
                name: "eVar",
                maxIndex: 75
            }
            ]
        })
    }
};
