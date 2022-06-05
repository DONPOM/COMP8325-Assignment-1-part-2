function amzn_ads(e) {
    "use strict";
    try {
        amznads.updateAds(e)
    } catch (a) {
        try {
            console.log("amzn_ads: " + a)
        } catch (t) {}
    }
}
function aax_write(e, a) {
    e.write(a), e.close()
}
function aax_render_ad(e) {
    if (e.passback)
        return void aax_write(document, e.html);
    var a = e.slotSize;
    if (!a)
        return void aax_write(document, e.html);
    var t = a.indexOf("x"), n = a.substring(0, t), o = a.substring(t + 1), d = "amznad" + Math.round(1e6 * Math.random());
    aax_write(document, '<iframe id="' + d + '" width="' + n + '" height="' + o + '" src="javascript:\'\'" scrolling="no" frameborder="0" marginwidth="0" marginheight="0" bgcolor="#FFFFFF" topmargin="0" leftmargin="0" rightmargin="0" bottommargin="0"></iframe>');
    var r;
    try {
        r = document.getElementById(d);
        var s = r.contentWindow || r.contentDocument;
        s.document && (s = s.document), aax_write(s, e.html)
    } catch (g) {
        r && (r.style.display = "none")
    }
}
var amzn_console = function() {
    "use strict";
    var e = {};
    return e.log = function() {}, e
}();
window.console && (amzn_console = window.console);
var amznads = function(e, a, t, n) {
    "use strict";
    var o = "https:" === a.location.protocol;
    return e.protocol = o ? "https://" : "http://", e.host = "aax.amazon-adsystem.com", e.dtb_svc = "/e/dtb/bid", e.pb_svc = "/x/getad", e.debug_mode = e.debug_mode ||!1, e.MIN_TIMEOUT = 0, e.DEFAULT_TIMEOUT = 1e3, e.targetingKey = "amznslots", e.vidKey = "amzn_vid", e.tasks = e.tasks || [], e.log = function(e) {
        try {
            n.log(e)
        } catch (a) {}
    }, e.debug_mode && e.log("Initiating amznads"), e.ads || (e.ads = {}), e.updateAds = function(a) {
        if (e.ads = a.ads, a.vads) {
            e.ads || (e.ads = {});
            var t;
            for (t in a.vads)
                a.vads.hasOwnProperty(t) && (e.ads[t] = a.vads[t], e.amzn_vid = a.vads[t])
        }
        e.debug_mode && e.log("Updated ads. Executing rest of the task queue"), e.doAllTasks(), e.tasks.push = function(a) {
            e.doTask(a)
        }
    }, e.saveAds = function(a) {
        e.saved_ads = a.ads, e.updateAds(a)
    }, e.getAdForSlot = function(t, n, o) {
        e.src_id = t;
        var d = d || {}, r = d.u;
        r || (r = e.getReferrerURL()), r&&-1 !== r.indexOf("amzn_debug_mode") && (e.debug_mode=!0), e.debug_mode && e.log("amznads.getAdForSlot: Using url=" + r);
        var s = "src=" + e.src_id + "&slot_uuid=" + n + "&c=100&u=" + r + "&cb=" + Math.round(1e7 * Math.random()), g = e.protocol + e.host + e.pb_svc + "?jsd=1&" + s;
        e.debug_mode && e.log("amznads.getAdAdForSlot: " + (o ? "Async " : "") + "Call to: " + g), o ? e.appendScriptTag(g) : aax_write(a, "<script type='text/javascript' src='" + g + "'></script>")
    }, e.getAdsCallback = function(a, t, n, o, d) {
        var r = null, d = d || {};
        try {
            t && "function" == typeof t && (r = e.handleCallBack(t, n))
        } catch (s) {
            e.debug_mode && e.log("amznads.getAdsAsyncWithCallback(): Error occured in setting up callback functionality." + s)
        }
        d.to || (d.to = n), e.doGetAdsAsync(a, o, d, r)
    }, e.getAdsAsync = function(a, t, n) {
        e.doGetAdsAsync(a, t, n)
    }, e.handleCallBack = function(a, n) {
        var o=!1, d = null, r = function(e) {
            if (!o) {
                try {
                    a(e), d && clearTimeout(d)
                } catch (t) {}
                o=!0
            }
        }, s = e.getValidMilliseconds(n);
        return d = s ? t.setTimeout(r, s) : t.setTimeout(r, e.DEFAULT_TIMEOUT), r
    }, e.getValidMilliseconds = function(a) {
        if (!a)
            return !1;
        try {
            var t=~~Number(a);
            if (t > e.MIN_TIMEOUT)
                return t
        } catch (n) {
            return e.debug_mode && e.log("amznads.isValidMilliseconds(): Invalid timeout specified." + n), !1
        }
        return !1
    }, e.getAds = function(t, n, o, d) {
        if (d)
            return void e.doGetAdsAsync(t, n, o);
        var r = e.getScriptSource(t, n, o);
        e.debug_mode && e.log("amznads.getAds: Call to: " + r), aax_write(a, "<script type='text/javascript' src='" + r + "'></script>")
    }, e.doGetAdsAsync = function(a, t, n, o) {
        var d = e.getScriptSource(a, t, n);
        e.debug_mode && e.log("amznads.getAds: Async Call to: " + d), e.appendScriptTag(d, o)
    }, e.getScriptSource = function(t, n, o) {
        e.src_id = t;
        var o = o || {}, d = o.u, r = o.d, s = o.to;
        if (d || (d = e.getReferrerURL()), d&&-1 !== d.indexOf("amzn_debug_mode") && (e.debug_mode=!0), e.ads && (e.debug_mode && e.log("amznads.getAds(): clear out existing ads"), e.clearTargetingFromGPTAsync(), e.ads = {}), e.saved_ads && (e.ads = e.saved_ads), r)
            try {
                a.domain = r, e.debug_mode && e.log("amznads.getAds(): Using domain=" + r)
        } catch (g) {
            e.debug_mode && e.log("amznads.getAds(): Unable to override document domain with '" + r + "'; exception=" + g)
        }
        e.debug_mode && e.log("amznads.getAds(): Using url=" + d);
        var i = "src=" + t + "&u=" + d + "&cb=" + Math.round(1e7 * Math.random());
        n && (i += "&sz=" + n), s && (i += "&t=" + s);
        var c = e.protocol + e.host + e.dtb_svc + "?" + i;
        return c
    }, e.getReferrerURL = function() {
        var n = encodeURIComponent(a.documentURI);
        try {
            n = encodeURIComponent(t.top.location.href), n && "undefined" != n || (n = e.detectIframeAndGetURL())
        } catch (o) {
            n = e.detectIframeAndGetURL()
        }
        return n
    }, e.detectIframeAndGetURL = function() {
        try {
            if (t.top !== t.self)
                return e.debug_mode && e.log("Script is loaded within iFrame. url=" + url), encodeURIComponent(a.referrer)
        } catch (n) {
            return encodeURIComponent(a.documentURI)
        }
    }, e.appendScriptTag = function(t, n) {
        var o = a.getElementsByTagName("script")[0], d = a.createElement("script");
        d.type = "text/javascript", d.async=!0, d.src = t;
        try {
            n && "function" == typeof n && (d.readyState ? (d.onreadystatechange = function() {
                ("loaded" == d.readyState || "complete" == d.readyState) && (d.onreadystatechange = null, n(!0))
            }, e.debug_mode && e.log("amznads.appendScriptTag: setting callBack function for < IE-8 ")) : (d.onload = function() {
                n(!0)
            }, e.debug_mode && e.log("amznads.appendScriptTag: setting callBack function for all other browsers exluding  < IE-8")))
        } catch (r) {
            e.debug_mode && e.log("amznads.appendScriptTag: Could not associate callBack function; " + r)
        }
        o.parentNode.insertBefore(d, o)
    }, e.renderAd = function(a, t) {
        if (e.debug_mode && e.log("amznads.renderAd: key=" + t + "; ad-tag=" + e.ads[t]), e.ads[t])
            aax_write(a, e.ads[t]);
        else {
            var n = new Object;
            n.c = "dtb", n.src = e.src_id, n.kvmismatch = 1, n.pubReturnedKey = t, n.aaxReturnedKeys = e.getTokens(), n.cb = Math.round(1e7 * Math.random());
            try {
                n.u = encodeURIComponent(location.host + location.pathname), navigator && (n.ua = encodeURIComponent(navigator.userAgent))
            } catch (o) {}
            var d = encodeURIComponent(JSON.stringify(n)), r = e.protocol + e.host + "/x/px/p/0/" + d;
            e.debug_mode && e.log("amznads.renderAd: keyValueMismatch detected, pubReturnedKey=" + t + ", aaxReturnedKeys=" + e.getTokens()), aax_write(a, "<img src='" + r + "'/>")
        }
    }, e.hasAds = function(a) {
        var t;
        if (!a)
            try {
                return Object.keys(e.ads).length > 0
        } catch (n) {
            e.debug_mode && e.log("amznads.hasAds: looks like IE 8 (and below): " + n);
            for (t in e.ads)
                if (e.ads.hasOwnProperty(t))
                    return !0
        }
        for (t in e.ads)
            if (e.ads.hasOwnProperty(t) && t.indexOf(a) > 0)
                return !0;
        return !1
    }, e.getTargeting = function() {
        var a = {};
        return a[e.targetingKey] = e.getTokens(), a[e.vidKey] = e.amzn_vid, a
    }, e.setTargeting = function(a, t) {
        var n;
        for (n in e.ads)
            if (e.ads.hasOwnProperty(n)) {
                if (t && n.indexOf(t) < 0)
                    continue;
                    a(n, "1")
            }
    }, e.setTargetingForGPTAsync = function(a) {
        try {
            if (a) {
                e.targetingKey = a;
                var t = e.getTokens();
                "undefined" != typeof t && t.length > 0 && googletag.cmd.push(function() {
                    googletag.pubads().setTargeting(a, t), googletag.pubads().setTargeting(e.vidKey, e.amzn_vid)
                })
            } else {
                var n;
                for (n in e.ads)
                    e.ads.hasOwnProperty(n)&&!function() {
                        var a = n;
                        e.debug_mode && e.log("amznads.setTargetingForGPTAsync: pushing localKey=" + a), googletag.cmd.push(function() {
                            amznads.debug_mode && amznads.log("amznads.setTargetingForGPTAsync: localKey=" + a), googletag.pubads().setTargeting(a, "1"), googletag.pubads().setTargeting(e.vidKey, e.amzn_vid)
                        })
                    }()
                }
            e.debug_mode && e.log("amznads.setTargetingForGPTAsync: Completed successfully. Number of ads returned by Amazon: " + Object.keys(e.ads).length)
        } catch (o) {
            e.debug_mode && e.log("amznads.setTargetingForGPTAsync: ERROR - " + o)
        }
    }, e.setTargetingForGPTSync = function(a) {
        try {
            if (a) {
                e.targetingKey = a;
                var t = e.getTokens();
                "undefined" != typeof t && t.length > 0 && (googletag.pubads().setTargeting(a, t), googletag.pubads().setTargeting(e.vidKey, e.amzn_vid))
            } else {
                var n;
                for (n in e.ads)
                    e.ads.hasOwnProperty(n) && (googletag.pubads().setTargeting(n, "1"), googletag.pubads().setTargeting(e.vidKey, e.amzn_vid))
                }
            e.debug_mode && e.log("amznads.setTargetingForGPTSync: Completed successfully. Number of ads returned by Amazon: " + Object.keys(e.ads).length)
        } catch (o) {
            e.debug_mode && e.log("amznads.setTargetingForGPTSync: ERROR - " + o)
        }
    }, e.clearTargetingFromGPTAsync = function() {
        try {
            googletag && googletag.pubads() && googletag.pubads().clearTargeting(e.targetingKey), googletag.pubads().clearTargeting(e.vidKey)
        } catch (a) {}
    }, e.appendTargetingToAdServerUrl = function(a) {
        var t = a;
        try {
            - 1 === a.indexOf("?") && (a += "?");
            var n;
            for (n in e.ads)
                e.ads.hasOwnProperty(n) && (a += "&" + n + "=1");
            e.debug_mode && e.log("amznads.appendTargetingToAdServerUrl: Completed successfully. Number of ads returned by Amazon: " + e.ads.length)
        } catch (o) {
            e.debug_mode && e.log("amznads.appendTargetingToAdServerUrl: ERROR - " + o)
        }
        return e.debug_mode && e.log("amznads.appendTargetingToAdServerUrl: input url: " + t + "\nreturning url: " + a), a
    }, e.appendTargetingToQueryString = function(a) {
        var t = a;
        try {
            var n;
            for (n in e.ads)
                e.ads.hasOwnProperty(n) && (a += "&" + n + "=1")
        } catch (o) {
            e.debug_mode && e.log("amznads.appendTargetingToQueryString: ERROR - " + o)
        }
        return e.debug_mode && e.log("amznads.appendTargetingToQueryString: input query-string:" + t + "\nreturning query-string:" + a), a
    }, e.getTokens = function(a) {
        var t, n = [];
        try {
            for (t in e.ads)
                if (e.ads.hasOwnProperty(t)) {
                    if (a && t.indexOf(a) < 0)
                        continue;
                        n.push(t)
                }
        } catch (o) {
            e.debug_mode && e.log("amznads.getTokens: ERROR - " + o)
        }
        return e.debug_mode && e.log("amznads.getTokens: returning tokens = " + n), n
    }, e.getKeys = e.getTokens, e.getVid = function() {
        return e.amzn_vid
    }, e.doAllTasks = function() {
        for (; e.tasks.length > 0;) {
            var a = e.tasks.shift();
            e.doTask(a)
        }
    }, e.doTask = function(a) {
        try {
            a.call()
        } catch (t) {
            e.debug_mode && e.log("Failed calling task: " + t)
        }
    }, e.tryGetAdsAsync = function() {
        e.asyncParams && e.getAdsCallback(e.asyncParams.id, e.asyncParams.callbackFn, e.asyncParams.timeout, e.asyncParams.size, e.asyncParams.data)
    }, e
}(amznads || {}, document, window, amzn_console);
amznads.tryGetAdsAsync(), window.amzn_ads = amzn_ads, window.amznads = amznads;
