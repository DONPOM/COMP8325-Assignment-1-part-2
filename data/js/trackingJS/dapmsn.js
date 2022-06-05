function verifyDapResize(n) {
    var t = dapMgr.adCont;
    dapMgr.lg("verifyDapResize; called for: " + n), t[n].resizeCalled || (dapMgr.lg("verifyDapResize; we haven't resized! we need to now! "), dap_Resize(t[n].ifrmid, t[n].w, t[n].h))
}
function dap_Resize(n, t, i) {
    dapMgr.lg("dap_Resize; width: " + t + " height: " + i);
    var r = document.getElementById(n);
    r && (r.width = t, r.height = i, t > 0 && i > 0 && dapMgr.saveAdResize(n, t, i))
}
function setCreativeData(n, t) {
    dapMgr.lg("setCreativeData; "), dapMgr.creativeData[n] = t, t.Viewability && t.Viewability == "true" && (dapMgr.lg("setCreativeData; spinning up viewability "), microsoft.advertising.sdk.products.viewability.viewableImpression(50, document.getElementById(n), t)), typeof microsoft.advertising.sdk.products.feedback != "undefined" && (microsoft.advertising.sdk.products.feedback.showIAF(n) || microsoft.advertising.sdk.products.feedback.showAdChoices(n)) && (dapMgr.lg("setCreativeData; calling displayFeedback()"), microsoft.advertising.sdk.products.feedback.displayFeedback(dapMgr.iafOpt[n].idx, dapMgr.iafOpt[n].ft))
}
function ShowAcb() {}
var _dapUtils = new function() {
    var n = navigator.userAgent.toLowerCase(), f = navigator.appVersion.toLowerCase(), i = parseFloat(f), u = parseInt(i), s = n.indexOf("mac")!=-1, o, e, r, t;
    this.is_opera = n.indexOf("opera")!=-1, this.is_ff = n.indexOf("firefox")!=-1, e = f.indexOf("trident/"), e!=-1 && (o = parseFloat(f.substr(e + 8, 3))), r = f.indexOf("msie"), r!=-1 && (s ? (r = n.indexOf("msie"), i = parseFloat(n.substring(r + 5, n.indexOf(";", r)))) : i = parseFloat(f.substring(r + 5, f.indexOf(";", r))), u = parseInt(i)), r==-1 && e!=-1 && (i = parseFloat(o) + 4, u = parseInt(i)), t = (r!=-1 || e!=-1)&&!this.is_opera, this.is_ie = t, this.is_ie5up = t && i >= 5, this.is_ie5_5 = t && n.indexOf("msie 5.5")!=-1, this.is_ie5_5up = t && i >= 5.5, this.is_ie6 = t && u == 6, this.is_ie6up = t && i >= 6, this.is_ie9 = t && u == 9, this.is_ie9down = t && i <= 9, this.is_ie9up = t && i >= 9, this.is_trident4up = t && o >= 4, this.is_trident5up = t && o >= 5, this.is_trident6up = t && o >= 6, this.is_win = n.indexOf("win")!=-1 || n.indexOf("16bit")!=-1, s && (this.is_win=!s), this.is_ff && (this.ffPos = n.indexOf("firefox"), n.length > this.ffPos + 8 && (u = parseInt(n.substring(this.ffPos + 8))), n.length > this.ffPos + 10 && (i = parseInt(n.substring(this.ffPos + 10)))), this.is_mac = s, this.minorVer = i, this.majorVer = u, this.getCurrentStyle = function(n) {
        return window.getComputedStyle ? window.getComputedStyle(n, null) ? window.getComputedStyle(n, null) : document.defaultView.getComputedStyle(n, null) : n.currentStyle
    }
}, dapMgr = new function() {
    var r = ["//rad.msn.com/ADSAdClient31.dll?GetSAd=", "//a.rad.msn.com/ADSAdClient31.dll?GetSAd=", "//b.rad.msn.com/ADSAdClient31.dll?GetSAd="], i, n, t;
    this.creativeData = {};
    var f = 0, e = 100, u = _dapUtils.is_ie && _dapUtils.majorVer < 8 ? 2: 6;
    this.adCont = [], this.unblockingOnLoad=!1, this.enableUnblockingOnload = function(n) {
        dapMgr.unblockingOnLoad = n, n==!0 && dapMgr.setOnloadHandler(window, function() {
            dapMgr.enableUnblockingOnload(!1)
        })
    }, i = function(n, t, i, r, u) {
        this.qs = n, this.divid = t, this.ifrmid = i, this.w = r, this.h = u, this.resizeCalled=!1
    }, this.lg = function() {}, this.renderAd = function(n, t, r, u, f, o, s) {
        var c, h, l;
        dapMgr.lg("renderAd; qs: " + t + " divid: " + n), c = this.adCont, h = dapMgr.getAdItemIndex(n);
        if (h>-1)
            c[h].qs = t, c[h].divid = n, c[h].w = r, c[h].h = u;
        else {
            if (c.length < e)
                l = new i(t, n, "dapIfM" + c.length, r, u), c.push(l);
            else 
                return;
            h = c.length - 1
        }
        dapMgr.iafOpt || (dapMgr.iafOpt = {}), dapMgr.iafOpt[c[h].ifrmid] = {
            idx: h,
            fb: f,
            ft: o,
            ac: s
        }, dapMgr.displayAd(h)
    }, this.saveAdResize = function(n, t, i) {
        var u, r;
        for (dapMgr.lg("saveAdResize; width: " + t + " height: " + i), u = this.adCont, r = 0; r < u.length; r++)
            if (u[r].ifrmid == n) {
                u[r].w = t, u[r].h = i, u[r].resizeCalled=!0, dapMgr.lg("saveAdResize; we have saved our resize! ");
                return 
            }
    }, this.enableACB = function() {}, this.getAdItemIndex = function(n) {
        for (var t = 0; t < dapMgr.adCont.length; t++)
            if (dapMgr.adCont[t].divid == n)
                return t
    }, this.setOnloadHandler = function(n, t) {
        var i = function() {
            n.attachEvent ? (n.detachEvent("onload", i), t()) : n.addEventListener && (n.removeEventListener("load", i, !1), t())
        };
        n.attachEvent ? n.attachEvent("onload", i) : n.addEventListener && n.addEventListener("load", i, !1)
    }, this.displayAd = function(n) {
        var l, a, h, i, v, c;
        dapMgr.lg("displayAd; idx: " + n);
        var e = dapMgr.adCont, s = r[Math.floor(f++%(r.length * u) / u)], o = document.getElementById(e[n].divid);
        if (!o)
            return;
        if (!e[n].qs || e[n].qs.length == 0)
            return;
        l = _dapUtils.getCurrentStyle(o);
        if (l) {
            a = l.display;
            if (a == "none" || a == "hidden")
                return 
        }
        h = e[n].ifrmid, i = document.createElement("iframe"), i.id = h, i.name = h, i.src = "about:blank", i.width = e[n].w, i.height = e[n].h, i.scrolling = "no", i.frameBorder = "0", i.allowTransparency=!0, s.length > 0 && (s += "&DPJS=8", s += microsoft.advertising.sdk.products.viewability.canReachTop==!0 ? "&VWS=1" : "&VWS=0"), (t.indexOf("&ID=")===-1 || t.indexOf("&MUID=")===-1) && (t = dapMgr.dapQSTracking()), s += t, v = e[n].qs.replace(/'/g, ""), c = dapMgr.getDapOutput(s + v, h, n);
        if (_dapUtils.is_ie && (_dapUtils.is_trident5up || dapMgr.unblockingOnLoad==!0 && (_dapUtils.majorVer >= 8 || _dapUtils.is_trident4up)))
            o.insertBefore(i, o.firstChild), dapMgr.lg("onloadHandler; using unblocking, idx: " + n), dapMgr.setOnloadHandler(i, function() {
                dapMgr.onloadHandler(i, n, c, e)
            });
        else {
            dapMgr.lg("onloadHandler; not using unblocking, idx: " + n), o.insertBefore(i, o.firstChild);
            dapMgr.onloadHandler(i, n, c, e)
        }
    }, this.onloadHandler = function(n, t, i, r) {
        dapMgr.lg("onloadHandler; idx: " + t);
        var u;
        try {
            u = n.contentDocument || n.contentWindow.document
        } catch (f) {
            u = null
        }
        u && u.write ? (u.write(i), _dapUtils.is_ie9down || _dapUtils.is_opera || u.close()) : n.src = "javascript:'" + i + "'", typeof microsoft.advertising.sdk.products.feedback != "undefined" && dapMgr.setOnloadHandler(n, function() {
            (microsoft.advertising.sdk.products.feedback.showIAF(r[t].ifrmid) || microsoft.advertising.sdk.products.feedback.showAdChoices(r[t].ifrmid)) && (dapMgr.lg("onloadHandler; calling displayFeedback() "), microsoft.advertising.sdk.products.feedback.displayFeedback(t, dapMgr.iafOpt[r[t].ifrmid].ft))
        })
    }, this.getDapOutput = function(n, t, i) {
        var r = '<html><head><title>Advertisement</title></head><body id="' + t + '" leftmargin="0" topmargin="0" style="background-color:transparent"><script type="text/javascript">var inDapIF=true; var inDapMgrIf=true;';
        return document.domain && location.hostname != document.domain && (r += 'document.domain="' + document.domain + '";'), r += '<\/script><script type="text/javascript">function startTimer(){if (event.srcElement.readyState == "complete") {parent.verifyDapResize(' + i + ');window.setTimeout("document.close();", 2000);}}<\/script><script type="text/javascript"  src="' + n + '" onreadystatechange="startTimer();" onload="parent.verifyDapResize(' + i + ');"><\/script></body></html>'
    }, n = function(n, t, i) {
        var u, r;
        if (n != null) {
            u = n.toLowerCase().indexOf(t.toLowerCase() + "=");
            if (u!=-1)
                return u += t.length + 1, r = n.indexOf(i, u), r = r==-1 ? n.length : r, n.substring(u, r)
        }
        return ""
    }, this.dapQSTracking = function() {
        var u = document.cookie, f = escape(n(u, "mh", ";")), r = f != "" ? "&PN=" + f: "", t = escape(n(n(u, "ANON", ";"), "A", "&")), i = escape(n(u, "MUID", ";"));
        return t == "" && (t = i), r += t != "" ? "&ID=" + t : "", r += i != "" ? "&MUID=" + i : ""
    }, t = this.dapQSTracking()
};
(function() {
    for (var l = "microsoft.advertising.sdk.products", t = window, o = l.split("."), i, s, e = 0; e < o.length; e++) {
        i = o[e], typeof t[i] == "undefined" && (t[i] = {});
        if (typeof t[i] != "object")
            break;
        t = t[i]
    }
    var c = 1e3, y, n = {}, a = "/view/", r, f, u, h = {
        PID: !0,
        UIT: !0,
        TargetID: !0,
        PG: !0,
        ASID: !0,
        ANID: !0,
        FN: !0
    };
    t.viewability = {
        impressionEvents: {
            Viewed: "EV",
            ViewabilityUndetermined: "EVU"
        },
        canReachTop: !0,
        isFrameViewable: function(n, t) {
            return dapMgr.lg("isFrameViewable; start "), document.hasFocus() ? this.isAdBlockerActive(t)?!1 : this.isPercentOfAdOnWebpage(n, t)?!0 : !1 : (dapMgr.lg("isFrameViewable; we don't have focus!"), !1)
        },
        isAdBlockerActive: function(n) {
            var t = n.height <= 1;
            return t && dapMgr.lg("isAdBlockerActive; this ad is 1x1! the ad is broken!"), t
        },
        checkViewability: function(t) {
            dapMgr.lg("checkViewability; ifrmId: " + t);
            if (this.isNullOrUndefined(n[t].ifrm)) {
                dapMgr.lg("checkViewability; viewableAds[key].ifrm is null, lets remove it "), delete n[t], this.objectSize(n) > 0 || this.destroy();
                return 
            }
            this.isFrameViewable(n[t].percent, n[t].ifrm) && (this.fireImpression(n[t].url + a + n[t].creativeData.CID + "?EVT=" + this.impressionEvents.Viewed, n[t].creativeData), delete n[t], this.objectSize(n) > 0 || this.destroy())
        },
        objectSize: function(n) {
            var i = 0, t;
            for (t in n)
                n.hasOwnProperty(t) && i++;
            return i
        },
        fireImpression: function(n, t) {
            var i = this.buildURL(n, t, h);
            this.sendImpression(i)
        },
        sendImpression: function(n) {
            dapMgr.lg("sendImpression; viewablilityUrl: " + n);
            var t = new Image;
            t.src = n
        },
        buildURL: function(n, t, i) {
            n.indexOf("?")===-1 ? n += "?" : n.charAt(n.length - 1) !== "?" && n.charAt(n.length - 1) !== "&" && (n += "&");
            for (var r in t)
                t.hasOwnProperty(r) && (!i || i[r]) && (n += r + "=" + t[r] + "&");
            return n += "msts=" + + new Date
        },
        viewableImpression: function(t, i, e) {
            dapMgr.lg("viewableImpression; start");
            if (this.canReachTop==!1) {
                dapMgr.lg("viewableImpression; canReachTop is false, we can't gage viewability.");
                return 
            }
            if (this.isNullOrUndefined(e)) {
                dapMgr.lg("viewableImpression; The creativeData passed in was null or undefined.");
                return 
            }
            if (this.isNullOrUndefined(e.GSERV)) {
                dapMgr.lg("viewableImpression; creativeData['GSERV'] is not valid.");
                return 
            }
            e.CID || (dapMgr.lg("viewableImpression; creativeData['CID'] is not valid."), e.CID = "");
            if (this.isNullOrUndefined(i)) {
                dapMgr.lg("viewableImpression; ifrm is null or undefined.");
                return 
            }
            if (this.isNullOrUndefined(t)) {
                dapMgr.lg("viewableImpression; percentRequired is null or undefined.");
                return 
            }
            if (this.invalidPercentage(t)) {
                dapMgr.lg("viewableImpression; percentRequired is invalid.");
                return 
            }
            var o = this, s = t / 100;
            n[i.id] = {
                percent: s,
                ifrm: i,
                creativeData: e,
                url: e.GSERV,
                viewableCheckTimer: null
            }, this.startTimer(i.id), this.objectSize(n) > 1 || (r = function() {
                o.restartAllTimers()
            }, f = function() {
                o.restartAllTimers()
            }, u = function() {
                o.restartAllTimers()
            }, this.registerEvent(top, "scroll", "onscroll", r), this.registerEvent(top, "resize", "onresize", f), this.registerEvent(top, "focus", "onfocus", u))
        },
        startTimer: function(t) {
            clearTimeout(n[t].viewableCheckTimer);
            var i = this;
            n[t].viewableCheckTimer = setTimeout(function() {
                i.checkViewability(t)
            }, c)
        },
        restartAllTimers: function() {
            for (var i in n)
                n.hasOwnProperty(i) && this.startTimer(i)
        },
        registerEvent: function(n, t, i, r) {
            n.addEventListener ? n.addEventListener(t, r, !1) : n.attachEvent && n.attachEvent(i, r)
        },
        removeEvent: function(n, t, i, r) {
            n.addEventListener ? n.removeEventListener(t, r, !1) : n.attachEvent && n.detachEvent(i, r)
        },
        invalidPercentage: function(n) {
            return n > 100 || n < 0
        },
        isPercentOfAdOnWebpage: function(n, t) {
            var i = t.getBoundingClientRect(), e = top.innerWidth || top.document.documentElement.clientWidth, f = top.innerHeight || top.document.documentElement.clientHeight, r, u;
            if (i.top > f)
                return !1;
            r = parseInt(t.width), u = parseInt(t.height), i.left < 0 ? (r = Math.max(0, i.left + parseInt(t.width)), r > e && (r = e)) : i.left + parseInt(t.width) > e && (r = Math.max(0, e - i.left)), i.top < 0 ? (u = Math.max(0, i.top + parseInt(t.height)), u > f && (u = f)) : i.top + parseInt(t.height) > f && (u = Math.max(0, f - i.top));
            var s = Math.max(1, parseInt(t.height) * parseInt(t.width)), h = r * u, o = h / s;
            return dapMgr.lg("isPercentOfAdOnWebpage; percentVisible: " + o), o >= n
        },
        destroy: function() {
            r && this.removeEvent(top, "scroll", "onscroll", r), f && this.removeEvent(top, "resize", "onresize", f), u && this.removeEvent(top, "focus", "onfocus", u)
        },
        isNullOrUndefined: function(n) {
            return typeof n == "undefined" || n === null
        }
    };
    try {
        s = top.document, self !== top && (microsoft.advertising.sdk.products.viewability.canReachTop=!1)
    } catch (v) {
        dapMgr.lg(" err: " + v), microsoft.advertising.sdk.products.viewability.canReachTop=!1
    }
    s === undefined && (microsoft.advertising.sdk.products.viewability.canReachTop=!1), dapMgr.lg(" canReachTop: " + microsoft.advertising.sdk.products.viewability.canReachTop)
})()
