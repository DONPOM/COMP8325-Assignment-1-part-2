function getQueryValue(n, t) {
    var r = new RegExp("[\\?&]" + t + "=([^&#]*)", "gi"), i = r.exec(n);
    return i == null ? "" : decodeURIComponent(i[1].replace(/\+/g, " "))
}
function getStore(n) {
    var t = "ClosestStore.asmx", r, i;
    $(".store-geo[data-GeoStoreLocalServiceURL]").length && (t = $(".store-geo").first().attr("data-GeoStoreLocalServiceURL")), i = "POST", typeof n != "undefined" && (r = {
        latitude: JSON.stringify(n.coords.latitude),
        longitude: JSON.stringify(n.coords.longitude)
    }, t = t + "ClientGeo", i = "GET"), $.ajax({
        url: t,
        type: i,
        timeout: 5e3,
        data: r,
        contentType: "application/json; charset=UTF-8",
        dataType: "json",
        error: function() {
            $(".store-geo").remove(), $(".store-editorial").fadeIn(1e3)
        },
        success: function(n) {
            if (typeof n != "undefined" && typeof n.d != "undefined" && typeof n.d.City != "undefined" && n.d.City != "" && n.d.StoreUrl != "undefined" && n.d.StoreUrl != "") {
                var t = $(".store-geo:first").text();
                $(".store-geo a").html(t + " " + n.d.City), $(".store-geo a").attr("href", n.d.StoreUrl), $(".store-editorial").remove(), $(".store-geo").fadeIn(1e3)
            } else 
                $(".store-geo").remove(), $(".store-editorial").fadeIn(1e3)
        }
    })
}
var MSCom;
(function() {
    var r = [], n = this, o;
    this.Asimov = this.Asimov || {}, this.Asimov.uploadUrl = "https://vortex.data.microsoft.com/collect/v1", this.Asimov.correlationVectorTag = "cV", this.Asimov.correlationVectorHeader = "MS-CV";
    var u = 63, f = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", c = function() {
        for (var t = "", n = 0; n < 16; n++)
            t += f.charAt(Math.floor(Math.random() * f.length));
        return t
    };
    this.Asimov.CorrelationVector = function() {
        var t = c(), n = 0, i = function(n) {
            return n.getValue().length + 2 <= u?!0 : !1
        }, r = function() {
            return t.length + 1 + (n + 1 + "").length <= u?!0 : !1
        };
        this.getValue = function() {
            return t.concat(".", n)
        }, this.setValue = function(i) {
            if (Asimov.CorrelationVector.isValid(i)) {
                var r = i.lastIndexOf(".");
                t = i.substr(0, r), n = parseInt(i.substr(r + 1), 10)
            } else 
                throw "Cannot set invalid correlation vector value";
            return t.concat(".", n)
        }, this.extend = function() {
            return i(this) && (t = t.concat(".", n), n = 0), this.getValue()
        }, this.increment = function() {
            return r() && (n = n + 1), this.getValue()
        }
    }, o = new RegExp("^[" + f + "]{16}(.[0-9]+)+$"), this.Asimov.CorrelationVector.isValid = function(n) {
        return o.test(n) && n.length <= u
    }, this.Asimov.cv = new this.Asimov.CorrelationVector;
    var l = function() {
        return n.jQuery?!n.JSON ||!n.JSON.stringify ? (t("Unable to write event: the global JSON.stringify method does not exist"), !1) : !0 : (t("Unable to write event: jQuery is not present"), !1)
    }, e = function(n, i) {
        t("Failure sending data to vortex: " + i)
    }, s = function(t, i) {
        n.console && n.console.log && n.console.log("JSLL: Success sending data to vortex: " + i)
    }, t = function(t) {
        n.console && n.console.error && n.console.error("JSLL: " + t)
    }, i = function(n, t) {
        if (n == "string")
            return typeof t == "string" || t instanceof String || t instanceof Date;
        if (n == "bool")
            return typeof t == "boolean" || t instanceof Boolean;
        if (!(typeof t == "number") || t instanceof Number)
            return !1;
        if (n == "uint8") {
            if (t < 0 || t > 255 || t%1 != 0)
                return !1
        } else if (n == "uint16") {
            if (t < 0 || t > 65535 || t%1 != 0)
                return !1
        } else if (n == "uint32") {
            if (t < 0 || t > 4294967295 || t%1 != 0)
                return !1
        } else if (n == "uint64") {
            if (t < 0 || t > 18446744073709551615 || t%1 != 0)
                return !1
        } else if (n == "int8") {
            if (t<-128 || t > 127 || t%1 != 0)
                return !1
        } else if (n == "int16") {
            if (t<-32768 || t > 32767 || t%1 != 0)
                return !1
        } else if (n == "int32") {
            if (t<-2147483648 || t > 2147483647 || t%1 != 0)
                return !1
        } else if (n == "int64") {
            if (t<-9223372036854775808 || t > 9223372036854775807 || t%1 != 0)
                return !1
        } else if (n == "float") {
            if (t<-3402823e32 || t > 3402823e32)
                return !1
        } else if (n == "double" && (t<-17976931348623157e292 || t > 17976931348623157e292))
            return !1;
        return !0
    }, h = function(n, r, u) {
        var c = r.fields, b, f, e, a, s, v, o, y, p, w, l;
        for (b in c) {
            if (f = c[b], e = u[f.name], e === null || e === undefined) {
                if (f.req)
                    return t("Missing required property: " + f.name), !1;
                continue
            }
            if (f.type == "map") {
                a = {};
                for (s in e) {
                    if (!i(f.key, s))
                        return t("A key in the map was of the wrong type: " + f.name), !1;
                    if (!i(f.element, e[s]))
                        return t("A value in the map was of the wrong type: " + f.name), !1;
                    a[s] = e[s]
                }
                n[f.name] = a
            } else if (f.type == "list" || f.type == "set") {
                if (v = [], Object.prototype.toString.call(e) === "[object Array]")
                    for (o = 0; o < e.length; o++) {
                        if (e[o] != null&&!i(f.element, e[o]))
                            return t("The list contains a value of the wrong type: " + f.name), !1;
                            v[o] = e[o]
                    } else 
                        return t("The " + f.type + " " + f.name + " was not an array as expected"), !1;
                n[f.name] = v
            } else if (f.type == "struct") {
                if (y = {}, !h(y, f.def, e))
                    return !1;
                n[f.name] = y
            } else {
                if (!i(f.type, e))
                    return t("Property is the wrong type: " + f.name), !1;
                n[f.name] = e
            }
        }
        for (p in u) {
            for (w=!1, l = 0; l < c.length; l++)
                if (c[l].name == p) {
                    w=!0;
                    break
                }
            w || t("An unexpected property was found in the event content and dropped: " + p)
        }
        return !0
    };
    this.Asimov.writeEvent = function(t) {
        var r, f, h, c, o, u, a, i;
        if (t && l()) {
            if (r = [], n.jQuery.isArray(t))
                for (f = 0; f < t.length; f++)
                    h = n.Asimov._validateAndTranslateEvent(t[f]), h.success && r.push(h.event);
            else 
                c = n.Asimov._validateAndTranslateEvent(t), c.success && r.push(c.event);
            if (r.length != 0) {
                for (o = "", u = 0; u < r.length; u++)
                    u > 0 && (o += "\n"), o += n.JSON.stringify(r[u]);
                a = {
                    accepts: {
                        text: "application/json"
                    },
                    url: n.Asimov.uploadUrl,
                    type: "post",
                    dataType: "text",
                    cache: !1,
                    data: o,
                    crossDomain: !0,
                    headers: {
                        "Content-Type": "application/x-json-stream"
                    }
                }, i = window.jQuery.ajax(a), i.fail(e), i.fail ? i.fail(e) : requst.error(e), i.done ? i.done(s) : i.success(s)
            }
        }
    }, this.Asimov._validateAndTranslateEvent = function(i) {
        var u = {}, s = {}, f = {
            event: u,
            success: !1
        }, c, e, a, l, v, o;
        if (!i)
            return t("Unable to write null event"), f;
        if (!i.name)
            return t("Unable to write event with missing name"), f;
        if (!r.hasOwnProperty(i.name))
            return t("Unable to write event: a schema for the event named {" + i.name + "} does not exist"), f;
        if (!i.content)
            return t("Unable to write event: the event is missing content"), f;
        c = r[i.name];
        for (e in c)
            if (e !== "name") {
                if (a = c[e], !i.content.hasOwnProperty(e))
                    return t("unable to write event: missing expected part: " + e), f;
                    if (a.part == "C" ? l = s : (l = {}, s.item = l, s.type = e), !h(l, a.def, i.content[e]))
                        return f
            }
        for (v in i.content)
            c[v].part || t("An unexpected property was found in the event content and dropped: " + v);
        if (u.name = i.name, u.time = new Date, u.data = s, u.tags = {}, o = n.navigator && n.navigator.appVersion ? n.navigator.appVersion : "", u.os = o.indexOf("Win")!=-1 ? "Windows" : o.indexOf("Mac")!=-1 ? "MacOS" : o.indexOf("X11")!=-1 ? "Unix" : o.indexOf("Linux")!=-1 ? "Linux" : "Unknown", i.hasOwnProperty(this.correlationVectorTag))
            if (this.CorrelationVector.isValid(i[this.correlationVectorTag]))
                u.tags[this.correlationVectorTag] = i[this.correlationVectorTag];
            else 
                return t("The correlation vector value is invalid: " + i[this.correlationVectorTag]), f;
        else 
            u.tags[this.correlationVectorTag] = this.cv.getValue();
        return f.success=!0, f
    }, this.Asimov._registerSchemas = function(n) {
        for (var t = 0; t < n.length; t++)
            r[n[t].name] = n[t]
    }
})(), function() {
    this.Asimov._registerSchemas([{
        name: "Microsoft.Infrastructure.Events.MSCOMRendering.ClientError",
        "Microsoft.Infrastructure.Events.MSCOMRendering.ClientError": {
            part: "C",
            def: {
                fields: [{
                    req: !0,
                    name: "ErrorInfo",
                    type: "string"
                }, {
                    name: "WasDisplayed",
                    type: "bool"
                }
                ]
            }
        }
    }, {
        name: "Microsoft.Infrastructure.Events.MSCOMRendering.PageAction",
        "Ms.Content.PageAction": {
            part: "B",
            def: {
                fields: [{
                    req: !0,
                    name: "pageName",
                    type: "string"
                }, {
                    name: "uri",
                    type: "string"
                }, {
                    name: "destUri",
                    type: "string"
                }, {
                    name: "pageType",
                    type: "string"
                }, {
                    name: "pageTags",
                    type: "string"
                }, {
                    name: "product",
                    type: "string"
                }, {
                    name: "screenState",
                    type: "int32"
                }, {
                    name: "customSessionGuid",
                    type: "string"
                }, {
                    name: "impressionGuid",
                    type: "string"
                }, {
                    name: "actionInputMethod",
                    type: "int32"
                }, {
                    name: "behavior",
                    type: "int32"
                }, {
                    name: "contentJsonVer",
                    type: "float"
                }, {
                    name: "content",
                    type: "string"
                }
                ]
            }
        },
        "Microsoft.Infrastructure.Events.MSCOMRendering.PageAction": {
            part: "C",
            def: {
                fields: [{
                    req: !0,
                    name: "VisitorId",
                    type: "string"
                }, {
                    name: "GroupId",
                    type: "string"
                }, {
                    name: "FlightId",
                    type: "string"
                }, {
                    name: "TimeToAction",
                    type: "int32"
                }, {
                    name: "Route",
                    type: "string"
                }, {
                    name: "PageVersion",
                    type: "string"
                }, {
                    name: "BrowserSize",
                    type: "string"
                }, {
                    name: "PageSize",
                    type: "string"
                }
                ]
            }
        }
    }, {
        name: "Microsoft.Infrastructure.Events.MSCOMRendering.PageView",
        "Ms.Content.PageView": {
            part: "B",
            def: {
                fields: [{
                    req: !0,
                    name: "pageName",
                    type: "string"
                }, {
                    name: "uri",
                    type: "string"
                }, {
                    name: "referrerUri",
                    type: "string"
                }, {
                    name: "pageType",
                    type: "string"
                }, {
                    name: "pageTags",
                    type: "string"
                }, {
                    name: "product",
                    type: "string"
                }, {
                    name: "screenState",
                    type: "int32"
                }, {
                    name: "customSessionGuid",
                    type: "string"
                }, {
                    name: "impressionGuid",
                    type: "string"
                }, {
                    name: "contentJsonVer",
                    type: "float"
                }, {
                    name: "content",
                    type: "string"
                }
                ]
            }
        },
        "Microsoft.Infrastructure.Events.MSCOMRendering.PageView": {
            part: "C",
            def: {
                fields: [{
                    name: "VisitorId",
                    type: "string"
                }, {
                    name: "GroupId",
                    type: "string"
                }, {
                    name: "FlightId",
                    type: "string"
                }, {
                    name: "ClientUTCOffset",
                    type: "int32"
                }, {
                    req: !0,
                    name: "UserAgent",
                    type: "string"
                }, {
                    name: "BrowserLanguage",
                    type: "string"
                }, {
                    name: "DNTStatus",
                    type: "string"
                }, {
                    name: "CookiesEnabled",
                    type: "bool"
                }, {
                    name: "SilverLightInstalled",
                    type: "bool"
                }, {
                    name: "SilverLightEnabledOnPage",
                    type: "bool"
                }, {
                    name: "SilverLightVersion",
                    type: "string"
                }, {
                    name: "FlashInstalled",
                    type: "bool"
                }, {
                    name: "FlashVersion",
                    type: "string"
                }, {
                    name: "BrowserSize",
                    type: "string"
                }, {
                    name: "Cookies",
                    type: "string"
                }, {
                    name: "PageLoadTime",
                    type: "int32"
                }, {
                    name: "PageTitle",
                    type: "string"
                }, {
                    name: "Route",
                    type: "string"
                }, {
                    name: "PageVersion",
                    type: "string"
                }, {
                    name: "ScreenResolution",
                    type: "string"
                }, {
                    name: "PageSize",
                    type: "string"
                }, {
                    name: "Scrl",
                    type: "string"
                }
                ]
            }
        }
    }, {
        name: "Microsoft.Infrastructure.Events.MSCOMRendering.TimeSpan",
        "Microsoft.Infrastructure.Events.MSCOMRendering.TimeSpan": {
            part: "C",
            def: {
                fields: [{
                    name: "RequestUrl",
                    type: "string"
                }, {
                    name: "Culture",
                    type: "string"
                }, {
                    name: "UserAgent",
                    type: "string"
                }, {
                    name: "BeginTime",
                    type: "string"
                }, {
                    name: "PageLoadTime",
                    type: "int32"
                }, {
                    name: "PageTimingDetails",
                    type: "string"
                }, {
                    name: "TotalRequests",
                    type: "int32"
                }, {
                    name: "SecondaryResourceDetails",
                    type: "string"
                }, {
                    name: "AdditionalInformation",
                    type: "string"
                }, {
                    name: "VisitorId",
                    type: "string"
                }
                ]
            }
        }
    }, {
        name: "Ms.Content.PageAction",
        "Ms.Content.PageAction": {
            part: "B",
            def: {
                fields: [{
                    req: !0,
                    name: "pageName",
                    type: "string"
                }, {
                    name: "uri",
                    type: "string"
                }, {
                    name: "destUri",
                    type: "string"
                }, {
                    name: "pageType",
                    type: "string"
                }, {
                    name: "pageTags",
                    type: "string"
                }, {
                    name: "product",
                    type: "string"
                }, {
                    name: "screenState",
                    type: "int32"
                }, {
                    name: "customSessionGuid",
                    type: "string"
                }, {
                    name: "impressionGuid",
                    type: "string"
                }, {
                    name: "actionInputMethod",
                    type: "int32"
                }, {
                    name: "behavior",
                    type: "int32"
                }, {
                    name: "contentJsonVer",
                    type: "float"
                }, {
                    name: "content",
                    type: "string"
                }
                ]
            }
        }
    }, {
        name: "Ms.Content.PageView",
        "Ms.Content.PageView": {
            part: "B",
            def: {
                fields: [{
                    req: !0,
                    name: "pageName",
                    type: "string"
                }, {
                    name: "uri",
                    type: "string"
                }, {
                    name: "referrerUri",
                    type: "string"
                }, {
                    name: "pageType",
                    type: "string"
                }, {
                    name: "pageTags",
                    type: "string"
                }, {
                    name: "product",
                    type: "string"
                }, {
                    name: "screenState",
                    type: "int32"
                }, {
                    name: "customSessionGuid",
                    type: "string"
                }, {
                    name: "impressionGuid",
                    type: "string"
                }, {
                    name: "contentJsonVer",
                    type: "float"
                }, {
                    name: "content",
                    type: "string"
                }
                ]
            }
        }
    }
    ])
}(), window.MSCOMRendering === undefined && (window.MSCOMRendering = {}), MSCOMRendering.Jll = function() {
    function a() {
        v(), e = y(), d(), $(window).load(function() {
            var e = window.location.href, t, n, u, f, i;
            if (window.performance != undefined) {
                var o = window.performance.timing.domComplete - window.performance.timing.fetchStart, l =+ new Date, a = l - window.performance.timing.navigationStart, v = {
                    domLoadTime: o,
                    timing: window.performance.timing
                }, s=!0, r = [];
                try {
                    for (t = window.performance.getEntries(), n = 0; n < t.length; n++)
                        u = t[n].name.indexOf("c.microsoft.com")>-1 || t[n].name.indexOf("tags.bluekai.com")>-1 ? t[n].name.substring(0, 100) : t[n].name, r.push({
                            name: u,
                            initiatorType: t[n].initiatorType,
                            entryType: t[n].entryType,
                            type: t[n].type,
                            startTime: t[n].startTime,
                            fetchStart: t[n].fetchStart,
                            duration: t[n].duration
                        })
                    } catch (y) {
                    s=!1
                }
                f = "", i = "", s ? (i = JSON.stringify(v), f = JSON.stringify(r)) : i = '{"domLoadTime": ' + o + ',"timing" {"navigationStart":' + window.performance.timing.navigationStart + ',"unloadEventStart":' + window.performance.timing.unloadEventStart + ',"unloadEventEnd":' + window.performance.timing.unloadEventEnd + ',"redirectStart":' + window.performance.timing.redirectStart + ',"redirectEnd":' + window.performance.timing.redirectEnd + ',"fetchStart":' + window.performance.timing.fetchStart + ',"domainLookupStart":' + window.performance.timing.domainLookupStart + ',"domainLookupEnd":' + window.performance.timing.domainLookupEnd + ',"connectStart":' + window.performance.timing.connectStart + ',"connectEnd":' + window.performance.timing.connectEnd + ',"requestStart":' + window.performance.timing.requestStart + ',"responseStart":' + window.performance.timing.responseStart + ',"responseEnd":' + window.performance.timing.responseEnd + ',"domLoading":' + window.performance.timing.domLoading + ',"domInteractive":' + window.performance.timing.domInteractive + ',"domContentLoadedEventStart":' + window.performance.timing.domContentLoadedEventStart + ',"domContentLoadedEventEnd":' + window.performance.timing.domContentLoadedEventEnd + ',"domComplete":' + window.performance.timing.domComplete + ',"loadEventStart":' + window.performance.timing.loadEventStart + ',"loadEventEnd":' + window.performance.timing.loadEventEnd + ',"msFirstPaint":' + window.performance.timing.msFirstPaint + "}}", h(window.performance.timing.navigationStart.toString(), r.length, c(), f, i, "", e, a)
            } else 
                h("0", 0, c(), "Not Supported", "Not Supported", "", e, 0)
        })
    }
    function v() {
        var t = document.getElementsByTagName("meta"), i = "", n;
        if (t)
            for (n = 0; n < t.length; n++)
                if (t[n].getAttribute("name") == "CorrelationVector") {
                    i = t[n].getAttribute("content");
                    break
                }
        Asimov.cv = new Asimov.CorrelationVector, i != "" && Asimov.cv.setValue(i)
    }
    function c() {
        var t = document.getElementsByTagName("meta"), n;
        if (t)
            for (n = 0; n < t.length; n++)
                if (t[n].getAttribute("name") == "MscomContentLocale")
                    return t[n].getAttribute("content");
        return ""
    }
    function y() {
        var u = document.cookie.indexOf(s + "="), n, t, r;
        if (u==-1)
            return n = b(), document.cookie = s + "=" + n, n;
        for (t = document.cookie.split("; "), i = 0; i < t.length; i++)
            if (r = t[i].split("="), s === r.shift())
                return r.join("=");
        return e
    }
    function p() {
        n.domain = window.location.host, n.siteStem = "", n.queryString = "", window.location.pathname != "" && (n.siteStem = window.location.pathname), window.location.search != "" && (n.queryString = window.location.search)
    }
    function w() {
        n.screenResolution = "", n.screenResolutionWidth = "", n.screenResolutionHeight = "", typeof screen == "object" && (n.screenResolution = screen.width + "x" + screen.height, n.screenResolutionHeight = screen.height, n.screenResolutionWidth = screen.width)
    }
    function r() {
        return ((1 + Math.random()) * 65536 | 0).toString(16).substring(1)
    }
    function b() {
        return r() + r() + "-" + r() + "-" + r() + "-" + r() + "-" + r() + r() + r()
    }
    function k() {
        var t = new Date;
        tz = t.getTimezoneOffset(), n.timeZoneOffSet = tz/-60
    }
    function d() {
        var r = document.cookie.indexOf("MUID="), n, t;
        if (r!=-1)
            for (n = document.cookie.split("; "), i = 0; i < n.length; i++)
                if (t = n[i].split("="), "MUID" === t[0]) {
                    f = t[1];
                    break
                }
    }
    function g() {
        var i = "", e = new Date, t = document.cookie.indexOf(u + "="), r, f;
        if (t==-1) {
            if (MscomSetTimeStamp(), sessionId = e.getTime(), n.cookieEnabled==!1)
                return;
            i = u + "=" + sessionId
        } else 
            r = t + u.length + 1, f = document.cookie.length, i = u + "=" + document.cookie.substring(r, f);
        document.cookie = i, t = document.cookie.indexOf(u + "="), n.cookieEnabled = t==-1?!1 : !0
    }
    function nt() {
        var i, t, u, r;
        if (n.flashInstalled=!1, n.flashVersion = "", i = (new Date).getFullYear() - 1992, navigator.userAgent.indexOf("MSIE")!=-1)
            for (t = i; t > 0; t--)
                try {
                    u = new ActiveXObject("ShockwaveFlash.ShockwaveFlash." + t), n.flashInstalled=!0, n.flashVersion = t + ".0";
                    break
        } catch (f) {} else 
            navigator.plugins["Shockwave Flash"] && (n.flashInstalled=!0, r = navigator.plugins["Shockwave Flash"], n.flashVersion = r.description.split(" ")[2])
    }
    function tt() {
        var u, t, i, r;
        n.silverlightEnabled=!1, n.silverlightInstalled=!1, n.silverlightVersion = "0.0", window.Silverlight != undefined && (n.silverlightEnabled=!0);
        try {
            navigator.plugins["Silverlight Plug-In"] ? (u = navigator.plugins["Silverlight Plug-In"], n.silverlightInstalled=!0, t = u.description, t && (i = t.split("."), t = i[0] + "." + i[1], n.silverlightVersion = t)) : navigator.userAgent.indexOf("MSIE")!=-1 && (r = new ActiveXObject("AgControl.AgControl"), r && (n.silverlightInstalled=!0, n.silverlightVersion = it(r)))
        } catch (f) {}
    }
    function it(n) {
        for (var t = "", u = (new Date).getYear() - 2004, r, i = u; i > 0; i--)
            for (r = 9; r >= 0; r--)
                if (t = i + "." + r, n.IsVersionSupported(t))
                    return t;
        return t
    }
    function rt() {
        n.browserSize = "", document.body && document.body.clientWidth != undefined ? n.browserSize = document.body.clientWidth + "x" + document.body.clientHeight : document.documentElement && document.documentElement.clientWidth != undefined ? n.browserSize = document.documentElement.clientWidth + "x" + document.documentElement.clientHeight : window.innerWidth != undefined && (n.browserSize = window.innerWidth + "x" + window.innerHeight)
    }
    function ut() {
        var t =+ new Date, n;
        return window.performance && window.performance.timing && (n = window.performance.timing.domComplete, n !== 0) ? t - n : - 1
    }
    function l() {
        n.title = document.title, n.referrer = document.referrer, k(), g(), w(), rt(), nt(), tt(), p()
    }
    function ft(n) {
        return typeof n == "number" && n%1 == 0
    }
    function t(n) {
        return n ? n : ""
    }
    function o(n) {
        return n && ft(n) ? n : 0
    }
    function et(i) {
        n = i, l();
        var r = [{
            name: "Microsoft.Infrastructure.Events.MSCOMRendering.PageView",
            content: {
                "Ms.Content.PageView": {
                    pageName: n.title,
                    uri: window.location.href,
                    referrerUri: n.referrer,
                    pageType: "WebPage",
                    pageTags: "{'InteractionType': '" + t(n.interactiontype) + "', 'LinkId': '" + t(n.linkid) + "', 'LinkType': '" + t(n.linktype) + "', 'Environment': '" + t(n.env) + "', 'Lang': '" + t(n.lang) + "', 'Loc': '" + t(n.loc) + "',}",
                    product: "",
                    screenState: 0,
                    customSessionGuid: e,
                    impressionGuid: "",
                    contentJsonVer: 2,
                    content: ""
                },
                "Microsoft.Infrastructure.Events.MSCOMRendering.PageView": {
                    VisitorId: f,
                    GroupId: "",
                    FlightId: "",
                    ClientUTCOffset: n.timeZoneOffSet,
                    UserAgent: navigator.userAgent,
                    BrowserLanguage: navigator.userLanguage,
                    DNTStatus: navigator.doNotTrack,
                    CookiesEnabled: n.cookieEnabled,
                    SilverLightInstalled: n.silverlightInstalled,
                    SilverLightEnabledOnPage: n.silverlightEnabled,
                    SilverLightVersion: n.silverlightVersion,
                    FlashInstalled: n.flashInstalled,
                    FlashVersion: n.flashVersion,
                    BrowserSize: t(n.viewport),
                    Cookies: "",
                    PageLoadTime: - 1,
                    PageTitle: n.title,
                    Route: t(n.route),
                    PageVersion: "",
                    ScreenResolution: n.screenResolution,
                    PageSize: n.browserSize,
                    Scrl: t(n.scrollable)
                }
            }
        }
        ];
        window.Asimov.writeEvent(r)
    }
    function ot(i) {
        n = i, l();
        var r = [{
            name: "Microsoft.Infrastructure.Events.MSCOMRendering.PageAction",
            content: {
                "Ms.Content.PageAction": {
                    pageName: n.title,
                    uri: window.location.href,
                    destUri: t(n["wcs.ct"]),
                    pageType: "WebPage",
                    pageTags: "{'InteractionType': '" + t(n.interactiontype) + "', 'LinkId': '" + t(n.linkid) + "', 'LinkType': '" + t(n.linktype) + "', 'Environment': '" + t(n.env) + "', 'ClickedObjectId': '" + t(n["wcs.cid"]) + "', 'ClickedTargetUrl': '" + t(n["wcs.ct"]) + "', 'ClickedObjectName': '" + t(n["wcs.cn"]) + "', 'ClickedObjectType': '" + o(n.cot) + "', 'SearchType': '" + t(n.searchtype) + "', 'SearchQuery': '" + t(n.searchquery) + "', 'Index': '" + o(n.index) + "', 'PageArea': '" + t(n.pgarea) + "', 'ComponentGroup': '" + t(n.cmpgrp) + "', 'ComponentName': '" + t(n.cmpnm) + "', 'ComponentType': '" + t(n.cmptyp) + "', 'Lang': '" + t(n.lang) + "', 'Loc': '" + t(n.loc) + "',}",
                    product: "",
                    screenState: 0,
                    customSessionGuid: e,
                    impressionGuid: "",
                    actionInputMethod: 0,
                    behavior: 0,
                    contentJsonVer: 2,
                    content: ""
                },
                "Microsoft.Infrastructure.Events.MSCOMRendering.PageAction": {
                    VisitorId: "MUID",
                    GroupId: "",
                    FlightId: "",
                    TimeToAction: ut(),
                    Route: t(n.route),
                    PageVersion: "",
                    BrowserSize: t(n.viewport),
                    PageSize: n.browserSize
                }
            }
        }
        ];
        window.Asimov.writeEvent(r)
    }
    function st(n, t, i, r, u) {
        var e = "{'ErrorTitle': '" + r + "', 'RequestURL': '" + t + "', 'LineNumber': '" + i + "', 'ErrorMessage': '" + n.replace(/'/g, '"') + "', 'VisitorId': '" + f + "',}";
        ht(e, u)
    }
    function ht(n, t) {
        var i = [{
            name: "Microsoft.Infrastructure.Events.MSCOMRendering.ClientError",
            content: {
                "Microsoft.Infrastructure.Events.MSCOMRendering.ClientError": {
                    ErrorInfo: n,
                    WasDisplayed: t
                }
            }
        }
        ];
        window.Asimov.writeEvent(i)
    }
    function h(n, t, i, r, u, e, s, h) {
        var c = [{
            name: "Microsoft.Infrastructure.Events.MSCOMRendering.TimeSpan",
            content: {
                "Microsoft.Infrastructure.Events.MSCOMRendering.TimeSpan": {
                    RequestUrl: s,
                    Culture: i,
                    UserAgent: navigator.userAgent,
                    BeginTime: n,
                    PageLoadTime: o(h),
                    PageTimingDetails: u,
                    TotalRequests: o(t),
                    SecondaryResourceDetails: r,
                    AdditionalInformation: e,
                    VisitorId: f
                }
            }
        }
        ];
        window.Asimov.writeEvent(c)
    }
    var n = {}, u = "MC0", f = "", s = "MSCOMBIID", e = "";
    return a(), {
        SendError: st,
        SentPageAction: ot,
        SentPageView: et,
        SendTimeSpan: h
    }
}(), window.onerror = function(n, t, i) {
    return MSCOMRendering.Jll.SendError(n, t, i, "ErrorTitle", !1), !0
}, $(document).ready(function() {
    $(".mscomAd:visible").length > 0 && window.Mscom.Helper.loadScript("http://Ads1.msn.com/library/dap.js", function() {
        $(".mscomAd:visible").each(function(n) {
            if (dapMgr != "undefined") {
                var t = "Ad" + n, i = $(this).attr("data-ad-pageGroup"), r = $(this).attr("data-ad-sizeCode"), u = $(this).attr("data-ad-width"), f = $(this).attr("data-ad-height");
                $(this).attr("id", t);
                try {
                    dapMgr.enableACB(t, !1), dapMgr.renderAd(t, "&PG=" + i + "&AP=" + r, u, f)
                } catch (e) {}
            }
        })
    })
}), window.Mscom === undefined && (window.Mscom = {
    init: function() {
        this.ResponsiveBP3Width = 899, this.ResponsiveBP2Width = 679, this.ResponsiveBP1Width = 539, this.Direction = $("html").css("direction"), this.Left = this.Direction == "ltr" ? "left" : "right", this.Right = this.Direction == "ltr" ? "right" : "left";
        var n = $('meta[name="MscomContentLocale"]').attr("content");
        this.ContentLocale = window.Mscom.Helper.IsValid(n) ? n : navigator.browserLanguage
    }
}), window.Mscom.BrowserDetect = {
    init: function() {
        this.browser = this.searchString(this.dataBrowser) || "Other", this.version = this.searchVersion(navigator.userAgent) || this.searchVersion(navigator.appVersion) || "Unknown"
    },
    searchString: function(n) {
        for (var i, t = 0; t < n.length; t++)
            if (i = n[t].string, this.versionSearchString = n[t].subString, i.indexOf(n[t].subString)!=-1)
                return n[t].identity
    },
    searchVersion: function(n) {
        var t = n.indexOf(this.versionSearchString);
        if (t!=-1)
            return parseFloat(n.substring(t + this.versionSearchString.length + 1))
    },
    dataBrowser: [{
        string: navigator.userAgent,
        subString: "Chrome",
        identity: "Chrome"
    }, {
        string: navigator.userAgent,
        subString: "MSIE",
        identity: "Explorer"
    }, {
        string: navigator.userAgent,
        subString: "Firefox",
        identity: "Firefox"
    }, {
        string: navigator.userAgent,
        subString: "Safari",
        identity: "Safari"
    }, {
        string: navigator.userAgent,
        subString: "Opera",
        identity: "Opera"
    }
    ]
}, window.Mscom.Helper = {
    htmlEncode: function(n) {
        return n == "undefined" || n == null ? n : $("<div/>").text(n).html()
    },
    htmlDecode: function(n) {
        return n == "undefined" || n == null ? n : $("<div/>").html(n).text()
    },
    htmlAttrEncode: function(n) {
        return n = this.htmlEncode(n), n = n.replace(/"/g, "&quot;"), n.replace(/'/g, "&#039;")
    },
    isIE7: function() {
        return navigator.appVersion.indexOf("MSIE 7.")!=-1
    },
    isResponsive: function() {
        return $(document.body).hasClass("mscom-responsive")?!0 : !1
    },
    IsValid: function(n) {
        return n == null || n == "undefined" || n.length == 0?!1 : !0
    },
    setCookie: function(n, t, i, r, u, f) {
        var o = new Date, e;
        i && (i = i * 864e5), e = new Date(o.getTime() + i), document.cookie = n + "=" + escape(t) + (i ? ";expires=" + e.toGMTString() : "") + (r ? ";path=" + r : "") + (u ? ";domain=" + u : "") + (f ? ";secure" : "")
    },
    getCookie: function(n, t) {
        for (var u = document.cookie.split("; "), r, i = 0; i < u.length; i++)
            if (r = u[i].split("="), n === r[0])
                return unescape(r[1]);
        return t
    },
    deleteCookie: function(n) {
        var t = new Date;
        t.setDate(t.getDate() - 1), document.cookie = n + "=;expires=" + t.toGMTString() + ";"
    },
    BiTrack: function(n, t, i, r) {
        if ($.bi)
            try {
                var u = $.bi.getLinkData(n);
                t && (u.title = t), this.IsValid(r) && (u.interactiontype = r), this.IsValid(i) && (u.cot = i), $.bi.record(u)
        } catch (f) {}
    },
    loadScript: function(n, t) {
        var i = document.createElement("script");
        i.src = n, document.body.appendChild(i), i.onload = i.onreadystatechange = i.onerror = function() {
            if ((!i ||!i.readyState ||!/^(?!(?:loaded|complete)$)/.test(i.readyState)) && typeof this.callbackComplete == "undefined") {
                try {
                    t && t()
                } catch (n) {
                    return 
                }
                this.callbackComplete=!0
            }
        }
    },
    GetQueryValue: function(n, t) {
        var r = new RegExp("[\\?&]" + t + "=([^&#]*)", "gi"), i = r.exec(n);
        return i == null ? "" : decodeURIComponent(i[1].replace(/\+/g, " "))
    },
    GetViewport: function() {
        var n = window, t = "inner";
        return "innerWidth"in window || (t = "client", n = document.documentElement || document.body), {
            width: n[t + "Width"],
            height: n[t + "Height"]
        }
    }
}, String.prototype.format = function() {
    var n = arguments;
    return this.replace(/{(\d+)}/g, function(t, i) {
        return n[i]
    })
}, typeof $ != "undefined" && $(function() {
    if (Mscom.init(), $(document.body).removeClass("mscom-nonjs"), Mscom.BrowserDetect.init(), Mscom.BrowserDetect.browser == "Explorer") {
        var n = "";
        Mscom.BrowserDetect.version < 9 && (n = " lt-ie9", Mscom.BrowserDetect.version < 8 && (n += " lt-ie8", Mscom.BrowserDetect.version < 7 && (n += " lt-ie7")), $(document.body).addClass(n))
    }
}), Mscom.Accordion = function(n) {
    this.Control = $("#" + n), this.ItemClass = ".mscom-accordion-item", this.ItemContainerClass = ".mscom-accordion-item-container", this.ItemLinkClass = ".mscom-accordion-item-link", this.AriaExpanded = "aria-expanded", this.AriaSelected = "aria-selected", $($.proxy(this.Init, this))
}, Mscom.Accordion.prototype = {
    Init: function() {
        this.Control.find("> ul > " + this.ItemClass + " > " + this.ItemLinkClass).click($.proxy(this.ItemClick, this))
    },
    ItemClick: function(n) {
        n.preventDefault();
        var t = $(n.currentTarget), i = t.closest(this.ItemClass).children(this.ItemContainerClass);
        i.is(":hidden") ? (i.slideDown(200), Mscom.Helper.BiTrack(n.target, null, 5, 9)) : (i.slideUp(200), Mscom.Helper.BiTrack(n.target, null, 5, 10)), t.closest(this.ItemClass).toggleClass("selected"), t.attr(this.AriaExpanded, !$.parseJSON(t.attr(this.AriaExpanded))), t.attr(this.AriaSelected, !$.parseJSON(t.attr(this.AriaSelected)))
    }
}, MSCom = MSCom || {}, MSCom.CMS = MSCom.CMS || {}, MSCom.CMS.LocalePickerLinkControl = {
    CookieName: "mslocale",
    CookieExpirationDays: 45,
    toggleLocaleFlyout: function(n) {
        $(".mscom-locale-flyout").is(":visible") ? (Mscom.Helper.BiTrack(n, null, 5, 10), $(".pagebody").show(), $(".mscom-locale-flyout").hide()) : (Mscom.Helper.BiTrack(n, null, 5, 9), $(".pagebody").hide(), $(".mscom-locale-flyout").show()), $("html").animate({
            scrollTop: 0
        }, "200")
    },
    SetCookie: function(n) {
        var f = MSCom.CMS.LocalePickerLinkControl.CookieName, h = MSCom.CMS.LocalePickerLinkControl.CookieExpirationDays, u = {}, t = "", e, i, o, r, s;
        document.cookie.length > 0 && document.cookie.indexOf(f + "=")!=-1 && (t = document.cookie.substr(document.cookie.indexOf(f + "=") + f.length + 1), t.indexOf(";") > 0 && (t = t.substring(0, t.indexOf(";"))), t = t.replace(/\|/g, ","), t = t.replace(/'/g, '"'), typeof JSON != "undefined" && (u = JSON.parse(t))), u.u = n, e = new Date, e.setDate(e.getDate() + h), typeof JSON != "undefined" ? i = JSON.stringify(u) : t.length == 0 ? i = '{"u":"' + u.u + '"}' : t.indexOf('"u":')==-1 ? (o = '{"u":"' + u.u + '",', i = t.replace("{", o)) : (o = '"u":"' + u.u + '"', r = t.substr(t.indexOf('"u":')), r = r.indexOf(",")!=-1 ? r.substring(0, r.indexOf(",")) : r.substring(0, r.length - 1), i = t.replace(r, o)), i = i.replace(/"/g, "'"), i = i.replace(/,/g, "|"), s = f + "=" + i + ";expires=" + e.toUTCString() + ";path=/", document.cookie = s
    },
    bindLocaleLinks: function(n) {
        var t = $(".mscom-locale-flyout a");
        t.bind("click", function(t) {
            var i, r, u;
            t.preventDefault(), i = $(this).attr("bi:locale"), n(i), r = "/" + Mscom.ContentLocale + "/", u = location.pathname.contains(r) ? location.pathname.replace(r, "/" + i + "/") : "/" + i + location.pathname, document.location.href = u + window.location.search
        })
    },
    localePickerhandler: function(n) {
        var r, i, t;
        if (typeof this.fragmentloaded == "undefined") {
            r = this, this.fragmentloaded=!1, i = $(".mscom-locale-flyout .mscom-ajax-contentinclude").attr("id"), t = MSCom.CMS.Mashup.ContentIncludes["_" + i] = new MSCom.CMS.Mashup.ContentInclude2($("#" + i)), t.handler = this, t.Control = n.delegateTarget;
            var u = this.bindLocaleLinks, f = this.toggleLocaleFlyout, e = this.SetCookie;
            t.render(function() {
                u(e), f(n.delegateTarget), r.fragmentloaded=!0
            })
        } else if (this.fragmentloaded===!0)
            this.toggleLocaleFlyout(n.delegateTarget);
        else 
            return;
        n.preventDefault()
    }
}, $(function() {
    $(".pagebody").length > 0 && $(".mscom-locale-flyout .mscom-ajax-contentinclude").length > 0 && $(".mscom-footer-localepicker a").bind("click", function(n) {
        MSCom.CMS.LocalePickerLinkControl.localePickerhandler(n)
    })
}), MSCom = MSCom || {}, MSCom.CMS = MSCom.CMS || {}, MSCom.CMS.Mashup = MSCom.CMS.Mashup || {}, MSCom.CMS.Mashup.ContentInclude = function(n, t, i, r, u, f, e, o) {
    e || (e = ""), this._url = n ? n + "/api/controls/contentinclude/" + e : window.location.protocol + "//" + window.location.host + "/api/controls/contentinclude/" + e, this._collection = getQueryValue(window.location.href, "CollectionId"), this._locale = i, this._pageId = t, this._ppaId = r, this._controlAttributeMapping = u, this._siteContextName = f, this._action = e, this._query = o
}, MSCom.CMS.Mashup.ContentInclude.prototype = {
    render: function(n) {
        var t, i;
        this._divToRender = n, t = this._url + "?locale=" + this._locale + "&pageId=" + this._pageId + "&site=" + this._siteContextName, this._collection && (t += "&CollectionId=" + this._collection), this._ppaId && (t += "&ProgrammableContentArea=" + this._ppaId);
        for (i in this._query)
            t += "&" + i + "=" + this._query[i];
        $.ajax({
            type: "POST",
            url: t,
            data: {
                controlAttributeMapping: this._controlAttributeMapping
            },
            xhrFields: {
                withCredentials: !0
            },
            success: function(t) {
                t != null && $(n).html(t)
            }
        })
    }
}, MSCom = MSCom || {}, MSCom.CMS = MSCom.CMS || {}, MSCom.CMS.Mashup = MSCom.CMS.Mashup || {}, MSCom.CMS.Mashup.ContentIncludes = MSCom.CMS.Mashup.ContentIncludes || {}, MSCom.CMS.Mashup.ContentInclude2 = function(n, t, i) {
    i || (i = "html"), this._locale = n.attr("data-urllocale"), this._url = t ? t + "/" + this._locale + "/api/controls/contentinclude/" + i : window.location.protocol + "//" + window.location.host + "/" + this._locale + "/api/controls/contentinclude/" + i, this._collection = this.getQueryValue(window.location.href, "CollectionId"), this._pageId = n.attr("data-defaultPageId"), this._ppaId = n.attr("data-ProgrammableContentArea"), this._host = n.attr("data-Host"), this._hostsegments = n.attr("data-host-segments"), this._hostquery = n.attr("data-host-querystring"), this._controlAttributeMapping = n.attr("data-ControlAttributesMapping"), this._action = i;
    var r = n.attr("data-ajaxQuery");
    r && (this._query = JSON.parse(r)), this._divToRender = "#" + n.attr("id")
}, MSCom.CMS.Mashup.ContentInclude2.prototype = {
    getQueryValue: function(n, t) {
        var r = new RegExp("[\\?&]" + t + "=([^&#]*)", "gi"), i = r.exec(n);
        return i == null ? "" : decodeURIComponent(i[1].replace(/\+/g, " "))
    },
    render: function(n) {
        var r = this._divToRender, t = this._url + "?pageId=" + this._pageId + "&host=" + this._host + "&segments=" + this._hostsegments + "&query=" + this._hostquery, i;
        this._collection && (t += "&CollectionId=" + this._collection), this._ppaId && (t += "&ProgrammableContentArea=" + this._ppaId);
        for (i in this._query)
            t += "&" + i + "=" + this._query[i];
        $.ajax({
            type: "POST",
            url: t,
            data: {
                controlAttributeMapping: this._controlAttributeMapping
            },
            xhrFields: {
                withCredentials: !0
            },
            success: function(t) {
                t != null && ($(r).html(t), n && n())
            }
        })
    }
}, $(document).ready(function() {
    $(".euCookie .mscom-alert-close").click(function(n) {
        n.preventDefault(), $(".euCookie").hide(), Mscom.Helper.setCookie("euCookie", 1, 365 / 2, "/", "microsoft.com"), Mscom.Helper.BiTrack(n.target, null, 5, 10)
    })
}), Mscom.Header = function(n) {
    this.Control = $("#" + n), this.navToggleClass = ".mscom-header-navtogglelink", this.searchToggleClass = ".mscom-header-searchtogglelink", this.searchSectionClass = ".mscom-header-search-section", this.navSectionClass = ".mscom-header-nav-section", $($.proxy(this.Ready, this))
}, Mscom.Header.prototype = {
    Ready: function() {
        this.Control.find(this.searchToggleClass).click($.proxy(this.SearchToggle, this)), this.Control.find(this.navToggleClass).click($.proxy(this.NavToggle, this))
    },
    SearchToggle: function(n) {
        n.preventDefault(), $(this.searchSectionClass).is(":hidden") ? ($(this.searchSectionClass).slideDown(200), Mscom.Helper.BiTrack(n.currentTarget, "SearchToogle", 5, 9)) : ($(this.searchSectionClass).slideUp(200), Mscom.Helper.BiTrack(n.currentTarget, "SearchToogle", 5, 10))
    },
    NavToggle: function(n) {
        n.preventDefault(), $(this.navSectionClass).is(":hidden") ? ($(this.navSectionClass).slideDown(200), Mscom.Helper.BiTrack(n.currentTarget, "NavToogle", 5, 9)) : ($(this.navSectionClass).slideUp(200), Mscom.Helper.BiTrack(n.currentTarget, "NavToogle", 5, 10))
    }
}, typeof $ != "undefined" && $(function() {
    Mscom && Mscom.Account && new Mscom.Account("divAccountControl")
}), Mscom.Account = function(n) {
    this.Control = $("#" + n), this._id = n, this.accountViewOneSection = ".mscom-accountcontrol-viewone", this.accountViewOneLinkClass = ".mscom-account-viewonelink", this._itemClass = ".mscom-accountcontrol-container", this._accountClass = ".mscom-account", this._accountLinkClass = ".mscom-account-link", this.Control.find(this._accountLinkClass).click($.proxy(this.ItemClick, this)).keydown($.proxy(this.ItemKeyDown, this)), $(this.accountViewOneLinkClass).click($.proxy(this.ItemViewOneClick, this)), $(document).click($.proxy(this.DocClick, this))
}, Mscom.Account.prototype = {
    DocClick: function(n) {
        var t = this.Control.find(".selected"), i;
        t.size() > 0 && (i = $.contains(t.get(0), n.target), i || this.HideAccounts(n))
    },
    ItemClick: function(n) {
        n.preventDefault(), this.IsAccountVisible(n) ? this.HideAccounts(n) : this.ShowAccount(n)
    },
    ItemViewOneClick: function(n) {
        n.preventDefault(), $(this.accountViewOneSection).is(":hidden") ? ($(this.accountViewOneSection).slideDown(200), Mscom.Helper.BiTrack(n.target, "AccountViewOne", 5, 9)) : ($(this.accountViewOneSection).slideUp(200), Mscom.Helper.BiTrack(n.target, "AccountViewOne", 5, 10))
    },
    ItemKeyDown: function(n) {
        n.which == 9 && n.shiftKey && this.HideAccounts(n)
    },
    IsAccountVisible: function(n) {
        return $(n.target).parents(this._itemClass, this.Control).find(this._accountClass).css("display") != "none"
    },
    ShowAccount: function(n) {
        $(n.target).parents(this._itemClass, this.Control).addClass("selected"), Mscom.Helper.BiTrack(n.target, "Account", 5, 9)
    },
    HideAccounts: function(n) {
        $(n.target).parents(this._itemClass, this.Control).removeClass("selected"), Mscom.Helper.BiTrack(n.target, "Account", 5, 10)
    }
}, typeof JSON != "object" && (JSON = {}), function() {
    "use strict";
    function i(n) {
        return n < 10 ? "0" + n : n
    }
    function o(n) {
        return e.lastIndex = 0, e.test(n) ? '"' + n.replace(e, function(n) {
            var t = s[n];
            return typeof t == "string" ? t : "\\u" + ("0000" + n.charCodeAt(0).toString(16)).slice( - 4)
        }) + '"' : '"' + n + '"'
    }
    function u(i, f) {
        var s, l, h, a, v = n, c, e = f[i];
        e && typeof e == "object" && typeof e.toJSON == "function" && (e = e.toJSON(i)), typeof t == "function" && (e = t.call(f, i, e));
        switch (typeof e) {
        case"string":
            return o(e);
        case"number":
            return isFinite(e) ? String(e) : "null";
        case"boolean":
        case"null":
            return String(e);
        case"object":
            if (!e)
                return "null";
            if (n += r, c = [], Object.prototype.toString.apply(e) === "[object Array]") {
                for (a = e.length, s = 0; s < a; s += 1)
                    c[s] = u(s, e) || "null";
                return h = c.length === 0 ? "[]" : n ? "[\n" + n + c.join(",\n" + n) + "\n" + v + "]" : "[" + c.join(",") + "]", n = v, h
            }
            if (t && typeof t == "object")
                for (a = t.length, s = 0; s < a; s += 1)
                    typeof t[s] == "string" && (l = t[s], h = u(l, e), h && c.push(o(l) + (n ? ": " : ":") + h));
            else 
                for (l in e)
                    Object.prototype.hasOwnProperty.call(e, l) && (h = u(l, e), h && c.push(o(l) + (n ? ": " : ":") + h));
            return h = c.length === 0 ? "{}" : n ? "{\n" + n + c.join(",\n" + n) + "\n" + v + "}" : "{" + c.join(",") + "}", n = v, h
        }
    }
    typeof Date.prototype.toJSON != "function" && (Date.prototype.toJSON = function() {
        return isFinite(this.valueOf()) ? this.getUTCFullYear() + "-" + i(this.getUTCMonth() + 1) + "-" + i(this.getUTCDate()) + "T" + i(this.getUTCHours()) + ":" + i(this.getUTCMinutes()) + ":" + i(this.getUTCSeconds()) + "Z" : null
    }, String.prototype.toJSON = Number.prototype.toJSON = Boolean.prototype.toJSON = function() {
        return this.valueOf()
    });
    var f = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, e = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, n, r, s = {
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
        '"': '\\"',
        "\\": "\\\\"
    }, t;
    typeof JSON.stringify != "function" && (JSON.stringify = function(i, f, e) {
        var o;
        if (n = "", r = "", typeof e == "number")
            for (o = 0; o < e; o += 1)
                r += " ";
        else 
            typeof e == "string" && (r = e);
        if (t = f, f && typeof f != "function" && (typeof f != "object" || typeof f.length != "number"))
            throw new Error("JSON.stringify");
        return u("", {
            "": i
        })
    }), typeof JSON.parse != "function" && (JSON.parse = function(n, t) {
        function r(n, i) {
            var f, e, u = n[i];
            if (u && typeof u == "object")
                for (f in u)
                    Object.prototype.hasOwnProperty.call(u, f) && (e = r(u, f), e !== undefined ? u[f] = e : delete u[f]);
            return t.call(n, i, u)
        }
        var i;
        if (n = String(n), f.lastIndex = 0, f.test(n) && (n = n.replace(f, function(n) {
            return "\\u" + ("0000" + n.charCodeAt(0).toString(16)).slice( - 4)
        })), /^[\],:{}\s]*$/.test(n.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, "@").replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:\s*\[)+/g, "")))
            return i = eval("(" + n + ")"), typeof t == "function" ? r({
                "": i
            }, "") : i;
        throw new SyntaxError("JSON.parse");
    })
}(), window.Mscom === undefined && (window.Mscom = {}), typeof $ != "undefined" && $(function() {
    Mscom && Mscom.Nav && new Mscom.Nav(".mscom-nav")
}), Mscom.Nav = function(n) {
    this.Control = $(n), this.navFlyoutLinkClass = ".mscom-nav-item-flyout-link", this.navFlyoutClass = ".mscom-nav-flyout", this.navFlyoutItemClass = ".mscom-navitem", $($.proxy(this.Ready, this))
}, Mscom.Nav.prototype = {
    Ready: function() {
        $(document).click($.proxy(this.DocClick, this)), $("a").focus($.proxy(this.DocClick, this)), this.Control.find(this.navFlyoutLinkClass).click($.proxy(this.navItemClick, this))
    },
    DocClick: function(n) {
        var t = this.Control.find(".selected"), i;
        t.size() > 0 && (i = $.contains(t.get(0), n.target), i || this.hideAllFlyout())
    },
    navItemClick: function(n) {
        n.preventDefault(), this.IsFlyoutVisible(n) ? this.hideFlyout(n) : this.showFlyout(n)
    },
    showFlyout: function(n) {
        this.hideAllFlyout(), $(n.target).parents(this.navFlyoutItemClass).addClass("selected"), jQuery.browser.opera || jQuery.browser.msie && document.documentMode <= 7 ? $(n.target).parents(this.navFlyoutItemClass).find(this.navFlyoutClass).show() : $(n.target).parents(this.navFlyoutItemClass).find(this.navFlyoutClass).slideDown(200), Mscom.Helper.BiTrack(n.target, null, 5, 9)
    },
    hideFlyout: function(n) {
        $(n.target).parents(this.navFlyoutItemClass).removeClass("selected"), jQuery.browser.opera || jQuery.browser.msie && document.documentMode <= 7 ? $(n.target).parents(this.navFlyoutItemClass).find(this.navFlyoutClass).hide() : $(n.target).parents(this.navFlyoutItemClass).find(this.navFlyoutClass).slideUp(200), Mscom.Helper.BiTrack(n.target, null, 5, 10)
    },
    hideAllFlyout: function() {
        var n = this.Control.find(".selected"), t;
        n.size() > 0 && (t = $(n.get(0)).find(".mscom-link"), Mscom.Helper.BiTrack(t, null, 5, 10)), this.Control.find(this.navFlyoutItemClass).removeClass("selected"), this.Control.find(this.navFlyoutItemClass).find(this.navFlyoutClass).slideUp(200)
    },
    IsFlyoutVisible: function(n) {
        return !$(n.target).parents(this.navFlyoutItemClass).find(this.navFlyoutClass).is(":hidden")
    }
}, window.Mscom === undefined && (window.Mscom = {}), typeof $ != "undefined" && $(function() {
    if (Mscom && Mscom.Pivot)
        var n = new Mscom.Pivot(".mscom-pivot-nav"), t = new Mscom.PivotTab(".mscom-pivot-tab")
}), Mscom.Pivot = function(n) {
    this.Control = $(n), this.showItemAlways=!1, this.Control.find(".mscom-pivot-container").hasClass("mscom-pivot-showitemalways") && (this.showItemAlways=!0), this.pivotFlyoutLinkClass = ".mscom-pivot-item-flyout-link", this.pivotFlyoutClass = ".mscom-pivot-flyout", this.pivotFlyoutItemClass = ".mscom-pivot-item", this.pivotContainerClass = ".mscom-pivot-container", $($.proxy(this.Ready, this))
}, Mscom.Pivot.prototype = {
    Ready: function() {
        this.Control.find(this.pivotFlyoutLinkClass).click($.proxy(this.pivotItemClick, this)), this.showItemAlways || $(document).click($.proxy(this.DocClick, this))
    },
    DocClick: function(n) {
        var i = this.Control.find(".selected"), t;
        i.size() > 0 && (t = $.contains(i.get(0), n.target), console.log(t), t || this.hideAllFlyout())
    },
    pivotItemClick: function(n) {
        n.preventDefault(), this.IsFlyoutVisible(n)&&!this.showItemAlways ? this.hideFlyout(n) : this.showFlyout(n)
    },
    showFlyout: function(n) {
        this.hideAllFlyout(), $(n.target).parents(this.pivotFlyoutItemClass).addClass("selected"), Mscom.Helper.BiTrack(n.target, null, 5, 9)
    },
    hideFlyout: function(n) {
        $(n.target).parents(this.pivotFlyoutItemClass).removeClass("selected"), Mscom.Helper.BiTrack(n.target, null, 5, 10)
    },
    hideAllFlyout: function() {
        this.Control.find(this.pivotFlyoutItemClass).removeClass("selected")
    },
    IsFlyoutVisible: function(n) {
        return !$(n.target).parents(this.pivotFlyoutItemClass).find(this.pivotFlyoutClass).is(":hidden")
    }
}, Mscom.PivotTab = function(n) {
    this.Control = $(n), this.showItemAlways=!1, this.Control.find(".mscom-pivot-container").hasClass("mscom-pivot-showitemalways") && (this.showItemAlways=!0), this.pivotFlyoutLinkClass = ".mscom-pivot-item-flyout-link", this.pivotFlyoutClass = ".mscom-pivot-flyout", this.pivotFlyoutItemClass = ".mscom-pivot-item", this.pivotContainerClass = ".mscom-pivot-container", $($.proxy(this.Ready, this))
}, Mscom.PivotTab.prototype = {
    Ready: function() {
        this.Control.find(this.pivotFlyoutLinkClass).click($.proxy(this.pivotItemClick, this)), !this.showItemAlways
    },
    DocClick: function(n) {
        var t = this.Control.find(".selected"), i;
        t.size() > 0 && (i = $.contains(t.get(0), n.target), i || this.hideAllFlyout())
    },
    pivotItemClick: function(n) {
        n.preventDefault(), this.IsFlyoutVisible(n)&&!this.showItemAlways ? this.hideFlyout(n) : this.showFlyout(n)
    },
    showFlyout: function(n) {
        Mscom.Helper.BiTrack(n.target, null, 5, 4), this.hideAllFlyout(), $(n.target).addClass("selected"), this.Control.find($(n.target).attr("id").replace("a-", "#")).addClass("selected")
    },
    hideFlyout: function(n) {
        Mscom.Helper.BiTrack(n.target, null, 5, 10), $(n.target).removeClass("selected"), this.Control.find($(n.target).attr("id").replace("a-", "#")).removeClass("selected")
    },
    hideAllFlyout: function() {
        this.Control.find("*").removeClass("selected")
    },
    IsFlyoutVisible: function(n) {
        return $(n.target).hasClass("selected")
    }
}, Mscom.Popup = function(n) {
    this.Control = $(n), this.CloseOnOutSideClick = this.Control.attr("data-closeonoutsideclick"), this.OpenInteractionType = this.Control.attr("data-openinteractiontype"), this.CloseInteractionType = this.Control.attr("data-closeinteractiontype"), Mscom.Helper.IsValid(this.OpenInteractionType) || (this.OpenInteractionType = 9), Mscom.Helper.IsValid(this.CloseInteractionType) || (this.CloseInteractionType = 10), this.Link = $(this.Control.find(".mscom-popup-link:first")), this.CloseButton = $(this.Control.find(".mscom-popup-close:first")), this.PopupContainer = $(this.Control.find(".mscom-popup-container:first")), this.IsContentIncludeLoaded=!1, this.ContentIncludeCount = 0, $($.proxy(this.Init, this))
}, Mscom.Popup.prototype = {
    Init: function() {
        this.Link.click($.proxy(this.OpenPopup, this)), this.CloseButton.click($.proxy(this.ClosePopup, this));
        $(document).on("keydown", $.proxy(this.DocumentKeyDown, this));
        this.CloseOnOutSideClick.toLowerCase() == "true" && ($(this.Control.find(".mscom-popup-mask:first")).click($.proxy(this.ClosePopup, this)), navigator.userAgent.match(/iemobile/i) && $(this.Control.find(".mscom-popup-layout:first")).click(function() {}))
    },
    DocumentKeyDown: function(n) {
        n.keyCode === 27 && this.CloseButton.trigger("click")
    },
    GetContenInclude: function() {
        var t = this;
        this.PopupContainer.find(".mscom-ajax-contentinclude").not(this.PopupContainer.find(".mscom-popup .mscom-ajax-contentinclude")).each(function() {
            var n = $(this), i;
            n.html().length === 0 && (t.ContentIncludeCount === 0 && t.showProgressBar(), t.ContentIncludeCount++, i = n.attr("data-defaultpageid"), MSCom.CMS.Mashup.ContentIncludes["_" + i] = new MSCom.CMS.Mashup.ContentInclude2(n), MSCom.CMS.Mashup.ContentIncludes["_" + i].render($.proxy(t.hideProgressBar, t)))
        })
    },
    showProgressBar: function() {
        this.PopupContainer.find(".mscom-popup-content").append("<div class='mscom-progressbar'><div>")
    },
    hideProgressBar: function() {
        this.ContentIncludeCount--, this.ContentIncludeCount === 0 && this.PopupContainer.find(".mscom-progressbar").remove()
    },
    OpenPopup: function(n) {
        n.preventDefault(), Mscom.Helper.BiTrack(n.target, null, 5, this.OpenInteractionType), $("html").css("overflow", "hidden"), this.IsContentIncludeLoaded || (this.IsContentIncludeLoaded=!0, this.GetContenInclude()), this.PopupContainer.addClass("mscom-show-popup"), this.PopupContainer.trigger("popupOpened")
    },
    ClosePopup: function(n) {
        n.preventDefault(), Mscom.Helper.BiTrack(n.target, "close", 5, this.CloseInteractionType), $("html").css("overflow", ""), this.PopupContainer.removeClass("mscom-show-popup"), this.PopupContainer.trigger("popupClosed")
    }
}, typeof $ != "undefined" && $(function() {
    Mscom && Mscom.Popup && $(".mscom-popup").each(function(n) {
        Mscom.Popup.Popups || (Mscom.Popup.Popups = []), Mscom.Popup.Popups[n] = new Mscom.Popup(this)
    })
}), Mscom.Search = function(n, t, i, r, u, f, e, o, s, h) {
    this.FadeSpeed = 200, this.watermarkTimer = null, this.dropdownTimer = null, this.IsAutoFocus = f, this.Control = $("#" + n), this.Textbox = this.Control.find(".mscom-search-TextBox"), this.Button = this.Control.find(".mscom-search-Button"), this.Control.find("#" + t).hide(), this.Scopes = u, this.CurrentScope = i, this.Control.find(".mscom-search-Source").val(this.CurrentScope), this.suggestionsRequestedText = "", this.suggestionsRequestedScope =- 1, this.CurrentLocale = r, this.WatermarkText = this.Textbox.attr("title"), this.TextboxWidth = this.Textbox.width() + 12, this.processedRequestTime = 0, this.BiTags = o, this.SearchHistoryOptions = {
        EnableSearchHistory: e,
        CookieName: "search-history",
        CookieExpiresDays: 5,
        CookiePath: "/",
        CookieDomain: "microsoft.com",
        MaxCount: 8,
        HideSearchScopes: !0
    }, this.EnableQuickSearch = s, s===!0 && h && $.trim(h.ServiceUrl).length > 0 && h.MaxCount > 0 ? this.QuickSearchFeature = new Mscom.Search.QuickSearchFeature(this, h) : this.EnableQuickSearch=!1, this.CurrentVisibleSearchHistory = [], this.MaxAutoSuggestionCount = 8, this.SearchHistoryContainer = this.Control.find(".mscom-search-history"), this.AutoSuggestionContainer = this.Control.find(".mscom-search-Suggestions"), this.SuggestionBorderStyle = this.AutoSuggestionContainer.css("border-top-width") + " " + this.AutoSuggestionContainer.css("border-top-style") + " " + this.AutoSuggestionContainer.css("border-top-color"), $($.proxy(this.initV3, this))
}, Mscom.Search.prototype = {
    initV3: function() {
        var n = this, t, i;
        this.SearchHistoryOptions.EnableSearchHistory && this.SearchHistoryOptions.HideSearchScopes && this.hideSearchScopes(), this.IsAutoFocus && this.Textbox.get(0).focus(), this.Control.find("#SearchControlForm").submit(function(t) {
            n.SearchClick(t)
        }), this.Control.find(".mscom-search-Button").click(function(t) {
            n.SearchClick(t)
        }), this.Textbox.bind("keydown", function(t) {
            n.textboxKeydown(t)
        }).bind("keyup", function(t) {
            n.textboxKeyup(t)
        }).bind("input", function(t) {
            n.textboxInput(t)
        }).bind("cut paste", function() {
            n.handleCutAndPaste()
        }), window.location.href.indexOf("q=")===-1 ? this.Textbox.val("") : this.lastTypedText = this.Textbox.val();
        this.Control.on("mouseenter", ".mscom-search-Dropdown a", function(t) {
            n.selectDropdownItem($(t.currentTarget), !1)
        });
        this.Control.on("mouseleave", ".mscom-search-Dropdown a", function() {
            n.selectDropdownItem(null, !1)
        });
        t = function() {
            n.dropdownTimer = window.setTimeout(function() {
                n.Control.find(".mscom-search-Dropdown").slideUp(200)
            }, 200)
        };
        this.Control.on("focus", ".mscom-search-TextBox, .mscom-search-Button, .mscom-search-Dropdown a", $.proxy(n.showDropDown, n));
        this.Control.on("blur", ".mscom-search-TextBox, .mscom-search-Button, .mscom-search-Dropdown a", t);
        this.Control.find(".mscom-search-Sources ul a").click(function(t) {
            n.selectSource($(this)), t.preventDefault()
        }), i = window.attachEvent ? "click" : "mousedown";
        this.Control.on(i, ".mscom-search-Suggestions ul a", function() {
            var t = $(this).text();
            n.Textbox.val(t), n.SearchHistoryOptions.EnableSearchHistory && n.saveSearchHistory(t), n.biTrack()
        });
        if (this.SearchHistoryOptions.EnableSearchHistory)
            this.SearchHistoryContainer.on("mousedown", "ul a", function() {
                var t = $(this).text();
                n.Textbox.val(t), n.saveSearchHistory(t), n.biTrack()
            })
    },
    showDropDown: function() {
        this.dropdownTimer != null && (window.clearTimeout(this.dropdownTimer), this.dropdownTimer = null), this.Control.find(".mscom-search-Dropdown").slideDown(200), this.SearchHistoryOptions.EnableSearchHistory && this.Textbox.val().length === 0 && this.showSearchHistory("")
    },
    selectSource: function(n) {
        var t, i;
        window.clearTimeout(this.dropdownTimer), this.dropdownTimer = null, t = this.Control.find(".mscom-search-Sources li").index(n.parent("li")), t !== this.CurrentScope && (this.CurrentScope = t, this.Control.find(".mscom-search-Source").val(t), this.Control.find(".mscom-search-Sources li").removeClass("currentScope"), n.parent("li").addClass("currentScope"), i = $.trim(n.text()), this.Textbox.attr("title", i), this.Button.attr("title", i), this.getSuggestions()), this.Textbox.get(0).focus()
    },
    shiftHighlight: function(n, t) {
        var r, u, s, f, e = ".mscom-search-Suggestions a:visible, .mscom-search-history a:visible", o = ".mscom-search-Suggestions a.selected, .mscom-search-history a.selected", i;
        this.EnableQuickSearch ? (r = this.Control.find(e + ", .quick-search-item:visible"), u = this.Control.find(o + ", .quick-search-item.selected")) : (r = this.Control.find(e), u = this.Control.find(o)), f = u.length ? r.index(u) : t === "down"?-1 : r.length, i = f, t === "down" ? (i++, i >= r.length && (i =- 1)) : t === "up" && (i === 0 ? i =- 1 : i--), i!==-1 ? this.selectDropdownItem($(r[i]), !0) : this.selectDropdownItem(null, !1)
    },
    selectDropdownItem: function(n, t) {
        this.Control.find(".mscom-search-Dropdown a").removeClass("selected"), this.Control.find(".mscom-search-Suggestions a, .mscom-search-history a").removeClass("selected"), this.EnableQuickSearch && this.Control.find(".quick-search-item").removeClass("selected"), n ? (n.addClass("selected"), t && (n.closest(".mscom-search-Suggestions").length || this.SearchHistoryOptions.EnableSearchHistory && n.closest(".mscom-search-history").length ? this.Textbox.val(n.text()) : this.restoreLastTypedText())) : this.SearchHistoryOptions.EnableSearchHistory || this.restoreLastTypedText()
    },
    restoreLastTypedText: function() {
        this.Textbox.val(this.lastTypedText)
    },
    textboxInput: function() {
        this.keyDown==!1 && (this.getSuggestions(), this.EnableQuickSearch && this.Textbox.val().length === 0 && this.QuickSearchFeature.clearSuggestions())
    },
    textboxKeydown: function(n) {
        var t, i, r;
        if (this.keyDown || (this.textOnKeydown = this.Textbox.val()), this.keyDown=!0, n.which === 13) {
            if (t = this.Control.find(".mscom-search-Sources a.selected"), t.length && t.closest("li.currentScope").length === 0) {
                this.selectSource(t), n.preventDefault();
                return 
            }
            if (this.EnableQuickSearch && (i = this.Control.find(".quick-search-item.selected:visible:first"), i.length > 0 && (r = i.attr("url"), r))) {
                window.location.href = r, n.preventDefault();
                return 
            }
        }
    },
    textboxKeyup: function(n) {
        if (this.keyDown=!1, n.which === 40)
            return this.shiftHighlight(n, "down"), !1;
        if (n.which === 38)
            return this.shiftHighlight(n, "up"), !1;
        this.Textbox.val() !== this.textOnKeydown && (this.lastTypedText = this.Textbox.val(), this.IsAutoFocus && this.showDropDown(), this.getSuggestions(), this.SearchHistoryOptions.EnableSearchHistory && this.showSearchHistory(this.Textbox.val()), this.EnableQuickSearch && this.QuickSearchFeature.showSuggestions(this.Textbox.val()))
    },
    handleCutAndPaste: function() {
        var n = this;
        setTimeout(function() {
            n.lastTypedText = n.Textbox.val(), n.getSuggestions()
        }, 0)
    },
    getSuggestions: function() {
        if (this.IsSuggestionsServiceEnabled()===!0 && this.MaxAutoSuggestionCount > 0) {
            var i = this.leftTrim(this.Textbox.val().replace(/\s+/gi, " ")), t, f, u, n, e, r;
            if (i !== this.suggestionsRequestedText || this.CurrentScope !== this.suggestionsRequestedScope) {
                this.suggestionsRequestedText = i, this.suggestionsRequestedScope = this.CurrentScope, u =+ new Date;
                try {
                    t = this.Scopes[this.CurrentScope], t.AutoSuggest && i.length >= t.AutoSuggest.MinChars ? (f = this["processSuggestions_" + t.AutoSuggest.ResultType], n = this, e = this.formatAutoSuggestRequest(i, t), r = {
                        success: function(r) {
                            if (u > n.processedRequestTime) {
                                n.processedRequestTime = u;
                                var s = n.MaxAutoSuggestionCount;
                                n.SearchHistoryOptions.EnableSearchHistory && (s = n.calculateMaxAutoSuggestionCount()), s > 0 && f.call(n, r, i, t, s, $.proxy(n.isDuplicate, n))
                            }
                        }
                    }, t.AutoSuggest.JsonpCallbackName ? (r.jsonp = "cb", r.dataType = "jsonp") : r.dataType = "json", $.ajax(e, r)) : (this.processedRequestTime = u, this.hideSuggestions())
                } catch (o) {}
            }
        }
    },
    processSuggestions_bingJSON: function(n, t, i, r, u) {
        var f, e, o, h=!1, s;
        try {
            if (n.AS.Results)
                for (e = n.AS.Results[0].Suggests, f = 0; f < e.length; ++f)
                    if (e[f].Txt) {
                        h=!0;
                        break
                    }
        } catch (c) {
            return 
        }
        if (h) {
            for (this.clearSuggestions(), s = 0, f = 0; f < e.length; ++f)
                if (o = e[f], o.Txt) {
                    if (u.call(this, o.Txt))
                        continue;
                        if (this.addSuggestion(o.Txt, this.formatSearchRequest(Mscom.Helper.htmlDecode(o.Txt), i)), s++, s + 1 >= r)
                            break
                }
            this.showSuggestions()
        } else 
            this.hideSuggestions()
    },
    processSuggestions_smcUnmarkedArray: function(n, t, i, r, u) {
        var f, e, o;
        if (n && n.length > 1) {
            for (this.clearSuggestions(), o = 0, f = 1; f < n.length; ++f)
                if (!u.call(this, n[f]) && (e = Mscom.Helper.htmlEncode(n[f]), this.addSuggestion(this.highlightQuery(e, t), this.formatSearchRequest(e, i)), o++, o >= r))
                    break;
            this.showSuggestions()
        } else 
            this.hideSuggestions()
    },
    processSuggestions_apiUnmarkedArray: function(n, t, i, r, u) {
        var f, e, o;
        if (n && n.length > 0) {
            for (this.clearSuggestions(), o = 0, f = 0; f < n.length; ++f)
                if (!u.call(this, n[f]) && (e = Mscom.Helper.htmlEncode(n[f]), this.addSuggestion(this.highlightQuery(e, t), this.formatSearchRequest(e, i)), o++, o >= r))
                    break;
            this.showSuggestions()
        } else 
            this.hideSuggestions()
    },
    highlightQuery: function(n, t) {
        var i = n.toLowerCase().indexOf(t.toLowerCase()), r = t.length;
        return i===-1 ? n : i === 0 ? n.substr(0, r) + "<strong>" + n.substring(r) + "<\/strong>" : "<strong>" + n.substr(0, i) + "<\/strong>" + n.substr(i, r) + "<strong>" + n.substring(r + i) + "<\/strong>"
    },
    formatAutoSuggestRequest: function(n, t) {
        return this.formatRequest(Mscom.Helper.htmlDecode(t.AutoSuggest.ServiceUrlFormat), n)
    },
    formatSearchRequest: function(n, t) {
        return this.formatRequest(t.SearchUrlFormat, n)
    },
    formatRequest: function(n, t) {
        return n.replace(/\{0\}/g, encodeURIComponent(t)).replace(/\{1\}/g, encodeURIComponent(this.CurrentLocale))
    },
    clearSuggestions: function() {
        this.Control.find(".mscom-search-Suggestions ul").html("")
    },
    addSuggestion: function(n, t) {
        this.Control.find(".mscom-search-Suggestions ul").append("<li><a href='" + t.replace(/'/g, "&#39;") + "' tabindex='-1' bi:track='false'><span class='mscom-search-Text'>" + n + "<\/span><\/a><\/li>")
    },
    fixDropdownContentOldIE: function(n) {
        var u, t, i, r;
        $.browser.msie && parseInt($.browser.version) <= 7 && (u = this, setTimeout(function() {
            t = u.Control.find(n), r = t.css("display"), t.css("display", ""), t.css("display", r), i = u.Control.find(".mscom-search-Sources"), r = i.css("display"), i.css("display", ""), i.css("display", r)
        }, 1))
    },
    hideSuggestions: function() {
        var n = this;
        this.AutoSuggestionContainer.slideUp(200, function() {
            n.fixDropdownContentOldIE(".mscom-search-Suggestions"), n.AutoSuggestionContainer.find("a.selected").removeClass("selected"), (n.SearchHistoryOptions.EnableSearchHistory || n.EnableQuickSearch) && n.updateBorderStyle()
        })
    },
    showSuggestions: function() {
        var n = this;
        this.AutoSuggestionContainer.slideDown(200, function() {
            n.fixDropdownContentOldIE(".mscom-search-Suggestions"), (n.SearchHistoryOptions.EnableSearchHistory || n.EnableQuickSearch) && (n.restrictSuggestionItemsDisplayCount(), n.updateBorderStyle())
        })
    },
    shiftFocus: function() {
        window.setTimeout($.proxy(function() {
            this.Textbox.get(0).focus()
        }, this), 0)
    },
    IsSuggestionsServiceEnabled: function() {
        var n;
        return n = location.protocol.toString().toLowerCase() == "http:"?!0 : !1
    },
    getSearchHistory: function() {
        var n = Mscom.Helper.getCookie(this.SearchHistoryOptions.CookieName, null);
        return Mscom.Helper.IsValid(n) ? JSON.parse(n) : []
    },
    reorderSearchHistory: function(n) {
        var t = this.getSearchHistory(), i = $.inArray(n, t);
        return i === 0 ? t : (i > 0 ? (t.splice(i, 1), t.splice(0, 0, n)) : (t.splice(0, 0, n), t.length >= this.SearchHistoryOptions.MaxCount && (t.length = this.SearchHistoryOptions.MaxCount)), t)
    },
    filterSearchHistory: function(n) {
        var i = [], r = this.getSearchHistory(), t;
        if (Mscom.Helper.IsValid(n)) {
            for (t = 0; t < r.length; t++)
                if (r[t].indexOf(n) === 0 && (i.push(r[t]), i.length >= this.SearchHistoryOptions.MaxCount))
                    break
        } else 
            i = r;
        return i
    },
    saveSearchHistory: function(n) {
        if (Mscom.Helper.IsValid(n)) {
            n = $.trim(n.replace(/\s+/gi, " "));
            var t = this.reorderSearchHistory(n), i = JSON.stringify(t);
            Mscom.Helper.setCookie(this.SearchHistoryOptions.CookieName, i, this.SearchHistoryOptions.CookieExpiresDays, this.SearchHistoryOptions.CookiePath, this.SearchHistoryOptions.CookieDomain)
        }
    },
    clearSearchHistory: function() {
        this.SearchHistoryContainer.find("ul").html("")
    },
    addSearchHistory: function(n, t) {
        this.SearchHistoryContainer.find("ul").append("<li><a href='" + t.replace(/'/g, "&#39;") + "' tabindex='-1' bi:track='false'><span class='mscom-search-Text'>" + n + "<\/span><\/a><\/li>")
    },
    hideSearchHistory: function() {
        var n = this;
        this.clearSearchHistory(), this.SearchHistoryContainer.slideUp(200, function() {
            n.fixDropdownContentOldIE(".mscom-search-history"), $(this).find("a.selected").removeClass("selected")
        })
    },
    hideSearchScopes: function() {
        var n = this.Control.find(".mscom-search-Sources");
        n.hide(), this.updateBorderStyle()
    },
    updateBorderStyle: function() {
        var i = this.Control.find(".mscom-search-Sources").is(":visible"), t, n;
        this.Control.find(".mscom-search-suggestion-container").css({
            "border-bottom": "none",
            "border-top": "none"
        }), this.Control.find(".mscom-search-suggestion-container:visible:last").css({
            "border-bottom": i ? "none": this.SuggestionBorderStyle,
            "border-top": "none"
        }), this.Control.find(".mscom-search-suggestion-container:visible:first").css("border-top", this.SuggestionBorderStyle), this.EnableQuickSearch && (t=!1, n = this.Control.find(".mscom-search-suggestion-container:visible"), n.length === 1 && n.hasClass("mscom-search-quick-search") && (t=!0), t && n.find(".quick-search-item:last").css("border-bottom", "none"))
    },
    showSearchHistory: function(n) {
        var i;
        n = $.trim(n.replace(/\s+/gi, " "));
        var u = this, f, t, e = this.Scopes[this.CurrentScope], r = this.filterSearchHistory(n);
        if (this.CurrentVisibleSearchHistory.length = 0, r && r.length > 0) {
            for (this.clearSearchHistory(), i = 0; i < r.length; i++)
                if (t = Mscom.Helper.htmlEncode(r[i]), f = n.length > 0 ? this.highlightQuery(t, n) : t, this.addSearchHistory(f, this.formatSearchRequest(t, e)), this.CurrentVisibleSearchHistory.push(t), i + 1 >= this.SearchHistoryOptions.MaxCount)
                    break;
            this.SearchHistoryContainer.slideDown(200, function() {
                u.restrictSuggestionItemsDisplayCount(), u.updateBorderStyle(), u.fixDropdownContentOldIE(".mscom-search-history")
            })
        } else 
            this.hideSearchHistory()
    },
    calculateMaxAutoSuggestionCount: function() {
        var n = 0, t = this.getVisibleSearchHistoryCount();
        return t < this.SearchHistoryOptions.MaxCount && (n = this.SearchHistoryOptions.MaxCount - t), n
    },
    getVisibleSearchHistoryCount: function() {
        return this.SearchHistoryContainer.find("ul li").length
    },
    isDuplicate: function(n) {
        return this.SearchHistoryOptions.EnableSearchHistory ? $.inArray(n, this.CurrentVisibleSearchHistory)!==-1 : !1
    },
    leftTrim: function(n) {
        return n.replace(/^\s+/, "")
    },
    restrictSuggestionItemsDisplayCount: function() {
        var t, n;
        if (this.EnableQuickSearch && (t = this.Control.find(".mscom-search-suggestion-container:visible > ul li").length, t > this.MaxAutoSuggestionCount)) {
            var u = t - this.MaxAutoSuggestionCount, i = this.Control.find(".mscom-search-history:visible > ul li, .mscom-search-Suggestions:visible > ul li"), r = 0;
            for (n = i.length - 1; n >= 0; n--)
                if ($(i[n]).remove(), r++, r >= u)
                    break;
            this.Control.find(".mscom-search-suggestion-container").each(function() {
                var n = $(this);
                n.find("> ul li").length === 0 && n.hide()
            })
        }
    },
    SearchClick: function(n) {
        n.preventDefault();
        var r = this.Scopes[this.CurrentScope], i, t = $.trim(this.Control.find(".mscom-search-TextBox").val());
        t.length ? (this.SearchHistoryOptions.EnableSearchHistory && this.saveSearchHistory(t), i = Mscom.Helper.htmlDecode(r.SearchUrlFormat).replace(/\{0\}/g, encodeURIComponent(t)).replace(/\{1\}/g, encodeURIComponent(this.CurrentLocale)), window.location = i, this.biTrack(n)) : this.Textbox.get(0).focus()
    },
    biTrack: function() {
        var i, r;
        if ($.bi) {
            var f = this.Control.find("form"), u = this.Control.find(".mscom-search-TextBox").val(), t = $.bi.getLinkData(f);
            if (t.interactiontype = 3, t.cot = 4, t.title = "searchbtn", t.searchtype = "MSCOM", t.searchquery = u, t.linkid = "searchmscomclick", t["wcs.cn"] = "searchbtn", t.priorterm = u, t["wcs.ct"] = window.location.href, Mscom.Helper.IsValid(this.BiTags)) {
                i = JSON.parse(this.BiTags);
                for (r in i)
                    t[r] = i[r]
            }
            $.bi.record(t)
        }
    }
}, Mscom.Search.QuickSearchFeature = function(n, t) {
    this.Options = {
        MaxCount: 3,
        MaxDescriptionLength: 100,
        ContainerSelector: ".mscom-search-quick-search"
    }, $.extend(this.Options, t), this.Container = n.Control.find(this.Options.ContainerSelector), this.QuickSearchList = this.Container.find(".quick-search-list"), this.SearchInstance = n, this.LatestRequestUrl = "", this.Options.DeeplinksMapping = [{
        MetaKey: "BuyUrl",
        DisplayText: this.Options.Deeplinks.BuyLinkText
    }, {
        MetaKey: "SupportUrl",
        DisplayText: this.Options.Deeplinks.SupportLinkText
    }, {
        MetaKey: "InstallUrl",
        DisplayText: this.Options.Deeplinks.InstallLinkText
    }, {
        MetaKey: "DownloadUrl",
        DisplayText: this.Options.Deeplinks.DownloadLinkText
    }
    ], $($.proxy(this.init, this))
}, Mscom.Search.QuickSearchFeature.prototype = {
    init: function() {
        this.Container.on("click", ".mscom-search-quick-search .quick-search-item", function(n) {
            var t = $(n.target), i = $(this).attr("url");
            t.is("a") || (window.location.href = i, n.preventDefault())
        });
        $.ajaxPrefilter(function(n, t, i) {
            i.originalUrl = t.url
        })
    },
    showSuggestions: function(n) {
        if (n = this.SearchInstance.leftTrim(n.replace(/\s+/gi, " ")), this.Options.MaxCount <= 0 || n.length === 0 || this.Options.ServiceUrl.length === 0) {
            this.clearSuggestions(), this.hideSuggestions();
            return 
        }
        var r, t, i, u;
        that = this, r =+ new Date, u = this.SearchInstance.Scopes[this.SearchInstance.CurrentScope];
        try {
            t = this.SearchInstance.formatRequest(Mscom.Helper.htmlDecode(this.Options.ServiceUrl), n), this.LatestRequestUrl = t, i = {
                url: t,
                jsonp: "cb",
                dataType: "jsonp",
                success: function(n, t, i) {
                    if (i.originalUrl && i.originalUrl === that.LatestRequestUrl) {
                        var r = $.trim(that.SearchInstance.Textbox.val());
                        that.processResponse(r, n)
                    }
                }
            }, $.ajax(t, i)
        } catch (f) {}
    },
    processResponse: function(n, t) {
        var i, v, c, l, f, s, e, a, r, o, u, h;
        if (this.clearSuggestions(), this.Options.MaxCount > 0 && n.length > 0 && t && t.ResultSets && t.ResultSets.length > 0 && t.ResultSets[0].Results.length > 0) {
            if (e = this, a = this.SearchInstance.Scopes[this.SearchInstance.CurrentScope], n.length < a.AutoSuggest.MinChars)
                return;
            for (r = 0; r < t.ResultSets[0].Results.length; r++) {
                if (i = t.ResultSets[0].Results[r], l = Mscom.Helper.htmlEncode(i.Title), f = this.truncateStringWithEllipsis(Mscom.Helper.htmlEncode(i.Description), this.Options.MaxDescriptionLength), c = i.Url, s = [], this.Options.DeeplinksMapping && this.Options.DeeplinksMapping.length > 0)
                    for (o = 0; o < this.Options.DeeplinksMapping.length; o++)(u = this.Options.DeeplinksMapping[o], $.trim(u.MetaKey).length != 0 && $.trim(u.DisplayText).length != 0) 
                        && (h = this.getMetaValue(i.Metas, u.MetaKey), $.trim(h).length > 0 && s.push({
                            Url: h,
                            DisplayText: u.DisplayText
                        }));
                if (n.length > 0 && (f = f.replace(new RegExp("(" + n + ")", "gi"), "<strong>$1<\/strong>")), this.addSuggestionItem({
                    Title: l,
                    Url: c,
                    ImageUrl: this.getMetaValue(i.Metas, "ImageUrl"),
                    Description: f,
                    DeepLinks: s
                }), r + 1 >= this.Options.MaxCount)
                    break
            }
            this.Container.slideDown(200, function() {
                e.SearchInstance.restrictSuggestionItemsDisplayCount(), e.SearchInstance.updateBorderStyle(), e.SearchInstance.fixDropdownContentOldIE(".mscom-search-quick-search")
            })
        } else 
            this.hideSuggestions()
    },
    getMetaValue: function(n, t) {
        var i, r;
        if (n)
            for (i = 0; i < n.length; i++)
                if (r = n[i], r.Key.toLowerCase() === t.toLowerCase())
                    return Mscom.Helper.htmlEncode(r.Value);
        return ""
    },
    hideSuggestions: function() {
        var n = this;
        this.Container.slideUp(200, function() {
            n.SearchInstance.updateBorderStyle()
        })
    },
    clearSuggestions: function() {
        this.QuickSearchList.html(""), this.SearchInstance.updateBorderStyle()
    },
    addSuggestionItem: function(n) {
        var t = "", i, r, u, f;
        if (n.DeepLinks.length > 0) {
            for (r = 0; r < n.DeepLinks.length; r++)
                i = n.DeepLinks[r], t += '<li><a title="' + i.DisplayText + '" href="' + i.Url + '">' + i.DisplayText + "<\/a><\/li>";
            t = '<ul class="quick-search-links">' + t + "<\/ul>"
        }
        u = "<img " + (n.ImageUrl.length > 0 ? "" : ' class="error"') + ' onerror="$(this).addClass(\'error\');" src="' + n.ImageUrl + '" alt="' + n.Title + '"/>', f = '<li><div class="quick-search-item" url="' + n.Url + '"><div class="quick-search-thumbnail">' + u + '<\/div><div class="quick-search-caption"><h3>' + n.Title + "<\/h3><p>" + n.Description + "<\/p>" + t + "<\/div><\/div><\/li>", this.QuickSearchList.append(f)
    },
    truncateStringWithEllipsis: function(n, t) {
        var i = n;
        return n.length > t && (i = i.substr(0, t) + "..."), i
    }
}, $(window).load(function() {
    var n = window.attachEvent ? "click": "mousedown", t;
    n == "mousedown" && typeof("ontouchstart"in window || window.DocumentTouch && document instanceof DocumentTouch) != "undefined" && (n += " touchstart");
    $(".heroitem .box, .heroitem .text, .heroitem .text-container").on("click", function(n) {
        if (n.target.tagName.toLowerCase() != "a") {
            n.stopPropagation();
            var t = $(this).parents(".heroitem").first().find("a").first();
            t.size() > 0 && (t.trigger("click"), window.location = t.attr("href"))
        }
    });
    if (!window.attachEvent)
        $(".heroitem .box").on(n, function(n) {
            if (n.target.tagName.toLowerCase() != "a") {
                n.stopPropagation();
                var t = $(this).parents(".heroitem").first().find("a").first();
                t.size() > 0 && t.eq(0).trigger(n.type)
            }
        });
    $(".heroitem .media a").on(n, function() {
        if ($.bi&&!this.touchflag) {
            this.touchflag=!0;
            var i = $.bi.getLinkData(this), t = $(this).parents(".heroitem").find(".box .box-title").text();
            t == "" && (t = $(this).find("img").attr("alt")), i.title = t, i["wcs.cn"] = t, $.bi.record(i)
        }
    });
    t = function(n) {
        var t;
        t = n ? $(".slideshow-hero .slides > li:eq(" + n + ") .heroitem") : $(".slideshow-hero .slides > li .heroitem:visible").eq(0), t.hasClass("light-foreground") ? $(".prev-next").addClass("light-foreground") : $(".prev-next").removeClass("light-foreground")
    }, $(".slideshow-hero .slides").responsiveSlideshow({
        speed: 4500,
        controlNav: $(".slideshow-hero .slides").children().length > 1 ? ".slideshow-hero .navigation .grid-container": null,
        directionNav: ".slideshow-hero .prev-next",
        init: function() {
            $(".heroitem.dark-foreground").removeClass("light-foreground"), t()
        },
        between: t
    })
}), window.Modernizr = function(n, t, i) {
    function k(n) {
        nt.cssText = n
    }
    function o(n, t) {
        return typeof n === t
    }
    function ft(n, t) {
        return !!~("" + n).indexOf(t)
    }
    function d(n, t) {
        var u, r;
        for (u in n)
            if (r = n[u], !ft(r, "-") && nt[r] !== i)
                return t == "pfx" ? r : !0;
        return !1
    }
    function et(n, t, r) {
        var f, u;
        for (f in n)
            if (u = t[n[f]], u !== i)
                return r===!1 ? n[f] : o(u, "function") ? u.bind(r || t) : u;
        return !1
    }
    function e(n, t, i) {
        var r = n.charAt(0).toUpperCase() + n.slice(1), u = (n + " " + it.join(r + " ") + r).split(" ");
        return o(t, "string") || o(t, "undefined") ? d(u, t) : (u = (n + " " + rt.join(r + " ") + r).split(" "), et(u, t, i))
    }
    var ot = "2.6.2", r = {}, v=!0, f = t.documentElement, s = "modernizr", g = t.createElement(s), nt = g.style, st, ht = {}.toString, y = " -webkit- -moz- -o- -ms- ".split(" "), tt = "Webkit Moz O ms", it = tt.split(" "), rt = tt.toLowerCase().split(" "), ut = {
        svg: "http://www.w3.org/2000/svg"
    }, u = {}, ct = {}, lt = {}, p = [], w = p.slice, h, c = function(n, i, r, u) {
        var l, a, c, v, e = t.createElement("div"), h = t.body, o = h || t.createElement("body");
        if (parseInt(r, 10))
            while (r--)
                c = t.createElement("div"), c.id = u ? u[r] : s + (r + 1), e.appendChild(c);
        return l = ["&#173;", '<style id="s', s, '">', n, "<\/style>"].join(""), e.id = s, (h ? e : o).innerHTML += l, o.appendChild(e), h || (o.style.background = "", o.style.overflow = "hidden", v = f.style.overflow, f.style.overflow = "hidden", f.appendChild(o)), a = i(e, n), h ? e.parentNode.removeChild(e) : (o.parentNode.removeChild(o), f.style.overflow = v), !!a
    }, b = {}.hasOwnProperty, l, a;
    l=!o(b, "undefined")&&!o(b.call, "undefined") ? function(n, t) {
        return b.call(n, t)
    } : function(n, t) {
        return t in n && o(n.constructor.prototype[t], "undefined")
    }, Function.prototype.bind || (Function.prototype.bind = function(n) {
        var t = this, i, r;
        if (typeof t != "function")
            throw new TypeError;
        return i = w.call(arguments, 1), r = function() {
            var f, e, u;
            return this instanceof r ? (f = function() {}, f.prototype = t.prototype, e = new f, u = t.apply(e, i.concat(w.call(arguments))), Object(u) === u ? u : e) : t.apply(n, i.concat(w.call(arguments)))
        }, r
    }), u.flexbox = function() {
        return e("flexWrap")
    }, u.canvas = function() {
        var n = t.createElement("canvas");
        return !!n.getContext&&!!n.getContext("2d")
    }, u.touch = function() {
        var i;
        return "ontouchstart"in n || n.DocumentTouch && t instanceof DocumentTouch ? i=!0 : c(["@media (", y.join("touch-enabled),("), s, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(n) {
            i = n.offsetTop === 9
        }), i
    }, u.backgroundsize = function() {
        return e("backgroundSize")
    }, u.cssanimations = function() {
        return e("animationName")
    }, u.csstransforms = function() {
        return !!e("transform")
    }, u.csstransforms3d = function() {
        var n=!!e("perspective");
        return n && "webkitPerspective"in f.style && c("@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}", function(t) {
            n = t.offsetLeft === 9 && t.offsetHeight === 3
        }), n
    }, u.csstransitions = function() {
        return e("transition")
    }, u.fontface = function() {
        var n;
        return c('@font-face {font-family:"font";src:url("https://")}', function(i, r) {
            var f = t.getElementById("smodernizr"), u = f.sheet || f.styleSheet, e = u ? u.cssRules && u.cssRules[0] ? u.cssRules[0].cssText: u.cssText || "": "";
            n = /src/i.test(e) && e.indexOf(r.split(" ")[0]) === 0
        }), n
    }, u.video = function() {
        var i = t.createElement("video"), n=!1;
        try {
            (n=!!i.canPlayType) && (n = new Boolean(n), n.ogg = i.canPlayType('video/ogg; codecs="theora"').replace(/^no$/, ""), n.h264 = i.canPlayType('video/mp4; codecs="avc1.42E01E"').replace(/^no$/, ""), n.webm = i.canPlayType('video/webm; codecs="vp8, vorbis"').replace(/^no$/, ""))
        } catch (r) {}
        return n
    }, u.audio = function() {
        var i = t.createElement("audio"), n=!1;
        try {
            (n=!!i.canPlayType) && (n = new Boolean(n), n.ogg = i.canPlayType('audio/ogg; codecs="vorbis"').replace(/^no$/, ""), n.mp3 = i.canPlayType("audio/mpeg;").replace(/^no$/, ""), n.wav = i.canPlayType('audio/wav; codecs="1"').replace(/^no$/, ""), n.m4a = (i.canPlayType("audio/x-m4a;") || i.canPlayType("audio/aac;")).replace(/^no$/, ""))
        } catch (r) {}
        return n
    }, u.svg = function() {
        return !!t.createElementNS&&!!t.createElementNS(ut.svg, "svg").createSVGRect
    }, u.inlinesvg = function() {
        var n = t.createElement("div");
        return n.innerHTML = "<svg/>", (n.firstChild && n.firstChild.namespaceURI) == ut.svg
    };
    for (a in u)
        l(u, a) && (h = a.toLowerCase(), r[h] = u[a](), p.push((r[h] ? "" : "no-") + h));
    return r.addTest = function(n, t) {
        if (typeof n == "object")
            for (var u in n)
                l(n, u) && r.addTest(u, n[u]);
        else {
            if (n = n.toLowerCase(), r[n] !== i)
                return r;
            t = typeof t == "function" ? t() : t, typeof v != "undefined" && v && (f.className += " " + (t ? "" : "no-") + n), r[n] = t
        }
        return r
    }, k(""), g = st = null, r._version = ot, r._prefixes = y, r._domPrefixes = rt, r._cssomPrefixes = it, r.testProp = function(n) {
        return d([n])
    }, r.testAllProps = e, r.testStyles = c, f.className = f.className.replace(/(^|\s)no-js(\s|$)/, "$1$2") + (v ? " js " + p.join(" ") : ""), r
}(this, this.document), function(n, t) {
    function c(n, t) {
        var i = n.createElement("p"), r = n.getElementsByTagName("head")[0] || n.documentElement;
        return i.innerHTML = "x<style>" + t + "<\/style>", r.insertBefore(i.lastChild, r.firstChild)
    }
    function u() {
        var n = i.elements;
        return typeof n == "string" ? n.split(" ") : n
    }
    function f(n) {
        var t = p[n[y]];
        return t || (t = {}, h++, n[y] = h, p[h] = t), t
    }
    function l(n, i, u) {
        if (i || (i = t), r)
            return i.createElement(n);
        u || (u = f(i));
        var e;
        return e = u.cache[n] ? u.cache[n].cloneNode() : it.test(n) ? (u.cache[n] = u.createElem(n)).cloneNode() : u.createElem(n), e.canHaveChildren&&!tt.test(n) ? u.frag.appendChild(e) : e
    }
    function w(n, i) {
        if (n || (n = t), r)
            return n.createDocumentFragment();
        i = i || f(n);
        for (var o = i.frag.cloneNode(), e = 0, s = u(), h = s.length; e < h; e++)
            o.createElement(s[e]);
        return o
    }
    function b(n, t) {
        t.cache || (t.cache = {}, t.createElem = n.createElement, t.createFrag = n.createDocumentFragment, t.frag = t.createFrag()), n.createElement = function(r) {
            return i.shivMethods ? l(r, n, t) : t.createElem(r)
        }, n.createDocumentFragment = Function("h,f", "return function(){var n=f.cloneNode(),c=n.createElement;h.shivMethods&&(" + u().join().replace(/\w+/g, function(n) {
            return t.createElem(n), t.frag.createElement(n), 'c("' + n + '")'
        }) + ");return n}")(i, t.frag)
    }
    function a(n) {
        n || (n = t);
        var u = f(n);
        return i.shivCSS&&!s&&!u.hasCSS && (u.hasCSS=!!c(n, "article,aside,figcaption,figure,footer,header,hgroup,main,nav,section{display:block}mark{background:#FF0;color:#000}")), r || b(n, u), n
    }
    function k(n) {
        for (var t, i = n.getElementsByTagName("*"), r = i.length, e = RegExp("^(?:" + u().join("|") + ")$", "i"), f = []; r--;)
            t = i[r], e.test(t.nodeName) && f.push(t.applyElement(d(t)));
        return f
    }
    function d(n) {
        for (var t, r = n.attributes, u = r.length, i = n.ownerDocument.createElement(e + ":" + n.nodeName); u--;)
            t = r[u], t.specified && i.setAttribute(t.nodeName, t.nodeValue);
        return i.style.cssText = n.style.cssText, i
    }
    function g(n) {
        for (var t, i = n.split("{"), r = i.length, f = RegExp("(^|[\\s,>+~])(" + u().join("|") + ")(?=[[\\s,>+~#.:]|$)", "gi"), o = "$1" + e + "\\:$2"; r--;)
            t = i[r] = i[r].split("}"), t[t.length - 1] = t[t.length - 1].replace(f, o), i[r] = t.join("}");
        return i.join("{")
    }
    function nt(n) {
        for (var t = n.length; t--;)
            n[t].removeNode()
    }
    function v(n) {
        function r() {
            clearTimeout(i._removeSheetTimer), t && t.removeNode(!0), t = null
        }
        var t, u, i = f(n), o = n.namespaces, s = n.parentWindow;
        return !ut || n.printShived ? n : (typeof o[e] == "undefined" && o.add(e), s.attachEvent("onbeforeprint", function() {
            r();
            for (var o, s, f, l = n.styleSheets, e = [], i = l.length, h = Array(i); i--;)
                h[i] = l[i];
            while (f = h.pop())
                if (!f.disabled && rt.test(f.media)) {
                    try {
                        o = f.imports, s = o.length
                    } catch (a) {
                        s = 0
                    }
                    for (i = 0; i < s; i++)
                        h.push(o[i]);
                        try {
                            e.push(f.cssText)
                        } catch (a) {}
            }
            e = g(e.reverse().join("")), u = k(n), t = c(n, e)
        }), s.attachEvent("onafterprint", function() {
            nt(u), clearTimeout(i._removeSheetTimer), i._removeSheetTimer = setTimeout(r, 500)
        }), n.printShived=!0, n)
    }
    var o = n.html5 || {}, tt = /^<|^(?:button|map|select|textarea|object|iframe|option|optgroup)$/i, it = /^<|^(?:a|b|button|code|div|fieldset|form|h1|h2|h3|h4|h5|h6|i|iframe|img|input|label|li|link|ol|option|p|param|q|script|select|span|strong|style|table|tbody|td|textarea|tfoot|th|thead|tr|ul)$/i, s, y = "_html5shiv", h = 0, p = {}, r, i;
    (function() {
        try {
            var n = t.createElement("a");
            n.innerHTML = "<xyz><\/xyz>", s = "hidden"in n, r = n.childNodes.length == 1 || function() {
                t.createElement("a");
                var n = t.createDocumentFragment();
                return typeof n.cloneNode == "undefined" || typeof n.createDocumentFragment == "undefined" || typeof n.createElement == "undefined"
            }()
        } catch (i) {
            s=!0, r=!0
        }
    })(), i = {
        elements: o.elements || "abbr article aside audio bdi canvas data datalist details figcaption figure footer header hgroup main mark meter nav output progress section summary time video",
        shivCSS: o.shivCSS!==!1,
        supportsUnknownElements: r,
        shivMethods: o.shivMethods!==!1,
        type: "default",
        shivDocument: a,
        createElement: l,
        createDocumentFragment: w
    }, n.html5 = i, a(t);
    var rt = /^$|\b(?:all|print)\b/, e = "html5shiv", ut=!r && function() {
        var i = t.documentElement;
        return typeof t.namespaces != "undefined" && typeof t.parentWindow != "undefined" && typeof i.applyElement != "undefined" && typeof i.removeNode != "undefined" && typeof n.attachEvent != "undefined"
    }();
    i.type += " print", i.shivPrint = v, v(t)
}(this, document), function(n, t, i) {
    function h(n) {
        return "[object Function]" == y.call(n)
    }
    function c(n) {
        return "string" == typeof n
    }
    function l() {}
    function w(n) {
        return !n || "loaded" == n || "complete" == n || "uninitialized" == n
    }
    function f() {
        var n = a.shift();
        v = 1, n ? n.t ? o(function() {
            ("c" == n.t ? u.injectCss : u.injectJs)(n.s, 0, n.a, n.x, n.e, 1)
        }, 0) : (n(), f()) : v = 0
    }
    function ut(n, i, s, h, c, l, y) {
        function k(t) {
            if (!nt && w(p.readyState) && (tt.r = nt = 1, !v && f(), p.onload = p.onreadystatechange = null, t)) {
                "img" != n && o(function() {
                    g.removeChild(p)
                }, 50);
                for (var u in r[i])
                    r[i].hasOwnProperty(u) && r[i][u].onload()
            }
        }
        var y = y || u.errorTimeout, p = t.createElement(n), nt = 0, b = 0, tt = {
            t: s,
            s: i,
            e: c,
            a: l,
            x: y
        };
        1 === r[i] && (b = 1, r[i] = []), "object" == n ? p.data = i : (p.src = i, p.type = n), p.width = p.height = "0", p.onerror = p.onload = p.onreadystatechange = function() {
            k.call(this, b)
        }, a.splice(h, 0, tt), "img" != n && (b || 2 === r[i] ? (g.insertBefore(p, d ? null : e), o(k, y)) : r[i].push(p))
    }
    function ft(n, t, i, r, u) {
        return v = 0, t = t || "j", c(n) ? ut("c" == t ? et : nt, n, t, this.i++, i, r, u) : (a.splice(this.i++, 0, n), 1 == a.length && f()), this
    }
    function b() {
        var n = u;
        return n.loader = {
            load: ft,
            i: 0
        }, n
    }
    var s = t.documentElement, o = n.setTimeout, e = t.getElementsByTagName("script")[0], y = {}.toString, a = [], v = 0, k = "MozAppearance"in s.style, d = k&&!!t.createRange().compareNode, g = d ? s: e.parentNode, s = n.opera && "[object Opera]" == y.call(n.opera), s=!!t.attachEvent&&!s, nt = k ? "object" : s ? "script" : "img", et = s ? "script" : nt, tt = Array.isArray || function(n) {
        return "[object Array]" == y.call(n)
    }, p = [], r = {}, it = {
        timeout: function(n, t) {
            return t.length && (n.timeout = t[0]), n
        }
    }, rt, u;
    u = function(n) {
        function a(n) {
            for (var n = n.split("!"), f = p.length, i = n.pop(), e = n.length, i = {
                url: i,
                origUrl: i,
                prefixes: n
            }, u, r, t = 0; t < e; t++)
                r = n[t].split("="), (u = it[r.shift()]) && (i = u(i, r));
            for (t = 0; t < f; t++)
                i = p[t](i);
            return i
        }
        function f(n, t, u, f, e) {
            var o = a(n), s = o.autoCallback;
            o.url.split(".").pop().split("?").shift(), o.bypass || (t && (t = h(t) ? t : t[n] || t[f] || t[n.split("/").pop().split("?")[0]]), o.instead ? o.instead(n, t, u, f, e) : (r[o.url] ? o.noexec=!0 : r[o.url] = 1, u.load(o.url, o.forceCSS ||!o.forceJS && "css" == o.url.split(".").pop().split("?").shift() ? "c" : i, o.noexec, o.attrs, o.timeout), (h(t) || h(s)) && u.load(function() {
                b(), t && t(o.origUrl, e, f), s && s(o.origUrl, e, f), r[o.url] = 2
            })))
        }
        function s(n, t) {
            function a(n, o) {
                if (n) {
                    if (c(n))
                        o || (i = function() {
                            var n = [].slice.call(arguments);
                            s.apply(this, n), u()
                        }), f(n, i, t, 0, e);
                    else if (Object(n) === n)
                        for (r in v = function() {
                            var t = 0, i;
                            for (i in n)
                                n.hasOwnProperty(i) && t++;
                                return t
                            }(), n)
                                n.hasOwnProperty(r) && (!o&&!--v && (h(i) ? i = function() {
                                    var n = [].slice.call(arguments);
                                    s.apply(this, n), u()
                                } : i[r] = function(n) {
                                    return function() {
                                        var t = [].slice.call(arguments);
                                        n && n.apply(this, t), u()
                                    }
                                }(s[r])), f(n[r], i, t, r, e))
                } else 
                    o || u()
            }
            var e=!!n.test, o = n.load || n.both, i = n.callback || l, s = i, u = n.complete || l, v, r;
            a(e ? n.yep : n.nope, !!o), o && a(o)
        }
        var e, t, o = this.yepnope.loader;
        if (c(n))
            f(n, 0, o, 0);
        else if (tt(n))
            for (e = 0; e < n.length; e++)
                t = n[e], c(t) ? f(t, 0, o, 0) : tt(t) ? u(t) : Object(t) === t && s(t, o);
        else 
            Object(n) === n && s(n, o)
    }, u.addPrefix = function(n, t) {
        it[n] = t
    }, u.addFilter = function(n) {
        p.push(n)
    }, u.errorTimeout = 1e4, null == t.readyState && t.addEventListener && (t.readyState = "loading", t.addEventListener("DOMContentLoaded", rt = function() {
        t.removeEventListener("DOMContentLoaded", rt, 0), t.readyState = "complete"
    }, 0)), n.yepnope = b(), n.yepnope.executeStack = f, n.yepnope.injectJs = function(n, i, r, s, h, c) {
        var a = t.createElement("script"), v, y, s = s || u.errorTimeout;
        a.src = n;
        for (y in r)
            a.setAttribute(y, r[y]);
        i = c ? f : i || l, a.onreadystatechange = a.onload = function() {
            !v && w(a.readyState) && (v = 1, i(), a.onload = a.onreadystatechange = null)
        }, o(function() {
            v || (v = 1, i(1))
        }, s), h ? a.onload() : e.parentNode.insertBefore(a, e)
    }, n.yepnope.injectCss = function(n, i, r, u, s, h) {
        var u = t.createElement("link"), c, i = h ? f: i || l;
        u.href = n, u.rel = "stylesheet", u.type = "text/css";
        for (c in r)
            u.setAttribute(c, r[c]);
        s || (e.parentNode.insertBefore(u, e), o(i, 0))
    }
}(this, document), Modernizr.load = function() {
    yepnope.apply(window, [].slice.call(arguments, 0))
}, function(n) {
    var t = [], u = function() {
        return typeof n.screen.systemXDPI != "undefined" && n.screen.systemXDPI >= n.screen.logicalXDPI ? n.screen.systemXDPI / n.screen.logicalXDPI : typeof n.devicePixelRatio != "undefined" ? n.devicePixelRatio : 1
    }, f = function() {
        var i=!1, t = i==!1 || n.screen.width > n.screen.height ? n.screen.width : n.screen.height, r = i==!1 || n.innerWidth > n.innerHeight ? n.innerWidth : n.innerHeight;
        return r > t && (t = r), t = t * u()
    }, e = function() {
        return navigator.userAgent.toLowerCase().indexOf("webkit")>-1 ? document.documentElement.clientWidth : n.innerWidth
    }, i = function(n) {
        return n = n.substring(0, n.length - 2), Number(n)
    }, s = function(n) {
        var o, r, s;
        if (n = $.trim(n), n.length > 0)
            for (n = n.toLowerCase(), n = n.substring(1, n.length - 1), o = n.split(") and ("), r = 0, s = o.length; r < s; r++) {
                var h = o[r].indexOf(":"), c = $.trim(o[r].substring(0, h)), t = $.trim(o[r].substring(h + 1));
                switch (c) {
                case"min-width":
                    if (t = i(t), t > e())
                        return !1;
                        break;
                    case"max-width":
                        if (t = i(t), t < e())
                            return !1;
                            break;
                        case"min-device-pixel-width":
                            if (t = i(t), t > f())
                                return !1;
                                break;
                            case"max-device-pixel-width":
                                if (t = i(t), t < f())
                                    return !1;
                                    break;
                                case"min-device-pixel-ratio":
                                    if (Number(t) > u())
                                        return !1;
                                        break;
                                    default:
                                        return !1
                }
            }
        return !0
    }, o = function(t) {
        var f, e, r, u, l, i, c, o;
        t.setAttribute("data-resolved", "true"), f = $(t).find("div[data-src]").get();
        n: for (u = f.length - 1; u >= 0; u--)
            if (l = s(f[u].getAttribute("data-media")), l==!0) {
                e = f[u].getAttribute("data-src"), r = u;
                break
            }
        if (typeof e != "undefined") {
            if (i = t.getElementsByTagName("img")[0], c=!1, i) {
                if (o = i.getAttribute("data-source-index"), o == r || t.getAttribute("data-disable-swap-above") !== null && o >= r || t.getAttribute("data-disable-swap-below") !== null && o <= r)
                    return 
            } else 
                c=!0, i = n.document.createElement("img"), i.alt = t.getAttribute("data-alt"), t.appendChild(i);
            c ? (i.src = e, i.setAttribute("data-source-index", r)) : h(i, e, r)
        } else 
            !i
    }, h = function(n, i, u) {
        if (typeof t[i] == "object" && t[i].complete)
            r(n, i, u);
        else if (document.images) {
            var f = $("<img />");
            f.bind("load", function() {
                r(n, f.attr("src"), u)
            }).bind("error", function() {
                c(i, f)
            }), t[i] = f, f.attr("src", i)
        } else 
            r(n, i, u)
    }, r = function(n, t, i) {
        n.src = t, n.setAttribute("data-source-index", i)
    }, c = function(n, i) {
        i.unbind("error").unbind("load"), i.get(0).src = "", t[n] = undefined
    };
    n.picturePolyfill = {}, n.picturePolyfill.resolveLast = function() {
        var n = $("div[data-picture]:not([data-resolved],[data-defer])").last().get()[0];
        o(n)
    }, n.picturePolyfill.init = function(n) {
        var i, t, r;
        for (typeof n != "boolean" && (n=!0), i = $("div[data-picture]").get(), t = 0, r = i.length; t < r; t++)(i[t].getAttribute("data-resolved") === null || i[t].getAttribute("data-disable-swap") === null) 
            && (n===!0 || i[t].getAttribute("data-defer") === null) && o(i[t])
        }, n.addEventListener ? (n.addEventListener("resize", n.picturePolyfill.init, !1), n.addEventListener("DOMContentLoaded", function() {
        n.picturePolyfill.init(!1)
    }, !1), n.addEventListener("load", n.picturePolyfill.init, !1)) : n.attachEvent && n.attachEvent("onload", n.picturePolyfill.init)
}(window), typeof window.MSCOM == "undefined" && (window.MSCOM = {}), window.MSCOM.Helper = {
    htmlEncode: function(n) {
        return n == "undefined" || n == null ? n : $("<div/>").text(n).html()
    },
    htmlDecode: function(n) {
        return n == "undefined" || n == null ? n : $("<div/>").html(n).text()
    },
    htmlAttrEncode: function(n) {
        return n = this.htmlEncode(n), n = n.replace(/"/g, "&quot;"), n.replace(/'/g, "&#039;")
    },
    isIE7: function() {
        return navigator.appVersion.indexOf("MSIE 7.")!=-1
    }
}, window.MSCOM.Helper.Content = {
    images: {},
    doc: {},
    queue: {},
    curThread: 0,
    maxThread: 2,
    nextDelay: 100,
    logenabled: !1,
    isBodyLoaded: !1,
    prefetchEnabled: !0,
    onload: function() {
        this.log("window.MSCOM.Helper.Content: Body Onload"), this.isBodyLoaded=!0, setTimeout($.proxy(this.next, this), this.nextDelay)
    },
    log: function(n) {
        if (this.logenabled)
            try {
                console.log(n)
        } catch (t) {} else 
            window.location.hash.indexOf("helplog") > 0 && (this.logenabled=!0)
    },
    sortQ: function() {
        var n = [], i = {}, t;
        for (t in this.queue)
            n.push(t);
        n.sort();
        for (t in n)
            i[n[t]] = this.queue[n[t]];
        this.queue = i
    },
    registerImage: function(n) {
        if (this.prefetchEnabled) {
            var t = {
                path: "",
                priority: 0,
                type: "img",
                callback: null,
                passvar: null,
                cache: !0
            };
            $.extend(t, n), this.addToQueue(t)
        }
        return this.prefetchEnabled
    },
    registerDoc: function(n) {
        if (this.prefetchEnabled) {
            var t = {
                path: "",
                priority: 0,
                type: "doc",
                callback: null,
                passvar: null,
                cache: !0
            };
            $.extend(t, n), this.addToQueue(t)
        }
        return this.prefetchEnabled
    },
    unregister: function(n) {
        var r, i, t;
        for (r in this.queue)
            for (i = this.queue[r], t = 0; t < i.length; ++t)
                if (i[t].path === n)
                    return this.log(["unregister", i[t]]), i.splice(t, 1), !0;
        return !1
    },
    addToQueue: function(n) {
        this.log(["addToQueue", n]), this.queue[n.priority] == undefined && (this.queue[n.priority] = [], this.sortQ()), this.queue[n.priority].push(n), setTimeout($.proxy(this.next, this), this.nextDelay)
    },
    next: function() {
        var n, t;
        if (!this.prefetchEnabled || this.curThread >= this.maxThread ||!$.isReady&&!this.isBodyLoaded)
            return !1;
        for (t in this.queue) {
            if (!this.isBodyLoaded && parseInt(t)>-1)
                return !1;
            if (this.queue[t].length > 0) {
                n = this.queue[t].shift();
                break
            }
        }
        n && (this.log(["Next:starting item", n]), this.curThread++, n.type == "img" && this.loadImage(n), n.type == "doc" && this.loadDocument(n))
    },
    loadImage: function(n) {
        if (n.cache && this.images[n.path] != undefined && this.images[n.path].complete)
            this.imageLoaded(this.images[n.path], n);
        else if (document.images) {
            var t = new Image, i = this;
            $(t).load(function() {
                i.imageLoaded(t, n)
            }), $(t).error(function() {
                i.imageError(t, n)
            }), t.src = n.path, this.images[n.path] = t
        }
    },
    loadDocument: function(n) {
        if (n.cache && this.doc[n.path] != undefined)
            this.docLoaded(this.doc[n.path], n);
        else {
            var t = this;
            $.get(n.path, function(i) {
                t.docLoaded(i, n)
            })
        }
    },
    imageLoaded: function(n, t) {
        this.curThread = this.curThread > 0 ? this.curThread - 1 : 0, t.callback && t.callback(n, t.passvar), setTimeout($.proxy(this.next, this), this.nextDelay)
    },
    imageError: function(n, t) {
        this.curThread = this.curThread > 0 ? this.curThread - 1 : 0, t.callback && t.callback(n, t.passvar), setTimeout($.proxy(this.next, this), this.nextDelay)
    },
    docLoaded: function(n, t) {
        this.curThread = this.curThread > 0 ? this.curThread - 1 : 0;
        try {
            t.callback && t.callback(n, t.passvar)
        } catch (i) {
            this.log(i)
        }
        setTimeout($.proxy(this.next, this), this.nextDelay)
    }
}, typeof $ != "undefined" ? $(window).load($.proxy(window.MSCOM.Helper.Content.onload, window.MSCOM.Helper.Content)) : document.addEventListener ? window.addEventListener("load", function() {
    window.MSCOM.Helper.Content.onload()
}, !1) : window.attachEvent && window.attachEvent("onload", function() {
    window.MSCOM.Helper.Content.onload()
}), $(function() {
    var n = $(".top-bar-storelink").children().clone(!0);
    $(".menulevel-storelink").prepend(n)
}), $(function() {
    if ($(".store-geo").length)
        if (typeof navigator.geolocation != "undefined" && $(".store-geo[data-clientgeo=true]").length) {
            var n = setTimeout(function() {
                $(".store-editorial").fadeIn(1e3)
            }, 1e4);
            navigator.geolocation.getCurrentPosition(function(t) {
                clearTimeout(n), getStore(t)
            })
        } else 
            getStore()
}), function(n, t, i, r) {
    n.responsiveSlideshow = function(u, f) {
        var c = n(u), o = c.children(), h = 0, v = {}, e, s = n.extend({}, n.responsiveSlideshow.defaults, f), a, l = [];
        n.data(u, "responsiveSlideshow", c), v = {
            init: function() {
                var f, t, l, u, a;
                e = this, s.init(), i.documentElement.dir == "rtl" ? o.css({
                    "margin-left": "-100%",
                    float: "right",
                    width: "100%"
                }) : o.css({
                    "margin-right": "-100%",
                    float: "left",
                    width: "100%"
                }), f = o.eq(h), o.not(f).hide(), e.goToSlide(h), f.data("shown", !0);
                c.on("mouseenter", function() {
                    e.handlePause("hover", "enter")
                }).on("mouseleave", function() {
                    e.handlePause("hover", "leave")
                });
                c.find("a").on("focus", function() {
                    e.handlePause("focus", "enter")
                }).on("blur", function() {
                    e.handlePause("focus", "leave")
                });
                if (s.controlNav) {
                    t = n(s.controlNav), l = function() {
                        o.each(function(i) {
                            var e = n(this).attr("selectBi");
                            e === r && (e = "");
                            var f = '<a href="{{target_id}}" title="{{title}}" bi:index="{{idx}}"' + e + '><span aria-hidden="true" class="bi-hidetext">{{icon}}<\/span> <span class="screen-reader-text">{{idx}}<\/span><\/a>', o = /{{\w*}}/gi, s = f.match(o), u = [];
                            u.idx = i, u.target_id = n(this).attr("id"), u.title = t.data("title-text") ? t.data("title-text").replace("{{idx}}", i + 1) : "", u.icon = t.data("icon-text") ? t.data("icon-text") : "&#x25CF;", n(s).each(function(n, t) {
                                f = f.replace(t, u[t.replace("{{", "").replace("}}", "")])
                            }), t.append(f)
                        })
                    }, t.children().length === 0 && l(), t.find("a").eq(0).addClass("active");
                    t.find("a").on("click", function(i) {
                        var s, u, c, f;
                        if ((i.preventDefault(), s = t.find("a").index(n(this)), h !== s) && (e.goToSlide(s), n.bi)) {
                            u = n.bi.getLinkData(this), c = n.bi.getAttrData(o.eq(s).find("a:visible"));
                            for (f in c)
                                u[f] === r ? u[f] = c[f] : u[f] += ";" + c[f];
                            u.typestructure += o.find("*[bi\\:type]").length > 0 ? ";" + o.find("*[bi\\:type]").attr("bi:type") : ";slideshowitem", u.interactiontype = 6, u.cot = 5, n.bi.record(u)
                        }
                    })
                }
                if (s.directionNav) {
                    u = n(s.directionNav), a = function() {
                        var t = n('<button class="prev">Previous<\/button><button class="next">Next<\/button>');
                        u.append(t)
                    }, u.children().length === 0 && a();
                    n(u).find(".prev").on("click", function(t) {
                        var s, i, f, u;
                        if (t.preventDefault(), s = e.getPrevSlideIndex(), e.prevSlide(), n.bi) {
                            i = n.bi.getLinkData(this), f = n.bi.getAttrData(o.eq(s).find("a:visible"));
                            for (u in f)
                                i[u] === r ? i[u] = f[u] : i[u] += ";" + f[u];
                            i.typestructure += o.find("*[bi\\:type]").length > 0 ? ";" + o.find("*[bi\\:type]").attr("bi:type") : ";slideshowitem", i.interactiontype = 5, i.cot = 5, i.index = s, n.bi.record(i)
                        }
                    });
                    n(u).find(".next").on("click", function(t) {
                        var s, i, f, u;
                        if (t.preventDefault(), s = e.getNextSlideIndex(), e.nextSlide(), n.bi) {
                            i = n.bi.getLinkData(this), f = n.bi.getAttrData(o.eq(s).find("a:visible"));
                            for (u in f)
                                i[u] === r ? i[u] = f[u] : i[u] += ";" + f[u];
                            i.typestructure += o.find("*[bi\\:type]").length > 0 ? ";" + o.find("*[bi\\:type]").attr("bi:type") : ";slideshowitem", i.interactiontype = 4, i.cot = 5, i.index = s, n.bi.record(i)
                        }
                    })
                }
                e.goToSlide(h)
            },
            handlePause: function(n, t) {
                if (l[n] = t === "enter", l[n])
                    e.clearTimer();
                else {
                    for (var i in l)
                        if (l[i])
                            return;
                    e.setTimer()
                }
            },
            setTimer: function() {
                s.slideshow && (a = t.setTimeout(e.autoNextSlide, s.speed))
            },
            clearTimer: function() {
                s.slideshow&&!!a && t.clearTimeout(a)
            },
            getNextSlideIndex: function() {
                var n = h + 1;
                return n === o.length ? 0 : n
            },
            getPrevSlideIndex: function() {
                var n = h - 1;
                return n < 0 ? o.length - 1 : n
            },
            autoNextSlide: function() {
                var f = e.getNextSlideIndex(), i = o.eq(f), s=!0, u, t;
                i.find("img").each(function(n, t) {
                    if (t.complete==!1)
                        return s=!1, !1
                }), s==!0 && i.find("div[data-picture]:not([data-resolved])").size() == 0 ? (e.nextSlide(), i.data("shown")!=!0 && (i.data("shown", !0), n.bi && (u = "slideshow;", u += o.find("*[bi\\:type]").length > 0 ? o.find("*[bi\\:type]").attr("bi:type") : "slideshowitem", t = n.bi.getAttrData(i.find("a:visible")), t.typestructure = u, t.index = f, t.interactiontype = 1, t.cot = 5, t.uridomain = r, t.uripath = r, t.uriquery = r, n.bi.record(t)))) : (e.clearTimer(), e.setTimer())
            },
            nextSlide: function() {
                e.goToSlide(e.getNextSlideIndex())
            },
            prevSlide: function() {
                e.goToSlide(e.getPrevSlideIndex())
            },
            goToSlide: function(i) {
                (s.before(), e.clearTimer(), e.setTimer(), h !== i) && (s.controlNav && n(s.controlNav).find("a").removeClass("active").eq(i).addClass("active"), o.eq(h).is(":visible") ? o.eq(h).fadeOut(s.animationDuration) : o.eq(h).hide(), h = i, o.eq(i).fadeIn(s.animationDuration, function() {
                    s.after()
                }), t.setTimeout(function() {
                    s.between(i)
                }, s.animationDuration / 2))
            }
        }, v.init()
    }, n.responsiveSlideshow.defaults = {
        slideshow: !0,
        speed: 5e3,
        directionNav: null,
        controlNav: null,
        animationDuration: 400,
        init: function() {},
        before: function() {},
        after: function() {},
        between: function() {}
    }, n.fn.responsiveSlideshow = function(t) {
        return n(this).each(function() {
            new n.responsiveSlideshow(n(this), t)
        })
    }
}(jQuery, window, document), Mscom.Video = function(n) {
    this.Control = $(n), this.VideoContainer = this.Control.find(".mscom-video-container"), this.Control.attr("data-loaded", "true"), this.VideoContainerID = this.VideoContainer.attr("id"), this.VideoID = this.VideoContainer.attr("data-videoid"), this.BiTags = this.VideoContainer.attr("data-bitag"), this.PlayerName = this.VideoContainer.attr("data-playername"), this.PlayerRefName = this.VideoContainer.attr("data-playerrefname"), this.AutoPlay = this.VideoContainer.attr("data-autoplay"), this.CSID = "ux-cms-en-us-msoffice", this.PlayerOverrides = {}, this.IsVideoLoaded=!1, this.IsVideoRendered=!1, this.VideoTitle, this.IsPopupVideo=!1, this.PopupContent, Mscom.Helper.IsValid(this.PlayerName) || (this.PlayerName = "ShowcaseMSN1"), Mscom.Helper.IsValid(this.PlayerRefName) || (this.PlayerRefName = "MPL_CMG_ShowcaseMSN1"), Mscom.Helper.IsValid(this.AutoPlay) || (this.AutoPlay = "False"), this.VideoWidgetId = this.VideoContainerID + "_ux1_1_1_1", $($.proxy(this.Init, this))
}, Mscom.Video.prototype = {
    Init: function() {
        var t = {
            videoFilter: {
                type: "Uuid",
                uuids: [this.VideoID]
            }
        }, n;
        if (this.PlayerOverrides[this.PlayerRefName + ".Width"] = "100%", this.PlayerOverrides[this.PlayerRefName + ".AutoPlayVideo"] = this.AutoPlay, this.PlayerOverrides[this.PlayerRefName + ".DefaultVideo"] = {}, this.PlayerOverrides[this.PlayerRefName + ".DefaultVideo"].videoQuery = t, n = this.Control.closest(".mscom-popup-container"), n.length > 0) {
            this.IsPopupVideo=!0, this.PopupContent = n.find(".mscom-popup-content").first();
            n.on("popupOpened", $.proxy(this.VideoPopupOpen, this));
            n.on("popupClosed", $.proxy(this.TriggerVideoEvent, this, "pauseVideo"));
            $(window).on("resize", $.proxy(this.ResizeVideo, this));
            this.VideoContainer.on("DOMSubtreeModified", $.proxy(this.VideoPopupLoaded, this))
        } else 
            this.Render("init")
    },
    VideoPopupOpen: function() {
        this.PauseAllVideo(), this.Render("popupOpened")
    },
    VideoPopupLoaded: function() {
        this.ResizeVideo(), this.VideoContainer.off("DOMSubtreeModified", this.VideoPopupLoaded)
    },
    Render: function() {
        this.IsVideoRendered ? this.AutoPlay === "True" && this.TriggerVideoEvent("playVideo") : (MsnVideoUx.render(this.PlayerName, this.VideoContainerID, this.PlayerOverrides, {
            csid: this.CSID
        }), MsnVideo2.addMessageReceiver({
            eventType: "playbackStatusChanged",
            widgetId: this.VideoWidgetId,
            funcCb: $.proxy(this.biTrack, this)
        }), this.IsVideoRendered=!0)
    },
    TriggerVideoEvent: function(n) {
        this.IsVideoRendered && MsnVideo2.sendMessage({
            type: n,
            targetId: this.VideoWidgetId
        })
    },
    PauseAllVideo: function() {
        if (Mscom.Helper.IsValid(Mscom.Video.Videos)) {
            var n = this;
            $.each(Mscom.Video.Videos, function() {
                this.VideoWidgetId !== n.VideoWidgetId && this.TriggerVideoEvent("pauseVideo")
            })
        }
    },
    ResizeVideo: function() {
        if (!(Mscom.BrowserDetect.browser == "Explorer" && Mscom.BrowserDetect.version < 9)) {
            var i = Mscom.Helper.GetViewport(), r = i.width - this.PopupContent.css("border-left-width").replace("px", "") * 2, u = i.height - 60, f = u / r, t = "", n = "";
            f < .5625 && (t = u * 1.77, n = t * .5625, t = "width:" + r + "px !important;", n = "padding-bottom:" + n + "px !important;"), this.Control.find(".vxp_content, .vxpMultiLitePlayer").attr("style", t), this.Control.find(".video_player_outer").attr("style", n), this.Control.attr("style", n), this.Control.find(".vxp_richEmbedContainer").attr("style", t + n)
        }
    },
    biTrack: function(n) {
        var u, f, t, i, r;
        if (n.sourceId === this.VideoWidgetId && (n.param.status == "loaded" && (this.IsPopupVideo && this.ResizeVideo(), this.IsVideoLoaded=!0, Mscom.BrowserDetect.browser == "Explorer" && Mscom.BrowserDetect.version <= 7 && (u = this.Control.find("div .vxp_richEmbedContainer object"), u.css("zoom", "1")), f = MsnVideo2.getProperties({
            type: "currentVideo",
            targetId: this.VideoWidgetId
        }), this.VideoTitle = f[0].param.video.title), (n.param.status == "videoPaused" || n.param.status == "videoPlayCompleted") && (this.IsVideoLoaded=!0), this.IsVideoLoaded==!0 && n.param.status === "videoPlaying" && (this.PauseAllVideo(), this.IsVideoLoaded=!1, $.bi))) {
            if (t = $.bi.getLinkData(this.Control), t["wcs.cn"] = this.VideoTitle, t.title = this.VideoTitle, Mscom.Helper.IsValid(this.BiTags)) {
                i = JSON.parse(this.BiTags);
                for (r in i)
                    t[r] = i[r]
            }
            $.bi.record(t)
        }
    }
}, typeof $ != "undefined" && $(function() {
    if (Mscom && Mscom.Video && $(".mscom-video").length > 0) {
        var n = "";
        n = window.location.protocol == "https:" ? "https://imgs1-video.ssl.catalog.video.msn.com/s/js/vxp.js" : "http://img1.video.s-msn.com/s/js/vxp.js", window.Mscom.Helper.loadScript(n, function() {
            $(".mscom-video ").not("[data-loaded=true]").each(function(n) {
                Mscom.Video.Videos || (Mscom.Video.Videos = []), Mscom.Video.Videos[n] = new Mscom.Video(this)
            })
        })
    }
}), Mscom.VideoBumper = function() {
    this.Control = $(".hero-video"), this.Video = document.getElementById("videoBumperVideo"), this.Slides = $(this.Control.find(".slides")).children(), this.FadeTransitionEnabled = this.Control.hasClass("video-fade-transition"), this.StageOverlayEnabled = this.Control.hasClass("stage-overlay"), this.IsMuteVideo = $(this.Video).attr("data-mute"), this.VideoTimeOut = $(this.Video).attr("data-timeout"), this.IsOverlayHidden=!1, this.OverlayContainer = null, this.Overlay = null, this.OverlayContent = null, $($.proxy(this.Init, this))
}, Mscom.VideoBumper.prototype = {
    Init: function() {
        try {
            this.Video.muted = Boolean(this.IsMuteVideo)
        } catch (n) {}
        this.FadeTransitionEnabled && (this.Control.find(".slides").css("position", "relative"), this.Slides.eq(2).find(".box").css("display", "none")), this.BindVideoEvents(), this.StageOverlayEnabled && this.BuildStageOverlay()
    },
    BuildStageOverlay: function() {
        this.OverlayContainer = $('<div class="overlay-container"><\/div>').css({
            width: "100%",
            position: "fixed",
            "background-color": "#fff",
            opacity: "0.70",
            filter: "alpha(opacity=70)",
            top: "0",
            left: "0",
            "z-index": "1001",
            overflow: "hidden"
        }), this.Control.css({
            "z-index": "1002"
        }), this.Control.after(this.OverlayContainer), this.SetOverlayHeight(), this.BindStageOverlayEvents()
    },
    BindStageOverlayEvents: function() {
        $(window).on("resize", $.proxy(this.SetOverlayHeight, this));
        $(document).on("DOMContentLoaded", $.proxy(this.SetOverlayHeight, this));
        $(document).on("click", $.proxy(this.HideOverlay, this));
        $(document).on("scroll", $.proxy(this.HideOverlay, this))
    },
    SetOverlayHeight: function() {
        var n = $(document).height();
        this.OverlayContainer.css({
            height: n
        })
    },
    HideOverlay: function() {
        this.IsOverlayHidden==!1 && (this.IsOverlayHidden=!0, this.Control.css({
            "z-index": "auto"
        }), this.OverlayContainer.fadeOut(600), $(window).off("resize", this.SetOverlayHeight), $(document).off("DOMContentLoaded", this.SetOverlayHeight), $(document).off("click", this.HideOverlay), $(document).off("scroll", this.HideOverlay, this))
    },
    BindVideoEvents: function() {
        var n = this;
        $(this.Video).bind("canplaythrough", function() {
            n.Slides.eq(2).is(":visible") || n.ShowVideoSlide()
        }), $(this.Video).bind("stalled", function() {
            n.StageOverlayEnabled && n.DelayOverlayHide(), n.Slides.eq(2).has("img") && n.GotoSlide(2)
        }), $(this.Video).bind("ended", function() {
            n.StageOverlayEnabled && n.DelayOverlayHide(), n.Control.removeClass("playing"), n.Slides.eq(2).has("img") && n.GotoSlide(2, 400)
        }), window.setTimeout(function() {
            n.DestroyVideoSlide()
        }, n.VideoTimeOut)
    },
    DelayOverlayHide: function() {
        var n = this;
        window.setTimeout(function() {
            n.HideOverlay()
        }, 800)
    },
    ShowVideoSlide: function() {
        if (this.Slides.eq(0).is(":visible"))
            if (this.Control.addClass("playing"), this.FadeTransitionEnabled) {
                $(this.Video).trigger("pause");
                var n = this;
                this.Slides.eq(1).css({
                    position: "absolute",
                    top: "0"
                }).fadeIn(500, function() {
                    n.Slides.eq(0).hide(), n.Slides.eq(1).css("position", "static"), $(n.Video).trigger("play")
                })
            } else 
                this.GotoSlide(1), $(this.Video).trigger("play")
    },
    GotoSlide: function(n, t) {
        var i, r;
        this.Slides.eq(n).is(":visible") || (this.FadeTransitionEnabled ? (i = this.Slides.filter(":visible"), typeof t != "number" && (t = 400), Modernizr !== undefined && Modernizr.cssanimations ? this.Slides.eq(2).find(".box").show() : this.Slides.eq(2).find(".box").fadeIn(400, "linear"), r = this, this.Slides.eq(n).css({
            position: "absolute",
            top: "0"
        }).fadeIn(t, function() {
            i.hide(), r.Slides.eq(n).css("position", "static")
        })) : this.Slides.hide().eq(n).show())
    },
    DestroyVideoSlide: function() {
        var n = this.Video.currentTime > 0;
        n || (this.GotoSlide(2), this.StageOverlayEnabled && this.DelayOverlayHide())
    }
}, Mscom.Slider = function(n, t) {
    this.Control = $("#" + n), this.SlideTransitionSpeed = parseInt(t), this.SliderContainer = $(this.Control.find(".mscom-slider-container")), this.Prev = $(this.Control.find(".mscom-prev")), this.Next = $(this.Control.find(".mscom-next")), this.SliderItems = $(this.Control.find(".mscom-slider-item")), this.SliderItemCount = this.SliderItems.length, this.DataView4 = parseInt(this.Control.attr("data-view4")), this.DataView3 = parseInt(this.Control.attr("data-view3")), this.DataView2 = parseInt(this.Control.attr("data-view2")), this.DataView1 = parseInt(this.Control.attr("data-view1")), this.SlideTransitionSpeedPerSlide = parseInt(this.SlideTransitionSpeed / this.DataView4), this.DataStyle4 = this.Control.attr("data-style4").toLowerCase(), this.DataStyle3 = this.Control.attr("data-style3").toLowerCase(), this.DataStyle2 = this.Control.attr("data-style2").toLowerCase(), this.DataStyle1 = this.Control.attr("data-style1").toLowerCase(), this.IsTouchEnabled=!1, this.CurrentItemIndex = 0, this.CurrentView = 4, this.CurrentViewItem = this.DataView4, this.CurrentViewStyle = this.DataStyle4, $($.proxy(this.Init, this))
}, Mscom.Slider.prototype = {
    Init: function() {
        Modernizr !== undefined && Modernizr.touch && (this.IsTouchEnabled=!0);
        $(window).on("resize", $.proxy(this.WindowResize, this));
        this.Prev.attr("href", "javascript:void(0);"), this.Next.attr("href", "javascript:void(0);"), this.WindowResize(), this.EnableNext(!0)
    },
    EnableNext: function(n) {
        this.BindNextEvents(n), this.Next.addClass("mscom-disable"), n && this.SliderItemCount > this.CurrentItemIndex + this.CurrentViewItem && this.Next.removeClass("mscom-disable")
    },
    EnablePrev: function(n) {
        this.BindPrevEvents(n), this.Prev.addClass("mscom-disable"), n && this.SliderItemCount > this.CurrentViewItem && this.CurrentItemIndex > 0 && this.Prev.removeClass("mscom-disable")
    },
    BindNextEvents: function(n) {
        if (this.IsTouchEnabled && this.SliderContainer.off("swipe" + Mscom.Left), this.Next.off("click"), n && this.SliderItemCount > this.CurrentItemIndex + this.CurrentViewItem) {
            if (this.IsTouchEnabled&&!this.IsVerticalStyle())
                this.SliderContainer.on("swipe" + Mscom.Left, $.proxy(this.SlideNext, this));
            this.Next.on("click", $.proxy(this.SlideNext, this))
        }
    },
    BindPrevEvents: function(n) {
        if (this.IsTouchEnabled && this.SliderContainer.off("swipe" + Mscom.Right), this.Prev.off("click"), n && this.SliderItemCount > this.CurrentViewItem && this.CurrentItemIndex > 0) {
            if (this.IsTouchEnabled&&!this.IsVerticalStyle())
                this.SliderContainer.on("swipe" + Mscom.Right, $.proxy(this.SlidePrev, this));
            this.Prev.on("click", $.proxy(this.SlidePrev, this))
        }
    },
    GetViewport: function() {
        var n = window, t = "inner";
        return "innerWidth"in window || (t = "client", n = document.documentElement || document.body), {
            width: n[t + "Width"],
            height: n[t + "Height"]
        }
    },
    WindowResize: function() {
        if (Mscom.BrowserDetect.browser == "Explorer" && Mscom.BrowserDetect.version < 9)
            this.SetCurrentView(4, this.DataView4, this.DataStyle4);
        else {
            var n = this.GetViewport().width;
            if (n < Mscom.ResponsiveBP1Width + 1) {
                this.SetCurrentView(1, this.DataView1, this.DataStyle1);
                return 
            }
            if (n < Mscom.ResponsiveBP2Width + 1) {
                this.SetCurrentView(2, this.DataView2, this.DataStyle2);
                return 
            }
            if (n < Mscom.ResponsiveBP3Width + 1) {
                this.SetCurrentView(3, this.DataView3, this.DataStyle3);
                return 
            }
            if (n > Mscom.ResponsiveBP3Width) {
                this.SetCurrentView(4, this.DataView4, this.DataStyle4);
                return 
            }
        }
    },
    SetCurrentView: function(n, t, i) {
        if (this.CurrentView != n || this.CurrentViewStyle != i) {
            this.CurrentView = n, this.CurrentViewItem = t, this.CurrentViewStyle = i, this.CurrentItemIndex = 0, this.EnableNext(!0), this.EnablePrev(!1), this.SliderItems.addClass("mscom-hide"), this.IsVerticalStyle() ? this.Control.addClass("mscom-slider-vertical") : this.Control.removeClass("mscom-slider-vertical");
            for (var r = 0; r < this.CurrentViewItem; r++)
                this.SliderItems.eq(r).removeClass("mscom-hide")
        }
    },
    IsVerticalStyle: function() {
        return this.CurrentViewStyle == "vertical"
    },
    GetItemMargin: function() {
        var n = $(this.Control.find(".mscom-slider-item:visible:first"));
        return parseInt(n.outerWidth(!0)) - parseInt(n.width())
    },
    GetItemDimesionToAdd: function(n, t) {
        return this.IsVerticalStyle() ? parseInt(n.outerHeight(t)) : parseInt(n.outerWidth(t))
    },
    SlidePrev: function(n) {
        n.preventDefault(), Mscom.Helper.BiTrack(n.target, "Prev", 5, 5), this.SlideClick(n, this.IsVerticalStyle() ? "bottom" : Mscom.Right, !1)
    },
    SlideNext: function(n) {
        n.preventDefault(), Mscom.Helper.BiTrack(n.target, "Next", 5, 4), this.SlideClick(n, this.IsVerticalStyle() ? "top" : Mscom.Left, !0)
    },
    SlideClick: function(n, t, i) {
        var r, u;
        this.BindNextEvents(!1), this.BindPrevEvents(!1);
        var s = 0, f = 0, e = 0, o = this.GetItemDimesionToAdd($(this.Control.find(".mscom-slider-items")), !1);
        for (i&&!this.IsVerticalStyle() && (o -= parseInt(this.GetItemMargin())), r = 0; r < this.CurrentViewItem; r++) {
            if (i) {
                if (this.CurrentItemIndex + this.CurrentViewItem + r >= this.SliderItemCount)
                    break;
                e = this.CurrentItemIndex + this.CurrentViewItem + r
            } else {
                if (this.CurrentItemIndex - r - 1 < 0)
                    break;
                e = this.CurrentItemIndex - r - 1
            }
            u = this.SliderItems.eq(e), u.css(t, o.toString() + "px"), u.addClass("mscom-currentslide"), u.removeClass("mscom-hide"), o += this.GetItemDimesionToAdd(u, !0), s += this.GetItemDimesionToAdd(u, !0), f += 1
        }
        i ? this.CurrentItemIndex += f : this.CurrentItemIndex -= f, this.SlideItems(t, s, f, i)
    },
    SlideItems: function(n, t, i) {
        var f = 0, u = this, o = parseInt(parseInt(this.SlideTransitionSpeedPerSlide) * parseInt(i)), e = {};
        e[n] = "-=" + t, this.SliderItems.animate(e, {
            duration: o,
            specialEasing: {
                width: "linear"
            },
            complete: function() {
                $(this).removeClass("mscom-currentslide"), $(this).removeAttr("style"), (f < u.CurrentItemIndex || f >= u.CurrentItemIndex + u.CurrentViewItem) && f != u.CurrentItemIndex && $(this).addClass("mscom-hide"), f >= u.SliderItemCount - 1 && (u.EnableNext(!0), u.EnablePrev(!0)), f += 1
            }
        })
    }
}, window.bk_async = function() {
    var n = document.createElement("iframe"), t, i, u, r;
    n.name = "__bkframe", n.location = "", n.src = "javascript:void(0)", n.height = 0, n.width = 0, n.frameBorder = 0, n.scrolling = "No", n.marginHeight = 0, n.marginWidth = 0, n.topMargin = 0, n.leftMargin = 0, n.style.display = "none", document.getElementsByTagName("body")[0].appendChild(n), t = "", i = document.cookie.split("; ");
    for (u in i)
        r = i[u].split("="), r[0].indexOf("MUID")!==-1 && (t = r[1]);
    bk_addPageCtx("muid", t), typeof ms != "undefined" && bk_addPageCtx("ms.linkid", ms["ms.linkid"]), BKTAG.doTag(13939, 4)
}, function() {
    var t = document.getElementsByTagName("script")[0], n = document.createElement("script");
    n.async=!0, n.src = "//tags.bkrtx.com/js/bk-coretag.js", t.parentNode.insertBefore(n, t)
}()
