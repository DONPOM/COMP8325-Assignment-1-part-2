!function(a, b) {
    "use strict";
    var c = a.adobe || (a.adobe = {});
    "undefined" == typeof c.dom && (c.dom = {}), c.dom.globalNavConfig = function() {
        var b = {
            ImsHost: a.location.hostname,
            protocol: a.location.protocol,
            ImsJS: "",
            ImsClientId: "adobedotcom2",
            ImsTestClientId: "adobedotcomTest2",
            CDN: adobeCDNHost,
            JSFile: "/assets/globalnav/" + adobeGlobalNavVersion + "-stage/js/globalnav.js",
            Typekit: "/content/dam/Adobe/typekit/yoe7ink.js",
            scope: "creative_cloud,AdobeID,openid,gnav,read_organizations,additional_info.projectedProductContext"
        };
        return b.ImsHost.indexOf("www.adobe.com")>-1 ? b.JSFile = "/assets/globalnav/" + adobeGlobalNavVersion + "/js/globalnav.js" : b.ImsHost.indexOf("stage")>-1 || b.ImsHost.indexOf(".wa.")>-1 || b.ImsHost.indexOf("helpx.qa")>-1 ? b.ImsJS = b.ImsHost.indexOf("stage2")>-1 ? "-fix" : "-stg1" : b.ImsHost.indexOf("qa")>-1&&-1 === b.ImsHost.indexOf("helpx.qa") ? (b.ImsJS = b.ImsHost.indexOf("qa01")>-1 ? "-fix" : b.ImsHost.indexOf("qa04")>-1 ? "-qa2" : "-qa1", b.ImsClientId = b.ImsTestClientId) : b.ImsHost.indexOf("dev")>-1 ? (b.ImsHost.indexOf("dev04")>-1 && (b.ImsJS = "-dev2", b.scope = "AdobeID,openid,gnav"), b.ImsHost.indexOf("dev03")>-1 ? b.ImsJS = "-stg1" : (b.ImsJS = "-dev1", b.ImsClientId = b.ImsTestClientId)) : (b.ImsHost.indexOf("127.0.0.1")>-1 || b.ImsHost.indexOf("localhost")>-1 || b.ImsHost.indexOf("local")>-1) && (b.ImsJS = "-stg1", b.CDN = "https:" === b.protocol ? "https://wwwimages2.stage.adobe.com" : "http://wwwimages.stage.adobe.com"), b
    }, c.dom.signOut = function() {
        $.removeCookie("AUID", {
            domain: "adobe.com"
        }), $.removeCookie("SCREENNAME", {
            domain: "adobe.com"
        }), $.removeCookie("WCDServer", {
            domain: "adobe.com"
        })
    }, c.dom.profile = function() {
        $.cookie("uasi", !0, {
            path: "/",
            domain: ".adobe.com"
        })
    }, c.dom.globalNavVars = c.dom.globalNavConfig();
    {
        var d = a.adobeid = {
            uses_redirect_mode: !0,
            client_id: c.dom.globalNavVars.ImsClientId,
            scope: c.dom.globalNavVars.scope,
            api_parameters: {
                authorize: {
                    state: {
                        ac: "adobe.com"
                    }
                }
            },
            onProfile: function(a) {
                a ? c.dom.profile() : c.dom.signOut()
            }
        };
        c.dom.adobeGlobalNavOptions = {
            adobeid: d,
            parentSelector: ".dom-globalnav"
        }
    }
    b.writeln('<script src="' + c.dom.globalNavVars.CDN + c.dom.globalNavVars.JSFile + '"></script>')
}(window, document), function(a) {
    "use strict";
    var b = a.adobe || (a.adobe = {});
    b.events = {
        _eventsRegistry: {},
        on: function(a, b) {
            if (0 === a.indexOf("ims:") || 0 === a.indexOf("gnav:"))
                this._addImsGnavCallback(a, b);
            else if (0 === a.indexOf("dom:")) {
                var c = this._eventsRegistry[a] || (this._eventsRegistry[a] = []);
                c.push(b)
            }
        },
        off: function(a, b) {
            var c = this._eventsRegistry[a];
            if (c) {
                var d = c.indexOf(b);
                - 1 !== d && c.splice(d, 1)
            }
        },
        has: function(a) {
            var b = this._eventsRegistry[a];
            return b && b.length > 0
        },
        _addImsGnavCallback: function(c, d) {
            var e, f = c.split(":"), g = f[0], h = f[1];
            if ("ims" === g)
                e = a.adobeid;
            else {
                var i = b.dom.adobeGlobalNavOptions;
                e = i.events || (i.events = {})
            }
            var j = this._eventsRegistry[c];
            if (!j) {
                j = this._eventsRegistry[c] = [];
                var k = e[h];
                k && j.push(k), e[h] = function() {
                    for (var a = j.slice(0), b = 0; b < a.length; b++) {
                        var c = a[b];
                        try {
                            c.apply(this, arguments)
                        } catch (d) {}
                    }
                }
            }
            j.push(d)
        }
    }
}(window, document);
