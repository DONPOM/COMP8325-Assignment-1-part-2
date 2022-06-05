var JSON;
if (!JSON) {
    JSON = {}
}(function() {
    function f(n) {
        return n < 10 ? "0" + n : n
    }
    if (typeof Date.prototype.toJSON !== "function") {
        Date.prototype.toJSON = function(key) {
            return isFinite(this.valueOf()) ? this.getUTCFullYear() + "-" + f(this.getUTCMonth() + 1) + "-" + f(this.getUTCDate()) + "T" + f(this.getUTCHours()) + ":" + f(this.getUTCMinutes()) + ":" + f(this.getUTCSeconds()) + "Z" : null
        };
        String.prototype.toJSON = Number.prototype.toJSON = Boolean.prototype.toJSON = function(key) {
            return this.valueOf()
        }
    }
    var cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, gap, indent, meta = {
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
        '"': '\\"',
        "\\": "\\\\"
    }, rep;
    function quote(string) {
        escapable.lastIndex = 0;
        return escapable.test(string) ? '"' + string.replace(escapable, function(a) {
            var c = meta[a];
            return typeof c === "string" ? c : "\\u" + ("0000" + a.charCodeAt(0).toString(16)).slice( - 4)
        }) + '"' : '"' + string + '"'
    }
    function str(key, holder) {
        var i, k, v, length, mind = gap, partial, value = holder[key];
        if (value && typeof value === "object" && typeof value.toJSON === "function") {
            value = value.toJSON(key)
        }
        if (typeof rep === "function") {
            value = rep.call(holder, key, value)
        }
        switch (typeof value) {
        case"string":
            return quote(value);
        case"number":
            return isFinite(value) ? String(value) : "null";
        case"boolean":
        case"null":
            return String(value);
        case"object":
            if (!value) {
                return "null"
            }
            gap += indent;
            partial = [];
            if (Object.prototype.toString.apply(value) === "[object Array]") {
                length = value.length;
                for (i = 0; i < length; i += 1) {
                    partial[i] = str(i, value) || "null"
                }
                v = partial.length === 0 ? "[]" : gap ? "[\n" + gap + partial.join(",\n" + gap) + "\n" + mind + "]" : "[" + partial.join(",") + "]";
                gap = mind;
                return v
            }
            if (rep && typeof rep === "object") {
                length = rep.length;
                for (i = 0; i < length; i += 1) {
                    if (typeof rep[i] === "string") {
                        k = rep[i];
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ": " : ":") + v)
                        }
                    }
                }
            } else {
                for (k in value) {
                    if (Object.prototype.hasOwnProperty.call(value, k)) {
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ": " : ":") + v)
                        }
                    }
                }
            }
            v = partial.length === 0 ? "{}" : gap ? "{\n" + gap + partial.join(",\n" + gap) + "\n" + mind + "}" : "{" + partial.join(",") + "}";
            gap = mind;
            return v
        }
    }
    if (typeof JSON.stringify !== "function") {
        JSON.stringify = function(value, replacer, space) {
            var i;
            gap = "";
            indent = "";
            if (typeof space === "number") {
                for (i = 0; i < space; i += 1) {
                    indent += " "
                }
            } else {
                if (typeof space === "string") {
                    indent = space
                }
            }
            rep = replacer;
            if (replacer && typeof replacer !== "function" && (typeof replacer !== "object" || typeof replacer.length !== "number")) {
                throw new Error("JSON.stringify")
            }
            return str("", {
                "": value
            })
        }
    }
    if (typeof JSON.parse !== "function") {
        JSON.parse = function(text, reviver) {
            var j;
            function walk(holder, key) {
                var k, v, value = holder[key];
                if (value && typeof value === "object") {
                    for (k in value) {
                        if (Object.prototype.hasOwnProperty.call(value, k)) {
                            v = walk(value, k);
                            if (v !== undefined) {
                                value[k] = v
                            } else {
                                delete value[k]
                            }
                        }
                    }
                }
                return reviver.call(holder, key, value)
            }
            text = String(text);
            cx.lastIndex = 0;
            if (cx.test(text)) {
                text = text.replace(cx, function(a) {
                    return "\\u" + ("0000" + a.charCodeAt(0).toString(16)).slice( - 4)
                })
            }
            if (/^[\],:{}\s]*$/.test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, "@").replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:\s*\[)+/g, ""))) {
                j = eval("(" + text + ")");
                return typeof reviver === "function" ? walk({
                    "": j
                }, "") : j
            }
            throw new SyntaxError("JSON.parse")
        }
    }
}());
window.page_unload = false;
(function(c) {
    var b = c.setTimeout, d = c.document, a = 0;
    c.jXHR = function() {
        var e, g, p, h, o = null, l = false;
        window.onbeforeunload = function(q) {
            window.page_unload = true
        };
        function m() {
            try {
                if (h && h.parentNode) {
                    h.parentNode.removeChild(h);
                    for (var r in h) {
                        delete h[r]
                    }
                    h = null
                }
            } catch (q) {}
        }
        function k() {
            g = false;
            e = "";
            m();
            h = null;
            i(0)
        }
        function f(r) {
            try {
                o.onerror.call(o, r, e)
            } catch (q) {
                throw new Error(r)
            }
        }
        function j() {
            if ((this.readyState && this.readyState !== "complete" && this.readyState !== "loaded") || g) {
                return 
            }
            this.onload = this.onreadystatechange = null;
            g = true;
            if (o.readyState !== 4) {
                f("Script failed to load [" + e + "].")
            }
            m()
        }
        function n() {
            if (!g) {
                f("Script failed to load [" + e + "], most probably due to an invalid URL or server error.")
            }
        }
        function i(q, r) {
            r = r || [];
            o.readyState = q;
            if (typeof o.onload === "function") {
                o.onload.apply(o, r);
                return 
            }
            if (typeof o.onreadystatechange === "function") {
                o.onreadystatechange.apply(o, r)
            }
        }
        o = {
            onerror: null,
            onreadystatechange: null,
            onload: null,
            readyState: 0,
            open: function(s, q) {
                k();
                var r = "cb" + (a++);
                (function(t) {
                    c.jXHR[t] = function() {
                        try {
                            i.call(o, 4, arguments)
                        } catch (u) {
                            Netdania.JsApi.Utilities.Log("Err: " + u.name + ";" + u.message + ";" + u.description);
                            o.readyState =- 1;
                            f("Script failed to run [" + e + "].")
                        }
                        c.jXHR[t] = null
                    }
                })(r);
                e = q.replace(/=\?/, "=jXHR." + r);
                i(1)
            },
            send: function() {
                b(function() {
                    h = d.createElement("script");
                    h.setAttribute("type", "text/javascript");
                    h.onload = h.onreadystatechange = function() {
                        j.call(h)
                    };
                    h.setAttribute("src", e);
                    d.getElementsByTagName("head")[0].appendChild(h);
                    h.onerror = function() {
                        if (!window.page_unload) {
                            n.call(h)
                        }
                    }
                }, 0);
                i(2)
            },
            setRequestHeader: function() {},
            getResponseHeader: function() {
                return ""
            },
            getAllResponseHeaders: function() {
                return []
            }
        };
        k();
        return o
    }
})(window);
var Netdania = Netdania || {};
Netdania.JsApi = {};
Netdania.JsApi.Utilities = {};
Netdania.JsApi.Response = {};
Netdania.JsApi.Request = {};
Netdania.JsApi.Utilities.ua = navigator.userAgent.toLowerCase();
if (Netdania.JsApi.Utilities.ua.indexOf(" chrome/") >= 0 || Netdania.JsApi.Utilities.ua.indexOf(" firefox/") >= 0 || Netdania.JsApi.Utilities.ua.indexOf(" gecko/") >= 0) {
    Netdania.JsApi.StringMaker = function() {
        this.str = "";
        this.length = 0;
        this.append = function(a) {
            this.str += a;
            this.length += a.length
        };
        this.prepend = function(a) {
            this.str = a + this.str;
            this.length += a.length
        };
        this.toString = function() {
            return this.str
        }
    }
} else {
    Netdania.JsApi.StringMaker = function() {
        this.parts = [];
        this.length = 0;
        this.append = function(a) {
            this.parts.push(a);
            this.length += a.length
        };
        this.prepend = function(a) {
            this.parts.unshift(a);
            this.length += a.length
        };
        this.toString = function() {
            return this.parts.join("")
        }
    }
}
Netdania.JsApi.Utilities.LogError = function(c, a) {
    var b = "";
    if (a !== undefined && a != null && a.message !== undefined && a.message != null) {
        b = a.message
    }
    Netdania.JsApi.Utilities.LogData(c + " " + b)
};
Netdania.JsApi.Utilities.Log = function(a) {
    if (false) {
        Netdania.JsApi.Utilities.LogData(a)
    }
};
Netdania.JsApi.Utilities.LogData = function(c, e) {
    if (typeof console == "undefined") {
        console = {
            log: function() {}
        }
    }
    console.log(c);
    var d = e || false;
    if (e) {
        var f = Netdania.JsApi.Utilities.GetURL();
        c += ";" + encodeURI(f);
        var a = null;
        var a = new jXHR();
        a.onerror = function(g) {
            console.log(g)
        };
        var b = "http://www.netdania.com/JSONLogging/LogJSON.aspx?data=" + c;
        a.open("GET", b, true);
        a.send()
    }
};
Netdania.JsApi.Utilities.keyStr = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.";
Netdania.JsApi.encode64 = function(c) {
    var a = new Netdania.JsApi.StringMaker();
    var k, h, f;
    var j, g, e, d;
    var b = 0;
    while (b < c.length) {
        k = c.charCodeAt(b++);
        h = c.charCodeAt(b++);
        f = c.charCodeAt(b++);
        j = k>>2;
        g = ((k & 3)<<4) | (h>>4);
        e = ((h & 15)<<2) | (f>>6);
        d = f & 63;
        if (isNaN(h)) {
            e = d = 64
        } else {
            if (isNaN(f)) {
                d = 64
            }
        }
        a.append(Netdania.JsApi.Utilities.keyStr.charAt(j) + Netdania.JsApi.Utilities.keyStr.charAt(g) + Netdania.JsApi.Utilities.keyStr.charAt(e) + Netdania.JsApi.Utilities.keyStr.charAt(d))
    }
    return a.toString()
};
Netdania.JsApi.Request.getReqObjPrice = function(c, b, a) {
    var e=++Netdania.JsApi.globalCurrentReqId;
    var d = {
        t: 1,
        i: e,
        m: a ? 1: 0,
        s: c,
        p: b
    };
    return d
};
Netdania.JsApi.Request.getReqObjChart = function(c, g, f, b, a) {
    var e=++Netdania.JsApi.globalCurrentReqId;
    var d = {
        t: 2,
        i: e,
        m: a ? 1: 0,
        s: c,
        p: b,
        ts: g,
        pt: f
    };
    return d
};
Netdania.JsApi.Request.getReqChartHistory = function(b, d, e, c, a, i, g) {
    var f=++Netdania.JsApi.globalCurrentReqId;
    var h = {
        t: 12,
        i: f,
        m: a ? 1: 0,
        s: b,
        p: c,
        ts: d,
        pt: e,
        sd: i,
        ed: g
    };
    return h
};
Netdania.JsApi.Request.getReqObjRemove = function(a) {
    var b = {
        t: 5,
        i: a
    };
    return b
};
Netdania.JsApi.Request.getReqObjHeadlines = function(d, c, b, a) {
    var f=++Netdania.JsApi.globalCurrentReqId;
    var e = {
        t: 3,
        i: f,
        m: a ? 1: 0,
        s: d,
        p: b,
        max: c
    };
    return e
};
Netdania.JsApi.Request.getReqObjStory = function(a, b) {
    var d=++Netdania.JsApi.globalCurrentReqId;
    var c = {
        t: 4,
        i: d,
        s: a,
        p: b
    };
    return c
};
Netdania.JsApi.Request.getReqObjInstrumentLookup = function(d, e, h, f, a, b, c) {
    var g=++Netdania.JsApi.globalCurrentReqId;
    var i = {
        t: 9,
        i: g,
        p: c,
        mkt: d,
        fid: e,
        str: h,
        mode: f,
        it: a,
        max: b
    };
    return i
};
Netdania.JsApi.Request.getReqObjNewsSearch = function(h, f, a, i, d, c, b) {
    var e=++Netdania.JsApi.globalCurrentReqId;
    var g = {
        t: 10,
        i: e,
        s: h,
        str: f,
        max: a,
        st: i,
        et: d,
        f: c,
        p: b
    };
    return g
};
Netdania.JsApi.Request.getReqObjAlertAddAlert = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.ADD_ALERT
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertGetUserActiveAlerts = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.GET_USER_INFORMATION
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertRequestPushDevices = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.GET_PUSH_DEVICES
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertEdit = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.EDIT_ALERT
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertGet = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.GET_SINGLE_ALERT
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertDelete = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.DELETE_ALERT
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertGetDeleted = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.GET_DELETED_ALERTS
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertGetTriggered = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.GET_TRIGGERED_ALERTS
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertMonitorUserActivity = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        m: 1,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.MONITOR_USER_ACTIVITIES
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertDisconnectMonitorUserActivity = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.DISCONNECT_USER_ACTIVITIES
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertMonitorUser = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        m: 1,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.MONITOR_USER
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertDisconnectMonitorUser = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.DISCONNECT_USER
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjAlertGetPushDevices = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.GET_PUSH_DEVICES
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Request.getReqObjIPLocation = function() {
    var b=++Netdania.JsApi.globalCurrentReqId;
    var a = {
        t: 38,
        i: b
    };
    return a
};
Netdania.JsApi.Request.getReqCloseConnection = function() {
    var b=++Netdania.JsApi.globalCurrentReqId;
    var a = {
        t: 39,
        i: b,
        c: 0
    };
    return a
};
Netdania.JsApi.Request.getReqObjAlertAddUser = function(a) {
    var c=++Netdania.JsApi.globalCurrentReqId;
    var b = {
        t: 37,
        i: c,
        o: {
            f: [],
            i: Netdania.JsApi.Alert.Commands.ADD_USER
        }
    };
    b.o.f = Netdania.JsApi.Alert.GetFieldsFromAlertObject(a);
    return b
};
Netdania.JsApi.Alert = Netdania.JsApi.Alert || {};
Netdania.JsApi.Alert.AlertObject = function() {
    this.ALERT_ID =- 1;
    this.USER_ID = "";
    this.PREVIOUS_OWNER = "";
    this.ALERT_CONDITION = "";
    this.DAYS_TO_LIVE = "";
    this.SMS_PHONE_NUMBER = "";
    this.SMS_FROM_NAME = "";
    this.SMS_MESSAGE = "";
    this.EMAIL_ADDRESS = "";
    this.EMAIL_FROM = "";
    this.EMAIL_SUBJECT = "";
    this.EMAIL_CONTENT = "";
    this.YAHOO_ID = "";
    this.YAHOO_MESSAGE = "";
    this.SKYPE_ID = "";
    this.SKYPE_MESSAGE = "";
    this.MSN_ID = "";
    this.MSN_MESSAGE = "";
    this.TRIGGER_DATE =- 1;
    this.ORDER_TYPE = "";
    this.PUSH_DEVICES_ID = "";
    this.PUSH_MESSAGE = "";
    this.IS_ALERT_CENTRAL = "";
    this.ALERTS_INCLUDED = "";
    this.USER_DETAILS_INCLUDED = "";
    this.COOKIE = "";
    this.STARTDATE = "";
    this.ENDDATE = "";
    this.TIMESCALE_SECONDS = 0;
    this.DELETION_REASON = ""
};
Netdania.JsApi.Alert.AlertObject2 = {
    USER_ID: - 1,
    previousOwner: "",
    alertCondition: "",
    daysToLive: "",
    smsPhoneNumber: "",
    smsFromName: "",
    smsMessage: "",
    emailAddress: "",
    emailFromName: "",
    emailSubject: "",
    emailContent: "",
    yahooID: "",
    yahooMessage: "",
    skypeID: "",
    skypeMessage: "",
    msnID: "",
    msnMessage: "",
    triggerDate: 0,
    orderType: "",
    pushDevices: "",
    pushMessage: "",
    pushSubject: ""
};
Netdania.JsApi.Alert.GetFieldsFromAlertObject = function(a) {
    var e = [];
    var c = Object.keys(new Netdania.JsApi.Alert.AlertObject());
    for (var b = 0; b < c.length; b++) {
        var d = {};
        d.v = [a[c[b]]];
        if (d.v != "" && d.v!=-1 && d.v != 0) {
            d.f = Netdania.JsApi.Alert.Fields[c[b]];
            d.t = Netdania.JsApi.Alert.FieldTypes[c[b]];
            e.push(d)
        }
    }
    d = {};
    d.f = 1;
    d.t = 2;
    d.v = [Netdania.JsApi.Utilities.GetHost()];
    return e
};
Netdania.JsApi.Response.MonitorChartResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c,
        get: function(e) {
            if (c != null) {
                for (var f = 0; f < c.length; f++) {
                    if (c[f].f == e) {
                        return c[f].v
                    }
                }
            }
            return null
        },
        getDecimals: function(e) {
            if (c != null) {
                for (var f = 0; f < c.length; f++) {
                    if (c[f].f == e) {
                        return c[f].d
                    }
                }
            }
            return 0
        }
    };
    return a
};
Netdania.JsApi.Response.ChartUpdateResponse = function(d, b, e, c) {
    var a = {
        type: b,
        id: e,
        data: d,
        ts: c,
        getClose: function() {
            if (d !== null) {
                for (var f = 0; f < d.length; f++) {
                    if (d[f].f == Netdania.JsApi.Fields.CHART_CLOSE) {
                        return d[f].v
                    }
                }
            }
            return "N/A"
        },
        getVolume: function() {
            if (d !== null) {
                for (var f = 0; f < d.length; f++) {
                    if (d[f].f == Netdania.JsApi.Fields.CHART_VOLUME) {
                        return d[f].v
                    }
                }
            }
            return "N/A"
        }
    };
    a.close = a.getClose();
    a.volume = a.getVolume();
    return a
};
Netdania.JsApi.Response.AlertAddedResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c,
        get: function(e) {
            for (var f = 0; f < c.length; f++) {
                if (c[f].f == e) {
                    return c[f].v
                }
            }
            return null
        }
    };
    return a
};
Netdania.JsApi.Response.AlertActive = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c,
        get: function(e) {
            for (var f = 0; f < c.length; f++) {
                if (c[f].f == e) {
                    return c[f].v
                }
            }
            return null
        }
    };
    return a
};
Netdania.JsApi.Response.MonitorPriceResponse = function(d, b, e) {
    var c = (d && d.f) ? d.f: null;
    var a = {
        type: b,
        id: e,
        data: c,
        modifiedFids: [],
        getFIDs: function() {
            var g = [];
            if (c !== null) {
                for (var f = 0; f < c.length; f++) {
                    g.push(c[f].f)
                }
                g
            }
            return g
        },
        get: function(g) {
            if (c !== null && this.getFIDs(c).contains(g)) {
                for (var f = 0; f < c.length; f++) {
                    if (c[f].f === g) {
                        return c[f].v
                    }
                }
            } else {
                return "N/A"
            }
        }
    };
    a.modifiedFids = a.getFIDs();
    return a
};
Netdania.JsApi.Response.NewsHistoryResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c
    };
    return a
};
Netdania.JsApi.Response.LookupResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c
    };
    return a
};
Netdania.JsApi.Response.MonitorNewsResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c
    };
    return a
};
Netdania.JsApi.Response.NewsStoryResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c
    };
    return a
};
Netdania.JsApi.Response.NewsSearchResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c
    };
    return a
};
Netdania.JsApi.Response.IPLocationResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c
    };
    return a
};
Netdania.JsApi.Response.GeneralMonitorResponse = function(c, b, d) {
    var a = {
        type: b,
        id: d,
        data: c
    };
    return a
};
Netdania.JsApi.getXReqPageSize = function() {
    var c = navigator.userAgent.toLowerCase();
    var a = 4;
    if (c.indexOf("chrome") >= 0 || c.indexOf("firefox") >= 0 || c.indexOf("gecko") >= 0) {
        a = 20
    }
    var b = c.indexOf("msie");
    if (b >= 0) {
        try {
            if (parseFloat(c.substring(b + 4)) > 7) {
                a = 100
            }
        } catch (d) {}
    }
    return a
};
Netdania.JsApi.unpackChartSeries = function(h, g) {
    var a = h.split(",");
    var d = new Array(a.length);
    var e = Math.pow(10, g);
    var f = 0;
    var b;
    for (var c = 0; c < a.length; c++) {
        b = parseInt(a[c], 36) + f;
        d[c] = b / e;
        f = b
    }
    a = e = null;
    return d
};
Netdania.JsApi.Utilities.GetHost = function() {
    return window.location.hostname
};
Netdania.JsApi.Utilities.GetURL = function() {
    return window.location.href
};
Netdania.JsApi.Events = {
    ONCONNECTED: "OnConnected",
    ONUPDATE: "OnUpdate",
    ONRAWUPDATE: "OnRawUpdate",
    ONDISCONNECTED: "OnDisconnected",
    ONINIT: "OnInit",
    ONRECONNECTED: "OnReconnect",
    ONPRICEUPDATE: "OnPriceUpdate",
    ONCHARTUPDATE: "OnChartUpdate",
    ONHISTORICALDATA: "OnHistoricalData",
    ONERROR: "OnError",
    ONINFO: "OnInfo",
    ONLOOKUP: "OnLookup",
    ONHISTORICALHEADLINES: "OnHistHeadlines",
    ONHEADLINEUPDATE: "OnHeadlineUpdate",
    ONNEWSSTORY: "OnNewsHist",
    ONNEWSSEARCH: "OnNewsSearch",
    ONALERTADDED: "OnAlertAdded",
    ONALERTDELETE: "OnAlertDelete",
    ONALERTDISCONNECTMONITORUSER: "OnAlertDisconnectMonitorUser",
    ONALERTDISCONNECTMONITORUSERACTIVITY: "OnAlertDisconnectMonitorUserActivity",
    ONALERTEDIT: "OnAlertEdit",
    ONALERTGET: "OnAlertGet",
    ONALERTGETDELETED: "OnAlertGetDeleted",
    ONALERTGETTRIGGERED: "OnAlertGetTriggered",
    ONALERTGETACTIVE: "OnAlertGetActive",
    ONALERTMONITORUSER: "OnAlertMonitorUser",
    ONALERTMONITORUSERACTIVITY: "OnAlertMonitorUserActivity",
    ONALERTGETPUSHDEVICES: "OnAlertGetPushDevices",
    ONIPLOCATIONRESPONSE: "OnIPLocationResponse"
};
Netdania.JsApi.ConnectionStatus = {
    LOGIN_OK: {
        code: 0,
        message: "LOGIN_OK"
    },
    LOGIN_ERROR: {
        code: 1,
        message: "LOGIN_ERROR"
    },
    LOGIN_ERROR_INVALID_USERGROUP: {
        code: 3,
        message: "LOGIN_ERROR_INVALID_USERGROUP"
    },
    LOGIN_ERROR_SERVICE_NOT_ALLOWED: {
        code: 4,
        message: "LOGIN_ERROR_SERVICE_NOT_ALLOWED"
    }
};
Netdania.JsApi.Messages = {
    DISCONNECT_MSG: "Disconnected from server",
    HISTORICAL_DATA_MSG: "historical data...",
    CONNECTING_MSG: "connecting...",
    RECONNECTING_MSG: "reconnecting...",
    PRICE_UPDATE_MSG: "price update...",
    CHART_UPDATE_MSG: "chart update...",
    HISTORICAL_HEADLINES_MSG: "historical headlines...",
    HEADLINE_UPDATE_MSG: "headline update...",
    HEADLINE_STORY_MSG: "headline update...",
    HEADLINE_SEARCH_MSG: "headline update...",
    LOOKUP_MSG: "headline update...",
    INVALID_HOST_ERR_MSG: "Error: Invalid host",
    INSTRUMENT_ALREADY_ADDED_ERR_MSG: "Error: instrument already added.",
    ALERT_ADDED: "Alert added.",
    ALERT_DELETED: "Alert deleted.",
    ALERT_DISCONNECTED_MONITORUSER: "Alert monitor user disconnected.",
    ALERT_DISCONNECTED_MONITORUSER_ACTIVITY: "Alert monitor user activity disconnected.",
    ALERT_EDIT: "Alert edited.",
    ALERT_GET: "Alert get.",
    ALERT_GET_DELETED: "Alert get deleted.",
    ALERT_GET_TRIGGERED: "Alert get triggered.",
    ALERT_GET_ACTIVE: "Alert get active received.",
    ALERT_MONITORUSER: "Alert monitor user.",
    ALERT_MONITORUSER_ACTIVITY: "Alert monitor user activity",
    ALERT_GET_PUSH_DEVICES: "Alert push devices received",
    IP_LOCATION: "IP Location received"
};
Netdania.JsApi.PoolingInterval = {
    AUTO: "0"
};
Netdania.JsApi.ConnectionType = {
    POLLING: "2",
    LONGPOLLING: "3",
    STREAMING: "1",
    AUTO: "0"
};
Netdania.JsApi.DetectCORSAvailability = function(a) {
    var b = 1;
    if ("withCredentials" in new XMLHttpRequest()) {
        a && (window.jXHR = XMLHttpRequest, b = 0)
    }
    return b
};
Netdania.JsApi.globalCurrentReqId = 0;
Netdania.JsApi.Connections = [];
Netdania.JsApi.TryCreateConnection = function(a, c) {
    var b;
    if (Netdania.JsApi.Connections !== undefined && Netdania.JsApi.Connections != null) {
        for (var d = 0; d < Netdania.JsApi.Connections.length; d++) {
            b = Netdania.JsApi.Connections[d];
            if (c.behavior !== undefined && (b.behavior == c.behavior || c.behavior == Netdania.JsApi.ConnectionType.AUTO)) {
                if (c.host !== undefined && b.host == c.host) {
                    if (c.v !== undefined && b.v == c.v) {
                        if (c.pollingInterval !== undefined && c.pollingInterval != Netdania.JsApi.PoolingInterval.AUTO && c.pollingInterval < b.interval) {
                            b.instance.SetPoolingInterval(c.pollingInterval);
                            b.interval = c.pollingInterval
                        }
                        return {
                            c: b,
                            n: false
                        }
                    }
                }
            }
        }
    }
    if (c.behavior == Netdania.JsApi.ConnectionType.AUTO) {
        c.behavior = Netdania.JsApi.ConnectionType.POLLING
    }
    if (c.pollingInterval == Netdania.JsApi.PoolingInterval.AUTO) {
        c.pollingInterval = 1000
    }
    b = {
        v: c.v,
        host: c.host,
        behavior: c.behavior,
        instance: a,
        interval: c.pollingInterval
    };
    Netdania.JsApi.Connections = Netdania.JsApi.Connections || [];
    Netdania.JsApi.Connections.push(b);
    return {
        c: b,
        n: true
    }
};
Netdania.JsApi.JSONConnection = function(y) {
    Netdania.JsApi.Utilities.Log("connection started...");
    if (!y.v) {
        y.v = 2
    }
    if (y.host === undefined || y.host == null || y.host === "") {
        y.host = "http://balancer11.netdania.com/StreamingServer/StreamingServer"
    }
    var m = this;
    m.config = y;
    var z = Netdania.JsApi.TryCreateConnection(this, y);
    if (!z.n) {
        return z.c.instance
    }
    this.Observer = new Netdania.UpdatesObserver();
    var l = Netdania.JsApi.getXReqPageSize();
    var n = y.pollingInterval >= 1000 ? y.pollingInterval: 1000;
    var b = y.behavior;
    var p = y.type;
    var u = null;
    var o = true;
    (y.useCORS != null && y.useCORS !== "undefined" && y.useCORS !== undefined) && (o = y.useCORS);
    p = Netdania.JsApi.DetectCORSAvailability(o);
    var t = false;
    var c = false;
    var r = [];
    var e = [];
    var s = false;
    var k = null;
    var x = null;
    var q = false;
    var h;
    Events.enable.call(this);
    this.SetPoolingInterval = function(A) {
        n = A
    };
    this.Connect = function(A) {
        var G = Netdania.JsApi.Utilities.GetHost();
        if (G === "" || G === null || G === undefined) {
            throw new Error(Netdania.JsApi.Messages.INVALID_HOST_ERR_MSG)
        }
        if (t || (x && q)) {
            return 
        }
        if ((e.length > 0 && s) || (A && r.length > 0)) {
            x = new jXHR();
            q = true;
            c = false;
            this.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.CONNECTING_MSG]);
            var F = {
                g: G,
                ai: "jsApi 1.0.1897",
                pr: b,
                au: Netdania.JsApi.Utilities.GetURL(),
                qup: 1
            };
            var D = Netdania.JsApi.encode64(JSON.stringify(F));
            if ((e.length > 0 && s) || (A && r.length > 0)) {
                var E = "";
                if (e.length > 0) {
                    var B = e.slice(0, l);
                    e = e.slice(l, e.length);
                    E = Netdania.JsApi.encode64(JSON.stringify(B));
                    Netdania.JsApi.Utilities.Log("requestQueue: " + JSON.stringify(B));
                    if (e.length == 0) {
                        s = false
                    }
                } else {
                    if (A && r.length > 0) {
                        E = Netdania.JsApi.encode64(JSON.stringify(r));
                        Netdania.JsApi.Utilities.Log("requestList: " + JSON.stringify(r));
                        s = false
                    }
                }
            }
            var C = m.config.host + "?xstream=1&v=" + y.v + "&dt=" + p + "&h=" + D + "&xcmd=" + E + "&cb=?&ts=" + Math.random();
            x.onerror = function(I, H) {
                m.fireEvent(Netdania.JsApi.Events.ONDISCONNECTED, [I, H]);
                q = false;
                t = false;
                v()
            };
            x.onload = function(I) {
                if (!p) {
                    if (this.responseText == undefined) {
                        return 
                    }
                    I = JSON.parse(this.responseText)
                }
                Netdania.JsApi.Utilities.Log("callback");
                if (this.readyState === undefined || this.readyState === 4) {
                    var H = I[1].s;
                    if (H !== Netdania.JsApi.ConnectionStatus.LOGIN_OK.code) {
                        m.fireEvent(Netdania.JsApi.Events.ONERROR, [I[1].m]);
                        m.fireEvent(Netdania.JsApi.Events.ONDISCONNECTED, [Netdania.JsApi.Messages.DISCONNECT_MSG]);
                        return 
                    }
                    u = I[1].m;
                    t = true;
                    m.fireEvent(Netdania.JsApi.Events.ONCONNECTED, [u]);
                    q = false;
                    u = I[1].m;
                    h = m.config.host + "?dt=" + p + "&sessid=" + u + "&cb=?&xpoll";
                    if (I[2]) {
                        j(I)
                    }
                    if (e.length > 0 && s) {
                        i(e);
                        e = [];
                        s = false
                    }
                    x.onreadystatechange = null;
                    x.onload = null;
                    k = setTimeout(a, 50);
                    k = null
                }
            };
            x.open("GET", C, true);
            x.send()
        }
    };
    this.Disconnect = function() {
        var A = Netdania.JsApi.Request.getReqCloseConnection();
        this.removeRequests(r);
        i([A]);
        t = false;
        e = [];
        r = [];
        c = true
    };
    var v = function() {
        Netdania.JsApi.Utilities.Log(encodeURI("reconnect..."));
        setTimeout(function() {
            m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.RECONNECTING_MSG]);
            if (q === false&&!t) {
                x = null;
                m.Connect(true)
            }
        }, 1000)
    };
    var a = function(B) {
        clearTimeout(k);
        Netdania.JsApi.Utilities.Log(encodeURI("polling..."));
        if (!t) {
            v()
        }
        if (q === false && h !== undefined&&!c) {
            var A = new jXHR();
            A.url = h + "&ts=" + Math.random();
            q = true;
            A.onerror = function(D, C) {
                m.fireEvent(Netdania.JsApi.Events.ONDISCONNECTED, [D, C]);
                q = false;
                t = false;
                v()
            };
            A.onload = function(C) {
                Netdania.JsApi.Utilities.Log(encodeURI("polling callback..."));
                if (!p) {
                    if (this.responseText === undefined) {
                        return 
                    }
                    C = JSON.parse(this.responseText)
                }
                if (this.readyState === undefined || this.readyState === 4) {
                    q = false;
                    j(C);
                    A.onreadystatechange = null;
                    A.onerror = null;
                    if (B === undefined) {
                        B = true
                    }
                    if (B) {
                        clearTimeout(k);
                        k = setTimeout(a, n)
                    }
                    A = null;
                    C = null
                }
            };
            A.open("GET", A.url, true);
            A.send()
        }
    };
    var j = function(T) {
        Netdania.JsApi.Utilities.Log("callback");
        var B = null, R = null;
        if (T !== "" && T !== undefined) {
            if (m.config.v === 2 || m.config.v === 3) {
                for (var O = 0; O < T.length; O++) {
                    if (T[O].t === 2) {
                        var J = [];
                        for (var N = 0; N < T[O].f.length; N++) {
                            var L = {};
                            L.f = T[O].f[N];
                            L.v = T[O].v[N];
                            J.push(L)
                        }
                        T[O].f = J
                    }
                }
            }
            m.fireEvent(Netdania.JsApi.Events.ONRAWUPDATE, [T]);
            for (var O = 0; O < T.length; O++) {
                switch (T[O].t) {
                case 4:
                    B = T[O].f;
                    for (var N = 0; N < B.length; N++) {
                        var I = B[N].f;
                        switch (I) {
                        case 107:
                        case 108:
                            break;
                        default:
                            B[N].v = Netdania.JsApi.unpackChartSeries(B[N].v, B[N].d);
                            break
                        }
                    }
                    var E = Netdania.JsApi.Response.MonitorChartResponse(B, T[O].t, T[O].i);
                    m.Observer.init(B, T[O].i);
                    m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                    m.fireEvent(Netdania.JsApi.Events.ONHISTORICALDATA, [E]);
                    m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.HISTORICAL_DATA_MSG]);
                    B = null;
                    break;
                case 2:
                    try {
                        B = {};
                        B.f = T[O].f;
                        B.v = T[O].v;
                        var G = Netdania.JsApi.Response.MonitorPriceResponse(B, T[O].t, T[O].i);
                        m.Observer.update(B, "", T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONPRICEUPDATE, [G]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.PRICE_UPDATE_MSG]);
                        B = null;
                        R = null
                    } catch (Q) {}
                    break;
                case 18:
                    try {
                        B = T[O].f;
                        R = T[O].rt;
                        var U = Netdania.JsApi.Response.ChartUpdateResponse(B, T[O].t, T[O].i, R);
                        m.Observer.update(B, R, T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, R, T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONCHARTUPDATE, [U]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.CHART_UPDATE_MSG]);
                        B = null;
                        R = null
                    } catch (Q) {}
                    break;
                case 6:
                    try {
                        B = T[O].h;
                        var K = Netdania.JsApi.Response.NewsHistoryResponse(B, T[O].t, T[O].i);
                        m.Observer.update(B, "", T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONHISTORICALHEADLINES, [K]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.HISTORICAL_HEADLINES_MSG]);
                        B = null;
                        R = null
                    } catch (Q) {}
                    break;
                case 19:
                    try {
                        B = T[O].h;
                        var V = Netdania.JsApi.Response.MonitorNewsResponse(B, T[O].t, T[O].i);
                        m.Observer.update(B, "", T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONHEADLINEUPDATE, [V]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.HEADLINE_UPDATE_MSG]);
                        B = null;
                        R = null
                    } catch (Q) {}
                    break;
                case 8:
                    try {
                        B = T[O].s;
                        R = T[O].rt;
                        var C = Netdania.JsApi.Response.NewsStoryResponse(B, T[O].t, T[O].i);
                        m.Observer.update(B, R, T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, R, T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONNEWSSTORY, [C]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.NEWS_STORY_MSG]);
                        B = null;
                        R = null
                    } catch (Q) {}
                    break;
                case 15:
                    try {
                        B = T[O].h;
                        var M = Netdania.JsApi.Response.NewsSearchResponse(B, T[O].t, T[O].i);
                        m.Observer.update(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONNEWSSEARCH, [M]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.NEWS_SEARCH_MSG]);
                        B = null;
                        R = null
                    } catch (Q) {}
                    break;
                case 13:
                    try {
                        B = T[O].a;
                        var H = Netdania.JsApi.Response.LookupResponse(B, T[O].t, T[O].i);
                        m.Observer.update(B, "", T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONLOOKUP, [H]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.LOOKUP_MSG]);
                        B = null
                    } catch (Q) {}
                    break;
                case 34:
                    B = T[O].o;
                    var A = B.i;
                    switch (A) {
                    case Netdania.JsApi.Alert.Commands.ADD_ALERT:
                        var D = Netdania.JsApi.Response.AlertAddedResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTADDED, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_ADDED]);
                        break;
                    case Netdania.JsApi.Alert.Commands.DELETE_ALERT:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTDELETE, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_DELETED]);
                        break;
                    case Netdania.JsApi.Alert.Commands.DISCONNECT_USER:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTDISCONNECTMONITORUSER, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_DISCONNECTED_MONITORUSER]);
                        break;
                    case Netdania.JsApi.Alert.Commands.DISCONNECT_USER_ACTIVITIES:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTDISCONNECTMONITORUSERACTIVITY, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_DISCONNECTED_MONITORUSER_ACTIVITY]);
                        break;
                    case Netdania.JsApi.Alert.Commands.EDIT_ALERT:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTEDIT, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_EDIT]);
                        break;
                    case Netdania.JsApi.Alert.Commands.GET_SINGLE_ALERT:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTGET, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_GET]);
                        break;
                    case Netdania.JsApi.Alert.Commands.GET_DELETED_ALERTS:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTGETDELETED, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_GET_DELETED]);
                        break;
                    case Netdania.JsApi.Alert.Commands.GET_TRIGGERED_ALERTS:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTGETTRIGGERED, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_GET_TRIGGERED]);
                        break;
                    case Netdania.JsApi.Alert.Commands.GET_USER_INFORMATION:
                        var D = Netdania.JsApi.Response.AlertActive(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTGETACTIVE, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_GET_ACTIVE]);
                        break;
                    case Netdania.JsApi.Alert.Commands.MONITOR_USER:
                        var D = Netdania.JsApi.Response.AlertActive(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSER, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_MONITORUSER]);
                        break;
                    case Netdania.JsApi.Alert.Commands.MONITOR_USER_ACTIVITIES:
                        var D = Netdania.JsApi.Response.AlertActive(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTDISCONNECTMONITORUSERACTIVITY, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_MONITORUSER_ACTIVITY]);
                        break;
                    case Netdania.JsApi.Alert.Commands.GET_PUSH_DEVICES:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTGETPUSHDEVICES, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.ALERT_GET_PUSH_DEVICES]);
                        break;
                    case Netdania.JsApi.Alert.Commands.NOTIFICATION:
                        var D = Netdania.JsApi.Response.AlertAddedResponse(B, A, T[O].i);
                        var S = m.getFieldValueFromSource(D.data.f, Netdania.JsApi.Alert.Fields.NOTIFICATION_TYPE)[0];
                        switch (S) {
                        case Netdania.JsApi.Alert.NotificationTypes.TRIGGERED_PRICE_ALERT:
                            m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSER, [D]);
                            break;
                        case Netdania.JsApi.Alert.NotificationTypes.ALERT_EDITED:
                            m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSERACTIVITY, [D]);
                            break;
                        case Netdania.JsApi.Alert.NotificationTypes.ALERT_ADDED:
                            m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSERACTIVITY, [D]);
                            break;
                        case Netdania.JsApi.Alert.NotificationTypes.ALERT_DELETED:
                            m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSERACTIVITY, [D]);
                            break;
                        case Netdania.JsApi.Alert.NotificationTypes.PUSH_DEVICES_CHANGED:
                            m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSERACTIVITY, [D]);
                            break
                        }
                        break;
                    case Netdania.JsApi.Alert.Commands.OK:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTDELETE, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTEDIT, [D]);
                        break;
                    case Netdania.JsApi.Alert.Commands.ERROR:
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, A, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONERROR, [D]);
                        break
                    }
                    m.Observer.update(B, "", T[O].i, T[O].t);
                    m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                    B = null;
                    break;
                case 20:
                    try {
                        B = null;
                        m.Observer.update(B, "", T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                        var E = Netdania.JsApi.Response.MonitorChartResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONHISTORICALDATA, [E]);
                        var G = Netdania.JsApi.Response.MonitorPriceResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONPRICEUPDATE, [G]);
                        var U = Netdania.JsApi.Response.ChartUpdateResponse(B, T[O].t, T[O].i, 0);
                        m.fireEvent(Netdania.JsApi.Events.ONCHARTUPDATE, [U]);
                        var K = Netdania.JsApi.Response.NewsHistoryResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONHISTORICALHEADLINES, [K]);
                        var V = Netdania.JsApi.Response.MonitorNewsResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONHEADLINEUPDATE, [V]);
                        var C = Netdania.JsApi.Response.NewsStoryResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONNEWSSTORY, [C]);
                        var M = Netdania.JsApi.Response.NewsSearchResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONNEWSSEARCH, [M]);
                        var H = Netdania.JsApi.Response.LookupResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONLOOKUP, [H]);
                        var F = Netdania.JsApi.Response.IPLocationResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONIPLOCATIONRESPONSE, [F]);
                        var D = Netdania.JsApi.Response.GeneralMonitorResponse(B, T[O].t, T[O].i);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTADDED, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTDELETE, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTEDIT, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSER, [D]);
                        m.fireEvent(Netdania.JsApi.Events.ONALERTMONITORUSERACTIVITY, [D]);
                        B = null;
                        R = null
                    } catch (Q) {}
                    break;
                case 35:
                    try {
                        B = {
                            country_code: T[O].cc,
                            country_name: T[O].cn,
                            region: T[O].rg,
                            city: T[O].ct,
                            ip: ""
                        };
                        var F = Netdania.JsApi.Response.IPLocationResponse(B, T[O].t, T[O].i);
                        m.Observer.update(B, "", T[O].i, T[O].t);
                        m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, "", T[O].i, T[O].t]);
                        m.fireEvent(Netdania.JsApi.Events.ONIPLOCATIONRESPONSE, [F]);
                        m.fireEvent(Netdania.JsApi.Events.ONINFO, [Netdania.JsApi.Messages.IP_LOCATION]);
                        B = null
                    } catch (Q) {}
                    break;
                case 36:
                    c = true;
                    break;
                default:
                    B = T[O].f || T[O].h || T[O].s || T[O].a || T[O].o;
                    R = T[O].rt;
                    var P = Netdania.JsApi.Response.GeneralMonitorResponse(B, T[O].t, T[O].i);
                    m.Observer.update(B, R, T[O].i, T[O].t);
                    m.fireEvent(Netdania.JsApi.Events.ONUPDATE, [B, R, T[O].i, T[O].t]);
                    B = null;
                    R = null
                }
            }
        }
        B = null;
        R = null;
        T = null;
        q = false
    };
    this.getFieldValueFromSource = function(C, A) {
        for (var B = 0; B < C.length; B++) {
            if (C[B].f == A) {
                return C[B].v
            }
        }
        return null
    };
    this.RemoveRequests = function(A) {
        if (A === undefined) {
            return 
        }
        var C = [];
        for (var B = 0; B < A.length; B++) {
            C.push(Netdania.JsApi.Request.getReqObjRemove(A[B]))
        }
        i(C);
        C = null
    };
    this.AddRequest = function(A) {
        if (!f(e, A)) {
            e.push(A);
            r.push(A)
        }
    };
    var i = function(C) {
        if (!t) {
            if (e.length === 0) {
                e = C;
                r = r.concat(C)
            } else {
                var A = [];
                for (var B = 0; B < C.length; B++) {
                    if (w(e, r, C[B])) {
                        continue
                    }
                    if (!f(e, C[B])) {
                        A.push(C[B])
                    }
                }
                e = e.concat(A);
                r = r.concat(A)
            }
            return 
        }
        if (C === undefined || C.length === 0) {
            return 
        }
        g(C)
    };
    var w = function(E, B, C) {
        var D = false;
        if (C.t == 5) {
            for (var A = 0; A < E.length; A++) {
                if (E[A].i === C.i) {
                    E.splice(A, 1);
                    D = true;
                    break
                }
            }
            for (var A = 0; A < B.length; A++) {
                if (B[A].i === C.i) {
                    B.splice(A, 1);
                    if (E.length == 0) {
                        D = true
                    }
                    break
                }
            }
            return D
        }
        return false
    };
    var f = function(C, B) {
        for (var A = 0; A < C.length; A++) {
            if (C[A].i === B.i) {
                return true;
                break
            }
        }
        return false
    };
    this.Flush = function() {
        s = true;
        if (!t) {
            i(e);
            this.Connect();
            return 
        } else {
            i(e);
            e = []
        }
    };
    this.GetRequestList = function() {
        return r
    };
    this.SetPendingRequest = function(A) {
        q = A
    };
    var d = function(A) {
        var D = [];
        for (var B = 0; B < A.length; B++) {
            if (A[B].t == 5) {
                D.push(A[B].i)
            }
        }
        var C = [];
        if (D.length > 0) {
            for (B = 0; B < A.length; B++) {
                if (D.indexOf(A[B].i)==-1) {
                    C.push(A[B])
                }
            }
            A = C
        }
        return A
    };
    var g = function(F) {
        q = true;
        if (F.length > 0) {
            var A = new jXHR();
            var C = F.slice(0, l);
            F = F.slice(l, F.length);
            var G = Netdania.JsApi.encode64(JSON.stringify(C));
            var B = [];
            for (var E = 0; E < C.length; E++) {
                if (w([], r, C[E])) {
                    continue
                }
                if (!f(r, C[E])) {
                    B.push(C[E])
                }
            }
            r = r.concat(B);
            r = d(r);
            var D = m.config.host + "?dt=" + p + "&sessid=" + u + "&xcmd=" + G + "&cb=?";
            A.onerror = function(I, H) {
                m.fireEvent(Netdania.JsApi.Events.ONDISCONNECTED, [I, H]);
                q = false;
                t = false;
                v()
            };
            A.onload = function(H) {
                if (!p) {
                    if (this.responseText == undefined) {
                        return 
                    }
                    H = JSON.parse(this.responseText)
                }
                if (this.readyState === undefined || this.readyState === 4) {
                    j(H)
                }
                clearTimeout(k);
                k = setTimeout(a, 50)
            };
            clearTimeout(k);
            A.open("GET", D, true);
            A.send();
            q = false;
            C = null;
            A = null
        }
        if (F.length > 0) {
            setTimeout(function() {
                g(F)
            }, 50)
        }
        q = false
    };
    this.isConnected = function() {
        return t
    };
    this.appendRequests = function(C) {
        s = true;
        if (!t) {
            if (e.length === 0) {
                e = C;
                r = r.concat(C)
            } else {
                var A = [];
                for (var B = 0; B < C.length; B++) {
                    if (!f(e, C[B])) {
                        A.push(C[B])
                    }
                }
                e = e.concat(A);
                r = r.concat(A)
            }
            return 
        }
        if (C === undefined || C.length === 0) {
            return 
        }
        g(C)
    };
    this.removeRequests = function(A) {
        this.RemoveRequests(A)
    }
};
if (!Array.prototype.forEach) {
    Array.prototype.forEach = function(b) {
        var a = this.length;
        if (typeof b != "function") {
            throw new TypeError()
        }
        var d = arguments[1];
        for (var c = 0; c < a; c++) {
            if (c in this) {
                b.call(d, this[c], c, this)
            }
        }
    }
}
if (!Array.prototype.contains) {
    Array.prototype.contains = function(b) {
        var a = this.length;
        while (a--) {
            if (this[a] === b) {
                return true
            }
        }
        return false
    }
}
if (!Array.prototype.indexOf) {
    Array.prototype.indexOf = function(c) {
        var a = this.length>>>0;
        var b = Number(arguments[1]) || 0;
        b = (b < 0) ? Math.ceil(b) : Math.floor(b);
        if (b < 0) {
            b += a
        }
        for (; b < a; b++) {
            if (b in this && this[b] === c) {
                return b
            }
        }
        return - 1
    }
}
Object.keys = Object.keys || (function() {
    var b = Object.prototype.hasOwnProperty, d=!{
        toString: null
    }.propertyIsEnumerable("toString"), a = ["toString", "toLocaleString", "valueOf", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable", "constructor"], c = a.length;
    return function(h) {
        if (typeof h != "object" && typeof h != "function" || h === null) {
            throw new TypeError("Object.keys called on a non-object")
        }
        var e = [];
        for (var f in h) {
            if (b.call(h, f)) {
                e.push(f)
            }
        }
        if (d) {
            for (var g = 0; g < c; g++) {
                if (b.call(h, a[g])) {
                    e.push(a[g])
                }
            }
        }
        return e
    }
})();
Netdania.UpdatesObserver = function() {
    this.components = []
};
var cachedDT = null;
function getRandomInt(b, a) {
    return Math.floor(Math.random() * (a - b + 1)) + b
}
Netdania.UpdatesObserver.prototype = {
    subscribe: function(a) {
        this.components.push(a)
    },
    unsubscribe: function(a) {
        this.components = this.components.filter(function(b) {
            if (b !== a) {
                return b
            }
        })
    },
    update: function(c, a, e, d) {
        var b = d || window;
        this.components.forEach(function(f) {
            if (f.ids.indexOf(e)>-1) {
                f.update(c, a, e)
            }
        })
    },
    init: function(b, d, c) {
        var a = c || window;
        this.components.forEach(function(e) {
            if (e.ids.indexOf(d)>-1) {
                e.init(b, d)
            }
        })
    }
};
var Events = {};
(function() {
    Events = {
        enable: function() {
            var a = this;
            a.listeners = {};
            a.fireEvent = function(c, b) {
                Events.fireEvent.call(a, c, b)
            };
            a.addListener = function(c, b) {
                Events.addListener.call(a, c, b)
            };
            a.removeListener = function(c, b) {
                Events.removeListener.call(a, c, b)
            }
        },
        fireEvent: function(d, a) {
            if (!!this.listeners[d]) {
                for (var b = 0; b < this.listeners[d].length; b++) {
                    var c = this.listeners[d][b];
                    c.apply(window, a)
                }
            }
        },
        addListener: function(b, a) {
            if (!this.listeners) {
                Events.enable.call(this, b)
            }
            if (!this.listeners[b]) {
                this.listeners[b] = []
            }
            if (a instanceof Function) {
                this.listeners[b].push(a)
            }
        },
        removeListener: function(c, b) {
            if (!!this.listeners[c] && this.listeners[c].length > 0) {
                if (!!b) {
                    var d = [];
                    for (var a = 0; a < this.listeners[c].length; a++) {
                        if (this.listeners[c][a] != b) {
                            d.push(this.listeners[c][a])
                        }
                    }
                    this.listeners[c] = d
                } else {
                    this.listeners[c] = []
                }
            }
        }
    }
}());
Netdania.JsApi.Fields = {
    ALLOWED_CHART_TIME_SCALES: [0, 1, 5, 10, 15, 30, 60, 120, 240, 480, 1440, 10080, 43200],
    CHART_TIME_STAMP: 100,
    CHART_OPEN: 101,
    CHART_HIGH: 102,
    CHART_LOW: 103,
    CHART_CLOSE: 104,
    CHART_VOLUME: 105,
    CHART_OPEN_INT: 106,
    CHART_BUYER_ID: 107,
    CHART_SELLER_ID: 108,
    TYPE_NUMERIC: 1,
    TYPE_STRING: 2,
    QUOTE_LAST: 6,
    QUOTE_OPEN_INT: 7,
    QUOTE_VOLUME_INC: 8,
    QUOTE_BID: 10,
    QUOTE_TIME_STAMP: 17,
    QUOTE_ASK: 11,
    QUOTE_MID_PRICE: 9,
    QUOTE_HIGH: 2,
    QUOTE_LOW: 3,
    QUOTE_OPEN: 4,
    QUOTE_CLOSE: 1,
    QUOTE_VOLUME: 5,
    QUOTE_CONTRIBUTOR: 23,
    QUOTE_BID_SIZE: 12,
    QUOTE_ASK_SIZE: 13,
    QUOTE_PRV_VOLUME: 18,
    QUOTE_SETTLE_PRICE: 20,
    QUOTE_DIVIDEND: 26,
    QUOTE_NAME: 25,
    QUOTE_AVG_PRICE: 16,
    QUOTE_EARN_PER_SHARE: 24,
    QUOTE_ISIN_CODE: 39,
    QUOTE_EQUITY_PER_SHARE: 40,
    QUOTE_SALES_PER_SHARE: 41,
    QUOTE_TOTAL_SHARES: 42,
    QUOTE_52W_HIGH: 21,
    QUOTE_52W_LOW: 22,
    QUOTE_YEAR_HIGH: 43,
    QUOTE_YEAR_LOW: 44,
    QUOTE_1W_HIGH: 120,
    QUOTE_1W_LOW: 121,
    QUOTE_1MONTH_HIGH: 122,
    QUOTE_1MONTH_LOW: 123,
    QUOTE_3MONTH_HIGH: 124,
    QUOTE_3MONTH_LOW: 125,
    QUOTE_6MONTH_HIGH: 126,
    QUOTE_6MONTH_LOW: 127,
    QUOTE_PRV_YEAR_CLOSE: 19,
    QUOTE_1WEEK_CLOSE: 27,
    QUOTE_1MONTH_CLOSE: 28,
    QUOTE_3MONTH_CLOSE: 29,
    QUOTE_1YEAR_CLOSE: 30,
    QUOTE_6MONTH_CLOSE: 117,
    QUOTE_EARN_PER_SHARE_EST: 97,
    QUOTE_BETA: 98,
    QUOTE_YIELD: 99,
    QUOTE_DEBT_TO_EQUITY: 100,
    QUOTE_INSTRUMENT_TYPE: 107,
    QUOTE_INDUSTRY_CODE: 108,
    QUOTE_INDUSTRY_NAME: 109,
    QUOTE_BUYER_ID: 110,
    QUOTE_SELLER_ID: 111,
    QUOTE_BOARD_LOT: 113,
    QUOTE_MARKET_ID: 115,
    QUOTE_CURRENCY: 116,
    _QUOTE_CHANGE: 14,
    _QUOTE_PERCENT_CHANGE: 15,
    _QUOTE_YEAR_CHANGE: 31,
    _QUOTE_YEAR_PERCENT_CHANGE: 32,
    _QUOTE_1WEEK_CHANGE: 33,
    _QUOTE_1WEEK_PERCENT_CHANGE: 34,
    _QUOTE_1MONTH_CHANGE: 35,
    _QUOTE_1MONTH_PERCENT_CHANGE: 36,
    _QUOTE_3MONTH_CHANGE: 45,
    _QUOTE_3MONTH_PERCENT_CHANGE: 46,
    _QUOTE_1YEAR_CHANGE: 37,
    _QUOTE_1YEAR_PERCENT_CHANGE: 38,
    _QUOTE_6MONTH_CHANGE: 118,
    _QUOTE_6MONTH_PERCENT_CHANGE: 119,
    _QUOTE_PRICE_PER_EARN: 101,
    _QUOTE_PRICE_PER_EARN_EST: 102,
    _QUOTE_EARN_PER_PRICE_EST: 103,
    _QUOTE_AMOUNT_TURNOVR: 104,
    _QUOTE_SYMBOL: 105,
    _QUOTE_LAST_BID_ASK: 106,
    _QUOTE_MARKET_CAP: 112,
    _QUOTE_BOARD_LOT_VALUE: 114,
    QUOTE_BID_ON: 47,
    QUOTE_ASK_ON: 48,
    QUOTE_BID_SN: 49,
    QUOTE_ASK_SN: 50,
    QUOTE_BID_TN: 51,
    QUOTE_ASK_TN: 52,
    QUOTE_BID_1W: 53,
    QUOTE_ASK_1W: 54,
    QUOTE_BID_2W: 55,
    QUOTE_ASK_2W: 56,
    QUOTE_BID_3W: 57,
    QUOTE_ASK_3W: 58,
    QUOTE_BID_1M: 59,
    QUOTE_ASK_1M: 60,
    QUOTE_BID_2M: 61,
    QUOTE_ASK_2M: 62,
    QUOTE_BID_3M: 63,
    QUOTE_ASK_3M: 64,
    QUOTE_BID_4M: 65,
    QUOTE_ASK_4M: 66,
    QUOTE_BID_5M: 67,
    QUOTE_ASK_5M: 68,
    QUOTE_BID_6M: 69,
    QUOTE_ASK_6M: 70,
    QUOTE_BID_7M: 71,
    QUOTE_ASK_7M: 72,
    QUOTE_BID_8M: 73,
    QUOTE_ASK_8M: 74,
    QUOTE_BID_9M: 75,
    QUOTE_ASK_9M: 76,
    QUOTE_BID_10M: 77,
    QUOTE_ASK_10M: 78,
    QUOTE_BID_11M: 79,
    QUOTE_ASK_11M: 80,
    QUOTE_BID_1Y: 81,
    QUOTE_ASK_1Y: 82,
    QUOTE_BID_2Y: 83,
    QUOTE_ASK_2Y: 84,
    QUOTE_BID_3Y: 85,
    QUOTE_ASK_3Y: 86,
    QUOTE_BID_4Y: 87,
    QUOTE_ASK_4Y: 88,
    QUOTE_BID_5Y: 89,
    QUOTE_ASK_5Y: 90,
    QUOTE_BID_6Y: 91,
    QUOTE_ASK_6Y: 92,
    QUOTE_BID_7Y: 93,
    QUOTE_ASK_7Y: 94,
    QUOTE_BID_10Y: 95,
    QUOTE_ASK_10Y: 96,
    QUOTE_DECIMALS: 137,
    QUOTE_UNITS: 138
};
Netdania.JsApi.Alert.Commands = {
    BIND_PUSH_DEVICE: 1,
    UNBIND_PUSH_DEVICE: 2,
    GET_PUSH_DEVICES: 3,
    GET_PUSH_BINDINGS: 4,
    SET_PUSH_BADGE: 5,
    ADD_ALERT: 6,
    GET_USER_INFORMATION: 7,
    USER_DETAILS: 8,
    ALERT: 9,
    USER_ALERTS: 10,
    GET_SINGLE_ALERT: 11,
    GET_TRIGGERED_ALERTS: 12,
    GET_DELETED_ALERTS: 13,
    EDIT_ALERT: 14,
    MONITOR_USER: 15,
    NEWS: 16,
    GET_UNREAD_NEWS: 17,
    MONITOR_USER_GROUP: 18,
    MARK_UNREAD_NEWS: 19,
    SEND_PUSH_MESSAGE: 20,
    MARK_UNREAD_PUSH_ALERTS: 21,
    GET_PUSH_ALERT_INFO: 22,
    SEND_NEWS_FLASH: 23,
    ADD_USER: 24,
    NOTIFICATION: 25,
    MONITOR_USER_ACTIVITIES: 26,
    DISCONNECT_USER_ACTIVITIES: 27,
    DELETE_USER: 28,
    DELETE_ALERT: 29,
    GET_USERGROUP_USERS: 30,
    DELETE_ALERT_LIST: 31,
    MOVE_ALERTS: 32,
    UPDATE_USER: 33,
    DISCONNECT_USER: 34,
    GET_SENT_MESSAGES: 35,
    ERROR: 255,
    OK: 0
};
Netdania.JsApi.Alert.FieldTypes = {
    ALERT_ID: 8,
    USER_ID: 2,
    PREVIOUS_OWNER: 2,
    ALERT_CONDITION: 2,
    DAYS_TO_LIVE: 7,
    SMS_PHONE_NUMBER: 2,
    SMS_FROM_NAME: 2,
    SMS_MESSAGE: 2,
    EMAIL_ADDRESS: 2,
    EMAIL_FROM: 2,
    EMAIL_SUBJECT: 2,
    EMAIL_CONTENT: 2,
    YAHOO_ID: 2,
    YAHOO_MESSAGE: 2,
    SKYPE_ID: 2,
    SKYPE_MESSAGE: 2,
    MSN_ID: 2,
    MSN_MESSAGE: 2,
    TRIGGER_DATE: 8,
    ORDER_TYPE: 2,
    PUSH_DEVICES: 2,
    PUSH_DEVICES_ID: 2,
    PUSH_MESSAGE: 2,
    DELETION_REASON: 2
};
Netdania.JsApi.Alert.Fields = {
    ERROR_CODE: 0,
    USER_GROUP: 1,
    USER_ID: 2,
    STARTDATE: 3,
    ENDDATE: 4,
    ALERT_ID: 5,
    PREVIOUS_OWNER: 6,
    ALERT_CONDITION: 7,
    DAYS_TO_LIVE: 8,
    EXPIRATION_DATE: 9,
    SMS_PHONE_NUMBER: 10,
    SMS_FROM_NAME: 11,
    SMS_MESSAGE: 12,
    EMAIL_ADDRESS: 13,
    EMAIL_FROM: 14,
    EMAIL_SUBJECT: 15,
    EMAIL_CONTENT: 16,
    YAHOO_ID: 17,
    YAHOO_MESSAGE: 18,
    SKYPE_ID: 19,
    SKYPE_MESSAGE: 20,
    MSN_ID: 21,
    MSN_MESSAGE: 22,
    TRIGGER_DATE: 23,
    USER_INFO: 24,
    MAX_ALERTS: 25,
    CURRENT_ALERTS_NO: 26,
    DELETION_DATE: 27,
    DELETION_REASON: 28,
    PUSH_DEVICES_ACTION: 29,
    PUSH_DEVICES_TOKEN: 30,
    PUSH_DEVICES_TYPE: 31,
    PUSH_DEVICES_ID: 32,
    PUSH_BADGE_VALUE: 33,
    PUSH_DEVICES_ARRAY: 34,
    PUSH_DEVICES: 35,
    ORDER_TYPE: 36,
    IS_ALERT_CENTRAL: 37,
    ALERTS_INCLUDED: 38,
    ALERTS_HOLDER: 39,
    USER_ALERTS: 40,
    USER_DETAILS_INCLUDED: 41,
    USER_HOLDER: 42,
    PUSH_DEVICES_NAME: 43,
    DATA_HOLDER: 44,
    MONITOR_HOLDER: 45,
    MONITOR_USER: 46,
    MONITOR_USER_ACTIVITIES: 47,
    TIMESTAMP: 48,
    STORY_ID: 49,
    HEADLINE: 50,
    NEWS_HOLDER: 51,
    MONITOR_USERS: 52,
    SOURCE_PROVIDER: 53,
    NEWS_ID: 54,
    MARKED_NEWS_NO: 55,
    PUSH_APP_NAME: 56,
    PUSH_MESSAGE: 57,
    PUSH_MESSAGE_TYPE: 58,
    INSTRUMENT: 59,
    PROVIDER: 60,
    INSTRUMENT_NAME: 61,
    TRIGGER_LEVEL: 62,
    OPERATOR: 63,
    ALERT_LEVEL: 64,
    NEWS_INCLUDED: 65,
    MARKED_ALERT_NO: 66,
    REAL_USER_GROUP: 67,
    PUSH_DEVICE_STATE: 68,
    DELETE_BIND: 69,
    REQUEST_ID: 70,
    COMPRESSION: 71,
    MONITOR: 72,
    NOTIFICATION_TYPE: 73,
    NEW_CONTACTS: 74,
    MOVE_SOURCE_DETAILS: 75,
    CONTACTS: 76,
    MOVED_ALERT_IDS: 77,
    NEWS_FIELD: 78,
    TRIGGER_VALUE: 79,
    SENT_DATE: 80,
    MAIL_HOLDER: 81,
    SMS_HOLDER: 82,
    DESTINATION_ID: 83,
    MESSAGE: 84,
    MESSAGE_HOLDER: 85,
    COOKIE: 86,
    HEARTBEAT_COUNTER: 87,
    ALIAS: 88,
    ACCOUNT_ID: 89,
    TRADING_EVENT_SUBSCRIPTION_ID: 90,
    TOKEN_VERSION: 91,
    TRADING_EVENTS_SUBSCRIBE: 92,
    PUSH_LAST_ACTION_TIMESTAMP: 93,
    PUSH_SOURCES: 94,
    START_INDEX: 95,
    NO_OFROWS: 96,
    PUSH_DEVICENAME: 97,
    DELETE_DEVICES: 98,
    PUSH_COUNTRY: 99,
    TRADE_AMOUNT: 100,
    TRADE_LEVEL: 101,
    TRADE_STATUS: 102,
    TRADE_INCLUDED: 103,
    TRADES_HOLDER: 104,
    TIMESCALE_SECONDS: 105
};
Netdania.JsApi.Alert.ObjectType = {
    TYPE_OBJECT_END: 0,
    TYPE_OBJECT: 1,
    _TYPE_STRING: 2,
    TYPE_BOOLEAN: 3,
    TYPE_BYTE: 4,
    TYPE_SHORT: 5,
    TYPE_CHAR: 6,
    TYPE_INT: 7,
    TYPE_LONG: 8,
    TYPE_FLOAT: 9,
    TYPE_DOUBLE: 10
};
Netdania.JsApi.Alert.NotificationTypes = {
    ALERT_ADDED: 1,
    ALERT_DELETED: 2,
    ALERT_EDITED: 3,
    TRIGGERED_NEWS_ALERT: 4,
    TRIGGERED_PRICE_ALERT: 5,
    ALERT_MOVED_FROM: 6,
    ALERT_MOVED_TO: 7,
    ALERT_USER_DELETED: 8,
    ALERT_USER_ADDED: 9,
    PUSH_DEVICES_CHANGED: 10
};
Netdania.JsApi.Alert.ErrorCodes = {
    GENERAL_ERROR: 0,
    ADD_ALERT_USER_NOT_FOUND_ERROR: 1,
    ADD_ALERT_INCORRECT_CONDITION_ERROR: 2,
    ADD_ALERT_ENGINE_ADD_ALERT_ERROR: 3,
    ADD_ALERT_MAX_ALERTS_EXCEEDED_ERROR: 4,
    ADD_ALERT_GENERAL_ERROR: 5,
    USER_ADD_GENERAL_ERROR: 6,
    USER_DELETE_GENERAL_ERROR: 7,
    USER_UPDATE_GENERAL_ERROR: 8,
    GET_USER_INFO_USER_NOT_FOUND_ERROR: 9,
    GET_USER_INFO_GENERAL_ERROR: 10,
    GET_TRIGGERED_ALERTS_GENERAL_ERROR: 11,
    GET_DELETED_ALERTS_GENERAL_ERROR: 12,
    GET_SENT_MESSAGES_GENERAL_ERROR: 13,
    DELETE_ALERT_USER_NOT_FOUND_ERROR: 14,
    DELETE_ALERT_ALERT_ID_NOT_FOUND_ERROR: 15,
    DELETE_ALERT_GENERAL_ERROR: 16,
    EDIT_ALERT_NO_NEW_ALERT_ERROR: 17,
    EDIT_ALERT_ALERT_ID_NOT_FOUND_ERROR: 18,
    EDIT_ALERT_USER_NOT_FOUND_ERROR: 19,
    EDIT_ALERT_MAX_ALERTS_VIOLATION_ERROR: 20,
    EDIT_ALERT_ENGINE_ADD_ALERT_ERROR: 21,
    EDIT_ALERT_GENERAL_ERROR: 22,
    GET_USER_GROUP_USERS_GENERAL_ERROR: 23,
    TRIGGER_ALERT_ALERT_ID_NOT_FOUND_ERROR: 24,
    TRIGGER_ALERT_USER_NOT_FOUND_ERROR: 25,
    TRIGGER_ALERT_GENERAL_ERROR: 26,
    GET_SINGLE_ALERT_GENERAL_ERROR: 27,
    GET_SINGLE_ALERT_ID_NOT_FOUND: 28,
    GET_USERGROUP_ALERTS_GENERAL_ERROR: 29,
    MOVE_ALERTS_GENERAL_ERROR: 30,
    MOVE_ALERTS_DESTINATION_USER_ERROR: 31,
    MOVE_ALERTS_INVALID_GROUP_LENGTH_ERROR: 32
};
