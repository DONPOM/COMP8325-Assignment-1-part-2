/* JS */
gapi.loaded_0(function(_) {
    var window = this;
    var oa, qa;
    _.q = function(a) {
        return function() {
            return _.aa[a].apply(this, arguments)
        }
    };
    var _DumpException = function(a) {
        throw a;
    };
    _.aa = [];
    _.ba = _.ba || {};
    _.r = this;
    _.ca = function(a) {
        return void 0 !== a
    };
    _.fa = function(a) {
        var b = typeof a;
        if ("object" == b)
            if (a) {
                if (a instanceof Array)
                    return "array";
                    if (a instanceof Object)
                        return b;
                        var c = Object.prototype.toString.call(a);
                        if ("[object Window]" == c)
                            return "object";
                            if ("[object Array]" == c || "number" == typeof a.length && "undefined" != typeof a.splice && "undefined" != typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("splice"))
                                return "array";
                                if ("[object Function]" == c || "undefined" != typeof a.call && "undefined" != typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("call"))
                                    return "function"
            } else 
                return "null";
                else if ("function" == b && "undefined" == typeof a.call)
    return "object";
return b
};
_.ha = function(a) {
    return "array" == _.fa(a)
};
_.ja = function(a) {
    return "string" == typeof a
};
_.na = "closure_uid_" + (1E9 * Math.random()>>>0);
oa = function(a, b, c) {
    return a.call.apply(a.bind, arguments)
};
qa = function(a, b, c) {
    if (!a)
        throw Error();
    if (2 < arguments.length) {
        var d = Array.prototype.slice.call(arguments, 2);
        return function() {
            var c = Array.prototype.slice.call(arguments);
            Array.prototype.unshift.apply(c, d);
            return a.apply(b, c)
        }
    }
    return function() {
        return a.apply(b, arguments)
    }
};
_.s = function(a, b, c) {
    _.s = Function.prototype.bind&&-1 != Function.prototype.bind.toString().indexOf("native code") ? oa : qa;
    return _.s.apply(null, arguments)
};
_.ra = function(a, b) {
    var c = Array.prototype.slice.call(arguments, 1);
    return function() {
        var b = c.slice();
        b.push.apply(b, arguments);
        return a.apply(this, b)
    }
};
_.sa = Date.now || function() {
    return + new Date
};
_.u = function(a, b) {
    function c() {}
    c.prototype = b.prototype;
    a.J = b.prototype;
    a.prototype = new c;
    a.prototype.constructor = a;
    a.nu = function(a, c, f) {
        return b.prototype[c].apply(a, Array.prototype.slice.call(arguments, 2))
    }
};
_.wa = window.gadgets || {};
_.ya = window.osapi = window.osapi || {};
_.google = window.google || {};
window.___jsl = window.___jsl || {};
(window.___jsl.cd = window.___jsl.cd || []).push({
    gwidget: {
        parsetags: "explicit"
    },
    appsapi: {
        plus_one_service: "/plus/v1"
    },
    client: {
        jsonpOverride: !1,
        rms: "migrated"
    },
    csi: {
        rate: .01
    },
    poshare: {
        hangoutContactPickerServer: "https://plus.google.com"
    },
    gappsutil: {
        required_scopes: ["https://www.googleapis.com/auth/plus.me", "https://www.googleapis.com/auth/plus.people.recommended"],
        display_on_page_ready: !1
    },
    appsutil: {
        required_scopes: ["https://www.googleapis.com/auth/plus.me", "https://www.googleapis.com/auth/plus.people.recommended"],
        display_on_page_ready: !1
    },
    "oauth-flow": {
        authUrl: "https://accounts.google.com/o/oauth2/auth",
        proxyUrl: "https://accounts.google.com/o/oauth2/postmessageRelay",
        redirectUri: "postmessage"
    },
    iframes: {
        sharebox: {
            params: {
                json: "&"
            },
            url: ":socialhost:/:session_prefix:_/sharebox/dialog"
        },
        plus: {
            url: ":socialhost:/:session_prefix:_/widget/render/badge?usegapi=1"
        },
        ":socialhost:": "https://apis.google.com",
        ":im_socialhost:": "https://plus.googleapis.com",
        domains_suggest: {
            url: "https://domains.google.com/suggest/flow"
        },
        card: {
            params: {
                s: "#",
                userid: "&"
            },
            url: ":socialhost:/:session_prefix:_/hovercard/internalcard"
        },
        ":signuphost:": "https://plus.google.com",
        ":gplus_url:": "https://plus.google.com",
        plusone: {
            url: ":socialhost:/:session_prefix:_/+1/fastbutton?usegapi=1"
        },
        plus_share: {
            url: ":socialhost:/:session_prefix:_/+1/sharebutton?plusShare=true&usegapi=1"
        },
        plus_circle: {
            url: ":socialhost:/:session_prefix:_/widget/plus/circle?usegapi=1"
        },
        plus_followers: {
            url: ":socialhost:/_/im/_/widget/render/plus/followers?usegapi=1"
        },
        configurator: {
            url: ":socialhost:/:session_prefix:_/plusbuttonconfigurator?usegapi=1"
        },
        appcirclepicker: {
            url: ":socialhost:/:session_prefix:_/widget/render/appcirclepicker"
        },
        page: {
            url: ":socialhost:/:session_prefix:_/widget/render/page?usegapi=1"
        },
        person: {
            url: ":socialhost:/:session_prefix:_/widget/render/person?usegapi=1"
        },
        community: {
            url: ":ctx_socialhost:/:session_prefix::im_prefix:_/widget/render/community?usegapi=1"
        },
        follow: {
            url: ":socialhost:/:session_prefix:_/widget/render/follow?usegapi=1"
        },
        commentcount: {
            url: ":socialhost:/:session_prefix:_/widget/render/commentcount?usegapi=1"
        },
        comments: {
            url: ":socialhost:/:session_prefix:_/widget/render/comments?usegapi=1"
        },
        youtube: {
            url: ":socialhost:/:session_prefix:_/widget/render/youtube?usegapi=1"
        },
        reportabuse: {
            url: ":socialhost:/:session_prefix:_/widget/render/reportabuse?usegapi=1"
        },
        additnow: {
            url: ":socialhost:/additnow/additnow.html"
        },
        ":source:": "1p"
    },
    poclient: {
        update_session: "google.updateSessionCallback"
    },
    "googleapis.config": {
        methods: {
            "pos.plusones.list": !0,
            "pos.plusones.get": !0,
            "pos.plusones.insert": !0,
            "pos.plusones.delete": !0,
            "pos.plusones.getSignupState": !0
        },
        requestCache: {
            enabled: !0
        },
        versions: {
            pos: "v1"
        },
        rpc: "/rpc",
        root: "https://content.googleapis.com",
        "root-1p": "https://clients6.google.com",
        sessionCache: {
            enabled: !0
        },
        transport: {
            isProxyShared: !0
        },
        xd3: "/static/proxy.html",
        developerKey: "AIzaSyCKSbrvQasunBoV16zDH9R33D88CeLr9gQ",
        auth: {
            useInterimAuth: !1
        }
    },
    report: {
        apis: ["iframes\\..*", "gadgets\\..*", "gapi\\.appcirclepicker\\..*", "gapi\\.client\\..*"],
        rate: 1E-4
    }
});

_.za = function(a, b) {
    return b
};
_.B = function(a, b) {
    var c = a.split("."), d = _.r;
    c[0]in d ||!d.execScript || d.execScript("var " + c[0]);
    for (var e; c.length && (e = c.shift());)
        !c.length && _.ca(b) ? d[e] = b : d[e] ? d = d[e] : d = d[e] = {}
};
_.C = function(a, b) {
    b = _.za(a, b);
    _.B(a, b)
};
_.Aa = window;
_.Ba = window.document;
_.Ea = _.Aa.location;
_.Ga = /\[native code\]/;
_.Ha = function(a, b, c) {
    return a[b] = a[b] || c
};
_.Ka = function() {
    var a;
    if ((a = Object.create) && _.Ga.test(a))
        a = a(null);
    else {
        a = {};
        for (var b in a)
            a[b] = void 0
    }
    return a
};
_.Ma = function(a, b) {
    return Object.prototype.hasOwnProperty.call(a, b)
};
_.Na = function(a, b) {
    if (!a)
        throw Error(b || "");
};
_.Sa = _.Ha(_.Aa, "gapi", {});
_.Ua = function(a, b, c) {
    var d = new RegExp("([#].*&|[#])" + b + "=([^&#]*)", "g");
    b = new RegExp("([?#].*&|[?#])" + b + "=([^&#]*)", "g");
    if (a = a && (d.exec(a) || b.exec(a)))
        try {
            c = (0, window.decodeURIComponent)(a[2])
    } catch (e) {}
    return c
};
_.Wa = function(a, b, c) {
    _.Va(a, b, c, "add", "at")
};
_.Va = function(a, b, c, d, e) {
    if (a[d + "EventListener"])
        a[d + "EventListener"](b, c, !1);
    else if (a[e + "tachEvent"])
        a[e + "tachEvent"]("on" + b, c)
};
_.Ya = _.Ha(_.Aa, "___jsl", _.Ka());
_.Ha(_.Ya, "I", 0);
_.Ha(_.Ya, "hel", 10);
var Za, $a, ab, bb, cb, db;
Za = function(a) {
    var b = window.___jsl = window.___jsl || {};
    b[a] = b[a] || [];
    return b[a]
};
$a = function(a) {
    var b = window.___jsl = window.___jsl || {};
    b.cfg=!a && b.cfg || {};
    return b.cfg
};
ab = function(a) {
    return "object" === typeof a && /\[native code\]/.test(a.push)
};
bb = function(a, b) {
    if (b)
        for (var c in b)
            b.hasOwnProperty(c) && (a[c] && b[c] && "object" === typeof a[c] && "object" === typeof b[c]&&!ab(a[c])&&!ab(b[c]) ? bb(a[c], b[c]) : b[c] && "object" === typeof b[c] ? (a[c] = ab(b[c]) ? [] : {}, bb(a[c], b[c])) : a[c] = b[c])
};
cb = function(a) {
    if (a&&!/^\s+$/.test(a)) {
        for (; 0 == a.charCodeAt(a.length - 1);)
            a = a.substring(0, a.length - 1);
        var b;
        try {
            b = window.JSON.parse(a)
        } catch (c) {}
        if ("object" === typeof b)
            return b;
        try {
            b = (new Function("return (" + a + "\n)"))()
        } catch (d) {}
        if ("object" === typeof b)
            return b;
        try {
            b = (new Function("return ({" + a + "\n})"))()
        } catch (e) {}
        return "object" === typeof b ? b : {}
    }
};
db = function(a) {
    $a(!0);
    var b = window.___gcfg, c = Za("cu");
    if (b && b !== window.___gu) {
        var d = {};
        bb(d, b);
        c.push(d);
        window.___gu = b
    }
    var b = Za("cu"), e = window.document.scripts || window.document.getElementsByTagName("script") || [], d = [], f = [];
    f.push.apply(f, Za("us"));
    for (var g = 0; g < e.length; ++g)
        for (var k = e[g], l = 0; l < f.length; ++l)
            k.src && 0 == k.src.indexOf(f[l]) && d.push(k);
    0 == d.length && 0 < e.length && e[e.length - 1].src && d.push(e[e.length - 1]);
    for (e = 0; e < d.length; ++e)
        d[e].getAttribute("gapi_processed") || (d[e].setAttribute("gapi_processed",
        !0), (f = d[e]) ? (g = f.nodeType, f = 3 == g || 4 == g ? f.nodeValue : f.textContent || f.innerText || f.innerHTML || "") : f = void 0, (f = cb(f)) && b.push(f));
    a && (d = {}, bb(d, a), c.push(d));
    d = Za("cd");
    a = 0;
    for (b = d.length; a < b; ++a)
        bb($a(), d[a]);
    d = Za("ci");
    a = 0;
    for (b = d.length; a < b; ++a)
        bb($a(), d[a]);
    a = 0;
    for (b = c.length; a < b; ++a)
        bb($a(), c[a])
};
_.E = function(a, b) {
    if (!a)
        return $a();
    for (var c = a.split("/"), d = $a(), e = 0, f = c.length; d && "object" === typeof d && e < f; ++e)
        d = d[c[e]];
    return e === c.length && void 0 !== d ? d : b
};
_.eb = function(a, b) {
    var c = a;
    if ("string" === typeof a) {
        for (var d = c = {}, e = a.split("/"), f = 0, g = e.length; f < g - 1; ++f)
            var k = {}, d = d[e[f]] = k;
        d[e[f]] = b
    }
    db(c)
};
var fb = function() {
    var a = window.__GOOGLEAPIS;
    a && (a.googleapis&&!a["googleapis.config"] && (a["googleapis.config"] = a.googleapis), _.Ha(_.Ya, "ci", []).push(a), window.__GOOGLEAPIS = void 0)
};
fb && fb();
db();
_.C("gapi.config.get", _.E);
_.C("gapi.config.update", _.eb);
var jb, lb, nb, ob, qb, rb, sb, ub, vb, wb, xb, zb, Db, Eb, Hb;
_.gb = function(a) {
    return !!a && "object" === typeof a && _.Ga.test(a.push)
};
_.hb = function(a, b, c) {
    if (a) {
        _.Na(_.gb(a), "arrayForEach was called with a non array value");
        for (var d = 0; d < a.length; d++)
            b.call(c, a[d], d)
    }
};
_.ib = function(a, b, c) {
    if (a)
        if (_.gb(a))
            _.hb(a, b, c);
        else {
            _.Na("object" === typeof a, "objectForEach was called with a non object value");
            c = c || a;
            for (var d in a)
                _.Ma(a, d) && void 0 !== a[d] && b.call(c, a[d], d)
        }
};
jb = function(a) {
    a = a.sort();
    for (var b = [], c = void 0, d = 0; d < a.length; d++) {
        var e = a[d];
        e != c && b.push(e);
        c = e
    }
    return b
};
lb = function() {
    var a = [], b = _.Ya.H;
    b && _.ib(b, function(b) {
        a.push.apply(a, b.L)
    });
    return jb(a)
};
_.mb = function(a) {
    if (_.Ga.test(Object.keys))
        return Object.keys(a);
    var b = [], c;
    for (c in a)
        _.Ma(a, c) && b.push(c);
    return b
};
ub = {};
vb = 0;
wb = _.Ka();
xb = _.Ka();
zb = function(a) {
    return "number" === typeof a && a > Math.random()
};
_.Ab = function(a) {
    if ("undefined" === typeof nb) {
        var b = _.E("report") || {}, c = b.rate;
        rb = b.timeout || 1E3;
        ob = b.host || "https://plus.google.com";
        qb = b.path || "/_/widget/report";
        nb = [];
        zb(c) && (nb = b.apis || []);
        var b = b.apiRate || {}, d;
        for (d in b)
            zb(b[d]) && nb.push(d)
    }
    for (d = 0; d < nb.length; ++d)
        if ((new RegExp("^" + nb[d] + "$")).test(a))
            return !0;
    return !1
};
Db = function(a) {
    delete ub[a]
};
Eb = function() {
    sb && (_.Aa.clearTimeout(sb), sb = 0);
    sb = _.Aa.setTimeout(function() {
        var a;
        a = window.document.location;
        a = a.protocol + "//" + a.host + a.pathname;
        var b = _.mb(xb).join(":");
        a = [ob, qb, "?api=", (0, window.encodeURIComponent)(b), "&url=", (0, window.encodeURIComponent)(a), "&loaded=", (0, window.encodeURIComponent)(lb().join(":"))].join("");
        xb = _.Ka();
        var b = new window.Image, c = vb++;
        ub[c] = b;
        b.onload = b.onerror = _.ra(Db, c);
        b.src = a;
        sb = 0
    }, rb)
};
_.Gb = function(a) {
    wb[a] || (xb[a]=!0, wb[a]=!0, Eb())
};
Hb = _.za;
_.za = function(a, b) {
    var c = Hb(a, b);
    "function" === typeof b && _.Ab(a) && (c = function(c) {
        _.Gb(a);
        return b.apply(this, arguments)
    });
    return c
};

_.K = {};
_.Pc = {};
window.iframer = _.Pc;
var Rc;
_.Qc = function(a) {
    var b = _.fa(a);
    return "array" == b || "object" == b && "number" == typeof a.length
};
Rc = 0;
_.Sc = function(a) {
    return a[_.na] || (a[_.na]=++Rc)
};
_.Tc = function(a) {
    var b = typeof a;
    return "object" == b && null != a || "function" == b
};
_.Uc = function(a) {
    return "function" == _.fa(a)
};
_.Vc = function(a) {
    return "number" == typeof a
};
_.Wc = function(a, b) {
    return 0 == a.lastIndexOf(b, 0)
};
_.Xc = String.prototype.trim ? function(a) {
    return a.trim()
} : function(a) {
    return a.replace(/^[\s\xa0]+|[\s\xa0]+$/g, "")
};
_.Yc = function(a, b) {
    return Array(b + 1).join(a)
};
_.Zc = 2147483648 * Math.random() | 0;
_.$c = function(a) {
    return String(a).replace(/\-([a-z])/g, function(a, c) {
        return c.toUpperCase()
    })
};
_.ad = Array.prototype;
_.bd = _.ad.indexOf ? function(a, b, c) {
    return _.ad.indexOf.call(a, b, c)
} : function(a, b, c) {
    c = null == c ? 0 : 0 > c ? Math.max(0, a.length + c) : c;
    if (_.ja(a))
        return _.ja(b) && 1 == b.length ? a.indexOf(b, c) : - 1;
    for (; c < a.length; c++)
        if (c in a && a[c] === b)
            return c;
    return - 1
};
_.cd = _.ad.lastIndexOf ? function(a, b, c) {
    return _.ad.lastIndexOf.call(a, b, null == c ? a.length - 1 : c)
} : function(a, b, c) {
    c = null == c ? a.length - 1 : c;
    0 > c && (c = Math.max(0, a.length + c));
    if (_.ja(a))
        return _.ja(b) && 1 == b.length ? a.lastIndexOf(b, c) : - 1;
    for (; 0 <= c; c--)
        if (c in a && a[c] === b)
            return c;
    return - 1
};
_.dd = _.ad.forEach ? function(a, b, c) {
    _.ad.forEach.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = _.ja(a) ? a.split("") : a, f = 0; f < d; f++)
        f in e && b.call(c, e[f], f, a)
};
_.ed = _.ad.filter ? function(a, b, c) {
    return _.ad.filter.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = [], f = 0, g = _.ja(a) ? a.split("") : a, k = 0; k < d; k++)
        if (k in g) {
            var l = g[k];
            b.call(c, l, k, a) && (e[f++] = l)
        }
    return e
};
_.fd = _.ad.map ? function(a, b, c) {
    return _.ad.map.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = Array(d), f = _.ja(a) ? a.split("") : a, g = 0; g < d; g++)
        g in f && (e[g] = b.call(c, f[g], g, a));
    return e
};
_.gd = _.ad.some ? function(a, b, c) {
    return _.ad.some.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = _.ja(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e && b.call(c, e[f], f, a))
            return !0;
    return !1
};
_.hd = _.ad.every ? function(a, b, c) {
    return _.ad.every.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = _.ja(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e&&!b.call(c, e[f], f, a))
            return !1;
    return !0
};
_.id = function(a, b) {
    return 0 <= (0, _.bd)(a, b)
};
_.jd = function(a, b) {
    for (var c = 1; c < arguments.length; c++) {
        var d = arguments[c], e;
        if (_.ha(d) || (e = _.Qc(d)) && Object.prototype.hasOwnProperty.call(d, "callee"))
            a.push.apply(a, d);
        else if (e)
            for (var f = a.length, g = d.length, k = 0; k < g; k++)
                a[f + k] = d[k];
        else 
            a.push(d)
    }
};

var ld, md, nd, od, qd, rd, sd;
_.L = function(a, b) {
    return _.aa[a] = b
};
_.kd = function(a, b) {
    return - 1 != a.indexOf(b)
};
ld = /[\x00&<>"']/;
md = /\x00/g;
nd = /'/g;
od = /"/g;
qd = />/g;
rd = /</g;
sd = /&/g;
_.td = function(a) {
    if (!ld.test(a))
        return a;
    - 1 != a.indexOf("&") && (a = a.replace(sd, "&amp;"));
    - 1 != a.indexOf("<") && (a = a.replace(rd, "&lt;"));
    - 1 != a.indexOf(">") && (a = a.replace(qd, "&gt;"));
    - 1 != a.indexOf('"') && (a = a.replace(od, "&quot;"));
    - 1 != a.indexOf("'") && (a = a.replace(nd, "&#39;"));
    - 1 != a.indexOf("\x00") && (a = a.replace(md, "&#0;"));
    return a
};
_.ud = function(a) {
    var b = [], c = 0, d;
    for (d in a)
        b[c++] = a[d];
    return b
};
_.vd = function(a) {
    var b = [], c = 0, d;
    for (d in a)
        b[c++] = d;
    return b
};
_.wd = function(a, b) {
    for (var c in a)
        if (a[c] == b)
            return !0;
    return !1
};
_.xd = function(a) {
    var b = arguments.length;
    if (1 == b && _.ha(arguments[0]))
        return _.xd.apply(null, arguments[0]);
    for (var c = {}, d = 0; d < b; d++)
        c[arguments[d]]=!0;
    return c
};
t: {
    var zd = _.r.navigator;
    if (zd) {
        var Ad = zd.userAgent;
        if (Ad) {
            _.yd = Ad;
            break t
        }
    }
    _.yd = ""
}
_.Bd = function(a) {
    return _.kd(_.yd, a)
};
_.Cd = function() {
    return _.Bd("Trident") || _.Bd("MSIE")
};
_.le = [];
_.me = [];
_.ne=!1;
_.oe = function(a) {
    _.le[_.le.length] = a;
    if (_.ne)
        for (var b = 0; b < _.me.length; b++)
            a((0, _.s)(_.me[b].wrap, _.me[b]))
};
var Bf, Df, Ef, Cf;
_.xf = function(a) {
    if (Error.captureStackTrace)
        Error.captureStackTrace(this, _.xf);
    else {
        var b = Error().stack;
        b && (this.stack = b)
    }
    a && (this.message = String(a))
};
_.u(_.xf, Error);
_.xf.prototype.name = "CustomError";
_.yf = function(a) {
    return a
};
_.zf = function(a) {
    var b = {}, c;
    for (c in a)
        b[c] = a[c];
    return b
};
_.Af = function(a) {
    for (var b = {}, c = 0, d = 0; d < a.length;) {
        var e = a[d++], f = _.Tc(e) ? "o" + _.Sc(e): (typeof e).charAt(0) + e;
        Object.prototype.hasOwnProperty.call(b, f) || (b[f]=!0, a[c++] = e)
    }
    a.length = c
};
Bf = function(a) {
    _.r.setTimeout(function() {
        throw a;
    }, 0)
};
_.Ff = function(a) {
    a = Cf(a);
    !_.Uc(_.r.setImmediate) || _.r.Window && _.r.Window.prototype.setImmediate == _.r.setImmediate ? (Df || (Df = Ef()), Df(a)) : _.r.setImmediate(a)
};
Ef = function() {
    var a = _.r.MessageChannel;
    "undefined" === typeof a && "undefined" !== typeof window && window.postMessage && window.addEventListener && (a = function() {
        var a = window.document.createElement("iframe");
        a.style.display = "none";
        a.src = "";
        window.document.documentElement.appendChild(a);
        var b = a.contentWindow, a = b.document;
        a.open();
        a.write("");
        a.close();
        var c = "callImmediate" + Math.random(), d = "file:" == b.location.protocol ? "*": b.location.protocol + "//" + b.location.host, a = (0, _.s)(function(a) {
            if (("*" == d || a.origin == d) &&
            a.data == c)
                this.port1.onmessage()
        }, this);
        b.addEventListener("message", a, !1);
        this.port1 = {};
        this.port2 = {
            postMessage: function() {
                b.postMessage(c, d)
            }
        }
    });
    if ("undefined" !== typeof a&&!_.Cd()) {
        var b = new a, c = {}, d = c;
        b.port1.onmessage = function() {
            if (_.ca(c.next)) {
                c = c.next;
                var a = c.wu;
                c.wu = null;
                a()
            }
        };
        return function(a) {
            d.next = {
                wu: a
            };
            d = d.next;
            b.port2.postMessage(0)
        }
    }
    return "undefined" !== typeof window.document && "onreadystatechange"in window.document.createElement("script") ? function(a) {
        var b = window.document.createElement("script");
        b.onreadystatechange = function() {
            b.onreadystatechange = null;
            b.parentNode.removeChild(b);
            b = null;
            a();
            a = null
        };
        window.document.documentElement.appendChild(b)
    } : function(a) {
        _.r.setTimeout(a, 0)
    }
};
Cf = _.yf;
_.oe(function(a) {
    Cf = a
});
var Lf = function(a, b) {
    Gf || Hf();
    If || (Gf(), If=!0);
    Jf.push(new Kf(a, b))
}, Gf, Hf = function() {
    if (_.r.Promise && _.r.Promise.resolve) {
        var a = _.r.Promise.resolve();
        Gf = function() {
            a.then(Mf)
        }
    } else 
        Gf = function() {
            _.Ff(Mf)
        }
}, If=!1, Jf = [], Mf = function() {
    for (; Jf.length;) {
        var a = Jf;
        Jf = [];
        for (var b = 0; b < a.length; b++) {
            var c = a[b];
            try {
                c.Fp.call(c.scope)
            } catch (d) {
                Bf(d)
            }
        }
    }
    If=!1
}, Kf = function(a, b) {
    this.Fp = a;
    this.scope = b
};
_.Nf = function(a) {
    a.prototype.then = a.prototype.then;
    a.prototype.$goog_Thenable=!0
};
_.Of = function(a) {
    if (!a)
        return !1;
    try {
        return !!a.$goog_Thenable
    } catch (b) {
        return !1
    }
};
_.Qf = function(a, b) {
    this.B = 0;
    this.F = void 0;
    this.A = this.C = null;
    this.D = this.G=!1;
    try {
        var c = this;
        a.call(b, function(a) {
            Pf(c, 2, a)
        }, function(a) {
            Pf(c, 3, a)
        })
    } catch (d) {
        Pf(this, 3, d)
    }
};
_.Qf.prototype.then = function(a, b, c) {
    return Rf(this, _.Uc(a) ? a : null, _.Uc(b) ? b : null, c)
};
_.Nf(_.Qf);
_.Qf.prototype.cancel = function(a) {
    0 == this.B && Lf(function() {
        var b = new Sf(a);
        Tf(this, b)
    }, this)
};
var Tf = function(a, b) {
    if (0 == a.B)
        if (a.C) {
            var c = a.C;
            if (c.A) {
                for (var d = 0, e =- 1, f = 0, g; g = c.A[f]; f++)
                    if (g = g.uj)
                        if (d++, g == a && (e = f), 0 <= e && 1 < d)
                            break;
                            0 <= e && (0 == c.B && 1 == d ? Tf(c, b) : (d = c.A.splice(e, 1)[0], Uf(c, d, 3, b)))
                        }
        } else 
            Pf(a, 3, b)
    }, Wf = function(a, b) {
    a.A && a.A.length || 2 != a.B && 3 != a.B || Vf(a);
    a.A || (a.A = []);
    a.A.push(b)
}, Rf = function(a, b, c, d) {
    var e = {
        uj: null,
        Hx: null,
        Ox: null
    };
    e.uj = new _.Qf(function(a, g) {
        e.Hx = b ? function(c) {
            try {
                var e = b.call(d, c);
                a(e)
            } catch (m) {
                g(m)
            }
        } : a;
        e.Ox = c ? function(b) {
            try {
                var e = c.call(d, b);
                !_.ca(e) && b instanceof
                Sf ? g(b) : a(e)
            } catch (m) {
                g(m)
            }
        } : g
    });
    e.uj.C = a;
    Wf(a, e);
    return e.uj
};
_.Qf.prototype.M = function(a) {
    this.B = 0;
    Pf(this, 2, a)
};
_.Qf.prototype.K = function(a) {
    this.B = 0;
    Pf(this, 3, a)
};
var Pf = function(a, b, c) {
    if (0 == a.B) {
        if (a == c)
            b = 3, c = new TypeError("Promise cannot resolve to itself");
        else {
            if (_.Of(c)) {
                a.B = 1;
                c.then(a.M, a.K, a);
                return 
            }
            if (_.Tc(c))
                try {
                    var d = c.then;
                    if (_.Uc(d)) {
                        Xf(a, c, d);
                        return 
                    }
            } catch (e) {
                b = 3, c = e
            }
        }
        a.F = c;
        a.B = b;
        Vf(a);
        3 != b || c instanceof Sf || Yf(a, c)
    }
}, Xf = function(a, b, c) {
    a.B = 1;
    var d=!1, e = function(b) {
        d || (d=!0, a.M(b))
    }, f = function(b) {
        d || (d=!0, a.K(b))
    };
    try {
        c.call(b, e, f)
    } catch (g) {
        f(g)
    }
}, Vf = function(a) {
    a.G || (a.G=!0, Lf(a.V, a))
};
_.Qf.prototype.V = function() {
    for (; this.A && this.A.length;) {
        var a = this.A;
        this.A = [];
        for (var b = 0; b < a.length; b++)
            Uf(this, a[b], this.B, this.F)
    }
    this.G=!1
};
var Uf = function(a, b, c, d) {
    if (2 == c)
        b.Hx(d);
    else {
        if (b.uj)
            for (; a && a.D; a = a.C)
                a.D=!1;
        b.Ox(d)
    }
}, Yf = function(a, b) {
    a.D=!0;
    Lf(function() {
        a.D && Zf.call(null, b)
    })
}, Zf = Bf, Sf = function(a) {
    _.xf.call(this, a)
};
_.u(Sf, _.xf);
Sf.prototype.name = "cancel";

_.H = _.H || {};
_.H = _.H || {};
(function() {
    function a(a, b) {
        return String.fromCharCode(b)
    }
    var b = {
        0: !1,
        10: !0,
        13: !0,
        34: !0,
        39: !0,
        60: !0,
        62: !0,
        92: !0,
        8232: !0,
        8233: !0,
        65282: !0,
        65287: !0,
        65308: !0,
        65310: !0,
        65340: !0
    };
    _.H.escape = function(a, b) {
        if (a) {
            if ("string" === typeof a)
                return _.H.Pl(a);
            if ("Array" === typeof a)
                for (var e = 0, f = a.length; e < f; ++e)
                    a[e] = _.H.escape(a[e]);
            else if ("object" === typeof a && b) {
                e = {};
                for (f in a)
                    a.hasOwnProperty(f) && (e[_.H.Pl(f)] = _.H.escape(a[f], !0));
                return e
            }
        }
        return a
    };
    _.H.Pl = function(a) {
        if (!a)
            return a;
        for (var d = [], e, f, g = 0, k = a.length; g < k; ++g)
            e = a.charCodeAt(g), f = b[e], !0 === f ? d.push("&#", e, ";") : !1 !== f && d.push(a.charAt(g));
        return d.join("")
    };
    _.H.VT = function(b) {
        return b ? b.replace(/&#([0-9]+);/g, a) : b
    }
})();

_.H = _.H || {};
(function() {
    function a(b) {
        var c = "";
        if (3 == b.nodeType || 4 == b.nodeType)
            c = b.nodeValue;
        else if (b.innerText)
            c = b.innerText;
        else if (b.innerHTML)
            c = b.innerHTML;
        else if (b.firstChild) {
            c = [];
            for (b = b.firstChild; b; b = b.nextSibling)
                c.push(a(b));
            c = c.join("")
        }
        return c
    }
    _.H.createElement = function(a) {
        var c;
        if (!window.document.body || window.document.body.namespaceURI)
            try {
                c = window.document.createElementNS("http://www.w3.org/1999/xhtml", a)
        } catch (d) {}
        return c || window.document.createElement(a)
    };
    _.H.Vo = function(a) {
        var c = _.H.createElement("iframe");
        try {
            var d = ["<", "iframe"], e = a || {}, f;
            for (f in e)
                e.hasOwnProperty(f) && (d.push(" "), d.push(f), d.push('="'), d.push(_.H.Pl(e[f])), d.push('"'));
            d.push("></");
            d.push("iframe");
            d.push(">");
            var g = _.H.createElement(d.join(""));
            g && (!c || g.tagName == c.tagName && g.namespaceURI == c.namespaceURI) && (c = g)
        } catch (k) {}
        d = c;
        a = a || {};
        for (var l in a)
            a.hasOwnProperty(l) && (d[l] = a[l]);
        return c
    };
    _.H.mv = function() {
        if (window.document.body)
            return window.document.body;
        try {
            var a = window.document.getElementsByTagNameNS("http://www.w3.org/1999/xhtml", "body");
            if (a && 1 == a.length)
                return a[0]
        } catch (c) {}
        return window.document.documentElement || window.document
    };
    _.H.AS = function(b) {
        return a(b)
    }
})();

_.Lb = function() {
    function a(a, b) {
        if (!(a < c) && d)
            if (2 === a && d.warn)
                d.warn(b);
            else if (3 === a && d.error)
                try {
                    d.error(b)
        } catch (g) {} else 
            d.log && d.log(b)
    }
    var b = function(b) {
        a(1, b)
    };
    _.Jb = function(b) {
        a(2, b)
    };
    _.Ib = function(b) {
        a(3, b)
    };
    _.Kb = function() {};
    b.INFO = 1;
    b.WARNING = 2;
    b.NONE = 4;
    var c = 1, d = window.console ? window.console: window.opera ? window.opera.postError: void 0;
    return b
}();

_.H = _.H || {};
(function() {
    var a = [];
    _.H.sT = function(b) {
        a.push(b)
    };
    _.H.GT = function() {
        for (var b = 0, c = a.length; b < c; ++b)
            a[b]()
    }
})();
_.Mb = function(a) {
    for (var b = 0; b < this.length; b++)
        if (this[b] === a)
            return b;
    return - 1
};
_.Nb = function(a, b) {
    var c = _.Ha(_.Ya, "watt", _.Ka());
    _.Ha(c, a, b)
};
_.H = _.H || {};
(function() {
    var a = null;
    _.H.Fb = function(b) {
        var c = "undefined" === typeof b;
        if (null !== a && c)
            return a;
        var d = {};
        b = b || window.location.href;
        var e = b.indexOf("?"), f = b.indexOf("#");
        b = ( - 1 === f ? b.substr(e + 1) : [b.substr(e + 1, f - e - 1), "&", b.substr(f + 1)].join("")).split("&");
        for (var e = window.decodeURIComponent ? window.decodeURIComponent : window.unescape, f = 0, g = b.length; f < g; ++f) {
            var k = b[f].indexOf("=");
            if ( - 1 !== k) {
                var l = b[f].substring(0, k), k = b[f].substring(k + 1), k = k.replace(/\+/g, " ");
                try {
                    d[l] = e(k)
                } catch (m) {}
            }
        }
        c && (a = d);
        return d
    };
    _.H.Fb()
})();
_.C("gadgets.util.getUrlParameters", _.H.Fb);
_.Ob = window.console;
_.Qb = function(a) {
    _.Ob && _.Ob.log && _.Ob.log(a)
};
_.Rb = function() {};
_.Sb = function() {
    var a = window.gadgets && window.gadgets.config && window.gadgets.config.get;
    a && _.eb(a());
    return {
        register: function(a, c, d) {
            d && d(_.E())
        },
        get: function(a) {
            return _.E(a)
        },
        update: function(a, c) {
            if (c)
                throw "Config replacement is not supported";
            _.eb(a)
        },
        gb: function() {}
    }
}();
_.C("gadgets.config.register", _.Sb.register);
_.C("gadgets.config.get", _.Sb.get);
_.C("gadgets.config.init", _.Sb.gb);
_.C("gadgets.config.update", _.Sb.update);
var Tb = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
    '"': '\\"',
    "\\": "\\\\"
}, Ub = function(a) {
    var b, c, d;
    b = /[\"\\\x00-\x1f\x7f-\x9f]/g;
    if (void 0 !== a) {
        switch (typeof a) {
        case "string":
            return b.test(a) ? '"' + a.replace(b, function(a) {
                var b = Tb[a];
                if (b)
                    return b;
                b = a.charCodeAt();
                return "\\u00" + Math.floor(b / 16).toString(16) + (b%16).toString(16)
            }) + '"' : '"' + a + '"';
        case "number":
            return (0, window.isFinite)(a) ? String(a) : "null";
        case "boolean":
        case "null":
            return String(a);
        case "object":
            if (!a)
                return "null";
            b = [];
            if ("number" ===
            typeof a.length&&!a.propertyIsEnumerable("length")) {
                d = a.length;
                for (c = 0; c < d; c += 1)
                    b.push(Ub(a[c]) || "null");
                return "[" + b.join(",") + "]"
            }
            for (c in a)
                !/___$/.test(c) && _.Ma(a, c) && "string" === typeof c && (d = Ub(a[c])) && b.push(Ub(c) + ":" + d);
            return "{" + b.join(",") + "}"
        }
        return ""
    }
}, Vb = function(a) {
    if (!a)
        return !1;
    if (/^[\],:{}\s]*$/.test(a.replace(/\\["\\\/b-u]/g, "@").replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:\s*\[)+/g, "")))
        try {
            return eval("(" + a + ")")
    } catch (b) {}
    return !1
}, Wb=!1, Zb;
try {
    Wb=!!window.JSON && '["a"]' === window.JSON.stringify(["a"]) && "a" === window.JSON.parse('["a"]')[0]
} catch (Xb) {}
Zb = function(a) {
    try {
        return window.JSON.parse(a)
    } catch (b) {
        return !1
    }
};
_.$b = Wb ? window.JSON.stringify : Ub;
_.ac = Wb ? Zb : Vb;

_.C("gadgets.json.stringify", _.$b);
_.C("gadgets.json.parse", _.ac);
_.Ua(_.Aa.location.href, "rpctoken") && _.Wa(_.Ba, "unload", function() {});
var bc, ec;
bc = function() {
    var a = _.Ba.readyState;
    return "complete" === a || "interactive" === a&&-1 == window.navigator.userAgent.indexOf("MSIE")
};
_.cc = function(a) {
    if (bc())
        a();
    else {
        var b=!1, c = function() {
            if (!b)
                return b=!0, a.apply(this, arguments)
        };
        _.Aa.addEventListener ? (_.Aa.addEventListener("load", c, !1), _.Aa.addEventListener("DOMContentLoaded", c, !1)) : _.Aa.attachEvent && (_.Aa.attachEvent("onreadystatechange", function() {
            bc() && c.apply(this, arguments)
        }), _.Aa.attachEvent("onload", c))
    }
};
_.dc = function(a, b) {
    if (!bc())
        try {
            a()
    } catch (c) {}
    _.cc(b)
};
ec = ec || {};
ec.Py = null;
ec.Ax = null;
ec.Gm = null;
ec.frameElement = null;
ec = ec || {};
ec.Hs || (ec.Hs = function() {
    function a(a) {
        "undefined" != typeof window.addEventListener ? window.addEventListener("message", a, !1) : "undefined" != typeof window.attachEvent && window.attachEvent("onmessage", a);
        window.___jsl = window.___jsl || {};
        var b = window.___jsl;
        b.RPMQ = b.RPMQ || [];
        b.RPMQ.push(a)
    }
    function b(a) {
        var b = (0, _.ac)(a.data);
        if (b && b.f) {
            (0, _.Kb)("gadgets.rpc.receive(" + window.name + "): " + a.data);
            var d = _.J.Of(b.f);
            e && ("undefined" !== typeof a.origin ? a.origin !== d : a.domain !== /^.+:\/\/([^:]+).*/.exec(d)[1]) ? _.Ib("Invalid rpc message origin. " +
            d + " vs " + (a.origin || "")) : c(b, a.origin)
        }
    }
    var c, d, e=!0;
    return {
        Mp: function() {
            return "wpm"
        },
        A: function() {
            return !0
        },
        gb: function(f, g) {
            _.Sb.register("rpc", null, function(a) {
                "true" === String((a && a.rpc || {}).disableForceSecure) && (e=!1)
            });
            c = f;
            d = g;
            a(b);
            d("..", !0);
            return !0
        },
        pb: function(a) {
            d(a, !0);
            return !0
        },
        call: function(a, b, c) {
            var d = _.J.Of(a), e = _.J.du(a);
            d ? window.setTimeout(function() {
                var a = (0, _.$b)(c);
                (0, _.Kb)("gadgets.rpc.send(" + window.name + "): " + a);
                e.postMessage(a, d)
            }, 0) : ".." != a && _.Ib("No relay set (used as window.postMessage targetOrigin), cannot send cross-domain message");
            return !0
        }
    }
}());
ec = ec || {};
ec.Tl || (ec.Tl = function() {
    function a(a, b) {
        G[b] = G[b] || function() {
            a.apply({}, arguments)
        }
    }
    function b() {
        if (null === v && window.document.body && l) {
            var a = l + "?cb=" + Math.random() + "&origin=" + z + "&jsl=1", c = window.document.createElement("div");
            c.style.height = "1px";
            c.style.width = "1px";
            a = '<object height="1" width="1" id="___xpcswf" type="application/x-shockwave-flash"><param name="allowScriptAccess" value="always"></param><param name="movie" value="' + a + '"></param><embed type="application/x-shockwave-flash" allowScriptAccess="always" src="' + a +
            '" height="1" width="1"></embed></object>';
            window.document.body.appendChild(c);
            c.innerHTML = a;
            v = c.firstChild
        }
        ++w;
        null !== t && (null !== v || 50 <= w) ? window.clearTimeout(t) : t = window.setTimeout(b, 100)
    }
    function c() {
        A[".."] || (k(".."), y++, 50 <= y && null !== D ? (window.clearTimeout(D), D = null) : D = window.setTimeout(c, 100))
    }
    function d() {
        if (null !== v && v.setup)
            for (; 0 < x.length;) {
                var a = x.shift(), b = a.VJ;
                v.setup(a.Si, ".." === b ? _.J.qo : b, ".." === b ? "INNER" : "OUTER")
            }
        null !== t && window.clearTimeout(t);
        t = null
    }
    function e() {
        A[".."] || null !==
        D || (D = window.setTimeout(c, 100))
    }
    function f(a, b, c) {
        b = _.J.Of(a);
        var d = _.J.Ag(a);
        v["sendMessage_" + (".." === a ? _.J.qo : a) + "_" + d + "_" + (".." === a ? "INNER" : "OUTER")].call(v, (0, _.$b)(c), b);
        return !0
    }
    function g(a, b) {
        var c = (0, _.ac)(a), d = c._scr;
        d ? (p(d, !0), A[d]=!0, ".." !== d && k(d, !0)) : window.setTimeout(function() {
            n(c, b)
        }, 0)
    }
    function k(a, b) {
        var c = _.J.qo, d = {};
        d._scr = b ? ".." : c;
        d._pnt = c;
        f(a, 0, d)
    }
    var l = null, m=!1, n = null, p = null, v = null, x = [], t = null, w = 0, y = 0, D = null, A = {}, z = window.location.protocol + "//" + window.location.host, G;
    window.___jsl =
    window.___jsl || {};
    G = window.___jsl._fm = {};
    _.Sb.register("rpc", null, function(a) {
        m && (l = a && a.rpc && a.rpc.commSwf || "//xpc.googleusercontent.com/gadgets/xpc.swf")
    });
    a(d, "ready");
    a(e, "setupDone");
    a(g, "receiveMessage");
    return {
        Mp: function() {
            return "flash"
        },
        A: function() {
            return !0
        },
        gb: function(a, b) {
            n = a;
            p = b;
            return m=!0
        },
        pb: function(a, c) {
            x.push({
                Si: c,
                VJ: a
            });
            null === v && null === t && (t = window.setTimeout(b, 100));
            return !0
        },
        call: f,
        bD: g,
        D: d,
        G: e
    }
}());
if (window.gadgets && window.gadgets.rpc)
    "undefined" != typeof _.J && _.J || (_.J = window.gadgets.rpc, _.J.config = _.J.config, _.J.register = _.J.register, _.J.uh = _.J.unregister, _.J.Ey = _.J.registerDefault, _.J.fA = _.J.unregisterDefault, _.J.hv = _.J.forceParentVerifiable, _.J.call = _.J.call, _.J.Rj = _.J.getRelayUrl, _.J.Ze = _.J.setRelayUrl, _.J.xn = _.J.setAuthToken, _.J.Dk = _.J.setupReceiver, _.J.Ag = _.J.getAuthToken, _.J.Dr = _.J.removeReceiver, _.J.Iv = _.J.getRelayChannel, _.J.qn = _.J.receive, _.J.Ay = _.J.receiveSameDomain, _.J.Ba = _.J.getOrigin,
    _.J.Of = _.J.getTargetOrigin, _.J.du = _.J._getTargetWin, _.J.aD = _.J._parseSiblingId);
else {
    _.J = function() {
        function a(a, b) {
            if (!Z[a]) {
                var c = ta;
                b || (c = pa);
                Z[a] = c;
                for (var d = $[a] || [], e = 0; e < d.length; ++e) {
                    var f = d[e];
                    f.t = A[a];
                    c.call(a, f.f, f)
                }
                $[a] = []
            }
        }
        function b() {
            function a() {
                Ta=!0
            }
            la || ("undefined" != typeof window.addEventListener ? window.addEventListener("unload", a, !1) : "undefined" != typeof window.attachEvent && window.attachEvent("onunload", a), la=!0)
        }
        function c(a, c, d, e, f) {
            A[c] && A[c] === d || (_.Ib("Invalid gadgets.rpc token. " +
            A[c] + " vs " + d), La(c, 2));
            f.onunload = function() {
                I[c]&&!Ta && (La(c, 1), _.J.Dr(c))
            };
            b();
            e = (0, _.ac)((0, window.decodeURIComponent)(e))
        }
        function d(b, c) {
            if (b && "string" === typeof b.s && "string" === typeof b.f && b.a instanceof Array)
                if (A[b.f] && A[b.f] !== b.t && (_.Ib("Invalid gadgets.rpc token. " + A[b.f] + " vs " + b.t), La(b.f, 2)), "__ack" === b.s)
                    window.setTimeout(function() {
                        a(b.f, !0)
                    }, 0);
                else {
                    b.c && (b.callback = function(a) {
                        _.J.call(b.f, (b.g ? "legacy__" : "") + "__cb", null, b.c, a)
                    });
                    if (c) {
                        var d = e(c);
                        b.origin = c;
                        var f = b.r, g;
                        try {
                            g = e(f)
                        } catch (k) {}
                        f &&
                        g == d || (f = c);
                        b.referer = f
                    }
                    d = (w[b.s] || w[""]).apply(b, b.a);
                    b.c && "undefined" !== typeof d && _.J.call(b.f, "__cb", null, b.c, d)
                }
            }
        function e(a) {
            if (!a)
                return "";
            a = a.split("#")[0].split("?")[0];
            a = a.toLowerCase();
            0 == a.indexOf("//") && (a = window.location.protocol + a);
            - 1 == a.indexOf("://") && (a = window.location.protocol + "//" + a);
            var b = a.substring(a.indexOf("://") + 3), c = b.indexOf("/");
            - 1 != c && (b = b.substring(0, c));
            a = a.substring(0, a.indexOf("://"));
            if ("http" !== a && "https" !== a && "chrome-extension" !== a && "file" !== a)
                throw Error("b");
            var c = "", d = b.indexOf(":");
            if ( - 1 != d) {
                var e = b.substring(d + 1), b = b.substring(0, d);
                if ("http" === a && "80" !== e || "https" === a && "443" !== e)
                    c = ":" + e
            }
            return a + "://" + b + c
        }
        function f(a) {
            if ("/" == a.charAt(0)) {
                var b = a.indexOf("|");
                return {
                    id: 0 < b ? a.substring(1, b): a.substring(1),
                    origin: 0 < b ? a.substring(b + 1): null
                }
            }
            return null
        }
        function g(a) {
            if ("undefined" === typeof a || ".." === a)
                return window.parent;
            var b = f(a);
            if (b)
                return window.top.frames[b.id];
            a = String(a);
            return (b = window.frames[a]) ? b : (b = window.document.getElementById(a)) && b.contentWindow ?
            b.contentWindow : null
        }
        function k(a, b) {
            if (!0 !== I[a]) {
                "undefined" === typeof I[a] && (I[a] = 0);
                var c = g(a);
                ".." !== a && null == c ||!0 !== ta.pb(a, b)?!0 !== I[a] && 10 > I[a]++?window.setTimeout(function() {
                    k(a, b)
                }, 500) : (Z[a] = pa, I[a]=!0) : I[a]=!0
            }
        }
        function l(a) {
            (a = y[a]) && "/" === a.substring(0, 1) && (a = "/" === a.substring(1, 2) ? window.document.location.protocol + a : window.document.location.protocol + "//" + window.document.location.host + a);
            return a
        }
        function m(a, b, c) {
            b&&!/http(s)?:\/\/.+/.test(b) && (0 == b.indexOf("//") ? b = window.location.protocol +
            b : "/" == b.charAt(0) ? b = window.location.protocol + "//" + window.location.host + b : - 1 == b.indexOf("://") && (b = window.location.protocol + "//" + b));
            y[a] = b;
            "undefined" !== typeof c && (D[a]=!!c)
        }
        function n(a, b) {
            b = b || "";
            A[a] = String(b);
            k(a, b)
        }
        function p(a) {
            a = (a.passReferrer || "").split(":", 2);
            V = a[0] || "none";
            xa = a[1] || "origin"
        }
        function v(b) {
            "true" === String(b.useLegacyProtocol) && (ta = ec.Gm || pa, ta.gb(d, a))
        }
        function x(a, b) {
            function c(d) {
                d = d && d.rpc || {};
                p(d);
                var f = d.parentRelayUrl || "", f = e(U.parent || b) + f;
                m("..", f, "true" === String(d.useLegacyProtocol));
                v(d);
                n("..", a)
            }
            !U.parent && b ? c({}) : _.Sb.register("rpc", null, c)
        }
        function t(a, b, c) {
            if (".." === a)
                x(c || U.rpctoken || U.ifpctok || "", b);
            else 
                t: {
                var d = null;
                if ("/" != a.charAt(0)) {
                    if (!_.H)
                        break t;
                        d = window.document.getElementById(a);
                        if (!d)
                            throw Error("c`" + a);
                }
                d = d && d.src;
                b = b || _.J.Ba(d);
                m(a, b);
                b = _.H.Fb(d);
                n(a, c || b.rpctoken)
            }
        }
        var w = {}, y = {}, D = {}, A = {}, z = 0, G = {}, I = {}, U = {}, Z = {}, $ = {}, V = null, xa = null, P = window.top !== window.self, ma = window.name, La = function() {}, X = window.console, ea = X && X.log && function(a) {
            X.log(a)
        }
        || function() {},
        pa = function() {
            function a(b) {
                return function() {
                    ea(b + ": call ignored")
                }
            }
            return {
                getCode: function() {
                    return "noop"
                },
                isParentVerifiable: function() {
                    return !0
                },
                init: a("init"),
                setup: a("setup"),
                call: a("call")
            }
        }();
        _.H && (U = _.H.Fb());
        var Ta=!1, la=!1, ta = function() {
            if ("flash" == U.rpctx)
                return ec.Tl;
            if ("rmr" == U.rpctx)
                return ec.Py;
            var a = "function" === typeof window.postMessage ? ec.Hs: "object" === typeof window.postMessage ? ec.Hs: window.ActiveXObject ? ec.Tl ? ec.Tl: ec.Ax ? ec.Ax: ec.Gm: 0 < window.navigator.userAgent.indexOf("WebKit") ?
            ec.Py: "Gecko" === window.navigator.product ? ec.frameElement: ec.Gm;
            a || (a = pa);
            return a
        }();
        w[""] = function() {
            ea("Unknown RPC service: " + this.s)
        };
        w.__cb = function(a, b) {
            var c = G[a];
            c && (delete G[a], c.call(this, b))
        };
        return {
            config: function(a) {
                "function" === typeof a.Wy && (La = a.Wy)
            },
            register: function(a, b) {
                if ("__cb" === a || "__ack" === a)
                    throw Error("d");
                if ("" === a)
                    throw Error("e");
                w[a] = b
            },
            uh: function(a) {
                if ("__cb" === a || "__ack" === a)
                    throw Error("f");
                if ("" === a)
                    throw Error("g");
                delete w[a]
            },
            Ey: function(a) {
                w[""] = a
            },
            fA: function() {
                delete w[""]
            },
            hv: function() {},
            call: function(a, b, c, d) {
                a = a || "..";
                var e = "..";
                ".." === a ? e = ma : "/" == a.charAt(0) && (e = _.J.Ba(window.location.href), e = "/" + ma + (e ? "|" + e : ""));
                ++z;
                c && (G[z] = c);
                var g = {
                    s: b,
                    f: e,
                    c: c ? z: 0,
                    a: Array.prototype.slice.call(arguments, 3),
                    t: A[a],
                    l: !!D[a]
                }, k;
                t: if ("bidir" === V || "c2p" === V && ".." === a || "p2c" === V && ".." !== a) {
                    k = window.location.href;
                    var l = "?";
                    if ("query" === xa)
                        l = "#";
                    else if ("hash" === xa)
                        break t;
                    l = k.lastIndexOf(l);
                    l =- 1 === l ? k.length : l;
                    k = k.substring(0, l)
                } else 
                    k = null;
                k && (g.r = k);
                if (".." === a || null != f(a) || window.document.getElementById(a))(k =
                Z[a]) || null === f(a) || (k = ta), 0 === b.indexOf("legacy__") && (k = ta, g.s = b.substring(8), g.c = g.c ? g.c : z), g.g=!0, g.r = e, k ? (D[a] && (k = ec.Gm), !1 === k.call(a, e, g) && (Z[a] = pa, ta.call(a, e, g))): $[a] ? $[a].push(g): $[a] = [g]
            }, Rj: l, Ze : m, xn : n, Dk : t, Ag : function(a) {
                return A[a]
            }, Dr: function(a) {
                delete y[a];
                delete D[a];
                delete A[a];
                delete I[a];
                delete Z[a]
            }, Iv: function() {
                return ta.Mp()
            }, qn: function(a, b) {
                4 < a.length ? ta.bD(a, d) : c.apply(null, a.concat(b))
            }, Ay: function(a) {
                a.a = Array.prototype.slice.call(a.a);
                window.setTimeout(function() {
                    d(a)
                },
                0)
            }, Ba: e, Of : function(a) {
                var b = null, c = l(a);
                c ? b = c : (c = f(a)) ? b = c.origin : b = ".." == a ? U.parent : window.document.getElementById(a).src;
                return e(b)
            }, gb: function() {
                !1 === ta.gb(d, a) && (ta = pa);
                P ? t("..") : _.Sb.register("rpc", null, function(a) {
                    a = a.rpc || {};
                    p(a);
                    v(a)
                })
            }, du: g, aD : f, A : "__ack", qo : ma || "..", D : 0, C : 1, B : 2
        }
    }();
    _.J.gb()
};
_.J.config({
    Wy: function(a) {
        throw Error("h`" + a);
    }
});
_.Kb = _.Rb;
_.C("gadgets.rpc.config", _.J.config);
_.C("gadgets.rpc.register", _.J.register);
_.C("gadgets.rpc.unregister", _.J.uh);
_.C("gadgets.rpc.registerDefault", _.J.Ey);
_.C("gadgets.rpc.unregisterDefault", _.J.fA);
_.C("gadgets.rpc.forceParentVerifiable", _.J.hv);
_.C("gadgets.rpc.call", _.J.call);
_.C("gadgets.rpc.getRelayUrl", _.J.Rj);
_.C("gadgets.rpc.setRelayUrl", _.J.Ze);
_.C("gadgets.rpc.setAuthToken", _.J.xn);
_.C("gadgets.rpc.setupReceiver", _.J.Dk);
_.C("gadgets.rpc.getAuthToken", _.J.Ag);
_.C("gadgets.rpc.removeReceiver", _.J.Dr);
_.C("gadgets.rpc.getRelayChannel", _.J.Iv);
_.C("gadgets.rpc.receive", _.J.qn);
_.C("gadgets.rpc.receiveSameDomain", _.J.Ay);
_.C("gadgets.rpc.getOrigin", _.J.Ba);
_.C("gadgets.rpc.getTargetOrigin", _.J.Of);

_.Mi = window.gapi && window.gapi.util || {};
var Ni = _.Mi = _.Mi || {};
window.___jsl = window.___jsl || {};
Ni.Jo = {
    A: function() {
        return window.___jsl.bsh
    },
    yv: function() {
        return window.___jsl.h
    },
    Rr: function(a) {
        window.___jsl.bsh = a
    },
    vJ: function(a) {
        window.___jsl.h = a
    }
};
var Qi, Ri;
Qi = /^https?:\/\/(?:\w|[\-\.])+\.google\.(?:\w|[\-:\.])+(?:\/[^\?\#]*)?\/u\/(\d)\//;
Ri = /^https?:\/\/(?:\w|[\-\.])+\.google\.(?:\w|[\-:\.])+(?:\/[^\?\#]*)?\/b\/(\d{10,})\//;
_.Si = function(a) {
    var b = _.E("googleapis.config/sessionIndex");
    null == b && (b = window.__X_GOOG_AUTHUSER);
    if (null == b) {
        var c = window.google;
        c && (b = c.authuser)
    }
    null == b && (a = a || window.location.href, b = _.Ua(a, "authuser") || null, null == b && (b = (b = a.match(Qi)) ? b[1] : null));
    return null == b ? null : String(b)
};
_.Ti = function(a) {
    var b = _.E("googleapis.config/sessionDelegate");
    null == b && (b = (a = (a || window.location.href).match(Ri)) ? a[1] : null);
    return null == b ? null : String(b)
};
_.Ui = function(a, b) {
    var c = _.Si(a) || b, d = _.Ti(a), e = "";
    c && (e += "u/" + c + "/");
    d && (e += "b/" + d + "/");
    return e || null
};

_.Vi = function(a) {
    var b;
    if (b = _.E("enableMultilogin"))
        if (b = a("cookie_policy"))
            b = String(a("immediate") || ""), a = String(a("prompt") || ""), b=!("true" === b.toLowerCase() || "none" === a.toLowerCase());
    return b?!0 : !1
};
_.Wi = function(a, b, c) {
    a = String(a);
    if (null != (_.Ua(a, "authuser") || null) || null != (_.Ua(a, "hd") || null))
        return a;
    b = _.Si(b);
    if (c) {
        var d = a, e = d.match(/^((https?:)?\/\/[^\/?#]*)?(\/[^\/?#]+)\/[0-9]+([\/][^?#]*)([?#].*)?$/);
        if (e && e[0]) {
            var f = e[1], g = e[4], k = e[5];
            e[3] == "/" + c && (d = (f || "") + (g || "/") + (k || ""))
        }
        if ((e = d.match(/^(((https?:)?\/\/[^\/?#]*)([\/][^?#]*)?|([\/][^?#]*))([?#].*)?$/)) && e[0]) {
            var f = e[2], l = e[4] || e[5], k = e[6];
            null != b && (d = (f || "") + "/" + c + "/" + (0, window.encodeURIComponent)(b) + (l || "/") + (k || ""));
            return d
        }
    }
    null ==
    b ? _.Vi(function(b) {
        return _.Ua(a, b) || null
    }) || (l = (0, window.encodeURIComponent)("authuser") + "=0") : l = b.match(/^([-a-z0-9]+[.])+[-a-z0-9]+$/) ? [(0, window.encodeURIComponent)("authuser") + "=", (0, window.encodeURIComponent)(String(b)), "&" + (0, window.encodeURIComponent)("hd") + "=", (0, window.encodeURIComponent)(b)].join("") : ["authuser=", (0, window.encodeURIComponent)(b)].join("");
    c = a.split("#");
    b = c[0].indexOf("?");
    0 > b ? c[0] = [c[0], "?", l].join("") : (d = [c[0]], b < c[0].length - 1 && d.push("&"), d.push(l), c[0] = d.join(""));
    return d = c.join("#")
};
_.google.oT = _.Wi;
_.google.$p = _.Si;
_.google.HS = _.Ti;
_.google.IS = _.Ui;

var Xi = function() {
    this.B =- 1
};
var Yi = function() {
    this.B =- 1;
    this.B = 64;
    this.A = [];
    this.F = [];
    this.K = [];
    this.D = [];
    this.D[0] = 128;
    for (var a = 1; a < this.B; ++a)
        this.D[a] = 0;
    this.G = this.C = 0;
    this.reset()
};
_.u(Yi, Xi);
Yi.prototype.reset = function() {
    this.A[0] = 1732584193;
    this.A[1] = 4023233417;
    this.A[2] = 2562383102;
    this.A[3] = 271733878;
    this.A[4] = 3285377520;
    this.G = this.C = 0
};
var Zi = function(a, b, c) {
    c || (c = 0);
    var d = a.K;
    if (_.ja(b))
        for (var e = 0; 16 > e; e++)
            d[e] = b.charCodeAt(c)<<24 | b.charCodeAt(c + 1)<<16 | b.charCodeAt(c + 2)<<8 | b.charCodeAt(c + 3), c += 4;
    else 
        for (e = 0; 16 > e; e++)
            d[e] = b[c]<<24 | b[c + 1]<<16 | b[c + 2]<<8 | b[c + 3], c += 4;
    for (e = 16; 80 > e; e++) {
        var f = d[e - 3]^d[e - 8]^d[e - 14]^d[e - 16];
        d[e] = (f<<1 | f>>>31) & 4294967295
    }
    b = a.A[0];
    c = a.A[1];
    for (var g = a.A[2], k = a.A[3], l = a.A[4], m, e = 0; 80 > e; e++)
        40 > e ? 20 > e ? (f = k^c & (g^k), m = 1518500249) : (f = c^g^k, m = 1859775393) : 60 > e ? (f = c & g | k & (c | g), m = 2400959708) : (f = c^g^k, m = 3395469782),
        f = (b<<5 | b>>>27) + f + l + m + d[e] & 4294967295, l = k, k = g, g = (c<<30 | c>>>2) & 4294967295, c = b, b = f;
    a.A[0] = a.A[0] + b & 4294967295;
    a.A[1] = a.A[1] + c & 4294967295;
    a.A[2] = a.A[2] + g & 4294967295;
    a.A[3] = a.A[3] + k & 4294967295;
    a.A[4] = a.A[4] + l & 4294967295
};
Yi.prototype.update = function(a, b) {
    if (null != a) {
        _.ca(b) || (b = a.length);
        for (var c = b - this.B, d = 0, e = this.F, f = this.C; d < b;) {
            if (0 == f)
                for (; d <= c;)
                    Zi(this, a, d), d += this.B;
            if (_.ja(a))
                for (; d < b;) {
                    if (e[f] = a.charCodeAt(d), ++f, ++d, f == this.B) {
                        Zi(this, e);
                        f = 0;
                        break
                    }
                } else 
                    for (; d < b;)
                        if (e[f] = a[d], ++f, ++d, f == this.B) {
                            Zi(this, e);
                            f = 0;
                            break
                        }
        }
        this.C = f;
        this.G += b
    }
};
Yi.prototype.M = function() {
    var a = [], b = 8 * this.G;
    56 > this.C ? this.update(this.D, 56 - this.C) : this.update(this.D, this.B - (this.C - 56));
    for (var c = this.B - 1; 56 <= c; c--)
        this.F[c] = b & 255, b/=256;
    Zi(this, this.F);
    for (c = b = 0; 5 > c; c++)
        for (var d = 24; 0 <= d; d -= 8)
            a[b] = this.A[c]>>d & 255, ++b;
    return a
};
_.$i = function() {
    this.A = new Yi
};
_.h = _.$i.prototype;
_.h.reset = function() {
    this.A.reset()
};
_.h.gA = function(a) {
    this.A.update(a)
};
_.h.Ou = function() {
    return this.A.M()
};
_.h.zs = function(a) {
    a = (0, window.unescape)((0, window.encodeURIComponent)(a));
    for (var b = [], c = 0, d = a.length; c < d; ++c)
        b.push(a.charCodeAt(c));
    this.gA(b)
};
_.h.Ef = function() {
    for (var a = this.Ou(), b = "", c = 0; c < a.length; c++)
        b += "0123456789ABCDEF".charAt(Math.floor(a[c] / 16)) + "0123456789ABCDEF".charAt(a[c]%16);
    return b
};

_.aj = function() {
    return Math.floor((new Date).getTime() / 1E3)
};
var cj, dj;
_.bj = function(a, b) {
    this.A = a;
    var c = b || {};
    this.D = c.tH;
    this.B = c.domain;
    this.C = c.path;
    this.G = c.ZI
};
cj = /^[-+/_=.:|%&a-zA-Z0-9@]*$/;
dj = /^[A-Z_][A-Z0-9_]{0,63}$/;
_.bj.prototype.bg = function() {
    for (var a = this.A + "=", b = window.document.cookie.split(/;\s*/), c = 0; c < b.length; ++c) {
        var d = b[c];
        if (0 == d.indexOf(a))
            return d.substr(a.length)
    }
};
_.bj.prototype.write = function(a, b) {
    if (!dj.test(this.A))
        throw "Invalid cookie name";
    if (!cj.test(a))
        throw "Invalid cookie value";
    var c = this.A + "=" + a;
    this.B && (c += ";domain=" + this.B);
    this.C && (c += ";path=" + this.C);
    var d = "number" === typeof b ? b: this.D;
    if (0 <= d) {
        var e = new Date;
        e.setSeconds(e.getSeconds() + d);
        c += ";expires=" + e.toUTCString()
    }
    this.G && (c += ";secure");
    window.document.cookie = c;
    return !0
};
_.bj.prototype.clear = function() {
    this.write("", 0)
};
_.bj.iterate = function(a) {
    for (var b = window.document.cookie.split(/;\s*/), c = 0; c < b.length; ++c) {
        var d = b[c].split("="), e = d.shift();
        a(e, d.join("="))
    }
};
var fj;
_.ej = function(a) {
    this.A = a
};
fj = {};
_.ej.prototype.bg = function() {
    if (fj.hasOwnProperty(this.A))
        return fj[this.A]
};
_.ej.prototype.write = function(a) {
    fj[this.A] = a;
    return !0
};
_.ej.prototype.clear = function() {
    delete fj[this.A]
};
_.ej.iterate = function(a) {
    for (var b in fj)
        fj.hasOwnProperty(b) && a(b, fj[b])
};
_.gj = function(a) {
    this.B = a;
    this.A = window.sessionStorage
};
_.gj.prototype.bg = function() {
    return this.A && this.A.getItem(this.B)
};
_.gj.prototype.write = function(a) {
    try {
        this.A && this.A.setItem(this.B, a)
    } catch (b) {
        return !1
    }
    return !0
};
_.gj.prototype.clear = function() {
    this.A && this.A.removeItem(this.B)
};
_.gj.iterate = function(a) {
    if (window.sessionStorage)
        for (var b = 0, c = window.sessionStorage.length; b < c; ++b) {
            var d = window.sessionStorage.key(b);
            a(d, window.sessionStorage[d])
        }
};
for (var hj = 0; 64 > hj; ++hj);
var ij;
ij = [".APPS.GOOGLEUSERCONTENT.COM", "@DEVELOPER.GSERVICEACCOUNT.COM"];
_.jj = "https:" === window.location.protocol;
_.kj = _.jj || "http:" === window.location.protocol ? _.bj : _.ej;
_.lj = function(a) {
    a = a.toUpperCase();
    for (var b = 0, c = ij.length; b < c; ++b) {
        var d = a.split(ij[b]);
        2 == d.length && "" === d[1] && (a = d[0])
    }
    a = a.replace(/-/g, "_").toUpperCase();
    40 < a.length && (b = new _.$i, b.zs(a), a = b.Ef().toUpperCase());
    return a
};
_.mj = function(a) {
    var b = a.substr(1), c = "", d = window.location.hostname;
    if ("" !== b) {
        c = (0, window.parseInt)(b, 10);
        if ((0, window.isNaN)(c))
            return null;
        b = d.split(".");
        if (b.length < c - 1)
            return null;
        b.length == c - 1 && (d = "." + d)
    } else 
        d = "";
    return {
        yb: "S" == a.charAt(0),
        domain: d,
        wd: c
    }
};
_.nj = function(a) {
    if (0 !== a.indexOf("GCSC"))
        return null;
    var b = {
        Dc: !1
    };
    a = a.substr(4);
    if (!a)
        return b;
    var c = a.charAt(0);
    a = a.substr(1);
    var d = a.lastIndexOf("_");
    if ( - 1 == d)
        return b;
    var e = _.mj(a.substr(d + 1));
    if (null == e)
        return b;
    a = a.substring(0, d);
    if ("_" !== a.charAt(0))
        return b;
    d = "E" === c && e.yb;
    return !d && ("U" !== c || e.yb) || d&&!_.jj ? b : {
        Dc: !0,
        yb: d,
        zu: a.substr(1),
        domain: e.domain,
        wd: e.wd
    }
};
var oj, pj, tj, uj, xj;
oj = _.Ka();
pj = _.Ka();
_.qj = _.Ka();
_.rj = _.Ka();
_.sj = null;
tj = "state code cookie_policy g_user_cookie_policy authuser prompt g-oauth-window status".split(" ");
uj = function(a) {
    this.B = a;
    this.A = null
};
uj.prototype.write = function(a) {
    var b = _.Ka(), c = _.Ka(), d;
    for (d in a)
        _.Ma(a, d) && (c[d] = a[d], b[d] = a[d]);
    d = 0;
    for (var e = tj.length; d < e; ++d)
        delete c[tj[d]];
    a = String(a.authuser || 0);
    d = _.Ka();
    d[a] = _.H.Fb("#" + _.vj(c));
    this.B.write((0, _.$b)(d));
    this.A = b
};
uj.prototype.bg = function() {
    return this.A
};
uj.prototype.clear = function() {
    this.B.clear();
    this.A = _.Ka()
};
_.wj = function(a) {
    if (!a)
        return null;
    "single_host_origin" !== a && (a = _.J.Ba(a));
    var b = window.location.hostname, c = b, d = _.jj;
    if ("single_host_origin" !== a) {
        c = a.split("://");
        if (2 == c.length)
            d = "https" === c.shift();
        else 
            return _.Qb("WARNING invalid cookie_policy: " + a), null;
        c = c[0]
    }
    if ( - 1 !== c.indexOf(":"))
        c = b = "";
    else {
        a = "." + c;
        if (b.lastIndexOf(a) !== b.length - a.length)
            return _.Qb("Invalid cookie_policy domain: " + c), null;
        c = a;
        b = c.split(".").length - 1
    }
    return {
        domain: c,
        yb: d,
        wd: b
    }
};
xj = function(a) {
    if (!a)
        return null;
    var b = a.client_id;
    if (!b)
        return null;
    b = _.lj(b);
    a = _.wj(a.cookie_policy);
    return a?!_.jj && a.yb ? (_.Qb("WARNING: https cookie_policy set for http domain"), null) : ["GCSC", a.yb ? "E": "U", "_", b, "_", a.yb ? "S": "H", a.wd].join("") : null
};
_.yj = function(a) {
    return a ? {
        domain: a.domain,
        path: "/",
        ZI: a.yb
    } : null
};
_.zj = function(a) {
    var b = oj[a];
    b || (b = new uj(new _.ej(a)), oj[a] = b);
    return {
        rh: b,
        key: a
    }
};
_.Aj = function(a, b) {
    var c = b ? pj: oj, d = b ? _.kj: _.gj, e = a && xj(a), f=!!e;
    a&&!a.g_user_cookie_policy && (d = _.ej, e = "token");
    if (!e)
        if (!b && _.sj)
            e = _.sj;
        else 
            return null;
    var g = c[e];
    if (!g) {
        g = _.nj(e);
        if (!("token" === e || g && g.Dc))
            return null;
        g = new d(e, _.yj(g));
        b || (g = new uj(g))
    }
    c[e] = g;
    return {
        rh: g,
        key: e,
        mE: f
    }
};
_.Bj = function(a, b, c) {
    a = a && "token" !== a ? _.zj(a) : _.Aj();
    if (!a)
        return null;
    if (c) {
        c = a.rh;
        _.Ka();
        var d = c.B.bg();
        c = null;
        try {
            c = (0, _.ac)(d)
        } catch (e) {}
        0 == c && (c = null);
        d = _.Si() || "0";
        d = String(d);
        c = c && c[d]
    } else 
        c = a.rh.bg();
    c && c.expires_at && _.aj() > c.expires_at && (a.rh.clear(), c = null);
    c && c.error&&!b && (c = null);
    return c
};
_.vj = function(a) {
    var b = "";
    if (!a)
        return b;
    for (var c in a)
        if ({}.hasOwnProperty.call(a, c)) {
            var d;
            d = a[c];
            if (null != d) {
                var e = [(0, window.encodeURIComponent)(c), "="];
                if (d instanceof Array) {
                    for (var f = [], g = 0; g < d.length; g++)
                        f.push((0, window.encodeURIComponent)(d[g]));
                        e.push(f.join("+"))
                    } else 
                        e.push((0, window.encodeURIComponent)(d));
                        d = e.join("")
                    } else 
                        d = "";
                        d && (b && (b += "&"), b += d)
        }
    return b
};
_.gj.iterate(function(a) {
    var b = _.nj(a);
    b && b.Dc && (oj[a] = new uj(new _.gj(a)))
});
_.kj.iterate(function(a) {
    oj[a] && (pj[a] = new _.kj(a, _.yj(_.nj(a))))
});

_.Oi = function(a) {
    if (!a)
        return "";
    a = a.split("#")[0].split("?")[0];
    a = a.toLowerCase();
    0 == a.indexOf("//") && (a = window.location.protocol + a);
    /^[\w\-]*:\/\//.test(a) || (a = window.location.href);
    var b = a.substring(a.indexOf("://") + 3), c = b.indexOf("/");
    - 1 != c && (b = b.substring(0, c));
    a = a.substring(0, a.indexOf("://"));
    if ("http" !== a && "https" !== a && "chrome-extension" !== a && "file" !== a)
        throw Error("b");
    var c = "", d = b.indexOf(":");
    if ( - 1 != d) {
        var e = b.substring(d + 1), b = b.substring(0, d);
        if ("http" === a && "80" !== e || "https" === a && "443" !== e)
            c = ":" + e
    }
    return a + "://" + b + c
};
_.Mi.Ba = function(a) {
    return _.Oi(a)
};
_.Pi = window.googleapis && window.googleapis.server || {};
_.Cj = function() {
    var a = /\s*;\s*/;
    return {
        get: function(b, c) {
            for (var d = b + "=", e = (window.document.cookie || "").split(a), f = 0, g; g = e[f]; ++f)
                if (0 == g.indexOf(d))
                    return g.substr(d.length);
            return c
        }
    }
}();
var Ej, Fj;
_.Dj = function(a) {
    this.A = a
};
Ej = /\s*;\s*/;
_.h = _.Dj.prototype;
_.h.isEnabled = function() {
    return window.navigator.cookieEnabled
};
_.h.set = function(a, b, c, d, e, f) {
    if (/[;=\s]/.test(a))
        throw Error("E`" + a);
    if (/[;\r\n]/.test(b))
        throw Error("F`" + b);
    _.ca(c) || (c =- 1);
    e = e ? ";domain=" + e : "";
    d = d ? ";path=" + d : "";
    f = f ? ";secure" : "";
    c = 0 > c ? "" : 0 == c ? ";expires=" + (new Date(1970, 1, 1)).toUTCString() : ";expires=" + (new Date((0, _.sa)() + 1E3 * c)).toUTCString();
    this.A.cookie = a + "=" + b + e + d + c + f
};
_.h.get = function(a, b) {
    for (var c = a + "=", d = (this.A.cookie || "").split(Ej), e = 0, f; f = d[e]; e++) {
        if (0 == f.lastIndexOf(c, 0))
            return f.substr(c.length);
        if (f == a)
            return ""
    }
    return b
};
_.h.remove = function(a, b, c) {
    var d = _.ca(this.get(a));
    this.set(a, "", 0, b, c);
    return d
};
_.h.fd = function() {
    return Fj(this).keys
};
_.h.xc = function() {
    return Fj(this).jA
};
_.h.isEmpty = function() {
    return !this.A.cookie
};
_.h.fb = function() {
    return this.A.cookie ? (this.A.cookie || "").split(Ej).length : 0
};
_.h.Mh = function(a) {
    for (var b = Fj(this).jA, c = 0; c < b.length; c++)
        if (b[c] == a)
            return !0;
    return !1
};
_.h.clear = function() {
    for (var a = Fj(this).keys, b = a.length - 1; 0 <= b; b--)
        this.remove(a[b])
    };
Fj = function(a) {
    a = (a.A.cookie || "").split(Ej);
    for (var b = [], c = [], d, e, f = 0; e = a[f]; f++)
        d = e.indexOf("="), - 1 == d ? (b.push(""), c.push(e)) : (b.push(e.substring(0, d)), c.push(e.substring(d + 1)));
    return {
        keys: b,
        jA: c
    }
};
_.Gj = new _.Dj(window.document);
_.Gj.B = 3950;

var Hj = function() {
    function a() {
        e[0] = 1732584193;
        e[1] = 4023233417;
        e[2] = 2562383102;
        e[3] = 271733878;
        e[4] = 3285377520;
        n = m = 0
    }
    function b(a) {
        for (var b = g, c = 0; 64 > c; c += 4)
            b[c / 4] = a[c]<<24 | a[c + 1]<<16 | a[c + 2]<<8 | a[c + 3];
        for (c = 16; 80 > c; c++)
            a = b[c - 3]^b[c - 8]^b[c - 14]^b[c - 16], b[c] = (a<<1 | a>>>31) & 4294967295;
        a = e[0];
        for (var d = e[1], f = e[2], k = e[3], l = e[4], m, n, c = 0; 80 > c; c++)
            40 > c ? 20 > c ? (m = k^d & (f^k), n = 1518500249) : (m = d^f^k, n = 1859775393) : 60 > c ? (m = d & f | k & (d | f), n = 2400959708) : (m = d^f^k, n = 3395469782), m = ((a<<5 | a>>>27) & 4294967295) + m + l + n + b[c] & 4294967295,
            l = k, k = f, f = (d<<30 | d>>>2) & 4294967295, d = a, a = m;
        e[0] = e[0] + a & 4294967295;
        e[1] = e[1] + d & 4294967295;
        e[2] = e[2] + f & 4294967295;
        e[3] = e[3] + k & 4294967295;
        e[4] = e[4] + l & 4294967295
    }
    function c(a, c) {
        if ("string" === typeof a) {
            a = (0, window.unescape)((0, window.encodeURIComponent)(a));
            for (var d = [], e = 0, g = a.length; e < g; ++e)
                d.push(a.charCodeAt(e));
            a = d
        }
        c || (c = a.length);
        d = 0;
        if (0 == m)
            for (; d + 64 < c;)
                b(a.slice(d, d + 64)), d += 64, n += 64;
        for (; d < c;)
            if (f[m++] = a[d++], n++, 64 == m)
                for (m = 0, b(f); d + 64 < c;)
                    b(a.slice(d, d + 64)), d += 64, n += 64
    }
    function d() {
        var a = [],
        d = 8 * n;
        56 > m ? c(k, 56 - m) : c(k, 64 - (m - 56));
        for (var g = 63; 56 <= g; g--)
            f[g] = d & 255, d>>>=8;
        b(f);
        for (g = d = 0; 5 > g; g++)
            for (var l = 24; 0 <= l; l -= 8)
                a[d++] = e[g]>>l & 255;
        return a
    }
    for (var e = [], f = [], g = [], k = [128], l = 1; 64 > l; ++l)
        k[l] = 0;
    var m, n;
    a();
    return {
        reset: a,
        update: c,
        M: d,
        Ef: function() {
            for (var a = d(), b = "", c = 0; c < a.length; c++)
                b += "0123456789ABCDEF".charAt(Math.floor(a[c] / 16)) + "0123456789ABCDEF".charAt(a[c]%16);
            return b
        }
    }
};
var Ij = function() {
    var a = window.__OVERRIDE_SID;
    null == a && (a = (new _.Dj(window.document)).get("SID"));
    return !!a
}, Kj = function(a, b, c) {
    if (1 == (_.ha(c) ? 2 : 1))
        return Jj([b, a].join(" "));
    var d = [], e = [];
    (0, _.dd)(c || [], function(a) {
        e.push(a.key);
        d.push(a.value)
    });
    c = (new Date).getTime();
    var f = [], f = 0 == d.length ? [c, b, a]: [d.join(":"), c, b, a];
    a = Jj(f.join(" "));
    a = [c, a];
    0 == e.length || a.push(e.join(""));
    return a.join("_")
}, Jj = function(a) {
    var b = Hj();
    b.update(a);
    return b.Ef().toLowerCase()
};
_.Lj = {
    Em: function(a) {
        var b = {
            SAPISIDHASH: !0,
            APISIDHASH: !0
        };
        return a && (a.OriginToken || a.Authorization && b[String(a.Authorization).split(" ")[0]])?!0 : !1
    },
    gH: Ij,
    $E: function() {
        var a = null;
        Ij() && (a = window.__PVT, null == a && (a = _.Cj.get("BEAT")));
        return a
    },
    Wl: function(a) {
        var b = _.Oi(String(window.location.href));
        if (Ij()) {
            var c = 0 == b.indexOf("https:") || 0 == b.indexOf("chrome-extension:"), d = c ? window.__SAPISID: window.__APISID;
            null == d && (d = (new _.Dj(window.document)).get(c ? "SAPISID" : "APISID"));
            if (d)
                return [c ? "SAPISIDHASH": "APISIDHASH", Kj(b, d, a)].join(" ")
        }
        return null
    }
};

_.H = _.H || {};
_.H.ll = function(a, b, c, d) {
    "undefined" != typeof a.addEventListener ? a.addEventListener(b, c, d) : "undefined" != typeof a.attachEvent ? a.attachEvent("on" + b, c) : _.Jb("cannot attachBrowserEvent: " + b)
};
_.H.uI = function(a) {
    var b = window;
    b.removeEventListener ? b.removeEventListener("mousemove", a, !1) : b.detachEvent ? b.detachEvent("onmousemove", a) : _.Jb("cannot removeBrowserEvent: mousemove")
};

_.Mj = function() {
    function a() {
        e[0] = 1732584193;
        e[1] = 4023233417;
        e[2] = 2562383102;
        e[3] = 271733878;
        e[4] = 3285377520;
        n = m = 0
    }
    function b(a) {
        for (var b = g, c = 0; 64 > c; c += 4)
            b[c / 4] = a[c]<<24 | a[c + 1]<<16 | a[c + 2]<<8 | a[c + 3];
        for (c = 16; 80 > c; c++)
            a = b[c - 3]^b[c - 8]^b[c - 14]^b[c - 16], b[c] = (a<<1 | a>>>31) & 4294967295;
        a = e[0];
        for (var d = e[1], f = e[2], k = e[3], l = e[4], m, n, c = 0; 80 > c; c++)
            40 > c ? 20 > c ? (m = k^d & (f^k), n = 1518500249) : (m = d^f^k, n = 1859775393) : 60 > c ? (m = d & f | k & (d | f), n = 2400959708) : (m = d^f^k, n = 3395469782), m = ((a<<5 | a>>>27) & 4294967295) + m + l + n + b[c] & 4294967295,
            l = k, k = f, f = (d<<30 | d>>>2) & 4294967295, d = a, a = m;
        e[0] = e[0] + a & 4294967295;
        e[1] = e[1] + d & 4294967295;
        e[2] = e[2] + f & 4294967295;
        e[3] = e[3] + k & 4294967295;
        e[4] = e[4] + l & 4294967295
    }
    function c(a, c) {
        if ("string" === typeof a) {
            a = (0, window.unescape)((0, window.encodeURIComponent)(a));
            for (var d = [], e = 0, g = a.length; e < g; ++e)
                d.push(a.charCodeAt(e));
            a = d
        }
        c || (c = a.length);
        d = 0;
        if (0 == m)
            for (; d + 64 < c;)
                b(a.slice(d, d + 64)), d += 64, n += 64;
        for (; d < c;)
            if (f[m++] = a[d++], n++, 64 == m)
                for (m = 0, b(f); d + 64 < c;)
                    b(a.slice(d, d + 64)), d += 64, n += 64
    }
    function d() {
        var a = [],
        d = 8 * n;
        56 > m ? c(k, 56 - m) : c(k, 64 - (m - 56));
        for (var g = 63; 56 <= g; g--)
            f[g] = d & 255, d>>>=8;
        b(f);
        for (g = d = 0; 5 > g; g++)
            for (var l = 24; 0 <= l; l -= 8)
                a[d++] = e[g]>>l & 255;
        return a
    }
    for (var e = [], f = [], g = [], k = [128], l = 1; 64 > l; ++l)
        k[l] = 0;
    var m, n;
    a();
    return {
        reset: a,
        update: c,
        M: d,
        Ef: function() {
            for (var a = d(), b = "", c = 0; c < a.length; c++)
                b += "0123456789ABCDEF".charAt(Math.floor(a[c] / 16)) + "0123456789ABCDEF".charAt(a[c]%16);
            return b
        }
    }
};
_.Nj = function() {
    function a(a) {
        var b = _.Mj();
        b.update(a);
        return b.Ef()
    }
    var b = window.crypto;
    if (b && "function" == typeof b.getRandomValues)
        return function() {
            var a = new window.Uint32Array(1);
            b.getRandomValues(a);
            return Number("0." + a[0])
        };
    var c = _.E("random/maxObserveMousemove");
    null == c && (c =- 1);
    var d = 0, e = Math.random(), f = 1, g = 1E6 * (window.screen.width * window.screen.width + window.screen.height), k = function(a) {
        a = a || window.event;
        var b = a.screenX + a.clientX<<16, b = b + (a.screenY + a.clientY), b = (new Date).getTime()%1E6 * b;
        f =
        f * b%g;
        0 < c&&++d == c && _.H.uI(k)
    };
    0 != c && _.H.ll(window, "mousemove", k, !1);
    var l = a(window.document.cookie + "|" + window.document.location + "|" + (new Date).getTime() + "|" + e);
    return function() {
        var b = f, b = b + (0, window.parseInt)(l.substr(0, 20), 16);
        l = a(l);
        return b / (g + Math.pow(16, 20))
    }
}();
_.C("shindig.random", _.Nj);
var Fd, Hd;
_.Dd = function() {};
_.Ed = function() {
    return _.Bd("Opera") || _.Bd("OPR")
};
Fd = function(a, b) {
    return a < b?-1 : a > b ? 1 : 0
};
_.Gd = function(a, b, c) {
    return 2 >= arguments.length ? _.ad.slice.call(a, b) : _.ad.slice.call(a, b, c)
};
Hd = "constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");
_.Id = function(a, b) {
    for (var c = 0, d = (0, _.Xc)(String(a)).split("."), e = (0, _.Xc)(String(b)).split("."), f = Math.max(d.length, e.length), g = 0; 0 == c && g < f; g++) {
        var k = d[g] || "", l = e[g] || "", m = RegExp("(\\d*)(\\D*)", "g"), n = RegExp("(\\d*)(\\D*)", "g");
        do {
            var p = m.exec(k) || ["", "", ""], v = n.exec(l) || ["", "", ""];
            if (0 == p[0].length && 0 == v[0].length)
                break;
            c = Fd(0 == p[1].length ? 0 : (0, window.parseInt)(p[1], 10), 0 == v[1].length ? 0 : (0, window.parseInt)(v[1], 10)) || Fd(0 == p[2].length, 0 == v[2].length) || Fd(p[2], v[2])
        }
        while (0 == c)
        }
    return c
};
_.Jd = function(a, b) {
    for (var c, d, e = 1; e < arguments.length; e++) {
        d = arguments[e];
        for (c in d)
            a[c] = d[c];
        for (var f = 0; f < Hd.length; f++)
            c = Hd[f], Object.prototype.hasOwnProperty.call(d, c) && (a[c] = d[c])
    }
};
_.Kd = function(a, b, c) {
    for (var d in a)
        b.call(c, a[d], d, a)
};
_.Ld = function(a) {
    var b = a.length;
    if (0 < b) {
        for (var c = Array(b), d = 0; d < b; d++)
            c[d] = a[d];
        return c
    }
    return []
};
var Wd, Xd, Zd, ae, ge, ke;
_.Sd = _.Ed();
_.M = _.Cd();
_.Td = _.Bd("Gecko")&&!_.kd(_.yd.toLowerCase(), "webkit")&&!(_.Bd("Trident") || _.Bd("MSIE"));
_.Ud = _.kd(_.yd.toLowerCase(), "webkit");
_.Vd = _.Ud && _.Bd("Mobile");
Wd = _.r.navigator || null;
Xd = Wd && Wd.platform || "";
_.Md = _.kd(Xd, "Mac");
_.Nd = _.kd(Xd, "Win");
_.Od = _.kd(Xd, "Linux");
var Yd = _.yd;
_.Pd=!!Yd && _.kd(Yd, "Android");
_.Qd=!!Yd && _.kd(Yd, "iPhone");
_.Rd=!!Yd && _.kd(Yd, "iPad");
Zd = _.r.navigator || null;
_.$d=!!Zd && _.kd(Zd.appVersion || "", "X11");
ae = function() {
    var a = _.r.document;
    return a ? a.documentMode : void 0
};
_.be = function() {
    var a = "", b;
    if (_.Sd && _.r.opera)
        return a = _.r.opera.version, _.Uc(a) ? a() : a;
    _.Td ? b = /rv\:([^\);]+)(\)|;)/ : _.M ? b = /\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/ : _.Ud && (b = /WebKit\/(\S+)/);
    b && (a = (a = b.exec(_.yd)) ? a[1] : "");
    return _.M && (b = ae(), b > (0, window.parseFloat)(a)) ? String(b) : a
}();
ge = {};
_.he = function(a) {
    return ge[a] || (ge[a] = 0 <= _.Id(_.be, a))
};
_.je = function(a) {
    return _.M && _.ie >= a
};
ke = _.r.document;
_.ie = ke && _.M ? ae() || ("CSS1Compat" == ke.compatMode ? (0, window.parseInt)(_.be, 10) : 5) : void 0;

var dh, fh, gh;
_.bh = function(a) {
    return /^[\s\xa0]*$/.test(a)
};
_.ch = function(a) {
    return _.bh(null == a ? "" : String(a))
};
dh = function(a) {
    return Array.prototype.join.call(arguments, "")
};
_.eh = function(a, b) {
    var c = a.length - b.length;
    return 0 <= c && a.indexOf(b, c) == c
};
fh = /^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#(.*))?$/;
_.hh = function(a) {
    if (gh) {
        gh=!1;
        var b = _.r.location;
        if (b) {
            var c = b.href;
            if (c && (c = (c = _.hh(c)[3] || null) ? (0, window.decodeURI)(c) : c) && c != b.hostname)
                throw gh=!0, Error();
        }
    }
    return a.match(fh)
};
gh = _.Ud;
_.ih = function(a) {
    if (a[1]) {
        var b = a[0], c = b.indexOf("#");
        0 <= c && (a.push(b.substr(c)), a[0] = b = b.substr(0, c));
        c = b.indexOf("?");
        0 > c ? a[1] = "?" : c == b.length - 1 && (a[1] = void 0)
    }
    return a.join("")
};
_.jh = function(a, b, c) {
    if (_.ha(b))
        for (var d = 0; d < b.length; d++)
            _.jh(a, String(b[d]), c);
    else 
        null != b && c.push("&", a, "" === b ? "" : "=", (0, window.encodeURIComponent)(String(b)))
};
_.kh = function(a, b) {
    for (var c in b)
        _.jh(c, b[c], a);
    return a
};
_.lh = function(a, b) {
    _.eh(a, "/") && (a = a.substr(0, a.length - 1));
    _.Wc(b, "/") && (b = b.substr(1));
    return dh(a, "/", b)
};

var Oj;
Oj = function(a, b) {
    var c = _.H.Vo({
        id: a,
        name: a
    });
    c.style.width = "1px";
    c.style.height = "1px";
    c.style.position = "absolute";
    c.style.top = "-100px";
    var d;
    if (window.navigator) {
        d = window.navigator.userAgent || "";
        var e = window.navigator.product || "";
        d = 0 != d.indexOf("Opera")&&-1 == d.indexOf("WebKit") && "Gecko" == e && 0 < d.indexOf("rv:1.")
    } else 
        d=!1;
    c.src = d ? "about:blank" : b;
    c.tabIndex = "-1";
    window.document.body.appendChild(c);
    d && (c.src = b);
    return c
};
_.Pj = function() {
    function a() {
        return !!m("auth/useFirstPartyAuthV2")
    }
    function b(a, b, c, d) {
        var e = m("proxy");
        if (c ||!e)
            var e = m("root"), f = m("root-1p") || e, g = m("xd3"), e = (c || String(b ? f : e)) + g;
        (b = _.H.Fb().jsh || _.Mi.Jo.yv()) && (e += (0 <= e.indexOf("?") ? "&" : "?") + "jsh=" + (0, window.encodeURIComponent)(b));
        m("push") && (e += (0 <= e.indexOf("?") ? "&" : "?") + "p=1");
        e += "#parent=" + (0, window.encodeURIComponent)(null != d ? String(d) : _.Mi.Ba(window.document.location.href));
        return e + ("&rpctoken=" + a)
    }
    function c(a, b, c, d) {
        var f = e(c, d);
        p[f] ||
        (c = Oj(f, b), _.J.register("ready:" + a, function() {
            _.J.uh("ready:" + a);
            if (!v[f]) {
                v[f]=!0;
                var b = x[f];
                x[f] = [];
                for (var c = 0, d = b.length; c < d; ++c) {
                    var e = b[c];
                    k(e.jh, e.BI, e.nc)
                }
            }
        }), _.J.Dk(f, b), p[f] = c)
    }
    function d(a, d) {
        var e = String(2147483647 * (0, _.Nj)() | 0), f = b(e, a, d);
        _.cc(function() {
            c(e, f, a, d)
        })
    }
    function e(a, c) {
        var d = b("", a, c, ""), e = n[d];
        e || (e = _.Mj(), e.update(d), e = e.Ef().toLowerCase(), e += Math.random(), n[d] = e);
        return "apiproxy" + e
    }
    function f(a) {
        if (!a)
            return null;
        var b = {}, c = a.headers || {}, d;
        for (d in c)({})
            .hasOwnProperty.call(c,
            d) && (b[d] = c[d]);
        b["Content-Type"] = "application/json";
        return [{
            key: "gapiRequest",
            params: {
                id: "gapiRequest",
                key: "gapiRequest",
                httpMethod: "POST",
                root: a.root,
                url: "/rpc",
                urlParams: a.urlParams,
                headers: b,
                clientName: a.clientName || null,
                clientVersion: a.clientVersion || null,
                body: a.requests || []
            }
        }
        ]
    }
    function g(a) {
        return function(b) {
            b = b.gapiRequest.data.body;
            var c = (0, _.ac)(b);
            if (c) {
                for (var d = {}, e = 0, f = c.length; e < f; ++e)
                    d[c[e].id] = c[e];
                a(d, b)
            } else 
                a(c, b)
        }
    }
    function k(a, b, c) {
        var k = void 0, m=!1;
        if ("makeRequest" !== a && "makeHttpRequests" !==
        a)
            throw 'only "makeRequest" and "makeHttpRequests" RPCs are implemented';
        "makeRequest" === a && (a = "makeHttpRequests", b = f(b), c = g(c));
        var n = function(a) {
            if (a) {
                if ("undefined" != typeof k && "undefined" != typeof a.root && k != a.root)
                    throw "all requests in a batch must have the same root URL";
                k = a.root || k;
                m = _.Lj.Em(a.headers)
            }
        };
        if (b) {
            for (var G = _.E("client/jsonpOverride"), I = 0, U = b.length; I < U; ++I) {
                var Z = b[I];
                Z && (Z = Z.params, n(Z), G && l(Z, c))
            }
            if (G)
                return 
        }
        var $ = e(m, k);
        p[$] || d(m, k);
        v[$] ? _.J.call($, a, function(a) {
            if (this.f == $ &&
            this.t == _.J.Ag(this.f) && this.origin == _.J.Of(this.f)) {
                var b = (0, _.ac)(a);
                c(b, a)
            }
        }, b) : (x[$] || (x[$] = []), x[$].push({
            jh: a,
            BI: b,
            nc: c
        }))
    }
    function l(a, b) {
        if ("GET" != a.httpMethod)
            throw "JSONP supports GET methods only.";
        var c = "jpcb" + String(2147483647 * (0, _.Nj)() | 0), d = window.document.createElement("script"), e = window.document.getElementsByTagName("head")[0];
        window[c] = function(a) {
            b(a);
            try {
                delete window[c]
            } catch (f) {
                window[c] = void 0
            }
            window.setTimeout(function() {
                e.removeChild(d)
            }, 1)
        };
        var f = a.root || m("root"), g = a.urlParams;
        g.callback = c;
        f = _.lh(f, a.url);
        g = _.ih(_.kh([f], g));
        d.setAttribute("id", c);
        d.setAttribute("src", g);
        d.setAttribute("charset", "utf-8");
        e.appendChild(d)
    }
    function m(a) {
        return _.E("googleapis.config/" + a)
    }
    var n = {}, p = {}, v = {}, x = {};
    return {
        fl: function(b, c, d) {
            var e = d || "auto";
            b = b || {};
            if ("none" == e)
                return b;
            c = c || window.location.href;
            d = b.Authorization;
            var f = b.OriginToken;
            if (!d&&!f) {
                f = _.Bj(void 0, void 0);
                !f && window.gapi.auth2 && "function" == typeof window.gapi.auth2._gt && (f = window.gapi.auth2._gt());
                f && f.access_token &&
                ("oauth2" == e || "auto" == e) && (d = String(f.token_type || "Bearer") + " " + f.access_token);
                if (f=!d)
                    f = (!!m("auth/useFirstPartyAuth") || "1p" == e) && "oauth2" != e;
                if (f && _.Lj.gH()) {
                    if (a()) {
                        d = m("primaryEmail");
                        var e = m("appDomain"), f = m("fogId"), g = [];
                        d && g.push({
                            key: "e",
                            value: d
                        });
                        e && g.push({
                            key: "a",
                            value: e
                        });
                        f && g.push({
                            key: "u",
                            value: f
                        });
                        d = _.Lj.Wl(g)
                    } else 
                        d = _.Lj.Wl();
                    d && (c = _.google.$p(c), c = b["X-Goog-AuthUser"] || c, _.ch(c) && (!a() || a() && _.ch(m("primaryEmail")) && _.ch(m("appDomain")) && _.ch(m("fogId"))) && (c = "0"), _.ch(c) || (b["X-Goog-AuthUser"] = c))
                }
                d ? b.Authorization = d : !1 !== m("auth/useOriginToken") && (f = _.Lj.$E()) && (b.OriginToken = f)
            }
            return b
        },
        A: l,
        fg: k
    }
}();

_.pe = function(a, b) {
    var c;
    t: {
        c = a.length;
        for (var d = _.ja(a) ? a.split("") : a, e = 0; e < c; e++)
            if (e in d && b.call(void 0, d[e], e, a)) {
                c = e;
                break t
            }
        c =- 1
    }
    return 0 > c ? null : _.ja(a) ? a.charAt(c) : a[c]
};
_.qe = function(a, b) {
    for (var c = a.split("."), d = b || _.r, e; e = c.shift();)
        if (null != d[e])
            d = d[e];
        else 
            return null;
    return d
};
_.re = function(a, b) {
    return 1 == _.ad.splice.call(a, b, 1).length
};
_.se = function(a, b) {
    var c = (0, _.bd)(a, b), d;
    (d = 0 <= c) && _.re(a, c);
    return d
};
_.te = function() {
    this.ce = this.ce;
    this.Sa = this.Sa
};
_.te.prototype.ce=!1;
_.te.prototype.ob = function() {
    return this.ce
};
_.te.prototype.ha = function() {
    this.ce || (this.ce=!0, this.W())
};
_.ve = function(a, b) {
    var c = _.ra(_.ue, b);
    a.Sa || (a.Sa = []);
    a.Sa.push(_.ca(void 0) ? (0, _.s)(c, void 0) : c)
};
_.te.prototype.W = function() {
    if (this.Sa)
        for (; this.Sa.length;)
            this.Sa.shift()()
};
_.ue = function(a) {
    a && "function" == typeof a.ha && a.ha()
};
_.we = function(a, b) {
    this.type = a;
    this.currentTarget = this.target = b;
    this.defaultPrevented = this.eh=!1;
    this.Oy=!0
};
_.we.prototype.W = function() {};
_.we.prototype.ha = function() {};
_.we.prototype.stopPropagation = function() {
    this.eh=!0
};
_.we.prototype.preventDefault = function() {
    this.defaultPrevented=!0;
    this.Oy=!1
};
var xe = function(a) {
    xe[" "](a);
    return a
};
xe[" "] = _.Dd;
var ye=!_.M || _.je(9), ze=!_.M || _.je(9), Be = _.M&&!_.he("9");
!_.Ud || _.he("528");
_.Td && _.he("1.9b") || _.M && _.he("8") || _.Sd && _.he("9.5") || _.Ud && _.he("528");
_.Td&&!_.he("8") || _.M && _.he("9");
_.Ce = _.M ? "focusin" : "DOMFocusIn";
_.De = _.M ? "focusout" : "DOMFocusOut";
_.Ee = _.Ud ? "webkitTransitionEnd" : _.Sd ? "otransitionend" : "transitionend";
_.Fe = function(a, b) {
    _.we.call(this, a ? a.type : "");
    this.relatedTarget = this.currentTarget = this.target = null;
    this.charCode = this.keyCode = this.button = this.screenY = this.screenX = this.clientY = this.clientX = this.offsetY = this.offsetX = 0;
    this.metaKey = this.shiftKey = this.altKey = this.ctrlKey=!1;
    this.state = null;
    this.jn=!1;
    this.A = null;
    a && this.gb(a, b)
};
_.u(_.Fe, _.we);
var Ge = [1, 4, 2];
_.Fe.prototype.gb = function(a, b) {
    var c = this.type = a.type;
    this.target = a.target || a.srcElement;
    this.currentTarget = b;
    var d = a.relatedTarget;
    if (d) {
        if (_.Td) {
            var e;
            t:
            {
                try {
                    xe(d.nodeName);
                    e=!0;
                    break t
                } catch (f) {}
                e=!1
            }
            e || (d = null)
        }
    } else 
        "mouseover" == c ? d = a.fromElement : "mouseout" == c && (d = a.toElement);
    this.relatedTarget = d;
    this.offsetX = _.Ud || void 0 !== a.offsetX ? a.offsetX : a.layerX;
    this.offsetY = _.Ud || void 0 !== a.offsetY ? a.offsetY : a.layerY;
    this.clientX = void 0 !== a.clientX ? a.clientX : a.pageX;
    this.clientY = void 0 !== a.clientY ? a.clientY :
    a.pageY;
    this.screenX = a.screenX || 0;
    this.screenY = a.screenY || 0;
    this.button = a.button;
    this.keyCode = a.keyCode || 0;
    this.charCode = a.charCode || ("keypress" == c ? a.keyCode : 0);
    this.ctrlKey = a.ctrlKey;
    this.altKey = a.altKey;
    this.shiftKey = a.shiftKey;
    this.metaKey = a.metaKey;
    this.jn = _.Md ? a.metaKey : a.ctrlKey;
    this.state = a.state;
    this.A = a;
    a.defaultPrevented && this.preventDefault()
};
_.He = function(a) {
    return (ye ? 0 == a.A.button : "click" == a.type?!0 : !!(a.A.button & Ge[0]))&&!(_.Ud && _.Md && a.ctrlKey)
};
_.Fe.prototype.stopPropagation = function() {
    _.Fe.J.stopPropagation.call(this);
    this.A.stopPropagation ? this.A.stopPropagation() : this.A.cancelBubble=!0
};
_.Fe.prototype.preventDefault = function() {
    _.Fe.J.preventDefault.call(this);
    var a = this.A;
    if (a.preventDefault)
        a.preventDefault();
    else if (a.returnValue=!1, Be)
        try {
            if (a.ctrlKey || 112 <= a.keyCode && 123 >= a.keyCode)
                a.keyCode =- 1
    } catch (b) {}
};
_.Fe.prototype.pE = function() {
    return this.A
};
_.Fe.prototype.W = function() {};
var Ie, Ke;
Ie = "closure_listenable_" + (1E6 * Math.random() | 0);
_.Je = function(a) {
    return !(!a ||!a[Ie])
};
Ke = 0;
var Le = function(a, b, c, d, e) {
    this.Zg = a;
    this.A = null;
    this.src = b;
    this.type = c;
    this.wl=!!d;
    this.Kc = e;
    this.key=++Ke;
    this.Hi = this.ul=!1
}, Me = function(a) {
    a.Hi=!0;
    a.Zg = null;
    a.A = null;
    a.src = null;
    a.Kc = null
};
var Ne = function(a) {
    this.src = a;
    this.A = {};
    this.B = 0
}, Pe, Oe;
Ne.prototype.add = function(a, b, c, d, e) {
    var f = a.toString();
    a = this.A[f];
    a || (a = this.A[f] = [], this.B++);
    var g = Oe(a, b, d, e);
    - 1 < g ? (b = a[g], c || (b.ul=!1)) : (b = new Le(b, this.src, f, !!d, e), b.ul = c, a.push(b));
    return b
};
Ne.prototype.remove = function(a, b, c, d) {
    a = a.toString();
    if (!(a in this.A))
        return !1;
    var e = this.A[a];
    b = Oe(e, b, c, d);
    return - 1 < b ? (Me(e[b]), _.re(e, b), 0 == e.length && (delete this.A[a], this.B--), !0) : !1
};
Pe = function(a, b) {
    var c = b.type;
    if (!(c in a.A))
        return !1;
    var d = _.se(a.A[c], b);
    d && (Me(b), 0 == a.A[c].length && (delete a.A[c], a.B--));
    return d
};
_.Qe = function(a, b, c, d, e) {
    a = a.A[b.toString()];
    b =- 1;
    a && (b = Oe(a, c, d, e));
    return - 1 < b ? a[b] : null
};
Oe = function(a, b, c, d) {
    for (var e = 0; e < a.length; ++e) {
        var f = a[e];
        if (!f.Hi && f.Zg == b && f.wl==!!c && f.Kc == d)
            return e
    }
    return - 1
};
var Re, Se, Te, Xe, Ye, cf, bf, Ze, df;
Re = "closure_lm_" + (1E6 * Math.random() | 0);
Se = {};
Te = 0;
_.N = function(a, b, c, d, e) {
    if (_.ha(b)) {
        for (var f = 0; f < b.length; f++)
            _.N(a, b[f], c, d, e);
        return null
    }
    c = _.Ue(c);
    return _.Je(a) ? a.P(b, c, d, e) : _.Ve(a, b, c, !1, d, e)
};
_.Ve = function(a, b, c, d, e, f) {
    if (!b)
        throw Error("k");
    var g=!!e, k = _.We(a);
    k || (a[Re] = k = new Ne(a));
    c = k.add(b, c, d, e, f);
    if (c.A)
        return c;
    d = Xe();
    c.A = d;
    d.src = a;
    d.Zg = c;
    a.addEventListener ? a.addEventListener(b.toString(), d, g) : a.attachEvent(Ye(b.toString()), d);
    Te++;
    return c
};
Xe = function() {
    var a = Ze, b = ze ? function(c) {
        return a.call(b.src, b.Zg, c)
    }
    : function(c) {
        c = a.call(b.src, b.Zg, c);
        if (!c)
            return c
    };
    return b
};
_.$e = function(a, b, c, d, e) {
    if (_.ha(b))
        for (var f = 0; f < b.length; f++)
            _.$e(a, b[f], c, d, e);
    else 
        c = _.Ue(c), _.Je(a) ? a.La(b, c, d, e) : a && (a = _.We(a)) && (b = _.Qe(a, b, c, !!d, e)) && _.af(b)
};
_.af = function(a) {
    if (_.Vc(a) ||!a || a.Hi)
        return !1;
    var b = a.src;
    if (_.Je(b))
        return Pe(b.Ee, a);
    var c = a.type, d = a.A;
    b.removeEventListener ? b.removeEventListener(c, d, a.wl) : b.detachEvent && b.detachEvent(Ye(c), d);
    Te--;
    (c = _.We(b)) ? (Pe(c, a), 0 == c.B && (c.src = null, b[Re] = null)) : Me(a);
    return !0
};
Ye = function(a) {
    return a in Se ? Se[a] : Se[a] = "on" + a
};
cf = function(a, b, c, d) {
    var e = 1;
    if (a = _.We(a))
        if (b = a.A[b.toString()])
            for (b = b.concat(), a = 0; a < b.length; a++) {
                var f = b[a];
                f && f.wl == c&&!f.Hi && (e&=!1 !== bf(f, d))
            }
    return Boolean(e)
};
bf = function(a, b) {
    var c = a.Zg, d = a.Kc || a.src;
    a.ul && _.af(a);
    return c.call(d, b)
};
Ze = function(a, b) {
    if (a.Hi)
        return !0;
    if (!ze) {
        var c = b || _.qe("window.event"), d = new _.Fe(c, this), e=!0;
        if (!(0 > c.keyCode || void 0 != c.returnValue)) {
            t:
            {
                var f=!1;
                if (0 == c.keyCode)
                    try {
                        c.keyCode =- 1;
                        break t
                } catch (g) {
                    f=!0
                }
                if (f || void 0 == c.returnValue)
                    c.returnValue=!0
            }
            c = [];
            for (f = d.currentTarget; f; f = f.parentNode)
                c.push(f);
            for (var f = a.type, k = c.length - 1; !d.eh && 0 <= k; k--)
                d.currentTarget = c[k], e&=cf(c[k], f, !0, d);
            for (k = 0; !d.eh && k < c.length; k++)
                d.currentTarget = c[k], e&=cf(c[k], f, !1, d)
            }
        return e
    }
    return bf(a, new _.Fe(b, this))
};
_.We = function(a) {
    a = a[Re];
    return a instanceof Ne ? a : null
};
df = "__closure_events_fn_" + (1E9 * Math.random()>>>0);
_.Ue = function(a) {
    if (_.Uc(a))
        return a;
    a[df] || (a[df] = function(b) {
        return a.handleEvent(b)
    });
    return a[df]
};
_.oe(function(a) {
    Ze = a(Ze)
});
_.ef = function() {
    _.te.call(this);
    this.Ee = new Ne(this);
    this.lr = this;
    this.td = null
};
_.u(_.ef, _.te);
_.ef.prototype[Ie]=!0;
_.h = _.ef.prototype;
_.h.$h = function() {
    return this.td
};
_.h.lh = _.q(1);
_.h.addEventListener = function(a, b, c, d) {
    _.N(this, a, b, c, d)
};
_.h.removeEventListener = function(a, b, c, d) {
    _.$e(this, a, b, c, d)
};
_.h.dispatchEvent = function(a) {
    var b, c = this.$h();
    if (c)
        for (b = []; c; c = c.$h()
            )b.push(c);
    var c = this.lr, d = a.type || a;
    if (_.ja(a))
        a = new _.we(a, c);
    else if (a instanceof _.we)
        a.target = a.target || c;
    else {
        var e = a;
        a = new _.we(d, c);
        _.Jd(a, e)
    }
    var e=!0, f;
    if (b)
        for (var g = b.length - 1; !a.eh && 0 <= g; g--)
            f = a.currentTarget = b[g], e = ff(f, d, !0, a) && e;
    a.eh || (f = a.currentTarget = c, e = ff(f, d, !0, a) && e, a.eh || (e = ff(f, d, !1, a) && e));
    if (b)
        for (g = 0; !a.eh && g < b.length; g++)
            f = a.currentTarget = b[g], e = ff(f, d, !1, a) && e;
    return e
};
_.h.W = function() {
    _.ef.J.W.call(this);
    if (this.Ee) {
        var a = this.Ee, b = 0, c;
        for (c in a.A) {
            for (var d = a.A[c], e = 0; e < d.length; e++)
                ++b, Me(d[e]);
            delete a.A[c];
            a.B--
        }
    }
    this.td = null
};
_.h.P = function(a, b, c, d) {
    return this.Ee.add(String(a), b, !1, c, d)
};
_.h.La = function(a, b, c, d) {
    return this.Ee.remove(String(a), b, c, d)
};
var ff = function(a, b, c, d) {
    b = a.Ee.A[String(b)];
    if (!b)
        return !0;
    b = b.concat();
    for (var e=!0, f = 0; f < b.length; ++f) {
        var g = b[f];
        if (g&&!g.Hi && g.wl == c) {
            var k = g.Zg, l = g.Kc || g.src;
            g.ul && Pe(a.Ee, g);
            e=!1 !== k.call(l, d) && e
        }
    }
    return e && 0 != d.Oy
};

_.gf = function(a) {
    if ("function" == typeof a.xc)
        return a.xc();
    if (_.ja(a))
        return a.split("");
    if (_.Qc(a)) {
        for (var b = [], c = a.length, d = 0; d < c; d++)
            b.push(a[d]);
        return b
    }
    return _.ud(a)
};
_.hf = function(a) {
    if ("function" == typeof a.fd)
        return a.fd();
    if ("function" != typeof a.xc) {
        if (_.Qc(a) || _.ja(a)) {
            var b = [];
            a = a.length;
            for (var c = 0; c < a; c++)
                b.push(c);
            return b
        }
        return _.vd(a)
    }
};
_.jf = function(a, b, c) {
    if ("function" == typeof a.forEach)
        a.forEach(b, c);
    else if (_.Qc(a) || _.ja(a))(0, _.dd)(a, b, c);
    else 
        for (var d = _.hf(a), e = _.gf(a), f = e.length, g = 0; g < f; g++)
            b.call(c, e[g], d && d[g], a)
};

_.kf = "StopIteration"in _.r ? _.r.StopIteration : Error("l");
_.lf = function() {};
_.lf.prototype.next = function() {
    throw _.kf;
};
_.lf.prototype.gf = function() {
    return this
};
_.mf = function(a, b) {
    this.B = {};
    this.A = [];
    this.D = this.C = 0;
    var c = arguments.length;
    if (1 < c) {
        if (c%2)
            throw Error("j");
        for (var d = 0; d < c; d += 2)
            this.set(arguments[d], arguments[d + 1])
    } else if (a) {
        a instanceof _.mf ? (c = a.fd(), d = a.xc()) : (c = _.vd(a), d = _.ud(a));
        for (var e = 0; e < c.length; e++)
            this.set(c[e], d[e])
    }
};
_.h = _.mf.prototype;
_.h.fb = function() {
    return this.C
};
_.h.xc = function() {
    nf(this);
    for (var a = [], b = 0; b < this.A.length; b++)
        a.push(this.B[this.A[b]]);
    return a
};
_.h.fd = function() {
    nf(this);
    return this.A.concat()
};
_.h.Mh = _.q(2);
_.h.equals = function(a, b) {
    if (this === a)
        return !0;
    if (this.C != a.fb())
        return !1;
    var c = b || of;
    nf(this);
    for (var d, e = 0; d = this.A[e]; e++)
        if (!c(this.get(d), a.get(d)))
            return !1;
    return !0
};
var of = function(a, b) {
    return a === b
};
_.mf.prototype.isEmpty = function() {
    return 0 == this.C
};
_.mf.prototype.clear = function() {
    this.B = {};
    this.D = this.C = this.A.length = 0
};
_.mf.prototype.remove = function(a) {
    return _.pf(this.B, a) ? (delete this.B[a], this.C--, this.D++, this.A.length > 2 * this.C && nf(this), !0) : !1
};
var nf = function(a) {
    if (a.C != a.A.length) {
        for (var b = 0, c = 0; b < a.A.length;) {
            var d = a.A[b];
            _.pf(a.B, d) && (a.A[c++] = d);
            b++
        }
        a.A.length = c
    }
    if (a.C != a.A.length) {
        for (var e = {}, c = b = 0; b < a.A.length;)
            d = a.A[b], _.pf(e, d) || (a.A[c++] = d, e[d] = 1), b++;
        a.A.length = c
    }
};
_.h = _.mf.prototype;
_.h.get = function(a, b) {
    return _.pf(this.B, a) ? this.B[a] : b
};
_.h.set = function(a, b) {
    _.pf(this.B, a) || (this.C++, this.A.push(a), this.D++);
    this.B[a] = b
};
_.h.forEach = function(a, b) {
    for (var c = this.fd(), d = 0; d < c.length; d++) {
        var e = c[d], f = this.get(e);
        a.call(b, f, e, this)
    }
};
_.h.clone = function() {
    return new _.mf(this)
};
_.h.gf = function(a) {
    nf(this);
    var b = 0, c = this.A, d = this.B, e = this.D, f = this, g = new _.lf;
    g.next = function() {
        for (; ;) {
            if (e != f.D)
                throw Error("n");
            if (b >= c.length)
                throw _.kf;
            var g = c[b++];
            return a ? g : d[g]
        }
    };
    return g
};
_.pf = function(a, b) {
    return Object.prototype.hasOwnProperty.call(a, b)
};

var rf, sf;
_.mf.prototype.Mh = _.L(2, function(a) {
    for (var b = 0; b < this.A.length; b++) {
        var c = this.A[b];
        if (_.pf(this.B, c) && this.B[c] == a)
            return !0
    }
    return !1
});
_.qf = function(a) {
    var b = 0, c;
    for (c in a)
        b++;
    return b
};
rf = function(a, b) {
    if ("function" == typeof a.every)
        return a.every(b, void 0);
    if (_.Qc(a) || _.ja(a))
        return (0, _.hd)(a, b, void 0);
    for (var c = _.hf(a), d = _.gf(a), e = d.length, f = 0; f < e; f++)
        if (!b.call(void 0, d[f], c && c[f], a))
            return !1;
    return !0
};
sf = function(a) {
    return "function" == typeof a.fb ? a.fb() : _.Qc(a) || _.ja(a) ? a.length : _.qf(a)
};
var vf;
_.uf = function(a) {
    this.A = new _.mf;
    if (a) {
        a = _.gf(a);
        for (var b = a.length, c = 0; c < b; c++)
            this.add(a[c])
    }
};
vf = function(a) {
    var b = typeof a;
    return "object" == b && a || "function" == b ? "o" + _.Sc(a) : b.substr(0, 1) + a
};
_.h = _.uf.prototype;
_.h.fb = function() {
    return this.A.fb()
};
_.h.add = function(a) {
    this.A.set(vf(a), a)
};
_.h.remove = function(a) {
    return this.A.remove(vf(a))
};
_.h.clear = function() {
    this.A.clear()
};
_.h.isEmpty = function() {
    return this.A.isEmpty()
};
_.h.contains = function(a) {
    a = vf(a);
    return _.pf(this.A.B, a)
};
_.h.xc = function() {
    return this.A.xc()
};
_.h.clone = function() {
    return new _.uf(this)
};
_.h.equals = function(a) {
    return this.fb() == sf(a) && wf(this, a)
};
var wf = function(a, b) {
    var c = sf(b);
    if (a.fb() > c)
        return !1;
    !("undefined" != typeof _.uf && b instanceof _.uf) && 5 < c && (b = new _.uf(b));
    return rf(a, function(a) {
        var c = b;
        return "function" == typeof c.contains ? c.contains(a) : "function" == typeof c.Mh ? c.Mh(a) : _.Qc(c) || _.ja(c) ? _.id(c, a) : _.wd(c, a)
    })
};
_.uf.prototype.gf = function() {
    return this.A.gf(!1)
};

var Ng, Tg, Vg, Wg, Ug;
Ng = function(a) {
    return /^\s*$/.test(a)?!1 : /^[\],:{}\s\u2028\u2029]*$/.test(a.replace(/\\["\\\/bfnrtu]/g, "@").replace(/"[^"\\\n\r\u2028\u2029\x00-\x08\x0a-\x1f]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:[\s\u2028\u2029]*\[)+/g, ""))
};
_.Og = function(a) {
    a = String(a);
    if (Ng(a))
        try {
            return eval("(" + a + ")")
    } catch (b) {}
    throw Error("o`" + a);
};
_.Pg = function(a) {
    return eval("(" + a + ")")
};
_.Sg = function(a) {
    return _.Qg(new _.Rg(void 0), a)
};
_.Rg = function(a) {
    this.A = a
};
_.Qg = function(a, b) {
    var c = [];
    Tg(a, b, c);
    return c.join("")
};
Tg = function(a, b, c) {
    switch (typeof b) {
    case "string":
        Ug(b, c);
        break;
    case "number":
        c.push((0, window.isFinite)(b)&&!(0, window.isNaN)(b) ? b : "null");
        break;
    case "boolean":
        c.push(b);
        break;
    case "undefined":
        c.push("null");
        break;
    case "object":
        if (null == b) {
            c.push("null");
            break
        }
        if (_.ha(b)) {
            var d = b.length;
            c.push("[");
            for (var e = "", f = 0; f < d; f++)
                c.push(e), e = b[f], Tg(a, a.A ? a.A.call(b, String(f), e) : e, c), e = ",";
            c.push("]");
            break
        }
        c.push("{");
        d = "";
        for (f in b)
            Object.prototype.hasOwnProperty.call(b, f) && (e = b[f], "function" != typeof e &&
            (c.push(d), Ug(f, c), c.push(":"), Tg(a, a.A ? a.A.call(b, f, e) : e, c), d = ","));
        c.push("}");
        break;
    case "function":
        break;
    default:
        throw Error("p`" + typeof b);
    }
};
Vg = {
    '"': '\\"',
    "\\": "\\\\",
    "/": "\\/",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\x0B": "\\u000b"
};
Wg = /\uffff/.test("\uffff") ? /[\\\"\x00-\x1f\x7f-\uffff]/g : /[\\\"\x00-\x1f\x7f-\xff]/g;
Ug = function(a, b) {
    b.push('"', a.replace(Wg, function(a) {
        if (a in Vg)
            return Vg[a];
        var b = a.charCodeAt(0), e = "\\u";
        16 > b ? e += "000" : 256 > b ? e += "00" : 4096 > b && (e += "0");
        return Vg[a] = e + b.toString(16)
    }), '"')
};

_.Xg = function(a) {
    for (var b in a)
        return !1;
    return !0
};
_.Yg = function(a, b) {
    _.ef.call(this);
    this.B = a || 1;
    this.A = b || _.r;
    this.C = (0, _.s)(this.XJ, this);
    this.D = (0, _.sa)()
};
_.u(_.Yg, _.ef);
_.h = _.Yg.prototype;
_.h.enabled=!1;
_.h.Nd = null;
_.h.setInterval = function(a) {
    this.B = a;
    this.Nd && this.enabled ? (_.Zg(this), this.start()) : this.Nd && _.Zg(this)
};
_.h.XJ = function() {
    if (this.enabled) {
        var a = (0, _.sa)() - this.D;
        0 < a && a < .8 * this.B ? this.Nd = this.A.setTimeout(this.C, this.B - a) : (this.Nd && (this.A.clearTimeout(this.Nd), this.Nd = null), this.dispatchEvent("tick"), this.enabled && (this.Nd = this.A.setTimeout(this.C, this.B), this.D = (0, _.sa)()))
    }
};
_.h.start = function() {
    this.enabled=!0;
    this.Nd || (this.Nd = this.A.setTimeout(this.C, this.B), this.D = (0, _.sa)())
};
_.Zg = function(a) {
    a.enabled=!1;
    a.Nd && (a.A.clearTimeout(a.Nd), a.Nd = null)
};
_.Yg.prototype.W = function() {
    _.Yg.J.W.call(this);
    _.Zg(this);
    delete this.A
};
_.$g = function(a, b, c) {
    if (_.Uc(a))
        c && (a = (0, _.s)(a, c));
    else if (a && "function" == typeof a.handleEvent)
        a = (0, _.s)(a.handleEvent, a);
    else 
        throw Error("q");
    return 2147483647 < b?-1 : _.r.setTimeout(a, b || 0)
};
_.ah = function(a) {
    _.r.clearTimeout(a)
};

_.mh = function(a) {
    switch (a) {
    case 200:
    case 201:
    case 202:
    case 204:
    case 206:
    case 304:
    case 1223:
        return !0;
    default:
        return !1
    }
};
var nh = function() {};
nh.prototype.A = null;
var ph = function(a) {
    var b;
    (b = a.A) || (b = {}, oh(a) && (b[0]=!0, b[1]=!0), b = a.A = b);
    return b
};
var rh, oh;
rh = function() {};
_.u(rh, nh);
_.sh = function(a) {
    return (a = oh(a)) ? new window.ActiveXObject(a) : new window.XMLHttpRequest
};
oh = function(a) {
    if (!a.B && "undefined" == typeof window.XMLHttpRequest && "undefined" != typeof window.ActiveXObject) {
        for (var b = ["MSXML2.XMLHTTP.6.0", "MSXML2.XMLHTTP.3.0", "MSXML2.XMLHTTP", "Microsoft.XMLHTTP"], c = 0; c < b.length; c++) {
            var d = b[c];
            try {
                return new window.ActiveXObject(d), a.B = d
            } catch (e) {}
        }
        throw Error("r");
    }
    return a.B
};
_.qh = new rh;
var uh, vh, wh;
_.th = function(a) {
    _.ef.call(this);
    this.headers = new _.mf;
    this.R = a || null;
    this.B=!1;
    this.Y = this.A = null;
    this.N = "";
    this.G = 0;
    this.C = this.da = this.V = this.U=!1;
    this.F = 0;
    this.K = null;
    this.D = "";
    this.ka = this.pa=!1
};
_.u(_.th, _.ef);
uh = /^https?$/i;
vh = ["POST", "PUT"];
wh = [];
_.xh = function(a, b, c, d, e, f) {
    var g = new _.th;
    wh.push(g);
    b && g.P("complete", b);
    g.Ee.add("ready", g.va, !0, void 0, void 0);
    f && (g.F = Math.max(0, f));
    g.send(a, c, d, e)
};
_.th.prototype.va = function() {
    this.ha();
    _.se(wh, this)
};
_.th.prototype.Zp = function() {
    return this.D
};
_.th.prototype.send = function(a, b, c, d) {
    if (this.A)
        throw Error("s`" + this.N + "`" + a);
    b = b ? b.toUpperCase() : "GET";
    this.N = a;
    this.G = 0;
    this.U=!1;
    this.B=!0;
    this.A = this.R ? _.sh(this.R) : _.sh(_.qh);
    this.Y = this.R ? ph(this.R) : ph(_.qh);
    this.A.onreadystatechange = (0, _.s)(this.wa, this);
    try {
        this.da=!0, this.A.open(b, String(a), !0), this.da=!1
    } catch (e) {
        yh(this);
        return 
    }
    a = c || "";
    var f = this.headers.clone();
    d && _.jf(d, function(a, b) {
        f.set(b, a)
    });
    d = _.pe(f.fd(), zh);
    c = _.r.FormData && a instanceof _.r.FormData;
    !_.id(vh, b) || d || c || f.set("Content-Type",
    "application/x-www-form-urlencoded;charset=utf-8");
    f.forEach(function(a, b) {
        this.A.setRequestHeader(b, a)
    }, this);
    this.D && (this.A.responseType = this.D);
    "withCredentials"in this.A && (this.A.withCredentials = this.pa);
    try {
        Ah(this), 0 < this.F && ((this.ka = Bh(this.A)) ? (this.A.timeout = this.F, this.A.ontimeout = (0, _.s)(this.Yc, this)) : this.K = _.$g(this.Yc, this.F, this)), this.V=!0, this.A.send(a), this.V=!1
    } catch (g) {
        yh(this)
    }
};
var Bh = function(a) {
    return _.M && _.he(9) && _.Vc(a.timeout) && _.ca(a.ontimeout)
}, zh = function(a) {
    return "content-type" == a.toLowerCase()
};
_.th.prototype.Yc = function() {
    "undefined" != typeof _.ba && this.A && (this.G = 8, this.dispatchEvent("timeout"), this.abort(8))
};
var yh = function(a) {
    a.B=!1;
    a.A && (a.C=!0, a.A.abort(), a.C=!1);
    a.G = 5;
    Ch(a);
    Dh(a)
}, Ch = function(a) {
    a.U || (a.U=!0, a.dispatchEvent("complete"), a.dispatchEvent("error"))
};
_.th.prototype.abort = function(a) {
    this.A && this.B && (this.B=!1, this.C=!0, this.A.abort(), this.C=!1, this.G = a || 7, this.dispatchEvent("complete"), this.dispatchEvent("abort"), Dh(this))
};
_.th.prototype.W = function() {
    this.A && (this.B && (this.B=!1, this.C=!0, this.A.abort(), this.C=!1), Dh(this, !0));
    _.th.J.W.call(this)
};
_.th.prototype.wa = function() {
    this.ob() || (this.da || this.V || this.C ? Eh(this) : this.ia())
};
_.th.prototype.ia = function() {
    Eh(this)
};
var Eh = function(a) {
    if (a.B && "undefined" != typeof _.ba && (!a.Y[1] || 4 != _.Fh(a) || 2 != a.getStatus()))
        if (a.V && 4 == _.Fh(a))
            _.$g(a.wa, 0, a);
        else if (a.dispatchEvent("readystatechange"), 4 == _.Fh(a)) {
            a.B=!1;
            try {
                _.Gh(a) ? (a.dispatchEvent("complete"), a.dispatchEvent("success")) : (a.G = 6, a.getStatus(), Ch(a))
            } finally {
                Dh(a)
            }
    }
}, Dh = function(a, b) {
    if (a.A) {
        Ah(a);
        var c = a.A, d = a.Y[0] ? _.Dd: null;
        a.A = null;
        a.Y = null;
        b || a.dispatchEvent("ready");
        try {
            c.onreadystatechange = d
        } catch (e) {}
    }
}, Ah = function(a) {
    a.A && a.ka && (a.A.ontimeout = null);
    _.Vc(a.K) &&
    (_.ah(a.K), a.K = null)
};
_.th.prototype.Cc = function() {
    return !!this.A
};
_.Gh = function(a) {
    var b = a.getStatus(), c;
    if (!(c = _.mh(b))) {
        if (b = 0 === b)
            a = _.hh(String(a.N))[1] || null, !a && window.self.location && (a = window.self.location.protocol, a = a.substr(0, a.length - 1)), b=!uh.test(a ? a.toLowerCase() : "");
        c = b
    }
    return c
};
_.Fh = function(a) {
    return a.A ? a.A.readyState : 0
};
_.th.prototype.getStatus = function() {
    try {
        return 2 < _.Fh(this) ? this.A.status : - 1
    } catch (a) {
        return - 1
    }
};
_.Hh = function(a) {
    try {
        return a.A ? a.A.responseText : ""
    } catch (b) {
        return ""
    }
};
_.Ih = function(a) {
    try {
        if (!a.A)
            return null;
        if ("response"in a.A)
            return a.A.response;
        switch (a.D) {
        case "":
        case "text":
            return a.A.responseText;
        case "arraybuffer":
            if ("mozResponseArrayBuffer"in a.A)
                return a.A.mozResponseArrayBuffer
        }
        return null
    } catch (b) {
        return null
    }
};
_.th.prototype.getAllResponseHeaders = function() {
    return this.A && 4 == _.Fh(this) ? this.A.getAllResponseHeaders() : ""
};
_.oe(function(a) {
    _.th.prototype.ia = a(_.th.prototype.ia)
});

var ___;
_.kc = window.tamings___ || [];
_.lc = window.caja___;
___ = window.___;
_.ya.sk = function() {
    var a = {}, b = [], c = function(a) {
        var b = {
            method: a.request.method,
            id: a.key
        };
        a.request.rpc && (b.params = a.request.rpc);
        return b
    };
    a.execute = function(a) {
        for (var e = {}, f = {}, g = 0, k = [], l = 0; l < b.length; l++) {
            var m = b[l].request.transport;
            f[m.name] || (k.push(m), g++);
            f[m.name] = f[m.name] || [];
            f[m.name].push(c(b[l]))
        }
        l = function(c) {
            c.error && (e.error = c.error);
            for (var f = 0; f < b.length; f++) {
                var k = b[f].key, l = c[k];
                l && (e[k] = l.error ? l : l.data || l.result)
            }
            g--;
            0 === g && a(e)
        };
        for (m = 0; m < k.length; m++)
            k[m].execute(f[k[m].name], l);
        0 == g && window.setTimeout(function() {
            a(e)
        }, 0)
    };
    a.add = function(c, e) {
        e && c && b.push({
            key: c,
            request: e
        });
        return a
    };
    return a
};
_.ya.dl = function(a, b) {
    if ("newBatch" !== a) {
        for (var c = a.split("."), d = window.osapi, e = 0; e < c.length - 1; e++)
            d[c[e]] = d[c[e]] || {}, d = d[c[e]];
        var f = c[c.length - 1];
        d[f] ? (d.__dupwarn || _.Jb("Skipping duplicate osapi method definition " + a + " on transport " + b.name + "; others may exist, but suppressing warnings"), d.__dupwarn=!0) : (d[f] = function(c) {
            c = c || {};
            c.userId = c.userId || "@viewer";
            c.groupId = c.groupId || "@self";
            return new mc(a, b, c)
        }, "undefined" !== typeof _.kc && _.kc.push(function() {
            _.lc.markTameAsFunction(d[f], a)
        }))
    }
};
var mc = function(a, b, c) {
    this.method = a;
    this.transport = b;
    this.rpc = c
};
mc.prototype.execute = function(a) {
    var b = "undefined" !== typeof _.lc && _.lc.getUseless && _.lc.getUseless(), c = b ? _.lc.getUseless(): this, d = b ? _.lc.untame(a): a;
    a = _.ya.sk();
    a.add(this.method, this);
    a.execute(function(a) {
        a.error ? d.call(c, a.error) : d.call(c, a[c.method])
    })
};
_.C("osapi.newBatch.add", _.ya.sk.add);
_.C("osapi.newBatch.execute", _.ya.sk.execute);
_.C("osapi._registerMethod", _.ya.dl);
_.C("osapi._BoundCall", mc);
_.C("osapi._BoundCall.prototype.execute", mc.prototype.execute);
var Qj = function(a) {
    a = a.split(".");
    for (var b = window.osapi || window, c; c = a.shift();)
        if (null != b[c])
            b = b[c];
        else 
            return null;
    return b
}, Rj = function(a, b) {
    var c = a.split("."), d = window;
    c[0]in d ||!d.execScript || d.execScript("var " + c[0]);
    for (var e; c.length && (e = c.shift());)
        c.length || void 0 === b ? d[e] ? d = d[e] : d = d[e] = {} : d[e] = b
};
var Tj, Vj, Yj, Zj, ak, Uj, bk, ck;
Tj = {};
Vj = function(a, b) {
    _.ya.dl(a, {
        name: "googleapis",
        execute: Uj,
        root: b
    });
    var c = Qj(a);
    Rj(a, c);
    if (0 != a.indexOf("googleapis.")) {
        var d = a.substring(a.indexOf(".") + 1), e = d.lastIndexOf(".delete");
        - 1 != e && e + 7 == d.length && (d = d.replace(".delete", ".remove"));
        Rj("googleapis." + d, c)
    }
};
_.Wj = function(a) {
    for (var b in a)
        a.hasOwnProperty(b) && (Tj[b] = a[b])
};
_.Xj = function(a) {
    null == a ? _.Sj && delete _.Sj.key : (_.Sj = _.Sj || {}, _.Sj.key = a)
};
Yj = function(a, b) {
    if ("trace" != a)
        throw 'only the "trace" parameter may be set using this API';
    null == b ? _.Sj && delete _.Sj[a] : (_.Sj = _.Sj || {}, _.Sj[a] = b)
};
Zj = function(a) {
    a = a || {};
    if (window.navigator) {
        for (var b = ["appVersion", "platform", "userAgent"], c = [], d = 0; d < b.length; d++)
            window.navigator[b[d]] && c.push((0, window.encodeURIComponent)(b[d]) + "=" + (0, window.encodeURIComponent)(window.navigator[b[d]]));
        a["X-ClientDetails"] = c.join("&")
    }
    return a
};
ak = function(a) {
    a.clientVersion = "1.0.0-alpha";
    _.Sj && (a.urlParams = _.Sj)
};
Uj = function(a, b) {
    for (var c = this.root || null, d = 0; d < a.length; d++) {
        var e = a[d], f;
        f = e.method;
        f = f.substring(0, f.indexOf("."));
        e.jsonrpc = "2.0";
        e.key = e.id;
        c = e.root || c;
        (f = Tj[f] || "v1", !e.apiVersion) && (e.apiVersion = f)
    }
    d = {};
    d = _.Pj.fl(d);
    d = Zj(d);
    c = {
        requests: a,
        headers: d,
        root: c
    };
    ak(c);
    _.Pj.fg("makeRequest", c, b)
};
bk = function(a, b) {
    for (var c = 0, d = a.length; c < d; c++) {
        var e = a[c];
        e.key = e.id;
        var f = e.params.headers || {}, f = _.Pj.fl(f), f = Zj(f);
        e.params.headers = f;
        ak(e.params)
    }
    _.Pj.fg("makeHttpRequests", a, b)
};
ck = function() {
    var a = _.H.Fb(), b = {
        debug: "googleapis.config/debug"
    }, c;
    for (c in b)
        a[c] && _.eb(b[c], "true" == a[c]);
    for (var d in _.E("googleapis.config/methods"))
        Vj(d);
    _.E("googleapis.config/versions") && _.Wj(_.E("googleapis.config/versions"));
    _.E("googleapis.config/developerKey") && _.Xj(_.E("googleapis.config/developerKey"));
    _.ya.dl("googleapis.newHttpRequest", {
        name: "googleapis",
        execute: bk
    });
    Rj("googleapis.newHttpRequest", Qj("googleapis.newHttpRequest"))
};
ck();
var dk = function() {
    this.A = {};
    this.B = _.ya.sk()
};
dk.prototype.add = function(a, b, c) {
    this.A[a] = c;
    this.B.add(a, b);
    return this
};
dk.prototype.execute = function(a) {
    var b = this;
    this.B.execute(function(c) {
        var d = {}, e=!1, f;
        for (f in c)
            if (c.hasOwnProperty(f)) {
                var g = c[f], k = b.A[f];
                k ? k(g) : (d[f] = g, e=!0)
            }
        e && a && a(d)
    })
};
_.C("googleapis.ApiClient.register", Vj);
_.C("googleapis.ApiClient.setVersions", _.Wj);
_.C("googleapis.ApiClient.setDeveloperKey", _.Xj);
_.C("googleapis.ApiClient.setUrlParameter", Yj);
_.C("googleapis.ApiClient.init", ck);
_.C("googleapis.Batch", dk);
_.C("googleapis.Batch.prototype.add", dk.prototype.add);
_.C("googleapis.Batch.prototype.execute", dk.prototype.execute);
_.C("googleapis.init", function() {
    ck()
});
_.C("googleapis.newBatch", function() {
    return new dk
});
_.C("googleapis.newRequest", function(a, b, c) {
    b = b || {};
    var d = {
        name: "googleapis",
        execute: Uj,
        root: c
    }, e = _.ya.sk(), f = this;
    c = {};
    c.method = a;
    c.transport = d;
    c.rpc = b;
    c.execute = function(c) {
        e.add(a, {
            method: a,
            rpc: b,
            transport: d
        });
        e.execute(function(b) {
            b.error ? c.call(f, b.error) : c.call(f, b[a])
        })
    };
    return c
});
_.C("googleapis.register", function(a, b) {
    Vj(a, b)
});
_.C("googleapis.setUrlParameter", function(a, b) {
    Yj(a, b)
});
_.C("googleapis.setDeveloperKey", function(a) {
    _.Xj(a)
});
_.C("googleapis.setVersions", function(a) {
    _.Wj(a)
});

var gk;
_.ek = function(a) {
    if (!_.Qc(a))
        return null;
    for (var b = {}, c = 0; c < a.length; c++)
        b[a[c]] = a[c];
    return b
};
_.fk = function(a, b) {
    var c = a.length;
    if (c != b.length)
        return !1;
    for (var d = 0; d < c; ++d) {
        var e = a.charCodeAt(d), f = b.charCodeAt(d);
        65 <= e && 90 >= e && (e += 32);
        65 <= f && 90 >= f && (f += 32);
        if (e != f)
            return !1
    }
    return !0
};
gk = null;
_.hk = function(a) {
    if (null === gk) {
        var b = _.E("client/headers/response");
        b || (b = _.E("googleapis/headers/response"));
        gk = _.ek(b)
    }
    if (null != gk) {
        if (gk.hasOwnProperty(a))
            return !0;
        for (var c in gk)
            if (gk.hasOwnProperty(c) && _.fk(c, a))
                return !0
    }
    return !1
};
_.ik = function(a) {
    a = String(a || "").split("\x00").join("");
    for (var b = [], c = 0, d = a.length; c < d; ++c) {
        var e = a.charAt(c), f = a.charCodeAt(c);
        if (55296 <= f && 56319 >= f && c + 1 < d) {
            var g = a.charAt(c + 1), k = a.charCodeAt(c + 1);
            56320 <= k && 57343 >= k && (e += g, f = 65536 + (f - 55296<<10) + (k - 56320), ++c)
        }
        if (!(0 <= f && 1114109 >= f) || 55296 <= f && 57343 >= f || 64976 <= f && 65007 >= f || 65534 == (f & 65534))
            f = 65533, e = String.fromCharCode(f);
        g=!(32 <= f && 126 >= f) || " " == e || ":" == e || "\\" == e;
        "%" == e && (c + 2 >= d ? g=!0 : (k = 16 * (0, window.parseInt)(a.charAt(c + 1), 16) + (0, window.parseInt)(a.charAt(c +
        2), 16), 0 <= k && 255 >= k ? (f = k, e = 0 == f ? "" : "%" + (256 + k).toString(16).toUpperCase().substr(1), c += 2) : g=!0));
        g && (e = (0, window.encodeURIComponent)(e), 1 >= e.length && (0 <= f && 127 >= f ? e = "%" + (256 + f).toString(16).toUpperCase().substr(1) : (f = 65533, e = (0, window.encodeURIComponent)(String.fromCharCode(f)))));
        b.push(e)
    }
    a = b.join("");
    a = a.split("#")[0];
    a = a.split("?");
    b = a[0].split("/");
    e = [];
    c = 0;
    for (d = b.length; c < d; ++c)
        f = b[c], g = f.split("%2E").join("."), g = g.split((0, window.encodeURIComponent)("\uff0e")).join("."), "." == g ? c + 1 == d && e.push("") :
        ".." == g ? (0 < e.length && e.pop(), c + 1 == d && e.push("")) : e.push(f);
    a[0] = e.join("/");
    for (a = a.join("?"); a && "/" == a.charAt(0);)
        a = a.substr(1);
    return "/" + a
};
_.kk = function(a) {
    var b = _.ik(a);
    if (String(a) != b)
        throw Error("G");
    (a = b) && "/" == a.charAt(a.length - 1) || (a = (a || "") + "/");
    _.J.register("init", function() {
        _.kk(a)
    });
    _.jk = a;
    _.H.Fb(window.location.href)
};
_.lk = "function" == typeof window.atob;
var xk = function(a) {
    this.B = a;
    this.A = _.Aa;
    this.G = this.F;
    this.M = /MSIE\s*[0-8](\D|$)/.test(window.navigator.userAgent);
    if (this.B.Hy) {
        this.A = this.B.Tv(this.A, this.B.Hy);
        a = this.A.document;
        var b = a.createElement("script");
        b.setAttribute("type", "text/javascript");
        b.text = "window.doPostMsg=function(w,s,o) {window.setTimeout(function(){w.postMessage(s,o);},0);};";
        a.body.appendChild(b);
        this.G = this.A.doPostMsg
    }
    this.C = {};
    this.D = {};
    a = (0, _.s)(this.K, this);
    _.Wa(this.A, "message", a);
    _.Ha(_.Ya, "RPMQ", []).push(a);
    this.A !=
    this.A.parent && wk(this, this.A.parent, "{h:'" + (0, window.escape)(this.A.name) + "'}", "*")
}, yk = function(a) {
    var b = null;
    0 === a.indexOf("{h:'") && a.indexOf("'}") === a.length - 2 && (b = (0, window.unescape)(a.substring(4, a.length - 2)));
    return b
}, zk = function(a) {
    a = (0, _.ac)(a);
    return null !== a && "object" === typeof a&&!!a.g
};
xk.prototype.K = function(a) {
    var b = String(a.data);
    (0, _.Rb)("gapi.rpc.receive(" + Ak + "): " + (!b || 512 >= b.length ? b : b.substr(0, 512) + "... (" + b.length + " bytes)"));
    var c = 0 !== b.indexOf("!_");
    c || (b = b.substring(2));
    var d = zk(b);
    if (!c&&!d) {
        if (!d && (c = yk(b))) {
            if (this.C[c])
                this.C[c]();
            else 
                this.D[c] = 1;
            return 
        }
        var e = a.origin, f = this.B.TD;
        this.M ? _.Aa.setTimeout(function() {
            f(b, e)
        }, 0) : f(b, e)
    }
};
xk.prototype.pb = function(a, b) {
    ".." === a || this.D[a] ? (b(), delete this.D[a]) : this.C[a] = b
};
var wk = function(a, b, c, d) {
    var e = zk(c) ? "": "!_";
    (0, _.Rb)("gapi.rpc.send(" + Ak + "): " + (!c || 512 >= c.length ? c : c.substr(0, 512) + "... (" + c.length + " bytes)"));
    a.G(b, e + c, d)
};
xk.prototype.F = function(a, b, c) {
    a.postMessage(b, c)
};
xk.prototype.send = function(a, b, c) {
    (a = this.B.Tv(this.A, a))&&!a.closed && wk(this, a, b, c)
};
var Bk, Ck, Dk, Ek, Fk, Gk, Hk, Ik, Jk, Ak, Kk, Lk, Mk, Nk, Ok, Pk, Qk, Rk, Wk, Xk, Zk, $k, bl, al, Sk, Tk, cl, dl, el, fl;
Bk = 0;
Ck = [];
Dk = {};
Ek = {};
Fk = _.H.Fb;
Gk = Fk();
Hk = Gk.rpctoken;
Ik = Gk.parent || _.Ba.referrer;
Jk = Gk.rly;
Ak = Jk || (_.Aa !== _.Aa.top || _.Aa.opener) && _.Aa.name || "..";
Kk = null;
Lk = {};
Mk = function() {};
Nk = {
    send: Mk,
    pb: Mk
};
Ok = function(a, b) {
    "/" == b.charAt(0) && (b = b.substring(1), a = _.Aa.top);
    for (var c = b.split("/"); c.length;) {
        var d;
        d = c.shift();
        "{" == d.charAt(0) && "}" == d.charAt(d.length - 1) && (d = d.substring(1, d.length - 1));
        if (".." === d)
            a = a == a.parent ? a.opener : a.parent;
        else if (".." !== d && a.frames[d]) {
            if (a = a.frames[d], !("postMessage"in a))
                throw "Not a window";
        } else 
            return null
    }
    return a
};
Pk = function(a) {
    return (a = Dk[a]) && a.Si
};
Qk = function(a) {
    if (a.f in{})
        return !1;
    var b = a.t, c = Dk[a.r];
    a = a.origin;
    return c && (c.Si === b ||!c.Si&&!b) && (a === c.origin || "*" === c.origin)
};
Rk = function(a) {
    var b = a.id.split("/"), c = b[b.length - 1], d = a.origin;
    return function(a) {
        var b = a.origin;
        return a.f == c && (d == b || "*" == d)
    }
};
_.Uk = function(a, b, c) {
    a = Sk(a);
    Ek[a.name] = {
        Fp: b,
        qk: a.qk,
        Xi: c || Qk
    };
    Tk()
};
_.Vk = function(a) {
    delete Ek[Sk(a).name]
};
Wk = {};
Xk = function(a, b) {
    var c = Wk["_" + a];
    c && c[1](this) && c[0].call(this, b)
};
Zk = function(a) {
    var b = a.c;
    if (!b)
        return Mk;
    var c = a.r, d = a.g ? "legacy__": "";
    return function() {
        var a = [].slice.call(arguments, 0);
        a.unshift(c, d + "__cb", null, b);
        _.Yk.apply(null, a)
    }
};
$k = function(a) {
    Kk = a
};
bl = function(a) {
    Lk[a] || (Lk[a] = _.Aa.setTimeout(function() {
        Lk[a]=!1;
        al(a)
    }, 0))
};
al = function(a) {
    var b = Dk[a];
    if (b && b.ready) {
        var c = b.xr;
        for (b.xr = []; c.length;)
            Nk.send(a, (0, _.$b)(c.shift()), b.origin)
    }
};
Sk = function(a) {
    return 0 === a.indexOf("legacy__") ? {
        name: a.substring(8),
        qk: !0
    } : {
        name: a,
        qk: !1
    }
};
Tk = function() {
    for (var a = _.E("rpc/residenceSec") || 60, b = (new Date).getTime() / 1E3, c = 0, d; d = Ck[c]; ++c) {
        var e = d.jh;
        if (!e || 0 < a && b - d.timestamp > a)
            Ck.splice(c, 1), --c;
        else {
            var f = e.s, g = Ek[f] || Ek["*"];
            if (g)
                if (Ck.splice(c, 1), --c, e.origin = d.origin, d = Zk(e), e.callback = d, g.Xi(e)) {
                    if ("__cb" !== f&&!!g.qk!=!!e.g)
                        break;
                        e = g.Fp.apply(e, e.a);
                        void 0 !== e && d(e)
                } else (0, _.Rb)("gapi.rpc.rejected(" + Ak + "): " + f)
            }
    }
};
cl = function(a, b, c) {
    Ck.push({
        jh: a,
        origin: b,
        timestamp: (new Date).getTime() / 1E3
    });
    c || Tk()
};
dl = function(a, b) {
    var c = (0, _.ac)(a);
    cl(c, b, !1)
};
el = function(a) {
    for (; a.length;)
        cl(a.shift(), this.origin, !0);
    Tk()
};
fl = function(a) {
    var b=!1;
    a = a.split("|");
    var c = a[0];
    0 <= c.indexOf("/") && (b=!0);
    return {
        id: c,
        origin: a[1] || "*",
        Lq: b
    }
};
_.gl = function(a, b, c, d) {
    var e = fl(a);
    d && (_.Aa.frames[e.id] = _.Aa.frames[e.id] || d);
    a = e.id;
    if (!Dk.hasOwnProperty(a)) {
        c = c || null;
        d = e.origin;
        if (".." === a)
            d = _.Mi.Ba(Ik), c = c || Hk;
        else if (!e.Lq) {
            var f = _.Ba.getElementById(a);
            f && (f = f.src, d = _.Mi.Ba(f), c = c || Fk(f).rpctoken)
        }
        "*" === e.origin && d || (d = e.origin);
        Dk[a] = {
            Si: c,
            xr: [],
            origin: d,
            LI: b,
            zy: function() {
                var b = a;
                Dk[b].ready = 1;
                al(b)
            }
        };
        Nk.pb(a, Dk[a].zy)
    }
    return Dk[a].zy
};
_.Yk = function(a, b, c, d) {
    a = a || "..";
    _.gl(a);
    a = a.split("|", 1)[0];
    var e = b, f = [].slice.call(arguments, 3), g = c, k = Ak, l = Hk, m = Dk[a], n = k, p = fl(a);
    if (m && ".." !== a) {
        if (p.Lq) {
            if (!(l = Dk[a].LI)) {
                l = null;
                Kk ? l = Kk.substring(1).split("/") : l = [Ak];
                for (var n = l.length - 1, v = _.Aa.parent; v !== _.Aa.top;) {
                    var x = v.parent;
                    if (!n--) {
                        for (var t = null, w = x.frames.length, y = 0; y < w; ++y)
                            x.frames[y] == v && (t = y);
                        l.unshift("{" + t + "}")
                    }
                    v = x
                }
                l = "/" + l.join("/")
            }
            n = l
        } else 
            n = k = "..";
        l = m.Si
    }
    g && p ? (m = Qk, p.Lq && (m = Rk(p)), Wk["_" + ++Bk] = [g, m], p = Bk) : p = null;
    f = {
        s: e,
        f: k,
        r: n,
        t: l,
        c: p,
        a: f
    };
    e = Sk(e);
    f.s = e.name;
    f.g = e.qk;
    Dk[a].xr.push(f);
    bl(a)
};
if ("function" === typeof _.Aa.postMessage || "object" === typeof _.Aa.postMessage)
    Nk = new xk({
        Hy: Jk ? "../" + Jk: null,
        TD: dl,
        Tv: Ok,
        hT: Ak,
        Qv: Pk,
        LT: $k
    }), _.Uk("__cb", Xk, function() {
        return !0
    }), _.Uk("_processBatch", el, function() {
        return !0
    }), _.gl("..");

_.hl = function(a) {
    return new _.Qf(a)
};
_.il = _.il || {};
_.il.pF = function() {
    var a = 0, b = 0;
    window.self.innerHeight ? (a = window.self.innerWidth, b = window.self.innerHeight) : window.document.documentElement && window.document.documentElement.clientHeight ? (a = window.document.documentElement.clientWidth, b = window.document.documentElement.clientHeight) : window.document.body && (a = window.document.body.clientWidth, b = window.document.body.clientHeight);
    return {
        width: a,
        height: b
    }
};

_.il = _.il || {};
(function() {
    function a(a, c) {
        window.getComputedStyle(a, "").getPropertyValue(c).match(/^([0-9]+)/);
        return (0, window.parseInt)(RegExp.$1, 10)
    }
    _.il.Jb = function() {
        var b = _.il.pF().height, c = window.document.body, d = window.document.documentElement;
        if ("CSS1Compat" === window.document.compatMode && d.scrollHeight)
            return d.scrollHeight !== b ? d.scrollHeight : d.offsetHeight;
        if (0 <= window.navigator.userAgent.indexOf("AppleWebKit")) {
            b = 0;
            for (c = [window.document.body]; 0 < c.length;) {
                var e = c.shift(), d = e.childNodes;
                if ("undefined" !==
                typeof e.style) {
                    var f = e.style.overflowY;
                    f || (f = (f = window.document.defaultView.getComputedStyle(e, null)) ? f.overflowY : null);
                    if ("visible" != f && "inherit" != f && (f = e.style.height, f || (f = (f = window.document.defaultView.getComputedStyle(e, null)) ? f.height : ""), 0 < f.length && "auto" != f))
                        continue
                }
                for (e = 0; e < d.length; e++) {
                    f = d[e];
                    if ("undefined" !== typeof f.offsetTop && "undefined" !== typeof f.offsetHeight)
                        var g = f.offsetTop + f.offsetHeight + a(f, "margin-bottom"), b = Math.max(b, g);
                    c.push(f)
                }
            }
            return b + a(window.document.body, "border-bottom") + a(window.document.body, "margin-bottom") + a(window.document.body, "padding-bottom")
        }
        if (c && d)
            return e = d.scrollHeight, f = d.offsetHeight, d.clientHeight !== f && (e = c.scrollHeight, f = c.offsetHeight), e > b ? e > f ? e : f : e < f ? e : f
    }
})();

var kl, jl, ql, rl, ll, ol, ml, sl, nl, jl;
_.pl = function() {
    var a;
    jl ? (a = new _.Aa.Uint32Array(1), kl.getRandomValues(a), a = Number("0." + a[0])) : (a = ll, a += (0, window.parseInt)(ml.substr(0, 20), 16), ml = nl(ml), a = a / (ol + Math.pow(16, 20)));
    return a
};
kl = _.Aa.crypto;
jl=!1;
ql = 0;
rl = 0;
ll = 1;
ol = 0;
ml = "";
sl = function(a) {
    a = a || _.Aa.event;
    var b = a.screenX + a.clientX<<16, b = b + (a.screenY + a.clientY), b = (new Date).getTime()%1E6 * b;
    ll = ll * b%ol;
    0 < ql&&++rl == ql && _.Va(_.Aa, "mousemove", sl, "remove", "de")
};
nl = function(a) {
    var b = new _.$i;
    b.zs(a);
    return b.Ef()
};
jl=!!kl && "function" == typeof kl.getRandomValues;
jl || (ol = 1E6 * (window.screen.width * window.screen.width + window.screen.height), ml = nl(_.Ba.cookie + "|" + _.Ba.location + "|" + (new Date).getTime() + "|" + Math.random()), ql = _.E("random/maxObserveMousemove") || 0, 0 != ql && _.Wa(_.Aa, "mousemove", sl));

var ul, wl, xl, yl, zl, Al, Bl, Cl, Dl, El, Hl, Il, Jl, Kl, Ll, Ml;
_.tl = function(a, b) {
    a = a || {};
    for (var c in a)
        _.Ma(a, c) && (b[c] = a[c])
};
ul = /^([^?#]*)(\?([^#]*))?(\#(.*))?$/;
_.vl = /^https?:\/\/[^\/%\\?#\s]+\/[^\s]*$/i;
wl = function(a, b) {
    var c = [];
    if (a)
        for (var d in a)
            if (_.Ma(a, d) && null != a[d]) {
                var e = b ? b(a[d]): a[d];
                c.push((0, window.encodeURIComponent)(d) + "=" + (0, window.encodeURIComponent)(e))
            }
    return c
};
xl = function(a) {
    return a.nu + (0 < a.query.length ? "?" + a.query.join("&") : "") + (0 < a.Ul.length ? "#" + a.Ul.join("&") : "")
};
yl = function(a) {
    a = a.match(ul);
    var b = _.Ka();
    b.nu = a[1];
    b.query = a[3] ? [a[3]] : [];
    b.Ul = a[5] ? [a[5]] : [];
    return b
};
zl = /'/g;
Al = /"/g;
Bl = />/g;
Cl = /</g;
Dl = /&/g;
El = function(a) {
    for (; a.firstChild;)
        a.removeChild(a.firstChild)
};
_.Fl = function(a, b) {
    var c = "";
    2E3 < b.length && (c = b.substring(2E3), b = b.substring(0, 2E3));
    var d = a.createElement("div"), e = a.createElement("a");
    e.href = b;
    d.appendChild(e);
    d.innerHTML = d.innerHTML;
    b = String(d.firstChild.href);
    d.parentNode && d.parentNode.removeChild(d);
    return b + c
};
_.Gl = function(a, b, c, d) {
    a = yl(a);
    a.query.push.apply(a.query, wl(b, d));
    a.Ul.push.apply(a.Ul, wl(c, d));
    return xl(a)
};
Hl = function(a) {
    return String(a).replace(Dl, "&amp;").replace(Cl, "&lt;").replace(Bl, "&gt;").replace(Al, "&quot;").replace(zl, "&#39;")
};
Jl = function() {
    var a = _.Ya.onl;
    if (!a) {
        a = _.Ka();
        _.Ya.onl = a;
        var b = _.Ka();
        a.e = function(a) {
            var d = b[a];
            d && (delete b[a], d())
        };
        a.a = function(a, d) {
            b[a] = d
        };
        a.r = function(a) {
            delete b[a]
        }
    }
    return a
};
Kl = function(a, b) {
    var c = b.onload;
    return "function" === typeof c ? (Jl().a(a, c), c) : null
};
Ll = function(a) {
    _.Na(/^\w+$/.test(a), "Unsupported id - " + a);
    Jl();
    return 'onload="window.___jsl.onl.e(&#34;' + a + '&#34;)"'
};
Ml = function(a) {
    Jl().r(a)
};
var Ol, Pl, Tl;
_.Nl = {
    allowtransparency: "true",
    frameborder: "0",
    hspace: "0",
    marginheight: "0",
    marginwidth: "0",
    scrolling: "no",
    style: "",
    tabindex: "0",
    vspace: "0",
    width: "100%"
};
Ol = {
    allowtransparency: !0,
    onload: !0
};
Pl = 0;
_.Ql = function(a, b) {
    var c, d = 0;
    do 
        c = b.id || ["I", Pl++, "_", (new Date).getTime()].join("");
    while (a.getElementById(c) && 5>++d);
    _.Na(5 > d, "Error creating iframe id");
    return c
};
_.Rl = function(a, b) {
    return a ? b + "/" + a : ""
};
_.Sl = function(a, b, c, d) {
    var e = {}, f = {};
    a.documentMode && 9 > a.documentMode && (e.hostiemode = a.documentMode);
    _.tl(d.queryParams || {}, e);
    _.tl(d.fragmentParams || {}, f);
    var g = d.connectWithQueryParams ? e: f, k = d.pfname, l = _.Ka();
    l.id = c;
    l.parent = a.location.protocol + "//" + a.location.host;
    c = _.Ua(a.location.href, "parent");
    k = k || "";
    !k && c && (k = _.Rl(_.Ua(a.location.href, "id", ""), _.Ua(a.location.href, "pfname", "")));
    l.pfname = k;
    _.tl(l, g);
    (l = _.Ua(b, "rpctoken") || e.rpctoken || f.rpctoken) || (l = g.rpctoken = d.rpctoken || String(Math.round(1E8 *
    _.pl())));
    d.rpctoken = l;
    g = a.location.href;
    a = _.Ka();
    (l = _.Ua(g, "_bsh", _.Ya.bsh)) && (a._bsh = l);
    (g = _.Ya.dpo ? _.Ya.h : _.Ua(g, "jsh", _.Ya.h)) && (a.jsh = g);
    d.hintInFragment ? _.tl(a, f) : _.tl(a, e);
    return _.Gl(b, e, f, d.paramsSerializer)
};
Tl = function(a) {
    _.Na(!a || _.vl.test(a), "Illegal url for new iframe - " + a)
};
_.Ul = function(a, b, c, d, e) {
    Tl(c.src);
    var f, g = Kl(d, c), k = g ? Ll(d): "";
    try {
        f = a.createElement('<iframe frameborder="' + Hl(String(c.frameborder)) + '" scrolling="' + Hl(String(c.scrolling)) + '" ' + k + ' name="' + Hl(String(c.name)) + '"/>')
    } catch (l) {
        f = a.createElement("iframe"), g && (f.onload = function() {
            f.onload = null;
            g.call(this)
        }, Ml(d))
    }
    for (var m in c)
        a = c[m], "style" === m && "object" === typeof a ? _.tl(a, f.style) : Ol[m] || f.setAttribute(m, String(a));
    (m = e && e.beforeNode || null) || e && e.dontclear || El(b);
    b.insertBefore(f, m);
    f = m ? m.previousSibling : b.lastChild;
    c.allowtransparency && (f.allowTransparency=!0);
    return f
};
var Vl, Yl;
Vl = /^:[\w]+$/;
_.Wl = /:([a-zA-Z_]+):/g;
_.Xl = function() {
    var a = _.Si() || "0", b = _.Ti(), c = _.Ui(void 0, a), d=!1 === _.E("isLoggedIn"), e = d ? "_/im/" : "";
    e && (c = "");
    var f = _.E("iframes/:socialhost:"), g = _.E("iframes/:im_socialhost:");
    return Il = {
        socialhost: f,
        ctx_socialhost: d ? g: f,
        session_index: a,
        session_delegate: b,
        session_prefix: c,
        im_prefix: e
    }
};
Yl = function(a, b) {
    return _.Xl()[b] || ""
};
_.Zl = function(a) {
    return _.Fl(_.Ba, a.replace(_.Wl, Yl))
};
_.$l = function(a) {
    var b = a;
    Vl.test(a) && (b = _.E("iframes/" + b.substring(1) + "/url"), _.Na(!!b, "Unknown iframe url config for - " + a));
    return _.Zl(b)
};
_.am = function(a, b, c) {
    var d = c || {};
    c = d.attributes || {};
    _.Na(!d.allowPost ||!c.onload, "onload is not supported by post iframe");
    a = _.$l(a);
    c = b.ownerDocument || _.Ba;
    var e = _.Ql(c, d);
    a = _.Sl(c, a, e, d);
    var f = _.Ka();
    _.tl(_.Nl, f);
    _.tl(d.attributes, f);
    f.name = f.id = e;
    f.src = a;
    d.eurl = a;
    if ((d || {}).allowPost && 2E3 < a.length) {
        var g = yl(a);
        f.src = "";
        f["data-postorigin"] = a;
        a = _.Ul(c, b, f, e);
        var k;
        if ( - 1 != window.navigator.userAgent.indexOf("WebKit")) {
            k = a.contentWindow.document;
            k.open();
            var f = k.createElement("div"), l = {}, m = e + "_inner";
            l.name = m;
            l.src = "";
            l.style = "display:none";
            _.Ul(c, f, l, m, d)
        }
        f = (d = g.query[0]) ? d.split("&") : [];
        d = [];
        for (l = 0; l < f.length; l++)
            m = f[l].split("=", 2), d.push([(0, window.decodeURIComponent)(m[0]), (0, window.decodeURIComponent)(m[1])]);
        g.query = [];
        f = xl(g);
        g = c.createElement("form");
        g.action = f;
        g.method = "POST";
        g.target = e;
        g.style.display = "none";
        for (e = 0; e < d.length; e++)
            f = c.createElement("input"), f.type = "hidden", f.name = d[e][0], f.value = d[e][1], g.appendChild(f);
        b.appendChild(g);
        g.submit();
        g.parentNode.removeChild(g);
        k && k.close();
        b = a
    } else 
        b = _.Ul(c, b, f, e, d);
    return b
};

var bm = function(a) {
    this.A = a
};
_.h = bm.prototype;
_.h.value = function() {
    return this.A
};
_.h.yf = function(a) {
    this.A.width = a;
    return this
};
_.h.Rb = function() {
    return this.A.width
};
_.h.kh = function(a) {
    this.A.height = a;
    return this
};
_.h.Jb = function() {
    return this.A.height
};
_.cm = function(a) {
    this.A = a
};
_.cm.prototype.kh = function(a) {
    this.A.height = a;
    return this
};
_.cm.prototype.Jb = function() {
    return this.A.height
};
_.cm.prototype.yf = function(a) {
    this.A.width = a;
    return this
};
_.cm.prototype.Rb = function() {
    return this.A.width
};
_.dm = function(a) {
    this.A = a || {}
};
_.dm.prototype.value = function() {
    return this.A
};
_.em = function(a, b) {
    a.A.url = b;
    return a
};
_.dm.prototype.getUrl = function() {
    return this.A.url
};
_.fm = function(a, b) {
    a.A.style = b;
    return a
};
_.dm.prototype.ca = function() {
    return this.A.id
};
_.gm = function(a, b) {
    a.A.queryParams = b;
    return a
};
_.hm = function(a, b) {
    a.A.relayOpen = b;
    return a
};
_.dm.prototype.Th = _.q(7);
_.dm.prototype.getContext = _.q(8);
_.dm.prototype.Kb = function() {
    return this.A.openerIframe
};
_.dm.prototype.Lj = function() {
    this.A.attributes = this.A.attributes || {};
    return new bm(this.A.attributes)
};

var km, mm, lm, om, qm, rm;
_.dm.prototype.getContext = _.L(8, function() {
    return this.A.context
});
_.dm.prototype.Th = _.L(7, function() {
    return this.A.apis
});
_.im = function(a) {
    a.A.waitForOnload=!0
};
_.jm = function(a) {
    return a.A.rpctoken
};
km = function(a, b) {
    a.A.onload = b
};
mm = function(a) {
    this.resolve = this.reject = null;
    this.promise = _.hl((0, _.s)(function(a, c) {
        this.resolve = a;
        this.reject = c
    }, this));
    a && (this.promise = lm(this.promise, a))
};
lm = function(a, b) {
    return a.then(function(a) {
        try {
            b(a)
        } catch (d) {}
        return a
    })
};
_.nm = function(a) {
    return new _.Qf(function(b, c) {
        var d = a.length, e = [];
        if (d)
            for (var f = function(a, c) {
                d--;
                e[a] = c;
                0 == d && b(e)
            }, g = function(a) {
                c(a)
            }, k = 0, l; l = a[k]; k++)
                l.then(_.ra(f, k), g);
        else 
            b(e)
    })
};
om = function(a) {
    return new _.Qf(function(b) {
        b(a)
    })
};
_.sm = function(a) {
    this.A = a || {}
};
_.u(_.sm, _.dm);
_.tm = function(a, b) {
    a.A.frameName = b;
    return a
};
_.sm.prototype.Ic = function() {
    return this.A.frameName
};
_.um = function(a, b) {
    a.A.rpcAddr = b;
    return a
};
_.sm.prototype.Ig = function() {
    return this.A.rpcAddr
};
_.vm = function(a, b) {
    a.A.retAddr = b;
    return a
};
_.wm = function(a) {
    return a.A.retAddr
};
_.sm.prototype.we = function(a) {
    this.A.origin = a;
    return this
};
_.sm.prototype.Ba = function() {
    return this.A.origin
};
_.sm.prototype.xf = function(a) {
    this.A.setRpcReady = a;
    return this
};
_.sm.prototype.Ni = function(a) {
    this.A.context = a;
    return this
};
var xm = function(a, b) {
    a.A._rpcReadyFn = b
};
_.sm.prototype.Fa = function() {
    return this.A.iframeEl
};
var ym, Cm;
ym = /^[\w\.\-]*$/;
_.zm = function(a) {
    return a.Wc === a.getContext().Wc
};
_.Am = function() {
    return !0
};
_.Bm = function(a) {
    for (var b = _.Ka(), c = 0; c < a.length; c++)
        b[a[c]]=!0;
    return function(a) {
        return !!b[a.Wc]
    }
};
Cm = function(a, b, c) {
    var d = rm[a];
    return function(a) {
        if (!b.ob() && (_.Na(this.origin === b.Wc, "Wrong origin " + this.origin + " != " + b.Wc), d && 0 < d.length)) {
            for (var f = this.callback, g = [], k = 0; k < d.length; k++)
                g.push(om(d[k].call(b, a, b)));
            c || _.nm(g).then(f)
        }
    }
};
_.Dm = function(a, b, c) {
    _.Na("_default" != a, "Cannot update default api");
    qm[a] = {
        map: b,
        filter: c
    }
};
_.Em = function(a, b, c) {
    _.Na("_default" != a, "Cannot update default api");
    _.Ha(qm, a, {
        map: {},
        filter: _.zm
    }).map[b] = c
};
_.Fm = function(a, b) {
    _.Ha(qm, "_default", {
        map: {},
        filter: _.Am
    }).map[a] = b;
    _.ib(_.pm.A, function(c) {
        c.register(a, b, _.Am)
    })
};
_.Gm = function() {
    return _.pm
};
_.Hm = function(a) {
    a = a || {};
    this.ce=!1;
    this.R = _.Ka();
    this.A = _.Ka();
    this.V = a._window || _.Aa;
    this.C = this.V.location.href;
    this.da = (this.ia = _.Ua(this.C, "parent", "")) ? _.Ua(this.C, "pfname", "") : "";
    this.qa = this.ia ? _.Ua(this.C, "id", "") : "";
    this.ee = _.Rl(this.qa, this.da);
    this.Wc = _.Mi.Ba(this.C);
    if (this.qa) {
        var b = new _.sm;
        _.um(b, a._parentRpcAddr || "..");
        _.vm(b, a._parentRetAddr || this.qa);
        b.we(_.Mi.Ba(_.Ua(this.C, "parent", this.C)));
        _.tm(b, this.da);
        this.B = this.Dj(b.value())
    } else 
        this.B = null
};
_.h = _.Hm.prototype;
_.h.ob = function() {
    return this.ce
};
_.h.ha = function() {
    if (!this.ob()) {
        for (var a = 0; a < this.A.length; a++)
            this.A[a].ha();
        this.ce=!0
    }
};
_.h.Ic = function() {
    return this.ee
};
_.h.Ja = function() {
    return this.V
};
_.h.Bn = _.q(9);
_.h.Qp = function(a) {
    return this.R[a]
};
_.h.Dj = function(a) {
    _.Na(!this.ob(), "Cannot attach iframe in disposed context");
    a = new _.sm(a);
    a.Ig() || _.um(a, a.ca());
    _.wm(a) || _.vm(a, "..");
    a.Ba() || a.we(_.Mi.Ba(a.getUrl()));
    a.Ic() || _.tm(a, _.Rl(a.ca(), this.ee));
    var b = a.Ic();
    if (this.A[b])
        return this.A[b];
    var c = a.Ig(), d = c;
    a.Ba() && (d = c + "|" + a.Ba());
    var e = _.wm(a), f = _.jm(a);
    f || (f = (f = a.Fa()) && (f.getAttribute("data-postorigin") || f.src) || a.getUrl(), f = _.Ua(f, "rpctoken"));
    xm(a, _.gl(d, e, f, a.A._popupWindow));
    d = ((window.gadgets || {}).rpc || {}).setAuthToken;
    f && d && d(c,
    f);
    var g = new _.Im(this, c, b, a), k = a.A.messageHandlersFilter;
    _.ib(a.A.messageHandlers, function(a, b) {
        g.register(b, a, k)
    });
    a.A.setRpcReady && g.xf();
    _.Jm(g, "_g_rpcReady");
    return g
};
_.h.Kr = function(a) {
    _.tm(a, null);
    var b = a.ca();
    !b || ym.test(b)&&!this.Ja().document.getElementById(b) || (_.Qb("Ignoring requested iframe ID - " + b), a.A.id = null)
};
_.h.Xf = function(a) {
    _.Na(!this.ob(), "Cannot open iframe in disposed context");
    var b = new _.sm(a);
    Km(this, b);
    var c = b.Ic();
    if (c && this.A[c])
        return this.A[c];
    this.Kr(b);
    c = b.getUrl();
    _.Na(c, "No url for new iframe");
    var d = b.A.queryParams || {};
    d.usegapi = "1";
    _.gm(b, d);
    d = this.ya && this.ya(c, b);
    d || (d = b.A.where, _.Na(!!d, "No location for new iframe"), c = _.am(c, d, a), b.A.iframeEl = c, d = c.getAttribute("id"));
    _.um(b, d).A.id = d;
    b.we(_.Mi.Ba(b.A.eurl || ""));
    this.Sa && this.Sa(b, b.Fa());
    c = this.Dj(a);
    c.Yr && c.Yr(c, a);
    (a = b.A.onCreate) &&
    a(c);
    b.A.disableRelayOpen || c.nj("_open");
    return c
};
var Lm = function(a, b, c) {
    var d = b.A.canvasUrl;
    if (!d)
        return c;
    _.Na(!b.A.allowPost, "Post is not supported when using canvas url");
    var e = b.getUrl();
    _.Na(e && _.Mi.Ba(e) === a.Wc && _.Mi.Ba(d) === a.Wc, "Wrong origin for canvas or hidden url " + d);
    _.em(b, d);
    _.im(b);
    b.A.canvasUrl = null;
    return function(a) {
        var b = a.Ja(), d = b.location.hash, d = _.$l(e) + (/#/.test(e) ? d.replace(/^#/, "&") : d);
        b.location.replace(d);
        c && c(a)
    }
}, Nm = function(a, b, c) {
    var d = b.A.relayOpen;
    if (d) {
        var e = a.B;
        d instanceof _.Im ? (e = d, _.hm(b, 0)) : 0 < Number(d) && _.hm(b,
        Number(d) - 1);
        if (e) {
            _.Na(!!e.Tx, "Relaying iframe open is disabled");
            if (d = b.A.style)
                if (d = _.Mm[d])
                    b.Ni(a), d(b.value()), b.Ni(null);
            b.A.openerIframe = null;
            c.resolve(e.Tx(b));
            return !0
        }
    }
    return !1
}, Rm = function(a, b, c) {
    var d = b.A.style;
    if (d)
        if (_.Na(!!_.Om, "Defer style is disabled, when requesting style " + d), _.Pm[d])
            Km(a, b);
        else 
            return Qm(d, function() {
                _.Na(!!_.Pm[d], "Fail to load style - " + d);
                c.resolve(a.open(b.value()))
            }), !0;
    return !1
};
_.Hm.prototype.open = function(a, b) {
    _.Na(!this.ob(), "Cannot open iframe in disposed context");
    var c = new _.sm(a), d = Lm(this, c, b), e = new mm(d);
    (d = c.getUrl()) && _.em(c, _.$l(d));
    if (Nm(this, c, e) || Rm(this, c, e) || Nm(this, c, e))
        return e.promise;
    var f;
    c.A.waitForOnload && km(c.Lj(), function() {
        e.resolve(f)
    });
    f = this.Xf(a);
    c.A.waitForOnload || e.resolve(f);
    return e.promise
};
_.Hm.prototype.U = _.q(10);
_.Im = function(a, b, c, d) {
    this.ce=!1;
    this.vb = a;
    this.wf = b;
    this.ee = c;
    this.Za = d;
    this.Li = _.wm(this.Za);
    this.Wc = this.Za.Ba();
    this.MG = this.Za.Fa();
    this.Vz = this.Za.A.where;
    this.Gi = [];
    this.nj("_default");
    a = this.Za.Th() || [];
    for (b = 0; b < a.length; b++)
        this.nj(a[b]);
    this.vb.A[c] = this
};
_.h = _.Im.prototype;
_.h.ob = function() {
    return this.ce
};
_.h.ha = function() {
    if (!this.ob()) {
        for (var a = 0; a < this.Gi.length; a++)
            this.uh(this.Gi[a]);
        delete _.pm.A[this.Ic()];
        this.ce=!0
    }
};
_.h.getContext = function() {
    return this.vb
};
_.h.Ig = function() {
    return this.wf
};
_.h.Ic = function() {
    return this.ee
};
_.h.Fa = function() {
    return this.MG
};
_.h.Pa = function() {
    return this.Vz
};
_.h.Ud = function(a) {
    this.Vz = a
};
_.h.xf = function() {
    (0, this.Za.A._rpcReadyFn)()
};
_.h.wb = function(a, b) {
    this.Za.value()[a] = b
};
_.h.Qj = function(a) {
    return this.Za.value()[a]
};
_.h.kb = function() {
    return this.Za.value()
};
_.h.ca = function() {
    return this.Za.ca()
};
_.h.Ba = function() {
    return this.Wc
};
_.h.register = function(a, b, c) {
    _.Na(!this.ob(), "Cannot register handler on disposed iframe " + a);
    _.Na((c || _.zm)(this), "Rejecting untrusted message " + a);
    c = this.ee + ":" + this.vb.ee + ":" + a;
    1 == _.Ha(rm, c, []).push(b) && (this.Gi.push(a), _.Uk(c, Cm(c, this, "_g_wasClosed" === a)))
};
_.h.uh = function(a, b) {
    var c = this.ee + ":" + this.vb.ee + ":" + a, d = rm[c];
    if (d) {
        if (b) {
            var e = _.Mb.call(d, b);
            0 <= e && d.splice(e, 1)
        } else 
            d.splice(0, d.length);
        0 == d.length && (e = _.Mb.call(this.Gi, a), 0 <= e && this.Gi.splice(e, 1), _.Vk(c))
    }
};
_.h.eF = function() {
    return this.Gi
};
_.h.nj = function(a) {
    this.Fo = this.Fo || [];
    if (!(0 <= _.Mb.call(this.Fo, a))) {
        this.Fo.push(a);
        a = qm[a] || {
            map: {}
        };
        for (var b in a.map)
            _.Ma(a.map, b) && this.register(b, a.map[b], a.filter)
    }
};
_.h.send = function(a, b, c, d) {
    _.Na(!this.ob(), "Cannot send message to disposed iframe - " + a);
    _.Na((d || _.zm)(this), "Wrong target for message " + a);
    c = new mm(c);
    _.Yk(this.wf, this.vb.ee + ":" + this.ee + ":" + a, c.resolve, b);
    return c.promise
};
_.Jm = function(a, b, c, d) {
    return a.send(b, c, d, _.Am)
};
_.Im.prototype.fI = function(a) {
    return a
};
_.Im.prototype.ping = _.q(11);
rm = _.Ka();
qm = _.Ka();
_.pm = new _.Hm;
_.Fm("_g_rpcReady", _.Im.prototype.xf);
_.Fm("_g_discover", _.Im.prototype.eF);
_.Fm("_g_ping", _.Im.prototype.fI);
var Qm, Km;
_.Pm = _.Ka();
_.Mm = _.Ka();
_.Om = function(a) {
    return _.Pm[a]
};
Qm = function(a, b) {
    _.Sa.load("gapi.iframes.style." + a, b)
};
Km = function(a, b) {
    var c = b.A.style;
    if (c) {
        _.fm(b, null);
        var d = _.Pm[c];
        _.Na(d, "No such style: " + c);
        b.Ni(a);
        d(b.value());
        b.Ni(null)
    }
};
var Sm, Tm;
Sm = {
    height: !0,
    width: !0
};
Tm = /^(?!-*(?:expression|(?:moz-)?binding))(?:[.#]?-?(?:[_a-z0-9-]+)(?:-[_a-z0-9-]+)*-?|-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[a-z]{1,2}|%)?|!important|)$/i;
_.Um = function(a) {
    "number" === typeof a && (a = String(a) + "px");
    return a
};
_.Im.prototype.Ja = function() {
    if (!_.zm(this))
        return null;
    var a = this.Za.A._popupWindow;
    if (a)
        return a;
    for (var b = this.wf.split("/"), a = this.getContext().Ja(), c = 0; c < b.length && a; c++) {
        var d = b[c];
        ".." === d ? a = a == a.parent ? a.opener : a.parent : a = a.frames[d]
    }
    return a
};
var Vm = function(a, b) {
    var c = a.B, d=!0;
    b.filter && (d = b.filter.call(b.Re, b.params));
    return om(d).then(function(d) {
        if (d && c) {
            var f;
            b.Wx && b.Wx.call(a, b.params);
            b.gz ? f = b.gz(b.params) : f = _.Jm(c, b.message, b.params);
            return b.pK ? f.then(function() {
                return !0
            }) : !0
        }
        return !1
    })
};
_.Hm.prototype.F = function(a, b, c) {
    a = Vm(this, {
        gz: function(a) {
            var b = _.pm.B;
            _.ib(_.pm.A, function(c) {
                c !== b && _.Jm(c, "_g_wasClosed", a)
            });
            var c = _.Jm(b, "_g_closeMe", a);
            _.Jm(b, "_g_wasClosed", a);
            return c
        },
        message: "_g_closeMe",
        params: a,
        Re: c,
        filter: this.Qp("onCloseSelfFilter")
    });
    b = new mm(b);
    b.resolve(a);
    return b.promise
};
_.Hm.prototype.Y = function(a, b, c) {
    a = a || {};
    b = new mm(b);
    b.resolve(Vm(this, {
        message: "_g_restyleMe",
        params: a,
        Re: c,
        filter: this.Qp("onRestyleSelfFilter"),
        pK: !0,
        Wx: this.va
    }));
    return b.promise
};
_.Hm.prototype.va = function(a) {
    "auto" === a.height && (a.height = _.il.Jb())
};
_.Wm = function(a) {
    var b = {};
    if (a)
        for (var c in a)
            _.Ma(a, c) && _.Ma(Sm, c) && Tm.test(a[c]) && (b[c] = a[c]);
    return b
};
_.h = _.Im.prototype;
_.h.close = function(a, b) {
    return _.Jm(this, "_g_close", a, b)
};
_.h.Ji = function(a, b) {
    return _.Jm(this, "_g_restyle", a, b)
};
_.h.Ki = function(a, b) {
    return _.Jm(this, "_g_restyleDone", a, b)
};
_.h.DD = function(a) {
    return this.getContext().F(a, void 0, this)
};
_.h.II = function(a) {
    if (a && "object" === typeof a)
        return this.getContext().Y(a, void 0, this)
};
_.h.JI = function(a) {
    var b = this.Za.A.onRestyle;
    b && b.call(this, a, this);
    a = a && "object" === typeof a ? _.Wm(a) : {};
    (b = this.Fa()) && a && "object" === typeof a && (_.Ma(a, "height") && (a.height = _.Um(a.height)), _.Ma(a, "width") && (a.width = _.Um(a.width)), _.tl(a, b.style))
};
_.h.ED = function(a) {
    var b = this.Za.A.onClose;
    b && b.call(this, a, this);
    this.Bu && this.Bu() || (a = this.Fa()) && a.parentNode && a.parentNode.removeChild(a);
    if (a = this.Za.A.controller)
        b = {}, b.frameName = this.Ic(), _.Jm(a, "_g_disposeControl", b)
};
_.Hm.prototype.wa = _.q(12);
_.Hm.prototype.pa = _.q(13);
_.Im.prototype.Gy = _.q(14);
_.Im.prototype.gh = function(a, b) {
    this.register("_g_wasClosed", a, b)
};
_.Im.prototype.rK = function() {
    this.getContext().Ja().setTimeout((0, _.s)(function() {
        this.ha()
    }, this), 0)
};
_.Fm("_g_close", _.Im.prototype.DD);
_.Fm("_g_closeMe", _.Im.prototype.ED);
_.Fm("_g_restyle", _.Im.prototype.II);
_.Fm("_g_restyleMe", _.Im.prototype.JI);
_.Fm("_g_wasClosed", _.Im.prototype.rK);

var Ym, an, dn, en, nn, on, qn, rn, sn;
_.Xm = function(a, b) {
    a.A.onClose = b
};
Ym = function(a, b) {
    a.A.apis = b;
    return a
};
_.Zm = function(a, b) {
    a.A.messageHandlersFilter = b;
    return a
};
_.$m = function(a, b) {
    a.A.messageHandlers = b;
    return a
};
an = function(a, b) {
    a.A.rpctoken = b;
    return a
};
_.bn = function(a, b) {
    a.A.where = b;
    return a
};
_.cn = function(a, b) {
    a.A.anchor = b;
    return a
};
dn = function(a, b, c) {
    this.promise = a;
    this.resolve = b;
    this.reject = c
};
en = function(a) {
    this.xd = a || {}
};
en.prototype.value = function() {
    return this.xd
};
var fn = function(a) {
    var b = new en;
    b.xd.role = a;
    return b
};
en.prototype.tb = function(a) {
    this.xd.handler = a;
    return this
};
en.prototype.ua = function() {
    return this.xd.handler
};
var gn = function(a, b) {
    a.xd.filter = b;
    return a
}, hn = function(a, b) {
    a.xd.apis = b;
    return a
};
en.prototype.Th = function() {
    return this.xd.apis
};
var jn = function(a) {
    a.xd.runOnce=!0;
    return a
}, kn = function(a) {
    this.A = a || {}
};
kn.prototype.value = function() {
    return this.A
};
kn.prototype.getIframe = function() {
    return this.A.iframe
};
var ln = function(a, b) {
    a.A.role = b;
    return a
}, mn = function(a, b) {
    a.A.data = b;
    return a
};
kn.prototype.xf = function(a) {
    this.A.setRpcReady = a;
    return this
};
nn = function(a, b) {
    a.A.rpctoken = b;
    return a
};
on = function(a) {
    a.A.selfConnect=!0;
    return a
};
_.pn = function() {
    var a, b, c = new _.Qf(function(c, e) {
        a = c;
        b = e
    });
    return new dn(c, a, b)
};
qn = /^https?:\/\/[^\/%\\?#\s]+$/i;
rn = {
    longdesc: !0,
    name: !0,
    src: !0,
    frameborder: !0,
    marginwidth: !0,
    marginheight: !0,
    scrolling: !0,
    align: !0,
    height: !0,
    width: !0,
    id: !0,
    "class": !0,
    title: !0,
    tabindex: !0,
    hspace: !0,
    vspace: !0,
    allowtransparency: !0
};
sn = function(a, b, c) {
    var d = a.wf, e = b.Li;
    _.vm(_.um(c, a.Li + "/" + b.wf), e + "/" + d);
    _.tm(c, b.Ic()).we(b.Wc)
};
_.Hm.prototype.M = function(a, b) {
    var c = new kn(a), d = new kn(b), e = c.A.setRpcReady, f = c.getIframe(), g = d.getIframe();
    if (g) {
        var k = c.A.rpctoken, l = new _.sm;
        sn(f, g, l);
        mn(ln(nn(new kn(l.value()), k), c.A.role), c.A.data).xf(e);
        var m = new _.sm;
        sn(g, f, m);
        mn(ln(nn(new kn(m.value()), k), d.A.role), d.A.data).xf(!0);
        _.Jm(f, "_g_connect", l.value(), function() {
            e || _.Jm(g, "_g_connect", m.value())
        });
        e && _.Jm(g, "_g_connect", m.value())
    } else 
        d = {}, mn(ln(on(new kn(d)), c.A.role), c.A.data), _.Jm(f, "_g_connect", d)
};
_.h = _.Im.prototype;
_.h.GD = function(a) {
    var b, c = new _.sm(a);
    a = new kn(c.value());
    a.A.selfConnect ? b = this : (_.Na(qn.test(c.Ba()), "Illegal origin for connected iframe - " + c.Ba()), b = this.vb.A[c.Ic()], b) ? c.A.setRpcReady && (b.xf(), _.Jm(b, "_g_rpcReady")) : (c = _.tm(_.vm(_.um(an(new _.sm, _.jm(c)), c.Ig()), _.wm(c)).we(c.Ba()), c.Ic()).xf(c.A.setRpcReady), b = this.vb.Dj(c.value()));
    var c = this.vb, d = a.A.role;
    a = a.A.data;
    tn(c);
    d = d || "";
    _.Ha(c.K, d, []).push({
        Re: b.Ic(),
        data: a
    });
    un(b, a, c.N[d])
};
_.h.Yr = function(a, b) {
    if (!(new _.sm(b)).A._relayedDepth) {
        var c = {};
        on(ln(new kn(c), "_opener"));
        _.Jm(a, "_g_connect", c)
    }
};
_.h.Tx = function(a) {
    var b = this, c = a.A.messageHandlers, d = a.A.messageHandlersFilter, e = a.A.onClose;
    _.Xm(_.Zm(_.$m(a, null), null), null);
    _.pn();
    return _.Jm(this, "_g_open", a.value()).then(function(f) {
        var g = new _.sm(f[0]), k = g.Ic();
        f = new _.sm;
        var l = b.Li, m = _.wm(g);
        _.vm(_.um(f, b.wf + "/" + g.Ig()), m + "/" + l);
        _.tm(f, k);
        f.we(g.Ba());
        Ym(f, g.Th());
        an(f, _.jm(a));
        _.$m(f, c);
        _.Zm(f, d);
        _.Xm(f, e);
        (g = b.vb.A[k]) || (g = b.vb.Dj(f.value()));
        return g
    })
};
_.h.Kr = function(a) {
    var b = a.getUrl();
    _.Na(!b || _.vl.test(b), "Illegal url for new iframe - " + b);
    var c = a.Lj().value(), b = {}, d;
    for (d in c)
        _.Ma(c, d) && _.Ma(rn, d) && (b[d] = c[d]);
    _.Ma(c, "style") && (d = c.style, "object" === typeof d && (b.style = _.Wm(d)));
    a.value().attributes = b
};
_.h.TH = function(a) {
    a = new _.sm(a);
    this.Kr(a);
    var b = a.A._relayedDepth || 0;
    a.A._relayedDepth = b + 1;
    a.A.openerIframe = this;
    _.pn();
    var c = _.jm(a);
    an(a, null);
    return this.vb.open(a.value()).then((0, _.s)(function(a) {
        var e = (new _.sm(a.kb())).Th(), f = new _.sm;
        sn(a, this, f);
        0 == b && ln(new kn(f.value()), "_opener");
        f.xf(!0);
        an(f, c);
        _.Jm(a, "_g_connect", f.value());
        f = new _.sm;
        _.tm(_.vm(_.um(Ym(f, e), a.Ig()), a.Li), a.Ic()).we(a.Ba());
        return f.value()
    }, this))
};
var tn = function(a) {
    a.K || (a.K = _.Ka(), a.N = _.Ka())
};
_.Hm.prototype.G = function(a, b, c, d) {
    tn(this);
    "object" === typeof a ? (b = new en(a), c = b.xd.role || "") : (b = gn(hn(fn(a).tb(b), c), d), c = a);
    d = this.K[c] || [];
    a=!1;
    for (var e = 0; e < d.length&&!a; e++)
        un(this.A[d[e].Re], d[e].data, [b]), a = b.xd.runOnce;
    c = _.Ha(this.N, c, []);
    a || b.xd.dontWait || c.push(b)
};
_.Hm.prototype.ka = _.q(15);
var un = function(a, b, c) {
    c = c || [];
    for (var d = 0; d < c.length; d++) {
        var e = c[d];
        if (e && a) {
            var f = e.xd.filter || _.zm;
            if (a && f(a)) {
                for (var f = e.Th() || [], g = 0; g < f.length; g++)
                    a.nj(f[g]);
                e.ua() && e.ua()(a, b);
                e.xd.runOnce && (c.splice(d, 1), --d)
            }
        }
    }
};
_.Hm.prototype.D = function(a, b, c) {
    this.G(jn(gn(hn(fn("_opener").tb(a), b), c)).value())
};
_.Im.prototype.HI = function(a) {
    this.getContext().D(function(b) {
        b.send("_g_wasRestyled", a, void 0, _.Am)
    }, null, _.Am)
};
var vn = _.pm.B;
vn && vn.register("_g_restyleDone", _.Im.prototype.HI, _.Am);
_.Fm("_g_connect", _.Im.prototype.GD);
var wn = {};
wn._g_open = _.Im.prototype.TH;
_.Dm("_open", wn, _.Am);

_.Sb.oM = function(a) {
    var b = [];
    if (1 < arguments.length)
        for (var c = 0, d; d = arguments[c]; ++c)
            b.push(d);
    else 
        b = a;
    return function(a) {
        for (var c = 0; b[c]; ++c)
            if (a === b[c])
                return !0;
        return !1
    }
};
_.Sb.SP = function(a) {
    return function(b) {
        return a.test(b)
    }
};
_.Sb.SA = function(a) {
    return "undefined" !== typeof a
};
_.Sb.IO = function(a) {
    return "string" === typeof a && 0 < a.length
};
_.Sb.YK = function(a) {
    return "boolean" === typeof a
};
_.Sb.LN = function(a) {
    return function(b) {
        for (var c in a)
            if (a.hasOwnProperty(c)&&!(0, a[c])(b[c]))
                return !1;
        return !0
    }
};

_.H = _.H || {};
_.H.Vm = function(a, b, c) {
    for (var d = [], e = 2, f = arguments.length; e < f; ++e)
        d.push(arguments[e]);
    return function() {
        for (var c = d.slice(), e = 0, f = arguments.length; e < f; ++e)
            c.push(arguments[e]);
        return b.apply(a, c)
    }
};
_.H.rk = function(a) {
    var b, c, d = {};
    for (b = 0; c = a[b]; ++b)
        d[c] = c;
    return d
};

_.H = _.H || {};
(function() {
    function a(a) {
        b = a["core.util"] || {}
    }
    var b = {}, c = {};
    _.Sb && _.Sb.register("core.util", null, a);
    _.H.yS = function(a) {
        return "undefined" === typeof b[a] ? null : b[a]
    };
    _.H.hasFeature = function(a) {
        return "undefined" !== typeof b[a]
    };
    _.H.GS = function() {
        return c
    }
})();

var ig, jg, kg;
ig=!_.M || _.je(9);
jg=!_.Td&&!_.M || _.M && _.je(9) || _.Td && _.he("1.9.1");
kg = _.M&&!_.he("9");
_.lg = _.M || _.Sd || _.Ud;
_.mg = _.M&&!_.je(9);
var qg, sg, ug, Hg, Ig, Jg, vg;
_.og = function(a, b) {
    var c = b || window.document;
    return c.querySelectorAll && c.querySelector ? c.querySelectorAll("." + a) : _.ng(window.document, "*", a, b)
};
_.pg = function(a, b) {
    var c = b || window.document, d = null;
    c.querySelectorAll && c.querySelector ? d = c.querySelector("." + a) : d = _.ng(window.document, "*", a, b)[0];
    return d || null
};
_.ng = function(a, b, c, d) {
    a = d || a;
    b = b && "*" != b ? b.toUpperCase() : "";
    if (a.querySelectorAll && a.querySelector && (b || c))
        return a.querySelectorAll(b + (c ? "." + c : ""));
    if (c && a.getElementsByClassName) {
        a = a.getElementsByClassName(c);
        if (b) {
            d = {};
            for (var e = 0, f = 0, g; g = a[f]; f++)
                b == g.nodeName && (d[e++] = g);
            d.length = e;
            return d
        }
        return a
    }
    a = a.getElementsByTagName(b || "*");
    if (c) {
        d = {};
        for (f = e = 0; g = a[f]; f++)
            b = g.className, "function" == typeof b.split && _.id(b.split(/\s+/), c) && (d[e++] = g);
        d.length = e;
        return d
    }
    return a
};
_.rg = function(a, b) {
    _.Kd(b, function(b, d) {
        "style" == d ? a.style.cssText = b : "class" == d ? a.className = b : "for" == d ? a.htmlFor = b : d in qg ? a.setAttribute(qg[d], b) : _.Wc(d, "aria-") || _.Wc(d, "data-") ? a.setAttribute(d, b) : a[d] = b
    })
};
qg = {
    cellpadding: "cellPadding",
    cellspacing: "cellSpacing",
    colspan: "colSpan",
    frameborder: "frameBorder",
    height: "height",
    maxlength: "maxLength",
    role: "role",
    rowspan: "rowSpan",
    type: "type",
    usemap: "useMap",
    valign: "vAlign",
    width: "width"
};
_.tg = function(a, b, c) {
    return sg(window.document, arguments)
};
sg = function(a, b) {
    var c = b[0], d = b[1];
    if (!ig && d && (d.name || d.type)) {
        c = ["<", c];
        d.name && c.push(' name="', _.td(d.name), '"');
        if (d.type) {
            c.push(' type="', _.td(d.type), '"');
            var e = {};
            _.Jd(e, d);
            delete e.type;
            d = e
        }
        c.push(">");
        c = c.join("")
    }
    c = a.createElement(c);
    d && (_.ja(d) ? c.className = d : _.ha(d) ? c.className = d.join(" ") : _.rg(c, d));
    2 < b.length && ug(a, c, b, 2);
    return c
};
ug = function(a, b, c, d) {
    function e(c) {
        c && b.appendChild(_.ja(c) ? a.createTextNode(c) : c)
    }
    for (; d < c.length; d++) {
        var f = c[d];
        !_.Qc(f) || _.Tc(f) && 0 < f.nodeType ? e(f) : (0, _.dd)(vg(f) ? _.Ld(f) : f, e)
    }
};
_.xg = function(a, b) {
    ug(_.wg(a), a, arguments, 1)
};
_.yg = function(a) {
    for (var b; b = a.firstChild;)
        a.removeChild(b)
};
_.zg = function(a, b) {
    b.parentNode && b.parentNode.insertBefore(a, b)
};
_.Ag = function(a) {
    return a && a.parentNode ? a.parentNode.removeChild(a) : null
};
_.Bg = function(a) {
    var b, c = a.parentNode;
    if (c && 11 != c.nodeType) {
        if (a.removeNode)
            return a.removeNode(!1);
        for (; b = a.firstChild;)
            c.insertBefore(b, a);
        return _.Ag(a)
    }
};
_.Cg = function(a) {
    return jg && void 0 != a.children ? a.children : (0, _.ed)(a.childNodes, function(a) {
        return 1 == a.nodeType
    })
};
_.Dg = function(a) {
    if (void 0 != a.firstElementChild)
        a = a.firstElementChild;
    else 
        for (a = a.firstChild; a && 1 != a.nodeType;)
            a = a.nextSibling;
    return a
};
_.Eg = function(a) {
    return _.Tc(a) && 1 == a.nodeType
};
_.Fg = function(a, b) {
    if (a.contains && 1 == b.nodeType)
        return a == b || a.contains(b);
    if ("undefined" != typeof a.compareDocumentPosition)
        return a == b || Boolean(a.compareDocumentPosition(b) & 16);
    for (; b && a != b;)
        b = b.parentNode;
    return b == a
};
_.wg = function(a) {
    return 9 == a.nodeType ? a : a.ownerDocument || a.document
};
_.Gg = function(a, b) {
    if ("textContent"in a)
        a.textContent = b;
    else if (3 == a.nodeType)
        a.data = b;
    else if (a.firstChild && 3 == a.firstChild.nodeType) {
        for (; a.lastChild != a.firstChild;)
            a.removeChild(a.lastChild);
        a.firstChild.data = b
    } else 
        _.yg(a), a.appendChild(_.wg(a).createTextNode(String(b)))
};
Hg = {
    SCRIPT: 1,
    STYLE: 1,
    HEAD: 1,
    IFRAME: 1,
    OBJECT: 1
};
Ig = {
    IMG: " ",
    BR: "\n"
};
_.Kg = function(a) {
    if (kg && "innerText"in a)
        a = a.innerText.replace(/(\r\n|\r|\n)/g, "\n");
    else {
        var b = [];
        Jg(a, b, !0);
        a = b.join("")
    }
    a = a.replace(/ \xAD /g, " ").replace(/\xAD/g, "");
    a = a.replace(/\u200B/g, "");
    kg || (a = a.replace(/ +/g, " "));
    " " != a && (a = a.replace(/^\s*/, ""));
    return a
};
_.Lg = function(a) {
    var b = [];
    Jg(a, b, !1);
    return b.join("")
};
Jg = function(a, b, c) {
    if (!(a.nodeName in Hg))
        if (3 == a.nodeType)
            c ? b.push(String(a.nodeValue).replace(/(\r\n|\r|\n)/g, "")) : b.push(a.nodeValue);
        else if (a.nodeName in Ig)
            b.push(Ig[a.nodeName]);
        else 
            for (a = a.firstChild; a;)
                Jg(a, b, c), a = a.nextSibling
};
vg = function(a) {
    if (a && "number" == typeof a.length) {
        if (_.Tc(a))
            return "function" == typeof a.item || "string" == typeof a.item;
        if (_.Uc(a))
            return "function" == typeof a.item
    }
    return !1
};
_.Mg = function(a) {
    this.A = a || _.r.document || window.document
};
_.h = _.Mg.prototype;
_.h.I = _.q(0);
_.h.T = function(a, b, c) {
    return sg(this.A, arguments)
};
_.h.createElement = function(a) {
    return this.A.createElement(a)
};
_.h.Ja = function() {
    var a = this.A;
    return a.parentWindow || a.defaultView
};
_.h.appendChild = function(a, b) {
    a.appendChild(b)
};
_.h.append = _.xg;
_.h.Vv = _.yg;
_.h.cx = _.zg;
_.h.removeNode = _.Ag;
_.h.fE = _.Bg;
_.h.tE = _.Cg;
_.h.vv = _.Dg;
_.h.isElement = _.Eg;
_.h.contains = _.Fg;

_.pi = _.xd("area base br col command embed hr img input keygen link meta param source track wbr".split(" "));
_.ri = function() {
    this.A = "";
    this.B = _.qi
};
_.ri.prototype.Vg=!0;
_.qi = {};
_.ri.prototype.Kg = function() {
    return this.A
};
_.si = function(a) {
    var b = new _.ri;
    b.A = a;
    return b
};
_.ti = _.si("");
var ui;
_.vi = function() {
    this.A = "";
    this.B = ui
};
_.vi.prototype.Vg=!0;
_.vi.prototype.Kg = function() {
    return this.A
};
_.vi.prototype.Fq=!0;
_.vi.prototype.Cg = function() {
    return 1
};
_.wi = function(a) {
    return a instanceof _.vi && a.constructor === _.vi && a.B === ui ? a.A : "type_error:SafeUrl"
};
ui = {};
var xi;
_.yi = function() {
    this.A = "";
    this.C = xi;
    this.B = null
};
_.yi.prototype.Fq=!0;
_.yi.prototype.Cg = function() {
    return this.B
};
_.yi.prototype.Vg=!0;
_.yi.prototype.Kg = function() {
    return this.A
};
_.zi = function(a) {
    return a instanceof _.yi && a.constructor === _.yi && a.C === xi ? a.A : "type_error:SafeHtml"
};
_.Bi = function(a) {
    if (a instanceof _.yi)
        return a;
    var b = null;
    a.Fq && (b = a.Cg());
    return _.Ai(_.td(a.Vg ? a.Kg() : String(a)), b)
};
_.Ci = _.xd("action", "cite", "data", "formaction", "href", "manifest", "poster", "src");
_.Di = _.xd("link", "script", "style");
xi = {};
_.Ai = function(a, b) {
    var c = new _.yi;
    c.A = a;
    c.B = b;
    return c
};
_.Ei = _.Ai("", 0);

_.Fn = _.Ka();
var Nr;
_.Er = function(a) {
    return "CSS1Compat" == a.compatMode
};
_.Fr = function(a) {
    return !_.Ud && _.Er(a) ? a.documentElement : a.body || a.documentElement
};
_.Gr = function(a, b) {
    this.width = a;
    this.height = b
};
_.h = _.Gr.prototype;
_.h.clone = function() {
    return new _.Gr(this.width, this.height)
};
_.h.isEmpty = function() {
    return !(this.width * this.height)
};
_.h.ceil = _.q(4);
_.h.floor = function() {
    this.width = Math.floor(this.width);
    this.height = Math.floor(this.height);
    return this
};
_.h.round = function() {
    this.width = Math.round(this.width);
    this.height = Math.round(this.height);
    return this
};
_.Hr = function(a, b) {
    this.x = _.ca(a) ? a : 0;
    this.y = _.ca(b) ? b : 0
};
_.h = _.Hr.prototype;
_.h.clone = function() {
    return new _.Hr(this.x, this.y)
};
_.h.ceil = _.q(5);
_.h.floor = function() {
    this.x = Math.floor(this.x);
    this.y = Math.floor(this.y);
    return this
};
_.h.round = function() {
    this.x = Math.round(this.x);
    this.y = Math.round(this.y);
    return this
};
_.h.translate = _.q(6);
_.Ir = function(a) {
    var b = _.Fr(a);
    a = a.parentWindow || a.defaultView;
    return _.M && _.he("10") && a.pageYOffset != b.scrollTop ? new _.Hr(b.scrollLeft, b.scrollTop) : new _.Hr(a.pageXOffset || b.scrollLeft, a.pageYOffset || b.scrollTop)
};
_.Jr = function(a) {
    a = a.document;
    a = _.Er(a) ? a.documentElement : a.body;
    return new _.Gr(a.clientWidth, a.clientHeight)
};
_.Hr.prototype.ceil = _.L(5, function() {
    this.x = Math.ceil(this.x);
    this.y = Math.ceil(this.y);
    return this
});
_.Gr.prototype.ceil = _.L(4, function() {
    this.width = Math.ceil(this.width);
    this.height = Math.ceil(this.height);
    return this
});
_.Mg.prototype.I = _.L(0, function(a) {
    return _.ja(a) ? this.A.getElementById(a) : a
});
_.Kr = function(a) {
    return _.Ir(a.A)
};
_.Lr = function(a) {
    return _.Er(a.A)
};
_.Mr = function(a) {
    return _.Jr(a || window)
};
_.Or = function(a) {
    return a ? new _.Mg(_.wg(a)) : Nr || (Nr = new _.Mg)
};
_.Pr = function(a) {
    return String(a).replace(/([-()\[\]{}+?*.$\^|,:#<!\\])/g, "\\$1").replace(/\x08/g, "\\x08")
};
_.Qr = function(a, b) {
    b.parentNode && b.parentNode.insertBefore(a, b.nextSibling)
};
_.Rr = function(a) {
    return a ? a.parentWindow || a.defaultView : window
};
_.Sr = function(a) {
    return _.ja(a) ? window.document.getElementById(a) : a
};
_.Tr = function(a) {
    var b = _.ja(void 0) ? _.Pr(void 0): "\\s";
    return a.replace(new RegExp("(^" + (b ? "|[" + b + "]+" : "") + ")([a-z])", "g"), function(a, b, e) {
        return b + e.toUpperCase()
    })
};
_.Ur = function(a, b) {
    for (var c in b)
        a[c] = b[c]
};
var Vr;
Vr = function() {
    return _.Ud ? "Webkit" : _.Td ? "Moz" : _.M ? "ms" : _.Sd ? "O" : null
};
_.Wr = function() {
    return _.Ud ? "-webkit" : _.Td ? "-moz" : _.M ? "-ms" : _.Sd ? "-o" : null
};
var bs, ks, gs;
_.Q = function(a, b, c) {
    if (_.ja(b))(b = _.Xr(a, b)) && (a.style[b] = c);
    else 
        for (var d in b) {
            c = a;
            var e = b[d], f = _.Xr(c, d);
            f && (c.style[f] = e)
        }
};
_.Xr = function(a, b) {
    var c = _.$c(b);
    if (void 0 === a.style[c]) {
        var d = Vr() + _.Tr(c);
        if (void 0 !== a.style[d])
            return d
    }
    return c
};
_.Yr = function(a, b) {
    var c = _.wg(a);
    return c.defaultView && c.defaultView.getComputedStyle && (c = c.defaultView.getComputedStyle(a, null)) ? c[b] || c.getPropertyValue(b) || "" : ""
};
_.Zr = function(a, b) {
    return a.currentStyle ? a.currentStyle[b] : null
};
_.$r = function(a, b) {
    return _.Yr(a, b) || _.Zr(a, b) || a.style && a.style[b]
};
_.as = function(a) {
    return _.$r(a, "position")
};
bs = function(a) {
    var b;
    try {
        b = a.getBoundingClientRect()
    } catch (c) {
        return {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0
        }
    }
    _.M && a.ownerDocument.body && (a = a.ownerDocument, b.left -= a.documentElement.clientLeft + a.body.clientLeft, b.top -= a.documentElement.clientTop + a.body.clientTop);
    return b
};
_.cs = function(a) {
    if (_.M&&!_.je(8))
        return a.offsetParent;
    var b = _.wg(a), c = _.$r(a, "position"), d = "fixed" == c || "absolute" == c;
    for (a = a.parentNode; a && a != b; a = a.parentNode)
        if (c = _.$r(a, "position"), d = d && "static" == c && a != b.documentElement && a != b.body, !d && (a.scrollWidth > a.clientWidth || a.scrollHeight > a.clientHeight || "fixed" == c || "absolute" == c || "relative" == c))
            return a;
    return null
};
_.ds = function(a) {
    var b, c = _.wg(a), d = _.$r(a, "position"), e = _.Td && c.getBoxObjectFor&&!a.getBoundingClientRect && "absolute" == d && (b = c.getBoxObjectFor(a)) && (0 > b.screenX || 0 > b.screenY), f = new _.Hr(0, 0), g;
    b = c ? _.wg(c) : window.document;
    g=!_.M || _.je(9) || _.Lr(_.Or(b)) ? b.documentElement : b.body;
    if (a == g)
        return f;
    if (a.getBoundingClientRect)
        b = bs(a), a = _.Kr(_.Or(c)), f.x = b.left + a.x, f.y = b.top + a.y;
    else if (c.getBoxObjectFor&&!e)
        b = c.getBoxObjectFor(a), a = c.getBoxObjectFor(g), f.x = b.screenX - a.screenX, f.y = b.screenY - a.screenY;
    else {
        b =
        a;
        do {
            f.x += b.offsetLeft;
            f.y += b.offsetTop;
            b != a && (f.x += b.clientLeft || 0, f.y += b.clientTop || 0);
            if (_.Ud && "fixed" == _.as(b)) {
                f.x += c.body.scrollLeft;
                f.y += c.body.scrollTop;
                break
            }
            b = b.offsetParent
        }
        while (b && b != a);
        if (_.Sd || _.Ud && "absolute" == d)
            f.y -= c.body.offsetTop;
        for (b = a; (b = _.cs(b)) && b != c.body && b != g;)
            f.x -= b.scrollLeft, _.Sd && "TR" == b.tagName || (f.y -= b.scrollTop)
    }
    return f
};
_.fs = function(a, b) {
    var c = new _.Hr(0, 0), d = _.Rr(_.wg(a)), e = a;
    do {
        var f = d == b ? _.ds(e): _.es(e);
        c.x += f.x;
        c.y += f.y
    }
    while (d && d != b && (e = d.frameElement) && (d = d.parent));
    return c
};
_.es = function(a) {
    var b;
    if (a.getBoundingClientRect)
        b = bs(a), b = new _.Hr(b.left, b.top);
    else {
        b = _.Kr(_.Or(a));
        var c = _.ds(a);
        b = new _.Hr(c.x - b.x, c.y - b.y)
    }
    if (_.Td&&!_.he(12)) {
        i:
        {
            c = _.$c("transform");
            if (void 0 === a.style[c] && (c = Vr() + _.Tr(c), void 0 !== a.style[c])) {
                c = _.Wr() + "-transform";
                break i
            }
            c = "transform"
        }
        a = (a = _.$r(a, c) || _.$r(a, "transform")) ? (a = a.match(gs)) ? new _.Hr((0, window.parseFloat)(a[1]), (0, window.parseFloat)(a[2])) : new _.Hr(0, 0) : new _.Hr(0, 0);
        a = new _.Hr(b.x + a.x, b.y + a.y)
    } else 
        a = b;
    return a
};
_.js = function(a, b, c) {
    if (b instanceof _.Gr)
        c = b.height, b = b.width;
    else if (void 0 == c)
        throw Error("K");
    a.style.width = _.hs(b, !0);
    _.is(a, c)
};
_.hs = function(a, b) {
    "number" == typeof a && (a = (b ? Math.round(a) : a) + "px");
    return a
};
_.is = function(a, b) {
    a.style.height = _.hs(b, !0)
};
_.ls = function(a) {
    var b = ks;
    if ("none" != _.$r(a, "display"))
        return b(a);
    var c = a.style, d = c.display, e = c.visibility, f = c.position;
    c.visibility = "hidden";
    c.position = "absolute";
    c.display = "inline";
    a = b(a);
    c.display = d;
    c.position = f;
    c.visibility = e;
    return a
};
ks = function(a) {
    var b = a.offsetWidth, c = a.offsetHeight, d = _.Ud&&!b&&!c;
    return _.ca(b)&&!d ||!a.getBoundingClientRect ? new _.Gr(b, c) : (a = bs(a), new _.Gr(a.right - a.left, a.bottom - a.top))
};
_.ns = function(a) {
    var b = _.Or(void 0), c = null, d = b.A;
    _.M && d.createStyleSheet ? (c = d.createStyleSheet(), _.ms(c, a)) : (d = _.ng(b.A, "head", void 0, void 0)[0], d || (c = _.ng(b.A, "body", void 0, void 0)[0], d = b.T("head"), c.parentNode.insertBefore(d, c)), c = b.T("style"), _.ms(c, a), b.appendChild(d, c));
    return c
};
_.ms = function(a, b) {
    _.M && _.ca(a.cssText) ? a.cssText = b : a.innerHTML = b
};
_.os = _.Td ? "MozUserSelect" : _.Ud ? "WebkitUserSelect" : null;
gs = /matrix\([0-9\.\-]+, [0-9\.\-]+, [0-9\.\-]+, [0-9\.\-]+, ([0-9\.\-]+)p?x?, ([0-9\.\-]+)p?x?\)/;

_.Ku = function(a) {
    if (8192 > a.length)
        return String.fromCharCode.apply(null, a);
    for (var b = "", c = 0; c < a.length; c += 8192)
        b += String.fromCharCode.apply(null, _.Gd(a, c, c + 8192));
    return b
};
var Mu, Nu, Ou;
_.Lu = null;
Mu = null;
Nu = null;
Ou = null;
_.Pu = _.Td || _.Ud || _.Sd || "function" == typeof _.r.atob;
_.Ru = function(a, b) {
    _.Qu();
    for (var c = b ? Ou : Mu, d = [], e = 0; e < a.length;) {
        var f = c[a.charAt(e++)], g = e < a.length ? c[a.charAt(e)]: 0;
        ++e;
        var k = e < a.length ? c[a.charAt(e)]: 64;
        ++e;
        var l = e < a.length ? c[a.charAt(e)]: 64;
        ++e;
        if (null == f || null == g || null == k || null == l)
            throw Error();
        d.push(f<<2 | g>>4);
        64 != k && (d.push(g<<4 & 240 | k>>2), 64 != l && d.push(k<<6 & 192 | l))
    }
    return d
};
_.Qu = function() {
    if (!_.Lu) {
        _.Lu = {};
        Mu = {};
        Nu = {};
        Ou = {};
        for (var a = 0; 65 > a; a++)
            _.Lu[a] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".charAt(a), Mu[_.Lu[a]] = a, Nu[a] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.".charAt(a), Ou[Nu[a]] = a, 62 <= a && (Mu["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.".charAt(a)] = a, Ou["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".charAt(a)] = a)
    }
};

_.Su = function(a) {
    this.A = a
};
_.Su.prototype.toString = function() {
    return this.A
};
_.Tu = function() {
    _.te.call(this);
    this.A = [];
    this.B = {}
};
_.u(_.Tu, _.te);
_.h = _.Tu.prototype;
_.h.$v = 1;
_.h.nn = 0;
_.h.subscribe = _.q(26);
_.h.Af = function(a) {
    if (0 != this.nn)
        return this.C || (this.C = []), this.C.push(a), !1;
    var b = this.A[a];
    if (b) {
        var c = this.B[b];
        c && _.se(c, a);
        delete this.A[a];
        delete this.A[a + 1];
        delete this.A[a + 2]
    }
    return !!b
};
_.h.mm = function(a, b) {
    var c = this.B[a];
    if (c) {
        this.nn++;
        for (var d = _.Gd(arguments, 1), e = 0, f = c.length; e < f; e++) {
            var g = c[e];
            this.A[g + 1].apply(this.A[g + 2], d)
        }
        this.nn--;
        if (this.C && 0 == this.nn)
            for (; c = this.C.pop();)
                this.Af(c);
        return 0 != e
    }
    return !1
};
_.h.clear = function(a) {
    if (a) {
        var b = this.B[a];
        b && ((0, _.dd)(b, this.Af, this), delete this.B[a])
    } else 
        this.A.length = 0, this.B = {}
};
_.h.fb = function(a) {
    if (a) {
        var b = this.B[a];
        return b ? b.length : 0
    }
    a = 0;
    for (b in this.B)
        a += this.fb(b);
    return a
};
_.h.W = function() {
    _.Tu.J.W.call(this);
    delete this.A;
    delete this.B;
    delete this.C
};
_.Uu = function() {
    _.te.call(this);
    this.A = new _.Tu;
    _.ve(this, this.A)
};
_.u(_.Uu, _.te);
_.Uu.prototype.subscribe = _.q(25);
_.Uu.prototype.Af = function(a) {
    return this.A.Af(a)
};
_.Uu.prototype.clear = function(a) {
    this.A.clear(_.ca(a) ? a.toString() : void 0)
};
_.Uu.prototype.fb = function(a) {
    return this.A.fb(_.ca(a) ? a.toString() : void 0)
};

var Vu = {}, Wu = function() {
    this.A = Vu
}, av, bv;
Wu.prototype.Vg=!0;
Wu.prototype.Kg = function() {
    return ""
};
Wu.prototype.toString = function() {
    return "Const{}"
};
var Xu = {}, Yu = /^[-.%_!# a-zA-Z0-9]+$/, Zu = function(a) {
    return a instanceof Wu && a.constructor === Wu && a.A === Vu ? "" : "type_error:Const"
}, $u = function() {
    this.A = Xu
};
$u.prototype.Vg=!0;
$u.prototype.Kg = function() {
    return ""
};
$u.prototype.Fq=!0;
$u.prototype.Cg = function() {
    return 1
};
av = function(a) {
    var b = 0, c = "", d = function(a) {
        _.ha(a) ? (0, _.dd)(a, d) : (a = _.Bi(a), c += _.zi(a), a = a.Cg(), 0 == b ? b = a : 0 != a && b != a && (b = null))
    };
    (0, _.dd)(arguments, d);
    return _.Ai(c, b)
};
bv = /^[a-zA-Z0-9-]+$/;
_.cv = function(a, b) {
    _.ha(b) || (b = [b]);
    var c = (0, _.fd)(b, function(a) {
        return _.ja(a) ? a : a.mn + " " + a.duration + "s " + a.timing + " " + a.vg + "s"
    });
    _.Q(a, "transition", c.join(","))
};
_.dv = function(a) {
    var b=!1, c;
    return function() {
        b || (c = a(), b=!0);
        return c
    }
}(function() {
    if (_.M)
        return _.he("10.0");
    var a = window.document.createElement("div"), b = _.Wr(), c = {
        transition: "opacity 1s linear"
    };
    b && (c[b + "-transition"] = "opacity 1s linear");
    b = {
        style: c
    };
    if (!bv.test("div"))
        throw Error("v`div");
    if ("div"in _.Di)
        throw Error("w`div");
    var c = null, d = "<div";
    if (b)
        for (var e in b) {
            if (!bv.test(e))
                throw Error("x`" + e);
                var f = b[e];
                if (null != f) {
                    var g;
                    g = e;
                    if (f instanceof Wu)
                        f = Zu(f);
                    else if ("style" == g.toLowerCase()) {
                        if (!_.Tc(f))
                            throw Error("A`" +
                            typeof f + "`" + f);
                            if (!(f instanceof _.ri)) {
                                var k = "", l = void 0;
                                for (l in f) {
                                    if (!/^[-_a-zA-Z0-9]+$/.test(l))
                                        throw Error("u`" + l);
                                        var m = f[l];
                                        null != m && (m instanceof Wu ? m = Zu(m) : Yu.test(m) || (m = "zClosurez"), k += l + ":" + m + ";")
                                    }
                                    f = k ? _.si(k) : _.ti
                            }
                            f = f instanceof _.ri && f.constructor === _.ri && f.B === _.qi ? f.A : "type_error:SafeStyle"
                        } else {
                            if (/^on/i.test(g))
                                throw Error("y`" + g + "`" + f);
                                if (g.toLowerCase()in _.Ci)
                                    if (f instanceof $u)
                                        f = f instanceof $u && f.constructor === $u && f.A === Xu ? "" : "type_error:TrustedResourceUrl";
                                    else if (f instanceof
                                    _.vi)
                                        f = _.wi(f);
                                    else 
                                        throw Error("z`" + g + "`div`" + f);
                        }
                        f.Vg && (f = f.Kg());
                        g = g + '="' + _.td(String(f)) + '"';
                        d = d + (" " + g)
                    }
            }
    e = void 0;
    _.ca(e) ? _.ha(e) || (e = [e]) : e = [];
    !0 === _.pi.div ? d += ">" : (c = av(e), d += ">" + _.zi(c) + "</div>", c = c.Cg());
    (b = b && b.dir) && (/^(ltr|rtl|auto)$/i.test(b) ? c = 0 : c = null);
    b = _.Ai(d, c);
    a.innerHTML = _.zi(b);
    a = a.firstChild;
    b = a.style[_.$c("transition")];
    return "" != ("undefined" !== typeof b ? b : a.style[_.Xr(a, "transition")] || "")
});

_.gv = function() {
    _.fv = "oauth2relay" + String(2147483647 * (0, _.Nj)() | 0);
    _.ev.proxy = _.fv
};
_.hv = new _.Uu;
_.iv = new _.Su("oauth");
_.ev = {};
_.gv();
var jv = _.E("oauth-flow/client_id");
_.ev.client_id = jv;
var kv = _.ev, lv;
var mv = String(_.E("oauth-flow/redirectUri"));
if (mv)
    lv = mv.replace(/[#][\s\S]*/, "");
else {
    var nv = _.Mi.Ba(window.location.href);
    lv = [_.E("oauth-flow/callbackUrl"), "?x_origin=", (0, window.encodeURIComponent)(nv)].join("")
}
kv.redirect_uri = lv;
_.ev.origin = _.Mi.Ba(window.location.href);
_.ev.response_type = "token";

_.xn = _.Ha(_.Ya, "rw", _.Ka());
var yn = function(a, b) {
    var c = _.xn[a];
    c && c.state < b && (c.state = b)
};
var zn = function(a) {
    a = (a = _.xn[a]) ? a.oid : void 0;
    if (a) {
        var b = _.Ba.getElementById(a);
        b && b.parentNode.removeChild(b);
        delete _.xn[a];
        zn(a)
    }
};
_.An = function(a) {
    a = a.container;
    "string" === typeof a && (a = window.document.getElementById(a));
    return a
};
_.Bn = function(a) {
    var b = a.clientWidth;
    return "position:absolute;top:-10000px;width:" + (b ? b + "px" : a.style.width || "300px") + ";margin:0px;border-style:none;"
};
_.Cn = function(a, b) {
    var c = {}, d = a.kb(), e = b && b.width, f = b && b.height, g = b && b.verticalAlign;
    g && (c.verticalAlign = g);
    e || (e = d.width || a.width);
    f || (f = d.height || a.height);
    d.width = c.width = e;
    d.height = c.height = f;
    d = a.Fa();
    e = a.ca();
    yn(e, 2);
    t: {
        e = a.Pa();
        c = c || {};
        if (_.Ya.oa) {
            var k = d.id;
            if (k) {
                f = (f = _.xn[k]) ? f.state : void 0;
                if (1 === f || 4 === f)
                    break t;
                zn(k)
            }
        }(f = e.nextSibling) && f.getAttribute && f.getAttribute("data-gapistub") && (e.parentNode.removeChild(f), e.style.cssText = "");
        var f = c.width, g = c.height, l = e.style;
        l.textIndent = "0";
        l.margin =
        "0";
        l.padding = "0";
        l.background = "transparent";
        l.borderStyle = "none";
        l.cssFloat = "none";
        l.styleFloat = "none";
        l.lineHeight = "normal";
        l.fontSize = "1px";
        l.verticalAlign = "baseline";
        e = e.style;
        e.display = "inline-block";
        d = d.style;
        d.position = "static";
        d.left = 0;
        d.top = 0;
        d.visibility = "visible";
        f && (e.width = d.width = f + "px");
        g && (e.height = d.height = g + "px");
        c.verticalAlign && (e.verticalAlign = c.verticalAlign);
        k && yn(k, 3)
    }(k = b ? b.title : null) && a.Fa().setAttribute("title", k)
};
_.Dn = function(a) {
    var b = a.Pa();
    b && b.removeChild(a.Fa())
};
_.En = function(a) {
    a.where = _.An(a);
    var b = a.messageHandlers = a.messageHandlers || {}, c = function(a) {
        _.Cn(this, a)
    };
    b._ready = c;
    b._renderstart = c;
    var d = a.onClose;
    a.onClose = function(a) {
        d && d.call(this, a);
        _.Dn(this)
    };
    a.onCreate = function(a) {
        a = a.Fa();
        a.style.cssText = _.Bn(a)
    }
};

_.K.ta = _.K.ta || {};
_.K.ta.uD = function(a) {
    try {
        return !!a.document
    } catch (b) {}
    return !1
};
_.K.ta.Rv = function(a) {
    var b = a.parent;
    return a != b && _.K.ta.uD(b) ? _.K.ta.Rv(b) : a
};
_.K.ta.MS = function(a) {
    var b = a.userAgent || "";
    a = a.product || "";
    return 0 != b.indexOf("Opera")&&-1 == b.indexOf("WebKit") && "Gecko" == a && 0 < b.indexOf("rv:1.")
};

var Eq = function() {
    _.K.Uw++;
    return ["I", _.K.Uw, "_", (new Date).getTime()].join("")
}, ar;
var Fq = function(a) {
    return a instanceof Array ? a.join(",") : a instanceof Object ? (0, _.$b)(a) : a
};
var Gq = function() {};
var Hq = function(a) {
    a && a.match(Iq) && _.eb("googleapis.config/gcv", a)
};
var Jq = function(a) {
    _.Mi.Jo.vJ(a)
};
var Kq = function(a) {
    _.Mi.Jo.Rr(a)
};
_.Lq = function(a, b) {
    var c = b || {}, d;
    for (d in a)
        a.hasOwnProperty(d) && (c[d] = a[d]);
    return c
};
_.Mq = function(a, b, c, d, e) {
    var f = [], g;
    for (g in a)
        if (a.hasOwnProperty(g)) {
            var k = b, l = c, m = a[g], n = d, p = Nq(g);
            p[k] = p[k] || {};
            n = _.H.Vm(n, m);
            m._iframe_wrapped_rpc_ && (n._iframe_wrapped_rpc_=!0);
            p[k][l] = n;
            f.push(g)
        }
    if (e)
        for (g in _.K.gi)
            _.K.gi.hasOwnProperty(g) && f.push(g);
    return f.join(",")
};
var Oq = function(a, b, c) {
    var d = {};
    if (a && a._methods) {
        a = a._methods.split(",");
        for (var e = 0; e < a.length; e++) {
            var f = a[e];
            d[f] = Pq(f, b, c)
        }
    }
    return d
};
var Qq = function(a) {
    if (a && a.disableMultiLevelParentRelay)
        a=!1;
    else {
        var b;
        if (b = _.Pc && _.Pc._open && "inline" != a.style&&!0 !== a.inline)
            a = a.container, b=!(a && ("string" == typeof a && window.document.getElementById(a) || window.document == (a.ownerDocument || a.document)));
        a = b
    }
    return a
};
var Rq = function(a, b) {
    var c = {}, d = b.params || {}, e;
    for (e in a)
        "#" == e.charAt(0) && (c[e.substring(1)] = a[e]), 0 == e.indexOf("fr-") && (c[e.substring(3)] = a[e]), "#" == d[e] && (c[e] = a[e]);
    for (var f in c)
        delete a["fr-" + f], delete a["#" + f], delete a[f];
    return c
};
var Sq = function(a) {
    if (":" == a.charAt(0)) {
        var b = _.E("iframes/" + a.substring(1));
        a = {};
        _.tl(b, a);
        (b = a.url) && (a.url = _.Zl(b));
        a.params || (a.params = {});
        return a
    }
    return {
        url: _.Zl(a)
    }
};
var Tq = function(a) {
    function b() {}
    b.prototype = Uq.prototype;
    a.prototype = new b
};
var Vq = function(a) {
    return _.K.Hk[a]
};
var Wq = function(a, b) {
    _.K.Hk[a] = b
};
var Xq = function(a) {
    a = a || {};
    "auto" === a.height && (a.height = _.il.Jb());
    var b = window && Yq && Yq.ja();
    b ? b.Ny(a.width || 0, a.height || 0) : _.Pc && _.Pc._resizeMe && _.Pc._resizeMe(a)
};
var Zq = function(a) {
    Hq(a)
};
ar = function(a) {
    var b = _.Ua(a.location.href, "urlindex");
    if (b = _.Ha(_.Ya, "fUrl", [])[b]) {
        var c = a.location.hash, b = b + (/#/.test(b) ? c.replace(/^#/, "&") : c);
        a.location.replace(b)
    }
};
_.br = function() {
    return _.Aa.location.origin || _.Aa.location.protocol + "//" + _.Aa.location.host
};
if (window.ToolbarApi)
    Yq = window.ToolbarApi, Yq.ja = window.ToolbarApi.getInstance, Yq.prototype = window.ToolbarApi.prototype, _.h = Yq.prototype, _.h.Ux = Yq.prototype.openWindow, _.h.Cu = Yq.prototype.closeWindow, _.h.Dz = Yq.prototype.setOnCloseHandler, _.h.tu = Yq.prototype.canClosePopup, _.h.Ny = Yq.prototype.resizeWindow;
else {
    var Yq = function() {}, cr = null;
    Yq.ja = function() {
        !cr && window.external && window.external.GTB_IsToolbar && (cr = new Yq);
        return cr
    };
    _.h = Yq.prototype;
    _.h.Ux = function(a) {
        return window.external.GTB_OpenPopup &&
        window.external.GTB_OpenPopup(a)
    };
    _.h.Cu = function(a) {
        window.external.GTB_ClosePopupWindow && window.external.GTB_ClosePopupWindow(a)
    };
    _.h.Dz = function(a, b) {
        window.external.GTB_SetOnCloseHandler && window.external.GTB_SetOnCloseHandler(a, b)
    };
    _.h.tu = function(a) {
        return window.external.GTB_CanClosePopup && window.external.GTB_CanClosePopup(a)
    };
    _.h.Ny = function(a, b) {
        return window.external.GTB_ResizeWindow && window.external.GTB_ResizeWindow(a, b)
    };
    window.ToolbarApi = Yq;
    window.ToolbarApi.getInstance = Yq.ja
};
var dr = function() {
    _.J.register("_noop_echo", function() {
        this.callback(_.K.XE(_.K.Uf[this.f]))
    })
}, er = function() {
    window.setTimeout(function() {
        _.J.call("..", "_noop_echo", _.K.XH)
    }, 0)
}, Pq = function(a, b, c) {
    var d = function(d) {
        var f = Array.prototype.slice.call(arguments, 0), g = f[f.length - 1];
        if ("function" === typeof g) {
            var k = g;
            f.pop()
        }
        f.unshift(b, a, k, c);
        _.J.call.apply(_.J, f)
    };
    d._iframe_wrapped_rpc_=!0;
    return d
}, Nq = function(a) {
    _.K.tn[a] || (_.K.tn[a] = {}, _.J.register(a, function(b, c) {
        var d = this.f;
        if (!("string" != typeof b ||
        b in{}
        || d in{})) {
            var e = this.callback, f = _.K.tn[a][d], g;
            f && Object.hasOwnProperty.call(f, b) ? g = f[b] : Object.hasOwnProperty.call(_.K.gi, a) && (g = _.K.gi[a]);
            if (g)
                return d = Array.prototype.slice.call(arguments, 1), g._iframe_wrapped_rpc_ && e && d.push(e), g.apply({}, d)
        }
        _.Ib(['Unregistered call in window "', window.name, '" for method "', a, '", via proxyId "', b, '" from frame "', d, '".'].join(""));
        return null
    }));
    return _.K.tn[a]
};
_.K.rD = function(a, b, c) {
    var d = Array.prototype.slice.call(arguments);
    _.K.Fv(function(a) {
        a.sameOrigin && (d.unshift("/" + a.claimedOpenerId + "|" + window.location.protocol + "//" + window.location.host), _.J.call.apply(_.J, d))
    })
};
_.K.pI = function(a, b) {
    _.J.register(a, b)
};
var Iq = /^[-_.0-9A-Za-z]+$/, fr = {
    open: "open",
    onready: "ready",
    close: "close",
    onresize: "resize",
    onOpen: "open",
    onReady: "ready",
    onClose: "close",
    onResize: "resize",
    onRenderStart: "renderstart"
}, gr = {
    onBeforeParentOpen: "beforeparentopen"
}, hr = {
    onOpen: function(a) {
        var b = a.kb();
        a.Gd(b.container || b.element);
        return a
    },
    onClose: function(a) {
        a.remove()
    }
};
_.K.Lj = function(a) {
    var b = _.Ka();
    _.tl(_.Nl, b);
    _.tl(a, b);
    return b
};
var Uq = function(a, b, c, d, e, f, g, k) {
    this.config = Sq(a);
    this.openParams = this.tk = b || {};
    this.params = c || {};
    this.methods = d;
    this.Kn=!1;
    ir(this, b.style);
    this.sj = {};
    jr(this, function() {
        var a;
        (a = this.tk.style) && _.K.Hk[a] ? a = _.K.Hk[a] : a ? (_.Jb(['Missing handler for style "', a, '". Continuing with default handler.'].join("")), a = null) : a = hr;
        if (a) {
            var b;
            if ("function" === typeof a)
                b = a(this);
            else {
                var c = {};
                for (b in a) {
                    var d = a[b];
                    c[b] = "function" === typeof d ? _.H.Vm(a, d, this) : d
                }
                b = c
            }
            for (var g in e)
                a = b[g], "function" === typeof a &&
                kr(this, e[g], _.H.Vm(b, a))
        }
        f && kr(this, "close", f)
    });
    this.vf = this.ac = g;
    this.jr = (k || []).slice();
    g && this.jr.unshift(g.ca())
};
Uq.prototype.kb = function() {
    return this.tk
};
Uq.prototype.He = function() {
    return this.params
};
Uq.prototype.dm = function() {
    return this.methods
};
Uq.prototype.Kb = function() {
    return this.vf
};
var ir = function(a, b) {
    if (!a.Kn) {
        var c = b&&!_.K.Hk[b] && _.K.Xo[b];
        c ? (a.Wo = [], c(function() {
            a.Kn=!0;
            for (var b = 0, c = a.Wo.length; b < c; ++b)
                a.Wo[b].call(a)
        })) : a.Kn=!0
    }
}, jr = function(a, b) {
    a.Kn ? b.call(a) : a.Wo.push(b)
};
Uq.prototype.Ob = function(a, b) {
    jr(this, function() {
        kr(this, a, b)
    })
};
var kr = function(a, b, c) {
    a.sj[b] = a.sj[b] || [];
    a.sj[b].push(c)
};
Uq.prototype.hh = function(a, b) {
    jr(this, function() {
        var c = this.sj[a];
        if (c)
            for (var d = 0, e = c.length; d < e; ++d)
                if (c[d] === b) {
                    c.splice(d, 1);
                    break
                }
    })
};
Uq.prototype.le = function(a, b) {
    var c, d = this.sj[a];
    if (d)
        for (var e = Array.prototype.slice.call(arguments, 1), f = 0, g = d.length; f < g; ++f)
            try {
                c = d[f].apply({}, e)
    } catch (k) {
        _.Ib(['Exception when calling callback "', a, '" with exception "', k.name, ": ", k.message, '".'].join(""))
    }
    return c
};
var lr = function(a) {
    return "number" == typeof a ? {
        value: a,
        Gp: a + "px"
    } : "100%" == a ? {
        value: 100,
        Gp: "100%",
        lx: !0
    } : null
}, mr = function(a, b, c, d, e, f, g) {
    Uq.call(this, a, b, c, d, fr, e, f, g);
    this.id = b.id || Eq();
    this.B = b.rpctoken && String(b.rpctoken) || Math.round(1E9 * (0, _.Nj)());
    this.D = Rq(this.params, this.config);
    this.Bp = {};
    jr(this, function() {
        this.le("open");
        _.Lq(this.Bp, this)
    })
};
Tq(mr);
_.h = mr.prototype;
_.h.Gd = function(a, b) {
    if (!this.config.url)
        return _.Ib("Cannot open iframe, empty URL."), this;
    var c = this.id;
    _.K.Uf[c] = this;
    var d = _.Lq(this.methods);
    d._ready = this.fn;
    d._close = this.close;
    d._open = this.kt;
    d._resizeMe = this.lt;
    d._renderstart = this.Px;
    var e = this.D;
    this.B && (e.rpctoken = this.B);
    e._methods = _.Mq(d, c, "", this, !0);
    this.C = a = "string" === typeof a ? window.document.getElementById(a) : a;
    d = {};
    d.id = c;
    if (b) {
        d.attributes = b;
        var f = b.style;
        if ("string" === typeof f) {
            var g;
            if (f) {
                g = [];
                for (var f = f.split(";"), k = 0, l = f.length; k <
                l; ++k) {
                    var m = f[k];
                    if (0 != m.length || k + 1 != l)
                        m = m.split(":"), 2 == m.length && m[0].match(/^[ a-zA-Z_-]+$/) && m[1].match(/^[ +.%0-9a-zA-Z_-]+$/) ? g.push(m.join(":")) : _.Ib(['Iframe style "', f[k], '" not allowed.'].join(""))
                    }
                g = g.join(";")
            } else 
                g = "";
            b.style = g
        }
    }
    this.kb().allowPost && (d.allowPost=!0);
    d.queryParams = this.params;
    d.fragmentParams = e;
    d.paramsSerializer = Fq;
    this.A = _.am(this.config.url, a, d);
    e = this.A.getAttribute("data-postorigin") || this.A.src;
    _.K.Uf[c] = this;
    _.J.xn(this.id, this.B);
    _.J.Ze(this.id, e);
    return this
};
_.h.Nc = function(a, b) {
    this.Bp[a] = b
};
_.h.ca = function() {
    return this.id
};
_.h.Fa = function() {
    return this.A
};
_.h.Pa = function() {
    return this.C
};
_.h.Ud = function(a) {
    this.C = a
};
_.h.fn = function(a) {
    var b = Oq(a, this.id, "");
    this.vf && "function" == typeof this.methods._ready && (a._methods = _.Mq(b, this.vf.ca(), this.id, this, !1), this.methods._ready(a));
    _.Lq(a, this);
    _.Lq(b, this);
    this.le("ready", a)
};
_.h.Px = function(a) {
    this.le("renderstart", a)
};
_.h.close = function(a) {
    a = this.le("close", a);
    delete _.K.Uf[this.id];
    return a
};
_.h.remove = function() {
    var a = window.document.getElementById(this.id);
    a && a.parentNode && a.parentNode.removeChild(a)
};
_.h.kt = function(a) {
    var b = Oq(a.params, this.id, a.proxyId);
    delete a.params._methods;
    "_parent" == a.openParams.anchor && (a.openParams.anchor = this.C);
    if (Qq(a.openParams))
        new nr(a.url, a.openParams, a.params, b, b._onclose, this, a.openedByProxyChain);
    else {
        var c = new mr(a.url, a.openParams, a.params, b, b._onclose, this, a.openedByProxyChain), d = this;
        jr(c, function() {
            var a = {
                childId: c.ca()
            }, f = c.Bp;
            f._toclose = c.close;
            a._methods = _.Mq(f, d.id, c.id, c, !1);
            b._onopen(a)
        })
    }
};
_.h.lt = function(a) {
    if (void 0 === this.le("resize", a) && this.A) {
        var b = lr(a.width);
        null != b && (this.A.style.width = b.Gp);
        a = lr(a.height);
        null != a && (this.A.style.height = a.Gp);
        this.A.parentElement && (null != b && b.lx || null != a && a.lx) && (this.A.parentElement.style.display = "block")
    }
};
var nr = function(a, b, c, d, e, f, g) {
    Uq.call(this, a, b, c, d, gr, e, f, g);
    this.url = a;
    this.A = null;
    this.wr = Eq();
    jr(this, function() {
        var a;
        this.le("beforeparentopen");
        var b = _.Lq(this.methods);
        b._onopen = this.SH;
        b._ready = this.fn;
        b._onclose = this.RH;
        this.params._methods = _.Mq(b, "..", this.wr, this, !0);
        b = {};
        for (a in this.params)
            b[a] = Fq(this.params[a]);
        var c = this.config.url;
        if (this.tk.hideUrlFromParent) {
            a = window.name;
            var d = c, c = _.Gl(this.config.url, this.params, {}, Fq), e = b, b = {};
            b._methods = e._methods;
            b["#opener"] = e["#opener"];
            b["#urlindex"] = e["#urlindex"];
            b["#opener"] && void 0 != e["#urlindex"] ? (b["#opener"] = a + "," + b["#opener"], a = d) : (d = _.Ha(_.Ya, "fUrl", []), e = d.length, d[e] = c, _.Ya.rUrl = ar, b["#opener"] = a, b["#urlindex"] = e, a = _.Mi.Ba(_.Aa.location.href), c = _.E("iframes/relay_url_" + (0, window.encodeURIComponent)(a)) || "/_/gapi/sibling/1/frame.html", a = a + c);
            c = a
        }
        _.Pc._open({
            url: c,
            openParams: this.tk,
            params: b,
            proxyId: this.wr,
            openedByProxyChain: this.jr
        })
    })
};
Tq(nr);
nr.prototype.B = function() {
    return this.A
};
nr.prototype.SH = function(a) {
    this.A = a.childId;
    var b = Oq(a, "..", this.A);
    _.Lq(b, this);
    this.close = b._toclose;
    _.K.Uf[this.A] = this;
    this.vf && this.methods._onopen && (a._methods = _.Mq(b, this.vf.ca(), this.A, this, !1), this.methods._onopen(a))
};
nr.prototype.fn = function(a) {
    var b = String(this.A), c = Oq(a, "..", b);
    _.Lq(a, this);
    _.Lq(c, this);
    this.le("ready", a);
    this.vf && this.methods._ready && (a._methods = _.Mq(c, this.vf.ca(), b, this, !1), this.methods._ready(a))
};
nr.prototype.RH = function(a) {
    if (this.vf && this.methods._onclose)
        this.methods._onclose(a);
    else 
        return a = this.le("close", a), delete _.K.Uf[this.A], a
};
var or = function(a, b, c, d, e, f, g) {
    Uq.call(this, a, b, c, d, gr, f, g);
    this.id = b.id || Eq();
    this.B = e;
    d._close = this.close;
    this.onClosed = this.A;
    this.nA = 0;
    jr(this, function() {
        this.le("beforeparentopen");
        var b = _.Lq(this.methods);
        this.params._methods = _.Mq(b, "..", this.wr, this, !0);
        b = {};
        b.queryParams = this.params;
        a = _.Sl(_.Ba, this.config.url, this.id, b);
        var c = e.Ux(a);
        this.canAutoClose = function(a) {
            a(e.tu(c))
        };
        e.Dz(c, this);
        this.nA = c
    })
};
Tq(or);
or.prototype.close = function(a) {
    a = this.le("close", a);
    this.B.Cu(this.nA);
    return a
};
or.prototype.A = function() {
    this.le("close")
};
(function() {
    _.K.Uf = {};
    _.K.Hk = {};
    _.K.Xo = {};
    _.K.Uw = 0;
    _.K.tn = {};
    _.K.gi = {};
    _.K.hn = null;
    _.K.gn = [];
    _.K.XH = function(a) {
        var b=!1;
        try {
            if (null != a)
                var e = window.parent.frames[a.id], b = e.iframer.id == a.id && e.iframes.openedId_(_.Pc.id)
        } catch (f) {}
        try {
            _.K.hn = {
                origin: this.origin,
                referer: this.referer,
                claimedOpenerId: a && a.id,
                claimedOpenerProxyChain: a && a.proxyChain || [],
                sameOrigin: b
            };
            for (a = 0; a < _.K.gn.length; ++a)
                _.K.gn[a](_.K.hn);
            _.K.gn = []
        } catch (g) {}
    };
    _.K.XE = function(a) {
        var b = a && a.vf, e = null;
        b && (e = {}, e.id = b.ca(), e.proxyChain =
        a.jr);
        return e
    };
    dr();
    if (window.parent != window) {
        var a = _.H.Fb();
        a.gcv && Hq(a.gcv);
        var b = a.jsh;
        b && Jq(b);
        _.Lq(Oq(a, "..", ""), _.Pc);
        _.Lq(a, _.Pc);
        er()
    }
    _.K.ua = Vq;
    _.K.tb = Wq;
    _.K.uJ = Zq;
    _.K.resize = Xq;
    _.K.BE = function(a) {
        return _.K.Xo[a]
    };
    _.K.Ur = function(a, b) {
        _.K.Xo[a] = b
    };
    _.K.My = Xq;
    _.K.FJ = Zq;
    _.K.Dm = {};
    _.K.Dm.get = Vq;
    _.K.Dm.set = Wq;
    _.K.gD = function(a, b) {
        Nq(a);
        _.K.gi[a] = b || window[a]
    };
    _.K.pS = function(a) {
        delete _.K.gi[a]
    };
    _.K.open = function(a, b, e, f, g, k) {
        3 == arguments.length ? f = {} : 4 == arguments.length && "function" === typeof f &&
        (g = f, f = {});
        var l = "bubble" === b.style && Yq ? Yq.ja(): null;
        return l ? new or(a, b, e, f, l, g, k) : Qq(b) ? new nr(a, b, e, f, g, k) : new mr(a, b, e, f, g, k)
    };
    _.K.close = function(a, b) {
        _.Pc && _.Pc._close && _.Pc._close(a, b)
    };
    _.K.ready = function(a, b, e) {
        2 == arguments.length && "function" === typeof b && (e = b, b = {});
        var f = a || {};
        "height"in f || (f.height = _.il.Jb());
        f._methods = _.Mq(b || {}, "..", "", _.Pc, !0);
        _.Pc && _.Pc._ready && _.Pc._ready(f, e)
    };
    _.K.Fv = function(a) {
        _.K.hn ? a(_.K.hn) : _.K.gn.push(a)
    };
    _.K.UH = function(a) {
        return !!_.K.Uf[a]
    };
    _.K.HE = function() {
        return ["https://ssl.gstatic.com/gb/js/",
        _.E("googleapis.config/gcv")].join("")
    };
    _.K.wy = function(a) {
        var b = {
            mouseover: 1,
            mouseout: 1
        };
        if (_.Pc._event)
            for (var e = 0; e < a.length; e++) {
                var f = a[e];
                f in b && _.H.ll(window.document, f, function(a) {
                    _.Pc._event({
                        event: a.type,
                        timestamp: (new Date).getTime()
                    })
                }, !0)
            }
    };
    _.K.xJ = Jq;
    _.K.Rr = Kq;
    _.K.ux = Gq;
    _.K.Zw = _.Pc
})();
_.C("iframes.allow", _.K.gD);
_.C("iframes.callSiblingOpener", _.K.rD);
_.C("iframes.registerForOpenedSibling", _.K.pI);
_.C("iframes.close", _.K.close);
_.C("iframes.getGoogleConnectJsUri", _.K.HE);
_.C("iframes.getHandler", _.K.ua);
_.C("iframes.getDeferredHandler", _.K.BE);
_.C("iframes.getParentInfo", _.K.Fv);
_.C("iframes.iframer", _.K.Zw);
_.C("iframes.open", _.K.open);
_.C("iframes.openedId_", _.K.UH);
_.C("iframes.propagate", _.K.wy);
_.C("iframes.ready", _.K.ready);
_.C("iframes.resize", _.K.resize);
_.C("iframes.setGoogleConnectJsVersion", _.K.uJ);
_.C("iframes.setBootstrapHint", _.K.Rr);
_.C("iframes.setJsHint", _.K.xJ);
_.C("iframes.setHandler", _.K.tb);
_.C("iframes.setDeferredHandler", _.K.Ur);
_.C("IframeBase", Uq);
_.C("IframeBase.prototype.addCallback", Uq.prototype.Ob);
_.C("IframeBase.prototype.getMethods", Uq.prototype.dm);
_.C("IframeBase.prototype.getOpenerIframe", Uq.prototype.Kb);
_.C("IframeBase.prototype.getOpenParams", Uq.prototype.kb);
_.C("IframeBase.prototype.getParams", Uq.prototype.He);
_.C("IframeBase.prototype.removeCallback", Uq.prototype.hh);
_.C("Iframe", mr);
_.C("Iframe.prototype.close", mr.prototype.close);
_.C("Iframe.prototype.exposeMethod", mr.prototype.Nc);
_.C("Iframe.prototype.getId", mr.prototype.ca);
_.C("Iframe.prototype.getIframeEl", mr.prototype.Fa);
_.C("Iframe.prototype.getSiteEl", mr.prototype.Pa);
_.C("Iframe.prototype.openInto", mr.prototype.Gd);
_.C("Iframe.prototype.remove", mr.prototype.remove);
_.C("Iframe.prototype.setSiteEl", mr.prototype.Ud);
_.C("Iframe.prototype.addCallback", mr.prototype.Ob);
_.C("Iframe.prototype.getMethods", mr.prototype.dm);
_.C("Iframe.prototype.getOpenerIframe", mr.prototype.Kb);
_.C("Iframe.prototype.getOpenParams", mr.prototype.kb);
_.C("Iframe.prototype.getParams", mr.prototype.He);
_.C("Iframe.prototype.removeCallback", mr.prototype.hh);
_.C("IframeProxy", nr);
_.C("IframeProxy.prototype.getTargetIframeId", nr.prototype.B);
_.C("IframeProxy.prototype.addCallback", nr.prototype.Ob);
_.C("IframeProxy.prototype.getMethods", nr.prototype.dm);
_.C("IframeProxy.prototype.getOpenerIframe", nr.prototype.Kb);
_.C("IframeProxy.prototype.getOpenParams", nr.prototype.kb);
_.C("IframeProxy.prototype.getParams", nr.prototype.He);
_.C("IframeProxy.prototype.removeCallback", nr.prototype.hh);
_.C("IframeWindow", or);
_.C("IframeWindow.prototype.close", or.prototype.close);
_.C("IframeWindow.prototype.onClosed", or.prototype.A);
_.C("iframes.util.getTopMostAccessibleWindow", _.K.ta.Rv);
_.C("iframes.handlers.get", _.K.Dm.get);
_.C("iframes.handlers.set", _.K.Dm.set);
_.C("iframes.resizeMe", _.K.My);
_.C("iframes.setVersionOverride", _.K.FJ);
Uq.prototype.send = function(a, b, c) {
    _.K.cz(this, a, b, c)
};
_.Pc.send = function(a, b, c) {
    _.K.cz(_.Pc, a, b, c)
};
Uq.prototype.register = function(a, b) {
    var c = this;
    c.Ob(a, function(a) {
        b.call(c, a)
    })
};
_.K.cz = function(a, b, c, d) {
    var e = [];
    void 0 !== c && e.push(c);
    d && e.push(function(a) {
        d.call(this, [a])
    });
    a[b] && a[b].apply(a, e)
};
_.K.Bf = function() {
    return !0
};
_.C("iframes.CROSS_ORIGIN_IFRAMES_FILTER", _.K.Bf);
_.C("IframeBase.prototype.send", Uq.prototype.send);
_.C("IframeBase.prototype.register", Uq.prototype.register);
_.C("Iframe.prototype.send", mr.prototype.send);
_.C("Iframe.prototype.register", mr.prototype.register);
_.C("IframeProxy.prototype.send", nr.prototype.send);
_.C("IframeProxy.prototype.register", nr.prototype.register);
_.C("IframeWindow.prototype.send", or.prototype.send);
_.C("IframeWindow.prototype.register", or.prototype.register);
_.C("iframes.iframer.send", _.K.Zw.send);

var Cr = _.K.tb, Dr = {
    open: function(a) {
        var b = _.An(a.kb());
        return a.Gd(b, {
            style: _.Bn(b)
        })
    },
    attach: function(a, b) {
        var c = _.An(a.kb()), d = b.id, e = b.getAttribute("data-postorigin") || b.src, f = /#(?:.*&)?rpctoken=(\d+)/.exec(e), f = f && f[1];
        a.id = d;
        a.B = f;
        a.C = c;
        a.A = b;
        _.K.Uf[d] = a;
        f = _.Lq(a.methods);
        f._ready = a.fn;
        f._close = a.close;
        f._open = a.kt;
        f._resizeMe = a.lt;
        f._renderstart = a.Px;
        _.Mq(f, d, "", a, !0);
        _.J.xn(a.id, a.B);
        _.J.Ze(a.id, e);
        var c = _.K.Lj({
            style: _.Bn(c)
        }), g;
        for (g in c)
            Object.prototype.hasOwnProperty.call(c, g) && ("style" == g ? a.A.style.cssText = c[g] : a.A.setAttribute(g, c[g]))
    }
};
Dr.onready = _.Cn;
Dr.onRenderStart = _.Cn;
Dr.close = _.Dn;
Cr("inline", Dr);

_.Li = (window.gapi || {}).load;
_.Gn = function(a) {
    var b = window;
    a = (a || b.location.href).match(/.*(\?|#|&)usegapi=([^&#]+)/) || [];
    return "1" === (0, window.decodeURIComponent)(a[a.length - 1] || "")
};
var Kn, Ln, Mn, Nn, On, Rn, Pn, Qn, Sn;
_.Hn = {
    apppackagename: 1,
    callback: 1,
    clientid: 1,
    cookiepolicy: 1,
    openidrealm: - 1,
    includegrantedscopes: - 1,
    requestvisibleactions: 1,
    scope: 1
};
_.In=!1;
_.Jn = function() {
    if (!_.In) {
        for (var a = window.document.getElementsByTagName("meta"), b = 0; b < a.length; ++b) {
            var c = a[b].name.toLowerCase();
            if (_.Wc(c, "google-signin-")) {
                var c = c.substring(14), d = a[b].content;
                _.Hn[c] && d && (_.Fn[c] = d)
            }
        }
        if (window.self !== window.top) {
            var a = window.document.location.toString(), e;
            for (e in _.Hn)
                0 < _.Hn[e] && (b = _.Ua(a, e, "")) && (_.Fn[e] = b)
            }
        _.In=!0
    }
    e = _.Ka();
    _.tl(_.Fn, e);
    return e
};
Kn = function(a) {
    return function(b, c) {
        return a ? _.Xl()[c] || a[c] || "" : _.Xl()[c] || ""
    }
};
Ln = {
    button: !0,
    div: !0,
    span: !0
};
Mn = function(a) {
    return _.Ha(_.Ya, "watt", _.Ka())[a]
};
Nn = function(a) {
    var b = _.Ha(_.Ya, "sws", []);
    return 0 <= _.Mb.call(b, a)
};
On = function() {};
Rn = function(a) {
    var b = a.Ph, c = function(a) {
        c.J.constructor.call(this, a);
        var b = this.ud.length;
        this.A = [];
        for (var d = 0; d < b; ++d)
            this.ud[d].jS || (this.A[d] = new this.ud[d](a))
    };
    _.u(c, b);
    for (var d = []; a;) {
        if (b = a.Ph) {
            b.ud && _.jd(d, b.ud);
            var b = b.prototype, e;
            for (e in b)
                if (b.hasOwnProperty(e) && _.Uc(b[e]) && b[e] !== a) {
                    var f=!!b[e].ZR, g = Pn(e, b, d, f);
                    (f = Qn(e, b, g, f)) && (c.prototype[e] = f)
                }
        }
        a = a.J && a.J.constructor
    }
    c.prototype.ud = d;
    return c
};
Pn = function(a, b, c, d) {
    for (var e = [], f = 0; f < c.length && (c[f].prototype[a] === b[a] || (e.push(f), !d)); ++f);
    return e
};
Qn = function(a, b, c, d) {
    var e;
    c.length ? d ? e = function(b) {
        var d = this.A[c[0]];
        return d ? d[a].apply(this.A[c[0]], arguments) : this.ud[c[0]].prototype[a].apply(this, arguments)
    } : b[a].tD ? e = function(b) {
        var d;
        t: {
            d = Array.prototype.slice.call(arguments, 0);
            for (var e = 0; e < c.length; ++e) {
                var l = this.A[c[e]];
                if (l = l ? l[a].apply(l, d) : this.ud[c[e]].prototype[a].apply(this, d)) {
                    d = l;
                    break t
                }
            }
            d=!1
        }
        return d
    } : b[a].sD ? e = function(b) {
        var d;
        t: {
            d = Array.prototype.slice.call(arguments, 0);
            for (var e = 0; e < c.length; ++e) {
                var l = this.A[c[e]], l = l ?
                l[a].apply(l, d): this.ud[c[e]].prototype[a].apply(this, d);
                if (null != l) {
                    d = l;
                    break t
                }
            }
            d = void 0
        }
        return d
    } : b[a].Bx ? e = function(b) {
        for (var d = Array.prototype.slice.call(arguments, 0), e = 0; e < c.length; ++e) {
            var l = this.A[c[e]];
            l ? l[a].apply(l, d) : this.ud[c[e]].prototype[a].apply(this, d)
        }
    } : e = function(b) {
        for (var d = Array.prototype.slice.call(arguments, 0), e = [], l = 0; l < c.length; ++l) {
            var m = this.A[c[l]];
            e.push(m ? m[a].apply(m, d) : this.ud[c[l]].prototype[a].apply(this, d))
        }
        return e
    } : d || b[a].tD || b[a].sD || b[a].Bx ? e = null : e = Sn;
    return e
};
Sn = function() {
    return []
};
On.prototype.B = function(a) {
    if (this.A)
        for (var b = 0; b < this.A.length; ++b)
            if (this.A[b]instanceof a)
                return this.A[b];
    return null
};
var Tn = function(a) {
    return this.Yo.B(a)
};
var Un;
Un = function(a) {
    var b;
    a.match(/^https?%3A/i) && (b = (0, window.decodeURIComponent)(a));
    return _.Fl(window.document, b ? b : a)
};
_.Vn = function(a) {
    a = a || "canonical";
    for (var b = window.document.getElementsByTagName("link"), c = 0, d = b.length; c < d; c++) {
        var e = b[c], f = e.getAttribute("rel");
        if (f && f.toLowerCase() == a && (e = e.getAttribute("href")) && (e = Un(e)) && null != e.match(/^https?:\/\/[\w\-\_\.]+/i))
            return e
    }
    return window.location.href
};
_.Wn = function() {
    return window.location.origin || window.location.protocol + "//" + window.location.host
};
_.Xn = function(a, b, c, d) {
    return (a = "string" == typeof a ? a : void 0) ? Un(a) : _.Vn(d)
};
_.Yn = function(a, b, c) {
    null == a && c && (a = c.db, null == a && (a = c.gwidget && c.gwidget.db));
    return a || void 0
};
_.Zn = function(a, b, c) {
    null == a && c && (a = c.ecp, null == a && (a = c.gwidget && c.gwidget.ecp));
    return a || void 0
};
_.$n = function(a, b) {
    return _.Xn(a, 0, 0, b.action ? void 0 : "publisher")
};
var ao, bo, co, eo, fo, go, io, ho;
ao = {
    se: "0"
};
bo = {
    post: !0
};
co = {
    style: "position:absolute;top:-10000px;width:450px;margin:0px;border-style:none"
};
eo = "onPlusOne _ready _close _open _resizeMe _renderstart oncircled drefresh erefresh".split(" ");
fo = _.Ha(_.Ya, "WI", _.Ka());
go = ["style", "data-gapiscan"];
io = function(a) {
    for (var b = _.Ka(), c = 0 != a.nodeName.toLowerCase().indexOf("g:"), d = 0, e = a.attributes.length; d < e; d++) {
        var f = a.attributes[d], g = f.name, k = f.value;
        0 <= _.Mb.call(go, g) || c && 0 != g.indexOf("data-") || "null" === k || "specified"in f&&!f.specified || (c && (g = g.substr(5)), b[g.toLowerCase()] = k)
    }
    a = a.style;
    (c = ho(a && a.height)) && (b.height = String(c));
    (a = ho(a && a.width)) && (b.width = String(a));
    return b
};
_.ko = function(a, b, c, d, e, f) {
    var g;
    c.rd ? g = b : (g = window.document.createElement("div"), b.setAttribute("data-gapistub", !0), g.style.cssText = "position:absolute;width:450px;left:-10000px;", b.parentNode.insertBefore(g, b));
    f.siteElement = g;
    g.id || (g.id = _.jo(a));
    b = _.Ka();
    b[">type"] = a;
    _.tl(c, b);
    a = _.am(d, g, e);
    f.iframeNode = a;
    f.id = a.getAttribute("id")
};
_.jo = function(a) {
    _.Ha(fo, a, 0);
    return "___" + a + "_" + fo[a]++
};
ho = function(a) {
    var b = void 0;
    "number" === typeof a ? b = a : "string" === typeof a && (b = (0, window.parseInt)(a, 10));
    return b
};
var lo, mo, no, oo, po = /(?:^|\s)g-((\S)*)(?:$|\s)/, qo = {
    plusone: !0,
    autocomplete: !0,
    profile: !0,
    signin: !0
};
lo = _.Ha(_.Ya, "SW", _.Ka());
mo = _.Ha(_.Ya, "SA", _.Ka());
no = _.Ha(_.Ya, "SM", _.Ka());
oo = _.Ha(_.Ya, "FW", []);
var ro = function(a, b) {
    var c;
    "string" === typeof a ? c = window.document.getElementById(a) : c = a;
    return c || b
}, uo = function(a, b) {
    var c, d, e, f;
    _.so("ps0");
    var g = ro(a, _.Ba);
    d = _.Ba.documentMode;
    if (g.querySelectorAll && (!d || 8 < d)) {
        d = b ? [b] : _.mb(lo).concat(_.mb(mo)).concat(_.mb(no));
        e = [];
        for (var k = 0; k < d.length; k++) {
            var l = d[k];
            e.push(".g-" + l, "g\\:" + l)
        }
        d = g.querySelectorAll(e.join(","))
    } else 
        d = g.getElementsByTagName("*");
    g = _.Ka();
    for (e = 0; e < d.length; e++) {
        k = d[e];
        var m = k, l = b, n = m.nodeName.toLowerCase(), p = void 0;
        m.getAttribute("data-gapiscan") ?
        l = null : (0 == n.indexOf("g:") ? p = n.substr(2) : (m = (m = String(m.className || m.getAttribute("class"))) && po.exec(m)) && (p = m[1]), l=!p ||!(lo[p] || mo[p] || no[p]) || l && p !== l ? null : p);
        l && (qo[l] || 0 == k.nodeName.toLowerCase().indexOf("g:") || 0 != _.mb(io(k)).length) && (k.setAttribute("data-gapiscan", !0), _.Ha(g, l, []).push(k))
    }
    for (c in g)
        oo.push(c);
    _.so("ps1");
    if (c = oo.join(":"))
        try {
            _.Sa.load(c, void 0)
    } catch (v) {
        _.Qb(v);
        return 
    }
    k = [];
    for (f in g)
        for (e = g[f], d = 0, c = e.length; d < c; d++)
            l = e[d], to(f, l, io(l), k, c)
};
var vo = function(a, b) {
    var c = Mn(a);
    b && c ? (c(b), (c = b.iframeNode) && c.setAttribute("data-gapiattached", !0)) : _.Sa.load(a, function() {
        var c = Mn(a), e = b && b.iframeNode;
        e && c ? (c(b), e.setAttribute("data-gapiattached", !0)) : (0, _.Sa[a].go)(e && e.parentNode)
    })
}, to = function(a, b, c, d, e, f) {
    switch (wo(b, a, f)) {
    case 0:
        a = no[a] ? a + "_annotation" : a;
        d = {};
        d.iframeNode = b;
        d.userParams = c;
        vo(a, d);
        break;
    case 1:
        if (b.parentNode) {
            for (var g in c) {
                if (f = _.Ma(c, g))
                    f = c[g], f=!!f && "object" === typeof f && (!f.toString || f.toString === Object.prototype.toString ||
                    f.toString === Array.prototype.toString);
                if (f)
                    try {
                        c[g] = (0, _.$b)(c[g])
                    } catch (k) {
                    delete c[g]
                }
            }
            g=!0;
            c.dontclear && (g=!1);
            delete c.dontclear;
            var l, m, n;
            f = {};
            var p = n = a;
            "plus" == a && c.action && (n = a + "_" + c.action, p = a + "/" + c.action);
            (n = _.E("iframes/" + n + "/url")) || (n = ":im_socialhost:/:session_prefix::im_prefix:_/widget/render/" + p + "?usegapi=1");
            for (m in ao)
                f[m] = m + "/" + (c[m] || ao[m]) + "/";
            m = _.Fl(_.Ba, n.replace(_.Wl, Kn(f)));
            p = "iframes/" + a + "/params/";
            f = {};
            _.tl(c, f);
            (n = _.E("lang") || _.E("gwidget/lang")) && (f.hl = n);
            bo[a] || (f.origin =
            _.Wn());
            f.exp = _.E(p + "exp");
            if (p = _.E(p + "location"))
                for (n = 0; n < p.length; n++) {
                    var v = p[n];
                    f[v] = _.Aa.location[v]
                }
            switch (a) {
            case "plus":
            case "follow":
                f.url = _.$n(f.href, c);
                delete f.href;
                break;
            case "plusone":
                p = (p = c.href) ? Un(p) : _.Vn();
                f.url = p;
                f.db = _.Yn(c.db, 0, _.E());
                f.ecp = _.Zn(c.ecp, 0, _.E());
                delete f.href;
                break;
            case "signin":
                f.url = _.Vn()
            }
            _.Ya.ILI && (f.iloader = "1");
            delete f["data-onload"];
            delete f.rd;
            for (var x in ao)
                f[x] && delete f[x];
            f.gsrc = _.E("iframes/:source:");
            x = _.E("inline/css");
            "undefined" !== typeof x &&
            0 < e && x >= e && (f.ic = "1");
            x = /^#|^fr-/;
            e = {};
            for (var t in f)
                _.Ma(f, t) && x.test(t) && (e[t.replace(x, "")] = f[t], delete f[t]);
            t = "q" == _.E("iframes/" + a + "/params/si") ? f : e;
            x = _.Jn();
            for (l in x)
                !_.Ma(x, l) || _.Ma(f, l) || _.Ma(e, l) || (t[l] = x[l]);
            l = [].concat(eo);
            t = _.E("iframes/" + a + "/methods");
            _.gb(t) && (l = l.concat(t));
            for (var w in c)
                _.Ma(c, w) && /^on/.test(w) && ("plus" != a || "onconnect" != w) && (l.push(w), delete f[w]);
            delete f.callback;
            e._methods = l.join(",");
            l = _.Gl(m, f, e);
            t = {
                allowPost: 1,
                attributes: co
            };
            t.dontclear=!g;
            w = {};
            w.userParams =
            c;
            w.url = l;
            w.type = a;
            _.ko(a, b, c, l, t, w);
            b = w.id;
            c = _.Ka();
            c.id = b;
            c.userParams = w.userParams;
            c.url = w.url;
            c.type = w.type;
            c.state = 1;
            _.xn[b] = c;
            b = w
        } else 
            b = null;
        b && ((c = b.id) && d.push(c), vo(a, b))
    }
}, wo = function(a, b, c) {
    if (a && 1 === a.nodeType && b) {
        if (c)
            return 1;
        if (no[b]) {
            if (Ln[a.nodeName.toLowerCase()])
                return (a = a.innerHTML) && a.replace(/^[\s\xa0]+|[\s\xa0]+$/g, "") ? 0 : 1
        } else {
            if (mo[b])
                return 0;
            if (lo[b])
                return 1
        }
    }
    return null
};
_.Ha(_.Sa, "platform", {}).go = function(a, b) {
    uo(a, b)
};
var xo = _.Ha(_.Ya, "perf", _.Ka()), yo = _.Ha(xo, "g", _.Ka()), zo = _.Ha(xo, "i", _.Ka()), Ao, Bo, Co, Do, Go, Ho;
_.Ha(xo, "r", []);
Ao = _.Ka();
Bo = _.Ka();
Co = function(a, b, c, d) {
    Ao[c] = Ao[c]||!!d;
    _.Ha(Bo, c, []);
    Bo[c].push([a, b])
};
Do = function(a, b, c) {
    var d = xo.r;
    "function" === typeof d ? d(a, b, c) : d.push([a, b, c])
};
_.so = function(a, b) {
    yo[a] = b || (new Date).getTime();
    Do(a)
};
_.Fo = function(a, b, c, d) {
    if ("_p" == b)
        throw Error("I");
    _.Eo(a, b, c, d)
};
_.Eo = function(a, b, c, d) {
    Go(b, c)[a] = d || (new Date).getTime();
    Do(a, b, c)
};
Go = function(a, b) {
    var c = _.Ha(zo, a, _.Ka());
    return _.Ha(c, b, _.Ka())
};
Ho = function(a, b, c) {
    var d = null;
    b && c && (d = Go(b, c)[a]);
    return d || yo[a]
};
_.Io = function(a, b, c, d, e, f) {
    _.Fo("wrt0", a, b, c);
    _.Fo("wrt1", a, b, d);
    _.Fo("wrt2", a, b, e);
    _.Fo("wrt3", a, b, f)
};
(function() {
    function a(a) {
        this.t = {};
        this.tick = function(a, b, c) {
            this.t[a] = [void 0 != c ? c: (new Date).getTime(), b];
            if (void 0 == c)
                try {
                    window.console.timeStamp("CSI/" + a)
            } catch (d) {}
        };
        this.tick("start", null, a)
    }
    var b;
    window.performance && (b = window.performance.timing);
    var c = b ? new a(b.responseStart): new a;
    window.__gapi_jstiming__ = {
        Timer: a,
        load: c
    };
    if (b) {
        var d = b.navigationStart, e = b.responseStart;
        0 < d && e >= d && (window.__gapi_jstiming__.srt = e - d)
    }
    if (b) {
        var f = window.__gapi_jstiming__.load;
        0 < d && e >= d && (f.tick("_wtsrt", void 0,
        d), f.tick("wtsrt_", "_wtsrt", e), f.tick("tbsd_", "wtsrt_"))
    }
    try {
        b = null, window.chrome && window.chrome.csi && (b = Math.floor(window.chrome.csi().pageT), f && 0 < d && (f.tick("_tbnd", void 0, window.chrome.csi().startE), f.tick("tbnd_", "_tbnd", d))), null == b && window.gtbExternal && (b = window.gtbExternal.pageT()), null == b && window.external && (b = window.external.pageT, f && 0 < d && (f.tick("_tbnd", void 0, window.external.startE), f.tick("tbnd_", "_tbnd", d))), b && (window.__gapi_jstiming__.pt = b)
    } catch (g) {}
})();
if (window.__gapi_jstiming__) {
    window.__gapi_jstiming__.pu = {};
    window.__gapi_jstiming__.yI = 1;
    var Jo = function(a, b, c) {
        var d = a.t[b], e = a.t.start;
        if (d && (e || c))
            return d = a.t[b][0], void 0 != c ? e = c : e = e[0], d - e
    };
    window.__gapi_jstiming__.getTick = Jo;
    window.__gapi_jstiming__.getLabels = function(a) {
        var b = [], c;
        for (c in a.t)
            b.push(c);
        return b
    };
    var Ko = function(a, b, c) {
        var d = "";
        window.__gapi_jstiming__.srt && (d += "&srt=" + window.__gapi_jstiming__.srt);
        window.__gapi_jstiming__.pt && (d += "&tbsrt=" + window.__gapi_jstiming__.pt);
        try {
            window.external &&
            window.external.tran ? d += "&tran=" + window.external.tran : window.gtbExternal && window.gtbExternal.tran ? d += "&tran=" + window.gtbExternal.tran() : window.chrome && window.chrome.csi && (d += "&tran=" + window.chrome.csi().tran)
        } catch (e) {}
        var f = window.chrome;
        if (f && (f = f.loadTimes)) {
            f().wasFetchedViaSpdy && (d += "&p=s");
            if (f().wasNpnNegotiated) {
                var d = d + "&npn=1", g = f().npnNegotiatedProtocol;
                g && (d += "&npnv=" + (window.encodeURIComponent || window.escape)(g))
            }
            f().wasAlternateProtocolAvailable && (d += "&apa=1")
        }
        var k = a.t, l = k.start, f =
        [], g = [], m;
        for (m in k)
            if ("start" != m && 0 != m.indexOf("_")) {
                var n = k[m][1];
                n ? k[n] && g.push(m + "." + Jo(a, m, k[n][0])) : l && f.push(m + "." + Jo(a, m))
            }
        if (b)
            for (var p in b)
                d += "&" + p + "=" + b[p];
        (b = c) || (b = "https:" == window.document.location.protocol ? "https://csi.gstatic.com/csi" : "http://csi.gstatic.com/csi");
        return [b, "?v=3", "&s=" + (window.__gapi_jstiming__.sn || "") + "&action=", a.name, g.length ? "&it=" + g.join(","): "", d, "&rt=", f.join(",")].join("")
    }, Lo = function(a, b, c) {
        a = Ko(a, b, c);
        if (!a)
            return "";
        b = new window.Image;
        var d = window.__gapi_jstiming__.yI++;
        window.__gapi_jstiming__.pu[d] = b;
        b.onload = b.onerror = function() {
            window.__gapi_jstiming__ && delete window.__gapi_jstiming__.pu[d]
        };
        b.src = a;
        b = null;
        return a
    };
    window.__gapi_jstiming__.report = function(a, b, c) {
        if ("prerender" == window.document.webkitVisibilityState) {
            var d=!1, e = function() {
                if (!d) {
                    b ? b.prerender = "1" : b = {
                        prerender: "1"
                    };
                    var f;
                    "prerender" == window.document.webkitVisibilityState ? f=!1 : (Lo(a, b, c), f=!0);
                    f && (d=!0, window.document.removeEventListener("webkitvisibilitychange", e, !1))
                }
            };
            window.document.addEventListener("webkitvisibilitychange", e, !1);
            return ""
        }
        return Lo(a, b, c)
    }
};
var Mo = [73, 74, 77, 78], No = {
    g: "gapi_global",
    m: "gapi_module",
    w: "gwidget"
}, Oo = function(a, b) {
    this.type = a ? "_p" == a ? "m" : "w" : "g";
    this.name = a;
    this.A = b
};
Oo.prototype.key = function() {
    switch (this.type) {
    case "g":
        return this.type;
    case "m":
        return this.type + "." + this.A;
    case "w":
        return this.type + "." + this.name + this.A
    }
};
var Po = new Oo, Qo = window.navigator.userAgent.match(/iPhone|iPad|Android|PalmWebOS|Maemo|Bada/), Ro = _.Ha(xo, "_c", _.Ka()), So = Math.random() < (_.E("csi/rate") || 0), Uo = function(a, b, c) {
    for (var d = new Oo(b, c), e = _.Ha(Ro, d.key(), _.Ka()), f = Bo[a] || [], g = 0; g < f.length; ++g) {
        var k = f[g], l = k[0], m = a, n = b, p = c, k = Ho(k[1], n, p), m = Ho(m, n, p);
        e[l] = k && m ? m - k : null
    }
    Ao[a] && So && (To(Po), To(d))
}, Vo = function(a, b) {
    b = b || [];
    for (var c = [], d = 0; d < b.length; d++)
        c.push(a + b[d]);
    return c
}, Wo = function(a, b) {
    var c = [];
    c.push("l" + (_.E("isPlusUser") ? "1" : "0"));
    var d = "m" + (Qo ? "1" : "0");
    c.push(d);
    if ("m" == a.type)
        c.push("p" + a.A);
    else if ("w" == a.type) {
        var e = "n" + a.A;
        c.push(e);
        "0" == a.A && c.push(d + e)
    }
    c.push("u" + (_.E("isLoggedIn") ? "1" : "0"));
    return Vo(b ? "e" + b : "", c)
}, To = function(a) {
    var b = _.Aa.__gapi_jstiming__;
    b.sn = No[a.type];
    var c = new b.Timer(0), d;
    t: {
        switch (a.type) {
        case "g":
            d = "global";
            break t;
        case "m":
            d = a.A;
            break t;
        case "w":
            d = a.name;
            break t
        }
        d = void 0
    }
    c.name = d;
    d=!1;
    var e = a.key(), f = Ro[e];
    c.tick("_start", null, 0);
    for (var g in f)
        c.tick(g, "_start", f[g]), d=!0;
    Ro[e] = _.Ka();
    if (d) {
        g =
        [];
        d = _.E("lexps");
        g = g.concat(Vo("e", d));
        g = g.concat(Wo(a));
        for (e = 0; e < Mo.length; e++)
            f = Mo[e], 0 <= _.Mb.call(d, f) && (g = g.concat(Wo(a, f)));
        g = Vo("abc_", g);
        b.report(c, {
            e: g.join(",")
        })
    }
};
Co("blt", "bs0", "bs1");
Co("psi", "ps0", "ps1");
Co("rpcqi", "rqe", "rqd");
Co("bsprt", "bsrt0", "bsrt1");
Co("bsrqt", "bsrt1", "bsrt2");
Co("bsrst", "bsrt2", "bsrt3");
Co("mli", "ml0", "ml1");
Co("mei", "me0", "me1", !0);
Co("wcdi", "wrs", "wcdi");
Co("wci", "wrs", "wdc");
Co("wdi", "wrs", "wrdi");
Co("wdt", "bs0", "wrdt");
Co("wri", "wrs", "wrri", !0);
Co("wrt", "bs0", "wrrt");
Co("wji", "wje0", "wje1", !0);
Co("wjli", "wjl0", "wjl1");
Co("whi", "wh0", "wh1", !0);
Co("wai", "waaf0", "waaf1", !0);
Co("wadi", "wrs", "waaf1", !0);
Co("wadt", "bs0", "waaf1", !0);
Co("wprt", "wrt0", "wrt1");
Co("wrqt", "wrt1", "wrt2");
Co("wrst", "wrt2", "wrt3", !0);
Co("fbprt", "fsrt0", "fsrt1");
Co("fbrqt", "fsrt1", "fsrt2");
Co("fbrst", "fsrt2", "fsrt3", !0);
Co("fdns", "fdns0", "fdns1");
Co("fcon", "fcon0", "fcon1");
Co("freq", "freq0", "freq1");
Co("frsp", "frsp0", "frsp1");
Co("fttfb", "fttfb0", "fttfb1");
Co("ftot", "ftot0", "ftot1", !0);
var Xo = xo.r;
if ("function" !== typeof Xo) {
    for (var Yo; Yo = Xo.shift();)
        Uo.apply(null, Yo);
    xo.r = Uo
};
var Zo = ["div"], $o = "onload", ap=!0, bp=!0, cp = function(a) {
    return a
}, dp = null, ep = function(a) {
    var b = _.E(a);
    return "undefined" !== typeof b ? b : _.E("gwidget/" + a)
}, dp = _.E(), ip, jp, kp, lp, mp, np, op, rp, pp, sp, xp, zp, Ap, Bp, Cp, tp, vp, Ep, up, Fp, Gp, Hp, Ip;
_.E("gwidget");
var fp = ep("parsetags"), $o = "explicit" === fp || "onload" === fp ? fp: $o, gp = ep("google_analytics");
"undefined" !== typeof gp && (ap=!!gp);
var hp = ep("data_layer");
"undefined" !== typeof hp && (bp=!!hp);
ip = function() {
    var a = this && this.ca();
    a && (_.Ya.drw = a)
};
jp = function() {
    _.Ya.drw = null
};
kp = function(a) {
    return function(b) {
        var c = a;
        "number" === typeof b ? c = b : "string" === typeof b && (c = b.indexOf("px"), - 1 != c && (b = b.substring(0, c)), c = (0, window.parseInt)(b, 10));
        return c
    }
};
lp = function(a) {
    "string" === typeof a && (a = window[a]);
    return "function" === typeof a ? a : null
};
mp = function() {
    return ep("lang") || "en-US"
};
np = function(a) {
    if (!_.K.ua("attach")) {
        var b = {}, c = _.K.ua("inline"), d;
        for (d in c)
            c.hasOwnProperty(d) && (b[d] = c[d]);
        b.open = function(a) {
            var b = a.kb().renderData.id, b = window.document.getElementById(b);
            if (!b)
                throw Error("J");
            return c.attach(a, b)
        };
        _.K.tb("attach", b)
    }
    a.style = "attach"
};
op = function() {
    var a = {};
    a.width = [kp(450)];
    a.height = [kp(24)];
    a.onready = [lp];
    a.lang = [mp, "hl"];
    a.iloader = [function() {
        return _.Ya.ILI
    }, "iloader"];
    return a
}();
rp = function(a) {
    var b = {};
    b.Mc = a[0];
    b.xh =- 1;
    b.aU = "___" + b.Mc + "_";
    b.sK = "g:" + b.Mc;
    b.VS = "g-" + b.Mc;
    b.Iy = [];
    b.config = {};
    b.vl = [];
    b.iA = {};
    b.Pn = {};
    var c = function(a) {
        for (var c in a)
            if (_.Ma(a, c)) {
                b.config[c] = [lp];
                b.vl.push(c);
                var d = a[c], k = null, l = null, m = null;
                "function" === typeof d ? k = d : d && "object" === typeof d && (k = d.LS, l = d.Qn, m = d.Un);
                m && (b.vl.push(m), b.config[m] = [lp], b.iA[c] = m);
                k && (b.config[c] = [k]);
                l && (b.Pn[c] = l)
            }
    }, d = function(a) {
        for (var c = {}, d = 0; d < a.length; ++d)
            c[a[d].toLowerCase()] = 1;
        c[b.sK] = 1;
        b.sH = c
    };
    a[1] && (b.Xx =
    a[1]);
    (function(a) {
        b.config = a;
        for (var c in op)
            op.hasOwnProperty(c)&&!b.config.hasOwnProperty(c) && (b.config[c] = op[c])
    })(a[2] || {});
    a[3] && c(a[3]);
    a[4] && d(a[4]);
    a[5] && (b.cg = a[5]);
    b.ST=!0 === a[6];
    b.jI = a[7];
    b.kK = a[8];
    b.sH || d(Zo);
    b.gr = function(a) {
        b.xh++;
        _.Fo("wrs", b.Mc, String(b.xh));
        var c = [], d = a.element, k = a.config, l = ":" + b.Mc;
        ":plus" == l && a.$f && a.$f.action && (l += "_" + a.$f.action);
        var m = pp(b, k), n = {};
        _.tl(_.Jn(), n);
        for (var p in a.$f)
            null != a.$f[p] && (n[p] = a.$f[p]);
        p = {
            container: d.id,
            renderData: a.wI,
            style: "inline",
            height: k.height,
            width: k.width
        };
        np(p);
        b.cg && (c[2] = p, c[3] = n, c[4] = m, b.cg("i", c));
        l = _.K.open(l, p, n, m);
        (0, _.qp)(b, l, k, d, a.OD);
        c[5] = l;
        b.cg && b.cg("e", c)
    };
    return b
};
pp = function(a, b) {
    for (var c = {}, d = a.vl.length - 1; 0 <= d; --d) {
        var e = a.vl[d], f = b[a.iA[e] || e] || b[e], g = b[e];
        g && f !== g && (f = function(a, b) {
            return function(c) {
                b.apply(this, arguments);
                a.apply(this, arguments)
            }
        }(f, g));
        f && (c[e] = f)
    }
    for (var k in a.Pn)
        a.Pn.hasOwnProperty(k) && (c[k] = sp(c[k] || function() {}, a.Pn[k]));
    c.drefresh = ip;
    c.erefresh = jp;
    return c
};
sp = function(a, b) {
    return function(c) {
        var d = b(c);
        if (d) {
            var e = c.href || null;
            if (ap) {
                if (window._gat)
                    try {
                        var f = window._gat._getTrackerByName("~0");
                        f && "UA-XXXXX-X" != f._getAccount() ? f._trackSocial("Google", d, e) : window._gaq && window._gaq.push(["_trackSocial", "Google", d, e])
                    } catch (g) {}
                if (window.ga && window.ga.getAll)
                    try {
                        for (var k = window.ga.getAll(), f = 0; f < k.length; f++)
                            k[f].send("social", "Google", d, e)
                        } catch (l) {}
            }
            if (bp && window.dataLayer)
                try {
                    window.dataLayer.push({
                        event: "social",
                        socialNetwork: "Google",
                        socialAction: d,
                        socialTarget: e
                    })
                } catch (m) {}
        }
        a.call(this, c)
    }
};
_.qp = function(a, b, c, d, e) {
    tp(b, c);
    up(b, d);
    vp(a, b, e);
    wp(a.Mc, a.xh.toString(), b);
    (new xp).Yo.Df(a, b, c, d, e)
};
xp = function() {
    if (!this.Yo) {
        for (var a = this.constructor; a&&!a.Ph;)
            a = a.J && a.J.constructor;
        a.Ph.Mu || (a.Ph.Mu = Rn(a));
        this.Yo = new a.Ph.Mu(this);
        this.B || (this.B = Tn)
    }
};
_.yp = function() {};
zp = xp;
_.yp.J || _.u(_.yp, On);
zp.Ph = _.yp;
_.yp.prototype.Df = function(a) {
    a = a ? a : function() {};
    a.Bx=!0;
    return a
}();
Ap = function(a) {
    return _.Im && a instanceof _.Im
};
Bp = function(a) {
    return Ap(a) ? "_renderstart" : "renderstart"
};
Cp = function(a) {
    return Ap(a) ? "_ready" : "ready"
};
_.Dp = function() {
    return !0
};
tp = function(a, b) {
    if (b.onready) {
        var c=!1, d = function() {
            c || (c=!0, b.onready.call(null))
        };
        a.register(Cp(a), d, _.Dp);
        a.register(Bp(a), d, _.Dp)
    }
};
vp = function(a, b, c) {
    var d = a.Mc, e = String(a.xh), f=!1, g = function() {
        f || (f=!0, c && _.Fo("wrdt", d, e), _.Fo("wrdi", d, e))
    };
    b.register(Bp(b), g, _.Dp);
    var k=!1;
    a = function() {
        k || (k=!0, g(), c && _.Fo("wrrt", d, e), _.Fo("wrri", d, e))
    };
    b.register(Cp(b), a, _.Dp);
    Ap(b) ? b.register("widget-interactive-" + b.id, a, _.Dp) : _.J.register("widget-interactive-" + b.id, a);
    _.J.register("widget-csi-tick-" + b.id, function(a, b, c) {
        "wdc" === a ? _.Fo("wdc", d, e, c) : "wje0" === a ? _.Fo("wje0", d, e, c) : "wje1" === a ? _.Fo("wje1", d, e, c) : "wh0" == a ? _.Eo("wh0", d, e, c) : "wh1" ==
        a ? _.Eo("wh1", d, e, c) : "wcdi" == a && _.Eo("wcdi", d, e, c)
    })
};
Ep = function(a) {
    return "number" == typeof a ? a + "px" : "100%" == a ? a : null
};
up = function(a, b) {
    var c = function(c) {
        c = c || a;
        var e = Ep(c.width);
        e && b.style.width != e && (b.style.width = e);
        (c = Ep(c.height)) && b.style.height != c && (b.style.height = c)
    };
    Ap(a) ? a.wb("onRestyle", c) : (a.register("ready", c, _.Dp), a.register("renderstart", c, _.Dp), a.register("resize", c, _.Dp))
};
Fp = function(a, b) {
    for (var c in op)
        if (op.hasOwnProperty(c)) {
            var d = op[c][1];
            d&&!b.hasOwnProperty(d) && (b[d] = a[d])
        }
    return b
};
Gp = function(a, b) {
    var c = {}, d;
    for (d in a)
        a.hasOwnProperty(d) && (c[a[d][1] || d] = (a[d] && a[d][0] || cp)(b[d.toLowerCase()], b, dp));
    return c
};
Hp = function(a) {
    if (a = a.jI)
        for (var b = 0; b < a.length; b++)(new window.Image)
            .src = a[b]
};
Ip = function(a, b) {
    var c = b.userParams, d = b.siteElement;
    d || (d = (d = b.iframeNode) && d.parentNode);
    if (d && 1 === d.nodeType) {
        var e = Gp(a.config, c);
        a.Iy.push({
            element: d,
            config: e,
            $f: Fp(e, Gp(a.Xx, c)),
            zT: 3,
            OD: !!c["data-onload"],
            wI: b
        })
    }
    c = a.Iy;
    for (d = a.gr; 0 < c.length;)
        d(c.shift())
};
_.Jp = function(a) {
    var b = rp(a);
    Hp(b);
    _.Nb(b.Mc, function(a) {
        Ip(b, a)
    });
    lo[b.Mc]=!0;
    var c = {
        Z: function(a, c) {
            var f = c || {};
            f.type = b.Mc;
            var g = f.type;
            delete f.type;
            var k = ro(a);
            if (k) {
                var l = {}, m;
                for (m in f)
                    _.Ma(f, m) && (l[m.toLowerCase()] = f[m]);
                l.rd = 1;
                (f=!!l.ri) && delete l.ri;
                to(g, k, l, [], 0, f)
            } else 
                _.Qb("string" === "gapi." + g + ".render: missing element " + typeof a ? a : "")
        },
        go: function(a) {
            uo(a, b.Mc)
        },
        A: function() {
            var a = _.Ha(_.Ya, "WI", _.Ka()), b;
            for (b in a)
                delete a[b]
        }
    };
    a = function() {
        "onload" === $o && c.go()
    };
    Nn(b.Mc) || _.dc(a, a);
    _.C("gapi." + b.Mc + ".go", c.go);
    _.C("gapi." + b.Mc + ".render", c.Z);
    return c
};
var Kp = Ip, Lp = function(a, b) {
    a.xh++;
    _.Fo("wrs", a.Mc, String(a.xh));
    var c = b.userParams, d = Gp(a.config, c), e = [], f = b.iframeNode, g = b.siteElement, k = pp(a, d), l = Gp(a.Xx, c);
    _.tl(_.Jn(), l);
    var l = Fp(d, l), c=!!c["data-onload"], m = _.pm, n = _.Ka();
    n.renderData = b;
    n.height = d.height;
    n.width = d.width;
    n.id = b.id;
    n.url = b.url;
    n.iframeEl = f;
    n.where = n.container = g;
    n.apis = ["_open"];
    n.messageHandlers = k;
    n.messageHandlersFilter = _.Am;
    _.En(n);
    f = l;
    a.cg && (e[2] = n, e[3] = f, e[4] = k, a.cg("i", e));
    k = m.Dj(n);
    k.id = b.id;
    k.Yr(k, n);
    (0, _.qp)(a, k, d, g, c);
    e[5] = k;
    a.cg && a.cg("e", e)
}, Ip = function(a, b) {
    var c = b.url;
    a.kK || _.Gn(c) ? _.Gm ? Lp(a, b) : (0, _.Li)("gapi.iframes.impl", function() {
        Lp(a, b)
    }) : _.K.open ? Kp(a, b) : (0, _.Li)("iframes", function() {
        Kp(a, b)
    })
};
var wp;
_.Mp = function() {
    var a = window;
    return !!a.performance&&!!a.performance.getEntries
};
_.Np = function(a) {
    var b = a.indexOf("#");
    - 1 != b && (a = a.substring(0, b));
    a = window.performance.getEntriesByName(a);
    if (1 > a.length)
        return null;
    a = a[0];
    return 0 == a.responseStart ? null : a
};
wp = function(a, b, c) {
    if (_.Mp()) {
        var d = function() {
            var a=!1;
            return function() {
                if (a)
                    return !0;
                a=!0;
                return !1
            }
        }(), e = function() {
            d() || window.setTimeout(function() {
                var d = _.Np(c.Fa().src);
                d && _.Io(a, b, Math.round(d.startTime), Math.round(d.requestStart), Math.round(d.responseStart), Math.round(d.responseEnd))
            }, 1E3)
        };
        c.register(Bp(c), e, _.Dp);
        c.register(Cp(c), e, _.Dp)
    }
};
_.C("gapi.widget.make", _.Jp);
var sv, rv, tv, uv, vv, wv, xv, yv, zv, Av, Bv, Cv, Dv, Ev, Fv, Gv, Hv, Iv, Kv, Lv, Mv, Nv, Ov, Pv, Rv, Sv, Tv, Uv, Vv, Wv, Xv, Yv, Zv, $v, aw, cw, dw, ew, fw, gw, iw, jw, kw, lw, mw, pw, qw, sw;
_.Tu.prototype.subscribe = _.L(26, function(a, b, c) {
    var d = this.B[a];
    d || (d = this.B[a] = []);
    var e = this.$v;
    this.A[e] = a;
    this.A[e + 1] = b;
    this.A[e + 2] = c;
    this.$v = e + 3;
    d.push(e);
    return e
});
_.Uu.prototype.subscribe = _.L(25, function(a, b, c) {
    return this.A.subscribe(a.toString(), b, c)
});
_.ov = function(a, b, c, d) {
    if (b = a.B[b]) {
        var e = a.A;
        (b = _.pe(b, function(a) {
            return e[a + 1] == c && e[a + 2] == d
        })) && a.Af(b)
    }
};
_.pv = function(a, b) {
    a.A.show = b;
    return a
};
_.qv = function(a) {
    a.A.closeClickDetection=!0;
    return a
};
sv = function(a) {
    var b = rv;
    _.cv(b, [{
        mn: "-webkit-transform",
        duration: 1,
        timing: "ease",
        vg: 0
    }
    ]);
    _.cv(b, [{
        mn: "transform",
        duration: 1,
        timing: "ease",
        vg: 0
    }
    ]);
    _.$g(function() {
        b.style.webkitTransform = "translate3d(0px," + a + "px,0px)";
        b.style.transform = "translate3d(0px," + a + "px,0px)"
    }, 0)
};
uv = function(a) {
    var b = _.wj(a);
    if (!b)
        return new _.ej("G_USERSTATE_");
    a = ["G_USERSTATE_", _.jj && b.yb ? "S": "H", b.wd].join("");
    var c = _.rj[a];
    c || (c = {
        tH: 63072E3
    }, _.tl(_.yj(b), c), c = new _.bj(a, c), _.rj[a] = c, b = c.bg(), "undefined" !== typeof b && null !== b && (window.document.cookie = a + "=; expires=Thu, 01 Jan 1970 00:00:01 GMT; path=/", c.write(b)));
    return c
};
vv = function(a) {
    a = a && a.id_token;
    if (!a ||!a.split(".")[1])
        return null;
    a = (a.split(".")[1] + "...").replace(/^((....)+).?.?.?$/, "$1");
    return _.Og(_.Ku(_.Ru(a, !0)))
};
wv = function() {
    rv.parentNode.removeChild(rv)
};
xv = function() {
    var a = tv + 88;
    sv(a);
    tv = a
};
yv = function() {
    var a = tv - 88;
    sv(a);
    tv = a
};
zv = function(a) {
    var b = uv(a).bg();
    a = _.Ka();
    if (b)
        for (var b = b.split(":"), c; c = b.shift();)
            c = c.split("="), a[c[0]] = c[1];
    return a
};
Av = function(a) {
    if (!a)
        return [];
    a = a.split("=");
    return a[1] ? a[1].split("|") : []
};
Bv = function(a) {
    return (a = vv(a)) ? a.sub : null
};
Cv = function(a) {
    var b = a ? xv: yv, c = a ? yv: xv;
    a = a ? "-" : "";
    tv = (0, window.parseInt)(a + 88, 10);
    rv.style.webkitTransform = "translate3d(0px," + a + 88 + "px,0px)";
    rv.style.transform = "translate3d(0px," + a + 88 + "px,0px)";
    rv.style.display = "";
    rv.style.visibility = "visible";
    b();
    _.$g(c, 4E3);
    _.$g(wv, 5E3)
};
Dv = function(a) {
    a = a.split(":");
    return {
        yu: a[0].split("=")[1],
        kJ: Av(a[1]),
        dT: Av(a[2]),
        uS: Av(a[3])
    }
};
Ev = function(a) {
    a = _.wj(a);
    if (!a || a.yb&&!_.jj)
        return null;
    var b = ["G_AUTHUSER_", _.jj && a.yb ? "S": "H", a.wd].join(""), c = _.qj[b];
    c || (c = new _.kj(b, _.yj(a)), _.qj[b] = c);
    return c
};
Fv = function(a) {
    var b = Bv(a);
    b ? (a = zv(a.cookie_policy), b = "0" == a[b] || "X" == a[b]) : b=!1;
    return b
};
Gv = function(a) {
    var b = _.E("oauth-flow/toast/position");
    "top" !== b && (b = "bottom");
    var c = window.document.createElement("div");
    rv = c;
    c.style.cssText = "position:fixed;left:0px;z-index:1000;width:100%;";
    var d;
    _.Q(c, "visibility", "hidden");
    _.Q(c, b, "-40px");
    _.Q(c, "height", "128px");
    d = c;
    if ("desktop" == _.E("deviceType")) {
        d = window.document.createElement("div");
        d.style.cssText = "float:left;position:relative;left:50%;";
        c.appendChild(d);
        var e = window.document.createElement("div");
        e.style.cssText = "float:left;position:relative;left:-50%";
        d.appendChild(e);
        d = e
    }
    e = "top" == b ? "-" : "";
    tv = (0, window.parseInt)(e + 88, 10);
    rv.style.webkitTransform = "translate3d(0px," + e + 88 + "px,0px)";
    rv.style.transform = "translate3d(0px," + e + 88 + "px,0px)";
    e = window;
    try {
        for (; e.parent != e && e.parent.document;)
            e = e.parent
    } catch (f) {}
    e = e.document.body;
    try {
        e.insertBefore(c, e.firstChild)
    } catch (g) {}
    _.pm.Xf({
        url: ":socialhost:/:session_prefix:_/widget/oauthflow/toast",
        queryParams: {
            clientId: a.client_id,
            idToken: a.id_token
        },
        where: d,
        onRestyle: function() {
            "top" === b ? Cv(!0) : Cv(!1)
        }
    })
};
Hv=!1;
Iv=!1;
_.Jv = function(a) {
    _.Ab(a) && _.Gb(a)
};
Lv = function(a) {
    var b, c = null;
    _.kj.iterate(function(a, d) {
        if (0 === a.indexOf("G_AUTHUSER_")) {
            var e = _.mj(a.substring(11));
            if (!b || e.yb&&!b.yb || e.yb == b.yb && e.wd > b.wd)
                b = e, c = d
        }
    });
    var d = a && _.lj(a);
    if (null !== c) {
        var e;
        _.kj.iterate(function(a, c) {
            var f = _.nj(a);
            f && f.Dc && (d && f.zu != d || f.yb == b.yb && f.wd == b.wd && (e = c))
        });
        if (e) {
            var f = Dv(e);
            a = f && f.kJ[Number(c)];
            f = f && f.yu;
            if (a)
                return {
                    pl: c,
                    KT: a,
                    yu: f
                }
        }
    }
    return null
};
Mv = function(a) {
    a = Ev(a.g_user_cookie_policy);
    _.eb("googleapis.config/sessionIndex", null);
    a.clear()
};
Nv = function(a) {
    return "true" === String(a.immediate)
};
Ov = function(a, b) {
    var c = null;
    if (a && b) {
        c = b.client_id = b.client_id || a.client_id;
        b.scope = b.scope || a.scope;
        b.g_user_cookie_policy = a.cookie_policy;
        b.cookie_policy = b.cookie_policy || a.cookie_policy;
        b.response_type = b.response_type || a.response_type;
        var d = _.E("iframes/signin/iframeType");
        !Hv && b.error && Nv(a) && (_.Jv("gapi.signin.impression." + d + "." + c), Hv=!0)
    }
    b && (b.issued_at || (b.issued_at = String(_.aj())), d = (0, window.parseInt)(b.expires_in, 10) || 86400, b.error && (d = _.E("oauth-flow/errorMaxAge") || 86400), b.expires_in =
    String(d), b.expires_at || (b.expires_at = String(_.aj() + d)), a && Nv(a) || (b["g-oauth-window"] = (Kv || {}).popup), b._aa || b.error || null != Lv(c) ||!Nv(a) || (b._aa = "1"), c = b.status = {}, c.google_logged_in=!!b.session_state, d = c.signed_in=!!b.access_token, c.method = d ? b["g-oauth-window"] ? "PROMPT" : "AUTO" : null);
    return b
};
Pv = {};
_.Qv = function(a, b) {
    var c, d;
    "string" == typeof a ? (c = b, d = a) : (c = a, d = "token");
    if (c) {
        var e = _.Aj(c, !0);
        if (e) {
            var f;
            if ((f = c) && f.session_state) {
                var g = [], k = [], l = [], m = (0, window.parseInt)(f.authuser, 10) || 0;
                g[m] = f.session_state;
                k[m] = f.issued_at;
                l[m] = f.expires_at;
                f = ["C=" + f.client_id, "S=" + g.join("|"), "I=" + k.join("|"), "X=" + l.join("|")].join(":")
            } else 
                f = null;
            f && e.rh.write(f);
            "token" == d && (e = Ev(c.g_user_cookie_policy)) && (c.error ? Mv(c) : c.session_state && e.write(c.authuser || "0"))
        }
    }
    if (d = "token" !== d ? _.zj(d) : _.Aj(c))
        if (c) {
            if (d.rh.write(c),
            !_.sj || d.mE && "token" !== _.sj)
                _.sj = d.key
        } else 
            d.rh.clear(), _.sj = null;
    _.Bj(void 0, void 0)
};
Rv = function(a, b) {
    var c = b && b.key || "token", d = a = Ov(b && b.params, a);
    !Fv(d) && 0 <= (" " + (d.scope || "") + " ").indexOf(" https://www.googleapis.com/auth/plus.login ") && _.E("isLoggedIn") && "1" === d._aa && (d._aa = "0", Iv || (Iv=!0, Gv(d)));
    _.Qv(c, a);
    a = _.Bj(c, void 0);
    if (b) {
        (c = b.iframe) && c.parentNode.removeChild(c);
        c = b.popup;
        d = b.after_redirect;
        if (c && "keep_open" != d)
            try {
                c.close()
            } catch (e) {}
        b.timeout && (window.clearTimeout(b.timeout), b.timeout = null);
        b.callback && (b.callback(a), b.callback = null)
    }
};
Sv = {};
Tv = {};
Uv = [];
Vv = function() {
    var a = Uv;
    if (null !== a) {
        Uv = null;
        for (var b = 0, c = a.length; b < c; b++)
            a[b]()
    }
};
Wv = function(a) {
    var b = _.fv;
    return function(c) {
        if (this.f == b && this.t == _.J.Ag(this.f) && this.origin == _.J.Of(this.f))
            return a.apply(this, arguments)
    }
};
Xv = 0;
Yv = function(a) {
    var b = {
        error: "user_signed_out"
    };
    b.client_id = a.client_id;
    b.g_user_cookie_policy = a.g_user_cookie_policy;
    b.scope = a.scope;
    b.response_type = a.response_type;
    b.session_state = a.session_state;
    return Ov(null, b)
};
Zv = function(a, b) {
    var c = a || {}, d;
    for (d in _.ev)
        _.ca(c[d]) || (c[d] = _.ev[d]);
    a = c;
    c = _.E("googleapis/overrideClientId");
    null != c && (a.client_id = c);
    if (!a.redirect_uri || "postmessage" === a.redirect_uri) {
        c = a;
        d = a.state || "";
        d = String(d);
        var e = "";
        if ({}.hasOwnProperty.call(Tv, d))
            e = Tv[d];
        else {
            for (var f = 2147483647 * (0, _.Nj)() | 0; ;) {
                e = String(f);
                if (!{}.hasOwnProperty.call(Pv, e))
                    break;
                f += (0, _.Nj)()
            }
            Pv[e] = d;
            Tv[d] = e
        }
        c.state = e + "|" + (0, _.Nj)();
        Sv[a.state] = b
    }
    c = a.authorize_uri || _.E("oauth-flow/authUrl");
    delete a.authorize_uri;
    c +=
    0 < c.indexOf("?") ? "&" : "?";
    c += _.vj(a);
    c = _.Wi(c);
    d = _.E("iframes/signin/iframeType");
    "blue" == d ? c += "&e=3100070" : "red" == d ? c += "&e=3100071" : "default" == d && (c += "&e=3100077");
    return c
};
$v = function(a) {
    var b = _.Jn(), c = b && b.scope, b = a && a.scope, b = "string" === typeof b ? b.split(" "): b || [];
    if (c) {
        for (var c = c.split(" "), d = 0; d < c.length; ++d) {
            var e = c[d];
            - 1 == _.Mb.call(b, e) && b.push(e)
        }
        0 < b.length && (a.scope = b.join(" "))
    }
    return a
};
aw = function(a) {
    if (!_.ca(a.include_granted_scopes)) {
        var b = _.E("include_granted_scopes"), c = _.Jn().include_granted_scopes;
        _.ca(b) ? a.include_granted_scopes=!!b : _.ca(c) && (a.include_granted_scopes = "1" == c || "true" == c)
    }
    return a
};
_.bw = function(a) {
    if (null === Uv)
        a && a();
    else {
        a && Uv.push(a);
        a = _.fv;
        var b = window.document.getElementById(a), c = (new Date).getTime();
        if (b) {
            if (Xv && 6E4 > c - Xv)
                return;
            b.parentNode.removeChild(b);
            if (/Firefox/.test(window.navigator.userAgent))
                try {
                    window.frames[a] = void 0
            } catch (d) {}
            _.gv();
            a = _.fv
        }
        Xv = c;
        var e = String(2147483647 * (0, _.Nj)() | 0);
        _.J.register("oauth2relayReady:" + e, Wv(function() {
            _.J.uh("oauth2relayReady:" + e);
            Vv()
        }));
        _.J.register("oauth2relayReady", Wv(function() {
            Vv()
        }));
        b = Wv(function(a) {
            var b = _.H.Fb;
            a = b(a);
            var b = a.state, c;
            c = b.replace(/\|.*$/, "");
            a.state = {}.hasOwnProperty.call(Pv, c) ? Pv[c] : null;
            null != a.state && (c = Sv[b], delete Sv[b], Rv(a, c))
        });
        _.J.register("oauth2callback:" + e, b);
        _.J.register("oauth2callback", b);
        var b = [_.E("oauth-flow/proxyUrl") || _.E("oauth-flow/relayUrl"), "?parent=", (0, window.encodeURIComponent)(_.Mi.Ba(window.location.href)), "#rpctoken=", e, "&forcesecure=1"].join(""), c = _.H.mv(), f = _.H.Vo({
            name: a,
            id: a
        });
        f.src = b;
        f.style.width = "1px";
        f.style.height = "1px";
        f.style.position = "absolute";
        f.style.top =
        "-100px";
        f.tabIndex = "-1";
        c.appendChild(f);
        _.J.Dk(a)
    }
};
cw = function(a, b, c, d) {
    var e = zv(c), f = e[a];
    e[a] = b ? "0" : "1";
    var g = [];
    _.ib(e, function(a, b) {
        g.push(b + "=" + a)
    });
    b = g.join(":");
    c = uv(c);
    b ? c.write(b) : c.clear();
    e[a] !== f && d && d()
};
dw = function(a, b) {
    for (var c = a.split(" "), d = b.split(" "), e = _.Ka(), f = 0, g = d.length; f < g; ++f)
        d[f] && (e[d[f]] = 1);
    f = 0;
    for (g = c.length; f < g; ++f)
        if (c[f]&&!e[c[f]])
            return !1;
    return !0
};
ew = function(a) {
    if (!a)
        return null;
    var b, c, d = _.lj(a);
    _.gj.iterate(function(a) {
        var f = _.nj(a);
        f && f.Dc && f.zu === d && (!b || f.yb&&!b.yb || f.yb == b.yb && f.wd > b.wd) && (b = f, c = a)
    });
    return c
};
fw = function(a) {
    var b = _.H.mv(), c = _.H.Vo();
    c.src = a.uri;
    c.style.width = "1px";
    c.style.height = "1px";
    c.style.position = "absolute";
    c.style.top = "-100px";
    a.timeout = window.setTimeout(function() {
        c.parentNode && c.parentNode.removeChild(c)
    }, 3E5);
    b.appendChild(c)
};
gw = function(a, b) {
    if (Kv) {
        var c = Kv.popup, d = Kv.after_redirect;
        if (c && "keep_open" != d&&!Nv(a))
            try {
                c.close()
            } catch (e) {}
    }
    c = Kv = {};
    "key"in a && (c.key = a.key, delete a.key);
    a = aw(a);
    c.params = $v(a);
    c.callback = function(c) {
        Nv(a) ||!c || c.error || cw(Bv(c), !1, c.cookie_policy);
        Fv(c) && Nv(a) && (c = Yv(c));
        (b || function() {})(Ov(a, c))
    };
    c.uri = Zv(a, c);
    return c
};
_.hw = function(a, b) {
    _.bw(function() {
        _.J.call(_.fv, "check_session_state", Wv(function(a) {
            b.call(null, a)
        }), a.session_state, a.client_id)
    })
};
iw = function(a) {
    a = _.H.Fb("#" + _.vj(a));
    if (!Nv(a))
        return null;
    var b = a.key || ew(a.client_id);
    return (b = _.Bj(b, !0, !0)) && b.client_id === a.client_id && dw(a.scope, b.scope) && dw(a.response_type, b.response_type) ? b : null
};
jw = function(a, b) {
    var c = _.H.Fb();
    a.hl = c.lang || c.hl || _.E("lang");
    var d = gw(a, b);
    a.after_redirect && (d.after_redirect = a.after_redirect);
    if (null != a.scope && null != a.client_id) {
        var e = function() {
            _.bw(function() {
                if (d.popup)
                    d.popup.focus();
                else if (Nv(a))
                    fw(d);
                else {
                    var b = Math.min(_.E("oauth-flow/authWindowWidth", 650), window.screen.width - 20), c = Math.min(_.E("oauth-flow/authWindowHeight", 600), window.screen.height - 30);
                    Kv.popup = window.open(d.uri, "_blank", ["toolbar=no", "location=" + (window.opera ? "no" : "yes"), "directories=no,status=no,menubar=no,scrollbars=yes,resizable=yes,copyhistory=no",
                    "width=" + b, "height=" + c, "top=" + (window.screen.height - c) / 2, "left=" + (window.screen.width - b) / 2].join())
                }
            })
        }, f = iw(a);
        !_.E("oauth-flow/disableOpt") && f ? _.hw(f, function(a) {
            a ? Rv(f, Kv) : e()
        }) : e()
    } else 
        Rv(null, Kv), _.Qb("Unable to perform authorization: scope and/or client_idparameters missing.")
};
kw = function(a) {
    a || (a = _.Bj(void 0, !0));
    a && "object" === typeof a || (a = {
        error: "invalid_request",
        error_description: "no callback data"
    });
    var b = a.error_description;
    b && window.console && (window.console.error(a.error), window.console.error(b));
    a.error || (_.Ya.drw = null);
    _.Qv(a);
    if (b = a.authuser)
        _.E("googleapis.config/sessionIndex"), _.eb("googleapis.config/sessionIndex", b);
    _.hv.A.mm(_.iv.toString(), a);
    return a
};
lw = function(a, b) {
    var c = a || {}, d = b || function() {};
    if (_.E("oauth-flow/disableOpt") || _.E("isLoggedIn") ||!Nv(c))
        jw(c, d);
    else {
        var e = _.Ka();
        e.client_id = c.client_id;
        e.session_state = null;
        _.hw(e, function(a) {
            a ? (a = _.Ka(), a.error = "immediate_failed_user_logged_out", d(Ov(c, a))) : (_.eb("isLoggedIn", !0), jw(c, d))
        })
    }
};
mw = function(a, b) {
    var c = Bv(a);
    c && (Mv(a), cw(c, !0, b, function() {
        kw && kw(Yv(a))
    }))
};
_.nw = function(a, b) {
    var c = new _.Qf(function(b, c) {
        var f = function(a) {
            null == a || a.error ? c(a) : b(a)
        };
        try {
            lw(a, f)
        } catch (g) {
            c(g)
        }
    });
    b && c.then(b, function(a) {
        b(kw(a))
    });
    return c
};
_.ow = function(a) {
    _.Ob && (_.Ob.error ? _.Ob.error(a) : _.Ob.log && _.Ob.log(a))
};
pw = _.Lj.Wl;
qw = null;
_.tw = function(a, b) {
    if ("force" !== a.approvalprompt) {
        var c = _.rw(a);
        c.immediate=!0;
        delete c.redirect_uri;
        delete c.approval_prompt;
        var d;
        if (d=!b)
            qw ? (c.client_id !== qw.client_id && window.console && window.console.log && window.console.log("Ignoring mismatched page-level auth param client_id=" + c.client_id), d=!0) : (qw = c, d=!1);
        d || sw(c)
    }
};
_.rw = function(a) {
    var b = a.redirecturi || "postmessage", c = (0, _.Xc)((a.scope || "").replace(/[\s\xa0]+/g, " ")), b = {
        client_id: a.clientid,
        redirect_uri: b,
        response_type: "code token id_token gsession",
        scope: c
    };
    a.approvalprompt && (b.approval_prompt = a.approvalprompt);
    a.state && (b.state = a.state);
    a.openidrealm && (b["openid.realm"] = a.openidrealm);
    c = "offline" == a.accesstype?!0 : (c = a.redirecturi) && "postmessage" != c;
    c && (b.access_type = "offline");
    a.requestvisibleactions && (b.request_visible_actions = (0, _.Xc)(a.requestvisibleactions.replace(/[\s\xa0]+/g,
    " ")));
    a.after_redirect && (b.after_redirect = a.after_redirect);
    a.cookiepolicy && "none" !== a.cookiepolicy && (b.cookie_policy = a.cookiepolicy);
    "none" === a.cookiepolicy && _.Jv("gapi.signin.cpnone");
    "undefined" != typeof a.includegrantedscopes && (b.include_granted_scopes = a.includegrantedscopes);
    a.e && (b.e = a.e);
    (a = a.authuser || _.E("googleapis.config/sessionIndex")) && (b.authuser = a);
    (a = _.E("useoriginassocialhost")) && (b.use_origin_as_socialhost = a);
    return b
};
sw = function(a) {
    _.Eo("waaf0", "signin", "0");
    _.nw(a, function(a) {
        _.Eo("waaf1", "signin", "0");
        kw(a)
    })
};
_.uw = function(a) {
    a = _.rw(a);
    _.eb("oauth-flow/authWindowWidth", 445);
    _.eb("oauth-flow/authWindowHeight", 615);
    sw(a)
};
_.vw = function(a) {
    _.ov(_.hv.A, _.iv.toString(), a, void 0);
    _.hv.subscribe(_.iv, a)
};
var Cw, Fw;
_.xw = function(a) {
    return a.cookiepolicy?!0 : (_.ww("cookiepolicy is a required field.  See https://developers.google.com/+/web/signin/#button_attr_cookiepolicy for more information."), !1)
};
_.ww = function(a) {
    window.console && (window.console.error ? window.console.error(a) : window.console.log && window.console.log(a))
};
_.Bw = function(a, b) {
    var c = _.Jn();
    _.tl(a, c);
    c = $v(c);
    if (_.xw(c)) {
        var d = _.yw(c);
        _.zw(c);
        b ? _.Wa(b, "click", function() {
            _.Aw(c, d)
        }) : _.Aw(c, d)
    }
};
_.yw = function(a) {
    var b = new Cw;
    _.vw(function(c) {
        if (b.A && c && (c.access_token && _.eb("isPlusUser", !0), c["g-oauth-window"]))
            if (b.A=!1, c.access_token && "consent" == c.prompt) {
                var d = c["g-oauth-window"];
                c = c.id_token;
                var e = a.apppackagename;
                if (e && d) {
                    var f=!0;
                    try {
                        f = d.closed
                    } catch (g) {}
                    f || (c = _.E("iframes/:socialhost:") + "/_/history/otaappinstall?clientId=" + (0, window.encodeURIComponent)(a.clientid) + "&appId=" + (0, window.encodeURIComponent)(e) + "&idToken=" + (0, window.encodeURIComponent)(c), c = _.Wi(c), d.location.href = c)
                }
            } else if (d =
            c["g-oauth-window"], a.apppackagename && d)
                try {
                    d.close()
        } catch (k) {}
    });
    return b
};
Cw = function() {
    this.A=!1
};
_.zw = function(a) {
    a = _.Dw(a);
    _.Ew(a.callback);
    _.bw(function() {
        _.tw(a)
    })
};
_.Dw = function(a) {
    Fw(a);
    a.apppackagename && (a.after_redirect = "keep_open");
    a.redirecturi && delete a.redirecturi;
    _.Vi(function(b) {
        return a[b]
    }) || (a.authuser = 0);
    a.apppackagename && (a.apppackagename = a.apppackagename.replace(/^[\s\xa0]+|[\s\xa0]+$/g, ""));
    return a
};
Fw = function(a) {
    /^\s*$/.test(a.scope || "") && (a.scope = "https://www.googleapis.com/auth/plus.login")
};
_.Ew = function(a) {
    if ("string" === typeof a)
        if (window[a])
            a = window[a];
        else {
            _.ww('Callback function named "' + a + '" not found');
            return 
        }
    a && _.vw(a)
};
_.Aw = function(a, b) {
    b.A=!0;
    var c;
    c = _.Dw(a);
    _.uw(c)
};
_.C("gapi.auth.authorize", _.nw);
_.C("gapi.auth.checkSessionState", _.hw);
_.C("gapi.auth.getAuthHeaderValueForFirstParty", pw);
_.C("gapi.auth.getToken", function(a, b) {
    return _.Bj(a, b)
});
_.C("gapi.auth.getVersionInfo", function(a, b) {
    _.bw(function() {
        var c = _.Lj.Wl() || "", d = null, e = null;
        c && (e = c.split(" "), 2 == e.length && (d = e[1]));
        d ? _.J.call(_.fv, "get_versioninfo", Wv(function(b) {
            a(b)
        }), d, b) : a()
    })
});
_.C("gapi.auth.init", _.bw);
_.C("gapi.auth.setToken", _.Qv);
_.C("gapi.auth.signIn", function(a) {
    _.Bw(a)
});
_.C("gapi.auth.signOut", function() {
    var a = _.Bj(void 0, void 0);
    a && mw(a, a.cookie_policy)
});
_.C("gapi.auth.unsafeUnpackIdToken", vv);
_.C("gapi.auth._pimf", _.tw);
_.C("gapi.auth._oart", Gv);
_.C("gapi.auth._guss", function(a) {
    return uv(a).bg()
});
var Gw = _.Jn();
Gw.clientid && Gw.scope && Gw.cookiepolicy && _.Jv("gapi.auth.pageLevelConfig(" + _.mb(Gw).sort().join("-") + ")");
Gw.clientid && Gw.scope && Gw.callback&&!_.E("disableRealtimeCallback") ? _.zw(Gw) : _.bw();
/*
 Portions of this code are from MochiKit, received by
 The Closure Authors under the MIT license. All other code is Copyright
 2005-2009 The Closure Authors. All Rights Reserved.
*/
_.$f = function(a, b) {
    this.F = [];
    this.U = a;
    this.R = b || null;
    this.D = this.B=!1;
    this.C = void 0;
    this.V = this.ya = this.K=!1;
    this.M = 0;
    this.A = null;
    this.G = 0
};
_.$f.prototype.cancel = function(a) {
    if (this.B)
        this.C instanceof _.$f && this.C.cancel();
    else {
        if (this.A) {
            var b = this.A;
            delete this.A;
            a ? b.cancel(a) : (b.G--, 0 >= b.G && b.cancel())
        }
        this.U ? this.U.call(this.R, this) : this.V=!0;
        this.B || this.Ol(new _.ag)
    }
};
_.$f.prototype.Y = function(a, b) {
    this.K=!1;
    bg(this, a, b)
};
var bg = function(a, b, c) {
    a.B=!0;
    a.C = c;
    a.D=!b;
    cg(a)
}, eg = function(a) {
    if (a.B) {
        if (!a.V)
            throw new dg;
        a.V=!1
    }
};
_.h = _.$f.prototype;
_.h.nc = function(a) {
    eg(this);
    bg(this, !0, a)
};
_.h.Ol = function(a) {
    eg(this);
    bg(this, !1, a)
};
_.h.Ob = function(a, b) {
    return this.Df(a, null, b)
};
_.h.Df = function(a, b, c) {
    this.F.push([a, b, c]);
    this.B && cg(this);
    return this
};
_.h.then = function(a, b, c) {
    var d, e, f = new _.Qf(function(a, b) {
        d = a;
        e = b
    });
    this.Df(d, function(a) {
        a instanceof _.ag ? f.cancel() : e(a)
    });
    return f.then(a, b, c)
};
_.Nf(_.$f);
_.$f.prototype.N = _.q(3);
var fg = function(a) {
    return (0, _.gd)(a.F, function(a) {
        return _.Uc(a[1])
    })
}, cg = function(a) {
    if (a.M && a.B && fg(a)) {
        var b = a.M, c = gg[b];
        c && (_.r.clearTimeout(c.qa), delete gg[b]);
        a.M = 0
    }
    a.A && (a.A.G--, delete a.A);
    for (var b = a.C, d = c=!1; a.F.length&&!a.K;) {
        var e = a.F.shift(), f = e[0], g = e[1], e = e[2];
        if (f = a.D ? g : f)
            try {
                var k = f.call(e || a.R, b);
                _.ca(k) && (a.D = a.D && (k == b || k instanceof Error), a.C = b = k);
                _.Of(b) && (d=!0, a.K=!0)
            } catch (l) {
            b = l, a.D=!0, fg(a) || (c=!0)
        }
    }
    a.C = b;
    d && (k = (0, _.s)(a.Y, a, !0), d = (0, _.s)(a.Y, a, !1), b instanceof _.$f ? (b.Df(k,
    d), b.ya=!0) : b.then(k, d));
    c && (b = new hg(b), gg[b.qa] = b, a.M = b.qa)
}, dg = function() {
    _.xf.call(this)
};
_.u(dg, _.xf);
dg.prototype.message = "Deferred has already fired";
dg.prototype.name = "AlreadyCalledError";
_.ag = function() {
    _.xf.call(this)
};
_.u(_.ag, _.xf);
_.ag.prototype.message = "Deferred was canceled";
_.ag.prototype.name = "CanceledError";
var hg = function(a) {
    this.qa = _.r.setTimeout((0, _.s)(this.B, this), 0);
    this.A = a
};
hg.prototype.B = function() {
    delete gg[this.qa];
    throw this.A;
};
var gg = {};

_.bt = function(a, b, c, d) {
    return _.ad.splice.apply(a, _.Gd(arguments, 1))
};
_.ct = function(a, b, c, d, e) {
    if (_.ha(b)) {
        for (var f = 0; f < b.length; f++)
            _.ct(a, b[f], c, d, e);
        return null
    }
    c = _.Ue(c);
    return _.Je(a) ? a.Ee.add(String(b), c, !0, d, e) : _.Ve(a, b, c, !0, d, e)
};
_.dt = function(a, b, c) {
    if (b in a)
        throw Error("i`" + b);
    a[b] = c
};
_.et = function(a) {
    if (!_.ha(a))
        for (var b = a.length - 1; 0 <= b; b--)
            delete a[b];
    a.length = 0
};
_.ft = function(a) {
    a.ja = function() {
        return a.dx ? a.dx : a.dx = new a
    }
};
_.gt = function(a) {
    _.te.call(this);
    this.Bc = a;
    this.C = {}
};
_.u(_.gt, _.te);
var ht = [];
_.gt.prototype.P = function(a, b, c, d) {
    return _.it(this, a, b, c, d)
};
_.it = function(a, b, c, d, e, f) {
    _.ha(c) || (c && (ht[0] = c.toString()), c = ht);
    for (var g = 0; g < c.length; g++) {
        var k = _.N(b, c[g], d || a.handleEvent, e ||!1, f || a.Bc || a);
        if (!k)
            break;
        a.C[k.key] = k
    }
    return a
};
_.gt.prototype.La = function(a, b, c, d, e) {
    if (_.ha(b))
        for (var f = 0; f < b.length; f++)
            this.La(a, b[f], c, d, e);
    else 
        c = c || this.handleEvent, e = e || this.Bc || this, c = _.Ue(c), d=!!d, b = _.Je(a) ? _.Qe(a.Ee, String(b), c, d, e) : a ? (a = _.We(a)) ? _.Qe(a, b, c, d, e) : null : null, b && (_.af(b), delete this.C[b.key]);
    return this
};
_.jt = function(a) {
    _.Kd(a.C, _.af);
    a.C = {}
};
_.gt.prototype.W = function() {
    _.gt.J.W.call(this);
    _.jt(this)
};
_.gt.prototype.handleEvent = function() {
    throw Error("L");
};

var Pt, cu, Yt, iu, Zt, au, $t, fu, bu, ku;
Pt = function(a) {
    return _.ad.concat.apply(_.ad, arguments)
};
_.Qt = function() {
    return Math.floor(2147483648 * Math.random()).toString(36) + Math.abs(Math.floor(2147483648 * Math.random())^(0, _.sa)()).toString(36)
};
_.Rt = function(a, b) {
    var c;
    a instanceof _.Rt ? (this.Lg = _.ca(b) ? b : a.Lg, _.St(this, a.re), _.Tt(this, a.ii), _.Ut(this, a.Ke), _.Vt(this, a.dh), this.setPath(a.getPath()), _.Wt(this, a.A.clone()), this.B(a.C())) : a && (c = _.hh(String(a))) ? (this.Lg=!!b, _.St(this, c[1] || "", !0), _.Tt(this, c[2] || "", !0), _.Ut(this, c[3] || "", !0), _.Vt(this, c[4]), this.setPath(c[5] || "", !0), _.Wt(this, c[6] || "", !0), this.B(c[7] || "", !0)) : (this.Lg=!!b, this.A = new _.Xt(null, 0, this.Lg))
};
_.h = _.Rt.prototype;
_.h.re = "";
_.h.ii = "";
_.h.Ke = "";
_.h.dh = null;
_.h.Wj = "";
_.h.Hp = "";
_.h.mx=!1;
_.h.Lg=!1;
_.h.toString = function() {
    var a = [], b = this.re;
    b && a.push(Yt(b, Zt, !0), ":");
    if (b = this.Ke) {
        a.push("//");
        var c = this.ii;
        c && a.push(Yt(c, Zt, !0), "@");
        a.push((0, window.encodeURIComponent)(String(b)).replace(/%25([0-9a-fA-F]{2})/g, "%$1"));
        b = this.dh;
        null != b && a.push(":", String(b))
    }
    if (b = this.getPath())
        this.Ke && "/" != b.charAt(0) && a.push("/"), a.push(Yt(b, "/" == b.charAt(0) ? $t : au, !0));
    (b = this.A.toString()) && a.push("?", b);
    (b = this.C()) && a.push("#", Yt(b, bu));
    return a.join("")
};
_.h.resolve = function(a) {
    var b = this.clone(), c=!!a.re;
    c ? _.St(b, a.re) : c=!!a.ii;
    c ? _.Tt(b, a.ii) : c=!!a.Ke;
    c ? _.Ut(b, a.Ke) : c = null != a.dh;
    var d = a.getPath();
    if (c)
        _.Vt(b, a.dh);
    else if (c=!!a.Wj) {
        if ("/" != d.charAt(0))
            if (this.Ke&&!this.Wj)
                d = "/" + d;
            else {
                var e = b.getPath().lastIndexOf("/");
                - 1 != e && (d = b.getPath().substr(0, e + 1) + d)
            }
        e = d;
        if (".." == e || "." == e)
            d = "";
        else if (_.kd(e, "./") || _.kd(e, "/.")) {
            for (var d = _.Wc(e, "/"), e = e.split("/"), f = [], g = 0; g < e.length;) {
                var k = e[g++];
                "." == k ? d && g == e.length && f.push("") : ".." == k ? ((1 < f.length ||
                1 == f.length && "" != f[0]) && f.pop(), d && g == e.length && f.push("")) : (f.push(k), d=!0)
            }
            d = f.join("/")
        } else 
            d = e
    }
    c ? b.setPath(d) : c = "" !== a.A.toString();
    c ? _.Wt(b, cu(a.A.toString())) : c=!!a.Hp;
    c && b.B(a.C());
    return b
};
_.h.clone = function() {
    return new _.Rt(this)
};
_.St = function(a, b, c) {
    _.du(a);
    a.re = c ? cu(b, !0) : b;
    a.re && (a.re = a.re.replace(/:$/, ""));
    return a
};
_.Tt = function(a, b, c) {
    _.du(a);
    a.ii = c ? cu(b) : b;
    return a
};
_.Ut = function(a, b, c) {
    _.du(a);
    a.Ke = c ? cu(b, !0) : b;
    return a
};
_.Vt = function(a, b) {
    _.du(a);
    if (b) {
        b = Number(b);
        if ((0, window.isNaN)(b) || 0 > b)
            throw Error("M`" + b);
        a.dh = b
    } else 
        a.dh = null;
    return a
};
_.Rt.prototype.getPath = function() {
    return this.Wj
};
_.Rt.prototype.setPath = function(a, b) {
    _.du(this);
    this.Wj = b ? cu(a, !0) : a;
    return this
};
_.Wt = function(a, b, c) {
    _.du(a);
    b instanceof _.Xt ? (a.A = b, eu(a.A, a.Lg)) : (c || (b = Yt(b, fu)), a.A = new _.Xt(b, 0, a.Lg));
    return a
};
_.Rt.prototype.Ka = function(a, b) {
    return _.Wt(this, a, b)
};
_.Rt.prototype.Ie = function() {
    return this.A.toString()
};
_.gu = function(a, b, c) {
    _.du(a);
    a.A.set(b, c);
    return a
};
_.Rt.prototype.Ld = function(a) {
    return this.A.get(a)
};
_.Rt.prototype.C = function() {
    return this.Hp
};
_.Rt.prototype.B = function(a, b) {
    _.du(this);
    this.Hp = b ? cu(a) : a;
    return this
};
_.du = function(a) {
    if (a.mx)
        throw Error("N");
};
_.hu = function(a) {
    return a instanceof _.Rt ? a.clone() : new _.Rt(a, void 0)
};
cu = function(a, b) {
    return a ? b ? (0, window.decodeURI)(a) : (0, window.decodeURIComponent)(a) : ""
};
Yt = function(a, b, c) {
    return _.ja(a) ? (a = (0, window.encodeURI)(a).replace(b, iu), c && (a = a.replace(/%25([0-9a-fA-F]{2})/g, "%$1")), a) : null
};
iu = function(a) {
    a = a.charCodeAt(0);
    return "%" + (a>>4 & 15).toString(16) + (a & 15).toString(16)
};
Zt = /[#\/\?@]/g;
au = /[\#\?:]/g;
$t = /[\#\?]/g;
fu = /[\#\?@]/g;
bu = /#/g;
_.Xt = function(a, b, c) {
    this.A = a || null;
    this.B=!!c
};
ku = function(a) {
    if (!a.Bb && (a.Bb = new _.mf, a.gd = 0, a.A))
        for (var b = a.A.split("&"), c = 0; c < b.length; c++) {
            var d = b[c].indexOf("="), e = null, f = null;
            0 <= d ? (e = b[c].substring(0, d), f = b[c].substring(d + 1)) : e = b[c];
            e = (0, window.decodeURIComponent)(e.replace(/\+/g, " "));
            e = ju(a, e);
            a.add(e, f ? (0, window.decodeURIComponent)(f.replace(/\+/g, " ")) : "")
        }
};
_.h = _.Xt.prototype;
_.h.Bb = null;
_.h.gd = null;
_.h.fb = function() {
    ku(this);
    return this.gd
};
_.h.add = function(a, b) {
    ku(this);
    this.A = null;
    a = ju(this, a);
    var c = this.Bb.get(a);
    c || this.Bb.set(a, c = []);
    c.push(b);
    this.gd++;
    return this
};
_.h.remove = function(a) {
    ku(this);
    a = ju(this, a);
    return _.pf(this.Bb.B, a) ? (this.A = null, this.gd -= this.Bb.get(a).length, this.Bb.remove(a)) : !1
};
_.h.clear = function() {
    this.Bb = this.A = null;
    this.gd = 0
};
_.h.isEmpty = function() {
    ku(this);
    return 0 == this.gd
};
var lu = function(a, b) {
    ku(a);
    b = ju(a, b);
    return _.pf(a.Bb.B, b)
};
_.h = _.Xt.prototype;
_.h.Mh = function(a) {
    var b = this.xc();
    return _.id(b, a)
};
_.h.fd = function() {
    ku(this);
    for (var a = this.Bb.xc(), b = this.Bb.fd(), c = [], d = 0; d < b.length; d++)
        for (var e = a[d], f = 0; f < e.length; f++)
            c.push(b[d]);
    return c
};
_.h.xc = function(a) {
    ku(this);
    var b = [];
    if (_.ja(a))
        lu(this, a) && (b = Pt(b, this.Bb.get(ju(this, a))));
    else {
        a = this.Bb.xc();
        for (var c = 0; c < a.length; c++)
            b = Pt(b, a[c])
    }
    return b
};
_.h.set = function(a, b) {
    ku(this);
    this.A = null;
    a = ju(this, a);
    lu(this, a) && (this.gd -= this.Bb.get(a).length);
    this.Bb.set(a, [b]);
    this.gd++;
    return this
};
_.h.get = function(a, b) {
    var c = a ? this.xc(a): [];
    return 0 < c.length ? String(c[0]) : b
};
_.mu = function(a, b, c) {
    a.remove(b);
    0 < c.length && (a.A = null, a.Bb.set(ju(a, b), _.Ld(c)), a.gd += c.length)
};
_.Xt.prototype.toString = function() {
    if (this.A)
        return this.A;
    if (!this.Bb)
        return "";
    for (var a = [], b = this.Bb.fd(), c = 0; c < b.length; c++)
        for (var d = b[c], e = (0, window.encodeURIComponent)(String(d)), d = this.xc(d), f = 0; f < d.length; f++) {
            var g = e;
            "" !== d[f] && (g += "=" + (0, window.encodeURIComponent)(String(d[f])));
            a.push(g)
        }
    return this.A = a.join("&")
};
_.Xt.prototype.clone = function() {
    var a = new _.Xt;
    a.A = this.A;
    this.Bb && (a.Bb = this.Bb.clone(), a.gd = this.gd);
    return a
};
var ju = function(a, b) {
    var c = String(b);
    a.B && (c = c.toLowerCase());
    return c
}, eu = function(a, b) {
    b&&!a.B && (ku(a), a.A = null, a.Bb.forEach(function(a, b) {
        var e = b.toLowerCase();
        b != e && (this.remove(b), _.mu(this, e, a))
    }, a));
    a.B = b
};
_.Xt.prototype.extend = function(a) {
    for (var b = 0; b < arguments.length; b++)
        _.jf(arguments[b], function(a, b) {
            this.add(b, a)
        }, this)
    };

var wD;
wD = function(a, b) {
    return a > b ? 1 : a < b?-1 : 0
};
_.xD = function(a) {
    var b = _.fa(a);
    if ("object" == b || "array" == b) {
        if (a.clone)
            return a.clone();
        var b = "array" == b ? []: {}, c;
        for (c in a)
            b[c] = _.xD(a[c]);
        return b
    }
    return a
};
_.yD = function(a, b) {
    a.sort(b || wD)
};
_.VK = {
    s: function(a, b, c) {
        return (0, window.isNaN)(c) || "" == c || a.length >= c ? a : a =- 1 < b.indexOf("-", 0) ? a + _.Yc(" ", c - a.length) : _.Yc(" ", c - a.length) + a
    },
    f: function(a, b, c, d, e) {
        d = a.toString();
        (0, window.isNaN)(e) || "" == e || (d = a.toFixed(e));
        var f;
        f = 0 > a ? "-" : 0 <= b.indexOf("+") ? "+" : 0 <= b.indexOf(" ") ? " " : "";
        0 <= a && (d = f + d);
        if ((0, window.isNaN)(c) || d.length >= c)
            return d;
        d = (0, window.isNaN)(e) ? Math.abs(a).toString() : Math.abs(a).toFixed(e);
        a = c - d.length - f.length;
        return d = 0 <= b.indexOf("-", 0) ? f + d + _.Yc(" ", a) : f + _.Yc(0 <= b.indexOf("0", 0) ? "0" : " ", a) + d
    },
    d: function(a, b, c, d, e, f, g, k) {
        return _.VK.f((0, window.parseInt)(a, 10), b, c, d, 0, f, g, k)
    }
};
_.VK.i = _.VK.d;
_.VK.u = _.VK.d;

var WK = function(a) {
    a = _.zf(a);
    _.Kd(a, function(b, c) {
        _.hk(c) && (a[c] = b)
    });
    return a
}, XK = {
    error: {
        code: - 1,
        message: "A network error occurred and the request could not be completed."
    }
}, YK = function(a) {
    return new _.Qf(function(b, c) {
        c(a)
    })
}, ZK = function(a) {
    a = _.zf(a);
    _.ib(a, function(b, c) {
        _.ha(b) && (a[c] = b.join(", "))
    });
    return a
}, $K = function(a, b, c, d) {
    _.th.call(this);
    this.mb = a;
    this.cb = b;
    this.Ia = c;
    this.ub = d;
    this.M = null
};
_.u($K, _.th);
$K.prototype.then = function(a) {
    this.M || (this.M = (new _.Qf(function(a, c) {
        this.P("error", (0, _.s)(function() {
            c(aL(this))
        }, this));
        this.P("success", (0, _.s)(function() {
            a(aL(this))
        }, this));
        this.send(this.mb, this.cb, this.Ia, this.ub)
    }, this)).then(function(a) {
        a.headers = WK(a.headers);
        return a
    }, function(a) {
        return a.status ? (a.headers = WK(a.headers), YK(a)) : YK({
            result: XK,
            body: '{"error":{"code":-1,"message":"A network error occurred and the request could not be completed."}}',
            headers: null,
            status: null,
            statusText: null
        })
    }));
    return this.M.then.apply(this.M, arguments)
};
var aL = function(a) {
    var b, c = a.getStatus(), d = _.Hh(a);
    204 == c ? b=!1 : b = "" == a.Zp() ? (0, _.ac)(d) : _.Ih(a);
    for (var e = {}, f = a.getAllResponseHeaders().split("\r\n"), g = 0; g < f.length; g++)
        if (!_.bh(f[g])) {
            var k;
            k = 2;
            for (var l = f[g].split(": "), m = []; 0 < k && l.length;)
                m.push(l.shift()), k--;
                l.length && m.push(l.join(": "));
                k = m;
                e[k[0]] = e[k[0]] ? e[k[0]] + (", " + k[1]) : k[1]
        }
    var n;
    try {
        n = 2 < _.Fh(a) ? a.A.statusText : ""
    } catch (p) {
        n = ""
    }
    return {
        result: b,
        body: d,
        headers: e,
        status: c,
        statusText: n
    }
}, bL = function(a, b) {
    for (var c = _.Qc(b), d = c ? b : arguments,
    c = c ? 0 : 1; c < d.length && (a = a[d[c]], _.ca(a)); c++);
    return a
}, cL = function(a) {
    var b = a.params || _.Ka();
    a = _.zf(b.headers) || {};
    var c = b.httpMethod || "GET", d = String(b.url || ""), e = b.body || null, b = b.responseType || null, f = _.Lj.Em(a);
    delete a["X-Referer"];
    a = ZK(a);
    a = new $K(d, c, e, a);
    a.pa = f;
    b && (a.D = b);
    return a
}, dL = function(a, b) {
    a.params.url = a.params.path;
    _.Pj.fg("makeHttpRequests", [a], b)
}, eL = {
    "'": "\\'"
}, fL = {
    "\x00": "\\0",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\x0B": "\\x0B",
    '"': '\\"',
    "\\": "\\\\"
}, gL = function(a,
b) {
    var c = function(a) {
        a = _.zf(a);
        delete a.result;
        a = {
            gapiRequest: {
                data: a
            }
        };
        b && b(a, (0, _.$b)(a))
    };
    cL(a).then(c, c)
}, hL = function(a) {
    return new _.Qf(function(b, c) {
        var d = function(a) {
            a && a.gapiRequest ? a = a.gapiRequest.data || a : c(a);
            a = {
                result: 204 != a.status && (0, _.ac)(a.body),
                body: a.body,
                headers: a.headers || null,
                status: a.status || null,
                statusText: a.statusText || null
            };
            _.mh(a.status) ? b(a) : c(a)
        };
        try {
            dL(a, d)
        } catch (e) {
            c(e)
        }
    })
}, iL = function(a) {
    var b;
    b = (a = bL(a, "params", "headers")) && _.Tc(a) ? a : {};
    a = "chrome-extension" == (_.hh(window.location.href)[1] ||
    null);
    b = _.Lj.Em(b);
    var c=!0, d = _.sh(_.qh);
    d && _.ca(d.withCredentials) || (c=!1);
    return !(a && b) && c
}, jL = function(a) {
    _.xf.call(this, a)
};
_.u(jL, _.xf);
jL.prototype.name = "gapi.client.Error";
var kL = function(a, b) {
    _.E("client/cors") && iL(a) ? gL(a, b) : dL(a, b)
}, lL = function(a) {
    if (!a ||!_.Uc(a))
        throw new jL("Must provide a function.");
    this.C = null;
    this.D = a
};
lL.prototype.then = function(a, b, c) {
    this.C || (this.C = this.D());
    return this.C.then(a, b, c)
};
lL.prototype.En = function(a) {
    this.C || (this.C = a)
};
var mL = function(a) {
    (0, window.isFinite)(a) && (a = String(a));
    return _.ja(a) ? /^\s*-?0x/i.test(a) ? (0, window.parseInt)(a, 16) : (0, window.parseInt)(a, 10) : window.NaN
}, nL = function(a) {
    a = String(a);
    if (a.quote)
        return a.quote();
    for (var b = ['"'], c = 0; c < a.length; c++) {
        var d = a.charAt(c), e = d.charCodeAt(0), f = c + 1, g;
        if (!(g = fL[d])) {
            if (!(31 < e && 127 > e))
                if (d in eL)
                    d = eL[d];
                else if (d in fL)
                    d = eL[d] = fL[d];
                else {
                    e = d;
                    g = d.charCodeAt(0);
                    if (31 < g && 127 > g)
                        e = d;
                    else {
                        if (256 > g) {
                            if (e = "\\x", 16 > g || 256 < g)
                                e += "0"
                        } else 
                            e = "\\u", 4096 > g && (e += "0");
                            e += g.toString(16).toUpperCase()
                        }
                        d = eL[d] = e
                }
            g = d
        }
        b[f] = g
    }
    b.push('"');
    return b.join("")
};
new _.ef;
var oL = function(a) {
    this.A = {};
    for (var b = 0; b < a.length; b++) {
        var c = a[b];
        this.A[c.A] = c
    }
}, pL = function(a) {
    a = _.ud(a.A);
    _.yD(a, function(a, c) {
        return a.A - c.A
    });
    return a
};
var qL = function(a, b) {
    this.A = a;
    this.D = b.name;
    this.C=!!b.AT;
    this.B = b.la;
    this.G = b.type
};
var rL = function() {
    this.yc = {};
    this.B = this.Wh().A;
    this.A = null
}, sL = function(a, b, c) {
    c = c || a;
    for (var d in a.yc) {
        var e = Number(d);
        a.B[e] || b.call(c, e, a.yc[d])
    }
};
_.h = rL.prototype;
_.h.Wh = function() {
    var a = this.constructor, b;
    if (!(b = a.Nu)) {
        var c;
        b = a.RD;
        var d = [];
        for (c in b)
            0 != c && d.push(new qL(c, b[c]));
        c = new oL(d);
        b = a.Nu = c
    }
    return b
};
_.h.has = function(a) {
    return null != this.yc[a.A]
};
_.h.get = function(a, b) {
    return tL(this, a.A, b)
};
_.h.set = function(a, b) {
    uL(this, a.A, b)
};
_.h.add = function(a, b) {
    vL(this, a.A, b)
};
_.h.clear = function(a) {
    wL(this, a.A)
};
_.h.equals = function(a) {
    if (!a || this.constructor != a.constructor)
        return !1;
    for (var b = pL(this.Wh()), c = 0; c < b.length; c++) {
        var d = b[c], e = d.A;
        if (null != this.yc[e] != (null != a.yc[e]))
            return !1;
        if (null != this.yc[e]) {
            var f = 11 == d.B || 10 == d.B, g = xL(this, e), e = xL(a, e);
            if (d.C) {
                if (g.length != e.length)
                    return !1;
                for (d = 0; d < g.length; d++) {
                    var k = g[d], l = e[d];
                    if (f?!k.equals(l) : k != l)
                        return !1
                }
            } else if (f?!g.equals(e) : g != e)
                return !1
        }
    }
    return !0
};
var yL = function(a, b) {
    for (var c = pL(a.Wh()), d = 0; d < c.length; d++) {
        var e = c[d], f = e.A;
        if (null != b.yc[f]) {
            a.A && delete a.A[e.A];
            var g = 11 == e.B || 10 == e.B;
            if (e.C)
                for (var e = xL(b, f) || [], k = 0; k < e.length; k++)
                    vL(a, f, g ? e[k].clone() : e[k]);
            else 
                e = xL(b, f), g ? (g = xL(a, f)) ? yL(g, e) : uL(a, f, e.clone()) : uL(a, f, e)
            }
    }
};
rL.prototype.clone = function() {
    var a = new this.constructor;
    a != this && (a.yc = {}, a.A && (a.A = {}), yL(a, this));
    return a
};
var xL = function(a, b) {
    var c = a.yc[b];
    return null == c ? null : c
}, tL = function(a, b, c) {
    var d = xL(a, b);
    return a.B[b].C ? d[c || 0] : d
}, uL = function(a, b, c) {
    a.yc[b] = c;
    a.A && (a.A[b] = c)
}, vL = function(a, b, c) {
    a.yc[b] || (a.yc[b] = []);
    a.yc[b].push(c);
    a.A && delete a.A[b]
}, wL = function(a, b) {
    delete a.yc[b];
    a.A && delete a.A[b]
}, zL = function(a, b) {
    a.RD = b;
    a.Wh = function() {
        return a.Nu || (new a).Wh()
    }
};
var AL = function() {
    rL.apply(this)
};
_.u(AL, rL);
var BL = function() {
    rL.apply(this)
};
_.u(BL, rL);
var CL = function() {
    rL.apply(this)
};
_.u(CL, rL);
var DL = function() {
    rL.apply(this)
};
_.u(DL, rL);
var EL = function() {
    rL.apply(this)
};
_.u(EL, rL);
var FL = function() {
    rL.apply(this)
};
_.u(FL, rL);
var GL = function() {
    rL.apply(this)
};
_.u(GL, rL);
var HL = function() {
    rL.apply(this)
};
_.u(HL, rL);
var IL = function() {
    rL.apply(this)
};
_.u(IL, rL);
var JL = function() {
    rL.apply(this)
};
_.u(JL, rL);
var KL = function() {
    rL.apply(this)
};
_.u(KL, rL);
KL.prototype.Oj = function() {
    return tL(this, 2)
};
KL.prototype.xu = function() {
    wL(this, 2)
};
var LL = function() {
    rL.apply(this)
};
_.u(LL, rL);
LL.prototype.getStatus = function() {
    return tL(this, 4)
};
var ML = function() {
    rL.apply(this)
};
_.u(ML, rL);
var NL = function() {
    rL.apply(this)
};
_.u(NL, rL);
var OL = function() {
    rL.apply(this)
};
_.u(OL, rL);
OL.prototype.Mp = function() {
    return tL(this, 1)
};
var PL = function() {
    rL.apply(this)
};
_.u(PL, rL);
PL.prototype.getStatus = function() {
    return tL(this, 1)
};
zL(AL, {
    0: {
        name: "LcsMessage",
        vc: "buzz.channel.LcsMessage"
    },
    1: {
        name: "session",
        la: 9,
        type: String
    },
    2: {
        name: "message_id",
        la: 9,
        type: String
    },
    3: {
        name: "data",
        la: 9,
        type: String
    }
});
zL(BL, {
    0: {
        name: "Version",
        vc: "buzz.channel.Version"
    },
    1: {
        name: "major_version",
        la: 5,
        type: Number
    },
    2: {
        name: "minor_version",
        la: 5,
        type: Number
    }
});
zL(CL, {
    0: {
        name: "ProtocolVersion",
        vc: "buzz.channel.ProtocolVersion"
    },
    1: {
        name: "version",
        la: 11,
        type: BL
    }
});
zL(DL, {
    0: {
        name: "ClientVersion",
        vc: "buzz.channel.ClientVersion"
    },
    1: {
        name: "version",
        la: 11,
        type: BL
    },
    2: {
        name: "platform",
        la: 9,
        type: String
    },
    3: {
        name: "language",
        la: 9,
        type: String
    },
    4: {
        name: "application_info",
        la: 9,
        type: String
    }
});
zL(EL, {
    0: {
        name: "SessionId",
        vc: "buzz.channel.SessionId"
    },
    1: {
        name: "service_name",
        la: 9,
        type: String
    },
    2: {
        name: "session_name",
        la: 9,
        type: String
    }
});
zL(FL, {
    0: {
        name: "ClientToServerMessage",
        vc: "buzz.channel.ClientToServerMessage"
    },
    1: {
        name: "header",
        la: 11,
        type: GL
    },
    2: {
        name: "init_endpoint_message",
        la: 11,
        type: HL
    },
    3: {
        name: "init_session_message",
        la: 11,
        type: IL
    },
    4: {
        name: "client_data_message_deprecated",
        la: 11,
        type: KL
    },
    5: {
        name: "send_on_disconnect_message",
        la: 11,
        type: KL
    },
    6: {
        name: "close_session_message",
        la: 11,
        type: JL
    }
});
zL(GL, {
    0: {
        name: "ClientHeader",
        vc: "buzz.channel.ClientHeader"
    },
    1: {
        name: "protocol_version",
        la: 11,
        type: CL
    },
    2: {
        name: "client_version",
        la: 11,
        type: DL
    },
    3: {
        name: "client_time_ms",
        la: 3,
        type: Number
    },
    4: {
        name: "max_known_server_time_ms",
        la: 3,
        type: Number
    },
    5: {
        name: "message_id",
        la: 9,
        type: String
    }
});
zL(HL, {
    0: {
        name: "InitEndpointMessage",
        vc: "buzz.channel.InitEndpointMessage"
    }
});
zL(IL, {
    0: {
        name: "InitSessionMessage",
        vc: "buzz.channel.InitSessionMessage"
    },
    1: {
        name: "session_id",
        la: 11,
        type: EL
    }
});
zL(JL, {
    0: {
        name: "CloseSessionMessage",
        vc: "buzz.channel.CloseSessionMessage"
    },
    1: {
        name: "session_id",
        la: 11,
        type: EL
    }
});
zL(KL, {
    0: {
        name: "DataMessage",
        vc: "buzz.channel.DataMessage"
    },
    1: {
        name: "session_id",
        la: 11,
        type: EL
    },
    2: {
        name: "message",
        la: 9,
        type: String
    }
});
zL(LL, {
    0: {
        name: "SessionStatusMessage",
        vc: "buzz.channel.SessionStatusMessage"
    },
    1: {
        name: "session_id",
        la: 11,
        type: EL
    },
    2: {
        name: "address",
        la: 9,
        type: String
    },
    3: {
        name: "is_broadcast_to_user",
        la: 8,
        type: Boolean
    },
    4: {
        name: "status",
        la: 11,
        type: OL
    }
});
zL(ML, {
    0: {
        name: "ServerToClientMessage",
        vc: "buzz.channel.ServerToClientMessage"
    },
    1: {
        name: "header",
        la: 11,
        type: NL
    },
    2: {
        name: "server_data_message",
        la: 11,
        type: KL
    },
    3: {
        name: "channel_status_message",
        la: 11,
        type: PL
    },
    4: {
        name: "session_status_message",
        la: 11,
        type: LL
    }
});
zL(NL, {
    0: {
        name: "ServerHeader",
        vc: "buzz.channel.ServerHeader"
    },
    1: {
        name: "protocol_version",
        la: 11,
        type: CL
    },
    4: {
        name: "server_time_ms",
        la: 3,
        type: Number
    },
    5: {
        name: "message_id",
        la: 9,
        type: String
    }
});
zL(OL, {
    0: {
        name: "StatusP",
        vc: "buzz.channel.StatusP"
    },
    1: {
        name: "code",
        la: 5,
        type: Number
    },
    2: {
        name: "description",
        la: 9,
        type: String
    }
});
zL(PL, {
    0: {
        name: "ChannelStatusMessage",
        vc: "buzz.channel.ChannelStatusMessage"
    },
    1: {
        name: "status",
        la: 11,
        type: OL
    },
    2: {
        name: "jid_resource",
        la: 9,
        type: String
    }
});
var QL = function() {};
var RL = function(a, b) {
    this.A=!!b
};
_.u(RL, QL);
var VL = function(a, b, c) {
    var d = pL(b.Wh());
    (0, _.dd)(d, function(a) {
        if (b.has(a)) {
            var d;
            d = a.A;
            d = b.B[d].C ? null != b.yc[d] ? b.yc[d].length : 0 : null != b.yc[d] ? 1 : 0;
            for (var g = 0; g < d; ++g) {
                c.append(a.D);
                11 == a.B || 10 == a.B ? (c.append(" {"), SL(c), c.A += 2) : c.append(": ");
                TL(this, b.get(a, g), a, c);
                if (11 == a.B || 10 == a.B)
                    c.A -= 2, c.append("}");
                SL(c)
            }
        }
    }, a);
    sL(b, function(a, b) {
        UL(this, a, b, c)
    }, a)
}, UL = function(a, b, c, d) {
    if (null != c)
        if (_.ha(c))(0, _.dd)(c, function(a) {
            UL(this, b, a, d)
        }, a);
    else {
        if (_.Tc(c)) {
            d.append(b);
            d.append(" {");
            SL(d);
            d.A +=
            2;
            if (c instanceof rL)
                VL(a, c, d);
            else 
                for (var e in c) {
                    var f = mL(e);
                    UL(a, f, c[e], d)
                }
                d.A -= 2;
                d.append("}")
            } else 
                _.ja(c) && (c = nL(c)), d.append(b), d.append(": "), d.append(c.toString());
                SL(d)
        }
}, TL = function(a, b, c, d) {
    switch (c.B) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 13:
    case 6:
    case 7:
    case 8:
    case 15:
    case 16:
    case 17:
    case 18:
        d.append(b);
        break;
    case 12:
    case 9:
        b = nL(b.toString());
        d.append(b);
        break;
    case 14:
        if (!a.A) {
            var e=!1;
            _.Kd(c.G, function(a, c) {
                a == b && (d.append(c), e=!0)
            })
        }
        e&&!a.A || d.append(b.toString());
        break;
    case 10:
    case 11:
        VL(a,
        b, d)
    }
}, WL = function() {
    this.A = 0;
    this.B = [];
    this.C=!0
};
WL.prototype.toString = function() {
    return this.B.join("")
};
WL.prototype.append = function(a) {
    if (this.C) {
        for (var b = 0; b < this.A; ++b)
            this.B.push(" ");
        this.C=!1
    }
    this.B.push(a.toString())
};
var SL = function(a) {
    a.B.push("\n");
    a.C=!0
};
var XL = new RL;
rL.prototype.toString = function() {
    var a = new WL;
    VL(XL, this, a);
    return a.toString()
};
var YL = function() {
    rL.apply(this)
};
_.u(YL, rL);
YL.prototype.Bg = function() {
    return tL(this, 1)
};
YL.prototype.yn = function(a) {
    uL(this, 1, a)
};
var ZL = function() {
    rL.apply(this)
};
_.u(ZL, rL);
_.h = ZL.prototype;
_.h.Ba = function() {
    return tL(this, 1)
};
_.h.we = function(a) {
    uL(this, 1, a)
};
_.h.Qv = function() {
    return tL(this, 3)
};
_.h.Gn = function(a) {
    uL(this, 3, a)
};
_.h.Bg = function() {
    return tL(this, 4)
};
_.h.yn = function(a) {
    uL(this, 4, a)
};
zL(YL, {
    0: {
        name: "BrowserChannelConfig",
        vc: "buzz.channel.proto.BrowserChannelConfig"
    },
    1: {
        name: "authuser",
        la: 4,
        type: Number
    },
    2: {
        name: "client_type",
        la: 9,
        type: String
    },
    3: {
        name: "init_delay_ms",
        la: 5,
        type: Number
    },
    4: {
        name: "service_override",
        la: 9,
        type: String
    },
    5: {
        name: "gapi_src",
        la: 9,
        type: String
    },
    6: {
        name: "property_id",
        la: 9,
        type: String
    },
    7: {
        name: "application_version",
        la: 9,
        type: String
    },
    8: {
        name: "retry_on_auth_error",
        la: 8,
        type: Boolean
    },
    9: {
        name: "gsid_servlet_path",
        la: 9,
        type: String
    }
});
zL(ZL, {
    0: {
        name: "AuthenticationParameters",
        vc: "buzz.channel.proto.AuthenticationParameters"
    },
    1: {
        name: "origin",
        la: 9,
        type: String
    },
    2: {
        name: "scheme",
        la: 5,
        type: Number
    },
    3: {
        name: "token",
        la: 9,
        type: String
    },
    4: {
        name: "authuser",
        la: 13,
        type: Number
    }
});
(function() {
    var a;
    return _.Nd ? (a = /Windows NT ([0-9.]+)/, (a = a.exec(_.yd)) ? a[1] : "0") : _.Md ? (a = /10[_.][0-9_.]+/, (a = a.exec(_.yd)) ? a[0].replace(/_/g, ".") : "10") : _.Pd ? (a = /Android\s+([^\);]+)(\)|;)/, (a = a.exec(_.yd)) ? a[1] : "") : _.Qd || _.Rd ? (a = /(?:iPhone|CPU)\s+OS\s+(\S+)/, (a = a.exec(_.yd)) ? a[1].replace(/_/g, ".") : "") : ""
})();
var $L = new CL, aM = new BL;
uL(aM, 1, 3);
uL(aM, 2, 2);
uL($L, 1, aM);
var bM = new BL;
uL(bM, 1, 3);
uL(bM, 2, 2);
var cM = new BL;
uL(cM, 1, 3);
uL(cM, 2, 2);
var dM = function(a) {
    _.te.call(this);
    this.ka = a || _.Or()
};
_.u(dM, _.te);
dM.prototype.aA = 0;
dM.prototype.X = function() {
    return this.aA
};
dM.prototype.Ja = function() {
    return this.ka.Ja()
};
var eM = function(a, b) {
    dM.call(this, b);
    this.A = a;
    this.K = this.A.B().ppu;
    this.va = this.A.B().lpu;
    this.F = []
}, fM, gM;
_.u(eM, dM);
_.h = eM.prototype;
_.h.hI = 5;
_.h.aA = 4;
_.h.jq = 0;
_.h.Yi=!1;
_.h.Ed=!1;
_.h.By = null;
var hM = function(a) {
    return "googlexpc_" + a.A.name + "_msg"
}, iM = function(a) {
    return "googlexpc_" + a.A.name + "_ack"
}, jM = function(a) {
    try {
        if (!a.ob() && a.A.D())
            return a.A.F().frames || {}
    } catch (b) {}
    return {}
};
eM.prototype.Hu = function() {
    if (!this.ob() && this.A.D()) {
        if (!this.Ed) {
            var a = hM(this);
            this.C = kM(this, a);
            this.V = this.Ja().frames[a];
            a = iM(this);
            this.B = kM(this, a);
            this.M = this.Ja().frames[a];
            this.Ed=!0
        }
        if (lM(this, hM(this)) && lM(this, iM(this)))
            this.ya = new mM(this, jM(this)[hM(this)], (0, _.s)(this.pa, this)), this.Y = new mM(this, jM(this)[iM(this)], (0, _.s)(this.wa, this)), this.U();
        else {
            if (1 == this.A.C()) {
                if (!(this.By || 0 < this.hI--)) {
                    for (var b = 10, a = ""; 0 < b--;)
                        a += "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".charAt(Math.floor(62 *
                        Math.random()));
                    this.A.V(a);
                    nM(this);
                    this.Ed=!1;
                    this.By = kM(this, "googlexpc_reconnect_" + this.A.name)
                }
            } else if (0 == this.A.C())
                for (var a = jM(this), c = a.length, d = 0; d < c; d++) {
                    try {
                        a[d] && a[d].name && (b = a[d].name)
                    } catch (e) {}
                    if (b) {
                        var f = b.split("_");
                        if (3 == f.length && "googlexpc" == f[0] && "reconnect" == f[1]) {
                            this.A.name = f[2];
                            nM(this);
                            this.Ed=!1;
                            break
                        }
                    }
            }
            this.Ja().setTimeout((0, _.s)(this.Hu, this), 100)
        }
    }
};
var kM = function(a, b) {
    var c = window.document.createElement("iframe"), d = c.style;
    d.position = "absolute";
    d.top = "-10px";
    d.left = "10px";
    d.width = "1px";
    d.height = "1px";
    c.id = c.name = b;
    c.src = a.K + "#INITIAL";
    a.Ja().document.body.appendChild(c);
    return c
}, nM = function(a) {
    a.C && (a.C.parentNode.removeChild(a.C), a.C = null, a.V = null);
    a.B && (a.B.parentNode.removeChild(a.B), a.B = null, a.M = null)
}, lM = function(a, b) {
    try {
        var c = jM(a)[b];
        if (!c || 0 != c.location.href.indexOf(a.va))
            return !1
        } catch (d) {
        return !1
    }
    return !0
};
eM.prototype.U = function() {
    var a = jM(this);
    a[iM(this)] && a[hM(this)] ? (this.da = new oM(this.K, this.V), this.G = new oM(this.K, this.M), this.Ja().setTimeout((0, _.s)(function() {
        this.da.send("SETUP");
        this.Yi=!0
    }, this), 100)) : (this.R || (this.R = (0, _.s)(this.U, this)), this.Ja().setTimeout(this.R, 100))
};
var pM = function(a) {
    if (a.ia && a.Ia && (a.A.M(), a.D)) {
        for (var b = 0, c; b < a.D.length; b++)
            c = a.D[b], a.A.G(c.service, c.$H);
        delete a.D
    }
};
eM.prototype.pa = function(a) {
    if ("SETUP" == a)
        this.G && (this.G.send("SETUP_ACK"), this.ia=!0, pM(this));
    else if (this.A.A() || this.ia) {
        var b = a.indexOf("|"), c = a.substring(0, b);
        a = a.substring(b + 1);
        b = c.indexOf(",");
        if ( - 1 == b) {
            var d;
            this.G.send("ACK:" + c);
            qM(this, a)
        } else 
            d = c.substring(0, b), this.G.send("ACK:" + d), c = c.substring(b + 1).split("/"), b = (0, window.parseInt)(c[0], 10), c = (0, window.parseInt)(c[1], 10), 1 == b && (this.N = []), this.N.push(a), b == c && (qM(this, this.N.join("")), delete this.N)
    }
};
eM.prototype.wa = function(a) {
    "SETUP_ACK" == a ? (this.Yi=!1, this.Ia=!0, pM(this)) : this.A.A() && this.Yi && (0, window.parseInt)(a.split(":")[1], 10) == this.jq && (this.Yi=!1, rM(this))
};
var rM = function(a) {
    if (!a.Yi && a.F.length) {
        var b = a.F.shift();
        ++a.jq;
        a.da.send(a.jq + b);
        a.Yi=!0
    }
}, qM = function(a, b) {
    var c = b.indexOf(":"), d = b.substr(0, c), c = b.substring(c + 1);
    a.A.A() ? a.A.G(d, c) : (a.D || (a.D = [])).push({
        service: d,
        $H: c
    })
};
eM.prototype.send = function(a, b) {
    var c = a + ":" + b;
    if (!_.M || 3800 >= b.length)
        this.F.push("|" + c);
    else 
        for (var d = b.length, e = Math.ceil(d / 3800), f = 0, g = 1; f < d;)
            this.F.push("," + g + "/" + e + "|" + c.substr(f, 3800)), g++, f += 3800;
    rM(this)
};
eM.prototype.W = function() {
    eM.J.W.call(this);
    var a = sM;
    _.se(a, this.ya);
    _.se(a, this.Y);
    this.ya = this.Y = null;
    _.Ag(this.C);
    _.Ag(this.B);
    this.V = this.M = this.C = this.B = null
};
var sM = [], tM = (0, _.s)(function() {
    var a = sM, b, c=!1;
    try {
        for (var d = 0; b = a[d]; d++)
            c = c || b.qn()
    } catch (e) {
        if (b.D.A.K(), !a.length)
            return 
    }
    a = (0, _.sa)();
    c && (fM = a);
    gM = window.setTimeout(tM, 1E3 > a - fM ? 10 : 100)
}, eM), uM = function() {
    fM = (0, _.sa)();
    gM && window.clearTimeout(gM);
    gM = window.setTimeout(tM, 10)
}, oM = function(a, b) {
    this.C = a;
    this.B = b;
    this.A = 0
};
oM.prototype.send = function(a) {
    this.A=++this.A%2;
    a = this.C + "#" + this.A + (0, window.encodeURIComponent)(a);
    try {
        _.Ud ? this.B.location.href = a : this.B.location.replace(a)
    } catch (b) {}
    uM()
};
var mM = function(a, b, c) {
    this.D = a;
    this.B = b;
    this.C = c;
    this.A = this.B.location.href.split("#")[0] + "#INITIAL";
    sM.push(this);
    uM()
};
mM.prototype.qn = function() {
    var a = this.B.location.href;
    if (a != this.A) {
        this.A = a;
        if (a = a.split("#")[1])
            a = a.substr(1), this.C((0, window.decodeURIComponent)(a));
        return !0
    }
    return !1
};
var vM = function(a) {
    this.C = a;
    this.D=!1;
    this.promise = {
        then: (0, _.s)(function(a, c, d) {
            this.D || (this.D=!0);
            this.B&&!this.A ? this.C.resolve(this.B) : this.A&&!this.B && this.C.reject(this.A);
            return this.C.promise.then(a, c, d)
        }, this)
    }
};
vM.prototype.resolve = function(a) {
    this.D ? this.C.resolve(a) : this.B || this.A || (this.B = a)
};
vM.prototype.reject = function(a) {
    this.D ? this.C.reject(a) : this.B || this.A || (this.A = a)
};
var wM = function(a) {
    lL.call(this, wM.prototype.G);
    if (!a || "object" != typeof a && "string" != typeof a)
        throw new jL("Invalid request parameters");
    var b;
    _.ja(a) ? b = {
        path: a
    } : b = a;
    if (!b.path)
        throw new jL('Missing required parameter: "path"');
    this.A = {
        path: b.path,
        method: b.method || "GET",
        params: b.params || {},
        headers: b.headers || {},
        body: b.body,
        root: b.root,
        responseType: b.responseType
    };
    this.B = b.XR || "auto"
};
_.u(wM, lL);
wM.prototype.ci = function() {
    return this.A
};
wM.prototype.wk = function() {
    var a = this.A, b = a.headers || {}, c = b, d = a.applicationName || null, e = a.clientName || null, f = a.clientVersion || null, f = f || "0.1", e = e || "google-api-javascript-client", g = [];
    d && (g.push(d), g.push(" "));
    g.push(e);
    f && (g.push("/"), g.push(f));
    c["X-JavaScript-User-Agent"] = g.join("");
    c = b["Content-Type"];
    a.body&&!c && (c = "application/json", b["Content-Type"] = c);
    "application/json" != c || a.params.alt || (a.params.alt = "json");
    (c = a.body || null) && _.Tc(c) && (a.body = (0, _.$b)(c));
    a.key = a.id;
    b = _.Pj.fl(b, void 0, this.B);
    c = _.Lj.Em(b);
    xM(b);
    (b = _.Sj.key)&&!a.params.key && (a.params.key = b);
    b = a.params;
    d = _.ik(a.path);
    e = String(_.E("googleapis.config/xd3") || "");
    18 <= e.length && "/static/proxy.html" == e.substring(e.length - 18) && (e = e.substring(0, e.length - 18));
    e || (e = "/");
    f = _.ik(e);
    if (e != f)
        throw Error("G");
    "/" != e.charAt(e.length - 1) && (e += "/");
    d = _.lh(e, d);
    _.eh(d, "/") && (d = d.substring(0, d.length - 1));
    d = _.ih(_.kh([d], b));
    a.path = d;
    b = a.path;
    c = (d = a.root) ? d : String(_.E(c ? "googleapis.config/root-1p" : "googleapis.config/root") || "");
    b = _.lh(c, b);
    a.url = b
};
var AM = function(a) {
    a.wk();
    a = a.A;
    return {
        key: "gapiRequest",
        params: {
            id: a.id,
            key: a.key,
            url: a.url,
            path: a.path,
            httpMethod: a.method,
            body: a.body || "",
            headers: a.headers || {},
            urlParams: {},
            root: a.root,
            clientName: yM(),
            clientVersion: zM()
        }
    }
};
wM.prototype.execute = function(a) {
    this.A.headers.Authorization || _.Bj(void 0, void 0);
    var b = AM(this);
    kL(b, function(b, d) {
        var e = b;
        b.gapiRequest && (e = b.gapiRequest);
        e && e.data && (e = e.data);
        var e = e instanceof Array ? e[0]: e, f;
        if (204 != e.status && e.body)
            try {
                f = (0, _.ac)(e.body)
        } catch (g) {}
        _.E("client/jsonpOverride") && (f = e);
        a && a(f, d)
    })
};
wM.prototype.G = function() {
    var a = AM(this);
    return _.E("client/cors") && iL(a) ? cL(a) : hL(a)
};
var BM = /<response-(.*)>/, CM = /^application\/http(;.+$|$)/, DM = function(a, b) {
    if (!a ||!b ||!_.Ma(a, b = b.toLowerCase()))
        throw new jL("Unable to retrieve header.");
    return a[b]
}, GM = function(a, b) {
    for (var c = "batch" + String(Math.round(2147483647 * (0, _.Nj)())) + String(Math.round(2147483647 * (0, _.Nj)())), d = {
        path: "/batch",
        headers: {
            "Content-Type": "multipart/mixed; boundary=" + c
        },
        method: "POST"
    }, c = "--" + c, e = [], f = 0; f < a.length; f++)
        e.push(EM(a[f].request, a[f].id));
    d.body = [c, e.join("\r\n" + c + "\r\n"), c + "--"].join("\r\n") + "\r\n";
    d.root = b || null;
    return FM(d)
}, EM = function(a, b) {
    var c = [], d = a.ci(), e = function(a, b) {
        _.ib(a, function(a, c) {
            b.push(c + ": " + a)
        })
    };
    e({
        "Content-Type": "application/http",
        "Content-Transfer-Encoding": "binary",
        "Content-ID": "<" + b + ">"
    }, c);
    c.push("");
    c.push(d.method + " " + d.path);
    e(d.headers, c);
    c.push("");
    d.body && c.push(d.body);
    return c.join("\r\n")
}, IM = function(a, b) {
    var c = HM(a, b), d = {};
    _.Kd(c, function(a, b) {
        d[b] = {
            result: a.result || a.body,
            rawResult: (0, _.$b)({
                id: b,
                result: a.result || a.body
            }),
            id: b
        }
    });
    return d
}, HM = function(a,
b) {
    a = (0, _.Xc)(a);
    _.eh(a, "--") && (a = a.substring(0, a.length - 2));
    for (var c = a.split(b), d = _.Ka(), e = 0; e < c.length; e++)
        if (c[e]) {
            var f;
            if (f = c[e]) {
                _.eh(f, "\r\n") && (f = f.substring(0, f.length - 2));
                f = JM(f);
                var g = _.Ka(), k = DM(f.outerHeaders, "Content-Type");
                if (null == CM.exec(k))
                    throw new jL("Unexpected Content-Type");
                    k = DM(f.outerHeaders, "Content-Id");
                    k = BM.exec(k);
                    if (!k)
                        throw new jL("Unable to recognize Content-Id.");
                        g.id = k[1];
                        k = (0, _.ac)(f.body);
                        g.response = {
                            result: k,
                            body: f.body,
                            status: f.status,
                            statusText: f.statusText,
                            headers: f.headers
                        };
                        f = g
            } else 
                f = null;
                f && f.id && (d[f.id] = f.response)
        }
    return d
}, JM = function(a) {
    if (!a)
        return null;
    a = a.split("\r\n");
    for (var b = 0, c = {
        headers: {},
        body: ""
    }, d = function(a) {
        var b = 0;
        _.ib(a, function() {
            b++
        });
        return b
    }; b < a.length && "" == a[b];)
        b++;
    c.outerHeaders = KM(a, b);
    var b = b + (d(c.outerHeaders) + 1), e = a[b++].split(" ");
    c.status = Number(e[1]);
    c.statusText = e[2];
    c.headers = KM(a, b);
    b += d(c.headers) + 1;
    c.body = a.slice(b).join("\r\n");
    return c
}, KM = function(a, b) {
    for (var c = _.Ka(), d = b; d < a.length && a[d]; d++) {
        var e = a[d].split(": "), f = (0, _.Xc)(e[0].toLowerCase()), e = (0, _.Xc)(e[1]);
        c[f] = e
    }
    return c
};
var LM = function(a) {
    lL.call(this, LM.prototype.G);
    this.A = [];
    this.B = a;
    this.F=!!a
};
_.u(LM, lL);
var MM = function(a, b) {
    for (var c = 0; c < a.A.length; c++)
        if (a.A[c].id == b)
            return !0;
    return !1
};
LM.prototype.add = function(a, b) {
    var c = b || _.Ka(), d = _.Ka();
    if (a)
        d.request = a;
    else 
        throw new jL("Batch entry " + (_.Ma(c, "id") ? '"' + c.id + '" ' : "") + "is missing a request method");
    var e;
    e = _.pn();
    e = new vM(e);
    d.vn = e;
    a.En(d.vn.promise);
    e = a.ci().root;
    if (!this.F) {
        if (e && this.B && e != this.B)
            throw new jL('The "root" provided in this request is not consistent with that of existing requests in the batch.');
        this.B = e || this.B
    }
    if (_.Ma(c, "id")) {
        e = c.id;
        if (MM(this, e))
            throw new jL('Batch ID "' + e + '" already in use, please use another.');
        d.id = e
    } else {
        do 
            d.id = String(Math.round(2147483647 * (0, _.Nj)()));
        while (MM(this, d.id))
        }
    d.nc = c.callback;
    this.A.push(d);
    return d.id
};
LM.prototype.execute = function(a) {
    1 > this.A.length || (_.hb(this.A, function(a) {
        a.request.wk()
    }), a = NM(this, a), GM(this.A, this.B).execute(a))
};
LM.prototype.G = function() {
    if (1 > this.A.length)
        return new _.Qf(function(a) {
            a({})
        });
    _.hb(this.A, function(a) {
        a.request.wk()
    });
    var a = GM(this.A, this.B), b = (0, _.s)(function(a) {
        OM(a);
        if (0 != DM(a.headers, "Content-Type").indexOf("multipart/mixed"))
            throw new jL("The response's Content-Type is not multipart/mixed.");
        var b = DM(a.headers, "Content-Type").split("boundary=")[1];
        if (!b)
            throw new jL("Boundary not indicated in response.");
        b = HM(a.body, "--" + b);
        a.result = b || {};
        for (var c = 0; c < this.A.length; c++) {
            var g = this.A[c],
            k = b[g.id];
            _.mh(k.status) ? g.vn.resolve(k) : g.vn.reject(k)
        }
        return a
    }, this), c = (0, _.s)(function(a) {
        for (var b = 0; b < this.A.length; b++)
            this.A[b].vn.reject({
                result: {
                    error: {
                        code: 0,
                        message: "The batch request could not be fulfilled."
                    }
                },
                body: '{"error":{"code":0,"message":"The batch request could not befulfilled"}}',
                headers: null,
                status: null,
                statusText: null
            });
        throw a;
    }, this);
    return a.then(b, c)
};
var NM = function(a, b) {
    return (0, _.s)(function(a, d) {
        var e;
        e = a;
        if (d&&!e) {
            var f = (0, _.ac)(d);
            f && (f.gapiRequest ? e = f.gapiRequest.data : e = f)
        }
        OM(e);
        if (e) {
            if (0 != DM(e.headers, "Content-Type").indexOf("multipart/mixed"))
                throw new jL("The response's Content-Type is not multipart/mixed.");
            if (200 <= e.status && 299 >= e.status) {
                var g = DM(e.headers, "Content-Type").split("boundary=")[1];
                if (!g)
                    throw new jL("Boundary not indicated in response.");
                g = IM(e.body, "--" + g);
                for (e = 0; e < this.A.length; e++)
                    if (f = this.A[e], delete g[f.id].rawResult, f.nc) {
                        var k = (0, _.$b)(g[f.id]);
                        f.nc(g[f.id] ||!1, k)
                    }
            }
            b && b(g || null, d)
        }
    }, a)
}, OM = function(a) {
    if (a && a.headers) {
        var b = _.Ka(), c;
        for (c in a.headers)
            b[c.toLowerCase()] = a.headers[c];
        a.headers = b
    }
};
var PM = ["appVersion", "platform", "userAgent"], QM = {
    "google-api-javascript-client": !0,
    "google-api-gwt-client": !0
}, RM = {}, SM = function(a, b) {
    var c;
    return c = "/discovery/v1/apis/" + ((0, window.encodeURIComponent)(a) + "/" + (0, window.encodeURIComponent)(b) + "/rpc?fields=methods%2f*%2fid&pp=0")
}, TM = function(a, b) {
    var c;
    return c = "/discovery/v1/apis/" + ((0, window.encodeURIComponent)(a) + "/" + (0, window.encodeURIComponent)(b) + "/rest?fields=servicePath%2Cresources%2Cparameters%2Cmethods&pp=0")
}, xM = function(a) {
    if (a && window.navigator) {
        for (var b =
        [], c = 0; c < PM.length; c++) {
            var d = window.navigator[PM[c]];
            d && b.push((0, window.encodeURIComponent)(PM[c]) + "=" + (0, window.encodeURIComponent)(d))
        }
        a["X-ClientDetails"] = b.join("&")
    }
}, yM = function() {
    var a = _.E("client/name", "google-api-javascript-client");
    return a in QM ? a : "google-api-javascript-client"
}, zM = function() {
    return String(_.E("client/version", "1.1.0-beta"))
}, WM = function(a, b, c, d, e) {
    if (!a ||!b)
        throw new jL("Missing required parameters.");
    var f = c || function() {};
    FM({
        path: SM(a, b),
        callback: function(a) {
            if (a.error)
                f(a);
            else {
                for (var c in a)
                    if (a.hasOwnProperty(c) && "methods" == c) {
                        var l = a[c], m;
                        for (m in l)
                            if (l.hasOwnProperty(m)) {
                                UM(m, {
                                    root: d
                                }, e);
                                var n = {};
                                n[m] = b;
                                VM(n)
                            }
                    }
                f.call(null)
            }
        },
        root: d
    })
}, YM = function(a, b, c, d) {
    function e(a) {
        function e(a) {
            _.ib(a, function(a) {
                if (x) {
                    if ("GET" == a.httpMethod && a.request && f) {
                        var e = _.Tc(d) ? bL(d, "root"): void 0, e = _.ja(e) ? e : void 0;
                        WM(f, b, c, e, !0);
                        x=!1
                    }
                    XM(p, a, v, d)
                }
            })
        }
        function g(a) {
            _.ib(a, function(a) {
                e(a.methods);
                (a = a.resources) && g(a)
            })
        }
        if (a.error)
            k(a);
        else {
            var p = a.servicePath, v = a.parameters, x=!0,
            t = a.resources;
            t && g(t);
            (a = a.methods) && e(a);
            x && k.call()
        }
    }
    var f, g;
    _.Tc(a) ? (g = a, f = a.name, b = a.version) : f = a;
    if (!f ||!b)
        throw new jL("Missing required parameters.");
    var k = c || function() {};
    g ? e(g) : FM({
        path: TM(f, b),
        callback: e,
        root: d && d.root
    })
}, FM = function(a, b) {
    var c;
    "object" == typeof a && (c = a.callback, delete a.callback);
    var d;
    ZM() ? d = new $M(a, b) : d = new wM(a);
    return c ? (d.execute(c), null) : d
}, aN = function(a, b) {
    b = _.zf(b);
    var c = function(a, b, c) {
        var d;
        if (a.hasOwnProperty(b))
            d = a[b], delete a[b];
        else if (c)
            throw new jL("Missing required parameters.");
        return d
    }, d = c(b, "servicePath", !0), e = String(c(b, "restPath", !0)), f = String(c(b, "httpMethod", !0)), g = c(b, "parameters");
    null != g ? g = _.zf(g) : g = null;
    var k = c(b, "parameterName");
    _.ja(k) || (k = null);
    c = c(b, "supportsSubscription");
    if (!e ||!f)
        throw new jL("Missing required parameters.");
    e = {
        id: a,
        path: e,
        parameters: g,
        httpMethod: f,
        request: {
            parameterName: k
        }
    };
    c && (e.supportsSubscription = c);
    XM(d, e, {
        path: {},
        query: {}
    }, b)
}, UM = function(a, b, c) {
    if (!a)
        throw new jL("Missing required parameters");
    for (var d = a.split("."), e = window.gapi.client,
    f = 0; f < d.length - 1; f++) {
        var g = d[f];
        e[g] = e[g] || {};
        e = e[g]
    }
    d = d[d.length - 1];
    if (!e[d] || c) {
        var k = b || {};
        e[d] = function(b) {
            var c;
            "string" == typeof k ? c = k : c = k.root;
            b && b.root && (c = b.root);
            b = {
                method: a,
                apiVersion: k.apiVersion,
                rpcParams: b,
                transport: {
                    name: "googleapis",
                    root: c
                }
            };
            return ZM() ? new $M(b, 2) : new bN(b)
        }
    }
}, XM = function(a, b, c, d) {
    if (!(b && b.id && b.path && b.httpMethod))
        throw new jL("Missing required parameters");
    var e = b.id.split("."), f = window.gapi.client, g;
    for (g = 0; g < e.length - 1; g++) {
        var k = e[g];
        f[k] = f[k] || {};
        f = f[k]
    }
    k =
    window.gapi.client[e[0]];
    k.Ks || (k.Ks = {
        servicePath: a || "",
        parameters: c
    });
    a = e[g];
    if (!f[a]) {
        if (d && d.hasOwnProperty("root"))
            var l = d.root;
        f[a] = _.ra(cN, {
            path: _.ja(b.path) ? b.path: null,
            httpMethod: _.ja(b.httpMethod) ? b.httpMethod: null,
            parameters: b.parameters,
            parameterName: (b.request || {}).parameterName || "",
            request: b.request,
            root: l
        }, k.Ks)
    }
}, cN = function(a, b, c, d) {
    var e = b.servicePath || "";
    _.Wc(e, "/") || (e = "/" + e);
    var f = dN(a.path, [a.parameters, b.parameters], c || {});
    b = f.$f;
    c = f.fK;
    e = _.lh(e, f.path);
    f = c.root;
    delete c.root;
    var g = a.parameterName;
    !g && 1 == _.qf(c) && c.hasOwnProperty("resource") && (g = "resource");
    var k;
    g && (k = c[g], delete c[g]);
    null != k || (k = d);
    null == k && a.request && (_.Xg(c) && (c = void 0), k = c);
    d = "complete" == _.E("client/rms") ? 0 : 1;
    return FM({
        path: e,
        method: a.httpMethod,
        params: b,
        body: k,
        root: f || a.root
    }, d) || null
}, dN = function(a, b, c) {
    c = _.zf(c);
    var d = {};
    _.hb(b, function(b) {
        _.ib(b, function(b, e) {
            var k = b.required;
            if ("path" == b.location)
                if (Object.prototype.hasOwnProperty.call(c, e))
                    _.kd(a, "{" + e + "}") ? (k = (0, window.encodeURIComponent)(String(c[e])),
                    a = a.replace("{" + e + "}", k)) : _.kd(a, "{+" + e + "}") && (k = (0, window.encodeURI)(String(c[e])), a = a.replace("{+" + e + "}", k)), delete c[e];
                else {
                    if (k)
                        throw new jL("Required path parameter " + e + " is missing.");
                } else 
                    "query" == b.location && Object.prototype.hasOwnProperty.call(c, e) && (d[e] = c[e], delete c[e])
        })
    });
    if (b = c.trace)
        d.trace = b, delete c.trace;
    return {
        path: a,
        $f: d,
        fK: c
    }
}, fN = function(a) {
    return ZM() ? new eN(a, 0) : new LM(a)
}, VM = function(a) {
    _.Wj(a);
    for (var b in a)
        Object.prototype.hasOwnProperty.call(a, b) && (RM[b] = a[b])
};
_.Xj(null);
var gN = function(a) {
    var b = ["complete"], c = _.E("client/rms");
    if (_.id(["rpcDeprecated", "migrated"], c))
        _.Qb(a + " is deprecated. See https://developers.google.com/api-client-library/javascript/reference/referencedocs");
    else if (_.id(b, c))
        throw new jL(a + " is discontinued.");
}, ZM = function() {
    return "migrated" == _.E("client/rms")
};
var hN = function() {
    this.A = [];
    this.C = this.B = null
};
hN.prototype.add = function(a, b) {
    var c = b || {}, d = {}, e = Object.prototype.hasOwnProperty;
    if (a)
        d.jh = a;
    else 
        throw new jL("Batch entry " + (e.call(c, "id") ? '"' + c.id + '" ' : "") + "is missing a request method");
    if (e.call(c, "id")) {
        for (var e = c.id, f = 0; f < this.A.length; f++)
            if (this.A[f].id == e)
                throw new jL('Batch ID "' + e + '" already in use, please use another.');
        d.id = e
    } else {
        do 
            d.id = String(2147483647 * (0, _.Nj)() | 0);
        while (e.call(this.A, d.id))
        }
    d.nc = c.callback;
    this.A.push(d)
};
hN.prototype.execute = function(a) {
    this.B = [];
    for (var b, c, d = 0; d < this.A.length; d++)
        b = this.A[d], c = b.jh, this.B.push(c.Wm(b.id)), this.C = c.gm() || this.C;
    b = {};
    b = _.Pj.fl(b);
    xM(b);
    b = {
        requests: this.B,
        headers: b,
        root: this.C,
        clientName: yM(),
        clientVersion: zM(),
        urlParams: {
            key: _.Sj.key
        }
    };
    _.Pj.fg("makeRequest", b, iN(this, a))
};
var iN = function(a, b) {
    return function(c, d) {
        for (var e = 0; e < a.A.length; e++) {
            var f = a.A[e];
            f.nc && f.nc(c[f.id] ||!1, d)
        }
        b && b(c, d)
    }
};
var bN = function(a) {
    if (!a || "object" != typeof a)
        throw new jL("Missing rpc parameters");
    if (!a.method)
        throw new jL("Missing rpc method");
    this.A = a
};
bN.prototype.gm = function() {
    var a = this.A.transport;
    return a ? a.root || null : null
};
bN.prototype.execute = function(a) {
    var b = new hN;
    b.add(this, {
        id: "gapiRpc",
        callback: jN(a)
    });
    b.execute()
};
bN.prototype.Wm = function(a) {
    var b = this.A.method;
    a = {
        jsonrpc: "2.0",
        id: a,
        method: b,
        apiVersion: String(this.A.apiVersion || RM[b] || "v1")
    };
    (b = this.A.rpcParams) && (a.params = b);
    return a
};
var jN = function(a) {
    return function(b, c) {
        var d;
        b ? b.error ? (d = b.error, null == d.error && (d.error = _.zf(b.error))) : (d = b.result || b.data, _.Tc(d) && null == d.result && (d.result = _.zf(b.result || b.data))) : d=!1;
        a(d, c)
    }
};
var $M = function(a, b) {
    this.B = b || 0;
    if (2 == this.B) {
        var c = null;
        null != a && _.Tc(a) && (c = {
            method: a.method,
            rpcParams: a.rpcParams,
            transport: a.transport,
            root: a.root,
            apiVersion: a.apiVersion
        });
        this.A = new bN(c)
    } else 
        0 == this.B && (c = a && a.callback) && (a.callback = kN(c)), c = null, null != a && (_.Tc(a) ? c = {
            path: a.path,
            method: a.method,
            params: a.params,
            headers: a.headers,
            body: a.body,
            root: a.root,
            responseType: a.responseType
        } : _.ja(a) && (c = a)), this.A = new wM(c)
}, kN = function(a) {
    return function(b) {
        var c;
        null != b && _.Tc(b) && b.error ? (c = lN(b),
        b = (0, _.$b)([{
            id: "gapiRpc",
            error: c
        }
        ]), c.error = _.xD(c)) : (null != b || (b = {}), c = _.xD(b), c.result = _.xD(b), b = (0, _.$b)([{
            id: "gapiRpc",
            result: b
        }
        ]));
        a(c, b)
    }
}, lN = function(a) {
    a = _.xD(a.error);
    return {
        code: a.code,
        data: a.errors,
        message: a.message
    }
};
_.h = $M.prototype;
_.h.execute = function(a) {
    var b;
    a && 1 == this.B ? b = kN(a) : b = a;
    this.A.execute(b)
};
_.h.then = function(a, b) {
    2 == this.B && mN('The "then" method is not available on this object.');
    return this.A.then(a, b)
};
_.h.En = function(a) {
    this.A.En && this.A.En(a)
};
_.h.ci = function() {
    if (this.A.ci)
        return this.A.ci()
};
_.h.wk = function() {
    this.A.ci && this.A.wk()
};
_.h.gm = function() {
    if (this.A.gm)
        return this.A.gm()
};
_.h.Wm = function(a) {
    if (this.A.Wm)
        return this.A.Wm(a)
};
var eN = function(a, b) {
    this.C = a;
    this.A = b || null;
    this.B = null
};
eN.prototype.add = function(a, b) {
    null === this.A && (this.A = a.B, 2 == this.A ? this.B = new hN : this.B = new LM(this.C));
    this.A !== a.B && mN("Unable to add item to batch.");
    var c = b && b.callback;
    1 == this.A && c && (b.callback = function(a) {
        a = nN(a);
        var b = (0, _.$b)([a]);
        c(a, b)
    });
    this.B.add(a, b)
};
eN.prototype.execute = function(a) {
    var b;
    a && 1 == this.A ? b = function(b) {
        var d = [];
        _.ib(b, function(a, e) {
            a = nN(a);
            b[e] = a;
            d.push(a)
        });
        var e = (0, _.$b)(d);
        a(b, e)
    } : b = a;
    this.B && this.B.execute(b)
};
var nN = function(a) {
    var b = a ? bL(a, "result"): null;
    _.Tc(b) && null != b.error && (b = lN(b), a = {
        id: a.id,
        error: b
    });
    return a
};
eN.prototype.then = function(a, b) {
    2 == this.A && mN('The "then" method is not available on this object.');
    return this.B.then(a, b)
};
var mN = function(a) {
    throw Error("ua`" + a);
};
_.C("gapi.client.load", function(a, b, c, d) {
    function e(c) {
        "migrated" == f || "complete" == f ? YM(a, b, c, {
            root: d
        }) : _.ja(a) && WM(a, b, c, _.ja(d) ? d : void 0)
    }
    var f = _.E("client/rms");
    if (c)
        e(c);
    else 
        return new _.Qf(function(a, b) {
            try {
                e(a)
            } catch (c) {
                b(c)
            }
        })
});
_.C("gapi.client.newBatch", function(a) {
    return ZM() ? new eN(a) : "complete" == _.E("client/rms") ? fN(a) : new hN
});
_.C("gapi.client.newRpcBatch", function() {
    gN("gapi.client.newRpcBatch");
    return ZM() ? new eN : new hN
});
_.C("gapi.client.newHttpBatch", fN);
_.C("gapi.client.register", function(a, b) {
    if ("complete" == _.E("client/rms"))
        aN(a, b);
    else {
        var c;
        b && (c = {
            apiVersion: b.apiVersion,
            root: b.root
        });
        UM(a, c)
    }
});
_.C("gapi.client.request", FM);
_.C("gapi.client.rpcRequest", function(a, b, c) {
    gN("gapi.client.rpcRequest");
    if (!a)
        throw new jL('Missing required parameter "method".');
    a = {
        method: a,
        apiVersion: b,
        rpcParams: c,
        transport: {
            name: "googleapis",
            root: c && c.root || ""
        }
    };
    return ZM() ? new $M(a, 2) : new bN(a)
});
_.C("gapi.client.setApiKey", _.Xj);
_.C("gapi.client.setApiVersions", VM);
_.C("gapi.client.AuthType", {
    KK: "auto",
    NONE: "none",
    JO: "oauth2",
    JM: "1p"
});
_.C("gapi.client.AuthType.AUTO", "auto");
_.C("gapi.client.AuthType.NONE", "none");
_.C("gapi.client.AuthType.OAUTH2", "oauth2");
_.C("gapi.client.AuthType.FIRST_PARTY", "1p");

});
// Google Inc.

