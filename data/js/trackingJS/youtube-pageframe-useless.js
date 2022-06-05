(function() {
    var g, aa = aa || {}, l = this;
    function n(a) {
        return void 0 !== a
    }
    function q(a, b, c) {
        a = a.split(".");
        c = c || l;
        a[0]in c ||!c.execScript || c.execScript("var " + a[0]);
        for (var d; a.length && (d = a.shift());)
            !a.length && n(b) ? c[d] = b : c[d] ? c = c[d] : c = c[d] = {}
    }
    function r(a, b) {
        for (var c = a.split("."), d = b || l, e; e = c.shift();)
            if (null != d[e])
                d = d[e];
            else 
                return null;
        return d
    }
    function s() {}
    function ba(a) {
        a.getInstance = function() {
            return a.ha ? a.ha : a.ha = new a
        }
    }
    function ca(a) {
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
}
function da(a) {
    return "array" == ca(a)
}
function ea(a) {
    var b = ca(a);
    return "array" == b || "object" == b && "number" == typeof a.length
}
function t(a) {
    return "string" == typeof a
}
function fa(a) {
    return "function" == ca(a)
}
function ga(a) {
    var b = typeof a;
    return "object" == b && null != a || "function" == b
}
function ha(a) {
    return a[ia] || (a[ia]=++ja)
}
var ia = "closure_uid_" + (1E9 * Math.random()>>>0), ja = 0;
function ka(a, b, c) {
    return a.call.apply(a.bind, arguments)
}
function la(a, b, c) {
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
}
function u(a, b, c) {
    u = Function.prototype.bind&&-1 != Function.prototype.bind.toString().indexOf("native code") ? ka : la;
    return u.apply(null, arguments)
}
function v(a, b) {
    var c = Array.prototype.slice.call(arguments, 1);
    return function() {
        var b = c.slice();
        b.push.apply(b, arguments);
        return a.apply(this, b)
    }
}
var w = Date.now || function() {
    return + new Date
};
function y(a, b) {
    function c() {}
    c.prototype = b.prototype;
    a.J = b.prototype;
    a.prototype = new c;
    a.base = function(a, c, f) {
        return b.prototype[c].apply(a, Array.prototype.slice.call(arguments, 2))
    }
}
Function.prototype.bind = Function.prototype.bind || function(a, b) {
    if (1 < arguments.length) {
        var c = Array.prototype.slice.call(arguments, 1);
        c.unshift(this, a);
        return u.apply(null, c)
    }
    return u(this, a)
};
function ma(a) {
    if (Error.captureStackTrace)
        Error.captureStackTrace(this, ma);
    else {
        var b = Error().stack;
        b && (this.stack = b)
    }
    a && (this.message = String(a))
}
y(ma, Error);
ma.prototype.name = "CustomError";
var na;
function oa(a, b) {
    for (var c = a.split("%s"), d = "", e = Array.prototype.slice.call(arguments, 1); e.length && 1 < c.length;)
        d += c.shift() + e.shift();
    return d + c.join("%s")
}
var pa = String.prototype.trim ? function(a) {
    return a.trim()
}
: function(a) {
    return a.replace(/^[\s\xa0]+|[\s\xa0]+$/g, "")
};
function qa(a) {
    return decodeURIComponent(a.replace(/\+/g, " "))
}
function sa(a) {
    if (!ta.test(a))
        return a;
    - 1 != a.indexOf("&") && (a = a.replace(ua, "&amp;"));
    - 1 != a.indexOf("<") && (a = a.replace(va, "&lt;"));
    - 1 != a.indexOf(">") && (a = a.replace(wa, "&gt;"));
    - 1 != a.indexOf('"') && (a = a.replace(xa, "&quot;"));
    - 1 != a.indexOf("'") && (a = a.replace(ya, "&#39;"));
    - 1 != a.indexOf("\x00") && (a = a.replace(za, "&#0;"));
    return a
}
var ua = /&/g, va = /</g, wa = />/g, xa = /"/g, ya = /'/g, za = /\x00/g, ta = /[\x00&<>"']/;
function Aa(a) {
    var b = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"'
    }, c;
    c = l.document.createElement("div");
    return a.replace(Ba, function(a, e) {
        var f = b[a];
        if (f)
            return f;
        if ("#" == e.charAt(0)) {
            var h = Number("0" + e.substr(1));
            isNaN(h) || (f = String.fromCharCode(h))
        }
        f || (c.innerHTML = a + " ", f = c.firstChild.nodeValue.slice(0, - 1));
        return b[a] = f
    })
}
function Ca(a) {
    return a.replace(/&([^;]+);/g, function(a, c) {
        switch (c) {
        case "amp":
            return "&";
        case "lt":
            return "<";
        case "gt":
            return ">";
        case "quot":
            return '"';
        default:
            if ("#" == c.charAt(0)) {
                var d = Number("0" + c.substr(1));
                if (!isNaN(d))
                    return String.fromCharCode(d)
                }
            return a
        }
    })
}
var Ba = /&([^;\s<&]+);?/g;
function Da(a, b) {
    return - 1 != a.indexOf(b)
}
function Ea(a, b) {
    for (var c = 0, d = pa(String(a)).split("."), e = pa(String(b)).split("."), f = Math.max(d.length, e.length), h = 0; 0 == c && h < f; h++) {
        var k = d[h] || "", m = e[h] || "", p = RegExp("(\\d*)(\\D*)", "g"), x = RegExp("(\\d*)(\\D*)", "g");
        do {
            var P = p.exec(k) || ["", "", ""], I = x.exec(m) || ["", "", ""];
            if (0 == P[0].length && 0 == I[0].length)
                break;
            c = Fa(0 == P[1].length ? 0 : parseInt(P[1], 10), 0 == I[1].length ? 0 : parseInt(I[1], 10)) || Fa(0 == P[2].length, 0 == I[2].length) || Fa(P[2], I[2])
        }
        while (0 == c)
        }
    return c
}
function Fa(a, b) {
    return a < b?-1 : a > b ? 1 : 0
}
function Ga(a) {
    for (var b = 0, c = 0; c < a.length; ++c)
        b = 31 * b + a.charCodeAt(c), b%=4294967296;
    return b
}
function Ha(a) {
    return String(a).replace(/\-([a-z])/g, function(a, c) {
        return c.toUpperCase()
    })
}
function Ia(a) {
    var b = t(void 0) ? "undefined".replace(/([-()\[\]{}+?*.$\^|,:#<!\\])/g, "\\$1").replace(/\x08/g, "\\x08"): "\\s";
    return a.replace(new RegExp("(^" + (b ? "|[" + b + "]+" : "") + ")([a-z])", "g"), function(a, b, e) {
        return b + e.toUpperCase()
    })
};
function Ja() {};
var Ka = Array.prototype, La = Ka.indexOf ? function(a, b, c) {
    return Ka.indexOf.call(a, b, c)
}
: function(a, b, c) {
    c = null == c ? 0 : 0 > c ? Math.max(0, a.length + c) : c;
    if (t(a))
        return t(b) && 1 == b.length ? a.indexOf(b, c) : - 1;
    for (; c < a.length; c++)
        if (c in a && a[c] === b)
            return c;
    return - 1
}, Ma = Ka.lastIndexOf ? function(a, b, c) {
    return Ka.lastIndexOf.call(a, b, null == c ? a.length - 1 : c)
}
: function(a, b, c) {
    c = null == c ? a.length - 1 : c;
    0 > c && (c = Math.max(0, a.length + c));
    if (t(a))
        return t(b) && 1 == b.length ? a.lastIndexOf(b, c) : - 1;
    for (; 0 <= c; c--)
        if (c in a && a[c] === b)
            return c;
    return - 1
}, z = Ka.forEach ? function(a, b, c) {
    Ka.forEach.call(a, b, c)
}
: function(a, b, c) {
    for (var d = a.length, e = t(a) ? a.split("") : a, f = 0; f < d; f++)
        f in e && b.call(c, e[f], f, a)
}, Na = Ka.filter ? function(a, b, c) {
    return Ka.filter.call(a, b, c)
}
: function(a, b, c) {
    for (var d = a.length, e = [], f = 0, h = t(a) ? a.split("") : a, k = 0; k < d; k++)
        if (k in h) {
            var m = h[k];
            b.call(c, m, k, a) && (e[f++] = m)
        }
    return e
}, A = Ka.map ? function(a, b, c) {
    return Ka.map.call(a, b, c)
}
: function(a, b, c) {
    for (var d = a.length, e = Array(d), f = t(a) ? a.split("") : a, h = 0; h < d; h++)
        h in f && (e[h] = b.call(c,
        f[h], h, a));
    return e
}, Oa = Ka.reduce ? function(a, b, c, d) {
    d && (b = u(b, d));
    return Ka.reduce.call(a, b, c)
}
: function(a, b, c, d) {
    var e = c;
    z(a, function(c, h) {
        e = b.call(d, e, c, h, a)
    });
    return e
}, Pa = Ka.some ? function(a, b, c) {
    return Ka.some.call(a, b, c)
}
: function(a, b, c) {
    for (var d = a.length, e = t(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e && b.call(c, e[f], f, a))
            return !0;
    return !1
}, Qa = Ka.every ? function(a, b, c) {
    return Ka.every.call(a, b, c)
}
: function(a, b, c) {
    for (var d = a.length, e = t(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e&&!b.call(c, e[f], f, a))
            return !1;
    return !0
};
function Ra(a, b, c) {
    b = Sa(a, b, c);
    return 0 > b ? null : t(a) ? a.charAt(b) : a[b]
}
function Sa(a, b, c) {
    for (var d = a.length, e = t(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e && b.call(c, e[f], f, a))
            return f;
    return - 1
}
function Ta(a, b) {
    return 0 <= La(a, b)
}
function Ua(a) {
    return 0 == a.length
}
function Va() {
    var a = Wa;
    if (!da(a))
        for (var b = a.length - 1; 0 <= b; b--)
            delete a[b];
    a.length = 0
}
function Xa(a, b) {
    Ta(a, b) || a.push(b)
}
function Ya(a, b) {
    var c = La(a, b), d;
    (d = 0 <= c) && Za(a, c);
    return d
}
function Za(a, b) {
    Ka.splice.call(a, b, 1)
}
function $a(a, b) {
    var c = Sa(a, b, void 0);
    0 <= c && Za(a, c)
}
function ab(a) {
    return Ka.concat.apply(Ka, arguments)
}
function bb(a) {
    var b = a.length;
    if (0 < b) {
        for (var c = Array(b), d = 0; d < b; d++)
            c[d] = a[d];
        return c
    }
    return []
}
function cb(a, b) {
    for (var c = 1; c < arguments.length; c++) {
        var d = arguments[c];
        if (ea(d)) {
            var e = a.length || 0, f = d.length || 0;
            a.length = e + f;
            for (var h = 0; h < f; h++)
                a[e + h] = d[h]
        } else 
            a.push(d)
    }
}
function db(a, b, c, d) {
    return Ka.splice.apply(a, eb(arguments, 1))
}
function eb(a, b, c) {
    return 2 >= arguments.length ? Ka.slice.call(a, b) : Ka.slice.call(a, b, c)
}
function fb(a, b, c) {
    if (!ea(a) ||!ea(b) || a.length != b.length)
        return !1;
    var d = a.length;
    c = c || gb;
    for (var e = 0; e < d; e++)
        if (!c(a[e], b[e]))
            return !1;
    return !0
}
function hb(a, b) {
    return a > b ? 1 : a < b?-1 : 0
}
function gb(a, b) {
    return a === b
};
function ib(a) {
    if (a.classList)
        return a.classList;
    a = a.className;
    return t(a) && a.match(/\S+/g) || []
}
function B(a, b) {
    return a.classList ? a.classList.contains(b) : Ta(ib(a), b)
}
function C(a, b) {
    a.classList ? a.classList.add(b) : B(a, b) || (a.className += 0 < a.className.length ? " " + b : b)
}
function jb(a, b) {
    if (a.classList)
        z(b, function(b) {
            C(a, b)
        });
    else {
        var c = {};
        z(ib(a), function(a) {
            c[a]=!0
        });
        z(b, function(a) {
            c[a]=!0
        });
        a.className = "";
        for (var d in c)
            a.className += 0 < a.className.length ? " " + d : d
    }
}
function D(a, b) {
    a.classList ? a.classList.remove(b) : B(a, b) && (a.className = Na(ib(a), function(a) {
        return a != b
    }).join(" "))
}
function kb(a, b) {
    a.classList ? z(b, function(b) {
        D(a, b)
    }) : a.className = Na(ib(a), function(a) {
        return !Ta(b, a)
    }).join(" ")
}
function mb(a, b, c) {
    c ? C(a, b) : D(a, b)
}
function nb(a, b) {
    var c=!B(a, b);
    mb(a, b, c)
}
function ob(a, b, c) {
    D(a, b);
    C(a, c)
};
function pb(a, b) {
    for (var c in a)
        b.call(void 0, a[c], c, a)
}
function qb(a, b, c) {
    var d = {}, e;
    for (e in a)
        b.call(c, a[e], e, a) && (d[e] = a[e]);
    return d
}
function rb(a) {
    var b = 0, c;
    for (c in a)
        b++;
    return b
}
function sb(a) {
    var b = [], c = 0, d;
    for (d in a)
        b[c++] = a[d];
    return b
}
function tb(a) {
    var b = [], c = 0, d;
    for (d in a)
        b[c++] = d;
    return b
}
function ub(a) {
    var b = vb, c;
    for (c in b)
        if (a.call(void 0, b[c], c, b))
            return c
}
function wb(a) {
    for (var b in a)
        return !1;
    return !0
}
function xb(a, b) {
    if (b in a)
        throw Error('The object already contains the key "' + b + '"');
    a[b]=!0
}
function yb(a) {
    var b = {}, c;
    for (c in a)
        b[c] = a[c];
    return b
}
function zb(a) {
    var b = ca(a);
    if ("object" == b || "array" == b) {
        if (a.clone)
            return a.clone();
        var b = "array" == b ? []: {}, c;
        for (c in a)
            b[c] = zb(a[c]);
        return b
    }
    return a
}
function Ab(a) {
    var b = {}, c;
    for (c in a)
        b[a[c]] = c;
    return b
}
var Bb = "constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");
function Cb(a, b) {
    for (var c, d, e = 1; e < arguments.length; e++) {
        d = arguments[e];
        for (c in d)
            a[c] = d[c];
        for (var f = 0; f < Bb.length; f++)
            c = Bb[f], Object.prototype.hasOwnProperty.call(d, c) && (a[c] = d[c])
    }
}
function Db(a) {
    var b = arguments.length;
    if (1 == b && da(arguments[0]))
        return Db.apply(null, arguments[0]);
    for (var c = {}, d = 0; d < b; d++)
        c[arguments[d]]=!0;
    return c
};
function Eb(a, b, c) {
    a && (a.dataset ? a.dataset[Fb(b)] = c : a.setAttribute("data-" + b, c))
}
function E(a, b) {
    return a ? a.dataset ? a.dataset[Fb(b)] : a.getAttribute("data-" + b) : null
}
function Gb(a, b) {
    a && (a.dataset ? delete a.dataset[Fb(b)] : a.removeAttribute("data-" + b))
}
var Hb = {};
function Fb(a) {
    return Hb[a] || (Hb[a] = String(a).replace(/\-([a-z])/g, function(a, c) {
        return c.toUpperCase()
    }))
};
function Ib() {
    this.$ = this.$;
    this.M = this.M
}
Ib.prototype.$=!1;
Ib.prototype.L = function() {
    return this.$
};
Ib.prototype.dispose = function() {
    this.$ || (this.$=!0, this.G())
};
function Jb(a, b) {
    a.M || (a.M = []);
    a.M.push(n(void 0) ? u(b, void 0) : b)
}
Ib.prototype.G = function() {
    if (this.M)
        for (; this.M.length;)
            this.M.shift()()
};
function Kb(a) {
    a && "function" == typeof a.dispose && a.dispose()
};
function Lb() {
    Ib.call(this);
    this.g = [];
    this.ja = {}
}
y(Lb, Ib);
g = Lb.prototype;
g.Wc = 1;
g.Wb = 0;
g.subscribe = function(a, b, c) {
    var d = this.ja[a];
    d || (d = this.ja[a] = []);
    var e = this.Wc;
    this.g[e] = a;
    this.g[e + 1] = b;
    this.g[e + 2] = c;
    this.Wc = e + 3;
    d.push(e);
    return e
};
g.unsubscribe = function(a, b, c) {
    if (a = this.ja[a]) {
        var d = this.g;
        if (a = Ra(a, function(a) {
            return d[a + 1] == b && d[a + 2] == c
        }))
            return this.ya(a)
    }
    return !1
};
g.ya = function(a) {
    if (0 != this.Wb)
        return this.j || (this.j = []), this.j.push(a), !1;
    var b = this.g[a];
    if (b) {
        var c = this.ja[b];
        c && Ya(c, a);
        delete this.g[a];
        delete this.g[a + 1];
        delete this.g[a + 2]
    }
    return !!b
};
g.publish = function(a, b) {
    var c = this.ja[a];
    if (c) {
        this.Wb++;
        for (var d = eb(arguments, 1), e = 0, f = c.length; e < f; e++) {
            var h = c[e];
            this.g[h + 1].apply(this.g[h + 2], d)
        }
        this.Wb--;
        if (this.j && 0 == this.Wb)
            for (; c = this.j.pop();)
                this.ya(c);
        return 0 != e
    }
    return !1
};
g.clear = function(a) {
    if (a) {
        var b = this.ja[a];
        b && (z(b, this.ya, this), delete this.ja[a])
    } else 
        this.g.length = 0, this.ja = {}
};
g.Ga = function(a) {
    if (a) {
        var b = this.ja[a];
        return b ? b.length : 0
    }
    a = 0;
    for (b in this.ja)
        a += this.Ga(b);
    return a
};
g.G = function() {
    Lb.J.G.call(this);
    delete this.g;
    delete this.ja;
    delete this.j
};
var Mb = window.yt && window.yt.config_ || {};
q("yt.config_", Mb, void 0);
q("yt.tokens_", window.yt && window.yt.tokens_ || {}, void 0);
var Nb = window.yt && window.yt.msgs_ || {};
q("yt.msgs_", Nb, void 0);
function Ob(a) {
    var b = arguments;
    if (1 < b.length) {
        var c = b[0];
        Mb[c] = b[1]
    } else 
        for (c in b = b[0], b)
            Mb[c] = b[c]
}
function F(a, b) {
    return a in Mb ? Mb[a] : b
}
function Pb() {
    return F("XSRF_TOKEN")
}
function G(a, b) {
    fa(a) && (a = Qb(a));
    return window.setTimeout(a, b)
}
function Rb(a, b) {
    fa(a) && (a = Qb(a));
    window.setInterval(a, b)
}
function H(a) {
    window.clearTimeout(a)
}
function Qb(a) {
    return a && window.yterr ? function() {
        try {
            return a.apply(this, arguments)
        } catch (b) {
            throw Sb(b), b;
        }
    } : a
}
function Sb(a, b) {
    var c = r("yt.www.errors.log");
    c ? c(a, b) : (c = F("ERRORS") || [], c.push([a, b]), Ob("ERRORS", c))
}
function Tb() {
    var a = {}, b = "FLASH_UPGRADE"in Nb ? Nb.FLASH_UPGRADE: 'You need to upgrade your Adobe Flash Player to watchthis video. <br> <a href="http://get.adobe.com/flashplayer/">Download it from Adobe.</a>';
    if (b)
        for (var c in a)
            b = b.replace(new RegExp("\\$" + c, "gi"), function() {
                return a[c]
            });
    return b
}
function Ub(a) {
    var b = "MASTHEAD_NOTIFICATIONS_LABEL"in Nb ? Nb.MASTHEAD_NOTIFICATIONS_LABEL: {}, c = F("I18N_PLURAL_RULES") || function(a) {
        return 1 == a ? "one" : "other"
    };
    return (b = b["case" + a] || b[c(a)]) ? b.replace("#", a.toString()) : a + ""
}
var Vb = "Microsoft Internet Explorer" == navigator.appName;
var Wb = r("yt.pubsub.instance_") || new Lb;
Lb.prototype.subscribe = Lb.prototype.subscribe;
Lb.prototype.unsubscribeByKey = Lb.prototype.ya;
Lb.prototype.publish = Lb.prototype.publish;
Lb.prototype.clear = Lb.prototype.clear;
q("yt.pubsub.instance_", Wb, void 0);
var Xb = r("yt.pubsub.subscribedKeys_") || {};
q("yt.pubsub.subscribedKeys_", Xb, void 0);
var Yb = r("yt.pubsub.topicToKeys_") || {};
q("yt.pubsub.topicToKeys_", Yb, void 0);
var Zb = r("yt.pubsub.isSynchronous_") || {};
q("yt.pubsub.isSynchronous_", Zb, void 0);
var $b = r("yt.pubsub.skipSubId_") || null;
q("yt.pubsub.skipSubId_", $b, void 0);
function J(a, b, c) {
    var d = ac();
    if (d) {
        var e = d.subscribe(a, function() {
            if (!$b || $b != e) {
                var d = arguments, h = function() {
                    Xb[e] && b.apply(c || window, d)
                };
                try {
                    Zb[a] ? h() : G(h, 0)
                } catch (k) {
                    Sb(k)
                }
            }
        }, c);
        Xb[e]=!0;
        Yb[a] || (Yb[a] = []);
        Yb[a].push(e);
        return e
    }
    return 0
}
function bc(a) {
    var b = ac();
    b && ("number" == typeof a ? a = [a] : "string" == typeof a && (a = [parseInt(a, 10)]), z(a, function(a) {
        b.unsubscribeByKey(a);
        delete Xb[a]
    }))
}
function K(a, b) {
    var c = ac();
    c && c.publish.apply(c, arguments)
}
function cc(a, b) {
    Zb[a]=!0;
    var c = ac();
    c && c.publish.apply(c, arguments);
    Zb[a]=!1
}
function dc(a) {
    Yb[a] && (a = Yb[a], z(a, function(a) {
        Xb[a] && delete Xb[a]
    }), a.length = 0)
}
function ec(a) {
    var b = ac();
    if (b)
        if (b.clear(a), a)
            dc(a);
        else 
            for (var c in Yb)
                dc(c)
}
function ac() {
    return r("yt.pubsub.instance_")
};
function fc(a, b) {
    if (window.spf) {
        var c = "";
        if (a) {
            var d = a.indexOf("jsbin/"), e = a.lastIndexOf(".js"), f = d + 6;
            - 1 < d&&-1 < e && e > f && (c = a.substring(f, e), c = c.replace(gc, ""), c = c.replace(hc, ""), c = c.replace("debug-", ""), c = c.replace("tracing-", ""))
        }
        spf.script.load(a, c, b)
    } else 
        ic(a, b)
}
function ic(a, b) {
    var c = jc(a), d = document.getElementById(c), e = d && E(d, "loaded"), f = d&&!e;
    if (e)
        b && b();
    else {
        if (b) {
            var e = J(c, b), h = "" + ha(b);
            kc[h] = e
        }
        f || (d = lc(a, c, function() {
            E(d, "loaded") || (Eb(d, "loaded", "true"), K(c), G(v(ec, c), 0))
        }))
    }
}
function lc(a, b, c) {
    var d = document.createElement("script");
    d.id = b;
    d.onload = function() {
        c && setTimeout(c, 0)
    };
    d.onreadystatechange = function() {
        switch (d.readyState) {
        case "loaded":
        case "complete":
            d.onload()
        }
    };
    d.src = a;
    a = document.getElementsByTagName("head")[0] || document.body;
    a.insertBefore(d, a.firstChild);
    return d
}
function mc(a, b) {
    if (a && b) {
        var c = "" + ha(b);
        (c = kc[c]) && bc(c)
    }
}
function jc(a) {
    var b = document.createElement("a");
    b.href = a;
    a = b.href.replace(/^[a-zA-Z]+:\/\//, "//");
    return "js-" + Ga(a)
}
var gc = /\.vflset|-vfl[a-zA-Z0-9_+=-]+/, hc = /-[a-zA-Z]{2,3}_[a-zA-Z]{2,3}(?=(\/|$))/, kc = {};
function nc(a, b) {
    if (window.spf) {
        var c = a.match(oc);
        spf.style.load(a, c ? c[1] : "", b);
        return null
    }
    return pc(a, b)
}
function qc(a, b, c) {
    if (a = F(a)) {
        var d = v(nc, a, b);
        if (c)
            var e = J(c, function() {
                bc(e);
                d()
            });
        else 
            d()
    }
}
function pc(a, b) {
    var c = rc(a), d = document.getElementById(c), e = d && E(d, "loaded"), f = d&&!e;
    if (e)
        return b && b(), d;
    b && (J(c, b), ha(b));
    return f ? d : d = sc(a, c, function() {
        E(d, "loaded") || (Eb(d, "loaded", "true"), K(c), G(v(ec, c), 0))
    })
}
function sc(a, b, c) {
    var d = document.createElement("link");
    d.id = b;
    d.rel = "stylesheet";
    d.onload = function() {
        c && setTimeout(c, 0)
    };
    d.href = a;
    (document.getElementsByTagName("head")[0] || document.body).appendChild(d);
    return d
}
function rc(a) {
    var b = document.createElement("a");
    b.href = a;
    a = b.href.replace(/^[a-zA-Z]+:\/\//, "//");
    return "css-" + Ga(a)
}
var oc = /cssbin\/(?:debug-)?([a-zA-Z0-9_-]+?)(?:-2x|-web|-rtl|-vfl|.css)/;
function L(a, b) {
    this.x = n(a) ? a : 0;
    this.y = n(b) ? b : 0
}
L.prototype.clone = function() {
    return new L(this.x, this.y)
};
function tc(a, b) {
    return new L(a.x - b.x, a.y - b.y)
}
L.prototype.ceil = function() {
    this.x = Math.ceil(this.x);
    this.y = Math.ceil(this.y);
    return this
};
L.prototype.floor = function() {
    this.x = Math.floor(this.x);
    this.y = Math.floor(this.y);
    return this
};
L.prototype.round = function() {
    this.x = Math.round(this.x);
    this.y = Math.round(this.y);
    return this
};
function uc(a, b) {
    this.width = a;
    this.height = b
}
g = uc.prototype;
g.clone = function() {
    return new uc(this.width, this.height)
};
g.isEmpty = function() {
    return !(this.width * this.height)
};
g.ceil = function() {
    this.width = Math.ceil(this.width);
    this.height = Math.ceil(this.height);
    return this
};
g.floor = function() {
    this.width = Math.floor(this.width);
    this.height = Math.floor(this.height);
    return this
};
g.round = function() {
    this.width = Math.round(this.width);
    this.height = Math.round(this.height);
    return this
};
var vc;
t: {
    var wc = l.navigator;
    if (wc) {
        var xc = wc.userAgent;
        if (xc) {
            vc = xc;
            break t
        }
    }
    vc = ""
};
var yc = Da(vc, "Opera") || Da(vc, "OPR"), N = Da(vc, "Trident") || Da(vc, "MSIE"), zc = Da(vc, "Gecko")&&!Da(vc.toLowerCase(), "webkit")&&!(Da(vc, "Trident") || Da(vc, "MSIE")), Ac = Da(vc.toLowerCase(), "webkit"), Bc = Da(vc, "Macintosh"), Cc = Da(vc, "Windows"), Dc = l.navigator || null, Ec=!!Dc && Da(Dc.appVersion || "", "X11");
function Fc() {
    var a = l.document;
    return a ? a.documentMode : void 0
}
var Gc = function() {
    var a = "", b;
    if (yc && l.opera)
        return a = l.opera.version, fa(a) ? a() : a;
    zc ? b = /rv\:([^\);]+)(\)|;)/ : N ? b = /\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/ : Ac && (b = /WebKit\/(\S+)/);
    b && (a = (a = b.exec(vc)) ? a[1] : "");
    return N && (b = Fc(), b > parseFloat(a)) ? String(b) : a
}(), Hc = {};
function Ic(a) {
    return Hc[a] || (Hc[a] = 0 <= Ea(Gc, a))
}
function Jc(a) {
    return N && Kc >= a
}
var Lc = l.document, Kc = Lc && N ? Fc() || ("CSS1Compat" == Lc.compatMode ? parseInt(Gc, 10) : 5): void 0;
var Mc=!N || Jc(9), Nc=!zc&&!N || N && Jc(9) || zc && Ic("1.9.1"), Oc = N&&!Ic("9"), Pc = N || yc || Ac;
function Qc(a) {
    return a ? new Rc(Sc(a)) : na || (na = new Rc)
}
function O(a) {
    return t(a) ? document.getElementById(a) : a
}
function Tc(a) {
    var b = document;
    return t(a) ? b.getElementById(a) : a
}
function Uc(a, b) {
    var c = b || document;
    return c.querySelectorAll && c.querySelector ? c.querySelectorAll("." + a) : Vc("*", a, b)
}
function Q(a, b) {
    var c = b || document, d = null;
    c.querySelectorAll && c.querySelector ? d = c.querySelector("." + a) : d = Vc("*", a, b)[0];
    return d || null
}
function Vc(a, b, c) {
    var d = document;
    c = c || d;
    a = a && "*" != a ? a.toUpperCase() : "";
    if (c.querySelectorAll && c.querySelector && (a || b))
        return c.querySelectorAll(a + (b ? "." + b : ""));
    if (b && c.getElementsByClassName) {
        c = c.getElementsByClassName(b);
        if (a) {
            for (var d = {}, e = 0, f = 0, h; h = c[f]; f++)
                a == h.nodeName && (d[e++] = h);
            d.length = e;
            return d
        }
        return c
    }
    c = c.getElementsByTagName(a || "*");
    if (b) {
        d = {};
        for (f = e = 0; h = c[f]; f++)
            a = h.className, "function" == typeof a.split && Ta(a.split(/\s+/), b) && (d[e++] = h);
        d.length = e;
        return d
    }
    return c
}
function Wc(a, b) {
    pb(b, function(b, d) {
        "style" == d ? a.style.cssText = b : "class" == d ? a.className = b : "for" == d ? a.htmlFor = b : d in Xc ? a.setAttribute(Xc[d], b) : 0 == d.lastIndexOf("aria-", 0) || 0 == d.lastIndexOf("data-", 0) ? a.setAttribute(d, b) : a[d] = b
    })
}
var Xc = {
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
function Yc(a) {
    a = a.document;
    a = Zc(a) ? a.documentElement : a.body;
    return new uc(a.clientWidth, a.clientHeight)
}
function $c(a) {
    var b = ad(a);
    a = bd(a);
    return N && Ic("10") && a.pageYOffset != b.scrollTop ? new L(b.scrollLeft, b.scrollTop) : new L(a.pageXOffset || b.scrollLeft, a.pageYOffset || b.scrollTop)
}
function ad(a) {
    return !Ac && Zc(a) ? a.documentElement : a.body || a.documentElement
}
function bd(a) {
    return a.parentWindow || a.defaultView
}
function cd(a, b, c) {
    var d = arguments, e = document, f = d[0], h = d[1];
    if (!Mc && h && (h.name || h.type)) {
        f = ["<", f];
        h.name && f.push(' name="', sa(h.name), '"');
        if (h.type) {
            f.push(' type="', sa(h.type), '"');
            var k = {};
            Cb(k, h);
            delete k.type;
            h = k
        }
        f.push(">");
        f = f.join("")
    }
    f = e.createElement(f);
    h && (t(h) ? f.className = h : da(h) ? f.className = h.join(" ") : Wc(f, h));
    2 < d.length && dd(e, f, d, 2);
    return f
}
function dd(a, b, c, d) {
    function e(c) {
        c && b.appendChild(t(c) ? a.createTextNode(c) : c)
    }
    for (; d < c.length; d++) {
        var f = c[d];
        !ea(f) || ga(f) && 0 < f.nodeType ? e(f) : z(ed(f) ? bb(f) : f, e)
    }
}
function fd(a) {
    var b = document, c = b.createElement("div");
    N ? (c.innerHTML = "<br>" + a, c.removeChild(c.firstChild)) : c.innerHTML = a;
    if (1 == c.childNodes.length)
        return c.removeChild(c.firstChild);
    for (a = b.createDocumentFragment(); c.firstChild;)
        a.appendChild(c.firstChild);
    return a
}
function Zc(a) {
    return "CSS1Compat" == a.compatMode
}
function gd(a, b) {
    dd(Sc(a), a, arguments, 1)
}
function hd(a) {
    for (var b; b = a.firstChild;)
        a.removeChild(b)
}
function id(a, b, c) {
    a.insertBefore(b, a.childNodes[c] || null)
}
function jd(a) {
    a && a.parentNode && a.parentNode.removeChild(a)
}
function kd(a, b) {
    var c = b.parentNode;
    c && c.replaceChild(a, b)
}
function ld(a) {
    return Nc && void 0 != a.children ? a.children : Na(a.childNodes, function(a) {
        return 1 == a.nodeType
    })
}
function md(a) {
    return void 0 != a.firstElementChild ? a.firstElementChild : nd(a.firstChild, !0)
}
function nd(a, b) {
    for (; a && 1 != a.nodeType;)
        a = b ? a.nextSibling : a.previousSibling;
    return a
}
function od(a) {
    if (!a)
        return null;
    if (a.firstChild)
        return a.firstChild;
    for (; a&&!a.nextSibling;)
        a = a.parentNode;
    return a ? a.nextSibling : null
}
function pd(a) {
    if (!a)
        return null;
    if (!a.previousSibling)
        return a.parentNode;
    for (a = a.previousSibling; a && a.lastChild;)
        a = a.lastChild;
    return a
}
function qd(a) {
    var b;
    if (Pc&&!(N && Ic("9")&&!Ic("10") && l.SVGElement && a instanceof l.SVGElement) && (b = a.parentElement))
        return b;
    b = a.parentNode;
    return ga(b) && 1 == b.nodeType ? b : null
}
function rd(a, b) {
    if (a.contains && 1 == b.nodeType)
        return a == b || a.contains(b);
    if ("undefined" != typeof a.compareDocumentPosition)
        return a == b || Boolean(a.compareDocumentPosition(b) & 16);
    for (; b && a != b;)
        b = b.parentNode;
    return b == a
}
function Sc(a) {
    return 9 == a.nodeType ? a : a.ownerDocument || a.document
}
function sd(a, b) {
    if ("textContent"in a)
        a.textContent = b;
    else if (3 == a.nodeType)
        a.data = b;
    else if (a.firstChild && 3 == a.firstChild.nodeType) {
        for (; a.lastChild != a.firstChild;)
            a.removeChild(a.lastChild);
        a.firstChild.data = b
    } else {
        hd(a);
        var c = Sc(a);
        a.appendChild(c.createTextNode(String(b)))
    }
}
var td = {
    SCRIPT: 1,
    STYLE: 1,
    HEAD: 1,
    IFRAME: 1,
    OBJECT: 1
}, ud = {
    IMG: " ",
    BR: "\n"
};
function vd(a, b, c) {
    if (!(a.nodeName in td))
        if (3 == a.nodeType)
            c ? b.push(String(a.nodeValue).replace(/(\r\n|\r|\n)/g, "")) : b.push(a.nodeValue);
        else if (a.nodeName in ud)
            b.push(ud[a.nodeName]);
        else 
            for (a = a.firstChild; a;)
                vd(a, b, c), a = a.nextSibling
}
function ed(a) {
    if (a && "number" == typeof a.length) {
        if (ga(a))
            return "function" == typeof a.item || "string" == typeof a.item;
        if (fa(a))
            return "function" == typeof a.item
    }
    return !1
}
function wd(a, b, c) {
    return b ? xd(a, function(a) {
        return !b || t(a.className) && Ta(a.className.split(/\s+/), b)
    }, c) : null
}
function yd(a, b) {
    return wd(a, b, void 0)
}
function xd(a, b, c) {
    for (var d = null == c, e = 0; a && (d || e <= c);) {
        if (b(a))
            return a;
        a = a.parentNode;
        e++
    }
    return null
}
function Rc(a) {
    this.g = a || l.document || document
}
Rc.prototype.wa = function(a) {
    return t(a) ? this.g.getElementById(a) : a
};
Rc.prototype.createElement = function(a) {
    return this.g.createElement(a)
};
function zd(a) {
    return Zc(a.g)
}
function Ad(a) {
    return $c(a.g)
}
Rc.prototype.appendChild = function(a, b) {
    a.appendChild(b)
};
Rc.prototype.contains = rd;
function Bd() {
    return Ac ? "Webkit" : zc ? "Moz" : N ? "ms" : yc ? "O" : null
};
function Cd(a, b, c, d) {
    this.top = a;
    this.right = b;
    this.bottom = c;
    this.left = d
}
g = Cd.prototype;
g.clone = function() {
    return new Cd(this.top, this.right, this.bottom, this.left)
};
g.contains = function(a) {
    return this && a ? a instanceof Cd ? a.left >= this.left && a.right <= this.right && a.top >= this.top && a.bottom <= this.bottom : a.x >= this.left && a.x <= this.right && a.y >= this.top && a.y <= this.bottom : !1
};
g.ceil = function() {
    this.top = Math.ceil(this.top);
    this.right = Math.ceil(this.right);
    this.bottom = Math.ceil(this.bottom);
    this.left = Math.ceil(this.left);
    return this
};
g.floor = function() {
    this.top = Math.floor(this.top);
    this.right = Math.floor(this.right);
    this.bottom = Math.floor(this.bottom);
    this.left = Math.floor(this.left);
    return this
};
g.round = function() {
    this.top = Math.round(this.top);
    this.right = Math.round(this.right);
    this.bottom = Math.round(this.bottom);
    this.left = Math.round(this.left);
    return this
};
function Dd(a, b, c, d) {
    this.left = a;
    this.top = b;
    this.width = c;
    this.height = d
}
g = Dd.prototype;
g.clone = function() {
    return new Dd(this.left, this.top, this.width, this.height)
};
g.contains = function(a) {
    return a instanceof Dd ? this.left <= a.left && this.left + this.width >= a.left + a.width && this.top <= a.top && this.top + this.height >= a.top + a.height : a.x >= this.left && a.x <= this.left + this.width && a.y >= this.top && a.y <= this.top + this.height
};
g.ceil = function() {
    this.left = Math.ceil(this.left);
    this.top = Math.ceil(this.top);
    this.width = Math.ceil(this.width);
    this.height = Math.ceil(this.height);
    return this
};
g.floor = function() {
    this.left = Math.floor(this.left);
    this.top = Math.floor(this.top);
    this.width = Math.floor(this.width);
    this.height = Math.floor(this.height);
    return this
};
g.round = function() {
    this.left = Math.round(this.left);
    this.top = Math.round(this.top);
    this.width = Math.round(this.width);
    this.height = Math.round(this.height);
    return this
};
function Ed(a, b, c) {
    t: if (c = Ha(c), void 0 === a.style[c]) {
        var d = Bd() + Ia(c);
        if (void 0 !== a.style[d]) {
            c = d;
            break t
        }
    }
    c && (a.style[c] = b)
}
function Fd(a, b) {
    var c = Sc(a);
    return c.defaultView && c.defaultView.getComputedStyle && (c = c.defaultView.getComputedStyle(a, null)) ? c[b] || c.getPropertyValue(b) || "" : ""
}
function Gd(a, b) {
    return Fd(a, b) || (a.currentStyle ? a.currentStyle[b] : null) || a.style && a.style[b]
}
function Hd(a) {
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
    N && a.ownerDocument.body && (a = a.ownerDocument, b.left -= a.documentElement.clientLeft + a.body.clientLeft, b.top -= a.documentElement.clientTop + a.body.clientTop);
    return b
}
function Id(a) {
    if (N&&!Jc(8))
        return a.offsetParent;
    var b = Sc(a), c = Gd(a, "position"), d = "fixed" == c || "absolute" == c;
    for (a = a.parentNode; a && a != b; a = a.parentNode)
        if (c = Gd(a, "position"), d = d && "static" == c && a != b.documentElement && a != b.body, !d && (a.scrollWidth > a.clientWidth || a.scrollHeight > a.clientHeight || "fixed" == c || "absolute" == c || "relative" == c))
            return a;
    return null
}
function Jd(a) {
    for (var b = new Cd(0, Infinity, Infinity, 0), c = Qc(a), d = c.g.body, e = c.g.documentElement, f = ad(c.g); a = Id(a);)
        if (!(N && 0 == a.clientWidth || Ac && 0 == a.clientHeight && a == d) && a != d && a != e && "visible" != Gd(a, "overflow")) {
            var h = Kd(a), k;
            k = a;
            if (zc&&!Ic("1.9")) {
                var m = parseFloat(Fd(k, "borderLeftWidth"));
                if (Ld(k))
                    var p = k.offsetWidth - k.clientWidth - m - parseFloat(Fd(k, "borderRightWidth")), m = m + p;
                    k = new L(m, parseFloat(Fd(k, "borderTopWidth")))
                } else 
                    k = new L(k.clientLeft, k.clientTop);
                    h.x += k.x;
                    h.y += k.y;
                    b.top = Math.max(b.top,
                    h.y);
                    b.right = Math.min(b.right, h.x + a.clientWidth);
                    b.bottom = Math.min(b.bottom, h.y + a.clientHeight);
                    b.left = Math.max(b.left, h.x)
        }
    d = f.scrollLeft;
    f = f.scrollTop;
    b.left = Math.max(b.left, d);
    b.top = Math.max(b.top, f);
    c = Yc(bd(c.g) || window);
    b.right = Math.min(b.right, d + c.width);
    b.bottom = Math.min(b.bottom, f + c.height);
    return 0 <= b.top && 0 <= b.left && b.bottom > b.top && b.right > b.left ? b : null
}
function Kd(a) {
    var b, c = Sc(a), d = Gd(a, "position"), e = zc && c.getBoxObjectFor&&!a.getBoundingClientRect && "absolute" == d && (b = c.getBoxObjectFor(a)) && (0 > b.screenX || 0 > b.screenY), f = new L(0, 0), h;
    b = c ? Sc(c) : document;
    h=!N || Jc(9) || zd(Qc(b)) ? b.documentElement : b.body;
    if (a == h)
        return f;
    if (a.getBoundingClientRect)
        b = Hd(a), a = Ad(Qc(c)), f.x = b.left + a.x, f.y = b.top + a.y;
    else if (c.getBoxObjectFor&&!e)
        b = c.getBoxObjectFor(a), a = c.getBoxObjectFor(h), f.x = b.screenX - a.screenX, f.y = b.screenY - a.screenY;
    else {
        b = a;
        do {
            f.x += b.offsetLeft;
            f.y +=
            b.offsetTop;
            b != a && (f.x += b.clientLeft || 0, f.y += b.clientTop || 0);
            if (Ac && "fixed" == Gd(b, "position")) {
                f.x += c.body.scrollLeft;
                f.y += c.body.scrollTop;
                break
            }
            b = b.offsetParent
        }
        while (b && b != a);
        if (yc || Ac && "absolute" == d)
            f.y -= c.body.offsetTop;
        for (b = a; (b = Id(b)) && b != c.body && b != h;)
            f.x -= b.scrollLeft, yc && "TR" == b.tagName || (f.y -= b.scrollTop)
    }
    return f
}
function Md(a) {
    var b;
    if (a.getBoundingClientRect)
        b = Hd(a), b = new L(b.left, b.top);
    else {
        b = Ad(Qc(a));
        var c = Kd(a);
        b = new L(c.x - b.x, c.y - b.y)
    }
    if (zc&&!Ic(12)) {
        i:
        {
            c = Ha("transform");
            if (void 0 === a.style[c] && (c = Bd() + Ia(c), void 0 !== a.style[c])) {
                c = (Ac ? "-webkit" : zc ? "-moz" : N ? "-ms" : yc ? "-o" : null) + "-transform";
                break i
            }
            c = "transform"
        }
        a = (a = Gd(a, c) || Gd(a, "transform")) ? (a = a.match(Nd)) ? new L(parseFloat(a[1]), parseFloat(a[2])) : new L(0, 0) : new L(0, 0);
        a = new L(b.x + a.x, b.y + a.y)
    } else 
        a = b;
    return a
}
function Od(a) {
    if (1 == a.nodeType)
        return Md(a);
    var b = fa(a.k), c = a;
    a.targetTouches && a.targetTouches.length ? c = a.targetTouches[0] : b && a.g.targetTouches && a.g.targetTouches.length && (c = a.g.targetTouches[0]);
    return new L(c.clientX, c.clientY)
}
function Pd(a, b) {
    "number" == typeof a && (a = (b ? Math.round(a) : a) + "px");
    return a
}
function Qd(a) {
    var b = Rd;
    if ("none" != Gd(a, "display"))
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
}
function Rd(a) {
    var b = a.offsetWidth, c = a.offsetHeight, d = Ac&&!b&&!c;
    return n(b)&&!d ||!a.getBoundingClientRect ? new uc(b, c) : (a = Hd(a), new uc(a.right - a.left, a.bottom - a.top))
}
function Sd(a) {
    var b = Kd(a);
    a = Qd(a);
    return new Dd(b.x, b.y, a.width, a.height)
}
function Ld(a) {
    return "rtl" == Gd(a, "direction")
}
function Td(a, b) {
    if (/^\d+px?$/.test(b))
        return parseInt(b, 10);
    var c = a.style.left, d = a.runtimeStyle.left;
    a.runtimeStyle.left = a.currentStyle.left;
    a.style.left = b;
    var e = a.style.pixelLeft;
    a.style.left = c;
    a.runtimeStyle.left = d;
    return e
}
function Ud(a, b) {
    var c = a.currentStyle ? a.currentStyle[b]: null;
    return c ? Td(a, c) : 0
}
function Vd(a) {
    if (N) {
        var b = Ud(a, "paddingLeft"), c = Ud(a, "paddingRight"), d = Ud(a, "paddingTop");
        a = Ud(a, "paddingBottom");
        return new Cd(d, c, a, b)
    }
    b = Fd(a, "paddingLeft");
    c = Fd(a, "paddingRight");
    d = Fd(a, "paddingTop");
    a = Fd(a, "paddingBottom");
    return new Cd(parseFloat(d), parseFloat(c), parseFloat(a), parseFloat(b))
}
var Wd = {
    thin: 2,
    medium: 4,
    thick: 6
};
function Xd(a, b) {
    if ("none" == (a.currentStyle ? a.currentStyle[b + "Style"] : null))
        return 0;
    var c = a.currentStyle ? a.currentStyle[b + "Width"]: null;
    return c in Wd ? Wd[c] : Td(a, c)
}
function Yd(a) {
    if (N&&!Jc(9)) {
        var b = Xd(a, "borderLeft"), c = Xd(a, "borderRight"), d = Xd(a, "borderTop");
        a = Xd(a, "borderBottom");
        return new Cd(d, c, a, b)
    }
    b = Fd(a, "borderLeftWidth");
    c = Fd(a, "borderRightWidth");
    d = Fd(a, "borderTopWidth");
    a = Fd(a, "borderBottomWidth");
    return new Cd(parseFloat(d), parseFloat(c), parseFloat(a), parseFloat(b))
}
var Nd = /matrix\([0-9\.\-]+, [0-9\.\-]+, [0-9\.\-]+, [0-9\.\-]+, ([0-9\.\-]+)p?x?, ([0-9\.\-]+)p?x?\)/;
function Zd(a) {
    var b = a.__yt_uid_key;
    b || (b = $d(), a.__yt_uid_key = b);
    return b
}
var $d = r("yt.dom.getNextId_");
if (!$d) {
    $d = function() {
        return ++ae
    };
    q("yt.dom.getNextId_", $d, void 0);
    var ae = 0
}
function be(a, b, c) {
    a = Vc(a, b, c);
    return a.length ? a[0] : null
}
function ce(a, b) {
    "disabled"in a && (a.disabled=!b);
    1 == a.nodeType && mb(a, "disabled", !b);
    if (a.hasChildNodes())
        for (var c = 0, d; d = a.childNodes[c]; ++c)
            ce(d, b)
}
function de(a) {
    a = a.replace(/^[\s\xa0]+/, "");
    var b = String(a.substr(0, 3)).toLowerCase();
    if (0 == ("<tr" < b?-1 : "<tr" == b ? 0 : 1))
        return a = fd("<table><tbody>" + a + "</tbody></table>"), be("tr", null, a);
    b = document.createElement("div");
    b.innerHTML = a;
    return md(b)
}
function ee(a) {
    mb(document.body, "hide-players", !0);
    a && mb(a, "preserve-players", !0)
}
function fe() {
    mb(document.body, "hide-players", !1);
    var a = Uc("preserve-players");
    z(a, function(a) {
        D(a, "preserve-players")
    })
};
function ge(a) {
    if (a = a || window.event) {
        for (var b in a)
            b in he || (this[b] = a[b]);
        this.gb = a;
        (b = a.target || a.srcElement) && 3 == b.nodeType && (b = b.parentNode);
        this.target = b;
        if (b = a.relatedTarget)
            try {
                b = b.nodeName ? b : null
        } catch (c) {
            b = null
        } else 
            "mouseover" == this.type ? b = a.fromElement : "mouseout" == this.type && (b = a.toElement);
        this.relatedTarget = b;
        this.clientX = void 0 != a.clientX ? a.clientX : a.pageX;
        this.clientY = void 0 != a.clientY ? a.clientY : a.pageY;
        this.keyCode = a.keyCode ? a.keyCode : a.which;
        this.charCode = a.charCode || ("keypress" ==
        this.type ? this.keyCode : 0);
        this.altKey = a.altKey;
        this.ctrlKey = a.ctrlKey;
        this.shiftKey = a.shiftKey;
        "MozMousePixelScroll" == this.type ? (this.wheelDeltaX = a.axis == a.HORIZONTAL_AXIS ? a.detail : 0, this.wheelDeltaY = a.axis == a.HORIZONTAL_AXIS ? 0 : a.detail) : window.opera ? (this.wheelDeltaX = 0, this.wheelDeltaY = a.detail) : 0 == a.wheelDelta%120 ? "WebkitTransform"in document.documentElement.style ? window.chrome && 0 == navigator.platform.indexOf("Mac") ? (this.wheelDeltaX = a.wheelDeltaX/-30, this.wheelDeltaY = a.wheelDeltaY/-30) : (this.wheelDeltaX =
        a.wheelDeltaX/-1.2, this.wheelDeltaY = a.wheelDeltaY/-1.2) : (this.wheelDeltaX = 0, this.wheelDeltaY = a.wheelDelta/-1.6) : (this.wheelDeltaX = a.wheelDeltaX/-3, this.wheelDeltaY = a.wheelDeltaY/-3)
    }
}
g = ge.prototype;
g.gb = null;
g.type = "";
g.target = null;
g.relatedTarget = null;
g.currentTarget = null;
g.data = null;
g.state = null;
g.keyCode = 0;
g.charCode = 0;
g.altKey=!1;
g.ctrlKey=!1;
g.shiftKey=!1;
g.clientX = 0;
g.clientY = 0;
g.wheelDeltaX = 0;
g.wheelDeltaY = 0;
g.preventDefault = function() {
    this.gb.returnValue=!1;
    this.gb.preventDefault && this.gb.preventDefault()
};
g.stopPropagation = function() {
    this.gb.cancelBubble=!0;
    this.gb.stopPropagation && this.gb.stopPropagation()
};
var he = {
    stopImmediatePropagation: 1,
    stopPropagation: 1,
    preventMouseEvent: 1,
    preventManipulation: 1,
    preventDefault: 1,
    layerX: 1,
    layerY: 1,
    scale: 1,
    rotation: 1
};
var vb = r("yt.events.listeners_") || {};
q("yt.events.listeners_", vb, void 0);
var ie = r("yt.events.counter_") || {
    count: 0
};
q("yt.events.counter_", ie, void 0);
function je(a, b, c, d) {
    return ub(function(e) {
        return e[0] == a && e[1] == b && e[2] == c && e[4]==!!d
    })
}
function R(a, b, c, d) {
    if (!a ||!a.addEventListener&&!a.attachEvent)
        return "";
    d=!!d;
    var e = je(a, b, c, d);
    if (e)
        return e;
    var e=++ie.count + "", f=!("mouseenter" != b && "mouseleave" != b ||!a.addEventListener || "onmouseenter"in document), h;
    h = f ? function(d) {
        d = new ge(d);
        if (!xd(d.relatedTarget, function(b) {
            return b == a
        }))
            return d.currentTarget = a, d.type = b, c.call(a, d)
    } : function(b) {
        b = new ge(b);
        b.currentTarget = a;
        return c.call(a, b)
    };
    h = Qb(h);
    vb[e] = [a, b, c, h, d];
    a.addEventListener ? "mouseenter" == b && f ? a.addEventListener("mouseover", h,
    d) : "mouseleave" == b && f ? a.addEventListener("mouseout", h, d) : "mousewheel" == b && "MozBoxSizing"in document.documentElement.style ? a.addEventListener("MozMousePixelScroll", h, d) : a.addEventListener(b, h, d) : a.attachEvent("on" + b, h);
    return e
}
function ke(a, b, c) {
    var d;
    return d = R(a, b, function() {
        le(d);
        c.apply(a, arguments)
    }, void 0)
}
function me(a, b, c, d) {
    return ne(a, b, c, function(a) {
        return B(a, d)
    })
}
function ne(a, b, c, d) {
    var e = a || document;
    return R(e, b, function(a) {
        var b = xd(a.target, function(a) {
            return a === e || d(a)
        });
        b && b !== e&&!b.disabled && (a.currentTarget = b, c.call(b, a))
    })
}
function le(a) {
    a && ("string" == typeof a && (a = [a]), z(a, function(a) {
        if (a in vb) {
            var c = vb[a], d = c[0], e = c[1], f = c[3], c = c[4];
            d.removeEventListener ? d.removeEventListener(e, f, c) : d.detachEvent && d.detachEvent("on" + e, f);
            delete vb[a]
        }
    }))
}
function oe(a) {
    a = a || window.event;
    a = a.target || a.srcElement;
    3 == a.nodeType && (a = a.parentNode);
    return a
};
function pe(a) {
    var b;
    b = b || 0;
    return function() {
        return a.apply(this, Array.prototype.slice.call(arguments, 0, b))
    }
};
var qe = "StopIteration"in l ? l.StopIteration: Error("StopIteration");
function re() {}
re.prototype.next = function() {
    throw qe;
};
re.prototype.fb = function() {
    return this
};
function se(a) {
    if (a instanceof re)
        return a;
    if ("function" == typeof a.fb)
        return a.fb(!1);
    if (ea(a)) {
        var b = 0, c = new re;
        c.next = function() {
            for (; ;) {
                if (b >= a.length)
                    throw qe;
                if (b in a)
                    return a[b++];
                b++
            }
        };
        return c
    }
    throw Error("Not implemented");
}
function te(a, b, c) {
    if (ea(a))
        try {
            z(a, b, c)
    } catch (d) {
        if (d !== qe)
            throw d;
    } else {
        a = se(a);
        try {
            for (; ;)
                b.call(c, a.next(), void 0, a)
            } catch (e) {
            if (e !== qe)
                throw e;
        }
    }
}
function ue(a) {
    if (ea(a))
        return bb(a);
    a = se(a);
    var b = [];
    te(a, function(a) {
        b.push(a)
    });
    return b
};
function ve(a, b) {
    this.j = {};
    this.g = [];
    this.B = this.k = 0;
    var c = arguments.length;
    if (1 < c) {
        if (c%2)
            throw Error("Uneven number of arguments");
        for (var d = 0; d < c; d += 2)
            this.set(arguments[d], arguments[d + 1])
    } else if (a) {
        a instanceof ve ? (c = a.Ca(), d = a.ta()) : (c = tb(a), d = sb(a));
        for (var e = 0; e < c.length; e++)
            this.set(c[e], d[e])
    }
}
g = ve.prototype;
g.Ga = function() {
    return this.k
};
g.ta = function() {
    we(this);
    for (var a = [], b = 0; b < this.g.length; b++)
        a.push(this.j[this.g[b]]);
    return a
};
g.Ca = function() {
    we(this);
    return this.g.concat()
};
g.equals = function(a, b) {
    if (this === a)
        return !0;
    if (this.k != a.Ga())
        return !1;
    var c = b || xe;
    we(this);
    for (var d, e = 0; d = this.g[e]; e++)
        if (!c(this.get(d), a.get(d)))
            return !1;
    return !0
};
function xe(a, b) {
    return a === b
}
g.isEmpty = function() {
    return 0 == this.k
};
g.clear = function() {
    this.j = {};
    this.B = this.k = this.g.length = 0
};
g.remove = function(a) {
    return ye(this.j, a) ? (delete this.j[a], this.k--, this.B++, this.g.length > 2 * this.k && we(this), !0) : !1
};
function we(a) {
    if (a.k != a.g.length) {
        for (var b = 0, c = 0; b < a.g.length;) {
            var d = a.g[b];
            ye(a.j, d) && (a.g[c++] = d);
            b++
        }
        a.g.length = c
    }
    if (a.k != a.g.length) {
        for (var e = {}, c = b = 0; b < a.g.length;)
            d = a.g[b], ye(e, d) || (a.g[c++] = d, e[d] = 1), b++;
        a.g.length = c
    }
}
g.get = function(a, b) {
    return ye(this.j, a) ? this.j[a] : b
};
g.set = function(a, b) {
    ye(this.j, a) || (this.k++, this.g.push(a), this.B++);
    this.j[a] = b
};
g.forEach = function(a, b) {
    for (var c = this.Ca(), d = 0; d < c.length; d++) {
        var e = c[d], f = this.get(e);
        a.call(b, f, e, this)
    }
};
g.clone = function() {
    return new ve(this)
};
g.fb = function(a) {
    we(this);
    var b = 0, c = this.g, d = this.j, e = this.B, f = this, h = new re;
    h.next = function() {
        for (; ;) {
            if (e != f.B)
                throw Error("The map has changed since the iterator was created");
            if (b >= c.length)
                throw qe;
            var h = c[b++];
            return a ? h : d[h]
        }
    };
    return h
};
function ye(a, b) {
    return Object.prototype.hasOwnProperty.call(a, b)
};
function ze(a, b, c) {
    for (var d = a.elements, e, f = 0; e = d[f]; f++)
        if (e.form == a&&!e.disabled && "fieldset" != e.tagName.toLowerCase()) {
            var h = e.name;
            switch (e.type.toLowerCase()) {
            case "file":
            case "submit":
            case "reset":
            case "button":
                break;
            case "select-multiple":
                e = Ae(e);
                if (null != e)
                    for (var k, m = 0; k = e[m]; m++)
                        c(b, h, k);
                        break;
                    default:
                        k = Ae(e), null != k && c(b, h, k)
                    }
        }
    d = a.getElementsByTagName("input");
    for (f = 0; e = d[f]; f++)
        e.form == a && "image" == e.type.toLowerCase() && (h = e.name, c(b, h, e.value), c(b, h + ".x", "0"), c(b, h + ".y", "0"))
    }
function Be(a, b, c) {
    var d = a.get(b);
    d || (d = [], a.set(b, d));
    d.push(c)
}
function Ce(a, b, c) {
    a.push(encodeURIComponent(b) + "=" + encodeURIComponent(c))
}
function Ae(a) {
    var b = a.type;
    if (!n(b))
        return null;
    switch (b.toLowerCase()) {
    case "checkbox":
    case "radio":
        return a.checked ? a.value : null;
    case "select-one":
        return b = a.selectedIndex, 0 <= b ? a.options[b].value : null;
    case "select-multiple":
        for (var b = [], c, d = 0; c = a.options[d]; d++)
            c.selected && b.push(c.value);
        return b.length ? b : null;
    default:
        return n(a.value) ? a.value : null
    }
};
function De(a) {
    a = String(a);
    if (/^\s*$/.test(a) ? 0 : /^[\],:{}\s\u2028\u2029]*$/.test(a.replace(/\\["\\\/bfnrtu]/g, "@").replace(/"[^"\\\n\r\u2028\u2029\x00-\x08\x0a-\x1f]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:[\s\u2028\u2029]*\[)+/g, "")))
        try {
            return eval("(" + a + ")")
    } catch (b) {}
    throw Error("Invalid JSON string: " + a);
}
function Ee(a) {
    return eval("(" + a + ")")
}
function S(a) {
    return Fe(new Ge(void 0), a)
}
function Ge(a) {
    this.g = a
}
function Fe(a, b) {
    var c = [];
    He(a, b, c);
    return c.join("")
}
function He(a, b, c) {
    switch (typeof b) {
    case "string":
        Ie(b, c);
        break;
    case "number":
        c.push(isFinite(b)&&!isNaN(b) ? b : "null");
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
        if (da(b)) {
            var d = b.length;
            c.push("[");
            for (var e = "", f = 0; f < d; f++)
                c.push(e), e = b[f], He(a, a.g ? a.g.call(b, String(f), e) : e, c), e = ",";
            c.push("]");
            break
        }
        c.push("{");
        d = "";
        for (f in b)
            Object.prototype.hasOwnProperty.call(b, f) && (e = b[f], "function" != typeof e && (c.push(d), Ie(f, c),
            c.push(":"), He(a, a.g ? a.g.call(b, f, e) : e, c), d = ","));
        c.push("}");
        break;
    case "function":
        break;
    default:
        throw Error("Unknown type: " + typeof b);
    }
}
var Je = {
    '"': '\\"',
    "\\": "\\\\",
    "/": "\\/",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\x0B": "\\u000b"
}, Ke = /\uffff/.test("\uffff") ? /[\\\"\x00-\x1f\x7f-\uffff]/g: /[\\\"\x00-\x1f\x7f-\xff]/g;
function Ie(a, b) {
    b.push('"', a.replace(Ke, function(a) {
        if (a in Je)
            return Je[a];
        var b = a.charCodeAt(0), e = "\\u";
        16 > b ? e += "000" : 256 > b ? e += "00" : 4096 > b && (e += "0");
        return Je[a] = e + b.toString(16)
    }), '"')
};
var Le = /^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#(.*))?$/;
function Me(a) {
    if (Ne) {
        Ne=!1;
        var b = l.location;
        if (b) {
            var c = b.href;
            if (c && (c = Oe(c)) && c != b.hostname)
                throw Ne=!0, Error();
        }
    }
    return a.match(Le)
}
var Ne = Ac;
function Oe(a) {
    return (a = Me(a)[3] || null) ? decodeURI(a) : a
}
function Pe() {
    var a = window.location.href, b = a.indexOf("#");
    return 0 > b ? null : a.substr(b + 1)
}
function Qe(a) {
    var b = Me(a);
    a = b[5];
    var c = b[6], b = b[7], d = "";
    a && (d += a);
    c && (d += "?" + c);
    b && (d += "#" + b);
    return d
}
function Re(a) {
    var b = a.indexOf("#");
    return 0 > b ? a : a.substr(0, b)
}
function Se(a, b, c) {
    if (da(b))
        for (var d = 0; d < b.length; d++)
            Se(a, String(b[d]), c);
    else 
        null != b && c.push("&", a, "" === b ? "" : "=", encodeURIComponent(String(b)))
}
function Te(a, b) {
    for (var c in b)
        Se(c, b[c], a);
    return a
}
function Ue(a) {
    a = Te([], a);
    a[0] = "";
    return a.join("")
}
function Ve(a, b) {
    var c = Te([a], b);
    if (c[1]) {
        var d = c[0], e = d.indexOf("#");
        0 <= e && (c.push(d.substr(e)), c[0] = d = d.substr(0, e));
        e = d.indexOf("?");
        0 > e ? c[1] = "?" : e == d.length - 1 && (c[1] = void 0)
    }
    return c.join("")
};
function We(a) {
    "?" == a.charAt(0) && (a = a.substr(1));
    a = a.split("&");
    for (var b = {}, c = 0, d = a.length; c < d; c++) {
        var e = a[c].split("=");
        if (1 == e.length && e[0] || 2 == e.length) {
            var f = qa(e[0] || ""), e = qa(e[1] || "");
            f in b ? da(b[f]) ? cb(b[f], e) : b[f] = [b[f], e] : b[f] = e
        }
    }
    return b
}
function Xe(a) {
    return - 1 != a.indexOf("?") ? (a = (a || "").split("#")[0], a = a.split("?", 2), We(1 < a.length ? a[1] : a[0])) : {}
}
var Ye = Oe;
function Ze(a, b) {
    var c = a.split("#", 2);
    a = c[0];
    var c = 1 < c.length ? "#" + c[1]: "", d = a.split("?", 2);
    a = d[0];
    var d = We(d[1] || ""), e;
    for (e in b)
        d[e] = b[e];
    return Ve(a, d) + c
}
function $e(a) {
    a = Ye(a);
    a = null === a ? null : a.split(".").reverse();
    return (null === a?!1 : "com" == a[0] && a[1].match(/^youtube(?:-nocookie)?$/)?!0 : !1) || (null === a?!1 : "google" == a[1]?!0 : "google" == a[2] ? "au" == a[0] && "com" == a[1]?!0 : "uk" == a[0] && "co" == a[1]?!0 : !1 : !1)
};
var af = null;
"undefined" != typeof XMLHttpRequest ? af = function() {
    return new XMLHttpRequest
} : "undefined" != typeof ActiveXObject && (af = function() {
    return new ActiveXObject("Microsoft.XMLHTTP")
});
function bf(a, b, c, d, e, f, h) {
    function k() {
        4 == (m && "readyState"in m ? m.readyState : 0) && b && Qb(b)(m)
    }
    var m = af && af();
    if (!("open"in m))
        return null;
    "onloadend"in m ? m.addEventListener("loadend", k, !1) : m.onreadystatechange = k;
    c = (c || "GET").toUpperCase();
    d = d || "";
    m.open(c, a, !0);
    f && (m.responseType = f);
    h && (m.withCredentials=!0);
    f = "POST" == c;
    if (e = cf(a, e))
        for (var p in e)
            m.setRequestHeader(p, e[p]), "content-type" == p.toLowerCase() && (f=!1);
    f && m.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    m.send(d);
    return m
}
function cf(a, b) {
    b = b || {};
    for (var c in df) {
        var d = F(df[c]), e;
        if (e = d) {
            e = a;
            var f = void 0;
            f = window.location.href;
            var h = Me(e)[1] || null, k = Ye(e);
            h && k ? (e = Me(e), f = Me(f), e = e[3] == f[3] && e[1] == f[1] && e[4] == f[4]) : e = k ? Ye(f) == k && (Number(Me(f)[4] || null) || null) == (Number(Me(e)[4] || null) || null) : !0;
            e || (e = c, f = F("CORS_HEADER_WHITELIST") || {}, e = (h = Ye(a)) ? (f = f[h]) ? Ta(f, e) : !1 : !0)
        }
        e && (b[c] = d)
    }
    return b
}
function ef(a, b) {
    var c = F("XSRF_FIELD_NAME"), d;
    b.headers && (d = b.headers["Content-Type"]);
    return !b.hf && (!Ye(a) || Ye(a) == document.location.hostname) && "POST" == b.method && (!d || "application/x-www-form-urlencoded" == d)&&!(b.ca && b.ca[c])
}
function ff(a, b) {
    var c = b.format || "JSON";
    b.gf && (a = document.location.protocol + "//" + document.location.hostname + (document.location.port ? ":" + document.location.port : "") + a);
    var d = F("XSRF_FIELD_NAME"), e = F("XSRF_TOKEN"), f = b.$a;
    f && (f[d] && delete f[d], a = Ze(a, f));
    var h = b.ae || "", f = b.ca;
    ef(a, b) && (f || (f = {}), f[d] = e);
    f && t(h) && (d = We(h), Cb(d, f), h = Ue(d));
    var k=!1, m, p = bf(a, function(a) {
        if (!k) {
            k=!0;
            m && H(m);
            var d;
            t:
            switch (a && "status"in a ? a.status : - 1) {
            case 200:
            case 201:
            case 202:
            case 203:
            case 204:
            case 205:
            case 206:
            case 304:
                d =
                !0;
                break t;
            default:
                d=!1
            }
            var e = null;
            if (d || 400 <= a.status && 500 > a.status)
                e = gf(c, a);
            if (d)
                t: {
                switch (c) {
                case "XML":
                    d = 0 == parseInt(e && e.return_code, 10);
                    break t;
                case "RAW":
                    d=!0;
                    break t
                }
                d=!!e
            }
            var e = e || {}, f = b.context || l;
            d ? b.V && b.V.call(f, a, e) : b.onError && b.onError.call(f, a, e);
            b.yc && b.yc.call(f, a, e)
        }
    }, b.method, h, b.headers, b.responseType, b.withCredentials);
    b.wb && 0 < b.timeout && (m = G(function() {
        k || (k=!0, p.abort(), H(m), b.wb.call(b.context || l, p))
    }, b.timeout));
    return p
}
function gf(a, b) {
    var c = null;
    switch (a) {
    case "JSON":
        var d = b.responseText, e = b.getResponseHeader("Content-Type") || "";
        d && 0 <= e.indexOf("json") && (c = Ee(d));
        break;
    case "XML":
        if (d = (d = b.responseXML) ? hf(d) : null)
            c = {}, z(d.getElementsByTagName("*"), function(a) {
                c[a.tagName] = jf(a)
            })
    }
    return c
}
function hf(a) {
    return a ? (a = ("responseXML"in a ? a.responseXML : a).getElementsByTagName("root")) && 0 < a.length ? a[0] : null : null
}
function jf(a) {
    var b = "";
    z(a.childNodes, function(a) {
        b += a.nodeValue
    });
    return b
}
var df = {
    "X-YouTube-Page-CL": "PAGE_CL",
    "X-YouTube-Page-Timestamp": "PAGE_BUILD_TIMESTAMP",
    "X-YouTube-Variants-Checksum": "VARIANTS_CHECKSUM"
};
Db("area base br col command embed hr img input keygen link meta param source track wbr".split(" "));
Db("action", "cite", "data", "formaction", "href", "manifest", "poster", "src");
Db("embed", "iframe", "link", "script", "style", "template");
function kf(a, b) {
    (a = O(a)) && a.style && (a.style.display = b ? "" : "none", mb(a, "hid", !b))
}
function lf(a) {
    return (a = O(a))?!("none" == a.style.display || B(a, "hid")) : !1
}
function mf(a) {
    if (a = O(a))
        lf(a) ? (a.style.display = "none", C(a, "hid")) : (a.style.display = "", D(a, "hid"))
}
function nf(a, b) {
    if (a = O(a))
        a.style.visibility = b ? "visible" : "hidden"
}
function of(a) {
    z(arguments, function(a) {
        kf(a, !0)
    })
}
function pf(a) {
    z(arguments, function(a) {
        kf(a, !1)
    })
}
function qf(a) {
    z(arguments, mf)
}
var rf = {};
function sf(a, b, c) {
    if ((a = O(a)) && a.style) {
        if (b in rf)
            b = rf[b];
        else {
            var d;
            if ((d = document.body.style) && b in d)
                d = b;
            else {
                var e = Bd();
                e ? (e = e.toLowerCase(), e += Ia(b), d=!n(d) || e in d ? e : null) : d = null
            }
            b = rf[b] = d
        }
        b && (a.style[b] = c)
    }
};
function tf(a) {
    tf[" "](a);
    return a
}
tf[" "] = s;
var uf=!N || Jc(9), vf = N&&!Ic("9");
!Ac || Ic("528");
zc && Ic("1.9b") || N && Ic("8") || yc && Ic("9.5") || Ac && Ic("528");
zc&&!Ic("8") || N && Ic("9");
function wf(a, b) {
    this.type = a;
    this.currentTarget = this.target = b;
    this.defaultPrevented = this.j=!1;
    this.Nc=!0
}
wf.prototype.dispose = function() {};
wf.prototype.stopPropagation = function() {
    this.j=!0
};
wf.prototype.preventDefault = function() {
    this.defaultPrevented=!0;
    this.Nc=!1
};
var xf = Ac ? "webkitTransitionEnd": yc ? "otransitionend": "transitionend";
function yf(a, b) {
    wf.call(this, a ? a.type : "");
    this.relatedTarget = this.currentTarget = this.target = null;
    this.charCode = this.keyCode = this.button = this.clientY = this.clientX = 0;
    this.shiftKey = this.altKey = this.ctrlKey=!1;
    this.g = this.state = null;
    a && this.init(a, b)
}
y(yf, wf);
yf.prototype.init = function(a, b) {
    var c = this.type = a.type;
    this.target = a.target || a.srcElement;
    this.currentTarget = b;
    var d = a.relatedTarget;
    if (d) {
        if (zc) {
            var e;
            t:
            {
                try {
                    tf(d.nodeName);
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
    this.clientX = void 0 !== a.clientX ? a.clientX : a.pageX;
    this.clientY = void 0 !== a.clientY ? a.clientY : a.pageY;
    this.button = a.button;
    this.keyCode = a.keyCode || 0;
    this.charCode = a.charCode || ("keypress" == c ? a.keyCode : 0);
    this.ctrlKey = a.ctrlKey;
    this.altKey = a.altKey;
    this.shiftKey = a.shiftKey;
    this.state = a.state;
    this.g = a;
    a.defaultPrevented && this.preventDefault()
};
yf.prototype.stopPropagation = function() {
    yf.J.stopPropagation.call(this);
    this.g.stopPropagation ? this.g.stopPropagation() : this.g.cancelBubble=!0
};
yf.prototype.preventDefault = function() {
    yf.J.preventDefault.call(this);
    var a = this.g;
    if (a.preventDefault)
        a.preventDefault();
    else if (a.returnValue=!1, vf)
        try {
            if (a.ctrlKey || 112 <= a.keyCode && 123 >= a.keyCode)
                a.keyCode =- 1
    } catch (b) {}
};
yf.prototype.k = function() {
    return this.g
};
var zf = "closure_listenable_" + (1E6 * Math.random() | 0), Af = 0;
function Bf(a, b, c, d, e) {
    this.listener = a;
    this.proxy = null;
    this.src = b;
    this.type = c;
    this.Ub=!!d;
    this.Yb = e;
    this.key=++Af;
    this.removed = this.Vb=!1
}
function Cf(a) {
    a.removed=!0;
    a.listener = null;
    a.proxy = null;
    a.src = null;
    a.Yb = null
};
function Df(a) {
    this.src = a;
    this.g = {};
    this.j = 0
}
Df.prototype.add = function(a, b, c, d, e) {
    var f = a.toString();
    a = this.g[f];
    a || (a = this.g[f] = [], this.j++);
    var h = Ef(a, b, d, e);
    - 1 < h ? (b = a[h], c || (b.Vb=!1)) : (b = new Bf(b, this.src, f, !!d, e), b.Vb = c, a.push(b));
    return b
};
Df.prototype.remove = function(a, b, c, d) {
    a = a.toString();
    if (!(a in this.g))
        return !1;
    var e = this.g[a];
    b = Ef(e, b, c, d);
    return - 1 < b ? (Cf(e[b]), Za(e, b), 0 == e.length && (delete this.g[a], this.j--), !0) : !1
};
function Ff(a, b) {
    var c = b.type;
    if (!(c in a.g))
        return !1;
    var d = Ya(a.g[c], b);
    d && (Cf(b), 0 == a.g[c].length && (delete a.g[c], a.j--));
    return d
}
Df.prototype.removeAll = function(a) {
    a = a && a.toString();
    var b = 0, c;
    for (c in this.g)
        if (!a || c == a) {
            for (var d = this.g[c], e = 0; e < d.length; e++)
                ++b, Cf(d[e]);
                delete this.g[c];
                this.j--
        }
    return b
};
function Gf(a, b, c, d, e) {
    a = a.g[b.toString()];
    b =- 1;
    a && (b = Ef(a, c, d, e));
    return - 1 < b ? a[b] : null
}
function Ef(a, b, c, d) {
    for (var e = 0; e < a.length; ++e) {
        var f = a[e];
        if (!f.removed && f.listener == b && f.Ub==!!c && f.Yb == d)
            return e
    }
    return - 1
};
var Hf = "closure_lm_" + (1E6 * Math.random() | 0), If = {}, Jf = 0;
function Kf(a, b, c, d, e) {
    if (da(b)) {
        for (var f = 0; f < b.length; f++)
            Kf(a, b[f], c, d, e);
        return null
    }
    c = Lf(c);
    if (a && a[zf])
        a = a.listen(b, c, d, e);
    else {
        if (!b)
            throw Error("Invalid event type");
        var f=!!d, h = Mf(a);
        h || (a[Hf] = h = new Df(a));
        c = h.add(b, c, !1, d, e);
        c.proxy || (d = Nf(), c.proxy = d, d.src = a, d.listener = c, a.addEventListener ? a.addEventListener(b.toString(), d, f) : a.attachEvent(Of(b.toString()), d), Jf++);
        a = c
    }
    return a
}
function Nf() {
    var a = Pf, b = uf ? function(c) {
        return a.call(b.src, b.listener, c)
    }
    : function(c) {
        c = a.call(b.src, b.listener, c);
        if (!c)
            return c
    };
    return b
}
function Qf(a, b, c, d, e) {
    if (da(b))
        for (var f = 0; f < b.length; f++)
            Qf(a, b[f], c, d, e);
    else 
        c = Lf(c), a && a[zf] ? a.oc(b, c, d, e) : a && (a = Mf(a)) && (b = Gf(a, b, c, !!d, e)) && Rf(b)
}
function Rf(a) {
    if ("number" == typeof a ||!a || a.removed)
        return !1;
    var b = a.src;
    if (b && b[zf])
        return Ff(b.Oa, a);
    var c = a.type, d = a.proxy;
    b.removeEventListener ? b.removeEventListener(c, d, a.Ub) : b.detachEvent && b.detachEvent(Of(c), d);
    Jf--;
    (c = Mf(b)) ? (Ff(c, a), 0 == c.j && (c.src = null, b[Hf] = null)) : Cf(a);
    return !0
}
function Of(a) {
    return a in If ? If[a] : If[a] = "on" + a
}
function Sf(a, b, c, d) {
    var e = 1;
    if (a = Mf(a))
        if (b = a.g[b.toString()])
            for (b = b.concat(), a = 0; a < b.length; a++) {
                var f = b[a];
                f && f.Ub == c&&!f.removed && (e&=!1 !== Tf(f, d))
            }
    return Boolean(e)
}
function Tf(a, b) {
    var c = a.listener, d = a.Yb || a.src;
    a.Vb && Rf(a);
    return c.call(d, b)
}
function Pf(a, b) {
    if (a.removed)
        return !0;
    if (!uf) {
        var c = b || r("window.event"), d = new yf(c, this), e=!0;
        if (!(0 > c.keyCode || void 0 != c.returnValue)) {
            t:
            {
                var f=!1;
                if (0 == c.keyCode)
                    try {
                        c.keyCode =- 1;
                        break t
                } catch (h) {
                    f=!0
                }
                if (f || void 0 == c.returnValue)
                    c.returnValue=!0
            }
            c = [];
            for (f = d.currentTarget; f; f = f.parentNode)
                c.push(f);
            for (var f = a.type, k = c.length - 1; !d.j && 0 <= k; k--)
                d.currentTarget = c[k], e&=Sf(c[k], f, !0, d);
            for (k = 0; !d.j && k < c.length; k++)
                d.currentTarget = c[k], e&=Sf(c[k], f, !1, d)
            }
        return e
    }
    return Tf(a, new yf(b, this))
}
function Mf(a) {
    a = a[Hf];
    return a instanceof Df ? a : null
}
var Uf = "__closure_events_fn_" + (1E9 * Math.random()>>>0);
function Lf(a) {
    if (fa(a))
        return a;
    a[Uf] || (a[Uf] = function(b) {
        return a.handleEvent(b)
    });
    return a[Uf]
};
function Vf(a) {
    Ib.call(this);
    this.j = a;
    this.g = {}
}
y(Vf, Ib);
var Wf = [];
g = Vf.prototype;
g.listen = function(a, b, c, d) {
    da(b) || (b && (Wf[0] = b.toString()), b = Wf);
    for (var e = 0; e < b.length; e++) {
        var f = Kf(a, b[e], c || this.handleEvent, d ||!1, this.j || this);
        if (!f)
            break;
        this.g[f.key] = f
    }
    return this
};
g.oc = function(a, b, c, d, e) {
    if (da(b))
        for (var f = 0; f < b.length; f++)
            this.oc(a, b[f], c, d, e);
    else 
        c = c || this.handleEvent, e = e || this.j || this, c = Lf(c), d=!!d, b = a && a[zf] ? Gf(a.Oa, String(b), c, d, e) : a ? (a = Mf(a)) ? Gf(a, b, c, d, e) : null : null, b && (Rf(b), delete this.g[b.key]);
    return this
};
g.removeAll = function() {
    pb(this.g, Rf);
    this.g = {}
};
g.G = function() {
    Vf.J.G.call(this);
    this.removeAll()
};
g.handleEvent = function() {
    throw Error("EventHandler.handleEvent not implemented");
};
function Xf() {
    Ib.call(this);
    this.Oa = new Df(this);
    this.Ge = this;
    this.na = null
}
y(Xf, Ib);
Xf.prototype[zf]=!0;
g = Xf.prototype;
g.vc = function(a) {
    this.na = a
};
g.addEventListener = function(a, b, c, d) {
    Kf(this, a, b, c, d)
};
g.removeEventListener = function(a, b, c, d) {
    Qf(this, a, b, c, d)
};
function Yf(a, b) {
    var c, d = a.na;
    if (d) {
        c = [];
        for (var e = 1; d; d = d.na)
            c.push(d), ++e
    }
    var d = a.Ge, e = b, f = e.type || e;
    if (t(e))
        e = new wf(e, d);
    else if (e instanceof wf)
        e.target = e.target || d;
    else {
        var h = e, e = new wf(f, d);
        Cb(e, h)
    }
    var h=!0, k;
    if (c)
        for (var m = c.length - 1; !e.j && 0 <= m; m--)
            k = e.currentTarget = c[m], h = Zf(k, f, !0, e) && h;
    e.j || (k = e.currentTarget = d, h = Zf(k, f, !0, e) && h, e.j || (h = Zf(k, f, !1, e) && h));
    if (c)
        for (m = 0; !e.j && m < c.length; m++)
            k = e.currentTarget = c[m], h = Zf(k, f, !1, e) && h
}
g.G = function() {
    Xf.J.G.call(this);
    this.removeAllListeners();
    this.na = null
};
g.listen = function(a, b, c, d) {
    return this.Oa.add(String(a), b, !1, c, d)
};
g.oc = function(a, b, c, d) {
    return this.Oa.remove(String(a), b, c, d)
};
g.removeAllListeners = function(a) {
    return this.Oa ? this.Oa.removeAll(a) : 0
};
function Zf(a, b, c, d) {
    b = a.Oa.g[String(b)];
    if (!b)
        return !0;
    b = b.concat();
    for (var e=!0, f = 0; f < b.length; ++f) {
        var h = b[f];
        if (h&&!h.removed && h.Ub == c) {
            var k = h.listener, m = h.Yb || h.src;
            h.Vb && Ff(a.Oa, h);
            e=!1 !== k.call(m, d) && e
        }
    }
    return e && 0 != d.Nc
};
function $f() {}
ba($f);
$f.prototype.g = 0;
function ag(a) {
    Xf.call(this);
    this.P = a || Qc();
    this.Ba = null;
    this.nb=!1;
    this.g = null;
    this.j = void 0;
    this.O = this.X = this.Z = null;
    this.rb=!1
}
y(ag, Xf);
g = ag.prototype;
g.Ae = $f.getInstance();
g.getId = function() {
    return this.Ba || (this.Ba = ":" + (this.Ae.g++).toString(36))
};
g.wa = function() {
    return this.g
};
function bg(a, b) {
    return a.g ? Q(b, a.g || a.P.g) : null
}
function cg(a) {
    a.j || (a.j = new Vf(a));
    return a.j
}
g.vc = function(a) {
    if (this.Z && this.Z != a)
        throw Error("Method not supported");
    ag.J.vc.call(this, a)
};
function dg(a, b) {
    if (a.nb)
        throw Error("Component already rendered");
    if (b) {
        a.rb=!0;
        var c = Sc(b);
        a.P && a.P.g == c || (a.P = Qc(b));
        a.g = b;
        a.Ja()
    } else 
        throw Error("Invalid element to decorate");
}
g.Ja = function() {
    this.nb=!0;
    eg(this, function(a) {
        !a.nb && a.wa() && a.Ja()
    })
};
g.Da = function() {
    eg(this, function(a) {
        a.nb && a.Da()
    });
    this.j && this.j.removeAll();
    this.nb=!1
};
g.G = function() {
    this.nb && this.Da();
    this.j && (this.j.dispose(), delete this.j);
    eg(this, function(a) {
        a.dispose()
    });
    !this.rb && this.g && jd(this.g);
    this.Z = this.g = this.O = this.X = null;
    ag.J.G.call(this)
};
function eg(a, b) {
    a.X && z(a.X, b, void 0)
}
g.removeChild = function(a, b) {
    if (a) {
        var c = t(a) ? a: a.getId(), d;
        this.O && c ? (d = this.O, d = (c in d ? d[c] : void 0) || null) : d = null;
        a = d;
        if (c && a) {
            d = this.O;
            c in d && delete d[c];
            Ya(this.X, a);
            b && (a.Da(), a.g && jd(a.g));
            c = a;
            if (null == c)
                throw Error("Unable to set parent component");
            c.Z = null;
            ag.J.vc.call(c, null)
        }
    }
    if (!a)
        throw Error("Child is not in parent component");
    return a
};
function fg(a) {
    ag.call(this, a);
    this.Aa = [];
    this.Ib = []
}
y(fg, ag);
fg.prototype.Da = function() {
    z(this.Aa, le);
    bc(this.Ib);
    fg.J.Da.call(this)
};
function gg() {
    fg.call(this)
}
y(gg, fg);
gg.prototype.Ja = function() {
    gg.J.Ja.call(this);
    this.D = E(this.wa(), "overflowable-list-orientation") || "horizontal";
    this.B = bg(this, "parent-list");
    this.C = bg(this, "overflow-container");
    this.k = bg(this, "overflow-list");
    bg(this, "overflowable-list-item");
    this.A = bg(this, "overflowable-list-more-button")
};
function hg() {
    var a = Uc("overflowable-list-root", ig), b = [];
    z(a, function(a) {
        var d = new gg;
        dg(d, a);
        b.push(d)
    });
    return b
}
function jg(a, b) {
    var c = kg(a), d = kg(b);
    return c < d?-1 : c == d ? 0 : 1
}
function lg(a, b) {
    var c = bb(Uc("overflowable-list-item", b));
    return Oa(c, function(a, b) {
        return a + mg(this, b)
    }, 0, a)
}
function mg(a, b) {
    return "vertical" == a.D ? Qd(b).height : Qd(b).width
}
function ng(a) {
    var b = Q("overflowable-list-item", a.wa());
    return b ? mg(a, b) : 0
}
function kg(a) {
    return lg(a, a.B) + lg(a, a.k)
};
function og(a, b, c, d, e, f, h) {
    var k, m;
    if (k = c.offsetParent) {
        var p = "HTML" == k.tagName || "BODY" == k.tagName;
        p && "static" == Gd(k, "position") || (m = Kd(k), p || (p = (p = Ld(k)) && zc?-k.scrollLeft : !p || N && Ic("8") || "visible" == Gd(k, "overflowX") ? k.scrollLeft : k.scrollWidth - k.clientWidth - k.scrollLeft, m = tc(m, new L(p, k.scrollTop))))
    }
    k = m || new L;
    m = Sd(a);
    if (p = Jd(a)) {
        var x = new Dd(p.left, p.top, p.right - p.left, p.bottom - p.top), p = Math.max(m.left, x.left), P = Math.min(m.left + m.width, x.left + x.width);
        if (p <= P) {
            var I = Math.max(m.top, x.top), x = Math.min(m.top +
            m.height, x.top + x.height);
            I <= x && (m.left = p, m.top = I, m.width = P - p, m.height = x - I)
        }
    }
    p = Qc(a);
    I = Qc(c);
    if (p.g != I.g) {
        var P = p.g.body, I = bd(I.g), x = new L(0, 0), W;
        W = (W = Sc(P)) ? bd(W) : window;
        var ra = P;
        do {
            var lb = W == I ? Kd(ra): Md(ra);
            x.x += lb.x;
            x.y += lb.y
        }
        while (W && W != I && (ra = W.frameElement) && (W = W.parent));
        P = tc(x, Kd(P));
        !N || Jc(9) || zd(p) || (P = tc(P, Ad(p)));
        m.left += P.x;
        m.top += P.y
    }
    a = pg(a, b);
    b = new L(a & 2 ? m.left + m.width : m.left, a & 1 ? m.top + m.height : m.top);
    b = tc(b, k);
    e && (b.x += (a & 2?-1 : 1) * e.x, b.y += (a & 1?-1 : 1) * e.y);
    var M;
    h && (M = Jd(c)) && (M.top -=
    k.y, M.right -= k.x, M.bottom -= k.y, M.left -= k.x);
    e = M;
    M = b.clone();
    b = pg(c, d);
    d = Qd(c);
    a = d.clone();
    M = M.clone();
    a = a.clone();
    k = 0;
    if (f || 0 != b)
        b & 2 ? M.x -= a.width + (f ? f.right : 0) : f && (M.x += f.left), b & 1 ? M.y -= a.height + (f ? f.bottom : 0) : f && (M.y += f.top);
    h && (e ? (f = M, b = a, k = 0, 65 == (h & 65) && (f.x < e.left || f.x >= e.right) && (h&=-2), 132 == (h & 132) && (f.y < e.top || f.y >= e.bottom) && (h&=-5), f.x < e.left && h & 1 && (f.x = e.left, k|=1), f.x < e.left && f.x + b.width > e.right && h & 16 && (b.width = Math.max(b.width - (f.x + b.width - e.right), 0), k|=4), f.x + b.width > e.right && h & 1 && (f.x =
    Math.max(e.right - b.width, e.left), k|=1), h & 2 && (k = k | (f.x < e.left ? 16 : 0) | (f.x + b.width > e.right ? 32 : 0)), f.y < e.top && h & 4 && (f.y = e.top, k|=2), f.y <= e.top && f.y + b.height < e.bottom && h & 32 && (b.height = Math.max(b.height - (e.top - f.y), 0), f.y = e.top, k|=8), f.y >= e.top && f.y + b.height > e.bottom && h & 32 && (b.height = Math.max(b.height - (f.y + b.height - e.bottom), 0), k|=8), f.y + b.height > e.bottom && h & 4 && (f.y = Math.max(e.bottom - b.height, e.top), k|=2), h & 8 && (k = k | (f.y < e.top ? 64 : 0) | (f.y + b.height > e.bottom ? 128 : 0)), h = k) : h = 256, k = h);
    f = new Dd(0, 0, 0, 0);
    f.left =
    M.x;
    f.top = M.y;
    f.width = a.width;
    f.height = a.height;
    h = k;
    h & 496 || (a = new L(f.left, f.top), M = zc && (Bc || Ec) && Ic("1.9"), a instanceof L ? (e = a.x, a = a.y) : (e = a, a = void 0), c.style.left = Pd(e, M), c.style.top = Pd(a, M), a = new uc(f.width, f.height), d == a || d && a && d.width == a.width && d.height == a.height || (f = a, d = Sc(c), e = zd(Qc(d)), !N || Ic("10") || e && Ic("8") ? (c = c.style, zc ? c.MozBoxSizing = "border-box" : Ac ? c.WebkitBoxSizing = "border-box" : c.boxSizing = "border-box", c.width = Math.max(f.width, 0) + "px", c.height = Math.max(f.height, 0) + "px") : (d = c.style,
    e ? (e = Vd(c), c = Yd(c), d.pixelWidth = f.width - c.left - e.left - e.right - c.right, d.pixelHeight = f.height - c.top - e.top - e.bottom - c.bottom) : (d.pixelWidth = f.width, d.pixelHeight = f.height))));
    return h
}
function pg(a, b) {
    return (b & 4 && Ld(a) ? b^2 : b)&-5
};
var qg;
var rg = {}, sg = "ontouchstart"in document;
function tg(a, b, c) {
    var d;
    switch (a) {
    case "mouseover":
    case "mouseout":
        d = 3;
        break;
    case "mouseenter":
    case "mouseleave":
        d = 9
    }
    return xd(c, function(a) {
        return B(a, b)
    }, d)
}
function ug(a) {
    var b = "mouseover" == a.type && "mouseenter"in rg || "mouseout" == a.type && "mouseleave"in rg, c = a.type in rg || b;
    if ("HTML" != a.target.tagName && c) {
        if (b) {
            var b = "mouseover" == a.type ? "mouseenter": "mouseleave", c = rg[b], d;
            for (d in c.ja) {
                var e = tg(b, d, a.target);
                e&&!xd(a.relatedTarget, function(a) {
                    return a == e
                }) && c.publish(d, e, b, a)
            }
        }
        if (b = rg[a.type])
            for (d in b.ja)(e = tg(a.type, d, a.target)
                ) && b.publish(d, e, a.type, a)
    }
}
R(document, "blur", ug, !0);
R(document, "change", ug, !0);
R(document, "click", ug);
R(document, "focus", ug, !0);
R(document, "mouseover", ug);
R(document, "mouseout", ug);
R(document, "mousedown", ug);
R(document, "keydown", ug);
R(document, "keyup", ug);
R(document, "keypress", ug);
R(document, "cut", ug);
R(document, "paste", ug);
sg && (R(document, "touchstart", ug), R(document, "touchend", ug), R(document, "touchcancel", ug));
function vg() {}
g = vg.prototype;
g.Ma = function(a) {
    return yd(a, T(this))
};
function T(a, b) {
    return "yt-uix" + (a.Ta ? "-" + a.Ta : "") + (b ? "-" + b : "")
}
g.init = s;
g.dispose = s;
g.Qb = function(a, b, c) {
    var d = this.T(a, b);
    if (d && (d = r(d))) {
        var e = eb(arguments, 2);
        db(e, 0, 0, a);
        d.apply(null, e)
    }
};
g.T = function(a, b) {
    return E(a, b)
};
function wg() {
    this.j = [];
    this.k = {}
}
y(wg, vg);
ba(wg);
wg.prototype.Ta = "button";
wg.prototype.g = null;
function xg(a, b) {
    var c = b.iframeMask;
    c || (c = document.createElement("iframe"), c.src = 'javascript:""', c.className = T(a, "menu-mask"), b.iframeMask = c);
    return c
}
function yg(a, b) {
    if (b) {
        var c = zg(a, b);
        if (c) {
            a.g = null;
            b.setAttribute("aria-pressed", "false");
            b.setAttribute("aria-expanded", "false");
            b.removeAttribute("aria-activedescendant");
            pf(c);
            a.Qb(b, "button-menu-action", !1);
            var d = xg(a, b), e = Zd(c).toString();
            delete a.k[e];
            G(function() {
                d && d.parentNode && d.parentNode.removeChild(d);
                c.originalParentNode && (c.parentNode.removeChild(c), c.originalParentNode.appendChild(c), c.originalParentNode = null, c.activeButtonNode = null)
            }, 1)
        }
        var e = yd(b, T(a, "group")), f = [T(a, "active")];
        e && f.push(T(a, "group-active"));
        kb(b, f);
        K("yt-uix-button-menu-hide", b);
        le(a.j);
        a.j.length = 0
    }
}
function zg(a, b) {
    if (!b.widgetMenu) {
        var c = a.T(b, "button-menu-id"), c = c && O(c), d = T(a, "menu");
        c ? jb(c, [d, T(a, "menu-external")]) : c = Q(d, b);
        b.widgetMenu = c
    }
    return b.widgetMenu
}
function Ag(a, b) {
    if (a.T(b, "button-toggle")) {
        var c = yd(b, T(a, "group")), d = T(a, "toggled"), e = B(b, d);
        if (c && a.T(c, "button-toggle-group")) {
            var f = a.T(c, "button-toggle-group"), c = Uc(T(a), c);
            z(c, function(a) {
                a != b || "optional" == f && e ? (D(a, d), a.removeAttribute("aria-pressed")) : (C(b, d), a.setAttribute("aria-pressed", "true"))
            })
        } else 
            e ? b.removeAttribute("aria-pressed") : b.setAttribute("aria-pressed", "true"), nb(b, d)
    }
};
function Bg(a) {
    l.setTimeout(function() {
        throw a;
    }, 0)
}
var Cg;
function Dg() {
    var a = l.MessageChannel;
    "undefined" === typeof a && "undefined" !== typeof window && window.postMessage && window.addEventListener && (a = function() {
        var a = document.createElement("iframe");
        a.style.display = "none";
        a.src = "";
        document.documentElement.appendChild(a);
        var b = a.contentWindow, a = b.document;
        a.open();
        a.write("");
        a.close();
        var c = "callImmediate" + Math.random(), d = "file:" == b.location.protocol ? "*": b.location.protocol + "//" + b.location.host, a = u(function(a) {
            if (("*" == d || a.origin == d) && a.data == c)
                this.port1.onmessage()
        },
        this);
        b.addEventListener("message", a, !1);
        this.port1 = {};
        this.port2 = {
            postMessage: function() {
                b.postMessage(c, d)
            }
        }
    });
    if ("undefined" !== typeof a&&!Da(vc, "Trident")&&!Da(vc, "MSIE")) {
        var b = new a, c = {}, d = c;
        b.port1.onmessage = function() {
            if (n(c.next)) {
                c = c.next;
                var a = c.md;
                c.md = null;
                a()
            }
        };
        return function(a) {
            d.next = {
                md: a
            };
            d = d.next;
            b.port2.postMessage(0)
        }
    }
    return "undefined" !== typeof document && "onreadystatechange"in document.createElement("script") ? function(a) {
        var b = document.createElement("script");
        b.onreadystatechange =
        function() {
            b.onreadystatechange = null;
            b.parentNode.removeChild(b);
            b = null;
            a();
            a = null
        };
        document.documentElement.appendChild(b)
    } : function(a) {
        l.setTimeout(a, 0)
    }
};
function Eg(a, b) {
    Fg || Gg();
    Hg || (Fg(), Hg=!0);
    Ig.push(new Jg(a, b))
}
var Fg;
function Gg() {
    if (l.Promise && l.Promise.resolve) {
        var a = l.Promise.resolve();
        Fg = function() {
            a.then(Kg)
        }
    } else 
        Fg = function() {
            var a = Kg;
            !fa(l.setImmediate) || l.Window && l.Window.prototype.setImmediate == l.setImmediate ? (Cg || (Cg = Dg()), Cg(a)) : l.setImmediate(a)
        }
}
var Hg=!1, Ig = [];
function Kg() {
    for (; Ig.length;) {
        var a = Ig;
        Ig = [];
        for (var b = 0; b < a.length; b++) {
            var c = a[b];
            try {
                c.g.call(c.scope)
            } catch (d) {
                Bg(d)
            }
        }
    }
    Hg=!1
}
function Jg(a, b) {
    this.g = a;
    this.scope = b
};
function Lg(a, b) {
    this.j = 0;
    this.C = void 0;
    this.g = this.k = null;
    this.B = this.A=!1;
    try {
        var c = this;
        a.call(b, function(a) {
            Mg(c, 2, a)
        }, function(a) {
            Mg(c, 3, a)
        })
    } catch (d) {
        Mg(this, 3, d)
    }
}
function Ng(a) {
    return new Lg(function(b) {
        b(a)
    })
}
Lg.prototype.then = function(a, b, c) {
    return Og(this, fa(a) ? a : null, fa(b) ? b : null, c)
};
Lg.prototype.then = Lg.prototype.then;
Lg.prototype.$goog_Thenable=!0;
Lg.prototype.cancel = function(a) {
    0 == this.j && Eg(function() {
        var b = new Pg(a);
        Qg(this, b)
    }, this)
};
function Qg(a, b) {
    if (0 == a.j)
        if (a.k) {
            var c = a.k;
            if (c.g) {
                for (var d = 0, e =- 1, f = 0, h; h = c.g[f]; f++)
                    if (h = h.Cb)
                        if (d++, h == a && (e = f), 0 <= e && 1 < d)
                            break;
                            0 <= e && (0 == c.j && 1 == d ? Qg(c, b) : (d = c.g.splice(e, 1)[0], Rg(c, d, 3, b)))
                        }
        } else 
            Mg(a, 3, b)
    }
function Sg(a, b) {
    a.g && a.g.length || 2 != a.j && 3 != a.j || Tg(a);
    a.g || (a.g = []);
    a.g.push(b)
}
function Og(a, b, c, d) {
    var e = {
        Cb: null,
        $c: null,
        ad: null
    };
    e.Cb = new Lg(function(a, h) {
        e.$c = b ? function(c) {
            try {
                var e = b.call(d, c);
                a(e)
            } catch (p) {
                h(p)
            }
        } : a;
        e.ad = c ? function(b) {
            try {
                var e = c.call(d, b);
                !n(e) && b instanceof Pg ? h(b) : a(e)
            } catch (p) {
                h(p)
            }
        } : h
    });
    e.Cb.k = a;
    Sg(a, e);
    return e.Cb
}
Lg.prototype.M = function(a) {
    this.j = 0;
    Mg(this, 2, a)
};
Lg.prototype.D = function(a) {
    this.j = 0;
    Mg(this, 3, a)
};
function Mg(a, b, c) {
    if (0 == a.j) {
        if (a == c)
            b = 3, c = new TypeError("Promise cannot resolve to itself");
        else {
            var d;
            if (c)
                try {
                    d=!!c.$goog_Thenable
            } catch (e) {
                d=!1
            } else 
                d=!1;
            if (d) {
                a.j = 1;
                c.then(a.M, a.D, a);
                return 
            }
            if (ga(c))
                try {
                    var f = c.then;
                    if (fa(f)) {
                        Ug(a, c, f);
                        return 
                    }
            } catch (h) {
                b = 3, c = h
            }
        }
        a.C = c;
        a.j = b;
        Tg(a);
        3 != b || c instanceof Pg || Vg(a, c)
    }
}
function Ug(a, b, c) {
    function d(b) {
        f || (f=!0, a.D(b))
    }
    function e(b) {
        f || (f=!0, a.M(b))
    }
    a.j = 1;
    var f=!1;
    try {
        c.call(b, e, d)
    } catch (h) {
        d(h)
    }
}
function Tg(a) {
    a.A || (a.A=!0, Eg(a.$, a))
}
Lg.prototype.$ = function() {
    for (; this.g && this.g.length;) {
        var a = this.g;
        this.g = [];
        for (var b = 0; b < a.length; b++)
            Rg(this, a[b], this.j, this.C)
    }
    this.A=!1
};
function Rg(a, b, c, d) {
    if (2 == c)
        b.$c(d);
    else {
        if (b.Cb)
            for (; a && a.B; a = a.k)
                a.B=!1;
        b.ad(d)
    }
}
function Vg(a, b) {
    a.B=!0;
    Eg(function() {
        a.B && Wg.call(null, b)
    })
}
var Wg = Bg;
function Pg(a) {
    ma.call(this, a)
}
y(Pg, ma);
Pg.prototype.name = "cancel";
function Xg(a, b) {
    Xf.call(this);
    this.g = a || 1;
    this.j = b || l;
    this.k = u(this.Kd, this);
    this.B = w()
}
y(Xg, Xf);
g = Xg.prototype;
g.enabled=!1;
g.la = null;
function Yg(a, b) {
    a.g = b;
    a.la && a.enabled ? (a.stop(), a.start()) : a.la && a.stop()
}
g.Kd = function() {
    if (this.enabled) {
        var a = w() - this.B;
        0 < a && a < .8 * this.g ? this.la = this.j.setTimeout(this.k, this.g - a) : (this.la && (this.j.clearTimeout(this.la), this.la = null), Yf(this, "tick"), this.enabled && (this.la = this.j.setTimeout(this.k, this.g), this.B = w()))
    }
};
g.start = function() {
    this.enabled=!0;
    this.la || (this.la = this.j.setTimeout(this.k, this.g), this.B = w())
};
g.stop = function() {
    this.enabled=!1;
    this.la && (this.j.clearTimeout(this.la), this.la = null)
};
g.G = function() {
    Xg.J.G.call(this);
    this.stop();
    delete this.j
};
function Zg(a, b, c) {
    if (fa(a))
        c && (a = u(a, c));
    else if (a && "function" == typeof a.handleEvent)
        a = u(a.handleEvent, a);
    else 
        throw Error("Invalid listener argument");
    return 2147483647 < b?-1 : l.setTimeout(a, b || 0)
};
function $g(a, b, c) {
    Ib.call(this);
    this.B = a;
    this.k = b;
    this.j = c;
    this.g = u(this.Jd, this)
}
y($g, Ib);
g = $g.prototype;
g.bc=!1;
g.zc = 0;
g.jb = null;
g.stop = function() {
    this.jb && (l.clearTimeout(this.jb), this.jb = null, this.bc=!1)
};
g.pause = function() {
    this.zc++
};
g.G = function() {
    $g.J.G.call(this);
    this.stop()
};
g.Jd = function() {
    this.jb = null;
    this.bc&&!this.zc && (this.bc=!1, ah(this))
};
function ah(a) {
    a.jb = Zg(a.g, a.k);
    a.B.call(a.j)
};
function bh(a) {
    this.g = a
}
var ch = /\s*;\s*/;
g = bh.prototype;
g.set = function(a, b, c, d, e, f) {
    if (/[;=\s]/.test(a))
        throw Error('Invalid cookie name "' + a + '"');
    if (/[;\r\n]/.test(b))
        throw Error('Invalid cookie value "' + b + '"');
    n(c) || (c =- 1);
    e = e ? ";domain=" + e : "";
    d = d ? ";path=" + d : "";
    f = f ? ";secure" : "";
    c = 0 > c ? "" : 0 == c ? ";expires=" + (new Date(1970, 1, 1)).toUTCString() : ";expires=" + (new Date(w() + 1E3 * c)).toUTCString();
    this.g.cookie = a + "=" + b + e + d + c + f
};
g.get = function(a, b) {
    for (var c = a + "=", d = (this.g.cookie || "").split(ch), e = 0, f; f = d[e]; e++) {
        if (0 == f.lastIndexOf(c, 0))
            return f.substr(c.length);
        if (f == a)
            return ""
    }
    return b
};
g.remove = function(a, b, c) {
    var d = n(this.get(a));
    this.set(a, "", 0, b, c);
    return d
};
g.Ca = function() {
    return dh(this).keys
};
g.ta = function() {
    return dh(this).values
};
g.isEmpty = function() {
    return !this.g.cookie
};
g.Ga = function() {
    return this.g.cookie ? (this.g.cookie || "").split(ch).length : 0
};
g.clear = function() {
    for (var a = dh(this).keys, b = a.length - 1; 0 <= b; b--)
        this.remove(a[b])
};
function dh(a) {
    a = (a.g.cookie || "").split(ch);
    for (var b = [], c = [], d, e, f = 0; e = a[f]; f++)
        d = e.indexOf("="), - 1 == d ? (b.push(""), c.push(e)) : (b.push(e.substring(0, d)), c.push(e.substring(d + 1)));
    return {
        keys: b,
        values: c
    }
}
var eh = new bh(document);
eh.j = 3950;
function fh(a, b, c, d, e) {
    eh.set("" + a, b, c, d || "/", e || "youtube.com")
}
function gh(a, b) {
    return eh.get("" + a, b)
}
function hh(a, b, c) {
    return eh.remove("" + a, b || "/", c || "youtube.com")
};
function ih() {
    var a = gh("PREF");
    if (a)
        for (var a = unescape(a).split("&"), b = 0; b < a.length; b++) {
            var c = a[b].split("="), d = c[0];
            (c = c[1]) && (jh[d] = c.toString())
        }
}
ba(ih);
var jh = r("yt.prefs.UserPrefs.prefs_") || {};
q("yt.prefs.UserPrefs.prefs_", jh, void 0);
function kh(a) {
    if (/^f([1-9][0-9]*)$/.test(a))
        throw "ExpectedRegexMatch: " + a;
}
function lh(a) {
    if (!/^\w+$/.test(a))
        throw "ExpectedRegexMismatch: " + a;
}
function mh(a) {
    a = void 0 !== jh[a] ? jh[a].toString() : null;
    return null != a && /^[A-Fa-f0-9]+$/.test(a) ? parseInt(a, 16) : null
}
ih.prototype.get = function(a, b) {
    lh(a);
    kh(a);
    var c = void 0 !== jh[a] ? jh[a].toString() : null;
    return null != c ? c : b ? b : ""
};
ih.prototype.set = function(a, b) {
    lh(a);
    kh(a);
    if (null == b)
        throw "ExpectedNotNull";
    jh[a] = b.toString()
};
function nh(a, b) {
    return !!((mh("f" + (Math.floor(b / 31) + 1)) || 0) & 1<<b%31)
}
function oh(a, b) {
    var c = "f" + (Math.floor(a / 31) + 1), d = 1<<a%31, e = mh(c) || 0, e = b ? e | d: e&~d;
    0 == e ? delete jh[c] : (d = e.toString(16), jh[c] = d.toString())
}
ih.prototype.remove = function(a) {
    lh(a);
    kh(a);
    delete jh[a]
};
ih.prototype.clear = function() {
    jh = {}
};
function ph() {
    var a = [], b;
    for (b in jh)
        a.push(b + "=" + escape(jh[b]));
    return a.join("&")
};
function qh(a) {
    if ("function" == typeof a.ta)
        return a.ta();
    if (t(a))
        return a.split("");
    if (ea(a)) {
        for (var b = [], c = a.length, d = 0; d < c; d++)
            b.push(a[d]);
        return b
    }
    return sb(a)
}
function rh(a, b) {
    if ("function" == typeof a.forEach)
        a.forEach(b, void 0);
    else if (ea(a) || t(a))
        z(a, b, void 0);
    else {
        var c;
        if ("function" == typeof a.Ca)
            c = a.Ca();
        else if ("function" != typeof a.ta)
            if (ea(a) || t(a)) {
                c = [];
                for (var d = a.length, e = 0; e < d; e++)
                    c.push(e)
            } else 
                c = tb(a);
        else 
            c = void 0;
        for (var d = qh(a), e = d.length, f = 0; f < e; f++)
            b.call(void 0, d[f], c && c[f], a)
    }
};
function sh(a, b) {
    var c;
    a instanceof sh ? (this.eb = n(b) ? b : a.eb, th(this, a.Sa), this.cb = a.cb, uh(this, a.Ia), vh(this, a.ab), this.xa = a.xa, wh(this, a.g.clone()), this.bb = a.bb) : a && (c = Me(String(a))) ? (this.eb=!!b, th(this, c[1] || "", !0), this.cb = xh(c[2] || ""), uh(this, c[3] || "", !0), vh(this, c[4]), this.xa = xh(c[5] || "", !0), wh(this, c[6] || "", !0), this.bb = xh(c[7] || "")) : (this.eb=!!b, this.g = new yh(null, 0, this.eb))
}
g = sh.prototype;
g.Sa = "";
g.cb = "";
g.Ia = "";
g.ab = null;
g.xa = "";
g.bb = "";
g.eb=!1;
g.toString = function() {
    var a = [], b = this.Sa;
    b && a.push(zh(b, Ah, !0), ":");
    if (b = this.Ia) {
        a.push("//");
        var c = this.cb;
        c && a.push(zh(c, Ah, !0), "@");
        a.push(encodeURIComponent(String(b)).replace(/%25([0-9a-fA-F]{2})/g, "%$1"));
        b = this.ab;
        null != b && a.push(":", String(b))
    }
    if (b = this.xa)
        this.Ia && "/" != b.charAt(0) && a.push("/"), a.push(zh(b, "/" == b.charAt(0) ? Bh : Ch, !0));
    (b = this.g.toString()) && a.push("?", b);
    (b = this.bb) && a.push("#", zh(b, Dh));
    return a.join("")
};
g.resolve = function(a) {
    var b = this.clone(), c=!!a.Sa;
    c ? th(b, a.Sa) : c=!!a.cb;
    c ? b.cb = a.cb : c=!!a.Ia;
    c ? uh(b, a.Ia) : c = null != a.ab;
    var d = a.xa;
    if (c)
        vh(b, a.ab);
    else if (c=!!a.xa) {
        if ("/" != d.charAt(0))
            if (this.Ia&&!this.xa)
                d = "/" + d;
            else {
                var e = b.xa.lastIndexOf("/");
                - 1 != e && (d = b.xa.substr(0, e + 1) + d)
            }
        e = d;
        if (".." == e || "." == e)
            d = "";
        else if (Da(e, "./") || Da(e, "/.")) {
            for (var d = 0 == e.lastIndexOf("/", 0), e = e.split("/"), f = [], h = 0; h < e.length;) {
                var k = e[h++];
                "." == k ? d && h == e.length && f.push("") : ".." == k ? ((1 < f.length || 1 == f.length && "" != f[0]) &&
                f.pop(), d && h == e.length && f.push("")) : (f.push(k), d=!0)
            }
            d = f.join("/")
        } else 
            d = e
    }
    c ? b.xa = d : c = "" !== a.g.toString();
    c ? wh(b, xh(a.g.toString())) : c=!!a.bb;
    c && (b.bb = a.bb);
    return b
};
g.clone = function() {
    return new sh(this)
};
function th(a, b, c) {
    a.Sa = c ? xh(b, !0) : b;
    a.Sa && (a.Sa = a.Sa.replace(/:$/, ""))
}
function uh(a, b, c) {
    a.Ia = c ? xh(b, !0) : b
}
function vh(a, b) {
    if (b) {
        b = Number(b);
        if (isNaN(b) || 0 > b)
            throw Error("Bad port number " + b);
        a.ab = b
    } else 
        a.ab = null
}
function wh(a, b, c) {
    b instanceof yh ? (a.g = b, Eh(a.g, a.eb)) : (c || (b = zh(b, Fh)), a.g = new yh(b, 0, a.eb))
}
function Gh(a, b, c) {
    a.g.set(b, c)
}
function Hh(a, b, c) {
    da(c) || (c = [String(c)]);
    Ih(a.g, b, c)
}
function Jh(a) {
    Gh(a, "zx", Math.floor(2147483648 * Math.random()).toString(36) + Math.abs(Math.floor(2147483648 * Math.random())^w()).toString(36));
    return a
}
function Kh(a) {
    return a instanceof sh ? a.clone() : new sh(a, void 0)
}
function Lh(a, b, c, d) {
    var e = new sh(null, void 0);
    a && th(e, a);
    b && uh(e, b);
    c && vh(e, c);
    d && (e.xa = d);
    return e
}
function xh(a, b) {
    return a ? b ? decodeURI(a) : decodeURIComponent(a) : ""
}
function zh(a, b, c) {
    return t(a) ? (a = encodeURI(a).replace(b, Mh), c && (a = a.replace(/%25([0-9a-fA-F]{2})/g, "%$1")), a) : null
}
function Mh(a) {
    a = a.charCodeAt(0);
    return "%" + (a>>4 & 15).toString(16) + (a & 15).toString(16)
}
var Ah = /[#\/\?@]/g, Ch = /[\#\?:]/g, Bh = /[\#\?]/g, Fh = /[\#\?@]/g, Dh = /#/g;
function yh(a, b, c) {
    this.g = a || null;
    this.j=!!c
}
function Nh(a) {
    if (!a.U && (a.U = new ve, a.ga = 0, a.g))
        for (var b = a.g.split("&"), c = 0; c < b.length; c++) {
            var d = b[c].indexOf("="), e = null, f = null;
            0 <= d ? (e = b[c].substring(0, d), f = b[c].substring(d + 1)) : e = b[c];
            e = qa(e);
            e = Oh(a, e);
            a.add(e, f ? qa(f) : "")
        }
}
g = yh.prototype;
g.U = null;
g.ga = null;
g.Ga = function() {
    Nh(this);
    return this.ga
};
g.add = function(a, b) {
    Nh(this);
    this.g = null;
    a = Oh(this, a);
    var c = this.U.get(a);
    c || this.U.set(a, c = []);
    c.push(b);
    this.ga++;
    return this
};
g.remove = function(a) {
    Nh(this);
    a = Oh(this, a);
    return ye(this.U.j, a) ? (this.g = null, this.ga -= this.U.get(a).length, this.U.remove(a)) : !1
};
g.clear = function() {
    this.U = this.g = null;
    this.ga = 0
};
g.isEmpty = function() {
    Nh(this);
    return 0 == this.ga
};
function Ph(a, b) {
    Nh(a);
    b = Oh(a, b);
    return ye(a.U.j, b)
}
g.Ca = function() {
    Nh(this);
    for (var a = this.U.ta(), b = this.U.Ca(), c = [], d = 0; d < b.length; d++)
        for (var e = a[d], f = 0; f < e.length; f++)
            c.push(b[d]);
    return c
};
g.ta = function(a) {
    Nh(this);
    var b = [];
    if (t(a))
        Ph(this, a) && (b = ab(b, this.U.get(Oh(this, a))));
    else {
        a = this.U.ta();
        for (var c = 0; c < a.length; c++)
            b = ab(b, a[c])
    }
    return b
};
g.set = function(a, b) {
    Nh(this);
    this.g = null;
    a = Oh(this, a);
    Ph(this, a) && (this.ga -= this.U.get(a).length);
    this.U.set(a, [b]);
    this.ga++;
    return this
};
g.get = function(a, b) {
    var c = a ? this.ta(a): [];
    return 0 < c.length ? String(c[0]) : b
};
function Ih(a, b, c) {
    a.remove(b);
    0 < c.length && (a.g = null, a.U.set(Oh(a, b), bb(c)), a.ga += c.length)
}
g.toString = function() {
    if (this.g)
        return this.g;
    if (!this.U)
        return "";
    for (var a = [], b = this.U.Ca(), c = 0; c < b.length; c++)
        for (var d = b[c], e = encodeURIComponent(String(d)), d = this.ta(d), f = 0; f < d.length; f++) {
            var h = e;
            "" !== d[f] && (h += "=" + encodeURIComponent(String(d[f])));
            a.push(h)
        }
    return this.g = a.join("&")
};
g.clone = function() {
    var a = new yh;
    a.g = this.g;
    this.U && (a.U = this.U.clone(), a.ga = this.ga);
    return a
};
function Oh(a, b) {
    var c = String(b);
    a.j && (c = c.toLowerCase());
    return c
}
function Eh(a, b) {
    b&&!a.j && (Nh(a), a.g = null, a.U.forEach(function(a, b) {
        var e = b.toLowerCase();
        b != e && (this.remove(b), Ih(this, e, a))
    }, a));
    a.j = b
};
var Qh = /^https?:\/\/([A-Za-z0-9-]{1,63}\.)*(corp\.google\.com|borg\.google\.com|prod\.google\.com|video\.google\.com|youtube\.com|youtube\.googleapis\.com|youtube-nocookie\.com|youtubeeducation\.com)(:[0-9]+)?\/embed\//;
var Rh = {}, Sh = 0, Th = r("yt.net.ping.workerUrl_") || null;
q("yt.net.ping.workerUrl_", Th, void 0);
function Uh(a) {
    var b = new Image, c = "" + Sh++;
    Rh[c] = b;
    b.onload = b.onerror = function() {
        delete Rh[c]
    };
    b.src = a;
    b = eval("null")
};
var Vh = window.performance || window.mozPerformance || window.msPerformance || window.webkitPerformance || {};
u(Vh.clearResourceTimings || Vh.webkitClearResourceTimings || Vh.mozClearResourceTimings || Vh.msClearResourceTimings || Vh.oClearResourceTimings || s, Vh);
var Wh = Vh.mark ? function(a) {
    Vh.mark(a)
}
: s;
function Xh(a) {
    if (!qg || a)
        qg = Yc(window);
    return qg
};
function Yh() {}
y(Yh, vg);
g = Yh.prototype;
g.Ma = function(a) {
    var b = vg.prototype.Ma.call(this, a);
    return b ? b : a
};
g.T = function(a, b) {
    var c = Yh.J.T.call(this, a, b);
    return c ? c : (c = Yh.J.T.call(this, a, "card-config")) && (c = r(c)) && c[b] ? c[b] : null
};
g.show = function(a) {
    var b = this.Ma(a);
    if (b) {
        C(b, T(this, "active"));
        var c = Zh(this, a, b);
        if (c) {
            c.cardTargetNode = a;
            c.cardRootNode = b;
            $h(this, a, c);
            var d = T(this, "card-visible"), e = this.T(a, "card-delegate-show") && this.T(b, "card-action");
            this.Qb(b, "card-action", a);
            this.j = a;
            pf(c);
            G(u(function() {
                e || (of(c), K("yt-uix-card-show", b, a, c));
                ai(c);
                C(c, d);
                K("yt-uix-kbd-nav-move-in-to", c)
            }, this), 10)
        }
    }
};
function Zh(a, b, c) {
    var d = c || b, e = T(a, "card");
    c = bi(a, d);
    var f = O(T(a, "card") + Zd(d));
    if (f)
        return a = Q(T(a, "card-body"), f), rd(a, c) || (jd(c), a.appendChild(c)), f;
    f = document.createElement("div");
    f.id = T(a, "card") + Zd(d);
    f.className = e;
    (d = a.T(d, "card-class")) && jb(f, d.split(/\s+/));
    d = document.createElement("div");
    d.className = T(a, "card-border");
    b = a.T(b, "orientation") || "horizontal";
    e = document.createElement("div");
    e.className = "yt-uix-card-border-arrow yt-uix-card-border-arrow-" + b;
    var h = document.createElement("div");
    h.className = T(a, "card-body");
    a = document.createElement("div");
    a.className = "yt-uix-card-body-arrow yt-uix-card-body-arrow-" + b;
    jd(c);
    h.appendChild(c);
    d.appendChild(a);
    d.appendChild(h);
    f.appendChild(e);
    f.appendChild(d);
    document.body.appendChild(f);
    return f
}
function $h(a, b, c) {
    var d = a.T(b, "orientation") || "horizontal", e = a.T(b, "position"), f=!!a.T(b, "force-position"), h = a.T(b, "position-fixed"), d = "horizontal" == d, k = "bottomright" == e || "bottomleft" == e, m = "topright" == e || "bottomright" == e, p, x;
    m && k ? (x = 7, p = 4) : m&&!k ? (x = 6, p = 5) : !m && k ? (x = 5, p = 6) : (x = 4, p = 7);
    var P = Ld(document.body), e = Ld(b);
    P != e && (x^=2);
    var I;
    d ? (e = b.offsetHeight / 2 - 12, I = new L( - 12, b.offsetHeight + 6)) : (e = b.offsetWidth / 2 - 6, I = new L(b.offsetWidth + 6, - 12));
    var W = Qd(c), e = Math.min(e, (d ? W.height : W.width) - 24 - 6);
    6 > e && (e =
    6, d ? I.y += 12 - b.offsetHeight / 2 : I.x += 12 - b.offsetWidth / 2);
    var ra = null;
    f || (ra = 10);
    W = T(a, "card-flip");
    a = T(a, "card-reverse");
    mb(c, W, m);
    mb(c, a, k);
    ra = og(b, x, c, p, I, null, ra);
    !f && ra && (ra & 48 && (m=!m, x^=2, p^=2), ra & 192 && (k=!k, x^=1, p^=1), mb(c, W, m), mb(c, a, k), og(b, x, c, p, I));
    h && (b = parseInt(c.style.top, 10), f = $c(document).y, sf(c, "position", "fixed"), sf(c, "top", b - f + "px"));
    P && (c.style.right = "", b = Sd(c), b.left = b.left || parseInt(c.style.left, 10), f = Yc(window), c.style.left = "", c.style.right = f.width - b.left - b.width + "px");
    b = Q("yt-uix-card-body-arrow",
    c);
    f = Q("yt-uix-card-border-arrow", c);
    d = d ? k ? "top" : "bottom" : !P && m || P&&!m ? "left" : "right";
    b.setAttribute("style", "");
    f.setAttribute("style", "");
    b.style[d] = e + "px";
    f.style[d] = e + "px";
    k = Q("yt-uix-card-arrow", c);
    m = Q("yt-uix-card-arrow-background", c);
    k && m && (c = "right" == d ? Qd(c).width - e - 13 : e + 11, e = c / Math.sqrt(2), k.style.left = c + "px", k.style.marginLeft = "1px", m.style.marginLeft =- e + "px", m.style.marginTop = e + "px")
}
g.hide = function(a) {
    if (a = this.Ma(a)) {
        var b = O(T(this, "card") + Zd(a));
        b && (D(a, T(this, "active")), D(b, T(this, "card-visible")), pf(b), this.j = null, b.cardTargetNode = null, b.cardRootNode = null, b.cardMask && (jd(b.cardMask), b.cardMask = null))
    }
};
function ci(a) {
    a.j && a.hide(a.j)
}
g.ee = function(a, b) {
    var c = this.Ma(a);
    if (c) {
        if (b) {
            var d = bi(this, c);
            if (!d)
                return;
            d.innerHTML = b
        }
        B(c, T(this, "active")) && (c = Zh(this, a, c), $h(this, a, c), of(c), ai(c))
    }
};
g.isActive = function(a) {
    return (a = this.Ma(a)) ? B(a, T(this, "active")) : !1
};
function bi(a, b) {
    var c = b.cardContentNode;
    if (!c) {
        var d = T(a, "content"), e = T(a, "card-content");
        (c = (c = a.T(b, "card-id")) ? O(c) : Q(d, b)) || (c = document.createElement("div"));
        ob(c, d, e);
        b.cardContentNode = c
    }
    return c
}
function ai(a) {
    var b = a.cardMask;
    b || (b = document.createElement("iframe"), b.src = 'javascript:""', jb(b, ["yt-uix-card-iframe-mask"]), a.cardMask = b);
    b.style.position = a.style.position;
    b.style.top = a.style.top;
    b.style.left = a.offsetLeft + "px";
    b.style.height = a.clientHeight + "px";
    b.style.width = a.clientWidth + "px";
    document.body.appendChild(b)
};
function di() {
    this.g = {};
    this.k = {}
}
y(di, Yh);
ba(di);
di.prototype.Ta = "clickcard";
di.prototype.show = function(a) {
    di.J.show.call(this, a);
    var b = this.Ma(a);
    if (!E(b, "click-outside-persists")) {
        var c = ha(a);
        if (this.g[c])
            return;
        var b = R(document, "click", u(this.B, this, a)), d = R(window, "blur", u(this.B, this, a));
        this.g[c] = [b, d]
    }
    a = R(window, "resize", u(this.ee, this, a, void 0));
    this.k[c] = a
};
di.prototype.hide = function(a) {
    di.J.hide.call(this, a);
    a = ha(a);
    var b = this.g[a];
    b && (le(b), this.g[a] = null);
    if (b = this.k[a])
        le(b), this.k[a] = null
};
di.prototype.B = function(a, b) {
    yd(b.target, "yt-uix" + (this.Ta ? "-" + this.Ta : "") + "-card") || this.hide(a)
};
function ei() {}
y(ei, Yh);
ba(ei);
ei.prototype.Ta = "hovercard";
function fi(a, b, c, d) {
    this.oa = a;
    this.A=!1;
    this.j = new Lb;
    this.C = me(this.oa, "click", u(this.be, this), "yt-dialog-dismiss");
    gi(this);
    this.M = b;
    this.$ = c;
    this.D = d;
    this.B = this.k = null
}
var hi = {
    LOADING: "loading",
    of: "content",
    sf: "working"
};
function ii(a, b) {
    a.L() || a.j.subscribe("post-all", b)
}
function gi(a) {
    a = Q("yt-dialog-fg-content", a.oa);
    var b = [];
    pb(hi, function(a) {
        b.push("yt-dialog-show-" + a)
    });
    kb(a, b);
    C(a, "yt-dialog-show-content")
}
g = fi.prototype;
g.show = function() {
    if (!this.L()) {
        document.activeElement && document.activeElement != document.body && document.activeElement.blur && document.activeElement.blur();
        if (!this.D) {
            this.g || (this.g = O("yt-dialog-bg"), this.g || (this.g = document.createElement("div"), this.g.id = "yt-dialog-bg", this.g.className = "yt-dialog-bg", document.body.appendChild(this.g)));
            var a;
            t:
            {
                var b = window, c = b.document;
                a = 0;
                if (c) {
                    a = c.body;
                    var d = c.documentElement;
                    if (!d ||!a) {
                        a = 0;
                        break t
                    }
                    b = Yc(b).height;
                    if (Zc(c) && d.scrollHeight)
                        a = d.scrollHeight != b ? d.scrollHeight :
                        d.offsetHeight;
                    else {
                        var c = d.scrollHeight, e = d.offsetHeight;
                        d.clientHeight != e && (c = a.scrollHeight, e = a.offsetHeight);
                        a = c > b ? c > e ? c : e : c < e ? c : e
                    }
                }
            }
            this.g.style.height = a + "px";
            of(this.g)
        }
        ee(this.oa);
        a = ji(this);
        ki(a);
        this.M || (this.k = R(document, "keydown", u(this.Od, this)));
        a = this.oa;
        d = J("player-added", this.Nd, this);
        Eb(a, "player-ready-pubsub-key", d);
        this.$ && (this.B = R(document, "click", u(this.Pd, this)));
        of(this.oa);
        C(document.body, "yt-dialog-active");
        a = wg.getInstance();
        a.g && yg(a, a.g);
        ci(di.getInstance());
        ci(ei.getInstance())
    }
};
function li() {
    var a = Uc("yt-dialog");
    return Pa(a, function(a) {
        return lf(a)
    })
}
g.Nd = function() {
    ee(this.oa)
};
function ji(a) {
    var b = Vc("iframe", null, a.oa);
    z(b, function(a) {
        var b = E(a, "onload");
        b && (b = r(b)) && R(a, "load", b);
        if (b = E(a, "src"))
            a.src = b
    }, a);
    return bb(b)
}
function ki(a) {
    z(document.getElementsByTagName("iframe"), function(b) {
        - 1 == La(a, b) && C(b, "iframe-hid")
    })
}
function mi() {
    var a = Uc("iframe-hid");
    z(a, function(a) {
        D(a, "iframe-hid")
    })
}
g.be = function(a) {
    a = a.currentTarget;
    a.disabled || (a = E(a, "action") || "", ni(this, a))
};
function ni(a, b) {
    if (!a.L()) {
        a.j.publish("pre-all");
        a.j.publish("pre-" + b);
        pf(a.oa);
        ci(di.getInstance());
        ci(ei.getInstance());
        li() || (pf(a.g), D(document.body, "yt-dialog-active"), fe(), mi());
        a.k && (le(a.k), a.k = null);
        a.B && (le(a.B), a.B = null);
        var c = a.oa;
        if (c) {
            var d = E(c, "player-ready-pubsub-key");
            d && (bc(d), Gb(c, "player-ready-pubsub-key"))
        }
        a.j.publish("post-all");
        K("yt-ui-dialog-hide-complete", a);
        "cancel" == b && K("yt-ui-dialog-cancelled", a);
        a.j && a.j.publish("post-" + b)
    }
}
g.Od = function(a) {
    G(u(function() {
        27 == a.keyCode && ni(this, "cancel")
    }, this), 0)
};
g.Pd = function(a) {
    "yt-dialog-base" == a.target.className && ni(this, "cancel")
};
g.L = function() {
    return this.A
};
g.dispose = function() {
    lf(this.oa) && ni(this, "dispose");
    le(this.C);
    this.j.dispose();
    this.j = null;
    this.A=!0
};
q("yt.ui.Dialog", fi, void 0);
function oi() {}
y(oi, vg);
ba(oi);
g = oi.prototype;
g.va = null;
g.pb = null;
g.Ta = "overlay";
function pi(a, b) {
    if (!a.va ||!lf(a.va.oa)) {
        var c = a.Ma(b), d = qi(c, b);
        c || (c = d ? d.overlayParentNode : null);
        if (c && d) {
            var e=!!a.T(c, "disable-shortcuts") ||!1;
            a.va = new fi(d, e);
            a.pb = c;
            var f = Q("yt-dialog-fg", d);
            if (f) {
                var h = a.T(c, "overlay-class") || "", k = a.T(c, "overlay-style") || "default", m = a.T(c, "overlay-shape") || "default", h = h ? h.split(" "): [];
                h.push(T(a, k));
                h.push(T(a, m));
                jb(f, h)
            }
            a.va.show();
            K("yt-uix-kbd-nav-move-to", d);
            ri(a);
            e || (e = u(function(a) {
                B(a.target, "yt-dialog-base") && si(this)
            }, a), d = Q("yt-dialog-base", d),
            a.g = R(d, "click", e));
            a.Qb(c, "overlay-shown")
        }
    }
}
function ri(a) {
    a.j || (a.j = J("yt-uix-overlay-hide", ti));
    a.va && ii(a.va, function() {
        var a = oi.getInstance();
        a.pb = null;
        a.va.dispose();
        a.va = null
    })
}
function si(a) {
    if (a.va) {
        var b = a.pb;
        ni(a.va, "overlayhide");
        a.Qb(b, "overlay-hidden");
        a.pb = null;
        a.g && (le(a.g), a.g = null);
        a.va = null
    }
}
function qi(a, b) {
    var c;
    if (a)
        if (c = Q("yt-dialog", a)) {
            var d = O("body-container");
            d && (d.appendChild(c), a.overlayContentNode = c, c.overlayParentNode = a)
        } else 
            c = a.overlayContentNode;
    else 
        b && (c = yd(b, "yt-dialog"));
    return c
}
function ui() {
    var a = oi.getInstance();
    if (a.pb)
        a = Q("yt-dialog-fg-content", a.pb.overlayContentNode);
    else 
        t: {
        if (a = Uc("yt-dialog-fg-content"))
            for (var b = 0; b < a.length; b++) {
                var c = yd(a[b], "yt-dialog");
                if (lf(c)) {
                    a = a[b];
                    break t
                }
            }
            a = null
    }
    return a
}
g.hide = function() {
    K("yt-uix-overlay-hide")
};
function ti() {
    si(oi.getInstance())
}
g.show = function(a) {
    pi(this, a)
};
function vi(a) {
    window.location = Ve(a, {}) + ""
}
function wi(a) {
    (window.ytspf || {}).enabled ? spf.navigate(a) : vi(a)
}
function xi(a, b, c) {
    var d = F("EVENT_ID");
    d && (b || (b = {}), b.ei || (b.ei = d));
    b && (d = Oe(a), d == Oe(window.location.href) ||!d && 0 == a.lastIndexOf("/", 0)) && (d = Qe(a), d = Re(d)) && (d = "s_tempdata-" + Ga(d), b = b ? Ue(b) : "", fh(d, b, 5));
    if (c)
        return !1;
    wi(a);
    return !0
};
var ig, yi, zi, Ai, Bi = [];
function Ci() {
    var a = Di(), b = E(a, "collection-id"), c = Q("subscription-picker-list-container", a), d = Uc("subscription-list-item", c), c = A(Na(d, function(a) {
        return B(a, "selected")&&!B(a, "saved")
    }), function(a) {
        return E(a, "channel-id")
    }), d = A(Na(d, function(a) {
        return !B(a, "selected") && B(a, "saved")
    }), function(a) {
        return E(a, "channel-id")
    });
    Ei(a);
    var e = Q("collection-name-input", a), f = e.attributes.value;
    (e = e.value) ? (Fi(a, !0), b ? c.length || d.length || f != e ? Gi(b, e, c, d) : (Ei(a), oi.getInstance().hide()) : Hi(e, c)) : Ii(a, !0)
}
function Ji(a, b) {
    Ei(a);
    oi.getInstance().hide();
    b.response && b.response.collection_url && vi(b.response.collection_url)
}
function Gi(a, b, c, d) {
    var e = Di(), f = db(c, 0, 40), h = db(d, 0, 40), f = {
        name: b,
        session_token: Pb(),
        collection_id: a,
        added_ids: f.join(","),
        removed_ids: h.join(",")
    };
    ff("/subscription_ajax?action_update_collection=1", {
        method: "POST",
        ca: f,
        V: function(f, h) {
            c.length || d.length ? Gi(a, b, c, d) : Ji(e, h)
        },
        onError: Ki(e)
    })
}
function Hi(a, b) {
    var c = Di();
    if (b.length) {
        var d = db(b, 0, 40), d = {
            name: a,
            session_token: Pb(),
            external_channel_ids: d.join(",")
        };
        ff("/subscription_ajax?action_add_collection=1", {
            method: "POST",
            ca: d,
            V: function(d, f) {
                if (b.length) {
                    var h = f.response.collection_id;
                    h && Gi(h, a, b, [])
                } else 
                    Ji(c, f)
            },
            onError: Ki(c)
        })
    } else 
        Ei(c), oi.getInstance().hide()
}
function Li() {
    var a = Di(), b = E(a, "collection-id");
    Ei(a);
    Fi(a, !0);
    ff("/subscription_ajax?action_remove_collection=1", {
        method: "POST",
        ca: {
            collection_id: b,
            session_token: Pb()
        },
        V: function() {
            Ei(a);
            oi.getInstance().hide();
            vi("/")
        },
        onError: Ki(a)
    })
}
function Mi() {
    var a = Di(), b = Q("subscription-picker-list-container", a), b = Q("subscription-picker-list", b).cloneNode(!0), c = Uc("subscription-list-item", b);
    z(c, function(a) {
        B(a, "selected") || jd(a)
    });
    c = Q("selected-channel-list-container", a);
    hd(c);
    c.appendChild(b);
    a = Q("selected-channels-button", a);
    kf(a, !!b)
}
function Ni(a) {
    var b = a.currentTarget;
    if (yd(b, "selected-channel-list-container")) {
        var c = E(b, "channel-id");
        a = Di();
        a = Q("subscription-picker-list-container", a);
        a = Uc("subscription-list-item", a);
        z(a, function(a) {
            E(a, "channel-id") == c && (b = a)
        })
    }
    nb(b, "selected");
    Mi()
}
function Oi() {
    var a = Di(), b = Q("subscription-search-input", a);
    Pi(a, b && b.value)
}
function Qi() {
    var a = Di();
    Q("subscription-search-input", a).value = "";
    Pi(a, "")
}
function Pi(a, b) {
    var c = Uc("subscription-list-item", a), d = new RegExp(b, "i");
    z(c, function(a) {
        var c = E(a, "channel-name");
        kf(a, !b || d.test(c))
    });
    c = Q("subscription-search-clear", a);
    nf(c, !!b)
}
function Ri(a) {
    var b = E(a, "panel-class");
    b && (a = ui(), a = Q("subscription-picker", a), a = Q(b, a), a = qd(a), z(ld(a), function(a) {
        kf(a, B(a, b))
    }))
}
function Si(a) {
    Ri(a.currentTarget)
}
function Di() {
    var a = ui();
    return Q("subscription-picker", a)
}
function Ei(a) {
    Ti(a, "");
    Ui(a, !1);
    Fi(a, !1);
    Ii(a, !1)
}
function Ui(a, b) {
    var c = Q("subscription-picker-loading", a);
    kf(c, b)
}
function Fi(a, b) {
    var c = Q("subscription-picker-saving", a);
    kf(c, b)
}
function Ii(a, b) {
    var c = Q("collection-name-empty-error", a);
    kf(c, b)
}
function Ti(a, b) {
    var c = Q("subscription-picker-errors", a);
    kf(c, "" != b);
    b && (c = Q("yt-alert-content", c), sd(c, b))
}
function Ki(a) {
    return function(b, c) {
        Ei(a);
        var d = c.errors;
        d && 0 < d.length && Ti(a, d[0])
    }
};
function Vi(a, b, c) {
    this.g = a;
    (a = b || null) || (a = Wi(this.g));
    a = "(" + a.join("|") + ")";
    a = oa("__%s__", a);
    this.k = new RegExp(a, "g");
    this.j = c || {}
}
var Xi = /__([a-z]+(?:_[a-z]+)*)__/g;
function Yi(a, b) {
    var c = Zi(O(a));
    return new Vi(c, b, void 0)
}
function Zi(a) {
    a = a.innerHTML;
    a = a.replace(/^\s*(\x3c!--\s*)?/, "");
    return a = a.replace(/(\s*--\x3e)?\s*$/, "")
}
function Wi(a) {
    var b = [], c = {};
    a.replace(Xi, function(a, e) {
        e in c || (c[e]=!0, b.push(e))
    });
    return b
}
function $i(a, b) {
    var c = u(function(a, c) {
        return sa(b[c] || this.j[c] || "")
    }, a);
    return a.g.replace(a.k, c)
};
function aj(a) {
    fg.call(this);
    this.A = a;
    this.ka = 0;
    this.D = this.H = this.C = this.B = this.k = this.ba = null
}
y(aj, fg);
g = aj.prototype;
g.Ja = function() {
    aj.J.Ja.call(this);
    this.S = this.wa();
    this.ka = parseInt(E(this.S, "max-title-length"), 10) || 0;
    this.C = bg(this, "create-button");
    this.H = bg(this, "cancel-button");
    this.B = bg(this, "privacy-button");
    var a = cg(this);
    this.k = bg(this, "title-input");
    a.listen(this.k, "keyup", this.Tc);
    a.listen(this.k, "paste", this.Tc);
    a.listen(this.S, "reset", this.te);
    a.listen(this.S, "submit", this.ue);
    this.ba = bg(this, "create-playlist-widget-privacy-menu");
    this.D = bg(this, "yt-uix-button-menu-item-selected");
    a = me(this.wa(),
    "click", u(this.ve, this), "privacy-option");
    this.Aa.push(a)
};
g.Da = function() {
    bj(this);
    this.D = this.H = this.C = this.ba = this.B = this.k = null;
    aj.J.Da.call(this)
};
g.Tc = function() {
    var a = ce, b = this.C, c = this.k.value.replace(/^\s+|\s+$/, "").length;
    a(b, 0 < c && (!this.ka || c <= this.ka))
};
g.ve = function(a) {
    (a = a.currentTarget) && cj(this, a);
    a = E(a, "value");
    bg(this, "privacy-value-input").value = a
};
function cj(a, b) {
    var c;
    c = wg.getInstance();
    c = Q(T(c, "content"), a.B);
    var d;
    Oc && "innerText"in b ? d = b.innerText.replace(/(\r\n|\r|\n)/g, "\n") : (d = [], vd(b, d, !0), d = d.join(""));
    d = d.replace(/ \xAD /g, " ").replace(/\xAD/g, "");
    d = d.replace(/\u200B/g, "");
    Oc || (d = d.replace(/ +/g, " "));
    " " != d && (d = d.replace(/^\s*/, ""));
    sd(c, pa(d));
    c = Q("yt-uix-button-menu-item-selected", a.ba);
    D(c, "yt-uix-button-menu-item-selected");
    C(b, "yt-uix-button-menu-item-selected");
    c = E(b, "privacy-state-of-menu");
    Eb(a.B, "privacy-state", c)
}
function dj(a, b) {
    ce(a.C, b);
    ce(a.H, b);
    ce(a.k, b);
    ce(a.B, b)
}
function bj(a) {
    a.k.value = "";
    var b = bg(a, "title-input-container");
    D(b, "yt-uix-form-error");
    b = Q("yt-uix-form-error-message", b);
    jd(b);
    ce(a.B, !0);
    (b = Q("yt-uix-button-menu-item-selected", a.ba)) && a.D && b != a.D && cj(a, a.D);
    ce(a.C, !1);
    ce(a.H, !0);
    ce(a.k, !0)
}
g.ue = function(a) {
    a.preventDefault();
    a = this.S;
    var b = {
        context: this,
        V: this.de,
        onError: this.ce
    };
    b.method = a.method.toUpperCase();
    if ("POST" == b.method) {
        var c;
        c = [];
        ze(a, c, Ce);
        c = c.join("&");
        b.ae = c
    } else {
        var d = new ve;
        ze(a, d, Be);
        we(d);
        c = {};
        for (var e = 0; e < d.g.length; e++) {
            var f = d.g[e];
            c[f] = d.j[f]
        }
        d = b.$a || {};
        Cb(d, c);
        b.$a = d
    }
    ff(a.action, b);
    dj(this, !1)
};
g.de = function(a, b) {
    bj(this);
    this.A && fa(this.A.Zb) && this.A.Zb({
        playlistId: b.response.playlistId,
        playlistName: b.response.playlistName,
        Ke: b.response.playlistEditorUrl
    });
    K("yt-uix-videoactionmenu-hide")
};
g.ce = function(a, b) {
    if (b && b.errors && b.errors.length) {
        var c = bg(this, "title-input-container"), d = b.errors[0];
        C(c, "yt-uix-form-error");
        var e = Q("yt-uix-form-error-message", c);
        e ? e.innerHTML = d : (d = cd("span", "yt-uix-form-error-message", document.createTextNode(String(d))), c.appendChild(d));
        dj(this, !0)
    }
};
g.te = function() {
    bj(this);
    this.A && fa(this.A.$b) && this.A.$b()
};
var ej = {};
function fj(a, b) {
    if (F("EVENTS_TRACKER_INSTALLED")) {
        var c = ej[a];
        if (!c) {
            var d = window._gaq._getAsyncTracker("eventsPageTracker");
            if (!d)
                return;
            window._gaq.push(function() {
                c = d._createEventTracker(a);
                ej[a] = c
            })
        }
        window._gaq.push(function() {
            c._trackEvent(b, void 0, void 0)
        })
    }
};
function gj(a, b) {
    fj(a, b || "null");
    var c = "/gen_204?" + ("a=" + a + (b ? "&" + b : "").replace(/\//g, "&"));
    c && Uh(c)
};
var hj, ij, jj, kj, lj, mj, nj;
nj = mj = lj = kj = jj = ij = hj=!1;
var oj = vc;
oj && ( - 1 != oj.indexOf("Firefox") ? hj=!0 : - 1 != oj.indexOf("Camino") ? ij=!0 : - 1 != oj.indexOf("iPad") ? kj=!0 : - 1 != oj.indexOf("iPhone")||-1 != oj.indexOf("iPod") ? jj=!0 : - 1 != oj.indexOf("Chrome") ? mj=!0 : - 1 != oj.indexOf("Android") ? lj=!0 : - 1 != oj.indexOf("Safari") && (nj=!0));
var pj = hj, qj = ij, rj = jj, sj = kj, tj = lj, uj = mj, vj = nj;
var wj, xj, yj = {
    pf: "content-snap-width-1",
    qf: "content-snap-width-2",
    rf: "content-snap-width-3"
};
function zj() {
    var a = [], b;
    for (b in yj)
        a.push(yj[b]);
    return a
};
function Aj() {
    Bj = Vc("html", void 0, void 0)[0];
    Cj = O("appbar-guide-button")
}
function Dj(a) {
    var b = B(Bj, "show-guide");
    mb(Bj, "show-guide", a);
    Cj && (Cj.setAttribute("aria-pressed", a), Cj.setAttribute("aria-expanded", a));
    if (a&&!b ||!a && b) {
        a = "";
        if (b = O("page"))
            a = b.className;
        a = Ue({
            "module-id": "guide-main",
            expanded: !0,
            auto: !1,
            "page-class": a,
            notification: !1
        });
        gj("guide-toggled", a)
    }(a = O("page")) && vj && (b = a.style.width, a.style.width = "99.99%", tf(a.offsetWidth), a.style.width = b)
}
function Ej() {
    return B(Bj, "show-guide")
}
function Fj() {
    return B(document.body, "guide-pinning-enabled")
}
function Gj() {
    return Fj() && B(Bj, "guide-pinned")
}
var Bj = null, Cj = null;
var Hj;
function Ij() {
    this.g = Jj
}
function Kj() {
    function a() {
        var a = Q("guide-likes-playlist-icon");
        if (a)
            return yd(a, "guide-notification-item")
    }
    Hj = O("appbar-main-guide-notification-container");
    Lj("yt-uix-playlistlike-liked", "appbar-guide-notification-playlist-like", function() {
        return Q("guide-item-container", O("behavior-id-guide-playlists-section"))
    }, Jj, "guide-playlist-section-updated");
    Mj("yt-uix-playlistlike-unliked", "appbar-guide-notification-playlist-unlike", Nj, "guide-playlist-section-updated");
    Oj("addto-menu-video-added", "appbar-guide-notification-playlist-video-added",
    Nj, null, new Ij);
    Mj("addto-menu-video-removed", "appbar-guide-notification-playlist-video-removed", Nj, "guide-playlist-section-updated");
    Lj("subscription-subscribe-success", "appbar-guide-notification-subscription", v(O, "guide-channels"), Pj, "guide-channel-section-updated", Qj);
    Mj("subscription-unsubscribe-success", "appbar-guide-notification-unsubscription", Rj, "guide-channel-section-updated");
    var b = v(O, "VLWL-guide-item");
    Oj("WATCH_LATER_VIDEO_ADDED", "appbar-guide-notification-watch-later-video-added", b,
    1);
    Oj("WATCH_LATER_VIDEO_REMOVED", "appbar-guide-notification-watch-later-video-removed", b, - 1);
    Oj("yt-uix-videolike-liked", "appbar-guide-notification-video-like", a, 1);
    Oj("yt-uix-videolike-unliked", "appbar-guide-notification-video-unlike", a, - 1)
}
function Jj(a) {
    if (!a || O(a.id))
        return null;
    var b = ["ID", "URL", "TITLE", "NOTIFICATION_OVERLAY_MESSAGE"];
    a = $i("RD" == a.playlistType ? Yi("appbar-guide-item-template-mix", b) : Yi("appbar-guide-item-template-playlist", b), {
        ID: a.id,
        URL: a.url,
        TITLE: a.title,
        NOTIFICATION_OVERLAY_MESSAGE: a.title
    });
    return de(a)
}
function Pj(a, b, c, d) {
    if (!d)
        return null;
    a = $i(Yi("appbar-guide-item-template-channel", ["ID", "URL", "TITLE", "THUMBNAIL_URL", "NOTIFICATION_OVERLAY_MESSAGE"]), {
        ID: d.external_id,
        URL: d.url,
        TITLE: d.title,
        THUMBNAIL_URL: d.thumbnail,
        NOTIFICATION_OVERLAY_MESSAGE: d.title
    });
    return de(a)
}
function Qj() {
    return O("guide-subscriptions-promo") ? (K("force-reload-subscriptions"), !1) : !0
}
function Nj(a) {
    return O(a.id + "-guide-item")
}
function Rj(a) {
    return a ? O(a + "-guide-item") : null
}
function Sj(a, b) {
    var c = Zi(O(a)), c = de(c);
    if (b) {
        var d = Q("appbar-guide-notification-text-content", c);
        if (d) {
            var e = document.createTextNode(String(" " + b));
            d && e && d.appendChild(e)
        }
    }
    return c
}
function Oj(a, b, c, d, e) {
    J(a, function(a) {
        var h = c.apply(null, arguments);
        if (d) {
            var k = Q("guide-count-value", h);
            if (k) {
                var m = k.innerHTML;
                "" === m.trim() || isNaN(m) || (k.innerHTML = parseInt(m, 10) + d)
            }
        }
        !h && e && (k = e.g(a), (m = Q("guide-item-container", O("behavior-id-guide-playlists-section"))) && k && id(m, k, 0));
        Tj(b, !!h, a ? a.title : null) || (k = Q("guide-item-update-notification", h), k || (k = Sj(b), C(k, "guide-item-update-notification"), h.appendChild(k)), G(v(C, h, "showing-update-notification"), 0), G(v(D, h, "showing-update-notification"),
        2E3))
    })
}
function Lj(a, b, c, d, e, f) {
    J(a, function() {
        if (!f || f.apply(arguments)) {
            var a = c();
            if (!Tj(b, !!a)) {
                var k = d.apply(null, arguments);
                id(a, k, 0);
                K(e)
            }
        }
    })
}
function Mj(a, b, c, d) {
    J(a, function() {
        var a = c.apply(null, arguments);
        if (!Tj(b, !!a)) {
            var f = Sj(b);
            C(f, "guide-item-removal-notification");
            a.appendChild(f);
            G(v(C, a, "removing-guide-item"), 0);
            G(function() {
                jd(a);
                K(d)
            }, 2E3)
        }
    })
}
function Tj(a, b, c) {
    if (b && Ej())
        return !1;
    hd(Hj);
    Hj.appendChild(Sj(a, c));
    C(document.body, "show-guide-button-notification");
    G(v(D, document.body, "show-guide-button-notification"), 2E3);
    return !0
};
var Uj = [], Vj = [], Wj = [];
function Xj() {
    ig = O("guide");
    yi = O("guide-channels");
    zi = O("appbar-guide-menu");
    Ai = O("appbar-guide-iframe-mask");
    Kj();
    ig && (Aj(), B(document.body, "exp-scrollable-guide") || Yj(), Bi.push(me(null, "click", Ci, "collection-save-button")), Bi.push(me(null, "click", Li, "collection-delete-button")), Bi.push(me(null, "click", Ni, "subscription-list-item")), Bi.push(me(null, "keyup", Oi, "subscription-search-input")), Bi.push(me(null, "click", Qi, "subscription-search-clear")), Bi.push(me(null, "click", Si, "tab-panel-trigger")), Uj.push(me(ig,
    "click", Zj, "guide-sort-choice")), B(document.body, "exp-scrollable-guide") ? (ak(Xh(!0)), Vj.push(J("page-resize", ak))) : Vj.push(J("page-resize", bk)), Vj.push(J("guide-channel-section-updated", ck)), Vj.push(J("guide-playlist-section-updated", v(bk, v(Xh, !0)))), Vj.push(J("force-reload-subscriptions", dk)), Vj.push(J("update-guide-subscriptions", ek)))
}
function fk(a, b) {
    z(Uc("guide-flyout-container", ig), function(c) {
        Uj.push(R(c, a, b))
    })
}
function gk(a) {
    var b = a.currentTarget;
    a = Q("guide-flyout", b);
    var c = Q("guide-flyout-trigger", b), b = Q("guide-flyout-iframe-mask", b);
    D(a, "flyout-shown");
    b && D(b, "flyout-shown");
    D(c, "on-hover")
}
function hk(a) {
    a = a.currentTarget;
    var b = Q("guide-flyout", a), c = Q("guide-flyout-trigger", a), d = Q("guide-flyout-iframe-mask", a), e = Q("guide-flyout", a), f = Vd(e), h = Q("guide-channels-list", a), k = Xh(!0), m = Q("guide-flyout-trigger", a), p = Q("guide-flyout-iframe-mask", a), x = k.height - (zi ? zi.offsetTop : 0);
    h.style.maxHeight = x - f.top - f.bottom + "px";
    f = Qd(e).height;
    h = Qd(m);
    m = Od(m);
    x = Math.min(x - f, Math.max(0, k.height - m.y - h.height / 2 - f / 2));
    h = m.x + h.width;
    k = k.width - m.x;
    e.style.bottom = x + "px";
    p && (p.style.height = f + "px", p.style.bottom =
    x + "px");
    "rtl" == document.body.getAttribute("dir") ? (e.style.right = k + "px", e.style.left = "", p && (p.style.right = k + "px", p.style.left = "")) : (e.style.left = h + "px", e.style.right = "", p && (p.style.left = h + "px", p.style.right = ""));
    C(b, "flyout-shown");
    d && C(d, "flyout-shown");
    C(c, "on-hover");
    K("yt-dom-content-change", a)
}
function ik() {
    Ai && zi && (Ai.style.height = Qd(zi).height + "px", Ai.style.marginTop = zi.style.marginTop, Ai.style.top = zi.style.top)
}
function Yj() {
    yi = O("guide-channels");
    Wj = hg();
    bk(Xh(!0));
    fk("mouseenter", hk);
    fk("mouseleave", gk)
}
function Zj(a) {
    var b = Q("guide-sort-button"), c = E(b, "guide-sort") || "", d = E(a.currentTarget, "guide-sort") || "";
    c != d && (Eb(b, "guide-sort", d), ff("/guide_channels_ajax?action_set_guide_sort=1", {
        method: "POST",
        $a: {
            sort: d
        },
        ca: {
            session_token: Pb()
        },
        format: "JSON",
        V: function(a, b) {
            ek(b);
            var c = Uc("guide-sort-choice");
            z(c, function(a) {
                var b = E(a, "guide-sort");
                wg.getInstance();
                a = a.parentNode;
                var b = d == b, c = be("span", "yt-uix-button-icon-wrapper", a);
                if (!c && b) {
                    var e = cd("span", {
                        "class": "yt-uix-button-icon-wrapper yt-uix-button-icon-checkbox"
                    }),
                    f = cd("div", {
                        "class": "yt-uix-button-icon-dropdown-checked"
                    });
                    e.appendChild(f);
                    id(a, e, 0)
                }
                kf(c, b)
            })
        }
    }))
}
function ek(a) {
    var b = a.channels;
    b && (a = O("guide-subscriptions-section"), b = fd(b), kd(b, a), Wj = hg(), jk(), a = Uc("yt-uix-tooltip-tip-visible"), z(a, function(a) {
        D(a, "yt-uix-tooltip-tip-visible")
    }))
}
function jk() {
    Yj();
    K("yt-dom-content-change", ig)
}
function ck() {
    yi && (yi.firstElementChild ? jk() : dk())
}
function kk(a) {
    var b = Uc("guide-item", ig);
    return Ra(b, function(b) {
        return E(b, "serialized-endpoint") == a
    })
}
function ak(a) {
    var b = zi;
    if (t("max-height"))
        Ed(b, a.height - 50 + "px", "max-height");
    else 
        for (var c in"max-height")
            Ed(b, "max-height"[c], c)
}
function bk(a) {
    if (Wj&&!B(document.body, "exp-scrollable-guide")) {
        var b = lk(a), c = Wj.length, d = 0;
        Wj.sort(jg || hb);
        z(Wj, function(a, f) {
            var h = kg(a) - d, h = Math.min(h, b / (c - f));
            b -= (c - f) * h;
            var h = d += h, h = h - (h >= kg(a) ? 0 : a.A ? mg(a, a.A) : 0), h = Math.max(ng(a), h), k = lg(a, a.B);
            if (k > h)
                for (; k > h;) {
                    var m;
                    m = a.B;
                    if (m = void 0 != m.lastElementChild ? m.lastElementChild : nd(m.lastChild, !1)
                        ) {
                            var p = mg(a, m), k = k - p;
                            id(a.k, m, 0)
                        } else 
                            break
            } else if (k < h) {
                for (m = document.createDocumentFragment(); !Ua(ld(a.k));)
                    if (p = md(a.k)) {
                        var x = mg(a, p);
                        if (k + x <= h)
                            k +=
                            x, m.appendChild(p);
                        else 
                            break
                    } else 
                        break;
                a.B.appendChild(m)
            }
            mb(a.C, "empty-overflow-list", !!Ua(ld(a.k)))
        });
        ik()
    }
}
function lk(a) {
    var b = zi, c = O("guide-container");
    if (!c ||!b)
        return 0;
    var d = Vd(c), d = d.top + d.bottom, c = md(c).clientHeight, e = 0;
    z(Wj, function(a) {
        e += a.B ? lg(a, a.B) : 0;
        e += Ua(ld(a.k)) ? 0 : a.A ? mg(a, a.A) : 0
    });
    c = d + c - e;
    return (Gj() ? b.clientHeight : a.height - 50) - c
}
function dk() {
    ff("/guide_channels_ajax?action_load_subs_and_footer=1", {
        format: "JSON",
        V: function(a, b) {
            ek(b)
        }
    })
};
function mk(a, b, c) {
    Ib.call(this);
    this.B = a;
    this.k = c;
    this.g = b || window;
    this.j = u(this.Id, this)
}
y(mk, Ib);
g = mk.prototype;
g.za = null;
g.ic=!1;
g.start = function() {
    this.stop();
    this.ic=!1;
    var a = nk(this), b = ok(this);
    a&&!b && this.g.mozRequestAnimationFrame ? (this.za = Kf(this.g, "MozBeforePaint", this.j), this.g.mozRequestAnimationFrame(null), this.ic=!0) : this.za = a && b ? a.call(this.g, this.j) : this.g.setTimeout(pe(this.j), 20)
};
g.stop = function() {
    if (this.isActive()) {
        var a = nk(this), b = ok(this);
        a&&!b && this.g.mozRequestAnimationFrame ? Rf(this.za) : a && b ? b.call(this.g, this.za) : this.g.clearTimeout(this.za)
    }
    this.za = null
};
g.isActive = function() {
    return null != this.za
};
g.Id = function() {
    this.ic && this.za && Rf(this.za);
    this.za = null;
    this.B.call(this.k, w())
};
g.G = function() {
    this.stop();
    mk.J.G.call(this)
};
function nk(a) {
    a = a.g;
    return a.requestAnimationFrame || a.webkitRequestAnimationFrame || a.mozRequestAnimationFrame || a.oRequestAnimationFrame || a.msRequestAnimationFrame || null
}
function ok(a) {
    a = a.g;
    return a.cancelRequestAnimationFrame || a.webkitCancelRequestAnimationFrame || a.mozCancelRequestAnimationFrame || a.oCancelRequestAnimationFrame || a.msCancelRequestAnimationFrame || null
};
var pk = window, qk = document, rk = pk.location;
function sk() {}
var tk = /\[native code\]/;
function U(a, b, c) {
    return a[b] = a[b] || c
}
function uk(a) {
    for (var b = 0; b < this.length; b++)
        if (this[b] === a)
            return b;
    return - 1
}
function vk(a) {
    a = a.sort();
    for (var b = [], c = void 0, d = 0; d < a.length; d++) {
        var e = a[d];
        e != c && b.push(e);
        c = e
    }
    return b
}
function wk() {
    var a;
    if ((a = Object.create) && tk.test(a))
        a = a(null);
    else {
        a = {};
        for (var b in a)
            a[b] = void 0
    }
    return a
}
var xk = U(pk, "gapi", {});
var yk;
yk = U(pk, "___jsl", wk());
U(yk, "I", 0);
U(yk, "hel", 10);
function zk() {
    var a = rk.href, b;
    if (yk.dpo)
        b = yk.h;
    else {
        b = yk.h;
        var c = RegExp("([#].*&|[#])jsh=([^&#]*)", "g"), d = RegExp("([?#].*&|[?#])jsh=([^&#]*)", "g");
        if (a = a && (c.exec(a) || d.exec(a)))
            try {
                b = decodeURIComponent(a[2])
            } catch (e) {}
    }
    return b
}
function Ak(a) {
    var b = U(yk, "PQ", []);
    yk.PQ = [];
    var c = b.length;
    if (0 === c)
        a();
    else 
        for (var d = 0, e = function() {
            ++d === c && a()
        }, f = 0; f < c; f++)
            b[f](e)
}
function Bk(a) {
    return U(U(yk, "H", wk()), a, wk())
};
var Ck = U(yk, "perf", wk());
U(Ck, "g", wk());
var Dk = U(Ck, "i", wk());
U(Ck, "r", []);
wk();
wk();
function Ek(a, b, c) {
    b && 0 < b.length && (b = Fk(b), c && 0 < c.length && (b += "___" + Fk(c)), 28 < b.length && (b = b.substr(0, 28) + (b.length - 28)), c = b, b = U(Dk, "_p", wk()), U(b, c, wk())[a] = (new Date).getTime(), b = Ck.r, "function" === typeof b ? b(a, "_p", c) : b.push([a, "_p", c]))
}
function Fk(a) {
    return a.join("__").replace(/\./g, "_").replace(/\-/g, "_").replace(/\,/g, "_")
};
var Gk = wk(), Hk = [];
function Ik(a) {
    throw Error("Bad hint" + (a ? ": " + a : ""));
};
Hk.push(["jsl", function(a) {
    for (var b in a)
        if (Object.prototype.hasOwnProperty.call(a, b)) {
            var c = a[b];
            "object" == typeof c ? yk[b] = U(yk, b, []).concat(c) : U(yk, b, c)
        }
    if (b = a.u)
        a = U(yk, "us", []), a.push(b), (b = /^https:(.*)$/.exec(b)) && a.push("http:" + b[1])
}
]);
var Jk = /^(\/[a-zA-Z0-9_\-]+)+$/, Kk = /^[a-zA-Z0-9\-_\.,!]+$/, Lk = /^gapi\.loaded_[0-9]+$/, Mk = /^[a-zA-Z0-9,._-]+$/;
function Nk(a, b, c, d) {
    var e = a.split(";"), f = Gk[e.shift()], h = null;
    f && (h = f(e, b, c, d));
    if (b = h)
        b = h, c = b.match(Ok), d = b.match(Pk), b=!!d && 1 === d.length && Qk.test(b)&&!!c && 1 === c.length;
    b || Ik(a);
    return h
}
function Rk(a, b, c, d) {
    function e(a) {
        return encodeURIComponent(a).replace(/%2C/g, ",")
    }
    a = Sk(a);
    Lk.test(c) || Ik("invalid_callback");
    b = Tk(b);
    d = d && d.length ? Tk(d) : null;
    return [encodeURIComponent(a.Te).replace(/%2C/g, ",").replace(/%2F/g, "/"), "/k=", e(a.version), "/m=", e(b), d ? "/exm=" + e(d): "", "/rt=j/sv=1/d=1/ed=1", a.gd ? "/am=" + e(a.gd): "", a.hd ? "/rs=" + e(a.hd): "", "/cb=", e(c)].join("")
}
function Sk(a) {
    "/" !== a.charAt(0) && Ik("relative path");
    for (var b = a.substring(1).split("/"), c = []; b.length;) {
        a = b.shift();
        if (!a.length || 0 == a.indexOf("."))
            Ik("empty/relative directory");
        else if (0 < a.indexOf("=")) {
            b.unshift(a);
            break
        }
        c.push(a)
    }
    a = {};
    for (var d = 0, e = b.length; d < e; ++d) {
        var f = b[d].split("="), h = decodeURIComponent(f[0]), k = decodeURIComponent(f[1]);
        2 == f.length && h && k && (a[h] = a[h] || k)
    }
    b = "/" + c.join("/");
    Jk.test(b) || Ik("invalid_prefix");
    c = Uk(a, "k", !0);
    d = Uk(a, "am");
    a = Uk(a, "rs");
    return {
        Te: b,
        version: c,
        gd: d,
        hd: a
    }
}
function Tk(a) {
    for (var b = [], c = 0, d = a.length; c < d; ++c) {
        var e = a[c].replace(/\./g, "_").replace(/-/g, "_");
        Mk.test(e) && b.push(e)
    }
    return b.join(",")
}
function Uk(a, b, c) {
    a = a[b];
    !a && c && Ik("missing: " + b);
    if (a) {
        if (Kk.test(a))
            return a;
        Ik("invalid: " + b)
    }
    return null
}
var Qk = /^https?:\/\/[a-z0-9_.-]+\.google\.com(:\d+)?\/[a-zA-Z0-9_.,!=\-\/]+$/, Pk = /\/cb=/g, Ok = /\/\//g;
function Vk() {
    var a = zk();
    if (!a)
        throw Error("Bad hint");
    return a
}
Gk.m = function(a, b, c, d) {
    (a = a[0]) || Ik("missing_hint");
    return "https://apis.google.com" + Rk(a, b, c, d)
};
var Wk = decodeURI("%73cript");
function Xk(a, b) {
    for (var c = [], d = 0; d < a.length; ++d) {
        var e = a[d];
        e && 0 > uk.call(b, e) && c.push(e)
    }
    return c
}
function Yk(a) {
    "loading" != qk.readyState ? Zk(a) : qk.write("<" + Wk + ' src="' + encodeURI(a) + '"></' + Wk + ">")
}
function Zk(a) {
    var b = qk.createElement(Wk);
    b.setAttribute("src", a);
    b.async = "true";
    (a = qk.getElementsByTagName(Wk)[0]) ? a.parentNode.insertBefore(b, a) : (qk.head || qk.body || qk.documentElement).appendChild(b)
}
function $k(a, b) {
    var c = b && b._c;
    if (c)
        for (var d = 0; d < Hk.length; d++) {
            var e = Hk[d][0], f = Hk[d][1];
            f && Object.prototype.hasOwnProperty.call(c, e) && f(c[e], a, b)
        }
}
function al(a, b) {
    bl(function() {
        var c;
        c = b === zk() ? U(xk, "_", wk()) : wk();
        c = U(Bk(b), "_", c);
        a(c)
    })
}
function cl(a, b) {
    var c = b || {};
    "function" == typeof b && (c = {}, c.callback = b);
    $k(a, c);
    var d = a ? a.split(":"): [], e = c.h || Vk(), f = U(yk, "ah", wk());
    if (f["::"] && d.length) {
        for (var h = [], k = null; k = d.shift();) {
            var m = k.split("."), m = f[k] || f[m[1] && "ns:" + m[0] || ""] || e, p = h.length && h[h.length - 1] || null, x = p;
            p && p.hint == m || (x = {
                hint: m,
                features: []
            }, h.push(x));
            x.features.push(k)
        }
        var P = h.length;
        if (1 < P) {
            var I = c.callback;
            I && (c.callback = function() {
                0==--P && I()
            })
        }
        for (; d = h.shift();)
            dl(d.features, c, d.hint)
    } else 
        dl(d || [], c, e)
}
function dl(a, b, c) {
    function d(a, b) {
        if (p)
            return 0;
        pk.clearTimeout(m);
        P.push.apply(P, W);
        var d = ((xk || {}).config || {}).update;
        d ? d(f) : f && U(yk, "cu", []).push(f);
        if (b) {
            Ek("me0", a, I);
            try {
                al(b, c)
            } finally {
                Ek("me1", a, I)
            }
        }
        return 1
    }
    a = vk(a) || [];
    var e = b.callback, f = b.config, h = b.timeout, k = b.ontimeout, m = null, p=!1;
    if (h&&!k ||!h && k)
        throw "Timeout requires both the timeout parameter and ontimeout parameter to be set";
    var x = U(Bk(c), "r", []).sort(), P = U(Bk(c), "L", []).sort(), I = [].concat(x);
    0 < h && (m = pk.setTimeout(function() {
        p=!0;
        k()
    }, h));
    var W = Xk(a, P);
    if (W.length) {
        var W = Xk(a, x), ra = U(yk, "CP", []), lb = ra.length;
        ra[lb] = function(a) {
            function b() {
                var a = ra[lb + 1];
                a && a()
            }
            function c(b) {
                ra[lb] = null;
                d(W, a) && Ak(function() {
                    e && e();
                    b()
                })
            }
            if (!a)
                return 0;
            Ek("ml1", W, I);
            0 < lb && ra[lb - 1] ? ra[lb] = function() {
                c(b)
            } : c(b)
        };
        if (W.length) {
            var M = "loaded_" + yk.I++;
            xk[M] = function(a) {
                ra[lb](a);
                xk[M] = null
            };
            a = Nk(c, W, "gapi." + M, x);
            x.push.apply(x, W);
            Ek("ml0", W, I);
            b.sync || pk.___gapisync ? Yk(a) : Zk(a)
        } else 
            ra[lb](sk)
    } else 
        d(W) && e && e()
};
function bl(a) {
    if (yk.hee && 0 < yk.hel)
        try {
            return a()
    } catch (b) {
        yk.hel--, cl("debug_error", function() {
            try {
                window.___jsl.hefn(b)
            } catch (a) {
                throw b;
            }
        })
    } else 
        return a()
};
xk.load = function(a, b) {
    return bl(function() {
        return cl(a, b)
    })
};
function el(a) {
    a = fa(a) ? {
        callback: a
    } : a || {};
    a._c && a._c.jsl && a._c.jsl.h || Cb(a, {
        _c: {
            jsl: {
                h: F("GAPI_HINT_PARAMS")
            }
        }
    });
    if (F("GAPI_HINT_OVERRIDE")) {
        var b = Xe(document.location.href).gapi_jsh;
        b && Cb(a, {
            _c: {
                jsl: {
                    h: b
                }
            }
        })
    }
    cl("iframes", a)
};
function fl() {
    var a = {
        action_get_delegate_accounts: 1,
        owner_picker_redirect_url: F("OWNER_PICKER_REDIRECT_URL"),
        session_token: Pb()
    };
    a.o = F("CREATOR_CONTEXT", "U");
    return a
};
function gl(a, b) {
    a=!!a;
    q("_lactCookie", a, window);
    if (null == r("_lact", window)) {
        if (a && b) {
            var c = gh("ACTIVITY", "-1");
            q("_lact", parseInt(c, 10), window)
        } else 
            q("_lact", - 1, window), hl();
        R(document, "keypress", hl);
        R(document, "mousedown", hl);
        R(document, "mouseup", hl)
    }
}
function hl() {
    var a = r("_lact", window);
    null == a && (gl(), a = r("_lact", window));
    var b = w();
    q("_lact", b, window);
    r("_lactCookie", window) && 1E3 <= b - a && fh("ACTIVITY", "" + b)
}
function il() {
    var a = r("_lact", window);
    return null == a?-1 : Math.max(w() - a, 0)
};
function jl(a, b, c, d, e, f) {
    this.g = null;
    this.D = c;
    this.B = a;
    this.M = b;
    this.k = d;
    this.A = F("GOOGLEPLUS_HOST") + (e ? "/u/" + e : "") + (f ? "/b/" + f : "") + "/_/notifications/frame";
    this.j = O(a)
}
function kl(a, b, c, d) {
    return {
        onOpen: u(function(a) {
            return a.openInto(c)
        }, a),
        onReady: v(function(a) {
            a && a()
        }, d ? b.showOnepick : void 0),
        onClose: v(function(a, b) {
            a && a();
            b.remove()
        }, d ? b.hideOnepick : void 0)
    }
}
function ll(a, b, c) {
    a && a[b] && a[b].apply(a, Array.prototype.slice.call(arguments, 2))
}
function ml(a, b) {
    var c = {
        setNotificationWidgetSize: u(function(a, b) {
            this.j.style.width = a;
            this.j.style.height = b
        }, a),
        setNotificationWidgetHeight: u(function(a) {
            this.j.style.height = a
        }, a),
        setNotificationText: u(function(a) {
            this.D(parseInt(a, 10))
        }, a),
        hideNotificationWidget: v(function(a) {
            a && a()
        }, b.hideNotificationWidget),
        openSharebox: function() {},
        onError: function() {}
    };
    a.g = iframes.open(a.A, {
        style: "iframe-style"
    }, {
        origin: window.location.protocol + "//" + window.location.hostname,
        source: "yt",
        sourceid: "36",
        hl: a.k
    },
    c, function() {})
}
jl.prototype.load = function(a) {
    iframes.setHandler("iframe-style", kl(this, a, this.B, !1));
    iframes.setHandler("onepick", kl(this, a, this.M, !0));
    ml(this, a)
};
jl.prototype.close = function() {
    ll(this.g, "onHide")
};
jl.prototype.C = function() {
    return Yc(window).height - 60 - 20
};
function nl(a, b) {
    ll(a.g, b ? "onIdle" : "onActive")
};
function ol(a, b, c) {
    this.A=!1;
    this.B = 0;
    this.k = O("sb-container");
    if (this.g = O("sb-button-notify"))
        this.M = be("SPAN", "yt-uix-button-content", this.g), this.D = be("IMG", "yt-uix-button-icon-bell", this.g);
    this.C = O("sb-onepick-target");
    this.j = new jl("sb-target", "sb-onepick-target", u(this.Cd, this), a, b, c);
    this.j.load({
        hideNotificationWidget: u(this.Ac, this),
        showOnepick: u(this.Bd, this),
        hideOnepick: u(this.zd, this)
    });
    this.$ = Qd(this.C);
    this.Bc();
    R(window, "resize", u(this.Bc, this));
    R(window, "click", u(this.Ac, this));
    gl();
    Rb(u(this.yd, this), 12E4)
}
g = ol.prototype;
g.$e = function() {
    this.A ? (pl(this), nl(this.j, !0)) : (G(u(this.Fe, this), 0), nl(this.j, !1))
};
g.Fe = function() {
    this.j.close();
    ql(this, !0);
    C(this.g, "sb-notif-clicked");
    var a = this.j, b = {
        maxWidgetHeight: a.C()
    };
    ll(a.g, "onShowNotificationsOnly", b);
    this.A=!0
};
function ql(a, b) {
    b ? (ob(a.k, "sb-off", "sb-on"), C(a.k, "sb-card-notif")) : (ob(a.k, "sb-on", "sb-off"), D(a.k, "sb-card-notif"))
}
function pl(a) {
    ql(a, !1);
    a.A=!1;
    D(a.g, "sb-notif-clicked")
}
g.Ac = function() {
    this.A && (this.j.close(), pl(this), rl(this))
};
g.Cd = function(a) {
    this.B = a;
    rl(this)
};
function rl(a) {
    a.M && sd(a.M, a.B + "");
    if (a.g) {
        0 == a.B ? ob(a.g, "sb-notif-on", "sb-notif-off") : ob(a.g, "sb-notif-off", "sb-notif-on");
        var b;
        b = Ub(a.B);
        a.D && a.D.setAttribute("alt", b)
    }
}
g.Bd = function() {
    ob(this.C, "sb-off", "sb-on")
};
g.zd = function() {
    ob(this.C, "sb-on", "sb-off")
};
g.Bc = function() {
    var a = Math.max((Yc(window).height - this.$.height) / 2, 0);
    this.C.style.top = a + "px"
};
g.yd = function() {
    6E5 < il() ? nl(this.j, !1) : nl(this.j, !0)
};
var sl = "";
function tl() {
    el(function() {
        var a = O("sb-button-notify"), b = new ol(F("SANDBAR_LOCALE"), F("SESSION_INDEX"), F("DELEGATED_SESSION_ID"));
        R(a, "click", u(b.$e, b));
        K("sandbar-init")
    })
}
function ul() {
    if ("U" == F("CREATOR_CONTEXT", "U")) {
        var a = Uc("yt-masthead-account-picker-user-option"), b = Qe(window.location.href);
        z(a, function(a) {
            Da(a.href, "action_handle_signin") && (a.href = Qe(Ze(a.href, {
                next: b
            })))
        })
    } else {
        var a = Uc("yt-masthead-account-picker-owner-option"), c = Pe();
        if (c) {
            var d = c.indexOf(";");
            - 1 != d && (c = c.substring(0, d))
        }
        c && z(a, function(a) {
            var b = Xe(a.href).next;
            b && (b = Re(b) + (c ? "#" + c : ""), a.href = Qe(Ze(a.href, {
                next: b
            })))
        })
    }
}
function vl() {
    D(document.body, "sitewide-ticker-visible");
    K("masthead-ticker-close")
}
function wl() {
    var a = F("SBOX_JS_URL") || sl;
    if (a) {
        sl = a;
        var b = r("yt.www.masthead.searchbox.init");
        b ? b() : xl(a, function() {
            var a;
            try {
                a = r("yt.www.masthead.searchbox.init"), a()
            } catch (b) {
                throw b.message = b.message + ' sbox type: "' + ca(a) + '"', b;
            }
        })
    }
}
function xl(a, b) {
    var c = O("masthead-search-term");
    if (c) {
        var d, e, f = function() {
            yl();
            wj = b;
            fc(a, b);
            le([d, e])
        };
        d = ke(c, "mouseover", f);
        e = ke(c, "keypress", f)
    }
}
function yl() {
    r("yt.www.masthead.searchbox.init") || (mc(sl, wj), wj = null)
};
var zl, Al, Bl, Cl;
var Dl=!1, El=!1;
function Fl() {
    if (zl = O("masthead-appbar")) {
        Al = O("masthead-positioner");
        Bl = O("masthead-positioner-height-offset");
        Aj();
        Gl();
        Hl();
        Il = new mk(Jl);
        var a = Kl;
        r("yt.scheduler.instance") && (a = function() {
            var a = Kl, c = void 0;
            void 0 === c && (c = NaN);
            var d = r("yt.scheduler.instance.addJob");
            d ? (isNaN(c) && (c = 0), d(a, 0, c)) : isNaN(c) ? a() : G(a, c || 0)
        });
        Ll.push(J("init", a));
        Ll.push(J("dispose", Ml));
        Nl = zj();
        ih.getInstance();
        Ol(nh(0, 128) ? nh(0, 129) : !0);
        Pl = Vc("html", void 0, void 0)[0];
        Ll.push(J("masthead-ticker-close", Ql));
        Ll.push(J("appbar-guide-delay-load",
        Gl));
        Ll.push(J("page-resize", Rl));
        Dl || Kl()
    }
}
function Kl() {
    Q("appbar-content-hidable");
    Sl = Q("appbar-content-trigger");
    (El = B(document.body, "always-autohide-masthead")) ? Gb(xj, "position-fixed") : Eb(xj, "position-fixed", "true");
    El || Sl ? Tl = R(window, "scroll", Ul) : Vl();
    Cl=!!O("appbar-guide-menu") && Fj();
    Dl ? (Cl || (mb(Bj, "guide-pinned", !1), Dj(!1)), Rl()) : Dl=!0;
    var a =- 1 * Od(Al).y, b = document.body.scrollHeight, c = a - (b - document.body.clientHeight);
    0 < c && (document.body.style.minHeight = b + c + "px");
    ad(document).scrollTop += a;
    Wl = $c(document).y;
    Xl();
    Yl();
    ik()
}
function Ml() {
    Sl = null;
    Vl()
}
function Xl() {
    if (Sl&&!Zl) {
        var a = Qd(Sl).height, b = Qd(Al).height, b = Od(Al).y + b, a = Math.floor(Od(Sl).y) + a, c = B(document.body, "appbar-hidden"), d = ad(document);
        c && a < b + 40 ? (d.scrollTop += 40, kf(zl, !0), D(document.body, "appbar-hidden"), $l()) : !c && b < a && (d.scrollTop -= 40, C(document.body, "appbar-hidden"), $l())
    }
}
function $l() {
    Zl=!0;
    G(function() {
        Zl=!1;
        B(document.body, "appbar-hidden") && kf(zl, !1)
    }, 300)
}
var Zl=!1;
function Rl() {
    Cl && (mb(Bj, "guide-pinned", 1251 <= (window.innerWidth || document.documentElement.clientWidth)), Dj(Gj() && am));
    bm()
}
function bm() {
    if (B(document.body, "flex-width-enabled-snap")) {
        var a;
        a = Ej();
        var b = (window.innerWidth || document.documentElement.clientWidth) - 21 - 50;
        1251 <= (window.innerWidth || document.documentElement.clientWidth) && a && (b -= 230);
        a = 1262 <= b ? "content-snap-width-3" : 1056 <= b ? "content-snap-width-2" : "content-snap-width-1";
        B(Pl, a) || (kb(Pl, Nl), C(Pl, a), K("yt-dom-content-change"))
    } else 
        kb(Pl, Nl)
}
function cm() {
    return O("appbar-guide-menu")
}
function Hl() {
    var a = B(document.body, "appbar-hidden"), b = Od(Al).y, c = Qd(Al).height;
    dm = a ? b + c : b + c - 40
}
function em(a) {
    Gj() || yd(a.target, "appbar-guide-clickable-ancestor") || Dj(!1)
}
function Jl() {
    var a = Math.max(0, $c(document).y), b = Wl;
    Wl = a;
    var c = b - a;
    0 < c && 0 == fm ? fm = b : 0 > c && (fm = 0);
    c = Math.min(0, Math.max(c + gm, - dm));
    var d = El;
    if (d) {
        var e = a < dm, b = a > b && Math.abs(gm) < dm;
        Ej() && b ? a=!1 : (a = d?!1 : 100 <= fm - a, a = e || b || a)
    } else 
        a=!1;
    a && gm != c && (Al.style.top = c + "px", hm = gm = c, im());
    Xl()
}
function im() {
    var a = cm();
    a && (a.style.top = hm + "px");
    ik()
}
function Gl() {
    if (!jm) {
        var a = cm();
        if (!a || Q("guide-module-loading", a)) {
            var b = O("appbar-guide-button");
            if (!b)
                return;
            km.push(ke(b, "click", function() {
                K("appbar-show-guide")
            }))
        }
        im();
        km.push(me(a, "click", Yl, "guide-item"));
        km.push(R(a, "mouseleave", lm));
        km.push(R(document.body, "click", em));
        a = Q("appbar-guide-toggle");
        km.push(R(a, "click", mm));
        jm=!0
    }
}
function Yl() {
    nm=!Gj()
}
function lm() {
    nm && (Dj(!1), nm=!1)
}
function mm() {
    var a=!Ej();
    Dj(a);
    a && K("yt-dom-content-change");
    Gj() && Ol(a);
    bm()
}
function Ol(a) {
    oh(128, !0);
    oh(129, a);
    fh("PREF", ph(), 63072E3);
    am = a
}
function Vl() {
    le(Tl);
    Il && Il.stop();
    Tl = ""
}
function Ul() {
    Il.isActive() || Il.start()
}
function Ql() {
    var a = Od(Al).y, b = Qd(Al).height;
    Bl.style.height = a + b + "px";
    var c = cm();
    c && (B(document.body, "appbar-hidden") ? c.style.marginTop = a + b + "px" : c.style.marginTop = a + b - 40 + "px", ik());
    Hl()
}
var km = [], Tl = "", Ll = [], Wl = 0, dm = 0, nm=!1, jm=!1, fm = 0, gm = 0, hm = 0, Il = null, Sl = null, Nl = null, am=!1, Pl = null;
function om() {};
function pm(a) {
    a && (this.id = a.id || a.name, this.name = a.name, this.app = a.app, this.type = a.type || "REMOTE_CONTROL", this.avatar = a.userAvatarUri || "", this.theme = a.theme || "u")
}
pm.prototype.id = "";
pm.prototype.name = "";
g = pm.prototype;
g.app = "";
g.type = "REMOTE_CONTROL";
g.avatar = "";
g.theme = "u";
g.equals = function(a) {
    return a ? this.id == a.id : !1
};
function qm(a, b) {
    this.action = a;
    this.params = b || null
};
function rm() {
    this.g = w()
}
new rm;
rm.prototype.set = function(a) {
    this.g = a
};
rm.prototype.get = function() {
    return this.g
};
function sm(a, b) {
    this.j = new Ge(a);
    this.g = b ? Ee : De
}
sm.prototype.stringify = function(a) {
    return Fe(this.j, a)
};
sm.prototype.parse = function(a) {
    return this.g(a)
};
function tm() {}
tm.prototype.g = null;
function um(a) {
    var b;
    (b = a.g) || (b = {}, vm(a) && (b[0]=!0, b[1]=!0), b = a.g = b);
    return b
};
var wm;
function xm() {}
y(xm, tm);
function ym(a) {
    return (a = vm(a)) ? new ActiveXObject(a) : new XMLHttpRequest
}
function vm(a) {
    if (!a.j && "undefined" == typeof XMLHttpRequest && "undefined" != typeof ActiveXObject) {
        for (var b = ["MSXML2.XMLHTTP.6.0", "MSXML2.XMLHTTP.3.0", "MSXML2.XMLHTTP", "Microsoft.XMLHTTP"], c = 0; c < b.length; c++) {
            var d = b[c];
            try {
                return new ActiveXObject(d), a.j = d
            } catch (e) {}
        }
        throw Error("Could not create ActiveXObject. ActiveX might be disabled, or MSXML might not be installed");
    }
    return a.j
}
wm = new xm;
function zm(a, b, c, d, e) {
    this.g = a;
    this.k = c;
    this.D = d;
    this.M = e || 1;
    this.A = 45E3;
    this.B = new Vf(this);
    this.j = new Xg;
    Yg(this.j, 250)
}
g = zm.prototype;
g.Ua = null;
g.ra=!1;
g.qb = null;
g.uc = null;
g.Ab = null;
g.ob = null;
g.Ka = null;
g.Qa = null;
g.Va = null;
g.Y = null;
g.Fb = 0;
g.ua = null;
g.Xb = null;
g.Xa = null;
g.ub =- 1;
g.Mc=!0;
g.Ya=!1;
g.hc = 0;
g.Ob = null;
var Am = {}, Bm = {};
g = zm.prototype;
g.setTimeout = function(a) {
    this.A = a
};
function Cm(a, b, c) {
    a.ob = 1;
    a.Ka = Jh(b.clone());
    a.Va = c;
    a.C=!0;
    Dm(a, null)
}
function Em(a, b, c, d, e) {
    a.ob = 1;
    a.Ka = Jh(b.clone());
    a.Va = null;
    a.C = c;
    e && (a.Mc=!1);
    Dm(a, d)
}
function Dm(a, b) {
    a.Ab = w();
    Fm(a);
    a.Qa = a.Ka.clone();
    Hh(a.Qa, "t", a.M);
    a.Fb = 0;
    a.Y = a.g.qc(a.g.zb() ? b : null);
    0 < a.hc && (a.Ob = new $g(u(a.Pc, a, a.Y), a.hc));
    a.B.listen(a.Y, "readystatechange", a.pe);
    var c = a.Ua ? yb(a.Ua): {};
    a.Va ? (a.Xb = "POST", c["Content-Type"] = "application/x-www-form-urlencoded", a.Y.send(a.Qa, a.Xb, a.Va, c)) : (a.Xb = "GET", a.Mc&&!Ac && (c.Connection = "close"), a.Y.send(a.Qa, a.Xb, null, c));
    a.g.qa(1)
}
g.pe = function(a) {
    a = a.target;
    var b = this.Ob;
    b && 3 == Gm(a) ? b.jb || b.zc ? b.bc=!0 : ah(b) : this.Pc(a)
};
g.Pc = function(a) {
    try {
        if (a == this.Y)
            t: {
            var b = Gm(this.Y), c = this.Y.B, d = this.Y.getStatus();
            if (N&&!Jc(10) || Ac&&!Ic("420+")) {
                if (4 > b)
                    break t
            } else if (3 > b || 3 == b&&!yc&&!Hm(this.Y))
                break t;
                this.Ya || 4 != b || 7 == c || (8 == c || 0 >= d ? this.g.qa(3) : this.g.qa(2));
                Im(this);
                var e = this.Y.getStatus();
                this.ub = e;
                var f = Hm(this.Y);
                (this.ra = 200 == e) ? (4 == b && Jm(this), this.C ? (Km(this, b, f), yc && this.ra && 3 == b && (this.B.listen(this.j, "tick", this.Ld), this.j.start())) : Lm(this, f), this.ra&&!this.Ya && (4 == b ? this.g.Nb(this) : (this.ra=!1, Fm(this)))) :
                (this.Xa = 400 == e && 0 < f.indexOf("Unknown SID") ? 3 : 0, V(), Jm(this), Mm(this))
            }
    } catch (h) {
        this.Y && Hm(this.Y)
    } finally {}
};
function Km(a, b, c) {
    for (var d=!0; !a.Ya && a.Fb < c.length;) {
        var e = Nm(a, c);
        if (e == Bm) {
            4 == b && (a.Xa = 4, V(), d=!1);
            break
        } else if (e == Am) {
            a.Xa = 4;
            V();
            d=!1;
            break
        } else 
            Lm(a, e)
    }
    4 == b && 0 == c.length && (a.Xa = 1, V(), d=!1);
    a.ra = a.ra && d;
    d || (Jm(a), Mm(a))
}
g.Ld = function() {
    var a = Gm(this.Y), b = Hm(this.Y);
    this.Fb < b.length && (Im(this), Km(this, a, b), this.ra && 4 != a && Fm(this))
};
function Nm(a, b) {
    var c = a.Fb, d = b.indexOf("\n", c);
    if ( - 1 == d)
        return Bm;
    c = Number(b.substring(c, d));
    if (isNaN(c))
        return Am;
    d += 1;
    if (d + c > b.length)
        return Bm;
    var e = b.substr(d, c);
    a.Fb = d + c;
    return e
}
function Om(a, b) {
    a.Ab = w();
    Fm(a);
    var c = b ? window.location.hostname: "";
    a.Qa = a.Ka.clone();
    Gh(a.Qa, "DOMAIN", c);
    Gh(a.Qa, "t", a.M);
    try {
        a.ua = new ActiveXObject("htmlfile")
    } catch (d) {
        Jm(a);
        a.Xa = 7;
        V();
        Mm(a);
        return 
    }
    var e = "<html><body>";
    b && (e += '<script>document.domain="' + c + '"\x3c/script>');
    e += "</body></html>";
    a.ua.open();
    a.ua.write(e);
    a.ua.close();
    a.ua.parentWindow.m = u(a.qe, a);
    a.ua.parentWindow.d = u(a.Rc, a, !0);
    a.ua.parentWindow.rpcClose = u(a.Rc, a, !1);
    c = a.ua.createElement("div");
    a.ua.parentWindow.document.body.appendChild(c);
    c.innerHTML = '<iframe src="' + a.Qa + '"></iframe>';
    a.g.qa(1)
}
g.qe = function(a) {
    Pm(u(this.Xe, this, a), 0)
};
g.Xe = function(a) {
    this.Ya || (Im(this), Lm(this, a), Fm(this))
};
g.Rc = function(a) {
    Pm(u(this.We, this, a), 0)
};
g.We = function(a) {
    this.Ya || (Jm(this), this.ra = a, this.g.Nb(this), this.g.qa(4))
};
g.cancel = function() {
    this.Ya=!0;
    Jm(this)
};
function Fm(a) {
    a.uc = w() + a.A;
    Qm(a, a.A)
}
function Qm(a, b) {
    if (null != a.qb)
        throw Error("WatchDog timer not null");
    a.qb = Pm(u(a.Ue, a), b)
}
function Im(a) {
    a.qb && (l.clearTimeout(a.qb), a.qb = null)
}
g.Ue = function() {
    this.qb = null;
    var a = w();
    0 <= a - this.uc ? (2 != this.ob && this.g.qa(3), Jm(this), this.Xa = 2, V(), Mm(this)) : Qm(this, this.uc - a)
};
function Mm(a) {
    a.g.Vc() || a.Ya || a.g.Nb(a)
}
function Jm(a) {
    Im(a);
    Kb(a.Ob);
    a.Ob = null;
    a.j.stop();
    a.B.removeAll();
    if (a.Y) {
        var b = a.Y;
        a.Y = null;
        Rm(b);
        b.dispose()
    }
    a.ua && (a.ua = null)
}
function Lm(a, b) {
    try {
        a.g.Xc(a, b), a.g.qa(4)
    } catch (c) {}
};
function Sm(a, b, c, d, e) {
    if (0 == d)
        c(!1);
    else {
        var f = e || 0;
        d--;
        Tm(a, b, function(e) {
            e ? c(!0) : l.setTimeout(function() {
                Sm(a, b, c, d, f)
            }, f)
        })
    }
}
function Tm(a, b, c) {
    var d = new Image;
    d.onload = function() {
        try {
            Um(d), c(!0)
        } catch (a) {}
    };
    d.onerror = function() {
        try {
            Um(d), c(!1)
        } catch (a) {}
    };
    d.onabort = function() {
        try {
            Um(d), c(!1)
        } catch (a) {}
    };
    d.ontimeout = function() {
        try {
            Um(d), c(!1)
        } catch (a) {}
    };
    l.setTimeout(function() {
        if (d.ontimeout)
            d.ontimeout()
    }, b);
    d.src = a
}
function Um(a) {
    a.onload = null;
    a.onerror = null;
    a.onabort = null;
    a.ontimeout = null
};
function Vm(a) {
    this.g = a;
    this.j = new sm(null, !0)
}
g = Vm.prototype;
g.fc = null;
g.ia = null;
g.Sb=!1;
g.Hc = null;
g.Rb = null;
g.lc = null;
g.tc = null;
g.ma = null;
g.Ha =- 1;
g.Db = null;
g.Eb = null;
g.connect = function(a) {
    this.tc = a;
    a = Wm(this.g, null, this.tc);
    V();
    this.Hc = w();
    var b = this.g.D;
    null != b ? (this.Db = b[0], (this.Eb = b[1]) ? (this.ma = 1, Xm(this)) : (this.ma = 2, Ym(this))) : (Hh(a, "MODE", "init"), this.ia = new zm(this, 0, void 0, void 0, void 0), this.ia.Ua = this.fc, Em(this.ia, a, !1, null, !0), this.ma = 0)
};
function Xm(a) {
    var b = Wm(a.g, a.Eb, "/mail/images/cleardot.gif");
    Jh(b);
    Sm(b.toString(), 5E3, u(a.we, a), 3, 2E3);
    a.qa(1)
}
g.we = function(a) {
    if (a)
        this.ma = 2, Ym(this);
    else {
        V();
        var b = this.g;
        b.pa = b.Na.Ha;
        Zm(b, 9)
    }
    a && this.qa(2)
};
function Ym(a) {
    var b = a.g.$;
    if (null != b)
        V(), b ? (V(), $m(a.g, a, !1)) : (V(), $m(a.g, a, !0));
    else if (a.ia = new zm(a, 0, void 0, void 0, void 0), a.ia.Ua = a.fc, b = a.g, b = Wm(b, b.zb() ? a.Db : null, a.tc), V(), !N || Jc(10))
        Hh(b, "TYPE", "xmlhttp"), Em(a.ia, b, !1, a.Db, !1);
    else {
        Hh(b, "TYPE", "html");
        var c = a.ia;
        a = Boolean(a.Db);
        c.ob = 3;
        c.Ka = Jh(b.clone());
        Om(c, a)
    }
}
g.qc = function(a) {
    return this.g.qc(a)
};
g.Vc = function() {
    return !1
};
g.Xc = function(a, b) {
    this.Ha = a.ub;
    if (0 == this.ma)
        if (b) {
            try {
                var c = this.j.parse(b)
            } catch (d) {
                c = this.g;
                c.pa = this.Ha;
                Zm(c, 2);
                return 
            }
            this.Db = c[0];
            this.Eb = c[1]
    } else 
        c = this.g, c.pa = this.Ha, Zm(c, 2);
    else if (2 == this.ma)
        if (this.Sb)
            V(), this.lc = w();
        else if ("11111" == b) {
            if (V(), this.Sb=!0, this.Rb = w(), c = this.Rb - this.Hc, !N || Jc(10) || 500 > c)
                this.Ha = 200, this.ia.cancel(), V(), $m(this.g, this, !0)
        } else 
            V(), this.Rb = this.lc = w(), this.Sb=!1
    };
g.Nb = function() {
    this.Ha = this.ia.ub;
    if (this.ia.ra)
        0 == this.ma ? this.Eb ? (this.ma = 1, Xm(this)) : (this.ma = 2, Ym(this)) : 2 == this.ma && (a=!1, (a=!N || Jc(10) ? this.Sb : 200 > this.lc - this.Rb?!1 : !0) ? (V(), $m(this.g, this, !0)) : (V(), $m(this.g, this, !1)));
    else {
        0 == this.ma ? V() : 2 == this.ma && V();
        var a = this.g;
        a.pa = this.Ha;
        Zm(a, 2)
    }
};
g.zb = function() {
    return this.g.zb()
};
g.isActive = function() {
    return this.g.isActive()
};
g.qa = function(a) {
    this.g.qa(a)
};
function an(a) {
    Xf.call(this);
    this.headers = new ve;
    this.X = a || null;
    this.j=!1;
    this.S = this.g = null;
    this.ka = this.H = "";
    this.B = 0;
    this.A = "";
    this.k = this.ba = this.D = this.Z=!1;
    this.C = 0;
    this.O = null;
    this.Aa = "";
    this.P = this.Ba=!1
}
y(an, Xf);
var bn = /^https?$/i, cn = ["POST", "PUT"];
g = an.prototype;
g.send = function(a, b, c, d) {
    if (this.g)
        throw Error("[goog.net.XhrIo] Object is active with another request=" + this.H + "; newUri=" + a);
    b = b ? b.toUpperCase() : "GET";
    this.H = a;
    this.A = "";
    this.B = 0;
    this.ka = b;
    this.Z=!1;
    this.j=!0;
    this.g = this.X ? ym(this.X) : ym(wm);
    this.S = this.X ? um(this.X) : um(wm);
    this.g.onreadystatechange = u(this.Cc, this);
    try {
        om(dn(this, "Opening Xhr")), this.ba=!0, this.g.open(b, String(a), !0), this.ba=!1
    } catch (e) {
        om(dn(this, "Error opening Xhr: " + e.message));
        en(this, e);
        return 
    }
    a = c || "";
    var f = this.headers.clone();
    d && rh(d, function(a, b) {
        f.set(b, a)
    });
    d = Ra(f.Ca(), fn);
    c = l.FormData && a instanceof l.FormData;
    !Ta(cn, b) || d || c || f.set("Content-Type", "application/x-www-form-urlencoded;charset=utf-8");
    f.forEach(function(a, b) {
        this.g.setRequestHeader(b, a)
    }, this);
    this.Aa && (this.g.responseType = this.Aa);
    "withCredentials"in this.g && (this.g.withCredentials = this.Ba);
    try {
        gn(this), 0 < this.C && (this.P = hn(this.g), om(dn(this, "Will abort after " + this.C + "ms if incomplete, xhr2 " + this.P)), this.P ? (this.g.timeout = this.C, this.g.ontimeout = u(this.Dc,
        this)) : this.O = Zg(this.Dc, this.C, this)), om(dn(this, "Sending request")), this.D=!0, this.g.send(a), this.D=!1
    } catch (h) {
        om(dn(this, "Send error: " + h.message)), en(this, h)
    }
};
function hn(a) {
    return N && Ic(9) && "number" == typeof a.timeout && n(a.ontimeout)
}
function fn(a) {
    return "content-type" == a.toLowerCase()
}
g.Dc = function() {
    "undefined" != typeof aa && this.g && (this.A = "Timed out after " + this.C + "ms, aborting", this.B = 8, dn(this, this.A), Yf(this, "timeout"), Rm(this, 8))
};
function en(a, b) {
    a.j=!1;
    a.g && (a.k=!0, a.g.abort(), a.k=!1);
    a.A = b;
    a.B = 5;
    jn(a);
    kn(a)
}
function jn(a) {
    a.Z || (a.Z=!0, Yf(a, "complete"), Yf(a, "error"))
}
function Rm(a, b) {
    a.g && a.j && (dn(a, "Aborting"), a.j=!1, a.k=!0, a.g.abort(), a.k=!1, a.B = b || 7, Yf(a, "complete"), Yf(a, "abort"), kn(a))
}
g.G = function() {
    this.g && (this.j && (this.j=!1, this.k=!0, this.g.abort(), this.k=!1), kn(this, !0));
    an.J.G.call(this)
};
g.Cc = function() {
    this.L() || (this.ba || this.D || this.k ? ln(this) : this.He())
};
g.He = function() {
    ln(this)
};
function ln(a) {
    if (a.j && "undefined" != typeof aa)
        if (a.S[1] && 4 == Gm(a) && 2 == a.getStatus())
            dn(a, "Local request error detected and ignored");
        else if (a.D && 4 == Gm(a))
            Zg(a.Cc, 0, a);
        else if (Yf(a, "readystatechange"), 4 == Gm(a)) {
            dn(a, "Request complete");
            a.j=!1;
            try {
                var b = a.getStatus(), c;
                t:
                switch (b) {
                case 200:
                case 201:
                case 202:
                case 204:
                case 206:
                case 304:
                case 1223:
                    c=!0;
                    break t;
                default:
                    c=!1
                }
                var d;
                if (!(d = c)) {
                    var e;
                    if (e = 0 === b) {
                        var f = Me(String(a.H))[1] || null;
                        if (!f && self.location)
                            var h = self.location.protocol, f = h.substr(0, h.length -
                            1);
                            e=!bn.test(f ? f.toLowerCase() : "")
                        }
                        d = e
                }
                if (d)
                    Yf(a, "complete"), Yf(a, "success");
                else {
                    a.B = 6;
                    var k;
                    try {
                        k = 2 < Gm(a) ? a.g.statusText : ""
                    } catch (m) {
                        k = ""
                    }
                    a.A = k + " [" + a.getStatus() + "]";
                    jn(a)
                }
            } finally {
                kn(a)
            }
    }
}
function kn(a, b) {
    if (a.g) {
        gn(a);
        var c = a.g, d = a.S[0] ? s: null;
        a.g = null;
        a.S = null;
        b || Yf(a, "ready");
        try {
            c.onreadystatechange = d
        } catch (e) {}
    }
}
function gn(a) {
    a.g && a.P && (a.g.ontimeout = null);
    "number" == typeof a.O && (l.clearTimeout(a.O), a.O = null)
}
g.isActive = function() {
    return !!this.g
};
function Gm(a) {
    return a.g ? a.g.readyState : 0
}
g.getStatus = function() {
    try {
        return 2 < Gm(this) ? this.g.status : - 1
    } catch (a) {
        return - 1
    }
};
function Hm(a) {
    try {
        return a.g ? a.g.responseText : ""
    } catch (b) {
        return ""
    }
}
function dn(a, b) {
    return b + " [" + a.ka + " " + a.H + " " + a.getStatus() + "]"
};
function mn(a, b, c) {
    this.M = a || null;
    this.g = 1;
    this.j = [];
    this.B = [];
    this.A = new sm(null, !0);
    this.D = b || null;
    this.$ = null != c ? c : null
}
function nn(a, b) {
    this.g = a;
    this.map = b;
    this.context = null
}
g = mn.prototype;
g.Hb = null;
g.ea = null;
g.W = null;
g.dc = null;
g.Kb = null;
g.Fc = null;
g.Pb = null;
g.yb = 0;
g.oe = 0;
g.aa = null;
g.Ra = null;
g.Fa = null;
g.Za = null;
g.Na = null;
g.Mb = null;
g.ib =- 1;
g.Lc =- 1;
g.pa =- 1;
g.xb = 0;
g.mb = 0;
g.Wa = 8;
var on = new Xf;
function pn(a) {
    wf.call(this, "statevent", a)
}
y(pn, wf);
function qn(a, b) {
    wf.call(this, "timingevent", a);
    this.size = b
}
y(qn, wf);
function rn(a) {
    wf.call(this, "serverreachability", a)
}
y(rn, wf);
g = mn.prototype;
g.connect = function(a, b, c, d, e) {
    V();
    this.dc = b;
    this.Hb = c || {};
    d && n(e) && (this.Hb.OSID = d, this.Hb.OAID = e);
    this.Na = new Vm(this);
    this.Na.fc = null;
    this.Na.j = this.A;
    this.Na.connect(a)
};
g.disconnect = function() {
    sn(this);
    if (3 == this.g) {
        var a = this.yb++, b = this.Kb.clone();
        Gh(b, "SID", this.k);
        Gh(b, "RID", a);
        Gh(b, "TYPE", "terminate");
        tn(this, b);
        a = new zm(this, 0, this.k, a, void 0);
        a.ob = 2;
        a.Ka = Jh(b.clone());
        (new Image).src = a.Ka;
        a.Ab = w();
        Fm(a)
    }
    un(this)
};
function sn(a) {
    if (a.Na) {
        var b = a.Na;
        b.ia && (b.ia.cancel(), b.ia = null);
        b.Ha =- 1;
        a.Na = null
    }
    a.W && (a.W.cancel(), a.W = null);
    a.Fa && (l.clearTimeout(a.Fa), a.Fa = null);
    vn(a);
    a.ea && (a.ea.cancel(), a.ea = null);
    a.Ra && (l.clearTimeout(a.Ra), a.Ra = null)
}
function wn(a, b) {
    if (0 == a.g)
        throw Error("Invalid operation: sending map when state is closed");
    a.j.push(new nn(a.oe++, b));
    2 != a.g && 3 != a.g || xn(a)
}
g.Vc = function() {
    return 0 == this.g
};
g.getState = function() {
    return this.g
};
function yn(a) {
    var b = 0;
    a.W && b++;
    a.ea && b++;
    return b
}
function xn(a) {
    a.ea || a.Ra || (a.Ra = Pm(u(a.Qc, a), 0), a.xb = 0)
}
g.Qc = function(a) {
    this.Ra = null;
    zn(this, a)
};
function zn(a, b) {
    if (1 == a.g) {
        if (!b) {
            a.yb = Math.floor(1E5 * Math.random());
            var c = a.yb++, d = new zm(a, 0, "", c, void 0);
            d.Ua = null;
            var e = An(a), f = a.Kb.clone();
            Gh(f, "RID", c);
            a.M && Gh(f, "CVER", a.M);
            tn(a, f);
            Cm(d, f, e);
            a.ea = d;
            a.g = 2
        }
    } else 
        3 == a.g && (b ? Bn(a, b) : 0 != a.j.length && (a.ea || Bn(a)))
}
function Bn(a, b) {
    var c, d;
    b ? 6 < a.Wa ? (a.j = a.B.concat(a.j), a.B.length = 0, c = a.yb - 1, d = An(a)) : (c = b.D, d = b.Va) : (c = a.yb++, d = An(a));
    var e = a.Kb.clone();
    Gh(e, "SID", a.k);
    Gh(e, "RID", c);
    Gh(e, "AID", a.ib);
    tn(a, e);
    c = new zm(a, 0, a.k, c, a.xb + 1);
    c.Ua = null;
    c.setTimeout(Math.round(1E4) + Math.round(1E4 * Math.random()));
    a.ea = c;
    Cm(c, e, d)
}
function tn(a, b) {
    if (a.aa) {
        var c = a.aa.fd(a);
        c && pb(c, function(a, c) {
            Gh(b, c, a)
        })
    }
}
function An(a) {
    var b = Math.min(a.j.length, 1E3), c = ["count=" + b], d;
    6 < a.Wa && 0 < b ? (d = a.j[0].g, c.push("ofs=" + d)) : d = 0;
    for (var e = 0; e < b; e++) {
        var f = a.j[e].g, h = a.j[e].map, f = 6 >= a.Wa ? e: f - d;
        try {
            rh(h, function(a, b) {
                c.push("req" + f + "_" + b + "=" + encodeURIComponent(a))
            })
        } catch (k) {
            c.push("req" + f + "_type=" + encodeURIComponent("_badmap"))
        }
    }
    a.B = a.B.concat(a.j.splice(0, b));
    return c.join("&")
}
function Cn(a) {
    a.W || a.Fa || (a.C = 1, a.Fa = Pm(u(a.ed, a), 0), a.mb = 0)
}
function Dn(a) {
    if (a.W || a.Fa || 3 <= a.mb)
        return !1;
    a.C++;
    a.Fa = Pm(u(a.ed, a), En(a, a.mb));
    a.mb++;
    return !0
}
g.ed = function() {
    this.Fa = null;
    this.W = new zm(this, 0, this.k, "rpc", this.C);
    this.W.Ua = null;
    this.W.hc = 0;
    var a = this.Fc.clone();
    Gh(a, "RID", "rpc");
    Gh(a, "SID", this.k);
    Gh(a, "CI", this.Mb ? "0" : "1");
    Gh(a, "AID", this.ib);
    tn(this, a);
    if (!N || Jc(10))
        Gh(a, "TYPE", "xmlhttp"), Em(this.W, a, !0, this.Pb, !1);
    else {
        Gh(a, "TYPE", "html");
        var b = this.W, c = Boolean(this.Pb);
        b.ob = 3;
        b.Ka = Jh(a.clone());
        Om(b, c)
    }
};
function $m(a, b, c) {
    a.Mb = c;
    a.pa = b.Ha;
    a.Re(1, 0);
    a.Kb = Wm(a, null, a.dc);
    xn(a)
}
g.Xc = function(a, b) {
    if (0 != this.g && (this.W == a || this.ea == a))
        if (this.pa = a.ub, this.ea == a && 3 == this.g)
            if (7 < this.Wa) {
                var c;
                try {
                    c = this.A.parse(b)
                } catch (d) {
                    c = null
                }
                if (da(c) && 3 == c.length)
                    if (0 == c[0])
                        t: {
                            if (!this.Fa) {
                                if (this.W)
                                    if (this.W.Ab + 3E3 < this.ea.Ab)
                                        vn(this), this.W.cancel(), this.W = null;
                                    else 
                                        break t;
                                        Dn(this);
                                        V()
                            }
                        } else 
                            this.Lc = c[1], 0 < this.Lc - this.ib && 37500 > c[2] && this.Mb && 0 == this.mb&&!this.Za && (this.Za = Pm(u(this.Td, this), 6E3));
        else 
            Zm(this, 11)
        } else 
            "y2f%" != b && Zm(this, 11);
    else if (this.W == a && vn(this), !/^[\s\xa0]*$/.test(b)) {
        c =
        this.A.parse(b);
        da(c);
        for (var e = 0; e < c.length; e++) {
            var f = c[e];
            this.ib = f[0];
            f = f[1];
            2 == this.g ? "c" == f[0] ? (this.k = f[1], this.Pb = f[2], f = f[3], null != f ? this.Wa = f : this.Wa = 6, this.g = 3, this.aa && this.aa.Kc(this), this.Fc = Wm(this, this.zb() ? this.Pb : null, this.dc), Cn(this)) : "stop" == f[0] && Zm(this, 7) : 3 == this.g && ("stop" == f[0] ? Zm(this, 7) : "noop" != f[0] && this.aa && this.aa.Jc(this, f), this.mb = 0)
        }
    }
};
g.Td = function() {
    null != this.Za && (this.Za = null, this.W.cancel(), this.W = null, Dn(this), V())
};
function vn(a) {
    null != a.Za && (l.clearTimeout(a.Za), a.Za = null)
}
g.Nb = function(a) {
    var b;
    if (this.W == a)
        vn(this), this.W = null, b = 2;
    else if (this.ea == a)
        this.ea = null, b = 1;
    else 
        return;
    this.pa = a.ub;
    if (0 != this.g)
        if (a.ra)
            1 == b ? (w(), Yf(on, new qn(on, a.Va ? a.Va.length : 0)), xn(this), this.B.length = 0) : Cn(this);
        else {
            var c = a.Xa, d;
            if (!(d = 3 == c || 7 == c || 0 == c && 0 < this.pa)) {
                if (d = 1 == b)
                    this.ea || this.Ra || 1 == this.g || 2 <= this.xb ? d=!1 : (this.Ra = Pm(u(this.Qc, this, a), En(this, this.xb)), this.xb++, d=!0);
                    d=!(d || 2 == b && Dn(this))
                }
                if (d)
                    switch (c) {
                    case 1:
                        Zm(this, 5);
                        break;
                    case 4:
                        Zm(this, 10);
                        break;
                    case 3:
                        Zm(this,
                        6);
                        break;
                    case 7:
                        Zm(this, 12);
                        break;
                    default:
                        Zm(this, 2)
                    }
                }
    };
function En(a, b) {
    var c = 5E3 + Math.floor(1E4 * Math.random());
    a.isActive() || (c*=2);
    return c * b
}
g.Re = function(a) {
    if (!Ta(arguments, this.g))
        throw Error("Unexpected channel state: " + this.g);
    };
function Zm(a, b) {
    if (2 == b || 9 == b) {
        var c = null;
        a.aa && (c = null);
        var d = u(a.Se, a);
        c || (c = new sh("//www.google.com/images/cleardot.gif"), Jh(c));
        Tm(c.toString(), 1E4, d)
    } else 
        V();
    Fn(a, b)
}
g.Se = function(a) {
    a ? V() : (V(), Fn(this, 8))
};
function Fn(a, b) {
    a.g = 0;
    a.aa && a.aa.Uc(a, b);
    un(a);
    sn(a)
}
function un(a) {
    a.g = 0;
    a.pa =- 1;
    if (a.aa)
        if (0 == a.B.length && 0 == a.j.length)
            a.aa.kc(a);
        else {
            var b = bb(a.B), c = bb(a.j);
            a.B.length = 0;
            a.j.length = 0;
            a.aa.kc(a, b, c)
        }
}
function Wm(a, b, c) {
    var d = Kh(c);
    if ("" != d.Ia)
        b && uh(d, b + "." + d.Ia), vh(d, d.ab);
    else 
        var e = window.location, d = Lh(e.protocol, b ? b + "." + e.hostname : e.hostname, e.port, c);
    a.Hb && pb(a.Hb, function(a, b) {
        Gh(d, b, a)
    });
    Gh(d, "VER", a.Wa);
    tn(a, d);
    return d
}
g.qc = function(a) {
    if (a)
        throw Error("Can't create secondary domain capable XhrIo object.");
    a = new an;
    a.Ba=!1;
    return a
};
g.isActive = function() {
    return !!this.aa && this.aa.isActive(this)
};
function Pm(a, b) {
    if (!fa(a))
        throw Error("Fn must not be null and must be a function");
    return l.setTimeout(function() {
        a()
    }, b)
}
g.qa = function() {
    Yf(on, new rn(on))
};
function V() {
    Yf(on, new pn(on))
}
g.zb = function() {
    return !(!N || Jc(10))
};
function Gn() {}
g = Gn.prototype;
g.Kc = function() {};
g.Jc = function() {};
g.Uc = function() {};
g.kc = function() {};
g.fd = function() {
    return {}
};
g.isActive = function() {
    return !0
};
function Hn(a, b) {
    Xg.call(this);
    if (fa(a))
        b && (a = u(a, b));
    else if (a && fa(a.handleEvent))
        a = u(a.handleEvent, a);
    else 
        throw Error("Invalid listener argument");
    this.D = a;
    Kf(this, "tick", u(this.C, this));
    this.stop();
    Yg(this, 5E3 + 2E4 * Math.random())
}
y(Hn, Xg);
Hn.prototype.A = 0;
Hn.prototype.C = function() {
    if (500 < this.g) {
        var a = this.g;
        24E4 > 2 * a && (a*=2);
        Yg(this, a)
    }
    this.D()
};
Hn.prototype.start = function() {
    Hn.J.start.call(this);
    this.A = w() + this.g
};
Hn.prototype.stop = function() {
    this.A = 0;
    Hn.J.stop.call(this)
};
function In(a, b) {
    this.O = a;
    this.B = b;
    this.k = new Lb;
    this.j = new Hn(this.xd, this);
    this.g = null;
    this.H=!1;
    this.C = null;
    this.$ = "";
    this.D = this.A = 0;
    this.M = []
}
y(In, Gn);
g = In.prototype;
g.subscribe = function(a, b, c) {
    return this.k.subscribe(a, b, c)
};
g.unsubscribe = function(a, b, c) {
    return this.k.unsubscribe(a, b, c)
};
g.ya = function(a) {
    return this.k.ya(a)
};
g.publish = function(a, b) {
    return this.k.publish.apply(this.k, arguments)
};
g.dispose = function() {
    this.H || (this.H=!0, this.k.clear(), this.disconnect(), Kb(this.k))
};
g.L = function() {
    return this.H
};
function Jn(a) {
    return {
        firstTestResults: [""],
        secondTestResults: !a.g.Mb,
        sessionId: a.g.k,
        arrayId: a.g.ib
    }
}
g.connect = function(a, b, c) {
    if (!this.g || 2 != this.g.getState()) {
        this.$ = "";
        this.j.stop();
        this.C = a || null;
        this.A = b || 0;
        a = this.O + "/test";
        b = this.O + "/bind";
        var d = new mn("1", c ? c.firstTestResults : null, c ? c.secondTestResults : null), e = this.g;
        e && (e.aa = null);
        d.aa = this;
        this.g = d;
        e ? (3 != e.getState() && 0 == yn(e) || e.getState(), this.g.connect(a, b, this.B, e.k, e.ib)) : c ? this.g.connect(a, b, this.B, c.sessionId, c.arrayId) : this.g.connect(a, b, this.B)
    }
};
g.disconnect = function(a) {
    this.D = a || 0;
    this.j.stop();
    this.g && (3 == this.g.getState() && zn(this.g), this.g.disconnect());
    this.D = 0
};
g.sendMessage = function(a, b) {
    var c = {
        _sc: a
    };
    b && Cb(c, b);
    this.j.enabled || 2 == (this.g ? this.g.getState() : 0) ? this.M.push(c) : this.g && 3 == this.g.getState() && wn(this.g, c)
};
g.Kc = function() {
    var a = this.j;
    a.stop();
    Yg(a, 5E3 + 2E4 * Math.random());
    this.C = null;
    this.A = 0;
    if (this.M.length) {
        a = this.M;
        this.M = [];
        for (var b = 0, c = a.length; b < c; ++b)
            wn(this.g, a[b])
    }
    this.publish("handlerOpened")
};
g.Uc = function(a, b) {
    var c = 2 == b && 401 == this.g.pa;
    if (4 != b&&!c) {
        if (6 == b || 410 == this.g.pa)
            c = this.j, c.stop(), Yg(c, 500);
        this.j.start()
    }
    this.publish("handlerError", b)
};
g.kc = function(a, b, c) {
    if (!this.j.enabled)
        this.publish("handlerClosed");
    else if (c)
        for (a = 0, b = c.length; a < b; ++a)
            this.M.push(c[a].map)
};
g.fd = function() {
    var a = {
        v: 2
    };
    this.$ && (a.gsessionid = this.$);
    0 != this.A && (a.ui = "" + this.A);
    0 != this.D && (a.ui = "" + this.D);
    this.C && Cb(a, this.C);
    return a
};
g.Jc = function(a, b) {
    if ("S" == b[0])
        this.$ = b[1];
    else if ("gracefulReconnect" == b[0]) {
        var c = this.j;
        c.stop();
        Yg(c, 500);
        this.j.start();
        this.g.disconnect()
    } else 
        this.publish("handlerMessage", new qm(b[0], b[1]))
};
function Kn(a, b) {
    (a.B.loungeIdToken = b) || a.j.stop()
}
g.getDeviceId = function() {
    return this.B.id
};
g.xd = function() {
    this.j.stop();
    0 != yn(this.g) ? this.j.start() : this.connect(this.C, this.A)
};
function Ln() {
    Ib.call(this);
    this.B = new Lb;
    Jb(this, v(Kb, this.B))
}
y(Ln, Ib);
Ln.prototype.subscribe = function(a, b, c) {
    return this.L() ? 0 : this.B.subscribe(a, b, c)
};
Ln.prototype.unsubscribe = function(a, b, c) {
    return this.L()?!1 : this.B.unsubscribe(a, b, c)
};
Ln.prototype.ya = function(a) {
    return this.L()?!1 : this.B.ya(a)
};
Ln.prototype.publish = function(a, b) {
    return this.L()?!1 : this.B.publish.apply(this.B, arguments)
};
function Mn() {
    this.g = [];
    this.j = []
}
g = Mn.prototype;
g.Ga = function() {
    return this.g.length + this.j.length
};
g.isEmpty = function() {
    return Ua(this.g) && Ua(this.j)
};
g.clear = function() {
    this.g = [];
    this.j = []
};
g.contains = function(a) {
    return Ta(this.g, a) || Ta(this.j, a)
};
g.remove = function(a) {
    var b = Ma(this.g, a);
    if (0 > b)
        return Ya(this.j, a);
    Za(this.g, b);
    return !0
};
g.ta = function() {
    for (var a = [], b = this.g.length - 1; 0 <= b; --b)
        a.push(this.g[b]);
    for (var c = this.j.length, b = 0; b < c; ++b)
        a.push(this.j[b]);
    return a
};
var Nn = w(), On = null, Pn = Array(50), Qn =- 1, Rn=!1;
function Sn(a) {
    Tn();
    On.push(a);
    Un(On)
}
function Vn(a, b) {
    Tn();
    var c = On, d = Wn(a, String(b));
    Ua(c) ? Xn(d) : (Un(c), z(c, function(a) {
        a(d)
    }))
}
function Tn() {
    On || (On = r("yt.mdx.remote.debug.handlers_") || [], q("yt.mdx.remote.debug.handlers_", On, void 0))
}
function Xn(a) {
    var b = (Qn + 1)%50;
    Qn = b;
    Pn[b] = a;
    Rn || (Rn = 49 == b)
}
function Un(a) {
    var b = Pn;
    if (b[0]) {
        var c = Qn, d = Rn ? c: - 1;
        do {
            var d = (d + 1)%50, e = b[d];
            z(a, function(a) {
                a(e)
            })
        }
        while (d != c);
        Pn = Array(50);
        Qn =- 1;
        Rn=!1
    }
}
function Wn(a, b) {
    var c = (w() - Nn) / 1E3;
    c.toFixed && (c = c.toFixed(3));
    var d = [];
    d.push("[", c + "s", "] ");
    d.push("[", "yt.mdx.remote", "] ");
    d.push(a + ": " + b, "\n");
    return d.join("")
};
function Yn(a) {
    a = a || {};
    this.name = a.name || "";
    this.id = a.id || a.screenId || "";
    this.token = a.token || a.loungeToken || "";
    this.uuid = a.uuid || a.dialId || ""
}
function Zn(a, b) {
    return !!b && (a.id == b || a.uuid == b)
}
function $n(a, b) {
    return a || b?!a!=!b?!1 : a.id == b.id && a.token == b.token && a.name == b.name && a.uuid == b.uuid : !0
}
function ao(a) {
    return {
        name: a.name,
        screenId: a.id,
        loungeToken: a.token,
        dialId: a.uuid
    }
}
function bo(a) {
    return new Yn(a)
}
function co(a) {
    return da(a) ? A(a, bo) : []
}
function eo(a) {
    return a ? '{name:"' + a.name + '",id:' + a.id.substr(0, 6) + "..,token:" + (a.token ? ".." + a.token.slice( - 6) : "-") + ",uuid:" + (a.uuid ? ".." + a.uuid.slice( - 6) : "-") + "}" : "null"
}
function fo(a) {
    return da(a) ? "[" + A(a, eo).join(",") + "]" : "null"
};
function go() {};
function ho() {}
y(ho, go);
ho.prototype.Ga = function() {
    var a = 0;
    te(this.fb(!0), function() {
        a++
    });
    return a
};
ho.prototype.clear = function() {
    var a = ue(this.fb(!0)), b = this;
    z(a, function(a) {
        b.remove(a)
    })
};
function io(a) {
    this.g = a
}
y(io, ho);
g = io.prototype;
g.isAvailable = function() {
    if (!this.g)
        return !1;
    try {
        return this.g.setItem("__sak", "1"), this.g.removeItem("__sak"), !0
    } catch (a) {
        return !1
    }
};
g.set = function(a, b) {
    try {
        this.g.setItem(a, b)
    } catch (c) {
        if (0 == this.g.length)
            throw "Storage mechanism: Storage disabled";
        throw "Storage mechanism: Quota exceeded";
    }
};
g.get = function(a) {
    a = this.g.getItem(a);
    if (!t(a) && null !== a)
        throw "Storage mechanism: Invalid value was encountered";
    return a
};
g.remove = function(a) {
    this.g.removeItem(a)
};
g.Ga = function() {
    return this.g.length
};
g.fb = function(a) {
    var b = 0, c = this.g, d = new re;
    d.next = function() {
        if (b >= c.length)
            throw qe;
        var d;
        d = c.key(b++);
        if (a)
            return d;
        d = c.getItem(d);
        if (!t(d))
            throw "Storage mechanism: Invalid value was encountered";
        return d
    };
    return d
};
g.clear = function() {
    this.g.clear()
};
g.key = function(a) {
    return this.g.key(a)
};
function jo() {
    var a = null;
    try {
        a = window.localStorage || null
    } catch (b) {}
    this.g = a
}
y(jo, io);
function ko() {
    var a = null;
    try {
        a = window.sessionStorage || null
    } catch (b) {}
    this.g = a
}
y(ko, io);
function lo(a) {
    this.g = a
}
lo.prototype.set = function(a, b) {
    n(b) ? this.g.set(a, S(b)) : this.g.remove(a)
};
lo.prototype.get = function(a) {
    var b;
    try {
        b = this.g.get(a)
    } catch (c) {
        return 
    }
    if (null !== b)
        try {
            return De(b)
    } catch (d) {
        throw "Storage: Invalid value was encountered";
    }
};
lo.prototype.remove = function(a) {
    this.g.remove(a)
};
function mo(a) {
    this.g = a
}
y(mo, lo);
function no(a) {
    this.data = a
}
function oo(a) {
    return !n(a) || a instanceof no ? a : new no(a)
}
mo.prototype.set = function(a, b) {
    mo.J.set.call(this, a, oo(b))
};
mo.prototype.j = function(a) {
    a = mo.J.get.call(this, a);
    if (!n(a) || a instanceof Object)
        return a;
    throw "Storage: Invalid value was encountered";
};
mo.prototype.get = function(a) {
    if (a = this.j(a)) {
        if (a = a.data, !n(a))
            throw "Storage: Invalid value was encountered";
    } else 
        a = void 0;
    return a
};
function po(a) {
    this.g = a
}
y(po, mo);
function qo(a) {
    var b = a.creation;
    a = a.expiration;
    return !!a && a < w()||!!b && b > w()
}
po.prototype.set = function(a, b, c) {
    if (b = oo(b)) {
        if (c) {
            if (c < w()) {
                po.prototype.remove.call(this, a);
                return 
            }
            b.expiration = c
        }
        b.creation = w()
    }
    po.J.set.call(this, a, b)
};
po.prototype.j = function(a, b) {
    var c = po.J.j.call(this, a);
    if (c)
        if (!b && qo(c))
            po.prototype.remove.call(this, a);
        else 
            return c
};
function ro(a) {
    this.g = a
}
y(ro, po);
function so(a, b) {
    var c = [];
    te(b, function(a) {
        var b;
        try {
            b = ro.prototype.j.call(this, a, !0)
        } catch (f) {
            if ("Storage: Invalid value was encountered" == f)
                return;
            throw f;
        }
        n(b) ? qo(b) && c.push(a) : c.push(a)
    }, a);
    return c
}
function to(a, b) {
    var c = so(a, b);
    z(c, function(a) {
        ro.prototype.remove.call(this, a)
    }, a)
}
function uo() {
    var a = vo;
    to(a, a.g.fb(!0))
};
function wo(a, b, c) {
    var d = c && 0 < c ? c: 0;
    c = d ? w() + 1E3 * d : 0;
    if ((d = d ? vo : xo) && window.JSON) {
        t(b) || (b = JSON.stringify(b, void 0));
        try {
            d.set(a, b, c)
        } catch (e) {
            d.remove(a)
        }
    }
}
function yo(a) {
    if (!xo&&!vo ||!window.JSON)
        return null;
    var b;
    try {
        b = xo.get(a)
    } catch (c) {}
    if (!t(b))
        try {
            b = vo.get(a)
    } catch (d) {}
    if (!t(b))
        return null;
    try {
        b = JSON.parse(b, void 0)
    } catch (e) {}
    return b
}
function zo(a) {
    xo && xo.remove(a);
    vo && vo.remove(a)
}
var vo, Ao = new jo;
vo = Ao.isAvailable() ? new ro(Ao) : null;
var xo, Bo = new ko;
xo = Bo.isAvailable() ? new ro(Bo) : null;
var Co = ["boadgeojelhgndaghljhdicfkmllpafd", "dliochdbjfkdbacpmhlcpmleaejidimm", "hfaagokkkhdbgiakmmlclaapfelnkoah", "fmfcbgogabcbclcofgocippekhfcmgfj", "enhhojjnijigcajfphajepfemndkmdlo"];
function Do(a) {
    var b = yo("yt-remote-cast-last-extension");
    b ? "none" == b ? a(null) : a(b) : Eo(0, a)
}
function Eo(a, b) {
    a == Co.length ? (wo("yt-remote-cast-last-extension", "none"), b(null)) : Fo(Co[a], function(c) {
        c ? (c = Co[a], wo("yt-remote-cast-last-extension", c), b(c)) : Eo(a + 1, b)
    })
}
function Go(a) {
    return "chrome-extension://" + a + "/cast_sender.js"
}
function Fo(a, b) {
    var c = new XMLHttpRequest;
    c.onreadystatechange = function() {
        4 == c.readyState && 200 == c.status && b(!0)
    };
    c.onerror = function() {
        b(!1)
    };
    try {
        c.open("GET", Go(a), !0), c.send()
    } catch (d) {
        b(!1)
    }
}
function Ho(a) {
    window.__onGCastApiAvailable = a;
    Do(function(b) {
        if (b) {
            Vn("bootstrap", "Found cast extension: " + b);
            q("chrome.cast.extensionId", b, void 0);
            var c = document.createElement("script");
            c.src = Go(b);
            c.onerror = function() {
                Io();
                zo("yt-remote-cast-last-extension");
                a(!1, "Extension JS failed to load.")
            };
            (document.head || document.documentElement).appendChild(c)
        } else 
            Vn("bootstrap", "No cast extension found"), a(!1, "No cast extension found")
    })
}
function Io() {
    window.__onGCastApiAvailable && delete window.__onGCastApiAvailable
};
function Jo(a) {
    this.port = this.k = "";
    this.g = "/api/lounge";
    this.j=!0;
    a = a || document.location.href;
    var b = Number(Me(a)[4] || null) || null || "";
    b && (this.port = ":" + b);
    this.k = Oe(a) || "";
    a = vc;
    0 <= a.search("MSIE") && (a = a.match(/MSIE ([\d.]+)/)[1], 0 > Ea(a, "10.0") && (this.j=!1))
}
function Ko(a, b, c, d) {
    var e = a.g;
    if (n(d) ? d : a.j)
        e = "https://" + a.k + a.port + a.g;
    return Ve(e + b, c || {})
}
Jo.prototype.sendRequest = function(a, b, c, d, e, f, h) {
    a = {
        format: f ? "RAW": "JSON",
        method: a,
        context: this,
        timeout: 5E3,
        withCredentials: !!h,
        V: v(this.A, d, !f),
        onError: v(this.B, e),
        wb: v(this.C, e)
    };
    c && (a.ca = c, a.headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    });
    return ff(b, a)
};
Jo.prototype.A = function(a, b, c, d) {
    b ? a(d) : a({
        text: c.responseText
    })
};
Jo.prototype.B = function(a, b) {
    a(Error("Request error: " + b.status))
};
Jo.prototype.C = function(a) {
    a(Error("request timed out"))
};
function Lo(a) {
    a && (this.id = a.id || "", this.name = a.name || "", this.activityId = a.activityId || "", this.status = a.status || "UNKNOWN")
}
Lo.prototype.id = "";
Lo.prototype.name = "";
Lo.prototype.activityId = "";
Lo.prototype.status = "UNKNOWN";
function Mo(a) {
    return {
        id: a.id,
        name: a.name,
        activityId: a.activityId,
        status: a.status
    }
}
Lo.prototype.toString = function() {
    return "{id:" + this.id + ",name:" + this.name + ",activityId:" + this.activityId + ",status:" + this.status + "}"
};
function No(a) {
    a = a || [];
    return "[" + A(a, function(a) {
        return a ? a.toString() : "null"
    }).join(",") + "]"
};
function Oo() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(a) {
        var b = 16 * Math.random() | 0;
        return ("x" == a ? b : b & 3 | 8).toString(16)
    })
}
function Po(a) {
    return A(a, function(a) {
        return Mo(a)
    })
}
function Qo() {
    return A(Wa, function(a) {
        return new Lo(a)
    })
}
function Ro(a, b) {
    return a || b ? a && b ? a.id == b.id && a.name == b.name : !1 : !0
}
function So(a, b) {
    return Ra(a, function(a) {
        return a.id == b
    })
}
function To(a, b) {
    return Ra(a, function(a) {
        return a || b?!a!=!b?!1 : a.id == b.id : !0
    })
}
function Uo(a, b) {
    return Ra(a, function(a) {
        return Zn(a, b)
    })
};
function Vo(a) {
    Ln.call(this);
    this.H = a;
    this.screens = []
}
y(Vo, Ln);
g = Vo.prototype;
g.da = function() {
    return this.screens
};
g.contains = function(a) {
    return !!To(this.screens, a)
};
g.get = function(a) {
    return a ? Uo(this.screens, a) : null
};
function Wo(a, b) {
    var c = a.get(b.uuid) || a.get(b.id);
    if (c) {
        var d = c.name;
        c.id = b.id || c.id;
        c.name = b.name;
        c.token = b.token;
        c.uuid = b.uuid || c.uuid;
        return c.name != d
    }
    a.screens.push(b);
    return !0
}
function Xo(a, b) {
    var c = a.screens.length != b.length;
    a.screens = Na(a.screens, function(a) {
        return !!To(b, a)
    });
    for (var d = 0, e = b.length; d < e; d++)
        c = Wo(a, b[d]) || c;
    return c
}
function Yo(a, b) {
    var c = a.screens.length;
    a.screens = Na(a.screens, function(a) {
        return !(a || b?!a!=!b ? 0 : a.id == b.id : 1)
    });
    return a.screens.length < c
}
g.info = function(a) {
    Vn(this.H, a)
};
g.warn = function(a) {
    Vn(this.H, a)
};
var Zo;
function $o() {
    var a = ap(), b = bp();
    Ta(a, b);
    if (cp()) {
        var c = a, d;
        d = 0;
        for (var e = c.length, f; d < e;) {
            var h = d + e>>1, k;
            k = hb(b, c[h]);
            0 < k ? d = h + 1 : (e = h, f=!k)
        }
        d = f ? d : ~d;
        0 > d && db(c, - (d + 1), 0, b)
    }
    a = dp(a);
    if (Ua(a))
        try {
            hh("remote_sid")
    } catch (m) {} else 
        try {
            fh("remote_sid", a.join(","))
    } catch (p) {}
}
function ap() {
    var a = yo("yt-remote-connected-devices") || [];
    a.sort(hb);
    return a
}
function dp(a) {
    if (Ua(a))
        return [];
    var b = a[0].indexOf("#"), c =- 1 == b ? a[0] : a[0].substring(0, b);
    return A(a, function(a, b) {
        return 0 == b ? a : a.substring(c.length)
    })
}
function ep(a) {
    wo("yt-remote-connected-devices", a, 86400)
}
function bp() {
    if (fp)
        return fp;
    var a = yo("yt-remote-device-id");
    a || (a = Oo(), wo("yt-remote-device-id", a, 31536E3));
    for (var b = ap(), c = 1, d = a; Ta(b, d);)
        c++, d = a + "#" + c;
    return fp = d
}
function gp() {
    return yo("yt-remote-session-browser-channel")
}
function cp() {
    return yo("yt-remote-session-screen-id")
}
function hp(a) {
    5 < a.length && (a = a.slice(a.length - 5));
    var b = A(ip(), function(a) {
        return a.loungeToken
    }), c = A(a, function(a) {
        return a.loungeToken
    });
    Qa(c, function(a) {
        return !Ta(b, a)
    }) && jp();
    wo("yt-remote-local-screens", a, 31536E3)
}
function ip() {
    return yo("yt-remote-local-screens") || []
}
function kp(a) {
    wo("yt-remote-load-account-screens", a, 31536E3)
}
function jp() {
    wo("yt-remote-lounge-token-expiration", !0, 86400)
}
function lp() {
    return !yo("yt-remote-lounge-token-expiration")
}
function mp(a) {
    wo("yt-remote-online-screens", a, 60)
}
function np() {
    return yo("yt-remote-online-screens") || []
}
function op(a) {
    wo("yt-remote-online-dial-devices", a, 30)
}
function pp(a, b) {
    wo("yt-remote-session-browser-channel", a);
    wo("yt-remote-session-screen-id", b);
    var c = ap(), d = bp();
    Ta(c, d) || c.push(d);
    ep(c);
    $o()
}
function qp(a) {
    a || (zo("yt-remote-session-screen-id"), zo("yt-remote-session-video-id"));
    $o();
    a = ap();
    Ya(a, bp());
    ep(a)
}
function rp() {
    if (!Zo) {
        var a;
        a = new jo;
        (a = a.isAvailable() ? a : null) && (Zo = new lo(a))
    }
    return Zo?!!Zo.get("yt-remote-use-staging-server") : !1
}
var fp = "";
function sp(a) {
    Vo.call(this, "AccountScreenService");
    this.g = a;
    this.j = NaN;
    this.screens = tp();
    this.info("Initializing with " + fo(this.screens))
}
y(sp, Vo);
g = sp.prototype;
g.start = function() {
    this.Zc()
};
g.add = function(a, b, c) {
    if (this.contains(a))
        this.Pa(a, a.name, b, c);
    else {
        var d = up(this, a.id);
        this.g.sendRequest("POST", d, {
            screen_name: a.name
        }, u(this.ze, this, a, b, c), u(this.Tb, this, "add", c), !0, !0)
    }
};
g.remove = function(a, b, c) {
    if (this.contains(a)) {
        var d = up(this, a.id);
        this.g.sendRequest("DELETE", d, null, u(this.Be, this, a, b, c), u(this.Tb, this, "remove", c), !0, !0)
    } else 
        a = "Trying to remove non-account screen: " + a.name, this.warn(a), c(Error(a))
};
g.Pa = function(a, b, c, d) {
    if (this.contains(a)) {
        var e = up(this, a.id);
        this.g.sendRequest("PUT", e, {
            screen_name: b
        }, u(this.Ce, this, a, b, c, d), u(this.Tb, this, "update", d), !0, !0)
    } else 
        a = "Trying to update non-account screen: " + a.name, this.warn(a), d(Error(a))
};
g.G = function() {
    H(this.j);
    sp.J.G.call(this)
};
function vp(a, b) {
    return a.length != b.length?!1 : fb(a, b, function(a, b) {
        return a || b?!a!=!b?!1 : a.id == b.id && a.name == b.name : !0
    })
}
function tp() {
    var a = co(ip()), b = co(np());
    return Na(b, function(b) {
        return !Uo(a, b.id)
    })
}
function up(a, b) {
    var c = Ko(a.g, "/screens");
    return b ? c + "/" + b : c
}
g.Zc = function() {
    if (!this.L()) {
        H(this.j);
        var a = up(this);
        this.g.sendRequest("GET", a, null, u(this.fe, this), u(this.Tb, this, "load", s), !1, !0)
    }
};
g.fe = function(a) {
    if (!this.L())
        if (a = co(sb(a)), vp(this.screens, a))
            for (var b = 0, c = this.screens.length; b < c; ++b) {
                var d = this.screens[b], e = Uo(a, d.id);
                e && e.token && (d.token = e.token)
            } else 
                this.info("Updated account screens: " + fo(a)), this.screens = a, this.publish("screenChange"), a = this.da().length ? 12E4 : 6E4, this.j = G(u(this.Zc, this), a)
};
g.ze = function(a, b) {
    this.L() || (Wo(this, a), this.publish("screenChange"), b(a))
};
g.Be = function(a, b) {
    this.L() || (Yo(this, a), this.publish("screenChange"), b(a))
};
g.Ce = function(a, b, c) {
    this.L() || (a = this.get(a.id), a.name = b, c(a), this.publish("screenChange"))
};
g.Tb = function(a, b, c) {
    this.L() || (a = "Failed to " + a + " account screen: " + c, this.warn(a), b(Error(a)))
};
function wp(a, b, c, d) {
    Ln.call(this);
    this.H = a;
    this.D = b;
    this.A = c;
    this.C = d;
    this.k = 0;
    this.g = null;
    this.j = NaN
}
y(wp, Ln);
var xp = [2E3, 2E3, 1E3, 1E3, 1E3, 2E3, 2E3, 5E3, 5E3, 1E4];
g = wp.prototype;
g.start = function() {
    !this.g && isNaN(this.j) && this.Ic()
};
g.stop = function() {
    this.g && (this.g.abort(), this.g = null);
    isNaN(this.j) || (H(this.j), this.j = NaN)
};
g.G = function() {
    this.stop();
    wp.J.G.call(this)
};
g.Ic = function() {
    this.j = NaN;
    this.g = ff(Ko(this.H, "/pairing/get_screen"), {
        method: "POST",
        ca: {
            pairing_code: this.D
        },
        timeout: 5E3,
        V: u(this.Rd, this),
        onError: u(this.Qd, this),
        wb: u(this.Sd, this)
    })
};
g.Rd = function(a, b) {
    this.g = null;
    var c = b.screen || {};
    c.dialId = this.A;
    c.name = this.C;
    this.publish("pairingComplete", new Yn(c))
};
g.Qd = function(a) {
    this.g = null;
    a.status && 404 == a.status ? this.k >= xp.length ? this.publish("pairingFailed", Error("DIAL polling timed out")) : (a = xp[this.k], this.j = G(u(this.Ic, this), a), this.k++) : this.publish("pairingFailed", Error("Server error " + a.status))
};
g.Sd = function() {
    this.g = null;
    this.publish("pairingFailed", Error("Server not responding"))
};
function yp(a) {
    Vo.call(this, "LocalScreenService");
    this.j = a;
    this.g = NaN;
    zp(this);
    this.info("Initializing with " + fo(this.screens))
}
y(yp, Vo);
g = yp.prototype;
g.start = function() {
    zp(this) && this.publish("screenChange");
    lp() && Ap(this);
    H(this.g);
    this.g = G(u(this.start, this), 1E4)
};
g.add = function(a, b) {
    zp(this);
    Wo(this, a);
    Bp(this, !1);
    this.publish("screenChange");
    b(a);
    a.token || Ap(this)
};
g.remove = function(a, b) {
    var c = zp(this);
    Yo(this, a) && (Bp(this, !1), c=!0);
    b(a);
    c && this.publish("screenChange")
};
g.Pa = function(a, b, c, d) {
    var e = zp(this), f = this.get(a.id);
    f ? (f.name != b && (f.name = b, Bp(this, !1), e=!0), c(a)) : d(Error("no such local screen."));
    e && this.publish("screenChange")
};
g.G = function() {
    H(this.g);
    yp.J.G.call(this)
};
function Ap(a) {
    if (a.screens.length) {
        var b = A(a.screens, function(a) {
            return a.id
        }), c = Ko(a.j, "/pairing/get_lounge_token_batch");
        a.j.sendRequest("POST", c, {
            screen_ids: b.join(",")
        }, u(a.Je, a), u(a.Ie, a))
    }
}
g.Je = function(a) {
    zp(this);
    var b = this.screens.length;
    a = a && a.screens || [];
    for (var c = 0, d = a.length; c < d; ++c) {
        var e = a[c], f = this.get(e.screenId);
        f && (f.token = e.loungeToken, --b)
    }
    Bp(this, !b);
    b && this.warn("Missed " + b + " lounge tokens.")
};
g.Ie = function(a) {
    this.warn("Requesting lounge tokens failed: " + a)
};
function zp(a) {
    var b = co(ip()), b = Na(b, function(a) {
        return !a.uuid
    });
    return Xo(a, b)
}
function Bp(a, b) {
    hp(A(a.screens, ao));
    b && jp()
};
function Cp(a, b) {
    Ln.call(this);
    this.C = b;
    this.g = Dp(this);
    this.D = a;
    this.k = this.A = NaN;
    this.j = null;
    this.H = u(this.Gd, this);
    Ep("Initialized with " + S(this.g))
}
y(Cp, Ln);
g = Cp.prototype;
g.start = function() {
    var a = parseInt(yo("yt-remote-fast-check-period") || "0", 10);
    (this.A = w() - 144E5 < a ? 0 : a) ? Fp(this, !1) : (this.A = w() + 3E5, wo("yt-remote-fast-check-period", this.A), this.wc());
    R(window, "storage", this.H)
};
g.isEmpty = function() {
    return wb(this.g)
};
g.update = function() {
    Ep("Updating availability on schedule.");
    var a = this.C(), b = qb(this.g, function(b, d) {
        return b&&!!Uo(a, d)
    }, this);
    Gp(this, b)
};
function Hp(a, b, c) {
    var d = Ko(a.D, "/pairing/get_screen_availability");
    a.D.sendRequest("POST", d, {
        lounge_token: b.token
    }, u(function(a) {
        a = a.screens || [];
        for (var d = 0, h = a.length; d < h; ++d)
            if (a[d].loungeToken == b.token) {
                c("online" == a[d].status);
                return 
            }
        c(!1)
    }, a), u(function() {
        c(!1)
    }, a))
}
g.G = function() {
    H(this.k);
    this.k = NaN;
    this.j && (this.j.abort(), this.j = null);
    var a = je(window, "storage", this.H, !1);
    a && le(a);
    Cp.J.G.call(this)
};
function Gp(a, b) {
    var c;
    t: if (rb(b) != rb(a.g))
        c=!1;
    else {
        c = tb(b);
        for (var d = 0, e = c.length; d < e; ++d)
            if (!a.g[c[d]]) {
                c=!1;
                break t
            }
        c=!0
    }
    c || (Ep("Updated online screens: " + S(a.g)), a.g = b, a.publish("screenChange"));
    Ip(a)
}
function Fp(a, b) {
    isNaN(a.k) || H(a.k);
    var c = u(a.wc, a), d;
    d = 0;
    b && (d = 2E3 + 8E3 * Math.random());
    d = 0 < a.A && a.A < w() ? 2E4 + d : 1E4 + d;
    a.k = G(c, d)
}
g.wc = function() {
    H(this.k);
    this.k = NaN;
    this.j && this.j.abort();
    var a = Jp(this);
    if (rb(a)) {
        var b = Ko(this.D, "/pairing/get_screen_availability"), c = {
            lounge_token: tb(a).join(",")
        };
        this.j = this.D.sendRequest("POST", b, c, u(this.Ee, this, a), u(this.De, this))
    } else 
        Gp(this, {}), Fp(this, !1)
};
g.Ee = function(a, b) {
    this.j = null;
    var c = tb(Jp(this));
    if (fb(c, tb(a))) {
        for (var c = b.screens || [], d = {}, e = 0, f = c.length; e < f; ++e)
            d[a[c[e].loungeToken]] = "online" == c[e].status;
        Gp(this, d);
        Fp(this, !1)
    } else 
        this.R("Changing Screen set during request."), this.wc()
};
g.De = function(a) {
    this.R("Screen availability failed: " + a);
    this.j = null;
    Fp(this, !1)
};
g.Gd = function(a) {
    "yt-remote-online-screen-ids" == a.key && (this.g = Dp(this), Fp(this, !0))
};
function Ep(a) {
    Vn("OnlineScreenService", a)
}
g.R = function(a) {
    Vn("OnlineScreenService", a)
};
function Jp(a) {
    var b = {};
    z(a.C(), function(a) {
        a.token ? b[a.token] = a.id : this.R("Requesting availability of screen w/o lounge token.")
    });
    return b
}
function Dp(a) {
    var b = yo("yt-remote-online-screen-ids") || "", b = b ? b.split(","): [], c = {};
    a = a.C();
    for (var d = 0, e = a.length; d < e; ++d) {
        var f = a[d].id;
        c[f] = Ta(b, f)
    }
    return c
}
function Ip(a) {
    var b = tb(qb(a.g, function(a) {
        return a
    }));
    b.sort(hb);
    b.length ? wo("yt-remote-online-screen-ids", b.join(","), 60) : zo("yt-remote-online-screen-ids");
    a = Na(a.C(), function(a) {
        return !!this.g[a.id]
    }, a);
    mp(A(a, ao))
};
function X(a, b) {
    Vo.call(this, "ScreenService");
    this.A = a;
    this.k = this.g = this.j = null;
    this.C = [];
    this.D = {};
    Kp(this, b)
}
y(X, Vo);
g = X.prototype;
g.start = function(a) {
    a ? (this.g || (this.g = new sp(this.A), this.g.subscribe("screenChange", u(this.Bb, this))), this.g.start()) : this.g && (Kb(this.g), this.g = null, Lp(this));
    this.j.start();
    this.k.start();
    this.screens.length && (this.publish("screenChange"), this.k.isEmpty() || this.publish("onlineScreenChange"))
};
g.add = function(a, b, c) {
    this.g ? this.g.add(a, u(function(a) {
        this.Bb();
        this.j.add(a, s, s);
        b(a)
    }, this), c) : this.j.add(a, b, c)
};
g.remove = function(a, b, c) {
    this.g && this.g.contains(a) ? (this.g.remove(a, u(function(a) {
        this.Bb();
        b(a)
    }, this), c), this.j.remove(a, s, s)) : this.j.remove(a, b, c);
    this.k.update()
};
g.Pa = function(a, b, c, d) {
    this.g && this.g.contains(a) ? (this.g.Pa(a, b, c, d), this.j.Pa(a, b, s, s)) : this.j.contains(a) ? this.j.Pa(a, b, c, d) : (a = "Updating name of unknown screen: " + a.name, this.warn(a), d(Error(a)))
};
g.da = function(a) {
    return a ? this.screens : ab(this.screens, Na(this.C, function(a) {
        return !this.contains(a)
    }, this))
};
g.Oc = function() {
    return Na(this.da(!0), function(a) {
        return !!this.k.g[a.id]
    }, this)
};
function Mp(a, b, c, d, e, f) {
    a.info("getAutomaticScreenByIds " + c + " / " + b);
    c || (c = a.D[b]);
    var h = a.da();
    if (h = (c ? Uo(h, c) : null) || Uo(h, b)) {
        h.uuid = b;
        var k = Np(a, h);
        Hp(a.k, k, function(a) {
            e(a ? k : null)
        })
    } else 
        c ? Op(a, c, u(function(a) {
            var f = Np(this, new Yn({
                name: d,
                screenId: c,
                loungeToken: a,
                dialId: b || ""
            }));
            Hp(this.k, f, function(a) {
                e(a ? f : null)
            })
        }, a), f) : e(null)
}
function Pp(a, b, c, d, e, f) {
    a.info("getDialScreenByPairingCode " + b + " / " + c);
    var h = new wp(a.A, b, c, d);
    h.subscribe("pairingComplete", u(function(a) {
        Kb(h);
        e(Np(this, a))
    }, a));
    h.subscribe("pairingFailed", function(a) {
        Kb(h);
        f(a)
    });
    h.start();
    return u(h.stop, h)
}
function Qp(a, b) {
    for (var c = 0, d = a.screens.length; c < d; ++c)
        if (a.screens[c].name == b)
            return a.screens[c];
    return null
}
g.cd = function(a, b) {
    for (var c = 2, d = b(a, c); Qp(this, d);) {
        c++;
        if (20 < c)
            return a;
        d = b(a, c)
    }
    return d
};
g.af = function(a, b, c, d) {
    ff(Ko(this.A, "/pairing/get_screen"), {
        method: "POST",
        ca: {
            pairing_code: a
        },
        timeout: 5E3,
        V: u(function(a, d) {
            var h = new Yn(d.screen || {});
            if (!h.name || Qp(this, h.name))
                h.name = this.cd(h.name, b);
            c(Np(this, h))
        }, this),
        onError: u(function(a) {
            d(Error("pairing request failed: " + a.status))
        }, this),
        wb: u(function() {
            d(Error("pairing request timed out."))
        }, this)
    })
};
g.G = function() {
    Kb(this.g);
    Kb(this.j);
    Kb(this.k);
    X.J.G.call(this)
};
function Op(a, b, c, d) {
    a.info("requestLoungeToken_ for " + b);
    var e = {
        ca: {
            screen_ids: b
        },
        method: "POST",
        context: a,
        V: function(a, e) {
            var k = e && e.screens || [];
            k[0] && k[0].screenId == b ? c(k[0].loungeToken) : d(Error("Missing lounge token in token response"))
        },
        onError: function() {
            d(Error("Request screen lounge token failed"))
        }
    };
    ff(Ko(a.A, "/pairing/get_lounge_token_batch"), e)
}
function Lp(a) {
    a.g ? a.screens = ab(a.g.da(), Na(a.j.da(), function(a) {
        return !this.g.contains(a)
    }, a)) : a.screens = a.j.da();
    for (var b = Ab(a.D), c = 0, d = a.screens.length; c < d; ++c) {
        var e = a.screens[c];
        e.uuid = b[e.id] || ""
    }
    a.info("Updated manual screens: " + fo(a.screens))
}
g.Bb = function() {
    Lp(this);
    this.publish("screenChange");
    this.k.update()
};
function Kp(a, b) {
    Rp(a);
    a.j = new yp(a.A);
    a.j.subscribe("screenChange", u(a.Bb, a));
    b && (a.g = new sp(a.A), a.g.subscribe("screenChange", u(a.Bb, a)));
    Lp(a);
    a.C = co(yo("yt-remote-automatic-screen-cache") || []);
    Rp(a);
    a.info("Initializing automatic screens: " + fo(a.C));
    a.k = new Cp(a.A, u(a.da, a, !0));
    a.k.subscribe("screenChange", u(function() {
        this.publish("onlineScreenChange")
    }, a))
}
function Np(a, b) {
    var c = a.get(b.id);
    c ? (c.uuid = b.uuid, b = c) : ((c = Uo(a.C, b.uuid)) ? (c.id = b.id, c.token = b.token, b = c) : a.C.push(b), wo("yt-remote-automatic-screen-cache", A(a.C, ao)));
    Rp(a);
    a.D[b.uuid] = b.id;
    wo("yt-remote-device-id-map", a.D, 31536E3);
    return b
}
function Rp(a) {
    a.D = yo("yt-remote-device-id-map") || {}
}
X.prototype.dispose = X.prototype.dispose;
function Sp(a, b, c) {
    Ln.call(this);
    this.Z = c;
    this.P = a;
    this.j = b;
    this.k = null
}
y(Sp, Ln);
function Tp(a, b) {
    a.k = b;
    a.publish("sessionScreen", a.k)
}
g = Sp.prototype;
g.fa = function(a) {
    this.L() || (a && this.warn("" + a), this.k = null, this.publish("sessionScreen", null))
};
g.info = function(a) {
    Vn(this.Z, a)
};
g.warn = function(a) {
    Vn(this.Z, a)
};
g.Yc = function() {
    return null
};
g.gc = function(a) {
    var b = this.j;
    a ? (b.displayStatus = new chrome.cast.ReceiverDisplayStatus(a, []), b.displayStatus.showStop=!0) : b.displayStatus = null;
    chrome.cast.setReceiverDisplayStatus(b, u(function() {
        this.info("Updated receiver status for " + b.friendlyName + ": " + a)
    }, this), u(function() {
        this.warn("Failed to update receiver status for: " + b.friendlyName)
    }, this))
};
g.G = function() {
    this.gc("");
    Sp.J.G.call(this)
};
function Up(a, b) {
    Sp.call(this, a, b, "CastSession");
    this.g = null;
    this.C = 0;
    this.A = null;
    this.H = u(this.he, this);
    this.D = u(this.ge, this);
    this.C = G(u(function() {
        Vp(this, null)
    }, this), 12E4)
}
y(Up, Sp);
g = Up.prototype;
g.mc = function(a) {
    if (this.g) {
        if (this.g == a)
            return;
        this.warn("Overriding cast sesison with new session object");
        this.g.removeUpdateListener(this.H);
        this.g.removeMessageListener("urn:x-cast:com.google.youtube.mdx", this.D)
    }
    this.g = a;
    this.g.addUpdateListener(this.H);
    this.g.addMessageListener("urn:x-cast:com.google.youtube.mdx", this.D);
    this.A && Wp(this);
    Xp(this, "getMdxSessionStatus")
};
g.lb = function(a) {
    this.info("launchWithParams: " + S(a));
    this.A = a;
    this.g && Wp(this)
};
g.stop = function() {
    this.g ? this.g.stop(u(function() {
        this.fa()
    }, this), u(function() {
        this.fa(Error("Failed to stop receiver app."))
    }, this)) : this.fa(Error("Stopping cast device witout session."))
};
g.gc = s;
g.G = function() {
    this.info("disposeInternal");
    H(this.C);
    this.C = 0;
    this.g && (this.g.removeUpdateListener(this.H), this.g.removeMessageListener("urn:x-cast:com.google.youtube.mdx", this.D));
    this.g = null;
    Up.J.G.call(this)
};
function Wp(a) {
    var b = a.A.videoId || a.A.videoIds[a.A.index];
    b && Xp(a, "flingVideo", {
        videoId: b,
        currentTime: a.A.currentTime || 0
    });
    a.A = null
}
function Xp(a, b, c) {
    a.info("sendYoutubeMessage_: " + b + " " + S(c));
    var d = {};
    d.type = b;
    c && (d.data = c);
    a.g ? a.g.sendMessage("urn:x-cast:com.google.youtube.mdx", d, s, u(function() {
        this.warn("Failed to send message: " + b + ".")
    }, a)) : a.warn("Sending yt message without session: " + S(d))
}
g.ge = function(a, b) {
    if (!this.L())
        if (b) {
            var c = Ee(b);
            if (c) {
                var d = "" + c.type, c = c.data || {};
                this.info("onYoutubeMessage_: " + d + " " + S(c));
                switch (d) {
                case "mdxSessionStatus":
                    Vp(this, c.screenId);
                    break;
                default:
                    this.warn("Unknown youtube message: " + d)
                }
            } else 
                this.warn("Unable to parse message.")
        } else 
            this.warn("No data in message.")
    };
function Vp(a, b) {
    H(a.C);
    b ? (a.info("onConnectedScreenId_: Received screenId: " + b), a.k && a.k.id == b || Mp(a.P, a.j.label, b, a.j.friendlyName, u(function(a) {
        a ? Tp(this, a) : this.fa(Error("Unable to fetch screen."))
    }, a), u(a.fa, a))) : a.fa(Error("Waiting for session status timed out."))
}
g.Yc = function() {
    return this.g
};
g.he = function(a) {
    this.L() || a || (this.warn("Cast session died."), this.fa())
};
function Yp(a, b) {
    Sp.call(this, a, b, "DialSession");
    this.C = this.O = null;
    this.S = "";
    this.A = null;
    this.H = s;
    this.D = NaN;
    this.X = u(this.$d, this);
    this.g = s
}
y(Yp, Sp);
g = Yp.prototype;
g.mc = function(a) {
    this.C = a;
    this.C.addUpdateListener(this.X)
};
g.lb = function(a) {
    this.A = a;
    this.H()
};
g.stop = function() {
    this.g();
    this.g = s;
    H(this.D);
    this.C ? this.C.stop(u(this.fa, this, null), u(this.fa, this, "Failed to stop DIAL device.")) : this.fa()
};
g.G = function() {
    this.g();
    this.g = s;
    H(this.D);
    this.C && this.C.removeUpdateListener(this.X);
    this.C = null;
    Yp.J.G.call(this)
};
function Zp(a) {
    a.g = Pp(a.P, a.S, a.j.label, a.j.friendlyName, u(function(a) {
        this.g = s;
        Tp(this, a)
    }, a), u(function(a) {
        this.g = s;
        this.fa(a)
    }, a))
}
g.$d = function(a) {
    this.L() || a || (this.warn("DIAL session died."), this.g(), this.g = s, this.fa())
};
function $p(a) {
    var b = {};
    b.pairingCode = a.S;
    if (a.A) {
        var c = a.A.index || 0, d = a.A.currentTime || 0;
        b.v = a.A.videoId || a.A.videoIds[c];
        b.t = d
    }
    rp() && (b.env_useStageMdx = 1);
    return Ue(b)
}
g.jc = function(a) {
    this.S = Oo();
    if (this.A) {
        var b = new chrome.cast.DialLaunchResponse(!0, $p(this));
        a(b);
        Zp(this)
    } else 
        this.H = u(function() {
            H(this.D);
            this.H = s;
            this.D = NaN;
            var b = new chrome.cast.DialLaunchResponse(!0, $p(this));
            a(b);
            Zp(this)
        }, this), this.D = G(u(function() {
            this.H()
        }, this), 100)
};
g.Md = function(a, b) {
    Mp(this.P, this.O.receiver.label, a, this.j.friendlyName, u(function(a) {
        a && a.token ? (Tp(this, a), b(new chrome.cast.DialLaunchResponse(!1))) : this.jc(b)
    }, this), u(function(a) {
        this.warn("Failed to get DIAL screen: " + a);
        this.jc(b)
    }, this))
};
function aq(a, b) {
    Sp.call(this, a, b, "ManualSession");
    this.g = G(u(this.lb, this, null), 150)
}
y(aq, Sp);
aq.prototype.stop = function() {
    this.fa()
};
aq.prototype.mc = s;
aq.prototype.lb = function() {
    H(this.g);
    this.g = NaN;
    var a = Uo(this.P.da(), this.j.label);
    a ? Tp(this, a) : this.fa(Error("No such screen"))
};
aq.prototype.G = function() {
    H(this.g);
    this.g = NaN;
    aq.J.G.call(this)
};
function bq(a) {
    Ln.call(this);
    this.j = a;
    this.g = null;
    this.C=!1;
    this.k = [];
    this.A = u(this.Hd, this)
}
y(bq, Ln);
g = bq.prototype;
g.init = function(a, b) {
    chrome.cast.timeout.requestSession = 3E4;
    var c = new chrome.cast.SessionRequest("233637DE");
    c.dialRequest = new chrome.cast.DialRequest("YouTube");
    var d = chrome.cast.AutoJoinPolicy.TAB_AND_ORIGIN_SCOPED, e = a ? chrome.cast.DefaultActionPolicy.CAST_THIS_TAB: chrome.cast.DefaultActionPolicy.CREATE_SESSION, c = new chrome.cast.ApiConfig(c, u(this.Sc, this), u(this.se, this), d, e);
    c.customDialLaunchCallback = u(this.re, this);
    chrome.cast.initialize(c, u(function() {
        this.L() || (chrome.cast.addReceiverActionListener(this.A),
        Sn(cq), this.j.subscribe("onlineScreenChange", u(this.Ec, this)), this.k = dq(this), chrome.cast.setCustomReceivers(this.k, s, u(function(a) {
            this.R("Failed to set initial custom receivers: " + S(a))
        }, this)), this.publish("yt-remote-cast2-availability-change", eq(this)), b(!0))
    }, this), function(a) {
        this.R("Failed to initialize API: " + S(a));
        b(!1)
    })
};
g.Ye = function(a, b) {
    fq("Setting connected screen ID: " + a + " -> " + b);
    if (this.g) {
        var c = this.g.k;
        if (!a || c && c.id != a)
            fq("Unsetting old screen status: " + this.g.j.friendlyName), Kb(this.g), this.g = null
    }
    if (a && b) {
        if (!this.g) {
            c = Uo(this.j.da(), a);
            if (!c) {
                fq("setConnectedScreenStatus: Unknown screen.");
                return 
            }
            var d = gq(this, c);
            d || (fq("setConnectedScreenStatus: Connected receiver not custom..."), d = new chrome.cast.Receiver(c.uuid ? c.uuid : c.id, c.name), d.receiverType = chrome.cast.ReceiverType.CUSTOM, this.k.push(d), chrome.cast.setCustomReceivers(this.k,
            s, u(function(a) {
                this.R("Failed to set initial custom receivers: " + S(a))
            }, this)));
            fq("setConnectedScreenStatus: new active receiver: " + d.friendlyName);
            hq(this, new aq(this.j, d), !0)
        }
        this.g.gc(b)
    } else 
        fq("setConnectedScreenStatus: no screen.")
};
function gq(a, b) {
    return b ? Ra(a.k, function(a) {
        return Zn(b, a.label)
    }, a) : null
}
g.Ze = function(a) {
    this.L() ? this.R("Setting connection data on disposed cast v2") : this.g ? this.g.lb(a) : this.R("Setting connection data without a session")
};
g.stopSession = function() {
    this.L() ? this.R("Stopping session on disposed cast v2") : this.g ? (this.g.stop(), Kb(this.g), this.g = null) : fq("Stopping non-existing session")
};
g.requestSession = function() {
    chrome.cast.requestSession(u(this.Sc, this), u(this.Ve, this))
};
g.G = function() {
    this.j.unsubscribe("onlineScreenChange", u(this.Ec, this));
    window.chrome && chrome.cast && chrome.cast.removeReceiverActionListener(this.A);
    var a = cq, b = r("yt.mdx.remote.debug.handlers_");
    Ya(b || [], a);
    Kb(this.g);
    bq.J.G.call(this)
};
function fq(a) {
    Vn("Controller", a)
}
g.R = function(a) {
    Vn("Controller", a)
};
function cq(a) {
    window.chrome && chrome.cast && chrome.cast.logMessage && chrome.cast.logMessage(a)
}
function eq(a) {
    return a.C||!!a.k.length||!!a.g
}
function hq(a, b, c) {
    Kb(a.g);
    (a.g = b) ? (c ? a.publish("yt-remote-cast2-receiver-resumed", b.j) : a.publish("yt-remote-cast2-receiver-selected", b.j), b.subscribe("sessionScreen", u(a.Gc, a, b)), b.k ? a.publish("yt-remote-cast2-session-change", b.k) : c && a.g.lb(null)) : a.publish("yt-remote-cast2-session-change", null)
}
g.Gc = function(a, b) {
    this.g == a && (b || hq(this, null), this.publish("yt-remote-cast2-session-change", b))
};
g.Hd = function(a, b) {
    if (!this.L())
        if (a)
            switch (fq("onReceiverAction_ " + a.label + " / " + a.friendlyName + "-- " + b), b) {
            case chrome.cast.ReceiverAction.CAST:
                if (this.g)
                    if (this.g.j.label != a.label)
                        fq("onReceiverAction_: Stopping active receiver: " + this.g.j.friendlyName), this.g.stop();
                    else {
                        fq("onReceiverAction_: Casting to active receiver.");
                        this.g.k && this.publish("yt-remote-cast2-session-change", this.g.k);
                        break
                    }
                    switch (a.receiverType) {
                    case chrome.cast.ReceiverType.CUSTOM:
                        hq(this, new aq(this.j, a));
                        break;
                    case chrome.cast.ReceiverType.DIAL:
                        hq(this,
                        new Yp(this.j, a));
                        break;
                    case chrome.cast.ReceiverType.CAST:
                        hq(this, new Up(this.j, a));
                        break;
                    default:
                        this.R("Unknown receiver type: " + a.receiverType);
                        return 
                    }
                    break;
                case chrome.cast.ReceiverAction.STOP:
                    this.g && this.g.j.label == a.label ? this.g.stop() : this.R("Stopping receiver w/o session: " + a.friendlyName)
            } else 
                this.R("onReceiverAction_ called without receiver.")
};
g.re = function(a) {
    if (this.L())
        return Promise.reject(Error("disposed"));
    var b = a.receiver;
    b.receiverType != chrome.cast.ReceiverType.DIAL && (this.R("Not DIAL receiver: " + b.friendlyName), b.receiverType = chrome.cast.ReceiverType.DIAL);
    var c = this.g ? this.g.j: null;
    if (!c || c.label != b.label)
        return this.R("Receiving DIAL launch request for non-clicked DIAL receiver: " + b.friendlyName), Promise.reject(Error("illegal DIAL launch"));
    if (c && c.label == b.label && c.receiverType != chrome.cast.ReceiverType.DIAL) {
        if (this.g.k)
            return fq("Reselecting dial screen."),
            this.publish("yt-remote-cast2-session-change", this.g.k), Promise.resolve(new chrome.cast.DialLaunchResponse(!1));
        this.R('Changing CAST intent from "' + c.receiverType + '" to "dial" for ' + b.friendlyName);
        hq(this, new Yp(this.j, b))
    }
    b = this.g;
    b.O = a;
    return b.O.appState == chrome.cast.DialAppState.RUNNING ? new Promise(u(b.Md, b, (b.O.extraData || {}).screenId || null)) : new Promise(u(b.jc, b))
};
g.Sc = function(a) {
    if (!this.L()) {
        fq("New cast session ID: " + a.sessionId);
        var b = a.receiver;
        if (b.receiverType != chrome.cast.ReceiverType.CUSTOM) {
            if (!this.g)
                if (b.receiverType == chrome.cast.ReceiverType.CAST)
                    fq("Got resumed cast session before resumed mdx connection."), hq(this, new Up(this.j, b), !0);
                else {
                    this.R("Got non-cast session without previous mdx receiver event, or mdx resume.");
                    return 
                }
            var c = this.g.j, d = Uo(this.j.da(), c.label);
            d && Zn(d, b.label) && c.receiverType != chrome.cast.ReceiverType.CAST && b.receiverType ==
            chrome.cast.ReceiverType.CAST && (fq("onSessionEstablished_: manual to cast session change " + b.friendlyName), Kb(this.g), this.g = new Up(this.j, b), this.g.subscribe("sessionScreen", u(this.Gc, this, this.g)), this.g.lb(null));
            this.g.mc(a)
        }
    }
};
g.bf = function() {
    return this.g ? this.g.Yc() : null
};
g.Ve = function(a) {
    this.L() || (this.R("Failed to estabilish a session: " + S(a)), a.code != chrome.cast.ErrorCode.CANCEL && hq(this, null))
};
g.se = function(a) {
    fq("Receiver availability updated: " + a);
    if (!this.L()) {
        var b = eq(this);
        this.C = a == chrome.cast.ReceiverAvailability.AVAILABLE;
        eq(this) != b && this.publish("yt-remote-cast2-availability-change", eq(this))
    }
};
function dq(a) {
    var b = a.j.Oc(), c = a.g && a.g.j;
    a = A(b, function(a) {
        c && Zn(a, c.label) && (c = null);
        var b = a.uuid ? a.uuid: a.id, f = gq(this, a);
        f ? (f.label = b, f.friendlyName = a.name) : (f = new chrome.cast.Receiver(b, a.name), f.receiverType = chrome.cast.ReceiverType.CUSTOM);
        return f
    }, a);
    c && (c.receiverType != chrome.cast.ReceiverType.CUSTOM && (c = new chrome.cast.Receiver(c.label, c.friendlyName), c.receiverType = chrome.cast.ReceiverType.CUSTOM), a.push(c));
    return a
}
g.Ec = function() {
    if (!this.L()) {
        var a = eq(this);
        this.k = dq(this);
        fq("Updating custom receivers: " + S(this.k));
        chrome.cast.setCustomReceivers(this.k, s, u(function() {
            this.R("Failed to set custom receivers.")
        }, this));
        var b = eq(this);
        b != a && this.publish("yt-remote-cast2-availability-change", b)
    }
};
bq.prototype.setLaunchParams = bq.prototype.Ze;
bq.prototype.setConnectedScreenStatus = bq.prototype.Ye;
bq.prototype.stopSession = bq.prototype.stopSession;
bq.prototype.getCastSession = bq.prototype.bf;
bq.prototype.requestSession = bq.prototype.requestSession;
bq.prototype.init = bq.prototype.init;
bq.prototype.dispose = bq.prototype.dispose;
function iq(a, b) {
    uj ? kq(a) && (lq(!0), window.chrome && chrome.cast && chrome.cast.isAvailable ? mq(b) : Ho(function(a, d) {
        a ? mq(b) : (nq("Failed to load cast API: " + d), oq(!1), lq(!1), zo("yt-remote-cast-available"), zo("yt-remote-cast-receiver"), pq(), b(!1))
    })) : jq("Cannot initialize because not running Chrome")
}
function pq() {
    jq("dispose");
    Io();
    var a = qq();
    a && a.dispose();
    rq = null;
    q("yt.mdx.remote.cloudview.instance_", null, void 0);
    sq(!1);
    bc(tq);
    tq.length = 0
}
function uq() {
    jq("clearCurrentReciever");
    zo("yt-remote-cast-receiver")
}
function vq() {
    return yo("yt-remote-cast-installed") ? qq() ? rq.getCastSession() : (nq("getCastSelector: Cast is not initialized."), null) : (nq("getCastSelector: Cast API is not installed!"), null)
}
function wq(a, b) {
    xq() ? qq().setConnectedScreenStatus(a, b) : nq("setConnectedScreenStatus called before ready.")
}
var rq = null;
function yq(a) {
    rq.init(!1, a)
}
function kq(a) {
    var b=!1;
    if (!rq) {
        var c = r("yt.mdx.remote.cloudview.instance_");
        c || (c = new bq(a), c.subscribe("yt-remote-cast2-availability-change", function(a) {
            wo("yt-remote-cast-available", a);
            K("yt-remote-cast2-availability-change", a)
        }), c.subscribe("yt-remote-cast2-receiver-selected", function(a) {
            jq("onReceiverSelected: " + a.friendlyName);
            wo("yt-remote-cast-receiver", a);
            K("yt-remote-cast2-receiver-selected", a)
        }), c.subscribe("yt-remote-cast2-receiver-resumed", function(a) {
            jq("onReceiverResumed: " + a.friendlyName);
            wo("yt-remote-cast-receiver", a)
        }), c.subscribe("yt-remote-cast2-session-change", function(a) {
            jq("onSessionChange: " + eo(a));
            a || zo("yt-remote-cast-receiver");
            K("yt-remote-cast2-session-change", a)
        }), q("yt.mdx.remote.cloudview.instance_", c, void 0), b=!0);
        rq = c
    }
    jq("cloudview.createSingleton_: " + b);
    return b
}
function qq() {
    rq || (rq = r("yt.mdx.remote.cloudview.instance_"));
    return rq
}
function mq(a) {
    oq(!0);
    lq(!1);
    yq(function(b) {
        b ? (sq(!0), K("yt-remote-cast2-api-ready")) : (nq("Failed to initialize cast API."), oq(!1), zo("yt-remote-cast-available"), zo("yt-remote-cast-receiver"), pq());
        a(b)
    })
}
function jq(a) {
    Vn("cloudview", a)
}
function nq(a) {
    Vn("cloudview", a)
}
function oq(a) {
    jq("setCastInstalled_ " + a);
    wo("yt-remote-cast-installed", a)
}
function xq() {
    return !!r("yt.mdx.remote.cloudview.apiReady_")
}
function sq(a) {
    jq("setApiReady_ " + a);
    q("yt.mdx.remote.cloudview.apiReady_", a, void 0)
}
function lq(a) {
    q("yt.mdx.remote.cloudview.initializing_", a, void 0)
}
var tq = [];
function zq(a) {
    Aq(this, a)
}
function Bq(a, b) {
    if (a.j)
        throw Error(b + " is not allowed in V3.");
}
function Cq(a) {
    a.B =- 1;
    a.k=!1;
    a.C = null;
    a.g =- 1;
    a.A = null;
    a.M = 0;
    a.D = w()
}
function Aq(a, b) {
    a.videoIds = [];
    a.j = "";
    Dq(a);
    b && (a.videoIds = b.videoIds, a.index = b.index, a.j = b.listId, a.videoId = b.videoId, a.g = b.playerState, a.A = b.errorReason, a.B = b.volume, a.k = b.muted, a.C = b.trackData, a.M = b.playerTime, a.D = b.playerTimeAt)
}
function Dq(a) {
    a.index =- 1;
    a.videoId = "";
    Cq(a)
}
function Eq(a) {
    return a.j ? a.videoId : a.videoIds[a.index]
}
function Fq(a, b) {
    a.M = b;
    a.D = w()
}
function Gq(a) {
    switch (a.g) {
    case 1:
        return (w() - a.D) / 1E3 + a.M;
    case - 1E3:
        return 0
    }
    return a.M
}
zq.prototype.setVideoId = function(a) {
    Bq(this, "setVideoId");
    var b = this.index;
    this.index = La(this.videoIds, a);
    b != this.index && Cq(this);
    return - 1 != b
};
function Hq(a, b, c) {
    var d = a.videoId;
    a.videoId = b;
    a.index = c;
    b != d && Cq(a)
}
function Iq(a, b, c) {
    Bq(a, "setPlaylist");
    c = c || Eq(a);
    fb(a.videoIds, b) && c == Eq(a) || (a.videoIds = bb(b), a.setVideoId(c))
}
zq.prototype.add = function(a) {
    Bq(this, "add");
    return a&&!Ta(this.videoIds, a) ? (this.videoIds.push(a), !0) : !1
};
zq.prototype.remove = function(a) {
    Bq(this, "remove");
    var b = Eq(this);
    return Ya(this.videoIds, a) ? (this.index = La(this.videoIds, b), !0) : !1
};
function Jq(a) {
    var b = {};
    b.videoIds = bb(a.videoIds);
    b.index = a.index;
    b.listId = a.j;
    b.videoId = a.videoId;
    b.playerState = a.g;
    b.errorReason = a.A;
    b.volume = a.B;
    b.muted = a.k;
    b.trackData = zb(a.C);
    b.playerTime = a.M;
    b.playerTimeAt = a.D;
    return b
}
zq.prototype.clone = function() {
    return new zq(Jq(this))
};
function Kq(a, b) {
    Ln.call(this);
    this.k = 0;
    this.A = a;
    this.H = [];
    this.D = new Mn;
    this.C = NaN;
    this.j = this.g = null;
    this.S = u(this.Ed, this);
    this.O = u(this.vb, this);
    this.P = u(this.Dd, this);
    var c = 0;
    a ? (c = a.getProxyState(), 3 != c && (a.subscribe("proxyStateChange", this.cc, this), Lq(this))) : c = 3;
    0 != c && (b ? this.cc(c) : G(u(function() {
        this.cc(c)
    }, this), 0));
    Mq(this, vq())
}
y(Kq, Ln);
g = Kq.prototype;
g.getState = function() {
    return this.k
};
function Nq(a) {
    return new zq(a.A.getPlayerContextData())
}
g.play = function() {
    Oq(this) ? (this.g ? this.g.play(null, s, u(function() {
        this.R("Failed to play video with cast v2 channel.");
        Pq(this, "play")
    }, this)) : Pq(this, "play"), Qq(this, 1, Gq(Nq(this))), Rq(this)) : Sq(this, this.play)
};
g.pause = function() {
    Oq(this) ? (this.g ? this.g.pause(null, s, u(function() {
        this.R("Failed to pause video with cast v2 channel.");
        Pq(this, "pause")
    }, this)) : Pq(this, "pause"), Qq(this, 2, Gq(Nq(this))), Rq(this)) : Sq(this, this.pause)
};
g.stop = function() {
    if (Oq(this)) {
        this.g ? this.g.stop(null, s, u(function() {
            this.R("Failed to stop video with cast v2 channel.");
            Pq(this, "stopVideo")
        }, this)) : Pq(this, "stopVideo");
        var a = Nq(this);
        Dq(a);
        Tq(this, a);
        Rq(this)
    } else 
        Sq(this, this.stop)
};
g.ld = function(a) {
    Oq(this) ? Pq(this, "addVideos", {
        listId: a
    }) : Sq(this, v(this.ld, a))
};
g.bd = function(a) {
    if (Oq(this)) {
        Pq(this, "removeVideo", {
            videoId: a
        });
        var b = Nq(this);
        b.j || (b.remove(a), Tq(this, b))
    } else 
        Sq(this, v(this.bd, a))
};
g.qd = function(a, b, c) {
    Oq(this) ? Pq(this, "moveVideo", {
        videoId: a,
        delta: (b >= c ? c : c - 1) - b
    }) : Sq(this, v(this.qd, a, b, c))
};
g.pd = function(a) {
    Oq(this) ? Pq(this, "insertVideo", {
        videoId: a
    }) : Sq(this, v(this.pd, a))
};
function Uq(a, b, c, d, e) {
    var f = Nq(a);
    d = d || 0;
    var h = {
        videoId: b,
        currentIndex: d,
        listId: e || f.j
    };
    Hq(f, b, d);
    n(c) && (Fq(f, c), h.currentTime = c);
    Pq(a, "setPlaylist", h);
    e || Tq(a, f)
}
g.jd = function(a, b) {
    if (Oq(this)) {
        var c = Nq(this);
        Hq(c, a, b);
        Pq(this, "previous");
        Tq(this, c)
    } else 
        Sq(this, v(this.jd, a, b))
};
g.kd = function(a, b) {
    if (Oq(this)) {
        var c = Nq(this);
        Hq(c, a, b);
        Pq(this, "next");
        Tq(this, c)
    } else 
        Sq(this, v(this.kd, a, b))
};
g.nd = function() {
    if (Oq(this)) {
        Pq(this, "clearPlaylist");
        var a = Nq(this);
        Aq(a);
        Tq(this, a);
        Rq(this)
    } else 
        Sq(this, this.nd)
};
g.dispose = function() {
    if (3 != this.k) {
        var a = this.k;
        this.k = 3;
        this.publish("proxyStateChange", a, this.k)
    }
    Kq.J.dispose.call(this)
};
g.G = function() {
    H(this.C);
    this.C = NaN;
    Vq(this);
    this.A = null;
    this.D.clear();
    Mq(this, null);
    Kq.J.G.call(this)
};
function Lq(a) {
    z(["remotePlayerChange", "remoteQueueChange"], function(a) {
        this.H.push(this.A.subscribe(a, v(this.Le, a), this))
    }, a)
}
function Vq(a) {
    z(a.H, function(a) {
        this.A.unsubscribeByKey(a)
    }, a);
    a.H.length = 0
}
function Oq(a) {
    return 1 == a.getState()
}
function Sq(a, b) {
    50 > a.D.Ga() && a.D.j.push(b)
}
function Qq(a, b, c) {
    var d = Nq(a);
    Fq(d, c);
    - 1E3 != d.g && (d.g = b);
    Tq(a, d)
}
function Pq(a, b, c) {
    a.A.sendMessage(b, c)
}
function Tq(a, b) {
    Vq(a);
    a.A.setPlayerContextData(Jq(b));
    Lq(a)
}
g.cc = function(a) {
    if ((a != this.k || 2 == a) && 3 != this.k && 0 != a) {
        var b = this.k;
        this.k = a;
        this.publish("proxyStateChange", b, a);
        if (1 == a)
            for (; !this.D.isEmpty();)
                b = a = this.D, Ua(b.g) && (b.g = b.j, b.g.reverse(), b.j = []), a.g.pop().apply(this);
        else 
            3 == a && this.dispose()
    }
};
function Rq(a) {
    H(a.C);
    a.C = G(u(function() {
        this.publish("remotePlayerChange");
        this.C = NaN
    }, a), 2E3)
}
g.Le = function(a) {
    ("remotePlayerChange" != a || isNaN(this.C)) && this.publish(a)
};
function Mq(a, b) {
    a.j && (a.j.removeUpdateListener(a.S), a.j.removeMediaListener(a.O), a.vb(null));
    a.j = b;
    a.j && (Vn("CP", "Setting cast session: " + a.j.sessionId), a.j.addUpdateListener(a.S), a.j.addMediaListener(a.O), a.j.media.length && a.vb(a.j.media[0]))
}
g.Ed = function(a) {
    if (!a)
        this.vb(null), Mq(this, null);
    else if (this.j.receiver.volume) {
        a = this.j.receiver.volume;
        var b = Nq(this);
        if (b.B != a.level || b.k != a.muted)
            Vn("CP", "Cast volume update: " + a.level + (a.muted ? " muted" : "")), b.B = Math.round(100 * a.level || 0), b.k=!!a.muted, Tq(this, b), Rq(this)
    }
};
g.vb = function(a) {
    Vn("CP", "Cast media: "+!!a);
    this.g && this.g.removeUpdateListener(this.P);
    if (this.g = a)
        this.g.addUpdateListener(this.P), Wq(this), Rq(this)
};
function Wq(a) {
    var b = a.g.customData;
    if (a.g.media) {
        var c = a.g.media, d = Nq(a);
        c.contentId != d.videoId && Vn("CP", "Cast changing video to: " + c.contentId);
        var e = c.customData;
        d.index = e.currentIndex;
        d.j = e.listId;
        d.videoId = c.contentId;
        d.g = b.playerState;
        Fq(d, a.g.getEstimatedTime());
        Tq(a, d)
    } else 
        Vn("CP", "No cast media video. Ignoring state update.")
}
g.Dd = function(a) {
    a ? (Wq(this), Rq(this)) : this.vb(null)
};
g.R = function(a) {
    Vn("CP", a)
};
function Y(a, b, c) {
    Ln.call(this);
    this.ba = a;
    this.O = [];
    this.O.push(R(window, "beforeunload", u(this.Ud, this)));
    this.j = [];
    this.F = new zq;
    3 == c["mdx-version"] && (this.F.j = "RQ" + b.token);
    this.P = b.id;
    this.g = Xq(this, c);
    this.g.subscribe("handlerOpened", this.Zd, this);
    this.g.subscribe("handlerClosed", this.Vd, this);
    this.g.subscribe("handlerError", this.Wd, this);
    this.F.j ? this.g.subscribe("handlerMessage", this.Xd, this) : this.g.subscribe("handlerMessage", this.Yd, this);
    Kn(this.g, b.token);
    this.subscribe("remoteQueueChange",
    function() {
        var a = this.F.videoId;
        cp() && wo("yt-remote-session-video-id", a)
    }, this)
}
y(Y, Ln);
g = Y.prototype;
g.Gb = NaN;
g.sc=!1;
g.Lb = NaN;
g.xc = NaN;
g.Jb = NaN;
g.connect = function(a, b) {
    if (b) {
        if (this.F.j) {
            var c = b.listId, d = b.videoId, e = b.index, f = b.currentTime || 0;
            5 >= f && (f = 0);
            h = {
                videoId: d,
                currentTime: f
            };
            c && (h.listId = c);
            n(e) && (h.currentIndex = e);
            c && (this.F.j = c);
            this.F.videoId = d;
            this.F.index = e || 0
        } else {
            var d = b.videoIds[b.index], f = b.currentTime || 0;
            5 >= f && (f = 0);
            var h = {
                videoIds: d,
                videoId: d,
                currentTime: f
            };
            this.F.videoIds = [d];
            this.F.index = 0
        }
        this.F.state = 3;
        Fq(this.F, f);
        this.N("Connecting with setPlaylist and params: " + S(h));
        this.g.connect({
            method: "setPlaylist",
            params: S(h)
        },
        a, gp())
    } else 
        this.N("Connecting without params"), this.g.connect({}, a, gp());
    Yq(this)
};
g.dispose = function() {
    this.L() || (this.publish("beforeDispose"), Zq(this, 3));
    Y.J.dispose.call(this)
};
g.G = function() {
    $q(this);
    ar(this);
    br(this);
    H(this.Jb);
    this.Jb = NaN;
    this.A = null;
    le(this.O);
    this.O.length = 0;
    this.g.dispose();
    Y.J.G.call(this);
    this.j = this.F = this.g = null
};
g.N = function(a) {
    Vn("conn", a)
};
g.Ud = function() {
    this.C(2)
};
function Xq(a, b) {
    return new In(Ko(a.ba, "/bc", void 0, !1), b)
}
function Zq(a, b) {
    a.publish("proxyStateChange", b)
}
function Yq(a) {
    a.Gb = G(u(function() {
        this.N("Connecting timeout");
        this.C(1)
    }, a), 2E4)
}
function $q(a) {
    H(a.Gb);
    a.Gb = NaN
}
function br(a) {
    H(a.Lb);
    a.Lb = NaN
}
function cr(a) {
    ar(a);
    a.xc = G(u(function() {
        this.k("getNowPlaying")
    }, a), 2E4)
}
function ar(a) {
    H(a.xc);
    a.xc = NaN
}
function dr(a) {
    var b = a.g;
    return !!b.g && 3 == b.g.getState() && isNaN(a.Gb)
}
g.Zd = function() {
    this.N("Channel opened");
    this.sc && (this.sc=!1, br(this), this.Lb = G(u(function() {
        this.N("Timing out waiting for a screen.");
        this.C(1)
    }, this), 15E3));
    pp(Jn(this.g), this.P)
};
g.Vd = function() {
    this.N("Channel closed");
    isNaN(this.Gb) ? qp(!0) : qp();
    this.dispose()
};
g.Wd = function(a) {
    qp();
    isNaN(this.D()) ? (this.N("Channel error: " + a + " without reconnection"), this.dispose()) : (this.sc=!0, this.N("Channel error: " + a + " with reconnection in " + this.D() + " ms"), Zq(this, 2))
};
function er(a, b) {
    b && ($q(a), br(a));
    b == dr(a) ? b && (Zq(a, 1), a.k("getSubtitlesTrack")) : b ? (a.H() && Aq(a.F), Zq(a, 1), a.k("getNowPlaying")) : a.C(1)
}
function fr(a, b) {
    var c = b.params.videoId;
    delete b.params.videoId;
    c == a.F.videoId && (wb(b.params) ? a.F.C = null : a.F.C = b.params, a.publish("remotePlayerChange"))
}
function gr(a, b) {
    var c = b.params.videoId || b.params.video_id, d = parseInt(b.params.currentIndex, 10);
    a.F.j = b.params.listId;
    Hq(a.F, c, d);
    a.publish("remoteQueueChange")
}
function hr(a, b) {
    b.params = b.params || {};
    gr(a, b);
    ir(a, b)
}
function ir(a, b) {
    var c = parseInt(b.params.currentTime || b.params.current_time, 10);
    Fq(a.F, isNaN(c) ? 0 : c);
    c = parseInt(b.params.state, 10);
    c = isNaN(c)?-1 : c;
    - 1 == c&&-1E3 == a.F.g && (c =- 1E3);
    a.F.g = c;
    var d = null;
    - 1E3 == c && (d = a.F.A || "unknown", n(b.params.currentError) && (d = De(b.params.currentError).reason || d));
    a.F.A = d;
    1 == a.F.g ? cr(a) : ar(a);
    a.publish("remotePlayerChange")
}
function jr(a, b) {
    var c = "true" == b.params.muted;
    a.F.B = parseInt(b.params.volume, 10);
    a.F.k = c;
    a.publish("remotePlayerChange")
}
g.Xd = function(a) {
    a.params ? this.N("Received: action=" + a.action + ", params=" + S(a.params)) : this.N("Received: action=" + a.action + " {}");
    switch (a.action) {
    case "loungeStatus":
        a = De(a.params.devices);
        this.j = A(a, function(a) {
            return new pm(a)
        });
        a=!!Ra(this.j, function(a) {
            return "LOUNGE_SCREEN" == a.type
        });
        er(this, a);
        break;
    case "loungeScreenConnected":
        er(this, !0);
        break;
    case "loungeScreenDisconnected":
        $a(this.j, function(a) {
            return "LOUNGE_SCREEN" == a.type
        });
        er(this, !1);
        break;
    case "remoteConnected":
        var b = new pm(De(a.params.device));
        Ra(this.j, function(a) {
            return a.equals(b)
        }) || Xa(this.j, b);
        break;
    case "remoteDisconnected":
        b = new pm(De(a.params.device));
        $a(this.j, function(a) {
            return a.equals(b)
        });
        break;
    case "gracefulDisconnect":
        break;
    case "playlistModified":
        gr(this, a);
        break;
    case "nowPlaying":
        hr(this, a);
        break;
    case "onStateChange":
        ir(this, a);
        break;
    case "onVolumeChanged":
        jr(this, a);
        break;
    case "onSubtitlesTrackChanged":
        fr(this, a);
        break;
    default:
        this.N("Unrecognized action: " + a.action)
    }
};
g.Yd = function(a) {
    a.params ? this.N("Received: action=" + a.action + ", params=" + S(a.params)) : this.N("Received: action=" + a.action);
    kr(this, a);
    lr(this, a);
    if (dr(this)) {
        var b = this.F.clone(), c=!1, d, e, f, h, k, m, p;
        a.params && (d = a.params.videoId || a.params.video_id, e = a.params.videoIds || a.params.video_ids, f = a.params.state, h = a.params.currentTime || a.params.current_time, k = a.params.volume, m = a.params.muted, n(a.params.currentError) && (p = De(a.params.currentError)));
        if ("onSubtitlesTrackChanged" == a.action)
            d == Eq(this.F) && (delete a.params.videoId,
            wb(a.params) ? this.F.C = null : this.F.C = a.params, this.publish("remotePlayerChange"));
        else if (Eq(this.F) || "onStateChange" != a.action)
            "playlistModified" != a.action && "nowPlayingPlaylist" != a.action || e ? (d || "nowPlaying" != a.action && "nowPlayingPlaylist" != a.action ? d || (d = Eq(this.F)) : this.F.setVideoId(""), e && (e = e.split(","), Iq(this.F, e, d))) : Iq(this.F, []), this.F.add(d) && this.k("getPlaylist"), d && this.F.setVideoId(d), b.index == this.F.index && fb(b.videoIds, this.F.videoIds) || this.publish("remoteQueueChange"), n(f) && (b = parseInt(f,
            10), b = isNaN(b)?-1 : b, - 1 == b&&-1E3 == this.F.g && (b =- 1E3), 0 == b && "0" == h && (b =- 1), c = c || b != this.F.g, this.F.g = b, d = null, - 1E3 == b && (d = this.F.A || "unknown", p && (d = p.reason || d)), c = c || this.F.A != d, this.F.A = d, 1 == this.F.g ? cr(this) : ar(this)), "onError" != a.action||-1 != this.F.g&&-1E3 != this.F.g || (a = De(a.params.errors) || [], 1 == a.length && "PLAYER_ERROR" == a[0].error && a[0].videoId == Eq(this.F) && (this.F.g =- 1E3, this.F.A = a[0].reason || "unknown", c=!0)), h && (b = parseInt(h, 10), Fq(this.F, isNaN(b) ? 0 : b), c=!0), n(k) && (b = parseInt(k, 10), isNaN(b) ||
        (c = c || this.F.B != b, this.F.B = b), n(m) && (m = "true" == m, c = c || this.F.k != m, this.F.k = m)), c && this.publish("remotePlayerChange")
    }
};
function kr(a, b) {
    switch (b.action) {
    case "loungeStatus":
        var c = De(b.params.devices);
        a.j = A(c, function(a) {
            return new pm(a)
        });
        break;
    case "loungeScreenDisconnected":
        $a(a.j, function(a) {
            return "LOUNGE_SCREEN" == a.type
        });
        break;
    case "remoteConnected":
        var d = new pm(De(b.params.device));
        Ra(a.j, function(a) {
            return a.equals(d)
        }) || Xa(a.j, d);
        break;
    case "remoteDisconnected":
        d = new pm(De(b.params.device)), $a(a.j, function(a) {
            return a.equals(d)
        })
    }
}
function lr(a, b) {
    var c=!1;
    if ("loungeStatus" == b.action)
        c=!!Ra(a.j, function(a) {
            return "LOUNGE_SCREEN" == a.type
        });
    else if ("loungeScreenConnected" == b.action)
        c=!0;
    else if ("loungeScreenDisconnected" == b.action)
        c=!1;
    else 
        return;
    if (!isNaN(a.Lb))
        if (c)
            br(a);
        else 
            return;
    c == dr(a) ? c && Zq(a, 1) : c ? ($q(a), a.H() && Aq(a.F), Zq(a, 1), a.k("getNowPlaying")) : a.C(1)
}
g.Fd = function() {
    if (this.A) {
        var a = this.A;
        this.A = null;
        this.F.videoId != a && this.k("getNowPlaying")
    }
};
Y.prototype.subscribe = Y.prototype.subscribe;
Y.prototype.unsubscribeByKey = Y.prototype.ya;
Y.prototype.Z = function() {
    var a = 3;
    this.L() || (a = 0, isNaN(this.D()) ? dr(this) && (a = 1) : a = 2);
    return a
};
Y.prototype.getProxyState = Y.prototype.Z;
Y.prototype.C = function(a) {
    this.N("Disconnecting with " + a);
    $q(this);
    this.publish("beforeDisconnect", a);
    1 == a && qp();
    this.g.disconnect(a);
    this.dispose()
};
Y.prototype.disconnect = Y.prototype.C;
Y.prototype.X = function() {
    var a = this.F;
    this.A && (a = this.F.clone(), Hq(a, this.A, a.index));
    return Jq(a)
};
Y.prototype.getPlayerContextData = Y.prototype.X;
Y.prototype.na = function(a) {
    var b = new zq(a);
    b.videoId && b.videoId != this.F.videoId && (this.A = b.videoId, H(this.Jb), this.Jb = G(u(this.Fd, this), 5E3));
    var c = [];
    this.F.j == b.j && this.F.videoId == b.videoId && this.F.index == b.index && fb(this.F.videoIds, b.videoIds) || c.push("remoteQueueChange");
    this.F.g == b.g && this.F.B == b.B && this.F.k == b.k && Gq(this.F) == Gq(b) && S(this.F.C) == S(b.C) || c.push("remotePlayerChange");
    Aq(this.F, a);
    z(c, function(a) {
        this.publish(a)
    }, this)
};
Y.prototype.setPlayerContextData = Y.prototype.na;
Y.prototype.S = function() {
    return this.g.B.loungeIdToken
};
Y.prototype.getLoungeToken = Y.prototype.S;
Y.prototype.H = function() {
    var a = this.g.getDeviceId(), b = Ra(this.j, function(b) {
        return "REMOTE_CONTROL" == b.type && b.id != a
    });
    return b ? b.id : ""
};
Y.prototype.getOtherConnectedRemoteId = Y.prototype.H;
Y.prototype.D = function() {
    var a = this.g;
    return a.j.enabled ? a.j.A - w() : NaN
};
Y.prototype.getReconnectTimeout = Y.prototype.D;
Y.prototype.ka = function() {
    if (!isNaN(this.D())) {
        var a = this.g.j;
        a.enabled && (a.stop(), a.start(), a.C())
    }
};
Y.prototype.reconnect = Y.prototype.ka;
Y.prototype.k = function(a, b) {
    b ? this.N("Sending: action=" + a + ", params=" + S(b)) : this.N("Sending: action=" + a);
    this.g.sendMessage(a, b)
};
Y.prototype.sendMessage = Y.prototype.k;
function mr() {
    if (!("cast"in window))
        return !1;
    var a = window.cast || {};
    return "ActivityStatus"in a && "Api"in a && "LaunchRequest"in a && "Receiver"in a
}
function nr(a) {
    Vn("CAST", a)
}
function or(a) {
    var b = pr();
    b && b.logMessage && b.logMessage(a)
}
function qr() {
    if (!r("yt.mdx.remote.castv2_")&&!rr && (Ua(Wa) && cb(Wa, yo("yt-remote-online-dial-devices") || []), mr())) {
        var a = pr();
        a ? (a.removeReceiverListener("YouTube", sr), a.addReceiverListener("YouTube", sr), nr("API initialized in the other binary")) : (a = new cast.Api, q("yt.mdx.remote.castapi.api_", a, void 0), a.addReceiverListener("YouTube", sr), a.setReloadTabRequestHandler && a.setReloadTabRequestHandler(function() {
            G(function() {
                window.location.reload(!0)
            }, 1E3)
        }), Sn(or), nr("API initialized"));
        rr=!0
    }
}
function tr(a) {
    var b = Sa(Wa, function(b) {
        return b.id == a.id
    });
    0 <= b && (Wa[b] = Mo(a))
}
function sr(a) {
    a.length && nr("Updating receivers: " + S(a));
    ur(a);
    K("yt-remote-cast-device-list-update");
    z(vr(), function(a) {
        wr(a.id)
    });
    z(a, function(a) {
        if (a.isTabProjected) {
            var c = xr(a.id);
            nr("Detected device: " + c.id + " is tab projected. Firing DEVICE_TAB_PROJECTED event.");
            G(function() {
                K("yt-remote-cast-device-tab-projected", c.id)
            }, 1E3)
        }
    })
}
function yr(a, b) {
    nr("Updating " + a + " activity status: " + S(b));
    var c = xr(a);
    c ? (b.activityId && (c.activityId = b.activityId), c.status = "running" == b.status ? "RUNNING" : "stopped" == b.status ? "STOPPED" : "error" == b.status ? "ERROR" : "UNKNOWN", "RUNNING" != c.status && (c.activityId = ""), tr(c), K("yt-remote-cast-device-status-update", c)) : nr("Device not found")
}
function vr() {
    qr();
    return Qo()
}
function ur(a) {
    a = A(a, function(a) {
        var c = a.id, d;
        d = a.name;
        d = Da(d, "&") ? "document"in l ? Aa(d) : Ca(d) : d;
        c = {
            id: c,
            name: d
        };
        if (a = xr(a.id))
            c.activityId = a.activityId, c.status = a.status;
        return c
    });
    Va();
    cb(Wa, a)
}
function xr(a) {
    var b = vr();
    return Ra(b, function(b) {
        return b.id == a
    }) || null
}
function wr(a) {
    var b = xr(a), c = pr();
    c && b && b.activityId && c.getActivityStatus(b.activityId, function(b) {
        "error" == b.status && (b.status = "stopped");
        yr(a, b)
    })
}
function zr(a) {
    qr();
    var b = xr(a), c = pr();
    c && b && b.activityId ? (nr("Stopping cast activity"), c.stopActivity(b.activityId, v(yr, a))) : nr("Dropping cast activity stop")
}
function pr() {
    return r("yt.mdx.remote.castapi.api_")
}
var rr=!1, Wa = r("yt.mdx.remote.castapi.devices_") || [];
q("yt.mdx.remote.castapi.devices_", Wa, void 0);
function Ar(a) {
    Ln.call(this);
    this.k = a;
    this.Ea = Br();
    this.N("Initializing local screens: " + fo(this.Ea));
    this.La = Cr();
    this.N("Initializing account screens: " + fo(this.La));
    this.ec = null;
    this.g = [];
    this.j = [];
    Dr(this, vr() || []);
    this.N("Initializing DIAL devices: " + No(this.j));
    a = co(np());
    Er(this, a);
    this.N("Initializing online screens: " + fo(this.g));
    this.A = w() + 3E5;
    Fr(this)
}
y(Ar, Ln);
g = Ar.prototype;
g.od = NaN;
g.N = function(a) {
    Vn("RM", a)
};
g.R = function(a) {
    Vn("RM", a)
};
function Cr() {
    var a = Br(), b = co(np());
    return Na(b, function(b) {
        return !To(a, b)
    })
}
function Br() {
    var a = co(ip());
    return Na(a, function(a) {
        return !a.uuid
    })
}
function Fr(a) {
    J("yt-remote-cast-device-list-update", function() {
        var a = vr();
        Dr(this, a || [])
    }, a);
    J("yt-remote-cast-device-status-update", a.Me, a);
    a.ac();
    var b = w() > a.A ? 2E4: 1E4;
    Rb(u(a.ac, a), b)
}
g.publish = function(a, b) {
    if (this.L())
        return !1;
    this.N("Firing " + a);
    return this.B.publish.apply(this.B, arguments)
};
g.ac = function() {
    var a = vr() || [];
    Ua(a) || Dr(this, a);
    a = Gr(this);
    Ua(a) || (Pa(a, function(a) {
        return !To(this.La, a)
    }, this) && lp() ? Hr(this) : Ir(this, a))
};
function Jr(a) {
    Er(a, a.g, !0);
    Dr(a, vr() || []);
    a.publish("managedScreenChange", Gr(a))
}
function Kr(a, b) {
    var c = Gr(a);
    return Na(b, function(a) {
        return a.uuid ? (a = So(this.j, a.uuid), !!a && "RUNNING" == a.status) : !!To(c, a)
    }, a)
}
function Dr(a, b) {
    var c=!1;
    z(b, function(a) {
        var b = Uo(this.Ea, a.id);
        b && b.name != a.name && (this.N("Renaming screen id " + b.id + " from " + b.name + " to " + a.name), b.name = a.name, c=!0)
    }, a);
    c && (a.N("Renaming due to DIAL."), Lr(a));
    op(Po(b));
    var d=!fb(a.j, b, Ro);
    d && a.N("Updating DIAL devices: " + No(a.j) + " to " + No(b));
    a.j = b;
    Er(a, a.g);
    d && a.publish("onlineReceiverChange")
}
g.Me = function(a) {
    var b = So(this.j, a.id);
    b && (this.N("Updating DIAL device: " + b.id + "(" + b.name + ") from status: " + b.status + " to status: " + a.status + " and from activityId: " + b.activityId + " to activityId: " + a.activityId), b.activityId = a.activityId, b.status = a.status, op(Po(this.j)));
    Er(this, this.g)
};
function Er(a, b, c) {
    var d = Kr(a, b), e=!fb(a.g, d, $n);
    if (e || c)
        Ua(b) || mp(A(d, ao));
    e && (a.N("Updating online screens: " + fo(a.g) + " -> " + fo(d)), a.g = d, a.publish("onlineReceiverChange"))
}
function Ir(a, b) {
    var c = [], d = {};
    z(b, function(a) {
        a.token && (d[a.token] = a, c.push(a.token))
    });
    var e = {
        method: "POST",
        ca: {
            lounge_token: c.join(",")
        },
        context: a,
        V: function(a, b) {
            var c = [];
            z(b.screens || [], function(a) {
                "online" == a.status && c.push(d[a.loungeToken])
            });
            var e = this.ec ? Mr(this, this.ec): null;
            e&&!To(c, e) && c.push(e);
            Er(this, c, !0)
        }
    };
    ff(Ko(a.k, "/pairing/get_screen_availability"), e)
}
function Hr(a) {
    var b = Gr(a), c = A(b, function(a) {
        return a.id
    });
    Ua(c) || (a.N("Updating lounge tokens for: " + S(c)), ff(Ko(a.k, "/pairing/get_lounge_token_batch"), {
        ca: {
            screen_ids: c.join(",")
        },
        method: "POST",
        context: a,
        V: function(a, c) {
            Nr(this, c.screens || []);
            this.Ea = Na(this.Ea, function(a) {
                return !!a.token
            });
            Lr(this);
            Ir(this, b)
        }
    }))
}
function Nr(a, b) {
    z(ab(a.Ea, a.La), function(a) {
        var d = Ra(b, function(b) {
            return a.id == b.screenId
        });
        d && (a.token = d.loungeToken)
    })
}
function Lr(a) {
    var b = Br();
    fb(a.Ea, b, $n) || (a.N("Saving local screens: " + fo(b) + " to " + fo(a.Ea)), hp(A(a.Ea, ao)), Jr(a))
}
g.cd = function(a, b) {
    for (var c = Gr(this), c = A(c, function(a) {
        return a.name
    }), d = a, e = 2; Ta(c, d);)
        d = b.call(l, e), e++;
    return d
};
function Mr(a, b) {
    var c = Uo(Gr(a), b);
    a.N("Found screen: " + eo(c) + " with key: " + b);
    return c
}
function Gr(a) {
    return ab(a.La, Na(a.Ea, function(a) {
        return !To(this.La, a)
    }, a))
}
function Or() {
    var a = Pr, b = Ko(a.k, "/screens");
    ff(b, {
        method: "GET",
        context: a,
        withCredentials: !0,
        V: function(a, b) {
            this.La = A(sb(b), function(a) {
                return new Yn(a)
            });
            this.N("Load account screens successfully.");
            this.ac();
            Jr(this);
            this.publish("accountScreenLoad")
        },
        onError: function() {
            this.R("Load account screens failed.");
            Qr(this)
        }
    })
}
function Qr(a) {
    Ua(a.La) && Ua(Cr()) || (a.N("Clear cached account screens."), a.La = [], a.ac(), Jr(a))
};
function Rr(a) {
    Vo.call(this, "ScreenServiceProxy");
    this.ha = a;
    this.g = [];
    this.g.push(this.ha.$_s("screenChange", u(this.ye, this)));
    this.g.push(this.ha.$_s("onlineScreenChange", u(this.xe, this)))
}
y(Rr, Vo);
g = Rr.prototype;
g.da = function(a) {
    return this.ha.$_gs(a)
};
g.contains = function(a) {
    return !!this.ha.$_c(a)
};
g.get = function(a) {
    return this.ha.$_g(a)
};
g.start = function(a) {
    this.ha.$_st(a)
};
g.add = function(a, b, c) {
    this.ha.$_a(a, b, c)
};
g.remove = function(a, b, c) {
    this.ha.$_r(a, b, c)
};
g.Pa = function(a, b, c, d) {
    this.ha.$_un(a, b, c, d)
};
g.G = function() {
    for (var a = 0, b = this.g.length; a < b; ++a)
        this.ha.$_ubk(this.g[a]);
    this.g.length = 0;
    this.ha = null;
    Rr.J.G.call(this)
};
g.ye = function() {
    this.publish("screenChange")
};
g.xe = function() {
    this.publish("onlineScreenChange")
};
X.prototype.$_st = X.prototype.start;
X.prototype.$_gspc = X.prototype.af;
X.prototype.$_c = X.prototype.contains;
X.prototype.$_g = X.prototype.get;
X.prototype.$_a = X.prototype.add;
X.prototype.$_un = X.prototype.Pa;
X.prototype.$_r = X.prototype.remove;
X.prototype.$_gs = X.prototype.da;
X.prototype.$_gos = X.prototype.Oc;
X.prototype.$_s = X.prototype.subscribe;
X.prototype.$_ubk = X.prototype.ya;
function Sr(a, b, c) {
    a ? q("yt.mdx.remote.castv2_", !0, void 0) : qr();
    vo && uo();
    $o();
    Tr || (Tr = new Jo, rp() && (Tr.g = "/api/loungedev"));
    Pr || a || (Pr = new Ar(Tr), Pr.subscribe("screenPair", Ur), Pr.subscribe("managedScreenChange", Vr), Pr.subscribe("onlineReceiverChange", function() {
        K("yt-remote-receiver-availability-change")
    }));
    Wr || (Wr = r("yt.mdx.remote.deferredProxies_") || [], q("yt.mdx.remote.deferredProxies_", Wr, void 0));
    Xr(b);
    b = Yr();
    if (a&&!b) {
        var d = new X(Tr, Zr(c));
        q("yt.mdx.remote.screenService_", d, void 0);
        b = Yr();
        iq(d,
        function(a) {
            a ? $r() && wq($r(), "YouTube TV") : d.subscribe("onlineScreenChange", function() {
                K("yt-remote-receiver-availability-change")
            })
        })
    }
    if (c&&!r("yt.mdx.remote.initialized_")) {
        q("yt.mdx.remote.initialized_", !0, void 0);
        as("Initializing: " + S(c));
        bs.push(J("yt-remote-cast2-availability-change", function() {
            K("yt-remote-receiver-availability-change")
        }));
        bs.push(J("yt-remote-cast2-receiver-selected", function() {
            cs();
            K("yt-remote-auto-connect", "cast-selector-receiver")
        }));
        bs.push(J("yt-remote-cast2-session-change",
        ds));
        bs.push(J("yt-remote-connection-change", function(a) {
            a ? wq($r(), "YouTube TV") : es() || (wq(null, null), uq())
        }));
        var e = fs();
        c.isAuto && (e.id += "#dial");
        e.name = c.device;
        e.app = c.app;
        as(" -- with channel params: " + S(e));
        gs(e);
        a && b.start(Zr(c));
        $r() || hs();
        Pr && (c.loadAccountScreens || $r() ? Or() : Qr(Pr))
    }
}
function is() {
    var a = $r();
    if (!a)
        return null;
    if (!Pr) {
        var b = Yr().da();
        return Uo(b, a)
    }
    return Mr(Pr, a)
}
function ds(a) {
    as("remote.onCastSessionChange_: " + eo(a));
    if (a) {
        var b = is();
        b && b.id == a.id ? wq(b.id, "YouTube TV") : (b && js(), ks(a, 1))
    } else 
        js()
}
function js() {
    if (Pr) {
        var a = Pr;
        H(a.od);
        a.od = NaN
    }
    t: {
        if (a = ls())
            if (a = a.getOtherConnectedRemoteId()) {
                as("Do not stop DIAL due to " + a);
                ms();
                break t
            }(a = r("yt.mdx.remote.currentDialId_")) ? (as("Stopping DIAL: " + a), zr(a), ms()) : (a = is()) && a.uuid && (as("Stopping DIAL: " + a.uuid), zr(a.uuid))
    }
    xq() ? qq().stopSession() : nq("stopSession called before API ready.");
    (a = ls()) ? a.disconnect(1) : (cc("yt-remote-before-disconnect", 1), cc("yt-remote-connection-change", !1));
    ns(null)
}
function os() {
    var a = ls();
    return a && 3 != a.getProxyState() ? new Kq(a, void 0) : null
}
function as(a) {
    Vn("remote", a)
}
function ps() {
    return r("yt.mdx.remote.screenService_")
}
function Yr() {
    if (!qs) {
        var a = ps();
        a && (qs = new Rr(a))
    }
    return qs
}
function Zr(a) {
    var b=!!yo("yt-remote-load-account-screens");
    if (a) {
        var c=!!a.isSignedIn;
        c ? (b|=!!a.loadAccountScreens, kp(b)) : kp(!1);
        return c && b
    }
    return b
}
function $r() {
    return r("yt.mdx.remote.currentScreenId_")
}
function rs(a) {
    q("yt.mdx.remote.currentScreenId_", a, void 0);
    if (Pr) {
        var b = Pr;
        b.A = w() + 3E5;
        if ((b.ec = a) && (a = Mr(b, a))&&!To(b.g, a)) {
            var c = bb(b.g);
            c.push(a);
            Er(b, c, !0)
        }
    }
}
function ms() {
    q("yt.mdx.remote.currentDialId_", "", void 0)
}
function cs() {
    q("yt.mdx.remote.connectData_", null, void 0)
}
function ls() {
    return r("yt.mdx.remote.connection_")
}
function ns(a) {
    var b = ls();
    cs();
    a ? Ja(!ls()) : (rs(""), ms());
    q("yt.mdx.remote.connection_", a, void 0);
    Wr && (z(Wr, function(b) {
        b(a)
    }), Wr.length = 0);
    b&&!a ? cc("yt-remote-connection-change", !1) : !b && a && K("yt-remote-connection-change", !0)
}
function es() {
    var a = cp();
    if (!a)
        return null;
    if (ps()) {
        var b = Yr().da();
        return Uo(b, a)
    }
    return Pr ? Mr(Pr, a) : null
}
function ks(a, b) {
    Ja(!$r());
    rs(a.id);
    var c = new Y(Tr, a, fs());
    c.connect(b, r("yt.mdx.remote.connectData_"));
    c.subscribe("beforeDisconnect", function(a) {
        cc("yt-remote-before-disconnect", a)
    });
    c.subscribe("beforeDispose", function() {
        ls() && (ls(), ns(null))
    });
    ns(c)
}
function hs() {
    var a = es();
    a ? (as("Resume connection to: " + eo(a)), ks(a, 0)) : (qp(), uq(), as("Skipping connecting because no session screen found."))
}
function Ur(a) {
    as("Paired with: " + eo(a));
    a ? ks(a, 1) : ns(null)
}
function Vr() {
    var a = $r();
    a&&!is() && (as("Dropping current screen with id: " + a), js());
    es() || qp()
}
var Tr = null, Wr = null, qs = null, Pr = null;
function Xr(a) {
    var b = fs();
    if (wb(b)) {
        var b = bp(), c = yo("yt-remote-session-name") || "", d = yo("yt-remote-session-app") || "", b = {
            device: "REMOTE_CONTROL",
            id: b,
            name: c,
            app: d
        };
        a && (b["mdx-version"] = 3);
        q("yt.mdx.remote.channelParams_", b, void 0)
    }
}
function fs() {
    return r("yt.mdx.remote.channelParams_") || {}
}
function gs(a) {
    a ? (wo("yt-remote-session-app", a.app), wo("yt-remote-session-name", a.name)) : (zo("yt-remote-session-app"), zo("yt-remote-session-name"));
    q("yt.mdx.remote.channelParams_", a, void 0)
}
var bs = [];
function ss(a) {
    fg.call(this);
    this.D = {
        Zb: a.Zb,
        $b: u(function() {
            di.getInstance().hide(this.wa());
            fa(a.$b) && a.$b()
        }, this)
    };
    this.B = this.k = null
}
y(ss, fg);
ss.prototype.Ja = function() {
    ss.J.Ja.call(this);
    this.k = O(E(this.wa(), "dialog-id"));
    if (!this.k)
        throw Error("Cannot find create playlist widget dialog.");
    cg(this).listen(document, "click", this.A);
    var a = J("yt-uix-card-show", u(this.C, this));
    this.Ib.push(a);
    a = Q("create-playlist-widget-form", this.k);
    this.B = new aj(this.D);
    dg(this.B, a)
};
ss.prototype.Da = function() {
    this.k = null;
    this.B.dispose();
    this.B = null;
    ss.J.Da.call(this)
};
ss.prototype.A = function(a) {
    rd(this.k, oe(a)) || di.getInstance().hide(this.wa())
};
ss.prototype.C = function(a) {
    a == this.wa() && bg(this.B, "title-input").focus()
};
function ts() {
    this.g = {}
}
y(ts, vg);
ba(ts);
ts.prototype.Ta = "scroller";
ts.prototype.dispose = function() {
    for (var a in this.g) {
        var b = this.g[a], c = this.T(b, "scroller-mousewheel-listener") || "";
        Eb(b, "scroller-mousewheel-listener", "");
        var d = this.T(b, "scroller-scroll-listener") || "";
        Eb(b, "scroller-scroll-listener", "");
        le(c);
        le(d);
        Eb(b, "scroller-scroll-listener", "");
        b && (b = ha(b).toString(), delete this.g[b])
    }
    this.g = {}
};
function us(a) {
    a = a || {};
    this.url = a.url || "";
    this.urlV8 = a.url_v8 || "";
    this.urlV9As2 = a.url_v9as2 || "";
    this.args = a.args || yb(vs);
    this.assets = a.assets || {};
    this.attrs = a.attrs || yb(ws);
    this.params = a.params || yb(xs);
    this.minVersion = a.min_version || "8.0.0";
    this.fallback = a.fallback || null;
    this.fallbackMessage = a.fallbackMessage || null;
    this.html5=!!a.html5;
    this.disable = a.disable || {};
    this.loaded=!!a.loaded;
    this.messages = a.messages || {}
}
var vs = {
    enablejsapi: 1
}, ws = {}, xs = {
    allowscriptaccess: "always",
    allowfullscreen: "true",
    bgcolor: "#000000"
};
function ys(a) {
    a instanceof us || (a = new us(a));
    return a
}
us.prototype.clone = function() {
    var a = new us, b;
    for (b in this) {
        var c = this[b];
        "object" == ca(c) ? a[b] = yb(c) : a[b] = c
    }
    return a
};
function zs() {
    return !1
}
function As() {
    return null
};
function Bs(a) {
    this.j = a;
    this.g = null;
    var b = this;
    a.then(function(a) {
        b.g = a
    })
}
function Cs(a, b) {
    var c = b || {}, d = Ds().then(function(b) {
        return b(a, {
            apiKey: c.jf || c.apiKey,
            environment: c.kf || c.environment,
            helpCenterPath: c.lf || c.helpCenterPath,
            locale: c.locale || c.locale,
            productData: c.mf || c.productData,
            receiverUri: c.nf || c.receiverUri,
            theme: c.theme || c.theme,
            window: c.window || c.window
        })
    });
    return new Bs(d)
}
var Es = null;
function Ds() {
    if (Es)
        return Es;
    var a = r("help.service.Lazy.create");
    return a ? Es = Ng(a) : Es = (new Lg(function(a, c) {
        var d = document.createElement("script");
        d.async=!0;
        d.src = "https://www.gstatic.com/inproduct_help/service/lazy.min.js";
        d.onload = d.onreadystatechange = function() {
            d.readyState && "loaded" != d.readyState && "complete" != d.readyState || a(null)
        };
        d.onerror = c;
        (document.head || document.getElementsByTagName("head")[0]).appendChild(d)
    })).then(function() {
        var a = r("help.service.Lazy.create");
        if (!a)
            throw Error("Failed to load help.service.Lazy.create from https://www.gstatic.com/inproduct_help/service/lazy.min.js");
        return a
    })
}
Bs.prototype.k = function(a) {
    Fs(this, "startFeedback", arguments)
};
Bs.prototype.B = function(a) {
    Fs(this, "startHelp", arguments)
};
function Fs(a, b, c) {
    a.j.then(function(a) {
        a[b].apply(a, c)
    })
};
var Gs=!1;
function Hs(a) {
    if (a = a.match(/[\d]+/g))
        a.length = 3, a.join(".")
}
if (navigator.plugins && navigator.plugins.length) {
    var Is = navigator.plugins["Shockwave Flash"];
    Is && (Gs=!0, Is.description && Hs(Is.description));
    navigator.plugins["Shockwave Flash 2.0"] && (Gs=!0)
} else if (navigator.mimeTypes && navigator.mimeTypes.length) {
    var Js = navigator.mimeTypes["application/x-shockwave-flash"];
    (Gs = Js && Js.enabledPlugin) && Hs(Js.enabledPlugin.description)
} else 
    try {
        var Ks = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"), Gs=!0;
        Hs(Ks.GetVariable("$version"))
} catch (Ls) {
    try {
        Ks = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"),
        Gs=!0
    } catch (Ms) {
        try {
            Ks = new ActiveXObject("ShockwaveFlash.ShockwaveFlash"), Gs=!0, Hs(Ks.GetVariable("$version"))
        } catch (Ns) {}
    }
};
function Os(a) {
    return (a = a.exec(vc)) ? a[1] : ""
}(function() {
    if (pj)
        return Os(/Firefox\/([0-9.]+)/);
    if (N || yc)
        return Gc;
    if (uj)
        return Os(/Chrome\/([0-9.]+)/);
    if (vj)
        return Os(/Version\/([0-9.]+)/);
    if (rj || sj) {
        var a;
        if (a = /Version\/(\S+).*Mobile\/(\S+)/.exec(vc))
            return a[1] + "." + a[2]
    } else {
        if (tj)
            return (a = Os(/Android\s+([0-9.]+)/)) ? a : Os(/Version\/([0-9.]+)/);
        if (qj)
            return Os(/Camino\/([0-9.]+)/)
    }
    return ""
})();
function Ps() {
    this.k = this.j = this.g = 0;
    this.B = "";
    var a = r("window.navigator.plugins"), b = r("window.navigator.mimeTypes"), a = a && a["Shockwave Flash"], b = b && b["application/x-shockwave-flash"], b = a && b && b.enabledPlugin && a.description || "";
    if (a = b) {
        var c = a.indexOf("Shockwave Flash");
        0 <= c && (a = a.substr(c + 15));
        for (var c = a.split(" "), d = "", a = "", e = 0, f = c.length; e < f; e++)
            if (d)
                if (a)
                    break;
                else 
                    a = c[e];
        else 
            d = c[e];
        d = d.split(".");
        c = parseInt(d[0], 10) || 0;
        d = parseInt(d[1], 10) || 0;
        e = 0;
        if ("r" == a.charAt(0) || "d" == a.charAt(0))
            e = parseInt(a.substr(1),
            10) || 0;
        a = [c, d, e]
    } else 
        a = [0, 0, 0];
    this.B = b;
    b = a;
    this.g = b[0];
    this.j = b[1];
    this.k = b[2];
    if (0 >= this.g) {
        var h, k, m, p;
        if (Vb)
            try {
                h = new ActiveXObject("ShockwaveFlash.ShockwaveFlash")
            } catch (x) {
            h = null
        } else 
            m = document.body, p = document.createElement("object"), p.setAttribute("type", "application/x-shockwave-flash"), h = m.appendChild(p);
        if (h && "GetVariable"in h)
            try {
                k = h.GetVariable("$version")
            } catch (P) {
            k = ""
        }
        m && p && m.removeChild(p);
        (h = k || "") ? (h = h.split(" ")[1].split(","), h = [parseInt(h[0], 10) || 0, parseInt(h[1], 10) || 0, parseInt(h[2],
        10) || 0]) : h = [0, 0, 0];
        this.g = h[0];
        this.j = h[1];
        this.k = h[2]
    }
}
ba(Ps);
Ps.prototype.getVersion = function() {
    return [this.g, this.j, this.k]
};
function Qs(a, b, c, d) {
    b = "string" == typeof b ? b.split(".") : [b, c, d];
    b[0] = parseInt(b[0], 10) || 0;
    b[1] = parseInt(b[1], 10) || 0;
    b[2] = parseInt(b[2], 10) || 0;
    return a.g > b[0] || a.g == b[0] && a.j > b[1] || a.g == b[0] && a.j == b[1] && a.k >= b[2]
}
function Rs(a) {
    return - 1 < a.B.indexOf("Gnash")&&-1 == a.B.indexOf("AVM2") || 9 == a.g && 1 == a.j || 9 == a.g && 0 == a.j && 1 == a.k?!1 : 9 <= a.g
}
function Ss(a) {
    return Cc?!Qs(a, 11, 2) : Bc?!Qs(a, 11, 3) : !Rs(a)
};
var Ts = {};
function Us(a, b) {
    var c = F("FEEDBACK_LOCALE_LANGUAGE"), d = F("FEEDBACK_LOCALE_EXTRAS", {});
    a ? Cb(Ts, a) : Cb(Ts, d);
    try {
        var e, f = r("yt.player.getPlayerByElement");
        (e = f ? f("player-api") : null) && e.pauseVideo && e.pauseVideo();
        var h = Ps.getInstance();
        Ts.flashVersion = h.getVersion().join(".");
        e && (Ts.playback_id = e.getVideoData().cpn)
    } catch (k) {}
    b && Cb(Ts, {
        trackingParam: b
    });
    return {
        helpCenterPath: "/youtube",
        locale: c,
        productData: Ts
    }
}
function Vs() {
    var a = F("SESSION_INDEX"), b = F("FEEDBACK_BUCKET_ID"), c = {
        abuseLink: "https://support.google.com/youtube/bin/answer.py?answer=140536",
        customZIndex: "2000000005"
    };
    a && (c.authuser = a + "");
    b && (c.bucket = b);
    return c
}
function Ws(a, b) {
    try {
        var c = (a || "59") + "", d = Us(b), e = Vs();
        Cs(c, d).k(e);
        return !1
    } catch (f) {
        return !0
    }
}
function Xs(a, b, c, d) {
    var e;
    d = (d || "59") + "";
    c = Us(c, void 0);
    a = {
        context: b,
        anchor: void 0,
        enableSendFeedback: !0,
        defaultHelpArticleId: a
    };
    Cb(a, Vs());
    try {
        Cs(d, c).B(a), e=!1
    } catch (f) {
        e=!0
    }
    return e
};
function Ys(a, b, c) {
    if (b) {
        a = t(a) ? Tc(a) : a;
        c = ys(c);
        var d = yb(c.attrs);
        d.tabindex = 0;
        var e = yb(c.params);
        e.flashvars = Ue(c.args);
        if (Vb) {
            d.classid = "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000";
            e.movie = b;
            b = document.createElement("object");
            for (var f in d)
                b.setAttribute(f, d[f]);
            for (f in e)
                d = document.createElement("param"), d.setAttribute("name", f), d.setAttribute("value", e[f]), b.appendChild(d)
            } else {
            d.type = "application/x-shockwave-flash";
            d.src = b;
            b = document.createElement("embed");
            b.setAttribute("name", d.id);
            for (f in d)
                b.setAttribute(f,
                d[f]);
            for (f in e)
                b.setAttribute(f, e[f])
            }
        e = document.createElement("div");
        e.appendChild(b);
        a.innerHTML = e.innerHTML
    }
}
function Zs(a, b, c) {
    if (a && a.attrs && a.attrs.id) {
        a = ys(a);
        var d=!!b, e = O(a.attrs.id), f = e ? e.parentNode : null;
        if (e && f) {
            if (window != window.top) {
                var h = null;
                if (document.referrer) {
                    var k = document.referrer.substring(0, 128);
                    $e(k) || (h = k)
                } else 
                    h = "unknown";
                h && (d=!0, a.args.framer = h)
            }
            h = $s();
            if (Qs(h, a.minVersion)) {
                var k = at(a, h), m = "";
                - 1 < navigator.userAgent.indexOf("Sony/COM2") || (m = e.getAttribute("src") || e.movie);
                (m != k || d) && Ys(f, k, a);
                Ss(h) && bt()
            } else 
                ct(f, a, h);
            c && c()
        } else 
            G(function() {
                Zs(a, b, c)
            }, 50)
    }
}
function ct(a, b, c) {
    0 == c.g && b.fallback ? b.fallback() : 0 == c.g && b.fallbackMessage ? b.fallbackMessage() : a.innerHTML = '<div id="flash-upgrade">' + Tb() + "</div>"
}
function at(a, b) {
    return Rs(b) && a.url || ( - 1 < navigator.userAgent.indexOf("Sony/COM2")&&!Qs(b, 9, 1, 58)?!1 : !0) && a.urlV9As2 || a.urlV8 || a.url
}
function $s() {
    var a = Ps.getInstance();
    ih.getInstance().set("fv", a.getVersion().join("."));
    fh("PREF", ph(), 63072E3);
    return a
}
function bt() {
    var a = O("flash10-promo-div"), b = nh(ih.getInstance(), 107);
    a&&!b && of(a)
};
var dt;
var et = vc, et = et.toLowerCase();
if (Da(et, "android")) {
    var ft = et.match(/android\D*(\d\.\d)[^\;|\)]*[\;\)]/);
    if (ft)
        dt = Number(ft[1]);
    else {
        var gt = {
            cupcake: 1.5,
            donut: 1.6,
            eclair: 2,
            froyo: 2.2,
            gingerbread: 2.3,
            honeycomb: 3,
            "ice cream sandwich": 4,
            jellybean: 4.1
        }, ht = et.match("(" + tb(gt).join("|") + ")");
        dt = ht ? gt[ht[0]] : 0
    }
} else 
    dt = void 0;
function it() {
    if (2.2 == dt)
        return !0;
    var a;
    a = r("yt.player.utils.videoElement_");
    a || (a = document.createElement("video"), q("yt.player.utils.videoElement_", a, void 0));
    try {
        return !(!a ||!a.canPlayType ||!a.canPlayType('video/mp4; codecs="avc1.42001E, mp4a.40.2"')&&!a.canPlayType('video/webm; codecs="vp8.0, vorbis"'))
    } catch (b) {
        return !1
    }
};
function jt() {
    var a = {
        volume: 100,
        muted: !1
    }, b = yo("yt-player-volume") || {};
    a.volume = isNaN(b.volume) ? 100 : Math.min(Math.max(b.volume, 0), 100);
    a.muted = void 0 == b.muted?!1 : b.muted;
    return a
};
function kt(a, b) {
    Ib.call(this);
    this.k = this.tb = a;
    this.O = b;
    this.C=!1;
    this.g = {};
    this.Z = this.H = null;
    this.P = new Lb;
    Jb(this, v(Kb, this.P));
    this.B = {};
    this.A = this.na = this.j = this.Aa = this.K = null;
    this.S=!1;
    this.ka = this.D = null;
    this.ba = {};
    this.Ib = ["onReady"];
    this.Ba = null;
    this.rb = 0;
    this.X = {};
    lt(this);
    this.hb("onVolumeChange", u(this.ud, this));
    this.hb("onError", u(this.sd, this));
    this.hb("onTabOrderChange", u(this.rd, this));
    this.hb("WATCH_LATER_VIDEO_ADDED", u(this.vd, this));
    this.hb("WATCH_LATER_VIDEO_REMOVED", u(this.wd,
    this))
}
y(kt, Ib);
g = kt.prototype;
g.getId = function() {
    return this.O
};
g.nc = function(a, b) {
    this.L() || (mt(this, a), nt(this, b), this.C && ot(this))
};
function mt(a, b) {
    a.Aa = b;
    a.K = b.clone();
    a.j = a.K.attrs.id || a.j;
    "video-player" == a.j && (a.j = a.O, a.K.attrs.id = a.O);
    a.k.id == a.j && (a.j = a.j + "-player", a.K.attrs.id = a.j);
    var c = a.K.args, d;
    if (!(d = a.K.args.eurl)) {
        d = document.location.toString();
        var e=!!d&&-1 != d.search(Qh), f =- 1 != d.indexOf("/embed/");
        e != f && Sb(Error("isIframeEmbed(" + d + ") behavior is not consistent"));
        d = f ? document.referrer && document.referrer.substring(0, 128) || "unknown" : d
    }
    c.eurl = d;
    a.K.args.enablejsapi = "1";
    a.K.args.playerapiid = a.O;
    a.na || (a.na = pt(a, a.K.args.jsapicallback ||
    "onYouTubePlayerReady"));
    a.K.args.jsapicallback = null;
    if (c = a.K.attrs.width)
        a.k.style.width = Pd(Number(c) || c, !0);
    if (c = a.K.attrs.height)
        a.k.style.height = Pd(Number(c) || c, !0);
    a.k.style.overflow = "hidden"
}
g.ie = function() {
    return this.Aa
};
function ot(a) {
    a.K.loaded || (a.K.loaded=!0, "0" != a.K.args.autoplay ? a.g.loadVideoByPlayerVars(a.K.args) : a.g.cueVideoByPlayerVars(a.K.args))
}
function qt(a) {
    if (!n(a.K.disable.flash)) {
        var b = a.K.disable, c;
        c = Qs($s(), a.K.minVersion);
        b.flash=!c
    }
    return !a.K.disable.flash
}
function rt(a) {
    var b = st(a);
    b && b.stopVideo && b.stopVideo();
    if (qt(a)) {
        var c = a.K;
        b && b.getUpdatedConfigurationData && (b = ys(b.getUpdatedConfigurationData()), b.args.video_id == c.args.video_id && (c = b));
        c.args.autoplay = 1;
        c.args.html5_unavailable = "1";
        mt(a, c);
        nt(a, "flash")
    }
}
function nt(a, b) {
    if (!a.L()) {
        if (!b) {
            var c;
            if (!(c=!a.K.html5 && qt(a))) {
                if (!n(a.K.disable.html5)) {
                    if (c = it())
                        c = tt(a) || a.K.assets.js;
                    a.K.disable.html5=!c;
                    c || (a.K.args.html5_unavailable = "1")
                }
                c=!!a.K.disable.html5
            }
            b = c ? qt(a) ? "flash" : "unsupported" : "html5"
        }("flash" == b ? a.Ne : "html5" == b ? a.Oe : a.Pe).call(a)
    }
}
function tt(a) {
    var b=!0, c = st(a);
    c && a.K && (a = a.K, b = E(c, "version") == a.assets.js);
    return b&&!!r("yt.player.Application.create")
}
g.Oe = function() {
    if (!this.S) {
        var a = tt(this);
        a && "html5" == ut(this) ? (this.A = "html5", this.C || this.kb()) : (vt(this), this.A = "html5", a && this.ka ? (this.tb.appendChild(this.ka), this.kb()) : (this.K.loaded=!0, this.D = u(function() {
            var a = this.tb, c = this.K.clone();
            r("yt.player.Application.create")(a, c);
            this.kb()
        }, this), this.S=!0, a ? this.D() : (fc(this.K.assets.js, this.D), nc(this.K.assets.css))))
    }
};
g.Ne = function() {
    var a = this.K.clone();
    a.attrs.width = a.attrs.width || "100%";
    a.attrs.height = a.attrs.height || "100%";
    if ("flash" == ut(this))
        this.A = "flash", this.C || Zs(a, !1, u(this.kb, this));
    else {
        vt(this);
        this.A = "flash";
        this.K.loaded=!0;
        var b = this.tb, b = t(b) ? Tc(b): b, a = ys(a);
        if (window != window.top) {
            var c = null;
            document.referrer && (c = document.referrer.substring(0, 128));
            a.args.framer = c
        }
        c = $s();
        Qs(c, a.minVersion) ? (c = at(a, c), Ys(b, c, a)) : ct(b, a, c);
        this.kb()
    }
};
function st(a) {
    var b = O(a.j);
    !b && a.k && a.k.querySelector && (b = a.k.querySelector("#" + a.j));
    return b
}
g.kb = function() {
    var a = st(this), b=!1;
    try {
        a && a.getApiInterface && a.getApiInterface() && (b=!0)
    } catch (c) {}
    if (b)
        if (this.S=!1, a.isNotServable && a.isNotServable())
            rt(this);
        else {
            lt(this);
            this.C=!0;
            a = st(this);
            a.addEventListener && (this.H = wt(this, a, "addEventListener"));
            a.removeEventListener && (this.Z = wt(this, a, "removeEventListener"));
            for (var b = a.getApiInterface(), b = b.concat(a.getInternalApiInterface()), d = 0; d < b.length; d++) {
                var e = b[d];
                this.g[e] || (this.g[e] = wt(this, a, e))
            }
            for (var f in this.B)
                this.H(f, this.B[f]);
                ot(this);
                this.na && this.na(this.g);
                this.P.publish("onReady", this.g)
        } else 
            this.rb = G(u(this.kb, this), 50)
    };
function wt(a, b, c) {
    var d = b[c];
    return function() {
        try {
            return a.Ba = null, d.apply(b, arguments)
        } catch (e) {
            "Bad NPObject as private data!" != e.message && (e.message += " (" + c + ")", a.Ba = e, Sb(e, "WARNING"))
        }
    }
}
function lt(a) {
    a.C=!1;
    if (a.Z)
        for (var b in a.B)
            a.Z(b, a.B[b]);
    for (var c in a.X)
        H(parseInt(c, 10));
    a.X = {};
    a.H = null;
    a.Z = null;
    for (var d in a.g)
        a.g[d] = null;
    a.g.addEventListener = u(a.hb, a);
    a.g.removeEventListener = u(a.le, a);
    a.g.destroy = u(a.dispose, a);
    a.g.getLastError = u(a.je, a);
    a.g.getPlayerType = u(a.ke, a);
    a.g.getCurrentVideoConfig = u(a.ie, a);
    a.g.loadNewVideoConfig = u(a.nc, a);
    a.g.isReady = u(a.ne, a)
}
g.ne = function() {
    return this.C
};
g.hb = function(a, b) {
    if (!this.L()) {
        var c = pt(this, b);
        if (c) {
            if (!Ta(this.Ib, a)&&!this.B[a]) {
                var d = xt(this, a);
                this.H && this.H(a, d)
            }
            this.P.subscribe(a, c);
            "onReady" == a && this.C && G(v(c, this.g), 0)
        }
    }
};
g.le = function(a, b) {
    if (!this.L()) {
        var c = pt(this, b);
        c && this.P.unsubscribe(a, c)
    }
};
function pt(a, b) {
    var c = b;
    if ("string" == typeof b) {
        if (a.ba[b])
            return a.ba[b];
        c = function() {
            var a = r(b);
            a && a.apply(l, arguments)
        };
        a.ba[b] = c
    }
    return c ? c : null
}
function xt(a, b) {
    var c = "ytPlayer" + b + a.O;
    a.B[b] = c;
    l[c] = function(c) {
        var e = G(function() {
            if (!a.L()) {
                a.P.publish(b, c);
                var f = a.X, h = e.toString();
                h in f && delete f[h]
            }
        }, 0);
        xb(a.X, e.toString())
    };
    return c
}
g.rd = function(a) {
    a = a ? pd : od;
    for (var b = a(document.activeElement); b && (1 != b.nodeType || (b.focus(), b != document.activeElement));)
        b = a(b)
};
g.ud = function(a) {
    var b = {};
    b.volume = isNaN(a.volume) ? jt().volume : Math.min(Math.max(a.volume, 0), 100);
    b.muted = void 0 == a.muted ? jt().muted : a.muted;
    wo("yt-player-volume", b, 2592E3)
};
g.sd = function(a) {
    5 == a && rt(this)
};
g.vd = function(a) {
    K("WATCH_LATER_VIDEO_ADDED", a)
};
g.wd = function(a) {
    K("WATCH_LATER_VIDEO_REMOVED", a)
};
g.Pe = function() {
    vt(this);
    this.A = "unsupported";
    var a = 'Adobe Flash Player or an HTML5 supported browser is required for video playback. <br> <a href="http://get.adobe.com/flashplayer/">Get the latest Flash Player</a> <br> <a href="/html5">Learn more about upgrading to an HTML5 browser</a>', b = navigator.userAgent.match(/Version\/(\d).*Safari/);
    b && 5 <= parseInt(b[1], 10) && (a = 'Adobe Flash Player or QuickTime is required for video playback. <br> <a href="http://get.adobe.com/flashplayer/"> Get the latest Flash Player</a> <br> <a href="http://www.apple.com/quicktime/download/">Get the latest version of QuickTime</a>');
    b = this.K.messages.player_fallback || a;
    a = O("player-unavailable");
    if (O("unavailable-submessage") && a) {
        O("unavailable-submessage").innerHTML = b;
        var b = Q("icon", a), c;
        if (c = b)
            c = b ? b.dataset ? Fb("icon")in b.dataset : b.hasAttribute?!!b.hasAttribute("data-icon") : !!b.getAttribute("data-icon") : !1;
        c && (b.src = E(b, "icon"));
        D(a, "hid");
        C(O("player"), "off-screen-trigger")
    }
};
g.ke = function() {
    return this.A || ut(this)
};
g.je = function() {
    return this.Ba
};
function ut(a) {
    return (a = st(a)) ? "div" == a.tagName.toLowerCase() ? "html5" : "flash" : null
}
function vt(a) {
    var b;
    (b = r("ytcsi.data_")) || (b = {
        tick: {},
        span: {},
        info: {}
    }, q("ytcsi.data_", b, void 0));
    b.tick.dcp = w();
    Wh("dcp");
    a.cancel();
    lt(a);
    a.A = null;
    a.K && (a.K.loaded=!1);
    b = st(a);
    "html5" == ut(a) ? a.ka = b : b && b.destroy && b.destroy();
    hd(a.tb)
}
g.cancel = function() {
    this.D && mc(this.K.assets.js, this.D);
    H(this.rb);
    this.S=!1
};
g.G = function() {
    vt(this);
    this.ba = null;
    for (var a in this.B)
        l[this.B[a]] = null;
    this.g = null;
    delete this.tb;
    delete this.k;
    this.K && (this.Aa = this.K = this.K.fallback = null);
    kt.J.G.call(this)
};
var zt = {}, At = "player_uid_" + (1E9 * Math.random()>>>0);
function Bt(a, b) {
    a = t(a) ? Tc(a) : a;
    b = ys(b);
    var c = At + "_" + ha(a), d = zt[c];
    if (d)
        return d.nc(b), d.g;
    d = new kt(a, c);
    zt[c] = d;
    K("player-added", d.g);
    Jb(d, v(Ct, d));
    G(function() {
        d.nc(b)
    }, 0);
    return d.g
}
function Dt() {
    for (var a in zt) {
        var b = zt[a];
        b && b.cancel()
    }
}
function Et(a) {
    if (a = O(a))
        a = At + "_" + ha(a), (a = zt[a]) && a.dispose()
}
function Ct(a) {
    zt[a.getId()] = null
}
function Ft(a) {
    a = O(a);
    if (!a)
        return null;
    var b = At + "_" + ha(a), c = zt[b];
    c || (c = new kt(a, b), zt[b] = c);
    return c.g
};
var Gt = r("yt.abuse.botguardInitialized") || zs;
q("yt.abuse.botguardInitialized", Gt, void 0);
var Ht = r("yt.abuse.invokeBotguard") || As;
q("yt.abuse.invokeBotguard", Ht, void 0);
var It = r("yt.player.exports.navigate") || xi;
q("yt.player.exports.navigate", It, void 0);
var Jt = r("yt.player.embed") || Bt;
q("yt.player.embed", Jt, void 0);
var Kt = r("yt.player.destroy") || Et;
q("yt.player.destroy", Kt, void 0);
var Lt = r("yt.player.cancelAll") || Dt;
q("yt.player.cancelAll", Lt, void 0);
var Mt = r("yt.player.getPlayerByElement") || Ft;
q("yt.player.getPlayerByElement", Mt, void 0);
var Nt = r("yt.player.exports.feedbackStart") || Ws;
q("yt.player.exports.feedbackStart", Nt, void 0);
var Ot = r("yt.player.exports.feedbackShowArticle") || Xs;
q("yt.player.exports.feedbackShowArticle", Ot, void 0);
var Pt = r("yt.util.activity.init") || gl;
q("yt.util.activity.init", Pt, void 0);
var Qt = r("yt.util.activity.getTimeSinceActive") || il;
q("yt.util.activity.getTimeSinceActive", Qt, void 0);
var Rt = r("yt.util.activity.setTimestamp") || hl;
q("yt.util.activity.setTimestamp", Rt, void 0);
var St=!1;
function Tt() {
    var a = r("yt.player.getPlayerByElement");
    return a ? a("player-api") : null
}
function Ut(a) {
    mb(O("player-mole-container"), "watch-mole", a);
    a = O("player");
    var b = O("player-mole-container");
    St=!a || B(a, "off-screen") ||!b || B(b, "watch-mole");
    (a = Tt()) && a.isReady() && a.setMinimized(St)
};
var Vt, Wt, Z, Xt, Yt, Zt, $t, au, bu, cu, du, eu, fu, gu, hu, $, iu;
var ju;
function ku(a, b) {
    this.type = a;
    this.videoIds = b || []
}
function lu(a) {
    t(a) && (a = [a]);
    var b = new ku(1, a);
    ju ? mu(function() {
        z(a, function(a) {
            ju.bd(a)
        })
    }, b, void 0) : nu(b, a)
}
function ou(a, b, c) {
    var d = new ku(2);
    ju && mu(function() {
        ju.qd(a, b, c)
    }, d, void 0)
}
function pu(a) {
    var b = new ku(2);
    ju && mu(function() {
        ju.pd(a)
    }, b, void 0)
}
function qu(a) {
    var b = new ku(2);
    ju && mu(function() {
        ju.ld(a)
    }, b, void 0)
}
function ru(a, b, c) {
    ju && mu(function() {
        b && Uq(ju, b, 0, 0, a)
    }, void 0, c)
}
function su(a, b, c) {
    ju && mu(function() {
        var c = ju;
        Uq(c, a);
        Pq(c, "updatePlaylist", {
            videoIds: b.join(",")
        })
    }, void 0, c)
}
function tu() {
    var a = new ku(2);
    ju && mu(function() {
        ju.stop();
        ju.nd()
    }, a, void 0)
}
function uu(a) {
    ju = a;
    ju.subscribe("remoteQueueChange", function() {
        K("queue-change", new ku(2))
    })
}
function vu() {
    var a = os();
    Kb(ju);
    ju = null;
    a ? uu(a) : K("queue-change", new ku(2))
}
function mu(a, b, c) {
    ju && 1 == ju.getState() && (a.call(l), c && G(function() {
        c()
    }, 0), n(b) && K("queue-change", b))
}
function nu(a, b) {
    ff("/watch_queue_ajax", {
        method: "POST",
        $a: {
            action_remove_from_watch_queue: 1
        },
        ca: {
            session_token: Pb(),
            list: "WQ",
            video_ids: b.join(",")
        },
        V: function() {
            K("queue-change", a)
        },
        onError: function() {}
    })
}
var wu = [];
var xu;
function yu() {
    if (!xu) {
        var a = O("watch-queue");
        if (!a)
            return [];
        xu = Q("watch-queue-items-list", a)
    }
    var b = [], a = ld(xu);
    z(a, function(a) {
        (a = E(a, "video-id")) && b.push(a)
    });
    return b
};
function zu() {
    K("yt-dom-content-change", $)
}
q("yt.www.watchqueue.loadThumbnails", zu, void 0);
function Au() {
    iu = O("watch-queue");
    bu = Q("watch-queue-control-bar", iu);
    cu = Q("watch-queue-count", bu);
    du = Q("yt-uix-button-dark-overflow-action-menu", bu);
    eu = Q("prev-watch-queue-button", iu);
    fu = Q("next-watch-queue-button", iu);
    gu = Q("play-watch-queue-button", iu);
    hu = Q("pause-watch-queue-button", iu);
    $ = Q("watch-queue-items-list", iu);
    Bu.push(me(iu, "click", Cu, "yt-uix-button"));
    Bu.push(me(iu, "click", Du, "watch-queue-video"));
    Bu.push(me(iu, "click", Eu, "control-bar-button"));
    Bu.push(R($, xf, Fu));
    Bu.push(me($, xf, Gu, "watch-queue-item"));
    Bu.push(me(null, "click", Hu, "watch-queue-video-menu-choice"));
    Bu.push(me(null, "click", Iu, "watch-queue-menu-choice"));
    $t = new Vi(O("watch-queue-count-msg").innerHTML, ["count", "total"]);
    au = Yi(O("watch-queue-loading-template"));
    Ju = 0;
    Xt=!1;
    var a = O("watch-queue-save-as-playlist-widget");
    a && (Zt = new ss({
        Zb: Ku
    }), dg(Zt, a));
    Wt = null;
    Sr(!!F("MDX_ENABLE_CASTV2"), !0, {
        device: "Desktop",
        app: "youtube-desktop",
        isSignedIn: F("LOGGED_IN")
    });
    Sr(!!F("MDX_ENABLE_CASTV2"), !!F("MDX_ENABLE_QUEUE"));
    wu.push(J("yt-remote-connection-change",
    vu));
    (a = os()) && uu(a);
    Lu.push(J("init", Mu));
    Lu.push(J("dispose", Nu));
    Lu.push(J("yt-remote-before-disconnect", Ou));
    Lu.push(J("yt-remote-connection-change", Pu));
    Pu();
    Mu();
    Lu.push(J("queue-change", Qu));
    Lu.push(J("watch-queue-addto-video-added", Ru));
    Lu.push(J("watch-queue-addto-video-removed", Ru));
    Lu.push(J("watch-queue-addto-video-play-next", Su));
    Lu.push(J("watch-queue-addto-video-play-now", Tu));
    Lu.push(J("watch-queue-addto-playlist-added", Ru));
    Lu.push(J("watch-queue-addto-playlist-removed", Ru));
    Lu.push(J("watch-queue-addto-playlist-play-next",
    Uu));
    Lu.push(J("watch-queue-addto-playlist-play-now", Vu))
}
function Wu() {
    var a = O("watch7-sidebar");
    return a ? qd(iu) == a : !1
}
function Xu() {
    var a=!1;
    if (!Z ||!lf(Vt) ||!F("PAGE_NAME"))
        return Yu(), a;
    var b = Zu(), c = $u();
    av() && c && b != c && (bv(c), b = c);
    !av() ||!b || b != F("VIDEO_ID") && b != Yt || F("LIST_ID") ? qd(iu) != Vt && (Vt.appendChild(iu), K("watch-queue-show", 0), zu(), a=!0) : (b = O("watch7-sidebar"))&&!Wu() && (D(Vt, "mole-notification"), id(b, iu, 0), K("watch-queue-show", 1), cv(), a=!0);
    Yu();
    return a
}
function bv(a) {
    if (Wt) {
        var b=!!Z, c = b&&!(a && (a == F("VIDEO_ID") || a == Yt));
        Ut(c);
        b && av() && (c ? K("watch-queue-show", 0) : K("watch-queue-show", 1));
        if (Wt.isReady())
            Wt.loadVideoById(a);
        else {
            var d = function() {
                Wt.loadVideoById(a);
                Wt.removeEventListener("onReady", d)
            };
            Wt.addEventListener("onReady", d)
        }
    }
}
function Zu() {
    return Wt && Wt.isReady() ? Wt.getVideoData().video_id : ""
}
function $u() {
    return Z ? Nq(Z).videoId : ""
}
function dv(a) {
    Zu() != ev() && fv(Zu());
    gv(1 == a);
    Xu()
}
function Yu() {
    Wt && (Zu() ? Xt || Ut(!!Z && (lf(Vt) || hv()) && qd(iu) == Vt) : Ut(!1))
}
function Mu() {
    D(Vt, "mole-notification");
    Yt = "";
    iv();
    Xu();
    var a = Wt;
    a && a.addEventListener("onStateChange", dv)
}
function Nu() {
    ob(Vt, "mole-expanded", "mole-collapsed");
    var a = Wt;
    a && a.removeEventListener("onStateChange", dv)
}
function av() {
    return "watch" == F("PAGE_NAME")
}
function jv(a) {
    if (a != ev()) {
        var b = yu(), c = b[0];
        kv() && (c = b[lv() + 1]);
        if (a != c)
            if (Ta(b, a)) {
                var d = La(b, a), b = La(b, c);
                ou(a, d, b)
            } else 
                pu(a)
    }
}
function Cu(a) {
    if (a.currentTarget == eu) {
        if (hv()) {
            var b = ld($);
            a = lv();
            b = b.length;
            a = (b + a - 1)%b;
            mv(a);
            cv();
            (b = ev()) && Z && Z.jd(b, a)
        }
    } else if (a.currentTarget == fu)
        hv() && (a = ld($), a = (lv() + 1)%a.length, mv(a), cv(), (b = ev()) && Z && Z.kd(b, a));
    else if (a.currentTarget == gu)
        pf(gu), of(hu), Z ? (a = yu(), !$u() && a[0] ? Uq(Z, a[0]) : Z.play()) : Wt && Wt.playVideo();
    else if (a.currentTarget == hu)
        pf(hu), of(gu), Z ? Z.pause() : Wt && Wt.pauseVideo();
    else if (B(a.currentTarget, "remove-item-watch-queue-button")) {
        if (a = wd(a.currentTarget, "watch-queue-item"),
        b = E(a, "video-id"))
            C(a, "fade-out"), lu(b)
    } else 
        B(a.currentTarget, "expand-video-watch-queue-button") && nv(a.currentTarget)
}
function Iu(a) {
    var b = E(a.currentTarget, "action");
    ov[b](a.currentTarget);
    a = a || window.event;
    a.cancelBubble=!0;
    a.stopPropagation && a.stopPropagation();
    yg(wg.getInstance(), du)
}
function Eu(a) {
    qd(iu) == Vt && a.currentTarget == bu && (a = oe(a), a && B(a, "overflow-menu-choice") || (nb(Vt, "mole-expanded"), nb(Vt, "mole-collapsed")))
}
function Hu(a) {
    var b = E(a.currentTarget, "action");
    pv[b](a.currentTarget)
}
function Fu() {
    D(Vt, "mole-notification");
    B(Vt, "mole-expanded") && zu();
    yg(wg.getInstance(), du)
}
function Gu(a) {
    B(a.currentTarget, "fade-out") && (jd(a.currentTarget), fv($u()), qv());
    zu()
}
function Du(a) {
    a = wd(a.currentTarget, "watch-queue-item");
    if (!B(a, "currently-playing")) {
        var b = ld($), b = La(b, a);
        mv(b);
        (a = E(a, "video-id")) && Z && Uq(Z, a, 0, b)
    }
}
function Ku(a) {
    var b = a.playlistId, c = a.Ke;
    b && c && (a = {
        videoIds: yu().join(","),
        Qe: b,
        V: function() {
            wi(c)
        }
    }, ff("/playlist_video_ajax?action_add_to_playlist=1", {
        method: "POST",
        $a: {
            feature: a.ef || null,
            authuser: a.df || null,
            pageid: a.pageId || null
        },
        ca: {
            video_ids: a.videoIds || null,
            source_playlist_id: a.sourcePlaylistId || null,
            full_list_id: a.Qe || null,
            delete_from_playlists: a.ff || null,
            add_to_playlists: a.cf || null,
            plid: F("PLAYBACK_ID") || null
        },
        context: a.context,
        onError: a.onError,
        V: a.V,
        yc: a.yc
    }))
}
function nv(a) {
    B(a, "disabled") || (a = E(a, "video-id")) && rv(a)
}
function kv() {
    return !!Q("currently-playing", $)
}
function lv() {
    var a = Q("currently-playing", $);
    if (a) {
        var b = ld($);
        return La(b, a)
    }
    return 0
}
function ev() {
    var a = Q("currently-playing", $);
    return a ? E(a, "video-id") || "" : ""
}
function fv(a) {
    a = La(yu(), a);
    0 <= a ? kv() && lv() == a || (mv(a), cv()) : sv()
}
function mv(a) {
    if (hv()) {
        sv();
        var b = ld($);
        b[a] && C(b[a], "currently-playing");
        tv()
    }
}
function sv() {
    if (hv()) {
        var a = Q("currently-playing", $);
        a && D(a, "currently-playing");
        tv()
    }
}
function Su(a) {
    jv(a);
    Ru()
}
function Tu(a) {
    Z && Uq(Z, a);
    a != ev() && (bv(a), Ru());
    Yt = a
}
function Uu(a) {
    yu();
    kv() && lv();
    qu(a);
    Ru()
}
function Vu(a, b, c) {
    b && (c ? su(b, c, function() {
        Yt = b
    }) : ru(a, b, function() {
        Yt = b
    }), Ru())
}
function Ru() {
    lf(Vt) && C(Vt, "mole-notification")
}
function uv() {
    var a = kv() ? lv() + 1: 0, b = yu().length;
    cu.innerHTML = 0 < a && 0 < b ? $i($t, {
        count: a,
        total: b
    }) : ""
}
function gv(a) {
    a ? (pf(gu), of(hu)) : (pf(hu), of(gu))
}
function tv() {
    var a = yu().length, b = 1 >= a, c = kv() ? lv(): - 1;
    eu.disabled = b || 0 == c;
    fu.disabled = b || c == a - 1;
    uv();
    if (0 != a) {
        var d = null, a = ld($);
        z(a, function(a) {
            var b = E(a, "video-id"), c = Q("play-next", a), k = Q("goto-video-page", a);
            mb(k, "disabled", av() && b == F("VIDEO_ID")&&!F("LIST_ID"));
            B(a, "currently-playing") ? (d = a, pf(c)) : d ? (d = null, mb(c, "disabled", !0)) : mb(c, "disabled", !1)
        })
    }
}
function qv() {
    tv();
    Ua(yu()) ? vv() : wv();
    K("watch-queue-update")
}
function xv() {
    Ju = 0;
    H(yv);
    yv = NaN
}
function zv() {
    Xt=!1;
    if (isNaN(yv)) {
        var a = Math.min(3E4, 1E3 * Math.pow(2, Ju));
        yv = G(function() {
            yv = NaN;
            Ju++;
            Av()
        }, a)
    }
}
function hv() {
    var a = md($);
    return !!a && B(a, "watch-queue-item")
}
function Av() {
    if (!Xt) {
        Xt=!0;
        var a;
        ju ? (a = ls(), a = "RQ" + (a ? a.getLoungeToken() : "")) : a = "WQ";
        ff("/watch_queue_ajax", {
            method: "GET",
            $a: {
                action_get_watch_queue_items: 1,
                list: a
            },
            V: Bv,
            onError: zv,
            wb: zv
        });
        hv() ? wv() : ($.innerHTML = $i(au, {}), qv())
    }
}
function Bv(a, b) {
    Xt=!1;
    if ($ && b && (b.html ||!$u()) && (xv(), Z)) {
        var c = yu(), d = [], e = fd(b.html), f = Na(e.childNodes, function(a) {
            return ga(a) && 1 == a.nodeType
        });
        z(f, function(a) {
            a = E(a, "video-id");
            d.push(a)
        });
        0 == c.length && ($.innerHTML = "");
        if (0 == d.length)
            $.innerHTML = "", K("watch-queue-empty"), qv();
        else if (!fb(c, d)) {
            if (d.length == c.length + 1) {
                for (var h = 0; h < c.length && c[h] == d[h];)
                    h++;
                var k = bb(c);
                db(k, h, 0, d[h]);
                if (fb(k, d)) {
                    id($, f[h], h);
                    C(f[h], "fade-in");
                    G(function() {
                        D(f[h], "fade-in")
                    }, 0);
                    fv($u());
                    zu();
                    qv();
                    return 
                }
            }
            if (d.length ==
            c.length - 1) {
                for (e = 0; e < d.length && c[e] == d[e];)
                    e++;
                k = bb(c);
                Za(k, e);
                if (fb(k, d)) {
                    c = ld($);
                    C(c[e], "fade-out");
                    fv($u());
                    return 
                }
            }
            $.innerHTML = "";
            Ua(f) || (z(f, function(a) {
                $.appendChild(a)
            }), zu(), fv($u()));
            qv()
        }
    }
}
function cv() {
    var a = Q("currently-playing", $);
    if (a) {
        var b = ts.getInstance(), c = $;
        if (c && a) {
            var d = Uc(T(b, "scroll-unit"), c), a = La(d, a);
            if (0 <= a&&!(!c || isNaN(a) || 0 > a) && (b = Uc(T(b, "scroll-unit"), c), !(0 >= b.length))) {
                a >= b.length && (a = b.length - 1);
                if (b.length) {
                    var d = b[0], e = d.offsetHeight;
                    1 < b.length && (e = b[1].offsetTop - d.offsetTop);
                    d = e
                } else 
                    d = 0;
                var e = c.offsetHeight, f = Math.max(Math.floor(c.scrollTop / d), 0);
                a > f - 1 && (d = Math.floor(e / d), e = b.length, a + d > e && (a = e - d + 1));
                0 > a && (a = 0);
                a = b[a].offsetTop;
                c&&!isNaN(a) && (0 > a && (a = 0), c.scrollTop =
                a, Eb(c, "scroller-offset", a + ""), K("yt-dom-content-change", c))
            }
        }
    }
    zu()
}
function rv(a) {
    a = Ve("/watch", {
        v: a
    });
    wi(a)
}
function Cv(a) {
    av() && (a != F("VIDEO_ID") || F("LIST_ID") ? (G(function() {
        rv(a)
    }, 100), Yt = a) : Xu())
}
function Ou(a) {
    Dv = a
}
function Pu() {
    var a = os();
    Kb(Z);
    Z = null;
    xv();
    $.innerHTML = "";
    qv();
    a ? (Z = a, Z.subscribe("proxyStateChange", Ev), Z.subscribe("remotePlayerChange", Fv), Dv = null) : vv();
    2 != Dv && (iv(), K("watch-queue-remote-connection", !!a))
}
function iv() {
    mb(O("page-container"), "remote-connected", !!Z)
}
function wv() {
    of(Vt);
    if (!Xu()) {
        var a = qd(iu) == Vt ? 0: 1;
        K("watch-queue-show", a)
    }
}
function vv() {
    qd(iu) != Vt && Vt.appendChild(iu);
    ob(Vt, "mole-expanded", "mole-collapsed");
    pf(Vt);
    Yu();
    K("watch-queue-hide", 0)
}
function Ev(a, b) {
    xv();
    1 == b && Av()
}
function Fv() {
    var a = Nq(Z);
    gv(1 == a.g)
}
function Qu() {
    if (Z) {
        var a = ev(), b = $u();
        b && b != a && fv(b);
        b && (Wu() || Yt == b) && Cv(b);
        Av()
    }
}
var ov = {
    "remove-all": function() {
        tu();
        $.innerHTML = "";
        qv();
        K("watch-queue-empty");
        vv()
    },
    "save-as-playlist": function() {
        if (!Ua(yu())) {
            var a = Zt;
            di.getInstance().show(a.wa())
        }
    },
    disconnect: function() {
        js()
    }
}, pv = {
    "play-next": function(a) {
        B(a, "disabled") || (a = E(a, "video-id")) && jv(a)
    },
    "goto-video-page": nv
}, Bu = [], Lu = [], Ju = 0, yv = NaN, Dv = null;
var Gv, Hv, Iv, Jv, Kv, Lv;
function Mv() {
    if (Gv = Q("watch-timeline"))
        Hv = Q("watch-timeline-sections", Gv), Iv = Q("watch-timeline-control-bar"), Jv = Q("watch-timeline-control-bar-info", Iv), Q("yt-uix-button-dark-overflow-action-menu", Iv), Kv = Q("prev-watch-timeline-button"), Lv = Q("next-watch-timeline-button"), Hv && K("yt-dom-content-change", Hv), me(Gv, "click", Nv, "yt-uix-button"), me(Gv, "click", Ov, "watch-timeline-video-link"), me(Hv, "click", Pv, "overflow-menu-choice")
}
function Qv() {
    if ("watch" == F("PAGE_NAME")) {
        var a = O("watch-timeline-mole-container");
        a && hd(a);
        Mv()
    }
}
function Rv() {
    if ("watch" == F("PAGE_NAME")) {
        var a = O("watch-timeline-mole-container"), b = O("watch-timeline-sidebar-container");
        a && b && gd(a, ld(b))
    }
}
function Nv(a) {
    a.currentTarget == Jv ? Sv() : a.currentTarget != Lv && a.currentTarget != Kv || Ov(a)
}
function Sv() {
    Tv() && (nb(Gv, "mole-expanded"), R(Hv, xf, function() {
        Hv && K("yt-dom-content-change", Hv)
    }))
}
function Ov(a) {
    if (Tv()) {
        var b = a || window.event;
        b.returnValue=!1;
        b.preventDefault && b.preventDefault();
        if (a = E(a.currentTarget, "video-id"))
            Ut(!0), (b = Tt()) && b.loadVideoById(a);
        K("watch-timeline-refresh", a)
    }
}
function Pv(a) {
    var b = E(a.currentTarget, "action");
    Uv[b](a.currentTarget)
}
function Tv() {
    return qd(Gv) == O("watch-timeline-mole-container")
}
var Uv = {
    "goto-video-page": function(a) {
        a = wd(a, "watch-timeline-item");
        a = Q("watch-timeline-video-link", a);
        wi(a.href)
    }
};
var Vv;
function Wv(a) {
    Xv(O("yt-picker-" + a + "-footer"), a, !1, !1, !0);
    O("footer").scrollIntoView()
}
function Xv(a, b, c, d, e) {
    c || (Vv && Vv != a && pf(Vv), Vv = a, e ? of(a) : qf(a));
    d ? Q("yt-close", a).focus() : (c = {}, c["action_" + b] = 1, c.base_url = window.location.href.split("#", 1)[0], ff("/picker_ajax", {
        format: "JSON",
        method: "GET",
        $a: c,
        V: u(Yv, null, a),
        onError: function() {
            pf(a)
        }
    }))
}
function Yv(a, b, c) {
    if (c && c.html) {
        Eb(a, "loaded", 1);
        a.innerHTML = c.html;
        Zv(a);
        b = document.body;
        var d = Kd(a), e = Kd(b), f = Yd(b);
        c = d.x - e.x - f.left;
        var d = d.y - e.y - f.top, e = b.clientHeight - a.offsetHeight, f = b.scrollLeft, h = b.scrollTop, f = f + Math.min(c, Math.max(c - (b.clientWidth - a.offsetWidth), 0)), h = h + Math.min(d, Math.max(d - e, 0));
        c = new L(f, h);
        b.scrollLeft = c.x;
        b.scrollTop = c.y;
        Q("yt-close", a).focus()
    } else 
        pf(a)
}
function Zv(a, b) {
    var c = Q("yt-picker-content", a);
    if (c) {
        var d = Vc(null, "yt-picker-section", c);
        z(d, u($v, null, Math.floor((b || c.offsetWidth) / 180)))
    } else 
        Sb(Error(".yt-picker-content missing"), "WARNING")
}
function $v(a, b) {
    for (var c = Vc(null, "yt-picker-item", b), d = Math.ceil(c.length / a), e, f = document.createElement("div"), h = 0; h < a; h++) {
        e = document.createElement("div");
        e.className = "yt-picker-grid";
        for (var k = d * h; k < d * (h + 1); k++)
            c[k] && e.appendChild(c[k]);
        e.children.length && f.appendChild(e)
    }
    b.innerHTML = f.innerHTML
};
q("yt.www.guide.collections.onCollectionPickerShown", function() {
    var a = Di();
    Qi();
    var b = Q("collection-name-input", a), c = E(a, "collection-id");
    c || (b.value = "");
    Ei(a);
    Ui(a, !0);
    var d = Q("subscription-picker-list-container", a);
    d.innerHTML = "";
    ff("/subscription_ajax?action_list_subscriptions=1", {
        method: "POST",
        ca: {
            session_token: Pb(),
            collection_id: c
        },
        V: function(b, f) {
            Ei(a);
            d.innerHTML = f.response.html_content;
            if (c) {
                Mi();
                var h = Q("selected-channels-button", a);
                Ag(wg.getInstance(), h);
                Ri(h)
            }
        },
        onError: Ki(a)
    })
}, void 0);
q("yt.www.guide.setup", Xj, void 0);
q("yt.www.guide.selectGuideItem", function(a) {
    if (ig) {
        var b = null, c = Q("guide-item-selected", ig);
        c && (b = E(c, "serialized-endpoint") || "");
        b != a && (b && D(c, "guide-item-selected"), a && (a = kk(a)) && C(a, "guide-item-selected"))
    }
}, void 0);
q("yt.www.masthead.dismissCookieAlert", function() {
    gj("GC_OK", void 0);
    ih.getInstance();
    oh(110, !0);
    fh("PREF", ph(), 63072E3)
}, void 0);
q("yt.www.masthead.dismissReturnToMobileMessage", function() {
    gj("NO_MWEB", void 0);
    pf(O("return-to-mobile"))
}, void 0);
q("yt.www.masthead.handleAccountPickerClick", function() {
    var a = O("yt-masthead-multilogin");
    E(a, "loaded") ? ul() : spf.load("/delegate_account_ajax", {
        method: "POST",
        postData: Ue(fl()),
        onDone: function() {
            Eb(a, "loaded", "true");
            ul()
        }
    })
}, void 0);
q("yt.www.masthead.loadSearchbox", wl, void 0);
q("yt.www.masthead.appbar.toggleMenu", function(a, b) {
    if (b) {
        var c;
        c = wg.getInstance();
        c = zg(c, a);
        K("yt-dom-content-change", c)
    }
}, void 0);
q("yt.net.cookies.set", fh, void 0);
q("yt.net.cookies.remove", hh, void 0);
q("yt.www.picker.load", function(a) {
    var b = E(a, "picker-key"), c = E(a, "picker-position"), d = "yt-picker-" + b + "-" + c;
    if (c = O(d))
        a = E(a, "button-menu-id") == d, d = E(c, "loaded"), Xv(c, b, !!a, !!d)
}, void 0);
q("yt.www.picker.displayLang", function() {
    Wv("language")
}, void 0);
q("yt.www.picker.applyGrid", Zv, void 0);
q("yt.www.watchtimeline.setupTimeline", Mv, void 0);
q("ytbin.www.pageframe.setup", function() {
    qc("GUIDE_DELAYLOADED_CSS");
    qc("PAGE_FRAME_DELAYLOADED_CSS", v(D, document.body, "delayed-frame-styles-not-in"), F("WAIT_TO_DELAYLOAD_FRAME_CSS") ? "yt-www-pageFrameCssNotifications-load" : null);
    Q("guide-module-loading") || Xj();
    xj = Q("yt-masthead-user-icon");
    me(O("ticker"), "click", vl, "yt-uix-close");
    F("SANDBAR_ENABLED") && tl();
    wl();
    Fl();
    (Vt = O("watch-queue-mole")) ? Au() : F("MDX_ENABLE_CASTV2") && Sr(!0, !1, {
        device: "Desktop",
        app: "youtube-desktop",
        isSignedIn: F("LOGGED_IN")
    });
    J("init", Qv);
    J("dispose", Rv);
    Qv();
    F("YOODLE_IS_ANIMATED") && fc(F("YOODLE_JS_URL"), function() {
        r("yt.www.masthead.AnimatedYoodle.init")(F("YOODLE_ANIMATION_FRAMES"), F("YOODLE_ANIMATION_DURATION"), F("YOODLE_ANIMATION_DELAY"), F("YOODLE_STOP_ON_LAST_FRAME"))
    });
    F("SAFETY_MODE_PENDING") && Wv("safetymode")
}, void 0);
q("ytbin.www.pageframe.cancelSetup", function() {
    yl()
}, void 0);
})();

