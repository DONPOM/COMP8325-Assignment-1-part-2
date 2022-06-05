function AC(t) {
    this.callback = t, this.items = 0, this.results = {}
}
function PL() {
    this.ls = this.checkLs() ? window.localStorage : null, this.lsVersion = "2", this.lsVersionKey = "lsVersion", this.isXDRSupport =- 1 === window.location.hash.indexOf("fail_xdr=1")&&!/fail_xdr=1(;|$)/.test(document.cookie)&&!!(window.XDomainRequest || window.XMLHttpRequest && "withCredentials"in new XMLHttpRequest), this.XDRSupportFailed=!1, this.checkLsVersion()
}
var ya = this.ya || {};
ya.zCookie = {}, ya.zCookie.insert = function(t, e) {
    var o, n, r, s = e.split(":"), i = [], a=!1;
    if (3 === s.length) {
        r = s[0], n = t.split("|");
        for (var l = 0, c = n.length; c > l; l++)
            n[l] && (o = n[l].split(":"), o[0] === r ? (i.push(e), a=!0) : i.push(n[l]));
        return a || i.push(e), i.join("|")
    }
    return !1
}, AC.prototype.add = function(t, e) {
    var o = this;
    return this.items++, function() {
        var n = e.apply(e, arguments);
        o.check(t, n)
    }
}, AC.prototype.check = function(t, e) {
    this.results[t] = e, this.items--, 0 === this.items && this.callback.call(this, this.results)
}, PL.prototype.checkLs = function() {
    try {
        return "localStorage"in window && null !== window.localStorage ? (localStorage.setItem("test", "1"), "1" === localStorage.getItem("test")) : !1
    } catch (t) {
        return !1
    }
}, PL.prototype.checkLsVersion = function() {
    if (this.ls)
        try {
            this.ls.getItem(this.lsVersionKey) !== this.lsVersion && this.ls.clear(), this.ls.setItem(this.lsVersionKey, this.lsVersion)
    } catch (t) {}
}, PL.prototype.clearLs = function(t) {
    var e, o, n, r, s = [];
    if (this.ls)
        try {
            for (var i = 0, a = this.ls.length; a > i; ++i)
                e = this.ls.key(i), e && e.indexOf("/")>-1 && (o = e.substr(0, e.indexOf("/")), r = this.ls.getItem(e), r && "v=" === r.substr(0, 2) && r.indexOf("@")>-1 && (n = r.substr(2, r.indexOf("@") - 2), n && o && t[o] !== n && s.push(e)));
                for (i = 0, a = s.length; a > i; ++i)
                    this.ls.removeItem(s[i])
    } catch (l) {}
}, PL.prototype.clearCookie = function(t, e) {
    for (var o, n, r, s = t.split("|"), i = [], a = 0, l = s.length; l > a; a++)
        s[a] && (r = s[a].split(":"), "m-" === r[0].substr(0, 2) && (o = r[0].substr(2), n = r[1], "-https" === o.substring(o.length - 6) && (o = o.substring(0, o.length - 6)), (n === e[o] || n === e[o + "-https"]) && i.push(s[a])));
    return i.join("|")
}, PL.prototype.load = function(t, e, o) {
    var n, r, s, i, a, l = this, c = new AC(function(t) {
        var n, r, s, i = {}, a = [];
        for (n in t)
            t.hasOwnProperty(n) && (r = "m-" + t[n].name + ":" + t[n].version + ":", i[r] || (i[r] = {
                0: 0,
                1: 0,
                2: 0
            }), i[r][String(t[n].result)]++);
        for (n in i)
            i.hasOwnProperty(n) && (s = n, r = i[n], r[0] || (!r[1] && r[2] ? s += "l" : r[1] && (s += "c"), a.push(s)));
        if (l.clearLs(e), a.length) {
            var c = decodeURIComponent((document.cookie.match(/z=([^;]+)/) || [])[1] || ""), u = c, p = function(t, e) {
                var o, n, r, s = e.split(":"), i = [], a=!1;
                if (3 === s.length) {
                    r = s[0], n = t.split("|");
                    for (var l = 0, c = n.length; c > l; l++)
                        n[l] && (o = n[l].split(":"), o[0] === r ? (i.push(e), a=!0) : i.push(n[l]));
                    return a || i.push(e), i.join("|")
                }
                return t
            };
            u = l.clearCookie(u, e);
            for (var h = 0, d = a.length; d > h; h++)
                u = p(u, a[h]);
            u && u !== c && (document.cookie = "z=" + encodeURIComponent(u) + "; expires=" + new Date( + new Date + 12096e5).toUTCString() + "; path=/")
        }
        o()
    });
    for (var u in t)
        if (t.hasOwnProperty(u)) {
            n = t[u], r = e[u], s = n.length;
            for (var p = 0; s > p; p++)
                i = n[p], a = this.getFileId(u, i), this.loadFile(r, u, i, c.add(a, this.loadFileHandler))
        }
}, PL.prototype.notloaded = {
    css: 0,
    js: 0
}, PL.prototype.markAsNotloaded = function(t) {
    this.notloaded[t]++
}, PL.prototype.markAsLoaded = function(t) {
    this.notloaded[t]--, window.$&&!this.notloaded[t] && $(window).trigger(t + "loaded.cacher")
}, PL.prototype.loadFile = function(t, e, o, n) {
    var r, s = this;
    if (this.ls && (r = this.loadFromLs(t, e, o)))
        return !o.dry && this["run" + o.type] && this["run" + o.type](r), void setTimeout(function() {
        n({
            version: t,
            name: e,
            result: 2
        })
    }, 0);
    if (this.markAsNotloaded(o.type), !o.dry) {
        var i = this.getExtension(o.url);
        this["get" + i] && this["get" + i](o.url, function() {
            s.markAsLoaded(o.type)
        })
    }
    this.XDRSupport() ? this.get(o.url, function(r) {
        var i=!1;
        r ? (i = s.saveToLs(t, e, o, r), n({
            version: t,
            name: e,
            result: i ? 2: 1
        })) : n({
            version: t,
            name: e,
            result: 1
        })
    }) : n({
        version: t,
        name: e,
        result: 1
    })
}, PL.prototype.loadFileHandler = function(t) {
    return t
}, PL.prototype.getFileId = function(t, e) {
    return t + e.name
}, PL.prototype.getVersion = function(t) {
    return "v=" + t + "@"
}, PL.prototype.getExtension = function(t) {
    return t.split(".").slice( - 1)[0]
}, PL.prototype.getXHR = function() {
    return window.XMLHttpRequest ? function() {
        return new window.XMLHttpRequest
    } : function() {
        try {
            return new window.ActiveXObject("Microsoft.XMLHTTP")
        } catch (t) {}
    }
}, PL.prototype.getXDR = function() {
    return window.XDomainRequest ? function() {
        return new window.XDomainRequest
    } : this.getXHR()
}, PL.prototype.XDRSupport = function() {
    return this.isXDRSupport
}, PL.prototype.appendToHead = function(t) {
    document.getElementsByTagName("head").item(0).appendChild(t)
}, PL.prototype.getcss = function(t, e) {
    var o = document.createElement("link");
    o.setAttribute("type", "text/css"), o.setAttribute("rel", "stylesheet"), o.setAttribute("href", t), /(opera|firefox)(?:.*version)?[ \/]([\w.]+)/i.test(navigator.userAgent) ? o.onload = e : "onreadystatechange"in o ? o.onreadystatechange = function() {
        /loaded|complete/.test(o.readyState) && e()
    } : !function n() {
        o && o.sheet ? e() : setTimeout(n, 20)
    }(), this.appendToHead(o)
}, PL.prototype.getjs = function(t, e) {
    var o = document.createElement("script");
    o.setAttribute("type", "text/javascript"), o.setAttribute("charset", "utf-8"), o.setAttribute("src", t), this.appendToHead(o);
    var n=!1;
    o.onload = o.onreadystatechange = function() {
        var t = this.readyState;
        n || t && "loaded" !== t && "complete" !== t || (n=!0, e(), o.onload = o.onreadystatechange = null)
    }
}, PL.prototype.get = function(t, e) {
    var o = this, n = /^\/\/|^https:\/\/|^http:\/\//.test(t) ? this.getXDR(): this.getXHR();
    if (n) {
        n = n(), n.open("GET", t, !0);
        var r = function() {
            n.onreadystatechange = function() {};
            try {
                n.onload = n.onerror = n.onprogress = function() {}
            } catch (t) {}
            r = s = null;
            var o = n.responseText;
            e(o)
        }, s = function() {
            n.onreadystatechange = function() {};
            try {
                n.onload = n.onerror = n.onprogress = function() {}
            } catch (t) {}
            r = s = null;
            try {
                n.abort()
            } catch (t) {}
            o.onXDRFail(), e()
        };
        "onreadystatechange"in n ? n.onreadystatechange = function() {
            return n.readyState < 4 ? void 0 : location.hash.indexOf("abort_xdr=1")>-1 ? void(s && s()) : void(200 == n.status && n.responseText ? r && r() : s && s())
        } : n.onload = r;
        try {
            n.onerror = s, n.onprogress = function() {}
        } catch (i) {}
        n.send(null)
    }
}, PL.prototype.onXDRFail = function() {
    this.XDRSupportFailed || (document.cookie = "fail_xdr=1;max-age=604800", this.isXDRSupport=!1, this.XDRSupportFailed=!0)
}, PL.prototype.runjs = function(t) {
    t && /\S/.test(t) && (window.execScript || function(t) {
        window.eval.call(window, t)
    })(t)
}, PL.prototype.appendStyle = function(t, e) {
    e = e || null;
    var o = document.createElement("style");
    o.setAttribute("type", "text/css"), e && (o.media = e), this.appendToHead(o), o.styleSheet ? o.styleSheet.cssText = t : o.appendChild(document.createTextNode(t))
}, PL.prototype.qRegexp = /@media([^\{]+)\{([\s\S]*?\})[\s]*?\}/gi, PL.prototype.fqRegexp = /@media *([^\{]+)\{([\S\s]+?)\}$/, PL.prototype.runcss = function(t) {
    this.appendStyle(t)
}, PL.prototype.loadFromLs = function(t, e, o) {
    if (!this.ls)
        return "";
    var n, r = this.getVersion(t), s = r.length, i = this.getFileId(e, o);
    try {
        n = this.ls.getItem(i)
    } catch (a) {
        return ""
    }
    if (n) {
        var l = n.substr(0, s);
        if (l === r) {
            var c = parseInt(n.substr(s + 2), 10), u = n.substr(s + 3 + (c + "").length);
            if (c === u.length)
                return u;
            this.sendStat("zcookie.error.get")
        }
        try {
            this.ls.removeItem(i)
        } catch (a) {}
    }
    return ""
}, PL.prototype.sendLog = function(t) {
    var e = setInterval(function() {
        "jQuery"in window && (clearTimeout(e), jQuery.get("/empty.html?lscachererr=" + encodeURIComponent(t)))
    }, 500)
}, PL.prototype.sendStat = function(t) {
    var e = setInterval(function() {
        window.cp && Lego && Lego.params && Lego.params.statRoot && (clearInterval(e), window.cp(Lego.params.statRoot + "." + t))
    }, 1e3)
}, PL.prototype.saveToLs = function(t, e, o, n) {
    if (!this.ls)
        return !1;
    if (o.size !== n.length)
        return this.sendStat("zcookie.error.load"), !1;
    var r, s = this.getFileId(e, o);
    try {
        if (r = this.getVersion(t) + "s=" + n.length + "@" + n, this.ls.setItem(s, r), this.ls.getItem(s).length !== r.length)
            return this.ls.removeItem(s), this.sendStat("zcookie.error.set"), !1
    } catch (i) {
        return !1
    }
    return !0
};
var pl = new PL;
