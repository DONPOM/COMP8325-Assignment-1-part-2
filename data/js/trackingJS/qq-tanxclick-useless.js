/* 2014-10-23T11:09:51.904Z */
!function(t, e) {
    function n(t) {
        var e = u(t);
        k = new Date - 0, _ = t.target, E = e.x, D = e.y
    }
    function a(t) {
        var e = u(t);
        if (!(_ != t.target || Math.abs(e.x - E) > 6 || Math.abs(e.y - D) > 6) && (k ? (T.mc = new Date - k, k = 0) : T.mc = 0, T.ex = E, T.ey = D, T.ml = 0, T.mox = 0, T.moy = 0, T.aml = 0, T.dt = 0, T.st = 0, T.mdx = E, T.mdy = D, h = l(t))) {
            var n = (h.getAttribute("href", 2).match(/http:\/\/([^\/]+)/i) || ["", ""])[1];
            d(n)
        }
    }
    function r(t) {
        var e = (new Date).getTime(), n = e - X;
        if (n > 5e3 ? T.mt = N = [] : T.dt += n, X = e, N.length < 5) {
            var a = u(t), r = a.x, o = a.y;
            (Math.abs(r - U) > 10 || Math.abs(o - M) > 10) && (U = r, M = o, N.push(r + "," + o))
        }
    }
    function o(t) {
        var e = t || window.event;
        r(e), k = 0
    }
    function i(t) {
        var e, n = t || window.event;
        (null === T.ex || null === T.ey) && (e = u(n), T.ex = e.x, T.ey = e.y), null === T.ml ? T.ml = 1 : T.ml++;
        var a = l(n);
        a && (C = new Date - 0, e = u(n), T.mox = e.x, T.moy = e.y, null === T.aml ? T.aml = 1 : T.aml++)
    }
    function l(t) {
        var e = t.srcElement ? t.srcElement: t.target;
        if ("A" != e.tagName.toUpperCase()) {
            for (var n = 5; n > 0; n--) {
                if (e = e.parentNode, !e ||!e.tagName)
                    return null;
                if ("A" == e.tagName.toUpperCase())
                    break
            }
            if ("A" != e.tagName.toUpperCase())
                return null
        }
        return "undefined" == typeof e.href ? null : "A" == e.tagName.toUpperCase() && 0 !== e.getAttribute("href", 2).indexOf("http") ? null : e
    }
    function u(t) {
        var e = t || window.event;
        if (b)
            try {
                return {
                    x: t.changedTouches[0].pageX,
                    y: t.changedTouches[0].pageY
                }
        } catch (e) {}
        return e.pageX || e.pageY ? {
            x: e.pageX,
            y: e.pageY
        } : {
            x: e.clientX + document.body.scrollLeft - document.body.clientLeft,
            y: e.clientY + document.body.scrollTop - document.body.clientTop
        }
    }
    function m(t) {
        var e = t || window.event;
        k = new Date - 0, T.st = k - C;
        var n = u(e);
        T.mdx = n.x, T.mdy = n.y, 4 == e.button && c(e), 2 == e.button && c(e)
    }
    function c(t) {
        k ? (T.mc = new Date - k, k = 0) : T.mc = 0;
        try {
            var e = t || window.event;
            if (h = l(e), !h)
                return;
            var n = (h.getAttribute("href", 2).match(/http:\/\/([^\/]+)/i) || ["", ""])[1];
            d(n)
        } catch (a) {}
    }
    function d(n) {
        var a = [1];
        a.push(T.ex + "," + T.ey), a.push(T.ml), a.push(T.mox + "," + T.moy), a.push(T.aml), a.push(T.mt.join("-")), a.push(T.dt), a.push(T.st), a.push(T.mdx + "," + T.mdy), a.push(T.mc), a.push(T.hl), a.push(T.brws_w + "x" + T.brws_h), a = a.join("_"), a = a + "|" + s(a);
        for (var r = 0; r < e.length; r++)
            if (p[e[r]]) {
                var o = (new Date).getTime();
                t.img = new Image, t.img.id = o, t.img.src = decodeURIComponent(p[e[r]]) + "&d_r=" + n + "_" + (new Date).getTime().toString().substr(4) + "&tr=" + a
            }
    }
    function s(t) {
        var e, n = 0;
        if (0 === t.length)
            return n;
        for (g = 0; g < t.length; g++)
            e = t.charCodeAt(g), n = (n<<5) - n + e;
        return (n>>>0).toString(16)
    }
    if (!t.__tanxclick_bind__) {
        t.__tanxclick_bind__=!0;
        var h, v = document.body, w = location.href.split("?");
        w.shift();
        for (var f = (w.join("?") || "").split("&"), p = {}, g = 0; g < f.length; g++) {
            var y = f[g].split("=");
            p[y[0]] = p[y[0]] || y[1]
        }
        var x=!(!window.attachEvent || window.opera), b = "ontouchstart"in window;
        if (b) {
            var _, E, D;
            document.addEventListener("touchstart", n, !1), document.addEventListener("touchend", a, !1)
        } else 
            x ? (v.attachEvent("onclick", c), v.attachEvent("onmousemove", o), v.attachEvent("onmouseover", i), v.attachEvent("onmousedown", m)) : (v.addEventListener("click", c, !1), v.addEventListener("mousemove", o, !1), v.addEventListener("mouseover", i, !1), v.addEventListener("mousedown", m, !1));
        var T = {
            v: 1,
            ex: null,
            ey: null,
            ml: null,
            mox: null,
            moy: null,
            aml: null,
            mt: [],
            dt: 0,
            st: null,
            mdx: null,
            mdy: null,
            mc: null,
            hl: window.history.length,
            brws_w: null,
            brws_h: null
        }, A = T.brws_w = v.clientWidth || 0, L = T.brws_h = v.clientHeight || 0;
        A && L || setTimeout(function() {
            T.brws_w = v.clientWidth, T.brws_h = v.clientHeight
        }, 20);
        var k = 0, C = 0, N = [], U = 0, M = 0, X = (new Date).getTime()
    }
}(window, ["tanxclick", "tanxdspv"]);
