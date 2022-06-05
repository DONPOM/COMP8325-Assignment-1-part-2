/*!
 * Shadowbox.js, version 3.0.3
 * http://shadowbox-js.com/
 *
 * Copyright 2007-2010, Michael J. I. Jackson
 * 2010-03-10 00:00:00 +0000
 */
(function(window, undefined) {
    var S = {
        version: "3.0.3"
    };
    var ua = navigator.userAgent.toLowerCase();
    if (ua.indexOf("windows")>-1 || ua.indexOf("win32")>-1) {
        S.isWindows = true
    } else {
        if (ua.indexOf("macintosh")>-1 || ua.indexOf("mac os x")>-1) {
            S.isMac = true
        } else {
            if (ua.indexOf("linux")>-1) {
                S.isLinux = true
            }
        }
    }
    S.isIE = ua.indexOf("msie")>-1;
    S.isIE6 = ua.indexOf("msie 6")>-1;
    S.isIE7 = ua.indexOf("msie 7")>-1;
    S.isGecko = ua.indexOf("gecko")>-1 && ua.indexOf("safari")==-1;
    S.isWebKit = ua.indexOf("applewebkit/")>-1;
    var inlineId = /#(.+)$/, galleryName = /^(light|shadow)box\[(.*?)\]/i, inlineParam = /\s*([a-z_]*?)\s*=\s*(.+)\s*/, fileExtension = /[0-9a-z]+$/i, scriptPath = /(.+\/)shadowbox\.js/i;
    var open = false, initialized = false, lastOptions = {}, slideDelay = 0, slideStart, slideTimer;
    S.current =- 1;
    S.dimensions = null;
    S.ease = function(a) {
        return 1 + Math.pow(a - 1, 3)
    };
    S.errorInfo = {
        fla: {
            name: "Flash",
            url: "http://www.adobe.com/products/flashplayer/"
        },
        qt: {
            name: "QuickTime",
            url: "http://www.apple.com/quicktime/download/"
        },
        wmp: {
            name: "Windows Media Player",
            url: "http://www.microsoft.com/windows/windowsmedia/"
        },
        f4m: {
            name: "Flip4Mac",
            url: "http://www.flip4mac.com/wmv_download.htm"
        }
    };
    S.gallery = [];
    S.onReady = noop;
    S.path = 'http://vindico.com/wp-content/uploads/shadowbox-js/src/';
    S.player = null;
    S.playerId = "sb-player";
    S.options = {
        animate: true,
        animateFade: true,
        autoplayMovies: true,
        continuous: false,
        enableKeys: true,
        flashParams: {
            bgcolor: "#000000",
            allowfullscreen: true
        },
        flashVars: {},
        flashVersion: "9.0.115",
        handleOversize: "resize",
        handleUnsupported: "link",
        onChange: noop,
        onClose: noop,
        onFinish: noop,
        onOpen: noop,
        showMovieControls: true,
        skipSetup: false,
        slideshowDelay: 0,
        viewportPadding: 20
    };
    S.getCurrent = function() {
        return S.current>-1 ? S.gallery[S.current] : null
    };
    S.hasNext = function() {
        return S.gallery.length > 1 && (S.current != S.gallery.length - 1 || S.options.continuous)
    };
    S.isOpen = function() {
        return open
    };
    S.isPaused = function() {
        return slideTimer == "pause"
    };
    S.applyOptions = function(a) {
        lastOptions = apply({}, S.options);
        apply(S.options, a)
    };
    S.revertOptions = function() {
        apply(S.options, lastOptions)
    };
    S.init = function(c, f) {
        if (initialized) {
            return 
        }
        initialized = true;
        if (S.skin.options) {
            apply(S.options, S.skin.options)
        }
        if (c) {
            apply(S.options, c)
        }
        if (!S.path) {
            var e, b = document.getElementsByTagName("script");
            for (var d = 0, a = b.length; d < a; ++d) {
                e = scriptPath.exec(b[d].src);
                if (e) {
                    S.path = e[1];
                    break
                }
            }
        }
        if (f) {
            S.onReady = f
        }
        bindLoad()
    };
    S.open = function(b) {
        if (open) {
            return 
        }
        var a = S.makeGallery(b);
        S.gallery = a[0];
        S.current = a[1];
        b = S.getCurrent();
        if (b == null) {
            return 
        }
        S.applyOptions(b.options || {});
        filterGallery();
        if (S.gallery.length) {
            b = S.getCurrent();
            if (S.options.onOpen(b) === false) {
                return 
            }
            open = true;
            S.skin.onOpen(b, load)
        }
    };
    S.close = function() {
        if (!open) {
            return 
        }
        open = false;
        if (S.player) {
            S.player.remove();
            S.player = null
        }
        if (typeof slideTimer == "number") {
            clearTimeout(slideTimer);
            slideTimer = null
        }
        slideDelay = 0;
        listenKeys(false);
        S.options.onClose(S.getCurrent());
        S.skin.onClose();
        S.revertOptions()
    };
    S.play = function() {
        if (!S.hasNext()) {
            return 
        }
        if (!slideDelay) {
            slideDelay = S.options.slideshowDelay * 1000
        }
        if (slideDelay) {
            slideStart = now();
            slideTimer = setTimeout(function() {
                slideDelay = slideStart = 0;
                S.next()
            }, slideDelay);
            if (S.skin.onPlay) {
                S.skin.onPlay()
            }
        }
    };
    S.pause = function() {
        if (typeof slideTimer != "number") {
            return 
        }
        slideDelay = Math.max(0, slideDelay - (now() - slideStart));
        if (slideDelay) {
            clearTimeout(slideTimer);
            slideTimer = "pause";
            if (S.skin.onPause) {
                S.skin.onPause()
            }
        }
    };
    S.change = function(a) {
        if (!(a in S.gallery)) {
            if (S.options.continuous) {
                a = (a < 0 ? S.gallery.length + a : 0);
                if (!(a in S.gallery)) {
                    return 
                }
            } else {
                return 
            }
        }
        S.current = a;
        if (typeof slideTimer == "number") {
            clearTimeout(slideTimer);
            slideTimer = null;
            slideDelay = slideStart = 0
        }
        S.options.onChange(S.getCurrent());
        load(true)
    };
    S.next = function() {
        S.change(S.current + 1)
    };
    S.previous = function() {
        S.change(S.current - 1)
    };
    S.setDimensions = function(o, f, m, n, e, a, k, h) {
        var j = o, d = f;
        var i = 2 * k + e;
        if (o + i > m) {
            o = m - i
        }
        var c = 2 * k + a;
        if (f + c > n) {
            f = n - c
        }
        var b = (j - o) / j, l = (d - f) / d, g = (b > 0 || l > 0);
        if (h && g) {
            if (b > l) {
                f = Math.round((d / j) * o)
            } else {
                if (l > b) {
                    o = Math.round((j / d) * f)
                }
            }
        }
        S.dimensions = {
            height: o + e,
            width: f + a,
            innerHeight: o,
            innerWidth: f,
            top: Math.floor((m - (o + i)) / 2 + k),
            left: Math.floor((n - (f + c)) / 2 + k),
            oversized: g
        };
        return S.dimensions
    };
    S.makeGallery = function(e) {
        var a = [], d =- 1;
        if (typeof e == "string") {
            e = [e]
        }
        if (typeof e.length == "number") {
            each(e, function(g, h) {
                if (h.content) {
                    a[g] = h
                } else {
                    a[g] = {
                        content: h
                    }
                }
            });
            d = 0
        } else {
            if (e.tagName) {
                var b = S.getCache(e);
                e = b ? b : S.makeObject(e)
            }
            if (e.gallery) {
                a = [];
                var f;
                for (var c in S.cache) {
                    f = S.cache[c];
                    if (f.gallery && f.gallery == e.gallery) {
                        if (d==-1 && f.content == e.content) {
                            d = a.length
                        }
                        a.push(f)
                    }
                }
                if (d==-1) {
                    a.unshift(e);
                    d = 0
                }
            } else {
                a = [e];
                d = 0
            }
        }
        each(a, function(g, h) {
            a[g] = apply({}, h)
        });
        return [a, d]
    };
    S.makeObject = function(d, c) {
        var e = {
            content: d.href,
            title: d.getAttribute("title") || "",
            link: d
        };
        if (c) {
            c = apply({}, c);
            each(["player", "title", "height", "width", "gallery"], function(f, g) {
                if (typeof c[g] != "undefined") {
                    e[g] = c[g];
                    delete c[g]
                }
            });
            e.options = c
        } else {
            e.options = {}
        }
        if (!e.player) {
            e.player = S.getPlayer(e.content)
        }
        var a = d.getAttribute("rel");
        if (a) {
            var b = a.match(galleryName);
            if (b) {
                e.gallery = escape(b[2])
            }
            each(a.split(";"), function(f, g) {
                b = g.match(inlineParam);
                if (b) {
                    e[b[1]] = b[2]
                }
            })
        }
        return e
    };
    S.getPlayer = function(c) {
        if (c.indexOf("#")>-1 && c.indexOf(document.location.href) == 0) {
            return "inline"
        }
        var d = c.indexOf("?");
        if (d>-1) {
            c = c.substring(0, d)
        }
        var b, a = c.match(fileExtension);
        if (a) {
            b = a[0].toLowerCase()
        }
        if (b) {
            if (S.img && S.img.ext.indexOf(b)>-1) {
                return "img"
            }
            if (S.swf && S.swf.ext.indexOf(b)>-1) {
                return "swf"
            }
            if (S.flv && S.flv.ext.indexOf(b)>-1) {
                return "flv"
            }
            if (S.qt && S.qt.ext.indexOf(b)>-1) {
                if (S.wmp && S.wmp.ext.indexOf(b)>-1) {
                    return "qtwmp"
                } else {
                    return "qt"
                }
            }
            if (S.wmp && S.wmp.ext.indexOf(b)>-1) {
                return "wmp"
            }
        }
        return "iframe"
    };
    function filterGallery() {
        var d = S.errorInfo, e = S.plugins, g, h, l, c, k, b, j, a;
        for (var f = 0; f < S.gallery.length; ++f) {
            g = S.gallery[f];
            h = false;
            l = null;
            switch (g.player) {
            case"flv":
            case"swf":
                if (!e.fla) {
                    l = "fla"
                }
                break;
            case"qt":
                if (!e.qt) {
                    l = "qt"
                }
                break;
            case"wmp":
                if (S.isMac) {
                    if (e.qt && e.f4m) {
                        g.player = "qt"
                    } else {
                        l = "qtf4m"
                    }
                } else {
                    if (!e.wmp) {
                        l = "wmp"
                    }
                }
                break;
            case"qtwmp":
                if (e.qt) {
                    g.player = "qt"
                } else {
                    if (e.wmp) {
                        g.player = "wmp"
                    } else {
                        l = "qtwmp"
                    }
                }
                break
            }
            if (l) {
                if (S.options.handleUnsupported == "link") {
                    switch (l) {
                    case"qtf4m":
                        k = "shared";
                        b = [d.qt.url, d.qt.name, d.f4m.url, d.f4m.name];
                        break;
                    case"qtwmp":
                        k = "either";
                        b = [d.qt.url, d.qt.name, d.wmp.url, d.wmp.name];
                        break;
                    default:
                        k = "single";
                        b = [d[l].url, d[l].name]
                    }
                    g.player = "html";
                    g.content = '<div class="sb-message">' + sprintf(S.lang.errors[k], b) + "</div>"
                } else {
                    h = true
                }
            } else {
                if (g.player == "inline") {
                    c = inlineId.exec(g.content);
                    if (c) {
                        j = get(c[1]);
                        if (j) {
                            g.content = j.innerHTML
                        } else {
                            h = true
                        }
                    } else {
                        h = true
                    }
                } else {
                    if (g.player == "swf" || g.player == "flv") {
                        a = (g.options && g.options.flashVersion) || S.options.flashVersion;
                        if (S.flash&&!S.flash.hasFlashPlayerVersion(a)) {
                            g.width = 310;
                            g.height = 177
                        }
                    }
                }
            }
            if (h) {
                S.gallery.splice(f, 1);
                if (f < S.current) {
                    --S.current
                } else {
                    if (f == S.current) {
                        S.current = f > 0 ? f - 1 : f
                    }
                }
                --f
            }
        }
    }
    function listenKeys(a) {
        if (!S.options.enableKeys) {
            return 
        }(a ? addEvent : removeEvent)(document, "keydown", handleKey)
    }
    function handleKey(c) {
        if (c.metaKey || c.shiftKey || c.altKey || c.ctrlKey) {
            return 
        }
        var b = keyCode(c), a;
        switch (b) {
        case 81:
        case 88:
        case 27:
            a = S.close;
            break;
        case 37:
            a = S.previous;
            break;
        case 39:
            a = S.next;
            break;
        case 32:
            a = typeof slideTimer == "number" ? S.pause : S.play;
            break
        }
        if (a) {
            preventDefault(c);
            a()
        }
    }
    function load(i) {
        listenKeys(false);
        var h = S.getCurrent();
        var e = (h.player == "inline" ? "html" : h.player);
        if (typeof S[e] != "function") {
            throw "unknown player " + e
        }
        if (i) {
            S.player.remove();
            S.revertOptions();
            S.applyOptions(h.options || {})
        }
        S.player = new S[e](h, S.playerId);
        if (S.gallery.length > 1) {
            var f = S.gallery[S.current + 1] || S.gallery[0];
            if (f.player == "img") {
                var d = new Image();
                d.src = f.content
            }
            var g = S.gallery[S.current - 1] || S.gallery[S.gallery.length - 1];
            if (g.player == "img") {
                var c = new Image();
                c.src = g.content
            }
        }
        S.skin.onLoad(i, waitReady)
    }
    function waitReady() {
        if (!open) {
            return 
        }
        if (typeof S.player.ready != "undefined") {
            var a = setInterval(function() {
                if (open) {
                    if (S.player.ready) {
                        clearInterval(a);
                        a = null;
                        S.skin.onReady(show)
                    }
                } else {
                    clearInterval(a);
                    a = null
                }
            }, 10)
        } else {
            S.skin.onReady(show)
        }
    }
    function show() {
        if (!open) {
            return 
        }
        S.player.append(S.skin.body, S.dimensions);
        S.skin.onShow(finish)
    }
    function finish() {
        if (!open) {
            return 
        }
        if (S.player.onLoad) {
            S.player.onLoad()
        }
        S.options.onFinish(S.getCurrent());
        if (!S.isPaused()) {
            S.play()
        }
        listenKeys(true)
    };
    if (!Array.prototype.indexOf) {
        Array.prototype.indexOf = function(b, c) {
            var a = this.length>>>0;
            c = c || 0;
            if (c < 0) {
                c += a
            }
            for (; c < a; ++c) {
                if (c in this && this[c] === b) {
                    return c
                }
            }
            return - 1
        }
    }
    function now() {
        return (new Date).getTime()
    }
    function apply(a, c) {
        for (var b in c) {
            a[b] = c[b]
        }
        return a
    }
    function each(d, e) {
        var b = 0, a = d.length;
        for (var c = d[0]; b < a && e.call(c, b, c) !== false; c = d[++b]) {}
    }
    function sprintf(b, a) {
        return b.replace(/\{(\w+?)\}/g, function(c, d) {
            return a[d]
        })
    }
    function noop() {}
    function get(a) {
        return document.getElementById(a)
    }
    function remove(a) {
        a.parentNode.removeChild(a)
    }
    var supportsOpacity = true, supportsFixed = true;
    function checkSupport() {
        var a = document.body, b = document.createElement("div");
        supportsOpacity = typeof b.style.opacity === "string";
        b.style.position = "fixed";
        b.style.margin = 0;
        b.style.top = "20px";
        a.appendChild(b, a.firstChild);
        supportsFixed = b.offsetTop == 20;
        a.removeChild(b)
    }
    S.getStyle = (function() {
        var a = /opacity=([^)]*)/, b = document.defaultView && document.defaultView.getComputedStyle;
        return function(f, e) {
            var d;
            if (!supportsOpacity && e == "opacity" && f.currentStyle) {
                d = a.test(f.currentStyle.filter || "") ? (parseFloat(RegExp.$1) / 100) + "" : "";
                return d === "" ? "1" : d
            }
            if (b) {
                var c = b(f, null);
                if (c) {
                    d = c[e]
                }
                if (e == "opacity" && d == "") {
                    d = "1"
                }
            } else {
                d = f.currentStyle[e]
            }
            return d
        }
    })();
    S.appendHTML = function(c, b) {
        if (c.insertAdjacentHTML) {
            c.insertAdjacentHTML("BeforeEnd", b)
        } else {
            if (c.lastChild) {
                var a = c.ownerDocument.createRange();
                a.setStartAfter(c.lastChild);
                var d = a.createContextualFragment(b);
                c.appendChild(d)
            } else {
                c.innerHTML = b
            }
        }
    };
    S.getWindowSize = function(a) {
        if (document.compatMode === "CSS1Compat") {
            return document.documentElement["client" + a]
        }
        return document.body["client" + a]
    };
    S.setOpacity = function(c, a) {
        var b = c.style;
        if (supportsOpacity) {
            b.opacity = (a == 1 ? "" : a)
        } else {
            b.zoom = 1;
            if (a == 1) {
                if (typeof b.filter == "string" && (/alpha/i).test(b.filter)) {
                    b.filter = b.filter.replace(/\s*[\w\.]*alpha\([^\)]*\);?/gi, "")
                }
            } else {
                b.filter = (b.filter || "").replace(/\s*[\w\.]*alpha\([^\)]*\)/gi, "") + " alpha(opacity=" + (a * 100) + ")"
            }
        }
    };
    S.clearOpacity = function(a) {
        S.setOpacity(a, 1)
    };
    function getTarget(a) {
        return a.target
    }
    function getPageXY(a) {
        return [a.pageX, a.pageY]
    }
    function preventDefault(a) {
        a.preventDefault()
    }
    function keyCode(a) {
        return a.keyCode
    }
    function addEvent(c, b, a) {
        jQuery(c).bind(b, a)
    }
    function removeEvent(c, b, a) {
        jQuery(c).unbind(b, a)
    }
    jQuery.fn.shadowbox = function(a) {
        return this.each(function() {
            var c = jQuery(this);
            var d = jQuery.extend({}, a || {}, jQuery.metadata ? c.metadata() : jQuery.meta ? c.data() : {});
            var b = this.className || "";
            d.width = parseInt((b.match(/w:(\d+)/) || [])[1]) || d.width;
            d.height = parseInt((b.match(/h:(\d+)/) || [])[1]) || d.height;
            Shadowbox.setup(c, d)
        })
    };
    var loaded = false, DOMContentLoaded;
    if (document.addEventListener) {
        DOMContentLoaded = function() {
            document.removeEventListener("DOMContentLoaded", DOMContentLoaded, false);
            S.load()
        }
    } else {
        if (document.attachEvent) {
            DOMContentLoaded = function() {
                if (document.readyState === "complete") {
                    document.detachEvent("onreadystatechange", DOMContentLoaded);
                    S.load()
                }
            }
        }
    }
    function doScrollCheck() {
        if (loaded) {
            return 
        }
        try {
            document.documentElement.doScroll("left")
        } catch (a) {
            setTimeout(doScrollCheck, 1);
            return 
        }
        S.load()
    }
    function bindLoad() {
        if (document.readyState === "complete") {
            return S.load()
        }
        if (document.addEventListener) {
            document.addEventListener("DOMContentLoaded", DOMContentLoaded, false);
            window.addEventListener("load", S.load, false)
        } else {
            if (document.attachEvent) {
                document.attachEvent("onreadystatechange", DOMContentLoaded);
                window.attachEvent("onload", S.load);
                var a = false;
                try {
                    a = window.frameElement === null
                } catch (b) {}
                if (document.documentElement.doScroll && a) {
                    doScrollCheck()
                }
            }
        }
    }
    S.load = function() {
        if (loaded) {
            return 
        }
        if (!document.body) {
            return setTimeout(S.load, 13)
        }
        loaded = true;
        checkSupport();
        S.onReady();
        if (!S.options.skipSetup) {
            S.setup()
        }
        S.skin.init()
    };
    S.plugins = {};
    if (navigator.plugins && navigator.plugins.length) {
        var names = [];
        each(navigator.plugins, function(a, b) {
            names.push(b.name)
        });
        names = names.join(",");
        var f4m = names.indexOf("Flip4Mac")>-1;
        S.plugins = {
            fla: names.indexOf("Shockwave Flash")>-1,
            qt: names.indexOf("QuickTime")>-1,
            wmp: !f4m && names.indexOf("Windows Media")>-1,
            f4m: f4m
        }
    } else {
        var detectPlugin = function(a) {
            var b;
            try {
                b = new ActiveXObject(a)
            } catch (c) {}
            return !!b
        };
        S.plugins = {
            fla: detectPlugin("ShockwaveFlash.ShockwaveFlash"),
            qt: detectPlugin("QuickTime.QuickTime"),
            wmp: detectPlugin("wmplayer.ocx"),
            f4m: false
        }
    };
    var relAttr = /^(light|shadow)box/i, expando = "shadowboxCacheKey", cacheKey = 1;
    S.cache = {};
    S.select = function(b) {
        var c = [];
        if (!b) {
            var a;
            each(document.getElementsByTagName("a"), function(f, g) {
                a = g.getAttribute("rel");
                if (a && relAttr.test(a)) {
                    c.push(g)
                }
            })
        } else {
            var e = b.length;
            if (e) {
                if (typeof b == "string") {
                    if (S.find) {
                        c = S.find(b)
                    }
                } else {
                    if (e == 2 && typeof b[0] == "string" && b[1].nodeType) {
                        if (S.find) {
                            c = S.find(b[0], b[1])
                        }
                    } else {
                        for (var d = 0; d < e; ++d) {
                            c[d] = b[d]
                        }
                    }
                }
            } else {
                c.push(b)
            }
        }
        return c
    };
    S.setup = function(a, b) {
        each(S.select(a), function(c, d) {
            S.addCache(d, b)
        })
    };
    S.teardown = function(a) {
        each(S.select(a), function(b, c) {
            S.removeCache(c)
        })
    };
    S.addCache = function(c, a) {
        var b = c[expando];
        if (b == undefined) {
            b = cacheKey++;
            c[expando] = b;
            addEvent(c, "click", handleClick)
        }
        S.cache[b] = S.makeObject(c, a)
    };
    S.removeCache = function(a) {
        removeEvent(a, "click", handleClick);
        delete S.cache[a[expando]];
        a[expando] = null
    };
    S.getCache = function(b) {
        var a = b[expando];
        return (a in S.cache && S.cache[a])
    };
    S.clearCache = function() {
        for (var a in S.cache) {
            S.removeCache(S.cache[a].link)
        }
        S.cache = {}
    };
    function handleClick(a) {
        S.open(this);
        if (S.gallery.length) {
            preventDefault(a)
        }
    };
    S.lang = {
        code: "en",
        of: "of",
        loading: "",
        cancel: "Cancel",
        next: "Next",
        previous: "Previous",
        play: "Play",
        pause: "Pause",
        close: "Close",
        errors: {
            single: 'You must install the <a href="{0}">{1}</a> browser plugin to view this content.',
            shared: 'You must install both the <a href="{0}">{1}</a> and <a href="{2}">{3}</a> browser plugins to view this content.',
            either: 'You must install either the <a href="{0}">{1}</a> or the <a href="{2}">{3}</a> browser plugin to view this content.'
        }
    };
    S.iframe = function(b, c) {
        this.obj = b;
        this.id = c;
        var a = get("sb-overlay");
        this.height = b.height ? parseInt(b.height, 10) : a.offsetHeight;
        this.width = b.width ? parseInt(b.width, 10) : a.offsetWidth
    };
    S.iframe.prototype = {
        append: function(a, c) {
            var b = '<iframe id="' + this.id + '" name="' + this.id + '" height="100%" width="100%" frameborder="0" marginwidth="0" marginheight="0" style="visibility:hidden" onload="this.style.visibility=\'visible\'" scrolling="auto"';
            if (S.isIE) {
                b += ' allowtransparency="true"';
                if (S.isIE6) {
                    b += " src=\"javascript:false;document.write('');\""
                }
            }
            b += "></iframe>";
            a.innerHTML = b
        },
        remove: function() {
            var a = get(this.id);
            if (a) {
                remove(a);
                if (S.isGecko) {
                    delete window.frames[this.id]
                }
            }
        },
        onLoad: function() {
            var a = S.isIE ? get(this.id).contentWindow: window.frames[this.id];
            a.location.href = this.obj.content
        }
    };
    var overlayOn = false, visibilityCache = [], pngIds = ["sb-nav-close", "sb-nav-next", "sb-nav-play", "sb-nav-pause", "sb-nav-previous"], container, overlay, wrapper, doWindowResize = true;
    function animate(c, m, j, h, n) {
        var a = (m == "opacity"), i = a ? S.setOpacity: function(o, p) {
            o.style[m] = "" + p + "px"
        };
        if (h == 0 || (!a&&!S.options.animate) || (a&&!S.options.animateFade)) {
            i(c, j);
            if (n) {
                n()
            }
            return 
        }
        var k = parseFloat(S.getStyle(c, m)) || 0;
        var l = j - k;
        if (l == 0) {
            if (n) {
                n()
            }
            return 
        }
        h*=1000;
        var d = now(), g = S.ease, f = d + h, e;
        var b = setInterval(function() {
            e = now();
            if (e >= f) {
                clearInterval(b);
                b = null;
                i(c, j);
                if (n) {
                    n()
                }
            } else {
                i(c, k + g((e - d) / h) * l)
            }
        }, 10)
    }
    function setSize() {
        container.style.height = S.getWindowSize("Height") + "px";
        container.style.width = S.getWindowSize("Width") + "px"
    }
    function setPosition() {
        container.style.top = document.documentElement.scrollTop + "px";
        container.style.left = document.documentElement.scrollLeft + "px"
    }
    function toggleTroubleElements(a) {
        if (a) {
            each(visibilityCache, function(b, c) {
                c[0].style.visibility = c[1] || ""
            })
        } else {
            visibilityCache = [];
            each(S.options.troubleElements, function(c, b) {
                each(document.getElementsByTagName(b), function(d, e) {
                    visibilityCache.push([e, e.style.visibility]);
                    e.style.visibility = "hidden"
                })
            })
        }
    }
    function toggleNav(c, a) {
        var b = get("sb-nav-" + c);
        if (b) {
            b.style.display = a ? "" : "none"
        }
    }
    function toggleLoading(a, f) {
        var e = get("sb-loading"), c = S.getCurrent().player, d = (c == "img" || c == "html");
        if (a) {
            S.setOpacity(e, 0);
            e.style.display = "block";
            var b = function() {
                S.clearOpacity(e);
                if (f) {
                    f()
                }
            };
            if (d) {
                animate(e, "opacity", 1, S.options.fadeDuration, b)
            } else {
                b()
            }
        } else {
            var b = function() {
                e.style.display = "none";
                S.clearOpacity(e);
                if (f) {
                    f()
                }
            };
            if (d) {
                animate(e, "opacity", 0, S.options.fadeDuration, b)
            } else {
                b()
            }
        }
    }
    function buildBars(m) {
        var f = S.getCurrent();
        get("sb-title-inner").innerHTML = f.title || "";
        var n, j, b, o, k;
        if (S.options.displayNav) {
            n = true;
            var l = S.gallery.length;
            if (l > 1) {
                if (S.options.continuous) {
                    j = k = true
                } else {
                    j = (l - 1) > S.current;
                    k = S.current > 0
                }
            }
            if (S.options.slideshowDelay > 0 && S.hasNext()) {
                o=!S.isPaused();
                b=!o
            }
        } else {
            n = j = b = o = k = false
        }
        toggleNav("close", n);
        toggleNav("next", j);
        toggleNav("play", b);
        toggleNav("pause", o);
        toggleNav("previous", k);
        var a = "";
        if (S.options.displayCounter && S.gallery.length > 1) {
            var l = S.gallery.length;
            if (S.options.counterType == "skip") {
                var e = 0, d = l, c = parseInt(S.options.counterLimit) || 0;
                if (c < l && c > 2) {
                    var g = Math.floor(c / 2);
                    e = S.current - g;
                    if (e < 0) {
                        e += l
                    }
                    d = S.current + (c - g);
                    if (d > l) {
                        d -= l
                    }
                }
                while (e != d) {
                    if (e == l) {
                        e = 0
                    }
                    a += '<a onclick="Shadowbox.change(' + e + ');"';
                    if (e == S.current) {
                        a += ' class="sb-counter-current"'
                    }
                    a += ">" + (++e) + "</a>"
                }
            } else {
                a = [S.current + 1, S.lang.of, l].join(" ")
            }
        }
        get("sb-counter").innerHTML = a;
        m()
    }
    function showBars(d) {
        var a = get("sb-title-inner"), c = get("sb-info-inner"), b = 0.35;
        a.style.visibility = c.style.visibility = "";
        if (a.innerHTML != "") {
            animate(a, "marginTop", 0, b)
        }
        animate(c, "marginTop", 0, b, d)
    }
    function hideBars(c, i) {
        var g = get("sb-title"), a = get("sb-info"), d = g.offsetHeight, e = a.offsetHeight, f = get("sb-title-inner"), h = get("sb-info-inner"), b = (c ? 0.35 : 0);
        animate(f, "marginTop", d, b);
        animate(h, "marginTop", e*-1, b, function() {
            f.style.visibility = h.style.visibility = "hidden";
            i()
        })
    }
    function adjustHeight(a, d, b, f) {
        var e = get("sb-wrapper-inner"), c = (b ? S.options.resizeDuration : 0);
        animate(wrapper, "top", d, c);
        animate(e, "height", a, c, f)
    }
    function adjustWidth(a, d, b, e) {
        var c = (b ? S.options.resizeDuration : 0);
        animate(wrapper, "left", d, c);
        animate(wrapper, "width", a, c, e)
    }
    function setDimensions(i, c) {
        var e = get("sb-body-inner"), i = parseInt(i), c = parseInt(c), b = wrapper.offsetHeight - e.offsetHeight, a = wrapper.offsetWidth - e.offsetWidth, g = overlay.offsetHeight, h = overlay.offsetWidth, f = parseInt(S.options.viewportPadding) || 20, d = (S.player && S.options.handleOversize != "drag");
        return S.setDimensions(i, c, g, h, b, a, f, d)
    }
    var K = {};
    K.markup = '<div id="sb-container"><div id="sb-overlay"></div><div id="sb-wrapper"><div id="sb-title"><div id="sb-title-inner"></div></div><div id="sb-info"><div id="sb-info-inner"><div id="sb-counter"></div><div id="sb-nav"><a id="sb-nav-close" title="{close}" onclick="Shadowbox.close()"></a><a id="sb-nav-next" title="{next}" onclick="Shadowbox.next()"></a><a id="sb-nav-play" title="{play}" onclick="Shadowbox.play()"></a><a id="sb-nav-pause" title="{pause}" onclick="Shadowbox.pause()"></a><a id="sb-nav-previous" title="{previous}" onclick="Shadowbox.previous()"></a></div></div></div><div id="sb-wrapper-inner"><div id="sb-body"><div id="sb-body-inner"></div><div id="sb-loading"><div id="sb-loading-inner"><span></span></div></div></div></div></div></div>';
    K.options = {
        animSequence: "sync",
        counterLimit: 10,
        counterType: "default",
        displayCounter: true,
        displayNav: true,
        fadeDuration: 0.35,
        initialHeight: 160,
        initialWidth: 320,
        modal: false,
        overlayColor: "#000",
        overlayOpacity: 0.5,
        resizeDuration: 0.35,
        showOverlay: true,
        troubleElements: ["select", "object", "embed", "canvas"]
    };
    K.init = function() {
        S.appendHTML(document.body, sprintf(K.markup, S.lang));
        K.body = get("sb-body-inner");
        container = get("sb-container");
        overlay = get("sb-overlay");
        wrapper = get("sb-wrapper");
        if (!supportsFixed) {
            container.style.position = "absolute"
        }
        if (!supportsOpacity) {
            var c, a, b = /url\("(.*\.png)"\)/;
            each(pngIds, function(e, f) {
                c = get(f);
                if (c) {
                    a = S.getStyle(c, "backgroundImage").match(b);
                    if (a) {
                        c.style.backgroundImage = "none";
                        c.style.filter = "progid:DXImageTransform.Microsoft.AlphaImageLoader(enabled=true,src=" + a[1] + ",sizingMethod=scale);"
                    }
                }
            })
        }
        var d;
        addEvent(window, "resize", function() {
            if (d) {
                clearTimeout(d);
                d = null
            }
            if (open) {
                d = setTimeout(K.onWindowResize, 10)
            }
        })
    };
    K.onOpen = function(a, c) {
        doWindowResize = false;
        container.style.display = "block";
        setSize();
        var b = setDimensions(S.options.initialHeight, S.options.initialWidth);
        adjustHeight(b.innerHeight, b.top);
        adjustWidth(b.width, b.left);
        if (S.options.showOverlay) {
            overlay.style.backgroundColor = S.options.overlayColor;
            S.setOpacity(overlay, 0);
            if (!S.options.modal) {
                addEvent(overlay, "click", S.close)
            }
            overlayOn = true
        }
        if (!supportsFixed) {
            setPosition();
            addEvent(window, "scroll", setPosition)
        }
        toggleTroubleElements();
        container.style.visibility = "visible";
        if (overlayOn) {
            animate(overlay, "opacity", S.options.overlayOpacity, S.options.fadeDuration, c)
        } else {
            c()
        }
    };
    K.onLoad = function(b, a) {
        toggleLoading(true);
        while (K.body.firstChild) {
            remove(K.body.firstChild)
        }
        hideBars(b, function() {
            if (!open) {
                return 
            }
            if (!b) {
                wrapper.style.visibility = "visible"
            }
            buildBars(a)
        })
    };
    K.onReady = function(d) {
        if (!open) {
            return 
        }
        var b = S.player, c = setDimensions(b.height, b.width);
        var a = function() {
            showBars(d)
        };
        switch (S.options.animSequence) {
        case"hw":
            adjustHeight(c.innerHeight, c.top, true, function() {
                adjustWidth(c.width, c.left, true, a)
            });
            break;
        case"wh":
            adjustWidth(c.width, c.left, true, function() {
                adjustHeight(c.innerHeight, c.top, true, a)
            });
            break;
        default:
            adjustWidth(c.width, c.left, true);
            adjustHeight(c.innerHeight, c.top, true, a)
        }
    };
    K.onShow = function(a) {
        toggleLoading(false, a);
        doWindowResize = true
    };
    K.onClose = function() {
        if (!supportsFixed) {
            removeEvent(window, "scroll", setPosition)
        }
        removeEvent(overlay, "click", S.close);
        wrapper.style.visibility = "hidden";
        var a = function() {
            container.style.visibility = "hidden";
            container.style.display = "none";
            toggleTroubleElements(true)
        };
        if (overlayOn) {
            animate(overlay, "opacity", 0, S.options.fadeDuration, a)
        } else {
            a()
        }
    };
    K.onPlay = function() {
        toggleNav("play", false);
        toggleNav("pause", true)
    };
    K.onPause = function() {
        toggleNav("pause", false);
        toggleNav("play", true)
    };
    K.onWindowResize = function() {
        if (!doWindowResize) {
            return 
        }
        setSize();
        var a = S.player, b = setDimensions(a.height, a.width);
        adjustWidth(b.width, b.left);
        adjustHeight(b.innerHeight, b.top);
        if (a.onWindowResize) {
            a.onWindowResize()
        }
    };
    S.skin = K; // expose
    window['Shadowbox'] = S;

})(window);

