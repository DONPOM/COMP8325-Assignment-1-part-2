(function(e) {
    "use strict";
    var t = e.GOC = e.GOC || {};
    t.queue = t.queue || [], t.opts = t.opts || {}
})(this), GOC.load = function(e, t) {
    "use strict";
    var n = document, r = n.getElementsByTagName("script")[0], i = n.createElement("script");
    i.type = "text/javascript", i.async = "async", i.onload = i.onreadystatechange = function(e, n) {
        var r = this, i = r.readyState, s = r.parentNode;
        s && (n ||!i || i === "complete" || i === "loaded") && (s.removeChild(this), !n && t && t())
    }, i.src = e, r.parentNode.insertBefore(i, r), i = n = r = null
}, function(e) {
    "use strict";
    var t = e.encodeURIComponent;
    e.GOC.load("//a.dilcdn.com/g/domains/" + t(e.location.hostname) + ".js")
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = {}, r = Array.prototype.slice;
    t.on = function(e, t, r) {
        var i = n[e] || (n[e] = []);
        return t && i.push({
            f: t,
            c: r
        }), this
    }, t.off = function(e, t, r) {
        var i, s, o = 0, u;
        if (!e)
            n = {};
        else if (!t)
            delete n[e];
        else if (i = n[e]) {
            s = n[e] = [];
            for (; o < i.length; ++o)
                u = i[o], u.f === t && (r ? u.c === r : u.c === this) && s.push(u)
        }
        return this
    }, t.trigger = function(e) {
        var t = n[e], i = 0, s, o = r.call(arguments, 1);
        if (t) {
            t = t.slice();
            for (; i < t.length; ++i)
                s = t[i], s.f.apply(s.c || this, o)
        }
        e !== "all" && this.trigger.apply(this, ["all", e].concat(o))
    }
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = function(e) {
        var n = e.apply ? e: t[e[0]];
        n && n.apply && n.apply(t, e.slice ? e.slice(1) : [])
    };
    e.setTimeout(function() {
        var e = t.queue || [], r = 0;
        for (; r < e.length; ++r)
            n(e[r]);
        t.queue = {
            push: n
        }
    }, 1)
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = e.encodeURIComponent;
    t.openCds = function(r) {
        e.location.href = "http://search.disney.com/goc/us/home/cds?url=" + n(t.go(r)) + "&js=1"
    }
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = e.document, r = n.documentElement, i = String.prototype.trim, s = Array.prototype.splice, o = Array.prototype.slice, u = i || function() {
        return this.toString().replace(/^\s+|\s+$/g, "")
    }, a = function(t) {
        var n = this.length || 0, r = 0, i = 0;
        if (t)
            if (t.nodeType || t === e)
                r = 1, this[n] = t;
            else {
                r = t.length || 0;
                for (; i < r; ++i)
                    this[n + i] = t[i]
            }
        this.length = n + r
    }, f = t.dom = function(e) {
        return new a(e)
    }, l = function(e) {
        var t = this, n = e.target || e.srcElement || r, i = e.fromElement;
        n.nodeType === 3 && (n = n.parentNode), t.type = e.type, t.target = n, t.original = e, t.altKey=!!e.altKey, t.metaKey=!!e.metaKey, t.shiftKey=!!e.shiftKey, t.ctrlKey=!!e.ctrlKey, t.charCode = e.charCode, t.keyCode = e.keyCode, t.relatedTarget = e.relatedTarget || (i === n ? e.toElement : i), t.defaultPrevented=!!(e.defaultPrevented || e.returnValue===!1 || e.getPreventDefault && e.getPreventDefault())
    };
    l.prototype = {
        preventDefault: function() {
            var e = this.original;
            this.defaultPrevented=!0, e.preventDefault ? e.preventDefault() : e.returnValue=!1
        },
        stopPropagation: function() {
            var e = this.original;
            e.stopPropagation ? e.stopPropagation() : e.cancelBubble=!0
        }
    }, a.prototype = {
        splice: s,
        add: a,
        ea: function(e) {
            var t = 0, n = this.length;
            for (; t < n; ++t)
                e.call(this[t], t);
            return this
        },
        eaN: function(e) {
            var t = 0, n = this.length, r;
            for (; t < n; ++t)
                r = this[t], r.nodeType === 1 && e.call(r, t);
            return this
        },
        before: function(e) {
            return this.ea(function() {
                e.parentNode.insertBefore(this, e)
            })
        },
        after: function(e) {
            var t = e.nextSibling;
            return t ? this.before(t) : this.ea(function() {
                e.parentNode.appendChild(this)
            })
        },
        rm: function() {
            return this.ea(function() {
                var e = this.parentNode;
                e && e.removeChild(this)
            }), this
        },
        on: function(t, n) {
            var r = o.call(arguments, 2);
            return this.eaN(function() {
                var i = this, s, o = function(i) {
                    i = i || e.event || {
                        type: t
                    }, s && s.call(this, i), n.apply(this, [new l(i)].concat(r))
                };
                i.addEventListener ? i.addEventListener(t, o, !1) : (s = i["on" + t], i["on" + t] = o), i = null
            })
        },
        hasC: function(e) {
            var t = 0, n = this.length, r;
            for (; t < n; ++t) {
                r = this[t];
                if (r.nodeType === 1 && (" " + r.className + " ").indexOf(" " + e + " ")>-1)
                    return !0
            }
            return !1
        },
        addC: function(e) {
            var t = e;
            return e = " " + e + " ", this.eaN(function() {
                var n = " " + (this.className || "") + " ";
                n.indexOf(e) < 0 && (this.className = u.call(n + t))
            })
        },
        rmC: function(e) {
            return e = " " + e + " ", this.eaN(function() {
                var t = " " + (this.className || "") + " ";
                while (t.indexOf(e)>-1)
                    t = t.replace(e, " ");
                this.className = u.call(t)
            })
        },
        byTag: function(e) {
            var t = f();
            return this.eaN(function() {
                t.add(this.getElementsByTagName(e))
            }), t
        },
        byClass: function(e) {
            var t = f();
            return this.eaN(function() {
                t.add(n.querySelectorAll("." + e))
            }), t
        }
    }, f.byId = function(e) {
        return f(n.getElementById(e))
    }, f.byTag = function(e) {
        return f(r).byTag(e)
    }, f.parse = function(e) {
        var t = n.createElement("div");
        return t.innerHTML = e, f(t.childNodes)
    }, f.contains = r.compareDocumentPosition ? function(e, t) {
        return !!(e.compareDocumentPosition(t) & 16)
    } : r.contains ? function(e, t) {
        var n = e.nodeType === 9 ? e.documentElement: e, r = t && t.parentNode;
        return e === r||!!(r && r.nodeType === 1 && n.contains && n.contains(r))
    } : function(e, t) {
        while (t = t.parentNode)
            if (t === e)
                return !0;
        return !1
    }, f.create = function(e) {
        return f(n.createElement(e))
    }
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = t.dom, r = e.encodeURIComponent, i = /^([^\/]+:)?\/\/([^\/]+)\//, s = /(?:^|.)([a-z0-9\-]+?(?:\.com?)?\.[a-z\-]+)$/, o = /(?:^|;)\s*SWID=([^;]+)/, u = {
        "go.com": "r.disney.",
        "disney.com": "r."
    }, a = function(e) {
        return (s.exec(e) || [e, e])[1]
    }, f = function(t) {
        var n = i.exec(t), s = (o.exec(e.document.cookie) || [])[1], f = e.location, l, c;
        return s && n && (n[1] || (t = f.protocol + t), f = f.hostname, l = a(f), n = a(n[2].toLowerCase()), u[n] && n !== l && (c = "http://" + u[n] + n + "/go?", t.substr(0, c.length) !== c && (t = c + "url=" + r(t) + "&dom=" + r(l) + "&swid=" + r(s)))), t
    }, l = function(e) {
        var t = e.target, n;
        while (t) {
            if (t.nodeName === "A" && (n = t.href)) {
                t.href = f(n);
                break
            }
            t = t.parentNode
        }
    };
    t.go = function(e) {
        return typeof e == "string" ? f(e) : (n(e).on("keypress", l).on("mousedown", l), e)
    }
}(this), GOC.css = function(e) {
    "use strict";
    var t = document, n = t.createElement("style"), r = t.getElementsByTagName("head")[0], i = (r || t).getElementsByTagName("script")[0];
    e && (n.setAttribute("type", "text/css"), i ? i.parentNode.insertBefore(n, i) : r.appendChild(n), n.styleSheet ? n.styleSheet.cssText = e : n.appendChild(t.createTextNode(e)))
}, function(e, t) {
    "use strict";
    var n = e.GOC, r = n.dom, i = "goc-logged-in", s = "goc-logged-out", o = "goc-user", u = ["login", "logout"], a, f = n.user = function(e) {
        var t = r.byId(o);
        t && (t.rmC(i), t.rmC(s), e===!1 || e === "false" || e === "logout" ? t.addC(s) : e && t.addC(i)), a = e
    }, l = function(e, t) {
        r(e).hasC("goc-" + t) && r(e).byTag("span").on("click", function() {
            n.trigger("click"), n.trigger("click:" + t)
        })
    }, c = function() {
        var e = r.byId(o).byTag("li"), i;
        e.ea(function() {
            for (i = 0; i < u.length; ++i)
                l(this, u[i])
        }), f(a !== t ? a : n.opts.user), n.off("accept:base", c)
    };
    n.on("accept:base", c)
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = e.document, r = n.documentElement, i = function(e, t) {
        var i, s, o, u = n.createElement("div"), a = n.body, f = a || n.createElement("body");
        return i = '&#173;<style id="goc-test-s">' + e + "</style>", u.id = "goc-test", (a ? u : f).innerHTML = i, f.appendChild(u), a || (f.style.background = "", f.style.overflow = "hidden", o = r.style.overflow, r.style.overflow = "hidden", r.appendChild(f)), s = t(u, e), a ? u.parentNode.removeChild(u) : (f.parentNode.removeChild(f), r.style.overflow = o), !!s
    }, s = t.mq = function(t) {
        var n = e.matchMedia || e.msMatchMedia;
        return n ? n(t).matches : i("@media " + t + "{#goc-test{position:absolute;}}", function(t) {
            var n = e.getComputedStyle;
            return (n ? n(t, null) : t.currentStyle).position === "absolute"
        })
    }, o = t.s = {
        du: !0,
        mq: s("only all")
    }, u = new e.Image;
    u.onload = u.onerror = function() {
        if (this.width !== 1 || this.height !== 1)
            r.className += " goc-no-data-uri", o.du=!1;
        u = null
    }, u.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = t.dom, r = "mtt_chrome", i = function(e) {
        return e = "" + e, e = e.replace(/[^\w\s\-]/g, ""), e = e.replace(/^\s+|\s+$/g, ""), e = e.replace(/[\s\-]+/g, "_"), e = e.toLowerCase(), e
    }, s = function(t) {
        var n = e.cto, r = e._gaq;
        n && n.trackLink(t);
        var i = t.linkName;
        t.href && (i = i + ":" + t.href), r && r.push(["_trackEvent", "Chrome: " + t.linkPosition, "Click", i])
    }, o = function(e, t, n) {
        var r = i(e.textContent || e.innerText), o = t + "/" + (n + 1) + "/none/link/" + r;
        s({
            linkName: o,
            linkPosition: t
        })
    }, u = function(e, t) {
        var r = n.byId(e).byTag("a");
        r.on("click", function() {
            var e = this;
            r.ea(function(n) {
                this === e && o(e, t, n)
            })
        })
    }, a = function(e) {
        var t = e.getAttribute("data-linkname"), n = e.getAttribute("data-linkposition"), r = e.getAttribute("data-linkinput");
        return t && n ? r ? [t, n, r] : [t, n] : !1
    }, f = {};
    t.on("change:search", function(t, n) {
        var r = e.cto;
        if (r&&!f[t]) {
            f[t]=!0;
            var i = {
                searchEvent: "attempt",
                engagementType: t,
                searchFlow: n
            };
            r.trackEvent(i)
        }
    }), t.on("change:search_phrase", function(r) {
        var i = e.cto;
        if (r && i) {
            var s = r.value, o = r.id, u = "autoComplete";
            n(r).hasC("instant-search-input") && t.is && (u = "instantSearch"), i.trackEvent({
                searchEvent: u,
                searchPhraseInput: s,
                searchPhrase: s,
                searchFlow: o
            })
        }
    }), t.on("accept:base", function() {
        u("goc-bar", "mtt_chrome")
    }), t.on("click:is_filter", function(t, n) {
        var r = e.cto;
        r && r.trackLink({
            searchRefinement: t,
            searchFlow: n
        })
    }), t.on("submit:search", function(t) {
        if (t === r) {
            var n = e.cto;
            n && n.trackLink("mtt_chrome/11/none/link/search_enter", "mtt_chrome")
        }
    }), t.on("click:search", function(e, t, i, o) {
        var u;
        e && (u = a(e.target)), u || (u = a(t)), n(t).hasC("all") && (o=!1);
        if (u[0] && u[0].indexOf(r) === 0) {
            var f = {
                linkName: u[0],
                linkPosition: u[1],
                href: t.href,
                searchPhraseInput: u[2],
                searchFlow: i
            };
            o && (f.searchEvent = "instantResultsClick"), s(f)
        }
    }), t.on("accept:menu", function() {
        u("goc-menu", "flyawaynav")
    })
}(this), function(e, t) {
    "use strict";
    var n = e.GOC, r = n.dom, i = e.document, s = r(i.documentElement), o = e.encodeURIComponent, u = e.location.protocol === "https:", a, f = "", l, c, h, p = "mtt_chrome", d = n.closeSearch = function() {
        var i = r.byId("goc-ac").rm();
        e.clearTimeout(c), a = f = l = t, i.length && (s.rmC("goc-open-search"), s.hasC("goc-open-menu") || (s.rmC("goc-open"), n.trigger("close")), n.trigger("close:search"))
    }, v = function(e) {
        var i = a && a.value;
        c = t;
        if (i !== t && i !== f) {
            f = i;
            if (i === "")
                d();
            else {
                var s = r(a).hasC("instant-search-input") && n.is ? n.is: n.a;
                n.load(s + o(i) + "?" + (u ? "s" : "") + "&p=" + e), n.trigger("change:search", e, a.id)
            }
        }
    }, m = function(e) {
        var t = r.byId("goc-ac").byTag("li"), n = t.length - 1;
        n >= 0 && (l = e, l < 0 && (l = n), l > n && (l = 0), t.rmC("goc-active"), r(t[l]).addC("goc-active"))
    }, g = function(e) {
        m(l !== t ? l + e : e > 0 ? 0 : - 1)
    }, y = function(e, t, r, i) {
        n.trigger("click"), n.trigger("click:search", e, this, r, i)
    }, b = {
        13: function(t) {
            var i = r(r.byId("goc-ac").byTag("li")[l]).byTag("a")[0];
            return i ? (y.call(i, a.value, a.id), e.location.href = i.href) : n.trigger("submit:search", t), d(), !!i
        },
        27: function() {
            if (a)
                return d(), !0
        },
        38: function() {
            return g( - 1), !0
        },
        40: function() {
            return g(1), !0
        }
    }, w = function(r, i) {
        var s = b[r.keyCode];
        a = this, s ? s.call(this, i) && r.preventDefault() : (c || (c = e.setTimeout(function() {
            v(i)
        }, 500)), h && e.clearTimeout(h), h = e.setTimeout(function() {
            h = t, n.trigger("change:search_phrase", a)
        }, 500))
    };
    n.ac = function(e, t) {
        return r(e).byTag("form").ea(function() {
            var e = r(this);
            e.hasC("goc-search") && e.byTag("input").ea(function() {
                this.name === "q" && (t = t || p, this.setAttribute("autocomplete", "off"), r(this).on("keydown", w, t))
            })
        }), e
    }, n._aIsf = function(e, t) {
        var i = r.byId("search-is-inner");
        n.go(r.parse(t)).before(i[0]), i.rm(), r.byId("search-is-inner").byTag("a").on("click", y, e, a.id, !0)
    }, n._aIs = function(e, t) {
        var i = r.byId("search-is"), s = r.parse(t);
        i.length > 0 ? (s.addC("show"), n.go(s).before(a), i.rm()) : (n.go(s).before(a), s.addC("show")), E(s, e)
    };
    var E = function(e, t) {
        e.on("touchstart", function() {
            r.byTag("input").ea(function() {
                this.blur()
            })
        }), e.byTag("a").on("click", y, t, a.id, !0), e.byClass("isFilter").ea(function() {
            r(this).on("click", function(t) {
                t.preventDefault(), t.stopPropagation();
                var i = t.target;
                if (i === this || r.contains(this, i))
                    r(e).byClass("isFilter").ea(function() {
                        r(this).rmC("active")
                    }), r(this).addC("active"), S(this.href);
                n.trigger("click:is_filter", this.href.split("f%5Bsearch_section%5D=")[1], a.id)
            })
        })
    }, S = function(e) {
        var r = {};
        e.replace(new RegExp("([^?=&]+)(=([^&]*))?", "g"), function(e, t, n, i) {
            r[t] = i
        }), r["f%5Bsearch_section%5D"] !== t ? n.load(n.is + r.q + "?" + (u ? "s" : "") + "&p=mtt_chrome&f%5Bsearch_section%5D=" + r["f%5Bsearch_section%5D"]) : n.load(n.is + r.q + "?" + (u ? "s" : "") + "&p=mtt_chrome")
    };
    n._aSac = function(e, i) {
        var o = r.byId("goc-ac").rm();
        o.rm(), o = o.length, a && (n.go(r.parse(i)).after(a).byTag("li").ea(function(n) {
            r(this).on("mouseover", function(e) {
                var t = e.target;
                (t === this || r.contains(this, t)) && m(n)
            }).on("mouseout", function(e) {
                var n = e.relatedTarget;
                (!n || n !== this&&!r.contains(this, n)) && m(t)
            }).byTag("a").on("click", y, e, a.id)
        }), o ? m(l) : (s.hasC("goc-open-menu") || n.trigger("open"), n.trigger("open:search"), s.addC("goc-open-search"), s.addC("goc-open")), n.trigger("accept:search"))
    }, s.on("click", function(t) {
        var n = t.target;
        a && n !== a && e.setTimeout(function() {
            d(), r.byId("search-is").rm()
        }, 0)
    })
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = t.dom, r, i, s = "goc-placeholder";
    "placeholder"in e.document.createElement("input") ? t.pl = function(e) {
        return e
    } : (r = function() {
        this.value.length < 1 && (this.value = this.getAttribute("placeholder"), n(this).addC(s))
    }, i = function() {
        var e = n(this);
        this.value === this.getAttribute("placeholder") && e.hasC(s) && (this.value = "", e.rmC(s))
    }, t.pl = function(e) {
        return e.byTag("input").eaN(function() {
            var e = this.getAttribute("placeholder");
            this.type === "text" && e && e.length > 0 && (n(this).on("focus", i).on("blur", r), r.call(this))
        }), e
    })
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = t.dom, r = e.document, i = /^[a-z0-9]+$/i, s = t.be = function(e) {
        var n = t.opts;
        return n.bg && i.exec(n.bg) && e.addC("goc-bg-" + n.bg), e
    };
    t._aBase = function(e, i, o, u, a) {
        var f = n.byId("goc");
        f.length || (f = n.create("div"), f.before(r.body.firstChild)), s(n.parse(e)).before(f[0]), f.rm(), t._aFooter(a), t.s.mq ? (o && (i += "@media all and (min-width: 1025px){" + o + "}"), u && (i += "@media only all and (max-width:1024px){" + u + "}"), t.css(i)) : t.css((i || "") + (o || "")), t.trigger("accept:base"), delete t._aBase
    }, t._aFooter = function(e, r) {
        var i = n.byId("goc-footer");
        i.length && (i[0].innerHTML = e, t.css(r || ""), t.trigger("accept:footer"), delete t._aFooter)
    }, t.on("accept:base", function() {
        var e = n.byId("goc-bar");
        t.opts.nav && (n.byId("goc-nav").addC("goc-nav"), n.byId("goc-e").on("click", function() {
            t.openMenu()
        })), t.ac(t.pl(t.go(e)))
    })
}(this), function(e, t) {
    "use strict";
    var n = e.GOC, r = n.dom, i = e.document, s = r(i.documentElement), o = e.setTimeout, u, a, f, l = function() {
        var e = "Moz Webkit O ms".split(" "), t = i.createElement("div").style;
        if ("transition"in t)
            return !0;
        for (var n = 0; n < e.length; ++n)
            if (e[n] + "Transition"in t)
                return !0
    }() ? 250: 1;
    n.oo = function(e, r, i) {
        var l = "goc-open-" + e;
        s.hasC(l) || (u && (s.rmC("goc-open-" + u), s.rmC("goc-animating-" + u), n.trigger("close:" + u)), a=!!i, u = e, f=!0, s.addC("goc-animating-" + e), o(function() {
            f = t, s[r ? "addC": "rmC"]("goc-masked"), s.addC("goc-open"), s.addC(l)
        }, 0), n.trigger("open"), n.trigger("open:" + e), s[0].clientHeight)
    }, n.co = function(e, r) {
        if (e === u) {
            u = t, s.addC("goc-close"), s.rmC("goc-masked"), s.rmC("goc-open"), s.rmC("goc-open-" + e);
            var i = function() {
                s.rmC("goc-animating-" + e), s.rmC("goc-close"), n.trigger("close"), n.trigger("close:" + e)
            };
            !l || r ? i() : o(i, l)
        }
    }, n.close = function() {
        u && n.co(u)
    }, s.on("click", function(e) {
        if (!f&&!a && u) {
            var t = e.target, i;
            while (i = t.parentNode) {
                if (r(i).hasC("goc-overlay"))
                    return;
                t = i
            }
            n.co(u)
        }
    }), s.on("keydown", function(e) {
        !e.defaultPrevented && u && e.keyCode === 27 && (e.preventDefault(), n.co(u))
    }), n.on("accept:base", function() {
        var e = r.create("div");
        e[0].id = "goc-mask", e[0].onclick = function() {}, e.before((i.getElementById("goc-body") || i.body).firstChild)
    })
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = t.dom, r = e.document, i = n(r.documentElement), s = t.openMenu = function() {
        t.oo("menu", !0)
    }, o = t.closeMenu = function() {
        t.co("menu")
    }, u = function(e) {
        e.preventDefault(), i.hasC("goc-open-menu") ? o() : s()
    };
    t._sMenu = function(e) {
        t.on("accept:base", function() {
            t.load(t.ao(e))
        }), delete t._sMenu
    }, t._aMenu = function(e, s, o, a) {
        t.s.mq ? (o && (s += "@media all and (min-width: 1025px){" + o + "}"), a && (s += "@media only all and (max-width:1024px){" + a + "}"), t.css(s)) : t.css((s || "") + (o || "")), t.go(t.ac(t.pl(t.be(n.parse(e))))).before(r.body.firstChild), n.byId("goc-button").on("click", u).length && n.byId("goc-bar").addC("goc-under-button"), i.on("keydown", function(e) {
            e.keyCode === 77 && e.shiftKey && e.ctrlKey&&!e.metaKey&&!e.altKey && u(e)
        });
        var f = n.byId("goc-button")[0].offsetWidth + "px";
        n.byId("goc-desktop-global")[0].style.paddingLeft = f, t.trigger("accept:menu"), delete t._aMenu
    }
}(this), function(e) {
    "use strict";
    var t = e.GOC, n = t.dom, r = function() {
        return t.mq("all and (max-width: 1024px)")
    }, i = function() {
        n.byId("goc-menu").byTag("dt").rmC("goc-open")
    }, s = function() {
        n.byId("goc-menu").byTag("dt").eaN(function() {
            var e = n(this);
            e.byTag("a").hasC("goc-active") && e.addC("goc-open")
        })
    }, o = function() {
        r() && e.scrollTo(0, 0)
    };
    t.isMobile = function(e) {
        var t = r();
        return e && e(t), t
    }, t.on("open:menu", i), t.on("open:menu", s), t.on("open:menu", o), t.on("close:menu", o), t.on("accept:menu", function() {
        var s = /\bAndroid\s+(\d[0-9\.]+\d)\b/.exec(e.navigator.userAgent), o = n(e.document.documentElement), u = e.matchMedia, a, f = n.byId("goc-menu");
        s && e.parseFloat(s[1]) < 2.2 && o.addC("goc-fixed-fix"), f.byTag("dt").on("click", function(e) {
            var t = n(this);
            t.hasC("goc-expandable") && r() && (e.preventDefault(), t.hasC("goc-open") ? t.rmC("goc-open") : (i(), t.addC("goc-open")))
        }).ea(function() {
            var e = this;
            do 
                e = e.nextSibling;
            while (e && e.nodeType !== 1);
            e && e.nodeName === "DD" && n(e).byTag("a").length && n(this).addC("goc-expandable")
        }), u && (a = u("all and (min-width: 1025px)")) && a.addListener && a.addListener(function() {
            t.co("menu", !0)
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.GOC;
    t.on("accept:menu", function() {
        function i() {
            if (n.byId("goc-body")[0])
                return;
            var e = n.byId("goc-menu")[0], t = [];
            n(e.parentNode.childNodes).ea(function() {
                this.id !== "goc-menu" && t.push(this)
            });
            var r = n.create("div")[0];
            r.id = "goc-body", n(r).after(e), n(t).ea(function() {
                r.appendChild(this)
            })
        }
        var n = t.dom, r = e.document;
        r.readyState === "interactive" || r.readyState === "complete" ? i() : r.addEventListener("DOMContentLoaded", i)
    })
}(this);
