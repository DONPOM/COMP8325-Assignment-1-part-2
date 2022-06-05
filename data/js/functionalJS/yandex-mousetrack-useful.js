function cp(t, e, n) {
    var i = "clck/lid=" + t;
    Lego.params.statRoot && (i += "/sid=" + Lego.params.msid);
    try {
        if (home && home.getData && home.getData("mordaFromCache.data"))
            return window.workerMessage("counter", {
                path: i
            }), !0;
        Lego.c(i, e, n)
    } catch (o) {}
}
function cpr(t, e, n) {
    try {
        Lego.cred(t, e, n)
    } catch (i) {}
}
function cpt(t, e) {
    t += (document.getElementById("form").text.value ? ".full" : ".empty") + "/u=" + (new Date).getTime(), cp(t, e)
}
function c0(t) {
    try {
        Lego.c(t)
    } catch (e) {}
}
function csh(t, e) {
    try {
        cp(t, e)
    } catch (n) {}
}
function csh_ifmsid(t, e) {
    Lego.params.statRoot && csh(t, e)
}
function cpa(t) {
    return t && "undefined" != typeof Lego.params.msid ? "onmousedown=\"Lego.cp('132','71946','" + t + "',this)\"" : ""
}
function statRedirect(t, e) {
    location.href = e && Lego.params.statRoot ? location.protocol + "//clck.yandex.ru/redir/dtype=clck/lid=" + e + "/sid=" + Lego.params.msid + "/rnd=" + ((new Date).getTime() + Math.round(100 * Math.random())) + "/*data=" + encodeURIComponent("url=" + encodeURIComponent(t.match(/^http/) ? t : location.protocol + "//" + location.host + (t.match("^/") ? t : "/" + t))) : t
}
function getcss(t) {
    var e = document.createElement("link");
    e.setAttribute("type", "text/css"), e.setAttribute("rel", "stylesheet"), e.setAttribute("href", t), document.getElementsByTagName("head").item(0).appendChild(e)
}
function tabs(t, e, n, i) {
    var o = "b-tabs", s = "b-tabs__selected", r = "b-tabs__notselected", a = "b-tabs__names", c = "b-tabs__items", u = "i-hidden";
    if ($(e).parent().is("." + s))
        return csh_ifmsid(n + ".link", e), !0;
    var l = $(e).closest("." + o);
    if ($("ul." + a + " > li", l).removeClass(s).addClass(r), $(e).closest("li").removeClass(r).addClass(s), $("div." + c + " > div", l).addClass(u), $("#" + t).removeClass(u), csh_ifmsid(n + ".select"), e.blur(), i) {
        var d = $(".b-tabs__frame");
        d.length || (d = $('<iframe class="b-tabs__frame" style="display:none"></iframe>'), $(document.body).append(d)), d.attr("src", i)
    }
    return !1
}
function CAWBrowser() {
    for (var t = [], e = 6; 15 > e; e++)
        t[e]=!1;
    var n = navigator.userAgent;
    this.msie = n && parseFloat(navigator.appVersion) >= 4 && n.indexOf("Opera") < 0 && n.indexOf("MSIE 4") < 0 && n.indexOf("MSIE") >= 0, this.win = n && (n.indexOf("Windows 95") >= 0 || n.indexOf("Windows NT") >= 0 || n.indexOf("Windows 98") >= 0), this.mac = navigator.platform&&-1 !== navigator.platform.indexOf("Mac"), this.opera7 =- 1 !== n.indexOf("Opera") && window.opera && document.readyState ? 1 : 0, this.gecko =- 1 !== n.toLowerCase().indexOf("gecko")&&-1 === n.indexOf("safari");
    var i = navigator.mimeTypes && navigator.mimeTypes["application/x-shockwave-flash"] ? navigator.mimeTypes["application/x-shockwave-flash"].enabledPlugin: 0;
    if (i)
        for (e = 6; 15 > e; e++)
            t[e] = i, t[e] = parseInt(t[e].description.substring(t[e].description.indexOf(".") - 2), 10) >= e;
    else if (this.msie && this.win&&!this.mac)
        for (e = 6; 15 > e; e++)
            try {
                t[e] = new ActiveXObject("ShockwaveFlash.ShockwaveFlash." + e)
    } catch (o) {}
    for (this.other=!((this.gecko || this.msie) && this.win&&!this.mac), this.flash = 0, e = 6; 15 > e; e++)
        t[e] && (this.flash = e)
}
function MordaRT() {
    this.listRE = {}
}
function workerMessage(t, e) {
    var n = home && home.getData && home.getData("worker-message.data");
    return n ? (window.chrome && window.chrome.runtime && window.chrome.runtime.sendMessage(n.extensionId, {
        to: "big_" + n.locale + "." + n.mordaZone,
        data: {
            type: "frommorda",
            message: t,
            params: e
        }
    }), !0) : !1
}
var xglobal = "undefined" != typeof global ? global: this;
!function(t, e) {
    "use strict";
    var n = t[e] = function() {}, i=!0;
    n.ns = function(t, e, n) {
        return t[e] || (t[e] = {}), "undefined" != typeof n && n.call(t[e]), this
    }, n.getEmptyString = function() {
        return ""
    }, n.stringf = function() {
        var t = arguments[0], e = Array.prototype.slice.call(arguments, 1);
        return t.replace(/%s/g, function() {
            return e.shift() || ""
        })
    }, n.generateId = function() {
        return String(Math.random()).substr(2)
    }, n.map = function(t, e) {
        return n.isArray(t) && e instanceof Function ? t.map(e) : void 0
    }, n.console = {
        slog: function() {
            this.log(n.stringf.apply(this, arguments))
        },
        log: function(e) {
            "undefined" != typeof debug ? debug(e) : t.console && t.console.log && t.console.log(e)
        },
        error: function(t) {
            throw new Error(t)
        },
        dir: function(t) {
            this.log(JSON.stringify(t))
        }
    }, n.log = n.console.log, n.error = n.console.error;
    var o = Array.prototype.filter ? function(t, e) {
        return t.filter(e)
    }
    : function(t, e) {
        var n = [];
        return x.each(t, function(t, i) {
            e(t, i, this) && n.push(t)
        }), n
    };
    n.filter = function(t, e) {
        return n.isArray(t) || n.error("first argument is expected to be an array instead of " + typeof t), n.isFunc(e) || n.error("second argument is expected to be a function instead of " + typeof e), o(t, e)
    };
    var s = Array.prototype.some ? function(t, e) {
        return t.some(e)
    }
    : function(t, e) {
        for (var n = 0, i = t.length; i > n; n++)
            if (e(t[n], n, t))
                return !0;
        return !1
    };
    n.some = function(t, e) {
        return n.isArray(t) || n.error("first argument is expected to be an array instead of " + typeof t), n.isFunc(e) || n.error("second argument is expected to be a function instead of " + typeof e), s(t, e)
    }, n.isArray = function(t) {
        return t instanceof Array
    }, n.isArrayLike = function(t) {
        return n.isObject(t)&&!n.isArray(t) && "length"in t
    }, n.isObject = function(t) {
        return "object" == typeof t
    }, n.eachListItem = function(t, e, n) {
        var i, o;
        if (n)
            for (i = 0, o = t.length; o > i; i++)
                e.call(n, t[i], i, t);
        else 
            for (i = 0, o = t.length; o > i; i++)
                e(t[i], i, t);
        return this
    }, n.eachProperty = function(t, e, n) {
        var i;
        if (n)
            for (i in t)
                t.hasOwnProperty(i) && e.call(n, t[i], i, t);
        else 
            for (i in t)
                t.hasOwnProperty(i) && e(t[i], i, t);
        return this
    }, n.trimRe = (new RegExp).compile(/^\s+|\s+$/g), n.trim = function(t) {
        var e = typeof t;
        return "string" !== e && n.error("Argument must be string but" + e + " passed"), t.trim instanceof Function ? t.trim() : t.replace(n.trimRe, "")
    }, x.stripSpaces = function(t) {
        return t.split(" ").join("")
    }, n.each = n.forEach = function(t, e, i) {
        var o = n.isArray(t) || n.isArrayLike(t) ? n.eachListItem: n.eachProperty;
        return o(t, e, i), this
    }, n.isFunction = n.isFunc = function() {
        return Array.prototype.every.call(arguments, function(t) {
            return t instanceof Function
        })
    }, n.isString = function() {
        return Array.prototype.every.call(arguments, function(t) {
            return "string" == typeof t
        })
    }, n.bind = function(t, e) {
        return function() {
            return t.apply(e, arguments)
        }
    }, n.beget = function(t, e) {
        var i, o = typeof t, s = typeof e;
        if ("object" !== o && n.error("Only object allowed to be prototype and tried to work with " + o), "undefined" !== s && "object" !== s && n.error("Only object allowed to be list of properties instead " + s), "function" == typeof Object.create)
            return i = {}, x.each(e, function(t, e) {
                i[e] = {
                    writable: !0,
                    enumerable: !0,
                    value: t
                }
            }), Object.create(t, i);
        var r = function() {};
        return t && "object" == typeof t && (r.prototype = t), i = new r, "undefined" != typeof e && x.each(e, function(t, e) {
            i[e] = t
        }), i
    }, n.Interface = function(t, e) {
        var i = this;
        2 !== arguments.length && n.error("Interface constructor called with " + arguments.length + "parameters, but expected exactly 2"), this.methods = [], this.name = t, x.each(e, function(t) {
            "string" != typeof t && n.error("Interface constructor expects method names to be passed in as a string"), i.methods.push(t)
        })
    }, n.Interface.ensureImplements = function(t, e) {
        2 !== arguments.length && n.error("Interface constructor called with " + arguments.length + "parameters, but expected at least 2"), x.each(e, function(e) {
            e.constructor !== n.Interface && n.error("Function Interface.ensureImplements expects arguments two and above to be instances of Interface."), x.each(e.methods, function(i) {
                t[i] && "function" == typeof t[i] || n.error("Function Interface.ensureImplements: object does not implement the " + e.name + " interface. Method " + i + " was not found.")
            })
        })
    }, n.extend = function(t, e) {
        var n = function() {};
        t.superclass = n.prototype = e.prototype, t.prototype = new n, t.prototype.constructor = t, e.prototype.constructor === Object.prototype.constructor && (e.prototype.constructor = e)
    }, n.getJPathValue = function(t, e) {
        for (var n = t, i = e.split("."), o = 0; n[i[o]];)
            n = n[i[o++]];
        return n
    }, n.create = function(t) {
        var e = function() {};
        return e.prototype = t, new e
    }, i && (n.ViewItem = new x.Interface("View", ["getCustomRender", "ruleFunction", "let", "get"])), n.View = function(t) {
        this.templates = t && t.templates ? x.beget(t.templates) : {}
    }, n.View.prototype = {
        templates: {},
        re: {
            parse: /\[%\s([^%]+)\s%\]/g,
            rule: /\s*:\s*/
        },
        getCustomRender: function() {
            var t = "undefined" !== arguments[0] ? arguments[0].split(this.re.rule): null;
            return t && 2 === t.length ? this.parseRules[t[0]] && this.parseRules[t[0]].call(this, t[1]) : void 0
        },
        parseRules: {
            l10n: function(t) {
                return function(e) {
                    return i&&!e.l10n(t) && n.console.error("There is no translation for " + t), e.l10n ? e.l10n(t) : t
                }
            },
            jpath: function(t) {
                return function(e) {
                    return x.getJPathValue(e, t) || ""
                }
            },
            view: function(t) {
                var e = this;
                return function(n) {
                    return e.get(t, n)
                }
            }
        },
        ruleFunction: function(t, e) {
            return this.getCustomRender(t) || function(n) {
                var o = n[t] || e[t];
                return i&&!o && x.console.log("Undefined variable" + t), o || ""
            }
        },
        get: function(t, e) {
            var i, o = this.templates[t];
            return i = n.isArray(o) ? x.map(o, function(t) {
                return n.isString(t) ? t : t(e)
            }).join("") : n.isString(o) ? o : o(e)
        }
    }, n.View.prototype.let = function(t, e, i) {
        var o = t in this.templates, s = [], r = t in this.templates ? this.templates[t]: null, a = this, c = 0;
        return n.isString(e) && (s = this.templates[t] = r ? x.create(r) : [], o && (s._super = r), e.replace(this.re.parse, function(t, n, o) {
            s.push(e.substr(c, o - c), a.ruleFunction(n, i)), c = o + t.length
        }), c < e.length && s.push(e.substr(c))), n.isFunc(e) && (this.templates[t] = e), this
    }, n.views = new n.View, n.getObject = function(t) {
        return t.beget = function(t) {
            return x.beget(this, t)
        }, x.beget(t)
    }, n.URLre = /^https?:\/\/\S+/i, n.isURL = function(t) {
        return "string" != typeof t && n.error("isURL expects string parameter instead " + typeof t), n.URLre.test(t)
    };
    var r = new n.Interface("intComponent", ["init", "data"]);
    n.ComponentBase = {
        data: function(t) {
            var e = this.getNodeElement();
            return e ? n.data(e, t) : void 0
        },
        getNodeElement: function() {
            var t = n.byId(this.id);
            return t.length ? t[0] : t
        },
        setId: function(t) {
            this.id = t
        }
    }, n.Component = function() {
        var t = [], e = {}, i = "data-xapp";
        return {
            extend: function(i) {
                var o = t.length, s = x.beget(n.ComponentBase, i);
                return n.Interface.ensureImplements(s, [r]), s.id || n.error("extend method expects id field in description object"), t.push(s), e[s.id] = o, t[o]
            },
            initByHTML: function(t) {
                var e = t || document, o = this, s = n.byAttr(i, e);
                x.each(s, function(t) {
                    var e = n.data(t, "xapp");
                    x.each(e.split(" "), function(e) {
                        o.initById(e, n.getElementId(t))
                    })
                })
            },
            initById: function(n, i) {
                var o = n in e ? t[e[n]]: null;
                o && ("undefined" != typeof i && o.setId(i), o.init(i))
            }
        }
    }();
    var a = function(t) {
        for (var e = document.all, n = [], i = 0, o = e.length; o > i; ++i)
            "string" == typeof e[i].getAttribute(t) && n.push(e[i]);
        return n
    };
    n.byAttr = function(e) {
        var n = arguments[1] || t.document || null;
        if (n && n.querySelectorAll)
            try {
                return Array.prototype.slice.call(n.querySelectorAll("[" + e + "]"))
        } catch (i) {}
        return a(e)
    }, n.byId = function(e) {
        var n = arguments[1] || t.document || null;
        return n && n.getElementById(e) || null
    };
    var c = 1, u = function(t) {
        return t && t.nodeType === c
    };
    n.data = function(t, e) {
        return u(t) || n.error("nodeElement expected as first argument instead of " + typeof t), t.getAttribute("data-" + e)
    }, n.getElementId = function(t) {
        u(t) || n.error("nodeElement expected as first argument instead of " + typeof t);
        var e = t.getAttribute("id");
        return e || (e = "_" + n.generateId(), t.setAttribute("id", e)), e
    }, n.constructURL = function(t, e) {
        var n, i = "", o = [], s = t.match(/((?:https?\:)?(?:\/)?\/[^?]+)\?([^#]+)?(\#\S+)?/), r = s && s[3] ? s[3]: "";
        if (s && s[2])
            for (var a = s[2].split("&"), c = a.length; c--;)
                o.push(a[c]);
        for (n in e)
            e.hasOwnProperty(n) && o.push(n + "=" + e[n]);
        return i = s ? s[1] + "?" + o.join("&") + r : t + "?" + o.join("&") + r
    }, n.toArray = function(t) {
        return Array.prototype.slice.call(t)
    }, function() {
        var t = {}, e = {};
        n.amd = {
            define: function(n, i, o) {
                return o = o ||!0, t[n] = i, o && (e[n]=!0), this
            },
            require: function() {
                var i = n.toArray(arguments).slice(0, arguments.length - 1), o = arguments[arguments.length - 1];
                return i = n.map(i, function(i) {
                    if (!t[i])
                        throw new Error("Unmet dependency:" + i);
                    return e[i] ? (t[i] = n.isFunction(t[i]) ? t[i]() : t[i], t[i]) : t[i]()
                }), o.apply(this, i), this
            }
        }
    }(), n.debounce = function(t, e, n) {
        var i;
        return function() {
            var o = this, s = arguments, r = function() {
                i = null, n || t.apply(o, s)
            }, a = n&&!i;
            clearTimeout(i), i = setTimeout(r, e), a && t.apply(o, s)
        }
    }, n.amd.define("statlog", function() {
        var t = null, e = {
            type: "show",
            root: "testRoot",
            statDataAttr: "statlog",
            dataNoRedirKey: "statlog-noredir",
            doc: document || null,
            sendAtWill: !0,
            dataReadyKey: "statlog-ready",
            clickEvent: "mousedown",
            skipVisibleCheck: !1,
            statEnabled: !1,
            clickCounter: function(t, n) {
                n = n || null, window.cp(e.root + "." + t, n)
            }
        }, i = function(e) {
            var n = [];
            return t && (n = JSON.stringify({
                event: e.type,
                parentPath: e.root,
                blocks: t.children
            })), n
        }, o = function() {
            return "undefined" != typeof Lego && Lego.params ? Lego.params.statEnabled : e.statEnabled
        }, s = function() {
            $(e.doc).on(e.clickEvent, "*[data-" + e.statDataAttr + "]", function(t) {
                if (o()) {
                    var n = $(this).data(e.statDataAttr), i = $(this).data(e.dataNoRedirKey);
                    n && (e.clickCounter(n, i ? null : this), i && (t.preventDefault(), t.stopPropagation()))
                }
            })
        };
        return s(), {
            params: e,
            isEnabled: o,
            log: function(i) {
                if ("undefined" == typeof i)
                    return "";
                var o = t ? t: t = {}, s = this;
                return this.isEnabled() && (n.forEach(i.split("."), function(t) {
                    var e = function(e) {
                        return e.ctag === t
                    };
                    o.children || (o.children = []);
                    var i = n.filter(o.children, e);
                    i.length || o.children.push({
                        ctag: t
                    }), o = n.filter(o.children, e)[0]
                }), e.sendAtWill && s.send()), this
            },
            logByDocument: function(t) {
                e = n.beget(e, t);
                var i = this, o = "data-" + e.statDataAttr, s = o.substr(5);
                $(e.doc).find("*[" + o + "]").each(function() {
                    if ($(this).is(":visible") || e.skipVisibleCheck) {
                        if (!o.match(/^data-/))
                            throw new Error("data attribute expected as statAttr");
                        var t = $(this).data(s);
                        $(this).data(e.dataReadyKey) || (i.log(t), $(this).data(e.dataReadyKey, t))
                    }
                })
            },
            dumpDATA: i,
            send: n.debounce(function() {
                t && $.ajax({
                    type: "POST",
                    url: "http://clck.yandex.ru/counter/*",
                    data: {
                        dtype: "clck",
                        session_id: Lego.params.msid,
                        events: i(e)
                    },
                    success: function() {
                        t = null
                    }
                })
            }, 1e3)
        }
    })
}(xglobal, "x");
var xglobal = "undefined" != typeof global ? global: this;
!function(t, e) {
    "use strict";
    var n = t[e] = function() {}, i=!0;
    n.ns = function(t, e, n) {
        return t[e] || (t[e] = {}), "undefined" != typeof n && n.call(t[e]), this
    }, n.getEmptyString = function() {
        return ""
    }, n.stringf = function() {
        var t = arguments[0], e = Array.prototype.slice.call(arguments, 1);
        return t.replace(/%s/g, function() {
            return e.shift() || ""
        })
    }, n.generateId = function() {
        return String(Math.random()).substr(2)
    }, n.map = function(t, e) {
        return n.isArray(t) && e instanceof Function ? t.map(e) : void 0
    }, n.console = {
        slog: function() {
            this.log(n.stringf.apply(this, arguments))
        },
        log: function(e) {
            "undefined" != typeof debug ? debug(e) : t.console && t.console.log && t.console.log(e)
        },
        error: function(t) {
            throw new Error(t)
        },
        dir: function(t) {
            this.log(JSON.stringify(t))
        }
    }, n.log = n.console.log, n.error = n.console.error;
    var o = Array.prototype.filter ? function(t, e) {
        return t.filter(e)
    }
    : function(t, e) {
        var n = [];
        return x.each(t, function(t, i) {
            e(t, i, this) && n.push(t)
        }), n
    };
    n.filter = function(t, e) {
        return n.isArray(t) || n.error("first argument is expected to be an array instead of " + typeof t), n.isFunc(e) || n.error("second argument is expected to be a function instead of " + typeof e), o(t, e)
    };
    var s = Array.prototype.some ? function(t, e) {
        return t.some(e)
    }
    : function(t, e) {
        for (var n = 0, i = t.length; i > n; n++)
            if (e(t[n], n, t))
                return !0;
        return !1
    };
    n.some = function(t, e) {
        return n.isArray(t) || n.error("first argument is expected to be an array instead of " + typeof t), n.isFunc(e) || n.error("second argument is expected to be a function instead of " + typeof e), s(t, e)
    }, n.isArray = function(t) {
        return t instanceof Array
    }, n.isArrayLike = function(t) {
        return n.isObject(t)&&!n.isArray(t) && "length"in t
    }, n.isObject = function(t) {
        return "object" == typeof t
    }, n.eachListItem = function(t, e, n) {
        var i, o;
        if (n)
            for (i = 0, o = t.length; o > i; i++)
                e.call(n, t[i], i, t);
        else 
            for (i = 0, o = t.length; o > i; i++)
                e(t[i], i, t);
        return this
    }, n.eachProperty = function(t, e, n) {
        var i;
        if (n)
            for (i in t)
                t.hasOwnProperty(i) && e.call(n, t[i], i, t);
        else 
            for (i in t)
                t.hasOwnProperty(i) && e(t[i], i, t);
        return this
    }, n.trimRe = (new RegExp).compile(/^\s+|\s+$/g), n.trim = function(t) {
        var e = typeof t;
        return "string" !== e && n.error("Argument must be string but" + e + " passed"), t.trim instanceof Function ? t.trim() : t.replace(n.trimRe, "")
    }, x.stripSpaces = function(t) {
        return t.split(" ").join("")
    }, n.each = n.forEach = function(t, e, i) {
        var o = n.isArray(t) || n.isArrayLike(t) ? n.eachListItem: n.eachProperty;
        return o(t, e, i), this
    }, n.isFunction = n.isFunc = function() {
        return Array.prototype.every.call(arguments, function(t) {
            return t instanceof Function
        })
    }, n.isString = function() {
        return Array.prototype.every.call(arguments, function(t) {
            return "string" == typeof t
        })
    }, n.bind = function(t, e) {
        return function() {
            return t.apply(e, arguments)
        }
    }, n.beget = function(t, e) {
        var i, o = typeof t, s = typeof e;
        if ("object" !== o && n.error("Only object allowed to be prototype and tried to work with " + o), "undefined" !== s && "object" !== s && n.error("Only object allowed to be list of properties instead " + s), "function" == typeof Object.create)
            return i = {}, x.each(e, function(t, e) {
                i[e] = {
                    writable: !0,
                    enumerable: !0,
                    value: t
                }
            }), Object.create(t, i);
        var r = function() {};
        return t && "object" == typeof t && (r.prototype = t), i = new r, "undefined" != typeof e && x.each(e, function(t, e) {
            i[e] = t
        }), i
    }, n.Interface = function(t, e) {
        var i = this;
        2 !== arguments.length && n.error("Interface constructor called with " + arguments.length + "parameters, but expected exactly 2"), this.methods = [], this.name = t, x.each(e, function(t) {
            "string" != typeof t && n.error("Interface constructor expects method names to be passed in as a string"), i.methods.push(t)
        })
    }, n.Interface.ensureImplements = function(t, e) {
        2 !== arguments.length && n.error("Interface constructor called with " + arguments.length + "parameters, but expected at least 2"), x.each(e, function(e) {
            e.constructor !== n.Interface && n.error("Function Interface.ensureImplements expects arguments two and above to be instances of Interface."), x.each(e.methods, function(i) {
                t[i] && "function" == typeof t[i] || n.error("Function Interface.ensureImplements: object does not implement the " + e.name + " interface. Method " + i + " was not found.")
            })
        })
    }, n.extend = function(t, e) {
        var n = function() {};
        t.superclass = n.prototype = e.prototype, t.prototype = new n, t.prototype.constructor = t, e.prototype.constructor === Object.prototype.constructor && (e.prototype.constructor = e)
    }, n.getJPathValue = function(t, e) {
        for (var n = t, i = e.split("."), o = 0; n[i[o]];)
            n = n[i[o++]];
        return n
    }, n.create = function(t) {
        var e = function() {};
        return e.prototype = t, new e
    }, i && (n.ViewItem = new x.Interface("View", ["getCustomRender", "ruleFunction", "let", "get"])), n.View = function(t) {
        this.templates = t && t.templates ? x.beget(t.templates) : {}
    }, n.View.prototype = {
        templates: {},
        re: {
            parse: /\[%\s([^%]+)\s%\]/g,
            rule: /\s*:\s*/
        },
        getCustomRender: function() {
            var t = "undefined" !== arguments[0] ? arguments[0].split(this.re.rule): null;
            return t && 2 === t.length ? this.parseRules[t[0]] && this.parseRules[t[0]].call(this, t[1]) : void 0
        },
        parseRules: {
            l10n: function(t) {
                return function(e) {
                    return i&&!e.l10n(t) && n.console.error("There is no translation for " + t), e.l10n ? e.l10n(t) : t
                }
            },
            jpath: function(t) {
                return function(e) {
                    return x.getJPathValue(e, t) || ""
                }
            },
            view: function(t) {
                var e = this;
                return function(n) {
                    return e.get(t, n)
                }
            }
        },
        ruleFunction: function(t, e) {
            return this.getCustomRender(t) || function(n) {
                var o = n[t] || e[t];
                return i&&!o && x.console.log("Undefined variable" + t), o || ""
            }
        },
        get: function(t, e) {
            var i, o = this.templates[t];
            return i = n.isArray(o) ? x.map(o, function(t) {
                return n.isString(t) ? t : t(e)
            }).join("") : n.isString(o) ? o : o(e)
        }
    }, n.View.prototype.let = function(t, e, i) {
        var o = t in this.templates, s = [], r = t in this.templates ? this.templates[t]: null, a = this, c = 0;
        return n.isString(e) && (s = this.templates[t] = r ? x.create(r) : [], o && (s._super = r), e.replace(this.re.parse, function(t, n, o) {
            s.push(e.substr(c, o - c), a.ruleFunction(n, i)), c = o + t.length
        }), c < e.length && s.push(e.substr(c))), n.isFunc(e) && (this.templates[t] = e), this
    }, n.views = new n.View, n.getObject = function(t) {
        return t.beget = function(t) {
            return x.beget(this, t)
        }, x.beget(t)
    }, n.URLre = /^https?:\/\/\S+/i, n.isURL = function(t) {
        return "string" != typeof t && n.error("isURL expects string parameter instead " + typeof t), n.URLre.test(t)
    };
    var r = new n.Interface("intComponent", ["init", "data"]);
    n.ComponentBase = {
        data: function(t) {
            var e = this.getNodeElement();
            return e ? n.data(e, t) : void 0
        },
        getNodeElement: function() {
            var t = n.byId(this.id);
            return t.length ? t[0] : t
        },
        setId: function(t) {
            this.id = t
        }
    }, n.Component = function() {
        var t = [], e = {}, i = "data-xapp";
        return {
            extend: function(i) {
                var o = t.length, s = x.beget(n.ComponentBase, i);
                return n.Interface.ensureImplements(s, [r]), s.id || n.error("extend method expects id field in description object"), t.push(s), e[s.id] = o, t[o]
            },
            initByHTML: function(t) {
                var e = t || document, o = this, s = n.byAttr(i, e);
                x.each(s, function(t) {
                    var e = n.data(t, "xapp");
                    x.each(e.split(" "), function(e) {
                        o.initById(e, n.getElementId(t))
                    })
                })
            },
            initById: function(n, i) {
                var o = n in e ? t[e[n]]: null;
                o && ("undefined" != typeof i && o.setId(i), o.init(i))
            }
        }
    }();
    var a = function(t) {
        for (var e = document.all, n = [], i = 0, o = e.length; o > i; ++i)
            "string" == typeof e[i].getAttribute(t) && n.push(e[i]);
        return n
    };
    n.byAttr = function(e) {
        var n = arguments[1] || t.document || null;
        if (n && n.querySelectorAll)
            try {
                return Array.prototype.slice.call(n.querySelectorAll("[" + e + "]"))
        } catch (i) {}
        return a(e)
    }, n.byId = function(e) {
        var n = arguments[1] || t.document || null;
        return n && n.getElementById(e) || null
    };
    var c = 1, u = function(t) {
        return t && t.nodeType === c
    };
    n.data = function(t, e) {
        return u(t) || n.error("nodeElement expected as first argument instead of " + typeof t), t.getAttribute("data-" + e)
    }, n.getElementId = function(t) {
        u(t) || n.error("nodeElement expected as first argument instead of " + typeof t);
        var e = t.getAttribute("id");
        return e || (e = "_" + n.generateId(), t.setAttribute("id", e)), e
    }, n.constructURL = function(t, e) {
        var n, i = "", o = [], s = t.match(/((?:https?\:)?(?:\/)?\/[^?]+)\?([^#]+)?(\#\S+)?/), r = s && s[3] ? s[3]: "";
        if (s && s[2])
            for (var a = s[2].split("&"), c = a.length; c--;)
                o.push(a[c]);
        for (n in e)
            e.hasOwnProperty(n) && o.push(n + "=" + e[n]);
        return i = s ? s[1] + "?" + o.join("&") + r : t + "?" + o.join("&") + r
    }, n.toArray = function(t) {
        return Array.prototype.slice.call(t)
    }, function() {
        var t = {}, e = {};
        n.amd = {
            define: function(n, i, o) {
                return o = o ||!0, t[n] = i, o && (e[n]=!0), this
            },
            require: function() {
                var i = n.toArray(arguments).slice(0, arguments.length - 1), o = arguments[arguments.length - 1];
                return i = n.map(i, function(i) {
                    if (!t[i])
                        throw new Error("Unmet dependency:" + i);
                    return e[i] ? (t[i] = n.isFunction(t[i]) ? t[i]() : t[i], t[i]) : t[i]()
                }), o.apply(this, i), this
            }
        }
    }(), n.debounce = function(t, e, n) {
        var i;
        return function() {
            var o = this, s = arguments, r = function() {
                i = null, n || t.apply(o, s)
            }, a = n&&!i;
            clearTimeout(i), i = setTimeout(r, e), a && t.apply(o, s)
        }
    }, n.amd.define("statlog", function() {
        var t = null, e = {
            type: "show",
            root: "testRoot",
            statDataAttr: "statlog",
            dataNoRedirKey: "statlog-noredir",
            doc: document || null,
            sendAtWill: !0,
            dataReadyKey: "statlog-ready",
            clickEvent: "mousedown",
            skipVisibleCheck: !1,
            statEnabled: !1,
            clickCounter: function(t, n) {
                n = n || null, window.cp(e.root + "." + t, n)
            }
        }, i = function(e) {
            var n = [];
            return t && (n = JSON.stringify({
                event: e.type,
                parentPath: e.root,
                blocks: t.children
            })), n
        }, o = function() {
            return "undefined" != typeof Lego && Lego.params ? Lego.params.statEnabled : e.statEnabled
        }, s = function() {
            $(e.doc).on(e.clickEvent, "*[data-" + e.statDataAttr + "]", function(t) {
                if (o()) {
                    var n = $(this).data(e.statDataAttr), i = $(this).data(e.dataNoRedirKey);
                    n && (e.clickCounter(n, i ? null : this), i && (t.preventDefault(), t.stopPropagation()))
                }
            })
        };
        return s(), {
            params: e,
            isEnabled: o,
            log: function(i) {
                if ("undefined" == typeof i)
                    return "";
                var o = t ? t: t = {}, s = this;
                return this.isEnabled() && (n.forEach(i.split("."), function(t) {
                    var e = function(e) {
                        return e.ctag === t
                    };
                    o.children || (o.children = []);
                    var i = n.filter(o.children, e);
                    i.length || o.children.push({
                        ctag: t
                    }), o = n.filter(o.children, e)[0]
                }), e.sendAtWill && s.send()), this
            },
            logByDocument: function(t) {
                e = n.beget(e, t);
                var i = this, o = "data-" + e.statDataAttr, s = o.substr(5);
                $(e.doc).find("*[" + o + "]").each(function() {
                    if ($(this).is(":visible") || e.skipVisibleCheck) {
                        if (!o.match(/^data-/))
                            throw new Error("data attribute expected as statAttr");
                        var t = $(this).data(s);
                        $(this).data(e.dataReadyKey) || (i.log(t), $(this).data(e.dataReadyKey, t))
                    }
                })
            },
            dumpDATA: i,
            send: n.debounce(function() {
                t && $.ajax({
                    type: "POST",
                    url: "http://clck.yandex.ru/counter/*",
                    data: {
                        dtype: "clck",
                        session_id: Lego.params.msid,
                        events: i(e)
                    },
                    success: function() {
                        t = null
                    }
                })
            }, 1e3)
        }
    })
}(xglobal, "x"), x.Component.extend({
    id: "widgets",
    init: function() {
        var t = ya.widgetsParams, e = t.pageType, n = {
            wauth: t.auth,
            layoutFixed: !t.editMode,
            addUrl: "/adddata/",
            patternId: t.patternid,
            yu: t.yu,
            domain: t.domain,
            locale: t.locale
        };
        e && (n[e + "_prototype"] = e), t.region && (n.region = t.region), t.geoid && (n.geoid = Number(t.geoid)), Widget.Init(n), wg.init(), wg.wa = n.wauth, t.settings && (wg.psettings = t.settings), wg.hpattern = 1, wg.editmode = t.editMode ? 1 : 0, wg.logged = t.Logged, wg.InMoscowRegion = t.InMoscowRegion, wg.isUkraine = t.NaUkraine, wg.statLogRoot = t.statLogRoot, wg.layoutType = t.layoutType, wg.hidePromo = t.hidePromo, wg.psettings.columnsCount || (wg.psettings.columnsCount = 5), e && (wg._prototype = e)
    }
}), $(function() {
    x.Component.initByHTML()
}), x.ns(this, "ya"), x.ns(this, "home"), ya.passRequest = function(t) {
    if (t) {
        var e = $("<iframe />");
        e.css({
            position: "absolute",
            height: "0",
            width: "0",
            left: "0",
            top: "0",
            visibility: "hidden"
        }), e.attr("src", t), e.appendTo(document.body)
    }
}, home.delayManager = function() {
    var t = {}, e = {}, n = function(n, i) {
        t[n] || (t[n] = []), e[n] ? i() : t[n].push(i)
    }, i = function(e) {
        if (t[e])
            for (var n = t[e].length, i = 0; n > i; i++)
                t[e].shift().call()
    };
    return {
        add: function(t, e) {
            n(t, e)
        },
        runQueue: function(t) {
            e[t]=!0, i(t)
        }
    }
}(), home.debounce = function(t, e, n) {
    var i, o, s, r, a;
    return function() {
        s = this, o = arguments, r = new Date;
        var c = function() {
            var u = new Date - r;
            e > u ? i = setTimeout(c, e - u) : (i = null, n || (a = t.apply(s, o)))
        }, u = n&&!i;
        return i || (i = setTimeout(c, e)), u && (a = t.apply(s, o)), a
    }
}, function(t) {
    function e(e, i, o) {
        var r=!1;
        if (s) {
            var a = [];
            t.each(c, function() {
                o.hasOwnProperty(this) && (r=!0) && a.push({
                    name: this,
                    val: o[this]
                })
            }), r && (t.each(o, function(t) {
                a.push({
                    name: t,
                    val: this
                })
            }), o = a)
        }
        t.each(o, function(o, s) {
            if (r && (o = s.name, s = s.val), t.isFunction(s) && (!n || s.toString().indexOf(".__base")>-1)) {
                var a = e[o] || function() {};
                i[o] = function() {
                    var t = this.__base;
                    this.__base = a;
                    var e = s.apply(this, arguments);
                    return this.__base = t, e
                }
            } else 
                i[o] = s
        })
    }
    var n = function() {}.toString().indexOf("_")>-1, i = function() {}, o = Object.create || function(t) {
        var e = function() {};
        return e.prototype = t, new e
    }, s=!0, r = {
        toString: ""
    };
    for (var a in r)
        r.hasOwnProperty(a) && (s=!1);
    var c = s ? ["toString", "valueOf"]: null;
    t.inherit = function() {
        var n = arguments, s = t.isFunction(n[0]), r = s ? n[0]: i, a = n[s ? 1: 0] || {}, c = n[s ? 2: 1], u = a.__constructor || s && r.prototype.__constructor ? function() {
            return this.__constructor.apply(this, arguments)
        }
        : function() {};
        if (!s)
            return u.prototype = a, u.prototype.__self = u.prototype.constructor = u, t.extend(u, c);
        t.extend(u, r);
        var l = r.prototype, d = u.prototype = o(l);
        return d.__self = d.constructor = u, e(l, d, a), c && e(r, u, c), u
    }, t.inheritSelf = function(t, n, i) {
        var o = t.prototype;
        return e(o, o, n), i && e(t, t, i), t
    }
}(jQuery), function(t) {
    var e = 0, n = "__" + + new Date, i = function() {
        return "uniq" + ++e
    };
    t.identify = function(t, e) {
        if (!t)
            return i();
        var o = "uniqueID"in t ? "uniqueID": n;
        return e || o in t ? t[o] : t[o] = i()
    }
}(jQuery), function(t) {
    t.isEmptyObject || (t.isEmptyObject = function(t) {
        for (var e in t)
            return !1;
        return !0
    })
}(jQuery), function(t) {
    t.extend({
        debounce: function(t, e, n, i) {
            3 == arguments.length && "boolean" != typeof n && (i = n, n=!1);
            var o;
            return function() {
                var s = arguments;
                i = i || this, n&&!o && t.apply(i, s), clearTimeout(o), o = setTimeout(function() {
                    n || t.apply(i, s), o = null
                }, e)
            }
        },
        throttle: function(t, e, n) {
            var i, o, s;
            return function() {
                o = arguments, s=!0, n = n || this, i || function() {
                    s ? (t.apply(n, o), s=!1, i = setTimeout(arguments.callee, e)) : i = null
                }()
            }
        }
    })
}(jQuery), function(t) {
    var e = "__" + + new Date + "storage", n = function(e, n) {
        return t.identify(e) + (n ? t.identify(n) : "")
    }, i = {
        buildEventName: function(t) {
            return t
        },
        on: function(i, o, s, r, a) {
            if ("string" == typeof i) {
                t.isFunction(o) && (r = s, s = o, o = void 0);
                for (var c, u = n(s, r), l = this[e] || (this[e] = {}), d = i.split(" "), h = 0; i = d[h++];)
                    if (i = this.buildEventName(i), c = l[i] || (l[i] = {
                        ids: {},
                        list: {}
                    }), !(u in c.ids)) {
                        var f = c.list, p = {
                            fn: s,
                            data: o,
                            ctx: r,
                            special: a
                        };
                        f.last ? (f.last.next = p, p.prev = f.last) : f.first = p, c.ids[u] = f.last = p
                    }
            } else {
                var m = this;
                t.each(i, function(t, e) {
                    m.on(t, e, o, a)
                })
            }
            return this
        },
        onFirst: function(t, e, n, i) {
            return this.on(t, e, n, i, {
                one: !0
            })
        },
        un: function(i, o, s) {
            if ("string" == typeof i || "undefined" == typeof i) {
                var r = this[e];
                if (r)
                    if (i) {
                        for (var a, c = i.split(" "), u = 0; i = c[u++];)
                            if (i = this.buildEventName(i), a = r[i])
                                if (o) {
                                    var l = n(o, s), d = a.ids;
                                    if (l in d) {
                                        var h = a.list, f = d[l], p = f.prev, m = f.next;
                                        p ? p.next = m : f === h.first && (h.first = m), m ? m.prev = p : f === h.last && (h.last = p), delete d[l]
                                    }
                                } else 
                                    delete this[e][i]
                    } else 
                        delete this[e]
            } else {
                var _ = this;
                t.each(i, function(t, e) {
                    _.un(t, e, s)
                })
            }
            return this
        },
        trigger: function(n, i) {
            var o, s = this, r = s[e];
            if ("string" == typeof n ? n = t.Event(s.buildEventName(o = n)) : n.type = s.buildEventName(o = n.type), n.target || (n.target = s), r && (r = r[n.type]))
                for (var a, c = r.list.first; c;)
                    n.data = c.data, a = c.fn.call(c.ctx || s, n, i), "undefined" != typeof a && (n.result = a, a===!1 && (n.preventDefault(), n.stopPropagation())), c.special && c.special.one && s.un(o, c.fn, c.ctx), c = c.next;
            return this
        }
    };
    t.observable = t.inherit(i, i)
}(jQuery), function(t, e) {
    function n(t, e, n) {
        return (t ? "__elem_" + t : "") + "__mod" + (e ? "_" + e : "") + (n ? "_" + n : "")
    }
    function i(e, i, o) {
        t.isFunction(e) ? i[n(o, "*", "*")] = e : t.each(e, function(e, s) {
            t.isFunction(s) ? i[n(o, e, "*")] = s : t.each(s, function(t, s) {
                i[n(o, e, t)] = s
            })
        })
    }
    function o(t, e) {
        return e ? Array.isArray(e) ? function(n) {
            for (var i = 0, o = e.length; o > i;)
                if (n.hasMod(t, e[i++]))
                    return !0;
            return !1
        } : function(n) {
            return n.hasMod(t, e)
        } : function(e) {
            return e.hasMod(t)
        }
    }
    var s = [], r = {}, a = {};
    this.BEM = t.inherit(t.observable, {
        __constructor: function(t, e, n) {
            var i = this;
            i._modCache = t || {}, i._processingMods = {}, i._params = e, i.params = null, n!==!1 ? i._init() : i.afterCurrentEvent(function() {
                i._init()
            })
        },
        _init: function() {
            return this._initing || this.hasMod("js", "inited") || (this._initing=!0, this.params || (this.params = t.extend(this.getDefaultParams(), this._params), delete this._params), this.setMod("js", "inited"), delete this._initing, this.hasMod("js", "inited") && this.trigger("init")), this
        },
        changeThis: function(t, e) {
            return t.bind(e || this)
        },
        afterCurrentEvent: function(t, e) {
            this.__self.afterCurrentEvent(this.changeThis(t, e))
        },
        trigger: function(t, e) {
            return this.__base(t = this.buildEvent(t), e).__self.trigger(t, e), this
        },
        buildEvent: function(e) {
            return "string" == typeof e && (e = t.Event(e)), e.block = this, e
        },
        hasMod: function(t, n, i) {
            var o = arguments.length, s=!1;
            1 == o ? (i = "", n = t, t = e, s=!0) : 2 == o && ("string" == typeof t ? (i = n, n = t, t = e) : (i = "", s=!0));
            var r = this.getMod(t, n) === i;
            return s?!r : r
        },
        getMod: function(t, e) {
            var n = typeof t;
            if ("string" === n || "undefined" === n) {
                e = t || e;
                var i = this._modCache;
                return e in i ? i[e] : i[e] = this._extractModVal(e)
            }
            return this._getElemMod(e, t)
        },
        _getElemMod: function(t, e, n) {
            return this._extractModVal(t, e, n)
        },
        getMods: function(t) {
            var n = t && "string" != typeof t, i = this, o = [].slice.call(arguments, n ? 1 : 0), s = i._extractMods(o, n ? t : e);
            return n || (o.length ? o.forEach(function(t) {
                i._modCache[t] = s[t]
            }) : i._modCache = s), s
        },
        setMod: function(n, i, o) {
            "undefined" == typeof o && (o = i, i = n, n = e);
            var s = this;
            if (!n || n[0]) {
                var r = (n && n[0] ? t.identify(n[0]) : "") + "_" + i;
                if (this._processingMods[r])
                    return s;
                var a, c = n ? s._getElemMod(i, n, a = s.__self._extractElemNameFrom(n)): s.getMod(i);
                if (c === o)
                    return s;
                this._processingMods[r]=!0;
                var u=!0, l = [i, o, c];
                n && l.unshift(n), [["*", "*"], [i, "*"], [i, o]].forEach(function(t) {
                    u = s._callModFn(a, t[0], t[1], l)!==!1 && u
                }), !n && u && (s._modCache[i] = o), u && s._afterSetMod(i, o, c, n, a), delete this._processingMods[r]
            }
            return s
        },
        _afterSetMod: function() {},
        toggleMod: function(t, n, i, o, s) {
            "string" == typeof t && (s = o, o = i, i = n, n = t, t = e), "undefined" == typeof o ? o = "" : "boolean" == typeof o && (s = o, o = "");
            var r = this.getMod(t, n);
            return (r == i || r == o) && this.setMod(t, n, "boolean" == typeof s ? s ? i : o : this.hasMod(t, n, i) ? o : i), this
        },
        delMod: function(t, n) {
            return n || (n = t, t = e), this.setMod(t, n, "")
        },
        _callModFn: function(t, i, o, s) {
            var r = n(t, i, o);
            return this[r] ? this[r].apply(this, s) : e
        },
        _extractModVal: function() {
            return ""
        },
        _extractMods: function() {
            return {}
        },
        channel: function(t, e) {
            return this.__self.channel(t, e)
        },
        getDefaultParams: function() {
            return {}
        },
        del: function(t) {
            var e = [].slice.call(arguments);
            return "string" == typeof t && e.unshift(this), this.__self.del.apply(this.__self, e), this
        },
        destruct: function() {}
    }, {
        _name: "i-bem",
        blocks: r,
        decl: function(n, s, a) {
            if ("string" == typeof n ? n = {
                block: n
            } : n.name && (n.block = n.name), n.baseBlock&&!r[n.baseBlock])
                throw 'baseBlock "' + n.baseBlock + '" for "' + n.block + '" is undefined';
            s || (s = {}), s.onSetMod && (i(s.onSetMod, s), delete s.onSetMod), s.onElemSetMod && (t.each(s.onElemSetMod, function(t, e) {
                i(e, s, t)
            }), delete s.onElemSetMod);
            var c = r[n.baseBlock || n.block] || this;
            if (n.modName) {
                var u = o(n.modName, n.modVal);
                t.each(s, function(n, i) {
                    t.isFunction(i) && (s[n] = function() {
                        var t;
                        if (u(this))
                            t = i;
                        else {
                            var o = c.prototype[n];
                            o && o !== s[n] && (t = this.__base)
                        }
                        return t ? t.apply(this, arguments) : e
                    })
                })
            }
            if (a && "boolean" == typeof a.live) {
                var l = a.live;
                a.live = function() {
                    return l
                }
            }
            var d;
            return n.block == c._name ? (d = t.inheritSelf(c, s, a))._processLive(!0) : (d = r[n.block] = t.inherit(c, s, a))._name = n.block, d
        },
        _processLive: function() {
            return !1
        },
        create: function(t, e) {
            return "string" == typeof t && (t = {
                block: t
            }), new r[t.block](t.mods, e)
        },
        getName: function() {
            return this._name
        },
        _extractElemNameFrom: function() {},
        afterCurrentEvent: function(t, e) {
            1 == s.push({
                fn: t,
                ctx: e
            }) && setTimeout(this._runAfterCurrentEventFns, 0)
        },
        _runAfterCurrentEventFns: function() {
            var t = s.length;
            if (t)
                for (var e, n = s.splice(0, t); e = n.shift();)
                    e.fn.call(e.ctx || this)
        },
        changeThis: function(t, e) {
            return t.bind(e || this)
        },
        del: function(t) {
            var e = "string" == typeof t, n = e ? 0: 1, i = arguments.length;
            for (e && (t = this); i > n;)
                delete t[arguments[n++]];
            return this
        },
        channel: function(n, i) {
            return "boolean" == typeof n && (i = n, n = e), n || (n = "default"), i ? void(a[n] && (a[n].un(), delete a[n])) : a[n] || (a[n] = new t.observable)
        }
    })
}(jQuery), function() {
    Object.keys || (Object.keys = function(t) {
        var e = [];
        for (var n in t)
            t.hasOwnProperty(n) && e.push(n);
        return e
    })
}(), function() {
    var t = Array.prototype, e = Object.prototype.toString, n = {
        indexOf: function(t, e) {
            e =+ (e || 0);
            var n = this, i = n.length;
            if (i > 0 && i > e)
                for (e = 0 > e ? Math.ceil(e) : Math.floor(e), - i > e && (e = 0), 0 > e && (e += i); i > e;) {
                    if (e in n && n[e] === t)
                        return e;
                        ++e
                }
            return - 1
        },
        forEach: function(t, e) {
            for (var n =- 1, i = this, o = i.length; ++n < o;)
                n in i && (e ? t.call(e, i[n], n, i) : t(i[n], n, i))
        },
        map: function(t, e) {
            for (var n =- 1, i = this, o = i.length, s = new Array(o); ++n < o;)
                n in i && (s[n] = e ? t.call(e, i[n], n, i) : t(i[n], n, i));
            return s
        },
        filter: function(t, e) {
            for (var n =- 1, i = this, o = i.length, s = []; ++n < o;)
                n in i && (e ? t.call(e, i[n], n, i) : t(i[n], n, i)) && s.push(i[n]);
            return s
        },
        reduce: function(t, e) {
            var n, i =- 1, o = this, s = o.length;
            if (arguments.length < 2) {
                for (; ++i < s;)
                    if (i in o) {
                        n = o[i];
                        break
                    }
            } else 
                n = e;
            for (; ++i < s;)
                i in o && (n = t(n, o[i], i, o));
            return n
        },
        some: function(t, e) {
            for (var n =- 1, i = this, o = i.length; ++n < o;)
                if (n in i && (e ? t.call(e, i[n], n, i) : t(i[n], n, i)))
                    return !0;
            return !1
        },
        every: function(t, e) {
            for (var n =- 1, i = this, o = i.length; ++n < o;)
                if (n in i&&!(e ? t.call(e, i[n], n, i) : t(i[n], n, i)))
                    return !1;
            return !0
        }
    };
    for (var i in n)
        t[i] || (t[i] = n[i]);
    Array.isArray || (Array.isArray = function(t) {
        return "[object Array]" === e.call(t)
    })
}(), function() {
    var t = Array.prototype.slice;
    Function.prototype.bind || (Function.prototype.bind = function(e) {
        var n = this, i = t.call(arguments, 1);
        return function() {
            return n.apply(e, i.concat(t.call(arguments)))
        }
    })
}(), function(t, e, n) {
    function i(t, e, n) {
        n.push(r, t, r, e)
    }
    function o(t, e, n, o) {
        o.push(t), n && i(e, n, o)
    }
    function s(t, e, s, r, c) {
        o(t, n, n, c), c.push(a, e), r && i(s, r, c)
    }
    var r = "_", a = "__", c = "[a-zA-Z0-9-]+";
    t.INTERNAL = {
        NAME_PATTERN: c,
        MOD_DELIM: r,
        ELEM_DELIM: a,
        buildModPostfix: function(t, e, n) {
            var o = n || [];
            return i(t, e, o), n ? o : o.join("")
        },
        buildClass: function(t, e, i, r, a) {
            var c = typeof i;
            if ("string" == c ? "string" != typeof r && (a = r, r = i, i = e, e = n) : "undefined" != c ? (a = i, i = n) : e && "string" != typeof e && (a = e, e = n), !(e || i || a))
                return t;
            var u = a || [];
            return e ? s(t, e, i, r, u) : o(t, i, r, u), a ? u : u.join("")
        },
        buildClasses: function(t, i, r, a) {
            i && "string" != typeof i && (a = r, r = i, i = n);
            var c = a || [];
            return i ? s(t, i, n, n, c) : o(t, n, n, c), r && e.each(r, function(e, n) {
                n && (c.push(" "), i ? s(t, i, e, n, c) : o(t, e, n, c))
            }), a ? c : c.join("")
        }
    }
}(BEM, jQuery), jQuery.cookie = function(t, e, n) {
    if ("undefined" == typeof e) {
        var i = null;
        if (document.cookie && "" != document.cookie)
            for (var o = document.cookie.split(";"), s = 0; s < o.length; s++) {
                var r = jQuery.trim(o[s]);
                if (r.substring(0, t.length + 1) == t + "=") {
                    i = decodeURIComponent(r.substring(t.length + 1));
                    break
                }
            }
        return i
    }
    n = n || {}, null === e && (e = "", n.expires =- 1);
    var a = "";
    if (n.expires && ("number" == typeof n.expires || n.expires.toUTCString)) {
        var c;
        "number" == typeof n.expires ? (c = new Date, c.setTime(c.getTime() + 24 * n.expires * 60 * 60 * 1e3)) : c = n.expires, a = "; expires=" + c.toUTCString()
    }
    var u = n.path ? "; path=" + n.path: "", l = n.domain ? "; domain=" + n.domain: "", d = n.secure ? "; secure": "";
    document.cookie = [t, "=", encodeURIComponent(e), a, u, l, d].join("")
}, function(t) {
    function e(t) {
        return t = t.replace(/%.{2}/g, function(t) {
            return i[t]
        })
    }
    function n(t, n) {
        var i = "";
        try {
            i = t(n)
        } catch (o) {
            i = t(e(n))
        }
        return i
    }
    var i = {
        "%D0": "%D0%A0",
        "%C0": "%D0%90",
        "%C1": "%D0%91",
        "%C2": "%D0%92",
        "%C3": "%D0%93",
        "%C4": "%D0%94",
        "%C5": "%D0%95",
        "%A8": "%D0%81",
        "%C6": "%D0%96",
        "%C7": "%D0%97",
        "%C8": "%D0%98",
        "%C9": "%D0%99",
        "%CA": "%D0%9A",
        "%CB": "%D0%9B",
        "%CC": "%D0%9C",
        "%CD": "%D0%9D",
        "%CE": "%D0%9E",
        "%CF": "%D0%9F",
        "%D1": "%D0%A1",
        "%D2": "%D0%A2",
        "%D3": "%D0%A3",
        "%D4": "%D0%A4",
        "%D5": "%D0%A5",
        "%D6": "%D0%A6",
        "%D7": "%D0%A7",
        "%D8": "%D0%A8",
        "%D9": "%D0%A9",
        "%DA": "%D0%AA",
        "%DB": "%D0%AB",
        "%DC": "%D0%AC",
        "%DD": "%D0%AD",
        "%DE": "%D0%AE",
        "%DF": "%D0%AF",
        "%E0": "%D0%B0",
        "%E1": "%D0%B1",
        "%E2": "%D0%B2",
        "%E3": "%D0%B3",
        "%E4": "%D0%B4",
        "%E5": "%D0%B5",
        "%B8": "%D1%91",
        "%E6": "%D0%B6",
        "%E7": "%D0%B7",
        "%E8": "%D0%B8",
        "%E9": "%D0%B9",
        "%EA": "%D0%BA",
        "%EB": "%D0%BB",
        "%EC": "%D0%BC",
        "%ED": "%D0%BD",
        "%EE": "%D0%BE",
        "%EF": "%D0%BF",
        "%F0": "%D1%80",
        "%F1": "%D1%81",
        "%F2": "%D1%82",
        "%F3": "%D1%83",
        "%F4": "%D1%84",
        "%F5": "%D1%85",
        "%F6": "%D1%86",
        "%F7": "%D1%87",
        "%F8": "%D1%88",
        "%F9": "%D1%89",
        "%FA": "%D1%8A",
        "%FB": "%D1%8B",
        "%FC": "%D1%8C",
        "%FD": "%D1%8D",
        "%FE": "%D1%8E",
        "%FF": "%D1%8F"
    };
    t.extend({
        decodeURI: function(t) {
            return n(decodeURI, t)
        },
        decodeURIComponent: function(t) {
            return n(decodeURIComponent, t)
        }
    })
}(jQuery), function(t, e, n) {
    function i(t, e, n) {
        (t[e] || (t[e] = [])).unshift(n)
    }
    function o(t, e) {
        return e.modName ? function(n) {
            (n._curBlock.mods || {})[e.modName] === e.modVal && t(n)
        } : t
    }
    function s(t, n) {
        var i, o = e.isArray(n);
        return e.isArray(t) ? o ? i = t.concat(n) : (i = t).push(n) : o ? (i = n).unshift(t) : i = [t, n], i
    }
    function r(t) {
        return t.replace(p, function(t) {
            return f[t]
        })
    }
    var a = t.INTERNAL, c = a.ELEM_DELIM, u = {
        area: 1,
        base: 1,
        br: 1,
        col: 1,
        command: 1,
        embed: 1,
        hr: 1,
        img: 1,
        input: 1,
        keygen: 1,
        link: 1,
        meta: 1,
        param: 1,
        source: 1,
        wbr: 1
    }, l = a.buildClass, d = a.buildClasses, h = {}, f = {
        '"': "&quot;",
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;"
    }, p = /["&<>]/g;
    t.HTML = {
        decl: function(t, n) {
            "string" == typeof t && (t = {
                block: t
            }), t.name && (t.block = t.name);
            var s = h[t.block] || (h[t.block] = {});
            n.onBlock && i(s, "_block", o(n.onBlock, t)), n.onElem && (e.isFunction(n.onElem) ? i(s, "_elem", o(n.onElem, t)) : e.each(n.onElem, function(e, n) {
                i(s, "_elem" + ("*" === e ? "" : c + e), o(n, t))
            }))
        },
        build: function(t) {
            var e = new this.Ctx(t);
            return e._buildAll(), e._flush()
        },
        Ctx: e.inherit({
            __constructor: function(t) {
                this._buffer = [], this._params = t, this._tParams = null, this._tParamsChanges = null, this._curBlock = n
            },
            pos: function() {
                return this._params._pos
            },
            isFirst: function() {
                return 1 === this._params._pos
            },
            isLast: function() {
                var t = this._params;
                return t._pos === t._siblingsCount
            },
            params: function(t) {
                var e = this;
                return "undefined" == typeof t ? e._params : (e._params = t, e)
            },
            param: function(t, n, i, o) {
                var s = this, r = s._params;
                return "undefined" == typeof n ? r[t] : (!i && t in r ? o && (r[t] = e.extend(n, r[t])) : r[t] = n, s)
            },
            attrs: function(t, e) {
                return this.param("attrs", t, e, !0)
            },
            attr: function(t, e, n) {
                var i = this;
                if ("undefined" == typeof e)
                    return (i._params.attrs || {})[t];
                var o = i._params.attrs;
                return o ? (n ||!(t in o)) && (o[t] = e) : (i._params.attrs = {})[t] = e, i
            },
            tag: function(t, e) {
                return this.param("tag", t, e)
            },
            cls: function(t, e) {
                return this.param("cls", t, e)
            },
            mods: function(t, e) {
                return this.param("mods", t, e, !0)
            },
            mod: function(t, e, n) {
                var i = this;
                if ("undefined" == typeof e)
                    return (i._params.mods || {})[t];
                var o = i._params.mods;
                return o ? (n ||!(t in o)) && (o[t] = e) : (i._params.mods = {})[t] = e, i
            },
            mix: function(t, e) {
                var n = this, i = n._params;
                return "undefined" == typeof t ? i.mix : (i.mix=!e && "mix"in i ? i.mix.concat(t) : t, n)
            },
            js: function(t) {
                return this.param("js", t)
            },
            content: function(t, e) {
                return this.param("content", t, e)
            },
            wrapContent: function(t) {
                var e = this, n = e._params;
                return t.content = n.content, n.content = t, e
            },
            beforeContent: function(t) {
                var e = this, n = e._params;
                return n.content = s(t, n.content), e
            },
            afterContent: function(t) {
                var e = this, n = e._params;
                return n.content = s(n.content, t), e
            },
            wrap: function(t) {
                var e = this, n = e._params;
                return t.block || (t._curBlock = e._curBlock), t.content = n._wrapper ? n._wrapper : n, n._wrapper = t, e
            },
            tParam: function(t, e) {
                var n = this, i = n._tParams || (n._tParams = {});
                if ("undefined" == typeof e)
                    return i[t];
                var o = n._tParamsChanges || (n._tParamsChanges = {});
                return t in o || (o[t] = i[t]), i[t] = e, n
            },
            generateId: function() {
                return e.identify()
            },
            stop: function() {
                this._params._isStopped=!0
            },
            _buildAll: function() {
                var t = this, n = t._buffer, i = t._params, o = typeof i;
                if ("string" == o || "number" == o)
                    n.push(i);
                else if (e.isArray(i))
                    for (var s, r, a = 0, c = i.length; c > a;)
                        t._params = s = i[a++], r = typeof s, "string" == r || "number" == r ? n.push(s) : s && (s._pos = a, s._siblingsCount = c, t._buildByDecl());
                else 
                    i && (t._params._pos = t._params._siblingsCount = 1, t._buildByDecl())
            },
            _build: function() {
                var t, n = this, i = n._buffer, o = n._params, s = o.tag || "div", a = o.block || o.elem, c = a && (o.block || n._curBlock.block), h=!1;
                o.js && ((t = {})[l(c, o.elem)] = o.js===!0 ? {} : o.js, h=!o.elem), i.push("<", s), (a || o.cls) && (i.push(' class="'), a && (d(c, o.elem, o.mods, i), o.mix && e.each(o.mix, function(e, n) {
                    n && (i.push(" "), d(n.block, n.elem, n.mods, i), n.js && ((t || (t = {}))[l(n.block, n.elem)] = n.js===!0 ? {} : n.js, h || (h=!n.elem)))
                })), o.cls && i.push(a ? " " : "", o.cls), h && i.push(" i-bem"), i.push('"')), t && i.push(' onclick="return ', r(JSON.stringify(t)), '"'), o.attrs && e.each(o.attrs, function(t, e) {
                    "undefined" != typeof e && null !== e && e!==!1 && i.push(" ", t, '="', e.toString().replace(/"/g, "&quot;"), '"')
                }), u[s] ? i.push("/>") : (i.push(">"), "undefined" != typeof o.content && (n._params = o.content, n._buildAll()), i.push("</", s, ">"))
            },
            _flush: function() {
                var t = this._buffer.join("");
                return delete this._buffer, t
            },
            _buildByDecl: function() {
                var t = this, n = t._curBlock, i = t._params;
                if (i._curBlock && (t._curBlock = i._curBlock), i.block && (t._curBlock = i), !i._wrapper) {
                    if (i.block || i.elem) {
                        var o = h[t._curBlock.block];
                        if (o) {
                            var s;
                            if (i.elem ? (s = o["_elem" + c + i.elem], o._elem && (s = s ? s.concat(o._elem) : o._elem)) : s = o._block, s)
                                for (var r, a = 0; (r = s[a++]) && (r(t), !i._isStopped););
                        }
                    }
                    if (i._wrapper)
                        return i._curBlock = t._curBlock, t._params = i._wrapper, t._buildAll()
                }
                var u = t._tParamsChanges;
                if (t._tParamsChanges = null, t._build(), t._curBlock = n, u) {
                    var l = t._tParams;
                    e.each(u, function(t, e) {
                        "undefined" == typeof e ? delete l[t] : l[t] = e
                    })
                }
            }
        })
    }
}(BEM, jQuery), function(t) {
    if (!window.JSON) {
        var e, n = Object.prototype.toString, i = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, o = {
            "\b": "\\b",
            "	": "\\t",
            "\n": "\\n",
            "\f": "\\f",
            "\r": "\\r",
            '"': '\\"',
            "\\": "\\\\"
        };
        window.JSON = {
            stringify: e = function(s) {
                if (null === s)
                    return "null";
                if ("undefined" == typeof s)
                    return t;
                switch (n.call(s)) {
                case"[object String]":
                    return i.lastIndex = 0, '"' + (i.test(s) ? s.replace(i, function(t) {
                        var e = o[t];
                        return "string" == typeof e ? e : "\\u" + ("0000" + t.charCodeAt(0).toString(16)).slice( - 4)
                    }) : s) + '"';
                case"[object Number]":
                case"[object Boolean]":
                    return "" + s;
                case"[object Array]":
                    for (var r, a = "[", c = 0, u = s.length; u > c;)
                        r = e(s[c]), a += (c++?"," : "") + ("undefined" == typeof r ? "null" : r);
                    return a + "]";
                case"[object Object]":
                    if ("[object Function]" === n.call(s.toJSON))
                        return e(s.toJSON());
                    var r, a = "{", c = 0;
                    for (var l in s)
                        s.hasOwnProperty(l) && (r = e(s[l]), "undefined" != typeof r && (a += (c++?"," : "") + '"' + l + '":' + r));
                    return a + "}";
                default:
                    return t
                }
            }
        }
    }
}(), function(t, e, n) {
    function i(t, n) {
        var i = t[0];
        e.each(a(i), function(r, a) {
            s(a, i, r, n);
            var c = m[a.uniqId];
            c ? c.domElem.index(i) < 0 && (c.domElem = c.domElem.add(t), e.extend(c._params, a)) : o(r, t, a)
        })
    }
    function o(t, i, o, r, c) {
        "boolean" == typeof o && (c = r, r = o, o = n);
        var u = i[0];
        o = s(o || a(u)[t], u, t);
        var l = o.uniqId;
        if (m[l])
            return m[l]._init();
        p[l] = p[l] ? p[l].add(i) : i;
        var d = u.parentNode;
        d && 11 !== d.nodeType || e.unique(p[l]);
        var h = v[t] || C.decl(t, {}, {
            live: !0
        });
        if (!(h._liveInitable=!!h._processLive()) || r || o.live===!1) {
            var f = new h(p[l], o, !!r);
            return delete p[l], c && c.apply(f, Array.prototype.slice.call(arguments, 4)), f
        }
    }
    function s(t, n, i, o) {
        (t || (t = {})).uniqId || (t.uniqId = (t.id ? i + "-id-" + t.id : e.identify()) + (o || e.identify()));
        var s = e.identify(n), r = _[s] || (_[s] = {});
        return r[i] || (r[i] = t), t
    }
    function r(t, e, n) {
        var i = t.find(e);
        return n ? i : i.add(t.filter(e))
    }
    function a(t) {
        var n = e.identify(t);
        return _[n] || (_[n] = c(t))
    }
    function c(t) {
        var n = t.onclick || t.ondblclick;
        if (!n && "body" == t.tagName.toLowerCase()) {
            var i = e(t), o = i.attr("onclick") || i.attr("ondblclick");
            o && (n = Function(o))
        }
        return n ? n() : {}
    }
    function u(t) {
        delete _[e.identify(t)]
    }
    function l(t, e) {
        1 === t.domElem.length ? t.destruct(!0) : t.domElem = t.domElem.not(e)
    }
    var d = e(window), h = e(document), f = e("body"), p = {}, m = {}, _ = {}, g = {}, b = {}, v = t.blocks, y = t.INTERNAL, w = y.NAME_PATTERN, k = y.MOD_DELIM, E = y.ELEM_DELIM, M = y.buildModPostfix, x = y.buildClass;
    e.fn.bem = function(t, e) {
        return o(t, this, e, !0)
    };
    var C = t.DOM = t.decl("i-bem__dom", {
        __constructor: function(t, n, i) {
            var o = this;
            o.domElem = t, o._eventNameCache = {}, o._elemCache = {}, m[o._uniqId = n.uniqId || e.identify(o)] = o, o._needSpecialUnbind=!1, o.__base(null, n, i)
        },
        findBlocksInside: function(t, e) {
            return this._findBlocks("find", t, e)
        },
        findBlockInside: function(t, e) {
            return this._findBlocks("find", t, e, !0)
        },
        findBlocksOutside: function(t, e) {
            return this._findBlocks("parents", t, e)
        },
        findBlockOutside: function(t, e) {
            return this._findBlocks("closest", t, e)[0] || null
        },
        findBlocksOn: function(t, e) {
            return this._findBlocks("", t, e)
        },
        findBlockOn: function(t, e) {
            return this._findBlocks("", t, e, !0)
        },
        _findBlocks: function(t, i, s, r) {
            s || (s = i, i = n);
            var a = i ? "string" == typeof i ? this.findElem(i): i: this.domElem, c = "string" == typeof s, u = c ? s: s.block || s.blockName, l = "." + (c ? x(u) : x(u, s.modName, s.modVal)) + (r ? ":first" : ""), d = a.filter(l);
            if (t && (d = d.add(a[t](l))), r)
                return d[0] ? o(u, d.eq(0), !0) : null;
            var h = [], f = {};
            return e.each(d, function(t, n) {
                var i = o(u, e(n), !0);
                f[i._uniqId] || (f[i._uniqId]=!0, h.push(i))
            }), h
        },
        bindToDomElem: function(t, n, i) {
            var o = this;
            return i ? t.bind(o._buildEventName(n), function(t) {
                return (t.data || (t.data = {})).domElem = e(this), i.apply(o, arguments)
            }) : e.each(n, function(e, n) {
                o.bindToDomElem(t, e, n)
            }), o
        },
        bindToDoc: function(t, e) {
            return this._needSpecialUnbind=!0, this.bindToDomElem(h, t, e)
        },
        bindToWin: function(t, e) {
            var n, i, o = e;
            return "resize" === t && (e = function() {
                var t = d.height(), e = d.width();
                (n !== t || i !== e) && (n = t, i = e, o.apply(this, arguments))
            }), this._needSpecialUnbind=!0, this.bindToDomElem(d, t, e)
        },
        bindTo: function(t, n, i) {
            return !n || e.isFunction(n) ? (i = n, n = t, t = this.domElem) : "string" == typeof t && (t = this.elem(t)), this.bindToDomElem(t, n, i)
        },
        unbindFromDomElem: function(t, e) {
            return t.unbind(this._buildEventName(e)), this
        },
        unbindFromDoc: function(t) {
            return this.unbindFromDomElem(h, t)
        },
        unbindFromWin: function(t) {
            return this.unbindFromDomElem(d, t)
        },
        unbindFrom: function(t, e) {
            return e ? "string" == typeof t && (t = this.elem(t)) : (e = t, t = this.domElem), this.unbindFromDomElem(t, e)
        },
        _buildEventName: function(t) {
            var e = this;
            return t.indexOf(" ") > 1 ? t.split(" ").map(function(t) {
                return e._buildOneEventName(t)
            }).join(" ") : e._buildOneEventName(t)
        },
        _buildOneEventName: function(t) {
            var e = this, n = e._eventNameCache;
            if (t in n)
                return n[t];
            var i = "." + e._uniqId;
            if (t.indexOf(".") < 0)
                return n[t] = t + i;
            var o = ".bem_" + e.__self._name;
            return n[t] = t.split(".").map(function(t, e) {
                return 0 == e ? t + o : o + "_" + t
            }).join("") + i
        },
        trigger: function(t, e) {
            return this.__base(t = this.buildEvent(t), e).domElem && this._ctxTrigger(t, e), this
        },
        _ctxTrigger: function(t, n) {
            var i = this, o = g[i.__self._buildCtxEventName(t.type)], s = {};
            o && i.domElem.each(function() {
                for (var r = this, a = o.counter; r && a;) {
                    var c = e.identify(r, !0);
                    if (c) {
                        if (s[c])
                            break;
                        var u = o.ctxs[c];
                        u && (e.each(u, function(e, o) {
                            o.fn.call(o.ctx || i, t, n)
                        }), a--), s[c]=!0
                    }
                    r = r.parentNode
                }
            })
        },
        setMod: function(t, n, i) {
            if (t && "undefined" != typeof i && t.length > 1) {
                var o = this;
                return t.each(function() {
                    var s = e(this);
                    s.__bemElemName = t.__bemElemName, o.setMod(s, n, i)
                }), o
            }
            return this.__base(t, n, i)
        },
        _extractModVal: function(t, e, n) {
            var i, o = (e || this.domElem)[0];
            return o && (i = o.className.match(this.__self._buildModValRE(t, n || e))), i ? i[2] : ""
        },
        _extractMods: function(t, e) {
            var n = {}, i=!t.length, o = 0;
            return ((e || this.domElem)[0].className.match(this.__self._buildModValRE("(" + (i ? w : t.join("|")) + ")", e, "g")) || []).forEach(function(t) {
                var e = (t = t.trim()).lastIndexOf(k), i = t.substr(0, e - 1).lastIndexOf(k);
                n[t.substr(i + 1, e - i - 1)] = t.substr(e + 1), ++o
            }), o < t.length && t.forEach(function(t) {
                t in n || (n[t] = "")
            }), n
        },
        _afterSetMod: function(t, n, i, o, s) {
            var r = this.__self, a = r._buildModClassPrefix(t, s), c = r._buildModValRE(t, s), u = "" === n;
            (o || this.domElem).each(function() {
                var t = this.className;
                t.indexOf(a)>-1 ? this.className = t.replace(c, u ? "" : "$1" + a + n) : u || e(this).addClass(a + n)
            }), s && this.dropElemCache(s, t, i).dropElemCache(s, t, n)
        },
        findElem: function(t, e, n, i) {
            arguments.length%2 ? (i = n, n = e, e = t, t = this.domElem) : "string" == typeof t && (t = this.findElem(t));
            var o = this.__self, s = "." + e.split(" ").map(function(t) {
                return x(o._name, t, n, i)
            }).join(",.");
            return r(t, s)
        },
        _elem: function(t, e, n) {
            var i, o = t + M(e, n);
            return (i = this._elemCache[o]) || (i = this._elemCache[o] = this.findElem(t, e, n), i.__bemElemName = t), i
        },
        elem: function(t, n, i) {
            if (n && "string" != typeof n)
                return n.__bemElemName = t, n;
            if (t.indexOf(" ") < 0)
                return this._elem(t, n, i);
            var o = e([]), s = this;
            return t.split(" ").forEach(function(t) {
                o = o.add(s._elem(t, n, i))
            }), o
        },
        dropElemCache: function(t, e, n) {
            if (t) {
                var i = this, o = M(e, n);
                t.indexOf(" ") < 0 ? delete i._elemCache[t + o] : t.split(" ").forEach(function(t) {
                    delete i._elemCache[t + o]
                })
            } else 
                this._elemCache = {};
            return this
        },
        elemParams: function(t) {
            var e;
            return "string" == typeof t ? (e = t, t = this.elem(t)) : e = this.__self._extractElemNameFrom(t), c(t[0])[x(this.__self.getName(), e)] || {}
        },
        elemify: function(t, n) {
            return (t = e(t)).__bemElemName = n, t
        },
        containsDomElem: function(t) {
            var e=!1;
            return this.domElem.each(function() {
                return !(e = t.parents().andSelf().index(this)>-1)
            }), e
        },
        buildSelector: function(t, e, n) {
            return this.__self.buildSelector(t, e, n)
        },
        destruct: function(t) {
            var n = this, i = n.__self;
            n._isDestructing=!0, n._needSpecialUnbind && i.doc.add(i.win).unbind("." + n._uniqId), n.dropElemCache().domElem.each(function(t, n) {
                var i = a(n);
                e.each(i, function(t, e) {
                    var o = m[e.uniqId];
                    o ? o._isDestructing || (l(o, n), delete i[t]) : delete p[e.uniqId]
                }), e.isEmptyObject(i) && u(n)
            }), t || n.domElem.remove(), delete m[n.un()._uniqId], delete n.domElem, delete n._elemCache, n.__base()
        }
    }, {
        scope: f,
        doc: h,
        win: d,
        _processLive: function(t) {
            var e = this, n = e._liveInitable;
            if ("live"in e) {
                var i = "undefined" == typeof n;
                i^t && (n = e.live()!==!1, e.live = function() {})
            }
            return n
        },
        init: function(t, n, o) {
            (!t || e.isFunction(t)) && (o = n, n = t, t = h);
            var s = e.identify();
            return r(t, ".i-bem").each(function() {
                i(e(this), s)
            }), n && this.afterCurrentEvent(function() {
                n.call(o || this, t)
            }), this._runAfterCurrentEventFns(), t
        },
        destruct: function(t, i, o) {
            "boolean" != typeof t && (o = i, i = t, t = n), r(i, ".i-bem", o).each(function(t, n) {
                var i = a(this);
                e.each(i, function(t, e) {
                    if (e.uniqId) {
                        var o = m[e.uniqId];
                        o ? (l(o, n), delete i[t]) : delete p[e.uniqId]
                    }
                }), e.isEmptyObject(i) && u(this)
            }), t || (o ? i.empty() : i.remove())
        },
        update: function(t, e, n, i) {
            this.destruct(t, !0), this.init(t.html(e), n, i)
        },
        replace: function(t, n) {
            this.destruct(!0, t), this.init(e(n).replaceAll(t))
        },
        append: function(t, n) {
            this.init(e(n).appendTo(t))
        },
        prepend: function(t, n) {
            this.init(e(n).prependTo(t))
        },
        before: function(t, n) {
            this.init(e(n).insertBefore(t))
        },
        after: function(t, n) {
            this.init(e(n).insertAfter(t))
        },
        _buildCtxEventName: function(t) {
            return this._name + ":" + t
        },
        _liveClassBind: function(t, n, i, o) {
            var s = this;
            if (n.indexOf(" ")>-1)
                n.split(" ").forEach(function(e) {
                    s._liveClassBind(t, e, i, o)
                });
            else {
                var r = b[n], a = e.identify(i);
                r || (r = b[n] = {}, h.bind(n, s.changeThis(s._liveClassTrigger, s))), r = r[t] || (r[t] = {
                    uniqIds: {},
                    fns: []
                }), a in r.uniqIds || (r.fns.push({
                    uniqId: a,
                    fn: s._buildLiveEventFn(i, o)
                }), r.uniqIds[a] = r.fns.length - 1)
            }
            return this
        },
        _liveClassUnbind: function(t, n, i) {
            var o = b[n];
            if (o)
                if (i) {
                    if (o = o[t]) {
                        var s = e.identify(i);
                        if (s in o.uniqIds) {
                            var r = o.uniqIds[s], a = o.fns.length - 1;
                            for (o.fns.splice(r, 1); a > r;)
                                o.uniqIds[o.fns[r++].uniqId] = r - 1;
                                delete o.uniqIds[s]
                        }
                    }
                } else 
                    delete o[t];
            return this
        },
        _liveClassTrigger: function(t) {
            var n = b[t.type];
            if (n) {
                var i = t.target, o = [];
                for (var s in n)
                    n.hasOwnProperty(s) && o.push(s);
                do 
                    for (var r = " " + i.className + " ", a = 0; s = o[a++];)
                        if (r.indexOf(" " + s + " ")>-1) {
                            for (var c, u = 0, l = n[s].fns, d=!1; c = l[u++];)
                                c.fn.call(e(i), t)===!1 && (d=!0);
                                if (d && t.preventDefault(), d || t.isPropagationStopped())
                                    return;
                                    o.splice(--a, 1)
                        }
                while (o.length && (i = i.parentNode))
                }
        }, _buildLiveEventFn: function(t, n) {
            var i = this;
            return function(s) {
                var r = [i._name, ((s.data || (s.data = {})).domElem = e(this)).closest(i.buildSelector()), !0], a = o.apply(null, n ? r.concat([t, s]) : r);
                return a&&!n && t ? t.apply(a, arguments) : void 0
            }
        }, liveInitOnEvent: function(t, e, n) {
            return this.liveBindTo(t, e, n, !0)
        }, liveBindTo: function(t, i, o, s) {
            (!i || e.isFunction(i)) && (o = i, i = t, t = n), t && "string" != typeof t || (t = {
                elem: t
            }), t.elemName && (t.elem = t.elemName);
            var r = this;
            return t.elem && t.elem.indexOf(" ") > 0 ? (t.elem.split(" ").forEach(function(e) {
                r._liveClassBind(x(r._name, e, t.modName, t.modVal), i, o, s)
            }), r) : r._liveClassBind(x(r._name, t.elem, t.modName, t.modVal), i, o, s)
        }, liveUnbindFrom: function(t, e, n) {
            var i = this;
            return t.indexOf(" ") > 1 ? (t.split(" ").forEach(function(t) {
                i._liveClassUnbind(x(i._name, t), e, n)
            }), i) : i._liveClassUnbind(x(i._name, t), e, n)
        }, _liveInitOnBlockEvent: function(t, e, n, i) {
            var o = this._name;
            return v[e].on(t, function(t) {
                var e = arguments, s = t.block[i](o);
                n && s.forEach(function(t) {
                    n.apply(t, e)
                })
            }), this
        }, liveInitOnBlockEvent: function(t, e, n) {
            return this._liveInitOnBlockEvent(t, e, n, "findBlocksOn")
        }, liveInitOnBlockInsideEvent: function(t, e, n) {
            return this._liveInitOnBlockEvent(t, e, n, "findBlocksOutside")
        }, liveInitOnBlockInit: function(t, e) {
            return this.liveInitOnBlockEvent("init", t, e)
        }, liveInitOnBlockInsideInit: function(t, e) {
            return this.liveInitOnBlockInsideEvent("init", t, e)
        }, on: function(t, e, n, i, o) {
            return t.jquery ? this._liveCtxBind(t, e, n, i, o) : this.__base(t, e, n, i)
        }, un: function(t, e, n, i) {
            return t.jquery ? this._liveCtxUnbind(t, e, n, i) : this.__base(t, e, n)
        }, liveCtxBind: function(t, e, n, i, o) {
            return this._liveCtxBind(t, e, n, i, o)
        }, _liveCtxBind: function(t, i, o, s, r) {
            var a = this;
            if ("string" == typeof i)
                if (e.isFunction(o) && (r = s, s = o, o = n), i.indexOf(" ")>-1)
                    i.split(" ").forEach(function(e) {
                        a._liveCtxBind(t, e, o, s, r)
                    });
                else {
                    var c = a._buildCtxEventName(i), u = g[c] || (g[c] = {
                        counter: 0,
                        ctxs: {}
                    });
                    t.each(function() {
                        var t = e.identify(this), n = u.ctxs[t];
                        n || (n = u.ctxs[t] = {}, ++u.counter), n[e.identify(s) + (r ? e.identify(r) : "")] = {
                            fn: s,
                            data: o,
                            ctx: r
                        }
                    })
                } else 
                    e.each(i, function(e, n) {
                        a._liveCtxBind(t, e, n, o)
                    });
            return a
        }, liveCtxUnbind: function(t, e, n, i) {
            return this._liveCtxUnbind(t, e, n, i)
        }, _liveCtxUnbind: function(t, n, i, o) {
            var s = this, r = g[n = s._buildCtxEventName(n)];
            return r && (t.each(function() {
                var t, n = e.identify(this, !0);
                n && (t = r.ctxs[n]) && (i && delete t[e.identify(i) + (o ? e.identify(o) : "")], (!i || e.isEmptyObject(t)) && (r.counter--, delete r.ctxs[n]))
            }), r.counter || delete g[n]), s
        }, _extractElemNameFrom: function(t) {
            if (t.__bemElemName)
                return t.__bemElemName;
            var e = t[0].className.match(this._buildElemNameRE());
            return e ? e[1] : n
        }, extractParams: c, _buildModClassPrefix : function(t, e) {
            return x(this._name) + (e ? E + ("string" == typeof e ? e : this._extractElemNameFrom(e)) : "") + k + t + k
        }, _buildModValRE: function(t, e, n) {
            return new RegExp("(\\s|^)" + this._buildModClassPrefix(t, e) + "(" + w + ")(?=\\s|$)", n)
        }, _buildElemNameRE: function() {
            return new RegExp(this._name + E + "(" + w + ")(?:\\s|$)")
        }, buildSelector: function(t, e, n) {
            return "." + x(this._name, t, e, n)
        }, getBlockByUniqId: function(t) {
            return m[t]
        }, getWindowSize: function() {
            return {
                width: d.width(),
                height: d.height()
            }
        }
    })
}(BEM, jQuery), function() {
    String.prototype.trim || (String.prototype.trim = function() {
        for (var t = this.replace(/^\s\s*/, ""), e = /\s/, n = t.length; e.test(t.charAt(--n)););
        return t.slice(0, n + 1)
    })
}(), function(t) {
    t || (t = window.Lego = {}), t.isSessionValid = function() {
        return !!t.getCookie("yandex_login")
    }
}(window.Lego), BEM.DOM.decl("i-global", {
    onSetMod: {
        js: function() {
            this.del(this.__self._params = $.extend({}, this.params), "uniqId", "name");
            var t = this.__self._params;
            t["passport-msg"] || (t["passport-msg"] = t.id), void 0 === t["show-counters"] && (t["show-counters"] = Math.round(100 * Math.random()) <= t["show-counters-percent"]), t.locale = t.lang, $(function() {
                t.oframebust && Lego.oframebust(t.oframebust)
            })
        }
    },
    getDefaultParams: function() {
        return {
            id: "",
            login: Lego.isSessionValid() ? $.cookie("yandex_login") || "": "",
            yandexuid: $.cookie("yandexuid"),
            lang: "ru",
            tld: "ru",
            retpath: encodeURI($.decodeURI(location.href)),
            "passport-host": "https://passport.yandex.ru",
            "pass-host": "//pass.yandex.ru",
            "social-host": "//social.yandex.ru",
            "lego-path": "/lego",
            "show-counters-percent": 100
        }
    }
}, {
    param: function(t) {
        return (this._params || {})[t]
    }
}), function(t) {
    function e(t) {
        return t.replace(/^(?:https?:)?\/\//, "")
    }
    t || (t = window.Lego = {}), !t.params && (t.params = {}), t.c = function(t, n, i) {
        var o = e(i && i.host || BEM.blocks["i-global"].param("click-host") || "clck.yandex.ru"), s = function(t, e, n, i) {
            return e = e.replace("'", "%27"), e.indexOf("/dtype=")>-1 ? e : location.protocol + "//" + o + "/" + n + "/dtype=" + t + "/rnd=" + ((new Date).getTime() + Math.round(100 * Math.random())) + (i ? "/*" + (e.match(/^http/) ? e : location.protocol + "//" + location.host + (e.match("^/") ? e : "/" + e)) : "/*data=" + encodeURIComponent("url=" + encodeURIComponent(e.match(/^http/) ? e : location.protocol + "//" + location.host + (e.match("^/") ? e : "/" + e))))
        }, r = function() {
            var e = document.getElementsByTagName("head")[0] || document.getElementsByTagName("body")[0], n = document.createElement("script");
            n.setAttribute("src", s(t, location.href, "jclck")), e.insertBefore(n, e.firstChild)
        };
        if (n)
            if (n.className.match(/b-link_pseudo_yes/) || n.href && n.href.match(/^mailto:/) || i && i.noRedirect===!0)
                r();
            else if (n.href) {
                var a = n.href;
                n.href = s(t, a, "redir"), setTimeout(function() {
                    n.href = a
                }, 500)
            } else if (n.form)
                if (n.type.match(/submit|button|image/)) {
                    var a = n.form.action;
                    n.form.action = s(t, a, "redir", !0), setTimeout(function() {
                        n.form.action = a
                    }, 500)
                } else 
                    r();
            else {
                if (!n.action)
                    throw "counter.js: not link and not form!";
                    n.action = s(t, n.action, "redir", !0)
            } else 
                r()
    }
}(window.Lego), function(t, e) {
    t || (t = window.Lego = {}), t.cp = function(n, i, o, s, r, a) {
        "string" == typeof s || (a = r, r = s, s = e), t.c("stred/pid=" + n + "/cid=" + i + (o ? "/path=" + o + (s ? "/vars=" + s : "") : ""), r, a)
    }
}(window.Lego), function(t) {
    t || (t = window.Lego = {}), t.ch = function(e, n, i) {
        BEM.blocks["i-global"].param("show-counters") && t.cp(0, 2219, e, n, i)
    }
}(window.Lego), function(t) {
    t.cred = function(t, e, n) {
        var i, o = function() {
            var e = document.getElementsByTagName("head")[0] || document.getElementsByTagName("body")[0], n = document.createElement("script");
            n.setAttribute("src", t), e.insertBefore(n, e.firstChild)
        };
        if (e)
            if (e.className.match(/b-link_pseudo_yes/) || e.href && e.href.match(/^mailto:/) || n && n.noRedirect===!0)
                o();
            else if (e.href)
                i = e.href, e.href = t, setTimeout(function() {
                    e.href = i
                }, 500);
            else if (e.form)
                e.type.match(/submit|button|image/) ? (i = e.form.action, e.form.action = t, setTimeout(function() {
                    e.form.action = i
                }, 500)) : o();
            else {
                if (!e.action)
                    throw "counter.js: not link and not form!";
                    e.action = t
            } else 
                o()
    }
}(window.Lego), window.csh_ifgsid = csh_ifmsid, function(t) {
    t || (t = window.Lego = {}), t.getCookie = function(t) {
        var e = document.cookie;
        if (e.length < 1)
            return !1;
        var n = e.indexOf(t + "=");
        if ( - 1 == n)
            return !1;
        n += t.length + 1;
        var i = e.indexOf(";", n);
        return decodeURIComponent( - 1 == i ? e.substring(n) : e.substring(n, i))
    }
}(window.Lego), function(t, e) {
    e || (e = window.Lego = {}), e.init || (e.init = function(n) {
        return (n = e.params = t.extend({
            id: "",
            login: e.isSessionValid() ? e.getCookie("yandex_login") || "": "",
            yandexuid: e.getCookie("yandexuid"),
            locale: "ru",
            retpath: window.location.toString(),
            "passport-host": "//passport.yandex.ru",
            "pass-host": "//pass.yandex.ru",
            "passport-msg": n.id,
            "social-host": "//social.yandex.ru",
            "lego-path": "/lego",
            "show-counters-percent": 100
        }, n, e.params))["show-counters"] = Math.round(100 * Math.random()) <= n["show-counters-percent"], BEM.blocks["i-global"]._params || t.extend(BEM.blocks["i-global"]._params = {}, n), t(function() {
            n.oframebust && e.oframebust(n.oframebust)
        }), n
    }), e.block || (e.block = {}), e.blockInit || (e.blockInit = function(n, i) {
        n = n || document, i = i || ".g-js", t(n).find(i).each(function() {
            var n = t(this), i = this.onclick ? this.onclick(): {}, o = i.name || "", s = e.block[o];
            s&&!n.data(o) && (s.call(n, i), n.data(o, !0).addClass(o + "_js_inited"))
        })
    }), e.blockInitBinded || (e.blockInitBinded=!!t(document).ready(function() {
        e.blockInit()
    }))
}(jQuery, window.Lego), function() {
    "use strict";
    BEM.decl("i-ua", {}, {});
    try {
        BEM.blocks["i-ua"].svg = document.documentElement.className.indexOf("i-ua_inlinesvg_yes")>-1
    } catch (t) {}
}(), function(t, e, n, i, o, s) {
    var r;
    e = t.documentElement, n = "className", i = "replace", e[n] = e[n][i]("i-ua_js_no", "i-ua_js_yes"), "CSS1Compat" !== t.compatMode && (e[n] = e[n][i]("i-ua_css_standart", "i-ua_css_quirks")), r = navigator.appVersion.match(/OS\s(\d)/), r = r && r[1] && parseInt(r[1], 10), /iphone|ipad|ipod/i.test(navigator.appVersion) && r >= 7 ? ((o = t.documentElement) && (s = o.classList), s.add("ios-simplified")) : /android/i.test(navigator.appVersion) && ((o = t.documentElement) && (s = o.classList), s.add("android"))
}(document), function(t) {
    t || (t = window.Lego = {}), t.messages = t.messages || {}, t.message = function(e, n) {
        return "ru" == t.params.locale ? n : t.messages[e] || n
    }
}(window.Lego), function(t) {
    var e, n;
    try {
        e = t.localStorage, n = "ya_ls_test", e.setItem(n, "1"), "1" !== e.getItem(n) ? e = null : e.removeItem(n)
    } catch (i) {
        e = null
    }
    var o = t.home || (t.home = {}), s = t.console && function(t) {
        console.log(t)
    }
    || function() {}, r = function(t) {
        var e = "localStorage is disabled. " + (t ? t : "");
        s(e)
    }, a = function(t) {
        try {
            return t()
        } catch (e) {
            r(e)
        }
    };
    e ? o.localStorage = {
        setItem: function(t, n) {
            return a(function() {
                return e.setItem(t, n)
            })
        },
        getItem: function(t) {
            return a(function() {
                return e.getItem(t)
            })
        },
        removeItem: function(t) {
            return a(function() {
                return e.removeItem(t)
            })
        },
        clear: function() {
            return a(function() {
                return e.clear()
            })
        },
        key: function(t) {
            return a(function() {
                return e.key(t)
            })
        }
    } : (o.localStorage = {
        removeItem: r,
        setItem: r,
        getItem: r,
        clear: r
    }, o.disabledLS=!0)
}(window), function(t) {
    "use strict";
    for (var e = ["ms", "moz", "webkit", "o"], n = 0; n < e.length&&!t.requestAnimationFrame; ++n)
        t.requestAnimationFrame = t[e[n] + "RequestAnimationFrame"];
    if (void 0 === t.requestAnimationFrame) {
        var i, o = [], s =+ new Date, r = 0, a = function() {
            var t = o;
            o = [], r =+ new Date, i = 0;
            for (var e = 0, n = t.length; n > e; e++)
                t[e](r - s)
        };
        t.requestAnimationFrame = function(e) {
            if (o.push(e), !i) {
                var n =+ new Date, s = Math.max(0, 16 - (n - r));
                i = t.setTimeout(a, s)
            }
            return i
        }, t.requestAnimationFrame.fallback=!0
    }
}(this), function(t) {
    var e, n;
    try {
        e = t.localStorage, n = "ya_ls_test", e.setItem(n, "1"), "1" !== e.getItem(n) ? e = null : e.removeItem(n)
    } catch (i) {
        e = null
    }
    var o = t.home || (t.home = {}), s = t.console && function(t) {
        console.log(t)
    }
    || function() {}, r = function(t) {
        var e = "localStorage is disabled. " + (t ? t : "");
        s(e)
    }, a = function(t) {
        try {
            return t()
        } catch (e) {
            r(e)
        }
    };
    e ? o.localStorage = {
        setItem: function(t, n) {
            return a(function() {
                return e.setItem(t, n)
            })
        },
        getItem: function(t) {
            return a(function() {
                return e.getItem(t)
            })
        },
        removeItem: function(t) {
            return a(function() {
                return e.removeItem(t)
            })
        },
        clear: function() {
            return a(function() {
                return e.clear()
            })
        },
        key: function(t) {
            return a(function() {
                return e.key(t)
            })
        }
    } : (o.localStorage = {
        removeItem: r,
        setItem: r,
        getItem: r,
        clear: r
    }, o.disabledLS=!0)
}(window), BEM.HTML.decl("b-link", {
    onBlock: function(t) {
        t.tag("a").attr("href", t.param("url"));
        for (var e, n = ["title", "target"]; e = n.pop();)
            t.param(e) && t.attr(e, t.param(e))
    }
}), function(t) {
    var e = t.event.special.leftclick = {
        setup: function() {
            t(this).bind("click", e.handler)
        },
        teardown: function() {
            t(this).unbind("click", e.handler)
        },
        handler: function(e) {
            e.button || (e.type = "leftclick", t.event.handle.apply(this, arguments), e.type = "click")
        }
    }
}(jQuery), BEM.DOM.decl({
    name: "b-link",
    modName: "pseudo",
    modVal: "yes"
}, {
    _onClick: function(t) {
        t.preventDefault(), this.hasMod("disabled", "yes") || this.afterCurrentEvent(function() {
            this.trigger("click")
        })
    }
}, {
    live: function() {
        this.__base.apply(this, arguments), this.liveBindTo({
            modName: "pseudo",
            modVal: "yes"
        }, "leftclick", function(t) {
            this._onClick(t)
        })
    }
}), BEM.HTML.decl({
    name: "b-link",
    modName: "pseudo",
    modVal: "yes"
}, {
    onBlock: function(t) {
        t.tag(t.param("url") ? "a" : "span"), t.wrapContent({
            elem: "inner"
        }), t.js(!0)
    },
    onElem: {
        inner: function(t) {
            t.tag("span")
        }
    }
}), function() {
    var t, e = {
        isAuthenticated: !1,
        isFeedbackReceived: !1,
        isRequestSent: !1
    }, n = function() {};
    BEM.decl("i-mda-auth", {}, {
        authenticate: function(i) {
            if (t = Lego.params["mda-auth-url"] || "", "" === t)
                return !1;
            var o = Lego.params["mda-auth-delay"] || 10;
            n = i || function() {}, setTimeout(function() {
                var n = document.createElement("iframe");
                n.style.visibility = "hidden", n.style.position = "absolute", n.setAttribute("src", t), n.style.top = n.style.left = n.style.width = n.style.height = 0, document.getElementsByTagName("body")[0].appendChild(n), e.isRequestSent=!0
            }, o)
        },
        isRequestSent: function() {
            return e.isRequestSent
        },
        isFeedbackReceived: function() {
            return e.isFeedbackReceived
        },
        isAuthenticated: function() {
            return e.isAuthenticated
        },
        getUrl: function() {
            return t
        },
        registerFeedback: function(t) {
            var i = "success" === t;
            e.isFeedbackReceived=!0, e.isAuthenticated = i, n(i)
        }
    })
}(), function(t) {
    var e;
    e = "ontouchstart"in window ? {
        type: "touch",
        down: "touchstart",
        move: "touchmove",
        up: "touchend",
        cancel: "touchcancel"
    } : {
        type: "mouse",
        down: "mousedown",
        move: "mousemove",
        up: "mouseup",
        cancel: "mouseleave"
    };
    var n = function(t) {
        var n = t.originalEvent;
        return "touch" === e.type ? (n = t.originalEvent, n.touches && n.touches[0] ? {
            x: n.touches[0].pageX,
            y: n.touches[0].pageY
        } : n.changedTouches && n.changedTouches[0] ? {
            x: n.changedTouches && n.changedTouches[0].pageX,
            y: n.changedTouches && n.changedTouches[0].pageY
        } : {
            x: 0,
            y: 0
        }) : {
            x: t.clientX,
            y: t.clientY
        }
    };
    t.event.special.pointerdown = {
        setup: function() {
            t(this).bind(e.down + ".pointerEvents", function(i) {
                "mouse" === e.type && (this._pointerIsDown=!0), i.pointer = n(i), i.type = "pointerdown", t.event.handle.apply(this, arguments)
            })
        },
        teardown: function() {
            t(this).unbind(e.down + ".pointerEvents")
        }
    }, t.event.special.pointerup = {
        setup: function() {
            t(this).bind(e.up + ".pointerEvents", function(i) {
                "mouse" === e.type && delete this._pointerIsDown, i.pointer = n(i), i.type = "pointerup", t.event.handle.apply(this, arguments)
            })
        },
        teardown: function() {
            t(this).unbind(e.up + ".pointerEvents")
        }
    }, t.event.special.tap = {
        setup: function() {
            if ("mouse" !== e.type&&!/Android/.test(navigator.userAgent)) {
                var n = {};
                t(this).bind({
                    "pointerdown.tapEvent": function(t) {
                        n.target = this, n.x1 = t.pointer.x, n.y1 = t.pointer.y
                    },
                    "pointermove.tapEvent": function(t) {
                        n.x2 = t.pointer.x, n.y2 = t.pointer.y
                    },
                    "pointerup.tapEvent": function(e) {
                        n.x2 = e.pointer.x, n.y2 = e.pointer.y, Math.abs(n.x2 - n.x1) < 5 && Math.abs(n.y2 - n.y1) < 5 && (9 !== n.target.nodeType && (t(document).one("click.preventClick", function(t) {
                            return t.preventDefault(), !1
                        }), t("body").css("pointer-events", "none"), setTimeout(function() {
                            t(document).unbind(".preventClick"), t("body").css("pointer-events", "auto")
                        }, 750)), e.type = "tap", t.event.handle.apply(this, arguments)), n = {}
                    },
                    "pointercancel.tapEvent": function() {
                        n = {}
                    }
                })
            }
        },
        teardown: function() {
            t(this).unbind(".tapEvent")
        }
    }
}(jQuery), function(t) {
    var e = t.browser.opera && t.browser.version < 12.1 ? "keypress": "keydown", n = Object.prototype.hasOwnProperty;
    BEM.DOM.decl("popup", {
        getDefaultParams: function() {
            var t = {
                left: 15,
                right: 15,
                top: 15,
                bottom: 15
            };
            return {
                directions: [{
                    to: "bottom",
                    axis: "center",
                    tail: {
                        axis: "center"
                    }
                }, {
                    to: "top",
                    axis: "center",
                    tail: {
                        axis: "center"
                    }
                }, {
                    to: "right",
                    axis: "middle",
                    tail: {
                        axis: "middle"
                    }
                }, {
                    to: "left",
                    axis: "middle",
                    tail: {
                        axis: "middle"
                    }
                }
                ],
                tail: {
                    width: 24.04,
                    height: 12.02,
                    offset: t
                }
            }
        },
        onSetMod: {
            js: function() {
                this._cache = {}, this._body = t("body"), this._inContainer = null, this._isParentFixed=!1, this._owner = null, this._userPosition = null, this._parent = null, this._childs = [], this._isShown=!1, this._isHiding=!1, this._positions = {}, this._currPos = {}, this._visibilityFactor = null, this._direction=!1, this._directions = {};
                var e = this.getDefaultParams(), n = this.params, i = this._repackDirParams(e.directions), o = this._repackDirParams(n.directions);
                n.tail && (this.params.tail = this._mergeParams(e.tail, n.tail, {
                    offset: this._offsetParamsHook
                })), this._order = o.keys.map(function(t) {
                    var e = o.directions[t], n = i.directions[t];
                    return n || (n = i.directions[e.to]), this._directions[t] = this._mergeParams(n, e, {
                        offset: this._offsetParamsHook,
                        tail: this._tailParamsHook
                    }), t
                }, this)
            },
            visibility: {
                visible: function() {
                    this._onShown()
                },
                "": function() {
                    this._onHidden()
                }
            }
        },
        show: function(e) {
            var n;
            if (e instanceof BEM)
                n = e.domElem;
            else if (e instanceof t)
                n = e;
            else if (!e)
                return;
            if (n) {
                this._owner && n[0] !== this._owner[0] && this.delMod("visibility"), this._owner = n;
                var i = this.findBlockOutside(n, "popup");
                i && this.setParent(i)
            } else 
                this._userPosition = e;
            return this.setMod("visibility", "outside").repaint()
        },
        hide: function() {
            if (this._isHiding)
                return this;
            if (this._isShown && (this._isHiding=!0, this._childs.forEach(function(t) {
                t.hide()
            }), this.hasMod("animate", "yes"))) {
                var t = this;
                return this.beforeHide(function() {
                    t.delMod("visibility")
                }), this
            }
            return this.delMod("visibility")
        },
        toggle: function(t) {
            return this._isShown&&!this._isHiding ? this.hide() : this.show(t || this._owner)
        },
        repaint: function() {
            this._moveToContainer();
            var t = this._pickDirection();
            return this.setMod("to", t.to)._show(this._positions[t.key], this._tailPos && this._tailPos[t.key]), this
        },
        getCurrPos: function() {
            return this._currPos
        },
        getCurrDirection: function() {
            return this._direction
        },
        setContent: function(t) {
            return BEM.DOM.update(this.elem("content"), t), this._isShown && this.repaint(), this
        },
        isShown: function() {
            return this._isShown
        },
        setParent: function(t) {
            return this._parent = t, this._isParentFixed = t.hasMod("position", "fixed"), t.addChild(this), this
        },
        addChild: function(t) {
            var e, n = this._childs.length;
            for (e = 0; n > e; e++)
                if (this._childs[e]._uniqId == t._uniqId)
                    return;
            this._childs.push(t), t.on("hide", function() {
                this.removeChild(t)
            }, this)
        },
        removeChild: function(t) {
            var e, n = this._childs.length;
            for (e = 0; n > e; e++)
                if (this._childs[e]._uniqId == t._uniqId) {
                    this._childs.splice(e, 1);
                    break
                }
        },
        _show: function(t, e) {
            return this._currPos = t, e && this.elem("tail").removeAttr("style").css(e), this.domElem.css(t), (!this._isShown || this._isHiding) && this.hasMod("animate", "yes") && this.afterShow(), this._isHiding=!1, this.setMod("visibility", "visible"), this
        },
        _onShown: function() {
            this.bindToDoc(e, function(e) {
                27 === e.keyCode && (t(this.__self.doc[0].activeElement).blur(), this.hide())
            }), this._attachUnder(), this._isShown=!0, this.hasMod("autoclosable", "yes") && this.afterCurrentEvent(function() {
                this._enableAutoclosable()
            }), this.hasMod("adaptive", "yes") && this._enableAdaptive();
            var n = t.browser;
            if (n.msie && parseInt(n.version, 10) >= 9) {
                var i = [this.domElem, this.elem("content")], o = i.map(function(t) {
                    return parseInt(t.css("z-index"), 10) || ""
                });
                this.domElem[0].onresize = function() {
                    i.forEach(function(t, e) {
                        t.css("z-index", o[e] + 1)
                    }), setTimeout(function() {
                        i.forEach(function(t, e) {
                            t.css("z-index", o[e])
                        })
                    }, 0)
                }
            }
            this.trigger("show")
        },
        _onHidden: function() {
            this.unbindFromDoc(e), this._detachUnder(), this.hasMod("autoclosable", "yes") && this._disableAutoclosable(), this.hasMod("adaptive", "yes") && this._disableAdaptive(), this._cache = {}, this._isShown=!1, this._isHiding=!1;
            var n = t.browser;
            n.msie && parseInt(n.version, 10) >= 9 && (this.domElem[0].onresize = null), this.trigger("hide")
        },
        _mergeParams: function(t, e, i) {
            var o = {};
            return i || (i = {}), t && "object" == typeof t && Object.keys(t).forEach(function(e) {
                o[e] = t[e]
            }), Object.keys(e).forEach(function(s) {
                var r = n.call(i, s) ? i[s].call(this, t[s], e[s]): e[s];
                o[s]=!r || "object" != typeof r || Array.isArray(r) ? r : t[s] ? this._mergeParams(t[s], r, i) : r
            }, this), o
        },
        _offsetParamsHook: function(t, e) {
            return "number" == typeof e ? {
                top: e,
                bottom: e,
                left: e,
                right: e
            } : e
        },
        _tailParamsHook: function(t, e) {
            return e.offset || (e.offset = this.params.tail.offset), e
        },
        _repackDirParams: function(e) {
            var n = {}, i = [];
            return ("string" == typeof e || t.isPlainObject(e)) && (e = [e]), i = e.map(function(t) {
                if ("string" == typeof t) {
                    var e = t.split("-");
                    t = {
                        to: e[0],
                        tail: {}
                    }, e[1] && (t.axis = e[1]), e[2] && (t.tail.axis = e[2])
                }
                var i = t.to;
                return n[i] || (n[i] = t), t.axis && (i += "-" + t.axis), t.key = i, n[i] = t, i
            }, this), {
                directions: n,
                keys: i
            }
        },
        _pickDirection: function() {
            var t, e = this._order, n = this.hasMod("adaptive", "yes") ? e.length: 1;
            for (this._visibilityFactor = 0, t = 0; n > t; t++) {
                var i = e[t], o = this._directions[i];
                this._resetPos(i)._pushPos(i, this._calcPos(o))._pushPos(i, this._calcOffsets(o)), this._hasTail() && this._resetTailPos(i)._pushTailPos(i, this._calcTailPos(o))._pushTailPos(i, this._calcTailOffset(o))._pushPos(i, this._calcOffsetByTail(o)), this._pushPos(i, this._getParentOffset());
                var s = this._calcVisibilityFactor(o);
                if ((s > this._visibilityFactor ||!this._direction) && (this._visibilityFactor = s, this._direction = this._directions[i], 100 == s))
                    break
            }
            return this._direction
        },
        _getParentOffset: function() {
            var t = this.domElem.offsetParent().offset();
            return t.left*=-1, t.top*=-1, t
        },
        _calcPos: function(t) {
            var e = this.getOwnerPos(), i = this.getOwnerSize(), o = this.getPopupSize(), s = t.axis, r = this.params.position || {}, a = {};
            switch (t.to) {
            case"bottom":
                a = {
                    top: n.call(r, "top") ? r.top: e.top + i.height,
                    left: n.call(r, "left") ? r.left: this._calcLeft(s)
                };
                break;
            case"top":
                a = {
                    top: n.call(r, "top") ? r.top: e.top - o.height,
                    left: n.call(r, "left") ? r.left: this._calcLeft(s)
                };
                break;
            case"left":
                a = {
                    top: n.call(r, "top") ? r.top: this._calcTop(s),
                    left: n.call(r, "left") ? r.left: e.left - o.width
                };
                break;
            case"right":
                a = {
                    top: n.call(r, "top") ? r.top: this._calcTop(s),
                    left: n.call(r, "left") ? r.left: e.left + i.width
                }
            }
            return a
        },
        _calcTop: function(t) {
            var e = 0, n = this.getPopupSize(), i = this.getOwnerPos(), o = this.getOwnerSize();
            return "top" == t ? e += i.top : "middle" == t ? e += i.top + o.height / 2 - n.height / 2 : "bottom" == t && (e += i.top + o.height - n.height), e
        },
        _calcLeft: function(t) {
            var e = 0, n = this.getPopupSize(), i = this.getOwnerPos(), o = this.getOwnerSize();
            return "left" == t ? e += i.left : "center" == t ? e += i.left + o.width / 2 - n.width / 2 : "right" == t && (e += i.left + o.width - n.width), e
        },
        _calcOffsets: function(t) {
            var e, n = this._cache.offset || (this._cache.offset = {}), i = t.key, o = t.offset;
            return n[i] ? n[i] : o ? (e = {
                left: 0,
                top: 0
            }, o.left && (e.left += o.left), o.right && (e.left -= o.right), o.top && (e.top += o.top), o.bottom && (e.top -= o.bottom), n[i] = e) : !1
        },
        _hasTail: function() {
            return 0 !== this.elem("tail").length
        },
        _moveToContainer: function() {
            var t = this._container = this._parent ? this._parent.domElem: this._body;
            this._inContainer ? t.children(":last")[0] === this.domElem[0] || this.domElem.appendTo(t) : this._inContainer=!!this.domElem.appendTo(t)
        },
        _resetPos: function(t) {
            return t ? this._positions[t] = null : this._positions = {}, this
        },
        _pushPosTo: function(t, e, n) {
            n!==!1 && ("string" == typeof e ? this._sum(t[e] || (t[e] = {}), n) : (n = e, Object.keys(t).forEach(function(e) {
                this._sum(t[e], n)
            }, this)))
        },
        _pushPos: function(t, e) {
            return this._pushPosTo(this._positions, t, e), this
        },
        _sum: function(t, e) {
            Object.keys(e).forEach(function(n) {
                t[n] = (t[n] || 0) + e[n]
            })
        },
        _getSizeOf: function(t) {
            return {
                height: t.outerHeight(),
                width: t.outerWidth()
            }
        },
        getOwnerSize: function() {
            return this._owner ? this._cache.ownerSize || (this._cache.ownerSize = this._getSizeOf(this._owner)) : {
                height: 0,
                width: 0
            }
        },
        getPopupSize: function() {
            return this._getSizeOf(this.domElem)
        },
        _getPosOf: function(t) {
            return t.offset() || {
                left: 0,
                top: 0
            }
        },
        getOwnerPos: function() {
            var t;
            if (this._owner && (t = this._getPosOf(this._owner), this.hasMod("position", "fixed"))) {
                var e = this.__self.win;
                t.top -= e.scrollTop(), t.left -= e.scrollLeft()
            }
            return t || this._userPosition
        },
        _calcVisibilityFactor: function(t) {
            var e, n, i = this.__self.getWindowSize(), o = this.getPopupSize(), s = this._positions[t.key], r = this.__self.win, a = this._parent ? this._parent.domElem.offset(): {
                top: 0,
                left: 0
            }, c = s.top + (this._isParentFixed ? a.top : - r.scrollTop()), u = s.left + (this._isParentFixed ? a.left : - r.scrollLeft()), l = u + o.width - i.width, d = c + o.height - i.height, h = {
                height: o.height,
                width: o.width
            }, f = 100;
            return d > 0 && (h.height -= d), 0 > c && (h.height += c), 0 > u && (h.width += u), l > 0 && (h.width -= l), h.height < 0 || h.width < 0 ? f = 0 : (n = Math.abs(h.height * h.width), e = o.height * o.width, e != n && (f = n / e * 100)), f
        },
        destruct: function() {
            var t = arguments;
            return this._childs.forEach(function(e) {
                e.destruct.apply(e, t)
            }), this.__base.apply(this, t)
        }
    }, {
        live: function() {
            this.liveBindTo("close", "leftclick", function() {
                this.hide()
            })
        }
    })
}(jQuery), BEM.DOM.decl("popup", {
    isHiding: function() {
        return this._isHiding
    }
}), BEM.DOM.decl({
    block: "popup",
    modName: "autoclosable",
    modVal: "yes"
}, {
    onSetMod: {
        autoclosable: {
            "": function() {
                this._disableAutoclosable()
            }
        }
    },
    _enableAutoclosable: function() {
        var t = this._under;
        this.hasMod(t, "type", "paranja") && (t.is("iframe") && (t = $([t[0].contentWindow, t[0].contentWindow.document])), this.bindTo(t, "leftclick tap", function(t) {
            t.stopPropagation(), this.hide()
        })), this.bindToDoc("leftclick tap", function(t) {
            if (!this._isRelatedNode($(t.target))) {
                var e = $.Event("outside-click");
                this.trigger(e, t), e.isDefaultPrevented() || this.hide()
            }
        })
    },
    _disableAutoclosable: function() {
        this.hasMod(this._under, "type", "paranja") && this.unbindFrom(this._under, "leftclick tap"), this.unbindFromDoc("leftclick tap")
    },
    _isRelatedNode: function(t) {
        if (this.containsDomElem(t) || this._owner && this.containsDomElem.call({
            domElem: this._owner
        }, t))
            return !0;
        var e, n = this._childs.length;
        for (e = 0; n > e; e++)
            if (this.containsDomElem.call({
                domElem: this._childs[e].domElem
            }, t))
                return !0;
        return !1
    }
}), BEM.DOM.decl("popup", {
    _enableAutoclosable: function() {
        this.__base.apply(this, arguments), this.bindToDoc("mousedown", function(t) {
            if (1 === t.which&&!this._isRelatedNode($(t.target)) && "yes" !== this.params.disableoutside) {
                var e = $.Event("outside-click");
                this.trigger(e, t), e.isDefaultPrevented() || (this._skipReturnFocus=!0, this.hide())
            }
        })
    },
    _disableAutoclosable: function() {
        this.__base.apply(this, arguments), this.unbindFromDoc("mousedown")
    }
}), BEM.DOM.decl({
    block: "popup",
    modName: "adaptive",
    modVal: "yes"
}, {
    onSetMod: {
        adaptive: {
            yes: function() {
                this._enableAdaptive()
            },
            no: function() {
                this._disableAdaptive()
            }
        },
        "watch-scroll": {
            yes: function() {
                this._watchScroll()
            },
            no: function() {
                this._unwatchScroll()
            }
        },
        visibility: {
            visible: function() {
                this.__base(), this.hasMod("adaptive", "yes") && this._enableAdaptive()
            },
            "": function() {
                this.__base(), this._disableAdaptive()
            }
        }
    },
    _enableAdaptive: function() {
        this.bindToWin("resize", this.onResize)._watchScroll()
    },
    _disableAdaptive: function() {
        this.unbindFromWin("resize")._unwatchScroll()
    },
    getScrollEvents: function() {
        return ["scroll"]
    },
    _watchScroll: function() {
        this._owner&&!this.hasMod("watch-scroll", "no") && this.bindTo(this._owner.parents().add(this.__self.win), this.getScrollEvents().join(" "), this.onScroll, this)
    },
    _unwatchScroll: function() {
        this._owner && this.unbindFromDomElem(this._owner.parents().add(this.__self.win), this.getScrollEvents().join(" "))
    },
    onResize: function() {
        this._cache = {}, this._isHiding || this.repaint()
    },
    onScroll: function() {
        this._cache = {}, this._isHiding || this.repaint()
    },
    destruct: function() {
        this._disableAdaptive(), this.__base.apply(this, arguments)
    }
}), BEM.DOM.decl({
    block: "popup",
    modName: "animate",
    modVal: "yes"
}, {
    afterShow: function() {
        var t = this.getCurrDirection();
        if (t) {
            var e = t.to, n = this.getCurrPos(), i = (this.getPopupSize(), this.params.tail, {
                opacity: 1,
                top: n.top,
                left: n.left
            }), o = {
                opacity: 0,
                top: n.top,
                left: n.left
            };
            "bottom" == e ? o.top += 10 : "top" == e ? o.top -= 10 : "left" == e ? o.left -= 10 : "right" == e && (o.left += 10), this.domElem.stop(!0).css(o).animate(i, 150)
        }
    },
    beforeHide: function(t) {
        var e = this.getCurrDirection();
        if (!e)
            return t();
        var n = e.to, i = this.getCurrPos(), o = (this.getPopupSize(), this.params.tail, {
            top: i.top,
            left: i.left,
            opacity: 0
        });
        return "bottom" == n ? o.top += 10 : "top" == n ? o.top -= 10 : "left" == n ? o.left -= 10 : "right" == n && (o.left += 10), this.domElem.stop(!0, !0).animate(o, 150, t)
    }
}), function() {
    var t = [];
    BEM.DOM.decl("popup", {
        onSetMod: {
            js: function() {
                this.__base.call(this);
                var t = this.findElem("under");
                this._underClassAttr = t.attr("class"), this.isDivEnough() ? this._under = t : (t.remove(), this._under = null), this._underInPool=!1
            }
        },
        isDivEnough: function() {
            return !1
        },
        _createUnder: function() {
            return $('<iframe frameBorder="0"/>')
        },
        _getUnder: function() {
            if (this._under)
                return this._under;
            var e = t.pop();
            return e && (this._underInPool=!1), this._under = e || this._createUnder()
        },
        _attachUnder: function() {
            var t = this._under = this._getUnder();
            t.attr("class", this._underClassAttr), this.hasMod(t, "type", "paranja") ? t.detach().insertBefore(this.domElem) : t.prependTo(this.domElem)
        },
        _detachUnder: function() {
            var e = this._under;
            t.push(e.detach()), this._under = null, this._underInPool=!0
        },
        destruct: function() {
            return this._underInPool && t.pop(), this.__base.apply(this, arguments)
        }
    })
}(), BEM.DOM.decl("popup", {
    _createUnder: function() {
        var t, e = $(".i-ua_browser_yandexbrowser").length || $(".i-ua_browser_chromemobile").length || /(iPad|iPhone|iPod)/g.test(navigator.userAgent);
        if (e)
            return $("<div></div>");
        try {
            t = location.hostname !== document.domain
        } catch (n) {
            t=!0
        }
        return $("<iframe" + ($.browser.msie && $.browser.version < 9 ? ' frameborder="0"' : "") + ($.browser.msie && t ? " src=\"javascript:void(function(){document.open();document.write('&lt;script&gt;document.domain=\\'" + document.domain + "\\'&lt;/script&gt;');document.close();})()\"" : "") + "/>")
    }
}), x.amd.require("statlog", function(t) {
    BEM.DOM.decl("b-sethome", {
        onSetMod: {
            js: function() {
                var e = this, n = this.__self, i = $(".b-line_bar");
                if (e.on = n._hasStorage(), !e.on)
                    return e.elem("close").addClass("i-hidden"), e.domElem.removeClass("i-hidden"), void i.removeClass("lifting");
                if (!n._hasBlock()) {
                    var o = e.findElem("close");
                    if (o) {
                        var s = o.data(t.params.statDataAttr);
                        t.log(s), o.data(t.params.dataReadyKey, s)
                    }
                    e.bindTo("close", "leftclick", function() {
                        e.domElem.addClass("i-hidden"), i.addClass("lifting"), n._blockSetHome()
                    }), e.domElem.removeClass("i-hidden"), i.removeClass("lifting")
                }
            }
        }
    }, {
        _hasStorage: function() {
            return !home.disabledLS
        },
        _blockSetHome: function() {
            home.localStorage.setItem("shforbidden", 1)
        },
        _hasBlock: function() {
            return !!home.localStorage.getItem("shforbidden")
        }
    }), BEM.DOM.decl({
        name: "b-sethome",
        modName: "type",
        modVal: "common"
    }, {
        onSetMod: {
            js: function() {
                this.__base(), !this.on || this.__self._hasBlock()
            }
        }
    }), BEM.DOM.decl({
        name: "b-sethome",
        modName: "type",
        modVal: "ff"
    }, {
        onSetMod: {
            js: function() {
                function t() {
                    n.popup.hide(), $(document).unbind("focus", t), n.dragIcon.unbind("mousedown", e)
                }
                function e() {
                    setTimeout(function() {
                        $(document).focus(t)
                    }, 500)
                }
                if (this.__base(), this.on&&!this.__self._hasBlock()) {
                    var n = this;
                    n.dropdown = n.findBlockInside("dropdown"), n.popup = n.findBlockInside({
                        block: "popup",
                        modName: "sethome",
                        modVal: "fx"
                    }), n.dragIcon = $(".b-sethome__drag-icon"), n.popup.on({
                        show: function() {
                            n.dragIcon.bind("mousedown", e)
                        },
                        hide: function() {
                            $(document).unbind("focus", t), n.dragIcon.unbind("mousedown", e)
                        }
                    }), n.bindTo("toggle-popup", "leftclick", function() {
                        n.popup.isShown() || n.popup.show(n.dropdown.elem("switcher"))
                    })
                }
            }
        }
    }), BEM.DOM.decl({
        name: "b-sethome",
        modName: "type",
        modVal: "ie"
    }, {
        onSetMod: {
            js: function() {
                if (this.__base(), this.on&&!this.__self._hasBlock()) {
                    var t = this;
                    t.homeUrl = location.protocol + "//" + location.hostname + "/", t._getHomePage(), t.bindTo("link", "leftclick", function() {
                        t._setHomePage()
                    })
                }
            }
        },
        _setHomePage: function() {
            try {
                return document.body.style.behavior = "url(#default#homepage)", document.body.setHomePage(this.homeUrl), this._getHomePage(), !1
            } catch (t) {}
        },
        _getHomePage: function() {
            var t = document.body, e = Lego.params["lang-zone"] || "ru";
            return t.style.behavior = "url(#default#homepage)", t.isHomePage("http://yandex." + e + "/") || t.isHomePage("http://www.yandex." + e + "/") || t.isHomePage(this.homeUrl) ? (this.setMod("show", "bar"), !0) : void 0
        }
    }), BEM.DOM.decl({
        name: "b-sethome",
        modName: "type",
        modVal: "chrome"
    }, {
        onSetMod: {
            js: function() {
                var e = this, n = "chrome_ext";
                e.__base(), e.on&&!e.__self._hasBlock() && (e.params["chrome-exp"] && (n += "_exp"), e.bindTo("link", "leftclick", function() {
                    return window.chrome.webstore.install("", function() {
                        e.domElem.addClass("i-hidden"), t.log("head.sethome." + n + ".yes")
                    }, function() {
                        t.log("head.sethome." + n + ".no")
                    }), !1
                }))
            }
        }
    })
}), $(function() {
    BEM.DOM.init()
}), BEM.DOM.decl("dropdown", {
    onSetMod: {
        js: function() {
            var t = this, e = this.params, n = this._popup = this.findBlockOn(this.elem(e.popupElem), "popup").on("show", function() {
                t.setMod("action", "open").trigger("show")
            }).on("hide", function() {
                t.setMod("action", "closed").trigger("hide")
            });
            e.popupMethods.map(function(e) {
                t[e] = n[e].bind(n)
            });
            var i = this.params.switcherBlock, o = this.elem("switcher");
            i && this.findBlockOn(o, i).on("click", function() {
                n.toggle(o)
            })
        }
    },
    getDefaultParams: function() {
        return {
            popupElem: "popup",
            popupMethods: ["toggle", "show", "hide", "setContent"]
        }
    },
    destruct: function() {
        this._popup.destruct.apply(this._popup, arguments), this.__base.apply(this, arguments)
    }
}, {
    live: function() {
        this.liveBindTo("switcher", "leftclick", function() {
            this.params.switcherBlock || this._popup.toggle(this.elem("switcher"))
        })
    }
}), BEM.DOM.decl("popup", {
    onSetMod: {
        js: function() {
            this.__base(), this._tailPos = {}
        }
    },
    _calcTailPos: function(t) {
        var e = t.to, n = this._positions[t.key], i = t.tail.axis, o = {};
        return "top" == e || "bottom" == e ? o.left = this._calcTailLeft(i, n) : ("left" == e || "right" == e) && (o.top = this._calcTailTop(i, n)), o
    },
    _calcTailTop: function(t, e) {
        var n = 0, i = this.getOwnerSize(), o = this.getOwnerPos(), s = this.params.tail.width, r = this.getPopupSize(), a = r.height, c = o.top - e.top, u = e.top + r.height - (o.top + i.height);
        return c > 0 && (n += c, a -= c), u > 0 && (a -= u), "middle" == t ? (a -= s, n += a / 2) : "bottom" == t && (a -= s, n += a), 0 > n && (n = 0), n
    },
    _calcTailLeft: function(t, e) {
        var n = 0, i = this.getOwnerSize(), o = this.getOwnerPos(), s = this.params.tail.width, r = this.getPopupSize(), a = o.left - e.left, c = r.width, u = e.left + r.width - (o.left + i.width);
        return a > 0 && (n += a, c -= a), u > 0 && (c -= u), "center" == t ? (c -= s, n += c / 2) : "right" == t && (c -= s, n += c), n
    },
    _calcOffsetByTail: function(t) {
        var e = this.params.tail, n = e.height, i = {};
        switch (t.to) {
        case"top":
            i = {
                top: - n
            };
            break;
        case"bottom":
            i = {
                top: n
            };
            break;
        case"right":
            i = {
                left: n
            };
            break;
        case"left":
            i = {
                left: - n
            }
        }
        return i
    },
    _calcTailOffset: function(t) {
        var e = {}, n = t.to, i = t.tail, o = i.offset, s = i.axis;
        return o ? ("top" == n || "bottom" == n ? (e.left = 0, "left" == s ? e.left += o.left : "center" == s ? (o.left && (e.left += o.left), o.right && (e.left -= o.right)) : "right" == s && (e.left -= o.right)) : ("left" == n || "right" == n) && (e.top = 0, "top" == s ? e.top += o.top : "middle" == s ? (o.top && (e.top += o.top), o.bottom && (e.top -= o.bottom)) : "bottom" == s && (e.top -= o.bottom)), e) : !1
    },
    _resetTailPos: function(t) {
        return t ? this._tailPos[t] = null : this._tailPos = {}, this
    },
    _pushTailPos: function(t, e) {
        return this._pushPosTo(this._tailPos, t, e), this
    }
}), function(t, e) {
    var n, i = {}, o = BEM.decl("i-loader", {}, {
        load: function(t, s, r, a, c) {
            "string" != typeof s && (c = a, a = r, r = s, s = t), void 0 === c && void 0 !== a && "function" != typeof a && (c = a, a = void 0);
            var u = i[t];
            if (u)
                return void(u.successFns ? (u.successFns.push(r), a && u.errorFns.push(a)) : this.afterCurrentEvent(function() {
                    r()
                }));
            var l = document, d = l.createElement("script"), h = function() {
                o._error(t)
            }, f = s.split("."), p = f.splice( - 2, 2);
            c && p.unshift("string" == typeof c ? c : BEM.blocks["i-global"].param("lang")), e.browser.msie && e.browser.version <= 8 && p.splice( - 1, 0, "ie"), d.type = "text/javascript", d.charset = "utf-8", d.src = f.concat(p).join("."), d.onerror = h, this.afterCurrentEvent(function() {
                (n || (n = e("head")[0])).insertBefore(d, n.firstChild)
            }), i[t] = {
                successFns: [r],
                errorFns: a ? [a]: [],
                timer: setTimeout(h, 2e4)
            }
        },
        loaded: function(e) {
            var n = i[e.id];
            if (clearTimeout(n.timer), e.js && e.js(), e.css && o._appendCss(e.css), e.hcss) {
                var s = [], r = t._ycssjs;
                e.hcss.forEach(function(t) {
                    if (r) {
                        if (t[0]in r)
                            return;
                        r(t[0])
                    }
                    s.push(t[1])
                }), s.length && o._appendCss(s.join(""))
            }
            if (e.hcss2) {
                var a, c = e.hcss2, u = document, l = u.createElement("div"), d = u.body, h = "ycssjs_", s = [];
                l.style.position = "absolute", d.insertBefore(l, d.lastChild.nextSibling), a = window.getComputedStyle ? window.getComputedStyle(l) : l.currentStyle, Object.keys(c).forEach(function(t) {
                    l.className = h + t, "-" === a.top.charAt(0) || s.push(c[t])
                }), a = null, d.removeChild(l), s.length && o._appendCss(s.join(""))
            }
            for (var f, p = n.successFns; f = p.shift();)
                f();
            delete n.successFns
        },
        _error: function(t) {
            var e, n = i[t], o = n.errorFns;
            for (clearTimeout(n.timer);
            e = o.shift();
            )e();
            delete i[t]
        },
        _appendCss: function(t) {
            var e = document.createElement("style");
            e.type = "text/css", n.insertBefore(e, n.firstChild), e.styleSheet ? e.styleSheet.cssText = t : e.appendChild(document.createTextNode(t))
        }
    })
}(this, jQuery), function(t, e) {
    t.DOM.decl("b-head-stripe", {
        onSetMod: {
            js: function() {
                var t = this.params.theme;
                this.hasMod("theme") ? this._show() : this._loadTheme(t, this._show)
            }
        },
        _install: function(t) {
            t.preventDefault();
            var e = this.params.download_url;
            e && (window.location = e)
        },
        _close: function(t) {
            t.preventDefault(), t.stopPropagation();
            var e = this.params.close_url;
            e && ((new Image).src = e), this.domElem.animate({
                "margin-top": "-20px"
            }, 500), this.setMod("state", "closed")
        },
        _loadTheme: function(n, i) {
            if (n) {
                var o = "b-head-stripe_theme_" + n, s = e.param("lego-static-host") + "/blocks-desktop/b-head-stripe/_theme/b-head-stripe_theme_" + n + ".bembundle.js";
                t.blocks["i-loader"].load(o, s, i)
            }
        },
        _show: function() {
            this._generateCounter().delMod("state")
        },
        _generateCounter: function() {
            var t = this.params.counter;
            return t && ((new Image).src = t), this
        }
    }, {
        live: function() {
            return this.liveBindTo("close", "leftclick", function(t) {
                this._close(t)
            }), this.liveBindTo("leftclick", function(t) {
                this._install(t)
            }), !1
        }
    })
}(BEM, BEM.blocks["i-global"], jQuery), BEM.DOM.decl("b-head-stripe", {
    onSetMod: {
        js: function() {
            "use strict";
            this.__base.apply(this, arguments)
        }
    },
    _install: function(t) {
        "use strict";
        var e, n;
        t.preventDefault(), t.target.href ? (e = this.params.set_cntr, n = t.target.href) : (e = this.params.stripe_cntr, n = this.params.stripe_url), e && cp(e), setTimeout(function() {
            n ? window.location = n : this.__base.apply(this, arguments)
        }, 200)
    },
    _close: function(t) {
        "use strict";
        this.__base.apply(this, arguments), $(window).trigger("popupredraw.lego");
        var e = $(".popup_visibility_visible:not(.b-topbar__popup)");
        e.length && e.bem("popup").hide(), t.preventDefault(), t.stopPropagation()
    },
    _changeTheme: function() {},
    _generateCounter: function() {
        "use strict";
        var t = this.__base, e = arguments, n = this;
        return home.delayManager.add("timeBanner.notshown", function() {
            t.apply(n, e)
        }), this
    }
}), BEM.HTML.decl("b-icon", {
    onBlock: function(t) {
        var e, n = {
            src: "//yandex.st/lego/_/La6qi18Z8LwgnZdsAr1qy1GwCwo.gif",
            alt: ""
        }, i = t.params(), o = ["alt", "width", "height"];
        for (i.url && (n.src = i.url); e = o.shift();)
            i[e] && (n[e] = i[e]);
        t.tag("img").attrs(n)
    }
}), $(function() {
    function t(t) {
        this.callback = t, this.items = 0, this.results = {}
    }
    if ("undefined" != typeof diskUploaderParams) {
        t.prototype.add = function(t, e) {
            var n = this;
            return this.items++, function() {
                var i = e.apply(e, arguments);
                n.check(t, i)
            }
        }, t.prototype.check = function(t, e) {
            this.results[t] = e, this.items--, 0 === this.items && this.callback.call(this, this.results)
        }, t.prototype.done = function() {
            this.callback.call(this, this.results)
        };
        var e = function() {
            this.sk = diskUploaderParams.sk, this.files = [], this.statusesTimers = {}, this.areaShown=!1, this.wrapTimer = null, this.uploadQueue = [], this.maxSize = 314572800, this.disabled=!1, this.forceDialogItemTemplate = ya.template('<div class="b-disk-uploader__force_item"><label class="b-disk-uploader__force_label" for="[% id %]"><input type="checkbox" id="[% id %]" class="b-disk-uploader__force_box">[% text %]</label></div>');
            var t = ya.template('<a href="[% href %]" class="b-link b-disk-uploader__go">[% text %]</a>');
            this.goLink = t({
                href: diskUploaderParams.uploadLinkUrl,
                text: diskUploaderParams.uploadLinkText
            })
        };
        e.prototype.push = function(t) {
            this.files.push({
                file: t,
                status: 0
            })
        }, e.prototype.store = function() {
            for (var e, n = new t(function(t) {
                var e, n, o = [], s = 0;
                for (n in t)
                    t.hasOwnProperty(n) && (e = t[n], 2 === e.status && o.push(e), s += e.file.size);
                if (o.length)
                    i.showForceDialog(o);
                else {
                    var r = ya.uid(), a = 0;
                    i.uploadQueue.push(r);
                    for (n in t)
                        t.hasOwnProperty(n) && (t[n].queueId = r, a++);
                    i.uploadFiles()
                }
            }), i = this, o = 0, s = this.files.length; s > o; o++)
                e = this.files[o], (0 === e.status || 3 === e.status) && (e.id || (e.id = ya.uid()), this.storeFile(e, n.add(e.id, this.storeFileHandler)));
            i.hideArea()
        }, e.prototype.storeFile = function(t, e) {
            if (3 === t.status)
                return void e(t);
            $.ajax({
                url: "https://mail.yandex." + diskUploaderParams.zone + "/api/disk_handler?handle=store&upload_from_main=1&sk=" + this.sk + "&path=/disk/" + t.file.name + (t.force ? "&force=1" : ""),
                method: "GET",
                xhrFields: {
                    withCredentials: !0
                },
                dataType: "json",
                success: function(n) {
                    n.oid && n.upload_url ? (t.status = 3, t.oid = n.oid, t.upload_url = n.upload_url) : 47 === n.code && (t.status = 2), e(t)
                },
                failure: function() {
                    t.status =- 1, e(t)
                }
            }), t.status = 1
        }, e.prototype.storeFileHandler = function(t) {
            return t
        }, e.prototype.uploadFiles = function() {
            for (var t, e = 0, n = 0, i = this.files.length; i > n; n++)
                t = this.files[n], 3 === t.status && (this.uploadFile(t), this.checkTimer(t, 1e3, !1), e++);
            1 === e ? csh_ifmsid(diskUploaderParams.statUploadOne) : e > 1 && csh_ifmsid(diskUploaderParams.statUploadMany), this.updateStatuses()
        }, e.prototype.uploadFile = function(t) {
            var e = new FormData, n = this;
            e.append("file", t.file), t.xhr = $.ajax({
                url: t.upload_url,
                data: e,
                cache: !1,
                contentType: !1,
                processData: !1,
                type: "POST",
                error: function() {
                    t.status > 0 && (t.status =- 1, n.updateStatuses())
                }
            }), t.status = 4
        }, e.prototype.showInfoDialog = function(t) {
            this.disabled=!0, this.infoPopupTitle.text(diskUploaderParams[t ? "incorrectFiles": "incorrectFilesPart"]), this.infoPopup.show()
        }, e.prototype.showForceDialog = function(t) {
            for (var e, n = "", i = 0, o = t.length; o > i; i++)
                e = t[i], n += this.forceDialogItemTemplate({
                    id: e.id,
                    text: e.file.name
                });
            $(document).trigger("popupsClose.lego");
            var s = $(".popup_visibility_visible:not(.b-topbar__popup)"), r = $(".i-popup_visibility_visible");
            s.length && s.bem("popup").hide(), r.length && r.bem("i-popup").hide(), this.disabled=!0, this.forcePopupContent.html(n), this.forcePopup.show(), 1 === o && ($(".b-disk-uploader__force_box").attr("checked", "checked").hide(), $(".b-disk-uploader__force_label").css("margin-left", 0)), this.fixParanja()
        }, e.prototype.collectForceFiles = function() {
            var t = this;
            $(".b-disk-uploader__force_box", this.forcePopupContent).each(function() {
                var e = t.gitFileInfo(this.id);
                e && 2 === e.status && (this.checked ? (e.status = 0, e.force=!0) : e.status =- 2)
            }), this.store(), this.forcePopup.hide()
        }, e.prototype.gitFileInfo = function(t) {
            for (var e = 0, n = this.files.length; n > e; e++)
                if (this.files[e].id === t)
                    return this.files[e];
            return null
        }, e.prototype.checkTimer = function(t, e, n) {
            var i = this;
            this.statusesTimers[t.id] = setTimeout(function() {
                n ? i.checkFocus(t) : i.checkStatus(t)
            }, e)
        }, e.prototype.focusEvents = function() {
            this.windowFocused=!0;
            var t = this;
            $.browser.msie ? ($(document).bind("focusout", function() {
                t.windowFocused=!1
            }), $(document).bind("focusin", function() {
                t.windowFocused=!0
            })) : ($(window).focus(function() {
                t.windowFocused=!0
            }), $(window).blur(function() {
                t.windowFocused=!1
            }))
        }, e.prototype.checkFocus = function(t) {
            var e = this;
            if (this.windowFocused)
                this.checkStatus(t);
            else {
                var n = function() {
                    $.browser.msie ? $(document).unbind("focusin", n) : $(window).unbind("focus", n);
                    var i = Math.round(4e3 * Math.random());
                    setTimeout(function() {
                        e.checkStatus(t)
                    }, i)
                };
                $.browser.msie ? $(document).bind("focusin", n) : $(window).bind("focus", n)
            }
        }, e.prototype.checkStatus = function(t) {
            var e = this;
            $.ajax({
                url: "https://mail.yandex." + diskUploaderParams.zone + "/api/disk_handler?handle=status&sk=" + this.sk + "&oid=" + t.oid,
                method: "GET",
                xhrFields: {
                    withCredentials: !0
                },
                dataType: "json",
                success: function(n) {
                    t.mpfs = n, t.status > 0 && ("WAITING" === t.mpfs.status || "EXECUTING" === t.mpfs.status ? e.checkTimer(t, 2e3, !0) : "DONE" === t.mpfs.status ? (t.status = 5, t.mpfs.progress = 100) : t.status =- 1, e.updateStatuses())
                },
                failure: function() {
                    t.status =- 1, e.updateStatuses()
                }
            })
        }, e.prototype.updateStatuses = function() {
            for (var t, e, n, i = 0, o = 0, s = [], r = 0, a = 0, c = 0, u = "", l = 0, d = this.uploadQueue.length; d > l; l++) {
                e = this.getFilesByQueue(this.uploadQueue[l]), i += e.length;
                for (var h = 0, f = e.length; f > h; h++)
                    n = e[h], 4 === n.status ? (t = n.mpfs && n.mpfs.progress ? n.mpfs.progress : 0, o += t, s.push(n), r++) : 5 === n.status ? (o += 100, a++) : - 1 === n.status && (u = n.file.name, c++)
            }
            if (r > 0)
                this.showStatusesWrap(), this.showBg(), this.showCancel(), this.progress.css("width", Math.floor(o / i) + "%"), this.title.removeClass("b-disk-uploader__statuses_title_error"), this.title.html(1 === r ? ya.htmlFilter(s[0].file.name) : diskUploaderParams.uploadingFiles.replace("%1", a).replace("%2", i));
            else {
                var p, m=!1;
                1 === c ? (p = diskUploaderParams.errorFile.replace("%1", ya.htmlFilter(u)), m=!0) : c > 1 ? (p = diskUploaderParams.errorFiles, m=!0) : p = diskUploaderParams.loaded, this.finishUpload(p, m)
            }
        }, e.prototype.finishUpload = function(t, e) {
            this.uploadQueue = [], this.hideStatusesWrap(), this.hideBg(), this.hideCancel(), this.title.removeClass("b-disk-uploader__statuses_title_error"), "undefined" != typeof e && e ? this.title.addClass("b-disk-uploader__statuses_title_error") : t += this.goLink, this.title.html(t)
        }, e.prototype.getFilesByQueue = function(t) {
            for (var e = [], n = 0, i = this.files.length; i > n; n++)
                this.files[n].queueId === t && e.push(this.files[n]);
            return e
        }, e.prototype.addCancelEvent = function() {
            var t, e = this;
            this.cancel.click(function() {
                for (var n = 0, i = e.files.length; i > n; n++)
                    t = e.files[n], t.xhr && 5 !== t.status && (t.status =- 3, t.xhr.abort(), clearTimeout(e.statusesTimers[t.id]));
                return e.finishUpload(diskUploaderParams.cancel, !0), !1
            })
        }, e.prototype.addCancelDialogEvent = function() {
            var t = this;
            this.infoPopup.on("hide", function() {
                t.disabled=!1
            }), this.forcePopup.on("hide", function() {
                t.disabled=!1, t.clearUploadQueue(), t.unfixParanja()
            }), $("#diskUploadCancel").click(function() {
                t.forcePopup.hide()
            })
        }, e.prototype.clearUploadQueue = function() {
            for (var t, e = 0, n = this.files.length; n > e; e++)
                t = this.files[e], (2 === t.status || 3 === t.status) && (t.status =- 3)
        }, e.prototype.fixParanja = function() {
            $(".i-popup__under_type_paranja").addClass("b-disk-force-popup__under"), $(".b-disk-force").closest(".i-popup").addClass("b-disk-force-popup"), this.hideBanner()
        }, e.prototype.unfixParanja = function() {
            $(".i-popup__under_type_paranja").removeClass("b-disk-force-popup__under"), $(".b-disk-force").closest(".i-popup").removeClass("b-disk-force-popup"), this.showBanner()
        }, e.prototype.checkFileTypes = function(t) {
            if (!t.types)
                return !1;
            var e=!1;
            if (t.types.length)
                for (var n = 0, i = t.types.length; i > n; n++)
                    "Files" === t.types[n] && (e=!0);
            return e
        }, e.prototype.checkFiles = function(e) {
            if (e.files && this.checkFileTypes(e)) {
                var n = new t(function(t) {
                    var e = 0, n = 0;
                    for (var o in t)
                        t.hasOwnProperty(o) && (++e, t[o].status!==!1 && (++n, i.push(t[o].file)));
                    e === n && e > 0 ? i.store() : i.showInfoDialog(0 === n)
                }), i = this;
                if (e.files.length)
                    for (var o = 0, s = e.files.length; s > o; o++)
                        this.isRegularFile(e.files[o], n.add("file" + o, this.checkFileHandler));
                else 
                    e.types.length && n.done()
            }
        }, e.prototype.isRegularFile = function(t, e) {
            var n;
            try {
                n = t.size
            } catch (i) {
                n = 0
            }
            if (0 === n)
                return void setTimeout(function() {
                e(!1, t)
            }, 0);
            if (t.size > 102400)
                return void setTimeout(function() {
                e(!0, t)
            }, 0);
            if (!window.FileReader || window.opera && navigator.platform.toLowerCase().indexOf("mac")>-1 && "12.00" === window.opera.version())
                setTimeout(function() {
                    e(null, t)
                }, 0);
            else 
                try {
                    var o = new FileReader;
                    o.onerror = function() {
                        o.onloadend = o.onprogress = o.onerror = null, e(!1, t)
                    }, o.onloadend = o.onprogress = function(n) {
                        o.onloadend = o.onprogress = o.onerror = null, "loadend" !== n.type && o.abort(), e(!0, t)
                    }, o.readAsDataURL(t)
            } catch (i) {
                setTimeout(function() {
                    e(!1, t)
                }, 0)
            }
        }, e.prototype.checkFileHandler = function(t, e) {
            return {
                file: e,
                status: t
            }
        }, e.prototype.stopLeaveTimer = function() {
            this.leaveTimer && (clearTimeout(this.leaveTimer), this.leaveTimer = null)
        }, e.prototype.startLeaveTimer = function() {
            this.stopLeaveTimer(), this.leaveTimer = setTimeout(function() {
                this.hideArea()
            }.bind(this), 50)
        }, e.prototype.init = function() {
            if (window.File && window.FileList && window.FileReader && $(".b-disk-uploader").length) {
                this.uploaderArea = $(".b-disk-uploader"), this.innerArea = $(".b-disk-uploader__inner"), this.forcePopup = $(".b-disk-force").bem("popup"), this.forcePopupContent = $(".b-disk-uploader__force_content"), this.infoPopup = $(".b-disk-info").bem("popup"), this.infoPopupTitle = $(".b-disk-uploader__info_title"), this.statusesWrap = $(".b-disk-uploader__statuses_wrap"), this.progress = $(".b-disk-uploader__statuses_progress"), this.bg = $(".b-disk-uploader__statuses_bg"), this.title = $(".b-disk-uploader__statuses_title"), this.cancel = $("#diskCancel");
                var t = this;
                $(window).bind({
                    dragenter: function(e) {
                        return !t.disabled && t.checkFileTypes(e.originalEvent.dataTransfer) ? (t.areaShown || t.showArea(), !1) : void 0
                    },
                    dragover: function() {
                        return t.stopLeaveTimer(), !1
                    },
                    dragleave: function() {
                        return !1
                    },
                    drop: function(e) {
                        return t.disabled || t.checkFiles(e.originalEvent.dataTransfer), !1
                    }
                }), this.uploaderArea.bind("dragleave", this.startLeaveTimer.bind(this)), this.uploaderArea.click(function() {
                    return t.hideArea(), !1
                }), $(document).mouseout(function(e) {
                    var n = e.relatedTarget || e.toElement;
                    n && "HTML" !== n.nodeName || t.hideArea()
                }), $(document).mouseover(function() {
                    t.areaShown && t.hideArea()
                }), $("#diskUploadOk").click(function() {
                    return t.collectForceFiles(), !1
                }), $("#diskUploadClose").click(function() {
                    return t.disabled=!1, t.infoPopup.hide(), t.store(), !1
                }), this.focusEvents(), this.addCancelEvent(), this.addCancelDialogEvent()
            }
        }, e.prototype.hideBanner = function() {
            window.home && home.leftBanner && home.leftBanner.hideBanner && home.leftBanner.hideBanner("b-disk-uploader")
        }, e.prototype.showBanner = function() {
            window.home && home.leftBanner && home.leftBanner.hideBanner && home.leftBanner.showBanner("b-disk-uploader")
        }, e.prototype.showArea = function() {
            this.areaShown || (this.uploaderArea.addClass("b-disk-uploader__highlighted"), this.hideBanner(), this.areaShown=!0)
        }, e.prototype.hideArea = function() {
            this.areaShown && (this.uploaderArea.removeClass("b-disk-uploader__highlighted"), this.showBanner(), this.areaShown=!1)
        }, e.prototype.showStatusesWrap = function() {
            this.wrapTimer && clearTimeout(this.wrapTimer), this.statusesWrap.hasClass("i-hidden") && (this.statusesWrap.css({
                bottom: - 44
            }).removeClass("i-hidden").animate({
                bottom: 0
            }, 500), this.hideBanner())
        }, e.prototype.hideStatusesWrap = function() {
            var t = this;
            this.statusesWrap.hasClass("i-hidden") || (this.wrapTimer = setTimeout(function() {
                t.statusesWrap.animate({
                    bottom: - 44
                }, 500, function() {
                    t.statusesWrap.css({
                        bottom: 0
                    }).addClass("i-hidden"), t.showBanner()
                })
            }, 1e4))
        }, e.prototype.showBg = function() {
            this.bg.removeClass("i-hidden")
        }, e.prototype.hideBg = function() {
            this.bg.addClass("i-hidden")
        }, e.prototype.showCancel = function() {
            this.cancel.removeClass("i-hidden")
        }, e.prototype.hideCancel = function() {
            this.cancel.addClass("i-hidden")
        };
        var n = new e;
        n.init()
    }
}), function() {
    "use strict";
    BEM.DOM.decl({
        name: "popup",
        modName: "focus",
        modVal: "capture"
    }, {
        _onShown: function() {
            this.__base.apply(this, arguments), this.domElem.attr("tabindex", "0").focus()
        },
        _onHidden: function() {
            this.__base.apply(this, arguments), this.domElem.removeAttr("tabindex")
        }
    })
}(), function() {
    function t() {
        var t = document.visibilityState || document.webkitVisibilityState || document.msVisibilityState;
        return "prerender" === t
    }
    function e() {
        return "//" + (BEM.blocks["i-global"].param("click-host") || "clck.yandex.ru") + "/click"
    }
    function n() {
        var t, n = "/dtype=stred/pid=1/cid=72202";
        n += "/reqid=" + l.reqid, n += "/path=690.1033";
        var o = {
            1036: d.domainLookupStart - h,
            1037: d.domainLookupEnd - d.domainLookupStart,
            1038: d.connectEnd - d.connectStart,
            1039: d.responseStart - d.requestStart,
            1040: d.responseEnd - d.responseStart,
            1040.318: d.responseEnd - h
        };
        for (t in _)
            _[t] > 0 ? o[t] = Math.min(f * m, _[t]) : 0 === _[t] && (o[t] = f * m);
        for (t in l.vars)
            l.vars.hasOwnProperty(t) && (o[t] = l.vars[t]);
        for (t in o)
            o[t] < 0 && delete o[t];
        o[1042] = encodeURIComponent(navigator.userAgent), n += "/vars=" + i(o), n += "/slots=" + (u("exp.data") || "") + "/*", (new Image).src = e() + n
    }
    function i(t) {
        var e = [];
        for (var n in t)
            t.hasOwnProperty(n) && e.push(n + "=" + t[n]);
        return e.join(",")
    }
    function o() {
        var e =- 1;
        if (t())
            return e;
        if (window.chrome && "function" == typeof window.chrome.loadTimes && h) {
            var n = window.chrome.loadTimes();
            e = n.firstPaintTime ? Math.floor(1e3 * n.firstPaintTime) - h : 0
        }
        return e
    }
    function s() {
        var e =- 1;
        return t() ? e : ("undefined" != typeof d.msFirstPaint && (e = d.msFirstPaint ? d.msFirstPaint - h : 0, d.msFirstPaint && d.responseStart && d.msFirstPaint < d.responseStart && (e =- 1)), e)
    }
    function r() {
        var e =- 1;
        if (t())
            return e;
        if (window.chrome && "function" == typeof window.chrome.loadTimes && h) {
            var n = window.chrome.loadTimes();
            e = n.firstPaintAfterLoadTime ? Math.floor(1e3 * n.firstPaintAfterLoadTime) - h : 0
        }
        return e
    }
    function a() {
        if (_[1051] > 0) {
            if (_[1041.659] > 0)
                return !0;
            if (_["1041.660"] > 0 && _["1050.660"] > 0)
                return !0
        }
    }
    function c() {
        ++p, _[1041.659] = s(), _["1041.660"] = o(), _["1050.660"] = r(), _[1041.659] > 0 ? _[1041.318] = _[1041.659] : _["1041.660"] > 0 && (_[1041.318] = _["1041.660"]), _[1041.318] > 0 && (_[1041.906] = _[1041.318] - (d.domainLookupStart - h), _[1041] = _[1041.318] - (d.responseStart - h)), l || (l = u("b-timing.data")), l && (a() || p === m) ? n() : setTimeout(c, f)
    }
    function u(t) {
        return home && home.getData && home.getData(t)
    }
    var l, d = window.performance && window.performance.timing;
    if (d && d.navigationStart) {
        var h = d.navigationStart, f = 1e3, p = 0, m = 10, _ = {
            1041.659: - 1,
            "1041.660": - 1,
            "1050.660": - 1,
            1041: - 1,
            1041.906: - 1,
            1041.318: - 1,
            1051: 0
        };
        $(function() {
            _[1051] =+ new Date - h
        }), setTimeout(c, f)
    }
}(), function(t) {
    t.getData = function(e, n) {
        var i, o, s, r, a = "export", c = e.split("."), u = c.length;
        if (u > 1) {
            for (o = t[a] || {}, s = 0; u > s; s++)
                if (i = c[s], o = o[i], "undefined" == typeof o)
                    return "undefined" == typeof o ? n : o;
            return "undefined" == typeof o ? n : o
        }
        return a = "export", r = t && t[a] && t[a][e], "undefined" == typeof r ? n : r
    }
}(home), $(function() {
    var t = $(".b-yabrowser-notice"), e = $("body");
    if (t.length) {
        var n = t.data(), i = n.show, o = n.close, s = n.bk;
        if (home.localStorage.getItem("znotice") === s)
            return o && ((new Image).src = o), void home.localStorage.clear("znotice");
        i && ((new Image).src = i), $.browser.webkit && (window.fifa2014Urls || window.olymp2014Urls) && $(".b-yabrowser-notice").css({
            "-webkit-transform": "translateZ(0)",
            transform: "translateZ(0)"
        }), $(".b-yabrowser-notice__link").click(function() {
            home.localStorage.setItem("znotice", s)
        }), setTimeout(function() {
            t.fadeIn(500), $(".b-yabrowser-notice__close").click(function() {
                return t.addClass("i-hidden"), e.removeClass("b-page_has_yabro-notice"), o && ((new Image).src = o), !1
            })
        }, 100)
    }
}), $(function() {
    var t, e;
    if ("undefined" != typeof Widget) {
        var n = function() {
            t || home && home.getData && (t = home.getData("b-domik-reloaded.reload-url"), e = home.getData("b-domik-reloaded.empty"))
        }, i = 3e5, o = {
            reload: function() {
                n(), t && $.get(t, function(t) {
                    if (t.Mail.Logged) {
                        var n = t.Mail.count, i = t.Mail.fullcountaccus, o = Number(n) ? "<b>" + String(n) + " " + i + "</b>": e;
                        $(".b-topbar__mail .b-topbar__notifier_type_mail .b-link__inner").html(o), $(".b-topbar").bem("b-topbar").popup.domElem.find(".b-topbar__mail .b-topbar__notifier_type_mail .b-link__inner").html(o), $(".b-topbar__body .b-topbar__notifier_type_mail .b-link__inner").html(Number(n) ? "<b>" + String(n) + "</b>" : e), s()
                    }
                }, "json")
            },
            id: "domik"
        }, s = function() {
            setTimeout(function() {
                Widget.prototype.iReload.call(o)
            }, i)
        };
        s()
    }
}), BEM.DOM.decl("button", {
    onSetMod: {
        js: function() {
            var t = this.isDisabled(), e = this.domElem;
            (this._href = e.attr("href")) && t && e.removeAttr("href"), e.attr("disabled", t), this._isFocusable = "a button".indexOf(e[0].tagName.toLowerCase())>-1
        },
        focused: {
            yes: function() {
                return this.isDisabled()?!1 : (this.bindToWin("unload", function() {
                    this.delMod("focused")
                }).bindTo("keydown", this._onKeyDown), this._isFocusable && (this._isControlFocused() || this.domElem.focus()), void this.afterCurrentEvent(function() {
                    this.trigger("focus")
                }))
            },
            "": function() {
                this.unbindFromWin("unload").unbindFrom("keydown"), this._isFocusable && this._isControlFocused() && this.domElem.blur(), this.afterCurrentEvent(function() {
                    this.trigger("blur")
                })
            }
        },
        disabled: function(t, e) {
            var n = "yes" == e, i = this.domElem;
            this._href && (n ? i.removeAttr("href") : i.attr("href", this._href)), n && i.keyup(), this.afterCurrentEvent(function() {
                i.attr("disabled", n)
            })
        },
        pressed: function(t, e) {
            this.setMod("focused", "yes"), this.isDisabled() || this.trigger("yes" == e ? "press" : "release")
        },
        hovered: {
            "": function() {
                this.delMod("pressed")
            }
        },
        "*": function(t) {
            return this.isDisabled() && "hovered pressed".indexOf(t)>-1?!1 : void 0
        }
    },
    isDisabled: function() {
        return this.hasMod("disabled", "yes")
    },
    url: function(t) {
        return "undefined" == typeof t ? this._href : (this._href = t, this.isDisabled() || this.domElem.attr("href", t), this)
    },
    _isControlFocused: function() {
        try {
            return this.containsDomElem($(document.activeElement))
        } catch (t) {
            return !1
        }
    },
    _onKeyDown: function(t) {
        var e = t.keyCode;
        13 != e && 32 != e || this._keyDowned || (this._keyDowned=!0, this.setMod("pressed", "yes").bindTo("keyup", function() {
            this.delMod("pressed").unbindFrom("keyup"), delete this._keyDowned, 32 == e && this.domElem.attr("href") && (document.location = this.domElem.attr("href"))
        }))
    },
    _onClick: function(t) {
        this.isDisabled() ? t.preventDefault() : this.afterCurrentEvent(function() {
            this.trigger("click")
        })
    },
    destruct: function() {
        this.delMod("focused"), this.__base.apply(this, arguments)
    }
}, {
    live: function() {
        var t = {
            mouseover: {
                name: "hovered",
                val: "yes"
            },
            mouseout: {
                name: "hovered"
            },
            mousedown: {
                name: "pressed",
                val: "yes"
            },
            mouseup: {
                name: "pressed"
            },
            focusin: {
                name: "focused",
                val: "yes"
            },
            focusout: {
                name: "focused"
            }
        };
        this.liveBindTo("leftclick", function(t) {
            this._onClick(t)
        }).liveBindTo("mouseover mouseout mouseup focusin focusout", function(e) {
            var n = t[e.type];
            this.setMod(n.name, n.val || "")
        }).liveBindTo("mousedown", function(e) {
            var n = t[e.type];
            1 == e.which && this.setMod(n.name, n.val || ""), this._isControlFocused() && e.preventDefault()
        })
    }
}), BEM.DOM.decl("b-themes-list", {
    onSetMod: {
        js: function() {
            var t = this;
            $(".b-themes-list__tab-item").live("click", function() {
                var e = $(this), n = e.data("tabid");
                t.showTabThemes(n), t.scrollToTheme(t.getSelectedThemeOffsetFromLeft()), t.updateHash(n)
            }), $(".b-themes-list__theme-link").click(function(e) {
                e.preventDefault(), t.addHashAndGo($(this).attr("href"), $(this).data("skinid"), $(this).data("stat"))
            }), $(".b-themes-list__arrow-prev").click(function() {
                t.scrolling(5, 1)
            }), $(".b-themes-list__arrow-next").click(function() {
                t.scrolling( - 5, 1)
            }), window.addEventListener && window.addEventListener("DOMMouseScroll", function(e) {
                t.wheel(e)
            }, !1), window.onmousewheel = document.onmousewheel = function(e) {
                t.wheel(e)
            };
            var e = t.hashQuery(), n = this.params.list;
            if (e.tab && n && n[e.tab] || (e.tab = "all"), this.showTabThemes(e.tab), e.pos)
                e.pos = parseInt(e.pos, 10);
            else {
                var i = t.getSelectedThemeOffsetFromLeft();
                i = t.check(i), e.pos =- i
            }
            if (t.scrollToTheme(e.pos, !0), $.browser.msie && (document.title = document.title.split("#")[0], $.browser.version < 10)) {
                var o = $(".b-themes-list");
                $(window).on("resize", function() {
                    o.toggleClass("ie8reflow")
                })
            }
        }
    },
    hashQuery: function() {
        var t, e = location.hash.substr(1), n = {};
        if (e) {
            e = e.split("&");
            for (var i = 0, o = e.length; o > i; i++)
                t = e[i].split("="), 2 === t.length && (n[t[0]] = t[1])
        }
        return n
    },
    getSelectedThemeOffsetFromLeft: function() {
        var t = this.elem("theme").not(".b-themes-list__theme-hidden"), e = t.index($(".b-themes-list__theme_selected"));
        return - 1 !== e?-68 * (e - 2) : 0
    },
    updateHash: function(t, e) {
        var n = this.hashQuery();
        "all" !== t && t ? n.tab = t : delete n.tab, e ? n.pos = e : delete n.pos;
        var i = "";
        for (var o in n)
            n.hasOwnProperty(o) && (i += o + "=" + n[o] + "&");
        i.length && (i = i.substr(0, i.length - 1)), location.hash = i, $.browser.msie && (document.title = document.title.split("#")[0])
    },
    showTabThemes: function(t) {
        var e = this.elem("theme"), n = e.eq(0).parent(), i = this.params.list;
        $(".b-themes-list__tab_selected").removeClass("b-themes-list__tab_selected"), $(".b-themes-list__tab_" + t).addClass("b-themes-list__tab_selected"), i[t] ? (e.addClass("b-themes-list__theme-hidden"), $.each(i[t], function(t, e) {
            var i = ".b-themes-list__theme_" + e;
            n.append($(i)), $(i).removeClass("b-themes-list__theme-hidden")
        })) : n.append(e.removeClass("b-themes-list__theme-hidden"))
    },
    isNewBrowser: function() {
        var t = /(msie) (\w+\.\w+)/.exec(navigator.userAgent.toLowerCase());
        return t && Number(t[2].split(".")[0]) > 8 ||!t
    },
    scrollToTheme: function(t, e) {
        var n;
        n = t?-t : 0, n = this.check(n);
        var i = $(".b-themes-list__list table");
        e ? (i.css("left", n).show(), setTimeout(function() {
            i.show()
        }, 0)) : (i.animate({
            left: n
        }), setTimeout(function() {
            i.show()
        }, 0))
    },
    showTheme: function(t, e, n) {
        var i = this.elem("theme"), o = i.eq(0).parent(), s = this.params.list;
        i.addClass("b-themes-list__theme-hidden"), $.each(s[t], function(t, e) {
            var n = ".b-themes-list__theme_" + e;
            o.append($(n)), $(".b-themes-list__theme_" + e).removeClass("b-themes-list__theme-hidden")
        }), this.showTabThemes(t), this.scrollToTheme(e || 0, n)
    },
    scrolling: function(t, e) {
        var n = $(".b-themes-list__list table"), i = this.check(parseInt(n.css("left"), 10) + 68 * t);
        e ? n.stop().animate({
            left: i
        }) : n.css({
            left: i
        })
    },
    check: function(t) {
        var e = this.widths(), n = e.windowWidth - e.themesTableWidth - 36;
        return t > 0 || n > 0 ? 0 : n > t ? n : t
    },
    widths: function() {
        var t = $(window).width(), e = parseFloat($(".b-themes-list__list table").width());
        return e = e === t - 36 ? 68 * this.elem("theme").length + 28 : e, {
            themesTableWidth: e,
            windowWidth: t
        }
    },
    scrollThemes: function(t) {
        - 1 > t && (t =- 1), t > 1 && (t = 1), this.scrolling(parseInt(t, 10), 0)
    },
    getCurrentTab: function() {
        return $(".b-themes-list__tab_selected > .b-themes-list__tab-item").data("tabid") || "all"
    },
    addHashAndGo: function(t, e, n) {
        return t += "#tab=" + this.getCurrentTab() + "&pos=" + Math.abs(parseInt($(".b-themes-list__list table").css("left"), 10)), window.statRedirect(t, n), !1
    },
    wheel: function(t) {
        var e, n = 0;
        t ? e = t.target : (t = window.event, e = t.srcElement), $(e).parents(".b-themes-list").length ? (t.wheelDelta ? n = t.wheelDelta / 120 : t.detail && (n =- t.detail / 3), n && (this.scrollThemes(n), t.preventDefault && t.preventDefault(), t.returnValue=!1)) : t.returnValue=!0
    }
}), window.ya = window.ya || {}, ya.history = window.history && history.pushState ? function() {
    return {
        navigate: function(t) {
            t = t || {}, t.url && (t.title || (t.title = ""), t.data || (t.data = ""), t.replace ? history.replaceState(t.data, t.title, t.url) : history.pushState(t.data, t.title, t.url))
        },
        onchange: function(t) {
            $(window).on("popstate", function(e) {
                t.call(this, e.originalEvent)
            })
        }
    }
}() : null, $(document).ready(function() {
    var t = home && home.getData && home.getData("i-change-url.data");
    if (t) {
        var e = t.url;
        if (e && ya.history) {
            var n = home && home.getData && home.getData("mordaFromCache.data");
            if (n) {
                var i = n.lastClid;
                document.location.search.indexOf("?clid=" + i)>-1 && ($("form.search input[name='clid']").length || $("form.search").append('<input type="hidden" name="clid" value="' + i + '">'))
            }
            ya.history.navigate({
                data: {
                    page: "morda"
                },
                url: e,
                replace: !0
            })
        }
    }
}), $(function() {
    function t(t, n) {
        t.append(r), r.fadeIn(500), s = 0, $("#b-wdgt-setup_button").attr("href", a + "#open=_" + n), clearTimeout(o), o = setTimeout(function() {
            e()
        }, 4e3)
    }
    function e() {
        s || r.fadeOut(500)
    }
    function n(n, r) {
        $(".b-newpromo").length || n.mouseenter(function() {
            s || (clearTimeout(o), i = setTimeout(function() {
                t(n, r)
            }, 1e3))
        }).mouseleave(function() {
            clearTimeout(i), clearTimeout(o), o = setTimeout(function() {
                s = 0, e()
            }, 1e3)
        })
    }
    if (!$(".b-page_widgets").length) {
        var i, o, s, r = $(".b-wdgt-control"), a = $("#b-wdgt-setup_button").attr("href");
        r.length && (r.css({
            right: 0,
            top: 0
        }), $("#b-wdgt-send_button").addClass("i-hidden"), $("#b-wdgt-del_button").addClass("i-hidden"), $(".b-wdgt-control").css({
            width: 18,
            height: 18
        }), $(".b-wdgt-control").mouseenter(function() {
            s = 1, clearTimeout(o)
        }), $(".b-wdgt-control").mouseleave(function() {
            o = setTimeout(function() {
                s = 0, e()
            }, 1e3)
        }), window.wdgtControl = n)
    }
}), BEM.DOM.decl({
    block: "popup",
    modName: "type",
    modVal: "modal"
}, {
    getDefaultParams: function() {
        var t = this.__base();
        return t.top = "50%", t.left = "50%", t
    },
    _isPercentVal: function(t) {
        return "string" == typeof t && t.indexOf("%") > 0
    },
    show: function(t) {
        this.setMod("visibility", "outside").setMod("adaptive", "no"), t || (t = {
            left: this.params.left,
            top: this.params.top
        });
        var e = this.getPopupSize();
        return this._isPercentVal(t.left)&&!t.marginLeft && (t.marginLeft = e.width/-(100 / parseInt(t.left, 10))), this._isPercentVal(t.top)&&!t.marginTop && (t.marginTop = e.height/-(100 / parseInt(t.top, 10))), this._show(t), this
    }
}), BEM.DOM.decl({
    block: "popup",
    modName: "position",
    modVal: "fixed"
}, {
    addChild: function(t) {
        this.__base.apply(this, arguments), t.setMod("watch-scroll", "no")
    }
}), BEM.DOM.decl("b-wdgt-hint", {
    onSetMod: {
        js: function() {
            this.bindTo("accept", "click", this.accept), this.bindTo("reject", "click", this.reject), this.popup = this.findBlockOn(this.domElem, "popup"), this.popup.on("hide", this.changeThis(function() {
                this.hasMod("widget")&&!this.hasMod("move") && this.saveValue(0, 1)
            }, this))
        },
        widget: function(t, e) {
            e ? (this._watch = setInterval(this.changeThis(function() {
                this.alignPopup()
            }, this), 300), this.showParanja(), this.bindEvents()) : (clearInterval(this._watch), this.hideParanja(), this.unbindEvents())
        },
        move: function(t, e) {
            this.popup[e ? "hide": "repaint"]()
        }
    },
    alignPopup: function() {
        if (this._widgetParams || (this._widgetParams = {
            offset: {}
        }), !this.hasMod("move")) {
            var t = this._widget.wrapper.$(), e = this._widgetParams, n = {
                offset: t.offset(),
                width: t.width(),
                height: t.height()
            };
            (e.offset.x !== n.offset.x || e.offset.y !== n.offset.y || e.width !== n.width || e.height !== n.height) && (this._widgetParams = n, this.repaint())
        }
    },
    bindEvents: function() {
        $(document).bind("keydown.wdgthint", this.changeThis(this.keydown))
    },
    unbindEvents: function() {
        $(document).unbind("keydown.wdgthint")
    },
    keydown: function(t) {
        switch (t.keyCode) {
        case 13:
            return this.accept(), !1;
        case 27:
            return this.reject(), !1
        }
    },
    showFor: function(t, e, n) {
        t && (this._widget = t, this._noCatalog = e, this._from = n, this.delMod("move"), t.params.yandex ? (this.elem("message").html(translatedMessages.securityMessages.yandexWidgetAdded), this.elem("no-yandex").addClass("i-hidden")) : (this.elem("message").html(translatedMessages.securityMessages.reallyAdd), this.elem("no-yandex").removeClass("i-hidden")))
    },
    showPopup: function() {
        this.popup.show(this._widget.wrapper.$()), this.setMod("widget", this._widget.id), this.delMod("move"), this.elem("accept").focus()
    },
    repaint: function() {
        this.hasMod("widget") && (this.popup._cache.ownerSize = null, this.popup.repaint())
    },
    wipe: function() {
        this._widget = this._from = this._noCatalog = 0, this.delMod("widget")
    },
    accept: function() {
        this.saveValue(1), this.popup.hide()
    },
    reject: function() {
        this.saveValue(0), this.popup.hide()
    },
    saveValue: function(t, e) {
        this._widget && (wg.confirmAddWidget({
            ans: t,
            noCatalog: this._noCatalog,
            from: this._from,
            id: this._widget.id,
            noUpdateHash: e
        }), this.wipe())
    },
    isShown: function() {
        return this.popup.isShown()
    },
    getParanja: function() {
        return $(".v_sform_bg")
    },
    showParanja: function() {
        var t = this.getParanja();
        t.one("click.wdgthint", this.changeThis(function() {
            this.reject()
        }, this)), t.show()
    },
    hideParanja: function() {
        var t = this.getParanja();
        t.off("click.wdgthint"), t.hide()
    }
}), BEM.DOM.decl("b-wdgt-share", {
    onSetMod: {
        js: function() {
            this.bindTo("close", "click", this.hide), this.bindTo("link", "click", this.linkClick)
        }
    },
    show: function(t) {
        this.updateLinks(t), this.getPopup().show()
    },
    hide: function() {
        this.getPopup().hide()
    },
    updateLinks: function(t) {
        var e = t.id.substr(0, t.id.lastIndexOf("-")), n = "http://www.yandex.#d#/?add=#id#&from=shared".replace(/#d#/, document.domain.replace(/yandex\./, "")).replace(/#id#/, e), i = this.params.shareText;
        this.elem("url").val(n), this.elem("link").each(function() {
            var e = $(this), o = e.data("service"), s = e.data("title") || i;
            e.attr("href", "http://share.yandex.ru/go.xml?service=" + encodeURIComponent(o) + "&url=" + encodeURIComponent(n) + "&title=" + encodeURIComponent(s.replace("%s", "«" + t.params.title + "»").replace("(URL)", n)))
        })
    },
    linkClick: function(t) {
        var e = t.currentTarget;
        if ("_blank" !== e.target) {
            {
                window.open(e.href, "shareWindow", "status=1, toolbar=0, menubar=0, resizable=1, scrollbars=1, width=800px, height=500px")
            }
            return !1
        }
    },
    getPopup: function() {
        return this.popup || (this.popup = this.findBlockOn(this.domElem, "popup"))
    }
}); var posRegionHint, initRegionHint; !function(t) {
    function e(t) {
        var e;
        i=!0, e = t.bem("popup"), e.show(n)
    }
    var n, i=!1;
    window.initRegionHint = function(o, s) {
        var r = t(".b-region__hint"), a = t(s), c = t(o), u = r.find(".b-region__hint__header")[0], l = u ? u.onclick(): {};
        r.length > 0 && setTimeout(function() {
            var o;
            if ("1" === l.ispopup) {
                if (n = c, 0 === n.length)
                    return !1;
                e(r)
            } else if (n = a, 0 === n.length)
                return !1;
            o = function() {
                e(r)
            }, n.click(o), t(document).keydown(function(t) {
                "27" === t.keyCode && r.bem("popup").hide()
            }), r.bem("popup").on("hide", function() {
                i=!1
            }), "1" === l.ispopup && r.bem("popup").on("hide", function() {
                n.unbind("click", o)
            }), t(window).resize(function() {
                setTimeout(function() {
                    if (i) {
                        r.bem("popup")._isShowed=!1, r.bem("popup").show(n), i=!0;
                        try {
                            document.body.focus()
                        } catch (t) {}
                    }
                }, 142)
            }), t(".b-region__ajax-action").click(function() {
                return t(document.body).append('<iframe style="display:none" src="' + this.href + '"></iframe>'), r.bem("popup").hide(), !1
            })
        }, 1e3)
    }
}(jQuery), $(document).ready(function() {
    !function() {
        function t() {
            return $(".b-mobile-balloon")[0].onclick()
        }
        $(".b-mobile-balloon__button_yes").click(function() {
            document.location.href = t().url
        }), $(".b-mobile-balloon__button_no").click(function() {
            var e = t().statNo;
            (new Image).src = t().closeurl, e && cp(e), $(".b-mobile-balloon").hide()
        })
    }()
}), $(document).ready(function() {
    function t(t, e) {
        var n = new Date(1e3 * t);
        return e.replace("%day", n.getDate()).replace("%month", i.monthNames[n.getMonth()]).replace("%year", n.getFullYear())
    }
    function e() {
        $(".b-head-stripe").hide(), $(".b-sethome-stripe").hide(), $(".b-https-stripe").hide();
        var t = $(".popup_visibility_visible:not(.b-topbar__popup)");
        t.length && t.bem("popup").hide(), setTimeout(function() {
            $(window).trigger("popupredraw.lego")
        }, 100)
    }
    function n(e, n) {
        r = e.html(), a = new Date(1e3 * n).getFullYear() === new Date(1e3 * i.timestamp).getFullYear() ? i.dateFormatShort : i.dateFormatLong, r = r.replace("%local_date", "<b>(" + t(n, a) + ")</b>").replace("%current_date", "<b>(" + t(i.timestamp, a) + ")</b>"), e.html(r)
    }
    var i = home && home.getData && home.getData("b-time-banner.data");
    if (!i || i.empty)
        return void home.delayManager.runQueue("timeBanner.notshown");
    var o, s, r, a, c, u, l = $(".b-time-banner"), d = Math.round((new Date).getTime() / 1e3);
    i.WrongDateFatal.show && (i.WrongDateFatal.till_ts > d || i.WrongDateFatal.from_ts && i.WrongDateFatal.from_ts < d) ? (o = i.WrongDateFatal, s = $(".b-time-banner_mode_fatal .b-time-banner__text"), n(s, d), $(".b-time-banner_mode_fatal").show(), u = i.showFatalCounter, c = o.show_url, e()) : i.WrongDateAlert.show && (i.WrongDateAlert.till_ts > d || i.WrongDateAlert.from_ts && i.WrongDateAlert.from_ts < d) ? (o = i.WrongDateAlert, s = $(".b-time-banner_mode_alert .b-time-banner__text"), n(s, d), $(".b-time-banner_mode_alert").show(), u = i.showAlertCounter, c = o.show_url, e()) : home.delayManager.runQueue("timeBanner.notshown"), c && ((new Image).src = c), u && csh_ifgsid(u), $(".attention-stripe__close").click(function(t) {
        (new Image).src = o.close_url, csh_ifgsid($(t.target).hasClass("attention-stripe__close_alert") ? i.closeAlertCounter : i.closeFatalCounter), l.hide(), t.preventDefault(), e()
    })
}), function() {
    $(document).ready(function() {
        function t(t) {
            var e = document.createElement("iframe");
            e.style.visibility = "hidden", e.style.position = "absolute", e.setAttribute("src", t), e.style.top = e.style.left = e.style.width = e.style.height = 0, document.getElementsByTagName("body")[0].appendChild(e)
        }
        if (home.getData("b-https-stripe.data.show")) {
            var e = home.getData("b-https-stripe.data.close_url"), n = home.getData("b-https-stripe.data.counter_url");
            t(n), cp(home.getData("b-https-stripe.counters.show")), $(".b-https-stripe .attention-stripe__close").click(function() {
                t(e), cp(home.getData("b-https-stripe.counters.close")), $(".b-https-stripe").hide()
            })
        }
    })
}(), BEM.DOM.decl("multiauth-promo", {
    onSetMod: {
        js: function() {
            if (this.popup = this.findBlockOn(this.domElem, "popup"), this.popup.show($(".user:first .user__icon")), $(".b-topbar").bem("b-topbar").onFirst("expanded", this.changeThis(this.hide)), this.repaint = this.changeThis(function() {
                this.popup.repaint()
            }), $(window).bind("popupredraw.lego", this.repaint), $(window).one("popupopen.lego", this.changeThis(this.hide)), this.params.close) {
                var t = new Image;
                t.src = this.params.close
            }
        }
    },
    hide: function() {
        $(window).unbind("popupredraw.lego", this.repaint), this.popup.delMod("domik"), this.popup.hide()
    }
}), $(function() {
    "use strict";
    var t = function() {
        var t = $(this), e = t.data("id"), n = t.data("cookie"), i = t.data("stat"), o = "b-tabs", s = "b-tabs__selected", r = "b-tabs__notselected", a = "b-tabs__names", c = "b-tabs__items", u = "i-hidden";
        if (t.parent().is("." + s))
            return i && csh_ifmsid(i + ".link", this), !0;
        var l = t.closest("." + o);
        if ($("." + a + " > .b-tabs__title-block", l).removeClass(s).addClass(r), t.closest(".b-tabs__title-block").removeClass(r).addClass(s), $("." + c + " > .b-tabs__block", l).addClass(u), $("#" + e).removeClass(u), i && csh_ifmsid(i + ".select"), t.blur(), n) {
            var d = $(".b-tabs__frame");
            d.length || (d = $('<iframe class="b-tabs__frame" style="display:none"></iframe>'), $(document.body).append(d)), d.attr("src", n)
        }
        if (l.data("message") && window.workerMessage) {
            var h = parseInt(t.attr("name").replace("tab", ""), 10);
            window.workerMessage("update", {
                name: l.data("message"),
                index: h
            })
        }
        return !1
    };
    $("body").on("click", ".b-tabs__title", t)
}), function() {
    var t = function() {
        this.serverDate = null, this.prevDay = null, this.loadDate = null, this.parent = null, this.placeHolders = {
            date: null,
            day: null,
            month: null,
            wday: null,
            time: null,
            h: null,
            m: null
        }, this.flicker = {
            timePH: null,
            init: function(t) {
                this.timePH = t
            },
            blink: function() {
                $(this.timePH).toggleClass("b-datetime__time_hidden")
            }
        }, this.timeTemplate = null
    };
    t.prototype._norm = function(t) {
        return t > 9 ? "" + t : "0" + t
    }, t.prototype.iterate = function() {
        var t = new Date, e = new Date(this.serverDate.getTime() + (t.getTime() - this.loadDate.getTime())), n = e.getDay();
        0 === e.getUTCHours() && 0 === e.getUTCMinutes() && this.prevDay !== n && (this.placeHolders.date.length > 0 && (this.placeHolders.date.replaceWith(this.timeTemplate({
            title: "",
            day: e.getDate(),
            month: this.locale.month[e.getMonth()],
            wday: this.locale.week[n]
        })), this.placeHolders.date = $(".b-datetime__date", this.parent)), this.prevDay = n, $(window).trigger({
            type: "morda.nextDay",
            date: e.getDate(),
            month: this.locale.month[e.getMonth()],
            wday: this.locale.week[n],
            monthServer: e.getMonth(),
            day: n
        })), this.placeHolders.h.html(this._norm(e.getUTCHours())), this.placeHolders.m.html(this._norm(e.getUTCMinutes()))
    }, t.prototype.init = function(t, e) {
        if (e) {
            if (this.parent = e, $(this.parent).hasClass("b-datetime__inited"))
                return;
            if ($(this.parent).addClass("b-datetime__inited"), this.serverDate = new Date(t + " GMT+0000"), this.loadDate = new Date, this.timeTemplate = ya.template(this.infoDate), this.serverDate && "Invalid Date" !== this.serverDate&&!isNaN(this.serverDate)) {
                this.placeHolders.date = $(".b-datetime__date", e), this.placeHolders.day = $(".b-datetime__day", e), this.placeHolders.month = $(".b-datetime__month", e), this.placeHolders.wday = $(".b-datetime__wday", e), this.placeHolders.time = $(".b-datetime__time", e), this.placeHolders.h = $("span:first", this.placeHolders.time), this.placeHolders.m = $("span:last", this.placeHolders.time), this.flicker.init($(".b-datetime__time b", e)), this.iterate();
                var n = this;
                setInterval(function() {
                    n.iterate(), n.flicker.blink()
                }, 1e3)
            }
        }
    }, window.MordaClock = t
}(), function(t) {
    var e = /\[%\s([^%]+)\s%\]/g;
    t.template = function(t, n) {
        n = n || {};
        try {
            var i = t.replace(/'/g, "\\'").replace(e, function(t, e) {
                return n.hasOwnProperty(e) ? "'+(hash.hasOwnProperty(\"" + e + '")?hash["' + e + '"]:"' + n[e] + "\")+'" : "'+(hash.hasOwnProperty(\"" + e + '")&&typeof(hash["' + e + "\"])!='undefined'?hash[\"" + e + "\"]:'')+'"
            });
            return new Function("hash", "return '" + i + "'")
        } catch (o) {
            debug && debug("template make failed:", t, i)
        }
    }, t.getClasses = function(t, e, n) {
        var i, o = t;
        if (n && (o += " " + n), e)
            for (i in e)
                e.hasOwnProperty(i) && e[i] && (o += " " + t + "_" + i + "_" + e[i]);
        return o
    }, t.getAttrs = function(t) {
        if (!t)
            return "";
        var e, n = "";
        for (e in t)
            t.hasOwnProperty(e) && (n += " " + e + '="' + t[e] + '"');
        return n
    }, t.getJsParams = function(t) {
        return JSON.stringify(t || {}).replace(/"/g, "'")
    }, t.decline = function(t, e) {
        return e && e.length ? 0 === t && e[3] ? e[3] : e[1] || e[2] ? e[2] ? (t%=100, t >= 20 && (t%=10), 1 === t ? e[0] : t > 1 && 5 > t ? e[1] : e[2]) : t >= 0 && 1 >= t ? e[0] : e[1] : e[1] : e
    }
}(ya), function() {
    function t() {
        var t = n.height(), e = i.height(), a = $(window).height();
        o || (o = t, s = t - e), i.css(o + r > a ? {
            height: Math.max(a - r - s, 200),
            overflowY: "auto",
            marginRight: 7
        } : {
            height: "auto",
            overflowY: "visible",
            marginRight: 0
        })
    }
    function e(e) {
        var i;
        i = e.is(".b-disaster") ? $(".b-link_pseudo_yes", e) : $(".b-disaster .b-link_pseudo_yes", e), i.click(function() {
            n && n.bem("popup").show(), t(), $(window).trigger("disasterpopup.show")
        })
    }
    var n, i, o = 0, s = 0, r = 70;
    $(function() {
        n = $(".b-disaster-popup"), i = $(".b-disaster-popup__content"), n.find(".b-disaster-content").each(function() {
            var t = this.onclick() || {};
            t = t.content || "", $(this).replaceWith(t)
        }), $(window).resize(function() {
            t()
        })
    }), window.initDisasterPopup = e
}(), function() {
    var t=!0;
    $.browser.msie ? ($(document).bind("focusin", function() {
        t=!0
    }), $(document).bind("focusout", function() {
        t=!1
    })) : ($(window).focus(function() {
        t=!0
    }), $(window).blur(function() {
        t=!1
    })), $(document).ready(function() {
        if (window.teaserId) {
            var e = $("#" + window.teaserId);
            if (e.length) {
                e = e[0];
                var n = e.src, i = 0, o = 6e4, s = setTimeout(function r() {
                    600 > i ? t ? (e.src = n + "?rnd=" + ya.uid(), i++, setTimeout(r, o)) : $.browser.msie ? $(document).one("focusin", r) : $(window).one("focus", r) : clearTimeout(s)
                }, o)
            }
        }
    })
}(), x.amd.require("statlog", function(t) {
    BEM.DOM.decl("themes-promo", {
        onSetMod: {
            js: function() {
                for (var e = this, n = e.params, i = n.stat, o = n.promos, s = null, r = null, a = null, c=!1, u = e._hasStorage(), l = new Date, d = new Date, h = 0, f = o.length; f > h; h += 1)
                    if (s = o[h], e._canShow(s)) {
                        $(".b-line_bar").addClass("has_themes-promo"), r = {
                            text: s.text,
                            url: "/themes/" + (s["default"] ? s["default"] : "") + "#tab=" + (s.group || "all"),
                            mix: "b-link_themes_promo ",
                            stat: i ? e._getStatAttr(s.counter + ".preview", !0): null
                        }, a = {
                            text: "×",
                            mix: "themes-promo__close b-link_pseudo_yes ",
                            stat: i ? e._getStatAttr(s.counter + ".close"): null
                        }, i && t.logByDocument(e.domElem[0]), e.elem("wrapper-link").html(e._link(r)), e.elem("wrapper-close").html(e._link(a)), s.promo_from_mail && ((new Image).src = s.show_url), u || s.promo_from_mail || e.elem("wrapper-close").addClass("i-hidden"), c=!0, e.setMod("state", "shown");
                        break
                    }
                c && (u && e.elem("wrapper-link").find(".b-link_themes_promo").click(function() {
                    e._banPromo(s.counter, Number(l.setMonth(l.getMonth() + 3)))
                }), e.bindTo("close", "leftclick", function() {
                    s.promo_from_mail ? (new Image).src = s.close_url : u && (e._banPromo(s.counter, Number(l.setMonth(l.getMonth() + 3))), e._banPromo("hot", Number(d.setHours(d.getHours() + 24)))), e.setMod("state", "hidden"), $(".b-line_bar").removeClass("has_themes-promo")
                }))
            }
        },
        _canShow: function(t) {
            return this._checkConditions(t) && this._checkLS("hot") && this._checkLS(t.counter)
        },
        _checkConditions: function(t) {
            return t["default"] !== this.params.skin
        },
        _checkLS: function(t) {
            if (!this._hasStorage())
                return !0;
            var e, n = Number(new Date), i = this._getUnlockDate(t);
            return i ? (e = i - n, e > 0?!1 : (this._cleanGarbage(t), !0)) : !0
        },
        _banPromo: function(t, e) {
            home.localStorage.setItem("tp:" + t, e)
        },
        _cleanGarbage: function(t) {
            home.localStorage.removeItem("tp:" + t)
        },
        _getUnlockDate: function(t) {
            var e = home.localStorage.getItem("tp:" + t);
            return null !== e ? Number(e) : null
        },
        _getStatAttr: function(t) {
            return 'data-statlog="head.skinpromo.' + t + '"'
        },
        _link: function(t) {
            return (t.url ? '<a href="' + t.url + '"' : "<span") + ' class="b-link i-bem' + (t.mix ? " " + t.mix : "") + '"' + (t.stat ? " " + t.stat : "") + 'onclick="return{' + (t.js || "'b-link':{}") + '}">' + (t.text || "") + (t.url ? "</a>" : "</span>")
        },
        _hasStorage: function() {
            return !home.disabledLS
        }
    })
}), function() {
    BEM.DOM.decl("i-menu", {
        onElemSetMod: {
            item: {
                state: {
                    current: function(t, e, n, i) {
                        if ("disabled" == i)
                            return !1;
                        var o = this.elem("item", "state", "current");
                        this.delMod(o, "state").trigger("current", {
                            prev: o,
                            current: t
                        })
                    }
                }
            }
        },
        onItemSelectorClick: function(t) {
            var e = this._getItemByEvent(t);
            this.setMod(e, "state", "current")
        },
        _getItemByEvent: function(t) {
            return t.data.domElem.closest(this.buildSelector("item"))
        }
    }, {
        live: function() {
            this.liveBindTo("item-selector", "leftclick", function(t) {
                this.onItemSelectorClick(t)
            })
        }
    })
}(), function() {
    BEM.DOM.decl({
        name: "b-menu-vert",
        baseBlock: "i-menu"
    }, {}, {
        live: function() {
            this.__base()
        }
    })
}(), BEM.DOM.decl({
    block: "dropdown-menu",
    baseBlock: "dropdown"
}), function() {
    BEM.DOM.decl({
        name: "b-menu-horiz",
        baseBlock: "i-menu"
    })
}(), BEM.DOM.decl({
    block: "popup",
    modName: "domik",
    modVal: "trigger"
}, {
    _dispatch: function(t) {
        $(window).trigger("popup" + t + ".lego")
    },
    show: function() {
        this._dispatch("open"), this.__base.apply(this, arguments)
    },
    hide: function() {
        this._dispatch("close"), this.__base.apply(this, arguments)
    }
}), BEM.DOM.decl("user", {
    onSetMod: {
        js: {
            inited: function() {
                this.__base.apply(this, arguments), this._isRetina = window.devicePixelRatio, this._isRetina > 1 && this.updateIcon()
            }
        }
    },
    _getIconPath: function() {
        var t = this.params.avatarHost, e = this.params.uid || BEM.blocks["i-global"].param("uid"), n = this._getIconSize(), i = (this._isRetina ? "/islands-retina-" : "/islands-") + n, o = [t, "/get-yapic/", e, i, "?rnd=", + new Date].join("");
        return o
    },
    _setIconBackground: function(t) {
        var e = this.elem("icon");
        e && e.css("background-image", "url(" + t + ")")
    },
    updateIcon: function() {
        var t = this._getIconPath();
        this._setIconBackground(t)
    },
    getDefaultParams: function() {
        return {
            avatarHost: "//avatars.yandex.net"
        }
    },
    _getIconSize: function() {
        return "middle"
    }
}), BEM.DOM.decl("user", {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments);
            var t = this, e = BEM.blocks["i-global"];
            0 === e.param("login").lastIndexOf("uid-", 0) && BEM.blocks["i-user-services"].get(function(e) {
                var n = ya.htmlFilter(e.displayName && e.displayName.name || e.login || " "), i = '<span class="user__name"><span class="user__first-letter">' + n[0] + "</span>" + n.slice(1) + "</span>";
                BEM.DOM.replace(t.elem("name"), i)
            })
        }
    }
}), BEM.DOM.decl("user", {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments);
            var t = this.elem("icon");
            if (t.length && 0 === $("html.i-ua_inlinesvg_yes").length) {
                var e = t.css("background-image");
                e = e.replace(/(?:[^\/]+)\.svg(.*)?\)$/, "SFzCr2GZxGw2r1fKJO9soo0otUI.png$1)"), t.css("background-image", e)
            }
        }
    }
}), BEM.DOM.decl("user", {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments);
            var t = this.findBlockOn("enter", "button");
            if (null !== t) {
                var e = BEM.create("domik_type_popup", {
                    mods: this.params.domikMods,
                    hidden: this.findBlockInside({
                        block: "domik",
                        modName: "type",
                        modVal: "hidden"
                    })
                });
                t.bindTo("leftclick", function() {
                    return e.open(), !1
                })
            }
        }
    }
}), BEM.DOM.decl("user", {
    onSetMod: {
        js: function() {
            var t = this, e = BEM.blocks["i-global"], n = function(t, e, n) {
                var i = "…";
                return t.length > e + n ? t.substring(0, e - 1) + i : t
            };
            0 === e.param("login").lastIndexOf("uid-", 0) && BEM.blocks["i-user-services"].get(function(e) {
                var i = n(ya.htmlFilter(e.displayName && e.displayName.name || e.login || " "), 20, 3), o = '<span class="user__name"><span class="user__first-letter">' + i[0] + "</span>" + i.slice(1) + "</span>";
                BEM.DOM.replace(t.elem("name"), o)
            });
            var i = this.findBlockOn("enter", "button");
            null !== i && i.bindTo("leftclick", function() {
                try {
                    var t = $(".popup_behavior_snap");
                    return t.length ? (t = t.bem("popup"), t.isShown()?!0 : $(".b-topbar_expanded_yes").length ? ($(".popup_visibility_visible").bem("popup").hide(), $(window).trigger("popupclose.lego"), !1) : !0) : !0
                } catch (e) {
                    return !1
                }
            })
        }
    }
}), BEM.decl("domik_type_popup", {
    destruct: function() {
        var t = this;
        t._form && t._form.destruct(), t._popup && t._popup.destruct(), t.__base.apply(t, arguments)
    },
    create: function() {
        var t = this;
        if (t._created)
            return t;
        var e, n, i = t.__self, o = i._buildTypePopup(t._uniqId, t.params.mods), s = $(BEMHTML.apply(o)), r = t.params.hidden;
        if (BEM.DOM.append($("body"), s), n = t._popup = s.bem("popup"), e = t._form = t._popup.findBlockInside("auth"), r) {
            var a = r.domElem;
            e.getUsernameBlock().val(a.find("input[name=login]").change().val()), e.getPasswordBlock().val(a.find("input[name=passwd]").change().val()), r.destruct()
        }
        return t._created=!0, t
    },
    open: function() {
        var t = this;
        return BEM.blocks["i-global"].param("login") ? t : (t._created || t.create(), t.close(), t._popup.show(), t.afterCurrentEvent(function() {
            t._form.getUsernameBlock().setMod("focused", "yes"), t._form.redraw()
        }), t)
    },
    close: function() {
        return this._popup.hide(), this
    },
    isOpened: function() {
        return this._popup && this._popup.isShowed() ||!1
    },
    getFormBlock: function() {
        return this._form
    }
}, {
    _buildTypePopup: function(t, e) {
        return {
            block: "popup",
            mix: [{
                block: "domik",
                elem: "popup"
            }
            ],
            mods: {
                type: "modal",
                position: "fixed",
                "body-scroll": "no",
                theme: "ffffff"
            },
            underMods: {
                type: "paranja"
            },
            content: {
                elem: "content",
                content: {
                    block: "domik",
                    js: {
                        _uniqId: t
                    },
                    mods: $.extend({
                        type: "popup"
                    }, e),
                    content: {
                        block: "auth",
                        mods: {
                            content: "auto"
                        }
                    }
                }
            }
        }
    }
}), BEM.DOM.decl({
    block: "popup",
    modName: "body-scroll",
    modVal: "no"
}, {
    onSetMod: {
        visibility: {
            visible: function() {
                this.__base(), this._body.css({
                    overflow: "hidden"
                })
            },
            "": function() {
                this.__base(), this._body.css({
                    overflow: ""
                })
            }
        }
    }
}), BEM.decl("i-common__string", {}, {
    cleverSubstring: function() {
        var t = "…";
        return function(e, n, i) {
            return e.length > n + i ? e.substring(0, n - 1) + t : e
        }
    }(),
    escapeHTML: function() {
        var t = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        };
        return function(e) {
            return String(e).replace(/&(?!\w+;)|[<>"']/g, function(e) {
                return t[e] || e
            })
        }
    }(),
    escapeRegExp: function(t) {
        return t.replace(/([.?*+^$[\]\\(){}|-])/g, "\\$1")
    },
    highlight: function(t, e) {
        if (!$.isArray(e))
            return t;
        var n, i = [], o = 0, s = e.sort(function(t, e) {
            return t[0] - e[0]
        }), r = [], a = "?", c=~t.indexOf(a), u = function() {
            for (var e, n = /\&#\d+;/gi; e = n.exec(t);)
                r.push(e[0]), t = t.replace(e[0], a)
        }, l = function(t) {
            if (c)
                return t;
            for (; ~t.indexOf(a) && r.length;)
                t = t.replace(a, r[0]), r.shift();
            return t
        };
        return !c && u(), s.forEach(function(e) {
            n = o > e[0] ? o : e[0], i.push({
                tag: "span",
                elem: "span",
                content: l(t.slice(o, n))
            }), i.push({
                tag: "em",
                elem: "em",
                content: l(t.slice(n, o = e[1]))
            })
        }, this), i.push({
            tag: "span",
            elem: "span",
            content: l(t.slice(o))
        }), i
    }
}), function() {
    var t, e, n = function() {
        for (var e, n = 0; e = t[n++];)
            e.val(e.elem("control").val())
    }, i = function(t) {
        try {
            return t.activeElement
        } catch (e) {}
    };
    BEM.DOM.decl("input", {
        onSetMod: {
            js: function() {
                var o = this, s = o.elem("control"), r = i(o.__self.doc[0]), a = o.params.autoFocus&&!(r && $(r).is("input, textarea"));
                o._val = s.val(), (r === s[0] || a) && (o.setMod("focused", "yes")._focused=!0), e || (t = [], e = o.channel("sys").on({
                    tick: n,
                    idle: function() {
                        e.un("tick", n)
                    },
                    wakeup: function() {
                        e.on("tick", n)
                    }
                })), o._instanceIndex = t.push(o.bindTo(s, {
                    focus: o._onFocus,
                    blur: o._onBlur
                })) - 1, o.on("change", o._updateClear)._updateClear(), o.bindTo("box", "click", function() {
                    o.setMod("focused", "yes")
                })
            },
            disabled: function(t, e) {
                this.elem("control").attr("disabled", "yes" === e)
            },
            focused: function(t, e) {
                if (this.hasMod("disabled", "yes"))
                    return !1;
                var n = "yes" == e;
                n ? this._focused || this._focus() : this._focused && this._blur(), this.afterCurrentEvent(function() {
                    this.trigger(n ? "focus" : "blur")
                })
            }
        },
        onElemSetMod: {
            message: {
                visibility: function(t, e, n) {
                    var i = this, o = i.getMod(t, "type");
                    if (o) {
                        var s=!0;
                        n || i.elem("message", "type", o).each(function() {
                            this != t[0] && i.hasMod($(this), "visibility", "visible") && (s=!1)
                        }), s && i.toggleMod("message-" + o, "yes", "", "visible" === n)
                    }
                }
            }
        },
        _onClearClick: function() {
            this.trigger("clear"), this.removeInsets && this.removeInsets(), this.val("", {
                source: "clear"
            }).setMod("focused", "yes")
        },
        _updateClear: function() {
            return this.toggleMod(this.elem("clear"), "visibility", "visible", "", !!this._val)
        },
        isDisabled: function() {
            return this.hasMod("disabled", "yes")
        },
        val: function(t, e) {
            if ("undefined" == typeof t)
                return this._val;
            if (this._val != t) {
                var n = this.elem("control");
                n.val() != t && n.val(t), this._val = t, this.trigger("change", e)
            }
            return this
        },
        getSelectionEnd: function() {
            var t = this.elem("control")[0], e = 0;
            if ("number" == typeof t.selectionEnd)
                e = t.selectionEnd;
            else {
                var n = document.selection.createRange();
                if (n && n.parentElement() == t) {
                    var i = t.value.length, o = t.createTextRange();
                    o.moveToBookmark(n.getBookmark());
                    var s = t.createTextRange();
                    s.collapse(!1), e = o.compareEndPoints("EndToEnd", s)>-1 ? i : - o.moveEnd("character", - i)
                }
            }
            return e
        },
        name: function() {
            return this.elem("control").attr("name")
        },
        _onFocus: function() {
            return this._focused=!0, this.setMod("focused", "yes")
        },
        _onBlur: function() {
            return this._focused=!1, this.delMod("focused")
        },
        _focus: function() {
            var t = this.elem("control")[0];
            if (t.createTextRange&&!t.selectionStart) {
                var e = t.createTextRange();
                e.move("character", t.value.length), e.select()
            } else 
                t.focus()
        },
        _blur: function() {
            this.elem("control").blur()
        },
        destruct: function() {
            this.__base.apply(this, arguments), this.params.shortcut && this.unbindFromDoc("keydown"), t.splice(this._instanceIndex, 1);
            for (var e, n = this._instanceIndex; e = t[n++];)
                --e._instanceIndex
        }
    }, {
        live: function() {
            return this.liveBindTo("clear", "leftclick", function(t) {
                this._onClearClick(), t.stopPropagation()
            }), !1
        }
    })
}(), BEM.DOM.decl("input", {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments), this.params.shortcut && this.bindToDoc("keydown", function(t) {
                t.ctrlKey && 38 === t.keyCode&&!$(t.target).is("input, textarea") && this.setMod("focused", "yes")
            })
        }
    }
}), function() {
    var t, e = 0, n=!1, i = 0, o = BEM.channel("sys"), s = 50;
    BEM.decl("i-system", {}, {
        start: function() {
            $(document).bind("mousemove keydown", function() {
                i = 0, n && (n=!1, o.trigger("wakeup"))
            }), this._tick()
        },
        _tick: function() {
            var r = this;
            o.trigger("tick", {
                counter: e++
            }), !n && (i += s) > 3e3 && (n=!0, o.trigger("idle")), t = setTimeout(function() {
                r._tick()
            }, s)
        }
    }).start()
}(), BEM.DOM.decl("checkbox", {
    onSetMod: {
        js: function() {
            this.setMod("checked", this.elem("control").attr("checked") ? "yes" : ""), this._isControlFocused() && this.setMod("focused", "yes")
        },
        focused: {
            yes: function() {
                return this.isDisabled()?!1 : (this._isControlFocused() || this.elem("control").focus(), this.setMod(this.elem("box"), "focused", "yes"), void this.afterCurrentEvent(function() {
                    this.trigger("focus")
                }))
            },
            "": function() {
                this._isControlFocused() && this.elem("control").blur(), this.delMod(this.elem("box"), "focused"), this.afterCurrentEvent(function() {
                    this.trigger("blur")
                })
            }
        },
        checked: function(t, e) {
            this.elem("control").attr("checked", "yes" == e), this.afterCurrentEvent(function() {
                this.trigger("change")
            }), this.toggleMod(this.elem("box"), "checked", "yes", "yes" == e)
        },
        disabled: function(t, e) {
            this.elem("control").attr("disabled", "yes" === e)
        }
    },
    isDisabled: function() {
        return this.hasMod("disabled", "yes")
    },
    isChecked: function() {
        return this.hasMod("checked", "yes")
    },
    toggle: function() {
        this.toggleMod("checked", "yes", "")
    },
    val: function(t) {
        var e = this.elem("control");
        return "undefined" == typeof t ? e.val() : (e.val(t), this)
    },
    _onClick: function(t) {
        this.isDisabled() || this.setMod("focused", "yes").toggle(), t.preventDefault()
    },
    _onChange: function(t) {
        t.target.checked ? this.setMod("checked", "yes") : this.delMod("checked")
    },
    _onFocusInFocusOut: function(t) {
        this.setMod("focused", "focusin" === t.type ? "yes" : "")
    },
    _onMouseOverMouseOut: function(t) {
        this.isDisabled() || this.setMod("hovered", "mouseover" === t.type ? "yes" : "")
    },
    _isControlFocused: function() {
        try {
            return this.containsDomElem($(this.__self.doc[0].activeElement))
        } catch (t) {
            return !1
        }
    }
}, {
    live: function() {
        return this.liveBindTo("leftclick", function(t) {
            this._onClick(t)
        }).liveBindTo("control", "leftclick", function(t) {
            t.stopPropagation()
        }).liveBindTo("control", "change", function(t) {
            this._onChange(t)
        }).liveBindTo("control", "focusin focusout", function(t) {
            this._onFocusInFocusOut(t)
        }).liveBindTo("mouseover mouseout", function(t) {
            this._onMouseOverMouseOut(t)
        }), !1
    }
}), BEM.DOM.decl("auth", {
    onSetMod: {
        js: function() {
            var t = this;
            t._button = t.findBlockInside("button", "button"), t._username = t.findBlockInside("username", "input"), t._password = t.findBlockInside("password", "input"), t._password.on("change", t._onInputChange, t), t._username.on("change", t._onInputChange, t), t.redraw(), t._errorOn = null, t.bindTo("submit", t._onBeforeSubmit)
        }
    },
    destruct: function() {
        var t = this;
        t._error && t._error.destruct(), t.__base.apply(t, arguments)
    },
    _appendErrorElem: function() {
        var t = this, e = $(BEMHTML.apply(t.__self._buildError()));
        t.__self.append(t.domElem, e), t._error = t.findBlockOn(e, "popup"), t._errorMsg = t.findElem(e, "error"), t._error.on("hide", function() {
            t._errorOn = null
        })
    },
    _checkInput: function(t) {
        var e = this, n = e.__self, i = e._username.val(), o = e._password.val();
        if (t) {
            if (0 === i.length)
                return e.showError(e._username, BEM.I18N("auth", "fill-input")), !1;
            if (0 === o.length)
                return e.showError(e._password, BEM.I18N("auth", "fill-input")), !1
        }
        return /[\u0400-\u04FF\u0500-\u052F]/.test(i) ? (e.showError(e._username, BEM.I18N("auth", "wrong-keyboard-layout")), !1) : /[\u0400-\u04FF\u0500-\u052F]/.test(o) ? (e.showError(e._password, BEM.I18N("auth", "wrong-keyboard-layout")), !1) : n._isUsernameValid(i) ? n._isPasswordValid(o) ? o && o === i ? (e.showError(e._password, BEM.I18N("auth", "wrong-password")), !1) : (e.hideError(), !0) : (e.showError(e._password, BEM.I18N("auth", "wrong-characters")), !1) : (e.showError(e._username, BEM.I18N("auth", "wrong-characters")), !1)
    },
    _onInputChange: function() {
        this._checkInput(!1)
    },
    enableButton: function() {
        return this._button.delMod("disabled"), this
    },
    disableButton: function() {
        return this._button.setMod("disabled", "yes"), this
    },
    _onBeforeSubmit: function(t) {
        return this._checkInput(!0) ? (this._onSubmit(t), void this.trigger("submit", t)) : !1
    },
    _onSubmit: function() {
        this.disableButton(), this.elem("retpath").val(this.__self._getRetpath()), this._timestamp || (this._timestamp = $('<input type="hidden" name="' + this.__self._timestampName + '"/>').appendTo(this.domElem[0])), this._timestamp.val( + new Date);
        var t = this.domElem.attr("action");
        t && this.domElem.attr("action", t.replace(/^(https?:)?/, "https:"))
    },
    showError: function(t, e) {
        var n = this;
        return n._error || n._appendErrorElem(), n._curErrorMsg !== e && (n._errorMsg.text(e), n._curErrorMsg = e), n._errorOn !== t && (n._error.hide(), n._error.show(t), n._errorOn = t), n
    },
    hideError: function() {
        return this._error && this._error.hide(), this._errorOn = null, this
    },
    getUsernameBlock: function() {
        return this._username
    },
    getPasswordBlock: function() {
        return this._password
    },
    getHaunterBlock: function() {
        var t = this;
        return !t._haunter && (t._haunter = t.findBlockInside("haunter", "checkbox")), t._haunter
    },
    submit: function() {
        return this.domElem.submit(), this
    },
    redraw: function() {
        return this._onInputChange(), this
    }
}, {
    _retpathName: "retpath",
    _timestampName: "timestamp",
    _isUsernameValid: function(t) {
        return !( - 1 === t.indexOf("@") && /[^-A-Za-z0-9\.]/.test(t))
    },
    _isPasswordValid: function(t) {
        return !/[^\-a-zA-Z0-9`!@#$%^&*()_=+\[\]{};:"\|\\,.<>\/?]/.test(t)
    },
    _getRetpath: function() {
        return BEM.blocks["i-global"].param("retpath")
    },
    _buildError: function() {
        return {
            block: "popup",
            mods: {
                autoclosable: "no"
            },
            js: {
                directions: ["right", "left"]
            },
            content: [{
                elem: "tail"
            }, {
                elem: "content",
                mix: [{
                    block: "auth",
                    elem: "error"
                }
                ]
            }
            ]
        }
    }
}), BEM.DOM.decl("auth", {
    onSetMod: {
        js: function() {
            var t = this;
            t.__base.apply(t, arguments);
            var e = document.createElement("b");
            e.innerHTML = "<!--[if IE 9]><i></i><![endif]-->", 1 === e.getElementsByTagName("i").length && t._username.elem("control").add(t._password.elem("control")).keydown(function(e) {
                return 13 === e.which ? (t.submit(), !1) : void 0
            })
        }
    }
}), BEM.DOM.decl("auth", {
    _checkInput: function(t) {
        var e = this, n = e.__self, i = e._username.val(), o = e._password.val();
        if (t) {
            if (0 === i.length)
                return e.showError(e._username, BEM.I18N("auth", "fill-input")), !1;
            if (0 === o.length)
                return e.showError(e._password, BEM.I18N("auth", "fill-input")), !1
        }
        return n._isUsernameValid(i) ? (e.hideError(), !0) : (e.showError(e._username, BEM.I18N("auth", "wrong-characters")), !1)
    }
}, {
    _isUsernameValid: function(t) {
        return !/\s/.test(t)
    },
    _buildError: function() {
        var t = this.__base.apply(this, arguments);
        return t.js.directions = ["left"], t
    }
}), BEM.DOM.decl({
    block: "auth",
    modName: "login",
    modVal: "known"
}, {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments), this.params.login || this.delMod("login"), this.findBlockInside("username", "input").val(this.params.login), this.bindTo("notme", "click", function() {
                this.findBlockInside("username", "input").val(""), this.elem("stale-login").remove(), this.channel("auth").trigger("notme"), this.delMod("login")
            })
        }
    }
}), BEM.DOM.decl("auth", {
    onSetMod: {
        js: function() {
            var t = this, e = BEM.blocks["i-global"];
            t.__base.apply(t, arguments);
            var n = e.param("social-providers");
            if (n)
                t._createSocial(n);
            else {
                var i = e.param("social-host");
                if (!i)
                    return;
                var o = t.elem("social");
                o.css("height", 28), $.getJSON(t.__self._getProvidersURL(i), function(e) {
                    o.css("height", ""), t._createSocial(e.providers)
                })
            }
        }
    },
    destruct: function() {
        var t = this;
        t._socialDropdown && t._socialDropdown.destruct(), t.__base.apply(t, arguments)
    },
    _createSocial: function(t) {
        var e, n = this, i = n.__self;
        return t.length ? (e = i._buildSocial(t, n._uniqId), i.update(n.elem("social"), BEMHTML.apply(e)), void(n._socialDropdown = n.findBlockInside("social", "dropdown"))) : void n.elem("social").remove()
    },
    _onSocialClick: function(t) {
        t.preventDefault();
        var e = BEM.blocks["i-global"], n = this.elemParams(t.data.domElem).provider;
        this._socialDropdown && this._socialDropdown.hide(), BEM.blocks["social-broker"].start(function() {
            e.param("retpath") && top.location.href != e.param("retpath") ? top.location.href = e.param("retpath") : top.location.reload()
        }, function() {}, {
            provider: n
        })
    }
}, {
    live: function() {
        return this.__base.apply(this, arguments), this.liveBindTo("social-link", "click", function(t) {
            this._onSocialClick(t)
        }), !1
    },
    _getProvidersURL: function(t) {
        return t + "/providers2.jsonp?callback=?"
    },
    _buildSocial: function(t, e) {
        var n, i, o = [];
        return t = t.filter(function(t) {
            return t.enabled
        }), n = t.filter(function(t) {
            return t.primary
        }), n = n.map(function(t) {
            return {
                block: "auth",
                elem: "social-link",
                js: {
                    provider: t.code
                },
                attrs: {
                    tabIndex: 103
                },
                mix: [{
                    elem: "social-icon",
                    elemMods: {
                        code: t.code
                    }
                }
                ]
            }
        }), [].push.apply(o, n), i = t.map(function(t) {
            return {
                block: "auth",
                elem: "social-link",
                js: {
                    provider: t.code
                },
                attrs: {
                    tabIndex: 103
                },
                content: [{
                    elem: "social-icon",
                    elemMods: {
                        code: t.code
                    }
                }, t.display_name]
            }
        }), n.length < t.length && o.push({
            block: "dropdown",
            js: !0,
            content: [{
                block: "button",
                mods: {
                    pseudo: "yes",
                    theme: "pseudo",
                    size: "s"
                },
                mix: [{
                    block: "dropdown",
                    elem: "switcher"
                }
                ],
                tabindex: 103,
                content: "· · ·"
            }, {
                elem: "popup",
                content: {
                    block: "auth",
                    elem: "social-popup",
                    mix: [{
                        block: "auth",
                        js: {
                            uniqId: e
                        }
                    }
                    ],
                    content: i
                }
            }
            ]
        }), o
    }
}), BEM.DOM.decl("auth", {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments), Lego.params.statRoot && this._socialDropdown && this._socialDropdown.bindTo("switcher", "click", function() {
                cp(Lego.params.statRoot + ".domik.loginsocial.more")
            })
        }
    },
    _onSocialClick: function(t) {
        t.preventDefault();
        var e = BEM.blocks["i-global"], n = this.elemParams(t.data.domElem).provider;
        this._socialDropdown && this._socialDropdown.hide(), Lego.params.statRoot && cp(Lego.params.statRoot + ".mail.logout.social." + n), BEM.blocks["social-broker"].start(function() {
            var t, n = e.param("retpath");
            return n && top.location.href !== n ? (t = setTimeout(function() {
                top.location.href = n
            }, 5e3), void $.ajax({
                url : Lego.params["pass-host"] + "/services", data : {
                    locale : BEM.blocks["i-global"].param("lang"), yu : $.cookie("yandexuid")
                }, dataType : "jsonp", success : function(e) {
                    clearTimeout(t);
                    for (var i = 0, o = e.length; o > i; i++) {
                        var s = e[i].url && (e[i].url.match(/^(?:https?:)?\/\/([^\/]+)/) || [])[1] || "";
                        if ( - 1 !== n.indexOf(s))
                            return void(top.location.href = n)
                    }
                    top.location.reload()
                }
            })): void top.location.reload()
        }, function() {}, {
            provider: n
        })
    }
}, {
    _buildSocial: function() {
        var t = this.__base.apply(this, arguments), e = t[t.length - 1];
        return "dropdown" === e.block && (e.content[0].content = "", e.content[1].js = {
            directions: {
                to: "bottom",
                axis: "right",
                offset: {
                    left: 10
                }
            }
        }), t
    }
}), BEM.decl("social-broker", {
    _windowWidth: 500,
    _windowHeight: 280,
    onSetMod: {
        js: function() {
            this._window = null, this._timer = 0
        }
    },
    start: function(t, e, n) {
        var i, o, s, r, a = this;
        if (this._cleanup(), this._onSuccess = t, this._onFailure = e, n = $.extend({}, a._getDefaultParams(), n), i = n.startUrl + "?", r = n.popupName, delete n.startUrl, delete n.popupName, o = n.display, s = n.retpath, o && "popup" != o)
            n.retpath = n.retnopopup || n.retpath || window.location.href, delete n.retnopopup;
        else {
            var c = "ddom=" + (n.ddom || (location.hostname == document.domain ? "" : document.domain));
            n.retpath += (s.match("#") ? "&" : "#") + c
        }
        return i += Object.keys(n).reduce(function(t, e) {
            return t.push(encodeURIComponent(e) + "=" + encodeURIComponent(n[e])), t
        }, []).join("&"), o && "popup" != o ? void(window.location.href = i) : (this._window = window.open(i, r, a._getPopupParams()), void(this._window && (this._window.focus(), clearTimeout(this._timer), this._timer = setInterval(function() {
            a.isClosed() && (clearInterval(a._timer), a._onFailure({
                status: "error"
            }))
        }, 500))))
    },
    isClosed: function() {
        if (!this._window)
            return !0;
        try {
            return this._window.closed
        } catch (t) {
            return !0
        }
    },
    _getDefaultParams: function() {
        var t = BEM.blocks["i-global"], e = t.param("id");
        return {
            startUrl: t.param("social-startUrl") || t.param("social-host") + "/broker/start",
            retpath: t.param("social-retpath") || location.protocol + "//" + location.host + t.param("lego-path") + "/blocks-common/i-social/closer/i-social__closer.html",
            retnopopup: t.param("social-retnopopup") ||!1,
            consumer: e || "morda",
            popupName: "social_" + (e || "morda"),
            action_if_anonymous: "authorize",
            result_location: "fragment"
        }
    },
    _getPositionedParams: function(t, e, n, i) {
        return "scrollbars=yes, resizable=1, menubar=0, toolbar=0, status=0, location=0, directories=0, left=" + e + ", top=" + t + ", width=" + n + ", height=" + i
    },
    _getPopupParams: function() {
        var t, e, n = this._windowWidth, i = this._windowHeight;
        return window.opera ? (t = (screen.availWidth - n) / 2, e = (screen.availHeight - i) / 2) : (t = (window.innerWidth - n) / 2, e = (window.innerHeight - i) / 2), t = Math.round(t), e = Math.round(e), this._getPositionedParams(e, t, n, i)
    },
    _cleanup: function() {
        clearInterval(this._timer), this.isClosed() || this._window.close(), this._window = null
    },
    onSuccess: function(t) {
        this._cleanup(), this._onSuccess && this._onSuccess(t)
    },
    onFailure: function(t) {
        this._cleanup(), this._onFailure && this._onFailure(t)
    }
}, {
    start: function() {
        this._current && this._current.destruct(), this._current = BEM.create("social-broker"), Lego.block["i-social"] = {
            broker: this._current
        }, this._current.start.apply(this._current, arguments)
    },
    onSuccess: function() {
        this._current.onSuccess.apply(this._current, arguments)
    },
    onFailure: function() {
        this._current.onFailure.apply(this._current, arguments)
    }
}), BEM.DOM.decl("input", {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments), (this._hasHint=!!this.elem("hint")[0]) && this.on("change", this._updateHint)._updateHint()
        },
        focused: function() {
            this.__base.apply(this, arguments), this._hasHint && this._updateHint()
        }
    },
    _updateHint: function() {
        this.toggleMod(this.elem("hint"), "visibility", "visible", !this.val())
    }
}), BEM.DOM.decl({
    name: "button",
    modName: "pseudo",
    modVal: "yes"
}, {
    _onClick: function(t) {
        this.__base.apply(this, arguments), this._href && t.preventDefault()
    }
}), BEM.DOM.decl({
    block: "user",
    modName: "menu",
    modVal: "yes"
}, {
    _onClick: function(t) {
        t.preventDefault(), this.__base.apply(this, arguments), this._dropdown || (this._dropdown = this.findBlockOutside("dropdown-menu")), this._dropdown && this._dropdown.toggle(this.elem(this.elem("icon").length ? "icon" : "name"))
    }
}, {
    live: function() {
        return this.__base.apply(this, arguments), this.liveBindTo("leftclick", function(t) {
            this._onClick(t)
        }), !1
    }
}), BEM.DOM.decl("user-menu-update", {
    onSetMod: {
        js: {
            inited: function() {
                var t = this;
                t.updateInformer(t.onUpdateSuccess, t.onUpdateError)
            }
        }
    },
    updateInformer: function(t, e) {
        var n = this;
        n._getExporter().get({
            key: $.cookie("yandexuid")
        }, function(n) {
            n.unread ? t.apply(this, arguments) : e && e.call(this)
        }, e)
    },
    onUpdateSuccess: function(t) {
        var e = parseInt(t.unread, 10) || 0, n = this.params.unreadMax;
        e && n && e > n && (e = n + "+");
        var i = e ? e + " ": "";
        BEM.DOM.update(this.domElem, i + BEM.I18N("user", "letter", {
            count: e
        }))
    },
    onUpdateError: function() {},
    _getExporter: function() {
        return this._exporter || (this._exporter = BEM.create("i-request_type_ajax", {
            url: this.__self.getExportApiUrl(),
            cache: !1,
            callbackCtx: this
        }))
    },
    getDefaultParams: function() {
        return {
            unreadMax: 99
        }
    }
}, {
    getExportApiUrl: function() {
        return BEM.blocks["i-global"].param("export-host") + "/for/unread.xml?callback=?"
    }
}), function() {
    var t = {};
    BEM.decl("i-request", {
        onSetMod: {
            js: function() {
                this._preventCache=!1
            }
        },
        get: function(t, e, n, i) {
            $.isFunction(n) || (i = n, n = this.params.onError), this._get(t, e, n, $.extend({}, this.params, i))
        },
        _get: function(e, n, i, o) {
            var s = this._buildCacheKey(e, o), r = t[o.cacheGroup];
            o.cache && r && s in r.data ? this.afterCurrentEvent(function() {
                n.call(this.params.callbackCtx, r.data[s])
            }, this) : this._do(e, n, i, o)
        },
        _do: function() {},
        _onSuccess: function(t, e, n, i) {
            i.cache&&!this._preventCache && this.putToCache(i, t, n), this._preventCache=!1
        },
        _buildCacheKey: function(t) {
            return "string" == typeof t ? t : $.param(t)
        },
        putToCache: function(e, n, i) {
            var o = t[e.cacheGroup] || (t[e.cacheGroup] = {
                keys: [],
                data: {}
            });
            o.keys.length >= e.cacheSize && delete o.data[o.keys.shift()];
            var s = this._buildCacheKey(n, e);
            o.data[s] = i, o.keys.push(s)
        },
        dropCache: function() {
            delete t[this.params.cacheGroup]
        },
        getDefaultParams: function() {
            return {
                cache: !1,
                cacheGroup: "default",
                cacheSize: 100,
                callbackCtx: this
            }
        }
    }, {
        _cache: t
    })
}(), BEM.decl({
    block: "i-request_type_ajax",
    baseBlock: "i-request"
}, {
    onSetMod: {
        js: function() {
            this.__base(), this._requestNumber = this._number = this._preventNumber = this._retryCount = 0
        }
    },
    _get: function(t, e, n, i) {
        this._number++, this._requestNumber++, this._retryCount = i.retryCount, this.__base.apply(this, arguments)
    },
    _do: function(t, e, n, i) {
        var o = this;
        if (o._number > o._preventNumber) {
            var s = arguments, r = {
                data: i.data ? $.extend({}, i.data, t): t
            }, a = o._wrapCallback(function(n, s, r) {
                o._onSuccess(o._buildCacheKey(t, i), t, n[0], i), o._allowCallback(s, r) && e.apply(i.callbackCtx, n)
            }), c = o._wrapCallback(function(t, e, r) {
                o._allowCallback(e, r) && (o._retryCount-->0 ? setTimeout(function() {
                    o._do.apply(o, s)
                }, i.retryInterval) : n && n.apply(i.callbackCtx, t))
            });
            $.each(["url", "dataType", "timeout", "type", "jsonp", "jsonpCallback"].concat(i.paramsToSettings || []), function(t, e) {
                r[e] = i[e]
            }), $.ajax(r).done(a).fail(c)
        }
    },
    _wrapCallback: function(t) {
        var e = this._requestNumber, n = this._number;
        return function(i) {
            null !== i && t(arguments, e, n)
        }
    },
    _allowCallback: function(t, e) {
        return e > this._preventNumber && this._requestNumber == t
    },
    _buildCacheKey: function(t, e) {
        return "string" == typeof t ? t : this.__base(t) + e.url
    },
    abort: function() {
        this._preventNumber=++this._number
    },
    preventCallbacks: function() {
        this.abort()
    },
    getDefaultParams: function() {
        return $.extend(this.__base(), {
            cache: !0,
            type: "GET",
            dataType: "jsonp",
            timeout: 2e4,
            retryCount: 0,
            retryInterval: 2e3
        })
    }
}), BEM.DOM.decl({
    block: "user",
    modName: "login",
    modVal: "known"
}, {
    onSetMod: {
        js: function() {
            this.channel("auth").on("notme", function() {
                this.delMod("login")
            }, this)
        }
    }
}), BEM.DOM.decl("userlist", {
    onSetMod: {
        js: function() {
            this.bindTo("item-user", "click", this.switchTo)
        }
    },
    getEmbeddedAuth: function() {
        return this.auth ? this.auth : this.auth = BEM.create("i-embeddedauth", this.params.auth)
    },
    switchTo: function(t) {
        var e = $(t.currentTarget).data("uid");
        return this.getEmbeddedAuth().setData({
            uid: e
        }, this.changeThis(this.switchResponse, this)).submit(), !1
    },
    switchResponse: function(t) {
        switch (t.status) {
        case"ok":
        case"action-not-required":
            return void this.goTo();
        default:
            this.goTo(this.params.passport)
        }
    },
    goTo: function(t) {
        t ? window.location = t : window.location.reload()
    }
}), BEM.decl("i-embeddedauth", {
    addCallback: function(t) {
        var e = "cb" + Math.random();
        return this.callbacks || (this.callbacks = {}), this.callbacks[e] = t, e
    },
    addQueryParam: function(t, e) {
        var n = t.split("?"), i = n[1] ? n[1].split("&"): [];
        return i.push(e), n[1] = i.join("&"), n.join("?")
    },
    bindCallback: function() {
        var t = this;
        window.mordaAuth = {
            onResponse: function(e) {
                if (!e)
                    return void t.processResponse();
                for (var n = e.split("?")[1].split("&"), i = {}, o = 0, s = n.length; s > o; o++) {
                    var r = n[o].split("=");
                    i[r.shift()] = decodeURIComponent(r.join("="))
                }
                t.processResponse(i)
            }
        }
    },
    submit: function() {
        this.getForm().trigger("submit")
    },
    setData: function(t, e) {
        if (t = $.extend({}, this.params.params || {}, t), this.idkey && (t.idkey = this.idkey), e) {
            var n = this.addCallback(e);
            t.retpath = this.addQueryParam(t.retpath, "from=" + n)
        }
        return this.getForm().html(this.stringifyParams(t)), this
    },
    setDomain: function() {
        var t = document.domain.match(/([^.]+(?:\.com?)?\.)?[^.]+$/)[0];
        try {
            document.domain !== t && (document.domain = t)
        } catch (e) {}
    },
    stringifyParams: function(t) {
        var e = "";
        for (var n in t)
            t.hasOwnProperty(n) && (e += this.getInputHtml(n, t[n]));
        return e
    },
    getInputHtml: function(t, e) {
        return '<input name="' + t + '" value="' + e + '"/>'
    },
    getForm: function() {
        if (this.form)
            return this.form;
        this.setDomain(), this.bindCallback();
        var t = "embeddedauthframe" + parseInt(1e4 * Math.random());
        return this.form = $('<form style="display:none" target="' + t + '" method="' + (this.params.method || "post") + '" action="' + this.params.action + '"></form>'), this.frame = $('<iframe style="display:none" name="' + t + '"></iframe>'), $("body").append(this.form, this.frame), this.form
    },
    processResponse: function(t) {
        var e = function() {};
        if (!t)
            return void this.trigger("no-data");
        switch (t.from && this.callbacks[t.from] && (e = this.callbacks[t.from], this.callbacks[t.from]=!1), this.idkey = t.idkey, t.status) {
        case"login-empty":
        case"password-empty":
        case"account-not-found":
        case"password-invalid":
        case"action-not-required":
        case"ok":
        case"uid-empty":
        case"sessionid-invalid":
        case"session-no-uid":
        case"yu-empty":
        case"yu-invalid":
        case"captcha-invalid":
        case"captcha-required":
            return e(t), void this.trigger("data-received", t);
        case"internal-exception":
            return this.idkey = "", this._interr ? (e(t), void this.trigger("data-received", t)) : (this._interr=!0, t.from && (this.callbacks[t.from] = e), void this.submit());
        default:
            t.url ? window.location = t.url : (e(t), this.trigger("data-received", t))
        }
    }
}), function(t) {
    var e = t.event.special.outsideclick = {
        add: function(n) {
            t(document).on("click." + n.guid, e.handler.bind(this))
        },
        remove: function(e) {
            t(document).off("click." + e.guid)
        },
        handler: function(e) {
            t.contains(this, e.target) || (e.type = "outsideclick", t.event.handle.apply(this, arguments), e.type = "click")
        }
    }
}(jQuery), BEM.DOM.decl("paranja", {
    onSetMod: {
        js: function() {
            var t = this;
            $(t.params.rel).each(function() {
                $(this.elem || t.domElem).on(this.event, t[this.method].bind(t))
            }), this.bindTo("leftclick tap", function() {
                this.trigger("click")
            })
        },
        state: {
            open: function() {
                this.domElem.trigger("open")
            },
            close: function() {
                this.domElem.trigger("close")
            }
        }
    },
    open: function() {
        this.setMod("state", "open")
    },
    close: function() {
        this.setMod("state", "close")
    }
}, {
    live: !1
}), function() {
    var t = $(window), e = {};
    BEM.DOM.decl("header__action", {
        onSetMod: {
            js: function() {
                var t = this.params.group;
                (e[t] || (e[t] = [])).push(this)
            },
            pressed: {
                "": function() {
                    var e = this.params.uniqId;
                    BEM.blocks.input.un("focus"), BEM.blocks.button.un("focus"), BEM.blocks.paranja.un("click"), t.off("scroll." + e), t.off("closeUnder"), this.domElem.trigger("released")
                },
                yes: function() {
                    var n = this, i = this.params.uniqId;
                    $(e[this.params.group]).each(function() {
                        this != n && this.delMod("pressed")
                    }), BEM.blocks.input.on("focus", function() {
                        n.delMod("pressed")
                    }), BEM.blocks.button.on("focus", function(t) {
                        t.block.findBlockOutside("services-table") || n.delMod("pressed")
                    }), BEM.blocks.paranja.on("click", function() {
                        n.delMod("pressed")
                    }), n.params.releaseByDocumentScroll && t.one("scroll." + i, function() {
                        n.delMod("pressed")
                    }), t.one("closeUnder", function() {
                        n.delMod("pressed")
                    }), this.domElem.trigger("pressed")
                }
            }
        },
        getDefaultParams: function() {
            return {
                group: Math.random(),
                releaseByHeaderOutsideClick: !0,
                releaseByDocumentScroll: !0
            }
        }
    }, {
        live: function() {
            return this.liveBindTo("leftclick tap", function(t) {
                t.preventDefault(), this.toggleMod("pressed", "yes", "")
            }), !1
        }
    })
}(), x.amd.require("statlog", function(t) {
    BEM.DOM.decl({
        block: "header__action",
        modName: "type",
        modVal: "settings"
    }, {
        onSetMod: {
            js: function() {
                this._dropdown || (this._dropdown = this.findBlockOutside("dropdown-menu")), this._dropdown.on("hide", this.changeThis(function() {
                    this.delMod("pressed")
                }))
            },
            pressed: {
                yes: function() {
                    var e = this;
                    Lego.params.statRoot && "undefined" != typeof cp && t.log("head.settings"), e.__base.apply(e, arguments), setTimeout(function() {
                        e._dropdown && e._dropdown.show(e)
                    }, 50)
                },
                "": function() {
                    this._dropdown && this._dropdown.hide(this)
                }
            }
        }
    })
}), BEM.DOM.decl("b-form-button", {
    onSetMod: {
        js: function() {
            var t = this.isDisabled();
            (this._href = this.domElem.attr("href")) && t && this.domElem.removeAttr("href"), this.elem("input").attr("disabled", t)
        },
        focused: {
            yes: function() {
                var t = this;
                return t.isDisabled()?!1 : (t.bindTo("keydown", this._onKeyDown).elem("input").is(":focus") || t.elem("input").focus(), void(t._unloadInited || (t._unloadInited = $(window).bind("unload", function() {
                    t.delMod("focused")
                }))))
            },
            "": function() {
                this.unbindFrom("keydown").elem("input").blur()
            }
        },
        disabled: function(t, e) {
            var n = "yes" == e;
            this._href && (n ? this.domElem.removeAttr("href") : this.domElem.attr("href", this._href)), n && this.domElem.keyup(), this.afterCurrentEvent(function() {
                this.domElem && this.elem("input").attr("disabled", n)
            })
        },
        pressed: function(t, e) {
            this.isDisabled() || this.trigger("yes" == e ? "press" : "release")
        },
        hovered: {
            "": function() {
                this.delMod("pressed")
            }
        },
        "*": function(t) {
            return this.isDisabled() && "hovered pressed".indexOf(t)>-1?!1 : void 0
        }
    },
    isDisabled: function() {
        return this.hasMod("disabled", "yes")
    },
    url: function(t) {
        return "undefined" == typeof t ? this._href : (this._href = t, this.isDisabled() || this.domElem.attr("href", t), this)
    },
    _onKeyDown: function(t) {
        var e = t.keyCode;
        13 != e && 32 != e || this._keyDowned || (this._keyDowned=!0, this.setMod("pressed", "yes").bindTo("keyup", function() {
            this.delMod("pressed").unbindFrom("keyup"), delete this._keyDowned, 32 == e && this.domElem.attr("href") && (document.location = this.domElem.attr("href"))
        }))
    },
    _onClick: function(t) {
        this.isDisabled() ? t.preventDefault() : this.afterCurrentEvent(function() {
            this.trigger("click")
        })
    },
    destruct: function() {
        this.delMod("focused"), this.__base.apply(this, arguments)
    }
}, {
    live: function() {
        var t = {
            mouseover: {
                name: "hovered",
                val: "yes"
            },
            mouseout: {
                name: "hovered"
            },
            mousedown: {
                name: "pressed",
                val: "yes"
            },
            mouseup: {
                name: "pressed"
            },
            focusin: {
                name: "focused",
                val: "yes"
            },
            focusout: {
                name: "focused"
            }
        };
        this.liveBindTo("leftclick", function(t) {
            this._onClick(t)
        }).liveBindTo("mouseover mouseout mouseup focusin focusout", function(e) {
            var n = t[e.type];
            this.setMod(n.name, n.val || "")
        }).liveBindTo("mousedown", function(e) {
            var n = t[e.type];
            1 == e.which && this.setMod(n.name, n.val || "")
        })
    }
}), BEM.HTML.decl("b-form-button", {
    onBlock: function(t) {
        t.tag(t.param("url") ? "a" : "span").attrs({
            href: t.param("url"),
            target: t.param("target")
        }).mods({
            size: t.mod("size"),
            theme: t.mod("theme")
        }).content([{
            elem: "left",
            tag: "i"
        }, {
            elem: "content",
            tag: "span",
            content: {
                elem: "text",
                tag: "span",
                content: t.content()
            }
        }
        ], !0).afterContent(t.param("type") ? {
            elem: "input",
            attrs: {
                value: t.param("value") || "",
                type: t.param("type"),
                name: t.param("name"),
                disabled: t.mod("disabled") && "disabled"
            }
        } : {
            elem: "click"
        }).js(!0)
    },
    onElem: {
        input: function(t) {
            t.tag("input")
        },
        click: function(t) {
            t.tag("i")
        }
    }
}), BEM.DOM.decl("b-fade", {
    onSetMod: {
        js: function() {
            var t = this, e = this.params.target ? this.findBlockOutside(this.params.target): this;
            e.domElem.bind("mouseenter", function() {
                t.setMod("hovered", "yes"), clearTimeout(t.countdown)
            }).bind("mouseleave", function() {
                t.hide()
            }), this._bindEventsOnPopups()
        }
    },
    hide: function() {
        var t = this;
        t.countdown = setTimeout(function() {
            t._hide()
        }, 200)
    },
    _bindEventsOnPopups: function() {
        var t, e = this._getDropdownas(), n = e.length, i = this, o = function() {
            clearTimeout(i.countdown)
        }, s = function() {
            i.hide()
        };
        for (t = 0; n > t; t++)
            e[t].getPopup().domElem.unbind("mouseenter.b-fade", o).unbind("mouseleave.b-fade", s).bind("mouseenter.b-fade", o).bind("mouseleave.b-fade", s)
    },
    _getDropdownas: function() {
        return this.dropdownas && this.dropdownas.length || (this.dropdownas = this.findBlocksInside("b-dropdowna") || []), this.dropdownas
    },
    _hide: function() {
        var t, e = this._getDropdownas(), n = e.length;
        for (this.setMod("hovered", ""), t = 0; n > t; t++)
            e[t].getPopup().hide()
    }
}), BEM.DOM.decl("b-keyboard-loader", {
    onSetMod: {
        js: function() {
            Lego.block["b-keyboard-loader"].call(this.domElem, this.params)
        }
    }
}), function(t, e) {
    e.block["b-keyboard-loader"] = function(n) {
        function i(n) {
            var i = n ? "addClass": "toggleClass", o = c.find(".b-keyboard-popup"), s = e.block["b-keyboard"]._lastFocusedIn;
            o.length && (c.find(".b-keyboard-popup__gap")[i]("i-hidden"), c.find(".b-keyboard-popup")[i]("i-hidden").hasClass("i-hidden") ? (u.trigger("keyboardClosed"), t(document).trigger("popupsClose.lego"), s && s.focus()) : u.trigger("keyboardOpened.lego"), o.data("isHidden", !1))
        }
        function o() {
            d || (e.cp(0, 70873, BEM.blocks["i-global"].param("id") + "." + (n.autoLoad ? "auto" : "manual")), d=!0)
        }
        function s(t) {
            return r()[t]
        }
        function r() {
            var e = {};
            return t.each(window.name.split("&"), function() {
                if (this.toString()) {
                    var t = this.split("=");
                    e[t[0]] = t[1]
                }
            }), e
        }
        function a(e, n) {
            var i = r(), o = [];
            i[e] = n, t.each(i, function(t, e) {
                o.push(t + "=" + e)
            }), window.name = o.join("&")
        }
        var c = t(document.body), u = t(window), l = this;
        l.click(function(o) {
            o.preventDefault();
            var s = t(this), r = (this.className.match(/b-keyboard-loader_lang_(\w+)/) || ["", ""])[1];
            if (c.find(".b-keyboard")[0])
                i();
            else {
                var a = BEM.blocks["i-global"].param("lego-static-host"), l = a + ("/" === a.charAt(a.length - 1) ? "" : "/") + "blocks-desktop/", d = [l + "b-keyboard/_keyboard.js"], h = t.browser.msie && (!document.documentMode || document.documentMode < 9) ? ".ie": "", f = [l + "b-keyboard/_keyboard" + h + ".css"];
                s.append('<i class="b-keyboard-loader__progress"/>'), t.xLazyLoader({
                    name: "b-keyboard",
                    js: d,
                    css: f,
                    success: function() {
                        e.block["b-keyboard-popup"]({
                            lang: r,
                            "for": n["for"]
                        }), c.append("<div class=\"b-keyboard i-bem\" onclick=\"return {'b-keyboard': { lang:'" + r + "'}};\"></div>"), BEM.DOM.init(t(".b-keyboard")), s.find(".b-keyboard-loader__progress").remove(), u.trigger("keyboardOpened.lego", {
                            afterLoad: !0
                        })
                    }
                })
            }
        }), u.bind("keyboardOpen.lego", function() {
            l.click()
        }).bind("keyboardClose.lego", i), n["for"] && u.bind("keyboardOpened.lego", function(e, i) {
            !(i && i.afterLoad && n.autoLoad) && t(n["for"]).focus()
        }), u.bind("keyboardLangChanged.lego keyboardSetLang.lego", function(t, e) {
            var n = l.find(".b-keyboard-loader__flag");
            n[0] && ("tt" === e && (e = "ru"), n.attr("src", n.attr("src").replace(/[a-z]+\.png$/, e + ".png")))
        }), n.storeState = n.storeState!==!1, n.storeState && u.bind("keyboardOpened.lego", function() {
            a("keyboard_state", "open");
            var e = s("keyboard_lang");
            e && t(window).trigger("keyboardSetLang.lego", e)
        }).bind("keyboardClosed.lego", function() {
            a("keyboard_state", "close")
        }).bind("keyboardLangChanged.lego", function(t, e) {
            a("keyboard_lang", e)
        }), (n.autoLoad = n.autoLoad || n.storeState && "open" == s("keyboard_state")) && l.click();
        var d;
        u.bind("keyboardOpened.lego", function() {
            o(), t("input:focus").focus()
        })
    }
}(jQuery, window.Lego), function(t) {
    function e() {
        function e(t, e) {
            l[t](e, function(t) {
                "error" == t ? h.push(e) : d.push(e) && r.each(e), o()
            }, "lazy-loaded-" + (r.name ? r.name : (new Date).getTime()))
        }
        function i(t) {
            r.complete(t, d, h), r[t]("error" == t ? h : d), clearTimeout(a), clearTimeout(c)
        }
        function o() {
            d.length == f.length ? i("success") : d.length + h.length == f.length && i("error")
        }
        function s() {
            h.push(this.src), o()
        }
        var r, a, c, u, l = this, d = [], h = [], f = [];
        this.init = function(n) {
            if (n) {
                if (r = t.extend({}, t.xLazyLoader.defaults, n), u = {
                    js: r.js,
                    css: r.css,
                    img: r.img
                }, t.each(u, function(t, e) {
                    "string" == typeof e && (e = e.split(",")), f = f.concat(e)
                }), !f.length)
                    return void i("error");
                r.timeout && (a = setTimeout(function() {
                    var e = d.concat(h);
                    t.each(f, function(n, i) {
                        - 1 == t.inArray(i, e) && h.push(i)
                    }), i("error")
                }, r.timeout)), t.each(u, function(n, i) {
                    t.isArray(i) ? t.each(i, function(t, i) {
                        e(n, i)
                    }) : "string" == typeof i && e(n, i)
                })
            }
        }, this.js = function(e, i) {
            var o = t('script[src*="' + e + '"]');
            if (o.length)
                return void(o.attr("pending") ? o.bind("scriptload", i) : i());
            var r = document.createElement("script");
            r.setAttribute("type", "text/javascript"), r.setAttribute("charset", "utf-8"), r.setAttribute("src", e), r.setAttribute("pending", 1), r.onerror = s, t(r).bind("scriptload", function() {
                t(this).removeAttr("pending"), i(), setTimeout(function() {
                    t(r).unbind("scriptload")
                }, 10)
            });
            var a=!1;
            r.onload = r.onreadystatechange = function() {
                a || this.readyState&&!/loaded|complete/.test(this.readyState) || (a=!0, r.onload = r.onreadystatechange = null, t(r).trigger("scriptload"))
            }, n.appendChild(r)
        }, this.css = function(e, i) {
            if (t('link[href*="' + e + '"]').length)
                return void i();
            var o = document.createElement("link");
            o.setAttribute("type", "text/css"), o.setAttribute("rel", "stylesheet"), o.setAttribute("href", e), t.browser.msie ? o.onreadystatechange = function() {
                /loaded|complete/.test(o.readyState) && i()
            } : t.browser.opera ? o.onload = i : !function() {
                o && o.sheet ? i() : c = setTimeout(arguments.callee, 20)
            }(), n.appendChild(o)
        }, this.img = function(t, e) {
            var n = new Image;
            n.onload = e, n.onerror = s, n.src = t
        }, this.disable = function(e) {
            t("#lazy-loaded-" + e, n).attr("disabled", "disabled")
        }, this.enable = function(e) {
            t("#lazy-loaded-" + e, n).removeAttr("disabled")
        }, this.destroy = function(e) {
            t("#lazy-loaded-" + e, n).remove()
        }
    }
    t.xLazyLoader = function(t, n) {
        "object" == typeof t && (n = t, t = "init"), (new e)[t](n)
    }, t.xLazyLoader.defaults = {
        js: [],
        css: [],
        img: [],
        name: null,
        timeout: 2e4,
        success: function() {},
        error: function() {},
        complete: function() {},
        each: function() {}
    };
    var n = document.getElementsByTagName("head")[0]
}(jQuery), BEM.DOM.decl("b-keyboard-loader", {
    onSetMod: {
        js: function() {
            var t = this;
            Lego.block["b-keyboard-loader"].call(this.domElem, this.params), $(window).on("keyboardOpened.lego", function() {
                t.params.statOpen && cp(t.params.statOpen)
            }).on("keyboardClosed.lego", function(e) {
                "lego" === e.namespace && t.params.statClose && cp(t.params.statClose)
            })
        }
    }
}), function(t, e) {
    e.block["b-keyboard-loader"] = function(n) {
        function i(n) {
            t("body").removeClass("b-page_keyboard");
            var i = n ? "addClass": "toggleClass", o = c.find(".b-keyboard-popup");
            if (o.length) {
                if (c.find(".b-keyboard-popup__gap")[i]("i-hidden"), c.find(".b-keyboard-popup")[i]("i-hidden").hasClass("i-hidden")) {
                    u.trigger("keyboardClosed"), t(document).trigger("popupsClose.lego");
                    var s = e.block["b-keyboard"]._lastFocusedIn;
                    void 0 !== s && s.focus()
                } else 
                    u.trigger("keyboardOpened.lego");
                o.data("isHidden", !1)
            }
        }
        function o() {
            h || (e.cp(0, 70873, BEM.blocks["i-global"].param("id") + "." + (n.autoLoad ? "auto" : "manual")), h=!0)
        }
        function s(t) {
            return r()[t]
        }
        function r() {
            var e = {};
            return t.each(window.name.split("&"), function() {
                if (this.toString()) {
                    var t = this.split("=");
                    e[t[0]] = t[1]
                }
            }), e
        }
        function a(e, n) {
            var i = r(), o = [];
            i[e] = n, t.each(i, function(t, e) {
                o.push(t + "=" + e)
            }), window.name = o.join("&")
        }
        var c = t(document.body), u = t(window), l = this;
        l.click(function(o) {
            o.preventDefault();
            var s = t(this), r = (this.className.match(/b-keyboard-loader_lang_(\w+)/) || ["", ""])[1];
            if (c.find(".b-keyboard")[0])
                i();
            else {
                var a = BEM.blocks["i-global"].param("lego-static-host-keyboard"), l = a + ("/" === a.charAt(a.length - 1) ? "" : "/") + "blocks/big/", d = [l + "b-keyboard/_keyboard.js"], h = t.browser.msie && (!document.documentMode || document.documentMode < 9) ? ".ie": "", f = [l + "b-keyboard/_keyboard" + h + ".css"];
                s.append('<i class="b-keyboard-loader__progress"/>'), t.xLazyLoader({
                    name: "b-keyboard",
                    js: d,
                    css: f,
                    success: function() {
                        e.block["b-keyboard-popup"]({
                            lang: r,
                            "for": n["for"]
                        }), c.append("<div class=\"b-keyboard i-bem\" onclick=\"return {'b-keyboard': { lang:'" + r + "'}};\"></div>"), BEM.DOM.init(t(".b-keyboard")), s.find(".b-keyboard-loader__progress").remove(), u.trigger("keyboardOpened.lego", {
                            afterLoad: !0
                        }), t(".b-keyboard").on("click", ".b-keyboard__key", function() {
                            var e = t("#text").offset().top;
                            t(".b-keyboard-popup").offset().top < e + 30 && window.scrollTo(0, e - 20)
                        })
                    }
                })
            }
        }), u.bind("keyboardOpen.lego", function() {
            l.click()
        }).bind("keyboardClose.lego", i), n["for"] && u.bind("keyboardOpened.lego", function(e, i) {
            !(i && i.afterLoad && n.autoLoad) && t(n["for"]).focus()
        }), u.bind("keyboardLangChanged.lego keyboardSetLang.lego", function(t, e) {
            var n = l.find(".b-keyboard-loader__flag");
            n[0] && ("tt" === e && (e = "ru"), n.attr("src", n.attr("src").replace(/[a-z]+\.png$/, e + ".png")))
        }), n.storeState = n.storeState!==!1, n.storeState && u.bind("keyboardOpened.lego", function() {
            a("keyboard_state", "open");
            var e = s("keyboard_lang");
            e && t(window).trigger("keyboardSetLang.lego", e)
        }).bind("keyboardClosed.lego", function() {
            t("body").removeClass("b-page_keyboard"), a("keyboard_state", "close")
        }).bind("keyboardLangChanged.lego", function(t, e) {
            a("keyboard_lang", e)
        });
        var d = "#kb" === window.location.hash;
        (n.autoLoad = n.autoLoad || n.storeState && "open" === s("keyboard_state") || d) && l.click();
        var h;
        u.bind("keyboardOpened.lego", function() {
            o(), t("input:focus").focus(), t("body").addClass("b-page_keyboard")
        })
    }
}(jQuery, window.Lego), function(t) {
    function e() {
        function e(t, e) {
            l[t](e, function(t) {
                "error" == t ? h.push(e) : d.push(e) && r.each(e), o()
            }, "lazy-loaded-" + (r.name ? r.name : (new Date).getTime()))
        }
        function i(t) {
            r.complete(t, d, h), r[t]("error" == t ? h : d), clearTimeout(a), clearTimeout(c)
        }
        function o() {
            d.length == f.length ? i("success") : d.length + h.length == f.length && i("error")
        }
        function s() {
            h.push(this.src), o()
        }
        var r, a, c, u, l = this, d = [], h = [], f = [];
        this.init = function(n) {
            if (n) {
                if (r = t.extend({}, t.xLazyLoader.defaults, n), u = {
                    js: r.js,
                    css: r.css,
                    img: r.img
                }, t.each(u, function(t, e) {
                    "string" == typeof e && (e = e.split(",")), f = f.concat(e)
                }), !f.length)
                    return void i("error");
                r.timeout && (a = setTimeout(function() {
                    var e = d.concat(h);
                    t.each(f, function(n, i) {
                        - 1 == t.inArray(i, e) && h.push(i)
                    }), i("error")
                }, r.timeout)), t.each(u, function(n, i) {
                    t.isArray(i) ? t.each(i, function(t, i) {
                        e(n, i)
                    }) : "string" == typeof i && e(n, i)
                })
            }
        }, this.js = function(e, i) {
            var o = t('script[src*="' + e + '"]');
            if (o.length)
                return void(o.attr("pending") ? o.bind("scriptload", i) : i());
            var r = document.createElement("script");
            r.setAttribute("type", "text/javascript"), r.setAttribute("charset", "utf-8"), r.setAttribute("src", e), r.setAttribute("pending", 1), r.onerror = s, t(r).bind("scriptload", function() {
                t(this).removeAttr("pending"), i(), setTimeout(function() {
                    t(r).unbind("scriptload")
                }, 10)
            });
            var a=!1;
            r.onload = r.onreadystatechange = function() {
                a || this.readyState&&!/loaded|complete/.test(this.readyState) || (a=!0, r.onload = r.onreadystatechange = null, t(r).trigger("scriptload"))
            }, n.appendChild(r)
        }, this.css = function(e, i) {
            if (t('link[href*="' + e + '"]').length)
                return void i();
            var o = document.createElement("link");
            o.setAttribute("type", "text/css"), o.setAttribute("rel", "stylesheet"), o.setAttribute("href", e), t.browser.msie ? o.onreadystatechange = function() {
                /loaded|complete/.test(o.readyState) && i()
            } : t.browser.opera ? o.onload = i : !function() {
                o && o.sheet ? i() : c = setTimeout(arguments.callee, 20)
            }(), n.appendChild(o)
        }, this.img = function(t, e) {
            var n = new Image;
            n.onload = e, n.onerror = s, n.src = t
        }, this.disable = function(e) {
            t("#lazy-loaded-" + e, n).attr("disabled", "disabled")
        }, this.enable = function(e) {
            t("#lazy-loaded-" + e, n).removeAttr("disabled")
        }, this.destroy = function(e) {
            t("#lazy-loaded-" + e, n).remove()
        }
    }
    t.xLazyLoader = function(t, n) {
        "object" == typeof t && (n = t, t = "init"), (new e)[t](n)
    }, t.xLazyLoader.defaults = {
        js: [],
        css: [],
        img: [],
        name: null,
        timeout: 2e4,
        success: function() {},
        error: function() {},
        complete: function() {},
        each: function() {}
    };
    var n = document.getElementsByTagName("head")[0]
}(jQuery), function(t) {
    BEM.DOM.decl("b-dropdowna", {
        onSetMod: {
            js: function() {
                this._getSwitcher().on("click", this._toggle, this)
            },
            disabled: function(t, e) {
                this._getSwitcher().setMod(t, e), "yes" == e && this.getPopup().hide()
            }
        },
        _getSwitcher: function() {
            return this._switcher || (this._switcher = this.findBlockInside("b-" + (this.getMod(this.elem("switcher"), "type") || "link")))
        },
        _toggle: function() {
            this.getPopup().toggle(this.elem("switcher"))
        },
        getPopup: function() {
            return this._popup || (this._popup = this.findBlockInside("b-popupa")).on("outside-click", function(e, n) {
                this._getSwitcher().containsDomElem(t(n.domEvent.target)) && e.preventDefault()
            }, this)
        },
        destruct: function() {
            var t = this._popup;
            t && t.destruct(), this.__base.apply(this, arguments)
        }
    }, {
        live: function() {
            this.liveInitOnEvent("switcher", "leftclick", function() {})
        }
    })
}(jQuery), BEM.HTML.decl("b-dropdowna", {
    onBlock: function(t) {
        t.js(!0)
    }
}), function(t) {
    var e = ["down-right", "down", "down-left", "up", "up-right", "up-left", "right-down", "right", "right-up", "left-down", "left", "left-up"];
    BEM.DOM.decl("b-popupa", {
        onSetMod: {
            js: function() {
                this._owner = null, this._isShowed=!1, this._direction = this.getMod("direction") || "down", this._lastDirection = null, this._hasTail=!!this.elem("tail").length, this._hasTail || Object.keys(this.params).forEach(function(t) {
                    0 === t.indexOf("tail") && (this.params[t] = 0)
                }, this)
            }
        },
        show: function(t) {
            return this._isShowed && this._owner === t || (this._owner = t, this._getUnder().show({
                left: - 1e4,
                top: - 1e4
            }), this.pos(), this._getUnder().setMod("animate", "yes")), this
        },
        hide: function() {
            return this._isShowed && this._getUnder().hide(), this
        },
        toggle: function(t) {
            return this.isShowed() ? this.hide() : this.show(t)
        },
        _pos: function() {
            var t = this._calcParams(this._owner);
            return this._hasTail && this.elem("tail").css(t.tailOffsets), this.setMod("direction", t.direction)._getUnder().show(t.offsets), this
        },
        pos: function() {
            return this._isShowed ? this._pos() : this
        },
        isShowed: function() {
            return this._isShowed
        },
        setDirection: function(t) {
            this._direction != t && (this._direction = t, this.isShowed() && this.pos())
        },
        setContent: function(t, e, n) {
            return BEM.DOM.update(this.elem("content"), t, e, n), this.isShowed() ? this.pos() : this
        },
        _isOwnerNode: function() {
            return !(!this._owner ||!this._owner.jquery)
        },
        _calcDimensions: function() {
            var t = this._getUnder().domElem, e = this._getUnder()._getUnder(), n = this.__self.win, i = this._owner, o = this._isOwnerNode(), s = o ? i.offset(): i, r = o ? i.outerWidth(): 0, a = o ? i.outerHeight(): 0, c = n.scrollLeft(), u = n.scrollTop(), l = this.__self.getWindowSize(), d = parseInt(this.elem("content").css("border-top-width"), 10);
            return {
                ownerWidth: r,
                ownerHeight: a,
                ownerLeft: s.left,
                ownerTop: s.top,
                ownerRight: s.left + r,
                ownerBottom: s.top + a,
                ownerHorizMiddle: s.left + r / 2,
                ownerVertMiddle: s.top + a / 2,
                posWidth: t.outerWidth(),
                posHeight: t.outerHeight(),
                underWidth: e.outerWidth(),
                underHeight: e.outerHeight(),
                borderWidth: isNaN(d) ? 0: d,
                windowLeft: c,
                windowRight: c + l.width,
                windowTop: u,
                windowBottom: u + l.height
            }
        },
        _calcParams: function() {
            var e = this._calcDimensions();
            if (this.hasMod("adjustable", "no"))
                return this.calcDirectionParams(this._direction, e);
            var n = {}, i = this.params.directions, o = t.inArray(this._direction, i);
            o>-1 || (o = 0);
            var s, r, a = o;
            do {
                if (s = i[o], r = n[s] = this.calcDirectionParams(s, e), !r.factor)
                    return this._lastDirection = s, r;
                ++o == i.length && (o = 0)
            }
            while (o !== a);
            return n[this._lastDirection || i[0]]
        },
        calcDirectionParams: function(t, e) {
            var n, i = this.params, o = {
                top: 0,
                left: 0
            }, s = {
                marginLeft: 0,
                marginTop: 0
            }, r = t.split("-")[0], a = this._hasTail && e.ownerWidth < i.tailWidthVertical ? (i.tailWidthVertical - e.ownerWidth) / 2: 0, c = this._hasTail && e.ownerHeight < i.tailHeightHorizontal ? (i.tailHeightHorizontal - e.ownerHeight) / 2: 0;
            switch (t) {
            case"up":
            case"down":
                o.left = e.ownerHorizMiddle - e.posWidth / 2, o.top = "down" == t ? e.ownerBottom + i.tailHeightVertical : e.ownerTop - e.posHeight - i.tailHeightVertical, s.marginLeft = e.posWidth / 2 - i.tailWidthVertical / 2, s.marginTop = "down" == t ? 0 - i.tailHeightVertical : 0;
                break;
            case"up-left":
            case"up-right":
            case"down-left":
            case"down-right":
                o.left = "down-right" == t || "up-right" == t ? e.ownerLeft - a : e.ownerRight - e.posWidth + a, o.top = "down" == r ? e.ownerBottom + i.tailHeightVertical : e.ownerTop - e.posHeight - i.tailHeightVertical, s.marginLeft = e.ownerWidth > e.posWidth ? e.posWidth / 2 - i.tailWidthVertical / 2 : e.ownerHorizMiddle - o.left - i.tailWidthVertical / 2, s.marginTop = "down" == r ? e.borderWidth - i.tailHeightVertical : 0 - e.borderWidth;
                break;
            case"left-down":
            case"right-down":
                o.left = "left-down" == t ? e.ownerLeft - e.posWidth - i.tailWidthHorizontal : e.ownerRight + i.tailWidthHorizontal, o.top = e.ownerTop - c, s.marginLeft = "left-down" == t ? 0 - e.borderWidth : e.borderWidth - i.tailWidthHorizontal, s.marginTop = e.ownerHeight < e.posHeight ? e.ownerVertMiddle - o.top - i.tailHeightHorizontal / 2 : e.posHeight / 2 - i.tailHeightHorizontal / 2;
                break;
            case"left":
            case"right":
                o.left = "left" == t ? e.ownerLeft - e.posWidth - i.tailWidthHorizontal : e.ownerRight + i.tailWidthHorizontal, o.top = e.ownerVertMiddle - e.posHeight / 2, s.marginLeft = "left" == t ? 0 - e.borderWidth : e.borderWidth - i.tailWidthHorizontal, s.marginTop = e.posHeight / 2 - i.tailHeightHorizontal / 2;
                break;
            case"left-up":
            case"right-up":
                o.left = "left-up" == t ? e.ownerLeft - e.posWidth - i.tailWidthHorizontal : e.ownerRight + i.tailWidthHorizontal, o.top = e.ownerBottom - e.posHeight + c, s.marginLeft = "left" == r ? 0 - e.borderWidth : e.borderWidth - i.tailWidthHorizontal, s.marginTop = e.ownerHeight > e.posHeight ? e.posHeight / 2 - i.tailHeightHorizontal / 2 : e.ownerVertMiddle - o.top - i.tailHeightHorizontal / 2
            }
            return n = this.calcInWindowFactor(o, e), {
                direction: r,
                factor: n,
                offsets: o,
                tailOffsets: s
            }
        },
        calcInWindowFactor: function(t, e) {
            var n = 0;
            return e.windowTop > t.top && (n += e.windowTop - t.top), t.top + e.underHeight > e.windowBottom && (n += t.top + e.underHeight - e.windowBottom), e.windowLeft > t.left && (n += e.windowLeft - t.left), t.left + e.underWidth > e.windowRight && (n += t.left + e.underWidth - e.windowRight), n
        },
        getDefaultParams: function() {
            return {
                tailOffset: 19,
                tailWidthHorizontal: 9,
                tailWidthVertical: 19,
                tailHeightHorizontal: 19,
                tailHeightVertical: 10,
                shadowSize: 7,
                directions: e
            }
        },
        destruct: function() {
            var t = this._under;
            t ? this._destructing || (this._destructing=!0, this.hide(), BEM.DOM.destruct(!1, t.domElem), this.__base(!0)) : this.__base.apply(this, arguments)
        },
        _getUnder: function() {
            if (!this._under) {
                var e = t(BEM.HTML.build({
                    block: "i-popup",
                    zIndex: this.params.zIndex,
                    mods: {
                        autoclosable: this.getMod("autoclosable") || "yes",
                        fixed: this.hasMod("direction", "fixed") && "yes"
                    },
                    underMods: this.params.underMods,
                    underMix: [{
                        block: "b-popupa",
                        elem: "under"
                    }
                    ]
                }));
                (this._under = this.findBlockOn(e, "i-popup")).on({
                    show: this._onUnderShowed,
                    hide: this._onUnderHidden,
                    "outside-click": this._onUnderOutsideClicked
                }, this).elem("content").append(this.domElem)
            }
            return this._under
        },
        _onUnderShowed: function() {
            this._isShowed=!0, this.bindToDomElem(t(window), "resize", this.pos)._isOwnerNode() && this.bindToDomElem(this._owner.parents().add(this.__self.win), "scroll", this.pos), this.trigger("show")
        },
        _onUnderHidden: function() {
            this._isShowed=!1, this.unbindFromDomElem(t(window), "resize")._isOwnerNode() && this.unbindFromDomElem(this._owner.parents().add(this.__self.win), "scroll"), this.trigger("hide")
        },
        _onUnderOutsideClicked: function() {
            this.trigger.apply(this, arguments)
        }
    }, {
        live: function() {
            this.liveBindTo("close", "leftclick", function() {
                this.hide()
            })
        }
    }), BEM.HTML.decl("b-popupa", {
        onBlock: function(e) {
            var n=!1;
            t.each(e.param("content"), function(t, e) {
                return !(n = "close" == e.elem)
            }), e.mods({
                theme: "ffffff",
                direction: "down",
                "has-close": n && "yes"
            }).js(!0).afterContent({
                elem: "shadow"
            })
        },
        onElem: {
            content: function(t) {
                t.wrap({
                    elem: "wrap-cell",
                    tag: "td"
                }).wrap({
                    tag: "tr"
                }).wrap({
                    elem: "wrap",
                    tag: "table"
                })
            },
            close: function(t) {
                t.tag("i")
            },
            shadow: function(t) {
                t.tag("i")
            },
            tail: function(t) {
                t.tag("i").wrapContent({
                    elem: "tail-i",
                    tag: "i"
                })
            }
        }
    })
}(jQuery), BEM.DOM.decl("b-popupa", {
    immediateHide: function() {
        return this._isShowed && this._getUnder().delMod("animate").delMod("visibility"), this
    },
    onSetMod: {
        overlapping: {
            yes: function() {
                this.afterCurrentEvent(function() {
                    this._bindOnShowHide()
                })
            }
        }
    },
    _pos: function() {
        var t = this._calcParams(this._owner);
        return this.elem("tail") && this.elem("tail").css(t.tailOffsets), this.setMod("direction", t.direction)._getUnder().show(t.offsets), this
    }
}), function(t) {
    function e() {
        return s.length ? s.shift() : o ? o.clone() : o = i()
    }
    function n(t) {
        s.push(t)
    }
    function i() {
        var e, n = (r.safari || r.webkit) && navigator.userAgent.toLowerCase().indexOf("mobile")>-1;
        if (n)
            return t("<div/>");
        try {
            e = location.hostname !== document.domain
        } catch (i) {
            e=!0
        }
        return t("<iframe" + (r.msie && r.version < 9 ? ' frameborder="0"' : "") + (r.msie && e ? " src=\"javascript:void(function(){document.open();document.write('&lt;script&gt;document.domain=\\'" + document.domain + "\\'&lt;/script&gt;');document.close();})()\"" : "") + "/>")
    }
    var o, s = [], r = t.browser;
    BEM.DOM.decl("i-popup", {
        onSetMod: {
            js: function() {
                var e = t.browser;
                e.msie && parseInt(e.version) >= 9 && (this.domElem[0].onresize = function() {
                    this.className += ""
                })
            },
            visibility: {
                visible: function() {
                    var t = this._getUnder(), e = t.parent();
                    this.hasMod(t, "type", "paranja") ? e.is("body") || t.appendTo("body") : e[0] !== this.domElem[0] && t.prependTo(this.domElem), this._inBody || (this._inBody=!!this.domElem.appendTo("body")), this.trigger("show")
                },
                "": function() {
                    var t = this._getUnder();
                    this.hasMod(t, "type", "paranja") && t.remove(), this._putUnder(), this.trigger("hide")
                }
            }
        },
        _getUnder: function() {
            return this._under || (this._under = e().attr("class", this._underClass || (this._underClass = this.findElem("under").remove().attr("class"))))
        },
        _putUnder: function() {
            n(this._under), delete this._under
        },
        show: function(t) {
            return t && this.domElem.css(t), this.setMod("visibility", "visible")
        },
        hide: function() {
            return this.delMod("animate").delMod("visibility")
        }
    }, {
        live: !0
    })
}(jQuery), BEM.HTML.decl("i-popup", {
    onBlock: function(t) {
        t.mod("autoclosable", "yes").js(!0).wrapContent({
            elem: "content"
        }).afterContent({
            elem: "under",
            mods: t.param("underMods"),
            mix: t.param("underMix")
        }).param("zIndex") && t.attr("style", "z-index:" + (32700 + t.param("zIndex")))
    }
}), function(t) {
    var e = t.browser.opera && t.browser.version < 12.1 ? "keypress": "keydown";
    BEM.DOM.decl({
        name: "i-popup",
        modName: "autoclosable",
        modVal: "yes"
    }, {
        onSetMod: {
            visibility: {
                visible: function() {
                    this.afterCurrentEvent(function() {
                        if (this.hasMod("visibility", "visible")) {
                            var n = this._under;
                            this.bindToDoc("leftclick", function(e) {
                                this.containsDomElem(t(e.target)) || this._onOutClick(e)
                            }).bindToDoc(e, function(t) {
                                27 == t.keyCode && this.hide()
                            }), n && n.is("iframe") && this.hasMod(n, "type", "paranja") && this.bindToDomElem(t([n[0].contentWindow, n[0].contentWindow.document]), "leftclick", this._onOutClick)
                        }
                    }), this.__base.apply(this, arguments)
                },
                "": function() {
                    return this.unbindFromDoc("leftclick " + e).__base.apply(this, arguments)
                }
            }
        },
        _onOutClick: function(e) {
            var n = t.Event("outside-click");
            this.trigger(n, {
                domEvent: e
            }), n.isDefaultPrevented() || this.hide()
        }
    })
}(jQuery), BEM.HTML.decl("b-dropdowna", {
    onElem: {
        switcher: function(t) {
            t.tag("span")
        }
    }
}), BEM.DOM.decl("b-menu", {
    onElemSetMod: {
        trigger: {
            state: function(t, e, n) {
                this.toggleMod(this.findElem(t.closest(this.buildSelector("layout-vert-cell")), "item-content").eq(0), "visibility", "visible", "opened" == n).trigger("trigger", {
                    domElem: t,
                    state: n
                })
            }
        },
        item: {
            state: {
                current: function(t) {
                    var e = this, n = e.__self.getName(), i = e.elem("item", "state", "current").filter(function() {
                        return $(this).closest(e.buildSelector()).bem(n) === e
                    });
                    e.findElem(t.parents(e.buildSelector("item-content")).prev(e.buildSelector("item")), "trigger").each(function() {
                        e.setMod($(this), "state", "opened")
                    }), e.delMod(i, "state").trigger("current", {
                        prev: i,
                        current: t
                    })
                }
            }
        }
    },
    onTriggerClick: function(t) {
        t.preventDefault(), this.toggleMod(t.data.domElem, "state", "opened")
    },
    onItemSelectorClick: function(t) {
        var e = this._getItemByEvent(t);
        this.hasMod(e, "state", "disabled") || this.setMod(e, "state", "current")
    },
    _getItemByEvent: function(t) {
        return t.data.domElem.closest(this.buildSelector("item"))
    }
}, {
    live: function() {
        this.liveBindTo("trigger", "leftclick", function(t) {
            this.onTriggerClick(t)
        }).liveBindTo("item-selector", "leftclick", function(t) {
            this.onItemSelectorClick(t)
        })
    }
}), BEM.DOM.decl("input", {
    onSetMod: {
        js: {
            inited: function() {
                this.__base.apply(this, arguments), BEM.blocks["b-link"].on(this.elem("samples"), "click", this._onSampleClick, this)
            }
        },
        disabled: function(t, e) {
            this.findBlocksInside(this.elem("samples"), "b-link").forEach(function(n) {
                n.setMod(t, e)
            })
        }
    },
    destruct: function() {
        BEM.blocks["b-link"].un(this.domElem, "click"), this.__base.apply(this, arguments)
    },
    _onSampleClick: function(t) {
        var e = t.target.domElem, n = this.elemParams(e);
        this.val("val"in n ? n.val : e.text(), {
            source: "sample"
        })
    }
}), BEM.DOM.decl("input", {
    _onSampleClick: function() {
        this.__base.apply(this, arguments), this.noSuggest=!0, this.setMod("focused", "yes"), this.noSuggest=!1
    }
}), BEM.DOM.decl({
    block: "input",
    modName: "keyboard",
    modVal: "yes"
}, {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments), $(window).bind("keyboardOpened.lego", this.changeThis(function() {
                this.setMod("focused", "yes")._focused=!0
            }))
        }
    }
}), BEM.DOM.decl({
    name: "input",
    modName: "ahead",
    modVal: "yes"
}, {
    onSetMod: {
        js: function() {
            var t = $.browser.opera && $.browser.version < 12.1 ? "keypress": "keydown", e = this;
            this.__base(), this._ahead = this.buildAhead(), this._intent=!1, this._text = this.val(), this._preventAhead=!1, this._lastBackspace = 0, this._filler = "", this._hint = "", this.on("change", this._onTextChange).bindTo("click", function(t) {
                if (!this._preventAhead && this.elem("input").is(t.target)) {
                    var n = t.pageX - this.domElem.offset().left - parseInt(this.elem("input").css("padding-left"), 10), i = this.elem("ahead-filler").width(), o = this.elem("ahead-hintholder").width();
                    n > i && o > n && (this.trigger("ahead-click"), setTimeout(function() {
                        e._fill({
                            usage: "mouse"
                        })
                    }, 0))
                }
            }).bindTo(t, this._onKeyDown), this.elem("ahead-hint").on("focus", function() {
                e.elem("input").focus()
            })
        }
    },
    _onKeyDown: function(t) {
        switch (this.__base.apply(this, arguments), new Date - this._lastBackspace > this.params.backspaceDelay && (this._preventAhead=!1), t.which) {
        case 8:
            if (!this.params.hideOnBackspace)
                break;
            this._preventAhead=!0, this._lastBackspace = new Date, this._intent = this._clearAhead();
            break;
        case 9:
            !this._isPopupShown || t.ctrlKey || t.altKey || (t.preventDefault(), this._fill({
                usage: "tab"
            }));
            break;
        case 39:
            this.val().length === this.getSelectionEnd() && this._fill({
                usage: "arrow"
            }) && t.preventDefault()
        }
    },
    setAhead: function(t, e, n) {
        return this._preventAhead?!1 : n.items && t === n.items[0] || n.val !== this.val() ? this._clearAhead() : (this.elem("ahead-hintholder").text(e), void(this._intent = this._showAhead(t, t + e)))
    },
    unsetAhead: function() {
        this._clearAhead(), this._intent=!1
    },
    onPopupShow: function() {
        this._isPopupShown=!0
    },
    onPopupHide: function() {
        this._preventAhead=!0, this._clearAhead()
    },
    _onTextChange: function() {
        var t = this._text.length - (this._text = this.val()).length;
        t > 0 && this.shiftAhead(t), this._intent&&!this._intent.match("^" + BEM.blocks["i-common__string"].escapeRegExp(this.val())) && this._clearAhead()
    },
    _fill: function(t) {
        var e = $.extend(t, {
            chars: this._intent.length - this._text.length
        });
        if ("string" == typeof this._intent && e.chars > 0) {
            try {
                var n = document.createEvent("TextEvent");
                n.initTextEvent("textInput", !0, !0, null, this._intent.slice(this.val().length), 9), this._input.selectionEnd = this._input.selectionStart = this._text.length, this._input.dispatchEvent(n)
            } catch (i) {
                var o = this.elem("input").get(0);
                if (this.val(this._intent), o.createTextRange) {
                    var s = o.createTextRange();
                    s.moveEnd("character", this._text.length), s.collapse(!1), s.select()
                }
            }
            this.trigger("ahead-fill", e)
        }
        return e.chars > 0
    },
    getSelectionEnd: function() {
        var t = this.elem("control")[0], e = 0;
        if ("number" == typeof t.selectionEnd)
            e = t.selectionEnd;
        else {
            var n = document.selection.createRange();
            if (n && n.parentElement() === t) {
                var i = t.value.length, o = t.createTextRange();
                o.moveToBookmark(n.getBookmark());
                var s = t.createTextRange();
                s.collapse(!1), e = o.compareEndPoints("EndToEnd", s)>-1 ? i : - o.moveEnd("character", - i)
            }
        }
        return e
    },
    getDefaultParams: function() {
        return $.extend(this.__base(), {
            backspaceDelay: 100,
            hideOnBackspace: !0
        })
    },
    buildAhead: function() {
        return this._ahead = $(BEMHTML.apply({
            block: "input",
            elem: "ahead"
        })), BEM.DOM.before(this.elem("control"), this._ahead), this._ahead
    },
    _setAhead: function(t, e) {
        return this.elem("ahead-filler").text(!this._preventAhead && (this._filler = t) || ""), this.elem("ahead-hint").val(!this._preventAhead && (this._hint = e) || ""), this.elem("ahead-hintholder").text(!this._preventAhead && e || ""), e
    },
    _showAhead: function(t, e) {
        return "undefined" == typeof t && this._clearAhead(), this._setAhead(t, e)
    },
    _clearAhead: function() {
        return this._setAhead("", ""), !1
    },
    shiftAhead: function(t) {
        var e = this._filler.slice(0, - t);
        return this._setAhead(e, this._hint)
    }
}), function() {
    var t = function(t) {
        var e = t.charCode || t.keyCode || t.which || 0, n = t.ctrlKey || t.altKey || t.metaKey, i = e >= 48 && 57 >= e, o = e >= 96 && 105 >= e, s = [e >= 65 && 90 >= e, e >= 1025 && 1071 >= e, 0 === e, 231 === e, 186 === e, String.fromCharCode(e).toLowerCase() !== String.fromCharCode(e).toUpperCase()].some(Boolean);
        return !n && (i || o || s)
    };
    BEM.DOM.decl({
        name: "input",
        modName: "autofocus",
        modVal: "capture"
    }, {
        onSetMod: {
            js: function() {
                this.__base.apply(this, arguments), this.bindToDoc("keydown", this._autoFocusBind)
            },
            autofocus: {
                yes: function() {
                    this.bindToDoc("keydown", this._autoFocusBind)
                },
                "": function() {
                    this.unbindFromDoc("keydown")
                }
            }
        },
        _autoFocusBind: function(e) {
            if (!this.hasMod("focused", "yes")&&!$(".popup_visibility_visible:not(.b-topbar__popup), .v_sform_bg:visible, .left-banner-opened").length) {
                var n;
                try {
                    n = document.activeElement.tagName.toLowerCase()
                } catch (e) {}
                if (t(e) && "input" !== n && "textarea" !== n) {
                    var i = this.elem("control")[0], o = this.val();
                    if (o.length > 0 && " " !== o.slice( - 1) && (o += " ", this.val(o)), i.createTextRange) {
                        var s = i.createTextRange();
                        s.collapse(!1), s.select()
                    } else 
                        i.selectionStart && i.setSelectionRange(o.length, o.length);
                    this.setMod("focused", "yes")
                }
            }
        }
    })
}(), BEM.DOM.decl("suggest2", {
    onSetMod: {
        js: {
            inited: function() {
                this.params.form || this.findBlockOutside("suggest2-form"), this._lastSubmitTime = 0, this._lastBlurTime = 0, this._clear()
            }
        }
    },
    getDefaultParams: function() {
        return {
            submitBySelect: !1,
            updateOnEnterByKeyboard: !0,
            onFocus: "request",
            requestOnEmptyInput: !1
        }
    },
    destruct: function() {
        this.getItems().forEach(function(t) {
            t.destruct()
        })
    },
    init: function() {
        var t = this;
        t.params.form.on("submit", function() {
            t._lastSubmitTime = (new Date).getTime(), t._currentSearchCGIParams ? t._currentSearchCGIParams=!1 : t._removeSearchCGIParams(), t._clear(), t._popup.hide()
        }), t._model = t.initModel(), t._view = t.initView(), t._counter = t.initCounter(), t._model.on({
            response: this.setItems,
            reject: function(e, n) {
                2 === n.code && (t._popup.hide(), t._clear())
            }
        }, t), t._input = t.params["suggest2-input"], t._popup = t.params["suggest2-popup"], t.bindEvents()
    },
    initModel: function() {
        return BEM.create("suggest2-model", this.params)
    },
    getModel: function() {
        return this._model
    },
    initView: function() {
        var t = this.getMod("theme") || "normal", e = {
            theme: t
        };
        return "large" === t && (e.group = "type"), "label" === this.getMod("group") && (e.group = "label"), BEM.create({
            block: "suggest2-view",
            mods: e
        })
    },
    getView: function() {
        return this._view
    },
    initCounter: function() {
        return this.params.counter ? this.params.counter.initValues().bindEvents() : void 0
    },
    getCounter: function() {
        return this._counter
    },
    setDataConverter: function(t) {
        return this.getModel().getProvider().convert = t, this
    },
    setRequestData: function(t) {
        return this.getModel().getProvider().getRequestData = t, this
    },
    setRequestUrl: function(t) {
        return this.getModel().getProvider().getRequestUrl = t, this
    },
    setOwner: function(t) {
        return this._popup.owner(t), this
    },
    bindEvents: function() {
        var t = this;
        return t._input.bindEvents({
            change: t._onChange,
            focus: t._onFocus,
            blur: t._onBlur,
            click: t._onFocus
        }, t), t._popup.on({
            preshow: function() {
                this.trigger("preshow")
            },
            show: function() {
                this.trigger("show")
            },
            hide: function(t, e) {
                this.trigger("hide", e)
            }
        }, this), t.params.button && t.params.button.bindTo("mousedown", function() {
            t._input.realVal(t._input.val()), t._popup.hide()
        }), t
    },
    setItems: function(t, e) {
        var n = this;
        return n._isFormSubmitDurinQuery() ||!e.data.items.length ? (n._popup.hide(), n._clear()) : n._input.val() !== e.val ? n._clear() : (n._text = e.val, n._pos = e.pos, n._meta = e.data.meta || {}, n._popup.show(n._view.html(e.data.items, n._meta)).width(n.getPopupWidth()), n._items = n.findBlocksInside("suggest2-item"), n._clearItems().trigger("update", {
            val: n._text,
            pos: n._pos,
            items: n._items
        }), n)
    },
    getItems: function() {
        return this._items || []
    },
    isBlur: function() {
        return this._isBlur
    },
    getPopupWidth: function() {},
    _isFormSubmitDurinQuery: function() {
        return this._lastSubmitTime && this._model.isTimeInRequestInterval(this._lastSubmitTime) ? (this._lastSubmitTime = 0, !0) : !1
    },
    _onChange: function(t, e) {
        this._model.request(this._input.realVal(this._input.val()), this._input.getCaretPosition(), e)
    },
    _onFocus: function() {
        !this._popup.isShown() && this._isBlur && this.params.onFocus && this._isLastBlurTimeLater(50) && (this.getItems().length && this._input.val() === this._text ? (this._clearItems(), this._popup.show()) : this._isMouseDown || "request" !== this.params.onFocus || this._onChange())
    },
    _onBlur: function() {
        this._lastBlurTime = (new Date).getTime(), this._isBlur ? (this._clear(), this._popup.hide()) : this._freezeInputFocus ? (this._clear(), this._input.focus()) : this._isMouseDown && this._input.focus(), this._isBlur=!0
    },
    _getItemIndex: function(t) {
        return this.getItems().indexOf(t)
    },
    _clear: function() {
        return this._isBlur=!0, this._freezeInputFocus=!1, this._clearItems()
    },
    _clearItems: function() {
        return this._isMouseDown=!1, this._enterItemIndex =- 1, this._unselectItems()
    },
    _isLastBlurTimeLater: function(t) {
        return (new Date).getTime() - this._lastBlurTime > t
    },
    _unselectItems: function(t) {
        var e = t ? [].slice.call(arguments): this.getItems();
        return e.forEach(function(t) {
            t.onLeave()
        }), this
    },
    _onSelectItem: function(t, e) {
        var n = this, i = n._getItemIndex(t), o = t.onSelect(e).val();
        return n._text = n._input.realVal(o), n._isBlur=!0, n._freezeInputFocus=!1, n.trigger("select", {
            val: n._text,
            pos: n._pos,
            item: t,
            meta: n._meta,
            itemIndex: i,
            byKeyboard: e
        }), n._setSearchCGIParams(t), o!==!1 && (n._items = [], n.params.submitBySelect && n.params.form.submit()), i
    },
    _currentSearchCGIParams: !1,
    _setSearchCGIParams: function(t) {
        if (this._removeSearchCGIParams(), t.params.searchCGI) {
            var e = [];
            t.params.searchCGI.forEach(function(t) {
                e.push({
                    block: "suggest2",
                    elem: "search-cgi",
                    tag: "input",
                    attrs: {
                        type: "hidden",
                        name: t[0],
                        value: t[1]
                    }
                })
            }), this._currentSearchCGIParams=!0, BEM.DOM.after(this._input.get().domElem, BEMHTML.apply(e))
        }
    },
    _removeSearchCGIParams: function() {
        BEM.DOM.destruct(this.findElem(this.params.form.domElem, "search-cgi"))
    },
    _onDownItem: function() {
        this._isBlur=!1, this._isMouseDown=!0
    }
}), BEM.DOM.decl("suggest2", {
    bindEvents: function() {
        var t = this;
        return t._input.bindEvents({
            keydown: t._onKeyDown,
            keyup: t._onKeyUp,
            keypress: t._onKeyPress
        }, t), BEM.blocks["suggest2-item"].on(t.domElem, {
            mouseover: function(e) {
                t._onEnterItem(e.block, !1)
            },
            mouseout: function(e) {
                t._onLeaveItem(e.block, !1)
            },
            mousedown: function() {
                t._onDownItem()
            },
            leftclick: function(e) {
                t._onSelectItem(e.block, !1)
            }
        }), t.__base.apply(t, arguments)
    },
    _onKeyDown: function(t) {
        var e = this.__self.keyboard, n = t.which;
        !this._popup.isShown() || n !== e.up && n !== e.down || (t.preventDefault(), this._enterByKeyboard(n))
    },
    _onKeyUp: function(t) {
        var e = this.__self.keyboard;
        if ((t.which === e.left || t.which === e.right) && (this.trigger("change-cursor-pos"), this._model.request(this._input.realVal(this._input.val()), this._input.getCaretPosition())), t.which === e.right&&~this._enterItemIndex && this._enterItemByKeyboard) {
            var n = this._items[this._enterItemIndex];
            n && this.trigger("ahead", {
                item: n,
                val: "link" === n.getMod("interact") ? n.elem("text").text(): n.val()
            })
        }
    },
    _onKeyPress: function(t) {
        if (t.which === this.__self.keyboard.enter&&~this._enterItemIndex && this._enterItemByKeyboard) {
            var e = this._items[this._enterItemIndex];
            e && (t.preventDefault(), this._onSelectItem(e, !0))
        }
    },
    _enterByKeyboard: function(t) {
        var e = t - 39, n = this._items.length, i = this._enterItemIndex + e;
        return - 1 === i || i === n ? (this._enterItemIndex = i, this._unselectItems(this._items[i]), this.trigger("out", {
            val: this._text,
            byKeyboard: !0
        })) : (0 > i ? i = n - 1 : i >= n && (i = 0), void this._onEnterItem(this._items[i], !0))
    },
    _onEnterItem: function(t, e) {
        this._enterItemIndex = this._getItemIndex(t), this._enterItemByKeyboard = e, this._unselectItems(), t.onEnter(e)
    },
    _onLeaveItem: function(t) {
        this._clearItems()._unselectItems(t)
    }
}, {
    keyboard: {
        enter: 13,
        left: 37,
        up: 38,
        right: 39,
        down: 40
    },
    live: function() {
        this.liveBindTo("mousedown", function() {
            this._onDownItem()
        })
    }
}), BEM.decl("suggest2", {
    bindEvents: function() {
        function t() {
            e._popup && e._popup.isShown() && e._popup.hide()
        }
        var e = this;
        e.__base.apply(e, arguments), e._input.bindEvents({
            click: function() {
                e._popup.isShown() || (e._tabChange=!1, e._onChange())
            }
        }), BEM.blocks["suggest2-item"].on(e.domElem, {
            mousedown: function() {
                e.bindToDoc("mouseup.suggest2", function() {
                    e._isBlur=!0, e._isMouseDown=!1, e.unbindFromDoc("mouseup.suggest2")
                })
            }
        }), $(window).bind("keyboardOpened.lego", function() {
            e._popup && e._popup.isShown() && e._popup.get() && e._popup.get().repaint()
        });
        var n=!0;
        $(window).on("focus", function() {
            var i;
            try {
                i = document.activeElement
            } catch (o) {}
            n || (n=!0, i && e._input.getControl() && e._input.getControl().is(i) && (e._tabChange=!0), t())
        }).on("blur", function() {
            n=!1, t()
        })
    },
    init: function() {
        this.__base.apply(this, arguments), this.trigger("model-ready")
    },
    _onChange: function() {
        return this._tabChange=!1, this.__base.apply(this, arguments)
    },
    _onFocus: function() {
        this._tabChange || this.__base.apply(this, arguments)
    }
}), BEM.DOM.decl("suggest2-detect", {
    onSetMod: {
        js: {
            inited: function() {
                this.bindToDoc("keydown", function(t) {
                    this.__self._pressedKeyCode = t.which
                }, this).bindToDoc("keyup", function() {
                    this.__self._pressedKeyCode = null
                }, this)
            }
        }
    }
}, {
    _pressedKeyCode: null,
    isPressedEscape: function() {
        return 27 === this._pressedKeyCode
    },
    isSupportInlineSVG: function() {
        var t = document.createElement("div");
        return t.innerHTML = "<svg/>", "http://www.w3.org/2000/svg" === (t.firstChild && t.firstChild.namespaceURI)
    }
}), BEM.decl("suggest2-model", {
    onSetMod: {
        js: {
            inited: function() {
                this._requests = []
            }
        }
    },
    destruct: function() {
        return this._provider && this._provider.destruct(), this.__base.apply(this, arguments)
    },
    getProvider: function() {
        return this._provider ? this._provider : (this._provider = this.setProvider(this.params), this._provider)
    },
    setProvider: function(t) {
        return this._provider = BEM.create("suggest2-provider", t), this._provider
    },
    getCancelRequestConditions: function() {
        return [{
            block: "suggest2",
            show: !1
        }, {
            block: "suggest2",
            event: "select"
        }, {
            block: "suggest2",
            event: "out"
        }, {
            block: "suggest2-item",
            show: !1
        }, {
            block: "suggest2-popup",
            event: "hide"
        }, {
            suggest: !1
        }, {
            source: "sample"
        }
        ]
    },
    request: function(t, e, n) {
        var i = this._isNeedRequest(t, n);
        if (!i.need)
            return this.trigger("reject", {
                val: t,
                pos: e,
                code: i.code,
                reason: i.reason || {}
            });
        var o = {
            val: t,
            requestTime: (new Date).getTime()
        };
        return this.trigger("request", {
            val: t,
            pos: e
        }), this.getProvider().get(t, e, function(t, e, n) {
            o.responseTime = (new Date).getTime(), this._requests.push(o), this._onResponse(t, e, n)
        }.bind(this)), this
    },
    isTimeInRequestInterval: function(t) {
        for (var e = this._requests.length - 1; e >= 0; e--)
            if (t > this._requests[e].requestTime && t < this._requests[e].responseTime)
                return !0;
        return !1
    },
    _isNeedRequest: function(t, e) {
        if ("" === t&&!this.params.requestOnEmptyInput)
            return {
                need: !1,
                code: 2
            };
        if (!e)
            return {
                need: !0
            };
        var n, i = this.getCancelRequestConditions();
        t: for (var o = 0; o < i.length; o++) {
            n = i[o];
            for (var s in n)
                if (n.hasOwnProperty(s) && n[s] !== e[s])
                    continue t;
            return {
                need: !1,
                code: 1,
                reason: {
                    condition: n
                }
            }
        }
        return {
            need: !0
        }
    },
    _onResponse: function(t, e, n) {
        this.trigger("response", {
            val: t,
            pos: e,
            data: n
        })
    }
}), BEM.decl("suggest2-provider", {
    getDefaultParams: function() {
        return {
            cache: !1,
            type: "GET",
            dataType: "jsonp",
            timeout: 2e3
        }
    },
    get: function(t, e, n) {
        var i = this, o = function() {
            n.call(i, t, e, i.convert(i._lastData))
        };
        return i._lastText === t ? o() : (i._lastText = t, void $.ajax(i._extendParamsData(i.getRequestData(t, e))).done(function(t) {
            i._lastData = t
        }).fail(function() {
            i._lastData = []
        }).always(o))
    },
    convert: function(t) {
        return {
            orig: t[0],
            items: t[1],
            meta: t[2]
        }
    },
    getRequestData: function(t, e) {
        var n = {};
        return BEM.blocks["suggest2-detect"].isSupportInlineSVG() && (n.svg = 1), $.extend({
            part: t,
            pos: e
        }, n)
    },
    getRequestUrl: function() {
        return this.params.url
    },
    _extendParamsData: function(t) {
        return this.params.url = this.getRequestUrl(), this.params.data = $.extend(this.params.data || {}, t), this.params
    },
    _lastText: void 0,
    _lastData: []
}), BEM.decl("suggest2-view", {
    html: function(t, e) {
        return BEMHTML.apply(this.build(t, e))
    },
    build: function(t, e) {
        return {
            block: "suggest2",
            elem: "content",
            mods: {
                theme: "normal"
            },
            content: this.buildItems(t, e)
        }
    },
    buildItems: function(t) {
        return t.reduce(function(t, e) {
            return t.push(this.buildItem(e)), t
        }.bind(this), [])
    },
    buildItem: function(t) {
        var e = this.__self._getPrefs(t), n = this.__self._getType(t, e), i = "html" === n ? {}
        : {
            type: n
        };
        return this._getItemByType(t, e, n, {
            text: {
                elem: "text",
                content: this.__self._highlight(this.__self._getText(t, e), e)
            },
            mods: i,
            js: e.js || {},
            props: {},
            content: []
        })
    },
    _getItemByType: function(t, e, n, i) {
        var o = i.text, s = i.mods, r = i.js, a = i.props, c = i.content;
        if (e.pers && (s.personal = "yes"), "fact" === n)
            c.push(o, {
                elem: "bullet"
            }, {
                elem: "fact",
                content: t[2]
            });
        else if ("weather" === n)
            c.push(o, {
                elem: "icon",
                elemMods: {
                    weather: t[3].replace("-", "minus-").replace("+", "plus-").replace(/_/g, "-")
                }
            }, {
                elem: "fact",
                content: t[2]
            });
        else if ("traffic" === n)
            c.push(o, {
                elem: "icon",
                elemMods: {
                    traffic: t[3]
                }
            }, {
                elem: "fact",
                content: t[2]
            });
        else if ("nav" === n)
            s.interact = "link", a.url = this.__self._resolveUrl(t[4]), c.push({
                elem: "text",
                content: t[3]
            }, {
                elem: "bullet"
            }, {
                elem: "fact",
                content: t[2]
            });
        else if ("icon" === n) {
            var u = {};
            e.icon.forEach(function(t, e, n) {
                Array.isArray(t) ? u[t[0]] = t[1] : 0 === e && (u[t] = n[e + 1])
            }), c.push(o, $.extend({
                elem: "icon",
                elemMods: {
                    type: "favicon"
                }
            }, u)), e.fact && c.push({
                elem: "fact",
                content: e.fact
            })
        } else 
            "html" === n ? (r.val = t[1], c.push(e.body)) : c.push(o);
        return e.url && (s.interact = "link", a.url = this.__self._resolveUrl(e.url), e.target && (a.target = e.target)), e.search_cgi && (r.searchCGI = e.search_cgi), $.extend({
            block: "suggest2-item",
            mods: s,
            js: r,
            content: c
        }, a)
    }
}, {
    _getPrefs: function(t) {
        if (!Array.isArray(t))
            return {};
        var e = t[t.length - 1];
        return $.isPlainObject(e) ? e : {}
    },
    _getType: function(t, e) {
        var n = Array.isArray(t) ? t[0] || "text": "text";
        return "icon" !== n || e.icon ? n : "text"
    },
    _getText: function(t) {
        return Array.isArray(t) ? t[1] : t
    },
    _highlight: function(t, e) {
        if (!e.hl)
            return t;
        var n, i = e.hl.sort(function(t, e) {
            return t[0] - e[0]
        }).filter(function(t) {
            return t[0] >= 0
        }), o = [], s = 0;
        return i.length ? (i.forEach(function(e) {
            n = s > e[0] ? s : e[0], o.push(t.slice(s, n)), o.push({
                elem: "highlight",
                content: t.slice(n, s = e[1])
            })
        }, this), o.push(t.slice(s)), o.filter(function(t) {
            return !!t
        })) : t
    },
    _resolveUrl: function(t) {
        return (/[\/]{2}/.test(t) ? "" : "http://") + t
    }
}), BEM.decl("suggest2-view", {}, {
    _getGroupType: function(t, e) {
        return t.pers ? "nah" : ~["nav"].indexOf(e) ? e : "text"
    },
    _getTextLabel: function(t) {
        return BEM.I18N("suggest2", t)
    }
}), BEM.decl("suggest2-popup", {
    onSetMod: {
        js: {
            inited: function() {
                this._popup = this.params.popup, this._popup.on("hide", function() {
                    BEM.blocks["suggest2-detect"].isPressedEscape() && this.trigger("hide", {})
                }, this), this.bindSuggestEvents()
            }
        }
    },
    show: function(t) {
        return this._popup.domElem ? (t && this._popup.setContent(t), this.isShown() ? this : (this.trigger("preshow"), this._popup.show(this.params.owner), this.trigger("show"), this)) : this
    },
    hide: function(t) {
        return this.isShown() && (this._popup.hide(), this.trigger("hide", t || {})), this
    },
    setContent: function(t) {
        return this._popup.setContent(t), this
    },
    isShown: function() {
        return this._popup ? this._popup.isShown() : !1
    },
    owner: function(t) {
        return t ? (this.params.owner = t, this) : this.params.owner
    },
    getOwnerPos: function() {
        return this._popup.getOwnerPos() || {}
    },
    get: function() {
        return this._popup
    },
    width: function(t) {
        return void 0 === t ? this._popup.domElem.outerWidth() : (this._popup.domElem.outerWidth(t), this)
    },
    destruct: function() {
        return this._popup ? this._popup.destruct() : void 0
    },
    bindSuggestEvents: function() {
        return this.params.suggest.on({
            select: function(t, e) {
                this.hide(e)
            },
            update: function(t, e) {
                e.items.length || this.hide({
                    response: "empty"
                })
            }
        }, this), this
    }
}), BEM.decl("suggest2-input", {
    onSetMod: {
        js: {
            inited: function() {
                var t = this;
                t._input = t.params.input, t.disableBrowserAutocomplete(), t.realVal(t.val()), t.bindEvents({
                    change: function(e, n) {
                        t.trigger("change", n)
                    },
                    clear: $.throttle(function(e, n) {
                        t.trigger("clear", n)
                    }, 300)
                }, t), t.bindSuggestEvents()
            }
        }
    },
    get: function() {
        return this._input
    },
    focus: function() {
        return this._input.setMod("focused", "yes"), this
    },
    blur: function() {
        return this._input.delMod("focused"), this
    },
    getControl: function() {
        return this._input.elem("control")
    },
    val: function() {
        return this._input.val.apply(this._input, arguments)
    },
    valWithoutSuggest: function(t, e) {
        return this._input.val.call(this._input, t, $.extend(e, {
            suggest: !1
        }))
    },
    realVal: function(t) {
        return void 0 === t ? this._realVal : ("string" == typeof t && (this._realVal = t), this._realVal)
    },
    width: function() {
        return this._input.domElem.width()
    },
    getCaretPosition: function() {
        var t, e = this.getControl()[0], n = this.val().length;
        return document.selection ? (t = document.selection.createRange(), t.moveStart("character", - n), t.text.length) : e.selectionStart || 0 === e.selectionStart ? e.selectionStart : n
    },
    moveCaretToEnd: function() {
        var t = this.getControl()[0], e = t.value.length;
        if ("number" == typeof t.selectionStart)
            t.scrollLeft = 999999, t.setSelectionRange(e, e);
        else if ("undefined" != typeof t.createTextRange) {
            t.focus();
            var n = t.createTextRange();
            n.collapse(!1), n.select()
        }
        return this
    },
    disableBrowserAutocomplete: function() {
        var t = this._input.hasMod("focused");
        t && this._input.delMod("focused"), this.getControl().attr({
            autocomplete: "off",
            autocorrect: "off",
            autocapitalize: "off",
            spellcheck: "false",
            "aria-autocomplete": "list"
        }), t && this._input.setMod("focused", "yes")
    },
    bindEvents: function(t, e) {
        return Object.keys(t).forEach(function(n) {
            this._eventsFilter(n).on(n, t[n].bind(e || this))
        }, this), this
    },
    bindSuggestEvents: function() {
        var t = this.params.suggest;
        return t.on({
            out: function(t, e) {
                this.a11yDeactive(), this._changeValueByEvent(t, e)
            }
        }, this), BEM.blocks["suggest2-item"].on(t.domElem, {
            tpah: function(t, e) {
                this._changeValueByEvent(t, e), this.moveCaretToEnd()
            },
            select: function(t, e) {
                this._changeValueByEvent(t, e), this.moveCaretToEnd()
            },
            enter: function(t, e) {
                this.a11yActive(e.id), e.byKeyboard && this.params.updateOnEnterByKeyboard && this._changeValueByEvent(t, e)
            }
        }, this), this
    },
    a11yActive: function(t) {
        this.getControl().attr("aria-activedescendant", t)
    },
    a11yDeactive: function() {
        this.getControl().removeAttr("aria-activedescendant")
    },
    _eventsFilter: function(t) {
        return ~["blur", "focus", "change", "clear"].indexOf(t) ? this._input : this.getControl()
    },
    _changeValueByEvent: function(t, e) {
        this.val(e && e.val || this.realVal(), $.extend({
            block: t.block.__self.getName(),
            event: t.type
        }, e))
    }
}), BEM.decl("suggest2-input", {
    onSetMod: {
        js: {
            inited: function() {
                this.__base.apply(this, arguments), this.bindEvents({
                    keydown: function(t) {
                        t.which === this.params.suggest.__self.keyboard.enter && this.trigger("enter")
                    }
                }, this)
            }
        }
    }
}), BEM.DOM.decl("suggest2-item", {
    val: function() {
        if (this.params.val)
            return this.params.val;
        var t = this.elem("text");
        return (t.length ? t : this.domElem).text()
    },
    onEnter: function(t) {
        return this.setMod("selected", "yes").triggerEvent("enter", {
            val: this.val(),
            byKeyboard: t
        })
    },
    onLeave: function() {
        return this.delMod("selected")
    },
    onSelect: function(t) {
        return this.triggerEvent("select", {
            val: this.val(),
            byKeyboard: t
        })
    },
    triggerEvent: function(t, e) {
        return this.trigger(t, {
            val: e.val,
            id: this.domElem.attr("id"),
            show: !1,
            byKeyboard: e.byKeyboard
        })
    }
}), BEM.DOM.decl("suggest2-item", {}, {
    live: function() {
        this.liveBindTo("mouseover mouseout mousedown leftclick", function(t) {
            this.trigger(t.type, t)
        })
    }
}), BEM.DOM.decl({
    block: "suggest2",
    modName: "theme",
    modVal: "normal"
}, {
    getPopupWidth: function() {
        return this._input.width()
    }
}), BEM.DOM.decl({
    block: "suggest2",
    modName: "theme",
    modVal: "normal"
}, {
    getPopupWidth: function() {
        var t = this._input.width();
        return $("html").hasClass("ios-simplified") ? t : t - 2
    }
}), BEM.DOM.decl({
    block: "suggest2",
    modName: "ahead",
    modVal: "yes"
}, {
    init: function() {
        var t = this.__base.apply(this, arguments), e = this;
        return e._model.on("response", function(t, n) {
            var i = n.data, o = i.items && i.items[0], s = e._input.val(), r = "object" == typeof o ? o[1]: o, a = "string" == typeof r && r.match("^" + BEM.blocks["i-common__string"].escapeRegExp(s.toLowerCase()) + "(.*)$");
            i.val = n.val, e.params.aheadFromServer ? e._input.get().setAhead(s, i.meta && i.meta["continue"] || "", i) : a && s.length >= e.params.showOverChars ? e._input.get().setAhead(s, e.params.showFullPhrase ? a[1] : a[1].split(" ")[0], {
                val: n.val
            }) : e._input.get().unsetAhead()
        }), e._input.on("change", function() {
            e._input.get().unsetAhead()
        }), e._input.get().onPopupShow && e._popup.on("show", function() {
            e._input.get().onPopupShow()
        }), e._popup.on("hide", function() {
            e._input.get().onPopupHide()
        }), t
    },
    getDefaultParams: function() {
        return $.extend(this.__base(), {
            aheadFromServer: !1,
            showOverChars: 3,
            showFullPhrase: !1
        })
    }
}), BEM.DOM.decl({
    block: "suggest2",
    modName: "adaptive",
    modVal: "yes"
}, {
    init: function() {
        this.__base.apply(this, arguments), this._winResizeCallback = function() {
            this._popup.width(this.getPopupWidth())
        }.bind(this), this.__self._bindToWinResize(this._winResizeCallback)
    },
    destruct: function() {
        this.__base.apply(this, arguments), BEM.DOM.blocks.suggest2._unbindFromWinResize(this._winResizeCallback)
    }
}, {
    _resizeCallbacks: [],
    _bindToWinResize: function(t) {
        return this._resizeCallbacks.push(t), this
    },
    _unbindFromWinResize: function(t) {
        for (var e = 0; e < this._resizeCallbacks.length; e++)
            if (this._resizeCallbacks[e] === t)
                return this._resizeCallbacks.splice(e, 1), this;
        return this
    },
    _getResizeCallback: function() {
        return $.throttle(function() {
            var t = arguments;
            this._resizeCallbacks.forEach(function(e) {
                e.apply(this, t)
            }, this)
        }, 300, this)
    },
    live: function() {
        this.__base.apply(this, arguments), this._resizeCallback = this._getResizeCallback(), BEM.DOM.win.on("resize", this._resizeCallback)
    }
}), BEM.DOM.decl("suggest2-form", {
    onSetMod: {
        js: {
            inited: function() {
                this._isReady=!1, this.initNodes()
            }
        }
    },
    getDefaultParams: function() {
        return {
            inputName: "input",
            buttonName: "button",
            popupName: "popup"
        }
    },
    getOwner: function() {},
    isReady: function() {
        return this._isReady
    },
    initNodes: function() {
        var t = this.getNodes();
        t.forEach(function(t) {
            t.suggest.init()
        }), this._isReady=!0, this.trigger("ready", {
            nodes: t
        })
    },
    getNodes: function() {
        return this.getNodeList().map(function(t) {
            var e = t.suggest, n = e.params;
            return n.input = t.input, n.button = t.button, n.form = this, n["suggest2-popup"] = BEM.create("suggest2-popup", {
                suggest: e,
                popup: e.findBlockOn(this.params.popupName),
                owner: this.getOwner(t) || n.input
            }), n["suggest2-input"] = BEM.create("suggest2-input", {
                suggest: e,
                input: n.input,
                updateOnEnterByKeyboard: n.updateOnEnterByKeyboard
            }), t.counter && (t.counter.params.suggest = e, t.counter.params.form = this, n.counter = t.counter), this.bindFormEvents(t), t
        }, this)
    },
    bindFormEvents: function(t) {
        var e = this;
        return t.button && t.button.bindTo("click", function(t) {
            e.trigger("button-click", t)
        }), e.bindTo("submit", function(t, n) {
            e.onSubmit(t, n)
        }), e
    },
    submit: function(t) {
        return this.domElem.trigger("submit", t || {}), this
    },
    onSubmit: function(t, e) {
        t.data = e || {}, this.trigger("submit", t)
    },
    getNodeList: function() {
        var t = this, e = [];
        return t.elem("node").each(function(n, i) {
            var o = $(i), s = t.findBlockInside(o, "suggest2"), r = {
                input: t.findElem(o, "input"),
                button: t.findElem(o, "button")
            };
            e.push({
                node: o,
                suggest: s,
                counter: t.findBlockOn("suggest2-counter"),
                input: t.findBlockOn(r.input, t.params.inputName),
                button: t.findBlockOn(r.button, t.params.buttonName),
                popup: s.findBlockOn(t.params.popupName)
            })
        }), e
    }
}), BEM.DOM.decl("suggest2-form", {
    onSetMod: {
        js: function() {
            this.__base(), this._initCounters()
        }
    },
    _prepareCounterStats: function(t) {
        var e = [];
        try {
            e.push(t.since_last_change), e.push(t.since_first_change), e.push(t.user_input.length), e.push(t.ratio.actionsCount), e.push(t.path.index), e.push("button_by_mouse" === t.path.submit ? "1" : "0"), e.push("edit" === t.path.state ? "1" : "0")
        } catch (n) {
            return ""
        }
        return e.join(",")
    },
    _bindCounters: function() {
        var t, e = this, n = e.findBlockInside("suggest2"), i = n.getCounter();
        i && e.params.statsExp===!0 && i.on("submit", function(n, i) {
            var o;
            t && t.remove(), o = e._prepareCounterStats(i), o && (t = $("<input>", {
                type: "hidden",
                name: "csg",
                value: o
            }), e.domElem.append(t))
        })
    },
    _initCounters: function() {
        this.isReady() ? this._bindCounters() : this.on("ready", this._bindCounters)
    }
}), BEM.DOM.decl("suggest2-counter", {
    onSetMod: {
        js: {
            inited: function() {
                this.clear()
            }
        }
    },
    getDefaultParams: function() {
        return {
            submitBySelect: !1,
            preventSubmit: !0,
            host: "//yandex.ru",
            path: "clck/jclck",
            dataUrl: "http://ya.ru",
            params: {},
            timeout: 300
        }
    },
    metaProps: {
        exprt: "exprt",
        r: "region",
        log: "log"
    },
    clear: function() {
        var t = this;
        return t._params = {
            user_input: "",
            text: ""
        }, t._times = [], t._path = {
            service: t.params.service,
            state: "not_shown",
            index: 0,
            personal: "nah_not_shown",
            submit: ""
        }, t._ratio = {
            len: 0,
            queryLen: 0,
            actionsCount: 0
        }, t._firstChange = 0, t._lastChange = 0, t._sinceChange = {
            first: 0,
            last: 0
        }, t._selectedText = "", t._session = 0, t._meta = {}, t._afterSendAjaxCallback, t
    },
    submit: function(t) {
        var e = this, n = e.params.suggest.params["suggest2-input"];
        e._params.text = n.val(), e._params.pos = n.getCaretPosition(), e._selectedText && e._selectedText !== e._params.text && (e._path.state = "edit"), e.sendAjax(e.getUrl(), $.extend({
            submitForm: !0
        }, t)), this.trigger("submit", this._getSubmitEventData()).clear(), this._params.prev_query = n.val()
    },
    _getSubmitEventData: function() {
        var t = this;
        return {
            path: t._path,
            times: t._times,
            prev_query: t._params.prev_query,
            user_input: t._params.user_input,
            text: t._params.text,
            pos: t._params.pos,
            ratio: t._ratio,
            since_first_change: t._sinceChange.first,
            since_last_change: t._sinceChange.last,
            session: t._session,
            meta: t._meta,
            url: t.getUrl()
        }
    },
    sendAjax: function(t, e) {
        return $.ajax({
            type: "GET",
            url: t,
            dataType: "script",
            timeout: this.params.timeout,
            cache: !0
        }).always(function() {
            this.params.preventSubmit && e.submitForm && this.params.form.submit({
                block: this.__self.getName(),
                prevent: !1
            }), this._afterSendAjaxCallback && this._afterSendAjaxCallback()
        }.bind(this)), this
    },
    afterSendAjax: function(t) {
        return this._afterSendAjaxCallback = t, this
    },
    getUrl: function() {
        return this.params.host + ["/", this.params.path].concat(this._getUrlParams()).join("/").replace(/(\/+)/g, "/")
    },
    _getUrlParams: function() {
        var t = this;
        return this._getParamsList().concat(t._getData()).reduce(function(e, n) {
            return e.concat(t._getParams(n))
        }, []).concat(["/"])
    },
    _getParamsList: function() {
        var t = this;
        return [t.params.params, t._getPath(), t._params, t._getRatio(), t._getSinceChange(), t._getSession()]
    },
    initValues: function() {
        return this._params.prev_query = this.params.suggest.params["suggest2-input"].val(), this
    },
    bindEvents: function() {
        var t, e = this, n = e.params, i = n.suggest;
        return n.form.on("button-click", function() {
            "keyboard" !== e._path.submit && (e._path.submit = "button_by_mouse")
        }), n.form.on("submit", function(t, i) {
            i.data.block !== e.__self.getName() && i.data.prevent!==!1 && (n.preventSubmit && (i.stopImmediatePropagation(), i.preventDefault()), e.submit())
        }), i.getModel().on({
            request: function() {
                t = new Date
            },
            response: function(n, i) {
                e._path.personal = e._hasPersonal(i.data.items) ? "nah_not_used" : "nah_not_shown", e._meta = i.data.meta || {}, e._addTime(new Date - t), e._setMetaProps(i)
            }
        }), i.params["suggest2-input"].on({
            change: function(t, n) {
                n && "clear" !== n.source || (e._params.user_input = t.block.val(), e[e._firstChange ? "_lastChange": "_firstChange"] = new Date), e._ratio.actionsCount++
            },
            enter: function() {
                e._path.submit = "keyboard"
            }
        }), i.params["suggest2-popup"].on({
            show: function() {
                e._path.state = "not_used"
            },
            hide: function(t, n) {
                "empty" === n.response && (e._path.state = "not_shown")
            }
        }), BEM.blocks["suggest2-item"].on(i.domElem, {
            enter: function(t, n) {
                n.byKeyboard && e.params.suggest.params.updateOnEnterByKeyboard && (e._selectedText = n.val, e._path.state = "suggest")
            }
        }), i.on({
            select: function(t, i) {
                e._path.state = i.byKeyboard ? "keyboard" : "mouse", e._path.index = i.itemIndex + 1, e._selectedText = i.val, i.item.hasMod("personal") && (e._path.personal = "nah_used"), n.submitBySelect && (e._path.submit = i.byKeyboard ? "keyboard" : "click_by_mouse", e._params.pos = i.pos, e._params.text = e._selectedText)
            }
        }, e), e
    },
    _hasPersonal: function(t) {
        return t.some(function(t) {
            return 1 === BEM.blocks["suggest2-view"]._getPrefs(t).pers
        })
    },
    _setMetaProps: function(t) {
        t.data.meta && Object.keys(this.metaProps).forEach(function(e) {
            this._params[this.metaProps[e]] = t.data.meta[e] || ""
        }, this)
    },
    _getParams: function(t) {
        return Object.keys(t).reduce(function(e, n) {
            return "" === t[n] ? e : (e.push(n + "=" + encodeURIComponent(t[n]).replace(/%5B/g, "[").replace(/%5D/g, "]").replace(/%2C/g, ",")), e)
        }, [])
    },
    _getPath: function() {
        return {
            path: [this._path.service, this._path.state, "p" + this._path.index, this._path.personal, this._path.submit].join(".")
        }
    },
    _getSession: function() {
        return this._session = (new Date).getTime() + Math.round(1e4 * Math.random()), {
            session: this._session
        }
    },
    _getSinceChange: function() {
        return {
            since_first_change: this._getSinceTime("first"),
            since_last_change: this._getSinceTime("last")
        }
    },
    _getSinceTime: function(t) {
        var e = "_" + t + "Change";
        return this._sinceChange[t] = this[e] ? (new Date).getTime() - this[e] : 0, this._sinceChange[t]
    },
    _getRatio: function() {
        return this._ratio.len = this._params.user_input.length, this._ratio.queryLen = this._params.text.length, {
            ratio: [this._ratio.len, this._ratio.queryLen, this._ratio.actionsCount].join(".")
        }
    },
    _addTime: function(t) {
        return this._times.push(t), this._params.times = this._times.join("."), this
    },
    _getData: function() {
        return {
            "*data": "url=" + this.params.dataUrl
        }
    }
}), BEM.DOM.decl({
    block: "suggest2-item",
    modName: "interact",
    modVal: "link"
}, {
    val: function() {
        return !1
    },
    onSelect: function(t) {
        return t && (window.location = this.elem("link").attr("href")), this.__base.apply(this, arguments)
    }
}), BEM.DOM.decl("arrow", {
    onSetMod: {
        js: function() {
            var t = this;
            this.__base(), this.input = this.findBlockInside("input"), this.suggest = this.findBlockInside("suggest2");
            var e = this.findBlockInside("input__sample");
            e && e.bindTo("click", function() {
                var e = t.domElem.find("input[name='nl']");
                t.params.nl&&!e.length && (e = $('<input type="hidden" name="nl" value="1"/>').insertAfter(t.input.domElem))
            }), this._initPrefetch()
        }
    },
    _getParams: function() {
        var t = this.domElem.find("input"), e = {};
        return t.each(function() {
            var t = $(this), n = t.prop("name");
            n && (e[t.prop("name")] = t.val())
        }), e.prefetch = 1, e
    },
    _serializeFormParams: function(t, e) {
        var n = [];
        t.text = e;
        for (var i in t)
            t.hasOwnProperty(i) && n.push(encodeURIComponent(i) + "=" + encodeURIComponent(t[i]));
        return "?" + n.join("&")
    },
    _initPrefetch: function() {
        var t = this, e=!1;
        t._prefetchStarted = {}, t._prefetchLoaded = {}, window.sessionStorage && t.params.prefetch && (this.domElem.on("submit", function() {
            var n = t.input.val();
            t._prefetchStarted[n] && (e || (e=!0, t.domElem.append('<input type="hidden" name="prefetch" value="1">'), t._prefetchLoaded[n] || t.domElem.append('<input type="hidden" name="unfinished" value="1">'))), window.sessionStorage.setItem("x", (new Date).getTime())
        }), "etalon" !== t.params.prefetch && this.suggest.on("model-ready", function() {
            t.suggest.getModel().on("response", function(e, n) {
                var i, o, s, r = n && n.data, a = r && r.meta && "yes" === r.meta.instant;
                if (a && r.items.length) {
                    if (i = "ahead" === t.params.prefetch ? r.meta && r.orig && r.meta["continue"] && r.orig + r.meta["continue"] : r.items[0]instanceof Array ? r.items[0][1] : r.items[0], !i)
                        return;
                    o = t._getParams(), t._prefetchStarted[i] || (t._prefetchStarted[i]=!0, s = new Image, s.src = t.domElem.attr("action") + t._serializeFormParams(o, i), s.onload = s.onerror = function() {
                        t._prefetchLoaded[i]=!0
                    })
                }
            })
        }))
    }
}), $(function() {
    "use strict";
    function t() {
        $(window).unbind("resize.b-twocolumns").bind("resize.b-twocolumns", t);
        var e, n = s.children(":visible").length - r.children(":visible").length;
        if ( - 1 > n)
            for (e = 0; - n > e; e += 2)
                s.append(r.children().first());
        else if (n > 1)
            for (e = 0; n > e; e += 2)
                r.prepend(s.children().last())
    }
    var e = $("<div></div>")[0], n = ["MozColumnCount", "columnCount", "WebkitColumnCount"].some(function(t) {
        return t in e.style
    });
    if (!n) {
        var i = $(".b-twocolumn"), o = i.not(".b-morda-search_nojs .b-twocolumn").children(), s = $('<div class="b-twocolumn__left"></div>'), r = $('<div class="b-twocolumn__right"></div>'), a = Math.ceil(o.length / 2);
        o.each(function(t) {
            (a > t ? s : r).append(this)
        }), i.append(s).append(r), $(window).bind("balance.b-twocolumns", t), $(window).bind("unbalance.b-twocolumns", function() {
            $(window).unbind("resize.b-twocolumns")
        })
    }
}), function(t, e) {
    BEM.DOM.decl("b-head-tabs", {
        onSetMod: {
            js: function() {
                e.block["b-head-tabs"].call(this, this.params)
            }
        }
    });
    var n = e.block["b-head-tabs"] = function(e) {
        function i(e, n) {
            var i = p[e] || {};
            return !i.clid && n && (i.clid = n), t.each(_[e] || [], function() {
                h[this] && (i[this] = h[this])
            }), i
        }
        function o(e, n) {
            var i = [].concat(g[e]), o = e + d;
            ("imageswww" == o || "wwwimages" == o) && i.push("site");
            var s = m[e] || {};
            return t.each(i, function() {
                h[this] && (s[this] = h[this])
            }), !s.clid && n && "www" !== e && (s.clid = n), s
        }
        function s(e) {
            var n = {};
            return e ? (t.each(e.substr(1).split("&"), function() {
                var e = this.split("="), i = e[0], o = e[1];
                if (i)
                    if (n.hasOwnProperty(i)) {
                        var s = n[i];
                        t.isArray(s) ? s.push(o) : n[i] = [s, o]
                    } else 
                        n[i] = o
            }), void 0 !== t.decodeURIComponent ? r(n) : n) : n
        }
        function r(e) {
            var n = ["site"];
            return t.each(e, function(i, o) {
                t.inArray(i, n)>-1 && (e[i] = t.isArray(o) ? o.map(function(e) {
                    return t.decodeURIComponent(e)
                }) : t.decodeURIComponent(o))
            }), e
        }
        var a, c = this.findBlockOutside("b-page").findBlockInside("b-head-search"), u = c && c.findBlockInside("b-search"), l = u && u.findBlockInside("b-form-input"), d = BEM.blocks["i-global"].param("id"), h = function() {
            var e = s(window.location.search), n = /^#.*(\?.+)$/.exec(window.location.hash);
            return n && t.extend(e, s(n[1])), e
        }(), f = {
            www: 505,
            search: 521,
            images: 526
        }
        [d] || 527, p = {
            market: {
                cvredirect: 2,
                clid: f
            }
        }, m = {
            news: {
                rpt: "nnews2",
                grhow: "clutop"
            },
            haber: {
                rpt: "nnews2",
                grhow: "clutop"
            },
            yaca: {
                rpt: "rs2"
            },
            market: {
                cvredirect: 2,
                clid: f
            },
            blogs: {
                ft: "blog"
            },
            people: {
                filter: "people"
            },
            appsearch: {
                filter: "mobile_apps"
            }
        }, _ = function() {
            var e = {};
            return t.each(["news", "yaca", "market", "blogs", "maps", "adresa", "video", "fotki", "slovari", "auto", "images", "music", "www", "aile", "yandex", "haber", "gorsel"], function() {
                e[this] = ["family"]
            }), e
        }(), g = function() {
            var e = {};
            return t.each(["news", "yaca", "market", "blogs", "maps", "adresa", "video", "fotki", "slovari", "auto", "images", "music", "www", "aile", "yandex", "haber", "gorsel"], function() {
                e[this] = ["noreask", "family"]
            }), e
        }(), b = this.domElem.find(':not(strong)>a.b-head-tabs__link:not([href$="/all"])').map(function() {
            var e = this.host.split(".")[0], r = s(this.search), a = r.clid, c = decodeURIComponent(t.param(t.extend(r, i(e, a)), !0)), u = this.pathname.replace(/^\/?(.*)/, "/$1"), l = n.getHostToUrlMap()[e];
            return l && u.indexOf(("/" === l.charAt(0) ? "" : "/") + l)>-1 && (l = ""), u = u.slice(1), u.length && (u = u.replace(/\/*$/, "/")), {
                node: this,
                clid: a,
                toServiceId: e,
                path: l,
                pathname: u,
                host: this.host,
                searchHost: n.getHostMap()[e],
                noSearchParamsUrl: c ? "?" + c: "",
                searchParams: t.extend(r, o(e, a))
            }
        }).filter(function() {
            return "undefined" != typeof this.path
        }), v = u && u.elem("advanced")[0];
        if (v && v.pathname) {
            var y = s(v.search), w = v.pathname.replace(/^\/?(.*)/, "$1");
            y.text = "", b.push({
                node: v,
                clid: y.clid,
                toServiceId: d,
                path: "",
                pathname: w,
                host: v.host,
                noSearchParamsUrl: "?" + t.param(y, !0),
                searchParams: y
            })
        }
        !function() {
            var i = l && t.trim(l.val()) || e["default"];
            a != i && (a = i, t.each(b, function() {
                i && (this.searchParams[n.getSearchParamName(this.toServiceId)] = i), this.node.href = [this.node.protocol + "/", i && this.searchHost ? this.searchHost: this.host, this.pathname + (i ? this.path + "?" + t.param(this.searchParams, !0) : this.noSearchParamsUrl)].join("/")
            })), setTimeout(arguments.callee, 200)
        }()
    };
    t.extend(n, {
        getHostToUrlMap: function() {
            var t = {
                www: "yandsearch",
                market: "search.xml",
                maps: "",
                music: "#!/search",
                video: "search"
            };
            return t.translate = t.ceviri = t.harita = t.maps, t.blogs = t.adresa = t.fotki = t.slovari = t.auto = t.rabota = t.market, t.appsearch = t.people = t.news = t.yaca = t.images = t.gorsel = t.haber = t.yandex = t.www, t.afisha = t.pogoda = t.hava = t.video, t
        },
        getHostMap: function() {
            return {}
        },
        getSearchParamName: function(t) {
            switch (t) {
            case"pogoda":
            case"hava":
                return "request";
            default:
                return "text"
            }
        }
    })
}(jQuery, window.Lego), $(document).ready(function() {
    function t(t, e) {
        return t + (e.length ? ( - 1 !== t.indexOf("?") ? "&" : "?") + e : "")
    }
    function e(t) {
        return tabParams.searchParamName && tabParams.searchParamName[t] ? tabParams.searchParamName[t] : ""
    }
    function n() {
        var n = encodeURIComponent($.trim(i.val()));
        r && n === s || (r=!0, s = n, o.map(function() {
            var i, o, s, r = (this.host, $(this).attr("id").split("-")[1]), a = "", c = "", u = "";
            if (tabParams.url[r] && (i = tabParams.url[r]), tabParams.searchUrl[r] && (s = tabParams.searchUrl[r]), n && s ||!tabParams.noSearchDefParamsMap ||!tabParams.noSearchDefParamsMap[r] ? n && tabParams.searchDefParamsMap && tabParams.searchDefParamsMap[r] && (a = $.param(tabParams.searchDefParamsMap[r])) : a = $.param(tabParams.noSearchDefParamsMap[r]), c = $.param(tabParams.commonParams), u = [c, a].filter(function(t) {
                return t.length
            }).join("&"), s && n || (o = t(i, u)), s && n) {
                var l = e(r);
                o = t(s, u), o = t(o, l + "=" + n)
            }
            $(this).attr("href", o)
        }))
    }
    if (window.tabParams) {
        var i = $("#text"), o = $("a.b-head-tabs__tab-link:not(.b-link_pseudo_yes)");
        if (i.length || (i = tabParams.elemSearch), !o.length && tabParams.elemTabs && (o = tabParams.elemTabs), i.length && o.length) {
            var s = "", r=!1;
            setInterval(n, 200)
        }
    }
}), BEM.DOM.decl("b-head-tabs", {
    onSetMod: {
        js: function() {
            var t = this.findBlockInside("b-popupa");
            t && (t.on("show", function() {
                this.afterCurrentEvent(function() {
                    $(window).trigger("balance.b-twocolumns")
                })
            }), t.on("hide", function() {
                $(window).trigger("unbalance.b-twocolumns")
            }))
        }
    }
}), function() {
    "use strict";
    var t = 250;
    BEM.DOM.decl("hint", {
        onSetMod: {
            js: function() {
                this.bindTo("item", "mouseenter", this._onItemOver), this.bindTo("item", "mouseleave", this._onItemOut)
            }
        },
        _onItemOver: function(e) {
            var n = this, i = $(e.target).closest(".hint__item");
            clearTimeout(this._timer), this._timer = setTimeout(function() {
                n._showTip(i)
            }, t)
        },
        _onItemOut: function() {
            clearTimeout(this._timer), this._hideTip()
        },
        _showTip: function(t) {
            this._hideTip(), this._$target = t, this.setMod(this._$target, "hint", "visible")
        },
        _hideTip: function() {
            this._$target && this._$target.length && (this.delMod(this._$target, "hint"), this._$target = null)
        }
    })
}(), BEM.DOM.decl("b-traffic-more", {
    onSetMod: {
        js: function() {
            var t = this;
            $(window).on("probkiRefreshState", function() {
                t.elem("link").find(".b-link__inner").html(t.params.altLinkText), t.delMod("visible")
            }), this.bindTo("link", "click", function(e) {
                t._showBlock(e)
            })
        }
    },
    _showBlock: function() {
        if (this.setMod("shown", "yes"), this.params.saveUrl) {
            var t = $(".b-traffic-more__frame");
            t.length || (t = $('<iframe class="b-traffic-more__frame" style="display:none"></iframe>'), $(document.body).append(t)), t.attr("src", this.params.saveUrl)
        }
    }
}), BEM.DOM.decl("b-traffic-jamspromo", {
    onSetMod: {
        js: function() {
            home.localStorage.getItem("b-traffic-jamspromo") || this.domElem.removeClass("i-hidden");
            try {
                location.hash.indexOf("open=_traffic")>-1 && home.localStorage.setItem("b-traffic-jamspromo", !0)
            } catch (t) {}
        }
    }
});
var AwapsJsonAPI = {
    Json: {}
};
AwapsJsonAPI.Json = function(t, e, n, i, o, s, r, a) {
    "undefined" == typeof a && (a = ""), a && a.length > 2 && a.indexOf(":") < 0 && (a += ":"), this.w = e, this.h = n;
    for (var c = String(n); c.length < 4;)
        c = "0" + c;
    for (c = e + c; c.length < 8;)
        c = "0" + c;
    o || (o = "awaps.yandex.ru"), i = i ? "&charset=" + i : "", s = s ? "&subsection=" + s : "", this.object_name = r ? r : "aw_json";
    var u = Math.round(65535 * Math.random());
    this.json_awaps_url = a + "//" + o + "/10/" + t + "/" + c + ".?charset=" + i + "&callback=" + this.object_name + ".awaps_callback&rnd=" + u + i + s, this.gif_code = '<a href="' + a + "//" + o + "/1/" + t + "/" + c + '.gif" target="_blank"  ><img src="' + a + "//" + o + "/0/" + t + "/" + c + ".gif?rnd=" + u + '" width="' + e + '" height="' + n + '" border="0" ></A>', this.awaps_div_code = '<div id="awaps_ad_div" width="' + e + '" height="' + n + '"></div>'
}, AwapsJsonAPI.Json.prototype.setID = function(t) {
    t && (this.html_cont_id = document.getElementById(t))
}, AwapsJsonAPI.Json.prototype.refreshAd = function() {
    this.callJson(this.json_awaps_url)
}, AwapsJsonAPI.Json.prototype.callJson = function(t) {
    if (t) {
        var e = document.getElementsByTagName("head")[0], n = document.createElement("script");
        n.type = "text/javascript", n.src = t, e.appendChild(n)
    }
}, AwapsJsonAPI.Json.prototype.showAwapsJson = function(t) {
    var e = t.media[0], n = this.getHTML(e);
    this.html_cont_id && (this.html_cont_id.innerHTML = n)
}, AwapsJsonAPI.Json.prototype.awaps_callback = function(t) {
    var e = t.media, n = e && e[0] && (e[0].bnImg || e[0].bnHtm);
    n && this.showAwapsJson(t)
}, AwapsJsonAPI.Json.prototype.getHTML = function(t) {
    var e = Math.round(65535 * Math.random());
    if (t.bnCount) {
        this.im = new Image;
        var n = String(t.bnCount);
        n = n.replace("%aw_random%", e), this.im.src = n
    }
    var i = new CAWBrowser, o = "", s = location.protocol || "http:";
    return t.bnHtm ? t.bnHtm : (i.flash >= t.flashV && t.bnFlash ? (s && 0 === s.toLowerCase().indexOf("https") && (t.bnFlash += "&cache=1"), o = "<object classid=clsid:D27CDB6E-AE6D-11cf-96B8-444553540000 codebase=" + s + "//download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab width=" + t.bnW + " height=" + t.bnH + ' ><param name="allowScriptAccess" value="Always" ><param name=movie value="' + t.bnFlash + '" ><param name=menu value=false><param name=quality value=high><param name=wmode value=opaque><param name="flashvars" value="' + t.flashvars + '" ><EMBED src="' + t.bnFlash + '"  flashvars="' + t.flashvars + '" quality=high  allowScriptAccess=Always wmode=opaque menu=false swLiveConnect=FALSE WIDTH=' + t.bnW + " HEIGHT=" + t.bnH + ' TYPE="application/x-shockwave-flash" PLUGINSPAGE="' + s + '//www.macromedia.com/shockwave/download/index.cgi?P1_Prod_Version=ShockwaveFlash"></EMBED> </object>') : (o = '<img src="' + t.bnImg + '" width="' + t.bnW + '" height="' + t.bnH + '" border=0 alt="' + t.bnAlt + '" >', t.gif_tizer || (o = '<a href="' + t.bnHref + '" target=_blank >' + o + "</a>")), o)
}, function() {
    "use strict";
    var t = (new Date).getTime(), e = function() {};
    window.home = window.home || {}, home.leftBanner = {};
    var n = home.leftBanner;
    $(function(i) {
        var o = home.getData("b-banner-left.data");
        if (o) {
            o.timestamp = t;
            var s, r, a, c, u = i(".b-page"), l = i(document.body), d = i(window), h = o.anim_duration, f=!1, p=!1, m = "two-slide" === o.mode, _ = "locked" === o.mode, g = "#leftBanner", b = 0, v = [], y = o.flash_ver, w = window.innerHeight, k = function(t) {
                var e = {};
                return {
                    isAnimating: function() {
                        return e.animation
                    },
                    isOpened: function() {
                        return t.is(".left-banner-opened")
                    },
                    isClosed: function() {
                        return !t.is(".left-banner-opened")
                    },
                    isHidden: function() {
                        return Number(Boolean(i("div#leftBanner").length || 0 !== parseInt(i(".b-banner-left").css("top")) || i(".b-banner-left:not(:visible)").length || i(".b-banner-left.hideBanner").length))
                    },
                    setClosed: function() {
                        t.removeClass("left-banner-opened")
                    },
                    setOpened: function() {
                        t.addClass("left-banner-opened")
                    },
                    setAnimated: function(n) {
                        e.animation=!!n, e.animation ? t.addClass("left-banner-animating") : t.removeClass("left-banner-animating")
                    },
                    setMinimized: function(t) {
                        e.minimized=!!t
                    },
                    setInited: function(t) {
                        e.initer = t
                    },
                    isInited: function() {
                        return !!e.initer
                    }
                }
            }(l);
            window.stateManager = k, window.AwapsRM = {
                Init: function() {
                    m ? setTimeout(function() {
                        B()
                    }, 50) : _ && B()
                },
                ClickSmall: function() {
                    "locked" === o.mode ? T() : F(!1)
                },
                ClickClose: home.debounce(function() {
                    H(!0, !1)
                }, 100, !0),
                ClickBig: function() {
                    H(!0, !1)
                },
                Error: function(t) {
                    E(), e("AwapsRM.Error(" + t + ")")
                }
            };
            var E = function() {
                var t=!1;
                if (o.canShowImage) {
                    o.isFlashDisabled=!0, i("#leftBanner").remove();
                    var e = i("<a>").attr({
                        id: "left-banner-fallback",
                        alt: o.image_alt || "",
                        href: o.click_url,
                        target: "_blank"
                    }).css({
                        backgroundColor: o.image_background_color,
                        visibility: "hidden"
                    }), s = i("<img>").attr({
                        id: "left-banner-fallback-image",
                        width: o.image_width,
                        height: o.image_height,
                        src: o.image_show_url
                    }).on("load", function() {
                        e.css("visibility", "visible");
                        var n = "none" === e.find("img").css("display");
                        n ? e.remove() : t=!0
                    }).on("error", function() {
                        e.remove()
                    });
                    e.append(s);
                    {
                        e.css("width")
                    }
                    e.css("width", 0), e.appendTo(i(".b-banner-left"));
                    var a = setInterval(function() {
                        t&&!I() && (clearInterval(a), e.css("width", 100), i(".left-banner-fallback-placeholder").replaceWith(e), r = i("#left-banner-fallback"), r.show(), n.showBanner("initialization"), $(), i(".b-banner-left").css("width", ""))
                    }, 500)
                }
            }, M = function(t, n, i) {
                if (i = Boolean(i), !m ||!k.isInited())
                    return !1;
                if (t = t || D(), n = n || P(), c && c.setSize) {
                    try {
                        var o = O();
                        c.setSize(parseInt(t * o), parseInt(n * o), i)
                    } catch (s) {
                        e(s + "; setSize call was failed.")
                    }
                    return {
                        width: t,
                        height: n
                    }
                }
                return null
            }, x = function(t) {
                if (m && c && c.closeBig && k.isInited())
                    try {
                        c.closeBig(t)
                } catch (n) {
                    e(n + "; closeBig call was failed.")
                }
            }, C = function() {
                var t = home.getData("b-banner-left.swfobject");
                return m && (t[6].big_flash_height = P(), t[6].big_flash_width = D()), t
            }, T = function() {
                var t = i(".b-banner-left")[0], e = t.innerHTML;
                t.innerHTML = "", t.innerHTML = e
            }, S = function() {
                n.hideBanner("initialization");
                var t = C(o);
                p=!o.isFlashDisabled && window.swfobject && window.swfobject.hasFlashPlayerVersion(y), p ? (swfobject.embedSWF.apply(swfobject, t), setTimeout(function() {
                    !k.isInited() && o.canShowImage && (e("flash init timeout, fallback to image"), E())
                }, 5e3), q()) : E(), o.noMediaQueries && i(".b-page-layout > div:not(.b-line_bar):not(.b-head-stripe):not(.b-skin-bg)").addClass("b-page-layout-movable"), $()
            }, B = function() {
                var t = i(".b-banner-left");
                k.setInited(!0), k.setMinimized(!0), k.setClosed(), t.addClass("b-banner-left__minimized"), t.removeClass("b-banner-left__image-mode"), setTimeout(function() {
                    i(".b-banner-left").addClass("b-banner-left__flash-mode")
                }, 1e3), n.showBanner("initialization"), m && (c = i(g)[0], i(window).on("resize", function() {
                    M(null, null, !0)
                }), u.add(i(".b-disk-uploader")).on("click.leftBanner", function(t, e) {
                    return e && e.verticalBannerIgnore ? void 0 : ($(), M(), k.isClosed() || k.isAnimating() || i(t.target).closest(".b-banner-left").length?!0 : (H(!0, !0), !0))
                }), i(document).keyup(function(t) {
                    27 === t.keyCode && k.isOpened()&&!k.isAnimating() && H(!0, !0)
                }), M()), d.on("resize", $), $()
            }, D = function() {
                return Math.max(o.bigBannerMinWidth, Math.min(parseInt(d.width() * (o.bannerToPageWidthFactorFull / 100)), o.bigBannerMaxWidth))
            }, I = function() {
                return o.pageWidthMin > parseInt(window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth) || o.allBannerMinHeight > P()
            }, O = function() {
                return 1
            }, P = function() {
                return parseInt(window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight)
            }, $ = home.debounce(function() {
                var t = I(), e = k.isHidden(), i = k.isOpened(), s=!i && window.innerHeight !== w;
                w = window.innerHeight, t ? i ? (n.hideBanner("resize.tooSmall"), k.setClosed(), q(), x(!0)) : n.hideBanner("resize.tooSmall") : e ? n.showBanner("resize.tooSmall") : i ? N() : M(null, null, s), o.noMediaQueries && (d.width() < 1265 ? l.addClass("width-lt-1265") : l.removeClass("width-lt-1265")), t || f ||!(f=!0) || (o.client_stat_url && ((new Image).src = o.client_stat_url), o.awaps_stat_url && ((new Image).src = o.awaps_stat_url))
            }, 100), j = function() {
                _ && i("#leftBanner").replaceWith(i('<div id="leftBanner">')), H(!1, !0), l.removeClass("b-banner-left-on"), i(".b-banner-left").addClass("hideBanner")
            }, A = function() {
                var t = i(".b-banner-left");
                _ ? t.is(".hideBanner") && S() : m && M(null, null, !0), t.removeClass("hideBanner"), l.addClass("b-banner-left-on")
            }, N = function(t) {
                var e = M();
                if (e) {
                    var n = e.width, s = e.height, r = i(o.bannerSelecor).width();
                    if ("#test" === location.hash) {
                        if (m) {
                            var a = a || D();
                            i(o.bannerSelecor).stop().animate({
                                width: a
                            })
                        }
                    } else 
                        r > n ? L(W, t?!1 : h, [n, s]) : W(n)
                }
            }, L = function(t, e, n) {
                0 === e || e > 0 ? setTimeout(function() {
                    t.apply(n)
                }, e) : t.apply(n)
            }, F = function() {
                o.show_big_stat_url && ((new Image).src = o.show_big_stat_url), W(), k.setAnimated(!0), k.setOpened(), clearTimeout(s), s = setTimeout(function() {
                    k.setAnimated(!1)
                }, h), U(), window.notificationPanel && window.notificationPanel.close(), l.trigger("clearScreen"), l.trigger("click", [{
                    verticalBannerIgnore: 1
                }
                ]), i(".b-topbar").bem("b-topbar")._hide()
            }, R = function() {
                o.close_big_stat_url && ((new Image).src = o.close_big_stat_url)
            }, H = function(t, e) {
                z(), t && R(), e && x(), clearTimeout(s), s = setTimeout(function() {
                    q()
                }, o.close_anim_duration || h)
            }, z = function() {
                k.setAnimated(!0), setTimeout(function() {
                    k.setAnimated(!1), k.setClosed()
                }, o.close_anim_duration || h)
            }, U = function() {
                k.setAnimated(!0), setTimeout(function() {
                    k.setAnimated(!1), k.setOpened()
                }, h)
            }, q = function() {
                var t = i(".b-banner-left");
                m&&!t.is(".b-banner-left__image-mode") && i(o.bannerSelecor).width(o.small_flash_width / O()), t.addClass("b-banner-left__minimized")
            }, W = function(t) {
                m && (t = t || D(), i(o.bannerSelecor).css({
                    width: t
                })), i(".b-banner-left").removeClass("b-banner-left__minimized")
            };
            n.hideBanner = function(t) {
                if (!t)
                    throw "hideBanner interface requires key param";
                return V(), - 1 !== v.indexOf(t)?!1 : (v.push(t), null === b && (b = 0), void(1===++b && j()))
            }, n.showBanner = function(t) {
                - 1 !== v.indexOf(t) && v.splice(v.indexOf(t), 1), null !== b&&--b <= 0&&!v.length && (b = null, a = setTimeout(function() {
                    A()
                }, 500))
            };
            {
                var V = function() {
                    a && (clearTimeout(a), a = 0)
                }, J=!0, K = 302, G = 0;
                "two-slide" !== home.getData("b-banner-left.data.mode")
            }
            if (window.performance && performance.timing && (G = performance.timing.responseEnd - performance.timing.responseStart, J = K > G), i.ajax({
                dataType: "script",
                cache: !0,
                url: o.swfobjectUrl
            }).done(function(t, e) {
                "error" === e ? E() : S()
            }).fail(function(t, n, i) {
                e("Failed to load swfobject: " + n + ", " + i), E()
            }), i(document).on("popupsClose.lego", function() {
                home.leftBanner.hideBanner("popupsClose.lego"), setTimeout(function() {
                    i(window).on("widgetSettingsClosed.banner-left", function() {
                        home.leftBanner.showBanner("popupsClose.lego"), i(window).off("widgetSettingsClosed.banner-left")
                    })
                }, 100)
            }), window.notificationPanel) {
                var Q = notificationPanel.open;
                notificationPanel.open = function() {
                    n.hideBanner("notification-panel"), Array.prototype.splice.call(arguments, 0, 1), Q.apply(notificationPanel, arguments)
                };
                var X = notificationPanel.close;
                notificationPanel.close = function() {
                    n.showBanner("notification-panel"), Array.prototype.splice.call(arguments, 0, 1), X.apply(notificationPanel, arguments)
                }
            }
        }
    })
}(), BEM.decl("i-statface", {
    onSetMod: {
        js: function() {
            this._data = {}, this._needSend=!1, this.hasMod("send", "manual") || $(window).unload(this.changeThis(this.send))
        }
    },
    set: function(t, e) {
        this._needSend=!0;
        var n = this._data;
        return "object" == typeof t ? $.each(t, function(t, e) {
            n[t] = e
        }) : n[t] = e, this
    },
    reset: function() {
        var t = this;
        return arguments[0] ? $.each(arguments, function(e, n) {
            delete t._data[n]
        }) : this._data = {}, $.isEmptyObject(this._data) && (this._needSend=!1), this
    },
    serialize: function() {
        var t = this;
        return $.map(t.params.keys, function(e) {
            return t._data[e]
        }).join(".") + (t.params.customKeys ? $.map(t.params.customKeys, function(e) {
            var n = t._data[e];
            return "/" + e + "=" + (void 0 == n ? "" : n)
        }).join("") : "")
    },
    send: function(t) {
        if (this._needSend) {
            var e = this.params, n = ["//", e.host, "/", e.path, "/dtype=stred", "/pid=", e.pid, "/cid=", e.cid, "/path=", this.serialize(), "/*data=", encodeURIComponent("url=" + encodeURIComponent(e.url))].join("");
            "click" == e.path ? document.createElement("IMG").src = n : $.ajax({
                type: "GET",
                url: n,
                data: null,
                complete: t || $.noop,
                dataType: "script",
                timeout: 500
            }), this._needSend=!1
        }
        return this
    },
    getDefaultParams: function() {
        return {
            host: "clck.yandex.ru",
            path: "jclck",
            url: location.href
        }
    }
}), function(t) {
    t = t || {}, t.oframebustMatchDomain = function(t, e) {
        t = "[object Array]" === Object.prototype.toString.call(t) ? t : function() {
            var e = [];
            for (var n in t)
                t.hasOwnProperty(n) && e.push(n);
            return e
        }();
        for (var n = 0, i = t.length; i > n; n++) {
            var o = t[n];
            if ("string" == typeof o) {
                if (/(\?|\*)/.test(o)) {
                    var s = o.replace(/\./g, "\\.").replace(/\*/g, ".*").replace(/\?/g, ".{1}");
                    if (new RegExp("^" + s + "$").test(e))
                        return !0
                } else if (e == o)
                    return !0
            } else 
                try {
                    if (o.test(e))
                        return !0
            } catch (r) {}
        }
    }
}(window.Lego), function(t) {
    t = t || {}, t.oframebust = function(e) {
        if (window.top.location != window.location) {
            var n = document.referrer.match(/^https?:\/\/([^:\/\s]+)\/?.*/);
            if (!n)
                return;
            !t.oframebustMatchDomain(e, n[1]) && (window.top.location = window.location)
        }
    }
}(window.Lego), BEM.DOM.decl({
    block: "b-head-stripe",
    modName: "type",
    modVal: "wide"
}, {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments), this.domElem.animate({
                "margin-top": 0
            }, 1e3);
            var t = this, e = setTimeout(function() {
                t._doCloseAnimation()
            }, 5e3);
            this.bindTo("mouseover", function() {
                clearTimeout(e)
            }), this.bindTo("mouseout", function() {
                e = setTimeout(function() {
                    t._doCloseAnimation()
                }, 1500)
            })
        }
    },
    _doCloseAnimation: function() {
        var t = this;
        this.domElem.animate({
            "margin-top": "-4em"
        }, 1e3, function() {
            t.setMod("state", "closed")
        })
    }
}), BEM.DOM.decl("b-head-universal", {
    onSetMod: {
        js: function() {
            "use strict";
            var t = this, e = t.params;
            if (e.animation_duration) {
                var n = e.animation_duration;
                t.domElem.css({
                    "-webkit-transition": "opacity " + n + "s",
                    transition: "opacity " + n + "s"
                }), setTimeout(function() {
                    t.setMod("state", "showed")
                }, 50)
            }
            e.auto_close && setTimeout(function() {
                t.domElem.hide()
            }, 1e3 * e.auto_close), t.bindTo("close", "leftclick", function(e) {
                t._closeCnt(e)
            }), t.bindTo("text1", "leftclick", function(e) {
                e.stopPropagation(), t._close()
            }), t.bindTo("text2", "leftclick", function(e) {
                e.stopPropagation(), t._close()
            }), t.bindTo("table", "leftclick", function() {
                t._close(), window.statRedirect(e.void_url, e.void_v12counter)
            }), e.background_color_file2 && ((new Image).src = e.background_color_file2, t.bindTo("text2", "mouseover", function() {
                t.elem("image").attr("src", e.background_color_file2)
            }), t.bindTo("text2", "mouseout", function() {
                t.elem("image").attr("src", e.background_image)
            })), e.agreement && (t.bindTo("text1", {
                mouseover: t._showAgreement,
                mouseout: t._hideAgreement
            }), t.bindTo("col_text2", {
                mouseover: t._showAgreement,
                mouseout: t._hideAgreement
            }), t.bindTo("agreement-link", "click", function(t) {
                t.stopPropagation()
            }))
        }
    },
    _showAgreement: function() {
        this.elem("agreement").addClass("b-head-universal__agreement_visible")
    },
    _hideAgreement: function() {
        this.elem("agreement").removeClass("b-head-universal__agreement_visible")
    },
    _closeCnt: function(t) {
        t.preventDefault(), t.stopPropagation(), this.params.close_v12counter && csh_ifmsid(this.params.close_v12counter), this.params.close_counter && ((new Image).src = this.params.close_counter), this._close()
    },
    _close: function() {
        this.setMod("state", "closed"), $(window).trigger("popupredraw.lego");
        var t = $(".popup_visibility_visible:not(.b-topbar__popup)");
        t.length && t.bem("popup").hide()
    }
}), BEM.DOM.decl("b-topbar", {
    onSetMod: {
        js: function() {
            var t = this;
            this.urls = BEM.DOM.extractParams(this.domElem[0])["b-topbar"], this._bindToggle(), this.popup = this._createPopup(), this._bindAutoclose(), this.has = $(".b-line_bar"), this._bindMailCount(), "yes" === this.getMod("expanded") ? (this.state = 1, window.opera ? setTimeout(function() {
                t._show()
            }, 50) : window.pl && window.pl.notloaded.css ? $(window).on("cssloaded.cacher", this.changeThis(function() {
                this._show()
            })) : this._show()) : this._notifyHasBlock()
        },
        expanded: {
            "": function() {
                this._notifyHasBlock(), this._switchCounters(this.domElem, ".big.", ".small."), this._switchCounters($(".b-popupa_overlapping_yes"), ".big.", ".small."), this._switchCounters($(".b-topbar__area"), ".big", ".small"), workerMessage("update", {
                    name: "domik",
                    visible: !1
                })
            },
            yes: function() {
                this._notifyHasBlock(), this._switchCounters(this.domElem, ".small.", ".big."), this._switchCounters($(".b-popupa_overlapping_yes"), ".small.", ".big."), this._switchCounters($(".b-topbar__area"), ".small", ".big"), workerMessage("update", {
                    name: "domik",
                    visible: !0
                })
            }
        }
    },
    _notifyHasBlock: function() {
        this.has.toggleClass("has_domik")
    },
    _switchCounters: function(t, e, n) {
        t.find(".b-link").each(function() {
            var t = $(this).attr("onmousedown");
            t && $(this).attr("onmousedown", t.replace(e, n))
        })
    },
    _createPopup: function() {
        var t = $(BEM.HTML.build({
            block: "popup",
            mods: {
                theme: "flat",
                behavior: "snap",
                autoclosable: "yes",
                direction: "down",
                animate: "yes"
            },
            mix: [{
                block: "b-topbar",
                elem: "popup"
            }
            ],
            js: {
                zIndex: - 31700
            },
            content: [{
                elem: "tail"
            }, {
                elem: "under"
            }, {
                elem: "content",
                mods: {
                    padded: "old"
                }
            }
            ]
        }));
        return t = this.findBlockOn(t, "popup"), t.elem("content").append($(".b-topbar__popup .popup__content > *")), t.on({
            "outside-click": function(t) {
                t.preventDefault()
            }
        }, this), t
    },
    _isPopupVisible: function(t) {
        return t.hasClass("popup_visibility_visible")
    },
    _bindAutoclose: function() {
        var t, e = this;
        $(window).bind("popupopen.lego", function() {
            clearTimeout(t), e.popup.hide()
        }).bind("popupclose.lego", function() {
            if (!e.popup.isShown() || e.popup.isHiding()) {
                var n = $(".header__action_type_settings.header__action_pressed_yes").length, i = $(".popup_langs_yes").hasClass("popup_visibility_visible"), o = $(".dropdown-menu__popup_extra_yes").hasClass("popup_visibility_visible");
                clearTimeout(t), t = setTimeout(function() {
                    var t = $(".popup_user_yes").hasClass("popup_visibility_visible");
                    !e.state || t && n || t && i || t && o || e._isPopupVisible($(".popup_langs_yes")) && i || e.popup.show(e._getSwitcher())
                }, 220 + (window.device ? 200 : 0))
            }
        }).bind("popupredraw.lego keyboardOpened.lego keyboardClosed", function() {
            e.state && e.popup.repaint()
        })
    },
    _bindMailCount: function() {
        var t = this;
        $(window).bind("popupmailcount.lego", function(e, n) {
            var i = n.count;
            t.popup.elem("content").find(".b-topbar__notifier_type_mail .b-link__inner").html("<b>" + (i ? i + " " : "") + ya.decline(i, n.words) + "</b>")
        })
    },
    _bindToggle: function() {
        for (var t = this.findBlocksInside($(".b-topbar__toggler"), "b-link"), e = 0; e < t.length; e++)
            t[e].on("click", this._toggle, this)
    },
    _show: function() {
        this.setMod("expanded", "yes"), this.trigger("expanded"), this._save(1);
        var t = this._getSwitcher();
        this.popup.show(t)
    },
    _hide: function() {
        this.delMod("expanded"), this._save(0), this.popup.hide()
    },
    _getSwitcher: function() {
        return this.switcher || (this.switcher = $($.browser.msie && $.browser.version < 9 ? $(".user__icon:first").length ? ".user__icon:first" : ".user__enter" : $(".user__name:first").length ? $(".user__name:first").width() > 30 ? ".user__name:first" : ".user__icon:first" : ".user__enter")), this.switcher
    },
    _toggle: function() {
        this.popup.isShown() ? this._hide() : this._show()
    },
    _save: function(t) {
        if (this.state !== t) {
            this.state = t;
            var e = $(".b-topbar__frame");
            e.length || (e = $('<iframe class="b-topbar__frame" style="display:none"></iframe>'), $(document.body).append(e)), e.attr("src", this.urls[t ? "open": "close"])
        }
    }
});
var notifyMailPulse = function() {
    var t, e, n = $(".b-topbar__notifier_type_mail");
    $(".b-ico-redmail", n).remove(), t = $("img.b-ico-mail", n), e = $("<span>").attr({
        "class": t.attr("class").replace("b-ico-mail", "b-ico-redmail")
    }), n.prepend(e), n.addClass("pulse-animation"), setTimeout(function() {
        n.removeClass("pulse-animation")
    }, 600)
}, notifyMailOld = function() {
    var t = $(".b-topbar__notifier_type_mail");
    t.addClass("exp2"), t.addClass("ring"), setTimeout(function() {
        t.removeClass("ring")
    }, 1600)
};
BEM.DOM.decl({
    block: "b-topbar",
    modName: "update",
    modVal: "auto"
}, {
    onSetMod: {
        js: function() {
            this.__base.apply(this, arguments), this.urls && this.urls.xivaUrl && (this.barMailCount = this.elem("notifier", "type", "mail").find(".b-link__inner"), this._mailCount = parseInt(this.barMailCount.text(), 10) || 0, this.connectAttempts = 0, this.connect(), this.words = this.urls.words || [])
        }
    },
    changeNumbers: function(t) {
        var e;
        null !== t && (this._mailCount = t), e = this._mailCount, this.barMailCount.html(e ? "<b>" + e + "</b>" : this.words[3] || e), $(window).trigger("popupmailcount.lego", {
            count: e,
            words: this.words
        }), home.getData("b-bar.mail-icon-exp") ? notifyMailPulse() : notifyMailOld()
    },
    _update: function(t) {
        "insert" === t.operation && this.changeNumbers(t.message.new_messages)
    },
    connect: function() {
        if (window.WebSocket) {
            var t = this, e = new WebSocket(this.urls.xivaUrl), n=!1;
            this.connectAttempts = 0, e.onopen = function() {
                n=!0, setTimeout(function() {
                    n && (t.connectAttempts = 0)
                }, 5e3)
            }, e.onmessage = function(e) {
                try {
                    var n = JSON.parse(e.data);
                    t._update(n)
                } catch (i) {}
            }, e.onclose = function(e) {
                t.connectAttempts++;
                var i = 6e4;
                n=!1, t.connectAttempts <= 5 ? i = 3e3 : t.connectAttempts <= 15 && (i = 1e4), 4e3 === e.code && (i = 3e5), setTimeout(function() {
                    t.connect()
                }, i)
            }
        }
    }
}), function() {
    var t = 24, e = 15, n = $.browser.opera && $.browser.version < 12.1 ? "keypress": "keydown";
    BEM.DOM.decl({
        block: "popup",
        modName: "behavior",
        modVal: "snap"
    }, {
        onSetMod: {
            visibility: {
                visible: function() {
                    this.__base(), this.hasMod("adaptive", "yes") || this._bindToResize()
                },
                "": function() {
                    this.__base(), this._unbindFromResize()
                }
            }
        },
        _calcLeft: function() {
            var n = this.getPopupSize(), i = this.getOwnerPos(), o = this.getOwnerSize(), s = i.left + o.width / 2 - t / 2, r = i.left + o.width, a = document.body.offsetWidth;
            if (a - e > r && a - e - s < n.width)
                return this.snap = "right", a - e - n.width;
            if (i.left > 0 && s < n.width + e)
                return this.snap = "left", e;
            this.snap=!1;
            var c = this.__base.apply(this, arguments), u = c + n.width - (document.body.scrollWidth - e);
            return u > 0 && (c -= u), c
        },
        _calcTop: function() {
            return this.__base.apply(this, arguments)
        },
        _show: function(t, e) {
            var n = {}, i = document.body.scrollWidth, o = document.body.scrollHeight, s = this.domElem.offsetParent();
            return n.top = n.right = n.bottom = n.left = "auto", t.left > i / 2 ? n.right = s.width() - this.getPopupSize().width - t.left : n.left = t.left, t.top > o / 2 ? n.bottom = s.height() - this.getPopupSize().height - t.top : n.top = t.top, this.__base.call(this, n, e)
        },
        getWindowSize: function() {
            return this._getSizeOf($(document))
        },
        onResize: function() {
            this._cache = {}, this._isHiding || this.repaint()
        },
        _onShown: function() {
            var t = this.__base.apply(this, arguments);
            return this.unbindFromDoc(n), t
        },
        _bindToResize: function() {
            this.bindToWin("resize", this.onResize)
        },
        _unbindFromResize: function() {
            this.unbindFromWin("resize")
        },
        _moveToContainer: function() {
            return this.domElem.parent().length ? this : (this.__base.apply(this, arguments), this)
        }
    })
}(), BEM.decl("i-user-services", {}, {
    get: function(t) {
        this._data ? t(this._data) : this._get(t)
    },
    getApiUrl: function() {
        return BEM.blocks["i-global"].param("pass-host") + "/services?callback=?"
    },
    _callbacks: [],
    _get: function(t) {
        var e = this;
        e._callbacks.push(t), 1 == e._callbacks.length && $.getJSON(e.getApiUrl(), {
            locale: BEM.blocks["i-global"].param("lang"),
            login: "yes",
            "current-login": BEM.blocks["i-global"].param("login"),
            yu: $.cookie("yandexuid")
        }, function(t) {
            e._data = t, $.each(e._callbacks, function() {
                this (t)
            }), e._callbacks.length = 0
        })
    }
}), BEM.decl("i-user-services", {}, {
    get: function(t) {
        if (this._data)
            this.trigger(this._data, t);
        else {
            var e = this;
            this._get(function(n) {
                e.trigger(n, t)
            })
        }
    },
    trigger: function(t, e) {
        e(t), $(window).trigger("popupredraw.lego")
    }
}), function(t, e, n, i) {
    n.DOM.decl("b-head-user", {
        onSetMod: {
            js: function() {
                var t = this;
                this._maxLength = parseInt(this.params.maxlength, 10), this._maxLengthRel = parseInt(this.params["maxlength-relative"], 10), ("" !== i.param("login") || i.param("displayName")) && n.blocks["i-user-services"].get(function(e) {
                    t.buildHtml(e)
                })
            }
        },
        getDefaultParams: function() {
            return {
                maxlength: 20,
                "maxlength-relative": 3
            }
        },
        getUserItems: function(e) {
            var o = e.displayName && e.displayName.social ? [{
                name: n.I18N("b-head-user", "profiles"),
                url: i.param("social-host")
            }
            ]: [{
                name: n.I18N("b-head-user", "passport"),
                url: i.param("passport-host") + "/passport?mode=passport"
            }, {
                name: n.I18N("b-head-user", "change-password"),
                url: i.param("passport-host") + "/passport?mode=changepass&retpath=" + encodeURIComponent(i.param("retpath")) + "/"
            }
            ], s = "";
            return t.each(o, function(t, e) {
                s += '<li class="b-menu__layout-vert-cell' + (t == o.length - 1 ? " b-menu__layout-vert-cell_position_last" : "") + '"><div class="b-menu__item"><a class="b-link b-head-user__link" href="' + e.url + '">' + e.name + "</a></div></li>"
            }), s
        },
        buildHtml: function(e) {
            if (e && e.services) {
                var o, s, r, a = this, c = e.displayName && e.displayName.social, u = n.blocks["i-common__string"].cleverSubstring(t("<div/>").text(a.getUserLogin(e)).html(), a._maxLength, a._maxLengthRel), l = this.findBlocksOn("i-ctr") && i.param("show-counters"), d = "";
                o = c ? '<b class="b-user b-user_social_yes"><span class="b-user__provider-ico" style="background:url(//static.yandex.st/social/current/icons/16/' + c.provider + '.png)"></span>' + u + "</b>" : '<b class="b-user"><b class="b-user__first-letter">' + u.substring(0, 1) + "</b>" + u.substring(1) + "</b>", d = a.params.href ? a.params.href : c ? i.param("social-host") : /\.com(.tr)?$/.test(location.hostname) ? i.param("passport-host") : i.param("i-host") || "https://i.yandex.ru", s = '<a class="b-link b-link_pseudo_yes ' + (l ? "i-ctr i-ctr_click_b-link " : "") + "i-bem\" onclick=\"return {'b-link':{}" + (l ? ",'i-ctr':{stat:'head-login'}" : "") + '}"' + (!c && a.hasMod("type", "com") ? "" : ' href="' + d + '"') + '><span class="b-link__inner">' + o + "</span></a>", r = '<div class="b-dropdowna b-dropdowna i-bem" onclick="return {\'b-dropdowna\':{}}"><span class="b-dropdowna__switcher">' + s + '</span><div class="i-popup i-popup_autoclosable_yes i-popup i-bem" onclick="return {\'i-popup\':{}}"><div class="i-popup__content"><div class="i-popup__under b-popupa__under"></div><div class="b-popupa b-popupa_theme_ffffff b-popupa_direction_down i-bem b-dropdowna__popup" onclick="return {\'b-popupa\':{}}"><i class="b-popupa__shadow"></i><i class="b-popupa__tail"><i class="b-popupa__tail-i"></i></i><table cellpadding="0" cellspasing="0" class="b-popupa__wrap"><tr><td class="b-popupa__wrap-cell"><div class="b-popupa__content"><div class="b-menu b-menu_layout_vert b-dropdowna__menu"><ul class="b-menu__layout-vert">', c && a.setMod("social", "yes"), t.each(e.services, function(t) {
                    r += '<li class="b-menu__layout-vert-cell' + (0 === t ? " b-menu__layout-vert-cell_position_first" : "") + '"><div class="b-menu__item">', r += i.param("id") == this.id && i.param("index") ? "<strong>" + this.title + "</strong>" : '<a class="b-link b-head-user__link" href="' + this.url + '" onmousedown="Lego.ch(\'' + i.param("id") + ".login.myservices." + this.id + "',this)\">" + this.title + "</a>", r += "</div></li>"
                }), e.services.length && (r += '<li class="b-menu__layout-vert-separator"><i class="b-menu__separator"></i></li>'), r += a.getUserItems(e), r += "</ul></div></div></td></tr></table></div></div></div></div>", n.DOM.update(a.findBlockInside("b-user").domElem, r)
            }
        },
        getUserLogin: function(t) {
            var e = i.param("displayName");
            return e && e.social && e.name || t.displayName && t.displayName.name || i.param("login") || t.login
        }
    })
}(jQuery, window.Lego, BEM, BEM.blocks["i-global"]), function() {
    function t() {
        Lego.c("lego-ctr/p=show(" + i(n()) + ")"), s = []
    }
    function e(t) {
        var e = t.split("-");
        return e.join("(") + new Array(e.length).join(")")
    }
    function n() {
        for (var t = [], e = 0, n = s.length; n > e; e++) {
            for (var i = s[e].split("-"), o = t, r = 0, a = i.length - 1; a > r; r++) {
                var c = i[r];
                o[c] || (o[c] = [], o.push(c)), o = o[c]
            }
            o.push(i[i.length - 1])
        }
        return t
    }
    function i(t) {
        for (var e = [], n = [], o = 0; o < t.length; o++) {
            var s = t[o];
            t[s] ? e.push(s + "(" + i(t[s]) + ")") : n.push(s)
        }
        return e.length && n.push(e.join("")), n.join(",")
    }
    function o(t) {
        var e = "-" + BEM.blocks["i-global"].param("id");
        return - 1 === t.indexOf(e, t.length - e.length) ? t += e : t
    }
    var s = [];
    BEM.DOM.decl("i-ctr", {
        onSetMod: {
            js: function() {
                this.setupShow(), this.setupClick()
            }
        },
        setupShow: function() {
            var e = o(this.params.stat);
            s.push(e), 1 == s.length && this.afterCurrentEvent(function() {
                t()
            })
        },
        setupClick: function() {
            var t = this.getMod("click");
            t ? this.findBlockOn(t).on("click", function() {
                this.sendClick()
            }, this) : this.bindTo("mousedown", this.sendClick)
        },
        sendClick: function(t) {
            Lego.c("lego-ctr/p=click(" + e(o(this.params.stat)) + ")", t && t.target && "A" === t.target.tagName.toUpperCase() && t.target)
        }
    })
}(), ya.pages = {}, ya.templates = {
    common: {},
    pages: {}
}, ya.generators = {
    common: {},
    pages: {}
}, ya.hash = {
    common: {},
    pages: {}
}, function(t, e) {
    t.make = function(t, e) {
        return t.hasOwnProperty(e) || (t[e] = {}), t[e]
    }, t.uid = function() {
        return String(Math.random()).substr(2, 12)
    }, t.htmlFilter = function(t) {
        return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    }, t.urlFilter = function(t) {
        return t.replace(/./g, function(t) {
            return t.match(/[&@\/;:=+?$]/) ? t : encodeURIComponent(t)
        })
    }, t.capitalize = function(t) {
        return t = t || "", t[0].toUpperCase() + t.substring(1)
    }, t.formatNumber = function(t, e, n) {
        if (isNaN(t))
            return t;
        t += "", e = e || ",", n = n || "&nbsp;";
        var i = t.lastIndexOf("."), o = [], s = "", r = 0, a = "";
        for ( - 1 !== i ? s = e + t.substr(i + 1) : i = t.length; --i>-1;)
            a = t[i] + a, r++, 3 === r && (o.push(a), a = "", r = 0);
        return a && o.push(a), o.reverse().join(n) + s
    }, t.Collection = function(t) {
        this.storage = [], this.idAttr = t || "id"
    }, t.Collection.prototype = {
        add: function(t) {
            this.storage.push(t), $(this).trigger("append." + t[this.idAttr], [t])
        },
        get: function(t) {
            var e = this;
            return this.storage.filter(function(n) {
                return n[e.idAttr] === t
            })
        }
    }, t.log = function() {
        if (env.debug)
            if (e.console)
                e.console.log.apply(this, arguments);
            else if (e.debug)
                e.debug.apply(this, arguments);
            else if (e.document && e.document.body) {
                var t = "debugWindow", n = document.getElementById(t);
                n ? n.innerHTML += "<hr/>msg" : document.body.innerHTML += '<div style="width:50%;height: 50px; margin-bottom: 200px; overflow: auto;position: absolute;bottom: 0; z-index: 9999; border:1px solid #ccc;background: #fff; color: #000; padding: 1em 2em;margin:2px;" id="' + t + '">Debug:<hr/>' + arguments[0] + "</div>"
            }
    }, t.zCookie = {
        insert: function(t, e) {
            var n, i, o, s = e.split(":"), r = [], a=!1;
            if (3 === s.length) {
                o = s[0], i = t.split("|");
                for (var c = 0, u = i.length; u > c; c++)
                    i[c] && (n = i[c].split(":"), n[0] === o ? (r.push(e), a=!0) : r.push(i[c]));
                return a || r.push(e), r.join("|")
            }
            return !1
        },
        get: function(t) {
            t || (t = "");
            for (var e = t.split("|"), n = {}, i = 0, o = e.length; o > i; i++) {
                var s = e[i].split(":");
                3 === s.length && (n[s[0].substr(2)] = s)
            }
            return function(t, e, i) {
                var o, s, r = n[t];
                if (r) {
                    o = r[1], s = "l" === r[2];
                    var a = {
                        version: o,
                        ls: s,
                        cache: "c" === r[2] && o === e,
                        fresh: s && o === e,
                        patchable: s && i && i.some(function(t) {
                            return t === o
                        })
                    };
                    return a
                }
            }
        }
    }, t.getStaticURL = function(t) {
        var n = e && e.staticVersion || ya && ya.page && ya.page.staticVersion || "1.274";
        return (env.devInstance ? "/v12" : "//yastatic.net/www/" + n + "/v12") + t
    }
}(ya, this), function() {
    function t(t) {
        var e = $(".b-etrains__track", t), n = $(".b-etrains__direction", t);
        n.length > 1 && n.unbind("click").click(function() {
            e.toggleClass("i-hidden");
            var t = this.parentNode.onclick();
            t && t.stat && csh_ifmsid(t.stat)
        }), $(".b-etrains__track_time", t).unbind("mouseover").mouseover(function() {
            var t = this.onclick().id;
            $("#" + t).removeClass("i-hidden")
        }).unbind("mouseout").mouseout(function() {
            var t = this.onclick().id;
            $("#" + t).addClass("i-hidden")
        })
    }
    window.initEtrains = t
}(), BEM.DOM.decl({
    name: "i-flashcookie",
    modName: "type",
    modVal: "inline"
}, {
    onSetMod: {
        js: function() {
            Lego.block["i-flashcookie"].call(this.domElem, this.params)
        }
    }
}), function() {
    var t = Lego.block["i-flashcookie"], e = function(t) {
        function e(t, e) {
            return '<param name="' + t + '" value="' + e + '" />'
        }
        function n(t, e) {
            return " " + t + '="' + e + '"'
        }
        function i(t, e, n) {
            var i = t.match(e);
            return i && i[n + 1 ? n: 1] || ""
        }
        function o() {
            return i(h.cookie, new RegExp("fuid" + c + "=([^;]+)"))
        }
        function s() {
            var t = h.domain;
            return t && i(t, /[^.]+\.[^.]{2,3}(\.tr)?$/, 0)
        }
        t || (t = {});
        var r, a, c, u, l = t.host || "kiks.yandex", d = window, h = document, f = navigator, p = f.userAgent, m = f.mimeTypes, _ = 9, g = p && /gecko/i.test(p), b = p && /MSIE/.test(p) && /Win/.test(p), v = g ? "embed": "object", y = "//" + l.split(".").slice(0, - 1).join(".") + ".%s/system/fc%d.html", w = "//" + l + ".ru/system/%s%d.swf", k = "//" + l + ".ru/fu", E = "//" + l + ".ru/fu", M = "application/x-shockwave-flash", x = "style", C = "position:absolute;margin-left:-999em;top:0;", T = ' classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"', S = "allowScriptAccess", B = "always", D = "height", I = "wmode", O = "window", P = "width", $ = "bgcolor", j = "#FFFFFF", A = "ya_fc", N = n(x, C) + n(P, 1) + n(D, 1) + n("id", A), L=!1;
        d.ya_fc_external = function(t, i, l, d) {
            r = t, a = i, c = l, u = d || "fc";
            var h, f = m && m[M] ? m[M].enabledPlugin: 0, p = L, y = 0, x = o() ? w.replace("%d", a).replace("%s", u): k.replace("%s", s());
            if (f)
                y = f.description.split("Shockwave Flash ")[1], p = parseInt(y, 10) >= _;
            else if (b) {
                p = L;
                try {
                    p = new ActiveXObject("ShockwaveFlash.ShockwaveFlash." + _)
                } catch (D) {}
            }
            return h = p ? g ? '<div style="' + C + '"><' + v + N + n("src", x) + n(I, O) + n($, j) + n(S, B) + n("type", M) + "/></div>" : (b ? "<" + v + T + N + ">" + e("movie", x) : "<" + v + N + n("type", M) + n("data", x) + ">") + e(I, O) + e($, j) + e(S, B) + "</" + v + ">" : '<img src="' + E.replace("%s", s()) + '" style="' + C + '" />'
        }, d.ya_fc_requestData = function() {
            var t = h.getElementById("ya_fc");
            try {
                t.setLocation(s(), o())
            } catch (e) {}
        }, d.ya_fc_setCookie = function(t) {
            var e = new Date;
            e.setTime(e.getTime() + 31536e7), h.cookie = "fuid" + c + "=" + t + ";expires=" + e.toGMTString() + ";path=/;domain=" + s()
        }, d.ya_fc_getIFrame = function(t) {
            var e = y.replace("%s", t).replace("%d", r), n = h.createElement("iframe"), i = h.getElementsByTagName("body")[0], o = "40px;";
            n.src = e, n.style.cssText = P + o + D + o + "border:0;" + C, n.frameBorder = 0, i && i.insertBefore(n, i.firstChild)
        };
        var F = this, R = arguments.callee.load = function(t) {
            var e, n;
            t.mode ? (e = "00", n = "kfc") : e = "07", F.replaceWith(d.ya_fc_external("07_2", e, "01", n))
        };
        F.hasClass("i-flashcookie_autoload_no") || R(t)
    };
    Lego.block["i-flashcookie"] = function(n) {
        var i = this.hasClass("i-flashcookie_type_inline") ? e: t;
        i && i.call(this, n)
    }
}(), BEM.DOM.decl("b-keyboard-popup", {
    onSetMod: {
        js: function() {
            Lego.block["b-keyboard-popup"].call(this.domElem, this.params)
        }
    }
}), function(t, e) {
    e.block["b-keyboard-popup"] = function(n) {
        var i = t('<div class="b-keyboard-popup__gap"/><div class="b-keyboard-popup"><div class="b-keyboard-popup__bar"><span class="b-keyboard-popup__bar-link b-keyboard-popup__close"><img class="b-icon" src="//yandex.st/lego/_/La6qi18Z8LwgnZdsAr1qy1GwCwo.gif">' + BEM.I18N("b-keyboard-popup", "close") + "</span></div><div class=\"b-keyboard i-bem\" onclick=\"return {name: 'b-keyboard',fake: false,lang: '" + n.lang + "'" + (n["for"] ? ",'for': '" + n["for"] + "'" : "") + '};"/></div>').appendTo("body"), o = i.filter(".b-keyboard-popup"), s = (i.filter(".b-keyboard-popup__gap"), function(t) {
            return "undefined" == typeof t ? o.data("isHidden") : void o.data("isHidden", t)
        }), r = function() {
            s() || (i.addClass("i-hidden"), t(window).trigger("keyboardClosed.lego"), s(!0))
        };
        t(document).bind("popupsClose.lego", function(e, n) {
            s() || n && n.source && t.contains(o[0], n.source) || r()
        }), i.find(".b-keyboard-popup__close").click(function() {
            r();
            var t = e.block["b-keyboard"]._lastFocusedIn;
            t && t.focus()
        })
    }
}(jQuery, window.Lego), x.Component.extend({
    id: "b-head-tabs__promo",
    init: function(t) {
        var e = $(x.byId(t)), n = "b-head-tabs__promo", i = this, o = null, s = i.data("name"), r = s.split("@");
        if (2 !== r.length)
            throw new Error("incorrect name in b-head-tabs__promo module");
        if ("undefined" != typeof JSON&&!home.disabledLS) {
            if (o = home.localStorage.getItem(n))
                try {
                    o = JSON.parse(o)
                } catch (a) {
                o = null
            }
            o && o.date && o.date !== r[1] && (o = null), (!o || o&&!o[s]) && (e.find(".b-head-tabs__promo-icon").addClass("b-head-tabs__promo_active"), e.find("a").on("mousedown", function() {
                var t = o || {};
                t.date = t.date || r[1], t[s]=!0;
                try {
                    home.localStorage.setItem(n, JSON.stringify(t))
                } catch (e) {}
            }))
        }
    }
}), BEM.DOM.decl("b-spin", {
    onSetMod: {
        js: function() {
            this._size = this.getMod("size") || /[\d]+/.exec(this.getMod("theme"))[0], this._bgProp = "background-position", this._posPrefix = "0 -", this.elem("icon").css("background-position-y") && (this._bgProp = "background-position-y", this._posPrefix = "-"), this._curFrame = 0, this.hasMod("progress") && this.channel("sys").on("tick", this._onTick, this)
        },
        progress: {
            yes: function() {
                this.channel("sys").on("tick", this._onTick, this)
            },
            "": function() {
                this.channel("sys").un("tick", this._onTick, this)
            }
        }
    },
    _onTick: function() {
        var t=++this._curFrame * this._size;
        t >= 36 * this._size && (this._curFrame = t = 0), this.elem("icon").css(this._bgProp, this._posPrefix + t + "px")
    },
    destruct: function() {
        this.channel("sys").un("tick", this._onTick, this), this.__base.apply(this, arguments)
    }
}), function() {
    var t = "//yastatic.net/browser-cookie/_fc.js", e = '<div style="height:0;overflow:hidden;"><object classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" width="1" height="1" id="fco" ><param name="movie" value="//yastatic.net/browser-cookie/flash-cookie.swf"/><param name="allowScriptAccess" value="Always"/><embed src="//yastatic.net/browser-cookie/flash-cookie.swf" allowScriptAccess="Always" width="1" height="1" id="fce"></embed></object></div>';
    setTimeout(function() {
        window.mordaCookieCopy && $.getScript(t, function() {
            $("body").append(e)
        })
    }, 5e3)
}(), $(document).ready(function() {
    window.setTimeout(function() {
        var t = home && home.getData && home.getData("b-cache.url");
        if (t) {
            var e = navigator.userAgent.toLowerCase();
            /crios/.test(e) && (/ipad/.test(e) || /ipod/.test(e) || /iphone/.test(e)) || ($.ajaxSetup({
                cache: !0
            }), $.getScript(t), $.ajaxSetup({
                cache: !1
            }))
        }
    }, 5e3)
}), function() {
    var t;
    t = function() {
        var t = document, e = window, n = {}, i = function() {};
        i.prototype.addEventListener = function(t, e) {
            this._listeners || (this._listeners = {}), t in this._listeners || (this._listeners[t] = []);
            var i = this._listeners[t];
            - 1 === n.arrIndexOf(i, e) && i.push(e)
        }, i.prototype.removeEventListener = function(t, e) {
            if (this._listeners && t in this._listeners) {
                var i = this._listeners[t], o = n.arrIndexOf(i, e);
                return - 1 !== o ? void(i.length > 1 ? this._listeners[t] = i.slice(0, o).concat(i.slice(o + 1)) : delete this._listeners[t]) : void 0
            }
        }, i.prototype.dispatchEvent = function(t) {
            var e = t.type, n = Array.prototype.slice.call(arguments, 0);
            if (this["on" + e] && this["on" + e].apply(this, n), this._listeners && e in this._listeners)
                for (var i = 0; i < this._listeners[e].length; i++)
                    this._listeners[e][i].apply(this, n)
        };
        var o = function(t, e) {
            if (this.type = t, "undefined" != typeof e)
                for (var n in e)
                    e.hasOwnProperty(n) && (this[n] = e[n])
        };
        o.prototype.toString = function() {
            var t = [];
            for (var e in this)
                if (this.hasOwnProperty(e)) {
                    var n = this[e];
                    "function" == typeof n && (n = "[function]"), t.push(e + "=" + n)
                }
            return "SimpleEvent(" + t.join(", ") + ")"
        };
        var s = function(t) {
            var e = this;
            e._events = t || [], e._listeners = {}
        };
        s.prototype.emit = function(t) {
            var e = this;
            if (e._verifyType(t), !e._nuked) {
                var n = Array.prototype.slice.call(arguments, 1);
                if (e["on" + t] && e["on" + t].apply(e, n), t in e._listeners)
                    for (var i = 0; i < e._listeners[t].length; i++)
                        e._listeners[t][i].apply(e, n)
            }
        }, s.prototype.on = function(t, e) {
            var n = this;
            n._verifyType(t), n._nuked || (t in n._listeners || (n._listeners[t] = []), n._listeners[t].push(e))
        }, s.prototype._verifyType = function(t) {
            var e = this;
            - 1 === n.arrIndexOf(e._events, t) && n.log("Event " + JSON.stringify(t) + " not listed " + JSON.stringify(e._events) + " in " + e)
        }, s.prototype.nuke = function() {
            var t = this;
            t._nuked=!0;
            for (var e = 0; e < t._events.length; e++)
                delete t[t._events[e]];
            t._listeners = {}
        };
        var r = "abcdefghijklmnopqrstuvwxyz0123456789_";
        n.random_string = function(t, e) {
            e = e || r.length;
            var n, i = [];
            for (n = 0; t > n; n++)
                i.push(r.substr(Math.floor(Math.random() * e), 1));
            return i.join("")
        }, n.random_number = function(t) {
            return Math.floor(Math.random() * t)
        }, n.random_number_string = function(t) {
            var e = ("" + (t - 1)).length, i = Array(e + 1).join("0");
            return (i + n.random_number(t)).slice( - e)
        }, n.getOrigin = function(t) {
            t += "/";
            var e = t.split("/").slice(0, 3);
            return e.join("/")
        }, n.isSameOriginUrl = function(t, n) {
            return n || (n = e.location.href), t.split("/").slice(0, 3).join("/") === n.split("/").slice(0, 3).join("/")
        }, n.getParentDomain = function(t) {
            if (/^[0-9.]*$/.test(t))
                return t;
            if (/^\[/.test(t))
                return t;
            if (!/[.]/.test(t))
                return t;
            var e = t.split(".").slice(1);
            return e.join(".")
        }, n.objectExtend = function(t, e) {
            for (var n in e)
                e.hasOwnProperty(n) && (t[n] = e[n]);
            return t
        };
        var a = "_jp";
        n.polluteGlobalNamespace = function() {
            a in e || (e[a] = {})
        }, n.closeFrame = function(t, e) {
            return "c" + JSON.stringify([t, e])
        }, n.userSetCode = function(t) {
            return 1e3 === t || t >= 3e3 && 4999 >= t
        }, n.countRTO = function(t) {
            var e;
            return e = t > 100 ? 3 * t : t + 200
        }, n.log = function() {
            e.console && console.log && console.log.apply && console.log.apply(console, arguments)
        }, n.bind = function(t, e) {
            return t.bind ? t.bind(e) : function() {
                return t.apply(e, arguments)
            }
        }, n.flatUrl = function(t) {
            return - 1 === t.indexOf("?")&&-1 === t.indexOf("#")
        }, n.amendUrl = function(e) {
            var i = t.location;
            if (!e)
                throw new Error("Wrong url for SockJS");
            if (!n.flatUrl(e))
                throw new Error("Only basic urls are supported in SockJS");
            return 0 === e.indexOf("//") && (e = i.protocol + e), 0 === e.indexOf("/") && (e = i.protocol + "//" + i.host + e), e = e.replace(/[/]+$/, "")
        }, n.arrIndexOf = function(t, e) {
            for (var n = 0; n < t.length; n++)
                if (t[n] === e)
                    return n;
            return - 1
        }, n.arrSkip = function(t, e) {
            var i = n.arrIndexOf(t, e);
            if ( - 1 === i)
                return t.slice();
            var o = t.slice(0, i);
            return o.concat(t.slice(i + 1))
        }, n.isArray = Array.isArray || function(t) {
            return {}.toString.call(t).indexOf("Array") >= 0
        }, n.delay = function(t, e) {
            return "function" == typeof t && (e = t, t = 0), setTimeout(e, t)
        };
        var c, u = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, l = {
            "\x00": "\\u0000",
            "": "\\u0001",
            "": "\\u0002",
            "": "\\u0003",
            "": "\\u0004",
            "": "\\u0005",
            "": "\\u0006",
            "": "\\u0007",
            "\b": "\\b",
            "	": "\\t",
            "\n": "\\n",
            "": "\\u000b",
            "\f": "\\f",
            "\r": "\\r",
            "": "\\u000e",
            "": "\\u000f",
            "": "\\u0010",
            "": "\\u0011",
            "": "\\u0012",
            "": "\\u0013",
            "": "\\u0014",
            "": "\\u0015",
            "": "\\u0016",
            "": "\\u0017",
            "": "\\u0018",
            "": "\\u0019",
            "": "\\u001a",
            "": "\\u001b",
            "": "\\u001c",
            "": "\\u001d",
            "": "\\u001e",
            "": "\\u001f",
            '"': '\\"',
            "\\": "\\\\",
            "": "\\u007f",
            "": "\\u0080",
            "": "\\u0081",
            "": "\\u0082",
            "": "\\u0083",
            "": "\\u0084",
            "": "\\u0085",
            "": "\\u0086",
            "": "\\u0087",
            "": "\\u0088",
            "": "\\u0089",
            "": "\\u008a",
            "": "\\u008b",
            "": "\\u008c",
            "": "\\u008d",
            "": "\\u008e",
            "": "\\u008f",
            "": "\\u0090",
            "": "\\u0091",
            "": "\\u0092",
            "": "\\u0093",
            "": "\\u0094",
            "": "\\u0095",
            "": "\\u0096",
            "": "\\u0097",
            "": "\\u0098",
            "": "\\u0099",
            "": "\\u009a",
            "": "\\u009b",
            "": "\\u009c",
            "": "\\u009d",
            "": "\\u009e",
            "": "\\u009f",
            "­": "\\u00ad",
            "؀": "\\u0600",
            "؁": "\\u0601",
            "؂": "\\u0602",
            "؃": "\\u0603",
            "؄": "\\u0604",
            "܏": "\\u070f",
            "឴": "\\u17b4",
            "឵": "\\u17b5",
            "‌": "\\u200c",
            "‍": "\\u200d",
            "‎": "\\u200e",
            "‏": "\\u200f",
            "\u2028": "\\u2028",
            "\u2029": "\\u2029",
            "‪": "\\u202a",
            "‫": "\\u202b",
            "‬": "\\u202c",
            "‭": "\\u202d",
            "‮": "\\u202e",
            " ": "\\u202f",
            "⁠": "\\u2060",
            "⁡": "\\u2061",
            "⁢": "\\u2062",
            "⁣": "\\u2063",
            "⁤": "\\u2064",
            "⁥": "\\u2065",
            "⁦": "\\u2066",
            "⁧": "\\u2067",
            "⁨": "\\u2068",
            "⁩": "\\u2069",
            "⁪": "\\u206a",
            "⁫": "\\u206b",
            "⁬": "\\u206c",
            "⁭": "\\u206d",
            "⁮": "\\u206e",
            "⁯": "\\u206f",
            "﻿": "\\ufeff",
            "￰": "\\ufff0",
            "￱": "\\ufff1",
            "￲": "\\ufff2",
            "￳": "\\ufff3",
            "￴": "\\ufff4",
            "￵": "\\ufff5",
            "￶": "\\ufff6",
            "￷": "\\ufff7",
            "￸": "\\ufff8",
            "￹": "\\ufff9",
            "￺": "\\ufffa",
            "￻": "\\ufffb",
            "￼": "\\ufffc",
            "�": "\\ufffd",
            "￾": "\\ufffe",
            "￿": "\\uffff"
        }, d = /[\x00-\x1f\ud800-\udfff\ufffe\uffff\u0300-\u0333\u033d-\u0346\u034a-\u034c\u0350-\u0352\u0357-\u0358\u035c-\u0362\u0374\u037e\u0387\u0591-\u05af\u05c4\u0610-\u0617\u0653-\u0654\u0657-\u065b\u065d-\u065e\u06df-\u06e2\u06eb-\u06ec\u0730\u0732-\u0733\u0735-\u0736\u073a\u073d\u073f-\u0741\u0743\u0745\u0747\u07eb-\u07f1\u0951\u0958-\u095f\u09dc-\u09dd\u09df\u0a33\u0a36\u0a59-\u0a5b\u0a5e\u0b5c-\u0b5d\u0e38-\u0e39\u0f43\u0f4d\u0f52\u0f57\u0f5c\u0f69\u0f72-\u0f76\u0f78\u0f80-\u0f83\u0f93\u0f9d\u0fa2\u0fa7\u0fac\u0fb9\u1939-\u193a\u1a17\u1b6b\u1cda-\u1cdb\u1dc0-\u1dcf\u1dfc\u1dfe\u1f71\u1f73\u1f75\u1f77\u1f79\u1f7b\u1f7d\u1fbb\u1fbe\u1fc9\u1fcb\u1fd3\u1fdb\u1fe3\u1feb\u1fee-\u1fef\u1ff9\u1ffb\u1ffd\u2000-\u2001\u20d0-\u20d1\u20d4-\u20d7\u20e7-\u20e9\u2126\u212a-\u212b\u2329-\u232a\u2adc\u302b-\u302c\uaab2-\uaab3\uf900-\ufa0d\ufa10\ufa12\ufa15-\ufa1e\ufa20\ufa22\ufa25-\ufa26\ufa2a-\ufa2d\ufa30-\ufa6d\ufa70-\ufad9\ufb1d\ufb1f\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufb4e\ufff0-\uffff]/g, h = JSON && JSON.stringify || function(t) {
            return u.lastIndex = 0, u.test(t) && (t = t.replace(u, function(t) {
                return l[t]
            })), '"' + t + '"'
        }, f = function(t) {
            var e, n = {}, i = [];
            for (e = 0; 65536 > e; e++)
                i.push(String.fromCharCode(e));
            return t.lastIndex = 0, i.join("").replace(t, function(t) {
                return n[t] = "\\u" + ("0000" + t.charCodeAt(0).toString(16)).slice( - 4), ""
            }), t.lastIndex = 0, n
        };
        n.quote = function(t) {
            var e = h(t);
            return d.lastIndex = 0, d.test(e) ? (c || (c = f(d)), e.replace(d, function(t) {
                return c[t]
            })) : e
        };
        var p = ["websocket", "xdr-streaming", "xhr-streaming", "iframe-eventsource", "iframe-htmlfile", "xdr-polling", "xhr-polling", "iframe-xhr-polling", "jsonp-polling"];
        n.probeProtocols = function() {
            for (var t = {}, e = 0; e < p.length; e++) {
                var n = p[e];
                t[n] = M[n] && M[n].enabled()
            }
            return t
        }, n.detectProtocols = function(t, e, n) {
            var i = {}, o = [];
            e || (e = p);
            for (var s = 0; s < e.length; s++) {
                var r = e[s];
                i[r] = t[r]
            }
            var a = function(t) {
                var e = t.shift();
                i[e] ? o.push(e) : t.length > 0 && a(t)
            };
            return n.websocket!==!1 && a(["websocket"]), i["xhr-streaming"]&&!n.null_origin ? o.push("xhr-streaming") : !i["xdr-streaming"] || n.cookie_needed || n.null_origin ? a(["iframe-eventsource", "iframe-htmlfile"]) : o.push("xdr-streaming"), i["xhr-polling"]&&!n.null_origin ? o.push("xhr-polling") : !i["xdr-polling"] || n.cookie_needed || n.null_origin ? a(["iframe-xhr-polling", "jsonp-polling"]) : o.push("xdr-polling"), o
        };
        var m = "_sockjs_global";
        n.createHook = function() {
            var t = "a" + n.random_string(8);
            if (!(m in e)) {
                var i = {};
                e[m] = function(t) {
                    return t in i || (i[t] = {
                        id: t,
                        del: function() {
                            delete i[t]
                        }
                    }), i[t]
                }
            }
            return e[m](t)
        }, n.attachMessage = function(t) {
            n.attachEvent("message", t)
        }, n.attachEvent = function(n, i) {
            "undefined" != typeof e.addEventListener ? e.addEventListener(n, i, !1) : (t.attachEvent("on" + n, i), e.attachEvent("on" + n, i))
        }, n.detachMessage = function(t) {
            n.detachEvent("message", t)
        }, n.detachEvent = function(n, i) {
            "undefined" != typeof e.addEventListener ? e.removeEventListener(n, i, !1) : (t.detachEvent("on" + n, i), e.detachEvent("on" + n, i))
        };
        var _ = {}, g=!1, b = function() {
            for (var t in _)
                _[t](), delete _[t]
        }, v = function() {
            g || (g=!0, b())
        };
        n.attachEvent("unload", v), n.unload_add = function(t) {
            var e = n.random_string(8);
            return _[e] = t, g && n.delay(b), e
        }, n.unload_del = function(t) {
            t in _ && delete _[t]
        }, n.createIframe = function(e, i) {
            var o, s, r = t.createElement("iframe"), a = function() {
                clearTimeout(o);
                try {
                    r.onload = null
                } catch (t) {}
                r.onerror = null
            }, c = function() {
                r && (a(), setTimeout(function() {
                    r && r.parentNode.removeChild(r), r = null
                }, 0), n.unload_del(s))
            }, u = function(t) {
                r && (c(), i(t))
            }, l = function(t, e) {
                try {
                    r && r.contentWindow && r.contentWindow.postMessage(t, e)
                } catch (n) {}
            };
            return r.src = e, r.style.display = "none", r.style.position = "absolute", r.onerror = function() {
                u("onerror")
            }, r.onload = function() {
                clearTimeout(o), o = setTimeout(function() {
                    u("onload timeout")
                }, 2e3)
            }, t.body.appendChild(r), o = setTimeout(function() {
                u("timeout")
            }, 15e3), s = n.unload_add(c), {
                post: l,
                cleanup: c,
                loaded: a
            }
        }, n.createHtmlfile = function(t, i) {
            var o, s, r, c = new ActiveXObject("htmlfile"), u = function() {
                clearTimeout(o)
            }, l = function() {
                c && (u(), n.unload_del(s), r.parentNode.removeChild(r), r = c = null, CollectGarbage())
            }, d = function(t) {
                c && (l(), i(t))
            }, h = function(t, e) {
                try {
                    r && r.contentWindow && r.contentWindow.postMessage(t, e)
                } catch (n) {}
            };
            c.open(), c.write('<html><script>document.domain="' + document.domain + '";</script></html>'), c.close(), c.parentWindow[a] = e[a];
            var f = c.createElement("div");
            return c.body.appendChild(f), r = c.createElement("iframe"), f.appendChild(r), r.src = t, o = setTimeout(function() {
                d("timeout")
            }, 15e3), s = n.unload_add(l), {
                post: h,
                cleanup: l,
                loaded: u
            }
        };
        var y = function() {};
        y.prototype = new s(["chunk", "finish"]), y.prototype._start = function(t, i, o, s) {
            var r = this;
            try {
                r.xhr = new XMLHttpRequest
            } catch (a) {}
            if (!r.xhr)
                try {
                    r.xhr = new e.ActiveXObject("Microsoft.XMLHTTP")
            } catch (a) {}(e.ActiveXObject || e.XDomainRequest) && (i += ( - 1 === i.indexOf("?") ? "?" : "&") + "t=" + + new Date), r.unload_ref = n.unload_add(function() {
                r._cleanup(!0)
            });
            try {
                r.xhr.open(t, i, !0)
            } catch (c) {
                return r.emit("finish", 0, ""), void r._cleanup()
            }
            if (s && s.no_credentials || (r.xhr.withCredentials = "true"), s && s.headers)
                for (var u in s.headers)
                    r.xhr.setRequestHeader(u, s.headers[u]);
            r.xhr.onreadystatechange = function() {
                if (r.xhr) {
                    var t = r.xhr;
                    switch (t.readyState) {
                    case 3:
                        try {
                            var e = t.status, n = t.responseText
                        } catch (t) {}
                        1223 === e && (e = 204), n && n.length > 0 && r.emit("chunk", e, n);
                        break;
                    case 4:
                        var e = t.status;
                        1223 === e && (e = 204), r.emit("finish", e, t.responseText), r._cleanup(!1)
                    }
                }
            }, r.xhr.send(o)
        }, y.prototype._cleanup = function(t) {
            var e = this;
            if (e.xhr) {
                if (n.unload_del(e.unload_ref), e.xhr.onreadystatechange = function() {}, t)
                    try {
                        e.xhr.abort()
                    } catch (i) {}
                e.unload_ref = e.xhr = null
            }
        }, y.prototype.close = function() {
            var t = this;
            t.nuke(), t._cleanup(!0)
        };
        var w = n.XHRCorsObject = function() {
            var t = this, e = arguments;
            n.delay(function() {
                t._start.apply(t, e)
            })
        };
        w.prototype = new y;
        var k = n.XHRLocalObject = function(t, e, i) {
            var o = this;
            n.delay(function() {
                o._start(t, e, i, {
                    no_credentials: !0
                })
            })
        };
        k.prototype = new y;
        var E = n.XDRObject = function(t, e, i) {
            var o = this;
            n.delay(function() {
                o._start(t, e, i)
            })
        };
        E.prototype = new s(["chunk", "finish"]), E.prototype._start = function(t, e, i) {
            var o = this, s = new XDomainRequest;
            e += ( - 1 === e.indexOf("?") ? "?" : "&") + "t=" + + new Date;
            var r = s.ontimeout = s.onerror = function() {
                o.emit("finish", 0, ""), o._cleanup(!1)
            };
            s.onprogress = function() {
                o.emit("chunk", 200, s.responseText)
            }, s.onload = function() {
                o.emit("finish", 200, s.responseText), o._cleanup(!1)
            }, o.xdr = s, o.unload_ref = n.unload_add(function() {
                o._cleanup(!0)
            });
            try {
                o.xdr.open(t, e), o.xdr.send(i)
            } catch (a) {
                r()
            }
        }, E.prototype._cleanup = function(t) {
            var e = this;
            if (e.xdr) {
                if (n.unload_del(e.unload_ref), e.xdr.ontimeout = e.xdr.onerror = e.xdr.onprogress = e.xdr.onload = null, t)
                    try {
                        e.xdr.abort()
                    } catch (i) {}
                e.unload_ref = e.xdr = null
            }
        }, E.prototype.close = function() {
            var t = this;
            t.nuke(), t._cleanup(!0)
        }, n.isXHRCorsCapable = function() {
            return e.XMLHttpRequest && "withCredentials"in new XMLHttpRequest ? 1 : e.XDomainRequest && t.domain ? 2 : N.enabled() ? 3 : 4
        };
        var M = function(t, i, o) {
            if (this === e)
                return new M(t, i, o);
            var s, r = this;
            r._options = {
                devel: !1,
                debug: !1,
                protocols_whitelist: [],
                info: void 0,
                rtt: void 0
            }, o && n.objectExtend(r._options, o), r._base_url = n.amendUrl(t), r._server = r._options.server || n.random_number_string(1e3), r._options.protocols_whitelist && r._options.protocols_whitelist.length ? s = r._options.protocols_whitelist : (s = "string" == typeof i && i.length > 0 ? [i] : n.isArray(i) ? i : null, s && r._debug('Deprecated API: Use "protocols_whitelist" option instead of supplying protocol list as a second parameter to SockJS constructor.')), r._protocols = [], r.protocol = null, r.readyState = M.CONNECTING, r._ir = q(r._base_url), r._ir.onfinish = function(t, e) {
                r._ir = null, t ? (r._options.info && (t = n.objectExtend(t, r._options.info)), r._options.rtt && (e = r._options.rtt), r._applyInfo(t, e, s), r._didClose()) : r._didClose(1002, "Can't connect to server", !0)
            }
        };
        M.prototype = new i, M.version = "0.3.4", M.CONNECTING = 0, M.OPEN = 1, M.CLOSING = 2, M.CLOSED = 3, M.prototype._debug = function() {
            this._options.debug && n.log.apply(n, arguments)
        }, M.prototype._dispatchOpen = function() {
            var t = this;
            t.readyState === M.CONNECTING ? (t._transport_tref && (clearTimeout(t._transport_tref), t._transport_tref = null), t.readyState = M.OPEN, t.dispatchEvent(new o("open"))) : t._didClose(1006, "Server lost session")
        }, M.prototype._dispatchMessage = function(t) {
            var e = this;
            e.readyState === M.OPEN && e.dispatchEvent(new o("message", {
                data: t
            }))
        }, M.prototype._dispatchHeartbeat = function() {
            var t = this;
            t.readyState === M.OPEN && t.dispatchEvent(new o("heartbeat", {}))
        }, M.prototype._didClose = function(t, e, i) {
            var s = this;
            if (s.readyState !== M.CONNECTING && s.readyState !== M.OPEN && s.readyState !== M.CLOSING)
                throw new Error("INVALID_STATE_ERR");
            s._ir && (s._ir.nuke(), s._ir = null), s._transport && (s._transport.doCleanup(), s._transport = null);
            var r = new o("close", {
                code: t,
                reason: e,
                wasClean: n.userSetCode(t)
            });
            if (!n.userSetCode(t) && s.readyState === M.CONNECTING&&!i) {
                if (s._try_next_protocol(r))
                    return;
                r = new o("close", {
                    code: 2e3,
                    reason: "All transports failed",
                    wasClean: !1,
                    last_event: r
                })
            }
            s.readyState = M.CLOSED, n.delay(function() {
                s.dispatchEvent(r)
            })
        }, M.prototype._didMessage = function(t) {
            var e = this, n = t.slice(0, 1);
            switch (n) {
            case"o":
                e._dispatchOpen();
                break;
            case"a":
                for (var i = JSON.parse(t.slice(1) || "[]"), o = 0; o < i.length; o++)
                    e._dispatchMessage(i[o]);
                break;
            case"m":
                var i = JSON.parse(t.slice(1) || "null");
                e._dispatchMessage(i);
                break;
            case"c":
                var i = JSON.parse(t.slice(1) || "[]");
                e._didClose(i[0], i[1]);
                break;
            case"h":
                e._dispatchHeartbeat()
            }
        }, M.prototype._try_next_protocol = function(e) {
            var i = this;
            for (i.protocol && (i._debug("Closed transport:", i.protocol, "" + e), i.protocol = null), i._transport_tref && (clearTimeout(i._transport_tref), i._transport_tref = null); ;) {
                var o = i.protocol = i._protocols.shift();
                if (!o)
                    return !1;
                if (M[o] && M[o].need_body===!0 && (!t.body || "undefined" != typeof t.readyState && "complete" !== t.readyState))
                    return i._protocols.unshift(o), i.protocol = "waiting-for-load", n.attachEvent("load", function() {
                        i._try_next_protocol()
                    }), !0;
                if (M[o] && M[o].enabled(i._options)) {
                    var s = M[o].roundTrips || 1, r = (i._options.rto || 0) * s || 5e3;
                    i._transport_tref = n.delay(r, function() {
                        i.readyState === M.CONNECTING && i._didClose(2007, "Transport timeouted")
                    });
                    var a = n.random_string(8), c = i._base_url + "/" + i._server + "/" + a;
                    return i._debug("Opening transport:", o, " url:" + c, " RTO:" + i._options.rto), i._transport = new M[o](i, c, i._base_url), !0
                }
                i._debug("Skipping transport:", o)
            }
        }, M.prototype.close = function(t, e) {
            var i = this;
            if (t&&!n.userSetCode(t))
                throw new Error("INVALID_ACCESS_ERR");
            return i.readyState !== M.CONNECTING && i.readyState !== M.OPEN?!1 : (i.readyState = M.CLOSING, i._didClose(t || 1e3, e || "Normal closure"), !0)
        }, M.prototype.send = function(t) {
            var e = this;
            if (e.readyState === M.CONNECTING)
                throw new Error("INVALID_STATE_ERR");
            return e.readyState === M.OPEN && e._transport.doSend(n.quote("" + t)), !0
        }, M.prototype._applyInfo = function(e, i, o) {
            var s = this;
            s._options.info = e, s._options.rtt = i, s._options.rto = n.countRTO(i), s._options.info.null_origin=!t.domain;
            var r = n.probeProtocols();
            s._protocols = n.detectProtocols(r, o, e)
        };
        var x = M.websocket = function(t, i) {
            var o = this, s = i + "/websocket";
            s = "https" === s.slice(0, 5) ? "wss" + s.slice(5) : "ws" + s.slice(4), o.ri = t, o.url = s;
            var r = e.WebSocket || e.MozWebSocket;
            o.ws = new r(o.url), o.ws.onmessage = function(t) {
                o.ri._didMessage(t.data)
            }, o.unload_ref = n.unload_add(function() {
                o.ws.close()
            }), o.ws.onclose = function() {
                o.ri._didMessage(n.closeFrame(1006, "WebSocket connection broken"))
            }
        };
        x.prototype.doSend = function(t) {
            this.ws.send("[" + t + "]")
        }, x.prototype.doCleanup = function() {
            var t = this, e = t.ws;
            e && (e.onmessage = e.onclose = null, e.close(), n.unload_del(t.unload_ref), t.unload_ref = t.ri = t.ws = null)
        }, x.enabled = function() {
            return !(!e.WebSocket&&!e.MozWebSocket)
        }, x.roundTrips = 2;
        var C = function() {};
        C.prototype.send_constructor = function(t) {
            var e = this;
            e.send_buffer = [], e.sender = t
        }, C.prototype.doSend = function(t) {
            var e = this;
            e.send_buffer.push(t), e.send_stop || e.send_schedule()
        }, C.prototype.send_schedule_wait = function() {
            var t, e = this;
            e.send_stop = function() {
                e.send_stop = null, clearTimeout(t)
            }, t = n.delay(25, function() {
                e.send_stop = null, e.send_schedule()
            })
        }, C.prototype.send_schedule = function() {
            var t = this;
            if (t.send_buffer.length > 0) {
                var e = "[" + t.send_buffer.join(",") + "]";
                t.send_stop = t.sender(t.trans_url, e, function(e, n) {
                    t.send_stop = null, e===!1 ? t.ri._didClose(1006, "Sending error " + n) : t.send_schedule_wait()
                }), t.send_buffer = []
            }
        }, C.prototype.send_destructor = function() {
            var t = this;
            t._send_stop && t._send_stop(), t._send_stop = null
        };
        var T = function(e, i, o) {
            var s = this;
            if (!("_send_form"in s)) {
                var r = s._send_form = t.createElement("form"), a = s._send_area = t.createElement("textarea");
                a.name = "d", r.style.display = "none", r.style.position = "absolute", r.method = "POST", r.enctype = "application/x-www-form-urlencoded", r.acceptCharset = "UTF-8", r.appendChild(a), t.body.appendChild(r)
            }
            var r = s._send_form, a = s._send_area, c = "a" + n.random_string(8);
            r.target = c, r.action = e + "/jsonp_send?i=" + c;
            var u;
            try {
                u = t.createElement('<iframe name="' + c + '">')
            } catch (l) {
                u = t.createElement("iframe"), u.name = c
            }
            u.id = c, r.appendChild(u), u.style.display = "none";
            try {
                a.value = i
            } catch (d) {
                n.log("Your browser is seriously broken. Go home! " + d.message)
            }
            r.submit();
            var h = function() {
                u.onerror && (u.onreadystatechange = u.onerror = u.onload = null, n.delay(500, function() {
                    u.parentNode.removeChild(u), u = null
                }), a.value = "", o(!0))
            };
            return u.onerror = u.onload = h, u.onreadystatechange = function() {
                "complete" == u.readyState && h()
            }, h
        }, S = function(t) {
            return function(e, n, i) {
                var o = new t("POST", e + "/xhr_send", n);
                return o.onfinish = function(t) {
                    i(200 === t || 204 === t, "http status " + t)
                }, function(t) {
                    i(!1, t)
                }
            }
        }, B = function(e, i) {
            var o, s, r = t.createElement("script"), a = function(t) {
                s && (s.parentNode.removeChild(s), s = null), r && (clearTimeout(o), r.parentNode.removeChild(r), r.onreadystatechange = r.onerror = r.onload = r.onclick = null, r = null, i(t), i = null)
            }, c=!1, u = null;
            if (r.id = "a" + n.random_string(8), r.src = e, r.type = "text/javascript", r.charset = "UTF-8", r.onerror = function() {
                u || (u = setTimeout(function() {
                    c || a(n.closeFrame(1006, "JSONP script loaded abnormally (onerror)"))
                }, 1e3))
            }, r.onload = function() {
                a(n.closeFrame(1006, "JSONP script loaded abnormally (onload)"))
            }, r.onreadystatechange = function() {
                if (/loaded|closed/.test(r.readyState)) {
                    if (r && r.htmlFor && r.onclick) {
                        c=!0;
                        try {
                            r.onclick()
                        } catch (t) {}
                    }
                    r && a(n.closeFrame(1006, "JSONP script loaded abnormally (onreadystatechange)"))
                }
            }, "undefined" == typeof r.async && t.attachEvent)
                if (/opera/i.test(navigator.userAgent))
                    s = t.createElement("script"), s.text = "try{var a = document.getElementById('" + r.id + "'); if(a)a.onerror();}catch(x){};", r.async = s.async=!1;
                else {
                    try {
                        r.htmlFor = r.id, r.event = "onclick"
                    } catch (l) {}
                    r.async=!0
                }
            "undefined" != typeof r.async && (r.async=!0), o = setTimeout(function() {
                a(n.closeFrame(1006, "JSONP script loaded abnormally (timeout)"))
            }, 35e3);
            var d = t.getElementsByTagName("head")[0];
            return d.insertBefore(r, d.firstChild), s && d.insertBefore(s, d.firstChild), a
        }, D = M["jsonp-polling"] = function(t, e) {
            n.polluteGlobalNamespace();
            var i = this;
            i.ri = t, i.trans_url = e, i.send_constructor(T), i._schedule_recv()
        };
        D.prototype = new C, D.prototype._schedule_recv = function() {
            var t = this, e = function(e) {
                t._recv_stop = null, e && (t._is_closing || t.ri._didMessage(e)), t._is_closing || t._schedule_recv()
            };
            t._recv_stop = I(t.trans_url + "/jsonp", B, e)
        }, D.enabled = function() {
            return !0
        }, D.need_body=!0, D.prototype.doCleanup = function() {
            var t = this;
            t._is_closing=!0, t._recv_stop && t._recv_stop(), t.ri = t._recv_stop = null, t.send_destructor()
        };
        var I = function(t, i, o) {
            var s = "a" + n.random_string(6), r = t + "?c=" + escape(a + "." + s), c = 0, u = function(t) {
                switch (c) {
                case 0:
                    delete e[a][s], o(t);
                    break;
                case 1:
                    o(t), c = 2;
                    break;
                case 2:
                    delete e[a][s]
                }
            }, l = i(r, u);
            e[a][s] = l;
            var d = function() {
                e[a][s] && (c = 1, e[a][s](n.closeFrame(1e3, "JSONP user aborted read")))
            };
            return d
        }, O = function() {};
        O.prototype = new C, O.prototype.run = function(t, e, n, i, o) {
            var s = this;
            s.ri = t, s.trans_url = e, s.send_constructor(S(o)), s.poll = new Z(t, i, e + n, o)
        }, O.prototype.doCleanup = function() {
            var t = this;
            t.poll && (t.poll.abort(), t.poll = null)
        };
        var P = M["xhr-streaming"] = function(t, e) {
            this.run(t, e, "/xhr_streaming", ie, n.XHRCorsObject)
        };
        P.prototype = new O, P.enabled = function() {
            return e.XMLHttpRequest && "withCredentials"in new XMLHttpRequest&&!/opera/i.test(navigator.userAgent)
        }, P.roundTrips = 2, P.need_body=!0;
        var $ = M["xdr-streaming"] = function(t, e) {
            this.run(t, e, "/xhr_streaming", ie, n.XDRObject)
        };
        $.prototype = new O, $.enabled = function() {
            return !!e.XDomainRequest
        }, $.roundTrips = 2;
        var j = M["xhr-polling"] = function(t, e) {
            this.run(t, e, "/xhr", ie, n.XHRCorsObject)
        };
        j.prototype = new O, j.enabled = P.enabled, j.roundTrips = 2;
        var A = M["xdr-polling"] = function(t, e) {
            this.run(t, e, "/xhr", ie, n.XDRObject)
        };
        A.prototype = new O, A.enabled = $.enabled, A.roundTrips = 2;
        var N = function() {};
        N.prototype.i_constructor = function(t, e, i) {
            var o = this;
            o.ri = t, o.origin = n.getOrigin(i), o.base_url = i, o.trans_url = e;
            var s = i + "/iframe.html";
            o.ri._options.devel && (s += "?t=" + + new Date), o.window_id = n.random_string(8), s += "#" + o.window_id, o.iframeObj = n.createIframe(s, function(t) {
                o.ri._didClose(1006, "Unable to load an iframe (" + t + ")")
            }), o.onmessage_cb = n.bind(o.onmessage, o), n.attachMessage(o.onmessage_cb)
        }, N.prototype.doCleanup = function() {
            var t = this;
            if (t.iframeObj) {
                n.detachMessage(t.onmessage_cb);
                try {
                    t.iframeObj.iframe.contentWindow && t.postMessage("c")
                } catch (e) {}
                t.iframeObj.cleanup(), t.iframeObj = null, t.onmessage_cb = t.iframeObj = null
            }
        }, N.prototype.onmessage = function(t) {
            var e = this;
            if (t.origin === e.origin) {
                var n = t.data.slice(0, 8), i = t.data.slice(8, 9), o = t.data.slice(9);
                if (n === e.window_id)
                    switch (i) {
                    case"s":
                        e.iframeObj.loaded(), e.postMessage("s", JSON.stringify([M.version, e.protocol, e.trans_url, e.base_url]));
                        break;
                    case"t":
                        e.ri._didMessage(o)
                    }
                }
        }, N.prototype.postMessage = function(t, e) {
            var n = this;
            n.iframeObj.post(n.window_id + t + (e || ""), n.origin)
        }, N.prototype.doSend = function(t) {
            this.postMessage("m", t)
        }, N.enabled = function() {
            var t = navigator && navigator.userAgent&&-1 !== navigator.userAgent.indexOf("Konqueror");
            return ("function" == typeof e.postMessage || "object" == typeof e.postMessage)&&!t
        };
        var L, F = function(t, i) {
            parent !== e ? parent.postMessage(L + t + (i || ""), "*") : n.log("Can't postMessage, no parent window.", t, i)
        }, R = function() {};
        R.prototype._didClose = function(t, e) {
            F("t", n.closeFrame(t, e))
        }, R.prototype._didMessage = function(t) {
            F("t", t)
        }, R.prototype._doSend = function(t) {
            this._transport.doSend(t)
        }, R.prototype._doCleanup = function() {
            this._transport.doCleanup()
        }, n.parent_origin = void 0, M.bootstrap_iframe = function() {
            var i;
            L = t.location.hash.slice(1);
            var o = function(t) {
                if (t.source === parent && ("undefined" == typeof n.parent_origin && (n.parent_origin = t.origin), t.origin === n.parent_origin)) {
                    var o = t.data.slice(0, 8), s = t.data.slice(8, 9), r = t.data.slice(9);
                    if (o === L)
                        switch (s) {
                        case"s":
                            var a = JSON.parse(r), c = a[0], u = a[1], l = a[2], d = a[3];
                            if (c !== M.version && n.log('Incompatibile SockJS! Main site uses: "' + c + '", the iframe: "' + M.version + '".'), !n.flatUrl(l) ||!n.flatUrl(d))
                                return void n.log("Only basic urls are supported in SockJS");
                                if (!n.isSameOriginUrl(l) ||!n.isSameOriginUrl(d))
                                    return void n.log("Can't connect to different domain from within an iframe. (" + JSON.stringify([e.location.href, l, d]) + ")");
                                    i = new R, i._transport = new R[u](i, l, d);
                                    break;
                                case"m":
                                    i._doSend(r);
                                    break;
                                case"c":
                                    i && i._doCleanup(), i = null
                        }
                }
            };
            n.attachMessage(o), F("s")
        };
        var H = function(t, e) {
            var i = this;
            n.delay(function() {
                i.doXhr(t, e)
            })
        };
        H.prototype = new s(["finish"]), H.prototype.doXhr = function(t, e) {
            var i = this, o = (new Date).getTime(), s = new e("GET", t + "/info"), r = n.delay(8e3, function() {
                s.ontimeout()
            });
            s.onfinish = function(t, e) {
                if (clearTimeout(r), r = null, 200 === t) {
                    var n = (new Date).getTime() - o, s = JSON.parse(e);
                    "object" != typeof s && (s = {}), i.emit("finish", s, n)
                } else 
                    i.emit("finish")
            }, s.ontimeout = function() {
                s.close(), i.emit("finish")
            }
        };
        var z = function(e) {
            var i = this, o = function() {
                var t = new N;
                t.protocol = "w-iframe-info-receiver";
                var n = function(e) {
                    if ("string" == typeof e && "m" === e.substr(0, 1)) {
                        var n = JSON.parse(e.substr(1)), o = n[0], s = n[1];
                        i.emit("finish", o, s)
                    } else 
                        i.emit("finish");
                    t.doCleanup(), t = null
                }, o = {
                    _options: {},
                    _didClose: n,
                    _didMessage: n
                };
                t.i_constructor(o, e, e)
            };
            t.body ? o() : n.attachEvent("load", o)
        };
        z.prototype = new s(["finish"]);
        var U = function() {
            var t = this;
            n.delay(function() {
                t.emit("finish", {}, 2e3)
            })
        };
        U.prototype = new s(["finish"]);
        var q = function(t) {
            if (n.isSameOriginUrl(t))
                return new H(t, n.XHRLocalObject);
            switch (n.isXHRCorsCapable()) {
            case 1:
                return new H(t, n.XHRLocalObject);
            case 2:
                return new H(t, n.XDRObject);
            case 3:
                return new z(t);
            default:
                return new U
            }
        }, W = R["w-iframe-info-receiver"] = function(t, e, i) {
            var o = new H(i, n.XHRLocalObject);
            o.onfinish = function(e, n) {
                t._didMessage("m" + JSON.stringify([e, n])), t._didClose()
            }
        };
        W.prototype.doCleanup = function() {};
        var V = M["iframe-eventsource"] = function() {
            var t = this;
            t.protocol = "w-iframe-eventsource", t.i_constructor.apply(t, arguments)
        };
        V.prototype = new N, V.enabled = function() {
            return "EventSource"in e && N.enabled()
        }, V.need_body=!0, V.roundTrips = 3;
        var J = R["w-iframe-eventsource"] = function(t, e) {
            this.run(t, e, "/eventsource", Y, n.XHRLocalObject)
        };
        J.prototype = new O;
        var K = M["iframe-xhr-polling"] = function() {
            var t = this;
            t.protocol = "w-iframe-xhr-polling", t.i_constructor.apply(t, arguments)
        };
        K.prototype = new N, K.enabled = function() {
            return e.XMLHttpRequest && N.enabled()
        }, K.need_body=!0, K.roundTrips = 3;
        var G = R["w-iframe-xhr-polling"] = function(t, e) {
            this.run(t, e, "/xhr", ie, n.XHRLocalObject)
        };
        G.prototype = new O;
        var Q = M["iframe-htmlfile"] = function() {
            var t = this;
            t.protocol = "w-iframe-htmlfile", t.i_constructor.apply(t, arguments)
        };
        Q.prototype = new N, Q.enabled = function() {
            return N.enabled()
        }, Q.need_body=!0, Q.roundTrips = 3;
        var X = R["w-iframe-htmlfile"] = function(t, e) {
            this.run(t, e, "/htmlfile", ne, n.XHRLocalObject)
        };
        X.prototype = new O;
        var Z = function(t, e, n, i) {
            var o = this;
            o.ri = t, o.Receiver = e, o.recv_url = n, o.AjaxObject = i, o._scheduleRecv()
        };
        Z.prototype._scheduleRecv = function() {
            var t = this, e = t.poll = new t.Receiver(t.recv_url, t.AjaxObject), n = 0;
            e.onmessage = function(e) {
                n += 1, t.ri._didMessage(e.data)
            }, e.onclose = function(n) {
                t.poll = e = e.onmessage = e.onclose = null, t.poll_is_closing || ("permanent" === n.reason ? t.ri._didClose(1006, "Polling error (" + n.reason + ")") : t._scheduleRecv())
            }
        }, Z.prototype.abort = function() {
            var t = this;
            t.poll_is_closing=!0, t.poll && t.poll.abort()
        };
        var Y = function(t) {
            var e = this, i = new EventSource(t);
            i.onmessage = function(t) {
                e.dispatchEvent(new o("message", {
                    data: unescape(t.data)
                }))
            }, e.es_close = i.onerror = function(t, s) {
                var r = s ? "user": 2 !== i.readyState ? "network": "permanent";
                e.es_close = i.onmessage = i.onerror = null, i.close(), i = null, n.delay(200, function() {
                    e.dispatchEvent(new o("close", {
                        reason: r
                    }))
                })
            }
        };
        Y.prototype = new i, Y.prototype.abort = function() {
            var t = this;
            t.es_close && t.es_close({}, !0)
        };
        var te, ee = function() {
            if (void 0 === te)
                if ("ActiveXObject"in e)
                    try {
                        te=!!new ActiveXObject("htmlfile")
            } catch (t) {} else 
                te=!1;
            return te
        }, ne = function(t) {
            var i = this;
            n.polluteGlobalNamespace(), i.id = "a" + n.random_string(6, 26), t += ( - 1 === t.indexOf("?") ? "?" : "&") + "c=" + escape(a + "." + i.id);
            var s, r = ee() ? n.createHtmlfile: n.createIframe;
            e[a][i.id] = {
                start: function() {
                    s.loaded()
                },
                message: function(t) {
                    i.dispatchEvent(new o("message", {
                        data: t
                    }))
                },
                stop: function() {
                    i.iframe_close({}, "network")
                }
            }, i.iframe_close = function(t, n) {
                s.cleanup(), i.iframe_close = s = null, delete e[a][i.id], i.dispatchEvent(new o("close", {
                    reason: n
                }))
            }, s = r(t, function() {
                i.iframe_close({}, "permanent")
            })
        };
        ne.prototype = new i, ne.prototype.abort = function() {
            var t = this;
            t.iframe_close && t.iframe_close({}, "user")
        };
        var ie = function(t, e) {
            var n = this, i = 0;
            n.xo = new e("POST", t, null), n.xo.onchunk = function(t, e) {
                if (200 === t)
                    for (; ;) {
                        var s = e.slice(i), r = s.indexOf("\n");
                        if ( - 1 === r)
                            break;
                            i += r + 1;
                            var a = s.slice(0, r);
                            n.dispatchEvent(new o("message", {
                                data: a
                            }))
                    }
            }, n.xo.onfinish = function(t, e) {
                n.xo.onchunk(t, e), n.xo = null;
                var i = 200 === t ? "network": "permanent";
                n.dispatchEvent(new o("close", {
                    reason: i
                }))
            }
        };
        return ie.prototype = new i, ie.prototype.abort = function() {
            var t = this;
            t.xo && (t.xo.close(), t.dispatchEvent(new o("close", {
                reason: "user"
            })), t.xo = null)
        }, M.getUtils = function() {
            return n
        }, M.getIframeTransport = function() {
            return N
        }, M
    }(), "_sockjs_onload"in window && setTimeout(_sockjs_onload, 1), "function" == typeof define && define.amd && define("sockjs", [], function() {
        return t
    }), function() {
        "use strict";
        var e, n, i, o = {}, s=!1, r = 0, a=!1, c=!0, u = [], l=!0;
        BEM.decl("i-xiva", {}, {
            listen: function(t, e, n) {
                var i=!o[t];
                i && (o[t] = {
                    handlers: []
                }), o[t].ts = e, o[t].handlers.push(n), this._init(), i && this._subscribe()
            },
            connect: function() {
                c=!0, this._connect()
            },
            disconnect: function(t) {
                c = t, e && e.close && e.close()
            },
            _sendQueue: function() {
                var t, e, n;
                for (t = 0, e = u.length; e > t; ++t)(n = u[t].handler)(u[t].data);
                u = []
            },
            _toggleFocus: function(t) {
                l = t, l && this._sendQueue()
            },
            _initFocusEvents: function() {
                var t = this;
                $.browser.msie ? $(document).bind("focusout", function() {
                    t._toggleFocus(!1)
                }).bind("focusin", function() {
                    t._toggleFocus(!0)
                }) : $(window).focus(function() {
                    t._toggleFocus(!0)
                }).blur(function() {
                    t._toggleFocus(!1)
                })
            },
            _bindOnUnload: function() {
                var t = this;
                $(window).on("beforeunload", function() {
                    t.disconnect(!1)
                })
            },
            _init: function() {
                s || (s=!0, this._initFocusEvents(), this._bindOnUnload(), this._connect = this._connect.bind(this), this._connect())
            },
            _hasXivaSupport: function() {
                return !!window.WebSocket
            },
            _connect: function() {
                var n = BEM.blocks["i-global"].param("xivaUrl"), i = this;
                this._hasXivaSupport() && n && (e = new t("//" + n + "/xiva", null, {
                    protocols_whitelist: ["websocket"]
                }), e.onopen = function() {
                    i._connected=!0, i._subscribe(), i._checkConnection()
                }, e.onmessage = i._onMessage, e.onclose = function() {
                    i._connected=!1, i._reconnect()
                })
            },
            _reconnect: function() {
                this._stopConnectionTimer(), c && (r++, 5 >= r ? setTimeout(this._connect, 3e3) : 15 >= r ? setTimeout(this._connect, 1e4) : setTimeout(this._connect, 6e4))
            },
            _stopConnectionTimer: function() {
                n && (clearTimeout(n), n = null)
            },
            _checkConnection: function() {
                this._stopConnectionTimer(), n = setTimeout(function() {
                    a && (r = 0)
                }, 5e3)
            },
            _subscribe: function() {
                var t = this;
                i && clearTimeout(i), i = setTimeout(function() {
                    t._subscribeSend()
                }, 100)
            },
            _subscribeSend: function() {
                var t;
                if (this._connected) {
                    t = {
                        subscribe: {}
                    };
                    for (var n in o)
                        o.hasOwnProperty(n) && (t.subscribe[n] = o[n].ts);
                    e.send(JSON.stringify(t))
                }
            },
            _onMessage: function(t) {
                var e, n, i, s, r = $.trim(t.data);
                try {
                    if (e = JSON.parse(r), i = e.ch + "_" + e.key, o[i] && (n = parseInt(e.ts), o[i].ts < n)) {
                        o[i].ts = n, s = o[i].handlers;
                        for (var a = 0, c = s.length; c > a; ++a)
                            l ? s[a](e) : u.push({
                                handler: s[a],
                                data: e
                            })
                        }
                } catch (d) {}
            }
        })
    }()
}(), Array.prototype.filter || (Array.prototype.filter = function(t) {
    var e = [], n = arguments[1] || this;
    if (null === this)
        throw new Error("There is empty context for filter function");
    if ("function" != typeof t)
        throw new Error("You should declare filter function as parameter");
    for (var i = 0, o = this.length; o > i; i++) {
        var s = n[i];
        t.call(n, s, i, this) && e.push(s)
    }
    return e
}), function(t) {
    t.Collection || (t.Collection = function(t) {
        this.storage = [], this.idAttr = t || "id"
    }, t.Collection.prototype = {
        add: function(t) {
            this.storage.push(t), $(this).trigger("append." + t[this.idAttr], [t])
        },
        get: function(t) {
            var e = this;
            return x.filter(this.storage, function(n) {
                return n[e.idAttr] === t
            })
        }
    })
}(ya, this), BEM.DOM.decl("w-rss", {
    onSetMod: {
        js: function() {
            var t = this.domElem, e = this.params.feedID, n = this.params.newsTimestamp, i = t.find(".w-rss__list"), o = t.parent().parent().find("h2.b-widget__title"), s = t.find(".w-rss__gradient"), r = t.find(".w-rss__gradient-up"), a = t.find(".w-rss__informer"), c = t.parent().find(".w-rss__arrow_type_up"), u = t.parent().find(".w-rss__arrow_type_down"), l = 0, d = 0, h = (new Date).getTime(), f = (new Date).getTime(), p = 3e4, m = 3e4, _ = 250, g=!1, b=!0, v=!1, y = function(t, e) {
                var n = "up" === t ? c: u, i = "w-rss__arrow_hovered-" + ("up" === t ? "up" : "down") + "_yes";
                e&&!n.get(0).clickable ? (n.get(0).clickable=!0, n.css({
                    cursor: "pointer"
                }), n.bind("mouseenter", function() {
                    n.addClass(i)
                }).bind("mouseleave", function() {
                    clearTimeout(d), clearInterval(l), n.removeClass(i)
                }).bind("click", function() {
                    $.browser.safari && window.getSelection && window.getSelection().empty();
                    try {
                        Lego.c("stred/pid=132/cid=71588/path=" + S.widget.id + "." + t)
                    } catch (e) {}
                    P.scroll(t)
                }).bind("mousedown", function() {
                    return d = setTimeout(function() {
                        l = setInterval(function() {
                            P.scroll(t)
                        }, 250)
                    }, 400), $.browser.opera?!1 : void 0
                }).bind("mouseup", function() {
                    clearTimeout(d), clearInterval(l)
                })) : !e && n.get(0).clickable && (n.get(0).clickable=!1, n.removeClass(i), n.css({
                    cursor: "default"
                }), clearTimeout(d), clearInterval(l), n.unbind("mouseenter").unbind("mouseleave").unbind("click").unbind("mousedown").unbind("mouseup"))
            }, w = 100, k=!1, E = [], M = [], x = function(t) {
                return wg.isColumnWide ? wg.isColumnWide(t) : 2 == t || 5 == t
            }, C = function(t) {
                return wg.isColumnNarrow ? wg.isColumnNarrow(t) : 4 == t
            }, T = function(t) {
                return wg.isColumnNarrow ? wg.isColumnNarrow(t) : 3 == t
            }, S = function() {
                var e = (t.closest(".b-widget-data").attr("id") + "").substr(3), n = Widget.Framework.get(e), i = n.params.position.split(":")[0];
                return {
                    widget: n,
                    col: i,
                    layout: x(i) ? {
                        horiz: !0
                    }
                    : {
                        vert: !0
                    },
                    mode: {
                        compact: 2 == n._prefs.wht,
                        large: 3 == n._prefs.wht
                    },
                    isHeightAuto: "" === n._prefs.wht || 0 === n._prefs.wht,
                    type: function() {
                        return t.hasClass("w-rss_type_text-img") ? {
                            "text-img": !0
                        } : t.hasClass("w-rss_type_magazine") ? {
                            magazine: !0
                        } : {
                            text: !0
                        }
                    }()
                }
            }(), B = S.widget.news.length, D = function() {
                if (S.layout.vert && t.hasClass("w-rss_layout_horiz") ? (t.removeClass("w-rss_layout_horiz"), t.addClass("w-rss_layout_vert")) : S.layout.horiz && t.hasClass("w-rss_layout_vert") && (t.removeClass("w-rss_layout_vert"), t.addClass("w-rss_layout_horiz")), S.isHeightAuto && (parseInt(S.widget._prefs.n, 10) > 8 ? (t.find(".w-rss__body").addClass("w-rss__body_height_large"), S.mode.large=!0) : S.layout.horiz || T(S.col) || parseInt(S.widget._prefs.n, 10) <= 5 || 5 >= B ? (t.find(".w-rss__body").addClass("w-rss__body_height_compact"), S.mode.compact=!0) : S.layout.vert&&!T(S.col) && (t.find(".w-rss__body").removeClass("w-rss__body_height_compact"), S.mode.compact=!1)), S.type["text-img"]&&!C(S.col)) {
                    var e = S.layout.vert ? "vert": "horiz", n = t.parent().find(".l-rss_type_" + e).clone();
                    return n.appendTo(i), n.find("tbody").html("")
                }
                return i
            }, I = D(), O = function() {
                return I.find(S.type["text-img"]&&!C(S.col) ? "tr" : "div.w-rss__item")
            }, P = function() {
                var e = 1, n = 0, o = 2, a = function(t, e) {
                    var n = O(), i = 0;
                    e || (e = n.length - 1);
                    for (var o = e; o >= t; o -= 1)
                        i += n.eq(o).outerHeight(!0);
                    return i
                };
                return S.type.text ? (e = 1, o = 4) : S.type.magazine && (o = 4), {
                    scroll: function(c) {
                        var u = O();
                        if (K(), S.type.magazine && n >= 0 && u.eq(n).removeClass("w-rss__item_position_first"), "down" === c) {
                            var l = a(u.length - 1 - o), d = a(n);
                            if (d - l <= t.height()&&!k && H(), n < u.length - 1) {
                                n += e, S.type.magazine&&!u.eq(n).hasClass("w-rss__item_previewable_no") && u.eq(n).addClass("w-rss__item_position_first");
                                var h = 0 - Math.floor(u.eq(n + e - 1).position().top);
                                h += q(u, h), i.animate({
                                    top: h
                                }, w), y("up", !0);
                                var f = i.height() - Math.abs(h), p = t.height() - s.height() / 3;
                                (n >= u.length - 1 || p >= f) && y("down", !1)
                            } else 
                                y("down", !1);
                            v && (s.addClass("i-hidden"), r.removeClass("i-hidden"))
                        } else 
                            n > 0 ? (n -= e, S.type.magazine&&!u.eq(n).hasClass("w-rss__item_previewable_no") && u.eq(n).addClass("w-rss__item_position_first"), i.animate({
                                top: 0 - u.eq(n).position().top
                            }, w), y("down", !0), 0 >= n && y("up", !1)) : y("up", !1), v && (s.removeClass("i-hidden"), r.addClass("i-hidden"), n || s.addClass("i-hidden"))
                    },
                    reset: function() {
                        n = 0, i.animate({
                            top: 0
                        }, w)
                    }
                }
            }(), j = function() {
                var e = t.parent().find(".w-rss__item").clone().remove();
                return function(t) {
                    var n = e.clone(), i = t.url;
                    i = i.replace(/^http:/, "").replace(/^https:/, ""), n.find(".w-rss__title").attr("href", i).html(t.title), n.find(".w-rss__date").html(t.date), t.descr ? n.find(".w-rss__description").html(t.descr) : n.find(".w-rss__description").remove();
                    var o = n.find(".w-rss__preview");
                    if (t.img) {
                        var s = S.type.magazine && S.mode.compact && t.imgCompact ? t.imgCompact: t.img;
                        s = s.replace(/^http:/, "").replace(/^https:/, ""), o.attr("href", i), o.css({
                            "background-image": "url(" + s + ")"
                        })
                    } else 
                        n.addClass("w-rss__item_previewable_no"), o.remove();
                    return n.show(), n
                }
            }(), A = t.parent().find(".l-rss_type_vert").find("tr:first").clone().remove(), N = t.parent().find(".l-rss_type_horiz").find("tr:first").clone().remove(), L = function(t) {
                var e = [];
                if (S.type["text-img"]&&!C(S.col)) {
                    var n, i = function(e, n) {
                        if ("" === $(n).html()) {
                            var i = t.shift();
                            i && ($(n).html('<div class="w-rss__item">' + j(i).html() + "</div>"), E.push(i))
                        }
                    }, o = I.find("tr:last").eq(0);
                    for (1 == o.length && o.find("td:not(.l-rss__gap)").each(i); t.length > 0;)
                        n = S.layout.vert ? A.clone() : N.clone(), n.find("td:not(.l-rss__gap)").each(i), $.merge(e, n)
                } else 
                    for (var s = 0, r = t.length - 1; r >= s; s += 1)
                        t[s] && (E.push(t[s]), $.merge(e, j(t[s])));
                I.append(e)
            }, F = function(t, i) {
                k=!0;
                var o = {
                    feed_id: e,
                    num: 12,
                    start: E.length + 1,
                    type: S.widget._prefs.type,
                    "no-text": S.widget._prefs["no-text"]
                };
                "object" == typeof t && $.extend(o, t), $.ajax({
                    url: S.widget.params.src,
                    dataType: "json",
                    data: o,
                    success: function(t) {
                        t && t.news && t.news.length > 0 && (t.tstamp && (n = t.tstamp), i(t))
                    },
                    complete: function() {
                        k=!1, G()
                    }
                })
            }, R = function(t) {
                return t.length > E.length || t[0].title !== E[0].title || t[0].date !== E[0].date?!0 : !1
            }, H = function() {
                F({
                    tstamp: n
                }, function(t) {
                    L(t.news), y("down", !0)
                })
            }, z = function(t) {
                t = t || E, P.reset(), i.html(""), I = D(), E = [], L(t), y("down", !0), V()
            }, U = function(e) {
                for (var n, i = O(), o = 0, r = 0; e > o && 100 > r;) {
                    if (n = $(i[r]).height(), o + n > e) {
                        e - o > n / 2 && (o += n);
                        break
                    }
                    o += n, r++
                }
                100 > r && r > 0 && (v=!0, $(".w-rss__body", t).height(o), s.addClass("i-hidden"))
            }, q = function(e, n) {
                if (!v)
                    return 0;
                for (var i, o = t.height(), s = 0, r = 0; o - n > s && 100 > r;) {
                    if (r > e.length - 1)
                        return 0;
                    i = $(e[r]).height(), s += i, r++
                }
                return 100 > r ? o - n - s : 0
            }, W = function() {
                if (!t.get(0).isInited) {
                    t.get(0).isInited=!0;
                    var e = t.height();
                    if (i.height() > e) {
                        S.type.magazine || setTimeout(function() {
                            U(e)
                        }, S.type["text-img"] ? 700 : 0);
                        var n=!0;
                        t.closest(".b-widget-fixed").mouseenter(function() {
                            u.removeClass("w-rss__arrow_visibility_hidden"), c.removeClass("w-rss__arrow_visibility_hidden")
                        }).mouseleave(function() {
                            u.addClass("w-rss__arrow_visibility_hidden"), c.addClass("w-rss__arrow_visibility_hidden")
                        }).mousemove(function() {
                            n && (n=!1, K(), setTimeout(function() {
                                n=!0
                            }, 250))
                        }), y("down", !0), y("up", !0), y("up", !1)
                    } else 
                        S.isHeightAuto && (t.find(".w-rss__body").removeClass("w-rss__body_height_large"), t.find(".w-rss__body").addClass("w-rss__body_height_compact"), S.mode.compact=!0, S.mode.large=!1, S.isHeightAuto=!1, z(), t.get(0).isInited=!1, W())
                }
            }, V = function() {
                if (setTimeout(W, 450), $(window).load(function() {
                    setTimeout(W, 250)
                }), S.type.magazine) {
                    var t = I.find("div.w-rss__item").eq(0);
                    t.hasClass("w-rss__item_previewable_no") || t.addClass("w-rss__item_position_first")
                }
            }, J = function() {
                return Math.round(2e3 * Math.random() + 3e3)
            }, K = function() {
                f = (new Date).getTime()
            }, G = function() {
                h = (new Date).getTime()
            };
            S.widget._onDropcallback = function(t) {
                var e = t.split(":")[0];
                e != S.col && (x(e)!=!!S.layout.horiz || S.type["text-img"] && (C(e) || C(S.col)) || T(e) || T(S.col)) && (S.col = e, S.layout = x(e) ? {
                    horiz: !0
                } : {
                    vert: !0
                }, z()), S.col = e
            }, $(document).bind("mousemove keydown", function() {
                b&&!g && (b=!1, setTimeout(function() {
                    b=!0
                }, _), (new Date).getTime() - h > p && (a.hasClass("w-rss__informer_visibility_hidden") ? (g=!0, setTimeout(function() {
                    F({
                        start: 0
                    }, function(t) {
                        g=!1, t.news.length > 0 && R(t.news) && ((new Date).getTime() - f > m ? z(t.news) : (a.removeClass("w-rss__informer_visibility_hidden"), M = t.news))
                    })
                }, J())) : (new Date).getTime() - f > m && M.length > 0 && (G(), z(M), M = [], a.addClass("w-rss__informer_visibility_hidden"))))
            }), o.append(a), o.addClass("b-widget__title_type_rss"), L(S.widget.news), V(), $.browser.msie && S.type["text-img"] && setTimeout(function() {
                var e = t.find(".w-rss__preview"), n = t.find("table td").eq(0).width();
                e.each(function(t, e) {
                    var i = $(e);
                    i.hide(), i.css({
                        width: n + "px"
                    }), i.css({
                        heigth: "70px"
                    }), i.show()
                }), t.hide(), t.css({
                    zoom: 1
                }), t.show()
            }, 600), setTimeout(function() {
                0 === t.height() && (t.parent().hide(), setTimeout(function() {
                    t.parent().show()
                }, 200))
            }, 2e3), a.click(function() {
                M.length > 0 && (z(M), M = []), K(), a.addClass("w-rss__informer_visibility_hidden")
            })
        }
    }
}), x.Component.extend({
    id: "check-https-cd",
    init: function() {
        var t = this.data("logger"), e = this.data("checker"), n = this.data("replace");
        if (e && t && n) {
            var i = new Image, o=!1;
            i.src = e, window.setTimeout(function() {
                o || (i.src = "")
            }, 1e3), $(i).on("error", function() {
                $("body").trigger("httpscheck", [!1]), o=!0
            }), $(i).on("load", function() {
                $("body").trigger("httpscheck", [!0]), o=!0
            }), $("body").on("httpscheck", function(e, o) {
                o ? $("form[action='" + n + "']").each(function() {
                    this.action = this.action.replace(/(http)(\:\/\/)/, "$1s$2")
                }) : i.src = t.replace(/%s/, ((new Date).getTime() / 1e3).toFixed())
            })
        }
    }
}), function() {
    "use strict";
    x.amd.require("statlog", function(t) {
        x.Component.extend({
            id: "statlogger",
            init: function() {
                var e = function() {
                    t.logByDocument({
                        type: "click",
                        root: "undefined" != typeof Lego && Lego.params && Lego.params.statRoot ? Lego.params.statRoot: "/testMordaRoot"
                    })
                };
                Lego.params["show-counters"] && (e(), $(window).on("resize", function() {
                    e()
                }))
            }
        })
    })
}(home), $(document).ready(function() {
    var t = home && home.getData && home.getData("awapsbw.data");
    t && window.performance && window.performance.timing && setTimeout(function() {
        var e = window.performance.timing.responseEnd - window.performance.timing.responseStart, n = e <= t.limit ? 101: 99, i=!0;
        e && ("com.tr" === t.domain && (i = window.plp && window.plp.white && window.plp.white[0] && window.plp.white[0].dry), i && ((new Image).src = "//awaps.yandex." + t.domain + "/0/" + t.counter + "/0.gif?setbw=" + n + "&ms=" + e))
    }, 2e3)
}), MordaRT.prototype.siteInDomainList = function(t, e) {
    for (var n, i = 0, o = t.length; o > i; i++)
        if (n = t[i], this.listRE[n] || (this.listRE[n] = new RegExp("^" + n.replace(/\./g, "\\.").replace(/\*/g, "[^.]+") + "$")), e.match(this.listRE[n]))
            return !0;
    return !1
}, MordaRT.prototype.getHostname = function(t) {
    var e = document.createElement("a");
    return e.href = t, e.hostname
}, MordaRT.prototype.getEntries = function(t, e) {
    if (!window.performance ||!window.performance.getEntries)
        return {
            resources: {},
            resourcesCount: 0
        };
    var n, i, o = window.performance.getEntries("resources"), s = {}, r = 0;
    if (o && o.length)
        for (var a = 0, c = o.length; c > a; a++)
            if (n = o[a].initiatorType || "unknown", i = this.getHostname(o[a].name), - 1 !== i.indexOf(".")&&!this.siteInDomainList(t, i)) {
                s[n] || (s[n] = {});
                var u = e ? o[a].name: i;
                s[n][u] ? s[n][u]++ : s[n][u] = 1, r++
            }
    return {
        resources: s,
        resourcesCount: r
    }
}, $(document).ready(function() {
    var t = home && home.getData && home.getData("resources-count.data");
    if (t) {
        var e = new MordaRT, n = ["*.*.yandex.*", "kiks.yandex.ru", "clck.yandex.*", "yabs.yandex.*", "mc.yandex.ru", "*.tns-counter.ru", "*.hit.gemius.pl"];
        setTimeout(function() {
            var i = e.getEntries(n);
            if (i.resourcesCount) {
                var o = {};
                o["resources-" + t.type] = i.resources, window.yaCounter722545 && window.yaCounter722545.params(o)
            }
        }, 11e3)
    }
}), $(document).ready(function() {
    function t(s) {
        var r = {};
        r["time-" + e.content] = {}, r["time-" + e.content][e.type] = {}, r["time-" + e.content][e.type][s] = 1, window.yaCounter722545 && window.yaCounter722545.params(r), n = o >= s ? setTimeout(function() {
            t(s + i)
        }, i) : null
    }
    var e = home && home.getData && home.getData("time-count.data");
    if (e) {
        var n = null, i = e.timeout, o = 3e5;
        t(0), $(document).bind("visibilitychange", function() {
            var t = document.visibilityState || document.webkitVisibilityState || document.msVisibilityState;
            n && "hidden" === t && (clearTimeout(n), n = null)
        }), $.browser.msie ? $(document).bind("focusout", function() {
            clearTimeout(n), n = null
        }) : $(window).blur(function() {
            clearTimeout(n), n = null
        })
    }
}), $(document).ready(function() {
    function t() {
        var t = null;
        "flash" === c.type ? t = c.custom.flashV && (new CAWBrowser).flash >= c.custom.flashV ? e() : n() : "img" === c.type ? t = n() : "html" !== c.type && (t = i());
        var r = o(), u = s();
        r && t ? $.extend(t, r) : r && (t = r), u && t ? $.extend(t, u) : u && (t = u), a(t)
    }
    function e() {
        var t = $("#banner > object:eq(0)").attr("classid"), e = $("#banner > #bc > noscript").length, n = $("#banner").children().length, i = $("#banner > #bc").children().length;
        return t && "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" === t && 1 === e && 2 === n && 1 === i ? null : {
            horizontal: {
                flash: 1
            }
        }
    }
    function n() {
        var t = $("#banner > #bc > noscript").length, e = $("#banner > #bc").children().length, n = "", i = "";
        return c.custom.gif ? n = $("#banner > img").attr("src") : (i = $("#banner > a").attr("href"), n = $("#banner > a > img").attr("src")), n !== c.custom.img ||!c.custom.gif && i !== c.custom.href || 1 !== t || 1 !== e ? c.custom.img ? {
            horizontal: {
                img: 1
            }
        } : {
            horizontal: {
                none: 1
            }
        } : null
    }
    function i() {
        var t = $(".b-line_adv").children().length;
        return 0 === t ? null : {
            horizontal: {
                none: 1
            }
        }
    }
    function o() {
        var t = $("img"), e = [], n=!1;
        if (t.each(function() {
            var t = $(this).parents("a");
            this.src && t.length && e.push(t.attr("href"))
        }), e.length)
            for (var i = 0, o = e.length; o > i; i++)
                if (!r(e[i])) {
                    n=!0;
                    break
                }
        return n ? {
            image: 1,
            ignore: 1
        } : null
    }
    function s() {
        var t = $("html")[0].style.cssText;
        return t ? {
            html: 1
        } : null
    }
    function r(t) {
        if (!t || "#" === t || 0 === $.trim(t).indexOf("javascript"))
            return !0;
        var e = u.getHostname(t);
        return u.siteInDomainList(l, e)
    }
    function a(t) {
        if (t&&!t.ignore) {
            var e = $('<iframe style="display:none"></iframe>');
            $("body").append(e), e.attr("src", c.url)
        }
        var n = {}, i = {
            check: 1
        };
        if (t && t.ignore && delete t.ignore, t) {
            n["consistency-" + c.content] = $.extend(t, i);
            var o = u.getEntries(l), s = u.getEntries(l, !0);
            o.resourcesCount && (n["resources-consistency-" + c.content] = o.resources), s.resourcesCount && o.resources.script && (n["resources-src-consistency-" + c.content] = s.resources.script)
        } else 
            n["consistency-" + c.content] = i;
        window.yaCounter722545 && window.yaCounter722545.params(n)
    }
    var c = home && home.getData && home.getData("consistency.data"), u = new MordaRT;
    if (c) {
        var l = ["*.*.yandex.*", "*.yandex.*", "yandex.*", "yastatic.net", "*.tns-counter.ru", "*.hit.gemius.pl"];
        setTimeout(function() {
            t()
        }, c.timeout)
    }
});
var i18n = i18n || {};
!function(t, e) {
    function n(t) {
        return "string" == typeof t && (t = {
            block: t
        }), t.block + (t.elem ? l + t.elem : "") + (t.modName ? u + t.modName + u + t.modVal : "")
    }
    function i(t) {
        var e = {};
        return t.split(l).forEach(function(t, n) {
            var i = [n ? "elem": "block", "mod", "val"];
            t.split(u).forEach(function(t, n) {
                e[i[n]] = t
            })
        }), e
    }
    function o(t) {
        return t ? c.push(t) : !1
    }
    function s() {
        return c.length && c.pop()
    }
    function r() {
        this._lang = "", this._prj = "lego", this._keyset = "", this._key = ""
    }
    if ("function" == typeof e.I18N && e.I18N._proto)
        return e.I18N;
    var a = {}, c = [], u = "_", l = "__", d = "ru";
    r.prototype = {
        lang: function(t) {
            return this._lang = t, this
        },
        project: function(t) {
            return this._prj = t, this
        },
        keyset: function(t, e) {
            return e && o(this._keyset), this._keyset = n(t), this
        },
        key: function(t) {
            return this._key = t, this
        },
        decl: function(t) {
            var e = i(this._keyset), n = "i-tanker" === e.block ? "tanker": this._prj, o = e.elem || this._keyset, s = this._key;
            n = i18n[n] || (i18n[n] = {}), o = n[o] || (n[o] = {}), o[s] = "function" == typeof t ? t : function() {
                return t
            };
            var r = a[this._lang] || (a[this._lang] = {}), c = r[this._keyset] || (r[this._keyset] = {});
            c[s] = t
        },
        val: function(t, e) {
            var n = a[this._lang] && a[this._lang][this._keyset];
            if (!n)
                return console && console.log && console.log("[Error] keyset: " + this._keyset + " key: " + this._key + " (lang: " + this._lang + ")"), "";
            if (n = n[this._key], !n)
                return "";
            try {
                return "string" == typeof n ? n : e ? n.call(e, t) : n.call(this, t)
            } catch (i) {
                throw "[Error] keyset: " + this._keyset + " key: " + this._key + " (lang: " + this._lang + ")"
            }
        },
        _c: function() {
            return a
        }
    }, e.I18N = function(t) {
        var e = function(t, n, i) {
            return e.keyset(t).key(n, i)
        };
        return e._proto = t, e.project = function(t) {
            return this._proto.project(t), this
        }, e.keyset = function(t) {
            return this._proto.keyset(t, !0), this
        }, e.key = function(t, n) {
            var i, o, r = this._proto;
            return r.lang(this._lang).key(t), i = r.val.call(r, n, e), o = s(), o && r.keyset(o, !1), i
        }, e.decl = function(t, e, n) {
            var i, o = this._proto;
            n || (n = {}), n.lang && o.lang(n.lang), o.keyset(t);
            for (i in e)
                e.hasOwnProperty(i) && o.key(i).decl(e[i]);
            return this
        }, e.lang = function(t) {
            return "undefined" != typeof t && (this._lang = t), this._lang
        }, e.lang(d), e
    }(new r), BEM = e
}(this, "undefined" == typeof BEM ? {} : BEM), BEM.I18N.decl("auth", {
    "fill-input": "Заполните это поле",
    login: "Логин",
    logon: "Войти",
    password: "Пароль",
    register: "Регистрация",
    remember: "Вспомнить пароль",
    temporary: "Чужой компьютер",
    "wrong-characters": "Недопустимый ввод",
    "wrong-keyboard-layout": "Смените раскладку",
    "wrong-password": "Пароль не может совпадать с логином"
}, {
    lang: "ru"
}), BEM.I18N.decl("b-head-search", {
    "my-searches-off": "Персональные подсказки ((выключены))",
    "my-searches-on": "Персональные подсказки ((включены))"
}, {
    lang: "ru"
}), BEM.I18N.decl("b-head-tabs", {
    more: "ещё",
    or: "или"
}, {
    lang: "ru"
}), BEM.I18N.decl("b-head-user", {
    "change-password": "Сменить пароль",
    passport: "Паспорт",
    profiles: "Мои профили"
}, {
    lang: "ru"
}), BEM.I18N.decl("b-keyboard-loader", {
    keyboard: "Клавиатура"
}, {
    lang: "ru"
}), BEM.I18N.decl("b-keyboard-popup", {
    close: "закрыть"
}, {
    lang: "ru"
}), BEM.I18N.decl("i-services", {
    404: "404",
    adresa: "Адреса",
    advertising: "Реклама",
    afisha: "Афиша",
    all: "Все сервисы",
    api: "API",
    appsearch: "Приложения",
    auto: "Авто",
    avia: "Авиабилеты",
    aziada: "Азиада",
    ba: "Баян",
    backapv: "Партнер Я.Карт",
    balance: "Баланс",
    bar: "Бар",
    "bar-ie": "Бар для ИЕ",
    "bar-ie9": "Бар для ИЕ9",
    bayan: "Баннеры Яндекса",
    blogs: "Блоги",
    books: "Книги",
    browser: "Браузер",
    calendar: "Календарь",
    captcha: "ой...",
    catalogwdgt: "Каталог виджетов",
    chrome: "Хром с поиском Яндекса",
    city: "Города",
    cityday: "День города",
    collection: "Коллекция",
    company: "Компания",
    contest: "Contest",
    desktop: "Персональный поиск",
    direct: "Директ",
    "direct.market": "Маркет",
    disk: "Диск",
    ege: "ЕГЭ",
    expert: "Эксперт",
    feedback: "Обратная связь",
    ff: "ФФ с поиском Яндекса",
    fotki: "Фотки",
    fresh: "Свежее",
    games: "Игрушки",
    geocontext: "Геоконтекст",
    goroda: "Города",
    help: "Помощь",
    i: "Мои сервисы",
    ie: "ИЕ с поиском Яндекса",
    images: "Картинки",
    "images-com": "Картинки",
    interests: "Интересы",
    internet: "Интернет",
    keyboard: "Клавиатура",
    kraski: "Краски",
    kuda: "Куда все идут",
    large: "Яндекс для слабовидящих",
    legal: "Правовые документы",
    lenta: "Лента",
    libra: "Библиотека",
    literacy: "Неделя борьбы за грамотность",
    local: "Локальная сеть",
    lost: "Незабудки",
    love: "День взаимного тяготения — 13 августа",
    mail: "Почта",
    maps: "Карты",
    "maps-wiki": "Народная карта",
    market: "Маркет",
    "market.advertising": "Маркет",
    metrika: "Метрика",
    metro: "Метро",
    mobile: "Мобильный",
    moikrug: "Мой Круг",
    money: "Деньги",
    museums: "Дни исторического и культурного наследия",
    music: "Музыка",
    "music-partner": "Музыка: статистика",
    nahodki: "Мои находки",
    nano: "Нано",
    newhire: "Наниматор",
    news: "Новости",
    oauth: "Авторизация",
    online: "Онлайн",
    openid: "OpenID",
    opera: "Opera Software",
    opinion: "Цитаты",
    partners: "Рекламная сеть",
    partnersearch: "Поиск для партнеров",
    passport: "Паспорт",
    pdd: "Почта для домена",
    peoplesearch: "Люди",
    perevod: "Перевод",
    probki: "Пробки",
    pulse: "блоги: пульс",
    punto: "Punto switcher",
    pvo: "Ответы",
    rabota: "Работа",
    ramazan: "",
    rasp: "Расписания",
    realty: "Недвижимость",
    referats: "Рефераты",
    rk: "Есть вопросы?",
    root: "Яндекс.Олимпиада для Unix администраторов",
    school: "Школа",
    search: "Поиск",
    server: "Сервер",
    site: "Поиск для сайта",
    slovari: "Словари",
    so: "Самооборона",
    social: "Социализм",
    soft: "Программы",
    sport: "Спорт",
    sprav: "Справочник",
    start: "Стартовая страница",
    stat: "Статистика",
    subs: "Подписки",
    taxi: "Такси",
    terms: "Разговорник",
    tests: "Тесты и опросы",
    tickets: "Билеты",
    time: "Яндекс.Время",
    toster: "Тосты",
    translate: "Перевод",
    tune: "Настройки",
    tv: "Телепрограмма",
    uslugi: "Услуги",
    video: "Видео",
    "video-com": "Видео",
    wdgt: "Виджеты",
    weather: "Погода",
    webmaster: "Вебмастер",
    widgets: "Виджеты",
    wordstat: "Статистика",
    wow: "Я.ру",
    www: "Поиск",
    xmlsearch: "XML",
    yaca: "Каталог",
    yamb: "Медийные баннеры",
    zakladki: "Закладки"
}, {
    lang: "ru"
}), BEM.I18N.decl("i-tanker__dynamic", {
    gender: function(t) {
        return function(t) {
            return t[t.gender]
        }.call(this, t)
    },
    plural: function(t) {
        return function(t) {
            var e = isNaN(parseInt(t.count)) ? 0: t.count, n = e%10, i = e%100;
            return 1 == n && 11 != i ? t.one : n > 1 && 5 > n && (10 > i || i > 20) ? t.some : t.many
        }.call(this, t)
    },
    plural_adv: function(t) {
        return function(t) {
            var e = isNaN(parseInt(t.count)) ? 0: t.count;
            return 0 === e ? t.none : this.keyset("i-tanker__dynamic").key("plural", {
                count: t.count,
                one: t.one,
                some: t.some,
                many: t.many
            })
        }.call(this, t)
    },
    toggle: function(t) {
        return function(t) {
            return Boolean(t.condition) ? t["true"] : t["false"]
        }.call(this, t)
    }
}, {
    lang: "ru"
}), BEM.I18N.decl("suggest2", {
    fact: "Факты",
    group: "Группа подсказок",
    nah: "Вы искали",
    nav: "Сайт",
    text: "Поиск",
    traffic: "Пробки",
    weather: "Погода"
}, {
    lang: "ru"
}), BEM.I18N.decl("suggest2-item", {
    "quick-answer": "Быстрый ответ"
}, {
    lang: "ru"
}), BEM.I18N.decl("user", {
    enter: "Войти",
    exit: "Выйти",
    letter: function(t) {
        return this.keyset("i-tanker__dynamic").key("plural_adv", {
            count: t.count,
            one: "новое письмо",
            some: "новых письма",
            many: "новых писем",
            none: "нет новых писем"
        })
    },
    letter_compose: "Написать письмо",
    passport: "Паспорт",
    upload_files: "Загрузить файлы",
    tune: "Настройка"
}, {
    lang: "ru"
}), BEM.I18N.lang("ru");
var BEMHTML = function() {
    var t, e = function(e) {
        function n() {
            function e() {
                if ("no-follow" === this.ctx.link)
                    return void 0;
                var t = this._links[this.ctx.link];
                return __r47 = this.ctx, this.ctx = t, __r48 = n.call(s), this.ctx = __r47, __r48
            }
            function i(t, e, n) {
                if (0 === e.length)
                    return void 0;
                if (Array.isArray(n)) {
                    for (var i = t, o = 0; o < n.length - 1; o++)
                        i = i[n[o]];
                    n = i[n[o]]
                }
                for (var s, r = t, o = 0; o < e.length - 1; o++)
                    r = r[e[o]];
                return s = r[e[o]], r[e[o]] = n, s
            }
            function o(t, e) {
                return (t || "") + "__" + (e || "")
            }
            var s = this;
            if (!!this.elem==!1 && "b-link" === this.block && "attrs" === this._mode) {
                for (var r = this.ctx, a = ["title", "target"], c = typeof r.url, u = {
                    href: "undefined" === c || "string" === c ? r.url: (c = [], __r86 = this._buf, this._buf = c, __r87 = this._mode, this._mode = "", __r88 = this.ctx, this.ctx = r.url, __r89 = n.call(s), this._buf = __r86, this._mode = __r87, this.ctx = __r88, c.join(""))
                }; c = a.pop();)
                    r[c] && (u[c] = r[c]);
                return u
            }
            if (!!this.elem==!1 && "b-link" === this.block && "tag" === this._mode)
                return "a";
            if ("popup" === this.block && "tail" === this.elem && "tag" === this._mode)
                return "i";
            if (!!this.elem==!1 && "popup" === this.block&&!(240546821 !== this.__$anflg)==!1 && "default" === this._mode) {
                var l = this.ctx.mods || {};
                if (l.theme || (l.theme = "ffffff"), l.autoclosable || (l.autoclosable = "yes"), l.adaptive || (l.adaptive = "yes"), l.animate || (l.animate = "yes"), this.ctx.mods = l, this.ctx.zIndex) {
                    var d = this.ctx.attrs || (this.ctx.attrs = {});
                    d.style || (d.style = ""), d.style += "z-index:" + this.ctx.zIndex
                }
                this.ctx.content = [{
                    elem: "under",
                    mods: this.ctx.underMods
                }, this.ctx.content];
                var h = this.__$anflg;
                this.__$anflg = 240546821;
                var f = this.ctx;
                this.ctx = this.ctx;
                var p = this._mode;
                return this._mode = "", n.call(s), this.ctx = f, this._mode = p, void(this.__$anflg = h)
            }
            if ("dropdown" === this.block && "popup" === this.elem&&!(595960171 !== this.__$anflg)==!1 && "default" === this._mode) {
                var m = [{
                    block: "dropdown",
                    elem: "popup"
                }
                ], _ = this.ctx;
                _.mix && (m = m.concat(_.mix));
                var g = this.__$anflg;
                this.__$anflg = 595960171;
                var b = this.ctx;
                this.ctx = {
                    block: "popup",
                    mix: m,
                    mods: _.mods,
                    attrs: _.attrs,
                    js: _.js,
                    content: [{
                        elem: "tail"
                    }, {
                        elem: "content",
                        content: _.content
                    }
                    ]
                };
                var v = this._mode;
                return this._mode = "", n.call(s), this.ctx = b, this._mode = v, void(this.__$anflg = g)
            }
            if (!!this.elem==!1 && "dropdown" === this.block && "mix" === this._mode)
                return [{
                    mods: {
                        action: "closed"
                    }
                }
                ];
            if (!!this.elem==!1 && "dropdown" === this.block && "js" === this._mode)
                return !0;
            if ("suggest2-item" === this.block&&!(this.mods && "link" === this.mods.interact)==!1&&!("nav" === this.mods.type || "icon" === this.mods.type)==!1 && "text" === this.elem && "mix" === this._mode)
                return {
                    mods: {
                        type: "url"
                    }
                };
            if (!!this.elem==!1 && "suggest2-item" === this.block&&!(this.mods && "link" === this.mods.interact)==!1 && "content" === this._mode)
                return {
                    elem: "link",
                    url: this.ctx.url,
                    target: this.ctx.target || "_blank",
                    content: this.ctx.content
                };
            if ("suggest2-item" === this.block && "bullet" === this.elem && "attrs" === this._mode)
                return {
                    "aria-hidden": !0
                };
            if ("suggest2-item" === this.block && "bullet" === this.elem && "tag" === this._mode)
                return "span";
            if ("suggest2" === this.block && "a11y" === this.elem && "tag" === this._mode)
                return "span";
            if ("suggest2-item" === this.block && "fact" === this.elem && "content" === this._mode)
                return [{
                    block: "suggest2",
                    elem: "a11y",
                    content: BEM.I18N("suggest2-item", "quick-answer") + ": "
                }, this.xmlEscape(this.ctx.content)];
            if ("suggest2-item" === this.block && "link" === this.elem && "attrs" === this._mode)
                return {
                    href: this.ctx.url,
                    target: this.ctx.target
                };
            if ("suggest2-item" === this.block && "link" === this.elem && "tag" === this._mode)
                return "a";
            if ("suggest2-item" === this.block && "icon" === this.elem&&!this.ctx.png==!1&&!this.ctx.svg==!1 && "attrs" === this._mode)
                return {
                    style: 'background-image:url("' + this.ctx[BEM.blocks["i-ua"] && BEM.blocks["i-ua"].svg ? "svg": "png"] + '")'
                };
            if ("suggest2-item" === this.block && "icon" === this.elem&&!this.ctx.svg==!1 && "attrs" === this._mode)
                return {
                    style: 'background-image:url("' + this.ctx.svg + '")'
                };
            if ("suggest2-item" === this.block && "icon" === this.elem&&!this.ctx.png==!1 && "attrs" === this._mode)
                return {
                    style: 'background-image:url("' + this.ctx.png + '")'
                };
            if ("suggest2-item" === this.block && "icon" === this.elem && "tag" === this._mode)
                return "span";
            if ("suggest2" === this.block&&!(this.mods && "normal" === this.mods.theme)==!1 && "content" === this.elem && "tag" === this._mode)
                return "ul";
            if ("suggest2-item" === this.block && "highlight" === this.elem && "content" === this._mode)
                return this.xmlEscape(this.ctx.content);
            if ("suggest2-item" === this.block && "highlight" === this.elem && "bem" === this._mode)
                return !1;
            if ("suggest2-item" === this.block && "highlight" === this.elem && "tag" === this._mode)
                return "b";
            if ("suggest2-item" === this.block && "fact" === this.elem && "tag" === this._mode)
                return "span";
            if ("suggest2-item" === this.block && "text" === this.elem && "content" === this._mode)
                return Array.isArray(this.ctx.content) ? this.ctx.content.map(function(t) {
                    return "string" == typeof t ? this.xmlEscape(t) : t
                }, this) : this.xmlEscape(this.ctx.content);
            if ("suggest2-item" === this.block && "text" === this.elem && "tag" === this._mode)
                return "span";
            if (!!this.elem==!1 && "suggest2-item" === this.block && "js" === this._mode)
                return !0;
            if (!!this.elem==!1 && "suggest2-item" === this.block && "tag" === this._mode)
                return "li";
            if (!!this.elem==!1 && "suggest2" === this.block && "mix" === this._mode)
                return [{
                    block: "suggest2-detect",
                    js: !0
                }
                ];
            if ("suggest2" === this.block && "items" === this.elem && "tag" === this._mode)
                return "ul";
            if (!!this.elem==!1 && "suggest2" === this.block && "js" === this._mode)
                return !0;
            if ("i-bem" === this.block && "i18n" === this.elem && "default" === this._mode) {
                if (!this.ctx)
                    return "";
                var y = this.ctx, w = y.keyset, k = y.key, E = y.params || {};
                if (!w&&!k)
                    return "";
                if (y.content) {
                    var M;
                    E.content = (M = [], __r76 = this._buf, this._buf = M, __r77 = this._mode, this._mode = "", __r78 = this.ctx, this.ctx = y.content, __r79 = n.call(s), this._buf = __r76, this._mode = __r77, this.ctx = __r78, M.join(""))
                }
                return void this._buf.push(BEM.I18N(w, k, E))
            }
            if ("input" === this.block && "clear" === this.elem && "content" === this._mode&&!!this.ctx.content==!1)
                return "";
            if ("input" === this.block && "clear" === this.elem && "tag" === this._mode)
                return "span";
            if ("input" === this.block && "control" === this.elem&&!!this.mods.clear==!1&&!(236227158 !== this.__$anflg)==!1 && "default" === this._mode) {
                var x = this.__$anflg;
                this.__$anflg = 236227158;
                var C = this.ctx;
                this.ctx = {
                    elem: "box",
                    tag: "span",
                    content: [this.ctx, {
                        elem: "clear"
                    }
                    ]
                };
                var T = this._mode;
                return this._mode = "", n.call(s), this.ctx = C, this._mode = T, void(this.__$anflg = x)
            }
            if ("input" === this.block && "control" === this.elem && "attrs" === this._mode) {
                var S = {
                    id: this._inputId,
                    name: this._name
                };
                return (this._value || 0 === this._value) && (S.value = this._value), this.mods.disabled && (S.disabled = "disabled"), S
            }
            if ("input" === this.block && "control" === this.elem && "tag" === this._mode)
                return "input";
            if ("input" === this.block && "ahead-hint" === this.elem && "attrs" === this._mode)
                return {
                    readonly: "readonly",
                    tabindex: "-1"
                };
            if ("input" === this.block && "ahead-hint" === this.elem && "tag" === this._mode)
                return "input";
            if ("input" === this.block && "ahead-hint" === this.elem && "mix" === this._mode)
                return [{
                    block: "input",
                    elem: "control"
                }
                ];
            if ("input" === this.block && "ahead-filler" === this.elem && "tag" === this._mode)
                return "span";
            if ("input" === this.block && "ahead" === this.elem && "content" === this._mode)
                return [{
                    elem: "ahead-filler"
                }, {
                    elem: "ahead-hint"
                }, {
                    elem: "ahead-hintholder"
                }
                ];
            if ("input" === this.block&&!!this.elem==!1 && "default" === this._mode&&!!(this.ctx.mods || {}).theme==!1)
                return __r68 = this.ctx, __r69 = __r68.mods, __r68.mods = this.ctx.mods || {}, __r70 = this.ctx.mods, __r71 = __r70.theme, __r70.theme = "normal", __r72 = n.call(s), __r68.mods = __r69, __r70.theme = __r71, __r72;
            if ("input" === this.block&&!!this.elem==!1 && "js" === this._mode)
                return "gradient" == this.mods.popup ? {
                    popupMods: {
                        gradient: "yes"
                    }
                } : !0;
            if ("input" === this.block&&!!this.elem==!1 && "default" === this._mode&&!(857832037 !== this.__$anflg)==!1&&!!this._inputId==!1) {
                var B = this.__$anflg;
                this.__$anflg = 857832037;
                var D = this._inputId;
                this._inputId = this.ctx.id || this.generateId();
                var I = this._name;
                this._name = this.ctx.name || "";
                var O = this._value;
                this._value = "undefined" != typeof this.ctx.value && null !== this.ctx.value ? this.ctx.value : "";
                var P = this._inputLink;
                this._inputLink=!0;
                var $ = this._disabled;
                return this._disabled = this.mods.disabled, n.call(s), this._inputId = D, this._name = I, this._value = O, this._inputLink = P, this._disabled = $, void(this.__$anflg = B)
            }
            if ("input" === this.block&&!!this.elem==!1 && "js" === this._mode)
                return !0;
            if ("input" === this.block&&!!this.elem==!1 && "tag" === this._mode)
                return "span";
            if ("" === this._mode&&!this._==!1&&!!this._.cleverSubstring==!1)
                return this._.cleverSubstring = function(t, e, n) {
                    return t.length > e + n ? t.substring(0, e - 1) + "…" : t
                }, void n.call(s);
            if (!!this.elem==!1 && "b-page" === this.block && "js-params" === this._mode) {
                var j, A = this["i-global"], N = {}, L = {
                    block: "i-global",
                    js: N
                };
                for (j in A)
                    A.hasOwnProperty(j) && (__r58 = this._mode, this._mode = "public-params", __r59 = this.block, this.block = "i-global", __r60 = this.elem, this.elem = j, __r61 = n.call(s), this._mode = __r58, this.block = __r59, this.elem = __r60, __r61) && (N[j] = A[j]);
                return L
            }
            if ("i-global" === this.block && "public-params" === this._mode&&!this.elem==!1)
                return {
                    id: 1,
                    lang: 1,
                    tld: 1,
                    "content-region": 1,
                    "user-region": 1,
                    login: 1,
                    displayName: 1,
                    index: 1,
                    yandexuid: 1,
                    "passport-host": 1,
                    "pass-host": 1,
                    "passport-msg": 1,
                    "static-host": 1,
                    "lego-static-host": 1,
                    "social-host": 1,
                    clck: 1,
                    "click-host": 1,
                    "export-host": 1,
                    "i-host": 1,
                    "social-retpath": 1,
                    "lego-path": 1,
                    sid: 1,
                    retpath: 1,
                    uid: 1
                }
                [this.elem] ||!1;
            if ("i-global" === this.block && "default" === this._mode && "lego-static-host" === this.elem)
                return "//yandex.st/lego/2.10-126";
            if ("i-global" === this.block && "default" === this._mode && "export-host" === this.elem)
                return "//export.yandex.ru";
            if ("i-global" === this.block && "default" === this._mode && "social-host" === this.elem)
                return "//social.yandex.ru";
            if ("i-global" === this.block && "default" === this._mode && "pass-host" === this.elem)
                return "//pass.yandex.ru";
            if ("i-global" === this.block && "default" === this._mode && "passport-host" === this.elem)
                return "https://passport.yandex.ru";
            if ("i-global" === this.block && "default" === this._mode && "click-host" === this.elem)
                return "//clck.yandex.ru";
            if ("i-global" === this.block && "default" === this._mode && "content-region" === this.elem)
                return "ru";
            if ("i-global" === this.block && "default" === this._mode && "tld" === this.elem)
                return "ru";
            if ("i-global" === this.block && "default" === this._mode && "lang" === this.elem)
                return "ru";
            if ("i-global" === this.block && "default" === this._mode&&!this.elem==!1)
                return "";
            if (!!this.elem!=!1 || "i-global" !== this.block || "default" !== this._mode) {
                if (!!this.elem==!1 && "i-global" === this.block && "env" === this._mode)
                    return {};
                if ("" === this._mode&&!!this["i-global"]==!1) {
                    for (var F, R = {}, H = ["lang", "tld", "content-region", "click-host", "passport-host", "pass-host", "social-host", "export-host", "login", "lego-static-host"]; F = H.shift();)
                        R[F] = (__r51 = this._mode, this._mode = "default", __r52 = this.block, this.block = "i-global", __r53 = this.elem, this.elem = F, __r54 = n.call(s), this._mode = __r51, this.block = __r52, this.elem = __r53, __r54);
                    return this["i-global"] = this._.extend(R, (__r55 = this._mode, this._mode = "env", __r56 = this.block, this.block = "i-global", __r57 = n.call(s), this._mode = __r55, this.block = __r56, __r57)), void n.call(s)
                }
                if (!!this.elem==!1 && "i-ua" === this.block&&!(989736432 !== this.__$anflg)==!1 && "content" === this._mode) {
                    var z = (__r49 = this.__$anflg, this.__$anflg = 989736432, __r50 = n.call(s), this.__$anflg = __r49, __r50);
                    return z += [";(function(d,e,c,r,n,w,v,f){", "e=d.documentElement;", 'c="className";', 'r="replace";', 'n="createElementNS";', 'f="firstChild";', 'w="http://www.w3.org/2000/svg";', 'e[c]+=" i-ua_svg_"+(!!d[n]&&!!d[n](w,"svg").createSVGRect?"yes":"no");', 'v=d.createElement("div");', 'v.innerHTML="<svg/>";', 'e[c]+=" i-ua_inlinesvg_"+((v[f]&&v[f].namespaceURI)==w?"yes":"no");', "})(document);"].join("")
                }
                if (!!this.elem==!1 && "i-ua" === this.block && "content" === this._mode)
                    return [";(function(d,e,c,r){", "e=d.documentElement;", 'c="className";', 'r="replace";', 'e[c]=e[c][r]("i-ua_js_no","i-ua_js_yes");', 'if(d.compatMode!="CSS1Compat")', 'e[c]=e[c][r]("i-ua_css_standart","i-ua_css_quirks")', "})(document);"].join("");
                if (!!this.elem==!1 && "i-ua" === this.block && "bem" === this._mode)
                    return !1;
                if (!!this.elem==!1 && "i-ua" === this.block && "tag" === this._mode)
                    return "script";
                if ("content" === this._mode)
                    return this.ctx.content;
                if ("mix" === this._mode)
                    return void 0;
                if ("bem" === this._mode)
                    return void 0;
                if ("jsAttr" === this._mode)
                    return void 0;
                if ("js" === this._mode)
                    return void 0;
                if ("cls" === this._mode)
                    return void 0;
                if ("attrs" === this._mode)
                    return void 0;
                if ("tag" === this._mode)
                    return void 0;
                if (!this.ctx==!1&&!!this._.isSimple(this.ctx)==!1&&!this.ctx.link==!1) {
                    if (!t ||!this._cacheLog)
                        return e.call(this);
                    var U = this._buf.slice(this._cachePos).join("");
                    this._cachePos = this._buf.length, this._cacheLog.push(U, {
                        log: this._localLog.slice(),
                        link: this.ctx.link
                    });
                    var q = e.call(this);
                    return this._cachePos = this._buf.length, q
                }
                if (!t==!1&&!this.ctx==!1&&!this.ctx.cache==!1) {
                    var W;
                    if (W = t.get(this.ctx.cache)) {
                        var V = this._links;
                        this.ctx.links && (this._links = this.ctx.links);
                        for (var J = 0; J < W.log.length; J++)
                            if ("string" != typeof W.log[J]) {
                                var K, G = W.log[J];
                                K = G.log.map(function(t) {
                                    return {
                                        key: t[0],
                                        value: i(this, t[0], t[1])
                                    }
                                }, this).reverse();
                                var Q = this.ctx, X = Q.cache;
                                Q.cache = null;
                                var Z = this._cacheLog;
                                this._cacheLog = null;
                                var Y = this.ctx, te = Y.link;
                                Y.link = G.link, n.call(s), Q.cache = X, this._cacheLog = Z, Y.link = te, K.forEach(function(t) {
                                    i(this, t.key, t.value)
                                }, this)
                            } else 
                                this._buf.push(W.log[J]);
                        return this._links = V, W.res
                    }
                    var ee, ne = [], ie = this.ctx, oe = ie.cache;
                    ie.cache = null;
                    var se = this._cachePos;
                    this._cachePos = this._buf.length;
                    var re = this._cacheLog;
                    this._cacheLog = ne;
                    var ae = this._localLog;
                    this._localLog = [], ee = n.call(s);
                    var ce = this._buf.slice(this._cachePos).join("");
                    return ce && ne.push(ce), ie.cache = oe, this._cachePos = se, this._cacheLog = re, this._localLog = ae, t.set(this.ctx.cache, {
                        log: ne,
                        res: ee
                    }), ee
                }
                if ("default" !== this._mode) {
                    if ("" === this._mode&&!this._.isSimple(this.ctx)==!1) {
                        this._listLength--;
                        var ue = this.ctx;
                        return void((ue && ue!==!0 || 0 === ue) && this._buf.push(ue))
                    }
                    if ("" === this._mode&&!!this.ctx==!1)
                        return void this._listLength--;
                    if ("" === this._mode&&!this._.isArray(this.ctx)==!1) {
                        var le = this.ctx, de = le.length, he = 0, fe = this.position, pe = this._notNewList;
                        for (pe ? this._listLength += de - 1 : (this.position = 0, this._listLength = de)
                            , this._notNewList=!0;
                        de > he;
                        ) {
                            var me = le[he++], _e = this.ctx;
                            this.ctx = null === me ? "" : me, n.call(s), this.ctx = _e
                        }
                        return void(pe || (this.position = fe))
                    }
                    if ("" === this._mode) {
                        var ge = this.ctx.block, be = this.ctx.elem, ve = this._currBlock || this.block;
                        this.ctx || (this.ctx = {});
                        var ye = this._mode;
                        this._mode = "default";
                        var we = this._links;
                        this._links = this.ctx.links || this._links;
                        var ke = this.block;
                        this.block = ge || (be ? ve : void 0);
                        var Ee = this._currBlock;
                        this._currBlock = ge || be ? void 0 : ve;
                        var Me = this.elem;
                        this.elem = this.ctx.elem;
                        var xe = this.mods;
                        this.mods = (ge ? this.ctx.mods : this.mods) || {};
                        var Ce = this.elemMods;
                        return this.elemMods = this.ctx.elemMods || {}, this.block || this.elem ? this.position = (this.position || 0) + 1 : this._listLength--, n.call(s), this._mode = ye, this._links = we, this.block = ke, this._currBlock = Ee, this.elem = Me, this.mods = xe, void(this.elemMods = Ce)
                    }
                } else {
                    var Te, Se = this, Be = Se.BEM, De = this.ctx, Ie = this._buf;
                    if (__r8 = this._mode, this._mode = "tag", __r9 = n.call(s), this._mode = __r8, Te = __r9, "undefined" != typeof Te || (Te = De.tag), "undefined" != typeof Te || (Te = "div"), Te) {
                        var Oe, Pe;
                        this.block && De.js!==!1 && (__r12 = this._mode, this._mode = "js", __r13 = n.call(s), this._mode = __r12, Pe = __r13, Pe = Pe ? this._.extend(De.js, Pe===!0 ? {} : Pe) : De.js===!0 ? {} : De.js, Pe && ((Oe = {})[Be.INTERNAL.buildClass(this.block, De.elem)] = Pe)), Ie.push("<", Te);
                        var $e = (__r14 = this._mode, this._mode = "bem", __r15 = n.call(s), this._mode = __r14, __r15);
                        "undefined" != typeof $e || ($e = "undefined" != typeof De.bem ? De.bem : De.block || De.elem);
                        var je = (__r16 = this._mode, this._mode = "cls", __r17 = n.call(s), this._mode = __r16, __r17);
                        je || (je = De.cls);
                        var Ae = De.block && Oe;
                        if ($e || je) {
                            if (Ie.push(' class="'), $e) {
                                Be.INTERNAL.buildClasses(this.block, De.elem, De.elemMods || De.mods, Ie);
                                var Ne = (__r18 = this._mode, this._mode = "mix", __r19 = n.call(s), this._mode = __r18, __r19);
                                if (De.mix && (Ne = Ne ? Ne.concat(De.mix) : De.mix), Ne) {
                                    var Le = {};
                                    Le[o(this.block, this.elem)]=!0, this._.isArray(Ne) || (Ne = [Ne]);
                                    for (var Fe = 0; Fe < Ne.length; Fe++) {
                                        var Re = Ne[Fe], He = Re.block || Re.elem, ze = Re.block || Re._block || Se.block, Ue = Re.elem || Re._elem || Se.elem;
                                        if (He && Ie.push(" "), Be.INTERNAL[He ? "buildClasses": "buildModsClasses"](ze, Re.elem || Re._elem || (Re.block ? void 0 : Se.elem), Re.elemMods || Re.mods, Ie), Re.js && ((Oe || (Oe = {}))[Be.INTERNAL.buildClass(ze, Re.elem)] = Re.js===!0 ? {} : Re.js, Ae || (Ae = ze&&!Re.elem)), He&&!Le[o(ze, Ue)]) {
                                            Le[o(ze, Ue)]=!0;
                                            var qe = (__r20 = this.block, this.block = ze, __r21 = this.elem, this.elem = Ue, __r22 = this._mode, this._mode = "mix", __r23 = n.call(s), this.block = __r20, this.elem = __r21, this._mode = __r22, __r23);
                                            if (qe)
                                                for (var We = 0; We < qe.length; We++) {
                                                    var Ve = qe[We];
                                                    (Ve.block || Ve.elem) && Le[o(Ve.block, Ve.elem)] || (Ve._block = ze, Ve._elem = Ue, Ne.splice(Fe + 1, 0, Ve))
                                                }
                                            }
                                    }
                                }
                            }
                            je && Ie.push($e ? " " : "", je), Ae && Ie.push(" i-bem"), Ie.push('"')
                        }
                        if (Oe) {
                            var Je = (__r26 = this._mode, this._mode = "jsAttr", __r27 = n.call(s), this._mode = __r26, __r27);
                            Ie.push(" ", Je || "onclick", '="return ', this._.attrEscape(JSON.stringify(Oe)), '"')
                        }
                        var Ke = (__r28 = this._mode, this._mode = "attrs", __r29 = n.call(s), this._mode = __r28, __r29);
                        if (Ke = this._.extend(Ke, De.attrs)) {
                            var Ge;
                            for (Ge in Ke)
                                void 0 !== Ke[Ge] && Ie.push(" ", Ge, '="', this._.attrEscape(Ke[Ge]), '"')
                        }
                    }
                    if (this._.isShortTag(Te))
                        Ie.push("/>");
                    else {
                        Te && Ie.push(">");
                        var Qe = (__r30 = this._mode, this._mode = "content", __r31 = n.call(s), this._mode = __r30, __r31);
                        if (Qe || 0 === Qe) {
                            var $e = this.block || this.elem, Xe = this._notNewList;
                            this._notNewList=!1;
                            var Ze = this.position;
                            this.position = $e ? 1 : this.position;
                            var Ye = this._listLength;
                            this._listLength = $e ? 1 : this._listLength;
                            var tn = this.ctx;
                            this.ctx = Qe;
                            var en = this._mode;
                            this._mode = "", n.call(s), this._notNewList = Xe, this.position = Ze, this._listLength = Ye, this.ctx = tn, this._mode = en
                        }
                        Te && Ie.push("</", Te, ">")
                    }
                }
            } else {
                var nn, on, sn, rn = this.ctx.params || {}, an = this["i-global"], cn = rn.tld && rn.tld !== an.tld;
                cn && (nn = rn.tld, on = "tr" === nn ? "yandex.com.tr" : "yandex." + nn, sn =- 1 != ["ua", "by", "kz"].indexOf(nn) ? "yandex.ru" : on, an["content-region"] = nn, an["click-host"] = "//clck." + sn, an["passport-host"] = "https://passport." + sn, an["pass-host"] = "//pass." + on, an["social-host"] = "//social." + on, an["export-host"] = "//export." + on);
                for (var un in rn)
                    an[un] = rn[un]
            }
        }
        return !function(t, e) {
            if (e.I18N)
                return void 0;
            t.BEM = e;
            var n = e.I18N = function(t, e) {
                return e
            };
            n.keyset = function() {
                return n
            }, n.key = function(t) {
                return t
            }, n.lang = function() {
                return void 0
            }
        }(this, "undefined" == typeof BEM ? {} : BEM), !function() {
            function t(t, e) {
                this.ctx = null === typeof t ? "" : t, this.apply = e, this._buf = [], this._ = this, this._start=!0, this._mode = "", this._listLength = 0, this._notNewList=!1, this.position = 0, this.block = void 0, this.elem = void 0, this.mods = void 0, this.elemMods = void 0
            }
            var i = {}, o = Object.prototype.toString, s = {
                area: 1,
                base: 1,
                br: 1,
                col: 1,
                command: 1,
                embed: 1,
                hr: 1,
                img: 1,
                input: 1,
                keygen: 1,
                link: 1,
                meta: 1,
                param: 1,
                source: 1,
                wbr: 1
            };
            !function(t, e) {
                function n(t, e, n) {
                    n.push(s, t, s, e)
                }
                function i(t, e, i, o) {
                    o.push(t), i && n(e, i, o)
                }
                function o(t, o, s, a, c) {
                    i(t, e, e, c), c.push(r, o), a && n(s, a, c)
                }
                var s = "_", r = "__", a = "[a-zA-Z0-9-]+";
                t.INTERNAL = {
                    NAME_PATTERN: a,
                    MOD_DELIM: s,
                    ELEM_DELIM: r,
                    buildModPostfix: function(t, e, i) {
                        var o = i || [];
                        return n(t, e, o), i ? o : o.join("")
                    },
                    buildClass: function(t, n, s, r, a) {
                        var c = typeof s;
                        if ("string" == c ? "string" != typeof r && (a = r, r = s, s = n, n = e) : "undefined" != c ? (a = s, s = e) : n && "string" != typeof n && (a = n, n = e), !(n || s || a))
                            return t;
                        var u = a || [];
                        return n ? o(t, n, s, r, u) : i(t, s, r, u), a ? u : u.join("")
                    },
                    buildModsClasses: function(t, e, n, s) {
                        var r = s || [];
                        if (n) {
                            var a;
                            for (a in n)
                                if (n.hasOwnProperty(a)) {
                                    var c = n[a];
                                    null != c && (c = n[a] + "", c && (r.push(" "), e ? o(t, e, a, c, r) : i(t, a, c, r)))
                                }
                        }
                        return s ? r : r.join("")
                    },
                    buildClasses: function(t, n, s, r) {
                        var a = r || [];
                        return n ? o(t, n, e, e, a) : i(t, e, e, a), this.buildModsClasses(t, n, s, r), r ? a : a.join("")
                    }
                }
            }(i);
            var r = function() {
                var t = {
                    '"': "&quot;",
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;"
                }, e = function(e) {
                    return t[e] || e
                };
                return function(t) {
                    return t = new RegExp(t, "g"), function(n) {
                        return ("" + n).replace(t, e)
                    }
                }
            }();
            t.prototype.isArray = function(t) {
                return "[object Array]" === o.call(t)
            }, t.prototype.isSimple = function(t) {
                var e = typeof t;
                return "string" === e || "number" === e || "boolean" === e
            }, t.prototype.isShortTag = function(t) {
                return s.hasOwnProperty(t)
            }, t.prototype.extend = function(t, e) {
                if (!t ||!e)
                    return t || e;
                var n, i = {};
                for (n in t)
                    t.hasOwnProperty(n) && (i[n] = t[n]);
                for (n in e)
                    e.hasOwnProperty(n) && (i[n] = e[n]);
                return i
            }, t.prototype.identify = function() {
                var t = 0, e = i.__id =+ new Date, n = "__" + e, o = function() {
                    return "uniq" + e + ++t
                };
                return function(t, e) {
                    return t ? e || t[n] ? t[n] : t[n] = o() : o()
                }
            }(), t.prototype.xmlEscape = r("[&<>]"), t.prototype.attrEscape = r('["&<>]'), t.prototype.BEM = i, t.prototype.isFirst = function() {
                return 1 === this.position
            }, t.prototype.isLast = function() {
                return this.position === this._listLength
            }, t.prototype.generateId = function() {
                return this.identify(this.ctx)
            }, e.apply = t.apply = function() {
                var e = new t(this, n);
                return e.apply(), e._buf.join("")
            }
        }(), e
    }("undefined" == typeof exports ? {} : exports);
    return function(n) {
        return n || (n = {}), t = n.cache, e.apply.call([this])
    }
}();
"undefined" == typeof exports || (exports.BEMHTML = BEMHTML);
