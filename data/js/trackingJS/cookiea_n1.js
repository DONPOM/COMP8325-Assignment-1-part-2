splashRadar._JSINIT=!0;
splashRadar._JSINITTime = (new Date).getTime();
splashRadar._JSLOADTime = (new Date).getTime() - splashRadar._JSLOADTime;
splashRadar("JSLoad", 1);
splashRadar("JSInit");
Splash.getUniqueId = function() {
    var a = 0;
    return function() {
        return a++
    }
}();
extend(mr, {
    getLf1Url: function(a, b) {
        var b = (b || location.href).replace(/https?:\/\/(\S+\.)*?mail.ru/i, "//lf1.mail.ru"), f =- 1 === b.indexOf("?") ? "?" : "&";
        if (a)
            for (var d in a)
                a.hasOwnProperty(d) && (b += f + d + "=" + encodeURIComponent(a[d]), f = "&");
        return b
    },
    tag: function(a, b, f) {
        if (f && document.querySelectorAll)
            return (b || document).querySelectorAll(a + "." + f);
        a = (b || document).getElementsByTagName(a);
        if (f) {
            for (var b = [], d = 0; d < a.length; d++) {
                var k = a[d];
                mr.hasClass(k, f) && b.push(k)
            }
            return b
        }
        return a
    },
    toggleClass: function(a, b,
    f, d) {
        !1 === d || void 0 === d && mr.hasClass(a, b) ? (mr.removeClass(a, b), f && mr.addClass(a, f)) : (mr.addClass(a, b), f && mr.removeClass(a, f))
    },
    hasParent: function(a, b) {
        var f;
        f = mr.isNode(b) ? function(d, a) {
            return d == a
        } : function(d, a) {
            return mr.hasClass(d, a)
        };
        for (var d = a.parentNode; d;) {
            if (f(d, b))
                return d;
            d = d.parentNode
        }
        return !1
    },
    isNode: function(a) {
        return a && 1 == a.nodeType
    },
    preventDefault: function(a) {
        a && a.preventDefault ? a.preventDefault() : window.event.returnValue=!1
    },
    s_cookie: __PH.cookie.s,
    data: function(a, b, f) {
        if (!a)
            return null;
        if (void 0 === f)
            return a.getAttribute("data-" + b);
        a.setAttribute("data-" + b, f)
    },
    proxy: function(a, b) {
        return function() {
            return a.apply(b, arguments)
        }
    },
    position: function(a) {
        if (a.getBoundingClientRect) {
            var a = a.getBoundingClientRect(), b = document.body, f = document.documentElement;
            return {
                left: a.left + (window.pageXOffset || f.scrollLeft || b.scrollLeft) - (f.clientLeft || b.clientLeft || 0),
                top: a.top + (window.pageYOffset || f.scrollTop || b.scrollTop) - (f.clientTop || b.clientTop || 0)
            }
        }
        b = parseFloat(a.offsetTop);
        for (f = parseFloat(a.offsetLeft); a =
        a.offsetParent;)
            b += parseFloat(a.offsetTop), f += parseFloat(a.offsetLeft);
        return {
            left: f,
            top: b
        }
    },
    getJson: function(a, b, f, d) {
        new Ajax(a, {
            json: !0,
            params: b,
            callback: f,
            error: d
        })
    },
    _jsonpId: 0,
    JSONP: function(a) {
        var b = a.url, f = a.params, d = a.getName || "callback", k = a.callback, e = a.error, a = a.timeout, c = document.createElement("script"), i = "__jsonp_" + mr._jsonpId++;
        window[i] = function(d) {
            clearTimeout(c._timeout);
            k(d)
        };
        a && (c._timeout = setTimeout(function() {
            e()
        }, a));
        f || (f = {});
        f[d] = i;
        f.rnd = (new Date).getTime() + Math.random();
        f =
        mr.formatQuery(f);
        c.src = b + (0 < b.indexOf("?") ? "&" : "?") + f;
        document.getElementsByTagName("head")[0].appendChild(c)
    },
    formatQuery: function(a) {
        var b = [], f;
        for (f in a)
            a.hasOwnProperty(f) && b.push(encodeURIComponent(f) + "=" + encodeURIComponent(a[f]));
        return b.join("&")
    },
    parseJson: function(a) {
        if ("string" !== typeof a ||!a)
            return null;
        try {
            return window.JSON && window.JSON.parse ? window.JSON.parse(a) : (new Function("return " + a))()
        } catch (b) {
            return window.logError && logError(b, "parseJson"), null
        }
    },
    plural: function(a) {
        return 1 ==
        a%10 && 11 != a%100 ? 0 : 2 <= a%10 && 4 >= a%10 && (10 > a%100 || 20 <= a%100) ? 1 : 2
    },
    placeSlot: function(a) {
        var b = mr.id("slot_" + a), a = mr.id("placeholder_" + a);
        b && a && (a.innerHTML = b.innerHTML)
    },
    placeSlots: function() {
        for (var a = 0; a < arguments.length; a++)
            mr.placeSlot(arguments[a])
    },
    extendClass: function(a, b) {
        var f = function() {};
        f.prototype = b.prototype;
        a.prototype = new f;
        a.prototype.constructor = a
    },
    vendor: /webkit/i.test(navigator.appVersion) ? "webkit": /firefox/i.test(navigator.userAgent) ? "Moz": /trident/i.test(navigator.userAgent) ? "ms":
    "opera"in window ? "O": ""
});
mr.hasTransform = "transform"in document.documentElement.style || mr.vendor + "Transform"in document.documentElement.style;
function Ajax(a, b) {
    var f = this.getTransport();
    if (f) {
        var d = b.method || "get", k = b.json ||!1, e = b.params || {}, c = b.callback || function() {}, i = b.error || function() {};
        "get" == d && b.params && (a += ( - 1 != a.indexOf("?") ? "&" : "?") + mr.formatQuery(e));
        f.onreadystatechange = function() {
            var d;
            if (4 === f.readyState) {
                if (200 === f.status)
                    if (k) {
                        var a;
                        try {
                            a = mr.parseJson(f.responseText)
                        } catch (e) {
                            window.logError && logError(e, "readyState"), d = e, window.console && window.console.error && console.error(e)
                        }
                        d || c(a)
                    } else 
                        c(f.responseText);
                if (200 !== f.status ||
                d)
                    i(d instanceof Error && d || Error("request error"))
            }
        };
        f.open(d, a, !0);
        f.send("get" == d ? null : mr.formatQuery(e))
    }
}
Ajax.prototype = {
    getTransport: window.ActiveXObject&&!window.XMLHttpRequest ? function() {
        try {
            return new window.ActiveXObject("Microsoft.XMLHTTP")
        } catch (a) {
            window.logError && logError(a, "getTransport_IE")
        }
    }
    : function() {
        try {
            return new window.XMLHttpRequest
        } catch (a) {
            window.logError && logError(a, "getTransport_XHR")
        }
    }
};
(function() {
    function a(a) {
        this._context = a || this;
        this._mixedIn = []
    }
    function b(a, d, b) {
        this.eventName = a;
        this.target = d;
        this.data = b
    }
    a.prototype.emit = function(a, d) {
        var k;
        if (!this._events ||!this._events[a] ||!(k = this._events[a]))
            return !1;
        for (var e = 0, c = k.length; e < c; e++) {
            var i = k[e];
            "function" === typeof i && i.apply(this._context, [new b(a, this._context, d)])
        }
        return !0
    };
    a.prototype.addListener = function(a, d) {
        if ("function" !== typeof d)
            throw Error("invalid argument");
        this._events || (this._events = {});
        this._events[a] || (this._events[a] =
        []);
        this._events[a].push(d);
        return this
    };
    a.prototype.on = a.prototype.addListener;
    a.prototype.once = function(a, d) {
        function b() {
            this.removeListener(a, b);
            d.apply(this._context, arguments)
        }
        if ("function" !== typeof d)
            throw Error("invalid argument");
        this.on(a, b);
        return this
    };
    a.prototype.removeListener = function(a, d) {
        if ("function" !== typeof d)
            throw Error("invalid argument");
        if (!this._events[a])
            return this;
        var b = this._events[a], e = b.indexOf(d);
        if (0 > e)
            return this;
        b.splice(e, 1);
        0 === b.length && delete this._events[a];
        return this
    };
    a.prototype.mixin = function(a) {
        a.bind = this.bind.bind(this);
        a.unbind = this.unbind.bind(this);
        a.dispatch = this.dispatch.bind(this);
        this._mixedIn.push(a)
    };
    a.prototype.bind = function(a, d) {
        this.on(a, d);
        return this
    };
    a.prototype.unbind = function(a, d) {
        this.removeListener(a, d)
    };
    a.prototype.dispatch = function(a, d) {
        this.emit.apply(this, arguments)
    };
    a.prototype.destroy = function() {
        for (var a = 0, d = this._mixedIn.length; a < d; a++) {
            var b = this._mixedIn[a];
            b.bind = b.unbind = b.dispatch = void 0
        }
        this._mixedIn = this._events =
        this._context = void 0
    };
    Splash.EventEmitter = a
})();
(function() {
    function a(a, b, e) {
        this._elem = a;
        this._name = b;
        this._fn = e;
        this._attached=!1;
        this._handler = mr.proxy(this._fn, this)
    }
    var b = 0, f = {};
    a.addListener = function(d, k, e) {
        if (Array.isArray(d)) {
            for (var c = 0, i = d.length; c < i; c++)
                d[c] && a.addListener(d[c], k, e);
            return this
        }
        k = Array.isArray(k) ? k : [k];
        c=!1;
        e.uuid && (c=!0);
        var j = e.uuid || b;
        e.uuid = j;
        j = f[j] = function(a) {
            var a = Event._fix(a || window.event), c = e.call(d, a);
            !1 === c && (a.preventDefault(), a.stopPropagation());
            return c
        };
        c || b++;
        c = 0;
        for (i = k.length; c < i; c++)
            document.addEventListener ?
            d.addEventListener(k[c], j, !1) : d.attachEvent("on" + k[c], j);
        return this
    };
    a.on = a.addListener;
    a.removeListener = function(d, b, e) {
        if (Array.isArray(d)) {
            for (var c = 0, i = d.length; c < i; c++)
                d[c] && a.removeListener(d[c], b, e);
            return this
        }
        b = Array.isArray(b) ? b : [b];
        c = 0;
        for (i = b.length; c < i; c++)
            document.removeEventListener ? d.removeEventListener(b[c], f[e.uuid], !1) : d.detachEvent("on" + b[c], f[e.uuid]);
        return this
    };
    a.once = function(d, b, e) {
        function c(i) {
            e.call(this, i);
            a.removeListener(d, b, c)
        }
        this.on(d, b, c);
        return this
    };
    a.prototype =
    {
        attach: function() {
            this._attached || (this._attached=!0, this._attach())
        },
        detach: function() {
            this._attached && (this._detach(), this._attached=!1)
        },
        _attach: function() {
            a.on(this._elem, this._name, this._handler)
        },
        _detach: function() {
            a.removeListener(this._elem, this._name, this._handler)
        },
        destroy: function() {
            this.detach();
            this._elem = this._name = this._fn = this._handler = null
        }
    };
    Event = {
        _fixHooks: {
            key: {
                props: ["charCode", "keyCode"],
                fix: function(a, b) {
                    a.which || (a.which = b.charCode ? b.charCode : b.keyCode);
                    return a
                }
            },
            mouse: {
                props: "clientX clientY offsetX offsetY pageX pageY screenX screenY".split(" "),
                fix: function(a, b) {
                    var e, c, i = b.button, f = b.fromElement;
                    null == a.pageX && null != b.clientX && (e = a.target.ownerDocument || document, c = e.documentElement, e = e.body, a.pageX = b.clientX + (c && c.scrollLeft || e && e.scrollLeft || 0) - (c && c.clientLeft || e && e.clientLeft || 0), a.pageY = b.clientY + (c && c.scrollTop || e && e.scrollTop || 0) - (c && c.clientTop || e && e.clientTop || 0));
                    !a.relatedTarget && f && (a.relatedTarget = f === a.target ? b.toElement : f);
                    !a.which && void 0 !== i && (a.which = i & 1 ? 1 : i & 2 ? 3 : i & 4 ? 2 : 0);
                    a.wheelDelta = b.wheelDelta;
                    a.wheelDeltaX = b.wheelDeltaX;
                    a.wheelDeltaY = b.wheelDeltaY;
                    return a
                }
            },
            touch: {
                props: ["touches", "targetTouches", "changedTouches"],
                fix: function(a) {
                    return a
                }
            }
        },
        _props: "altKey bubbles cancelable ctrlKey currentTarget eventPhase shiftKey target timeStamp which".split(" "),
        _fix: function(a) {
            var b = {
                preventDefault: function() {
                    a.preventDefault ? a.preventDefault() : a.returnValue=!1
                },
                stopPropagation: function() {
                    a.stopPropagation ? a.stopPropagation() : a.cancelBubble=!0
                }
            }, e = a.type, c = /^(key|mouse|touch)/.exec(e);
            c&&-1 !== ["key", "mouse", "touch"].indexOf(c[1]) &&
            (e = c[1]);
            "click" === e && (e = "mouse");
            for (var e = this._fixHooks[e] || {}, c = e.props ? this._props.concat(e.props) : this._props, i = c.length; i;) {
                var f = c[--i];
                b[f] = a[f]
            }
            b.target && 3 === b.target.nodeType && (b.target = b.target.parentNode);
            b.target || (b.target = a.srcElement || document);
            e.fix && e.fix(b, a);
            b.originalEvent = a;
            b.type = a.type;
            b.timeStamp = a && a.timeStamp || (new Date).valueOf();
            return b
        }
    };
    Splash.DOMEvent = a
})();
(function(a, b) {
    function f(b) {
        function c(a, b) {
            return z.add(I, a, b)
        }
        function e(a, b) {
            a && z.rm(I, a, b)
        }
        function l(a, b) {
            a && z.fire(I, a, b)
        }
        function i() {
            return {
                current: R.curr,
                recent: R.rcnt,
                previous: R.prev,
                value: F,
                data: J,
                fullData: P,
                focused: L(N),
                method: ga,
                selected: L(ea),
                result: oa(ea),
                itemData: $(ea),
                hovered: L(Q),
                items: n.mkarr(S)
            }
        }
        function u() {
            P = J = F = null;
            j()
        }
        function j() {
            Q = N = ea = ga = null;
            ha=!1
        }
        function w(a) {
            a = a.currentTarget;
            t.mousePreview ? (a = L(a), pa(a), ma(a)) : (N = null, ta(a))
        }
        function q(a) {
            ha && w(a)
        }
        function k() {}
        function B() {}
        function r() {}
        function D(a) {
            ha || w(a);
            ha=!0
        }
        function E(a) {
            a.stopPropagation();
            a.preventDefault();
            a = L(a.currentTarget);
            - 1 != a && (ga = "mouse", ua(a))
        }
        function v() {
            if (/^(?:relative|absolute|fixed)$/i.test(n.css(G, "position"))) {
                var b = fa({}, A.getBoundingClientRect());
                b.left += a.pageXOffset + (t.shiftX || 0);
                b.top += a.pageYOffset + A.offsetHeight + (t.shiftY || 0);
                n.css(G, "left", b.left + "px");
                n.css(G, "top", b.top + "px")
            }
        }
        function W(a) {
            a = a ? "on" : "off";
            n.attr(A, "autocomplete", a);
            n.attr(A, "autocapitalize", a);
            n.attr(A, "autocorrect",
            a);
            aa && (g(aa), aa = null);
            n.hasFocus(A) && (A.blur(), aa = s(function() {
                (!d.activeElement ||!/^(?:a|input|textarea|select|button|)$/i.test(d.activeElement.tagName) || n.hasFocus(A)) && A.focus();
                aa && (g(aa), aa = null)
            }, 25))
        }
        function ba() {
            g(ia);
            ia = null
        }
        function M() {
            ia = s(function() {
                va ? M() : (g(ia), ia = null)
            }, 50)
        }
        function T() {
            va=!0
        }
        function sa() {
            va=!1;
            z.fire(A, "focus")
        }
        function U() {
            g(qa);
            qa = null
        }
        function ca(a) {
            if (229 != a.keyCode) {
                var b = a.keyCode, c = a.shiftKey, g=!a.ctrlKey&&!c;
                switch (!0) {
                case g && (38 == b || 40 == b)&&!V():
                    a.preventDefault()
                }
                if (!qa) {
                    qa =
                    s(U, t.keynavDelay);
                    c=!1;
                    switch (!0) {
                    case 13 == b:
                        !V() && N && (ga = "keyboard", b = ua(N, !0), (!t.autoSubmit ||!1 === b) && a.preventDefault());
                        break;
                    case 27 == b:
                        V() || (X(), ra(), s(ra, 15), a.preventDefault());
                        break;
                    case g && 38 == b:
                        V() || (wa( - 1), c=!0);
                        break;
                    case g && 39 == b:
                        !V() && (null != P && null != J && null != F && N && oa(N) !== F) && xa(oa(N));
                        break;
                    case g && 40 == b:
                        V() ? null != P && (null != J && null != F) && N ? (ya(), c=!0) : xa() : (wa(1), c=!0)
                    }
                    c && t.preview && (N ? ma(L(N)) : ra())
                }
            }
        }
        function da() {
            U()
        }
        function L(a) {
            var b;
            "number" === typeof a ? b = a : a && a.nodeType &&
            (b = n.indexOf(S, a));
            return "number" === typeof b&&!isNaN(b) ? b : - 1
        }
        function za(a) {
            var b;
            if ("number" === typeof a)
                b = S[a];
            else if (a && a.nodeType&&~(a = n.indexOf(S, a)))
                b = S[a];
            return b || null
        }
        function $(a) {
            a = L(a);
            return J && null != J[a] ? J[a] : null
        }
        function ta(a) {
            var b = za(a), c = t.hover;
            !Q&&!b || Q === b || (c && ("string" === typeof c ? (Q && n.rmCls(Q, c), b && n.addCls(b, c)) : O(c) && (Q && (a = L(Q), c.call(I, !1, Q, $(a), a, F)), b && (a = L(b), c.call(I, !0, b, $(a), a, F)))), Q = b)
        }
        function pa(a) {
            var a = L(a), b = za(a), c = $(a);
            ta(a);
            l(y.focus, [c, a, F]);
            N = b;
            l(y.focusEnd,
            [c, a, F])
        }
        function wa(a) {
            var b = N || Q, c = b ? L(b): - 1, b = S.length - 1, a = c + a;
            - 1 > a ? a = b : a > b && (a =- 1);
            pa(a)
        }
        function oa(a) {
            var a = L(a), b = $(a);
            return null != b ? t.result.call(I, b, a, P, F) : null
        }
        function ma(a) {
            var a = L(a), b = $(a), a = null != b ? t.result(b, a, P, F): F;
            R.setIgnore(a)
        }
        function ra() {
            R.setIgnore(F);
            A.value = F
        }
        function ua(b, c) {
            b = L(b);
            ea =- 1 == b ? null : za(b);
            pa(b);
            $(b);
            var g = t.select, e, H=!1;
            l(y.select, i());
            X();
            ra();
            O(g) && (e = g.call(I, i()));
            if (!1 !== e && (g = L(b), (g = oa(g)) && R.setIgnore(g), Y && t.autoSubmit))
                if (c)
                    H=!0;
                else {
                    var f=!1;
                    d.createEvent ?
                    (z.natOne(Y, "submit", function(b) {
                        b = b || a.event;
                        f = z.natIsDefaultPrevented(b);
                        z.Event.prototype.preventDefault.call({
                            origEvt: b
                        })
                    }), g = d.createEvent("HTMLEvents"), g.initEvent("submit", !1, !0), Y.dispatchEvent(g)) : d.createEventObject && (g = d.createEventObject(), f=!Y.fireEvent("onsubmit", g));
                    !1 === f && Y.submit()
                }
            H ? "addEventListener"in d ? z.natOne(Y, "submit", Aa) : z.natOne(Y, "submit", function() {
                s(Aa, 50)
            }) : Aa();
            return e
        }
        function Aa() {
            var a = t.unselect;
            O(a) && a.call(I, i());
            l(y.selectEnd, i());
            ea = ga = null
        }
        function xa(a) {
            a ?
            R.fireVal(a, !0) : R.fire(!0)
        }
        function V() {
            var a=!0, b = t.switcher, c = t.switchChecker;
            b ? "string" === typeof b ? a=!n.hasCls(G, t.switcher) : O(b) && O(c) && (a=!c.call(I)) : a = "none" === n.css(G, "display");
            return a
        }
        function ya() {
            if (V()) {
                l(y.open, [F]);
                var a = t.switcher;
                a ? "string" === typeof a ? n.addCls(G, a) : O(a) && a.call(I, !0, G) : n.css(G, "display", "block");
                t.correction && v();
                ha=!1;
                l(y.openEnd, [F])
            }
        }
        function X() {
            if (!V()) {
                l(y.close, [F]);
                var a = t.switcher;
                a ? "string" === typeof a ? n.rmCls(G, a) : O(a) && a(!1, G) : n.css(G, "display", "none");
                l(y.closeEnd, [F])
            }
        }
        function na(a) {
            a.target !== A && X()
        }
        function Ba() {
            v()
        }
        function Da() {
            ja && (l(y.enable), z.add(d, "click", na), z.add(G, "click", C), z.add(A, "focus", ba), z.add(A, "blur", M), z.add(G, "mousedown", T), z.add(G, "mouseup", sa), z.add(A, "keydown", ca), z.add(A, "keyup", da), t.correction && z.add(a, "resize", Ba), W(!1), R.init(), ja=!1, l(y.enableEnd))
        }
        function Ea() {
            ja || (l(y.disable), X(), R.off(), Fa.abortAll(), W(!0), z.rm(d, "click", na), z.rm(G, "click", C), z.rm(A, "focus", ba), z.rm(A, "blur", M), z.rm(G, "mousedown", T), z.rm(G,
            "mouseup", sa), z.rm(A, "keydown", ca), z.rm(A, "keyup", da), t.correction && Ba && z.rm(a, "resize", Ba), n.empty(ka), S.length = 0, j(), ja=!0, l(y.disableEnd))
        }
        function Ca(a) {
            try {
                console.log(a.type, arguments)
            } catch (b) {}
        }
        function Ga(a, b) {
            var g = arguments.length;
            if (0 == g)
                return t;
            if (1 == g && "object" !== typeof a)
                return n.walker(t, a);
            "object" !== typeof a && (g = {}, g[a] = b, a = g);
            g = a;
            if (null == g)
                g = null;
            else {
                var l = {};
                (function Ha(a, b) {
                    null != a && (n.isObj(a) ? n.objEach(a, function(a, c) {
                        Ha(a, (b ? b + "." : "") + c)
                    }) : b && (l[b] = a))
                })(g);
                g = l
            }
            g && n.objEach(g,
            function(a, b) {
                var g = n.walker(t, b, a);
                if (/(?:^on[A-Z])/.test(b.substr(0, 3)) && O(a)) {
                    var l = b.charAt(2).toLowerCase() + b.slice(3);
                    if (l = f.evt[l])
                        g && e(l, g), c(l, a)
                }
            })
        }
        var I = this;
        if (!(I instanceof f))
            return new f(b);
        var z = f.Event, R = {
            run: !1,
            did: null,
            prev: null,
            rcnt: null,
            curr: null,
            ignr: null,
            init: function() {
                this.run || (this.curr = this.prev = A.value, this.on())
            },
            handler: function(a) {
                var b = this.curr = A.value, g = this.prev;
                if (a || b !== g) {
                    !a && null == this.ignr && (this.rcnt = g);
                    this.prev = b;
                    if (a || null == this.ignr || this.ignr !== b) {
                        l(y.valueChange,
                        [b, g, this.rcnt]);
                        var a = String(b), c = a.length, g=!0;
                        if (t.valMin && c < t.valMin || t.valMax && c > t.valMax)
                            g=!1;
                        c = t.valFilter;
                        g && c && (g = c instanceof RegExp ? c.test(a) : O(c) ? c(a) : c === a);
                        g ? (l(y.passFilter, [b]), Fa.send(b, function(a, b) {
                            if (b === A.value) {
                                var c = t.dataGet;
                                F = b;
                                J = (P = a) && O(c) ? c.call(I, P) : P;
                                c=!1;
                                if (!P ||!J ||!J.length || t.min && t.min > J.length)
                                    c=!0;
                                if (c)
                                    l(y.rejectData, [a, b]), u(), X();
                                else {
                                    l(y.acceptData, [a, b]);
                                    n.empty(ka);
                                    S.length = 0;
                                    j();
                                    l(y.render, [J, F]);
                                    var c = t.max ? Math.min(J.length, t.max): J.length, g = 0, e, H, d = O(t.item) ?
                                    t.item: n.tmpl(t.item);
                                    t.itemExtraData && n.isObj(t.itemExtraData) && (H = t.itemExtraData);
                                    for (; g < c; g++) {
                                        e = J[g];
                                        e = {
                                            itemData: e,
                                            fullData: P,
                                            index: g,
                                            value: F
                                        };
                                        H && n.ext(e, H);
                                        (e = d.call(I, e)) && "string" === typeof e && (e = n.cre(e));
                                        if (!e ||!e.nodeType)
                                            break;
                                        z.add(e, "click", E);
                                        z.add(e, "mouseenter", q);
                                        z.add(e, "mouseleave", k);
                                        z.add(e, "mousedown", B);
                                        z.add(e, "mouseup", r);
                                        z.add(e, "mousemove", D);
                                        S.push(e);
                                        ka.appendChild(e)
                                    }
                                    l(y.renderEnd, [S, J, F]);
                                    ya()
                                }
                            }
                        }, function(a, b) {
                            b === A.value && (u(), X())
                        })) : (l(y.failFilter, [b]), u(), X())
                    }
                    this.ignr =
                    null
                }
            },
            fire: function(a) {
                this.off();
                this.handler(a);
                this.on()
            },
            fireVal: function(a, b) {
                this.off();
                A.value = a;
                this.handler(b);
                this.on()
            },
            on: function() {
                if (!this.run) {
                    var a = this;
                    this.run=!0;
                    this.did = x(function() {
                        a.handler()
                    }, t.delay)
                }
            },
            off: function() {
                this.run && (this.run=!1, K(this.did), this.did = null)
            },
            setIgnore: function(a) {
                A.value !== a && (this.ignr = A.value = a)
            }
        }, la = {
            c: {},
            i: 0,
            set: function(a, b) {
                l(y.setCache, [a, b]);
                t.cchLimit && this.i + 1 >= t.cchLimit && this.flush();
                this.c[a] = b;
                this.i++;
                l(y.setCacheEnd, [a, b, this.i])
            },
            get: function(a) {
                l(y.getCache, [a]);
                var b = this.has(a) ? this.c[a]: null;
                l(y.getCacheEnd, [a, b]);
                return b
            },
            has: function(a) {
                return null != this.c[a] && n.hasOwn(this.c, a)
            },
            flush: function() {
                l(y.flushCache);
                var a = this.c, b;
                for (b in a)
                    n.hasOwn(a, b) && delete a[b];
                this.i = 0;
                l(y.flushCacheEnd)
            }
        }, Fa = {
            act: [],
            actVals: {},
            isAct: function(a) {
                return !arguments.length?!!this.act.length : !!this.actVals[a] && n.hasOwn(this.actVals, a)
            },
            add: function(a, b) {
                var c = this.act, g = t.reqMax || t.ajax.stackSize;
                g && (c.length && c.length >= g) && this.rm(c[0]);
                this.actVals[b] = a;
                return c.push(a)
            },
            rm: function(a) {
                if (a) {
                    a.abort();
                    var b = n.indexOf(this.act, a);
                    ~b && this.act.splice(b, 1);
                    var c = this.actVals;
                    for (b in c)
                        if (c[b] === a) {
                            delete c[b];
                            break
                        }
                    return this.act.length
                }
            },
            abortAll: function() {
                for (var a = this.act, b = this.actVals, c = a.length; c--;)
                    a[c] && a[c].abort();
                a.length = 0;
                for (c in b)
                    delete b[c]
            },
            stack: function() {
                return n.mkarr(this.act)
            },
            send: function(a, b, c) {
                if (!a)
                    return !1;
                var g = this, e, H = {}, d = t.url, i = {
                    query: a
                };
                if (t.cch && la.has(a))
                    return b && b(la.get(a), a), !0;
                if (this.isAct(a))
                    return l(y.blockRequest,
                    ["running", a]), !1;
                "object" === typeof d ? d = n.url(n.objFormat(d, i)) : (i.query = p(i.query), d = n.format(d, i));
                t.ajax && fa(!0, H, t.ajax);
                var i = {
                    dataType: t.reqDataType,
                    timeout: t.reqTimeout,
                    jsonp: t.callbackParam,
                    scriptCharset: t.scriptCharset,
                    data: t.reqData
                }, u;
                for (u in i)
                    null != i[u] && (H[u] = i[u]);
                e = f.Ajax(d, H);
                e.on("success", function(c, H) {
                    l(y.successRequest, [H, a]);
                    l(y.completeRequest, [H, a]);
                    var d = t.dataFilter;
                    O(d) && (H = d.call(g, H, a));
                    g.rm(e) || l(y.stopRequest, [a]);
                    t.cch && la.set(a, H);
                    b && b(H, a)
                });
                e.on("error", function(b,
                H, d) {
                    l(y.errorRequest, [d, a]);
                    l(y.completeRequest, [d, a]);
                    g.rm(e) || l(y.stopRequest, [a]);
                    c && c(d, a)
                });
                1 == g.add(e, a) && l(y.startRequest, [a]);
                l(y.sendRequest, [a]);
                e.send();
                return !0
            }
        }, P = null, J = null, F = null, ha=!1, aa, ia, qa, t = {}, b = fa(!0, {}, f.opts, b);
        Ga(b);
        var A = t.field && n.$(t.field), Y = A && A.form, G = n.$(t.cont), ka = t.list && G ? n.$(t.list, G): G, S = [], Q, N, ea, va, ja=!0, ga = null;
        if (!A ||!G ||!ka)
            throw "";
        t.debug && n.objEach(y, function(a) {
            c(a, Ca)
        });
        fa(I, {
            field: A,
            form: Y,
            cont: G,
            list: ka,
            show: xa,
            focus: pa,
            moveFocus: wa,
            select: ua,
            open: ya,
            close: X,
            isClosed: V,
            enable: Da,
            disable: Ea,
            isDisabled: function() {
                return ja
            },
            destroy: function() {
                l(y.destroy);
                Ea();
                t.cch && la.flush();
                e();
                l(y.destroyEnd);
                delete f.instances[I.guid]
            },
            on: c,
            off: e,
            getState: i,
            opts: Ga,
            flushCache: function() {
                la.flush()
            }
        });
        I.guid = Z++;
        f.instances[I.guid] = I;
        t.enabled && Da()
    }
    var d = a.document, k = Object.prototype, e = k.hasOwnProperty, c = k.toString, i = Array.prototype.slice, j = Array.prototype.forEach, r = Array.prototype.indexOf, w = Array.prototype.map, q = String.prototype.trim, v = Function.prototype.bind,
    p = encodeURIComponent, s = a.setTimeout, x = a.setInterval, g = a.clearTimeout, K = a.clearInterval, k = Date.now || function() {
        return + new Date
    }, B = /\s+/, E = /\{(\w*)\}/mg, l = /([-\\()^$.?+*])/g, u = "SG" + k(), Z = 1, C = function() {
        return !1
    }, D = function() {}, W;
    a.getComputedStyle ? W = function(b, c) {
        var g, e = a.getComputedStyle(b, null);
        e && (g = e.getPropertyValue(c) || e[c]);
        return g
    } : d.documentElement.currentStyle && (W = function(a, b) {
        var c = a.currentStyle && a.currentStyle[b], g = a.style;
        null == c && (g && g[b]) && (c = g[b]);
        return "" === c ? "auto" : c
    });
    var n = {
        hasOwn: function(a,
        b) {
            return e.call(a, b)
        },
        isArr: Array.isArray || function(a) {
            return "[object Array]" === c.call(a)
        },
        isFn: function(a) {
            return "[object Function]" === c.call(a)
        },
        isObj: function(a) {
            if (!a || ("object" !== typeof a || n.isArr(a) || n.isFn(a) || a.nodeType || a.setTimeout) || a.constructor&&!n.hasOwn(a, "constructor")&&!n.hasOwn(a.constructor.prototype, "isPrototypeOf"))
                return !1;
            for (var c in a);
            return c === b || n.hasOwn(a, c)
        },
        isWin: function(a) {
            return !!a && "object" === typeof a && "setInterval"in a
        },
        bind: function(a, b) {
            if (v)
                return v.apply(a,
                i.call(arguments, 1));
            var c = i.call(arguments, 2), g = function() {
                return a.apply(this instanceof D ? this : b || {}, c.concat(i.call(arguments)))
            };
            D.prototype = a.prototype;
            g.prototype = new D;
            return g
        },
        ext: function() {
            var a, c, g, e, l, d = arguments, i = d[0] || {}, f = 1, u = d.length, j=!1;
            "boolean" === typeof i && (j = i, i = d[1] || {}, f = 2);
            for ("object" !== typeof i&&!n.isFn(i) && (i = {}); f < u; f++)
                if (null != (a = d[f]))
                    for (c in a)
                        g = i[c], e = a[c], i !== e && (j && e && (n.isObj(e) || (l = n.isArr(e))) ? (l ? (l=!1, g = g && n.isArr(g) ? g : []) : g = g && n.isObj(g) ? g : {}, i[c] = fa(j,
                        g, e)) : e !== b && (i[c] = e));
            return i
        },
        copy: function(a) {
            return n.ext(!0, n.isArr(a) ? [] : {}, a)
        },
        trim: function(a) {
            a = String(a);
            return q ? q.call(a) : a.replace(/^\s+|\s+$/g, "")
        },
        arrEach: function(a, b, c) {
            if (j)
                j.call(a, b, c);
            else 
                for (var g = 0, e = a.length; g < e; g++)
                    b.call(c, a[g], g)
        },
        objEach: function(a, b, c) {
            for (var g in a)
                n.hasOwn(a, g) && b.call(c, a[g], g)
        },
        each: function(a, b, c) {
            n["number" === typeof a.length ? "arrEach": "objEach"](a, b, c)
        },
        map: function(a, b, c) {
            if (w)
                return w.call(a, b, c);
            for (var g = 0, e = a.length, l = Array(e); g < e; g++)
                l[g] =
                b.call(c, a[g], g);
            return l
        },
        indexOf: function(a, b, c) {
            if (r)
                return r.call(a, b, c);
            var c = c || 0, g =- 1, e = 0, l = a.length;
            if (c < l)
                for (0 > c && (e = Math.max(l - Math.abs(c), 0)); e < l; e++)
                    if (a[e] === b) {
                        g = e;
                        break
                    }
            return g
        },
        format: function(a, b) {
            return a && b ? String(a).replace(E, function(a, c) {
                return n.hasOwn(b, c) ? b[c] : ""
            }) : ""
        },
        objFormat: function(a, b) {
            var c = {};
            n.objEach(a, function(a, g) {
                c[g] = "string" === typeof a ? n.format(a, b) : n.isObj(a) ? n.objFormat(a, b) : a
            });
            return c
        },
        walker: function(a, b, c) {
            for (var g = 2 < arguments.length, e = b.split("."),
            l = 0, d = e.length, i = a, f, u = null, j, w; l < d; l++)
                if (f = e[l], u = (i = u || i)[f], w = l == d - 1, null == u ||!w&&!n.isObj(u))
                    if (g)
                        w || (u = i[f] = {});
                    else {
                        u = null;
                        break
                    }
            l == d && g ? (j = i[f], i[f] = c) : c = u;
            return g ? j : c
        },
        resc: function(a) {
            return String(a).replace(l, "\\$1")
        },
        from: function(a) {
            return null == a ? [] : n.isArr(a) ? a : [a]
        },
        mkarr: function(a) {
            var c, g;
            if (a && (g = a.length) !== b)
                try {
                    c = i.call(a)
            } catch (e) {
                for (c = Array(g); g--;)
                    c[g] = a[g]
            }
            return c || []
        },
        prm: function(a) {
            var b, c, g = [];
            for (b in a)
                n.hasOwn(a, b) && g.push(n.map([b, n.isFn(c = a[b]) ? c(): c], p).join("="));
            return g.join("&")
        },
        aprm: function(a, b) {
            if (!b)
                return a;
            var a = String(a), c = [a], g = a.charAt(a.length - 1);
            c.push(!~a.indexOf("?") ? "?" : "?" !== g && "&" !== g ? "&" : "");
            c.push("object" === typeof b ? n.prm(b) : b);
            return c.join("")
        },
        url: function(a) {
            var b = [];
            a.scheme && b.push(a.scheme, ":");
            a.authority && b.push("//", a.authority);
            a.path && b.push("/", !a.path.indexOf("/") ? a.path.slice(1) : a.path);
            a.query && b.push(n.aprm("?", a.query));
            a.fragment && (a = a.fragment, b.push("#", "object" === typeof a ? n.prm(a) : a));
            return b.join("")
        },
        css: function(a,
        b, c) {
            if (!a)
                return null;
            var g = null;
            2 < arguments.length && b ? a.style[b] = c : b && (g = W(a, b));
            return g
        },
        addCls: function(a, b) {
            for (var c = n.trim(b).split(B), g = c.length, e = a.className; g--;)
                b = c[g], n.hasCls(a, b) || (e += " " + b);
            a.className = n.trim(e)
        },
        rmCls: function(a, b) {
            if (1 < arguments.length) {
                for (var c = n.trim(b).split(B), g = c.length, e = a.className; g--;)
                    b = c[g], e = e.replace(RegExp("(?:^|\\s+)" + n.resc(b) + "(?:\\s+|$)", "g"), " ");
                a.className = n.trim(e)
            } else 
                a.className = ""
        },
        hasCls: function(a, b) {
            b = n.trim(b);
            return !(!a ||!a.nodeType ||
            !b||!~(" " + a.className + " ").indexOf(" " + b + " "))
        },
        cres: function(a) {
            var a = n.trim(a), b = d.createDocumentFragment().appendChild(d.createElement("div"));
            b.innerHTML = a;
            return n.mkarr(b.childNodes)
        },
        cre: function(a) {
            a = n.trim(a);
            return "<" === a.charAt(0) && ">" === a.charAt(a.length - 1) ? n.cres(a)[0] : d.createElement(a)
        },
        rme: function(a) {
            a && a.parentNode && (f.Event.rm(a), a.parentNode.removeChild(a))
        },
        empty: function(a) {
            for (var b; a && (b = a.firstChild);)
                n.rme(b)
        },
        attr: function(a, b, c) {
            2 < arguments.length && a.setAttribute(b, c);
            return a.getAttribute(b)
        },
        hasFocus: function(a) {
            return !!a && a.ownerDocument.activeElement === a
        },
        contains: d.documentElement.contains ? function(a, b) {
            return a !== b && (a.contains ? a.contains(b) : !0)
        }
        : d.documentElement.compareDocumentPosition ? function(a, b) {
            return !!(a.compareDocumentPosition(b) & 16)
        }
        : C,
        parseXML: function(c) {
            if ("string" !== typeof c ||!c)
                return null;
            var g, e;
            try {
                a.DOMParser ? (e = new DOMParser, g = e.parseFromString(c, "text/xml")) : (g = new ActiveXObject("Microsoft.XMLDOM"), g.async = "false", g.loadXML(c))
            } catch (l) {
                g =
                b
            }
            if (!g ||!g.documentElement || g.getElementsByTagName("parsererror").length)
                throw "Invalid XML: " + c;
            return g
        }
    }, ba = /^[\],:{}\s]*$/, M = /\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, T = /"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, sa = /(?:^|:|,)(?:\s*\[)+/g;
    n.parseJSON = function(b) {
        if ("string" !== typeof b ||!b)
            return null;
        b = n.trim(b);
        if (a.JSON && a.JSON.parse)
            return a.JSON.parse(b);
        if (ba.test(b.replace(M, "@").replace(T, "]").replace(sa, "")))
            return (new Function("return " + b))();
        throw "Invalid JSON: " +
        b;
    };
    var ta = n, da = function(a, b) {
        var c, g = "string" === typeof a, e=!1, l;
        g ? (a = n.trim(a), e = (!a.indexOf("#") ||!a.indexOf(".")) && 0 > a.indexOf(ma) + a.indexOf(ca)) : a && (l=!!a.nodeType);
        if (e || l) {
            if (l && a[u] ? c = U[a[u]] : l || (c = U[a]), !c && (e = n.$(a)))
                c = U[e[u] || (e[u] = Z++)] = da(e.innerHTML || ""), g && (U[a] = c)
        } else 
            c = a, c = U[c] || (U[c] = Function("_o", "_e", "var _=[],print=function(){_.push.apply(p,arguments)};with(_o||{}){_.push('" + c.replace(Ca, " ").split(ma).join("\t").replace(Ia, "$1\r").replace(Ja, function(a, b) {
                b = n.trim(b);
                return !b ?
                "" : ["');", "try{_.push(" + b + ")}catch(e){_e.push(e)}", "_.push('"].join("")
            }).split("\t").join("');").split(ca).join("_.push('").split("\r").join("\\'") + "')}return _.join('')"));
        c || (c = b ? function() {
            return ""
        } : "");
        return b ? c(b, na) : c
    }, ma = "<%", ca = "%>", Ca = /[\r\t\n]/g, Ia = RegExp("((^|" + n.resc(ca) + ")[^\\t]*)'", "g"), Ja = RegExp("\\t=(.*?)" + n.resc(ca), "g"), na = [], U = {};
    da.errs = na;
    da.cch = U;
    ta.tmpl = da;
    var fa = n.ext, O = n.isFn, y = f.evt = {};
    n.arrEach("open openEnd close closeEnd enable enableEnd disable disableEnd destroy destroyEnd blockRequest successRequest completeRequest stopRequest errorRequest startRequest sendRequest passFilter failFilter rejectData acceptData render renderEnd focus focusEnd select selectEnd valueChange flushCache flushCacheEnd setCache setCacheEnd getCache getCacheEnd".split(" "),
    function(a) {
        this[a] = a
    }, y);
    f.instances = {};
    f.opts = {
        enabled: !0,
        cont: "#sg, .sg",
        delay: 250,
        valMin: 1,
        valMax: 255,
        valFilter: /(?:\S)/,
        url: "http://suggests.go.mail.ru/sg_u?q={query}",
        ajax: {
            dataType: "jsonp",
            jsonp: "callback",
            scriptCharset: "utf-8",
            timeout: 5E3,
            stackSize: 2
        },
        dataFilter: function(a) {
            return a
        },
        dataGet: function(a) {
            return a && a.items ? a.items : []
        },
        cch: !0,
        cchLimit: 128,
        max: 10,
        min: 0,
        autoSubmit: !0,
        hover: "sg__item_hover",
        item: '<div class="sg__item"><%= itemData.textMarked %></div>',
        result: function(a) {
            return a.text
        },
        select: function() {},
        keynavDelay: 150,
        preview: !0,
        mousePreview: !1,
        debug: !1
    };
    f.setup = function(a) {
        fa(!0, f.opts, a)
    };
    f.expando = u;
    f.guid = function() {
        return Z++
    };
    f.now = k;
    f.utils = n;
    f.tmpl = n.tmpl;
    a.SG = a.SG || f
})(window);
(function(a, b) {
    var f = a.document, d = b.utils, k = d.mkarr, e=!!f.getElementsByClassName, c=!!f.querySelectorAll, i = function(a, b) {
        var d = [], b = b ? i(b)[0]: f;
        if (!a)
            return d;
        if (a.nodeType || "object" === typeof a && "setTimeout"in a)
            return [a];
        if ("object" === typeof a && ("[object Array]" === Object.prototype.toString.call(a) || void 0 !== a.length))
            return k(a);
        if (!b)
            return d;
        if (/^[\w#.][\w\]*^|=!]*$/.test(a)) {
            var q = 0;
            switch (a.charAt(0)) {
            case "#":
                q = a.slice(1);
                (d = f.getElementById(q)) && d.id !== q && (d = f.all[q]);
                d = d ? [d] : [];
                break;
            case ".":
                var v =
                a.slice(1);
                if (e)
                    d = (d = b.getElementsByClassName(v)).length ? d : [];
                else {
                    for (var v = " " + v + " ", p = b.getElementsByTagName("*"), s = 0, x; x = p[s++];) 
                        - 1 != (" " + x.className + " ").indexOf(v) && (d[q++] = x);
                    d = q ? d : []
                }
                break;
            default:
                d = (d = b.getElementsByTagName(a)).length ? d : []
            }
        } else {
            p=!1;
            s=~a.indexOf("!=");
            if (c&&!s)
                try {
                    d = b.querySelectorAll(a.replace(/=([^\]]+)/, '="$1"'))
                } catch (g) {
                p=!0
            }
            if (p ||!c || s) {
                x = a.split(/ *, */);
                for (var K = x.length - 1, B=!!K, E, l, u, Z, C, D, W, n, ba, M, T; p = x[K--];) {
                    l = (E = p.replace(/(\([^)]*)\+/, "$1%").replace(/(\[[^\]]+)~/,
                    "$1&").replace(/(~|>|\+)/, " $1 ").split(/ +/)).length;
                    s = 0;
                    u = " ";
                    for (p = [b]; q = E[s++];)
                        if (" " !== q && ">" !== q && "~" !== q && "+" !== q && p) {
                            q = q.match(/([^[:.#]+)?(?:#([^[:.#]+))?(?:\.([^[:.]+))?(?:\[([^!&^*|$[:=]+)([!$^*|&]?=)?([^:\]]+)?\])?(?:\:([^(]+)(?:\(([^)]+)\))?)?/);
                            Z = q[1] || "*";
                            C = q[2];
                            v = q[3] ? " " + q[3] + " " : "";
                            D = [];
                            q = W = 0;
                            for (ba = s == l; n = p[W++];)
                                switch (u) {
                                case " ":
                                    n = n.getElementsByTagName(Z);
                                    for (T = 0; M = n[T++];)
                                        if ((!C || M.id === C) && (!v||-1 != (" " + M.className + " ").indexOf(v))&&!M.yeasss)
                                            ba && (M.yeasss = 1), D[q++] = M
                                }
                                p =
                                D
                        } else 
                            u = q;
                    if (B) {
                        if (!p.concat) {
                            D = [];
                            for (T = 0; M = p[T];)
                                D[T++] = M;
                            p = D
                        }
                        d = p.concat(1 == d.length ? d[0] : d)
                    } else 
                        d = p
                }
                for (q = d.length; q--;)
                    d[q].yeasss = d[q].nodeIndex = d[q].nodeIndexLast = null
            }
        }
        return k(d)
    };
    b.$ = d.$ = function(a, b) {
        return i(a, b)[0] || null
    };
    b.$$ = d.$$ = i
})(window, SG);
(function(a, b) {
    function f(a) {
        var b = a.relatedTarget, c=!1, d = a.type;
        a.type = x[d];
        b !== this && (b && (c = e.contains(this, b)), c || (b = e.mkarr(arguments), p.fire(this, b.shift(), b), a.type = d))
    }
    var d = a.document, k = b.expando, e = b.utils, c = /\s+/, i = "altKey attrChange attrName bubbles button cancelable charCode clientX clientY ctrlKey currentTarget data detail eventPhase fromElement handler keyCode layerX layerY metaKey newValue offsetX offsetY pageX pageY prevValue relatedNode relatedTarget screenX screenY shiftKey srcElement target toElement view wheelDelta which".split(" "), j =
    d.addEventListener ? function(a, b, c) {
        a.addEventListener(b, c, !1)
    }
    : d.attachEvent ? function(a, b, c) {
        a.attachEvent("on" + b, c)
    }
    : emptyFn, r = d.addEventListener ? function(a, b, c) {
        a.removeEventListener(b, c, !1)
    }
    : d.attachEvent ? function(a, b, c) {
        a.detachEvent("on" + b, c)
    }
    : emptyFn, w = function(a) {
        return "defaultPrevented"in a ? a.defaultPrevented : "returnValue"in a?!a.returnValue : "getPreventDefault"in a ? a.getPreventDefault() : !1
    }, q = function() {
        return !0
    }, v = function() {
        return !1
    }, p = {
        fired: null,
        c: {},
        custom: {},
        hlink: function(a, c) {
            c[k] =
            a[k] = a[k] || b.guid();
            return c
        },
        fix: function(a) {
            if (a[k])
                return a;
            for (var b = a, a = new p.Event(a), c = 0, e; e = i[c++];)
                a[e] = b[e];
            a.target || (a.target = a.srcElement || d);
            3 === a.target.nodeType && (a.target = a.target.parentNode);
            !a.relatedTarget && a.fromElement && (a.relatedTarget = a.fromElement === a.target ? a.toElement : a.fromElement);
            null == a.pageX && null != a.clientX && (c = a.target.ownerDocument || d, b = c.documentElement, c = c.body, a.pageX = a.clientX + (b && b.scrollLeft || c && c.scrollLeft || 0) - (b && b.clientLeft || c && c.clientLeft || 0), a.pageY =
            a.clientY + (b && b.scrollTop || c && c.scrollTop || 0) - (b && b.clientTop || c && c.clientTop || 0));
            if (null == a.which && (null != a.charCode || null != a.keyCode))
                a.which = null != a.charCode ? a.charCode : a.keyCode;
            !a.metaKey && a.ctrlKey && (a.metaKey = a.ctrlKey);
            !a.which && void 0 !== a.button && (a.which = a.button & 1 ? 1 : a.button & 2 ? 3 : a.button & 4 ? 2 : 0);
            return a
        },
        handler: function(b, c) {
            b = p.fix(b || a.event);
            b.currentTarget = this;
            var d = p.c[this[k]][b.type], i;
            (c = c && e.isArr(c) ? e.mkarr(c) : null) && c.unshift(b);
            for (var l = 0, f = d.length; l < f&&!(i = d[l], i = c ? i.apply(this,
            c) : i.call(this, b), !1 === i && (b.preventDefault(), b.stopPropagation()), b.isImmediatePropagationStopped()); l++);
        },
        add: function(g, d, i, f) {
            if (g && d) {
                g.setInterval && (g != a&&!g.frameElement) && (g = a);
                var l = e.from(i), u = f ? "unshift": "push";
                e.arrEach(d.split(c), function(a) {
                    e.arrEach(l, function(c) {
                        if (c && e.isFn(c)) {
                            c[k] || (c[k] = b.guid());
                            var d = g[k] = g[k] || b.guid(), d = p.c[d] = p.c[d] || {}, d = d[a] = d[a] || [], l = d.handle = function(a, b) {
                                if (!a || a.type !== p.fired)
                                    return p.handler.call(l.elem, a, b)
                            };
                            l.elem = g;
                            if (1 == d[u](c) && (g.nodeType ||
                            g.setInterval))
                                c = p.custom && p.custom[a] && p.custom[a].setup, (!c ||!1 === c.call(g, l)) && j(g, a, d.handle)
                        }
                    })
                })
            }
        },
        one: function(a, b, c) {
            p.add(a, b, p.hlink(c, function() {
                p.rm(this, b, arguments.callee);
                return c.apply(this, arguments)
            }))
        },
        first: function(a, b, c) {
            p.add(a, b, c, !0)
        },
        rm: function(b, d, i) {
            if (b)
                if (d)
                    b.setInterval && (b != a&&!b.frameElement) && (b = a), e.arrEach(d.split(c), function(a) {
                        var c = b[k], d = c && p.c[c], l = d && d[a];
                        if (c && d && l) {
                            if (e.isFn(i) && i[k])
                                for (var f = l.length; f--;) {
                                    if (l[f][k] == i[k]) {
                                        l.splice(f, 1);
                                        break
                                    }
                                } else 
                                    l.length =
                                    0;
                                    if (!l.length) {
                                        if (b.nodeType || b.setInterval)
                                            f = p.custom && p.custom[a] && p.custom[a].teardown, (!f ||!1 === f.call(b, l.handle)) && r(b, a, l.handle);
                                            delete d[a];
                                            for (a in d)
                                                return;
                                                delete p.c[c]
                                    }
                        }
                    });
                    else {
                        var d = (d = b[k]) && p.c[d], f = [], l;
                        if (d) {
                            for (l in d)
                                e.hasOwn(d, l) && f.push(l);
                                p.rm(b, f.join(" "))
                            }
                    }
                },
fire: function(a, b, c) {
    if (a && b) {
        var c = e.from(c), d = b.type || b, l, i = "on" + d, f = a[k];
        if (f = (f = f && p.c[f]) && f[d])
            l = b[k] ? b : "object" === typeof b ? new p.Event(d, b) : new p.Event(b), l.type = d, l.target = l.currentTarget = a, f.handle.call(a, l,
            c);
        if (!l ||!l.isDefaultPrevented()) {
            var j;
            try {
                if ((a.nodeType || a.setInterval) && i && a[d])(j = a[i]) && (a[i] = null), p.fired = d, a[d]()
                } catch (q) {}
            j && (a[i] = j)
        }
        p.fired = null
    }
}, copy: function(a, b, c) {
    var a = (a = a[k]) && p.c[a], d = null;
    null != c && a && a[c] ? (d = {}, d[c] = a[c]) : a && (d = a);
    e.objEach(d, function(a, c) {
        p.add(b, c, a)
    })
}, Event: function(a, c) {
    a && a.type ? (this.origEvent = a, this.type = a.type, this.isDefaultPrevented = w(a) ? q : v) : this.type = a;
    c && extend(this, c);
    this.timeStamp = b.now();
    this[k]=!0
}
},
s = p.Event.prototype;
s.preventDefault = function() {
    this.isDefaultPrevented =
    q;
    var a = this.origEvent;
    a && (a.preventDefault ? a.preventDefault(): a.returnValue=!1)
};
s.stopPropagation = function() {
    this.isPropagationStopped = q;
    var a = this.origEvent;
    a && (a.stopPropagation && a.stopPropagation(), a.cancelBubble=!0)
};
s.stopImmediatePropagation = function() {
    this.isImmediatePropagationStopped = q;
    this.stopPropagation()
};
s.isDefaultPrevented = v;
s.isPropagationStopped = v;
s.isImmediatePropagationStopped = v;
var x = {
    mouseover: "mouseenter",
    mouseout: "mouseleave"
};
e.objEach(x, function(a, b) {
    p.custom[a] = {
        setup: function() {
            p.add(this,
            b, f)
        },
        teardown: function() {
            p.rm(this, b, f)
        }
    }
});
p.protoMixin = {
    on: function(a, b) {
        p.add(this, a, b)
    },
    off: function(a, b) {
        p.rm(this, a, b)
    },
    fire: function(a, b) {
        p.fire(this, a, b)
    }
};
p.natAdd = j;
p.natRm = r;
p.natOne = function(a, b, c) {
    j(a, b, function() {
        r(a, b, arguments.callee);
        return c.apply(a, arguments)
    })
};
p.natIsDefaultPrevented = w;
b.Event = p;
e.Event = p
})(window, SG);
(function(a, b) {
    function f() {
        try {
            return new XMLHttpRequest
        } catch (a) {}
        try {
            return new ActiveXObject("Microsoft.XMLHTTP")
        } catch (b) {}
        return null
    }
    function d(a, b) {
        return new e(a, b)
    }
    function k(a, b) {
        this._url = a.replace(/#.*$/, "");
        this._options = b
    }
    function e(a, b) {
        var e, b = q.ext(!0, {}, d.defaults, b);
        b.method = b.method.toUpperCase();
        null == b.crossDomain && (e = p.exec(a.toLowerCase()), b.crossDomain=!(!e ||!(e[1] != x[1] || e[2] != x[2] || (e[3] || ("http:" === e[1] ? 80 : 443)) != (x[3] || ("http:" === x[1] ? 80 : 443)))));
        if (!b.dataType)
            throw Error("dataType must be specified");
        b.dataType = b.dataType.toLowerCase();
        this._url = a;
        this._urlParts = e;
        this._options = b;
        this._degradeEnabled = this._degraded=!1;
        b.CORSDegrade && b.crossDomain && (this._degradeEnabled=!0);
        this._CORSFailed=!1;
        if ("jsonp" === b.dataType || this._degradeEnabled && (!K || e[1] + "//" + e[2]in B))
            b.dataType = "jsonp", this._degraded=!0, this._degradeEnabled=!1;
        var g = this._basecls = new ("jsonp" === b.dataType ? i : c)(a, b);
        q.arrEach(["success", "complete", "error"], function(a) {
            var c = b[a];
            if (c && q.isFn(c))
                g.on(a, c)
        })
    }
    function c(a, b) {
        k.call(this,
        a, b);
        this._xhr = null;
        this._aborted = this._completed = this._processing=!1;
        this._tmid = null;
        this._setReqHeaders = {}
    }
    function i(a, b) {
        k.call(this, a, b);
        this._jsonpCallback = this._script = null;
        this._aborted = this._completed = this._processing=!1;
        this._tmid = null
    }
    function j(a, b, c) {
        a = a.toUpperCase();
        d[a] = b;
        E[b] = c
    }
    var r = a.document, w = a.location, q = b.utils, v = q.Event, p = /^([\w\+\.\-]+:)(?:\/\/([^\/?#:]*)(?::(\d+))?)?/, s, x;
    try {
        s = w.href
    } catch (g) {
        s = r.createElement("a"), s.href = "", s = s.href
    }
    x = p.exec(s.toLowerCase()) || [];
    var w =
    f(), K = "withCredentials"in w, w = null, B = {};
    k.prototype = {
        readyState: 0,
        status: - 1,
        statusText: "",
        responseText: null,
        responseXML: null,
        send: function() {},
        abort: function() {},
        destroy: function() {},
        setRequestHeader: function() {},
        getAllResponseHeaders: function() {},
        getResponseHeader: function() {}
    };
    q.ext(k.prototype, v.protoMixin);
    e.prototype = q.ext({}, {
        send: function() {
            var a = this, b = a._options, c = a._basecls;
            a._degradeEnabled && v.first(c, "success complete error", function(d) {
                if ("error" === d.type && (4 == this.readyState && 0 == this.status) &&
                (a._CORSFailed=!0, a._degradeEnabled=!1, b.CORSUrlDegrade)) {
                    var e = a._urlParts;
                    B[e[1] + "//" + e[2]]=!0
                }
                a._CORSFailed && (d.stopImmediatePropagation(), "complete" === d.type && (c.off("success complete error", arguments.callee), b.dataType = "jsonp", new_basecls = new i(a._url, b), v.copy(c, new_basecls), c.destroy(), c = null, a._basecls = new_basecls, a.send()))
            });
            return c.send.apply(c, arguments)
        }
    });
    q.objEach(k.prototype, function(a, b) {
        b in this || q.isFn(a) && (this[b] = function() {
            var a = this._basecls;
            return a[b].apply(a, arguments)
        })
    },
    e.prototype);
    c.prototype = q.ext({}, k.prototype, {
        _dataTypes: {
            json: q.parseJSON,
            xml: q.parseXML
        },
        _getXHR: function() {
            return f()
        },
        _handleResponse: function() {
            var a = this.responseText, b;
            return (b = this._dataTypes[this._options.dataType]) ? b(a) : String(a)
        },
        _onreadystatechange: function() {
            var a = this._xhr, b = this._aborted, c = null, e;
            if (!b) {
                this.readyState = a.readyState;
                this.status = a.status;
                try {
                    this.statusText = a.statusText
                } catch (g) {
                    this.statusText = ""
                }
                if (4 == this.readyState) {
                    if ((e = a.responseXML) && e.documentElement)
                        this.responseXML =
                        e;
                    try {
                        this.responseText = a.responseText
                    } catch (i) {}
                }
            }
            if (b || 4 == this.readyState)
                a.onreadystatechange = null, clearTimeout(this._tmid), b || (a = this.status, this._processing=!1, this._completed=!0, 200 <= a && 300 > a || 304 === a ? (c = this._handleResponse(), a = d.SUCCESS, b = E[a], this.fire("success", [c, b])) : (a = d.BAD_STATUS, b = E[a], this.fire("error", [a, b])), this.fire("complete", [c, a, b]), this._cleanup())
        },
        _abort: function(a) {
            var b = E[a];
            this._xhr.abort();
            this._aborted=!0;
            this._processing=!1;
            this._completed=!0;
            this._onreadystatechange();
            this.readyState = 4;
            a != d.SYSTEM_ABORT && (this.fire("error", [a, b]), this.fire("complete", [null, a, b]));
            this._cleanup()
        },
        _cleanup: function() {
            this._tmid = this._xhr = null;
            this._setReqHeaders = {};
            this.off()
        },
        send: function() {
            var a = this;
            if (a._processing)
                throw Error("already processing");
            var b = a._options, c, e, g, i, f = "";
            c = a._getXHR();
            if (!c)
                throw Error("XHR not supported in your browser");
            a._xhr = c;
            e = b.method;
            g = a._url;
            "POST" === e && (i = g.indexOf("?"), ~i && (f = g.slice(i + 1), g = g.substring(0, i)));
            c.open(e, g, !!b.async);
            c.onreadystatechange =
            q.bind(a._onreadystatechange, a);
            "POST" === e && a.setRequestHeader("Content-Type", b.contentType);
            b.cache || a.setRequestHeader("If-Modified-Since", "Sat, 1 Jan 1970 00:00:00 GMT");
            !b.crossDomain&&!a._setReqHeaders["X-Requested-With"] && (a._setReqHeaders["X-Requested-With"] = "XMLHttpRequest");
            q.objEach(a._setReqHeaders, function(a, b) {
                c.setRequestHeader(b, a)
            });
            b.xhrFields && q.objEach(b.xhrFields, function(a, b) {
                c[b] = a
            });
            a._processing=!0;
            b.timeout && (a._tmid = setTimeout(function() {
                a._abort(d.ABORT_TIMEOUT)
            }, b.timeout));
            return c.send("POST" === e ? f : null)
        },
        abort: function() {
            this._completed || this._abort(d.ABORT_USER)
        },
        setRequestHeader: function(a, b) {
            this._setReqHeaders[a] = b
        },
        destroy: function() {
            this._completed || this._abort(d.SYSTEM_ABORT)
        }
    });
    i.prototype = q.ext({}, k.prototype, {
        _onload: function() {
            var a = this._script, b = this._aborted;
            if (b ||!a.readyState || "loaded" === a.readyState)
                a.onload = a.onreadystatechange = null, q.rme(a), this._script = null, this._complete(), b && this._replGlCb2Rm()
        },
        _glCb: function(a) {
            var b = d.SUCCESS, c = E[b];
            this._complete();
            this._rmGlCb();
            this.fire("success", [a, c]);
            this.fire("complete", [a, b, c]);
            this._cleanup()
        },
        _complete: function() {
            clearTimeout(this._tmid);
            this._completed=!0;
            this._processing=!1
        },
        _addGlCb: function() {
            a[this._jsonpCallback] = q.bind(this._glCb, this)
        },
        _rmGlCb: function() {
            try {
                delete a[this._jsonpCallback]
            } catch (b) {}
        },
        _replGlCb2Rm: function() {
            a[this._jsonpCallback] = q.bind(this._rmGlCb, this)
        },
        _abort: function(a) {
            var b = E[a];
            this._aborted=!0;
            this._onload();
            a != d.SYSTEM_ABORT && (this.fire("error", [a, b]), this.fire("complete",
            [null, a, b]));
            this._cleanup()
        },
        _cleanup: function() {
            this._tmid = null;
            this.off()
        },
        send: function() {
            var a = this;
            if (a._processing)
                throw Error("already processing");
            var c = a._options, e = r.head || q.$("head,body") || r.documentElement, g = c.jsonp, i = c.jsonpCallback, f = c.scriptCharset, j;
            j = {};
            q.isFn(i) && (i = i());
            c.data && (j = q.ext(j, c.data));
            j[g] = i;
            c.cache || (j._ = b.now());
            g = a._url;
            g = q.aprm(g, j);
            j = q.cre("script");
            j.async=!0;
            f && (j.charset = f);
            j.onload = j.onreadystatechange = q.bind(a._onload, a);
            j.src = g;
            a._script = j;
            a._jsonpCallback =
            i;
            a._processing=!0;
            a._addGlCb();
            c.timeout && (a._tmid = setTimeout(function() {
                a._abort(d.ABORT_TIMEOUT)
            }, c.timeout));
            e.insertBefore(j, e.firstChild);
            return !0
        },
        abort: function() {
            this._completed || this._abort(d.ABORT_USER)
        },
        destroy: function() {
            this._completed || this._abort(d.SYSTEM_ABORT)
        }
    });
    d.XHR = c;
    d.JSONP = i;
    d.Wrapper = e;
    d.CORSResticted = B;
    d.defaults = {
        dataType: "text",
        method: "GET",
        url: s,
        async: !0,
        contentType: "application/x-www-form-urlencoded; charset=utf-8",
        xhrFields: {
            withCredentials: !1
        },
        jsonp: "callback",
        jsonpCallback: function() {
            return b.expando +
            "_" + b.guid()
        },
        scriptCharset: "utf-8",
        CORSDegrade: !1,
        CORSUrlDegrade: "domain"
    };
    s = 1;
    var E = {};
    j("ABORT_USER", s++, "aborted");
    j("ABORT_TIMEOUT", s++, "timeout");
    j("BAD_STATUS", s++, "bad status");
    j("SUCCESS", s++, "OK");
    j("SYSTEM_ABORT", s++, "system aborted");
    d.STATUSES = E;
    d.base = k;
    d.supportCORS = K;
    b.Ajax = d
})(window, SG);
(function(a, b) {
    function f(a, f, e, c, i, j) {
        this.content = a;
        this.tipSide = e || "bottom";
        this.arrowShift = j;
        this.listenTo = f;
        this.mod = c;
        this.shift = i;
        this.node = b.createElement("div");
        this.node.className = "hint" + (this.mod ? " hint_" + this.mod : "");
        - 1 !== navigator.userAgent.indexOf("MSIE 6") && (this.nodeIE = b.createElement("iframe"), this.nodeIE.className = "hint hint_iframe");
        this._prepareHTML();
        this.arrow = this.node.getElementsByTagName("i")[0];
        this._prepareEvents();
        this.enable()
    }
    f.prototype = new Splash.EventEmitter;
    mr.extend(f.prototype,
    {
        isVisible: function() {
            return this._visible
        },
        show: function(a) {
            if (!(a&&!this._visible ||!a && this._visible)) {
                this._inBody || (this.nodeIE && b.body.appendChild(this.nodeIE), b.body.appendChild(this.node), this._inBody=!0);
                a || this.emit("show");
                this.node.style.visibility = "hidden";
                this.node.style.display = "block";
                var a = mr.position(this.listenTo), f = this.node.offsetHeight, e = this.node.offsetWidth, c = this.listenTo.offsetWidth, i = this.listenTo.offsetHeight, j = 22, r = 6;
                this.shift && (j += this.shift.x, r += this.shift.y);
                var w =
                0, q = 0;
                this.arrowShift && (w += this.arrowShift.x, q += this.arrowShift.y);
                this.arrow.style.marginLeft = w + "px";
                this.arrow.style.marginTop = q + "px";
                this.node.style.left = a.left + c / 2 - j + "px";
                "bottom" == this.tipSide ? this.node.style.top = a.top - f - r + "px" : "top" == this.tipSide && (this.node.style.top = a.top + i + r + "px");
                this.nodeIE && (this.nodeIE.style.width = e, this.nodeIE.style.height = f, this.nodeIE.style.left = this.node.style.left, this.nodeIE.style.top = this.node.style.top, this.nodeIE.style.display = this.node.style.display, this.nodeIE.style.visibility =
                "visible");
                this.node.style.visibility = "visible";
                this._visible=!0
            }
        },
        hide: function() {
            this.node.style.display = "none";
            this.nodeIE && (this.nodeIE.style.display = this.node.style.display);
            this._visible=!1
        },
        _prepareHTML: function() {
            var a = '<i class="hint__tip hint__tip_' + this.tipSide + (this.mod ? " hint__tip_" + this.mod + " hint__tip_" + this.mod + "_" + this.tipSide : "") + '">' + ( - 1 !== navigator.userAgent.indexOf("MSIE 6") ? '<i class="hint__tip_ie"></i>' : "") + '</i><div class="hint__content' + (this.mod ? " hint__conten_" + this.mod :
            "") + '">' + this.content;
            this.node.innerHTML = a;
            this.show(!0)
        },
        setContent: function(a) {
            this.content = a;
            this._prepareHTML()
        },
        _prepareEvents: function() {
            var a;
            this._showEvent = new Splash.DOMEvent(this.listenTo, "mouseover", mr.proxy(function() {
                clearTimeout(a);
                a = setTimeout(mr.proxy(function() {
                    this.show()
                }, this), 100)
            }, this));
            var b = mr.proxy(function() {
                clearTimeout(a);
                a = setTimeout(mr.proxy(function() {
                    this.hide()
                }, this), 400)
            }, this);
            this._hideEvent = new Splash.DOMEvent(this.listenTo, "mouseout", b);
            mr.bind(this.node,
            "mouseover", function() {
                clearTimeout(a)
            });
            mr.bind(this.node, "mouseout", b)
        },
        enable: function() {
            this._showEvent.attach();
            this._hideEvent.attach()
        },
        disable: function() {
            this._showEvent.detach();
            this._hideEvent.detach()
        },
        destroy: function() {
            this.node.parentNode.removeChild(this.node);
            this._showEvent.destroy();
            this._hideEvent.destroy();
            this._events.destroy();
            this._events = this._hideEvent = this._showEvent = this.content = this.tipSide = this.listenTo = this.node = null
        }
    });
    Splash.Hint = f
})(window, document, void 0);
(function(a, b, f) {
    function d(e, c) {
        var d = this, j, r, w, q, v, p;
        d.container = e;
        d.options = c;
        d.is_notswitched=!0;
        if (d.options.animation)
            a: {
            r = document.createElement("fakeelement");
            w = {
                WebkitAnimation: "webkitAnimationEnd",
                OAnimation: "oAnimationEnd",
                msAnimation: "MSAnimationEnd",
                animation: "animationend"
            };
            for (j in w)
                if (void 0 !== r.style[j]) {
                    j = w[j];
                    break a
                }
                j=!1
        } else 
            j=!1;
        d.animationend = j;
        d.transitionend = d.options.balloon && k() ? k() : !1;
        d.mask = b.id(c.mask.maskId);
        d.contentContainer = b.id(c.content.containerId);
        d.more = b.id(c.tabs.moreId);
        d.moreContainer = b.id(c.tabs.moreDropdownId);
        w = d.container.children[0].getElementsByTagName("td");
        d.__onClick = function(a) {
            d.is_notswitched && (delete d.is_notswitched, d.closeBalloon && d.closeBalloon());
            d._onClick(this, a)
        };
        if (d.options.extended) {
            q = b.id("news-list-slot-extend").children;
            d.clones = {};
            d.clones.informer = document.createElement("div");
            for (j = 0; j < q.length; j++)
                if ("SCRIPT" !== q[j].nodeName)
                    if ( - 1 !== q[j].className.indexOf("w-text-banner_outer")) {
                        p = document.createElement("div");
                        p.className = q[j].className;
                        v = q[j].children;
                        for (r = 0; r < v.length; r++)
                            "IMG" === v[r].nodeName || "SCRIPT" === v[r].nodeName || p.appendChild(v[r].cloneNode(!0));
                            - 1 === p.className.indexOf("w-text-banner_outer-hiddable") && (d.clones.banner = p.cloneNode(!0), d.clones.banner.className += "-hiddable");
                            d.clones.informer.appendChild(p)
                    } else 
                        d.clones.informer.appendChild(q[j].cloneNode(!0));
            if (d.options.balloon && void 0 == b.s_cookie.getGlobal("s_balloonshowed")
                ) {
                var s = b.id("extendNewsNotify"), x = b.id("extendNewsNotifyOpen"), g = b.id("extendNewsNotifyClose");
                d.closeBalloon = function() {
                    b.removeClass(s, "news__tabs__balloon_fadein");
                    b.addClass(s, "news__tabs__balloon_fadeout");
                    b.s_cookie.setGlobal("s_balloonshowed", "1");
                    b.counter("clb4364312");
                    d.transitionend ? b.bind(s, d.transitionend, function() {
                        b.removeClass(s, "news__tabs__balloon_show");
                        b.removeClass(s, "news__tabs__balloon_fadeout")
                    }) : (b.removeClass(s, "news__tabs__balloon_show"), b.removeClass(s, "news__tabs__balloon_fadeout"))
                };
                f.DOMEvent.addListener(a, "load", function() {
                    d.is_notswitched && (f.DOMEvent.addListener(x,
                    "click", function(a) {
                        var b = 0;
                        a.preventDefault();
                        if (!(0>+d.options.balloon_opentab))
                            for (; b < d.tabs.length; b++)
                                if ( + d.tabs[b].getAttribute("data-slot") === d.options.balloon_opentab) {
                                    d.__onClick.call(d.tabs[b], a);
                                    d.closeBalloon();
                                    break
                                }
                    }), f.DOMEvent.addListener(g, "click", function(a) {
                        a.preventDefault();
                        d.closeBalloon()
                    }), b.counter("d4364309"), b.addClass(s, "news__tabs__balloon_show"), setTimeout(function() {
                        b.addClass(s, "news__tabs__balloon_fadein")
                    }, 0))
                })
            }
        }
        d.tabs = w;
        for (j = 0; j < w.length; j++)
            d._initTab(w[j]);
        d._closeMore =
        d._closeMore.bind(d);
        f.DOMEvent.addListener(d.more, "click", d._openMore.bind(d))
    }
    function k() {
        var a, b = document.createElement("fakeelement"), d = {
            WebkitTransition: "webkitTransitionEnd",
            OTransition: "otransitionend",
            oATransition: "oTransitionEnd",
            msTransition: "msTransitionEnd",
            Transition: "transitionend"
        };
        for (a in d)
            if (void 0 !== b.style[a])
                return d[a];
        return !1
    }
    d.prototype = new f.EventEmitter;
    b.extend(d.prototype, {
        _initTab: function(a) {
            var b = this, d = a.getElementsByTagName("a")[0];
            b.isActive(a) && (b.activeTab = a);
            b._loadTabContent(a);
            b._initDouble(a);
            f.DOMEvent.addListener(d, "mousedown", function(d) {
                b.activeTab !== a && d.stopPropagation()
            });
            f.DOMEvent.addListener(d, "click", function(d) {
                b.__onClick.call(a, d)
            })
        },
        loaded: function() {
            this._waiting && this._waiting();
            this.emit("loaded")
        },
        _initDouble: function(a) {
            var c = this;
            b.hasClass(a, c.options.tabs.doubleTabClass) && (a._double = document.createElement("div"), a._double.className = c.options.tabs.moreDropdownItemClass, a._double.innerHTML = a.innerHTML.replace(RegExp(c.options.tabs.tabLinkClass,
            "g"), c.options.tabs.moreDropdownItemLinkClass), a._double.tab = a, f.DOMEvent.addListener(a._double.getElementsByTagName("a")[0], "click", function(b) {
                c.__onClick.call(a, b)
            }), c.moreContainer.appendChild(a._double))
        },
        _openMore: function(a) {
            !b.hasParent(a.target, this.moreContainer)&&!b.hasClass(this.more, this.options.tabs.moreOpenedClass) && (b.counter("clb818633"), b.addClass(this.more, this.options.tabs.moreOpenedClass), setTimeout(function() {
                f.DOMEvent.addListener(document, "click", this._closeMore)
            }.bind(this),
            0))
        },
        _closeMore: function() {
            b.removeClass(this.more, this.options.tabs.moreOpenedClass);
            f.DOMEvent.removeListener(document, "click", this._closeMore)
        },
        _loadTabContent: function(a) {
            var c, d, f, k;
            if (a._content)
                return !0;
            c = b.id("news:" + a.getAttribute("data-n"));
            if (!c)
                return !1;
            k = this.options.extended ? function() {
                var f, j, k, p, s = c.children[c.children.length - 1], x;
                if (s&&!s.children.length && null == s.getAttribute("data-type")) {
                    this.clones.informer = this.clones.informer.cloneNode(!0);
                    x = this.clones.informer.getElementsByTagName("a");
                    for (f = 0; f < x.length; f++)
                        if ((j = x[f].id)&&-1 < j.indexOf("regional-info-link-"))
                            k =+ j.substr(j.length - 1, 1) + 1, x[f].id = "regional-info-link-" + k, b.bind(x[f], "click", function() {
                                regionalSelector.show()
                            });
                    x = this.clones.informer.getElementsByTagName("span");
                    p = [];
                    for (f = 0; f < x.length; f++)
                        if (j = x[f].id)
                            x[f].id = j + "-" + k, p.push(x[f]);
                    x = null;
                    b.SetDate && new b.SetDate(p);
                    s.appendChild(this.clones.informer);
                    a._banner=!0
                }
                s && (!a._banner && this.clones.banner&&!b.hasClass(c.parentNode, this.options.content.tabClass)) && (d = this.clones.banner.cloneNode(!0),
                s.appendChild(d), a._banner=!0)
            } : function() {};
            b.hasClass(c.parentNode, this.options.content.tabClass) ? (a._content = c.parentNode, b.addClass(a._content, this.options.content.tabActiveClass), k.apply(this)) : (f = document.createElement("div"), f.className = this.options.content.tabClass, c.style.display = "", k.apply(this), this.contentContainer.appendChild(f), f.appendChild(c), a._content = f);
            return !0
        },
        _onClick: function(a, b) {
            a.tab && (a = a.tab);
            a&&!this.isActive(a) && (b.preventDefault(), this.activateTab(a))
        },
        isActive: function(a) {
            return b.hasClass(a,
            this.options.tabs.tabActiveClass)
        },
        activateTab: function(a) {
            var c = this, d = a.getAttribute("data-c"), f = this.activeTab, k = function() {
                c.isAnimating=!1;
                c.deactivateTab(f);
                b.unbind(a._content, c.animationend, k)
            };
            if (!c.isAnimating ||!c.animationend)
                c.isAnimating=!0, d && b.counter(d), f&&!c.animationend && c.deactivateTab(f), b.addClass(a, c.options.tabs.tabActiveClass), c._loadTabContent(a) ? (b.addClass(a._content, c.options.content.tabActiveClass), c.animationend && (Array.prototype.indexOf.call(f.parentNode.children, f) <
                Array.prototype.indexOf.call(a.parentNode.children, a) ? (b.addClass(a._content, c.options.content.animate.fromright), b.addClass(f._content, c.options.content.animate.toleft)) : (b.addClass(a._content, c.options.content.animate.fromleft), b.addClass(f._content, c.options.content.animate.toright)), b.removeClass(f, c.options.tabs.tabActiveClass), b.bind(a._content, c.animationend, k)), c._waiting = null, c.mask.style.display = "none") : (c.mask.style.display = "block", c._waiting = function() {
                    b.counter("d3745149");
                    c._loadTabContent(a);
                    c.isAnimating=!1;
                    b.removeClass(f._content, c.options.content.tabActiveClass);
                    b.removeClass(f, c.options.tabs.tabActiveClass);
                    b.addClass(a._content, c.options.content.tabActiveClass);
                    c.mask.style.display = "none"
                }), a._double ? (b.addClass(a._double, c.options.tabs.moreDropdownItemSelectedClass), b.addClass(c.more, c.options.tabs.moreSelectedClass)) : b.removeClass(c.more, c.options.tabs.moreSelectedClass), c.activeTab = a, c.tabsSwitched=!0, c.emit("tab:activate", a)
        },
        deactivateTab: function(a) {
            this._loadTabContent(a) &&
            b.removeClass(a._content, this.options.content.tabActiveClass);
            this.animationend && (b.removeClass(this.activeTab._content, this.options.content.animate.fromleft), b.removeClass(this.activeTab._content, this.options.content.animate.fromright));
            this._loadTabContent(a) && this.animationend && (b.removeClass(a._content, this.options.content.animate.toleft), b.removeClass(a._content, this.options.content.animate.toright));
            a._double && b.removeClass(a._double, this.options.tabs.moreDropdownItemSelectedClass);
            this.animationend ||
            (b.removeClass(a, this.options.tabs.tabActiveClass), this.activeTab = null)
        }
    });
    f.News = d
})(window, window.mr, window.Splash);
(function(a, b, f) {
    function d(a) {
        this.__id=++k;
        this.__content = a.content || "empty content";
        this.__tpl = a.tpl;
        this.__hovering = a.hovering ||!1;
        this.__afterInsert = a.afterInsert || null;
        this.__position = a.position || "center";
        this.__class_names = a.class_names;
        this.__showed=!1;
        this.once_close = a.once_close;
        this.counter_show = a.counter_show;
        this.counter_hide = a.counter_hide;
        this.setContent()
    }
    var k = 0, e = a.document;
    d.prototype.setContent = function(a) {
        var b = e.createElement("div");
        "string" === typeof a && (this.__content = a);
        this.tpl =
        this.__tpl.call(this);
        b.innerHTML = this.tpl;
        this.block = b.children[0]
    };
    d.prototype.setPosition = function(a) {
        this.__position = a || this.__position;
        b.addClass(this.block, "notify_" + this.__position)
    };
    d.prototype.show = function() {
        if (!(this.__showed || 1 === this.once_close)) {
            var a = this;
            this.__insertBlock();
            b.removeClass(this.block, "notify_hidden");
            ({
                center: function() {
                    this.block.style.marginLeft =- this.block.offsetWidth / 2 + "px"
                }
            })[this.__position].call(this);
            setTimeout(function() {
                b.addClass(a.block, "notify_show");
                a.__showed =
                !0
            }, 10);
            this.counter_show && b.counter(this.counter_show)
        }
    };
    d.prototype.hide = function() {
        this.once_close && (this.once_close = 1);
        b.removeClass(this.block, "notify_show");
        b.addClass(this.block, "notify_hidden");
        this.__showed=!1;
        this.counter_hide && b.counter(this.counter_hide)
    };
    d.prototype.__container = b.id("main-wrap");
    d.prototype.__insertBlock = function() {
        if (!this.__is_appended) {
            var a = this;
            this.__container.appendChild(this.block);
            this.__is_appended=!0;
            this.block_inner = b.id("notify:inner:" + this.__id);
            this.block_close =
            b.id("notify:close:" + this.__id);
            f.DOMEvent.on(this.block_close, "click", function(b) {
                b.preventDefault();
                a.hide()
            });
            null !== this.__afterInsert && "function" == typeof this.__afterInsert && this.__afterInsert.apply(this)
        }
    };
    f.Notify = d;
    f._notify = {}
})(window, window.mr, window.Splash);
(function() {
    var a = 0, b=!1, f = SG, d = function(a) {
        if (i) {
            var b = i.itemData, c = i.value, d = k.cre("form");
            d.action = "//go.mail.ru/search";
            d.method = "get";
            d.target = "_blank";
            k.objEach({
                q: b.link,
                sg: c,
                sgsig: b.sig,
                ce: 1,
                fm: 1
            }, function(a, b) {
                var c = k.cre("input");
                c.type = "hidden";
                c.name = b;
                c.value = a;
                d.appendChild(c)
            });
            document.body.appendChild(d);
            d.submit();
            setTimeout(function() {
                k.rme(d);
                d = null
            }, 20);
            a.preventDefault();
            return !1
        }
    }, k = f.utils, e = f.Event;
    k.suppDataURI=!1;
    if ( - 1 == navigator.userAgent.indexOf("MSIE 6")&&-1 == navigator.userAgent.indexOf("MSIE 7")) {
        var c =
        new Image;
        e.add(c, "load error", function(a) {
            e.rm(c, "load error");
            k.suppDataURI = "load" === a.type;
            c = null
        });
        c.src = "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw=="
    }
    f.SHOW_MUSICON=!0;
    var i = null;
    f.setup({
        enabled: !1,
        field: "#q",
        cont: ".go-suggests",
        list: ".go-suggests__items",
        switcher: "go-suggests_open",
        autoSubmit: !0,
        preview: !0,
        hover: "go-suggests__item_hover",
        max: 7,
        url: {
            scheme: "",
            authority: "suggests.go.mail.ru",
            path: "sg_u",
            query: {
                q: "{query}"
            }
        },
        ajax: {
            dataType: "jsonp",
            timeout: 3E3,
            data: {}
        },
        dataFilter: function(a) {
            if (a && (!a.items && (a.sites && a.sites.length) && (a.items = []), a.items)) {
                var b;
                for (b = a.items.length; b--;)
                    f.SHOW_MUSICON || delete a.items[b].mus;
                if (a.sites)
                    for (b = a.sites.length; b--;)
                        a.items.unshift(a.sites[b])
            }
            return a
        },
        item: ".go-suggests__suggest-tmpl",
        result: function(a) {
            return "site" === a.type ? a.site : a.text
        },
        select: function(c) {
            var d = c.itemData, e = c.value, f = this.form;
            a = 1;
            "site" === d.type ? (i = c, b=!0) : (c = {
                us: e.length,
                usln: c.selected + 1,
                usstr: encodeURIComponent(e)
            }, d.spcAuto && (c.ussp =
            d.spcAuto), d.mus && (c.usmus = 1, c.rch = "l", k.rme(f.elements.rch)), (d = k.$$(".js-srch-inp-to-del")) && k.isArr(d) && k.arrEach(d, function(a) {
                k.rme(a)
            }), k.objEach(c, function(a, b) {
                if (null != e) {
                    var c = k.cre("input");
                    c.type = "hidden";
                    c.name = b;
                    c.value = a;
                    c.className = "js-srch-inp-to-del";
                    f.appendChild(c)
                }
            }))
        },
        unselect: function() {
            i = null;
            b=!1
        },
        onEnable: function() {
            e.add(this.form, "submit", d)
        },
        onDisable: function() {
            e.rm(this.form, "submit", d)
        }
    });
    var j = mr.id("search"), r = mr.id("q"), w = mr.id("search__button__wrapper__field"),
    q = mr.id("search__projects-list"), v = [null, "615232", "615227", "615230"], p, s;
    Splash.DOMEvent.on(w, "mousedown", function() {
        a = 2
    });
    Splash.DOMEvent.on(j, "submit", function(c) {
        clearTimeout(p);
        p = setTimeout(function() {
            j._submited=!1
        }, 300);
        if (j._submited)
            return c.preventDefault(), !1;
        if (/^\s*$/.test(r.value))
            return r.focus(), c.preventDefault(), !1;
        1 != a && 2 != a && (a = 3);
        mr.counter("clb" + v[a]);
        a = 0;
        j._submited=!0;
        b || (clearTimeout(s), s = setTimeout(function() {
            j.submit()
        }, 100));
        c.preventDefault();
        return !1
    });
    Splash.DOMEvent.on(q,
    "click", function(a) {
        var b = a.target, c = r.value, d = mr.data(b, "search-href"), e = mr.data(b, "search-param");
        "A" === b.tagName && (c && d) && (e ? (b = document.createElement("form"), b.action = d, b.method = "get", b.setAttribute("accept-charset", "windows-1251"), d = document.createElement("input"), d.type = "hidden", d.name = e, d.value = c, b.appendChild(d), document.body.appendChild(b), b.submit()) : location.href = d + encodeURIComponent(c), a.preventDefault())
    });
    SG().enable()
})();
(function() {
    var a = mr.id("Auth"), b = mr.id("js-unread"), f = mr.tag("*", b, "js-txt")[0], d = mr.id("js-readed"), k = mr.id("js-mailbox-user"), e = mr.id("js-mailbox-avatar"), c = mr.id("js-mailbox-writemail"), i = mr.id("js-mailbox-exit"), j = mr.id("mailbox__login"), r = mr.id("mailbox__login__domain"), w = mr.id("domain-widget"), q = mr.id("mailbox__password"), v = mr.id("mailbox__auth__remember__checkbox"), p = mr.id("js-mailbox-enter"), s = {
        ActiveEmail: k.innerHTML,
        TotalUnread: parseInt(f.innerHTML)
    }, x = new Splash.EventEmitter, g = mr.id("js-unread-icon");
    mr.data(a, "loadsubmit-counter", "d703061");
    window.Splash.ajaxUpdates.push({
        data: "mailbox",
        cb: function(a) {
            s && a.hasAuth !== s.hasAuth && x.emit("switch", {
                data: a,
                oldData: s
            });
            mr.activeEmail = a && a.ActiveEmail || "";
            p.href = a.HasAuth ? mr.data(p, "enter-href") : mr.data(p, "login-href");
            if (a.HasAuth) {
                k.innerHTML = a.ActiveEmail;
                mr.isMultiAuth && (mr.multiuser.innerHTML = a.ActiveEmail);
                a.ActiveEmail !== s.ActiveEmail && (e.style.backgroundImage = "url(//filin.mail.ru/pic?from=splash&email=" + a.ActiveEmail + (mr.retina ? "&width=90&height=90" :
                "&width=45&height=45") + (a.fullName ? "&name=" + encodeURIComponent(a.fullName) : "") + ")");
                e.name = a.TotalUnread ? mr.data(e, "counter-unread") : mr.data(e, "counter-readed");
                c.name = a.TotalUnread ? mr.data(c, "counter-unread") : mr.data(c, "counter-readed");
                i.name = a.TotalUnread ? mr.data(i, "counter-unread") : mr.data(i, "counter-readed");
                if (a.TotalUnread) {
                    if (f.innerHTML = a.TotalUnread + " " + _plural(a.TotalUnread, Splash.data.mailbox.newL) + " " + _plural(a.TotalUnread, Splash.data.mailbox.letters), b.style.display = "inline", d.style.display =
                    "none", a.ActiveEmail !== s.ActiveEmail || a.TotalUnread > s.TotalUnread)
                        clearTimeout(E), mr.removeClass(g, "animated"), mr.removeClass(g, "animated_flash"), setTimeout(function() {
                            mr.addClass(g, "animated");
                            mr.addClass(g, "animated_flash")
                        }, 0), E = setTimeout(function() {
                            mr.removeClass(g, "animated");
                            mr.removeClass(g, "animated_flash")
                        }, 2E3)
                    } else 
                        d.style.display = "inline", b.style.display = "none";
                mr.removeClass(document.body, "status__user_hasNoAuth");
                mr.addClass(document.body, "status__user_hasAuth")
            } else {
                if (document.activeElement &&
                - 1 !== document.activeElement.className.indexOf("mailbox"))
                    return;
                var u = null, B = null;
                a.User && (u = a.User + "@" + a.UserDomain);
                j.value && (B = j.value);
                u != B && (q.value = "");
                mailboxLoginPlaceholder.toggle();
                for (mailboxPasswordPlaceholder.toggle(); r && r.firstChild;)
                    r.removeChild(r.firstChild);
                if (a.Domains[0].Domain) {
                    mr.domains = [];
                    for (var u = a.Domains.length, B = null, C = 0; C < u; C++)
                        B = a.Domains[C].Domain, mr.domains.push(B)
                    } else 
                        mr.domains = a.Domains, mr.extDomains = a.ExtDomains;
                u = 0;
                for (B = mr.domains.length; u < B; u++)
                    if (C = mr.domains[u],
                    r) {
                        var D = document.createElement("option");
                        D.value = C;
                        D.selected = a.UserDomain == C;
                        D.innerHTML = "@" + C;
                        r.appendChild(D)
                    }
                if (mr.externalDomainsEnabled) {
                    D = document.createElement("option");
                    D.disabled=!0;
                    D.innerHTML = "------";
                    r.appendChild(D);
                    u = 0;
                    for (B = mr.extDomains.length; u < B; u++)
                        C = mr.extDomains[u], r && (D = document.createElement("option"), D.value = C, D.selected = a.UserDomain == C, D.innerHTML = "@" + C, r.appendChild(D))
                    } else 
                        a.UserDomainExt && (u = document.createElement("option"), u.value = a.UserDomainExt, u.selected=!0, u.innerHTML =
                        "@" + a.UserDomainExt, r.appendChild(u));
                w && w.mrChange();
                u = __PH.cookie.s.getGlobal("a") || mr.cookie("loginS");
                v.checked = "string" !== typeof u || "1" === u;
                v.mrChange();
                x.emit("reset", a);
                mr.removeClass(document.body, "status__user_hasAuth");
                mr.addClass(document.body, "status__user_hasNoAuth")
            }
            !a.HasAuth&&!a.User ? mr.addClass(document.body, "status__user_hasNoAuth_noCookie") : mr.removeClass(document.body, "status__user_hasNoAuth_noCookie");
            try {
                __PH && __PH.updateCounter({
                    mail: a.TotalUnread || 0
                })
            } catch (K) {}
            s = a
        }
    });
    var K =
    mr.id("password_warning"), B = new Splash.Hint("", K);
    B.on("show", function() {
        mr.counter("d721322")
    });
    x.on("switch", B.hide);
    x.on("reset", function() {
        q.value || (B.hide(), K.style.display = "none")
    });
    new function() {
        this.input = q;
        this.allowedSymbols = /^[0-9a-zA-Z!@#$%^&*()\-<>'"_+=;:,./?\\|`~[\]{}]+$/i;
        this.wrongAlpha = /[\u0430-\u044f\u0410-\u042f\u0454\u0457\u0491\u0456]/i;
        this.wrongSymbolsMsg = "\u0412\u044b \u0432\u0432\u0435\u043b\u0438<br/>\u043d\u0435\u0434\u043e\u043f\u0443\u0441\u0442\u0438\u043c\u044b\u0435 \u0441\u0438\u043c\u0432\u043e\u043b\u044b";
        this.wrongBadLangMsg = "\u0421\u043c\u0435\u043d\u0438\u0442\u0435 \u0440\u0430\u0441\u043a\u043b\u0430\u0434\u043a\u0443";
        this.warningEvent = new Splash.DOMEvent(this.input, "keyup", mr.proxy(function() {
            this.input.value&&!this.allowedSymbols.test(this.input.value) ? (this.wrongAlpha.test(this.input.value) ? B.setContent(this.wrongBadLangMsg) : B.setContent(this.wrongSymbolsMsg), K.style.display = "block", mr.counter("d755623")) : (B.hide(), K.style.display = "none")
        }, this));
        this.warningEvent.attach()
    };
    var E
})();
(function() {
    var a = mr.id("js-my-messages"), b = mr.id("js-my-messages-counter"), f = mr.id("js-my-messages-link"), d = mr.id("js-my-messages-icon"), k = mr.id("js-my-friendships"), e = mr.id("js-my-friendships-counter"), c = mr.id("js-my-friendships-link"), i = mr.id("js-my-friendships-icon");
    window.Splash.ajaxUpdates.push({
        data: "my",
        cb: function(j) {
            j.HasMy && (b.innerHTML = j.MyMessages_cnt, e.innerHTML = j.MyFriendships_cnt, f.title = j.MyMessages_cnt ? j.MyMessages_cnt + " " + _plural(j.MyMessages_cnt, Splash.data.my.messages[0]) : Splash.data.my.messages[1],
            c.title = j.MyFriendships_cnt ? j.MyFriendships_cnt + " " + _plural(j.MyFriendships_cnt, Splash.data.my.friendships[0]) + Splash.data.my.friendships[1] : Splash.data.my.friendships[2], c.href = c.href.split("?")[0] + "?from=splash&requests=" + j.MyFriendships_cnt + "#notify=people", mr["0" == j.MyMessages_cnt ? "removeClass": "addClass"](d, "icon_counter_my-messages_unread"), mr["0" == j.MyFriendships_cnt ? "removeClass": "addClass"](i, "icon_counter_my-friendship_unread"));
            a.style.display=!j.MyMessages_cnt ? "none" : "";
            k.style.display =
            !j.MyFriendships_cnt ? "none" : ""
        }
    })
})();
mr.mraSuggest && function(a) {
    function b(a) {
        if (!d) {
            clearTimeout(k);
            mr.counter("d929045");
            var b = mr.id("mraSuggest"), f = mr.id("mraTitle"), j = mr.id("mraDescription"), r = a.items && a.items.length;
            try {
                if ("ok" === a.status && 0 != r) {
                    for (var w = 0; w < r; w++) {
                        var q = a.items[w], v = document.createElement("a"), p = "http://agent.mail.ru/friends?partnerid=ava", s = 1 == r ? "\u0443\u0436\u0435 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442": "\u0443\u0436\u0435 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u044e\u0442";
                        v.className =
                        "social_mra__suggest__avatars__image";
                        v.href = p;
                        v.style.backgroundImage = "url(//avt.foto.mail.ru/" + q.photoBox + "/_avatar22)";
                        v.title = q.nick;
                        v.name = "clb877223";
                        b.appendChild(v)
                    }
                    j.innerHTML = s;
                    j.href = p;
                    j.name = "clb877221";
                    f.href = p;
                    f.name = "clb877221";
                    b.style.visibility = "visible";
                    mr.counter("d876981")
                } else 
                    "ok" === a.status && 0 == r ? mr.counter("d876980") : mr.counter("d876974")
                } catch (x) {
                window.logError && logError(x, "createMraSuggests"), mr.counter("d929044")
            }
            j.style.visibility = "visible";
            d=!0
        }
    }
    function f() {
        k = setTimeout(function() {
            b({
                status: "error"
            })
        },
        750);
        mr.getScript("//mrimraker1.mail.ru/agent_suggest?json=1&callback=getMraFriends", !1, "utf-8")
    }
    var d=!1, k;
    a.getMraFriends = function(a) {
        b(a)
    };
    "opera"in a ? a.addEventListener("DOMContentLoaded", function() {
        f()
    }, !1) : f()
}(window, document);
(function() {
    var a = {}, b = 0, f = [], d, k;
    window.reloadSlots = function(e, c) {
        return function() {
            var i = "reloadSlots" + b++;
            window[i] = function(b) {
                try {
                    var c = b.length, e, f, i, j, g;
                    for (d = 0; d < c; d++) {
                        g = b[d];
                        e = g.slot;
                        j = mr.id("slot_" + e);
                        f = g.banner;
                        var k = g.html, r = g.slot || "", E = "u" + r + (new Date).getTime() * parseInt(1E5 * Math.random()), k = k.replace(/(<script[^>]*>)\s*(<\!--|\/\/\s*<!\[CDATA\[)/gi, "$1;").replace(/\/\/\s*(--\>|\]\]>)\s*(<\/script>)/gi, ";$2");
                        i = k = "<script>window.qwe" + E + r + ' = "";<\/script>' + k.replace(/\w+\.write(ln)?\(/ig,
                        "qwe" + E + r + "+=(").replace(/<noscript>/ig, "<\!--").replace(/<\/noscript>/ig, "--\>") + '<span id="' + E + r + '"></span><script>mr.id("' + E + r + '").innerHTML = "<i style=\'display:none;\'>&nbsp;</i>" + qwe' + E + r + ";<\/script>";
                        null != j.getAttribute("data-exb") && j.setAttribute("data-exb", f);
                        j.innerHTML = '<i style="display:none;">&nbsp;</i>' + i;
                        setTimeout(function(b) {
                            return function() {
                                try {
                                    var c = b.getElementsByTagName("script"), d = function(b) {
                                        function e() {
                                            c[b + 1] && d(b + 1)
                                        }
                                        var f = c[b], g = f.text || f.textContent || f.innerHTML ||
                                        "", i = document.getElementsByTagName("head")[0] || document.documentElement, j = document.createElement("script");
                                        j.type = "text/javascript";
                                        if (f.src)
                                            a[f.src] ? e() : (j.setAttribute("src", f.src + "?rnd=" + Math.random()), j.onload = function() {
                                                i.removeChild(j);
                                                a[f.src] = 1;
                                                e()
                                            }, i.insertBefore(j, i.firstChild));
                                        else {
                                            try {
                                                j.appendChild(document.createTextNode(g))
                                            } catch (k) {
                                                j.text = g
                                            }
                                            i.insertBefore(j, i.firstChild);
                                            i.removeChild(j);
                                            e()
                                        }
                                    };
                                    c.length && d(0)
                                } catch (e) {
                                    window.logError && logError(e, "execInnerScripts")
                                }
                            }
                        }(j), 0)
                    }
                } catch (l) {
                    window.logError &&
                    logError(l, "reloadSlots")
                }
            };
            setTimeout(function() {
                window[i] = function() {}
            }, 6E4);
            var j = [];
            if (c)
                for (var r in c)
                    c.hasOwnProperty(r) && j.push([r, c[r]].join("="));
            d = 0;
            for (k = e.length; d < k; d++)(r = mr.id("slot_" + e[d]).getAttribute("data-exb")
                ) && f.push(r);
            r = "//ad.mail.ru/adq/?callback=" + i + "&q=" + e.join("&q=");
            _EXPERIMENTID && (r += "&test_id=" + _EXPERIMENTID);
            _PREVIEW && (r += "&preview=" + _PREVIEW);
            j.length && (r += "&" + j.join("&"));
            f.length && (r += "&exb=" + f.join(";"), f.length = 0);
            mr.getScript(r, !1, "utf-8")
        }
    }
})();
(function() {
    function a(a, c) {
        if (!a.data.email&&!mr.activeEmail && 0!==++d%5)
            return c(), !1;
        f = c;
        var e = mr.ajaxUrl + (window.__PH && a.data && a.data.email ? "&Login=" + encodeURIComponent(a.data.email) : "") + "&rnd=" + (new Date).getTime() + Math.random();
        percent(mr.LF1Percent) && ((new Image).src = mr.getLf1Url({
            zero: 1
        }, e));
        mr.getJson(e, {}, b);
        return !1
    }
    function b(a) {
        try {
            if (a)
                for (var b = 0, c = window.Splash.ajaxUpdates.length; b < c; b++) {
                    var d = window.Splash.ajaxUpdates[b];
                    d.cb(d.data ? a[d.data] : a)
                }
            } catch (e) {
            window.logError && logError(e,
            "getSplash")
        }
        "function" === typeof f && f()
    }
    var f = function() {}, d = 0;
    if (__mailUpdated)
        __mailUpdated=!0, __PH.on("update", function(b, c) {
            a(b, c)
        });
    else 
        __PH.once("update", function(b, c) {
            __mailUpdated=!0;
            __PH.on("update", function(b, c) {
                a(b, c)
            });
            c()
        });
    __PH.on("authChange", function(b, c) {
        a(b, c);
        k()
    });
    var k = reloadSlots([4015, 217, 4499]);
    if (mr.onFocusBannerReload) {
        var e=!1;
        __PH.on("visibilitychange", function(a) {
            a.data.visible && e && (c(), k(), mr.counter("d1464827"))
        });
        var c = function() {
            e=!1;
            setTimeout(function() {
                e=!0
            }, 6E5)
        };
        c()
    }
})();
(function() {
    var a;
    a=!0;
    "O" === mr.vendor && (a = navigator.userAgent.match(/Version\/([0-9]+)/), a = null !== a && 12 < a[1]);
    a = {
        extended: newsTabs && newsTabs.extended?!0: !1,
        animation: newsTabs && newsTabs.animation && a?!0: !1,
        balloon: newsTabs && newsTabs.balloon?!0: !1,
        balloon_opentab: newsTabs && newsTabs.balloon_opentab ? newsTabs.balloon_opentab: !1,
        tabs: {
            tabClass: "news__tabs__item",
            tabActiveClass: "news__tabs__item_selected",
            doubleTabClass: "news__tabs__item_double",
            tabLinkClass: "news__tabs__item__link",
            moreId: "news__more",
            moreIconId: "news__more__icon",
            moreDropdownId: "news__more__dropdown",
            moreOpenedClass: "news__tabs__more_opened",
            moreSelectedClass: "news__tabs__more_selected",
            moreDropdownItemClass: "news__tabs__more__dropdown__item",
            moreDropdownItemSelectedClass: "news__tabs__more__dropdown__item_selected",
            moreDropdownItemLinkClass: "news__tabs__more__dropdown__item__link"
        }, content : {
            containerId: "news__wrap", tabClass : "news__list", tabActiveClass : "news__list_active", slotClass : "news__list__slot", animate : {
                fromleft: "news__list_fromleft", fromright : "news__list_fromright",
                toleft : "news__list_toleft", toright : "news__list_toright"
            }
        }, mask: {
            maskId: "news__mask"
        }
    }; mr.cookie("newsT", null);
    window.newsTabs = new Splash.News(mr.id("news"), a)
})();
function splashMultiAuthInit() {
    mr.isMultiAuth = "isMultiAuth"in __PH&&!0 === __PH.isMultiAuth();
    if (mr.isMultiAuth) {
        mr.multiuser = mr.id("MA_user-email");
        var a = Splash, b = function(b) {
            mr.hasClass(c, i) && (clearTimeout(r), mr.removeClass(c, i), c.style.width = "", a.DOMEvent.removeListener(document, "click", f), b && b.preventDefault())
        }, f = function(a) {
            var e;
            mr.hasParent(a.target, c) ? (e = mr.hasParent(a.target, "js-logout") || mr.hasClass(a.target, "js-logout") && (e = a.target)) ? (__PH.logoutAccount(mr.hasParent(e, "js-list-user").getAttribute("data-email"),
            d), mr.counter("clb1199007"), a.preventDefault()) : (e = mr.hasParent(a.target, "js-list-user") || mr.hasClass(a.target, "js-list-user") && (e = a.target)) ? (__PH.switchAccount(e.getAttribute("data-email")), mr.counter("clb1195611"), a.preventDefault(), b()) : mr.hasParent(a.target, "js-button") || mr.hasClass(a.target, "js-button") && a.target ? (a.preventDefault(), b()) : mr.hasParent(a.target, "js-login") || mr.hasClass(a.target, "js-login") && a.target ? (mr.counter("clb1195612"), b(), mr.isPHAuthForm && (a.preventDefault(), __PH.authForm.show(j))) :
            mr.isNode(a.target) && "a" === a.target.tagName.toLowerCase() && b() : b()
        }, d = function(a) {
            var b = mr.id("multiAuthMenuList");
            if ("error" !== a.status) {
                a = a.accounts;
                1 < a.length ? (mr.addClass(c, "multiauth__menu_multi"), 9 < a.length ? mr.addClass(c, "multiauth__menu_limit") : mr.removeClass(c, "multiauth__menu_limit")) : (mr.removeClass(c, "multiauth__menu_multi"), mr.removeClass(c, "multiauth__menu_limit"));
                for (var d = [], e = 0, f = a.length; e < f; e++) {
                    var i = "", g = a[e], j = "";
                    g.firstName && g.lastName && (j = (g.firstName + " " + g.lastName).replace(/</g,
                    "&lt;").replace(/>/g, "&gt;"));
                    0 === e ? i = " multiauth__menu__dropdown__list__item_first" : e === a.length && (i = " multiauth__menu__dropdown__list__item_last");
                    g.active && (i += " multiauth__menu__dropdown__list__item_current");
                    var k = g.counters && g.counters.mail;
                    d.push("<div" + (!g.active ? ' tabindex="0"' : "") + ' class="multiauth__menu__dropdown__list__item' + (!g.active ? " js-list-user" : "") + i + '" data-email="' + g.email + '" onkeypress="if ((event.which || event.charCode || event.keyCode) === 13){var t = (event.target || event.srcElement); t.nodeType == 3 && (t = t.parentNode); t.click();}"><div class="multiauth__menu__dropdown__list__item__avatar" style="background-image:url(//filin.mail.ru/pic?from=splash&email=' +
                    g.email + (mr.retina ? "&width=90&height=90" : "&width=45&height=45") + "&name=" + encodeURIComponent((g.firstName ? g.firstName.replace(/'/g, "&#39;").replace(/"/g, "&#34;").replace(/\(/g, "&#40;").replace(/\)/g, "&#41;") + " " : "") + (g.lastName.replace(/'/g, "&#39;").replace(/"/g, "&#34;").replace(/\(/g, "&#40;").replace(/\)/g, "&#41;") || "")) + ');"></div><div class="multiauth__menu__dropdown__list__item__info"><div class="multiauth__menu__dropdown__list__item__info__email">' + (!g.active ? '<div class="multiauth__menu__dropdown__list__item__info__email__counter' +
                    (k ? " multiauth__menu__dropdown__list__item__info__email__counter_visible" : "") + '"><i class="multiauth__menu__dropdown__list__item__info__email__counter__icon"></i><span class="multiauth__menu__dropdown__list__item__info__email__counter__text">' + k + "</span></div>" : "") + '<div class="multiauth__menu__dropdown__list__item__info__email__text">' + g.email + '</div></div><div class="multiauth__menu__dropdown__list__item__info__text">' + j + '</div></div><div class="multiauth__menu__dropdown__list__item__logout js-logout"><span class="portal-button portal-button_only-icon"><span class="portal-button__fake"><span class="portal-button__fake__icon portal-button__fake__icon_multiauth portal-button__fake__icon_multiauth_exit"></span></span></span></div></div>')
                }
                b.innerHTML =
                d.join("")
            }
        }, k = mr.id("js-mailbox-user"), e = mr.id("js-mailbox-user-wrap"), c = mr.id("multiAuthMenu"), i = "multiauth__menu_open", j = {};
        mr.id("multiAuthHolder").appendChild(c);
        mr.addClass(e, "w-mailbox__userinfo__email_multiauth");
        a.DOMEvent.on(k, "click", function(e) {
            if (mr.hasClass(c, i))
                b(e);
            else if (!mr.hasClass(c, i)) {
                var j = function() {
                    k && (mr.addClass(c, i), a.DOMEvent.on(document, "click", f), k=!1)
                }, k=!0, p;
                mr.addClass(c, "multiauth__menu__dropdown_loading");
                mr.removeClass(c, "multiauth__menu__dropdown_error");
                p = setTimeout(j,
                100);
                __PH.loadAccountsList(function(a) {
                    clearTimeout(p);
                    "timeout" === a.status || "error" === a.status ? (mr.removeClass(c, "multiauth__menu__dropdown_loading"), mr.addClass(c, "multiauth__menu__dropdown_error")) : (d(a), mr.removeClass(c, "multiauth__menu__dropdown_loading"));
                    j();
                    c.style.width = c.offsetWidth + "px"
                });
                mr.counter("clb1195610");
                e && e.preventDefault()
            }
        });
        a.DOMEvent.on(c, "mouseover", function() {
            clearTimeout(r)
        });
        a.DOMEvent.on(c, "mouseout", function() {
            clearTimeout(r);
            r = setTimeout(b, 4E3)
        });
        __PH.on("loginRequest",
        function(a, b) {
            try {
                __PH.authForm.show(j), b()
            } catch (c) {
                b(c)
            }
        });
        var r
    }
}(function(a, b) {
    function f(a) {
        a.preventDefault();
        mr.slidotype_opened ? e() : k()
    }
    function d(a) {
        a = "wheelDelta"in a ? a.wheelDelta : a.detail;
        mr.slidotype_opened&&!(void 0 != a && 0 < a) && (clearTimeout(q), q = setTimeout(function() {
            if (0 >= r.clientHeight - i.clientHeight - (void 0 !== window.pageYOffset ? window.pageYOffset : (document.documentElement || document.body.parentNode || document.body).scrollTop)
                )clearTimeout(w), setTimeout(function() {
                e()
            }, 200)
        }, 10))
    }
    function k() {
        mr.slidotype_opened || (clearTimeout(p), p = setTimeout(function() {
            mr.slidotype_opened =
            !0;
            mr.addClass(v, "slidotype_open");
            mr.addClass(j, "layout_slidotype");
            mr.slidotype_id && mr.counter("d" + mr.slidotype_id)
        }, 10))
    }
    function e() {
        if (!mr.slidotype_opened)
            return !1;
        clearTimeout(w);
        w = setTimeout(function() {
            mr.slidotype_opened=!1;
            mr.removeClass(v, "slidotype_open");
            mr.removeClass(j, "layout_slidotype");
            mr.counter("clb11790322")
        }, 10)
    }
    var c = mr.id("slidotype:fold");
    if (a && c) {
        var i = document.body, j = mr.id("layout"), r = mr.id("main-wrap"), w, q, v = mr.id("slidotype");
        if (a)
            Splash.DOMEvent.on(a, "click", f);
        if (b)
            Splash.DOMEvent.on(b,
            "click", f);
        Splash.DOMEvent.on(c, "click", function(a) {
            a.preventDefault();
            e()
        });
        Splash.DOMEvent.on(window, "mousewheel", d);
        Splash.DOMEvent.on(document, "mousewheel", d);
        Splash.DOMEvent.on(document, "DOMMouseScroll", d);
        Splash.DOMEvent.on(window, "scroll", d);
        var p;
        mr.slidotypeClicked && k()
    }
})(mr.logoTrigger, mr.slidoTrigger);
Splash.tpl.notify = function() {
    function a(a) {
        K(a + '\nin block "' + p + '" at line: ' + v + "\nfile: " + q)
    }
    function b(a) {
        return g[a]
    }
    function f(a) {
        if ("string" === typeof a) {
            if (x.test(a))
                return a.replace(s, b)
        } else if ("undefined" === typeof a)
            return "";
        return a
    }
    var d = "", k = [], e, c = [], i, j, r = "", w = {}, q = "", v = "", p = "", s = /[&<>"]/g, x = /[&<>"]/, g = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;"
    }, K;
    K = "undefined" === typeof __fest_error ? "undefined" !== typeof console && console.error ? function() {
        return Function.prototype.apply.call(console.error,
        console, arguments)
    } : function() {} : __fest_error;
    try {
        c[0] = f(this.__id)
    } catch (B) {
        c[0] = "", a(B.message)
    }
    try {
        c[1] = f(" notify_" + this.__position + (this.__hovering ? " notify_hovering" : ""))
    } catch (E) {
        c[1] = "", a(E.message)
    }
    d += '<div id="notify:' + c[0] + '" class="notify notify_hidden' + c[1] + '">';
    try {
        c[0] = f(this.__id)
    } catch (l) {
        c[0] = "", a(l.message)
    }
    d += '<div id="notify:close:' + c[0] + '" class="notify__close"></div>';
    try {
        c[0] = f(this.__id)
    } catch (u) {
        c[0] = "", a(u.message)
    }
    d += '<div id="notify:inner:' + c[0] + '" class="notify__inner">';
    try {
        d += this.__content
    } catch (Z) {
        a(Z.message + "3")
    }
    d += "</div></div>";
    if (i = k.length) {
        for (c = 0; c < i; c++)
            if (e = k[c], "string" === typeof e)
                r += e;
            else if (j = w[e.name]) {
                var C = e.params;
                if (e.cp)
                    for (e in e = void 0, C)
                        "function" == typeof C[e] && C[e].param && (C[e] = C[e]());
                        e = j.call(this, C);
                        r += e
            }
        return r + d
    }
    return d
};
if (mr.up) {
    var blocks = null, now = new Date, h = now.getHours(), m = now.getMinutes();
    Splash._notify.up = new Splash.Notify({
        hovering: !0,
        once_close: !0,
        tpl: Splash.tpl.notify,
        content: '<span class="notify__up">\u041f\u043e\u044f\u0432\u0438\u043b\u0438\u0441\u044c \u0441\u0432\u0435\u0436\u0438\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0438\t<span id="notify:up:refresh" class="notify__up__refresh">\u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c</span></span>',
        afterInsert: function() {
            Splash.DOMEvent.on(this.block,
            "click", function(a) {
                a.target&&-1 < a.target.id.indexOf("notify:close") || (mr.counter("clb10646802"), window.location.reload())
            })
        },
        counter_show: "d10646802"
    });
    Splash.ajaxUpdates.push({
        data: "update",
        cb: function(a) {
            var b, f;
            if (!blocks)
                for (b in blocks = {}, a.news)
                    blocks[b] = mr.id("news:" + b);
            for (b in blocks)
                f = a.news[b], slot = blocks[b].getAttribute("data-s"), blocks[b].getAttribute("data-b"), f.is_default && mr.timestamp < f.up && Splash._notify.up.show()
        }
    })
}
splashRadar("JSInit", 1);
splashRadar._JSINITTime = (new Date).getTime() - splashRadar._JSINITTime;
mr.scriptLoaded=!0;
window.onload = function() {
    onloadRadar("load", 1)("all", 1)(!0)
};

