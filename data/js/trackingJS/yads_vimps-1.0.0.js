/* UADF-Core 0.10.0 (built at 2014-08-29 13:20:38) */
!function(t) {
    "use strict";
    function e() {
        return s.__moduleMap || (s.__moduleMap = {}), s.__moduleMap
    }
    function i(t, i, n) {
        var a = e();
        a[t] || (a[t] = {
            name: t,
            deps: i,
            callback: n,
            initialized: null
        })
    }
    for (var n = "YJ_UADF", a=!1, o = t; t.top != o;) {
        o = o.parent;
        try {
            {
                o.document.URL
            }
        } catch (r) {
            a=!0;
            break
        }
    }
    if (a) {
        var s = {}, h = function() {}, p = function() {}, c = function() {};
        s.init = function() {
            return !1
        }
    } else {
        var s = function(e) {
            return t.top[e] || (t.top[e] = {}), t.top[e]
        }(n), h = function(t) {
            if (t.initialized)
                return t.initialized;
            for (var i = e(), n = [], a = 0; a < t.deps.length; a++) {
                var o = t.deps[a];
                n.push(h(i[o]))
            }
            var r = t.callback.apply(t, n);
            return t.initialized = r ? r : {}, t.initialized
        }, p = function(t) {
            var i = e();
            if (!i[t])
                return null;
            var n = h(i[t]);
            return n
        }, c = function(t, e) {
            for (var i in e)
                e.hasOwnProperty(i) && (t[i] = e[i])
        };
        s.init = function() {
            return p("core/uadf"), !0
        }
    }
    i("core/loader", [], function() {}), i("dev/flags", [], function() {
        return {
            version: "0.10.0",
            debug: !1,
            debugLevel: 0
        }
    }), i("core/core_util", ["dev/flags"], function() {
        return {
            inherit: function(t, e) {
                if (Object.create)
                    t.prototype = Object.create(e.prototype);
                else {
                    var i = function() {};
                    i.prototype = e.prototype, t.prototype = new i
                }
                t.prototype.constructor = t, t.super_ = e
            },
            isInherited: function(t, e) {
                for (var i = t; ;) {
                    if (i === e)
                        return !0;
                    if (!i.super_)
                        return !1;
                    i = i.super_
                }
            },
            isArray: function(t) {
                return "[object Array]" === Object.prototype.toString.call(t)
            },
            isString: function(t) {
                return "[object String]" === Object.prototype.toString.call(t)
            },
            isNumber: function(t) {
                return "[object Number]" === Object.prototype.toString.call(t)
            },
            forEach: function(t, e, i) {
                if (t.forEach)
                    t.forEach(e, i);
                else 
                    for (var n = 0; n < t.length; n++)
                        e.call(i, t[n], n, t)
            },
            map: function(t, e, i) {
                if (t.map)
                    return t.map(e, i);
                for (var n = [], a = 0; a < t.length; a++)
                    n.push(e.call(i, t[a], a, t));
                return n
            },
            filter: function(t, e, i) {
                if (t.filter)
                    return t.filter(e, i);
                for (var n = [], a = 0; a < t.length; a++) {
                    var o = e.call(i, t[a], a, t);
                    o && n.push(t[a])
                }
                return n
            },
            forEachHash: function(t, e, i) {
                for (var n in t)
                    t.hasOwnProperty(n) && e.call(i, n, t[n], t)
            },
            formatObject: function(t, e, i, n, a) {
                var o = "";
                for (var r in t)
                    if (t.hasOwnProperty(r)) {
                        o.length > 0 && (o += i);
                        var s = n ? n(r): r, h = a ? a(t[r]): t[r];
                        o += s + e + h
                    }
                return o
            },
            extend: function(t) {
                for (var e = 1; e < arguments.length; e++) {
                    var i = arguments[e];
                    for (var n in i)
                        i.hasOwnProperty(n) && (t[n] = i[n])
                }
                return t
            },
            makeRandomString: function() {
                return Math.floor(1e9 * Math.random()).toString(32)
            }
        }
    }), i("core/dom_util", ["dev/flags", "core/core_util"], function(e, i) {
        function n(t) {
            this.top = t.top, this.left = t.left, t.width ? (this.width = t.width, this.height = t.height, this.bottom = this.top + this.height, this.right = this.left + this.width) : (this.bottom = t.bottom, this.right = t.right, this.width = this.right - this.left, this.height = this.bottom - this.top)
        }
        function a(t) {
            return t.ownerDocument.defaultView || t.ownerDocument.parentWindow
        }
        function o(t, e) {
            function i(e) {
                var n = t(e);
                if (!n)
                    return !1;
                for (var a = e.frames, o = 0; o < a.length; o++) {
                    var r = a[o];
                    try {
                        {
                            r.document.URL
                        }
                    } catch (s) {
                        continue
                    }
                    var h = i(r);
                    if (!h)
                        return !1
                }
                return !0
            }
            e = e ? e : g, i(e)
        }
        function r(t) {
            return t.currentStyle || a(t).getComputedStyle(t, "")
        }
        function s(t) {
            var e = r(t);
            return e.overflow.indexOf("scroll") >= 0 || e.overflow.indexOf("hidden") >= 0?!0 : !1
        }
        function h(t, e, i) {
            e = e ? e : 0, i = i ? i : 0;
            for (var o = 0, r = 0, s = t; ;) {
                var h = a(s).frameElement;
                if (!h)
                    break;
                var p = h.getBoundingClientRect();
                o += p.top, r += p.left, s = h
            }
            var c = t.getBoundingClientRect();
            return new n({
                top: Math.round(c.top + i + o),
                left: Math.round(c.left + e + r),
                bottom: Math.round(c.bottom + i + o),
                right: Math.round(c.right + e + r)
            })
        }
        function p(t) {
            var e = t.getBoundingClientRect();
            return new n({
                top: Math.round(e.top),
                left: Math.round(e.left),
                bottom: Math.round(e.bottom),
                right: Math.round(e.right)
            })
        }
        function c() {
            var t = g.innerWidth || y.documentElement.clientWidth, e = g.innerHeight || y.documentElement.clientHeight, i = 0, a = 0;
            return y.body && (i = y.body.scrollLeft || y.documentElement.scrollLeft, a = y.body.scrollTop || y.documentElement.scrollTop), new n({
                top: Math.round(a),
                left: Math.round(i),
                width: Math.round(t),
                height: Math.round(e)
            })
        }
        function l() {
            var t = 0, e = 0;
            return y.body && (e = y.body.scrollWidth || y.documentElement.scrollWidth, t = y.body.scrollHeight || y.documentElement.scrollHeight), {
                width: Math.round(e),
                height: Math.round(t)
            }
        }
        function u() {
            var t = g.innerWidth || y.documentElement.clientWidth, e = g.innerHeight || y.documentElement.clientHeight;
            return {
                width: Math.round(t),
                height: Math.round(e)
            }
        }
        function d(t) {
            t = t ? t : g;
            var e = y.body.scrollLeft || y.documentElement.scrollLeft, i = y.body.scrollTop || y.documentElement.scrollTop;
            return {
                left: Math.round(e),
                top: Math.round(i)
            }
        }
        function f(t, e) {
            this.elem = t, this.trimmedRect = null, e && (this.trimmedRect = e)
        }
        var g = t.top, y = g.document;
        return n.prototype.toString = function() {
            return "Rect W:" + this.width + " H:" + this.height + " T:" + this.top + " L:" + this.left + " B:" + this.bottom + " R:" + this.right
        }, n.prototype.transfer = function(t, e) {
            this.left += t, this.right += t, this.top += e, this.bottom += e
        }, n.prototype.copy = function() {
            return new n(this)
        }, n.prototype.getOverlappedArea = function(t) {
            var e = Math.max(this.top, t.top), i = Math.min(this.bottom, t.bottom), a = Math.max(this.left, t.left), o = Math.min(this.right, t.right);
            return new n(i - e > 0 && o - a > 0 ? {
                top: e,
                bottom: i,
                left: a,
                right: o
            } : {
                top: 0,
                bottom: 0,
                left: 0,
                right: 0
            })
        }, f.prototype.calcLayoutPosition = function() {
            function t(t) {
                for (var i = t; ;) {
                    if (i = i.parentNode, i === i.ownerDocument.body)
                        break;
                    s(i) && (e.unshift(i), n += i.scrollTop, o += i.scrollLeft)
                }
            }
            var e = [], i = d(), n = i.top, o = i.left;
            t(this.elem);
            for (var r = a(this.elem); r != g;) {
                e.unshift(r.frameElement);
                var p = d(r);
                n += p.top, o += p.left, t(r.frameElement), r = r.parent
            }
            this.layoutPosition = h(this.elem, o, n), this.scrollParents = e
        }, f.prototype.getLayoutPosition = function(t) {
            return (t ||!this.layoutPosition) && this.calcLayoutPosition(), this.layoutPosition
        }, f.prototype.getViewableRect = function(t) {
            this.scrollParents || this.calcLayoutPosition();
            var e = t ? t: c(), n = i.map(this.scrollParents, function(t) {
                return t
            });
            n.push(this.elem);
            for (var a = null, o = e, r = 0; r < n.length; r++) {
                var s = n[r];
                if (a = h(s, e.left, e.top), o = o.getOverlappedArea(a), 0 === o.width || 0 === o.height)
                    break
            }
            if (this.trimmedRect) {
                var p = this.trimmedRect.copy();
                return p.transfer(a.left, a.top), o.getOverlappedArea(p)
            }
            return o
        }, {
            Rect: n,
            traverseAllWindows: o,
            getElementRect: p,
            getViewedRect: c,
            getPageSize: l,
            getScreenSize: u,
            getScrollPos: d,
            ElementPosition: f
        }
    }), i("core/event", ["dev/flags", "core/core_util", "core/dom_util"], function(e, i, n) {
        function a(t, e, i, n) {
            t.addEventListener ? t.addEventListener(e, i, n) : t.attachEvent && t.attachEvent("on" + e, i)
        }
        function o(t, e, i, n) {
            t.removeEventListener ? t.removeEventListener(e, i, n) : t.detachEvent && t.detachEvent("on" + e, i)
        }
        var r = t.top, h = function(t) {
            this.pollInterval = t, this.targets = [], this.timer = null, this.activeNums = 0
        };
        h.prototype.start = function() {
            var t = this;
            this.timer = r.setInterval(function() {
                t.poll()
            }, this.pollInterval)
        }, h.prototype.stop = function() {
            r.clearInterval(this.timer), this.timer = null
        }, h.prototype.poll = function() {
            var t = (new Date).getTime();
            i.forEach(this.targets, function(e) {
                e.active && t - e.lastFiredAt >= e.interval && (e.method.call(e.context, "Polling"), e.lastFiredAt = t, e.lazyMode && (e.lazyCounter++, e.lazyCounter > 30 && (e.interval = 1e3), e.lazyCounter > 60 && (e.interval = 3e3), e.lazyCounter > 100 && (e.interval = 1e4)))
            }, this)
        }, h.prototype.registerHandler = function(t, e, i, n) {
            var a = (new Date).getTime(), o = n ? 200: t, r = {
                active: !1,
                interval: o,
                context: e,
                method: i,
                lastFiredAt: a,
                lazyMode: !!n,
                lazyCounter: 0
            }, s = this.targets.length;
            return this.targets.push(r), s
        }, h.prototype.activateHandler = function(t) {
            var e = this.targets[t];
            e.active=!0, 0 === this.activeNums && this.start(), this.activeNums++
        }, h.prototype.deactivateHandler = function(t) {
            var e = this.targets[t];
            e.active=!1, this.activeNums--, 0 === this.activeNums && this.stop()
        };
        var p = function(t, e) {
            this.manager = t, this.context = e.context, this.method = e.method, this.name = e.name || this.method.name || "UNNAMED", this.events = e.events || [], this.polling = e.polling||-1, this.lazyPolling = e.lazyPolling, this.minInterval = e.minInterval || 0, this.lastFiredAt = 0, this.eventKeys = [], this.pollingKey =- 1, this.registerHandlers()
        };
        p.prototype.registerHandlers = function() {
            i.forEach(this.events, function(t) {
                var e = this.manager.getEvent(t), i = e.registerHandler(this, this.fire);
                this.eventKeys.push({
                    event: e,
                    key: i
                })
            }, this), (this.polling >= 0 || this.lazyPolling) && (this.pollingKey = this.manager.poller.registerHandler(this.polling, this, this.fire, this.lazyPolling)), this.activate()
        }, p.prototype.activate = function() {
            i.forEach(this.eventKeys, function(t) {
                t.event.activateHandler(t.key)
            }, this), this.pollingKey >= 0 && this.manager.poller.activateHandler(this.pollingKey)
        }, p.prototype.deactivate = function() {
            i.forEach(this.eventKeys, function(t) {
                t.event.deactivateHandler(t.key)
            }, this), this.pollingKey >= 0 && this.manager.poller.deactivateHandler(this.pollingKey)
        }, p.prototype.fire = function(t, e) {
            var i = (new Date).getTime(), n=!1;
            if (0 === this.minInterval || 0 === this.lastFiredAt ? n=!0 : i - this.lastFiredAt > this.minInterval && (n=!0), n) {
                var a = e ? e: {};
                this.lastFiredAt = i;
                var o = i - s.getMaster().baseTime;
                this.method.call(this.context, o, t, a)
            }
        };
        var c = function() {
            this.poller = new h(50), this.events = {}
        };
        c.prototype.newAction = function(t) {
            return new p(this, t)
        }, c.prototype.getEvent = function(t) {
            if (this.events[t])
                return this.events[t];
            if (!g[t])
                return null;
            var e = new g[t](this);
            return this.events[t] = e, e
        };
        var l = function() {
            this.eventName = "", this.handlers = [], this.activeNums = 0
        };
        l.prototype.registerHandler = function(t, e) {
            var i = {
                active: !1,
                context: t,
                method: e
            }, n = this.handlers.length;
            return this.handlers.push(i), n
        }, l.prototype.activateHandler = function(t) {
            var e = this.handlers[t];
            e.active=!0, 0 === this.activeNums && this.setup(), this.activeNums++
        }, l.prototype.deactivateHandler = function(t) {
            var e = this.handlers[t];
            e.active=!1, this.activeNums--, 0 === this.activeNums && this.cleanup()
        }, l.prototype.setup = function() {}, l.prototype.fire = function(t) {
            i.forEach(this.handlers, function(e) {
                e.active && e.method.call(e.context, this.eventName, t)
            }, this)
        }, l.prototype.cleanup = function() {};
        var u = function() {
            l.apply(this, arguments), this.eventName = "onLoad"
        };
        i.inherit(u, l), u.prototype.setup = function() {
            var t = this;
            this.boundHandler = function(e) {
                t.fire({
                    nativeEvent: e
                })
            }, a(r, "load", this.boundHandler, !1)
        }, u.prototype.cleanup = function() {
            o(r, "load", this.boundHandler, !1)
        };
        var d = function() {
            l.apply(this, arguments), this.eventName = "pageLeave"
        };
        i.inherit(d, l), d.prototype.setup = function() {
            var t = this;
            this.nativeWrapper = function(e) {
                t.fire({
                    nativeEvent: e
                })
            }, a(r, "pagehide", this.nativeWrapper, !1)
        }, d.prototype.cleanup = function() {
            o(r, "pagehide", this.nativeWrapper, !1)
        };
        var f = function() {
            l.apply(this, arguments), this.eventName = "viewPosChange"
        };
        i.inherit(f, l), f.prototype.setup = function() {
            var t = this;
            this.nativeWrapper = function(e) {
                t.nativeHandler({
                    nativeEvent: e
                })
            }, a(r, "scroll", this.nativeWrapper, !1), a(r, "touchmove", this.nativeWrapper, !1)
        }, f.prototype.nativeHandler = function(t) {
            var e = n.getViewedRect();
            t.rect = e, this.lastViewedRect ? e.width !== this.lastViewedRect.width || e.height !== this.lastViewedRect.height ? (this.fire(t), this.lastViewedRect = e) : (Math.abs(e.top - this.lastViewedRect.top) >= this.changeThreshold || Math.abs(e.left - this.lastViewedRect.left) >= this.changeThreshold) && (this.fire(t), this.lastViewedRect = e) : (this.fire(t), this.lastViewedRect = e)
        }, f.prototype.changeThreshold = 50, f.prototype.cleanup = function() {
            o(r, "scroll", this.nativeWrapper, !1), o(r, "touchmove", this.nativeWrapper, !1)
        };
        var g = {
            onLoad: u,
            pageLeave: d,
            viewPosChange: f
        };
        return {
            EventManager: c,
            MetaEvent: l,
            MetaEventDefinition: g
        }
    }), i("core/master", ["dev/flags", "core/core_util", "core/event"], function(t, e, i) {
        var n = function(t) {
            this.baseTime = t, this.detectors = [], this.incrementedKey = 0, this.eventManager = new i.EventManager
        };
        return n.prototype.getElapsedTime = function() {
            return (new Date).getTime() - this.baseTime
        }, n.prototype.getAutoIncrementedKey = function() {
            return ++this.incrementedKey
        }, {
            Master: n
        }
    }), i("core/beacon_sender", ["dev/flags", "core/core_util"], function(t, e) {
        var i = function(t) {
            if ("https:" === location.protocol) {
                if (!t.httpsUri)
                    return;
                this.uri = t.httpsUri
            } else 
                this.uri = t.uri;
            this.requestType = t.requestType, this.dryRun=!!t.dryRun
        };
        return i.prototype.MAX_LENGTH_FOR_EACH_REQUEST = 2e3, i.prototype.getQueryMaxLength = function() {
            return this.MAX_LENGTH_FOR_EACH_REQUEST - this.uri.length
        }, i.prototype.sendString = function(t) {
            this.doRequest(t)
        }, i.prototype.doRequest = function(i) {
            if (t.debug) {
                e.map(i.split("&"), function(t) {
                    var e = t.split("=");
                    return 1 === e.length ? decodeURIComponent(e[0]) : decodeURIComponent(e[0]) + ":" + decodeURIComponent(e[1])
                }).join(", ")
            }
            var n = this.getQueryMaxLength();
            if (i.length > n && (i = i.substring(0, n)), !this.dryRun) {
                var a = document.createElement("img"), o = this.uri;
                i.length > 0 && (o += "?" + i + "&EOQ", a.src = o)
            }
        }, {
            BeaconSender: i
        }
    }), i("core/beacon_query", ["dev/flags", "core/core_util"], function(t, e) {
        function i(t) {
            return t instanceof a ? t : e.isString(t) || e.isNumber(t) ? new o(t.toString()) : null
        }
        var n = function(t) {
            this.dataDef = t, this.lastElapsedTime =- 1, this.history = [], this.history.push([0, {}
            ]), this.dataSize = 0, this.separatorNums = {
                sep1: 0,
                sep2: 0,
                sep3: 0,
                sep4: 0
            }
        };
        n.prototype.newItem = function(t, e) {
            this.lastElapsedTime = t;
            var i = {}, n = 0;
            for (var a in e)
                if (e.hasOwnProperty(a)) {
                    var o = encodeURIComponent(e[a].toString());
                    this.hasDiff(a, o) && (i[a] = o, this.dataSize += a.length, this.dataSize += o.length, this.separatorNums.sep2++, n++, n > 1 && this.separatorNums.sep3++)
                }
            n > 0 && (this.history.push([t, i]), this.dataSize += t.toString().length, this.separatorNums.sep1++, this.history.length > 2 && this.separatorNums.sep4++)
        }, n.prototype.hasDiff = function(t, e) {
            for (var i = this.history.length - 1; i >= 0; i--) {
                var n = this.history[i][1];
                if (n.hasOwnProperty(t))
                    return n[t] === e.toString()?!1 : !0
            }
            return !0
        }, n.prototype.calcDataSize = function(t) {
            var e = {
                sep1: encodeURIComponent(t.sep1).length,
                sep2: encodeURIComponent(t.sep2).length,
                sep3: encodeURIComponent(t.sep3).length,
                sep4: encodeURIComponent(t.sep4).length
            };
            return this.dataSize + this.separatorNums.sep1 * e.sep1 + this.separatorNums.sep2 * e.sep2 + this.separatorNums.sep3 * e.sep3 + this.separatorNums.sep4 * e.sep4
        }, n.prototype.shrink = function() {
            for (var t = {}, i = this.history.length, n = (function(e, i) {
                e in t || (t[e] = i)
            }), a = i - 1; a >= 0; a--) {
                var o = this.history[a][1];
                e.forEachHash(o, n, this)
            }
            var r = this.history[i - 1][0];
            this.history = [], this.history.push([r, t]), this.dataSize = 0, this.separatorNums = {
                sep1: 0,
                sep2: 0,
                sep3: 0,
                sep4: 0
            }
        }, n.prototype.forEach = function(t, i) {
            if (!(this.history.length <= 1)) {
                var n = this.history.length, a = {}, o = function(t, e) {
                    a[t] = e
                };
                e.forEachHash(this.history[0][1], o);
                for (var r = 1; n > r; r++) {
                    var s = this.history[r][0], h = this.history[r][1];
                    e.forEachHash(h, o), t.call(i, s, a)
                }
            }
        }, n.prototype.serialize = function(t) {
            var i = "";
            return e.forEach(this.history, function(n, a) {
                i.length > 0 && (i += encodeURIComponent(t.sep4)), a > 0 && (i += n[0], i += encodeURIComponent(t.sep1), i += e.formatObject(n[1], encodeURIComponent(t.sep2), encodeURIComponent(t.sep3)))
            }), i
        };
        var a = function() {
            this.dataSize = 0, this.parent = null
        };
        a.prototype.incrementSize = function(t) {
            this.dataSize += t, this.parent && this.parent.incrementSize(t)
        }, a.prototype.decrementSize = function(t) {
            this.dataSize -= t, this.parent && this.parent.decrementSize(t)
        }, a.prototype.getDataSize = function() {
            return this.dataSize
        }, a.prototype.exports = function() {};
        var o = function(t, e) {
            a.apply(this, arguments), this.value = e ? t : encodeURIComponent(t), this.incrementSize(this.value.length)
        };
        e.inherit(o, a), o.prototype.exports = function() {
            return this.value
        };
        var r = function(t) {
            a.apply(this, arguments), this.sep = encodeURIComponent(t), this.children = []
        };
        e.inherit(r, a), r.prototype.exports = function() {
            return e.map(this.children, function(t) {
                return t.exports()
            }, this).join(this.sep)
        }, r.prototype.push = function(t) {
            var e = i(t);
            this.children.length > 0 && this.incrementSize(this.sep.length), this.incrementSize(e.dataSize), this.children.push(e), e.parent = this
        }, r.prototype.clear = function() {
            this.children = [], this.dataSize = 0
        };
        var s = function(t, e) {
            a.apply(this, arguments), this.keySep = encodeURIComponent(t), this.valSep = encodeURIComponent(e), this.hash = {}, this.keyNums = 0
        };
        e.inherit(s, a), s.prototype.exports = function() {
            var t = "";
            return e.forEachHash(this.hash, function(e, i) {
                t.length > 0 && (t += this.valSep), t += e + this.keySep + i.exports()
            }, this), t
        }, s.prototype.update = function(t, e) {
            var n = i(e), a = encodeURIComponent(t);
            this.hash[a] && this.remove(t), this.keyNums++, this.keyNums > 1 && this.incrementSize(this.valSep.length), this.incrementSize(a.length), this.incrementSize(this.keySep.length), this.incrementSize(n.dataSize), this.hash[a] = n, n.parent = this
        }, s.prototype.remove = function(t) {
            var e = encodeURIComponent(t), i = this.hash[e];
            i && (this.decrementSize(e.length), this.decrementSize(this.keySep.length), this.decrementSize(i.dataSize), this.keyNums -= 1, 0 !== this.keyNums && this.decrementSize(this.valSep.length), delete this.hash[e])
        }, s.prototype.get = function(t) {
            return this.hash[encodeURIComponent(t)]
        };
        var h = function(t, i) {
            this.options = i ? i : {}, this.definition = {}, this.exportOrder = [], this.staticValueNums = 0, this.staticValueSize = 0, e.forEach(t, function(t) {
                t._encodedKey = encodeURIComponent(t.key), "static" === t.type && (t._encodedValue = encodeURIComponent(t.value), this.staticValueSize += t._encodedKey.length, this.staticValueSize += 1, this.staticValueSize += t._encodedValue.length, this.staticValueNums > 0 && (this.staticValueSize += 1), this.staticValueNums++), this.definition[t.name] = t, this.exportOrder.push(t.name)
            }, this)
        };
        h.prototype.getInstance = function() {
            return new p(this)
        }, h.prototype.isDefinedVar = function(t) {
            return this.definition[t] && "var" === this.definition[t].type
        }, h.prototype.getStaticValueSize = function() {
            return this.staticValueSize
        }, h.prototype.setLazyStaticValue = function(t, e) {
            var i = this.definition[t];
            i._encodedValue && (this.staticValueSize -= i._encodedKey.length, this.staticValueSize -= 1, this.staticValueSize -= i._encodedValue.length, i._encodedValue = null, this.staticValueNums -= 1, 0 !== this.staticValueNums && (this.staticValueSize -= 1)), e && (i._encodedValue = encodeURIComponent(e), this.staticValueSize += i._encodedKey.length, this.staticValueSize += 1, this.staticValueSize += i._encodedValue.length, this.staticValueNums > 0 && (this.staticValueSize += 1), this.staticValueNums++)
        };
        var p = function(t) {
            this.template = t, this.vars = {}
        };
        return p.prototype.bind = function(t, e, n) {
            var a;
            a = n ? new o(e, !0) : i(e), this.template.isDefinedVar(t) && (this.vars[t] = a)
        }, p.prototype.exports = function() {
            var t = "";
            return e.forEach(this.template.exportOrder, function(i) {
                var n = this.template.definition[i];
                "static" === n.type && (t.length > 0 && (t += "&"), t += n._encodedKey + "=" + n._encodedValue), "lazy-static" === n.type ? n._encodedValue && (t.length > 0 && (t += "&"), t += n._encodedKey + "=" + n._encodedValue) : "random" === n.type ? (t.length > 0 && (t += "&"), t += n._encodedKey + "=" + e.makeRandomString()) : "var" === n.type && this.vars[i] && (t.length > 0 && (t += "&"), t += n._encodedKey + "=" + this.vars[i].exports())
            }, this), t
        }, {
            HistoricalData: n,
            StringBeaconData: o,
            ArrayBeaconData: r,
            HashBeaconData: s,
            BeaconQueryTemplate: h,
            BeaconQuery: p
        }
    }), i("core/uadf", ["dev/flags", "core/core_util", "core/dom_util", "core/master", "core/event", "core/beacon_sender", "core/beacon_query"], function(t, e, i, n, a, o, r) {
        function h() {
            return s.__masterInstance || (s.__masterInstance = new n.Master(f)), s.__masterInstance
        }
        function l(t) {
            for (var e = "", i = 0; i < t.length; i++) {
                var n = t[i];
                n.toLowerCase() !== n ? (e.length > 0 && (e += "_"), e += n.toLowerCase()) : e += n
            }
            return e
        }
        function u(t) {
            var i = function() {
                t.apply(this, arguments)
            };
            return e.inherit(i, t), i
        }
        function d(t) {
            var i = l(t), n = "detector/" + i, a = p(n);
            if (!a)
                return null;
            var o = {};
            return e.forEachHash(a, function(t, e) {
                o[t] = u(e)
            }), o
        }
        var f = (new Date).getTime();
        return c(s, {
            version: t.version,
            getMaster: h,
            exportDetector: d
        }), {
            CoreUtil: e,
            DOMUtil: i,
            Event: a,
            BeaconSender: o,
            BeaconQuery: r,
            getMaster: h
        }
    }), i("detector/detector", ["dev/flags", "core/uadf"], function(t, e) {
        var i = e.CoreUtil, n = function() {
            this.serialNumber = e.getMaster().getAutoIncrementedKey(), this.registerAllActions()
        };
        return n.prototype.actions = {}, n.prototype.registerAllActions = function() {
            i.forEachHash(this.actions, function(t, e) {
                var i = this.registerAction(t, e);
                this.actions[t]._registered = i
            }, this)
        }, n.prototype.deactivateAllActions = function() {
            i.forEachHash(this.actions, function(t, e) {
                e._registered.deactivate()
            }, this)
        }, n.prototype.registerAction = function(t, i) {
            var n = {
                context: this,
                method: this[t],
                name: t,
                events: i.events,
                polling: i.polling,
                lazyPolling: i.lazyPolling,
                minInterval: i.minInterval
            }, a = e.getMaster().eventManager, o = a.newAction(n);
            return o
        }, n.prototype.isCollectedElem = function(t) {
            if (t.__uadfCollectedBy)
                for (var e = t.__uadfCollectedBy, i = 0; i < e.length; i++)
                    if (e[i] === this.serialNumber)
                        return !0;
            return !1
        }, n.prototype.setAsCollected = function(t) {
            t.__uadfCollectedBy || (t.__uadfCollectedBy = []), t.__uadfCollectedBy.push(this.serialNumber)
        }, {
            Detector: n
        }
    }), i("detector/simple_viewable_detector", ["dev/flags", "core/uadf", "detector/detector"], function(t, e, i) {
        var n = e.CoreUtil, a = e.DOMUtil, o = i.Detector, r = e.BeaconSender, s = e.BeaconQuery.BeaconQueryTemplate, h = e.BeaconQuery.ArrayBeaconData, p = function() {
            this.data = new h(this.separator)
        };
        p.prototype.separator = "!", p.prototype.push = function(t) {
            this.data.push(t)
        };
        var c = function(t, e) {
            this.rawElement = t, this.trimmedRect = e && e.trimmedRect ? e.trimmedRect : null, this.position = new a.ElementPosition(this.rawElement, this.trimmedRect), this.layoutPosition = this.position.getLayoutPosition(), this.lastViewedIn=!1, this.lastViewedStart = 0, this.currentViewedTime = 0, this.viewedEnoughTime=!1, this.notifiedOnce=!1, this.viewInTimes = [], this.viewOutTimes = [], this.options = e
        };
        c.prototype.VIEWED_TIME_CONDITION = 1e3, c.prototype.VIEWED_AREA_CONDITION = .5, c.prototype.calcViewableArea = function(t, e) {
            var i = this.position.getViewableRect(), n = i.width * i.height;
            if (0 === n)
                return 0;
            var o = this.trimmedRect ? this.trimmedRect: a.getElementRect(this.rawElement), r = o.width * o.height, s = t.width * t.height;
            return 0 === r || 0 === s ? 0 : n / Math.min(r, s)
        }, c.prototype.check = function(t, e, i) {
            var n = this.calcViewableArea(e, i), a = n >= this.VIEWED_AREA_CONDITION;
            this.lastViewedIn && a ? this.handleViewContinue(t) : this.lastViewedIn&&!a ? this.handleViewOut(t) : !this.lastViewedIn && a && this.handleViewIn(t)
        }, c.prototype.handleViewIn = function(t) {
            this.lastViewedIn=!0, this.lastViewedStart = t, this.currentViewedTime = 0, this.viewInTimes.push(this.lastViewedStart)
        }, c.prototype.handleViewContinue = function() {
            this.currentViewedTime = e.getMaster().getElapsedTime() - this.lastViewedStart, this.currentViewedTime >= this.VIEWED_TIME_CONDITION && (this.viewedEnoughTime=!0)
        }, c.prototype.handleViewOut = function(t) {
            this.currentViewedTime = t - this.lastViewedStart, this.currentViewedTime >= this.VIEWED_TIME_CONDITION && (this.viewedEnoughTime=!0), this.viewOutTimes.push(t), this.lastViewedIn=!1, this.lastViewedStart = 0, this.currentViewedTime = 0
        };
        var l = function() {
            o.apply(this, arguments), this.targets = [], this.beaconSender = new this.BeaconSenderConstructor(this.beaconConfig), this.viewedQueryTmpl = new this.BeaconQueryTemplateConstructor(this.viewedQueryTmplDef)
        };
        return n.inherit(l, o), l.prototype.BeaconSenderConstructor = r.BeaconSender, l.prototype.BeaconQueryTemplateConstructor = s, l.prototype.BeaconDataListConstructor = p, l.prototype.ViewableTargetConstructor = c, l.prototype.actions = {
            collectTargets: {
                events: ["onLoad"],
                polling: 5e3
            },
            checkViewableState: {
                events: ["onLoad", "viewPosChange"],
                polling: 300,
                minInterval: 300
            },
            viewedBeacon: {
                events: ["pageLeave"],
                lazyPolling: !0
            }
        }, l.prototype.collectTargetElements = function() {
            return []
        }, l.prototype.collectTargets = function() {
            var t = this.collectTargetElements();
            t && n.forEach(t, function(t) {
                this.registerTarget(t)
            }, this)
        }, l.prototype.registerTarget = function(t, e) {
            if (!this.isCollectedElem(t) || e && e.canDuplicate) {
                var i = new this.ViewableTargetConstructor(t, e);
                return this.targets.push(i), this.setAsCollected(t), i
            }
            return null
        }, l.prototype.checkViewableState = function(t, e, i) {
            var o = i.rect ? i.rect: a.getViewedRect(), r=!1;
            n.forEach(this.targets, function(e) {
                e.check(t, o), this.isBeaconedTarget(e) && (r=!0)
            }, this), this.checkViewableState_InstantMode && this.viewedBeacon(t, e, i)
        }, l.prototype.checkViewableStateStart = function(t, e, i) {
            return this.checkViewableStateInProgress?!0 : (this.checkViewableStateIndex = 0, this.checkViewableStateInProgress=!0, void this.checkViewableStateStep(t, e, i))
        }, l.prototype.checkViewableStateStep = function(t, e, i) {
            if (this.checkViewableStateInProgress) {
                for (var n = i.rect ? i.rect : a.getViewedRect(), o = this.checkViewableStateIndex + 200, r=!1; this.checkViewableStateIndex < o;) {
                    var s = this.targets[this.checkViewableStateIndex];
                    if (!s) {
                        r=!0;
                        break
                    }
                    s.check(t, n), this.checkViewableStateIndex++
                }
                r && (this.checkViewableStateIndex = 0, this.checkViewableStateInProgress=!1), this.checkViewableState_InstantMode && this.viewedBeacon(t, e, i)
            }
        }, l.prototype.checkViewableState_InstantMode=!1, l.prototype.isBeaconedTarget = function(t) {
            return t.viewedEnoughTime && (!this.sendBeaconOnlyOnce ||!t.notifiedOnce)
        }, l.prototype.getBeaconDataForTarget = function(t) {
            return this.getBeaconDataForTargetElement(t.rawElement)
        }, l.prototype.getBeaconDataForTargetElement = function() {
            return ""
        }, l.prototype.viewedBeacon = function(t, e, i) {
            var a = n.filter(this.targets, function(t) {
                return this.isBeaconedTarget(t)
            }, this), o = n.map(a, function(t) {
                return t.notifiedOnce=!0, this.getBeaconDataForTarget(t)
            }, this);
            if (o.length > 0) {
                var r = new this.BeaconDataListConstructor;
                n.forEach(o, function(t) {
                    r.push(t)
                }, this);
                var s = this.viewedQueryTmpl.getInstance();
                s.bind(this.viewedBeaconDataKey, r.data), this.viewedBeaconModifyQuery(t, e, i, s), this.beaconSender.sendString(s.exports())
            }
        }, l.prototype.viewedBeaconModifyQuery = function() {}, l.prototype.viewedBeaconDataKey = "viewed", l.prototype.sendBeaconOnlyOnce=!0, l.prototype.beaconConfig = {
            uri: "http://" + location.host + "/beacon.gif",
            httpsUri: "https://s." + location.host + "/beacon.gif",
            requestType: "image"
        }, l.prototype.viewedQueryTmplDef = [{
            name: "viewed",
            type: "var",
            key: "viewed"
        }
        ], {
            SimpleBeaconDataList: p,
            SimpleViewableTarget: c,
            SimpleViewableDetector: l
        }
    }), i("detector/deep_viewable_detector", ["dev/flags", "core/uadf", "detector/simple_viewable_detector"], function(t, e, i) {
        var n = e.CoreUtil, a = e.DOMUtil, o = (e.BeaconQuery.BeaconQueryTemplate, e.BeaconQuery.StringBeaconData, e.BeaconQuery.ArrayBeaconData), r = e.BeaconQuery.HashBeaconData, s = e.BeaconQuery.HistoricalData, h = i.SimpleViewableDetector, p = i.SimpleViewableTarget, c = function() {
            this.data = new r(this.separators.rootKeySep, this.separators.rootValSep), this.hasDiff=!1
        };
        c.prototype.separators = {
            rootKeySep: ":",
            rootValSep: ",",
            inOutsSep: ""
        }, c.prototype.keyNameMap = {
            id: "r",
            width: "w",
            height: "h",
            posX: "px",
            posY: "py",
            inOuts: "io"
        }, c.prototype.updateStringValue = function(t, e) {
            var i=!1, n = this.data.get(this.keyNameMap[t]), a = this.formatter[t] ? this.formatter[t].apply(this, e) : e[0];
            n ? a !== n.value && (i=!0) : i=!0, i && (this.data.update(this.keyNameMap[t], a), this.hasDiff=!0)
        }, c.prototype.updateId = function(t) {
            this.data.update(this.keyNameMap.id, t)
        }, c.prototype.updateWidth = function(t) {
            this.updateStringValue("width", [t])
        }, c.prototype.updateHeight = function(t) {
            this.updateStringValue("height", [t])
        }, c.prototype.updatePosX = function(t) {
            this.updateStringValue("posX", [t])
        }, c.prototype.updatePosY = function(t) {
            this.updateStringValue("posY", [t])
        }, c.prototype.formatter = {}, c.prototype.getInOutsArray = function() {
            var t = this.data.get(this.keyNameMap.inOuts);
            return t || (t = new o(this.separators.inOutsSep), this.data.update(this.keyNameMap.inOuts, t)), t
        }, c.prototype.pushInOuts = function(t, e) {
            var i = this.getInOutsArray();
            "in" === t ? (i.push("i" + e), this.hasDiff=!0) : "out" === t && (i.push("o" + e), this.hasDiff=!0)
        }, c.prototype.clearAfterSent = function() {
            this.data.remove(this.keyNameMap.width), this.data.remove(this.keyNameMap.height), this.data.remove(this.keyNameMap.posX), this.data.remove(this.keyNameMap.posY), this.data.remove(this.keyNameMap.inOuts), this.hasDiff=!1
        }, c.prototype.getDataSize = function() {
            return this.data.getDataSize()
        };
        var l = function() {
            p.apply(this, arguments), this.beaconData = new this.BeaconDataConstructor, this.beaconData.updateId(this.getId()), this.beaconData.updateWidth(this.layoutPosition.width), this.beaconData.updateHeight(this.layoutPosition.height), this.beaconData.updatePosX(this.layoutPosition.left), this.beaconData.updatePosY(this.layoutPosition.top)
        };
        n.inherit(l, p), l.prototype.BeaconDataConstructor = c, l.prototype.getId = function() {
            return Math.floor(1e3 * Math.random()).toString()
        }, l.prototype.getGroupingId = function() {
            return "g"
        }, l.prototype.handleViewIn = function() {
            l.super_.prototype.handleViewIn.apply(this, arguments), this.beaconData.pushInOuts("in", this.viewInTimes[this.viewInTimes.length - 1])
        }, l.prototype.handleViewOut = function() {
            l.super_.prototype.handleViewOut.apply(this, arguments), this.beaconData.pushInOuts("out", this.viewOutTimes[this.viewOutTimes.length - 1])
        };
        var u = function() {
            this.data = new r(this.separators.rootKeySep, this.separators.rootValSep)
        };
        u.prototype.separators = {
            rootKeySep: "_",
            rootValSep: "!,",
            arraySep: "|"
        }, u.prototype.addItem = function(t, e) {
            var i = this.data.get(t);
            i || (i = new o(this.separators.arraySep), this.data.update(t, i)), i.push(e)
        }, u.prototype.exports = function() {
            return this.data.exports()
        };
        var d = function() {
            h.apply(this, arguments), this.stateQueryTmpl = new this.BeaconQueryTemplateConstructor(this.stateQueryTmplDef), this.pageState = new s(this.convertPageStateKey()), this.updatePageState(e.getMaster().getElapsedTime())
        };
        return n.inherit(d, h), d.prototype.ViewableTargetConstructor = l, d.prototype.TargetDataSetConstructor = u, d.prototype.actions || (d.prototype.actions = {}), d.prototype.actions.updatePageState = {
            events: ["viewPosChange"]
        }, d.prototype.actions.stateBeaconThresholdMode = {
            polling: 2e3
        }, d.prototype.actions.stateBeaconForceMode = {
            events: ["pageLeave"],
            polling: 2e4
        }, t.debug && (d.prototype.actions.dumpStateData = {
            polling: 1e4
        }), d.prototype.scrollMainElement = null, d.prototype.getCurrentPageState = function(t, e, i) {
            var n, a;
            return this.scrollMainElement ? (n = this.scrollMainElement.scrollLeft, a = this.scrollMainElement.scrollTop) : (n = e.left, a = e.top), {
                screenWidth: e.width,
                screenHeight: e.height,
                scrollX: n,
                scrollY: a,
                pageWidth: i.width,
                pageHeight: i.height
            }
        }, d.prototype.updatePageState = function(t, e, i) {
            var n = i && i.rect ? i.rect: a.getViewedRect(), o = a.getPageSize(), r = this.getCurrentPageState(t, n, o);
            this.pageState.newItem(t, this.convertPageStateKey(r))
        }, d.prototype.convertPageStateKey = function(t) {
            var e = {};
            return n.forEachHash(t, function(t, i) {
                e[this.pageStateKey[t]] = i
            }, this), e
        }, d.prototype.pageStateKey = {
            screenWidth: "bw",
            screenHeight: "bh",
            pageWidth: "pw",
            pageHeight: "ph",
            scrollX: "sx",
            scrollY: "sy"
        }, d.prototype.pageStateSeparators = {
            sep1: ":",
            sep2: "",
            sep3: ",",
            sep4: ";"
        }, d.prototype.stateBeacon = function(t, e, i, a) {
            var o = [], r = 0;
            n.forEach(this.targets, function(t) {
                var e = t.beaconData;
                e.hasDiff && (r += e.getDataSize(), o.push(t))
            }, this);
            var s = this.pageState.calcDataSize(this.pageStateSeparators);
            r += s;
            var h = a;
            if (a || (t <= this.stateBeaconEarlyTimeThreshold && r > 0 ? h=!0 : r > this.stateBeaconDataSizeThreshold && (h=!0)), h) {
                var p = this.stateQueryTmpl.getInstance();
                if (o.length > 0) {
                    var c = new this.TargetDataSetConstructor;
                    n.forEach(o, function(t) {
                        var e = t.beaconData, i = t.getGroupingId();
                        c.addItem(i, e.data)
                    }, this), p.bind(this.stateBeaconDataKey, c.data)
                }
                s > 0 && (p.bind(this.stateBeaconPageDataKey, this.pageState.serialize(this.pageStateSeparators), !0), this.pageState.shrink()), this.stateBeaconModifyQuery(t, e, i, p), this.beaconSender.sendString(p.exports()), n.forEach(o, function(t) {
                    var e = t.beaconData;
                    e.clearAfterSent()
                }, this)
            }
        }, d.prototype.stateBeaconForceMode = function(t, e, i) {
            this.stateBeacon(t, e, i, !0)
        }, d.prototype.stateBeaconThresholdMode = function(t, e, i) {
            this.stateBeacon(t, e, i, !1)
        }, d.prototype.stateBeaconModifyQuery = function() {}, d.prototype.stateBeaconDataKey = "state", d.prototype.stateBeaconPageDataKey = "pageState", d.prototype.stateBeaconDataSizeThreshold = 500, d.prototype.stateBeaconEarlyTimeThreshold = 1e4, d.prototype.stateQueryTmplDef = [{
            name: "state",
            type: "var",
            key: "state"
        }, {
            name: "pageState",
            type: "var",
            key: "pageState"
        }
        ], d.prototype.dumpStateData = function() {
            var t = {};
            n.forEach(this.targets, function(e) {
                var i = e.beaconData;
                if (i.hasDiff) {
                    var n = e.getGroupingId();
                    t[n] || (t[n] = []), t[n].push(i.data)
                }
            }, this);
            var e = "AD:" + n.formatObject(t, "$", "!", null, function(t) {
                return n.map(t, function(t) {
                    return decodeURIComponent(t.exports())
                }).join("|")
            });
            e += "PD:" + decodeURIComponent(this.pageState.serialize(this.pageStateSeparators))
        }, {
            DeepViewableDetector: d,
            DeepViewableTarget: l,
            DeepTargetData: c,
            DeepTargetDataSet: u
        }
    }), i("porting/yads_vimps", ["dev/flags", "core/uadf", "detector/deep_viewable_detector", "detector/simple_viewable_detector"], function(e, i, n, a) {
        var o = i.CoreUtil, r = i.DOMUtil, s = i.BeaconSender.BeaconSender, h = n.DeepViewableDetector, p = n.DeepViewableTarget, c = n.DeepTargetData, l = n.DeepTargetDataSet, u = a.SimpleBeaconDataList, d = t.top, f = d.document, g = function() {
            u.apply(this, arguments)
        };
        o.inherit(g, u), g.prototype.getDataSize = function() {
            return this.data.getDataSize()
        }, g.prototype.separator = "&";
        var y = function() {
            c.apply(this, arguments)
        };
        o.inherit(y, c), y.prototype.separators = {
            rootKeySep: "",
            rootValSep: "!",
            inOutsSep: ""
        }, y.prototype.keyNameMap = {
            id: "rk",
            width: "sw",
            height: "sh",
            posX: "px",
            posY: "py",
            inOuts: "io"
        };
        var m = function() {
            l.apply(this, arguments)
        };
        o.inherit(m, l), m.prototype.separators = {
            rootKeySep: ":",
            rootValSep: ";",
            arraySep: "'"
        };
        var v = function(t, e) {
            this.tagId = e.tagId, this.requestId = e.requestId, this.rank = e.rank, this.viewableInfo = e.viewableInfo, p.apply(this, arguments)
        };
        o.inherit(v, p), v.prototype.BeaconDataConstructor = y, v.prototype.getId = function() {
            return this.rank
        }, v.prototype.getGroupingId = function() {
            return this.tagId + "@" + this.requestId
        };
        var S = function() {
            s.apply(this, arguments), this.beaconAllowed=!1, this.beaconStack = []
        };
        o.inherit(S, s), S.prototype.sendString = function(t, e) {
            e || this.beaconAllowed ? S.super_.prototype.sendString.call(this, t) : this.beaconStack.push(t)
        }, S.prototype.setAllowed = function() {
            this.beaconAllowed=!0, this.sendIndex = 0;
            var t = this;
            this.timer = setInterval(function() {
                t.sendStackedBeacon()
            }, 10)
        }, S.prototype.sendStackedBeacon = function() {
            this.beaconStack[this.sendIndex] ? (S.super_.prototype.sendString.call(this, this.beaconStack[this.sendIndex]), this.sendIndex++) : (this.beaconStack = [], clearInterval(this.timer))
        };
        var w = function() {
            this.httpBeaconEntry && (this.beaconConfig = {
                uri: this.httpBeaconEntry,
                requestType: "image"
            }, this.beaconConfigForReport = {
                uri: this.httpBeaconEntryForReport,
                requestType: "image"
            }, h.apply(this, arguments), this.requestIdTable = {}, this.beaconNum = 0, this.disallowed=!1, this.beaconSenderForReport = new this.BeaconSenderConstructor(this.beaconConfigForReport), this.pvRandom = o.makeRandomString(), this.viewedQueryTmpl.setLazyStaticValue("pvRandom", this.pvRandom), this.stateQueryTmpl.setLazyStaticValue("pvRandom", this.pvRandom), this.messageQueryTmpl = new this.BeaconQueryTemplateConstructor(this.messageQueryTmplDef), this.setMessageHandler(), this.sendStartBeacon())
        };
        o.inherit(w, h), w.prototype.ViewableTargetConstructor = v, w.prototype.TargetDataSetConstructor = m, w.prototype.BeaconSenderConstructor = S, w.prototype.BeaconDataListConstructor = g, w.prototype.actions = {
            updatePageState: {
                events: ["viewPosChange"],
                minInterval: 100
            },
            checkViewableState: {
                events: ["onLoad", "viewPosChange"],
                polling: 1e3,
                minInterval: 100
            },
            viewedBeacon: {
                events: ["pageLeave"]
            },
            stateBeaconThresholdMode: {
                polling: 4e3
            },
            stateBeaconForceMode: {
                events: ["pageLeave"],
                polling: 6e4
            }
        }, e.debug && (w.prototype.actions.dumpStateData = {
            polling: 1e4
        }), w.prototype.checkViewableState_InstantMode=!0, w.prototype.stateBeaconDataSizeThreshold = 800, w.prototype.stateBeaconEarlyTimeThreshold = 2e4, w.prototype.sendStartBeacon = function() {
            var t = this.viewedQueryTmpl.getInstance();
            t.bind("beaconType", "1"), t.bind("seqNum", ++this.beaconNum), t.bind("elapsedTime", i.getMaster().getElapsedTime()), t.bind("pageUrl", location.href), t.bind("referrerUrl", f.referrer), t.bind("startTime", i.getMaster().baseTime), this.beaconSender.sendString(t.exports())
        }, w.prototype.handleVImpsMode = function(t) {
            return this.disallowed?!1 : 0 === t ? (this.terminate(), !1) : (this.beaconSender.setAllowed(), this.beaconSenderForReport.setAllowed(), !0)
        }, w.prototype.terminate = function() {
            this.disallowed=!0, this.deactivateAllActions(), this.sendMessageBeacon("101")
        }, w.prototype.notifyRendered = function(t) {
            if (!o.isNumber(t.vimpsMode)&&!isNaN(t.vimpsMode))
                return this.sendMessageBeacon("201"), void this.terminate();
            var e = this.handleVImpsMode(t.vimpsMode);
            if (e)
                if (o.isString(t.adProdSetCode) && 0 !== t.adProdSetCode.length && o.isString(t.requestId) && 0 !== t.requestId.length || this.sendMessageBeacon("202"), "multi" === t.type) {
                    var i = t.parentIdList;
                    o.isArray(i) || this.sendMessageBeacon("202"), o.forEach(i, function(e, i) {
                        var n = f.getElementById(e);
                        n && this.collectTextAdsIn(n, t.adProdSetCode, t.requestId, t.viewableInfoList[i])
                    }, this)
                } else if ("single_with_parent" === t.type) {
                    o.isString(t.parentId) && 0 !== t.parentId.length && o.isArray(t.viewableInfo) || this.sendMessageBeacon("202");
                    var n = t.parentId, a = f.getElementById(n);
                    a && this.collectTextAdsIn(a, t.adProdSetCode, t.requestId, t.viewableInfo)
                } else 
                    "doc_write" === t.type ? t.viewableInfo ? (t.docWriteCaller && "script" === t.docWriteCaller.tagName.toLowerCase() && o.isArray(t.viewableInfo) || this.sendMessageBeacon("202"), this.collectTextAdsIn(t.docWriteCaller.parentNode, t.adProdSetCode, t.requestId, t.viewableInfo)) : this.sendMessageBeacon("102", t.adProdSetCode + "@" + t.requestId) : this.sendMessageBeacon("202")
        }, w.prototype.collectTextAdsIn = function(t, i, n, a) {
            var r = 0;
            e.debug && (r = this.targets.length);
            var s = 0, h = t.getElementsByTagName("a");
            if (o.forEach(h, function(t) {
                if (0 === t.href.indexOf("http://rd.ane.yahoo.co.jp/rd?")) {
                    n in this.requestIdTable || (this.requestIdTable[n] = 0);
                    var e=++this.requestIdTable[n];
                    this.registerTarget(t, {
                        tagId: i,
                        requestId: n,
                        rank: e,
                        viewableInfo: a[s]
                    }), s++
                }
            }, this), e.debug) {
                this.targets.length - r
            }
        }, w.prototype.messageHandlers = {
            frameStarted: {
                method: "handleFrameStartedMessage"
            },
            frameLoaded: {
                method: "handleFrameLoadedMessage"
            }
        }, w.prototype.setMessageHandler = function() {
            var t = this, e = function(e) {
                t.handleMessage(e)
            };
            d.addEventListener ? d.addEventListener("message", e, !1) : d.attachEvent("onmessage", e)
        }, w.prototype.handleMessage = function(t) {
            var e = /\.yahoo\.co\.jp(:[0-9]+)?$/;
            if ("http://i.yimg.jp" === t.origin || "https://s.yimg.jp" === t.origin || e.test(t.origin)) {
                var i = t.data.split(","), n = i[0], a = this.messageHandlers[n];
                a && this[a.method].call(this, i, t)
            }
        }, w.prototype.handleFrameStartedMessage = function(t) {
            if (3 !== t.length)
                return void this.sendMessageBeacon("203");
            var e = parseInt(t[1], 10);
            if (isNaN(e))
                return this.sendMessageBeacon("201"), void this.terminate();
            var i = this.handleVImpsMode(e);
            if (i) {
                t[2]
            }
        }, w.prototype.handleFrameLoadedMessage = function(t, e) {
            5 !== t.length && this.sendMessageBeacon("203");
            var i = parseInt(t[1], 10);
            if (isNaN(i))
                return this.sendMessageBeacon("201"), void this.terminate();
            var n = this.handleVImpsMode(i);
            if (n) {
                var a;
                if (r.traverseAllWindows(function(t) {
                    for (var i = t.document.getElementsByTagName("iframe"), n = 0; n < i.length; n++)
                        if (i[n].contentWindow == e.source)
                            return a = i[n], !1;
                    return !0
                }, d), !a)
                    return void this.sendMessageBeacon("203");
                var s = t[2], h = t[3];
                if ((0 === s.length || 0 === h.length) && this.sendMessageBeacon("203"), "non_ydn" === h)
                    return void this.registerTarget(a, {
                    tagId: s,
                    requestId: h,
                    rank: "1",
                    viewableInfo: "non_ydn"
                });
                var p = function(t) {
                    var e = t.split("/");
                    if (5 !== e.length)
                        return null;
                    var i = parseInt(e[0], 10), n = parseInt(e[1], 10), a = parseInt(e[2], 10), o = parseInt(e[3], 10);
                    return isNaN(i) || isNaN(n) || isNaN(a) || isNaN(o) ? null : e[4] && 0 !== e[4].length ? 0 === i && 0 === n && 0 === a && 0 === o ? [e[4], null] : [e[4], new r.Rect({
                        top: i,
                        right: n,
                        bottom: a,
                        left: o
                    })] : null
                };
                if (0 === t[4].indexOf("ERR"))
                    return void this.sendMessageBeacon("203", t[4]);
                var c = t[4].split("|");
                o.forEach(c, function(t, e) {
                    var i = p(t);
                    if (!i)
                        return void this.sendMessageBeacon("203", t);
                    var n = i[0], o = i[1], r = {
                        canDuplicate: !0,
                        tagId: s,
                        requestId: h,
                        rank: e + 1,
                        viewableInfo: n
                    };
                    o && (r.trimmedRect = o), this.registerTarget(a, r)
                }, this)
            }
        }, w.prototype.getBeaconDataForTarget = function(t) {
            return t.viewableInfo
        }, w.prototype.viewedBeacon = function(t, e, i) {
            var n = o.filter(this.targets, function(t) {
                return this.isBeaconedTarget(t)
            }, this), a = o.map(n, function(t) {
                return t.notifiedOnce=!0, this.getBeaconDataForTarget(t)
            }, this);
            if (a.length > 0) {
                var r = new this.BeaconDataListConstructor;
                if (o.forEach(a, function(n) {
                    if ("non_ydn" !== n && (r.push(n), r.getDataSize() > 1500)) {
                        var a = this.viewedQueryTmpl.getInstance();
                        a.bind(this.viewedBeaconDataKey, r.data), this.viewedBeaconModifyQuery(t, e, i, a), this.beaconSenderForReport.sendString(a.exports()), r = new this.BeaconDataListConstructor
                    }
                }, this), r.getDataSize() > 0) {
                    var s = this.viewedQueryTmpl.getInstance();
                    s.bind(this.viewedBeaconDataKey, r.data), this.viewedBeaconModifyQuery(t, e, i, s), this.beaconSenderForReport.sendString(s.exports())
                }
            }
        }, w.prototype.viewedBeaconModifyQuery = function(t, e, i, n) {
            n.bind("beaconType", "2"), n.bind("seqNum", ++this.beaconNum), n.bind("elapsedTime", t), "pageLeave" === e && n.bind("pageLeaveFlag", "1")
        }, w.prototype.stateBeaconModifyQuery = function(t, e, i, n) {
            n.bind("beaconType", "3"), n.bind("seqNum", ++this.beaconNum), n.bind("elapsedTime", t), "pageLeave" === e && n.bind("pageLeaveFlag", "1")
        }, w.prototype.getCurrentPageState = function() {
            var t = w.super_.prototype.getCurrentPageState.apply(this, arguments);
            return d.yads_pv_timestamp && (t.yadsPVTS = d.yads_pv_timestamp), t
        }, w.prototype.sendMessageBeacon = function(t, n) {
            var a = this.messageQueryTmpl.getInstance();
            a.bind("beaconType", "4"), a.bind("seqNum", ++this.beaconNum), a.bind("elapsedTime", i.getMaster().getElapsedTime()), a.bind("messageCode", t), n && a.bind("optionMessage", n), this.beaconSender.sendString(a.exports()), e.debug && "1" !== t[0]
        }, w.prototype.pageStateKey = {
            screenWidth: "bw",
            screenHeight: "bh",
            pageWidth: "pw",
            pageHeight: "ph",
            scrollX: "sx",
            scrollY: "sy",
            yadsPVTS: "ts"
        }, w.prototype.pageStateSeparators = {
            sep1: "|",
            sep2: "",
            sep3: "!",
            sep4: "'"
        }, w.prototype.httpBeaconEntry = null;
        var b = [{
            name: "rs",
            type: "static",
            key: "rs",
            value: "yads_vb"
        }, {
            name: "beaconType",
            type: "var",
            key: "bt"
        }, {
            name: "version",
            type: "static",
            key: "v",
            value: e.version
        }, {
            name: "pvRandom",
            type: "lazy-static",
            key: "pv"
        }, {
            name: "beaconRandom",
            type: "random",
            key: "rand"
        }, {
            name: "seqNum",
            type: "var",
            key: "seq"
        }, {
            name: "elapsedTime",
            type: "var",
            key: "et"
        }, {
            name: "retryNum",
            type: "var",
            key: "rn"
        }, {
            name: "pageLeaveFlag",
            type: "var",
            key: "pl"
        }, {
            name: "pageUrl",
            type: "var",
            key: "u"
        }, {
            name: "referrerUrl",
            type: "var",
            key: "ref"
        }, {
            name: "startTime",
            type: "var",
            key: "st"
        }, {
            name: "reportData",
            type: "var",
            key: "vi"
        }, {
            name: "pageStateData",
            type: "var",
            key: "pd"
        }, {
            name: "adStateData",
            type: "var",
            key: "ad"
        }, {
            name: "messageCode",
            type: "var",
            key: "code"
        }, {
            name: "optionMessage",
            type: "var",
            key: "msg"
        }
        ];
        return w.prototype.viewedQueryTmplDef = b, w.prototype.stateQueryTmplDef = b, w.prototype.messageQueryTmplDef = b, w.prototype.viewedBeaconDataKey = "reportData", w.prototype.stateBeaconDataKey = "adStateData", w.prototype.stateBeaconPageDataKey = "pageStateData", {
            YADSViewableDetector: w
        }
    }), i("porting/yads_vimps_main", ["dev/flags", "core/uadf", "porting/yads_vimps"], function(t, e, i) {
        var n = i.YADSViewableDetector, a = function() {
            n.apply(this, arguments)
        };
        e.CoreUtil.inherit(a, n);
        var o = null, r = null;
        o = "http://b94.yahoo.co.jp/gb.gif", r = "http://b94.yahoo.co.jp/v", a.prototype.httpBeaconEntry = o, a.prototype.httpBeaconEntryForReport = r;
        var h = new a;
        c(s, {
            YADSViewable: {
                notifyRendered: function() {
                    h.notifyRendered.apply(h, arguments)
                }
            }
        })
    }), p("porting/yads_vimps_main")
}(this);
