function getViewPort() {
    var e, t;
    return typeof window.innerWidth == "number" ? (e = window.innerWidth, t = window.innerHeight) : document.documentElement && (document.documentElement.clientWidth || document.documentElement.clientHeight) ? (e = document.documentElement.clientWidth, t = document.documentElement.clientHeight) : document.body && (document.body.clientWidth || document.body.clientHeight) && (e = document.body.clientWidth, t = document.body.clientHeight), {
        width: e,
        height: t
    }
}
function getTagInfo() {
    window.groupId = window.groupId == undefined ? Math.round(Math.random() * 1e3) : window.groupId;
    var e = getViewPort(), t = "";
    e.width <= 768 ? t = "mobile" : e.width > 768 && e.width <= 1024 ? t = "tablet" : t = "display";
    var n = t == "mobile" ? "a": t == "tablet" ? "a": "adserver";
    return {
        domain: n,
        alias: t,
        groupId: groupId,
        viewPort: e
    }
}(function() {
    function C(e, t, n) {
        if (e === t)
            return e !== 0 || 1 / e == 1 / t;
        if (e == null || t == null)
            return e === t;
        e._chain && (e = e._wrapped), t._chain && (t = t._wrapped);
        if (e.isEqual && S.isFunction(e.isEqual))
            return e.isEqual(t);
        if (t.isEqual && S.isFunction(t.isEqual))
            return t.isEqual(e);
        var r = a.call(e);
        if (r != a.call(t))
            return !1;
        switch (r) {
        case"[object String]":
            return e == String(t);
        case"[object Number]":
            return e!=+e ? t!=+t : e == 0 ? 1 / e == 1 / t : e ==+ t;
        case"[object Date]":
        case"[object Boolean]":
            return + e ==+ t;
        case"[object RegExp]":
            return e.source == t.source && e.global == t.global && e.multiline == t.multiline && e.ignoreCase == t.ignoreCase
        }
        if (typeof e != "object" || typeof t != "object")
            return !1;
        var i = n.length;
        while (i--)
            if (n[i] == e)
                return !0;
        n.push(e);
        var s = 0, o=!0;
        if (r == "[object Array]") {
            s = e.length, o = s == t.length;
            if (o)
                while (s--)
                    if (!(o = s in e == s in t && C(e[s], t[s], n)))
                        break
        } else {
            if ("constructor"in e != "constructor"in t || e.constructor != t.constructor)
                return !1;
            for (var u in e)
                if (S.has(e, u)) {
                    s++;
                    if (!(o = S.has(t, u) && C(e[u], t[u], n)))
                        break
                }
            if (o) {
                for (u in t)
                    if (S.has(t, u)&&!(s--))
                        break;
                o=!s
            }
        }
        return n.pop(), o
    }
    var e = this, t = e._, n = {}, r = Array.prototype, i = Object.prototype, s = Function.prototype, o = r.slice, u = r.unshift, a = i.toString, f = i.hasOwnProperty, l = r.forEach, c = r.map, h = r.reduce, p = r.reduceRight, d = r.filter, v = r.every, m = r.some, g = r.indexOf, y = r.lastIndexOf, b = Array.isArray, w = Object.keys, E = s.bind, S = function(e) {
        return new P(e)
    };
    typeof exports != "undefined" ? (typeof module != "undefined" && module.exports && (exports = module.exports = S), exports._ = S) : e._ = S, S.VERSION = "1.3.3";
    var x = S.each = S.forEach = function(e, t, r) {
        if (e == null)
            return;
        if (l && e.forEach === l)
            e.forEach(t, r);
        else if (e.length ===+ e.length) {
            for (var i = 0, s = e.length; i < s; i++)
                if (i in e && t.call(r, e[i], i, e) === n)
                    return 
        } else 
            for (var o in e)
                if (S.has(e, o) && t.call(r, e[o], o, e) === n)
                    return 
    };
    S.map = S.collect = function(e, t, n) {
        var r = [];
        return e == null ? r : c && e.map === c ? e.map(t, n) : (x(e, function(e, i, s) {
            r[r.length] = t.call(n, e, i, s)
        }), e.length ===+ e.length && (r.length = e.length), r)
    }, S.reduce = S.foldl = S.inject = function(e, t, n, r) {
        var i = arguments.length > 2;
        e == null && (e = []);
        if (h && e.reduce === h)
            return r && (t = S.bind(t, r)), i ? e.reduce(t, n) : e.reduce(t);
        x(e, function(e, s, o) {
            i ? n = t.call(r, n, e, s, o) : (n = e, i=!0)
        });
        if (!i)
            throw new TypeError("Reduce of empty array with no initial value");
        return n
    }, S.reduceRight = S.foldr = function(e, t, n, r) {
        var i = arguments.length > 2;
        e == null && (e = []);
        if (p && e.reduceRight === p)
            return r && (t = S.bind(t, r)), i ? e.reduceRight(t, n) : e.reduceRight(t);
        var s = S.toArray(e).reverse();
        return r&&!i && (t = S.bind(t, r)), i ? S.reduce(s, t, n, r) : S.reduce(s, t)
    }, S.find = S.detect = function(e, t, n) {
        var r;
        return T(e, function(e, i, s) {
            if (t.call(n, e, i, s))
                return r = e, !0
        }), r
    }, S.filter = S.select = function(e, t, n) {
        var r = [];
        return e == null ? r : d && e.filter === d ? e.filter(t, n) : (x(e, function(e, i, s) {
            t.call(n, e, i, s) && (r[r.length] = e)
        }), r)
    }, S.reject = function(e, t, n) {
        var r = [];
        return e == null ? r : (x(e, function(e, i, s) {
            t.call(n, e, i, s) || (r[r.length] = e)
        }), r)
    }, S.every = S.all = function(e, t, r) {
        var i=!0;
        return e == null ? i : v && e.every === v ? e.every(t, r) : (x(e, function(e, s, o) {
            if (!(i = i && t.call(r, e, s, o)))
                return n
        }), !!i)
    };
    var T = S.some = S.any = function(e, t, r) {
        t || (t = S.identity);
        var i=!1;
        return e == null ? i : m && e.some === m ? e.some(t, r) : (x(e, function(e, s, o) {
            if (i || (i = t.call(r, e, s, o)))
                return n
        }), !!i)
    };
    S.include = S.contains = function(e, t) {
        var n=!1;
        return e == null ? n : g && e.indexOf === g ? e.indexOf(t)!=-1 : (n = T(e, function(e) {
            return e === t
        }), n)
    }, S.invoke = function(e, t) {
        var n = o.call(arguments, 2);
        return S.map(e, function(e) {
            return (S.isFunction(t) ? t || e : e[t]).apply(e, n)
        })
    }, S.pluck = function(e, t) {
        return S.map(e, function(e) {
            return e[t]
        })
    }, S.max = function(e, t, n) {
        if (!t && S.isArray(e) && e[0] ===+ e[0])
            return Math.max.apply(Math, e);
        if (!t && S.isEmpty(e))
            return - Infinity;
        var r = {
            computed: - Infinity
        };
        return x(e, function(e, i, s) {
            var o = t ? t.call(n, e, i, s): e;
            o >= r.computed && (r = {
                value: e,
                computed: o
            })
        }), r.value
    }, S.min = function(e, t, n) {
        if (!t && S.isArray(e) && e[0] ===+ e[0])
            return Math.min.apply(Math, e);
        if (!t && S.isEmpty(e))
            return Infinity;
        var r = {
            computed: Infinity
        };
        return x(e, function(e, i, s) {
            var o = t ? t.call(n, e, i, s): e;
            o < r.computed && (r = {
                value: e,
                computed: o
            })
        }), r.value
    }, S.shuffle = function(e) {
        var t = [], n;
        return x(e, function(e, r, i) {
            n = Math.floor(Math.random() * (r + 1)), t[r] = t[n], t[n] = e
        }), t
    }, S.sortBy = function(e, t, n) {
        var r = S.isFunction(t) ? t: function(e) {
            return e[t]
        };
        return S.pluck(S.map(e, function(e, t, i) {
            return {
                value: e,
                criteria: r.call(n, e, t, i)
            }
        }).sort(function(e, t) {
            var n = e.criteria, r = t.criteria;
            return n === void 0 ? 1 : r === void 0?-1 : n < r?-1 : n > r ? 1 : 0
        }), "value")
    }, S.groupBy = function(e, t) {
        var n = {}, r = S.isFunction(t) ? t: function(e) {
            return e[t]
        };
        return x(e, function(e, t) {
            var i = r(e, t);
            (n[i] || (n[i] = [])).push(e)
        }), n
    }, S.sortedIndex = function(e, t, n) {
        n || (n = S.identity);
        var r = 0, i = e.length;
        while (r < i) {
            var s = r + i>>1;
            n(e[s]) < n(t) ? r = s + 1 : i = s
        }
        return r
    }, S.toArray = function(e) {
        return e ? S.isArray(e) ? o.call(e) : S.isArguments(e) ? o.call(e) : e.toArray && S.isFunction(e.toArray) ? e.toArray() : S.values(e) : []
    }, S.size = function(e) {
        return S.isArray(e) ? e.length : S.keys(e).length
    }, S.first = S.head = S.take = function(e, t, n) {
        return t != null&&!n ? o.call(e, 0, t) : e[0]
    }, S.initial = function(e, t, n) {
        return o.call(e, 0, e.length - (t == null || n ? 1 : t))
    }, S.last = function(e, t, n) {
        return t != null&&!n ? o.call(e, Math.max(e.length - t, 0)) : e[e.length - 1]
    }, S.rest = S.tail = function(e, t, n) {
        return o.call(e, t == null || n ? 1 : t)
    }, S.compact = function(e) {
        return S.filter(e, function(e) {
            return !!e
        })
    }, S.flatten = function(e, t) {
        return S.reduce(e, function(e, n) {
            return S.isArray(n) ? e.concat(t ? n : S.flatten(n)) : (e[e.length] = n, e)
        }, [])
    }, S.without = function(e) {
        return S.difference(e, o.call(arguments, 1))
    }, S.uniq = S.unique = function(e, t, n) {
        var r = n ? S.map(e, n): e, i = [];
        return e.length < 3 && (t=!0), S.reduce(r, function(n, r, s) {
            if (t ? S.last(n) !== r ||!n.length : !S.include(n, r))
                n.push(r), i.push(e[s]);
            return n
        }, []), i
    }, S.union = function() {
        return S.uniq(S.flatten(arguments, !0))
    }, S.intersection = S.intersect = function(e) {
        var t = o.call(arguments, 1);
        return S.filter(S.uniq(e), function(e) {
            return S.every(t, function(t) {
                return S.indexOf(t, e) >= 0
            })
        })
    }, S.difference = function(e) {
        var t = S.flatten(o.call(arguments, 1), !0);
        return S.filter(e, function(e) {
            return !S.include(t, e)
        })
    }, S.zip = function() {
        var e = o.call(arguments), t = S.max(S.pluck(e, "length")), n = new Array(t);
        for (var r = 0; r < t; r++)
            n[r] = S.pluck(e, "" + r);
        return n
    }, S.indexOf = function(e, t, n) {
        if (e == null)
            return - 1;
        var r, i;
        if (n)
            return r = S.sortedIndex(e, t), e[r] === t ? r : - 1;
        if (g && e.indexOf === g)
            return e.indexOf(t);
        for (r = 0, i = e.length; r < i; r++)
            if (r in e && e[r] === t)
                return r;
        return - 1
    }, S.lastIndexOf = function(e, t) {
        if (e == null)
            return - 1;
        if (y && e.lastIndexOf === y)
            return e.lastIndexOf(t);
        var n = e.length;
        while (n--)
            if (n in e && e[n] === t)
                return n;
        return - 1
    }, S.range = function(e, t, n) {
        arguments.length <= 1 && (t = e || 0, e = 0), n = arguments[2] || 1;
        var r = Math.max(Math.ceil((t - e) / n), 0), i = 0, s = new Array(r);
        while (i < r)
            s[i++] = e, e += n;
        return s
    };
    var N = function() {};
    S.bind = function(t, n) {
        var r, i;
        if (t.bind === E && E)
            return E.apply(t, o.call(arguments, 1));
        if (!S.isFunction(t))
            throw new TypeError;
        return i = o.call(arguments, 2), r = function() {
            if (this instanceof r) {
                N.prototype = t.prototype;
                var e = new N, s = t.apply(e, i.concat(o.call(arguments)));
                return Object(s) === s ? s : e
            }
            return t.apply(n, i.concat(o.call(arguments)))
        }
    }, S.bindAll = function(e) {
        var t = o.call(arguments, 1);
        return t.length == 0 && (t = S.functions(e)), x(t, function(t) {
            e[t] = S.bind(e[t], e)
        }), e
    }, S.memoize = function(e, t) {
        var n = {};
        return t || (t = S.identity), function() {
            var r = t.apply(this, arguments);
            return S.has(n, r) ? n[r] : n[r] = e.apply(this, arguments)
        }
    }, S.delay = function(e, t) {
        var n = o.call(arguments, 2);
        return setTimeout(function() {
            return e.apply(null, n)
        }, t)
    }, S.defer = function(e) {
        return S.delay.apply(S, [e, 1].concat(o.call(arguments, 1)))
    }, S.throttle = function(e, t) {
        var n, r, i, s, o, u, a = S.debounce(function() {
            o = s=!1
        }, t);
        return function() {
            n = this, r = arguments;
            var f = function() {
                i = null, o && e.apply(n, r), a()
            };
            return i || (i = setTimeout(f, t)), s ? o=!0 : u = e.apply(n, r), a(), s=!0, u
        }
    }, S.debounce = function(e, t, n) {
        var r;
        return function() {
            var i = this, s = arguments, o = function() {
                r = null, n || e.apply(i, s)
            };
            n&&!r && e.apply(i, s), clearTimeout(r), r = setTimeout(o, t)
        }
    }, S.once = function(e) {
        var t=!1, n;
        return function() {
            return t ? n : (t=!0, n = e.apply(this, arguments))
        }
    }, S.wrap = function(e, t) {
        return function() {
            var n = [e].concat(o.call(arguments, 0));
            return t.apply(this, n)
        }
    }, S.compose = function() {
        var e = arguments;
        return function() {
            var t = arguments;
            for (var n = e.length - 1; n >= 0; n--)
                t = [e[n].apply(this, t)];
            return t[0]
        }
    }, S.after = function(e, t) {
        return e <= 0 ? t() : function() {
            if (--e < 1)
                return t.apply(this, arguments)
        }
    }, S.keys = w || function(e) {
        if (e !== Object(e))
            throw new TypeError("Invalid object");
        var t = [];
        for (var n in e)
            S.has(e, n) && (t[t.length] = n);
        return t
    }, S.values = function(e) {
        return S.map(e, S.identity)
    }, S.functions = S.methods = function(e) {
        var t = [];
        for (var n in e)
            S.isFunction(e[n]) && t.push(n);
        return t.sort()
    }, S.extend = function(e) {
        return x(o.call(arguments, 1), function(t) {
            for (var n in t)
                e[n] = t[n]
        }), e
    }, S.pick = function(e) {
        var t = {};
        return x(S.flatten(o.call(arguments, 1)), function(n) {
            n in e && (t[n] = e[n])
        }), t
    }, S.defaults = function(e) {
        return x(o.call(arguments, 1), function(t) {
            for (var n in t)
                e[n] == null && (e[n] = t[n])
        }), e
    }, S.clone = function(e) {
        return S.isObject(e) ? S.isArray(e) ? e.slice() : S.extend({}, e) : e
    }, S.tap = function(e, t) {
        return t(e), e
    }, S.isEqual = function(e, t) {
        return C(e, t, [])
    }, S.isEmpty = function(e) {
        if (e == null)
            return !0;
        if (S.isArray(e) || S.isString(e))
            return e.length === 0;
        for (var t in e)
            if (S.has(e, t))
                return !1;
        return !0
    }, S.isElement = function(e) {
        return !!e && e.nodeType == 1
    }, S.isArray = b || function(e) {
        return a.call(e) == "[object Array]"
    }, S.isObject = function(e) {
        return e === Object(e)
    }, S.isArguments = function(e) {
        return a.call(e) == "[object Arguments]"
    }, S.isArguments(arguments) || (S.isArguments = function(e) {
        return !!e&&!!S.has(e, "callee")
    }), S.isFunction = function(e) {
        return a.call(e) == "[object Function]"
    }, S.isString = function(e) {
        return a.call(e) == "[object String]"
    }, S.isNumber = function(e) {
        return a.call(e) == "[object Number]"
    }, S.isFinite = function(e) {
        return S.isNumber(e) && isFinite(e)
    }, S.isNaN = function(e) {
        return e !== e
    }, S.isBoolean = function(e) {
        return e===!0 || e===!1 || a.call(e) == "[object Boolean]"
    }, S.isDate = function(e) {
        return a.call(e) == "[object Date]"
    }, S.isRegExp = function(e) {
        return a.call(e) == "[object RegExp]"
    }, S.isNull = function(e) {
        return e === null
    }, S.isUndefined = function(e) {
        return e === void 0
    }, S.has = function(e, t) {
        return f.call(e, t)
    }, S.noConflict = function() {
        return e._ = t, this
    }, S.identity = function(e) {
        return e
    }, S.times = function(e, t, n) {
        for (var r = 0; r < e; r++)
            t.call(n, r)
    }, S.escape = function(e) {
        return ("" + e).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#x27;").replace(/\//g, "&#x2F;")
    }, S.result = function(e, t) {
        if (e == null)
            return null;
        var n = e[t];
        return S.isFunction(n) ? n.call(e) : n
    }, S.mixin = function(e) {
        x(S.functions(e), function(t) {
            B(t, S[t] = e[t])
        })
    };
    var k = 0;
    S.uniqueId = function(e) {
        var t = k++;
        return e ? e + t : t
    }, S.templateSettings = {
        evaluate: /<%([\s\S]+?)%>/g,
        interpolate: /<%=([\s\S]+?)%>/g,
        escape: /<%-([\s\S]+?)%>/g
    };
    var L = /.^/, A = {
        "\\": "\\",
        "'": "'",
        r: "\r",
        n: "\n",
        t: "	",
        u2028: "\u2028",
        u2029: "\u2029"
    };
    for (var O in A)
        A[A[O]] = O;
    var M = /\\|'|\r|\n|\t|\u2028|\u2029/g, _ = /\\(\\|'|r|n|t|u2028|u2029)/g, D = function(e) {
        return e.replace(_, function(e, t) {
            return A[t]
        })
    };
    S.template = function(e, t, n) {
        n = S.defaults(n || {}, S.templateSettings);
        var r = "__p+='" + e.replace(M, function(e) {
            return "\\" + A[e]
        }).replace(n.escape || L, function(e, t) {
            return "'+\n_.escape(" + D(t) + ")+\n'"
        }).replace(n.interpolate || L, function(e, t) {
            return "'+\n(" + D(t) + ")+\n'"
        }).replace(n.evaluate || L, function(e, t) {
            return "';\n" + D(t) + "\n;__p+='"
        }) + "';\n";
        n.variable || (r = "with(obj||{}){\n" + r + "}\n"), r = "var __p='';var print=function(){__p+=Array.prototype.join.call(arguments, '')};\n" + r + "return __p;\n";
        var i = new Function(n.variable || "obj", "_", r);
        if (t)
            return i(t, S);
        var s = function(e) {
            return i.call(this, e, S)
        };
        return s.source = "function(" + (n.variable || "obj") + "){\n" + r + "}", s
    }, S.chain = function(e) {
        return S(e).chain()
    };
    var P = function(e) {
        this._wrapped = e
    };
    S.prototype = P.prototype;
    var H = function(e, t) {
        return t ? S(e).chain() : e
    }, B = function(e, t) {
        P.prototype[e] = function() {
            var e = o.call(arguments);
            return u.call(e, this._wrapped), H(t.apply(S, e), this._chain)
        }
    };
    S.mixin(S), x(["pop", "push", "reverse", "shift", "sort", "splice", "unshift"], function(e) {
        var t = r[e];
        P.prototype[e] = function() {
            var n = this._wrapped;
            t.apply(n, arguments);
            var r = n.length;
            return (e == "shift" || e == "splice") && r === 0 && delete n[0], H(n, this._chain)
        }
    }), x(["concat", "join", "slice"], function(e) {
        var t = r[e];
        P.prototype[e] = function() {
            return H(t.apply(this._wrapped, arguments), this._chain)
        }
    }), P.prototype.chain = function() {
        return this._chain=!0, this
    }, P.prototype.value = function() {
        return this._wrapped
    }
}).call(this), function() {
    var e = this, t = e.Backbone, n = [], r = n.push, i = n.slice, s = n.splice, o;
    typeof exports != "undefined" ? o = exports : o = e.Backbone = {}, o.VERSION = "1.0.0";
    var u = e._;
    !u && typeof require != "undefined" && (u = require("underscore")), o.$ = e.jQuery || e.Zepto || e.ender || e.$, o.noConflict = function() {
        return e.Backbone = t, this
    }, o.emulateHTTP=!1, o.emulateJSON=!1;
    var a = o.Events = {
        on: function(e, t, n) {
            if (!l(this, "on", e, [t, n]) ||!t)
                return this;
            this._events || (this._events = {});
            var r = this._events[e] || (this._events[e] = []);
            return r.push({
                callback: t,
                context: n,
                ctx: n || this
            }), this
        },
        once: function(e, t, n) {
            if (!l(this, "once", e, [t, n]) ||!t)
                return this;
            var r = this, i = u.once(function() {
                r.off(e, i), t.apply(this, arguments)
            });
            return i._callback = t, this.on(e, i, n)
        },
        off: function(e, t, n) {
            var r, i, s, o, a, f, c, h;
            if (!this._events ||!l(this, "off", e, [t, n]))
                return this;
            if (!e&&!t&&!n)
                return this._events = {}, this;
            o = e ? [e] : u.keys(this._events);
            for (a = 0, f = o.length; a < f; a++) {
                e = o[a];
                if (s = this._events[e]) {
                    this._events[e] = r = [];
                    if (t || n)
                        for (c = 0, h = s.length; c < h; c++)
                            i = s[c], (t && t !== i.callback && t !== i.callback._callback || n && n !== i.context) && r.push(i);
                    r.length || delete this._events[e]
                }
            }
            return this
        },
        trigger: function(e) {
            if (!this._events)
                return this;
            var t = i.call(arguments, 1);
            if (!l(this, "trigger", e, t))
                return this;
            var n = this._events[e], r = this._events.all;
            return n && c(n, t), r && c(r, arguments), this
        },
        stopListening: function(e, t, n) {
            var r = this._listeners;
            if (!r)
                return this;
            var i=!t&&!n;
            typeof t == "object" && (n = this), e && ((r = {})[e._listenerId] = e);
            for (var s in r)
                r[s].off(t, n, this), i && delete this._listeners[s];
            return this
        }
    }, f = /\s+/, l = function(e, t, n, r) {
        if (!n)
            return !0;
        if (typeof n == "object") {
            for (var i in n)
                e[t].apply(e, [i, n[i]].concat(r));
            return !1
        }
        if (f.test(n)) {
            var s = n.split(f);
            for (var o = 0, u = s.length; o < u; o++)
                e[t].apply(e, [s[o]].concat(r));
            return !1
        }
        return !0
    }, c = function(e, t) {
        var n, r =- 1, i = e.length, s = t[0], o = t[1], u = t[2];
        switch (t.length) {
        case 0:
            while (++r < i)(n = e[r]).callback.call(n.ctx);
            return;
        case 1:
            while (++r < i)(n = e[r]).callback.call(n.ctx, s);
            return;
        case 2:
            while (++r < i)(n = e[r]).callback.call(n.ctx, s, o);
            return;
        case 3:
            while (++r < i)(n = e[r]).callback.call(n.ctx, s, o, u);
            return;
        default:
            while (++r < i)(n = e[r]).callback.apply(n.ctx, t)
            }
    }, h = {
        listenTo: "on",
        listenToOnce: "once"
    };
    u.each(h, function(e, t) {
        a[t] = function(t, n, r) {
            var i = this._listeners || (this._listeners = {}), s = t._listenerId || (t._listenerId = u.uniqueId("l"));
            return i[s] = t, typeof n == "object" && (r = this), t[e](n, r, this), this
        }
    }), a.bind = a.on, a.unbind = a.off, u.extend(o, a);
    var p = o.Model = function(e, t) {
        var n, r = e || {};
        t || (t = {}), this.cid = u.uniqueId("c"), this.attributes = {}, u.extend(this, u.pick(t, d)), t.parse && (r = this.parse(r, t) || {});
        if (n = u.result(this, "defaults"))
            r = u.defaults({}, r, n);
        this.set(r, t), this.changed = {}, this.initialize.apply(this, arguments)
    }, d = ["url", "urlRoot", "collection"];
    u.extend(p.prototype, a, {
        changed: null,
        validationError: null,
        idAttribute: "id",
        initialize: function() {},
        toJSON: function(e) {
            return u.clone(this.attributes)
        },
        sync: function() {
            return o.sync.apply(this, arguments)
        },
        get: function(e) {
            return this.attributes[e]
        },
        escape: function(e) {
            return u.escape(this.get(e))
        },
        has: function(e) {
            return this.get(e) != null
        },
        set: function(e, t, n) {
            var r, i, s, o, a, f, l, c;
            if (e == null)
                return this;
            typeof e == "object" ? (i = e, n = t) : (i = {})[e] = t, n || (n = {});
            if (!this._validate(i, n))
                return !1;
            s = n.unset, a = n.silent, o = [], f = this._changing, this._changing=!0, f || (this._previousAttributes = u.clone(this.attributes), this.changed = {}), c = this.attributes, l = this._previousAttributes, this.idAttribute in i && (this.id = i[this.idAttribute]);
            for (r in i)
                t = i[r], u.isEqual(c[r], t) || o.push(r), u.isEqual(l[r], t) ? delete this.changed[r] : this.changed[r] = t, s ? delete c[r] : c[r] = t;
            if (!a) {
                o.length && (this._pending=!0);
                for (var h = 0, p = o.length; h < p; h++)
                    this.trigger("change:" + o[h], this, c[o[h]], n)
            }
            if (f)
                return this;
            if (!a)
                while (this._pending)
                    this._pending=!1, this.trigger("change", this, n);
            return this._pending=!1, this._changing=!1, this
        },
        unset: function(e, t) {
            return this.set(e, void 0, u.extend({}, t, {
                unset : !0
            }))
        },
        clear: function(e) {
            var t = {};
            for (var n in this.attributes)
                t[n] = void 0;
            return this.set(t, u.extend({}, e, {
                unset: !0
            }))
        },
        hasChanged: function(e) {
            return e == null?!u.isEmpty(this.changed) : u.has(this.changed, e)
        },
        changedAttributes: function(e) {
            if (!e)
                return this.hasChanged() ? u.clone(this.changed) : !1;
            var t, n=!1, r = this._changing ? this._previousAttributes : this.attributes;
            for (var i in e) {
                if (u.isEqual(r[i], t = e[i]))
                    continue;
                (n || (n = {}))[i] = t
            }
            return n
        },
        previous: function(e) {
            return e == null ||!this._previousAttributes ? null : this._previousAttributes[e]
        },
        previousAttributes: function() {
            return u.clone(this._previousAttributes)
        },
        fetch: function(e) {
            e = e ? u.clone(e) : {}, e.parse === void 0 && (e.parse=!0);
            var t = this, n = e.success;
            return e.success = function(r) {
                if (!t.set(t.parse(r, e), e))
                    return !1;
                n && n(t, r, e), t.trigger("sync", t, r, e)
            }, j(this, e), this.sync("read", this, e)
        },
        save: function(e, t, n) {
            var r, i, s, o = this.attributes;
            e == null || typeof e == "object" ? (r = e, n = t) : (r = {})[e] = t;
            if (r && (!n ||!n.wait)&&!this.set(r, n))
                return !1;
            n = u.extend({
                validate: !0
            }, n);
            if (!this._validate(r, n))
                return !1;
            r && n.wait && (this.attributes = u.extend({}, o, r)), n.parse === void 0 && (n.parse=!0);
            var a = this, f = n.success;
            return n.success = function(e) {
                a.attributes = o;
                var t = a.parse(e, n);
                n.wait && (t = u.extend(r || {}, t));
                if (u.isObject(t)&&!a.set(t, n))
                    return !1;
                f && f(a, e, n), a.trigger("sync", a, e, n)
            }, j(this, n), i = this.isNew() ? "create" : n.patch ? "patch" : "update", i === "patch" && (n.attrs = r), s = this.sync(i, this, n), r && n.wait && (this.attributes = o), s
        },
        destroy: function(e) {
            e = e ? u.clone(e) : {};
            var t = this, n = e.success, r = function() {
                t.trigger("destroy", t, t.collection, e)
            };
            e.success = function(i) {
                (e.wait || t.isNew()) && r(), n && n(t, i, e), t.isNew() || t.trigger("sync", t, i, e)
            };
            if (this.isNew())
                return e.success(), !1;
            j(this, e);
            var i = this.sync("delete", this, e);
            return e.wait || r(), i
        },
        url: function() {
            var e = u.result(this, "urlRoot") || u.result(this.collection, "url") || B();
            return this.isNew() ? e : e + (e.charAt(e.length - 1) === "/" ? "" : "/") + encodeURIComponent(this.id)
        },
        parse: function(e, t) {
            return e
        },
        clone: function() {
            return new this.constructor(this.attributes)
        },
        isNew: function() {
            return this.id == null
        },
        isValid: function(e) {
            return this._validate({}, u.extend(e || {}, {
                validate: !0
            }))
        },
        _validate: function(e, t) {
            if (!t.validate ||!this.validate)
                return !0;
            e = u.extend({}, this.attributes, e);
            var n = this.validationError = this.validate(e, t) || null;
            return n ? (this.trigger("invalid", this, n, u.extend(t || {}, {
                validationError: n
            })), !1) : !0
        }
    });
    var v = ["keys", "values", "pairs", "invert", "pick", "omit"];
    u.each(v, function(e) {
        p.prototype[e] = function() {
            var t = i.call(arguments);
            return t.unshift(this.attributes), u[e].apply(u, t)
        }
    });
    var m = o.Collection = function(e, t) {
        t || (t = {}), t.url && (this.url = t.url), t.model && (this.model = t.model), t.comparator !== void 0 && (this.comparator = t.comparator), this._reset(), this.initialize.apply(this, arguments), e && this.reset(e, u.extend({
            silent: !0
        }, t))
    }, g = {
        add: !0,
        remove: !0,
        merge: !0
    }, y = {
        add: !0,
        merge: !1,
        remove: !1
    };
    u.extend(m.prototype, a, {
        model: p,
        initialize: function() {},
        toJSON: function(e) {
            return this.map(function(t) {
                return t.toJSON(e)
            })
        },
        sync: function() {
            return o.sync.apply(this, arguments)
        },
        add: function(e, t) {
            return this.set(e, u.defaults(t || {}, y))
        },
        remove: function(e, t) {
            e = u.isArray(e) ? e.slice() : [e], t || (t = {});
            var n, r, i, s;
            for (n = 0, r = e.length; n < r; n++) {
                s = this.get(e[n]);
                if (!s)
                    continue;
                delete this._byId[s.id], delete this._byId[s.cid], i = this.indexOf(s), this.models.splice(i, 1), this.length--, t.silent || (t.index = i, s.trigger("remove", s, this, t)), this._removeReference(s)
            }
            return this
        },
        set: function(e, t) {
            t = u.defaults(t || {}, g), t.parse && (e = this.parse(e, t)), u.isArray(e) || (e = e ? [e] : []);
            var n, i, o, a, f, l, c = t.at, h = this.comparator && c == null && t.sort!==!1, p = u.isString(this.comparator) ? this.comparator: null, d = [], v = [], m = {};
            for (n = 0, i = e.length; n < i; n++) {
                if (!(o = this._prepareModel(e[n], t)))
                    continue;
                (f = this.get(o)) ? (t.remove && (m[f.cid]=!0), t.merge && (f.set(o.attributes, t), h&&!l && f.hasChanged(p) && (l=!0))) : t.add && (d.push(o), o.on("all", this._onModelEvent, this), this._byId[o.cid] = o, o.id != null && (this._byId[o.id] = o))
            }
            if (t.remove) {
                for (n = 0, i = this.length; n < i; ++n)
                    m[(o = this.models[n]).cid] || v.push(o);
                v.length && this.remove(v, t)
            }
            d.length && (h && (l=!0), this.length += d.length, c != null ? s.apply(this.models, [c, 0].concat(d)) : r.apply(this.models, d)), l && this.sort({
                silent: !0
            });
            if (t.silent)
                return this;
            for (n = 0, i = d.length; n < i; n++)(o = d[n])
                .trigger("add", o, this, t);
            return l && this.trigger("sort", this, t), this
        },
        reset: function(e, t) {
            t || (t = {});
            for (var n = 0, r = this.models.length; n < r; n++)
                this._removeReference(this.models[n]);
            return t.previousModels = this.models, this._reset(), this.add(e, u.extend({
                silent: !0
            }, t)), t.silent || this.trigger("reset", this, t), this
        },
        push: function(e, t) {
            return e = this._prepareModel(e, t), this.add(e, u.extend({
                at: this.length
            }, t)), e
        },
        pop: function(e) {
            var t = this.at(this.length - 1);
            return this.remove(t, e), t
        },
        unshift: function(e, t) {
            return e = this._prepareModel(e, t), this.add(e, u.extend({
                at: 0
            }, t)), e
        },
        shift: function(e) {
            var t = this.at(0);
            return this.remove(t, e), t
        },
        slice: function(e, t) {
            return this.models.slice(e, t)
        },
        get: function(e) {
            return e == null ? void 0 : this._byId[e.id != null ? e.id: e.cid || e]
        },
        at: function(e) {
            return this.models[e]
        },
        where: function(e, t) {
            return u.isEmpty(e) ? t ? void 0 : [] : this[t ? "find": "filter"](function(t) {
                for (var n in e)
                    if (e[n] !== t.get(n))
                        return !1;
                return !0
            })
        },
        findWhere: function(e) {
            return this.where(e, !0)
        },
        sort: function(e) {
            if (!this.comparator)
                throw new Error("Cannot sort a set without a comparator");
            return e || (e = {}), u.isString(this.comparator) || this.comparator.length === 1 ? this.models = this.sortBy(this.comparator, this) : this.models.sort(u.bind(this.comparator, this)), e.silent || this.trigger("sort", this, e), this
        },
        sortedIndex: function(e, t, n) {
            t || (t = this.comparator);
            var r = u.isFunction(t) ? t: function(e) {
                return e.get(t)
            };
            return u.sortedIndex(this.models, e, r, n)
        },
        pluck: function(e) {
            return u.invoke(this.models, "get", e)
        },
        fetch: function(e) {
            e = e ? u.clone(e) : {}, e.parse === void 0 && (e.parse=!0);
            var t = e.success, n = this;
            return e.success = function(r) {
                var i = e.reset ? "reset": "set";
                n[i](r, e), t && t(n, r, e), n.trigger("sync", n, r, e)
            }, j(this, e), this.sync("read", this, e)
        },
        create: function(e, t) {
            t = t ? u.clone(t) : {};
            if (!(e = this._prepareModel(e, t)))
                return !1;
            t.wait || this.add(e, t);
            var n = this, r = t.success;
            return t.success = function(i) {
                t.wait && n.add(e, t), r && r(e, i, t)
            }, e.save(null, t), e
        },
        parse: function(e, t) {
            return e
        },
        clone: function() {
            return new this.constructor(this.models)
        },
        _reset: function() {
            this.length = 0, this.models = [], this._byId = {}
        },
        _prepareModel: function(e, t) {
            if (e instanceof p)
                return e.collection || (e.collection = this), e;
            t || (t = {}), t.collection = this;
            var n = new this.model(e, t);
            return n._validate(e, t) ? n : (this.trigger("invalid", this, e, t), !1)
        },
        _removeReference: function(e) {
            this === e.collection && delete e.collection, e.off("all", this._onModelEvent, this)
        },
        _onModelEvent: function(e, t, n, r) {
            if ((e === "add" || e === "remove") && n !== this)
                return;
            e === "destroy" && this.remove(t, r), t && e === "change:" + t.idAttribute && (delete this._byId[t.previous(t.idAttribute)], t.id != null && (this._byId[t.id] = t)), this.trigger.apply(this, arguments)
        }
    });
    var b = ["forEach", "each", "map", "collect", "reduce", "foldl", "inject", "reduceRight", "foldr", "find", "detect", "filter", "select", "reject", "every", "all", "some", "any", "include", "contains", "invoke", "max", "min", "toArray", "size", "first", "head", "take", "initial", "rest", "tail", "drop", "last", "without", "indexOf", "shuffle", "lastIndexOf", "isEmpty", "chain"];
    u.each(b, function(e) {
        m.prototype[e] = function() {
            var t = i.call(arguments);
            return t.unshift(this.models), u[e].apply(u, t)
        }
    });
    var w = ["groupBy", "countBy", "sortBy"];
    u.each(w, function(e) {
        m.prototype[e] = function(t, n) {
            var r = u.isFunction(t) ? t: function(e) {
                return e.get(t)
            };
            return u[e](this.models, r, n)
        }
    });
    var E = o.View = function(e) {
        this.cid = u.uniqueId("view"), this._configure(e || {}), this._ensureElement(), this.initialize.apply(this, arguments), this.delegateEvents()
    }, S = /^(\S+)\s*(.*)$/, x = ["model", "collection", "el", "id", "attributes", "className", "tagName", "events"];
    u.extend(E.prototype, a, {
        tagName: "div",
        $: function(e) {
            return this.$el.find(e)
        },
        initialize: function() {},
        render: function() {
            return this
        },
        remove: function() {
            return this.$el.remove(), this.stopListening(), this
        },
        setElement: function(e, t) {
            return this.$el && this.undelegateEvents(), this.$el = e instanceof o.$ ? e : o.$(e), this.el = this.$el[0], t!==!1 && this.delegateEvents(), this
        },
        delegateEvents: function(e) {
            if (!e&&!(e = u.result(this, "events")))
                return this;
            this.undelegateEvents();
            for (var t in e) {
                var n = e[t];
                u.isFunction(n) || (n = this[e[t]]);
                if (!n)
                    continue;
                var r = t.match(S), i = r[1], s = r[2];
                n = u.bind(n, this), i += ".delegateEvents" + this.cid, s === "" ? this.$el.on(i, n) : this.$el.on(i, s, n)
            }
            return this
        },
        undelegateEvents: function() {
            return this.$el.off(".delegateEvents" + this.cid), this
        },
        _configure: function(e) {
            this.options && (e = u.extend({}, u.result(this, "options"), e)), u.extend(this, u.pick(e, x)), this.options = e
        },
        _ensureElement: function() {
            if (!this.el) {
                var e = u.extend({}, u.result(this, "attributes"));
                this.id && (e.id = u.result(this, "id")), this.className && (e["class"] = u.result(this, "className"));
                var t = o.$("<" + u.result(this, "tagName") + ">").attr(e);
                this.setElement(t, !1)
            } else 
                this.setElement(u.result(this, "el"), !1)
        }
    }), o.sync = function(e, t, n) {
        var r = T[e];
        u.defaults(n || (n = {}), {
            emulateHTTP: o.emulateHTTP,
            emulateJSON: o.emulateJSON
        });
        var i = {
            type: r,
            dataType: "json"
        };
        n.url || (i.url = u.result(t, "url") || B()), n.data == null && t && (e === "create" || e === "update" || e === "patch") && (i.contentType = "application/json", i.data = JSON.stringify(n.attrs || t.toJSON(n))), n.emulateJSON && (i.contentType = "application/x-www-form-urlencoded", i.data = i.data ? {
            model: i.data
        } : {});
        if (n.emulateHTTP && (r === "PUT" || r === "DELETE" || r === "PATCH")) {
            i.type = "POST", n.emulateJSON && (i.data._method = r);
            var s = n.beforeSend;
            n.beforeSend = function(e) {
                e.setRequestHeader("X-HTTP-Method-Override", r);
                if (s)
                    return s.apply(this, arguments)
            }
        }
        i.type !== "GET"&&!n.emulateJSON && (i.processData=!1), i.type === "PATCH" && window.ActiveXObject && (!window.external ||!window.external.msActiveXFilteringEnabled) && (i.xhr = function() {
            return new ActiveXObject("Microsoft.XMLHTTP")
        });
        var a = n.xhr = o.ajax(u.extend(i, n));
        return t.trigger("request", t, a, n), a
    };
    var T = {
        create: "POST",
        update: "PUT",
        patch: "PATCH",
        "delete": "DELETE",
        read: "GET"
    };
    o.ajax = function() {
        return o.$.ajax.apply(o.$, arguments)
    };
    var N = o.Router = function(e) {
        e || (e = {}), e.routes && (this.routes = e.routes), this._bindRoutes(), this.initialize.apply(this, arguments)
    }, C = /\((.*?)\)/g, k = /(\(\?)?:\w+/g, L = /\*\w+/g, A = /[\-{}\[\]+?.,\\\^$|#\s]/g;
    u.extend(N.prototype, a, {
        initialize: function() {},
        route: function(e, t, n) {
            u.isRegExp(e) || (e = this._routeToRegExp(e)), u.isFunction(t) && (n = t, t = ""), n || (n = this[t]);
            var r = this;
            return o.history.route(e, function(i) {
                var s = r._extractParameters(e, i);
                n && n.apply(r, s), r.trigger.apply(r, ["route:" + t].concat(s)), r.trigger("route", t, s), o.history.trigger("route", r, t, s)
            }), this
        },
        navigate: function(e, t) {
            return o.history.navigate(e, t), this
        },
        _bindRoutes: function() {
            if (!this.routes)
                return;
            this.routes = u.result(this, "routes");
            var e, t = u.keys(this.routes);
            while ((e = t.pop()) != null)
                this.route(e, this.routes[e])
        },
        _routeToRegExp: function(e) {
            return e = e.replace(A, "\\$&").replace(C, "(?:$1)?").replace(k, function(e, t) {
                return t ? e : "([^/]+)"
            }).replace(L, "(.*?)"), new RegExp("^" + e + "$")
        },
        _extractParameters: function(e, t) {
            var n = e.exec(t).slice(1);
            return u.map(n, function(e) {
                return e ? decodeURIComponent(e) : null
            })
        }
    });
    var O = o.History = function() {
        this.handlers = [], u.bindAll(this, "checkUrl"), typeof window != "undefined" && (this.location = window.location, this.history = window.history)
    }, M = /^[#\/]|\s+$/g, _ = /^\/+|\/+$/g, D = /msie [\w.]+/, P = /\/$/;
    O.started=!1, u.extend(O.prototype, a, {
        interval: 50,
        getHash: function(e) {
            var t = (e || this).location.href.match(/#(.*)$/);
            return t ? t[1] : ""
        },
        getFragment: function(e, t) {
            if (e == null)
                if (this._hasPushState ||!this._wantsHashChange || t) {
                    e = this.location.pathname;
                    var n = this.root.replace(P, "");
                    e.indexOf(n) || (e = e.substr(n.length))
                } else 
                    e = this.getHash();
            return e.replace(M, "")
        },
        start: function(e) {
            if (O.started)
                throw new Error("Backbone.history has already been started");
            O.started=!0, this.options = u.extend({}, {
                root: "/"
            }, this.options, e), this.root = this.options.root, this._wantsHashChange = this.options.hashChange!==!1, this._wantsPushState=!!this.options.pushState, this._hasPushState=!!(this.options.pushState && this.history && this.history.pushState);
            var t = this.getFragment(), n = document.documentMode, r = D.exec(navigator.userAgent.toLowerCase()) && (!n || n <= 7);
            this.root = ("/" + this.root + "/").replace(_, "/"), r && this._wantsHashChange && (this.iframe = o.$('<iframe src="javascript:0" tabindex="-1" />').hide().appendTo("body")[0].contentWindow, this.navigate(t)), this._hasPushState ? o.$(window).on("popstate", this.checkUrl) : this._wantsHashChange && "onhashchange"in window&&!r ? o.$(window).on("hashchange", this.checkUrl) : this._wantsHashChange && (this._checkUrlInterval = setInterval(this.checkUrl, this.interval)), this.fragment = t;
            var i = this.location, s = i.pathname.replace(/[^\/]$/, "$&/") === this.root;
            if (this._wantsHashChange && this._wantsPushState&&!this._hasPushState&&!s)
                return this.fragment = this.getFragment(null, !0), this.location.replace(this.root + this.location.search + "#" + this.fragment), !0;
            this._wantsPushState && this._hasPushState && s && i.hash && (this.fragment = this.getHash().replace(M, ""), this.history.replaceState({}, document.title, this.root + this.fragment + i.search));
            if (!this.options.silent)
                return this.loadUrl()
        },
        stop: function() {
            o.$(window).off("popstate", this.checkUrl).off("hashchange", this.checkUrl), clearInterval(this._checkUrlInterval), O.started=!1
        },
        route: function(e, t) {
            this.handlers.unshift({
                route: e,
                callback: t
            })
        },
        checkUrl: function(e) {
            var t = this.getFragment();
            t === this.fragment && this.iframe && (t = this.getFragment(this.getHash(this.iframe)));
            if (t === this.fragment)
                return !1;
            this.iframe && this.navigate(t), this.loadUrl() || this.loadUrl(this.getHash())
        },
        loadUrl: function(e) {
            var t = this.fragment = this.getFragment(e), n = u.any(this.handlers, function(e) {
                if (e.route.test(t))
                    return e.callback(t), !0
            });
            return n
        },
        navigate: function(e, t) {
            if (!O.started)
                return !1;
            if (!t || t===!0)
                t = {
                    trigger: t
                };
            e = this.getFragment(e || "");
            if (this.fragment === e)
                return;
            this.fragment = e;
            var n = this.root + e;
            if (this._hasPushState)
                this.history[t.replace ? "replaceState": "pushState"]({}, document.title, n);
            else {
                if (!this._wantsHashChange)
                    return this.location.assign(n);
                this._updateHash(this.location, e, t.replace), this.iframe && e !== this.getFragment(this.getHash(this.iframe)) && (t.replace || this.iframe.document.open().close(), this._updateHash(this.iframe.location, e, t.replace))
            }
            t.trigger && this.loadUrl(e)
        },
        _updateHash: function(e, t, n) {
            if (n) {
                var r = e.href.replace(/(javascript:|#).*$/, "");
                e.replace(r + "#" + t)
            } else 
                e.hash = "#" + t
        }
    }), o.history = new O;
    var H = function(e, t) {
        var n = this, r;
        e && u.has(e, "constructor") ? r = e.constructor : r = function() {
            return n.apply(this, arguments)
        }, u.extend(r, n, t);
        var i = function() {
            this.constructor = r
        };
        return i.prototype = n.prototype, r.prototype = new i, e && u.extend(r.prototype, e), r.__super__ = n.prototype, r
    };
    p.extend = m.extend = N.extend = E.extend = O.extend = H;
    var B = function() {
        throw new Error('A "url" property or function must be specified')
    }, j = function(e, t) {
        var n = t.error;
        t.error = function(r) {
            n && n(e, r, t), e.trigger("error", e, r, t)
        }
    }
}.call(this), function(e) {
    "use strict";
    var t = e.jQuery, n = e._, r = e.Backbone, i = n.sortedIndex, s = r.Events, o = r.View, u = o.prototype.remove, a = "breakpoints", f = function(e, t, n) {
        var r = i(n, e + 1), s = i(n, t + 1);
        return r !== s ? n.slice(Math.min(r, s), Math.max(r, s)) : []
    }, l = r.Resize = {
        trigger: s.trigger,
        width: function() {
            return t(e).width()
        }
    };
    l[a] = [], n.each(["on", "off", "bind", "unbind"], function(t) {
        l[t] = function(r, i, o) {
            s[t].call(this, r, i, o), this[a] = n.compact(n.map(n.keys(this._events || {}), function(t) {
                return e.parseInt(t)
            })).sort(function(e, t) {
                return e - t
            })
        }
    }), r.View = o.extend({
        constructor: function(e) {
            o.call(this, e), this[a] && l.on(this[a], this.resize, this)
        },
        resize: function() {
            this.render()
        },
        remove: function() {
            return this[a] && l.off(this[a], this.resize, this), u.call(this)
        }
    }), t(function() {
        var n = l.width();
        t(e).resize(function() {
            var e, t = l.width();
            t !== n && (e = f(n, t, l[a]), e.length && l.trigger(e.join(" "), t, n < t ? "up" : "down"), n = t)
        })
    })
}(this), function(e) {
    "use strict"
    ;
    var t = e.Spatula = {}, n = e.jQuery, r = e._, i = e.Backbone, s = r.isFunction, o = r.map, u = t.whenAll = function(e) {
        return n.when.apply(n, e || [])
    }, a = t.tryCall = function(e, t) {
        var n = e && e[t];
        if (s(n))
            return n.apply(e, r.rest(arguments, 2))
    }, f = {}, l = {}, c = {
        model: !0,
        collection: !0
    }, h = [], p = [], d = function(e, t, n) {
        f[e] = t, l[e] = n
    };
    t.ContainerView = i.View.extend({
        mod: 0,
        transition: function(e) {
            if (e === this.current)
                return e.render();
            var t = this, r=++t.mod, i = t.current, s = u();
            return t.current = e, t.$el.css("minHeight", t.$el.height()), i || t.$el.empty(), t.trigger("begin", e), i && (i.hide && (s = n.when(i.hide(), s)), s.done(function() {
                i.remove()
            })), s = u(o(this._cb, function(t) {
                return t.call(this, e)
            }).concat([a(e, "load"), s])), s.fail(function() {
                var n = Array.prototype.slice.call(arguments);
                t.mod === r && (t.current = null, t.$el.empty(), t.trigger.apply(t, ["error", e].concat(n)))
            }), s = s.pipe(function() {
                if (t.mod === r)
                    return e.render(), t.$el.prepend(e.el), a(e, "ready"), t.trigger("ready", e), a(e, "show")
            }), s.pipe(function() {
                if (t.mod !== r)
                    return n.Deferred().reject();
                t.$el.css("minHeight", ""), t.trigger("end", e)
            })
        },
        loader: function(e) {
            (this._cb || (this._cb = [])).push(e)
        },
        render: function() {
            return a(this.current, "render"), this
        },
        ready: function() {
            return a(this.current, "ready"), this
        }
    }), t.RichView = i.View.extend({
        assets: c,
        load: function(e, t) {
            var n = this;
            return n._load || (n._load = u(o(n.assets, function(e, t) {
                return f[t].call(n, e).done(function() {
                    n.trigger("load:" + t)
                })
            })).done(function() {
                n.trigger("load")
            })), n._load.then(e, t)
        },
        render: function() {
            var e = this, t = e.load();
            return t.state() === "resolved" ? e.safeRender() : e._render || (e._render=!0, t.then(function() {
                e.safeRender()
            })), e
        },
        safeRender: function() {},
        remove: function() {
            var e = this;
            return e.destroy(), e.trigger("destroy"), e.off(), r.each(e.assets, function(e, t) {
                a(l, t, e)
            }), i.View.prototype.remove.apply(e, arguments), e
        },
        destroy: function() {}
    }, {
        assetLoader: d
    }), d("scripts", function(e) {
        return e = r.difference(e, h), u(o(e, function(e) {
            return h.push(e), n.ajax(e, {
                dataType: "script",
                cache: !0,
                error: function() {
                    h = r.without(h, e)
                }
            })
        }))
    }), n(function() {
        h = r.union(h, n("script[src]").not("[async]").map(function() {
            return n(this).attr("src")
        }).get())
    }), d("styles", function(e) {
        return u(o(e, function(e) {
            n("<link/>", {
                "class": "css-" + this.cid,
                rel: "stylesheet",
                type: "text/css",
                href: e
            }).appendTo("head")
        }))
    }, function() {
        n("link.css-" + this.cid).remove()
    }), d("templates", function(e, t) {
        return t || (t = this.templates = {}), u(o(e, function(e, r) {
            return n.ajax(e, {
                cache: !0,
                success: function(e) {
                    t[r] = e
                }
            })
        }))
    }), d("data", function(e) {
        var t = this;
        return n.ajax(e, {
            dataType: "json",
            success: function(e) {
                t.data = e
            }
        })
    }), r.each(c, function(e, t) {
        d(t, function(e) {
            return e = s(e.fetch) ? e : this[t], a(e, "fetch") || u()
        })
    }), t.start = function(s) {
        var o = i.history, u = /^([^\/]+?:\/\/[^\/]+?)(\/[^?#]*)/.exec(e.location.href), a;
        s = r.extend({
            pushState: !0,
            hashChange: !1
        }, s), u && o.start(s) && (u = u[1] + o.options.root, a = u.length, n("body").on("click.spatula", "a[href]", function(e) {
            var n = this.href;
            !e.metaKey&&!e.ctrlKey&&!e.isDefaultPrevented() && n && n.substr(0, a) === u && (n = n.substr(a), t.test(n) && r.any(o.handlers, function(t) {
                if (t.route.test(n))
                    return e.preventDefault(), o.navigate(n, !0), !0
            }))
        }))
    }, t.test = function(e) {
        return !r.any(p, function(t) {
            return t.test(e)
        })
    }, t.exclude = function(e) {
        r.isRegExp(e) || (e = i.Router.prototype._routeToRegExp.call({}, e)), p.push(e)
    }
}(this), function(e) {
    "use strict";
    var t = e.Spatula;
    t.exclude(/\?(?:.*&)?cds(?:[=&]|$)/)
}(this);
if (!jQuery.support.cors && window.XDomainRequest) {
    var httpRegEx = /^https?:\/\//i, getOrPostRegEx = /^get|post$/i, sameSchemeRegEx = new RegExp("^" + location.protocol, "i"), xmlRegEx = /\/xml/i;
    jQuery.ajaxTransport("text html xml json", function(e, t, n) {
        if (e.crossDomain && e.async && getOrPostRegEx.test(e.type) && httpRegEx.test(t.url) && sameSchemeRegEx.test(t.url)) {
            var r = null, i = (t.dataType || "").toLowerCase();
            return {
                send: function(n, s) {
                    r = new XDomainRequest, /^\d+$/.test(t.timeout) && (r.timeout = t.timeout), r.ontimeout = function() {
                        s(500, "timeout")
                    }, r.onload = function() {
                        var e = "Content-Length: " + r.responseText.length + "\r\nContent-Type: " + r.contentType, t = {
                            code: 200,
                            message: "success"
                        }, n = {
                            text: r.responseText
                        };
                        try {
                            if (i === "json")
                                try {
                                    n.json = JSON.parse(r.responseText)
                                } catch (o) {
                                t.code = 500, t.message = "parseerror"
                            } else if (i === "xml" || i !== "text" && xmlRegEx.test(r.contentType)) {
                                var u = new ActiveXObject("Microsoft.XMLDOM");
                                u.async=!1;
                                try {
                                    u.loadXML(r.responseText)
                                } catch (o) {
                                    u = undefined
                                }
                                if (!u ||!u.documentElement || u.getElementsByTagName("parsererror").length)
                                    throw t.code = 500, t.message = "parseerror", "Invalid XML: " + r.responseText;
                                    n.xml = u
                            }
                        } catch (a) {
                            throw a
                        } finally {
                            s(t.code, t.message, n, e)
                        }
                    }, r.onerror = function() {
                        s(500, "error", {
                            text: r.responseText
                        })
                    }, r.open(e.type, e.url), r.send()
                },
                abort: function() {
                    r && r.abort()
                }
            }
        }
    })
}(function(e) {
    "use strict";
    var t = e.Disney, n = e.Spatula, r = e.GOC, i = e.jQuery, s = e._, o = e.Backbone, u = o.sync, a = /^\/(?:[^\/]|$)/, f = t.ajaxBase;
    i(function() {
        var r = t.redirect, o = t.top = new n.ContainerView({
            el: "#main"
        });
        o.on("ready", function(t) {
            var n = t.title;
            i.isFunction(n) && (n = n.call(t)), n && (e.document.title = n), i("html,body").scrollTop(0)
        }), r && o.loader(r.loader), s.defer(n.start)
    }), r && r.queue.push(["go", e.document.body]), i.ajaxSetup({
        timeout: 2e4
    }), f && (o.sync = function(e, t, n) {
        var r = n.url || t && t.url;
        return a.exec(r) && (n = s.extend({
            url: f + r.substr(1)
        }, n)), u(e, t, n)
    })
})(this), function(e, t) {
    "use strict";
    t.support.css3 = function() {
        function t(t, n) {
            var r = e.document.body || e.document.documentElement, i = r.style, s = ["Webkit", "Moz", "ms", "O"];
            if (i[t] !== undefined)
                return n ? t : !0;
            t = t.charAt(0).toUpperCase() + t.substr(1);
            for (var o = 0; o < s.length; o++)
                if (i[s[o] + t] !== undefined)
                    return n ? s[o] + t : !0
        }
        return t
    }(), t.fn.selectTransitionEnd = function() {
        var t, n = e.document.createElement("fakeelement"), r = {
            transition: "transitionend",
            WebkitTransition: "webkitTransitionEnd",
            MozTransition: "transitionend",
            OTransition: "otransitionend"
        };
        for (t in r)
            if (n.style[t] !== undefined)
                return r[t]
    }
}(this, jQuery), function(e) {
    "use strict";
    var t = e.console, n = e.Grill = {}, r = e.Spatula, i = e.jQuery, s = e._, o = e.Backbone, u = r.whenAll, a = o.Model, f = o.Collection, l = o.View, c = r.tryCall, h, p = function(n) {
        var r = e.onerror;
        t && t.error && t.error(n.stack || n.toString()), r && r(n.toString())
    }, d = function(e, t) {
        try {
            return c(e, t)
        } catch (n) {
            p(n)
        }
    }, v = function(e) {
        return s.map(this.views, function(t) {
            return d(t, e)
        })
    }, m = i.support.css3("transition") ? function(t) {
        var n = i.Deferred(), r = i(e).selectTransitionEnd() + ".grill";
        return i(".overlay", t).on(r, function(e) {
            this === e.target && (n.resolve(), t.off(r))
        }), n
    }
    : function() {
        return !0
    }, g = n.Module = a.extend({
        initialize: function() {
            this.data = this.data(this.get("data")), this.on("change:data", function(e, t) {
                e.data.reset(t)
            })
        },
        data: function(e) {
            return new f(e)
        }
    }), y = n.Burger = f.extend({
        model: g,
        grillName: "burger",
        checkGrill: function(e) {
            var t = n[this.grillName];
            if (t)
                return delete n[this.grillName], this.reset(this.parse(t)), e && e(this, t), !0
        },
        fetch: function(e) {
            var t = e && e.success;
            return this.checkGrill(t) ? i.when() : f.prototype.fetch.call(this, e)
        },
        parse: function(e) {
            for (var t in e)
                t in y.prototype || (this[t] = e[t]);
            return this.trigger("parse", e), e.stack
        }
    }), b = l.extend({
        initialize: function(e) {
            this.template = e.template, this.render(), this.model.on("change", this.render, this)
        },
        render: function() {
            var e = this.$el;
            this.setElement(i(this.template.render(this.model))), e.replaceWith(this.$el)
        }
    }), w = n.ListView = l.extend({
        tagName: "ul",
        initialize: function(e) {
            var t = this, n = t.collection;
            t.template = e.template, t.reset(), n.on("add", function(e, n) {
                var r = n.indexOf(e), i = t.spawn(e);
                t.views.splice(r, 0, i), r === 0 ? t.$el.prepend(i.el) : t.views[r - 1].$el.after(i.el)
            }), n.on("remove", function(e, n, r) {
                t.views.splice(r.index, 1)[0].remove()
            }), n.on("reset", t.reset, t)
        },
        reset: function() {
            var e = this;
            return e.views && s.invoke(e.views, "remove"), e.views = e.collection.map(function(t) {
                var n = e.spawn(t);
                return e.$el.append(n.el), n
            }), e
        },
        spawn: function(e) {
            return new b({
                model: e,
                template: this.template
            })
        }
    }), E = {}, S = n.ModuleView = r.RichView.extend({
        assets: {},
        tagName: "section",
        className: "module",
        safeRender: function() {
            var e = this.template;
            e && e.render && this.$el.html(e.render(this.model.toJSON()))
        }
    }, {
        register: function(e, t, n) {
            return E[e] = this.extend(t, n)
        },
        create: function(e, t) {
            var n = E[e] || this, r = new n(t);
            return r.$el.addClass(e).attr("data-module", e), r
        }
    }), x = w.prototype, T = n.BurgerView = w.extend({
        id: "burger",
        tagName: "article",
        className: "hidden",
        subview: S,
        initialize: function(e) {
            x.initialize.call(this, e), i("<div/>", {
                "class": "overlay"
            }).appendTo(this.el)
        },
        load: function() {
            var e = this;
            return this.collection.fetch().pipe(function() {
                return u(v.call(e, "load"))
            })
        },
        spawn: function(e) {
            return S.create(e.get("view"), {
                model: e
            })
        },
        itemOptions: function() {
            return {}
        },
        title: function() {
            return this.collection.title
        },
        url: function() {
            return this.collection.url
        },
        render: function() {
            v.call(this, "render")
        },
        ready: function() {
            u(v.call(this, "ready"))
        },
        show: function() {
            return this.$el.hasClass("hidden") ? (this.$el.removeClass("hidden"), u([m(this.$el)].concat(v.call(this, "show")))) : !0
        },
        hide: function() {
            return this.$el.hasClass("hidden")?!0 : (this.$el.addClass("hidden"), u([m(this.$el)].concat(v.call(this, "hide"))))
        },
        remove: function() {
            v.call(this, "remove"), x.remove.call(this)
        }
    });
    n.BunView = S.extend({
        className: "bun skip-bottom-border",
        tagName: "div"
    }), h = {
        model: y,
        view: T
    }, n.config = function(e) {
        s.extend(h, e)
    }, n.order = function(e, t) {
        var n = s.extend({}, h, t), r = n.model, i = n.view, o = new r;
        o.url = e, n.container.transition(new i({
            collection: o
        }))
    }
}(this), function(e) {
    "use strict";
    new (e.Backbone.Router.extend({
        routes: {
            "*p": "a"
        },
        a: function(t) {
            var n = "/_grill/json/";
            t && (n += t), e.Grill.order(n)
        }
    }))
}(this), function(e) {
    "use strict";
    var t = [], n, r = "unknown", i = e.Disney.age = function(e) {
        t ? t.push(e) : e(r)
    }, s = function() {
        if (t) {
            for (var i = 0; i < t.length; ++i)
                t[i](r);
            e.clearTimeout(n), t = n = null
        }
    };
    i.known = function(e) {
        r = typeof e == "string" && e || "unknown", s()
    }, n = e.setTimeout(s, 5e3)
}(this), function(e) {
    "use strict";
    var t = e.console, n = e.Disney, r = n.Ads = n.Ads || {}, i = e._;
    r.MessageBus = function() {
        function a(e, t) {
            for (var n = 0; n < t.length; n++) {
                var r = t[n];
                if (!r)
                    continue;
                try {
                    r.callback(e)
                } catch (i) {
                    u(i)
                }
            }
        }
        function f(e, t) {
            var n = {
                name: e,
                body: t
            }, i = (r[e] || []).slice(0);
            a(n, i), i = (r["*"] || []).slice(0), a(n, i)
        }
        function l(e, t, n) {
            i.each(o, function(r) {
                if (r !== n)
                    try {
                        r.bus[r.functionName](e, t)
                } catch (i) {}
            })
        }
        var r = {}, s = this, o = {}, u = function(n) {
            var r = e.onerror;
            t && t.error && t.error("MessageBus:: " + n.stack || "MessageBus:: " + n.toString()), r && r("MessageBus:: " + n.toString())
        };
        typeof n.Ads.MessageBus.Connections == "undefined" && (n.Ads.MessageBus.Connections = {}), this.addConnection = function(e, t, r) {
            if (!r) {
                var i = e + "_" + Math.round(Math.random() * 1e6), u = "Disney.Ads.MessageBus.Connections." + i, a = i + "_primarySendMessage", c = i + "_secondarySendMessage", h = function(e, t) {
                    f(e, t), l(e, t, d)
                }, p = {
                    id: e,
                    bus: s,
                    "function": h,
                    functionName: a,
                    uniqueId: i,
                    globalId: u
                };
                n.Ads.MessageBus.Connections[i] = {}, n.Ads.MessageBus.Connections[i][a] = h, s[a] = h;
                var d = {
                    id: e,
                    bus: {
                        sendMessage: t.sendMessage,
                        addMessageListener: t.addMessageListener,
                        removeMessageListener: t.removeMessageListener,
                        addConnection: t.addConnection,
                        removeConnection: t.removeConnection
                    },
                    functionName: c,
                    uniqueId: i,
                    globalId: u
                };
                r = {
                    primary: p,
                    secondary: d
                }, n.Ads.MessageBus.Connections[i].connection = r, o[e] = d;
                try {
                    t.addConnection(e, s, r)
                } catch (v) {}
                d.bus = t
            }
        }, this.removeConnection = function(e) {
            var t = o[e];
            if (!t)
                return;
            var r = n.Ads.MessageBus.Connections[t.uniqueId].connection;
            s[r.primary.functionName] = null, n.Ads.MessageBus.Connections[t.uniqueId] = null, o[e] = null, t && t.bus && t.bus.removeConnection && t.bus.removeConnection(e)
        }, this.sendMessage = function(e, t) {
            f(e, t), l(e, t)
        }, this.addMessageListener = function(e, t) {
            var n = {
                name: e,
                callback: t
            };
            r[e] || (r[e] = []), r[e].push(n)
        }, this.removeMessageListener = function(e, t) {
            var n = r[e] || [];
            for (var i = 0; i < n.length; i++) {
                var s = n[i];
                s && s.name === e && s.callback === t && (n[i] = null)
            }
        }, this.destroy = function() {
            i.each(o, function(e) {
                s.removeConnection(e)
            }), o = null, i.each(r, function(e) {
                s.removeMessageListener(e.name, e.callback)
            }), r = null, s = null
        }
    }
}(this), function(e) {
    "use strict";
    var t = "http://pubads.g.doubleclick.net/gampad/ads?sz=768x432&iu=/7046/video&impl=s&gdfp_req=1&env=vp&output=xml_vast2&unviewed_position_start=1&url=[referrer_url]&correlator=[timestamp]&cmsid=604", n = e.Disney, r = n.Ads = n.Ads || {}, i = e.jQuery, s = e._, o = s.extend, u = function() {}, a = function(e) {
        var t = this, n = function() {
            t.apply(this, arguments)
        };
        return u.prototype = this.prototype, n.prototype = new u, o(n.prototype, e), n.sub = this.sub, n
    }, f = r.AbstractAdPlayer = function(e, n) {
        this.config = {
            adTagUrl: t + r.isMobile ? "&m_ast=vast": ""
        }, e || this.init(e, n)
    };
    f.sub = a, f.prototype = {
        id: null,
        config: null,
        messageBus: null,
        init: function(e, t) {
            this.id = e, i.extend(this.config, t)
        },
        destroy: function() {
            this.messageBus && (this.messageBus.destroy(), this.messageBus = null)
        }
    }
}(this), function(e) {
    "use strict";
    function N() {
        s.ajax(t, {
            dataType: "script",
            cache: !0,
            success: C,
            error: k
        })
    }
    function C() {
        h = e.google.ima, L()
    }
    function k() {
        x.messageBus.sendMessage("adOpportunityEnd")
    }
    function L() {
        A(), O(), y.requestAds(mt())
    }
    function A() {
        var e = s("#" + x.id);
        e.html('<div class="adPlayer"><div class="overlay adControls" style="display:none;"><div class="adCountdown"><h3>' + x.config.translations.advertisement + "</h3></div></div>" + '<div class="overlay playControls" style="display:none; z-index:1;"><div class="playButton"></div></div>' + '<div class="overlay adClickTarget" style="display:none; z-index:2;"/>' + '<video class="playerElement" autoplay="true" width="100%" oncontextmenu="return false;" oncontextmenu="return false;"></video>' + '<div class="adsContainer" style="display:none;"></div></div>')
    }
    function O() {
        M(), D(), H(), I(), R(), j(), wt()
    }
    function M() {
        m = s("#" + x.id + " .adControls")[0]
    }
    function _() {
        m = null
    }
    function D() {
        E = s("#" + x.id + " .playControls"), E.bind("click", z)
    }
    function P() {
        E && (E.unbind("click", z), E = null)
    }
    function H() {
        d = s("#" + x.id + " .adClickTarget")[0]
    }
    function B() {
        d && (d = null)
    }
    function j() {
        y = new h.AdsLoader(g), y.addEventListener(h.AdsManagerLoadedEvent.Type.ADS_MANAGER_LOADED, dt, !1), y.addEventListener(h.AdErrorEvent.Type.AD_ERROR, vt, !1)
    }
    function F() {
        y && (y.removeEventListener(h.AdsManagerLoadedEvent.Type.ADS_MANAGER_LOADED, dt, !1), y.removeEventListener(h.AdErrorEvent.Type.AD_ERROR, vt, !1), y = null)
    }
    function I() {
        T = s("#" + x.id + " .playerElement")[0], T && T instanceof a && (s(T).on("ended", V), s(T).on("timeupdate", W), T.addEventListener("loadeddata", X, !0), T.addEventListener("play", J, !1), T.addEventListener("pause", K, !1), T.addEventListener("stalled", $, !0))
    }
    function q() {
        T && T instanceof a && (s(T).off("ended", V), s(T).off("timeupdate", W), T.removeEventListener("loadeddata", X, !0), T.removeEventListener("play", J, !1), T.removeEventListener("pause", K, !1), T.removeEventListener("stalled", $, !0), T.hasOwnProperty("pause") && T.pause(), T = null)
    }
    function R() {
        var e = s("#" + x.id + " .adsContainer")[0];
        g = new h.AdDisplayContainer(e, T, d)
    }
    function U() {
        g = null
    }
    function z() {
        if (v) {
            v=!1;
            return 
        }
        G(), et(), T.play()
    }
    function W() {
        T.duration > n && m.style.display === "none" ? tt() : T.duration <= n && m.style.display === "inline" && nt();
        var e = Math.round(T.duration - T.currentTime);
        e = isNaN(e) ? 0 : e, s("#" + x.id + " .adCountdown h3").html(x.config.translations.advertisement + ": " + e)
    }
    function X() {
        T.play()
    }
    function V() {
        if (u.ios)
            try {
                T.webkitExitFullScreen()
        } catch (e) {} else 
            o.exitFullscreen ? o.exitFullscreen() : o.mozCancelFullScreen ? o.mozCancelFullScreen() : o.webkitExitFullScreen && o.webkitExitFullScreen()
    }
    function $() {
        T.play()
    }
    function J() {
        x.messageBus.sendMessage("played")
    }
    function K() {
        !u.kindle && p && (T.play(), Y(), Z()), p=!1, x.messageBus.sendMessage("paused")
    }
    function Q() {
        F(), pt(), _(), P(), B(), q(), U(), Et()
    }
    function G() {
        d.style.display = "inline"
    }
    function Y() {
        d.style.display = "none"
    }
    function Z() {
        E[0].style.display = "inline"
    }
    function et() {
        E[0].style.display = "none"
    }
    function tt() {
        m.style.display = "inline"
    }
    function nt() {
        m.style.display = "none"
    }
    function rt() {
        T.hasAttribute("controls") && T.removeAttribute("controls")
    }
    function it() {
        G(), rt(), T.pause()
    }
    function st() {
        w && x.messageBus.sendMessage("adEnd"), x.messageBus.sendMessage("adOpportunityEnd"), Y(), et(), nt()
    }
    function ot() {
        v=!0, p=!0, f(at, 0)
    }
    function ut() {
        T && T.duration > r && (w=!0, x.messageBus.sendMessage("adStart"))
    }
    function at() {
        v=!1
    }
    function ft() {
        it()
    }
    function lt() {
        f(st, 0)
    }
    function ct() {}
    function ht(e) {
        try {
            b = e.getAdsManager(T)
        } catch (t) {}
        b && (b.addEventListener(h.AdEvent.Type.CLICK, ct), b.addEventListener(h.AdEvent.Type.LOADED, ct), b.addEventListener(h.AdEvent.Type.CONTENT_PAUSE_REQUESTED, ct), b.addEventListener(h.AdEvent.Type.CONTENT_RESUME_REQUESTED, ct), b.addEventListener(h.AdErrorEvent.Type.AD_ERROR, ct), b.addEventListener(h.AdEvent.Type.STARTED, ct), b.addEventListener(h.AdEvent.Type.FIRST_QUARTILE, ct), b.addEventListener(h.AdEvent.Type.MIDPOINT, ct), b.addEventListener(h.AdEvent.Type.THIRD_QUARTILE, ct), b.addEventListener(h.AdEvent.Type.COMPLETE, ct), b.addEventListener(h.AdEvent.Type.ALL_ADS_COMPLETED, ct), b.addEventListener(h.AdErrorEvent.Type.AD_ERROR, ct), b.addEventListener(h.AdEvent.Type.STARTED, ut), b.addEventListener(h.AdEvent.Type.CLICK, ot), b.addEventListener(h.AdEvent.Type.CONTENT_PAUSE_REQUESTED, ft), b.addEventListener(h.AdEvent.Type.CONTENT_RESUME_REQUESTED, lt))
    }
    function pt() {
        if (b) {
            b.removeEventListener(h.AdEvent.Type.CLICK, ct), b.removeEventListener(h.AdEvent.Type.LOADED, ct), b.removeEventListener(h.AdEvent.Type.CONTENT_PAUSE_REQUESTED, ct), b.removeEventListener(h.AdEvent.Type.CONTENT_RESUME_REQUESTED, ct), b.removeEventListener(h.AdErrorEvent.Type.AD_ERROR, ct), b.removeEventListener(h.AdEvent.Type.STARTED, ct), b.removeEventListener(h.AdEvent.Type.FIRST_QUARTILE, ct), b.removeEventListener(h.AdEvent.Type.MIDPOINT, ct), b.removeEventListener(h.AdEvent.Type.THIRD_QUARTILE, ct), b.removeEventListener(h.AdEvent.Type.COMPLETE, ct), b.removeEventListener(h.AdEvent.Type.ALL_ADS_COMPLETED, ct), b.removeEventListener(h.AdErrorEvent.Type.AD_ERROR, ct), b.removeEventListener(h.AdEvent.Type.STARTED, ut), b.removeEventListener(h.AdEvent.Type.CLICK, ot), b.removeEventListener(h.AdEvent.Type.CONTENT_PAUSE_REQUESTED, ft), b.removeEventListener(h.AdEvent.Type.CONTENT_RESUME_REQUESTED, lt);
            try {
                b.destroy()
            } catch (e) {}
            b = null
        }
    }
    function dt(e) {
        ht(e);
        if (b)
            try {
                b.init(s(T).width(), s(T).height(), h.ViewMode.NORMAL), b.start()
        } catch (t) {}
    }
    function vt() {
        st()
    }
    function mt() {
        var e = new h.AdsRequest;
        return e.adTagUrl = x.config.adTagUrl, e.adType = "video", e.linearAdSlotWidth = s(T).width(), e.linearAdSlotHeight = s(T).height(), e.nonLinearAdSlotWidth = s(T).width(), e.nonLinearAdSlotHeight = s(T).height(), e
    }
    function gt() {
        T && T.play()
    }
    function yt() {
        T&&!T.paused && (S=!0, T.pause())
    }
    function bt() {
        T && S && (T.play(), S=!1)
    }
    function wt() {
        x.messageBus.addMessageListener("play", gt), x.messageBus.addMessageListener("pause", yt), x.messageBus.addMessageListener("resume", bt)
    }
    function Et() {
        x && x.messageBus && (x.messageBus.removeMessageListener("play", gt), x.messageBus.removeMessageListener("pause", yt), x.messageBus.removeMessageListener("resume", bt))
    }
    var t = "http://s0.2mdn.net/instream/html5/ima3.js", n = 6, r = 2, i = e.Disney, s = e.jQuery, o = e.document, u = e.Modernizr, a = e.HTMLVideoElement, f = e.setTimeout, l = i.Ads = i.Ads || {}, c = l.AbstractAdPlayer, h, p, d, v, m, g, y, b, w, E, S, x, T;
    l.HTMLAdPlayer = c.sub({
        init: function(e, t) {
            if (!e)
                return;
            x = this, c.prototype.init.call(this, e, t), x.messageBus.sendMessage("adOpportunityStart"), N()
        },
        destroy: function() {
            Q(), pt(), F();
            var e = s("#" + x.id);
            e.empty(), x = null, c.prototype.destroy.call(this)
        }
    })
}(this), function(e) {
    "use strict";
    function c() {
        var e = "#" + f.id;
        a[l].ready = h, s(e, {
            src: "//a.dilcdn.com/a/adplayer-fc2dbad8bf4c.swf",
            version: t,
            onFail: p,
            bgcolor: "#000000",
            width: "100%",
            height: "100%",
            wmode: "opaque",
            menu: "false",
            id: f.id + "_playerElement"
        }, {
            adTagUrl: f.config.adTagUrl,
            advertisement: f.config.translations.advertisement,
            skipAds: f.config.translations.skipAds,
            messageBusReady: "Disney.Ads.Players." + l + ".ready"
        })
    }
    function h() {
        f.messageBus.addConnection(l, i.getElementById(f.id + "_playerElement"))
    }
    function p() {
        f.messageBus.sendMessage("adOpportunityEnd")
    }
    var t = [10, 3], n = e.Disney, r = e.jQuery, i = e.document, s = e.flashembed, o = n.Ads = n.Ads || {}, u = o.AbstractAdPlayer, a = o.Players = o.Players || {}, f, l;
    o.FlashAdPlayer = u.sub({
        init: function(e, t) {
            if (!e)
                return;
            f = this, u.prototype.init.call(this, e, t), l = "axxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx".replace(/[xy]/g, function(e) {
                var t = Math.random() * 16 | 0, n = e === "x" ? t: t & 3 | 8;
                return n.toString(16)
            }), a[l] = {}, f.messageBus.sendMessage("adOpportunityStart"), c()
        },
        destroy: function() {
            a[l] = null;
            var e = r("#" + f.id);
            e.empty(), f = null, u.prototype.destroy.call(this)
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Modernizr, r = t.Ads = t.Ads || {}, i = r.FlashAdPlayer, s = r.HTMLAdPlayer, o = e.location.href.indexOf("html5=true")>-1;
    t.Ads.AdPlayer = function(e, r) {
        var u;
        return !n.flash || o ? u = new s : u = new i, u.messageBus = new t.Ads.MessageBus, u.init(e, r), u
    }
}(this), function(e) {
    "use strict";
    function v(e) {
        return s.Deferred(function(n) {
            t.age(function(t) {
                t !== "unknown" && e.setTargeting("age", t), n.resolve()
            })
        }).promise()
    }
    function H() {
        return e.ActiveXObject && typeof u.documentElement.style.opacity != "undefined"
    }
    var t = e.Disney, n = t.Ads = t.Ads || {}, r = e.Modernizr, i = e.Backbone, s = e.jQuery, o = e._, u = e.document, a = o.each, f = o.extend, l, c = e.location.href.indexOf("gameVideoAd=true")>-1, h = function(t) {
        e.googletag.cmd.push(function() {
            t(e.googletag)
        })
    }, p = function() {}, d = function(e) {
        var t = this, n = function() {
            t.apply(this, arguments)
        };
        return p.prototype = this.prototype, n.prototype = new p, f(n.prototype, e), n.sub = this.sub, n
    }, m = function(e, t, n, r) {
        var i = this;
        this.id || (this.id = o.uniqueId("gpt-")), o.isArray(n[0]) || (n = [n]), i._w = n[0][0], i._h = n[0][1], i.name || (i.name = e), i._d = n, h(function(e) {
            i.define(e, t, r)
        })
    };
    m.sub = d, m.prototype = {
        define: function(e, t, n) {
            var r = this._s = e.defineSlot(t, this._d, this.id);
            a(n, function(e, t) {
                r.setTargeting(t, e)
            }), r.addService(e.pubads())
        },
        tag: function() {
            return '<aside class="gpt ' + this.name + '" id="' + this.id + '" style="' + this.style() + '"></aside>'
        },
        style: function() {
            return "width:" + this._w + "px;height:" + this._h + "px"
        },
        ready: function() {
            var e = this;
            e._$ = s("#" + this.id), h(function(t) {
                l(), v(e._s).then(function() {
                    t.display(e.id)
                })
            })
        },
        restore: function() {
            this._$&&!this._$.closest("html").length && s("#" + this.id).remove()
        },
        remove: function() {
            this._$ && this._$.appendTo("body").empty().remove()
        },
        refresh: function() {
            var e = this._s;
            e && h(function(t) {
                t.pubads().refresh([e])
            })
        },
        sizes: function() {
            return o.map(this._d, function(e) {
                return e[0] + "x" + e[1]
            })
        }
    };
    var g = m.sub({
        ready: function() {
            m.prototype.ready.call(this);
            var t = this._$, n = function() {
                s.contains(u.body, t[0]) ? t.children().length ? (t.css("display") === "inline-block" && t.css("width", ""), t.css("height", "")) : e.setTimeout(n, 100) : t = null
            };
            n()
        }
    }), y = "gpt-companion", b = "companion", w, E = m.sub({
        id: y,
        name: b,
        define: function() {},
        ready: function() {
            var e = this;
            h(function() {
                l(), e._s = w
            }), m.prototype.ready.call(this)
        }
    }), S = m.sub({
        name: b
    }), x = E.sub({
        ready: function() {
            var e = this;
            h(function() {
                l(), e._s = w
            }), this._$ = s("#" + this.id)
        },
        refresh: function() {}
    }), T = m.sub({
        name: b,
        refresh: function() {},
        showVideoCompanion: function() {
            this._$.addClass("hide"), this._$.after(_.tag("gameVideoCompanion")), M.gameVideoCompanion.ready()
        },
        hideVideoCompanion: function() {
            this._$.removeClass("hide"), M.gameVideoCompanion && M.gameVideoCompanion.remove()
        }
    }), N = m.sub({
        adPlayed: !1,
        tag: function() {
            return '<aside class="gpt" id="' + this.id + '-container" style="margin:0px auto;' + this.style() + '">' + '<div class="' + this.name + '" id="' + this.id + '" style="' + this.style() + '"></div>' + '<div class="' + this.name + '" id="' + this.id + '-video" style="' + this.style() + ';display:none"></div>' + "</aside>"
        },
        ready: function() {
            var t = this;
            t._$ = s("#" + this.id + "-container"), h(function(n) {
                l(), v(t._s).then(function() {
                    n.display(t.id), c && e.setTimeout(function() {
                        e.disableAutoUnload(), t.videoMode()
                    }, 0)
                })
            })
        },
        remove: function() {
            !this.adPlayed && M && M.gameCompanion && M.gameCompanion.hideVideoCompanion(), t.Ads.setPlayer(null), this.adPlayer && (this.adPlayer.messageBus && this.adPlayer.messageBus.removeMessageListener("adStart", this.onAdStart), this.adPlayer.destroy(), this.adPlayer = null), m.prototype.remove.call(this)
        },
        videoMode: function() {
            s("#" + this.id).hide(), s("#" + this.id + "-video").show(), M && M.gameCompanion && M.gameCompanion.showVideoCompanion();
            var e = {
                adTagUrl: t.Ads.getVideoAdTagUrl(),
                translations: {
                    advertisement: "Advertisement",
                    skip_ad: "Skip Ad"
                }
            };
            this.onAdStart = o.bind(function() {
                this.adPlayed=!0
            }, this), this.adPlayer = new t.Ads.AdPlayer(this.id + "-video", e), this.adPlayer.messageBus.addMessageListener("adStart", this.onAdStart), t.Ads.setPlayer(this.adPlayer)
        }
    }), C, k, L = [300, 60];
    r.mobile ? (k = "gpt_mobile.js", C = {
        sponsor: [m, [270, 80]],
        banner: [m, [320, 50]],
        rectAll: [m, [320, 50]],
        modalInterstitial: [m, [300, 254]]
    }, i.Resize.width() >= 728 ? (C.videoCompanion = [E, L], C.pushdown = [m, [728, 90]], C.rectAll = [m, [300, 250]]) : C.mobile = [m, [320, 50]]) : (k = "gpt.js", C = {
        videoCompanion: [E, L],
        gameCompanion: [T, L],
        gameVideoCompanion: [x, L],
        activitykitCompanion: [S, L],
        gameLoader: [N, [750, 500]],
        sponsor: [m, [270, 80]],
        pushdown: [g, [[970, 90], [970, 66], [728, 90], [970, 250]]],
        rectangle: [m, [300, 250]],
        rectAll: [m, [300, 250]],
        banner: [m, [320, 50]],
        modalInterstitial: [m, [300, 254]]
    });
    var A, O, M, _, D = n.DoubleClickAdContext = function(e) {
        this.adData = e, this.slots = M = {}, this.unit = A = r.mobile && this.adData.config.mobile || this.adData.config.web, this._t = O = this.adData.config.targeting, _ = this, _.adSystem = this.adData.adSystem, _.ageReady = s.Deferred(function(e) {
            t.age(e.resolve)
        }).promise(), _.status = function() {
            return _.ageReady
        }
    };
    D.prototype = {
        noAds: function() {
            return !this.unit
        },
        tag: function(e, t) {
            var n = this.slots[e], r = C[e], i;
            return n ? n.tag() : r ? (i = r[0], r = new i(e, this.unit, r[1], f({}, this._t, t)), this.slots[e] = r, r.tag()) : ""
        },
        showCDSAds: function() {
            return !1
        },
        cds: function() {
            return !!this.adData.cds
        },
        refresh: function() {
            var n = this;
            h(function(r) {
                t.age(function() {
                    e.setTimeout(function() {
                        r.pubads().refresh(o.pluck(n.slots, "_s"))
                    }, 0)
                })
            })
        },
        ready: function() {
            a(this.slots, function(e) {
                e.ready()
            }), this.refresh()
        },
        helpers: function(e) {
            var t = this, n = {};
            return a(C, function(r, i) {
                n[i + "Ad"] = function() {
                    return t.tag(i, f({}, e, this.adTargeting))
                }
            }), n
        },
        setTargeting: function(e, t) {
            this._t[e] = t
        },
        sizes: function() {
            return o.union.apply(o, o.invoke(this.slots, "sizes"))
        }
    }, a(["restore", "remove"], function(e) {
        D.prototype[e] = function() {
            a(this.slots, function(t) {
                t[e]()
            })
        }
    });
    var P = function() {
        return e.location.hostname
    };
    e.googletag = {
        cmd: []
    }, l = o.once(function() {
        var t = e.googletag, n = t.pubads();
        n.setTargeting("hostname", P()), w = t.defineSlot(A, L, y), w.addService(t.companionAds()).addService(n), t.companionAds().setRefreshUnfilledSlots(!0), n.enableAsyncRendering(), n.collapseEmptyDivs(), n.disableInitialLoad(), t.enableServices()
    }), s(function() {
        var e = {
            dataType: "script",
            cache: !0
        };
        s.ajax("//www.googletagservices.com/tag/js/" + k, e)
    }), e.sendJSEvent = function(e) {
        try {
            e === "pauseVideo" ? t.Ads.getPlayer().messageBus.sendMessage("pause") : e === "playVideo" && t.Ads.getPlayer().messageBus.sendMessage("resume")
        } catch (n) {}
    };
    var B, j = function(e) {
        if (u.createEvent) {
            var t = u.createEvent("Event");
            t.initEvent(e, !0, !0), u.dispatchEvent(t)
        }
    }, F = function() {
        u.attachEvent&&!H() ? u.documentElement.fakePlayEvent++ : j("DisneyMediaPlayerPlay")
    }, I = function() {
        u.attachEvent&&!H() ? u.documentElement.fakePauseEvent++ : j("DisneyMediaPlayerStop")
    };
    s("document").ready(function() {
        u.attachEvent&&!H() && (u.documentElement.fakePlayEvent = 0, u.documentElement.fakePauseEvent = 0)
    }), o.extend(n, {
        isMobile: r.mobile,
        setPlayer: function(e) {
            B && B.messageBus && (B.messageBus.removeMessageListener("played", F), B.messageBus.removeMessageListener("paused", I)), B = e, B && B.messageBus && (B.messageBus.addMessageListener("played", F), B.messageBus.addMessageListener("paused", I))
        },
        getPlayer: function() {
            return B
        },
        getAdSlots: function() {
            return M
        },
        getAdContext: function() {
            return _
        },
        getAdUnit: function() {
            return A
        },
        getTargeting: function() {
            return O
        },
        getTargetingString: function() {
            var e = "";
            return O&&!O.hostname && (O.hostname = P()), o.each(O, function(t, n) {
                n && t && (e += n + "=" + t + "&")
            }), e.slice(0, e.length - 1)
        },
        getCompanionSizes: function() {
            return [L]
        },
        getCompanionSizesString: function() {
            var e = t.Ads.getCompanionSizes(), n = "";
            return o.each(e, function(e) {
                e && (n += e[0] + "x" + e[1] + ",")
            }), n.slice(0, n.length - 1)
        },
        getVideoAdTagUrl: function(e) {
            var n = encodeURIComponent(t.Ads.getAdUnit()), i = encodeURIComponent(t.Ads.getCompanionSizesString()), s = encodeURIComponent(t.Ads.getTargetingString()), o = "http://pubads.g.doubleclick.net/gampad/ads?sz=768x432&iu=" + n + "&ciu_szs=" + i + "&impl=s" + "&gdfp_req=1" + "&env=vp" + "&output=xml_vast2" + "&unviewed_position_start=1" + "&url=[referrer_url]" + "&correlator=[timestamp]" + "&cmsid=604" + "&cust_params=" + s;
            return e && e.length > 0 && (o += "&vid=" + e), r.mobile && (o += "&m_ast=vast"), o
        }
    })
}(this), function(e) {
    "use strict";
    function c(e, t, n) {
        var r = e.sizes(), i = '<aside class="gpt ' + e.name + '" id="' + e.id + '" style="' + e.style() + '">' + '<span id="comp_' + r + '" class="_fwph">';
        return i += '<form id="_fw_form_comp_' + r + '" style="display:none" >', i += '<input type="hidden" name="_fw_input_comp_' + r + '" id="_fw_input_comp_' + r + '" value="' + t + '" /></form>', i += '<span id="_fw_container_comp_' + r + n + "></span></span></aside>", i
    }
    var t = e.Disney, n = t.Ads = t.Ads || {}, r = e.Modernizr, i = e.jQuery, s = e._, o = s.each, u = s.pluck, a = s.extend, f = function() {}, l = function(e) {
        var t = this, n = function() {
            t.apply(this, arguments)
        };
        return f.prototype = this.prototype, n.prototype = new f, a(n.prototype, e), n.sub = this.sub, n
    }, h = function(e, t, n) {
        var r = this;
        this.id || (this.id = s.uniqueId("vevo-")), s.isArray(n[0]) || (n = [n]), r._w = s.max(u(n, 0)), r._h = s.max(u(n, 1)), r.name || (r.name = e), r._d = n
    };
    h.sub = l, h.prototype = {
        define: function() {},
        tag: function() {
            return c(this, "ptgt=p&envp=g_iframe_js&slau=" + this.sizes() + "_Companion1&slid=comp_" + this.sizes() + "&cd=" + this._w + "," + this._h + "&flag=+fcai+niic", '"')
        },
        style: function() {
            return "width:" + this._w + "px;height:" + this._h + "px"
        },
        ready: function() {
            var e = this;
            e._$ = i("#" + this.id)
        },
        restore: function() {
            this._$&&!this._$.closest("html").length && i("#" + this.id).remove()
        },
        remove: function() {
            this._$ && this._$.appendTo("body").empty().remove()
        },
        refresh: function() {
            var e = this._s;
            if (e)
                return 
        },
        sizes: function() {
            return s.map(this._d, function(e) {
                return e[0] + "x" + e[1]
            })
        }
    };
    var p = h.sub({
        ready: function() {
            h.prototype.ready.call(this)
        },
        tag: function() {
            return c(this, "ptgt=p&envp=g_iframe_js&slid=comp_" + this.sizes() + "&w=" + this._w + "&h=" + this._h + "&flag=-cmpn", '" class="_fwac"')
        }
    }), d = "vevo-companion", v = "companion", m = h.sub({
        id: d,
        name: v,
        define: function() {},
        ready: function() {
            h.prototype.ready.call(this)
        },
        tag: function() {
            return c(this, "ptgt=p&envp=g_iframe_js&slau=" + this.sizes() + "_Companion1&slid=comp_" + this.sizes() + "&cd=" + this._w + "," + this._h + "&flag=+fcai+niic", '"')
        }
    }), g, y = [300, 60];
    r.mobile ||!r.flash || e.location.href.indexOf("html5=true")>-1 ? g = {} : g = {
        videoCompanion: [m, y],
        pushdown: [p, [[728, 90]]],
        rectangle: [h, [300, 250]]
    };
    var b, w, E, S = n.VevoAdContext = function(e, n) {
        function r() {
            return i.Deferred(function(e) {
                t.age(function(t) {
                    t === "adult" || t === "teen" || t === "unknown" ? w.siteSection = E.config.ads : w.siteSection = E.config.coppa, e.resolve()
                })
            }).promise()
        }
        this.slots = b = {}, w = this, E = e, w.adSystem = E.adSystem, w.siteSection = E.config.coppa, w.ageReady = i.Deferred(function(e) {
            t.age(e.resolve)
        }).promise(), w.status = function() {
            return w.ageReady
        }, w.cds()===!0 || E.disableAds===!0 ? w.siteSection = E.config.noads : r(), n.on("login", r), n.on("logout", r)
    };
    S.prototype = {
        tag: function(e) {
            if (this.cds())
                return "";
            var t = this.slots[e], n = g[e], r;
            return t ? t.tag() : n ? (r = n[0], n = new r(e, "", n[1], a({}, this._t, "")), this.slots[e] = n, n.tag()) : ""
        },
        cds: function() {
            return !!E.cds
        },
        noAds: function() {
            return !1
        },
        showCDSAds: function() {
            return !0
        },
        removeSlot: function(e) {
            b[e].remove()
        },
        helpers: function(e) {
            var t = this, n = {};
            return o(g, function(r, i) {
                n[i + "Ad"] = function() {
                    return t.tag(i, a({}, e, this.adTargeting))
                }
            }), n
        },
        sizes: function() {
            return s.union.apply(s, s.invoke(this.slots, "sizes"))
        }
    }, o(["ready", "restore", "remove", "refresh"], function(e) {
        S.prototype[e] = function() {
            o(this.slots, function(t) {
                t[e]()
            })
        }
    })
}(this), function() {
    function e(e, u) {
        e = e || "", u = u || {};
        for (var a in t)
            t.hasOwnProperty(a) && (u.autoFix && (u["fix_" + a]=!0), u.fix = u.fix || u["fix_" + a]);
        var f = {
            comment: /^\x3c!--/,
            endTag: /^<\//,
            atomicTag: /^<\s*(script|style|noscript|iframe|textarea)[\s>]/i,
            startTag: /^</,
            chars: /^[^<]/
        }, l = {
            comment: function() {
                var t = e.indexOf("-->");
                if (0 <= t)
                    return {
                        content: e.substr(4, t),
                        length: t + 3
                    }
            },
            endTag: function() {
                var t = e.match(r);
                if (t)
                    return {
                        tagName: t[1],
                        length: t[0].length
                    }
            },
            atomicTag: function() {
                var t = l.startTag();
                if (t) {
                    var n = e.slice(t.length);
                    if (n.match(RegExp("</\\s*" + t.tagName + "\\s*>", "i")) && (n = n.match(RegExp("([\\s\\S]*?)</\\s*" + t.tagName + "\\s*>", "i"))))
                        return {
                            tagName: t.tagName,
                            attrs: t.attrs,
                            content: n[1],
                            length: n[0].length + t.length
                        }
                }
            },
            startTag: function() {
                var t = e.match(n);
                if (t) {
                    var r = {};
                    return t[2].replace(i, function(e, t, n, i, o) {
                        e = n || i || o || s.test(t) && t || null, r[t] = e
                    }), {
                        tagName: t[1],
                        attrs: r,
                        unary: !!t[3],
                        length: t[0].length
                    }
                }
            },
            chars: function() {
                var t = e.indexOf("<");
                return {
                    length: 0 <= t ? t: e.length
                }
            }
        }, c = function() {
            for (var t in f)
                if (f[t].test(e)) {
                    o && console.log("suspected " + t);
                    var n = l[t]();
                    return n ? (o && console.log("parsed " + t, n), n.type = n.type || t, n.text = e.substr(0, n.length), e = e.slice(n.length), n) : null
                }
        };
        if (u.fix) {
            var h = /^(AREA|BASE|BASEFONT|BR|COL|FRAME|HR|IMG|INPUT|ISINDEX|LINK|META|PARAM|EMBED)$/i, d = /^(COLGROUP|DD|DT|LI|OPTIONS|P|TD|TFOOT|TH|THEAD|TR)$/i, m = [];
            m.last = function() {
                return this[this.length - 1]
            }, m.lastTagNameEq = function(e) {
                var t = this.last();
                return t && t.tagName && t.tagName.toUpperCase() === e.toUpperCase()
            }, m.containsTagName = function(e) {
                for (var t = 0, n; n = this[t]; t++)
                    if (n.tagName === e)
                        return !0;
                return !1
            };
            var y = function(e) {
                return e && "startTag" === e.type && (e.unary = h.test(e.tagName) || e.unary), e
            }, b = c, E = function() {
                e = "</" + m
                .pop().tagName + ">" + e
            }, S = {
                startTag: function(t) {
                    var n = t.tagName;
                    "TR" === n.toUpperCase() && m.lastTagNameEq("TABLE") ? (e = "<TBODY>" + e, x()) : u.fix_selfClose && d.test(n) && m.containsTagName(n) ? m.lastTagNameEq(n) ? E() : (e = "</" + t.tagName + ">" + e, x()) : t.unary || m.push(t)
                },
                endTag: function(e) {
                    m.last() ? u.fix_tagSoup&&!m.lastTagNameEq(e.tagName) ? E() : m.pop() : u.fix_tagSoup && (b(), x())
                }
            }, x = function() {
                var t = e, n = y(b());
                e = t, n && S[n.type] && S[n.type](n)
            }, c = function() {
                return x(), y(b())
            }
        }
        return {
            append: function(t) {
                e += t
            },
            readToken: c,
            readTokens: function(e) {
                for (var t; (t = c()) && (!e[t.type] ||!1 !== e[t.type](t)););
            },
            clear: function() {
                var t = e;
                return e = "", t
            },
            rest: function() {
                return e
            },
            stack: []
        }
    }
    var t = function() {
        var e = {}, t = this.document.createElement("div");
        return t.innerHTML = "<P><I></P></I>", e.tagSoup = "<P><I></P></I>" !== t.innerHTML, t.innerHTML = "<P><i><P></P></i></P>", e.selfClose = 2 === t.childNodes.length, e
    }(), n = /^<([\-A-Za-z0-9_]+)((?:\s+[\w\-]+(?:\s*=\s*(?:(?:"[^"]*")|(?:'[^']*')|[^>\s]+))?)*)\s*(\/?)>/, r = /^<\/([\-A-Za-z0-9_]+)[^>]*>/, i = /([\-A-Za-z0-9_]+)(?:\s*=\s*(?:(?:"((?:\\.|[^"])*)")|(?:'((?:\\.|[^'])*)')|([^>\s]+)))?/g, s = /^(checked|compact|declare|defer|disabled|ismap|multiple|nohref|noresize|noshade|nowrap|readonly|selected)$/i, o=!1;
    e.supports = t, e.tokenToString = function(e) {
        var t = {
            comment: function(e) {
                return "<--" + e.content + "-->"
            },
            endTag: function(e) {
                return "</" + e.tagName + ">"
            },
            atomicTag: function(e) {
                return console.log(e), t.startTag(e) + e.content + t.endTag(e)
            },
            startTag: function(e) {
                var t = "<" + e.tagName, n;
                for (n in e.attrs)
                    var r = e.attrs[n], t = t + (" " + n + '="' + (r ? r.replace(/(^|[^\\])"/g, '$1\\"') : "") + '"');
                return t + (e.unary ? "/>" : ">")
            },
            chars: function(e) {
                return e.text
            }
        };
        return t[e.type](e)
    }, e.escapeAttributes = function(e) {
        var t = {}, n;
        for (n in e) {
            var r = e[n];
            t[n] = r && r.replace(/(^|[^\\])"/g, '$1\\"')
        }
        return t
    };
    for (var u in t)
        e.browserHasFlaw = e.browserHasFlaw ||!t[u] && u;
    this.htmlParser = e
}(), function() {
    function e() {}
    function t(e, t, n) {
        var r, i = e && e.length || 0;
        for (r = 0; r < i; r++)
            t.call(n, e[r], r)
    }
    function n(e, t, n) {
        for (var r in e)
            e.hasOwnProperty(r) && t.call(n, r, e[r])
    }
    function r(e, t) {
        return n(t, function(t, n) {
            e[t] = n
        }), e
    }
    function i(e) {
        try {
            return o.call(e)
        } catch (n) {
            var r = [];
            return t(e, function(e) {
                r.push(e)
            }), r
        }
    }
    var s = this;
    if (!s.postscribe) {
        var o = Array.prototype.slice, u = function(e, t, n) {
            var r = f + t;
            if (2 === arguments.length)
                return r = e.getAttribute(r), null == r ? r : String(r);
            null != n && "" !== n ? e.setAttribute(r, n) : e.removeAttribute(r)
        }, a = function(e, t) {
            var n = e.ownerDocument;
            r(this, {
                root: e,
                options: t,
                win: n.defaultView || n.parentWindow,
                doc: n,
                parser: s.htmlParser("", {
                    autoFix: !0
                }),
                actuals: [e],
                proxyHistory: "",
                proxyRoot: n.createElement(e.nodeName),
                scriptStack: [],
                writeQueue: []
            }), u(this.proxyRoot, "proxyof", 0)
        }, f = "data-ps-";
        a.prototype.write = function() {
            [].push.apply(this.writeQueue, arguments);
            for (var e; !this.deferredRemote && this.writeQueue.length;)
                e = this.writeQueue.shift(), "function" == typeof e ? this.callFunction(e) : this.writeImpl(e)
        }, a.prototype.callFunction = function(e) {
            var t = {
                type: "function",
                value: e.name || e.toString()
            };
            this.onScriptStart(t), e.call(this.win, this.doc), this.onScriptDone(t)
        }, a.prototype.writeImpl = function(e) {
            this.parser.append(e);
            var t;
            for (e = []; (t = this.parser.readToken())&&!/^script$/i.test(t.tagName);)
                e.push(t);
            this.writeStaticTokens(e), t && this.handleScriptToken(t)
        }, a.prototype.writeStaticTokens = function(e) {
            e = this.buildChunk(e);
            if (e.actual)
                return e.html = this.proxyHistory + e.actual, this.proxyHistory += e.proxy, this.proxyRoot.innerHTML = e.html, this.walkChunk(), e
        }, a.prototype.buildChunk = function(e) {
            var n = this.actuals.length, r = [], i = [], s = [];
            return t(e, function(e) {
                r.push(e.text);
                if (e.attrs) {
                    if (!/^noscript$/i.test(e.tagName)) {
                        var t = n++;
                        i.push(e.text.replace(/(\/?>)/, " " + f + "id=" + t + " $1")), "ps-script" !== e.attrs.id && s.push("atomicTag" === e.type ? "" : "<" + e.tagName + " " + f + "proxyof=" + t + (e.unary ? "/>" : ">"))
                    }
                } else 
                    i.push(e.text), s.push("endTag" === e.type ? e.text : "")
            }), {
                tokens: e,
                raw: r.join(""),
                actual: i.join(""),
                proxy: s.join("")
            }
        }, a.prototype.walkChunk = function() {
            for (var e, t = [this.proxyRoot]; null != (e = t.shift());) {
                var n = 1 === e.nodeType;
                if (!n ||!u(e, "proxyof"))
                    n && (this.actuals[u(e, "id")] = e, u(e, "id", null)), (n = e.parentNode && u(e.parentNode, "proxyof")) && this.actuals[n].appendChild(e);
                t.unshift.apply(t, i(e.childNodes))
            }
        }, a.prototype.handleScriptToken = function(e) {
            var t = this.parser.clear();
            t && this.writeQueue.unshift(t), e.src = e.attrs.src || e.attrs.SRC, e.src && this.scriptStack.length ? this.deferredRemote = e : this.onScriptStart(e);
            var n = this;
            this.writeScriptToken(e, function() {
                n.onScriptDone(e)
            })
        }, a.prototype.onScriptStart = function(e) {
            e.outerWrites = this.writeQueue, this.writeQueue = [], this.scriptStack.unshift(e)
        }, a.prototype.onScriptDone = function(e) {
            e !== this.scriptStack[0] ? this.options.error({
                message: "Bad script nesting or script finished twice"
            }) : (this.scriptStack.shift(), this.write.apply(this, e.outerWrites), !this.scriptStack.length && this.deferredRemote && (this.onScriptStart(this.deferredRemote), this.deferredRemote = null))
        }, a.prototype.writeScriptToken = function(e, t) {
            var n = this.buildScript(e);
            e.src && (n.src = e.src, this.scriptLoadHandler(n, t));
            try {
                this.insertScript(n), e.src || t()
            } catch (r) {
                this.options.error(r), t()
            }
        }, a.prototype.buildScript = function(e) {
            var t = this.doc.createElement(e.tagName);
            return n(e.attrs, function(e, n) {
                t.setAttribute(e, n)
            }), e.content && (t.text = e.content), t
        }, a.prototype.insertScript = function(e) {
            this.writeImpl('<span id="ps-script"/>');
            var t = this.doc.getElementById("ps-script");
            t.parentNode.replaceChild(e, t)
        }, a.prototype.scriptLoadHandler = function(e, t) {
            function n() {
                e = e.onload = e.onreadystatechange = e.onerror = null, t()
            }
            var i = this.options.error;
            r(e, {
                onload: function() {
                    n()
                },
                onreadystatechange: function() {
                    /^(loaded|complete)$/.test(e.readyState) && n()
                },
                onerror: function() {
                    i({
                        message: "remote script failed " + e.src
                    }), n()
                }
            })
        };
        var l, c = function() {
            var e = v.shift();
            e && (e.stream = h.apply(null, e))
        }, h = function(t, n, i) {
            function s(e) {
                e = i.beforeWrite(e), m.write(e), i.afterWrite(e)
            }
            m = new a(t, i), m.id = d++, m.name = i.name || m.id, p.streams[m.name] = m;
            var o = t.ownerDocument, u = {
                write: o.write,
                writeln: o.writeln
            };
            r(o, {
                write: s,
                writeln: function(e) {
                    s(e + "\n")
                }
            });
            var f = m.win.onerror || e;
            return m.win.onerror = function(e, t, n) {
                i.error({
                    msg: e + " - " + t + ":" + n
                }), f.apply(m.win, arguments)
            }, m.write(n, function() {
                r(o, u), m.win.onerror = f, i.done(), m = null, c()
            }), m
        }, p = function(t, r, i) {
            "function" == typeof i && (i = {
                done: i
            });
            var o = i;
            i = {
                done: e,
                error: function(e) {
                    throw e
                },
                beforeWrite: function(e) {
                    return e
                },
                afterWrite: e
            }, o = o || {}, n(i, function(e, t) {
                null == o[e] && (o[e] = t)
            }), i = o, t = /^#/.test(t) ? s.document.getElementById(t.substr(1)) : t.jquery ? t[0] : t;
            var u = [t, r, i];
            return t.postscribe = {
                cancel: function() {
                    u.stream ? u.stream.abort() : u[1] = e
                }
            }, v.push(u), m || c(), t.postscribe
        }, d = 0, v = [], m = null;
        l = r(p, {
            streams: {},
            queue: v,
            WriteStream: a
        }), s.postscribe = l
    }
}(), window.adgroupid == undefined && (window.adgroupid = Math.round(Math.random() * 1e3));
var ADTECH = ADTECH || {};
ADTECH.loadAd = function(e, t, n) {
    var r = document.getElementById(e);
    r.innerHTML = "", postscribe(r, t, {
        done: n
    })
}, function(e) {
    "use strict";
    function c(n, i) {
        var s = "/_ad/tags?placement=" + n + "&site_id=" + t.Ads.getAdContext().adtechSiteId + "&uri=" + e.window.location.pathname;
        r.get(s, i)
    }
    var t = e.Disney, n = t.Ads = t.Ads || {}, r = e.jQuery, i = e._, s = e.ADTECH, o = i.each, u = i.pluck, a = i.extend, f = function() {}, l = function(e) {
        var t = this, n = function() {
            t.apply(this, arguments)
        };
        return f.prototype = this.prototype, n.prototype = new f, a(n.prototype, e), n.sub = this.sub, n
    }, h = function(e, t, n, r) {
        var s = this;
        s.id || (s.id = i.uniqueId("adtech-")), i.isArray(n[0]) || (n = [n]), s._w = i.max(u(n, 0)), s._h = i.max(u(n, 1)), s.name || (s.name = e), s._d = n, s._placement = r
    };
    h.sub = l, h.prototype = {
        define: function() {},
        tag: function() {
            var e = '<div class="adtech"><aside id="' + this.id + '" class="gpt adtech ' + this.name.split("_custom_")[0] + '"></aside></div>';
            return e
        },
        style: function() {
            return "width:" + this._w + "px;height:" + this._h + "px"
        },
        validate_ad: function(t, n) {
            var r = new e.Image;
            r.src = n.attr("src"), r.onload = function() {
                var e = "visible";
                r.width <= 1 && (e = "hidden", t.closest('div[class^="bound"').addClass("noAd")), t.css("visibility", e)
            }
        },
        ready: function() {
            var e = this;
            c(this.placement(), function(t) {
                s.loadAd(e.id, t, function() {
                    var t = r("#" + e.id), n = t.find("img");
                    n.length ? e.validate_ad(t, n) : t.css("visibility", "visible")
                })
            })
        },
        placement: function() {
            return this._placement || (this._placement = {
                "970x250": "mh_leaderboard_top",
                "970x90": "mh_leaderboard_top",
                "728x90": "mh_leaderboard_top",
                "300x50": "mh_leaderboard_top",
                "300x250": "mh_mpu1",
                "300x60": "mh_mpu1"
            }
            [this.size()]), this._placement
        },
        size: function() {
            return this._w + "x" + this._h
        },
        restore: function() {
            this._$&&!this._$.closest("html").length && r("#" + this.id).remove()
        },
        remove: function() {
            this._$ && this._$.appendTo("body").empty().remove()
        },
        refresh: function() {
            var e = this._s;
            if (e)
                return 
        },
        sizes: function() {
            return i.map(this._d, function(e) {
                return e[0] + "x" + e[1]
            })
        }
    };
    var p = h.sub({
        ready: function() {
            h.prototype.ready.call(this)
        }
    }), d = h.sub({
        id: "adtech-companion",
        name: "companion",
        ready: function() {
            h.prototype.ready.call(this)
        }
    }), v = {
        pushdown: [p, [[728, 90]]],
        rectangle: [h, [300, 250]],
        videoCompanion: [d, [300, 250]]
    }, m, g, y, b = n.AdTechAdContext = function(e) {
        this.slots = m = {}, g = this, y = e, g.adSystem = y.adSystem, g.adtechSiteId = y.config.adtechSiteId, g.vastPrerollUrl = "", g.vastPostrollUrl = "", g.prerollReady = r.Deferred(), g.postrollReady = r.Deferred(), g.status = function() {
            return r.when(g.prerollReady, g.postrollReady)
        };
        var t = this;
        c("preroll", function(e) {
            t.vastPrerollUrl = e, t.prerollReady.resolve()
        }), c("postroll", function(e) {
            t.vastPostrollUrl = e, t.postrollReady.resolve()
        })
    };
    b.prototype = {
        hideAd: function(t) {
            return e.screen.width < 400 && t.placement().match(/mh_mpu/) !== null
        },
        tag: function(e) {
            if (this.cds())
                return "";
            var t = this.slots[e], n = v[e], r;
            return t ? t.tag() : n ? (r = n[0], n = new r(e, "", n[1]), this.hideAd(n) ? "" : (this.slots[e] = n, n.tag())) : ""
        },
        cds: function() {
            return !!y.cds
        },
        noAds: function() {
            return !1
        },
        showCDSAds: function() {
            return !0
        },
        createSlot: function(e, t) {
            var n = v[e], r = e + "_custom_" + (Object.keys(this.slots).length + 1), i = new n[0](r, null, n[1], t);
            return this.slots[r] = i, i
        },
        removeSlot: function(e) {
            m[e].remove()
        },
        helpers: function(e) {
            var t = this, n = {};
            return o(v, function(r, i) {
                n[i + "Ad"] = function() {
                    return t.tag(i, a({}, e, this.adTargeting))
                }
            }), n
        },
        sizes: function() {
            return i.union.apply(i, i.invoke(this.slots, "sizes"))
        },
        getVastPrerollUrl: function(e) {
            return g.vastPrerollUrl + ";vidAS=pre_roll;vidRT=VAST;cors=yes;vidRTV=2.0.1;kvdproperty=" + e + ";misc=" + (new Date).getTime()
        },
        getVastPostrollUrl: function(e) {
            return g.vastPostrollUrl + ";vidAS=post_roll;vidRT=VAST;cors=yes;vidRTV=2.0.1;kvdproperty=" + e + ";misc=" + (new Date).getTime()
        }
    }, o(["ready", "restore", "remove", "refresh"], function(e) {
        b.prototype[e] = function() {
            o(this.slots, function(t) {
                t[e]()
            })
        }
    }), i.extend(n, {
        getAdContext: function() {
            return g
        },
        getVastUrl: function(e) {
            return e
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = t.Ads = t.Ads || {}, r = function() {
        return {}
    }, i = e.jQuery, s = n.NoAdContext = function(e) {
        var t = i.Deferred().resolve().promise();
        this.cds = function() {
            return e.cds
        }, this.noAds = function() {
            return e.disableAds
        }, this.status = function() {
            return t
        }
    };
    s.prototype = {
        adSystem: "noads",
        showCDSAds: function() {
            return !1
        },
        helpers: r,
        ready: r,
        restore: r,
        remove: r,
        setTargeting: r,
        tag: function() {
            return ""
        },
        sizes: function() {
            return []
        }
    }
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = t.Ads;
    t.Ads.AdContext = function(e) {
        var t = e.adData;
        switch (t.adSystem) {
        case"dfp":
            return new n.DoubleClickAdContext(t);
        case"vevo":
            return new n.VevoAdContext(t, e.guestContext);
        case"adtech":
            return new n.AdTechAdContext(t);
        default:
            return new n.NoAdContext(t)
        }
    }
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = t.Cookie, r = e.Backbone, i = e._, s = /^([0-9a-f]+):([^:]+)/, o = "dolMttAbn", u = 7, a = function() {
        var e = s.exec(n.get(o) || "");
        return e && {
            id: e[1],
            group: e[2]
        }
    }, f = function(e) {
        n.set(o, e.id + ":" + e.group, {
            domain: n.domain(),
            path: "/",
            expires: u
        })
    }, l = t.abn = i.extend({
        testcase: function() {
            return a()
        },
        parse: function(e) {
            var r = e.abn, s = r && r.cases, o = a(), u, f, l =- 1;
            return s && (f = i.pluck(s, "weight"), u = i.keys(s), n.enabled() && (!o || o.id === r.id) && (o ? l = i.indexOf(u, o.group) : (l = t.randi(f), l >= 0 && (o = {
                id: r.id,
                group: u[l]
            }))), e = s[u[l < 0 ? 0: l]].data, l >= 0 && this.trigger("view", o, e)), e
        }
    }, r.Events);
    l.on("view", f), t.randi = function(t) {
        var n = 0, r, i, s = t.length;
        for (r = s - 1; r>-1; --r)
            n+=+t[r] || 0;
        i = e.Math.random() * n;
        for (r = s - 1; i >= 0 && r>-2; --r)
            i-=+t[r] || 0;
        return r + 1
    }
}(this), function(e, t) {
    "use strict";
    var n = e.Disney, r = e.Backbone, i = e._, s = e.devicePixelRatio >= 1.5;
    s && n.geo && n.geo(function(e, t) {
        i.contains(["dialup", "mobile", "satellite", "wireless"], t) && (s=!1)
    });
    var o = n.Entity = r.Model.extend({
        primaryImageName: function() {
            var e = this.get("type");
            return this.get("primaryImageName") || (i.contains(["Video", "Artist", "Game", "Character"], e) ? "thumb" : i.contains(["Collection"], e) ? "square" : i.contains(["Movie"], e) ? "poster" : i.contains(["Credit", "Gallery"], e) ? "photo" : i.contains(["Hero"], e) ? "minihero" : i.contains(["Show", "Product", "Contest", "Shoutout"], e) ? "logo" : "")
        },
        primaryImage: function() {
            var e = this.primaryImageName();
            return i.isFunction(this[e]) ? this[e]() : this.get(e) || t
        }
    });
    i.each({
        thumb: "//a.dilcdn.com/a/thumb_placeholder-dc025e737a0d.jpeg",
        tthumb: "//a.dilcdn.com/a/thumb_placeholder-dc025e737a0d.jpeg",
        photo: "//a.dilcdn.com/a/logo_placeholder-ea619842fa24.jpeg",
        logo: "//a.dilcdn.com/a/entity_placeholder-6d659f726188.jpeg",
        poster: "//a.dilcdn.com/a/poster_placeholder-6ffdcfa05047.jpeg",
        square: "//a.dilcdn.com/a/logo_placeholder-ea619842fa24.jpeg",
        hero: "//a.dilcdn.com/a/placeholder_hero-36547564f418.jpg",
        minihero: "//a.dilcdn.com/a/placeholder_minihero-46da2100b36f.jpg",
        hero_mobile: "//a.dilcdn.com/a/placeholder_hero_mobile-e9f682fa7427.jpg",
        landing: "//a.dilcdn.com/a/placeholder_landing-fc21e75a7d94.jpg",
        landing_mobile: "//a.dilcdn.com/a/placeholder_landing_mobile-e9f682fa7427.jpg",
        header: t,
        header_mobile: t,
        overlayLeft: t,
        overlayRight: t,
        image: t,
        catalogImage: t,
        promo: t,
        src: t,
        sidekick_desktop: t
    }, function(e, t) {
        o.prototype[t] = function() {
            var n = this.get(t + "2x");
            return s && n || this.get(t) || n || e
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.jQuery, n = e._, r = e.Backbone, i = e.Math, s = r.Collection, o = function(e, t) {
        var n = new Array(e), r;
        if (t !== n[0])
            for (r = 0; r < e; ++r)
                n[r] = t;
        return n
    }, u = function(e) {
        return {
            models: e
        }
    };
    r.LazyCollection = s.extend({
        params: {
            limit: "limit",
            offset: "offset"
        },
        constructor: function(e, t) {
            t || (t = {}), this.count = t.count, this.min = t.min || e && e.length || 1, t.params && (this.params = n.defaults(t.params, this.params)), s.call(this, e, t)
        },
        ensure: function(e) {
            var n = this, r = n.count, s = n.length, o = n._ensure, u = n.params, a = i.min(e - s, n.min), f, l;
            return r && (e = i.min(e, r)), e <= s ? l = t.when() : o ? (l = t.Deferred(), o.always(function() {
                n.ensure(e).then(function() {
                    l.resolveWith(arguments)
                }, function() {
                    l.rejectWith(arguments)
                })
            })) : (f = {}, f[u.limit] = a, f[u.offset] = s, n.trigger("loading", n, !0), l = n._ensure = n.fetch({
                add: !0,
                remove: !1,
                data: f
            }).then(function() {
                n.length < s + a && (n.count = n.length)
            }).always(function() {
                n._ensure = null, n.trigger("loading", n, !1)
            })), l
        },
        remaining: function() {
            var e = this.count;
            return e ? e - this.length : null
        },
        loading: function() {
            return !!this._ensure
        },
        pages: function(e, t) {
            var r = [], s, a, f = t && t.whole, l = t && t.pad, c = t && t.nest, h = t && t.fill;
            e =+ e || 1;
            for (a = 0; a < this.length; a += e)
                r.push(this.models.slice(a, a + e));
            a = r.length - 1, f && a > 0 && r[a].length < e && (--a, r.pop()), l && (a >= 0 && (r[a] = r[a].concat(o(e - r[a].length, h))), s = o(e, h)), c && (r = n.map(r, u), s = u(s));
            if (l) {
                a = i.floor(this.count / e);
                while (r.length < a)
                    r.push(s);
                a = this.count - a * e, a > 0&&!f && (s = o(a, h), r.push(c ? u(s) : s))
            }
            return r
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = e.Math, i = /%%([a-z0-9]+)%%/g, s = function(e, t) {
        return t === "rng" ? "" + r.floor(r.random() * 1e18) : ""
    }, o = t.ping = function(t) {
        if (t) {
            var n = new e.Image;
            n.src = t.toString().replace(i, s)
        }
    };
    n("body").on("click", "a[data-ping]", function() {
        o(this.getAttribute("data-ping"))
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Backbone, r = e._, i = "change:watched", s = function(e, t) {
        t ? this.retain() : this.release()
    }, o = function(e) {
        e !== i && this.trigger.apply(this, arguments)
    }, u = function(e) {
        this.current = e || {}
    }, a = function() {
        this.current = void 0
    }, f = {}, l = ["login", "logout", "refresh", "updateProfile"], c = t.GuestContext = function(e) {
        this._h = e || {}, this.on("login", u), this.on("update", u), this.on("logout", a)
    };
    c.actions = l, r.extend(c.prototype, n.Events, {
        _r: 0,
        retain: function() {
            this._r++===0 && this.trigger(i, this, !0)
        },
        release: function() {
            this._r > 0&&--this._r === 0 && this.trigger(i, this, !1)
        },
        watched: function() {
            return this._r > 0
        },
        spawn: function() {
            var e = new c(f);
            return e._p = this, e.current = this.current, e.on(i, s, this), this.on("all", o, e), e
        },
        disown: function(e) {
            e.off(i, s, this), this.off("all", o, e), e._r > 0 && this.release()
        }
    });
    var h = function() {
        throw "Not implemented"
    };
    r.each(l, function(e) {
        c.prototype[e] = function() {
            return (this._h[e] || h).apply(this, arguments)
        }, f[e] = function() {
            return this._p[e].apply(this._p, arguments)
        }
    })
}(this), function(e, t) {
    "use strict";
    function f() {}
    var n = e.Disney, r = e._, i = e.JSON, s = r.rest, o = {
        query: f,
        provide: f,
        pub: f,
        sub: f
    }, u, a;
    i && e.postMessage && (e.addEventListener ? (u = function(t) {
        e.addEventListener("message", t, !1)
    }, a = function(t) {
        e.removeEventListener("message", t, !1)
    }) : e.attachEvent && (u = function(t) {
        e.attachEvent("onmessage", t)
    }, a = function(t) {
        e.detachEvent("onmessage", t)
    })), n.xdc = function(e, n, f, l) {
        function g(t) {
            var r = h + i.stringify([t].concat(s(arguments)));
            e.postMessage(r, n)
        }
        function y(e) {
            var o, u, a = c.length;
            e.origin === n && typeof e.data == "string" && e.data.substr(0, a) === c && (o = i.parse(e.data.substr(a)), typeof o[0] == "string" && (u = r.has(m, o[0]) && m[o[0]]) && u.apply(t, s(o)))
        }
        var c = "xdc " + f + " ", h = "xdc " + (l || f) + " ", p = {}, d = {}, v = {}, m;
        return u ? (u(y), m = {
            call: function(e, n, i) {
                function a() {
                    if (o++)
                        throw "reply already sent";
                    g("reply", n, r.toArray(arguments))
                }
                var s = r.has(d, e) && d[e], o = 0, u=!1;
                if (s)
                    try {
                        s.apply(t, i.concat([a])), u=!0
                } finally {
                    u || g("error", n, "unexpected remote exception")
                } else 
                    g("error", n, "undefined remote procedure: " + e)
            },
            reply: function(e, n) {
                var i = r.has(p, e) && p[e];
                i && (delete p[e], i.apply(t, n))
            },
            notify: function(e, n) {
                var i = r.has(v, e) && v[e], s, o;
                if (typeof n.length == "number") {
                    if (i)
                        for (s = 0, o = i.length; s < o; ++s)
                            i[s].apply(t, n);
                    e !== "all" && m.notify("all", [e].concat(n))
                }
            },
            error: function(e, t) {
                var n = r.has(p, e) && p[e];
                if (n)
                    throw delete p[e], "" + t
            }
        }, {
            query: function(e) {
                var t = s(arguments), n = t[t.length - 1], i = r.uniqueId();
                n && (p[i] = n, g("call", e, i, r.initial(t)))
            },
            provide: function(e, t) {
                d[e] = t
            },
            pub: function(e) {
                g("notify", e, s(arguments))
            },
            sub: function(e, t) {
                var n = v[e];
                n || (n = v[e] = []), n.push(t)
            },
            close: function() {
                a(y)
            }
        }) : o
    }
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = e._, i = e.document, s = /^([^\/:]+:\/\/[^\/]+)\//;
    t.initGuestConsumer = function(o, u) {
        var a = s.exec(o) || s.exec(e.location.href), f = n.Deferred(), l = {};
        return r.each(t.GuestContext.actions, function(e) {
            l[e] = function() {
                return f.then(function(t) {
                    return n.Deferred(function(n) {
                        t.query(e, function(e, t) {
                            t ? n.reject(t) : n.resolve(e)
                        })
                    })
                })
            }
        }), l = new t.GuestContext(l), f.then(function(e) {
            e.sub("all", function() {
                l.trigger.apply(l, r.toArray(arguments))
            })
        }), a ? function() {
            function c() {
                r.style.visibility = "hidden", r.style.minHeight = "0", r.style.left = "-100%"
            }
            var e, r = i.createElement("div"), s = i.createElement("iframe");
            s.setAttribute("allowtransparency", "true"), s.setAttribute("frameBorder", "0"), s.setAttribute("width", "100%"), s.setAttribute("height", "100%"), s.setAttribute("scrolling", "no"), s.setAttribute("style", "overflow:hidden;display:block;"), s.src = o, r.setAttribute("style", "position:absolute;top:0;bottom:0;width:100%;z-index:10000;"), r.appendChild(s), c(), u = u || i.body, u.appendChild(r), u.style.position = "relative", e = t.xdc(s.contentWindow, a[1], "guest"), e.provide("ready", function(t) {
                t(), f.resolve(e)
            }), l.on("open", function() {
                r.style.visibility = "visible", r.style.minHeight = "2000px", r.style.left = "0", s.focus && s.focus(), n("html,body").scrollTop(0)
            }), l.on("close", c)
        }() : f.reject(), l
    }
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e._, r = e.jQuery;
    t.initEmptyGuestConsumer = function() {
        var e = {};
        return n.each(t.GuestContext.actions, function(t) {
            e[t] = function() {
                return r.Deferred()
            }
        }), new t.GuestContext(e)
    }
}(this), function(e) {
    "use strict";
    var t = e.Disney;
    t.gocGuest = function(e, t) {
        function n(t) {
            e.queue.push(t)
        }
        function r() {
            t.watched() || t.current ? n(["user", !!t.current]) : n(["user", void 0])
        }
        r(), t.on("change:watched", r), t.on("login", r), t.on("logout", r), n(["on", "click:login", function() {
            t.login()
        }
        ]), n(["on", "click:logout", function() {
            t.logout()
        }
        ])
    }
}(this), function(e) {
    "use strict";
    function n(e) {
        t.age.known((e.profile.ageBand || "unknown").toLowerCase())
    }
    function r() {
        t.age.known(null)
    }
    var t = e.Disney;
    t.ageGuest = function(e) {
        e.refresh().fail(r), e.on("login", n), e.on("update", n), e.on("logout", r)
    }
}(this), function(e) {
    "use strict";
    function s() {
        return n.mobile ? "mobile" : "desktop"
    }
    var t = e.Disney, n = e.Modernizr, r = e._, i = e.$;
    t.vis = function(e) {
        return r.reduce(e, function(e, t) {
            var n = t.visibility || {};
            if (!r.contains(n.invisible_device_types, s())) {
                if (!r.isEmpty(n.allowed_regions) ||!r.isEmpty(n.disallowed_regions))
                    t.style.geo_hidden=!0;
                e.push(t)
            }
            return e
        }, [])
    }, i(function() {
        t.top.on("ready", function(e) {
            r.each(e.views, function(e) {
                var n = e.model.get("visibility");
                n && e.$el.hasClass("geo-hidden") && t.geo(function(t) {
                    t&&!r.include(n.disallowed_regions, t) && (r.isEmpty(n.allowed_regions) || r.include(n.allowed_regions, t)) && e.$el.removeClass("geo-hidden").addClass("geo-accepted")
                })
            })
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Grill, n = e.jQuery, r = e.Backbone, i = e.Disney, s = t.BurgerView, o = s.prototype, u = t.Burger, a = u.prototype, f = e.location, l = /[&?]draft_id=([^&]+)/.exec(f.search), c = i.didBase ? i.initGuestConsumer(i.didBase + f.host): i.initEmptyGuestConsumer(), h = t.ValueAdedBurgerView = s.extend({
        initialize: function() {
            this.guest = c.spawn(), s.prototype.initialize.apply(this, arguments)
        },
        spawn: function(e) {
            var n = this.collection, r = this.ads;
            r || (this.ads = r = new i.Ads.AdContext({
                adData: n.adData,
                guestContext: c
            }));
            var s = {
                model: e,
                ads: r,
                guest: this.guest,
                cds: r.cds(),
                noAds: r.noAds()
            };
            return t.ModuleView.create(e.get("view"), s)
        },
        remove: function() {
            var e = n("#nav-pushdown"), t = e.height();
            return c.disown(this.guest), this.ads && (t > 0 && e.css("min-height", t), this.ads.remove()), o.remove.call(this)
        },
        optimizely: function() {
            var e = this.collection.optimizely;
            return e
        },
        cto: function() {
            var e = this.collection.cto, t = this.ads;
            t && t.unit && (e = n.extend({
                adPageName: t.unit,
                adSizeList: t.sizes().join(",")
            }, e));
            var r = i.abn.testcase();
            return r && (e = n.extend({
                abTest: r.id + "|" + r.group
            }, e)), e
        },
        url: function() {
            return this.collection.href
        },
        branding: function() {
            return this.collection.branding
        }
    }), p = /^\/_grill\/json(\/.*)/, d = t.Module.extend({
        data: function(e) {
            var t = new r.LazyCollection(e, {
                model: i.Entity,
                params: {
                    offset: "o",
                    limit: "l"
                },
                count: this.get("count")
            }), s = this.collection.url, o;
            if (o = p.exec(s))
                s = o[1];
            var u = s.indexOf("?") >= 0 ? "&": "?";
            return s = "/_grill/more" + s + u + n.param({
                r: this.get("ref")
            }), l && (s += "&draft_id=" + l[1]), t.url = s, t
        },
        subject: function() {
            return this.data.first()
        }
    }), v = t.MysteryMeat = u.extend({
        model: d,
        parse: function(e) {
            return e = i.abn.parse(e), e.stack = i.vis(e.stack), a.parse.call(this, e)
        }
    });
    e.GOC && i.gocGuest(e.GOC, c), i.ageGuest(c), n(function() {
        var t = i.top;
        e.Grill.config({
            container: t,
            model: v,
            view: h
        }), t.on("ready", function(e) {
            var t = e.ads, r, s = e.collection, o = n("#nav-pushdown").css("min-height", "");
            t && t.helpers && (r = t.helpers(), r.pushdownAd && s && s.adData.pda && o.is(":visible") && o.html(r.pushdownAd()), s && s.ping && i.ping(s.ping), t.ready())
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Spatula, i = e._, s = e.$, o = n.TopperView = n.ValueAdedBurgerView.extend({
        className: "topper",
        id: "topper"
    }), u = n.MysteryTopper = n.MysteryMeat.extend({
        grillName: "topper",
        fetch: function(e) {
            var t = e && e.success;
            return this.checkGrill(t), s.when()
        }
    });
    s(function() {
        var n = t.topper_nav = new r.ContainerView({
            el: "#topper-nav"
        });
        n.on("ready", function(e) {
            var t = e.ads;
            t && t.ready && t.ready(e)
        }), i.defer(function() {
            e.Grill.order("/_grill/json/nav", {
                container: n,
                model: u,
                view: o
            })
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney.Style, n = e.jQuery;
    t.backgroundStyles = function(t, r) {
        var i = n("#nav-local"), s = n("#takeover-area .overlay.module-override"), o = function(e, t, r, i, s, o) {
            var u = t || "#f8f9fa", a = [], f = [{
                prefix: "-moz-",
                direction: "top"
            }, {
                prefix: "-webkit-",
                direction: "top"
            }, {
                prefix: "-o-",
                direction: "top"
            }, {
                prefix: "-ms-",
                direction: "top"
            }, {
                prefix: "",
                direction: "to bottom"
            }
            ], l = "rgba(" + parseInt(e.substring(1, 3), 16) + ", " + parseInt(e.substring(3, 5), 16) + ", " + parseInt(e.substring(5, 7), 16) + ", ", c = l + "1)", h = l + "0)", p = i === "top" ? c: h, d = i === "top" ? h: c;
            return n.each(f, function(e, t) {
                var n = t.prefix + "linear-gradient(" + t.direction + ", " + p + " 0%, ";
                s && (n += p + " " + s + "px, "), o && (n += d + " " + o + ", "), n += d + " " + r + ")", a.push(n)
            }), a.push('progid:DXImageTransform.Microsoft.gradient(startColorstr="#' + e.substring(1) + '", endColorstr="#' + u.substring(1) + '", GradientType=0)'), a
        };
        r ? s = n("#takeover-area .overlay.default") : s.addClass("active");
        if (t) {
            var u = t.takeover, a = t.body_bg;
            if (a) {
                n("#body-bg").css("opacity", 1), a.footerInverted && n("footer,#goc-ft").addClass("inverted"), a.footerDark && n("footer,#goc-ft").removeClass("inverted"), a.color && (n("#body-bg .main").css("background-color", a.color), n("#body-bg .safety-color").css("background-color", a.color));
                if (a.gradient) {
                    a.color = a.color || "#f8f9fa";
                    var f = a.fadeBelow ? n("#takeover-area").height() + 40: 40, l = a.quickFade ? f + 900 + "px": "72%", c = o(a.gradient, a.color, "92%", "bottom", f, l);
                    n.each(c, function(e, t) {
                        var r = t.indexOf("progid")===-1 ? "background-image": "filter";
                        n("#body-bg .color-fade").css(r, t)
                    })
                }
            }
            if (u) {
                u.inverted === undefined && (u.inverted=!0), (!r ||!n("#takeover-area .overlay.module-override").hasClass("active")) && i.toggleClass("inverted", u.inverted);
                var h = u.image, p=!u.color && u.gradient ? "#f8f9fa" : u.color, d = u.gradient, v = u.safety, m = u.bgRepeat, g = u.inverted, y = g ? "light" : "dark";
                s.removeClass("light-nav dark-nav"), s.addClass(y + "-nav"), r && (i.addClass("default-" + y), i.toggleClass("inverted", g)), p ? (s.find(".main-color").css("background-color", p), s.find(".safety-color").css("background-color", p)) : (s.find(".main-color").attr("style", ""), s.find(".safety-color").attr("style", ""));
                if (h) {
                    var b = new e.Image;
                    b.src = h, v && n(b).on("load", function() {
                        s.find(".safety-color .color-block").css({
                            width: b.width,
                            "background-color": v
                        }), b = null
                    }), s.find(".main-image").css("background-image", "url(" + h + ")")
                } else 
                    s.find(".main-image").attr("style", "");
                if (m || d) {
                    if (m) {
                        var w = "url(" + m + ")";
                        s.find(".repeating").css("background-image") !== w && s.find(".repeating").css("background-image", w), v && s.find(".safety-color").css("background-color", v)
                    } else {
                        var E = o(d, u.color, "100%", "top", 216);
                        n.each(E, function(e, t) {
                            var n = t.indexOf("progid")===-1 ? "background-image": "filter";
                            s.find(".repeating").css(n, t)
                        }), s.find(".safety-color").css("background-color", d)
                    }
                    s.find(".repeating").css("opacity") < 1 && s.find(".repeating").css("opacity", 1)
                } else 
                    s.find(".repeating").attr("style", "").css("opacity", 0);
                s.css("opacity", 1)
            }
        } else {
            s.css("opacity", 0);
            var S = i.hasClass("default-light");
            s.removeClass("light-nav dark-nav"), i.toggleClass("inverted", S)
        }
    }
}(this), function(e, t) {
    "use strict";
    var n = e.jQuery, r = e.GOC || {
        queue: []
    };
    n("#nav-e").click(function(e) {
        e.preventDefault(), r.queue.push(["openMenu"])
    }), n(function() {
        var t = e.Backbone.history;
        t.on("route", function() {
            var i = t.options.root + t.getFragment(), s = /^([^\/]+?:\/\/[^\/]+?)\//.exec(e.location.href);
            s && i.substr(0, 1) === "/" && (i = s[1] + i), n("#nav-local li a[href]").each(function() {
                n(this).toggleClass("active", this.href === i)
            }), r.queue.push(["closeMenu"])
        })
    })
}(this), function(e, t) {
    "use strict";
    var n = e.jQuery, r = e.GOC || {
        queue: []
    };
    n("#nav-e").click(function(e) {
        e.preventDefault(), r.queue.push(["openMenu"])
    }), n(function() {
        var t = e.Backbone.history;
        t.on("route", function() {
            var e = n("#nav-local-channels a"), i = t.options.root + t.getFragment();
            e.removeClass("active"), n("#nav-local-channels a[href='" + i + "']").addClass("active"), r.queue.push(["closeMenu"])
        });
        if (n(e).width() >= 1024) {
            var i = n("#nav-local-channels .channel-slider"), s = i.hasClass("channel") ? .09: .045;
            i.swipeasaurus({
                arrows: !0,
                loop: !0,
                right: s
            })
        }
        n(".channel-bg > .pages > li > a").each(function() {
            var e = n(this), t = e.html(), r = t.split(" "), i = '<span class="firstWord">' + r[0] + "</span><br />";
            r.shift(), n.each(r, function(e, t) {
                i += " " + t
            }), e.html(i)
        }), n(".xd-bg > .pages > li > a").each(function() {
            var e = n(this), t = e.html(), r = t.split(" "), i = '<span class="firstWord">' + r[0] + "</span><br />";
            r.shift(), n.each(r, function(e, t) {
                i += " " + t
            }), e.html(i)
        })
    })
}(this);
var Hogan = {};
(function(e, t) {
    "use strict";
    function a(e) {
        return String(e === null || e === undefined ? "" : e)
    }
    function f(e) {
        return e = a(e), u.test(e) ? e.replace(n, "&amp;").replace(r, "&lt;").replace(i, "&gt;").replace(s, "&#39;").replace(o, "&quot;") : e
    }
    e.Template = function(e, n, r, i) {
        this.r = e || this.r, this.c = r, this.options = i, this.text = n || "", this.buf = t ? [] : ""
    }, e.Template.prototype = {
        r: function() {
            return ""
        },
        v: f,
        t: a,
        render: function(t, n, r) {
            return this.ri([t], n || {}, r)
        },
        ri: function(e, t, n) {
            return this.r(e, t, n)
        },
        rp: function(e, t, n, r) {
            var i = n[e];
            return i ? (this.c && typeof i == "string" && (i = this.c.compile(i, this.options)), i.ri(t, n, r)) : ""
        },
        rs: function(e, t, n) {
            var r = e[e.length - 1];
            if (!l(r) && (typeof r != "object" ||!l(r = r.models))) {
                n(e, t, this);
                return 
            }
            for (var i = 0; i < r.length; i++)
                e.push(r[i]), n(e, t, this), e.pop()
        },
        s: function(e, t, n, r, i, s, o) {
            var u;
            return l(e) && e.length === 0?!1 : (typeof e == "function" && (e = this.ls(e, t, n, r, i, s, o)), u = e === ""||!!e, !r && u && t && t.push(typeof e == "object" ? e : t[t.length - 1]), u)
        },
        d: function(e, t, n, r) {
            var i = e.split("."), s = this.f(i[0], t, n, r), o = null;
            if (e === "." && l(t[t.length - 2]))
                return t[t.length - 1];
            for (var u = 1; u < i.length; u++)
                s && typeof s == "object" && (i[u]in s || (s = s.attributes) && typeof s == "object" && i[u]in s) ? (o = s, s = s[i[u]]) : s = "";
            return r&&!s?!1 : (!r && typeof s == "function" && (t.push(o), s = this.lv(s, t, n), t.pop()), s)
        },
        f: function(e, t, n, r) {
            var i=!1, s = null, o=!1;
            for (var u = t.length - 1; u >= 0; u--) {
                s = t[u];
                if (s && typeof s == "object" && (e in s || (s = s.attributes) && typeof s == "object" && e in s)) {
                    i = s[e], o=!0;
                    break
                }
            }
            return o ? (!r && typeof i == "function" && (i = this.lv(i, t, n)), i) : r?!1 : ""
        },
        ho: function(e, t, n, r, i) {
            var s = this.c, o = this.options;
            return o.delimiters = i, r = e.call(t, r), r = r == null ? String(r) : r.toString(), this.b(s.compile(r, o).render(t, n)), !1
        },
        b: t ? function(e) {
            this.buf.push(e)
        }
        : function(e) {
            this.buf += e
        },
        fl: t ? function() {
            var e = this.buf.join("");
            return this.buf = [], e
        }
        : function() {
            var e = this.buf;
            return this.buf = "", e
        },
        ls: function(e, t, n, r, i, s, o) {
            var u = t[t.length - 1], a = null;
            if (!r && this.c && e.length > 0)
                return this.ho(e, u, n, this.text.substring(i, s), o);
            a = e.call(u);
            if (typeof a == "function") {
                if (r)
                    return !0;
                if (this.c)
                    return this.ho(a, u, n, this.text.substring(i, s), o)
            }
            return a
        },
        lv: function(e, t, n) {
            var r = t[t.length - 1], i = e.call(r);
            if (typeof i == "function") {
                i = a(i.call(r));
                if (this.c&&~i.indexOf("{{"))
                    return this.c.compile(i, this.options).render(r, n)
            }
            return a(i)
        }
    };
    var n = /&/g, r = /</g, i = />/g, s = /\'/g, o = /\"/g, u = /[&<>\"\']/, l = Array.isArray || function(e) {
        return Object.prototype.toString.call(e) === "[object Array]"
    }
})(typeof exports != "undefined" ? exports : Hogan), this.Whiskers || function(e) {
    "use strict";
    var t = function(e, t) {
        var n = this, r;
        e = e.split(".");
        while (e.length > 1)
            r = e.shift(), n.hasOwnProperty(r) ? n = n[r] : n = n[r] = {};
        n[e[0]] = t
    }, n = e.Whiskers = function(r, i) {
        t.call(n, r, new e.Hogan.Template(i))
    }
}(this), Whiskers("errors.e404", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<article id="error_page" class="e404 bound"> <div class="inner"> <img src="'), r.b(r.v(r.f("image", e, t, 0))), r.b('"> <div class="message"> <h1>Uh Oh!</h1> <h2>Couldn&rsquo;t find that. <br> Try something else?</h2> <form id="search_404" action="'), r.b(r.v(r.f("search_error", e, t, 0))), r.b('" method="GET"> <input id="search_404_input" name="q" type="text" placeholder="Search..."> </form> </div> </div> </article>'), r.fl()
}), Whiskers("errors.e500", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<article id="error_page" class="e500 bound"> <div class="inner"> <img src="'), r.b(r.v(r.f("image", e, t, 0))), r.b('"> <div class="message"> <h1>Gawrsh!</h1> <h2>Something is not working right.<br> Please try again soon.</h2> </div> </div> </article>'), r.fl()
}), Whiskers("errors.enet", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<article id="error_page" class="enet bound"> <div class="inner"> <img src="'), r.b(r.v(r.f("image", e, t, 0))), r.b('"> <div class="message"> <h1>Aw Phooey!</h1> <h2>The connection was lost. Please try again.</h2> </div> </div> </article>'), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Backbone, r = e.jQuery, i = e.Whiskers, s = ["//a.dilcdn.com/a/e404_matte-22219aa5f291.png", "//a.dilcdn.com/a/e500_matte-09657841c7d6.png", "//a.dilcdn.com/a/enet_matte-c080e9f4b159.png"], o = r("#goc-bar .goc-search").attr("action"), u = n.View.extend({
        render: function() {
            this.el.innerHTML = this.template.render({
                image: this.image,
                search_error: o
            })
        },
        cto: function() {
            return {
                cmsId: "MH",
                account: "wdgdsec",
                categoryCode: "dcom",
                pageName: this.error,
                breadcrumbs: "error",
                pageType: "errorpage",
                ctoSrcPath: "/stat/dolWebAnalytics_DEV.js"
            }
        }
    }), a = u.extend({
        image: s[0],
        search_error: o,
        template: i.errors.e404,
        error: "404"
    }), f = u.extend({
        image: s[1],
        template: i.errors.e500,
        error: "500"
    }), l = u.extend({
        image: s[2],
        template: i.errors.enet,
        error: "network"
    });
    r(function() {
        t.top.on("error", function(e, n, r, i) {
            r === "timeout" ? t.top.transition(new l) : i === "Not Found" ? t.top.transition(new a) : r !== "abort" && t.top.transition(new f)
        }), t.ajaxBase === "/" && e.setTimeout(function() {
            for (var t = 0; t < s.length; ++t)(new e.Image)
                .src = s[t];
            s = null
        }, 4e3)
    })
}(this), function(e) {
    "use strict";
    e.jQuery(function() {
        var t = e.Backbone.history
        , n = 10;
        t.on("route", function() {
            --n < 0 && (t._hasPushState=!1)
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = t.Spinner = {};
    n(function() {
        var i = t.top, s, o = n("#spinner"), u = function() {
            o.fadeIn("fast")
        }, a = function() {
            e.clearTimeout(s), o.hide()
        };
        i.on("begin", function() {
            a(), s = e.setTimeout(u, 1600)
        }).on("ready", a).on("error", a), r = t.Spinner = {
            show: function() {
                u()
            },
            hide: function() {
                a()
            }
        }
    })
}(this), function(e) {
    "use strict";
    e.jQuery(function(t) {
        var n = "title", r = t("body");
        e.Modernizr.touch || r.on("mouseenter", "[title]", function() {
            var e = t(this), r = e.data(n) || e.attr(n);
            e.data(n, r).attr(n, "")
        }).on("mouseleave", "[title]", function() {
            var e = t(this), r = e.data(n);
            r && e.attr(n, r)
        })
    })
}(this), function(e) {
    var t = e.document, n = e.jQuery, r = [84, 69, 65, 77, 32, 77, 65, 84, 84, 69, 82, 72, 79, 82, 78], i = 0, s = function(o) {
        o.keyCode === r[i++] ? i === r.length && (n(t).off("keydown", s), n("body").fadeOut(function() {
            e.location.href = "/its/a/secret/to/everybody/index.html"
        })) : i = 0
    };
    n(t).on("keydown", s)
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = t.Cookie, r = e.jQuery, i = e.navigator, s = e.location.protocol === "https:", o = Array.prototype.slice, u, a = [], f = function(e) {
        a === null ? e(u) : a.push(e)
    }, l = e.cto = {
        trackPagination: function(e) {
            e += "";
            while (e.length < 2)
                e = "0" + e;
            f(function(t) {
                var n = t.pageName;
                n && (t.pageName = n.replace(/\|page_\d+$/, "") + "|page_" + e, t.track())
            })
        }
    };
    r.each(["resetObj", "track", "trackApp", "trackEvent", "trackLink", "trackVideo", "trackGame"], function(e, t) {
        l[t] = function() {
            var e = o.call(arguments), n = e[0];
            n && n.ctoSrcPath && c(n.ctoSrcPath), f(function(n) {
                n[t].apply(n, e)
            })
        }
    });
    var c = function(t) {
        function o() {
            u = new e.CTO;
            var t = 0, n = function() {
                t < a.length ? (a[t](u), ++t, e.setTimeout(n, 200)) : a = null
            };
            n()
        }
        if (i && i.loadPurpose === "preview")
            return;
        if (e.cto.account)
            return;
        var f, l;
        r(e).one("CTOReady SWIDReady", function(e) {
            var t = e.type;
            t === "CTOReady" ? f=!0 : t === "SWIDReady" && (l=!0), f && l && o()
        });
        var c = r.ajax((s ? "https://global" : "http://aglobal") + ".go.com" + t, {
            dataType: "script",
            cache: !0
        });
        !s&&!n.get("SWID") && n.enabled() ? (c = r.when(c, r.Deferred(function(e) {
            r.ajax("http://r.disney.com/swid", {
                dataType: "text"
            }).then(function(t) {
                n.set("SWID", t, {
                    domain: n.domain(),
                    expires: 7300
                }), e.resolve()
            }, function() {
                e.resolve()
            })
        })), c.then(function() {
            r(e).trigger("SWIDReady")
        })) : r(e).trigger("SWIDReady")
    }
}(this), function(e) {
    "use strict";
    var t = e.jQuery, n = e.cto, r = e.Tracker = {
        safe_cto: function(t, r) {
            n && n[t] ? n[t].apply(this, r) : e.console.log("Undefined method: " + t + " on cto")
        },
        page: function(e) {
            var n = {
                pageName: ""
            };
            t.extend(n, e), r.track({
                pageName: n.pageName
            })
        },
        event: function(e) {
            var n = e.eventCategory + ":" + e.action, i = {
                eventCategory: null,
                action: null,
                label: null,
                value: null,
                engagementType: n
            };
            t.extend(i, e), r.trackEvent(i)
        },
        link: function(e) {
            var n = {
                linkName: null,
                linkPosition: null,
                isChrome: !1,
                href: null
            };
            t.extend(n, e), n.isChrome && (n.linkPosition = "Chrome: " + n.linkPosition), r.trackLink(n.linkName, n.linkPosition, undefined, n.href)
        },
        video: function(e) {
            var n = {
                analyticsData: null
            };
            t.extend(n, e), r.trackVideo(n.analyticsData)
        },
        setCto: function(e) {
            n = e
        }
    }, i = ["resetObj", "track", "trackPagination", "trackApp", "trackLink", "trackEvent", "trackVideo"];
    t.map(i, function(e) {
        r[e] = function() {
            r.safe_cto(e, arguments)
        }
    }), t(function() {
        t("#nav-local ul").on("click", "a", null, function(n) {
            var r = t(n.currentTarget), i = t.trim(r.find("img").attr("alt") || r.text()), s = r.attr("href"), o = t("#nav-local a").index(r), u = ["sitenav", o, "none", "link"].join("/"), a = [u, i].join("/");
            e.Tracker.link({
                linkName: a,
                linkPosition: u,
                href: s
            })
        }), t("#nav-local-channels .pages").on("click", "a", null, function(n) {
            var r = t(n.currentTarget), i = t.trim(r.find("a").attr("alt") || r.text()), s = i.replace(/[^a-z0-9 ]/gi, "").replace(/\s+/g, "-"), o = r.attr("href"), u = ["Topnav", "mainlinks"].join("/"), a = [u, "image", s].join("/");
            e.Tracker.link({
                linkName: a,
                linkPosition: u,
                href: o
            })
        }), t("#nav-local-channels .channel-slider-container").on("click", "a", null, function(n) {
            var r = t(n.currentTarget), i = t.trim(r.find("img").attr("alt") || r.text()), s = i.replace(/[^a-z0-9 ]/gi, "").replace(/\s+/g, "-"), o = r.attr("href"), u = ["CharacterNav", "char"].join("/"), a = [u, "image", s].join("/");
            e.Tracker.link({
                linkName: a,
                linkPosition: u,
                href: o
            })
        }), t("footer a").on("click", null, null, function(n) {
            var r = t.trim(t(n.currentTarget).text()), i = t(n.currentTarget).attr("href"), s = t(n.currentTarget).data("module-title");
            s && (r = s + "/" + r), e.Tracker.link({
                linkName: "globalfooter/link/" + r,
                linkPosition: "globalfooter",
                href: i
            })
        }), t.each(["#searchField_SRP", "#searchField_BlogChrome"], function(n, r) {
            t(r).one("input", function() {
                e.cto.trackEvent({
                    searchFlow: r.replace("#", ""),
                    searchEvent: "attempt"
                })
            })
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = e.window, i = e.Tracker;
    n(function() {
        var e = n("footer #utility");
        t.top.on("ready", function(t) {
            n(".select-language-wrapper").length && n(".language-picker").change(function() {
                var e = n(".language-picker option:selected"), t = e.data("url");
                i.link({
                    linkPosition: "globalfooter/lanuage-picker",
                    linkName: "globalfooter/lanuage-picker/" + e.text(),
                    href: t
                }), t && (r.location = t)
            });
            var s = t.branding;
            n.isFunction(s) && (s = s.call(t)), n("footer .branding").first().text(s), n(".bun.search").length && e.addClass("hide-search")
        }), t.portal === "music.disney.com" && n("#goc-ft").addClass("music-footer"), n("#error_page").length && e.addClass("hide-search"), n("footer form.search").on("submit", function() {
            i.link({
                linkName: "globalfooter/searchbox/search_enter",
                linkPosition: "globalfooter/searchbox"
            })
        }), n("footer form.search input").one("input", function() {
            i.event({
                searchAttempt: "y",
                engagementType: "globalfooter",
                searchFlow: "searchField_Footer"
            })
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = t.Cookie;
    n(function() {
        t.top.on("ready", function() {
            r.get("cookies-accepted") !== "true" && (n(".regional-cookie-warning").show(), n("#regional-cookie-accept").click(function() {
                r.set("cookies-accepted", "true", {
                    domain: r.domain(),
                    expires: 365
                }), n(".regional-cookie-warning").hide()
            }))
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Modernizr, r = t.Cookie, i = e.jQuery;
    i(function() {
        t.top.on("ready", function() {
            var e = i("#browser_warning"), t = r.get("accept-browser-warning");
            n.mq("only all") || t !== "true" && (e.addClass("modal-show"), i("#modal_accept").click(function() {
                e.removeClass("modal-show"), r.set("accept-browser-warning", "true", {
                    domain: r.domain(),
                    expires: 1
                })
            }))
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery;
    n(function() {
        t.top.on("begin", function() {
            n("#takeover-area .overlay.module-override").css("opacity", 0), n("#body-bg").css("opacity", 0), n("footer").removeClass("inverted")
        }), t.top.on("ready", function(e) {
            if (e.collection && e.collection.style) {
                var r = {
                    takeover: {
                        image: e.collection.style.takeover && e.collection.style.takeover.image ? e.collection.style.takeover.image: e.collection.style.takeover_image,
                        color: e.collection.style.main_background_color,
                        gradient: e.collection.style.top_gradient_color,
                        bgRepeat: e.collection.style.takeover_repeating_image,
                        safety: e.collection.style.pushdown_safety_color,
                        inverted: e.collection.style.local_chrome_color && e.collection.style.local_chrome_color === "dark"?!1: !0
                    },
                    body_bg: {}
                };
                e.collection.style.background_color || e.collection.style.gradient_color ? (r.body_bg = {
                    fadeBelow: e.collection.style.background_position && e.collection.style.background_position === "below_takeover"?!0: !1,
                    quickFade: e.collection.style.background_transition && e.collection.style.background_transition === "rapid"?!0: !1,
                    footerDark: e.collection.style.footer_text_color && e.collection.style.footer_text_color === "dark"?!0: !1,
                    footerInverted: e.collection.style.footer_text_color && e.collection.style.footer_text_color === "light"?!0: !1
                }, r.body_bg.color = e.collection.style.background_color, r.body_bg.gradient = e.collection.style.gradient_color) : r.body_bg = null, t.Style.backgroundStyles(r, !0)
            }
            t.portal === "princess.disney.com" && n("#burger").addClass("serif"), n(".module.header, .bun.hero_slider, .bun.hero_universal, .bun.minihero, .module.sponsor").next().not(".border-override-true").css("border-top", "none")
        })
    })
}(this), function(e) {
    "use strict";
    e.Disney.FormElements = e.Disney.FormElements || {};
    var t = e.Disney.FormElements, n = e.Whiskers, r = e._, i = e.jQuery;
    i(function() {
        t.dateDropdown = {
            renderTemplate: function(t) {
                var i = t || {}, s = r.extend({
                    month: "Month",
                    day: "Day",
                    year: "Year",
                    month1: "January",
                    month2: "February",
                    month3: "March",
                    month4: "April",
                    month5: "May",
                    month6: "June",
                    month7: "July",
                    month8: "August",
                    month9: "September",
                    month10: "October",
                    month11: "November",
                    month12: "December"
                }, i.translations), o = [], u = [], a = [], f = new e.Date, l = f.getFullYear();
                for (var c = 1; c <= 31; c++)
                    o.push(c);
                for (var h = 0; h < 12; h++)
                    u.push({
                        value: h,
                        month: s["month" + (h + 1)]
                    });
                for (var p = l; p >= l - 100; p--)
                    a.push(p);
                var d = {
                    days: o,
                    months: u,
                    years: a,
                    translations: s
                };
                return n.partials.date_dropdown.render(d)
            }
        }, t.characterCountdown = {
            renderTemplate: function(e) {
                var t = e.maxLength || 140, r = e.tag, i = e.label, s = {
                    maxLength: t,
                    tag: r,
                    label: i
                };
                return n.partials.character_countdown.render(s)
            },
            onReady: function(e, t, n) {
                e.on("keyup", function() {
                    var e = i(this).val(), r = t - e.length;
                    n.html(r)
                })
            }
        }
    })
}(this), Whiskers("shared.search_navigation", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<span id="search-e" role="button" title="Search"></span> <div id="nav-search" class="goc-nav-search"> <div class="overlay"></div> <div class="search-form"> <form class="goc-search" method="GET" action="'), r.b(r.v(r.f("search_action", e, t, 0))), r.b('"> <span id="search-clear" role="button" title="Clear"></span> <input type="text" name="q" class="instant-search-input" placeholder="'), r.b(r.v(r.f("search", e, t, 0))), r.b('" autocomplete="off"/> </form> <span id="search-close" role="button" title="Cancel"></span> </div> </div>'), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Whiskers, r = e.jQuery, i = e.GOC, s = e.Tracker;
    t.ModuleView.register("search_navigation", {
        template: n.shared.search_navigation,
        $navSearch: null,
        $searchClear: null,
        $searchInput: null,
        $searchClose: null,
        $navLocal: null,
        open: function() {
            var e = ["sitenav", r("#nav-local a").length, "none", "link"].join("/");
            s.link({
                linkName: "searchOpen",
                linkPosition: e
            }), this.$navLocal.addClass("search-open")
        },
        close: function() {
            return this.$navLocal.removeClass("search-open"), this.clearForm(!0), !0
        },
        clearForm: function(e) {
            this.$searchClear.removeClass("show"), r("#search-is").remove();
            var t = this.$searchInput.val("").focus();
            e && t.blur()
        },
        initialize: function() {
            var t = this, n = r("#goc-bar .goc-search"), o = n.attr("action"), u = n.find("input[type=text]").attr("placeholder"), a = this.template.render({
                search_action: o,
                search: u
            });
            r("#nav-e + ul").after(a), r("form.goc-search input[type='text']").keyup(function(e) {
                r(e.target).val() === "" && t.clearForm(!1)
            }), t.$navSearch = r("#nav-search"), t.$navLocal = r(".nav-local"), t.$searchClose = r("#search-close"), t.$searchClose.html(r("#search_cancel_text").html()), t.$navSearch.on("click", "a[href]", function() {
                t.close()
            });
            var f = e.setInterval(function() {
                typeof i.ac == "function" && (i.ac(t.$navSearch, "mtt_chrome"), e.clearInterval(f))
            }, 1500);
            r("#search-e").bind("click touchstart", function() {
                return t.open(), !1
            }), t.$searchClear = r("#search-clear"), t.$searchInput = r("#nav-search .search-form form input[type=text]"), t.$searchClose.bind("click touchstart", function() {
                s.link({
                    linkName: "searchClose",
                    searchPhraseInput: t.$searchInput.val()
                })
            }), t.$searchClear.bind("click touchstart", function() {
                s.link({
                    linkName: "searchClear",
                    searchPhraseInput: t.$searchInput.val()
                })
            }), t.$navSearch.find(".overlay, #search-close").bind("click touchstart", function() {
                return t.close(), !1
            }), t.$searchInput.attr("autocomplete", "off").attr("autocorrect", "off"), t.$navSearch.find(".search-form").on(r.fn.selectTransitionEnd(), function() {
                return t.$searchInput.focus(), !1
            }), t.$searchClear.bind("click touchstart", function() {
                t.clearForm(!1)
            }), t.$searchInput.keyup(function() {
                t.$searchClear.addClass("show")
            })
        }
    }), r(function() {
        t.ModuleView.create("search_navigation")
    })
}(this), Whiskers("partials.age_gate_modal", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="age-gate-modal" class="modal-prompt-window"> <div class="loading-status"><span>'), r.b(r.v(r.d("translations.loading", e, t, 0))), r.b('</span></div> <div class="modal-close top-right"></div> <div class="modal-info"> <div class="enter-birthday"> <p class="modal-title">'), r.b(r.v(r.d("translations.enter", e, t, 0))), r.b("</p> <form>"), r.b(r.t(r.f("date_dropdown", e, t, 0))), r.b('</form> <p class="error alert">'), r.b(r.v(r.d("translations.error", e, t, 0))), r.b('</p> <a class="submit-button button blue large">'), r.b(r.v(r.d("translations.submit", e, t, 0))), r.b('</a> <a class="modal-close button" data-title="close">'), r.b(r.v(r.d("translations.nevermind", e, t, 0))), r.b('</a> </div> <div class="too-young"> <p class="modal-title">'), r.b(r.v(r.d("translations.sorry", e, t, 0))), r.b("</p> <p>"), r.b(r.v(r.d("translations.too_young", e, t, 0))), r.b('</p> <a class="button-close button blue small">'), r.b(r.v(r.d("translations.close", e, t, 0))), r.b("</a> </div> </div> "), r.s(r.f("isSocial", e, t, 1), e, t, 0, 717, 1497, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div id="social-buttons"> <p>'), n.b(n.v(n.d("translations.share_this_with_your_friends", e, t, 0))), n.b('</p> <div id="social-logos"> <a class="fb-logo" title="Facebook" data-share-url="'), n.b(n.v(n.f("fbsource", e, t, 0))), n.b('" href="#">Facebook</a> <a class="tw-logo" title="Twitter" target="_blank" href="https://twitter.com/intent/tweet?url='), n.b(n.v(n.f("twtsource", e, t, 0))), n.b("&text="), n.b(n.v(n.f("titleEncoded", e, t, 0))), n.b('">Twitter</a> '), n.s(n.f("thumbEncoded", e, t, 1), e, t, 0, 1069, 1259, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a class="pin-logo" title="Pinterest" target="_blank" href="http://www.pinterest.com/pin/create/button/?url='), n.b(n.v(n.f("pinsource", e, t, 0))), n.b("&media="), n.b(n.v(n.f("thumbEncoded", e, t, 0))), n.b("&description="), n.b(n.v(n.f("titleEncoded", e, t, 0))), n.b('">Pinterest</a> ')
        }), e.pop()), n.b(' </div> </div> <div id="share-links"> <p>'), n.b(n.v(n.d("translations.link", e, t, 0))), n.b('</p> <input type="text" value="'), n.b(n.v(n.f("source", e, t, 0))), n.b('"> '), n.s(n.f("embedCode", e, t, 1), e, t, 0, 1396, 1475, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <p>"), n.b(n.v(n.d("translations.embed", e, t, 0))), n.b('</p> <textarea rows="2"> '), n.b(n.t(n.f("embedCode", e, t, 0))), n.b(" </textarea> ")
        }), e.pop()), n.b(" </div> ")
    }), e.pop()), r.b(' </div> <div class="modal-overlay"></div>'), r.fl()
}), Whiskers("partials.date_dropdown", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="date_dropdown"> <div class="input-group month"> <select name="month" required> <option value="" selected disabled>'), r.b(r.v(r.d("translations.month", e, t, 0))), r.b("</option> "), r.s(r.f("months", e, t, 1), e, t, 0, 169, 215, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <option value="'), n.b(n.v(n.f("value", e, t, 0))), n.b('">'), n.b(n.v(n.f("month", e, t, 0))), n.b("</option> ")
    }), e.pop()), r.b(' </select> </div> <div class="input-group day"> <select name="day" required> <option value="" selected disabled>'), r.b(r.v(r.d("translations.day", e, t, 0))), r.b("</option> "), r.s(r.f("days", e, t, 1), e, t, 0, 377, 415, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <option value="'), n.b(n.v(n.d(".", e, t, 0))), n.b('">'), n.b(n.v(n.d(".", e, t, 0))), n.b("</option> ")
    }), e.pop()), r.b(' </select> </div> <div class="input-group year"> <select name="year" required> <option value="" selected disabled>'), r.b(r.v(r.d("translations.year", e, t, 0))), r.b("</option> "), r.s(r.f("years", e, t, 1), e, t, 0, 579, 617, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <option value="'), n.b(n.v(n.d(".", e, t, 0))), n.b('">'), n.b(n.v(n.d(".", e, t, 0))), n.b("</option> ")
    }), e.pop()), r.b(" </select> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    e.Disney.AgeGate = e.Disney.AgeGate || {};
    var t = e.Disney, n = t.AgeGate, r = e.Whiskers, i = t.Cookie, s = e.Tracker, o = e._, u = e.jQuery;
    n = {
        createAgeGate: function() {
            return null
        },
        getAge: function(e) {
            var t = e.guest.current;
            return new Date(t ? t.profile.dateOfBirth : decodeURI(i.get("age-gate-user")))
        },
        renderModal: function(e, n) {
            var i = n.collection.age_gate_text || {}, s = {
                loading: "Loading",
                enter: "Enter your birthday to continue:",
                submit: "Submit",
                close: "Close",
                error: "Please fill out all fields",
                sorry: "Sorry",
                too_young: "You are not eligible.",
                link: "Link",
                embed: "Embed",
                share_this_with_your_friends: "Share this with your friends",
                month: "Month",
                day: "Day",
                year: "Year"
            };
            o.defaults(i, s);
            var u = o.extend({
                date_dropdown: t.FormElements.dateDropdown.renderTemplate({
                    translations: i
                }),
                translations: i
            }, e);
            return r.partials.age_gate_modal.render(u)
        },
        checkAge: function() {
            var t = u("#age-gate-modal .date_dropdown"), n = t.find(".month option:selected").val(), r = t.find(".day option:selected").val(), i = t.find(".year option:selected").val(), s = new e.Date(i, n, r);
            n === "" || r === "" || i === "" ? this.errorText() : s < this.minBday() ? (this.setCookie(s), this.oldEnough()) : (this.setCookie(s), this.tooYoungText())
        },
        minBday: function() {
            var t = new e.Date;
            return t.setFullYear(t.getFullYear() - this.gatedAge)
        },
        bindModalEvents: function(e) {
            var t = this;
            u("#age-gate-modal .modal-close, #age-gate-modal .button-close, .modal-overlay").on("click", function() {
                t.closeModal()
            }), u("#age-gate-modal .submit-button").on("click", function() {
                t.checkAge()
            }), typeof e == "function" && e()
        },
        closeModal: function() {
            u("#age-gate-modal, #age-gate-modal + .modal-overlay").delay(100).remove(), u("html").removeClass("modal-no-scroll")
        },
        openModal: function() {
            var e = this.renderModal();
            u("#burger-container").append(e), u("#age-gate-modal").addClass("modal-show"), this.bindModalEvents()
        },
        errorText: function() {
            u(".error").delay(400).fadeIn()
        },
        tooYoungText: function() {
            u(".enter-birthday").hide(), u(".too-young").delay(100).fadeIn()
        },
        oldEnough: function() {
            return null
        },
        setCookie: function(e) {
            i.set("age-gate-user", encodeURI(e), {
                domain: i.domain(),
                expires: 1
            })
        }
    };
    var a = o.defaults({
        createAgeGate: function() {
            var e = this;
            u("#burger-container").on("click", ".social-link-btn", function() {
                var n = u(this);
                e.view = t.top.current, e.bday = e.getAge(e.view), e.isSocial = n.hasClass("social-link-btn"), e.gatedLink = n.data("url"), e.source = n.data("source"), e.thumb = n.data("thumb"), e.title = n.data("title"), e.gatedAge = e.isSocial ? 13 : parseInt(n.attr("data-age"), 10), e.openModal(), e.bday && e.bday >= e.minBday() && (e.showDropDown(), e.tooYoungText())
            })
        },
        checkSuccess: function() {
            var e = this.bday < this.minBday();
            e ? this.oldEnough() : this.gatedAge && this.showDropDown()
        },
        showDropDown: function() {
            u("#social-buttons").hide(0, function() {
                u("#age-gate-modal .modal-info").fadeIn()
            })
        },
        openModal: function() {
            var n = this, r = {
                isSocial: this.isSocial,
                source: this.source,
                fbsource: this.source + "?cmp=" + encodeURIComponent("GSL|none|natural|fb||social-share-buttons"),
                twtsource: encodeURIComponent(this.source + "?cmp=" + encodeURIComponent("GSL|none|natural|twt||social-share-buttons")),
                pinsource: encodeURIComponent(this.source + "?cmp=" + encodeURIComponent("GSL|none|natural|pin||social-share-buttons")),
                thumbEncoded: encodeURIComponent(this.thumb),
                titleEncoded: encodeURIComponent(this.title),
                embedCode: this.buildEmbedCode
            };
            u("#burger-container").append(this.renderModal(r, this.view)), u("html").addClass("modal-no-scroll");
            var i = u("#age-gate-modal");
            i.addClass("modal-show"), i.find(".modal-info").hide(), this.bindModalEvents(function() {
                i.find("#social-logos > a").on("click", function(e) {
                    e.preventDefault(), n.$el = u(this), n.checkSuccess()
                }), i.find("input[type=text], textarea").on("click", function() {
                    this.select()
                })
            }), t.FB && t.FB.load(), s && s.event({
                eventCategory: "socialelement",
                action: "share",
                label: e.location.href
            })
        },
        oldEnough: function() {
            var e = this.$el;
            e.hasClass("fb-logo") ? this.shareFacebook(e.data("share-url")) : e.hasClass("pin-logo") ? this.sharePinterest(e.attr("href")) : e.hasClass("tw-logo") && this.shareTwitter(e.attr("href")), this.closeModal()
        },
        shareFacebook: function(e) {
            t.FB.load(function(t) {
                t.ui({
                    method: "share",
                    href: e
                })
            })
        },
        sharePinterest: function(t) {
            e.open(t, "pinterest", "height=440,width=550,resizable=1")
        },
        shareTwitter: function(t) {
            e.open(t, "twitter", "height=440,width=550,resizable=1")
        },
        buildEmbedCode: function() {
            var t = u('div[itemprop="video"]').find('meta[itemprop="embedId"]').attr("content"), n = u(".bun.watch").find('meta[itemprop="title"]').attr("content"), r = "http://" + e.location.host + "/embed/" + t, i = t ? '<iframe width="560" height="315" src="' + r + '" frameborder="0" allowfullscreen scrolling="no"></iframe><br/><a href="' + e.location.href + '">' + n + " on Disney Video</a>": null;
            return i
        }
    }, n), f = o.defaults({
        createAgeGate: function() {
            var e = this;
            u("#burger-container").on("click", ".age-gated", function(n) {
                n.preventDefault();
                var r = u(this), i = t.top.current;
                e.bday = e.getAge(i), e.view = i, e.gatedLink = r, e.gatedAge = parseInt(r.attr("data-age"), 10), e.agePass = e.bday < e.minBday(), e.checkSuccess()
            })
        },
        checkSuccess: function() {
            this.agePass ? this.oldEnough() : this.gatedAge && (this.openModal(), this.bday.getTime() && this.tooYoungText())
        },
        openModal: function() {
            var e = this.renderModal({}, this.view);
            u("#burger-container").append(e), u("#age-gate-modal").addClass("modal-show"), this.bindModalEvents(), u("html").addClass("modal-no-scroll")
        },
        oldEnough: function() {
            this.gatedLink.context.target ? (e.open(this.gatedLink.context.href), this.closeModal()) : e.location = this.gatedLink.context.href
        }
    }, n);
    u(function() {
        f.createAgeGate(), a.createAgeGate()
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = 0, i = /^(?:(?:[^:\/]+:)?\/\/[^\/]+)?(\/[^#]*)/;
    n(function() {
        t.top.on("ready", function() {
            var t = i.exec(e.location.href);
            r++&&n.ajax("/_grill/pv" + (t ? t[1] : "/"), {
                converters: {
                    "*": e.String
                }
            })
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = n.isFunction, i = function(t) {
        var n = e.onerror, r = e.console;
        r && r.error && r.error(t.stack || t.toString()), n && n(t.toString())
    };
    n(function() {
        t.top.on("ready", function(t) {
            var n = e.Tracker, s = {};
            r(t.cto) && (s = t.cto());
            if (s && s.account)
                try {
                    n.resetObj(), n.track(s)
            } catch (o) {
                i(o)
            }
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Tracker, n = e.jQuery, r = "#burger > [data-module]", i = ".slides li", s = function(e) {
        return e != null && (e = n.trim(e).replace(/[^a-z0-9_:, \-]/gi, "").replace(/\s+/g, " ")), e
    }, o = function(e, t, r) {
        var i = e.closest(t);
        return i.length ? n(t, r).index(i) : - 1
    }, u = function(e, t, u, a) {
        var f = [], l = n(r).index(t), c = o(e, ".entity,.item,.col,.product", t), h = o(e, i, t), p = o(e, ".button", e.closest(i + "," + r)), d = o(e, ".all", t), v = s(n.trim(t.find("h1,h2,h3,h4,h5,h6").eq(0).text()));
        return l === 0 && v.indexOf("hero")>-1 && (v = e.closest(i).find(".background img[alt]").attr("alt") || v), f.push(v, "mod" + (l + 1)), c >= 0 ? f.push("item" + (c + 1)) : h >= 0 && (f.push("slide" + (h + 1)), u = "full"), a && f.push(a), p >= 0 ? u = "button" + (p + 1) : o(e, ".poster", t) >= 0 ? u = "poster" : o(e, ".trailer-thumb", t) >= 0 ? u = "trailer-thumb" : o(e, ".gallery-item", t) >= 0 && (u = null), u && f.push(u), d >= 0 && f.push("viewall"), f.join("/")
    }, a = function(e, t, n) {
        var r = o(e, ".suggested", t), i = o(e, ".card .next", t);
        return r >= 0 ? "card/suggested/p" + r + "/" + n : i >= 0 ? "card/next/p" + i + "/" + n : o(e, "#next_container", t) >= 0 ? (n === "link" && (n = "details"), "upnext/" + n) : u(e, t, n)
    }, f = function(e, t, i) {
        var a = s(n.trim(t.find("h1,h2,h3,h4,h5,h6").eq(0).text())), f = n(r).index(t), l = o(e, ".modal-window .arrow-next", t), c = o(e, ".modal-window .arrow-prev", t), h = o(e, ".modal-window .modal-close", t);
        if (l >= 0 || c >= 0 || h >= 0) {
            var p = e.data("viewtype"), d = parseInt(e.data("currentitem"), 10) + 1, v = p === "ad" ? "ad": "item" + d, m = a + "/mod" + (f + 1) + "/" + v + "/modal";
            return m
        }
        return u(e, t, i)
    };
    n("body").on("click", "#burger a[href], #burger .button, #burger .clicon, #burger .launches-modal", function(e) {
        var i = n(this), o = n(e.target), l = o.is("img"), c = i.closest(r), h = s(c.data("module")), p = s(i.closest("[data-title]").data("title") || l && o.attr("alt") || i.text()), d = l ? "thumb": "link", v = s(i.closest("[data-convert]").data("convert")), m, g;
        v === "buy" && (p = v + "_" + p, g = s(i.closest("[data-retailer]").data("retailer")), g && (p = g + "/" + p));
        if (h === "search")
            return;
        if (c.length) {
            var y = "";
            m = (h === "watch" ? a : h === "image_gallery" ? f : u)(i, c, d, v);
            if (h === "songs") {
                if (i.context.className === "like-icon clicon clicked")
                    y = "/like-song", p = "like-song/" + i.data("likeinfo").replace(/\s+/g, "_") + "";
                else if (i.context.className === "like-icon clicon")
                    return;
                if (v === "buy_song") {
                    var b = n(e.currentTarget), w = b.closest(".icon-container").data("buyaylitics").replace(/\s+/g, "_");
                    m = m.replace(/buy_song/, "cart"), p = w, b.data("purchase") && (p = b.data("retailer") + "/" + w)
                }
            }
            t.link({
                linkName: h + "/" + m + "/" + p,
                linkPosition: h + "/" + m + y,
                href: i.attr("href")
            })
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = e._, i = function(e, t, n) {
        var i = this;
        r.each(n, function(e, t) {
            i[t] = function() {
                var t = this, n = arguments;
                t.load(function(r) {
                    e.apply(t, [r].concat(n))
                })
            }
        }), i._u = e, i._c = [], i._i = t
    };
    i.prototype = {
        load: function(e) {
            var t = this;
            return t._p || (t._p = n.ajax(t._u, {
                dataType: "script",
                cache: !0
            }).pipe(function() {
                var e = t._i();
                return r.invoke(t._c, "call", t, e), t._c = null, e
            })), this._p.then(e)
        },
        ready: function(e) {
            this._p ? this._p.then(e) : this._c.push(e)
        }
    }, t.FB = new i("https://connect.facebook.net/en_US/all.js", function() {
        var t = e.FB;
        return t.init({
            appId: n("meta[property='fb:app_id']").attr("content"),
            status: !1
        }), t
    }, {
        go: function(e) {
            e.XFBML.parse()
        }
    }), n(function() {
        n("<div/>", {
            id: "fb-root"
        }).appendTo("body")
    }), e.___gcfg = {
        parsetags: "explicit"
    }, t.P1 = new i("https://apis.google.com/js/plusone.js", function() {
        return e.gapi.plusone
    }, {
        go: function(e) {
            e.go()
        }
    })
}(this), function(e) {
    "use strict";
    var t = function(t) {
        t = "socialelement/" + t, e.Tracker.event({
            eventCategory: t,
            action: "like"
        })
    };
    e.Disney.FB.ready(function(e) {
        e.Event.subscribe("edge.create", function() {
            t("facebook")
        })
    }), e.renderPlusone = function() {
        t("google")
    }
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.jQuery, r = e.Spatula || {};
    n(function() {
        t.top.on("ready", function(t) {
            if (!t ||!n.isFunction(t.optimizely))
                return;
            var i = t.optimizely();
            if (i&&!n.isEmptyObject(i)) {
                r.test = function() {
                    return !1
                };
                if (e.optimizely)
                    e.optimizely.push(["activate"]);
                else {
                    var s = "http://cdn.optimizely.com/js/" + i.project_id + ".js";
                    n.getScript([s], function() {
                        e.optimizely && e.optimizely.push(["activate"])
                    })
                }
            }
        })
    })
}(this), Whiskers("modules.activity_kit_shared", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("kitData", e, t, 1), e, t, 0, 12, 28, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.b(n.rp("kitStyles", e, t, "")), n.b(" ")
    }), e.pop()), r.b(" "), r.s(r.f("data", e, t, 1), e, t, 0, 50, 283, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="content-container '), n.b(n.v(n.f("type", e, t, 0))), n.b("-container "), n.b(n.v(n.f("theme", e, t, 0))), n.b('"> <div class="content-wrapper"> '), n.b(n.rp("container", e, t, "")), n.b(' </div> </div> <div class="content-toolbar-container"> <div id="toolbar" class="inverted"> <div id="meta"></div> </div> </div> ')
    }), e.pop()), r.fl()
}), Whiskers("partials.activity_kit_content", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="stage"> <div id="activity-kit" class="'), r.b(r.v(r.f("view", e, t, 0))), r.b('"></div> <div class="print-footer"> <div class="print-notice"> &copy; 2014 Disney </div> <h1 class="print-title" '), r.s(r.f("kitData", e, t, 1), e, t, 0, 183, 245, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.s(n.f("theme_color", e, t, 1), e, t, 0, 199, 229, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('style="color: '), n.b(n.v(n.f("theme_color", e, t, 0))), n.b('"')
        }), e.pop())
    }), e.pop()), r.b(">"), r.b(r.v(r.f("title", e, t, 0))), r.b("</h1> </div> </div>"), r.fl()
}), Whiskers("partials.activity_kit_theming", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<style type="text/css"> '), r.s(r.f("theme_color", e, t, 1), e, t, 0, 40, 725, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" #activity-kit.avatar .finish .back { color: "), n.b(n.v(n.f("theme_color", e, t, 0))), n.b(";} #activity-kit.avatar .finish .back .icon { background-color: "), n.b(n.v(n.f("theme_color", e, t, 0))), n.b("; } #activity-kit.avatar .finish .print { background-color: "), n.b(n.v(n.f("theme_color", e, t, 0))), n.b("; } #activity-kit.avatar .toolbar .tab.done.active { background-color: "), n.b(n.v(n.f("theme_color", e, t, 0))), n.b("; } #activity-kit.stickerbook .choices, #activity-kit.stickerbook .choices > .progress, #activity-kit.stickerbook .finish, .landscape #activity-kit.stickerbook .toolbar .tabs .tab.active .tab-content, .portrait #activity-kit.stickerbook .toolbar .tabs .tab.active .tab-content { background-color: "), n.b(n.v(n.f("theme_color", e, t, 0))), n.b("; } #activity-kit .print-footer .print-title { color: "), n.b(n.v(n.f("theme_color", e, t, 0))), n.b("; } ")
    }), e.pop()), r.b(" "), r.s(r.f("theme_arrows", e, t, 1), e, t, 0, 759, 1368, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" #activity-kit.avatar .toolbar .eye .next{ background-image: url("), n.b(n.v(n.f("next", e, t, 0))), n.b("); background-size: 96% auto; } #activity-kit.avatar .toolbar .eye .prev{ background-image: url("), n.b(n.v(n.f("prev", e, t, 0))), n.b("); background-size: 96% auto; } #activity-kit.stickerbook .omnislide .next{ background-image: url("), n.b(n.v(n.f("next", e, t, 0))), n.b("); background-size: 50px; } #activity-kit.stickerbook .omnislide .prev{ background-image: url("), n.b(n.v(n.f("prev", e, t, 0))), n.b("); background-size: 50px; } #activity-kit.dressup .omnislide .next{ background-image: url("), n.b(n.v(n.f("next", e, t, 0))), n.b("); background-size: 50px; } #activity-kit.dressup .omnislide .prev{ background-image: url("), n.b(n.v(n.f("prev", e, t, 0))), n.b("); background-size: 50px; } ")
    }), e.pop()), r.b(" </style>"), r.fl()
}), function(e) {
    e.ActivityKit = e.ActivityKit || {}, e.ActivityKit.rule_templates = {
        Background: {
            zIndex: 1,
            required: !0,
            min: 1,
            max: 1,
            selfReplace: !0,
            width: 1,
            height: 1
        },
        CharacterSticker: {
            movable: !0,
            zIndex: 2,
            max: 100
        },
        PropSticker: {
            movable: !0,
            zIndex: 4,
            max: 100
        },
        DressupCharacter: {
            zIndex: 2,
            max: 1,
            movable: !0,
            selfReplace: !0
        }
    }
}(this), function(e) {
    "use strict";
    var t = e.ActivityKit, n = e.Disney, r = e._, i = [], s = e.devicePixelRatio || 1, o = e.document.documentElement.clientWidth, u = function() {
        for (var e = 0; e < i.length; ++e)
            i[e](o);
        u = i = null
    };
    t.hiRes = function(e) {
        i ? i.push(e) : e(o)
    }, s > 1 ? n.geo(function(e) {
        r.contains(["dialup", "mobile", "satellite", "wireless"], e) || (o = s * o), u()
    }) : u()
}(this), function(e) {
    "use strict";
    var t = e.ActivityKit, n = e.jQuery, r = e.Backbone, i = e._, s = r.Model, o = r.Collection, u = function() {}, a = 1024, f, l = function() {
        f = null
    }, c = t.unload = function() {
        f || (f = [], i.defer(l, 1e3)), f.push(this), this.src = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAIAOw=="
    }, h = function(r, s) {
        var o = r.prototype;
        i.each(s, function(e) {
            o[e] = function() {
                var t = this.get(e), n = i.keys(t || {});
                return n.length === 0 ? undefined : t[i.min(n, function(e) {
                    var t = e - a;
                    return t < 0 ? t * t : t
                })]
            }
        }), o.prefetch = i.memoize(function() {
            var r = this, o = n.Deferred(), a = 0, f = s.length, l, h, p, d;
            return f > 0 ? (p = function() {
                this.onload = this.onabort = this.onerror = u, c.call(this), o.notify(e.Math.min(++a, f) / f), a >= f && (l = null, e.clearInterval(h), o.resolve())
            }, d = function(e) {
                e && e.onload === p && e.complete && e.onload()
            }, h = e.setInterval(function() {
                i.each(l, d)
            }, 500), t.hiRes(function() {
                l = i.map(s, function(t) {
                    var n = r[t](), i;
                    return i = new e.Image, i.src = n, i.complete ? p.call(i) : i.onload = i.onabort = i.onerror = p, i
                })
            })) : o.resolve(), o.promise()
        }, function() {
            return this.cid
        })
    }, p = function() {
        return n.when.apply(n, this.map(function(e) {
            return e.prefetch()
        })).pipe(null, null, function() {
            return i.reduce(arguments, function(e, t) {
                return e + (t || 0)
            }, 0) / arguments.length
        })
    }, d = s.extend({});
    t.GroupList = o.extend({
        model: d,
        prefetch: p
    });
    var v = s.extend({
        defaults: {
            zIndex: 0,
            movable: !1,
            min: 0
        },
        applyTo: function(e, t, r) {
            var s = e.ofRule(this), o = s.length, u = this.get("required"), a = this.get("min"), f = this.get("max"), l, c, h, p, d;
            if (o > f) {
                var v = i.first(s.slice(0, o - f));
                l = v.get("x"), c = v.get("y"), h = v.get("width"), p = v.get("height"), d = v.get("rotation"), r && (r.set("x", l), r.set("y", c), r.set("width", h), r.set("height", p), r.set("rotation", d)), e.remove(v)
            } else if (u && o < a)
                e.add(i.map(t.ofRule(this).slice(0, a), function(e) {
                    return e.clone()
                }));
            else if (r && r.creationEvent) {
                var m;
                r.creationEvent.originalEvent ? m = r.creationEvent.originalEvent : m = r.creationEvent;
                var g = {
                    x: (m.x - n(m.target).first("img").offset().left) / n(m.target).first("img").width(),
                    y: (m.y - n(m.target).first("img").offset().top) / n(m.target).first("img").height()
                };
                r.set("x", (n(m.target).offset().left - n(".scene").offset().left) / n(".scene")[0].offsetWidth - .03), r.set("y", (n(m.target).offset().top - n(".scene").offset().top) / n(".scene")[0].offsetHeight - .5), r.set("relLoc", g)
            }
            return e
        }
    });
    t.RuleList = o.extend({
        model: v,
        applyTo: function(e, t, n) {
            return this.each(function(r) {
                r.applyTo(e, t, n)
            }), e
        },
        forLayer: function(e) {
            return this.get(e.get("rule"))
        }
    });
    var m = s.extend({
        defaults: {
            rotation: 0
        },
        initialize: function() {
            var r = this;
            (this.get("x") == null || this.get("y") == null) && this.positionAt(.4, .5), this.get("modifier_rule") === "Background" && this.positionAt(0, 0);
            var i = new e.Image;
            i.src = r.src(), n(i).on("load", function() {
                r.set("natural", {
                    width: i.naturalWidth,
                    height: i.naturalHeight
                });
                if (!t.bgSize && r.get("modifier_rule") === "Background")
                    t.bgSize = r.get("natural");
                else if (t.bgSize && r.get("modifier_rule") !== "Background") {
                    var e = {
                        width: i.naturalWidth / t.bgSize.width,
                        height: i.naturalHeight / t.bgSize.height
                    };
                    if (!r.get("width")) {
                        var n = 1;
                        r.get("modifier_rule") === "PropSticker" ? n = .3 : r.get("modifier_rule") === "CharacterSticker" ? n = .6 : r.get("modifier_rule") === "DressupCharacter" && (n = .8), r.set("width", e.width * n), r.set("height", e.height * n)
                    }
                }
            })
        },
        positionAt: function(e, t) {
            this.set("x", e), this.set("y", t)
        },
        groups: function() {
            var e = this.get("group");
            return this.get("groups") || (e ? [e] : [])
        },
        isRule: function(e) {
            return this.get("rule") === e
        },
        inGroup: function(e) {
            return e.id && (e = e.id), i.contains(this.groups(), e)
        }
    }), g = t.LayerList = o.extend({
        model: m,
        prefetch: p,
        ofRule: function() {
            var e =
            i.map(arguments, function(e) {
                return e.id && (e = e.id), e
            });
            return this.select(function(t) {
                return i.include(e, t.get("rule"))
            })
        },
        ofGroup: function(e) {
            return this.filter(function(t) {
                return t.inGroup(e)
            })
        }
    }), y = s.extend({
        rules: function() {
            var e = this.get("rule");
            return this.get("rules") || (e ? [e] : [])
        },
        findChoices: function(e, t, n) {
            if (this.get("group"))
                return t.toArray();
            var r = new g(n.ofGroup(e));
            return r.ofRule.apply(r, this.rules())
        }
    });
    t.TabList = o.extend({
        model: y,
        prefetch: p
    }), h(d, ["icon"]), h(m, ["src"]), t.hiRes(function(e) {
        a = e
    })
}(this), function(e, t, n) {
    "use strict";
    var r = "swipeasaurus", i = "." + r, s = {
        effect: "slide",
        loop: !0,
        index: 0,
        speed: 300,
        delay: 0
    }, o = Math.abs, u = Math.min, a = e.setTimeout, f = e.clearTimeout, l = ["Webkit", "Moz", "ms", "O"], c = e.document.documentElement, h = function(e) {
        var t, r = 0;
        if (c.style[e] !== n)
            return e;
        t = e.charAt(0).toUpperCase() + e.substr(1);
        for (; r < l.length; r++)
            if (c.style[l[r] + t] !== n)
                return l[r] + t
    }, p=!!h("perspective"), d=!!h("transition"), v=!!h("transform"), m = (h("transform") || "transform").replace(/([A-Z])/g, function(e, t) {
        return "-" + t.toLowerCase()
    }).replace(/^ms-/, "-ms-"), g = function() {
        return (new Date).getTime()
    }, y = function(e, n) {
        return t("<span/>", {
            "class": e,
            title: n,
            role: "button",
            onclick: ""
        })
    }, b = {
        slide: {
            c: {
                overflow: "hidden"
            },
            s: {
                opacity: 1,
                transitionProperty: m + ", visibility"
            },
            a: d && v ? function(e, t, n) {
                n = n ? n + "ms" : "0ms";
                var r = o(t) < 2;
                e.css({
                    transitionDuration: n + ", 0ms",
                    transitionDelay: "0ms, " + (r ? "0ms" : n),
                    visibility: r ? "visible": "hidden",
                    transform: p ? "translate3d(" + t * 100 + "%,0,0)": "translateX(" + t * 100 + "%)"
                })
            }
            : function(e, t, n) {
                var r = this, i = function() {
                    e.css("left", (t * (1 - r._r - r._l) + r._l) * 100 + "%")
                }, s = e.width();
                n && s ? e.stop().css("left", e.position().left).animate({
                    left: t * s + "px"
                }, n, i) : i()
            }
        },
        fade: {
            c: {
                overflow: ""
            },
            s: {
                transform: "translate3d(0,0,0)",
                transitionProperty: "opacity, visibility"
            },
            a: d ? function(e, t, n) {
                n = n ? n + "ms" : "0";
                var r = o(t), i = r < 1, s = r < .5;
                e.css({
                    transitionDuration: n + ", 0ms",
                    transitionDelay: "0ms, " + (i ? "0ms" : n),
                    visibility: i ? "visible": "hidden",
                    opacity: 1 - u(1, r),
                    zIndex: s ? 1: 0
                })
            }
            : function(e, t, n) {
                var r = o(t), i = r < 1;
                i && e.css({
                    visibility: "visible"
                }), e.stop().animate({
                    opacity: 1 - u(1, r)
                }, n, function() {
                    i || e.css({
                        visibility: "hidden"
                    })
                })
            }
        }
    }, w = {
        scroll: function() {
            t(this.$el).scrollLeft(0)
        }
    }, E = function(e, t) {
        var n = e[t] || e.peek;
        return n > 0 && n < .5 ? n : 0
    }, S = function(e, n) {
        var r = this, s, o = r.$slides = e.children(), u = r.length = o.length, a = r._w=!!n.loop, f = r._l = E(n, "left"), l = r._r = E(n, "right"), c = 1 - f - l;
        r._s =+ n.speed, r.index = n.index, r.$el = e, t.each(w, function(t, s) {
            var o = e;
            t.substr(0, 5) === "mouse" && n.arrows && (o = e.parent()), o.on(t + i, function(e) {
                s.call(r, e)
            })
        }), o.on("focusin", function() {
            r.slide(t(this).index())
        }), n.dots && u > 1 && (s = o.map(function(e) {
            return y("dot", e)[0]
        }), s.on("click", function(e) {
            var n = t(e.currentTarget).index();
            r.slide(n)
        }), e.on("slide", function(e, t) {
            s.removeClass("active").eq(t).addClass("active")
        }), e.after(t("<div/>", {
            "class": "dots",
            "aria-hidden": "true"
        }).append(s))), t.each({
            prev: ["Previous", 0],
            next: ["Next", u - 1]
        }, function(i, s) {
            var o;
            n[i] ? o = t(n[i]) : n.arrows && (o = e.siblings("." + i), !o.length && u > 1 && e.after(o = y(i, s[0]))), o && (o.on("click", t.proxy(r[i], r)), a || e.on("slide", function(e, t) {
                o.toggleClass("disabled", t === s[1])
            }))
        }), e.css({
            zIndex: 1,
            position: "relative"
        }), o.each(function(e) {
            t(this).css({
                display: "block",
                position: e === 0 ? "relative": "absolute",
                width: 100 * c + "%",
                left: 100 * f + "%",
                top: 0
            })
        }), r.effect(n.effect), r.start(n.delay)
    };
    S.prototype = {
        _c: 0,
        effect: function(e) {
            var n = b[e] || s.effect;
            this._a = function(e, r) {
                var i = 0, s = this.length, o = this.$slides;
                for (; i < s; ++i)
                    n.a.call(this, t(o[i]), i - e, r)
            }, this.$el.css(n.c), this.$slides.css(n.s), this.slide(this.index, !0)
        },
        slide: function(e, t) {
            var n = this, r = t ? 0: n._s;
            e < 0 && (e = 0), e >= n.length && (e = n.length - 1);
            if (!(n._c++) || n.index !== e)
                n.$el.trigger("slide", n.index = e), f(n._t), n._t = a(function() {
                    n.$el.trigger("slideEnd", e)
                }, r);
            n._a(n.index, r), n.resume()
        },
        prev: function() {
            var e = this.index;
            e === 0 ? this._w && this.slide(this.length - 1) : this.slide(e - 1)
        },
        next: function() {
            var e = this.index;
            e === this.length - 1 ? this._w && this.slide(0) : this.slide(e + 1)
        },
        resume: function() {
            var e = this;
            e.pause(), e._d > 0 && (e._i = a(function() {
                t(e.$el).data(r) === e && e.next()
            }, e._d + e._s))
        },
        pause: function() {
            f(this._i)
        },
        start: function(e) {
            this._d = e || this._d, this.resume()
        },
        stop: function() {
            this._d = 0, this.pause()
        }
    }, t.extend(w, e.Modernizr.touch ? {
        touchstart: function(e) {
            var t = e.originalEvent.touches;
            if (t.length > 1)
                return w.touchcancel.call(this, e);
            this.touch = {
                x: t[0].pageX,
                y: t[0].pageY,
                delta: 0,
                dist: 0,
                test: !0,
                time: g(),
                width: this.$el.width()
            }
        },
        touchmove: function(e) {
            var t = this.touch, n = e.originalEvent.touches, r = this.index, i = this.length, s, u;
            if (t) {
                s = n[0].pageX - t.x, t.dist += o(s - t.delta), t.delta = s;
                if (t.test) {
                    u = n[0].pageY - t.y, t.test=!1;
                    if (o(s) < o(u))
                        return w.touchcancel.call(this, e)
                    }
                e.preventDefault(), this.pause();
                if (r === 0 && s > 0 || r === i - 1 && s < 0)
                    s/=o(s) / t.width + 1;
                this._a(r - s / t.width)
            }
        },
        touchcancel: function() {
            this.touch = null, this.slide(this.index)
        },
        touchend: function(e) {
            var t = this.touch, n = 0;
            if (t) {
                if (o(t.delta) * 2 > t.width || g() - t.time < 250 && t.dist > 20)
                    n = t.delta < 0 ? 1 : - 1;
                t.dist > 20 && (e.preventDefault(), this.bust = g()), this.touch = null, this.slide(this.index + n)
            }
        },
        click: function(e) {
            this.bust && g() - this.bust < 250 && e.preventDefault()
        }
    } : {
        mouseenter: function() {
            this.pause()
        },
        mouseleave: function() {
            this.resume()
        }
    }), t.fn[r] = function(e) {
        var n = arguments;
        return this.each(function() {
            var i = t(this), o;
            typeof e == "string" ? (o = i.data(r), o[e].apply(o, Array.prototype.slice.call(n, 1))) : (e = t.extend({}, s, e), i.data(r, new S(i, e)))
        })
    }
}(this, jQuery), function(e) {
    "use strict";
    var t = e.ActivityKit, n = e.jQuery, r = e.Backbone, i = e.Math, s = e.Tracker, o = e._, u = r.View, a = function(e) {
        s.link({
            linkName: "activity_kit",
            linkPosition: "activity_kit/" + e
        })
    }, f = function() {
        this.removeAttribute("width"), this.removeAttribute("height")
    }, l = t.ProgressView = u.extend({
        className: "progress",
        percent: 0,
        initialize: function(e) {
            var t = this, r = o.bind(t.remove, t);
            t.promise = n.when(e.promise).then(function() {
                r()
            }, r, o.bind(t.progress, t)), t.render()
        },
        progress: function(e) {
            this.percent = e, this.$(".bar").css("width", e * 100 + "%")
        },
        render: function() {
            var e = n("<div/>", {
                "class": "track"
            });
            e.append(n("<div/>", {
                "class": "bar"
            })), this.$el.empty().append(e), this.progress(this.percent)
        }
    }, {
        forPromise: function(e, t) {
            e = n.when(e), e.state() === "pending" && n(t).prepend((new l({
                promise: e
            })).el)
        }
    }), c = u.extend({
        tagName: "div",
        className: "layer",
        transformControl: '<div class="transformers-container"><div class="transformers"><div class="rotate handle"><div class="handle-aspect"></div></div><div class="resize handle"><div class="handle-aspect"></div></div></div></div>',
        endPos: {},
        events: {
            mousedown: "mousedown",
            touchstart: "touchstart",
            dragstart: function(e) {
                e.preventDefault()
            }
        },
        initialize: function() {
            var t = new e.Image, r = this.$el, i = this;
            t.src = this.model.src(), this.render(), r.append(t), this.options.rule.get("selfReplace") ? r.css("z-index", this.options.rule.get("zIndex")) : (this.options.old=!1, r.css("z-index", 999)), t.loaded ? i.assignOrientation(t, r) : n(t).on("load", function() {
                i.assignOrientation(t, r)
            }), this.options.rule.get("movable") && r.append(this.transformControl), this.$el.append('<div class="glow" />'), this.model.on("change", this.render, this)
        },
        assignOrientation: function(e, t) {
            var n = e.naturalHeight / e.naturalWidth, r = n > 2.25 ? "ultra-portrait": n > 1 ? "portrait": n > .75 ? "square": "landscape";
            t.addClass(r + "-image")
        },
        startgesture: function(e, t, r) {
            var i = this;
            if (this.movable()) {
                var s = this.model, u = this.$el.closest(".scene"), a = u.offset();
                a.width = u.outerWidth(), a.height = u.outerHeight(), this.gesture = {
                    offset: o.extend({}, a),
                    initial: {
                        x: s.get("x"),
                        y: s.get("y"),
                        width: s.get("width"),
                        rotation: s.get("rotation")
                    },
                    start: {
                        x: e - a.left,
                        y: t - a.top,
                        target: r,
                        direction: 0
                    }
                }
            }
            this.options.rule.get("min") > 0 && this.options.trash.toggleVisible(!1);
            var f = [];
            n.each(this.$el.siblings(), function(e, t) {
                f.push({
                    el: t,
                    index: parseInt(n(t).css("z-index"), 10)
                })
            }), f = f.sort(function(e, t) {
                return e.index - t.index
            }), n.each(f, function(e, t) {
                t.index = e + 1, e > 0 && n(t.el).css("z-index", t.index)
            }), !this.options.rule.get("selfReplace") && this.options.old && this.$el.css("z-index", o.last(f).index + 1), n.each(this.model.collection.models, function(e, t) {
                t === i.model ? i.model.inFocus=!i.model.inFocus : t.inFocus=!1
            }), n(r).hasClass("handle-aspect") || (n(".layer.focus").removeClass("focus"), n(r).closest(".layer").hasClass("background-rule-type") || this.$el.closest(".scene").removeClass("focus-on-layer"))
        },
        movegesture: function(e, t) {
            var r = this.model, s = this.gesture, o = this.options.trash;
            if (s) {
                var u = {
                    x: e - s.offset.left,
                    y: t - s.offset.top
                }, a = u.x - s.start.x, f = u.y - s.start.y, l = n(s.start.target).closest(".layer"), c = {
                    top: l.position().top + l.height() / 2,
                    left: l.position().left + l.width() / 2
                }, h = i.atan2(u.y - c.top, u.x - c.left) - i.atan2(s.start.y - c.top, s.start.x - c.left), p = s.start.x - c.left, d = u.x - c.left, v = i.abs(d) - i.abs(p);
                n(s.start.target).parent().hasClass("rotate") ? this.$el.hasClass("rotating") || this.$el.addClass("rotating") : n(s.start.target).parent().hasClass("resize") ? this.$el.addClass("resizing") : this.$el.addClass("dragging"), n(s.start.target).parent().hasClass("rotate") ? (r.set("rotation", s.initial.rotation + h), this.$el.find(".transformers").css({
                    transform: "rotate(" + h + "rad)"
                })) : n(s.start.target).parent().hasClass("resize") ? r.set("width", s.initial.width + v / s.offset.width) : (r.set("x", s.initial.x + a / s.offset.width), r.set("y", s.initial.y + f / s.offset.height)), a !== 0 && f !== 0 && (this.model.inFocus=!0)
            }
            this.options.rule.get("required") || o.toggleVisible(!0), o.toggleHover(o.inHitZone(e, t)&&!this.options.rule.get("required")), this.endPos = {
                left: e,
                top: t
            }
        },
        cancelgesture: function() {
            var e = this.gesture;
            e && this.endgesture(e.start.x, e.start.y)
        },
        endgesture: function(e, t) {
            var r = this.options.trash, i = this.$el;
            e != null && t != null && (t > n(".choices").offset().top && (t = n("#activity-kit").height() / 3 * 2), this.movegesture(e, t)), this.options.old || (i.css("z-index", 997), this.options.old=!0), this.gesture = null, i.closest("#activity-kit").off("." + this.cid), i.removeClass("dragging rotating resizing"), this.toggleFocus(this.model.inFocus), this.$el.find(".transformers").css({
                transform: "rotate(0rad)"
            }), r.inHitZone(e, t)&&!this.options.rule.get("required") && this.throwOut()
        },
        mousedown: function(e) {
            var t = this.cid;
            this.startgesture(e.pageX, e.pageY, e.target), this.$el.closest("#activity-kit").on("mousemove." + t, o.bind(this.mousemove, this)).on("mouseup." + t + " mouseleave." + t, o.bind(this.mouseup, this))
        },
        mousemove: function(e) {
            e.preventDefault(), this.movegesture(e.pageX, e.pageY)
        },
        mouseup: function(e) {
            this.endgesture(e.pageX, e.pageY)
        },
        touchstart: function(e) {
            var t = this.cid, r, i;
            e.originalEvent ? i = e : i = n.Event(e, {
                target: this.el
            }), r = i.originalEvent.touches;
            if (r.length > 1)
                this.cancelgesture();
            else if (this.movable()) {
                this.startgesture(r[0].pageX, r[0].pageY, r[0].target);
                var s = this.$el.closest(".scene");
                s.on("touchmove." + t, o.bind(this.touchmove, this)).on("touchend." + t, o.bind(this.touchend, this)).on("touchcancel." + t, o.bind(this.cancelgesture, this))
            }
        },
        touchmove: function(e) {
            var t;
            e.originalEvent ? t = e.originalEvent.touches : t = e.touches, this.gesture && (e.preventDefault(), this.movegesture(t[0].pageX, t[0].pageY))
        },
        touchend: function() {
            this.endgesture(this.endPos.left, this.endPos.top), this.endPos = {}
        },
        render: function() {
            var e = this.model, t = this.$el;
            t.addClass(e.attributes.rule + " " + e.attributes.modifier_rule.toLowerCase() + "-rule-type"), t.attr("data-linkedTo", this.options.rule.get("linkedTo")), this.$el.css({
                width: e.get("width") * 100 + "%",
                top: e.get("y") * 100 + "%",
                left: e.get("x") * 100 + "%",
                position: "absolute"
            }).each(f), this.$el.find("img").css({
                transform: "rotate(" + e.get("rotation") + "rad)"
            })
        },
        movable: function() {
            return this.options.editable && this.options.rule.get("movable")
        },
        toggleFocus: function(t) {
            var n = this, r = this.options.trash, i=!this.options.rule.get("required");
            e.clearInterval(r.focusInterval), t ? (n.$el.addClass("focus"), r.toggleVisible(i), r.targetLayer = i ? this : null, r.focusInterval = e.setTimeout(function() {
                n.$el.removeClass("focus"), r.toggleVisible(!1)
            }, 5e3)) : (n.$el.removeClass("focus"), r.toggleVisible(!1), r.targetLayer = null)
        },
        throwOut: function() {
            this.options.model.collection.remove(this.model), e.clearInterval(this.options.trash.focusInterval), this.options.trash.toggleHover(!1), this.options.trash.toggleVisible(!1)
        },
        remove: function() {
            u.prototype.remove.call(this), t.unload.call(this.el)
        }
    });
    t.SceneView = u.extend({
        className: "scene",
        initialize: function() {
            var e = this, t = e.collection;
            e.render(), n("#activity-kit").addClass(this.options.activityType), t.on("add", function(t, r) {
                var i = r.indexOf(t), s = e.layer(t), o = s.el;
                i === 0 ? e.$el.prepend(o) : e.items().eq(i - 1).after(o);
                if (t.creationEvent) {
                    var u;
                    t.creationEvent.originalEvent ? u = t.creationEvent.originalEvent : u = t.creationEvent, e.options.rules.forLayer(t).get("selfReplace") || (t.set("x", (n(u.target).offset().left - n(".scene").offset().left) / n(".scene")[0].offsetWidth - .03), t.set("y", (n(u.target).offset().top - n(".scene").offset().top) / n(".scene")[0].offsetHeight - .5))
                }
            }), t.on("remove", function(t, n, r) {
                e.items().eq(r.index).remove()
            }), t.on("reset", e.render, e), n("#activity-kit").on("touchmove", function(e) {
                e.preventDefault()
            })
        },
        render: function() {
            var e = this;
            return e.items().remove(), e.collection.each(function(t) {
                var r = e.layer(t).el;
                e.options.rules.forLayer(t).get("movable") && n(r).append(e.transformControl), e.$el.append(r), n(r).css("z-index", e.options.rules.forLayer(t).get("zIndex"))
            }), e
        },
        layer: function(e) {
            var t = e;
            t.inFocus=!1;
            var n = new c({
                model: t,
                old: !0,
                rule: this.options.rules.forLayer(e),
                editable: this.options.editable,
                trash: this.options.trash
            });
            return n.render(), {
                el: n.el,
                layer: n
            }
        },
        items: function() {
            return this.$el.children()
        }
    }), t.ChoicesView = u.extend({
        className: "choices",
        choicesPerRow: function() {
            return 5
        },
        transitionEffect: function() {
            return "slide"
        },
        events: {
            "mousedown .layer": function(e) {
                e.preventDefault();
                var t = e.currentTarget;
                this.applyChoice(this.$(".layer").index(t), e)
            },
            "touchstart .layer": function(e) {
                e.preventDefault();
                var t = e.currentTarget;
                this.applyChoice(this.$(".layer").index(t), e)
            },
            "touchmove .layer": function(e) {
                e.preventDefault()
            },
            "mousemove .layer": function(e) {
                e.preventDefault()
            }
        },
        initialize: function(e) {
            var t = this.scene = e.scene, n = this.collection;
            n.on("reset", this.render, this), this.render(), t.on("add", this.selection, this), t.on("remove", this.selection, this), t.on("reset", this.selection, this)
        },
        selection: function() {
            var e = this.scene;
            this.$(".layer").each(function() {
                var t = n(this), r = t.data("id");
                t.toggleClass("selected", !!e.get(r))
            })
        },
        render: function() {
            var e = this, r = n("<div/>"), i, s = this.collection;
            r.css("display", "none"), s.each(function(t, s) {
                s%e.choicesPerRow() === 0 && (i = n("<ul/>", {
                    "class": t.get("rule") + " " + t.get("modifier_rule").toLowerCase() + "-rule-type"
                }).appendTo(r));
                var o = n("<img/>", {
                    "class": "icon",
                    src: t.get("src")[768]
                }), u = n("<div/>", {
                    "class": "layer " + t.get("rule") + " " + t.get("modifier_rule").toLowerCase() + "-rule-type",
                    data: {
                        id: t.id
                    }
                }).append(o);
                i.append(n("<li/>").append(u))
            }), this.$("img").each(t.unload), this.$el.empty().append(r), this.$("img").each(f), r.swipeasaurus({
                arrows: !0,
                right: 0,
                loop: !1,
                effect: this.transitionEffect()
            }), this.selection(), r.fadeIn(), l.forPromise(s.prefetch(), this.$el)
        },
        applyChoice: function(e, t) {
            var n = this.options, r = this.collection, i = r.at(e), s = this.options.rules, o, u = this.scene, f;
            if (n.groups.include(i)) {
                var l = n.toolbar.options;
                n.scenes[l.active.id].reset(u.toArray()), u.reset(n.scenes[i.id].toArray()), n.active = l.active = i
            } else 
                o = s.forLayer(i), u.get(i.id) ? o.get("min") < u.ofRule(o).length && u.remove(i) : (f = i.clone(), o.get("selfReplace") || (f.creationEvent = t), u.add(f)), s.applyTo(u, i, f);
            a("img/" + r.title + (e + 1))
        }
    }), t.ToolbarView = u.extend({
        index: 0,
        className: "toolbar",
        events: {
            "click .tabs li": function(e) {
                var t = e.currentTarget;
                this.switchToTab(n(t).prevAll().length, !0)
            },
            "click .next": function() {
                this.activeTab ? n(this.activeTab).find(".next").trigger("click") : n(".omnislide .choices.selected").find(".next").trigger("click")
            },
            "click .prev": function() {
                this.activeTab ? n(this.activeTab).find(".prev").trigger("click") : n(".omnislide .choices.selected").find(".prev").trigger("click")
            }
        },
        initialize: function(e) {
            this.index = e.index || this.index, this.render()
        },
        render: function() {
            var e = n("<ul/>", {
                "class": "tabs tabs-" + this.options.tabs.length
            }), t = n("<div/>", {
                "class": "eye"
            }).append(n("<div/>", {
                "class": "next",
                role: "button",
                title: "Next"
            })).append(n("<div/>", {
                "class": "prev",
                role: "button",
                title: "Prev"
            }));
            this.options.tabs.each(function(t) {
                e.append(n("<li/>", {
                    "class": "tab " + t.get("rule")
                }).append(n("<div/>", {
                    "class": "tab-content"
                }).append(n("<img/>", {
                    "class": "lit",
                    src: t.get("icon_active") ? t.get("icon_active"): t.get("active_icon_image_fog") ? t.get("active_icon_image_fog"): t.get("icon_image_fog")
                })).append(n("<img/>", {
                    "class": "dim",
                    src: t.get("icon_default") ? t.get("icon_default"): t.get("icon_image_fog")
                })).append(n("<span/>", {
                    text: t.get("title"),
                    "class": "title"
                }))))
            }), e.find("li").first().addClass("active"), this.$el.empty().append(t).append(e), this.$("img").each(f), this.switchToTab(this.index)
        },
        switchToTab: function(n, r) {
            var s = this.options, o = this.index =+ n, u = s.tabs.at(o), f = u.get("title").toLowerCase(), l = 100, c = 1.5, h = 96, p = i.max(this.options.tabs.length, 5);
            u.get("finish") ? (s.omnislide && s.omnislide.children(".choices").removeClass("selected"), this.options.finish.$el.delay(l).fadeIn(), this.$(".eye").css("top", 80 + c + "%")) : (this.options.finish.$el.fadeOut(150), t.data.activityType === "stickerbook" && this.collection.reset(u.findChoices(s.active, s.groups, s.choices)), this.$(".eye").css("top", o * h / p + c + "%")), this.$(".eye").attr("class", "eye " + u.get("rule")), this.$el.attr("class", "toolbar " + u.get("rule") + " tab-" + o), e.setTimeout(function() {
                this.$(".tabs li").removeClass("active").eq(o).addClass("active")
            }, 100), r && a("button/" + f), this.collection.title = f
        }
    }), t.TrashView = u.extend({
        className: "trash",
        tagName: "div",
        focusInterval: null,
        targetLayer: null,
        events: {
            click: function() {
                var e = this.targetLayer;
                e && (e.throwOut(), this.toggleVisible(!1))
            }
        },
        inHitZone: function(e, t) {
            var n = {};
            return n.left = this.$el.offset().left, n.right = n.left + this.$el.width(), n.top = this.$el.offset().top, n.bottom = n.top + this.$el.height(), e > n.left && t > n.top && e < n.right && t < n.bottom
        },
        toggleVisible: function(e) {
            this.$el.toggleClass("active", e)
        },
        toggleHover: function(e) {
            e ? (this.$el.addClass("hovering"), n(".dragging").css("opacity", "0.15")) : (this.$el.removeClass("hovering"), n(".dragging").css("opacity", "1"))
        }
    }), t.FinishView = u.extend({
        className: "finish-container",
        events: {
            "click .print": function() {
                a("button/print"), e.print()
            },
            "click .back": function() {
                a("button/back");
                var e = this.options.toolbar.$el.find("ul li").get(0);
                n(e).trigger("click")
            }
        },
        initialize: function() {
            var r = this.$el.empty(), i = n("<div/>", {
                "class": "finish"
            }), s = n("<div/>", {
                "class": "print",
                role: "button",
                title: t.translations.print,
                text: t.translations.print
            }), o = n("<div/>", {
                "class": "reload back",
                role: "button",
                title: t.translations.back,
                text: t.translations.back
            });
            i.append(o), e.print && i.append(s), r.append(i)
        }
    }), n(e).on("touchstart", function() {})
}(this), function(e) {
    "use strict";
    var t = e.ActivityKit, n = e.jQuery, r = e._, i = e.Backbone;
    t.data = {};
    var s = function(e) {
        var t = 0, n = function() {
            t < e.length && e[t++].prefetch().then(n)
        };
        e = e.slice(), n(), n()
    };
    t.init = function(o, u, a, f) {
        r.extend(t.data, {
            activityType: a,
            hasPrimary: !1,
            orientation: "landscape"
        }), t.translations = {
            print: "Print",
            done: "Done",
            back: "Back"
        }, r.extend(t.translations, f);
        var l = u, c = [], h = [], p;
        o.rules = [], n.each(o.categories, function(e, r) {
            r.rule = r.title.replace(/ /g, "-").replace(/'/g, "").toLowerCase(), o.rules.push({
                id: r.rule,
                template: r.modifier_rule
            }), r.primary && (t.data.hasPrimary=!0, n.each(r.assets, function(e, t) {
                h.push(t.id)
            }))
        }), t.data.hasPrimary || (p = new t.GroupList({
            id: "emptyPrimary"
        }), h.push("emptyPrimary")), n.each(o.rules, function(e, t) {
            r.extend(t, l[t.template])
        }), n.each(o.categories, function(e, r) {
            n.each(r.assets, function(e, t) {
                r.primary ? t.groups = [t.id] : t.groups = h, t.id = t.id.toString(), t.src = {
                    320: t.iphone_image_fog,
                    640: t.iphone_retina_image_fog,
                    768: t.ipad_image_fog,
                    1024: t.desktop_image_fog,
                    1536: t.ipad_retina_image_fog,
                    2048: t.desktop_retina_image_fog
                }, t.rule = r.title.replace(/ /g, "-").replace(/'/g, "").toLowerCase(), t.modifier_rule = r.modifier_rule, c.push(t)
            }), r.primary && (r.group=!0, p = new t.GroupList(r.assets))
        }), o.categories.push({
            title: t.translations.done,
            rule: "done",
            finish: !0,
            icon_default: "//a.dilcdn.com/a/activity_kit_app/icon-done-2479b5d7e6bd.png",
            icon_active: "//a.dilcdn.com/a/activity_kit_app/icon-done_active-aa07dcc2a77f.png"
        });
        var d = /(?:\?|&)group=([^&]*)/.exec(e.location.search), v = new t.RuleList(o.rules), m = new t.TabList(o.categories), g = new t.LayerList(c, {
            comparator: function(e) {
                return e.groups().length
            }
        }), y = new t.LayerList, b = p.reduce(function(e, n) {
            var r = e[n.id] = new t.LayerList;
            return v.applyTo(r, new t.LayerList(g.ofGroup(n))), y.add(r.models), e
        }, {}), w = d && p.get(d[1]) || (d = null) || p.first(), E = b[w.id].clone(), S = new t.LayerList, x = new t.TrashView({
            scene: E
        }), T = new t.SceneView({
            activityType: t.data.activityType,
            collection: E,
            rules: v,
            editable: !0,
            trash: x
        }), N = new t.FinishView({
            choices: g,
            tabs: m
        }), C = new t.ToolbarView({
            activityType: t.data.activityType,
            collection: S,
            active: w,
            scene: E,
            tabs: m,
            index: d ? 1: 0,
            choices: g,
            groups: p,
            finish: N
        }), k = [];
        t.data.activityType === "stickerbook" && (k[0] = new t.ChoicesView({
            activityType: t.data.activityType,
            collection: S,
            active: w,
            scene: E,
            scenes: b,
            rules: v,
            groups: p,
            toolbar: C,
            trash: x
        }));
        var L = [];
        n.each(k, function(e, t) {
            L.push(t.el)
        });
        var A = n("<div/>", {
            "class": "omnislide"
        }).append(N.el);
        n.each(L, function(e, t) {
            A.append(t)
        }), r.extend(N.options, {
            toolbar: C
        }), r.extend(C.options, {
            omnislide: A
        });
        var O = n("<div/>", {
            "class": "scaler"
        }).append(T.el), M = n("<div/>", {
            "class": "controls"
        }).prepend(x.el).prepend(C.el).prepend(A), D = function() {
            var r = n(".stage"), s = n(e).height() - r.offset().top, o = i.Resize.width(), u = s / o, a = u > 1 ? "portrait": "landscape";
            o > 1024 && (a = "landscape"), a !== t.data.orientatation && (r.removeClass(t.data.orientation).addClass(a), t.data.orientation = a), n(".loading-spacer").addClass("complete")
        }, P = y.add(m).prefetch().then(function() {
            n("#activity-kit").prepend(M).prepend(O), D(), n(e).on("resize", r.throttle(D, 100)), s(g)
        });
        N.$el.hide(), t.ProgressView.forPromise(P, O), n(e.document).on("keydown", function(e) {
            var t = x.targetLayer;
            e.keyCode === 8 && e.preventDefault();
            if (t) {
                var n = t.model.get("rotation"), r = t.model.get("width");
                e.keyCode === 8 ? (e.preventDefault(), t.throwOut()) : e.keyCode === 27 ? (t.model.inFocus=!1, t.toggleFocus(!1)) : e.keyCode === 38 ? (r += .03, t.model.set("width", r)) : e.keyCode === 40 ? r > .1 && (r -= .03, t.model.set("width", r)) : e.keyCode === 37 ? (n = n === undefined ? 0 : n - .04, t.model.set("rotation", n)) : e.keyCode === 39 && (n = n === undefined ? 0 : n + .04, t.model.set("rotation", n))
            }
        })
    }
}(this), Whiskers("shared.contenttoolbar", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="toolbar"> <div id="metadata" class="bound"> <div class="module_body"> <div class="contentarea col-container"> <div class="content"> <div class="item"> '), r.s(r.f("contentToolbarTitleLink", e, t, 1), e, t, 0, 188, 328, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.s(n.f("asset_desktop_image", e, t, 1), e, t, 0, 213, 303, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('" class="icon"> <img src="'), n.b(n.v(n.f("asset_desktop_image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </a> ')
        }), e.pop()), n.b(" ")
    }), e.pop()), r.b(' <div class="productlink"> <h1>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h1> "), r.s(r.f("badge", e, t, 1), e, t, 0, 412, 498, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <span class="watch_icon"> <img src="'), n.b(n.v(n.f("badge", e, t, 0))), n.b('" class="watchpageicon" alt=""> </span> ')
    }), e.pop()), r.b(" "), r.s(r.f("contentToolbarTitleLink", e, t, 1), e, t, 0, 537, 570, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a> ")
    }), e.pop()), r.b(' </div> </div> <p itemprop="description">'), r.b(r.v(r.f("description", e, t, 0))), r.b(r.v(r.f("rating", e, t, 0))), r.b("</p> "), r.s(r.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (r.b(" "), r.s(r.f("use_deprecated_social", e, t, 1), e, t, 0, 704, 1023, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <ul id="social-links"'), n.s(n.f("linksInverted", e, t, 1), e, t, 0, 744, 761, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' class="inverted"')
        }), e.pop()), n.b("> "), n.s(n.f("socialLinks", e, t, 1), e, t, 0, 797, 819, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.rp("socialLinkPartial", e, t, ""))
        }), e.pop()), n.b(" "), n.s(n.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || n.b('<li class="facebook"> <div class="fb-like" data-send="false" data-layout="button_count" data-show-faces="false" data-width="90"></div> </li>'), n.b(" </ul> ")
    }), e.pop()), r.b(" "), r.s(r.f("use_deprecated_social", e, t, 1), e, t, 1, 0, 0, "") || (r.b(" "), r.s(r.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || (r.b(" "), r.b(r.rp("socialLinkPartial", e, t, "")), r.b(" ")), r.b(" ")), r.b(" ")), r.b(" "), r.s(r.f("contentToolbarPromoLink", e, t, 1), e, t, 0, 1206, 1563, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('" class="watch_icon item" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <div class="messageLink"> '), n.s(n.f("asset_desktop_image", e, t, 1), e, t, 0, 1316, 1401, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="watch-icon"> <img src="'), n.b(n.v(n.f("asset_desktop_image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </div> ')
        }), e.pop()), n.b(' <div class="copy'), n.s(n.f("asset_desktop_image", e, t, 1), e, t, 1, 0, 0, "") || n.b(" noImage "), n.b('"> <p aria-hidden="true">'), n.b(n.v(n.f("description", e, t, 0))), n.b("</p> </div> </div> </a> ")
    }), e.pop()), r.b(" "), r.s(r.f("purchase", e, t, 1), e, t, 0, 1605, 1703, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div id="purchase" class="bound"> <p> <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('" target="_blank">'), n.b(n.v(n.f("text", e, t, 0))), n.b("</a> </p> </div> ")
    }), e.pop()), r.b(" </div> </div> "), r.s(r.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (r.b(' <div class="ad-container"> <div class="companionad">'), r.b(r.t(r.f("companionAd", e, t, 0))), r.b("</div> "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" </div> ")), r.b(" </div> </div> </div>"), r.fl()
}), Whiskers("partials.social_link", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<li class="social-dropdown-btn '), r.b(r.v(r.f("share_type", e, t, 0))), r.b('"> <div class="label" role="button">'), r.b(r.v(r.f("share_text", e, t, 0))), r.b('</div> <!-- dropdown area --> <div class="dropdown-panel"> <span class="dropdown-label">'), r.b(r.v(r.f("output_label", e, t, 0))), r.b(':</span> <textarea rows="1" class="text-output"></textarea> </div> </li>'), r.fl()
}), Whiskers("partials.social_btn", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<button type="button" class="social-link-btn" data-source="'), r.s(r.f("shareOverrides", e, t, 1), e, t, 0, 78, 101, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.v(n.d("shareOverrides.href", e, t, 0)))
    }), e.pop()), r.s(r.f("shareOverrides", e, t, 1), e, t, 1, 0, 0, "") || r.b(r.v(r.f("href", e, t, 0))), r.b('" data-thumb="'), r.b(r.v(r.f("thumb", e, t, 0))), r.b('" data-title="'), r.s(r.f("shareOverrides", e, t, 1), e, t, 0, 222, 246, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.v(n.d("shareOverrides.title", e, t, 0)))
    }), e.pop()), r.s(r.f("shareOverrides", e, t, 1), e, t, 1, 0, 0, "") || r.b(r.v(r.f("title", e, t, 0))), r.b('"> <span class="icon-social"></span> <span class="social-share-title">'), r.b(r.v(r.d("translations.share", e, t, 0))), r.b("</span> </button>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.jQuery, n = e.document, r = e.Tracker;
    t(function() {
        var i = e.location;
        t(n).on("click", ".social-dropdown-btn .label", function() {
            var n = t(this).closest(".social-dropdown-btn"), s = function(e) {
                var n = t(e).closest('div[itemprop="video"]').find('meta[itemprop="embedId"]').attr("content"), r = t(e).closest(".bun.watch").find('meta[itemprop="title"]').attr("content"), s = "http://" + i.host + "/embed/" + n;
                t(e).find(".text-output").text('<iframe width="560" height="315" src="' + s + '" frameborder="0" allowfullscreen scrolling="no"></iframe><br/><a href="' + i.href + '">' + r + " on Disney Video</a>")
            }, o = function(e) {
                t(e).removeClass("active"), t("body").off("click.watch"), t(e).hasClass("active") && (t(e).removeClass("active"), u(e))
            }, u = function(n) {
                r && r.event({
                    eventCategory: "socialelement",
                    action: "embed",
                    label: e.location.href
                }), t(n).hasClass("share") ? t(n).find(".text-output").text(i.href) : t(n).hasClass("embed") && s(n), t(n).addClass("active"), t("body").on("click.watch", function(e) {
                    var r = t(e.target);
                    r.closest(".dropdown-panel").length === 0 && r.closest(".social-dropdown-btn").length === 0 && o(n)
                })
            };
            n.hasClass("active") ? o(n) : (t.each(t(".social-dropdown-btn.active"), function(e, t) {
                o(t)
            }), i = e.location, u(n))
        });
        var s = t(".dropdown-panel");
        t(n).on("click", s, function() {
            t(this).find(".text-output").select()
        })
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Backbone, r = e.jQuery, i = e._, s = e.Whiskers, o = e.Tracker;
    t.ContentToolbarView = n.View.extend({
        initialize: function() {
            this.render(), this.init()
        },
        render: function() {
            var e = this.options.ads, t = this.model.get("type") === "video", n = t ? s.partials.social_link: s.partials.social_btn, r = this.model.get("translations") || {}, o = i.extend({
                use_deprecated_social: t,
                translations: r,
                cds: this.options.cdsMode&&!e.showCDSAds(),
                socialLinks: this.options.socialLinks,
                companionAd: function() {
                    return e.tag(this.type.toLowerCase() + "Companion")
                }
            }, this.model.get("type") === "activitykit" ? this.model.get("data")[1] : this.model.get("data")[0], e ? e.helpers() : {});
            this.$el.html(s.shared.contenttoolbar.render(o, {
                socialLinkPartial: n
            }))
        },
        init: function() {
            var n = this;
            n.$(".share-embed-btn").on("click", function() {
                var t = function() {
                    n.$("#share_embed").removeClass("active"), r("body").off("click.watch"), n.$("#share_link").hasClass("active") && (n.$("#share_link").removeClass("active"), i())
                }, i = function() {
                    o && o.event({
                        eventCategory: "socialelement",
                        action: "embed",
                        label: e.location.href
                    }), n.$("#share_embed").addClass("active"), r("body").on("click.watch", function(e) {
                        var n = r(e.target);
                        n.closest("#share_embed").length === 0 && n.closest(".share-embed-btn").length === 0 && t()
                    })
                };
                n.$("#sharing .active").length ? t() : i()
            }), n.$(".share-link-btn").on("click", function() {
                var t = function() {
                    n.$("#share_link").removeClass("active"), r("body").off("click.watch"), n.$("#share_embed").hasClass("active") && (n.$("#share_embed").removeClass("active"), i())
                }, i = function() {
                    o && o.event({
                        eventCategory: "socialelement",
                        action: "share",
                        label: e.location.href
                    }), n.$("#share_link").addClass("active"), r("body").on("click.watch", function(e) {
                        var n = r(e.target);
                        n.closest("#share_link").length === 0 && n.closest(".share-link-btn").length === 0 && t()
                    })
                };
                n.$("#sharing .active").length ? t() : i()
            });
            var s = r("#share_embed");
            r(".code", s).on("click", function() {
                r(this).select()
            });
            var u = r("#share_link");
            r(".link", u).on("click", function() {
                r(this).select()
            });
            var a = i.first(this.model.get("data")), f = e.location, l = "http://" + f.host + "/embed/" + a.id;
            n.$("#share_embed .code").text('<iframe width="560" height="315" src="' + l + '" frameborder="0" allowfullscreen scrolling="no"></iframe><br/><a href="' + f.href + '">' + a.title + " on Disney Video</a>"), n.$(".quick_link input[type=text]").val(f.href), t.FB.go()
        },
        remove: function() {
            r("body").off("click.watch");
            var e = r("#share_embed"), t = r("#share_link");
            r(".code", e).off("click"), r(".link", t).off("click"), r("#sharing").appendTo("body").remove()
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.ActivityKit, n = e.Disney, r = e.Grill, i = e._, s = e.jQuery, o = e.Math, u = e.Whiskers, a = r.ModuleView.extend({
        bodyClassName: "content-page",
        template: u.modules.activity_kit_shared,
        initialize: function() {
            var e = this, n = i.first(e.model.get("data")), r = e.model.get("view"), s = t.rule_templates;
            e.translations = e.model.get("translations"), i.defaults(e.translations, {
                print: "Print",
                done: "Done",
                back: "Back"
            });
            var o = i.last(this.model.get("data")).pslug, u = i.last(this.model.get("data")).ptitle;
            !o && u && (o = u.replace(/ /g, "-").replace(/'/g, "").toLowerCase()), o && (this.bodyClassName += " " + o), t.init(n, s, r, e.translations)
        },
        render: function() {
            var e = this, t = i.last(e.model.get("data")), n = e.model.get("data").length > 1 && i.first(e.model.get("data")), r = i.extend({
                data: t,
                kitData: n,
                cds: e.options.cds,
                theme: e.model.get("style").default_color_theme,
                view: e.model.get("view"),
                upNext: this.upNext && this.upNext(),
                translations: e.translations
            }, this.options.ads.helpers());
            e.options.cds && (r.allurl += "?cds"), e.$el.html(e.template.render(r, {
                kitStyles: u.partials.activity_kit_theming,
                container: u.partials.activity_kit_content
            })), s("#body-wrapper").attr("class", e.bodyClassName)
        },
        ready: function() {
            var t = this, r = t.options, o = i.last(t.model.get("data")), u = t.translations, a = e.Tracker;
            this.tb = new n.ContentToolbarView({
                el: this.$("#meta"),
                model: this.model,
                socialLinks: [{
                    share_type: "embed",
                    share_text: u.embed,
                    output_label: u.embed
                }, {
                    share_type: "share",
                    share_text: u.share,
                    output_label: u.quick_link
                }
                ],
                cdsMode: r.cds,
                ads: r.ads
            }), e.setTimeout(function() {
                var e = o.ptitle || "", n = o.slug.replace(/-/g, "") || "", r = o.id || "", i = t.generateUUID() || "", s = (new Date).getTime() || "";
                a.app({
                    assetTypeCode: "app",
                    appEvent: "load",
                    appBuCode: "dcom",
                    appPropertyCode: e,
                    appOwnerName: "dcom",
                    appTypeCode: "misc",
                    appGenreCode: "",
                    appName: n,
                    assetId: r,
                    assetName: "app|dcom|" + e + "|dcom|misc||" + r + "|" + n || "",
                    appTime: s,
                    appSessionId: i
                })
            }, 1e3), s("html").addClass("single-page-print")
        },
        containerType: function() {
            return i.contains(["stickerbook", "avatar"], this.model.get("view")) ? "activity_kit" : i.last(this.model.get("data")).type.toLowerCase()
        },
        generateUUID: function() {
            var e = function() {
                return ((1 + o.random()) * 65536 | 0).toString(16)
                .substring(1)
            };
            return e() + e() + "-" + e() + "-" + e() + "-" + e() + "-" + e() + e() + e()
        },
        remove: function() {
            r.ModuleView.prototype.remove.call(this), s("#body-wrapper").removeClass(this.bodyClassName), s("html").removeClass("single-page-print"), this.tb.remove()
        }
    });
    a.register("stickerbook", {})
}(this), Whiskers("modules.banner", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> '), r.b(r.t(r.f("bannerAd", e, t, 0))), r.b(" </div>"), r.fl()
}), Whiskers("modules.sponsor", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> '), r.b(r.t(r.f("sponsorAd", e, t, 0))), r.b(" </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = t.Style.breakpoints, r = t.Style, i = e.Backbone, s = e.jQuery;
    r.moduleStyle = function(t, r, o) {
        var u = i.Resize.width() >= n.midHigh, a = e.devicePixelRatio > 1.5, f = t.$el, l = t.model, c = l.get("style") || {}, h = "rgba(0, 0, 0, 0.12)", p = f.parents("#main").length > 0, d=!1;
        r !== undefined && (p = r), o !== undefined && (d = o), c.top_border_override && (p = c.top_border_override.toLowerCase() === "border", p && f.addClass("border-override-true")), c.bottom_border_override && (d = c.bottom_border_override.toLowerCase() === "border"), c.geo_hidden&&!f.hasClass("geo-accepted") && f.addClass("geo-hidden"), p && f.css("border-top", "1px solid " + h), d && f.css("border-bottom", "1px solid " + h), f.css("border-color", h);
        var v = {}, m = {}, g = {};
        v.dark = c.module_dark_background_color || l.collection.style.background_color_dark, v.light = c.module_light_background_color || l.collection.style.background_color_light, v.white = "#ffffff", m.dark = c.module_dark_heading_color, m.light = c.heading_color, m.white = c.heading_color, g.dark = c.module_dark_body_text_color, g.light = c.module_light_body_text_color;
        var y = {
            theme: c.color_theme ? c.color_theme.toLowerCase(): c.default_color_theme || "light",
            color: v[c.default_color_theme],
            header_color: m[c.default_color_theme],
            text_color: g[c.default_color_theme]
        };
        switch (y.theme) {
        case"white":
            y.color = null, y.header_color = m.light, y.text_color = g.light;
            break;
        case"light":
            y.color = v.light, y.header_color = m.light, y.text_color = g.light;
            break;
        case"dark":
            y.color = v.dark, y.header_color = m.dark, y.text_color = g.dark
        }
        c.background_color && (y.color = c.background_color);
        var b = c && c.images && c.images.background_image, w = b && b.url, E = b && b.retina_url || w;
        y.image = a ? E : w, f.addClass(y.theme), u ? (y.color && f.css({
            "background-color": y.color,
            "background-image": "none"
        }), y.image && (y.position = (b && b.alignment) + " left", f.css({
            "background-image": "url(" + y.image + ")",
            "background-repeat": "no-repeat",
            "background-position": y.position,
            "background-size": "100% auto"
        }), y.color && f.css("background-color", y.color))) : (y.image && f.css("background-image", "none"), y.color && f.css("background-color", y.color));
        if (y.header_color) {
            var S = f.find("h1, h2, h1 a, h2 a, h3").not(".all a");
            S.each(function(e, t) {
                var n = s(t);
                n.closest(".skip-styles").length < 1 && n.css("color", y.header_color)
            })
        }
        if (y.text_color) {
            var x = f.find("p, .item-titles-below h4");
            x.each(function(e, t) {
                var n = s(t);
                n.closest(".skip-styles").length < 1 && n.css("color", y.text_color)
            })
        }
    }
}(this), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e._, s = t.ModuleView.extend({
        render: function() {
            var e = n.Ads.getAdContext();
            if (e instanceof n.Ads.AdTechAdContext) {
                var t = this.model.get("advertisement_type"), r = this.model.get("advertisement_placement_id"), s = null;
                t === "leaderboard" ? (s = "pushdown", r = "mh_" + t + "_" + r) : (s = "rectangle", r = "mh_mpu" + r);
                var o = e.createSlot(s, r);
                this.$el.html(o.tag())
            } else {
                var u = i.extend({}, this.options.ads.helpers(), this.model);
                this.$el.html(this.template.render(u)), n.Style.moduleStyle(this, !0, !0)
            }
        }
    });
    s.register("ad_banner", {
        className: "module skip-bottom-border full-height full-width",
        template: r.modules.banner
    }), s.register("sponsor", {
        className: "module full-height full-width",
        template: r.modules.sponsor
    })
}(this), Whiskers("modules.all", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.s(r.f("singleRow", e, t, 1), e, t, 0, 55, 66, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" single-row")
    }), e.pop()), r.s(r.f("item_titles_below", e, t, 1), e, t, 0, 102, 120, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" item-titles-below")
    }), e.pop()), r.b('"> <div class="header module_header"> '), r.s(r.f("title", e, t, 1), e, t, 0, 190, 208, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("<h2>"), n.b(n.v(n.f("title", e, t, 0))), n.b("</h2>")
    }), e.pop()), r.b(' <div class="filters"> <a href="" class="filter"> <span>'), r.b(r.v(r.d("translations.all", e, t, 0))), r.b('</span> </a> <div class="filter_mobile"> <span>'), r.b(r.v(r.d("translations.all", e, t, 0))), r.b('</span> <select class="mobile_filter_list"> <option value="0" data-filter="All" selected="selected">'), r.b(r.v(r.d("translations.all", e, t, 0))), r.b('</option> </select> </div> </div> </div> <div class="filter_drawer"> <div class="filter_list"> </div> </div> <div id="stream_items"> '), r.b(r.t(r.f("list", e, t, 0))), r.b(" </div> "), r.s(r.f("hasMore", e, t, 1), e, t, 0, 644, 844, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="show_more_container"> <div class="show_more blue button large" role="button" onclick=""> <span class="label">'), n.b(n.v(n.d("translations.show_more", e, t, 0))), n.b('</span> <span class="spinner"></span> </div> </div> ')
    }), e.pop()), r.b(" </div>"), r.fl()
}), Whiskers("modules.col", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<span class="col '), r.s(r.f("lastRow", e, t, 1), e, t, 0, 29, 37, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("last-row")
    }), e.pop()), r.b('">'), r.b(r.rp("entity", e, t, "")), r.b('<span class="arrow"></span></span>'), r.fl()
}), Whiskers("partials.module_elements.module_header", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="module_header'), r.s(r.f("icon_image", e, t, 1), e, t, 0, 40, 45, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" icon")
    }), e.pop()), r.s(r.f("desc", e, t, 1), e, t, 1, 0, 0, "") || r.b(" no-desc"), r.b('"> '), r.b(r.rp("iconHeader", e, t, "")), r.b(' <div class="title"> <h2>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h2> "), r.s(r.f("seeAll", e, t, 1), e, t, 0, 155, 216, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<a class="all" href="'), n.b(n.v(n.f("seeAll", e, t, 0))), n.b('">'), n.b(n.v(n.d("translations.see_all", e, t, 0))), n.b("</a>")
    }), e.pop()), r.b(" </div> "), r.s(r.f("desc", e, t, 1), e, t, 0, 244, 259, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("<p>"), n.b(n.v(n.f("desc", e, t, 0))), n.b("</p>")
    }), e.pop()), r.b(" </div>"), r.fl()
}), Whiskers("partials.icon_header", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("icon_image", e, t, 1), e, t, 0, 15, 673, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="header_icon"style="background-color: '), n.b(n.v(n.f("icon_background_color", e, t, 0))), n.b("; background-image: -webkit-gradient(linear, left top, left bottom, from("), n.b(n.v(n.f("icon_highlight_color", e, t, 0))), n.b("), to("), n.b(n.v(n.f("icon_background_color", e, t, 0))), n.b(")); background-image: -webkit-linear-gradient(top, "), n.b(n.v(n.f("icon_highlight_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("icon_background_color", e, t, 0))), n.b("); background-image: -moz-linear-gradient(top, "), n.b(n.v(n.f("icon_highlight_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("icon_background_color", e, t, 0))), n.b("); background-image: -o-linear-gradient(top, "), n.b(n.v(n.f("icon_highlight_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("icon_background_color", e, t, 0))), n.b("); background-image: linear-gradient(to bottom, "), n.b(n.v(n.f("icon_highlight_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("icon_background_color", e, t, 0))), n.b(');"> <div class="icon_'), n.b(n.v(n.f("icon_type", e, t, 0))), n.b(' icon_img"></div> </div> ')
    }), e.pop()), r.fl()
}), Whiskers("entities.album", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <span class="square-box"><span class="aspect"> <img src="'), r.b(r.v(r.f("square", e, t, 0))), r.b('" class="square" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </span></span> <h3 class="title" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> </a>"), r.fl()
}), Whiskers("entities.artist", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b(r.rp("entities/personality", e, t, "")), r.fl()
}), Whiskers("entities.character", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b(r.rp("entities/personality", e, t, "")), r.fl()
}), Whiskers("entities.collection", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <span class="square-box"><span class="aspect"> <img src="'), r.b(r.v(r.f("square", e, t, 0))), r.b('" class="square" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </span></span> <h3 class="title" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> </a>"), r.fl()
}), Whiskers("entities.comments", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="header module_header"> <h2>Comments <span class="comment_count">(<span class="count">0</span>)</span></h2> </div> <div class="gutter"> <form> <div class="emblem"> <span class="avatar"></span> <span class="change">Change</span> </div> <div class="emblems"> <ul> '), r.s(r.f("emblems", e, t, 1), e, t, 0, 305, 376, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li data-id="'), n.b(n.v(n.f("id", e, t, 0))), n.b('"><img src="'), n.b(n.v(n.f("large_image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"></li> ')
    }), e.pop()), r.b(' </ul> </div> <div class="fields"> <div class="words"> <textarea name="shoutout" maxlength="150" placeholder="What do you have to say?" required></textarea> <span class="remaining"></span> </div> <div class="nas"> <div class="input-group name"> <input type="text" name="name" maxlength="20" required placeholder="Your name"></input> </div> <div class="input-group age"> <input type="text" name="age" maxlength="3" required placeholder="Age"></input> </div> <div class="input-group state"> <select name="state" class="empty" required> <option value="" selected>State</option> <option value="AL">AL</option> <option value="AK">AK</option> <option value="AZ">AZ</option> <option value="AR">AR</option> <option value="CA">CA</option> <option value="CO">CO</option> <option value="CT">CT</option> <option value="DE">DE</option> <option value="DC">DC</option> <option value="FL">FL</option> <option value="GA">GA</option> <option value="HI">HI</option> <option value="ID">ID</option> <option value="IL">IL</option> <option value="IN">IN</option> <option value="IA">IA</option> <option value="KS">KS</option> <option value="KY">KY</option> <option value="LA">LA</option> <option value="ME">ME</option> <option value="MD">MD</option> <option value="MA">MA</option> <option value="MI">MI</option> <option value="MN">MN</option> <option value="MS">MS</option> <option value="MO">MO</option> <option value="MT">MT</option> <option value="NE">NE</option> <option value="NV">NV</option> <option value="NH">NH</option> <option value="NJ">NJ</option> <option value="NM">NM</option> <option value="NY">NY</option> <option value="NC">NC</option> <option value="ND">ND</option> <option value="OH">OH</option> <option value="OK">OK</option> <option value="OR">OR</option> <option value="PA">PA</option> <option value="RI">RI</option> <option value="SC">SC</option> <option value="SD">SD</option> <option value="TN">TN</option> <option value="TX">TX</option> <option value="UT">UT</option> <option value="VT">VT</option> <option value="VA">VA</option> <option value="WA">WA</option> <option value="WV">WV</option> <option value="WI">WI</option> <option value="WY">WY</option> <option value="Other">Other</option> </select> </div> </div> <button type="submit" class="button blue small">Submit</button> <p class="notification"></p> </div> </form> <div class="list"></div> <div class="show_more_container"> <div class="show_more blue button large" role="button" onclick=""> <span class="label">Show More</span> <span class="spinner"></span> </div> </div> </div> </div>'), r.fl()
}), Whiskers("entities.comments.message", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<li class="message'), r.s(r.f("pending", e, t, 1), e, t, 0, 30, 38, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" pending")
    }), e.pop()), r.b('"> <div class="arrow"></div> <div class="inner"> <q>'), r.b(r.v(r.f("text", e, t, 0))), r.b("</q> "), r.s(r.f("emblem", e, t, 1), e, t, 0, 126, 197, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="avatar"><img src="'), n.b(n.v(n.f("large_image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"></div> ')
    }), e.pop()), r.b(" "), r.s(r.f("emblem", e, t, 1), e, t, 1, 0, 0, "") || r.b(' <div class="avatar"></div> '), r.b(" <span> "), r.s(r.f("dname", e, t, 1), e, t, 0, 277, 298, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.b(n.v(n.f("dname", e, t, 0))), n.b("</address> ")
    }), e.pop()), r.b(" "), r.s(r.f("name", e, t, 1), e, t, 0, 318, 373, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" <strong>"), n.b(n.v(n.f("name", e, t, 0))), n.b("</strong> Age "), n.b(n.v(n.f("age", e, t, 0))), n.b(", from "), n.b(n.v(n.f("state", e, t, 0))), n.b(" ")
    }), e.pop()), r.b(" </span> </div> </li>"), r.fl()
}), Whiskers("entities.console_game", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <div class="game-box '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 93, 101, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isMobile")
    }), e.pop()), r.b(" "), r.s(r.f("isConsole", e, t, 1), e, t, 0, 129, 138, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isConsole")
    }), e.pop()), r.b('"><div class="aspect"> <img src="'), r.b(r.v(r.f("thumb", e, t, 0))), r.b('" class="game" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </div></div> '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 252, 303, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<h3 class="title app-title isMobile">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</h3>")
    }), e.pop()), r.b(' <div class="duplicate-title title '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 364, 372, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isMobile")
    }), e.pop()), r.b('"> <h3>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> <h4>"), r.b(r.v(r.f("ptitle", e, t, 0))), r.b("</h4> </div> </a>"), r.fl()
}), Whiskers("entities.contest", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="contest_heading'), r.s(r.f("logo", e, t, 1), e, t, 0, 36, 40, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" img")
    }), e.pop()), r.b('"> <div class="module_header"> <h2 class="'), r.s(r.f("logo", e, t, 1), e, t, 0, 100, 128, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.s(n.f("hide_ad", e, t, 1), e, t, 1, 0, 0, "") || n.b("noAd")
    }), e.pop()), r.b('">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h2> </div> "), r.s(r.f("logo", e, t, 1), e, t, 0, 170, 240, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img class="'), n.s(n.f("logo", e, t, 1), e, t, 0, 192, 214, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.s(n.f("noAd", e, t, 1), e, t, 0, 201, 205, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b("noAd")
            }), e.pop())
        }), e.pop()), n.b('"src="'), n.b(n.v(n.f("logo", e, t, 0))), n.b('"> ')
    }), e.pop()), r.b(' <div class="contestant_container'), r.s(r.f("logo", e, t, 1), e, t, 1, 0, 0, "") || r.s(r.f("noAd", e, t, 1), e, t, 0, 300, 305, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" noAd")
    }), e.pop()), r.b(" "), r.s(r.f("logo", e, t, 1), e, t, 0, 333, 366, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.s(n.f("noAd", e, t, 1), e, t, 0, 342, 357, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" noAd hasHeader")
        }), e.pop())
    }), e.pop()), r.b('"> <ul class="contestants'), r.s(r.f("single_option", e, t, 1), e, t, 0, 418, 425, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" single")
    }), e.pop()), r.b(" "), r.s(r.f("logo", e, t, 1), e, t, 1, 0, 0, "") || r.s(r.f("noAd", e, t, 1), e, t, 0, 462, 502, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" noAd"), n.s(n.f("showFour", e, t, 1), e, t, 0, 480, 489, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" showFour")
        }), e.pop())
    }), e.pop()), r.b(" "), r.s(r.f("logo", e, t, 1), e, t, 0, 530, 563, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.s(n.f("noAd", e, t, 1), e, t, 0, 539, 554, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" noAd hasHeader")
        }), e.pop())
    }), e.pop()), r.b('"> '), r.s(r.f("contestants", e, t, 1), e, t, 0, 591, 1796, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li class="option '), n.b(n.v(n.f("type", e, t, 0))), n.s(n.f("clickable", e, t, 1), e, t, 0, 632, 642, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" clickable")
        }), e.pop()), n.s(n.f("results", e, t, 1), e, t, 1, 0, 0, "") || n.s(n.f("disabled", e, t, 1), e, t, 0, 681, 690, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" disabled")
        }), e.pop()), n.s(n.f("results", e, t, 1), e, t, 0, 727, 734, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" active")
        }), e.pop()), n.b('" data-id="'), n.b(n.v(n.f("id", e, t, 0))), n.b('"> '), n.s(n.f("results", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div class="actions checkboxes"> '), n.s(n.f("vote_button", e, t, 1), e, t, 0, 828, 1006, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <span class="vote checkbox'), n.s(n.f("votes", e, t, 1), e, t, 0, 865, 873, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" checked")
            }), e.pop()), n.s(n.f("disabled", e, t, 1), e, t, 0, 896, 905, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" disabled")
            }), e.pop()), n.s(n.f("loading", e, t, 1), e, t, 0, 930, 938, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" loading")
            }), e.pop()), n.b('" title="'), n.b(n.v(n.f("text", e, t, 0))), n.b('"> <span class="check"></span> </span> ')
        }), e.pop()), n.b(" "), n.s(n.f("checkboxes", e, t, 1), e, t, 0, 1038, 1204, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <span class="checkbox'), n.s(n.f("checked", e, t, 1), e, t, 0, 1072, 1080, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" checked")
            }), e.pop()), n.s(n.f("disabled", e, t, 1), e, t, 0, 1105, 1114, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" disabled")
            }), e.pop()), n.b('" data-index="'), n.b(n.v(n.f("index", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("text", e, t, 0))), n.b('"> <span class="check"></span> </span> ')
        }), e.pop()), n.b(" </div> ")), n.b(" "), n.s(n.f("results", e, t, 1), e, t, 0, 1252, 1549, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="actions results"> '), n.s(n.f("image_poll", e, t, 1), e, t, 0, 1298, 1386, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" "), n.s(n.f("results", e, t, 1), e, t, 0, 1311, 1373, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(' <div class="bar" style="height:'), n.b(n.v(n.f("percent_formatted", e, t, 0))), n.b('"></div> ')
                }), e.pop()), n.b(" ")
            }), e.pop()), n.b(" "), n.s(n.f("show_percentage", e, t, 1), e, t, 0, 1422, 1452, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" <p>"), n.b(n.v(n.f("percent_formatted", e, t, 0))), n.b("</p> ")
            }), e.pop()), n.b(" "), n.s(n.f("show_percentage", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" <p>"), n.b(n.v(n.f("total_formatted", e, t, 0))), n.b("</p> ")), n.b(" </div> ")
        }), e.pop()), n.b(' <div class="item"> '), n.s(n.f("image_poll", e, t, 1), e, t, 0, 1596, 1633, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <img src="'), n.b(n.v(n.f("logo", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("text", e, t, 0))), n.b('"> ')
        }), e.pop()), n.b(" "), n.s(n.f("image_poll", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.s(n.f("results", e, t, 1), e, t, 0, 1677, 1738, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="bar" style="width:'), n.b(n.v(n.f("percent_formatted", e, t, 0))), n.b('"></div> ')
        }), e.pop()), n.b(" ")), n.b(" <p>"), n.b(n.v(n.f("text", e, t, 0))), n.b("</p> </div> </li> ")
    }), e.pop()), r.b(" </ul> "), r.s(r.f("message", e, t, 1), e, t, 0, 1831, 1968, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="message '), n.b(n.v(n.f("type", e, t, 0))), n.s(n.f("single_option", e, t, 1), e, t, 0, 1878, 1885, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" single")
        }), e.pop()), n.b('"> <p>'), n.b(n.v(n.f("text", e, t, 0))), n.b('<span class="again"> Vote again?</span></p> </div> ')
    }), e.pop()), r.b(" "), r.s(r.f("master_button", e, t, 1), e, t, 0, 1999, 2229, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.s(n.f("complete", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div class="master_button '), n.s(n.f("logo", e, t, 1), e, t, 0, 2049, 2082, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.s(n.f("noAd", e, t, 1), e, t, 0, 2058, 2073, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" noAd hasHeader")
            }), e.pop())
        }), e.pop()), n.b('"> <div class="button large blue'), n.s(n.f("disabled", e, t, 1), e, t, 0, 2136, 2145, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" disabled")
        }), e.pop()), n.s(n.f("loading", e, t, 1), e, t, 0, 2170, 2178, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" loading")
        }), e.pop()), n.b('">'), n.b(n.v(n.f("text", e, t, 0))), n.b(" </div> </div> ")), n.b(" ")
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), Whiskers("entities.credit", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b(r.rp("entities/personality", e, t, "")), r.fl()
}), Whiskers("entities.custom-character", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b(r.rp("entities/character", e, t, "")), r.fl()
}), Whiskers("entities.gallery", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="gallery-boundary"> <a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('" class="gallery-box"'), r.s(r.f("ping", e, t, 1), e, t, 0, 78, 99, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <div class="stack-box"> <div class="aspect"> <img src="'), r.b(r.v(r.f("thumbnail", e, t, 0))), r.b('"> </div> <span class="thumb-title"> <h3 class="overlay-title" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b('</h3> </span> </div> <span class="stack-inner top"></span> <span class="stack-inner bot"></span> </a> <a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 398, 419, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <span class="duplicate-title"> <h3 aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> </span> </a> </div>"), r.fl()
}), Whiskers("entities.game", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <div class="game-box '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 93, 101, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isMobile")
    }), e.pop()), r.b(" "), r.s(r.f("isConsole", e, t, 1), e, t, 0, 129, 138, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isConsole")
    }), e.pop()), r.b('"><div class="aspect"> <img src="'), r.b(r.v(r.f("thumb", e, t, 0))), r.b('" class="game" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </div></div> '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 252, 303, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<h3 class="title app-title isMobile">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</h3>")
    }), e.pop()), r.b(' <div class="duplicate-title title '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 364, 372, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isMobile")
    }), e.pop()), r.b('"> <h3>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> <h4>"), r.b(r.v(r.f("ptitle", e, t, 0))), r.b("</h4> </div> </a>"), r.fl()
}), Whiskers("entities.mobile_app", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <div class="game-box '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 93, 101, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isMobile")
    }), e.pop()), r.b(" "), r.s(r.f("isConsole", e, t, 1), e, t, 0, 129, 138, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isConsole")
    }), e.pop()), r.b('"><div class="aspect"> <img src="'), r.b(r.v(r.f("thumb", e, t, 0))), r.b('" class="game" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </div></div> '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 252, 303, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<h3 class="title app-title isMobile">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</h3>")
    }), e.pop()), r.b(' <div class="duplicate-title title '), r.s(r.f("isMobile", e, t, 1), e, t, 0, 364, 372, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isMobile")
    }), e.pop()), r.b('"> <h3>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> <h4>"), r.b(r.v(r.f("ptitle", e, t, 0))), r.b("</h4> </div> </a>"), r.fl()
}), Whiskers("entities.movie", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <span class="poster-box"><span class="aspect"> <img src="'), r.b(r.v(r.f("poster", e, t, 0))), r.b('" class="poster" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </span></span> <h3 class="title" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> </a>"), r.fl()
}), Whiskers("entities.music", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <span class="square-box"><span class="aspect"> <img src="'), r.b(r.v(r.f("square", e, t, 0))), r.b('" class="square" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </span></span> <h3 class="title" aria-hidden="true"><a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</a></h3> </a>"), r.fl()
}), Whiskers("entities.personality", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('" class="'), r.s(r.f("unstyled", e, t, 1), e, t, 1, 0, 0, "") || r.b("personality"), r.s(r.f("unstyled", e, t, 1), e, t, 0, 76, 84, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("unstyled")
    }), e.pop()), r.b('-box entity-box" '), r.s(r.f("ping", e, t, 1), e, t, 0, 123, 144, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <div class="aspect"> '), r.s(r.f("thumb", e, t, 1), e, t, 0, 186, 309, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<img src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('" class="'), n.s(n.f("unstyled", e, t, 1), e, t, 1, 0, 0, "") || n.b("personality"), n.s(n.f("unstyled", e, t, 1), e, t, 0, 264, 272, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("unstyled")
        }), e.pop()), n.b('-image" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">')
    }), e.pop()), r.b(" "), r.s(r.f("thumb", e, t, 1), e, t, 1, 0, 0, "") || (r.b('<img src="'), r.b(r.v(r.f("photo", e, t, 0))), r.b('" class="'), r.s(r.f("unstyled", e, t, 1), e, t, 1, 0, 0, "") || r.b("personality"), r.s(r.f("unstyled", e, t, 1), e, t, 0, 408, 416, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("unstyled")
    }), e.pop()), r.b('-image" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('">')), r.b(' </div> <h3 class="title '), r.s(r.f("item_titles_below", e, t, 1), e, t, 1, 0, 0, "") || r.b("skip-styles"), r.b('" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> </a>"), r.fl()
}), Whiskers("entities.product", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b(' class="skip-styles"> <div class="square-box"> <div class="aspect"> <div class="img-wrapper"> <img src="'), r.b(r.v(r.f("thumb", e, t, 0))), r.b('" class="square product" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </div> </div> </div> <div class="product-details"> '), r.s(r.f("title", e, t, 1), e, t, 0, 273, 319, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<h3 class="title product-title">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</h3>")
    }), e.pop()), r.b(" "), r.s(r.f("price", e, t, 1), e, t, 0, 340, 370, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<p class="price">'), n.b(n.v(n.f("price", e, t, 0))), n.b("</p>")
    }), e.pop()), r.b(" </div> </a>"), r.fl()
}), Whiskers("entities.product_modal", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="product-tech"> <div class="close_modal"></div> <div class="product-modal"> <p class="title tablet">'), r.b(r.v(r.f("title", e, t, 0))), r.b('</p> <div class="leftside"> <div class="boundingBox"> <img class="logo" src="'), r.b(r.v(r.f("thumb", e, t, 0))), r.b('"> </div> <div class="product-button"></div> </div> <div class="details"> <h2 class="title desktop">'), r.b(r.v(r.f("title", e, t, 0))), r.b('</h2> <div class="desc_modal"> <h3 class="special">Special Features</h3> <ul class="desc"> '), r.s(r.f("tech_lines", e, t, 1), e, t, 0, 421, 437, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" <li>"), n.b(n.v(n.d(".", e, t, 0))), n.b("</li> ")
    }), e.pop()), r.b(" </ul> "), r.s(r.f("spec_truth", e, t, 1), e, t, 0, 474, 516, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<h3 class="tech_spec">Technical Specs</h3>')
    }), e.pop()), r.b(" "), r.s(r.f("specs", e, t, 1), e, t, 0, 542, 741, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="spec_col"> <h2 class="spec_type">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</h2> "), n.s(n.f("specifications", e, t, 1), e, t, 0, 622, 714, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <ul class="tech_list"> <li><span class="tech_title">'), n.b(n.v(n.f("title", e, t, 0))), n.b(":</span> "), n.b(n.v(n.f("value", e, t, 0))), n.b("</li> </ul> ")
        }), e.pop()), n.b(" </div> ")
    }), e.pop()), r.b(" </div> </div> </div> </div>"), r.fl()
}), Whiskers("entities.products", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("product_data", e, t, 1), e, t, 0, 17, 863, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="product"> '), n.s(n.f("is2Items", e, t, 1), e, t, 0, 53, 226, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="leftside"> <div class="boundingBox"> <img class="logo" src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('"> </div> <div class="product-button '), n.s(n.f("more_lines", e, t, 1), e, t, 0, 187, 195, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b("hasModal")
            }), e.pop()), n.b('"></div> </div> ')
        }), e.pop()), n.b(" "), n.s(n.f("is2Items", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div class="boundingBox"> <img class="logo" src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('"> </div> ')), n.b(' <div class="details"> <h2 class="title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h2> <div class="product-button"></div> <ul class="desc"> '), n.s(n.f("product_lines", e, t, 1), e, t, 0, 462, 478, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <li>"), n.b(n.v(n.d(".", e, t, 0))), n.b("</li> ")
        }), e.pop()), n.b(' </ul> <a class="modal_info button '), n.s(n.f("more_lines", e, t, 1), e, t, 1, 0, 0, "") || n.b("no_modal"), n.b('">'), n.b(n.v(n.d("translations.show_more", e, t, 0))), n.b('</a> </div> <ul class="desc_mobile '), n.s(n.f("more_lines", e, t, 1), e, t, 0, 647, 655, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("hasModal")
        }), e.pop()), n.b('"> '), n.s(n.f("product_lines", e, t, 1), e, t, 0, 691, 707, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <li>"), n.b(n.v(n.d(".", e, t, 0))), n.b("</li> ")
        }), e.pop()), n.b(" "), n.s(n.f("specs", e, t, 1), e, t, 0, 736, 839, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("specifications", e, t, 1), e, t, 0, 756, 819, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <li><span class="tech_title">'), n.b(n.v(n.f("title", e, t, 0))), n.b(":</span> "), n.b(n.v(n.f("value", e, t, 0))), n.b("</li> ")
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(" </ul> </div> ")
    }), e.pop()), r.fl()
}), Whiskers("entities.promo", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('" '), r.s(r.f("age_gate", e, t, 1), e, t, 0, 32, 75, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' class="age-gated" data-age="'), n.b(n.v(n.f("age_gate", e, t, 0))), n.b('" ')
    }), e.pop()), r.s(r.f("ping", e, t, 1), e, t, 0, 97, 118, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b(' title="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> <span class="'), r.b(r.v(r.f("primaryImageName", e, t, 0))), r.b("-box "), r.b(r.v(r.f("contentType", e, t, 0))), r.b("-box entity-box "), r.s(r.f("isMobile", e, t, 1), e, t, 0, 229, 237, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("isMobile")
    }), e.pop()), r.b('"> <span class="aspect"> <img src="'), r.s(r.f("imgSource", e, t, 1), e, t, 0, 299, 312, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.v(n.f("imgSource", e, t, 0)))
    }), e.pop()), r.s(r.f("imgSource", e, t, 1), e, t, 1, 0, 0, "") || r.b(r.v(r.f("thumb", e, t, 0))), r.b('" class="'), r.b(r.v(r.f("primaryImageName", e, t, 0))), r.b('" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </span> </span> <h3 class="title" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> </a>"), r.fl()
}), Whiskers("entities.shoutout", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="header module_header"> '), r.s(r.f("logo", e, t, 1), e, t, 0, 64, 84, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<img src="'), n.b(n.v(n.f("logo", e, t, 0))), n.b('">')
    }), e.pop()), r.b(" <h2>"), r.b(r.v(r.f("question", e, t, 0))), r.b('</h2> </div> <div class="left"> '), r.s(r.f("logo", e, t, 1), e, t, 0, 151, 171, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<img src="'), n.b(n.v(n.f("logo", e, t, 0))), n.b('">')
    }), e.pop()), r.b(' </div> <div class="right"> <form> <div class="fields"> <div class="words"> <textarea name="shoutout" maxlength="70" placeholder="What do you have to say?" required></textarea> <span class="remaining">50</span> </div> <div class="nas"> <div class="input-group name"> <input type="text" name="name" maxlength="20" required placeholder="First name"></input> </div> <div class="input-group age"> <input type="text" name="age" maxlength="3" required placeholder="Age"></input> </div> <div class="input-group state"> <select name="state" class="empty" required> <option value="" selected>State</option> <option value="AL">AL</option> <option value="AK">AK</option> <option value="AZ">AZ</option> <option value="AR">AR</option> <option value="CA">CA</option> <option value="CO">CO</option> <option value="CT">CT</option> <option value="DE">DE</option> <option value="DC">DC</option> <option value="FL">FL</option> <option value="GA">GA</option> <option value="HI">HI</option> <option value="ID">ID</option> <option value="IL">IL</option> <option value="IN">IN</option> <option value="IA">IA</option> <option value="KS">KS</option> <option value="KY">KY</option> <option value="LA">LA</option> <option value="ME">ME</option> <option value="MD">MD</option> <option value="MA">MA</option> <option value="MI">MI</option> <option value="MN">MN</option> <option value="MS">MS</option> <option value="MO">MO</option> <option value="MT">MT</option> <option value="NE">NE</option> <option value="NV">NV</option> <option value="NH">NH</option> <option value="NJ">NJ</option> <option value="NM">NM</option> <option value="NY">NY</option> <option value="NC">NC</option> <option value="ND">ND</option> <option value="OH">OH</option> <option value="OK">OK</option> <option value="OR">OR</option> <option value="PA">PA</option> <option value="RI">RI</option> <option value="SC">SC</option> <option value="SD">SD</option> <option value="TN">TN</option> <option value="TX">TX</option> <option value="UT">UT</option> <option value="VT">VT</option> <option value="VA">VA</option> <option value="WA">WA</option> <option value="WV">WV</option> <option value="WI">WI</option> <option value="WY">WY</option> <option value="Other">Other</option> </select> </div> </div> <button type="submit" class="button blue small">Submit</button> <p class="notification"></p> </div> </form> <div class="list"></div> <div class="show_more_container"> <div class="show_more blue button large" role="button" onclick=""> <span class="label">Show More</span> <span class="spinner"></span> </div> </div> </div> </div>'
    ), r.fl()
}), Whiskers("entities.shoutout.message", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<li class="message'), r.s(r.f("pending", e, t, 1), e, t, 0, 30, 38, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" pending")
    }), e.pop()), r.b('"> <q>'), r.b(r.v(r.f("text", e, t, 0))), r.b("</q> <span> "), r.s(r.f("dname", e, t, 1), e, t, 0, 86, 107, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.b(n.v(n.f("dname", e, t, 0))), n.b("</address> ")
    }), e.pop()), r.b(" "), r.s(r.f("name", e, t, 1), e, t, 0, 127, 182, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" <strong>"), n.b(n.v(n.f("name", e, t, 0))), n.b("</strong> Age "), n.b(n.v(n.f("age", e, t, 0))), n.b(", from "), n.b(n.v(n.f("state", e, t, 0))), n.b(" ")
    }), e.pop()), r.b(" </span> </li>"), r.fl()
}), Whiskers("entities.show", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b('> <span class="logo-box'), r.s(r.f("channelSite", e, t, 1), e, t, 0, 96, 104, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" channel")
    }), e.pop()), r.b('"><span class="aspect"> <img src="'), r.b(r.v(r.f("logo", e, t, 0))), r.b('" class="logo" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> </span></span> <h3 class="title" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> </a>"), r.fl()
}), Whiskers("entities.upload_stunt.step1", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="content-wrapper"> <div class="content"> <h3>'), r.b(r.v(r.f("start_header", e, t, 0))), r.b('</h3> <p class="description">'), r.b(r.v(r.f("start_description", e, t, 0))), r.b('</p> <div class="button gold large enter-contest">'), r.b(r.v(r.f("start_button_text", e, t, 0))), r.b('</div> <div class="footnotes"> '), r.b(r.t(r.f("start_footnotes", e, t, 0))), r.b(" </div> </div> "), r.s(r.f("image_url", e, t, 1), e, t, 0, 274, 321, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img src="'), n.b(n.v(n.f("image_url", e, t, 0))), n.b('" class="image-asset"> ')
    }), e.pop()), r.b(" </div>"), r.fl()
}), Whiskers("entities.upload_stunt.step2", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="content-wrapper"> '), r.s(r.f("image_url", e, t, 1), e, t, 0, 44, 91, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img src="'), n.b(n.v(n.f("image_url", e, t, 0))), n.b('" class="image-asset"> ')
    }), e.pop()), r.b(' <div class="content"> <div class="content-container left"> <form action="'), r.b(r.v(r.f("form_action", e, t, 0))), r.b('"> '), r.s(r.f("fields", e, t, 1), e, t, 0, 208, 1369, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <fieldset class="info-fs"> <legend>Submission Info</legend> '), n.s(n.f("name", e, t, 1), e, t, 0, 278, 361, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <input type="text" name="name" maxlength="20" required placeholder="'), n.b(n.v(n.f("label", e, t, 0))), n.b('" /> ')
        }), e.pop()), n.b(" "), n.s(n.f("age", e, t, 1), e, t, 0, 379, 747, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("select", e, t, 1), e, t, 0, 391, 588, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <div class="input-group age select"> <select name="age" class="empty" required> <option value="">'), n.b(n.v(n.f("label", e, t, 0))), n.b("</option> "), n.s(n.f("options", e, t, 1), e, t, 0, 520, 558, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(' <option value="'), n.b(n.v(n.d(".", e, t, 0))), n.b('">'), n.b(n.v(n.d(".", e, t, 0))), n.b("</option> ")
                }), e.pop()), n.b(" </select> </div> ")
            }), e.pop()), n.b(" "), n.s(n.f("select", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div class="input-group age input"> <input type="text" name="age" maxlength="3" required placeholder="'), n.b(n.v(n.f("label", e, t, 0))), n.b('" /> </div> ')), n.b(" ")
        }), e.pop()), n.b(" "), n.s(n.f("state", e, t, 1), e, t, 0, 766, 960, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="input-group state"> <select name="state" class="empty" required> <option value="">'), n.b(n.v(n.f("label", e, t, 0))), n.b("</option> "), n.s(n.f("options", e, t, 1), e, t, 0, 892, 930, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <option value="'), n.b(n.v(n.d(".", e, t, 0))), n.b('">'), n.b(n.v(n.d(".", e, t, 0))), n.b("</option> ")
            }), e.pop()), n.b(" </select> </div> ")
        }), e.pop()), n.b(" </fieldset> "), n.s(n.f("extra", e, t, 1), e, t, 0, 993, 1358, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <fieldset class="extra-fs"> '), n.s(n.f("select", e, t, 1), e, t, 0, 1033, 1245, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <div class="input-group extra select"> <select name="'), n.b(n.v(n.f("tag", e, t, 0))), n.b('" class="empty" required> <option value="" selected>'), n.b(n.v(n.f("label", e, t, 0))), n.b("</option> "), n.s(n.f("options", e, t, 1), e, t, 0, 1177, 1215, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(' <option value="'), n.b(n.v(n.d(".", e, t, 0))), n.b('">'), n.b(n.v(n.d(".", e, t, 0))), n.b("</option> ")
                }), e.pop()), n.b(" </select> </div> ")
            }), e.pop()), n.b(" "), n.s(n.f("select", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div class="input-group extra"> '), n.b(n.t(n.f("character_countdown", e, t, 0))), n.b(" </div> ")), n.b(" </fieldset> ")
        }), e.pop()), n.b(" ")
    }), e.pop()), r.b(' <div class="file-name"></div> <div class="button silver small select-file"> '), r.b(r.v(r.f("select_file_button_text", e, t, 0))), r.b(' <input type="file" name="file" required /> </div> <div class="button gold small begin-upload"> '), r.b(r.v(r.f("upload_file_button_text", e, t, 0))), r.b(' </div> <div class="notification"></div> <input type="hidden" name="campaign_id" value="'), r.b(r.v(r.f("campaign_id", e, t, 0))), r.b('"> </form> <div class="footnotes">'), r.b(r.t(r.f("form_footnotes", e, t, 0))), r.b('</div> </div> <div class="content-container right"> '), r.s(r.f("has_tips", e, t, 1), e, t, 0, 1829, 2117, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" <h4>"), n.b(n.v(n.f("helpful_tip_header", e, t, 0))), n.b('</h4> <div class="tips"> <div class="tips-container left"> <h5>DO</h5> <ul> '), n.s(n.f("dos", e, t, 1), e, t, 0, 1940, 1956, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <li>"), n.b(n.v(n.d(".", e, t, 0))), n.b("</li> ")
        }), e.pop()), n.b(' </ul> </div> <div class="tips-container right"> <h5>DON\'T</h5> <ul> '), n.s(n.f("donts", e, t, 1), e, t, 0, 2043, 2059, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <li>"), n.b(n.v(n.d(".", e, t, 0))), n.b("</li> ")
        }), e.pop()), n.b(" </ul> </div> </div> "), n.b(n.t(n.f("helpful_tip_footnote", e, t, 0))), n.b(" ")
    }), e.pop()), r.b(' </div> </div> </div> <div class="progress"> <div class="center"> <div>Uploading</div> <div class="spinner"></div> <div class="split"></div> <div class="bar-container"> <div class="bar"></div> </div> <div class="button black large cancel-upload">Cancel Upload</div> </div> </div>'), r.fl()
}), Whiskers("entities.upload_stunt.step3", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="content-wrapper"> '), r.s(r.f("background_image", e, t, 1), e, t, 0, 51, 110, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img src="'), n.b(n.v(n.f("background_image", e, t, 0))), n.b('" class="background-image"> ')
    }), e.pop()), r.b(" "), r.s(r.f("image_url", e, t, 1), e, t, 0, 146, 193, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img src="'), n.b(n.v(n.f("image_url", e, t, 0))), n.b('" class="image-asset"> ')
    }), e.pop()), r.b(' <div class="content"> <h3>'), r.b(r.v(r.f("thank_you_header", e, t, 0))), r.b('</h3> <p class="description">'), r.b(r.v(r.f("thank_you_text", e, t, 0))), r.b("</p> </div> </div>"), r.fl()
}), Whiskers("entities.upload_stunt.step4", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="content-wrapper"> '), r.s(r.f("image_url", e, t, 1), e, t, 0, 44, 91, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img src="'), n.b(n.v(n.f("image_url", e, t, 0))), n.b('" class="image-asset"> ')
    }), e.pop()), r.b(' <div class="content"> '), r.s(r.f("message", e, t, 1), e, t, 0, 140, 197, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" <h3>"), n.b(n.v(n.f("header", e, t, 0))), n.b('</h3> <p class="description">'), n.b(n.v(n.f("body", e, t, 0))), n.b("</p> ")
    }), e.pop()), r.b(" "), r.s(r.f("url", e, t, 1), e, t, 0, 218, 279, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('" class="checkout">Checkout with Paypal</a> ')
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), Whiskers("entities.uploadstunt", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.fl()
}), Whiskers("entities.video", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"'), r.s(r.f("ping", e, t, 1), e, t, 0, 27, 48, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
    }), e.pop()), r.b(' class="'), r.s(r.f("item_titles_below", e, t, 1), e, t, 1, 0, 0, "") || r.b("skip-styles"), r.b('"> <span class="thumb-box"><span class="aspect"> <img src="'), r.b(r.v(r.f("thumb", e, t, 0))), r.b('" class="thumb" alt="'), r.b(r.v(r.f("title", e, t, 0))), r.b('"> '), r.s(r.f("badge", e, t, 1), e, t, 0, 231, 273, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<img src="'), n.b(n.v(n.f("badge", e, t, 0))), n.b('" class="badge" alt="">')
    }), e.pop()), r.b(' <span class="thumb-title"> <h3 class="overlay-title" aria-hidden="true">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> <h4>"), r.b(r.v(r.f("ptitle", e, t, 0))), r.b("</h4> "), r.s(r.f("duration", e, t, 1), e, t, 0, 404, 450, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <time aria-hidden="true">'), n.b(n.v(n.f("duration", e, t, 0))), n.b("</time> ")
    }), e.pop()), r.b(' </span> </span></span> <div class="duplicate-title"> <h3>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h3> "), r.s(r.f("ptitle", e, t, 1), e, t, 0, 547, 568, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" <h4>"), n.b(n.v(n.f("ptitle", e, t, 0))), n.b("</h4> ")
    }), e.pop()), r.b(" </div> </a>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = t.placeSwap = function(e, t) {
        if (!t)
            return function(t) {
                t && n(e, t)
            };
        t.render(), e.$("#" + t.cid).replaceWith(t.el)
    };
    t.placeHtml = function(e) {
        return e ? '<div id="' + e.cid + '"></div>' : null
    }
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Backbone, i = e.Whiskers, s = e.Math, o = e.encodeURIComponent, u = e._, a = e.jQuery, f = t.Style.breakpoints, l = e.Tracker, c = u.values(f);
    n.ModuleView.register("all", {
        className: "module stream-view",
        template: i.modules.all,
        breakpoints: c.join(" "),
        page: 1,
        events: {
            "click .show_more": function(e) {
                var t = this, n=++t.page * t.cols() * this.rows();
                e.preventDefault(), t.model.data.ensure(n).then(function() {
                    t.fit(), l.pagination(t.page)
                })
            },
            "click .filters a.filter": function(e) {
                var t = this;
                e.preventDefault(), t.$(".filters a.filter").toggleClass("active");
                var n = a(".filter_list").height() + 44;
                t.$(".filter_drawer").hasClass("active") ? t.$(".filter_drawer").css("height", "0").removeClass("active") : t.$(".filter_drawer").css("height", n).addClass("active")
            }
        },
        initialize: function() {
            var e = this, t = e.model, r = t.data;
            e.entityType = u.first(r.models).get("entityType") || t.get("type"), this.list = new n.ListView({
                collection: r,
                template: {
                    render: function(t) {
                        return i.modules.col.render(t, {
                            entity: i.entities[e.entityType]
                        })
                    }
                }
            }), r.on("loading", function(t, n) {
                e.updateButton(n)
            })
        },
        render: function() {
            var e = this.list, n = this.model.data, r = u.extend({
                entityType: this.entityType,
                list: t.placeHtml(e),
                hasMore: n.remaining() > 0,
                singleRow: n.count <= this.cols()
            }, this.model);
            this.$el.html(this.template.render(r, {
                module_header: i.partials.module_elements.module_header,
                iconHeader: i.partials.icon_header
            })), t.Style.moduleStyle(this), t.placeSwap(this, e), this.updateButton(!1)
        },
        rows: function() {
            var e = r.Resize.width();
            return e >= f.max ? 4 : e >= f.midHigh ? 4 : 5
        },
        cols: function() {
            var e = r.Resize.width(), t = this.entityType;
            switch (t) {
            case"movie":
            case"console_game":
                return e >= f.max ? 5 : e >= f.midHigh ? 4 : e >= f.midLow ? 3 : 2;
            case"mobile_app":
                return e >= f.max ? 5 : e >= f.midHigh ? 4 : e >= f.midLow ? 3 : 1;
            case"video":
            case"game":
                return e >= f.max ? 4 : e >= f.midHigh ? 3 : 1;
            case"show":
                return e >= f.max ? 4 : e >= f.midHigh ? 3 : 1;
            default:
                return e >= f.max ? 4 : e >= f.midHigh ? 3 : 1
            }
        },
        getPlatforms: function() {
            var e = this, t = e.model, n = t.data, r = [];
            return u.each(n.models, function(e) {
                a.merge(r, e.attributes.platforms || [])
            }, this), u.uniq(r).sort()
        },
        getGameTypes: function() {
            var e = this, n = e.model, r = n.data, i = [];
            return u.each(r.models, function(e) {
                i.push(e.attributes.game_type)
            }, this), u.uniq(i).length === 1 && t.portal === "games.disney.com" ? i[0] : "All"
        },
        ready: function() {
            this.gameFilterType = this.getGameTypes(), this.createUI()
        },
        createUI: function() {
            var e = this, n = e.model.get("filters"), r = this.$(".filter_list"), i = this.$(".mobile_filter_list"), s = this.model.get("all_button_text"), o = e.getPlatforms();
            n = t.portal === "games.disney.com" ? o : n, n && n.length ? (r.append('<a class="active" href="" data-filter="All"><span>All ' + s + "</span></a>"), this.$(".filters a.filter span").replaceWith("<span>All " + s + "</span>"), this.$(".filter_mobile span").replaceWith("<span>All " + s + "</span>"), a.each(n, function(e) {
                var t = n[e];
                r.append('<a href="" data-filter="' + t + '"><span>' + t + "</span></a>"), i.append('<option value="' + (e + 1) + '" data-filter="' + t + '">' + t + "</option>")
            }), this.$(".filter_drawer a").click(function(t) {
                t.preventDefault(), e.$(".filter_drawer a").removeClass("active"), a(this).addClass("active"), e.$(".filter_drawer").css("height", "0").removeClass("active"), e.$(".filters a.filter").removeClass("active"), a(this).data("filter") === "All" ? (e.$(".filters a.filter span").replaceWith("<span>" + a(this).data("filter") + " " + s + "</span>"), e.$(".filter_mobile span").replaceWith("<span>" + a(this).data("filter") + " " + s + "</span>")) : (e.$(".filters a.filter span").replaceWith("<span>" + a(this).data("filter") + "</span>"), e.$(".filter_mobile span").replaceWith("<span>" + a(this).data("filter") + "</span>")), e.hideStreamItems()
            }), this.$(".mobile_filter_list").change(function() {
                e.$(".mobile_filter_list option:selected").each(function() {
                    a(this).data("filter") === "All" ? (e.$(".filters a.filter span").replaceWith("<span>" + a(this).data("filter") + " " + s + "</span>"), e.$(".filter_mobile span").replaceWith("<span>" + a(this).data("filter") + " " + s + "</span>")) : (e.$(".filters a.filter span").replaceWith("<span>" + a(this).data("filter") + "</span>"), e.$(".filter_mobile span").replaceWith("<span>" + a(this).data("filter") + "</span>")), e.$(".mobile_filter_list").blur(), e.hideStreamItems()
                })
            }), r.css("display", "")) : (r.hide(), this.$(".filters a.filter").hide(), this.$(".filter_mobile").hide()), this.resize()
        },
        hideStreamItems: function() {
            var t = this, n = this.$("#stream_items ul"), r = this.$("#stream_items"), i = n.outerHeight(), s = a(e).selectTransitionEnd() + ".stream_items";
            n.css("height", i).addClass("hide"), r.addClass("load"), a.support.css3("transition") ? n.on(s, function(e) {
                this === e.target && (t.fetch(), n.off(s))
            }) : t.fetch()
        },
        fetch: function() {
            var e = this, n = r.Resize.width(), i = e.list.collection, s = this.$(".filter_drawer a.active").data("filter") || "All", u = this.$("select.mobile_filter_list option:selected").data("filter") || "All", a;
            n < f.midHigh ? a = "/all-filter.json?f=" + o(u) : (s.replace(/'?'/g, ""), a = "/all-filter.json?f=" + o(s)), t.portal === "games.disney.com" && (a = a + "&g=" + e.gameFilterType), e.page = 1, i.reset(), i.url = a;
            var l = e.page * e.cols() * this.rows();
            i.ensure(l).then(function() {
                e.$("#stream_items").removeClass("load"), e.$("#stream_items ul").css("height", "auto").removeClass("hide")
            })
        },
        updateButton: function(e) {
            var t = this.$el, n = this.model.data;
            t.toggleClass("loading", e), t.toggleClass("has-more", n.remaining() > 0)
        },
        resize: function() {
            var e = this.cols(), t = e > 1 ? "grid-layout": "list-layout";
            this.page = s.floor(this.model.data.length / this.rows() / e) || 1, this.list.$el.attr("class", "col-container cols-" + e + " " + t), this.fit()
        },
        fit: function() {
            var e = this.list.collection.count, t = this.page * this.cols() * this.rows();
            this.model.data.remaining() > 0 ? t < e ? (this.list.$el.children().show().slice(t).hide(), this.$(".col").removeClass("last-row"), this.list.$el.children().slice(t - this.cols()).addClass("last-row")) : this.list.$el.children().show() : (this.list.$el.children().show(), this.$(".col").removeClass("last-row"))
        }
    })
}(this), Whiskers("modules.article", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 1330, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound"> <div class="article_container" itemscope="" itemtype="http://schema.org/BlogPosting"> <img src="'), n.b(n.v(n.f("asset_image", e, t, 0))), n.b('" class="article_mobile_image" /> '), n.s(n.f("author_name", e, t, 1), e, t, 0, 191, 520, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="author_info"> <span class="author_img"> <div class="scaler"></div> <img src="'), n.b(n.v(n.f("author_image", e, t, 0))), n.b('" /> </span> <div class="author"> <p class="posted_by">Posted By</p> <p class="author_name">'), n.b(n.v(n.f("author_name", e, t, 0))), n.b('</p> </div> <time class="post_date" datetime="'), n.b(n.v(n.f("schema_date", e, t, 0))), n.b('" itemprop="datePublished">'), n.b(n.v(n.f("post_date", e, t, 0))), n.b("</time> </div> ")
        }), e.pop()), n.b(' <div class="article_text"> <div class="header module_header"><h2 class="blog_title title" itemprop="name headline">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h2></div> <div class="article_body" itemprop="articleBody"> '), n.b(n.t(n.f("article_text", e, t, 0))), n.b(' </div> </div> <div class="author_info_mobile"> <span class="author_img"> <img src="'), n.b(n.v(n.f("author_image", e, t, 0))), n.b('"/> </span> <div class="author"> <p class="posted_by">Posted By</p> <p class="author_name">'), n.b(n.v(n.f("author_name", e, t, 0))), n.b('</p> </div> <div class="post_date">'), n.b(n.v(n.f("post_date", e, t, 0))), n.b('</div> </div> <div class="article_image_container"> <div class="caption_img"> <img src="'), n.b(n.v(n.f("asset_image", e, t, 0))), n.b('" /> '), n.s(n.f("image_caption", e, t, 1), e, t, 0, 1121, 1171, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<div class="image_caption">'), n.b(n.v(n.f("image_caption", e, t, 0))), n.b("</div>")
        }), e.pop()), n.b(' </div> <div class="artAd_medRect ad-container"> '), n.b(n.t(n.f("rectangleAd", e, t, 0))), n.b(" </div> </div> </div> "), n.b(" "), n.b(n.t(n.f("mobileAd", e, t, 0))), n.b(" </div> ")
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e._, i = t.ModuleView.extend({
        render: function() {
            var e = r.extend({}, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(e)), n.Style.moduleStyle(this)
        }
    });
    i.register("article", {
        template: e.Whiskers.modules.article
    })
}(this), Whiskers("modules.article_stream", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class ="bound"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 37, 55, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(" <ul>"), r.s(r.f("data", e, t, 1), e, t, 0, 85, 94, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("list", e, t, ""))
    }), e.pop()), r.b("</ul> </div>"), r.fl()
}), Whiskers("partials.article_stream_list", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<li class="article_container"> <div class="image_container"> <a href = "'), r.b(r.v(r.f("href", e, t, 0))), r.b('"><img src="'), r.b(r.v(r.f("asset_image", e, t, 0))), r.b('"></img></a> </div> <div class="article_text_container"> <div class="header_article"> <a href = "'), r.b(r.v(r.f("href", e, t, 0))), r.b('"> <h2>'), r.b(r.v(r.f("title", e, t, 0))), r.b('</h2> </a> </div> <p class="articletext">'), r.b(r.v(r.f("article_excerpt", e, t, 0))), r.b(' <a href = "'), r.b(r.v(r.f("href", e, t, 0))), r.b('">'), r.b(r.v(r.d("translations.continue_reading", e, t, 0))), r.b('</a> </p> <div class ="articlefooter"> <div class="author_info"> <img src="'), r.b(r.v(r.f("author_image", e, t, 0))), r.b('" class="circle"></img> </div> <p class ="postdate">'), r.b(r.v(r.d("translations.posted", e, t, 0))), r.b(" "), r.b(r.v(r.f("post_date_diff", e, t, 0))), r.b("</p> "), r.b(" </div> </div> </li>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e._, i = e.Whiskers, s = n.ModuleView.extend({
        initialize: function() {
            var e = this.model.get("translations"), t = {
                continue_reading: "Continue Reading",
                posted: "Posted"
            };
            r.defaults(e, t), r.extend(this, {
                translations: e
            })
        },
        render: function() {
            var e = r.extend(this.model, {
                translations: this.translations
            });
            this.$el.html(this.template.render(e, {
                module_header: i.partials.module_elements.module_header,
                list: i.partials.article_stream_list
            })), t.Style.moduleStyle(this)
        }
    });
    s.register("article_stream", {
        className: "module full-height",
        template: i.modules.article_stream
    })
}(this), Whiskers("modules.artist", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 380, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound"> <div class="container"> <div class="branding"> <img src="'), n.b(n.v(n.f("hero", e, t, 0))), n.b('"> </div> <div class="details '), n.s(n.f("style", e, t, 1), e, t, 0, 135, 156, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.v(n.f("module_font_color", e, t, 0)))
        }), e.pop()), n.b('"> <h1>'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h1> <div class="headshot"> <span class="personality-box"><span class="aspect"> <img src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </span></span> </div> <p class="desc">'), n.b(n.v(n.f("description", e, t, 0))), n.b("</p> </div> </div> </div> ")
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e.Backbone, u = e._, a = t.ModuleView.extend({
        className: "module skip-bottom-border full-height full-width",
        render: function() {
            var e = u.first(this.model.get("data")), t = o.Resize.width() >= r.midHigh, i = u.extend({
                posterClass: t && "wide",
                posterImg: t ? e.poster: e.poster_mobile
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(i)), n.Style.moduleStyle(this, !1)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    });
    a.register("artist", {
        breakpoints: i,
        template: s.modules.artist
    })
}(this), Whiskers("modules.catalog", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 2687, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound"> <div class="container" data-title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <div class="title"> <h1>'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h1> <p class="desc"> '), n.b(n.v(n.f("desc", e, t, 0))), n.b(" "), n.s(n.f("price", e, t, 1), e, t, 0, 152, 188, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<span class="price">'), n.b(n.v(n.f("price", e, t, 0))), n.b("</span>")
        }), e.pop()), n.b(" </p> "), n.s(n.f("watch_message", e, t, 1), e, t, 0, 222, 314, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="watch_message"> <div class="watch_message_text">'), n.b(n.v(n.f("watch_message", e, t, 0))), n.b("</div> </div> ")
        }), e.pop()), n.b(" "), n.s(n.f("meta", e, t, 1), e, t, 0, 342, 807, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<p class="meta"> '), n.s(n.f("rating", e, t, 1), e, t, 0, 370, 436, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" <span><strong>"), n.b(n.v(n.d("translations.rated", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("rating", e, t, 0))), n.b("</span> ")
            }), e.pop()), n.b(" "), n.s(n.f("release", e, t, 1), e, t, 0, 460, 673, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" <span><strong>"), n.s(n.f("reRelease", e, t, 1), e, t, 0, 489, 507, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(n.v(n.f("reReleasedText", e, t, 0)))
                }), e.pop()), n.s(n.f("reRelease", e, t, 1), e, t, 1, 0, 0, "") || (n.b(n.v(n.d("translations.released", e, t, 0))), n.b(":")), n.b("</strong> "), n.s(n.f("reRelease", e, t, 1), e, t, 1, 0, 0, "") || n.b(n.v(n.f("release", e, t, 0))), n.s(n.f("reRelease", e, t, 1), e, t, 0, 638, 651, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(n.v(n.f("reRelease", e, t, 0)))
                }), e.pop()), n.b("</span> ")
            }), e.pop()), n.b(" "), n.s(n.f("platform_data", e, t, 1), e, t, 0, 704, 784, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" <span><strong>"), n.b(n.v(n.d("translations.available_on", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("platform_data", e, t, 0))), n.b("</span> ")
            }), e.pop()), n.b(" </p>")
        }), e.pop()), n.b(' <div class="movie-actions-alt"> '), n.s(n.f("retailerData", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.s(n.f("buttons", e, t, 1), e, t, 0, 879, 1011, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button large '), n.b(n.v(n.f("type", e, t, 0))), n.b(" "), n.b(n.v(n.f("button_color", e, t, 0))), n.b('"'), n.s(n.f("convert", e, t, 1), e, t, 0, 957, 984, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' data-convert="'), n.b(n.v(n.f("convert", e, t, 0))), n.b('"')
            }), e.pop()), n.b(">"), n.b(n.v(n.f("title", e, t, 0))), n.b("</a> ")
        }), e.pop()), n.b(" ")), n.b(" "), n.s(n.f("retailerData", e, t, 1), e, t, 0, 1059, 1092, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="retail-link"></div> ')
        }), e.pop()), n.b(' </div> </div> <div class="poster-container'), n.s(n.f("watch_message", e, t, 1), e, t, 0, 1170, 1184, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" watch-padding")
        }), e.pop()), n.b('"> <div class="poster-inner poster-box"> <div class="poster '), n.s(n.f("isMobile", e, t, 1), e, t, 0, 1275, 1283, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("isMobile")
        }), e.pop()), n.b('"> <div class="aspect"><img src="'), n.b(n.v(n.f("catalogImage", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" width="100%"></div> '), n.s(n.f("sponsor_banner", e, t, 1), e, t, 0, 1402, 1486, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<img class="sponsor-banner" src="'), n.b(n.v(n.f("sponsor_banner", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" width="100%">')
        }), e.pop()), n.b(" </div> </div> "), n.s(n.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.s(n.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.b(n.rp("socialLinkPartial", e, t, "")), n.b(" ")), n.b(" ")), n.b(" </div> "), n.s(n.f("wide", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div class="desc-alt"> <p class="desc'), n.s(n.f("watch_message", e, t, 1), e, t, 0, 1675, 1689, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" watch-padding")
        }), e.pop()), n.b('"> '), n.b(n.v(n.f("desc", e, t, 0))), n.b(" "), n.s(n.f("price", e, t, 1), e, t, 0, 1729, 1766, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<span class="price">$'), n.b(n.v(n.f("price", e, t, 0))), n.b("</span>")
        }), e.pop()), n.b(" </p> "), n.s(n.f("watch_message", e, t, 1), e, t, 0, 1800, 1892, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="watch_message"> <div class="watch_message_text">'), n.b(n.v(n.f("watch_message", e, t, 0))), n.b("</div> </div> ")
        }), e.pop()), n.b(' <p class="meta"> '), n.s(n.f("rating", e, t, 1), e, t, 0, 1939, 2005, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <span><strong>"), n.b(n.v(n.d("translations.rated", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("rating", e, t, 0))), n.b("</span> ")
        }), e.pop()), n.b(" "), n.s(n.f("release", e, t, 1), e, t, 0, 2029, 2225, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <span><strong>"), n.s(n.f("reRelease", e, t, 1), e, t, 0, 2058, 2076, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(n.v(n.f("reReleasedText", e, t, 0)))
            }), e.pop()), n.s(n.f("reRelease", e, t, 1), e, t, 1, 0, 0, "") || n.b("Released:"), n.b("</strong> "), n.s(n.f("reRelease", e, t, 1), e, t, 1, 0, 0, "") || n.b(n.v(n.f("release", e, t, 0))), n.s(n.f("reRelease", e, t, 1), e, t, 0, 2190, 2203, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(n.v(n.f("reRelease", e, t, 0)))
            }), e.pop()), n.b("</span> ")
        }), e.pop()), n.b(" "), n.s(n.f("platform_data", e, t, 1), e, t, 0, 2256, 2336, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <span><strong>"), n.b(n.v(n.d("translations.available_on", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("platform_data", e, t, 0))), n.b("</span> ")
        }), e.pop()), n.b(" </p> </div> ")), n.b(' <div class="movie-actions"> '), n.s(n.f("retailerData", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.s(n.f("buttons", e, t, 1), e, t, 0, 2435, 2567, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button large '), n.b(n.v(n.f("type", e, t, 0))), n.b(" "), n.b(n.v(n.f("button_color", e, t, 0))), n.b('"'), n.s(n.f("convert", e, t, 1), e, t, 0, 2513, 2540, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' data-convert="'), n.b(n.v(n.f("convert", e, t, 0))), n.b('"')
            }), e.pop()), n.b(">"), n.b(n.v(n.f("title", e, t, 0))), n.b("</a> ")
        }), e.pop()), n.b(" ")), n.b(" "), n.s(n.f("retailerData", e, t, 1), e, t, 0, 2615, 2648, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="retail-link"></div> ')
        }), e.pop()), n.b(" </div> </div> </div> ")
    }), e.pop()), r.fl()
}), Whiskers("shared.productbutton", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="purchase_product"> <a class="button large green '), r.s(r.f("retail_links", e, t, 1), e, t, 1, 0, 0, "") || r.b("clear"), r.b('"> '), r.s(r.f("retail_links", e, t, 1), e, t, 1, 0, 0, "") || r.b('<div class="arrow-left"></div>'), r.b(" "), r.s(r.f("purchase_cta", e, t, 1), e, t, 0, 184, 200, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.v(n.f("purchase_cta", e, t, 0)))
    }), e.pop()), r.b(" "), r.s(r.f("purchase_cta", e, t, 1), e, t, 1, 0, 0, "") || (r.b(" "), r.s(r.f("retail_links", e, t, 1), e, t, 0, 253, 270, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.v(n.f("pre_order_now", e, t, 0)))
    }), e.pop()), r.b(" "), r.s(r.f("retail_links", e, t, 1), e, t, 1, 0, 0, "") || r.b(r.v(r.f("coming_soon", e, t, 0))), r.b(" ")), r.b(" "), r.s(r.f("retail_links", e, t, 1), e, t, 1, 0, 0, "") || r.b('<div class="arrow-right"></div>'), r.b(' </a> <div class="buy '), r.s(r.f("retail_links", e, t, 1), e, t, 1, 0, 0, "") || r.b("no_flyout"), r.b('" data-convert="buy"> <ul> '), r.s(r.f("retail_links", e, t, 1), e, t, 0, 530, 691, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li><a href="'), n.b(n.v(n.f("content_url", e, t, 0))), n.b('" target="_blank" '), n.s(n.f("age_gate", e, t, 1), e, t, 0, 590, 632, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' class="age-gated" data-age="'), n.b(n.v(n.f("age_gate", e, t, 0))), n.b('"')
        }), e.pop()), n.b(' data-retailer="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></li> ")
    }), e.pop()), r.b(" </ul> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Backbone, r = e._, i = e.Whiskers;
    t.ProductButtonView = n.View.extend({
        _flyoutVisible: !1,
        flyoutVisible: function(e) {
            if (typeof e == "undefined")
                return this._flyoutVisible;
            this._flyoutVisible = e, this.renderFlyout()
        },
        initialize: function(e) {
            this.translations = e.translations, this.age_gate = e.age_gate, this.render()
        },
        render: function() {
            var e = this, t = r.extend(e.model, e.translations);
            t.age_gate = this.age_gate, this.$el.html(i.shared.productbutton.render(t)), this.$el.click(function() {
                e.flyoutVisible(!e.flyoutVisible()), e.trigger("change", e)
            })
        },
        renderFlyout: function() {
            var e = this.$(".buy");
            this.flyoutVisible() ? e.addClass("active") : e.removeClass("active")
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e._;
    t.BunView.register("catalog", {
        breakpoints: i,
        template: s.modules.catalog,
        render: function() {
            var e = this, t = e.model.data.first(), r = e.model.get("data")[0], i = e.model.get("translations"), u = new Date, a = u.getUTCDate(), f = u.getUTCMonth() + 1, l = u.getUTCFullYear(), c=!1, h;
            a < 10 && (a = "0" + a), f < 10 && (f = "0" + f), u = l + "-" + f + "-" + a;
            var p = (r.re_release_date <= u ? i.re_released : i.re_releasing) + ":";
            r.type === "Game" && (h = r.platforms, h && h.length > 0 && (h = h.join(", ")));
            if (h || r.rating || r.release)
                c=!0;
            var d = o.extend({
                platform_data: h,
                meta: c,
                reReleasedText: p,
                linksInverted: !0,
                cds: !!this.options.cds
            }, e.options.ads.helpers(), e.model);
            e.$el.html(e.template.render(d, {
                socialLinkPartial: s.partials.social_btn
            }));
            var v = t.get("retailerData");
            v && v.retail_links.length && new n.ProductButtonView({
                el: e.$(".retail-link"),
                model: v,
                translations: i
            })
        },
        ready: function() {
            var e = this.model.get("style").background_color ? this.model.get("style").background_color: !1;
            e && this.$el.css({
                "background-color": e
            }), n.FB.go()
        },
        resize: function() {
            this.options.ads.restore()
        }
    })
}(this), Whiskers("modules.character_bio", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("entityType", e, t, 0))), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 51, 69, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="character-bio-list"> <ul> '), r.s(r.f("data", e, t, 1), e, t, 0, 133, 487, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li> <div class="personality-box entity-box"> <div class="aspect"> '), n.s(n.f("thumb", e, t, 1), e, t, 0, 211, 274, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<img src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('" class="personality-image" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">')
        }), e.pop()), n.b(" "), n.s(n.f("thumb", e, t, 1), e, t, 1, 0, 0, "") || (n.b('<img src="'), n.b(n.v(n.f("photo", e, t, 0))), n.b('" class="personality-image" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">')), n.b(' </div> <div class="copy-box"> <h3 class="title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h3> <p class="desc">'), n.b(n.v(n.f("short_desc", e, t, 0))), n.b("</p> </div> </div> </li> ")
    }), e.pop()), r.b(" </ul> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e._, s = t.ModuleView.extend({
        template: r.modules.character_bio,
        render: function() {
            var e = i.extend({}, this.model);
            this.$el.html(this.template.render(e), {
                module_header: r.partials.module_elements.module_header,
                iconHeader: r.partials.icon_header
            }), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render()
        }
    });
    s.register("character_bio", {}), s.register("character_bio_left_align", {})
}(this), Whiskers("modules.contest", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 197, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound'), n.s(n.f("noAd", e, t, 1), e, t, 0, 36, 41, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" noAd")
        }), e.pop()), n.b('"> <div class="container"></div> '), n.s(n.f("data", e, t, 1), e, t, 0, 92, 180, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("noAd", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div class="ad-container"> '), n.b(n.t(n.f("rectangleAd", e, t, 0))), n.b(" "), n.b(n.t(n.f("mobileAd", e, t, 0))), n.b(" </div> ")), n.b(" ")
        }), e.pop()), n.b(" </div> ")
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    function l(e, t) {
        arguments.length < 2 && (t = e>>>0, e = f);
        var n = o.Deferred();
        return a(function() {
            n.resolve(e)
        }, t), n.promise()
    }
    function h(e) {
        return String(e).replace(/\b(\d+)((\.\d+)*)\b/g, function(e, t, n) {
            return (t.charAt(0) > 0&&!(n || ".").lastIndexOf(".") ? t.replace(/(\d)(?=(\d{3})+$)/g, "$1,") : t) + n
        })
    }
    var t = e.Grill, n = e.Whiskers, r = e.Backbone, i = e.Disney, s = e.Tracker, o = e.jQuery, u = e._, a = e.setTimeout
    , f, c = function(e, t) {
        if (!e)
            throw new Error(t)
    }, p = r.View.extend({
        className: "",
        template: n.entities.contest,
        events: {
            "click .master_button .button": "_handleMasterClick",
            "click span.checkbox": "_handleCheckboxClick",
            "click div.item": "_handleTextOrImageClick",
            "click span.vote": "_handleSingleVoteClick",
            "click .again": "_handleVoteAgainClick"
        },
        initialize: function() {
            u.bindAll(this), this._initializeContest()
        },
        render: function() {
            var e = this._generateRenderContext();
            this.$el.html(this.template.render(e))
        },
        _displayAlert: function() {},
        _showVoteConfirmation: function() {},
        _generateRenderContext: function() {
            var e = {}, t = this._contest ? this._contest.state: "pre", n = 1;
            e.title = this.model.get("title"), e.logo = this.model.get("logo"), e.image_poll = this.model.get("image_poll"), e.noAd = this.model.get("noAd"), e.showFour = this.model.get("showFour"), e.one_click = this.model.get("one_click"), e.show_percentage = this.model.get("show_percentage"), e.contestants = [], u.each(this.model.get("contestants"), function(r) {
                var i = {};
                i.id = r.id, i.title = r.title, i.description = r.description, i.text = r.text, i.logo = r.logo, i.type = e.image_poll ? "img" : "text", i.checkboxes = [], i.vote_button = null;
                var s = this._userVotes ? this._userVotes.votes[i.id]: null;
                if (e.one_click) {
                    var o = t !== "voting" ||!s || s.pending || s.available_votes < 1, a = s && s.total_period_votes ? s.total_period_votes: 0, f = s && s.pending ? "Saving...": this.model.get("vote_button_text");
                    i.vote_button = {
                        disabled: o,
                        text: f,
                        votes: a,
                        loading: s && s.pending
                    }
                } else 
                    for (var l = 0; l < n; ++l)
                        if (s) {
                            var c = s.total_period_votes, p = s.available_votes, d = s.pending_votes, v = s.pending?!0 : !1, m = l < c + d, g=!v && l >= c && l < c + d + p;
                            i.checkboxes.push({
                                index: l + 1,
                                checked: m,
                                disabled: !g
                            })
                        } else 
                            i.checkboxes.push({
                                index: l + 1,
                                checked: !1,
                                disabled: !0
                            });
                (t === "voting" || t === "closed" || t === "finished") && this._userVotes && this._userVotes.max_available_votes === 0 && this._results && (i.results = this._results[i.id], i.results && (i.results.percent !== undefined ? i.results.percent_formatted = String(Math.round(i.results.percent * 100)) + "%" : i.results.percent_formatted = "0%", i.results.total !== undefined ? i.results.total_formatted = h(i.results.total) : i.results.total_formatted = "0")), i.checkboxes && i.checkboxes.length > 0 && (i.disabled = u.every(i.checkboxes, function(e) {
                    return e.disabled===!0
                }), i.clickable=!i.disabled), i.vote_button && (i.clickable=!i.vote_button.disabled), e.contestants.push(i)
            }, this), e.single_option = e.contestants.length === 1, t === "pre" ? e.message = {
                type: "pre",
                text: this.model.get("pre_open_text")
            } : t === "voting" ? (e.message = {
                type: "voting",
                text: null
            }, this._userVotes && (this._userVotes.max_available_votes > 0 || this._userVotes.total_period_votes > 0 && (e.message.type = "voted", e.message.text = this.model.get("voted_text")))) : t === "closed" ? e.message = {
                type: "closed",
                text: this.model.get("closed_text")
            } : t === "finished" && (e.message = {
                type: "finished",
                text: this.model.get("finished_text")
            }), e.message&&!e.message.text && (e.message = null);
            if (!e.one_click && t === "voting") {
                var r = e.master_button = {
                    disabled: !0,
                    loading: !1,
                    text: this.model.get("vote_button_text"),
                    complete: !1
                };
                this._userVotes && (this._userVotes.pending ? (r.loading=!0, r.text = "Saving...") : this._userVotes.max_available_votes > 0 ? r.disabled=!1 : this._userVotes.total_period_votes > 0 && (r.complete=!0))
            }
            return e
        },
        _handleMasterClick: function(e) {
            var t = o(e.currentTarget);
            if (t.hasClass("disabled")) {
                this.model.get("requires_registration")&&!this._currentUid;
                return 
            }
            this._registerVotes(), this._trackLink(null, t.text()), e.preventDefault()
        },
        _handleTextOrImageClick: function(e) {
            var t = o(e.currentTarget);
            if (this.model.get("one_click"))
                this._handleSingleVoteClick(e);
            else {
                var n = t.closest(".option");
                if (n.length) {
                    if (n.hasClass("disabled")) {
                        e.preventDefault();
                        return 
                    }
                    var r = n.data("id");
                    if (this._userVotes && this._userVotes.votes[r]) {
                        var i = this._userVotes.votes[r];
                        i.pending_votes === 0 ? this._setCandidatePendingVotes(r, i.available_votes) : this._setCandidatePendingVotes(r, 0), this.render()
                    }
                    this._trackLink(r)
                }
                e.preventDefault()
            }
        },
        _handleCheckboxClick: function(e) {
            if (this.model.get("one_click"))
                return;
            var t = o(e.currentTarget), n = t.closest(".option"), r = n.find(".checkbox"), i = r.not(".disabled"), s = r.filter(".disabled.checked"), u = 0;
            if (t.hasClass("checkbox")) {
                if (t.hasClass("disabled"))
                    return;
                u = t.data("index"), t.hasClass("checked")&&!t.next().hasClass("checked")&&--u, u -= s.length
            } else 
                i.last().hasClass("checked") ? u = 0 : u = s.length - i.length;
            var a = n.data("id");
            this._setCandidatePendingVotes(a, u), this.render(), this._trackLink(a), e.preventDefault()
        },
        _handleSingleVoteClick: function(e) {
            var t = o(e.currentTarget), n = t.closest(".option");
            if (t.hasClass("disabled")) {
                this.model.get("requires_registration") && this._currentUid === null, e.preventDefault();
                return 
            }
            var r = n.data("id");
            this._registerVote(r), e.preventDefault()
        },
        _handleVoteAgainClick: function() {
            this.render()
        },
        _trackLink: function(e, t) {
            if (!e&&!t)
                return;
            var n = this.options, r = ["contest", "mod" + n.moduleView.index, "item" + n.index];
            if (e) {
                var i = this.model.get("contestants"), o = u.find(i, function(t) {
                    return t.id === e
                }), a = u.indexOf(i, o);
                t = t || o.text || o.title, r.push("poll" + a)
            } else 
                r.push("button");
            var f = r.join("/"), l = f + "/" + t.replace(/[^a-z0-9 ]/gi, "").replace(/\s+/g, "-");
            s.link({
                linkName: l,
                linkPosition: f
            })
        },
        _initializeContest: function() {
            this._userVotes = null;
            if (this.model.get("requires_registration") && this._currentUid === null) {
                this.render();
                return 
            }
            this._loadContestInfo().then(this._handleLoadContestInfoSuccess, this._handleLoadContestInfoFailed).then(this.render)
        },
        _handleLoadContestInfoSuccess: function(e) {
            var t = e.results, n = e.user, r = e.contest;
            if (!n)
                throw new Error("Contest Info is missing user field");
            this._storageSaveContestInfo({
                results: t,
                user: n
            }), this._userVotes = this._inflateUserVotes(n), this._results = this._inflateResults(t), this._contest = r
        },
        _handleLoadContestInfoFailed: function() {},
        _handleSubmitVotesSuccess: function(e) {
            var t = this._storageGetContestInfo();
            for (var n in this._userVotes.votes)
                this._userVotes.votes.hasOwnProperty(n) && (this._setCandidatePending(n, !1), this._commitCandidatePendingVotes(n, t ? t.user : null));
            this._storageSaveContestInfo(t), this._results = this._inflateResults(e.results, this._userVotes), this.render(), this._showVoteConfirmation(), this._beginVoteAgainCountdown()
        },
        _handleSubmitVotesError: function() {
            for (var e in this._userVotes.votes)
                this._userVotes.votes.hasOwnProperty(e) && this._setCandidatePending(e, !1);
            this.render(), this._displayAlert("Error submitting votes...")
        },
        _handleSubmitVoteSuccess: function(e) {
            var t = e.candidate_id, n = this._storageGetContestInfo();
            this._setCandidatePending(t, !1), this._commitCandidatePendingVotes(t, n ? n.user : null), this._storageSaveContestInfo(n), this._results = this._inflateResults(e.results, this._userVotes), this.render(), this._showVoteConfirmation(t), this._beginVoteAgainCountdown()
        },
        _handleSubmitVoteError: function(e) {
            var t = e.candidate_id;
            this._setCandidatePendingVotes(t, 0), this._setCandidatePending(t, !1), this.render(), this._displayAlert("Error submitting vote, please try again.")
        },
        _loadContestInfo: function() {
            var e;
            return e = this._storageGetContestInfo(), e ? o.when(e) : this._fetchContestInfo()
        },
        _fetchContestInfo: function() {
            var e = this.model.get("cws_url") + "/voting/tenants/" + encodeURIComponent(this.model.get("cws_tenant")) + "/contests/" + encodeURIComponent(this.model.get("contest_id"));
            return o.ajax({
                type: "get",
                context: this,
                dataType: "json",
                cache: !1,
                url: e,
                xhrFields: {
                    withCredentials: !0
                }
            })
        },
        _submitVotes: function(e) {
            var t = this.model.get("cws_url") + "/voting/tenants/" + encodeURIComponent(this.model.get("cws_tenant")) + "/contests/" + encodeURIComponent(this.model.get("contest_id")) + "/vote", n = (new Date).valueOf(), r = function(e) {
                var t = (new Date).valueOf(), r = t - n - 0;
                return r < 0 ? l(e, r*-1) : e
            };
            return o.ajax({
                type: "post",
                context: this,
                dataType: "json",
                url: t,
                data: {
                    votes: e
                },
                xhrFields: {
                    withCredentials: !0
                }
            }).then(r)
        },
        _beginVoteAgainCountdown: function() {
            this._contest.state === "voting" && this._userVotes.max_available_votes <= 0 && this._contest.time_rules && this._contest.time_rules.period >= 0 && (this._activeVoteAgainTimer = a(this._handleVoteAgainTimeout, this._contest.time_rules.period * 1e3))
        },
        _handleVoteAgainTimeout: function() {
            var e = this;
            this._loadContestInfo().then(this._handleLoadContestInfoSuccess, this._handleLoadContestInfoFailed).then(function() {
                this._userVotes.max_available_votes > 0 && e.$(".again").fadeIn(500)
            })
        },
        _registerVote: function(e) {
            var t = this._userVotes;
            if (!t)
                return !1;
            var n = t.votes[e];
            if (!n)
                return !1;
            if (n.pending_votes)
                return !1;
            var r = this._setCandidatePendingVotes(e, 1);
            if (r===!1)
                return !1;
            this._setCandidatePending(e, !0);
            var i = [String(e)];
            return this.render(), this._submitVotes(i).then(function(t) {
                return t.candidate_id = e, t
            }, function(t) {
                return {
                    candidate_id: e,
                    xhr: t
                }
            }).then(this._handleSubmitVoteSuccess, this._handleSubmitVoteError)
        },
        _registerVotes: function() {
            if (!this._userVotes)
                return !1;
            if (!this._userVotes.pending_votes)
                return this._displayAlert(), !1;
            this._displayAlert("");
            var e = [];
            for (var t in this._userVotes.votes)
                if (this._userVotes.votes.hasOwnProperty(t)) {
                    var n = this._userVotes.votes[t];
                    this._setCandidatePending(t, !0);
                    for (var r = 0; r < n.pending_votes; ++r)
                        e.push(String(t))
                }
            return this.render(), this._submitVotes(e).then(this._handleSubmitVotesSuccess, this._handleSubmitVotesError)
        },
        _inflateUserVotes: function(e) {
            e.pending_votes = 0, e.max_available_votes = e.available_votes, e.pending=!1;
            for (var t in e.votes)
                if (e.votes.hasOwnProperty(t)) {
                    var n = e.votes[t];
                    n.max_available_votes = n.available_votes, n.pending_votes = 0, n.pending=!1
                }
            return e
        },
        _inflateResults: function(e, t) {
            if (!e)
                return null;
            var n = 0, r = 0, i = {};
            for (var s in this._userVotes.votes)
                if (this._userVotes.votes.hasOwnProperty(s)) {
                    var o = e[s] || 0;
                    if (t && t.votes) {
                        var u = t.votes[s] || 0;
                        u && (o += u.total_period_votes)
                    }
                    n += o, i[s] = {
                        total: o
                    }, o > r && (r = o)
                }
            if (n > 0)
                for (var a in i)
                    if (i.hasOwnProperty(a)) {
                        var f = i[a];
                        f.winner = r === f.total, f.percent = f.total / n
                    }
            return i
        },
        _setCandidatePendingVotes: function(e, t) {
            var n = this._userVotes;
            if (!n)
                return !1;
            var r = n.votes[e];
            if (!r)
                return !1;
            var i = t - r.pending_votes;
            if (i > r.available_votes)
                return !1;
            r.pending_votes = t, n.pending_votes += i, n.available_votes -= i;
            for (var s in n.votes)
                n.votes.hasOwnProperty(s) && (r = n.votes[s], r.available_votes = Math.min(n.available_votes, r.max_available_votes - r.pending_votes));
            return !0
        },
        _commitCandidatePendingVotes: function(e, t) {
            var n = this._userVotes, r = n.votes[e], i;
            n.total_votes += r.pending_votes, n.total_period_votes += r.pending_votes, n.max_available_votes -= r.pending_votes, n.pending_votes -= r.pending_votes, t && (t.total_votes = n.total_votes, t.total_period_votes = n.total_period_votes, t.available_votes = n.max_available_votes), r.total_votes += r.pending_votes, r.total_period_votes += r.pending_votes, r.max_available_votes -= r.pending_votes, r.pending_votes = 0, t && (i = t.votes[e], i.total_votes = r.total_votes, i.total_period_votes = r.total_period_votes);
            for (e in n.votes)
                n.votes.hasOwnProperty(e) && (r = n.votes[e], r.max_available_votes = Math.min(n.max_available_votes, r.max_available_votes), c(r.max_available_votes >= r.available_votes, "available_votes is out of sync"), t && (i = t.votes[e], i.available_votes = r.max_available_votes))
            },
        _setCandidatePending: function(e, t) {
            var n = this._userVotes, r = n.votes[e];
            n.pending=!1, r.pending = t;
            for (e in n.votes)
                if (n.votes.hasOwnProperty(e)) {
                    r = n.votes[e];
                    if (r.pending) {
                        n.pending=!0;
                        break
                    }
                }
            return n.pending
        },
        _storageGetContestInfo: function() {
            return null
        },
        _storageSaveContestInfo: function() {
            return null
        }
    }), d = t.ModuleView.extend({
        className: "module contest",
        template: n.modules.contest,
        initialize: function() {
            this.cds=!!this.options.cds, this.contest_views = [], this.index = this.model.collection.indexOf(this.model), this.model.data.each(function(e, t) {
                this.contest_views.push(new p({
                    model: e,
                    index: t,
                    moduleView: this
                }))
            }, this)
        },
        render: function() {
            var e = u.extend({
                iconStyles: this.model.get("style"),
                noAd: !!this.options.cds
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(e, {
                iconHeader: n.partials.icon_header
            }));
            var t = this.$el.find(".container");
            u.each(this.contest_views, function(e) {
                e.render(), t.append(e.$el)
            }, this), i.Style.moduleStyle(this)
        },
        resize: function() {
            this.cds || this.options.ads.restore(), this.render(), this.ready()
        }
    });
    d.register("contest")
}(this), Whiskers("modules.credits", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> '), r.s(r.f("noAd", e, t, 1), e, t, 1, 0, 0, "") || (r.b(" "), r.b(r.t(r.f("rectangleAd", e, t, 0))), r.b(" "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" ")), r.b(' <div class="content container"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 121, 139, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="credits-list"> <div class="role_contain"> <div class="credit_contain"> '), r.s(r.f("roles", e, t, 1), e, t, 0, 249, 513, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="role_group"> <h3>'), n.b(n.v(n.f("role", e, t, 0))), n.b('</h3> <div class="cast_list"> '), n.s(n.f("cast", e, t, 1), e, t, 0, 326, 489, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="cast"> <span class="talent">'), n.b(n.v(n.f("person", e, t, 0))), n.b('</span> <span class="character '), n.b(n.v(n.f("role", e, t, 0))), n.b('"> '), n.s(n.f("title", e, t, 1), e, t, 0, 427, 436, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(n.v(n.f("title", e, t, 0)))
            }), e.pop()), n.b(" "), n.s(n.f("title", e, t, 1), e, t, 1, 0, 0, "") || n.b(n.v(n.f("role", e, t, 0))), n.b(" </span> </p> ")
        }), e.pop()), n.b(" </div> </div> ")
    }), e.pop()), r.b(" </div> </div> </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e.Backbone, u = e._, a = t.ModuleView.extend({
        breakpoints: i,
        template: s.modules.credits,
        getRoles: function() {
            var e = {};
            return u.each(this.model.get("data"), function(t) {
                var n = t.role;
                if (!n)
                    return;
                e[n] || (e[n] = {
                    role: n,
                    cast: []
                }), e[n].cast.push({
                    person: t.person,
                    title: t.title || n
                })
            }), u.values(e)
        },
        render: function() {
            var e = o.Resize.width() >= r.midHigh, t = this.getRoles(), i = u.extend({
                roles: t,
                creditsClass: e && "wide",
                noAd: this.model.get("view") === "credits_no_ad"
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(i, {
                module_header: s.partials.module_elements.module_header
            })), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    });
    a.register("credits", {}), a.register("credits_no_ad", {})
}(this), Whiskers("modules.disclaimer", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="links"> '), r.s(r.f("data", e, t, 1), e, t, 0, 49, 125, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button blue small" target="_blank">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a> ")
    }), e.pop()), r.b(" </div> "), r.s(r.f("desc", e, t, 1), e, t, 0, 151, 186, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<p class="description">'), n.b(n.v(n.f("desc", e, t, 0))), n.b("</p>")
    }), e.pop()), r.b(" </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e._, i = e.Whiskers;
    n.ModuleView.register("disclaimer", {
        template: i.modules.disclaimer,
        render: function() {
            var e = r.extend({}, this.model);
            this.$el.html(this.template.render(e)), t.Style.moduleStyle(this, !1)
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Grill.BunView, n = e.jQuery, r = e._, i = e.Math, s = function() {
        var t = n(e), r = t.height(), s = t.width(), o = i.min(1.3333, i.max(r / s, .5625));
        r = i.round(s * o), r !== this.h && (this.h = r, this.$("iframe.stage").height(r))
    };
    t.register("dressup", {
        initialize: function() {
            n(e).on("resize." + this.cid, r.throttle(r.bind(s, this), 100))
        },
        render: function() {
            var t = n("<iframe/>", {
                "class": "stage",
                src: "/apps/dressup/princess" + e.location.search,
                frameborder: 0,
                scrolling: "no"
            });
            this.$el.empty().append(t)
        },
        ready: s,
        remove: function() {
            t.prototype.remove.call(this), n(e).off("resize." + this.cid)
        }
    }), n(e).on("touchstart", function() {})
}(this), Whiskers("modules.empty_example", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b(' <div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b('"> '), r.b(" "), r.b(" "), r.s(r.f("header_data", e, t, 1), e, t, 0, 379, 397, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(" "), r.b(" "), r.s(r.f("displayImages", e, t, 1), e, t, 0, 629, 878, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="entity-box"><div class="aspect"><img src="'), n.b(n.v(n.f("starterKitteh", e, t, 0))), n.b('"></div></div> '), n.b(" ")
    }), e.pop()), r.b(" "), r.b(' <div class="text-container skip-styles"> '), r.b(' <h2 class="hello_world">'), r.b(r.v(r.d("translations.hello_world", e, t, 0))), r.b("</h2> "), r.b(" "), r.s(r.f("displayImages", e, t, 1), e, t, 0, 1538, 1922, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.b(' <div class="surprise large button '), n.b(n.v(n.f("buttonColor", e, t, 0))), n.b('">'), n.b(n.v(n.d("translations.surprise_me", e, t, 0))), n.b("</div> "), n.b(" ")
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    function h(e, t) {
        return s.floor(s.random() * (t - e) + e)
    }
    function p(e, t) {
        var n = e, r = e >= a.max ? e * .32: e >= a.midHigh ? e * .36: e >= a.midLow ? e * .44: e * .5, i = h(n, n + 20), s = h(r, r + 20), o = t + i + "/" + s;
        return o
    }
    function d(e, t) {
        t.find(".aspect").html(e), t.removeClass("loading")
    }
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e.Backbone, s = e.Math, o = e._, u = e.jQuery, a = n.Style.breakpoints, f = o.values(a), l = t.ModuleView.extend({
        className: "module empty-view",
        template: r.modules.empty_example,
        breakpoints: f.join(" "),
        initialize: function() {
            var e = this.model.get("translations"), t = {
                hello_world: "Oh haiiiiii world!",
                surprise_me: "Surprise Me!"
            };
            o.defaults(e, t), o.extend(this, {
                translations: e
            })
        },
        render: function() {
            var e = o.extend(this.model, {
                translations: this.translations
            });
            this.extendContext && o.extend(e, this.extendContext()), this.$el.html(this.template.render(e, {
                module_header: r.partials.module_elements.module_header,
                iconHeader: r.partials.icon_header
            })), n.Style.moduleStyle(this)
        }
    }), c = l.extend({
        className: "module empty-view empty-image-view",
        rootUrl: "http://placekitten.com/",
        buttonColor: "blue",
        displayImages: !0,
        extendContext: function() {
            var e = this, t = {
                starterKitteh: function() {
                    var t = s.min(i.Resize.width(), 920);
                    return p(t, e.rootUrl)
                },
                buttonColor: e.buttonColor,
                displayImages: e.displayImages
            };
            return t
        },
        events: {
            "click .surprise": function() {
                var t = this.$(".entity-box"), n = new e.Image;
                t.addClass("loading"), n.src = p(t.width(), this.rootUrl), n.complete ? d(n, t) : u(n).on("load", function() {
                    d(n, t)
                })
            }
        }
    });
    l.register("empty_example", {}), c.register("empty_example_image", {}), c.register("empty_example_image_extension", {
        rootUrl: "http://placepuppy.it/",
        buttonColor: "green"
    })
}(this), Whiskers("modules.error", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="error_page" class="e'), r.b(r.v(r.f("error_code", e, t, 0))), r.b(' bound"> <div class="inner"> '), r.s(r.f("style", e, t, 1), e, t, 0, 82, 226, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.s(n.f("images", e, t, 1), e, t, 0, 94, 214, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("foreground_image", e, t, 1), e, t, 0, 116, 192, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <div class="rich_text_image '), n.b(n.v(n.f("alignment", e, t, 0))), n.b('_align"><img src="'), n.b(n.v(n.f("url", e, t, 0))), n.b('"></div> ')
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(" ")
    }), e.pop()), r.b(' <div class="message"> <h1>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</h1> <h2>"), r.b(r.v(r.f("desc", e, t, 0))), r.b("</h2> <div id=error_search> </div> </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.jQuery, i = e.Whiskers, s=!1;
    n.ModuleView.register("error", {
        template: i.modules.error,
        render: function() {
            s = this.model.get("error_code") === 404, this.$el.html(this.template.render(this.model)), t.Style.moduleStyle(this, !1)
        },
        ready: function() {
            if (s) {
                var e = r("footer #utility form");
                e.clone().appendTo(r("#error_page #error_search"))
            }
        }
    })
}(this), Whiskers("modules.featured_gallery", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 536, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound"> <div class="container"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" class="gallery-box gallery featured"> <div class="stack-box"> <div class="aspect"> <img src="'), n.b(n.v(n.f("thumb_src", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </div> </div> <span class="stack-inner top"></span> <span class="stack-inner bot"></span> </a> <div class="header module_header details"> '), n.s(n.f("data", e, t, 1), e, t, 0, 381, 505, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <h2>"), n.b(n.v(n.f("title", e, t, 0))), n.b("</h2> <p>"), n.b(n.v(n.f("description", e, t, 0))), n.b('</p> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button large blue">'), n.b(n.v(n.d("translations.view_slideshow", e, t, 0))), n.b("</a> ")
        }), e.pop()), n.b(" </div> </div> </div> ")
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Whiskers, i = e._;
    n.ModuleView.register("featured_gallery", {
        template: r.modules.featured_gallery,
        render: function() {
            var e = i.first(this.model.get("data")), n = i.extend({
                thumb_src: e.thumbnail || e.images && e.images[0].mobile_image
            }, this.model);
            this.$el.html(this.template.render(n)), t.Style.moduleStyle(this)
        }
    })
}(this), Whiskers("modules.featured_media", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container '), r.b(r.v(r.f("layout", e, t, 0))), r.b('"> '), r.s(r.f("data", e, t, 1), e, t, 0, 64, 427, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.s(n.f("isMobile", e, t, 1), e, t, 0, 104, 111, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" mobile")
        }), e.pop()), n.b('"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <div class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b("-box entity-box "), n.s(n.f("isMobile", e, t, 1), e, t, 0, 202, 210, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("isMobile")
        }), e.pop()), n.b(" "), n.b(n.v(n.f("primaryImageName", e, t, 0))), n.b("-box"), n.s(n.f("isConsole", e, t, 1), e, t, 0, 262, 270, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" console")
        }), e.pop()), n.b('"> <div class="aspect"> <img src="'), n.b(n.v(n.f("imageUrl", e, t, 0))), n.b('" class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b(" "), n.s(n.f("isMobile", e, t, 1), e, t, 0, 367, 376, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" isMobile")
        }), e.pop()), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </div> </div> </a> ')
    }), e.pop()), r.b(' </div> <div class="details '), r.b(r.v(r.f("objectType", e, t, 0))), r.b('-details"> '), r.s(r.f("data", e, t, 1), e, t, 0, 498, 793, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <h2 class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b('_title"><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</a></h2> <p class="desc">'), n.b(n.v(n.f("short_desc", e, t, 0))), n.b('</p> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button large blue more"> '), n.s(n.f("cta_button_title", e, t, 1), e, t, 0, 677, 697, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.v(n.f("cta_button_title", e, t, 0)))
        }), e.pop()), n.b(" "), n.s(n.f("cta_button_title", e, t, 1), e, t, 1, 0, 0, "") || n.b(n.v(n.d("translations.more_info", e, t, 0))), n.b(" </a> ")
    }), e.pop()), r.b(" </div> </div> </div>"), r.fl()
}), Whiskers("modules.featured_game", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container '), r.b(r.v(r.f("layout", e, t, 0))), r.b(' "> '), r.s(r.f("data", e, t, 1), e, t, 0, 65, 341, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b('"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <div class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b("-box entity-box "), n.s(n.f("isMobile", e, t, 1), e, t, 0, 170, 178, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("isMobile")
        }), e.pop()), n.b('"> <div class="aspect"> <img src="'), n.b(n.v(n.f("imageUrl", e, t, 0))), n.b('" class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b(" "), n.s(n.f("isMobile", e, t, 1), e, t, 0, 274, 283, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" isMobile")
        }), e.pop()), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </div> </div> </a> </div> ')
    }), e.pop()), r.b(' <div class="details '), r.b(r.v(r.f("objectType", e, t, 0))), r.b('-details"> '), r.s(r.f("data", e, t, 1), e, t, 0, 405, 576, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <h2 class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b('_title title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h2> <p class="desc">'), n.b(n.v(n.f("short_desc", e, t, 0))), n.b('</p> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button large blue more">'), n.b(n.v(n.d("translations.play_game", e, t, 0))), n.b("</a> ")
    }), e.pop()), r.b(" </div> </div> </div>"), r.fl()
}), Whiskers("modules.featured_movie", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container '), r.b(r.v(r.f("layout", e, t, 0))), r.b('"> '), r.s(r.f("data", e, t, 1), e, t, 0, 64, 280, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b('"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <div class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b('-box poster-box entity-box"> <div class="aspect"> <img src="'), n.b(n.v(n.f("imageUrl", e, t, 0))), n.b('" class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </div> </div> </a> </div> ')
    }), e.pop()), r.b(' <div class="details '), r.b(r.v(r.f("objectType", e, t, 0))), r.b('-details"> '), r.s(r.f("data", e, t, 1), e, t, 0, 344, 509, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <h2 class="'), n.b(n.v(n.f("objectType", e, t, 0))), n.b('_title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h2> <p class="desc">'), n.b(n.v(n.f("short_desc", e, t, 0))), n.b('</p> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button large blue more">'), n.b(n.v(n.d("translations.more_info", e, t, 0))), n.b("</a> ")
    }), e.pop()), r.b(" </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e._, u = t.ModuleView.extend({
        breakpoints: i,
        render: function() {
            var e = this, t = e.model, r = o.first(e.model.get("data")), i = this.$el, s = t.get("type"), u = t.get("style").layout ? t.get("style").layout: "left";
            s === "game" && r.isConsole && (s = "movie");
            var a = o.extend({
                layout: u,
                objectType: s,
                seeAll: t.get("seeAll"),
                imageUrl: t.get("type") === "game" || t.get("type") === "video" ? r.thumb: t.get("type") === "movie" ? r.poster: null
            }, this.options.ads.helpers(), e.model);
            i.addClass("featured_media"), i.html(e.template.render(a)), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    });
    u.register("featured_media", {
        template: s.modules.featured_media
    }), u.register("featured_game", {
        template: s.modules.featured_game
    }), u.register("featured_movie", {
        template: s.modules.featured_movie
    })
}(this), Whiskers("modules.game_logos", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container"> '), r.s(r.f("game_rating", e, t, 1), e, t, 0, 60, 302, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="logo-container"> <span class="esrb"><img src="'), n.b(n.v(n.f("game_rating", e, t, 0))), n.b('"></span> '), n.s(n.f("data", e, t, 1), e, t, 0, 153, 285, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" target="_blank"> <span class="game-logo-box"><img src="'), n.b(n.v(n.f("icon", e, t, 0))), n.b('" class="game-logo" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"></span> </a> ')
        }), e.pop()), n.b(" </div> ")
    }), e.pop()), r.b(' <div class="game_legal"> '), r.b(r.v(r.f("legal_text", e, t, 0))), r.b(" </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e._;
    t.ModuleView.register("game_logos", {
        template: r.modules.game_logos,
        render: function() {
            var e = i.extend({}, this.model);
            this.$el.html(this.template.render(e)), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render()
        }
    })
}(this), Whiskers("modules.glance", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 36, 54, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(" "), r.s(r.f("data", e, t, 1), e, t, 0, 80, 1195, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="item" data-title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <div class="poster"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"><span class="poster-box"><img src="'), n.b(n.v(n.f("poster", e, t, 0))), n.b('" width="100%"></span></a> </div><div class="details"> <h3><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</a></h3> <p class="desc">'), n.b(n.v(n.f("short_desc", e, t, 0))), n.b('</p> <div class="meta"> '), n.s(n.f("reRelease", e, t, 1), e, t, 0, 373, 459, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="rerelease"><strong>'), n.b(n.v(n.d("translations.re_release", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("reRelease", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(" "), n.s(n.f("release", e, t, 1), e, t, 0, 486, 557, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="release"><strong>'), n.b(n.v(n.f("release_text", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("release", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(" "), n.s(n.f("rating", e, t, 1), e, t, 0, 581, 656, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="rating"><strong>'), n.b(n.v(n.d("translations.rated", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("rating", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(' </div> <div class="actions"> '), n.s(n.f("buttons", e, t, 1), e, t, 0, 709, 852, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button small '), n.b(n.v(n.f("type", e, t, 0))), n.b(" "), n.b(n.v(n.f("button_color", e, t, 0))), n.b('"'), n.s(n.f("convert", e, t, 1), e, t, 0, 792, 819, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' data-convert="'), n.b(n.v(n.f("convert", e, t, 0))), n.b('"')
            }), e.pop()), n.b(">"), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></div> ")
        }), e.pop()), n.b(' </div> </div><div class="trailer-thumb"> <a href="'), n.b(n.v(n.f("thref", e, t, 0))), n.b('"> <span class="thumb-box"><span class="aspect"> <img src="'), n.b(n.v(n.f("tthumb", e, t, 0))), n.b('" class="thumb" alt="'), n.b(n.v(n.f("ttitle", e, t, 0))), n.b('"> <span class="thumb-title"> <h3 class="title" aria-hidden="true">'), n.b(n.v(n.f("ttitle", e, t, 0))), n.b('</h3> <time aria-hidden="true">'), n.b(n.v(n.f("tduration", e, t, 0))), n.b("</time> </span> </span></span> </a> </div> </div> ")
    }), e.pop()), r.b(" </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e._;
    t.ModuleView.register("glance", {
        breakpoints: i,
        template: s.modules.glance,
        render: function() {
            var e = this.model.get("translations"), t = o.extend({
                release_text: e.release
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(t));
            var r = this.$el;
            r.find(".divide:last").css("display", "none"), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    })
}(this), Whiskers("modules.glimpse", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <hr class="topborder"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 59, 77, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="col-container container cols-2">'), r.s(r.f("data", e, t, 1), e, t, 0, 147, 865, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<div class="col item" data-title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> '), n.b(n.rp("entity", e, t, "")), n.b(' <div class="details"> <div class="meta"> '), n.s(n.f("reRelease", e, t, 1), e, t, 0, 260, 346, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="rerelease"><strong>'), n.b(n.v(n.d("translations.re_release", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("reRelease", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(" "), n.s(n.f("release", e, t, 1), e, t, 0, 373, 452, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="release"><strong>'), n.b(n.v(n.d("translations.release", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("release", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(" "), n.s(n.f("rating", e, t, 1), e, t, 0, 476, 551, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="rating"><strong>'), n.b(n.v(n.d("translations.rated", e, t, 0))), n.b(":</strong> "), n.b(n.v(n.f("rating", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(" "), n.s(n.f("albumShortDesc", e, t, 1), e, t, 0, 582, 627, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p class="shortdesc">'), n.b(n.v(n.f("albumShortDesc", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(' </div> <div class="actions"> '), n.s(n.f("buttons", e, t, 1), e, t, 0, 688, 831, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button small '), n.b(n.v(n.f("type", e, t, 0))), n.b(" "), n.b(n.v(n.f("button_color", e, t, 0))), n.b('"'), n.s(n.f("convert", e, t, 1), e, t, 0, 771, 798, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' data-convert="'), n.b(n.v(n.f("convert", e, t, 0))), n.b('"')
            }), e.pop()), n.b(">"), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></div> ")
        }), e.pop()), n.b(" </div> </div> </div> ")
    }), e.pop()), r.b("</div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e._;
    t.ModuleView.register("glimpse", {
        template: r.modules.glimpse,
        render: function() {
            var e = this.model, t = i.extend({
                albumShortDesc: function() {
                    return e.get("type") === "album" && this.get("short_desc") ? this.get("short_desc") : null
                }
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(t, {
                module_header: r.partials.module_elements.module_header,
                iconHeader: r.partials.icon_header,
                entity: r.entities[e.get("type")]
            })), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    })
}(this), Whiskers("modules.browse", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '
    ), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.s(r.f("singleRow", e, t, 1), e, t, 0, 55, 66, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" single-row")
    }), e.pop()), r.s(r.f("item_titles_below", e, t, 1), e, t, 0, 102, 120, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" item-titles-below")
    }), e.pop()), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 161, 179, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="cols-'), r.b(r.v(r.f("rpp", e, t, 0))), r.b(" "), r.b(r.v(r.f("layoutType", e, t, 0))), r.b(' col-container">'), r.s(r.f("data", e, t, 1), e, t, 0, 260, 323, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<span class="col">'), n.b(n.rp("entity", e, t, "")), n.b('<span class="arrow"></span></span>')
    }), e.pop()), r.b("</div> </div>"), r.fl()
}), Whiskers("modules.recommended", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.s(r.f("singleRow", e, t, 1), e, t, 0, 55, 66, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" single-row")
    }), e.pop()), r.s(r.f("item_titles_below", e, t, 1), e, t, 0, 102, 120, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" item-titles-below")
    }), e.pop()), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 161, 179, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="cols-'), r.b(r.v(r.f("rpp", e, t, 0))), r.b(' col-container"> '), r.s(r.f("trimmedData", e, t, 1), e, t, 0, 253, 291, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <span class="col">'), n.b(n.rp("entity", e, t, "")), n.b("</span> ")
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e.Backbone, s = e._, o = n.Style.breakpoints, u = s.values(o), a = s.contains(["disneychannel.disney.com", "disneyxd.disney.com"], n.portal), f = t.ModuleView.extend({
        breakpoints: u.join(" "),
        initialize: function() {
            var e = this.model, t = e.data;
            this.entityType = s.first(t.models).get("entityType") || e.get("type")
        },
        render: function() {
            var e = i.Resize.width(), t = this.cols(e), o = this.model, u = this.model.get("type"), f = this.model.get("style"), l = o.data.models.slice(0, this.limit(t)), c = f ? f.images ? f.images.background_image ? f.images.background_image.url ? f.images.background_image.url: null: null: null: null, h = s.extend({
                rpp: t,
                trimmedData: l,
                layoutType: t > 1 ? "grid-layout": "list-layout",
                channelSite: a,
                entityType: this.entityType,
                singleRow: l.length <= t
            }, this.model);
            this.$el.html(this.template.render(h, {
                module_header: r.partials.module_elements.module_header,
                iconHeader: r.partials.icon_header,
                entity: r.entities[this.entityType]
            })), u === "character" && n.portal === "princess.disney.com"&&!c ? this.$el.addClass("no-bg") : n.Style.moduleStyle(this)
        },
        limit: function(e) {
            var t = this.model.get("type"), n = this.model.data.length;
            return t === "character" ? n : n > e ? n - n%e : n
        }
    });
    f.register("browse", {
        template: r.modules.browse,
        cols: function(e) {
            var t = this.entityType;
            if (this.model.get("style").items_per_row_override && e >= o.max)
                return this.model.get("style").items_per_row_override;
            switch (t) {
            case"personality":
            case"console_game":
                return e >= o.max ? 5 : e >= o.midHigh ? 4 : e >= o.midLow ? 3 : 2;
            case"mobile_app":
                return e >= o.max ? 5 : e >= o.midHigh ? 4 : e >= o.midLow ? 3 : 1;
            case"movie":
                return e >= o.max ? 5 : e >= o.midHigh ? 5 : 1;
            case"video":
            case"game":
                return e >= o.max ? 4 : e >= o.midHigh ? 3 : 1;
            default:
                return e >= o.max ? 4 : e >= o.midHigh ? 3 : 1
            }
        }
    }), f.register("recommended", {
        template: r.modules.recommended,
        cols: function(e) {
            var t = this.entityType;
            return this.model.get("style").items_per_row_override && e >= o.midHigh ? this.model.get("style").items_per_row_override : t === "movie" ? e >= o.max ? 5 : e >= o.midHigh ? 4 : e >= o.midLow ? 3 : 2 : e >= o.midHigh ? 4 : 2
        },
        limit: function(e) {
            return a ? this.model.data.length : e
        }
    }), f.register("enchilada", {
        className: "browse module",
        template: r.modules.recommended,
        cols: function(e) {
            return e >= o.max ? 5 : e >= o.midHigh ? 4 : 1
        },
        limit: function() {
            return this.model.data.length
        }
    })
}(this), Whiskers("modules.guide", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 60, 78, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="schedule"> '), r.s(r.f("data", e, t, 1), e, t, 0, 127, 345, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="row"> <span class="cell time">'), n.b(n.v(n.f("time", e, t, 0))), n.b('</span> <span class="cell info"> <div class="title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</div> <div class="series">'), n.b(n.v(n.f("series", e, t, 0))), n.b('</div> </span> <span class="cell desc">'), n.b(n.v(n.f("description", e, t, 0))), n.b("</span> </div> ")
    }), e.pop()), r.b(" </div> </div> </div>"), r.fl()
}), Whiskers("modules.schedule", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="shows_schedule"> <div class="container"> <div class="header">'), r.b(r.v(r.f("title", e, t, 0))), r.b('</div> <ul class="three_up"> '), r.s(r.f("data", e, t, 1), e, t, 0, 120, 207, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li><span class="time">'), n.b(n.v(n.f("time", e, t, 0))), n.b('</span><span class="show_title">'), n.b(n.v(n.f("series", e, t, 0))), n.b("</span></li> ")
    }), e.pop()), r.b(' </ul> <div class="full_schedule"> <a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"><span class="see_mobile">'), r.b(r.v(r.f("see", e, t, 0))), r.b(" &nbsp;</span>"), r.b(r.v(r.f("linkTitle", e, t, 0))), r.b("</a> </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Modernizr, r = e.Whiskers, i = e.jQuery, s = e._, o = {
        title: "Today's Schedule",
        schedule_title: "What's On:"
    }, u = t.ModuleView.extend({
        initialize: function() {
            var e = this, t = e.buildUrl();
            e.fetchData(t), e.$el.hide()
        },
        zeroPad: function(e, t) {
            var n = e.toString();
            while (n.length < t)
                n = "0" + n;
            return n
        },
        toParseableDate: function(e) {
            var t = {
                Jan: 1,
                Feb: 2,
                Mar: 3,
                Apr: 4,
                May: 5,
                Jun: 6,
                Jul: 7,
                Aug: 8,
                Sep: 9,
                Oct: 10,
                Nov: 11,
                Dec: 12
            }, n = e.split(" ");
            return e = n[3] + "/" + parseInt(t[n[2]], 10) + "/" + parseInt(n[1], 10) + " " + n[4], new Date(e)
        },
        buildUrl: function() {
            var e = this, t = new Date, n = new Date, r = "";
            n.setDate(n.getDate() + 1);
            var i = t.getFullYear() + e.zeroPad(t.getMonth() + 1, 2) + e.zeroPad(t.getDate(), 2), s = n.getFullYear() + e.zeroPad(n.getMonth() + 1, 2) + e.zeroPad(n.getDate(), 2), o =- (t.getTimezoneOffset() / 60), u = "001", a = this.model.get("data")[0].title.toLowerCase(), f = null;
            this.model.get("channel_schedule") ? f = this.model.get("channel_schedule").toLowerCase() : f = "false";
            var l = "";
            return a.indexOf("junior")!==-1 || f.indexOf("junior")!==-1 ? r = "008" : a.indexOf("xd")!==-1 || f.indexOf("xd")!==-1 ? r = "009" : r = "004", l = "/watch/livetvguide/" + r + "/" + u + "/" + i + "/" + s + "/" + o, l
        },
        fetchData: function(e) {
            var t = this;
            i.getJSON(e, function(e) {
                var r = [], u = e.schedule.video, a = 0, f = new Date;
                i.each(u, function(e, n) {
                    var i = n.airdates.airdate, s = t.toParseableDate(i), o = s.getTime() + parseInt(n.duration, 10) >= f.getTime();
                    if (o && a < 10) {
                        var u = s.getHours(), l = s.getMinutes(), c = u >= 12 ? "pm": "am";
                        u = u === 0 ? 12 : u > 12 ? u - 12 : u;
                        var h = u + ":" + t.zeroPad(l, 2) + c;
                        a++, r.push({
                            time: h,
                            title: n.title,
                            series: n.show.title,
                            description: n.description
                        })
                    }
                }), o = s.extend(o, {
                    data: r
                }), n.flash&&!n.mobile && t.$el.hide().html(t.template.render(o)).fadeIn()
            })
        }
    });
    u.register("live_tv_guide", {
        template: r.modules.guide
    }), u.register("schedule", {
        template: r.modules.schedule,
        fetchData: function(e) {
            var t = this, n = this.model.get("data")[0].href, r = this.model.get("data")[0].title, u = this.model.get("title"), a = this.model.get("translations");
            i.getJSON(e, function(e) {
                var f = [], l = e.schedule.video, c = 0, h = new Date;
                i.each(l, function(e, n) {
                    var r = n.airdates.airdate, i = t.toParseableDate(r), s = i.getTime() + parseInt(n.duration, 10) >= h.getTime();
                    if (s && c < 3) {
                        var o = i.getHours(), u = i.getMinutes(), a = o >= 12 ? "p": "a";
                        o = o === 0 ? 12 : o > 12 ? o - 12 : o;
                        var l = o + ":" + t.zeroPad(u, 2) + a;
                        c++, f.push({
                            time: l,
                            series: n.show.title
                        })
                    }
                }), o = s.extend(o, {
                    data: f,
                    href: n,
                    title: u,
                    linkTitle: r,
                    translations: a
                }), t.$el.hide().html(t.template.render(o)).fadeIn()
            })
        }
    })
}(this), Whiskers("modules.hero_universal", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound skip-styles"> <div class="carousel'), r.s(r.f("identity", e, t, 1), e, t, 0, 65, 112, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" identity"), n.s(n.f("hasThumbs", e, t, 1), e, t, 1, 0, 0, "") || n.b(" no-thumbs")
    }), e.pop()), r.b(" "), r.b(r.v(r.f("heroSize", e, t, 0))), r.b(' universal"> <ul class="slides"> '), r.s(r.f("slides", e, t, 1), e, t, 0, 182, 207, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" <li>"), n.b(n.rp("heroSlide", e, t, "")), n.b("</li> ")
    }), e.pop()), r.b(" </ul> </div> "), r.s(r.f("identity", e, t, 1), e, t, 0, 245, 272, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("identityOverlayPartial", e, t, ""))
    }), e.pop()), r.b(" </div>"), r.fl()
}), Whiskers("modules.rich_image", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("image", e, t, 1), e, t, 0, 10, 221, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<div class="bound"> '), n.s(n.f("data", e, t, 1), e, t, 0, 39, 178, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" '), n.s(n.f("age_gate", e, t, 1), e, t, 0, 71, 114, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' class="age-gated" data-age="'), n.b(n.v(n.f("age_gate", e, t, 0))), n.b('" ')
            }), e.pop()), n.b(' title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" target="_blank">')
        }), e.pop()), n.b(' <img src="'), n.b(n.v(n.f("image", e, t, 0))), n.b('"> </a> </div>')
    }), e.pop()), r.fl()
}), Whiskers("partials.hero_slide", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="branding"> <div class="scaler"></div> '), r.s(r.f("overlayLeft", e, t, 1), e, t, 0, 66, 137, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="left-overlay overlay"> <img src="'), n.b(n.v(n.f("overlayLeft", e, t, 0))), n.b('"> </div> ')
    }), e.pop()), r.b(" "), r.s(r.f("overlayRight", e, t, 1), e, t, 0, 171, 244, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="right-overlay overlay"> <img src="'), n.b(n.v(n.f("overlayRight", e, t, 0))), n.b('"> </div> ')
    }), e.pop()), r.b(" "), r.s(r.f("href", e, t, 1), e, t, 0, 271, 290, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">')
    }), e.pop()), r.b(' <div class="background"> '), r.s(r.f("slide", e, t, 1), e, t, 0, 335, 457, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img class="mobile-image" src="'), n.b(n.v(n.f("hero_mobile", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <img class="wide-image" src="'), n.b(n.v(n.f("wide", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> ')
    }), e.pop()), r.b(" </div> "), r.s(r.f("show_data", e, t, 1), e, t, 0, 489, 1183, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.s(n.f("section", e, t, 1), e, t, 0, 502, 546, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<h2 class="channel_section">'), n.b(n.v(n.f("section", e, t, 0))), n.b("</h2>")
        }), e.pop()), n.b(' <div class="channel_mini_hero"> <div class="featured_channel_img"> <img src="'), n.b(n.v(n.f("asset_logo_image", e, t, 0))), n.b('"> <p>'), n.b(n.v(n.f("watch_message", e, t, 0))), n.b("</p> </div> "), n.s(n.f("featured_channel_shortDesc", e, t, 1), e, t, 0, 722, 885, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="featured_link"> <div class="featured_channel"> <img src="'), n.b(n.v(n.f("featured_content_image", e, t, 0))), n.b('"> <p>'), n.b(n.v(n.f("featured_channel_shortDesc", e, t, 0))), n.b("</p> </div> </a> ")
        }), e.pop()), n.b(" "), n.s(n.f("featured_channel_shortDesc", e, t, 1), e, t, 1, 0, 0, "") || n.b('<div class="no_desc_box"></div>'), n.b(' <div class="cds_box '), n.s(n.f("featured_channel_shortDesc", e, t, 1), e, t, 1, 0, 0, "") || n.b("notxt"), n.s(n.f("featured_channel_shortDesc", e, t, 1), e, t, 0, 1129, 1136, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" hastxt")
        }), e.pop()), n.b('"></div> </div> ')
    }), e.pop()), r.b(" "), r.s(r.f("animated", e, t, 1), e, t, 0, 1211, 1266, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.s(n.f("noVideoSupport", e, t, 1), e, t, 1, 0, 0, "") || n.b(n.rp("videoPartial", e, t, ""))
    }), e.pop()), r.b(" "), r.s(r.f("href", e, t, 1), e, t, 0, 1289, 1293, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("</a>")
    }), e.pop()), r.b(" </div> "), r.s(r.f("show_data", e, t, 1), e, t, 1, 0, 0, "") || (r.b(' <div class="content '), r.b(r.v(r.f("contentPosition", e, t, 0))), r.b('"> <div class="logo"> '), r.s(r.f("featured_content_image", e, t, 1), e, t, 0, 1413, 1478, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <img src="'), n.b(n.v(n.f("featured_content_image", e, t, 0))), n.b('"> </a> ')
    }), e.pop()), r.b(" "), r.s(r.f("show_data", e, t, 1), e, t, 1, 0, 0, "") || (r.b(' <a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('" class="heading"> <h2 class="'), r.b(r.v(r.f("editorialColor", e, t, 0))), r.b('" '), r.s(r.f("headerColor", e, t, 1), e, t, 0, 1604, 1635, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('style="color: '), n.b(n.v(n.f("headerColor", e, t, 0))), n.b(';"')
    }), e.pop()), r.b(">"), r.b(r.v(r.f("section", e, t, 0))), r.b('</h2> <h3 class="'), r.b(r.v(r.f("editorialColor", e, t, 0))), r.b('">'), r.b(r.v(r.f("featured_channel_shortDesc", e, t, 0))), r.b("</h3> </a> ")), r.b(" "), r.s(r.f("buttons", e, t, 1), e, t, 0, 1768, 2385, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button large blue" style="background-color: '), n.b(n.v(n.f("bg_top_color", e, t, 0))), n.b("; background-image: -webkit-gradient(linear, left top, left bottom, from("), n.b(n.v(n.f("bg_bottom_color", e, t, 0))), n.b("), to("), n.b(n.v(n.f("bg_top_color", e, t, 0))), n.b(")); background-image: -webkit-linear-gradient(top, "), n.b(n.v(n.f("bg_bottom_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("bg_top_color", e, t, 0))), n.b("); background-image: -moz-linear-gradient(top, "), n.b(n.v(n.f("bg_bottom_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("bg_top_color", e, t, 0))), n.b("); background-image: -o-linear-gradient(top, "), n.b(n.v(n.f("bg_bottom_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("bg_top_color", e, t, 0))), n.b("); background-image: linear-gradient(to bottom, "), n.b(n.v(n.f("bg_bottom_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("bg_top_color", e, t, 0))), n.b("); border: 1px solid "), n.b(n.v(n.f("border_color", e, t, 0))), n.b("; color: "), n.b(n.v(n.f("text_color", e, t, 0))), n.b(';">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a> ")
    }), e.pop()), r.b(" </div> </div> ")), r.fl()
}), Whiskers("partials.hero_video", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<!-- begin video partial--> <div class="video-wrapper '), r.s(r.f("loopOption", e, t, 1), e, t, 0, 69, 76, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("looping")
    }), e.pop()), r.b(" "), r.s(r.f("loopOption", e, t, 1), e, t, 1, 0, 0, "") || r.b("single-play"), r.b('"> <video '), r.s(r.f("loopOption", e, t, 1), e, t, 0, 158, 162, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("loop")
    }), e.pop()), r.b('> <source src="'), r.b(r.v(r.f("mp4", e, t, 0))), r.b('" type="video/mp4"> <source src="'), r.b(r.v(r.f("webm", e, t, 0))), r.b('" type="video/webm"> <img src="'), r.b(r.v(r.f("fallback", e, t, 0))), r.b('"> </video> <!--<a class="sound-toggle on"></a>--> '), r.s(r.f("loopOption", e, t, 1), e, t, 1, 0, 0, "") || (r.b('<div class="replay-button-box '), r.b(r.v(r.f("editorialColor", e, t, 0))), r.b('-theme"><a class="button large replay" style="">Replay</a></div>')), r.b(" </div><!-- end video partial -->"), r.fl()
}), Whiskers("partials.hero_identity_overlay", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="details '), r.b(r.v(r.f("contentPosition", e, t, 0))), r.b(" "), r.b(r.v(r.f("portal", e, t, 0))), r.b(" "), r.b(r.v(r.f("editorialColor", e, t, 0))), r.b('" style="background-color: '), r.b(r.v(r.f("darkColor", e, t, 0))), r.b(';"> <h1 '), r.s(r.f("headerColor", e, t, 1), e, t, 0, 133, 164, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('style="color: '), n.b(n.v(n.f("headerColor", e, t, 0))), n.b(';"')
    }), e.pop()), r.b(">"), r.s(r.f("isPrincess", e, t, 1), e, t, 0, 196, 201, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("Meet ")
    }), e.pop()), r.b('<span class="name">'), r.b(r.v(r.f("title", e, t, 0))), r.b("</span></h1> "), r.s(r.f("hasThumbs", e, t, 1), e, t, 0, 271, 407, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="headshot"> <span class="personality-box"><span class="aspect"> <img src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </span></span> </div> ')
    }), e.pop()), r.b(' <p class="desc '), r.b(r.v(r.f("contentPosition", e, t, 0))), r.b('">'), r.b(r.v(r.f("description", e, t, 0))), r.b("</p> "), r.s(r.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (r.b(' <div id="sharing"> <ul class="social"> '), r.s(r.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || r.b(' <li class="facebook"> <div class="fb-like" data-send="false" data-layout="button_count" data-show-faces="false" data-width="90"></div> </li> '), r.b(" </ul> </div> <!-- End of sharing --> ")), r.b(" </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill.BunView, n = e.Disney, r = e.Grill, i = e.Whiskers, s = e.Modernizr, o = e.jQuery, u = e._, a = e.Backbone, f = e.devicePixelRatio > 1.5, l = n.Style.breakpoints, c = [l.midHigh, l.max], h = r.BunView.extend({
        className: "bun skip-bottom-border full-height full-width",
        template: i.modules.hero_universal,
        heroImg: "hero",
        wideImage: null,
        mobileImage: null,
        breakpoints: c.join(" "),
        currentPlayer: undefined,
        current: undefined,
        oldPlayer: undefined,
        mergedObj: null,
        darkColor: null,
        identityThumbs: null,
        slideDelay: 3e3,
        delayAfterVideo: 4e3,
        showArrows: !0,
        showDots: !0,
        hidden: !1,
        socialLoaded: !1,
        initialize: function() {
            this.model.get("slide_timer") && (this.slideDelay = this.model.get("slide_timer") * 1e3);
            var e = this.model.get("data"), t = this.model.get("style"), n = this, r = null;
            if (e.length > 0 && e[0].type === "Hero")
                this.darkColor = this.model.get("style").module_dark_background_color || this.model.collection.style.background_color_dark, this.mergedObj = {
                    heroes: e
                }, o(e).slice( - 1)[0].type !== "Hero" && (r = o(e).slice( - 1)[0], this.mergedObj.heroes.pop(), this.identityThumbs = r.type === "Character" || r.type === "Artist"), this.mergedObj.identity = r;
            else {
                n.template = i.modules.rich_image;
                if (t && t.images && t.images.foreground_image) {
                    var s = t.images.foreground_image;
                    n.wideImage = f ? s.retina_url ? s.retina_url : s.url : s.url, n.mobileImage = f ? s.mobile_retina_url ? s.mobile_retina_url : s.mobile_url ? s.mobile_url : null : s.mobile_url ? s.mobile_url : null
                }
            }
        },
        render: function() {
            var t = this, r = this.model, c = r.get("data"), h = u.first(r.get("data")), p = r.get("style"), d = a.Resize.width(), v = d >= l.midHigh, m = d >= l.max, g=!!this.options.cds, y = u.extend({
                data: c,
                wide: v,
                desktop: m,
                image: v ? t.wideImage: t.mobileImage,
                cds: g,
                noVideoSupport: !s.video || s.mobile,
                darkColor: t.darkColor,
                heroSize: h ? h.heroSize: null,
                layout: p.layout,
                contentPosition: h ? h.contentPosition: null,
                slides: c,
                identity: function() {
                    if (t.mergedObj.identity)
                        return t.mergedObj.identity.darkColor = t.darkColor, t.mergedObj.identity.hasThumbs = t.identityThumbs, t.mergedObj.identity.isPrincess = n.portal === "princess.disney.com", t.mergedObj.identity.editorialColor = u.first(t.mergedObj.heroes).editorialColor, t.mergedObj.identity.headerColor = u.first(t.mergedObj.heroes).titleColor || u.first(t.mergedObj.heroes).editorialColor, t.mergedObj.identity
                },
                slide: function() {
                    var e = f ? this[t.heroImg + "2x"] ? this[t.heroImg + "2x"]: this[t.heroImg]: this[t.heroImg], n = this.hero_mobile, r = {
                        wide: e,
                        mobile: n
                    };
                    return r
                },
                headerColor: function() {
                    return this.titleColor
                }
            }, this.options.ads.helpers(), r);
            this.$el.html(t.template.render(y, {
                heroSlide: i.partials.hero_slide,
                videoPartial: i.partials.hero_video,
                identityOverlayPartial: i.partials.hero_identity_overlay
            })), this.$(".slides").swipeasaurus({
                effect: d >= l.max ? "fade": t.mergedObj && t.mergedObj.heroes && t.mergedObj.heroes.length > 1 ? "slide": "fade",
                dots: this.showDots,
                arrows: this.showArrows
            }), n.Style.moduleStyle(this, !1), o(e).on("scroll." + this.cid, u.throttle(u.bind(this.scrollCheck, this), 300))
        },
        resize: function() {
            var t = this.$(".slides"), n = this, r = (t.data("swipeasaurus") || {}).index / o("li", t).length, i = [];
            this.$(".slides li.finished").each(function(e, t) {
                i.push(o(t).index())
            }), t = n.$(".slides"), t.swipeasaurus("slide", e.Math.floor(r * o("li", t).length), !0), this.options.ads.restore(), t.swipeasaurus("stop"), o(i).each(function(e) {
                o(n.$(".slides li")[e]).addClass("finished")
            }), this.ready(), this.model.get("data").length > 0 && this.model.get("data")[0].type !== "Hero" && this.render()
        },
        ready: function() {
            var e = a.Resize.width();
            this.mergedObj && this.mergedObj.identity && this.mergedObj.identity.headerColor && (e < l.midHigh ? this.$(".details h1").css("color", "#fff") : this.$(".details h1").css("color", this.mergedObj.identity.headerColor));
            if (e < l.midHigh)
                return;
            this.takeover(0);
            var t = this.$(".slides"), r = this.$(".button.replay"), i = this;
            i.current = o(".dot.active")[0] ? o(".dot.active").index() : 0, r.on("click", function(e) {
                e.preventDefault();
                var n = o(e.target).parent(".content").siblings(".branding").find("video")[0];
                o(n).closest("li").removeClass("finished"), t.swipeasaurus("stop"), i.resetSlide(o(e.target).closest(".slides li").index(), !0)
            }), t.on("slide", function(e, t) {
                a.Resize.width() >= l.max && i.takeover(t)
            }).swipeasaurus("start", i.slideDelay), i.onNewSlide(i.current);
            var s = function(e, t) {
                if (t === i.current)
                    return 
            };
            t.on("slideEnd", s), i.scrollCheck(), !i.socialLoaded && i.mergedObj && i.mergedObj.identity&&!this.options.cds && (n.FB.go(), i.socialLoaded=!0)
        },
        scrollCheck: function() {
            var t = this, n = this.$(".slides"), r = o(e).height(), i = t.$el.height(), s = t.$el.offset().top, u = o(e).scrollTop();
            if (t.hidden) {
                if (t.hidden === "below" && r + u > s + i / 5 || t.hidden === "above" && u < s + i)
                    t.hidden=!1, t.currentPlayer && (o(t.currentPlayer).attr("loop") === "loop" ? t.currentPlayer.play() : t.onNewSlide(t.current)), n.swipeasaurus("resume")
            } else 
                u > s + i ? (t.hidden = "above", t.currentPlayer && o(t.currentPlayer).attr("loop") === "loop" && t.currentPlayer.pause(), n.swipeasaurus("pause")) : r + u < s && (t.hidden = "below", t.currentPlayer && o(t.currentPlayer).attr("loop") === "loop" && t.currentPlayer.pause(), n.swipeasaurus("pause"))
        },
        pauseAndRewind: function(e) {
            e.pause(), e.currentTime = 0
        },
        startVideo: function(e) {
            var t = this;
            e && (e.play(), o(e).on("ended", function() {
                o(e).closest("li").addClass("finished"), t.showEndcard(e)
            }))
        },
        showEndcard: function(e) {
            var t = this;
            o(e).closest("li").hasClass("finished") ? a.Resize.width() > l.midHigh && t.$(".slides").swipeasaurus("start", t.delayAfterVideo) : (e.currentTime = e.duration, o(e).closest("li").addClass("finished"))
        },
        resetSlide: function(e, t) {
            var n = this;
            o(n.$(".slides li")[e]).removeClass("finished");
            var r = o(n.$(".slides li")[e]).find("video")[0];
            r && (n.pauseAndRewind(r), t && n.startVideo(r))
        },
        onNewSlide: function(e) {
            var t = this, n = t.$(".slides");
            t.current = e, a.Resize.width() > l.midHigh && (t.oldPlayer && t.oldPlayer !== t.currentPlayer && (o(t.oldPlayer).closest("li").hasClass("finished") || (t.pauseAndRewind(t.oldPlayer), o(t.oldPlayer).closest(".video-wrapper").hasClass("single-play") && t.showEndcard(t.oldPlayer))), o(t.$(".slides > li")[e]).find("video") && (t.currentPlayer = o(t.$(".slides > li")[e]).find("video")[0]), t.oldPlayer = t.currentPlayer, t.hidden===!1 && (t.currentPlayer ? o(t.currentPlayer).closest("li").hasClass("finished") || o(t.currentPlayer).attr("loop") === "loop" ? (o(t.currentPlayer).attr("loop") === "loop" && t.startVideo(t.currentPlayer), n.swipeasaurus("start", t.slideDelay)) : (n.swipeasaurus("stop"), t.startVideo(t.currentPlayer)) : n.swipeasaurus("start", t.slideDelay)), a.Resize.width() >= l.max && t.takeover(e))
        },
        takeover: function(e) {
            var t = this.model.get("data"), r = t[e].background_styles;
            n.Style.backgroundStyles(r)
        },
        remove: function() {
            t.prototype.remove.call(this), o(e).off("scroll." + this.cid)
        }
    });
    h.register("hero_universal", {}), h.register("minihero", {
        heroImg: "minihero"
    }), h.register("channel_hero", {}), h.register("hero_slider", {}), h.register("empty_takeover", {}), h.register("rich_media", {
        takeover: function() {
            return 
        },
        showArrows: !1,
        showDots: !1,
        hidden: "below"
    })
}(this), Whiskers("modules.image_gallery", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 3106, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div id="print-image" class="skip-styles"> <div class="print-frame"> <div class="img-wrapper"> <img> <div class="info"> <p class="print-notice"> &copy; 2013 Disney </p> <span class="caption"></span> </div> </div> </div> </div> <div class="gallery_header"> <div class="gallery_banner_container"> <div class="banner_img"> <div class="scaler"></div> <img src="'), n.b(n.v(n.f("banner", e, t, 0))), n.b('"> <h2 class="'), n.s(n.f("cds", e, t, 1), e, t, 0, 399, 412, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" gallery-noad")
        }), e.pop()), n.b('">'), n.b(n.v(n.f("gallery_title", e, t, 0))), n.b('</h2> <span class="gallery_shade"></span> </div> </div> <div class="gallery_link_container'), n.s(n.f("cds", e, t, 1), e, t, 0, 537, 550, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" gallery-noad")
        }), e.pop()), n.b('"> <p class="description">'), n.b(n.v(n.f("description", e, t, 0))), n.b('</p> <ul class="gallery_external_links"> '), n.s(n.f("parent_page", e, t, 1), e, t, 0, 656, 743, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <li class="parent_page_link"><a href="'), n.b(n.v(n.f("parent_page", e, t, 0))), n.b('">'), n.b(n.v(n.f("parent_page_title", e, t, 0))), n.b("</a></li> ")
        }), e.pop()), n.b(' <li class="image-total">'), n.b(n.v(n.f("numImages", e, t, 0))), n.b(" "), n.b(n.v(n.d("translations.images", e, t, 0))), n.b("</li> "), n.b(" "), n.s(n.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.s(n.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <li class="social"> '), n.s(n.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.b(n.rp("socialLinkPartial", e, t, "")), n.b(" ")), n.b(" </li> ")), n.b(" ")), n.b(' </ul> </div> </div> <div class="gallery_ad">'), n.b(n.t(n.f("rectAllAd", e, t, 0))), n.b('</div> <div id="gallery_thumb_grid"> <ul class ="img-cols"> '), n.s(n.f("thumbViewList", e, t, 1), e, t, 0, 1141, 1160, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.t(n.f("thumbViewList", e, t, 0)))
        }), e.pop()), n.b(" "), n.s(n.f("thumbViewList", e, t, 1), e, t, 1, 0, 0, "") || (n.b(" "), n.b(" "), n.s(n.f("noscript_images", e, t, 1), e, t, 0, 1280, 1307, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.rp("partials/gallery_thumb", e, t, ""))
        }), e.pop()), n.b(" ")), n.b(" </ul> </div> "), n.s(n.f("show_pagination", e, t, 1), e, t, 0, 1380, 1620, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="static_pagination"> '), n.s(n.f("prev_page", e, t, 1), e, t, 0, 1427, 1501, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b('<a href="?page='), n.b(n.v(n.f("prev_page", e, t, 0))), n.b('" class="prev_page"> &laquo; Previous Page</a>')
            }), e.pop()), n.b(" "), n.s(n.f("next_page", e, t, 1), e, t, 0, 1530, 1598, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b('<a href="?page='), n.b(n.v(n.f("next_page", e, t, 0))), n.b('" class="next_page">Next Page&raquo;</a>')
            }), e.pop()), n.b(" </div> ")
        }), e.pop()), n.b(" "), n.s(n.f("hasMore", e, t, 1), e, t, 0, 1653, 1842, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="show_more_container"> <div class="show_more blue button large" role="button"> <span class="label">'), n.b(n.v(n.d("translations.show_more", e, t, 0))), n.b('</span> <span class="spinner"></span> </div> </div> ')
        }), e.pop()), n.b(" "), n.b(' <div class="modal-window skip-styles"> <div class="modal-header"></div> <div class="modal-content"> '), n.b(n.t(n.f("imageViewList", e, t, 0))), n.b(' <div class="arrows"> <div class="arrow-sizer prev"><span class="arrow-prev clicon" data-title="prev"></span></div> <div class="arrow-sizer next"><span class="arrow-next clicon" data-title="next"></span></div> </div> </div> <div class="modal-footer"> <span class="print"></span> <span class="image-count"><span class="current-count">1</span> '), n.b(n.v(n.d("translations.of", e, t, 0))), n.b(" "), n.b(n.v(n.f("numImages", e, t, 0))), n.b('</span> </div> <div class="loading-status"><span>'), n.b(n.v(n.d("translations.loading", e, t, 0))), n.b('...</span></div> <div class="modal-info"> <h2 class="modal-title">'), n.b(n.v(n.f("gallery_title", e, t, 0))), n.b("</h2> "), n.b(" "), n.b(" "), n.b(" "), n.b(" "), n.b(' </div> <div class="modal-close clicon" data-title="close"></div> </div> <div class="modal-overlay"></div> ')
    }), e.pop()), r.fl()
}), Whiskers("partials.gallery_thumb", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<li class="gallery-item item thumb launches-modal '), r.s(r.f("active", e, t, 1), e, t, 0, 61, 67, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("active")
    }), e.pop()), r.b('" data-title="thumb" itemscope itemtype="http://schema.org/ImageObject"> <div class="center-crop"> <div class="img-aspect"> <a itemprop="contentURL" href="'), r.b(r.v(r.f("image_fog", e, t, 0))), r.b('"> <img class="thumbnail-image" title="'), r.b(r.v(r.f("seo_title", e, t, 0))), r.b('" alt="'), r.b(r.v(r.f("seoCaption", e, t, 0))), r.b('" src="'), r.b(r.v(r.f("thumbnail_image_fog", e, t, 0))), r.b('" itemprop="thumbnailUrl"> <div class="active-indicator"></div> </a> </div> <div class="zoom-overlay" '), r.s(r.f("touchDevice", e, t, 1), e, t, 0, 467, 484, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' ontouchstart=" "')
    }), e.pop()), r.b('></div> </div> <meta itemprop="caption" content="'), r.b(r.v(r.f("caption", e, t, 0))), r.b('"> <meta itemprop="name" content="'), r.b(r.v(r.f("seoTitle", e, t, 0))), r.b('"> </li>'), r.fl()
}), Whiskers("partials.image_gallery_slide", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<li class="content-container item-'), r.b(r.v(r.f("slide", e, t, 0))), r.b(" "), r.b(r.v(r.f("type", e, t, 0))), r.b("-"), r.b(r.v(r.f("index", e, t, 0))), r.b(" "), r.b(r.v(r.f("type", e, t, 0))), r.b('" data-id="'), r.b(r.v(r.f("id", e, t, 0))), r.b('"> <div class="image-info"> '), r.s(r.f("caption", e, t, 1), e, t, 0, 128, 166, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<p class="description">'), n.b(n.v(n.f("caption", e, t, 0))), n.b("</p>")
    }), e.pop()), r.b(' </div> <div class="gallery-image-sizer"> '), r.s(r.f("ad", e, t, 1), e, t, 0, 227, 458, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b('<div class="interstitial-container"> <div class="gallery_ad"><div class="ad-sizer"></div></div> </div> <p class="skip-ad"><span class="skip-prev"></span><span class="skip-text">'), n.b(n.v(n.f("skipText", e, t, 0))), n.b('</span><span class="skip-next"></span></p>')
    }), e.pop()), r.b(" "), r.s(r.f("ad", e, t, 1), e, t, 1, 0, 0, "") || r.b(' <img class="gallery-image"> '), r.b(" </div> </li>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Math, r = e.jQuery;
    t.Style.fitToScreen = function(e) {
        if (e) {
            var t = e[0], i = t && t.item, s = t.height / t.width || 0, o = t.limitMax, u = e[1], a = e[2] && e[2].height() || 0, f = e[3], l = f && f.height() || 0, c = e[4] || .5, h = u.height(), p = h - a - l, d = u.width(), v = o ? t.width: n.min(p / s, d), m = o ? t.height: v * s, g = n.max(n.abs(p - m) * c, 0), y = g + m;
            f.css({
                top: y
            }), i && r(i).css({
                height: m,
                width: v,
                top: g
            })
        }
    }
}(this), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = e.Modernizr, s = e.Whiskers, o = e.Backbone, u = e.Tracker, a = e._, f = e.jQuery;
    t.ModuleView.register("image_gallery", {
        page: 1,
        currentImage: 0,
        currentSlide: 0,
        autoPaginate: !0,
        imageIds: [],
        resizeParams: {},
        adjacents: [],
        template: s.modules.image_gallery,
        social_link: s.partials.social_link,
        events: {
            "click .modal-close": function() {
                this.closeModal()
            },
            "click .skip-ad": function(e) {
                var t = this.$(".modal-slides");
                f(e.target).hasClass("skip-prev") ? t.swipeasaurus("prev") : t.swipeasaurus("next")
            },
            "click .gallery-item": function(e) {
                e.preventDefault();
                var t = f(e.target).closest(".gallery-item").index();
                this.openModal(t)
            },
            "click .show_more_container": function() {
                this.loadMoreThumbs(), u && this.cleanPageName && u.page({
                    pageName: this.cleanPageName + "_" + this.page
                })
            },
            "click .print": function() {
                e.print()
            },
            "touchstart .print": function() {
                e.print()
            }
        },
        initializeModalAd: function(e) {
            var t = this.options.ads.tag("modalInterstitial"), n = e.$el.find(".interstitial-container .gallery_ad");
            n.html(t), this.options.ads.slots.modalInterstitial.ready(), this.modalAd = n
        },
        initialize: function() {
            var e = this, n=!!this.options.cds, r = a.first(this.model.get("data")), u = r.images;
            a.extend(this, {
                touchDevice: i.touch,
                cds: n,
                allThumbs: u,
                numImages: u.length,
                modalImages: [],
                visibleThumbs: u && u.slice(0, this.page * this.numPerBatch()),
                adInterval: n ? null: r.adInterval
            });
            var l = this.adInterval;
            this.adDivisor = l ? l : 1e3, f.each(this.allThumbs, function(t, n) {
                e.imageIds.push(n.id), n.index = t, n.slide = e.modalImages.length, n.type = "image", e.modalImages.push(n), l && (t + 1)%e.adDivisor === 0 && e.modalImages.push({
                    type: "ad",
                    index: Math.floor(t / e.adDivisor),
                    ad: !0,
                    slide: e.modalImages.length,
                    skipText: "Skip ad"
                })
            }), this.numSlides = e.modalImages.length, this.thumbCollection = new o.LazyCollection(this.visibleThumbs), this.modalCollection = new o.LazyCollection(this.modalImages), this.thumbViewList = new t.ListView({
                collection: this.thumbCollection,
                template: {
                    render: function(e) {
                        return s.partials.gallery_thumb.render(e, {})
                    }
                }
            }), this.imageViewList = new t.ListView({
                className: "modal-slides",
                collection: this.modalCollection,
                template: {
                    render: function(e) {
                        return a.extend(e, {
                            imgSrc: function() {
                                return o.Resize.width() <= 568 ? this.get("mobile_image") : this.get("image")
                            }
                        }), s.partials.image_gallery_slide.render(e, {})
                    }
                }
            }), f.each(this.imageViewList.views, function(e, t) {
                t.printSrc = t.model.get("image") || null
            });
            var c = this.model.collection;
            c && c.cto && (this.cleanPageName = c.cto.pageName, c.cto.pageName += "_1", c.cto.breadcrumbs = "image-gallery")
        },
        render: function() {
            var e = this.thumbViewList, t = this.imageViewList, r = this.model.get("translations"), i = {
                images: "Images"
            };
            a.defaults(r, i);
            var o = a.extend(this.options.ads.helpers(), this.model, {
                numImages: this.numImages,
                hasMore: this.hasMore(),
                cds: this.cds,
                thumbViewList: n.placeHtml(e),
                imageViewList: n.placeHtml(t),
                translations: r
            });
            this.$el.html(this.template.render(o, {
                socialLinkPartial: s.partials.social_btn
            })), n.placeSwap(this, e), n.placeSwap(this, t), n.Style.moduleStyle(this, !1)
        },
        ready: function() {
            var t = this, n = this.slides = this.$(".modal-slides");
            n.swipeasaurus({
                arrows: !0,
                next: ".arrow-next",
                prev: ".arrow-prev",
                loop: !1,
                speed: 500
            }), this.$(".content-container").css("visibility", "hidden"), n.on("slide", function(e, n) {
                t.onSlide(n)
            }), this.updateButton(!1), f("body").addClass(this.hoverClass), f("html").addClass("single-page-print");
            var r = 27, i = 37, s = 39;
            f(e.document).keyup(function(e) {
                switch (e.keyCode) {
                case r:
                    f(".modal-window .modal-close"
                    ).trigger("click");
                    break;
                case i:
                    n.swipeasaurus("prev");
                    break;
                case s:
                    n.swipeasaurus("next")
                }
            }), this.modalAd = f(".interstitial-container .gallery_ad").first();
            if (e.location.hash) {
                var o = /(image\/)([a-z0-9]*)/, u = o.exec(e.location.hash), a = u && u[2] ? t.imageIds.indexOf(u[2]): - 1;
                a >= 0 && (t.autoPaginate=!1, t.openModal(a))
            }
        },
        dimensions: function() {
            var e = o.Resize.width(), t = e >= r.max ? "desktop": e >= r.midHigh ? "tablet": "mobile", n = {
                desktop: 4,
                tablet: 3,
                mobile: 2
            }, i = {
                desktop: 3,
                tablet: 3,
                mobile: 3
            }, s = {
                cols: n[t],
                rows: i[t]
            };
            return s
        },
        numPerBatch: function() {
            return this.dimensions().rows * this.dimensions().cols
        },
        bindResize: function(t) {
            var r = this, i = r.fetchAdjacents(), s = function() {
                n.Style.fitToScreen(r.fetchResizeParams(t)), i.length > 0 && f(i).each(function(e, t) {
                    n.Style.fitToScreen(r.fetchResizeParams(t))
                })
            };
            f(e).on("resize." + this.cid + "slide" + t, a.debounce(s, 100))
        },
        unbindResize: function(t) {
            f(e).off("resize." + this.cid + "slide" + t)
        },
        setResizeParams: function(e, t) {
            var n = t.$el, r = this.$(".modal-window"), i = this.$(".modal-header"), s = n.find(".image-info"), o = .46, u = "item" + t.model.get("slide");
            return this.resizeParams[u] = [e, r, i, s, o], this.resizeParams[u]
        },
        fetchResizeParams: function(e) {
            return this.resizeParams["item" + e]
        },
        fetchAdjacents: function() {
            return this.adjacents
        },
        itemLoaded: function(e) {
            var t=!1;
            if (!e.model.get("ad")) {
                var n = e.$el.find("img")[0];
                t = n&&!!n.src
            }
            return t
        },
        preloadItem: function(t) {
            var n = this, r = t.model, i=!!r.get("ad");
            if (i)
                n.insertNewItem(t);
            else {
                var s = o.Resize.width() <= 568 ? r.get("mobile_image"): r.get("image"), u = new e.Image;
                u.className = "gallery-image", u.src = s, u.complete ? n.insertNewItem(t, u) : f(u).on("load", function() {
                    n.insertNewItem(t, u)
                })
            }
        },
        insertNewItem: function(e, t) {
            var n;
            if (e.model.get("ad"))
                n = this.modalAd;
            else {
                var r = e.$el.find("img")[0] || null;
                r && t && f(r).replaceWith(t), n = t
            }
            this.sizeItem(n, e), this.$(".modal-window").removeClass("loading")
        },
        onSlide: function(t) {
            var r = this, i = this.$(".modal-window"), s = this.viewForIndex(t), o = s.$el.data("id"), l = s.model, c = l.get("ad"), h = t < this.numSlides - 1 ? t + 1: null, p = t > 0 ? t - 1: null, d = {
                next: h,
                prev: p
            };
            r.itemLoaded(s) || (this.preloadItem(s), c || i.addClass("loading")), this.unbindResize(this.currentSlide), this.bindResize(t), this.adjacents = d, a.extend(this, {
                currentImage: l.get("index"),
                currentSlide: t,
                currentView: s
            }), this.$(".current").removeClass("current"), s.$el.addClass("current"), c ? (this.modalAd.find(".ad-sizer").length > 0 && this.initializeModalAd(s), this.insertAd(s)) : r.autoPaginate && this.currentImage >= this.visibleThumbs.length && this.loadMoreThumbs(), f.each(d, function(e, t) {
                if (t) {
                    var i = r.viewForIndex(t);
                    r.itemLoaded(i) ? n.Style.fitToScreen(r.fetchResizeParams(t)) : r.preloadItem(i)
                }
            }), this.setNewText(s), this.insertPrintData(s), s.$el.css("visibility", "visible"), s.$el.css("opacity", 1), i.hasClass("first-shown") && e.setTimeout(function() {
                r.$(".modal-window").removeClass("first-shown")
            }, 800), n.Style.fitToScreen(r.fetchResizeParams(t)), u && this.cleanPageName && u.page({
                pageName: this.cleanPageName
            }), o ? e.location.replace("#image/" + s.$el.data("id")) : e.location.replace("#")
        },
        centerThumb: function(e) {
            var t = e.outerHeight(), n = e.offset().top, r = t / 2, i = Math.max(n - r, 0);
            f("body").stop(!0).animate({
                scrollTop: i
            }, 1e3)
        },
        setNewText: function(e) {
            var t = parseInt(e.model.get("index"), 10), n = this.fetchAdjacents().next, r = this.fetchAdjacents().prev, i = n ? this.viewForIndex(n).model: e.model, s = r ? this.viewForIndex(r).model: e.model, o = e.model.get("ad"), u = this.$(".modal-window");
            o ? u.addClass("ad-view") : (this.$(".modal-window .current-count").html(t + 1), u.removeClass("ad-view")), this.$(".modal-window .arrow-prev").data({
                currentitem: n ? i.get("index"): this.numSlides,
                viewtype: i.get("type")
            }), this.$(".modal-window .arrow-next").data({
                currentitem: r ? s.get("index"): 0,
                viewtype: s.get("type")
            })
        },
        sizeItem: function(t, r) {
            var i, s, o;
            if (t === this.modalAd) {
                var u = t.find(".ad-sizer").length > 0 ? t.find(".ad-sizer"): t.find(".gpt");
                i = u.height(), s = u.width(), o=!0
            } else {
                var a = new e.Image;
                a.src = t.src, i = a.height, s = a.width, o=!1
            }
            var f = {
                height: i,
                width: s,
                limitMax: o,
                item: t
            };
            r.aspectRatio = i / s, n.Style.fitToScreen(this.setResizeParams(f, r))
        },
        openModal: function(e) {
            var t = e + Math.floor(e / this.adDivisor), n = this.viewForIndex(t), r = this.$(".modal-slides"), i = this.$(".modal-window");
            i.addClass("first-shown"), r.swipeasaurus("effect", "fade"), this.onSlide(t), r.swipeasaurus("slide", t), r.swipeasaurus("effect", "slide"), i.addClass("modal-show"), n.$el.css("visibility", "visible")
        },
        closeModal: function() {
            var t = this;
            this.$(".modal-window .modal-close").data("currentitem", this.currentImage), this.$(".modal-window").removeClass("modal-show loading"), this.$(".gallery-item.active").removeClass("active"), this.$(".content-container").css("visibility", "hidden"), t.autoPaginate ? (this.thumbViewList.collection.at(t.currentImage).set("active", !0), e.setTimeout(function() {
                t.thumbViewList.collection.at(t.currentImage).set("active", !1)
            }, 1200), this.centerThumb(this.thumbViewList.views[this.currentImage].$el)) : t.autoPaginate=!0, this.unbindResize(this.currentSlide), this.cds || this.refreshAd("rectAll")
        },
        updateButton: function(e) {
            this.$el.toggleClass("loading", e).toggleClass("has-more", this.hasMore())
        },
        hasMore: function() {
            return this.page * this.numPerBatch() < this.numImages
        },
        loadMoreThumbs: function() {
            this.page++, this.updateButton(!0), this.visibleThumbs = this.allThumbs.slice(0, this.page * this.numPerBatch()), this.thumbViewList.collection.add(this.visibleThumbs), this.updateButton(!1)
        },
        insertAd: function(e) {
            var t = this.modalAd.detach();
            e.$el.find(".interstitial-container").html(t), this.refreshAd("modalInterstitial")
        },
        refreshAd: function(e) {
            this.options.ads.slots[e].refresh()
        },
        viewForIndex: function(e) {
            return this.imageViewList.views[e]
        },
        remove: function() {
            this.unbindResize(this.currentSlide), f("body").removeClass(this.hoverClass), f("html").removeClass("single-page-print"), t.ModuleView.prototype.remove.call(this)
        },
        insertPrintData: function(e) {
            var t = f("#print-image img")[0], n = e.printSrc, r = e.model.get("caption");
            t && (t.src = n), f("#print-image .caption").html(r)
        }
    })
}(this), Whiskers("modules.landing", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 146, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound"> <div class="landing-box '), n.b(n.v(n.f("landingClass", e, t, 0))), n.b('"> <img src="'), n.b(n.v(n.f("landingImg", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" class="landing"> </div> </div> ')
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e.jQuery, u = e.Backbone, a = e._, f = t.BunView.extend({
        className: "module full-width full-height",
        template: s.modules.landing,
        breakpoints: i,
        takeover: function() {
            o("#takeover-area").addClass("module-takeover");
            var e = this.model.get("data"), t = a.first(e).background_styles;
            n.Style.backgroundStyles(t)
        },
        render: function() {
            var e = this.model.data.first(), t = u.Resize.width() >= r.midHigh, i = a.extend({
                landingClass: t && "wide",
                landingImg: e[t ? "header": "header_mobile"]() || e[t ? "hero": "hero_mobile"]()
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(i)), n.Style.moduleStyle(this, !1);
            var s = this.model.get("data"), o = a.first(s).takeover;
            n.Style.backgroundStyles(o)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        },
        ready: function() {
            this.takeover()
        }
    });
    f.register("landing", {}), f.register("youtube", {
        ready: function() {
            n.Style.moduleStyle(this)
        }
    })
}(this), Whiskers("modules.live", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="video_wrapper" class="inverted"> <div class="bound"> <div class="scaler"></div> <div class="live player" id="live_watch_player" width="100%" height="100%" vp:wmode="direct" vp:pfallback="flash" vp:autoplay="true" vp:preloader="http://cdn.edgedatg.com/vp2/m/vp2k/1.0.0/prod/preloader.swf"> </div> </div>'), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.flashembed, r = t.Capabilities = {
        MIN_FP_VERSION: [9, 0],
        isHTML5Capable: function() {
            return e.Modernizr.canvas
        },
        isFlashCapable: function(e) {
            var t = e || r.MIN_FP_VERSION;
            return n.isSupported(t)
        },
        isMobileBrowser: function() {
            return e.navigator.userAgent.match(/(ipod|iphone|ipad|android|silk|iemobile)/i)
        },
        isiOSBrowser: function() {
            return e.navigator.userAgent.match(/(ipod|iphone|ipad)/i)
        },
        isAndroidBrowser: function() {
            return e.navigator.userAgent.match(/(android)/i)
        },
        isKindleBrowser: function() {
            return e.navigator.userAgent.match(/(kindle|silk)/i)
        }
    }
}(this), function(e) {
    "use strict";
    function d() {
        e.flashembed = u
    }
    function v(t) {
        t.providerId || e.Backbone.history.navigate(c[h].url, !0)
    }
    function m() {}
    var t = e.Grill, n = e.Disney, r = e._, i = e.jQuery, s = e.Whiskers, o = e.Modernizr, u = e.flashembed, a = e.Math, f=!1, l=!1, c = {
        dch: {
            url: "/collections/disney-channel-4befcbb33c18eb29118e2222",
            id: "9723272"
        },
        xd: {
            url: "/collections/disney-xd-4befcbb5e4276b29118e2222",
            id: "9723252"
        },
        djr: {
            url: "/collections/disney-junior-4befcbb481d3cb29118e2222",
            id: "9723262"
        }
    }, h, p = function() {
        if (!l ||!f)
            return;
        if (o.flash&&!o.mobile) {
            i("#live_watch_player").attr("vp:config", c[h].id);
            var t = e.com.disney.datg.videoplatforms.vp2k.PlayerManager.create("live_watch_player");
            t.on(t.PROVIDERSELECT, v), t.on(t.PROVIDERDIALOG, m), t.on(t.LOADSTART, d)
        } else 
            i("#live_watch_player").html('<div class="player_error">Sorry, but your device does not support watching live TV.</div>')
    };
    t.BunView.register("live_watch", {
        initialize: function() {
            var t = this;
            t.h = i(e).on("resize.live_watch", r.throttle(function() {
                var n = i(e).height();
                n !== t.h && (t.h = n, t.layout())
            }, 100)).height();
            var n = this.model.get("data")[0].title.toLowerCase();
            n.indexOf("junior")!==-1 ? h = "djr" : n.indexOf("xd")!==-1 ? h = "xd" : h = "dch", i("document").ready(function() {
                i.ajax("http://cdn.edgedatg.com/vp2/m/vp2k/1.0.0/prod/embed.js?prod=true", {
                    dataType: "script",
                    cache: !0,
                    success: function() {
                        l=!0, e.setTimeout(p, 0)
                    }
                })
            })
        },
        render: function() {
            this.$el.html(s.modules.live.render()), n.Style.moduleStyle(this, !1)
        },
        ready: function() {
            f=!0, e.setTimeout(p, 0)
        },
        layout: function() {
            var e = this.$("#video_wrapper .bound");
            e.css("maxWidth", a.min(1280, a.max(320, a.ceil((this.h - 120) / .5625))))
        },
        remove: function() {}
    })
}(this), Whiskers("partials.logo", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<a href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('" title="Disney '), r.b(r.v(r.f("title", e, t, 0))), r.b('" class="nav-logo"'), r.s(r.f("hide_logo", e, t, 1), e, t, 0, 74, 96, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' class="disable_image"')
    }), e.pop()), r.s(r.f("logo_override", e, t, 1), e, t, 0, 128, 205, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' id="nav-logo-override"> <img src="'), n.b(n.v(n.f("logo_override", e, t, 0))), n.b('" alt="Disney '), n.b(n.v(n.f("title", e, t, 0))), n.b('">')
    }), e.pop()), r.s(r.f("logo_override", e, t, 1), e, t, 1, 0, 0, "") || (r.b('id="nav-logo"><span>'), r.b(r.v(r.f("title", e, t, 0))), r.b("</span> ")), r.b("</a>"), r.fl()
}), Whiskers("modules.navigation.channels", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="nav-local-channels" class="nav-local '), r.b(r.v(r.f("style", e, t, 0))), r.b('"> <div class="'), r.b(r.v(r.f("style", e, t, 0))), r.b('-bound"> <div class="'), r.b(r.v(r.f("style", e, t, 0))), r.b('-bg"> '), r.b(r.rp("logo", e, t, "")), r.b(' <span id="nav-e" role="button" title="Navigate"></span> <ul class="pages"> '), r.s(r.f("links", e, t, 1), e, t, 0, 210, 269, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></li> ")
    }), e.pop()), r.b(' </ul> <div class="channel-slider-container '), r.b(r.v(r.f("style", e, t, 0))), r.b('"> <div class="peek"></div> <div class="channel-slider '), r.b(r.v(r.f("style", e, t, 0))), r.b('"> '), r.s(r.f("slider", e, t, 1), e, t, 0, 410, 705, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="images"> '), n.s(n.f("page", e, t, 1), e, t, 0, 441, 688, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <div class="image-box-nav"> <img src="'), n.b(n.v(n.f("asset_thumbnail_image", e, t, 0))), n.b('" class="image-nav" id="'), n.b(n.v(n.f("id", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" /> <img src="'), n.b(n.v(n.f("asset_thumbnail_hover_image", e, t, 0))), n.b('" class="image-over" id="'), n.b(n.v(n.f("id", e, t, 0))), n.b('-over" /> </div> </a> ')
        }), e.pop()), n.b(" </div> ")
    }), e.pop()), r.b(' </div> </div> </div> </div> <span id="search_cancel_text">'), r.b(r.v(r.f("search_cancel_text", e, t, 0))), r.b("</span> </div>"), r.fl()
}), Whiskers("modules.navigation.disneychannelde", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="nav-local-channels" class="nav-local '), r.b(r.v(r.f("style", e, t, 0))), r.b('"> <div class="'), r.b(r.v(r.f("style", e, t, 0))), r.b('-bound"> <div class="'), r.b(r.v(r.f("style", e, t, 0))), r.b('-bg"> '), r.b(r.rp("logo", e, t, "")), r.b(' <span id="nav-e" role="button" title="Navigate"></span> <ul class="pages"> '), r.s(r.f("links", e, t, 1), e, t, 0, 210, 269, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></li> ")
    }), e.pop()), r.b(' </ul> </div> </div> <span id="search_cancel_text">'), r.b(r.v(r.f("search_cancel_text", e, t, 0))), r.b("</span> </div>"), r.fl()
}), Whiskers("modules.navigation.matterhorn", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="nav-local" class="inverted nav-local '), r.s(r.f("exclude_from_desktop", e, t, 1), e, t, 0, 71, 92, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("excluded_from_desktop")
    }), e.pop()), r.b('"> <div class="bound"> '), r.b(r.rp("logo", e, t, "")), r.b(' <span id="nav-e" role="button" title="Navigate"></span> <ul> '), r.s(r.f("links", e, t, 1), e, t, 0, 221, 264, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></li> ")
    }), e.pop()), r.b(' </ul> <span id="search_cancel_text">'), r.b(r.v(r.f("search_cancel_text", e, t, 0))), r.b("</span> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.$, n = e.Backbone, r = e.GOC || {
        queue: []
    }, i = e.Grill, s = e.Disney.Style.breakpoints, o = e.Whiskers;
    i.ModuleView.register("local_chrome", {
        breakpoints: s.chromeMax,
        index: 0,
        ready: function() {
            this.index = this.$el.index(), this.$el.find("#nav-e").click(function(e) {
                e.preventDefault(), r.queue.push(["openMenu"])
            });
            var n = this.model.get("navigation"), s = e.Backbone.history;
            if (n.template === "channels") {
                var o = s.options.root + s.getFragment();
                s.on("route", function() {
                    var e = t("#nav-local-channels a");
                    e.removeClass("active"), t("#nav-local-channels a[href='" + o + "']").addClass("active"), r.queue.push(["closeMenu"])
                });
                if (t(e).width() >= 1024) {
                    var u = t("#nav-local-channels .channel-slider"), a = u.hasClass("channel") ? .09: .045;
                    u.swipeasaurus({
                        arrows: !0,
                        loop: !0,
                        right: a
                    })
                }
                this.$el.find(".channel-bg > .pages > li > a, .xd-bg > .pages > li > a").each(function() {
                    var e = t(this), n = e.html(), r = n.split(" "), i = '<span class="firstWord">' + r[0] + "</span><br />";
                    r.shift(), t.each(r, function(e, t) {
                        i += " " + t
                    }), e.html(i)
                })
            } else 
                s.on("route", function() {
                    var n = s.options.root + s.getFragment(), i = /^([^\/]+?:\/\/[^\/]+?)\//.exec(e.location.href);
                    i && n.substr(0, 1) === "/" && (n = i[1] + n), t("#nav-local li a[href]").each(function() {
                        t(this).toggleClass("active", this.href === n)
                    }), r.queue.push(["closeMenu"])
                });
            i.ModuleView.create("search_navigation"), this.resize()
        },
        render: function() {
            var e = this.model.get("navigation"), t = o.modules.navigation[e.template];
            this.$el.html(t.render(e, {
                logo: o.partials.logo
            }))
        },
        resize: function() {
            var e = n.Resize.width();
            if (e < s.chromeMax)
                this.$el.parent().prepend(this.$el);
            else if (this.index !== this.$el.index()) {
                var t = this.$el.parent().children();
                this.index >= t.size() - 1 ? this.$el.parent().append(this.$el) : this.$el.insertBefore(t.eq(this.index + 1))
            }
        }
    })
}(this), Whiskers("modules.personality_hero", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 767, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound skip-styles"> <div class="container"> <div class="branding"> <img src="'), n.b(n.v(n.f("imageUrl", e, t, 0))), n.b('"> </div> <div class="details '), n.b(n.v(n.f("layout", e, t, 0))), n.b(" "), n.b(n.v(n.f("portal", e, t, 0))), n.b('" style="background-color: '), n.b(n.v(n.f("bgColor", e, t, 0))), n.b(';"> <h1>'), n.s(n.f("meetPrefix", e, t, 1), e, t, 0, 223, 228, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("Meet ")
        }), e.pop()), n.b('<span class="name">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</span></h1> <div class="headshot"> <span class="personality-box entity-box"><span class="aspect"> <img src="'), n.b(n.v(n.f("imageUrl", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </span></span> </div> <p class="desc">'), n.b(n.v(n.f("description", e, t, 0))), n.b("</p> "), n.s(n.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div id="sharing"> <ul class="social"> '), n.s(n.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || n.b(' <li class="facebook"> <div class="fb-like" data-send="false" data-layout="button_count" data-show-faces="false" data-width="90"></div> </li> '), n.b(" </ul> </div> <!-- End of sharing --> ")), n.b(" </div> </div> </div> ")
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e.Backbone, s = n.Style.breakpoints, o = s.midHigh.toString(), u = e._, a = e.devicePixelRatio > 1.5, f = t.ModuleView.extend({
        render: function() {
            var e = u.first(this.model.get("data")), t = i.Resize.width() >= s.midHigh, r=!!this.options.cds, o = this.$el, f = t ? "none" : this.model.get("style").module_dark_background_color || this.model.collection.style.background_color_dark, l = n.portal === "princess.disney.com", c = this.model.get("style").layout ? this.model.get("style").layout : "left", h = this.model.get("style").images.foreground_image, p = null, d = null, v = null;
            h ? p = a ? h.retina_url ? h.retina_url : h.url : h.url : p = e.hero, d = a ? e.thumb2x ? e.thumb2x : e.thumb : e.thumb ? e.thumb : e.photo ? e.photo : p, v = t ? p : d;
            var m = u.extend({
                imageUrl: v,
                cds: r,
                meetPrefix: l,
                bgColor: f,
                layout: c,
                portal: (n.portal || "").split(".")[0]
            }, this.model);
            o.html(this.template.render(m))
        },
        resize: function() {
            this.render(), n.FB.go()
        },
        ready: function() {
            n.FB.go()
        }
    });
    f.register("personality_hero", {
        className: "module full-height full-width",
        breakpoints: o,
        template: r.modules.personality_hero
    })
}(this), Whiskers("modules.play", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 1193, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <!-- MUSTACHE TEMPLATE - GAME CONTAINER MODULE--> <div class="bound"> <div class="game_wrapper"> <div class="bun_game '), n.b(n.v(n.f("createClass", e, t, 0))), n.b('"> <div id="game_container_outer" class="inverted"> <noscript> <div class="player_error">'), n.b(n.v(n.d("translations.no_javascript", e, t, 0))), n.b("</div> </noscript> "), n.s(n.f("showGameLoader", e, t, 1), e, t, 0, 300, 318, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.t(n.f("gameLoaderAd", e, t, 0)))
        }), e.pop()), n.b(' <div itemscope itemtype="http://schema.org/WebApplication"> <meta itemprop="name" content="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <meta itemprop="description" content="'), n.b(n.v(n.f("desc", e, t, 0))), n.b('"> <meta itemprop="url" content="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <meta itemprop="image" content="'), n.b(n.v(n.f("square", e, t, 0))), n.b('"> <meta itemprop="applicationCategory" content="'), n.b(n.v(n.f("type", e, t, 0))), n.b('"> <meta itemprop="applicationSubCategory" content="'), n.b(n.v(n.f("genres", e, t, 0))), n.b(" "), n.b(n.v(n.f("type", e, t, 0))), n.b('"> <div id="game_container" class="inverted"> <div id=\'DisneyGame-'), n.b(n.v(n.f("id", e, t, 0))), n.b("' contentid='"), n.b(n.v(n.f("id", e, t, 0))), n.b('\'></div> </div> <div class="game_container_too_small"> <div class="orphan"><p>'), n.b(n.v(n.d("translations.too_small", e, t, 0))), n.b('</p></div> </div> <div class="iOS"> <div class="orphan"><p>'), n.b(n.v(n.d("translations.ios", e, t, 0))), n.b('</p></div> </div> <div class="flashNotPresent"> <div class="orphan"><p>'), n.b(n.v(n.d("translations.no_flash", e, t, 0))), n.b('</p></div> </div> </div> </div> </div> </div> </div> <div class="content-toolbar-container"> <div id="meta"></div> </div> ')
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = t.Style.breakpoints, r = [n.midHigh, n.max], i = e.Grill, s = e.Backbone, o = e.Modernizr, u = e._, a = e.jQuery, f = e.Whiskers, l=!1, c=!1, h=!1, p = null, d=!1, v = function(t) {
        var n = e.onerror, r = e.console;
        r && r.error && r.error(t.stack || t.toString()), n && n(t.toString())
    };
    i.ModuleView.register("play", {
        className: "module full-width full-height",
        breakpoints: r.join(" "),
        template: f.modules.play,
        tabletSize: 768,
        isMobile: o.mobile,
        assets: {
            scripts: ["//a.dilcdn.com/a/disneygames-debug-6f0676094850.js"]
        },
        render: function() {
            var t = s.Resize.width() >= this.tabletSize;
            this.gameType = this.model.get("data")[0].game_type, this.isCreateApp() || (this.gameData = {
                wmode: "transparent"
            }, e.disneyGames.processConfig(this.model.get("data")[0].game_data, this.gameData), this.setAdTargeting(this.gameData));
            var n = u.extend({
                cds: this.options.cds,
                createClass: this.isCreateApp() ? "create_module": "game_module",
                showGameLoader: t
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(n))
        },
        isCreateApp: function() {
            return this.gameType === "Create App"
        },
        hideAllMessages: function() {
            this.$(".flashNotPresent").hide(), this.$(".game_container_too_small").hide(), this.$(".create_container_too_small").hide(), this.$(".iOS").hide()
        },
        resize: function() {
            if (h)
                return this.$(".flashNotPresent").hide(), this.$(".game_container_too_small").hide(), this.$(".create_container_too_small").hide(), this.$(".bun_game").addClass("error_holder"), this.$(".iOS").show(), null;
            this.$(".iOS").hide(), c ? (this.$(".bun_game").addClass("error_holder"), this.$(".flashNotPresent").show(), this.$(".game_container_too_small").hide(), this.$(".create_container_too_small").hide()) : this.isMobile || (this.$(".flashNotPresent").hide(), this.isCreateApp() ? this.handleGameContainerShowHide(s.Resize.width() >= n.max) : this.handleGameContainerShowHide(s.Resize.width() >= this.tabletSize))
        },
        handleGameContainerShowHide: function(t) {
            var n = e.disneyGames;
            t ? (l===!1 ? this.loadGameContainer() : n.showGame(), this.$("#game_container").show(), this.$(".bun_game").removeClass("error_holder"), this.$(".game_container_too_small").hide()) : (this.$(".bun_game").addClass("error_holder"), this.$(".game_container_too_small").show(), l===!0 && (n.hideGame(), e.setTimeout(u.bind(this.removeAd, this), 0)), this.$("#game_container").hide())
        },
        setAdTargeting: function(e) {
            var t = this.options.ads;
            t.setTargeting("assetId", e.assetId), t.setTargeting("assetName", e.assetName), t.setTargeting("assetType", e.assetType), t.setTargeting("bu", e.bu), t.setTargeting("gameName", e.gameName), t.setTargeting("gameType", e.gameType), t.setTargeting("genre", e.genre), t.setTargeting("owner", e.owner), t.setTargeting("property", e.property)
        },
        loadGameContainer: function() {
            var t = e.disneyGames, n = this.model.get("data")[0].id;
            l=!0, e.disneyGamesCommunication.addMessageListener(u.bind(this.handleMessage, this));
            if (this.isCreateApp())
                t.addGame({
                    elementId: "DisneyGame-" + n,
                    contentId: this.model.get("data")[0].create_id,
                    provider: "create"
                });
            else {
                var r = {
                    elementId: "DisneyGame-" + n,
                    contentId: this.getContentId(),
                    gameData: this.gameData,
                    iframePath: "/games/syndication/iframeGame.html"
                };
                if (this.options && this.options.ads && this.options.ads.slots) {
                    var i = this.options.ads.slots.gameLoader;
                    i && (r.adDivId = i.id, r.externalAdControl=!0)
                }
                t.addGame(r)
            }
        },
        getContentId: function() {
            var e;
            return this.model.get("data")[0].go_pub_id ? e = this.model.get("data")[0].go_pub_id : e = this.model.get("data")[0].id, e
        },
        removeAd: function() {
            if (this.options && this.options.ads && this.options.ads.slots) {
                var e = this.options.ads.slots.gameLoader;
                e && e.remove()
            }
        },
        handleMessage: function(t) {
            var n = e.disneyGames;
            switch (t) {
            case"adStarted":
                p.hideAllMessages();
                break;
            case"startGame":
                s.Resize.width() <= this.tabletSize&&!this.isMobile && p.resize();
                break;
            case"removeAd":
                e.setTimeout(u.bind(this.removeAd, this), 0), p.resize();
                break;
            case"unsupportedDevice":
                h=!0, p.resize(), this.$("#game_container").hide(), l===!0 && n.hideGame();
                break;
            case"flashFailed":
                c=!0, p.resize(), this.$("#game_container").hide(), l===!0 && n.hideGame();
                break;
            case"HTML5Works":
                this.$(".bun_game").css("padding", "0");
                break;
            default:
            }
        },
        refreshAds: function() {
            a("#nav-pushdown").is(":visible") && this.options.ads.refresh()
        },
        ready: function() {
            var n = this, r = n.model.get("translations");
            p = n, c=!1, h=!1, new t.ContentToolbarView({
                el: this.$("#meta"),
                model: this.model,
                socialLinks: [{
                    share_type: "share",
                    share_text: r.share,
                    output_label: r.quick_link
                }
                ],
                cdsMode: this.options.cds,
                ads: this.options.ads
            }), l===!1 && this.loadGameContainer(), this.hideAllMessages(), this.resize(), !d&&!this.options.cds && (d = e.setInterval(function() {
                n.refreshAds()
            }, 18e4))
        },
        remove: function() {
            try {
                e.disneyGamesCommunication.clearMessageListeners()
            } catch (t) {
                v(t)
            }
            try {
                e.disneyGames.removeGame()
            } catch (t) {
                v(t)
            }
            i.BunView.prototype.remove.call(this), l=!1
        }
    })
}(this), Whiskers("modules.playlist", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <ul class="list"> '), r.s(r.f("data", e, t, 1), e, t, 0, 47, 517, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.s(n.f("videoPlaylist", e, t, 1), e, t, 0, 65, 499, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <li> <div class="order"><span>'), n.b(n.v(n.f("order", e, t, 0))), n.b('</span></div> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <span class="thumb-box"><span class="aspect"> <img src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('" class="thumb" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" '), n.b(n.t(n.f("ctoImgAttr", e, t, 0))), n.b("> "), n.s(n.f("badge", e, t, 1), e, t, 0, 264, 306, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b('<img src="'), n.b(n.v(n.f("badge", e, t, 0))), n.b('" class="badge" alt="">')
            }), e.pop()), n.b(' <span class="thumb-title"> <time>'), n.b(n.v(n.f("duration", e, t, 0))), n.b('</time> </span> </span> </span> </a> <div class="details"> <h3><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></h3> <p>"), n.b(n.v(n.f("description", e, t, 0))), n.b("</p> </div> </li> ")
        }), e.pop())
    }), e.pop()), r.b(" </ul> </div>"), r.fl()
}), Whiskers("modules.playlist_header_image", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container"> '), r.s(r.f("data", e, t, 1), e, t, 0, 53, 671, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="playlist_item"> <a href="'), n.b(n.v(n.f("playlistLink", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <span class="playlist_play_btn"></span> '), n.s(n.f("thumbnails", e, t, 1), e, t, 0, 183, 220, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<img src="'), n.b(n.v(n.f("image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">')
        }), e.pop()), n.b(' </a> </div> <div class="details"> <h2>'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h2> <div class="meta"> <span class="total">'), n.b(n.v(n.f("totalVideos", e, t, 0))), n.b(" "), n.b(n.v(n.d("translations.videos", e, t, 0))), n.b("</span> "), n.s(n.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || n.b(' <span class="fb-like" data-send="false" data-layout="button_count" data-show-faces="false" data-width="90"></span> '), n.b(' </div> <p class="desc">'), n.b(n.v(n.f("description", e, t, 0))), n.b('</p> <a class="button large blue" href="'), n.b(n.v(n.f("playlistLink", e, t, 0))), n.b('">'), n.b(n.v(n.d("translations.watch_playlist", e, t, 0))), n.b("</a> </div> ")
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), Whiskers("modules.featured_playlist", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container"> '), r.s(r.f("data", e, t, 1), e, t, 0, 53, 513, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="playlist_item"> <a href="'), n.b(n.v(n.f("playlistLink", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <span class="playlist_play_btn" ></span> '), n.s(n.f("thumbnails", e, t, 1), e, t, 0, 184, 221, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<img src="'), n.b(n.v(n.f("image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">')
        }), e.pop()), n.b(' </a> </div> <div class="details"> <h3><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</a></h3> <div class="meta"> <span class="total"><strong>'), n.b(n.v(n.f("totalVideos", e, t, 0))), n.b(' Videos</strong><span class="pipeColor"> | </span><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">'), n.b(n.v(n.d("translation.see_playlist", e, t, 0))), n.b("</a></span> </div> <p>"), n.b(n.v(n.f("short_desc", e, t, 0))), n.b("</p> ")
    }), e.pop()), r.b(" </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Whiskers, i = e.jQuery, s = e._, o = n.ModuleView.register("playlist", {
        template: r.modules.playlist,
        render: function() {
            var e = this.model.get("data")[0].id, t = this, n = s.extend({}, t.model);
            t.$el.html(t.template.render(n)), t.$("a").each(function(t, n) {
                i(n).attr("href", i(n).attr("href") + "?playlist=" + e)
            })
        },
        ready: function() {
            t.FB.go()
        }
    });
    o.register("playlist_header_image", {
        template: r.modules.playlist_header_image
    }), o.register("featured_playlist", {
        template: r.modules.featured_playlist
    })
}(this), Whiskers("modules.product", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 36, 54, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="product_list"> '), r.b(r.rp("products", e, t, "")), r.b(" "), r.s(r.f("data", e, t, 1), e, t, 0, 121, 133, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.b(n.rp("modal", e, t, "")), n.b(" ")
    }), e.pop()), r.b(" </div> "), r.s(r.f("hasMore", e, t, 1), e, t, 0, 162, 362, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="show_more_container"> <div class="show_more blue button large" role="button" onclick=""> <span class="label">'), n.b(n.v(n.d("translations.show_more", e, t, 0))), n.b('</span> <span class="spinner"></span> </div> </div> ')
    }), e.pop()), r.b(" </div>"), r.fl()
}), Whiskers("modules.sidekick_product", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 60, 78, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="module_body"> <div class="product_list one ad"> '), r.s(r.f("data", e, t, 1), e, t, 0, 164, 841, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="product"> <div class="leftside"> <div class="boundingBox"><img class="logo" src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('"></div> <div class="product-button"></div> </div> <div class="details"> <h2 class="title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h2> <div class="product-button"></div> <ul class="desc"> '), n.s(n.f("product_lines", e, t, 1), e, t, 0, 444, 458, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("<li>"), n.b(n.v(n.d(".", e, t, 0))), n.b("</li>")
        }), e.pop()), n.b(" </ul> "), n.s(n.f("more_lines", e, t, 1), e, t, 0, 498, 557, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<a class="modal_info button">'), n.b(n.v(n.d("translations.more_info", e, t, 0))), n.b("</a>")
        }), e.pop()), n.b(' </div> <ul class="desc_mobile '), n.s(n.f("more_lines", e, t, 1), e, t, 0, 618, 626, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("hasModal")
        }), e.pop()), n.b('"> '), n.s(n.f("product_lines", e, t, 1), e, t, 0, 662, 676, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("<li>"), n.b(n.v(n.d(".", e, t, 0))), n.b("</li>")
        }), e.pop()), n.b(" "), n.s(n.f("specs", e, t, 1), e, t, 0, 705, 806, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.s(n.f("specifications", e, t, 1), e, t, 0, 724, 787, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <li><span class="tech_title">'), n.b(n.v(n.f("title", e, t, 0))), n.b(":</span> "), n.b(n.v(n.f("value", e, t, 0))), n.b("</li> ")
            }), e.pop())
        }), e.pop()), n.b(" </ul> </div> "), n.b(n.rp("modal", e, t, "")), n.b(" ")
    }), e.pop()), r.b(' </div> <div class="ad-container"> '), r.b(r.t(r.f("rectangleAd", e, t, 0))), r.b(" "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" </div> </div> </div>"), r.fl()
}), function(e, t) {
    "use strict";
    var n = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
    e.fn.imagesLoaded = function(r) {
        function c() {
            var t = e(f), n = e(l);
            s && (l.length ? s.reject(u, t, n) : s.resolve(u)), e.isFunction(r) && r.call(i, u, t, n)
        }
        function h(e) {
            p(e.target, e.type === "error")
        }
        function p(t, r) {
            if (t.src === n || e.inArray(t, a)!==-1)
                return;
            a.push(t), r ? l.push(t) : f.push(t), e.data(t, "imagesLoaded", {
                isBroken: r,
                src: t.src
            }), o && s.notifyWith(e(t), [r, u, e(f), e(l)]), u.length === a.length && (setTimeout(c), u.unbind(".imagesLoaded", h))
        }
        var i = this, s = e.isFunction(e.Deferred) ? e.Deferred(): 0, o = e.isFunction(s.notify), u = i.find("img").add(i.filter("img")), a = [], f = [], l = [];
        return e.isPlainObject(r) && e.each(r, function(e, t) {
            e === "callback" ? r = t : s && s[e](t)
        }), u.length ? u.bind("load.imagesLoaded error.imagesLoaded", h).each(function(r, i) {
            var s = i.src, o = e.data(i, "imagesLoaded");
            if (o && o.src === s) {
                p(i, o.isBroken);
                return 
            }
            if (i.complete && i.naturalWidth !== t) {
                p(i, i.naturalWidth === 0 || i.naturalHeight === 0);
                return 
            }
            if (i.readyState || i.complete)
                i.src = n, i.src = s
        }) : c(), s ? s.promise(i) : i
    }
}(jQuery), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e.Backbone, u = e.jQuery, a = e._, f = t.ModuleView.extend({
        breakpoints: i,
        template: s.modules.product,
        initialize: function() {
            this.itemCount = 0, this.two_button_margin = 10, this.is2Items=!1, this.translations = this.model.get("translations")
        },
        render: function() {
            a.extend(this, {
                itemCount: this.model.get("data").length,
                allProducts: this.model.get("data")
            }), this.allProducts.length > 4 ? (this.productData = this.allProducts.slice(0, 4), this
            .stashedData = this.allProducts.slice(4)) : this.productData = this.allProducts, this.is2Items = this.itemCount !== 1 && this.itemCount !== 3?!0 : !1;
            var e = a.extend({
                is2Items: this.is2Items,
                hasMore: this.itemCount > 4,
                title: this.model.get("title"),
                desc: this.model.get("desc"),
                product_data: this.productData
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(e, {
                products: s.entities.products,
                modal: s.entities.product_modal
            })), n.Style.moduleStyle(this), this.updateButton(this.itemCount > 4)
        },
        resize: function() {
            var e = o.Resize.width() >= r.midHigh;
            this.options.ads.restore(), e ? this.$(".desc_mobile, .modal_info").removeClass("fade") : this.$(".product-tech, .product").removeClass("fade")
        },
        allImagesLoaded: function(e) {
            var t = this;
            for (var n = 0; n < e.length; n++) {
                var r = e[n];
                if (t.is2Items) {
                    var i = u(r).parent(), s = r.clientHeight - i.height() + t.two_button_margin;
                    i.parent().find(".product-button").css("margin-top", s + "px")
                }
            }
        },
        ready: function() {
            this.buttonViews = [];
            var e = this, t = this.$(".product_list .product"), r = this.$(".product_list"), i = this.$(".product-tech").length;
            switch (t.length) {
            case 1:
                r.addClass("one");
                break;
            case 3:
                r.addClass("three");
                break;
            default:
                r.addClass("two")
            }
            this.$(".product_list .product").each(function(t, r) {
                e.$(r).find(".product-button").each(function(r, i) {
                    var s = new n.ProductButtonView({
                        el: i,
                        model: e.model.get("data")[t],
                        translations: e.model.get("translations"),
                        age_gate: e.model.get("age_gate")
                    });
                    s.on("change", e.buttonChanged, e), e.buttonViews.push(s)
                })
            }), this.$(".product-tech").each(function(t, r) {
                e.$(r).find(".product-button").each(function(r, i) {
                    var s = new n.ProductButtonView({
                        el: i,
                        model: e.model.get("data")[t],
                        age_gate: e.model.get("age_gate")
                    });
                    s.on("change", e.buttonChanged, e), e.buttonViews.push(s)
                })
            });
            for (var s = 0; s < i; s++)
                this.$(".modal_info").eq(s).addClass("" + s)
        },
        updateButton: function(e) {
            var t = this.$el;
            t.toggleClass("has-more", e)
        },
        modalDisplay: function(e) {
            var t = o.Resize.width() >= r.midHigh, n = this.$(".product-tech").length, i = null;
            if (t) {
                this.$(".product_list .product").addClass("fade");
                for (i = 0; i < n; i++)
                    u(e.currentTarget).hasClass(i) && this.$(".product-tech").eq(i).addClass("fade")
            } else 
                for (i = 0; i < n; i++)
                    u(e.currentTarget).hasClass(i) && this.$(".desc_mobile").eq(i).addClass("fade")
        },
        modalRemove: function() {
            this.$(".product-tech, .product, .desc_mobile").removeClass("fade"), this.$(".buy").removeClass("active")
        },
        events: {
            "click .modal_info": "modalDisplay",
            "click .close_modal": "modalRemove",
            "click .product.fade": "modalRemove",
            "click .show_more_container": function(e) {
                e.preventDefault(), this.buttonViews = [];
                var t = this, r = this.productData, i = this.stashedData, o = this.$(".product_list");
                i.length > 4 ? (r = i.slice(0, 4), i = i.slice(4)) : (r = i, i = []);
                var f = a.extend({
                    is2Items: !0,
                    hasMore: i.length > 0,
                    product_data: r
                }), l = s.entities.products.render(f);
                u(l).insertAfter(o.find(".product").last()), this.$(".product_list .product").slice(1).each(function(e, r) {
                    t.$(r).find(".product-button").slice(0).each(function(r, i) {
                        var s = new n.ProductButtonView({
                            el: i,
                            model: t.model.get("data")[e],
                            age_gate: t.model.get("age_gate")
                        });
                        s.on("change", t.buttonChanged, t), t.buttonViews.push(s)
                    })
                });
                for (var c = 4; c < r.length + 4; c++)
                    this.$(".modal_info").eq(c).addClass("" + c);
                this.updateButton(!1)
            }
        }
    });
    f.register("product", {}), f.register("sidekick_product", {
        className: "module sidekick",
        template: s.modules.sidekick_product
    })
}(this), Whiskers("modules.property_hero", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 735, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="branding"> <div class="scaler"></div> '), n.s(n.f("wide", e, t, 1), e, t, 0, 69, 286, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("overlayLeft", e, t, 1), e, t, 0, 86, 159, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <span class="left-overlay overlay"> <img src="'), n.b(n.v(n.f("overlayLeft", e, t, 0))), n.b('"> </span> ')
            }), e.pop()), n.b(" "), n.s(n.f("overlayRight", e, t, 1), e, t, 0, 193, 268, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <span class="right-overlay overlay"> <img src="'), n.b(n.v(n.f("overlayRight", e, t, 0))), n.b('"> </span> ')
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(' <span class="background"> <img src="'), n.b(n.v(n.f("propertyHeader", e, t, 0))), n.b('"> </span> </div> '), n.s(n.f("wide", e, t, 1), e, t, 0, 377, 725, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="content '), n.b(n.v(n.f("contentPosition", e, t, 0))), n.b(' skip-styles"> '), n.s(n.f("video", e, t, 1), e, t, 0, 442, 707, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" "), n.s(n.f("id", e, t, 1), e, t, 0, 450, 699, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(' <div class="video-container"> <div class="scaler"></div> '), n.s(n.f("expired", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div id="player_property_hero" class="player video trailer-box"> <noscript> <div class="player_error">'), n.b(n.v(n.d("translations.no_javascript", e, t, 0))), n.b("</div> </noscript> </div> ")), n.b(" </div> ")
                }), e.pop()), n.b(" ")
            }), e.pop()), n.b(" </div> ")
        }), e.pop()), n.b(" ")
    }), e.pop()), r.fl()
}), Whiskers("modules.property_nav", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 2067, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <style type="text/css"> '), n.s(n.f("pn_style", e, t, 1), e, t, 0, 47, 1730, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" .property_nav_wrap{ background-color: "), n.b(n.v(n.f("background_color", e, t, 0))), n.b("; background-image: -webkit-gradient(linear, left top, left bottom, from("), n.b(n.v(n.f("gradient_color", e, t, 0))), n.b("), to("), n.b(n.v(n.f("background_color", e, t, 0))), n.b(")); background-image: -webkit-linear-gradient(top, "), n.b(n.v(n.f("gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("background_color", e, t, 0))), n.b("); background-image: -moz-linear-gradient(top, "), n.b(n.v(n.f("gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("background_color", e, t, 0))), n.b("); background-image: -o-linear-gradient(top, "), n.b(n.v(n.f("gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("background_color", e, t, 0))), n.b("); background-image: linear-gradient(to bottom, "), n.b(n.v(n.f("gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("background_color", e, t, 0))), n.b("); } .property_nav_wrap ul li:hover{ background-color: "), n.b(n.v(n.f("hover_color", e, t, 0))), n.b("; background-image: -webkit-gradient(linear, left top, left bottom, from("), n.b(n.v(n.f("hover_gradient_color", e, t, 0))), n.b("), to("), n.b(n.v(n.f("hover_color", e, t, 0))), n.b(")); background-image: -webkit-linear-gradient(top, "), n.b(n.v(n.f("hover_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("hover_color", e, t, 0))), n.b("); background-image: -moz-linear-gradient(top, "), n.b(n.v(n.f("hover_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("hover_color", e, t, 0))), n.b("); background-image: -o-linear-gradient(top, "), n.b(n.v(n.f("hover_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("hover_color", e, t, 0))), n.b("); background-image: linear-gradient(to bottom, "), n.b(n.v(n.f("hover_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("hover_color", e, t, 0))), n.b("); } .property_nav_wrap ul li.active{ background-color: "), n.b(n.v(n.f("active_color", e, t, 0))), n.b("; background-image: -webkit-gradient(linear, left top, left bottom, from("), n.b(n.v(n.f("active_gradient_color", e, t, 0))), n.b("), to("), n.b(n.v(n.f("active_color", e, t, 0))), n.b(")); background-image: -webkit-linear-gradient(top, "), n.b(n.v(n.f("active_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("active_color", e, t, 0))), n.b("); background-image: -moz-linear-gradient(top, "), n.b(n.v(n.f("active_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("active_color", e, t, 0))), n.b("); background-image: -o-linear-gradient(top, "), n.b(n.v(n.f("active_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("active_color", e, t, 0))), n.b("); background-image: linear-gradient(to bottom, "), n.b(n.v(n.f("active_gradient_color", e, t, 0))), n.b(", "), n.b(n.v(n.f("active_color", e, t, 0))), n.b("); } .property_nav_wrap ul li a{ color: "), n.b(n.v(n.f("font_color", e, t, 0))), n.b(" }; ")
        }), e.pop()), n.b(' </style> <div class="property_nav_wrap"> <div class="property_nav_scroller"> <ul class="links-'), n.b(n.v(n.f("numLinks", e, t, 0))), n.b('"> '), n.s(n.f("pn_links", e, t, 1), e, t, 0, 1866, 2033, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<li class="'), n.s(n.f("icon", e, t, 1), e, t, 0, 1886, 1890, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b("icon")
            }), e.pop()), n.b('"><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"'), n.s(n.f("ping", e, t, 1), e, t, 0, 1946, 1967, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' data-ping="'), n.b(n.v(n.f("ping", e, t, 0))), n.b('"')
            }), e.pop()), n.b(">"), n.s(n.f("icon", e, t, 1), e, t, 0, 1986, 2006, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b('<img src="'), n.b(n.v(n.f("icon", e, t, 0))), n.b('">')
            }), e.pop()), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></li>")
        }), e.pop()), n.b(" </ul> </div> </div> ")
    }), e.pop()), r.fl()
}), Whiskers("modules.property_description", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="module_body"> '), r.s(r.f("data", e, t, 1), e, t, 0, 55, 905, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="content" data-title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <div class="header"> <h1 style="color:'), n.s(n.f("style", e, t, 1), e, t, 0, 149, 166, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.v(n.f("heading_color", e, t, 0)))
        }), e.pop()), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</h1> <div class="meta"> '), n.s(n.f("rating", e, t, 1), e, t, 0, 223, 300, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p><span class="meta-heading">'), n.b(n.v(n.d("translations.rated", e, t, 0))), n.b(":</span> "), n.b(n.v(n.f("rating", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(" "), n.s(n.f("release", e, t, 1), e, t, 0, 324, 409, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <p><span class="meta-heading">'), n.b(n.v(n.d("translations.release_date", e, t, 0))), n.b(":</span> "), n.b(n.v(n.f("release", e, t, 0))), n.b("</p> ")
        }), e.pop()), n.b(' </div> </div> <p class="desc">'), n.b(n.v(n.f("desc", e, t, 0))), n.b("</p> "), n.s(n.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (n.b('<div class="movie-actions"> '), n.s(n.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || n.b(' <ul class="social"> <li class="facebook"> <div class="fb-like" data-send="false" data-layout="button_count" data-show-faces="false" data-width="90"></div> </li> </ul> '), n.b(" "), n.s(n.f("buttons", e, t, 1), e, t, 0, 722, 870, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" target="_blank" class="button small '), n.b(n.v(n.f("type", e, t, 0))), n.b(" "), n.b(n.v(n.f("button_color", e, t, 0))), n.b('"'), n.s(n.f("convert", e, t, 1), e, t, 0, 816, 843, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' data-convert="'), n.b(n.v(n.f("convert", e, t, 0))), n.b('"')
            }), e.pop()), n.b(">"), n.b(n.v(n.f("title", e, t, 0))), n.b("</a> ")
        }), e.pop()), n.b(" </div>")), n.b(" </div> ")
    }), e.pop()), r.b(' <div class="ad-container"> '), r.b(r.t(r.f("rectangleAd", e, t, 0))), r.b(" "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    function c(e) {
        return e.split("?")[0]
    }
    var t = e.Disney, n = t.Style.breakpoints, r = n.midHigh.toString(), i = e.Grill, s = e.Whiskers, o = e.Backbone, u = e.jQuery, a = e._, f = i.ModuleView.extend({
        breakpoints: r,
        render: function() {
            var e = a.first(this.model.get("data")), r = e.background_styles, i=!!this.options.cds, s = o.Resize.width() >= n.midHigh, f = a.extend({
                cds: i,
                wide: s,
                propertyHeader: s ? e.hero: e.hero_mobile,
                numLinks: e.pn_links ? e.pn_links.length: 0
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(f)).toggleClass("cds", i), this.$el.addClass(this.model.cid), t.Style.moduleStyle(this, !1), r && (u("#takeover-area").addClass("module-takeover"), t.Style.backgroundStyles(r))
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    });
    f.register("property_hero", {
        template: s.modules.property_hero,
        initialize: function() {
            this.property().get("video") && (this.assets = {
                scripts: ["//a.dilcdn.com/a/videoplayer-e4e73a5b3283.js"]
            })
        },
        property: function() {
            return this.model.data.first()
        },
        resize: function() {
            this.render(), this.options.ads.restore(), this.ready()
        },
        ready: function() {
            var e = o.Resize.width();
            if (e < n.midHigh)
                return;
            var r = this.property(), i = r.get("video"), s = r.get("image");
            i && (this.player = new t.MatterhornVideoPlayer("player_property_hero", {
                video: i,
                clickToPlay: !0,
                poster: s,
                ads: {
                    enabled: !1
                }
            }))
        },
        remove: function() {
            this.player && (this.player.destroy(), this.player = null), i.ModuleView.prototype.remove.call(this)
        }
    }), f.register("property_description", {
        template: s.modules.property_description,
        breakpoints: null,
        ready: function() {
            t.FB.go()
        }
    });
    var l = f.extend({
        className: "module property_nav",
        template: s.modules.property_nav,
        ready: function() {
            this.$("a").each(function() {
                u(this).parent("li").toggleClass("active", c(this.href) === c(e.location.href))
            })
        }
    });
    l.register("property_nav"), l.register("property_nav_mobile", {
        className: "module property_nav parks-store-nav"
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = t.Style.breakpoints, r = n.midHigh.toString(), i = e.Grill, s = e.Whiskers, o = e.Backbone, u = e._, a = e.devicePixelRatio > 1.5;
    i.ModuleView.register("rich_image", {
        breakpoints: r,
        className: "module full-width full-height",
        template: s.modules.rich_image,
        render: function() {
            var e = o.Resize.width() >= n.midHigh, r = this, i = u.first(this.model.get("data")), s = r.model.get("style").images.foreground_image, f = null;
            if (s) {
                var l = a ? s.retina_url ? s.retina_url: s.url: s.url, c = a ? s.mobile_retina_url ? s.mobile_retina_url: s.mobile_url ? s.mobile_url: null: s.mobile_url ? s.mobile_url: null;
                f = e ? l : c
            }
            var h = u.extend({
                image: f ? f: null,
                data: i ? i: null
            }, this.model);
            this.$el.html(this.template.render(h)), t.Style.moduleStyle(this, !1)
        },
        resize: function() {
            this.render()
        }
    })
}(this), Whiskers("modules.rich_text", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="rich_text_container"> '), r.s(r.f("style", e, t, 1), e, t, 0, 64, 208, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.s(n.f("images", e, t, 1), e, t, 0, 76, 196, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("foreground_image", e, t, 1), e, t, 0, 98, 174, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <div class="rich_text_image '), n.b(n.v(n.f("alignment", e, t, 0))), n.b('_align"><img src="'), n.b(n.v(n.f("url", e, t, 0))), n.b('"></div> ')
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(" ")
    }), e.pop()), r.b(" "), r.s(r.f("rich_text", e, t, 1), e, t, 0, 233, 284, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="rich_text_body">'), n.b(n.t(n.f("rich_text", e, t, 0))), n.b("</div> ")
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Whiskers, i = e._;
    n.ModuleView.register("rich_text", {
        template: r.modules.rich_text,
        render: function() {
            var e = i.first(this.model.get("data")), n = i.extend({
                data: e
            }, this.model);
            this.$el.html(this.template.render(n)), t.Style.moduleStyle(this, !1)
        }
    })
}(this), Whiskers("modules.search", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<form class="search-form goc-search" action="'), r.b(r.v(r.f("search_action", e, t, 0))), r.b('" method="GET"> <a class="search-nav-logo" href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('"> </a> '), r.b(' <a class="textSearchHeading" href="'), r.b(r.v(r.f("href", e, t, 0))), r.b('">'), r.b(r.v(r.d("translations.search", e, t, 0))), r.b('</a> <div class="query-field"> '), r.b(' <input name="q" type="text" placeholder="" id="searchField_SRP"> </div> <input class="large button blue" type="submit" value="'), r.b(r.v(r.d("translations.search", e, t, 0))), r.b('"> </form>'), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.GOC, r = e.jQuery, i = e.Whiskers, s = e.Tracker, o = e._;
    t.BunView.register("search", {
        template: i.modules.search,
        ready: function() {
            n && n.queue.push(["ac", this.el, "srp/searchbox"]), r('input[name="q"]').focus()
        },
        events: {
            "click a": function(e) {
                var t = r(e.currentTarget), n = t.attr("href");
                e.target.nodeName.toLowerCase() === "img" && (t = r(e.target)), s.link({
                    linkName: t.data("linkname"),
                    linkPosition: t.data("linkposition"),
                    href: n
                })
            },
            "submit form.goc-search": function() {
                s.link({
                    linkName: "srp/searchbox/search_enter",
                    linkPosition: "srp/searchbox"
                })
            }
        },
        render: function() {
            var e = this, t = r(".goc-bound .goc-search").attr("action") || "/search", n = o.extend({
                search_action: t
            }, e.model);
            e.$el.html(e.template.render(n))
        }
    })
}(this), Whiskers("modules.search_stream", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b(" "), r.s(r.f("item_titles_below", e, t, 1), e, t, 0, 64, 81, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("item-titles-below")
    }), e.pop()), r.b('"> <div class="header module_header"> '), r.s(r.f("title", e, t, 1), e, t, 0, 151, 169, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("<h2>"), n.b(n.v(n.f("title", e, t, 0))), n.b("</h2>")
    }), e.pop()), r.b(' <div class="filters" data-title="filters"> '), r.s(r.f("sortOptions", e, t, 1), e, t, 0, 239, 342, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="sort"> <a href="" class="popular active">Popular</a><a href="" class="new">New</a> </div> ')
    }), e.pop()), r.b(' <a href="" class="filter"> <span>All Games</span> </a> <div class="filter_mobile"> <span>All Games</span> <select class="mobile_filter_list"> <option value="0" data-filter="All" selected="selected">All Games</option> </select> </div> </div> </div> <div class="filter_drawer"> <div class="filter_list"> </div> </div> <div id="stream_items"> '), r.b(r.t(r.f("list", e, t, 0))), r.b(' </div> <div class="show_more_container"> <div class="show_more blue button large" role="button" onclick=""> <span class="label">Show More</span> <span class="spinner"></span> </div> </div> </div>'), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Backbone, i = e.Modernizr, s = e.Whiskers, o = e.Tracker, u = e.jQuery, a = e._, f = t.Style.breakpoints, l = a.values(f), c = r.LazyCollection.extend({
        filters: [],
        parse: function(e) {
            return this.setFilters(e.filters), this.setCount(e.total_found), e.data
        },
        setFilters: function(e) {
            this.filters = e
        },
        setCount: function(e) {
            this.count = e
        }
    }), h = n.ModuleView.extend({
        sortTypes: ["created_at", "popularity"],
        className: "module stream-view",
        compatibility: i.canvas && i.flash ? "html5,flash": i.canvas&&!i.flash ? "html5": "flash",
        page: 1,
        events: {
            "click .show_more": function(e) {
                var t = this;
                e.preventDefault(), this.getMoreItems().then(function() {
                    t.showHideButton(), o.pagination(++t.page)
                })
            },
            "click .sort a.popular": function(e) {
                var t = this;
                e.preventDefault(), t.$(".popular").addClass("active"), t.$(".new").removeClass("active"), this.hideStreamItems()
            },
            "click .sort a.new": function(e) {
                var t = this;
                e.preventDefault(), t.$(".new").addClass("active"), t.$(".popular").removeClass("active"), this.hideStreamItems()
            },
            "click .filters a.filter": function(e) {
                var t = this;
                e.preventDefault(), t.$(".filters a.filter").toggleClass("active");
                var n = u(".filter_list").height() + 44;
                t.$(".filter_drawer").hasClass("active") ? t.$(".filter_drawer").css("height", "0").removeClass("active") : t.$(".filter_drawer").css("height", n).addClass("active")
            }
        },
        initialize: function() {
            var e = this, t = e.model, r = t.data = new c(null, {
                min: t.data.min,
                params: {
                    limit: "l",
                    offset: "o"
                }
            });
            this.itemsRetrieved=!1, this.readyCalled=!1, this.list = new n.ListView({
                collection: r,
                template: {
                    render: function(e) {
                        return s.modules.col.render(e, {
                            entity: s.entities[t.get("type")]
                        })
                    }
                }
            }), r.on("loading", function(t, n) {
                e.updateButton(n)
            }), this.resetAndGetMoreItems().then(function() {
                e.itemsRetrieved=!0, e.createUI(e.model.data.filters), e.showHideButton()
            })
        },
        getMoreItems: function() {
            var e = this, t = r.Resize.width(), n = e.list.collection, i = this.$(".filter_drawer a.active").data("filter") || "All", s = this.$("select.mobile_filter_list option:selected").data("filter") || "All", o = this.$(".sort > .active").text().toLowerCase(), u = o !== "new" ? e.sortTypes[1]: e.sortTypes[0];
            t < 768 && (i = s);
            var a = "/search-filter.json?f=" + i + "&s=" + u + "&c=" + e.compatibility;
            return n.url = a, this.itemCount += this.cols() * this.rows(), this.model.data.ensure(this.itemCount)
        },
        showHideButton: function() {
            this.model.data.length >= this.model.data.count ? this.$(".show_more_container").hide() : this.$(".show_more_container").show()
        },
        hideStreamItems: function() {
            var t = this, n = this.$("#stream_items ul"), r = this.$("#stream_items"), i = n.outerHeight(), s = u(e).selectTransitionEnd() + ".stream_items";
            n.css("height", i).addClass("hide"), r.addClass("load"), u.support.css3("transition") ? n.on(s, function(e) {
                this === e.target && (t.resetAndGetMoreItems(), n.off(s))
            }) : t.resetAndGetMoreItems()
        },
        render: function() {
            var e = this.list, n = {
                title: this.model.get("title"),
                type: this.model.get("type"),
                item_titles_below: this.model.get("item_titles_below"),
                list: t.placeHtml(e),
                sortOptions: !0
            };
            this.$el.html(this.template.render(n)), t.Style.moduleStyle(this), t.placeSwap(this, e), this.updateButton(!1)
        },
        ready: function() {
            this.readyCalled=!0, this.resize(), this.createUI(this.model.data.filters)
        },
        updateButton: function(e) {
            var t = this.$el, n = this.model.data;
            t.toggleClass("loading", e), t.toggleClass("has-more", n.count > 0)
        },
        resize: function() {
            var e = this.cols();
            this.list.$el.removeClass().addClass("cols-" + e), this.fit()
        },
        fit: function() {
            var e = this.list.collection.count, t = this.list.collection.length, n = this.page * this.cols() * this.rows();
            n < e && t < e ? this.list.$el.children().show().slice(this.page * this.cols() * this.rows()).hide() : this.list.$el.children().show()
        },
        createUI: function(e) {
            if (!this.itemsRetrieved ||!this.readyCalled)
                return;
            var t = this, n = e, r = this.$(".filter_list"), i = this.$(".mobile_filter_list"), s = this.model.get("all_button_text");
            n ? (r.append('<a class="active" href="" data-filter="All"><span>All ' + s + "</span></a>"), this.$(".filters a.filter span").replaceWith("<span>All " + s + "</span>"), this.$(".filter_mobile span").replaceWith("<span>All " + s + "</span>"), u.each(n, function(e) {
                var t = n[e];
                r.append('<a href="" data-filter="' + t + '"><span>' + t + "</span></a>"), i.append('<option value="' + (e + 1) + '" data-filter="' + t + '">' + t + "</option>")
            }), this.$(".filter_drawer a").click(function(e) {
                e.preventDefault(), t.$(".filter_drawer a").removeClass("active"), u(this).addClass("active"), t.$(".filter_drawer").css("height", "0").removeClass("active"), t.$(".filters a.filter").removeClass("active"), u(this).data("filter") === "All" ? (t.$(".filters a.filter span").replaceWith("<span>All " + s + "</span>"), t.$(".filter_mobile span").replaceWith("<span>All " + s + "</span>")) : (t.$(".filters a.filter span").replaceWith("<span>" + u(this).data("filter") + "</span>"), t.$(".filter_mobile span").replaceWith("<span>" + u(this).data("filter") + "</span>")), t.hideStreamItems()
            }), this.$(".mobile_filter_list").change(function() {
                t.$(".mobile_filter_list option:selected").each(function() {
                    u(this).data("filter") === "All" ? (t.$(".filters a.filter span").replaceWith("<span>All " + s + "</span>"), t.$(".filter_mobile span").replaceWith("<span>All " + s + "</span>")) : (t.$(".filters a.filter span").replaceWith("<span>" + u(this).data("filter") + "</span>"), t.$(".filter_mobile span").replaceWith("<span>" + u(this).data("filter") + "</span>")), t.$(".mobile_filter_list").blur(), t.hideStreamItems()
                })
            }), r.css("display", "")) : (r.hide(), this.$(".filters a.filter").hide(), this.$(".filter_mobile").hide()), this.$(".selectBox").append('<div class="arrow_down"></div>')
        },
        resetAndGetMoreItems: function() {
            var e = this, t = e.list.collection;
            this.page = 1, this.itemCount = 0, t.reset();
            var n = this.getMoreItems();
            return n.then(function() {
                e.itemsRetrieved=!0, e.showHideButton(), e.$("#stream_items").removeClass("load"), e.$("#stream_items ul").css("height", "auto").removeClass("hide")
            }), this.$(".arrow_down").length || this.$(".selectBox").append('<div class="arrow_down"></div>'), n
        }
    });
    h.register("search_stream", {
        breakpoints: l.join(" "),
        template: s.modules.search_stream,
        rows: function() {
            var e = r.Resize.width();
            return e < f.midHigh ? 4 : e < f.max ? 4 : 3
        },
        cols: function() {
            var e = r.Resize.width(), n = this.model.get("type");
            switch (n) {
            case"movie":
                return t.portal === "princess.disney.com" ? e < f.midHigh ? 1 : e < f.max ? 3 : 4 : e < f.midHigh ? 1 : e < f.max ? 4 : 5;
            case"video":
                return e < f.midHigh ? 1 : e < f.max ? 4 : 5;
            default:
                return e < f.midHigh ? 1 : e < f.max ? 3 : 4
            }
        }
    })
}(this), Whiskers("modules.shoutout", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 53, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="container"> '), n.b(n.rp("entity", e, t, "")), n.b(" </div> ")
    }), e.pop()), r.fl()
}), Whiskers("modules.comments", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 53, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="container"> '), n.b(n.rp("entity", e, t, "")), n.b(" </div> ")
    }), e.pop()), r.fl()
}), function(e, t) {
    "use strict";
    function s(e) {
        var n = t(e.currentTarget), r = n.val(), s = n.attr(i);
        s && r === s && (n.val(""), n.removeClass("empty"))
    }
    function o(e) {
        var n = t(e.currentTarget), r = n.val(), s = n.attr(i);
        s&&!r && (n.val(s), n.addClass("empty"))
    }
    var n, r = "input,textarea", i = "placeholder";
    i in e.document.createElement("input") ? n = function() {
        return this
    } : n = function() {
        return this.on("focusin", r, s), this.on("focusout", r, o), this.find(r).each(function() {
            o({
                currentTarget: this
            })
        }), this
    }, t.fn.polyfillPlaceholder = n
}(this, jQuery), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e.Backbone, u = e.jQuery, a = e._, f = e.Tracker, l = t.ListView.extend({}), c = o.LazyCollection.extend({
        initialize: function(e, t) {
            this.options = t || {}
        },
        setItemEmblem: function(e) {
            var t = this.options.emblems;
            if (e.emblem && t) {
                var n = t.get(e.emblem);
                e.emblem = n ? n.toJSON() : null
            }
            return e
        },
        parse: function(e) {
            this.count = e.number_of_messages_stored;
            var t = this;
            return a.each(e.msgs, function(e) {
                e.id = e._id, delete e._id, t.setItemEmblem(e)
            }), e.msgs
        },
        remaining: function() {
            return this.count || this.count === 0 ? this.count - this.length : null
        }
    }), h = o.LazyCollection.extend({
        pages: o.LazyCollection.prototype.pages
    }), p = o.View.extend({
        className: "container",
        breakpoints: i,
        page: 1,
        posts: 0,
        events: {
            "click .show_more_container": function(e) {
                e.preventDefault(), this.getNextPage()
            },
            "change select": function() {
                this.updateSelect()
            },
            'keyup [name="shoutout"]': function() {
                this.updateRemaining()
            },
            'keypress [name="shoutout"]': function(e) {
                if (e.which !== 0 && e.which !== 8) {
                    var t = u(e.currentTarget), n = t.val() || "", r = t.attr("maxlength");
                    r && n.length >= r && e.preventDefault()
                }
            },
            'change [name="shoutout"]': function() {
                this.updateRemaining()
            },
            'keypress input[name="name"]': function(e) {
                if (e.which !== 0 && e.which !== 8) {
                    var t = String.fromCharCode(e.which);
                    /[a-z0-9]/i.test(t) || e.preventDefault()
                }
            },
            'change input[name="name"]': function(e) {
                var t = u(e.currentTarget), n = t.val() || "";
                t.val(n.replace(/[^a-z0-9]/gi, ""))
            },
            "click div.emblem": function() {
                this.toggleEmblemPicker()
            },
            "click div.emblems ul li": function(e) {
                var t = u(e.currentTarget), n = t.data("id");
                if (!n)
                    return;
                var r = this.model.emblems.get(n);
                this.setEmblem(r), this.hideEmblemPicker()
            },
            "submit form": function(e) {
                var t = u(e.currentTarget);
                this.processForm(t), e.preventDefault()
            }
        },
        initialize: function() {
            a.bindAll(this), this.template = this.options.template || this.template, this.message_template = this.options.message_template || this.message_template, this.animateMessageIntoView = this.options.animateMessageIntoView || this.animateMessageIntoView;
            var e = this.model.get("shoutout_id"), t = this.model.get("cws_url"), n = this.model.get("cws_tenant");
            this.filter_url = t + "/language_filter";
            var r = this.model.emblems = new h(this.model.get("emblems")), i = this.model.data = new c(null, {
                min: this.rows(),
                params: {
                    limit: "length"
                },
                emblems: r
            });
            i.url = t + "/shoutouts/tenants/" + encodeURIComponent(n) + "/questions/" + encodeURIComponent(e) + "/messages", i.on("loading", this.onDataLoading), this.messages_view = new l({
                collection: i,
                template: this.message_template
            }), this.emblemPickerOpen=!1, this.currentEmblem = r.at(0), i.ensure(this.rows())
        },
        resize: function(e) {
            if (e < r.midHigh) {
                var t = this.model.data, n = t.models.slice(0, this.rows());
                t.reset(n), t.ensure(this.rows())
            }
        },
        trackLink: function(e) {
            var t = this.options, n = ["shoutout", "mod" + t.moduleView.index, "item" + t.index, "button"], r = n.join("/"), i = r + "/" + e.replace(/[^a-z0-9 ]/gi, "").replace(/\s+/g, "-");
            f.link({
                linkName: i,
                linkPosition: r
            })
        },
        getNextPage: function() {
            var e = this.model.data;
            if (e.loading() || e.remaining() === 0)
                return !1;
            var t=++this.page * this.rows();
            this.trackLink("get more"), this.model.data.ensure(t).then(function() {})
        },
        onScroll: function(e) {
            var t = u(e.currentTarget), n = t.children(), r = t.height(), i = n.height(), s = t.scrollTop(), o = i - r - s;
            o < .5 * r && this.getNextPage()
        },
        onDataLoading: function(e, t) {
            this.loading = t, this.updateShowMore()
        },
        onMessageSaveSuccess: function() {
            this.posts += 1, this.trackLink("submit"), this.posts >= 3 ? this.updateSubmitButton({
                disabled: !0
            }) : this.updateSubmitButton();
            var e = this._pendingMessage;
            this._pendingMessage = undefined, e.set("pending", !0), this.model.data.unshift(this.model.data.setItemEmblem(e.toJSON())), this.animateMessageIntoView(this.messages_view.$("li:first-child")), this.clearForm(), this.model.data.count += 1, this.updateCount()
        },
        onMessageSaveError: function(e) {
            typeof e != "string" && (e = "Saving has failed, sorry!"), this.trackLink("submit-error"), this.showAlert(e), this.updateSubmitButton()
        },
        animateMessageIntoView: function(e) {
            e.addClass("show")
        },
        toggleEmblemPicker: function() {
            this.emblemPickerOpen ? this.hideEmblemPicker() : this.showEmblemPicker()
        },
        updateRemaining: function() {
            var e = this.$(".remaining"), t = this.$('[name="shoutout"]'), n = t.attr("maxlength"), r = t.hasClass("empty") ? 0: (t.val() || "").length, i = n - r;
            e.text(i)
        },
        setEmblem: function(e) {
            var t = e ? e.attributes: {}, n = u("<img/>").attr({
                src: t.large_image
            }), r = this.$("div.emblem .avatar");
            r.empty(), t.large_image && r.append(n), this.currentEmblem = e
        },
        updateCount: function() {
            var e = this.$(".count");
            e.text(this.model.data.count ? this.model.data.count : 0)
        },
        updateShowMore: function() {
            var e = this.$(".show_more_container .show_more"), t = this.$(".show_more_container .loading");
            if (this.loading)
                e.hide(), t.show();
            else {
                var n = this.model.data.remaining();
                n === null || n > 0 ? e.show() : e.hide(), t.hide(), this.updateCount()
            }
        },
        showEmblemPicker: function() {
            if (this.emblemPickerOpen)
                return;
            this.$(".emblems").addClass("active"), this.emblemPickerOpen=!0
        },
        hideEmblemPicker: function() {
            if (!this.emblemPickerOpen)
                return;
            this.$(".emblems").removeClass("active"), this.emblemPickerOpen=!1
        },
        render: function() {
            var e = this.model.toJSON();
            this.$el.html(this.template.render(e)), this.$el.find(".list").append(this.messages_view.$el), this.updateSelect(), this.setEmblem(this.currentEmblem), this.updateRemaining(), this.bindScroll(), this.updateCount(), this.updateShowMore(), this.$el.polyfillPlaceholder()
        },
        bindScroll: function() {
            this.$(".list").scroll(this.onScroll)
        },
        updateSelect: function() {
            var e = this.$("select");
            e.val() ? e.removeClass("empty") : e.addClass("empty")
        },
        rows: function() {
            return 6
        },
        clearForm: function() {
            var e = this.$("form");
            e.find('[name="shoutout"]').val(""), e.find('[name="name"]').val(""), e.find('[name="age"]').val(""), e.find('[name="state"]').val(""), this.updateSelect(), this.addPlaceholders && this.addPlaceholders(), this.updateRemaining()
        },
        processForm: function(e) {
            var t = this, n = e.find('[name="shoutout"]'), r = e.find('[name="name"]'), i = e.find('[name="age"]'), s = e.find('[name="state"]');
            this.clearPlaceholders && this.clearPlaceholders();
            var u = n.val(), a = r.val(), f = i.val(), l = s.val();
            this.addPlaceholders && this.addPlaceholders();
            if (u.length === 0)
                return this.showAlert("Type your shoutout in the box above"), !1;
            var c = n.attr("maxlength");
            if (u.length > c)
                return this.showAlert("Your message may be no more than " + c + " characters"), !1;
            if (a.length === 0 || a.indexOf(" ")!==-1)
                return this.showAlert("Enter your first name (no spaces)"), !1;
            if (f.length === 0 ||!f.match(/^\d+$/))
                return this.showAlert("Enter your age (numbers only)"), !1;
            f = parseInt(f, 10);
            if (f < 1 || f > 123)
                return this.showAlert("Enter a valid age"), !1;
            if (!l)
                return this.showAlert("Select the state you live in"), !1;
            var h = new o.Model({
                text: u,
                name: a,
                age: f,
                state: l,
                emblem: this.currentEmblem ? this.currentEmblem.get("id"): null
            }, {
                collection: this.model.data
            });
            this.updateSubmitButton({
                saving: !0
            }), this._pendingMessage = h;
            var p = u + "|" + a, d = o.sync("read", t.model, {
                url: this.filter_url + "/validate",
                data: {
                    message: p
                }
            });
            return d.then(function(e) {
                return e && e.reject===!1 ? h.save().then(t.onMessageSaveSuccess, t.onMessageSaveError) : t.onMessageSaveError("Please submit another message.")
            }, this.onMessageSaveError)
        },
        showAlert: function(e) {
            this.$(".notification").removeClass("notice").addClass("alert").stop(!0).hide().text(e).slideDown(500).delay(2500).slideUp(300)
        },
        showNotice: function(e) {
            this.$(".notification").removeClass("alert").addClass("notice").stop(!0).hide().text(e).slideDown(500).delay(2500).slideUp(300)
        },
        updateSubmitButton: function(e) {
            e = e || {};
            var t = this.$("button[type=submit]");
            e.saving ? t.attr("disabled", !0).addClass("disabled").text("Saving...") : e.disabled ? t.hide() : t.attr("disabled", !1).removeClass("disabled").text("Submit")
        }
    }), d = t.ModuleView.extend({
        initialize: function() {
            this.shoutout_views = [], this.index = this.model.collection.indexOf(this.model), this.model.data.each(function(e, t) {
                this.shoutout_views.push(new p({
                    model: e,
                    index: t,
                    moduleView: this,
                    template: this.entity,
                    message_template: this.message,
                    animateMessageIntoView: this.animateMessageIntoView
                }))
            }, this)
        },
        render: function() {
            var e = {};
            this.$el.html(this.template.render(e)), a.each(this.shoutout_views, function(e) {
                e.render(), this.$el.append(e.$el)
            }, this), n.Style.moduleStyle(this)
        }
    });
    d.register("shoutout", {
        className: "module shoutout",
        template: s.modules.shoutout,
        entity: s.entities.shoutout,
        message: s.entities.shoutout.message,
        animateMessageIntoView: function(e) {
            e.addClass("show")
        }
    }), d.register("comments", {
        className: "module comments",
        template: s.modules.comments,
        entity: s.entities.comments,
        message: s.entities.comments.message
        ,
        animateMessageIntoView: function(e) {
            e.addClass("show")
        }
    })
}(this), Whiskers("modules.side_by_side", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b('"> '), r.s(r.f("data", e, t, 1), e, t, 0, 53, 111, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="item"> '), n.b(n.rp("module_header", e, t, "")), n.b(" "), n.b(n.rp("entity", e, t, "")), n.b(" </div> ")
    }), e.pop()), r.b(" "), r.s(r.f("foregroundImage", e, t, 1), e, t, 0, 141, 197, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <img class="item '), n.b(n.v(n.f("fgPos", e, t, 0))), n.b('" src="'), n.b(n.v(n.f("foregroundImage", e, t, 0))), n.b('"> ')
    }), e.pop()), r.b(" </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e.Backbone, u = e._, a = e.devicePixelRatio > 1.5, f = t.ModuleView.extend({
        breakpoints: i,
        className: "module half-view",
        template: s.modules.side_by_side,
        render: function() {
            var e = o.Resize.width() >= r.midHigh, t = this.model, i = this.$el, f = t.get("style").layout ? t.get("style").layout: "left", l = t.get("style").images.foreground_image, c = null;
            l && (c = a ? l.retina_url ? l.retina_url : l.url ? l.url : null : l.url ? l.url : null);
            var h = u.extend({
                layout: f,
                foregroundImage: e ? c ? c: null: null,
                fgPos: e ? l ? l.alignment ? l.alignment: null: null: null,
                contentType: function() {
                    return this.get("type").toLowerCase()
                },
                isMobile: function() {
                    return this.get("game_type") === "Mobile App"
                }
            }, this.options.ads.helpers(), this.model);
            i.html(this.template.render(h, {
                entity: s.entities[this.model.get("type")],
                module_header: s.partials.module_elements.module_header
            })), i.addClass(f), n.Style.moduleStyle(this, !this.noBorder)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    });
    f.register("side_by_side", {}), f.register("halfling", {
        noBorder: !0
    }), f.register("age_gate", {})
}(this), Whiskers("modules.sidekick_shared", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b(" "), r.s(r.f("two_up", e, t, 1), e, t, 0, 53, 114, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.s(n.f("item_titles_below", e, t, 1), e, t, 0, 75, 92, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("item-titles-below")
        }), e.pop())
    }), e.pop()), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 144, 162, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="module_body"> <div class="col-container'), r.s(r.f("one_up", e, t, 1), e, t, 0, 241, 248, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" cols-1")
    }), e.pop()), r.s(r.f("two_up", e, t, 1), e, t, 0, 270, 277, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" cols-2")
    }), e.pop()), r.s(r.f("gameType", e, t, 1), e, t, 0, 301, 314, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.b(n.v(n.f("gameType", e, t, 0)))
    }), e.pop()), r.b('"> '), r.s(r.f("data", e, t, 1), e, t, 0, 339, 916, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="col"> <div class="thumbnail">'), n.b(n.rp("entity", e, t, "")), n.b('</div> <div class="details"> <h2 class="title" aria-hidden="false"><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></h2> "), n.s(n.f("short_desc", e, t, 1), e, t, 0, 512, 546, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<p class="desc">'), n.b(n.v(n.f("short_desc", e, t, 0))), n.b("</p>")
        }), e.pop()), n.b(" "), n.s(n.f("item_titles_below", e, t, 1), e, t, 0, 584, 629, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("ptitle", e, t, 1), e, t, 0, 596, 617, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" <h4>"), n.b(n.v(n.f("ptitle", e, t, 0))), n.b("</h4> ")
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(" "), n.s(n.f("one_up", e, t, 1), e, t, 0, 663, 890, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('" class="button large blue" data-linkname="moreinfo"> '), n.s(n.f("cta_button_title", e, t, 1), e, t, 1, 0, 0, "") || n.b(n.v(n.d("translations.more_info", e, t, 0))), n.b(" "), n.s(n.f("cta_button_title", e, t, 1), e, t, 0, 843, 863, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(n.v(n.f("cta_button_title", e, t, 0)))
            }), e.pop()), n.b(" </a> ")
        }), e.pop()), n.b(" </div> </div> ")
    }), e.pop()), r.b(' </div> <div class="ad-container"> '), r.b(r.t(r.f("rectangleAd", e, t, 0))), r.b(" "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" </div> </div> </div>"), r.fl()
}), Whiskers("modules.sidekick_featured", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 60, 78, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="module_body"> <div class="col-container"> '), r.s(r.f("data", e, t, 1), e, t, 0, 158, 345, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="thumb-container"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <span class="thumb-box"><span class="aspect"> <img src="'), n.b(n.v(n.f("sidekick_desktop", e, t, 0))), n.b('" class="thumb" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </span></span> </a> </div> ')
    }), e.pop()), r.b(' </div> <div class="ad-container"> '), r.b(r.t(r.f("rectangleAd", e, t, 0))), r.b(" "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" </div> </div> </div>"), r.fl()
}), Whiskers("modules.sidekick_video", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 60, 78, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="module_body"> <div class="col-container"> '), r.s(r.f("data", e, t, 1), e, t, 0, 158, 797, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="col"> <div class="video-container trailer-box"> <div class="aspect"> '), n.s(n.f("expired", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div id="player_sidekick" class="player video trailer-box"> <noscript> <div class="player_error">'), n.b(n.v(n.d("translations.no_javascript", e, t, 0))), n.b("</div> </noscript> </div> ")), n.b(" "), n.s(n.f("expired", e, t, 1), e, t, 0, 431, 494, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<div class="orphan"><p>'), n.b(n.v(n.d("translations.video_expired", e, t, 0))), n.b("</p></div>")
        }), e.pop()), n.b(' <div id="card" class="card embed"> <div class="card_header"> <p class="btn_replay">'), n.b(n.v(n.d("translations.replay", e, t, 0))), n.b('</p> <p class="btn_watch">'), n.b(n.v(n.d("translations.watch_on", e, t, 0))), n.b('</p> </div> <div class="suggested"> <ul> '), n.s(n.f("suggested", e, t, 1), e, t, 0, 719, 741, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <li>"), n.b(n.rp("entity", e, t, "")), n.b("</li> ")
        }), e.pop()), n.b(" </ul> </div> </div> </div> </div> </div> ")
    }), e.pop()), r.b(' </div> <div class="ad-container"> '), r.b(r.t(r.f("rectangleAd", e, t, 0))), r.b(" "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" </div> </div> </div>"), r.fl()
}), Whiskers("modules.watch", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 1773, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div itemprop="video" itemscope itemtype="http://schema.org/VideoObject" > <div id="video_wrapper" class="inverted"> <div class="bound"> <div class="scaler"></div> <div id="frame"> <!-- Orphan video markup --> <meta itemprop="duration" content="'), n.b(n.v(n.f("duration_iso", e, t, 0))), n.b('"> <meta itemprop="thumbnail" content="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('"> <meta itemprop="name" content="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <meta itemprop="embedId" content="'), n.b(n.v(n.f("id", e, t, 0))), n.b('"> <meta itemprop="title" content="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <meta itemprop="description" content="'), n.b(n.v(n.f("description", e, t, 0))), n.b('"> <meta itemprop="URL" content="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> '), n.s(n.f("embedURL", e, t, 1), e, t, 0, 562, 613, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <meta itemprop="embedURL" content="'), n.b(n.v(n.f("embedURL", e, t, 0))), n.b('"> ')
        }), e.pop()), n.b(" "), n.s(n.f("expired", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div id="player" class="player"> <noscript> <div class="player_error">'), n.b(n.v(n.d("translations.no_javascript", e, t, 0))), n.b("</div> </noscript> </div> ")), n.b(" "), n.s(n.f("expired", e, t, 1), e, t, 0, 791, 856, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="orphan"><p>'), n.b(n.v(n.d("translations.video_expired", e, t, 0))), n.b("</p></div> ")
        }), e.pop()), n.b(' <!-- /Orphan video markup --> <div id="card" class="card watch skip-styles"> <div class="card_header"> <p class="btn_replay">'), n.b(n.v(n.d("translations.replay_video", e, t, 0))), n.b('</p> <p class="countdown_intro">'), n.b(n.v(n.d("translations.next_video_start", e, t, 0))), n.b(' <span class="countdown">10</span><span class="autoplay">'), n.b(n.v(n.d("translations.autoplay_off", e, t, 0))), n.b("</span></p> "), n.s(n.f("upNext", e, t, 1), e, t, 0, 1197, 1277, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"><p class="btn_next">'), n.b(n.v(n.d("translations.watch_next_video", e, t, 0))), n.b("</p></a>")
        }), e.pop()), n.b(' </div> <div class="suggested"> <ul> '), n.s(n.f("upNext", e, t, 1), e, t, 0, 1336, 1375, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<li class="priority"> '), n.b(n.rp("entity", e, t, "")), n.b(" </li>")
        }), e.pop()), n.b(" "), n.s(n.f("suggested", e, t, 1), e, t, 0, 1401, 1423, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("<li> "), n.b(n.rp("entity", e, t, "")), n.b(" </li>")
        }), e.pop()), n.b(' </ul> </div> </div> </div> </div> </div> <!-- End of video_wrapper --> <!-- PLAYLIST --> <div id="playlist-holder" class="open"></div> <!-- END PLAYLIST --> <div class="content-toolbar-container"> <div id="toolbar" class="inverted"> <div id="meta"></div> </div> <!-- End of toolbar --> </div> <!-- End of bound --> </div> <!-- End --> ')
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Modernizr, o = e.Whiskers, u = e._, a = t.ModuleView.extend({
        className: "module sidekick",
        template: o.modules.sidekick_shared,
        numObjects: 1,
        render: function() {
            var e = this.model.get("type"), t = this.model.get("view"), r = this.model.get("data"), i;
            e === "game" && (r = this.getGameData(this.model)), r[0].game_type && (i = r[0].game_type.split(" ").join("_").toLowerCase()), this.model.data.models = u.first(r, this.numObjects);
            var s = u.extend({
                isPromo: this.model.get("type") === "promo",
                gameType: i,
                one_up: t === "sidekick_one_up",
                two_up: t === "sidekick_two_up",
                suggested: this.model.get("suggested")
            }, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(s, {
                module_header: o.partials.module_elements.module_header,
                iconHeader: o.partials.icon_header,
                entity: o.entities[this.model.get("type")]
            })).toggleClass("cds", !!this.options.cds&&!this.options.ads.showCDSAds()), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.options.ads.restore()
        },
        getGameData: function(e) {
            var t = e.get("data");
            u.each(t, function(e) {
                e.thumb = e.consoleThumb || e.thumb
            });
            if (!s.flash && s.canvas) {
                var n = [], r = [];
                u.each(t, function(e) {
                    e.compatibility_flags && (u.contains(e.compatibility_flags, "html5") || u.contains(e.compatibility_flags, "mobile")) ? n.push(e) : r.push(e)
                }), t = n.concat(r)
            }
            return t
        }
    });
    a.register("sidekick_two_up", {
        numObjects: 2
    }), a.register("sidekick_one_up", {}), a.register("sidekick_featured", {
        template: o.modules.sidekick_featured
    }), a.register("sidekick_video", {
        assets: {
            scripts: ["//a.dilcdn.com/a/videoplayer-e4e73a5b3283.js"]
        },
        template: o.modules.sidekick_video,
        breakpoints: i,
        ready: function() {
            var e = u.first(this.model.get("data")), t = this.model.get("autoplay")?!0 : !1;
            this.player = new n.MatterhornVideoPlayer("player_sidekick", {
                video: e,
                autoPlay: t,
                clickToPlay: !0,
                showTitle: !0,
                ads: {
                    enabled: !this.options.cds&&!this.options.noAds,
                    context: this.options.ads
                },
                translations: this.model.get("translations")
            })
        },
        remove: function() {
            this.player && (this.player.destroy(), this.player = null), t.BunView.prototype.remove.call(this)
        }
    })
}(this), Whiskers("modules.slider", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.b(" "), r.s(r.f("item_titles_below", e, t, 1), e, t, 0, 64, 81, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("item-titles-below")
    }), e.pop()), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 122, 140, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(" "), r.b(' <div class="slides peek-'), r.b(r.v(r.f("rpp", e, t, 0))), r.b(" "), r.s(r.f("peeking", e, t, 1), e, t, 1, 0, 0, "") || r.b("no-peek"), r.b('"> <ol class="slider-list peek-'), r.b(r.v(r.f("rpp", e, t, 0))), r.b('"> '), r.s(r.f("pages", e, t, 1), e, t, 0, 290, 470, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li class="slider-page"> <ul class="col-container cols-'), n.b(n.v(n.f("rpp", e, t, 0))), n.b('"> '), n.s(n.f("models", e, t, 1), e, t, 0, 367, 446, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <li class="col '), n.s(n.f("widthClass", e, t, 1), e, t, 0, 398, 412, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(n.v(n.f("widthClass", e, t, 0)))
            }), e.pop()), n.b('">'), n.b(n.rp("entity", e, t, "")), n.b("</li> ")
        }), e.pop()), n.b(" </ul> </li> ")
    }), e.pop()), r.b(' </ol> <div class="peek"></div> '), r.b(" </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Backbone, i = e.Whiskers, s = e.jQuery, o = e._, u = n.Style.breakpoints, a = o.values(u), f = r.Resize.width, l = t.ModuleView.extend({
        className: "module slider-view pagination-styles",
        breakpoints: a.join(" "),
        template: i.modules.slider,
        events: {
            slide: function(t, n) {
                t.preventDefault();
                var r = this, i = this.visibleThumbs.length;
                this.loadedPages = e.Math.max(this.loadedPages, n + this.startingCount), this.$(".slides").toggleClass("peeking", n !== this.length - 1), this.visibleThumbs = this.allThumbs.slice(0, this.rpp() * this.loadedPages + 1), !this.init && i < this.visibleThumbs.length && (this.collection.add(this.visibleThumbs), this.lastPage = n, e.setTimeout(function() {
                    r.render(), r.ready()
                }, this.slideTime))
            }
        },
        initialize: function() {
            var e = this, t = e.model, n = t.data;
            e.entityType = o.first(n.models).get("entityType") || t.get("type"), this.lastPage = 0, this.startingCount = this.loadedPages = 1, this.slideTime = 500, this.init=!1, this.allThumbs = t.get("data"), this.visibleThumbs = this.allThumbs.slice(0, this.rpp() * this.loadedPages + 1), this.collection = new r.LazyCollection(this.visibleThumbs), this.configureData && this.configureData()
        },
        peeking: function() {
            return this.model.get("data").length > this.rpp()
        },
        wide: function() {
            return f() >= u.midHigh
        },
        handleLoaded: function() {
            var e = this, t = this.$(".col img:first-of-type"), n = this.placeholders ? this.placeholders.length: 0;
            t.slice(this.rpp() * (this.loadedPages - 1) + 1 + n).each(function() {
                var t = s(this);
                t.hide(), this.complete ? t.fadeIn() : e.$(this).load(function() {
                    t.fadeIn()
                })
            })
        },
        render: function() {
            var e = this, t = e.model, n = e.rpp(), r = e.peeking(), s = this.collection, u = t.get("type") === "custom-character";
            this.formatFeatured && this.formatFeatured();
            if (s) {
                var a = o.extend({
                    rpp: n,
                    peeking: r,
                    entityType: e.entityType,
                    seeAll: t.get("seeAll"),
                    unstyled: u,
                    pages: s.pages(n, {
                        nest: 1
                    })
                }, e.options.ads.helpers(), this.model);
                e.length = a.pages.length, e.$el.html(e.template.render(a, {
                    module_header: i.partials.module_elements.module_header,
                    iconHeader: i.partials.icon_header,
                    entity: i.entities[e.entityType]
                })), this.handleLoaded()
            }
        },
        ready: function() {
            this.init=!0, this.$(".slides ol").swipeasaurus({
                arrows: !0,
                loop: !1,
                right: .25 / Math.sqrt(this.rpp() / .3),
                index: this.lastPage,
                speed: this.slideTime
            }), this.init=!1, n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render(), this.ready()
        }
    });
    l.register("slider", {
        rpp: function() {
            var e = f(), t = this.wide(), n = this.entityType;
            if (this.model.get("style").items_per_row_override && t)
                return this.model.get("style").items_per_row_override;
            switch (n) {
            case"movie":
            case"album":
                return t ? 4 : 3;
            case"personality":
            case"show":
                return e >= u.max ? 5 : t ? 4 : e >= u.midLow ? 3 : 2;
            case"product":
            case"promo":
            case"mobile_app":
                return e >= u.max ? 4 : t ? 3 : 2;
            case"video":
                return t ? 3 : 2;
            case"game":
                return t ? 3 : 2;
            default:
                return t ? 3 : 1
            }
        }
    }), l.register("characters", {
        rpp: function() {
            var e = f(), t = this.wide();
            return e >= u.max ? 5 : t ? 4 : e >= u.midLow ? 3 : 2
        }
    }), l.register("slider_rfy", {
        rpp: function() {
            var e = this.wide();
            return this.model.get("style").items_per_row_override && e ? this.model.get("style").items_per_row_override : e ? 3 : 2
        },
        initialize: function() {
            var e = this;
            e.entityType = "video", e.$el.hide(), this.lastPage = 0, this.startingCount = this.loadedPages = 1, this.slideTime = 500, this.init=!1, e.allThumbs = this.model.get("data"), e.fetchData()
        },
        fetchData: function() {
            var t = this, i = n.Cookie, o = /visitor_id=([^&]*)/.exec(e.location.search), u = o && o[1], a = null;
            if (!u) {
                var f = i.get("ctoLocalVisitor");
                if (f) {
                    var l = JSON.parse(decodeURIComponent(f)), c = l.localVisitorId;
                    c && (a = c)
                }
            } else 
                a = u;
            if (a) {
                var h = "/_recommendations/" + a;
                s.getJSON(h, function(e) {
                    if (!e.error) {
                        t.allThumbs = e;
                        for (var n = 0; n < t.allThumbs.length; n++)
                            t.allThumbs[n].id = t.allThumbs[n].id || n;
                        t.visibleThumbs = t.allThumbs.slice(0, t.rpp() * t.loadedPages + 1), t.collection = new r.LazyCollection(t.visibleThumbs), t.render(), t.ready(), t.$el.show()
                    }
                })
            }
        },
        peeking: function() {
            return !0
        }
    }), l.register("featured_promo_slider", {
        numFeatured: 1,
        configureData: function() {
            var t = this.collection.models, n = this.numFeatured, i = this.placeholders = [], o = e.state;
            if (e.state) {
                s.each(t, function(e, t) {
                    var n = t.get("location"), r = n && n.toLowerCase();
                    t.set("inState", r === o)
                });
                var u = this.collection.where({
                    inState: !0
                });
                this.collection.remove(u), this.collection.add(u, {
                    at: 0
                })
            }
            s.each(t, function(e, t) {
                if (e < n) {
                    var s = new r.Model;
                    s.set("widthClass", "placeholder"), i.push(s), t.set("widthClass", "featured"), t.set("featured", !0)
                }
            })
        },
        formatFeatured: function() {
            var e = this.collection.models, t = this.placeholders, n = this.wide();
            s.each(e, function(e, t) {
                t.get("featured") && n ? t.set("imgSource", t.get("promo_featured")) : t.set("imgSource", t.get("promo"))
            }), n ? this.collection.add(t, {
                at: 1
            }) : this.collection.remove(t)
        },
        rpp: function() {
            return this.wide() ? 4 : 2
        }
    })
}(this), Whiskers("modules.songs.songs_desktop", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 36, 54, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class = "container"> <div class = "label-container"> <p>Request</p> <div class="right"> <p>Video</p> <p>Purchase</p> </div> </div> <div class = "track-container"> <ul> '), r.s(r.f("data", e, t, 1), e, t, 0, 253, 1093, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li class="song"> <a class = "like-icon clicon" data-likeinfo="request-'), n.b(n.v(n.f("title", e, t, 0))), n.b("-"), n.b(n.v(n.f("artist", e, t, 0))), n.b('"></a> <span class = "number"></span> <div class = "track-title"> <div class="song-title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</div> <div class = "artist">'), n.b(n.v(n.f("artist", e, t, 0))), n.b('</div> </div> <div class ="icon-container" data-buyaylitics = "'), n.b(n.v(n.f("title", e, t, 0))), n.b("-"), n.b(n.v(n.f("artist", e, t, 0))), n.b('"> <a class = "music-video '), n.s(n.f("watch_link", e, t, 1), e, t, 0, 608, 627, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("active hover clicon")
        }), e.pop()), n.b('" '), n.s(n.f("watch_link", e, t, 1), e, t, 0, 659, 682, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' href="'), n.b(n.v(n.f("watch_link", e, t, 0))), n.b('" ')
        }), e.pop()), n.b('></a> <div class = "cart '), n.s(n.f("retail", e, t, 1), e, t, 0, 733, 752, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("active hover clicon")
        }), e.pop()), n.b('" '), n.s(n.f("retail", e, t, 1), e, t, 0, 776, 799, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('data-convert="buy_song"')
        }), e.pop()), n.b('> <div class="buy '), n.s(n.f("retail", e, t, 1), e, t, 1, 0, 0, "") || n.b("no_flyout"), n.b('"> <ul> '), n.s(n.f("retail", e, t, 1), e, t, 0, 878, 1048, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("retail_links", e, t, 1), e, t, 0, 896, 1030, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <li class = "fly-li"><a href="'), n.b(n.v(n.f("content_url", e, t, 0))), n.b('" target="_blank" data-purchase="external" data-retailer="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></li> ")
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(" </ul> </div> </div> </div> </li> ")
    }), e.pop()), r.b(" </ul> </div> </div> </div>"), r.fl()
}), Whiskers("modules.songs.songs_mobile", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 36, 54, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="container"> <div class="track-container"> <ul> '), r.s(r.f("data", e, t, 1), e, t, 0, 139, 1065, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li class="song"> <span class = "number"></span> <div class = "track-title"> <div class="song-title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</div> <div class = "artist">'), n.b(n.v(n.f("artist", e, t, 0))), n.b('</div> </div> <div class = "expand-icon clicon" data-expand="'), n.b(n.v(n.f("song", e, t, 0))), n.b('"></div> <div class = "icon-container" data-buyaylitics = "'), n.b(n.v(n.f("title", e, t, 0))), n.b("-"), n.b(n.v(n.f("artist", e, t, 0))), n.b('"> <div class="inner"> <a class = "like-icon clicon" data-like="'), n.b(n.v(n.f("song", e, t, 0))), n.b('" data-likeinfo="request-'), n.b(n.v(n.f("title", e, t, 0))), n.b("-"), n.b(n.v(n.f("artist", e, t, 0))), n.b('"></a> '), n.s(n.f("watch_link", e, t, 1), e, t, 0, 576, 679, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a class = "music-video active hover clicon" href = "'), n.b(n.v(n.f("watch_link", e, t, 0))), n.b('" data-linkname="music-video"></a> ')
        }), e.pop()), n.b(" "), n.s(n.f("retail", e, t, 1), e, t, 0, 706, 984, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class = "cart active hover clicon" data-convert="buy_song"> <div class="buy"> <ul> '), n.s(n.f("retail_links", e, t, 1), e, t, 0, 812, 946, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <li class = "fly-li"><a href="'), n.b(n.v(n.f("content_url", e, t, 0))), n.b('" target="_blank" data-purchase="external" data-retailer="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a></li> ")
            }), e.pop()), n.b(" </ul> </div> </div> ")
        }), e.pop()), n.b(' <div class="close" data-close ="'), n.b(n.v(n.f("song", e, t, 0))), n.b('"></div> </div> </div> </li> ')
    }), e.pop()), r.b(" </ul> </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    function h(t) {
        s.mq("only all") && (t.template = u(e.window).width() >= r.midHigh ? f.songs_desktop : f.songs_mobile)
    }
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Modernizr, o = e.Whiskers, u = e.jQuery, a = e._, f = e.Whiskers.modules.songs, l = 0, c = t.ModuleView.extend({
        breakpoints: i,
        template: f.songs_desktop,
        initialize: function() {
            l = u(e.window).width(), h(this)
        },
        render: function() {
            var e = 0, t = a.extend({}, this.model);
            this.$el.html(this.template.render(t, {
                module_header: o.partials.module_elements.module_header,
                iconHeader: o.partials.icon_header
            }));
            var r = this.$("span.number"), i = this.$("li.song");
            r.each(function(t, n) {
                e++;
                var r = u(n);
                r.append(e)
            }), i.each(function(t, n) {
                e++;
                var r = e%2, i = u(n);
                r === 1 ? i.addClass("odd") : i.addClass("even")
            }), n.Style.moduleStyle(this)
        },
        events: {
            "click .like-icon": function(e) {
                u(e.currentTarget).addClass("clicked")
            },
            "click .clicked": function(e) {
                u(e.currentTarget).removeClass("clicked")
            },
            "click .cart": function(e) {
                var t = u(e.target), n = t.children(".buy");
                t.context.className.match(/cart active hover/) && t.addClass("clicon"), n.hasClass("active") ? n.removeClass("active") : (this.$(".buy").removeClass("active"), n.addClass("active"))
            },
            "click .fly-li": function(e) {
                u(e.currentTarget).closest("div.cart").removeClass("clicon")
            },
            "click .expand-icon": function(e) {
                var t = u(e.currentTarget.nextElementSibling);
                this.$(".buy").removeClass("active"), this.$(".icon-container.open").removeClass("open"), t.toggleClass("open")
            },
            "click .close": function() {
                this.$(".buy").removeClass("active"), this.$(".icon-container.open").removeClass("open")
            }
        },
        resize: function() {
            h(this), this.render()
        }
    });
    c.register("songs", {})
}(this), Whiskers("modules.storybook", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("story", e, t, 1), e, t, 0, 10, 2229, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div id="story-container" class="reading '), n.s(n.f("noCover", e, t, 1), e, t, 1, 0, 0, "") || n.b(" start-page"), n.s(n.f("noAudio", e, t, 1), e, t, 0, 99, 108, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" no-audio")
        }), e.pop()), n.b('"> <div class="artwork carousel"> <ul id="storybook-artwork"> '), n.s(n.f("panels", e, t, 1), e, t, 0, 193, 561, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("image", e, t, 1), e, t, 0, 204, 550, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(" <li> "), n.s(n.f("bg", e, t, 1), e, t, 0, 217, 246, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b('<img src="'), n.b(n.v(n.f("bg", e, t, 0))), n.b('" class="bg">')
                }), e.pop()), n.b(" "), n.s(n.f("mid", e, t, 1), e, t, 0, 262, 293, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b('<img src="'), n.b(n.v(n.f("mid", e, t, 0))), n.b('" class="mid">')
                }), e.pop()), n.b(" "), n.s(n.f("fg", e, t, 1), e, t, 0, 309, 338, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b('<img src="'), n.b(n.v(n.f("fg", e, t, 0))), n.b('" class="fg">')
                }), e.pop()), n.b(" "), n.s(n.f("ambient_sound", e, t, 1), e, t, 0, 364, 525, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(' <audio id="ambient-loop" preload loop> '), n.s(n.f("mp3", e, t, 1), e, t, 0, 412, 451, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                        n.b('<source src="'), n.b(n.v(n.f("mp3", e, t, 0))), n.b('" type="audio/mp3">')
                    }), e.pop()), n.b(" "), n.s(n.f("ogg", e, t, 1), e, t, 0, 468, 507, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                        n.b('<source src="'), n.b(n.v(n.f("ogg", e, t, 0))), n.b('" type="audio/ogg">')
                    }), e.pop()), n.b(" </audio> ")
                }), e.pop()), n.b(" </li> ")
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(' </ul> <div class="fade"></div> </div> <div class="narrative carousel"> <ul id="storybook-narrative"> '), n.s(n.f("text", e, t, 1), e, t, 0, 683, 1611, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <li> "), n.s(n.f("startPage", e, t, 1), e, t, 1, 0, 0, "") || n.s(n.f("buyButton", e, t, 1), e, t, 1, 0, 0, "") || (n.b("<p>"), n.b(n.v(n.f("text", e, t, 0))), n.b("</p>")), n.b(" "), n.s(n.f("startPage", e, t, 1), e, t, 0, 775, 1078, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <div class="start-buttons"> <div id="read" class="large blue button start-button">'), n.b(n.v(n.d("translations.read", e, t, 0))), n.b('</div> <div id="listen" class="large green button start-button">'), n.b(n.v(n.d("translations.listen", e, t, 0))), n.b('</div> <div id="read-without-audio" class="large green button start-button">'), n.b(n.v(n.d("translations.begin", e, t, 0))), n.b("</div> </div> ")
            }), e.pop()), n.b(" "), n.s(n.f("buyButton", e, t, 1), e, t, 0, 1107, 1590, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <div class="end '), n.s(n.f("identity", e, t, 1), e, t, 1, 0, 0, "") || n.b("no-association"), n.b('"> <div class="end-block left"> <span class="replay small silver button">'), n.b(n.v(n.d("translations.read_again", e, t, 0))), n.b('</span> <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('"><span class="purchase small green button">'), n.b(n.v(n.f("text", e, t, 0))), n.b("</span></a> </div> "), n.s(n.f("identity", e, t, 1), e, t, 0, 1372, 1569, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b(' <div class="end-block right"> <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('"> <span class="visit"> '), n.s(n.f("image", e, t, 1), e, t, 0, 1453, 1492, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                        n.b('<img class="portrait" src="'), n.b(n.v(n.f("retina", e, t, 0))), n.b('">')
                    }), e.pop()), n.b(" <span>"), n.b(n.v(n.d("translations.visit", e, t, 0))), n.b(" "), n.b(n.v(n.f("title", e, t, 0))), n.b("</span> </span> </a> </div> ")
                }), e.pop()), n.b(" </div> ")
            }), e.pop()), n.b(" </li> ")
        }), e.pop()), n.b(" </ul> </div> "), n.s(n.f("audio", e, t, 1), e, t, 0, 1644, 2002, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" "), n.s(n.f("narration", e, t, 1), e, t, 0, 1659, 1809, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b('<audio id="narrative-sprite"> '), n.s(n.f("mp3", e, t, 1), e, t, 0, 1697, 1736, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b('<source src="'), n.b(n.v(n.f("mp3", e, t, 0))), n.b('" type="audio/mp3">')
                }), e.pop()), n.b(" "), n.s(n.f("ogg", e, t, 1), e, t, 0, 1753, 1792, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b('<source src="'), n.b(n.v(n.f("ogg", e, t, 0))), n.b('" type="audio/ogg">')
                }), e.pop()), n.b(" </audio>")
            }), e.pop()), n.b(" "), n.s(n.f("music", e, t, 1), e, t, 0, 1834, 1991, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b('<audio id="music-loop" preload loop> '), n.s(n.f("mp3", e, t, 1), e, t, 0, 1879, 1918, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b('<source src="'), n.b(n.v(n.f("mp3", e, t, 0))), n.b('" type="audio/mp3">')
                }), e.pop()), n.b(" "), n.s(n.f("ogg", e, t, 1), e, t, 0, 1935, 1974, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                    n.b('<source src="'), n.b(n.v(n.f("ogg", e, t, 0))), n.b('" type="audio/ogg">')
                }), e.pop()), n.b(" </audio>")
            }), e.pop()), n.b(" ")
        }), e.pop()), n.b(' <div id="storybook-counter"> <div class="page-count"><span class="current-count"></span> '), n.b(n.v(n.d("translations.of", e, t, 0))), n.b(' <span class="total-count"></span></div> </div> '), n.s(n.f("noAudio", e, t, 1), e, t, 1, 0, 0, "") || n.b('<div id="mute-button"></div>'), n.b(" </div> ")
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Whiskers, r = e.Disney, i = e.Modernizr, s = e.Tracker, o = e.jQuery, u = e._, a = r.Style.breakpoints, f = u.values(a);
    t.ModuleView.register("storybook", {
        breakpoints: f.join(" "),
        template: n.modules.storybook,
        story: {},
        textData: [],
        book: undefined,
        audInProgress: !1,
        narrationAudio: undefined,
        musicAudio: undefined,
        ambientAudio: undefined,
        currTextIndex: 0,
        end: 0,
        muted: !0,
        keyEvents: !1,
        fadeInIntervals: {
            ambient: [],
            music: [],
            narration: []
        },
        fadeOutIntervals: {
            ambient: [],
            music: [],
            narration: []
        },
        narrativeTimeouts: [],
        pulseInterval: undefined,
        timeCheckInterval: undefined,
        startTimeout: undefined,
        replayTimeout: undefined,
        touchDevice: i.touch,
        blinkArrow: function(t) {
            var n = this;
            t ? (o(".next").addClass("turn-page"), n.pulseInterval = e.setInterval(function() {
                o(".next").toggleClass("pulse")
            }, 1e3)) : (e.clearInterval(n.pulseInterval), n.pulseInterval = null, o(".next").removeClass("turn-page pulse"))
        },
        stopAndReset: function(e, t) {
            var n = this;
            e && (t&&!n.touchDevice ? n.fadeOut(["narration"], 200) : (e.pause(), n.audInProgress=!1)), o(".next, .prev").removeClass("playing")
        },
        fadeIn: function(t, n) {
            var r = this;
            o.each(t, function(t, i) {
                var s = r[i + "Audio"];
                if (s) {
                    s.volume = 0, s.play();
                    var o = r.currTextIndex;
                    r.fadeInIntervals[i][o] = e.setInterval(function() {
                        s && s.volume < 1 - 20 / n ? s.volume += 20 / n : (s && (s.volume = 1), e.clearInterval(r.fadeInIntervals[i][o]), r.fadeInIntervals[i][o] = null)
                    }, 20)
                }
            })
        },
        fadeOut: function(t, n) {
            var r = this;
            o.each(t, function(t, i) {
                var s = r[i + "Audio"];
                if (s) {
                    var o = r.currTextIndex;
                    r.fadeOutIntervals[i][o] = e.setInterval(function() {
                        s && s.volume > 20 / n ? s.volume -= 20 / n : (s && (s.volume = 0, s.pause(), r.audInProgress=!1), e.clearInterval(r.fadeOutIntervals[i][o]), r.fadeOutIntervals[i][o] = null)
                    }, 20)
                }
            })
        },
        playNarrative: function(t) {
            var n = this, r = n.$("#storybook-narrative");
            n.textData[t].audio && (n.narrationAudio.currentTime = n.textData[t].audio.startTime, n.end = n.textData[t].audio.endTime, n.narrationAudio.volume = 1, n.narrationAudio.play(), o(".next, .prev").addClass("playing"), n.audInProgress=!0, n.timeCheckInterval = e.setInterval(function() {
                n.audInProgress && n.narrationAudio.currentTime > n.end && t + 1 < n.textData.length && (n.stopAndReset(n.narrationAudio, !0), e.clearInterval(n.timeCheckInterval), n.timeCheckInterval = null, n.currTextIndex === 0 ? (e.setTimeout(function() {
                    r.swipeasaurus("next")
                }, 1500), n.blinkArrow(!1)) : n.blinkArrow(!0))
            }, 10))
        },
        setBlankPage: function(e, t) {
            var n = this;
            e ? (o("#story-container").addClass("start-page reading"), o(".start-buttons").fadeIn("fast"), n.keyEvents=!1, o("#storybook-counter", n.$el).css("opacity", "0"), t || (o("#story-container").removeClass("started listening"), o("#mute-button").removeClass("muted").attr("style", ""), n.musicAudio && n.currTextIndex === 1 && n.fadeOut(["music"], 1e3), n.narrationAudio && n.currTextIndex === 1 && n.fadeOut(["narration"], 500))) : (o("#story-container").removeClass("start-page").addClass("started"), n.keyEvents=!0, o("#storybook-counter", n.$el).css("opacity", "1"))
        },
        muteButtonPress: function(t) {
            var n = this;
            t ? (n.audInProgress=!1, n.stopAndReset(n.narrationAudio, !0), n.musicAudio && n.fadeOut(["music", "ambient"], 800), o("#mute-button").addClass("muted"), n.timeCheckInterval && e.clearInterval(n.timeCheckInterval), n.muted=!0) : (n.playNarrative(n.currTextIndex), n.musicAudio && n.fadeIn(["music", "ambient"], 400), o("#mute-button").removeClass("muted"), n.muted=!1)
        },
        render: function() {
            var e = this, t = e.model, n = u.first(t.get("data")), r = e.story, i = e.textData, s = this.model.get("translations"), a = {
                read: "Read",
                listen: "Listen",
                begin: "Let's Begin!",
                visit: "Visit",
                read_again: "Read Again"
            };
            u.defaults(s, a), r = n, e.book = r.title, i.length = 0, r.location = r.endcard.image.bg.split("/")[2] || "http://cdnvideo.dolimg.com", o.each(r.panels, function(t, n) {
                n.text_blocks && o.each(n.text_blocks, function(e, n) {
                    i.push({
                        page: t + 1,
                        text: n.text,
                        audio: n.audio
                    })
                }), n.image && (o.each(["bg", "mid", "fg"], function(e, t) {
                    n.image[t] && (n.image[t] = "http://" + r.location + n.image[t])
                }), o.each(["mp3", "ogg"], function(e, t) {
                    n.image.ambient_sound && n.image.ambient_sound[t] && (n.image.ambient_sound[t] = "http://" + r.location + n.image.ambient_sound[t])
                }), e.touchDevice && (n.image.ambient_sound = null))
            }), r.cover && (r.panels.splice(0, 0, r.cover), i.splice(0, 0, {
                text: " ",
                page: 0,
                audio: r.cover.text_blocks[0].audio,
                startPage: !0
            })), r.endcard && (r.panels.push(r.endcard), i.push({
                text: null,
                page: r.panels.length,
                audio: r.endcard.text_blocks[0].audio,
                buyButton: r.endcard.buyButton
            })), o.each(i, function(e, t) {
                t.audio && o.each(["startTime", "endTime"], function(e, n) {
                    t.audio[n] && t.audio[n].split(":").length > 1 ? t.audio[n] = (parseFloat(t.audio[n].split(":")[0], 10) * 60 + parseFloat(t.audio[n].split(":")[1], 10)).toFixed(1) : t.audio[n] && (t.audio[n] = parseFloat(t.audio[n]))
                })
            }), r.text = i, e.touchDevice && r.audio && (r.audio.music = null), e.story = r;
            var f = u.extend(e.options.ads.helpers(), e.model, {
                story: r,
                noCover: r.cover?!1: !0,
                noAudio: r.audio === null,
                translations
                : s
            });
            e.$el.html(this.template.render(f), {}), e.$("#storybook-artwork").swipeasaurus(), e.$("#storybook-narrative").swipeasaurus({
                arrows: !0,
                loop: !1
            })
        },
        ready: function() {
            var t = this;
            t.narrationAudio = o("#narrative-sprite")[0], t.musicAudio = o("#music-loop")[0];
            var n = t.story, i = t.$("#storybook-artwork"), u = t.$("#storybook-narrative"), a = t.$("#story-container"), f = t.$("#storybook-counter .current-count"), l = t.$("#storybook-counter .total-count"), c = n.cover ? i.children("li").length - 1: i.children("li").length, h = t.textData;
            o("html").hasClass("no-media-query") && a.addClass("no-audio"), n.cover ? o("#storybook-counter").css("opacity", 0) : t.keyEvents=!0, f.text("1"), l.text(c), o(t.narrationAudio).on("ended", function() {
                o(".end").css("opacity", 1), o("#mute-button").css("opacity", 0), t.replayTimeout = e.setTimeout(function() {
                    t.musicAudio && t.fadeOut(["music"], 2e3), t.ambientAudio && t.fadeIn(["ambient"], 2e3)
                }, 5e3), t.blinkArrow(!1), t.keyEvents=!1, t.stopAndReset(t.narrationAudio), t.audInProgress=!1
            }), i.on("slide", function(e, r) {
                n.cover ? r === 0 && t.currTextIndex === 1 || r === c ? t.setBlankPage(!0, r === c) : r === 1 && t.currTextIndex === 0 ? (t.setBlankPage(!1, !1), f.text(r)) : f.text(r) : r === c - 1 ? t.setBlankPage(!0, !0) : f.text(r);
                var u = o(i.find("li")[r]).find("audio");
                u.length > 0 ? (t.ambientAudio = u[0], t.muted || (t.ambientAudio.load(), o(t.ambientAudio).on("canplay", function() {
                    t.fadeIn(["ambient"], 2e3), o(t.ambientAudio).off("canplay")
                }))) : t.ambientAudio = null, s.track({
                    breadcrumbs: t.book,
                    pageName: "dcom|pri|" + t.book + "|" + (r + 1)
                })
            }), u.on("slide", function(r, s) {
                var o;
                s !== 0 && (e.clearTimeout(t.startTimeout), t.stopAndReset(t.narrationAudio, !0)), e.clearTimeout(t.narrativeTimeouts[t.currTextIndex]), t.narrativeTimeouts[t.currTextIndex] = null, t.timeCheckInterval && e.clearInterval(t.timeCheckInterval), t.blinkArrow(!1), h[t.currTextIndex].page !== h[s].page ? (t.ambientAudio && t.fadeOut(["ambient"], 500), i.swipeasaurus("slide", n.cover ? h[s].page : h[s].page - 1), t.ambientAudio ? o = 3e3 : o = 900) : o = 400, !t.muted && h[s].audio && a.hasClass("started")&&!a.hasClass("replay") && (t.narrativeTimeouts[s] = e.setTimeout(function() {
                    t.playNarrative(s)
                }, o)), t.currTextIndex = s
            }), o("#mute-button").on("click", function() {
                o(this).hasClass("muted") ? t.muteButtonPress(!1) : t.muteButtonPress(!0)
            }), (n.identity && n.identity.takeover || t.model.collection.style && t.model.collection.style.takeover.image) && r.Style.backgroundStyles(t.model.collection.style.takeover && t.model.collection.style.takeover.image ? {
                takeover: t.model.collection.style.takeover
            } : {
                takeover: n.identity.takeover
            }), o("#storybook-narrative .replay").click(function() {
                o(".end").attr("style", ""), t.musicAudio && t.fadeOut(["music"], 2e3), t.ambientAudio && t.fadeOut(["ambient"], 2e3), n.cover ? t.setBlankPage(!0, !1) : t.setBlankPage(!1, !1), a.addClass("replay"), u.swipeasaurus("slide", 0), s.link({
                    linkName: "storybook",
                    linkPosition: "storybook/button/replay"
                }), o("#mute-button").removeClass("muted")
            }), o(e.document).keydown(function(e) {
                if (t.keyEvents) {
                    if (e.keyCode === 37)
                        return u.swipeasaurus("prev", 0), !1;
                    if (e.keyCode === 39)
                        return u.swipeasaurus("next", 0), !1
                }
            }), t.resize(), o(".start-button").on("click", function() {
                t.currTextIndex = 0, t.startTimeout = e.setTimeout(function() {
                    o(".start-buttons").hide()
                }, 1500), e.clearInterval(t.fadeOutIntervals.music[h.length - 1]), e.clearTimeout(t.replayTimeout), o(this).attr("id") === "listen" ? (t.muted=!1, t.audInProgress=!1, a.removeClass("reading replay").addClass("listening started"), t.narrationAudio.readyState < 3 && t.narrationAudio.load(), t.musicAudio ? t.musicAudio.readyState >= 3 ? (t.musicAudio.currentTime = 0, t.fadeIn(["music"], 300), e.setTimeout(function() {
                    u.trigger("slide", 0)
                }, 1500)) : (t.musicAudio.load(), o(t.musicAudio).on("canplay", function() {
                    t.fadeIn(["music"], 300), o(t.musicAudio).off("canplay"), t.narrationAudio.readyState >= 3 ? e.setTimeout(function() {
                        u.trigger("slide", 0)
                    }, 1500) : o(t.narrationAudio).on("canplay", function() {
                        t.narrationAudio.volume = 1, u.trigger("slide", 0)
                    })
                })) : t.narrationAudio.readyState > 2 ? u.trigger("slide", 0) : o(t.narrationAudio).on("canplay", function() {
                    u.trigger("slide", 0)
                }), s.link({
                    linkName: "storybook",
                    linkPosition: "storybook/button/listen"
                })) : (t.muted=!0, t.musicAudio && t.fadeOut(["music", "ambient"], 400), a.addClass("reading started").removeClass("start-page replay listening"), u.swipeasaurus("slide", 1), t.keyEvents=!0, s.link({
                    linkName: "storybook",
                    linkPosition: "storybook/button/read"
                }))
            })
        },
        resize: function() {
            o("#storybook-narrative li p").css("top", function() {
                return (parseInt(o(this).parent().height(), 10) - parseInt(o(this).outerHeight(), 10)) / 2
            })
        }
    })
}(this), Whiskers("modules.stream", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound '), r.b(r.v(r.f("type", e, t, 0))), r.b(" "), r.b(r.v(r.f("entityType", e, t, 0))), r.s(r.f("singleRow", e, t, 1), e, t, 0, 55, 66, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" single-row")
    }), e.pop()), r.s(r.f("enablePeek", e, t, 1), e, t, 0, 95, 103, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" peeking")
    }), e.pop()), r.s(r.f("item_titles_below", e, t, 1), e, t, 0, 140, 158, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" item-titles-below")
    }), e.pop()), r.b('"> '), r.s(r.f("header_data", e, t, 1), e, t, 0, 199, 217, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(" "), r.b(r.t(r.f("list", e, t, 0))), r.b(' <div class="show_more_container"> <div class="bottom-peek"></div> <div class="show_more blue button large" role="button" onclick=""> <span class="label">'), r.b(r.v(r.d("translations.show_more", e, t, 0))), r.b('</span> <span class="spinner"></span> </div> </div> </div>'), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Backbone, i = e.Whiskers, s = e.Math, o = e._, u = t.Style.breakpoints, a = o.values(u);
    n.ModuleView.register("stream", {
        className: "module stream-view",
        breakpoints: a.join(" "),
        page: 1,
        template: i.modules.stream,
        events: {
            "click .show_more": function(t) {
                this.loadMoreThumbs(), this.has_revealed=!0, t.preventDefault(), this.fit(), e.Tracker.pagination(this.page)
            }
        },
        initialize: function() {
            var e = this, t = this.model, s = t.data;
            this.allThumbs = this.model.get("data"), this.count = this.allThumbs.length, this.entityType = o.first(s.models).get("entityType") || t.get("type"), this.visibleThumbs = this.allThumbs.slice(0, this.page * this.rows() * this.cols()), this.thumbCollection = new r.LazyCollection(this.visibleThumbs), this.has_revealed=!1, this.list = new n.ListView({
                collection: e.thumbCollection,
                template: {
                    render: function(t) {
                        return i.modules.col.render(t, {
                            entity: i.entities[e.entityType]
                        })
                    }
                }
            })
        },
        updateView: function() {
            this.visibleThumbs = this.allThumbs.slice(0, this.desired()), this.list.collection.add(this.visibleThumbs)
        },
        loadMoreThumbs: function() {
            this.page++, this.updateView(), t.Style.moduleStyle(this)
        },
        updateButton: function() {
            var e = this.model.get("enable_peek"), t = this.visibleThumbs.length < this.count;
            this.$(".col").hasClass("last-row") && e ||!e && t ? this.$el.addClass("has-more") : this.$el.removeClass("has-more")
        },
        desired: function() {
            return this.page * this.rows() * this.cols()
        },
        render: function() {
            var e = this.list, n = o.extend({
                entityType: this.entityType,
                list: t.placeHtml(e),
                enablePeek: this.model.get("enable_peek"),
                singleRow: this.count <= this.cols()
            }, this.model);
            this.$el.html(this.template.render(n, {
                module_header: i.partials.module_elements.module_header,
                iconHeader: i.partials.icon_header
            })), t.Style.moduleStyle(this), t.placeSwap(this, e)
        },
        resize: function() {
            var e = this.cols(), t = e > 1 ? "grid-layout": "list-layout";
            this.page = s.floor(this.visibleThumbs.length / this.rows() / e) || 1, this.list.$el.attr("class", "col-container cols-" + e + " " + t), this.fit()
        },
        fit: function() {
            var e = this.desired();
            this.$(".col").removeClass("last-row"), e < this.count ||!this.has_revealed ? (this.updateView(), this.list.$el.children().show().slice(e).hide(), this.list.$el.children().slice(e - this.cols()).addClass("last-row")) : this.list.$el.children().show(), this.updateButton()
        },
        rows: function() {
            var e = r.Resize.width(), t = this.model.get("type"), n = this.entityType;
            if (t === "game")
                return e < u.midHigh ? 4 : 4;
            switch (n) {
            case"personality":
                return e < u.midLow ? 3 : 2;
            default:
                return e < u.midHigh ? 3 : 2
            }
        },
        cols: function() {
            var e = r.Resize.width(), n = this.entityType;
            switch (n) {
            case"movie":
                return t.portal === "princess.disney.com" ? e >= u.max ? 4 : e >= u.midHigh ? 3 : 1 : e >= u.max ? 5 : e >= u.midHigh ? 4 : 1;
            case"video":
            case"gallery":
                return e >= u.max ? 3 : e >= u.midHigh ? 3 : e >= u.midLow ? 2 : 1;
            case"product":
                return e >= u.max ? 5 : e >= u.midHigh ? 4 : e >= u.midLow ? 2 : 1;
            case"personality":
                return e >= u.max ? 5 : e >= u.midHigh ? 4 : e >= u.midLow ? 3 : 2;
            default:
                return e >= u.max ? 4 : e >= u.midHigh ? 3 : 1
            }
        },
        ready: function() {
            t.Style.moduleStyle(this), this.resize()
        }
    })
}(this), function(e, t) {
    "use strict";
    e(function() {
        function e(e, t, n, i) {
            return r(e).then(t, n, i)
        }
        function n(e, t) {
            this.then = e, this.inspect = t
        }
        function r(e) {
            return o(function(t) {
                t(e)
            })
        }
        function i(t) {
            return e(t, f)
        }
        function s() {
            function i(i, s, o) {
                e.resolve = e.resolver.resolve = function(e) {
                    return n ? r(e) : (n=!0, i(e), t)
                }, e.reject = e.resolver.reject = function(e) {
                    return n ? r(f(e)) : (n=!0, s(e), t)
                }, e.notify = e.resolver.notify = function(e) {
                    return o(e), e
                }
            }
            var e, t, n;
            return e = {
                promise: D,
                resolve: D,
                reject: D,
                notify: D,
                resolver: {
                    resolve: D,
                    reject: D,
                    notify: D
                }
            }, e.promise = t = o(i), e
        }
        function o(e) {
            function s(e, n, i) {
                return o(function(s, o, u) {
                    r ? r.push(function(t) {
                        t.then(e, n, i).then(s, o, u)
                    }) : P(function() {
                        t.then(e, n, i).then(s, o, u)
                    })
                })
            }
            function a() {
                return t ? t.inspect() : x()
            }
            function h(e) {
                if (!r)
                    return;
                t = u(e), c(r, t), r = D
            }
            function p(e) {
                h(f(e))
            }
            function d(e) {
                r && c(r, l(e))
            }
            var t, r = [];
            try {
                e(h, p, d)
            } catch (i) {
                p(i)
            }
            return new n(s, a)
        }
        function u(e) {
            return e instanceof n ? e : e === Object(e) && "then"in e ? o(function(t, n, r) {
                P(function() {
                    try {
                        var i = e.then;
                        typeof i == "function" ? C(i, e, t, n, r) : t(a(e))
                    } catch (s) {
                        n(s)
                    }
                })
            }) : a(e)
        }
        function a(e) {
            var t = new n(function(n) {
                try {
                    return typeof n == "function" ? u(n(e)) : t
                } catch (r) {
                    return f(r)
                }
            }, function() {
                return E(e)
            });
            return t
        }
        function f(e) {
            var t = new n(function(n, r) {
                try {
                    return typeof r == "function" ? u(r(e)) : t
                } catch (i) {
                    return f(i)
                }
            }, function() {
                return S(e)
            });
            return t
        }
        function l(e) {
            var t = new n(function(n, r, i) {
                try {
                    return typeof i == "function" ? l(i(e)) : t
                } catch (s) {
                    return l(s)
                }
            });
            return t
        }
        function c(e, t) {
            P(function() {
                var n, r = 0;
                while (n = e[r++])
                    n(t)
            })
        }
        function h(e) {
            return e && typeof e.then == "function"
        }
        function p(t, n, r, i, s) {
            return e(t, function(t) {
                function u(r, i, s) {
                    function d(e) {
                        c(e)
                    }
                    function v(e) {
                        l(e)
                    }
                    var o, u, a, f, l, c, h, p;
                    h = t.length>>>0, o = Math.max(0, Math.min(n, h)), a = [], u = h - o + 1, f = [];
                    if (!o)
                        r(a);
                    else {
                        c = function(e) {
                            f.push(e), --u || (l = c = j, i(f))
                        }, l = function(e) {
                            a.push(e), --o || (l = c = j, r(a))
                        };
                        for (p = 0; p < h; ++p)
                            p in t && e(t[p], v, d, s)
                    }
                }
                return o(u).then(r, i, s)
            })
        }
        function d(e, t, n, r) {
            function i(e) {
                return t ? t(e[0]) : e[0]
            }
            return p(e, 1, i, n, r)
        }
        function v(e, t, n, r) {
            return b(e, j).then(t, n, r)
        }
        function m() {
            return b(arguments, j)
        }
        function g(e) {
            return b(e, E, S)
        }
        function y(e, t) {
            return b(e, t)
        }
        function b(t, n, r) {
            return e(t, function(t) {
                function i(i, s, o) {
                    var u, a, f, l, c;
                    f = a = t.length>>>0, u = [];
                    if (!f) {
                        i(u);
                        return 
                    }
                    l = function(t, a) {
                        e(t, n, r).then(function(e) {
                            u[a] = e, --f || i(u)
                        }, s, o)
                    };
                    for (c = 0; c < a; c++)
                        c in t ? l(t[c], c) : --f
                }
                return o(i)
            })
        }
        function w(t, n) {
            var r = C(N, arguments, 1);
            return e(t, function(t) {
                var i;
                return i = t.length, r[0] = function(t, r, s) {
                    return e(t, function(t) {
                        return e(r, function(e) {
                            return n(t, e, s, i)
                        })
                    })
                }, T.apply(t, r)
            })
        }
        function E(e) {
            return {
                state: "fulfilled",
                value: e
            }
        }
        function S(e) {
            return {
                state: "rejected",
                reason: e
            }
        }
        function x() {
            return {
                state: "pending"
            }
        }
        function P(e) {
            L.push(e) === 1 && H()
        }
        function H() {
            k(B)
        }
        function B() {
            var e, t = 0;
            while (e = L[t++])
                e();
            L = []
        }
        function j(e) {
            return e
        }
        e.defer = s, e.resolve = r, e.reject = i, e.join = m, e.all = v, e.map = y, e.reduce = w, e.settle = g, e.any = d, e.some = p, e.isPromise = h, e.promise = o, n.prototype = {
            otherwise: function(e) {
                return this.then(D, e)
            },
            ensure: function(e) {
                function t() {
                    return r(e())
                }
                return this.then(t, t).yield(this)
            },
            yield: function(e) {
                return this.then(function() {
                    return e
                })
            }, spread : function(e) {
                return this.then(function(t) {
                    return v(t, function(t) {
                        return e.apply(D, t)
                    })
                })
            }, always: function(e, t) {
                return this.then(e, e, t)
            }
        };
        var T, N, C, k, L, A, O, M, _, D;
        return L = [], A = t.setTimeout, k = typeof setImmediate == "function" ? setImmediate.bind(t) : typeof process == "object" && process.nextTick ? process.nextTick : typeof vertx == "object" ? vertx.runOnLoop : function(e) {
            A(e, 0)
        }, O = Function.prototype, M = O.call, C = O.bind ? M.bind(M) : function(e, t) {
            return e.apply(t, N.call(arguments, 2))
        }, _ = [], N = _.slice, T = _.reduce || function(e) {
            var t, n, r, i, s;
            s = 0, t = Object(this), i = t.length>>>0, n = arguments;
            if (n.length <= 1)
                for (; ;) {
                    if (s in t) {
                        r = t[s++];
                        break
                    }
                    if (++s >= i)
                        throw new TypeError
                } else 
                    r = n[1];
            for (; s < i; ++s)
                s in t && (r = e(r, t[s], s, t));
            return r
        }, e
    })
}(function(e) {
    this.when = e()
}, this), Whiskers("modules.sweepstakes", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 324, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="bound'), n.s(n.f("noCode", e, t, 1), e, t, 0, 38, 46, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" no-code")
        }), e.pop()), n.b('"> <div class="container"> <div class="header module_header sweeps"> <h2>'), n.b(n.v(n.f("title", e, t, 0))), n.b("</h2> </div> "), n.b(n.rp("sweeps_code", e, t, "")), n.b(" "), n.b(n.rp("sweeps_tos", e, t, "")), n.b(" "), n.b(n.rp("sweeps_complete", e, t, "")), n.b(' <div class="sweeps_error"></div> <p class="sweep_legal"> '), n.s(n.f("metadata", e, t, 1), e, t, 0, 276, 291, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(n.v(n.f("short_rules", e, t, 0)))
        }), e.pop()), n.b(" </p> </div> </div> ")
    }), e.pop()), r.fl()
}), Whiskers("partials.sweeps_code", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="sweeps_code"> <p class="sweep_desc">'), r.b(r.v(r.f("description", e, t, 0))), r.b('</p> <div class="sweep_form"> <div class="denied_txt"> <span class="try_again">Try Again!</span> <span class="hint">Hint: '), r.s(r.f("metadata", e, t, 1), e, t, 0, 198, 223, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.v(n.f("codeword_hint_message", e, t, 0)))
    }), e.pop()), r.b('</span> </div> <form> <input class="code_form" type="text" name="code" maxlength="20" required placeholder="Enter Codeword"></input> </form> <button class="button blue large code">Enter</button> </div> </div>'), r.fl()
}), Whiskers("partials.sweeps_tos", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="sweeps_tos"> <p class="terms">'), r.b(r.v(r.f("legal_text", e, t, 0))), r.b('</p> <div class="sweep_form"> <div class="tos_check"> <p class="tos">I agree to the above Terms and Conditions</p> <span class="vote checkbox" title="'), r.b(r.v(r.f("text", e, t, 0))), r.b('"> <span class="check"></span> </span> </div> <button type="submit" class="button blue large tos disabled">Enter</button> </div> </div>'), r.fl()
}), Whiskers("partials.sweeps_complete", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="sweeps_complete module_header"> <h2 class="sweep_desc complete">'), r.s(r.f("metadata", e, t, 1), e, t, 0, 89, 108, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.v(n.f("success_message", e, t, 0)))
    }), e.pop()), r.b("</h2> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e.jQuery, u = e._, a = t.ModuleView.extend({
        breakpoints: i,
        template: s.modules.sweepstakes,
        sweeps_error: "",
        user_code: "",
        initialize: function() {
            this.options.guest.retain()
        },
        render: function() {
            var e = u.extend({}, this.model);
            this.$el.html(this.template.render(e, {
                sweeps_tos: s.partials.sweeps_tos,
                sweeps_code: s.partials.sweeps_code,
                sweeps_complete: s.partials.sweeps_complete
            })), n.Style.moduleStyle(this), this.$el.polyfillPlaceholder()
        },
        resize: function() {
            n.Style.moduleStyle(this)
        },
        codeWords: function() {
            var e = this, t = this.options.guest.login();
            t.then(function() {
                var t = {
                    sweeps_id: e.model.get("data")[0].sweep_id,
                    entered_code: e.user_code
                };
                e.guestCanEnter() ? (e.user_code = e.$("input.code_form").val(), e.$("button.code").text("Verify..."), o.ajax({
                    type: "POST",
                    url: "/_sweeps/sweeps_code",
                    processData: !1,
                    data: JSON.stringify(t),
                    contentType: "application/json",
                    success: function(t) {
                        t === "false" ? (e.$(".denied_txt").addClass("active"), e.$("button.code").text("Submit")) : (e.$(".sweeps_code").remove(), e.$(".sweeps_tos").fadeIn(500))
                    },
                    error: function() {
                        e.showError()
                    }
                })) : e.showError()
            })
        },
        checkboxChecking: function() {
            this.$(".checkbox").hasClass("checked") ? (this.$(".button.tos").addClass("disabled").removeClass("compelte"), this.$(".checkbox").removeClass("checked")) : (this.$(".button.tos").removeClass("disabled").addClass("compelte"), this.$(".checkbox").addClass("checked"))
        },
        sweepsTerms: function() {
            var e = this, t = {
                sweeps_id: this.model.get("data")[0].sweep_id,
                disid: this.options.guest.current.profile.swid.replace("{", "").replace("}", ""),
                entered_code: e.user_code
            };
            this.$("button.tos").text("Verify..."), o.ajax({
                type: "POST",
                url: "/_sweeps/sweeps_info",
                processData: !1,
                data: JSON.stringify(t),
                contentType: "application/json",
                success: function() {
                    e.$(".sweeps_complete").fadeIn(500), e.$(".container").addClass("finished"), e.$(".header.sweeps").css("visibility", "hidden"), e.$(".sweeps_tos").remove()
                }
            })
        },
        events: {
            "click span.checkbox": "checkboxChecking",
            "click .button.code": "codeWords",
            "keypress .code_form": function(t) {
                e.event.keyCode === 13 && (this.codeWords(), t.preventDefault())
            },
            "click .button.tos.compelte": "sweepsTerms"
        },
        guestCanEnter: function() {
            var e = this, t = parseInt(e.model.get("data")[0].metadata.min_age, 10) || 0, n = parseInt(e.model.get("data")[0].metadata.max_age, 10), r = new Date(e.options.guest.current.profile.dateOfBirth), i = new Date, s = new Date, o = n ? new Date: null;
            return s.setFullYear(i.getFullYear() - t), o && o.setFullYear(i.getFullYear() - n - 1), r > s ? (e.sweeps_error = "min_age_gate", !1) : o && r < o ? (e.sweeps_error = "max_age_gate", !1) : !0
        },
        showError: function() {
            var e = this, t;
            switch (e.sweeps_error) {
            case"min_age_gate":
                t = e.model.get("data")[0].metadata.min_age_message;
                break;
            case"max_age_gate":
                t = e.model.get("data")[0].metadata.max_age_message;
                break;
            default:
                t = "There was an error entering the sweepstakes. Please try again."
            }
            t = "<h2>" + t + "</h2>", e.$(".sweeps_code").remove(), e.$(".header.sweeps").css("visibility", "hidden"), e.$(".sweeps_error").html(t).fadeIn(500)
        },
        destroy: function() {
            this.options.guest && this.options.guest.release()
        }
    });
    a.register("sweepstakes", {})
}(this), Whiskers("modules.header", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="header module_header"> '), r.s(r.f("title", e, t, 1), e, t, 0, 65, 83, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("<h2>"), n.b(n.v(n.f("title", e, t, 0))), n.b("</h2>")
    }), e.pop()), r.b(" "), r.s(r.f("desc", e, t, 1), e, t, 0, 103, 120, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b("<h3>"), n.b(n.v(n.f("desc", e, t, 0))), n.b("</h3>")
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.Whiskers, i = e._, s = n.ModuleView.extend({
        className: "module full-height",
        template: r.modules.header,
        render: function() {
            var e = i.first(this.model.get("data")), n = i.extend({
                data: e ? e: null
            }, this.model);
            this.$el.html(this.template.render(n)), t.Style.moduleStyle(this, !1)
        }
    });
    s.register("header", {})
}(this), Whiskers("modules.trailer", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <div class="container"> '), r.s(r.f("data", e, t, 1), e, t, 0, 53, 818, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="video-container trailer-box"> <div class="scaler"></div> '), n.s(n.f("expired", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div id="player_trailer" class="player video trailer-box"> <noscript> <div class="player_error">'), n.b(n.v(n.d("translations.no_javascript", e, t, 0))), n.b("</div> </noscript> </div> ")), n.b(" "), n.s(n.f("expired", e, t, 1), e, t, 0, 313, 378, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="orphan"><p>'), n.b(n.v(n.d("translations.video_expired", e, t, 0))), n.b("</p></div> ")
        }), e.pop()), n.b(' <div id="card" class="card embed skip-styles"> <div class="card_header"> <p class="btn_replay">'), n.b(n.v(n.d("translations.replay", e, t, 0))), n.b('</p> <p class="btn_watch">'), n.b(n.v(n.d("translations.watch_on", e, t, 0))), n.b('</p> </div> <div class="suggested"> <ul> '), n.s(n.f("suggested", e, t, 1), e, t, 0, 615, 637, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(" <li>"), n.b(n.rp("entity", e, t, "")), n.b("</li> ")
        }), e.pop()), n.b(' </ul> </div> </div> </div> <div class="module_header details"> <h2><a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" title="'), n.b(n.v(n.f("title", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</a></h2> <p class="desc">'), n.b(n.v(n.f("description", e, t, 0))), n.b("</p> </div> ")
    }), e.pop()), r.b(" </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = t.Style.breakpoints, r = n.midHigh.toString(), i = e.Grill, s = e.Whiskers, o = e.$, u = e._;
    i.ModuleView.register("trailer", {
        breakpoints: r,
        template: s.modules.trailer,
        assets: {
            scripts: ["//a.dilcdn.com/a/videoplayer-e4e73a5b3283.js"]
        },
        initialize: function() {
            var t = this;
            t.centerSuggested = u.throttle(function() {
                var n = t.$el.find(".card.embed .card_header"), r = t.$el.find(".card.embed ul"), i = r.find("li");
                if (o(e).width() <= 1008) {
                    var s = (r.outerHeight() - n.outerHeight() - i.outerHeight()) / 2;
                    r.css("margin-top", s > 0 ? s : 0)
                } else 
                    r.css("margin-top", 4)
            }, 100), o(e).on("resize." + t.cid, t.centerSuggested)
        },
        render: function() {
            var e = u.extend({}, this.options.ads.helpers(), this.model);
            this.$el.html(this.template.render(e, {
                suggested: this.model.get("suggested"),
                entity: s.entities.video
            })), t.Style.moduleStyle(this)
        },
        ready: function() {
            this.$el.find(".card.embed .suggested li:nth-child(n+3)").addClass("li-nth-child-np3"), this.centerSuggested();
            var e = u.first(this.model.get("data")), n = this.model.get("autoplay")?!0 : !1;
            this.player = new t.MatterhornVideoPlayer("player_trailer", {
                video: e,
                autoPlay: n,
                clickToPlay: !0,
                ads: {
                    enabled: !this.options.cds&&!this.options.noAds,
                    context: this.options.ads
                },
                translations: this.model.get("translations")
            })
        },
        resize: function() {
            this.render(), this.options.ads.restore(), this.ready()
        },
        remove: function() {
            this.player && (this.player.destroy(), this.player = null), o(e).off("resize." + this.cid), i.ModuleView.prototype.remove.call(this)
        }
    })
}(this), Whiskers("modules.trifecta", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("header_data", e, t, 1), e, t, 0, 16, 34, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(n.rp("module_header", e, t, ""))
    }), e.pop()), r.b(' <div class="container"> '), r.s(r.f("data", e, t, 1), e, t, 0, 84, 475, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="item"> <div class="logo-container"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> <span class="logo-box"><span class="aspect"> <img src="'), n.b(n.v(n.f("logo", e, t, 0))), n.b('" class="logo" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </span></span> </a> </div> <div class="description-container"> <p class="desc">'), n.b(n.v(n.f("short_desc", e, t, 0))), n.b('</p> </div> <div class="btn"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" class="button small blue" role="button">'), n.b(n.v(n.d("translations.watch_now", e, t, 0))), n.b("</a> </div> </div> ")
    }), e.pop()), r.b(" </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = n.Style.breakpoints, i = r.midHigh.toString(), s = e.Whiskers, o = e._;
    t.ModuleView.register("trifecta", {
        template: s.modules.trifecta,
        breakpoints: i,
        render: function() {
            var e = o.extend({}, this.model);
            this.$el.html(this.template.render(e, {
                entity: s.entities[this.model.get("type")]
            })), n.Style.moduleStyle(this)
        },
        resize: function() {
            this.render(), this.options.ads.restore()
        }
    })
}(this), Whiskers("modules.upload_stunt", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 63, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="upload_stunt_entity"> '), n.b(n.rp("entity", e, t, "")), n.b(" </div> ")
    }), e.pop()), r.fl()
}), function(e) {
    "use strict";
    var t = e.Grill, n = e.Disney, r = e.Whiskers, i = e._, s = t.ModuleView.extend({
        template: r.modules.upload_stunt,
        assets: {
            scripts: ["//a.dilcdn.com/a/entities/upload_stunt-625df51ae519.js"]
        },
        initialize: function() {
            this.options.guest && this.options.guest.retain()
        },
        createViews: function() {
            this.views = [], this.index = this.model.collection.indexOf(this.model), this.model.data.each(function(e, t) {
                this.views.push(new this.subview({
                    model: e,
                    index: t,
                    moduleView: this,
                    template: this.entity,
                    guest: this.options.guest
                }))
            }, this)
        },
        safeRender: function() {
            this.subview = this.getView(), this.createViews();
            var e = {};
            this.$el.html(this.template.render(e)), i.each(this.views, function(e) {
                e.render(), this.$el.append(e.$el)
            }, this), n.Style.moduleStyle(this)
        },
        destroy: function() {
            this.options.guest && this.options.guest.release(), i.each(this.views, function(e) {
                e.remove()
            }, this)
        }
    });
    s.register("upload_stunt", {
        getView: function() {
            return n.Views.UploadStuntEntityView
        }
    }), s.register("ugc_consent", {
        getView: function() {
            return n.Views.UploadStuntConsentView
        }
    })
}(this), Whiskers("modules.voting", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="bound"> <h3 class="voting-title">'), r.b(r.v(r.f("title", e, t, 0))), r.b('</h3> <h2 class="voting-question">'), r.b(r.v(r.d("header_data.desc", e, t, 0))), r.b('</h2> <ul class="voting-items"> '), r.s(r.f("randData", e, t, 1), e, t, 0, 153, 744, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li data-id="'), n.b(n.v(n.f("id", e, t, 0))), n.b('"> <div class="item-container"> <div class="thumb" style="background-image: url('), n.b(n.v(n.f("thumb", e, t, 0))), n.b(');"> <span class="thumb-sizer"></span> </div> <div class="details"> <div class="title">'), n.b(n.v(n.f("title", e, t, 0))), n.b('</div> <div class="description">'), n.b(n.v(n.f("description", e, t, 0))), n.b('</div> </div> <div class="btn-container vote-container clicon"> <div class="vote-btn"> <div class="vote-btn-checkmark"></div> <div class="vote-btn-title">'), n.b(n.v(n.d("translations.vote", e, t, 0))), n.b('</div> </div> </div> <div class="btn-container share-vote">'), n.b(n.rp("socialLinkPartial", e, t, "")), n.b('</div> </div> <div class="notification"></div> <span class="bg-fill"></span> </li> ')
    }), e.pop()), r.b(" </ul> </div>"), r.fl()
}), Whiskers("modules.voting_player", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="voting-video"> <div class="aspect"> <!-- Orphan video markup --> '), r.s(r.f("expired", e, t, 1), e, t, 1, 0, 0, "") || r.b(' <div id="voting-video-player" class="voting-video-player"></div> '), r.b(" "), r.s(r.f("expired", e, t, 1), e, t, 0, 177, 242, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div class="orphan"><p>'), n.b(n.v(n.d("translations.video_expired", e, t, 0))), n.b("</p></div> ")
    }), e.pop()), r.b(" <!-- /Orphan video markup --> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e.jQuery, i = e._, s = e.Whiskers;
    n.ModuleView.register("voting", {
        template: s.modules.voting,
        assets: {
            scripts: ["//a.dilcdn.com/a/videoplayer-e4e73a5b3283.js"]
        },
        events: {
            "click .vote-container": function(e) {
                this.$voteItem = r(e.currentTarget).closest("li"), this.handleVoteClick()
            },
            "click .thumb": function(e) {
                var t = r(e.currentTarget), n = t.closest("li"), i = n.data("id");
                if (!this.$voteItem || this.$voteItem.data("id") !== i)
                    this.$(".thumb").removeClass("active"), t.addClass("active"), this.$voteItem = n, this.initVideoPlayer(this.$voteItem)
            }
        },
        initialize: function() {
            this.options.guest.retain()
        },
        render: function() {
            var e = this.model.get("data"), n = i.shuffle(e);
            this.model.set("randData", n);
            var o = i.extend({
                shareOverrides: {
                    href: r("meta[property='og:url']").attr("content"),
                    title: r("meta[property='og:title']").attr("content")
                }
            }, this.model);
            this.$el.html(this.template.render(o, {
                socialLinkPartial: s.partials.social_btn
            })), t.Style.moduleStyle(this, !1)
        },
        handleVoteClick: function() {
            var e = this;
            r.when(this.options.guest.login()).then(function(t) {
                e.makeVoteRequest(t)
            })
        },
        makeVoteRequest: function(e) {
            var t = this, n = t.model.get("translations"), i = n.error_default, s = {
                swid: e.profile.swid,
                accessToken: e.tokens.access,
                eventID: this.model.get("campaign_id"),
                voteOptionID: this.$voteItem.data("id")
            };
            r.ajax({
                url: t.model.get("cellophane_host") + "/ugc/content/v1/vote",
                headers: {
                    Authorization: t.model.get("cellophane_auth_key")
                },
                type: "POST",
                data: s,
                dataType: "json"
            }).done(function(e) {
                e.data && e.data._id ? t.animateVoteSuccess() : t.animateVoteFail(i)
            }).fail(function(e) {
                e.status === 409 && (i = n.error_maxnum), t.animateVoteFail(i)
            })
        },
        animateVoteSuccess: function() {
            this.$(".selected").removeClass("selected"), this.$voteItem.addClass("selected")
        },
        animateVoteFail: function(e) {
            this.$(".alert").removeClass("alert"), this.$voteItem.addClass("alert").find(".notification").text(e)
        },
        destroy: function() {
            this.options.guest && this.options.guest.release()
        },
        initVideoPlayer: function(e) {
            var t = e.data("id"), n = this.model.data.findWhere({
                id: t
            });
            this.player && this.player.remove(), this.player = new o({
                model: n
            }), e.find(".thumb-sizer").html(this.player.render()), this.player.init(n)
        }
    });
    var o = n.ModuleView.register("video-player", {
        className: "voting-video-container",
        template: s.modules.voting_player,
        render: function() {
            return this.$el.html(this.template.render()), this.el
        },
        init: function(e) {
            this.player = new t.MatterhornVideoPlayer("voting-video-player", {
                video: e.attributes,
                autoPlay: !0,
                clickToPlay: !0,
                showTitle: !0,
                ads: {
                    enabled: !1,
                    context: !1
                }
            })
        },
        remove: function() {
            this.$(".voting-video-container").remove(), this.player.destroy(), this.player = null, n.BunView.prototype.remove.call(this)
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e._, i = e.jQuery, s = e.Whiskers, o = e.Math, u, a = e.sessionStorage, f = e.JSON;
    n.BunView.register("watch", {
        className: "bun full-width full-height",
        assets: {
            scripts: ["//a.dilcdn.com/a/videoplayer-e4e73a5b3283.js"]
        },
        initialize: function() {
            var t = this;
            t.h = i(e).on("resize." + t.cid, r.throttle(function() {
                var n = i(e).height();
                n !== t.h && (t.h = n, t.layout())
            }, 100)).height()
        },
        render: function() {
            var e = r.extend({
                tags: this.model.get("data")[0].tags,
                contenttype: this.model.get("data")[0].contenttype,
                upNext: this.upNext(),
                cds: this.options.cds
            }, this.options.ads.helpers(), this.model.toJSON());
            this.$el.html(s.modules.watch.render(e, {
                entity: s.entities.video
            })), u = e.data[0], this.layout(), t.Style.moduleStyle(this, !1)
        },
        ready: function() {
            var e = this, n = e.options, i = e.model.get("data")[0].not_embeddable, s = e.model.get("translations"), o = [{
                share_type: "share",
                share_text: s.share,
                output_label: s.quick_link
            }
            ];
            i!==!0 && o.unshift({
                share_type: "embed",
                share_text: s.embed,
                output_label: s.embed
            }), this.tb = new t.ContentToolbarView({
                el: this.$("#meta"),
                model: this.model,
                socialLinks: o,
                cdsMode: n.cds,
                ads: n.ads
            });
            var u = r.first(this.model.get("data"));
            this.player = new t.MatterhornVideoPlayer("player", {
                video: u,
                autoPlay: !0,
                ads: {
                    enabled: !n.cds&&!n.noAds,
                    context: n.ads
                },
                translations: s
            }), this.watchPlaylist = new t.WatchPlaylistView({
                el: this.$("#playlist-holder"),
                model: this.model,
                videoPlayer: this.player,
                cdsMode: n.cds
            })
        },
        upNext: function() {
            var e = r.first(this.model.get("data")), t = a && f.parse(a.getItem("videoHistoryOnDisneyVideo") || "[]"), n = e.next || [], i = [];
            r.each(n, function(e) {
                i.push(e.id)
            });
            var s = r.first(r.difference(i, t)), o = r.find(n, function(e) {
                return e.id === s
            });
            o || (o = r.first(n)), t.push(e.id), t.length > 10 && t.shift();
            try {
                a.setItem("videoHistoryOnDisneyVideo", f.stringify(t))
            } catch (u) {}
            return o
        },
        layout: function() {
            var e = this.$("#video_wrapper .bound");
            e.css("maxWidth", o.min(1024, o.max(320, o.ceil((this.h - 120) / .5625))))
        },
        remove: function() {
            this.player && (this.player.destroy(), this.player = null), n.BunView.prototype.remove.call(this), i(e).off("resize." + this.cid), this.tb.remove(), this.watchPlaylist.remove()
        }
    })
}(this), Whiskers("modules.watch2", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.s(r.f("data", e, t, 1), e, t, 0, 9, 2195, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div id="video_watch_container" itemprop="video" itemscope itemtype="http://schema.org/VideoObject"> <div id="video_wrapper" class="inverted"> <div class="bound"> <div class="scaler"></div> <div id="frame"> <!-- Orphan video markup --> <meta itemprop="duration" content="'), n.b(n.v(n.f("duration_iso", e, t, 0))), n.b('"> <meta itemprop="thumbnail" content="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('"> <meta itemprop="name" content="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <meta itemprop="embedId" content="'), n.b(n.v(n.f("id", e, t, 0))), n.b('"> <meta itemprop="title" content="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <meta itemprop="description" content="'), n.b(n.v(n.f("description", e, t, 0))), n.b('"> <meta itemprop="URL" content="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"> '), n.s(n.f("embedURL", e, t, 1), e, t, 0, 588, 639, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <meta itemprop="embedURL" content="'), n.b(n.v(n.f("embedURL", e, t, 0))), n.b('"> ')
        }), e.pop()), n.b(" "), n.s(n.f("expired", e, t, 1), e, t, 1, 0, 0, "") || (n.b(' <div id="player" class="player"> <noscript> <div class="player_error">'), n.b(n.v(n.d("translations.no_javascript", e, t, 0))), n.b("</div> </noscript> </div> ")), n.b(" "), n.s(n.f("expired", e, t, 1), e, t, 0, 817, 882, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <div class="orphan"><p>'), n.b(n.v(n.d("translations.video_expired", e, t, 0))), n.b("</p></div> ")
        }), e.pop()), n.b(' <!-- /Orphan video markup --> <div id="card" class="card watch skip-styles"> <div class="card_header"> <p class="btn_replay">'), n.b(n.v(n.d("translations.replay_video", e, t, 0))), n.b('</p> <p class="countdown_intro">'), n.b(n.v(n.d("translations.next_video_start", e, t, 0))), n.b(' <span class="countdown">10</span><span class="autoplay">'), n.b(n.v(n.d("translations.autoplay_off", e, t, 0))), n.b("</span></p> "), n.s(n.f("upNext", e, t, 1), e, t, 0, 1223, 1303, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('"><p class="btn_next">'), n.b(n.v(n.d("translations.watch_next_video", e, t, 0))), n.b("</p></a>")
        }), e.pop()), n.b(' </div> <div class="suggested"> <ul> '), n.s(n.f("upNext", e, t, 1), e, t, 0, 1362, 1401, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b('<li class="priority"> '), n.b(n.rp("entity"
            , e, t, "")), n.b(" </li>")
        }), e.pop()), n.b(" "), n.s(n.f("suggested", e, t, 1), e, t, 0, 1427, 1449, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b("<li> "), n.b(n.rp("entity", e, t, "")), n.b(" </li>")
        }), e.pop()), n.b(' </ul> </div> </div> </div> </div> </div> <!-- End of video_wrapper --> <div class="content-toolbar-container"> <div id="toolbar" class="inverted"> <div id="meta"></div> </div> <!-- End of toolbar --> </div> <!-- End of bound --> '), n.s(n.f("contentToolbarPromoLink", e, t, 1), e, t, 0, 1721, 2078, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('" class="watch_icon item" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> <div class="messageLink"> '), n.s(n.f("asset_desktop_image", e, t, 1), e, t, 0, 1831, 1916, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b(' <div class="watch-icon"> <img src="'), n.b(n.v(n.f("asset_desktop_image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </div> ')
            }), e.pop()), n.b(' <div class="copy'), n.s(n.f("asset_desktop_image", e, t, 1), e, t, 1, 0, 0, "") || n.b(" noImage "), n.b('"> <p aria-hidden="true">'), n.b(n.v(n.f("description", e, t, 0))), n.b("</p> </div> </div> </a> ")
        }), e.pop()), n.b(' <div id="playlist-holder" class="open"></div> <!-- end playlist --> </div> <!-- End --> ')
    }), e.pop()), r.fl()
}), Whiskers("shared.contenttoolbar2", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div id="toolbar"> <div id="metadata" class="bound"> <div class="module_body"> <div class="contentarea col-container"> <div class="content"> <h1>'), r.b(r.v(r.f("title", e, t, 0))), r.b('</h1> <p itemprop="description">'), r.b(r.v(r.f("description", e, t, 0))), r.b(r.v(r.f("rating", e, t, 0))), r.b('</p> </div> <!-- end .content --> </div><!-- end .contentarea --> <div class="secondary-content"> <div class="promo-social-wrapper"> <div class="meta-content"> <div class="productlink"> '), r.s(r.f("contentToolbarTitleLink", e, t, 1), e, t, 0, 425, 565, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(" "), n.s(n.f("asset_desktop_image", e, t, 1), e, t, 0, 450, 540, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('" class="icon"> <img src="'), n.b(n.v(n.f("asset_desktop_image", e, t, 0))), n.b('" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> </a> ')
        }), e.pop()), n.b(" ")
    }), e.pop()), r.b(" "), r.s(r.f("badge", e, t, 1), e, t, 0, 604, 690, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <span class="watch_icon"> <img src="'), n.b(n.v(n.f("badge", e, t, 0))), n.b('" class="watchpageicon" alt=""> </span> ')
    }), e.pop()), r.b(" "), r.s(r.f("contentToolbarTitleLink", e, t, 1), e, t, 0, 729, 781, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <a class="icon_title" href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</a> ")
    }), e.pop()), r.b(' </div> <!-- end .productlink --> <div class="social-links-wrapper"> '), r.s(r.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (r.b(" "), r.s(r.f("suppress_social", e, t, 1), e, t, 1, 0, 0, "") || (r.b(" "), r.b(r.rp("socialLinkPartial", e, t, "")), r.b(" ")), r.b(" ")), r.b(" "), r.s(r.f("purchase", e, t, 1), e, t, 0, 974, 1072, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <div id="purchase" class="bound"> <p> <a href="'), n.b(n.v(n.f("url", e, t, 0))), n.b('" target="_blank">'), n.b(n.v(n.f("text", e, t, 0))), n.b("</a> </p> </div> ")
    }), e.pop()), r.b("<!-- end #purchase --> </div><!-- .social-links-wrapper --> </div><!-- end .meta-content --> "), r.s(r.f("cds", e, t, 1), e, t, 1, 0, 0, "") || (r.b(' <div class="ad-container"> <div class="companionad">'), r.b(r.t(r.f("companionAd", e, t, 0))), r.b("</div> "), r.b(r.t(r.f("mobileAd", e, t, 0))), r.b(" </div> ")), r.b("<!-- end .ad-container--> </div> <!-- end .promo-social-wrapper --> </div><!-- end .subcontent --> </div> </div> </div>"), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = e.Backbone, r = e.jQuery, i = e._, s = e.Whiskers;
    t.ContentToolbarView2 = n.View.extend({
        initialize: function() {
            this.render()
        },
        render: function() {
            var e = this.options.ads, t = this.model.get("translations"), n = i.extend({
                translations: t,
                cds: this.options.cdsMode&&!e.showCDSAds(),
                companionAd: function() {
                    return e.tag(this.type.toLowerCase() + "Companion")
                }
            }, this.model.get("data")[0], e ? e.helpers() : {});
            this.$el.html(s.shared.contenttoolbar2.render(n, {
                socialLinkPartial: s.partials.social_btn
            }))
        },
        remove: function() {
            r("body").off("click.watch")
        }
    })
}(this), function(e) {
    "use strict";
    var t = e.Disney, n = e.Grill, r = e._, i = e.jQuery, s = e.Whiskers, o = e.Math, u, a = e.sessionStorage, f = e.JSON;
    n.BunView.register("watch_2", {
        className: "bun full-width full-height",
        assets: {
            scripts: ["//a.dilcdn.com/a/videoplayer-e4e73a5b3283.js"]
        },
        initialize: function() {
            var t = this;
            t.h = i(e).on("resize." + t.cid, r.throttle(function() {
                var n = i(e).height();
                n !== t.h && (t.h = n, t.layout())
            }, 100)).height()
        },
        render: function() {
            var e = r.extend({
                tags: this.model.get("data")[0].tags,
                contenttype: this.model.get("data")[0].contenttype,
                upNext: this.upNext(),
                cds: this.options.cds
            }, this.options.ads.helpers(), this.model.toJSON());
            this.$el.html(s.modules.watch2.render(e, {
                entity: s.entities.video
            })), u = e.data[0], this.layout(), t.Style.moduleStyle(this, !1)
        },
        ready: function() {
            var e = this, n = e.options;
            this.tb = new t.ContentToolbarView2({
                el: this.$("#meta"),
                model: this.model,
                cdsMode: n.cds,
                ads: n.ads
            });
            var i = r.first(this.model.get("data"));
            this.player = new t.MatterhornVideoPlayer("player", {
                video: i,
                autoPlay: !0,
                ads: {
                    enabled: !n.cds&&!n.noAds,
                    context: n.ads
                },
                translations: this.model.get("translations")
            }), this.watchPlaylist = new t.WatchPlaylistView({
                el: this.$("#playlist-holder"),
                model: this.model,
                videoPlayer: this.player,
                cdsMode: n.cds
            })
        },
        upNext: function() {
            var e = r.first(this.model.get("data")), t = a && f.parse(a.getItem("videoHistoryOnDisneyVideo") || "[]"), n = e.next || [], i = [];
            r.each(n, function(e) {
                i.push(e.id)
            });
            var s = r.first(r.difference(i, t)), o = r.find(n, function(e) {
                return e.id === s
            });
            o || (o = r.first(n)), t.push(e.id), t.length > 10 && t.shift();
            try {
                a.setItem("videoHistoryOnDisneyVideo", f.stringify(t))
            } catch (u) {}
            return o
        },
        layout: function() {
            var e = this.$("#video_wrapper .bound");
            e.css("maxWidth", o.min(1024, o.max(320, o.ceil((this.h - 120) / .5625))))
        },
        remove: function() {
            this.player && (this.player.destroy(), this.player = null), n.BunView.prototype.remove.call(this), i(e).off("resize." + this.cid), this.tb.remove(), this.watchPlaylist.remove()
        }
    })
}(this), Whiskers("modules.watch_playlist", function(e, t, n) {
    "use strict";
    var r = this;
    return r.b(n = n || ""), r.b('<div class="watch-playlist slider-view pagination-styles"> <div class="bound"> <div class="hider"></div> <div class="slides peek-'), r.b(r.v(r.f("rpp", e, t, 0))), r.b(' peeking"> <div class="peek left"></div> <div class="peek"></div> <ol class="slider-list peek-'), r.b(r.v(r.f("rpp", e, t, 0))), r.b('"> '), r.s(r.f("pages", e, t, 1), e, t, 0, 250, 869, "{{ }}") && (r.rs(e, t, function(e, t, n) {
        n.b(' <li class="slider-page"> <ul class="col-container cols-'), n.b(n.v(n.f("rpp", e, t, 0))), n.b('"> '), n.s(n.f("models", e, t, 1), e, t, 0, 327, 845, "{{ }}") && (n.rs(e, t, function(e, t, n) {
            n.b(' <li class="col"> <a href="'), n.b(n.v(n.f("href", e, t, 0))), n.b('" id="'), n.b(n.v(n.f("id", e, t, 0))), n.b('" data-next="'), n.b(n.v(n.f("nextItem", e, t, 0))), n.b('"> <span class="thumb-box"> <span class="aspect"> <span class="now-playing-title"> <h3 class="overlay-title" aria-hidden="true">'), n.b(n.v(n.f("now_playing", e, t, 0))), n.b('</h3> </span> <span class="thumbnail"> <img src="'), n.b(n.v(n.f("thumb", e, t, 0))), n.b('" class="thumb" alt="'), n.b(n.v(n.f("title", e, t, 0))), n.b('"> '), n.s(n.f("badge", e, t, 1), e, t, 0, 643, 685, "{{ }}") && (n.rs(e, t, function(e, t, n) {
                n.b('<img src="'), n.b(n.v(n.f("badge", e, t, 0))), n.b('" class="badge" alt="">')
            }), e.pop()), n.b(' </span> </span> <div class="hover-title skip-styles"> <div class="text-hidden"><h3 aria-hidden="true">'), n.b(n.v(n.f("title", e, t, 0))), n.b("</h3></div> </div> </span> </a> </li> ")
        }), e.pop()), n.b(" </ul> </li> ")
    }), e.pop()), r.b(' </ol> <div class="count"> <p><span class="title">'), r.b(r.v(r.f("playlistTitle", e, t, 0))), r.b(" &nbsp;|&nbsp; </span>"), r.b(r.v(r.f("currentIndex", e, t, 0))), r.b(" "), r.b(r.v(r.d("translations.of", e, t, 0))), r.b(" "), r.b(r.v(r.f("totolItems", e, t, 0))), r.b('</p> </div> </div> </div> </div> <div class="pulldown-bar"><div class="arrow up"></div></div>'), r.fl()
}), function(e) {
    "use strict";
    var t = e.Disney, n = t.Style.breakpoints, r = [n.midLow, n.midHigh], i = e.Backbone, s = e.jQuery, o = e.Whiskers, u = t.Cookie, a = e._, f=!1, l = null, c = i.Resize.width, h = "manual-play", p = /^(?:(?:[a-z]+:)?\/\/[a-z0-9:.\-]+)?\/(.*)$/;
    t.WatchPlaylistView = i.View.extend({
        videoPlayer: null,
        breakpoints: r.join(" "),
        events: {
            "click .pulldown-bar": function() {
                this.showPlaylist()
            },
            "click .hider": function() {
                this.showPlaylist()
            },
            "click .watch-playlist .slides": function(e) {
                e.stopPropagation()
            }
        },
        showPlaylist: function() {
            if (l)
                return;
            f ? (f=!1, s("#playlist-holder").addClass("closed").removeClass("open"), s(".hider").fadeIn()) : (f=!0, s("#playlist-holder").addClass("open").removeClass("closed"), s(".hider").fadeOut())
        },
        initialize: function() {
            this.videoPlayer = this.options.videoPlayer, this.init()
        },
        render: function() {
            var t = this, r = this.rpp(), i = this.model.get("translations"), u = a.indexOf(a.pluck(this.listData.models, "id"), this.model.get("data")[0].id), f = Math.floor(u / r), h = {
                cds: this.options.cdsMode,
                rpp: r,
                currentTitle: t.listData.models[u].get("title"),
                currentIndex: t.listData.models[u].get("order"),
                totolItems: t.listData.length,
                playlistTitle: t.playlistTitle,
                pages: t.listData.pages(r, {
                    nest: 1
                }),
                translations: i
            };
            this.$el.html(o.modules.watch_playlist.render(h)), this.$(".slides ol").swipeasaurus({
                arrows: !0,
                loop: !1,
                right: .2 / (r + .2),
                index: f
            }), t.$(".slides ol").delegate("a", "click", function(n) {
                var r = /^(?:(?:[a-z]+:)?\/\/[a-z0-9:.\-]+)?\/(.*)$/, i = this;
                if (i) {
                    var o = r.exec(i);
                    if (o && o[1]) {
                        if (e.Tracker) {
                            var u = s(n.currentTarget), a = s.trim(u.find("img").attr("alt") || u.text()), f = u.attr("href"), l = t.$(".slides ol a").index(u), c = ["stream", "playlist", l, "link"].join("/"), h = [c, a].join("/");
                            e.Tracker.link({
                                linkName: h,
                                linkPosition: c,
                                href: f
                            })
                        }
                        e.Backbone.history.navigate(o[1], !0)
                    }
                }
                return !1
            });
            var p = "#" + this.listData.models[u].get("id");
            s(p).find(".thumb-box").toggleClass("active"), s(p).find(".thumbnail").toggleClass("active"), s(p).find(".now-playing-title").toggleClass("active"), s(".hider").fadeOut(), c() >= n.midHigh && (l = e.setTimeout(this.closePlaylist, 5e3)), this.videoPlayer.adPattern.setPlaylistLoaded(!0), this.playlistLoaded=!0
        },
        closePlaylist: function() {
            l && (e.clearTimeout(l), l = null), s("#playlist-holder").addClass("closed").removeClass("open"), s(".hider").fadeIn()
        },
        init: function() {
            var n = /(?:\?|^|&)playlist=([0-9a-f]+)/, r = this, o = n.exec(e.location.search);
            if (o) {
                o = o[1];
                var u = this.options.cdsMode ? "?cds": "";
                s.ajax((t.ajaxBase || "/") + "lists/" + o + ".json" + u, {
                    dataType: "json",
                    cache: !0
                }).then(function(e) {
                    var t = e.data;
                    r.playlistTitle = e.title, r.videoPlayer.adPattern.setPlaylistAdTime(e.adTime), r.videoPlayer.adPattern.setPlaylistAdPattern(e.adPattern), r.videoPlayer.messageBus.addMessageListener("playComplete", a.bind(r.loadNextPlaylistItem, r));
                    var n = r.options.cdsMode ? "&": "?";
                    for (var s = 0; s < t.length; s++)
                        t[s].href = t[s].href + n + "playlist=" + o, t[s].nextItem = t[s + 1] ? t[s + 1].href + n + "playlist=" + o : "", t[s].order = s + 1;
                    r.listData = new i.LazyCollection(t), r.render(), r.configureEndCard(r.videoPlayer)
                })
            }
        },
        rpp: function() {
            var e = c() >= n.midHigh;
            return e ? 4 : 2
        },
        isPlaylistLoaded: function() {
            return this.playlistLoaded
        },
        configureEndCard: function() {
            var t = s(".card"), n = "#" + this.options.videoPlayer.config.video.id, r = s(n).attr("data-next");
            r ? this.options.videoPlayer.config.showEndCard=!1 : (this.options.videoPlayer.config.showEndCard=!0, t.find(".countdown_intro").css("display", "none"), t.find(".btn_next").css("display", "none"), t.find(".priority").removeClass("priority"), t.find(".btn_replay").css("left", "410px"), u.set(h, "ON"), this.options.videoPlayer.endCard.replay = function() {
                var t = a.first(s(".col")), n = s(t).find("a").attr("href");
                if (n) {
                    var r = p.exec(n);
                    r && r[1] && e.Backbone.history.navigate(r[1], !0)
                }
            })
        },
        loadNextPlaylistItem: function() {
            var t = "#" + this.options.videoPlayer.config.video.id, n = s(t).attr("data-next");
            if (n) {
                var r = p.exec(n);
                r && r[1] && e.Backbone.history.navigate(r[1], !0)
            }
        }
    })
}(this);
