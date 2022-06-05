/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("intl", function(e, t) {
    var n = {}, r = "yuiRootLang", i = "yuiActiveLang", s = [];
    e.mix(e.namespace("Intl"), {
        _mod: function(e) {
            return n[e] || (n[e] = {}), n[e]
        },
        setLang: function(e, t) {
            var n = this._mod(e), s = n[i], o=!!n[t];
            return o && t !== s && (n[i] = t, this.fire("intl:langChange", {
                module: e,
                prevVal: s,
                newVal: t === r ? "": t
            })), o
        },
        getLang: function(e) {
            var t = this._mod(e)[i];
            return t === r ? "" : t
        },
        add: function(e, t, n) {
            t = t || r, this._mod(e)[t] = n, this.setLang(e, t)
        },
        get: function(t, n, r) {
            var s = this._mod(t), o;
            return r = r || s[i], o = s[r] || {}, n ? o[n] : e.merge(o)
        },
        getAvailableLangs: function(t) {
            var n = e.Env._loader, r = n && n.moduleInfo[t], i = r && r.lang;
            return i ? i.concat() : s
        }
    }), e.augment(e.Intl, e.EventTarget), e.Intl.publish("intl:langChange", {
        emitFacade: !0
    })
}, "3.11.0", {
    requires: ["intl-base", "event-custom"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("array-extras", function(e, t) {
    var n = e.Array, r = e.Lang, i = Array.prototype;
    n.lastIndexOf = r._isNative(i.lastIndexOf) ? function(e, t, n) {
        return n || n === 0 ? e.lastIndexOf(t, n) : e.lastIndexOf(t)
    } : function(e, t, n) {
        var r = e.length, i = r - 1;
        if (n || n === 0)
            i = Math.min(n < 0 ? r + n : n, r);
        if (i>-1 && r > 0)
            for (; i>-1; --i)
                if (i in e && e[i] === t)
                    return i;
        return - 1
    }, n.unique = function(e, t) {
        var n = 0, r = e.length, i = [], s, o, u, a;
        e: for (; n < r; n++) {
            a = e[n];
            for (s = 0, u = i.length; s < u; s++) {
                o = i[s];
                if (t) {
                    if (t.call(e, a, o, n, e))
                        continue e
                } else if (a === o)
                    continue e
            }
            i.push(a)
        }
        return i
    }, n.filter = r._isNative(i.filter) ? function(e, t, n) {
        return i.filter.call(e, t, n)
    } : function(e, t, n) {
        var r = 0, i = e.length, s = [], o;
        for (; r < i; ++r)
            r in e && (o = e[r], t.call(n, o, r, e) && s.push(o));
        return s
    }, n.reject = function(e, t, r) {
        return n.filter(e, function(e, n, i) {
            return !t.call(r, e, n, i)
        })
    }, n.every = r._isNative(i.every) ? function(e, t, n) {
        return i.every.call(e, t, n)
    } : function(e, t, n) {
        for (var r = 0, i = e.length; r < i; ++r)
            if (r in e&&!t.call(n, e[r], r, e))
                return !1;
        return !0
    }, n.map = r._isNative(i.map) ? function(e, t, n) {
        return i.map.call(e, t, n)
    } : function(e, t, n) {
        var r = 0, s = e.length, o = i.concat.call(e);
        for (; r < s; ++r)
            r in e && (o[r] = t.call(n, e[r], r, e));
        return o
    }, n.reduce = r._isNative(i.reduce) ? function(e, t, n, r) {
        return i.reduce.call(e, function(e, t, i, s) {
            return n.call(r, e, t, i, s)
        }, t)
    } : function(e, t, n, r) {
        var i = 0, s = e.length, o = t;
        for (; i < s; ++i)
            i in e && (o = n.call(r, o, e[i], i, e));
        return o
    }, n.find = function(e, t, n) {
        for (var r = 0, i = e.length; r < i; r++)
            if (r in e && t.call(n, e[r], r, e))
                return e[r];
        return null
    }, n.grep = function(e, t) {
        return n.filter(e, function(e, n) {
            return t.test(e)
        })
    }, n.partition = function(e, t, r) {
        var i = {
            matches: [],
            rejects: []
        };
        return n.each(e, function(n, s) {
            var u = t.call(r, n, s, e) ? i.matches: i.rejects;
            u.push(n)
        }), i
    }, n.zip = function(e, t) {
        var r = [];
        return n.each(e, function(e, n) {
            r.push([e, t[n]])
        }), r
    }, n.flatten = function(e) {
        var t = [], i, s, o;
        if (!e)
            return t;
        for (i = 0, s = e.length; i < s; ++i)
            o = e[i], r.isArray(o) ? t.push.apply(t, n.flatten(o)) : t.push(o);
        return t
    }
}, "3.11.0", {
    requires: ["yui-base"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("escape", function(e, t) {
    var n = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "/": "&#x2F;",
        "`": "&#x60;"
    }, r = {
        html: function(e) {
            return (e + "").replace(/[&<>"'\/`]/g, r._htmlReplacer)
        },
        regex: function(e) {
            return (e + "").replace(/[\-$\^*()+\[\]{}|\\,.?\s]/g, "\\$&")
        },
        _htmlReplacer: function(e) {
            return n[e]
        }
    };
    r.regexp = r.regex, e.Escape = r
}, "3.11.0", {
    requires: ["yui-base"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("autocomplete-base", function(e, t) {
    function T() {}
    var n = e.Escape, r = e.Lang, i = e.Array, s = e.Object, o = r.isFunction, u = r.isString, a = r.trim, f = e.Attribute.INVALID_VALUE, l = "_functionValidator", c = "_sourceSuccess", h = "allowBrowserAutocomplete", p = "inputNode", d = "query", v = "queryDelimiter", m = "requestTemplate", g = "results", y = "resultListLocator", b = "value", w = "valueChange", E = "clear", S = d, x = g;
    T.prototype = {
        initializer: function() {
            e.before(this._bindUIACBase, this, "bindUI"), e.before(this._syncUIACBase, this, "syncUI"), this.publish(E, {
                defaultFn: this._defClearFn
            }), this.publish(S, {
                defaultFn: this._defQueryFn
            }), this.publish(x, {
                defaultFn: this._defResultsFn
            })
        },
        destructor: function() {
            this._acBaseEvents && this._acBaseEvents.detach(), delete this._acBaseEvents, delete this._cache, delete this._inputNode, delete this._rawSource
        },
        clearCache: function() {
            return this._cache && (this._cache = {}), this
        },
        sendRequest: function(t, n) {
            var r, i = this.get("source");
            return t || t === "" ? this._set(d, t) : t = this.get(d) || "", i && (n || (n = this.get(m)), r = n ? n.call(this, t) : t, i.sendRequest({
                query: t,
                request: r,
                callback: {
                    success: e.bind(this._onResponse, this, t)
                }
            })), this
        },
        _bindUIACBase: function() {
            var t = this.get(p), n = t && t.tokenInput;
            n && (t = n.get(p), this._set("tokenInput", n));
            if (!t) {
                e.error("No inputNode specified.");
                return 
            }
            this._inputNode = t, this._acBaseEvents = new e.EventHandle([t.on(w, this._onInputValueChange, this), t.on("blur", this._onInputBlur, this), this.after(h + "Change", this._syncBrowserAutocomplete), this.after("sourceTypeChange", this._afterSourceTypeChange), this.after(w, this._afterValueChange)])
        },
        _syncUIACBase: function() {
            this._syncBrowserAutocomplete(), this.set(b, this.get(p).get(b))
        },
        _createArraySource: function(e) {
            var t = this;
            return {
                type: "array",
                sendRequest: function(n) {
                    t[c](e.concat(), n)
                }
            }
        },
        _createFunctionSource: function(e) {
            var t = this;
            return {
                type: "function",
                sendRequest: function(n) {
                    function i(e) {
                        t[c](e || [], n)
                    }
                    var r;
                    (r = e(n.query, i)) && i(r)
                }
            }
        },
        _createObjectSource: function(e) {
            var t = this;
            return {
                type: "object",
                sendRequest: function(n) {
                    var r = n.query;
                    t[c](s.owns(e, r) ? e[r] : [], n)
                }
            }
        },
        _functionValidator: function(e) {
            return e === null || o(e)
        },
        _getObjectValue: function(e, t) {
            if (!e)
                return;
            for (var n = 0, r = t.length; e && n < r; n++)
                e = e[t[n]];
            return e
        },
        _parseResponse: function(e, t, r) {
            var i = {
                data: r,
                query: e,
                results: []
            }, s = this.get(y), o = [], u = t && t.results, a, f, l, c, h, p, d, v, m, g, b;
            u && s && (u = s.call(this, u));
            if (u && u.length) {
                a = this.get("resultFilters"), b = this.get("resultTextLocator");
                for (p = 0, d = u.length; p < d; ++p)
                    m = u[p], g = b ? b.call(this, m) : m.toString(), o.push({
                        display: n.html(g),
                        raw: m,
                        text: g
                    });
                for (p = 0, d = a.length; p < d; ++p) {
                    o = a[p].call(this, e, o.concat());
                    if (!o)
                        return;
                    if (!o.length)
                        break
                }
                if (o.length) {
                    l = this.get("resultFormatter"), h = this.get("resultHighlighter"), v = this.get("maxResults"), v && v > 0 && o.length > v && (o.length = v);
                    if (h) {
                        c = h.call(this, e, o.concat());
                        if (!c)
                            return;
                        for (p = 0, d = c.length; p < d; ++p)
                            m = o[p], m.highlighted = c[p], m.display = m.highlighted
                    }
                    if (l) {
                        f = l.call(this, e, o.concat());
                        if (!f)
                            return;
                        for (p = 0, d = f.length; p < d; ++p)
                            o[p].display = f[p]
                    }
                }
            }
            i.results = o, this.fire(x, i)
        },
        _parseValue: function(e) {
            var t = this.get(v);
            return t && (e = e.split(t), e = e[e.length - 1]), r.trimLeft(e)
        },
        _setEnableCache: function(e) {
            this._cache = e ? {} : null
        },
        _setLocator: function(e) {
            if (this[l](e))
                return e;
            var t = this;
            return e = e.toString().split("."), function(n) {
                return n && t._getObjectValue(n, e)
            }
        },
        _setRequestTemplate: function(e) {
            return this[l](e) ? e : (e = e.toString(), function(t) {
                return r.sub(e, {
                    query: encodeURIComponent(t)
                })
            })
        },
        _setResultFilters: function(t) {
            var n, s;
            return t === null ? [] : (n = e.AutoCompleteFilters, s = function(e) {
                return o(e) ? e : u(e) && n && o(n[e]) ? n[e] : !1
            }, r.isArray(t) ? (t = i.map(t, s), i.every(t, function(e) {
                return !!e
            }) ? t : f) : (t = s(t), t ? [t] : f))
        },
        _setResultHighlighter: function(t) {
            var n;
            return this[l](t) ? t : (n = e.AutoCompleteHighlighters, u(t) && n && o(n[t]) ? n[t] : f)
        },
        _setSource: function(t) {
            var n = this.get("sourceType") || r.type(t), i;
            return t && o(t.sendRequest) || t === null || n === "datasource" ? (this._rawSource = t, t) : (i = T.SOURCE_TYPES[n]) ? (this._rawSource = t, r.isString(i) ? this[i](t) : i(t)) : (e.error("Unsupported source type '" + n + "'. Maybe autocomplete-sources isn't loaded?"), f)
        },
        _sourceSuccess: function(e, t) {
            t.callback.success({
                data: e,
                response: {
                    results: e
                }
            })
        },
        _syncBrowserAutocomplete: function() {
            var e = this.get(p);
            e.get("nodeName").toLowerCase() === "input" && e.setAttribute("autocomplete", this.get(h) ? "on" : "off")
        },
        _updateValue: function(e) {
            var t = this.get(v), n, s, o;
            e = r.trimLeft(e), t && (n = a(t), o = i.map(a(this.get(b)).split(t), a), s = o.length, s > 1 && (o[s - 1] = e, e = o.join(n + " ")), e = e + n + " "), this.set(b, e)
        },
        _afterSourceTypeChange: function(e) {
            this._rawSource && this.set("source", this._rawSource)
        },
        _afterValueChange: function(e) {
            var t = e.newVal, n = this, r = e.src === T.UI_SRC, i, s, o, u;
            r || n._inputNode.set(b, t), o = n.get("minQueryLength"), u = n._parseValue(t) || "", o >= 0 && u.length >= o ? r ? (i = n.get("queryDelay"), s = function() {
                n.fire(S, {
                    inputValue: t,
                    query: u,
                    src: e.src
                })
            }, i ? (clearTimeout(n._delay), n._delay = setTimeout(s, i)) : s()) : n._set(d, u) : (clearTimeout(n._delay), n.fire(E, {
                prevVal: e.prevVal ? n._parseValue(e.prevVal): null,
                src: e.src
            }))
        },
        _onInputBlur: function(e) {
            var t = this.get(v), n, i, s;
            if (t&&!this.get("allowTrailingDelimiter")) {
                t = r.trimRight(t), s = i = this._inputNode.get(b);
                if (t)
                    while ((i = r.trimRight(i)) && (n = i.length - t.length) && i.lastIndexOf(t) === n)
                        i = i.substring(0, n);
                else 
                    i = r.trimRight(i);
                i !== s && this.set(b, i)
            }
        },
        _onInputValueChange: function(e) {
            var t = e.newVal;
            t !== this.get(b) && this.set(b, t, {
                src: T.UI_SRC
            })
        },
        _onResponse: function(e, t) {
            e === (this.get(d) || "") && this._parseResponse(e || "", t.response, t.data)
        },
        _defClearFn: function() {
            this._set(d, null), this._set(g, [])
        },
        _defQueryFn: function(e) {
            this.sendRequest(e.query)
        },
        _defResultsFn: function(e) {
            this._set(g, e[g])
        }
    }, T.ATTRS = {
        allowBrowserAutocomplete: {
            value: !1
        },
        allowTrailingDelimiter: {
            value: !1
        },
        enableCache: {
            lazyAdd: !1,
            setter: "_setEnableCache",
            value: !0
        },
        inputNode: {
            setter: e.one,
            writeOnce: "initOnly"
        },
        maxResults: {
            value: 0
        },
        minQueryLength: {
            value: 1
        },
        query: {
            readOnly: !0,
            value: null
        },
        queryDelay: {
            value: 100
        },
        queryDelimiter: {
            value: null
        },
        requestTemplate: {
            setter: "_setRequestTemplate",
            value: null
        },
        resultFilters: {
            setter: "_setResultFilters",
            value: []
        },
        resultFormatter
        : {
            validator: l,
            value: null
        },
        resultHighlighter: {
            setter: "_setResultHighlighter",
            value: null
        },
        resultListLocator: {
            setter: "_setLocator",
            value: null
        },
        results: {
            readOnly: !0,
            value: []
        },
        resultTextLocator: {
            setter: "_setLocator",
            value: null
        },
        source: {
            setter: "_setSource",
            value: null
        },
        sourceType: {
            value: null
        },
        tokenInput: {
            readOnly: !0
        },
        value: {
            value: ""
        }
    }, T._buildCfg = {
        aggregates: ["SOURCE_TYPES"],
        statics: ["UI_SRC"]
    }, T.SOURCE_TYPES = {
        array: "_createArraySource",
        "function": "_createFunctionSource",
        object: "_createObjectSource"
    }, T.UI_SRC = e.Widget && e.Widget.UI_SRC || "ui", e.AutoCompleteBase = T
}, "3.11.0", {
    optional: ["autocomplete-sources"],
    requires: ["array-extras", "base-build", "escape", "event-valuechange", "node-base"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("autocomplete-sources", function(e, t) {
    var n = e.AutoCompleteBase, r = e.Lang, i = "_sourceSuccess", s = "maxResults", o = "requestTemplate", u = "resultListLocator";
    e.mix(n.prototype, {
        _YQL_SOURCE_REGEX: /^(?:select|set|use)\s+/i,
        _beforeCreateObjectSource: function(t) {
            return t instanceof e.Node && t.get("nodeName").toLowerCase() === "select" ? this._createSelectSource(t) : e.JSONPRequest && t instanceof e.JSONPRequest ? this._createJSONPSource(t) : this._createObjectSource(t)
        },
        _createIOSource: function(t) {
            function a(n) {
                var o = n.request;
                if (r._cache && o in r._cache) {
                    r[i](r._cache[o], n);
                    return 
                }
                s && s.isInProgress() && s.abort(), s = e.io(r._getXHRUrl(t, n), {
                    on: {
                        success: function(t, s) {
                            var u;
                            try {
                                u = e.JSON.parse(s.responseText)
                            } catch (a) {
                                e.error("JSON parse error", a)
                            }
                            u && (r._cache && (r._cache[o] = u), r[i](u, n))
                        }
                    }
                })
            }
            var n = {
                type: "io"
            }, r = this, s, o, u;
            return n.sendRequest = function(t) {
                o = t;
                if (u)
                    return;
                u=!0, e.use("io-base", "json-parse", function() {
                    n.sendRequest = a, a(o)
                })
            }, n
        },
        _createJSONPSource: function(t) {
            function u(e) {
                var n = e.request, s = e.query;
                if (r._cache && n in r._cache) {
                    r[i](r._cache[n], e);
                    return 
                }
                t._config.on.success = function(t) {
                    r._cache && (r._cache[n] = t), r[i](t, e)
                }, t.send(s)
            }
            var n = {
                type: "jsonp"
            }, r = this, s, o;
            return n.sendRequest = function(i) {
                s = i;
                if (o)
                    return;
                o=!0, e.use("jsonp", function() {
                    t instanceof e.JSONPRequest || (t = new e.JSONPRequest(t, {
                        format: e.bind(r._jsonpFormatter, r)
                    })), n.sendRequest = u, u(s)
                })
            }, n
        },
        _createSelectSource: function(e) {
            var t = this;
            return {
                type: "select",
                sendRequest: function(n) {
                    var r = [];
                    e.get("options").each(function(e) {
                        r.push({
                            html: e.get("innerHTML"),
                            index: e.get("index"),
                            node: e,
                            selected: e.get("selected"),
                            text: e.get("text"),
                            value: e.get("value")
                        })
                    }), t[i](r, n)
                }
            }
        },
        _createStringSource: function(e) {
            return this._YQL_SOURCE_REGEX.test(e) ? this._createYQLSource(e) : e.indexOf("{callback}")!==-1 ? this._createJSONPSource(e) : this._createIOSource(e)
        },
        _createYQLSource: function(t) {
            function c(o) {
                var u = o.query, a = n.get("yqlEnv"), f = n.get(s), c, h, p;
                p = r.sub(t, {
                    maxResults: f > 0 ? f: 1e3,
                    request: o.request,
                    query: u
                });
                if (n._cache && p in n._cache) {
                    n[i](n._cache[p], o);
                    return 
                }
                c = function(e) {
                    n._cache && (n._cache[p] = e), n[i](e, o)
                }, h = {
                    proto: n.get("yqlProtocol")
                }, l ? (l._callback = c, l._opts = h, l._params.q = p, a && (l._params.env = a)) : l = new e.YQLRequest(p, {
                    on: {
                        success: c
                    },
                    allowCache: !1
                }, a ? {
                    env: a
                } : null, h), l.send()
            }
            var n = this, o = {
                type: "yql"
            }, a, f, l;
            return n.get(u) || n.set(u, n._defaultYQLLocator), o.sendRequest = function(t) {
                a = t, f || (f=!0, e.use("yql", function() {
                    o.sendRequest = c, c(a)
                }))
            }, o
        },
        _defaultYQLLocator: function(t) {
            var n = t && t.query && t.query.results, i;
            return n && r.isObject(n) ? (i = e.Object.values(n) || [], n = i.length === 1 ? i[0] : i, r.isArray(n) || (n = [n])) : n = [], n
        },
        _getXHRUrl: function(e, t) {
            var n = this.get(s);
            return t.query !== t.request && (e += t.request), r.sub(e, {
                maxResults: n > 0 ? n: 1e3,
                query: encodeURIComponent(t.query)
            })
        },
        _jsonpFormatter: function(e, t, n) {
            var i = this.get(s), u = this.get(o);
            return u && (e += u(n)), r.sub(e, {
                callback: t,
                maxResults: i > 0 ? i: 1e3,
                query: encodeURIComponent(n)
            })
        }
    }), e.mix(n.ATTRS, {
        yqlEnv: {
            value: null
        },
        yqlProtocol: {
            value: "http"
        }
    }), e.mix(n.SOURCE_TYPES, {
        io: "_createIOSource",
        jsonp: "_createJSONPSource",
        object: "_beforeCreateObjectSource",
        select: "_createSelectSource",
        string: "_createStringSource",
        yql: "_createYQLSource"
    }, !0)
}, "3.11.0", {
    optional: ["io-base", "json-parse", "jsonp", "yql"],
    requires: ["autocomplete-base"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("lang/autocomplete-list_en", function(e) {
    e.Intl.add("autocomplete-list", "en", {
        item_selected: "{item} selected.",
        items_available: "Suggestions are available. Use up and down arrows to select."
    })
}, "3.11.0");
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("selector-css2", function(e, t) {
    var n = "parentNode", r = "tagName", i = "attributes", s = "combinator", o = "pseudos", u = e.Selector, a = {
        _reRegExpTokens: /([\^\$\?\[\]\*\+\-\.\(\)\|\\])/,
        SORT_RESULTS: !0,
        _isXML: function() {
            var t = e.config.doc.createElement("div").tagName !== "DIV";
            return t
        }(),
        shorthand: {
            "\\#(-?[_a-z0-9]+[-\\w\\uE000]*)": "[id=$1]",
            "\\.(-?[_a-z]+[-\\w\\uE000]*)": "[className~=$1]"
        },
        operators: {
            "": function(t, n) {
                return e.DOM.getAttribute(t, n) !== ""
            },
            "~=": "(?:^|\\s+){val}(?:\\s+|$)",
            "|=": "^{val}-?"
        },
        pseudos: {
            "first-child": function(t) {
                return e.DOM._children(t[n])[0] === t
            }
        },
        _bruteQuery: function(t, n, r) {
            var i = [], s = [], o, a = u._tokenize(t), f = a[a.length - 1], l = e.DOM._getDoc(n), c, h, p, d, v;
            if (f) {
                h = f.id, p = f.className, d = f.tagName || "*";
                if (n.getElementsByTagName)
                    h && (n.all || n.nodeType === 9 || e.DOM.inDoc(n)) ? s = e.DOM.allById(h, n) : p ? s = n.getElementsByClassName(p) : s = n.getElementsByTagName(d);
                else {
                    o = [], c = n.firstChild, v = d === "*";
                    while (c) {
                        while (c)
                            c.tagName > "@" && (v || c.tagName === d) && s.push(c), o.push(c), c = c.firstChild;
                        while (o.length > 0&&!c)
                            c = o.pop().nextSibling
                    }
                }
                s.length && (i = u._filterNodes(s, a, r))
            }
            return i
        },
        _filterNodes: function(t, n, r) {
            var i = 0, s, o = n.length, a = o - 1, f = [], l = t[0], c = l, h = e.Selector.getters, p, d, v, m, g, y, b, w;
            for (i = 0; c = l = t[i++];) {
                a = o - 1, m = null;
                e:
                while (c && c.tagName) {
                    v = n[a], b = v.tests, s = b.length;
                    if (s&&!g)
                        while (w = b[--s]) {
                            p = w[1], h[w[0]] ? y = h[w[0]](c, w[0]) : (y = c[w[0]], w[0] === "tagName"&&!u._isXML && (y = y.toUpperCase()), typeof y != "string" && y !== undefined && y.toString ? y = y.toString() : y === undefined && c.getAttribute && (y = c.getAttribute(w[0], 2)));
                            if (p === "=" && y !== w[2] || typeof p != "string" && p.test&&!p.test(y) ||!p.test && typeof p == "function"&&!p(c, w[0], w[2])) {
                                if (c = c[m])
                                    while (c && (!c.tagName || v.tagName && v.tagName !== c.tagName))
                                        c = c[m];
                                        continue e
                            }
                        }
                    a--;
                    if (!!g ||!(d = v.combinator)) {
                        f.push(l);
                        if (r)
                            return f;
                        break
                    }
                    m = d.axis, c = c[m];
                    while (c&&!c.tagName)
                        c = c[m];
                    d.direct && (m = null)
                }
            }
            return l = c = null, f
        },
        combinators: {
            " ": {
                axis: "parentNode"
            },
            ">": {
                axis: "parentNode",
                direct: !0
            },
            "+": {
                axis: "previousSibling",
                direct: !0
            }
        },
        _parsers: [{
            name: i,
            re: /^\uE003(-?[a-z]+[\w\-]*)+([~\|\^\$\*!=]=?)?['"]?([^\uE004'"]*)['"]?\uE004/i,
            fn: function(t, n) {
                var r = t[2] || "", i = u.operators, s = t[3] ? t[3].replace(/\\/g, ""): "", o;
                if (t[1] === "id" && r === "=" || t[1] === "className" && e.config.doc.documentElement.getElementsByClassName && (r === "~=" || r === "="))
                    n.prefilter = t[1], t[3] = s, n[t[1]] = t[1] === "id" ? t[3] : s;
                r in i && (o = i[r], typeof o == "string" && (t[3] = s.replace(u._reRegExpTokens, "\\$1"), o = new RegExp(o.replace("{val}", t[3]))), t[2] = o);
                if (!n.last || n.prefilter !== t[1])
                    return t.slice(1)
            }
        }, {
            name: r,
            re: /^((?:-?[_a-z]+[\w-]*)|\*)/i,
            fn: function(e, t) {
                var n = e[1];
                u._isXML || (n = n.toUpperCase()), t.tagName = n;
                if (n !== "*" && (!t.last || t.prefilter))
                    return [r, "=", n];
                t.prefilter || (t.prefilter = "tagName")
            }
        }, {
            name: s,
            re: /^\s*([>+~]|\s)\s*/,
            fn: function(e, t) {}
        }, {
            name: o,
            re: /^:([\-\w]+)(?:\uE005['"]?([^\uE005]*)['"]?\uE006)*/i,
            fn: function(e, t) {
                var n = u[o][e[1]];
                return n ? (e[2] && (e[2] = e[2].replace(/\\/g, "")), [e[2], n]) : !1
            }
        }
        ],
        _getToken: function(e) {
            return {
                tagName: null,
                id: null,
                className: null,
                attributes: {},
                combinator: null,
                tests: []
            }
        },
        _tokenize: function(t) {
            t = t || "", t = u._parseSelector(e.Lang.trim(t));
            var n = u._getToken(), r = t, i = [], o=!1, a, f, l, c;
            e: do {
                o=!1;
                for (l = 0; c = u._parsers[l++];)
                    if (a = c.re.exec(t)) {
                        c.name !== s && (n.selector = t), t = t.replace(a[0], ""), t.length || (n.last=!0), u._attrFilters[a[1]] && (a[1] = u._attrFilters[a[1]]), f = c.fn(a, n);
                        if (f===!1) {
                            o=!1;
                            break e
                        }
                        f && n.tests.push(f);
                        if (!t.length || c.name === s)
                            i.push(n), n = u._getToken(n), c.name === s && (n.combinator = e.Selector.combinators[a[1]]);
                            o=!0
                    }
            }
            while (o && t.length);
            if (!o || t.length)
                i = [];
            return i
        },
        _replaceMarkers: function(e) {
            return e = e.replace(/\[/g, "\ue003"), e = e.replace(/\]/g, "\ue004"), e = e.replace(/\(/g, "\ue005"), e = e.replace(/\)/g, "\ue006"), e
        },
        _replaceShorthand: function(t) {
            var n = e.Selector.shorthand, r;
            for (r in n)
                n.hasOwnProperty(r) && (t = t.replace(new RegExp(r, "gi"), n[r]));
            return t
        },
        _parseSelector: function(t) {
            var n = e.Selector._replaceSelector(t), t = n.selector;
            return t = e.Selector._replaceShorthand(t), t = e.Selector._restore("attr", t, n.attrs), t = e.Selector._restore("pseudo", t, n.pseudos), t = e.Selector._replaceMarkers(t), t = e.Selector._restore("esc", t, n.esc), t
        },
        _attrFilters: {
            "class": "className",
            "for": "htmlFor"
        },
        getters: {
            href: function(t, n) {
                return e.DOM.getAttribute(t, n)
            },
            id: function(t, n) {
                return e.DOM.getId(t)
            }
        }
    };
    e.mix(e.Selector, a, !0), e.Selector.getters.src = e.Selector.getters.rel = e.Selector.getters.href, e.Selector.useNative && e.config.doc.querySelector && (e.Selector.shorthand["\\.(-?[_a-z]+[-\\w]*)"] = "[class~=$1]")
}, "3.11.0", {
    requires: ["selector-native"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("selector-css3", function(e, t) {
    e.Selector._reNth = /^(?:([\-]?\d*)(n){1}|(odd|even)$)*([\-+]?\d*)$/, e.Selector._getNth = function(t, n, r, i) {
        e.Selector._reNth.test(n);
        var s = parseInt(RegExp.$1, 10), o = RegExp.$2, u = RegExp.$3, a = parseInt(RegExp.$4, 10) || 0, f = [], l = e.DOM._children(t.parentNode, r), c;
        u ? (s = 2, c = "+", o = "n", a = u === "odd" ? 1 : 0) : isNaN(s) && (s = o ? 1 : 0);
        if (s === 0)
            return i && (a = l.length - a + 1), l[a - 1] === t?!0 : !1;
        s < 0 && (i=!!i, s = Math.abs(s));
        if (!i) {
            for (var h = a - 1, p = l.length; h < p; h += s)
                if (h >= 0 && l[h] === t)
                    return !0
        } else 
            for (var h = l.length - a, p = l.length; h >= 0; h -= s)
                if (h < p && l[h] === t)
                    return !0;
        return !1
    }, e.mix(e.Selector.pseudos, {
        root: function(e) {
            return e === e.ownerDocument.documentElement
        },
        "nth-child": function(t, n) {
            return e.Selector._getNth(t, n)
        },
        "nth-last-child": function(t, n) {
            return e.Selector._getNth(t, n, null, !0)
        },
        "nth-of-type": function(t, n) {
            return e.Selector._getNth(t, n, t.tagName)
        },
        "nth-last-of-type": function(t, n) {
            return e.Selector._getNth(t, n, t.tagName, !0)
        },
        "last-child": function(t) {
            var n = e.DOM._children(t.parentNode);
            return n[n.length - 1] === t
        },
        "first-of-type": function(t) {
            return e.DOM._children(t.parentNode, t.tagName)[0] === t
        },
        "last-of-type": function(t) {
            var n = e.DOM._children(t.parentNode, t.tagName);
            return n[n.length - 1] === t
        },
        "only-child": function(t) {
            var n = e.DOM._children(t.parentNode);
            return n.length === 1 && n[0] === t
        },
        "only-of-type": function(t) {
            var n = e.DOM._children(t.parentNode, t.tagName);
            return n.length === 1 && n[0] === t
        },
        empty: function(e) {
            return e.childNodes.length === 0
        },
        not: function(t, n) {
            return !e.Selector.test(t, n)
        },
        contains: function(e, t) {
            var n = e.innerText || e.textContent || "";
            return n.indexOf(t)>-1
        },
        checked: function(e) {
            return e.checked===!0 || e.selected===!0
        },
        enabled: function(e) {
            return e.disabled !== undefined&&!e.disabled
        },
        disabled: function(e) {
            return e.disabled
        }
    }), e.mix(e.Selector.operators, {
        "^=": "^{val}",
        "$=": "{val}$",
        "*=": "{val}"
    }), e.Selector.combinators["~"] = {
        axis: "previousSibling"
    }
}, "3.11.0", {
    requires: ["selector-native", "selector-css2"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("shim-plugin", function(e, t) {
    function n(e) {
        this.init(e)
    }
    n.CLASS_NAME = "yui-node-shim", n.TEMPLATE = '<iframe class="' + n.CLASS_NAME + '" frameborder="0" title="Node Stacking Shim"' + 'src="javascript:false" tabindex="-1" role="presentation"' + 'style="position:absolute; z-index:-1;"></iframe>', n.prototype = {
        init: function(e) {
            this._host = e.host, this.initEvents(), this.insert(), this.sync()
        },
        initEvents: function() {
            this._resizeHandle = this._host.on("resize", this.sync, this)
        },
        getShim: function() {
            return this._shim || (this._shim = e.Node.create(n.TEMPLATE, this._host.get("ownerDocument")))
        },
        insert: function() {
            var e = this._host;
            this._shim = e.insertBefore(this.getShim(), e.get("firstChild"))
        },
        sync: function() {
            var e = this._shim, t = this._host;
            e && e.setAttrs({
                width: t.getStyle("width"),
                height: t.getStyle("height")
            })
        },
        destroy: function() {
            var e = this._shim;
            e && e.remove(!0), this._resizeHandle.detach()
        }
    }, n.NAME = "Shim", n.NS = "shim", e.namespace("Plugin"), e.Plugin.Shim = n
}, "3.11.0", {
    requires: ["node-style", "node-pluginhost"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("autocomplete-list", function(e, t) {
    var n = e.Lang, r = e.Node, i = e.Array, s = e.UA.ie && e.UA.ie < 7, o = 9, u = "_CLASS_ITEM", a = "_CLASS_ITEM_ACTIVE", f = "_CLASS_ITEM_HOVER", l = "_SELECTOR_ITEM", c = "activeItem", h = "alwaysShowList", p = "circular", d = "hoveredItem", v = "id", m = "item", g = "list", y = "result", b = "results", w = "visible", E = "width", S = "select", x = e.Base.create("autocompleteList", e.Widget, [e.AutoCompleteBase, e.WidgetPosition, e.WidgetPositionAlign], {
        ARIA_TEMPLATE: "<div/>",
        ITEM_TEMPLATE: "<li/>",
        LIST_TEMPLATE: "<ul/>",
        UI_EVENTS: function() {
            var t = e.merge(e.Node.DOM_EVENTS);
            return delete t.valuechange, delete t.valueChange, t
        }(),
        initializer: function() {
            var t = this.get("inputNode");
            if (!t) {
                e.error("No inputNode specified.");
                return 
            }
            this._inputNode = t, this._listEvents = [], this.DEF_PARENT_NODE = t.get("parentNode"), this[u] = this.getClassName(m), this[a] = this.getClassName(m, "active"), this[f] = this.getClassName(m, "hover"), this[l] = "." + this[u], this.publish(S, {
                defaultFn: this._defSelectFn
            })
        },
        destructor: function() {
            while (this._listEvents.length)
                this._listEvents.pop().detach();
            this._ariaNode && this._ariaNode.remove().destroy(!0)
        },
        bindUI: function() {
            this._bindInput(), this._bindList()
        },
        renderUI: function() {
            var t = this._createAriaNode(), n = this.get("boundingBox"), r = this.get("contentBox"), i = this._inputNode, o = this._createListNode(), u = i.get("parentNode");
            i.addClass(this.getClassName("input")).setAttrs({
                "aria-autocomplete": g,
                "aria-expanded": !1,
                "aria-owns": o.get("id")
            }), u.append(t), s && n.plug(e.Plugin.Shim), this._ariaNode = t, this._boundingBox = n, this._contentBox = r, this._listNode = o, this._parentNode = u
        },
        syncUI: function() {
            this._syncResults(), this._syncVisibility()
        },
        hide: function() {
            return this.get(h) ? this : this.set(w, !1)
        },
        selectItem: function(e, t) {
            if (e) {
                if (!e.hasClass(this[u]))
                    return this
            } else {
                e = this.get(c);
                if (!e)
                    return this
            }
            return this.fire(S, {
                itemNode: e,
                originEvent: t || null,
                result: e.getData(y)
            }), this
        },
        _activateNextItem: function() {
            var e = this.get(c), t;
            return e ? t = e.next(this[l]) || (this.get(p) ? null : e) : t = this._getFirstItemNode(), this.set(c, t), this
        },
        _activatePrevItem: function() {
            var e = this.get(c), t = e ? e.previous(this[l]): this.get(p) && this._getLastItemNode();
            return this.set(c, t || null), this
        },
        _add: function(t) {
            var r = [];
            return i.each(n.isArray(t) ? t : [t], function(e) {
                r.push(this._createItemNode(e).setData(y, e))
            }, this), r = e.all(r), this._listNode.append(r.toFrag()), r
        },
        _ariaSay: function(e, t) {
            var r = this.get("strings." + e);
            this._ariaNode.set("text", t ? n.sub(r, t) : r)
        },
        _bindInput: function() {
            var e = this._inputNode, t, n, r;
            this.get("align") === null && (r = this.get("tokenInput"), t = r && r.get("boundingBox") || e, this.set("align", {
                node: t,
                points: ["tl", "bl"]
            }), !this.get(E) && (n = t.get("offsetWidth")) && this.set(E, n)), this._listEvents = this._listEvents.concat([e.after("blur", this._afterListInputBlur, this), e.after("focus", this._afterListInputFocus, this)])
        },
        _bindList: function() {
            this._listEvents = this._listEvents.concat([e.one("doc").after("click", this._afterDocClick, this), e.one("win").after("windowresize", this._syncPosition, this), this.after({
                mouseover: this._afterMouseOver,
                mouseout: this._afterMouseOut,
                activeItemChange: this._afterActiveItemChange,
                alwaysShowListChange: this._afterAlwaysShowListChange,
                hoveredItemChange: this._afterHoveredItemChange,
                resultsChange: this._afterResultsChange,
                visibleChange: this._afterVisibleChange
            }), this._listNode.delegate("click", this._onItemClick, this[l], this)])
        },
        _clear: function() {
            this.set(c, null), this._set(d, null), this._listNode.get("children").remove(!0)
        },
        _createAriaNode: function() {
            var e = r.create(this.ARIA_TEMPLATE);
            return e.addClass(this.getClassName("aria")).setAttrs({
                "aria-live": "polite",
                role: "status"
            })
        },
        _createItemNode: function(t) {
            var n = r.create(this.ITEM_TEMPLATE);
            return n.addClass(this[u]).setAttrs({
                id: e.stamp(n),
                role: "option"
            }).setAttribute("data-text", t.text).append(t.display)
        },
        _createListNode: function() {
            var t = this.get("listNode") || r.create(this.LIST_TEMPLATE);
            return t.addClass(this.getClassName(g)).setAttrs({
                id: e.stamp(t),
                role: "listbox"
            }), this._set("listNode", t), this.get("contentBox").append(t), t
        },
        _getFirstItemNode: function() {
            return this._listNode.one(this[l])
        },
        _getLastItemNode: function() {
            return this._listNode.one(this[l] + ":last-child")
        },
        _syncPosition: function() {
            this._syncUIPosAlign(), this._syncShim()
        },
        _syncResults: function(e) {
            e || (e = this.get(b)), this._clear(), e.length && (this._add(e), this._ariaSay("items_available")), this._syncPosition(), this.get("activateFirstItem")&&!this.get(c) && this.set(c, this._getFirstItemNode())
        },
        _syncShim: s ? function() {
            var e = this._boundingBox.shim;
            e && e.sync()
        }
        : function() {},
        _syncVisibility: function(t) {
            this.get(h) && (t=!0, this.set(w, t)), typeof t == "undefined" && (t = this.get(w)), this._inputNode.set("aria-expanded", t), this._boundingBox.set("aria-hidden", !t), t ? this._syncPosition() : (this.set(c, null), this._set(d, null), this._boundingBox.get("offsetWidth")), e.UA.ie === 7 && e.one("body").addClass("yui3-ie7-sucks").removeClass("yui3-ie7-sucks")
        },
        _afterActiveItemChange: function(t) {
            var n = this._inputNode, r = t.newVal, i = t.prevVal, s;
            i && i._node && i.removeClass(this[a]), r ? (r.addClass(this[a]), n.set("aria-activedescendant", r.get(v))) : n.removeAttribute("aria-activedescendant"), this.get("scrollIntoView") && (s = r || n, (!s.inRegion(e.DOM.viewportRegion(), !0) ||!s.inRegion(this._contentBox, !0)) && s.scrollIntoView())
        },
        _afterAlwaysShowListChange: function(e) {
            this.set(w, e.newVal || this.get(b).length > 0)
        },
        _afterDocClick: function(e) {
            var t = this._boundingBox, n = e.target;
            n !== this._inputNode && n !== t && n.ancestor("#" + t.get("id"), !0) && this.hide()
        },
        _afterHoveredItemChange: function(e) {
            var t = e.newVal, n = e.prevVal;
            n && n.removeClass(this[f]), t && t.addClass(this[f])
        },
        _afterListInputBlur: function() {
            this._listInputFocused=!1, this.get(w)&&!this._mouseOverList && (this._lastInputKey !== o ||!this.get("tabSelect") ||!this.get(c)) && this.hide()
        },
        _afterListInputFocus: function() {
            this._listInputFocused=!0
        },
        _afterMouseOver: function(e) {
            var t = e.domEvent.target.ancestor(this[l], !0);
            this._mouseOverList=!0, t && this._set(d, t)
        },
        _afterMouseOut: function() {
            this._mouseOverList=!1, this._set(d, null)
        },
        _afterResultsChange: function(e) {
            this._syncResults(e.newVal
            ), this.get(h) || this.set(w, !!e.newVal.length)
        },
        _afterVisibleChange: function(e) {
            this._syncVisibility(!!e.newVal)
        },
        _onItemClick: function(e) {
            var t = e.currentTarget;
            this.set(c, t), this.selectItem(t, e)
        },
        _defSelectFn: function(e) {
            var t = e.result.text;
            this._inputNode.focus(), this._updateValue(t), this._ariaSay("item_selected", {
                item: t
            }), this.hide()
        }
    }, {
        ATTRS: {
            activateFirstItem: {
                value: !1
            },
            activeItem: {
                setter: e.one,
                value: null
            },
            alwaysShowList: {
                value: !1
            },
            circular: {
                value: !0
            },
            hoveredItem: {
                readOnly: !0,
                value: null
            },
            listNode: {
                writeOnce: "initOnly",
                value: null
            },
            scrollIntoView: {
                value: !1
            },
            strings: {
                valueFn: function() {
                    return e.Intl.get("autocomplete-list")
                }
            },
            tabSelect: {
                value: !0
            },
            visible: {
                value: !1
            }
        },
        CSS_PREFIX: e.ClassNameManager.getClassName("aclist")
    });
    e.AutoCompleteList = x, e.AutoComplete = x
}, "3.11.0", {
    lang: ["en", "es", "hu", "it"],
    requires: ["autocomplete-base", "event-resize", "node-screen", "selector-css3", "shim-plugin", "widget", "widget-position", "widget-position-align"],
    skinnable: !0
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("autocomplete-list-keys", function(e, t) {
    function u() {
        e.before(this._bindKeys, this, "bindUI"), this._initKeys()
    }
    var n = 40, r = 13, i = 27, s = 9, o = 38;
    u.prototype = {
        _initKeys: function() {
            var e = {}, t = {};
            e[n] = this._keyDown, t[r] = this._keyEnter, t[i] = this._keyEsc, t[s] = this._keyTab, t[o] = this._keyUp, this._keys = e, this._keysVisible = t
        },
        destructor: function() {
            this._unbindKeys()
        },
        _bindKeys: function() {
            this._keyEvents = this._inputNode.on("keydown", this._onInputKey, this)
        },
        _unbindKeys: function() {
            this._keyEvents && this._keyEvents.detach(), this._keyEvents = null
        },
        _keyDown: function() {
            this.get("visible") ? this._activateNextItem() : this.show()
        },
        _keyEnter: function(e) {
            var t = this.get("activeItem");
            if (!t)
                return !1;
            this.selectItem(t, e)
        },
        _keyEsc: function() {
            this.hide()
        },
        _keyTab: function(e) {
            var t;
            if (this.get("tabSelect")) {
                t = this.get("activeItem");
                if (t)
                    return this.selectItem(t, e), !0
            }
            return !1
        },
        _keyUp: function() {
            this._activatePrevItem()
        },
        _onInputKey: function(e) {
            var t, n = e.keyCode;
            this._lastInputKey = n, this.get("results").length && (t = this._keys[n], !t && this.get("visible") && (t = this._keysVisible[n]), t && t.call(this, e)!==!1 && e.preventDefault())
        }
    }, e.Base.mix(e.AutoCompleteList, [u])
}, "3.11.0", {
    requires: ["autocomplete-list", "base-build"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("autocomplete-plugin", function(e, t) {
    function r(e) {
        e.inputNode = e.host, !e.render && e.render!==!1 && (e.render=!0), r.superclass.constructor.apply(this, arguments)
    }
    var n = e.Plugin;
    e.extend(r, e.AutoCompleteList, {}, {
        NAME: "autocompleteListPlugin",
        NS: "ac",
        CSS_PREFIX: e.ClassNameManager.getClassName("aclist")
    }), n.AutoComplete = r, n.AutoCompleteList = r
}, "3.11.0", {
    requires: ["autocomplete-list", "node-pluginhost"]
});
YUI.add("bo-selecta-transjax", function(a) {
    a.transjax.add("bo-selecta", {
        me: 'me',
        loading: 'Loading...',
        uber_contact_list_view_profile: 'View Profile',
        uber_contact_list_friend_and_family: 'Friend &amp; Family',
        uber_contact_list_friend: 'Friend',
        uber_contact_list_family: 'Family',
        uber_contact_list_contact: 'Contact',
        uber_contact_list_removed: 'Removed',
        uber_contact_list_edit: 'Edit',
        uber_contact_list_default_text: 'screen name, real name, or email',
        uber_contact_list_max_results: 'Showing [results_shown] of [total_results] results. <a href="[url]">See all...</a>',
        uber_contact_list_no_realname: 'No real name given',
        bo_selecta_no_contacts_found: 'No contacts found.',
        bo_selecta_no_members_found: 'No members found.',
        bo_selecta_global_search_msg: 'Search through all Flickr members?',
        bo_selecta_global_search_msg_2: 'Search all Flickr members?',
        too_many_results: 'We found too many results to display here. Please type a few more characters.'
    })
}, "0.0.1", {
    requires: ["transjax-base"]
}); /*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("datasource-local", function(e, t) {
    var n = e.Lang, r = function() {
        r.superclass.constructor.apply(this, arguments)
    };
    e.mix(r, {
        NAME: "dataSourceLocal",
        ATTRS: {
            source: {
                value: null
            }
        },
        _tId: 0,
        transactions: {},
        issueCallback: function(e, t) {
            var n = e.on || e.callback, r = n && n.success, i = e.details[0];
            i.error = e.error || e.response.error, i.error && (t.fire("error", i), r = n && n.failure), r && r(i)
        }
    }), e.extend(r, e.Base, {
        initializer: function(e) {
            this._initEvents()
        },
        _initEvents: function() {
            this.publish("request", {
                defaultFn: e.bind("_defRequestFn", this),
                queuable: !0
            }), this.publish("data", {
                defaultFn: e.bind("_defDataFn", this),
                queuable: !0
            }), this.publish("response", {
                defaultFn: e.bind("_defResponseFn", this),
                queuable: !0
            })
        },
        _defRequestFn: function(e) {
            var t = this.get("source"), r = e.details[0];
            n.isUndefined(t) && (r.error = new Error("Local source undefined")), r.data = t, this.fire("data", r)
        },
        _defDataFn: function(e) {
            var t = e.data, r = e.meta, i = {
                results: n.isArray(t) ? t: [t],
                meta: r ? r: {}
            }, s = e.details[0];
            s.response = i, this.fire("response", s)
        },
        _defResponseFn: function(e) {
            r.issueCallback(e, this)
        },
        sendRequest: function(e) {
            var t = r._tId++, n;
            return e = e || {}, n = e.on || e.callback, this.fire("request", {
                tId: t,
                request: e.request,
                on: n,
                callback: n,
                cfg: e.cfg || {}
            }), t
        }
    }), e.namespace("DataSource").Local = r
}, "3.11.0", {
    requires: ["base"]
});
YUI.add("bo-selecta-global-search-datasource", function(a) {
    a.BoSelectaGlobalSearchDataSource = function(b) {
        a.BoSelectaGlobalSearchDataSource.superclass.constructor.apply(this, arguments);
        this.request = null;
        this.selector = b.selector
    };
    a.extend(a.BoSelectaGlobalSearchDataSource, a.DataSource.Local, {
        _defRequestFn: function(d) {
            var c, b = this;
            if (this.request && this.request.readyState < 4) {
                this.request.abort()
            }
            this.selector.loadingBox.setStyle("display", "block");
            c = decodeURIComponent(d.request);
            this.request = a.flickrAPI.callMethod("flickr.people.search", {
                username: c
            }, {
                success: function(j, p) {
                    var q = j.data, i = j.params;
                    if (!b.selector.searchingGlobally) {
                        b.selector.loadingBox.setStyle("display", "none");
                        return 
                    }
                    var l = [];
                    var r = q.people.person;
                    var k, h;
                    for (var g = 0, m = r.length; g < m; g++) {
                        h = r[g];
                        k = {
                            n: h.nsid,
                            e: h.email,
                            u: h.username,
                            r: h.realname,
                            a: h.path_alias,
                            d: h.friend,
                            y: h.family,
                            s: h.iconserver,
                            f: h.iconfarm
                        };
                        k[0] = h.name;
                        k.i = (k.f && k.f !== 0 && k.s && k.s !== 0) ? "http://farm" + k.f + ".static" + (a.config.flickr.dev ? "-dev" : "") + ".flickr.com/" + k.s + "/buddyicons/" + k.n + ".jpg" : "http://www.flickr.com/images/buddyicon.gif";
                        l.push(k)
                    }
                    if (l.length === 0) {
                        if (!b.selector.hideNoContactMessage) {
                            l.push({
                                message: a.transjax.get("bo-selecta", "bo_selecta_no_members_found"),
                                disableActive: true
                            })
                        }
                        if (b.selector.zeroResultsMessages.length > 0) {
                            var f, o, e;
                            for (var g = 0, m = b.selector.zeroResultsMessages.length; g < m; g++) {
                                f = b.selector.zeroResultsMessages[g];
                                if (f.validator(b.selector)) {
                                    l.push(f)
                                }
                            }
                        }
                    }
                    b.selector.loadingBox.setStyle("display", "none");
                    b.fire("data", a.mix({
                        data: l
                    }, d))
                },
                failure: function(g, f) {
                    var e = g.data, i = g.params, h = [];
                    h.push({
                        message: a.transjax.get("bo-selecta", "too_many_results"),
                        totalResultsMessage: true
                    });
                    b.selector.loadingBox.setStyle("display", "none");
                    b.fire("data", a.mix({
                        data: h
                    }, d))
                }
            });
            return d.tId
        }
    })
}, "0.0.1", {
    requires: ["bo-selecta-transjax", "datasource-local", "gallery-flickr-api"]
}); /*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("datasource-function", function(e, t) {
    var n = e.Lang, r = function() {
        r.superclass.constructor.apply(this, arguments)
    };
    e.mix(r, {
        NAME: "dataSourceFunction",
        ATTRS: {
            source: {
                validator: n.isFunction
            }
        }
    }), e.extend(r, e.DataSource.Local, {
        _defRequestFn: function(e) {
            var t = this.get("source"), n = e.details[0];
            if (t)
                try {
                    n.data = t(e.request, this, e)
            } catch (r) {
                n.error = r
            } else 
                n.error = new Error("Function data failure");
            return this.fire("data", n), e.tId
        }
    }), e.DataSource.Function = r
}, "3.11.0", {
    requires: ["datasource-local"]
});
YUI.add("image-fader", function(b) {
    function a(c, d) {
        d = b.Lang.isNumber(d) ? d : 1;
        if (b.Lang.isString(c)) {
            c = b.all(c)
        }
        c.on("load", function(g) {
            var f;
            if (g.target.inDoc()) {
                f = new b.Anim({
                    node: g.target,
                    to: {
                        opacity: d
                    },
                    duration: 0.2
                });
                f.run()
            }
        })
    }
    b.imageFader = {
        attach: a
    }
}, "0.0.1", {
    requires: ["node", "anim"]
});
YUI.add("bo-selecta-3", function(b) {
    b.BoSelecta3 = function(d, c) {
        c = c || {};
        this.showUsername = c.showUsername || true;
        this.showSubtitle = c.showSubtitle || false;
        this.showIcon = c.showIcon || false;
        this.showRelationship = c.showRelationship || false;
        this.usePersonMenu = c.usePersonMenu || false;
        this.linkUsernameToPhotostream = c.linkUsernameToPhotostream || false;
        this.linkSubtitle = c.linkSubtitle || false;
        this.showEditRelationshipLink = c.showEditRelationshipLink || false;
        this.searchOnUsername = c.searchOnUsername || true;
        this.searchOnRealname = c.searchOnRealname || false;
        this.searchOnEmail = c.searchOnEmail || false;
        this.searchOnPathAlias = c.searchOnPathAlias || false;
        this.searchUsingSoundex = c.searchUsingSoundex || false;
        this.includeUser = c.includeUser || false;
        this.includeAddressBook = c.includeAddressBook || false;
        this.maxResultsDisplayed = c.maxResultsDisplayed || 5;
        this.globalMaxResultsDisplayed = c.globalMaxResultsDisplayed || 10;
        this.showTotalResults = c.showTotalResults || false;
        this.disableMouseHover = c.disableMouseHover || false;
        this.selectFirstItem = c.selectFirstItem || false;
        this.hideNoContactMessage = c.hideNoContactMessage || false;
        this.zeroResultsMessages = c.zeroResultsMessages || [];
        this.preloadText = c.preloadText || "";
        this.defaultText = c.defaultText || "";
        this.loadingText = c.loadingText || b.transjax.get("bo-selecta", "loading");
        this.focusOnFetch = c.focusOnFetch || false;
        this.fetchDataImmediately = c.fetchDataImmediately || false;
        this.allowNoContacts = c.allowNoContacts || false;
        this.allowFormSubmit = c.allowFormSubmit || false;
        this.respectCanTagFlag = c.respectCanTagFlag || false;
        this.apiURL = c.apiURL || b.config.flickr.people.api_url;
        this.enableGlobalSearch = c.enableGlobalSearch || false;
        this.defaultContainerWidth = c.defaultContainerWidth || 250;
        this.minListWidth = c.minListWidth || 228;
        this.maxListWidth = c.maxListWidth || 500;
        this.loadTransparently = c.loadTransparently || false;
        if (this.includeUser) {
            this.meString = b.transjax.get("bo-selecta", "me").toLowerCase()
        }
        this.contacts = null;
        this.skipList = {};
        this.shadowInit = false;
        this.fetchingData = false;
        this.haveTriedToFetchData = false;
        this.searchingGlobally = false;
        this._initHTML(d);
        this._initAutoComplete();
        if (this.fetchDataImmediately) {
            this.fetchData()
        }
    };
    b.augment(b.BoSelecta3, b.EventTarget);
    b.BoSelecta3.prototype._initHTML = function(e) {
        if (b.Lang.isString(e)) {
            e = b.one(e)
        }
        if (!e instanceof b.Node) {
            e = b.one("#BoSelecta_input")
        }
        this.inputField = e;
        this.container = this.inputField.get("parentNode");
        var c = this.container.all(".no-js-fallback");
        c.setStyle("display", "none");
        this.container.addClass("bo-selecta-3");
        this.container.addClass("yui-skin-sam");
        this.containerId = b.stamp(this.container);
        this.inputField.addClass("input");
        var d = this.container.all(".loading");
        if (d.size() > 0) {
            this.loadingBox = d.item(0);
            this.loadingBox.setStyle("display", this.fetchDataImmediately&&!this.loadTransparently ? "block" : "none")
        } else {
            this.loadingBox = b.Node.create('<div class="loading">' + this.loadingText + "</div>");
            this.loadingBox.setStyle("display", this.fetchDataImmediately&&!this.loadTransparently ? "block" : "none");
            this.container.append(this.loadingBox)
        }
        this.inputField.set("disabled", false);
        this.inputField.set("value", this.preloadText);
        this.inputField.addClass("grey");
        this.inputField.on("focus", this.clearDefaultText, this, true);
        this.inputField.on("blur", function(f) {
            if (this.inputField.get("value") === "") {
                this.restoreDefaultText()
            }
        }, this, true);
        if (!this.allowFormSubmit) {
            this.form = this.container.ancestor("form", true);
            if (this.form) {
                this.form.on("submit", function(f) {
                    f.halt()
                })
            }
        }
        this.maxInputWidth = parseInt(this.container.getStyle("width"), 10) || this.defaultContainerWidth;
        this.defaultInputWidth = parseInt(this.inputField.getStyle("width"), 10) || this.defaultContainerWidth
    };
    b.BoSelecta3.prototype._initAutoComplete = function() {
        var c = this;
        this.contactsCacheDataSource = new b.DataSource.Function({
            source: function(d) {
                return c._searchContacts(d)
            }
        });
        this.globalSearchDataSource = new b.BoSelectaGlobalSearchDataSource({
            selector: this
        });
        this.autoComp = new b.AutoComplete({
            inputNode: this.inputField,
            source: this.contactsCacheDataSource,
            resultTextLocator: function(d) {
                if (d && d.u) {
                    return b.BoSelecta3.sanitizeString(d.u)
                }
                return ""
            },
            resultFormatter: function(f, d, e) {
                return c._formatResult(f, d, e)
            },
            activateFirstItem: this.selectFirstItem,
            queryDelay: 0,
            render: true
        });
        this.autoComp.on("select", this._onItemSelectEvent, this, true);
        this.autoComp.on("query", this._handleQueryChange, this, true);
        this.autoComp.after("query", this._resizeListToFit, this, true);
        this.inputField.on("keydown", this._onInputFieldKeyDown, this, true);
        this.inputField.on("focus", function(f) {
            var d = c.getQuery();
            if (d) {
                c.autoComp.sendRequest(d)
            }
        });
        this.inputField.on("blur", function(d) {
            var f = c.autoComp.get("visible");
            if (f) {
                if (!c.resultsHideHandler) {
                    c.resultsHideHandle = b.one("document").on("click", function(g) {
                        if (g.target.ancestor(".yui3-aclist-item")) {
                            return 
                        }
                        b.fire("BoSelecta:widgetBlur");
                        b.fire("flickr-menus:hide");
                        c.resultsHideHandle.detach()
                    })
                }
            } else {
                b.fire("BoSelecta:widgetBlur")
            }
        });
        b.on("flickr-menus:hide", this.hideResults, this, true)
    };
    b.BoSelecta3.prototype._fetchData = function() {
        if (this.fetchingData) {
            return 
        }
        this.fetchingData = true;
        this.haveTriedToFetchData = true;
        this.fire("BoSelecta:dataFetchStart");
        if (!this.loadTransparently) {
            this.inputField.setStyle("display", "none");
            this.loadingBox.setStyle("display", "block")
        }
        var c = this.apiURL;
        var d = this;
        var e = {
            success: function(f, g, l) {
                if (g.responseText === "-1" || g.responseText === "") {
                    if (d.allowNoContacts) {
                        d.contacts = [];
                        if (!d.loadTransparently) {
                            d.restoreDefaultText();
                            d.loadingBox.setStyle("display", "none");
                            d.inputField.setStyle("display", "inline");
                            if (d.inputField.get("value") === "" || d.focusOnFetch) {
                                d.clear(true)
                            }
                        }
                        d.fire("BoSelecta:dataFetchComplete")
                    } else {
                        e.failure()
                    }
                    return 
                }
                var k = g.responseText.split("\u0002")[1];
                var p = "\u0003";
                var m = "\u0001";
                d.contacts = k.split(p);
                var i;
                for (var h = 0, j = d.contacts.length; h < j; h++) {
                    i = d.contacts[h].split(m);
                    if (i.length === 0) {
                        continue
                    }
                    d.contacts[h] = {};
                    d.contacts[h].n = i[0];
                    d.contacts[h].e = i[1];
                    d.contacts[h].u = i[2];
                    d.contacts[h].r = i[3];
                    d.contacts[h].s = i[4];
                    d.contacts[h].f = i[5];
                    d.contacts[h].a = i[6];
                    d.contacts[h].d = i[7];
                    d.contacts[h].y = i[8];
                    d.contacts[h].c = i[9];
                    d.contacts[h].us = b.StringFilters.soundex(d.contacts[h].u);
                    d.contacts[h].rs = b.StringFilters.soundex(d.contacts[h].r);
                    d.contacts[h][0] = d.contacts[h].u
                }
                if (!d.loadTransparently) {
                    d.restoreDefaultText();
                    d.loadingBox.setStyle("display", "none");
                    d.inputField.setStyle("display", "inline");
                    if (d.focusOnFetch) {
                        d.clear(true)
                    }
                }
                d.fetchingData = false;
                d.fire("BoSelecta:dataFetchComplete")
            },
            failure: function(i, h, g) {
                d.autoComp.destroy();
                if (d.form && d.form.get("tagName").toLowerCase() === "form") {
                    b.Event.purgeElement(d.form)
                }
                var f = d.inputField.get("parentNode");
                f.all(".no-js-fallback").setStyle("display", "");
                d._destroy();
                d.fetchingData = false;
                d.fire("BoSelecta:dataError")
            }
        };
        if (!c) {
            e.failure();
            return 
        }
        b.io(c, {
            method: "GET",
            on: e
        })
    };
    b.BoSelecta3.prototype._destroy = function(d) {
        var c = this.inputField.get("parentNode");
        b.Event.purgeElement(this.inputField);
        this.inputField.removeClass("grey");
        this.inputField.removeClass("yui-ac-input");
        this.inputField.set("value", "");
        this.inputField.setStyle("display", "");
        this.inputField.removeClass("input");
        if (d) {
            this.inputField.remove()
        }
        this.loadingBox.remove();
        c.removeClass("yui-skin-sam");
        c.removeClass("bo-selecta-3")
    };
    b.BoSelecta3.prototype._searchContacts = function(g) {
        var v, e = [], o, j = this;
        if (g === "" || (g && g === this.defaultText)) {
            return e
        }
        if (g && typeof g === "string" && this.contacts) {
            o = g;
            g = this._escapeForRegEx(g);
            g = b.StringFilters.substitute_equivalent_chars(g);
            var d = new RegExp(g, "i"), u = b.StringFilters.soundex(o), s, k;
            for (var q = 0, t = this.contacts.length; q < t; q++) {
                k = false;
                s = this.contacts[q];
                s.matched = null;
                if (s.n && (!this.includeUser || o === this.meString) && s.n === b.config.flickr.user.nsid) {
                    if (o === this.meString&&!this.skipList[s.n]) {
                        v = s
                    }
                    continue
                }
                if (this.skipList && (this.skipList[s.n] || this.skipList[s.e])) {
                    continue
                }
                if (this.searchOnUsername && s.u && s.u.search(d)!==-1) {
                    k = true;
                    s.matched = "username";
                    s.matchedPos = s.u.search(d)
                }
                if (!k && this.searchOnRealname && s.r && s.r.search(d)!==-1) {
                    k = true;
                    s.matched = "realname";
                    s.matchedPos = s.r.search(d)
                }
                if (!k && this.searchOnEmail && s.e && g !== "@" && s.e.search(d)!==-1) {
                    k = true;
                    s.matched = "email";
                    s.matchedPos = s.e.search(d)
                }
                if (!k && this.searchOnPathAlias && s.a && s.a.search(d)!==-1) {
                    k = true;
                    s.matched = "path_alias";
                    s.matchedPos = s.a.search(d)
                }
                if (this.searchUsingSoundex) {
                    if (!k && this.searchOnUsername && s.us && b.StringFilters.isEqualSoundex(s.us, u)) {
                        k = true;
                        s.matched = "username-soundex";
                        s.matchedPos = (s.us === u) ? 0 : b.StringFilters.isEqualSoundex(s.us, u)
                    }
                    if (!k && this.searchOnRealname && s.rs && b.StringFilters.isEqualSoundex(s.rs, u)) {
                        k = true;
                        s.matched = "realname-soundex";
                        s.matchedPos = (s.rs === u) ? 0 : b.StringFilters.isEqualSoundex(s.rs, u)
                    }
                }
                if (k) {
                    e.push(s)
                }
            }
        }
        if (e.length === 0&&!(o === this.meString && v)) {
            if (!this.hideNoContactMessage) {
                e.push({
                    message: b.transjax.get("bo-selecta", "bo_selecta_no_contacts_found"),
                    notFoundLink: true,
                    disableActive: true
                });
                if (this.enableGlobalSearch) {
                    e.push({
                        globalSearchLink: true,
                        query: o,
                        message: b.transjax.get("bo-selecta", "bo_selecta_global_search_msg")
                    })
                }
            }
            if (this.zeroResultsMessages.length > 0) {
                var h, p, c;
                for (var q = 0, t = this.zeroResultsMessages.length; q < t; q++) {
                    h = this.zeroResultsMessages[q];
                    if (h.validator(this)) {
                        e.push(h)
                    }
                }
            }
        } else {
            for (var q = 0, t = e.length; q < t; q++) {
                if (!e[q].n&&!e[q].u) {
                    for (var r = 0, t = e.length; r < t; r++) {
                        if (q !== r && e[q] && e[r] && e[q].e === e[r].e) {
                            delete e[q];
                            break
                        }
                    }
                }
            }
            for (var q = 0, t = e.length; q < t;) {
                if (!e[q]) {
                    e.splice(q, 1);
                    t--
                } else {
                    q++
                }
            }
            var i = {
                username: 0,
                realname: 1,
                email: 2,
                path_alias: 3,
                "username-soundex": 4,
                "realname-soundex": 5
            };
            e.sort(function(x, m) {
                var w = (x.n && x.u) ? "contact": "addressBookEntry";
                var y = (m.n && m.u) ? "contact": "addressBookEntry";
                if (w !== y) {
                    return (w === "contact")?-1 : 1
                }
                if (x.matchedPos === m.matchedPos) {
                    if (i[x.matched] === i[m.matched]) {
                        var z = x.u.charCodeAt(0), n = m.u.charCodeAt(0);
                        if (z === n) {
                            return 0
                        } else {
                            if (z < n) {
                                return - 1
                            } else {
                                return 1
                            }
                        }
                    } else {
                        if (i[x.matched] < i[m.matched]) {
                            return - 1
                        } else {
                            return 1
                        }
                    }
                } else {
                    if (x.matchedPos < m.matchedPos) {
                        return - 1
                    } else {
                        return 1
                    }
                }
            });
            if (o === this.meString && v) {
                v.matched = "username";
                e.unshift(v)
            }
            var l = e.length;
            e = e.splice(0, this.maxResultsDisplayed);
            if (this.showTotalResults) {
                var f = "/search/people/?q=%s&m=names&see=all";
                e.push({
                    message: b.transjax.get("bo-selecta", "uber_contact_list_max_results").replace(/\[results_shown\]/, Math.min(e.length, this.maxResultsDisplayed)).replace(/\[total_results\]/, l).replace(/\[url\]/, f),
                    totalResultsMessage: true
                })
            }
        }
        if (this.showEditRelationshipLink&&!this.relationshipLinksInitialized) {
            b.use("contact-changer", function(m) {
                j.container.delegate("click", function(n) {
                    m.contact_changer.show(n.target.getAttribute("data-nsid"));
                    n.preventDefault()
                }, ".edit-rel a");
                j.relationshipLinksInitialized = true
            })
        }
        return e
    };
    b.BoSelecta3.prototype._formatResult = function(h, q, i) {
        var j = this, s, l = [], v, e, d, m, c=!h ? null : new RegExp(b.StringFilters.substitute_equivalent_chars(j._escapeForRegEx(h)), "gi"), k, g, o, u, r, f;
        if (!h) {
            h = ""
        }
        for (var p = 0, t = q.length; p < t; p++) {
            s = q[p].raw;
            if (!s) {
                continue
            }
            if (s.globalSearchLink) {
                v = '<p class="bs-global-search-link bs-message bs-onclick"><span>' + s.message + "</span></p>";
                l.push(v);
                continue
            }
            if (s.totalResultsMessage) {
                v = '<p class="bs-message bs-total-results bs-disable-active"><span>' + s.message + "</span></p>";
                l.push(v);
                continue
            }
            if (s.message) {
                v = '<p class="bs-message' + (s.notFoundLink ? "" : " bs-onclick") + (s.disableActive ? " bs-disable-active" : "") + (s.className ? " " + s.className : "") + '">' + s.message + "</p>";
                l.push(v);
                continue
            }
            m = (j.respectCanTagFlag && s.c === "0") ? ' class="disabled"' : "";
            if (s.e&&!s.u&&!s.n) {
                if (!s.i) {
                    s.i = "/images/icon_unread.gif"
                }
                e = j.showIcon ? '<img src="' + s.i + '" width="16" height="16" class="BuddyIconX bs-email-icon">' : "";
                d = '<p email="' + s.e + '"' + m + '><span class="name name-email">' + e + '<strong class="username">' + a(j._sanitizeString(s.e), c) + '</strong></span><span class="relationship"></span></p>';
                l.push(d);
                continue
            }
            k = (s.a) ? s.a : s.n;
            g = ' / <a href="/people/' + k + '/">' + b.transjax.get("bo-selecta", "uber_contact_list_view_profile") + "</a>";
            if (!s.i) {
                s.i = (s.f && s.f !== 0 && s.s && s.s !== 0) ? "http://farm" + s.f + ".static" + (b.config.flickr.dev ? "-dev" : "") + ".flickr.com/" + s.s + "/buddyicons/" + s.n + ".jpg" : "http://www.flickr.com/images/buddyicon.gif"
            }
            if (F.config.flickr.is_secure) {
                s.i = s.i.replace("http:", "https:")
            }
            o = "";
            if (j.showRelationship) {
                if (s.d === "1" && s.y === "1") {
                    o += b.transjax.get("bo-selecta", "uber_contact_list_friend_and_family")
                } else {
                    if (s.d === "1") {
                        o += b.transjax.get("bo-selecta", "uber_contact_list_friend")
                    } else {
                        if (s.y === "1") {
                            o += b.transjax.get("bo-selecta", "uber_contact_list_family")
                        } else {
                            if (s.removed === "1") {
                                o += '<em class="contact-removed">' + b.transjax.get("bo-selecta", "uber_contact_list_removed") + "</em>"
                            } else {
                                if (j.searchingGlobally) {
                                    o += ""
                                } else {
                                    o += b.transjax.get("bo-selecta", "uber_contact_list_contact")
                                }
                            }
                        }
                    }
                }
                if (j.showEditRelationshipLink&&!j.searchingGlobally) {
                    o += ' <span class="edit-rel">(<a href="/people/' + k + '/relationship/" data-nsid="' + s.n + '">' + b.transjax.get("bo-selecta", "uber_contact_list_edit") + "</a>)</span>"
                }
            }
            e = j.showIcon ? '<img src="' + s.i + '" width="24" height="24" class="BuddyIconX fade-in" nsid="' + s.n + '">' : "";
            e = j.linkUsernameToPhotostream ? '<a href="/photos/' + k + '/">' + e + "</a>" : e;
            u = j.showUsername ? a(j._sanitizeString(s.u), c) : "";
            u = j.linkUsernameToPhotostream ? '<strong class="username"><a href="/photos/' + k + '/">' + u + "</a></strong>" : '<strong class="username">' + u + "</strong>";
            r = "";
            if (j.showSubtitle) {
                if (!s.matched || s.matched === "realname" || s.matched === "realname-soundex") {
                    r = s.r ? a(j._sanitizeString(s.r), c) : b.transjax.get("bo-selecta", "uber_contact_list_no_realname")
                } else {
                    if (s.matched === "username" || s.matched === "username-soundex") {
                        r = s.r ? j._sanitizeString(s.r) : b.transjax.get("bo-selecta", "uber_contact_list_no_realname")
                    } else {
                        if (s.matched === "email") {
                            r = a(j._sanitizeString(s.e), c)
                        } else {
                            if (s.matched === "path_alias" && s.a) {
                                r = "flickr.com/people/" + a(j._sanitizeString(s.a), c) + "/"
                            }
                        }
                    }
                }
                r = j.linkSubtitle ? '<a title="' + b.transjax.get("bo-selecta", "uber_contact_list_view_profile") + '" href="/people/' + k + '/" class="realname">' + r + "</a>" : '<span class="realname">' + r + "</span>"
            }
            f = b.Node.create('<p nsid="' + s.n + '"' + m + '><span class="name">' + e + u + r + '</span><span class="relationship">' + o + "</span></p>");
            b.imageFader.attach(f.all("img"));
            l.push(f)
        }
        return l
    };
    function a(g, e) {
        if (!g) {
            return g
        }
        var c, d, f;
        g = b.StringFilters.unescape_entities(g, true);
        d = e ? g.search(e) : - 1;
        if (d===-1) {
            return b.BoSelecta3.sanitizeString(g)
        }
        c = g.match(e);
        f = d + c[0].length;
        return b.StringFilters.escape_entities(g.substring(0, d), true) + '<span class="term-highlight">' + b.StringFilters.escape_entities(g.substring(d, f), true) + "</span>" + b.StringFilters.escape_entities(g.substring(f, g.length), true)
    }
    b.BoSelecta3.prototype._resizeListToFit = function(j) {
        var h, g, d = this.minListWidth, f = this.maxListWidth, i, k, c;
        if (j.type === "autocompleteList:query" || (j.type === "autocompleteList:visibleChange" && j.newVal === true)) {
            h = j.currentTarget;
            g = h.get("listNode");
            g.all("li p").each(function(e) {
                i = e.one(".name");
                k = e.one(".relationship");
                if (i && k) {
                    c = i.get("offsetWidth") + k.get("offsetWidth") + 12;
                    if (c > d) {
                        d = c
                    }
                }
            });
            if (!this.baseListWidth) {
                this.baseListWidth = h.get("width")
            }
            if (d > f) {
                d = f
            }
            if (d < this.baseListWidth) {
                d = this.baseListWidth
            }
            h.set("width", d)
        }
    };
    b.BoSelecta3.prototype._attachIconEvents = function(e) {
        if (!personmenu_process_img || typeof personmenu_process_img !== "function") {
            return 
        }
        var g = e.all("img"), d;
        for (var f = 0, c = g.size(); f < c; f++) {
            d = b.Node.getDOMNode(g.item(f));
            if (g.item(f).hasClass("BuddyIconX")) {
                d.nsid = g.item(f).getAttribute("nsid");
                personmenu_process_img(d)
            }
        }
    };
    b.BoSelecta3.prototype._useGlobalSearch = function(d) {
        var c = this;
        this.searchingGlobally = true;
        if (d) {
            this.inputField.set("value", decodeURIComponent(d))
        }
        this.autoComp.set("source", this.globalSearchDataSource);
        this.autoComp.set("minQueryLength", 2);
        this.autoComp.set("maxResultsDisplayed", this.globalMaxResultsDisplayed);
        this.autoComp.set("queryDelay", 600);
        this.loadingBox.addClass("loading-global").removeClass("loading");
        this.loadingBox.set("innerHTML", "");
        this.globalSearchRemover = this.inputField.on("keyup", function(f) {
            if (c.inputField.get("value") === "") {
                c.globalSearchRemover.detach();
                c._useContactsCache()
            }
        });
        d = this.getQuery();
        this.autoComp.sendRequest(d)
    };
    b.BoSelecta3.prototype._useContactsCache = function(c) {
        this.searchingGlobally = false;
        if (c) {
            this.inputField.set("value", decodeURIComponent(c))
        }
        this.autoComp.set("source", this.contactsCacheDataSource);
        this.autoComp.set("minQueryLength", 1);
        this.autoComp.set("maxResultsDisplayed", 0);
        this.autoComp.set("queryDelay", 0)
    };
    b.BoSelecta3.prototype._onItemSelectEvent = function(f, d) {
        var c = f.result.raw;
        if (c.message) {
            if (c.globalSearchLink) {
                this._useGlobalSearch(c.query)
            } else {
                if (c.onclick && typeof c.onclick === "function") {
                    c.onclick(f, this)
                } else {
                    this.clear()
                }
            }
            f.preventDefault();
            return 
        }
        this.fire("BoSelecta:resultSelect", c, c.query);
        if (this.searchingGlobally) {
            this._useContactsCache()
        }
    };
    b.BoSelecta3.prototype._handleQueryChange = function(c) {
        this.fire("BoSelecta:queryChange", c.query)
    };
    b.BoSelecta3.prototype._onInputFieldKeyDown = function(d) {
        if (d.keyCode === 13 && this.autoComp) {
            var f = false;
            if (this.autoComp._oCurItem) {
                var c = b.one(this.autoComp._oCurItem).all("p");
                if (c.size() > 0) {
                    f = (c.item(0).hasClass("bs-message")&&!c.item(0).hasClass("bs-onclick"))
                }
            }
            if (!this.autoComp._oCurItem || f) {
                this.fire("BoSelecta:inputEnter")
            }
        }
    };
    b.BoSelecta3.prototype._sanitizeString = function(c) {
        c = c.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        return c
    };
    b.BoSelecta3.prototype._unsanitizeString = function(c) {
        c = c.replace(/\&lt;/g, "<").replace(/\&gt;/g, ">");
        return c
    };
    b.BoSelecta3.prototype._escapeForRegEx = function(c) {
        c = decodeURIComponent(c);
        c = c.replace(/\\/g, "\\\\").replace(/\./g, "\\.").replace(/\*/g, "\\*").replace(/\$/g, "\\$").replace(/\^/g, "\\^").replace(/\?/g, "\\?");
        c = c.replace(/\|/g, "\\|").replace(/\+/g, "\\+").replace(/\(/g, "\\(").replace(/\)/g, "\\)").replace(/\[/g, "\\[").replace(/\]/g, "\\]");
        return c
    };
    b.BoSelecta3.prototype.fetchData = function() {
        if (!this.contacts&&!this.haveTriedToFetchData) {
            this._fetchData()
        }
    };
    b.BoSelecta3.prototype.refreshData = function() {
        this._fetchData()
    };
    b.BoSelecta3.prototype.hideResults = function() {
        if (this.autoComp && this.autoComp.hide) {
            this.autoComp.hide()
        }
        if (this.resultsHideHandle && this.resultsHideHandle.detach) {
            this.resultsHideHandle.detach()
        }
    };
    b.BoSelecta3.prototype.clear = function(c) {
        this.inputField.removeClass("grey");
        this.inputField.set("value", "");
        if (c) {
            try {
                this.inputField.focus()
            } catch (d) {}
        }
        this.hideResults()
    };
    b.BoSelecta3.prototype.clearDefaultText = function(c) {
        if (this.inputField.hasClass("grey")) {
            this.clear(c)
        }
    };
    b.BoSelecta3.prototype.restoreDefaultText = function() {
        this.clear();
        this.inputField.addClass("grey");
        this.inputField.set("value", this.defaultText);
        this.inputField.blur()
    };
    b.BoSelecta3.prototype.getQuery = function() {
        if (this.inputField.hasClass("grey")) {
            return ""
        }
        return this.inputField.get("value")
    };
    b.BoSelecta3.prototype.refreshResults = function() {
        if (this.autoComp && this.getQuery()) {
            this.autoComp.sendRequest(this.getQuery())
        }
    };
    b.BoSelecta3.prototype.skipContact = function(c) {
        this.skipList[(c.n ? c.n : c.e)] = true
    };
    b.BoSelecta3.prototype.unskipContact = function(c) {
        this.skipList[(c.n ? c.n : c.e)] = false
    };
    b.BoSelecta3.prototype.unskipAll = function() {
        this.skipList = {}
    };
    b.BoSelecta3.prototype.showInFieldLoading = function() {
        this.loadingBox.addClass("loading-global");
        this.loadingBox.removeClass("loading");
        this.loadingBox.setStyle("display", "block")
    };
    b.BoSelecta3.prototype.hideInFieldLoading = function() {
        this.loadingBox.setStyle("display", "")
    };
    b.BoSelecta3.prototype.destroy = function() {
        this._destroy(true)
    };
    b.BoSelecta3.sanitizeString = b.BoSelecta3.prototype._sanitizeString;
    b.BoSelecta3.unsanitizeString = b.BoSelecta3.prototype._unsanitizeString
}, "0.0.1", {
    requires: F.config.modules["bo-selecta-3"].requires || [],
    optional: F.config.modules["bo-selecta-3"].optional || []
});
YUI.add("nav-selecta-transjax", function(a) {
    a.transjax.add("nav-selecta", {
        search_on_flickr: "Search <strong>Everyone's Uploads</strong>",
        search_more: "More search types...</strong>",
        jump_to_the_search_field: "Jump to the search field",
        search_photostream: "<strong>Your Photostream</strong>",
        search_users_photostream: "<strong>%s Photostream</strong>",
        search_favorites: "<strong>Your Favorites</strong>",
        search_users_favorites: "<strong>%s Favorites</strong>",
        search_contacts: "<strong>Your Contacts' Photos</strong>",
        search_people: "<strong>Flickr Members</strong>",
        search_tags: "<strong>Tags</strong>",
        search_groups: "<strong>Groups</strong>",
        search_group: "<strong>This Group</strong>",
        search_location: "<strong>Places</strong>",
        search_apps: "<strong>Apps</strong>",
        search_help: "<strong>The Help Forum</strong>",
        search_advanced: "<strong>Advanced Search</strong>",
        section_your_photostream: 'Your Photostream',
        section_your_sets: 'Your Sets',
        section_your_collections: 'Your Collections',
        section_your_galleries: 'Your Galleries',
        section_your_archives: 'Your Archives',
        section_your_tags: 'Your Tags',
        section_your_map: 'Your Map',
        section_your_favorites: 'Your Favorites',
        section_your_stats: 'Your Stats',
        section_your_apps: 'Your Apps',
        section_recent_activity: 'Recent Activity',
        section_photos_of_you: 'Photos of You',
        section_upload_photos: 'Upload Photos',
        section_your_account: 'Your Account',
        section_your_profile: 'Your Profile',
        section_flickrmail: 'FlickrMail',
        section_organize_and_create: 'Organize & Create',
        section_prints_and_photo_products: 'Prints &amp; photo products',
        section_contacts: "Contacts' Recent Uploads",
        section_photos_of_your_contacts: 'Photos of Your Contacts',
        section_contact_list: 'Contact List',
        section_find_your_friends: 'Find Your Friends',
        section_invite_your_friends: 'Invite your Friends',
        section_invite_history: 'Invite History',
        section_guest_pass_history: 'Guest Pass History',
        section_your_groups: 'Your Groups',
        section_recent_changes_in_your_groups: 'Recent Changes in Your Groups',
        section_create_a_new_group: 'Create a New Group',
        section_the_tour: 'The Tour',
        section_explore: 'Explore',
        section_last_7_days_interesting: 'Last 7 Days Interesting',
        section_popular_tags: 'Popular Tags',
        section_calendar: 'Calendar',
        section_most_recent_uploads_to_flickr: 'Most Recent Uploads to Flickr',
        section_video_on_flickr: 'Video on Flickr',
        section_galleries: 'Galleries',
        section_explore_analog: 'Explore Analog',
        section_flickr_clock: 'Flickr Clock',
        section_world_map: 'World Map',
        section_places: '',
        section_the_commons: 'The Commons',
        section_creative_commons: 'Creative Commons',
        section_flickrblog: 'FlickrBlog',
        section_getty_images: "Getty Images",
        section_code_flickr: "code.flickr",
        section_the_app_garden: 'The App Garden',
        section_camera_finder: 'Camera Finder',
        section_developer_guidelines: 'Developer Guide',
        section_api_docs: 'API Documentation',
        section_feeds: 'Feeds',
        section_help: 'Help',
        section_community_guidelines: 'Community Guidelines',
        section_the_help_forum: 'The Help Forum',
        section_faq: 'FAQ',
        section_tools: 'Tools',
        section_sitemap: 'Sitemap',
        section_about_flickr: 'About Flickr',
        section_jobs: 'Jobs at Flickr',
        section_terms_of_service: 'Terms of Service',
        section_terms_of_use: 'Terms of Use',
        section_your_privacy: 'Your Privacy',
        section_copyright_ip_policy: 'Copyright/IP Policy',
        section_report_abuse: 'Report Abuse',
        section_mobile: 'Mobile',
        section_photosession: 'Photo Session',
        section_alias_graphs: 'Graphs',
        section_alias_referrers: 'Referrers',
        section_alias_views: 'Views',
        section_alias_settings: 'Settings',
        section_alias_preferences: 'Preferences',
        section_alias_batch: 'Batch',
        section_alias_prints: 'Prints',
        section_alias_calendar: 'Calendar',
        section_alias_book: 'Book',
        section_alias_film: 'Film',
        section_alias_license: 'License',
        section_alias_buy: 'Buy',
        section_alias_sell: 'Sell',
        section_alias_developer: 'Developer',
        section_alias_programming: 'Programming',
        section_alias_code: 'Code',
        section_alias_documentation: 'Documentation',
        section_alias_career: 'Career',
        section_alias_work: 'Work',
        section_alias_staff: 'Staff',
        section_alias_upload: 'Upload',
        section_alias_plugin: 'Plug-in',
        section_alias_: '',
        section_alias_: '',
        section_alias_: '',
        section_alias_: '',
        section_alias_: '',
        section_alias_: '',
        section_alias_: ''
    })
}, "0.0.1", {
    requires: F.config.modules["nav-selecta-transjax"].requires || []
});
YUI.add("nav-selecta", function(a) {
    var f = false;
    a.on("keydown", function(k) {
        if (k.metaKey || k.ctrlKey) {
            f = true
        }
    });
    a.on("keyup", function(k) {
        if (f) {
            f = false
        }
    });
    a.NavSelecta = function(q, m) {
        m = m || {};
        m.showSubtitle = (typeof m.showSubtitle !== "undefined") ? m.showSubtitle : true;
        m.showIcon = (typeof m.showIcon !== "undefined") ? m.showIcon : true;
        m.maxResultsDisplayed = m.maxResultsDisplayed || 15;
        m.searchOnRealname = (typeof m.searchOnRealname !== "undefined") ? m.searchOnRealname : true;
        m.searchOnEmail = (typeof m.searchOnEmail !== "undefined") ? m.searchOnEmail : true;
        m.searchOnPathAlias = (typeof m.searchOnPathAlias !== "undefined") ? m.searchOnPathAlias : true;
        m.selectFirstItem = (typeof m.selectFirstItem !== "undefined") ? m.selectFirstItem : true;
        m.loadTransparently = (typeof m.loadTransparently !== "undefined") ? m.loadTransparently : true;
        m.allowFormSubmit = (typeof m.allowFormSubmit !== "undefined") ? m.allowFormSubmit : true;
        m.minListWidth = m.minListWidth || (F.config.flickr.flags.enable_global_nav ? 241 : 266);
        a.NavSelecta.superclass.constructor.call(this, q, m);
        this.inputField.removeClass("grey");
        this.maxContactsDisplayed = m.maxContactsDisplayed || 3;
        this.maxSectionsDisplayed = m.maxSectionsDisplayed || 3;
        this.on("BoSelecta:resultSelect", e);
        this.on("BoSelecta:resultClick", e);
        if (a.config.flickr.user && a.config.flickr.user.nsid) {
            this.inputField.once("mouseover", this.fetchData, this, true);
            this.inputField.once("focus", this.fetchData, this, true);
            this.on("BoSelecta:dataFetchComplete", this.refreshResults, this, true)
        }
        this.container.addClass("nav-selecta");
        var l = this.autoComp.get("listNode").ancestor(".yui3-aclist-content"), o = "search-menu-hover";
        function n(t, s) {
            if (this._hover_hide_timeout) {
                this._hover_hide_timeout.cancel()
            }
        }
        function p(t, s) {
            n(t, s);
            this._hover_hide_timeout = a.later(1000, this, function() {
                a.fire("flickr-menus:hide", s);
                this._hover_hide_timeout = null
            })
        }
        function r(t, s) {
            t.target.addClass("search-menu-hover");
            n(t, s)
        }
        function k(t, s) {
            t.target.removeClass("search-menu-hover");
            p(t, s)
        }
        l.on("mouseover", r, this);
        l.on("mouseout", k, this);
        this.container.on("mouseover", r, this, "source:nav-selecta-container");
        this.container.on("mouseout", k, this, "source:nav-selecta-container");
        this.container.on("keyup", function(s) {
            if (!s.target.hasClass("search-menu-hover")) {
                k.apply(this, arguments)
            }
        }, this);
        a.keyboardShortcutManager.register({
            keystring: "83",
            handler: d,
            context: "",
            scope: this,
            legend: {
                key: "S",
                description: a.transjax.get("nav-selecta", "jump_to_the_search_field"),
                group: a.transjax.get("keyboard-shortcut-legend", "navigation"),
                order: 99
            }
        });
        this.addSections(a.config.flickr.nav_selecta.additional_sections)
    };
    var c = {}, i = {};
    function e(l, k) {
        if (l.type === "contact") {
            if (!f) {
                g("/photos/" + (l.a ? l.a : l.n) + "/")
            }
        } else {
            if (l.type === "section") {
                if (!f) {
                    g(l.url)
                }
            } else {
                if (!f) {
                    g(l.url)
                }
            }
        }
    }
    function d(k) {
        k.preventDefault();
        window.scrollTo(0, 0);
        this.inputField.focus()
    }
    function g(k) {
        a.rapidTracker.beacon("Search-search_icon");
        a.fire("NavSelecta:beforeNavigatingAway", k);
        window.location = k
    }
    function b(n) {
        var p = location.search.split("?")[1] || null;
        if (p) {
            var m = p.split("&"), s = n.split("&"), o = null, k = {}, l = null, r = null, q = "";
            for (l = 0; s.length > l; l++) {
                o = s[l].split("=");
                k[o[0]] = o[1]
            }
            for (l = 0; m.length > l; l++) {
                o = m[l].split("=");
                if (!o[0].match(/^[a-zA-Z][a-zA-Z0-9]*/)) {
                    continue
                }
                if (k[o[0]]) {
                    r = k[o[0]];
                    k[o[0]] = null
                } else {
                    r = o[1]
                }
                q += o[0] + "=" + a.StringFilters.escape_entities(r) + "&"
            }
            for (l in k) {
                if (k[l]) {
                    q += l + "=" + k[l] + "&"
                }
            }
            return q.substring(0, q.length - 1)
        } else {
            return n
        }
    }
    c._searchContacts = function(q) {
        var J, l = [], z = q, s, u = this, t = false, I = false, A = false, H = false, x = false;
        l.push({
            type: "search",
            message: a.transjax.get("nav-selecta", "search_on_flickr", '<em class="term-highlight">' + z + "</em>"),
            onclick: function(m) {
                if (!f) {
                    g("/search/?" + b("q=" + encodeURIComponent(z)))
                }
            },
            url: "/search/?" + b("q=" + encodeURIComponent(z))
        });
        if (a.config.flickr.nav_selecta.photostream_search_user) {
            if (a.config.flickr.user && a.config.flickr.user.nsid && a.config.flickr.user.nsid === a.config.flickr.nav_selecta.photostream_search_user.nsid) {
                l.push({
                    type: "search-photostream",
                    message: a.transjax.get("nav-selecta", "search_photostream", '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=" + a.config.flickr.user.nsid + "&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=" + a.config.flickr.user.nsid + "&q=" + encodeURIComponent(z))
                });
                t = true
            } else {
                l.push({
                    type: "search-users-photostream",
                    message: a.transjax.get("nav-selecta", "search_users_photostream", a.StringFilters.escape_entities(a.config.flickr.nav_selecta.photostream_search_user.owner_name_possessive), '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=" + a.config.flickr.nav_selecta.photostream_search_user.nsid + "&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=" + a.config.flickr.nav_selecta.photostream_search_user.nsid + "&q=" + encodeURIComponent(z))
                });
                I = true
            }
        }
        if (a.config.flickr.nav_selecta.favorites_search_user) {
            if (a.config.flickr.user && a.config.flickr.user.nsid && a.config.flickr.user.nsid === a.config.flickr.nav_selecta.favorites_search_user.nsid) {
                l.push({
                    type: "search-favorites",
                    message: a.transjax.get("nav-selecta", "search_favorites", '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=faves&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=faves&q=" + encodeURIComponent(z))
                });
                A = true
            } else {
                l.push({
                    type: "search-users-favorites",
                    message: a.transjax.get("nav-selecta", "search_users_favorites", a.StringFilters.escape_entities(a.config.flickr.nav_selecta.favorites_search_user.owner_name_possessive), '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=faves-" + a.config.flickr.nav_selecta.favorites_search_user.nsid + "&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=faves-" + a.config.flickr.nav_selecta.favorites_search_user.nsid + "&q=" + encodeURIComponent(z))
                });
                H = true
            }
        }
        if (a.config.flickr.nav_selecta.search_group && a.config.flickr.nav_selecta.search_group.nsid) {
            l.push({
                type: "search-group",
                message: a.transjax.get("nav-selecta", "search_group", '<em class="term-highlight">' + z + "</em>"),
                onclick: function() {
                    if (!f) {
                        g("/search/groups/?" + b("w=" + encodeURIComponent(a.config.flickr.nav_selecta.search_group.nsid) + "&m=pool&q=" + encodeURIComponent(z)))
                    }
                },
                url: "/search/groups/?" + b("w=" + encodeURIComponent(a.config.flickr.nav_selecta.search_group.nsid) + "&m=pool&q=" + encodeURIComponent(z))
            });
            x = true
        }
        if (!this.showMoreSearches&&!a.config.flickr.nav_selecta.search_view) {
            l.push({
                type: "search-more",
                message: a.transjax.get("nav-selecta", "search_more", '<em class="term-highlight">' + z + "</em>"),
                onclick: function() {
                    u.showMoreSearches = true;
                    u.selectSecondItem = u.autoComp.after("results", function(m) {
                        if (t || I || A || H || x) {
                            u.autoComp.set("activeItem", u.autoComp._getFirstItemNode().next().next())
                        } else {
                            u.autoComp.set("activeItem", u.autoComp._getFirstItemNode().next())
                        }
                        u.selectSecondItem.detach()
                    });
                    u.refreshResults()
                }
            })
        } else {
            if (a.config.flickr.user && a.config.flickr.user.nsid&&!t) {
                l.push({
                    type: "search-photostream",
                    message: a.transjax.get("nav-selecta", "search_photostream", '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=" + a.config.flickr.user.nsid + "&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=" + a.config.flickr.user.nsid + "&q=" + encodeURIComponent(z))
                })
            }
            if (a.config.flickr.nav_selecta.photostream_search_user && (!a.config.flickr.user || (a.config.flickr.user.nsid !== a.config.flickr.nav_selecta.photostream_search_user.nsid))&&!I) {
                l.push({
                    type: "search-users-photostream",
                    message: a.transjax.get("nav-selecta", "search_users_photostream", a.StringFilters.escape_entities(a.config.flickr.nav_selecta.photostream_search_user.owner_name_possessive), '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=" + a.config.flickr.nav_selecta.photostream_search_user.nsid + "&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=" + a.config.flickr.nav_selecta.photostream_search_user.nsid + "&q=" + encodeURIComponent(z))
                })
            }
            if (a.config.flickr.nav_selecta.favorites_search_user && (!a.config.flickr.user || (a.config.flickr.user.nsid !== a.config.flickr.nav_selecta.favorites_search_user.nsid))&&!H) {
                l.push({
                    type: "search-users-favorites",
                    message: a.transjax.get("nav-selecta", "search_users_favorites", a.StringFilters.escape_entities(a.config.flickr.nav_selecta.favorites_search_user.owner_name_possessive), '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=" + a.config.flickr.nav_selecta.favorites_search_user.nsid + "&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=" + a.config.flickr.nav_selecta.favorites_search_user.nsid + "&q=" + encodeURIComponent(z))
                })
            }
            if (a.config.flickr.user && a.config.flickr.user.nsid) {
                l.push({
                    type: "search-contacts",
                    message: a.transjax.get("nav-selecta", "search_contacts", '<em class="term-highlight">' + z + "</em>"),
                    onclick: function() {
                        if (!f) {
                            g("/search/?" + b("w=contacts&q=" + encodeURIComponent(z)))
                        }
                    },
                    url: "/search/?" + b("w=contacts&q=" + encodeURIComponent(z))
                })
            }
            l.push({
                type: "search-groups",
                message: a.transjax.get("nav-selecta", "search_groups", '<em class="term-highlight">' + z + "</em>"),
                onclick: function() {
                    if (!f) {
                        g("/search/groups/?" + b("q=" + encodeURIComponent(z)))
                    }
                },
                url: "/search/groups/?" + b("q=" + encodeURIComponent(z))
            });
            l.push({
                type: "search-people",
                message: a.transjax.get("nav-selecta", "search_people", '<em class="term-highlight">' + z + "</em>"),
                onclick: function() {
                    if (!f) {
                        g("/search/people/?" + b("m=names&q=" + encodeURIComponent(z)))
                    }
                },
                url: "/search/people/?" + b("m=names&q=" + encodeURIComponent(z))
            })
        }
        if (q && typeof q === "string") {
            q = this._escapeForRegEx(q);
            q = a.StringFilters.substitute_equivalent_chars(q);
            var k = new RegExp(q, "i"), E, v, C, D, G, r;
            if (this.contacts) {
                var B = false, y = [];
                for (C = 0, G = this.contacts.length; C < G; C++) {
                    v = false;
                    E = this.contacts[C];
                    E.matched = null;
                    if (E.n && (!this.includeUser || z === this.meString) && E.n === a.config.flickr.user.nsid) {
                        if (z === this.meString&&!this.skipList[E.n]) {
                            J = E
                        }
                        continue
                    }
                    if (E.e&&!E.u&&!E.n) {
                        continue
                    }
                    if (this.skipList && (this.skipList[E.n] || this.skipList[E.e])) {
                        continue
                    }
                    if (this.searchOnUsername && E.u && E.u.search(k)!==-1) {
                        v = true;
                        E.matched = "username";
                        E.matchedPos = E.u.search(k)
                    }
                    if (!v && this.searchOnRealname && E.r && E.r.search(k)!==-1) {
                        v = true;
                        E.matched = "realname";
                        E.matchedPos = E.r.search(k)
                    }
                    if (!v && this.searchOnEmail && E.e && q !== "@" && E.e.search(k)!==-1) {
                        v = true;
                        E.matched = "email";
                        E.matchedPos = E.e.search(k)
                    }
                    if (!v && this.searchOnPathAlias && E.a && E.a.search(k)!==-1) {
                        v = true;
                        E.matched = "path_alias";
                        E.matchedPos = E.a.search(k)
                    }
                    if (v) {
                        B = true;
                        E.type = "contact";
                        y.push(E)
                    }
                }
                if (B) {
                    s = {
                        username: 0,
                        realname: 1,
                        email: 2,
                        path_alias: 3
                    };
                    y.sort(function(L, m) {
                        var K = (L.n && L.u) ? "contact": "addressBookEntry";
                        var M = (m.n && m.u) ? "contact": "addressBookEntry";
                        if (K !== M) {
                            return (K === "contact")?-1 : 1
                        }
                        if (L.matchedPos === m.matchedPos) {
                            if (s[L.matched] === s[m.matched]) {
                                var N = L.u.charCodeAt(0), n = m.u.charCodeAt(0);
                                if (N === n) {
                                    return 0
                                } else {
                                    if (N < n) {
                                        return - 1
                                    } else {
                                        return 1
                                    }
                                }
                            } else {
                                if (s[L.matched] < s[m.matched]) {
                                    return - 1
                                } else {
                                    return 1
                                }
                            }
                        } else {
                            if (L.matchedPos < m.matchedPos) {
                                return - 1
                            } else {
                                return 1
                            }
                        }
                    });
                    if (z === this.meString && J) {
                        J.matched = "username";
                        l.unshift(J)
                    }
                    l = l.concat(y.slice(0, this.maxContactsDisplayed))
                }
            } else {
                if (a.config.flickr.user && a.config.flickr.user.nsid) {
                    this.fetchData()
                }
            }
            if (this.groups) {}
            if (a.NavSelecta.nav_sections) {
                var p = false, w = [], o;
                for (C = 0, G = a.NavSelecta.nav_sections.length; C < G; C++) {
                    v = false;
                    section = a.NavSelecta.nav_sections[C];
                    section.matched = null;
                    if (section.name && section.name.search(k)!==-1) {
                        v = true;
                        section.matched = "name";
                        section.matchedPos = section.name.search(k)
                    }
                    if (!v && section.aliases && section.aliases.length) {
                        for (D = 0, r = section.aliases.length; D < r; D++) {
                            o = section.aliases[D].search(k);
                            if (o!==-1 && (!section.matchedPos || o < section.matchedPos)) {
                                v = true;
                                section.matched = "alias";
                                section.matchedPos = o
                            }
                        }
                    }
                    if (!v && this.searchOnURL && section.url && section.url.search(k)!==-1) {
                        v = true;
                        section.matched = "url";
                        section.matchedPos = section.url.search(k)
                    }
                    if (v) {
                        p = true;
                        section.type = "section";
                        w.push(section)
                    }
                }
                if (p) {
                    s = {
                        name: 0,
                        alias: 1,
                        url: 2
                    };
                    w.sort(function(K, m) {
                        if (K.matchedPos === m.matchedPos) {
                            if (s[K.matched] === s[m.matched]) {
                                var L = K.name.charCodeAt(0), n = m.name.charCodeAt(0);
                                if (L === n) {
                                    return 0
                                } else {
                                    if (L < n) {
                                        return - 1
                                    } else {
                                        return 1
                                    }
                                }
                            } else {
                                if (s[K.matched] < s[m.matched]) {
                                    return - 1
                                } else {
                                    return 1
                                }
                            }
                        } else {
                            if (K.matchedPos < m.matchedPos) {
                                return - 1
                            } else {
                                return 1
                            }
                        }
                    });
                    l = l.concat(w.slice(0, this.maxSectionsDisplayed))
                }
            }
        }
        return l
    };
    c._formatResult = function(q, z, r) {
        var s = this, B, v = [], G, m, l, w, k=!q ? null : new RegExp(a.StringFilters.substitute_equivalent_chars(s._escapeForRegEx(q)), "gi"), t, p, x, D, A, o, u = false, E = false;
        if (!q) {
            q = ""
        }
        for (var y = 0, C = z.length; y < C; y++) {
            B = z[y].raw;
            if (!B) {
                continue
            }
            if (B.type === "search-more") {
                G = '<p class="bs-message bs-onclick ns-search ns-search-more"><span class="inner"><span class="icon"></span>' + B.message + "</span></p>";
                v.push(G);
                continue
            }
            if (B.type.match(/^search/)) {
                G = '<p class="bs-message bs-onclick ns-search"><a class="inner" href="' + B.url + '"><span class="icon"></span>' + B.message + "</a></p>";
                v.push(G);
                continue
            }
            if (B.type === "section") {
                G = '<p class="bs-message ns-section ns-section-' + (B.sectionType ? B.sectionType : "generic") + (E ? "" : " first-section") + '"><a href="' + B.url + '" class="inner"><span class="icon"></span>' + h(B.name, k) + "</a></p>";
                v.push(G);
                E = true;
                continue
            }
            if (B.globalSearchLink) {
                G = '<p class="bs-global-search-link bs-message bs-onclick"><span>' + B.message + "</span></p>";
                v.push(G);
                continue
            }
            if (B.message) {
                G = '<p class="bs-message' + (B.notFoundLink ? "" : " bs-onclick") + (B.disableActive ? " bs-disable-active" : "") + (B.className ? " " + B.className : "") + '">' + B.message + "</p>";
                v.push(G);
                continue
            }
            pClasses = [];
            if (!u) {
                pClasses.push("first-contact")
            }
            if (z[y + 1] && z[y + 1].raw && z[y + 1].raw.type !== "contact") {
                pClasses.push("last-contact")
            }
            w = (pClasses.length) ? ' class="' + pClasses.join(" ") + '"' : "";
            if (B.e&&!B.u&&!B.n) {
                if (!B.i) {
                    B.i = "/images/icon_unread.gif"
                }
                m = s.showIcon ? '<img src="' + B.i + '" width="16" height="16" class="BuddyIconX bs-email-icon">' : "";
                l = '<p email="' + B.e + '"' + w + '><span class="name name-email">' + m + '<strong class="username">' + h(B.e, k) + '</strong></span><span class="relationship"></span></p>';
                v.push(l);
                u = true;
                continue
            }
            t = (B.a) ? B.a : B.n;
            p = ' / <a href="/people/' + t + '/">' + a.transjax.get("bo-selecta", "uber_contact_list_view_profile") + "</a>";
            if (!B.i) {
                B.i = (B.f && B.f !== 0 && B.s && B.s !== 0) ? "http://farm" + B.f + "." + a.config.flickr.static_domain + "/" + B.s + "/buddyicons/" + B.n + ".jpg" : "http://www.flickr.com/images/buddyicon.gif"
            }
            if (F.config.flickr.is_secure) {
                B.i = B.i.replace("http:", "https:")
            }
            x = "";
            if (s.showRelationship) {
                if (B.d === "1" && B.y === "1") {
                    x += a.transjax.get("bo-selecta", "uber_contact_list_friend_and_family")
                } else {
                    if (B.d === "1") {
                        x += a.transjax.get("bo-selecta", "uber_contact_list_friend")
                    } else {
                        if (B.y === "1") {
                            x += a.transjax.get("bo-selecta", "uber_contact_list_family")
                        } else {
                            if (B.removed === "1") {
                                x += '<em class="contact-removed">' + a.transjax.get("bo-selecta", "uber_contact_list_removed") + "</em>"
                            } else {
                                if (s.searchingGlobally) {
                                    x += ""
                                } else {
                                    x += a.transjax.get("bo-selecta", "uber_contact_list_contact")
                                }
                            }
                        }
                    }
                }
                if (s.showEditRelationshipLink&&!s.searchingGlobally) {
                    x += ' <span class="edit-rel">(<a href="/people/' + t + '/relationship/" onclick="icon_windowOpenFromLink(\'' + B.n + "'); return false;\">" + a.transjax.get("bo-selecta", "uber_contact_list_edit") + "</a>)</span>"
                }
            }
            m = s.showIcon ? '<img src="' + B.i + '" width="24" height="24" class="BuddyIconX fade-in" nsid="' + B.n + '" ' + s.iconAnimCode + ">" : "";
            m = s.linkUsernameToPhotostream ? '<a href="/photos/' + t + '/">' + m + "</a>" : m;
            D = s.showUsername ? h(s._sanitizeString(B.u, k)) : "";
            D = s.linkUsernameToPhotostream ? '<strong class="username"><a href="/photos/' + t + '/">' + D + "</a></strong>" : '<strong class="username">' + D + "</strong>";
            A = "";
            if (s.showSubtitle) {
                if (!B.matched || B.matched === "realname") {
                    A = B.r ? h(B.r, k) : a.transjax.get("bo-selecta", "uber_contact_list_no_realname")
                } else {
                    if (B.matched === "username") {
                        A = B.r ? s._sanitizeString(B.r) : a.transjax.get("bo-selecta", "uber_contact_list_no_realname")
                    } else {
                        if (B.matched === "email") {
                            A = h(B.e, k)
                        } else {
                            if (B.matched === "path_alias" && B.a) {
                                A = "flickr.com/people/" + h(B.a, k) + "/"
                            }
                        }
                    }
                }
                A = s.linkSubtitle ? '<a title="' + a.transjax.get("bo-selecta", "uber_contact_list_view_profile") + '" href="/people/' + t + '/" class="realname">' + A + "</a>" : '<span class="realname">' + A + "</span>"
            }
            o = a.Node.create('<p nsid="' + B.n + '"' + w + '><span class="name">' + m + D + A + '</span><span class="relationship">' + x + "</span></p>");
            a.imageFader.attach(o.all("img"));
            v.push(o);
            u = true
        }
        return v
    };
    function h(o, m) {
        if (!o) {
            return o
        }
        var k, l, n;
        o = a.StringFilters.unescape_entities(o, true);
        l = m ? o.search(m) : - 1;
        if (l===-1) {
            return a.BoSelecta3.sanitizeString(o)
        }
        k = o.match(m);
        n = l + k[0].length;
        return a.StringFilters.escape_entities(o.substring(0, l), true) + '<span class="term-highlight">' + a.StringFilters.escape_entities(o.substring(l, n), true) + "</span>" + a.StringFilters.escape_entities(o.substring(n, o.length), true)
    }
    function j(k) {
        return a.transjax.get("nav-selecta", k)
    }
    c.addSections = function(k) {
        a.NavSelecta.nav_sections = a.NavSelecta.nav_sections.concat(k)
    };
    i.nav_sections = [{
        name: j("section_your_photostream"),
        sectionType: "photos",
        url: "/photos/me/",
        aliases: []
    }, {
        name: j("section_your_sets"),
        sectionType: "photos",
        url: "/photos/me/sets/",
        aliases: []
    }, {
        name: j("section_your_collections"),
        sectionType: "photos",
        url: "/photos/me/collections/",
        aliases: []
    }, {
        name: j("section_your_galleries"),
        sectionType: "photos",
        url: "/photos/me/galleries/",
        aliases: []
    }, {
        name: j("section_your_archives"),
        sectionType: "photos",
        url: "/photos/me/archives/",
        aliases: []
    }, {
        name: j("section_your_tags"),
        sectionType: "photos",
        url: "/photos/me/tags/",
        aliases: []
    }, {
        name: j("section_your_map"),
        sectionType: "map",
        url: "/photos/me/map/",
        aliases: []
    }, {
        name: j("section_your_favorites"),
        sectionType: "photos",
        url: "/photos/me/favorites/",
        aliases: []
    }, {
        name: j("section_your_stats"),
        sectionType: "stats",
        url: "/photos/me/stats/",
        aliases: ["Graphs", "Referrers", "Referers", "Views", j("section_alias_graphs"), j("section_alias_referrers"), j("section_alias_views")]
    }, {
        name: j("section_your_apps"),
        sectionType: "",
        url: "/services/apps/by/me/",
        aliases: []
    }, {
        name: j("section_recent_activity"),
        sectionType: "activity",
        url: "/activity/",
        aliases: []
    }, {
        name: j("section_photos_of_you"),
        sectionType: "photos",
        url: "/photosof/me/",
        aliases: []
    }, {
        name: j("section_upload_photos"),
        sectionType: "photos",
        url: "/photos/upload/",
        aliases: ["Uploadr", j("section_alias_upload")]
    }, {
        name: j("section_your_account"),
        sectionType: "text",
        url: "/account/",
        aliases: ["Settings", "Preferences", j("section_alias_settings"), j("section_alias_preferences")]
    }, {
        name: j("section_your_profile"),
        sectionType: "",
        url: "/people/me/",
        aliases: []
    }, {
        name: j("section_flickrmail"),
        sectionType: "text",
        url: "/mail/",
        aliases: []
    }, {
        name: j("section_organize_and_create"),
        sectionType: "photos",
        url: "/photos/organize/",
        aliases: ["Batch", j("section_alias_batch")]
    }, {
        name: j("section_prints_and_photo_products"),
        sectionType: "photos",
        url: "/photos/organize/?start_tab=print",
        aliases: ["Snapfish", "Prints", "Calendar", "Book", "Buy", j("section_alias_prints"), j("section_alias_calendar"), j("section_alias_book"), j("section_alias_buy")]
    }, {
        name: j("section_contacts"),
        sectionType: "photos",
        url: "/photos/contacts/",
        aliases: []
    }, {
        name: j("section_photos_of_your_contacts"),
        sectionType: "photos",
        url: "/photosof/contacts/",
        aliases: []
    }, {
        name: j("section_contact_list"),
        sectionType: "contacts",
        url: "/people/me/contacts/",
        aliases: []
    }, {
        name: j("section_find_your_friends"),
        sectionType: "contacts",
        url: "/import/people/",
        aliases: ["Facebook", "Gmail", "Hotmail", "Yahoo Mail"]
    }, {
        name: j("section_invite_your_friends"),
        sectionType: "contacts",
        url: "/invite/",
        aliases: []
    }, {
        name: j("section_invite_history"),
        sectionType: "text",
        url: "/invite/history/",
        aliases: []
    }, {
        name: j("section_guest_pass_history"),
        sectionType: "text",
        url: "/invite/history/guests/",
        aliases: ["Guestpass"]
    }, {
        name: j("section_your_groups"),
        sectionType: "photos",
        url: "/groups/",
        aliases: []
    }, {
        name: j("section_recent_changes_in_your_groups"),
        sectionType: "activity",
        url: "/recent.gne",
        aliases: []
    }, {
        name: j("section_create_a_new_group"),
        sectionType: "text",
        url: "/groups_create.gne",
        aliases: []
    }, {
        name: j("section_the_tour"),
        sectionType: "text",
        url: "/tour/",
        aliases: []
    }, {
        name: j("section_explore"),
        sectionType: "photos",
        url: "/explore/",
        aliases: []
    }, {
        name: j("section_last_7_days_interesting"),
        sectionType: "photos",
        url: "/explore/interesting/7days/",
        aliases: []
    }, {
        name: j("section_popular_tags"),
        sectionType: "photos",
        url: "/photos/tags/",
        aliases: []
    }, {
        name: j("section_most_recent_uploads_to_flickr"),
        sectionType: "photos",
        url: "/photos/",
        aliases: []
    }, {
        name: j("section_video_on_flickr"),
        sectionType: "video",
        url: "/explore/video/",
        aliases: []
    }, {
        name: j("section_galleries"),
        sectionType: "photos",
        url: "/galleries/",
        aliases: []
    }, {
        name: j("section_explore_analog"),
        sectionType: "photos",
        url: "/analog/",
        aliases: ["Film", j("section_alias_film")]
    }, {
        name: j("section_world_map"),
        sectionType: "map",
        url: "/map/",
        aliases: []
    }, {
        name: j("section_places"),
        sectionType: "map",
        url: "/places/",
        aliases: []
    }, {
        name: j("section_the_commons"),
        sectionType: "photos",
        url: "/commons/",
        aliases: []
    }, {
        name: j("section_creative_commons"),
        sectionType: "photos",
        url: "/creativecommons/",
        aliases: ["CC", "License", j("section_alias_license")]
    }, {
        name: j("section_flickrblog"),
        sectionType: "photos",
        url: "http://blog.flickr.com/",
        aliases: []
    }, {
        name: j("section_getty_images"),
        sectionType: "photos",
        url: "/gettyimages/",
        aliases: ["License", "Buy", "Sell", j("section_alias_license"), j("section_alias_buy"), j("section_alias_sell")]
    }, {
        name: j("section_code_flickr"),
        sectionType: "code",
        url: "http://code.flickr.com/",
        aliases: ["API", "Developer", "Programming", "Documentation", j("section_alias_developer"), j("section_alias_programming"), j("section_alias_documentation")]
    }, {
        name: j("section_the_app_garden"),
        sectionType: "text",
        url: "/services/",
        aliases: ["Apps"]
    }, {
        name: j("section_camera_finder"),
        sectionType: "text",
        url: "/cameras/",
        aliases: []
    }, {
        name: j("section_developer_guidelines"),
        sectionType: "code",
        url: "/services/developer/",
        aliases: ["API", "Programming", "Code", "Documentation", j("section_alias_programming"), j("section_alias_code"), j("section_alias_documentation")]
    }, {
        name: j("section_api_docs"),
        sectionType: "code",
        url: "/services/api/",
        aliases: ["API", "Developer", "Programming", "Code", "Documentation", j("section_alias_developer"), j("section_alias_programming"), j("section_alias_code"), j("section_alias_documentation")]
    }, {
        name: j("section_feeds"),
        sectionType: "text",
        url: "/services/feeds/",
        aliases: ["RSS", "Atom", "Data"]
    }, {
        name: j("section_help"),
        sectionType: "text",
        url: "/help/",
        aliases: []
    }, {
        name: j("section_community_guidelines"),
        sectionType: "text",
        url: "/help/guidelines/",
        aliases: []
    }, {
        name: j("section_the_help_forum"),
        sectionType: "text",
        url: "/help/forum/",
        aliases: []
    }, {
        name: j("section_faq"),
        sectionType: "text",
        url: "/help/faq/",
        aliases: ["Frequently"]
    }, {
        name: j("section_tools"),
        sectionType: "text",
        url: "/tools/",
        aliases: ["Upload", "Plug-in", "Mac", "Windows", j("section_alias_upload"), j("section_alias_plugin")]
    }, {
        name: j("section_about_flickr"),
        sectionType: "meta",
        url: "/about/",
        aliases: ["Staff", j("section_alias_staff")]
    }, {
        name: j("section_jobs"),
        sectionType: "meta",
        url: "/jobs/",
        aliases: ["Career", "Work", j("section_alias_career"), j("section_alias_work")]
    }, {
        name: j("section_mobile"),
        sectionType: "",
        url: "/mobile/",
        aliases: ["Cellphone"]
    }, {
        name: "Android",
        sectionType: "",
        url: "/android/",
        aliases: []
    }
    ];
    a.extend(a.NavSelecta, a.BoSelecta3, c, i)
}, "0.0.1", {
    requires: F.config.modules["nav-selecta"].requires || [],
    optional: F.config.modules["nav-selecta"].optional || []
}); /*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("history-base", function(e, t) {
    function p() {
        this._init.apply(this, arguments)
    }
    function d(e) {
        return n.type(e) === "object"
    }
    var n = e.Lang, r = e.Object, i = YUI.namespace("Env.History"), s = e.Array, o = e.config.doc, u = o.documentMode, a = e.config.win, f = {
        merge: !0
    }, l = "change", c = "add", h = "replace";
    e.augment(p, e.EventTarget, null, null, {
        emitFacade: !0,
        prefix: "history",
        preventable: !1,
        queueable: !0
    }), i._state || (i._state = {}), p.NAME = "historyBase", p.SRC_ADD = c, p.SRC_REPLACE = h, p.html5=!!(a.history && a.history.pushState && a.history.replaceState && ("onpopstate"in a || e.UA.gecko >= 2) && (!e.UA.android || e.UA.android >= 2.4)), p.nativeHashChange = ("onhashchange"in a || "onhashchange"in o) && (!u || u > 7), e.mix(p.prototype, {
        _init: function(e) {
            var t;
            e = this._config = e || {}, this.force=!!e.force, t = this._initialState = this._initialState || e.initialState || null, this.publish(l, {
                broadcast: 2,
                defaultFn: this._defChangeFn
            }), t && this.replace(t)
        },
        add: function() {
            var e = s(arguments, 0, !0);
            return e.unshift(c), this._change.apply(this, e)
        },
        addValue: function(e, t, n) {
            var r = {};
            return r[e] = t, this._change(c, r, n)
        },
        get: function(t) {
            var n = i._state, s = d(n);
            return t ? s && r.owns(n, t) ? n[t] : undefined : s ? e.mix({}, n, !0) : n
        },
        replace: function() {
            var e = s(arguments, 0, !0);
            return e.unshift(h), this._change.apply(this, e)
        },
        replaceValue: function(e, t, n) {
            var r = {};
            return r[e] = t, this._change(h, r, n)
        },
        _change: function(t, n, r) {
            return r = r ? e.merge(f, r) : f, r.merge && d(n) && d(i._state) && (n = e.merge(i._state, n)), this._resolveChanges(t, n, r), this
        },
        _fireEvents: function(e, t, n) {
            this.fire(l, {
                _options: n,
                changed: t.changed,
                newVal: t.newState,
                prevVal: t.prevState,
                removed: t.removed,
                src: e
            }), r.each(t.changed, function(t, n) {
                this._fireChangeEvent(e, n, t)
            }, this), r.each(t.removed, function(t, n) {
                this._fireRemoveEvent(e, n, t)
            }, this)
        },
        _fireChangeEvent: function(e, t, n) {
            this.fire(t + "Change", {
                newVal: n.newVal,
                prevVal: n.prevVal,
                src: e
            })
        },
        _fireRemoveEvent: function(e, t, n) {
            this.fire(t + "Remove", {
                prevVal: n,
                src: e
            })
        },
        _resolveChanges: function(e, t, n) {
            var s = {}, o, u = i._state, a = {};
            t || (t = {}), n || (n = {}), d(t) && d(u) ? (r.each(t, function(e, t) {
                var n = u[t];
                e !== n && (s[t] = {
                    newVal: e,
                    prevVal: n
                }, o=!0)
            }, this), r.each(u, function(e, n) {
                if (!r.owns(t, n) || t[n] === null)
                    delete t[n], a[n] = e, o=!0
            }, this)) : o = t !== u, (o || this.force) && this._fireEvents(e, {
                changed: s,
                newState: t,
                prevState: u,
                removed: a
            }, n)
        },
        _storeState: function(e, t) {
            i._state = t || {}
        },
        _defChangeFn: function(e) {
            this._storeState(e.src, e.newVal, e._options)
        }
    }, !0), e.HistoryBase = p
}, "3.11.0", {
    requires: ["event-custom-complex"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("history-html5", function(e, t) {
    function a() {
        a.superclass.constructor.apply(this, arguments)
    }
    var n = e.HistoryBase, r = e.Lang, i = e.config.win, s = e.config.useHistoryHTML5, o = "popstate", u = n.SRC_REPLACE;
    e.extend(a, n, {
        _init: function(t) {
            var n = i.history.state;
            e.Object.isEmpty(n) && (n = null), t || (t = {}), t.initialState && r.type(t.initialState) === "object" && r.type(n) === "object" ? this._initialState = e.merge(t.initialState, n) : this._initialState = n, e.on("popstate", this._onPopState, i, this), a.superclass._init.apply(this, arguments)
        },
        _storeState: function(t, n, r) {
            t !== o && i.history[t === u ? "replaceState": "pushState"](n, r.title || e.config.doc.title || "", r.url || null), a.superclass._storeState.apply(this, arguments)
        },
        _onPopState: function(e) {
            this._resolveChanges(o, e._event.state || null)
        }
    }, {
        NAME: "historyhtml5",
        SRC_POPSTATE: o
    }), e.Node.DOM_EVENTS.popstate || (e.Node.DOM_EVENTS.popstate = 1), e.HistoryHTML5 = a;
    if (s===!0 || s!==!1 && n.html5)
        e.History = a
}, "3.11.0", {
    optional: ["json"],
    requires: ["event-base", "history-base", "node-base"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("history-hash", function(e, t) {
    function p() {
        p.superclass.constructor.apply(this, arguments)
    }
    var n = e.HistoryBase, r = e.Lang, i = e.Array, s = e.Object, o = YUI.namespace("Env.HistoryHash"), u = "hash", a, f, l, c = e.config.win, h = e.config.useHistoryHTML5;
    e.extend(p, n, {
        _init: function(t) {
            var n = p.parseHash();
            t = t || {}, this._initialState = t.initialState ? e.merge(t.initialState, n) : n, e.after("hashchange", e.bind(this._afterHashChange, this), c), p.superclass._init.apply(this, arguments)
        },
        _change: function(e, t, n) {
            return s.each(t, function(e, n) {
                r.isValue(e) && (t[n] = e.toString())
            }), p.superclass._change.call(this, e, t, n)
        },
        _storeState: function(e, t) {
            var r = p.decode, i = p.createHash(t);
            p.superclass._storeState.apply(this, arguments), e !== u && r(p.getHash()) !== r(i) && p[e === n.SRC_REPLACE ? "replaceHash": "setHash"](i)
        },
        _afterHashChange: function(e) {
            this._resolveChanges(u, p.parseHash(e.newHash), {})
        }
    }, {
        NAME: "historyHash",
        SRC_HASH: u,
        hashPrefix: "",
        _REGEX_HASH: /([^\?#&]+)=([^&]+)/g,
        createHash: function(e) {
            var t = p.encode, n = [];
            return s.each(e, function(e, i) {
                r.isValue(e) && n.push(t(i) + "=" + t(e))
            }), n.join("&")
        },
        decode: function(e) {
            return decodeURIComponent(e.replace(/\+/g, " "))
        },
        encode: function(e) {
            return encodeURIComponent(e).replace(/%20/g, "+")
        },
        getHash: e.UA.gecko ? function() {
            var t = e.getLocation(), n = /#(.*)$/.exec(t.href), r = n && n[1] || "", i = p.hashPrefix;
            return i && r.indexOf(i) === 0 ? r.replace(i, "") : r
        }
        : function() {
            var t = e.getLocation(), n = t.hash.substring(1), r = p.hashPrefix;
            return r && n.indexOf(r) === 0 ? n.replace(r, "") : n
        },
        getUrl: function() {
            return location.href
        },
        parseHash: function(e) {
            var t = p.decode, n, i, s, o, u = {}, a = p.hashPrefix, f;
            e = r.isValue(e) ? e : p.getHash();
            if (a) {
                f = e.indexOf(a);
                if (f === 0 || f === 1 && e.charAt(0) === "#")
                    e = e.replace(a, "")
            }
            s = e.match(p._REGEX_HASH) || [];
            for (n = 0, i = s.length; n < i; ++n)
                o = s[n].split("="), u[t(o[0])] = t(o[1]);
            return u
        },
        replaceHash: function(t) {
            var n = e.getLocation(), r = n.href.replace(/#.*$/, "");
            t.charAt(0) === "#" && (t = t.substring(1)), n.replace(r + "#" + (p.hashPrefix || "") + t)
        },
        setHash: function(t) {
            var n = e.getLocation();
            t.charAt(0) === "#" && (t = t.substring(1)), n.hash = (p.hashPrefix || "") + t
        }
    }), a = o._notifiers, a || (a = o._notifiers = []), e.Event.define("hashchange", {
        on: function(t, n, r) {
            (t.compareTo(c) || t.compareTo(e.config.doc.body)) && a.push(r)
        },
        detach: function(e, t, n) {
            var r = i.indexOf(a, n);
            r!==-1 && a.splice(r, 1)
        }
    }), f = p.getHash(), l = p.getUrl(), n.nativeHashChange ? o._hashHandle || (o._hashHandle = e.Event.attach("hashchange", function(e) {
        var t = p.getHash(), n = p.getUrl();
        i.each(a.concat(), function(r) {
            r.fire({
                _event: e,
                oldHash: f,
                oldUrl: l,
                newHash: t,
                newUrl: n
            })
        }), f = t, l = n
    }, c)) : o._hashPoll || (o._hashPoll = e.later(50, null, function() {
        var e = p.getHash(), t, n;
        f !== e && (n = p.getUrl(), t = {
            oldHash: f,
            oldUrl: l,
            newHash: e,
            newUrl: n
        }, f = e, l = n, i.each(a.concat(), function(e) {
            e.fire(t)
        }))
    }, null, !0)), e.HistoryHash = p;
    if (h===!1 ||!e.History && h!==!0 && (!n.html5 ||!e.HistoryHTML5))
        e.History = p
}, "3.11.0", {
    requires: ["event-synthetic", "history-base", "yui-later"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("history-hash-ie", function(e, t) {
    if (e.UA.ie&&!e.HistoryBase.nativeHashChange) {
        var n = e.Do, r = YUI.namespace("Env.HistoryHash"), i = e.HistoryHash, s = r._iframe, o = e.config.win;
        i.getIframeHash = function() {
            if (!s ||!s.contentWindow)
                return "";
            var e = i.hashPrefix, t = s.contentWindow.location.hash.substr(1);
            return e && t.indexOf(e) === 0 ? t.replace(e, "") : t
        }, i._updateIframe = function(e, t) {
            var n = s && s.contentWindow && s.contentWindow.document, r = n && n.location;
            if (!n ||!r)
                return;
            t ? r.replace(e.charAt(0) === "#" ? e : "#" + e) : (n.open().close(), r.hash = e)
        }, n.before(i._updateIframe, i, "replaceHash", i, !0), s || e.on("domready", function() {
            var t = i.getHash();
            s = r._iframe = e.Node.getDOMNode(e.Node.create('<iframe src="javascript:0" style="display:none" height="0" width="0" tabindex="-1" title="empty"/>')), e.config.doc.documentElement.appendChild(s), i._updateIframe(t || "#"), e.on("hashchange", function(e) {
                t = e.newHash, i.getIframeHash() !== t && i._updateIframe(t)
            }, o), e.later(50, null, function() {
                var e = i.getIframeHash();
                e !== t && i.setHash(e)
            }, null, !0)
        })
    }
}, "3.11.0", {
    requires: ["history-hash", "node-base"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("router", function(e, t) {
    function f() {
        f.superclass.constructor.apply(this, arguments)
    }
    var n = e.HistoryHash, r = e.QueryString, i = e.Array, s = e.config.win, o = [], u = [], a = "ready";
    e.Router = e.extend(f, e.Base, {
        _regexPathParam: /([:*])([\w\-]+)?/g,
        _regexUrlQuery: /\?([^#]*).*$/,
        _regexUrlOrigin: /^(?:[^\/#?:]+:\/\/|\/\/)[^\/]*/,
        initializer: function(t) {
            var n = this;
            n._html5 = n.get("html5"), n._routes = [], n._url = n._getURL(), n._setRoutes(t && t.routes ? t.routes : n.get("routes")), n._html5 ? (n._history = new e.HistoryHTML5({
                force: !0
            }), n._historyEvents = e.after("history:change", n._afterHistoryChange, n)) : n._historyEvents = e.on("hashchange", n._afterHistoryChange, s, n), n.publish(a, {
                defaultFn: n._defReadyFn,
                fireOnce: !0,
                preventable: !1
            }), n.once("initializedChange", function() {
                e.once("load", function() {
                    setTimeout(function() {
                        n.fire(a, {
                            dispatched: !!n._dispatched
                        })
                    }, 20)
                })
            }), o.push(this)
        },
        destructor: function() {
            var e = i.indexOf(o, this);
            e>-1 && o.splice(e, 1), this._historyEvents && this._historyEvents.detach()
        },
        dispatch: function() {
            return this.once(a, function() {
                this._ready=!0, this.upgrade() || this._dispatch(this._getPath(), this._getURL())
            }), this
        },
        getPath: function() {
            return this._getPath()
        },
        hasRoute: function(e) {
            var t;
            return this._hasSameOrigin(e) ? (this._html5 || (e = this._upgradeURL(e)), t = this.removeQuery(this.removeRoot(e)), !!this.match(t).length) : !1
        },
        match: function(e) {
            return i.filter(this._routes, function(t) {
                return e.search(t.regex)>-1
            })
        },
        removeRoot: function(e) {
            var t = this.get("root");
            return e = e.replace(this._regexUrlOrigin, ""), t && e.indexOf(t) === 0 && (e = e.substring(t.length)), e.charAt(0) === "/" ? e : "/" + e
        },
        removeQuery: function(e) {
            return e.replace(/\?.*$/, "")
        },
        replace: function(e) {
            return this._queue(e, !0)
        },
        route: function(e, t) {
            t = i.flatten(i(arguments, 1, !0));
            var n = [];
            return this._routes.push({
                callbacks: t,
                keys: n,
                path: e,
                regex: this._getRegex(e, n),
                callback: t[0]
            }), this
        },
        save: function(e) {
            return this._queue(e)
        },
        upgrade: function() {
            if (!this._html5)
                return !1;
            var e = this._getHashPath();
            return e ? (this.once(a, function() {
                this.replace(e)
            }), !0) : !1
        },
        _decode: function(e) {
            return decodeURIComponent(e.replace(/\+/g, " "))
        },
        _dequeue: function() {
            var t = this, n;
            return YUI.Env.windowLoaded ? (n = u.shift(), n ? n() : this) : (e.once("load", function() {
                t._dequeue()
            }), this)
        },
        _dispatch: function(t, n, r) {
            var s = this, o = s._decode, u = s.match(t), a = [], f, l, c;
            return s._dispatching = s._dispatched=!0, !u ||!u.length ? (s._dispatching=!1, s) : (l = s._getRequest(t, n, r), c = s._getResponse(l), l.next = function(n) {
                var r, h, p;
                if (n)
                    n === "route" ? (a = [], l.next()) : e.error(n);
                else if (r = a.shift())typeof r == "string" && (h = r, r = s[h], r || e.error("Router: Callback not found: " + h, null, "router")), l.pendingCallbacks = a.length, r.call(s, l, c, l.next);
                else if (p = u.shift())
                    a = p.callbacks.concat(), f = i.map(p.regex.exec(t) || [], o), f.length === p.keys.length + 1 ? l.params = i.hash(p.keys, f.slice(1)) : l.params = f.concat(), l.pendingRoutes = u.length, l.next()
            }, l.next(), s._dispatching=!1, s._dequeue())
        },
        _getHashPath: function(e) {
            return e || (e = n.getHash()), e && e.charAt(0) === "/" ? this._joinURL(e) : ""
        },
        _getOrigin: function() {
            var t = e.getLocation();
            return t.origin || t.protocol + "//" + t.host
        },
        _getPath: function() {
            var t=!this._html5 && this._getHashPath() || e.getLocation().pathname;
            return this.removeQuery(this.removeRoot(t))
        },
        _getPathRoot: function() {
            var t = "/", n = e.getLocation().pathname, r;
            return n.charAt(n.length - 1) === t ? n : (r = n.split(t), r.pop(), r.join(t) + t)
        },
        _getQuery: function() {
            var t = e.getLocation(), r, i;
            return this._html5 ? t.search.substring(1) : (r = n.getHash(), i = r.match(this._regexUrlQuery), r && i ? i[1] : t.search.substring(1))
        },
        _getRegex: function(e, t) {
            return e instanceof RegExp ? e : e === "*" ? /.*/ : (e = e.replace(this._regexPathParam, function(e, n, r) {
                return r ? (t.push(r), n === "*" ? "(.*?)" : "([^/#?]*)") : n === "*" ? ".*" : e
            }), new RegExp("^" + e + "$"))
        },
        _getRequest: function(e, t, n) {
            return {
                path: e,
                query: this._parseQuery(this._getQuery()),
                url: t,
                src: n
            }
        },
        _getResponse: function(e) {
            var t = function() {
                return e.next.apply(this, arguments)
            };
            return t.req = e, t
        },
        _getRoutes: function() {
            return this._routes.concat()
        },
        _getURL: function() {
            var t = e.getLocation().toString();
            return this._html5 || (t = this._upgradeURL(t)), t
        },
        _hasSameOrigin: function(t) {
            var n = (t && t.match(this._regexUrlOrigin) || [])[0];
            return n && n.indexOf("//") === 0 && (n = e.getLocation().protocol + n), !n || n === this._getOrigin()
        },
        _joinURL: function(e) {
            var t = this.get("root");
            return e = this.removeRoot(e), e.charAt(0) === "/" && (e = e.substring(1)), t && t.charAt(t.length - 1) === "/" ? t + e : t + "/" + e
        },
        _normalizePath: function(e) {
            var t = "..", n = "/", r, i, s, o, u, a;
            if (!e || e === n)
                return n;
            o = e.split(n), a = [];
            for (r = 0, i = o.length; r < i; ++r)
                u = o[r], u === t ? a.pop() : u && a.push(u);
            return s = n + a.join(n), s !== n && e.charAt(e.length - 1) === n && (s += n), s
        },
        _parseQuery: r && r.parse ? r.parse: function(e) {
            var t = this._decode, n = e.split("&"), r = 0, i = n.length, s = {}, o;
            for (; r < i; ++r)
                o = n[r].split("="), o[0] && (s[t(o[0])] = t(o[1] || ""));
            return s
        },
        _queue: function() {
            var t = arguments, n = this;
            return u.push(function() {
                return n._html5 ? e.UA.ios && e.UA.ios < 5 ? n._save.apply(n, t) : setTimeout(function() {
                    n._save.apply(n, t)
                }, 1) : (n._dispatching=!0, n._save.apply(n, t)), n
            }), this._dispatching ? this : this._dequeue()
        },
        _resolvePath: function(t) {
            return t ? (t.charAt(0) !== "/" && (t = this._getPathRoot() + t), this._normalizePath(t)) : e.getLocation().pathname
        },
        _resolveURL: function(t) {
            var n = t && t.match(this._regexURL), r, i, s, o, u;
            return n ? (r = n[1], i = n[2], s = n[3], o = n[4], r ? (r.indexOf("//") === 0 && (r = e.getLocation().protocol + r), r + (i || "/") + (s || "") + (o || "")) : (u = this._getOrigin() + this._resolvePath(i), i || s ? u + (s || "") + (o || "") : (s = this._getQuery(), u + (s ? "?" + s : "") + (o || "")))) : e.getLocation().toString()
        },
        _save: function(t, r) {
            var i = typeof t == "string", s, o, u;
            if (i&&!this._hasSameOrigin(t))
                return e.error("Security error: The new URL must be of the same origin as the current URL."), this;
            i && (t = this._joinURL(t)), this._ready=!0;
            if (this._html5)
                this._history[r ? "replace": "add"](null, {
                    url: t
                });
            else {
                s = e.getLocation().pathname, o = this.get("root"), u = n.getHash(), i || (t = u);
                if (o === s || o === this._getPathRoot())
                    t = this.removeRoot(t);
                t === u ? e.Router.dispatch() : n[r ? "replaceHash": "setHash"](t)
            }
            return this
        },
        _setRoutes: function(e) {
            return this._routes = [], i.each(e, function(e) {
                var t = e.callbacks || e.callback
                ;
                this.route(e.path, t)
            }, this), this._routes.concat()
        },
        _upgradeURL: function(t) {
            if (!this._hasSameOrigin(t))
                return t;
            var n = (t.match(/#(.*)$/) || [])[1] || "", r = e.HistoryHash.hashPrefix, i;
            r && n.indexOf(r) === 0 && (n = n.replace(r, ""));
            if (n) {
                i = this._getHashPath(n);
                if (i)
                    return this._resolveURL(i)
            }
            return t
        },
        _afterHistoryChange: function(e) {
            var t = this, n = e.src, r = t._url, i = t._getURL();
            t._url = i;
            if (n === "popstate" && (!t._ready || r.replace(/#.*$/, "") === i.replace(/#.*$/, "")))
                return;
            t._dispatch(t._getPath(), i, n)
        },
        _defReadyFn: function(e) {
            this._ready=!0
        }
    }, {
        NAME: "router",
        ATTRS: {
            html5: {
                valueFn: function() {
                    return e.Router.html5
                },
                writeOnce: "initOnly"
            },
            root: {
                value: ""
            },
            routes: {
                value: [],
                getter: "_getRoutes",
                setter: "_setRoutes"
            }
        },
        html5: e.HistoryBase.html5 && (!e.UA.android || e.UA.android >= 3),
        _instances: o,
        dispatch: function() {
            var e, t, n;
            for (e = 0, t = o.length; e < t; e += 1)
                n = o[e], n && n._dispatch(n._getPath(), n._getURL())
        }
    }), e.Controller = e.Router
}, "3.11.0", {
    optional: ["querystring-parse"],
    requires: ["array-extras", "base-build", "history"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("pjax-base", function(e, t) {
    function s() {}
    var n = e.config.win, r = e.ClassNameManager.getClassName("pjax"), i = "navigate";
    s.prototype = {
        _regexURL: /^((?:[^\/#?:]+:\/\/|\/\/)[^\/]*)?([^?#]*)(\?[^#]*)?(#.*)?$/,
        initializer: function() {
            this.publish(i, {
                defaultFn: this._defNavigateFn
            }), this.get("html5") && this._pjaxBindUI()
        },
        destructor: function() {
            this._pjaxEvents && this._pjaxEvents.detach()
        },
        navigate: function(t, n) {
            return t = this._resolveURL(t), this._navigate(t, n)?!0 : (this._hasSameOrigin(t) || e.error("Security error: The new URL must be of the same origin as the current URL."), !1)
        },
        _isLinkSameOrigin: function(t) {
            var n = e.getLocation(), r = n.protocol, i = n.hostname, s = parseInt(n.port, 10) || null, o;
            return t.get("protocol") !== r || t.get("hostname") !== i?!1 : (o = parseInt(t.get("port"), 10) || null, r === "http:" ? (s || (s = 80), o || (o = 80)) : r === "https:" && (s || (s = 443), o || (o = 443)), o === s)
        },
        _navigate: function(t, r) {
            t = this._upgradeURL(t);
            if (!this.hasRoute(t))
                return !1;
            r = e.merge(r, {
                url: t
            });
            var s = this._getURL(), o, u;
            u = t.replace(/(#.*)$/, function(e, t, n) {
                return o = t, e.substring(n)
            });
            if (o && u === s.replace(/#.*$/, "")) {
                if (!this.get("navigateOnHash"))
                    return !1;
                r.hash = o
            }
            return "replace"in r || (r.replace = t === s), this.get("html5") || r.force ? this.fire(i, r) : n && (r.replace ? n.location.replace(t) : n.location = t), !0
        },
        _pjaxBindUI: function() {
            this._pjaxEvents || (this._pjaxEvents = e.one("body").delegate("click", this._onLinkClick, this.get("linkSelector"), this))
        },
        _defNavigateFn: function(e) {
            this[e.replace ? "replace": "save"](e.url), n && this.get("scrollToTop") && setTimeout(function() {
                n.scroll(0, 0)
            }, 1)
        },
        _onLinkClick: function(e) {
            var t, n, r;
            if (e.button !== 1 || e.ctrlKey || e.metaKey)
                return;
            t = e.currentTarget;
            if (t.get("tagName").toUpperCase() !== "A")
                return;
            if (!this._isLinkSameOrigin(t))
                return;
            n = t.get("href"), n && (r = this._navigate(n, {
                originEvent: e
            }), r && e.preventDefault())
        }
    }, s.ATTRS = {
        linkSelector: {
            value: "a." + r,
            writeOnce: "initOnly"
        },
        navigateOnHash: {
            value: !1
        },
        scrollToTop: {
            value: !0
        }
    }, e.PjaxBase = s
}, "3.11.0", {
    requires: ["classnamemanager", "node-event-delegate", "router"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("view", function(e, t) {
    function n() {
        n.superclass.constructor.apply(this, arguments)
    }
    e.View = e.extend(n, e.Base, {
        containerTemplate: "<div/>",
        events: {},
        template: "",
        _allowAdHocAttrs: !0,
        initializer: function(t) {
            t || (t = {}), t.containerTemplate && (this.containerTemplate = t.containerTemplate), t.template && (this.template = t.template), this.events = t.events ? e.merge(this.events, t.events) : this.events, this.after("containerChange", this._afterContainerChange)
        },
        destroy: function(e) {
            return e && (e.remove || e["delete"]) && this.onceAfter("destroy", function() {
                this._destroyContainer()
            }), n.superclass.destroy.call(this)
        },
        destructor: function() {
            this.detachEvents(), delete this._container
        },
        attachEvents: function(t) {
            var n = this.get("container"), r = e.Object.owns, i, s, o, u;
            this.detachEvents(), t || (t = this.events);
            for (u in t) {
                if (!r(t, u))
                    continue;
                s = t[u];
                for (o in s) {
                    if (!r(s, o))
                        continue;
                    i = s[o], typeof i == "string" && (i = this[i]);
                    if (!i)
                        continue;
                    this._attachedViewEvents.push(n.delegate(o, i, u, this))
                }
            }
            return this
        },
        create: function(t) {
            return t ? e.one(t) : e.Node.create(this.containerTemplate)
        },
        detachEvents: function() {
            return e.Array.each(this._attachedViewEvents, function(e) {
                e && e.detach()
            }), this._attachedViewEvents = [], this
        },
        remove: function() {
            var e = this.get("container");
            return e && e.remove(), this
        },
        render: function() {
            return this
        },
        _destroyContainer: function() {
            var e = this.get("container");
            e && e.remove(!0)
        },
        _getContainer: function(e) {
            return this._container || (e ? (this._container = e, this.attachEvents()) : (e = this._container = this.create(), this._set("container", e))), e
        },
        _afterContainerChange: function() {
            this.attachEvents(this.events)
        }
    }, {
        NAME: "view",
        ATTRS: {
            container: {
                getter: "_getContainer",
                setter: e.one,
                writeOnce: !0
            }
        },
        _NON_ATTRS_CFG: ["containerTemplate", "events", "template"]
    })
}, "3.11.0", {
    requires: ["base-build", "node-event-delegate"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("app-base", function(e, t) {
    var n = e.Lang, r = e.Object, i = e.PjaxBase, s = e.Router, o = e.View, u = e.ClassNameManager.getClassName, a = e.config.win, f;
    f = e.Base.create("app", e.Base, [o, s, i], {
        views: {},
        initializer: function(t) {
            function i(t, r) {
                n[r] = e.merge(n[r], t)
            }
            t || (t = {});
            var n = {};
            r.each(this.views, i), r.each(t.views, i), this.views = n, this._viewInfoMap = {}, this.after("activeViewChange", e.bind("_afterActiveViewChange", this)), this.get("serverRouting") || this._pjaxBindUI()
        },
        createView: function(t, i) {
            var s = this.getViewInfo(t), u = s && s.type || o, a, f;
            return a = n.isString(u) ? r.getValue(e, u.split(".")) : u, f = new a(i), this._viewInfoMap[e.stamp(f, !0)] = s, f
        },
        getViewInfo: function(t) {
            return n.isString(t) ? this.views[t] : t && this._viewInfoMap[e.stamp(t, !0)]
        },
        render: function() {
            var t = e.App.CLASS_NAMES, n = this.get("container"), r = this.get("viewContainer"), i = this.get("activeView"), s = i && i.get("container"), o = n.compareTo(r);
            return n.addClass(t.app), r.addClass(t.views), i&&!r.contains(s) && r.appendChild(s), !n.contains(r)&&!o && n.appendChild(r), this
        },
        showView: function(t, r, i, s) {
            var o, u;
            return i || (i = {}), s ? i = e.merge(i, {
                callback: s
            }) : n.isFunction(i) && (i = {
                callback: i
            }), n.isString(t) && (o = this.getViewInfo(t), o && o.preserve && o.instance ? (t = o.instance, this._viewInfoMap[e.stamp(t, !0)] = o) : (t = this.createView(t, r), u=!0)), i.update&&!u && t.setAttrs(r), "render"in i ? i.render && t.render() : u && t.render(), this._set("activeView", t, {
                options: i
            })
        },
        _attachView: function(e, t) {
            if (!e)
                return;
            var n = this.getViewInfo(e), r = this.get("viewContainer");
            e.addTarget(this), n && (n.instance = e), r[t ? "prepend": "append"](e.get("container"))
        },
        _destroyContainer: function() {
            var t = e.App.CLASS_NAMES, n = this.get("container"), r = this.get("viewContainer"), i = n.compareTo(r);
            if (e.one("body").compareTo(n)) {
                this.detachEvents(), n.removeClass(t.app), i ? n.removeClass(t.views) : r.remove(!0);
                return 
            }
            r.remove(!0), i || n.remove(!0)
        },
        _detachView: function(t) {
            if (!t)
                return;
            var n = this.getViewInfo(t) || {};
            n.preserve ? t.remove() : (t.destroy({
                remove: !0
            }), delete this._viewInfoMap[e.stamp(t, !0)], t === n.instance && delete n.instance), t.removeTarget(this)
        },
        _getViewContainer: function(e) {
            return !e&&!this._viewContainer && (e = this._viewContainer = this.create(), this._set("viewContainer", e)), e
        },
        _initHtml5: function() {
            return this.get("serverRouting")===!1?!1 : s.html5
        },
        _isChildView: function(e, t) {
            var n = this.getViewInfo(e), r = this.getViewInfo(t);
            return n && r ? this.getViewInfo(n.parent) === r : !1
        },
        _isParentView: function(e, t) {
            var n = this.getViewInfo(e), r = this.getViewInfo(t);
            return n && r ? this.getViewInfo(r.parent) === n : !1
        },
        _navigate: function(t, n) {
            return this.get("serverRouting") || (n = e.merge({
                force: !0
            }, n)), i.prototype._navigate.call(this, t, n)
        },
        _save: function(t, n) {
            var r;
            return this.get("serverRouting")&&!this.get("html5") ? this._hasSameOrigin(t) ? (a && (r = this._joinURL(t || ""), n ? a.location.replace(r) : a.location = r), this) : (e.error("Security error: The new URL must be of the same origin as the current URL."), this) : s.prototype._save.apply(this, arguments)
        },
        _uiSetActiveView: function(e, t, n) {
            n || (n = {});
            var r = n.callback, i = this._isChildView(e, t), s=!i && this._isParentView(e, t), o=!!n.prepend || s;
            if (e === t)
                return r && r.call(this, e);
            this._attachView(e, o), this._detachView(t), r && r.call(this, e)
        },
        _afterActiveViewChange: function(e) {
            this._uiSetActiveView(e.newVal, e.prevVal, e.options)
        }
    }, {
        ATTRS: {
            activeView: {
                value: null,
                readOnly: !0
            },
            container: {
                valueFn: function() {
                    return e.one("body")
                }
            },
            html5: {
                valueFn: "_initHtml5"
            },
            linkSelector: {
                value: "a"
            },
            serverRouting: {
                valueFn: function() {
                    return e.App.serverRouting
                },
                writeOnce: "initOnly"
            },
            viewContainer: {
                getter: "_getViewContainer",
                setter: e.one,
                writeOnce: !0
            }
        },
        _NON_ATTRS_CFG: ["views"]
    }), e.namespace("App").Base = f, e.App = e.mix(e.Base.create("app", f, []), e.App, !0), e.App.CLASS_NAMES = {
        app: u("app"),
        views: u("app", "views")
    }
}, "3.11.0", {
    requires: ["classnamemanager", "pjax-base", "router", "view"]
});
(function() {
    var a = "flickr-app";
    YUI.add(a, function(g) {
        var f;
        function e() {
            if (!f) {
                f = new g.App({
                    serverRouting: true,
                    linkSelector: false,
                    scrollToTop: false
                });
                f.upgradeClick = d;
                f.saveIfRoutable = c;
                f.replaceIfRoutable = b;
                g.FlickrApp = f;
                if (g.config.flickr.user.admin_user === "hartsell") {
                    F.FlickrApp = f
                }
            }
        }
        function c(h) {
            var i;
            i = f.removeRoot(h);
            if (f.hasRoute(i)) {
                f.save(i);
                return true
            }
            return false
        }
        function b(h) {
            var i;
            i = f.removeRoot(h);
            if (f.hasRoute(i)) {
                f.replace(i);
                return true
            }
            return false
        }
        function d(k) {
            var i, h, j;
            if (k.altKey || k.ctrlKey || k.metaKey || k.shiftKey) {
                return 
            }
            i = k.target.ancestor("a", true);
            h = i && i.getAttribute("href");
            if (h) {
                j = f.removeRoot(h);
                if (f.hasRoute(j)) {
                    k.preventDefault();
                    f.save(j);
                    return true
                }
            }
            return false
        }
        g.FlickrApp = {
            init: e
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
YUI.add("flickr-tooltips", function(a) {
    var q = null, g = null, b = 0, k = null, c = null, n = null;
    function h() {
        q = document.createElement("div");
        g = document.createElement("div");
        q.style.display = "none";
        q.style.zIndex = "200000";
        document.body.appendChild(q);
        q.appendChild(g)
    }
    function m(s, u, v, r, z) {
        var t, B, C, A;
        s = a.one(s);
        z = z || "ToolTip";
        if (!q) {
            h()
        }
        if (s.getStyle("visibility") === "hidden") {
            return 
        }
        q.className = z;
        if (b) {
            if (k === s._node) {
                d();
                return 
            }
            d()
        }
        r = (r === undefined) ? "" : r;
        v += r.length;
        n = document.getElementById(u);
        g.innerHTML = n.innerHTML + r;
        g.style.width = "auto";
        q.style.display = "block";
        q.style.visibility = "hidden";
        if (z !== "ToolTipSmall") {
            w = 150;
            if (v > 200) {
                w = 300
            } else {
                if (v > 100) {
                    w = 200
                }
            }
            g.style.width = w + "px"
        }
        t = q.offsetWidth;
        C = s.getX();
        A = s.getY();
        if (z === "ToolTipSmall") {
            C -= Math.round((t - s.get("offsetWidth")) / 2) - 5;
            if (s._node.tip_placement === "below") {
                A += s.get("offsetHeight") + 10
            } else {
                A -= 22
            }
        } else {
            A += 20
        }
        B = f() - 40;
        if (C + t > B) {
            C = B - t
        }
        C = Math.max(10, C);
        q.style.left = C + "px";
        q.style.top = A + "px";
        b = 1;
        q.style.display = "block";
        q.style.visibility = "visible";
        k = s._node;
        document.onmousedown = l
    }
    function l(s) {
        var r = e(s);
        if (r === k || (o(r) && j(r))) {
            document.onmousedown = function() {}
        } else {
            d()
        }
    }
    function d(r) {
        if (r) {
            if (k !== r) {
                return 
            }
        }
        document.onmousedown = function() {};
        if (!q) {
            return false
        }
        b = 0;
        q.style.display = "none";
        k = "null";
        return false
    }
    function j(r) {
        var s = r;
        while (s) {
            if (s === q) {
                return 1
            }
            s = s.parentNode
        }
        return 0
    }
    function o(r) {
        var s = r;
        while (s) {
            if (s.href) {
                return 1
            }
            s = s.parentNode
        }
        return 0
    }
    function i() {
        if (window.innerHeight) {
            return window.innerHeight
        } else {
            if (document.documentElement.clientHeight) {
                return document.documentElement.clientHeight
            } else {
                if (document.body.clientHeight) {
                    return document.body.clientHeight
                }
            }
        }
    }
    function f() {
        if (window.innerWidth) {
            return window.innerWidth
        } else {
            if (document.documentElement.clientWidth) {
                return document.documentElement.clientWidth
            } else {
                if (document.body.clientWidth) {
                    return document.body.clientWidth
                }
            }
        }
    }
    function e(r) {
        if (r && r.target) {
            return r.target
        }
        if (r && r.srcElement) {
            return r.srcElement
        }
        if (window.event) {
            return window.event.srcElement
        }
        return null
    }
    function p() {
        window.show_tooltip = m
    }
    a.tooltips = {
        init: p
    }
}, "0.0.1", {
    requires: ["node", "event", "event-delegate"]
});
YUI.add("personmenu-transjax", function(b) {
    var a = {
        svn_bump: "",
        error_timeout: 'Uh oh! There was a problem contacting the server.',
        error_getting: 'Uh oh! There was a problem getting data from the server.'
    };
    b.transjax.add("personmenu", a)
}, "0.0.1", {
    requires: ["transjax-base"]
});
YUI.add("personmenu-rapid", function(a) {
    a.on("personmenu:show", function(b) {
        if (b) {
            a.rapidTracker.addModules(b)
        }
    })
}, "0.0.1", {
    requires: F.config.modules["personmenu-rapid"].requires || []
});
YUI.add("personmenu", function(c) {
    var u, H, o, y, d, Q, A, x, F, T;
    var S = c.config.flickr.flags.enable_photo_page_icon_lite_sidebar ? 70: 48;
    function O(V) {
        if (!O.is_init) {
            H = {
                open_on_icon_click: false,
                get_data_on_arrow_hover: true,
                show_delay: 100,
                hide_delay: 500,
                spinner_delay: 200,
                io_timeout: 10000,
                io_cache_ttl: 120000,
                zeus: false
            };
            if (!c.Lang.isObject(V)) {
                V = {}
            }
            H = c.mix(V, H);
            u = c.one("body");
            if (c.config.flickr.user.nsid || c.config.flickr.flags.enable_grease) {
                u.on("mouseover", b);
                u.on("mouseout", b);
                u.on("click", t);
                c.on("resize", t);
                c.on("contact_changer:change", L);
                c.on("contact_changer:open", M);
                c.on("PageContext:set", g)
            }
            O.is_init = true
        }
    }
    function h(W) {
        var V, X;
        if (!h.cache) {
            h.cache = new c.CacheSimple()
        }
        V = h.cache;
        if (c.Lang.isString(W)) {
            X = V.retrieve(W);
            if (X && X.response && (new Date()).getTime() < X.payload.expires) {
                return X.response
            } else {
                return null
            }
        } else {
            return V
        }
    }
    function f(V, W) {
        if (!c.Lang.isString(V)) {
            return false
        }
        h().add(V, W, {
            expires: (new Date()).getTime() + H.io_cache_ttl
        });
        return true
    }
    function n(V) {
        f(V, null)
    }
    function B(V) {
        return (function() {
            var W;
            return function() {
                var Z, aa, Y, X;
                if (!W || arguments.length !== W.length) {
                    Z = true
                } else {
                    Y = Array.prototype.slice.call(arguments);
                    X = Array.prototype.slice.call(W);
                    for (aa in Y) {
                        if (Y.hasOwnProperty(aa)) {
                            if (Y[aa] !== X[aa]) {
                                Z = true;
                                break
                            }
                        }
                    }
                }
                if (Z) {
                    V.apply(this, arguments)
                }
                W = arguments
            }
        }())
    }
    function l(V) {
        if (c.Lang.isString(V)) {
            V = c.one(V)
        }
        if (!V instanceof c.Node) {
            return false
        }
        return V
    }
    function b(V) {
        if (!V.target.hasClass("personmenu-trigger")) {
            return 
        }
        switch (V.type) {
        case"mouseover":
            if ((d || A) && Q) {
                return 
            }
            C(V.target);
            break;
        case"mouseout":
            N();
            break
        }
    }
    function L(V) {
        n(V.nsid)
    }
    function z(V) {
        switch (V.type) {
        case"mouseover":
            U();
            if (H.get_data_on_arrow_hover) {
                if (c.one(".personmenu .personmenu-arrow") && (V.target.test(".personmenu .personmenu-arrow") || c.one(".personmenu .personmenu-arrow").contains(V.target))) {
                    if (!d&&!h(o)) {
                        P(o)
                    }
                }
            }
            break;
        case"mouseout":
            if (!d&&!A) {
                I()
            }
            break;
        case"click":
            if (c.one(".personmenu .personmenu-arrow") && (V.target.test(".personmenu .personmenu-arrow") || c.one(".personmenu .personmenu-arrow").contains(V.target))) {
                V.preventDefault();
                w()
            } else {
                if (H.zeus && (c.one(".personmenu .personmenu-more") && (V.target.test(".personmenu .personmenu-more") || c.one(".personmenu .personmenu-more").contains(V.target)))) {
                    if (!d) {
                        V.preventDefault();
                        V.halt();
                        k()
                    }
                } else {
                    if (H.open_on_icon_click && c.one(".personmenu .personmenu-icon-link") && (V.target.test(".personmenu .personmenu-icon-link") || c.one(".personmenu .personmenu-icon-link").contains(V.target))) {
                        if (!d) {
                            V.preventDefault();
                            k()
                        }
                    } else {
                        if (c.one(".personmenu .personmenu-relationship-change") && (V.target.test(".personmenu .personmenu-relationship-change") || c.one(".personmenu .personmenu-relationship-change").contains(V.target))) {
                            if (c.config.modules["contact-changer"]) {
                                V.preventDefault();
                                s()
                            }
                            c.use("contact-changer", function(X) {
                                i();
                                X.contact_changer.show(o)
                            })
                        } else {
                            if (c.one(".personmenu .photos-trigger") && V.target.test(".personmenu .photos-trigger")) {
                                var W = V.target.ancestor("li").one(".photos");
                                if (W) {
                                    V.preventDefault();
                                    W.toggleClass("photos-open")
                                }
                            } else {
                                c.fire("personmenu:navigate", {
                                    id: p().get("id"),
                                    url: V.target.get("href")
                                })
                            }
                        }
                    }
                }
            }
            break
        }
    }
    function t(V) {
        if (V.type === "click") {
            if (H.open_on_icon_click && V.target.hasClass("personmenu-trigger")) {
                V.preventDefault();
                V.halt();
                k(V.target)
            } else {
                if (Q&&!c.one(".personmenu").contains(V.target)) {
                    M()
                }
            }
        }
        if (V.type === "resize") {
            if (Q) {
                m(y)
            }
        }
    }
    function g(V) {
        if (V.newVal !== "") {
            M()
        }
    }
    function U() {
        if (F) {
            window.clearTimeout(F);
            F = null
        }
    }
    function N() {
        if (x) {
            window.clearTimeout(x);
            x = null
        }
    }
    function C(V) {
        U();
        N();
        if (Q&&!H.zeus) {
            r(V)
        } else {
            x = window.setTimeout(function() {
                x = null;
                r(V)
            }, H.show_delay)
        }
    }
    function I() {
        if (!F) {
            F = window.setTimeout(function() {
                F = null;
                M()
            }, H.hide_delay)
        }
    }
    function j(W) {
        var V;
        W = l(W);
        V = W.get("data-defer-src") || W.get("src");
        return V
    }
    function R(W) {
        var V;
        W = l(W);
        V = a(W);
        return V ? "/photos/" + V : false
    }
    function a(X) {
        var W, V;
        W = j(X);
        V = W.split("#")[1];
        return J(V) ? V : false
    }
    function J(V) {
        return (/^[0-9]+@N[0-9]{2}$/).test(V)
    }
    function p() {
        var V;
        if (!p.menu) {
            if (H.zeus) {
                V = '<div class="personmenu personmenu-hidden personmenu-zeus"><div class="personmenu-hd"><a data-track="icon-link" href="" class="personmenu-icon-link rapidnofollow"><img class="personmenu-icon"></a><span class="personmenu-spinner"></span><span class="personmenu-name"></span></div><div class="personmenu-bd"></div></div>'
            } else {
                V = '<div class="personmenu personmenu-hidden"><div class="personmenu-hd rapidnofollow"><a data-track="icon-link" href="" class="personmenu-icon-link rapidnofollow"><span class="personmenu-spinner"></span><img class="personmenu-icon"></a><a data-track="show-menu" href="" class="rapidnofollow personmenu-arrow" role="button"><span></span></a></div><span class="personmenu-hd-shadow"></span><div class="personmenu-bd"></div><span class="personmenu-bd-border-blocker"></span></div>'
            }
            c.one("body").append(V);
            p.menu = c.one(".personmenu");
            p.menu.on("mouseover", z);
            p.menu.on("mouseout", z);
            p.menu.on("click", z)
        }
        return p.menu
    }
    function w() {
        if (d) {
            r()
        } else {
            k()
        }
    }
    function q(V, W) {
        if (V) {
            p().one(".personmenu-icon-link").set("href", V)
        }
        if (W) {
            p().one(".personmenu-icon").set("src", W)
        }
    }
    q = B(q);
    function s() {
        A = true;
        p().addClass("personmenu-spinning");
        if (H.zeus) {
            if (p().one(".personmenu-spinner")) {
                p().one(".personmenu-spinner").addClass("personmenu-spinner-on")
            }
        } else {
            T = window.setTimeout(function() {
                p().one(".personmenu-spinner").addClass("personmenu-spinner-on");
                T = null
            }, H.spinner_delay)
        }
    }
    function i() {
        A = false;
        if (T) {
            window.clearTimeout(T);
            T = null
        }
        p().removeClass("personmenu-spinning");
        if (p().one(".personmenu-spinner")) {
            p().one(".personmenu-spinner").removeClass("personmenu-spinner-on")
        }
    }
    function G() {
        Q = true;
        p().removeClass("personmenu-hidden")
    }
    function M() {
        Q = false;
        p().addClass("personmenu-hidden")
    }
    function m(V) {
        var W, X;
        W = V.getXY();
        X = H.zeus ? 8 : 6;
        W[0] -= parseInt((S - V.get("width")) / 2, 10) + X;
        W[1] -= parseInt((S - V.get("height")) / 2, 10) + X;
        p().setStyle("position", "absolute");
        p().setStyle("left", W[0] + "px");
        p().setStyle("top", W[1] + "px")
    }
    function D(V) {
        var X, Z;
        X = R(V);
        Z = j(V);
        q(X, Z);
        if (H.zeus) {
            var W = a(V);
            var Y = '<ul><li><a href="/people/' + W + '" class="personmenu-item">Profile</a></li><li><a href="' + X + '" class="personmenu-item">Photostream</a></li><li class="personmenu-sep-bottom"><a href="/photos/' + W + '/favorites" class="personmenu-item">Favorites</a></li><li><a href="" class="personmenu-item personmenu-more">More...</a></li></ul>';
            p().one(".personmenu-bd").set("innerHTML", Y);
            p().one(".personmenu-name").set("innerHTML", W)
        }
    }
    D = B(D);
    function P(V) {
        var Y, X, W;
        P.in_progress_nsids = P.in_progress_nsids || {};
        Y = P.in_progress_nsids;
        X = {
            success: function(ac, ab, Z) {
                var aa = ab.responseText;
                if (aa) {
                    f(Z, aa)
                }
                if (A && o === Z) {
                    E(Z)
                }
            },
            failure: function(ab, aa, Z) {
                if (A && o === Z) {
                    if (aa.statusText === "timeout") {
                        e(c.transjax.get("personmenu", "error_timeout"))
                    } else {
                        e(c.transjax.get("personmenu", "error_getting"))
                    }
                }
            },
            end: function(aa, Z) {
                delete Y[Z]
            }
        };
        W = "/personmenu_fragment.gne?nsid=" + V + "&magic_cookie=" + c.config.flickr.magic_cookie;
        if (H.zeus) {
            W = W + "&zeus=1"
        }
        if (Y[V]) {
            return false
        }
        Y[V] = V;
        c.io(W + "&cachebust=" + (new Date()).getTime(), {
            timeout: H.io_timeout,
            on: X,
            "arguments": V
        })
    }
    function E(V) {
        var W = h(V);
        if (W) {
            if (H.zeus) {
                p().set("innerHTML", W)
            } else {
                p().one(".personmenu-bd").set("innerHTML", W)
            }
            i();
            var X = y.getAttribute("data-menu-id");
            if (X) {
                p().set("id", X)
            } else {
                X = false
            }
            c.fire("personmenu:show", X);
            v()
        } else {
            s();
            P(V)
        }
    }
    function e(W) {
        var V = '<div class="alert">' + W + "</div>";
        p().one(".personmenu-bd").set("innerHTML", V);
        i();
        v()
    }
    function r(V) {
        var W;
        V = l(V) || y;
        if (!V) {
            return false
        }
        y = V;
        W = a(V);
        if (!W) {
            return false
        }
        o = W;
        i();
        K();
        D(V);
        m(V);
        G();
        if (H.zeus&&!h(W)) {
            P(W)
        }
        return true
    }
    function k(V) {
        var W;
        V = l(V) || y;
        if (!V) {
            return false
        }
        y = V;
        W = a(V);
        if (!W) {
            return false
        }
        o = W;
        r(V);
        E(W);
        return true
    }
    function K() {
        p().addClass("personmenu-closed");
        p().removeClass("personmenu-open");
        d = false
    }
    function v() {
        p().removeClass("personmenu-closed");
        p().addClass("personmenu-open");
        d = true;
        p().one(".personmenu-bd").vis.animScrollIntoView()
    }
    c.personmenu = {
        init: O,
        showClosed: r,
        showOpen: k,
        hide: M
    }
}, "0.0.1", {
    requires: ["node", "anim", "anim-scroll", "node-visibility", "event-custom", "cache-simple", "io-base", "personmenu-transjax", "personmenu-css"],
    optional: ["contact-changer"]
});
(function() {
    var a = "flickr";
    YUI.add(a, function(c) {
        function b() {
            c.FlickrApp.init();
            if (c.config.flickr.flags.enable_grease || c.config.flickr.user && c.config.flickr.user.nsid) {
                c.personmenu.init()
            }
            if (c.config.flickr.sharing.use_share_this_v3) {
                c.use("share-this-v3-menu", function(h) {
                    var i = (h.config.flickr.flags.enable_2013_photo_page && h.config.flickr.page_type === "photo") ? "br": "t";
                    var f, e;
                    (function g() {
                        function l(m) {
                            var n = {
                                x: 0,
                                y: 0
                            };
                            while (m) {
                                n.x += m.offsetLeft;
                                n.y += m.offsetTop;
                                m = m.offsetParent
                            }
                            return n
                        }
                        function j(m) {
                            if (document.all) {
                                return document.all[m]
                            } else {
                                if (document.getElementById) {
                                    return document.getElementById(m)
                                } else {
                                    return null
                                }
                            }
                        }
                        var k = l(document.getElementById("share-this-v3-more-button"));
                        if (k.y < 411) {
                            i = "tr";
                            f = true;
                            e = true
                        }
                    })();
                    h.shareThisV3Menu.init({
                        arrowPosition: i,
                        overrideRefresh: f,
                        scroll: e
                    })
                });
                if (c.config.flickr.sharing.share_immediately_service) {
                    c.use("share-this-v3-dialog", function(e) {
                        var f = e.config.flickr.sharing.share_immediately_service;
                        e.shareThisV3Dialog.share_by_service(f.service_id, f.service_type_id)
                    })
                }
            }
            if (c.config.flickr.flags.enable_tumblr_trackr) {
                c.use("tumblr-trackr", function(e) {
                    e.tumblrTrackr.init()
                })
            }
            if (c.config.flickr.flags.view_count_on_visible) {
                c.ViewCount.onVisible("img.view-count-on-visible")
            }
            c.tooltips.init();
            if (window.devicePixelRatio > 1 && document.cookie.indexOf("rtna=1")===-1) {
                var d = new Date();
                d.setTime(d.getTime() + 2592000000);
                document.cookie = "rtna=1; expires=" + d.toGMTString() + "; path=/"
            }
            if (c.config.flickr.flags.enable_sitekey_fetcher) {
                window.setInterval(function() {
                    if (c.config.flickrAPI && c.config.flickrAPI.api_key) {
                        c.use("gallery-flickr-api", function(f) {
                            var h, e, g;
                            h = "flickr.site.getKey";
                            e = {};
                            g = {
                                success: function(i) {
                                    if (i && i.data && i.data.key && i.data.key._content && f.Lang.isString(i.data.key._content)) {
                                        f.config.flickrAPI.api_key = i.data.key._content
                                    }
                                },
                                failure: function(i) {}
                            };
                            f.flickrAPI.callMethod(h, e, g)
                        })
                    }
                }, 7 * 60 * 60 * 1000)
            }
        }
        c.flickr = b
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}()); /*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("json-stringify", function(e, t) {
    var n = ":", r = e.config.global.JSON;
    e.mix(e.namespace("JSON"), {
        dateToString: function(e) {
            function t(e) {
                return e < 10 ? "0" + e : e
            }
            return e.getUTCFullYear() + "-" + t(e.getUTCMonth() + 1) + "-" + t(e.getUTCDate()) + "T" + t(e.getUTCHours()) + n + t(e.getUTCMinutes()) + n + t(e.getUTCSeconds()) + "Z"
        },
        stringify: function() {
            return r.stringify.apply(r, arguments)
        },
        charCacheThreshold: 100
    })
}, "3.11.0", {
    requires: ["yui-base"]
});
YUI.add("share-menu-transjax", function(b) {
    var a = null;
    b.transjax.add("share-menu", {
        menu_fragment_error: "Sorry, there was a problem loading the sharing menu.",
        services_fragment_error: "Sorry, we couldn't find any services for you to share with.",
        grab_the_link_fragment_error: 'There was an error',
        share_blog_no_blogs: "Sorry, we couldn't find any blogs for your account.",
        share_blog_choose_blog: "Choose a blog below:",
        sharing_email_error: "Oops! That doesn't look like a real email address. Try again?",
        sharing_no_contact: "Hmm... we couldn't find %s in your contacts list. Do you want to try entering their email address instead?",
        sharing_no_member: "Hmm... we couldn't find %s on Flickr. Do you want to try entering their email address instead?",
        uber_contact_list_default_text: "screen name, real name, or email",
        uber_contact_list_max_results: "Showing [results_shown] of [total_results] results. <a href='[url]'>See all...</a>",
        uber_contact_list_no_realname: "No real name given",
        generating_guest_pass_msg: "Generating Guest Pass...",
        sharing_initial_instructions: "Enter email addresses or a contact's screen name",
        sharing_instructions_one_added: "1 recipient so far... add another?",
        sharing_instructions_many_added: "%s recipients so far... add another?",
        sharing_msg_over_limit: "Please limit your message to %s characters or fewer.",
        spam_url_warning: "Hey! A URL or phrase that you tried to post has been used for abuse on Flickr before. (ex. SPAM) If that's you, stop it! If that's not you, sorry, but you can't post it.",
        spam_share_warning: "Slow down, partner! To protect the Flickr community against spam, we place a few restrictions on new accounts. You can read more about these restrictions in <a href=\'/help/website/#870679\'>this FAQ</a>.",
        status_sending_msg: "Sending message..."
    })
}, "0.0.1", {
    requires: ["transjax-base"]
});
(function() {
    var a = "flanal";
    YUI.add(a, function(c) {
        function b(e, g, d) {
            d = d || {};
            var f = "/flanal_event.gne?target=" + e + "&title=" + g + ("value" in d ? "&value=" + d.value : "") + ("units" in d ? "&units=" + d.units : "") + "&rand=" + Math.random();
            if (d.sync) {
                c.use("io", function(h) {
                    h.io(f, {
                        sync: true
                    })
                })
            } else {
                (new Image()).src = f
            }
            c.log(e + " (" + g + ")", "info", "flanal")
        }
        c.flanal = b
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
YUI.add("share-this-v3-dialog-transjax", function(a) {
    a.transjax.add("share-this-v3-dialog", {
        try_again_later: "D'oh! That didn't work. Please try again later.",
        following_flickr_waiting: "Following...",
        following_flickr_success: "Awesome, you're following <a href='http://twitter.com/Flickr' target='_blank'>@Flickr</a>!"
    })
}, "0.0.1", {
    requires: ["transjax-base"]
});
YUI.add("share-this-v3-service-email-transjax", function(a) {
    a.transjax.add("share-this-v3-service-email", {
        email_field_default_text: "screen name, real name, or email",
        send_an_email: "Send an email to this address",
        no_realname: "No real name given",
        sharing_email_error: "Oops! That doesn't look like a real email address. Try again?",
        sharing_no_contact: "Hmm... we couldn't find %s in your contacts list. Do you want to try entering their email address instead?"
    })
}, "0.0.1", {
    requires: ["transjax-base"]
});
YUI.add("share-this-v3-service-twitter-transjax", function(a) {
    a.transjax.add("share-this-v3-service-twitter", {
        over_limit_1: "%s whoa!",
        over_limit_5: "%s too much!",
        over_limit_10: "%s less is more!",
        over_limit_20: "%s way over!",
        over_limit_30: "%s booooring!"
    })
}, "0.0.1", {
    requires: ["transjax-base"]
});
YUI.add("share-this-v3-services", function(e) {
    var d = {};
    function c(f) {
        d[f.name] = f
    }
    function b() {
        return d
    }
    function a(g) {
        var f, h;
        for (f in d) {
            if (d.hasOwnProperty(f)) {
                if (d[f].id === g) {
                    h = true;
                    break
                }
            }
        }
        return h ? d[f] : undefined
    }
    e.shareThisV3Services = {
        add_service: c,
        get_services: b,
        get_service: a
    }
}, "0.0.1", {
    requires: []
});
YUI.add("share-this-v3-service-blogger", function(b) {
    var a = b.shareThisV3Services;
    if (!a) {
        b.log("Y.shareThisV3Services null/undefined?")
    } else {
        a.add_service({
            name: "blogger",
            id: 3,
            setup_url: "/share.gne?action=setup&service_type_id=3",
            use_popup: true,
            popup_params: {
                width: 780,
                height: 480
            },
            default_text: null,
            on_window_closed: function() {
                var c = b.shareThisV3Dialog.get_current_service();
                b.log("auth (maybe) complete for " + c.name);
                b.shareThisV3Dialog.fetch_sharing_fragment({
                    on_complete: function() {
                        b.shareThisV3Dialog.refresh_dialog_state();
                        b.fire("shareDialog:servicesAuthComplete")
                    }
                })
            },
            on_show_screen: {
                setup: function() {
                    b.log("blogger: setup screen")
                },
                setup_choose_a_blog: function() {
                    b.log("blogger: setup_choose_a_blog()")
                },
                share: function() {
                    b.log("blogger: share screen");
                    var d = b.shareThisV3Dialog.get_current_service(), e = b.shareThisV3Dialog.service_helpers;
                    b.global_dialog.show();
                    var c = b.one("#blogger-text");
                    d.default_text = c.getAttribute("data-default-text");
                    c.on("focus", b.shareThisV3Dialog.service_helpers.maybeClearText);
                    c.on("blur", b.shareThisV3Dialog.service_helpers.maybeRestoreText);
                    e.addDescriptionEvents(b.one("#share-this-v3-dialog .add-object-description"), c)
                },
                share_confirmblog: function() {
                    b.log("share_confirmblog()");
                    b.global_dialog.show()
                }
            },
            on_auth_failed: function() {
                b.log("on_auth_failed");
                set_dialog_state("state-setup state-setup-error");
                b.global_dialog.reposition()
            },
            on_submit_form: {
                share: function() {
                    b.shareThisV3Dialog.service_helpers.maybeClearText({
                        target: b.one("#blogger-text")
                    })
                },
                share_chooseblog: function() {
                    b.global_dialog.hide();
                    b.global_dialog.set_loading(true)
                }
            },
            misc: {
                maybe_highlight_endpoint: function(f) {
                    var g = f.target, h = f.target.getAttribute("data-default-url"), c = f.target.get("value"), d = c.indexOf(h);
                    if (d!==-1) {
                        b.shareThisV3Dialog.service_helpers.setSelectionRange(b.Node.getDOMNode(g), d, d + h.length)
                    }
                }
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-service-email", function(c) {
    c.config.flickr.user.apiURL = null;
    var q = c.shareThisV3Services, o, h = {}, v = false, p = "#share-form-email", k = false;
    function a() {
        if (!v) {
            c.shareThisV3Dialog.service_helpers.disableForm(c.one(p));
            v = true
        }
    }
    function g() {
        if (v) {
            c.shareThisV3Dialog.service_helpers.enableForm(c.one(p));
            v = false
        }
    }
    function u() {
        var x = c.shareThisV3Dialog.service_helpers, z = x.is_guestpass_required(), y = x.is_guestpass_automatic(), w = s(), A = false;
        if (!z || y) {
            if (w) {
                g()
            } else {
                a()
            }
        } else {
            if (!w) {
                a()
            } else {
                A = x.is_guestpass_chosen();
                if (A) {
                    g()
                } else {
                    a()
                }
            }
        }
    }
    function s() {
        var w = c.one("#sharing-email-input"), y = w.get("value"), x = (m(y) || j());
        return x
    }
    function n() {
        var z = b();
        var x = c.shareThisV3Dialog.get_form_params();
        z = c.merge(z, x);
        var w = c.one("#email-text");
        var y = w.get("value");
        if (y && y !== w.getAttribute("data-default-text")) {
            z.message_text = y
        }
        return z
    }
    if (!q) {
        c.log("Y.shareThisV3Services null/undefined?")
    } else {
        q.add_service({
            name: "email",
            id: 130,
            default_text: null,
            on_show_screen: {
                share: function() {
                    var z = c.shareThisV3Dialog.service_helpers, x = c.one("#sharing-email-input"), y = c.one("#email-text"), w = c.shareThisV3Dialog.get_current_service();
                    k = false;
                    w.default_text = y.getAttribute("data-default-text");
                    y.on("focus", z.maybeClearText);
                    y.on("blur", z.maybeRestoreText);
                    c.one("#to-flickr").set("value", "");
                    c.one("#to-email").set("value", "");
                    h = {};
                    add_by_email = function(A) {
                        if (r()) {
                            i();
                            c.global_dialog.reposition()
                        }
                    };
                    o = new c.BoSelecta3(x, {
                        showSubtitle: true,
                        showIcon: true,
                        maxResultsDisplayed: 4,
                        includeAddressBook: !!(c.config.flickr.user.nsid),
                        searchOnUsername: true,
                        searchOnRealname: true,
                        searchOnEmail: true,
                        searchOnPathAlias: true,
                        showTotalResults: true,
                        allowNoContacts: true,
                        defaultText: c.transjax.get("share-this-v3-service-email", "email_field_default_text"),
                        purgeFormOnDataFailure: false,
                        focusOnFetch: true,
                        fetchDataImmediately: true,
                        selectFirstItem: true,
                        allowFormSubmit: false,
                        hideNoContactMessage: true,
                        enableGlobalSearch: false,
                        zeroResultsMessages: [{
                            message: "<span>" + c.transjax.get("share-this-v3-service-email", "send_an_email") + "</span>",
                            className: "bs-send-email-link",
                            onclick: add_by_email,
                            validator: function(A) {
                                var B = A.getQuery(), C = (t(B) && m(B)&&!A.searchingGlobally);
                                return C
                            }
                        }
                        ],
                        apiURL: (c.config.flickr.user.nsid ? x.getAttribute("data-api-url") : null)
                    });
                    o.on("BoSelecta:resultSelect", d);
                    o.on("BoSelecta:resultClick", d);
                    c.one(p).one('.Butt[type="submit"]').on("mousedown", function(A) {
                        k = true
                    });
                    c.one(p).on("submit", i);
                    if (z.is_guestpass_required()) {
                        c.all('#guestpass-box input[type="checkbox"]').on("click", u)
                    }
                    v = false;
                    a();
                    c.one("#sharing-email-input").on("keyup", u);
                    c.delegate("click", f, "#bo-selecta-results-box", "a.delete-x")
                }
            },
            on_submit_form: {
                share: function(w) {
                    var x = b();
                    c.one("#to-flickr").set("value", x.to_flickr || null);
                    c.one("#to-email").set("value", x.to_email || null);
                    if (c.config.flickr.flags.enable_grease&&!c.config.flickr.user.nsid) {
                        w.preventDefault();
                        return false
                    } else {
                        c.shareThisV3Dialog.service_helpers.maybeClearText({
                            target: c.one("#email-text")
                        })
                    }
                }
            }
        })
    }
    function d(w) {
        if (typeof h[(w.n ? w.n : w.e)] === "undefined") {
            h[(w.n ? w.n : w.e)] = w;
            o.skipContact(w);
            if (F.config.flickr.is_secure) {
                w.i = w.i.replace("http:", "https:")
            }
            var x = c.one("#bo-selecta-results-box"), z, y;
            liEl = document.createElement("li");
            liEl.innerHTML = (w.n) ? '<p nsid="' + w.n + '"><span class="name"><img src="' + w.i + '" width="24" height="24" class="BuddyIconX" nsid="' + w.n + '" id="pt-icon-' + w.n + '"><strong class="username">' + o._sanitizeString(w.u) + '</strong><span class="realname">' + (w.r ? o._sanitizeString(w.r) : c.transjax.get("share-this-v3-service-email", "no_realname")) + '</span></span><span class="controls"><a href="#" class="delete-x" class="pt-remove" id="pt-remove-' + w.n + '"><span>[x]</span></a></span></p>' : '<p email="' + o._sanitizeString(w.e) + '" class="email-contact"><span class="name"><img src="' + w.i + '" width="16" height="16" class="BuddyIconX bs-email-icon"><strong class="username email">' + o._sanitizeString(w.e) + '</strong></span><span class="controls"><a href="#" class="delete-x" class="pt-remove" id="pt-remove-' + o._sanitizeString(w.e) + '"><span>[x]</span></a></span></p>';
            x.appendChild(liEl);
            x.addClass("not-empty");
            if (c.is_ie) {
                x.setStyle("display", "block")
            }
            y = x.all("li.last");
            if (y) {
                y.each(function() {
                    this.removeClass("last")
                })
            }
            c.one(liEl).addClass("last");
            if (x.getElementsByTagName("li").length > 5) {
                x.addClass("overflow")
            }
            x.set("scrollTop", x.get("scrollHeight"))
        }
        setTimeout(function() {
            o.clear(true)
        }, 1);
        u();
        c.global_dialog.reposition();
        if (typeof e !== "undefined") {
            e.preventDefault()
        }
        return false
    }
    function i(B) {
        var C = o.getQuery(), x = c.one("#sharing-email-input"), A, w, z, y;
        w = C.replace(/([\;\s<>]+)/g, ",");
        A = w.split(",");
        if (!A.length) {
            return l(false, B)
        } else {
            for (z = 0, y = A.length; z < y; z++) {
                if (A[z]) {
                    x.set("value", A[z]);
                    l(true, B)
                }
            }
            x.set("value", "")
        }
    }
    function l(z, B) {
        var A = o.getQuery();
        if (r(z)) {
            var x = {
                e: o.getQuery()
            };
            if (typeof h[x.e] === "undefined") {
                h[x.e] = x;
                var y = c.one("#bo-selecta-results-box");
                var w = document.createElement("li");
                w.innerHTML = '<p email="' + o._sanitizeString(x.e) + '" class="email-contact"><span class="name"><img src="/images/icon_unread.gif" width="16" height="16" class="BuddyIconX bs-email-icon"><strong class="username email">' + o._sanitizeString(x.e) + '</strong></span><span class="controls"><a href="#" class="pt-remove delete-x" id="pt-remove-' + o._sanitizeString(x.e) + '"><span>[x]</span></a></span></p>';
                y.appendChild(w);
                y.addClass("not-empty");
                if (c.UA.ie) {
                    y.setStyle("display", "block")
                }
                y.all("li").removeClass("last");
                c.one(w).addClass("last");
                if (y.all("li").size() > 5) {
                    y.addClass("overflow")
                }
                y.set("scrollTop", y.get("scrollHeight"));
                o.clear(true)
            }
            u()
        }
        if (B && A&&!k) {
            B.halt();
            return false
        } else {
            if (c.config.flickr.flags.enable_grease&&!c.config.flickr.user.nsid) {
                c.use("grease", function(D) {
                    var C = n();
                    D.grease.shareV3(C);
                    D.grease.authenticate(function(E) {})
                });
                if (B) {
                    B.preventDefault()
                }
                return false
            }
        }
    }
    function m(w) {
        var x = /[a-z_0-9\.\-\+\']+@[a-z_0-9\.\-\+]+\.[a-z]/i;
        return x.test(w)
    }
    function t(x) {
        var w = /\.|@/;
        return w.test(x)
    }
    function r(w) {
        var x = o.getQuery();
        if (x !== "") {
            if (!m(x)) {
                var y = c.config.flickr.flags.enable_grease ? c.transjax.get("share-this-v3-service-email", "sharing_email_error", x): c.transjax.get("share-this-v3-service-email", "sharing_no_contact", x);
                if (t(x)) {
                    y = c.transjax.get("share-this-v3-service-email", "sharing_email_error")
                }
                if (!w) {
                    alert(y)
                }
                return false
            } else {
                return true
            }
        }
        return false
    }
    function f(A) {
        A.halt();
        var z = A.target.ancestor("p");
        var B = z.getAttribute("nsid") ? z.getAttribute("nsid"): z.getAttribute("email");
        o.unskipContact(h[B]);
        delete h[B];
        var w = z.get("parentNode");
        var x = c.one("#bo-selecta-results-box");
        x.removeChild(w);
        o.clear(true);
        var y = x.all("li");
        if (y.size() === 0) {
            x.removeClass("not-empty");
            if (c.is_ie) {
                x.setStyle("display", "")
            }
        } else {
            y.removeClass("last");
            if (y.size()) {
                x.all("li:list-child").addClass("last")
            }
        }
        if (y.size() <= 5) {
            x.removeClass("overflow")
        }
        u()
    }
    function b() {
        var z = {}, x = [], y = [], w;
        for (w in h) {
            if (h.hasOwnProperty(w)) {
                if (h[w].n) {
                    y.push(h[w].n)
                } else {
                    if (h[w].e) {
                        x.push(h[w].e)
                    }
                }
            }
        }
        if (x.length > 0) {
            z.to_email = x.join()
        }
        if (y.length > 0) {
            z.to_flickr = y.join()
        }
        return z
    }
    function j() {
        var w = b();
        return (w.to_email || w.to_flickr)
    }
}, "0.0.1", {
    requires: F.config.modules["share-this-v3-service-email"].requires || [],
    optional: F.config.modules["share-this-v3-service-email"].optional || []
});
YUI.add("share-this-v3-service-facebook", function(a) {
    var g = a.shareThisV3Services;
    var b, c, l, o, h, k, e;
    var m = false;
    var n = "#share-form-facebook";
    var f = "#guestpass-box";
    var d = false;
    function j() {
        c = a.one(n);
        l = a.one(f);
        o = a.shareThisV3Dialog.service_helpers.is_guestpass_required();
        h = a.shareThisV3Dialog.service_helpers.is_guestpass_automatic();
        k = null;
        e = 0;
        if (!o || h) {
            if (m) {
                a.shareThisV3Dialog.service_helpers.enableForm(c);
                m = false
            }
            return false
        }
        if (l) {
            k = l.all('input[type="checkbox"]');
            if (k) {
                k.each(function() {
                    if (this.get("checked")) {
                        e++
                    }
                });
                if (e && m) {
                    a.shareThisV3Dialog.service_helpers.enableForm(c);
                    m = false
                } else {
                    if (!e&&!m) {
                        a.shareThisV3Dialog.service_helpers.disableForm(c);
                        m = true
                    }
                }
            } else {}
        }
    }
    function i() {
        a.all(f + ' input[type="checkbox"]').on("change", j)
    }
    if (!g) {
        a.log("Y.shareThisV3Services null/undefined?")
    } else {
        g.add_service({
            name: "facebook",
            id: 128,
            setup_url: "/share.gne?action=setup&service_type_id=128",
            use_popup: true,
            signed_out_url: "/share.gne?action=signedOutShare&service_type_id=128",
            popup_params: {
                width: 780,
                height: 500
            },
            default_text: null,
            setup_only: false,
            share_signed_out: function(s) {
                var q = a.shareThisV3Dialog.get_current_service(), p = "&" + a.shareThisV3Dialog.get_share_this_data(true).join("&"), r = q.signed_out_url + "&" + p;
                a.shareThisV3Dialog.webCirca1999(r, {
                    width: 640,
                    height: 320
                });
                return false
            },
            on_window_closed: function() {
                var p = a.shareThisV3Dialog.get_current_service();
                a.log("auth (maybe) complete for " + p.name);
                if (p.setup_only) {
                    a.shareThisV3Dialog.service_helpers.setupOnly()
                } else {
                    a.shareThisV3Dialog.fetch_sharing_fragment()
                }
            },
            on_auth_failed: function() {
                a.log("D'oh, auth did not work, or something. Retry?")
            },
            on_show_screen: {
                share: function() {
                    if (m) {
                        m = false;
                        a.shareThisV3Dialog.service_helpers.enableForm(c)
                    }
                    var p = a.one("#facebook-text");
                    var q = a.shareThisV3Dialog.get_current_service();
                    q.default_text = p.getAttribute("data-default-text");
                    p.on("focus", a.shareThisV3Dialog.service_helpers.maybeClearText);
                    p.on("blur", a.shareThisV3Dialog.service_helpers.maybeRestoreText);
                    i();
                    j();
                    a.log("on_show_screen.share()")
                },
                error: function() {
                    a.fire("shareDialog:servicesAuthComplete")
                }
            },
            on_submit_form: {
                share: function() {
                    a.shareThisV3Dialog.service_helpers.maybeClearText({
                        target: a.one("#facebook-text")
                    })
                }
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-service-livejournal", function(d) {
    var b = d.shareThisV3Services;
    var c, e, a;
    if (!b) {
        d.log("Y.shareThisV3Services null/undefined?")
    } else {
        b.add_service({
            name: "livejournal",
            id: 5,
            use_setup_dialog: true,
            default_text: null,
            on_show_screen: {
                setup: function() {
                    d.log("livejournal: setup screen");
                    var f = d.shareThisV3Dialog.get_current_service();
                    c = d.one("#lj-setup-api-endpoint");
                    e = d.one("#lj-setup-username");
                    a = d.one("#lj-setup-password");
                    if (a) {
                        a.on("focus", f.misc.set_password_field)
                    }
                    if (e) {
                        e.on("keypress", f.misc.update_endpoint_url);
                        e.on("keyup", f.misc.update_endpoint_url)
                    }
                },
                setup_choose_a_blog: function() {
                    d.log("livejournal: setup_choose_a_blog()")
                },
                share: function() {
                    var g = d.shareThisV3Dialog.get_current_service(), h = d.shareThisV3Dialog.service_helpers, f = d.one("#livejournal-text");
                    g.default_text = f.getAttribute("data-default-text");
                    f.on("focus", h.maybeClearText);
                    f.on("blur", h.maybeRestoreText);
                    h.addDescriptionEvents(d.one("#share-this-v3-dialog .add-object-description"), f)
                },
                password: function() {
                    var g = d.shareThisV3Dialog.get_current_service(), f = d.one("#lj-setup-password");
                    if (f) {
                        f.focus()
                    }
                }
            },
            on_submit_form: {
                share: function() {
                    d.shareThisV3Dialog.service_helpers.maybeClearText({
                        target: d.one("#livejournal-text")
                    })
                }
            },
            on_auth_failed: function() {
                d.shareThisV3Dialog.set_dialog_state("state-setup state-setup-error");
                d.global_dialog.reposition()
            },
            misc: {
                update_endpoint_url: function(f) {
                    c.set("value", c.getAttribute("data-default-url").replace("%s", e.get("value")))
                },
                set_password_field: function(f) {
                    var g = f.target;
                    if (g) {
                        d.shareThisV3Dialog.service_helpers.setPasswordField(g)
                    }
                }
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-service-pinterest", function(b) {
    var a = b.shareThisV3Services;
    if (!a) {
        b.log("Y.shareThisV3Services null/undefined?")
    } else {
        a.add_service({
            name: "pinterest",
            id: 132,
            use_popup: true,
            signed_out_url: "/share.gne?action=signedOutShare&service_type_id=132",
            share_signed_out: function(g) {
                var d = b.shareThisV3Dialog.get_current_service(), c = "&" + b.shareThisV3Dialog.get_share_this_data(true).join("&"), f = d.signed_out_url + c;
                b.shareThisV3Dialog.webCirca1999(f, {
                    width: 685,
                    height: 520
                });
                b.fire("flickr-menus:hide");
                return false
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-service-tumblr", function(b) {
    var a = b.shareThisV3Services;
    if (!a) {
        b.log("Y.shareThisV3Services null/undefined?")
    } else {
        a.add_service({
            name: "tumblr",
            id: 12,
            setup_url: "/share.gne?action=setup&service_type_id=12",
            use_popup: true,
            popup_params: {
                width: 920,
                height: 600
            },
            signed_out_url: "/share.gne?action=signedOutShare&service_type_id=12",
            default_text: null,
            share_signed_out: function(g) {
                var d = b.shareThisV3Dialog.get_current_service(), c = "&" + b.shareThisV3Dialog.get_share_this_data(true).join("&"), f = d.signed_out_url + c;
                b.shareThisV3Dialog.webCirca1999(f, {
                    width: 788,
                    height: 375
                });
                return false
            },
            on_window_closed: function() {
                var c = b.shareThisV3Dialog.get_current_service();
                b.log("auth (maybe) complete for " + c.name);
                if (c.setup_only) {
                    b.shareThisV3Dialog.fetch_sharing_fragment()
                } else {
                    b.shareThisV3Dialog.fetch_sharing_fragment()
                }
            },
            on_auth_failed: function() {
                b.log("D'oh, auth did not work, or something. Retry?")
            },
            on_show_screen: {
                share: function() {
                    b.log("on_show_screen:share");
                    b.global_dialog.show();
                    var d = b.shareThisV3Dialog.get_current_service(), f = b.shareThisV3Dialog.service_helpers, c = b.one("#tumblr-text"), e = b.one("#tumblr-butt-post");
                    if (!c) {
                        b.log("Warn: could not find textarea");
                        return false
                    }
                    f.addDescriptionEvents(b.one("#share-this-v3-dialog .add-object-description"), c);
                    f.addMoreOptionsEvents();
                    d.default_text = c.getAttribute("data-default-text");
                    c.on("focus", f.maybeClearText);
                    c.on("blur", f.maybeRestoreText)
                }
            },
            on_submit_form: {
                share: function() {
                    b.shareThisV3Dialog.service_helpers.maybeClearText({
                        target: b.one("#tumblr-text")
                    })
                },
                setup_choose_a_blog: function() {
                    b.global_dialog.hide();
                    b.global_dialog.set_loading(true)
                }
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-service-twitter", function(b) {
    var a = b.shareThisV3Services;
    if (!a) {
        b.log("Y.shareThisV3Services null/undefined?")
    } else {
        a.add_service({
            name: "twitter",
            id: 9,
            setup_url: "/share.gne?action=setup&service_type_id=9",
            use_popup: true,
            popup_params: {
                width: 780,
                height: 500
            },
            signed_out_url: "/share.gne?action=signedOutShare&service_type_id=9",
            max_tweet_length: 140,
            default_text: null,
            flickr_shorturl_length: null,
            over_limit_classname: "over-limit",
            share_signed_out: function(g) {
                var d = b.shareThisV3Dialog.get_current_service(), c = "&" + b.shareThisV3Dialog.get_share_this_data(true).join("&"), f = d.signed_out_url + c;
                b.shareThisV3Dialog.webCirca1999(f, {
                    width: 780,
                    height: 425
                });
                return false
            },
            on_window_closed: function() {
                var c = b.shareThisV3Dialog.get_current_service();
                b.log("auth (maybe) complete for " + c.name);
                if (c.setup_only) {
                    b.shareThisV3Dialog.service_helpers.setupOnly()
                } else {
                    b.shareThisV3Dialog.fetch_sharing_fragment()
                }
            },
            on_auth_failed: function() {
                b.log("D'oh, auth did not work, or something. Retry?")
            },
            on_show_screen: {
                share: function() {
                    b.log("on_show_screen:share");
                    var k = b.shareThisV3Dialog.get_current_service();
                    var l = b.one("#tweet-text"), c = b.one("#tweet-characters"), m = null, g = false, h, n, i = {
                        "1": "over_limit_1",
                        "5": "over_limit_5",
                        "10": "over_limit_10",
                        "20": "over_limit_20",
                        "30": "over_limit_30"
                    }, j = b.one("#twitter-butt-post");
                    if (!l) {
                        b.log("Warn: could not find textarea");
                        return false
                    }
                    function o(r, s, p) {
                        if (r._node.setSelectionRange) {
                            r._node.focus();
                            r._node.setSelectionRange(s, p)
                        } else {
                            if (r._node.createTextRange) {
                                var q = r._node.createTextRange();
                                q.collapse(true);
                                q.moveStart("character", s);
                                q.moveEnd("character", p);
                                q.select()
                            }
                        }
                    }
                    var e = b.Lang.trim(b.one("#tweet-shorturl").get("title"));
                    var f = l.get("value");
                    l.set("text", f + " " + e);
                    o(l, 0, f.length);
                    k.max_tweet_length = parseInt(l.getAttribute("data-available-chars"), 10);
                    k.default_text = l.getAttribute("data-default-text");
                    function d() {
                        var q = parseInt(l.get("value").length, 10), p = (140 - parseInt(q, 10)), r;
                        if (!isNaN(p) && p !== m) {
                            m = p;
                            if (p < 0&&!g) {
                                g = true;
                                c.addClass(k.over_limit_classname);
                                j.replaceClass("Butt", "DisabledButt");
                                j.set("disabled", true)
                            } else {
                                if (p >= 0 && g) {
                                    g = false;
                                    c.removeClass(k.over_limit_classname);
                                    j.replaceClass("DisabledButt", "Butt");
                                    j.set("disabled", false);
                                    c.setStyle("fontSize", "11px")
                                }
                            }
                            if (p >= 0) {
                                c.set("innerHTML", p)
                            } else {
                                r = Math.abs(p);
                                if (r >= 1 && r < 5) {
                                    h = "1";
                                    n = 1
                                } else {
                                    if (r >= 5 && r < 10) {
                                        h = "5";
                                        n = 1.25
                                    } else {
                                        if (r >= 10 && r < 20) {
                                            h = "10";
                                            n = 1.5
                                        } else {
                                            if (r >= 20 && r < 30) {
                                                h = "20";
                                                n = 1.75
                                            } else {
                                                if (r >= 30) {
                                                    h = "30";
                                                    n = 2
                                                }
                                            }
                                        }
                                    }
                                }
                                over_limit_string = b.transjax.get("share-this-v3-service-twitter", i[h], p);
                                c.set("innerHTML", over_limit_string);
                                c.setStyle("fontSize", Math.min(32, parseInt(11 * n + (r > 30 ? (r - 30) / 10 : 0), 10)) + "px")
                            }
                        } else {
                            if (isNaN(p)) {
                                c.setStyle("display", "none")
                            }
                        }
                        b.one("#tweet-shorturl a").on("click", function(u) {
                            var t = /([^\s^]*)([http\:\/\/]+)?flic\.kr([\/a-zA-Z0-9]+)?([^\s$]*)/;
                            if (l.get("value").search(t) > 0) {
                                l.set("value", l.get("value").replace(t, e))
                            } else {
                                l.set("value", l.get("value") + " " + e)
                            }
                            o(l, l.get("value").length, l.get("value").length);
                            u.preventDefault()
                        });
                        var s = new RegExp("(^|\\s)" + e.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&") + "($|\\s)");
                        if (l.get("value").search(s) < 0) {
                            b.one("#tweet-shorturl span").show();
                            j.replaceClass("Butt", "DisabledButt");
                            j.set("disabled", true)
                        } else {
                            b.one("#tweet-shorturl span").hide();
                            if (!g) {
                                j.replaceClass("DisabledButt", "Butt");
                                j.set("disabled", false)
                            }
                        }
                    }
                    b.shareThisV3Dialog.service_helpers.addMoreOptionsEvents();
                    l.on("focus", b.shareThisV3Dialog.service_helpers.maybeClearText);
                    l.on("focus", d);
                    l.on("blur", b.shareThisV3Dialog.service_helpers.maybeRestoreText);
                    l.on("keyup", d);
                    l.on("change", d);
                    d()
                },
                success: function() {
                    var f = b.one(".flickr-follow-yes");
                    var d = b.one(".flickr-follow-no");
                    if (f && d) {
                        f.on("click", e);
                        d.on("click", e)
                    }
                    function e(k) {
                        var j = k.target.getAttribute("data-action");
                        var i = b.shareThisV3Dialog.get_current_service();
                        var g = [];
                        g.push("action=" + j);
                        g.push("service_type_id=9");
                        g.push("service_id=" + i.service_id);
                        g.push("magic_cookie=" + b.config.flickr.magic_cookie);
                        g = g.join("&");
                        var h = "/share.gne";
                        if (j == "followFlickr") {
                            b.one("#follow-flickr").set("innerHTML", '<p class="follow-working">' + b.transjax.get("share-this-v3-dialog", "following_flickr_waiting") + "</p>")
                        } else {
                            b.one("#follow-flickr").remove();
                            b.one(".follow-flickr-headline").remove()
                        }
                        b.io(h, {
                            method: "POST",
                            data: g,
                            on: {
                                complete: function(n, l, m) {
                                    if (c(l.responseText)) {
                                        b.one("#follow-flickr").set("innerHTML", '<p class="follow-error">' + b.transjax.get("share-this-v3-dialog", "try_again_later") + "</p>")
                                    } else {
                                        if (j == "followFlickr") {
                                            b.one("#follow-flickr").set("innerHTML", '<p class="follow-success">' + b.transjax.get("share-this-v3-dialog", "following_flickr_success") + "</p>")
                                        }
                                    }
                                }
                            }
                        });
                        k.preventDefault();
                        return 
                    }
                    function c(g) {
                        return (g && g.match(/state-error/i))
                    }
                }
            },
            on_submit_form: {
                share: function() {
                    b.shareThisV3Dialog.service_helpers.maybeClearText({
                        target: b.one("#tweet-text")
                    })
                }
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-service-wordpress", function(c) {
    var b = c.shareThisV3Services;
    function a() {
        var e = c.one("#share-form-wordpress"), g = c.one("#wp-setup-api-endpoint"), d = c.one("#wp-setup-username"), f = false;
        if (d.get("value") !== "" && g.get("value") !== "") {
            f = true
        }
        if (f) {
            c.shareThisV3Dialog.service_helpers.enableForm(e)
        } else {
            c.shareThisV3Dialog.service_helpers.disableForm(e)
        }
    }
    if (!b) {
        c.log("Y.shareThisV3Services null/undefined?")
    } else {
        b.add_service({
            name: "wordpress",
            id: 7,
            use_setup_dialog: true,
            default_text: null,
            on_show_screen: {
                setup: function() {
                    c.log("wordpress: setup screen");
                    var e = c.shareThisV3Dialog.get_current_service(), f = c.one("#wp-setup-api-endpoint"), d = c.one("#wp-setup-username");
                    if (f) {
                        f.on("focus", e.misc.maybe_highlight_endpoint);
                        f.focus();
                        f.on("keypress", a);
                        f.on("keyup", a)
                    }
                    if (d) {
                        d.on("keypress", a);
                        d.on("keyup", a)
                    }
                    a()
                },
                setup_choose_a_blog: function() {
                    c.log("wordpress: choose_a_blog()")
                },
                share: function() {
                    c.log("wordpress: share screen");
                    var g = c.shareThisV3Dialog.service_helpers, e = c.shareThisV3Dialog.get_current_service(), d = c.one("#wordpress-text"), f = c.one("#wordpress-subject");
                    c.global_dialog.set_loading(false);
                    c.global_dialog.show();
                    e.default_text = d.getAttribute("data-default-text");
                    g.addDescriptionEvents(c.one("#share-this-v3-dialog .add-object-description"), d);
                    d.on("focus", g.maybeClearText);
                    d.on("blur", g.maybeRestoreText);
                    if (f) {
                        f.focus()
                    }
                },
                password: function() {
                    c.log("wordpress: password screen");
                    var e = c.shareThisV3Dialog.get_current_service(), d = c.one("#wp-setup-password");
                    if (d) {
                        d.focus()
                    }
                }
            },
            on_submit_form: {
                share: function() {
                    c.shareThisV3Dialog.service_helpers.maybeClearText({
                        target: c.one("#wordpress-text")
                    })
                },
                setup_choose_a_blog: function() {
                    c.global_dialog.hide();
                    c.global_dialog.set_loading(true)
                }
            },
            on_auth_failed: function() {
                c.log("on_auth_failed");
                set_dialog_state("state-setup state-setup-error");
                c.global_dialog.reposition();
                c.log("setup screen shown - now what, complain?")
            },
            misc: {
                maybe_highlight_endpoint: function(g) {
                    var h = g.target, i = g.target.getAttribute("data-default-url"), d = g.target.get("value"), f = d.indexOf(i);
                    if (f!==-1) {
                        c.shareThisV3Dialog.service_helpers.setSelectionRange(c.Node.getDOMNode(h), f, f + i.length)
                    }
                }
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-service-yahoo-pulse", function(c) {
    var b = c.shareThisV3Services, d = "#pulse-text", a;
    if (!b) {
        c.log("Y.shareThisV3Services null/undefined?")
    } else {
        b.add_service({
            name: "yahoo",
            id: 129,
            default_text: null,
            flickr_shorturl_length: null,
            share_signed_out: function(f) {},
            on_show_screen: {
                share: function() {
                    c.log("on_show_screen:share");
                    var f = c.shareThisV3Dialog.get_current_service(), e = c.one(d);
                    f.default_text = e.getAttribute("data-default-text");
                    e.on("focus", c.shareThisV3Dialog.service_helpers.maybeClearText);
                    e.on("blur", c.shareThisV3Dialog.service_helpers.maybeRestoreText)
                }
            },
            on_submit_form: {
                share: function() {
                    c.shareThisV3Dialog.service_helpers.maybeClearText({
                        target: c.one(d)
                    });
                    c.fire("shareDialog:servicesAuthComplete")
                }
            }
        })
    }
}, "0.0.1", {
    requires: ["share-this-v3-services"]
});
YUI.add("share-this-v3-dialog", function(c) {
    var O = null;
    var L = null;
    var S = null;
    var b = null;
    var z = null;
    var B = null;
    var C = [];
    var n = "state-";
    var Q = "share-this-v3-dialog";
    var d = (function() {
        var ac = "#guestpass-box", ag = "data-required", ad = "data-auto-guestpass";
        function ai() {
            var an = c.one(ac);
            if (!an) {
                return false
            }
            return (an.getAttribute(ag) === "true")
        }
        function ak() {
            var ap = c.one(ac), an = (ap ? ap.all('input[type="checkbox"]') : null), ao = false;
            if (an) {
                an.each(function() {
                    if (this.get("checked")) {
                        ao = true
                    }
                })
            }
            return ao
        }
        function U() {
            var an = c.one(ac);
            if (!an) {
                return false
            }
            return (an.getAttribute(ad) === "true")
        }
        var aa = {
            setup: function() {},
            share: function() {
                if (c.one(ac)) {
                    r()
                }
            },
            success: function() {
                c.use("flanal", function(an) {
                    an.flanal("flickr.sharing." + B.name + "_share_complete", B.name + "_share_complete")
                })
            }
        };
        function aj() {
            v();
            M()
        }
        function V() {
            if (typeof B.setup_only !== "undefined" && B.setup_only) {
                c.global_dialog.hide();
                c.global_dialog.set_loading(true);
                setTimeout(function() {
                    window.location.href = "/account/?tab=extend"
                }, 20);
                return true
            } else {
                return false
            }
        }
        function Z(ao) {
            var an = ao.target, ap = an.get("value");
            if (ap === B.default_text) {
                an.set("value", "");
                an.removeClass("default-text");
                if (typeof updateWhileTyping !== "undefined") {
                    updateWhileTyping()
                }
            }
        }
        function al(an) {}
        function af(at) {
            var ar = at, ao = ar.get("type");
            if (ao !== "password") {
                if (c.UA.ie) {
                    var aq = c.Node.getDOMNode(ar), ap = aq.cloneNode(true), an;
                    ap.type = "password";
                    ap.value = "";
                    an = aq.parentNode.insertBefore(ap, aq);
                    aq.parentNode.removeChild(aq);
                    setTimeout(function() {
                        try {
                            an.focus()
                        } catch (au) {}
                    }, 10)
                } else {
                    ar.set("type", "password");
                    ar.set("value", "")
                }
            }
        }
        function ae(ao, ap, aq) {
            if (ao.setSelectionRange) {
                ao.focus();
                ao.setSelectionRange(ap, aq)
            } else {
                if (ao.createTextRange) {
                    var an = ao.createTextRange();
                    an.collapse(true);
                    an.moveEnd("character", aq);
                    an.moveStart("character", ap);
                    an.select()
                }
            }
        }
        function Y(an) {
            var ao = (an ? an : c.one("#share-this-v3-dialog textarea.share-object-description"));
            if (ao) {
                return (ao.get("value"))
            }
        }
        function am(an, ap) {
            Z({
                target: an
            });
            var aq = Y(), ao = an.get("value");
            if (aq) {
                an.set("value", (ao ? ao + "\n\n" : "") + aq)
            }
            ap.preventDefault();
            return false
        }
        function ah(ao, an) {
            if (ao) {
                ao.on("click", function(ap) {
                    c.shareThisV3Dialog.service_helpers.addObjectDescription(an, ap);
                    ap.target.ancestor(".add-description").addClass("description-added")
                })
            }
        }
        function W() {
            var ao = c.one("#share-this-v3-dialog"), ap = ao.one(".more-options"), an = ao.one(".more-options-toggler");
            if (ap && an && an.one("a")) {
                an.one("a").on("click", function(aq) {
                    aq.preventDefault();
                    ap.toggleClass("more-options-open");
                    an.toggleClass("more-options-toggler-open")
                })
            }
        }
        function T(ao) {
            var an = ao.all('input[type="submit"]');
            an.each(function() {
                this.replaceClass("Butt", "DisabledButt");
                this.setAttribute("disabled", "disabled")
            })
        }
        function X(ao) {
            var an = ao.all('input[type="submit"]');
            an.each(function() {
                this.replaceClass("DisabledButt", "Butt");
                this.removeAttribute("disabled")
            })
        }
        function ab() {
            c.global_dialog.remove_existing_dialog();
            c.global_dialog.set_loading(false);
            c.global_dialog.create_alert_dialog("<b>" + c.transjax.get("share-this-v3-dialog", "try_again_later") + "</b>", function() {
                c.global_dialog.hide()
            })
        }
        return {
            assign_events: aj,
            addDescriptionEvents: ah,
            addObjectDescription: am,
            addMoreOptionsEvents: W,
            disableForm: T,
            enableForm: X,
            genericError: ab,
            is_guestpass_automatic: U,
            is_guestpass_chosen: ak,
            is_guestpass_required: ai,
            on_show_screen: aa,
            maybeClearText: Z,
            maybeRestoreText: al,
            setupOnly: V,
            setPasswordField: af,
            setSelectionRange: ae
        }
    }());
    var k = {};
    function q(U) {
        var V, T = null;
        U = parseInt(U, 10);
        for (V in k) {
            if (k.hasOwnProperty(V) && k[V].id === U) {
                T = k[V].name;
                break
            }
        }
        return T
    }
    function P(T) {
        var U, W = [], V = {
            object_id: y(),
            owner_nsid: a(),
            object_type: w()
        };
        if (!T) {
            return V
        }
        for (U in V) {
            if (V.hasOwnProperty(U)) {
                W.push(U + "=" + V[U])
            }
        }
        return W
    }
    function y() {
        if (c.config && c.config.flickr && c.config.flickr.sharing && c.config.flickr.sharing.object) {
            return c.config.flickr.sharing.object.id
        } else {
            return null
        }
    }
    function a() {
        if (c.config && c.config.flickr && c.config.flickr.sharing && c.config.flickr.sharing.object) {
            return c.config.flickr.sharing.object.owner_nsid
        } else {
            return null
        }
    }
    function w() {
        if (c.config && c.config.flickr && c.config.flickr.sharing && c.config.flickr.sharing.object) {
            return c.config.flickr.sharing.object.type
        } else {
            return null
        }
    }
    function D() {
        var U = c.one("#share-this-v3-dialog"), T = null;
        if (U) {
            T = U.getAttribute("data-state")
        }
        return T
    }
    function l(V) {
        var U = "#share-this-dialog-state", T;
        if (c.one(U)) {
            c.one(U).setAttribute("class", V)
        }
        if (V && V.length > n.length) {
            V = V.substr(n.length);
            if (V !== "working") {
                m()
            }
            T = c.one("#share-this-v3-dialog .auto-focus");
            if (T) {
                T.focus()
            }
            if (d.on_show_screen && d.on_show_screen[V]) {
                d.on_show_screen[V]()
            }
            if (B && B.on_show_screen && B.on_show_screen[V]) {
                B.on_show_screen[V]()
            }
        }
    }
    function A() {
        l("state-setup")
    }
    function H() {
        function U(W) {
            return (parseInt(W.get("offsetHeight"), 10) - 20)
        }
        var T = {
            left: c.one("#global-dialog-container .left-item"),
            right: c.one("#global-dialog-container .right-item")
        }, V = {};
        if (T.left && T.right) {
            V = {
                left: U(T.left),
                right: U(T.right)
            };
            if (V.left > V.right) {
                T.right.setStyle("minHeight", V.left + "px")
            } else {
                T.left.setStyle("minHeight", V.right + "px")
            }
        }
        c.global_dialog.reposition()
    }
    function i(T) {
        j()
    }
    function p(T) {
        C.push(T)
    }
    function m() {
        for (var T = C.length; T--;) {
            C[T].detach()
        }
        C = []
    }
    function r() {
        p(c.all('#guestpass-box input[type="checkbox"]').on("click", i))
    }
    function M() {
        p(c.all("#global-dialog-container form.share-form").on("submit", o))
    }
    function v() {
        var T = "#global-dialog-container a, #global-dialog-container input", V = c.one(".fb-needs-reconnect");
        function U(X) {
            var W = X.target;
            if (W && (W.hasClass("close-x") || W.hasClass("CancelButt") || W.hasClass("dismissDialog"))) {
                c.global_dialog.hide();
                X.preventDefault();
                if (c.PageContext) {
                    c.PageContext.unset(Q)
                }
                return false
            }
        }
        c.all(T).on("click", U);
        if (V) {
            V.on("click", function(W) {
                W.preventDefault();
                c.global_dialog.hide();
                if (c.PageContext) {
                    c.PageContext.unset(Q)
                }
                c.shareThisV3Dialog.share_by_service(B.id, B.service_type_id, true, false, null, W)
            }, this)
        }
    }
    function I() {
        B.needs_setup = true;
        if (B.on_auth_failed) {
            B.on_auth_failed()
        }
    }
    function K(W, X, T) {
        var V;
        if (B && B.needs_setup) {
            if (X.responseText) {
                c.one("#share-this-v3-dialog").get("parentNode").set("innerHTML", X.responseText);
                V = D();
                if (V && V.match(/share/i)) {
                    c.fire("shareDialog:servicesAuthComplete")
                }
                l(V)
            } else {
                var U = d.setupOnly();
                if (!U) {
                    I()
                }
            }
        } else {
            if (G(X.responseText)) {
                if (c.one("#share-this-v3-dialog")) {
                    c.one("#share-this-v3-dialog").get("parentNode").set("innerHTML", X.responseText);
                    V = D();
                    l(V)
                } else {
                    d.genericError()
                }
            } else {
                d.genericError()
            }
        }
        d.assign_events()
    }
    function E() {
        var T = D();
        if (T) {
            l(T)
        }
    }
    function o(V) {
        var W = V.target, U, T;
        if (W.hasClass("scrumjax-exclude")) {
            c.global_dialog.hide();
            c.global_dialog.set_loading(true);
            return true
        }
        if (B) {
            T = D();
            if (T) {
                U = T.substr(n.length);
                if (B.on_submit_form && B.on_submit_form[U]) {
                    B.on_submit_form[U](V)
                }
            }
        }
        c.use("io-form", function() {
            var X = {
                method: "POST",
                form: {
                    id: W.getAttribute("id"),
                    useDisabled: false
                },
                on: {
                    success: K,
                    failure: K,
                    timeout: K
                }
            };
            if (B && B.setup_only) {
                l("state-working setup-only")
            } else {
                l("state-working")
            }
            var Y = W.getAttribute("action");
            if (!Y) {
                Y = W.getAttribute("data-ie-action")
            }
            if (c.one("#" + W.getAttribute("id"))) {
                c.io(Y, X)
            }
        });
        V.preventDefault();
        return false
    }
    function G(T) {
        return (T && T.match(/share\-this\-v3/i))
    }
    function e(W, U, V, T) {
        var Y = {
            success: function(ad, ab, ac) {
                var Z = ab.responseText, aa;
                if (G(Z)) {
                    c.global_dialog.remove_existing_dialog();
                    c.global_dialog.create_dialog(Z);
                    aa = D();
                    c.global_dialog.show({
                        modal: true,
                        width: (!c.UA.ie || c.config.is_zeus ? 422 : 440)
                    });
                    if (V && V.on_complete) {
                        V.on_complete()
                    } else {
                        l(aa)
                    }
                    c.global_dialog.reposition();
                    c.global_dialog.set_loading(false);
                    c.global_dialog.set_modal(true);
                    d.assign_events()
                } else {
                    if (B && B.setup_only) {
                        d.setupOnly()
                    } else {
                        d.genericError()
                    }
                }
            },
            failure: function(ab, Z, aa) {
                if (Z.statusText === "timeout") {
                    if (B && B.needs_auth && B.on_auth_failed) {
                        B.on_auth_failed()
                    }
                } else {
                    if (B && B.needs_auth && B.on_auth_failed) {
                        B.on_auth_failed()
                    } else {
                        c.global_dialog.hide();
                        d.genericError()
                    }
                }
            }
        };
        var X = (typeof T !== "undefined" ? T : Y);
        c.global_dialog.set_loading(true);
        c.io(W, {
            method: "POST",
            data: U,
            on: X,
            "arguments": U
        })
    }
    function x(U) {
        var W = (U || null), V = "/share.gne", T = [];
        T.push("action=delete");
        if (U.service_id) {
            T.push("service_id=" + U.service_id)
        }
        if (U.service_type_id) {
            T.push("service_type_id=" + U.service_type_id)
        }
        T.push("cachebust=" + (new Date()).getTime());
        T = T.join("&");
        W = {
            on_complete: function() {
                var Y = c.one("#share-confirm-delete");
                if (Y) {
                    try {
                        Y.focus()
                    } catch (X) {}
                }
                p(c.one("#global-dialog-container form.scrumjax-exclude").on("submit", o))
            }
        };
        e(V, T, W)
    }
    function h(U) {
        var W = (U || null), V = "/share.gne", T = [];
        T.push("action=retrieve_fragments");
        if (B.needs_setup) {
            T.push("fragments[]=dialog-setup")
        } else {
            T.push("fragments[]=dialog-share")
        }
        if (L) {
            T.push("service_name=" + L)
        }
        if (S) {
            T.push("object_id=" + S)
        }
        if (b) {
            T.push("object_type=" + b)
        }
        if (z) {
            T.push("owner_nsid=" + z)
        }
        T.push("cachebust=" + (new Date()).getTime());
        if (B) {
            if (B.setup_only) {
                T.push("setup_only=1")
            }
            if (B.service_id) {
                T.push("service_id=" + B.service_id)
            }
            if (B.service_type_id) {
                T.push("service_type_id=" + B.service_type_id)
            }
        }
        T = T.join("&");
        e(V, T, W)
    }
    function j(U) {
        var W = (U || null), V = "/share.gne", T = [];
        T.push("action=retrieve_fragments");
        T.push("fragments[]=dialog-share-thumbnails");
        if (L) {
            T.push("service_name=" + L)
        }
        if (S) {
            T.push("object_id=" + S)
        }
        if (b) {
            T.push("object_type=" + b)
        }
        if (z) {
            T.push("owner_nsid=" + z)
        }
        if (B) {
            if (B.service_id) {
                T.push("service_id=" + B.service_id)
            }
            if (B.service_type_id) {
                T.push("service_type_id=" + B.service_type_id)
            }
        }
        c.one("#global_dialog").all("input.gp-type:checked").each(function(Y) {
            T.push(Y.getAttribute("name") + "=1")
        });
        T = T.join("&");
        W = {
            on_complete: function() {}
        };
        var X = {
            success: function(ab, Z, aa) {
                var Y = Z.responseText;
                c.one("#share-this-thumbnail").get("parentNode").set("innerHTML", Y)
            }
        };
        c.one("#share-this-thumbnail-loading").setStyle("display", "block");
        e(V, T, W, X)
    }
    function g(V, Z, U, X, W, T) {
        if (c.PageContext) {
            c.PageContext.set(Q)
        }
        S = y();
        b = w();
        z = a();
        L = q(Z);
        if (!k[L]) {
            return false
        } else {
            B = k[L]
        }
        var Y = T && typeof c.config.flickr.user.admin_user !== "undefined" ? T.altKey: false;
        if (!W&&!Y && c.config && c.config.flickr && c.config.flickr.user && c.config.flickr.user.nsid) {} else {
            if (!W&&!Y && typeof global_nsid !== "undefined" && global_nsid) {} else {
                if (B.share_signed_out) {
                    B.share_signed_out(T);
                    T.preventDefault();
                    return false
                }
            }
        }
        c.fire("flickr-menus:hide");
        B.needs_setup = U;
        B.service_id = V;
        B.service_type_id = Z;
        B.setup_only = X;
        if (U) {
            if (B.use_popup && B.setup_url) {
                s(B.setup_url, B.popup_params, function() {
                    if (B.on_window_closed) {
                        B.on_window_closed();
                        c.fire("shareDialog:servicesAuthComplete")
                    }
                })
            } else {
                if (B.use_setup_dialog) {
                    h({
                        on_complete: function() {
                            if (B.needs_setup) {
                                A()
                            }
                        }
                    })
                } else {
                    h({
                        on_complete: function() {
                            if (B.needs_setup) {
                                A()
                            }
                        }
                    })
                }
            }
            return false
        } else {
            h()
        }
        return false
    }
    function u() {
        return B
    }
    function N() {
        c.global_dialog.set_loading(true)
    }
    function t(T) {
        if (O && O.closed) {
            T()
        } else {
            setTimeout(function() {
                t(T)
            }, 1000)
        }
    }
    function s(V, T, W) {
        var X, U = [];
        for (X in T) {
            if (T.hasOwnProperty(X)) {
                U.push(X + "=" + T[X])
            }
        }
        U.push("resizable=1");
        U.push("scrollbars=1");
        try {
            O = window.open(V, "newWindow", U.join(","));
            if (O.focus) {
                O.focus()
            }
        } catch (Y) {
            return true
        }
        if (W) {
            N();
            t(W)
        }
        return false
    }
    function f() {
        var U = {}, T;
        U.service_type_id = c.one("#form_service_type_id").get("value");
        U.object_type = c.one("#form_object_type").get("value");
        U.object_id = c.one("#form_object_id").get("value");
        T = c.one("#form_owner_nsid");
        if (T) {
            U.owner_nsid = T.get("value")
        }
        return U
    }
    function R(U) {
        var W = [];
        for (var V in U) {
            if (U.hasOwnProperty(V)) {
                W.push(V + "=" + encodeURIComponent(U[V]))
            }
        }
        var T = {
            method: "POST",
            data: W.join("&"),
            on: {
                complete: function(Z, X) {
                    var Y = X.responseText;
                    if (G(Y)) {
                        c.global_dialog.create_dialog(Y).show({
                            modal: true,
                            width: (!c.UA.ie || c.config.is_zeus ? 422 : 440)
                        });
                        v()
                    } else {
                        c.shareThisV3Dialog.service_helpers.genericError()
                    }
                }
            }
        };
        c.io("/share.gne", T)
    }
    function J() {
        if (c.shareThisV3Services) {
            k = c.shareThisV3Services.get_services()
        }
    }
    J();
    c.shareThisV3Dialog = {
        assign_dialog_dismiss_events: v,
        confirm_delete: x,
        do_share_action: R,
        fetch_sharing_fragment: h,
        get_current_service: u,
        get_form_params: f,
        init: J,
        refresh_dialog_state: E,
        services: k,
        service_helpers: d,
        get_share_this_data: P,
        share_by_service: g,
        set_dialog_state: l,
        show_setup_state: A,
        webCirca1999: s
    }
}, "0.0.1", {
    requires: F.config.modules["share-this-v3-dialog"].requires || [],
    optional: F.config.modules["share-this-v3-dialog"].optional || []
});
YUI.add("share-this-v3-triggers", function(c) {
    function b(d) {
        d.on("click", a)
    }
    function a(j) {
        var f, m, k, i, o, l, d, n, g, h;
        f = j.target.ancestor("a.share-dialog-action", true);
        if (!f && j.target.hasClass("Butt")) {
            f = j.target.one("a.share-dialog-action")
        }
        if (!f) {
            return 
        }
        if (f.hasClass("share-disabled")) {
            j.preventDefault();
            return 
        }
        k = f.getAttribute("data-service-id");
        i = f.getAttribute("data-service-type-id");
        if (!i) {
            j.preventDefault();
            return 
        }
        l = f.getAttribute("data-setup-only");
        d = (f.getAttribute("data-service-needs-setup") === "true");
        n = (f.getAttribute("data-service-always-use-popup") === "true");
        if (!d && j.shiftKey) {
            c.log("Allowing set-up override");
            d = true
        }
        if (!l) {
            m = c.shareThisV3Services.get_service(parseInt(i, 10));
            o = m && m.name && m.name.replace(/[^a-zA-Z0-9_\-]/g, "_");
            if (o) {
                if (c.config.flickr.user.nsid) {
                    if (d) {
                        h = o + "_button_needs_setup"
                    } else {
                        h = o + "_button_already_setup"
                    }
                } else {
                    h = o + "_button_signed_out"
                }
                g = "flickr.sharing_triggers." + h;
                c.flanal(g, h)
            }
        }
        c.shareThisV3Dialog.share_by_service(k, i, d, l, n, j);
        j.preventDefault();
        return false
    }
    c.shareThisV3Triggers = {
        listen: b,
        handleClick: a
    }
}, "0.0.1", {
    requires: F.config.modules["share-this-v3-triggers"].requires || [],
    optional: F.config.modules["share-this-v3-triggers"].optional || []
});
YUI.add("share-this-v3-menu", function(c) {
    var T, P, I, M, e, w = "share-options-open", b = "share-options-deferred", t = "share-options-placeholder", B = "share-menu", K = {}, m = false, E = c.config.flickr.page_type === "photo", J = c.config.flickr.page_type === "faves", p = c.config.flickr.flags.enable_2013_photo_page_extras, x = c.config.flickr.page_type === "sihp", g = c.config.flickr.page_type === "photostream", G = c.config.flickr.page_type === "set", s = c.config.flickr.page_type === "faves", k = c.config.flickr.page_type === "group_pool", D = p && (E || J), C = D || x || (p && (g || G || s)), v;
    function N(X) {
        var Y, V, Z, W = X || {};
        V = W.localPlacement || "#photo-actions #stats_ul";
        function U() {
            P = c.menus.add({
                id: "share-menu-v3",
                popover: Z,
                context: B,
                button_node: W.button_node || q(),
                constrain_node: D ? c.one(document.body): null,
                scroll: p && (!(g || G || s || k) || c.config.flickr.flags.enable_justified_view_inline_sharing) ? W.scroll: null,
                onHide: function() {
                    if (W && W.onHide) {
                        W.onHide()
                    }
                }
            })
        }
        T = c.one("#share-menu-v3");
        if (!T) {
            c.log("share menu v3:: WARN: Could not find menuNode", "error", "share-this-v3-menu")
        }
        e = c.one("#post-upload-actions");
        if (e) {
            c.shareThisV3Triggers.listen(e)
        }
        M = c.one("#share-buttons, " + (D && E ? "#stats_ul" : "#button-bar"));
        if (M) {
            c.shareThisV3Triggers.listen(M);
            if (!c.config.flickr.is_touch_device) {
                Y = M.on("mouseover", function(aa) {
                    if (q().contains(aa.target)) {
                        Y.detach();
                        u()
                    }
                })
            }
        }
        if (T) {
            T.on("click", n);
            T.on("mousedown", o);
            T.on("mouseup", l);
            if (!c.config.flickr.is_touch_device) {
                T.on("mouseover", c.betterThrottle(function(ab) {
                    var aa = ab.target.ancestor("li." + b, true);
                    if (aa) {
                        u(aa)
                    }
                }, 100))
            }
            i();
            Z = new c.Popover({
                srcNode: "#share-menu-v3",
                visible: false,
                width: "400px",
                accordian: true,
                arrowPosition: W.arrowPosition || "tr",
                isRefresh: W.overrideRefresh || D,
                refreshBottom: W.bottom || "37px",
                refreshRight: W.right || "66px",
                showSquare: C,
                extraCSS: C ? "square-menu-css": ""
            });
            Z.on("visibleChange", function(aa) {
                if (aa.newVal) {
                    a();
                    u()
                }
            });
            Z.render((D&&!W.overrideRefresh) ? c.one(V) : undefined);
            if (c.one("#post-upload-share-more-button")) {
                I = c.menus.add({
                    id: "share-menu-v3",
                    popover: Z,
                    context: B,
                    button_node: c.one("#post-upload-share-more-button")
                });
                c.on("postUploadSharing:close", U)
            } else {
                U()
            }
        }
        c.on("shareDialog:servicesAuthComplete", function() {
            if (P) {
                P.hide()
            }
            y();
            window.setTimeout(function() {
                A((p ? ["menu-services"] : ["buttons", "menu-services"]), {
                    reload: true
                })
            }, 1000)
        });
        if (c.config.flickr.sharing.shareability_is_unknown) {
            A((p ? ["menu-full"] : ["buttons", "menu-full"]), {
                menuCallback: W.afterInit
            })
        }
        if (F.actionQueue && c.Lang.isObject(F.actionQueue)) {
            F.actionQueue["share-this-v3-more-button"] = {
                display_share: function() {
                    if (P) {
                        P.show()
                    }
                }
            };
            F.actionQueue.module_loaded("share-this-v3-more-button")
        }
        c.fire("shareMenu:initialized");
        m = true
    }
    function o(U) {
        if (U.target.test("#share-options-grablink-input") || U.target.test(".embed-markup")) {
            U.target.select()
        }
    }
    function l(U) {
        if (U.target.test("#share-options-grablink-input") || U.target.test(".embed-markup")) {
            U.preventDefault()
        }
    }
    function n(U) {
        if (U.target.test("#share-options-grablink-input") || U.target.test(".embed-markup")) {
            U.target.select();
            U.preventDefault()
        } else {
            if (U.target.ancestor("li.share-options div.share-options-header", true)) {
                U.preventDefault();
                j(U.target.ancestor("li"))
            } else {
                if (U.target.ancestor(".share-more-ways", true)) {
                    U.preventDefault();
                    T.one(".share-services-small").toggleClass("share-hidden")
                } else {
                    if (U.target.ancestor("ul.share-options-list li, ul.share-services li, ul.share-services-small li", true)) {
                        c.shareThisV3Triggers.handleClick(U)
                    } else {
                        if (U.target.ancestor("form#grab-link input", true)) {
                            L(U)
                        } else {
                            if (U.target.ancestor("form#sharing-get-html-form input", true)) {
                                z(U)
                            }
                        }
                    }
                }
            }
        }
    }
    function a(U) {
        U = U || T.one("." + w);
        if (U && U.getAttribute("data-panel-name") === "services") {
            d(U)
        }
    }
    function u(U) {
        U = U || T.one("." + w);
        if (U && U.hasClass(b)) {
            if (U.get("id") === "share-options-grablink") {
                A(["menu-grab-the-link"])
            } else {
                if (U.get("id") === "share-options-services") {
                    A(["menu-services"])
                }
            }
            U.removeClass(b)
        }
    }
    function j(V) {
        var Z, U, ae, ab, ag, W, af, ad, Y, ac, X, ah, aa;
        if (!V || V.hasClass(w)) {
            return false
        }
        ae = T.one("." + w);
        a(ae);
        u(V);
        U = V.one(".share-options-header");
        Z = V.one(".share-options-inner");
        Z.setStyle("display", "block");
        W = Z.get("offsetHeight") + parseInt(Z.getComputedStyle("marginTop"), 10) + parseInt(Z.getComputedStyle("marginBottom"), 10) + U.get("offsetHeight") + parseInt(U.getComputedStyle("marginTop"), 10) + parseInt(U.getComputedStyle("marginBottom"), 10);
        ad = new c.Anim({
            node: V,
            to: {
                height: W
            },
            duration: 0.42,
            easing: c.Easing.easeBoth
        });
        ad.on("end", function() {
            V.setStyle("height", "");
            V.addClass(w)
        });
        if (!c.UA.ie || c.UA.ie >= 8) {
            Y = new c.Anim({
                node: U,
                from: {
                    backgroundColor: "#0063DC",
                    color: "#ffffff"
                },
                to: {
                    backgroundColor: "#ffffff",
                    color: "#000000"
                },
                duration: 0.42,
                easing: c.Easing.easeBoth
            });
            Y.on("end", function() {
                U.setStyle("backgroundColor", "");
                U.setStyle("color", "")
            })
        }
        if (ae) {
            ag = ae.one(".share-options-header");
            ab = ae.one(".share-options-inner");
            af = ag.get("offsetHeight") + parseInt(ag.getComputedStyle("marginTop"), 10) + parseInt(ag.getComputedStyle("marginBottom"), 10);
            ac = new c.Anim({
                node: ae,
                to: {
                    height: af
                },
                duration: 0.25,
                easing: c.Easing.easeBoth
            });
            ac.on("end", function() {
                ae.setStyle("height", "");
                ae.removeClass(w);
                ab.setStyle("display", "none")
            });
            if (!c.UA.ie || c.UA.ie >= 8) {
                X = new c.Anim({
                    node: ag,
                    from: {
                        backgroundColor: "#ffffff",
                        color: "#000000"
                    },
                    to: {
                        backgroundColor: "#F6F6F6",
                        color: "#0063dc"
                    },
                    duration: 0.25,
                    easing: c.Easing.easeBoth
                });
                X.on("end", function() {
                    ag.setStyle("backgroundColor", "");
                    ag.setStyle("color", "")
                })
            }
            ac.run();
            if (X) {
                X.run()
            }
        }
        ah = V.getAttribute("data-panel-name");
        ad.run();
        if (Y) {
            Y.run()
        }
        a(V);
        aa = {
            services: 1,
            link: 2,
            embed: 3,
            party: 0
        };
        if (ah && aa[ah]) {
            if (j.pending) {
                window.clearTimeout(j.pending);
                j.pending = undefined
            }
            j.pending = window.setTimeout(function() {
                c.flickrAPI.callMethod("flickr.prefs.setPrefs", {
                    share_this_panel: aa[ah]
                })
            }, 2000)
        }
    }
    function q() {
        return M && M.one(".share-dropdown-arrow")
    }
    function Q() {
        var U, W, V;
        U = T.one("#share-options-grablink-input");
        W = T.one("#short-url input");
        if (W) {
            V = W.get("checked") ? W.getAttribute("data-url-short") : W.getAttribute("data-url-long")
        }
        if (V) {
            U.set("value", V);
            U.select()
        }
    }
    function L(U) {
        if (U.target.get("id") === "gp-short-url") {
            Q()
        } else {
            if (U.target.hasClass("gp-type")) {
                if (T.one("form#grab-link input.gp-type:checked")) {
                    T.one("#gp-add-button").replaceClass("DisabledButt", "Butt")
                } else {
                    T.one("#gp-add-button").replaceClass("Butt", "DisabledButt")
                }
            } else {
                if (U.target.get("id") === "gp-add-button") {
                    U.preventDefault();
                    if (!U.target.hasClass("DisabledButt")) {
                        window.setTimeout(function() {
                            A(["menu-grab-the-link"], {
                                reload: true,
                                guestpass: true
                            })
                        }, 10)
                    }
                }
            }
        }
    }
    function z(V) {
        var U = V.target.get("id");
        if (U === "code-type-bbcode" || U === "code-type-html" || U === "code-type-playr") {
            h()
        }
    }
    function h() {
        var V, Y, X, U, W, Z;
        V = {
            playr: {
                Thumbnail: "share-options-embed-textarea-t-playr",
                Square: "share-options-embed-textarea-sq-playr",
                "Large Square": "share-options-embed-textarea-q-playr",
                Small: "share-options-embed-textarea-s-playr",
                "Small 320": "share-options-embed-textarea-n-playr",
                Medium: "share-options-embed-textarea-m-playr",
                "Medium 640": "share-options-embed-textarea-z-playr",
                "Medium 800": "share-options-embed-textarea-c-playr",
                Large: "share-options-embed-textarea-l-playr",
                "Large 1600": "share-options-embed-textarea-h-playr",
                "Large 2048": "share-options-embed-textarea-k-playr",
                Original: "share-options-embed-textarea-o-playr"
            },
            html: {
                Thumbnail: "share-options-embed-textarea-t",
                Square: "share-options-embed-textarea-sq",
                "Large Square": "share-options-embed-textarea-q",
                Small: "share-options-embed-textarea-s",
                "Small 320": "share-options-embed-textarea-n",
                Medium: "share-options-embed-textarea-m",
                "Medium 640": "share-options-embed-textarea-z",
                "Medium 800": "share-options-embed-textarea-c",
                Large: "share-options-embed-textarea-l",
                "Large 1600": "share-options-embed-textarea-h",
                "Large 2048": "share-options-embed-textarea-k",
                Original: "share-options-embed-textarea-o"
            },
            bbcode: {
                Thumbnail: "share-options-embed-textarea-t-bbcode",
                Square: "share-options-embed-textarea-sq-bbcode",
                "Large Square": "share-options-embed-textarea-q-bbcode",
                Small: "share-options-embed-textarea-s-bbcode",
                "Small 320": "share-options-embed-textarea-n-bbcode",
                Medium: "share-options-embed-textarea-m-bbcode",
                "Medium 640": "share-options-embed-textarea-z-bbcode",
                "Medium 800": "share-options-embed-textarea-c-bbcode",
                Large: "share-options-embed-textarea-l-bbcode",
                "Large 1600": "share-options-embed-textarea-h-bbcode",
                "Large 2048": "share-options-embed-textarea-k-bbcode",
                Original: "share-options-embed-textarea-o-bbcode"
            }
        };
        Y = {
            html: 0,
            bbcode: 1
        };
        if (c.config.flickr.flags.enable_playr) {
            Y = {
                html: 2,
                bbcode: 1,
                playr: 0
            }
        }
        X = {
            Square: 1,
            Thumbnail: 2,
            Small: 3,
            Medium: 4,
            "Medium 640": 5,
            Large: 6,
            Original: 7,
            "Large Square": 8,
            "Small 320": 9,
            "Medium 800": 10,
            "Large 1600": 11,
            "Large 2048": 12
        };
        U = c.one("#sharing_size").get("value");
        if (c.one("#code-type-bbcode").get("checked")) {
            W = "bbcode"
        } else {
            if (c.one("#code-type-playr") && c.one("#code-type-playr").get("checked")) {
                W = "playr"
            } else {
                W = "html"
            }
        }
        Z = V[W][U];
        c.all(".embed-markup").each(function(aa) {
            aa.setStyle("display", "none")
        });
        c.one("#" + Z).setStyle("display", "block");
        if (h.pending) {
            window.clearTimeout(h.pending);
            h.pending = undefined
        }
        h.pending = window.setTimeout(function() {
            c.flickrAPI.callMethod("flickr.prefs.setPrefs", {
                photo_embed_type: Y[W],
                photo_embed_size_new: X[U]
            })
        }, 2000)
    }
    function i() {
        if (!K.embed_size_change_handle && c.one("#sharing_size")) {
            K.embed_size_change_handle = c.one("#sharing_size").on("change", function(U) {
                h()
            })
        }
    }
    function f() {
        c.Object.each(K, function(V, U) {
            if (V && V.detach) {
                V.detach();
                K[U] = undefined
            }
        })
    }
    function y() {
        var U;
        if (M) {
            U = M.all(".share-this-v3");
            if (U && U.size()) {
                U.each(function(X) {
                    var V = X.one(".Butt, .DisabledButt"), W;
                    if (V) {
                        V.removeClass("Butt").addClass("DisabledButt share-disabled");
                        W = V.one("a");
                        if (W) {
                            W.removeAttribute("data-service-type-id");
                            W.removeAttribute("data-service-id");
                            W.removeAttribute("data-service-needs-setup")
                        }
                    }
                })
            }
        }
    }
    function O(V, X, ab) {
        var ac = "...", Y = ab.get("innerHTML"), Z = Y.match(/\.\.\./), U = Y.length - (Z ? 3 : 0), aa = 8, W = V.get("offsetWidth");
        if (W > X && U > aa) {
            ab.set("innerHTML", Y.substr(0, Y.length - (Z ? 4 : 1)) + ac);
            O.apply(this, arguments)
        }
        V.setAttribute("data-truncated", 1)
    }
    function d(U) {
        if (U && U.one("li")) {
            var V = U.one("li").get("offsetWidth"), W = U.all(".service-username");
            if (W) {
                W.each(function() {
                    if (!this.getAttribute("data-truncated")) {
                        O(this, V, this.one(".service-truncated"))
                    }
                })
            }
        }
    }
    function S() {
        return c.JSON.stringify(c.config.flickr.sharing.object)
    }
    function R(W, V) {
        var U = V ? ["menu-full"]: ["buttons", "menu-full"];
        A(U, {
            menuCallback: W
        })
    }
    function H() {
        var U;
        if (M) {
            U = M.all(".share-this-v3");
            if (U && U.size()) {
                U.remove()
            }
        }
    }
    function A(aa, X) {
        var W, ab, U, Z = [], V = "/share.gne", ac = [], Y;
        if (!aa) {
            return false
        }
        if (c.Lang.isString(aa)) {
            aa = [aa]
        }
        if (!c.Lang.isArray(aa)) {
            return false
        }
        X = X || {};
        W = S();
        U = {
            buttons: {
                on: {
                    start: function() {
                        if (!M) {
                            return false
                        }
                        if (p) {
                            return false
                        }
                        y();
                        return true
                    },
                    success: function(ad) {
                        var ae;
                        if (M) {
                            ae = c.Node.create(ad);
                            if (ae) {
                                M.all(".share-this-v3").remove();
                                if (c.config.flickr.flags.enable_2013_photo_page && (E || J)) {
                                    M.all(".divider").insert(ae, "before")
                                } else {
                                    M.append(ae)
                                }
                                P.set("button_node", q());
                                if (I) {
                                    I.set("button_node", c.one("#post-upload-share-more-button"))
                                }
                            }
                        }
                    }
                }
            },
            "menu-grab-the-link": {
                on: {
                    start: function() {
                        if (X.reload) {
                            if (!ab["menu-grab-the-link"].placeholder_node) {
                                ab["menu-grab-the-link"].placeholder_node = T.one("#share-options-grablink ." + t);
                                if (!ab["menu-grab-the-link"].placeholder_node) {
                                    ab["menu-grab-the-link"].placeholder_node = T.one("#share-options-grablink .share-options-inner")
                                }
                            }
                            if (X.guestpass) {
                                ac.push("guestpass=1");
                                ab["menu-grab-the-link"].placeholder_node.all("form#grab-link input.gp-type:checked").each(function(ad) {
                                    ac.push(ad.getAttribute("name") + "=1")
                                })
                            }
                            ab["menu-grab-the-link"].loaded = false;
                            ab["menu-grab-the-link"].placeholder_node.addClass(t);
                            ab["menu-grab-the-link"].placeholder_node.set("innerHTML", "")
                        }
                        if (ab["menu-grab-the-link"].loaded) {
                            return false
                        }
                        if (T.one("form#grab-link")) {
                            ab["menu-grab-the-link"].loaded = true;
                            return false
                        }
                        if (!ab["menu-grab-the-link"].placeholder_node) {
                            ab["menu-grab-the-link"].placeholder_node = T.one("#share-options-grablink ." + t);
                            if (!ab["menu-grab-the-link"].placeholder_node) {
                                c.log("Missing placeholder node", "error", "share-this-v3-menu");
                                return false
                            }
                        }
                        ab["menu-grab-the-link"].placeholder_node.set("innerHTML", "").addClass(t);
                        return true
                    },
                    success: function(ad) {
                        ab["menu-grab-the-link"].loaded = true;
                        ab["menu-grab-the-link"].placeholder_node.set("innerHTML", ad);
                        if (X.guestpass) {
                            if (c.config.flickr.page_type === "photo" && c.photo && c.photo.getCurrentPhoto) {
                                c.use("photo-data", function(af) {
                                    var ae = af.photo.getCurrentPhoto();
                                    if (ae) {
                                        ae.increment("guestpass_count")
                                    }
                                })
                            }
                        }
                    },
                    failure: function() {
                        ab["menu-grab-the-link"].placeholder_node.set("innerHTML", c.transjax.get("share-menu", "grab_the_link_fragment_error"))
                    },
                    complete: function() {
                        ab["menu-grab-the-link"].placeholder_node.removeClass(t)
                    }
                }
            },
            "menu-services": {
                on: {
                    start: function() {
                        if (X.reload) {
                            if (!ab["menu-services"].placeholder_node) {
                                ab["menu-services"].placeholder_node = T.one("#share-options-services ." + t);
                                if (!ab["menu-services"].placeholder_node) {
                                    ab["menu-services"].placeholder_node = T.one("#share-options-services .share-options-inner")
                                }
                            }
                            ab["menu-services"].loaded = false;
                            ab["menu-services"].placeholder_node.addClass(t);
                            ab["menu-services"].placeholder_node.set("innerHTML", "")
                        }
                        if (ab["menu-services"].loaded) {
                            return false
                        }
                        if (T.one("ul.share-services")) {
                            ab["menu-services"].loaded = true;
                            return false
                        }
                        if (!ab["menu-services"].placeholder_node) {
                            ab["menu-services"].placeholder_node = T.one("#share-options-services ." + t);
                            if (!ab["menu-services"].placeholder_node) {
                                c.log("Missing placeholder node", "error", "share-this-v3-menu");
                                return false
                            }
                        }
                        return true
                    },
                    success: function(ad) {
                        ab["menu-services"].loaded = true;
                        ab["menu-services"].placeholder_node.removeClass(t);
                        ab["menu-services"].placeholder_node.set("innerHTML", ad)
                    },
                    failure: function(af, ae, ad) {
                        ab["menu-services"].placeholder_node.set("innerHTML", c.transjax.get("share-menu", "services_fragment_error"))
                    },
                    complete: function() {
                        ab["menu-services"].placeholder_node.removeClass(t)
                    }
                }
            },
            "menu-full": {
                on: {
                    start: function() {
                        f();
                        return true
                    },
                    success: function(ad) {
                        if (T) {
                            T.set("innerHTML", ad)
                        }
                        i();
                        if (X.menuCallback) {
                            X.menuCallback()
                        }
                    }
                }
            }
        };
        if (!A.memo) {
            A.memo = {};
            c.Object.each(U, function(ae, ad) {
                A.memo[ad] = {
                    fetching: {}
                }
            })
        }
        ab = A.memo;
        c.Array.each(aa, function(af, ad) {
            var ae;
            if (ab[af].fetching[W]) {
                return false
            }
            ae = U[af].on.start.apply(window);
            if (ae) {
                Z.push(af)
            }
        });
        if (Z.length === 0) {
            return false
        }
        if (c.config.flickr.sharing.object.id) {
            ac.push("object_id=" + c.config.flickr.sharing.object.id)
        }
        if (c.config.flickr.sharing.object.type) {
            ac.push("object_type=" + c.config.flickr.sharing.object.type)
        }
        if (c.config.flickr.sharing.object.owner_nsid) {
            ac.push("owner_nsid=" + c.config.flickr.sharing.object.owner_nsid);
            ac.push("object_owner_nsid=" + c.config.flickr.sharing.object.owner_nsid)
        }
        ac.push("cachebust=" + (new Date()).getTime());
        ac.push("action=retrieve_fragments");
        c.Array.each(Z, function(ad) {
            ac.push("fragments[]=" + ad)
        });
        if (F.config.flickr.page_type === "photo") {
            ac.push("is_photo_page=true")
        }
        ac = ac.join("&");
        V = V + "?" + ac;
        Y = {
            on: {
                success: function(af, ae) {
                    var ad = ae.responseText.split("<!-- Fragment -->");
                    c.Array.each(Z, function(ah, ag) {
                        if (W === S()) {
                            if (c.Lang.isFunction(U[ah].on.success)) {
                                U[ah].on.success.apply(window, [ad[ag]])
                            }
                            if (c.Lang.isFunction(U[ah].on.complete)) {
                                U[ah].on.complete.apply(window, [])
                            }
                        }
                        delete ab[ah].fetching[W]
                    })
                },
                failure: function(af, ae) {
                    var ad = arguments;
                    c.Array.each(Z, function(ag) {
                        if (W === S()) {
                            if (c.Lang.isFunction(U[ag].on.failure)) {
                                U[ag].on.failure.apply(window, ad)
                            }
                            if (c.Lang.isFunction(U[ag].on.complete)) {
                                U[ag].on.complete.apply(window, [])
                            }
                        }
                        delete ab[ag].fetching[W]
                    })
                }
            }
        };
        c.Array.each(Z, function(ad) {
            ab[ad].fetching[W] = true
        });
        c.io(V, Y)
    }
    function r() {
        if (P) {
            P.hide()
        }
    }
    c.shareThisV3Menu = {
        init: N,
        show: function() {
            P.show()
        },
        hide: r,
        openPanel: j,
        disableButtons: y,
        refresh: R,
        clear: H,
        getMenu: function() {
            return P
        },
        isInitialized: function() {
            return m
        }
    }
}, "0.0.1", {
    requires: F.config.modules["share-this-v3-menu"].requires || [],
    optional: F.config.modules["share-this-v3-menu"].optional || []
});
YUI.add("popup-login", function(a) {
    var e = "/photo_grease_postlogin.gne", i = false, r = false, p = false, s, q = "head-upload-link", c = "signin-popup", m = /cookie_session=[^;]+/gi, b = a.global_dialog;
    function o() {
        var t = {
            method: "post",
            data: "random=0",
            on: {
                success: function(w, v, u) {
                    if (v && (v.responseText === "1,1" || v.responseText === "1,0")) {
                        a.fire("login:authenticated", v.responseText)
                    } else {
                        d()
                    }
                }
            }
        };
        a.later(100, this, h);
        a.io("/fragment.gne?name=social-auth-fragment", t)
    }
    function h() {
        if (b) {
            b.show({
                loading: true,
                modal: true
            })
        }
    }
    function d() {
        if (b) {
            b.hide()
        }
    }
    function l(w) {
        var z = 650;
        var x;
        if (a.config.flickr.flags.enable_ads_on_login_page) {
            x = a.one(window).get("winWidth") || 0;
            if (!isNaN(z)) {
                z = Math.max(880, Math.min(1200, x))
            }
        }
        if (a.config.flickr.is_touch_device) {
            var v = e + "?d=" + window.location + "&notpopup=1";
            window.location = w + "?popup=0&redir=" + encodeURIComponent(v);
            return false
        }
        var u = w + "?popup=1&redir=" + e + "?d=" + window.location;
        try {
            s = window.open(u, "newWindow", "width=" + z + ",height=650,resizable=1,scrollbars=1,location=yes")
        } catch (y) {
            return true
        }
        try {
            if (s.focus) {
                s.focus()
            }
        } catch (t) {}
        a.later(500, this, function() {
            if (!s || s.closed || typeof s.closed === "undefined" ||!i) {
                var A = e + "?d=" + window.location + "&notpopup=1";
                window.location = w + "?popup=0&redir=" + encodeURIComponent(A)
            }
        });
        return false
    }
    function j() {
        a.later(20, this, f);
        r = true;
        l("/signin")
    }
    var k = 0;
    function f() {
        if (document.cookie.match(m)) {
            r = false;
            o()
        } else {
            a.later(20, this, f)
        }
    }
    function g() {
        var t = false;
        if (s) {
            try {
                s.close();
                t = true
            } catch (u) {}
        }
        return t
    }
    function n() {
        if (a.config.flickr.user.nsid) {
            return 
        }
        a.one("body").on("click", function(v) {
            var u = v.target.get("id");
            if (v.target.ancestor("." + c, true)) {
                v.preventDefault();
                v.halt();
                j()
            } else {
                if (u === q) {
                    p = true;
                    v.halt();
                    j()
                }
            }
        });
        var t = a.one("window");
        t.on("login|blur", function(u) {
            i = true
        });
        t.on("login|focus", function(u) {
            i = false;
            if (r) {
                o()
            }
        });
        a.on("login:authenticated", function(v) {
            var u = window.location.toString();
            g();
            if (v === "1,1") {
                window.location = "/"
            } else {
                if (p) {
                    window.location = "/photos/upload"
                } else {
                    if (u.match(/\/new/i)) {
                        window.location = "/"
                    } else {
                        if (window.location.pathname && window.location.pathname === "/") {
                            window.location = "/"
                        } else {
                            window.location.reload()
                        }
                    }
                }
            }
        })
    }
    a.popup_login = {
        init: n
    }
}, "0.0.1", {
    requires: ["node", "event", "io-base"]
});
YUI.add("liquid-resizer", function(b) {
    function a(c) {
        b.on("resize", b.betterThrottle(function() {
            var d = F.liquid.resizePage(true);
            b.fire("Liquid:resize", d)
        }, 20))
    }
    b.liquidResizer = {
        init: a
    }
}, "0.0.1", {
    requires: F.config.modules["liquid-resizer"].requires || [],
    optional: F.config.modules["liquid-resizer"].optional || []
});
if (typeof deconcept == "undefined") {
    var deconcept = new Object()
}
if (typeof deconcept.util == "undefined") {
    deconcept.util = new Object()
}
if (typeof deconcept.SWFObjectUtil == "undefined") {
    deconcept.SWFObjectUtil = new Object()
}
deconcept._playerVersion = null;
deconcept.SWFObject = function(f, d, m, g, j, l, o, i, a, e, n) {
    if (!document.getElementById) {
        return 
    }
    this.DETECT_KEY = e ? e : "detectflash";
    this.skipDetect = deconcept.util.getRequestParameter(this.DETECT_KEY);
    this.params = new Object();
    this.variables = new Object();
    this.attributes = new Array();
    if (f) {
        this.setAttribute("swf", f)
    }
    if (d) {
        this.setAttribute("id", d)
    }
    if (m) {
        this.setAttribute("width", m)
    }
    if (g) {
        this.setAttribute("height", g)
    }
    this.setAttribute("base", (n) ? n : ".");
    if (j) {
        this.setAttribute("version", new deconcept.PlayerVersion(j.toString().split(".")))
    }
    this.installedVer = deconcept.SWFObjectUtil.getPlayerVersion();
    if (!window.opera && document.all && this.installedVer.major > 7) {
        deconcept.SWFObject.doPrepUnload = true
    }
    if (l) {
        this.addParam("bgcolor", l)
    }
    var b = o ? o: "high";
    this.addParam("quality", b);
    this.setAttribute("useExpressInstall", false);
    this.setAttribute("doExpressInstall", false);
    var k = (i) ? i: window.location;
    this.setAttribute("xiRedirectUrl", k);
    this.setAttribute("redirectUrl", "");
    if (a) {
        this.setAttribute("redirectUrl", a)
    }
};
deconcept.SWFObject.prototype = {
    useExpressInstall: function(a) {
        this.xiSWFPath=!a ? "expressinstall.swf" : a;
        this.setAttribute("useExpressInstall", true)
    },
    setAttribute: function(a, b) {
        this.attributes[a] = b
    },
    getAttribute: function(a) {
        return this.attributes[a]
    },
    addParam: function(a, b) {
        this.params[a] = b
    },
    getParams: function() {
        return this.params
    },
    addVariable: function(a, b) {
        this.variables[a] = b
    },
    getVariable: function(a) {
        return this.variables[a]
    },
    getVariables: function() {
        return this.variables
    },
    getVariablePairs: function() {
        var a = new Array();
        var b;
        var c = this.getVariables();
        for (b in c) {
            a[a.length] = b + "=" + c[b]
        }
        return a
    },
    getSWFHTML: function(e) {
        var d = "";
        if (navigator.plugins && navigator.mimeTypes && navigator.mimeTypes.length) {
            if (this.getAttribute("doExpressInstall")) {
                this.addVariable("MMplayerType", "PlugIn");
                this.setAttribute("swf", this.xiSWFPath)
            }
            d = '<embed type="application/x-shockwave-flash" src="' + this.getAttribute("swf") + '" width="' + this.getAttribute("width") + '" height="' + this.getAttribute("height") + '" style="' + this.getAttribute("style") + '"';
            d += ' id="' + this.getAttribute("id") + '" name="' + this.getAttribute("id") + '" base="' + this.getAttribute("base") + '" ';
            var c = this.getParams();
            for (var a in c) {
                d += [a] + '="' + c[a] + '" '
            }
            var b = this.getVariablePairs().join("&");
            if (b.length > 0) {
                d += 'flashvars="' + b + '"'
            }
            d += "/>"
        } else {
            if (this.getAttribute("doExpressInstall")) {
                this.addVariable("MMplayerType", "ActiveX");
                this.setAttribute("swf", this.xiSWFPath)
            }
            d = '<object id="' + this.getAttribute("id") + '" classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" width="' + this.getAttribute("width") + '" height="' + this.getAttribute("height") + '" style="' + this.getAttribute("style") + '">';
            d += '<param name="movie" value="' + this.getAttribute("swf") + '" />';
            d += '<param name="base" value="' + this.getAttribute("base") + '" />';
            var c = this.getParams();
            for (var a in c) {
                d += '<param name="' + a + '" value="' + c[a] + '" />'
            }
            var b = this.getVariablePairs().join("&");
            if (b.length > 0) {
                d += '<param name="flashvars" value="' + b + '" />'
            }
            d += "</object>"
        }
        d = (typeof e != "undefined" ? e : "") + d;
        return d
    },
    write: function(a, d) {
        if (this.getAttribute("useExpressInstall")) {
            var b = new deconcept.PlayerVersion([6, 0, 65]);
            if (this.installedVer.versionIsValid(b)&&!this.installedVer.versionIsValid(this.getAttribute("version"))) {
                this.setAttribute("doExpressInstall", true);
                this.addVariable("MMredirectURL", escape(this.getAttribute("xiRedirectUrl")));
                document.title = document.title.slice(0, 47) + " - Flash Player Installation";
                this.addVariable("MMdoctitle", document.title)
            }
        }
        if (this.skipDetect || this.getAttribute("doExpressInstall") || this.installedVer.versionIsValid(this.getAttribute("version"))) {
            var c = (typeof a == "string") ? document.getElementById(a): a;
            c.innerHTML = this.getSWFHTML(d);
            return true
        } else {
            if (this.getAttribute("redirectUrl") != "") {
                document.location.replace(this.getAttribute("redirectUrl"))
            }
        }
        return false
    }
};
deconcept.SWFObjectUtil.getPlayerVersion = function() {
    if (deconcept._playerVersion) {
        return deconcept._playerVersion
    }
    var c = new deconcept.PlayerVersion([0, 0, 0]);
    if (navigator.plugins && navigator.mimeTypes.length) {
        var a = navigator.plugins["Shockwave Flash"];
        if (a && a.description) {
            a.description.match(/[a-zA-Z\s]([0-9]+)\.([0-9]+)\s*[rbd]([0-9]+)/);
            var g = [RegExp.$1, RegExp.$2, RegExp.$3];
            if (navigator.platform.match(/win32/i)&&!g[0]) {
                a.description.match(/[a-zA-Z\s]([0-9]+)\.([0-9]+|([0-9]+.[0-9]+.))\s*[rbd]([0-9]+)/);
                g = [RegExp.$1, RegExp.$2, RegExp.$3]
            }
            c = new deconcept.PlayerVersion(g)
        }
    } else {
        if (navigator.userAgent && navigator.userAgent.indexOf("Windows CE") >= 0) {
            var d = 1;
            var b = 3;
            while (d) {
                try {
                    b++;
                    d = new ActiveXObject("ShockwaveFlash.ShockwaveFlash." + b);
                    c = new deconcept.PlayerVersion([b, 0, 0])
                } catch (f) {
                    d = null
                }
            }
        } else {
            try {
                var d = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7")
            } catch (f) {
                try {
                    var d = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");
                    c = new deconcept.PlayerVersion([6, 0, 21]);
                    d.AllowScriptAccess = "always"
                } catch (f) {
                    if (c.major == 6) {
                        return c
                    }
                }
                try {
                    d = new ActiveXObject("ShockwaveFlash.ShockwaveFlash")
                } catch (f) {}
            }
            if (d != null) {
                c = new deconcept.PlayerVersion(d.GetVariable("$version").split(" ")[1].split(","))
            }
        }
    }
    deconcept._playerVersion = c;
    return c
};
deconcept.PlayerVersion = function(a) {
    this.major = a[0] != null ? parseInt(a[0]) : 0;
    this.minor = a[1] != null ? parseInt(a[1]) : 0;
    this.rev = a[2] != null ? parseInt(a[2]) : 0
};
deconcept.PlayerVersion.prototype.versionIsValid = function(a) {
    if (this.major < a.major) {
        return false
    }
    if (this.major > a.major) {
        return true
    }
    if (this.minor < a.minor) {
        return false
    }
    if (this.minor > a.minor) {
        return true
    }
    if (this.rev < a.rev) {
        return false
    }
    return true
};
deconcept.util = {
    getRequestParameter: function(d) {
        var c = document.location.search || document.location.hash;
        if (d == null) {
            return c
        }
        if (c) {
            var b = c.substring(1).split("&");
            for (var a = 0; a < b.length; a++) {
                if (b[a].substring(0, b[a].indexOf("=")) == d) {
                    return b[a].substring((b[a].indexOf("=") + 1))
                }
            }
        }
        return ""
    }
};
deconcept.SWFObjectUtil.cleanupSWFs = function() {
    var c = document.getElementsByTagName("OBJECT");
    for (var b = c.length - 1; b >= 0; b--) {
        c[b].style.display = "none";
        for (var a in c[b]) {
            if (typeof c[b][a] == "function") {
                c[b][a] = function() {}
            }
        }
    }
};
if (deconcept.SWFObject.doPrepUnload) {
    if (!deconcept.unloadSet) {
        deconcept.SWFObjectUtil.prepUnload = function() {
            __flash_unloadHandler = function() {};
            __flash_savedUnloadHandler = function() {};
            window.attachEvent("onunload", deconcept.SWFObjectUtil.cleanupSWFs)
        };
        window.attachEvent("onbeforeunload", deconcept.SWFObjectUtil.prepUnload);
        deconcept.unloadSet = true
    }
}
if (!document.getElementById && document.all) {
    document.getElementById = function(a) {
        return document.all[a]
    }
}
var getQueryParamValue = deconcept.util.getRequestParameter;
var FlashObject = deconcept.SWFObject;
var SWFObject = deconcept.SWFObject;
function SWFMacMouseWheel(a) {
    this.so = a;
    var b = navigator.appVersion.toLowerCase().indexOf("mac")!=-1;
    if (b) {
        this.init()
    }
}
SWFMacMouseWheel.prototype = {
    init: function() {
        SWFMacMouseWheel.instance = this;
        if (window.addEventListener) {
            window.addEventListener("DOMMouseScroll", SWFMacMouseWheel.instance.wheel, false)
        }
        window.onmousewheel = document.onmousewheel = SWFMacMouseWheel.instance.wheel
    },
    handle: function(a) {
        document[this.so.getAttribute("id")].externalMouseEvent(a)
    },
    wheel: function(a) {
        var b = 0;
        if (a.wheelDelta) {
            b = a.wheelDelta / 120;
            if (window.opera) {
                b =- b
            }
        } else {
            if (a.detail) {
                b =- a.detail / 3
            }
        }
        if (/AppleWebKit/.test(navigator.userAgent)) {
            b/=3
        }
        if (b) {
            SWFMacMouseWheel.instance.handle(b)
        }
        if (a.preventDefault) {
            a.preventDefault()
        }
        a.returnValue = false
    }
};
(function() {
    var a = "urls";
    YUI.add(a, function(b) {
        var h = {
            photo: /^\/photos\/([^\/]+)\/(\d+)(?:\/in\/([^\/]+))?(?:(?:\/page(\d+))|(?:\/(lightbox))|(?:\/(tour)))?/,
            photostream: /^\/photos\/([^\/]+)(?:\/page(\d+))?(?:\/with\/(\d+))?\/?(lightbox\/?)?/,
            delete_photo: /^\/photos\/([^\/]+)\/?\?delete=(\d+)(?:&context=([^&]+))?/,
            faves: /^\/photos\/([^\/]+)\/favorites(?:\/from\/([^\/]+))?(?:\/page(\d+))?(?:\/with\/(\d+))?(?:\/(lightbox))?\/?(?:\?(?:[^&]*&)?view=([^&]+))?/,
            explore: /^\/explore\/?(\d{4}\/\d{2}\/\d{2})?(?:\/?with\/([0-9]+))?\/?(lightbox\/?)?/,
            group_pool: /^\/groups\/([^\/]+)\/?(?:pool)?(?:\/((?:(?!with|page|lightbox)[^\/\?])+))?(?:\/with\/(\d+))?(?:\/page(\d+))?(\/lightbox)?\/?(?:\?(?:[^&]*&)?view=([^&]+))?/,
            gallery: /^\/photos\/([^\/]+)\/galleries\/([0-9]+)(?:\/page(\d+))?(?:\/with\/(\d+))?(?:\/(lightbox))?/,
            contacts: /^\/photos\/(?:friends|contacts)(?:\/page(\d+))?(?:\/with\/(\d+))?(?:\/(lightbox))?\/?(?:\?(?:[^&]*&)?view=([^&]+))?/,
            contactsfaves: /^\/photos\/contacts\/favorites(?:\/page(\d+))?(?:\/with\/(\d+))?(?:\/(lightbox))?\/?(?:\?(?:[^&]*&)?view=([^&]+))?/
        };
        function e(r, t) {
            var q, s = {};
            if (!h[r]) {
                b.log('[urls] no regex found for page_type "' + r + '"');
                return 
            }
            q = h[r].exec(t);
            switch (r) {
            case"photo":
                s.path_alias = q[1] && decodeURIComponent(q[1]);
                s.photo_id = q[2];
                s.context_id = q[3] && decodeURIComponent(q[3]);
                s.page = q[4];
                s.lightbox = q[5] ? true : false;
                s.tour = q[6] ? true : false;
                break;
            case"photostream":
                s.path_alias = q[1] && decodeURIComponent(q[1]);
                s.page = q[2];
                s.photo_id = q[3];
                s.lightbox = q[4] ? true : false;
                break;
            case"delete_photo":
                s.path_alias = q[1] && decodeURIComponent(q[1]);
                s.photo_id = q[2];
                s.context_id = q[3];
                break;
            case"faves":
                s.path_alias = q[1] && decodeURIComponent(q[1]);
                s.from = q[2] && decodeURIComponent(q[2]);
                s.page = q[3];
                s.photo_id = q[4];
                s.lightbox = q[5] ? true : false;
                s.view_mode = q[6];
                break;
            case"group_pool":
                s.group_id = q[1] && decodeURIComponent(q[1]);
                s.owner_id = q[2] && decodeURIComponent(q[2]);
                s.photo_id = q[3];
                s.page = q[4];
                s.lightbox = q[5] ? true : false;
                s.view_mode = q[6];
                break;
            case"gallery":
                s.path_alias = q[1] && decodeURIComponent(q[1]);
                s.gallery_id = q[2] && decodeURIComponent(q[2]);
                s.page = q[3];
                s.photo_id = q[4];
                s.lightbox = q[5] ? true : false;
                break;
            case"contacts":
                s.page = q[1];
                s.photo_id = q[2];
                s.lightbox = q[3] ? true : false;
                s.view_mode = q[4];
                break;
            case"contactsfaves":
                s.page = q[1];
                s.photo_id = q[2];
                s.lightbox = q[3] ? true : false;
                s.view_mode = q[4];
                break;
            case"explore":
                s.date = q[1];
                s.photo_id = q[2];
                s.lightbox = q[3]
            }
            return s
        }
        function c(s, r) {
            var q;
            r = r || {};
            if (r.photo && b.Lang.isString(r.photo)) {
                r.photo_id = r.photo;
                if (b.PhotoData) {
                    r.photo = b.PhotoData.get(r.photo)
                }
                if (!b.Lang.isObject(r.photo)) {
                    r.photo = undefined;
                    delete r.photo
                }
            }
            if (b.Lang.isObject(r.photo)) {
                r.photo_id = r.photo.get("id")
            }
            switch (s) {
            case"photo":
                q = p(r);
                break;
            case"photostream":
                q = l(r);
                break;
            case"delete_photo":
                q = o(r);
                break;
            case"faves":
                q = n(r);
                break;
            case"group_pool":
                q = i(r);
                break;
            case"gallery":
                q = d(r);
                break;
            case"contacts":
                q = m(r);
                break;
            case"contactsfaves":
                q = k(r);
                break;
            case"explore":
                q = g(r);
                break
            }
            return q
        }
        function p(v) {
            var s, t, u, r, z, x, B, y, A, w, q;
            s = v.photo;
            t = v.photo_id;
            u = v.path_alias;
            if (!b.Lang.isObject(s)&&!t) {
                b.log("[urls] missing photo object or ID");
                return 
            }
            if (!b.Lang.isObject(s)&&!u) {
                b.log("[urls] missing photo object or path alias");
                return 
            }
            t = t || s.get("id");
            u = u || s.get("pathalias") || s.get("owner");
            if (v.context) {
                if (b.Lang.isString(v.context)) {
                    r = v.context
                } else {
                    if (b.Lang.isObject(v.context) && b.Lang.isString(v.context.id)) {
                        r = v.context.id
                    }
                }
            }
            if (v.page) {
                z = parseInt(v.page, 10);
                if (z === 1 ||!b.Lang.isNumber(z)) {
                    z = false
                }
            }
            x = b.Lang.isBoolean(v.lightbox) ? v.lightbox : false;
            B = (b.Lang.isBoolean(v.sizes) || b.Lang.isString(v.sizes)) ? v.sizes : false;
            y = b.Lang.isString(v.hash) ? v.hash : false;
            A = b.Lang.isBoolean(v.exif) ? v.exif : false;
            w = b.Lang.isBoolean(v.faves) ? v.faves : false;
            q = b.Lang.isBoolean(v.details) ? v.details : false;
            return "/photos/" + u + "/" + t + "/" + (B ? "sizes/" + (b.Lang.isString(B) ? B + "/" : "") : "") + (A ? "meta/" : "") + (w ? "favorites/" : "") + (q ? "edit-details/" : "") + (r ? ("in/" + r + "/") : "") + (z ? "page" + z + "/" : "") + (x ? "lightbox/" : "") + (y ? "#" + y : "")
        }
        function l(r) {
            var s, q;
            if (r.photo) {
                r.path_alias = r.photo.get("pathalias") || r.photo.get("owner")
            }
            s = r.path_alias;
            if (!s) {
                b.log("[urls] missing path_alias");
                return 
            }
            q = "/photos/" + s + "/";
            if (r.person) {
                q += "people/" + r.person + "/"
            } else {
                if (r.tag) {
                    if (b.Lang.isArray(r.tag)) {
                        r.tag = r.tag.join(",")
                    }
                    q += "tags/" + r.tag + "/"
                } else {
                    if (r.photo_id) {
                        q += "with/" + r.photo_id + "/"
                    }
                }
            }
            if (r.lightbox) {
                q += "lightbox/"
            }
            return q
        }
        function o(r) {
            var s, q;
            s = r.path_alias;
            if (!s) {
                b.log("[urls] missing path_alias");
                return 
            }
            photo_id = r.photo_id;
            if (!photo_id) {
                b.log("[urls] missing photo_id");
                return 
            }
            context_id = r.context_id;
            q = "/photos/" + s + "/?delete=" + photo_id + (context_id ? "&context=" + context_id : "") + "&magic_cookie=" + b.config.flickr.magic_cookie;
            return q
        }
        function n(r) {
            var s, q;
            s = r.path_alias;
            if (!s) {
                b.log("[urls] missing path_alias");
                return 
            }
            q = "/photos/" + s + "/favorites/";
            if (r.from) {
                q += "from/" + r.from + "/"
            }
            if (r.page) {
                q += "page" + r.page + "/"
            }
            if (r.photo_id) {
                q += "with/" + r.photo_id + "/"
            }
            if (r.lightbox) {
                q += "lightbox/"
            }
            if (r.view_mode) {
                q += "?view=" + r.view_mode
            }
            return q
        }
        function i(r) {
            var s, q;
            s = r.group_id;
            if (!s) {
                b.log("[urls] missing group_id");
                return 
            }
            q = "/groups/" + s + "/pool/";
            if (r.owner_id) {
                q += r.owner_id + "/"
            }
            if (r.photo_id) {
                q += "with/" + r.photo_id + "/"
            }
            if (r.page) {
                q += "page" + r.page + "/"
            }
            if (r.lightbox) {
                q += "lightbox/"
            }
            if (r.view_mode) {
                q += "?view=" + r.view_mode
            }
            return q
        }
        function m(r) {
            var q;
            q = "/photos/friends/";
            if (r.page) {
                q += "page" + r.page + "/"
            }
            if (r.photo_id) {
                q += "with/" + r.photo_id + "/"
            }
            if (r.lightbox) {
                q += "lightbox/"
            }
            if (r.view_mode) {
                q += "?view=" + r.view_mode
            }
            return q
        }
        function k(r) {
            var q;
            q = "/photos/contacts/favorites/";
            if (r.page) {
                q += "page" + r.page + "/"
            }
            if (r.photo_id) {
                q += "with/" + r.photo_id + "/"
            }
            if (r.lightbox) {
                q += "lightbox/"
            }
            if (r.view_mode) {
                q += "?view=" + r.view_mode
            }
            return q
        }
        function d(s) {
            var t, q, r;
            t = s.path_alias;
            q = s.gallery_id;
            if (!t) {
                b.log("[urls] missing path_alias");
                return 
            }
            if (!q) {
                b.log("[urls] missing gallery_id");
                return 
            }
            r = "/photos/" + t + "/galleries/" + q + "/";
            if (s.photo_id) {
                r += "with/" + s.photo_id + "/"
            }
            if (s.lightbox) {
                r += "lightbox/"
            }
            return r
        }
        function g(s) {
            var r, t, q;
            r = s.date;
            t = s.with_id;
            q = "/explore";
            if (r) {
                q += "/" + r
            }
            if (t) {
                q += "/with/" + t
            }
            if (s.lightbox) {
                q += "lightbox/"
            }
            return q
        }
        function j(r) {
            if (!b.Lang.isNumber(r)) {
                r = parseInt(r, 10)
            }
            var q = "", t = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ";
            var u = r, s;
            while (r >= 58) {
                u = r / 58;
                s = r - (58 * Math.floor(u));
                q = "" + t.substr(s, 1) + q;
                r = Math.floor(u)
            }
            return (u) ? "" + t.substr(u, 1) + q : q
        }
        function f(v) {
            var u = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ";
            var r = v.length;
            var q = 0;
            var t = 1;
            for (var s = (r - 1); s >= 0; s--) {
                q = q + t * u.indexOf(v[s]);
                t = t * u.length
            }
            return q
        }
        b.urls = {
            get: c,
            parse: e,
            getShortID: j,
            getLongID: f,
            regexes: h
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
YUI.add("html5-balls", function(d) {
    var b = '<div class="html5-balls"><div class="blueball"></div><div class="pinkball"></div></div>';
    if (d.UA.ie && d.UA.ie < 9) {
        b = '<div class="html4-balls"></div>'
    }
    var c = {
        width: {
            value: 24
        },
        node: {
            validator: function(e) {
                if (e.one) {
                    return true
                } else {
                    return false
                }
            }
        }
    };
    var a = function(e) {
        this.set("node", e.node);
        this.set("width", e.width);
        var f = this.get("node");
        var g = f.create(b);
        g.setStyle("width", this.get("width"));
        f.set("innerHTML", b)
    };
    d.augment(a, d.Attribute);
    d.Balls = a
}, "0.0.1", {
    requires: F.config.modules["html5-balls"].requires || [],
    optional: F.config.modules["html5-balls"].optional || []
});
(function() {
    var a = "view-count";
    YUI.add(a, function(g) {
        var f;
        f = {
            queued: {},
            logging: {},
            logged: {},
            pendingFlush: undefined
        };
        function b(k, j, h) {
            var l, i;
            if (!k ||!c(k)) {
                return 
            }
            if (!j ||!g.Lang.isString(j)) {
                j = document.referrer
            }
            if (j.charAt(0) === "/") {
                l = true
            }
            i = {};
            i.photo_id = k;
            if (h) {
                i.async = true
            }
            if (l) {
                i.path = j
            } else {
                i.referrer = j
            }
            f.queued[k] = i;
            if (!f.pendingFlush) {
                f.pendingFlush = setTimeout(e, 1000)
            }
        }
        function e() {
            if (f.pendingFlush) {
                clearTimeout(f.pendingFlush);
                f.pendingFlush = undefined;
                d()
            }
        }
        function d() {
            var i, h, j, n, k, m, l;
            i = g.config.flickr.max_photo_ids_per_api_call;
            h = f.queued;
            f.queued = {};
            j = g.Object.keys(h);
            if (i && i < j.length) {
                g.Array.each(j, function(p, o) {
                    if (o >= i) {
                        b(h[p]);
                        h[p] = undefined;
                        delete h[p]
                    }
                })
            }
            g.mix(f.logging, h);
            n = "flickr.photos.addView";
            k = {
                views: g.JSON.stringify(g.Object.values(h))
            };
            m = {
                success: function(o) {
                    g.mix(f.logged, h);
                    l()
                },
                failure: function(o) {
                    l()
                }
            };
            l = function() {
                g.Object.each(h, function(o, p) {
                    f.logging[p] = undefined;
                    delete f.logging[p]
                })
            };
            g.flickrAPI.callMethod(n, k, m)
        }
        function c(h) {
            if (f.queued[h] || f.logging[h] || f.logged[h]) {
                return false
            }
            return true
        }
        g.ViewCount = {
            canQueue: c,
            queue: b,
            flushQueue: e
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
(function() {
    var a = "view-count-on-visible";
    YUI.add(a, function(c) {
        var f = 30, b = {};
        function h() {
            var l;
            if (c.config.flickr.flags.view_count_on_visible || c.config.flickr.flags.enable_sihp_viewcount || c.config.flickr.flags.enable_photostream_viewcount) {
                l = true
            } else {
                l = false
            }
            return l
        }
        function k(l) {
            if (!h()) {
                return 
            }
            if (c.Lang.isString(l)) {
                l = c.all(l)
            }
            if (l instanceof c.Node) {
                l = c.all([l._node])
            }
            if (!l instanceof c.NodeList ||!l.size()) {
                return false
            }
            c.later(20, null, function() {
                l.each(function(q, p, n) {
                    var r = q.get("id"), o, m;
                    if (!r) {
                        r = c.guid();
                        q.set("id", r)
                    }
                    if (b[r]) {
                        if (b[r].node === q) {
                            return 
                        }
                        if (b[r].handler) {
                            g(b[r].node, b[r].triggered)
                        }
                        if (b[r].triggered) {
                            return 
                        }
                    }
                    node_width = q.get("offsetWidth");
                    o = q.get("offsetHeight");
                    m = [0 - Math.floor(0.8 * (node_width || f)), 0 - Math.floor(0.8 * (o || f))];
                    if (b[r]) {
                        b[r].handler = q.vis.onVisible(j, m);
                        b[r].node = q
                    } else {
                        b[r] = {
                            handler: q.vis.onVisible(j, m),
                            node: q,
                            id: r,
                            triggered: false
                        }
                    }
                })
            });
            return {
                unregister: function() {
                    g(l, false)
                },
                checkVisible: function() {
                    d(l)
                },
                setThreshold: function(m) {
                    e(l, m)
                }
            }
        }
        function g(m, l) {
            if (m instanceof c.Node) {
                m = c.all([m._node])
            }
            m.each(function(p, o, n) {
                var q = p.get("id");
                if (c.Lang.isObject(b[q]) && c.Lang.isFunction(b[q].handler)) {
                    b[q].handler.cancel();
                    b[q].triggered = (l === false) ? false : true;
                    b[q].handler = null;
                    b[q].node = null
                }
            })
        }
        function j(l) {
            var o, n, m;
            o = [l.get("offsetHeight"), l.get("offsetWidth")];
            if (Math.max(o[0], o[1]) >= f) {
                n = l.getAttribute("src") || l.getAttribute("data-defer-src");
                if (n) {
                    m = c.Node.create("<img>");
                    m.on("load", function() {
                        i(l);
                        m.remove(true)
                    });
                    m.setAttribute("src", n)
                }
            }
        }
        function i(o) {
            var n, m, l;
            if (o) {
                n = o.ancestor("[data-photo-id]", true);
                if (n) {
                    m = n.getAttribute("data-photo-id");
                    if (m) {
                        l = window.location.href.toLowerCase();
                        c.ViewCount.queue(m, l, true)
                    }
                }
                g(o, true)
            }
        }
        function d(l) {
            l.each(function(o, n, m) {
                if (c.Lang.isFunction(b[o.get("id")].handler)) {
                    b[o.get("id")].handler.check()
                }
            })
        }
        function e(l, m) {
            l.each(function(p, o, n) {
                if (c.Lang.isFunction(b[p.get("id")].handler)) {
                    b[p.get("id")].handler.set("threshold", m)
                }
            })
        }
        c.ViewCount.onVisible = k;
        c.ViewCount.onVisibleEnabled = h
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}()); /*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("io-form", function(e, t) {
    var n = encodeURIComponent;
    e.IO.stringify = function(t, n) {
        n = n || {};
        var r = e.IO.prototype._serialize({
            id: t,
            useDisabled: n.useDisabled
        }, n.extra && typeof n.extra == "object" ? e.QueryString.stringify(n.extra) : n.extra);
        return r
    }, e.mix(e.IO.prototype, {
        _serialize: function(t, r) {
            var i = [], s = t.useDisabled ||!1, o = 0, u = typeof t.id == "string" ? t.id: t.id.getAttribute("id"), a, f, l, c, h, p, d, v, m, g;
            u || (u = e.guid("io:"), t.id.setAttribute("id", u)), f = e.config.doc.getElementById(u);
            if (!f ||!f.elements)
                return r || "";
            for (p = 0, d = f.elements.length; p < d; ++p) {
                a = f.elements[p], h = a.disabled, l = a.name;
                if (s ? l : l&&!h) {
                    l = n(l) + "=", c = n(a.value);
                    switch (a.type) {
                    case"select-one":
                        a.selectedIndex>-1 && (g = a.options[a.selectedIndex], i[o++] = l + n(g.attributes.value && g.attributes.value.specified ? g.value : g.text));
                        break;
                    case"select-multiple":
                        if (a.selectedIndex>-1)
                            for (v = a.selectedIndex, m = a.options.length; v < m; ++v)
                                g = a.options[v], g.selected && (i[o++] = l + n(g.attributes.value && g.attributes.value.specified ? g.value : g.text));
                        break;
                    case"radio":
                    case"checkbox":
                        a.checked && (i[o++] = l + c);
                        break;
                    case"file":
                    case undefined:
                    case"reset":
                    case"button":
                        break;
                    case"submit":
                    default:
                        i[o++] = l + c
                    }
                }
            }
            return r && (i[o++] = r), i.join("&")
        }
    }, !0)
}, "3.11.0", {
    requires: ["io-base", "node-base"]
});
/*
YUI 3.11.0 (build d549e5c)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("substitute", function(e, t) {
    var n = e.Lang, r = "dump", i = " ", s = "{", o = "}", u = /(~-(\d+)-~)/g, a = /\{LBRACE\}/g, f = /\{RBRACE\}/g, l = function(t, l, c, h) {
        var p, d, v, m, g, y, b = [], w, E, S = t.length;
        for (; ;) {
            p = t.lastIndexOf(s, S);
            if (p < 0)
                break;
            d = t.indexOf(o, p);
            if (p + 1 >= d)
                break;
            w = t.substring(p + 1, d), m = w, y = null, v = m.indexOf(i), v>-1 && (y = m.substring(v + 1), m = m.substring(0, v)), g = l[m], c && (g = c(m, g, y)), n.isObject(g) ? e.dump ? n.isArray(g) ? g = e.dump(g, parseInt(y, 10)) : (y = y || "", E = y.indexOf(r), E>-1 && (y = y.substring(4)), g.toString === Object.prototype.toString || E>-1 ? g = e.dump(g, parseInt(y, 10)) : g = g.toString()) : g = g.toString() : n.isUndefined(g) && (g = "~-" + b.length + "-~", b.push(w)), t = t.substring(0, p) + g + t.substring(d + 1), h || (S = p - 1)
        }
        return t.replace(u, function(e, t, n) {
            return s + b[parseInt(n, 10)] + o
        }).replace(a, s).replace(f, o)
    };
    e.substitute = l, n.substitute = l
}, "3.11.0", {
    requires: ["yui-base"],
    optional: ["dump"]
});
YUI.add("flickr-dialog-base", function(f) {
    f.FlickrDialog = {};
    var d = '<div class="flickr-dialog {dialogClass} hidden" role="{role}"><div class="flickr-dialog-container"><div class="flickr-dialog-content-wrapper">{dialogContentHTML}</div></div></div>', c = '<div class="flickr-dialog-mask hidden"></div>', a = 20001, b = 20000, e = 0.35;
    f.FlickrDialog.Base = function(g) {
        f.Attribute.call(this, f.FlickrDialog.ATTRS, g);
        this._createAndInsertHTML();
        this._attachEvents()
    };
    f.extend(f.FlickrDialog.Base, f.Attribute);
    f.FlickrDialog.ATTRS = {
        dialogContentHTML: {
            value: "",
            validator: f.Lang.isString
        },
        width: {
            value: 330,
            validator: function(g) {
                return (f.Lang.isNumber(g) || g === "auto")
            }
        },
        modal: {
            value: true,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        context: {
            valueFn: function() {
                return "flickr-dialog-" + f.guid()
            },
            validator: f.Lang.isString,
            writeOnce: "initOnly"
        },
        dialogClass: {
            value: "flickr-dialog-base",
            validator: f.Lang.isString
        },
        role: {
            value: "dialog",
            validator: f.Lang.isString,
            writeOnce: "initOnly"
        },
        allowClicksOnMaskToHide: {
            value: false,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        allowReturnKeyToConfirm: {
            value: false,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        allowEscKeyToCancel: {
            value: true,
            validator: f.Lang.isBoolean
        },
        allowEventsToPropagate: {
            value: true,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        allowFocusOnFirstElement: {
            value: true,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        restoreFocusOnHide: {
            value: true,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        setAriaLabelledby: {
            value: true,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        allowModalToShrink: {
            value: false,
            validator: f.Lang.isBoolean,
            writeOnce: "initOnly"
        },
        shrunkWidth: {
            value: 280,
            validator: f.Lang.isNumber
        },
        shrunkClass: {
            value: "flickr-dialog-shrunk",
            validator: f.Lang.isString
        },
        autoCenter: {
            value: true,
            validator: f.Lang.isNumber
        },
        resizeDetectionThrottle: {
            value: 50,
            validator: f.Lang.isNumber
        },
        dialogNode: {
            value: null
        },
        maskNode: {
            value: null
        },
        primaryControl: {
            value: null
        },
        timestamp: {
            value: null,
            validator: f.Lang.isNumber
        },
        showing: {
            value: false,
            validator: f.Lang.isBoolean
        },
        stackingOrder: {
            value: 0,
            validator: f.Lang.isNumber
        }
    };
    f.FlickrDialog.Base.prototype._createAndInsertHTML = function() {
        var h, g;
        this.set("timestamp", (new Date()).getTime());
        this.set("dialogNode", f.Node.create(f.substitute(d, {
            dialogClass: this.get("dialogClass"),
            role: this.get("role"),
            dialogContentHTML: this.get("dialogContentHTML")
        })));
        f.one("body").append(this.get("dialogNode"));
        this.set("maskNode", f.Node.create(c));
        f.one("body").append(this.get("maskNode"));
        this.set("primaryControl", (this.get("dialogNode").one("input, button, textarea") || this.get("dialogNode")));
        this.get("dialogNode").setStyle("width", (this.get("width") === "auto" ? this.get("width") : this.get("width") + "px"));
        if (this.get("setAriaLabelledby")) {
            h = this.get("dialogNode").one(".hd h1, .hd h2, .hd h3, .hd h4, .hd h5, .hd h6");
            if (h) {
                g = h.generateID();
                this.get("dialogNode").setAttribute("aria-labelledby", g)
            }
        }
    };
    f.FlickrDialog.Base.prototype._attachEvents = function() {
        if (this.get("allowClicksOnMaskToHide")) {
            this.get("maskNode").on("click", this._hideAdapter, this)
        }
        if (this.get("allowReturnKeyToConfirm")) {
            f.keyboardShortcutManager.register({
                keystring: "13",
                handler: this._confirmAdapter,
                context: this.get("context"),
                scope: this
            })
        }
        f.keyboardShortcutManager.register({
            keystring: "27",
            handler: this._hideAdapter,
            context: this.get("context"),
            scope: this
        });
        if (this.get("modal")) {
            f.on(["FlickrDialog:show", "FlickrDialog:hide"], this._adjustMaskOpacity, this)
        }
        this.after("dialogContentHTMLChange", function(g) {
            this.get("dialogNode").one(".flickr-dialog-content-wrapper").set("innerHTML", g.newVal)
        });
        this.after(["widthChange", "shrunkWidthChange"], function(g) {
            this.get("dialogNode").setStyle("width", (this.get("width") === "auto" ? this.get("width") : this.get("width") + "px"));
            if (this.get("autoCenter")) {
                if (this.get("allowModalToShrink")) {
                    this._modalWidth = this.get("dialogNode").get("offsetWidth");
                    this._modalHeight = this.get("dialogNode").get("offsetHeight");
                    this._checkWindowSize({
                        widthWasChanged: true
                    })
                } else {
                    this.center()
                }
            }
        });
        this.after("dialogClassChange", function(g) {
            this.get("dialogNode").removeClass(g.prevVal).addClass(g.newVal)
        });
        this.after("shrunkClassChange", function(g) {
            if (this._currentlyShrunk) {
                this.get("dialogNode").removeClass(g.prevVal).addClass(g.newVal)
            }
        })
    };
    f.FlickrDialog.Base.prototype._hideAdapter = function(g) {
        if (g.charCode === 27&&!this.get("allowEscKeyToCancel")) {
            return 
        }
        if (!this.get("allowEventsToPropagate")) {
            g.stopPropagation()
        }
        this.hide()
    };
    f.FlickrDialog.Base.prototype._confirmAdapter = function(g) {
        var h = f.focusTracker.get();
        if (!h ||!h.hasClass("CancelButt")) {
            this.hide(true)
        }
    };
    f.FlickrDialog.Base.prototype._stack = function(g) {
        this.get("dialogNode").setStyle("zIndex", a + (2 * f.FlickrDialog.Base.visibleDialogs));
        this.get("maskNode").setStyle("zIndex", b + (2 * f.FlickrDialog.Base.visibleDialogs))
    };
    f.FlickrDialog.Base.prototype._unstack = function(g) {
        this.get("dialogNode").setStyle("zIndex", "");
        this.get("maskNode").setStyle("zIndex", "")
    };
    f.FlickrDialog.Base.prototype._adjustMaskOpacity = function(j) {
        var g, i, h;
        if (!this.get("showing")) {
            return 
        }
        if (this.get("stackingOrder") === 0 || f.FlickrDialog.Base.visibleModalDialogs === 1) {
            this.get("maskNode").setStyle("opacity", e)
        } else {
            g = 1 - e;
            i = Math.pow(g, 1 / f.FlickrDialog.Base.visibleModalDialogs);
            h = 1 - i;
            this.get("maskNode").setStyle("opacity", h)
        }
    };
    f.FlickrDialog.Base.prototype._enforceModality = function(g) {
        if (this.isTopDialog()&&!this.get("dialogNode").contains(g.target)) {
            this.get("primaryControl").focus()
        }
    };
    f.FlickrDialog.Base.prototype._checkWindowSize = function(k) {
        var g = f.DOM.winWidth(), h = f.DOM.winHeight(), i = (!this._currentlyShrunk && (this._modalWidth > g || this._modalHeight > h)) || (this._currentlyShrunk && (this._modalWidth <= g && this._modalHeight <= h)), j = k.widthWasChanged;
        if (this.get("allowModalToShrink") && (i || j)) {
            if ((!this._currentlyShrunk || j) && (this._modalWidth > g || this._modalHeight > h)) {
                this.get("dialogNode").setStyle("width", (this.get("shrunkWidth") === "auto" ? this.get("shrunkWidth") : this.get("shrunkWidth") + "px"));
                this.get("dialogNode").addClass(this.get("shrunkClass"));
                this._currentlyShrunk = true;
                this.fire("shrunk")
            } else {
                this.get("dialogNode").setStyle("width", this.get("width") === "auto" ? this.get("width") : this.get("width") + "px");
                this.get("dialogNode").removeClass(this.get("shrunkClass"));
                this._currentlyShrunk = false;
                this.fire("unshrunk")
            }
            if (this.get("autoCenter")) {
                this.center()
            }
        }
        this.fire("viewportResized", {
            viewportHeight: h,
            viewportWidth: g,
            hasChanged: i,
            modalHeight: this._modalHeight,
            modalWidth: this._modalWidth
        })
    };
    f.FlickrDialog.Base.prototype.show = function(g) {
        if (this.get("showing")) {
            return 
        }
        if (this.get("restoreFocusOnHide")) {
            this._previouslySelectedNode = f.focusTracker.get()
        }
        this.get("dialogNode").removeClass("hidden");
        this.center();
        this._stack();
        if (this.get("modal")) {
            this.get("maskNode").removeClass("hidden");
            this._modalityHandler = f.one(f.config.doc).on("focus", this._enforceModality, this)
        }
        this._modalWidth = this.get("dialogNode").get("offsetWidth");
        this._modalHeight = this.get("dialogNode").get("offsetHeight");
        this._windowResizeHandler = f.on("resize", f.betterThrottle(this._checkWindowSize, this.get("resizeDetectionThrottle"), this));
        f.PageContext.set(this.get("context"));
        this.set("showing", true);
        f.FlickrDialog.Base.visibleDialogs++;
        if (this.get("modal")) {
            f.FlickrDialog.Base.visibleModalDialogs++
        }
        this.set("stackingOrder", f.FlickrDialog.Base.visibleDialogs);
        this.fire("show", g);
        f.fire("FlickrDialog:show");
        if (this.get("allowFocusOnFirstElement")) {
            f.focusTracker.blur();
            this.get("primaryControl").focus()
        }
    };
    f.FlickrDialog.Base.prototype.hide = function(g) {
        if (!this.get("showing")) {
            return 
        }
        if (this.get("modal")) {
            this._modalityHandler.detach()
        }
        if (this.get("allowModalToShrink")) {
            this._windowResizeHandler.detach()
        }
        if (f.focusTracker.isInput()) {
            f.focusTracker.blur()
        }
        this.get("dialogNode").addClass("hidden");
        this.get("maskNode").addClass("hidden");
        if (this._previouslySelectedNode) {
            f.later(0, this._previouslySelectedNode, this._previouslySelectedNode.focus)
        }
        f.PageContext.unset(this.get("context"));
        this._unstack();
        this.set("showing", false);
        f.FlickrDialog.Base.visibleDialogs--;
        if (this.get("modal")) {
            f.FlickrDialog.Base.visibleModalDialogs--
        }
        this.set("stackingOrder", 0);
        this.fire("hide", g);
        f.fire("FlickrDialog:hide")
    };
    f.FlickrDialog.Base.prototype.center = function() {
        var g = this.get("dialogNode").get("offsetWidth"), i = this.get("dialogNode").get("offsetHeight");
        this.get("dialogNode").setStyle("marginLeft", - (g / 2) + "px");
        this.get("dialogNode").setStyle("marginTop", - (i / 2) + "px");
        this.fire("center")
    };
    f.FlickrDialog.Base.prototype.isTopDialog = function() {
        return (this.get("stackingOrder") === f.FlickrDialog.Base.visibleDialogs)
    };
    f.FlickrDialog.Base.prototype.destroy = function(g) {
        this.hide();
        this.get("dialogNode").empty();
        this.get("maskNode").empty();
        this.get("dialogNode").remove(true);
        this.get("maskNode").remove(true);
        this.fire("destroy", g);
        f.fire("FlickrDialog:destroy")
    };
    f.FlickrDialog.Base.visibleDialogs = 0;
    f.FlickrDialog.Base.visibleModalDialogs = 0
}, "0.0.1", {
    requires: F.config.modules["flickr-dialog-base"].requires || [],
    optional: F.config.modules["flickr-dialog-base"].optional || []
});
YUI.add("flickr-dialog-alert", function(d) {
    var c = '<div class="hd">{message}</div><div class="bd button-container"><input type="button" class="{confirmButtonClass}" value="{confirmButtonText}" id="{confirmButtonId}"></div>';
    d.FlickrDialog.Alert = function(e) {
        var f, g;
        f = {
            message: "",
            confirmButtonText: d.transjax.get("global-dialog", "button_confirm"),
            confirmButtonClass: "Butt",
            confirmButtonId: d.guid(),
            callback: function(h) {},
            dialogClass: "flickr-dialog-alert",
            role: "alertdialog",
            allowReturnKeyToConfirm: true
        };
        e = d.mix(e || {}, f);
        e.dialogContentHTML = d.substitute(c, e);
        d.FlickrDialog.Alert.superclass.constructor.call(this, e);
        g = {
            message: {
                value: f.message,
                validator: d.Lang.isString,
                writeOnce: "initOnly"
            },
            confirmButtonText: {
                value: f.confirmButtonText,
                validator: d.Lang.isString,
                writeOnce: "initOnly"
            },
            confirmButtonClass: {
                value: f.confirmButtonClass,
                validator: d.Lang.isString,
                writeOnce: "initOnly"
            },
            confirmButtonId: {
                value: f.confirmButtonId,
                validator: d.Lang.isString,
                writeOnce: "initOnly"
            },
            callback: {
                value: f.callback,
                validator: d.Lang.isFunction,
                writeOnce: "initOnly"
            }
        };
        this.addAttrs(g, e);
        d.one("#" + this.get("confirmButtonId")).on("click", this.hide, this);
        this.on("hide", this.get("callback"), this)
    };
    var a = {}, b = {};
    d.extend(d.FlickrDialog.Alert, d.FlickrDialog.Base, a, b)
}, "0.0.1", {
    requires: F.config.modules["flickr-dialog-alert"].requires || [],
    optional: F.config.modules["flickr-dialog-alert"].optional || []
});
YUI.add("flickr-dialog-confirmation", function(c) {
    var e = '{titleHTML}<div class="bd">{message}{photoHTML}</div><div class="bd button-container"><input type="button" class="{confirmButtonClass}" value="{confirmButtonText}" id="{confirmButtonId}"> <input type="button" class="{cancelButtonClass}" value="{cancelButtonText}" id="{cancelButtonId}"></div>', d = '<div class="hd"><h3>{title}</h3></div>';
    PHOTO_TEMPLATE = '<div class="photo-container"><img src="{url}" width="{width}" height="{height}"></div>';
    c.FlickrDialog.ConfirmationDialog = function(f) {
        var g, h;
        g = {
            title: "",
            message: "",
            photo: null,
            confirmButtonText: c.transjax.get("global-dialog", "button_confirm"),
            confirmButtonClass: "Butt",
            confirmButtonId: c.guid(),
            cancelButtonText: c.transjax.get("global-dialog", "button_cancel"),
            cancelButtonClass: "CancelButt",
            cancelButtonId: c.guid(),
            callback: function(i) {},
            dialogClass: "flickr-dialog-confirmation",
            role: "alertdialog",
            allowReturnKeyToConfirm: true
        };
        f = c.mix(f || {}, g);
        f.titleHTML = (f.title) ? c.substitute(d, f) : "";
        f.photoHTML = (f.photo) ? c.substitute(PHOTO_TEMPLATE, f.photo) : "";
        f.dialogContentHTML = c.substitute(e, f);
        c.FlickrDialog.ConfirmationDialog.superclass.constructor.call(this, f);
        h = {
            title: {
                value: g.title,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            message: {
                value: g.message,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            photo: {
                value: g.photo,
                writeOnce: "initOnly"
            },
            confirmButtonText: {
                value: g.confirmButtonText,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            confirmButtonClass: {
                value: g.confirmButtonClass,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            confirmButtonId: {
                value: g.confirmButtonId,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            cancelButtonText: {
                value: g.cancelButtonText,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            cancelButtonClass: {
                value: g.cancelButtonClass,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            cancelButtonId: {
                value: g.cancelButtonId,
                validator: c.Lang.isString,
                writeOnce: "initOnly"
            },
            callback: {
                value: g.callback,
                writeOnce: "initOnly"
            }
        };
        this.addAttrs(h, f);
        c.one("#" + this.get("confirmButtonId")).on("click", this._checkConfirmationAndHide, this, true);
        c.one("#" + this.get("cancelButtonId")).on("click", this._checkConfirmationAndHide, this);
        this.on("hide", this._invokeCallback, this)
    };
    var a = {}, b = {};
    a._checkConfirmationAndHide = function(f, g) {
        this.hide(g)
    };
    a._invokeCallback = function(f, g) {
        if (typeof this.get("callback") === "function") {
            this.get("callback")(g)
        } else {
            if (g && typeof this.get("callback").confirmed === "function") {
                this.get("callback").confirmed()
            } else {
                if (!g && typeof this.get("callback").cancelled === "function") {
                    this.get("callback").cancelled()
                }
            }
        }
    };
    c.extend(c.FlickrDialog.ConfirmationDialog, c.FlickrDialog.Base, a, b)
}, "0.0.1", {
    requires: F.config.modules["flickr-dialog-confirmation"].requires || [],
    optional: F.config.modules["flickr-dialog-confirmation"].optional || []
});
YUI.add("flickr-dialog-destructive-confirmation", function(c) {
    c.FlickrDialog.DestructiveConfirmationDialog = function(d) {
        var e;
        e = {
            confirmButtonClass: "DeleteButt"
        };
        d = c.mix(d || {}, e);
        c.FlickrDialog.DestructiveConfirmationDialog.superclass.constructor.call(this, d);
        this.set("primaryControl", c.one("#" + this.get("cancelButtonId")))
    };
    var a = {}, b = {};
    c.extend(c.FlickrDialog.DestructiveConfirmationDialog, c.FlickrDialog.ConfirmationDialog, a, b)
}, "0.0.1", {
    requires: F.config.modules["flickr-dialog-destructive-confirmation"].requires || [],
    optional: F.config.modules["flickr-dialog-destructive-confirmation"].optional || []
});
YUI.add("flickr-dialog-frame", function(c) {
    c.FlickrDialog.Frame = function(d) {
        var e = {
            dialogClass: "flickr-dialog-frame"
        };
        d = c.mix(d || {}, e);
        c.FlickrDialog.Frame.superclass.constructor.call(this, d)
    };
    var a = {}, b = {};
    c.extend(c.FlickrDialog.Frame, c.FlickrDialog.Base, a, b)
}, "0.0.1", {
    requires: F.config.modules["flickr-dialog-frame"].requires || [],
    optional: F.config.modules["flickr-dialog-frame"].optional || []
});
YUI.add("flickr-dialog-short-message", function(d) {
    var a = '<span class="short-msg-{type}"></span>{message}';
    d.FlickrDialog.ShortMessage = function(e) {
        var f, g;
        f = {
            message: "",
            width: "auto",
            type: "no-icon",
            callback: function(h) {},
            dialogClass: "flickr-dialog-short-message",
            role: "alertdialog",
            allowClicksOnMaskToHide: true,
            allowReturnKeyToConfirm: true,
            timeToHide: 4000
        };
        e = d.mix(e || {}, f);
        e.dialogContentHTML = d.substitute(a, e);
        d.FlickrDialog.ShortMessage.superclass.constructor.call(this, e);
        g = {
            message: {
                value: f.message,
                validator: d.Lang.isString,
                writeOnce: "initOnly"
            },
            callback: {
                value: f.callback,
                writeOnce: "initOnly"
            },
            timeToHide: {
                value: f.timeToHide,
                validator: d.Lang.isNumber,
                writeOnce: "initOnly"
            }
        };
        this.addAttrs(g, e);
        this.on("hide", this.get("callback"), this);
        if (this.get("timeToHide")) {
            this.on("show", this._startFinalCountdown, this);
            this.on("hide", this._cancelFinalCountdown, this)
        }
    };
    var b = {}, c = {};
    b._startFinalCountdown = function() {
        this._timer = d.later(this.get("timeToHide"), this, this.hide)
    };
    b._cancelFinalCountdown = function() {
        this._timer.cancel()
    };
    d.extend(d.FlickrDialog.ShortMessage, d.FlickrDialog.Base, b, c)
}, "0.0.1", {
    requires: F.config.modules["flickr-dialog-short-message"].requires || [],
    optional: F.config.modules["flickr-dialog-short-message"].optional || []
});
YUI.add("flickr-dialog-spinner", function(c) {
    c.FlickrDialog.Spinner = function(d) {
        var e = {
            dialogClass: "flickr-dialog-spinner",
            width: "auto",
            allowEscKeyToCancel: false
        };
        d = c.mix(d || {}, e);
        c.FlickrDialog.Spinner.superclass.constructor.call(this, d)
    };
    var a = {}, b = {};
    c.extend(c.FlickrDialog.Spinner, c.FlickrDialog.Base, a, b)
}, "0.0.1", {
    requires: F.config.modules["flickr-dialog-spinner"].requires || [],
    optional: F.config.modules["flickr-dialog-spinner"].optional || []
});
YUI.add("flickr-dialog", function(a) {}, "0.0.1", {
    requires: F.config.modules["flickr-dialog"].requires || [],
    optional: F.config.modules["flickr-dialog"].optional || []
});
YUI.add("photo-comments-transjax", function(a) {
    a.transjax.add("photo-comments", {
        error_delete_permission: "Sorry, but you don\'t have permission to delete that comment, or you aren\'t signed into Flickr.",
        error_delete_generic: 'Sorry, but there was a problem deleting the comment.',
        error_edit_getting: 'Sorry, but there was a problem getting the comment to edit.',
        error_edit_generic: 'Sorry, but there was a problem editing the comment.',
        error_add_rate_limit: 'Slow down, partner! To protect the Flickr community against spam, we place a few restrictions on new accounts. You can read more about these restrictions in <a href="/help/website/#870679">this FAQ</a>.',
        error_add_generic: 'We had a problem saving your comment.',
        error_getting_comments: 'Sorry, but there was a problem getting the comments for this photo.',
        error_empty_body: "An empty comment box? That won't work!",
        reply_wipe_confirmation: 'You already started typing a comment. If you continue, that text will be lost!',
        reply_button_confirm: 'OKAY',
        reply_button_cancel: 'CANCEL',
        delete_confirmation: 'Are you sure you want to delete this comment?',
        delete_button_confirm: 'YES, DELETE IT',
        delete_button_cancel: 'CANCEL',
        hide_formatting_tips: 'Hide formatting tips',
        faves_rollup_placeholder: '...and %s more favorites',
        block_user: 'Block this user from commenting on your stuff in the future.',
        user_blocked: 'You are currently blocking this Flickr member. They cannot add any more comments to any of your stuff. You can remove this via the blocked list on your "Contacts" page.',
        deleting: 'Deleting...',
        trigger_recompile: "",
        bert_lorem_ipsum: "Lorem Ip\'sum lorem"
    })
}, "0.0.1", {
    requires: ["transjax-base"]
});
YUI.add("gallery-storage-lite", function(b) {
    var s = b.config.doc, j = b.config.win, c = b.JSON, r = b.namespace("StorageLite"), i = "yui_storage_lite", v = "YUI StorageLite data", y = 1048576, p = "1.0", t = "ready", x = 0, h = 1, a = 2, m = 3, k = 4, f = "yui_storage_lite", l = "data", u = {}, n, g;
    try {
        if (j.localStorage) {
            g = h
        } else {
            if (j.globalStorage) {
                g = a;
                var o = j.globalStorage[j.location.hostname]
            } else {
                if (j.openDatabase && navigator.userAgent.indexOf("Chrome")===-1) {
                    g = m
                } else {
                    if (b.UA.ie >= 5) {
                        g = k
                    } else {
                        g = x
                    }
                }
            }
        }
    } catch (q) {
        g = x
    }
    b.StorageFullError = function(d) {
        b.StorageFullError.superclass.constructor.call(d);
        this.name = "StorageFullError";
        this.message = d || "Maximum storage capacity reached";
        if (b.UA.ie) {
            this.description = this.message
        }
    };
    b.extend(b.StorageFullError, Error);
    b.augment(r, b.EventTarget, true, null, {
        emitFacade: true,
        prefix: "storage-lite",
        preventable: false
    });
    r.publish(t, {
        fireOnce: true
    });
    b.mix(r, {
        clear: function() {},
        getItem: function(e, d) {
            return null
        },
        length: function() {
            return 0
        },
        removeItem: function(d) {},
        setItem: function(d, e) {}
    });
    if (g === h || g === a) {
        b.mix(r, {
            length: function() {
                return n.length
            },
            removeItem: function(d) {
                n.removeItem(d)
            },
            setItem: function(e, w, d) {
                n.setItem(e, d ? c.stringify(w) : w)
            }
        }, true);
        if (g === h) {
            n = j.localStorage;
            b.mix(r, {
                clear: function() {
                    n.clear()
                },
                getItem: function(w, e) {
                    try {
                        return e ? c.parse(n.getItem(w)) : n.getItem(w)
                    } catch (d) {
                        return null
                    }
                }
            }, true)
        } else {
            if (g === a) {
                n = j.globalStorage[j.location.hostname];
                b.mix(r, {
                    clear: function() {
                        for (var d in n) {
                            if (n.hasOwnProperty(d)) {
                                n.removeItem(d);
                                delete n[d]
                            }
                        }
                    },
                    getItem: function(w, e) {
                        try {
                            return e ? c.parse(n[w].value) : n[w].value
                        } catch (d) {
                            return null
                        }
                    }
                }, true)
            }
        }
        r.fire(t)
    } else {
        if (g === m || g === k) {
            b.mix(r, {
                clear: function() {
                    u = {};
                    r._save()
                },
                getItem: function(e, d) {
                    return u.hasOwnProperty(e) ? u[e] : null
                },
                length: function() {
                    var e = 0, d;
                    for (d in u) {
                        if (u.hasOwnProperty(d)) {
                            e += 1
                        }
                    }
                    return e
                },
                removeItem: function(d) {
                    if (u) {
                        delete u[d];
                        r._save()
                    } else {
                        u = {}
                    }
                },
                setItem: function(e, w, d) {
                    u[e] = w;
                    r._save()
                }
            }, true);
            if (g === m) {
                n = j.openDatabase(i, p, v, y);
                b.mix(r, {
                    _save: function() {
                        n.transaction(function(d) {
                            d.executeSql("REPLACE INTO " + i + " (name, value) VALUES ('data', ?)", [c.stringify(u)])
                        })
                    }
                }, true);
                n.transaction(function(d) {
                    d.executeSql("CREATE TABLE IF NOT EXISTS " + i + "(name TEXT PRIMARY KEY, value TEXT NOT NULL)");
                    d.executeSql("SELECT value FROM " + i + " WHERE name = 'data'", [], function(z, w) {
                        if (w.rows.length) {
                            try {
                                u = c.parse(w.rows.item(0).value)
                            } catch (e) {
                                u = {}
                            }
                        }
                        r.fire(t)
                    })
                })
            } else {
                if (g === k) {
                    n = s.createElement("span");
                    n.addBehavior("#default#userData");
                    b.mix(r, {
                        _save: function() {
                            var e = c.stringify(u);
                            try {
                                n.setAttribute(l, e);
                                n.save(f)
                            } catch (d) {
                                throw new b.StorageFullError()
                            }
                        }
                    }, true);
                    b.on("domready", function() {
                        s.body.appendChild(n);
                        n.load(f);
                        try {
                            u = c.parse(n.getAttribute(l))
                        } catch (d) {
                            u = {}
                        }
                        r.fire(t)
                    })
                }
            }
        } else {
            r.fire(t)
        }
    }
}, "1.0.0", {
    requires: ["event-base", "event-custom", "event-custom-complex", "json"]
});
YUI.add("occult", function(e) {
    if (typeof window.FLICKR === "undefined") {
        window.FLICKR = {};
        var d = window.FLICKR
    }
    function c() {
        var f = arguments, l = null, h, g, k;
        for (h = 0; h < f.length; h = h + 1) {
            k = ("" + f[h]).split(".");
            l = FLICKR;
            for (g = (k[0] == "FLICKR") ? 1 : 0; g < k.length; g = g + 1) {
                l[k[g]] = l[k[g]] || {};
                l = l[k[g]]
            }
        }
        return l
    }
    function b(h, l) {
        c(h);
        var g = h.split("."), f = g.length, k = FLICKR;
        for (var j = 0; j < g.length - 1; j++) {
            k = k[g[j]]
        }
        k[g[f - 1]] = l
    }
    function a(f, g) {
        return d[f]
    }
    e.occult = {
        register: b,
        get: a
    }
}, "0.0.1", {});
YUI.add("photo-transjax", function(a) {
    a.transjax.add("photo", {
        your_favorites: 'Your favorites',
        untitled: 'Untitled',
        window_title: '%s | Flickr - Photo Sharing!'
    })
}, "0.0.1", {
    requires: ["transjax-base"]
});
(function() {
    var a = "type-cast";
    YUI.add(a, function(h) {
        function e(i) {
            if (h.Lang.isNumber(i)) {
                i = i + ""
            }
            return h.Lang.isString(i) ? i : ""
        }
        function c(i) {
            if (i === "0") {
                i = 0
            }
            return !!i
        }
        function d(i) {
            i = parseInt(i, 10);
            if (!h.Lang.isNumber(i)) {
                return undefined
            }
            return i
        }
        function f(i) {
            i = d(i);
            if (!h.Lang.isNumber(i) || i < 0 || i > 3) {
                return undefined
            }
            return i
        }
        function b(i) {
            return h.Lang.isArray(i) ? i : []
        }
        function g(i) {
            return h.Lang.isObject(i) ? i : {}
        }
        h.TypeCast = {
            string: e,
            bool: c,
            integer: d,
            integer0123: f,
            arr: b,
            obj: g
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
(function() {
    var a = "grease";
    YUI.add(a, function(d) {
        var X = false, S, al, P = d.StorageLite, T = false, s, t, z, Q, W, E = "pc", y = "pic", u = "pf", g = "pct", R = "ps", Z = "ps3", j = "pp", q = "pps", aa = "/photo_grease_postlogin.gne", x = "fave", ad = "contact", ae = "share", G = "share_v3", h = "comment", e = /cookie_session=[^;]+/gi, l = ";", an = 3, O = 1000, L = [E, y, R, Z, u, g, u, j, q];
        if (d.config.flickr.flags.enable_grease) {
            X = true
        }
        Q = d.global_dialog;
        function p() {
            d.Array.each(L, function(Y) {
                P.removeItem(Y)
            })
        }
        var ah = (function() {
            var Y = null, ay = false, aq = 0, at = false, ar = false;
            function aw() {
                O = (O >= 10000) ? O : O * 2;
                return O
            }
            function ax() {
                if (ar) {
                    if (Y) {
                        Q.remove_existing_dialog();
                        Y.cancel();
                        ay = false
                    }
                    ar = false
                }
            }
            function au() {
                ar = false;
                if (Y) {
                    Y.cancel()
                }
                ax();
                Q.remove_existing_dialog();
                ay = false;
                Y = null;
                at = false;
                aq = 0;
                O = 1000;
                at = false
            }
            function ap() {
                if (Y) {
                    Y.cancel()
                }
                at = true;
                d.fire("grease:cancel")
            }
            function av() {
                if (at) {
                    return 
                }
                if (Y) {
                    Y.cancel();
                    Y = null
                }
                var az = {
                    method: "post",
                    data: "random=0",
                    on: {
                        success: function(aC, aB, aA) {
                            if (aB && aB.responseText.substring(0, 1) === "1") {
                                d.fire("grease:authenticated")
                            } else {
                                if (aq++<an) {
                                    m();
                                    if (!Y) {
                                        Y = d.later(aw(), this, av)
                                    }
                                }
                            }
                        },
                        failure: function() {
                            if (aq++<1) {
                                if (!Y) {
                                    Y = d.later(aw(), this, av)
                                }
                            }
                        }
                    }
                };
                d.io("/fragment.gne?name=social-auth-fragment", az)
            }
            return {
                start: function() {
                    if (!ar) {
                        ar = true;
                        av()
                    } else {
                        O = 50;
                        Y && Y.cancel();
                        av();
                        return 
                    }
                },
                stop: ax,
                cancel: ap,
                pause: function() {
                    Y && Y.cancel();
                    O = 1000000
                },
                canceled: at,
                reset: au
            }
        }());
        function af(Y) {
            if (!X) {
                return 
            }
            if (Y && d.config.flickr.user.nsid) {
                P.on("storage-lite:ready", function() {
                    var aq = ag(E), av = ag(y), aB = ag(u), at = ag(g), ax = ag(R, true), aw = ag(Z, true), ap = ag(j), aA = ag(q);
                    if (aq) {
                        z = "comment";
                        var az = aq.indexOf(";"), ar = aq.substring(0, az), ay = aq.substring(az + 1);
                        if (ar === d.photo.getCurrentPhoto().get("id")) {
                            ab(ay)
                        }
                    } else {
                        if (av) {
                            z = "comment";
                            var az = av.indexOf(";"), ar = av.substring(0, az), ay = av.substring(az + 1);
                            C(ay, ar)
                        } else {
                            if (aB) {
                                z = "fave";
                                V(aB)
                            } else {
                                if (at) {
                                    z = "contact";
                                    o(at)
                                } else {
                                    if (ax) {
                                        z = "share";
                                        I(ax)
                                    } else {
                                        if (aw) {
                                            z = "sharev3";
                                            aj(aw)
                                        } else {
                                            var au = d.one("#global-dialog-background");
                                            if (au) {
                                                au.setStyle("display", "none")
                                            }
                                            p()
                                        }
                                    }
                                }
                            }
                        }
                    }
                })
            } else {
                P.on("storage-lite:ready", function() {
                    p()
                })
            }
        }
        function c() {
            d.detach("grease|*");
            S = false;
            m();
            W(false);
            ah.reset()
        }
        function f(Y) {
            if (Y.target.hasClass("close-x") || Y.target.hasClass("Butt")) {
                Y.preventDefault();
                Y.halt();
                d.global_dialog.hide()
            }
        }
        function A() {
            Q.show({
                loading: true,
                modal: true
            })
        }
        function m() {
            Q.hide()
        }
        function ak(ap) {
            var aq = d.one("body");
            var Y = aq.create('<img height="1" width="1" src="' + ap + '">');
            aq.appendChild(Y)
        }
        function J() {
            Q.create_dialog(d.one("#grease-confirm-" + z).get("innerHTML"), {
                width: 355
            }).show({
                loading: false,
                modal: true
            });
            d.one("#global_dialog").on("click", f, this);
            d.later(10000, this, d.global_dialog.hide);
            if (d.config.flickr.flags.enable_rapid_tracking && d.one("#reg-tour")) {
                d.use("rapid-tracker", function(ap) {
                    ap.rapidTracker.addModules("grease-conversion-tracker");
                    ap.rapidTracker.beaconClick("grease-conversion-tracker", z);
                    switch (z) {
                    case x:
                        ak("https://pclick.internal.yahoo.com/p/s=2143640346&t=" + Math.random());
                        break;
                    case ad:
                        ak("https://pclick.internal.yahoo.com/p/s=2143640345&t=" + Math.random());
                        break;
                    case ae:
                        ak("https://pclick.internal.yahoo.com/p/s=2143640344&t=" + Math.random());
                        break;
                    default:
                        ak("https://pclick.internal.yahoo.com/p/s=2143640343&t=" + Math.random())
                    }
                })
            }
        }
        function V(ap) {
            var at, Y, au, aq, ar = true;
            if (d.photo && d.photo.getCurrentPhoto && d.photo.getCurrentPhoto() && d.photo.getCurrentPhoto().get("is_owner")) {
                if (Q) {
                    Q.remove_existing_dialog()
                }
                aq = d.one("#global-dialog-background");
                if (aq) {
                    aq.setStyle("display", "none")
                }
                p();
                return 
            }
            if (d.PhotoData) {
                at = d.PhotoData.get(ap);
                if (at) {
                    if (!at.get("is_fave")) {
                        at.toggle_fave()
                    }
                    J();
                    p()
                } else {
                    if (typeof d.FlickrAppExplore !== "undefined") {
                        au = d.FlickrAppExplore.getListModel();
                        if (au) {
                            au.toggleFave(ap);
                            ar = false
                        }
                    }
                    if (typeof d.FlickrAppPhotostream !== "undefined") {
                        au = d.FlickrAppPhotostream.getListModel();
                        if (au) {
                            au.toggleFave(ap);
                            neeedsAPI = false
                        }
                    }
                    if (ar) {
                        d.flickrAPI.callMethod("flickr.favorites.add", {
                            photo_id: ap
                        })
                    }
                    J();
                    p()
                }
            }
        }
        function I(Y) {
            A();
            p();
            d.flickrAPI.callMethod("flickr.sharing.share", Y, {
                success: function(ap) {
                    m();
                    J()
                },
                failure: r
            })
        }
        function aj(Y) {
            A();
            p();
            Y.magic_cookie = d.config.flickr.magic_cookie;
            Y.from_grease = 1;
            d.use("share-this-v3-dialog", function(ap) {
                ap.shareThisV3Dialog.do_share_action(Y)
            })
        }
        function r() {
            p();
            Q.create_dialog(d.one("#grease-error").get("innerHTML")).show({
                loading: false,
                modal: true,
                width: 355,
                style: "grease-error"
            });
            d.one("#global_dialog").on("click", f, this);
            d.later(5000, this, d.global_dialog.hide)
        }
        function o(Y) {
            A();
            contactArr = Y.split(";");
            d.flickrAPI.callMethod("flickr.contacts.add", {
                user_id: contactArr[0],
                friend: contactArr[1],
                family: contactArr[2]
            }, {
                success: function(ap) {
                    m();
                    J();
                    p()
                },
                failure: r
            })
        }
        function n() {
            if (document.cookie.match(e)) {
                S = false;
                A();
                ah.start()
            } else {
                d.later(20, this, n)
            }
        }
        function ab(aq) {
            A();
            p();
            var ap = d.one("#message");
            if (!ap) {
                r();
                return 
            }
            d.one("#message").set("innerHTML", aq);
            var Y = {
                method: "post",
                data: "preview=0",
                form: {
                    id: d.one(".add-comment-form form")
                },
                on: {
                    success: function(aw, av, at) {
                        m();
                        J();
                        var ar = d.one("#comments");
                        ar.set("innerHTML", av.responseText);
                        var au = ar.one(".Problem");
                        if (au) {
                            Q.create_alert_dialog(au.get("innerHTML"), function() {
                                ar.one("#message").scrollIntoView()
                            })
                        }
                    }
                }
            };
            d.io("/photo_comments_fragment.gne", Y)
        }
        function C(ar, aq) {
            p();
            var Y = "magic_cookie=" + d.config.flickr.magic_cookie + "&photo=" + aq + "&addcomment=1flash_ver=unknown&offset=" + ( - 1) + "&limit=4&show-chrome=true&view-more-count=20&hide-meta=true&message=" + ar;
            var ap = {
                method: "post",
                data: Y,
                on: {
                    success: function(ay, ax, au) {
                        var at = d.one("div[data-photo-id=" + aq + "]");
                        if (at) {
                            var av = at.getAttribute("data-comments-total-count");
                            av = av === "" ? 0 : parseInt(av, 10);
                            av++;
                            at.setAttribute("data-comments-total-count", av);
                            var aw = at.one(".comment-count");
                            if (aw) {
                                if (av < 100) {
                                    aw.setHTML(av)
                                } else {
                                    aw.setHTML("99+")
                                }
                            }
                        }
                        m()
                    },
                    failure: function(at, au) {
                        m();
                        showError(d.transjax.get("photo-comments", "error_add_generic"))
                    }
                }
            };
            d.io("/photo_comments_fragment_2013.gne?cachebust=" + (new Date()).getTime(), ap)
        }
        function B(ar) {
            var aq;
            var aw = 650;
            var at;
            if (d.config.flickr.flags.enable_ads_on_login_page) {
                at = d.one(window).get("winWidth") || 0;
                if (!isNaN(aw)) {
                    aw = Math.max(880, Math.min(1200, at))
                }
            }
            ah.reset();
            var ap = ar + "&popup=1&redir=" + aa + "?d=" + encodeURIComponent(window.location + "?reg=1") + "&src=" + z;
            if (d.config.flickr.is_touch_device) {
                aq = aa + "?d=" + window.location + "&notpopup=1&src=" + z + "&reg=1";
                window.location = ar + "&popup=0&redir=" + encodeURIComponent(aq);
                return false
            }
            try {
                al = window.open(ap, "newWindow", "width=" + aw + ",height=650,resizable=1,scrollbars=1,location=yes")
            } catch (av) {
                return true
            }
            try {
                if (al.focus) {
                    al.focus()
                }
            } catch (Y) {}
            if (!al || al.closed || typeof al.closed === "undefined") {
                aq = aa + "?d=" + window.location + "&notpopup=1&src=" + z + "&reg=1";
                window.location = ar + "&popup=0&redir=" + encodeURIComponent(aq)
            } else {
                window.setTimeout(n, 500)
            }
            var au = d.one("window");
            au.on("grease|focus", function() {
                if (!ah.canceled) {
                    ah.start()
                }
                if (!document.cookie.match(e)) {
                    m();
                    W(false);
                    n()
                }
            });
            d.on("grease:cancel", function(ax) {
                S = false;
                c()
            });
            au.on("grease|blur", function() {
                if (!ah.canceled) {
                    ah.pause()
                }
            });
            au.on("grease:timeout", function(ax) {
                S = false;
                c()
            });
            d.on("grease:authenticated", function(aB) {
                if (al) {
                    try {
                        al.close()
                    } catch (az) {}
                }
                A();
                S = false;
                W(true);
                var aC = (Z), ax = ["reg=1", "src=" + z];
                if (aC) {
                    var aA = window.location.toString(), ay = aA.match(/\?/i);
                    aA += (!ay ? "?" : "&") + ax.join("&");
                    if (aA.indexOf("#?")!==-1) {
                        aA = aA.replace("#", "")
                    }
                    window.location = aA
                } else {
                    window.location = d.urls.get("photo", {
                        photo: d.photo.getCurrentPhoto()
                    }) + "?" + ax.join("&")
                }
            });
            return false
        }
        function b(aq, ar, ap) {
            var Y = ap || false;
            P.setItem(aq, ar, Y)
        }
        function ag(Y, aq) {
            var ap = aq || false;
            return P.getItem(Y, ap)
        }
        function H(Y) {
            if (!S) {
                ah.reset();
                W = Y || function() {};
                d.global_dialog.remove_existing_dialog();
                d.global_dialog.show({
                    loading: true,
                    modal: true
                });
                S = true;
                B("/signin?src=" + z + "&reg=1")
            } else {
                c();
                H(Y)
            }
        }
        function M(ap, Y) {
            Y = Y || d.photo.getCurrentPhoto().get("id");
            p();
            b(Y ? y : E, Y + l + ap);
            z = "comment"
        }
        function w(Y) {
            p();
            b(u, Y);
            z = "fave"
        }
        function ac(Y) {
            p();
            if (d.Lang.isArray(Y)) {
                b(g, Y.join(";"));
                z = "contact"
            }
        }
        function U(Y) {
            p();
            b(R, Y, true);
            z = "share"
        }
        function ai(Y) {
            p();
            b(Z, Y, true);
            z = "sharev3"
        }
        function v(Y) {
            p();
            b(j, Y);
            z = "party"
        }
        function ao(Y) {
            p();
            b(q, d.JSON.stringify(Y));
            z = "partyStart"
        }
        d.grease = {
            init: af,
            enabled: X,
            authenticate: H,
            postComment: M,
            addFave: w,
            addContact: ac,
            share: U,
            shareV3: ai,
            photoParty: v,
            photoPartyStart: ao
        };
        if (d.config.flickr.user.admin_user === "saw") {
            d.occult.register("postComment", function(Y) {
                ab(Y)
            });
            d.occult.register("showdialog", function(Y) {
                z = "fave";
                J()
            });
            d.occult.register("fakeerror", function(Y) {
                r(Y)
            })
        }
        var i = d.one("#reg-tour");
        function k() {
            for (var Y = 1; Y < 4; Y++) {
                am.removeClass(K + Y)
            }
        }
        function N(Y) {
            Y.preventDefault();
            k();
            d.Event.purgeElement("#reg-tour");
            i.addClass("hidden")
        }
        function D(Y) {
            if (Y.test("#reg-tour")) {
                return 
            }
            if (Y.test("ol li")) {
                k();
                switch (Y.get("id")) {
                case"step-1":
                    am.addClass(K + 1);
                    break;
                case"step-2":
                    am.addClass(K + 2);
                    break;
                case"step-3":
                    am.addClass(K + 3);
                    break
                }
            } else {
                D(Y.get("parentNode"))
            }
        }
        if (i) {
            var K = "tour-step", am = d.one("body");
            i.one(".close-x").on("click", N);
            i.on("mouseover", function(Y) {
                D(Y.target)
            });
            i.on("mouseout", function(Y) {
                k()
            })
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
YUI.add("photo-data", function(g) {
    var b = {}, a = g.TypeCast;
    function f(i) {
        var h;
        if (i instanceof d) {
            h = i
        } else {
            if (b[i.id]) {
                h = b[i.id];
                h.merge_data(i)
            } else {
                h = new d(i);
                b[i.id] = h
            }
        }
        return h
    }
    function c(h) {
        if (typeof h === "undefined") {
            return b
        } else {
            return b[h]
        }
    }
    function e(h) {
        return g.Lang.isNumber(h) && parseInt(h, 10) === h
    }
    g.PhotoData = {
        add: f,
        get: c
    };
    function d(h) {
        d.superclass.constructor.apply(this, arguments)
    }
    d.NAME = "PhotoData";
    d.ATTRS = {
        id: {
            value: ""
        },
        title: {
            broadcast: 1,
            value: "",
            setter: function(i, h) {
                i = a.string(i);
                return i.replace(/^\s+$/, "")
            }
        },
        description: {
            broadcast: 1,
            value: "",
            setter: a.string
        },
        owner: {
            value: "",
            setter: function(h) {
                if (!g.Lang.isString(h)) {
                    if (g.Lang.isObject(h) && h.nsid) {
                        h = h.nsid
                    } else {
                        h = ""
                    }
                }
                return h
            }
        },
        ownername: {
            value: ""
        },
        pathalias: {
            value: ""
        },
        iconserver: {
            value: ""
        },
        iconfarm: {
            value: ""
        },
        url: {
            value: "",
            getter: function(i, h) {
                return g.urls.get("photo", {
                    photo: this
                })
            }
        },
        date_posted: {
            broadcast: 1,
            getter: a.integer
        },
        is_public: {
            broadcast: 1,
            getter: a.bool
        },
        is_friend: {
            broadcast: 1,
            getter: a.bool
        },
        is_family: {
            broadcast: 1,
            getter: a.bool
        },
        perm_comment: {
            broadcast: 1,
            getter: a.integer0123
        },
        perm_addmeta: {
            broadcast: 1,
            getter: a.integer0123
        },
        perms: {
            getter: function() {
                return {
                    is_public: this.get("is_public"),
                    is_friend: this.get("is_friend"),
                    is_family: this.get("is_family"),
                    perm_comment: this.get("perm_comment"),
                    perm_addmeta: this.get("perm_addmeta")
                }
            }
        },
        license: {
            broadcast: 1,
            getter: a.integer,
            setter: a.integer
        },
        safety_level: {
            broadcast: 1,
            getter: a.integer,
            setter: a.integer
        },
        needs_interstitial: {
            broadcast: 1,
            value: false,
            getter: a.bool,
            setter: a.bool
        },
        media: {
            broadcast: 1,
            value: "photo"
        },
        is_video: {
            getter: function() {
                return (this.get("media") === "video")
            }
        },
        secret: {
            value: ""
        },
        deleted: {
            value: false
        },
        dmca_takedown: {
            value: false
        },
        geo: {
            broadcast: 1,
            getter: function(h) {
                if (!g.Lang.isObject(h)) {
                    h = {}
                }
                return h
            }
        },
        has_geo: {
            getter: function() {
                var h = this.get("geo");
                return (h && h.latitude && h.longitude && h.accuracy)
            }
        },
        exif_latlong: {},
        perm_viewgeo: {
            broadcast: 1,
            value: false
        },
        is_bad: {
            broadcast: 1,
            getter: a.integer,
            setter: a.integer
        },
        content_type: {
            broadcast: 1,
            getter: a.integer,
            setter: a.integer
        },
        is_fave: {
            broadcast: 1,
            value: false,
            setter: function(j, h) {
                j = a.bool(j);
                var i = this.get(h);
                if (j !== i) {
                    if (j) {
                        this.increment("fave_count")
                    } else {
                        this.decrement("fave_count")
                    }
                }
                if (this.get("fave_changes_pending") === 0) {
                    this.set("is_confirmed_fave", j)
                }
                return j
            },
            getter: a.bool
        },
        is_confirmed_fave: {
            value: undefined
        },
        fave_changes_pending: {
            value: 0
        },
        fave_count: {
            broadcast: 1,
            value: 0,
            getter: a.integer,
            setter: a.integer
        },
        comment_count: {
            broadcast: 1,
            value: 0,
            getter: a.integer,
            setter: a.integer
        },
        can_addmeta: {
            value: false,
            getter: a.bool
        },
        can_comment: {
            value: false,
            getter: a.bool
        },
        can_fave: {
            value: false,
            getter: function() {
                if (this.get("is_ignored")) {
                    return false
                }
                if (this.get("dmca_takedown")) {
                    return false
                }
                if (this.get("is_owner")) {
                    return false
                }
                if (!g.config.flickr.user.nsid&&!g.config.flickr.flags.enable_grease) {
                    return false
                }
                return true
            }
        },
        group_count: {
            broadcast: 1,
            getter: a.integer,
            setter: a.integer
        },
        people_count: {
            broadcast: 1,
            getter: a.integer,
            setter: a.integer
        },
        guestpass_count: {
            broadcast: 1,
            getter: a.integer,
            setter: a.integer
        },
        is_owner: {
            value: false,
            getter: function() {
                return g.config.flickr.user.nsid && g.config.flickr.user.nsid === this.get("owner")
            }
        },
        is_ignored: {
            value: false
        },
        tags: {
            broadcast: 1,
            getter: a.arr
        },
        sizes: {
            value: {},
            getter: function(m, i) {
                var j, l, h, k;
                m = a.obj(m);
                if (g.config.flickr.flags.enable_2048_images) {
                    j = m.o || m.l || m.z;
                    if (j && (j.width >= 2048 || j.height >= 2048)) {
                        l = j.width / j.height;
                        if (l > 1) {
                            k = 2048;
                            h = Math.round(2048 / l)
                        } else {
                            h = 2048;
                            k = Math.round(2048 * l)
                        }
                        m["2048"] = {
                            url: m.t.url.replace("_t", "_2048"),
                            height: h + "",
                            width: k + "",
                            label: "2048"
                        }
                    }
                }
                return m
            }
        }
    };
    g.extend(d, g.Base, {
        initializer: function(h) {
            this.merge_data(h)
        },
        merge_data: function(i) {
            var j, h;
            if (typeof i.id !== "undefined" && i.id !== this.get("id")) {
                this.set("id", i.id)
            }
            if (typeof i.title !== "undefined" && i.title !== this.get("title")) {
                this.set("title", i.title)
            }
            if (typeof i.description !== "undefined" && i.description !== this.get("description")) {
                this.set("description", i.description)
            }
            if (typeof i.owner !== "undefined" && i.owner !== this.get("owner")) {
                this.set("owner", i.owner)
            }
            if (typeof i.ownername !== "undefined" && i.ownername !== this.get("ownername")) {
                this.set("ownername", i.ownername)
            }
            if (typeof i.pathalias !== "undefined" && i.pathalias !== this.get("pathalias")) {
                this.set("pathalias", i.pathalias)
            }
            if (typeof i.iconserver !== "undefined" && i.iconserver !== this.get("iconserver")) {
                this.set("iconserver", i.iconserver)
            }
            if (typeof i.iconfarm !== "undefined" && i.iconfarm !== this.get("iconfarm")) {
                this.set("iconfarm", i.iconfarm)
            }
            if (typeof i.url !== "undefined" && i.url !== this.get("url")) {
                this.set("url", i.url)
            }
            if (typeof i.date_posted !== "undefined" && i.date_posted !== this.get("date_posted")) {
                this.set("date_posted", i.date_posted)
            }
            if (typeof i.date_taken !== "undefined" && i.date_taken !== this.get("date_taken")) {
                this.set("date_taken", i.date_taken)
            }
            if (typeof i.is_public !== "undefined" && i.is_public !== this.get("is_public")) {
                this.set("is_public", i.is_public)
            }
            if (typeof i.is_friend !== "undefined" && i.is_friend !== this.get("is_friend")) {
                this.set("is_friend", i.is_friend)
            }
            if (typeof i.is_family !== "undefined" && i.is_family !== this.get("is_family")) {
                this.set("is_family", i.is_family)
            }
            if (typeof i.can_comment !== "undefined" && i.can_comment !== this.get("can_comment")) {
                this.set("can_comment", i.can_comment)
            }
            if (typeof i.can_addmeta !== "undefined" && i.can_addmeta !== this.get("can_addmeta")) {
                this.set("can_addmeta", i.can_addmeta)
            }
            if (typeof i.perm_comment !== "undefined" && i.perm_comment !== this.get("perm_comment")) {
                this.set("perm_comment", i.perm_comment)
            }
            if (typeof i.perm_addmeta !== "undefined" && i.perm_addmeta !== this.get("perm_addmeta")) {
                this.set("perm_addmeta", i.perm_addmeta)
            }
            if (typeof i.license !== "undefined" && i.license !== this.get("license")) {
                this.set("license", i.license)
            }
            if (typeof i.needs_interstitial !== "undefined" && i.needs_interstitial !== this.get("needs_interstitial")) {
                this.set("needs_interstitial", i.needs_interstitial)
            }
            if (typeof i.safety_level !== "undefined" && i.safety_level !== this.get("safety_level")) {
                this.set("safety_level", i.safety_level)
            }
            if (typeof i.media !== "undefined" && i.media !== this.get("media")) {
                this.set("media", i.media)
            }
            if (typeof i.secret !== "undefined" && i.secret !== this.get("secret")) {
                this.set("secret", i.secret)
            }
            if (typeof i.is_bad !== "undefined" && i.is_bad !== this.get("is_bad")) {
                this.set("is_bad", i.is_bad)
            }
            if (typeof i.content_type !== "undefined" && i.content_type !== this.get("content_type")) {
                this.set("content_type", i.content_type)
            }
            if (typeof i.is_faved !== "undefined" && i.is_faved !== this.get("is_fave")) {
                this.set("is_fave", i.is_faved)
            }
            if (typeof i.is_fave !== "undefined" && i.is_fave !== this.get("is_fave")) {
                this.set("is_fave", i.is_fave)
            }
            if (typeof i.fave_count !== "undefined" && i.fave_count !== this.get("fave_count")) {
                this.set("fave_count", i.fave_count)
            }
            if (typeof i.count_faves !== "undefined" && i.count_faves !== this.get("fave_count")) {
                this.set("fave_count", i.count_faves)
            }
            if (typeof i.comment_count !== "undefined" && i.comment_count !== this.get("comment_count")) {
                this.set("comment_count", i.comment_count)
            }
            if (typeof i.count_comments !== "undefined" && i.count_comments !== this.get("comment_count")) {
                this.set("comment_count", i.count_comments)
            }
            if (typeof i.geo !== "undefined" && i.geo !== this.get("geo")) {
                this.set("geo", i.geo)
            }
            if (typeof i.exif_latlong !== "undefined" && i.exif_latlong !== this.get("exif_latlong")) {
                this.set("exif_latlong", i.exif_latlong)
            }
            if (typeof i.perm_viewgeo !== "undefined" && i.perm_viewgeo !== this.get("perm_viewgeo")) {
                this.set("perm_viewgeo", i.perm_viewgeo)
            }
            if (typeof i.woe !== "undefined" && i.woe !== this.get("woe")) {
                this.set("woe", i.woe)
            }
            if (typeof i.tags !== "undefined" && i.tags !== this.get("tags")) {
                this.set("tags", i.tags)
            }
            if (typeof i.group_count !== "undefined" && i.group_count !== this.get("group_count")) {
                this.set("group_count", i.group_count)
            }
            if (typeof i.people_count !== "undefined" && i.people_count !== this.get("people_count")) {
                this.set("people_count", i.people_count)
            }
            if (typeof i.guestpass_count !== "undefined" && i.guestpass_count !== this.get("guestpass_count")) {
                this.set("guestpass_count", i.guestpass_count)
            }
            if (i.sizes && g.Lang.isObject(i.sizes)) {
                j = i.sizes
            } else {
                j = {};
                if (i.url_sq) {
                    j.sq = {
                        url: i.url_sq,
                        width: parseInt(i.width_sq, 10),
                        height: parseInt(i.height_sq, 10)
                    }
                }
                if (i.url_t) {
                    j.t = {
                        url: i.url_t,
                        width: parseInt(i.width_t, 10),
                        height: parseInt(i.height_t, 10)
                    }
                }
                if (i.url_s) {
                    j.s = {
                        url: i.url_s,
                        width: parseInt(i.width_s, 10),
                        height: parseInt(i.height_s, 10)
                    }
                }
                if (i.url_m) {
                    j.m = {
                        url: i.url_m,
                        width: parseInt(i.width_m, 10),
                        height: parseInt(i.height_m, 10)
                    }
                }
                if (i.url_z) {
                    j.z = {
                        url: i.url_z,
                        width: parseInt(i.width_z, 10),
                        height: parseInt(i.height_z, 10)
                    }
                }
                if (i.url_l) {
                    j.l = {
                        url: i.url_l,
                        width: parseInt(i.width_l, 10),
                        height: parseInt(i.height_l, 10)
                    }
                }
                if (i.url_h) {
                    j.h = {
                        url: i.url_h,
                        width: parseInt(i.width_h, 10),
                        height: parseInt(i.height_h, 10)
                    }
                }
                if (i.url_k) {
                    j.k = {
                        url: i.url_k,
                        width: parseInt(i.width_k, 10),
                        height: parseInt(i.height_k, 10)
                    }
                }
                if (i.url_o) {
                    j.o = {
                        url: i.url_o,
                        width: parseInt(i.width_o, 10),
                        height: parseInt(i.height_o, 10)
                    }
                }
                if (i.media === "video") {
                    if (i.height_v) {
                        j.v = {
                            width: parseInt(i.width_v, 10),
                            height: parseInt(i.height_v, 10)
                        }
                    }
                }
            }
            if (g.Object.size(j)) {
                h = this.get("sizes") || {};
                g.Object.each(h, function(k, l) {
                    if (j[l]) {
                        j[l] = g.merge(k, j[l])
                    }
                });
                this.set("sizes", g.merge(h, j))
            }
        },
        toggle_fave: function(j) {
            var i = this, h, m, k, l;
            if (!this.get("can_fave")) {
                return 
            }
            if (g.config.flickr.flags.enable_grease&&!g.config.flickr.user.nsid) {
                g.grease.addFave(this.get("id"));
                g.grease.authenticate();
                return 
            }
            h=!this.get("is_fave");
            m = "flickr.favorites." + (h ? "add" : "remove");
            k = {
                photo_id: this.get("id")
            };
            l = {
                success: function() {
                    var n, o, p, q;
                    n = i.get("id");
                    i.decrement("fave_changes_pending");
                    i.set("is_confirmed_fave", h);
                    if (i.get("fave_changes_pending") === 0) {
                        i.set("is_fave", h)
                    }
                    o = "faves-" + (g.config.flickr.user.pathalias ? g.config.flickr.user.pathalias : g.config.flickr.user.nsid);
                    p = g.ContextData.getContext(o);
                    if (p) {
                        q = p.getPhotoPosition(n)
                    }
                    if (h) {
                        if (p) {
                            if (g.ContextData.getPrevLimit(o) === q) {
                                g.ContextData.unremovePhoto(o, q)
                            } else {
                                g.ContextData.invalidate(o)
                            }
                        } else {
                            if (g.config.flickr.page_type === "photo" && i.get("id") === g.photo.getCurrentPhoto().get("id")) {
                                g.ContextData.add({
                                    id: o,
                                    user_id: g.config.flickr.user.nsid,
                                    type: "faves",
                                    title: g.transjax.get("photo", "your_favorites"),
                                    url: g.config.flickr.user.photos_url + "favorites/",
                                    photos: [{
                                        position: "0",
                                        photo: i
                                    }
                                    ]
                                })
                            }
                        }
                    } else {
                        if (p && g.Lang.isNumber(q)) {
                            g.ContextData.removePhoto(o, q)
                        }
                    }
                },
                failure: function() {
                    i.decrement("fave_changes_pending");
                    if (i.get("fave_changes_pending") === 0) {
                        i.set("is_fave", i.get("is_confirmed_fave"))
                    }
                }
            };
            this.set("is_fave", h);
            this.increment("fave_changes_pending");
            g.flickrAPI.callMethod(m, k, l)
        },
        record_interstitial_clickthrough: function() {
            var h = this.get("owner");
            g.Object.each(b, function(i, j) {
                if (i.get("owner") === h) {
                    i.set("needs_interstitial", false)
                }
            });
            g.flickrAPI.callMethod("flickr.people.recordInterstitialClickthrough", {
                user_id: h
            }, function() {})
        },
        updateWoeAttributeByeId: function(i) {
            var h = this;
            g.use("flickr-geo", function(j) {
                j.flickrGeo.getWoeById(i, function(k) {
                    if (k.success && j.Lang.isObject(k.woe)) {
                        h.set("woe", k.woe)
                    } else {
                        j.log("Unable to update WOE. Expected data was not received from the API.", "error", "photo-data")
                    }
                }, this)
            })
        },
        increment: function(h) {
            if (!e(this.get(h))) {
                return false
            }
            return this.add(h, 1)
        },
        decrement: function(h) {
            if (!e(this.get(h))) {
                return false
            }
            return this.add(h, - 1)
        },
        add: function(i, j) {
            var k = this.get(i), h;
            if (!g.Lang.isNumber(k) ||!g.Lang.isNumber(j)) {
                return false
            }
            h = k + j;
            if (this.set(i, h)) {
                return h
            } else {
                return false
            }
        }
    })
}, "0.0.1", {
    requires: F.config.modules["photo-data"].requires || [],
    optional: F.config.modules["photo-data"].optional || []
});
(function() {
    var a = "refresh-sihp-comment";
    YUI.add(a, function(b) {
        var p, e;
        var n = {};
        var c = function(q) {
            o(q)
        };
        b.config[a] = b.config[a] || {
            open_cards: {}
        };
        function o(r) {
            var q, s;
            if (!b.Lang.isObject(r)) {
                console.warn("[" + a + "]", "No config");
                return 
            }
            q = r.container;
            if (b.Lang.isString(q)) {
                q = b.one(q)
            }
            if (!b.Lang.isObject(q) ||!q.getDOMNode) {
                console.warn("[" + a + "]", "No container node");
                return 
            }
            p = r;
            q.delegate("click", h, r.commentIconSelector);
            q.delegate("click", g, r.commentReplySelector || ".comment-reply");
            m(q);
            e = "refresh-sihp-comment:dismiss-dialog";
            b.on(e, f)
        }
        function h(w) {
            if (w.preventDefault) {
                w.preventDefault()
            }
            if (b.shareThisV3Menu.isInitialized()) {
                b.shareThisV3Menu.hide()
            }
            var y = w.target.ancestor(p.photoContainerSelector), A = p.commentContainerSelector ? y.ancestor(p.commentContainerSelector): y, B = p.photoNodeSelector ? y.one(p.photoNodeSelector): y, x = w.target.ancestor(p.commentButtonNodeSelector, true), s = B.getAttribute("data-photo-id"), u = b.one(p.container).getComputedStyle("width"), z = B.get("offsetHeight"), v = (z + 15) + "px", q = p.arrowOffset || 39;
            if (w.target.ancestor(".leftPortrait")) {
                q += 267
            }
            if (b.one(".comments-popover-" + s)) {
                f();
                return 
            } else {
                if (b.one(".comments-popover")) {
                    f()
                }
            }
            if (b.all(".comments-popover")) {
                b.all(".comments-popover").remove()
            }
            if (b.all(p.commentIconSelector)) {
                b.all(p.commentIconSelector).removeClass("active")
            }
            totalComments = parseInt(B.getAttribute("data-comments-total-count"), 10);
            if (x.one(".li_icon")) {
                x.one(".li_icon").remove()
            } else {
                if (p.onLoading) {
                    p.onLoading(x)
                }
            }
            x.append('<span class="html5-balls"><span class="blueball"></span><span class="pinkball"></span></span>');
            var r = "magic_cookie=" + b.config.flickr.magic_cookie + "&photo=" + s + "&offset=" + (totalComments - 4) + "&limit=4&show-chrome=true&view-more-count=20&truncate-comment-body=120&hide-meta=true";
            var t = {
                method: "post",
                data: r,
                on: {
                    success: function(C, D, K) {
                        b.config[a].open_cards[parseInt(s, 10)] = true;
                        var N, J;
                        A.append('<div id="comments" class="comments-popover comments-popover-' + s + '"><div class="arrow"></div><div class="comments-inner">' + D.responseText + "</div></div>");
                        if (p.edgeDetection) {
                            if (x.getX() < 570) {
                                A.one("#comments").setStyle("right", "");
                                A.one("#comments").setStyle("left", "10px");
                                var L = A.getComputedStyle("width");
                                L = L.replace("px", "");
                                L = parseInt(L, 10);
                                q = L - 75 + 10
                            }
                            if (x.getY() < 570) {
                                var H = A.getComputedStyle("height");
                                H = H.replace("px", "");
                                H = parseInt(H, 10);
                                A.one("#comments").setStyle("top", (H - 3) + "px");
                                A.one("#comments").setStyle("bottom", "inherit");
                                var M = A.one(".arrow");
                                var E = M.getStyle("bottom");
                                E = E.replace("px", "");
                                E = parseInt(E, 10);
                                M.setStyle("top", E);
                                M.setStyle("bottom", "inherit");
                                M.setStyle("borderBottomColor", M.getStyle("borderTopColor"));
                                M.setStyle("borderTopColor", "transparent")
                            }
                        } else {
                            if (y.hasClass("topRightLandscapeContainer")) {
                                var H = A.getComputedStyle("height");
                                if (H) {
                                    H = parseInt(H.replace("px", ""), 10);
                                    A.one("#comments").setStyle("bottom", ((H / 2) + 37) + "px")
                                }
                            } else {
                                if (y.hasClass("leftPortraitContainer")) {
                                    A.one("#comments").setStyle("right", "");
                                    A.one("#comments").setStyle("left", "10px");
                                    var L = A.getComputedStyle("width");
                                    L = L.replace("px", "");
                                    L = parseInt((L / 2) + "px", 10);
                                    q = L - 75
                                }
                            }
                        }
                        x.one(".html5-balls").remove();
                        x.addClass("active");
                        if (p.onSuccess) {
                            p.onSuccess(y, x, totalComments)
                        } else {
                            x.append('<span class="li_icon"></span>')
                        }
                        N = A.one(".comment-button-post");
                        if (N) {
                            N.on("click", function(O) {
                                j(O, s)
                            })
                        }
                        if (A && A.one("form #message")) {
                            A.one("form #message").on("keyup", function(O) {
                                n[s] = O.target.get("value")
                            });
                            if (n[s]) {
                                A.one("form #message").set("value", n[s])
                            }
                        }
                        if (x.hasClass("counter1")) {
                            q += 5
                        } else {
                            if (x.hasClass("counter2")) {
                                q += 9
                            } else {
                                if (x.hasClass("counter3")) {
                                    q += 13
                                }
                            }
                        }
                        J = A.one(".arrow");
                        if (J) {
                            if ((p.edgeDetection && x.getX() < 570) || y.hasClass("leftPortraitContainer")) {
                                J.setStyle("left", q + "px")
                            } else {
                                J.setStyle("right", q + "px")
                            }
                        }
                        if (b.one("#load-more-next-comments")) {
                            b.one("#load-more-next-comments").hide()
                        }
                        var G = window.innerWidth + 200;
                        y.append("<div class='popover-shim' style='position: absolute; top: 0; left: -500px; height: 1000px; width: " + G + "px; z-index: 1;'></div>");
                        if (b.one(".popover-shim")) {
                            b.one(".popover-shim").on("click", f)
                        }
                        b.keyboardShortcutManager.register({
                            keystring: "27",
                            handler: f,
                            context: "*"
                        });
                        try {
                            if (p.edgeDetection && x.getY() < 570) {
                                A.one(".comment-content textarea").vis.animScrollIntoView({
                                    bottom: true,
                                    top: false,
                                    onfinish: function() {
                                        A.one(".comment-content textarea").focus()
                                    }
                                }, 100, true)
                            } else {
                                A.one(".comment-content textarea").focus()
                            }
                        } catch (I) {}
                    },
                    failure: function(E, D, C) {
                        x.one(".html5-balls").remove();
                        if (p.onSuccess) {
                            p.onSuccess(y, x, totalComments)
                        } else {
                            x.append('<span class="li_icon"></span>')
                        }
                        i(b.transjax.get("photo-comments", "error_getting_comments"))
                    }
                }
            };
            b.io("/photo_comments_fragment_2013.gne?cachebust=" + (new Date()).getTime(), t)
        }
        function j(A, u) {
            A.preventDefault();
            var v = A.target.ancestor(p.photoContainerSelector) || A.target.ancestor(p.commentContainerSelector), D = v.one(p.photoNodeSelector + "[data-photo-id=" + u + "]") ? v.one(p.photoNodeSelector + "[data-photo-id=" + u + "]"): p.commentContainerSelector ? A.target.ancestor(p.commentContainerSelector): v, C = (v.ancestor(p.commentContainerSelector) || v), x = C.one(".comments-popover, .comments-overlay"), B = D.one(p.commentButtonNodeSelector), r = (C.one(".comments-popover")) ? true: false, z, q = D.getAttribute("data-comments-total-count"), t = parseInt(q === "" ? 0 : q, 10);
            if (b.config.flickr.flags.enable_grease&&!b.config.flickr.user.nsid && b.grease) {
                var y = encodeURIComponent(x.one("form #message").get("value"));
                b.grease.postComment(y, u);
                b.grease.authenticate(function(E) {});
                return false
            }
            var s = "magic_cookie=" + b.config.flickr.magic_cookie + "&photo=" + u + "&addcomment=1flash_ver=unknown&offset=" + (t - 3) + "&limit=4&show-chrome=true&view-more-count=20&truncate-comment-body=" + d(2) + "&hide-meta=true&message=" + encodeURIComponent(x.one("form #message").get("value"));
            var w = {
                method: "post",
                data: s,
                on: {
                    success: function(M, L, H) {
                        var E;
                        var I = b.Node.create(L.responseText).one(".Problem");
                        if (I) {
                            if (I.get("text") === "An empty comment box? That won't work!") {
                                i(b.transjax.get("photo-comments", "error_empty_body"))
                            } else {
                                i(b.transjax.get("photo-comments", "error_add_generic"))
                            }
                            x.one(".comment-content .html5-balls").remove();
                            x.one(".comment-content form").show();
                            return 
                        }
                        n = {};
                        if (r) {
                            var G = t, K = G + 1, J;
                            z = D.get("parentNode");
                            B.setAttribute("class", "");
                            if (K < 10) {
                                B.addClass("counter1");
                                J = "counter1"
                            } else {
                                if (K < 100) {
                                    B.addClass("counter2");
                                    J = "counter2"
                                } else {
                                    B.addClass("counter3");
                                    J = "counter3";
                                    B.one(p.counterSelector || ".li_counter").hide();
                                    if (B.one(".li_99_plus")) {
                                        B.one(".li_99_plus").show()
                                    }
                                }
                            }
                            B.addClass(p.commentButtonNodeClass || "il_comments");
                            B.addClass("rapidnofollow");
                            B.removeClass("active");
                            B.one(p.counterSelector || ".li_counter").set("text", K);
                            if (z.get("parentNode").hasClass("card-bundle")) {
                                E = D.getAttribute("data-photo-id");
                                z.one(".id" + E).setAttribute("data-comment_class", J).setAttribute("data-comment_count", K)
                            }
                            f()
                        } else {
                            x.one(".comments-inner").set("innerHTML", L.responseText);
                            if (x.one(".comment-button-post")) {
                                x.one(".comment-button-post").on("click", function(N) {
                                    j(N, u)
                                })
                            }
                        }
                        D.setAttribute("data-comments-total-count", t + 1)
                    },
                    failure: function(E, G) {
                        x.one(".comment-content .html5-balls").remove();
                        x.one(".comment-content form").show();
                        x.one(".comment-content textarea").focus();
                        i(b.transjax.get("photo-comments", "error_add_generic"))
                    }
                }
            };
            x.one(".comment-content form").hide();
            x.one(".add-comment-form .comment-content").append('<span class="html5-balls"><span class="blueball"></span><span class="pinkball"></span></span>');
            b.io("/photo_comments_fragment_2013.gne?cachebust=" + (new Date()).getTime(), w)
        }
        function m(q) {
            q.all(".card-comment").each(function(w) {
                if (!w.one(".comments-overlay")) {
                    var v = w, r = v.getAttribute("data-photo-id"), u = v.one(".commentsWrapper");
                    totalComments = parseInt(v.getAttribute("data-comments-total-count"));
                    var s = "magic_cookie=" + b.config.flickr.magic_cookie + "&photo=" + r + "&offset=" + (totalComments - 4) + "&limit=4&show-chrome=true&view-more-count=20&truncate-comment-body=" + d(2) + "&hide-meta=true";
                    var t = {
                        method: "post",
                        data: s,
                        on: {
                            success: function(A, z, y) {
                                if (u.one(".comments-overlay")) {
                                    return 
                                }
                                var x;
                                if (u.one(".action-view-comments")) {
                                    u.one(".action-view-comments").remove()
                                }
                                u.append('<div id="comments" class="comments-overlay"><div class="comments-inner">' + z.responseText + "</div></div>");
                                x = u.one(".comment-button-post");
                                if (x) {
                                    x.on("click", function(B) {
                                        j(B, r)
                                    })
                                }
                                b.refreshSIHPSidebar.updateConfig()
                            },
                            failure: function(z, y, x) {}
                        }
                    };
                    b.io("/photo_comments_fragment_2013.gne?cachebust=" + (new Date()).getTime(), t)
                }
            })
        }
        function d(q) {
            var r = b.one(p.container).get("offsetWidth");
            if (r <= 640) {
                return 50 * q
            } else {
                return 70 * q
            }
        }
        function k(r) {
            r.preventDefault();
            var q = r.target.ancestor("li.comment-block");
            q.one(".truncated-comment").hide();
            q.one(".whole-comment").show()
        }
        function g(s) {
            s.preventDefault();
            var q, r;
            q = s.target.ancestor(".comment-block").one(".comment-buddy-icon-link");
            if (q) {
                r = "[http://www.flickr.com/photos/" + q.get("href").replace(/^.*?\/photos\/(.*?)\/$/, "$1") + "]"
            }
            if (r) {
                var t = r + " ";
                s.target.ancestor(".comments-inner").one("#message").focus();
                s.target.ancestor(".comments-inner").one("#message").set("value", t)
            }
        }
        function i(q) {
            i.alert = i.alert || new b.FlickrDialog.Alert({
                message: q,
                width: 280
            });
            i.alert.show()
        }
        function l(q) {
            h({
                target: q.one(".comments-inline-btn")
            })
        }
        var f = function() {
            var t = b.one(".comments-popover"), s;
            if (t) {
                var u = b.one(".popover-shim"), r = t.ancestor(p.commentContainerSelector), q = r.all(p.commentButtonNodeSelector || ".il_comments");
                t.remove();
                u.remove();
                q.removeClass("active");
                if (p.onDismiss) {
                    p.onDismiss(r)
                }
                delete b.config[a].open_cards[parseInt(r.getData("photo-id"), 10)];
                n = {}
            }
        };
        c.prototype.dismissCommentsPopover = f;
        c.prototype.loadCommentsCards = m;
        c.prototype.openForNode = l;
        c.prototype.getDismissEvent = function() {
            return e
        };
        b.refreshSIHPComment = c
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
(function() {
    var a = "refresh-sihp-feed-ads";
    YUI.add(a, function(b) {
        var d = b.one("#activityFeed"), k = 8, q = 15, c, n, l, i = 0, h=!!(b.config.flickr.flags.enable_n_to_ldrb_ads), g = (h ? "LDRB" : "N"), f = (h ? "LDRB2" : "N");
        function m(s, u, t) {
            var v = {
                spaceID: 792600147,
                position: t,
                adID: u
            };
            if (F.adHelper) {
                F.adHelper.onEmptyAd(v, function() {
                    s.remove();
                    s = null;
                    v = null;
                    u = null
                })
            }
        }
        function r(t) {
            var s = 0;
            if (t.size() > 0) {
                t.each(function(v, u) {
                    if (v.one(".is-adult")) {
                        s = u + 1
                    }
                })
            }
            return s
        }
        function o(u, y) {
            var t, w, v, s, x;
            w = c.cloneNode(true);
            t = b.guid();
            v = w.one("iframe");
            x = v.get("src");
            if (y > k) {
                s = f;
                x = x.replace("position=" + g, "position=" + f, "gi")
            } else {
                s = g
            }
            v.set("src", x + "adid=" + t);
            u.insert(w, "after");
            m(w, t, s)
        }
        function j() {
            var t, v, u, w, s;
            v = d.all("div.card:not(.ad-card)");
            i = v.size();
            while (n <= i) {
                s = 0;
                u = v.slice(n, n + 3);
                if (u.size() < 3) {
                    return 
                }
                s = r(u);
                if (s > 0) {
                    n += s
                } else {
                    w = v.slice(n - 3, n);
                    s = 0;
                    s = r(w);
                    if (s > 0) {
                        n += s
                    } else {
                        t = v.item(n - 1);
                        o(t, n);
                        n += q
                    }
                }
            }
        }
        function e(t) {
            c = c.get("innerHTML");
            var y = "<!--", s = "-->", x = c.indexOf(y), w = c.lastIndexOf(s), z = d.all("div.card:not(.ad-card)"), v, u;
            i = z.size();
            c = c.substr(x + y.length, w - x - y.length);
            c = b.Node.create('<div class="card ad-card">' + c + "</div>");
            v = c.one("iframe");
            u = v.get("src");
            v.set("src", u.substring(0, u.indexOf("adid")));
            if (!d.one(".ad-card")) {} else {
                d.all(".ad-card").each(function(A) {
                    o(A, n);
                    A.remove();
                    n += q
                })
            }
            l = b.on("SIHP:appended", j)
        }
        function p(s) {
            if (s && s.feedAds) {
                k = s.feedAds.initialAdPos || k;
                q = s.feedAds.frequency || q
            }
            n = k;
            c = b.one("#feed-adframe");
            if (c) {
                e(s)
            } else {}
        }
        b.refreshSIHPFeedAds = {
            init: p
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
(function() {
    var a = "refresh-sihp-favorite";
    YUI.add(a, function(g) {
        function f(i) {
            var h;
            if (!g.Lang.isObject(i)) {
                console.warn("[" + a + "]", "No config");
                return 
            }
            h = i.container;
            if (g.Lang.isString(h)) {
                h = g.one(h)
            }
            if (!g.Lang.isObject(h) ||!h.getDOMNode) {
                console.warn("[" + a + "]", "No container node");
                return 
            }
            h.on("click", c)
        }
        function c(h) {
            if (h.target.ancestor(".il_fav", true)) {
                if (!h.target.ancestor(".own-photo", true)) {
                    d(h)
                } else {
                    h.preventDefault()
                }
            }
        }
        function d(p) {
            var r, v, l, j, k, i, o, n, t, u, q, m, h, s = "is_fav";
            p.preventDefault();
            r = p.target.ancestor(".il_fav", true);
            v = r.ancestor(".imgWrapper");
            m = v.get("parentNode").get("parentNode");
            if (m.hasClass("card-bundle")) {
                o = v.getAttribute("data-photo-id");
                h = m.one(".id" + o)
            }
            q = v.getAttribute("data-photo-owner");
            if (F.config.flickr.user.nsid == q) {
                return 
            }
            if (!r ||!v) {
                console.warn("[" + a + "]", "No fave or photo node");
                return 
            }
            l = v.getAttribute("data-photo-id");
            j = v.getAttribute("data-photo-media");
            k=!r.hasClass(s);
            u = v.one(".il_fav .li_counter");
            i = "flickr.favorites." + (k ? "add" : "remove");
            n = {
                photo_id: l
            };
            t = {
                success: function() {
                    if (u) {
                        e(u, (k) ? 1 : - 1, h)
                    }
                },
                failure: function() {
                    b(r, !k, j, s)
                }
            };
            b(r, k, j, s);
            g.flickrAPI.callMethod(i, n, t)
        }
        function e(m, l, o) {
            var i, n, k, h = (l==-1) ? "0": "1", j;
            i = m.get("innerHTML");
            j = parseInt(i, 10);
            j = j + l;
            if (j < 0) {
                j = 0
            }
            n = m.get("parentNode");
            n.removeClass("counter0");
            n.removeClass("counter1");
            n.removeClass("counter2");
            n.removeClass("counter3");
            if (j) {
                switch (j.toString().length) {
                case 0:
                    n.addClass("counter0");
                    k = "counter0";
                    break;
                case 1:
                    n.addClass("counter1");
                    k = "counter1";
                    break;
                case 2:
                    n.addClass("counter2");
                    k = "counter2";
                    break;
                default:
                    n.addClass("counter3");
                    k = "counter3";
                    break
                }
            } else {
                n.addClass("counter0");
                k = "counter0"
            }
            m.set("innerHTML", j);
            if (o) {
                o.setAttribute("data-fav_class", k).setAttribute("data-fav_count", j).setAttribute("data-is_fav", h)
            }
        }
        function b(i, h, k, j) {
            var l;
            if (h) {
                i.addClass(j);
                l = k === "video" ? "remove_fave_video" : "remove_fave_photo"
            } else {
                i.removeClass(j);
                l = k === "video" ? "add_fave_video" : "add_fave_photo"
            }
            i.setAttribute("title", g.transjax.get("refresh-sihp", l))
        }
        g.refreshSIHPFavorite = {
            init: f
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
(function() {
    var a = "refresh-sihp-keyboard";
    YUI.add(a, function(b) {
        var d, h, e;
        function l(m) {
            var n;
            d = m.container;
            if (b.Lang.isString(d)) {
                d = b.one(d)
            }
            if (!b.Lang.isObject(d) ||!d.getDOMNode) {
                console.warn("[" + a + "]", "No container node");
                return 
            }
            n = b.one("#global-nav");
            if (n) {
                e = (n.get("offsetHeight") + n.getY() - n.get("docScrollY"))
            }
            if (!b.Lang.isNumber(e)) {
                e = 0
            }
            b.keyboardShortcutManager.register({
                keystring: "74,75",
                handler: function(o) {
                    if (parseInt(o.keyCode, 10) === 75) {
                        k(true)
                    } else {
                        k()
                    }
                },
                context: ""
            });
            b.on("SIHP:appended", function(o) {
                delete c.memo
            })
        }
        function c() {
            if (!c.memo) {
                c.memo = d.all(".card")
            }
            return c.memo
        }
        function k(t) {
            var o, w, n, u, m, q, r, v, p, s;
            u = c();
            if (!u.size()) {
                return 
            }
            m = i(u);
            q = m[0];
            r = m[1];
            v = m[2];
            p = h && h.get("running") && k.last_move && k.last_move.direction === "prev";
            s = h && h.get("running") && k.last_move && k.last_move.direction === "next";
            if (t) {
                if (p) {
                    w = k.last_move.destination - 1
                } else {
                    if (b.Lang.isNumber(q) || b.Lang.isNumber(r)) {
                        if (b.Lang.isNumber(q) && b.Lang.isNumber(r) && r < q) {
                            w = r
                        } else {
                            if (b.Lang.isNumber(q)) {
                                w = q - 1
                            } else {
                                if (b.Lang.isNumber(r)) {
                                    if (v) {
                                        w = r
                                    } else {
                                        w = r - 1
                                    }
                                }
                            }
                        }
                    }
                }
                if (b.Lang.isNumber(w)) {
                    if (w < 0) {
                        o = b.one("body")
                    } else {
                        o = u.item(w)
                    }
                }
            } else {
                if (s) {
                    w = k.last_move.destination + 1
                } else {
                    if (b.Lang.isNumber(q) || b.Lang.isNumber(r)) {
                        if (b.Lang.isNumber(q) && b.Lang.isNumber(r) && r < q) {
                            w = q
                        } else {
                            if (b.Lang.isNumber(q)) {
                                w = q
                            } else {
                                if (b.Lang.isNumber(r)) {
                                    if (v) {
                                        w = r + 1
                                    } else {
                                        w = r
                                    }
                                }
                            }
                        }
                    }
                }
                if (b.Lang.isNumber(w)) {
                    if (w < u.size()) {
                        o = u.item(w);
                        n = parseInt(o.getY() - o.get("docScrollY"), 10);
                        if (n <= (10 + e)) {
                            w = w + 1;
                            o = u.item(w)
                        }
                    }
                    if (w >= u.size()) {
                        o = b.one("#foot")
                    }
                }
            }
            if (o) {
                k.last_move = {
                    direction: t ? "prev": "next",
                    destination: w
                };
                g(o, true, 0.4);
                return true
            } else {
                return false
            }
        }
        function i(p, u) {
            var m = p, r, q, t = m.size(), n, w, v, x, o, s;
            for (r = 0; r < t; r++) {
                if (u) {
                    q = t - r - 1
                } else {
                    q = r
                }
                x = f(m.item(q));
                o = x[0];
                s = x[1];
                if (o) {
                    if (b.Lang.isNumber(n) || b.Lang.isNumber(w)) {
                        v = true
                    }
                    if (!b.Lang.isNumber(n)) {
                        n = q
                    }
                } else {
                    if (s) {
                        if (b.Lang.isNumber(n) || b.Lang.isNumber(w)) {
                            v = true
                        }
                        if (!b.Lang.isNumber(w)) {
                            w = q
                        }
                    }
                }
                if (b.Lang.isNumber(n) && b.Lang.isNumber(w) && (v || (!o&&!s))) {
                    break
                }
            }
            return [n, w, v]
        }
        function f(n) {
            var p, m, o, q;
            if (!n) {
                return 
            }
            m = n.get("id");
            if (!m) {
                m = b.guid();
                n.set("id", m)
            }
            if (!f.regions) {
                f.regions = {}
            }
            if (f.regions[m]) {
                p = f.regions[m]
            } else {
                p = n.get("region");
                f.regions[m] = p
            }
            o = j(n, true, p);
            q = j(n, false, p);
            return [o, q]
        }
        function g(n, m, p) {
            var r, q, o = 10;
            if (!b.Lang.isNumber(p)) {
                p = 0.4
            }
            if (m) {
                r = n.getY() - o
            } else {
                r = n.getY() + n.get("offsetHeight") - n.get("winHeight") + o
            }
            r -= e;
            if (r < 0) {
                r = 0
            }
            q = {
                scroll: [0, r]
            };
            if (!h) {
                h = new b.Anim({
                    node: b.one("window"),
                    to: q,
                    duration: p,
                    easing: b.Easing.easeOut
                })
            } else {
                if (h.get("running")) {
                    h.stop()
                }
                h.set("to", q)
            }
            h.run()
        }
        g = b.betterThrottle(g, 50);
        function j(n, m, o) {
            var p;
            p = b.DOM.viewportRegion(n);
            p.top += e;
            p.height -= e;
            return b.DOM.inRegion(n, p, m, o)
        }
        b.refreshSIHPKeyboard = {
            init: l
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
(function() {
    var a = "refresh-sihp-muting";
    YUI.add(a, function(f) {
        function e(h) {
            var g;
            if (!f.Lang.isObject(h)) {
                console.warn("[" + a + "]", "No config");
                return 
            }
            g = h.container;
            if (f.Lang.isString(g)) {
                g = f.one(g)
            }
            if (!f.Lang.isObject(g) ||!g.getDOMNode) {
                console.warn("[" + a + "]", "No container node");
                return 
            }
            f.one("#activityFeed").delegate("click", c, ".mute")
        }
        function c(m) {
            m.preventDefault();
            var p = m.target.ancestor(".imgContainer") || m.target.ancestor(".micro-card"), s = p.one(".imgWrapper") || m.target.ancestor(".micro-card"), i = s.getAttribute("data-photo-id"), h = s.getAttribute("data-photo-owner"), n = p.getComputedStyle("width"), o = p.getComputedStyle("height"), j = s.getAttribute("data-card-type"), r = s.getAttribute("data-share-id"), l = p.hasClass("is_muted"), g = "flickr.activity." + (l ? "unmuteObject" : "muteObject"), k = {
                photo_id: i
            }, q = {
                success: function() {}
            };
            if (j === "shared_to") {
                k.share_id = r
            }
            if (!i) {
                k = {
                    contact_id: h
                }
            }
            if (l) {
                b(p);
                q.failure = function() {
                    d(p, n, o)
                }
            } else {
                d(p, n, o);
                q.failure = function() {
                    b(p)
                }
            }
            f.flickrAPI.callMethod(g, k, q)
        }
        function d(h, i, g) {
            h.addClass("is_muted");
            h.append("<div class='mute-overlay'></div>")
        }
        function b(g) {
            g.removeClass("is_muted");
            g.one(".mute-overlay").remove()
        }
        f.refreshSIHPMuting = {
            init: e
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
(function() {
    var a = "refresh-sihp-share";
    YUI.add(a, function(b) {
        var d, e, k;
        var c;
        function j(l) {
            if (!b.Lang.isObject(l)) {
                return 
            }
            k = l;
            c = l.container;
            if (b.Lang.isString(c)) {
                c = b.one(c)
            }
            if (!b.Lang.isObject(c) ||!c.getDOMNode) {
                return 
            }
            k.shareIconSelector = k.shareIconSelector || ".il_share .li_icon";
            k.a_nodeSelector = k.a_nodeSelector || "li";
            k.imgWrapperSelector = k.imgWrapperSelector || ".imgWrapper";
            k.shareButtonSelector = k.shareButtonSelector || "li";
            c.delegate("click", g, k.shareIconSelector);
            if (k.altButtonSelector) {
                b.one(k.altButtonSelector).on("click", i)
            }
            e = k.dismissCommentEvent
        }
        function i(n) {
            n.preventDefault();
            b.config.flickr.sharing = {
                use_share_this_v3: true,
                object: {
                    id: k.altId,
                    owner_nsid: k.altOwner,
                    type: k.altType,
                },
                shareability_is_unknown: true
            };
            var l = b.one(k.altButtonSelector);
            var m = l.one("a");
            h(m);
            m.removeClass("sn-ico-share");
            if (!b.shareThisV3Menu.isInitialized()) {
                b.refreshSIHPShare.menu = b.shareThisV3Menu.init({
                    button_node: l,
                    arrowPosition: "tr",
                    overrideRefresh: k.overrideRefresh || overrideRefresh,
                    scroll: true,
                    bottom: k.bottom,
                    right: k.right,
                    afterInit: function() {
                        f(m);
                        m.addClass("sn-ico-share");
                        b.shareThisV3Menu.show()
                    }
                })
            } else {
                if (!b.shareThisV3Menu.getMenu().get("popover").get("visible")) {
                    b.shareThisV3Menu.hide();
                    b.shareThisV3Menu.clear();
                    b.shareThisV3Menu.getMenu().set("button_node", b.one(k.altButtonSelector));
                    b.shareThisV3Menu.refresh(function() {
                        f(m);
                        m.addClass("sn-ico-share");
                        b.shareThisV3Menu.show()
                    })
                } else {
                    f(m);
                    m.addClass("sn-ico-share")
                }
            }
        }
        function g(q) {
            q.preventDefault();
            b.fire(e);
            var r = q.target.ancestor(k.shareIconSelector, true), u = q.target.ancestor(k.a_nodeSelector), v = r.ancestor(k.imgWrapperSelector), m = v && v.getAttribute("data-photo-id"), n = v && v.getAttribute("data-photo-owner"), p = q.target.ancestor(k.shareButtonSelector);
            if (m && n) {
                if (d === v.getDOMNode().id) {
                    d = null
                } else {
                    d = v.getDOMNode().id;
                    b.config.flickr.sharing = {
                        use_share_this_v3: true,
                        shareability_is_unknown: true,
                        object: {
                            id: m,
                            owner_nsid: n,
                            type: "photo"
                        }
                    };
                    var s = "br", l, t;
                    if (v.ancestor(".firstCard")) {
                        (function o() {
                            function x(y) {
                                var z = {
                                    x: 0,
                                    y: 0
                                };
                                while (y) {
                                    z.x += y.offsetLeft;
                                    z.y += y.offsetTop;
                                    y = y.offsetParent
                                }
                                return z
                            }
                            var w = x(r.getDOMNode());
                            if (w.y < 411) {
                                s = "tr";
                                l = true;
                                t = true
                            }
                        })()
                    }
                    if (!b.shareThisV3Menu.isInitialized()) {
                        h(u);
                        b.refreshSIHPShare.menu = b.shareThisV3Menu.init({
                            button_node: r,
                            arrowPosition: k.pos || s,
                            overrideRefresh: k.overrideRefresh || l,
                            scroll: k.scroll || t,
                            bottom: k.bottom,
                            right: k.right,
                            afterInit: function() {
                                f(u);
                                b.shareThisV3Menu.show();
                                if (k.inlineRealign) {
                                    var w = b.one("#share-menu-v3").siblings(".yui3-popover-arrow").item(0);
                                    var y = w.getStyle("top");
                                    var x = w.getStyle("right");
                                    if (y && x) {
                                        y = parseInt(y.replace("px", ""), 10);
                                        x = parseInt(x.replace("px", ""), 10);
                                        y = y + k.adjustTop;
                                        x = x + k.adjustRight;
                                        w.setStyle("top", y + "px").setStyle("right", x + "px")
                                    }
                                }
                                p.addClass("active");
                                if (k.onSuccess) {
                                    k.onSuccess(v, p)
                                }
                            },
                            onHide: function() {
                                if (k.onDismiss) {
                                    k.onDismiss(v, p)
                                }
                            }
                        })
                    } else {
                        b.shareThisV3Menu.hide();
                        b.shareThisV3Menu.clear();
                        h(u);
                        b.shareThisV3Menu.getMenu().set("button_node", r);
                        b.shareThisV3Menu.getMenu().set("onHide", function() {
                            if (k.onDismiss) {
                                k.onDismiss(v, p)
                            }
                        });
                        b.shareThisV3Menu.refresh(function() {
                            f(u);
                            b.shareThisV3Menu.show();
                            p.addClass("active");
                            if (k.onSuccess) {
                                k.onSuccess(v, p)
                            }
                            if (k.inlineRealign) {
                                var w = b.one("#share-menu-v3").siblings(".yui3-popover-arrow").item(0);
                                var y = w.getStyle("top");
                                var x = w.getStyle("right");
                                if (y && x) {
                                    y = parseInt(y.replace("px", ""), 10);
                                    x = parseInt(x.replace("px", ""), 10);
                                    y = y + 6;
                                    x = x + 8;
                                    w.setStyle("top", y + "px").setStyle("right", x + "px")
                                }
                            }
                        })
                    }
                }
            }
        }
        function h(l) {
            if (l.one(".li_icon")) {
                l.one(".li_icon").hide()
            } else {
                if (k.onLoading) {
                    k.onLoading(l)
                }
            }
            l.append('<span class="html5-balls"><span class="blueball"></span><span class="pinkball"></span></span>')
        }
        function f(l) {
            l.one(".html5-balls").remove();
            if (l.one(".li_icon")) {
                l.one(".li_icon").show()
            }
        }
        b.refreshSIHPShare = {
            init: j
        }
    }, "0.0.1", {
        requires: F.config.modules[a].requires || [],
        optional: F.config.modules[a].optional || []
    })
}());
