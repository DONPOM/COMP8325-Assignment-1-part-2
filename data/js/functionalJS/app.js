/*! Dust - Asynchronous Templating - v2.3.4
* http://linkedin.github.io/dustjs/
* Copyright (c) 2014 Aleksander Williams; Released under the MIT License */
function printStackTrace(a) {
    var a = a || {
        guess: !0
    }, b = a.e || null, a=!!a.guess, c = new printStackTrace.implementation, b = c.run(b);
    return a ? c.guessFunctions(b) : b
}
var Hoptoad = {
    VERSION: "0.1.0",
    NOTICE_XML: '<?xml version="1.0" encoding="UTF-8"?><notice version="2.0"><api-key></api-key><notifier><name>airbrake_js</name><version>0.1.0</version><url>http://airbrake.io</url></notifier><error><class>EXCEPTION_CLASS</class><message>EXCEPTION_MESSAGE</message><backtrace>BACKTRACE_LINES</backtrace></error><request><url>REQUEST_URL</url><component>REQUEST_COMPONENT</component><action>REQUEST_ACTION</action></request><server-environment><project-root>PROJECT_ROOT</project-root><environment-name>production</environment-name></server-environment></notice>',
    ROOT: window.location.protocol + "//" + window.location.host,
    BACKTRACE_MATCHER: /^(.*)\@(.*)\:(\d+)$/,
    backtrace_filters: [/notifier\.js/],
    IEVersion: /MSIE (\d+\.\d+);/.test(navigator.userAgent) ? parseFloat(RegExp.$1): NaN,
    notify: function(a) {
        var b = escape(Hoptoad.generateXML(a)), c = Hoptoad.host || "airbrake.io", d = "//" + c + "/notifier_api/v2/notices?data=" + b, e = document.createElement("iframe");
        e.style.width = "1px", e.style.height = "1px", e.style.display = "none", e.src = d, document.body && (document.body.appendChild(e), setTimeout(function() {
            document.body.removeChild(e)
        }, 2e3))
    },
    setEnvironment: function(a) {
        var b = /<environment-name>.*<\/environment-name>/;
        Hoptoad.NOTICE_XML = Hoptoad.NOTICE_XML.replace(b, "<environment-name>" + a + "</environment-name>")
    },
    setHost: function(a) {
        Hoptoad.host = a
    },
    setKey: function(a) {
        var b = /<api-key>.*<\/api-key>/;
        Hoptoad.NOTICE_XML = Hoptoad.NOTICE_XML.replace(b, "<api-key>" + a + "</api-key>")
    },
    setErrorDefaults: function(a) {
        Hoptoad.errorDefaults = a
    },
    generateXML: function(a) {
        var b = Hoptoad.mergeDefault(Hoptoad.errorDefaults, a), c = Hoptoad.NOTICE_XML, d = b.url || "", e = location.hash || "", f = Hoptoad.escapeText(d + e || ""), g = Hoptoad.escapeText(b.component || ""), h = Hoptoad.escapeText(b.action || ""), i = Hoptoad.escapeText(b.type || "Error"), j = Hoptoad.escapeText(b.message || "Unknown error."), k = Hoptoad.IEVersion < 9 ? []: Hoptoad.generateBacktrace(b);
        j.length > 2e3 && (j = j.substr(0, 2e3));
        if (Hoptoad.trim(f) === "" && Hoptoad.trim(g) === "")
            c = c.replace(/<request>.*<\/request>/, "");
        else {
            var l = "", m = b["cgi-data"] || {};
            m.HTTP_USER_AGENT = navigator.userAgent, l += "<cgi-data>", l += Hoptoad.generateVariables(m), l += "</cgi-data>";
            var n = ["params", "session"];
            for (var o = 0; o < 2; o++) {
                var p = n[o];
                b[p] && (l += "<" + p + ">", l += Hoptoad.generateVariables(b[p]), l += "</" + p + ">")
            }
            c = c.replace("</request>", l + "</request>").replace("REQUEST_URL", f).replace("REQUEST_ACTION", h).replace("REQUEST_COMPONENT", g)
        }
        return c.replace("PROJECT_ROOT", Hoptoad.ROOT).replace("EXCEPTION_CLASS", i).replace("EXCEPTION_MESSAGE", j).replace("BACKTRACE_LINES", k.join(""))
    },
    generateBacktrace: function(a) {
        a = a || {};
        if (typeof a.stack != "string")
            try {
                (0)()
        } catch (b) {
            a.stack = b.stack
        }
        var c = [], d = Hoptoad.getStackTrace(a);
        for (var e = 0, f = d.length; e < f; e++) {
            var g = d[e], h = g.match(Hoptoad.BACKTRACE_MATCHER);
            if (h && Hoptoad.validBacktraceLine(g)) {
                var i = h[2].replace(Hoptoad.ROOT, "[PROJECT_ROOT]");
                e == 0 && h[2].match(document.location.href) && c.push('<line method="" file="internal: " number=""/>'), c.push('<line method="' + Hoptoad.escapeText(h[1]) + '" file="' + Hoptoad.escapeText(i) + '" number="' + h[3] + '" />')
            }
        }
        return c
    },
    getStackTrace: function(a) {
        var b = printStackTrace({
            e: a,
            guess: !1
        });
        for (var c = 0, d = b.length; c < d; c++) {
            if (b[c].match(/\:\d+$/))
                continue;
            b[c].indexOf("@")==-1 && (b[c] += "@unsupported.js"), b[c] += ":0"
        }
        return b
    },
    validBacktraceLine: function(a) {
        for (var b = 0; b < Hoptoad.backtrace_filters.length; b++)
            if (a.match(Hoptoad.backtrace_filters[b]))
                return !1;
        return !0
    },
    generateVariables: function(a) {
        var b, c = "";
        for (b in a)
            c += '<var key="' + Hoptoad.escapeText(b) + '">' + Hoptoad.escapeText(a[b]) + "</var>";
        return c
    },
    escapeText: function(a) {
        return a.replace(/&/g, "&#38;").replace(/</g, "&#60;").replace(/>/g, "&#62;").replace(/'/g, "&#39;").replace(/"/g, "&#34;")
    },
    trim: function(a) {
        return a.toString().replace(/^\s+/, "").replace(/\s+$/, "")
    },
    mergeDefault: function(a, b) {
        var c = {}, d;
        for (d in b)
            c[d] = b[d];
        for (d in a)
            c.hasOwnProperty(d) || (c[d] = a[d]);
        return c
    }
};
printStackTrace.implementation = function() {}, printStackTrace.implementation.prototype = {
    run: function(a) {
        var a = a || this.createException(), b = this.mode(a);
        return b === "other" ? this.other(arguments.callee) : this[b](a)
    },
    createException: function() {
        try {
            return this.undef(), null
        } catch (a) {
            return a
        }
    },
    mode: function(a) {
        return a.arguments && a.stack ? this._mode = "chrome" : typeof window != "undefined" && window.opera ? this._mode = a.stacktrace ? "opera10" : "opera" : a.stack ? this._mode = "firefox" : this._mode = "other"
    },
    instrumentFunction: function(a, b, c) {
        a = a || window, a["_old" + b] = a[b], a[b] = function() {
            return c.call(this, printStackTrace()), a["_old" + b].apply(this, arguments)
        }, a[b]._instrumented=!0
    },
    deinstrumentFunction: function(a, b) {
        a[b].constructor === Function && a[b]._instrumented && a["_old" + b].constructor === Function && (a[b] = a["_old" + b])
    },
    chrome: function(a) {
        return a.stack.replace(/^[^\(]+?[\n$]/gm, "").replace(/^\s+at\s+/gm, "").replace(/^Object.<anonymous>\s*\(/gm, "{anonymous}()@").split("\n")
    },
    firefox: function(a) {
        return a.stack.replace(/(?:\n@:0)?\s+$/m, "").replace(/^\(/gm, "{anonymous}(").split("\n")
    },
    opera10: function(a) {
        var a = a.stacktrace.split("\n"), b = /.*line (\d+), column (\d+) in ((<anonymous function\:?\s*(\S+))|([^\(]+)\([^\)]*\))(?: in )?(.*)\s*$/i, c, d, e;
        c = 2, d = 0;
        for (e = a.length; c < e - 2; c++)
            if (b.test(a[c])) {
                var f = RegExp.$6 + ":" + RegExp.$1 + ":" + RegExp.$2, g = RegExp.$3, g = g.replace(/<anonymous function\:?\s?(\S+)?>/g, "{anonymous}");
                a[d++] = g + "@" + f
            }
        return a.splice(d, a.length - d), a
    },
    opera: function(a) {
        var a = a.message.split("\n"), b = /Line\s+(\d+).*script\s+(http\S+)(?:.*in\s+function\s+(\S+))?/i, c, d, e;
        c = 4, d = 0;
        for (e = a.length; c < e; c += 2)
            b.test(a[c]) && (a[d++] = (RegExp.$3 ? RegExp.$3 + "()@" + RegExp.$2 + RegExp.$1 : "{anonymous}()@" + RegExp.$2 + ":" + RegExp.$1) + " -- " + a[c + 1].replace(/^\s+/, ""));
        return a.splice(d, a.length - d), a
    },
    other: function(a) {
        for (var b = /function\s*([\w\-$]+)?\s*\(/i, c = [], d, e; a && c.length < 10;)
            d = b.test(a.toString()) ? RegExp.$1 || "{anonymous}" : "{anonymous}", e = Array.prototype.slice.call(a.arguments || []), c[c.length] = d + "(" + this.stringifyArguments(e) + ")", a = a.caller;
        return c
    },
    stringifyArguments: function(a) {
        for (var b = 0; b < a.length; ++b) {
            var c = a[b];
            c === void 0 ? a[b] = "undefined" : c === null ? a[b] = "null" : c.constructor && (c.constructor === Array ? a[b] = c.length < 3 ? "[" + this.stringifyArguments(c) + "]" : "[" + this.stringifyArguments(Array.prototype.slice.call(c, 0, 1)) + "..." + this.stringifyArguments(Array.prototype.slice.call(c, - 1)) + "]" : c.constructor === Object ? a[b] = "#object" : c.constructor === Function ? a[b] = "#function" : c.constructor === String && (a[b] = '"' + c + '"'))
        }
        return a.join(",")
    },
    sourceCache: {},
    ajax: function(a) {
        var b = this.createXMLHTTPObject();
        if (b)
            return b.open("GET", a, !1), b.setRequestHeader("User-Agent", "XMLHTTP/1.0"), b.send(""), b.responseText
    },
    createXMLHTTPObject: function() {
        for (var a, b = [function() {
            return new XMLHttpRequest
        }, function() {
            return new ActiveXObject("Msxml2.XMLHTTP")
        }, function() {
            return new ActiveXObject("Msxml3.XMLHTTP")
        }, function() {
            return new ActiveXObject("Microsoft.XMLHTTP")
        }
        ], c = 0; c < b.length; c++)
            try {
                return a = b[c](), this.createXMLHTTPObject = b[c], a
        } catch (d) {}
    },
    isSameDomain: function(a) {
        return a.indexOf(location.hostname)!==-1
    },
    getSource: function(a) {
        return a in this.sourceCache || (this.sourceCache[a] = this.ajax(a).split("\n")), this.sourceCache[a]
    },
    guessFunctions: function(a) {
        for (var b = 0; b < a.length; ++b) {
            var c = a[b], d = /\{anonymous\}\(.*\)@(\w+:\/\/([\-\w\.]+)+(:\d+)?[^:]+):(\d+):?(\d+)?/.exec(c);
            if (d) {
                var e = d[1], d = d[4];
                e && this.isSameDomain(e) && d && (e = this.guessFunctionName(e, d), a[b] = c.replace("{anonymous}", e))
            }
        }
        return a
    },
    guessFunctionName: function(a, b) {
        var c;
        try {
            c = this.guessFunctionNameFromLines(b, this.getSource(a))
        } catch (d) {
            c = "getSource failed with url: " + a + ", exception: " + d.toString()
        }
        return c
    },
    guessFunctionNameFromLines: function(a, b) {
        for (var c = /function ([^(]*)\(([^)]*)\)/, d = /['"]?([0-9A-Za-z_]+)['"]?\s*[:=]\s*(function|eval|new Function)/, e = "", f = 0; f < 10; ++f)
            if (e = b[a - f] + e, e !== void 0) {
                var g = d.exec(e);
                if (g && g[1])
                    return g[1];
                    if ((g = c.exec(e)) && g[1])
                        return g[1]
            }
        return "(?)"
    }
}, define("internal/sitebuilder/common/airbrakeNotifier", [], function() {});
var root = this;
define("underscore", [], function() {
    var a = root._, b = {}, c = Array.prototype, d = Object.prototype, e = Function.prototype, f = c.push, g = c.slice, h = c.concat, i = c.unshift, j = d.toString, k = d.hasOwnProperty, l = c.forEach, m = c.map, n = c.reduce, o = c.reduceRight, p = c.filter, q = c.every, r = c.some, s = c.indexOf, t = c.lastIndexOf, u = Array.isArray, v = Object.keys, w = e.bind, x = function(a) {
        if (a instanceof x)
            return a;
        if (!(this instanceof x))
            return new x(a);
        this._wrapped = a
    };
    typeof exports != "undefined" ? (typeof module != "undefined" && module.exports && (exports = module.exports = x), exports._ = x) : root._ = x, x.VERSION = "1.4.2";
    var y = x.each = x.forEach = function(a, c, d) {
        if (a == null)
            return;
        if (l && a.forEach === l)
            a.forEach(c, d);
        else if (a.length ===+ a.length) {
            for (var e = 0, f = a.length; e < f; e++)
                if (c.call(d, a[e], e, a) === b)
                    return 
        } else 
            for (var g in a)
                if (x.has(a, g) && c.call(d, a[g], g, a) === b)
                    return 
    };
    x.map = x.collect = function(a, b, c) {
        var d = [];
        return a == null ? d : m && a.map === m ? a.map(b, c) : (y(a, function(a, e, f) {
            d[d.length] = b.call(c, a, e, f)
        }), d)
    }, x.reduce = x.foldl = x.inject = function(a, b, c, d) {
        var e = arguments.length > 2;
        a == null && (a = []);
        if (n && a.reduce === n)
            return d && (b = x.bind(b, d)), e ? a.reduce(b, c) : a.reduce(b);
        y(a, function(a, f, g) {
            e ? c = b.call(d, c, a, f, g) : (c = a, e=!0)
        });
        if (!e)
            throw new TypeError("Reduce of empty array with no initial value");
        return c
    }, x.reduceRight = x.foldr = function(a, b, c, d) {
        var e = arguments.length > 2;
        a == null && (a = []);
        if (o && a.reduceRight === o)
            return d && (b = x.bind(b, d)), arguments.length > 2 ? a.reduceRight(b, c) : a.reduceRight(b);
        var f = a.length;
        if (f!==+f) {
            var g = x.keys(a);
            f = g.length
        }
        y(a, function(h, i, j) {
            i = g ? g[--f] : --f, e ? c = b.call(d, c, a[i], i, j) : (c = a[i], e=!0)
        });
        if (!e)
            throw new TypeError("Reduce of empty array with no initial value");
        return c
    }, x.find = x.detect = function(a, b, c) {
        var d;
        return z(a, function(a, e, f) {
            if (b.call(c, a, e, f))
                return d = a, !0
        }), d
    }, x.filter = x.select = function(a, b, c) {
        var d = [];
        return a == null ? d : p && a.filter === p ? a.filter(b, c) : (y(a, function(a, e, f) {
            b.call(c, a, e, f) && (d[d.length] = a)
        }), d)
    }, x.reject = function(a, b, c) {
        var d = [];
        return a == null ? d : (y(a, function(a, e, f) {
            b.call(c, a, e, f) || (d[d.length] = a)
        }), d)
    }, x.every = x.all = function(a, c, d) {
        c || (c = x.identity);
        var e=!0;
        return a == null ? e : q && a.every === q ? a.every(c, d) : (y(a, function(a, f, g) {
            if (!(e = e && c.call(d, a, f, g)))
                return b
        }), !!e)
    };
    var z = x.some = x.any = function(a, c, d) {
        c || (c = x.identity);
        var e=!1;
        return a == null ? e : r && a.some === r ? a.some(c, d) : (y(a, function(a, f, g) {
            if (e || (e = c.call(d, a, f, g)))
                return b
        }), !!e)
    };
    x.contains = x.include = function(a, b) {
        var c=!1;
        return a == null ? c : s && a.indexOf === s ? a.indexOf(b)!=-1 : (c = z(a, function(a) {
            return a === b
        }), c)
    }, x.invoke = function(a, b) {
        var c = g.call(arguments, 2);
        return x.map(a, function(a) {
            return (x.isFunction(b) ? b : a[b]).apply(a, c)
        })
    }, x.pluck = function(a, b) {
        return x.map(a, function(a) {
            return a[b]
        })
    }, x.where = function(a, b) {
        return x.isEmpty(b) ? [] : x.filter(a, function(a) {
            for (var c in b)
                if (b[c] !== a[c])
                    return !1;
            return !0
        })
    }, x.max = function(a, b, c) {
        if (!b && x.isArray(a) && a[0] ===+ a[0] && a.length < 65535)
            return Math.max.apply(Math, a);
        if (!b && x.isEmpty(a))
            return - Infinity;
        var d = {
            computed: - Infinity
        };
        return y(a, function(a, e, f) {
            var g = b ? b.call(c, a, e, f): a;
            g >= d.computed && (d = {
                value: a,
                computed: g
            })
        }), d.value
    }, x.min = function(a, b, c) {
        if (!b && x.isArray(a) && a[0] ===+ a[0] && a.length < 65535)
            return Math.min.apply(Math, a);
        if (!b && x.isEmpty(a))
            return Infinity;
        var d = {
            computed: Infinity
        };
        return y(a, function(a, e, f) {
            var g = b ? b.call(c, a, e, f): a;
            g < d.computed && (d = {
                value: a,
                computed: g
            })
        }), d.value
    }, x.shuffle = function(a) {
        var b, c = 0, d = [];
        return y(a, function(a) {
            b = x.random(c++), d[c - 1] = d[b], d[b] = a
        }), d
    };
    var A = function(a) {
        return x.isFunction(a) ? a : function(b) {
            return b[a]
        }
    };
    x.sortBy = function(a, b, c) {
        var d = A(b);
        return x.pluck(x.map(a, function(a, b, e) {
            return {
                value: a,
                index: b,
                criteria: d.call(c, a, b, e)
            }
        }).sort(function(a, b) {
            var c = a.criteria, d = b.criteria;
            if (c !== d) {
                if (c > d || c === void 0)
                    return 1;
                if (c < d || d === void 0)
                    return - 1
            }
            return a.index < b.index?-1 : 1
        }), "value")
    };
    var B = function(a, b, c, d) {
        var e = {}, f = A(b);
        return y(a, function(b, g) {
            var h = f.call(c, b, g, a);
            d(e, h, b)
        }), e
    };
    x.groupBy = function(a, b, c) {
        return B(a, b, c, function(a, b, c) {
            (x.has(a, b) ? a[b] : a[b] = []).push(c)
        })
    }, x.countBy = function(a, b, c) {
        return B(a, b, c, function(a, b, c) {
            x.has(a, b) || (a[b] = 0), a[b]++
        })
    }, x.sortedIndex = function(a, b, c, d) {
        c = c == null ? x.identity : A(c);
        var e = c.call(d, b), f = 0, g = a.length;
        while (f < g) {
            var h = f + g>>>1;
            c.call(d, a[h]) < e ? f = h + 1 : g = h
        }
        return f
    }, x.toArray = function(a) {
        return a ? a.length ===+ a.length ? g.call(a) : x.values(a) : []
    }, x.size = function(a) {
        return a.length ===+ a.length ? a.length : x.keys(a).length
    }, x.first = x.head = x.take = function(a, b, c) {
        return b != null&&!c ? g.call(a, 0, b) : a[0]
    }, x.initial = function(a, b, c) {
        return g.call(a, 0, a.length - (b == null || c ? 1 : b))
    }, x.last = function(a, b, c) {
        return b != null&&!c ? g.call(a, Math.max(a.length - b, 0)) : a[a.length - 1]
    }, x.rest = x.tail = x.drop = function(a, b, c) {
        return g.call(a, b == null || c ? 1 : b)
    }, x.compact = function(a) {
        return x.filter(a, function(a) {
            return !!a
        })
    };
    var C = function(a, b, c) {
        return y(a, function(a) {
            x.isArray(a) ? b ? f.apply(c, a) : C(a, b, c) : c.push(a)
        }), c
    };
    x.flatten = function(a, b) {
        return C(a, b, [])
    }, x.without = function(a) {
        return x.difference(a, g.call(arguments, 1))
    }, x.uniq = x.unique = function(a, b, c, d) {
        var e = c ? x.map(a, c, d): a, f = [], g = [];
        return y(e, function(c, d) {
            if (b?!d || g[g.length - 1] !== c : !x.contains(g, c)
                )g.push(c), f.push(a[d])
        }), f
    }, x.union = function() {
        return x.uniq(h.apply(c, arguments))
    }, x.intersection = function(a) {
        var b = g.call(arguments, 1);
        return x.filter(x.uniq(a), function(a) {
            return x.every(b, function(b) {
                return x.indexOf(b, a) >= 0
            })
        })
    }, x.difference = function(a) {
        var b = h.apply(c, g.call(arguments, 1));
        return x.filter(a, function(a) {
            return !x.contains(b, a)
        })
    }, x.zip = function() {
        var a = g.call(arguments), b = x.max(x.pluck(a, "length")), c = new Array(b);
        for (var d = 0; d < b; d++)
            c[d] = x.pluck(a, "" + d);
        return c
    }, x.object = function(a, b) {
        var c = {};
        for (var d = 0, e = a.length; d < e; d++)
            b ? c[a[d]] = b[d] : c[a[d][0]] = a[d][1];
        return c
    }, x.indexOf = function(a, b, c) {
        if (a == null)
            return - 1;
        var d = 0, e = a.length;
        if (c) {
            if (typeof c != "number")
                return d = x.sortedIndex(a, b), a[d] === b ? d : - 1;
            d = c < 0 ? Math.max(0, e + c) : c
        }
        if (s && a.indexOf === s)
            return a.indexOf(b, c);
        for (; d < e; d++)
            if (a[d] === b)
                return d;
        return - 1
    }, x.lastIndexOf = function(a, b, c) {
        if (a == null)
            return - 1;
        var d = c != null;
        if (t && a.lastIndexOf === t)
            return d ? a.lastIndexOf(b, c) : a.lastIndexOf(b);
        var e = d ? c: a.length;
        while (e--)
            if (a[e] === b)
                return e;
        return - 1
    }, x.range = function(a, b, c) {
        arguments.length <= 1 && (b = a || 0, a = 0), c = arguments[2] || 1;
        var d = Math.max(Math.ceil((b - a) / c), 0), e = 0, f = new Array(d);
        while (e < d)
            f[e++] = a, a += c;
        return f
    };
    var D = function() {};
    x.bind = function(b, c) {
        var d, e;
        if (b.bind === w && w)
            return w.apply(b, g.call(arguments, 1));
        if (!x.isFunction(b))
            throw new TypeError;
        return e = g.call(arguments, 2), d = function() {
            if (this instanceof d) {
                D.prototype = b.prototype;
                var a = new D, f = b.apply(a, e.concat(g.call(arguments)));
                return Object(f) === f ? f : a
            }
            return b.apply(c, e.concat(g.call(arguments)))
        }
    }, x.bindAll = function(a) {
        var b = g.call(arguments, 1);
        return b.length == 0 && (b = x.functions(a)), y(b, function(b) {
            a[b] = x.bind(a[b], a)
        }), a
    }, x.memoize = function(a, b) {
        var c = {};
        return b || (b = x.identity), function() {
            var d = b.apply(this, arguments);
            return x.has(c, d) ? c[d] : c[d] = a.apply(this, arguments)
        }
    }, x.delay = function(a, b) {
        var c = g.call(arguments, 2);
        return setTimeout(function() {
            return a.apply(null, c)
        }, b)
    }, x.defer = function(a) {
        return x.delay.apply(x, [a, 1].concat(g.call(arguments, 1)))
    }, x.throttle = function(a, b) {
        var c, d, e, f, g, h, i = x.debounce(function() {
            g = f=!1
        }, b);
        return function() {
            c = this, d = arguments;
            var j = function() {
                e = null, g && (h = a.apply(c, d)), i()
            };
            return e || (e = setTimeout(j, b)), f ? g=!0 : (f=!0, h = a.apply(c, d)), i(), h
        }
    }, x.debounce = function(a, b, c) {
        var d, e;
        return function() {
            var f = this, g = arguments, h = function() {
                d = null, c || (e = a.apply(f, g))
            }, i = c&&!d;
            return clearTimeout(d), d = setTimeout(h, b), i && (e = a.apply(f, g)), e
        }
    }, x.once = function(a) {
        var b=!1, c;
        return function() {
            return b ? c : (b=!0, c = a.apply(this, arguments), a = null, c)
        }
    }, x.wrap = function(a, b) {
        return function() {
            var c = [a];
            return f.apply(c, arguments), b.apply(this, c)
        }
    }, x.compose = function() {
        var a = arguments;
        return function() {
            var b = arguments;
            for (var c = a.length - 1; c >= 0; c--)
                b = [a[c].apply(this, b)];
            return b[0]
        }
    }, x.after = function(a, b) {
        return a <= 0 ? b() : function() {
            if (--a < 1)
                return b.apply(this, arguments)
        }
    }, x.keys = v || function(a) {
        if (a !== Object(a))
            throw new TypeError("Invalid object");
        var b = [];
        for (var c in a)
            x.has(a, c) && (b[b.length] = c);
        return b
    }, x.values = function(a) {
        var b = [];
        for (var c in a)
            x.has(a, c) && b.push(a[c]);
        return b
    }, x.pairs = function(a) {
        var b = [];
        for (var c in a)
            x.has(a, c) && b.push([c, a[c]]);
        return b
    }, x.invert = function(a) {
        var b = {};
        for (var c in a)
            x.has(a, c) && (b[a[c]] = c);
        return b
    }, x.functions = x.methods = function(a) {
        var b = [];
        for (var c in a)
            x.isFunction(a[c]) && b.push(c);
        return b.sort()
    }, x.extend = function(a) {
        return y(g.call(arguments, 1), function(b) {
            for (var c in b)
                a[c] = b[c]
        }), a
    }, x.pick = function(a) {
        var b = {}, d = h.apply(c, g.call(arguments, 1));
        return y(d, function(c) {
            c in a && (b[c] = a[c])
        }), b
    }, x.omit = function(a) {
        var b = {}, d = h.apply(c, g.call(arguments, 1));
        for (var e in a)
            x.contains(d, e) || (b[e] = a[e]);
        return b
    }, x.defaults = function(a) {
        return y(g.call(arguments, 1), function(b) {
            for (var c in b)
                a[c] == null && (a[c] = b[c])
        }), a
    }, x.clone = function(a) {
        return x.isObject(a) ? x.isArray(a) ? a.slice() : x.extend({}, a) : a
    }, x.tap = function(a, b) {
        return b(a), a
    };
    var E = function(a, b, c, d) {
        if (a === b)
            return a !== 0 || 1 / a == 1 / b;
        if (a == null || b == null)
            return a === b;
        a instanceof x && (a = a._wrapped), b instanceof x && (b = b._wrapped);
        var e = j.call(a);
        if (e != j.call(b))
            return !1;
        switch (e) {
        case"[object String]":
            return a == String(b);
        case"[object Number]":
            return a!=+a ? b!=+b : a == 0 ? 1 / a == 1 / b : a ==+ b;
        case"[object Date]":
        case"[object Boolean]":
            return + a ==+ b;
        case"[object RegExp]":
            return a.source == b.source && a.global == b.global && a.multiline == b.multiline && a.ignoreCase == b.ignoreCase
        }
        if (typeof a != "object" || typeof b != "object")
            return !1;
        var f = c.length;
        while (f--)
            if (c[f] == a)
                return d[f] == b;
        c.push(a), d.push(b);
        var g = 0, h=!0;
        if (e == "[object Array]") {
            g = a.length, h = g == b.length;
            if (h)
                while (g--)
                    if (!(h = E(a[g], b[g], c, d)))
                        break
        } else {
            var i = a.constructor, k = b.constructor;
            if (i !== k&&!(x.isFunction(i) && i instanceof i && x.isFunction(k) && k instanceof k))
                return !1;
            for (var l in a)
                if (x.has(a, l)) {
                    g++;
                    if (!(h = x.has(b, l) && E(a[l], b[l], c, d)))
                        break
                }
            if (h) {
                for (l in b)
                    if (x.has(b, l)&&!(g--))
                        break;
                h=!g
            }
        }
        return c.pop(), d.pop(), h
    };
    x.isEqual = function(a, b) {
        return E(a, b, [], [])
    }, x.isEmpty = function(a) {
        if (a == null)
            return !0;
        if (x.isArray(a) || x.isString(a))
            return a.length === 0;
        for (var b in a)
            if (x.has(a, b))
                return !1;
        return !0
    }, x.isElement = function(a) {
        return !!a && a.nodeType === 1
    }, x.isArray = u || function(a) {
        return j.call(a) == "[object Array]"
    }, x.isObject = function(a) {
        return a === Object(a)
    }, y(["Arguments", "Function", "String", "Number", "Date", "RegExp"], function(a) {
        x["is" + a] = function(b) {
            return j.call(b) == "[object " + a + "]"
        }
    }), x.isArguments(arguments) || (x.isArguments = function(a) {
        return !!a&&!!x.has(a, "callee")
    }), typeof /./ != "function" && (x.isFunction = function(a) {
        return typeof a == "function"
    }), x.isFinite = function(a) {
        return x.isNumber(a) && isFinite(a)
    }, x.isNaN = function(a) {
        return x.isNumber(a) && a!=+a
    }, x.isBoolean = function(a) {
        return a===!0 || a===!1 || j.call(a) == "[object Boolean]"
    }, x.isNull = function(a) {
        return a === null
    }, x.isUndefined = function(a) {
        return a === void 0
    }, x.has = function(a, b) {
        return k.call(a, b)
    }, x.noConflict = function() {
        return root._ = a, this
    }, x.identity = function(a) {
        return a
    }, x.times = function(a, b, c) {
        for (var d = 0; d < a; d++)
            b.call(c, d)
    }, x.random = function(a, b) {
        return b == null && (b = a, a = 0), a + (0 | Math.random() * (b - a + 1))
    };
    var F = {
        escape: {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;"
        }
    };
    F.unescape = x.invert(F.escape);
    var G = {
        escape: new RegExp("[" + x.keys(F.escape).join("") + "]", "g"),
        unescape: new RegExp("(" + x.keys(F.unescape).join("|") + ")", "g")
    };
    x.each(["escape", "unescape"], function(a) {
        x[a] = function(b) {
            return b == null ? "" : ("" + b).replace(G[a], function(b) {
                return F[a][b]
            })
        }
    }), x.result = function(a, b) {
        if (a == null)
            return null;
        var c = a[b];
        return x.isFunction(c) ? c.call(a) : c
    }, x.mixin = function(a) {
        y(x.functions(a), function(b) {
            var c = x[b] = a[b];
            x.prototype[b] = function() {
                var a = [this._wrapped];
                return f.apply(a, arguments), L.call(this, c.apply(x, a))
            }
        })
    };
    var H = 0;
    x.uniqueId = function(a) {
        var b = H++;
        return a ? a + b : b
    }, x.templateSettings = {
        evaluate: /<%([\s\S]+?)%>/g,
        interpolate: /<%=([\s\S]+?)%>/g,
        escape: /<%-([\s\S]+?)%>/g
    };
    var I = /(.)^/, J = {
        "'": "'",
        "\\": "\\",
        "\r": "r",
        "\n": "n",
        "	": "t",
        "\u2028": "u2028",
        "\u2029": "u2029"
    }, K = /\\|'|\r|\n|\t|\u2028|\u2029/g;
    x.template = function(a, b, c) {
        c = x.defaults({}, c, x.templateSettings);
        var d = new RegExp([(c.escape || I).source, (c.interpolate || I).source, (c.evaluate || I).source].join("|") + "|$", "g"), e = 0, f = "__p+='";
        a.replace(d, function(b, c, d, g, h) {
            f += a.slice(e, h).replace(K, function(a) {
                return "\\" + J[a]
            }), f += c ? "'+\n((__t=(" + c + "))==null?'':_.escape(__t))+\n'" : d ? "'+\n((__t=(" + d + "))==null?'':__t)+\n'" : g ? "';\n" + g + "\n__p+='" : "", e = h + b.length
        }), f += "';\n", c.variable || (f = "with(obj||{}){\n" + f + "}\n"), f = "var __t,__p='',__j=Array.prototype.join,print=function(){__p+=__j.call(arguments,'');};\n" + f + "return __p;\n";
        try {
            var g = new Function(c.variable || "obj", "_", f)
        } catch (h) {
            throw h.source = f, h
        }
        if (b)
            return g(b, x);
        var i = function(a) {
            return g.call(this, a, x)
        };
        return i.source = "function(" + (c.variable || "obj") + "){\n" + f + "}", i
    }, x.chain = function(a) {
        return x(a).chain()
    };
    var L = function(a) {
        return this._chain ? x(a).chain() : a
    };
    return x.mixin(x), y(["pop", "push", "reverse", "shift", "sort", "splice", "unshift"], function(a) {
        var b = c[a];
        x.prototype[a] = function() {
            var c = this._wrapped;
            return b.apply(c, arguments), (a == "shift" || a == "splice") && c.length === 0 && delete c[0], L.call(this, c)
        }
    }), y(["concat", "join", "slice"], function(a) {
        var b = c[a];
        x.prototype[a] = function() {
            return L.call(this, b.apply(this._wrapped, arguments))
        }
    }), x.extend(x.prototype, {
        chain: function() {
            return this._chain=!0, this
        },
        value: function() {
            return this._wrapped
        }
    }), x.noConflict()
}), define("backbone", ["jquery", "underscore"], function(a, b) {
    var c = window, d = c.Backbone, e = Array.prototype, f = e.push, g = e.slice, h = e.splice, i;
    typeof exports != "undefined" ? i = exports : i = c.Backbone = {}, i.VERSION = "0.9.2", i.$ = a, i.noConflict = function() {
        return c.Backbone = d, this
    }, i.emulateHTTP=!1, i.emulateJSON=!1;
    var j = /\s+/, k = i.Events = {
        on: function(a, b, c) {
            var d, e, f;
            if (!b)
                return this;
            a = a.split(j), d = this._callbacks || (this._callbacks = {});
            while (e = a.shift())
                f = d[e] || (d[e] = []), f.push(b, c);
            return this
        },
        off: function(a, c, d) {
            var e, f, g, h;
            if (!(f = this._callbacks))
                return this;
            if (!(a || c || d))
                return delete this._callbacks, this;
            a = a ? a.split(j) : b.keys(f);
            while (e = a.shift()) {
                if (!(g = f[e]) ||!c&&!d) {
                    delete f[e];
                    continue
                }
                for (h = g.length - 2; h >= 0; h -= 2)
                    c && g[h] !== c || d && g[h + 1] !== d || g.splice(h, 2)
            }
            return this
        },
        trigger: function(a) {
            var b, c, d, e, f, g, h, i;
            if (!(c = this._callbacks))
                return this;
            i = [], a = a.split(j);
            for (e = 1, f = arguments.length; e < f; e++)
                i[e - 1] = arguments[e];
            while (b = a.shift()) {
                if (h = c.all)
                    h = h.slice();
                if (d = c[b])
                    d = d.slice();
                if (d)
                    for (e = 0, f = d.length; e < f; e += 2)
                        d[e].apply(d[e + 1] || this, i);
                if (h) {
                    g = [b].concat(i);
                    for (e = 0, f = h.length; e < f; e += 2)
                        h[e].apply(h[e + 1] || this, g)
                    }
            }
            return this
        }
    };
    k.bind = k.on, k.unbind = k.off;
    var l = i.Model = function(a, c) {
        var d, e = a || {};
        c && c.collection && (this.collection = c.collection), c && c.parse && (e = this.parse(e));
        if (d = b.result(this, "defaults"))
            e = b.extend({}, d, e);
        this.attributes = {}, this._escapedAttributes = {}, this.cid = b.uniqueId("c"), this.changed = {}, this._changes = {}, this._pending = {}, this.set(e, {
            silent: !0
        }), this.changed = {}, this._changes = {}, this._pending = {}, this._previousAttributes = b.clone(this.attributes), this.initialize.apply(this, arguments)
    };
    b.extend(l.prototype, k, {
        changed: null,
        _changes: null,
        _pending: null,
        _changing: null,
        idAttribute: "id",
        initialize: function() {},
        toJSON: function(a) {
            return b.clone(this.attributes)
        },
        sync: function() {
            return i.sync.apply(this, arguments)
        },
        get: function(a) {
            return this.attributes[a]
        },
        escape: function(a) {
            var c;
            if (c = this._escapedAttributes[a])
                return c;
            var d = this.get(a);
            return this._escapedAttributes[a] = b.escape(d == null ? "" : "" + d)
        },
        has: function(a) {
            return this.get(a) != null
        },
        set: function(a, c, d) {
            var e, f;
            if (a == null)
                return this;
            b.isObject(a) ? (f = a, d = c) : (f = {})[a] = c;
            var g = d && d.silent, h = d && d.unset;
            f instanceof l && (f = f.attributes);
            if (h)
                for (e in f)
                    f[e] = void 0;
            if (!this._validate(f, d))
                return !1;
            this.idAttribute in f && (this.id = f[this.idAttribute]);
            var i = this._changing, j = this.attributes, k = this._escapedAttributes, m = this._previousAttributes || {};
            for (e in f) {
                c = f[e];
                if (!b.isEqual(j[e], c) || h && b.has(j, e))
                    delete k[e], this._changes[e]=!0;
                h ? delete j[e] : j[e] = c, !b.isEqual(m[e], c) || b.has(j, e) !== b.has(m, e) ? (this.changed[e] = c, g || (this._pending[e]=!0)) : (delete this.changed[e], delete this._pending[e], i || delete this._changes[e]), i && b.isEqual(j[e], i[e]) && delete this._changes[e]
            }
            return g || this.change(d), this
        },
        unset: function(a, c) {
            return c = b.extend({}, c, {
                unset: !0
            }), this.set(a, null, c)
        },
        clear: function(a) {
            return a = b.extend({}, a, {
                unset: !0
            }), this.set(b.clone(this.attributes), a)
        },
        fetch: function(a) {
            a = a ? b.clone(a) : {};
            var c = this, d = a.success;
            return a.success = function(b, e, f) {
                if (!c.set(c.parse(b, f), a))
                    return !1;
                d && d(c, b, a)
            }, this.sync("read", this, a)
        },
        save: function(a, c, d) {
            var e, f, g;
            a == null || b.isObject(a) ? (e = a, d = c) : a != null && ((e = {})[a] = c), d = d ? b.clone(d) : {};
            if (d.wait) {
                if (!this._validate(e, d))
                    return !1;
                f = b.clone(this.attributes)
            }
            var h = b.extend({}, d, {
                silent: !0
            });
            if (e&&!this.set(e, d.wait ? h : d))
                return !1;
            if (!e&&!this._validate(null, d))
                return !1;
            var i = this, j = d.success;
            d.success = function(a, c, f) {
                g=!0;
                var h = i.parse(a, f);
                d.wait && (h = b.extend(e || {}, h));
                if (!i.set(h, d))
                    return !1;
                j && j(i, a, d)
            };
            var k = this.sync(this.isNew() ? "create" : "update", this, d);
            return !g && d.wait && (this.clear(h), this.set(f, h)), k
        },
        destroy: function(a) {
            a = a ? b.clone(a) : {};
            var c = this, d = a.success, e = function() {
                c.trigger("destroy", c, c.collection, a)
            };
            a.success = function(b) {
                (a.wait || c.isNew()) && e(), d && d(c, b, a)
            };
            if (this.isNew())
                return a.success(), !1;
            var f = this.sync("delete", this, a);
            return a.wait || e(), f
        },
        url: function() {
            var a = b.result(this, "urlRoot") || b.result(this.collection, "url") || E();
            return this.isNew() ? a : a + (a.charAt(a.length - 1) === "/" ? "" : "/") + encodeURIComponent(this.id)
        },
        parse: function(a, b) {
            return a
        },
        clone: function() {
            return new this.constructor(this.attributes)
        },
        isNew: function() {
            return this.id == null
        },
        change: function(a) {
            var c = this._changing, d = this._changing = {};
            for (var e in this._changes)
                this._pending[e]=!0;
            var f = this._changes;
            this._changes = {};
            var g = [];
            for (var e in f)
                d[e] = this.get(e), g.push(e);
            for (var h = 0, i = g.length; h < i; h++)
                this.trigger("change:" + g[h], this, d[g[h]], a);
            if (c)
                return this;
            while (!b.isEmpty(this._pending)) {
                this._pending = {}, this.trigger("change", this, a);
                for (var e in this.changed) {
                    if (this._pending[e] || this._changes[e])
                        continue;
                    delete this.changed[e]
                }
                this._previousAttributes = b.clone(this.attributes)
            }
            return this._changing = null, this
        },
        hasChanged: function(a) {
            return a == null?!b.isEmpty(this.changed) : b.has(this.changed, a)
        },
        changedAttributes: function(a) {
            if (!a)
                return this.hasChanged() ? b.clone(this.changed) : !1;
            var c, d=!1, e = this._previousAttributes;
            for (var f in a) {
                if (b.isEqual(e[f], c = a[f]))
                    continue;
                (d || (d = {}))[f] = c
            }
            return d
        },
        previous: function(a) {
            return a == null ||!this._previousAttributes ? null : this._previousAttributes[a]
        },
        previousAttributes: function() {
            return b.clone(this._previousAttributes)
        },
        isValid: function(a) {
            return !this.validate ||!this.validate(this.attributes, a)
        },
        _validate: function(a, c) {
            if (c && c.silent ||!this.validate)
                return !0;
            a = b.extend({}, this.attributes, a);
            var d = this.validate(a, c);
            return d ? (c && c.error && c.error(this, d, c), this.trigger("error", this, d, c), !1) : !0
        }
    });
    var m = i.Collection = function(a, b) {
        b || (b = {}), b.model && (this.model = b.model), b.comparator !== void 0 && (this.comparator = b.comparator), this._reset(), this.initialize.apply(this, arguments), a && (b.parse && (a = this.parse(a)), this.reset(a, {
            silent: !0,
            parse: b.parse
        }))
    };
    b.extend(m.prototype, k, {
        model: l,
        initialize: function() {},
        toJSON: function(a) {
            return this.map(function(b) {
                return b.toJSON(a)
            })
        },
        sync: function() {
            return i.sync.apply(this, arguments)
        },
        add: function(a, c) {
            var d, e, g, i, j, k = c && c.at;
            a = b.isArray(a) ? a.slice() : [a];
            for (d = 0, g = a.length; d < g; d++) {
                if (a[d] = this._prepareModel(a[d], c))
                    continue;
                throw new Error("Can't add an invalid model to a collection")
            }
            for (d = a.length - 1; d >= 0; d--) {
                i = a[d], j = i.id != null && this._byId[i.id];
                if (j || this._byCid[i.cid]) {
                    c && c.merge && j && j.set(i, c), a.splice(d, 1);
                    continue
                }
                i.on("all", this._onModelEvent, this), this._byCid[i.cid] = i, i.id != null && (this._byId[i.id] = i)
            }
            this.length += a.length, e = [k != null ? k: this.models.length, 0], f.apply(e, a), h.apply(this.models, e), this.comparator && k == null && this.sort({
                silent: !0
            });
            if (c && c.silent)
                return this;
            while (i = a.shift())
                i.trigger("add", i, this, c);
            return this
        },
        remove: function(a, c) {
            var d, e, f, g;
            c || (c = {}), a = b.isArray(a) ? a.slice() : [a];
            for (d = 0, e = a.length; d < e; d++) {
                g = this.getByCid(a[d]) || this.get(a[d]);
                if (!g)
                    continue;
                delete this._byId[g.id], delete this._byCid[g.cid], f = this.indexOf(g), this.models.splice(f, 1), this.length--, c.silent || (c.index = f, g.trigger("remove", g, this, c)), this._removeReference(g)
            }
            return this
        },
        push: function(a, b) {
            return a = this._prepareModel(a, b), this.add(a, b), a
        },
        pop: function(a) {
            var b = this.at(this.length - 1);
            return this.remove(b, a), b
        },
        unshift: function(a, c) {
            return a = this._prepareModel(a, c), this.add(a, b.extend({
                at: 0
            }, c)), a
        },
        shift: function(a) {
            var b = this.at(0);
            return this.remove(b, a), b
        },
        slice: function(a, b) {
            return this.models.slice(a, b)
        },
        get: function(a) {
            return a == null ? void 0 : this._byId[a.id != null ? a.id: a]
        },
        getByCid: function(a) {
            return a && this._byCid[a.cid || a]
        },
        at: function(a) {
            return this.models[a]
        },
        where: function(a) {
            return b.isEmpty(a) ? [] : this.filter(function(b) {
                for (var c in a)
                    if (a[c] !== b.get(c))
                        return !1;
                return !0
            })
        },
        sort: function(a) {
            if (!this.comparator)
                throw new Error("Cannot sort a set without a comparator");
            return b.isString(this.comparator) || this.comparator.length === 1 ? this.models = this.sortBy(this.comparator, this) : this.models.sort(b.bind(this.comparator, this)), (!a ||!a.silent) && this.trigger("reset", this, a), this
        },
        pluck: function(a) {
            return b.invoke(this.models, "get", a)
        },
        reset: function(a, c) {
            for (var d = 0, e = this.models.length; d < e; d++)
                this._removeReference(this.models[d]);
            return this._reset(), a && this.add(a, b.extend({
                silent: !0
            }, c)), (!c ||!c.silent) && this.trigger("reset", this, c), this
        },
        fetch: function(a) {
            a = a ? b.clone(a) : {}, a.parse === void 0 && (a.parse=!0);
            var c = this, d = a.success;
            return a.success = function(b, e, f) {
                c[a.add ? "add": "reset"](c.parse(b, f), a), d && d(c, b, a)
            }, this.sync("read", this, a)
        },
        create: function(a, c) {
            var d = this;
            c = c ? b.clone(c) : {}, a = this._prepareModel(a, c);
            if (!a)
                return !1;
            c.wait || d.add(a, c);
            var e = c.success;
            return c.success = function(a, b, c) {
                c.wait && d.add(a, c), e && e(a, b, c)
            }, a.save(null, c), a
        },
        parse: function(a, b) {
            return a
        },
        clone: function() {
            return new this.constructor(this.models)
        },
        chain: function() {
            return b(this.models).chain()
        },
        _reset: function(a) {
            this.length = 0, this.models = [], this._byId = {}, this._byCid = {}
        },
        _prepareModel: function(a, b) {
            if (a instanceof l)
                return a.collection || (a.collection = this), a;
            b || (b = {}), b.collection = this;
            var c = new this.model(a, b);
            return c._validate(c.attributes, b) ? c : !1
        },
        _removeReference: function(a) {
            this === a.collection && delete a.collection, a.off("all", this._onModelEvent, this)
        },
        _onModelEvent: function(a, b, c, d) {
            if ((a === "add" || a === "remove") && c !== this)
                return;
            a === "destroy" && this.remove(b, d), b && a === "change:" + b.idAttribute && (delete this._byId[b.previous(b.idAttribute)], b.id != null && (this._byId[b.id] = b)), this.trigger.apply(this, arguments)
        }
    });
    var n = ["forEach", "each", "map", "collect", "reduce", "foldl", "inject", "reduceRight", "foldr", "find", "detect", "filter", "select", "reject", "every", "all", "some", "any", "include", "contains", "invoke", "max", "min", "sortedIndex", "toArray", "size", "first", "head", "take", "initial", "rest", "tail", "last", "without", "indexOf", "shuffle", "lastIndexOf", "isEmpty"];
    b.each(n, function(a) {
        m.prototype[a] = function() {
            var c = g.call(arguments);
            return c.unshift(this.models), b[a].apply(b, c)
        }
    });
    var o = ["groupBy", "countBy", "sortBy"];
    b.each(o, function(a) {
        m.prototype[a] = function(c, d) {
            var e = b.isFunction(c) ? c: function(a) {
                return a.get(c)
            };
            return b[a](this.models, e, d)
        }
    });
    var p = i.Router = function(a) {
        a || (a = {}), a.routes && (this.routes = a.routes), this._bindRoutes(), this.initialize.apply(this, arguments)
    }, q = /\((.*?)\)/g, r = /:\w+/g, s = /\*\w+/g, t = /[-{}[\]+?.,\\^$|#\s]/g;
    b.extend(p.prototype, k, {
        initialize: function() {},
        route: function(a, c, d) {
            return b.isRegExp(a) || (a = this._routeToRegExp(a)), d || (d = this[c]), i.history.route(a, b.bind(function(b) {
                var e = this._extractParameters(a, b);
                d && d.apply(this, e), this.trigger.apply(this, ["route:" + c].concat(e)), i.history.trigger("route", this, c, e)
            }, this)), this
        },
        navigate: function(a, b) {
            return i.history.navigate(a, b), this
        },
        _bindRoutes: function() {
            if (!this.routes)
                return;
            var a, c = b.keys(this.routes);
            while ((a = c.pop()) != null)
                this.route(a, this.routes[a])
        },
        _routeToRegExp: function(a) {
            return a = a.replace(t, "\\$&").replace(q, "(?:$1)?").replace(r, "([^/]+)").replace(s, "(.*?)"), new RegExp("^" + a + "$")
        },
        _extractParameters: function(a, b) {
            return a.exec(b).slice(1)
        }
    });
    var u = i.History = function() {
        this.handlers = [], b.bindAll(this, "checkUrl"), typeof window != "undefined" && (this.location = window.location, this.history = window.history)
    }, v = /^[#\/]/, w = /^\/+|\/+$/g, x = /msie [\w.]+/, y = /\/$/;
    u.started=!1, b.extend(u.prototype, k, {
        interval: 50,
        getHash: function(a) {
            var b = (a || this).location.href.match(/#(.*)$/);
            return b ? b[1] : ""
        },
        getFragment: function(a, b) {
            if (a == null)
                if (this._hasPushState ||!this._wantsHashChange || b) {
                    a = this.location.pathname;
                    var c = this.root.replace(y, "");
                    a.indexOf(c) || (a = a.substr(c.length))
                } else 
                    a = this.getHash();
            return decodeURIComponent(a.replace(v, ""))
        },
        start: function(a) {
            if (u.started)
                throw new Error("Backbone.history has already been started");
            u.started=!0, this.options = b.extend({}, {
                root: "/"
            }, this.options, a), this.root = this.options.root, this._wantsHashChange = this.options.hashChange!==!1, this._wantsPushState=!!this.options.pushState, this._hasPushState=!!(this.options.pushState && this.history && this.history.pushState);
            var c = this.getFragment(), d = document.documentMode, e = x.exec(navigator.userAgent.toLowerCase()) && (!d || d <= 7);
            this.root = ("/" + this.root + "/").replace(w, "/"), e && this._wantsHashChange && (this.iframe = i.$('<iframe src="javascript:0" tabindex="-1" />').hide().appendTo("body")[0].contentWindow, this.navigate(c)), this._hasPushState ? i.$(window).bind("popstate", this.checkUrl) : this._wantsHashChange && "onhashchange"in window&&!e ? i.$(window).bind("hashchange", this.checkUrl) : this._wantsHashChange && (this._checkUrlInterval = setInterval(this.checkUrl, this.interval)), this.fragment = c;
            var f = this.location, g = f.pathname.replace(/[^\/]$/, "$&/") === this.root;
            if (this._wantsHashChange && this._wantsPushState&&!this._hasPushState&&!g)
                return this.fragment = this.getFragment(null, !0), this.location.replace(this.root + this.location.search + "#" + this.fragment), !0;
            this._wantsPushState && this._hasPushState && g && f.hash && (this.fragment = this.getHash().replace(v, ""), this.history.replaceState({}, document.title, this.root + this.fragment + f.search));
            if (!this.options.silent)
                return this.loadUrl()
        },
        stop: function() {
            i.$(window).unbind("popstate", this.checkUrl).unbind("hashchange", this.checkUrl), clearInterval(this._checkUrlInterval), u.started=!1
        },
        route: function(a, b) {
            this.handlers.unshift({
                route: a,
                callback: b
            })
        },
        checkUrl: function(a) {
            var b = this.getFragment();
            b === this.fragment && this.iframe && (b = this.getFragment(this.getHash(this.iframe)));
            if (b === this.fragment)
                return !1;
            this.iframe && this.navigate(b), this.loadUrl() || this.loadUrl(this.getHash())
        },
        loadUrl: function(a) {
            var c = this.fragment = this.getFragment(a), d = b.any(this.handlers, function(a) {
                if (a.route.test(c))
                    return a.callback(c), !0
            });
            return d
        },
        navigate: function(a, b) {
            if (!u.started)
                return !1;
            if (!b || b===!0)
                b = {
                    trigger: b
                };
            a = this.getFragment(a || "");
            if (this.fragment === a)
                return;
            this.fragment = a;
            var c = this.root + a;
            if (this._hasPushState)
                this.history[b.replace ? "replaceState": "pushState"]({}, document.title, c);
            else {
                if (!this._wantsHashChange)
                    return this.location.assign(c);
                this._updateHash(this.location, a, b.replace), this.iframe && a !== this.getFragment(this.getHash(this.iframe)) && (b.replace || this.iframe.document.open().close(), this._updateHash(this.iframe.location, a, b.replace))
            }
            b.trigger && this.loadUrl(a)
        },
        _updateHash: function(a, b, c) {
            if (c) {
                var d = a.href.replace(/(javascript:|#).*$/, "");
                a.replace(d + "#" + b)
            } else 
                a.hash = "#" + b
        }
    }), i.history = new u;
    var z = i.View = function(a) {
        this.cid = b.uniqueId("view"), this._configure(a || {}), this._ensureElement(), this.initialize.apply(this, arguments), this.delegateEvents()
    }, A = /^(\S+)\s*(.*)$/, B = ["model", "collection", "el", "id", "attributes", "className", "tagName"];
    b.extend(z.prototype, k, {
        tagName: "div",
        $: function(a) {
            return this.$el.find(a)
        },
        initialize: function() {},
        render: function() {
            return this
        },
        dispose: function() {
            return this.undelegateEvents(), this.model && this.model.off && this.model.off(null, null, this), this.collection && this.collection.off && this.collection.off(null, null, this), this
        },
        remove: function() {
            return this.dispose(), this.$el.remove(), this
        },
        make: function(a, b, c) {
            var d = document.createElement(a);
            return b && i.$(d).attr(b), c != null && i.$(d).html(c), d
        },
        setElement: function(a, b) {
            return this.$el && this.undelegateEvents(), this.$el = a instanceof i.$ ? a : i.$(a), this.el = this.$el[0], b!==!1 && this.delegateEvents(), this
        },
        delegateEvents: function(a) {
            if (!a&&!(a = b.result(this, "events")))
                return;
            this.undelegateEvents();
            for (var c in a) {
                var d = a[c];
                b.isFunction(d) || (d = this[a[c]]);
                if (!d)
                    throw new Error('Method "' + a[c] + '" does not exist');
                var e = c.match(A), f = e[1], g = e[2];
                d = b.bind(d, this), f += ".delegateEvents" + this.cid, g === "" ? this.$el.bind(f, d) : this.$el.delegate(g, f, d)
            }
        },
        undelegateEvents: function() {
            this.$el.unbind(".delegateEvents" + this.cid)
        },
        _configure: function(a) {
            this.options && (a = b.extend({}, this.options, a)), b.extend(this, b.pick(a, B)), this.options = a
        },
        _ensureElement: function() {
            if (!this.el) {
                var a = b.extend({}, b.result(this, "attributes"));
                this.id && (a.id = b.result(this, "id")), this.className && (a["class"] = b.result(this, "className")), this.setElement(this.make(b.result(this, "tagName"), a), !1)
            } else 
                this.setElement(b.result(this, "el"), !1)
        }
    });
    var C = {
        create: "POST",
        update: "PUT",
        "delete": "DELETE",
        read: "GET"
    };
    i.sync = function(a, c, d) {
        var e = C[a];
        d || (d = {});
        var f = {
            type: e,
            dataType: "json"
        };
        d.url || (f.url = b.result(c, "url") || E()), !d.data && c && (a === "create" || a === "update") && (f.contentType = "application/json", f.data = JSON.stringify(c)), i.emulateJSON && (f.contentType = "application/x-www-form-urlencoded", f.data = f.data ? {
            model: f.data
        } : {});
        if (i.emulateHTTP && (e === "PUT" || e === "DELETE")) {
            f.type = "POST", i.emulateJSON && (f.data._method = e);
            var g = d.beforeSend;
            d.beforeSend = function(a) {
                a.setRequestHeader("X-HTTP-Method-Override", e);
                if (g)
                    return g.apply(this, arguments)
            }
        }
        f.type !== "GET"&&!i.emulateJSON && (f.processData=!1);
        var h = d.success;
        d.success = function(a, b, e) {
            h && h(a, b, e), c.trigger("sync", c, a, d)
        };
        var j = d.error;
        return d.error = function(a, b, e) {
            j && j(c, a, d), c.trigger("error", c, a, d)
        }, i.ajax(b.extend(f, d))
    }, i.ajax = function() {
        return i.$.ajax.apply(i.$, arguments)
    };
    var D = function(a, c) {
        var d = this, e;
        a && b.has(a, "constructor") ? e = a.constructor : e = function() {
            d.apply(this, arguments)
        }, b.extend(e, d, c);
        var f = function() {
            this.constructor = e
        };
        return f.prototype = d.prototype, e.prototype = new f, a && b.extend(e.prototype, a), e.__super__ = d.prototype, e
    };
    l.extend = m.extend = p.extend = z.extend = u.extend = D;
    var E = function() {
        throw new Error('A "url" property or function must be specified')
    };
    return i
}), define("webs/props", [], function() {
    var a = window.webs && window.webs.props;
    if (!a)
        try {
            typeof top.webs.props != "undefined" ? a = top.webs.props : a = {}
    } catch (b) {
        a = {}
    }
    return a.imageProcessorServer = a.imageProcessorServer || "http://imageprocessor.websimages.com", a
}), define("localize", [], function() {
    function a(a) {
        return a.replace(/^\s+|\s+$/g, "")
    }
    function b(b, c) {
        var d = a(b).split(/\s+/), e = d[0];
        if (typeof c[e] == "function")
            return c[e].call(c, d, c)
    }
    function c(a, b) {
        return b[a]
    }
    function d(b, c) {
        return b.replace(/{([^}]*)}/g, function(b, d) {
            return e.localize(a(d), c)
        })
    }
    var e = {
        localize: function(e, f) {
            if (typeof location != "undefined" && location.search && location.search.indexOf("notranslate=1") > 0)
                return e;
            var g = c(e, f);
            return typeof g != "undefined" ? d(String(g), f) : (g = b(e, f), g ? d(g, f) : (this.logMissingLocalization(e, f), g = e, g))
        },
        logMissingLocalization: function(b, c) {
            typeof console != "undefined" && console.warn && console.warn("No localized string for", b, c.name), typeof mpq != "undefined" && mpq.push && mpq.push(["track", "localizationMissing", {
                key: b,
                locale: c.name
            }
            ])
        }
    };
    return e
}), define("locale", ["localize"], function(a) {
    function b(a, b) {
        function c() {}
        c.prototype = a;
        var d = new c;
        for (var e in b)
            d[e] = b[e];
        return d
    }
    function e(a, e) {
        typeof e == "undefined" && (e = 1), d[a] = b(c, {
            pluralType: "" + e,
            name: a
        })
    }
    function f(a) {
        typeof console != "undefined" && console.warn && console.warn("No such locale", a), typeof mpq != "undefined" && mpq.push && mpq.push(["track", "localeMissing", {
            locale: a
        }
        ])
    }
    var c = {
        name: "base",
        pluralize: function(b, c) {
            var d = a.localize(b[2], c), e = c.pluralTypes[c.pluralType](d), f = b[1] + "." + e;
            return a.localize(f, c)
        },
        pluralTypes: {
            1: function(a) {
                return a == "1" ? "1" : "other"
            },
            2: function(a) {
                return a == "0" || a == "1" ? "1" : "other"
            }
        },
        formatDate: function(a, b) {
            var c = b[a[1]];
            c = new Date(c);
            if (c instanceof Date) {
                var d = c.getDate(), e = c.getMonth() + 1, f = c.getFullYear();
                return e + "/" + d + "/" + f
            }
            return ""
        }
    }, d = {};
    return e("da-DK", 1), e("de-DE", 1), e("en-GB", 1), e("en-AU", 1), e("en-US", 1), e("es-ES", 1), e("es-MX", 1), e("es-US", 1), e("fr-CA", 2), e("fr-FR", 2), e("it-IT", 1), e("nb-NO", 1), e("nl-NL", 1), e("sv-SE", 1), e("zz-ZZ", 1), function(a) {
        var e = d[a];
        return e ? e : (f(a), b(c, {}))
    }
}), define("translate", ["localize", "locale", "jquery"], function(a, b, c) {
    function m(a, b) {
        function c() {}
        c.prototype = a;
        var d = new c;
        for (var e in b)
            d[e] = b[e];
        return d
    }
    if (typeof window != "undefined" && window.parent && window.parent !== window)
        try {
            return window.parent.require("translate")
    } catch (d) {}
    var e = {}, f = typeof webs != "undefined" && webs.locale || "en-US", g, h = b(f), i = "none";
    g = "/s/resources/" + f + "/", typeof webs != "undefined" && webs.props && webs.props.dynamicAssetServer && (g = webs.props.dynamicAssetServer + g);
    var j = function(a) {
        var b = a.split(".");
        for (var c = 0; c < b.length; c++) {
            var d = e[b.slice(0, c + 1).join(".")];
            if (d)
                return d
        }
        return !1
    }, k = function(a) {
        var b = j(a);
        if (b)
            return b;
        var d = g + a + "/", f = c.Deferred(), h = f.promise();
        return require([d + "?callback=define"], function(a) {
            l(a), f.resolve(a)
        }), e[a] = h, h
    }, l = function(a) {
        for (var b in a)
            a.hasOwnProperty(b) && (h[b] = a[b])
    }, n = function(b, c) {
        var d = m(h, c);
        return a.localize(b, d)
    };
    try {
        require.s.contexts._.specified.prefetchTranslationNamespaces && require(["prefetchTranslationNamespaces"], function(a) {
            c.each(a, function(a, b) {
                k(b)
            })
        })
    } catch (d) {}
    return {
        load: function(a, b, c, d) {
            a = String(a).toString(), d.isBuild ? c() : a === i ? c(n) : k(a).then(function() {
                c(n)
            })
        },
        translate: n,
        add: l
    }
}), define("ui/inlineEdit", ["translate!webs.validations", "jquery"], function(a, b) {
    var c = {
        setAttribute: function(a) {
            if (a.type === "keydown" && a.keyCode !== 13 || a.type === "click" && this.tagName === "INPUT")
                return;
            a.type === "click" && a.preventDefault();
            var d = b(a.target).closest(".editable"), e = d.find('[data-editable="true"]'), f = {}, g;
            c.clearErrors(d), e.each(function() {
                var a = b(this).attr("name");
                f[a] = b(this).val()
            }), g = b.Event("inline:submit", {
                inlineData: f
            }), d.trigger(g), c.handleSave(g)
        },
        showEditMode: function(a) {
            a.preventDefault(), b(this).closest(".editable").addClass("editing").find('[data-editable="true"]:first').focus()
        },
        leaveEditMode: function(a) {
            var d = b(a.target).closest(".editing").removeClass("editing");
            c.clearErrors(d);
            var e = b.Event("inline:cancel", {});
            d.trigger(e)
        },
        setErrorTarget: function(a) {
            typeof a.targetInput != "undefined" && b(a.target).closest(".editing").data("errorTarget", a.targetInput)
        },
        getErrorTarget: function(a) {
            return a.data("errorTarget") || a
        },
        clearErrors: function(a) {
            c.getErrorTarget(a).removeClass("error").find(".hint").remove()
        },
        handleSave: function(d) {
            d.isDefaultPrevented() ? typeof d.targetInput != "undefined" ? (c.setErrorTarget(d), d.targetInput.addClass("error").append('<span class="hint">' + a("webs.validations." + d.errors[0]) + "</span>")) : b(d.target).addClass("error").append('<span class="hint">' + a("webs.validations." + d.errors[0]) + "</span>") : c.leaveEditMode(d)
        }
    };
    return b("body").on("click", ".editable .set", c.setAttribute), b("body").on("click", ".editable .edit", c.showEditMode), b("body").on("click", ".editable .cancel", c.leaveEditMode), b("body").on("click", ".editable .rename", c.showEditMode), b("body").on("keydown", '.editable [data-editable="true"]', c.setAttribute), c
}), define("ui/contextMenu", ["jquery"], function(a) {
    var b = {
        toggle: function(b) {
            b.preventDefault(), b.stopPropagation();
            var c = a(b.target), d = c.hasClass("showMenu"), e = c.find(".w-menu");
            a(".settings").removeClass("showMenu"), d || (e.removeClass("above below"), c.offset().top + c.height() + 10 >= a("body").height() - e.height() ? e.addClass("above") : e.addClass("below"), c.addClass("showMenu")), top.$("body").off("permissionSubmit, hideSubmit"), top.$("body").on("permissionSubmit", {
                $el: a(this)
            }, function(a, b) {
                b !== "0" ? a.data.$el.closest("li").trigger("page:change:setProtected") : a.data.$el.closest("li").trigger("page:change:unsetProtected")
            }), top.$("body").on("hideSubmit", {
                $el: a(this)
            }, function(a, b) {
                a.data.$el.closest("li").trigger("page:change:hidden", {
                    event: a,
                    isHidden: b
                })
            })
        },
        hide: function(b) {
            b.preventDefault(), b.stopPropagation(), a(".settings").removeClass("showMenu")
        },
        showDialog: function(b) {
            var c = a(b.target);
            (new top.Popover(c.attr("href"), {
                width: c.data("width"),
                height: c.data("height"),
                heading: c.text()
            })).show()
        }
    };
    return a("body").on("click", ".settings", b.toggle), a("body").on("click", ".w-menu", b.hide), a("body").on("blur", ".w-menu", b.hide), a("body").on("click", function() {
        a(".w-menu").blur()
    }), a("body").on("click", '[data-role="w-menu-option"]', b.showDialog), b
}), define("dust-core", [], function() {
    function Context(a, b, c, d) {
        this.stack = a, this.global = b, this.blocks = c, this.templateName = d
    }
    function Stack(a, b, c, d) {
        this.tail = b, this.isObject = a && typeof a == "object", this.head = a, this.index = c, this.of = d
    }
    function Stub(a) {
        this.head = new Chunk(this), this.callback = a, this.out = ""
    }
    function Stream() {
        this.head = new Chunk(this)
    }
    function Chunk(a, b, c) {
        this.root = a, this.next = b, this.data = [], this.flushable=!1, this.taps = c
    }
    function Tap(a, b) {
        this.head = a, this.tail = b
    }
    var root = window, dust = {}, NONE = "NONE", ERROR = "ERROR", WARN = "WARN", INFO = "INFO", DEBUG = "DEBUG", loggingLevels = [DEBUG, INFO, WARN, ERROR, NONE], EMPTY_FUNC = function() {}, logger = {}, originalLog, loggerContext;
    dust.debugLevel = NONE, dust.silenceErrors=!1, root && root.console && root.console.log && (loggerContext = root.console, originalLog = root.console.log), logger.log = loggerContext ? function() {
        typeof originalLog == "function" ? logger.log = function() {
            originalLog.apply(loggerContext, arguments)
        } : logger.log = function() {
            var a = Array.prototype.slice.apply(arguments).join(" ");
            originalLog(a)
        }, logger.log.apply(this, arguments)
    } : function() {}, dust.log = function(a, b) {
        dust.isDebug && dust.debugLevel === NONE && (logger.log('[!!!DEPRECATION WARNING!!!]: dust.isDebug is deprecated.  Set dust.debugLevel instead to the level of logging you want ["debug","info","warn","error","none"]'), dust.debugLevel = INFO), b = b || INFO, dust.indexInArray(loggingLevels, b) >= dust.indexInArray(loggingLevels, dust.debugLevel) && (dust.logQueue || (dust.logQueue = []), dust.logQueue.push({
            message: a,
            type: b
        }), logger.log("[DUST " + b + "]: " + a));
        if (!dust.silenceErrors && b === ERROR)
            throw typeof a == "string" ? new Error(a) : a
    }, dust.onError = function(a, b) {
        logger.log("[!!!DEPRECATION WARNING!!!]: dust.onError will no longer return a chunk object."), dust.log(a.message || a, ERROR);
        if (!dust.silenceErrors)
            throw a;
        return b
    }, dust.helpers = {}, dust.cache = {}, dust.register = function(a, b) {
        if (!a)
            return;
        dust.cache[a] = b
    }, dust.render = function(a, b, c) {
        var d = (new Stub(c)).head;
        try {
            dust.load(a, d, Context.wrap(b, a)).end()
        } catch (e) {
            dust.log(e, ERROR)
        }
    }, dust.stream = function(a, b) {
        var c = new Stream;
        return dust.nextTick(function() {
            try {
                dust.load(a, c.head, Context.wrap(b, a)).end()
            } catch (d) {
                dust.log(d, ERROR)
            }
        }), c
    }, dust.renderSource = function(a, b, c) {
        return dust.compileFn(a)(b, c)
    }, dust.compileFn = function(a, b) {
        b = b || null;
        var c = dust.loadSource(dust.compile(a, b));
        return function(a, d) {
            var e = d ? new Stub(d): new Stream;
            return dust.nextTick(function() {
                typeof c == "function" ? c(e.head, Context.wrap(a, b)).end() : dust.log(new Error("Template [" + b + "] cannot be resolved to a Dust function"), ERROR)
            }), e
        }
    }, dust.load = function(a, b, c) {
        var d = dust.cache[a];
        return d ? d(b, c) : dust.onLoad ? b.map(function(b) {
            dust.onLoad(a, function(d, e) {
                if (d)
                    return b.setError(d);
                dust.cache[a] || dust.loadSource(dust.compile(e, a)), dust.cache[a](b, c).end()
            })
        }) : b.setError(new Error("Template Not Found: " + a))
    }, dust.loadSource = function(source, path) {
        return eval(source)
    }, Array.isArray ? dust.isArray = Array.isArray : dust.isArray = function(a) {
        return Object.prototype.toString.call(a) === "[object Array]"
    }, dust.indexInArray = function(a, b, c) {
        c =+ c || 0;
        if (Array.prototype.indexOf)
            return a.indexOf(b, c);
        if (a === undefined || a === null)
            throw new TypeError('cannot call method "indexOf" of null');
        var d = a.length;
        Math.abs(c) === Infinity && (c = 0), c < 0 && (c += d, c < 0 && (c = 0));
        for (; c < d; c++)
            if (a[c] === b)
                return c;
        return - 1
    }, dust.nextTick = function() {
        return function(a) {
            setTimeout(a, 0)
        }
    }(), dust.isEmpty = function(a) {
        return dust.isArray(a)&&!a.length?!0 : a === 0?!1 : !a
    }, dust.filter = function(a, b, c) {
        if (c)
            for (var d = 0, e = c.length; d < e; d++) {
                var f = c[d];
                f === "s" ? (b = null, dust.log("Using unescape filter on [" + a + "]", DEBUG)) : typeof dust.filters[f] == "function" ? a = dust.filters[f](a) : dust.log("Invalid filter [" + f + "]", WARN)
            }
        return b && (a = dust.filters[b](a)), a
    }, dust.filters = {
        h: function(a) {
            return dust.escapeHtml(a)
        },
        j: function(a) {
            return dust.escapeJs(a)
        },
        u: encodeURI,
        uc: encodeURIComponent,
        js: function(a) {
            return JSON ? JSON.stringify(a) : (dust.log("JSON is undefined.  JSON stringify has not been used on [" + a + "]", WARN), a)
        },
        jp: function(a) {
            return JSON ? JSON.parse(a) : (dust.log("JSON is undefined.  JSON parse has not been used on [" + a + "]", WARN), a)
        }
    }, dust.makeBase = function(a) {
        return new Context(new Stack, a)
    }, Context.wrap = function(a, b) {
        return a instanceof Context ? a : new Context(new Stack(a), {}, null, b)
    }, Context.prototype.get = function(a, b) {
        return typeof a == "string" && (a[0] === "." && (b=!0, a = a.substr(1)), a = a.split(".")), this._get(b, a)
    }, Context.prototype._get = function(a, b) {
        var c = this.stack, d = 1, e, f, g, h;
        dust.log("Searching for reference [{" + b.join(".") + "}] in template [" + this.getTemplateName() + "]", DEBUG), f = b[0], g = b.length;
        if (a && g === 0)
            h = c, c = c.head;
        else {
            if (!a) {
                while (c) {
                    if (c.isObject) {
                        h = c.head, e = c.head[f];
                        if (e !== undefined)
                            break
                    }
                    c = c.tail
                }
                e !== undefined ? c = e : c = this.global ? this.global[f] : undefined
            } else 
                c && (c.head ? c = c.head[f] : c = undefined);
            while (c && d < g)
                h = c, c = c[b[d]], d++
        }
        if (typeof c == "function") {
            var i = function() {
                try {
                    return c.apply(h, arguments)
                } catch (a) {
                    return dust.log(a, ERROR)
                }
            };
            return i.isFunction=!0, i
        }
        return c === undefined && dust.log("Cannot find the value for reference [{" + b.join(".") + "}] in template [" + this.getTemplateName() + "]"), c
    }, Context.prototype.getPath = function(a, b) {
        return this._get(a, b)
    }, Context.prototype.push = function(a, b, c) {
        return new Context(new Stack(a, this.stack, b, c), this.global, this.blocks, this.getTemplateName())
    }, Context.prototype.rebase = function(a) {
        return new Context(new Stack(a), this.global, this.blocks, this.getTemplateName())
    }, Context.prototype.current = function() {
        return this.stack.head
    }, Context.prototype.getBlock = function(a, b, c) {
        if (typeof a == "function") {
            var d = new Chunk;
            a = a(d, this).data.join("")
        }
        var e = this.blocks;
        if (!e) {
            dust.log("No blocks for context[{" + a + "}] in template [" + this.getTemplateName() + "]", DEBUG);
            return 
        }
        var f = e.length, g;
        while (f--) {
            g = e[f][a];
            if (g)
                return g
        }
    }, Context.prototype.shiftBlocks = function(a) {
        var b = this.blocks, c;
        return a ? (b ? c = b.concat([a]) : c = [a], new Context(this.stack, this.global, c, this.getTemplateName())) : this
    }, Context.prototype.getTemplateName = function() {
        return this.templateName
    }, Stub.prototype.flush = function() {
        var a = this.head;
        while (a) {
            if (!a.flushable) {
                if (a.error) {
                    this.callback(a.error), dust.log("Chunk error [" + a.error + "] thrown. Ceasing to render this template.", WARN), this.flush = EMPTY_FUNC;
                    return 
                }
                return 
            }
            this.out += a.data.join(""), a = a.next, this.head = a
        }
        this.callback(null, this.out)
    }, Stream.prototype.flush = function() {
        var a = this.head;
        while (a) {
            if (!a.flushable) {
                if (a.error) {
                    this.emit("error", a.error), dust.log("Chunk error [" + a.error + "] thrown. Ceasing to render this template.", WARN), this.flush = EMPTY_FUNC;
                    return 
                }
                return 
            }
            this.emit("data", a.data.join("")), a = a.next, this.head = a
        }
        this.emit("end")
    }, Stream.prototype.emit = function(a, b) {
        if (!this.events)
            return dust.log("No events to emit", INFO), !1;
        var c = this.events[a];
        if (!c)
            return dust.log("Event type [" + a + "] does not exist", WARN), !1;
        if (typeof c == "function")
            c(b);
        else if (dust.isArray(c)) {
            var d = c.slice(0);
            for (var e = 0, f = d.length; e < f; e++)
                d[e](b)
        } else 
            dust.log("Event Handler [" + c + "] is not of a type that is handled by emit", WARN)
    }, Stream.prototype.on = function(a, b) {
        return this.events || (this.events = {}), this.events[a] ? typeof this.events[a] == "function" ? this.events[a] = [this.events[a], b] : this.events[a].push(b) : (dust.log("Event type [" + a + "] does not exist. Using just the specified callback.", WARN), b ? this.events[a] = b : dust.log("Callback for type [" + a + "] does not exist. Listener not registered.", WARN)), this
    }, Stream.prototype.pipe = function(a) {
        return this.on("data", function(b) {
            try {
                a.write(b, "utf8")
            } catch (c) {
                dust.log(c, ERROR)
            }
        }).on("end", function() {
            try {
                return a.end()
            } catch (b) {
                dust.log(b, ERROR)
            }
        }).on("error", function(b) {
            a.error(b)
        }), this
    }, Chunk.prototype.write = function(a) {
        var b = this.taps;
        return b && (a = b.go(a)), this.data.push(a), this
    }, Chunk.prototype.end = function(a) {
        return a && this.write(a), this.flushable=!0, this.root.flush(), this
    }, Chunk.prototype.map = function(a) {
        var b = new Chunk(this.root, this.next, this.taps), c = new Chunk(this.root, b, this.taps);
        return this.next = c, this.flushable=!0, a(c), b
    }, Chunk.prototype.tap = function(a) {
        var b = this.taps;
        return b ? this.taps = b.push(a) : this.taps = new Tap(a), this
    }, Chunk.prototype.untap = function() {
        return this.taps = this.taps.tail, this
    }, Chunk.prototype.render = function(a, b) {
        return a(this, b)
    }, Chunk.prototype.reference = function(a, b, c, d) {
        if (typeof a == "function") {
            a.isFunction=!0, a = a.apply(b.current(), [this, b, null, {
                auto: c,
                filters: d
            }
            ]);
            if (a instanceof Chunk)
                return a
        }
        return dust.isEmpty(a) ? this : this.write(dust.filter(a, c, d))
    }, Chunk.prototype.section = function(a, b, c, d) {
        if (typeof a == "function") {
            a = a.apply(b.current(), [this, b, c, d]);
            if (a instanceof Chunk)
                return a
        }
        var e = c.block, f = c["else"];
        d && (b = b.push(d));
        if (dust.isArray(a)) {
            if (e) {
                var g = a.length, h = this;
                if (g > 0) {
                    b.stack.head && (b.stack.head.$len = g);
                    for (var i = 0; i < g; i++)
                        b.stack.head && (b.stack.head.$idx = i), h = e(h, b.push(a[i], i, g));
                    return b.stack.head && (b.stack.head.$idx = undefined, b.stack.head.$len = undefined), h
                }
                if (f)
                    return f(this, b)
                }
        } else if (a===!0) {
            if (e)
                return e(this, b)
        } else if (a || a === 0) {
            if (e)
                return e(this, b.push(a))
        } else if (f)
            return f(this, b);
        return dust.log("Not rendering section (#) block in template [" + b.getTemplateName() + "], because above key was not found", DEBUG), this
    }, Chunk.prototype.exists = function(a, b, c) {
        var d = c.block, e = c["else"];
        if (!dust.isEmpty(a)) {
            if (d)
                return d(this, b)
        } else if (e)
            return e(this, b);
        return dust.log("Not rendering exists (?) block in template [" + b.getTemplateName() + "], because above key was not found", DEBUG), this
    }, Chunk.prototype.notexists = function(a, b, c) {
        var d = c.block, e = c["else"];
        if (dust.isEmpty(a)) {
            if (d)
                return d(this, b)
        } else if (e)
            return e(this, b);
        return dust.log("Not rendering not exists (^) block check in template [" + b.getTemplateName() + "], because above key was found", DEBUG), this
    }, Chunk.prototype.block = function(a, b, c) {
        var d = c.block;
        return a && (d = a), d ? d(this, b) : this
    }, Chunk.prototype.partial = function(a, b, c) {
        var d;
        d = dust.makeBase(b.global), d.blocks = b.blocks, b.stack && b.stack.tail && (d.stack = b.stack.tail), c && (d = d.push(c)), typeof a == "string" && (d.templateName = a), d = d.push(b.stack.head);
        var e;
        return typeof a == "function" ? e = this.capture(a, d, function(a, b) {
            d.templateName = d.templateName || a, dust.load(a, b, d).end()
        }) : e = dust.load(a, this, d), e
    }, Chunk.prototype.helper = function(a, b, c, d) {
        var e = this;
        try {
            return dust.helpers[a] ? dust.helpers[a](e, b, c, d) : (dust.log("Invalid helper [" + a + "]", WARN), e)
        } catch (f) {
            return dust.log(f, ERROR), e
        }
    }, Chunk.prototype.capture = function(a, b, c) {
        return this.map(function(d) {
            var e = new Stub(function(a, b) {
                a ? d.setError(a) : c(b, d)
            });
            a(e.head, b).end()
        })
    }, Chunk.prototype.setError = function(a) {
        return this.error = a, this.root.flush(), this
    }, Tap.prototype.push = function(a) {
        return new Tap(a, this)
    }, Tap.prototype.go = function(a) {
        var b = this;
        while (b)
            a = b.head(a), b = b.tail;
        return a
    };
    var HCHARS = new RegExp(/[&<>\"\']/), AMP = /&/g, LT = /</g, GT = />/g, QUOT = /\"/g, SQUOT = /\'/g;
    dust.escapeHtml = function(a) {
        return typeof a == "string" ? HCHARS.test(a) ? a.replace(AMP, "&amp;").replace(LT, "&lt;").replace(GT, "&gt;").replace(QUOT, "&quot;").replace(SQUOT, "&#39;") : a : a
    };
    var BS = /\\/g, FS = /\//g, CR = /\r/g, LS = /\u2028/g, PS = /\u2029/g, NL = /\n/g, LF = /\f/g, SQ = /'/g, DQ = /"/g, TB = /\t/g;
    dust.escapeJs = function(a) {
        return typeof a == "string" ? a.replace(BS, "\\\\").replace(FS, "\\/").replace(DQ, '\\"').replace(SQ, "\\'").replace(CR, "\\r").replace(LS, "\\u2028").replace(PS, "\\u2029").replace(NL, "\\n").replace(LF, "\\f").replace(TB, "\\t") : a
    }, typeof exports == "object" ? module.exports = dust : root.dust = dust
}), define("dust-helpers", ["webs/props", "translate!none", "dust-core"], function(a, b) {
    function c(a, b, c) {
        if (a)
            switch (b || typeof a) {
            case"number":
                return + a;
            case"string":
                return String(a);
            case"boolean":
                return a = a === "false"?!1 : a, Boolean(a);
            case"date":
                return new Date(a);
            case"context":
                return c.get(a)
            }
        return a
    }
    var d = dust.makeBase({
        websProp: function(b, c, d, e) {
            return b.write(a[e.name])
        },
        l: function(a, c, d, e) {
            e = e || {};
            var f = a.data.length;
            d.block(a, c);
            var g = a.data.splice(f, a.data.length - f), h = b(g.join(""), e);
            return a.write(h)
        },
        ne: function(a, b, d, e) {
            if (e && c(e.key, e.type, b) !== c(e.value, e.type, b)) {
                var f = d.block;
                return f ? a.render(f, b) : (console.log("Missing body block in the 'ne' helper "), a)
            }
            return a
        },
        eq: function(a, b, d, e) {
            if (e && c(e.key, e.type, b) === c(e.value, e.type, b)) {
                var f = d.block;
                return f ? a.render(f, b) : (console.log("Missing body block in the 'eq' helper "), a)
            }
            return a
        },
        gt: function(a, b, d, e) {
            if (e && c(e.key, e.type, b) > c(e.value, e.type, b)) {
                var f = d.block;
                return f ? a.render(f, b) : (console.log("Missing body block in the 'gt' helper "), a)
            }
            return a
        },
        sep: function(a, b, c) {
            var d = c.block;
            return b.stack.index === b.stack.of - 1 ? a : d ? c.block(a, b) : a
        }
    });
    return d
}), function(a) {
    function f(a, b) {
        return a.lastIndexOf(".") <= a.lastIndexOf("/") ? a + "." + b : a
    }
    function g(a, c) {
        var d = a[b]("link");
        return d.rel = "stylesheet", d.type = "text/css", d.href = c, d
    }
    function h(a, b) {
        return a.replace(d, b + "//")
    }
    var b = "createElement", c = a.document, d = /^\/\//, e;
    c && (e = c.head || (c.head = c.getElementsByTagName("head")[0])), define("link", {
        load: function(a, b, d, i) {
            var j, k, l;
            j = b.toUrl(f(a, "css")), l = "fixSchemalessUrls"in i ? i.fixSchemalessUrls : c && c.location.protocol, j = l ? h(j, l) : j, c ? (k = g(c, j), e.appendChild(k), d(k.sheet || k.styleSheet)) : d(j)
        }
    })
}(this), define("internal/common/tooltip", ["jquery", "link"], function(a) {
    return require(["link!../../static/projects/finch/css/tooltip"], function() {}), a.tooltip = a.fn.tooltip = function b(c) {
        if (this.jquery) {
            var d = a([]);
            return this.each(function(e, f) {
                d.push(b(a.extend({}, c, {
                    anchor: f
                }))[0])
            }), d
        }
        c = a.extend({}, {
            content: "&nbsp;",
            style: "info",
            position: "southeast",
            offset: [5, 5],
            anchor: "cursor"
        }, c);
        if (typeof c.anchor == "string" && c.anchor !== "cursor")
            return a(c.anchor).tooltip(c);
        c.anchor.nodeType && (c.anchor = a(c.anchor));
        var e = a("<div></div>").addClass("w-tooltip").html(c.content).appendTo(a("body")), f = a('<div class="tooltip-tip"><span class="tip">&nbsp;</span><span class="overlay">&nbsp;</span></div>');
        c.style && e.addClass("w-tooltip-" + c.style), c.hideicon && e.addClass("w-tooltip-noicon"), c.anchor === "cursor" ? (e.css({
            position: "fixed"
        }), a("body").bind("mousemove", function(b) {
            if (!e.hasClass("active"))
                return;
            var d = parseInt(b.clientX + e.width(), 10), f = c.position;
            d > a("body").width() - e.width() * 1.5 && (f = "southwest"), f.match(/(top|north)/i) ? e.css({
                bottom: a(window).height() - b.clientY + c.offset[1] + "px",
                top: ""
            }) : e.css({
                top: b.clientY + c.offset[1] + "px",
                bottom: ""
            }), f.match(/(left|west)/i) ? e.css({
                right: a(window).width() - b.clientX + c.offset[0] + "px",
                left: ""
            }) : e.css({
                left: b.clientX + c.offset[0] + "px",
                right: ""
            })
        })) : c.showTip && (e.append(f), c.tipOffset = c.tipOffset || [0, 0]), e.reposition = function() {
            var d = 0, f = 0;
            if (c.anchor === "cursor")
                return;
            if (c.anchor && c.anchor.constructor === Array)
                e.css({
                    position: "absolute"
                }), c.position.match(/(top|north)/i) ? e.css("bottom", a("body").height() - c.anchor[1] + c.offset[1] + "px") : e.css("top", c.anchor[1] + c.offset[1] + "px"), c.position.match(/(left|west)/i) ? e.css("right", a("body").width() - c.anchor[0] + c.offset[0] + "px") : e.css("left", c.anchor[0] + c.offset[0] + "px"), c.showTip && g(0, c);
            else {
                if (!c.anchor ||!c.anchor.jquery)
                    throw ["Unsupported tooltip definition", c];
                e.css({
                    position: "absolute"
                });
                var h = c.anchor.offset();
                c.position.match(/(top|north)/i) ? e.css("bottom", a("body").height() - h.top + c.anchor.outerHeight() + c.offset[1] + "px") : e.css("top", h.top + c.anchor.outerHeight() + c.offset[1] + "px");
                if (c.position.match(/(left|west)/i))
                    e.css("right", a("body").width() - h.left + c.offset[0] + "px");
                else if (c.position.match(/center/i))
                    e.css("left", h.left - e.outerWidth() / 2 + c.offset[0] + "px");
                else {
                    var i = Math.abs(parseInt(e.css("margin-left"), 10)), j = a("body").width() - i;
                    d = h.left + c.anchor.outerWidth() + c.offset[0], f = d, d < i ? d = i : d > j && (d = j), e.css("left", d + "px")
                }
                c.showTip && g(d - f, c)
            }
        }, e.updateOffset = function(b) {
            c.offset !== b && (c.offset = b, e.reposition())
        };
        var g = function(a, b) {
            var c = e.width(), d = parseInt(e.css("padding-left"), 10), g = b.anchor.offset().left, h = parseInt(e.css("left"), 10), i = b.anchor.width(), j = g - h + i / 4;
            j > c + d && (j = c + d), f.css("left", j + "px")
        }, h = function() {
            e.reposition()
        };
        return a(window).resize(h), e.reposition(), e
    }, a.tooltip
}), define("nodeDataTooltip", ["jquery", "underscore", "internal/common/tooltip"], function(b, c) {
    var d = [], e = 0, f, g = {
        content: "",
        style: "mini"
    }, h = function(a) {
        var d = c.reduce(a.parents(), function(a, d) {
            return c.extend(a, b(d).data())
        }, a.data()), e = c.reduce(c.keys(d), function(a, b) {
            return b.substr(0, 7) === "tooltip" && (a[b.substr(7).toLowerCase()] = d[b]), a
        }, c.clone(g));
        if (e.offsetleft || e.offsettop)
            e.offset = [parseInt(e.offsetleft, 10), parseInt(e.offsettop, 10)];
        return e.anchor === "this" && (e.anchor = a), e
    }, i = function(a) {
        j(), a.data("existingTooltip") ? f = d[a.data("existingTooltip")] : (f = b.tooltip(h(a)), f.prepend(a.data("tooltip")), a.data("tooltip-title") && f.prepend("<h6>" + a.data("tooltip-title") + "</h6>"), d[e] = f, a.data({
            existingTooltip: e
        }), e++), f.addClass("active"), f.show()
    }, j = function() {
        f && (f.removeClass("active"), f.hide())
    };
    b(function() {
        b("body").on("mouseover", "*[data-tooltip]", function() {
            i(b(this))
        }).on("mouseout", "*[data-tooltip]", function() {
            j()
        }), b("body").on("showTooltip", function(a, c) {
            i(b(c))
        }).on("hideTooltip", function(a, b) {
            j()
        })
    })
}), define("tracking/trackingHub", ["underscore", "backbone"], function(a, b) {
    var c = a.extend({
        track: function(a, b) {
            b.url && (b.url = this.sanitizeUrl(b.url)), this.trigger(a, b)
        },
        parsePathname: function(a) {
            var b = document.createElement("a");
            return b.href = a, b.pathname
        },
        sanitizeUrl: function(a) {
            var b = this.parsePathname(a);
            return b.replace(/\/\d+/, "")
        }
    }, b.Events);
    return c
}), define("tracking/googleTagManagerSender", ["tracking/trackingHub", "webs/props"], function(a, b) {
    function c() {
        typeof _gtmTrack == "undefined" && (_gtmTrack = []), typeof webs != "undefined" && (webs.site && webs.site.id && _gtmTrack.push({
            siteId: webs.site.id
        }), webs.account && webs.account.id && _gtmTrack.push({
            websId: webs.account.id
        }));
        var a = b && b.environment === "production";
        !a && typeof location != "undefined" && location.search && location.search.indexOf("gtm=1") > 0 && (a=!0);
        if (a) {
            var c = "www.googletagmanager.com/gtm.js", d = "GTM-H7L7", e = document.getElementsByTagName("script");
            for (var f = 0; f < e.length; f++)
                if (e[f].src.indexOf(c)!=-1 && e[f].src.indexOf(d)!=-1) {
                    a=!1;
                    break
                }
            a && function(a, b, d, e, f) {
                a[e] = a[e] || [], a[e].push({
                    "gtm.start": (new Date).getTime(),
                    event: "gtm.js"
                });
                var g = b.getElementsByTagName(d)[0], h = b.createElement(d), i = e != "dataLayer" ? "&l=" + e: "";
                h.async=!0, h.src = "//" + c + "?id=" + f + i, g.parentNode.insertBefore(h, g)
            }(window, document, "script", "_gtmTrack", d)
        }
    }
    var d = {
        pageView: function(a) {
            _gtmTrack.push({
                event: "pageView",
                url: a.url
            })
        },
        click: function(a) {
            _gtmTrack.push({
                event: "click",
                name: a.name
            })
        },
        impression: function(a) {
            _gtmTrack.push({
                event: "impression",
                name: a.name
            })
        },
        campaign: function(a) {
            _gtmTrack.push({
                event: "campaign",
                medium: a.medium,
                campaign: a.campaign,
                content: a.content,
                source: a.source
            })
        }
    };
    c(), a.on("all", function(a, b) {
        a in d && d[a].call(null, b)
    })
}), define("tracking/domTracker", ["tracking/trackingHub", "underscore", "jquery"], function(a, b, c) {
    function d() {
        c("body").on("click", "[data-track-click]", function(b) {
            var d = c(b.target).data("trackClick");
            a.track("click", {
                name: d
            })
        }), c("body").on("track:rendered", function(d) {
            c(d.target).find("[data-track-impression]:visible").each(function() {
                var b = c(this).data("trackImpression");
                a.track("impression", {
                    data: b
                })
            }), c(d.target).find("[data-track-timing]:visible").each(function() {
                var d = window._performanceLabels || [], e = c(this).data("trackTiming"), f = c(this).data("trackVariable");
                a.track("trackTiming", {
                    category: e,
                    variable: f,
                    label: b.uniq(d).sort().join(","),
                    upperBound: 36e5
                })
            })
        }), c("body").trigger("track:rendered");
        var d = window.location.search, e = /[?&]utm_([^=]*)=([^?&]*)/ig, f, g = {}, h=!1;
        while (f = e.exec(d))
            g[f[1]] = f[2], h=!0;
        h && a.track("campaign", g)
    }
    jQuery(d)
}), function() {
    function a(a) {
        a(function() {
            var b = a("#help"), c = a("<div class='w-more_cover' />").insertBefore(b);
            a("body").on("mousedown", function(d) {
                a(d.target).is("#help, #help *") || (b.removeClass("active"), c.removeClass("active"))
            }).on("click", "#help", function(a) {
                b.toggleClass("active"), c.toggleClass("active"), a.preventDefault()
            }).on("click", "[data-dialog]", function(b) {
                b.preventDefault();
                var c = a(this).data();
                (new Popover(c.dialog, {
                    width: c.width,
                    height: c.height,
                    heading: c.heading
                })).show()
            })
        })
    }
    typeof define == "function" && define.amd ? define("ui/layout", ["jquery"], a) : a(jQuery)
}(), function() {
    var a = window.webs || top.webs;
    a && require(["internal/sitebuilder/common/airbrakeNotifier"], function() {
        Hoptoad.setKey("3752f02c4fca4dd061add28609bae37b"), Hoptoad.setEnvironment(a && a.props && a.props.environment || "development"), Hoptoad.setErrorDefaults({
            url: window.location,
            component: "InternalRedesign"
        })
    })
}(), define("apps/App", ["jquery", "underscore", "backbone", "webs/props", "ui/inlineEdit", "ui/contextMenu", "dust-core", "dust-helpers", "nodeDataTooltip", "tracking/googleTagManagerSender", "tracking/domTracker", "ui/layout"], function(a, b, c, d) {
    var e = c.View.extend({});
    return a.ajaxSetup({
        cache: !1
    }), e
})
