Date.prototype.ToString = function(q, y, z) {
    q = q + "";
    z = z || false;
    var v = new Array();
    var w;
    for (var s = 0; s < q.length; ++s) {
        var c = q.charAt(s);
        if (c === "%" && s !== q.length - 1) {
            w = q.charAt(++s);
            switch (w) {
            case"d":
                var p = z ? this.getUTCDate(): this.getDate();
                if (p < 10) {
                    p = "0" + p
                }
                v.push(p);
                break;
            case"m":
                var u = "";
                if (q.charAt(s + 1) === "m") {
                    ++s;
                    u = z ? this.getUTCMonth() : this.getMonth() + 1;
                    switch (u) {
                    case 1:
                        if (y) {
                            u = y.jan
                        } else {
                            u = "Jan"
                        }
                        break;
                    case 2:
                        if (y) {
                            u = y.feb
                        } else {
                            u = "Feb"
                        }
                        break;
                    case 3:
                        if (y) {
                            u = y.mar
                        } else {
                            u = "Mar"
                        }
                        break;
                    case 4:
                        if (y) {
                            u = y.apr
                        } else {
                            u = "Apr"
                        }
                        break;
                    case 5:
                        if (y) {
                            u = y.may
                        } else {
                            u = "May"
                        }
                        break;
                    case 6:
                        if (y) {
                            u = y.jun
                        } else {
                            u = "Jun"
                        }
                        break;
                    case 7:
                        if (y) {
                            u = y.jul
                        } else {
                            u = "Jul"
                        }
                        break;
                    case 8:
                        if (y) {
                            u = y.aug
                        } else {
                            u = "Aug"
                        }
                        break;
                    case 9:
                        if (y) {
                            u = y.sep
                        } else {
                            u = "Sep"
                        }
                        break;
                    case 10:
                        if (y) {
                            u = y.oct
                        } else {
                            u = "Oct"
                        }
                        break;
                    case 11:
                        if (y) {
                            u = y.nov
                        } else {
                            u = "Nov"
                        }
                        break;
                    case 12:
                        if (y) {
                            u = y.dec
                        } else {
                            u = "Dec"
                        }
                        break
                    }
                } else {
                    u = z ? this.getUTCMonth() : this.getMonth() + 1;
                    if (u < 10) {
                        u = "0" + u
                    }
                }
                v.push(u);
                break;
            case"y":
                var A = this.getYear();
                if ((ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE9 || ISQ.Http.browser.isIE10 || ISQ.Http.browser.isIE11) && (!ISQ.Http.browser.isOpera || ISQ.Http.browser.full >= "opera10.52")) {
                    A += 1900
                }
                v.push(A);
                break;
            case"Y":
                var A = this.getYear();
                if ((ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE9 || ISQ.Http.browser.isIE10 || ISQ.Http.browser.isIE11) && (!ISQ.Http.browser.isOpera || ISQ.Http.browser.full >= "opera10.52")) {
                    A += 1900
                }
                if (A) {
                    v.push(A.toString().substr(2, 2))
                }
                break;
            case"H":
                var r = z ? this.getUTCHours(): this.getHours();
                if (r < 10) {
                    r = "0" + r
                }
                v.push(r);
                break;
            case"M":
                var t = z ? this.getUTCMinutes(): this.getMinutes();
                if (t < 10) {
                    t = "0" + t
                }
                v.push(t);
                break;
            case"S":
                var x = z ? this.getUTCSeconds(): this.getSeconds();
                if (x < 10) {
                    x = "0" + x
                }
                v.push(x);
                break;
            case"t":
                var x = z ? this.getUTCMilliseconds(): this.getMilliseconds();
                if (x < 10) {
                    x = "0" + x
                }
                v.push(x);
                break;
            default:
                v.push(q.charAt(s - 1));
                v.push(q.charAt(s));
                break
            }
        } else {
            v.push(c)
        }
    }
    return v.join("")
};
String.prototype.Trim = function(f) {
    if (f) {
        var e = this.startsWith_repeat(f);
        if (e < this.length) {
            var d = this.endsWith_repeat(f);
            if (e != 0 || d != this.length - 1) {
                return this.substring(e, d + 1)
            } else {
                return this.toString()
            }
        } else {
            return ""
        }
    }
    return this.replace(/^\s+|\s+$/g, "")
};
String.prototype.startsWith_repeat = function(e) {
    if (this.length < e.length) {
        return 0
    }
    var f = 0;
    var d = 0;
    for (; d < this.length; d++) {
        if (this.charAt(d) != e.charAt(f)) {
            d -= f;
            break
        }
        if (f == e.length - 1) {
            f = 0
        } else {
            f++
        }
    }
    return d
};
String.prototype.endsWith_repeat = function(e) {
    if (this.length < e.length) {
        return this.length
    }
    var f = e.length - 1;
    var d = this.length - 1;
    for (; d > 0; d--) {
        if (this.charAt(d) != e.charAt(f)) {
            d += f;
            break
        }
        if (f == 0) {
            f = e.length - 1
        } else {
            f--
        }
    }
    return d
};
String.prototype.startsWith = function(d) {
    if (this.length < d.length) {
        return false
    }
    for (var c = 0; c < d.length; ++c) {
        if (this.charAt(c) !== d.charAt(c)) {
            return false
        }
    }
    return true
};
String.prototype.endsWith = function(e) {
    if (this.length < e.length) {
        return false
    }
    var f = e.length - 1;
    for (var d = this.length - 1; d > this.length - 1 - e.length; --d) {
        if (e.charAt(f--) !== this.charAt(d)) {
            return false
        }
    }
    return true
};
String.prototype.contains = function(b) {
    return this.indexOf(b)!==-1
};
String.prototype.LastIndexOf = function(k, j) {
    if (this.length === 0 || k === null) {
        return - 1
    }
    if (k.length > this.length) {
        return - 1
    }
    if (isNaN(j)) {
        j = this.length - k.length
    }
    var f = false;
    for (var g = j; g >= 0; --g) {
        f = true;
        for (var l = 0; l < k.length; ++l) {
            if (this.charAt(g + l) !== k.charAt(l)) {
                f = false;
                break
            }
        }
        if (f) {
            return g
        }
    }
    return - 1
};
String.prototype.LastIndexOf_char = function(d, f) {
    f = f || this.length - 1;
    for (var e = f; e >= 0; --e) {
        if (this.charAt(e) === d) {
            return e
        }
    }
    return - 1
};
String.prototype.setCharAt = function(d, c) {
    if (d > this.length - 1) {
        return str
    }
    return this.substr(0, d) + c + this.substr(d + 1)
};
String.prototype.countCharAppearances = function(d) {
    var e = 0;
    for (var f = 0; f < this.length; ++f) {
        if (this.charAt(f) == d) {
            ++e
        }
    }
    return e
};
String.prototype.equals = function(n, m, g) {
    if (!m) {
        m = 0
    }
    if (m + n.length >= this.length) {
        return false
    }
    var l = 0;
    var k = m + n.length;
    for (var j = m; j < k; ++j) {
        if (g) {
            if (this.charAt(j).toLowerCase() != n.charAt(l++).toLowerCase()) {
                return false
            }
        } else {
            if (this.charAt(j) != n.charAt(l++)) {
                return false
            }
        }
    }
    return true
};
String.prototype.indexOfAny = function(l, k) {
    var f = this.length;
    for (var g = 0; g < l.length; ++g) {
        var j = this.indexOf(l[g], k);
        if (j!=-1 && j < f) {
            f = j
        }
    }
    if (f == this.length) {
        return - 1
    }
    return f
};
String.prototype.replaceMultipleOccurancesWithOne = function(j, n) {
    var m = this.indexOf(j);
    if (m===-1) {
        return this.toString()
    }
    var o = [];
    var l = m;
    var p;
    for (var k = 0; k < this.length; ++k) {
        p = this.charAt(k);
        if (p !== j) {
            o.push(p)
        } else {
            if (l != k - 1) {
                o.push(n)
            }
            l = k
        }
    }
    return o.join("")
};
String.prototype.replacer = function(e) {
    var j = null;
    for (var f = 0; f < this.length; ++f) {
        var g;
        for (g = 0; g < e.length; ++g) {
            if (this.charAt(f) === e[g].from) {
                if (!j) {
                    j = [];
                    j.push(this.substring(0, f))
                }
                if (e[g].to) {
                    j.push(e[g].to)
                }
                break
            }
        }
        if (g === e.length && j) {
            j.push(this.charAt(f))
        }
    }
    if (j) {
        return j.join("")
    } else {
        return this
    }
};
String.prototype.escaper = function() {
    return this.replacer(String.escapeData)
};
String.escapeData = [{
    from: "\r",
    to: "\\r"
}, {
    from: "\n",
    to: "\\n"
}, {
    from: "\t",
    to: "\\t"
}, {
    from: "'",
    to: "'"
}, {
    from: '"',
    to: '\\"'
}, {
    from: "\\",
    to: "\\\\"
}
];
String.prototype.replaceAll = function(e, f, d) {
    if (typeof(d) === "undefined") {
        d = true
    }
    return this.replace(new RegExp(e.replace(/([\/\,\!\\\^\$\{\}\[\]\(\)\.\*\+\?\|\<\>\-\&])/g, "\\$&"), (d ? "gi" : "g")), (typeof(f) == "string") ? f.replace(/\$/g, "$$$$") : f)
};
function initializeNS(e) {
    var f = e.split(".");
    var j = window;
    for (var g = 0; g < f.length; ++g) {
        j = j[f[g]];
        if (typeof(j) === "undefined" ||!j) {
            return {}
        }
    }
    return j
}
function isNSexists(e) {
    var f = e.split(".");
    var j = window;
    for (var g = 0; g < f.length; ++g) {
        j = j[f[g]];
        if (typeof(j) === "undefined" ||!j) {
            return false
        }
    }
    return true
}
ISQ = initializeNS("ISQ");
nanoRep = ISQ;
ISQ.version = "2.45.3.1";
var debugLevel = 0;
ISQ.Base = function Class() {};
ISQ.Base.polymorphismCtor = function(d, c) {
    c = ISQ.Tools.toArray(c);
    if (this.mBase !== null && typeof(this.mBase.prototype.polymorphismCtor) === "function") {
        this.initBase.apply(this, c);
        this.mBase.prototype.polymorphismCtor(d, c)
    }
    this.ctor.apply(d, c)
};
ISQ.Base.extend = function(j) {
    var f = function() {
        if (typeof(j) === "object" && j.abstractClass && this instanceof f) {
            throw ("Trying to create an instance of an abstract class")
        }
        if (this.mBase !== null && typeof(this.mBase.abstractClass) === "object") {
            for (var a in this.mBase.abstractClass) {
                if (typeof(this[a]) !== "function") {
                    throw ("un-implemented abstract method: " + a)
                }
            }
        }
        this.polymorphismCtor(this, arguments);
        if (debugLevel > 2) {
            profiler.registerObjectCreation(this)
        }
    };
    for (var g in this.prototype) {
        f.prototype[g] = this.prototype[g]
    }
    if (typeof(j) === "object" && j.abstractClass) {
        f.abstractClass = {}
    }
    f.prototype.polymorphismCtor = ISQ.Base.polymorphismCtor;
    f.extend = this.extend;
    f.prototype.mBase = this;
    if (typeof(j) !== "object" || typeof(j.name) !== "string") {
        throw ("Class name is missing (ctorObject.name)")
    }
    var e = j.name.Trim();
    if (!e) {
        throw ("Invalid class name")
    }
    f.prototype.__className__ = f.__className__ = e;
    return f
};
ISQ.Base.__className__ = "Base Class";
ISQ.Base.prototype.__className__ = "Base Class";
ISQ.Base.prototype.mBase = null;
ISQ.Base.prototype.mIsDisposed = false;
ISQ.Base.prototype.ctor = function() {};
ISQ.Base.prototype.dtor = function() {
    delete this.mBase
};
ISQ.Base.prototype.initBase = function() {};
ISQ.Base.prototype.isDisposed = function() {
    return this.mIsDisposed
};
ISQ.Base.prototype.executeBaseMethod = function(d, c) {
    if (!this.mBase) {
        throw ("ISQ.Base.prototype.executeBaseMethod: no base!")
    }
    this.mBase.prototype[d].apply(this, c || [])
};
ISQ.Base.prototype.dispose = function() {
    if (this.mIsDisposed) {
        return false
    }
    this.mIsDisposed = true;
    var d = this.dtor.apply(this, arguments);
    if (d === false) {
        this.mIsDisposed = false;
        return false
    }
    if (debugLevel > 2) {
        profiler.registerObjectDisposal(this)
    }
    var c = this.mBase;
    while (c !== null) {
        c.prototype.dtor.apply(this, arguments);
        c = c.prototype.mBase
    }
    this.__dispose__();
    return true
};
ISQ.Base.prototype.__dispose__ = function() {
    for (var e in this) {
        try {
            switch (e) {
            case"mBase":
            case"__dispose__":
            case"polymorphismCtor":
            case"extend":
            case"__className__":
                continue;
                break
            }
            if (typeof(this[e]) === "undefined" || this[e] === null) {
                continue
            }
            if (typeof(this[e]) !== "function" && typeof(this[e]) !== "object") {
                continue
            }
            if (typeof(this[e].dispose) === "function") {
                this[e].dispose()
            }
            if (this[e] instanceof Array) {
                for (var f in this[e]) {
                    if (this[e][f] && typeof(this[e][f].dispose) === "function") {
                        this[e][f].dispose()
                    }
                }
            }
            delete this[e]
        } catch (d) {
            if (debugLevel > 1) {
                alert("base::__dispose__: error disposing:" + e)
            }
        }
    }
};
ISQ.Base.prototype.isDerivedOf = function(e, g) {
    g = g === true;
    if (g && this.__className__ === e.__className__) {
        return false
    }
    if (this.__className__ === e.__className__) {
        return true
    }
    var j = false;
    var f = this.mBase;
    while (f !== null&&!j) {
        if (e.__className__ === f.__className__) {
            j = true
        }
        f = f.prototype.mBase
    }
    return j
};
ISQ.Base.isDerivedOf = function(c, d) {
    if (c === null) {
        return false
    }
    if (typeof(c) !== "object" || typeof(c.isDerivedOf) !== "function") {
        return false
    }
    return c.isDerivedOf(d)
};
ISQ.Profiler = function() {
    this.mAllocationArray = new Array();
    this.mDisposalArray = new Array();
    this.mCurrentArray = new Array();
    this.mPointerArray = new Array()
};
ISQ.Profiler.prototype.dispose = function() {
    delete this.mAllocationArray;
    delete this.mDisposalArray;
    delete this.mCurrentArray;
    delete this.mPointerArray
};
ISQ.Profiler.prototype.registerObjectCreation = function(g) {
    var e = g.__className__;
    var f = this.mAllocationArray[e];
    if (typeof(f) === "undefined") {
        f = 0
    }
    this.mAllocationArray[e]=++f;
    var f = this.mCurrentArray[e];
    if (typeof(f) === "undefined") {
        f = 0
    }
    this.mCurrentArray[e]=++f;
    var j = this.mPointerArray[e];
    if (!j) {
        j = new Array();
        this.mPointerArray[e] = j
    }
    j.push(g)
};
ISQ.Profiler.prototype.registerObjectDisposal = function(k) {
    if (this.inCleaningProcedure) {
        return 
    }
    var f = k.__className__;
    var g = this.mDisposalArray[f];
    if (typeof(g) === "undefined") {
        g = 0
    }
    this.mDisposalArray[f]=++g;
    g = this.mCurrentArray[f];
    this.mCurrentArray[f]=--g;
    var l = this.mPointerArray[f];
    if (l) {
        for (var j = 0; j < l.length; ++j) {
            if (l[j] === k) {
                l.splice(j, 1);
                break
            }
        }
    }
};
ISQ.Profiler.prototype.reset = function() {
    this.inCleaningProcedure = false;
    this.mAllocationArray = new Array();
    this.mDisposalArray = new Array();
    this.mCurrentArray = new Array();
    this.mPointerArray = new Array()
};
ISQ.Profiler.prototype.validatePointerArray = function() {
    var e = new Array();
    for (var g in this.mPointerArray) {
        var f = this.mPointerArray[g];
        var j = this.mCurrentArray[g];
        if (j !== f.length) {
            e[g] = true
        }
    }
    return e
};
ISQ.Profiler.prototype.analyzeHandlers = function() {
    var l = this.mPointerArray["ISQ.Handler"];
    var f = new Array();
    for (var k = 0; k < l.length; ++k) {
        var j = l[k];
        var g = j.context;
        if (!ISQ.Base.isDerivedOf(g, ISQ.Base)) {
            continue
        }
        if (g.isDisposed()) {
            f.push({
                name: g.__className__,
                instance: g,
                func: j.func
            })
        }
    }
    return f
};
ISQ.Profiler.prototype.emptyEvents = function() {
    var f = this.mPointerArray["ISQ.Event"];
    var d = 0;
    for (var e = 0; e < f.length; ++e) {
        if (f[e].count() === 0) {
            ++d
        }
    }
    return d
};
ISQ.Profiler.prototype.emptyEventsContainers = function() {
    var k = new Array();
    var m;
    for (var e in this.mPointerArray) {
        if (e === "ISQ.Handler" || e == "ISQ.Event") {
            continue
        }
        for (var l = 0; l < this.mPointerArray[e].length; ++l) {
            m = this.mPointerArray[e][l];
            for (var n in m) {
                try {
                    switch (n) {
                    case"dispose":
                    case"mIsDisposed":
                    case"ctor":
                    case"dtor":
                    case"mBase":
                    case"__dispose__":
                    case"polymorphismCtor":
                    case"extend":
                    case"__className__":
                        continue;
                        break
                    }
                    if (typeof(m[n]) === "undefined" || m[n] === null) {
                        continue
                    }
                    if (typeof(m[n]) !== "function" && typeof(m[n]) !== "object") {
                        continue
                    }
                    if (m[n].__className__ === "ISQ.Event") {
                        if (m[n].count() === 0) {
                            if (!k[e]) {
                                k[e] = new Array()
                            }
                            if (typeof(k[e][n]) === "undefined") {
                                k[e][n] = 1
                            } else {
                                ++k[e][n]
                            }
                        }
                    }
                } catch (j) {}
            }
        }
    }
    return k
};
ISQ.Profiler.prototype.inCleaningProcedure = false;
ISQ.Profiler.prototype.allocations = function() {
    return this.mAllocationArray
};
ISQ.Profiler.prototype.current = function() {
    return this.mCurrentArray
};
ISQ.Profiler.prototype.disposals = function() {
    return this.mDisposalArray
};
ISQ.Profiler.prototype.pointers = function() {
    return this.mPointerArray
};
if (typeof(profiler) === "undefined" && debugLevel > 2) {
    profiler = new ISQ.Profiler()
}
ISQ.Tools = initializeNS("ISQ.Tools");
ISQ.Tools.stateEnum = {
    KEY: 5,
    VALUE: 6
};
ISQ.Tools.stringToPairArray = function(s, m, v, t, u) {
    var n = s.length;
    if (n === 0) {
        return null
    }
    var q = new Array();
    t = t !== false;
    u = u !== false;
    var r = ISQ.Tools.stateEnum.KEY;
    var o = "";
    var l = new Array();
    var p = 0;
    while (p < n) {
        switch (r) {
        case ISQ.Tools.stateEnum.KEY:
            if (s.charAt(p) !== m) {
                l.push(s.charAt(p))
            } else {
                r = ISQ.Tools.stateEnum.VALUE;
                o = ISQ.Tools.getStringValue(l, t);
                l = new Array()
            }
            break;
        case ISQ.Tools.stateEnum.VALUE:
            if (s.charAt(p) !== v) {
                l.push(s.charAt(p))
            } else {
                r = ISQ.Tools.stateEnum.KEY;
                q[o] = ISQ.Tools.getStringValue(l, u);
                l = new Array()
            }
            break
        }
        ++p
    }
    if (r === ISQ.Tools.stateEnum.VALUE) {
        q[o] = ISQ.Tools.getStringValue(l)
    }
    return q
};
ISQ.Tools.getStringValue = function(d, e) {
    e = e === true;
    var f = unescape(d.join("").toString()).Trim();
    if (e) {
        f = f.toLowerCase()
    }
    return f
};
ISQ.Tools.getNumber = function(g, e) {
    var j = typeof(g);
    if (typeof(g) === "number") {
        return g
    }
    if (typeof(g) !== "string") {
        g = g.toString()
    }
    var f;
    if (g.contains(".")) {
        f = parseFloat(g)
    } else {
        f = parseInt(g)
    }
    if (isNaN(f)) {
        return typeof(e) === "number" ? e : NaN
    }
    return f
};
ISQ.Tools.verifyNumber = function(d) {
    var c = ISQ.Tools.getNumber(d);
    if (isNaN(c)) {
        throw ("ISQ.Tools.verifyNumber: given value is not a number")
    }
};
ISQ.Tools.getEnumValue = function(d, f) {
    if (typeof(d) !== "object" || f === null) {
        return null
    }
    for (var e in d) {
        if (d[e] == f) {
            return e
        }
    }
    return null
};
ISQ.Tools.getOpacityFromRGBA = function(d) {
    if (d === "transparent") {
        return 0
    }
    var c = d.LastIndexOf(",");
    if (c===-1) {
        throw ("ISQ.Tools.getOpacityFromRGBA(): Invalid string")
    }
    if (d.countCharAppearances(",") === 2) {
        return 100
    }
    d = d.substring(c + 1);
    return parseInt(d)
};
ISQ.Tools.RGBtoHex = function(f, e, d) {
    return ISQ.Tools.toHex(f) + ISQ.Tools.toHex(e) + ISQ.Tools.toHex(d)
};
ISQ.Tools.RGBtoHex2 = function(b) {
    b = b.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
    return "#" + ISQ.Tools.toHex(b[1]) + ISQ.Tools.toHex(b[2]) + ISQ.Tools.toHex(b[3])
};
ISQ.Tools.toHex = function(b) {
    if (typeof(b) === "string") {
        b = parseInt(b)
    }
    if (typeof(b) !== "number" || b === 0) {
        return "00"
    }
    b = Math.max(0, b);
    b = Math.min(b, 255);
    b = Math.round(b);
    return "0123456789ABCDEF".charAt((b - b%16) / 16) + "0123456789ABCDEF".charAt(b%16)
};
ISQ.Tools.hexToRGB = function(c) {
    var d = {
        r: - 1,
        g: - 1,
        b: - 1
    };
    c = (c.charAt(0) == "#") ? c.substring(1) : c;
    d.r = parseInt(c.substring(0, 2), 16);
    d.g = parseInt(c.substring(2, 4), 16);
    d.b = parseInt(c.substring(4, 6), 16);
    return d
};
ISQ.Tools.getPrettyNumber = function(x, q, s, u) {
    x = parseFloat(x);
    if (isNaN(x)) {
        return x
    }
    var z = "";
    if (s) {
        if (x >= 1073741824) {
            x = x / 1073741824;
            z = "GB"
        } else {
            if (x >= 1048576) {
                x = x / 1048576;
                z = "MB"
            } else {
                if (x >= 1024) {
                    x = x / 1024;
                    z = "KB"
                } else {
                    z = "B"
                }
            }
        }
    } else {
        if (u && x >= 1000) {
            x = x / 1000;
            z = "K"
        }
    }
    if (isNaN(q) && q !== true && q !== false) {
        q = true
    }
    switch (q) {
    case false:
        break;
    case 0:
        x = parseInt(x);
        break;
    case true:
        if (Math.abs(x) < 10) {
            q = 2
        } else {
            if (Math.abs(x) < 100) {
                q = 1
            } else {
                q = 0
            }
        }
        x = x.toFixed(q);
        break;
    default:
        x = x.toFixed(q);
        break
    }
    if (x%1 === 0) {
        x = parseInt(x)
    }
    var y = x + "";
    var r = null;
    var v = y.indexOf(".");
    if (v!==-1) {
        r = y.substr(v);
        var A =- 1;
        for (var t = 0; t < r.length; ++t) {
            if (r.charAt(t) === "0") {
                if (A==-1) {
                    A = t
                }
            } else {
                A =- 1
            }
        }
        if (A!=-1) {
            r = r.substr(0, A)
        }
        if (r.length === 1) {
            r = null
        }
        y = y.substr(0, v)
    }
    x = parseInt(y);
    if ((x / 1000) >= 1) {
        var o = new Array(y.length + Math.floor((y.length - 1) / 3));
        var w = o.length;
        var p = 0;
        for (var t = y.length - 1; t>-1; --t) {
            if (t < y.length - 1 && p%3 === 0) {
                o[w--] = ","
            }
            o[w--] = y.charAt(t);
            ++p
        }
    } else {
        o = [];
        o.push(x)
    }
    if (r !== "") {
        o.push(r)
    }
    if (z) {
        o.push(z)
    }
    return o.join("")
};
ISQ.Tools.formatPercentage = function(f) {
    var e;
    var d = Math.abs(f);
    if (d > 1000) {
        e = (f / 1000).toFixed(0) + "K "
    } else {
        if (d < 1) {
            e = f.toFixed(2)
        } else {
            if (d < 10) {
                e = f.toFixed(1)
            } else {
                e = f.toFixed(0)
            }
        }
    }
    return e + "%"
};
ISQ.Tools.toArray = function(f) {
    if (f instanceof Array) {
        return f
    }
    var a = new Array();
    if (typeof(f.length) !== "number") {
        a.push(f)
    } else {
        var e;
        if (f.length === 0) {
            for (e in f) {
                a[e] = f[e]
            }
        } else {
            if (f.length > 10000) {
                _debug().insertText("Error converting object to an array. Object length: " + f.length);
                return new Array()
            }
            for (e = 0; e < f.length; ++e) {
                a[e] = f[e]
            }
        }
    }
    return a
};
ISQ.Tools.getDateTicks = function(m, o, p, q, u) {
    m = m.Trim();
    if (m === "") {
        return - 1
    }
    var n, r, v;
    var s = o.indexOf("dd");
    n = m.substr(s, 2);
    if (isNaN(n)) {
        return - 1
    }
    s = o.indexOf("mm");
    r = m.substr(s, 2);
    if (isNaN(r)) {
        return - 1
    }
    s = o.indexOf("yyyy");
    if (s===-1) {
        s = o.indexOf("yy")
    }
    v = m.substr(s, 4);
    if (isNaN(v)) {
        return - 1
    }
    var t = new Date();
    p = typeof(p) == "undefined" ? t.getHours() : p;
    q = typeof(q) == "undefined" ? t.getMinutes() : q;
    u = typeof(u) == "undefined" ? t.getSeconds() : u;
    var l = new Date(v, r - 1, n, p, q, u);
    return Date.UTC(l.getUTCFullYear(), l.getUTCMonth(), l.getUTCDate(), l.getUTCHours(), l.getUTCMinutes(), l.getUTCSeconds())
};
ISQ.Tools.getIPString = function(g) {
    var f = false;
    if (g < 0) {
        g =- g;
        f = true
    }
    var j = new Array();
    if (f) {
        j.push(256 - (g%256))
    } else {
        j.push(g%256)
    }
    for (var e = 2; e >= 0; e--) {
        g = Math.floor(g / 256);
        j.push(".");
        if (f) {
            j.push(255 - (g%256))
        } else {
            j.push(g%256)
        }
    }
    return j.join("")
};
ISQ.Tools.getContentTypeByExtension = function(e, f) {
    var d;
    switch (e) {
    case"bmp":
        d = "image/bmp";
        break;
    case"gif":
        d = "image/gif";
        break;
    case"jpg":
    case"jpeg":
        d = "image/jpeg";
        break;
    case"png":
        d = "image/png";
        break;
    case"tiff":
    case"tif":
        d = "image/tiff";
        break;
    case"pdf":
        d = "application/pdf";
        break;
    default:
        d = "";
        break
    }
    if (f) {
        return d
    }
    switch (e) {
    case"doc":
        d = "application/msword";
        break;
    case"docx":
        d = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
        break;
    case"pdf":
        d = "application/pdf";
        break;
    case"ppt":
        d = "application/vnd.ms-powerpoint";
        break;
    case"pptx":
        d = "application/vnd.openxmlformats-officedocument.presentationml.presentation";
        break;
    case"xlsx":
        d = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
        break;
    case"xls":
        d = "application/vnd.ms-excel";
        break;
    case"csv":
        d = "text/csv";
        break;
    case"xml":
        d = "text/xml";
        break;
    case"txt":
        d = "text/plain";
        break;
    case"zip":
        d = "application/zip";
        break;
    case"ogg":
        d = "application/ogg";
        break;
    case"mp3":
        d = "audio/mpeg";
        break;
    case"wma":
        d = "audio/x-ms-wma";
        break;
    case"wav":
        d = "audio/x-wav";
        break;
    case"wmv":
        d = "audio/x-ms-wmv";
        break;
    case"swf":
        d = "application/x-shockwave-flash";
        break;
    case"avi":
        d = "video/avi";
        break;
    case"mp4":
        d = "video/mp4";
        break;
    case"mpg":
    case"mpeg":
        d = "video/mpeg";
        break;
    case"qt":
        d = "video/quicktime";
        break;
    default:
        d = ""
    }
    return d
};
ISQ.Tools.extend = function(k, q) {
    q = q || {};
    k = k === true;
    var m = arguments.length;
    var o = null;
    for (var l = 2; l < m; l++) {
        o = arguments[l];
        if (typeof(o) === "undefined" || o === null) {
            continue
        }
        for (var n in o) {
            var p = q[n], j = o[n];
            if (q === j) {
                continue
            }
            if (typeof(j) === "undefined") {
                continue
            }
            if (k && j !== null && typeof(j) === "object"&&!j.nodeType) {
                if (typeof(j.isDerivedOf) !== "function" ||!j.isDerivedOf(ISQ.Base)) {
                    q[n] = ISQ.Tools.extend(true, p || (j.length != null ? [] : {}), j)
                } else {
                    q[n] = j
                }
            } else {
                if (j !== undefined) {
                    q[n] = j
                }
            }
        }
    }
    return q
};
ISQ.Tools.addArrayToGC = function(d, c) {
    d.add({
        dispose: function() {
            for (var a in c) {
                c[a].dispose();
                c[a] = null
            }
            c = null
        }
    })
};
ISQ.Tools.foreach = function(e, f) {
    for (var g in e) {
        var j = null;
        if (ISQ.Base.isDerivedOf(f, ISQ.Handler)) {
            j = f.handle(e[g], g)
        } else {
            j = f(e[g], g)
        }
        if (j === false) {
            break
        }
    }
};
window.foreach = ISQ.Tools.foreach;
ISQ.Tools.iteration = function(f, j, g) {
    if (!g) {
        for (var k = 0; k < f.length; ++k) {
            var l = null;
            if (ISQ.Base.isDerivedOf(j, ISQ.Handler)) {
                l = j.handle(f[k], k)
            } else {
                l = j(f[k], k)
            }
            if (l === false) {
                break
            }
        }
    } else {
        for (var k = f.length - 1; k >= 0; --k) {
            var l = null;
            if (ISQ.Base.isDerivedOf(j, ISQ.Handler)) {
                l = j.handle(f[k], k)
            } else {
                l = j(f[k], k)
            }
            if (l === false) {
                break
            }
        }
    }
};
window.iteration = window._it = ISQ.Tools.iteration;
ISQ.Tools.addRemoveClickOutsideElementsListener = function(g, j, e, f) {
    if (!ISQ.Tools.clickOutsideListeners) {
        ISQ.Tools.clickOutsideListeners = new Array()
    }
    if (j) {
        if (!g) {
            throw ("ISQ.Tools.addRemoveClickOutsideElementsListener: invalid id")
        }
        if (!ISQ.Base.isDerivedOf(f, ISQ.Handler)) {
            throw ("ISQ.Tools.addRemoveClickOutsideElementsListener: invalid handler")
        }
        if (!ISQ.Tools.clickOutsideListeners[g]) {
            ISQ.Tools.clickOutsideListeners[g] = {}
        }
        if (ISQ.Tools.clickOutsideListeners[g].registered) {
            if (document.removeEventListener) {
                document.removeEventListener("click", ISQ.Tools.clickOutsideListeners[g].func, true);
                if (ISQ.Http.browser.isTablet) {
                    document.removeEventListener("touchstart", ISQ.Tools.clickOutsideListeners[g].func, true)
                }
            } else {
                if (document.detachEvent) {
                    document.detachEvent("onclick", ISQ.Tools.clickOutsideListeners[g].func)
                }
            }
        }
        ISQ.Tools.clickOutsideListeners[g].func = function(a) {
            if (ISQ.Tools.clickOutsideListeners[g].ignoreNextClick) {
                ISQ.Tools.clickOutsideListeners[g].ignoreNextClick = false;
                return 
            }
            if (ISQ.Tools.clickOutsideElements(a, e)) {
                f.handle(a)
            }
        };
        ISQ.Tools.clickOutsideListeners[g].registered = true;
        if (document.addEventListener) {
            document.addEventListener("click", ISQ.Tools.clickOutsideListeners[g].func, true);
            if (ISQ.Http.browser.isTablet) {
                document.addEventListener("touchstart", ISQ.Tools.clickOutsideListeners[g].func, true)
            }
        } else {
            if (document.attachEvent) {
                document.attachEvent("onclick", ISQ.Tools.clickOutsideListeners[g].func)
            }
        }
    } else {
        if (ISQ.Tools.clickOutsideListeners[g] && ISQ.Tools.clickOutsideListeners[g].registered) {
            ISQ.Tools.clickOutsideListeners[g].registered = false;
            if (document.removeEventListener) {
                document.removeEventListener("click", ISQ.Tools.clickOutsideListeners[g].func, true);
                if (ISQ.Http.browser.isTablet) {
                    document.removeEventListener("touchstart", ISQ.Tools.clickOutsideListeners[g].func, true)
                }
            } else {
                if (document.detachEvent) {
                    document.detachEvent("onclick", ISQ.Tools.clickOutsideListeners[g].func)
                }
            }
        }
    }
};
ISQ.Tools.fireMouseEvents = function(f, e) {
    if (document.createEvent) {
        var d = document.createEvent("MouseEvents");
        d.initEvent(e, true, false);
        f.dispatchEvent(d)
    } else {
        if (document.createEventObject) {
            f.fireEvent("on" + e)
        }
    }
};
ISQ.Tools.clickOutsideElements = function(g, f) {
    if (!f ||!(f instanceof Array)) {
        return false
    }
    var e = ISQ.Html.getDomEvent(g);
    while (e.target !== document.body && e.target !== null) {
        for (var j in f) {
            if (e.target == f[j]) {
                return false
            }
        }
        e.target = e.target.parentNode
    }
    if (e.target !== document.body && e.target !== null) {
        return false
    }
    return true
};
ISQ.Tools.ignoreNextClickOutsideElementsEvent = function(b) {
    if (!ISQ.Tools.clickOutsideListeners[b]) {
        return 
    }
    ISQ.Tools.clickOutsideListeners[b].ignoreNextClick = true
};
ISQ.Tools.pagerSeperator = "<page_separator/>";
ISQ.Data = initializeNS("ISQ.Data");
ISQ.Data.getEmailAddress = function(m) {
    var g = /[\w-_\&.]+\@[\w-_]+\.+[\w-_.]+/g;
    var l = m.match(g);
    if (l === null || l.length === 0) {
        return null
    }
    var n = new RegExp();
    n.compile("[~!#$%^&*()=`'\"\\/><?[]{}|]", "g");
    var j = new Array();
    for (var k = 0; k < l.length; ++k) {
        if (l[k].search(n)===-1) {
            j.push(l[k])
        }
    }
    return j
};
ISQ.Data.checkString = function(n, o) {
    var m = true;
    var l;
    switch (o) {
    case"email":
        if (n.contains("..")) {
            return false
        }
        if (n.contains(".@") || n.contains("@.")) {
            return false
        }
        var j = /^[\w-_\&\+.]+\@([\w-_]+\.){1,}[\w-_]+$/;
        var p = new RegExp("[~!#$%^&*()=''\"\\/><?[]{}|]", "g");
        return (n.search(p)===-1) && (n.match(j)) && (n !== "");
        break;
    case"number":
        var k = /^[-]?[0-9]+\.?[0-9]?$/;
        return k.test(n);
        break;
    case"phone":
        var k = /^\+?[0-9\s(-)-\[\].\ ]*[#x]?[0-9\s]*?$/;
        return k.test(n);
        break;
    case"username":
        l = new RegExp(".+@.+");
        return (l.test(n));
        break;
    case"url":
        l = new RegExp("^https?://([a-zA-Z0-9]+.){1,}[a-zA-Z0-9]+");
        return (l.test(n));
        break
    }
    return m
};
ISQ.Data.validateRangeInputs = function(m, p, o, j) {
    var n = m.value.Trim();
    if (!n) {
        return 
    }
    var q = p.value.Trim();
    if (!q) {
        return 
    }
    var k = ISQ.Tools.getDateTicks(n, j);
    var l = ISQ.Tools.getDateTicks(q, j);
    if (k > l) {
        if (o === m) {
            m.value = q
        } else {
            p.value = n
        }
    }
};
ISQ.Tools.blinkElement = function(e, j, g, f) {
    if (g) {
        ISQ.CSS.addClass(e, f);
        j--
    } else {
        ISQ.CSS.removeClass(e, f)
    }
    if (j) {
        setTimeout(function() {
            ISQ.Tools.blinkElement(e, j, !g, f)
        }, 150)
    }
};
ISQ.Tools.toggleBlinkElement = function(d, e, f) {
    if (d.blinkInterval) {
        clearInterval(d.blinkInterval);
        ISQ.CSS.removeClass(d, e);
        delete d.blinkInterval;
        return 
    }
    d.blinkInterval = setInterval(function() {
        if (d.blinkOn) {
            d.blinkOn = false;
            ISQ.CSS.removeClass(d, e)
        } else {
            d.blinkOn = true;
            ISQ.CSS.addClass(d, e)
        }
    }, f || 800)
};
ISQ.Data.checkTextWidth = function(f, e) {
    var d = ISQ.Html._("ellipsisDiv");
    if (!d) {
        d = createDiv(null, "position:fixed;white-space:nowrap;top:0px;left:0px;z-index:-5;visibility:hidden");
        d.id = "ellipsisDiv"
    } else {
        clearNode(d)
    }
    ISQ.Data.copyElementFontStyle(e, d);
    d.innerHTML = f;
    document.body.insertBefore(d, document.body.firstChild);
    return d.offsetWidth
};
ISQ.Data.copyElementFontStyle = function(e, f) {
    if (!f ||!e) {
        return 
    }
    var d = null;
    if (ISQ.Http.browser.isIE9 && document.body.style.overflow) {
        d = document.body.style.overflow;
        document.body.style.overflow = "hidden"
    }
    f.style.fontSize = ISQ.CSS.getElementStyle(e, "font-size");
    f.style.fontWeight = ISQ.CSS.getElementStyle(e, "font-weight");
    f.style.fontFamily = ISQ.CSS.getElementStyle(e, "font-family");
    f.style.fontStyle = ISQ.CSS.getElementStyle(e, "font-style");
    f.style.textDecoration = ISQ.CSS.getElementStyle(e, "text-decoration");
    f.style.color = ISQ.CSS.getElementStyle(e, "color");
    if (d) {
        document.body.style.overflow = d
    }
};
ISQ.Data.autoEllipseText = function(C, E, r, u, s, q) {
    var D;
    var t = false;
    q = q || null;
    if (u) {
        D = u.offsetWidth
    } else {
        D = ISQ.Data.checkTextWidth(C, r);
        u = ISQ.Html._("ellipsisDiv")
    }
    var z = C;
    if (D > E) {
        t = true;
        u.innerHTML = "";
        var B = 0, v = C.length, y;
        var A = v;
        do {
            y = B + Math.floor((v - B) / 2);
            var x = false;
            var w;
            for (w = v; w > 0 && (w > y || x); --w) {
                if (C.charAt(w) == ">") {
                    x = true
                } else {
                    if (C.charAt(w) == "<") {
                        x = false
                    }
                }
            }
            y = w;
            u.innerHTML = C.substr(0, y) + "...";
            if (u.offsetWidth < E) {
                B = y
            } else {
                v = y
            }
            if (A == y) {
                break
            }
            A = y
        }
        while ((v - B) > 1);
        if (u.offsetWidth > E || A == y) {
            C = C.substr(0, y - 1) + "..."
        } else {
            C = u.innerHTML
        }
    } else {
        if (s) {
            return 
        }
    }
    if (r) {
        r.innerHTML = C;
        if (t) {
            r.setAttribute("title", z)
        }
    } else {
        return C
    }
};
ISQ.Data.indexOfArray = function(d, f) {
    var e = d.length;
    while (e--) {
        if (d[e] == f) {
            return e
        }
    }
    return - 1
};
ISQ.Data.inArray = function(c, d) {
    return ISQ.Data.indexOfArray(c, d)!==-1
};
ISQ.Data.isObjectEmpty = function(d) {
    for (var c in d) {
        return false
    }
    return true
};
ISQ.Data.jsonToString = function(m, n, l) {
    if (!l&&!window.Prototype) {
        try {
            return JSON.stringify(m)
        } catch (e) {}
    }
    if (!n) {
        n = new Array()
    }
    if (m instanceof Array) {
        n.push("[");
        for (var k = 0; k < m.length; ++k) {
            if (k !== 0) {
                n.push(",")
            }
            ISQ.Data.jsonToString(m[k], n, true)
        }
        n.push("]")
    } else {
        if (m === null) {
            n.push("null")
        } else {
            if (typeof(m) === "object") {
                n.push("{");
                var j = true;
                for (var k in m) {
                    if (!j) {
                        n.push(",")
                    }
                    j = false;
                    n.push('"');
                    n.push(k);
                    n.push('"');
                    n.push(":");
                    ISQ.Data.jsonToString(m[k], n, true)
                }
                n.push("}")
            } else {
                if (typeof(m) === "string") {
                    n.push('"');
                    n.push(m.escaper());
                    n.push('"')
                } else {
                    n.push(m)
                }
            }
        }
    }
    if (!l) {
        return n.join("")
    }
};
ISQ.Data.blockCreditCardNumbers = function(t, r) {
    var q = /(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})/g;
    var u = t.replace(/ /g, "").replace(/-/g, "").replace(/\n/g, "").replace(/\r/g, "").replace(/\./g, "");
    var w = u;
    var p = u.match(q);
    if (!p || p.length == 0) {
        return t
    }
    for (var j = 0; j < p.length; ++j) {
        var o = p[j];
        var s = new Array();
        for (var n = 0; n < o.length; ++n) {
            s.push(r)
        }
        w = w.replace(o, s.join(""))
    }
    var v = 0;
    for (var j = 0; j < t.length; ++j) {
        if (t.charAt(j) == " " || t.charAt(j) == "-") {
            continue
        }
        if (u.charAt(v) != w.charAt(v)) {
            t = t.setCharAt(j, r)
        }
        v++
    }
    return t
};
ISQ.Data.createLink = function(n, q, o, l, k, p, j) {
    if (!n ||!q) {
        return ""
    }
    if (k && p) {
        n = ISQ.Data.unHTML(ISQ.Data.breakWords(n))
    } else {
        if (k) {
            n = ISQ.Data.breakWords(n)
        } else {
            if (p) {
                n = ISQ.Data.unHTML(n)
            }
        }
    }
    if (j&&!q.startsWith("http://")&&!q.startsWith("https://")) {
        q = "http://" + q
    }
    var m = new Array();
    m.push('<a href="');
    m.push(q);
    if (o) {
        m.push('" target="');
        m.push(o)
    }
    if (l) {
        m.push('" class="');
        m.push(l)
    }
    m.push('">');
    m.push(n);
    m.push("</a>");
    return m.join("")
};
ISQ.Data.parseURL = function(c, d) {
    if (!c) {
        return ""
    }
    return c.replace(/[A-Za-z]+:\/\/[A-Za-z0-9-_]+\.[A-Za-z0-9-_:%&\?\/.=]+/, function(a) {
        return ISQ.Data.createLink(a, a, d)
    })
};
ISQ.Data.parseTwitterStatus = function(j) {
    if (!j) {
        return ""
    }
    var g = new ISQ.Data.EntityParser(j);
    var e = new Array();
    var f = {};
    while (g.getNextSegment(f)) {
        f.segment = ISQ.Data.parseURL(f.segment, "_blank");
        f.segment = f.segment.replace(/[@]+[A-Za-z0-9-_]+/g, function(a) {
            var c = a.replace("@", "");
            var b = "http://twitter.com/" + c;
            return ISQ.Data.createLink(a, b, "_blank")
        });
        f.segment = f.segment.replace(/[#]+[A-Za-z0-9-_]+/g, function(a) {
            var b = a.replace("#", "%23");
            var c = "http://search.twitter.com/search?q=" + b;
            return ISQ.Data.createLink(a, c, "_blank")
        });
        e.push(f.segment);
        e.push(f.entity)
    }
    return e.join("")
};
ISQ.Data.parseArticleContext = function(n, m) {
    result = new Array();
    var l = 0, j = false, k = 0, g;
    while (l < n.length) {
        l = n.indexOf(",", k);
        if (l===-1) {
            l = n.length
        }
        j=!j;
        if (j) {
            g = n.substring(k, l++)
        } else {
            if (!m) {
                result[g] = n.substring(k, l++)
            } else {
                result.push(n.substring(k, l++))
            }
        }
        k = l
    }
    return result
};
ISQ.Data.uniqueId = function() {
    return Math.floor(Math.random() * 10000000)
};
ISQ.Data.nowMili = new Date().getTime();
ISQ.Data.dayInMs = 24 * 60 * 60 * 1000;
ISQ.Tools.fixIE7DropDownInIframe = function(d) {
    var f, e;
    f =+ this.getAttribute("size") || 1;
    e = 1;
    if (d === "click" && f === 1) {
        e = this.options.length
    }
    this.setAttribute("size", e)
};
ISQ.Tools.forceNumericValue = function(c, d) {
    if (d) {
        c.setAttribute("maxLength", d)
    }
    $(document).ready(function() {
        $(c).keydown(function(a) {
            if (a.keyCode == 46 || a.keyCode == 8 || a.keyCode == 9 || a.keyCode == 27 || (a.keyCode == 65 && a.ctrlKey === true) || (a.keyCode >= 35 && a.keyCode <= 39)) {
                return 
            } else {
                if (a.shiftKey || (a.keyCode < 48 || a.keyCode > 57) && (a.keyCode < 96 || a.keyCode > 105)) {
                    a.preventDefault()
                }
            }
        })
    })
};
ISQ.Tools.postDataUsingFrame = function(g) {
    var m = "frm_" + ISQ.Data.uniqueId();
    var l = ISQ.Html.createIframe(m, m, "display:none;", "");
    document.body.appendChild(l);
    var j = createDiv(document.body, "display:none");
    var k = createElement("form", j);
    k.target = m;
    k.action = "/KB";
    k.method = "post";
    var n = createInput("hidden", "info", k, escape(g));
    n.value = unescape(n.value);
    k.submit();
    document.body.removeChild(j);
    setTimeout(function() {
        try {
            document.body.removeChild(l)
        } catch (a) {}
    }, 30000)
};
ISQ.Data.ADDRESS_DELIMITER = String.fromCharCode(187);
ISQ.Data.MailAddress = function(c, d) {
    this.address = c || null;
    this.displayName = d || null;
    this.isValid = this.validate()
};
ISQ.Data.MailAddress.prototype.validate = function() {
    if (!this.address) {
        return false
    }
    return ISQ.Data.checkString(this.address, "email")
};
ISQ.Data.MailAddress.prototype.asString = function() {
    var b = new Array();
    if (this.displayName) {
        if (this.displayName.indexOf('"')==-1) {
            b.push('"');
            b.push(this.displayName.replace(/\"/g, '\\"'));
            b.push('"')
        }
        if (this.address.indexOf("<")==-1) {
            b.push("<");
            b.push(this.address);
            b.push(">")
        }
        b.push(" ")
    } else {
        b.push(this.address)
    }
    return b.join("")
};
ISQ.Data.parseEmailAddress = function(q) {
    if (!q) {
        return null
    }
    var m = {
        none: 0,
        address: 1,
        name: 2,
        nameQuotes: 3
    };
    var k, l;
    var n = new Array();
    var j = function() {
        if (k == null) {
            if (l != null) {
                n.push(new ISQ.Data.MailAddress(l.Trim()))
            } else {
                throw ("ISQ.Data.parseEmailAddress: Error parsing input string: " + q)
            }
        } else {
            if (l == null) {
                n.push(new ISQ.Data.MailAddress(k))
            } else {
                n.push(new ISQ.Data.MailAddress(k, l))
            }
        }
        k = null;
        l = null
    };
    var r = m.none;
    var p = 0, o = 0;
    for (o = 0; o < q.length; ++o) {
        switch (r) {
        case m.none:
            if (/\s/.test(q[o])) {
                continue
            } else {
                switch (q[o]) {
                case"<":
                    r = m.address;
                    p = o + 1;
                    break;
                case'"':
                    r = m.nameQuotes;
                    p = o + 1;
                    break;
                case";":
                case",":
                    j();
                    r = m.none;
                    break;
                default:
                    r = m.name;
                    p = o;
                    break
                }
            }
            break;
        case m.address:
            if (q[o] == ">") {
                k = q.substring(p, o);
                r = m.none
            }
            break;
        case m.name:
            switch (q[o]) {
            case"<":
                l = q.substring(p, o);
                r = m.address;
                p = o + 1;
                break;
            case";":
            case",":
                l = q.substring(p, o);
                j();
                r = m.none;
                break
            }
            break;
        case m.nameQuotes:
            switch (q[o]) {
            case"\\":
                if (q.length > o + 1) {
                    ++o
                }
                break;
            case'"':
                l = q.substring(p, o);
                r = m.none;
                break
            }
            break
        }
    }
    switch (r) {
    case m.address:
        k = q.substring(p, o);
        break;
    case m.name:
        if (q[o - 1] == '"' || q[o - 1] == ">") {
            l = q.substring(p, o - 1)
        } else {
            l = q.substring(p, o)
        }
        break
    }
    j();
    r = m.none;
    return n
};
ISQ.Data.createDelimitedEmailAddressList = function(o) {
    var n = new Array();
    var j = new Array();
    var l = new Array();
    try {
        l = ISQ.Data.parseEmailAddress(o) || new Array()
    } catch (e) {
        j.push(o)
    }
    for (var k = 0; k < l.length; k++) {
        var m = l[k];
        if (!m.address ||!m.isValid) {
            j.push(m);
            continue
        }
        if (n.length > 0) {
            n.push(ISQ.Data.ADDRESS_DELIMITER)
        }
        n.push(m.asString())
    }
    return {
        list: n.join(""),
        errors: j
    }
};
ISQ.Data.htmlEncode = function(d) {
    var c = createElement("div");
    c.innerHTML = d;
    if (typeof(c.innerText) !== "undefined") {
        return c.innerText.Trim()
    } else {
        if (typeof(c.textContent) !== "undefined") {
            return c.textContent.Trim()
        }
    }
    return c.innerHTML.Trim()
};
ISQ.Data.htmlDecode = function(d) {
    var c = createElement("div");
    c.innerHTML = d;
    return c.innerHTML.Trim()
};
ISQ.Data.getTextFromHtml = function(d) {
    var c = createElement("div");
    c.innerHTML = d;
    return ($(c).text()).Trim()
};
ISQ.Data.stripHtml = function(k, n, l) {
    var q = 0, p =- 1, m = 0;
    var j = new Array();
    var o;
    n = n || "";
    q = k.indexOf("<" + n, q);
    while (q>-1) {
        j.push(k.substr(m, q - m));
        p = k.indexOf(">", q);
        m = q + 1;
        if (p!=-1) {
            o = k.substr(q + 1, p - q - 1).toLowerCase();
            if (o.startsWith("br") || o.startsWith("p")) {
                if (l) {
                    j.push("<br/>")
                } else {
                    j.push("\n ")
                }
            }
            m = p + 1
        } else {
            j.push("<")
        }
        q = k.indexOf("<", m)
    }
    if (m != k.length) {
        j.push(k.substr(m, k.length - m))
    }
    return j.join("")
};
ISQ.Data.unHTML = function(b) {
    if (!b) {
        return ""
    }
    b = b.replace(/\</g, "&lt;");
    b = b.replace(/\>/g, "&gt;");
    b = b.replace(/&lt;wbr\/&gt;/g, "<wbr/>");
    return b
};
ISQ.Data.toHTML = function(b) {
    if (!b) {
        return ""
    }
    b = b.replace(/&lt;/g, "<");
    b = b.replace(/&gt;/g, ">");
    return b
};
ISQ.Data.escapeJS = function(b) {
    if (!b) {
        return ""
    }
    b = b.replace(/\'/g, "\\'");
    return b
};
ISQ.Data.processHTMLstring = function(Q, af) {
    af = af || {};
    var ag = {
        text: "",
        script: ""
    };
    if (!Q) {
        return ag
    }
    var al = Q.indexOf("<");
    if (al===-1) {
        if (af.decodeEntities) {
            Q = ISQ.Data.htmlEncode(Q)
        }
        if (af.wordBreak) {
            ag.text = ISQ.Data.breakWords(Q)
        } else {
            ag.text = Q
        }
        return ag
    }
    var R =- 1;
    var ab = 0;
    var ac =- 1;
    var ai = new Array();
    var ah = null;
    var M;
    do {
        if (ab !== al) {
            M = Q.substring(ab, al);
            if (af.decodeEntities) {
                M = ISQ.Data.htmlEncode(M)
            }
            if (af.wordBreak) {
                ISQ.Data.breakWords(M, ai, true)
            } else {
                ai.push(M)
            }
        }
        ac = al;
        R = Q.indexOf(">", al + 1);
        if (R===-1) {
            break
        }
        var aa = "";
        var an = Q.charAt(al + 1);
        if (Q.charAt(al + 2) === " " && (an === "A" || an === "a")) {
            R = Q.indexOf("</a>", al);
            if (R===-1) {
                R = Q.indexOf("</A>", al);
                if (R===-1) {
                    break
                }
            }
            R += 4;
            var am = Q.substring(al, R);
            var K = createDiv(null);
            K.innerHTML = am;
            var Y = K.firstChild;
            if (K.parentNode != null) {
                K.parentNode.removeChild(K)
            }
            K = null;
            if (Y.nodeName !== "A") {
                break
            }
            var L = Y.getAttribute("nanoreplinkid");
            if (L && af.questionTagFunction) {
                af.questionTagFunction(ai, L, Y.innerHTML, am, af.questionTagCustomParams || null)
            } else {
                var X = Y.getAttribute("href");
                if (X && X != "javascript:void(0)" && X.charAt(0) != "#" && af.allowLinksInSameWindow !== true) {
                    var H;
                    if (af.url&&!X.startsWith("\\") && ((H = X.indexOf(":/"))===-1) && ((H = X.indexOf(":\\"))===-1)&&!X.startsWith("mailto:")) {
                        if (X.startsWith("/")) {
                            var ak = 0;
                            if (af.url.startsWith("https://")) {
                                ak += 8
                            } else {
                                if (af.url.startsWith("http://")) {
                                    ak += 7
                                } else {
                                    throw "ISQ.Data.processHTMLstring: invalid options.url"
                                }
                            }
                            var P = af.url.substring(0, af.url.indexOf("/", ak));
                            X = P + X
                        } else {
                            var ad = af.url.indexOf("?");
                            var ae = af.url.indexOf("#");
                            var T;
                            if (ad===-1) {
                                if (ae===-1) {
                                    T = af.url.length - 1
                                } else {
                                    T = ae
                                }
                            } else {
                                if (ae===-1) {
                                    T = ad
                                } else {
                                    T = Math.min(ad, ae)
                                }
                            }
                            var S = af.url.substring(0, af.url.LastIndexOf("/", T) + 1);
                            X = S + X
                        }
                        Y.setAttribute("href", X)
                    }
                    var J = Y.getAttribute("target");
                    switch (J) {
                    case"blank":
                    case"top":
                    case"_blank":
                    case"_top":
                    case"_parent":
                    case"parent":
                        break;
                    default:
                        Y.setAttribute("target", "_blank");
                        break
                    }
                } else {
                    if (X && X.charAt(0) == "#" && af.namedAnchorFunction) {
                        Y.setAttribute("onclick", "return " + af.namedAnchorFunction + "(this.hash)")
                    }
                }
                aa = elementToString(Y, ISQ.Data.processHTMLstring(Y.innerHTML, af).text)
            }
        } else {
            if (Q.equals("table ", al + 1) || Q.equals("table>", al + 1)) {
                R = Q.indexOf("</table>", al);
                if (R===-1) {
                    R = Q.indexOf("</TABLE>", al);
                    if (R===-1) {
                        break
                    }
                }
                R += 8;
                var am = Q.substring(al, R);
                var K = createDiv(null);
                K.innerHTML = am;
                var Z = K.firstChild;
                if (K.parentNode != null) {
                    K.parentNode.removeChild(K)
                }
                K = null;
                if (Z.nodeName !== "TABLE") {
                    break
                }
                for (var aj in Z.rows) {
                    var O = Z.rows[aj];
                    for (var I in O.childNodes) {
                        var N = O.cells.length;
                        if (O.childNodes[I].nodeName != "TD") {
                            continue
                        }
                        O.childNodes[I].setAttribute("width", 100 / N + "%");
                        O.childNodes[I].style.width = 100 / N + "%"
                    }
                }
                var U = Z.getAttribute("width");
                Z.setAttribute("width", "100%");
                Z.style.width = "100%";
                Z.style.tableLayout = "fixed";
                aa = elementToString(Z, ISQ.Data.processHTMLstring(Z.innerHTML, af).text)
            } else {
                if (Q.equals("script ", al + 1) || Q.equals("script>", al + 1) && af.doNotProcessScriptTag !== true) {
                    var W = Q.indexOf(">", al + 6);
                    if (W===-1) {
                        break
                    }
                    var V = Q.indexOf("</", W);
                    if (V===-1) {
                        break
                    }
                    if (ah === null) {
                        ah = new Array()
                    }
                    ah.push(Q.substring(W + 1, V));
                    ah.push(";");
                    R = Q.indexOf(">", V + 2);
                    if (R===-1) {
                        break
                    }
                    ++R
                } else {
                    aa = Q.substring(al, ++R)
                }
            }
        }
        ai.push(aa);
        ab = R;
        al = Q.indexOf("<", ab)
    }
    while (al!==-1);
    if (R===-1) {
        M = Q.substring(ac);
        if (af.decodeEntities) {
            M = ISQ.Data.htmlEncode(M)
        }
        if (af.wordBreak) {
            ISQ.Data.breakWords(M, ai, true)
        } else {
            ai.push(M)
        }
    } else {
        M = Q.substring(ab);
        if (af.decodeEntities) {
            M = ISQ.Data.htmlEncode(M)
        }
        if (af.wordBreak) {
            ISQ.Data.breakWords(M, ai, true)
        } else {
            ai.push(M)
        }
    }
    ag.text = ai.join("");
    if (ah) {
        ag.script = ah.join("")
    }
    return ag
};
ISQ.Data.getDBAnswerAsHtml = function(o, n, l, k, j, m) {
    try {
        return ISQ.Data.processHTMLstring(o, {
            wordBreak: n !== true,
            questionTagFunction: l,
            questionTagCustomParams: k,
            namedAnchorFunction: j,
            decodeEntities: false,
            url: m
        })
    } catch (e) {
        if (typeof(_debug) === "function") {
            _debug().insertText("ISQ.Data.getDBAnswerAsHtml exception: " + o)
        }
        return {
            text: "",
            error: e
        }
    }
};
ISQ.Data.breakWords = function(m, l, n) {
    if (!m) {
        if (l) {
            return 
        }
        return ""
    }
    var k = new ISQ.Data.EntityParser(m);
    var j = {};
    var g;
    if (l) {
        g = l
    } else {
        g = new Array()
    }
    while (k.getNextSegment(j)) {
        ISQ.Data.breakWords_internal(j.segment, g, n);
        if (j.entity) {
            g.push(j.entity)
        }
    }
    k.dispose();
    if (l) {
        return 
    }
    return g.join("")
};
ISQ.Data.breakWords_internal = function(p, o, q) {
    if (!p ||!o) {
        return 
    }
    var r = "<wbr/>";
    if (ISQ.Http.browser.full.startsWith("safari3") || ISQ.Http.browser.isOpera || ISQ.Http.browser.isIE8 || ISQ.Http.browser.isIE9 || ISQ.Http.browser.isIE10) {
        r = "&#8203;"
    }
    if (q && ISQ.Http.browser.isIE8) {
        r = "&shy;"
    }
    var m = p.length;
    var s = 0;
    var n =- 1;
    var k = false;
    var l = 0;
    for (l = 0; l < m; ++l) {
        if (p.charAt(l) === " ") {
            s = 0
        } else {
            if (s >= 20 && s%2 === 0) {
                o.push(p.substring(n + 1, l + 1));
                n = l;
                o.push(r);
                ++s
            } else {
                ++s
            }
        }
    }
    o.push(p.substring(n + 1))
};
ISQ.Data.getTextDirAlign = function(n) {
    if (typeof(n) !== "string") {
        throw ("getTextDirAlign: no txt")
    }
    var l = {
        align: "left",
        dir: "ltr",
        alignOpp: "right"
    };
    var k = 0;
    var m = 0;
    for (var j = 0; j < n.length; ++j) {
        var g = n.charCodeAt(j);
        if (g >= 48 && g <= 57) {
            continue
        }
        if (g >= 1456 && g <= 1524) {
            ++m
        } else {
            ++k
        }
    }
    if (m >= k) {
        l.align = "right";
        l.dir = "rtl";
        l.alignOpp = "left"
    }
    return l
};
ISQ.Data.getURLencodedString = function(b) {
    if (!b) {
        return ""
    }
    return escape(ISQ.Tools.Base64._utf8_encode(b))
};
ISQ.Data.getURLdecodedString = function(b) {
    if (!b) {
        return ""
    }
    return ISQ.Tools.Base64._utf8_decode(unescape(b))
};
ISQ.Data.EntityParser = ISQ.Base.extend({
    name: "ISQ.Data.EntityParser"
});
ISQ.Data.EntityParser.prototype.ctor = function(b) {
    if (!b) {
        throw ("ISQ.Data.EntityParser: no txt")
    }
    this.mOriginalText = b;
    this.mIndex = 0
};
ISQ.Data.EntityParser.prototype.dtor = function() {};
ISQ.Data.EntityParser.prototype.getNextSegment = function(g) {
    if (this.mIndex == this.mOriginalText.length) {
        return false
    }
    var f = this.mOriginalText.indexOf("&", this.mIndex);
    var e = this.mOriginalText.indexOf(";", f);
    if (f===-1 || e===-1 || e - f > 8) {
        g.entity = null;
        g.segment = this.mOriginalText.substring(this.mIndex);
        this.mIndex = this.mOriginalText.length
    } else {
        if (f!==-1) {
            g.entity = this.mOriginalText.substring(f, ++e);
            var j = this.mOriginalText.substring(this.mIndex, f);
            g.segment = j;
            this.mIndex = e
        }
    }
    return true
};
ISQ.Data.EntityParser.prototype.mOriginalText;
ISQ.Data.EntityParser.prototype.mLastEntityFound = null;
ISQ.Data.EntityParser.prototype.mAlteredText;
ISQ.Data.EntityParser.prototype.mIndex = 0;
ISQ.Data.EntityParser.prototype.mNextIndex =- 1;
ISQ.Data.decodeEntities = function(d) {
    if (typeof(jQuery) === "function") {
        return $("<div/>").html(d).text()
    }
    try {
        if (!ISQ.Data.entityDecoderElement) {
            ISQ.Data.entityDecoderElement = createElement("textarea", document.body, "display:none;visibility:hidden;position:absolute;top:100px;left:13px;")
        }
        ISQ.Data.entityDecoderElement.innerHTML = ISQ.Data.unHTML(d);
        return ISQ.Data.entityDecoderElement.value
    } catch (c) {
        return d
    }
};
ISQ.Data.Xml = initializeNS("ISQ.Data.Xml");
ISQ.Data.Xml.toXML = function(g) {
    var j = null;
    if (ISQ.Http.browser.app === "ie") {
        try {
            j = new ActiveXObject("Microsoft.XMLDOM");
            j.async = "false";
            j.loadXML(g)
        } catch (e) {
            j = null
        }
    }
    if (j == null) {
        try {
            var f = new DOMParser();
            j = f.parseFromString(g, "text/xml");
            f = null
        } catch (e) {
            j = null
        }
    }
    return j
};
ISQ.Data.Xml.getChildsByName = function(l, j) {
    if (!l) {
        return null
    }
    var g = l.childNodes.length;
    var k = null;
    for (var f = 0; f < g; ++f) {
        if (l.childNodes[f].nodeName === j) {
            if (k === null) {
                k = new Array()
            }
            k.push(l.childNodes[f])
        }
    }
    return k
};
ISQ.Data.Xml.downloadXML = function(f, d) {
    f = f || null;
    d = d || null;
    if (!f || typeof(d) !== "function") {
        return 
    }
    var e = null;
    if (ISQ.Http.browser.app === "ie"&&!ISQ.Http.browser.isIE11) {
        e = new ActiveXObject("Microsoft.XMLDOM");
        e.async = true;
        e.onreadystatechange = function() {
            if (e.readyState === 4) {
                d(e)
            }
        };
        e.load(f)
    } else {
        e = new XMLHttpRequest();
        if (ISQ.Http.browser.app === "ff" && (ISQ.Http.browser.version === "3" || ISQ.Http.browser.version === "3.6")) {
            e.onload = function() {
                var b = e.responseText;
                var a = new DOMParser();
                var c = a.parseFromString(b, "text/xml");
                d(c)
            }
        } else {
            e.onreadystatechange = function() {
                if (e.readyState === 4) {
                    var b = e.responseText;
                    var a = new DOMParser();
                    var c = a.parseFromString(b, "text/xml");
                    d(c)
                }
            }
        }
        e.open("GET", f, true);
        e.send(null)
    }
};
ISQ.Data.Xml.getFirstElementByName = function(l, j) {
    if (!l) {
        return null
    }
    if (l.nodeName === j) {
        return l
    }
    var f = l.childNodes;
    if (f.length === 0) {
        return null
    }
    for (var g = 0; g < f.length; ++g) {
        var k = ISQ.Data.Xml.getFirstElementByName(f[g], j);
        if (k !== null) {
            return k
        }
    }
    return null
};
ISQ.Tools.FileLoader = initializeNS("ISQ.Tools.FileLoader");
ISQ.Tools.FileLoader.ajaxPool = null;
ISQ.Tools.FileLoader.getAjaxPool = function() {
    if (ISQ.Tools.FileLoader.ajaxPool === null) {
        var b = (ISQ.Http.browser.isIE6 || ISQ.Http.browser.isIE7) ? 1: 2;
        ISQ.Tools.FileLoader.ajaxPool = new Ajax.Pool("/kb", b, Ajax.methodEnum.GET, Ajax.typeEnum.ASYNC, 10000);
        ISQ.Tools.FileLoader.ajaxPool.incomingResponseEvent.addHandler(ISQ.Tools.FileLoader.loadResponse);
        if (typeof("sessionGC") === "function") {
            sessionGC().add(ISQ.Tools.FileLoader.ajaxPool)
        }
    }
    return ISQ.Tools.FileLoader.ajaxPool
};
ISQ.Tools.FileLoader.load = function(m, j, n) {
    if (j&&!ISQ.Base.isDerivedOf(j, ISQ.Handler)) {
        j = new ISQ.Handler(j)
    }
    var l = m.indexOf("?");
    var k;
    if (l===-1) {
        k = m
    } else {
        k = m.substr(0, l)
    }
    var g = ISQ.Tools.FileLoader.getAjaxPool();
    g.add(null, m, new Ajax.Token({
        callback: j,
        tag: n
    }))
};
ISQ.Tools.FileLoader.loadResponse = function(response) {
    try {
        if (response.request.mUrl.endsWith(".css")) {
            ISQ.CSS.addNewStyle(response.text)
        } else {
            if (response.request.mUrl.endsWith(".js")) {
                eval(response.text)
            }
        }
    } catch (ex) {
        _debug().insertText("ISQ.Tools.FileLoader.loadResponse: Error loading file ( " + response.request.mUrl + ").\n" + ex)
    }
    if (!response.token ||!response.token.callback) {
        return 
    }
    var tag = response.token.tag;
    response.token.callback.handle(tag, response);
    response.token.callback.dispose();
    response.token.callback = null
};
ISQ.Tools.appendScript = function(f, d) {
    var e = document.createElement("script");
    e.type = "text/javascript";
    e.async = true;
    e.defer = true;
    if (f instanceof Array) {
        f = f.join("")
    }
    if (d) {
        if (ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE11) {
            e.onload = d
        } else {
            e.onreadystatechange = function() {
                if (this.readyState !== "complete" && this.readyState !== "loaded") {
                    return
                }
                d()
            }
        }
    }
    e.src = f;
    document.getElementsByTagName("head")[0].appendChild(e)
};
ISQ.Tools.Time = initializeNS("ISQ.Tools.Time");
ISQ.Tools.Time.ticksDiff = 62135596800000;
ISQ.Tools.Time.toClientTicks = function(b) {
    return b - ISQ.Tools.Time.ticksDiff
};
ISQ.Tools.Time.toServerTicks = function(b) {
    if (b===-1) {
        return b
    }
    return b + ISQ.Tools.Time.ticksDiff
};
ISQ.Tools.Time.secondsInTime = function(g, e, j) {
    var f = {};
    g = parseInt(g);
    j = j || 86400;
    f.days = parseInt(g / j);
    if (e) {
        f.days = ISQ.Tools.getPrettyNumber(f.days, 0)
    }
    g = g%j;
    f.hours = parseInt(g / 3600);
    g = g%3600;
    f.minutes = parseInt(g / 60);
    f.seconds = g%60;
    return f
};
ISQ.Tools.Time.stringToSpan = function(q) {
    if (!q.contains(":")) {
        throw ("ISQ.Tools.Time.stringToSpan: Incorrect string format")
    }
    var p = q.split(":");
    if (p.length > 4) {
        throw ("ISQ.Tools.Time.stringToSpan: Incorrect string format")
    }
    var k = 0, l = 0, n = 0, o = 0;
    for (var m = 0; m < p.length; ++m) {
        if (p[m].startsWith("0")) {
            var j = 0;
            while (p[m].charAt(j) === "0") {
                ++j
            }
            if (j !== p[m].length) {
                p[m] = p[m].substring(j)
            }
        }
    }
    if (p.length == 4) {
        k = parseInt(p[0]);
        l = parseInt(p[1]);
        n = parseInt(p[2]);
        o = parseInt(p[3])
    }
    if (p.length == 3) {
        l = parseInt(p[0]);
        n = parseInt(p[1]);
        o = parseInt(p[2])
    }
    if (p.length == 2) {
        l = parseInt(p[0]);
        n = parseInt(p[1])
    }
    return {
        days: k,
        hours: l,
        minutes: n,
        seconds: o
    }
};
ISQ.Tools.Time.drawSLATime = function(g, f) {
    var j = new Array();
    var e = false;
    ISQ.Tools.Time.appendTimeToStringBuilder(g, j);
    if (j.length === 0) {
        j.push(" N/A ")
    }
    f.innerHTML = j.join("")
};
ISQ.Tools.Time.appendTimeToStringBuilder = function(l, k, f) {
    f = f !== false;
    var g = k.length;
    var j = false;
    if (l.days > 0) {
        k.push(l.days);
        k.push("d");
        f = false
    }
    if (l.hours != 0 || l.days > 0) {
        if (k.length !== g) {
            k.push(" ")
        }
        k.push(l.hours);
        k.push("h");
        f = false;
        j = true
    }
    if (l.days === 0 && (l.minutes > 0 || j)) {
        if (k.length !== g) {
            k.push(" ")
        }
        k.push(l.minutes);
        k.push("m")
    }
    if ((f && l.seconds > 0) || k.length === g) {
        if (k.length !== g) {
            k.push(" ")
        }
        k.push(l.seconds);
        k.push("s")
    }
};
ISQ.Tools.Time.elapsedTimeTicker = function() {
    var b = {
        build: function(a) {
            this.startTime = a.startTime;
            this.endTime = a.endTime;
            this.parent = a.parent;
            this.interval = null;
            this.timerElapsed();
            if (a.start) {
                this.start()
            }
            return this
        },
        start: function() {
            var a = this;
            this.interval = setInterval(function() {
                a.timerElapsed.apply(a)
            }, 1000)
        },
        stop: function() {
            clearInterval(this.interval);
            this.interval = null
        },
        running: function() {
            return this.interval != null
        },
        timerElapsed: function() {
            var d = this.endTime || new Date();
            if (d < this.startTime) {
                this.startTime = d
            }
            var a = new Date(d - this.startTime);
            this.parent.innerHTML = a.ToString("%H:%M:%S", null, true)
        }
    };
    return b
};
ISQ.Tools.Base64 = {
    _keyStr: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=",
    encode: function(t) {
        var u = "";
        var l, m, n, o, p, q, r;
        var s = 0;
        while (s < t.length) {
            l = t.charCodeAt(s++);
            m = t.charCodeAt(s++);
            n = t.charCodeAt(s++);
            o = l>>2;
            p = ((l & 3)<<4) | (m>>4);
            q = ((m & 15)<<2) | (n>>6);
            r = n & 63;
            if (isNaN(m)) {
                q = r = 64
            } else {
                if (isNaN(n)) {
                    r = 64
                }
            }
            u = u + this._keyStr.charAt(o) + this._keyStr.charAt(p) + this._keyStr.charAt(q) + this._keyStr.charAt(r)
        }
        return u
    },
    decode: function(t) {
        var u = "";
        var l, m, n;
        var o, p, q, r;
        var s = 0;
        t = t.replace(/[^A-Za-z0-9\+\/\=]/g, "");
        while (s < t.length) {
            o = this._keyStr.indexOf(t.charAt(s++));
            p = this._keyStr.indexOf(t.charAt(s++));
            q = this._keyStr.indexOf(t.charAt(s++));
            r = this._keyStr.indexOf(t.charAt(s++));
            l = (o<<2) | (p>>4);
            m = ((p & 15)<<4) | (q>>2);
            n = ((q & 3)<<6) | r;
            u = u + String.fromCharCode(l);
            if (q != 64) {
                u = u + String.fromCharCode(m)
            }
            if (r != 64) {
                u = u + String.fromCharCode(n)
            }
        }
        return u
    },
    _utf8_encode: function(g) {
        g = g.replace(/\r\n/g, "\n");
        var j = "";
        for (var f = 0; f < g.length; f++) {
            var c = g.charCodeAt(f);
            if (c < 128) {
                j += String.fromCharCode(c)
            } else {
                if ((c > 127) && (c < 2048)) {
                    j += String.fromCharCode((c>>6) | 192);
                    j += String.fromCharCode((c & 63) | 128)
                } else {
                    j += String.fromCharCode((c>>12) | 224);
                    j += String.fromCharCode(((c>>6) & 63) | 128);
                    j += String.fromCharCode((c & 63) | 128)
                }
            }
        }
        return j
    },
    _utf8_decode: function(j) {
        var g = "";
        var f = 0;
        var c = c1 = c2 = 0;
        while (f < j.length) {
            c = j.charCodeAt(f);
            if (c < 128) {
                g += String.fromCharCode(c);
                f++
            } else {
                if ((c > 191) && (c < 224)) {
                    c2 = j.charCodeAt(f + 1);
                    g += String.fromCharCode(((c & 31)<<6) | (c2 & 63));
                    f += 2
                } else {
                    c2 = j.charCodeAt(f + 1);
                    c3 = j.charCodeAt(f + 2);
                    g += String.fromCharCode(((c & 15)<<12) | ((c2 & 63)<<6) | (c3 & 63));
                    f += 3
                }
            }
        }
        return g
    },
    encodeString: function(j) {
        if (!j) {
            return ""
        }
        var e = false;
        for (var f = 0; f < j.length&&!e; ++f) {
            if (j.charCodeAt(f) > 127) {
                e = true
            }
        }
        var g;
        if (e) {
            j = ISQ.Tools.Base64._utf8_encode(j)
        }
        return ISQ.Tools.Base64.encode(j)
    },
    decodeString: function(e) {
        var j = ISQ.Tools.Base64.decode(e);
        var f = false;
        for (var g = 0; g < j.length&&!f; ++g) {
            if (j.charCodeAt(g) > 127) {
                f = true
            }
        }
        if (f) {
            j = ISQ.Tools.Base64._utf8_decode(j)
        }
        return j
    }
};
ISQ.Tools.BigUInt256 = null;
ISQ.Tools.ConvertFromByteArray = function(l, s, n) {
    if (ISQ.Tools.BigUInt256 === null) {
        ISQ.Tools.BigUInt256 = new ISQ.Tools.BigUInt("256")
    }
    var o;
    var k;
    switch (s) {
    case"Int64":
        o = n + 8;
        break;
    case"Int32":
        o = n + 4;
        break;
    default:
        throw ("ISQ.Tools.ConvertFromByteArray: invalid type");
        break
    }
    if (l.length < o) {
        throw ("ISQ.Tools.ConvertFromByteArray: invalid offset")
    }
    var k = new ISQ.Tools.BigUInt("0");
    for (var m = o - 1; m >= n; --m) {
        var p = k.multiply(ISQ.Tools.BigUInt256);
        k = p;
        var r = new ISQ.Tools.BigUInt(l.charCodeAt(m));
        p = k.add(r);
        k = p
    }
    var q = k.toString();
    return q
};
ISQ.Tools.ConvertToByteArray = function(p, q) {
    if (ISQ.Tools.BigUInt256 === null) {
        ISQ.Tools.BigUInt256 = new ISQ.Tools.BigUInt("256")
    }
    var k = "";
    var j;
    var m = 0;
    switch (q) {
    case"Int64":
        m = 8;
        break;
    case"Int32":
        m = 4;
        break;
    default:
        throw ("ISQ.Tools.ConvertFromByteArray: invalid type");
        break
    }
    var j = new ISQ.Tools.BigUInt(p);
    for (var l = 0; l < m; ++l) {
        var n = {};
        var o = j.divide(ISQ.Tools.BigUInt256, n);
        j = o;
        k = k + String.fromCharCode(n.value.toString())
    }
    return k
};
ISQ.Tools.getHex = function(p) {
    var j = new ISQ.Tools.BigUInt(16);
    var l = new ISQ.Tools.BigUInt(p);
    var o = new Array();
    var n = {
        value: null
    };
    var k = 0;
    while (true&&++k <= 100) {
        l = l.divide(j, n);
        var m = n.value.toString();
        switch (m) {
        case"":
            o.push("0");
            break;
        case"0":
        case"1":
        case"2":
        case"3":
        case"4":
        case"5":
        case"6":
        case"7":
        case"8":
        case"9":
            o.push(m);
            break;
        case"10":
            o.push("A");
            break;
        case"11":
            o.push("B");
            break;
        case"12":
            o.push("C");
            break;
        case"13":
            o.push("D");
            break;
        case"14":
            o.push("E");
            break;
        case"15":
            o.push("F");
            break
        }
        var m = l.toString();
        if (!m || m === "0") {
            break
        }
    }
    o.reverse();
    return o.join("")
};
ISQ.Tools.compareArrays_flat = function(e, f, j) {
    if (e === f) {
        return true
    }
    if (!e&&!f) {
        return true
    }
    if (!e ||!f) {
        return false
    }
    if (e.length !== f.length) {
        return false
    }
    for (var g = 0; g < e.length; ++g) {
        if (e[g] !== f[g]) {
            return false
        }
    }
    return true
};
ISQ.Tools.compareArrays = function(j, k, o) {
    if (j === k) {
        return true
    }
    if (!j&&!k) {
        return true
    }
    if (!j ||!k) {
        return false
    }
    if (j.length !== k.length) {
        return false
    }
    for (var m in j) {
        var p = j[m];
        var l = false;
        for (var n in k) {
            if (k[n] === p) {
                l = true;
                break
            }
        }
        if (!l) {
            return false
        }
    }
    if (o) {
        return true
    }
    return ISQ.Tools.compareArrays(k, j, true)
};
ISQ.Tools.reverseArray = function(g) {
    var l = g.length;
    for (var j = 0; j < g.length; ++j) {
        var m = 0;
        for (var k = 1; k < l; ++k) {
            var n = g[m];
            g[m] = g[k];
            g[k] = n;
            ++m
        }
        --l
    }
};
ISQ.Tools.disposeArray = function(c) {
    if (!c) {
        return 
    }
    for (var d = 0; d < c.length; ++d) {
        if (!c[d]) {
            continue
        }
        if (c[d] instanceof Array) {
            ISQ.Tools.disposeArray(c[d]);
            continue
        }
        if (typeof(c[d].dispose) === "function") {
            c[d].dispose()
        }
        c[d] = null;
        delete c[d]
    }
};
ISQ.Tools.isInCSVList = function(f, e) {
    if (typeof(f) !== "string" || typeof(e) !== "string") {
        return false
    }
    if (!f ||!e) {
        return false
    }
    e = e.Trim();
    f = f.Trim();
    var d = f.indexOf(e);
    if (d==-1) {
        return false
    }
    if (d > 0 && f[d - 1] != ",") {
        return false
    }
    if (d + e.length < f.length && f[d + e.length] != ",") {
        return false
    }
    return true
};
ISQ.Tools.Languages = {
    ar: "Arabic",
    bg: "Bulgarian",
    cs: "Czech",
    da: "Danish",
    de: "German",
    el: "Greek, Modern",
    en: "English",
    es: "Spanish",
    et: "Estonian",
    fi: "Finnish",
    fr: "French",
    hi: "Hindi",
    hu: "Hungarian",
    id: "Indonesian",
    it: "Italian",
    iw: "Hebrew",
    ja: "Japanese",
    ko: "Korean",
    lt: "Lithuanian",
    lv: "Latvian",
    nl: "Dutch",
    no: "Norwegian",
    pl: "Polish",
    pt: "Portuguese",
    ro: "Romanian, Moldavian, Moldovan",
    ru: "Russian",
    sk: "Slovak",
    sl: "Slovene",
    sv: "Swedish",
    th: "Thai",
    tr: "Turkish",
    uk: "Ukrainian",
    vi: "Vietnamese",
    yi: "Yiddish",
    zh: "Chinese"
};
ISQ.Tools.LanguageSynonym = {
    "zh-CHS": "Chinese",
    "zh-CHT": "Chinese",
    "zh-CN": "Chinese",
    he: "Hebrew"
};
ISQ.Tools.UniformLanguageCode = {
    he: "iw",
    "zh-CHS": "zh",
    "zh-CHT": "zh",
    "zh-CN": "zh"
};
ISQ.Tools.CountryNames = {
    AF: "Afghanistan",
    AX: "Aland Islands",
    AL: "Albania",
    DZ: "Algeria",
    AS: "American Samoa",
    AD: "Andorra",
    AO: "Angola",
    AI: "Anguilla",
    AQ: "Antarctica",
    AG: "Antigua And Barbuda",
    AR: "Argentina",
    AM: "Armenia",
    AW: "Aruba",
    AU: "Australia",
    AT: "Austria",
    AZ: "Azerbaijan",
    BS: "Bahamas",
    BH: "Bahrain",
    BD: "Bangladesh",
    BB: "Barbados",
    BY: "Belarus",
    BE: "Belgium",
    BZ: "Belize",
    BJ: "Benin",
    BM: "Bermuda",
    BT: "Bhutan",
    BO: "Bolivia",
    BA: "Bosnia And Herzegovina",
    BW: "Botswana",
    BV: "Bouvet Island",
    BR: "Brazil",
    IO: "British Indian Ocean Territory",
    BN: "Brunei Darussalam",
    BG: "Bulgaria",
    BF: "Burkina Faso",
    BI: "Burundi",
    KH: "Cambodia",
    CM: "Cameroon",
    CA: "Canada",
    CV: "Cape Verde",
    KY: "Cayman Islands",
    CF: "Central African Republic",
    TD: "Chad",
    CL: "Chile",
    CN: "China",
    CX: "Christmas Island",
    CC: "Cocos (Keeling) Islands",
    CO: "Colombia",
    KM: "Comoros",
    CG: "Congo",
    CD: "Congo, Democratic Republic",
    CK: "Cook Islands",
    CR: "Costa Rica",
    CI: "Cote D'Ivoire",
    HR: "Croatia",
    CU: "Cuba",
    CY: "Cyprus",
    CZ: "Czech Republic",
    DK: "Denmark",
    DJ: "Djibouti",
    DM: "Dominica",
    DO: "Dominican Republic",
    EC: "Ecuador",
    EG: "Egypt",
    SV: "El Salvador",
    GQ: "Equatorial Guinea",
    ER: "Eritrea",
    EE: "Estonia",
    ET: "Ethiopia",
    FK: "Falkland Islands (Malvinas)",
    FO: "Faroe Islands",
    FJ: "Fiji",
    FI: "Finland",
    FR: "France",
    GF: "French Guiana",
    PF: "French Polynesia",
    TF: "French Southern Territories",
    GA: "Gabon",
    GM: "Gambia",
    GE: "Georgia",
    DE: "Germany",
    GH: "Ghana",
    GI: "Gibraltar",
    GR: "Greece",
    GL: "Greenland",
    GD: "Grenada",
    GP: "Guadeloupe",
    GU: "Guam",
    GT: "Guatemala",
    GG: "Guernsey",
    GN: "Guinea",
    GW: "Guinea-Bissau",
    GY: "Guyana",
    HT: "Haiti",
    HM: "Heard Island & Mcdonald Islands",
    VA: "Holy See (Vatican City State)",
    HN: "Honduras",
    HK: "Hong Kong",
    HU: "Hungary",
    IS: "Iceland",
    IN: "India",
    ID: "Indonesia",
    IR: "Iran, Islamic Republic Of",
    IQ: "Iraq",
    IE: "Ireland",
    IM: "Isle Of Man",
    IL: "Israel",
    IT: "Italy",
    JM: "Jamaica",
    JP: "Japan",
    JE: "Jersey",
    JO: "Jordan",
    KZ: "Kazakhstan",
    KE: "Kenya",
    KI: "Kiribati",
    KR: "Korea",
    KW: "Kuwait",
    KG: "Kyrgyzstan",
    LA: "Lao People's Democratic Republic",
    LV: "Latvia",
    LB: "Lebanon",
    LS: "Lesotho",
    LR: "Liberia",
    LY: "Libyan Arab Jamahiriya",
    LI: "Liechtenstein",
    LT: "Lithuania",
    LU: "Luxembourg",
    MO: "Macao",
    MK: "Macedonia",
    MG: "Madagascar",
    MW: "Malawi",
    MY: "Malaysia",
    MV: "Maldives",
    ML: "Mali",
    MT: "Malta",
    MH: "Marshall Islands",
    MQ: "Martinique",
    MR: "Mauritania",
    MU: "Mauritius",
    YT: "Mayotte",
    MX: "Mexico",
    FM: "Micronesia, Federated States Of",
    MD: "Moldova",
    MC: "Monaco",
    MN: "Mongolia",
    ME: "Montenegro",
    MS: "Montserrat",
    MA: "Morocco",
    MZ: "Mozambique",
    MM: "Myanmar",
    NA: "Namibia",
    NR: "Nauru",
    NP: "Nepal",
    NL: "Netherlands",
    AN: "Netherlands Antilles",
    NC: "New Caledonia",
    NZ: "New Zealand",
    NI: "Nicaragua",
    NE: "Niger",
    NG: "Nigeria",
    NU: "Niue",
    NF: "Norfolk Island",
    MP: "Northern Mariana Islands",
    NO: "Norway",
    OM: "Oman",
    PK: "Pakistan",
    PW: "Palau",
    PS: "Palestinian Territory, Occupied",
    PA: "Panama",
    PG: "Papua New Guinea",
    PY: "Paraguay",
    PE: "Peru",
    PH: "Philippines",
    PN: "Pitcairn",
    PL: "Poland",
    PT: "Portugal",
    PR: "Puerto Rico",
    QA: "Qatar",
    RE: "Reunion",
    RO: "Romania",
    RU: "Russian Federation",
    RW: "Rwanda",
    BL: "Saint Barthelemy",
    SH: "Saint Helena",
    KN: "Saint Kitts And Nevis",
    LC: "Saint Lucia",
    MF: "Saint Martin",
    PM: "Saint Pierre And Miquelon",
    VC: "Saint Vincent And Grenadines",
    WS: "Samoa",
    SM: "San Marino",
    ST: "Sao Tome And Principe",
    SA: "Saudi Arabia",
    SN: "Senegal",
    RS: "Serbia",
    SC: "Seychelles",
    SL: "Sierra Leone",
    SG: "Singapore",
    SK: "Slovakia",
    SI: "Slovenia",
    SB: "Solomon Islands",
    SO: "Somalia",
    ZA: "South Africa",
    GS: "South Georgia And Sandwich Isl.",
    ES: "Spain",
    LK: "Sri Lanka",
    SD: "Sudan",
    SR: "Suriname",
    SJ: "Svalbard And Jan Mayen",
    SZ: "Swaziland",
    SE: "Sweden",
    CH: "Switzerland",
    SY: "Syrian Arab Republic",
    TW: "Taiwan",
    TJ: "Tajikistan",
    TZ: "Tanzania",
    TH: "Thailand",
    TL: "Timor-Leste",
    TG: "Togo",
    TK: "Tokelau",
    TO: "Tonga",
    TT: "Trinidad And Tobago",
    TN: "Tunisia",
    TR: "Turkey",
    TM: "Turkmenistan",
    TC: "Turks And Caicos Islands",
    TV: "Tuvalu",
    UG: "Uganda",
    UA: "Ukraine",
    AE: "United Arab Emirates",
    GB: "United Kingdom",
    US: "United States",
    UM: "United States Outlying Islands",
    UY: "Uruguay",
    UZ: "Uzbekistan",
    VU: "Vanuatu",
    VE: "Venezuela",
    VN: "Viet Nam",
    VG: "Virgin Islands, British",
    VI: "Virgin Islands, U.S.",
    WF: "Wallis And Futuna",
    EH: "Western Sahara",
    YE: "Yemen",
    ZM: "Zambia",
    ZW: "Zimbabwe"
};
ISQ.Tools.getUniformLanguageCode = function(c) {
    if (!c) {
        return null
    }
    var d = ISQ.Tools.UniformLanguageCode[c];
    if (d) {
        return d
    }
    return c
};
ISQ.Tools.getLanguageName = function(d) {
    var c = ISQ.Tools.Languages[d];
    if (!c) {
        c = ISQ.Tools.LanguageSynonym[d]
    }
    if (!c) {
        return null
    }
    return c
};
ISQ.Tools.isLanguageRTL = function(b) {
    b = ISQ.Tools.getUniformLanguageCode(b);
    switch (b) {
    case"iw":
    case"ar":
    case"yi":
    case"ur":
        return true
    }
    return false
};
ISQ.Tools.HTML = initializeNS("ISQ.Tools.HTML");
ISQ.Tools.HTML.openPopup = function(p, l, q, j, n) {
    var k = (screen.width / 2) - (q / 2);
    var o = (screen.height / 2) - (j / 2);
    var m = n || "no";
    return window.open(p, l, "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=" + m + ", resizable=no, copyhistory=no, width=" + q + ", height=" + j + ", top=" + o + ", left=" + k)
};
ISQ.Tools.Storage = initializeNS("ISQ.Tools.Storage");
ISQ.Tools.Storage.HTML5 = false;
try {
    ISQ.Tools.Storage.HTML5 = window.localStorage !== null && typeof(window.localStorage) !== "undefined"
} catch (ex) {}
ISQ.Tools.Storage.get = function(c, d) {
    if (!ISQ.Tools.Storage.HTML5) {
        return null
    }
    if (d) {
        return window.sessionStorage.getItem(c)
    } else {
        return window.localStorage.getItem(c)
    }
};
ISQ.Tools.Storage.set = function(d, f, e) {
    if (!ISQ.Tools.Storage.HTML5) {
        return false
    }
    if (e) {
        window.sessionStorage.setItem(d, f)
    } else {
        window.localStorage.setItem(d, f)
    }
    return true
};
ISQ.Tools.Storage.remove = function(c, d) {
    if (!ISQ.Tools.Storage.HTML5) {
        return 
    }
    if (d) {
        window.sessionStorage.removeItem(c)
    } else {
        window.localStorage.removeItem(c)
    }
};
ISQ.Tools.Storage.clear = function(b) {
    if (!ISQ.Tools.Storage.HTML5) {
        return 
    }
    if (b) {
        window.sessionStorage.clear()
    } else {
        window.localStorage.clear()
    }
};
ISQ.Event = ISQ.Base.extend({
    name: "ISQ.Event"
});
ISQ.Event.prototype.ctor = function() {
    this.mCount = 0
};
ISQ.Event.prototype.dtor = function() {
    if (this.handlers !== null) {
        var d = this.handlers.length;
        for (var c = 0; c < d; ++c) {
            if (this.handlers[c] !== null) {
                this.handlers[c].dispose()
            }
        }
        delete this.handlers
    }
    this.handlers = null
};
ISQ.Event.prototype.handlers = null;
ISQ.Event.prototype.mCount = 0;
ISQ.Event.prototype.mRemovedHandler = false;
ISQ.Event.prototype.dispatch = function() {
    if (this.isDisposed()) {
        return
    }
    if (this.handlers === null) {
        return 
    }
    var j = this.handlers.length;
    for (var g = 0; g < j; ++g) {
        if (this.isDisposed()) {
            return 
        }
        var f = this.handlers[g];
        try {
            if (!f) {
                continue
            }
            if (f.isDisposed()) {
                this.mRemovedHandler = true;
                this.handlers[g] = null
            } else {
                f.handle.apply(f, arguments)
            }
        } catch (e) {
            if (debugLevel && typeof(ISQ.Debug) !== "undefined") {
                ISQ.Debug.insertText("Exception in event dispatch: " + e);
                ISQ.Debug.insertText("Handler:");
                ISQ.Debug.insertText("Func: " + f.func);
                ISQ.Debug.insertText("Context: " + f.context);
                ISQ.Debug.insertStackCallback();
                ISQ.Debug.insertText("Exception is rolled away");
                throw (e)
            }
        }
    }
    if (this.isDisposed()) {
        return 
    }
    if (this.mRemovedHandler) {
        this.cleanDisposedHandlers()
    }
};
ISQ.Event.prototype.cleanDisposedHandlers = function() {
    if (!this.mRemovedHandler) {
        return 
    }
    var d = this.handlers.length;
    for (var c = 0; c < d; ++c) {
        if (this.handlers[c] === null) {
            this.handlers.splice(c, 1);
            --c;
            --d
        }
    }
    this.mRemovedHandler = false
};
ISQ.Event.prototype.clear = function() {
    if (!this.handlers) {
        return 
    }
    var d = this.handlers.length;
    for (var c = 0; c < d; ++c) {
        this.handlers[c].dispose()
    }
    this.handlers = null
};
ISQ.Event.prototype.addHandlerObject = function(e) {
    if (this.isDisposed()) {
        return 
    }
    if (this.handlers === null) {
        this.handlers = new Array()
    }
    var f = this.handlers.length;
    for (i = 0; i < f; ++i) {
        var d = this.handlers[i];
        if (d !== null && d.func === e.func && d.context === e.context) {
            return 
        }
    }
    this.handlers.push(e);
    ++this.mCount;
    this.cleanDisposedHandlers();
    return e
};
ISQ.Event.prototype.addHandler = function(g, f) {
    if (this.isDisposed()) {
        return 
    }
    if (this.handlers === null) {
        this.handlers = new Array()
    }
    var l = this.handlers.length;
    for (i = 0; i < l; ++i) {
        var j = this.handlers[i];
        if (j !== null && j.func === g && j.context === f) {
            return 
        }
    }
    var k = new ISQ.Handler(g, f);
    this.handlers.push(k);
    ++this.mCount;
    g = null;
    f = null;
    this.cleanDisposedHandlers();
    return k
};
ISQ.Event.prototype.removeHandlerObject = function(f) {
    if (this.handlers === null) {
        return 
    }
    var j = this.handlers.length;
    var e = null;
    var g;
    for (g = 0; g < j; ++g) {
        if (this.handlers[g] === f) {
            --this.mCount;
            this.handlers[g] = null;
            break
        }
    }
    this.mRemovedHandler = true;
    f.dispose()
};
ISQ.Event.prototype.removeHandler = function(g, f) {
    if (this.handlers === null) {
        return
    }
    f = f || null;
    var l = this.handlers.length;
    var j = null;
    var k;
    for (k = 0; (k < l && j === null); ++k) {
        if (this.handlers[k] !== null && this.handlers[k].func === g && this.handlers[k].context === f) {
            j = this.handlers[k];
            this.handlers[k] = null;
            --this.mCount;
            break
        }
    }
    this.mRemovedHandler = true;
    if (j !== null) {
        j.dispose()
    }
    g = null;
    f = null
};
ISQ.Event.prototype.count = function() {
    return this.mCount
};
ISQ.Handler = ISQ.Base.extend({
    name: "ISQ.Handler"
});
ISQ.Handler.toHandler = function(b) {
    if (ISQ.Base.isDerivedOf(b, ISQ.Handler)) {
        return b
    }
    return new ISQ.Handler(b)
};
ISQ.Handler.prototype.ctor = function(d, c) {
    d = d || null;
    c = c || null;
    if (typeof(d) !== "function") {
        if (debugLevel > 1) {}
        throw ("Error creating event handler: no function is passed")
    }
    this.func = d;
    this.context = c;
    return this
};
ISQ.Handler.prototype.dtor = function() {
    delete this.func;
    delete this.context;
    this.func = null;
    this.context = null
};
ISQ.Handler.prototype.handle = function() {
    if (!this.func && debugLevel) {}
    return this.func.apply(this.context, arguments)
};
ISQ.Handler.prototype.handleArgumentArray = function() {
    if (arguments.length === 0) {
        throw ("invalid use of ISQ.Handler.handleArgumentArray")
    }
    if (typeof(arguments[0].length) !== "number") {
        throw ("invalid array in ISQ.Handler.handleArgumentArray")
    }
    return this.func.apply(this.context, ISQ.Tools.toArray(arguments[0]))
};
ISQ.Handler.prototype.func = null;
ISQ.Handler.prototype.context = null;
ISQ.Timer = ISQ.Base.extend({
    name: "ISQ.Timer"
});
ISQ.Timer.prototype.ctor = function(e, d) {
    if (!ISQ.Base.isDerivedOf(e, ISQ.Handler)) {
        throw ("ISQ.Timer.ctor: invalid handler")
    }
    this.mHandler = e;
    this.mDisposeAfterTimerElapsed = d !== false;
    this.mParams = new Array();
    if (arguments.length <= 2) {
        return 
    }
    for (var f = 2; f < arguments.length; ++f) {
        this.mParams.push(arguments[f])
    }
};
ISQ.Timer.prototype.dtor = function() {
    this.start = null;
    if (this.mTimer) {
        clearTimeout(this.mTimer);
        this.mTimer = null
    }
};
ISQ.Timer.prototype.start = function(e) {
    if (typeof(e) !== "number") {
        throw ("Timer interval illegal")
    }
    if (arguments.length > 1) {
        this.mParams = new Array();
        for (var d = 1; d < arguments.length; ++d) {
            this.mParams.push(arguments[d])
        }
    }
    var f = this;
    this.mTimer = setTimeout(function() {
        if (f.mTimer === null) {
            return
        }
        f.timerElapsed.apply(f, new Array())
    }, e)
};
ISQ.Timer.prototype.timerElapsed = function() {
    this.mHandler.handleArgumentArray(this.mParams);
    if (this.mDisposeAfterTimerElapsed) {
        this.dispose()
    }
};
ISQ.Timer.prototype.cancel = function() {
    if (this.mTimer === null) {
        return 
    }
    clearTimeout(this.mTimer);
    this.mTimer = null
};
ISQ.Timer.prototype.isRunning = function() {
    return this.mTimer !== null
};
ISQ.Timer.prototype.mDisposeAfterTimerElapsed = true;
ISQ.Timer.prototype.mTimer = null;
ISQ.Timer.prototype.mHandler = null;
ISQ.Timer.prototype.mParams = null;
ISQ.Http = initializeNS("ISQ.Http");
ISQ.Http.MAX_URL_SIZE = 2047;
ISQ.Http.getNoHashUrl = function() {
    var e = window.location.hash;
    var d = window.location.href;
    if (e !== "") {
        var f;
        do {
            f = d.indexOf(e);
            if (f!==-1) {
                d = d.substring(0, f)
            }
        }
        while (f!==-1)
        }
    return d
};
ISQ.Http.detectBroswer = function() {
    var k = navigator.appVersion;
    var o = navigator.userAgent;
    var j = 0;
    ISQ.Http.browser = {};
    if (k.indexOf("Android")!==-1 || o.indexOf("Android")!==-1) {
        ISQ.Http.browser.app = "android";
        var l = k.indexOf("Android ") + 8;
        var p = k.substring(l, k.indexOf(";", l));
        ISQ.Http.browser.version = p;
        ISQ.Http.browser.isMobile = k.indexOf("Mobile")>-1 || o.indexOf("Mobile")!==-1;
        ISQ.Http.browser.isTablet=!ISQ.Http.browser.isMobile;
        ISQ.Http.browser.usesViewport = true;
        ISQ.Http.browser.isAndroid = true;
        if (parseInt(ISQ.Http.browser.version) < 4) {
            ISQ.Http.browser.oldAndroid = true
        }
    } else {
        if (navigator.appVersion.indexOf("iPhone")!==-1) {
            var p = navigator.appVersion.substring(0, navigator.appVersion.indexOf(" "));
            ISQ.Http.browser.app = "safari";
            ISQ.Http.browser.version = "iphone " + parseInt(p);
            ISQ.Http.browser.isMobile = true;
            ISQ.Http.browser.usesViewport = true;
            ISQ.Http.browser.isIPhone = true;
            if (k.indexOf("OS 6_")!==-1) {
                ISQ.Http.browser.isIOS6 = true
            } else {
                if (k.indexOf("OS 7_")!==-1) {
                    ISQ.Http.browser.isIOS7 = true
                }
            }
            ISQ.Http.browser.isIOS = ISQ.Http.browser.isIOS6 || ISQ.Http.browser.isIOS7
        } else {
            if (navigator.appVersion.indexOf("iPad")!==-1) {
                var p = navigator.appVersion.substring(0, navigator.appVersion.indexOf(" "));
                ISQ.Http.browser.app = "safari";
                ISQ.Http.browser.version = "ipad " + parseInt(p);
                ISQ.Http.browser.isTablet = true;
                ISQ.Http.browser.usesViewport = true;
                ISQ.Http.browser.isIPad = true;
                if (k.indexOf("OS 6_")!==-1) {
                    ISQ.Http.browser.isIOS6 = true
                } else {
                    if (k.indexOf("OS 7_")!==-1) {
                        ISQ.Http.browser.isIOS7 = true
                    }
                }
                ISQ.Http.browser.isIOS = ISQ.Http.browser.isIOS6 || ISQ.Http.browser.isIOS7
            }
        }
    }
    if ((j = k.indexOf("MSIE "))!==-1) {
        ISQ.Http.browser.app = "ie";
        j += 5;
        var n = k.indexOf(" ", j);
        if (n===-1) {
            ISQ.Http.browser.version = k.substring(j)
        } else {
            ISQ.Http.browser.version = k.substring(j, n)
        }
        if (ISQ.Http.browser.version.startsWith("10")) {
            ISQ.Http.browser.isIE10 = true
        }
        if (ISQ.Http.browser.version.startsWith("9")) {
            ISQ.Http.browser.isIE9 = true
        }
        if (ISQ.Http.browser.version.startsWith("8")) {
            ISQ.Http.browser.isIE8 = true
        }
        if (ISQ.Http.browser.version.startsWith("7")) {
            ISQ.Http.browser.isIE7 = true
        }
        if (ISQ.Http.browser.version.startsWith("6")) {
            ISQ.Http.browser.isIE6 = true
        }
        ISQ.Http.browser.isIE = true
    } else {
        if (k.indexOf("Chrome")!==-1) {
            ISQ.Http.browser.app = "chrome";
            ISQ.Http.browser.version = "0"
        } else {
            if ((j = k.indexOf("Safari"))!==-1 && ISQ.Http.browser.app != "android") {
                var p = k.substring(j + 7);
                ISQ.Http.browser.app = "safari";
                if (p <= "530.17") {
                    ISQ.Http.browser.version = "3"
                } else {
                    ISQ.Http.browser.version = "4"
                }
            } else {
                if ((j = o.indexOf("Firefox/"))!==-1) {
                    ISQ.Http.browser.app = "ff";
                    var n = o.indexOf(" ", j);
                    if (n===-1) {
                        ISQ.Http.browser.version = o.substring(j + 8)
                    } else {
                        ISQ.Http.browser.version = o.substring(j + 8, n)
                    }
                    if (ISQ.Http.browser.version.startsWith("3")) {
                        ISQ.Http.browser.isFF3 = true
                    }
                    if (ISQ.Http.browser.version.startsWith("3.0")) {
                        ISQ.Http.browser.isFF3_0 = true
                    }
                    if (ISQ.Http.browser.version.startsWith("3.6")) {
                        ISQ.Http.browser.isFF3_6 = true
                    }
                } else {
                    if (o.indexOf("Opera/")!==-1) {
                        ISQ.Http.browser.app = "opera";
                        var l = o.indexOf("Version/");
                        if (l===-1) {
                            var m = o.indexOf(" ", 6);
                            ISQ.Http.browser.version = o.substring(6, m)
                        } else {
                            var m = o.indexOf(" ", l);
                            if (m===-1) {
                                ISQ.Http.browser.version = o.substring(l + 8)
                            } else {
                                ISQ.Http.browser.version = o.substring(l + 8, m)
                            }
                        }
                        ISQ.Http.browser.isOpera = true
                    } else {
                        if (o.contains("Symbian")) {
                            ISQ.Http.browser.app = "symbian";
                            ISQ.Http.browser.version = 1;
                            ISQ.Http.browser.full = ISQ.Http.browser.app + ISQ.Http.browser.version
                        }
                    }
                }
            }
        }
    }
    if (!ISQ.Http.browser.app) {
        if (o.indexOf("AppleWebKit")!==-1) {
            ISQ.Http.browser.app = "safari";
            ISQ.Http.browser.version = "5"
        } else {
            if (k.indexOf("Trident/7")!==-1) {
                ISQ.Http.browser.app = "ie";
                ISQ.Http.browser.isIE = true;
                var l = k.indexOf("rv:");
                if (l!==-1) {
                    l += 3;
                    if (k.equals("11", l)) {
                        ISQ.Http.browser.version = "11";
                        ISQ.Http.browser.isIE11 = true
                    }
                }
            } else {
                ISQ.Http.browser.app = "default";
                ISQ.Http.browser.version = "--"
            }
        }
    }
    ISQ.Http.browser.full = ISQ.Http.browser.app + ISQ.Http.browser.version;
    o = null;
    k = null
};
if (typeof(ISQ.Http.browser) === "undefined" ||!ISQ.Http.browser) {
    ISQ.Http.detectBroswer()
}
ISQ.Http.isIEQuirksMode = function() {
    return (ISQ.Http.browser.isIE && document.compatMode !== "CSS1Compat")
};
ISQ.Http.Cookies = initializeNS("ISQ.Http.Cookies");
ISQ.Http.Cookies._read = function(m) {
    if (!m) {
        return null
    }
    var r = document.cookie;
    var o = 0;
    var j;
    var n;
    var p = 0;
    var l = {
        KEY: 1,
        VALUE: 2
    };
    var q = l.KEY;
    while (p < r.length) {
        j = r.charAt(p++);
        if (j === " ") {
            continue
        }
        if (q === l.KEY) {
            if (j === "=") {
                q = l.VALUE;
                o = 0;
                continue
            }
            n = m.charAt(o);
            if (j === n || j.toLowerCase() === n.toLowerCase()) {
                if (++o === m.length) {
                    if (r.charAt(p) === "=") {
                        var k = r.indexOf(";", ++p);
                        if (k===-1) {
                            k = r.length
                        }
                        return r.substring(p, k)
                    } else {
                        o = 0
                    }
                }
            } else {
                o = 0
            }
        } else {
            if (j === ";") {
                q = l.KEY
            }
        }
    }
    return null
};
ISQ.Http.Cookies._readCaseSensitive = function(f) {
    if (!f) {
        return null
    }
    var j = document.cookie;
    var g = j.indexOf(f);
    while (g!==-1) {
        g += f.length;
        if (j.charAt(g) === "=") {
            var e = j.indexOf(";", ++g);
            if (e===-1) {
                e = j.length
            }
            return j.substring(g, e)
        } else {
            g = j.indexOf(f)
        }
    }
    return null
};
ISQ.Http.Cookies._write = function(n, q, o, k, p, m) {
    n = n || null;
    q=!isNaN(q) ? q : q || null;
    o = o || "/";
    k = k || null;
    p = p === true;
    m = m || null;
    if (n === null || q === null) {
        return 
    }
    var l = new Date();
    if (m !== null) {
        l.setSeconds(l.getSeconds() + m);
        l = l.toGMTString()
    }
    var j = [];
    if (n.indexOf("%20")!==-1 || n.toLowerCase().indexOf("%3a")!==-1) {
        j.push(n)
    } else {
        j.push(escape(n))
    }
    j.push("=");
    if (q.toString().indexOf("%20")!==-1 || q.toString().toLowerCase().indexOf("%3a")!==-1) {
        j.push(q)
    } else {
        j.push(escape(q))
    }
    j.push("; path=");
    j.push(o);
    if (k !== null) {
        j.push("; domain=");
        j.push(k)
    }
    if (p) {
        j.push("; secure=");
        j.push(p)
    }
    if (m !== null) {
        j.push("; expires=");
        j.push(l)
    }
    document.cookie = j.join("")
};
ISQ.Http.Cookies._delete = function(m, n, k, o) {
    m = m || null;
    n = n || null;
    k = k || null;
    o = o || null;
    if (m === null) {
        return 
    }
    var p = ISQ.Http.Cookies._read(m);
    if (p === null) {
        return 
    }
    var l = new Date();
    l.setMonth(l.getMonth() - 1);
    l = l.toGMTString();
    var j = [];
    j.push(escape(m));
    j.push("=1");
    j.push("; expires=");
    j.push(l);
    n = n || "/";
    j.push("; path=");
    j.push(n);
    k = k || ISQ.Http.domain(false);
    j.push("; domain=");
    j.push(k);
    if (o !== null) {
        j.push("; secure=");
        j.push(o)
    }
    document.cookie = j.join("")
};
ISQ.Http.Cookies.encode64 = function(d) {
    d += "";
    if (d.length === 0) {
        return d
    }
    var c = ISQ.Tools.Base64.encodeString(d);
    return c.replace(/\=/g, "_").replace(/\+/g, "|")
};
ISQ.Http.Cookies.decode64 = function(d) {
    d += "";
    if (d.length === 0) {
        return d
    }
    var c = d.replace(/\_/g, "=").replace(/\|/g, "+");
    return ISQ.Tools.Base64.decodeString(c)
};
ISQ.Http.domain = function(m, p) {
    m = m === true;
    var k;
    var l;
    if (typeof(ISQ.Http.mDomain) === "undefined" ||!ISQ.Http.mDomain || p) {
        p = p || window.location.href;
        var n = p.indexOf("//") + 2;
        k = p.substring(n);
        var o = k.indexOf("/");
        if (o>-1) {
            l = k.substring(0, o)
        } else {
            l = k
        }
        var j = k.indexOf(":");
        if (j>-1) {
            k = l.substring(0, j)
        } else {
            k = l
        }
    }
    if (!p) {
        ISQ.Http.mIncludePort = l;
        ISQ.Http.mDomain = k
    }
    if (m) {
        return l
    } else {
        return k
    }
};
ISQ.Http.relativePath = function(l) {
    l = l || window.location.href;
    var k = {
        path: "",
        filename: ""
    };
    var f = l.indexOf("//") + 2;
    f = l.indexOf("/", f);
    if (f == l.length - 1) {
        return k
    }
    var j = l.indexOf("?");
    var g = l.LastIndexOf_char("/", j===-1 ? 0 : j);
    k.path = l.substring(f, g++);
    k.filename = l.substring(g);
    f = k.filename.indexOf("#");
    if (f!==-1) {
        k.filename = k.filename.substring(0, f)
    }
    return k
};
if (typeof(ISQ.Http.urlArguments) === "undefined" ||!ISQ.Http.urlArguments) {
    if (window.location.search.length !== 0) {
        ISQ.Http.urlArguments = ISQ.Tools.stringToPairArray(window.location.search.substring(1), "=", "&", true, false)
    } else {
        ISQ.Http.urlArguments = []
    }
}
if (typeof(ISQ.Http.hashArguments) === "undefined" ||!ISQ.Http.hashArguments) {
    if (window.location.hash.length !== 0) {
        ISQ.Http.hashArguments = ISQ.Tools.stringToPairArray(window.location.hash.substring(1), "=", "&", true, false)
    } else {
        ISQ.Http.hashArguments = []
    }
}
ISQ.Http.getUrlParam = function(f, l) {
    f = f.replace(/[\[]/, "\\[").replace(/[\]]/, "\\]");
    var j = "[\\?&]" + f + "=([^&#]*)";
    var g = new RegExp(j);
    l = l || window;
    var k = g.exec(l.location.href);
    if (k == null) {
        return ""
    } else {
        return k[1]
    }
};
ISQ.Http.getUrlFakePart = function() {
    var d = window.location.href.indexOf("/~");
    if (d===-1) {
        return ""
    }
    d += 2;
    if (window.location.search && d > window.location.href.indexOf("?")) {
        return ""
    }
    if (window.location.hash && d > window.location.href.indexOf("#")) {
        return ""
    }
    var c = window.location.href.indexOf("/", d);
    if (c===-1) {
        return ""
    }
    return window.location.href.substring(d, c)
};
ISQ.Html = initializeNS("ISQ.Html");
ISQ.Html._ = function(j, e) {
    if (!j) {
        return null
    }
    var f = null;
    if (typeof(j) === "string") {
        if (ISQ.Http.browser.app === "ie") {
            if (!ISQ.Html.IE_idMap) {
                ISQ.Html.IE_idMap = new Array();
                if (typeof(ISQ.Console) !== "undefined" && typeof(sessionGC) === "function") {
                    sessionGC().add(ISQ.Html.IE_idMap);
                    ISQ.Html.IE_idMap.dispose = function() {
                        ISQ.Html.IE_idMap = new Array()
                    }
                }
            }
            if (!e) {
                f = ISQ.Html.IE_idMap[j];
                if (ISQ.Http.browser.isIE && f) {
                    var g = f;
                    do {
                        g = g.parentNode
                    }
                    while (g !== null && g !== document);
                    if (g === null) {
                        f = null
                    }
                }
            }
            if (!f || e === true) {
                f = document.getElementById(j);
                if (f) {
                    ISQ.Html.IE_idMap[j] = f
                }
            }
        } else {
            f = document.getElementById(j)
        }
    } else {
        if (typeof(j) === "object" && typeof(j.parentNode) !== "undefined" && typeof(j.nodeName) !== "undefined") {
            f = j
        }
    }
    return f
};
if (typeof(window._) !== "function") {
    window._ = ISQ.Html._
}
ISQ.Html.pageDirection = function() {
    if (!ISQ.Html._pageDirection) {
        ISQ.Html._pageDirection = document.body.getAttribute("dir");
        if (!ISQ.Html._pageDirection) {
            ISQ.Html._pageDirection = ISQ.CSS.getValue(document.body, "direction")
        }
        if (!ISQ.Html._pageDirection) {
            ISQ.Html._pageDirection = "ltr"
        }
    }
    return ISQ.Html._pageDirection
};
ISQ.Html.position_new = function(d, f) {
    if (f) {
        return ISQ.Html.position(d, f)
    }
    if (typeof(d.getBoundingClientRect) == "undefined" ||!d.getBoundingClientRect) {
        return ISQ.Html.position(d, f)
    }
    var e = d.getBoundingClientRect();
    e.x = e.left;
    e.y = e.top;
    return e
};
ISQ.Html.position = function(k, p, m) {
    var o = {
        x: - 1,
        y: - 1
    };
    k = ISQ.Html._(k);
    if (k === null) {
        return o
    }
    var n = ISQ.Html._(p) || document.body;
    o.x = 0;
    o.y = 0;
    var j = k;
    while (j !== null && j !== n && j !== document.body && j.nodeName !== "HTML") {
        o.x += j.offsetLeft;
        o.x -= j.scrollLeft;
        var l = parseInt(ISQ.CSS.getElementStyle(j, "margin-left"));
        o.x -= isNaN(l) ? 0 : l;
        o.y += j.offsetTop;
        o.y -= j.scrollTop;
        j = j.offsetParent
    }
    if (n !== document.body && (j === document.body || j === null)) {
        o.x =- 1;
        o.y =- 1
    }
    k = null;
    p = null;
    return o
};
ISQ.Html.setMinHeight = function(c, d) {
    c = ISQ.Html._(c);
    if (c === null) {
        return 
    }
    if (c.offsetHeight < d) {
        c.style.height = d + "px"
    }
};
ISQ.Html.actualHeight = function(m) {
    var l = $(m);
    var n = l.height();
    var k = l.css("border-top");
    if (k) {
        n += parseInt(k)
    }
    var j = l.css("border-bottom");
    if (j) {
        n += parseInt(j)
    }
    var p = l.css("padding-top");
    if (p) {
        n += parseInt(p)
    }
    var o = l.css("padding-bottom");
    if (o) {
        n += parseInt(o)
    }
    return n
};
ISQ.Html.screenSize = function() {
    return ISQ.Html.pageSize(true)
};
ISQ.Html.scrollBarWidth = function() {
    return 17
};
ISQ.Html.scrollBarWidth_afterCheck = function() {
    var b = ISQ.Html.screenSize();
    if (document.body.scrollHeight === b.h) {
        return 0
    }
    return ISQ.Html.scrollBarWidth()
};
ISQ.Html.pageSize = function(j, g) {
    var k = {
        w: 0,
        h: 0
    };
    var l = {};
    if (self.innerHeight) {
        l.w = self.innerWidth;
        l.h = self.innerHeight
    } else {
        if (document.documentElement && document.documentElement.clientHeight) {
            l.w = document.documentElement.clientWidth;
            l.h = document.documentElement.clientHeight
        } else {
            if (document.body) {
                l.w = document.body.clientWidth;
                l.h = document.body.clientHeight
            }
        }
    }
    var n = ISQ.Html.getScroll("width");
    var m = ISQ.Html.getScroll("height");
    if (j || l.w > n) {
        k.w = l.w
    } else {
        k.w = n
    }
    if ((j && (!ISQ.Http.browser.isIPad || window.top !== window)) || l.h > m) {
        k.h = l.h
    } else {
        k.h = m
    }
    if (m > l.h) {
        if (ISQ.CSS.getElementStyle(document.body, "overflow") === "hidden") {
            return k
        }
        if (ISQ.CSS.getElementStyle(document.body, "overflow-y") === "hidden") {
            return k
        }
        if (ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE10 || ISQ.Http.browser.isIE9 || ISQ.Http.browser.isIE11) {
            if (!g) {
                k.w -= ISQ.Html.scrollBarWidth()
            }
        } else {
            if (g) {
                k.w += ISQ.Html.scrollBarWidth()
            }
        }
    }
    return k
};
ISQ.Html.getScroll = function(f) {
    f = f || "top";
    var d = 0;
    var e = 0;
    if (f === "top") {
        if (ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE10 || ISQ.Http.browser.isIE11) {
            e = window.pageYOffset
        }
        if (ISQ.Http.browser.app !== "ie") {
            e = ISQ.Html.returnMaxScroll(e, window.scrollY)
        }
        e = ISQ.Html.returnMaxScroll(e, document.body.scrollTop);
        e = ISQ.Html.returnMaxScroll(e, document.documentElement.scrollTop);
        e = ISQ.Html.returnMaxScroll(e, document.body.parentNode.scrollTop)
    } else {
        if (f === "left") {
            if (ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE10 || ISQ.Http.browser.isIE11) {
                e = window.pageXOffset
            }
            if (ISQ.Http.browser.app !== "ie") {
                e = ISQ.Html.returnMaxScroll(e, window.scrollX)
            }
            e = ISQ.Html.returnMaxScroll(e, document.body.scrollLeft);
            e = ISQ.Html.returnMaxScroll(e, document.documentElement.scrollLeft);
            e = ISQ.Html.returnMaxScroll(e, document.body.parentNode.scrollLeft)
        } else {
            if (f === "height") {
                switch (ISQ.Http.browser.app) {
                case"chrome":
                case"ff":
                case"opera":
                case"safari":
                    if (window.top === window) {
                        e = document.documentElement.scrollHeight
                    } else {
                        e = document.body.scrollHeight
                    }
                    break;
                default:
                    e = ISQ.Html.returnMaxScroll(e, document.body.scrollHeight);
                    e = ISQ.Html.returnMaxScroll(e, document.documentElement.scrollHeight);
                    e = ISQ.Html.returnMaxScroll(e, document.body.parentNode.scrollHeight);
                    break
                }
            } else {
                if (f === "width") {
                    e = ISQ.Html.returnMaxScroll(e, document.body.scrollWidth);
                    e = ISQ.Html.returnMaxScroll(e, document.documentElement.scrollWidth);
                    e = ISQ.Html.returnMaxScroll(e, document.body.parentNode.scrollWidth)
                }
            }
        }
    }
    return e
};
ISQ.Html.returnMaxScroll = function(c, d) {
    if (c > d) {
        return c
    }
    return d
};
ISQ.Html.getEventTarget = function(b) {
    return window.event ? window.event.srcElement : b ? b.target : null
};
ISQ.Html.getDomEvent = function(b) {
    if (ISQ.Base.isDerivedOf(b, ISQ.Html.DomEvent)) {
        return b
    }
    return new ISQ.Html.DomEvent(b)
};
ISQ.Html.DomEvent = ISQ.Base.extend({
    name: "ISQ.Html.DomEvent"
});
ISQ.Html.DomEvent.prototype.ctor = function(b) {
    this.domEvent = window.event || b;
    this.target = ISQ.Html.getEventTarget(b)
};
ISQ.Html.DomEvent.prototype.dtor = function() {
    delete target;
    delete domEvent
};
ISQ.Html.DomEvent.prototype.cancel = function() {
    if (this.domEvent.stopPropagation) {
        this.domEvent.stopPropagation()
    }
    if (this.domEvent.cancelBubble != null) {
        this.domEvent.cancelBubble = true
    }
};
ISQ.Html.DomEvent.prototype.target = null;
ISQ.Html.DomEvent.prototype.domEvent = null;
window.addEvents = function(d, e, f) {
    if (d.addEventListener) {
        if (e.indexOf("on") == 0) {
            e = e.charAt(2).toLowerCase() + e.slice(3)
        }
        d.addEventListener(e, f)
    } else {
        if (d.attachEvent) {
            if (e.indexOf("on") != 0) {
                e = "on" + e
            }
            d.attachEvent(e, f)
        }
    }
};
window.removeEvents = function(d, e, f) {
    if (d.removeEventListener) {
        if (e.indexOf("on") == 0) {
            e = e.charAt(2).toLowerCase() + e.slice(3)
        }
        d.removeEventListener(e, f)
    } else {
        if (d.detachEvent) {
            if (e.indexOf("on") != 0) {
                e = "on" + e
            }
            d.detachEvent(e, f)
        }
    }
};
ISQ.Html.Scroll = initializeNS("ISQ.Html.Scroll");
ISQ.Html.Scroll.init = function() {
    if (ISQ.Html.Scroll._wasInit) {
        return 
    }
    ISQ.Html.Scroll._wasInit = true;
    ISQ.Html.Scroll._running = true;
    if (typeof(ISQ.Console) !== "undefined" && typeof(sessionGC) === "function") {
        sessionGC().add(ISQ.Html.Scroll)
    }
    ISQ.Html.Scroll.event = new ISQ.Event();
    if (window.addEventListener) {
        window.addEventListener("scroll", ISQ.Html.Scroll.scrollHandler, true)
    } else {
        if (window.attachEvent) {
            window.attachEvent("onscroll", ISQ.Html.Scroll.scrollHandler)
        }
    }
    if (ISQ.Http.browser.isIE8 || ISQ.Http.browser.isIE7 || ISQ.Http.browser.isIE6) {
        ISQ.Html.Scroll.interval = 30;
        ISQ.Html.Scroll.diff = 65
    }
};
ISQ.Html.Scroll.scrollHandler = function(d) {
    if (!ISQ.Html.Scroll._running) {
        return 
    }
    var f = ISQ.Html.getEventTarget(d);
    if (f !== null && f !== document) {
        return 
    }
    var e = {
        left: ISQ.Html.getScroll("left"),
        top: ISQ.Html.getScroll("top")
    };
    if (ISQ.Html.Scroll.mLastScroll && ISQ.Html.Scroll.mLastScroll.left === e.left && ISQ.Html.Scroll.mLastScroll.top === e.top) {
        return 
    }
    ISQ.Html.Scroll.mLastScroll = e;
    ISQ.Html.Scroll.timeStamp = new Date().getTime();
    if (ISQ.Html.Scroll.timer !== null) {
        return 
    }
    ISQ.Html.Scroll.timer = setInterval(function() {
        if (ISQ.Html.Scroll._running) {
            var a = new Date().getTime();
            if (a - ISQ.Html.Scroll.timeStamp < ISQ.Html.Scroll.diff) {
                return 
            }
            ISQ.Html.Scroll.event.dispatch(ISQ.Html.Scroll.mLastScroll, ISQ.Html.pageSize(), ISQ.Html.screenSize())
        }
        clearInterval(ISQ.Html.Scroll.timer);
        ISQ.Html.Scroll.timer = null
    }, ISQ.Html.Scroll.interval)
};
ISQ.Html.Scroll.dispose = function() {
    ISQ.Html.Scroll._running = false;
    if (window.removeEventListener) {
        window.removeEventListener("scroll", ISQ.Html.Scroll.scrollHandler, true)
    } else {
        if (window.detachEvent) {
            window.detachEvent("onscroll", ISQ.Html.Scroll.scrollHandler)
        }
    }
    ISQ.Html.Scroll.event.dispose();
    ISQ.Html.Scroll.event = null
};
ISQ.Html.Scroll.interval = 5;
ISQ.Html.Scroll.diff = 11;
ISQ.Html.Scroll.timer = null;
ISQ.Html.ReSize = initializeNS("ISQ.Html.ReSize");
ISQ.Html.ReSize.enumListenTo = {
    width: 1,
    height: 2,
    both: 3
};
ISQ.Html.ReSize.init = function(c, d) {
    if (!this.mListenTo) {
        this.mListenTo = 0
    }
    this.mListenTo|=c;
    ISQ.Html.ReSize.mSkipTimer|=d === true;
    if (ISQ.Html.ReSize.wasInit()) {
        return 
    }
    ISQ.Html.ReSize._wasInit = true;
    ISQ.Html.ReSize._running = true;
    if (typeof(ISQ.Console) !== "undefined" && typeof(sessionGC) == "function") {
        sessionGC().add(ISQ.Html.ReSize)
    }
    ISQ.Html.ReSize.Event = new ISQ.Event();
    ISQ.Html.ReSize.mWindowSize_thisEvent = ISQ.Html.screenSize();
    ISQ.Html.ReSize.mWindowSize_browserEvent = ISQ.Html.screenSize();
    if (window.addEventListener) {
        window.addEventListener("resize", ISQ.Html.ReSize.resizeHandler, true);
        if (ISQ.Http.browser.isMobile) {
            window.addEventListener("zoom", ISQ.Html.ReSize.resizeHandler, true)
        }
    } else {
        if (window.attachEvent) {
            window.attachEvent("onresize", ISQ.Html.ReSize.resizeHandler)
        }
    }
};
ISQ.Html.ReSize.resizeHandler = function() {
    if (!ISQ.Html.ReSize._running) {
        return 
    }
    var b = ISQ.Html.screenSize();
    if (!ISQ.Html.ReSize.mWindowSize_thisEvent ||!ISQ.Html.ReSize.mWindowSize_browserEvent) {
        return 
    }
    if ((b.w === ISQ.Html.ReSize.mWindowSize_thisEvent.w || (ISQ.Html.ReSize.mListenTo & ISQ.Html.ReSize.enumListenTo.width) === 0) && (b.h === ISQ.Html.ReSize.mWindowSize_thisEvent.h || (ISQ.Html.ReSize.mListenTo & ISQ.Html.ReSize.enumListenTo.height) === 0)) {
        if (ISQ.Html.ReSize.timer) {
            clearInterval(ISQ.Html.ReSize.timer);
            ISQ.Html.ReSize.timer = null
        }
        return 
    }
    if ((b.w === ISQ.Html.ReSize.mWindowSize_browserEvent.w || (ISQ.Html.ReSize.mListenTo & ISQ.Html.ReSize.enumListenTo.width) === 0) && (b.h === ISQ.Html.ReSize.mWindowSize_browserEvent.h || (ISQ.Html.ReSize.mListenTo & ISQ.Html.ReSize.enumListenTo.height) === 0)) {
        return 
    }
    ISQ.Html.ReSize.mWindowSize_browserEvent = b;
    ISQ.Html.ReSize.timeStamp = new Date().getTime();
    if (ISQ.Html.ReSize.timer != null && typeof(ISQ.Html.ReSize.timer) !== "undefined") {
        return 
    }
    if (ISQ.Html.ReSize.mSkipTimer) {
        ISQ.Html.ReSize.Event.dispatch();
        ISQ.Html.ReSize.mWindowSize_thisEvent = ISQ.Html.ReSize.mWindowSize_browserEvent;
        return 
    }
    ISQ.Html.ReSize.timer = setInterval(function() {
        if (ISQ.Html.ReSize._running) {
            var a = new Date().getTime();
            if (a - ISQ.Html.ReSize.timeStamp < ISQ.Html.ReSize.diff) {
                return 
            }
            ISQ.Html.ReSize.Event.dispatch();
            ISQ.Html.ReSize.mWindowSize_thisEvent = ISQ.Html.ReSize.mWindowSize_browserEvent
        }
        clearInterval(ISQ.Html.ReSize.timer);
        ISQ.Html.ReSize.timer = null
    }, ISQ.Html.ReSize.interval)
};
ISQ.Html.ReSize.wasInit = function() {
    if (typeof(this._wasInit) === "undefined") {
        return false
    }
    return this._wasInit
};
ISQ.Html.ReSize.suspend = function() {
    if (!ISQ.Html.ReSize.wasInit()) {
        return 
    }
    ISQ.Html.ReSize._running = false
};
ISQ.Html.ReSize.resume = function() {
    if (!ISQ.Html.ReSize.wasInit()) {
        return 
    }
    ISQ.Html.ReSize._running = true
};
ISQ.Html.ReSize.dispose = function() {
    if (!ISQ.Html.ReSize.wasInit()) {
        return 
    }
    ISQ.Html.ReSize._running = false;
    if (window.removeEventListener) {
        window.removeEventListener("resize", ISQ.Html.ReSize.resizeHandler, true);
        if (ISQ.Http.browser.isMobile) {
            window.removeEventListener("zoom", ISQ.Html.ReSize.resizeHandler, true)
        }
    } else {
        if (window.detachEvent) {
            window.detachEvent("onresize", ISQ.Html.ReSize.resizeHandler)
        }
    }
    ISQ.Html.ReSize.Event.dispose();
    ISQ.Html.ReSize.Event = null
};
ISQ.Html.ReSize.interval = 100;
ISQ.Html.ReSize.diff = 199;
ISQ.Html.WindowActive = initializeNS("ISQ.Html.WindowActive");
ISQ.Html.WindowActive.isInit = false;
ISQ.Html.WindowActive.hidden = null;
ISQ.Html.WindowActive.event = null;
ISQ.Html.WindowActive.init = function() {
    if (ISQ.Html.WindowActive.isInit) {
        return 
    }
    ISQ.Html.WindowActive.isInit = true;
    ISQ.Html.WindowActive.event = new ISQ.Event();
    ISQ.Html.WindowActive.hidden = "hidden";
    if (ISQ.Html.WindowActive.hidden in document) {
        document.addEventListener("visibilitychange", ISQ.Html.WindowActive.onchange)
    } else {
        if ((ISQ.Html.WindowActive.hidden = "mozHidden") in document) {
            document.addEventListener("mozvisibilitychange", ISQ.Html.WindowActive.onchange)
        } else {
            if ((ISQ.Html.WindowActive.hidden = "webkitHidden") in document) {
                document.addEventListener("webkitvisibilitychange", ISQ.Html.WindowActive.onchange)
            } else {
                if ((ISQ.Html.WindowActive.hidden = "msHidden") in document) {
                    document.addEventListener("msvisibilitychange", ISQ.Html.WindowActive.onchange)
                } else {
                    if ("onfocusin" in document) {
                        document.onfocusin = document.onfocusout = ISQ.Html.WindowActive.onchange
                    } else {
                        window.onpageshow = window.onpagehide = window.onfocus = window.onblur = ISQ.Html.WindowActive.onchange
                    }
                }
            }
        }
    }
};
ISQ.Html.WindowActive.onchange = function(g) {
    var f = true, k = false;
    var j = {
        focus: f,
        focusin: f,
        pageshow: f,
        blur: k,
        focusout: k,
        pagehide: k
    };
    g = g || window.event;
    var l;
    if (g.type in j) {
        l = j[g.type]
    } else {
        l = this[ISQ.Html.WindowActive.hidden] ? k : f
    }
    ISQ.Html.WindowActive.event.dispatch(l)
};
ISQ.Html.WindowActive.dispose = function() {
    if (!ISQ.Html.WindowActive.isInit) {
        return 
    }
    if ("hidden" in document) {
        document.removeEventListener("visibilitychange", ISQ.Html.WindowActive.onchange)
    } else {
        if ("mozHidden" in document) {
            document.removeEventListener("mozvisibilitychange", ISQ.Html.WindowActive.onchange)
        } else {
            if ("webkitHidden" in document) {
                document.removeEventListener("webkitvisibilitychange", ISQ.Html.WindowActive.onchange)
            } else {
                if ("msHidden" in document) {
                    document.removeEventListener("msvisibilitychange", ISQ.Html.WindowActive.onchange)
                } else {
                    if ("onfocusin" in document) {
                        document.onfocusin = document.onfocusout = null
                    } else {
                        window.onpageshow = window.onpagehide = window.onfocus = window.onblur = null
                    }
                }
            }
        }
    }
    ISQ.Html.WindowActive.event.dispose();
    ISQ.Html.WindowActive.event = null
};
ISQ.Html._doc = document;
ISQ.Html.createElement = function(l, j, k) {
    var g = ISQ.Html._doc.createElement(l);
    var f = ISQ.Html._(j);
    if (f !== null) {
        f.appendChild(g)
    }
    if (typeof(k) === "string") {
        ISQ.CSS.setStyle(g, k)
    }
    return g
};
var createElement = ISQ.Html.createElement;
ISQ.Html.createDiv = function(g, j) {
    var f = ISQ.Html._doc.createElement("div");
    var e = ISQ.Html._(g);
    if (e !== null) {
        e.appendChild(f)
    }
    if (typeof(j) === "string") {
        ISQ.CSS.setStyle(f, j)
    }
    return f
};
var createDiv = ISQ.Html.createDiv;
ISQ.Html.createTable = function(k, n, j, l, g) {
    var m = createElement("table", k);
    m.setAttribute("cellPadding", 0);
    m.setAttribute("cellSpacing", 0);
    m.setAttribute("border", 0);
    if (typeof(g) === "string") {
        m.setAttribute("align", g)
    }
    if (typeof(n) === "number" || typeof(n) === "string") {
        m.setAttribute("width", n)
    }
    if (typeof(j) === "string") {
        m.setAttribute("dir", j);
        m.style.direction = j
    }
    if (typeof(l) === "string") {
        ISQ.CSS.setStyle(m, l)
    }
    return createElement("tbody", m)
};
var createTable = ISQ.Html.createTable;
ISQ.Html.createRow = function(f, e, j) {
    var g = createElement("tr", f);
    if (typeof(e) === "string") {
        ISQ.CSS.setStyle(g, e)
    }
    if (typeof(j) === "string") {
        g.setAttribute("valign", j)
    }
    return g
};
var createRow = ISQ.Html.createRow;
ISQ.Html.addSpaceRow = function(l, j, g, k) {
    var n = createRow(l);
    var m = createElement("td", n);
    m.style.height = j;
    if (typeof(k) === "string") {
        ISQ.CSS.setStyle(n, k)
    }
    if (g) {
        m.setAttribute("colSpan", g)
    }
    return n
};
var addSpaceRow = ISQ.Html.addSpaceRow;
ISQ.Html.addSpaceTd = function(k, l, g) {
    var j = createElement("td", k);
    if (typeof(g) === "number") {
        j.setAttribute("rowSpan", g)
    }
    if (!l) {
        return 
    }
    j.style.width = l;
    var f = createDiv(j, "width:" + l);
    f.innerHTML = "&nbsp;";
    return j
};
var addSpaceTd = ISQ.Html.addSpaceTd;
ISQ.Html.createTd = function(m, g, n, k, j) {
    var l = createElement("td", m);
    if (n) {
        l.setAttribute("vAlign", n)
    }
    if (g) {
        l.setAttribute("align", g)
    }
    if (typeof(k) === "string") {
        ISQ.CSS.setStyle(l, k)
    }
    if (j) {
        l.setAttribute("colSpan", j)
    }
    return l
};
var createTd = ISQ.Html.createTd;
ISQ.Html.clearNode = function(d, f) {
    if (typeof(d) === "undefined" || d === null) {
        return null
    }
    if (d.nodeType !== 1) {
        return d
    }
    f = f !== false;
    if (f) {
        var e = d.nodeName;
        switch (e) {
        case"TR":
        case"TBODY":
        case"TABLE":
            break;
        default:
            d.innerHTML = "";
            return d
        }
    }
    while (d.firstChild !== null) {
        if (f) {
            ISQ.Html.clearNode(d.firstChild, true)
        }
        d.removeChild(d.firstChild)
    }
    return d
};
var clearNode = ISQ.Html.clearNode;
ISQ.Html.createInput = function(w, q, t, x, n, s, u) {
    if (typeof(arguments[0]) == "object") {
        var r = arguments[0];
        return createInput(r.type, r.name, r.parent, r.value || null, r.errorStyle || null, r.style || null, r.placeholder || null)
    }
    var o;
    q = q || "";
    var p = false;
    if (ISQ.Http.browser.app === "ie"&&!ISQ.Http.browser.isIE11) {
        try {
            var v = new Array();
            v.push("<input type='");
            v.push(w);
            v.push("' name='");
            v.push(q);
            v.push("' id='");
            v.push(q);
            v.push("' />");
            o = ISQ.Html._doc.createElement(v.join(""));
            if (t) {
                t.appendChild(o)
            }
            p = true
        } catch (e) {
            p = false
        }
    }
    if (!p) {
        o = createElement("input", t);
        o.type = w;
        o.name = o.id = q
    }
    if (typeof(x) !== "undefined" && x !== null) {
        o.value = x
    }
    if (typeof(u) !== "undefined" && u !== null) {
        o.placeholder = u
    }
    if (typeof(n) === "string") {
        o.errorStyle = n;
        o.error = function() {
            ISQ.CSS.setStyle(this, this.errorStyle);
            o.hasError = true
        }
    }
    o.okStyle = typeof(s) === "string" ? s : "";
    ISQ.CSS.setStyle(o, o.okStyle);
    o.ok = function() {
        ISQ.CSS.setStyle(this, this.okStyle);
        o.hasError = false
    };
    return o
};
var createInput = ISQ.Html.createInput;
ISQ.Html.createCheckbox = function(s, q, v, y, n, p, u) {
    if (typeof(arguments[0]) == "object" && arguments[0] != null) {
        var t = arguments[0];
        return createCheckbox(t.name, t.id, t.parent, t.text, t.style, t.checked, t.onclick)
    }
    var w = createTable(v);
    var q = q || s || "cb_" + Math.floor(Math.random() * 1000000);
    if (n) {
        ISQ.CSS.setStyle(w.parentNode, n)
    }
    var z = createRow(w);
    var x = createTd(z);
    var o = createInput("checkbox", q, x);
    o.style.margin = "2px 3px 0";
    o.id = q;
    o.checked = o.defaultChecked = p === true;
    if (typeof(u) != "undefined") {
        o.onclick = u
    }
    if (y) {
        x = createTd(z);
        var r = createElement("label", x);
        r.setAttribute("for", o.id);
        r.innerHTML = y;
        o.label = r
    }
    return o
};
var createCheckbox = ISQ.Html.createCheckbox;
ISQ.Html.createRadio = function(y, w, B, F, H, s, t, r, A, v) {
    if (typeof(arguments[0]) == "object") {
        var z = arguments[0];
        return ISQ.Html.createRadio(z.name, z.id, z.parent, z.text, z.value, z.style, z.checked, z.floatStyle, z.onclick, z.holderStyle)
    }
    var D = createTable(B);
    if (r) {
        D.parentNode.setAttribute("align", r);
        var u = (r.toLowerCase() === "left" ? "right" : "left");
        ISQ.CSS.setStyle(D.parentNode, "margin-" + u + ": 5px")
    }
    var G = createRow(D);
    var E = createTd(G, "", "", v);
    var C = createInput("radio", y, E, H, s, s);
    C.style.margin = "0 3px 0";
    C.id = w || "radio_" + Math.floor(Math.random() * 1000000);
    if (typeof(t) != "undefined") {
        C.checked = t
    }
    if (typeof(A) != "undefined") {
        C.onclick = A
    }
    E = createTd(G);
    var x = createElement("label", E);
    x.setAttribute("for", C.id);
    x.innerHTML = F;
    return C
};
var createRadio = ISQ.Html.createRadio;
ISQ.Html.createSelect = function(f) {
    var j = createElement("select", f.parent, f.style || "");
    if (f.defaultValue) {
        var g = createElement("option", j);
        g.value = g.innerHTML = f.defaultValue;
        g.selected = true
    }
    for (var e = 0; e < f.values.length; ++e) {
        var g = createElement("option", j);
        if (typeof(f.values[e]) === "object") {
            g.innerHTML = ISQ.Data.unHTML(f.values[e].html.toString());
            g.value = f.values[e].value
        } else {
            g.value = f.values[e];
            g.innerHTML = ISQ.Data.unHTML(f.values[e].toString())
        }
        if (f.selected && g.value == f.selected) {
            g.selected = true
        }
    }
    return j
};
var createSelect = ISQ.Html.createSelect;
ISQ.Html.updateSelectOptions = function(e) {
    clearNode(e.select);
    for (var d = 0; d < e.values.length; ++d) {
        var f = createElement("option", e.select);
        if (typeof(e.values[d]) === "object") {
            f.innerHTML = e.values[d].html;
            f.value = e.values[d].value
        } else {
            f.value = f.innerHTML = e.values[d]
        }
        if (e.selected && f.value == e.selected) {
            f.selected = true
        }
    }
    return e.select
};
var updateSelectOptions = ISQ.Html.updateSelectOptions;
ISQ.Html.select_selectedOption = function(b) {
    return b[b.selectedIndex]
};
ISQ.Html.createForm = function(j) {
    var e = createElement("form", j.parent);
    if (typeof(j.action) === "string") {
        e.action = j.action
    }
    if (typeof(j.method) === "string") {
        e.method = j.method
    }
    if (typeof(j.id) === "string") {
        e.id = from.name = j.id
    }
    if (typeof(j.name) === "string") {
        from.name = j.name;
        if (typeof(j.id) !== "string") {
            e.id = from.name
        }
    }
    var f = typeof(j.onsubmit) !== "undefined" && j.onsubmit !== null ? j.onsubmit: null;
    if (f !== null) {
        f.executed = false
    }
    e.onsubmit = function(a) {
        if (f !== null) {
            if (!f.executed) {
                f.handle(ISQ.Html.getDomEvent(a));
                f.executed = true
            } else {
                f.executed = false
            }
        }
        return false
    };
    function g(b) {
        var a = ISQ.Html.getDomEvent(b);
        if (a.domEvent.keyCode !== 13) {
            return false
        }
        if (f !== null) {
            if (!f.executed) {
                f.handle(a);
                f.executed = true
            } else {
                f.executed = false
            }
        }
        a.cancel();
        return false
    }
    if (ISQ.Html._doc.addEventListener) {
        if (ISQ.Http.browser.app !== "chrome" && ISQ.Http.browser.app !== "opera") {
            e.addEventListener("keypress", g, true)
        } else {
            e.addEventListener("keyup", g, true)
        }
    } else {
        if (ISQ.Html._doc.attachEvent) {
            e.attachEvent("onkeyup", g)
        }
    }
    return e
};
var createForm = ISQ.Html.createForm;
ISQ.Html.createIframe = function(g, k, f, l) {
    l = l || "about:blank";
    var j = createElement("iframe");
    j.setAttribute("name", k);
    j.setAttribute("id", g);
    j.setAttribute("src", l);
    if (f) {
        ISQ.CSS.setStyle(j, f)
    }
    return j
};
ISQ.Html.createImage = function(d) {
    if (!d.src) {
        throw ("createImage: invalid src")
    }
    var c = new Image();
    c.src = d.src;
    if (d.onload) {
        c.onload = d.onload
    }
    if (d.id) {
        c.id = d.id
    }
    if (d.align) {
        c.setAttribute("align", d.align)
    }
    if (d.style) {
        ISQ.CSS.setStyle(c, d.style)
    }
    if (d.parent) {
        d.parent.appendChild(c)
    }
    if (d.altText) {
        c.setAttribute("title", d.altText)
    }
    if (d.overSrc) {
        ISQ.Html.setImageMouseOverAndOut(c, d.overSrc)
    }
    return c
};
var createImage = ISQ.Html.createImage;
ISQ.Html.createLink = function(d) {
    var a = createElement("a", d.parent, d.style);
    if (d.text) {
        a.innerHTML = d.text
    }
    if (typeof(d.href) === "string") {
        a.href = d.href
    } else {
        a.href = "javascript:void(0)"
    }
    if (typeof(d.title) === "string") {
        a.title = d.title
    }
    if (typeof(d.target) === "string") {
        a.target = d.target
    }
    if (typeof(d.id) === "string") {
        a.id = d.id
    }
    ISQ.Html.setOnclick(a, d.onclick);
    return a
};
var createLink = ISQ.Html.createLink;
window.setOnclick = ISQ.Html.setOnclick = function(c, d) {
    if (c.nodeName !== "A") {
        if (d) {
            c.onclick = d
        }
        return 
    }
    c.onclick = function(a) {
        if (typeof(ISQ.Console) !== "undefined" && typeof(ISQ.Console.SessionManager) !== "undefined") {
            ISQ.Console.SessionManager.lastClickedLinkHref = this.href ? this.href : null;
            setTimeout(function() {
                delete ISQ.Console.SessionManager.lastClickedLinkHref
            }, 10)
        }
        if (typeof(d) !== "undefined") {
            d.apply(this, new Array(a))
        }
    }
};
ISQ.Html.createHR = function(l, k, g, j) {
    var n = createRow(l);
    var m = createElement("td", n);
    m.style.height = k;
    m.style.background = g;
    if (typeof(j) === "number") {
        m.setAttribute("colSpan", j)
    }
    return n
};
var createHR = ISQ.Html.createHR;
ISQ.Html.createTextNode = function(j, g) {
    var f = ISQ.Html._doc.createTextNode(j);
    var e = ISQ.Html._(g);
    if (e !== null) {
        e.appendChild(f)
    }
    return f
};
var createTextNode = ISQ.Html.createTextNode;
ISQ.Html.moveChildNodesToElement = function(f, e) {
    clearNode(e);
    var d = f.childNodes;
    while (d.length > 0) {
        e.appendChild(d[0])
    }
};
var moveChildNodesToElement = ISQ.Html.moveChildNodesToElement;
ISQ.Html.domRecursion = function(d, e) {
    if (!d) {
        return 
    }
    e(d);
    for (var f = 0; f < d.childNodes.length; ++f) {
        domRecursion(d.childNodes[f], e)
    }
};
window.domRecursion = ISQ.Html.domRecursion;
ISQ.Html.setImageMouseOverAndOut = function(d, f) {
    var e = d.src;
    d.onmouseover = function() {
        this.src = f
    };
    d.onmouseout = function() {
        this.src = e
    }
};
var setImageMouseOverAndOut = ISQ.Html.setImageMouseOverAndOut;
ISQ.Html.elementToString = function(j, l) {
    var n = new Array();
    n.push("<");
    n.push(j.nodeName);
    n.push(" ");
    for (var k = 0; k < j.attributes.length; ++k) {
        var g = j.attributes[k];
        if (!g.nodeValue) {
            continue
        }
        var m = null;
        if (typeof(g.nodeValue) === "string") {
            m = g.nodeValue.indexOf("'")>-1 ? '"' : "'"
        }
        n.push(g.nodeName);
        n.push("=");
        if (m) {
            n.push(m)
        }
        n.push(g.nodeValue);
        if (m) {
            n.push(m)
        }
        n.push(" ")
    }
    n.push(">");
    if (l) {
        n.push(l)
    } else {
        n.push(j.innerHTML)
    }
    n.push("</");
    n.push(j.nodeName);
    n.push(">");
    return n.join("")
};
var elementToString = ISQ.Html.elementToString;
ISQ.Html.getElementsByClassName = function(d, c) {
    if (typeof(document.getElementsByClassName) === "function") {
        return d.getElementsByClassName(c)
    } else {
        return d.querySelectorAll("." + c)
    }
};
ISQ.Html.setPlaceHolder = function(m, p, j, o) {
    var n = m.parentNode || o;
    if (!n) {
        return 
    }
    var q = createDiv(n);
    q.className = j;
    q.innerHTML = p;
    q.onclick = function() {
        m.focus()
    };
    var k = function() {
        if (m.value == "") {
            q.style.display = "block"
        } else {
            q.style.display = "none"
        }
    };
    var l = function() {
        if (m.value == "") {
            q.style.display = "block"
        }
    };
    window.addEvents(m, "onkeyup", k);
    window.addEvents(m, "onblur", l);
    m.autoCheck = k
};
ISQ.Html.createInvisibleIframe = function(c) {
    var d = createElement("iframe", document.body, "visible:hidden;position:absolute;top:0px;left:0px;width:1px;height:1px");
    setTimeout(function() {
        if (c.isDisposed()) {
            document.body.removeChild(d);
            return 
        }
        ISQ.Html.waitUntilAvailable(d, c)
    }, 0);
    return d
};
createInvisibleIframe = ISQ.Html.createInvisibleIframe;
ISQ.Html.waitUntilAvailable = function(k, g) {
    var j = null;
    var l = null;
    try {
        j = k.contentDocument || k.contentWindow.document || null;
        l = k.contentWindow || k.contentDocument.defaultView || null
    } catch (f) {}
    if (!j ||!l || (!j.body&&!l.document.body)) {
        setTimeout(function() {
            if (g.isDisposed()) {
                document.body.removeChild(k);
                return 
            }
            ISQ.Html.waitUntilAvailable(k, g)
        }, 10);
        return 
    }
    if (j.body) {
        ISQ.Html.iframeReady(j, g)
    } else {
        ISQ.Html.iframeReady(l.document, g)
    }
};
ISQ.Html.iframeReady = function(d, c) {
    d.open();
    d.writeln("<html><body></body></html>");
    d.close();
    c.handle(d);
    c.dispose()
};
ISQ.Html.detectDoctype = function() {
    var d = "loose";
    try {
        var e = null;
        if (window.document.doctype) {
            e = window.document.doctype.systemId;
            if (!e) {
                e = window.document.doctype.publicId
            }
            if (!e && window.document.doctype.name.toLowerCase() === "html") {
                e = "doctype html"
            }
        } else {
            e = window.document.firstChild;
            if (e.innerHTML.length !== 0) {
                e = null
            } else {
                e = e.nodeValue
            }
        }
        if (e === null) {
            d = "nodoctype"
        } else {
            e = e.toLowerCase();
            if (e === "doctype html" || e === "ctype ht") {
                d = "html5"
            } else {
                if (e.indexOf("xhtml")!==-1 || e.indexOf("rdfa 1.1")!==-1) {
                    d = "xhtml"
                } else {
                    if (e.indexOf("strict")!==-1) {
                        d = "strict"
                    } else {
                        if (e.indexOf("loose")!==-1) {
                            d = "loose"
                        } else {
                            if (e.indexOf("transitional")!==-1) {
                                if (e.indexOf("4.01")!==-1) {
                                    d = "transitional401"
                                } else {
                                    if (e.indexOf("4.0")!==-1) {
                                        d = "transitional4"
                                    } else {
                                        d = "transitional"
                                    }
                                }
                            } else {
                                if (e.indexOf("frameset")!==-1) {
                                    d = "frameset"
                                } else {
                                    d = "nodoctype"
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (f) {
        d = "nodoctype"
    }
    return d
};
ISQ.CSS = initializeNS("ISQ.CSS");
ISQ.CSS.getVerticalMargin = function(c) {
    var d = 0;
    tp = ISQ.CSS.getElementStyle(c, "margin-top");
    tp = parseInt(tp);
    if (!isNaN(tp)) {
        d += tp
    }
    bt = ISQ.CSS.getElementStyle(c, "margin-bottom");
    bt = parseInt(bt);
    if (!isNaN(bt)) {
        d += bt
    }
    return d
};
ISQ.CSS.getHorizonalMargin = function(c) {
    var d = 0;
    left = ISQ.CSS.getElementStyle(c, "margin-left");
    left = parseInt(left);
    if (!isNaN(left)) {
        d += left
    }
    right = ISQ.CSS.getElementStyle(c, "margin-right");
    right = parseInt(right);
    if (!isNaN(right)) {
        d += right
    }
    return d
};
ISQ.CSS.setStyle = function(j, l, f) {
    j = ISQ.Html._(j);
    if (j === null) {
        return 
    }
    if (typeof(l) !== "string") {
        return 
    }
    if (typeof(f) === "undefined") {
        f = true
    }
    if (f) {
        j.className = "";
        j.style.cssText = ""
    }
    var k = l.split("&");
    for (var g = 0; g < k.length; ++g) {
        if (k[g].charAt(0) === ".") {
            j.className += " " + k[g].substring(1)
        } else {
            j.style.cssText += ";" + k[g]
        }
    }
};
ISQ.CSS.setOpacity = function(e, f) {
    e = ISQ.Html._(e);
    if (e === null) {
        return 
    }
    if (typeof(f) !== "number" || (f < 0 || f > 100)) {
        return 
    }
    if (ISQ.Http.browser.isIE6 || ISQ.Http.browser.isIE7) {
        if (f === 100) {
            e.style.removeAttribute("filter");
            if (e.nodeName === "TR") {
                for (var d = 0; d < e.childNodes.length; ++d) {
                    ISQ.CSS.setOpacity(e.childNodes[d], f)
                }
            }
        } else {
            e.style.filter = "alpha(opacity=" + f + ")";
            if (e.nodeName === "TR") {
                for (var d = 0; d < e.childNodes.length; ++d) {
                    ISQ.CSS.setOpacity(e.childNodes[d], f)
                }
            }
        }
    } else {
        if (ISQ.Http.browser.isIE8 || ISQ.Http.browser.isIE9) {
            if (f === 100) {
                e.style.removeAttribute("filter");
                if (e.nodeName === "TR") {
                    for (var d = 0; d < e.childNodes.length; ++d) {
                        e.childNodes[d].style.removeAttribute("filter")
                    }
                }
            } else {
                e.style.filter = "progid:DXImageTransform.Microsoft.Alpha(Opacity=" + f + ")";
                if (e.nodeName === "TR") {
                    for (var d = 0; d < e.childNodes.length; ++d) {
                        e.childNodes[d].style.filter = "inherit"
                    }
                }
            }
        } else {
            if (f === 100) {
                e.style.removeProperty("opacity");
                if (ISQ.Http.browser.app === "opera" && e.nodeName === "TR") {
                    for (var d = 0; d < e.childNodes.length; ++d) {
                        e.childNodes[d].style.removeProperty("opacity")
                    }
                }
            } else {
                e.style.opacity = (f / 100);
                if (ISQ.Http.browser.app === "opera" && e.nodeName === "TR") {
                    for (var d = 0; d < e.childNodes.length; ++d) {
                        e.childNodes[d].style.opacity = "inherit"
                    }
                }
            }
        }
    }
};
ISQ.CSS.toggleCss = function(e, m) {
    var n = false;
    var q = "cssRules";
    if (ISQ.Http.browser.app === "ie"&&!ISQ.Http.browser.isIE11) {
        q = "rules"
    }
    var p = document.styleSheets.length;
    for (var o = p - 1; o >= 0; --o) {
        var s = document.styleSheets[o];
        try {
            if (s.href || s[q].length == 0) {
                continue
            }
            var r = s[q][0].selectorText;
            if (!r) {
                continue
            }
            if (r.toLowerCase() !== e.toLowerCase()) {
                continue
            }
            s.disabled=!m;
            n = true
        } catch (l) {}
    }
    return n
};
ISQ.CSS.bringToFront = function(e) {
    var m = false;
    var q = "cssRules";
    if (ISQ.Http.browser.app === "ie"&&!ISQ.Http.browser.isIE11) {
        q = "rules"
    }
    var p = document.styleSheets.length;
    for (var o = p - 1; o >= 0; --o) {
        var s = document.styleSheets[o];
        try {
            if (s.href || s[q].length == 0) {
                continue
            }
            var r = s[q][0].selectorText;
            if (!r) {
                continue
            }
            if (r.toLowerCase() !== e.toLowerCase()) {
                continue
            }
            if (s.ownerNode) {
                styleElement = s.ownerNode
            } else {
                styleElement = s.owningElement
            }
            var n = styleElement.parentNode;
            if (n) {
                n.appendChild(styleElement)
            }
            m = true
        } catch (l) {}
    }
    return m
};
ISQ.CSS.getElementStyle = function(k, m) {
    k = ISQ.Html._(k);
    var n = null;
    if (window.getComputedStyle) {
        n = document.defaultView.getComputedStyle(k, null).getPropertyValue(m)
    }
    if (!n && k.runtimeStyle) {
        if (m.indexOf("-")!==-1) {
            var e = [];
            for (var l = 0; l < m.length; l++) {
                if (m.charAt(l) === "-") {
                    e.push(m.charAt(++l).toUpperCase())
                } else {
                    e.push(m.charAt(l))
                }
            }
            m = e.join("")
        }
        if (typeof(document.defaultView) !== "undefined") {
            try {
                n = document.defaultView.getComputedStyle(k, null)[m]
            } catch (j) {
                n = null
            }
        }
        if (!n) {
            try {
                n = k.runtimeStyle[m]
            } catch (j) {
                n = null
            }
        }
        if (!n) {
            if ((ISQ.Http.browser.isIE7 || ISQ.Http.browser.isIE6) && m.contains("border")) {
                n = null
            } else {
                if (k.currentStyle) {
                    n = k.currentStyle[m]
                }
            }
        }
    }
    return n
};
ISQ.CSS.getElementWidth = function(d, e) {
    var f = d.offsetWidth;
    f -= ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(d, "padding-left"), 0);
    f -= ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(d, "padding-right"), 0);
    if (e) {
        f -= ISQ.CSS.getHorizonalMargin(d)
    }
    return f
};
ISQ.CSS.getElementHeight = function(e, g, f) {
    var j = e.offsetHeight;
    if (j === 0) {
        if (f) {
            return 0
        }
        j = ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(e, "height"), 0)
    }
    j -= ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(e, "padding-top"), 0);
    j -= ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(e, "padding-bottom"), 0);
    if (g) {
        j -= ISQ.CSS.getVerticalMargin(e)
    }
    return j
};
ISQ.CSS.getPaddingWidth = function(c) {
    var d = ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(c, "padding-left"), 0);
    d += ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(c, "padding-right"), 0);
    return d
};
ISQ.CSS.setWidthIncludingPadding = function(c, d) {
    d -= ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(c, "padding-left"), 0);
    d -= ISQ.Tools.getNumber(ISQ.CSS.getElementStyle(c, "padding-right"), 0);
    c.style.width = d + "px"
};
ISQ.CSS.addClass = function(f, e) {
    if (f.className) {
        var j = f.className.split(" ");
        for (var g in j) {
            if (j[g] == e) {
                return 
            }
        }
    }
    f.className += " " + e
};
ISQ.CSS.removeClass = function(j, g) {
    if (!j.className) {
        return 
    }
    var f = new Array();
    var l = j.className.split(" ");
    for (var k in l) {
        if (l[k] && l[k] != g) {
            f.push(l[k])
        }
    }
    j.className = f.join(" ")
};
ISQ.CSS.addNewStyle = function(f, e) {
    var g = document.createElement("style");
    g.type = "text/css";
    if (e) {
        g.id = e
    }
    document.getElementsByTagName("head")[0].appendChild(g);
    if (ISQ.Http.browser.app === "ie"&&!ISQ.Http.browser.isIE11) {
        g.styleSheet.cssText = f
    } else {
        var j = document.createTextNode(f);
        g.appendChild(j)
    }
    return g
};
ISQ.CSS.setNodeStyleText = function(c, d) {
    c.setAttribute("style", d);
    c.style.cssText = d
};
ISQ.CSS.removeStyle = function(j) {
    if (!j) {
        return 
    }
    var e = document.getElementsByTagName("head")[0];
    for (var g = 0; g < e.childNodes.length; g++) {
        var f = e.childNodes[g];
        if (f.id == j) {
            e.removeChild(f);
            --g
        }
    }
};
ISQ.CSS.generateCSSstring = function(j, g) {
    if (j === null || g === null) {
        return ""
    }
    var e = [];
    e.push(g);
    e.push("{" + j + "}");
    var f = e.join("");
    return f
};
ISQ.CSS.getAttributeValueFromStyleString = function(k, f) {
    var l = 0;
    var j = k.indexOf(f);
    if (j>-1) {
        j = k.indexOf(":", j);
        var g = k.indexOf(";", j);
        if (g>-1) {
            l = k.substring(j + 1, g)
        } else {
            l = k.substring(j + 1)
        }
    }
    return l
};
ISQ.CSS.getBackgroundImage = function(k) {
    var l = "";
    var f = ISQ.CSS.getAttributeValueFromStyleString(k, "background-image");
    if (f === 0) {
        f = ISQ.CSS.getAttributeValueFromStyleString(k, "background")
    }
    var j = f.indexOf("url(");
    if (j>-1) {
        j += 4;
        var g = f.indexOf(")", j);
        g -= 1;
        l = f.substring(j, g);
        l = l.replace('"', "")
    }
    return l
};
ISQ.CSS.getSelectorStyle = function(B, C) {
    if (typeof(B) !== "string") {
        return 
    }
    B = B.toLowerCase();
    if (typeof(C) !== "string") {
        C = null
    } else {
        C = C.toLowerCase()
    }
    var A = "cssRules";
    if (ISQ.Http.browser.app === "ie"&&!ISQ.Http.browser.isIE11) {
        A = "rules"
    }
    try {
        var x = document.styleSheets.length;
        var F = new Array();
        for (var n = x - 1; n >= 0; --n) {
            var G = document.styleSheets[n];
            try {
                if (G.disabled) {
                    continue
                }
                var z = G[A].length;
                for (var w = z - 1; w >= 0; --w) {
                    var E = G[A][w].selectorText;
                    if (!E) {
                        continue
                    }
                    var D = E.toLowerCase().split(",");
                    var y = D.length;
                    for (h = y - 1; h >= 0; --h) {
                        var l = D[h].Trim();
                        if (B === l || (C !== null && C === l)) {
                            var H;
                            for (var j in G[A][w].style) {
                                if (F[j]) {
                                    continue
                                }
                                H = G[A][w].style[j];
                                if (!H) {
                                    continue
                                }
                                F[j] = H
                            }
                        }
                    }
                }
            } catch (m) {}
        }
        return F
    } catch (m) {}
    return null
};
ISQ.CSS.getValueBySelectors = function(f, j, k) {
    if (typeof(f) !== "string") {
        return 
    }
    if (typeof(j) !== "string") {
        return 
    }
    if (typeof(k) !== "string") {
        k = null
    }
    var g = ISQ.CSS.getSelectorStyle(j, k);
    if (g === null) {
        return null
    }
    value = g[f];
    if (typeof(value) !== "string") {
        return null
    }
    var l = value.Trim();
    if (l === "") {
        return null
    }
    return l
};
ISQ.CSS.changeValueBySelector = function(j, e, g) {
    if (typeof(e) !== "string") {
        return 
    }
    if (typeof(j) !== "string") {
        return 
    }
    if (typeof(g) !== "string") {
        return 
    }
    var f = ISQ.CSS.getSelectorStyle(j);
    if (!f) {
        return null
    }
    f[e] = g
};
ISQ.CSS.getValue = function(n, m) {
    var k = ISQ.Html._(n);
    if (k === null) {
        return null
    }
    var l = "#" + k.id;
    var j = "." + k.className;
    var o = k.nodeName.toLowerCase();
    m = m.toLowerCase();
    var q = null;
    q = k.style[m];
    if (q) {
        return q.Trim()
    }
    var p = ISQ.CSS.getValueBySelectors(m, j, l);
    if (p === null) {
        return ISQ.CSS.getValueBySelectors(m, o + j, o + l)
    }
};
ISQ.Widget = initializeNS("ISQ.Widget");
ISQ.API = initializeNS("ISQ.API");
ISQ.Widget.supportsPostMessage = (typeof(window.postMessage) === "function" || typeof(window.postMessage) === "object");
ISQ.Widget.getInnerFrameUrl = function(s, x, p) {
    s = s || "CDCframe.html";
    var m = x.indexOf("://") + 3;
    var r;
    var n;
    var q = "";
    if (m !== 2) {
        r = x.indexOf("/", m);
        q = x.substring(0, m);
        n = x.substring(m, r)
    } else {
        r = x.indexOf("/");
        n = x.substring(0, r)
    }
    var t = s.indexOf("://") + 3;
    var w;
    var u;
    var v = "";
    if (t !== 2) {
        w = s.indexOf("/", t);
        v = s.substring(0, t);
        u = s.substring(t, w)
    } else {
        w = s.indexOf("/");
        u = s.substring(0, w)
    }
    if (q && u === n) {
        if (v !== q) {
            return s.replace(v, q)
        }
        return s
    } else {
        if (p) {
            if (u === ISQ.Http.domain()) {
                return s
            }
            return ""
        }
    }
    var o = "";
    if (s.charAt(0) === "/") {
        o = q + n + s
    } else {
        if (t !== 2) {
            o = q + n + s.substring(w)
        } else {
            o = q + n + "/" + s
        }
    }
    return o
};
ISQ.Widget.extractDataFromScript = function(o) {
    var n = {};
    var p = document.getElementsByTagName("script");
    var q, j;
    for (var k = 0; k < p.length; ++k) {
        q = p[k].getAttribute("src");
        if (!q) {
            continue
        }
        if (!q.contains(o)) {
            continue
        }
        if (!q.contains("nanorep.com") && q.startsWith("http")) {
            continue
        }
        var l =- 1;
        if (q.startsWith("/")) {
            l = window.location.href.indexOf("/", 8);
            if (l===-1) {
                n.serverUrl = window.location.href
            } else {
                n.serverUrl = window.location.href.substr(0, l)
            }
        } else {
            l = q.indexOf("/", 8);
            if (l!==-1) {
                n.serverUrl = q.substr(0, l)
            }
        }
        q = q.substring(q.indexOf("?") + 1);
        j = ISQ.Tools.stringToPairArray(q, "=", "&", true, false);
        n.cdcFrame = j.innerframeurl;
        if (!n.cdcFrame) {
            n.cdcFrame = j.cdcframe
        }
        if (n.cdcFrame) {
            n.cdcVersion = j.cdcversion;
            if (!n.cdcVersion) {
                n.cdcVersion = 1
            }
            n.cdcVersion = parseInt(n.cdcVersion)
        }
        n.account = j.account;
        var m = j.preview;
        if (m === "true") {
            n.preview = true;
            n.referer = j.referer
        }
        n.debug = j.debug || false;
        break
    }
    return n
};
ISQ.Widget.appendConfigurationsToIframeSource = function(c, d) {
    c.push("#account=");
    c.push(d.account);
    if (d.kb) {
        c.push("&kb=");
        c.push(d.kb)
    }
    if (d.splitTestGroup) {
        c.push("&splittestgroup=");
        c.push(d.splitTestGroup)
    }
    if (d.preview) {
        c.push("&preview=true")
    }
    c.push("&doctype=");
    c.push(ISQ.Widget.doctype);
    c.push("&refencoded=true&referer=");
    c.push(ISQ.Http.Cookies.encode64(d.referer));
    if (!ISQ.Widget.supportsPostMessage && d.cdcFrame) {
        c.push("&cdcFrame=");
        c.push(ISQ.Widget.getInnerFrameUrl(d.cdcFrame, window.location.href));
        c.push("&cdcversion=");
        c.push(d.cdcVersion)
    }
    if (ISQ.Widget.supportsPostMessage) {
        c.push("&cnf=true")
    }
    if (d.pageId) {
        c.push("&pageId=");
        c.push(d.pageId)
    }
    if (d.searchDelay) {
        c.push("&searchDelay=");
        c.push(d.searchDelay)
    }
    c.push("&reportEvents=true");
    if (d.debug) {
        c.push("&debug=isq")
    }
    ISQ.Widget.appendCustomValuesToStringBuilder(c, d.customValues)
};
ISQ.Widget.appendCustomValuesToStringBuilder = function(g, e) {
    if (!e) {
        return 
    }
    var j;
    for (var f in e) {
        j = typeof(e[f]);
        if (j === "function" || j === "object") {
            continue
        }
        g.push("&");
        g.push(f);
        g.push("=");
        if (j === "string") {
            if (f === "onloadQuestionId" || f === "languageCode" || f == "configId") {
                g.push(e[f])
            } else {
                g.push(ISQ.Http.Cookies.encode64(e[f]))
            }
        } else {
            g.push(e[f])
        }
    }
};
ISQ.Widget.downloadAndSendCNF = function(d) {
    if (!ISQ.Widget.CNFStatus) {
        ISQ.Widget.CNFStatus = []
    }
    var c = ISQ.Widget.CNFStatus[d.id] = {
        downloaded: false,
        waitingForDownload: false
    };
    if (!window._nRepData.eCNFLoaded) {
        window._nRepData.eCNFLoaded = new ISQ.Event()
    }
    if (d.useExistingCNF && ((d.id === "float" && typeof(ISQ.Cnf_Float) === "object") || (d.id === "embed" && typeof(ISQ.Cnf_Embed) === "object"))) {
        if (!ISQ.Widget.supportsPostMessage) {
            return false
        }
        ISQ.Widget.cnfLoaded(d);
        d.useExistingCNF = false;
        c.downloaded = true;
        return true
    }
    ISQ.Widget.downloadCnf(d, function() {
        ISQ.Widget.cnfLoaded(d);
        if (!ISQ.Widget.supportsPostMessage) {
            return 
        }
        c.downloaded = true;
        if (!c.waitingForDownload) {
            return 
        }
        if (d.type === "float") {
            ISQ.Widget.sendCNFData(ISQ.Cnf_Float, d)
        } else {
            ISQ.Widget.sendCNFData(ISQ.Cnf_Embed, d)
        }
    });
    return ISQ.Widget.supportsPostMessage
};
ISQ.Widget.cnfLoaded = function(widgetData) {
    var cnf = null;
    if (widgetData.type === "float") {
        cnf = ISQ.Cnf_Float
    } else {
        cnf = ISQ.Cnf_Embed
    }
    if (!cnf) {
        return 
    }
    var chatConfig = eval("(" + cnf.chatConfiguration + ")");
    if (chatConfig.active === "LP") {
        if (typeof(lpMTagConfig) != "undefined" && typeof(lpMTagConfig.LPSID_VAR) != "undefined") {
            chatConfig.LP.visitorId = lpMTagConfig.LPSID_VAR;
            cnf.chatConfiguration = ISQ.Data.jsonToString(chatConfig)
        } else {
            var lptagInterval = setInterval(function() {
                if (typeof(lpMTagConfig) != "undefined" && typeof(lpMTagConfig.LPSID_VAR) != "undefined") {
                    stringBuilder = [];
                    stringBuilder.push("subject=visitorId&data=");
                    stringBuilder.push(lpMTagConfig.LPSID_VAR);
                    widgetData.cdcUser.sendMessage(stringBuilder.join(""));
                    clearInterval(lptagInterval)
                }
            }, 500)
        }
    }
    window._nRepData.eCNFLoaded.dispatch(widgetData.id);
    if (widgetData.eCnfLoaded) {
        widgetData.eCnfLoaded()
    }
};
ISQ.Widget.downloadCnf = function(n, g) {
    var m = null;
    if (n.preview) {
        m = escape(n.referer)
    } else {
        if (n.customValues && n.customValues.configId) {
            m = "!" + n.customValues.configId
        }
    }
    var k = document.getElementsByTagName("head")[0];
    var j = createElement("script");
    j.setAttribute("type", "text/javascript");
    j.setAttribute("src", ISQ.Widget.buildCNFSource(n.serverUrl, n.account, n.type === "float", n.kb, m, n.preview, n.splitTestGroup));
    j.async = true;
    j.defer = true;
    if (g) {
        var l = function() {
            if (ISQ.Referer.handlerObjects[j.src].pendingLoad === true) {
                return 
            }
            g()
        };
        ISQ.Referer = initializeNS("ISQ.Referer");
        ISQ.Referer.handlerObjects = ISQ.Referer.handlerObjects || [];
        ISQ.Referer.handlerObjects[j.src] = {
            callback: g
        };
        if (ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE11) {
            j.onload = l
        } else {
            j.onreadystatechange = function() {
                if (this.readyState !== "complete" && this.readyState !== "loaded") {
                    return 
                }
                l();
                l = null
            }
        }
    }
    k.appendChild(j);
    return true
};
ISQ.Widget.buildCNFSource = function(p, j, k, l, o, n, q) {
    var m = [];
    if (p) {
        m.push(p)
    }
    m.push("/~");
    m.push(j.toLowerCase());
    m.push("/widget/scripts/cnf.js?account=");
    m.push(j);
    m.push("&key=");
    m.push(ISQ.Http.Cookies.encode64(ISQ.Http.domain(true, o ? o : null)));
    if (q) {
        m.push("&splittestgroup=");
        m.push(q)
    }
    if (o) {
        m.push("&referer=");
        m.push(o)
    }
    if (k) {
        m.push("&isFloat=true")
    }
    if (l) {
        m.push("&kb=");
        m.push(l)
    }
    if (n) {
        m.push("&preview=true")
    }
    if (ISQ.Http.browser.isIPhone) {
        m.push("&d=1")
    }
    return m.join("")
};
ISQ.Widget.sendCNFData = function(k, p) {
    var l = ISQ.Widget.CNFStatus[p.id];
    l.waitingForDownload = true;
    if (!l.downloaded) {
        return 
    }
    var j = 1800;
    var m = function(b, a) {
        stringBuilder = [];
        stringBuilder.push("subject=cnf&data=");
        stringBuilder.push(b);
        if (a) {
            stringBuilder.push("&finished=true")
        }
        p.cdcUser.sendMessage(stringBuilder.join(""))
    };
    var o = ISQ.Data.jsonToString(k);
    o = ISQ.Http.Cookies.encode64(o);
    var n = 0;
    if (j < o.length) {
        while (n < o.length) {
            if (n + j > o.length) {
                j = o.length - n
            }
            m(o.substr(n, j));
            n += j
        }
        m("", true)
    } else {
        m(o, true)
    }
    if (p.eAfterCnfSent) {
        p.eAfterCnfSent()
    }
};
ISQ.Widget.sendingMessagesToWidget = function(b) {
    return ISQ.Widget.supportsPostMessage || b.formValues instanceof Array || typeof(b.context) === "object" || typeof(b.customParams) === "object"
};
ISQ.Widget.initCDC = function(j, f) {
    if (ISQ.Widget.supportsPostMessage) {
        j.cdcVersion = 3
    } else {
        if (!j.cdcVersion) {
            j.cdcVersion = 0
        }
    }
    var g = ISQ.Widget.sendingMessagesToWidget(j);
    j.cdcUser = null;
    switch (j.cdcVersion) {
    case 0:
    case 1:
    case 2:
        if (j.cdcVersion !== 0) {
            ISQ.Widget.cdcMessageEvent.addHandler(f, j)
        }
        if (g) {
            j.cdcUser = new ISQ.CDC.User({
                targetWindowName: j.iframe.id,
                proxyIFrameUrl: j.serverUrl + "/common/cdc/cdc.html",
                server: j.serverUrl
            })
        }
        break;
    case 3:
        var e = {
            allowMessagesFrom: j.serverUrl,
            widget: j.type
        };
        if (g) {
            e.server = j.serverUrl;
            e.targetWindowName = j.iframe.id;
            e.proxyIFrameUrl = j.serverUrl + "/common/cdc/cdc.html"
        }
        j.cdcUser = new ISQ.CDC.User(e);
        j.cdcUser.incomingMessageEvent.addHandler(f, j);
        break
    }
};
ISQ.Widget.APIKeyEndDelimiter = "<";
ISQ.Widget.APIValueEndDelimiter = "(";
ISQ.Widget.sendAPItoWidget = function(w) {
    var s = ISQ.Widget.sendingMessagesToWidget(w);
    if (!w.cdcUser ||!s) {
        return 
    }
    var t = [];
    t.push("subject=api");
    var p = w.formValues;
    if ((p instanceof Array) && p.length !== 0) {
        t.push("&form=");
        var r, v, u;
        for (var q = 0; q < p.length; ++q) {
            var o = p[q];
            if (!(o instanceof Array) || o.length < 2) {
                continue
            }
            if (o.length === 2) {
                r = o[0];
                v = o[1];
                u = null
            } else {
                r = o[1];
                u = o[0];
                v = o[2]
            }
            if (q !== 0) {
                t.push(ISQ.Widget.APIKeyEndDelimiter)
            }
            t.push(ISQ.Http.Cookies.encode64(r));
            t.push(ISQ.Widget.APIValueEndDelimiter);
            t.push(ISQ.Http.Cookies.encode64(v));
            if (u) {
                t.push(ISQ.Widget.APIValueEndDelimiter);
                t.push(ISQ.Http.Cookies.encode64(u))
            }
        }
    }
    var m = w.context;
    if (typeof(m) === "object") {
        t.push("&context=");
        ISQ.Widget.CDC_appendKeyValueToMessage(m, t)
    }
    var n = w.customParams;
    if (typeof(n) === "object") {
        t.push("&customparams=");
        ISQ.Widget.CDC_appendKeyValueToMessage(n, t)
    }
    w.cdcUser.sendMessage(t.join(""))
};
ISQ.Widget.CDC_getKeyValueArrayFromMessage = function(m, l) {
    var k;
    if (l) {
        k = {}
    } else {
        k = []
    }
    var q = m.split(ISQ.Widget.APIKeyEndDelimiter);
    var p, j, o;
    for (var n = 0; n < q.length; ++n) {
        p = q[n].split(ISQ.Widget.APIValueEndDelimiter);
        j = ISQ.Http.Cookies.decode64(p[0]);
        o = ISQ.Http.Cookies.decode64(p[1]);
        k[j] = o
    }
    return k
};
ISQ.Widget.CDC_appendKeyValueToMessage = function(e, j) {
    var f = true;
    for (var g in e) {
        if (!f) {
            j.push(ISQ.Widget.APIKeyEndDelimiter)
        }
        f = false;
        j.push(ISQ.Http.Cookies.encode64(g));
        j.push(ISQ.Widget.APIValueEndDelimiter);
        j.push(ISQ.Http.Cookies.encode64(e[g]))
    }
};
ISQ.Widget.fixStealFocus = function(f) {
    if (!f.iframe) {
        return 
    }
    var e = f.setFocus;
    if (!e) {
        return 
    }
    e = ISQ.Html._(e);
    if (!e) {
        return 
    }
    var d = function() {
        try {
            e.focus()
        } catch (a) {}
    };
    if (window.attachEvent) {
        f.iframe.attachEvent("onload", d)
    } else {
        if (window.addEventListener) {
            f.iframe.addEventListener("load", d, true)
        }
    }
};
ISQ.Widget.resizeIframeOnLoad_IE = function(g, j, e, f) {
    if (ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE11) {
        return false
    }
    if (ISQ.Http.browser.isIE6 || ISQ.Http.browser.isIE7) {
        if (document.readyState === "complete") {
            return false
        }
        if (!g.registeredOnload) {
            g.registeredOnload = true;
            window.attachEvent("onload", function() {
                f(g, parseInt(j), parseInt(e) + 1)
            })
        }
        return true
    } else {
        if (document.readyState !== "complete" || window != window.top) {
            f(g, parseInt(j), parseInt(e) + 1);
            g.cdcUser.sendMessage("subject=refreshSize");
            return true
        } else {
            return false
        }
    }
};
ISQ.Widget.buildNRLinkUrl = function(j, m, p) {
    var o = new Array();
    o.push("http://widget.nanorep.com/From-Widget/?s=poweredBy&aid=");
    o.push(j);
    if (m) {
        o.push("&utm_source=");
        var n = ISQ.Http.domain(false, m).split(".");
        var l = o.length;
        for (var k = n.length - 1; k >= 0; k--) {
            o.splice(l, 0, n[k]);
            if (n[k].length > 3) {
                break
            }
            if (k > 0) {
                o.splice(l, 0, ".")
            }
        }
    }
    o.push("&utm_medium=widget&utm_content=");
    o.push(p);
    return o.join("")
};
ISQ.Widget.getAttachmentUrl = function(f, d) {
    var e = f.indexOf("/");
    if (e==-1) {
        return 
    }
    d.push("http://");
    d.push(f.substring(0, e));
    d.push(".s3.amazonaws.com/");
    d.push(f.substring(e + 1))
};
ISQ.Widget.incomingCDCMessageOrginizer = null;
ISQ.Widget.incomingCDCMessage = function(o) {
    if (o.charAt(0) === "?") {
        o = o.substr(1)
    }
    var p = ISQ.Tools.stringToPairArray(o, "=", "&", true, false);
    if (!ISQ.Widget.incomingCDCMessageOrginizer) {
        ISQ.Widget.incomingCDCMessageOrginizer = []
    }
    var e = ISQ.Widget.incomingCDCMessageOrginizer[p.widget];
    if (!e) {
        e = ISQ.Widget.incomingCDCMessageOrginizer[p.widget] = {};
        e.lastHandledOrdinal = 0;
        e.pendingMessages = null
    }
    if (p.subject && p.subject == "reset") {
        e.lastHandledOrdinal = 0;
        e.pendingMessages = null
    }
    var s = parseInt(p.ord);
    var l = s - e.lastHandledOrdinal;
    if (l <= 0) {
        return 
    }
    if (l > 1) {
        if (e.pendingMessages === null) {
            e.pendingMessages = [];
            e.pendingMessagesCount = 0
        }
        e.pendingMessages[s] = p;
        ++e.pendingMessagesCount
    } else {
        try {
            ISQ.Widget.cdcMessageEvent.dispatch(p)
        } catch (m) {}
        if (e.pendingMessages !== null) {
            var r = e.pendingMessagesCount;
            var q = s + 1;
            for (var n = 0; n < r; ++n, ++q) {
                if (!e.pendingMessages[q]) {
                    break
                }
                ISQ.Widget.cdcMessageEvent.dispatch(e.pendingMessages[q]);
                s = q;
                --e.pendingMessagesCount
            }
            if (e.pendingMessagesCount === 0) {
                e.pendingMessages = null
            }
        }
        e.lastHandledOrdinal = s
    }
};
if (!ISQ.Widget.cdcMessageEvent) {
    ISQ.Widget.cdcMessageEvent = new ISQ.Event()
}
ISQ.Widget.handleCommonCDCMessage = function(messageArgs, widgetData) {
    switch (messageArgs.subject) {
    case"cnf":
        if (widgetData.id === "float") {
            ISQ.Widget.sendCNFData(ISQ.Cnf_Float, widgetData)
        } else {
            ISQ.Widget.sendCNFData(ISQ.Cnf_Embed, widgetData)
        }
        break;
    case"event":
        ISQ.Widget.handleEventMessage(messageArgs, widgetData);
        break;
    case"pagescript":
        ISQ.API.runPageScript(messageArgs.data, true);
        break;
    case"chatResumeRequest":
        if (_nRepData.chatInResume) {
            ISQ.Widget.preventChatResuming(widgetData);
            break
        }
        if (!_nRepData.chatInResume) {
            _nRepData.chatInResume = true
        }
        ISQ.Widget.aboutToResumeChat(widgetData);
        break;
    case"chatEnded":
        _nRepData.chatInResume = false;
        break;
    case"loadChat":
    case"startCustomChat":
        var config = eval("(" + messageArgs.config + ")");
        switch (config.active) {
        case"LiveChatInc":
            ISQ.Widget.Proxy.Chat.LiveChatInc.start(messageArgs.subject, config, widgetData);
            break;
        case"Zopim":
            ISQ.Widget.Proxy.Chat.Zopim.start(messageArgs.subject, config, widgetData);
            break;
        case"Olark":
            ISQ.Widget.Proxy.Chat.Olark.start(messageArgs.subject, config, widgetData);
            break;
        case"SnapEngage":
            ISQ.Widget.Proxy.Chat.SnapEngage.start(messageArgs.subject, config, widgetData);
            break
        }
        break
    }
};
ISQ.Widget.aboutToResumeChat = function(b) {
    b.cdcUser.sendMessage("subject=chatResumeApproved")
};
ISQ.Widget.preventChatResuming = function(b) {
    b.cdcUser.sendMessage("subject=preventChatResuming")
};
ISQ.Widget.doctype = ISQ.Html.detectDoctype();
ISQ.Widget.createMobileBlockerLink = function(m, j, n, k) {
    if (!ISQ.Http.browser.isMobile) {
        return
    }
    var a = m.mobileBlockerLink = document.createElement("a");
    a.className = "nR_noRemove";
    a.style.cssText = "display:block;width: " + n + ";height:" + m.height + "px;position:relative;left:0px;top:-" + m.height + "px;z-index:1001;background:transparent;";
    j.appendChild(a);
    a.href = m.iframe.src;
    a.target = "_blank";
    a.onclick = function(b) {
        if (m.backImgDiv && ISQ.Cnf_Float.floatBackImageAsk) {
            this.style.display = "none";
            var c = document.elementFromPoint(b.clientX, b.clientY);
            while (c != null && c !== m.backImgDiv && c !== document.body) {
                c = c.parentNode
            }
            this.style.display = "block";
            if (c == m.backImgDiv && ISQ.Cnf_Float.floatBackImageAsk && ISQ.Cnf_Float.floatBackImageAsk > 0) {
                var d = this.href;
                if (!this.hrefLength) {
                    this.hrefLength = d.length
                }
                if (d.contains("#")) {
                    d += "&onloadquestionid=" + ISQ.Cnf_Float.floatBackImageAsk
                } else {
                    d += "#onloadquestionid=" + ISQ.Cnf_Float.floatBackImageAsk
                }
                this.href = d
            } else {
                if (this.hrefLength && this.href.length != this.hrefLength) {
                    this.href = this.href.substring(0, this.hrefLength)
                }
            }
        }
        m.cdcUser.sendMessage("subject=mobileOpenNewWindow")
    };
    var l = _("ozen");
    if (m.id === "float" && l && l.nodeName === "A") {
        l.href = m.iframe.src;
        l.target = "_blank";
        l.onclick = function(b) {
            if (m.backImgDiv && ISQ.Cnf_Float.floatBackImageAsk) {
                this.style.display = "none";
                var c = document.elementFromPoint(b.clientX, b.clientY);
                while (c != null && c !== m.backImgDiv && c !== document.body) {
                    c = c.parentNode
                }
                this.style.display = "block";
                if (c == m.backImgDiv && ISQ.Cnf_Float.floatBackImageAsk && ISQ.Cnf_Float.floatBackImageAsk > 0) {
                    var d = this.href;
                    if (!this.hrefLength) {
                        this.hrefLength = d.length
                    }
                    if (d.contains("#")) {
                        d += "&onloadquestionid=" + ISQ.Cnf_Float.floatBackImageAsk
                    } else {
                        d += "#onloadquestionid=" + ISQ.Cnf_Float.floatBackImageAsk
                    }
                    this.href = d
                } else {
                    if (this.hrefLength && this.href.length != this.hrefLength) {
                        this.href = this.href.substring(0, this.hrefLength)
                    }
                }
            }
            m.cdcUser.sendMessage("subject=mobileOpenNewWindow")
        }
    }
};
ISQ.Widget.addCssStyles = function(d, c) {
    if (typeof(d) === "string" && d) {
        if (d.contains(" ")) {
            ISQ.CSS.addNewStyle(d, c)
        } else {
            ISQ.CSS.addNewStyle(ISQ.Tools.Base64.decode(d), c)
        }
    }
};
ISQ.Widget.replacePlaceholders = function(D, B, F, t, u) {
    var C = {};
    C.success = false;
    C.text = D;
    if (!D) {
        C.success = true;
        return C
    }
    var q = new Array();
    var E = 0;
    var v = 0;
    var w = ["|", ":"];
    while (E < D.length) {
        E = D.indexOf(F, v);
        if (E===-1) {
            q.push(D.substring(v));
            break
        }
        q.push(D.substring(v, E));
        v = D.indexOf(t, E);
        if (v==-1) {
            q.push(D.substring(E));
            break
        }
        var z = D.substring(E + 2, v);
        v += 2;
        var A = z.split("|");
        var s;
        for (var y = 0; y < A.length; ++y) {
            z = A[y];
            if (y == A.length - 1) {
                var r = z.indexOf(":");
                if (r!==-1) {
                    s = z.substring(r + 1);
                    z = z.substring(0, r)
                }
            }
            if (!B) {
                if (y == A.length - 1) {
                    x = s
                }
                continue
            } else {
                var x = B[z];
                if (x) {
                    break
                }
                x = s
            }
        }
        if (!x) {
            if (!u) {
                return C
            }
            x = ""
        }
        q.push(x)
    }
    C.text = q.join("");
    C.success = true;
    return C
};
ISQ.Widget.combineContext = function(m, n, o) {
    if (!m) {
        return n
    }
    if (n) {
        for (var p in n) {
            var q = p.toLowerCase();
            n[q] = n[p];
            if (q !== p) {
                delete n[p]
            }
        }
    }
    var s = null;
    if (n || o) {
        for (var l in m) {
            if (n && n[l.toLowerCase()]) {
                continue
            }
            for (var k = 0; k < m[l].length; ++k) {
                var r = null;
                if (o) {
                    r = o[m[l][k]]
                }
                if (!r) {
                    continue
                }
                if (!s) {
                    s = [];
                    for (var t in n) {
                        s[t] = n[t]
                    }
                }
                s[l.toLowerCase()] = r;
                break
            }
        }
    }
    return s || n
};
ISQ.Widget.Mobile = {};
ISQ.Widget.Mobile.resizeStart = function() {
    if (!ISQ.Http.browser.isMobile || ISQ.Http.browser.oldAndroid) {
        return 
    }
    if (!ISQ.Widget.Mobile.resizeInterval) {
        ISQ.Widget.Mobile.resizeInterval = setInterval(ISQ.Widget.Mobile.resizeTimerElapsed, 200)
    }
    window.addEventListener("orientationchange", ISQ.Widget.Mobile.orientationChanged)
};
ISQ.Widget.Mobile.resizeStop = function() {
    if (ISQ.Widget.Mobile.resizeInterval) {
        clearInterval(ISQ.Widget.Mobile.resizeInterval)
    }
    ISQ.Widget.Mobile.resizeInterval = null;
    window.removeEventListener("orientationchange", ISQ.Widget.Mobile.orientationChanged)
};
ISQ.Widget.Mobile.resizeTimerElapsed = function() {
    var d = ISQ.Widget.Mobile.sizerDiv;
    if (!d) {
        d = createDiv(document.body);
        d.style.cssText = "width:0; height:0; position:fixed; bottom:0;right:0;";
        ISQ.Widget.Mobile.sizerDiv = d
    }
    if (d.offsetLeft != ISQ.Widget.Mobile.oldViewportWidth || ISQ.Widget.Mobile.forceNextAction) {
        if (ISQ.Cnf_Embed) {
            ISQ.Widget.Mobile.forceNextAction = ISQ.Embed.Mobile.resizeAction(d.offsetLeft, d.offsetTop)
        }
        ISQ.Widget.Mobile.oldViewportWidth = d.offsetLeft;
        var c = ISQ.Cnf_Float || ISQ.Cnf_Embed;
        if (!c.mobileWidgetEnabled) {
            return 
        }
        ISQ.Widget.Mobile.forceNextAction|=ISQ.Widget.Mobile.resizeAction(d.offsetLeft, d.offsetTop)
    }
};
ISQ.Widget.Mobile.orientationChanged = function() {
    ISQ.Widget.Mobile.oldViewportWidth = null
};
ISQ.Widget.Mobile.resizeAction = function(d, c) {
    if (_nRepData["float"]&&!_nRepData["float"].widgetLoaded) {
        return true
    }
    if (!ISQ.Widget.Mobile.teaser) {
        ISQ.Widget.Mobile.buildTeaserElement(d, c);
        ISQ.Widget.Mobile.teaser.scale(d, c);
        return false
    }
    if (ISQ.Widget.Mobile.timer) {
        clearTimeout(ISQ.Widget.Mobile.timer)
    }
    ISQ.Widget.Mobile.teaser.style.display = "none";
    ISQ.Widget.Mobile.timer = setTimeout(function() {
        ISQ.Widget.Mobile.teaser.scale(d, c)
    }, 200);
    return false
};
ISQ.Widget.Mobile.buildTeaserElement = function(s, r) {
    var t = _nRepData["float"] || _nRepData.embed;
    var k = ISQ.Cnf_Float || ISQ.Cnf_Embed;
    var l = _nRepData["float"] && _nRepData["float"].dockMethod == nanoRep.Float.enumDockMethod.RIGHT;
    var q = ISQ.Widget.Mobile.teaser = createDiv(document.body, "visibility:hidden; text-align:center; width:100%; bottom:0; z-index:1000; font-family:Arial;");
    q.style.color = k.titleColor || "#FFF";
    q.style.backgroundColor = k.titleBGColor || "#009FD4";
    q.style.position = "fixed";
    if (ISQ.Tools.isLanguageRTL(k.kbLanguageCode)) {
        q.style.direction = "rtl"
    }
    q.heightRatio_portrait = 1 / 18;
    q.heightRatio_landscape = 1 / 9;
    q.fontRatio = 0.8;
    var o = r > s ? q.heightRatio_portrait: q.heightRatio_landscape;
    var n = r * o;
    var m = Math.floor(n * q.fontRatio) + "px";
    q.style.height = Math.ceil(n) + "px";
    q.style.lineHeight = Math.ceil(n) + "px";
    q.style.fontSize = m;
    q.style.bottom =- 1 * Math.ceil(n / 3) + "px";
    var p = createElement("span", q, "text-overflow:nowrap; font-size: inherit; color: inherit;");
    p.innerHTML = k.titleTeaserText;
    q.textSpan = p;
    q.onclick = function() {
        ISQ.Widget.Mobile.openWidgetInNewWindow(t)
    };
    q.widthRatio = n / (p.offsetWidth * 1.15);
    q.leftRatio = 1 / 5;
    q.paddingRatio = 1 / 9;
    q.borderRatio = 1 / 5;
    q.scale = function(f, e) {
        var c = e > f ? this.heightRatio_portrait: this.heightRatio_landscape;
        var b = e * c;
        var g = b / this.widthRatio;
        this.style.width = Math.ceil(g) + "px";
        this.style.height = this.style.lineHeight = Math.ceil(b) + "px";
        var d = Math.ceil(b * this.leftRatio) + "px";
        if (!l) {
            this.style.left = d
        } else {
            this.style.right = d
        }
        this.style.letterSpacing = g > 350 ? "-2px" : g > 150 ? "-1px" : "";
        this.style.fontSize = Math.floor(b * this.fontRatio) + "px";
        var a = Math.ceil(b * this.borderRatio);
        this.style.borderRadius = a + "px " + a + "px 0 0";
        this.style.visibility = "visible";
        this.style.display = "block"
    };
    nanoRep.Widget.PositionAnimator.setBottom(q, 0);
    nanoRep.Widget.PositionAnimator.start();
    return q
};
ISQ.Widget.Mobile.openWidgetInNewWindow = function(b) {
    b.cdcUser.sendMessage("subject=mobileOpenNewWindow");
    window.open(b.iframe.src)
};
ISQ.Widget.Mobile.forceNextAction = false;
nanoRep.Widget.PositionAnimator = {};
nanoRep.Widget.PositionAnimator.element = null;
nanoRep.Widget.PositionAnimator.side = 0;
nanoRep.Widget.PositionAnimator.bottom = 0;
nanoRep.Widget.PositionAnimator.dockRight = false;
nanoRep.Widget.PositionAnimator.isRunning = false;
nanoRep.Widget.PositionAnimator.timer = null;
nanoRep.Widget.PositionAnimator.totalTime = 1000;
nanoRep.Widget.PositionAnimator.remainingTimeBottom = 0;
nanoRep.Widget.PositionAnimator.remainingTimeSide = 0;
nanoRep.Widget.PositionAnimator.factorBottom = 0;
nanoRep.Widget.PositionAnimator.factorSide = 0;
nanoRep.Widget.PositionAnimator.interval = 20;
nanoRep.Widget.PositionAnimator.lastTimerDate = 0;
nanoRep.Widget.PositionAnimator.stopAnimationInterval = 100;
nanoRep.Widget.PositionAnimator.setSide = function(e, f, d) {
    nanoRep.Widget.PositionAnimator.element = e;
    nanoRep.Widget.PositionAnimator.side = f;
    nanoRep.Widget.PositionAnimator.remainingTimeSide = nanoRep.Widget.PositionAnimator.totalTime - 100;
    nanoRep.Widget.PositionAnimator.factorSide = Math.abs(nanoRep.Widget.PositionAnimator.side - nanoRep.Widget.PositionAnimator.getCurrentSide()) / 20;
    nanoRep.Widget.PositionAnimator.dockRight = d
};
nanoRep.Widget.PositionAnimator.setBottom = function(c, d) {
    if (d == 0) {
        d =- 1
    }
    nanoRep.Widget.PositionAnimator.element = c;
    nanoRep.Widget.PositionAnimator.bottom = d;
    nanoRep.Widget.PositionAnimator.remainingTimeBottom = nanoRep.Widget.PositionAnimator.totalTime - 100;
    nanoRep.Widget.PositionAnimator.factorBottom = Math.abs(nanoRep.Widget.PositionAnimator.bottom - nanoRep.Widget.PositionAnimator.getCurrentBottom()) / 20
};
nanoRep.Widget.PositionAnimator.start = function() {
    if (!nanoRep.Widget.PositionAnimator.side&&!nanoRep.Widget.PositionAnimator.bottom) {
        return
    }
    if (nanoRep.Widget.PositionAnimator.isRunning) {
        return 
    }
    nanoRep.Widget.PositionAnimator.isRunning = true;
    nanoRep.Widget.PositionAnimator.lastTimerDate = new Date();
    nanoRep.Widget.PositionAnimator.timer = setTimeout(nanoRep.Widget.PositionAnimator.timerElapsed, 0)
};
nanoRep.Widget.PositionAnimator.getCurrentSide = function() {
    var b;
    if (nanoRep.Widget.PositionAnimator.dockRight) {
        b = parseInt(nanoRep.Widget.PositionAnimator.element.style.right)
    } else {
        b = parseInt(nanoRep.Widget.PositionAnimator.element.style.left)
    }
    if (isNaN(b)) {
        b = 0
    }
    return b
};
nanoRep.Widget.PositionAnimator.getCurrentBottom = function() {
    return parseInt(nanoRep.Widget.PositionAnimator.element.style.bottom)
};
nanoRep.Widget.PositionAnimator.timerElapsed = function() {
    var m = new Date() - nanoRep.Widget.PositionAnimator.lastTimerDate;
    if (m >= nanoRep.Widget.PositionAnimator.stopAnimationInterval) {
        nanoRep.Widget.PositionAnimator.isRunning = false;
        if (nanoRep.Widget.PositionAnimator.side !== 0) {
            if (nanoRep.Widget.PositionAnimator.dockRight) {
                nanoRep.Widget.PositionAnimator.element.style.right = nanoRep.Widget.PositionAnimator.side + "px"
            } else {
                nanoRep.Widget.PositionAnimator.element.style.left = nanoRep.Widget.PositionAnimator.side + "px"
            }
        }
        if (nanoRep.Widget.PositionAnimator.bottom !== 0) {
            nanoRep.Widget.PositionAnimator.element.style.bottom = nanoRep.Widget.PositionAnimator.bottom + "px"
        }
    }
    var k = nanoRep.Widget.PositionAnimator.getCurrentBottom();
    var l = nanoRep.Widget.PositionAnimator.getCurrentSide();
    var j, p;
    if (nanoRep.Widget.PositionAnimator.side === 0) {
        p = 0
    } else {
        p = nanoRep.Widget.PositionAnimator.side - l
    }
    if (nanoRep.Widget.PositionAnimator.bottom === 0) {
        j = 0
    } else {
        j = nanoRep.Widget.PositionAnimator.bottom - k
    }
    var o;
    if (p !== 0) {
        p/=Math.abs(p);
        var n = nanoRep.Widget.PositionAnimator.remainingTimeSide / (nanoRep.Widget.PositionAnimator.totalTime);
        n*=nanoRep.Widget.PositionAnimator.factorSide;
        if (n < 2) {
            n = 2
        }
        o = l + (p * n);
        nanoRep.Widget.PositionAnimator.remainingTimeSide -= nanoRep.Widget.PositionAnimator.interval;
        if (p > 0 && o > nanoRep.Widget.PositionAnimator.side) {
            o = nanoRep.Widget.PositionAnimator.side
        }
        if (p < 0 && o < nanoRep.Widget.PositionAnimator.side) {
            o = nanoRep.Widget.PositionAnimator.side
        }
        if (nanoRep.Widget.PositionAnimator.dockRight) {
            nanoRep.Widget.PositionAnimator.element.style.right = o + "px"
        } else {
            nanoRep.Widget.PositionAnimator.element.style.left = o + "px"
        }
        if (o === nanoRep.Widget.PositionAnimator.side) {
            nanoRep.Widget.PositionAnimator.side = 0
        }
    } else {
        nanoRep.Widget.PositionAnimator.side = 0
    }
    if (j !== 0) {
        j/=Math.abs(j);
        var n = nanoRep.Widget.PositionAnimator.remainingTimeBottom / (nanoRep.Widget.PositionAnimator.totalTime);
        n*=nanoRep.Widget.PositionAnimator.factorBottom;
        if (n < 2) {
            n = 2
        }
        o = k + (j * n);
        nanoRep.Widget.PositionAnimator.remainingTimeBottom -= nanoRep.Widget.PositionAnimator.interval;
        if (j > 0 && o > nanoRep.Widget.PositionAnimator.bottom) {
            o = nanoRep.Widget.PositionAnimator.bottom
        }
        if (j < 0 && o < nanoRep.Widget.PositionAnimator.bottom) {
            o = nanoRep.Widget.PositionAnimator.bottom
        }
        nanoRep.Widget.PositionAnimator.element.style.bottom = o + "px";
        if (o === nanoRep.Widget.PositionAnimator.bottom) {
            nanoRep.Widget.PositionAnimator.bottom = 0
        }
    } else {
        nanoRep.Widget.PositionAnimator.bottom = 0
    }
    if (nanoRep.Widget.PositionAnimator.bottom || nanoRep.Widget.PositionAnimator.side) {
        nanoRep.Widget.PositionAnimator.lastTimerDate = new Date();
        nanoRep.Widget.PositionAnimator.timer = setTimeout(nanoRep.Widget.PositionAnimator.timerElapsed, nanoRep.Widget.PositionAnimator.interval)
    } else {
        nanoRep.Widget.PositionAnimator.isRunning = false
    }
};
ISQ.API.init = function(b) {
    ISQ.API.defaultWidgetData = b
};
ISQ.API.getWidgetData = function(c) {
    var d = null;
    if (c) {
        d = _nRepData[c]
    }
    return d || ISQ.API.defaultWidgetData
};
ISQ.API.openContactForm = function(c) {
    var d = ISQ.API.getWidgetData(c);
    if (!d ||!d.cdcUser) {
        return 
    }
    if (d.id == "float") {
        if (ISQ.Http.browser.isMobile) {
            return 
        }
    } else {
        if (ISQ.Http.browser.oldAndroid) {
            return 
        }
    }
    d.cdcUser.sendMessage("subject=openContactForm")
};
ISQ.API.startChat = function(e, f) {
    var j = ISQ.API.getWidgetData(f);
    if (!j ||!j.cdcUser) {
        return 
    }
    if (j.id == "float") {
        if (ISQ.Http.browser.isMobile) {
            return 
        }
        nanoRep.Float.changeState(nanoRep.Float.enumStates.expand)
    } else {
        if (ISQ.Http.browser.oldAndroid) {
            return 
        }
    }
    if (j.autoMinimizeTimer) {
        clearTimeout(j.autoMinimizeTimer)
    }
    var g = "subject=startChat";
    if (e) {
        g += "&countStatsOnly=true"
    }
    j.cdcUser.sendMessage(g)
};
ISQ.API.customizeWidget = function(e, f) {
    if (typeof(e) != "object") {
        return 
    }
    var j = ISQ.API.getWidgetData(f);
    if (!j ||!j.cdcUser) {
        return 
    }
    var g = "subject=customize";
    if (e.bgcolor) {
        g += "&bgcolor=" + e.bgcolor
    }
    if (e.title) {
        g += "&title=" + e.title
    }
    j.cdcUser.sendMessage(g)
};
ISQ.API.ask = function(e, d) {
    if (!e) {
        e = ""
    }
    var f = ISQ.API.getWidgetData(d);
    if (!f ||!f.cdcUser) {
        return 
    }
    f.cdcUser.sendMessage("subject=ask&question=" + e)
};
ISQ.API.askById = function(j, f, k) {
    if (!j) {
        return 
    }
    var l = ISQ.API.getWidgetData(f);
    if (!l ||!l.cdcUser) {
        return 
    }
    var g = "subject=askById&questionid=" + j;
    if (k) {
        g += "&sender=" + k
    }
    l.cdcUser.sendMessage(g)
};
ISQ.API.importCustomParams = function(d) {
    if (d.table) {
        d.customParams = ISQ.API.getCustomParamsFromTable(d)
    } else {
        if (d.ul) {
            d.customParams = ISQ.API.getCustomParamsFromUl(d)
        }
    }
    if (!d.customParams) {
        return 
    }
    var c = function(f) {
        var a = f ? window._nRepData[f]: _nRepData;
        if (!a) {
            return 
        }
        if (!a.customParams) {
            a.customParams = d.customParams
        } else {
            for (var b in d.customParams) {
                if (a.customParams[b]) {
                    continue
                }
                a.customParams[b] = d.customParams[b]
            }
        }
    };
    c("");
    if (d.widget) {
        c(d.widget)
    } else {
        c("float");
        c("embed")
    }
};
ISQ.API.getCustomParamsFromTable = function(v) {
    if (!v.table) {
        return 
    }
    var z = window._(v.table);
    if (!z) {
        return 
    }
    var r = [];
    var E = z.getElementsByTagName("TR");
    var B, C, s = false, w, x, u;
    var q =- 1;
    var y = "";
    for (var D = 0; D < E.length; ++D) {
        y = "noText";
        w = x = "";
        B = [];
        C = E[D].getElementsByTagName("TH");
        for (var t = 0; t < C.length; ++t) {
            B.push(C[t])
        }
        C = E[D].getElementsByTagName("TD");
        for (var t = 0; t < C.length; ++t) {
            B.push(C[t])
        }
        q++;
        for (var A = 0; A < B.length; ++A) {
            u = B[A].innerText || B[A].textContent;
            if (!u) {
                continue
            }
            u = u.Trim();
            y = "hasText";
            if (v.skipRows && ISQ.Data.indexOfArray(v.skipRows, q)!==-1) {
                y = "skip";
                break
            } else {
                if (!w) {
                    w = u
                } else {
                    if (!x) {
                        x = u;
                        break
                    }
                }
            }
        }
        if (y === "noText") {
            --q;
            continue
        }
        if (y === "skip") {
            continue
        }
        if (!w ||!x) {
            continue
        }
        r[w] = x
    }
    return r
};
ISQ.API.getCustomParamsFromUl = function(p) {
    if (!p.ul) {
        return 
    }
    var s = window._(p.ul);
    if (!s) {
        return 
    }
    var o = s.getElementsByTagName("LI");
    var j = [];
    var m, q, r = "";
    for (var l = 0; l < o.length; ++l) {
        q = r = "";
        for (var n = 0; n < o[l].children.length; ++n) {
            if (o[l].children[n].localName == "small") {
                q = o[l].children[n].innerHTML.replace(/:/g, "").Trim()
            } else {
                if (o[l].children[n].localName == "b") {
                    r = o[l].children[n].innerHTML.Trim()
                }
            }
        }
        if (!q ||!r) {
            return 
        }
        j[q] = r
    }
    return j
};
ISQ.API.runPageScript = function(script, encoded) {
    if (!script) {
        return 
    }
    try {
        if (encoded) {
            eval(ISQ.Tools.Base64.decodeString(script))
        } else {
            eval(script)
        }
    } catch (e) {}
};
ISQ.Widget.handleEventMessage = function(messageArgs, widgetData) {
    var type = messageArgs.type;
    var handl, query;
    var timer = 3000;
    switch (type) {
    case"query":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events[type];
        query = ISQ.Tools.Base64.decodeString(messageArgs.data);
        var isAutoQuestion = messageArgs.isautoquestion == "true";
        if (handl) {
            if (isAutoQuestion) {
                handl(query, true)
            } else {
                this.queryTimer = setTimeout(function() {
                    handl(query, false)
                }, timer)
            }
            if (false) {
                if (typeof(lpMTagConfig) === "undefined") {
                    return 
                }
                if (typeof(lpMTagConfig.pageVar) === "undefined") {
                    lpMTagConfig.pageVar = []
                }
                lpMTagConfig.pageVar[lpMTagConfig.pageVar.length] = "nanoRep_question=" + query
            }
        }
        var noWaitHandl = widgetData.events.query_noWait;
        if (noWaitHandl) {
            noWaitHandl(query, isAutoQuestion)
        }
        break;
    case"nodata":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events.noData;
        if (handl) {
            handl()
        }
        break;
    case"scriptload":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events.scriptload;
        if (handl) {
            handl()
        }
        break;
    case"load":
        if (messageArgs.integrate_ga === "true") {
            var events = widgetData.events;
            if (!events) {
                events = widgetData.events = {}
            }
            var userActiveFunc = events.userActive;
            events.userActive = function() {
                if (typeof(ga) === "function") {
                    ga("send", "event", "nanoRep", "nanoRep User Active")
                }
                if (typeof(_gaq) !== "undefined") {
                    _gaq.push(["_trackEvent", "nanoRep", "nanoRep User Active"])
                }
                if (userActiveFunc) {
                    try {
                        userActiveFunc()
                    } catch (e) {
                        console.log(e)
                    }
                }
            };
            var queryFunc = events.query;
            events.query = function(queryText, isAutoQuestion) {
                if (!isAutoQuestion) {
                    if (typeof(ga) === "function") {
                        ga("send", "event", "nanoRep", "nanoRep Query", queryText)
                    }
                    if (typeof(_gaq) !== "undefined") {
                        _gaq.push(["_trackEvent", "nanoRep", "nanoRep Query", queryText])
                    }
                }
                if (queryFunc) {
                    try {
                        queryFunc(queryText, isAutoQuestion)
                    } catch (e) {
                        console.log(e)
                    }
                }
            };
            var incomingSearchResultsFunc = events.incomingSearchResults;
            events.incomingSearchResults = function(hasResults, isAutoQuestion, query, results) {
                if (hasResults&&!isAutoQuestion) {
                    if (typeof(ga) === "function") {
                        ga("send", "event", "nanoRep", "nanoRep Results Shown")
                    }
                    if (typeof(_gaq) !== "undefined") {
                        _gaq.push(["_trackEvent", "nanoRep", "nanoRep Results Shown"])
                    }
                }
                if (incomingSearchResultsFunc) {
                    try {
                        incomingSearchResultsFunc(hasResults, isAutoQuestion, query, results)
                    } catch (e) {
                        console.log(e)
                    }
                }
            };
            var escFunc = events.escalation;
            events.escalation = function(type, queryText) {
                if (typeof(ga) === "function") {
                    ga("send", "event", "nanoRep", "nanoRep Escalation", queryText)
                }
                if (typeof(_gaq) !== "undefined") {
                    _gaq.push(["_trackEvent", "nanoRep", "nanoRep Escalation", queryText])
                }
                if (escFunc) {
                    try {
                        escFunc(type, queryText)
                    } catch (e) {
                        console.log(e)
                    }
                }
            };
            var likeFunc = events.like;
            events.like = function(queryText, answer) {
                if (typeof(ga) === "function") {
                    ga("send", "event", "nanoRep", "nanoRep Like", answer)
                }
                if (typeof(_gaq) !== "undefined") {
                    _gaq.push(["_trackEvent", "nanoRep", "nanoRep Like", answer])
                }
                if (likeFunc) {
                    try {
                        likeFunc(queryText, answer)
                    } catch (e) {
                        console.log(e)
                    }
                }
            }
        }
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events[type];
        if (handl) {
            handl()
        }
        break;
    case"userstartedtyping":
        if (this.queryTimer) {
            clearTimeout(this.queryTimer)
        }
        if (this.incomingSearchResultsTimer) {
            clearTimeout(this.incomingSearchResultsTimer)
        }
        break;
    case"incomingsearchresults":
        if (!widgetData.events) {
            break
        }
        isAutoQuestion = messageArgs.isautoquestion == "true";
        var hasResults = messageArgs.hasresults === "true";
        var results = eval("(" + ISQ.Tools.Base64.decodeString(messageArgs.results) + ")");
        var query = messageArgs.query ? ISQ.Tools.Base64.decodeString(messageArgs.query): null;
        handl = widgetData.events.incomingSearchResults;
        if (handl) {
            if (isAutoQuestion) {
                handl(hasResults, true, query, results)
            } else {
                this.incomingSearchResultsTimer = setTimeout(function() {
                    handl(hasResults, false, query, results)
                }, timer)
            }
        }
        noWaitHandl = widgetData.events.incomingSearchResults_noWait;
        if (noWaitHandl) {
            noWaitHandl(hasResults, isAutoQuestion, query, results)
        }
        break;
    case"escalation":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events[type];
        if (!handl) {
            break
        }
        query = ISQ.Tools.Base64.decodeString(messageArgs.query);
        var escType = messageArgs.esctype;
        handl(escType, query);
        break;
    case"like":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events[type];
        if (!handl) {
            break
        }
        query = ISQ.Tools.Base64.decodeString(messageArgs.query);
        var answer = ISQ.Tools.Base64.decodeString(messageArgs.answer);
        handl(query, answer);
        break;
    case"useractive":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events.userActive;
        if (handl) {
            handl()
        }
        break;
    case"directionchanged":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events.directionChanged;
        if (handl) {
            handl(messageArgs.rtl === "true")
        }
        break;
    case"chatavailable":
        if (!widgetData.events) {
            break
        }
        handl = widgetData.events.chatAvailable;
        if (handl) {
            handl(messageArgs.value === "true")
        }
        break
    }
};
ISQ.Widget.buildAttachmentUrls_forEvent = function(j) {
    if (!j) {
        return null
    }
    var e = [];
    var g = j.split(",");
    for (var f = 0; f < g.length; ++f) {
        if (e.length != 0) {
            e.push(",")
        }
        ISQ.Widget.getAttachmentUrl(g[f], e)
    }
    return e.join("")
};
if (typeof(ISQ.Widget.state) === "undefined") {
    ISQ.Widget.API = {};
    ISQ.Widget.API.customParams = null;
    ISQ.Widget.API.getCurrentURL = function() {
        return window.location.href
    };
    ISQ.Widget.API.getCustomParams = function() {
        return _nRepData.customParams || {}
    };
    ISQ.Widget.API.openContactForm = function() {
        ISQ.API.openContactForm()
    };
    ISQ.Widget.API.updateContext = function(d, c) {};
    ISQ.Widget.API.reAsk = function() {}
}
ISQ.Widget.Proxy = initializeNS("ISQ.Widget.Proxy");
ISQ.Widget.Proxy.Chat = {
    sendMessage: function(d, c) {
        _it(d, function(a) {
            a.cdcUser.sendMessage("subject=chat&" + c)
        })
    }
};
ISQ.Widget.Proxy.Chat.LiveChatInc = {
    enumStates: {
        none: 0,
        loaded: 1,
        availabilityReceived: 2
    },
    mState: 0,
    mAvailable: false,
    mPendingWidgets: [],
    start: function(d, e, f) {
        if (d == "loadChat") {
            this.load(e, f)
        } else {
            LC_API.open_chat_window()
        }
    },
    load: function(j, k) {
        if (this.mState >= this.enumStates.loaded) {
            if (this.mState >= this.enumStates.availabilityReceived) {
                if (k.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage([k], "type=available&data=" + this.mAvailable)
                } else {
                    if (this.mAvailable === "true") {
                        k.mState|=ISQ.Widget.iChat.enumState.available
                    } else {
                        k.mState&=~ISQ.Widget.iChat.enumState.available
                    }
                }
            } else {
                this.mPendingWidgets.push(k)
            }
            return 
        } else {
            this.mPendingWidgets.push(k)
        }
        this.mState = this.enumStates.loaded;
        var l = document.head.getElementsByTagName("script");
        for (var f = 0; f < l.length; f++) {
            if (l[f].src.indexOf("livechatinc")!=-1) {
                this.mAvailable = LC_API.agents_are_available();
                if (k.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage([k], "type=available&data=" + this.mAvailable)
                } else {
                    if (this.mAvailable === "true") {
                        k.mState|=ISQ.Widget.iChat.enumState.available
                    } else {
                        k.mState&=~ISQ.Widget.iChat.enumState.available
                    }
                }
                this.mState = this.enumStates.availabilityReceived;
                return 
            }
        }
        window.__lc = {};
        window.LC_API = {};
        window.__lc.license = j.LiveChatInc.id;
        window.LC_API.on_chat_ended = function() {
            window.LC_API.hide_chat_window()
        };
        if (!j.popup.enabled) {
            window.LC_API.on_chat_started = function(a) {
                window.LC_API.open_chat_window()
            }
        }
        var g = this;
        window.LC_API.on_after_load = function() {
            if (!window.LC_API.chat_running() || j.popup.enabled) {
                window.LC_API.hide_chat_window()
            }
            g.mAvailable = LC_API.agents_are_available();
            if (k.cdcUser) {
                ISQ.Widget.Proxy.Chat.sendMessage(g.mPendingWidgets, "type=available&data=" + g.mAvailable)
            } else {
                if (g.mAvailable == true) {
                    k.mState|=ISQ.Widget.iChat.enumState.available
                } else {
                    k.mState&=~ISQ.Widget.iChat.enumState.available
                }
            }
            g.mState = g.enumStates.availabilityReceived
        };
        url = (window.location.protocol === "https:" ? "https:" : "http:") + "//" + j.LiveChatInc.server + "/tracking.js";
        ISQ.Tools.appendScript(url)
    }
};
ISQ.Widget.Proxy.Chat.Zopim = {
    enumStates: {
        none: 0,
        loaded: 1,
        availabilityReceived: 2
    },
    mState: 0,
    mAvailable: false,
    mPendingWidgets: [],
    start: function(d, e, f) {
        if (d == "loadChat") {
            this.load(e, f)
        } else {
            if (e.popup.enabled) {
                $zopim.livechat.window.openPopout()
            } else {
                $zopim.livechat.window.show()
            }
        }
    },
    load: function(k, l) {
        if (this.mState >= this.enumStates.loaded) {
            if (this.mState >= this.enumStates.availabilityReceived) {
                if (l.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage([l], "type=available&data=" + this.mAvailable)
                } else {
                    if (this.mAvailable == true) {
                        l.mState|=ISQ.Widget.iChat.enumState.available
                    } else {
                        l.mState&=~ISQ.Widget.iChat.enumState.available
                    }
                }
            } else {
                this.mPendingWidgets.push(l)
            }
            return 
        } else {
            this.mPendingWidgets.push(l)
        }
        this.mState = this.enumStates.loaded;
        var j = this;
        var n = function() {
            if (!$zopim.livechat) {
                return 
            }
            if (!$zopim.livechat.isChatting() || k.popup.enabled) {
                $zopim.livechat.hideAll()
            }
            $zopim.livechat.setOnChatEnd(function() {
                $zopim.livechat.hideAll()
            });
            $zopim.livechat.window.onHide(function() {
                if (!$zopim.livechat.isChatting() || k.popup.enabled) {
                    $zopim.livechat.hideAll()
                }
            });
            var a = function(b) {
                if (b == "online") {
                    j.mAvailable = true
                } else {
                    j.mAvailable = false
                }
                if (l.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage(j.mPendingWidgets, "type=available&data=" + j.mAvailable)
                } else {
                    if (j.mAvailable == true) {
                        l.mState|=ISQ.Widget.iChat.enumState.available
                    } else {
                        l.mState&=~ISQ.Widget.iChat.enumState.available
                    }
                }
                j.mState = j.enumStates.availabilityReceived
            };
            $zopim.livechat.setOnStatus(a)
        };
        var m = document.head.getElementsByTagName("script");
        for (var g = 0; g < m.length; g++) {
            if (m[g].src.indexOf("zopim")!=-1) {
                n();
                return 
            }
        }
        window.$zopim || (function(b, d) {
            var e = $zopim = function(f) {
                e._.push(f)
            }, a = e.s = b.createElement(d), c = b.getElementsByTagName(d)[0];
            e.set = function(f) {
                e.set._.push(f)
            };
            e._ = [];
            e.set._ = [];
            a.async=!0;
            a.setAttribute("charset", "utf-8");
            a.src = (window.location.protocol === "https:" ? "https:" : "http:") + k.Zopim.server + "?" + k.Zopim.id;
            e.t =+ new Date;
            a.type = "text/javascript";
            if (n) {
                if (ISQ.Http.browser.app !== "ie" || ISQ.Http.browser.isIE11) {
                    a.onload = n
                } else {
                    a.onreadystatechange = function() {
                        if (this.readyState !== "complete" && this.readyState !== "loaded") {
                            return 
                        }
                        n()
                    }
                }
            }
            c.parentNode.insertBefore(a, c)
        })(document, "script")
    }
};
ISQ.Widget.Proxy.Chat.SnapEngage = {
    enumStates: {
        none: 0,
        loaded: 1,
        availabilityReceived: 2
    },
    mState: 0,
    mAvailable: false,
    mPendingWidgets: [],
    start: function(d, e, f) {
        if (d == "loadChat") {
            this.load(e, f)
        } else {
            SnapEngage.startLink()
        }
    },
    load: function(l, o) {
        if (this.mState >= this.enumStates.loaded) {
            if (this.mState >= this.enumStates.availabilityReceived) {
                if (o.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage(k.mPendingWidgets, "type=available&data=" + k.mAvailable)
                }
            } else {
                this.mPendingWidgets.push(o)
            }
            return 
        } else {
            this.mPendingWidgets.push(o)
        }
        this.mState = this.enumStates.loaded;
        var k = this;
        var m = function() {
            if (typeof SnapEngage == "undefined") {
                return 
            }
            SnapEngage.hideButton();
            SnapEngage.getAgentStatusAsync(function(a) {
                if (a) {
                    k.mAvailable = true
                } else {
                    k.mAvailable = false
                }
                if (o.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage(k.mPendingWidgets, "type=available&data=" + k.mAvailable)
                }
                k.mState = k.enumStates.availabilityReceived
            });
            SnapEngage.setCallback("Close", function(b, a) {
                SnapEngage.hideButton()
            })
        };
        var p = document.head.getElementsByTagName("script");
        for (var j = 0; j < p.length; j++) {
            if (p[j].src.indexOf("snapengage")!=-1) {
                m();
                return 
            }
        }
        var n = (window.location.protocol === "https:" ? "https:" : "http:") + l.SnapEngage.server + l.SnapEngage.id + ".js";
        ISQ.Tools.appendScript(n, m)
    }
};
ISQ.Widget.Proxy.Chat.Olark = {
    enumStates: {
        none: 0,
        loaded: 1,
        availabilityReceived: 2
    },
    mState: 0,
    mAvailable: false,
    mPendingWidgets: [],
    start: function(d, e, f) {
        if (d == "loadChat") {
            this.load(e, f)
        } else {
            olark("api.box.expand")
        }
    },
    load: function(l, m) {
        if (this.mState >= this.enumStates.loaded) {
            if (this.mState >= this.enumStates.availabilityReceived) {
                if (m.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage([m], "type=available&data=" + this.mAvailable)
                }
            } else {
                this.mPendingWidgets.push(m)
            }
            return 
        } else {
            this.mPendingWidgets.push(m)
        }
        this.mState = this.enumStates.loaded;
        var j = this;
        var k = function() {
            var a = function(b) {
                j.mAvailable = b;
                if (m.cdcUser) {
                    ISQ.Widget.Proxy.Chat.sendMessage(j.mPendingWidgets, "type=available&data=" + j.mAvailable)
                }
                j.mState = j.enumStates.availabilityReceived
            };
            olark.configure("box.start_hidden", true);
            olark("api.chat.onOperatorsAvailable", function() {
                a(true)
            });
            olark("api.chat.onOperatorsAway", function() {
                a(false)
            })
        };
        var n = document.head.getElementsByTagName("script");
        for (var g = 0; g < n.length; g++) {
            if (n[g].src.indexOf("olark")!=-1) {
                k();
                return 
            }
        }
        window.olark || (function(a) {
            var c = window, b = document, d = c.location.protocol == "https:" ? "https:": "http:", o = a.name, f = "load";
            var e = function() {
                c[o] = function() {
                    (q.s = q.s || []).push(arguments)
                };
                var q = c[o]._ = {}, t = a.methods.length;
                while (t--) {
                    (function(p) {
                        c[o][p] = function() {
                            c[o]("call", p, arguments)
                        }
                    })(a.methods[t])
                }
                q.l = a.loader;
                q.i = e;
                q.p = {
                    0: + new Date
                };
                q.P = function(p) {
                    q.p[p] = new Date - q.p[0]
                };
                function u() {
                    q.P(f);
                    c[o](f)
                }
                c.addEventListener ? c.addEventListener(f, u, false) : c.attachEvent("on" + f, u);
                var s = function() {
                    function K(y) {
                        y = "head";
                        return ["<", y, "></", y, "><", T, ' onload="var d=', R, ";d.getElementsByTagName('head')[0].", p, "(d.", S, "('script')).", r, "='", d, "//", q.l, "'", '"', "></", T, ">"].join("")
                    }
                    var T = "body", v = b[T];
                    if (!v) {
                        return setTimeout(s, 100)
                    }
                    q.P(1);
                    var p = "appendChild", S = "createElement", r = "src", w = b[S]("div"), M = w[p](b[S](o)), P = b[S]("iframe"), R = "document", Q = "domain", x;
                    w.style.display = "none";
                    v.insertBefore(w, v.firstChild).id = o;
                    P.frameBorder = "0";
                    P.id = o + "-loader";
                    if (/MSIE[ ]+6/.test(navigator.userAgent)) {
                        P.src = "javascript:false"
                    }
                    P.allowTransparency = "true";
                    M[p](P);
                    try {
                        P.contentWindow[R].open()
                    } catch (N) {
                        a[Q] = b[Q];
                        x = "javascript:var d=" + R + ".open();d.domain='" + b.domain + "';";
                        P[r] = x + "void(0);"
                    }
                    try {
                        var L = P.contentWindow[R];
                        L.write(K());
                        L.close()
                    } catch (O) {
                        P[r] = x + 'd.write("' + K().replace(/"/g, String.fromCharCode(92) + '"') + '");d.close();'
                    }
                    q.P(2)
                };
                s()
            };
            e()
        })({
            loader: "static.olark.com/jsclient/loader0.js",
            name: "olark",
            methods: ["configure", "extend", "declare", "identify"]
        });
        olark.identify(l.Olark.id);
        k()
    }
};
ISQ.CDC = initializeNS("ISQ.CDC");
if (typeof(nanoRep) === "undefined") {
    window.nanoRep = {}
}
if (typeof(nanoRep.CDC) === "undefined") {
    nanoRep.CDC = {}
}
nanoRep.CDC.Helper = {
    splitURLArguments: function(e) {
        var g = new Array();
        var t = e;
        var c = t.length;
        if (c === 0) {
            return g
        }
        var b = function(j) {
            return unescape(j.join("").toString()).replace(/^\s+|\s+$/g, "")
        };
        var s = 1;
        var d = "";
        var f = 0;
        var a = new Array();
        while (f < c) {
            switch (s) {
            case 1:
                if (t.charAt(f) !== "=") {
                    a.push(t.charAt(f))
                } else {
                    s = 2;
                    d = b(a).toLowerCase();
                    a = new Array()
                }
                break;
            case 2:
                if (t.charAt(f) !== "&") {
                    a.push(t.charAt(f))
                } else {
                    s = 1;
                    g[d] = b(a);
                    a = new Array()
                }
                break
            }
            ++f
        }
        if (s === 2) {
            g[d] = b(a)
        }
        return g
    },
    getTargetWindow: function(a, c) {
        if (!a) {
            return window.top
        }
        var g = window;
        if (c) {
            g = window.parent
        }
        if (a === "parent") {
            return g.parent
        }
        var f = false;
        var d;
        do {
            try {
                if (g === window.top) {
                    f = true
                }
                d = g.frames[a];
                if (d) {
                    return d
                }
            } catch (b) {}
            g = g.parent
        }
        while (!f);
        return window.top
    },
    parseMessage: function(a) {
        if (!a) {
            return ""
        }
        var b = nanoRep.CDC.Helper.splitURLArguments(a);
        b.s = unescape(b.s);
        b.t = unescape(b.t);
        return b
    }
};
ISQ.CDC.User = ISQ.Base.extend({
    name: "ISQ.CDC.User"
});
ISQ.CDC.User.prototype.ctor = function(d) {
    this.mTargetWindowName = d.targetWindowName || "";
    this.mProxyIFrameUrl = d.proxyIFrameUrl || null;
    this.mAllowMessagesFrom = d.allowMessagesFrom || null;
    this.incomingMessageEvent = new ISQ.Event();
    this.mWidget = d.widget || null;
    this.eFilterMessages = new ISQ.Event();
    if (typeof(ISQ.CDC.users) === "undefined") {
        ISQ.CDC.users = new Array()
    }
    ISQ.CDC.users.push(this);
    if (typeof(window.postMessage) != "function" && typeof(window.postMessage) != "object") {
        this.mLegacyMode = true;
        this.mHost_l = window.location.protocol + "//" + window.location.host;
        if (this.mProxyIFrameUrl) {
            if (this.mProxyIFrameUrl.contains("?")) {
                this.mProxyIFrameUrl += "&srv=" + d.server
            } else {
                this.mProxyIFrameUrl += "?srv=" + d.server
            }
        }
    } else {
        var c = this;
        this.mPostMessageEventListener = function(b) {
            if (b.origin !== c.mAllowMessagesFrom) {
                return 
            }
            var a = ISQ.CDC.Helper.splitURLArguments(b.data);
            a.s = unescape(b.origin);
            a.t = "*";
            c.dispatchMessage(a)
        };
        if (window.addEventListener) {
            window.addEventListener("message", this.mPostMessageEventListener, false)
        } else {
            if (window.attachEvent) {
                window.attachEvent("onmessage", this.mPostMessageEventListener)
            }
        }
    }
};
ISQ.CDC.User.prototype.dtor = function() {
    if (this.mProxyIFrame_l && this.mProxyIFrame_l.parentNode !== null) {
        document.body.removeChild(this.mProxyIFrame_l)
    }
    if (!this.mLegacyMode) {
        if (window.removeEventListener) {
            window.removeEventListener("message", this.mPostMessageEventListener, false)
        } else {
            if (window.detachEvent) {
                window.detachEvent("onmessage", this.mPostMessageEventListener)
            }
        }
    }
};
ISQ.CDC.User.prototype.sendMessage = function(j, m) {
    m = m || "*";
    if (!this.mLegacyMode) {
        var n = ISQ.CDC.Helper.getTargetWindow(this.mTargetWindowName);
        j += "&ord=" + (++this.mMessageOrdinal);
        n.postMessage(j, m)
    } else {
        if (!this.mMessagesOnSent) {
            this.mMessagesOnSent = new Array()
        }
        if (!this.mProxyIFrameBaseUrl_l) {
            var g = new Array();
            g.push(this.mProxyIFrameUrl);
            g.push("#");
            g.push("s=");
            g.push(escape(this.mHost_l));
            g.push("&t=");
            g.push(escape(m));
            g.push("&twnd=");
            g.push(this.mTargetWindowName);
            this.mProxyIFrameBaseUrl_l = g.join("")
        }
        var k = function(a, f, b, q) {
            var r = new Array();
            r.push(a.mProxyIFrameBaseUrl_l);
            r.push("&");
            r.push(b);
            r.push("&ord=");
            r.push(f);
            var p = "position: absolute; left: 0px; top:-1100px;height:70px;width:70px;";
            var c = document.createElement("iframe");
            var d = ISQ.CDC.User.getIframeId();
            c.setAttribute("id", d);
            c.setAttribute("name", d);
            c.setAttribute("src", r.join(""));
            c.setAttribute("frameBorder", "1");
            c.setAttribute("scrolling", "auto");
            c.setAttribute("width", "70px");
            c.setAttribute("height", "70px");
            c.setAttribute("style", p);
            if (c.style.setAttribute) {
                c.style.setAttribute("cssText", p)
            }
            document.body.appendChild(c);
            var e = function() {
                setTimeout(function() {
                    a.mMessagesOnSent[f] = null
                }, 100)
            };
            if (ISQ.Http.browser.app === "opera") {
                c.onload = e
            } else {
                if (document.addEventListener) {
                    c.addEventListener("load", e, true)
                } else {
                    if (document.attachEvent) {
                        c.attachEvent("onload", e)
                    }
                }
            }
        };
        var l=++this.mMessageOrdinal;
        this.mMessagesOnSent[l] = new k(this, l, j, m)
    }
};
ISQ.CDC.User.prototype.dispatchMessage = function(m) {
    if (m.widget && this.mWidget !== m.widget) {
        return 
    }
    var p = {
        message: m,
        filter: false
    };
    this.eFilterMessages.dispatch(p);
    if (p.filter) {
        return 
    }
    if (m.subject && m.subject == "reset") {
        this.mLastHandledMessageOrdinal = 0;
        this.mMessageOrdinal = 0;
        this.mReceivedMessages = null;
        this.mWasReset_debug = true
    }
    var q = parseInt(m.ord);
    var e = q - this.mLastHandledMessageOrdinal;
    if (e <= 0) {
        return 
    }
    if (e > 1) {
        if (this.mReceivedMessages === null) {
            this.mReceivedMessages = new Array();
            this.mReceivedMessagesCount = 0
        }
        this.mReceivedMessages[q] = m;
        ++this.mReceivedMessagesCount
    } else {
        try {
            this.incomingMessageEvent.dispatch(m)
        } catch (k) {}
        if (this.mReceivedMessages !== null) {
            var o = this.mReceivedMessagesCount;
            var n = q + 1;
            for (var l = 0; l < o; ++l, ++n) {
                if (!this.mReceivedMessages[n]) {
                    break
                }
                this.incomingMessageEvent.dispatch(this.mReceivedMessages[n]);
                q = n;
                --this.mReceivedMessagesCount
            }
            if (this.mReceivedMessagesCount === 0) {
                this.mReceivedMessages = null
            }
        }
        this.mLastHandledMessageOrdinal = q
    }
};
ISQ.CDC.User.prototype.targetWindowName = function() {
    return this.mTargetWindowName
};
ISQ.CDC.User.prototype.allowMessagesFrom = function() {
    return this.mAllowMessagesFrom
};
ISQ.CDC.User.prototype.mTargetWindowName = null;
ISQ.CDC.User.prototype.mLegacyMode = false;
ISQ.CDC.User.prototype.mHost_l = "";
ISQ.CDC.User.prototype.mProxyIFrameUrl = null;
ISQ.CDC.User.prototype.mProxyIFrameBaseUrl_l = null;
ISQ.CDC.User.prototype.mCurrentHeight = 70;
ISQ.CDC.User.prototype.incomingMessageEvent = null;
ISQ.CDC.User.prototype.eFilterMessages = null;
ISQ.CDC.User.prototype.mPostMessageEventListener = null;
ISQ.CDC.User.prototype.mIsCDCStation = true;
ISQ.CDC.User.prototype.mMessageOrdinal = 0;
ISQ.CDC.User.prototype.mAllowMessagesFrom = null;
ISQ.CDC.User.prototype.mMessagesOnSent = null;
ISQ.CDC.User.prototype.mReceivedMessages = null;
ISQ.CDC.User.prototype.mLastHandledMessageOrdinal = 0;
ISQ.CDC.User.prototype.mReceivedMessagesCount = 0;
ISQ.CDC.User.prototype.mWasReset_debug = false;
ISQ.CDC.User.getIframeId = function() {
    return "nR_CDC" + ISQ.CDC.User.iframeIdCounter++
};
ISQ.CDC.User.iframeIdCounter = new Date().getTime();
nanoRep.FAQ = initializeNS("nanoRep.FAQ");
nanoRep.FAQ.Texts = {
    en: {
        like: "Good answer!",
        likeActive: "You like this",
        dislike: "Needs improvement",
        dislikeActive: "You dislike this",
        ok: "OK",
        cancel: "Cancel",
        dislikeOptionsTitle: "What's wrong with this answer?",
        likeTexts: {
            0: "Like",
            2: "Answer is incorrect or inaccurate",
            3: "Missing important information",
            4: "Answer irrelevant to the page",
            5: "Didn't find what I was looking for",
            6: "Missing or incorrect information"
        }
    },
    pl: {
        like: "Like",
        likeActive: "Chcesz tego",
        dislike: "Nie lubić",
        dislikeActive: "Nie lubić",
        ok: "OK",
        cancel: "Anuluj",
        dislikeOptionsTitle: "Jaki jest problem z odpowiedzią?",
        likeTexts: {
            0: "Like",
            2: "Odpowiedź jest niepoprawna bądź nieprecyzyjna",
            3: "Brakuje ważnych informacji",
            4: "Odpowiedź jest niezwiązana z tematem",
            5: "Nie mogę znaleźć to, czego szukasz",
            6: "Brak lub nieprawidłowe informacje"
        }
    },
    da: {
        like: "Like",
        likeActive: "Du kan lide denne",
        dislike: "Dislike",
        dislikeActive: "Du kan ikke lide denne",
        ok: "OK",
        cancel: "Annuller",
        dislikeOptionsTitle: "Hvad er der galt med dette svar?",
        likeTexts: {
            0: "Like",
            2: "Svaret er forkert eller upræcist",
            3: "Mangler vigtig information",
            4: "Svaret er irrelevant",
            5: "Jeg kunne ikke finde, hvad jeg ledte efter",
            6: "Manglende eller forkerte oplysninger"
        }
    },
    hu: {
        like: "Like",
        likeActive: "You like this",
        dislike: "Dislike",
        dislikeActive: "You dislike this",
        ok: "Rendben",
        cancel: "Mégse",
        dislikeOptionsTitle: "Mi a baj a válasszal?",
        likeTexts: {
            0: "Like",
            2: "A válasz helytelen vagy pontatlan",
            3: "Fontos információ hiányzik belőle",
            4: "A válasz nem releváns",
            5: "Én nem találtam, amit kerestem",
            6: "Hiányzó vagy helytelen információ"
        }
    },
    tr: {
        like: "Like",
        likeActive: "You like this",
        dislike: "Dislike",
        dislikeActive: "You dislike this",
        ok: "OK",
        cancel: "Cancel",
        dislikeOptionsTitle: "What's wrong with this answer?",
        likeTexts: {
            0: "Like",
            2: "Answer is incorrect or inaccurate",
            3: "Missing important information",
            4: "Answer irrelevant to the page",
            5: "Didn't find what I was looking for",
            6: "Missing or incorrect information"
        }
    },
    sv: {
        like: "Liksom",
        likeActive: "Du gillar det här inlägget",
        dislike: "Motvilja",
        dislikeActive: "Du gillar inte det här",
        ok: "OK",
        cancel: "Avbryt",
        dislikeOptionsTitle: "Vad är det för fel på detta svar?",
        likeTexts: {
            0: "Liksom",
            2: "Svaret är felaktigt eller oriktigt",
            3: "Saknar viktig information",
            4: "Svaret är irrelevant",
            5: "Jag hittade inte det jag letade efter",
            6: "Saknas eller felaktig information"
        }
    },
    de: {
        like: "Gefällt mir",
        likeActive: "Sie mögen dieses",
        dislike: "Dislike",
        dislikeActive: "Sie mögen diese",
        ok: "OK",
        cancel: "Abbrechen",
        dislikeOptionsTitle: "Was ist falsch mit dieser Antwort?",
        likeTexts: {
            0: "Gefällt mir",
            2: "Antwort ist nicht richtig oder ungenau",
            3: "Es fehlen wichtige Informationen",
            4: "Die Antwort ist irrelevant",
            5: "Ich fand nicht, was ich gesucht habe",
            6: "Fehlende oder falsche Informationen"
        }
    },
    ar: {
        like: "مثل",
        likeActive: "كنت مثل هذا",
        dislike: "كره",
        dislikeActive: "كنت تكره هذا",
        ok: "موافق",
        cancel: "إلغاء",
        dislikeOptionsTitle: "ما الخطأ في هذه الإجابة؟",
        likeTexts: {
            0: "مثل",
            2: "الإجابة غير صحيحة أو غير دقيقة",
            3: "معلومات هامة ناقصة",
            4: "الإجابة غير ذات صلة",
            5: "لم أجد ما كنت أبحث عنه",
            6: "المعلومات الناقصة أو غير صحيحة"
        }
    },
    zh: {
        like: "Like",
        likeActive: "你喜欢这个",
        dislike: "反感",
        dislikeActive: "你不喜欢",
        ok: "确定",
        cancel: "取消",
        dislikeOptionsTitle: "这个答案有什么问题？",
        likeTexts: {
            0: "Like",
            2: "答案错误或不准确",
            3: "遗漏了重要信息",
            4: "答案不切题",
            5: "我没有找到我要找的",
            6: "缺少或不正确的信息"
        }
    },
    ru: {
        like: "Like",
        likeActive: "Вам понравилось это",
        dislike: "не любить",
        dislikeActive: "Вы не понравилось",
        ok: "OK",
        cancel: "Отмена",
        dislikeOptionsTitle: "В чем проблема с этим ответом?",
        likeTexts: {
            0: "Like",
            2: "Ответ неверный или неточный",
            3: "В нем отсутствует важная информация",
            4: "Ответ не относится к делу",
            5: "Мне не удалось найти то, что я искал",
            6: "Отсутствие или неправильную информацию"
        }
    },
    ja: {
        like: "Like",
        likeActive: "あなたはこれが好き",
        dislike: "嫌い",
        dislikeActive: "あなたはこれを嫌い",
        ok: "OK",
        cancel: "キャンセル",
        dislikeOptionsTitle: "この回答の問題点は?",
        likeTexts: {
            0: "Like",
            2: "回答が不正または不精確",
            3: "重要情報が欠けている",
            4: "回答内容が無関係",
            5: "私は私が探していたものを見つけることができませんでした",
            6: "情報の修正、不足はあります"
        }
    },
    ko: {
        like: "Like",
        likeActive: "당신이 좋아",
        dislike: "싫어",
        dislikeActive: "당신이 싫어",
        ok: "확인",
        cancel: "취소",
        dislikeOptionsTitle: "이 답의 무엇이 잘못되었나요?",
        likeTexts: {
            0: "Like",
            2: "답이 잘못되었거나 정확하지 않습니다.",
            3: "중요한 정보가 빠졌습니다.",
            4: "답이 적절하지 않습니다.",
            5: "나는 내가 무엇을 찾고 찾을 수 없습니다",
            6: "누락 또는 잘못된 정보"
        }
    },
    fr: {
        like: "J'aime",
        likeActive: "Vous aimez ce",
        dislike: "Antipathie",
        dislikeActive: "Yous n'aime pas ce",
        ok: "OK",
        cancel: "annuler",
        dislikeOptionsTitle: "Pourquoi cette réponse est-elle erronée ?",
        likeTexts: {
            0: "J'aime",
            2: "La réponse est inexacte ou imprécise",
            3: "Des informations importantes sont manquantes",
            4: "La réponse est hors sujet",
            5: "Je n'ai pas trouvé ce que je cherchais",
            6: "Information manquante ou incorrecte"
        }
    },
    it: {
        like: "Mi piace",
        likeActive: "Ti piace questo",
        dislike: "Antipatia",
        dislikeActive: "Non ti piace questo",
        ok: "OK",
        cancel: "Annulla",
        dislikeOptionsTitle: "Cosa c'è di sbagliato in questa risposta?",
        likeTexts: {
            0: "Mi piace",
            2: "La risposta è errata o inaccurata",
            3: "Mancano informazioni importanti",
            4: "La risposta è irrilevante",
            5: "Non ho trovato quello che cercavo",
            6: "Informazioni mancanti o errate"
        }
    },
    nl: {
        like: "Vind ik leuk",
        likeActive: "U houdt van dit",
        dislike: "Afkeer",
        dislikeActive: "Je vind dit niet leuk",
        ok: "OK",
        cancel: "Annuleren",
        dislikeOptionsTitle: "Wat klopt er niet aan dit antwoord?",
        likeTexts: {
            0: "Vind ik leuk",
            2: "Antwoord is onjuist of onnauwkeurig",
            3: "Belangrijke informatie ontbreekt",
            4: "Het antwoord doet niet ter zake",
            5: "Heeft u niet gevonden wat ik zocht",
            6: "Ontbrekende of onjuiste informatie"
        }
    },
    pt: {
        like: "Gosto",
        likeActive: "Você gosta disso",
        dislike: "Antipatia",
        dislikeActive: "Você não gosta deste",
        ok: "OK",
        cancel: "Cancelar",
        dislikeOptionsTitle: "O que há de errado com esta resposta?",
        likeTexts: {
            0: "Gosto",
            2: "A resposta é incorreta ou inexata",
            3: "Faltam informações importantes",
            4: "A resposta é irrelevante",
            5: "Não encontrou o que estava procurando",
            6: "Informações ausentes ou incorretas"
        }
    },
    es: {
        like: "Me gusta",
        likeActive: "Si le gusta este",
        dislike: "No me gusta",
        dislikeActive: "No le gusta este",
        ok: "Aceptar",
        cancel: "Cancelar",
        dislikeOptionsTitle: "¿Cuál es el problema con esta respuesta?",
        likeTexts: {
            0: "Me gusta",
            2: "La respuesta es incorrecta o inexacta",
            3: "Falta información importante",
            4: "La respuesta es irrelevante",
            5: "No encontré lo que buscaba",
            6: "Información faltante o incorrecta"
        }
    },
    iw: {
        like: "אהבתי",
        likeActive: "אהבת את זה",
        dislike: "לא אהבתי",
        dislikeActive: "לא אהבת את זה",
        ok: "אישור",
        cancel: "ביטול",
        dislikeOptionsTitle: "מה לא תקין בתשובה זו?",
        likeTexts: {
            0: "אהבתי",
            2: "תשובה לא נכונה / לא מדוייקת",
            3: "חסר מידע חשוב",
            4: "התשובה אינה רלוונטית לעמוד",
            5: "לא מצאתי את מה שחיפשתי",
            6: "מידע חסר או לא מדוייק"
        }
    }
};
nanoRep.FAQ.Localization = {};
nanoRep.FAQ.init = function(c, d) {
    nanoRep.FAQ.Localization = nanoRep.FAQ.Texts[c] || nanoRep.FAQ.Texts.en;
    nanoRep.FAQ.serverUrl = d
};
nanoRep.FAQ.serverUrl = null;
nanoRep.FAQ.Likes = {};
nanoRep.FAQ.Likes.enumFeedback = {
    positive: 1,
    wrongAnswer: 2,
    wrongContent: 4,
    wrongContext: 8
};
nanoRep.FAQ.Likes.mData = {
    div: null,
    options: [],
    pastActions: null,
    hasContext: false
};
nanoRep.FAQ.Likes.buildThumbUpIcon = function(s, D, y, G, t, r, w, E, B, F, C) {
    B = B || nanoRep.FAQ.Localization;
    var H = createDiv(null, ".nanoRepMP_likeIcon");
    H.onclick = function() {
        nanoRep.FAQ.Likes.saveData(H, s, G, t, r, w, E, F);
        nanoRep.FAQ.Likes.sendThumbAction(nanoRep.FAQ.Likes.enumFeedback.positive, B)
    };
    var x = nanoRep.FAQ.Likes.getLastAction(s);
    nanoRep.FAQ.Likes.setThumbStyles(x.action, H, null, false, B);
    if (x.action == 0 && x.cacheVar == t) {
        y++
    }
    if (C) {
        D.appendChild(H);
        H.className += " nanoRepMP_likeIconStandAlone";
        H.innerHTML = y ? ISQ.Tools.getPrettyNumber(y, 0, false, true) : "0";
        H.likes = y;
        return H
    }
    var z = createDiv(D, "white-space:nowrap;");
    var A = createDiv(null, "white-space:nowrap;display:inline;");
    var v = (ISQ.Widget.HTML.dynamicRTLCssLoaded) ? "rtl": "ltr";
    if (v == "ltr") {
        z.appendChild(H);
        z.appendChild(A);
        createDiv(A, ".newLikesLeftEdge&display:inline-block;float:none;vertical-align:top;")
    }
    var u = createDiv(A, ".newlikes&display:inline-block;float:none;vertical-align:top");
    if (v == "rtl") {
        u.className += " newlikesRTL";
        createDiv(A, ".newLikesRightEdge&display:inline-block;float:none;vertical-align:top;");
        A.appendChild(u);
        z.appendChild(H);
        z.appendChild(A)
    }
    u.innerHTML = y ? ISQ.Tools.getPrettyNumber(y, 0, false, true) : "0";
    H.likes = y;
    H.countElement = u;
    H.countContainer = A;
    return H
};
nanoRep.FAQ.Likes.buildThumbDownIcon = function(o, u, r, x, p, n, q, v, s, w, t, z) {
    s = s || nanoRep.FAQ.Localization;
    var y = createDiv(u, ".nanoRepMP_likeIcon &.nanoRepMP_dislikeIcon");
    y.widgetAnswer = z || false;
    r.thumbDown = y;
    y.onclick = function() {
        nanoRep.FAQ.Likes.saveData(r, o, x, p, n, q, v, w);
        var a = nanoRep.FAQ.Likes.getLastAction(o);
        if (a.action > nanoRep.FAQ.Likes.enumFeedback.positive) {
            nanoRep.FAQ.Likes.sendThumbAction(a.action, s)
        } else {
            nanoRep.FAQ.Likes.toggleDislikeOptions(false, s)
        }
    };
    nanoRep.FAQ.Likes.setThumbStyles(nanoRep.FAQ.Likes.getLastAction(o).action, null, y, false, s);
    return y
};
nanoRep.FAQ.Likes.getLastAction = function(articleId) {
    var pastActions = nanoRep.FAQ.Likes.mData.pastActions;
    var lastAction = {
        action: null,
        cacheVar: null
    };
    if (!pastActions) {
        var currentVal = ISQ.Tools.Storage.get("faqActions");
        if (currentVal) {
            pastActions = eval("(" + currentVal + ")")
        }
    }
    if (!pastActions) {
        pastActions = {}
    }
    nanoRep.FAQ.Likes.mData.pastActions = pastActions;
    if (pastActions[articleId] && typeof(pastActions[articleId].action) !== "undefined") {
        lastAction = pastActions[articleId]
    }
    return lastAction
};
nanoRep.FAQ.Likes.saveData = function(t, l, s, m, k, o, q, r, n, p) {
    nanoRep.FAQ.Likes.mData.thumbUp = t;
    nanoRep.FAQ.Likes.mData.thumbDown = t.thumbDown || null;
    nanoRep.FAQ.Likes.mData.articleId = l;
    nanoRep.FAQ.Likes.mData.account = k;
    nanoRep.FAQ.Likes.mData.rtl = s || false;
    nanoRep.FAQ.Likes.mData.cacheVar = m;
    nanoRep.FAQ.Likes.mData.account = k;
    nanoRep.FAQ.Likes.mData.kb = o;
    nanoRep.FAQ.Likes.mData.preview = q;
    nanoRep.FAQ.Likes.mData.referrer = r;
    nanoRep.FAQ.Likes.mData.keywordsetId = p || null;
    nanoRep.FAQ.Likes.mData.callback = n || null
};
nanoRep.FAQ.Likes.setThumbStyles = function(f, k, j, l, g) {
    g = g || nanoRep.FAQ.Localization;
    k = k || nanoRep.FAQ.Likes.mData.thumbUp;
    j = j || nanoRep.FAQ.Likes.mData.thumbDown;
    if (k) {
        if (f == nanoRep.FAQ.Likes.enumFeedback.positive) {
            ISQ.CSS.addClass(k, "nanoRepMP_likeIconCurrent");
            k.title = g.likeActive
        } else {
            ISQ.CSS.removeClass(k, "nanoRepMP_likeIconCurrent");
            k.title = g.like
        }
    }
    if (j) {
        if (f > nanoRep.FAQ.Likes.enumFeedback.positive) {
            ISQ.CSS.addClass(j, "nanoRepMP_dislikeIconCurrent");
            j.title = g.dislikeActive
        } else {
            ISQ.CSS.removeClass(j, "nanoRepMP_dislikeIconCurrent");
            j.title = g.dislike
        }
    }
    if (l) {
        nanoRep.FAQ.Likes.updateLikeCount(k, f == nanoRep.FAQ.Likes.enumFeedback.positive)
    }
};
nanoRep.FAQ.Likes.updateLikeCount = function(d, c) {
    d.likes = c ? d.likes + 1 : d.likes - 1;
    if (d.countElement) {
        d.countElement.innerHTML = d.likes > 0 ? ISQ.Tools.getPrettyNumber(d.likes, 0, false, true) : "0"
    } else {
        d.innerHTML = d.likes > 0 ? ISQ.Tools.getPrettyNumber(d.likes, 0, false, true) : "0"
    }
};
nanoRep.FAQ.Likes.toggleDislikeOptions = function(j, l) {
    l = l || nanoRep.FAQ.Localization;
    if (typeof(nanoRep.FAQ.Likes.mData) == "undefined") {
        return 
    }
    var f = nanoRep.FAQ.Likes.mData.thumbDown;
    if (!f) {
        return 
    }
    if (!nanoRep.FAQ.Likes.mData.div) {
        nanoRep.FAQ.Likes.buildDislikeOptions(f)
    }
    var g = nanoRep.FAQ.Likes.mData.div;
    if (g.style.display == "block" || j) {
        g.style.display = "none";
        f.title = f.oldTitle;
        ISQ.CSS.removeClass(f, "nanoRepMP_dislikeIconOpen");
        if (g.faqFader) {
            g.faqFader.parentNode.removeChild(g.faqFader);
            g.faqFader = null
        }
        f.removeChild(g);
        for (var k in nanoRep.FAQ.Likes.mData.options) {
            nanoRep.FAQ.Likes.mData.options[k].checked = false
        }
        ISQ.Tools.addRemoveClickOutsideElementsListener("dislikeOptions", false);
        clearNode(nanoRep.FAQ.Likes.mData.div);
        nanoRep.FAQ.Likes.mData.div = null
    } else {
        f.appendChild(g);
        f.oldTitle = f.title;
        f.title = "";
        g.style.display = "block";
        ISQ.CSS.addClass(f, "nanoRepMP_dislikeIconOpen");
        nanoRep.FAQ.Likes.setDislikeOptionsPosition();
        ISQ.Tools.addRemoveClickOutsideElementsListener("dislikeOptions", true, [g], new ISQ.Handler(function(a) {
            if (ISQ.Html.getEventTarget(a) == nanoRep.FAQ.Likes.mData.thumbDown) {
                return 
            }
            nanoRep.FAQ.Likes.toggleDislikeOptions(true, l);
            if (nanoRep.FAQ.Likes.dislikeTimer) {
                clearInterval(nanoRep.FAQ.Likes.dislikeTimer)
            }
        }))
    }
    ISQ.Tools.blinkElement(g, 0, false, "validationErr")
};
nanoRep.FAQ.Likes.dislikeTimer = null;
nanoRep.FAQ.Likes.setDislikeOptionsPosition = function() {
    var k = nanoRep.FAQ.Likes.mData.div;
    if (!k.parentNode) {
        return 
    }
    k.style.height = "auto";
    if (ISQ.Widget.HTML && ISQ.Widget.HTML.contentWrapper.offsetWidth < 420) {
        k.className += " narrowWidget_dislikeOptions"
    }
    var m = "0";
    var n = ISQ.Html.pageSize(true);
    if (n.h > 400) {
        if (k.parentNode.offsetTop + 150 > n.h - 30) {
            m = "-130"
        } else {
            m = "0"
        }
    }
    k.style.bottom = "auto";
    if (ISQ.Widget.HTML && ISQ.Widget.HTML.contentWrapper) {
        var l = createDiv(ISQ.Widget.HTML.widgetContainer, ".widgetFAQBlocker");
        k.faqFader = l;
        k.faqFader.style.position = "fixed";
        k.faqFader.onclick = function() {};
        var o = (ISQ.Widget.HTML.contentWrapper.offsetWidth - k.offsetWidth) / 2;
        k.style.left = o + "px";
        var p = ISQ.Widget.HTML.widgetContainer.offsetHeight;
        if (ISQ.Widget.isFloat) {
            var m = (ISQ.Widget.HTML.contentWrapper.offsetHeight - k.offsetHeight) / 2;
            m += ISQ.Widget.HTML.searchWrapper.offsetHeight + ISQ.Widget.HTML.titleWrapper.offsetHeight;
            k.style.top = m + "px";
            if (ISQ.Http.browser.isMobile) {
                k.style.top = (screen.height - k.offsetHeight) / 2 + "px"
            }
        } else {
            var j = k.parentNode.getBoundingClientRect();
            k.style.top = j.top - (k.offsetHeight) / 2 + "px";
            if (j.top + k.offsetHeight > p) {
                k.style.top = p - k.offsetHeight - ISQ.Widget.HTML.footerWrapper.offsetHeight + "px"
            }
        }
        if (ISQ.Http.browser.isMobile) {
            nanoRep.FAQ.Likes.dislikeTimer = setInterval(function() {
                var a = (ISQ.Widget.HTML.contentWrapper.offsetWidth - k.offsetWidth) / 2;
                k.style.left = a + "px"
            }, 250)
        }
    } else {
        if (nanoRep.FAQ.Likes.mData.rtl) {
            k.style.left = "20px"
        } else {
            k.style.right = "20px"
        }
    }
};
nanoRep.FAQ.Likes.buildDislikeOptions = function(o, v) {
    v = v || nanoRep.FAQ.Localization;
    var w = ISQ.Http.browser.isMobile ? " &.mobile": "";
    var q = nanoRep.FAQ.Likes.mData.div = createDiv(document.body, ".nanoRepMP_dislikeOptions" + w);
    q.onclick = function(a) {
        ISQ.Html.getDomEvent(a).cancel()
    };
    createLink({
        parent: q,
        text: "x",
        style: ".closeIcon",
        onclick: function() {
            nanoRep.FAQ.Likes.toggleDislikeOptions(false, v)
        }
    });
    createDiv(q, ".nanoRepMP_dislikeOptionsTitle").innerHTML = v.dislikeOptionsTitle;
    var x = [];
    var r = nanoRep.FAQ.Likes.mData.hasContext || (typeof(_session) === "function" && _session().hasContext());
    if (o.widgetAnswer) {
        x = [6, 5];
        if (r) {
            x.push(4)
        }
    } else {
        if (r) {
            x = [6, 4]
        } else {
            x = [2, 3]
        }
    }
    var A = {
        0: 1,
        2: 4,
        3: 4,
        4: 8,
        5: 2,
        6: 4
    };
    var z = v.likeTexts;
    for (var s = 0; s < x.length; s++) {
        var t = x[s];
        var y = createRadio({
            type: "radio",
            name: "dislike",
            parent: q,
            text: z[t],
            value: A[t],
            holderStyle: "padding-top:2px;",
            onclick: function() {
                ISQ.Tools.blinkElement(q, 0, false, "validationErr")
            }
        });
        nanoRep.FAQ.Likes.mData.options.push(y)
    }
    var p = createDiv(q, ".nanoRepMP_dislikeOptionsBtns");
    var u = createLink({
        parent: p,
        text: v.ok,
        style: ".button",
        onclick: function() {
            var a =- 1;
            for (var b in nanoRep.FAQ.Likes.mData.options) {
                if (nanoRep.FAQ.Likes.mData.options[b].checked) {
                    a = parseInt(nanoRep.FAQ.Likes.mData.options[b].value)
                }
            }
            if (a==-1) {
                ISQ.Tools.blinkElement(q, 3, true, "validationErr");
                return 
            }
            if (nanoRep.FAQ.Likes.mData.callback) {
                nanoRep.FAQ.Likes.mData.callback(nanoRep.FAQ.Likes.mData.articleId, nanoRep.FAQ.Likes.mData.keywordsetId, a)
            } else {
                nanoRep.FAQ.Likes.sendThumbAction(a, v)
            }
            nanoRep.FAQ.Likes.toggleDislikeOptions(false, v)
        }
    });
    u = createLink({
        parent: p,
        text: v.cancel,
        style: ".button &.buttonLink",
        onclick: function() {
            nanoRep.FAQ.Likes.toggleDislikeOptions(false, v)
        }
    })
};
nanoRep.FAQ.Likes.sendThumbAction = function(n, s) {
    s = s || nanoRep.FAQ.Localization;
    var o = nanoRep.FAQ.Likes.mData.articleId;
    var t = nanoRep.FAQ.Likes.mData.pastActions;
    var p = nanoRep.FAQ.Likes.mData.cacheVar;
    var m = nanoRep.FAQ.Likes.mData.account;
    var r = nanoRep.FAQ.Likes.mData.kb;
    var u = nanoRep.FAQ.Likes.mData.preview;
    var v = nanoRep.FAQ.Likes.mData.referrer;
    if (!o) {
        throw ("articleId must be specified")
    }
    var w = null;
    if (typeof(t[o]) !== "undefined" && typeof(t[o].action) !== "undefined") {
        w = t[o].action;
        if (w == n) {
            n = null
        }
    }
    if (n == null) {
        delete t[o]
    } else {
        t[o] = {};
        t[o].action = n;
        if (p) {
            t[o].cacheVar = p
        }
    }
    if (!u) {
        ISQ.Tools.Storage.set("faqActions", ISQ.Data.jsonToString(t));
        nanoRep.FAQ.Likes.mData.pastActions = t;
        var x = new Array();
        x.push(nanoRep.FAQ.serverUrl);
        x.push("/~");
        x.push(m.toLowerCase());
        x.push("/common/API/mostPopular/action.gif?rnd=");
        x.push(Math.floor(Math.random() * 1000000));
        x.push("&account=");
        x.push(m);
        if (r) {
            x.push("&kb=" + r)
        }
        x.push("&id=");
        x.push(o);
        if (n != null) {
            x.push("&action=" + n)
        }
        if (w != null) {
            x.push("&undo=" + w)
        }
        if (v) {
            x.push("&referer=" + escape(v))
        }
        var q = new Image();
        q.src = x.join("");
        q.style.width = "1px";
        q.style.height = "1px";
        q.style.display = "none"
    }
    nanoRep.FAQ.Likes.setThumbStyles(n, null, null, n == nanoRep.FAQ.Likes.enumFeedback.positive || w == nanoRep.FAQ.Likes.enumFeedback.positive, s)
};
nanoRep.FAQ.combineAndFormatContext = function(k, n) {
    if (!k) {
        k = {}
    }
    var l;
    if (typeof(_nRepData) == "undefined" ||!_nRepData.customParams) {
        l = k
    }
    if (!n) {
        if (ISQ.Cnf_Float) {
            l = ISQ.Widget.combineContext(ISQ.Cnf_Float.contextAttributes, k, _nRepData.customParams)
        } else {
            if (ISQ.Cnf_Embed) {
                l = ISQ.Widget.combineContext(ISQ.Cnf_Embed.contextAttributes, k, _nRepData.customParams)
            }
        }
    }
    var o = false;
    if (l) {
        for (var j in l) {
            o = true;
            break
        }
    }
    if (o) {
        nanoRep.FAQ.Likes.mData.hasContext = true;
        var p = [{
            from: ",",
            to: " "
        }, {
            from: ":",
            to: " "
        }
        ];
        var m = [];
        for (var j in l) {
            if (!l[j] || typeof(l[j]) !== "string") {
                continue
            }
            if (m.length != 0) {
                m.push(",")
            }
            m.push(j.replacer(p));
            m.push(":");
            m.push(l[j].replacer(p))
        }
        return "&context=" + ISQ.Http.Cookies.encode64(m.join(""))
    }
    return ""
};
nanoRep.FAQ.requestFaqLists = function(e) {
    var f = new Array();
    var d = e.account || _nRepData.account;
    f.push(e.serverUrl);
    f.push("/~" + d.toLowerCase() + "/common/API/mostPopular/faqList.js");
    f.push("?account=");
    f.push(d);
    if (e.isFloat) {
        f.push("&isFloat=true")
    }
    if (e.kb) {
        f.push("&kb=" + e.kb)
    }
    f.push("&referer=");
    if (e.configId) {
        f.push("!" + e.configId)
    } else {
        if (window.location.href.contains("nanorep.com") && e.referer) {
            f.push(escape(e.referer))
        } else {
            f.push(escape(window.location.href))
        }
    }
    if (e.encoding) {
        f.push("&encoding=" + e.encoding)
    }
    f.push(nanoRep.FAQ.combineAndFormatContext(e.context, e.disableContextCombine));
    ISQ.Tools.appendScript(f, function() {
        if (!nanoRep.FAQ.faqData) {
            throw ("Invalid FAQ Data received")
        }
        if (e.callback) {
            e.callback(nanoRep.FAQ.faqData)
        } else {
            nanoRep.FAQ.faqData = ISQ.Data.jsonToString(nanoRep.FAQ.faqData);
            var a = ISQ.Http.Cookies.encode64(nanoRep.FAQ.faqData);
            e.cdcUser.sendMessage("subject=faqData&data=" + a)
        }
        delete nanoRep.FAQ.faqData
    })
};
if (typeof(window.ISQ) === "undefined") {
    window.ISQ = {}
}
window.ISQ.removeNanoRepLoading = function(e) {
    var l = document.getElementsByTagName("DIV");
    if (typeof(l) === "undefined" ||!l) {
        return 
    }
    var j = null;
    for (var o = 0; o < l.length; ++o) {
        if (l[o].className !== "nanorep_loadingData" && l[o].id !== "nanoRepProxyContainer") {
            continue
        }
        for (var k = 0; k < l[o].childNodes.length; ++k) {
            if (l[o].childNodes[k].style && (l[o].childNodes[k].getAttribute("nR_noRemove") === "true" || l[o].childNodes[k].getAttribute("nR_noRemove") === true || l[o].childNodes[k].className.indexOf("nR_noRemove")!==-1)) {
                continue
            }
            if (((navigator.appVersion.indexOf("MSIE 6")!==-1) || (navigator.appVersion.indexOf("MSIE 7")!==-1)) && document.readyState !== "complete" && window.location.protocol.startsWith("https")) {
                if (l[o].childNodes[k].style) {
                    l[o].childNodes[k].style.display = "none"
                } else {
                    if (!j) {
                        j = new Array()
                    }
                    j.push(l[o].childNodes[k])
                }
            } else {
                try {
                    l[o].removeChild(l[o].childNodes[k]);
                    --k
                } catch (m) {
                    if (l[o].childNodes[k].style) {
                        l[o].childNodes[k].style.display = "none"
                    }
                }
            }
        }
    }
    if (j) {
        var n = function(a) {
            if (document.readyState !== "complete") {
                setTimeout(function() {
                    n(a)
                }, 100);
                return 
            }
            for (var b = 0; b < a.length; ++b) {
                if (a[b].parentNode) {
                    a[b].parentNode.removeChild(a[b])
                }
            }
        };
        n(j)
    }
};
window.ISQ.removeNanoRepLoading();
ISQ.Embed = {};
(function() {
    var a = window._nRepData;
    if (!a) {
        throw ("nanoRep widget: invalid widget data (0)")
    }
    var d = a.embed;
    if (!d) {
        throw ("nanoRep widget: invalid widget data (1)")
    }
    var f = ISQ.Widget.extractDataFromScript("embed.js");
    var c = typeof(d) === "object"&&!(d instanceof Array) && d.processed;
    if (!c) {
        var j, e, b;
        var g;
        if (d instanceof Array) {
            g = {
                width: d[0],
                height: d[1],
                container: d[2]
            }
        } else {
            if (typeof(d) === "object") {
                g = d
            }
        }
        g.referer = ISQ.Http.getNoHashUrl();
        if (f.preview && f.referer && g.referer.startsWith(f.serverUrl)) {
            g.referer = f.referer
        }
        g.type = "embed";
        g.id = g.id || "embed";
        g.serverUrl = f.serverUrl || (window.location.protocol + "//" + ISQ.Http.domain(true));
        g.account = f.account || d.account || window._nRepData.account;
        g.preview = f.preview || false;
        g.debug = f.debug || false;
        g.cdcUser = null;
        g.dynamicSize = d.dynamicSize || false;
        g.cdcFrame = d.cdcFrame || "";
        g.cdcVersion = d.cdcVersion || 0;
        g.maxHeight = d.maxHeight;
        g.useExistingCNF = d.useExistingCNF === true;
        a.embed = g
    }
})();
ISQ.Embed.dispose = function(b) {
    if (!_nRepData.embed) {
        return 
    }
    if (!_nRepData.embed.processed) {
        return 
    }
    var a = ISQ.Html._(_nRepData.embed.container);
    if (_nRepData.embed.cdcVersion === 2) {
        ISQ.Widget.cdcMessageEvent.removeHandler(ISQ.Embed.cdcMessageHandler, _nRepData.embed)
    } else {
        if (_nRepData.embed.cdcVersion === 3) {
            _nRepData.embed.cdcUser.dispose()
        }
    }
    if (a) {
        a.style.height = "auto"
    }
    _nRepData.embed.sentMessageToWidget = false;
    if (b) {
        delete _nRepData.embed
    }
    clearNode(a)
};
ISQ.Embed.reloadWidget = function(a) {
    ISQ.Embed.dispose(a ? true : false);
    if (a) {
        _nRepData.embed = a
    }
    ISQ.Embed.loadWidget(_nRepData.embed)
};
ISQ.Embed.loadWidget = function(e) {
    if (ISQ.Http.browser.isIE6) {
        if (window._nRepData.windowLoaded !== true) {
            setTimeout(function() {
                ISQ.Embed.loadWidget(e)
            }, 300);
            return 
        }
    }
    var b = ISQ.Html._(e.container);
    if (b === null) {
        if (window._nRepData.windowLoaded !== true) {
            setTimeout(function() {
                ISQ.Embed.loadWidget(e)
            }, 200)
        } else {
            throw ("nanoRep widget: invalid container")
        }
        return 
    }
    var d = [];
    d.push(e.serverUrl);
    d.push("/widget/widget.html");
    ISQ.Widget.appendConfigurationsToIframeSource(d, e);
    if (e.dynamicSize) {
        d.push("&dynamicsize=true");
        if (e.maxHeight) {
            d.push("&dynamicmaxsize=");
            d.push(e.maxHeight)
        }
        e.height = 1
    }
    d.push("&notifyclick=true");
    var g = typeof(e.width) === "string" && e.width.endsWith("%") ? e.width: e.width + "px";
    var c = "nanoRep_frame_embed_" + new Date().getTime();
    var a = document.createElement("iframe");
    a.style.height = e.height + "px";
    a.style.width = g;
    a.style.background = "transparent";
    a.style.position = "static";
    a.style.zIndex = 20;
    a.style.border = "0px";
    a.style.margin = "0px";
    a.setAttribute("allowTransparency", true);
    a.setAttribute("frameBorder", false);
    a.setAttribute("scrolling", false);
    a.setAttribute("id", c);
    a.setAttribute("name", c);
    a.setAttribute("title", "Instant answers frame, access this frame to ask a question.");
    a.setAttribute("nanoRep_loading_noRemove", true);
    e.iframe = a;
    ISQ.Embed.initCDC(e);
    e.eCnfLoaded = ISQ.Embed.cnfLoaded;
    e.eAfterCnfSent = ISQ.Embed.afterCnfSent;
    ISQ.Widget.downloadAndSendCNF(e);
    if (ISQ.Widget.sendingMessagesToWidget && (!ISQ.Widget.supportsPostMessage&&!e.dynamicSize)) {
        var f = function() {
            ISQ.Widget.sendAPItoWidget(e);
            e.sentMessageToWidget = true
        };
        if (ISQ.Http.browser.app === "opera") {
            e.iframe.onload = f
        } else {
            if (document.addEventListener) {
                e.iframe.addEventListener("load", f, true)
            } else {
                if (document.attachEvent) {
                    e.iframe.attachEvent("onload", f)
                }
            }
        }
    }
    a.src = d.join("");
    ISQ.Widget.fixStealFocus(e);
    b.appendChild(a);
    if (ISQ.Http.browser.oldAndroid) {
        ISQ.Widget.createMobileBlockerLink(e, b, g, e.height + "px")
    }
    e.size = [e.width, e.height]
};
ISQ.Embed.cnfLoaded = function() {
    var b = _nRepData.embed;
    if (ISQ.Cnf_Embed.customEmbeddedSize) {
        var a = ISQ.Cnf_Embed.customEmbeddedSize.split(",");
        if (a.length === 2) {
            a[0] = parseInt(a[0]);
            a[1] = parseInt(a[1]);
            b.width = a[0];
            b.height = a[1];
            if (b.iframe) {
                b.iframe.style.height = a[0] + "px";
                b.iframe.style.width = a[1] + "px"
            }
        }
    }
    if (ISQ.Cnf_Embed.onLoadPageScript) {
        ISQ.API.runPageScript(ISQ.Cnf_Embed.onLoadPageScript, false)
    }
    ISQ.Widget.Mobile.resizeStart()
};
ISQ.Embed.afterCnfSent = function() {
    if (typeof(ISQ.Cnf_Embed.faqData) === "string") {
        var a = _nRepData.embed;
        nanoRep.FAQ.requestFaqLists({
            serverUrl: a.serverUrl,
            account: a.account,
            kb: a.kb,
            referer: a.referer,
            encoding: a.encoding,
            context: a.context,
            isFloat: false,
            callback: null,
            cdcUser: a.cdcUser,
            configId: a.customValues && a.customValues.configId ? a.customValues.configId: null
        })
    }
};
ISQ.Embed.Mobile = {};
ISQ.Embed.Mobile.resizeAction = function(c) {
    if (!_nRepData.embed.sentMessageToWidget) {
        return true
    }
    var d = _nRepData.embed;
    var b = d.size[0];
    var a = 275;
    if (c < a) {
        c = a
    }
    if (c < b) {
        ISQ.Embed.Mobile.setWidgetWidth(c - 20, d, b)
    } else {
        if (d.iframe.parentNode.offsetWidth < b) {
            ISQ.Embed.Mobile.setWidgetWidth(b - 20, d, b)
        }
    }
    return false
};
ISQ.Embed.Mobile.setWidgetWidth = function(c, b, a) {
    if (c == 0 || c > a) {
        c = a
    }
    if (b.iframe.offsetWidth === c) {
        return 
    }
    b.iframe.parentNode.style.width = c + "px";
    b.iframe.style.width = c + "px";
    if (b.dynamicSize) {
        b.cdcUser.sendMessage("subject=dynamicSizeRequest")
    }
};
ISQ.Embed.initCDC = function(a) {
    ISQ.Widget.initCDC(a, ISQ.Embed.cdcMessageHandler);
    ISQ.API.init(a)
};
ISQ.Embed.cdcMessageHandler = function(f) {
    if (f.widget !== "embed") {
        return 
    }
    var g = f.subject;
    if (!this.sentMessageToWidget) {
        ISQ.Widget.sendAPItoWidget(this);
        this.sentMessageToWidget = true
    }
    switch (g) {
    case"reset":
        this.sentMessageToWidget = false;
        this.size[1] = 1;
        break;
    case"goUp":
        var a = ISQ.Html.position(this.iframe.parentNode).y;
        if (a < ISQ.Html.getScroll("top")) {
            window.scrollTo(ISQ.Html.getScroll("left"), a)
        }
        break;
    case"setSize":
        var d = f.height;
        var c = (this.size[1] === 1);
        if (this.iframe.offsetHeight == d) {
            return 
        }
        if (ISQ.Http.browser.isMobile) {
            if (c) {
                this.size[1] = d
            }
        }
        if (!ISQ.Widget.resizeIframeOnLoad_IE(this, 0, d, ISQ.Embed.windowLoaded)) {
            this.iframe.parentNode.style.height = d + "px";
            this.iframe.style.height = d + "px";
            if (this.mobileBlockerLink) {
                this.mobileBlockerLink.style.height = d + "px";
                this.mobileBlockerLink.style.top = "-" + d + "px"
            }
            if (!c) {
                var a = ISQ.Html.position(this.iframe.parentNode).y;
                if (a + this.iframe.parentNode.offsetHeight < ISQ.Html.getScroll("top")) {
                    window.scrollTo(0, a)
                }
            }
        }
        if (ISQ.Http.browser.isIOS7) {
            this.cdcUser.sendMessage("subject=setHeight&size=" + d)
        }
        if (_nRepData.embed.cdcUser.resizeIframe) {
            _nRepData.embed.cdcUser.resizeIframe.apply(this, [d])
        }
        break;
    case"click":
        try {
            if (typeof(nanoRep.FAQ) != "undefined" && typeof(nanoRep.FAQ.Likes) != "undefined") {
                nanoRep.FAQ.Likes.toggleDislikeOptions(true)
            }
        } catch (b) {}
        break;
    default:
        ISQ.Widget.handleCommonCDCMessage(f, this);
        break
    }
};
ISQ.Embed.windowLoaded = function(b, c, a) {
    b.iframe.parentNode.style.height = a + "px";
    b.iframe.style.height = a + "px"
};
window._nRepData.Widgets = window._nRepData.Widgets || {};
window._nRepData.Widgets._widgetId = 0;
window._nRepData.Widgets.getNextWidgetId = function() {
    return "widget_" + window._nRepData.Widgets._widgetId++
};
window._nRepData.Widgets.mergeCommonProperties = function(f) {
    var d = window._nRepData;
    var e = function(a, b) {
        if (!b) {
            if (d[a]&&!f[a]) {
                f[a] = d[a]
            }
        } else {
            if (d[a] && (!f.customValues ||!f.customValues[a])) {
                if (!f.customValues) {
                    f.customValues = {}
                }
                f.customValues[a] = d[a]
            }
        }
    };
    e("pageId");
    e("conversionTag");
    e("searchDelay");
    e("splitTestGroup");
    e("kb");
    e("formValues");
    e("context");
    e("customParams");
    e("setFocus");
    e("onloadQuestionId", true);
    e("configId", true);
    if (f.type === "float") {
        e("anchorX")
    }
    if (!f.events && (d.events || d.reportQueryHandler)) {
        if (d.events) {
            f.events = d.events
        } else {
            f.events = {};
            f.events.query = d.reportQueryHandler
        }
    }
    if (f.events && f.events.reportQueryHandler) {
        f.events.query = f.events.reportQueryHandler
    }
    ISQ.Widget.handleEventMessage({
        type: "scriptload"
    }, f)
};
(function() {
    var b = window._nRepData;
    if (typeof(window.nanoRep.Embed) != "undefined" && b.embed&&!b.embed.processed) {
        b.embed.processed = true;
        window._nRepData.Widgets.mergeCommonProperties(b.embed);
        window.nanoRep.Embed.loadWidget(b.embed)
    }
    if (typeof(window.nanoRep.Float) != "undefined" && b["float"]&&!b["float"].processed) {
        b["float"].processed = true;
        window._nRepData.Widgets.mergeCommonProperties(b["float"]);
        window.nanoRep.Float.preInit()
    }
})();
