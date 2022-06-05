(function() {
    var cookieName = "__CG";
    var debugQueryParam = "Debug";
    var debugURL = "g.3gl.net/jp/v2/D";
    var testUserId = "test";
    var cpV = 2;
    var RT = parent.window;
    var RP = RT.RProfiler;
    var RC = undefined;
    if (!document.getElementById ||!(window.attachEvent || window.addEventListener)) {
        return;
    }
    var tload = 'load';
    var tBefUnload = 'beforeunload';
    var pFld = "data";
    var stOk = 0;
    var pFurl, json, inD;
    var prot = RT.location.protocol + "//";
    function addTg(eUrl, t) {
        var s = t.document.createElement('script');
        s.type = 'text/javascript';
        s.src = prot + eUrl;
        t.document.body.appendChild(s);
    }
    function qStV(n) {
        var q = RT.location.search.substring(1);
        var v = q.split("&");
        for (var i = 0; i < v.length; i++) {
            var p = v[i].split("=");
            if (p[0] == n) {
                return p[1];
            }
        }
        return false;
    }
    function stop() {
        if (!RP) {
            return;
        }
        RP.$E.clearAll();
    }
    function setCookie(n, v, y, d) {
        var e = "";
        if (y) {
            var t = new Date();
            t.setTime(t.getTime() + (y * 24 * 60 * 60 * 1000));
            e = "; expires=" + t.toGMTString();
        }
        document.cookie = n + "=" + escape(v) + e + "; path=/; domain=" + d + ";";
    }
    function getCookie(r) {
        var c, s = document.cookie.split(';');
        for (var i = 0; i < s.length; i++) {
            c = s[i];
            if (r.test(c)) {
                return unescape(c.substring(c.indexOf("=") + 1, c.length));
            }
        }
        return null;
    }
    var sD = (function () {
        var cNrg = new RegExp("^(\\s)*" + cookieName + "=", "i");
        var sEx = 30 * 60 * 1000, cEx = 365;
        var uF = null;
        var dmain;
        function empt() {
            return {
                uI: - 1,
                sI: 0,
                st: 0,
                pvc: 0,
                pf: - 1,
                doi: - 1
            };
        }
        function upSt() {
            if (!uF) {
                return;
            }
            var t = (new Date()).getTime();
            if (t - uF.st > sEx) {
                uF.sI = 0;
            }
            uF.st = t;
            upCk();
        }
        function upCk() {
            if (!uF) {
                return;
            }
            var d, h, v = "u:" + uF.uI + "," + "s:" + uF.sI + "," + "t:" + uF.st + "," + "c:" + uF.pvc + "," + "k:" + uF.cks + "," + "f:" + uF.pf;
            if (!dmain) {
                var a, i = uF.doi, m = document.domain.split(".");
                if (i==-1) {
                    d = m[m.length - 1];
                    for (h = m.length - 2; h >= 0; h--) {
                        d = m[h] + "." + d;
                        setCookie(cookieName, v + ",i:" + h, cEx, d);
                        if (getCookie(cNrg)) {
                            uF.doi = h;
                            dmain = d;
                            break;
                        }
                    }
                    return;
                }
                if (i < m.length) {
                    d = m[m.length - 1];
                    for (h = m.length - 2; h >= i; h--) {
                        d = m[h] + "." + d;
                    }
                }
                dmain = d;
            }
            v = v + ",i:" + uF.doi;
            setCookie(cookieName, v, cEx, dmain);
        }
        function uIFrmCk(r) {
            var c, l, kv, v, uI =- 1, s = document.cookie.split(';');
            for (var k = 0; k < s.length; k++) {
                c = s[k];
                if (!r.test(c)) {
                    continue;
                }
                c = unescape(c.substring(c.indexOf("=") + 1, c.length));
                l = c.split(',');
                for (var i = 0; i < l.length; i++) {
                    if (l[i].indexOf("u:") !== 0) {
                        continue;
                    }
                    kv = l[i].split(':');
                    if (kv.length !== 2 || kv[1].length <= 0) {
                        break;
                    }
                    if (kv[1] !== testUserId) {
                        v = parseInt(kv[1], 10);
                        if (!isNaN(v) && v!=-1) {
                            return v;
                        }
                    } else if (userId===-1) {
                        uI = kv[1];
                    }
                }
            }
            return uI;
        }
        function readCk() {
            var c = getCookie(cNrg);
            uF = empt();
            if (!c) {
                return;
            }
            var uId, kv, v, i, l = c.split(',');
            for (i = 0; i < l.length; i++) {
                kv = l[i].split(':');
                if (kv.length === 2) {
                    switch (kv[0]) {
                    case 'u':
                        if (kv[1] === testUserId) {
                            uF.uI = kv[1];
                        } else {
                            v = parseInt(kv[1], 10);
                            if (!isNaN(v)) {
                                uF.uI = v;
                            }
                        }
                        break;
                    case 's':
                        v = parseInt(kv[1], 10);
                        if (!isNaN(v)) {
                            uF.sI = v;
                        }
                        break;
                    case 't':
                        v = parseInt(kv[1], 10);
                        if (!isNaN(v)) {
                            uF.st = v;
                        }
                        break;
                    case 'c':
                        v = parseInt(kv[1], 10);
                        if (!isNaN(v)) {
                            uF.pvc = v;
                        }
                        break;
                    case 'k':
                        uF.cks = kv[1];
                        break;
                    case 'f':
                        v = parseInt(kv[1], 10);
                        if (!isNaN(v)) {
                            uF.pf = v;
                        }
                        break;
                    case 'i':
                        v = parseInt(kv[1], 10);
                        if (!isNaN(v)) {
                            uF.doi = v;
                        }
                        break;
                    }
                }
            }
        }
        function setPf() {
            uF.pf = 0;
            if (!RC) {
                uF.pf =- 1;
                return;
            }
            if ((uF.uI%10000) <= RC.r) {
                uF.pf = 1;
            }
        }
        function CkSC(u) {
            var i, n = 0, hn = getHostName(u).replace(":", "-");
            var x = u.indexOf("?");
            for (i = hn.length; i < u.length; i++) {
                n += (u.charCodeAt(i)%i);
            }
            return hn + "/" + ((x < 0) ? u.length : x) + "/" + u.length + "/" + n;
        }
        function getHostName(u) {
            var s = u.indexOf("//") + 2;
            var e = u.indexOf("/", s);
            if (s < 2 && e===-1) {
                return u;
            }
            if (e===-1) {
                e = u.length;
            }
            return u.substring(s, e);
        }
        function loadCk() {
            var s = RP.d.s;
            readCk();
            if (typeof uF.uI !== 'number' || uF.uI===-1) {
                uId = uIFrmCk(cNrg);
                if (uId !== testUserId && uId>-1) {
                    uF.uI = uId;
                    setPf();
                }
            }
            if (uF.uI===-1 || uF.uI === testUserId) {
                uF.uI = Math.floor(1 + (Math.random() * ((Math.pow(2, 64) - 2) / 2)));
                setPf();
            }
            if (uF.sI === 0 || (s - uF.st > sEx)) {
                uF.sI = Math.floor(1 + (Math.random() * ((Math.pow(2, 32) - 2) / 2)));
                uF.pvc = 1;
            } else {
                if (uF.pvc < 65535) {
                    uF.pvc++;
                }
                if (CkSC(RT.document.referrer) === uF.cks && uF.st > 0) {
                    uF.rT = s - uF.st;
                }
            }
            uF.st = (new Date()).getTime();
            uF.pvI = Math.floor(1 + (Math.random() * ((Math.pow(2, 16) - 2) / 2)));
            uF.cks = CkSC(window.location.href);
            upCk();
            if (!getCookie(cNrg)) {
                return false;
            }
            return true;
        }
        return {
            lc: loadCk,
            ust: upSt,
            pf: (function () {
                return (!!uF) ? uF.pf : 0;
            }),
            uI: (function () {
                return (!!uF) ? uF.uI : - 1;
            }),
            sI: (function () {
                return (!!uF) ? uF.sI : 0;
            }),
            pvc: (function () {
                return (!!uF) ? uF.pvc : 0;
            }),
            pvI: (function () {
                return (!!uF) ? uF.pvI : 0;
            }),
            rT: (function() {
                return (!!uF) ? uF.rT : 0;
            })
        };
    })();
    function cJson() {
        function f(n) {
            return n < 10 ? '0' + n : n;
        }
        function toJSON(d) {
            return d.getUTCFullYear() + '-' + f(d.getUTCMonth() + 1) + '-' + f(d.getUTCDate()) + 'T' + f(d.getUTCHours()) + ':' + f(d.getUTCMinutes()) + ':' + f(d.getUTCSeconds()) + 'Z';
        }
        function escape(h) {
            var c = m[h];
            if (c) {
                return c;
            }
            c = h.charCodeAt();
            return '\\u00' + Math.floor(c / 16).toString(16) + (c%16).toString(16);
        }
        var m = {
            '\b': '\\b',
            '\t': '\\t',
            '\n': '\\n',
            '\f': '\\f',
            '\r': '\\r',
            '"': '\\"',
            '\\': '\\\\'
        };
        var r = /["\\\x00-\x1f\x7f-\x9f]/g;
        function stringify(value, whitelist) {
            var a = [], i, k, l, v, h;
            var propertyNameList = whitelist || value;
            switch (typeof value) {
            case 'string':
                return r.test(value) ? '"' + value.replace(r, escape) + '"' : '"' + value + '"';
            case 'number':
                return isFinite(value) ? String(value) : 'null';
            case 'boolean':
            case 'null':
                return String(value);
            case 'object':
                if (!value) {
                    return 'null';
                }
                if (value.constructor === Date) {
                    return stringify(toJSON(value));
                }
                if (typeof value.length === 'number' && !(value.propertyIsEnumerable('length'))) {
                    l = value.length;
                    for (i = 0; i < l; i += 1) {
                        a.push(stringify(value[i], whitelist) || 'null');
                    }
                    return '[' + a.join(',') + ']';
                }
                for (k in propertyNameList) {
                    if (typeof k === 'string') {
                        v = stringify(value[k], whitelist);
                        if (v) {
                            a.push(stringify(k.toLowerCase()) + ':' + v);
                        }
                    }
                }
                return '{' + a.join(',') + '}';
            }
        }
        return {
            stringify: stringify,
            parse: function(text, filter) {
                var j;
                function walk(k, v) {
                    var i, n;
                    if (v && typeof v === 'object') {
                        for (i in v) {
                            if (Object.prototype.hasOwnProperty.apply(v, [i])) {
                                n = walk(i, v[i]);
                                if (n !== undefined) {
                                    v[i] = n;
                                }
                            }
                        }
                    }
                    return filter(k, v);
                }
                var textString = text.replace(/\\./g, '@');
                textString = testString.replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(:?[eE][+\-]?\d+)?/g, ']');
                textString = testString.replace(/(?:^|:|,)(?:\s*\[)+/g, '');
                if (/^[\],:{}\s]*$/.test(textString)) {
                    j = eval('(' + text + ')');
                    return typeof filter === 'function' ? walk('', j) : j;
                }
                throw new SyntaxError('parseJSON');
            }
        };
    }
    function dPost(dstr) {
        var root = document.createElement('iframe');
        root.src = pFurl + "?" + pFld + "=" + dstr;
        var s = root.style;
        s.position = 'absolute';
        s.top = '-10000px';
        s.left = '-1000px';
        document.body.appendChild(root);
        if ("onreadystatechange" in root) {
            root.onreadystatechange = function() {
                if (root.readyState == 'complete') {
                    iMang(root);
                }
            };
        } else {
            root.onload = function() {
                iMang(root);
            }
        }
    }
    function iMang(root) {
        setTimeout(function() {
            root.parentNode.removeChild(root);
        }, 100);
    }
    function doPstRp(y) {
        var so = crtPostObj(y);
        if (so == null || so.dI===-1) {
            return;
        }
        var jstr = json.stringify(so);
        dPost(jstr);
    }
    function doDbgRp(y) {
        var so = crtPostObj(y);
        if (so == null) {
            return;
        }
        RP.updateDisplay(y, so);
    }
    function crtPostObj(y) {
        if (y !== tload) {
            return null;
        }
        var c = window.RPClient;
        var s = {
            v: cpV,
            y: y || 'unknown',
            uI: sD.uI(),
            sI: sD.sI(),
            pI: sD.pvI(),
            dI: ((c && c.c) ? c.c : - 1),
            pc: sD.pvc()
        };
        if (window.performance) {
            var prfT = RT.performance.timing;
            s.dn = prfT.domainLookupEnd - prfT.domainLookupStart;
            s.fc = prfT.connectEnd - prfT.connectStart;
            if (prfT.secureConnectionStart) {
                s.sc = prfT.connectEnd - prfT.secureConnectionStart;
            }
            s.wt = prfT.responseStart - prfT.requestStart;
            s.ld = prfT.responseEnd - prfT.responseStart;
            s.de = prfT.domInteractive - prfT.navigationStart;
            s.dL = prfT.domContentLoadedEventStart - prfT.navigationStart;
            s.dc = prfT.domComplete - prfT.navigationStart;
            s.rp = prfT.responseEnd - prfT.navigationStart;
            s.cL = prfT.loadEventStart - prfT.domLoading;
            s.rd = prfT.redirectEnd - prfT.redirectStart;
            if (window.performance.navigation) {
                prfT = RT.performance.navigation;
                s.rc = prfT.redirectCount;
            }
        }
        if (RP.d.jsC !== undefined) {
            s.jsC = RP.d.jsC;
        }
        s.dh = screen.height;
        s.dw = screen.width;
        var exTE = sD.rT();
        var enTL = RP.d.lT - RP.d.s;
        if (exTE > 0) {
            s.xe = exTE;
        }
        if (enTL > 0) {
            s.el = enTL;
        }
        if (window.chrome) {
            var l = RT.chrome.loadTimes();
            if (l.wasFetchedViaSpdy) {
                s.sv = l.npnNegotiatedProtocol;
            }
        }
        s.rf = escape(RT.location.href);
        if (document.webkitVisibilityState === "prerender") {
            s.pr = 1;
        }
        if (RP.a !== undefined) {
            var it;
            for (name in RP.a) {
                switch (name) {
                case 'appError':
                    it = RP.a[name];
                    var n;
                    if (it && typeof it === 'object') {
                        for (i in it) {
                            i = Number(i);
                            if (isNaN(i)) {
                                continue;
                            }
                            n = it[i];
                            if (n && typeof n == 'string') {
                                if (n.length > 32) {
                                    n = n.subtring(0, 32);
                                }
                                s.ae = {};
                                s.ae[i] = n;
                                break;
                            }
                        }
                    }
                    break;
                case 'conversion':
                    it = RP.a[name];
                    s.cv = 1;
                    if (it) {
                        if (typeof it === 'object') {
                            for (i in it) {
                                i = Number(i);
                                if (isNaN(i)) {
                                    continue;
                                }
                                n = it[i];
                                if (n && typeof n == 'number') {
                                    s.rv = i;
                                    s.ri = n;
                                    break;
                                }
                            }
                        }
                    }
                    break;
                case 'indicator':
                    it = {};
                    if (insightBld('number', name, it)) {
                        s.ind = it;
                    }
                    break;
                case 'tracepoint':
                    it = {};
                    if (insightBld('string', name, it)) {
                        s.tra = it;
                    }
                    break;
                case 'pageGroup':
                    it = RP.a[name];
                    if (it !== undefined && typeof it == 'string') {
                        s.pg = it;
                    }
                    break;
                case 'variation':
                    it = RP.a[name];
                    if (it !== undefined && typeof it == 'string') {
                        s.ab = it;
                    }
                    break;
                }
            }
        }
        return s;
    }
    function insightBld(valT, name, result) {
        var it = RP.a[name];
        var n = undefined;
        if (it && typeof it === 'object') {
            for (i in it) {
                if (i) {
                    n = it[i];
                    if (n && typeof n == valT) {
                        result[i] = n;
                    }
                }
            }
        }
        return !!n;
    }
    function shrtUnBL() {
        RP.$E.add(tBefUnload, RT, function () {
            sD.ust();
        });
    }
    function registerPageLoad() {
        if (inD) {
            RP.onDbgLd = function () {
                doDbgRp(tload);
                stop();
            };
            addTg(debugURL + '?' + Math.random(), RT);
            return;
        }
        if (sD.pf() == 1) {
            if (RP.d.Lf != 1 && RP.d.Lf!=-2) {
                RP.d.Lf =- 2;
                setTimeout(registerPageLoad, 1);
                return;
            }
            doPstRp(tload);
        }
        stop();
        shrtUnBL();
    }
    function start() {
        if (!RP ||!RP.d || stOk == 1) {
            return;
        }
        stOk = 1;
        RC = window.RPClient;
        json = cJson();
        pFurl = prot + RC.pu;
        inD = qStV(debugQueryParam);
        var nhc=!sD.lc();
        if (!inD && nhc) {
            stop();
            return;
        }
        if (RP.d.Lf == 1) {
            registerPageLoad();
        } else {
            RP.$E.add(tload, RT, registerPageLoad);
        }
    }
    if (window.RPClient) {
        start();
    } else {
        document.onConfigLoaded = start;
    }
})();

