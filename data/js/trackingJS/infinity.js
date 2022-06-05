if (typeof(g367CB268B1094004A3689751E7AC568F) == "undefined" ||!g367CB268B1094004A3689751E7AC568F.Core) {
    (function(x, s) {
        var d = "", t = "?", e = "function", r = "undefined", l = "object", f = "major", j = "model", k = "name", p = "type", v = "vendor", w = "version", b = "architecture", c = "console", i = "mobile", o = "tablet";
        var u = {
            has: function(y, z) {
                return z.toLowerCase().indexOf(y.toLowerCase())!==-1
            },
            lowerize: function(y) {
                return y.toLowerCase()
            }
        };
        var g = {
            rgx: function() {
                for (var I, z = 0, A, B, E, G, D, C, y = arguments; z < y.length; z += 2) {
                    var H = y[z], F = y[z + 1];
                    if (typeof I === r) {
                        I = {};
                        for (E in F) {
                            G = F[E];
                            if (typeof G === l) {
                                I[G[0]] = s
                            } else {
                                I[G] = s
                            }
                        }
                    }
                    for (A = B = 0; A < H.length; A++) {
                        D = H[A].exec(this.getUA());
                        if (!!D) {
                            for (E = 0; E < F.length; E++) {
                                C = D[++B];
                                G = F[E];
                                if (typeof G === l && G.length > 0) {
                                    if (G.length == 2) {
                                        if (typeof G[1] == e) {
                                            I[G[0]] = G[1].call(this, C)
                                        } else {
                                            I[G[0]] = G[1]
                                        }
                                    } else {
                                        if (G.length == 3) {
                                            if (typeof G[1] === e&&!(G[1].exec && G[1].test)) {
                                                I[G[0]] = C ? G[1].call(this, C, G[2]) : s
                                            } else {
                                                I[G[0]] = C ? C.replace(G[1], G[2]) : s
                                            }
                                        } else {
                                            if (G.length == 4) {
                                                I[G[0]] = C ? G[3].call(this, C.replace(G[1], G[2])) : s
                                            }
                                        }
                                    }
                                } else {
                                    I[G] = C ? C : s
                                }
                            }
                            break
                        }
                    }
                    if (!!D) {
                        break
                    }
                }
                return I
            },
            str: function(B, A) {
                for (var y in A) {
                    if (typeof A[y] === l && A[y].length > 0) {
                        for (var z = 0; z < A[y].length; z++) {
                            if (u.has(A[y][z], B)) {
                                return y === t ? s : y
                            }
                        }
                    } else {
                        if (u.has(A[y], B)) {
                            return y === t ? s : y
                        }
                    }
                }
                return B
            }
        };
        var h = {
            browser: {
                oldsafari: {
                    major: {
                        1: ["/8", "/1", "/3"],
                        2: "/4",
                        "?": "/"
                    },
                    version: {
                        "1.0": "/8",
                        1.2: "/1",
                        1.3: "/3",
                        "2.0": "/412",
                        "2.0.2": "/416",
                        "2.0.3": "/417",
                        "2.0.4": "/419",
                        "?": "/"
                    }
                }
            },
            device: {
                sprint: {
                    model: {
                        "Evo Shift 4G": "7373KT"
                    },
                    vendor: {
                        HTC: "APA",
                        Sprint: "Sprint"
                    }
                }
            },
            os: {
                windows: {
                    version: {
                        ME: "4.90",
                        "NT 3.11": "NT3.51",
                        "NT 4.0": "NT4.0",
                        2000: "NT 5.0",
                        XP: ["NT 5.1", "NT 5.2"],
                        Vista: "NT 6.0",
                        7: "NT 6.1",
                        8: "NT 6.2",
                        8.1: "NT 6.3",
                        RT: "ARM"
                    }
                }
            }
        };
        var n = {
            browser: [[/(opera\smini)\/((\d+)?[\w\.-]+)/i, /(opera\s[mobiletab]+).+version\/((\d+)?[\w\.-]+)/i, /(opera).+version\/((\d+)?[\w\.]+)/i, /(opera)[\/\s]+((\d+)?[\w\.]+)/i], [k, w, f], [/\s(opr)\/((\d+)?[\w\.]+)/i], [[k, "Opera"], w, f], [/(kindle)\/((\d+)?[\w\.]+)/i, /(lunascape|maxthon|netfront|jasmine|blazer)[\/\s]?((\d+)?[\w\.]+)*/i, /(avant\s|iemobile|slim|baidu)(?:browser)?[\/\s]?((\d+)?[\w\.]*)/i, /(?:ms|\()(ie)\s((\d+)?[\w\.]+)/i, /(rekonq)((?:\/)[\w\.]+)*/i, /(chromium|flock|rockmelt|midori|epiphany|silk|skyfire|ovibrowser|bolt|iron)\/((\d+)?[\w\.-]+)/i], [k, w, f], [/(trident).+rv[:\s]((\d+)?[\w\.]+).+like\sgecko/i], [[k, "IE"], w, f], [/(yabrowser)\/((\d+)?[\w\.]+)/i], [[k, "Yandex"], w, f], [/(comodo_dragon)\/((\d+)?[\w\.]+)/i], [[k, /_/g, " "], w, f], [/(chrome|omniweb|arora|[tizenoka]{5}\s?browser)\/v?((\d+)?[\w\.]+)/i], [k, w, f], [/(dolfin)\/((\d+)?[\w\.]+)/i], [[k, "Dolphin"], w, f], [/((?:android.+)crmo|crios)\/((\d+)?[\w\.]+)/i], [[k, "Chrome"], w, f], [/version\/((\d+)?[\w\.]+).+?mobile\/\w+\s(safari)/i], [w, f, [k, "Mobile Safari"]], [/version\/((\d+)?[\w\.]+).+?(mobile\s?safari|safari)/i], [w, f, k], [/webkit.+?(mobile\s?safari|safari)((\/[\w\.]+))/i], [k, [f, g.str, h.browser.oldsafari.major], [w, g.str, h.browser.oldsafari.version]], [/(konqueror)\/((\d+)?[\w\.]+)/i, /(webkit|khtml)\/((\d+)?[\w\.]+)/i], [k, w, f], [/(navigator|netscape)\/((\d+)?[\w\.-]+)/i], [[k, "Netscape"], w, f], [/(swiftfox)/i, /(icedragon|iceweasel|camino|chimera|fennec|maemo\sbrowser|minimo|conkeror)[\/\s]?((\d+)?[\w\.\+]+)/i, /(firefox|seamonkey|k-meleon|icecat|iceape|firebird|phoenix)\/((\d+)?[\w\.-]+)/i, /(mozilla)\/((\d+)?[\w\.]+).+rv\:.+gecko\/\d+/i, /(uc\s?browser|polaris|lynx|dillo|icab|doris|amaya|w3m|netsurf|qqbrowser)[\/\s]?((\d+)?[\w\.]+)/i, /(links)\s\(((\d+)?[\w\.]+)/i, /(gobrowser)\/?((\d+)?[\w\.]+)*/i, /(ice\s?browser)\/v?((\d+)?[\w\._]+)/i, /(mosaic)[\/\s]((\d+)?[\w\.]+)/i], [k, w, f]],
            cpu: [[/(?:(amd|x(?:(?:86|64)[_-])?|wow|win)64)[;\)]/i], [[b, "amd64"]], [/((?:i[346]|x)86)[;\)]/i], [[b, "ia32"]], [/windows\s(ce|mobile);\sppc;/i], [[b, "arm"]], [/((?:ppc|powerpc)(?:64)?)(?:\smac|;|\))/i], [[b, /ower/, "", u.lowerize]], [/(sun4\w)[;\)]/i], [[b, "sparc"]], [/(ia64(?=;)|68k(?=\))|arm(?=v\d+;)|(?:irix|mips|sparc)(?:64)?(?=;)|pa-risc)/i], [b, u.lowerize]],
            device: [[/\((ipad|playbook);[\w\s\);-]+(rim|apple)/i], [j, v, [p, o]], [/(hp).+(touchpad)/i, /(kindle)\/([\w\.]+)/i, /\s(nook)[\w\s]+build\/(\w+)/i, /(dell)\s(strea[kpr\s\d]*[\dko])/i], [v, j, [p, o]], [/\((ip[honed|\s\w*]+);.+(apple)/i], [j, v, [p, i]], [/(blackberry)[\s-]?(\w+)/i, /(blackberry|benq|palm(?=\-)|sonyericsson|acer|asus|dell|huawei|meizu|motorola)[\s_-]?([\w-]+)*/i, /(hp)\s([\w\s]+\w)/i, /(asus)-?(\w+)/i], [v, j, [p, i]], [/\((bb10);\s(\w+)/i], [[v, "BlackBerry"], j, [p, i]], [/android.+((transfo[prime\s]{4,10}\s\w+|eeepc|slider\s\w+))/i], [[v, "Asus"], j, [p, o]], [/(sony)\s(tablet\s[ps])/i], [v, j, [p, o]], [/(nintendo)\s([wids3u]+)/i], [v, j, [p, c]], [/((playstation)\s[3portablevi]+)/i], [[v, "Sony"], j, [p, c]], [/(sprint\s(\w+))/i], [[v, g.str, h.device.sprint.vendor], [j, g.str, h.device.sprint.model], [p, i]], [/(htc)[;_\s-]+([\w\s]+(?=\))|\w+)*/i, /(zte)-(\w+)*/i, /(alcatel|geeksphone|huawei|lenovo|nexian|panasonic|(?=;\s)sony)[_\s-]?([\w-]+)*/i], [v, [j, /_/g, " "], [p, i]], [/\s((milestone|droid(?:[2-4x]|\s(?:bionic|x2|pro|razr))?(:?\s4g)?))[\w\s]+build\//i, /(mot)[\s-]?(\w+)*/i], [[v, "Motorola"], j, [p, i]], [/android.+\s((mz60\d|xoom[\s2]{0,2}))\sbuild\//i], [[v, "Motorola"], j, [p, o]], [/android.+((sch-i[89]0\d|shw-m380s|gt-p\d{4}|gt-n8000|sgh-t8[56]9))/i], [[v, "Samsung"], j, [p, o]], [/((s[cgp]h-\w+|gt-\w+|galaxy\snexus))/i, /(sam[sung]*)[\s-]*(\w+-?[\w-]*)*/i, /sec-((sgh\w+))/i], [[v, "Samsung"], j, [p, i]], [/(sie)-(\w+)*/i], [[v, "Siemens"], j, [p, i]], [/(maemo|nokia).*(n900|lumia\s\d+)/i, /(nokia)[\s_-]?([\w-]+)*/i], [[v, "Nokia"], j, [p, i]], [/android\s3\.[\s\w-;]{10}((a\d{3}))/i], [[v, "Acer"], j, [p, o]], [/android\s3\.[\s\w-;]{10}(lg?)-([06cv9]{3,4})/i], [[v, "LG"], j, [p, o]], [/((nexus\s4))/i, /((nexus\s5))/i, /(lg)[e;\s-\/]+(\w+)*/i], [[v, "LG"], j, [p, i]], [/(mobile|tablet);.+rv\:.+gecko\//i], [p, v, j]],
            engine: [[/(presto)\/([\w\.]+)/i, /(webkit|trident|netfront|netsurf|amaya|lynx|w3m)\/([\w\.]+)/i, /(khtml|tasman|links)[\/\s]\(?([\w\.]+)/i, /(icab)[\/\s]([23]\.[\d\.]+)/i], [k, w], [/rv\:([\w\.]+).*(gecko)/i], [w, k]],
            os: [[/(windows)\snt\s6\.2;\s(arm)/i, /(windows\sphone(?:\sos)*|windows\smobile|windows)[\s\/]?([ntce\d\.\s]+\w)/i], [k, [w, g.str, h.os.windows.version]], [/(win(?=3|9|n)|win\s9x\s)([nt\d\.]+)/i], [[k, "Windows"], [w, g.str, h.os.windows.version]], [/\((bb)(10);/i], [[k, "BlackBerry"], w], [/(blackberry)\w*\/?([\w\.]+)*/i, /(tizen)\/([\w\.]+)/i, /(android|webos|palm\os|qnx|bada|rim\stablet\sos|meego)[\/\s-]?([\w\.]+)*/i], [k, w], [/(symbian\s?os|symbos|s60(?=;))[\/\s-]?([\w\.]+)*/i], [[k, "Symbian"], w], [/mozilla.+\(mobile;.+gecko.+firefox/i], [[k, "Firefox OS"], w], [/(nintendo|playstation)\s([wids3portablevu]+)/i, /(mint)[\/\s\(]?(\w+)*/i, /(joli|[kxln]?ubuntu|debian|[open]*suse|gentoo|arch|slackware|fedora|mandriva|centos|pclinuxos|redhat|zenwalk)[\/\s-]?([\w\.-]+)*/i, /(hurd|linux)\s?([\w\.]+)*/i, /(gnu)\s?([\w\.]+)*/i], [k, w], [/(cros)\s[\w]+\s([\w\.]+\w)/i], [[k, "Chromium OS"], w], [/(sunos)\s?([\w\.]+\d)*/i], [[k, "Solaris"], w], [/\s([frentopc-]{0,4}bsd|dragonfly)\s?([\w\.]+)*/i], [k, w], [/(ip[honead]+)(?:.*os\s*([\w]+)*\slike\smac|;\sopera)/i], [[k, "iOS"], [w, /_/g, "."]], [/(mac\sos\sx)\s?([\w\s\.]+\w)*/i], [k, [w, /_/g, "."]], [/(haiku)\s(\w+)/i, /(aix)\s((\d)(?=\.|\)|\s)[\w\.]*)*/i, /(macintosh|mac(?=_powerpc)|plan\s9|minix|beos|os\/2|amigaos|morphos|risc\sos)/i, /(unix)\s?([\w\.]+)*/i], [k, w]]
        };
        var q = function(z) {
            var y = z || (x && x.navigator && x.navigator.userAgent ? x.navigator.userAgent : d);
            if (!(this instanceof q)) {
                return new q(z).getResult()
            }
            this.getBrowser = function() {
                return g.rgx.apply(this, n.browser)
            };
            this.getCPU = function() {
                return g.rgx.apply(this, n.cpu)
            };
            this.getDevice = function() {
                return g.rgx.apply(this, n.device)
            };
            this.getEngine = function() {
                return g.rgx.apply(this, n.engine)
            };
            this.getOS = function() {
                return g.rgx.apply(this, n.os)
            };
            this.getResult = function() {
                return {
                    ua: this.getUA(),
                    browser: this.getBrowser(),
                    engine: this.getEngine(),
                    os: this.getOS(),
                    device: this.getDevice(),
                    cpu: this.getCPU()
                }
            };
            this.getUA = function() {
                return y
            };
            this.setUA = function(A) {
                y = A;
                return this
            };
            this.setUA(y)
        };
        if (typeof exports !== r) {
            if (typeof module !== r && module.exports) {
                exports = module.exports = q
            }
            exports.UAParser = q
        } else {
            x.UAParser = q;
            if (typeof define === e && define.amd) {
                define(function() {
                    return q
                })
            }
            if (typeof x.jQuery !== r) {
                var a = x.jQuery;
                var m = new q;
                a.ua = m.getResult();
                a.ua.get = function() {
                    return m.getUA()
                };
                a.ua.set = function(A) {
                    m.setUA(A);
                    var z = m.getResult();
                    for (var y in z) {
                        a.ua[y] = z[y]
                    }
                }
            }
        }
    })(this);
    var g367CB268B1094004A3689751E7AC568F = {
        Core: true,
        Guid: 0,
        Version: 0,
        InfinityHost: "//engine.phn.doublepimp.com",
        CDNHost: "//cdn.engine.phn.doublepimp.com",
        EmailInfoAddress: "help@doublepimp.com",
        ClickTagEnabled: true,
        Modules: {},
        Media: {},
        PopLoaded: false,
        PassParams: "",
        Bypass: "0",
        IsFastPop: true,
        PopRedirectDelay: 200,
        ScriptHost: "engine.phn.doublepimp.com",
        IsAdblockRequest: false,
        SubID: "",
        RequestTrackingScripts: false,
        PlaceComScorePixel: false,
        PlaceLotamePixel: false,
        PlaceLotameImpressionPixel: false,
        LotameTrackingId: "0",
        _Top: null,
        ComScoreTagType: "0",
        ComScoreClientId: "0",
        ComScoreChannelId: "0",
        IsPremiumLinkDisabled: "true",
        LotameClientId: 3722,
        UseAjax: false,
        EncodeUrl: false,
        DFPImpressionUrl: null,
        Init: function() {
            var a = this;
            var y;
            var w = document.getElementById("infinity");
            this.Guid = w.getAttribute("data-guid");
            this.Version = w.getAttribute("data-version");
            this.PassParams = w.getAttribute("data-pass-params") || "";
            this.UseAjax = w.getAttribute("data-ajax");
            this.EncodeUrl = w.getAttribute("data-encode-url");
            this.DFPImpressionUrl = w.getAttribute("data-dfp-impression") || "";
            if (this.SubID.length === 0) {
                y = w.getAttribute("data-subid");
                this.SubID = y != null ? y : ""
            }
            this._Top = self;
            if (top != self) {
                try {
                    if (top.document.location.toString()) {
                        this._Top = top;
                        var m, o = top.document.getElementsByTagName("iframe");
                        for (var q = 0; q < o.length; q++) {
                            try {
                                var f = o[q].contentDocument || o[q].contentWindow.document || o[q].document;
                                if (f === self.document) {
                                    m = o[q];
                                    break
                                }
                            } catch (h) {}
                        }
                        if (m) {
                            var n = document.createAttribute("data-inf-script-frame");
                            m.setAttributeNode(n)
                        }
                    }
                } catch (k) {}
            }
            if (window.opera) {
                return 
            }
            if (true) {
                if (navigator.userAgent.indexOf("7.0.2 Safari")!=-1) {
                    return 
                }
            }
            if (this.IsFastPop) {
                window.console = window.console || {
                    log: function(e) {}
                };
                console.log("INF_FPU: Initiated");
                g367CB268B1094004A3689751E7AC568F.PopSettings = {};
                g367CB268B1094004A3689751E7AC568F.PopSettings.Cap = 1;
                g367CB268B1094004A3689751E7AC568F.PopSettings.CapLength = 86400;
                g367CB268B1094004A3689751E7AC568F.PopSettings.Width = 0;
                g367CB268B1094004A3689751E7AC568F.PopSettings.Height = 0;
                g367CB268B1094004A3689751E7AC568F.PopSettings.RespectTargetBlank = true;
                g367CB268B1094004A3689751E7AC568F.PopSettings.UseTabUnder = true;
                g367CB268B1094004A3689751E7AC568F.PopSettings.SameTabAd = true;
                g367CB268B1094004A3689751E7AC568F.PopSettings.MobileHistory = true;
                g367CB268B1094004A3689751E7AC568F.PopSettings.UseRemoteMediaHost = "false";
                g367CB268B1094004A3689751E7AC568F.PopSettings.UsePopCustomHost = "false";
                g367CB268B1094004A3689751E7AC568F.PopSettings.PopCustomHostUrl = "";
                g367CB268B1094004A3689751E7AC568F.PopSettings.PublisherDomainAliasUrl = "engine.phn.doublepimp.com";
                g367CB268B1094004A3689751E7AC568F.PopSettings.ScriptHost = "engine.phn.doublepimp.com";
                g367CB268B1094004A3689751E7AC568F.PopSettings.KeywordFilter = null;
                g367CB268B1094004A3689751E7AC568F.PopSettings.OffsetFilter = {
                    Type: 'Any',
                    Offset: 1 
                };
                g367CB268B1094004A3689751E7AC568F.PopSettings.BrowserFilter = null;
                g367CB268B1094004A3689751E7AC568F.PopSettings.DeviceFilter = null;
                g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter = null;
                g367CB268B1094004A3689751E7AC568F.PopSettings.ReferrerFilter = null;
                var s = function(R, P) {
                    var d = (top != self && typeof(top.document.location.toString()) === "string") ? top: self;
                    var M = null;
                    P = (P || {});
                    var Q = (P.name || Math.floor((Math.random() * 1000) + 1));
                    var e = function() {
                        var U = navigator.userAgent.toLowerCase();
                        var T = {
                            webkit: /webkit/.test(U),
                            mozilla: (/mozilla/.test(U)) && (!/(compatible|webkit)/.test(U)),
                            chrome: /chrome/.test(U),
                            msie: (/msie/.test(U)) && (!/opera/.test(U)),
                            firefox: /firefox/.test(U),
                            safari: (/safari/.test(U)&&!(/chrome/.test(U))),
                            opera: /opera/.test(U)
                        };
                        T.version = (T.safari) ? (U.match(/.+(?:ri)[\/: ]([\d.]+)/) || [])[1] : (U.match(/.+(?:ox|me|ra|ie)[\/: ]([\d.]+)/) || [])[1];
                        return T
                    }();
                    function F() {
                        if (!g367CB268B1094004A3689751E7AC568F.PopSettings.KeywordFilter) {
                            return true
                        }
                        var W = false;
                        var T = window.location.href.toLowerCase();
                        for (var U in g367CB268B1094004A3689751E7AC568F.PopSettings.KeywordFilter.Keywords) {
                            var V = g367CB268B1094004A3689751E7AC568F.PopSettings.KeywordFilter.Keywords[U].toLowerCase();
                            W = g367CB268B1094004A3689751E7AC568F.PopSettings.KeywordFilter.IsRegex ? new RegExp(V).test(T) : T.indexOf(V)!=-1;
                            if (W) {
                                break
                            }
                        }
                        return W != g367CB268B1094004A3689751E7AC568F.PopSettings.KeywordFilter.Exclusive
                    }
                    function I() {
                        if (!g367CB268B1094004A3689751E7AC568F.PopSettings.ReferrerFilter) {
                            return true
                        }
                        var V = false;
                        var T = window.document.referrer.toLowerCase();
                        for (var U in g367CB268B1094004A3689751E7AC568F.PopSettings.ReferrerFilter.Referrers) {
                            var W = g367CB268B1094004A3689751E7AC568F.PopSettings.ReferrerFilter.Referrers[U].toLowerCase();
                            V = g367CB268B1094004A3689751E7AC568F.PopSettings.ReferrerFilter.IsRegex ? new RegExp(W).test(T) : T.indexOf(W)!=-1;
                            if (V) {
                                break
                            }
                        }
                        return V != g367CB268B1094004A3689751E7AC568F.PopSettings.ReferrerFilter.Exclusive
                    }
                    function G() {
                        if (!g367CB268B1094004A3689751E7AC568F.PopSettings.OffsetFilter) {
                            return true
                        }
                        var T = a.Storage.GetCookie("FastPopSessionRequestNumber");
                        if (T == null || isNaN(T)) {
                            T = 0
                        }
                        T++;
                        a.Storage.SetCookie("FastPopSessionRequestNumber", T);
                        switch (g367CB268B1094004A3689751E7AC568F.PopSettings.OffsetFilter.Type) {
                        case"Any":
                            return true;
                        case"OnlyOn":
                            return T == g367CB268B1094004A3689751E7AC568F.PopSettings.OffsetFilter.Offset;
                        case"Before":
                            return T < g367CB268B1094004A3689751E7AC568F.PopSettings.OffsetFilter.Offset;
                        case"After":
                            return T > g367CB268B1094004A3689751E7AC568F.PopSettings.OffsetFilter.Offset
                        }
                    }
                    function i() {
                        if (!g367CB268B1094004A3689751E7AC568F.PopSettings.BrowserFilter) {
                            return true
                        }
                        var U = new UAParser();
                        var T = U.getBrowser();
                        return (g367CB268B1094004A3689751E7AC568F.PopSettings.BrowserFilter.TargetChrome && T.name == "Chrome") || (g367CB268B1094004A3689751E7AC568F.PopSettings.BrowserFilter.TargetFirefox && T.name == "Firefox") || (g367CB268B1094004A3689751E7AC568F.PopSettings.BrowserFilter.TargetIe && T.name == "IE") || (g367CB268B1094004A3689751E7AC568F.PopSettings.BrowserFilter.TargetSafari && T.name == "Safari") || (T.name != "Chrome" && T.name != "Firefox" && T.name != "IE" && T.name != "Safari" && g367CB268B1094004A3689751E7AC568F.PopSettings.BrowserFilter.TargetOthers)
                    }
                    function E() {
                        if (!g367CB268B1094004A3689751E7AC568F.PopSettings.DeviceFilter) {
                            return true
                        }
                        var U = new UAParser();
                        var T = U.getDevice();
                        return (g367CB268B1094004A3689751E7AC568F.PopSettings.DeviceFilter.TargetDesktop && (T.type == "console" || T.type == undefined)) || (g367CB268B1094004A3689751E7AC568F.PopSettings.DeviceFilter.TargetMobile && T.type == "mobile") || (g367CB268B1094004A3689751E7AC568F.PopSettings.DeviceFilter.TargetTablet && T.type == "tablet")
                    }
                    function H() {
                        if (!g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter) {
                            return true
                        }
                        var U = new UAParser();
                        var T = U.getOS();
                        return (g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetAndroid && T.name == "Android") || (g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetIOS && T.name == "Mac OS") || (g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetOSX && T.name == "iOS") || (g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetWindows && T.name == "Windows" && S(T)) || (T.name != "Android" && T.name != "Mac OS" && T.name != "iOS" && T.name != "Windows" && g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetOthers)
                    }
                    function S(T) {
                        return (T.version == "Vista" && g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetWindowsVista) || (T.version == "7" && g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetWindows7) || (T.version == "8" && g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetWindows8) || (T.version != "Vista" && T.version != "7" && T.version != "8" && g367CB268B1094004A3689751E7AC568F.PopSettings.OSFilter.TargetOtherVersions)
                    }
                    function g(ad, Z, ae, Y, ab, ac) {
                        var U = g367CB268B1094004A3689751E7AC568F;
                        var aa = "toolbar=no,scrollbars=yes,location=yes,statusbar=yes,menubar=no,resizable=1,width=" + ae.toString() + ",height=" + Y.toString() + ",screenX=" + ab + ",screenY=" + ac;
                        var X = U.PopSettings.ScriptHost;
                        var T = U.PopSettings.PublisherDomainAliasUrl;
                        var V = X != T;
                        var W = function(ai) {
                            if (U.PopRunning) {
                                return 
                            }
                            U.PopRunning = true;
                            try {
                                window.console = window.console || {
                                    log: function(au) {}
                                };
                                console.log(ai)
                            } catch (ah) {}
                            if (U.PopLoaded || U.IsCapped("InfNumFastPops", "InfNumFastPopsExpire", U.PopSettings.Cap)) {
                                return 
                            }
                            var ak = ai.target || ai.srcElement;
                            if (ak.nodeName.toLowerCase() !== "a") {
                                ak = U.GetParentLink(ak)
                            }
                            if (U.IsPremiumLinkDisabled === "true" && (ak.getAttribute("target") === "_blank" || ak.getAttribute("rel") === "nofollow")) {
                                U.PopRunning = false;
                                return 
                            }
                            if (U.IsMobile.any(U._Top)) {
                                var ar = new UAParser();
                                var am = ar.getOS();
                                var ag = ar.getDevice();
                                if (am.name == "iOS" && am.version < "6") {
                                    return 
                                }
                                if (ag.vendor == "Samsung" && am.name == "Android") {
                                    return 
                                }
                                if (U.IsMobile.Windows(U._Top)) {
                                    return 
                                }
                                if (U.IsMobile.iOS(U._Top) && (navigator.userAgent.match("CriOS"))) {
                                    return 
                                }
                                if (ak.nodeName.toLowerCase() === "a" && ak.getAttribute("href") != "#" && ak.getAttribute("href").indexOf("javascript:")==-1&&!(U.IsPremiumLinkDisabled === "true" && (ak.getAttribute("target") === "_blank" || ak.getAttribute("rel") === "nofollow"))) {
                                    if (g367CB268B1094004A3689751E7AC568F.PopSettings.MobileHistory) {
                                        var al = a._Top.window.open("about:blank");
                                        al.history.replaceState({
                                            previous: a._Top.location.href
                                        }, null, a._Top.location.href);
                                        al.addEventListener("popstate", function(au) {
                                            al.location = au.state.previous
                                        });
                                        al.location = ak.getAttribute("href")
                                    } else {
                                        window.open(ak.getAttribute("href"))
                                    }
                                    U.IncrementCap("InfNumFastPops", "InfNumFastPopsExpire", U.PopSettings.CapLength);
                                    if (am.name == "Android") {
                                        ak.setAttribute("href", u)
                                    }
                                    a._Top.document.location = u;
                                    if (ai.preventDefault !== undefined) {
                                        ai.preventDefault()
                                    }
                                    g367CB268B1094004A3689751E7AC568F.PopRunning = false;
                                    return false
                                } else {
                                    g367CB268B1094004A3689751E7AC568F.PopRunning = false;
                                    return true
                                }
                            }
                            if ((e.chrome && U.PopSettings.UseTabUnder) || navigator.userAgent.indexOf("Chrome/33.0.1750.146")>-1) {
                                if (U.PopSettings.SameTabAd && V && ak.tagName.toLowerCase() === "a" && ak.getAttribute("href").lastIndexOf("javascript:", 0) !== 0) {
                                    ai.preventDefault();
                                    var ap = "inftabwindow_" + Math.floor((Math.random() * 100000000) + 1).toString();
                                    var at;
                                    var af = top.document.location.href;
                                    var aq = ak.href;
                                    try {
                                        at = window.open("about:blank", ap);
                                        at.history.pushState(null, null, af);
                                        at.location = aq
                                    } catch (ah) {
                                        at = window.open(aq, ap)
                                    }
                                    at.focus();
                                    g367CB268B1094004A3689751E7AC568F.PopLoaded = true;
                                    U.IncrementCap("InfNumFastPops", "InfNumFastPopsExpire", U.PopSettings.CapLength);
                                    window.location.href = ad + "&hosted=true"
                                } else {
                                    var aj = document.createElement("a");
                                    aj.setAttribute("data-tabunder", true);
                                    if (U.PopSettings.UseRemoteMediaHost === "true" && V) {
                                        if (U.PopSettings.UsePopCustomHost === "true") {
                                            aj.href = U.PopSettings.PopCustomHostUrl
                                        } else {
                                            aj.href = document.location.href
                                        }
                                        setTimeout(function() {
                                            at.location.href = ad + "&hosted=true"
                                        }, U.PopRedirectDelay)
                                    } else {
                                        aj.href = ad + "&hosted=true"
                                    }
                                    var ap = "inftabwindow_" + Math.floor((Math.random() * 100000000) + 1).toString();
                                    aj.target = ap;
                                    top.window.document.body.appendChild(aj);
                                    var ah = document.createEvent("MouseEvents");
                                    ah.initMouseEvent("click", true, true, window, 0, 0, 0, 0, 0, true, false, false, true, 0, null);
                                    aj.dispatchEvent(ah);
                                    U.IncrementCap("InfNumFastPops", "InfNumFastPopsExpire", U.PopSettings.CapLength)
                                }
                            } else {
                                if (e.chrome) {
                                    if (U.PopSettings.UseRemoteMediaHost === "true" && V) {
                                        var an;
                                        if (U.PopSettings.UsePopCustomHost === "true") {
                                            an = U.PopSettings.PopCustomHostUrl
                                        } else {
                                            an = document.location.href
                                        }
                                        var ao = window.open(an, Z, aa);
                                        setTimeout(function() {
                                            ao.location.href = ad;
                                            U.PopLoaded = true
                                        }, U.PopRedirectDelay)
                                    } else {
                                        window.open(ad, Z, aa)
                                    }
                                    U.IncrementCap("InfNumFastPops", "InfNumFastPopsExpire", U.PopSettings.CapLength)
                                } else {
                                    if (history.pushState && U.PopSettings.SameTabAd && V && ak.tagName.toLowerCase() === "a" && ak.getAttribute("href").lastIndexOf("javascript:", 0) !== 0) {
                                        ai.preventDefault();
                                        var ap = "inftabwindow_" + Math.floor((Math.random() * 100000000) + 1).toString();
                                        var at;
                                        var af = top.document.location.href;
                                        var aq = ak.href;
                                        try {
                                            at = window.open("about:blank", ap);
                                            at.history.pushState(null, null, af);
                                            at.location = aq
                                        } catch (ah) {
                                            at = window.open(aq, ap)
                                        }
                                        at.focus();
                                        g367CB268B1094004A3689751E7AC568F.PopLoaded = true;
                                        U.IncrementCap("InfNumFastPops", "InfNumFastPopsExpire", U.PopSettings.CapLength);
                                        window.location.href = ad + "&hosted=true"
                                    } else {
                                        M = d.window.open(ad, Z, aa);
                                        if (M) {
                                            U.PopLoaded = true;
                                            U.IncrementCap("InfNumFastPops", "InfNumFastPopsExpire", U.PopSettings.CapLength);
                                            L()
                                        }
                                    }
                                }
                            }
                            U.PopLoaded = true;
                            g367CB268B1094004A3689751E7AC568F.PopRunning = false
                        };
                        a.BindOnDocmentClick(W)
                    }
                    function L() {
                        try {
                            M.blur();
                            M.opener.window.focus();
                            window.self.window.focus();
                            window.focus();
                            if (e.firefox) {
                                K()
                            }
                            if (e.webkit) {
                                J()
                            }
                            if (e.msie) {
                                setTimeout(function() {
                                    M.blur();
                                    M.opener.window.focus();
                                    window.self.window.focus();
                                    window.focus()
                                }, 1000)
                            }
                            if (e.chrome) {
                                var U = document.createElement("A");
                                U.id = "inffake";
                                document.body.appendChild(U);
                                U.webkitRequestFullscreen();
                                var T = document.createEvent("MouseEvents");
                                T.initMouseEvent("click", target ? true : false, true, window, 0, 0, 0, 0, 0, false, false, true, false, 0, null);
                                MgPop.currentBlock.fakeLink.dispatchEvent(T);
                                document.webkitCancelFullScreen();
                                setTimeout(function() {
                                    window.getSelection().empty()
                                }, 250)
                            }
                        } catch (T) {}
                    }
                    function K() {
                        var T = top.window.open("about:blank");
                        T.focus();
                        T.close()
                    }
                    function J() {
                        var V = "";
                        var U = top.window.document.createElement("a");
                        U.href = "data:text/html,<scr" + V + "ipt>window.close();</scr" + V + "ipt>";
                        document.getElementsByTagName("body")[0].appendChild(U);
                        var T = top.window.document.createEvent("MouseEvents");
                        T.initMouseEvent("click", false, true, window, 0, 0, 0, 0, 0, true, false, false, true, 0, null);
                        U.dispatchEvent(T);
                        U.parentNode.removeChild(U)
                    }
                    if (p.PopLoaded || p.IsCapped("InfNumFastPops", "InfNumFastPopsExpire", p.PopSettings.Cap) ||!F() ||!G() ||!i() ||!E() ||!H() ||!I()) {
                        return 
                    } else {
                        var N = (a.GetWindowLeft() + (a.GetWindowWidth() / 2) - (1024 / 2));
                        var O = (a.GetWindowTop() + (a.GetWindowHeight() / 2) - (768 / 2));
                        g(R, Q, 1024, 768, N, O)
                    }
                };
                var p = g367CB268B1094004A3689751E7AC568F;
                var r = p.InfinityHost;
                var D = window.location;
                var u;
                var B = encodeURIComponent(new Date().getTimezoneOffset());
                var v = g367CB268B1094004A3689751E7AC568F.b32encode(encodeURIComponent(window.document.referrer));
                var t = encodeURIComponent(Math.floor(Math.random() * 100000 + 1));
                var z = (this.SubID.length > 0 ? "&subId=" + this.SubID : "null");
                u = r + "/fp.engine?id=" + p.Guid + "&rand=" + t + "&ver=" + this.Version + "&time=" + B + "&referrerUrl=" + v + "&subId=" + (this.SubID.length > 0 ? "&subId=" + this.SubID : "");
                s(u)
            }
            if (!true && true) {
                y = g367CB268B1094004A3689751E7AC568F.SubID;
                var A = function() {
                    g367CB268B1094004A3689751E7AC568F.Script.Load(g367CB268B1094004A3689751E7AC568F.InfinityHost + "/sitetestclickcount.engine?id=" + g367CB268B1094004A3689751E7AC568F.Guid + "&rand=" + encodeURIComponent(Math.random()) + (y.length > 0 ? "&subId=" + y : ""), true)
                };
                a.BindOnDocmentClick(A)
            }
            function x() {
                if (a.PlaceLotamePixel || a.RequestTrackingScripts) {
                    var d = document.createElement("script");
                    d.setAttribute("src", a.RequestTrackingScripts ? a.CDNHost + "/Scripts/ltm?id=" + a.Guid : "//tags.crwdcntrl.net/c/" + a.LotameTrackingId + "/cc_af.js");
                    d.async = true;
                    var e = document.getElementsByTagName("script")[0];
                    if (e) {
                        e.parentNode.insertBefore(d, e)
                    }
                }
            }
            if (a.PlaceComScorePixel || a.RequestTrackingScripts) {
                var b = document.createElement("script");
                b.setAttribute("src", a.RequestTrackingScripts ? a.CDNHost + "/Scripts/cms?id=" + a.Guid : (document.location.protocol == "https:" ? "https://sb" : "http://b") + ".scorecardresearch.com/beacon.js?c1=" + a.ComScoreTagType + "&c2=" + a.ComScoreClientId + "&c3=" + a.ComScoreChannelId);
                b.async = true;
                var c = document.getElementsByTagName("script")[0];
                if (c) {
                    c.parentNode.insertBefore(b, c)
                }
            }
            var l = 123;
            var C = new Date().getTimezoneOffset();
            var v = window.document.referrer;
            if (g367CB268B1094004A3689751E7AC568F.IsAdblockRequest) {
                var B = encodeURIComponent(C);
                var t = encodeURIComponent(Math.floor(Math.random() * 100000 + 1));
                var j = g367CB268B1094004A3689751E7AC568F.b32encode(encodeURIComponent(v));
                var z = (this.SubID.length > 0 ? this.SubID : "null");
                this.Script.Add(this.InfinityHost + "/" + (B ? B : "null") + "/" + (this.Guid ? this.Guid : "null") + "/" + (t ? t : "null") + "/" + (this.Version ? this.Version : "null") + "/null/" + (l ? l : "null") + "/" + z + "/Tag.engine", true)
            } else {
                this.Script.Add(this.InfinityHost + "/Tag.engine?time=" + encodeURIComponent(C) + "&id=" + this.Guid + "&rand=" + encodeURIComponent(Math.floor(Math.random() * 100000 + 1)) + "&ver=" + this.Version + "&referrerUrl=" + encodeURIComponent(v) + "&fingerPrint=" + l + (this.SubID.length > 0 ? "&subId=" + this.SubID : ""), true)
            }
            x();
            this.Script.Load(function() {
                if (typeof a.Media.Items == "undefined" || a.Media.Items === undefined || a.Media.Items === null) {
                    return 
                }
                if (a.IsClientSideFiltersPassed()) {
                    for (var d = 0; d < a.Media.Items.length; d++) {
                        a.Script.Add(a.CDNHost + "/Scripts/MediaScripts/" + a.Media.Items[d].ScriptUri, true)
                    }
                    a.Script.Load(function() {
                        var g = "";
                        for (var e = 0; e < a.Media.Items.length; e++) {
                            g = a.Media.Items[e].MediaType;
                            if (typeof a.Modules[g] != "undefined") {
                                a.Modules[g](a.Media.Items[e], a._Top)
                            }
                        }
                    })
                }
                g367CB268B1094004A3689751E7AC568F.Lotame.PlaceMappingPixel()
            });
            if (typeof(InfCustomerCallback) === "function") {
                InfCustomerCallback()
            }
        },
        IsClientSideFiltersPassed: function() {
            if (this.Media.ClientSideFilters == null || this.Media.ClientSideFilters.length < 1) {
                return true
            }
            for (var c in this.Media.ClientSideFilters) {
                var a = this.Media.ClientSideFilters[c];
                var b = false;
                if (a.Type == "PublisherSiteFrequencyCap") {
                    b = this.IsPublisherSiteFrequencyCapPassed(a)
                }
                if (!b) {
                    return false
                }
            }
            return true
        },
        IsPublisherSiteFrequencyCapPassed: function(a) {
            var b=!this.IsCapped("InfNumPops", "InfNumPopsExpire", a.Cap, true);
            this.IncrementCap("InfNumPops", "InfNumPopsExpire", a.CapLengthInSeconds, true);
            return b
        },
        FetchMediaUrl: function(f, d) {
            var e = g367CB268B1094004A3689751E7AC568F.InfinityHost;
            if (f.Redirect == 1) {
                e = this.InfinityHost.replace("http://", "https://")
            }
            var i = this._Top.location;
            var h;
            if (g367CB268B1094004A3689751E7AC568F.IsAdblockRequest) {
                var c = g367CB268B1094004A3689751E7AC568F.b32encode(encodeURIComponent(i));
                var b = "b32_" + g367CB268B1094004A3689751E7AC568F.b32encode(f.Country);
                h = "/" + (f.PerformanceTest ? f.PerformanceTest : "null") + "/" + (f.MediaId ? f.MediaId : "null") + "/" + (f.PlacementId ? f.PlacementId : "null") + "/" + (f.MediaSegmentId ? f.MediaSegmentId : "null") + "/" + (f.PoolId ? f.PoolId : "null") + "/" + (f.SiteId ? f.SiteId : "null") + "/" + (f.ZoneId ? f.ZoneId : "null") + "/" + (b ? b : "null") + "/" + (f.Bid ? f.Bid : "null") + "/" + (f.MaxBid ? f.MaxBid : "null") + "/" + (this.SubID.length > 0 ? this.SubID : "null") + "/null/Redirect.eng"
            } else {
                h = "/Redirect.engine?PerformanceTest=" + f.PerformanceTest + "&MediaId=" + f.MediaId + "&PId=" + f.PlacementId + "&MediaSegmentId=" + f.MediaSegmentId + "&PoolId=" + f.PoolId + "&SiteId=" + f.SiteId + "&ZoneId=" + f.ZoneId + "&Country=" + f.Country + "&Bid=" + f.Bid + "&MaxBid=" + f.MaxBid + (this.SubID.length > 0 ? "&subId=" + this.SubID : "") + "&currentUrl=" + encodeURIComponent(i)
            }
            if (d === true && f.Settings.Width < 1024) {
                var a = "/Bridge/Index?width=" + f.Settings.Width + "&height=" + f.Settings.Height + "&url=" + encodeURIComponent(h);
                return g367CB268B1094004A3689751E7AC568F.DFPImpressionUrl + e + a
            }
            if (f.MediaType === "Link" && f.Settings.PassParams === true && g367CB268B1094004A3689751E7AC568F.PassParams.length > 0) {
                var g = h + "&" + g367CB268B1094004A3689751E7AC568F.PassParams;
                if (g.length <= 2048) {
                    h = g
                }
                h = g367CB268B1094004A3689751E7AC568F.DFPImpressionUrl + h
            }
            g367CB268B1094004A3689751E7AC568F.Impressions.Record(f);
            if (g367CB268B1094004A3689751E7AC568F.EncodeUrl !== undefined && g367CB268B1094004A3689751E7AC568F.EncodeUrl) {
                e = e.replace("https://", "").replace("http://", "").replace("//", "");
                return g367CB268B1094004A3689751E7AC568F.DFPImpressionUrl + "//" + window.getUri(e + h)
            }
            return g367CB268B1094004A3689751E7AC568F.DFPImpressionUrl + e + h
        },
        BindOnDocumentTouch: function(b) {
            var c = self;
            if (top != self) {
                try {
                    if (top.document.location.toString()) {
                        c = top
                    }
                } catch (a) {}
            }
            if (c.document.addEventListener) {
                c.document.addEventListener("touchstart", b, false)
            } else {
                c.document.attachEvent("touchstart", b)
            }
        },
        BindOnDocmentClick: function(b) {
            var c = self;
            if (top != self) {
                try {
                    if (top.document.location.toString()) {
                        c = top
                    }
                } catch (a) {}
            }
            if (c.document.addEventListener) {
                c.document.addEventListener("click", b, false)
            } else {
                c.document.attachEvent("onclick", b)
            }
        },
        GetExpiration: function(b) {
            var c = (new Date()).valueOf() + b * 1000;
            var a = new Date();
            a.setTime(c);
            return a
        },
        IsExpired: function(b) {
            var a = new Date();
            var c = a.valueOf() >= Date.parse(b);
            return c
        },
        IsCapped: function(d, e, a, f) {
            f=!!f;
            var b = f ? this.Storage.GetSessionStorage(d): this.Storage.GetLocalStorage(d);
            var c = f ? this.Storage.GetSessionStorage(e): this.Storage.GetLocalStorage(e);
            if (b === undefined || b === 0 || c === undefined || this.IsExpired(c) || isNaN(b)) {
                return false
            }
            return a !== undefined && b >= a
        },
        IncrementCap: function(d, e, c, f) {
            f=!!f;
            var a = f ? this.Storage.GetSessionStorage(d): this.Storage.GetLocalStorage(d);
            var b = f ? this.Storage.GetSessionStorage(e): this.Storage.GetLocalStorage(e);
            if (a === undefined || a === 0 || b === undefined || this.IsExpired(b) || isNaN(a)) {
                a = 0;
                b = this.GetExpiration(c)
            }
            f ? this.Storage.SetSessionStorage(d, ++a) : this.Storage.SetLocalStorage(d, ++a);
            f ? this.Storage.SetSessionStorage(e, b) : this.Storage.SetLocalStorage(e, b)
        },
        GetWindowHeight: function() {
            var a = 0;
            if (typeof(this._Top.window.innerHeight) == "number") {
                a = this._Top.window.innerHeight
            } else {
                if (this._Top.document.documentElement && this._Top.document.documentElement.clientHeight) {
                    a = this._Top.document.documentElement.clientHeight
                } else {
                    if (this._Top.document.body && this._Top.document.body.clientHeight) {
                        a = this._Top.document.body.clientHeight
                    }
                }
            }
            return a
        },
        GetWindowWidth: function() {
            var a = 0;
            if (typeof(this._Top.window.innerWidth) == "number") {
                a = this._Top.window.innerWidth
            } else {
                if (this._Top.document.documentElement && this._Top.document.documentElement.clientWidth) {
                    a = this._Top.document.documentElement.clientWidth
                } else {
                    if (this._Top.document.body && this._Top.document.body.clientWidth) {
                        a = this._Top.document.body.clientWidth
                    }
                }
            }
            return a
        },
        GetWindowTop: function() {
            return (this._Top.window.screenTop !== undefined) ? this._Top.window.screenTop : this._Top.window.screenY
        },
        GetWindowLeft: function() {
            return (this._Top.window.screenLeft !== undefined) ? this._Top.window.screenLeft : this._Top.window.screenX
        },
        GetParentLink: function(c) {
            var a = c;
            if (a.getAttribute("target") == null && a.getAttribute("rel") == null && a.nodeName.toLowerCase() != "html") {
                var b = 0;
                while (a.parentNode && b <= 4 && a.nodeName.toLowerCase() != "html") {
                    b++;
                    a = a.parentNode;
                    if (a.nodeName.toLowerCase() === "a" && a.href !== "") {
                        break
                    }
                }
            }
            return a
        },
        Querystring: {
            Init: function() {
                var f;
                var b = /\+/g;
                var h = /([^&=]+)=?([^&]*)/g;
                var c = function(a) {
                    return decodeURIComponent(a.replace(b, " "))
                };
                var g = window.location.search.substring(1);
                while (f = h.exec(g)) {
                    this.Params[c(f[1])] = c(f[2])
                }
            },
            Params: []
        },
        Script: {
            Items: [],
            Add: function(b, a) {
                if (!this.Exists(b)) {
                    this.Items.push({
                        url: b,
                        cache: a,
                        complete: false
                    });
                    return true
                }
                return false
            },
            Exists: function(b) {
                for (var a = 0; a < this.Items.length; a++) {
                    if (this.Items[a].url == b) {
                        return true
                    }
                }
                return false
            },
            Complete: function(b) {
                for (var a = 0; a < b.length; a++) {
                    if (!b[a].complete) {
                        return false
                    }
                }
                return true
            },
            Create: function(d, a) {
                var c = document.createElement("script");
                c.setAttribute("type", "text/javascript");
                if (a) {
                    c.src = d
                } else {
                    var b = Math.floor(89999999 * Math.random() + 10000000);
                    c.src = (d.indexOf("?")>-1) ? d + "&" + b : d + "?" + b
                }
                return c
            },
            AppendToDom: function(e, d, b) {
                var a = this;
                var c = document.getElementsByTagName("head")[0];
                if (g367CB268B1094004A3689751E7AC568F.UseAjax !== undefined && g367CB268B1094004A3689751E7AC568F.UseAjax) {
                    g367CB268B1094004A3689751E7AC568F.AjaxLoad(e[d].url, function(j) {
                        var i = document.createElement("script");
                        i.type = "text/javascript";
                        i.innerHTML = j.responseText;
                        var h = document.getElementsByTagName("script")[0];
                        h.parentNode.insertBefore(i, h);
                        e[d].complete = true;
                        if (a.Complete(e) && typeof b == "function") {
                            b()
                        }
                    })
                } else {
                    var g = e[d].url;
                    if (g367CB268B1094004A3689751E7AC568F.EncodeUrl !== undefined && g367CB268B1094004A3689751E7AC568F.EncodeUrl) {
                        g = "//" + window.getUri(g)
                    }
                    var f = this.Create(g, e[d].cache);
                    f.onload = f.onreadystatechange = function(h) {
                        if (!e[d].complete && (!this.readyState || this.readyState == "loaded" || this.readyState == "complete")) {
                            e[d].complete = true;
                            f.onload = f.onreadystatechange = null;
                            if (a.Complete(e) && typeof b == "function") {
                                b()
                            }
                        }
                    };
                    c.appendChild(f)
                }
            },
            Load: function(a, b, c) {
                if (arguments.length === 0 || arguments.length === 1) {
                    var e = this.Items;
                    this.Items = [];
                    for (var d = 0; d < e.length; d++) {
                        this.AppendToDom(e, d, a)
                    }
                } else {
                    this.AppendToDom([{
                        url: a,
                        cache: b,
                        complete: false
                    }
                    ], 0, c)
                }
            }
        },
        Lotame: {
            PlaceMappingPixel: function() {
                if (g367CB268B1094004A3689751E7AC568F.MappingPixel.UserId && g367CB268B1094004A3689751E7AC568F.MappingPixel.PlaceMappingPixel) {
                    var b = document.createElement("img");
                    b.setAttribute("width", "1");
                    b.setAttribute("height", "1");
                    b.setAttribute("style", "display: block;");
                    b.setAttribute("src", "//bcp.crwdcntrl.net/map/c=" + g367CB268B1094004A3689751E7AC568F.LotameClientId + "/tp=ADSP/tpid=" + g367CB268B1094004A3689751E7AC568F.MappingPixel.UserId.replace(/-/g, ""));
                    var a = document.body;
                    if (a) {
                        a.insertBefore(b, a.firstChild)
                    }
                }
            }
        },
        Impressions: {
            Record: function(c) {
                if (g367CB268B1094004A3689751E7AC568F.PlaceLotameImpressionPixel && g367CB268B1094004A3689751E7AC568F.ImpressionPixel.PlaceImpressionTrackingPixel) {
                    var b = document.createElement("img");
                    b.setAttribute("width", "1");
                    b.setAttribute("height", "1");
                    b.setAttribute("style", "display: block;");
                    b.setAttribute("src", "//bcp.crwdcntrl.net/5/c=" + g367CB268B1094004A3689751E7AC568F.LotameClientId + "/ctax=Campaigns^Expose^Viewers^Placement " + c.PlacementId + " - Viewer");
                    var a = document.body;
                    if (a) {
                        a.insertBefore(b, a.firstChild)
                    }
                }
            },
        },
        Storage: {
            GetLocalStorage: function(a) {
                var c;
                if (this.SupportsLocalStorage()) {
                    c = localStorage[a]
                } else {
                    c = this.GetCookie(a)
                }
                if (c) {
                    var b = c.split("__")[0];
                    if (b == this.GetCookie("ISSH") || b == "undefined") {
                        c = c.split("__")[1]
                    }
                }
                return c
            },
            SetLocalStorage: function(b, d) {
                var c = this.GetCookie("ISSH") + "__" + d;
                if (this.SupportsLocalStorage()) {
                    try {
                        localStorage[b] = c
                    } catch (a) {
                        this.FailedLocalStorage = true;
                        return 
                    }
                } else {
                    this.SetCookie(b, c)
                }
            },
            SupportsLocalStorage: function() {
                try {
                    sessionStorage.setItem("storageSupport", 1);
                    sessionStorage.removeItem("storageSupport");
                    return "localStorage" in window && window.localStorage !== null
                } catch (a) {
                    if (a.code === DOMException.QUOTA_EXCEEDED_ERR && sessionStorage.length === 0) {
                        window.console = window.console || {
                            log: function(b) {}
                        };
                        console.log("Safari private mode detected")
                    }
                    return false
                }
            },
            GetSessionStorage: function(a) {
                var c;
                if (this.SupportsSessionStorage()) {
                    c = sessionStorage[a]
                } else {
                    c = this.GetCookie(a)
                }
                if (c) {
                    var b = c.split("__")[0];
                    if (b == this.GetCookie("ISSH") || b == "undefined") {
                        c = c.split("__")[1]
                    }
                }
                return c
            },
            SetSessionStorage: function(b, d) {
                var c = this.GetCookie("ISSH") + "__" + d;
                if (this.SupportsSessionStorage()) {
                    try {
                        sessionStorage[b] = c
                    } catch (a) {
                        this.FailedSessionStorage = true
                    }
                } else {
                    this.SetCookie(b, c)
                }
            },
            SupportsSessionStorage: function() {
                try {
                    sessionStorage.setItem("storageSupport", 1);
                    sessionStorage.removeItem("storageSupport");
                    return "sessionStorage" in window && window.sessionStorage !== null
                } catch (a) {
                    if (a.code === DOMException.QUOTA_EXCEEDED_ERR && sessionStorage.length === 0) {
                        window.console = window.console || {
                            log: function(b) {}
                        };
                        console.log("Safari private mode detected")
                    }
                    return false
                }
            },
            SetCookie: function(a, e, d) {
                var c = new Date();
                c.setSeconds(c.getSeconds() + d);
                var b = escape(e) + ((d == null) ? "" : "; expires=" + c.toUTCString());
                document.cookie = a + "=" + b
            },
            GetCookie: function(b) {
                var c, d, e, a = document.cookie.split(";");
                for (c = 0; c < a.length; c++) {
                    d = a[c].substr(0, a[c].indexOf("="));
                    e = a[c].substr(a[c].indexOf("=") + 1);
                    d = d.replace(/^\s+|\s+$/g, "");
                    if (d == b) {
                        return unescape(e)
                    }
                }
            },
            FailedLocalStorage: false,
            FailedSessionStorage: false
        },
        IsMobile: {
            Android: function(a) {
                return a.navigator.userAgent.match(/Android/i)
            },
            BlackBerry: function(a) {
                return a.navigator.userAgent.match(/BlackBerry/i)
            },
            iOS: function(a) {
                return a.navigator.userAgent.match(/iPhone|iPad|iPod/i)
            },
            Opera: function(a) {
                return a.navigator.userAgent.match(/Opera Mini/i)
            },
            Windows: function(a) {
                return a.navigator.userAgent.match(/IEMobile/i)
            },
            any: function(a) {
                return a.navigator.userAgent.match(/Android|BlackBerry|iPhone|iPad|iPod|Opera Mini|IEMobile/i)
            }
        },
        b32encode: function(g) {
            var a = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
            var d = [];
            var e = Math.floor((g.length / 5));
            var c = g.length%5;
            if (c != 0) {
                for (var b = 0; b < (5 - c); b++) {
                    g += "\x00"
                }
                e += 1
            }
            for (b = 0; b < e; b++) {
                d.push(a.charAt(g.charCodeAt(b * 5)>>3));
                d.push(a.charAt(((g.charCodeAt(b * 5) & 7)<<2) | (g.charCodeAt(b * 5 + 1)>>6)));
                d.push(a.charAt(((g.charCodeAt(b * 5 + 1) & 63)>>1)));
                d.push(a.charAt(((g.charCodeAt(b * 5 + 1) & 1)<<4) | (g.charCodeAt(b * 5 + 2)>>4)));
                d.push(a.charAt(((g.charCodeAt(b * 5 + 2) & 15)<<1) | (g.charCodeAt(b * 5 + 3)>>7)));
                d.push(a.charAt(((g.charCodeAt(b * 5 + 3) & 127)>>2)));
                d.push(a.charAt(((g.charCodeAt(b * 5 + 3) & 3)<<3) | (g.charCodeAt(b * 5 + 4)>>5)));
                d.push(a.charAt(((g.charCodeAt(b * 5 + 4) & 31))))
            }
            var f = 0;
            if (c == 1) {
                f = 6
            } else {
                if (c == 2) {
                    f = 4
                } else {
                    if (c == 3) {
                        f = 3
                    } else {
                        if (c == 4) {
                            f = 1
                        }
                    }
                }
            }
            for (b = 0; b < f; b++) {
                d.pop()
            }
            return d.join("")
        },
        AjaxLoad: function(g, a) {
            var j;
            if (typeof XMLHttpRequest !== "undefined") {
                j = new XMLHttpRequest()
            } else {
                var h = ["MSXML2.XmlHttp.5.0", "MSXML2.XmlHttp.4.0", "MSXML2.XmlHttp.3.0", "MSXML2.XmlHttp.2.0", "Microsoft.XmlHttp"];
                for (var d = 0, f = h.length; d < f; d++) {
                    try {
                        j = new ActiveXObject(h[d]);
                        break
                    } catch (b) {}
                }
            }
            var c = function() {
                if (j.readyState < 4) {
                    return 
                }
                if (j.status !== 200) {
                    return 
                }
                if (j.readyState === 4) {
                    a(j)
                }
            };
            j.onreadystatechange = c;
            j.open("GET", g, true);
            j.send()
        }
    };
    (function(a) {
        a.Fingerprint = 123
    })(g367CB268B1094004A3689751E7AC568F);
    g367CB268B1094004A3689751E7AC568F.Init()
};
