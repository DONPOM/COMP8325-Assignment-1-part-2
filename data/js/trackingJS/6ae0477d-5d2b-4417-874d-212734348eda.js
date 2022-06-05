var SnapABug = function() {
    var f = ("https:" == document.location.protocol ? "https" : "http") + "://www.snapengage.com", k = ("https:" == document.location.protocol ? "https" : "http") + "://commondatastorage.googleapis.com/code.snapengage.com", u=!0, g=!0, r=!1, d = "", n = "", x = "", q = "", v = "", B = "", A = "left", C = "55%", z = 35, w = 73, D = "", W=!1, l = "", O = "", p = {}, M = "en", J = "-4px", H = 0, F = "", $=!0, X =- 1, K = 0, Y = "", P = "", L = "", Z = "white", ha = "", Q = "", G = "", R = 0, a = 0, c = 1, m = 1, e = 0, Ha=!0, Ia = k + "/wbg/jimdo_form.png", Ja=!1, ra = null, aa = null, Ka=!1, La=!1, Ma = "", Aa = 0, Na=!0,
    V, Ba = 0, S = "", sa=!1, T = null, Ca=!1, ba = null, Da = 2, ia = "", ja = "", Oa = 30, Pa = 0, Qa=!0, Ra=!0, ka=!0, Sa=!0, Ta=!0, ca = "", Ua=!0, ta=!1, ua = "", Va = null, Wa = null, Xa = null, Ya = null, Za = null, $a = null, ab = null, bb = null, cb = null, da = null, db = null, Ea=!1, eb=!0, fb = "", va = "", la=!1, ma = 0, Fa = 110, wa = null, I = null, E = null, gb = 0, hb = 0, na = "", ib = "", jb = "", kb = "", lb=!1, xa=!1, ea = 0, oa = {}, ya = [], fa =- 1, ga=!1, za = null, pa = null, mb =- 1, N = "", Ga = "", nb=!1;
    return {
        setLocale: function(b) {
            b && (M = b, 6 < b.length && (M = b.substr(0, 5)))
        },
        startLink: function() {
            SnapABug.startPreChat();
            return !1
        },
        startChat: function(b) {
            if (la)
                SnapABug.populateFormStatus( - 2);
            else 
                return this.preventProactiveChat(), this.hideButton(), r || this.handleChatInProgress() || (ta=!0, S = b, this.sendGetWidgetConfig(!1)), !1
        },
        startCall: function() {
            if (la)
                SnapABug.populateFormStatus( - 2);
            else 
                return this.preventProactiveChat(), r || (this.showLoading(), this.sendGetCallConfig()), !1
        },
        setUserEmail: function(b, t) {
            if (b && "invalid@example.com" != b)
                try {
                    D = b.replace(/^[\s\+]+|[\s\+]+$/g, "")
            } catch (a) {}
            !0 == t && (W=!0)
        },
        setDescription: function(b) {
            l =
            b
        },
        setButtonEffect: function(b) {
            J = b;
            this.buttonOut()
        },
        setFormTopPosition: function(b) {
            H = b
        },
        setChatFormPosition: function(b) {
            F = b
        },
        getChatFormPosition: function() {
            return F
        },
        setDomain: function(b) {
            "" == d&&-1 != document.location.href.indexOf(b) && (Y = b)
        },
        setSecureConnection: function() {
            f = f.replace(/http\:/, "https:")
        },
        setStartChat: function(b) {
            ca = b
        },
        allowOffline: function(b) {
            (Qa = b) || this.showScreenshotOption(!1)
        },
        allowChat: function(b) {
            (ka = b) || (e=!1)
        },
        allowScreenshot: function(b) {
            Ra = b
        },
        allowProactiveChat: function(b) {
            Sa =
            b
        },
        showScreenshotOption: function(b) {
            Ta = b
        },
        setWidgetId: function(b) {
            d = b
        },
        allowChatSound: function(b) {
            Ha=!0 == b?!0 : !1
        },
        allowPageChangeNotification: function(b) {
            Ua = b
        },
        allowChatHistory: function(b) {
            eb = b
        },
        clearAllCookies: function() {
            SnapABugChat.deleteChatCookies();
            SnapABugChat.setCookie("SnapABugRef", "", 0);
            SnapABugChat.deleteCookie("SnapABugRef");
            SnapABugChat.setCookie("SnapABugNoProactiveChat", "", 0);
            SnapABugChat.deleteCookie("SnapABugNoProactiveChat");
            SnapABugChat.setCookie("SnapABugUserEmail", "", 0);
            SnapABugChat.deleteCookie("SnapABugUserEmail");
            SnapABugChat.setCookie("SnapABugVisit", "", 0);
            SnapABugChat.deleteCookie("SnapABugVisit");
            SnapABugChat.setCookie("SnapABugHistory", "", 0);
            SnapABugChat.deleteCookie("SnapABugHistory");
            D = ""
        },
        sendTextToChat: function(b) {
            return r ? (SnapABugChat.sendChatMessage(6, b), !0) : !1
        },
        openProactiveChat: function(b, t, a) {
            xa ? SnapABug.openProactiveV2APICall(b, t, a) : SnapABug.openProactiveChatDo(b, t, a)
        },
        setNoProactiveChatDelay: function(b) {
            Oa = b
        },
        setVisitorCookieLifeTime: function(b) {
            Pa = b
        },
        getVisitorCookieLifeTime: function() {
            return Pa
        },
        setProactiveAutocloseDelay: function(b) {
            Da = b
        },
        setCallback: function(b, t) {
            switch (b.toLowerCase()) {
            case "open":
                Va = t;
                break;
            case "openproactive":
                Wa = t;
                break;
            case "close":
                Xa = t;
                break;
            case "messagesubmit":
                Ya = t;
                break;
            case "startchat":
                Za = t;
                break;
            case "chatmessagesent":
                $a = t;
                break;
            case "chatmessagereceived":
                ab = t;
                break;
            case "beforecallme":
                bb = t;
                break;
            case "startcallme":
                cb = t;
                break;
            case "agentonline":
                null != t ? (da || (da = []), da.push(t)) : da = null;
                break;
            case "agentcommand":
                db = t
            }
        },
        getAgentStatusAsync: function(b) {
            this.setCallback("agentOnline",
            b);
            - 1 != ma && (b = (new Date).getTime(), 0 == ma || ma + 1E3 * Fa < b ? (ma =- 1, this.sendGetWidgetConfig(!1)) : (this.callCallback("agentOnline", SnapABug.chatAvailable()?!0 : !1), this.setCallback("agentOnline", null)))
        },
        trackConversion: function(b) {
            var t = "ci=" + encodeURIComponent(n);
            "" == n && (t = "c=" + encodeURIComponent(x));
            t += "&a=" + encodeURIComponent(b);
            this.rpc(f + "/snapabug/ServiceTrackConversion?" + t, 3, !0)
        },
        hideButton: function() {
            var b = document.getElementById("SnapABug_bImg");
            b && (b.style.display = "none")
        },
        showButton: function() {
            var b =
            document.getElementById("SnapABug_bImg");
            b && (b.style.display = "inline")
        },
        disableLightbox: function() {
            Z = ""
        },
        setSecureConnexion: function() {
            SnapABug.setSecureConnection()
        },
        chatAvailable: function() {
            return e
        },
        setLanguage: function(b) {
            var t = "en";
            - 1 != b.indexOf("Dansk") ? t = "da" : - 1 != b.indexOf("Deutsch") ? t = "de" : - 1 != b.indexOf("Espa") ? t = "es" : - 1 != b.indexOf("Italiano") ? t = "it" : - 1 != b.indexOf("Suomi") ? t = "fi" : - 1 != b.indexOf("Fran") ? t = "es" : - 1 != b.indexOf("Hungarian") ? t = "hu" : - 1 != b.indexOf("Italiano") ? t = "it" : - 1 != b.indexOf("Hebrew") ?
            t = "iw" : - 1 != b.indexOf("Japanese") ? t = "ja" : - 1 != b.indexOf("Norsk") ? t = "nb" : - 1 != b.indexOf("Nederlands") ? t = "nl" : - 1 != b.indexOf("Polish") ? t = "pl" : - 1 != b.indexOf("Portugu") ? t = "pt" : - 1 != b.indexOf("Romana") ? t = "ro" : - 1 != b.indexOf("Slovak") ? t = "sk" : - 1 != b.indexOf("Svenska") ? t = "sv" : - 1 != b.indexOf("Turkish") ? t = "tr" : - 1 != b.indexOf("Simplified Chinese") && (t = "zh");
            this.setLocale(t)
        },
        init: function(b) {
            this.initWidget(b, !1)
        },
        initAsync: function(b) {
            this.initWidget(b, !0)
        },
        addButton: function(b, t, a, c) {
            if ("undefined" != typeof t && "undefined" !=
            typeof a && (!SnapABug.isMobile() || "undefined" == typeof SNAPENGAGE_MOBILE)) {
                switch (t) {
                case "0":
                    A = "left";
                    break;
                case "1":
                    A = "right";
                    break;
                case "2":
                    A = "top";
                    break;
                case "3":
                    A = "bottom";
                    break;
                default:
                    A = "left"
                }
                C = a
            }
            this.injectButton(b, c);
            this.init(b)
        },
        add: function(b, t, a, c, U, e) {
            this.setLocale(t);
            if ("undefined" != typeof a && "undefined" != typeof c && (!SnapABug.isMobile() || "undefined" == typeof SNAPENGAGE_MOBILE)) {
                switch (a) {
                case "0":
                    A = "left";
                    break;
                case "1":
                    A = "right";
                    break;
                case "2":
                    A = "top";
                    break;
                case "3":
                    A = "bottom";
                    break;
                default:
                    A = "left"
                }
                C = c;
                this.injectButton(b, !1, !0, U, e)
            }
            this.initWidget(b, !0)
        },
        setButton: function(b, t, a) {
            B = b;
            z = t;
            w = a;
            J = "0px"
        },
        rpc: function(b, t, a) {
            Ma = 2E3 >= b.length ? b : b.substring(0, 2E3);
            Aa = t;
            Na = a;
            this.doRpc()
        },
        doRpc: function() {
            YAHOO.util.Get.script(Ma, {
                onFailure: function() {
                    SnapABug.callbackError(!0)
                },
                onTimeout: function() {
                    SnapABug.callbackError(!0)
                },
                timeout: 3E4
            })
        },
        callbackError: function(b) {
            b && 0 < Aa ? (rcpRetries = Aa - 1, this.doRpc()) : Na || this.populateFormStatus(6)
        },
        getWindowWidth: function() {
            var b = 0;
            window.innerWidth ?
            b = window.innerWidth : self.innerWidth ? b = self.innerWidth : document.documentElement && document.documentElement.clientWidth ? b = document.documentElement.clientWidth : document.body && (b = document.body.clientWidth);
            return b
        },
        getWindowHeight: function() {
            var b = 0;
            window.innerHeight ? b = window.innerHeight : self.innerHeight ? b = self.innerHeight : document.documentElement && document.documentElement.clientHeight ? b = document.documentElement.clientHeight : document.body && (b = document.body.clientHeight);
            return b
        },
        getOSBarHeight: function() {
            try {
                if ( - 1 !=
                navigator.appVersion.indexOf("Win"))
                    return 28;
                if ( - 1 != navigator.appVersion.indexOf("Mac"))
                    return 75
            } catch (b) {}
            return 0
        },
        isIE6: function() {
            try {
                return - 1 != navigator.userAgent.toLowerCase().indexOf("msie 6")
            } catch (b) {
                return !1
            }
        },
        isIE7: function() {
            try {
                return - 1 != navigator.userAgent.toLowerCase().indexOf("msie 7")
            } catch (b) {
                return !1
            }
        },
        isIE: function() {
            try {
                return - 1 != navigator.appVersion.indexOf("MSIE")?!0 : !1
            } catch (b) {
                return !1
            }
        },
        isIE11: function() {
            try {
                return - 1 != navigator.appVersion.indexOf("Trident")?!0 : !1
            } catch (b) {
                return !1
            }
        },
        isFirefox: function() {
            try {
                return - 1 != navigator.userAgent.indexOf("Firefox")?!0 : !1
            } catch (b) {
                return !1
            }
        },
        isMobile: function() {
            try {
                return navigator.userAgent.match(/(iPhone|iPod|iPad|BlackBerry|Android|IEMobile)/)
            } catch (b) {
                return !1
            }
        },
        getMobileUI: function() {
            return wa
        },
        isUSCanada: function() {
            return !0
        },
        setCursorPosition: function(b, t) {
            try {
                if (b) {
                    if (b.createTextRange) {
                        var a = b.createTextRange();
                        a.collapse(!0);
                        a.moveEnd(t);
                        a.moveStart(t);
                        a.select();
                        return !0
                    }
                    if (b.setSelectionRange)
                        return b.setSelectionRange(t, t),
                        !0
                } else 
                    return !1
            } catch (c) {}
        },
        getBaseURL: function() {
            return f
        },
        getStaticBaseURL: function() {
            return k
        },
        getUserEmail: function() {
            return D
        },
        getCaseId: function() {
            return x
        },
        getDbCaseId: function() {
            return n
        },
        getWidgetId: function() {
            return d
        },
        getCallMeButton: function() {
            return ia
        },
        setCallMePhone: function(b) {
            ja = b
        },
        getCallMePhone: function() {
            return ja
        },
        getEntryPageRef: function() {
            var b = decodeURIComponent(SnapABugChat.getCookie("SnapABugRef"));
            try {
                if (b && "" != b)
                    return b.split(" ")[1]
            } catch (a) {}
            return ""
        },
        getEntryPageUrl: function() {
            var b =
            decodeURIComponent(SnapABugChat.getCookie("SnapABugRef"));
            try {
                if (b && "" != b)
                    return b.split(" ")[0]
            } catch (a) {}
            return ""
        },
        getChatSoundAllowed: function() {
            return Ha
        },
        setEnableInternationalPhoneCalls: function(b) {
            SnapABug.enableInternationalPhoneCalls = b
        },
        getEnableInternationalPhoneCalls: function() {
            return SnapABug.enableInternationalPhoneCalls
        },
        preventProactiveChat: function() {
            T && (clearTimeout(T), S = "", sa=!1, T = null);
            SnapABugChat && (SnapABugChat.setNoProactiveChatCookie(), SnapABugChat.clearProactive())
        },
        getNoProactiveChatDelay: function() {
            return Oa
        },
        setCvval: function(b, a) {
            p[b] = a
        },
        getLocale: function() {
            return M
        },
        getDirection: function() {
            return ua
        },
        setDirection: function(b) {
            ua = b
        },
        callCallback: function(b, a, c, y) {
            try {
                switch (b.toLowerCase()) {
                case "open":
                    return Va(a);
                case "openproactive":
                    return Wa(a, c);
                case "close":
                    return Xa(a, c);
                case "messagesubmit":
                    return Ya(a, c);
                case "startchat":
                    return Za(a, c, y);
                case "chatmessagesent":
                    return $a(a);
                case "chatmessagereceived":
                    return ab(a, c);
                case "beforecallme":
                    return bb(a);
                case "startcallme":
                    return cb(a);
                case "agentonline":
                    for (b =
                    0; b < da.length; b++)
                        da[b](a);
                    break;
                case "agentcommand":
                    return db(a)
                }
            } catch (U) {}
        },
        openProactiveChatDo: function(b, a, c, y) {
            "1" == SnapABugChat.getBannedCookie() ||!ka || r ||!0 != b && SnapABugChat.isNoProactiveChatCookie() || (SnapABugChat.reInit(), T && clearTimeout(T), Ca = a, c && (S = c), SnapABug.sendGetProactiveChatAgent(!1, y))
        },
        openProactiveV2APICall: function(b, a, c) {
            "1" == SnapABugChat.getBannedCookie() ||!ka || r ||!0 != b && SnapABugChat.isNoProactiveChatCookie() || (SnapABugChat.reInit(), Ca = a, c ? (S = c, SnapABug.sendGetProactiveChatAgent(!1)) :
            null !== pa ? SnapABug.executeOpenProactiveChatAPIRule(b, a, pa) : 3 > ea && SnapABug.setProactiveChatEvaluationFinishedCallback(function() {
                SnapABug.executeOpenProactiveChatAPIRule(b, a, pa)
            }))
        },
        setProactiveChatEvaluationFinishedCallback: function(b) {
            za = b
        },
        clearProactiveChatEvaluationFinishedCallback: function(b) {
            za = null
        },
        executeProactiveChatEvaluationFinishedCallback: function() {
            "function" === typeof za && za()
        },
        executeOpenProactiveChatAPIRule: function(b, a, c) {
            ga || (ga=!0, SnapABug.openProactiveChatDo(b, a, decodeURIComponent(c.message)))
        },
        initWidget: function(b, a) {
            d = b;
            SnapABug.isMobile() && "undefined" != typeof SNAPENGAGE_MOBILE ? SnapABug.initMobile() : this.injectDialog(a);
            this.handleChatInProgress() || ("" !== SnapABugChat.getCookie("SnapABugChatWindow") && SnapABugChat.setChatWindowCookieData("isMinimized", !1), T || (T = setTimeout("SnapABug.sendGetWidgetConfig(true);", 14E3)), null != SnapABug.getMobileUI() && (SnapABug.getMobileUI().hideTab(), SnapABug.getAgentStatusAsync(function(b) {
                !b || I && I.btnColor && "none" == I.btnColor || SnapABug.getMobileUI().showTab()
            })));
            try {
                var c = SnapABugChat.getCookie("SnapABugRef");
                c && "" != c || SnapABugChat.setCookie("SnapABugRef", encodeURIComponent(document.location.href + " " + document.referrer), 120)
            } catch (y) {}
            SnapABugChat.getVisitCookie() || (SnapABugChat.setHistoryCookieIncrementVisits(), SnapABugChat.setVisitCookie());
            SnapABugChat.incrementVisitCookie()
        },
        startPreChat: function() {
            null != SnapABug.getMobileUI() ? (SnapABug.getMobileUI().showChat(), SnapABug.getAgentStatusAsync(function(b) {
                b || (SnapABug.getMobileUI().hideChat(), SnapABug.getMobileUI().hideTab(),
                SnapABug.injectDialog(!0), SnapABug.mobileUI = null, SnapABug.loadForm())
            })) : la ? SnapABug.populateFormStatus( - 2) : (this.preventProactiveChat(), r || (this.getPreChatSearchEnabled() && 1 == this.getPreChatSearch() ? SnapABug.startPreChatSearch() : this.handleChatInProgress() || ("" == ca ? this.loadForm() : this.startChat(ca), this.handleChatInProgress(), Ea=!0)))
        },
        isScreenshotCapable: function() {
            if (!Ra)
                return !1;
            try {
                "" == P && (P = this.getJREVersions());
                if (0 != P.indexOf("1.6") && 0 != P.indexOf("1.7"))
                    return !1;
                var b = navigator.userAgent,
                b = b.toLowerCase();
                return - 1 != b.indexOf("intel mac os x 10.5")||-1 != b.indexOf("intel mac os x 10_5")||-1 != b.indexOf("intel mac os x 10.7")||-1 != b.indexOf("intel mac os x 10_7")||-1 != b.indexOf("intel mac os x 10.8")||-1 != b.indexOf("intel mac os x 10_8")||-1 != b.indexOf("intel mac os x 10.9")||-1 != b.indexOf("intel mac os x 10_9")||-1 != b.indexOf("intel mac os x 10.10")||-1 != b.indexOf("intel mac os x 10_10")?!1 : "none" != P && "undefined" != P
            } catch (a) {
                return !1
            }
        },
        isSneakPeek: function() {
            return !1
        },
        getPreChatSearch: function() {
            return mb
        },
        setPreChatSearch: function(b) {
            mb = b
        },
        getDomain: function() {
            return Y
        },
        getPageChangeNotificationAllowed: function() {
            return Ua
        },
        setChatStatus: function(b) {
            r = b
        },
        setCountry: function(b) {
            fb = b
        },
        getCountry: function() {
            return fb
        },
        setProactiveChatAutocloseTimeout: function() {
            ba && clearTimeout(ba);
            try {
                Da && (ba = setTimeout("SnapABug.slideFormOut();", 6E4 * Da))
            } catch (b) {}
        },
        clearProactiveChatAutocloseTimeout: function() {
            ba && (clearTimeout(ba), ba = null)
        },
        handleChatInProgress: function() {
            if (SnapABugChat) {
                var b = SnapABugChat.getCaseIdCookie();
                if (null != b && "" != b && "closed" != b)
                    return this.continueChat(b), !0
            }
            return !1
        },
        continueChat: function(b) {
            "" == x && "" == n && (r=!0, x = b, SnapABugChat.restartChat(x, f), this.hideButton())
        },
        getVisitorCountry: function() {
            return na ? na : ""
        },
        setVisitorCountry: function(b) {
            na = b
        },
        enableCallMe: function() {
            ia = '<div style="' + this.getCSS(void 0, void 0, 78, 16, "overflow:hidden;cursor:pointer;cursor:hand;") + '"  onclick="SnapABug.clickedCallMe(); return true;"><img style="position:relative;margin:0;border:none;padding:0;" src="' + SnapABug.getStaticBaseURL() +
            '/wbg/snapabug_callme_en.png" border="0" alt="Call me now" width="72" height="32" onmouseover="this.style.top=\'-16px\';" onmouseout="this.style.top=\'0px\';" /></div>';
            var b = document.getElementById("SnapABug_CallMe");
            b && (b.innerHTML = ia)
        },
        disableCallMe: function() {
            ia = "";
            var b = document.getElementById("SnapABug_CallMe");
            b && (b.innerHTML = ia)
        },
        clickedCallMe: function() {
            this.callCallback("BeforeCallme", "") || SnapABugChat.populateCallMeForm()
        },
        injectDialog: function(b) {
            SnapABug.styleTag = document.getElementById("SnapABug_Style");
            SnapABug.styleTag || (SnapABug.styleTag = document.createElement("style"), SnapABug.styleTag.id = "SnapABug_Style", SnapABug.styleTag.type = "text/css", SnapABug.styleTag.styleSheet ? SnapABug.styleTag.styleSheet.cssText = "" : SnapABug.styleTag.appendChild(document.createTextNode("")), (document.head || document.getElementsByTagName("head")[0]).appendChild(SnapABug.styleTag), SnapABug.addStyleRule || (SnapABug.addStyleRule = function(b, a) {
                var t = "", c = "";
                if ("string" === typeof b)
                    t = "#SnapABug_WP" !== b.substr(0, 12) ? "#SnapABug_WP " :
                    "", c = t + b + " { " + a + " } ";
                else if ("object" === typeof b && "undefined" !== typeof b.length) {
                    for (var qa = 0; qa < b.length; qa++)
                        t = "#SnapABug_WP" !== b[qa].substr(0, 12) ? "#SnapABug_WP " : "", c += t + b[qa], qa !== b.length - 1 && (c += ", ");
                    c += "{ " + a + " } "
                }
                SnapABug.styleTag.styleSheet ? SnapABug.styleTag.styleSheet.cssText += c : SnapABug.styleTag.textContent += c
            }), SnapABug.addStyleRule('#SnapABug_WP div span iframe h1 h2 h3 h4 h5 h6 hr p a big em img q small strong b u i fieldset input input[type="submit"] button textarea form label'.split(" "),
            "margin: 0; padding: 0; border: 0; font-size: 100%; font: inherit; vertical-align: baseline; float: none;max-width: none; max-height: none; letter-spacing: normal; -webkit-transform: none; -moz-transform: none; -ms-transform: none; -o-transform: none; transform: none; -webkit-transition-property: none; -moz-transition-property: none; -o-transition-property: none; transition: none;-webkit-box-shadow: none; -moz-box-shadow: none; box-shadow: none;font-feature-settings: normal; -webkit-font-feature-settings: normal; -moz-font-feature-settings: normal; -ms-font-feature-settings: normal; -o-font-feature-settings: normal; "),
            SnapABug.addStyleRule(["*:before", "*:after"], "display:none;"));
            var a = "position:fixed;_position:absolute;display:none;top:0%;_top:expression(eval(document.body.scrollTop));left:0px;width:100%;height:100%;text-align:left;z-index:2147483644;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;list-style:none;text-align:left;font-size:100%;font-weight:normal;outline:none;background-image:url('" + SnapABug.getBlankImg() + "');background-repeat:repeat;", c = '<div id="SnapABug_O" style="position:absolute;width:100%;height:100%;top:0px;left:0px;text-align:left;z-index:auto;-moz-opacity:0.5;opacity:.50;filter:alpha(opacity=50);background-color:' +
            Z + ';"></div>', y = '<div id="SnapABug_P" style="position:absolute;left:0px;top:0px;margin:0;padding:0;border:none;text-align:left;z-index:1;overflow:hidden;background-color:transparent;"></div><div id="SnapABug_IP" style="position:absolute;left:0px;top:0px;z-index:0;"></div><div id="SnapABug_DB" style="position:absolute;left:0px;top:0px;margin:0;border:0;padding:0;z-index:1;cursor:move;background-image:url(\'' + SnapABug.getBlankImg() + "');background-repeat:repeat;\"></div>";
            if (b) {
                b = document.getElementById("SnapABug_W");
                b || (b = document.createElement("div"), b.id = "SnapABug_W", document.body.appendChild(b));
                var U = document.getElementById("SnapABug_WP");
                U || (U = document.createElement("div"), U.id = "SnapABug_WP", document.body.appendChild(U));
                var e = document.getElementById("SnapABug_Applet");
                e || (e = document.createElement("div"), e.id = "SnapABug_Applet", document.body.appendChild(e));
                try {
                    b.setAttribute("style", a), U.setAttribute("style", "position:fixed;_position:absolute;display:none;z-index:2147483645;overflow:hidden;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;list-style:none;outline:none;"),
                    e.setAttribute("style", "position:fixed;_position:absolute;display:none;top:0%;_top:expression(eval(document.body.scrollTop));left:0px;width:100%;height:2px;text-align:left;z-index:2147483646;")
                } catch (d) {}
                try {
                    b.style.cssText = a, U.style.cssText = "position:fixed;_position:absolute;display:none;z-index:2147483645;overflow:hidden;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;list-style:none;outline:none;", e.style.cssText = "position:fixed;_position:absolute;display:none;top:0%;_top:expression(eval(document.body.scrollTop));left:0px;width:100%;height:2px;text-align:left;z-index:2147483646;"
                } catch (m) {}
                b.innerHTML =
                c;
                U.innerHTML = y + SnapABug.soundNotificationDiv()
            } else 
                document.write('<div id="SnapABug_W" style="' + a + '">' + c + '</div><div id="SnapABug_WP" style="position:fixed;_position:absolute;display:none;z-index:2147483645;overflow:hidden;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;list-style:none;outline:none;">' + y + SnapABug.soundNotificationDiv() + '</div><div id="SnapABug_Applet" style="position:fixed;_position:absolute;display:none;top:0%;_top:expression(eval(document.body.scrollTop));left:0px;width:100%;height:2px;text-align:left;z-index:2147483646;"></div>')
        },
        setMobileStyle: function(b) {
            null == I && (I = {});
            "undefined" !== typeof b && ("undefined" !== typeof b.btnColor && (I.btnColor = b.btnColor), "undefined" !== typeof b.tabPosition && (I.tabPosition = b.tabPosition), "undefined" !== typeof b.tabOffset && (I.tabOffset = b.tabOffset))
        },
        initMobile: function() {
            var b = {
                baseUrl: k + "/m",
                chat: {
                    showPoweredBy: SnapABug.isBranded(),
                    chatSoundEnabled: SnapABug.getChatSoundAllowed(),
                    chatUpdatedCallback: function(b, a) {
                        SnapABug.sendCreateCase(SnapABug.getUserEmail(), a, !0) || b && SnapABugChat.sendChatMessage(1,
                        a.replace("\n", ""))
                    },
                    onClose: function() {
                        SnapABugChat.closeChat(!0)
                    }
                },
                tab: {
                    text: "Chat",
                    textColor: "#fff",
                    counterColor: "#333",
                    position: {
                        pos: "left",
                        offset: "30%"
                    }
                },
                emoticons: {
                    smile: {
                        url: k + "/m/icons/smile-light.png",
                        characters: ":)"
                    },
                    sad: {
                        url: k + "/m/icons/sad-light.png",
                        characters: ":("
                    },
                    ok: {
                        url: k + "/m/icons/ok-light.png",
                        characters: ":|"
                    }
                }
            };
            "undefined" !== typeof I && ("undefined" !== typeof I.btnColor && "none" != I.btnColor && (b.tab.backgroundColor = I.btnColor), "undefined" !== typeof I.tabPosition && (b.tab.position.pos =
            I.tabPosition), "undefined" !== typeof I.tabOffset && (b.tab.position.offset = I.tabOffset));
            wa = new SNAPENGAGE_MOBILE(b);
            wa.init();
            wa.showTab()
        },
        soundNotificationDiv: function() {
            if (SnapABug.hasFlash())
                return '<div id="SnapABug_Snd" style="width:0px;height:0px;font-size:0px;margin:0;padding:0;background-color:transparent;"></div>';
            SnapABug.allowChatSound(!1);
            return ""
        },
        positionDragger: function(b, a, c, y) {
            var e = document.getElementById("SnapABug_DB");
            e.style.left = b + "px";
            e.style.top = a + "px";
            e.style.width = c + "px";
            e.style.height =
            y + "px"
        },
        moveForm: function(b, a) {
            var c = document.getElementById("SnapABug_WP");
            0 > b && (b = 0);
            b > SnapABug.getWindowWidth() - c.offsetWidth && (b = SnapABug.getWindowWidth() - c.offsetWidth);
            0 > a && (a = 0);
            a > SnapABug.getWindowHeight() - c.offsetHeight && (a = SnapABug.getWindowHeight() - c.offsetHeight);
            c.style.left = b + "px";
            c.style.top = a + "px";
            SnapABug.formPosX = b;
            SnapABug.formPosY = a
        },
        setupDraggerHandler: function() {
            try {
                var b = document.getElementById("SnapABug_WP");
                document.getElementById("SnapABug_DB").onmousedown = function(b) {
                    SnapABug.dragging =
                    !0;
                    document.getElementById&&!document.all ? (SnapABug.dragFromX = b.pageX, SnapABug.dragFromY = b.pageY) : (b = document.compatMode && "BackCompat" != document.compatMode ? document.documentElement : document.body, SnapABug.dragFromX = event.clientX + b.scrollLeft, SnapABug.dragFromY = event.clientY + b.scrollTop);
                    return !1
                };
                Ka || ra ||!document.onmouseup || "function" !== typeof document.onmouseup || (ra = document.onmouseup);
                Ka=!0;
                document.onmouseup = function(b) {
                    SnapABug.dragging && SnapABugChat.setChatWindowCookieData("chatBoxPosition",
                    [SnapABug.formPosX, - 1, SnapABug.formPosY, - 1]);
                    SnapABug.dragging=!1;
                    try {
                        ra && ra(b)
                    } catch (a) {}
                };
                La || aa ||!document.onmousemove || "function" !== typeof document.onmousemove || (aa = document.onmousemove);
                La=!0;
                document.onmousemove = function(a) {
                    if (SnapABug.dragging) {
                        var c = 0, t = 0;
                        if (document.getElementById&&!document.all)
                            c = b.offsetLeft + a.pageX - SnapABug.dragFromX, t = b.offsetTop + a.pageY - SnapABug.dragFromY, SnapABug.dragFromX = a.pageX, SnapABug.dragFromY = a.pageY;
                        else {
                            var e = document.compatMode && "BackCompat" != document.compatMode ?
                            document.documentElement: document.body, c = b.offsetLeft + event.clientX + e.scrollLeft - SnapABug.dragFromX, t = b.offsetTop + event.clientY + e.scrollTop - SnapABug.dragFromY;
                            SnapABug.dragFromX = event.clientX + e.scrollLeft;
                            SnapABug.dragFromY = event.clientY + e.scrollTop
                        }
                        SnapABug.moveForm(c, t);
                        try {
                            aa && aa(a)
                        } catch (d) {}
                        return !1
                    }
                    try {
                        aa && aa(a)
                    } catch (m) {}
                    return !0
                }
            } catch (a) {}
        },
        injectButton: function(b, a, c, y, e) {
            "https:" == document.location.protocol && (y && (y = y.replace("http://", "https://")), e && (e = e.replace("http://", "https://")));
            var d = "", m = C, l;
            l = "%" == m.substr(m.length - 1, 1) ? "left" == A || "right" == A ? Math.round(m.substr(0, m.length - 1) * this.getWindowHeight() / 100) : Math.round(m.substr(0, m.length - 1) * this.getWindowWidth() / 100) : m.substr(0, m.length - 2);
            var n = "";
            a || "" != B || (n = this.getMessage(0));
            y && e && y == e ? (B = y, w = z =- 1) : a || y && e ? (B = f + "/statusImage?w=" + b + "&rand=" + encodeURIComponent(Math.floor(4294967295 * Math.random()).toString(16)), w = z =- 1, y && e && (B += "&on=" + encodeURIComponent(y) + "&off=" + encodeURIComponent(e))) : "" == B && (B = k + "/btn/help_" + A + "_" + M,
            B = this.isIE6() ? B + ".gif" : B + ".png", "top" == A || "bottom" == A) && (b = z, z = w, w = b);
            switch (A) {
            case "left":
                d += "body .SnapABug_Button{cursor:pointer;cursor:hand;overflow:hidden;position:fixed;_position:absolute;display:block;top:" + m + ";_top:expression(eval(document.body.scrollTop)+" + l + ");left:0px;z-index:2147000000;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;outline:none;font-size:0px;line-height:0px;}";
                break;
            case "right":
                d += "body .SnapABug_Button{cursor:pointer;cursor:hand;overflow:hidden;position:fixed;_position:absolute;display:block;top:" +
                m + ";_top:expression(eval(document.body.scrollTop)+" + l + ");right:0px;z-index:2147000000;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;outline:none;font-size:0px;line-height:0px;}";
                break;
            case "top":
                d += "body .SnapABug_Button{cursor:pointer;cursor:hand;overflow:hidden;position:fixed;_position:absolute;display:block;top:0px;left:" + m + ";_left:expression(eval(document.body.scrollLeft)+" + l + ");z-index:2147000000;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;outline:none;font-size:0px;line-height:0px;}";
                break;
            case "bottom":
                d += "body .SnapABug_Button{cursor:pointer;cursor:hand;overflow:hidden;position:fixed;_position:absolute;display:block;bottom:0px;left:" + m + ";_left:expression(eval(document.body.scrollLeft)+" + l + ");z-index:2147000000;margin:0;padding:0;border-collapse:collapse;border-spacing:0;border:none;outline:none;font-size:0px;line-height:0px;}"
            }
            d += " @media print{body .SnapABug_Button{display:none;}}";
            m = "";
            0 < z && 0 < w && (m = 'width="' + z + '" height="' + w + '"');
            n = '<img id="SnapABug_bImg" style="position:relative;" src="' +
            B + '" alt="' + n + '" border="0" ' + m + ' onmouseover="SnapABug.buttonOver();" onmouseout="SnapABug.buttonOut();" />';
            c ? (c = document.getElementsByTagName("head")[0], m = document.createElement("style"), m.type = "text/css", m.styleSheet ? m.styleSheet.cssText = d : m.appendChild(document.createTextNode(d)), c.appendChild(m), d = document.getElementById("SnapABug_Button"), d || (d = document.createElement("div"), d.id = "SnapABug_Button", d.className = "SnapABug_Button", d.onclick = SnapABug.startLink, document.body.appendChild(d)), d.innerHTML =
            n) : (document.write('<style type="text/css">' + d + "</style>"), document.write('<div class="SnapABug_Button" onclick="SnapABug.startLink();">' + n + "</div>"));
            this.buttonOut()
        },
        buttonOver: function() {
            var b = document.getElementById("SnapABug_bImg").style;
            switch (A) {
            case "left":
                b.left = "0px";
                break;
            case "right":
                b.right = "0px";
                break;
            case "top":
                b.top = "0px";
                break;
            case "bottom":
                b.bottom = "0px"
            }
            this.preloadLoadingImg()
        },
        buttonOut: function() {
            try {
                var b = document.getElementById("SnapABug_bImg").style;
                switch (A) {
                case "left":
                    b.left =
                    J;
                    break;
                case "right":
                    b.right = J;
                    break;
                case "top":
                    b.top = J;
                    break;
                case "bottom":
                    b.bottom = J
                }
            } catch (a) {}
        },
        isWebCapture: function() {
            return u && (!r || a) && this.isScreenshotCapable()
        },
        showForm: function(b, a, c, y, e, d, m) {
            SnapABug.formWidth = a;
            SnapABug.formHeight = c;
            var f = document.getElementById("SnapABug_P"), l = document.getElementById("SnapABug_IP"), n = document.getElementById("SnapABug_WP"), g = document.getElementById("SnapABug_W"), p = document.getElementById("SnapABug_O");
            n.style.width = a + "px";
            n.style.height = c + "px";
            f.style.width =
            a + "px";
            f.style.height = c + "px";
            l.style.left = y + "px";
            l.style.top = e + "px";
            l.style.width = d + "px";
            l.style.height = m + "px";
            n.style.display = "block";
            f.style.display = "block";
            l.style.display = 0 < d && 0 < m ? "block" : "none";
            b && "" != Z ? (g.style.display = "block", p.style.display = "block", p.style.width = this.getWindowWidth() + "px", p.style.height = this.getWindowHeight() + "px") : (g.style.display = "none", p.style.display = "none");
            a = [ - 1, - 1, - 1, - 1];
            SnapABug.formPosX && SnapABug.formPosY ? (a[0] = SnapABug.formPosX, a[2] = SnapABug.formPosY) : a = this.getDefaultChatPosition(b);
            l.innerHTML = '<iframe width="100%" height="100%" frameborder="0" src="' + SnapABug.getBaseURL() + '/empty.html"></iframe>';
            this.positionForm(a[0], a[1], a[2], a[3]);
            this.setupDraggerHandler()
        },
        getDefaultChatPosition: function(b) {
            var a =- 1, c =- 1, y =- 1, e =- 1;
            switch (F) {
            case "tl":
                y = c = 0;
                break;
            case "tr":
                y = a = 0;
                break;
            case "bl":
                e = c = 0;
                break;
            default:
                e = a = 0
            }
            if (b || "c" == F)
                b = document.getElementById("SnapABug_P"), e = a =- 1, c = Math.round(this.getWindowWidth() / 2 - b.offsetWidth / 2), 0 >= c && (c = 0), 0 !== H ? y = H : (y = Math.round(this.getWindowHeight() /
                2 - b.offsetHeight / 2), 0 >= y && (y = 0));
            return [c, a, y, e]
        },
        positionForm: function(b, a, c, y, e) {
            if (!SnapABug.isAnimating() || e)
                if (e = document.getElementById("SnapABug_WP"), this.isIE6()) {
                    var d = document.documentElement.scrollTop ? document.documentElement.scrollTop: document.body.scrollTop, m = document.documentElement.scrollLeft ? document.documentElement.scrollLeft: document.body.scrollLeft;
                    e.style.top = c ? d + c + "px" : d + SnapABug.getWindowHeight() - y - e.offsetHeight + "px";
                    e.style.left = b ? m + b + "px" : m + SnapABug.getWindowWidth() - a - e.offsetWidth +
                    "px"
                } else 
                    e.style.top =- 1 != c ? c + "px" : "", e.style.bottom =- 1 != y ? y + "px" : "", e.style.left =- 1 != b ? b + "px" : "", e.style.right =- 1 != a ? a + "px" : ""
        },
        isAnimating: function(b) {
            "undefined" !== typeof b && (nb = b);
            return nb
        },
        slideFormOut: function() {
            if (!x&&!n) {
                for (var b = SnapABug.getWindowWidth(), a = document.getElementById("SnapABug_WP"), c = a.offsetLeft, a = a.offsetTop, e = 0, d = 0; 20 > d; d++)
                    e = c > b / 2 ? c + (b - c) * (1 - Math.cos(d * Math.PI / 20 / 2)) : c - (c + SnapABug.formWidth) * (1 - Math.cos(d * Math.PI / 20 / 2)), 0 == d && (SnapABug.isAnimating(!0), setTimeout("SnapABug.isAnimating(false);",
                    1E3)), setTimeout("SnapABug.positionForm(" + e + ", -1, " + a + ", -1, true);", 1E3 * d / 20);
                setTimeout("SnapABug.closeForm(true);", 1E3)
            }
        },
        slideFormIn: function() {
            for (var b = SnapABug.getWindowWidth(), a = document.getElementById("SnapABug_WP"), c = a.offsetLeft, a = a.offsetTop, e = 0, d = 0; 20 > d; d++)
                e = c > b / 2 ? c + (b - c) * (1 - Math.sin(d * Math.PI / 20 / 2)) : c - (c + SnapABug.formWidth) * (1 - Math.sin(d * Math.PI / 20 / 2)), 0 == d ? (SnapABug.positionForm(e, - 1, a, - 1), SnapABug.isAnimating(!0), setTimeout("SnapABug.isAnimating(false);", 600), e = SnapABug.getDefaultChatPosition(),
                setTimeout("SnapABug.positionForm(" + e[0] + "," + e[1] + "," + e[2] + "," + e[3] + ");", 601)) : setTimeout("SnapABug.positionForm(" + e + ", -1, " + a + ", -1, true);", 600 * d / 20)
        },
        getCSS: function(b, a, c, e, d, m) {
            var f = ";position:absolute;display:block;margin:0;padding:0;border:none;float:none;clear:none;";
            void 0 != b && (f += "left:" + b + "px;");
            void 0 != a && (f += "top:" + a + "px;");
            void 0 != c && (f += "width:" + c + "px;max-width:" + c + "px;min-width:" + c + "px;");
            void 0 != e && (f += "height:" + e + "px;max-height:" + e + "px;min-height:" + e + "px;");
            f += "z-index:2;background-color:transparent;background:none;font-style:normal;font-weight:normal;font-size:100%;line-height:normal;text-shadow:none;list-style:none;" +
            ("" == ua ? "text-align:left;" : "text-align:right;") + "outline:none;text-indent:0px;-moz-box-shadow:none;-webkit-box-shadow:none;" + ua;
            void 0 != d && (f += d.replace(/"/g, "'"));
            void 0 != m && (f += m.replace(/"/g, "'"));
            return f
        },
        getBlankImg: function() {
            this.isIE6();
            return k + "/wbg/blank.gif"
        },
        preloadLoadingImg: function() {
            Ja || (Ja=!0, (new Image).src = k + "/wbg/loading.gif")
        },
        showLoading: function() {
            document.getElementById("SnapABug_P").innerHTML = '<div style="position:absolute;left:0;top:0;z-index:1;"><img src="' + k + '/wbg/loading.gif" style="' +
            this.getCSS(0, 0, 31, 31) + '" galleryimg="no" width="31" height="31" border="0" alt="Loading..." /></div>';
            this.showForm(!0, 31, 31, 0, 0, 0, 0)
        },
        loadForm: function() {
            this.showLoading();
            var b = new Image;
            b.onload = function() {
                SnapABug.populateForm()
            };
            b.src = Ia;
            return !1
        },
        getSnapABugSiteUrl: function() {
            return "http://www.snapengage.com/partner?src=widget&amp;id=" + d + "&amp;l=" + SnapABug.getLocale()
        },
        getLogoClickATag: function() {
            var b = SnapABugChat.getPageTitle(!1).toLowerCase(), b =- 1 != b.indexOf("live chat")||-1 != b.indexOf("online chat");
            return '<a href="' + SnapABug.getSnapABugSiteUrl() + '" target="_blank" style="text-indent:0px;border:none;"' + (b ? "" : ' rel="nofollow"') + ">"
        },
        logoLink: function() {
            return "9a9bb60a-5a65-400c-a6a6-c4095f88c3a7" == d || "594d526a-8b21-4dea-9937-941fab9c959d" == d || "8f3f2320-a57b-4cd9-9cc2-7030c2984f76" == d || "de9efea5-50ce-405b-ba38-d597c5d63bc7" == d || "ad15d298-5080-406e-acad-0fa1eb07c1e4" == d || "e5277a72-aea3-4184-b8eb-f056273a52e9" == d || "e09b64f1-5bf5-415d-8fc0-1a633f5f01b5" == d || "c314e9a6-57ca-4db9-bab6-6a902284e0da" ==
            d || "23a8596b-1662-4f10-b111-e733c61e2294" == d || "7c9d6f66-5ac9-4d4c-aa76-3b803cb4d873" == d || "fc798239-0a16-462d-b012-654cf57b0bc8" == d || "8c5066d4-7432-484a-a297-3f8e20b09a54" == d || "40277fbf-4da8-4569-8ea2-573323e291f6" == d || "c4c0c4a0-03e7-49bf-ba4e-09467f70d658" == d || "9e7310f0-e8a4-4210-9033-98d3ef28f95c" == d || "d960b840-4485-43ca-9831-0ab9a0429f24" == d || "e268792b-888d-427c-8ebc-e8f6f1265df7" == d || "299e9557-a29b-4948-b143-fa5a6fce3faf" == d?!1 : $
        },
        initMinimizeArea: function(b, a, c, e) {
            "undefined" === typeof SnapABugChat.getChatWindowCookieData("isMinimized") &&
            SnapABugChat.setChatWindowCookieData("isMinimized", !1);
            SnapABugChat.minimizedDimensions = [b, a, c, e];
            var d = document.getElementById("SnapABug_RB");
            if (!d) {
                d = document.createElement("div");
                d.id = "SnapABug_RB";
                SnapABug.addStyleRule("#SnapABug_WP #SnapABug_RB", "position: absolute; z-index: 12; cursor: pointer; left: " + b + "px; top: " + a + "px; width: " + c + "px; height: " + e + "px; display: none");
                SnapABug.addStyleRule("#SnapABug_WP.minimized #SnapABug_RB", "display: block");
                try {
                    d.setAttribute("onclick", "SnapEngageChat.minimizeToggle()")
                } catch (m) {}
                try {
                    d.onclick =
                    function() {
                        SnapABugChat.minimizeToggle()
                    }
                } catch (f) {}
                document.getElementById("SnapABug_WP").appendChild(d)
            }
        },
        initMinimizeNotification: function(b, a, c, e, d, m, f, l, n) {
            var p = document.getElementById("SnapABug_WP"), g = document.getElementById("SnapABug_NMN"), k = document.getElementById("SnapABug_NMNi");
            g || (g = document.createElement("div"), g.id = "SnapABug_NMN");
            k || (k = document.createElement("span"), k.id = "SnapABug_NMNi");
            c = c ? c : 25;
            e = e ? e : 20;
            m = m ? m : 3;
            d = d ? d : 16;
            l = l ? l : "#FFFFFF";
            b = "display: none; position: absolute; left:" +
            (b ? b : 0) + "px; top:" + (a ? a : 0) + "px; z-index: 2; font-size:" + d + "px; text-align: center; cursor: pointer;  background-color: " + (f ? f : "#FF0000") + "; color: " + l + "; -webkit-border-radius:" + m + "px; border-radius:" + m + "px;" + (n ? n : "");
            l = "display: inline-block; box-sizing: border-box;  color: " + l + "; font-size: 1em; font-weight: bold; padding:0 0.3em;";
            a = e === c;
            d = (e / d).toFixed(4);
            l = a ? l + (" line-height:" + d + "em; min-width:" + d + "em;") : l + (" line-height:" + e + "px; min-width:" + c + "px;");
            try {
                g.setAttribute("style", b), g.setAttribute("onclick",
                "SnapEngageChat.minimizeToggle()"), k.setAttribute("style", l)
            } catch (q) {}
            try {
                g.style.cssText = b, g.onclick = function() {
                    SnapABugChat.minimizeToggle()
                }, k.style.cssText = l
            } catch (x) {}
            p.appendChild(g);
            g.appendChild(k)
        },
        initChatBoxMenuButton: function(b, a, c, e, d, m, f, l) {
            var n = document.getElementById("SnapABug_WP"), g = document.getElementById("SnapABug_CBMBtn"), p = document.getElementById("SnapABug_CBMBtnI");
            if (!g) {
                g = document.createElement("div");
                g.id = "SnapABug_CBMBtn";
                p || (p = document.createElement("i"), p.id = "SnapABug_CBMBtnI");
                g.appendChild(p);
                b = b ? b : SnapABug.formWidth - 30;
                a = a ? a : 10;
                c = c ? c : 20;
                e = e ? e : 20;
                d = d ? d : "black";
                m = m ? m : "transparent";
                extraCSS = f ? f : "";
                if (l)
                    SnapABug.menuButtonIconSpriteSheetURL = l;
                else 
                    switch (l = SnapABug.getStaticBaseURL() + "/wbg/minBtn_sprites_v2_", d) {
                    case "white":
                    case "#FFFFFF":
                    case "#ffffff":
                        SnapABug.menuButtonIconSpriteSheetURL = l + "white.png";
                        break;
                    default:
                        SnapABug.menuButtonIconSpriteSheetURL = l + "black.png"
                    }
                SnapABug.addStyleRule("#SnapABug_CBMBtn", "position: absolute; z-index: 2;left:" + b + "px; top:" + a + "px; width:" +
                c + "px; height:" + e + "px; cursor: pointer; display: block; background-color: " + m + "; color: " + d + "; " + f);
                SnapABug.addStyleRule("#SnapABug_CBMBtn>i", "position: absolute; z-index: 2;right:0px; top:0px; width:20px; height:20px; background-image: url(" + SnapABug.menuButtonIconSpriteSheetURL + "); background-repeat: no-repeat; background-position: " + SnapABug.getIconCoords("chevron-down") + ";");
                SnapABug.addStyleRule("#SnapABug_WP.minimized #SnapABug_CBMBtn>i", "background-position: " + SnapABug.getIconCoords("chevron-up") +
                ";");
                SnapABug.addStyleRule(["#SnapABug_CBMBtn:hover>i", "#SnapABug_CBM"], "display:none;");
                n.appendChild(g)
            }
        },
        initChatBoxMenu: function(b, a, c, e, d, m) {
            var f = document.getElementById("SnapABug_CBMBtn");
            if (f) {
                var l = document.getElementById("SnapABug_CBM");
                if (!l) {
                    l = document.createElement("div");
                    l.id = "SnapABug_CBM";
                    var n = SnapABug.getStaticBaseURL() + "/wbg/minBtn_sprites_v2_";
                    b = b ? b : "500%";
                    extraCSS = d ? d : "";
                    switch (c) {
                    case "white":
                    case "#FFFFFF":
                    case "#ffffff":
                        d = "#FFFFFF";
                        e = e ? e : "#555555";
                        a = a ? a : "#000000";
                        SnapABug.menuIconSpriteSheetURL =
                        m ? m : n + "white.png";
                        break;
                    default:
                        d = "#000000", e = e ? e : "#EEEEEE", a = a ? a : "#FFFFFF", SnapABug.menuIconSpriteSheetURL = m ? m : n + "black.png"
                    }
                    SnapABug.addStyleRule(["#SnapABug_CBMBtn:hover #SnapABug_CBM:hover", "#SnapABug_CBMBtn:hover #SnapABug_CBM", "#SnapABug_CBMBtn #SnapABug_CBM:hover"], "display:block;");
                    SnapABug.addStyleRule("#SnapABug_CBM", "position: absolute; z-index: 2;width: " + b + "; padding: 1px 3px; border-radius: 1px;top: 0px; right: 0px; background: " + a + "; color: " + d + "; box-shadow: 0 1px 4px 0 " + d + "; -webkit-box-shadow: 0 2px 4px 0 " +
                    d + ";border: thin solid " + e + ";" + extraCSS);
                    SnapABug.addStyleRule(".SnapABug_CBMItem", "position:relative; height: 20px; width: 100%;");
                    SnapABug.addStyleRule(".SnapABug_CBMItem:hover", "background-color: " + e + ";");
                    SnapABug.addStyleRule(".SnapABug_CBMItem span", "font-size: 14px; white-space: nowrap; padding-left: 10px;");
                    SnapABug.addStyleRule(".SnapABug_CBMItem i", "position:absolute; opacity: 0.6; top:0; right:0; height: 20px; width: 20px; display: inline-block;");
                    SnapABug.addStyleRule(".SnapABug_CBMItemSplitter",
                    "height:0; width: 90%; margin: 0px auto;  border-bottom: thin solid " + c + ";clear: both;opacity: 0.15; -ms-filter: progid:DXImageTransform.Microsoft.Alpha(Opacity=15);filter: alpha(opacity=15);");
                    SnapABug.addStyleRule("#SnapABug_closeBtn", "margin-top: 3px;");
                    SnapABug.addStyleRule("#SnapABug_minBtn", "margin-bottom: 3px;");
                    SnapABug.appendMenuItem(l, "SnapABug_minBtn", "Minimize", "div", SnapABug.getIconCoords("minimize"), "SnapABugChat.minimizeToggle();") && SnapABug.appendMenuSplitter(l);
                    SnapABug.appendMenuItem(l,
                    "SnapABug_closeBtn", "End Chat", "div", SnapABug.getIconCoords("close"), "SnapABug.closeForm();");
                    f.appendChild(l)
                }
            }
        },
        appendMenuItem: function(b, a, c, e, d, m) {
            if (null !== document.getElementById(a))
                return !1;
            e = document.createElement(e);
            var f = document.createElement("span"), l = document.createElement("i");
            e.id = a;
            e.className = "SnapABug_CBMItem";
            newElemIconCSS = "background-image: url('" + SnapABug.menuIconSpriteSheetURL + "'); background-repeat: no-repeat; background-position: " + d + ";";
            f.innerHTML = c;
            try {
                e.setAttribute("onclick",
                m), l.setAttribute("style", newElemIconCSS)
            } catch (n) {}
            try {
                e.onclick = function() {
                    eval(m)
                }, l.style || (l.style.cssText = newElemIconCSS)
            } catch (g) {}
            e.appendChild(f);
            e.appendChild(l);
            b.appendChild(e);
            return !0
        },
        appendMenuSplitter: function(b) {
            var a = document.createElement("div");
            a.className = "SnapABug_CBMItemSplitter";
            b.appendChild(a)
        },
        getIconCoords: function(b) {
            switch (b) {
            case "chevron-down":
                return "0 0";
            case "chevron-up":
                return "0 -20px";
            case "minimize":
                return "0 -40px";
            case "close":
                return "0 -60px";
            default:
                return "-100px -100px"
            }
        },
        hideChatBoxMenu: function() {
            var b = document.getElementById("SnapABug_WP"), a = document.getElementById("SnapABug_CBMBtn");
            a && b.removeChild(a)
        },
        isBranded: function() {
            return !1
        },
        populateForm: function() {
            var b = "", a = "background-image:url('" + this.getBlankImg() + "');background-repeat:repeat;", c = this.getCSS(322, 7, 51, 44, "cursor:pointer;"), e = this.getCSS(0, 0, 380, 400);
            SnapABug.positionDragger(7, 7, 315, 46);
            var d = this.getCSS(30, 60, 320, 20, "font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif; font-size: 12px; font-weight:600;color:#0087c1;"),
            m = this.getCSS(35, 85, 310, 20, "font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif; font-size:13px;color:black;", a), f = this.getCSS(30, 120, 320, 20, "font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif; font-size: 12px; font-weight:600;color:#0087c1;"), n = this.getCSS(35, 142, 314, 156, "font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif; font-size:13px;color:black;", a), a = this.getCSS(30, 310, 330, 20, "font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif; font-size: 12px; font-weight:600;color:#0087c1;"),
            g = this.getCSS(30, 335, 90, 25), p = this.getCSS(30, 370, 320, 20, "font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif; font-size: 12px; font-weight:600;color:#0087c1;"), q = this.getCSS(120, 330, 235, 25, "text-align:right;");
            W && (b = 'readonly="1" disabled="1"');
            "" == D && "" != SnapABugChat.getUserEmailCookie() && (D = SnapABugChat.getUserEmailCookie());
            b = '<div style="left:0;top:0;z-index:1;"><img src="' + Ia + '" style="' + e + '" galleryimg="no" width="380" height="400" /></div><div id="SnapABug_H" style="position:absolute;left:37px;top:126px;width:200px;z-index:100000;font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif; ;text-align:left;color:white;background-color:#67a8f3;border:4px solid white;padding:4px;font-size:9pt;display:none;"></div><form name="SnapABug_Form" accept-charset="UTF-8" style="margin:0px;" action="#" onsubmit="return SnapABug.submitForm(false);"><div id="SnapABug_email" style="' +
            d + '">' + this.getMessage(1) + "</div><div><input " + b + ' style="' + m + '" type="text" id="email" name="email" value="' + D + '" size="10"/></div><div id="SnapABug_desc" style="' + f + '">' + this.getMessage(2) + '</div><div><textarea name="description" style="' + n + '">' + l + "</textarea></div>";
            this.isScreenshotCapable() && Ta && (b += '<div style="' + a + '"><input type="checkbox" ' + (R ? 'checked="1"' : "") + ' name="screenshot" style="float:none;clear:none;width:12px;height:12px;vertical-align:middle;margin:0;padding:0;display:inline;" /><span style="vertical-align:middle;color:#0087c1"> ' +
            this.getMessage(3) + ' </span><span onmouseover="SnapABug.showHelp();" onmouseout="SnapABug.hideHelp();" style="display:inline;"><img style="display:inline;margin:0;padding:0;border:none;vertical-align:middle;width:12px;height:12px;" src="' + k + "/wbg/help." + (this.isIE6() ? "gif" : "png") + '" width="12" height="12" border="0" /></span></div>');
            b += '<div id="SnapABug_FormButtons" style="' + q + 'text-align:right;"><input id="SnapABug_OSB" onclick="return SnapABug.submitForm(false);" style="background:#0087c1; cursor: pointer; border: none; -webkit-border-radius:5px; -moz-border-radius:5px; border-radius:5px; margin-left: 5px; padding: 7px 15px; width:auto; overflow:visible;font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif;  font-size:12px; font-weight:400; color:#ffffff; text-decoration:none; vertical-align:middle;display:none;" type="button" value="' +
            this.getMessage(12) + '" onmouseover="this.style.color=\'#ffffff\';" onmouseout="this.style.color=\'#ffffff\';" /><input id="SnapABug_OCB" onclick="return SnapABug.submitForm(true);" style="background:#0087c1; cursor: pointer; border: none; -webkit-border-radius:5px; -moz-border-radius:5px; border-radius:5px; margin-left: 5px; padding: 7px 15px; width:auto; overflow:visible;font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif;  font-size:12px; font-weight:400; color:#ffffff; text-decoration:none; vertical-align:middle;display:none;font-weight:bold;" type="button" value="' +
            this.getMessage(13) + '" onmouseover="this.style.color=\'#ffffff\';" onmouseout="this.style.color=\'#ffffff\';" /><input id="SnapABug_SB" onclick="return SnapABug.submitForm(false);" style="background:#0087c1; cursor: pointer; border: none; -webkit-border-radius:5px; -moz-border-radius:5px; border-radius:5px; margin-left: 5px; padding: 7px 15px; width:auto; overflow:visible;font-family: lucida grande, Helvetica Neue, Helvetica, Arial, sans-serif;  font-size:12px; font-weight:400; color:#ffffff; text-decoration:none; vertical-align:middle;font-weight:bold;" type="button" value="' +
            this.getMessage(12) + '" onmouseover="this.style.color=\'#ffffff\';" onmouseout="this.style.color=\'#ffffff\';" /></div></form><div id="SnapABug_FormFbk" style="' + p + '"><span id="SnapABug_FormFbkT"></span> <a id="SnapABug_FormFbkL" target="_blank" href="#" style="color:#0087c1;font-weight:bold;font-size:12px;"></a></div>';
            b += '<div style="' + c + '" onclick="SnapABug.closeForm();" title="' + this.getMessage(10) + '"><img style="margin:0;border:none;padding:0;width:51px;height:44px;" src="' + this.getBlankImg() +
            '" width="51" height="44" alt="' + this.getMessage(10) + '" /></div>';
            this.isBranded() && (b += '<div style="' + g + '">', this.logoLink() && (b += '<a href="' + this.getSnapABugSiteUrl() + '" target="_blank" style="text-indent:0px;border:none;">'), b += '<img style="margin:0;border:none;padding:0;" src="' + k + '/wbg/logo.png" border="0" alt="powered by SnapEngage" width="84" height="24" />', this.logoLink() && (b += "</a>"), b += "</div>");
            document.getElementById("SnapABug_P").innerHTML = b;
            this.sendGetWidgetConfig(!1);
            this.showForm(!0,
            380, 400, 13, 13, 341, 358);
            this.showFeedback();
            this.updateFormButtons();
            (c = document.SnapABug_Form) && ("" == D ? c.email.focus() : c.description.focus());
            return !1
        },
        updateFormButtons: function() {
            if (la)(document.getElementById("SnapABug_OCB") || document.getElementById("SnapABug_OSB") || document.getElementById("SnapABug_CL")) && SnapABug.populateFormStatus( - 2);
            else {
                try {
                    var b = document.getElementById("SnapABug_SB"), a = document.getElementById("SnapABug_OSB"), c = document.getElementById("SnapABug_OCB");
                    e ? (b && (b.style.display =
                    "none"), Qa && a && (a.style.display = "inline"), c && (c.style.display = "inline")) : (b && (b.style.display = "inline"), a && (a.style.display = "none"), c && (c.style.display = "none"))
                } catch (d) {}(b = document.SnapABug_Form) && b.screenshot && this.isScreenshotCapable() && (b.screenshot.checked = 1 == R)
            }
        },
        showFeedback: function() {
            try {
                var b = document.getElementById("SnapABug_FormFbk");
                b && (ha || G ? (document.getElementById("SnapABug_FormFbkT").innerHTML = ha, document.getElementById("SnapABug_FormFbkL").href = Q, document.getElementById("SnapABug_FormFbkL").innerHTML =
                G, b.style.display = "block") : b.style.display = "none")
            } catch (a) {}
        },
        populateFormStatus: function(b) {
            if (null != SnapABug.getMobileUI() && 4 == b && r && SnapABugChat)
                return SnapABugChat.startChat(x, f), !1;
            var a = this.getMessage(9), c = '<div style="margin-top:30px;background:white;width:100%;border:none;text-align:left;"><div id="SnapABug_PB" style="background:#61656d;width:' + Math.round(100 * b / 4) + '%;height:20px;border:none;text-align:left;"></div></div>';
            1 == b ? a = this.getMessage(6) : 4 == b ? a = this.getMessage(7) : 5 == b ? a = this.getMessage(7) +
            ".." : 6 == b ? a = this.getMessage(8) : 7 == b ? a = this.getMessage(11) : - 2 == b && (a = "Sorry, this SnapEngage account is currently inactive");
            if ((4 == b || 5 == b) && r && SnapABugChat)
                return SnapABugChat.isWebCapInProgress() ? SnapABugChat.webCaptureDone(X) : SnapABugChat.startChat(x, f), !1;
            var e = "", d = $;
            4 == b || 5 == b || 6 == b || 7 == b||-2 == b ? (e = "SnapABug.closeForm();", c = '<div style="text-align:middle;margin-top:20px;"><input type="button" value="' + this.getMessage(10) + '" onclick="' + e + '" style="float:none;clear:none;width:auto;height:28px;margin:0;padding:3px 5px 3px 5px;font-size:12px;font-family:lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif;cursor:pointer;" /></div>') :
            $=!1;
            if (!r && "" == ja || 6 == a || 7 == a||-2 == a)
                b = F, F = "c", SnapABugChat.populateSubChatForm('<div style="margin:50px 20px;font-size:14px;font-family:lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif;color:#61656d !important;">' + a + c + "</div>", e), F = b;
            $ = d
        },
        updadateFormStatusBar: function(b, a) {
            var c = document.getElementById("SnapABug_PB");
            c && (c.style.width = Math.round(100 * b / 4 + 100 * a / 4) + "%")
        },
        closeForm: function(b) {
            var a = r;
            g && (this.hideOverlay(!1), r ? (SnapABugChat.closeChat(!0), r=!1) : SnapABugChat.reInit());
            SnapABugChat &&
            !0 != b && SnapABugChat.setNoProactiveChatCookie();
            SnapABug.proactiveMsg = "";
            SnapABug.proactiveAgentNumber = "";
            SnapABugChat.clearProactive();
            SnapABugChat.isMinimized() && SnapABugChat.unminimizeChat();
            SnapABug.hideChatBoxMenu();
            this.showButton();
            !0 != b && SnapABug.callCallback("Close", SnapABugChat.isProactive() ? "proactive" : a ? "chat" : "form", e ? "online" : "offline")
        },
        hideOverlay: function(b) {
            b || (document.getElementById("SnapABug_W").style.display = "none");
            document.getElementById("SnapABug_O").style.display = "none";
            document.getElementById("SnapABug_P").style.display =
            "none";
            document.getElementById("SnapABug_WP").style.display = "none";
            window.scrollBy(1, 0);
            window.scrollBy( - 1, 0)
        },
        submitForm: function(b) {
            var a = document.SnapABug_Form;
            if (a) {
                u = a.screenshot && a.screenshot.checked;
                K = 0;
                a.screenshot && (K = a.screenshot.checked ? 1 : 2);
                "" == L ? L = document.getElementById("SnapABug_email").style.color : (document.getElementById("SnapABug_email").style.color = L, document.getElementById("SnapABug_desc").style.color = L);
                var e=!1;
                "" != a.description.value || b&&!m || (document.getElementById("SnapABug_desc").style.color =
                "red", a.description.focus(), e=!0);
                !1 != /^([A-Za-z0-9_\-\.\+])*([A-Za-z0-9_\-\+])\@([A-Za-z0-9\-\.])+\.([A-Za-z]{2,6})$/.test(a.email.value) || b&&!c || (document.getElementById("SnapABug_email").style.color = "red", a.email.focus(), e=!0);
                e || (g=!1, SnapABugChat.setUserEmailCookie(a.email.value), D = a.email.value, this.sendCreateCase(a.email.value, SnapABug.deactivateHTML(a.description.value), b) || (document.getElementById("SnapABug_desc").style.color = "red", a.description.focus(), g=!0))
            }
            return !1
        },
        startChatGo: function() {
            ta =
            !1;
            e ? SnapABugChat.proactive(S, !1, "", "") : (ca = "", this.startLink())
        },
        sendClickToCall: function(b, a) {
            var c = "ci=" + encodeURIComponent(n);
            "" == n && (c = "c=" + encodeURIComponent(x));
            c += "&p=" + encodeURIComponent(b) + "&src=" + a;
            this.rpc(f + "/snapabug/ServiceClickToCall?" + c, 3, !0);
            this.callCallback("StartCallme", b)
        },
        callbackClickToCall: function(b) {},
        sendGetCallConfig: function() {
            var b = "w=" + encodeURIComponent(d);
            this.rpc(f + "/snapabug/ServiceGetCallConfig?" + b, 3, !0)
        },
        callbackGetCallConfig: function(b) {
            0 < b ? SnapABugChat.populateCallMeForm() :
            (ca = "", this.startLink())
        },
        sendGetWidgetConfig: function(b) {
            var a = "w=" + encodeURIComponent(d);
            b && (a += "&p=1");
            this.rpc(f + "/snapabug/ServiceGetConfig?" + a, 3, !0)
        },
        callbackGetWidgetConfig: function(b, d, f, y, l, n, g, p, k) {
            - 1 == b ? SnapABug.hideButton() : - 2 == b ? (la=!0, $=!1, (document.getElementById("SnapABug_OCB") || document.getElementById("SnapABug_OSB") || document.getElementById("SnapABug_CL") || ta) && SnapABug.populateFormStatus( - 2)) : (this.callCallback("agentOnline", b?!0 : !1), this.setCallback("agentOnline", null), ma = (new Date).getTime(),
            Fa*=1.1, k && 0 < k&&!SnapABugChat.isNoProactiveChatCookie()&&!xa && (T && clearTimeout(T), 15 >= k ? SnapABug.sendGetProactiveChatAgent() : T = setTimeout("SnapABug.sendGetProactiveChatAgent(false);", 1E3 * (k - 15))), R = l, e = b, "1" != SnapABugChat.getBannedCookie() && ka || (e = 0), this.updateFormButtons(), a = n, c = g, m = p, ha = d, Q = y, G = f, this.showFeedback(), ta && (this.startChatGo(), b && (SnapABug.callCallback("Open", "online"), Ga && SnapABug.sendCreateCase("", Ga, !0), SnapABug.getPreChatSearchEnabled() && SnapABug.resetPreChatSearch())), Ea && document.getElementById("SnapABug_OCB") &&
            (SnapABug.callCallback("Open", b ? "online" : "offline"), Ea=!1, SnapABug.getPreChatSearchEnabled() && SnapABug.resetPreChatSearch()))
        },
        sendGetProactiveChatAgent: function(b, a) {
            if (Sa) {
                var c = "w=" + encodeURIComponent(d);
                b && (c += "&c=1");
                "undefined" !== typeof a && (c += "&rid=" + a);
                this.rpc(f + "/snapabug/ServiceGetProactiveChatAgent?" + c, 3, !0)
            }
        },
        prepareProactiveMessage: function(b, a) {
            var c = b;
            return c = b.replace("{alias}", a)
        },
        callbackProactiveChatAgent: function(b, a, c, e, d, m) {
            ga=!1;
            SnapABug.proactiveAgentNumber = a;
            "" != ja ? (SnapABugChat.proactive(" ",
            !1, b, m), u=!1, SnapABug.sendCreateCase(SnapABug.getUserEmail(), "Call-me request", !0)) : ("" != S && (e = SnapABug.prepareProactiveMessage(S, b), sa=!0), "" != a && "" != e ? (S = e, d && "1" == d && (sa=!0), SnapABugChat.proactive(S, sa, b, m), this.hideButton(), E && (E.length = 0)) : Ca && this.startLink());
            c ? SnapABug.enableCallMe() : SnapABug.disableCallMe()
        },
        setFromUrlOverride: function(b) {
            va = b
        },
        sendCreateCase: function(b, a, c) {
            this.hideButton();
            if (r)
                return !1;
            "" == b && "" != D && (b = D);
            "" == b && "" != SnapABugChat.getUserEmailCookie() && (b = SnapABugChat.getUserEmailCookie());
            "" == a && (a = "(none)");
            var e = "e=" + encodeURIComponent(b) + "&w=" + encodeURIComponent(d) + "&l=" + encodeURIComponent(M) + "&d=" + encodeURIComponent(a.replace(/\n/g, "<br />")) + "&k=" + encodeURIComponent(Math.floor(4294967295 * Math.random()).toString(16));
            "undefined" !== typeof va && "" !== va && (e += "&fo=" + encodeURIComponent(va));
            e = f + "/snapabug/ServiceCreateCase?" + e;
            if (2E3 < e.length)
                return !1;
            r = c;
            null!=!SnapABug.getMobileUI() && (this.isWebCapture() ? this.populateFormStatus(1) : this.populateFormStatus(2), SnapABugChat && c && (SnapABugChat.setQuestion(a),
            SnapABugChat.preloadChatFormImg(), SnapABugChat.chatStarting()));
            this.rpc(e, 3, !1);
            c ? SnapABug.callCallback("StartChat", b, a, SnapABugChat.isProactive() ? "proactive" : "manual") : SnapABug.callCallback("MessageSubmit", b, a);
            return !0
        },
        callbackCreateCase: function(b, a, c, e, d, m) {
            if ("false" == b)
                return "Error" == a ? (g=!0, this.populateFormStatus(6)) : "Duplicate" == a && (g=!0, this.populateFormStatus(7)), r=!1;
            x = b;
            n = d;
            q = c;
            v = e;
            b = SnapABugChat.getHistoryCookie();
            "" != b && eb && SnapABug.setCvval("__sehist", b);
            SnapABugChat.getPageTitle(!1) &&
            SnapABug.setCvval("__sepaget", SnapABugChat.getPageTitle(!1));
            b = SnapABugChat.getTokensCookies();
            "" != b && SnapABug.setCvval("__setokens", b);
            SnapABugChat.setHistoryCookieAppendCase(x);
            try {
                this.isWebCapture() ? this.startWebCapture() : (this.populateFormStatus(3), this.sendFinalizeCase())
            } catch (f) {
                SnapABug.sendFinalizeCase()
            }
        },
        sendFinalizeCase: function() {
            var b = "ci=" + encodeURIComponent(n) + "&v=" + encodeURIComponent(this.getPlugins());
            - 1 != X && (b += "&e=" + encodeURIComponent(X));
            var b = b + ("&s=" + encodeURIComponent(K)),
            b = b + (r ? "&m=c2" : "&m=h"), b = b + ("&u=" + encodeURIComponent(this.getEntryPageUrl())), b = b + ("&r=" + encodeURIComponent(this.getEntryPageRef())), a = this.getJSVariables(), c = "&vjs=" + encodeURIComponent(a), c = f + "/snapabug/ServiceFinalizeCase?" + b + c;
            if (2E3 < c.length) {
                V = [];
                try {
                    for (var e = f + "/snapabug/ServiceAddToCase?ci=" + encodeURIComponent(n) + "&vjs=", d = 2E3 - e.length, m = encodeURIComponent(a), l = a = 0; ;)
                        if (l = a + d, l >= m.length) {
                            V.push(e + m.substr(a));
                            break
                        } else {
                            if ("%" == m.charAt(l - 1) || "%" == m.charAt(l - 2) || "%" == m.charAt(l - 3)) {
                                for (var g =
                                1; 4 > g; g++)
                                    if ("%" == m.charAt(l - g)) {
                                        l -= g;
                                        break
                                    }
                                    for (; ;) {
                                        var p = parseInt(m.charAt(l - 2) + m.charAt(l - 1), 16);
                                        if (127 < p && 192 > p)
                                            l -= 3;
                                        else {
                                            l -= 3;
                                            break
                                        }
                                    }
                            }
                            V.push(e + m.substring(a, l));
                            a = l
                        }
                    V.push((f + "/snapabug/ServiceFinalizeCase?" + b).substring(0, 2E3));
                    V.reverse();
                    Ba = V.length;
                    this.callbackAddToCase(!0)
                } catch (k) {
                    this.rpc(c.substring(0, 2E3), 3, !1)
                }
            } else 
                this.rpc(c, 3, !1)
        },
        callbackFinalizeCase: function(b) {
            b ? (g=!0, this.populateFormStatus(4)) : this.callbackError(!0)
        },
        callbackAddToCase: function(b) {
            b ? 0 < V.length && (0 < Ba && this.updadateFormStatusBar(3,
            1 - V.length / Ba), this.rpc(V.pop(), 3, !1)) : this.callbackError(!0)
        },
        sendCheckPhone: function(b) {
            b = "w=" + encodeURIComponent(d) + "&p=" + encodeURIComponent(b);
            this.rpc(f + "/snapabug/ServiceCheckPhone?" + b, 3, !0)
        },
        callbackCheckPhone: function(b) {
            b ? (x || n ? SnapABug.sendClickToCall(ja, 2) : this.sendGetProactiveChatAgent(!0), SnapABug.disableCallMe(), SnapABugChat.formCallMeDone()) : SnapABugChat.populateCallMeForm(!0, this.getCallMePhone())
        },
        startWebCapture: function() {
            try {
                var b =- 1 != navigator.appVersion.indexOf("Mac") ? 1E3 : 500;
                X =- 1;
                document.getElementById("SnapABug_Applet").style.display = "block";
                document.getElementById("SnapABug_Applet").innerHTML = '<applet name="SnapABug_Applet" archive="sWebCaptureApplet.jar" code="webCaptureApplet.WebCaptureApplet" codebase="' + f + '/" width="' + this.getWindowWidth() + '" width="100%" height="2" mayscript style="position:absolute;background-position-x:0px;background-position-y:0px;display:inline;font-size:2px;width:' + this.getWindowWidth() + 'px;height:2px;" ><param name="u" value="' + f + '"><param name="h" value="' +
                this.getWindowHeight() + '"><param name="w" value="' + this.getWindowWidth() + '"><param name="c" value="' + x + '"><param name="ci" value="' + n + '"><param name="k" value="' + ( - 1 != navigator.userAgent.toLowerCase().indexOf("msie") ? 1 : 0) + '"><param name="d" value="' + b + '"><param name="t" value="' + this.getOSBarHeight() + '"><param name="ns" value="SnapABug"></applet>'
            } catch (a) {
                this.webCaptureUploadDone()
            }
        },
        stopWebCapture: function() {
            try {
                document.getElementById("SnapABug_Applet").style.display = "none"
            } catch (b) {}
        },
        webCaptureReady: function() {
            this.hideOverlay(!0)
        },
        webCaptureSnapshotDone: function() {
            this.populateFormStatus(2)
        },
        webCaptureUploadDone: function() {
            this.populateFormStatus(3);
            this.stopWebCapture();
            SnapABugChat && SnapABugChat.isWebCapInProgress() ? this.populateFormStatus(4) : this.sendFinalizeCase()
        },
        webCaptureFailed: function(b) {
            X = b;
            g=!0;
            this.webCaptureUploadDone()
        },
        thankYouNote: function() {
            g=!0;
            this.populateFormStatus(4)
        },
        showHelp: function() {
            try {
                h = document.getElementById("SnapABug_H"), h.innerHTML = this.getMessage(5), h.style.display = "block"
            } catch (b) {}
            return !1
        },
        hideHelp: function() {
            try {
                document.getElementById("SnapABug_H").style.display = "none"
            } catch (b) {}
        },
        getFlashVersion: function() {
            var b =- 1 != navigator.appVersion.indexOf("MSIE")?!0 : !1, a =- 1 != navigator.appVersion.toLowerCase().indexOf("win")?!0 : !1, c =- 1 != navigator.userAgent.indexOf("Opera")?!0 : !1;
            try {
                var e = "none";
                if (null != navigator.plugins && 0 < navigator.plugins.length) {
                    if (navigator.plugins["Shockwave Flash 2.0"] || navigator.plugins["Shockwave Flash"]) {
                        var d = navigator.plugins["Shockwave Flash" + (navigator.plugins["Shockwave Flash 2.0"] ?
                        " 2.0" : "")].description.split(" "), m = d[2].split("."), f = m[0], l = m[1], g = d[3];
                        "" == g && (g = d[4]);
                        "d" == g[0] ? g = g.substring(1) : "r" == g[0] && (g = g.substring(1), 0 < g.indexOf("d") && (g = g.substring(0, g.indexOf("d"))));
                        e = f + "." + l + "." + g
                    }
                } else if ( - 1 != navigator.userAgent.toLowerCase().indexOf("webtv/2.6"))
                    e = 4;
                else if ( - 1 != navigator.userAgent.toLowerCase().indexOf("webtv/2.5"))
                    e = 3;
                else if ( - 1 != navigator.userAgent.toLowerCase().indexOf("webtv"))
                    e = 2;
                else if (b && a&&!c) {
                    var b = "", n;
                    try {
                        n = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"),
                        b = n.GetVariable("$version")
                    } catch (p) {}
                    if (!b)
                        try {
                            n = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"), b = "WIN 6,0,21,0", n.AllowScriptAccess = "always", b = n.GetVariable("$version")
                        } catch (k) {}
                    if (!b)
                        try {
                            n = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.3"), b = n.GetVariable("$version")
                        } catch (q) {}
                    if (!b)
                        try {
                            new ActiveXObject("ShockwaveFlash.ShockwaveFlash.3"), b = "WIN 3,0,18,0"
                    } catch (x) {}
                    if (!b)
                        try {
                            new ActiveXObject("ShockwaveFlash.ShockwaveFlash"), b = "WIN 2,0,0,11"
                    } catch (O) {}
                    e = b
                }
                return e.replace(/,/g, ".")
            } catch (v) {
                return "none"
            }
        },
        getJREVersions: function() {
            function b(b) {
                try {
                    return window.ActiveXObject ? null != new window.ActiveXObject("JavaWebStart.isInstalled." + b + ".0") : !1
                } catch (a) {
                    return !1
                }
            }
            function a(b, c) {
                try {
                    var e = b.split("."), d = c.split("."), m = e.length;
                    d.length > m && (m = d.length);
                    for (var t = 0; t < m; t++) {
                        if (e.length == t)
                            return c;
                        if (d.length == t)
                            return b;
                        if (e[t] < d[t])
                            return c;
                        if (e[t] > d[t])
                            return b
                    }
                } catch (f) {}
                return b
            }
            if (!navigator.javaEnabled())
                return "none";
            var c = "undefined";
            try {
                if ( - 1 != navigator.userAgent.toLowerCase().indexOf("msie"))
                    c =
                    b("1.8.0") ? "1.8.0" : b("1.7.0") ? "1.7.0" : b("1.6.0") ? "1.6.0" : b("1.5.0") ? "1.5.0" : b("1.4.2") ? "1.4.2" : "none";
                else if (navigator.mimeTypes)
                    for (var e = 0; e < navigator.mimeTypes.length; e++)
                        s = navigator.mimeTypes[e].type, null != s.match(/^application\/x-java-applet\x3Bversion=([0-9\.]+)/) && (c = "undefined" != c ? a(c, RegExp.$1) : RegExp.$1)
            } catch (d) {}
            return c
        },
        getPlugins: function() {
            if ("true" == q)
                try {
                    return "Flash=" + this.getFlashVersion() + ",Java=" + this.getJREVersions()
            } catch (b) {
                return "E"
            }
            return ""
        },
        hasFlash: function() {
            var b = SnapABug.getFlashVersion();
            return "none" != b && "" != b?!0 : !1
        },
        getJSVariables: function() {
            try {
                O = "";
                for (var b in p)
                    this.setVarVal(b + "=" + p[b]);
                if ("" != v) {
                    var a = v.split(",");
                    for (b = 0; b < a.length; b++)
                        "" != a[b] && "undefined" == typeof p[a[b]] && eval("if(typeof " + a[b] + '== "undefined"){SnapABug.setVarVal("' + a[b] + '=undefined");}else{SnapABug.setVarVal("' + a[b] + '="+' + a[b] + ");};")
                    }
                return O
            } catch (c) {
                return "E"
            }
        },
        setVarVal: function(b) {
            try {
                b = b.replace(/,/g, "&#44;"), "" != O && (O += ","), O += b
            } catch (a) {}
        },
        getMessage: function(b) {
            switch (SnapABug.getLocale()) {
            case "ar":
                switch (SnapABug.setDirection("direction:rtl;"),
                b) {
                case 0:
                    return "\u0644\u0644\u0645\u0633\u0627\u0639\u062f\u0629";
                case 1:
                    return "\u0639\u0646\u0648\u0627\u0646 \u0627\u0644\u0627\u064a\u0645\u064a\u0644";
                case 2:
                    return "\u0633\u0624\u0627\u0644\u0643";
                case 3:
                    return "\u0627\u0636\u0641 \u0644\u0642\u0637\u0629 \u0645\u0646 \u0627\u0644\u0635\u0641\u062d\u0629 \u0627\u0644\u062d\u0627\u0644\u064a\u0629";
                case 4:
                    return "\u0627\u0631\u0633\u0644";
                case 5:
                    return "Send a screen shot of what I am seeing on the web page so that the support team can provide a fast answer.";
                case 6:
                    return 'Preparing for snapshot...<br /><br />Please click "Run" or "Allow" to let our applet send a snapshot of the web page to the support team.';
                case 7:
                    return "\u0634\u0643\u0631\u0627 \u0644\u0642\u062f \u062a\u0645\u0651 \u0627\u0631\u0633\u0627\u0644 \u0631\u0633\u0627\u0644\u062a\u0643";
                case 8:
                    return "\u0639\u0630\u0631\u0627, \u0644\u0645 \u064a\u062a\u0645 \u0627\u0631\u0633\u0627\u0644 \u0631\u0633\u0627\u0644\u062a\u0643 \u0628\u0646\u062c\u0627\u062d. \u0646\u062d\u0646 \u0646\u062a\u062d\u0642\u0642 \u0645\u0646 \u0627\u0644\u0645\u0634\u0643\u0644\u0629. \u0627\u0644\u0631\u062c\u0627\u0621 \u0627\u0644\u0645\u062d\u0627\u0648\u0644\u0629 \u0645\u0646 \u062c\u062f\u064a\u062f \u0644\u0627\u062d\u0642\u0627.";
                case 9:
                    return "\u062c\u0627\u0631\u064a \u0627\u0644\u0627\u062a\u0651\u0635\u0627\u0644";
                case 10:
                    return "\u0627\u063a\u0644\u0627\u0642";
                case 11:
                    return "\u0644\u0642\u062f \u0633\u0628\u0642 \u0627\u0646 \u0631\u0627\u0633\u0644\u062a \u0627\u0644\u0641\u0631\u0639 \u0627\u0644\u0641\u0646\u0651\u064a \u0628\u062e\u0635\u0648\u0635 \u0647\u0630\u0647 \u0627\u0644\u0645\u0634\u0643\u0644\u0629.";
                case 12:
                    return "\u0631\u0627\u0633\u0644\u0646\u0627 \u0639\u0628\u0631 \u0627\u0644\u0627\u064a\u0645\u064a\u0644";
                case 13:
                    return "\u062f\u0631\u062f\u0634 \u0645\u0639\u0646\u0627";
                case 14:
                    return "\u0623\u0643\u062a\u0628 \u0631\u0633\u0627\u0644\u062a\u0643 \u0647\u0646\u0627";
                case 15:
                    return "\u0627\u0646\u0627"
                }
            case "az":
                switch (b) {
                case 0:
                    return "K\u00f6m\u0259k";
                case 1:
                    return "Sizin e-po\u00e7t:";
                case 2:
                    return "Probleminizi t\u0259svir edin:";
                case 3:
                    return "Hal-haz\u0131rdak\u0131 s\u0259hif\u0259nin qrafiki sur\u0259tini d\u0259 \u0259lav\u0259 et";
                case 4:
                    return "G\u00f6nd\u0259r";
                case 5:
                    return "D\u0259st\u0259kd\u0259n tez cavab al, veb s\u0259hif\u0259d\u0259 g\u00f6rd\u00fcy\u00fcm\u00fcn qrafiki sur\u0259tini g\u00f6nd\u0259r.";
                case 6:
                    return 'Qrafiki sur\u0259tin haz\u0131rlanmas\u0131... <br /><br /> Z\u0259hm\u0259t olmasa, "Run" (\u0130\u015f\u0259 sal) v\u0259 ya "Allow" (Icaz\u0259 ver) d\u00fcym\u0259l\u0259rini klik edin ki, proqram veb sayt\u0131n sur\u0259tini d\u0259st\u0259y\u0259 g\u00f6nd\u0259r\u0259 bilsin.';
                case 7:
                    return "T\u0259\u015f\u0259kk\u00fcrl\u0259r,<br/>Sizin m\u0259ktubunuz g\u00f6nd\u0259rildi.";
                case 8:
                    return "Ba\u011f\u0131\u015flay\u0131n, sizin m\u0259ktubunuz g\u00f6nd\u0259rilm\u0259di.<br /> Biz, sizin problem il\u0259 m\u0259\u015fquluq. <br/>Z\u0259hm\u0259t olmasa, bir q\u0259d\u0259r g\u00f6zl\u0259yin.";
                case 9:
                    return "Y\u00fckl\u0259nir...";
                case 10:
                    return "Ba\u011fla";
                case 11:
                    return "Siz bu problem haqq\u0131nda ismar\u0131c\u0131 d\u0259st\u0259y\u0259 art\u0131q g\u00f6nd\u0259rmisiniz.";
                case 12:
                    return "M\u0259ktub";
                case 13:
                    return "\u00c7at";
                case 14:
                    return "Operator cavab ver\u0259n kimi mesaj\u0131n\u0131z\u0131 bura daxil edib [enter] - \u0259 bas\u0131n";
                case 15:
                    return "M\u0259n"
                }
            case "cs":
                switch (b) {
                case 0:
                    return "N\u00e1pov\u011bda";
                case 1:
                    return "V\u00e1\u0161 email:";
                case 2:
                    return "Jak V\u00e1m m\u016f\u017eeme pomoci?";
                case 3:
                    return "Vlo\u017eit sn\u00edmek obrazovky aktu\u00e1ln\u00ed str\u00e1nky";
                case 4:
                    return "Odeslat";
                case 5:
                    return "Odeslat sn\u00edmek toho co pr\u00e1v\u011b vid\u00edm na str\u00e1nk\u00e1ch, aby z\u00e1kaznick\u00fd servis mohl rychle a p\u0159esn\u011b odpov\u011bd\u011bt.";
                case 6:
                    return 'P\u0159ipravuje se sn\u00edm\u00e1n\u00ed obrazovky...<br/><br/> Pros\u00edm klikn\u011bte na "Spustit" nebo "Umo\u017enit" pro odesl\u00e1n\u00ed sn\u00edmku obrazovky na\u0161emu z\u00e1kaznick\u00e9mu servisu.';
                case 7:
                    return "D\u011bkujeme,<br /> Va\u0161e zpr\u00e1va byla odesl\u00e1na.";
                case 8:
                    return "Omlouv\u00e1me se, Va\u0161e zpr\u00e1va nemohla b\u00fdt odesl\u00e1na.<br />V sou\u010dasnosti zji\u0161\u0165ujeme mo\u017en\u00e9 pot\u00ed\u017ee.<br />Pros\u00edme zkuste to pozd\u011bji.";
                case 9:
                    return "Nahr\u00e1v\u00e1 se...";
                case 10:
                    return "Zav\u0159\u00edt";
                case 11:
                    return "Tento dotaz byl ji\u017e odesl\u00e1n z\u00e1kaznick\u00e9mu servisu.";
                case 12:
                    return "Napi\u0161te n\u00e1m email";
                case 13:
                    return "Chatujte s n\u00e1mi";
                case 14:
                    return "napi\u0161te sem svoji zpr\u00e1vu a ode\u0161lete ji stisknut\u00edm tla\u010d\u00edtka [Enter]...";
                case 15:
                    return "J\u00e1"
                }
            case "da":
                switch (b) {
                case 0:
                    return "Hj&#230;lp";
                case 1:
                    return "Din emailadresse:";
                case 2:
                    return "Hvad kan vi hj\u00e6lpe dig med?";
                case 3:
                    return "Vedh&#230;ft et screenshot af siden";
                case 4:
                    return "Send";
                case 5:
                    return "Send et screenshot af det jeg ser p&#229; hjemmesiden&#44; s&#229; support teamet hurtigt kan afhj&#230;lpe problemet.";
                case 6:
                    return "G&#248;r klar til at tage et screenshot...<br /><br />Klik &#34;K&#248;r&#34; eller &#34;Accepter&#34; for at tillade&#44; at der bliver sendt et screenshot til support teamet.";
                case 7:
                    return "Tak,<br />din besked er sendt.";
                case 8:
                    return "Beklager&#44; din besked kunne ikke sendes.<br />Vi unders&#248;ger hvorfor&#44; venligst pr&#248;v igen senere.";
                case 9:
                    return "Sender...";
                case 10:
                    return "Luk";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Send email";
                case 13:
                    return "Chat med os";
                case 14:
                    return "Skriv din besked her og tryk p\u00e5 [Enter]";
                case 15:
                    return "mig"
                }
            case "de":
                switch (b) {
                case 0:
                    return "Hilfe";
                case 1:
                    return "Ihre E-Mail-Adresse:";
                case 2:
                    return "Ihre Frage:";
                case 3:
                    return "Bildschirmaufnahme der Seite hinzuf\u00fcgen";
                case 4:
                    return "Senden";
                case 5:
                    return "Durch anklicken senden Sie uns eine Bildschirmaufnahme der Webseite, damit wir Ihnen rascher helfen k\u00f6nnen.";
                case 6:
                    return "Vorbereitung der Bildschirmaufnahme...<br />Bitte bewilligen Sie die Sicherheitsabfrage des Java-Applets, um eine Bildschirmaufnahme der Web-Seite f&uuml;r das Support Team zu machen.";
                case 7:
                    return "Vielen Dank,<br />Ihre Nachricht wurde gesendet.";
                case 8:
                    return "Leider konnte Ihre Nachricht nicht gesendet werden.<br />Wir untersuchen das Problem.<br />Bitte versuchen Sie es sp&auml;ter erneut.";
                case 9:
                    return "Hochladen...";
                case 10:
                    return "Schlie&szlig;en";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "E-Mail senden";
                case 13:
                    return "Chat starten";
                case 14:
                    return "Bitte hier eine Nachricht eingeben und auf [Enter] dr\u00fccken um fortzufahren...";
                case 15:
                    return "ich"
                }
            case "el":
                switch (b) {
                case 0:
                    return "\u0392\u03bf\u03ae\u03b8\u03b5\u03b9\u03b1";
                case 1:
                    return "\u03a4\u03bf email \u03c3\u03b1\u03c2:";
                case 2:
                    return "\u03a0\u03b5\u03c1\u03b9\u03b3\u03c1\u03b1\u03c6\u03ae \u03c0\u03c1\u03bf\u03b2\u03bb\u03ae\u03bc\u03b1\u03c4\u03bf\u03c2:";
                case 3:
                    return "\u03a3\u03c5\u03bc\u03c0\u03b5\u03c1\u03b9\u03bb\u03ac\u03b2\u03b5\u03c4\u03b5 \u03bc\u03b9\u03b1 \u03b1\u03c0\u03b5\u03b9\u03ba\u03cc\u03bd\u03b9\u03c3\u03b7 \u03c4\u03b7\u03c2 \u03c0\u03b1\u03c1\u03bf\u03cd\u03c3\u03b1\u03c2 \u03c3\u03b5\u03bb\u03af\u03b4\u03b1\u03c2";
                case 4:
                    return "\u0391\u03c0\u03bf\u03c3\u03c4\u03bf\u03bb\u03ae";
                case 5:
                    return "\u03a3\u03c4\u03b5\u03af\u03bb\u03c4\u03b5 \u03bc\u03bf\u03c5 \u03bc\u03b9\u03b1 \u03b1\u03c0\u03b5\u03b9\u03ba\u03cc\u03bd\u03b9\u03c3\u03b7 \u03b1\u03c0\u03cc \u03c4\u03b7 \u03c3\u03b5\u03bb\u03af\u03b4\u03b1 \u03c0\u03bf\u03c5 \u03b2\u03bb\u03ad\u03c0\u03b5\u03c4\u03b5, \u03ce\u03c3\u03c4\u03b5 \u03c4\u03bf \u03c4\u03b5\u03c7\u03bd\u03b9\u03ba\u03cc \u03bc\u03b1\u03c2 \u03c4\u03bc\u03ae\u03bc\u03b1 \u03bd\u03b1 \u03c3\u03b1\u03c2 \u03b4\u03ce\u03c3\u03b5\u03b9 \u03bc\u03b9\u03b1 \u03c3\u03cd\u03bd\u03c4\u03bf\u03bc\u03b7 \u03b1\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7.";
                case 6:
                    return '\u03a0\u03c1\u03bf\u03b5\u03c4\u03bf\u03b9\u03bc\u03b1\u03c3\u03af\u03b1 \u03b3\u03b9\u03b1 \u03b1\u03c0\u03b5\u03b9\u03ba\u03cc\u03bd\u03b9\u03c3\u03b7...<br/><br/>\u03a0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb\u03ce \u03c0\u03b1\u03c4\u03ae\u03c3\u03c4\u03b5 "\u0395\u03ba\u03c4\u03ad\u03bb\u03b5\u03c3\u03b7" \u03ae "\u0395\u03c0\u03b9\u03c4\u03c1\u03ad\u03c0\u03c9" \u03ce\u03c3\u03c4\u03b5 \u03b7 \u03b5\u03c6\u03b1\u03c1\u03bc\u03bf\u03b3\u03ae \u03bd\u03b1 \u03c3\u03c4\u03b5\u03af\u03bb\u03b5\u03b9 \u03ad\u03bd\u03b1 snapshot \u03c4\u03b7\u03c2 \u03c3\u03b5\u03bb\u03af\u03b4\u03b1\u03c2 \u03c3\u03c4\u03bf \u03c4\u03b5\u03c7\u03bd\u03b9\u03ba\u03cc \u03bc\u03b1\u03c2 \u03c4\u03bc\u03ae\u03bc\u03b1.';
                case 7:
                    return "\u0395\u03c5\u03c7\u03b1\u03c1\u03b9\u03c3\u03c4\u03bf\u03cd\u03bc\u03b5,<br/>\u03a4\u03bf \u03bc\u03ae\u03bd\u03c5\u03bc\u03b1 \u03c3\u03c4\u03ac\u03bb\u03b8\u03b7\u03ba\u03b5.";
                case 8:
                    return "\u039b\u03c5\u03c0\u03bf\u03cd\u03bc\u03b1\u03c3\u03c4\u03b5, \u03c4\u03bf \u03bc\u03ae\u03bd\u03c5\u03bc\u03b1 \u03b4\u03b5\u03bd \u03bc\u03c0\u03bf\u03c1\u03b5\u03af \u03bd\u03b1 \u03c3\u03c4\u03b1\u03bb\u03b5\u03af.<br/>\u0395\u03c1\u03b5\u03c5\u03bd\u03bf\u03cd\u03bc\u03b5 \u03c4\u03bf \u03b8\u03ad\u03bc\u03b1.<br/>\u03a0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb\u03cc\u03cd\u03bc\u03b5 \u03c0\u03c1\u03bf\u03c3\u03c0\u03b1\u03b8\u03ae\u03c3\u03c4\u03b5 \u03b1\u03c1\u03b3\u03cc\u03c4\u03b5\u03c1\u03b1.";
                case 9:
                    return "\u0386\u03bd\u03b5\u03b2\u03b1\u03c3\u03bc\u03b1 \u03b1\u03c1\u03c7\u03b5\u03af\u03bf\u03c5...";
                case 10:
                    return "\u039a\u03bb\u03b5\u03af\u03c3\u03b9\u03bc\u03bf";
                case 11:
                    return "\u0388\u03c7\u03b5\u03c4\u03b5 \u03ae\u03b4\u03b7 \u03c3\u03c4\u03b5\u03af\u03bb\u03b5\u03b9 \u03c4\u03bf \u03c3\u03c5\u03b3\u03ba\u03b5\u03ba\u03c1\u03b9\u03bc\u03ad\u03bd\u03bf \u03c0\u03c1\u03cc\u03b2\u03bb\u03b7\u03bc\u03b1 \u03c3\u03c4\u03bf \u03c4\u03b5\u03c7\u03bd\u03b9\u03ba\u03cc \u03bc\u03b1\u03c2 \u03c4\u03bc\u03ae\u03bc\u03b1.";
                case 12:
                    return "\u0391\u03c0\u03bf\u03c3\u03c4\u03bf\u03bb\u03ae";
                case 13:
                    return "\u039c\u03b9\u03bb\u03ae\u03c3\u03c4\u03b5 \u03bc\u03b1\u03b6\u03af \u03bc\u03b1\u03c2";
                case 14:
                    return "\u03b3\u03c1\u03ac\u03c8\u03c4\u03b5 \u03c4\u03bf \u03bc\u03ae\u03bd\u03c5\u03bc\u03ac \u03c3\u03b1\u03c2 \u03ba\u03b1\u03b9 \u03c0\u03b1\u03c4\u03ae\u03c3\u03c4\u03b5 [enter] \u03b3\u03b9\u03b1 \u03bd\u03b1 \u03c3\u03c4\u03b1\u03bb\u03b5\u03af...";
                case 15:
                    return "\u0395\u03bc\u03ad\u03bd\u03b1"
                }
            case "es":
                switch (b) {
                case 0:
                    return "Ayuda";
                case 1:
                    return "Su email:";
                case 2:
                    return "\u00bfEn qu\u00e9 puedo ayudarle?";
                case 3:
                    return "Incluye una instant\u00e1nea de esta p\u00e1gina";
                case 4:
                    return "Enviar";
                case 5:
                    return "Envia una instant\u00e1nea de lo que estoy viendo en la p\u00e1gina web para que el equipo de soporte puedan darme una respuesta.";
                case 6:
                    return 'Preparando para la instant\u00e1nea ...<br /><br />Por favor haga clic en "Run" o "Allow" para permitir que nuestro applet env\u00ede una captura de pantalla de este sitio al equipo de soporte.';
                case 7:
                    return "Gracias,<br />Su mensaje ha sido enviado.";
                case 8:
                    return "Lo sentimos, su mensaje no pudo enviar.<br />Estamos investigando el problema.<br />Por favor, int\u00e9ntelo m\u00e1s tarde.";
                case 9:
                    return "Cargando...";
                case 10:
                    return "Cerrar";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Enviar";
                case 13:
                    return "Chat con nosotros";
                case 14:
                    return "Escriba su mensaje aqu\u00ed y presione [enter] para enviar...";
                case 15:
                    return "yo"
                }
            case "et":
                switch (b) {
                case 0:
                    return "Abi";
                case 1:
                    return "Sinu email:";
                case 2:
                    return "Kirjelda oma probleemi:";
                case 3:
                    return "Lisa chatile oma ekraanist tehtud pilt";
                case 4:
                    return "Saada";
                case 5:
                    return "Saada minu ekraanist tehtud pilt tehnilise toe ekspertidele, et saada kiirem vastus.";
                case 6:
                    return 'Teen ekraanist pilti...<br /><br />Palun vajuta "Run-Jooksuta" v\u00f5i "Allow-Luba" et saaksime n\u00e4ha pilti veebilehest millel hetkel olete.';
                case 7:
                    return "Ait\u00e4h,<br />Teie teade on edasi saadetud postituviga.";
                case 8:
                    return "Kahjuks polnud v\u00f5imalik teadet postituviga edastada.<br />Me uurime hetkel mis juhtus.<br />Palun proovige natukese aja p\u00e4rast uuesti.";
                case 9:
                    return "Laen \u00fcles...";
                case 10:
                    return "Sulge";
                case 11:
                    return "Selline probleem on juba tehnilise toe ekspertidele esitatud.";
                case 12:
                    return "Kirjuta meile";
                case 13:
                    return "R\u00e4\u00e4gi meiega";
                case 14:
                    return "Palun kirjutage oma teade siia ning vajutage [enter] et see edasi saata...";
                case 15:
                    return "Mina"
                }
            case "fi":
                switch (b) {
                case 0:
                    return "Palaute";
                case 1:
                    return "S&#228;hk&#246;postiosoitteesi:";
                case 2:
                    return "Lis\u00e4\u00e4 kysymyksesi:";
                case 3:
                    return "Liit&#228; kuvakaappaus t&#228;m&#228;nhetkisest&#228; sivusta";
                case 4:
                    return "L&#228;het&#228;";
                case 5:
                    return "L&#228;het&#228; kuvakaappaus n&#228;kym&#228;st&#228;&#44; jotta tukipalvelu voi vastata kysymykseesi nopeammin.";
                case 6:
                    return "Valmistellaan kuvakaappausta...<br /><br />Ole hyv&#228; ja valitse &#34;Suorita&#34; tai &#34;Hyv&#228;ksy&#34; valtuuttaaksesi ohjelman ottamaan kuvakaappauksen tukipalvelua varten.";
                case 7:
                    return "Kiitos&#44;<br />Viestisi on l&#228;hetetty.";
                case 8:
                    return "Valitettavasti viesti&#228; ei pystytty l&#228;hett&#228;m&#228;&#228;n.&<br />Ongelmaa tutkitaan.<br />Ole hyv&#228; ja yrit&#228; my&#246;hemmin uudelleen.";
                case 9:
                    return "L&#228;hetet&#228;&#228;n...";
                case 10:
                    return "Sulje";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "L\u00e4het\u00e4 Email";
                case 13:
                    return "Avaa Chat";
                case 14:
                    return "Kirjoita viestisi t\u00e4h\u00e4n ja paina [enter]...";
                case 15:
                    return "Min&#228;"
                }
            case "fr":
                switch (b) {
                case 0:
                    return "Aide";
                case 1:
                    return "Votre email :";
                case 2:
                    return "Votre question :";
                case 3:
                    return "Envoyez une copie d'&eacute;cran";
                case 4:
                    return "Envoyez";
                case 5:
                    return "Envoie une image de la page courante pour que l'&eacute;quipe de support puisse fournir une r&eacute;ponse rapidement.";
                case 6:
                    return "Pr&eacute;paration pour la capture d'&eacute;cran...<br />Merci de cliquer sur \"Run\" ou \"Allow\" pour que l'applet puisse envoyer l'image de l'&eacute;cran &agrave; l'&eacute;quipe de support.";
                case 7:
                    return "Merci,<br />votre message a &eacute;t&eacute; envoy&eacute;.";
                case 8:
                    return "D&eacute;sol&eacute;, votre message n'a pas pu &ecirc;tre envoy&eacute;.<br />Nous recherchons la cause du probl&egrave;me.<br />Merci de r&egrave;essayer plus tard.";
                case 9:
                    return "Envoi en cours...";
                case 10:
                    return "Fermer";
                case 11:
                    return "Vous avez d&eacute;j&agrave; envoy&eacute; ce message.";
                case 12:
                    return "Envoyez";
                case 13:
                    return "Tchattez avec nous";
                case 14:
                    return "Tapez votre message ici et appuyez sur [Entr\u00e9e] pour envoyer...";
                case 15:
                    return "moi"
                }
            case "fr_CA":
                switch (b) {
                case 0:
                    return "Aide";
                case 1:
                    return "Entrez votre adresse courriel :";
                case 2:
                    return "Posez votre question :";
                case 3:
                    return "Envoyez une copie d'&eacute;cran";
                case 4:
                    return "Envoyez";
                case 5:
                    return "Envoie une image de la page courante pour que l'&eacute;quipe de support puisse fournir une r&eacute;ponse rapidement.";
                case 6:
                    return "Pr&eacute;paration pour la capture d'&eacute;cran...<br />Merci de cliquer sur \"Run\" ou \"Allow\" pour que l'applet puisse envoyer l'image de l'&eacute;cran &agrave; l'&eacute;quipe de support.";
                case 7:
                    return "Merci,<br />votre message a &eacute;t&eacute; envoy&eacute;.";
                case 8:
                    return "D&eacute;sol&eacute;, votre message n'a pas pu &ecirc;tre envoy&eacute;.<br />Nous recherchons la cause du probl&egrave;me.<br />Merci de r&egrave;essayer plus tard.";
                case 9:
                    return "Envoi en cours...";
                case 10:
                    return "Fermer";
                case 11:
                    return "Vous avez d&eacute;j&agrave; envoy&eacute; ce message.";
                case 12:
                    return "Envoyez";
                case 13:
                    return "Clavardez avec nous";
                case 14:
                    return "Tapez votre message ici et appuyez sur [Entr\u00e9e] pour envoyer...";
                case 15:
                    return "moi"
                }
            case "hu":
                switch (b) {
                case 0:
                    return "Seg&iacute;ts&eacute;g";
                case 1:
                    return "E-mail c&iacute;med:";
                case 2:
                    return "&Iacute;rd le a k&eacute;rd&eacute;sed:";
                case 3:
                    return "Mell&eacute;keld az oldal snapshotj&aacute;t";
                case 4:
                    return "K&uuml;ldd";
                case 5:
                    return "K&uuml;ldd el az oldal k&eacute;p&eacute;t /screenshot/, hogy k&ouml;nnyebben seg&iacute;thess&uuml;nk.";
                case 6:
                    return 'El&otilde;k&eacute;sz&uuml;let a snapshotra...<br /><br />K&eacute;r&uuml;nk kattints a "Run" vagy "Allow"-ra &eacute;s a rendszer&uuml;nk elk&uuml;ldi azt a supportunknak.';
                case 7:
                    return "K&ouml;sz&ouml;nj&uuml;k,<br />az &uuml;zenetet elk&uuml;ldted.";
                case 8:
                    return "Sajn&aacute;ljuk, az &uuml;zenetet nem tudtuk elk&uuml;ldeni. Megvizsg&aacute;ljuk az ok&aacute;t. K&eacute;r&uuml;nk, pr&oacute;b&aacute;ld meg k&eacute;s&otilde;bb.";
                case 9:
                    return "Felt&ouml;lt&eacute;s...";
                case 10:
                    return "Bez&aacute;r&aacute;s";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "K\u00fcldd el";
                case 13:
                    return "Csetelj vel\u00fcnk";
                case 14:
                    return "ide \u00edrja be \u00fczenet\u00e9t \u00e9s nyomja meg az [Enter] gombot az elk\u00fcld\u00e9shez...";
                case 15:
                    return "\u00c9n"
                }
            case "it":
                switch (b) {
                case 0:
                    return "Help";
                case 1:
                    return "La sua email:";
                case 2:
                    return "La sua richiesta:";
                case 3:
                    return "Includi uno screenshot della pagina";
                case 4:
                    return "Invia";
                case 5:
                    return "Invia una schermata di quello che stai vedendo sulla pagina web per aiutare il team di supporto a fornirti una risposta pi&ugrave; rapidamente";
                case 6:
                    return "Preparazione per la cattura dello schermo...<br />Si prega di attivare  l'applet Java.";
                case 7:
                    return "Grazie,<br /> Il tuo messaggio &egrave; stato inviato.";
                case 8:
                    return "Mi dispiace, il tuo messaggio non &egrave; stato inviato.<br />Stiamo indagando sulla questione.<br />Si prega di riprovare pi&ugrave; tardi.";
                case 9:
                    return "Caricamento...";
                case 10:
                    return "Chiudi";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Invia";
                case 13:
                    return "Chatta con noi";
                case 14:
                    return "scrivi qui il tuo messaggio e premi [Invio]...";
                case 15:
                    return "me"
                }
            case "is":
                switch (b) {
                case 0:
                    return "Hj\u00e1lp";
                case 1:
                    return "Netfangi\u00f0 \u00feitt:";
                case 2:
                    return "L\u00fdstu vandam\u00e1linu:";
                case 3:
                    return "L\u00e1ta skj\u00e1skot af n\u00faverandi s\u00ed\u00f0u fylgja me\u00f0";
                case 4:
                    return "Senda";
                case 5:
                    return "Senda skj\u00e1skot af s\u00ed\u00f0unni sem \u00e9g er \u00e1 svo \u00fej\u00f3nustufulltr\u00faar geti a\u00f0sto\u00f0a\u00f0 mig betur.";
                case 6:
                    return 'Undib\u00fd skj\u00e1skot...<br /><br />Vinsamlegast smelltu \u00e1 "Run" e\u00f0a "Allow" svo a\u00f0 forriti\u00f0 geti sent skj\u00e1skot af vefs\u00ed\u00f0unni til \u00fej\u00f3nustufulltr\u00faanna.';
                case 7:
                    return "Takk fyrir.<br />Skilabo\u00f0in \u00fe\u00edn hafa veri\u00f0 send.";
                case 8:
                    return "\u00dev\u00ed mi\u00f0ur var ekki h\u00e6gt a\u00f0 senda skilabo\u00f0in \u00fe\u00edn.<br />Vi\u00f0 munum kanna \u00feetta vandam\u00e1l.<br />Vinsamlegast reyndu aftur s\u00ed\u00f0ar.";
                case 9:
                    return "Sendi skr\u00e1...";
                case 10:
                    return "Loka";
                case 11:
                    return "\u00de\u00fa hefur n\u00fa \u00feegar sent inn \u00feetta m\u00e1l til \u00fej\u00f3nustufulltr\u00faa.";
                case 12:
                    return "T\u00f6lvup\u00f3st";
                case 13:
                    return "Spjalla\u00f0u vi\u00f0 okkur";
                case 14:
                    return "Getur \u00fe\u00fa slegi\u00f0 inn skilabo\u00f0 og slegi\u00f0 \u00e1 [enter] til a\u00f0 senda...";
                case 15:
                    return "\u00c9g"
                }
            case "iw":
                switch (SnapABug.setDirection("direction:rtl;"), b) {
                case 0:
                    return "\u05e2\u05d6\u05e8\u05d4";
                case 1:
                    return '\u05db\u05ea\u05d5\u05d1\u05ea \u05d4\u05d3\u05d5\u05d0"\u05dc \u05e9\u05dc\u05da';
                case 2:
                    return "\u05d4\u05e9\u05d0\u05dc\u05d4 \u05e9\u05dc\u05da";
                case 3:
                    return "\u05d4\u05d5\u05e1\u05e3 \u05e6\u05d9\u05dc\u05d5\u05dd \u05de\u05e1\u05da \u05e9\u05dc \u05d4\u05d3\u05e3 \u05d4\u05e0\u05d5\u05db\u05d7\u05d9";
                case 4:
                    return "\u05e9\u05dc\u05d7";
                case 5:
                    return "\u05dc\u05ea\u05d2\u05d5\u05d1\u05d4 \u05de\u05d4\u05d9\u05e8\u05d4 \u05e9\u05dc \u05e6\u05d5\u05d5\u05ea \u05d4\u05ea\u05de\u05d9\u05db\u05d4, \u05d4\u05d5\u05e1\u05e3 \u05d1\u05d1\u05e7\u05e9\u05d4 \u05e6\u05d9\u05dc\u05d5\u05dd \u05de\u05e1\u05da \u05e9\u05dc \u05e2\u05de\u05d5\u05d3 \u05d4\u05d0\u05d9\u05e0\u05d8\u05e8\u05e0\u05d8";
                case 6:
                    return '\u05de\u05ea\u05db\u05d5\u05e0\u05df \u05dc\u05e6\u05d9\u05dc\u05d5\u05dd \u05d4\u05de\u05e1\u05da ...<br /><br />\u05dc\u05d7\u05e5 \u05e2\u05dc "\u05d0\u05e9\u05e8" \u05d0\u05d5 "\u05d4\u05e8\u05e5"  \u05dc\u05e9\u05dc\u05d9\u05d7\u05ea \u05e6\u05d9\u05dc\u05d5\u05dd \u05d4\u05de\u05e1\u05da \u05dc\u05e6\u05d5\u05d5\u05ea \u05d4\u05ea\u05de\u05d9\u05db\u05d4.';
                case 7:
                    return "\u05d4\u05d5\u05d3\u05e2\u05ea\u05da \u05e0\u05e9\u05dc\u05d7\u05d4. \u05ea\u05d5\u05d3\u05d4.";
                case 8:
                    return "\u05de\u05e6\u05d8\u05e2\u05e8\u05d9\u05dd, \u05dc\u05d0 \u05e0\u05d9\u05ea\u05df \u05dc\u05e9\u05dc\u05d5\u05d7 \u05d0\u05ea \u05d4\u05d5\u05d3\u05e2\u05ea\u05da.<br />\u05d0\u05e0\u05d5 \u05d1\u05d5\u05d7\u05e0\u05d9\u05dd \u05d0\u05ea \u05d4\u05e0\u05d5\u05e9\u05d0.<br />\u05d0\u05e0\u05d0 \u05e0\u05e1\u05d5 \u05e9\u05d5\u05d1 \u05d1\u05de\u05d5\u05e2\u05d3 \u05de\u05d0\u05d5\u05d7\u05e8 \u05d9\u05d5\u05ea\u05e8";
                case 9:
                    return "\u05d8\u05d5\u05e2\u05df ...";
                case 10:
                    return "\u05e1\u05d2\u05d5\u05e8";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "\u05e9\u05dc\u05d7 \u05d0\u05d9\u05de\u05d9\u05d9\u05dc";
                case 13:
                    return "\u05e9\u05d9\u05d7\u05ea \u05e6'\u05d0\u05d8 \u05d0\u05d9\u05ea\u05e0\u05d5";
                case 14:
                    return "\u05d4\u05e7\u05dc\u05d3 \u05d0\u05ea \u05d4\u05d4\u05d5\u05d3\u05e2\u05d4 \u05e9\u05dc\u05da \u05db\u05d0\u05df \u05d5\u05dc\u05d7\u05e5 [Enter] \u05db\u05d3\u05d9 \u05dc\u05e9\u05dc\u05d5\u05d7...";
                case 15:
                    return "\u05d0\u05e0\u05d9"
                }
            case "ja":
                switch (b) {
                case 0:
                    return "\u30d8\u30eb\u30d7";
                case 1:
                    return "\u3042\u306a\u305f\u306e\u30e1\u30fc\u30eb\u30a2\u30c9\u30ec\u30b9:";
                case 2:
                    return "\u3054\u8cea\u554f:";
                case 3:
                    return "\u73fe\u5728\u306e\u30da\u30fc\u30b8\u306e\u30b9\u30ca\u30c3\u30d7\u30b7\u30e7\u30c3\u30c8\u3092\u52a0\u3048\u308b";
                case 4:
                    return "\u9001\u308b";
                case 5:
                    return "\u30b5\u30dd\u30fc\u30c8\u30c1\u30fc\u30e0\u304c\u8fc5\u901f\u306b\u56de\u7b54\u3067\u304d\u308b\u3088\u3046\u3001\u30a6\u30a7\u30d6\u30b5\u30a4\u30c8\u3067\u79c1\u304c\u898b\u3066\u3044\u308b\u3082\u306e\u3092\u30b9\u30af\u30ea\u30fc\u30f3\u30b7\u30e7\u30c3\u30c8\u3067\u9001\u4fe1\u3059\u308b\u3002";
                case 6:
                    return '\u30b9\u30ca\u30c3\u30d7\u30b7\u30e7\u30c3\u30c8\u3092\u6e96\u5099\u3057\u3066\u3044\u307e\u3059<br /><br />\u6211\u3005\u306e\u30a2\u30d7\u30ec\u30c3\u30c8\u306b\u30a6\u30a7\u30d6\u30b5\u30a4\u30c8\u306e\u30b9\u30ca\u30c3\u30d7\u30b7\u30e7\u30c3\u30c8\u3092\u30b5\u30dd\u30fc\u30c8\u30c1\u30fc\u30e0\u306b\u9001\u4fe1\u3055\u305b\u308b\u305f\u3081\u306b\u3001"\u5b9f\u884c\u3059\u308b"\u3082\u3057\u304f\u306f"\u8a31\u53ef\u3059\u308b"\u3092\u30af\u30ea\u30c3\u30af\u3057\u3066\u4e0b\u3055\u3044\u3002';
                case 7:
                    return "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\u3001<br />\u3042\u306a\u305f\u306e\u30e1\u30c3\u30bb\u30fc\u30b8\u306f\u9001\u4fe1\u3055\u308c\u307e\u3057\u305f\u3002";
                case 8:
                    return "\u7533\u3057\u8a33\u3054\u3056\u3044\u307e\u305b\u3093\u3001\u3042\u306a\u305f\u306e\u30e1\u30c3\u30bb\u30fc\u30b8\u306f\u9001\u4fe1\u3055\u308c\u307e\u305b\u3093\u3067\u3057\u305f\u3002<br />\u6211\u3005\u306f\u554f\u984c\u3092\u8abf\u67fb\u3057\u3066\u3044\u307e\u3059\u3002<br />\u6642\u9593\u3092\u304a\u3044\u3066\u3082\u3046\u4e00\u5ea6\u9001\u4fe1\u3057\u3066\u4e0b\u3055\u3044\u3002";
                case 9:
                    return "\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9...";
                case 10:
                    return "\u9589\u3058\u308b";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "\u30e1\u30fc\u30eb\u3059\u308b";
                case 13:
                    return "\u30c1\u30e3\u30c3\u30c8\u3059\u308b";
                case 14:
                    return "\u3053\u3053\u306b\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u5165\u529b\u3057\u3001\u9001\u4fe1\u3059\u308b\u306b\u306f[enter]\u3092\u62bc\u3057\u3066\u304f\u3060\u3055\u3044\u3002";
                case 15:
                    return "\u304a\u5ba2\u69d8"
                }
            case "ko":
                switch (b) {
                case 0:
                    return "\uace0\uac1d \uc9c0\uc6d0";
                case 1:
                    return "\uc774\uba54\uc77c \uc8fc\uc18c:";
                case 2:
                    return "\ubb38\uc81c\uc810 \ub610\ub294 \uc758\uacac:";
                case 3:
                    return "\ud604\uc7ac \ud398\uc774\uc9c0\ub97c \ucea1\ucc98\ud574 \ucca8\ubd80";
                case 4:
                    return "\ubcf4\ub0b4\uae30";
                case 5:
                    return "\uc9c0\uae08 \ubcf4\uace0 \uc788\ub294 \uc6f9\ud398\uc774\uc9c0\ub97c \ucea1\ucc98\ud574 \ubcf4\ub0b4\uba74 \uc9c0\uc6d0\ud300\uc774 \ub354\uc6b1 \ube60\ub974\uace0 \uc815\ud655\ud558\uac8c \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc2b5\ub2c8\ub2e4.";
                case 6:
                    return '\ud654\uba74 \ucea1\ucc98 \uc900\ube44\uc911...<br /><br />"Run"\uc774\ub098 "Allow" \ub4f1\uc744 \ud074\ub9ad\ud574\uc57c \ud654\uba74\uc744 \ucea1\ucc98\ud574\uc11c \uc9c0\uc6d0\ud300\uc5d0\uac8c \ubcf4\ub0bc \uc218 \uc788\uc2b5\ub2c8\ub2e4.';
                case 7:
                    return "\uace0\ub9d9\uc2b5\ub2c8\ub2e4.<br />\uc791\uc131\ud55c \ub0b4\uc6a9\uc774 \uc804\uc1a1\ub410\uc2b5\ub2c8\ub2e4.";
                case 8:
                    return "\uc8c4\uc1a1\ud569\ub2c8\ub2e4\ub9cc, \uc791\uc131\ud55c \ub0b4\uc6a9\uc744 \uc804\uc1a1\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.<br />\uc804\uc1a1\ud558\uc9c0 \ubabb\ud55c \uc6d0\uc778\uc740 \ub0b4\ubd80\uc801\uc73c\ub85c \ub354 \ud655\uc778\ud574\ubcf4\uaca0\uc2b5\ub2c8\ub2e4.<br />\uc7a0\uc2dc \ud6c4\uc5d0 \ub2e4\uc2dc \uc2dc\ub3c4\ud574\uc8fc\uc138\uc694.";
                case 9:
                    return "\ud30c\uc77c \uc62c\ub9ac\ub294 \uc911...";
                case 10:
                    return "\ub2eb\uae30";
                case 11:
                    return "\uc774\ubbf8 \ud574\ub2f9 \uac74\uc744 \uc9c0\uc6d0\ud300\uc5d0\uac8c \uc804\uc1a1\ud588\uc2b5\ub2c8\ub2e4.";
                case 12:
                    return "\uc774\uba54\uc77c \ubcf4\ub0b4\uae30";
                case 13:
                    return "\uc9c0\uc6d0\ud300\uacfc \uba54\uc2e0\uc800";
                case 14:
                    return "\uc5ec\uae30\uc5d0 \ub300\ud654 \ub0b4\uc6a9\uc744 \uc785\ub825\ud558\uace0 [\uc5d4\ud130]\ud0a4\ub97c \ub204\ub974\uba74 \ub300\ud654\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4...";
                case 15:
                    return "\ubcf8\uc778"
                }
            case "lt":
                switch (b) {
                case 0:
                    return "Pagalba";
                case 1:
                    return "J\u016bs\u0173 el. pa\u0161tas:";
                case 2:
                    return "Apra\u0161ykite savo problem\u0105:";
                case 3:
                    return "Prid\u0117ti esamo ekrano nuotrauk\u0105";
                case 4:
                    return "Si\u0173sti";
                case 5:
                    return "Nusi\u0173sti dabartinio nar\u0161ykl\u0117je matomo vaizdo nuotrauk\u0105, kad IT komanda gal\u0117t\u0173 lengviau ir grei\u010diau identifikuoti problem\u0105.";
                case 6:
                    return 'Ruo\u0161iama ekrano nuotrauka..<br/><br/>Paspauskite "Run" arba "Allow" leisdama m\u016bs\u0173 \u012fskiepiui nusi\u0173sti nar\u0161ykl\u0117s ekrano nuotrauk\u0105 tech. pagalbos komandai.';
                case 7:
                    return "D\u0117kojame,<br />J\u016bs\u0173 \u017einut\u0117 buvo i\u0161si\u0173sta.";
                case 8:
                    return "Atsipra\u0161ome, J\u016bs\u0173 \u017einut\u0117 nebuvo nusi\u0173sta.<br />\u0160iuo metu ie\u0161kome kur gal\u0117jo b\u016bti problema.<br />Pra\u0117jus \u0161iek tiek laiko, pabandykite dar kart\u0105.";
                case 9:
                    return "Duomenys siun\u010diami...";
                case 10:
                    return "U\u017everti";
                case 11:
                    return "J\u016bs jau esate prane\u0161\u0119s apie \u0161i\u0105 problem\u0105.";
                case 12:
                    return "Para\u0161ykite mums el";
                case 13:
                    return "Para\u0161ykite";
                case 14:
                    return "Ra\u0161ykite savo \u017einut\u0119 \u010dia ir spauskite [enter] nor\u0117dami j\u0105 i\u0161si\u0173sti...";
                case 15:
                    return "man"
                }
            case "lv":
                switch (b) {
                case 0:
                    return "Pal\u012bdz\u012bba";
                case 1:
                    return "J\u016bsu e-pasts:";
                case 2:
                    return "Uzrakstiet savu jaut\u0101jumu:";
                case 3:
                    return "Pievienot eso\u0161\u0101s lapas ekr\u0101nuz\u0146\u0113mumu";
                case 4:
                    return "Nos\u016bt\u012bt";
                case 5:
                    return "Nos\u016bt\u012bt ekr\u0101nuz\u0146\u0113mumu, ko es redzu weblap\u0101, lai atbalsta grupa var nodro\u0161in\u0101t \u0101tru atbildi.";
                case 6:
                    return 'Sagatavo\u0161an\u0101s momentuz\u0146\u0113mumam...<br /><br />L\u016bdzu, noklik\u0161\u0137iniet uz "Run" vai "At\u013caut", lai \u013cautu m\u016bsu s\u012bklietotnei nos\u016bt\u012bt weblapas momentuz\u0146\u0113mumu atbalsta grupai.';
                case 7:
                    return "Paldies, J\u016bsu zi\u0146a tika nos\u016bt\u012bta.";
                case 8:
                    return "Atvainojiet, J\u016bsu v\u0113stuli nevar nos\u016bt\u012bt. <br /> M\u0113s \u0161obr\u012bd izmekl\u0113jam \u0161o lietu. <br /> L\u016bdzu, m\u0113\u0123iniet v\u0113lreiz v\u0113l\u0101k.";
                case 9:
                    return "Aug\u0161upiel\u0101de...";
                case 10:
                    return "Aizv\u0113rt";
                case 11:
                    return "J\u016bs jau esat iesniedzis \u0161o lietu atbalsta komandai.";
                case 12:
                    return "Nos\u016bti mums e-pastu";
                case 13:
                    return "S\u0101kt t\u0113rz\u0113\u0161anu";
                case 14:
                    return "Ierakstiet savu zi\u0146u \u0161eit, un nospiediet [Enter], lai nos\u016bt\u012btu...";
                case 15:
                    return "Es"
                }
            case "nb":
                switch (b) {
                case 0:
                    return "Hjelp";
                case 1:
                    return "Din epostadresse:";
                case 2:
                    return "Ditt sp\u00f8rsm\u00e5l:";
                case 3:
                    return "Legg ved et skjermbilde av denne siden";
                case 4:
                    return "Send inn";
                case 5:
                    return "Legg ved et skjermbilde av det jeg ser p&#229; websiden, slik at de som skal yte brukerst&#248;tte kan gi raskere og bedre svar.";
                case 6:
                    return 'Forbereder skjermbilde...<br /><br />Om du f&#229;r sp&#248;rsm&#229;l om &#229; tillate at programmet kj&#248;rer, er det viktig at du trykker "Tillat", "Kj&#248;r" eller lignende positivt svar.';
                case 7:
                    return "Takk,<br />Meldingen din har blitt sendt.";
                case 8:
                    return "Beklager, men meldingen kunne ikke sendes.<br />Vi fors&#248;ker &#229; rette feilen.<br />V&#230;r vennlig &#229; fors&#248;k igjen senere.";
                case 9:
                    return "Laster opp...";
                case 10:
                    return "Lukk";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Send inn";
                case 13:
                    return "Chat med oss";
                case 14:
                    return "Skriv din melding her og trykk [Enter]...";
                case 15:
                    return "jeg"
                }
            case "nl":
                switch (b) {
                case 0:
                    return "Help";
                case 1:
                    return "Uw e-mailadres:";
                case 2:
                    return "Uw vraag:";
                case 3:
                    return "Voeg een screenshot van de pagina toe";
                case 4:
                    return "Verzenden";
                case 5:
                    return "Verstuur een screenshot van de webpagina zodat het support team u snel kan helpen.";
                case 6:
                    return 'Screenshot wordt voorbereid...<br /><br />Kies a.u.b. "Run" of "Allow" zodat een screenshot kan worden verzonden naar het support team.';
                case 7:
                    return "Bedankt,<br />Uw bericht is verzonden.";
                case 8:
                    return "Excuses, uw bericht kon niet verzonden worden.<br />Wij zullen het probleem onderzoeken.<br />Probeert u het later nog eens.";
                case 9:
                    return "Bezig met versturen...";
                case 10:
                    return "Sluiten";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Email ons";
                case 13:
                    return "Chat met ons";
                case 14:
                    return "Typ hier uw bericht en druk op [enter] om het te verzenden\u2026";
                case 15:
                    return "ik"
                }
            case "nn":
                switch (b) {
                case 0:
                    return "Hjelp";
                case 1:
                    return "Di e-postadresse:";
                case 2:
                    return "Ditt sp\u00f8rsm\u00e5l:";
                case 3:
                    return "Legg ved eit skjermbilete av denne sida";
                case 4:
                    return "Send inn";
                case 5:
                    return "Legg ved eit skjermbilete av det eg ser p\u00e5 websida, slik at dei som skal yte brukarst\u00f8tte kan gje raskare og betre svar.";
                case 6:
                    return 'F\u00f8rebur skjermbilete...<br /><br />Om du f\u00e5r sp\u00f8rsm\u00e5l om \u00e5 tillate atprogrammet k\u00f8yrer, er det viktig at du trykker "Tillat", "K\u00f8yr" ellerlignande positivt svar.';
                case 7:
                    return "Takk,<br />Meldinga di har blitt sendt.";
                case 8:
                    return "Beklager, men meldingen kunne ikkje sendast.<br />Vi fors\u00f8ker \u00e5 rettefeilen.<br />V\u00e6r venleg \u00e5 fors\u00f8ke igjen seinare.";
                case 9:
                    return "Lastar opp...";
                case 10:
                    return "Lat att";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Send inn";
                case 13:
                    return "Chat med oss";
                case 14:
                    return "Skriv di melding her og trykk [Enter]...";
                case 15:
                    return "Eg"
                }
            case "pl":
                switch (b) {
                case 0:
                    return "Pomoc";
                case 1:
                    return "Adres email:";
                case 2:
                    return "Tre\u015b\u0107 pytania:";
                case 3:
                    return "Za\u0142\u0105cz zrzut ekranu bie\u017c\u0105cej strony";
                case 4:
                    return "Wy\u015blij";
                case 5:
                    return "Prosz\u0119 wys\u0142a\u0107 zrzut ekranu bie\u017c\u0105cej strony, aby otrzyma\u0107 jak najszybsz\u0105 odpowied\u017a.";
                case 6:
                    return "Przygotowanie do zrobienia zrzutu ekranu...<br /><br />Aplikacja Java przetwarza tre\u015b\u0107 wiadomo\u015bci.";
                case 7:
                    return "Dzi\u0119kujemy,<br />Wiadomo\u015b\u0107 zosta\u0142a wys\u0142ana.";
                case 8:
                    return "Przepraszamy, wiadomo\u015b\u0107 nie zosta\u0142a wys\u0142ana.<br />Nasz zesp\u00f3\u0142 analizuje przyczyn\u0119 problemu.<br />Prosz\u0119 spr\u00f3bowa\u0107 ponownie p\u00f3\u017aniej.";
                case 9:
                    return "\u0141adowanie...";
                case 10:
                    return "Zamknij";
                case 11:
                    return "Ju\u017c wys\u0142a\u0142e\u015b to zg\u0142oszenie do wsparcia.";
                case 12:
                    return "Wy\u015blij email";
                case 13:
                    return "Wejd\u017a na chat";
                case 14:
                    return "Prosz\u0119 wpisa\u0107 tre\u015b\u0107 pytania i wcisn\u0105\u0107 enter.";
                case 15:
                    return "Ja"
                }
            case "pt":
                switch (b) {
                case 0:
                    return "Ajuda";
                case 1:
                    return "Seu e-mail:";
                case 2:
                    return "Seu coment&aacute;rio:";
                case 3:
                    return "Inclua uma captura de tela da p&aacute;gina atual";
                case 4:
                    return "Envie";
                case 5:
                    return "Envie a captura de tela daquilo que eu estou vendo na p&aacute;gina da web para que o time de suporte possa fornecer uma resposta r&aacute;pida.";
                case 6:
                    return 'Por favor, clique "Run" ou "Allow" para permitir nosso applet enviar uma captura de tela da p&aacute;gina da web para o time de suporte.';
                case 7:
                    return "Obrigado,<br />sua mensagem foi enviada.";
                case 8:
                    return "Desculpe, sua mensagem n&atilde;o pode ser enviada. N&oacute;s estamos investigando o problema. Por favor, tente novamente mais tarde.";
                case 9:
                    return "Carregando...";
                case 10:
                    return "Feche";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Envie-nos";
                case 13:
                    return "Converse conosco";
                case 14:
                    return "Digite sua mensagem aqui e tecle [enter] para enviar...";
                case 15:
                    return "eu"
                }
            case "ro":
                switch (b) {
                case 0:
                    return "Ajutor";
                case 1:
                    return "Email-ul dvs:";
                case 2:
                    return "\u00centrebarea dvs:";
                case 3:
                    return "Include\u021bi o captur\u0103 a paginii curente";
                case 4:
                    return "Trimte";
                case 5:
                    return "Trimite\u021bi o captur\u0103 a ceea ce vede\u021bi pe pagina web pentru ca echipa tehnic\u0103 s\u0103 v\u0103 poat\u0103 r\u0103spunde c\u00e2t mai rapid.";
                case 6:
                    return 'Preg\u0103tire pentru captur\u0103... <br /><br />V\u0103 rug\u0103m ap\u0103sa\u021bi "Run" sau "Allow" pentru a l\u0103sa modulul s\u0103 trimit\u0103 o captur\u0103 a paginii c\u0103tre echipa tehnic\u0103.';
                case 7:
                    return "Mul\u021bumim,<br />Mesajul dumneavoastr\u0103 a fost trimis.";
                case 8:
                    return "Scuze, mesajul dumneavoastr\u0103 nu a fost trimis.<br />Acum investig\u0103m problema.<br />V\u0103 rug\u0103m re\u00eencerca\u021bi mai t\u00e2rziu.";
                case 9:
                    return "Se \u00eencarc\u0103...";
                case 10:
                    return "\u00cenchide";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "email";
                case 13:
                    return "Converseaz\u0103 prin chat";
                case 14:
                    return "Scrie\u021bi mesajul dumneavoastr\u0103 aici \u0219i ap\u0103sa\u021bi [enter] pentru a-l trimite...";
                case 15:
                    return "Eu"
                }
            case "ru":
                switch (b) {
                case 0:
                    return "\u041f\u043e\u043c\u043e\u0449\u044c";
                case 1:
                    return "\u0412\u0430\u0448 email:";
                case 2:
                    return "\u0412\u0430\u0448 \u0437\u0430\u043f\u0440\u043e\u0441:";
                case 3:
                    return "\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0441\u043d\u0438\u043c\u043e\u043a \u044d\u043a\u0440\u0430\u043d\u0430 \u0442\u0435\u043a\u0443\u0449\u0435\u0439 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u044b";
                case 4:
                    return "\u041e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c";
                case 5:
                    return "\u041e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c \u0441\u043d\u0438\u043c\u043e\u043a \u044d\u043a\u0440\u0430\u043d\u0430 (\u0442\u043e\u0433\u043e \u0447\u0442\u043e \u044f \u0432\u0438\u0436\u0443 \u043d\u0430 \u0432\u0435\u0431-\u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0435), \u0447\u0442\u043e\u0431\u044b \u0441\u043b\u0443\u0436\u0431\u0430 \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0438 \u0441\u043c\u043e\u0433\u043b\u0430 \u0431\u044b\u0441\u0442\u0440\u043e \u043e\u0442\u0432\u0435\u0442\u0438\u0442\u044c.";
                case 6:
                    return "\u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430 \u043a \u0441\u043d\u044f\u0442\u0438\u044e \u0441\u043d\u0438\u043c\u043a\u0430 \u044d\u043a\u0440\u0430\u043d\u0430<br />\u041f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430 \u0440\u0430\u0437\u0440\u0435\u0448\u0438\u0442\u0435 Java-\u0430\u043f\u043f\u043b\u0435\u0442\u0443 \u0441\u0434\u0435\u043b\u0430\u0442\u044c \u044d\u0442\u043e \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435.";
                case 7:
                    return "\u0421\u043f\u0430\u0441\u0438\u0431\u043e,<br />\u0412\u0430\u0448\u0435 \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u043e\u0442\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u043e.";
                case 8:
                    return "\u041a \u0441\u043e\u0436\u0430\u043b\u0435\u043d\u0438\u044e, \u0412\u0430\u0448\u0435 \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u043d\u0435 \u043c\u043e\u0436\u0435\u0442 \u0431\u044b\u0442\u044c \u043e\u0442\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u043e.<br />\u041c\u044b \u0440\u0430\u0437\u0431\u0438\u0440\u0430\u0435\u043c\u0441\u044f \u0441 \u0432\u043e\u0437\u043d\u0438\u043a\u0448\u0435\u0439 \u043f\u0440\u043e\u0431\u043b\u0435\u043c\u043e\u0439.<br />\u041f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430, \u043f\u043e\u0432\u0442\u043e\u0440\u0438\u0442\u0435 \u043f\u043e\u043f\u044b\u0442\u043a\u0443 \u043f\u043e\u0437\u0436\u0435.";
                case 9:
                    return "\u0418\u0434\u0435\u0442 \u0437\u0430\u0433\u0440\u0443\u0437\u043a\u0430...";
                case 10:
                    return "\u0417\u0430\u043a\u0440\u044b\u0442\u044c";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "\u041e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c";
                case 13:
                    return "\u0427\u0430\u0442 \u0441\u043e \u0441\u043f\u0435\u0446\u0438\u0430\u043b\u0438\u0441\u0442\u043e\u043c";
                case 14:
                    return "\u0412\u0432\u0435\u0434\u0438\u0442\u0435 \u0441\u0432\u043e\u0435 \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u0437\u0434\u0435\u0441\u044c \u0438 \u043d\u0430\u0436\u043c\u0438\u0442\u0435 [enter] \u0434\u043b\u044f \u043e\u0442\u043f\u0440\u0430\u0432\u043a\u0438...";
                case 15:
                    return "\u042f"
                }
            case "sk":
                switch (b) {
                case 0:
                    return "Pomoc";
                case 1:
                    return "V&#225;&#353; email:";
                case 2:
                    return "Pop&#237;&#353;te Va&#353;u ot&#225;zku:";
                case 3:
                    return "Pripoji&#357; snapshot aktu&#225;lnej str&#225;nky";
                case 4:
                    return "Po&#353;li";
                case 5:
                    return "Po&#353;li fotku toho &#269;o vid&#237;m na str&#225;nke aby mohla na&#353;a podpora poskytn&#250;&#357; r&#253;chlu odpove&#271;.";
                case 6:
                    return "Pripravujeme fotku...<br/><br/>Pros&#237;me umo&#382;nite appletu n&#225;m posla&#357; fotku aktu&#225;lnej web str&#225;nky.";
                case 7:
                    return "&#270;akujeme V&#225;m!<br/>Va&#353;a spr&#225;va bola poslan&#225;.";
                case 8:
                    return "Prep&#225;&#269;te, Va&#353;a spr&#225;va nebola poslan&#225;.<br/>Rie&#353;ime vyskytnut&#253; probl&#233;m.<br/>Pros&#237;me pok&#250;ste sa nesk&#244;r znova.";
                case 9:
                    return "Nahr&#225;va sa...";
                case 10:
                    return "Zatvori&#357;";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Po\u0161li email";
                case 13:
                    return "Chatuj s nami";
                case 14:
                    return "Nap\u00ed\u0161te svoj dotaz a stla\u010dte [Enter]...";
                case 15:
                    return "ja"
                }
            case "sv":
                switch (b) {
                case 0:
                    return "Hj\u00e4lp";
                case 1:
                    return "Din emailadress:";
                case 2:
                    return "Vad kan vi hj\u00e4lpa till med?";
                case 3:
                    return "Bifoga sk&auml;rmdump av aktuell webbsida";
                case 4:
                    return "Skicka";
                case 5:
                    return "Skicka en sk&auml;rmdump av det jag ser p&aring; webbsidan s&aring; att supportteamet l&auml;ttare kan ge ett snabbt svar.";
                case 6:
                    return 'F&ouml;rbereder en sk&auml;rmdump...<br /><br />V&auml;nligen klicka p&aring; "K&ouml;r" alternativt "Till&aring;t" s&aring; att v&aring;r applet ta en sk&auml;rmdump och skicka den till supportteamet.';
                case 7:
                    return "Tack s&aring; mycket,<br />ditt meddelande har skickats.";
                case 8:
                    return "Tyv&auml;rr, ditt meddelande gick inte att skicka.<br />Vi unders&ouml;ker problemet.<br />V&auml;nligen f&ouml;rs&ouml;k igen vid ett senare tillf&auml;lle.";
                case 9:
                    return "Laddar upp...";
                case 10:
                    return "St&auml;ng";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Skicka ett email";
                case 13:
                    return "Chatta med oss";
                case 14:
                    return "Fyll i ditt meddelande h\u00e4r och tryck p\u00e5 [enter] f\u00f6r att skicka...";
                case 15:
                    return "Jag"
                }
            case "tr":
                switch (b) {
                case 0:
                    return "Yard&#305;m";
                case 1:
                    return "E-posta Adresiniz:";
                case 2:
                    return "Sorununuzun detaylar&#305;n&#305; yaz&#305;n:";
                case 3:
                    return "&#350;u an bulundu&#287;um sayfan&#305;n g&#246;r&#252;n&#252;m&#252;n&#252; de ekle";
                case 4:
                    return "G&#246;nder";
                case 5:
                    return "&#350;u anda siz ne g&#246;r&#252;yorsan&#305;z ekranda destek eleman&#305;n&#305;n da ayn&#305; g&#246;r&#252;nt&#252;y&#252; g&#246;rebilmesi ve size daha h&#305;zl&#305; geri d&#246;nmesini sa&#287;lar.";
                case 6:
                    return 'Ekran g\u00f6r\u00fcn\u00fcm\u00fc haz\u0131rlan\u0131yor...<br /><br />L\u00fctfen "\u00c7al\u0131\u015ft\u0131r" veya "\u0130zin Ver" tu\u015flar\u0131ndan birine bas\u0131n b\u00f6ylece yaz\u0131l\u0131m\u0131m\u0131z ekran g\u00f6r\u00fcnt\u00fcs\u00fcn\u00fc al\u0131p destek personeline g\u00f6nderebilecek.';
                case 7:
                    return "Te&#351;ekk&#252;r ederiz,<br />Mesaj&#305;n&#305;z g&#246;nderildi.";
                case 8:
                    return "&#214;z&#252;r dileriz fakat mesaj&amp;#305;n&amp;#305;z g&#246;nderilemedi.<br />&amp;#350;u anda bu konu ile ilgili hata g&#246;nderildi inceliyoruz<br />L&#252;tfen daha sonra tekrar deneyiniz.";
                case 9:
                    return "Y&#252;kleniyor...";
                case 10:
                    return "Kapat";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "&#304;leti&#351;im";
                case 13:
                    return "Bizimle Chat Yap&#305;n";
                case 14:
                    return "Mesajinizi yazin ve [enter] tusuna basin...";
                case 15:
                    return "ben"
                }
            case "zh":
                switch (b) {
                case 0:
                    return "\u5e2e\u52a9";
                case 1:
                    return "\u60a8\u7684\u7535\u5b50\u90ae\u4ef6";
                case 2:
                    return "\u63cf\u8ff0\u60a8\u7684\u95ee\u9898\uff1a";
                case 3:
                    return "\u5305\u62ec\u5bf9\u5f53\u524d\u7f51\u9875\u7684\u5feb\u7167";
                case 4:
                    return "\u53d1\u9001";
                case 5:
                    return "\u0017\u5bc4\u4e86\u4ec0\u4e48\uff0c\u6211\u770b\u5230\u5728\u7f51\u9875\u4e0a\uff0c\u8ba9\u652f\u6301\u5c0f\u7ec4\u53ef\u4ee5\u63d0\u4f9b\u5feb\u901f\u7684\u7b54\u6848\u5c4f\u5e55\u5feb\u7167";
                case 6:
                    return "\u51c6\u5907\u5c4f\u5e55\u6355\u83b7...<br />\u8bf7\u76f8\u4fe1Java\u5c0f\u7a0b\u5e8f";
                case 7:
                    return "\u8c22\u8c22\u60a8<br />\u60a8\u7684\u90ae\u4ef6\u5df2\u53d1\u9001";
                case 8:
                    return "\u5bf9\u4e0d\u8d77\uff0c\u60a8\u7684\u90ae\u4ef6\u65e0\u6cd5\u53d1\u9001<br />\u6211\u4eec\u6b63\u5728\u8c03\u67e5\u8fd9\u4e2a\u95ee\u9898<br />\u8bf7\u7a0d\u540e\u518d\u8bd5";
                case 9:
                    return "\u4e0a\u4f20...";
                case 10:
                    return "\u5173\u95ed";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "\u90ae\u4ef6\u8054\u7cfb";
                case 13:
                    return "\u5728\u7ebf\u804a\u5929";
                case 14:
                    return "\u5728\u8fd9\u91cc\u8f93\u5165\u540e\u6309\u56de\u8f66\u952e\u53d1\u9001";
                case 15:
                    return "me"
                }
            case "zh_TW":
                switch (b) {
                case 0:
                    return "\u6392\u89e3\u7591\u96e3";
                case 1:
                    return "\u4f60\u7684\u96fb\u90f5\u5730\u5740\uff1a";
                case 2:
                    return "\u4f60\u7684\u554f\u984c\uff1a";
                case 3:
                    return "Include a snapshot of the current page";
                case 4:
                    return "\u50b3\u9001";
                case 5:
                    return "Send a screen shot of what I am seeing on the web page so that the support team can provide a fast answer.";
                case 6:
                    return 'Preparing for snapshot...<br /><br />Please click "Run" or "Allow" to let our applet send a snapshot of the web page to the support team.';
                case 7:
                    return "\u8b1d\u8b1d\u3002<br />\u4f60\u7684\u8a0a\u606f\u5df2\u9001\u51fa";
                case 8:
                    return "\u5c0d\u4e0d\u8d77\uff0c\u4f60\u7684\u8a0a\u606f\u50b3\u9001\u767c\u751f\u932f\u8aa4\u3002<br />\u6211\u5011\u6b63\u5728\u8abf\u67e5\u4e2d,<br />\u8acb\u7a0d\u5f8c\u518d\u8a66\u3002";
                case 9:
                    return "\u4e0a\u50b3\u4e2d\u2026...";
                case 10:
                    return "\u95dc\u9589";
                case 11:
                    return "\u4f60\u7684\u500b\u6848\u5df2\u7d93\u63d0\u4ea4\u5230\u652f\u63f4\u5c0f\u7d44";
                case 12:
                    return "\u4ee5\u96fb\u90f5\u806f\u7d61\u6211\u5011";
                case 13:
                    return "\u804a\u5929\u5c0d\u8a71";
                case 14:
                    return "\u5728\u9019\u88e1\u8f38\u5165\u5f8c\u6309\u4e0b Enter \u9375\u767c\u9001";
                case 15:
                    return "\u6211"
                }
            default:
                switch (b) {
                case 0:
                    return "Help";
                case 1:
                    return "Your email:";
                case 2:
                    return "Your question:";
                case 3:
                    return "Include a snapshot of the current page";
                case 4:
                    return "Send";
                case 5:
                    return "Send a screen shot of what I am seeing on the web page so that the support team can provide a fast answer.";
                case 6:
                    return 'Preparing for snapshot...<br /><br />Please click "Run" or "Allow" to let our applet send a snapshot of the web page to the support team.';
                case 7:
                    return "Thank you,<br />Your message has been sent.";
                case 8:
                    return "Sorry, your message couldn't be sent.<br />We are investigating the issue.<br />Please try again later.";
                case 9:
                    return "Uploading...";
                case 10:
                    return "Close";
                case 11:
                    return "You have already submitted this case to the support team.";
                case 12:
                    return "Email us";
                case 13:
                    return "Chat with us";
                case 14:
                    return "Type your message here and press [enter] to send...";
                case 15:
                    return "me"
                }
            }
            return "Message not found. msgId: " + b
        },
        prepareEventHandling: function() {
            document.addEventListener ? (SnapABug.addEvent = function(b, a, c) {
                b.addEventListener(a, c, !1);
                return c
            }, SnapABug.removeEvent = function(b, a, c) {
                b.removeEventListener(a, c, !1)
            }) : document.attachEvent && (SnapABug.addEvent =
            function(b, a, c) {
                var e = function() {
                    return c.apply(b, arguments)
                };
                b.attachEvent("on" + a, e);
                return e
            }, SnapABug.removeEvent = function(b, a, c) {
                b.detachEvent("on" + a, c)
            })
        },
        sendGetProactiveGeoData: function() {
            if (!lb) {
                lb=!0;
                var b = "w=" + encodeURIComponent(d);
                this.rpc(f + "/snapabug/servicegetproactivegeodata?" + b, 3, !0)
            }
        },
        setProactiveChatEventListener: function(b, a, c) {
            c = [c];
            switch (b.toLowerCase()) {
            case "scroll":
                SnapABug.addEvent(window, "scroll", function(b) {
                    SnapABug.evaluateProactiveRule("intrule", a, (window.pageYOffset ||
                    document.documentElement.scrollyPos) + (window.innerHeight || document.documentElement.offsetHeight)) && SnapABug.proactiveChatEvaluateCallback(!0, c)
                });
                break;
            case "click":
                SnapABug.addEvent(document, "click", function() {
                    gb++;
                    SnapABug.evaluateProactiveRule("intrule", a, gb) && SnapABug.proactiveChatEvaluateCallback(!0, c)
                });
                break;
            case "exit":
                SnapABug.addEvent(document, "mouseout", function(b) {
                    b = b ? b : window.event;
                    var a = b.relatedTarget || b.toElement;
                    b = b.pageY || b.clientY;
                    a && "HTML" != a.nodeName||-1 != b && 150 > b && SnapABug.proactiveChatEvaluateCallback(!0,
                    c)
                })
            }
        },
        prepareForRegexMatch: function(b) {
            return b.replace(/[-[\]{}()+?.,\\^$|#\s]/g, "\\$&").replace(/\*+/g, ".*")
        },
        evaluateProactiveRule: function(b, a, c) {
            switch (b.toLowerCase()) {
            case "stringrule":
                switch (a.operator.toLowerCase()) {
                case "equals":
                    return a = new RegExp("^" + SnapABug.prepareForRegexMatch(decodeURIComponent(a.value)) + "$"), a.test(c);
                case "notequals":
                    return a = new RegExp("^" + SnapABug.prepareForRegexMatch(decodeURIComponent(a.value)) + "$"), !a.test(c);
                case "contains":
                    return a = new RegExp(SnapABug.prepareForRegexMatch(decodeURIComponent(a.value))),
                    a.test(c);
                case "notcontains":
                    return a = new RegExp(SnapABug.prepareForRegexMatch(decodeURIComponent(a.value))), !a.test(c);
                case "beginswith":
                    return a = new RegExp("^" + SnapABug.prepareForRegexMatch(decodeURIComponent(a.value))), a.test(c);
                case "endwith":
                    return a = new RegExp(SnapABug.prepareForRegexMatch(decodeURIComponent(a.value)) + "$"), a.test(c);
                case "matchregex":
                    return a = new RegExp(decodeURIComponent(a.value)), a.test(c)
                }
                break;
            case "intrule":
                switch (a.operator.toLowerCase()) {
                case "equals":
                    return c == a.value;
                case "notequals":
                    return c != a.value;
                case "greater":
                    return c > a.value;
                case "less":
                    return c < a.value;
                case "greaterequal":
                    return c >= a.value;
                case "lessequal":
                    return c <= a.value
                }
                break;
            case "boolrule":
                return SnapABug.booleanToString(c) == SnapABug.booleanToString(a.value);
            case "timeofdayrule":
                b = new Date(a.toEpoch);
                var e = new Date(a.fromEpoch);
                c = new Date(c);
                for (var d=!1, m=!1, f = a.days.length; 0 <= f; f--)
                    a.days[f] == c.getUTCDay() && (m=!0);
                if (!m)
                    return !1;
                e.getUTCHours() === c.getUTCHours() && e.getUTCMinutes() <= c.getUTCMinutes() &&
                (d=!0);
                c.getUTCHours() === b.getUTCHours() && c.getUTCMinutes() <= b.getUTCMinutes() && (d=!0);
                if (d)
                    return !0;
                b.getUTCHours() < e.getUTCHours() ? e.getUTCHours() < c.getUTCHours() ? d=!0 : c.getUTCHours() < b.getUTCHours() && (d=!0) : e.getUTCHours() < c.getUTCHours() && c.getUTCHours() < b.getUTCHours() && (d=!0);
                return d
            }
        },
        booleanToString: function(b) {
            return "boolean" === typeof b ? b.toString() : b
        },
        getSearchParamsFromUrl: function(b) {
            for (var a = [], c = ["google", "bing", "ask", "msn", "yahoo"], e = "q", d=!1, m = 0; m < c.length; m++) 
                - 1 != b.indexOf(c[m]) &&
                (d=!0, "yahoo" === c[m] && (e = "p"));
            if (d)
                for (c = b.slice(b.indexOf("?") + 1).split("&"), m = 0; m < c.length; m++)
                    b = c[m].split("="), a.push(b[0]), a[b[0]] = b[1];
            return void 0 !== a[e] ? decodeURIComponent(a[e].replace(/\+/g, " ")) : ""
        },
        setOpenProactiveChatAPIRule: function(b) {
            null !== pa || SnapABug.isRuleSetAgentRouted(b) || (pa = b)
        },
        proactiveChatRuleEvalStep1: function() {
            var b, a, c, e, d, m, f=!1, l=!0;
            ea = 1;
            var f=!1, g = E.length;
            for (; 0 < g--;) {
                xa || (xa=!0);
                b = E[g];
                a = b.operator;
                c = b.pageRules;
                var n = c.length;
                b:
                for (; 0 < n--;) {
                    e = c[n];
                    m=!1;
                    switch (e.name.toLowerCase()) {
                    case "pageurl":
                        d =
                        SnapABug.proactiveUrlEvaluation(e.ruleProperty, window.location.href);
                        break;
                    case "entryurl":
                        d = SnapABug.proactiveUrlEvaluation(e.ruleProperty, SnapABug.getEntryPageUrl());
                        break;
                    case "comingfrom":
                        d = SnapABug.proactiveUrlEvaluation(e.ruleProperty, SnapABug.getEntryPageRef());
                        break;
                    case "keywords":
                        d = SnapABug.evaluateProactiveRule("stringrule", e.ruleProperty, SnapABug.getSearchParamsFromUrl(SnapABug.getEntryPageRef()));
                        break;
                    case "jsvar":
                        d = SnapABug.evaluateProactiveRule("stringrule", e.ruleProperty, SnapABug.getJSVarValue(e.ruleProperty.jsVar));
                        break;
                    case "newvisitor":
                        d = SnapABug.evaluateProactiveRule("boolrule", e.ruleProperty, SnapABugChat.getVisitorIsFirstTimeVisiting());
                        l=!1;
                        break;
                    case "chattedbefore":
                        d = SnapABug.evaluateProactiveRule("boolrule", e.ruleProperty, 0 < SnapABugChat.getVisitorNumberOfPreviousChats());
                        l=!1;
                        break;
                    case "pagesvisited":
                        d = SnapABug.evaluateProactiveRule("intrule", e.ruleProperty, SnapABugChat.getVisitorNumberOfPagesVisited());
                        break;
                    case "lang":
                    case "timeofday":
                    case "country":
                    case "state":
                    case "city":
                        f=!0;
                    default:
                        m=!0
                    }
                    if (!m)
                        if ("any" ===
                        a) {
                            if (d) {
                                l && SnapABug.setOpenProactiveChatAPIRule(b);
                                SnapABug.openProactiveV2Chat(g);
                                return 
                            }
                        } else if (d)
                            if (1 < c.length)
                                E[g].pageRules.splice(n, 1);
                            else {
                                l && SnapABug.setOpenProactiveChatAPIRule(b);
                                SnapABug.openProactiveV2Chat(g);
                                return 
                            } else {
                                E.splice(g, 1);
                                break b
                            }
                }
            }
            f ? SnapABug.sendGetProactiveGeoData() : SnapABug.proactiveChatRuleEvalStep3()
        },
        proactiveUrlEvaluation: function(b, a) {
            if (!a ||!b)
                return !1;
            var c = SnapABug.evaluateProactiveRule("stringrule", b, a);
            c || (c = "", c = "/" === a.charAt(a.length - 1) ? a.substring(0, a.length -
            1) : a + "/", c = SnapABug.evaluateProactiveRule("stringrule", b, c));
            return c
        },
        callbackGetProactiveGeoData: function(b, a, c, e, d) {
            na = b;
            ib = a;
            jb = c;
            hb = 1E3 * e;
            kb = d;
            SnapABug.proactiveChatRuleEvalStep2()
        },
        proactiveChatRuleEvalStep2: function() {
            var b, a, c, e, d, m;
            ea = 2;
            var f = E.length;
            for (; 0 < f--;) {
                b = E[f];
                a = b.operator;
                c = b.pageRules;
                var l = c.length;
                b:
                for (; 0 < l--;) {
                    e = c[l];
                    m=!1;
                    switch (e.name.toLowerCase()) {
                    case "timeofday":
                        d = SnapABug.evaluateProactiveRule("timeofdayrule", e.ruleProperty, hb);
                        break;
                    case "country":
                        d = SnapABug.evaluateProactiveRule("stringrule",
                        e.ruleProperty, na);
                        break;
                    case "state":
                        d = SnapABug.evaluateProactiveRule("stringrule", e.ruleProperty, ib);
                        break;
                    case "city":
                        d = SnapABug.evaluateProactiveRule("stringrule", e.ruleProperty, jb);
                        break;
                    case "lang":
                        d = SnapABug.evaluateProactiveRule("stringrule", e.ruleProperty, kb);
                        break;
                    default:
                        m=!0
                    }
                    if (!m)
                        if ("any" === a) {
                            if (d) {
                                SnapABug.setOpenProactiveChatAPIRule(b);
                                SnapABug.openProactiveV2Chat(f);
                                return 
                            }
                        } else if (d)
                            if (1 < c.length)
                                E[f].pageRules.splice(l, 1);
                            else {
                                SnapABug.setOpenProactiveChatAPIRule(b);
                                SnapABug.openProactiveV2Chat(f);
                                return 
                            } else {
                                E.splice(f, 1);
                                break b
                            }
                }
            }
            SnapABug.proactiveChatRuleEvalStep3()
        },
        proactiveChatRuleEvalStep3: function() {
            function b(b, a) {
                null === b.timer || b.timer > a.ruleProperty.value ? (b.eventIDList = [a.eventID], b.timer = a.ruleProperty.value) : b.timer === a.ruleProperty.value && b.eventIDList.push(a.eventID);
                return b
            }
            var a, c, e, d = E.length;
            3 > ea && (SnapABug.prepareEventHandling(), firstPass=!0, ea = 3);
            var m = {
                timer: null,
                eventIDList: []
            }, f = {
                timer: null,
                eventIDList: []
            };
            for (; 0 < d--;) {
                a = E[d];
                c = a.pageRules;
                var l = c.length, g=!0;
                for (; 0 <
                l--;) {
                    e = c[l];
                    firstPass && (e.eventID = "eventID" + d + l);
                    switch (e.name.toLowerCase()) {
                    case "visitorclicks":
                        firstPass && (SnapABug.setProactiveChatEventListener("click", e.ruleProperty, e.eventID), g=!1);
                        break;
                    case "scrolldown":
                        firstPass && (SnapABug.setProactiveChatEventListener("scroll", e.ruleProperty, e.eventID), g=!1);
                        break;
                    case "visitorexits":
                        firstPass && (SnapABug.setProactiveChatEventListener("exit", e.ruleProperty, e.eventID), g=!1);
                        break;
                    case "timesite":
                        m = b(m, e);
                        break;
                    case "timepage":
                        f = b(f, e)
                    }
                    firstPass && g && 0 ===
                    l && SnapABug.setOpenProactiveChatAPIRule(a)
                }
            }
            SnapABug.executeProactiveChatEvaluationFinishedCallback();
            m.timer && window.setTimeout(function() {
                SnapABug.proactiveChatEvaluateCallback(!0, m.eventIDList)
            }, 1E3 * (m.timer - SnapABugChat.getVisitorTimeOnSite()));
            f.timer && window.setTimeout(function() {
                SnapABug.proactiveChatEvaluateCallback(!0, f.eventIDList)
            }, 1E3 * (f.timer - SnapABugChat.getVisitorTimeOnPage()))
        },
        proactiveChatEvaluateCallback: function(b, a) {
            var c, e, d = E.length;
            for (; 0 < d--;) {
                c = E[d];
                e = c.pageRules;
                c = c.operator;
                var m = e.length;
                for (; 0 < m--;)
                    SnapABug.isInArray(e[m].eventID, a) && ("any" === c ? b && SnapABug.openProactiveV2Chat(d) : b ? 1 < e.length ? E[d].pageRules.splice(m, 1) : SnapABug.openProactiveV2Chat(d) : E.splice(d, 1))
            }
        },
        isInArray: function(b, a) {
            if (b && a)
                for (var c = 0; c < a.length; c++)
                    if (a[c] == b)
                        return !0;
            return !1
        },
        openProactiveV2Chat: function(b) {
            var a = E[b];
            ga || (ga=!0, SnapABug.isRuleSetAgentRouted(a) ? (SnapABug.setPreferredAgentList(b), SnapABug.openProactiveChatWithAgentRouting(b)) : SnapABug.openProactiveChatDo(!1, !1, decodeURIComponent(a.message),
            b))
        },
        isRuleSetAgentRouted: function(b) {
            return b.advanced.agents && b.advanced.agents instanceof Array?!0 : !1
        },
        setPreferredAgentList: function(b) {
            b = E[b];
            for (var a = 0; a < b.advanced.agents.length; a++)
                ya.push(b.advanced.agents[a])
        },
        openProactiveChatWithAgentRouting: function(b) {
            var a = E[b];
            fa = b;
            "1" == SnapABugChat.getBannedCookie() ||!ka || r || SnapABugChat.isNoProactiveChatCookie() || (SnapABugChat.reInit(), S = decodeURIComponent(a.message), SnapABug.sendGetAvailableAgentList())
        },
        sendGetAvailableAgentList: function() {
            var b =
            "w=" + encodeURIComponent(d);
            this.rpc(f + "/snapabug/servicegetallavailableagents?" + b, 3, !0)
        },
        callbackGetAvailableAgentList: function(b) {
            if (b) {
                b = b.split("&");
                for (var a = 0; a < b.length; a++) {
                    var c = b[a].split(". ");
                    oa[c[0]] = c[1]
                }
            }
            for (a = 0; a < ya.length; a++)
                for (var e in oa)
                    if (oa[e] == ya[a]) {
                        SnapABug.sendGetProactiveDataFromAgentNumberAndHash(oa[e], e);
                        return 
                    }
            SnapABug.proactiveChatWithAgentRoutingErrorFallback()
        },
        proactiveChatWithAgentRoutingErrorFallback: function() {
            ga=!1;
            SnapABug.clearProactiveChatEvaluationFinishedCallback();
            - 1 != fa && (E.splice(fa, 1), fa =- 1);
            oa = {};
            ya = [];
            switch (ea) {
            case 1:
                SnapABug.proactiveChatRuleEvalStep1();
                break;
            case 2:
                SnapABug.proactiveChatRuleEvalStep2();
                break;
            case 3:
                SnapABug.proactiveChatRuleEvalStep3()
            }
        },
        sendGetProactiveDataFromAgentNumberAndHash: function(b, a) {
            var c = "w=" + encodeURIComponent(d), c = c + ("&h=" + encodeURIComponent(b)), c = c + ("&nr=" + encodeURIComponent(a));
            this.rpc(f + "/snapabug/servicegetproactivedatafromnrhash?" + c, 3, !0);
            - 1 != fa && E.splice(fa, 1)
        },
        proactiveChatInitialize: function(b) {
            if (b) {
                b = eval("(" +
                b + ")");
                E = b.ruleSets.reverse();
                ruleSetArrayOperator = b.operator;
                b.proactiveDisableTime && SnapABug.setNoProactiveChatDelay(b.proactiveDisableTime);
                if (b.newVisitorCookieLifeTime) {
                    SnapABug.setVisitorCookieLifeTime(b.newVisitorCookieLifeTime);
                    if (SnapABugChat.getVisitCookie()) {
                        var a = SnapABugChat.getVisitCookie();
                        SnapABugChat.setCookie("SnapABugVisit", a, SnapABug.getVisitorCookieLifeTime())
                    }
                    SnapABug.setVisitorCookieLifeTime(b.newVisitorCookieLifeTime)
                }
                SnapABug.proactiveChatRuleEvalStep1()
            }
        },
        getJSVarValue: function(b) {
            v =
            b;
            var a = this.getJSVariables().split("=");
            return a[0] == b ? a[1] : ""
        },
        messagesInit: function(b) {
            b = eval("(" + b + ")");
            SnapABug.messagesObj = b;
            SnapABug.messagesObj && (SnapABug.getMessage = function(b) {
                SnapABug.messagesObj.isSingle && (SnapABug.setLocale = function() {
                    return !1
                });
                var a = SnapABug.getLocale();
                SnapABug.setDirection("ar" === a || "iw" === a ? "direction:rtl;" : "");
                return SnapABug.messagesObj[a][b]
            })
        },
        setPreChatSearchEnabled: function(b) {
            this.enablePreChatSearch = b;
            !0 == b && this.setPreChatSearch(1)
        },
        getPreChatSearchEnabled: function() {
            return this.enablePreChatSearch
        },
        resetPreChatSearch: function() {
            N = "";
            SnapABug.setPreChatSearch(1)
        },
        startPreChatSearch: function() {
            function b() {
                document.getElementById("preChatSearchInstructions").innerHTML = SnapABug.getMessage(16);
                document.getElementById("preChatSearchQuestionTextArea").value = N ? N : "";
                document.getElementById("preChatSearchQuestionForm").style.display = "";
                document.getElementById("preChatSearchNextButton").setAttribute("style", "float: none; clear: none; height: 25px; margin: 0px 0px 0px 4px; cursor: pointer; outline: none; border: 1px solid rgb(160, 161, 160); background-image: -webkit-gradient(linear, 0% 0%, 0% 100%, from(rgb(255, 255, 255)), to(rgb(199, 201, 199))); padding: 2px 8px 3px 8px; width: auto; overflow: visible; border-top-left-radius: 10px; border-top-right-radius: 10px; border-bottom-right-radius: 10px; border-bottom-left-radius: 10px; color: rgb(81, 85, 93); font-size: 12px; font-family: 'lucida grande', 'Helvetica Neue', Helvetica, Arial, sans-serif; text-decoration: none; vertical-align: middle; font-weight: bold; display: inline-block;float: right;");
                document.getElementById("preChatSearchNextButton").value = SnapABug.getMessage(18);
                document.getElementById("preChatSearchSearching").style.display = "none";
                document.getElementById("preChatSearchSearching").innerHTML = "<img src='" + k + "/wbg/loading.gif'/>";
                document.getElementById("preChatSearchResultsForm").style.display = "none";
                document.getElementById("preChatSearchSearchResults").innerHTML = "";
                document.getElementById("preChatSearchBackButton").innerHTML = SnapABug.getMessage(19);
                document.getElementById("preChatSearchHelpButton").setAttribute("style",
                "float: none; clear: none; height: 25px; margin: 0px 0px 0px 4px; cursor: pointer; outline: none; border: 1px solid rgb(160, 161, 160); background-image: -webkit-gradient(linear, 0% 0%, 0% 100%, from(rgb(255, 255, 255)), to(rgb(199, 201, 199))); padding: 2px 8px 3px 8px; width: auto; overflow: visible; border-top-left-radius: 10px; border-top-right-radius: 10px; border-bottom-right-radius: 10px; border-bottom-left-radius: 10px; color: rgb(81, 85, 93); font-size: 12px; font-family: 'lucida grande', 'Helvetica Neue', Helvetica, Arial, sans-serif; text-decoration: none; vertical-align: middle; font-weight: bold; display: inline-block;float: right;");
                document.getElementById("preChatSearchHelpButton").value = SnapABug.getMessage(20)
            }
            SnapEngageChat.populateSubChatForm('<div style="padding: 0px 5px 0px 5px; text-align: left;">               <div id="preChatSearchInstructions" style="padding-bottom: 10px; font-style: normal; font-weight: normal; font-size: 100%; line-height: normal; text-shadow: none; list-style: none; text-align: left; outline: none; text-indent: 0px; -moz-box-shadow: none; -webkit-box-shadow: none; font-family: lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif; font-size: 12px; font-weight: 600; color: #61656d;"></div>               <div id="preChatSearchQuestionForm">                 <textarea id="preChatSearchQuestionTextArea" autofocus rows="15" cols="40" maxlength="500" style="width: 100%; height: auto; resize: none; outline: none; border: 1px solid #999999 !important; box-shadow:none !important; font-family: arial,sans-serif; font-size: 12px; font-weight: normal; font-style: normal; line-height: normal; -webkit-box-sizing: border-box; -moz-box-sizing: border-box; box-sizing: border-box;"></textarea>                 <div style="padding-top: 15px;">                   <input id="preChatSearchNextButton" value="Next" type="submit">                 </div>               </div>               <div id="preChatSearchSearching" style="text-align: center; display: block; line-height: 200px;"></div>               <div id="preChatSearchResultsForm">                 <div id="preChatSearchSearchResults" style="height: 220px; overflow-y: scroll; overflow-x: hidden;"></div>                 <div style="padding-top: 15px;">                   <div style="display: inline-block; float: left; padding-top: 5px;"><a id="preChatSearchBackButton" style="color: #999; text-decoration: none;" href="#">Back</a></div>                   <div style="display: inline-block; float: right;"><input id="preChatSearchHelpButton" value="None of these helped..." type="button"></div>                 </div>               </div>             </div>',
            "SnapABug.closeForm();");
            var a = document.getElementById("SnapABug_CM");
            a && (a.parentNode.style.borderColor = "white");
            try {
                SnapABug.setTitle("")
            } catch (c) {
                document.getElementById("preChatSearchQuestionTextArea").rows = 9, document.getElementById("preChatSearchSearchResults").style.height = "200px"
            }
            document.getElementById("preChatSearchQuestionTextArea").onkeyup = function(b) {
                13 == b.keyCode && SnapABug.preChatSearchSubmit()
            };
            document.getElementById("preChatSearchNextButton").onclick = function() {
                SnapABug.preChatSearchSubmit()
            };
            document.getElementById("preChatSearchBackButton").onclick = function() {
                b()
            };
            document.getElementById("preChatSearchHelpButton").onclick = function() {
                SnapABug.preChatSearchStartChat()
            };
            b()
        },
        preChatSearchSubmit: function() {
            document.getElementById("preChatSearchInstructions").style.display = "none";
            document.getElementById("preChatSearchQuestionForm").style.display = "none";
            document.getElementById("preChatSearchResultsForm").style.display = "none";
            document.getElementById("preChatSearchSearching").style.display =
            "";
            N = document.getElementById("preChatSearchQuestionTextArea").value;
            var b = f + "/snapabug/preChatSearch?criteria=" + encodeURIComponent(N) + "&widget=" + SnapABug.getWidgetId();
            SnapABug.rpc(b, 3, !0)
        },
        callbackSearchAction: function(b) {
            if (b && 0 != b.length) {
                for (var a = "<ul style='list-style-type: none; padding: 0; margin: 0px;'>", c = 0; c < b.length; c += 1)
                    a = a + "<li><div style='white-space: nowrap; padding: 5px; cursor: pointer;' onclick='window.open(\"" + b[c].link + "\", \"_blank\");' onmouseover='this.style.backgroundColor=\"#e0ecfd\"'; onmouseout='this.style.backgroundColor=\"white\"'; style='background-color: white'> <img style='padding-right: 3px;' height='12' width='12' src='" +
                    f + "/img/document.png'/><span style='font-family: lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif; font-size: 12px; font-weight: bold; text-overflow: ellipsis;'><a target='_blank' style='color: #5d7584;' href='" + b[c].link + "'>" + b[c].title + "</a></span><div style='font-family: lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif; font-size: 11px; white-space: normal; line-height: normal;'>" + b[c].summary + "</div></div><div style='padding-top: 3px; margin-bottom: 3px; border-bottom: 1px solid #d0d0d0;'></div></li>";
                a += "</ul>";
                document.getElementById("preChatSearchInstructions").innerHTML = SnapABug.getMessage(17);
                document.getElementById("preChatSearchSearchResults").innerHTML = a;
                document.getElementById("preChatSearchSearching").style.display = "none";
                document.getElementById("preChatSearchInstructions").style.display = "";
                document.getElementById("preChatSearchResultsForm").style.display = ""
            } else 
                SnapABug.preChatSearchStartChat()
        },
        preChatSearchStartChat: function() {
            SnapABug.setPreChatSearch(0);
            if ("" == ca) {
                var b = [ - 1, - 1,
                - 1, - 1];
                SnapABug.formPosX && SnapABug.formPosY ? (b[0] = SnapABug.formPosX, b[2] = SnapABug.formPosY) : b = SnapABug.getDefaultChatPosition();
                SnapABug.setDescription(N);
                SnapABug.loadForm();
                SnapABug.positionForm(b[0], b[1], b[2], b[3])
            } else 
                N && (SnapABug.setDescription(N), SnapABug.setStartChat(N)), Ga = N ? N : "n/a", SnapEngage.startChat(N, !0)
        },
        deactivateHTML: function(b) {
            return b.replace(/>/g, "&gt;").replace(/</g, "&lt;")
        }
    }
}(), SnapABugChat = function() {
    var f = "", k = "", u = null, g = 0, r = 0, d = "", n = "agent", x = "", q = "", v = "", B = "", A=!1, C=!1,
    z = "", w = "", D=!0, W = "", l = "", O = "", p = "", M = 0, J=!1, H = "color:#75A7fE;", F = SnapABug.getStaticBaseURL() + "/wbg/jimdo_chat_form.png", H = "color:#0087c1;", $ = SnapABug.getStaticBaseURL() + "/wbg/jimdo_chat_form.png", X = document.title, K = "", Y = null, P = "", L = "", Z = "", ha =+ new Date, Q, G = null, R = null;
    return {
        startChat: function(a, c) {
            this.initChat(a, c);
            this.sendChatAction()
        },
        restartChat: function(a, c) {
            D=!1;
            this.initChat(a, c);
            SnapABug.getPageChangeNotificationAllowed() ? this.sendChatMessage(7, window.location.href) : this.sendChatMessage(7,
            "");
            this.setReadOnly(!1);
            this.setPreviousChatWindowState();
            this.startPolling()
        },
        preloadChatFormImg: function() {
            null == SnapABug.getMobileUI() && ((new Image).src = F)
        },
        proactive: function(a, c, d, e) {
            null == SnapABug.getMobileUI() && (k = SnapABug.getBaseURL(), q = e, "" == f && (n = d, " " != a ? "/info " == a.substr(0, 6) ? K = a.substr(6) : (K = this.createTranscriptLine(d, a), Z = a) : K = " ", this.showFormChat(), SnapABug.slideFormIn(), c && this.newMessageNotification(), this.setReadOnly(!1), "" != d && SnapABug.callCallback("OpenProactive", d, a), SnapABug.setProactiveChatAutocloseTimeout()))
        },
        chatStarting: function() {
            document.getElementById("SnapABug_CL") || null != SnapABug.getMobileUI() || (k = SnapABug.getBaseURL(), this.showFormChat(), this.setReadOnly(!0))
        },
        initChat: function(a, c) {
            f = a;
            k = c;
            M = g = 0;
            n = "agent";
            J = C = A=!1;
            Y = {};
            var d = document.SnapABug_Form;
            w = d && d.chat ? d.chat.value : "";
            "" == z && "" != K && (z = K);
            K = "";
            SnapABug.disableCallMe();
            this.setCaseIdCookie();
            this.setChatViewCookie();
            null == SnapABug.getMobileUI() && this.showFormChat()
        },
        setCustomIntroMsg: function(a) {
            W = a
        },
        reInit: function() {
            B = z = Z = x = v = q = "";
            this.setReadOnly(!0);
            Q && (Q = window.clearTimeout(Q))
        },
        isProactive: function() {
            return "" != K
        },
        clearProactive: function() {
            K = ""
        },
        sendChatAction: function() {
            var a = "ci=" + encodeURIComponent(SnapABug.getDbCaseId());
            SnapABug.proactiveAgentNumber && (a += "&a=" + SnapABug.proactiveAgentNumber);
            Z && (a += "&pm=" + encodeURIComponent(Z));
            SnapABug.rpc(k + "/snapabug/ServiceChatAction?" + a, 3, !0)
        },
        callbackChatAction: function(a) {
            this.setReadOnly(!1);
            SnapABugChat.updateHistoryCookieForLastCase(f);
            this.startPolling();
            "" != SnapABug.getCallMePhone() && (SnapABug.sendClickToCall(SnapABug.getCallMePhone(),
            2), SnapABug.setCallMePhone(""))
        },
        setReassignmentTimerAfterRestartChat: function(a) {
            if ( - 1 < a&&!Q) {
                var c = this.getHistoryCookieLastCase();
                c[0] == f && (c = (new Date).getTime() - c[1], c < 1E3 * a && (Q = window.setTimeout(function() {
                    SnapABugChat.callAssignNewAgent()
                }, 1E3 * a - c)))
            }
        },
        callAssignNewAgent: function() {
            if (f&&!J && "1" != SnapABugChat.getBannedCookie() && "" == SnapABug.getCallMePhone()) {
                var a = document.getElementById("SnapABug_Typing");
                a && "visible" == a.style.visibility ? Q = window.setTimeout(function() {
                    SnapABugChat.callAssignNewAgent()
                },
                1E4) : (a = "c=" + encodeURIComponent(SnapABug.getCaseId()), SnapABug.rpc(k + "/snapabug/ServiceAgentReassign?" + a, 3, !0))
            }
        },
        callbackAssignNewAgent: function(a) {
            a && (SnapABugChat.updateHistoryCookieForLastCase(f), Q = null)
        },
        soundNotificationDiv: function() {
            return ""
        },
        populateChatForm: function() {
            var a = SnapABug.getCSS(324, 8, 48, 42, "cursor:pointer;"), c = SnapABug.getCSS(0, 0, 380, 400);
            SnapABug.positionDragger(9, 7, 314, 61);
            var d = SnapABug.getCSS(30, 72, 313, 216, "background:none;font-family:arial,sans-serif;font-size:12px;font-weight:normal;color:black;",
            "overflow:auto;margin-left:7px;"), e = SnapABug.getCSS(35, 302, 315, 62, "font-family:arial,sans-serif;font-size:12px;font-weight:normal;color:black;", "vertical-align:top;resize:none;background-image:url('" + SnapABug.getBlankImg() + "');background-repeat:repeat;overflow:auto;"), f = SnapABug.getCSS(352, 150, 16, 138);
            SnapABug.getCSS(216, 372, 138, 16);
            var l = SnapABug.getCSS(32, 370, 78, 16), c = '<div style="left:0;top:0;z-index:1;"><img src="' + F + '" style="' + c + '" galleryimg="no" width="380" height="400" /></div><div id="SnapABug_CL" style="' +
            d + '"></div><form name="SnapABug_Form" accept-charset="UTF-8" action="#" onsubmit="return SnapABugChat.submitChat();"><textarea id="SnapABug_CE" style="' + e + '" type="text" name="chat" onkeydown="return SnapABugChat.keyDownHandler(event);" onclick="return SnapABugChat.keyDownHandler(event);"></textarea></form>', c = c + ('<div style="' + a + '" onclick="SnapABug.closeForm();" title="' + SnapABug.getMessage(10) + '"><img style="margin:0;border:none;padding:0;width:48px;height:42px;" src="' + SnapABug.getBlankImg() +
            '" width="48" height="42" alt="' + SnapABug.getMessage(10) + '" /></div>' + this.soundNotificationDiv() + "");
            SnapABug.isBranded() && (c += '<div style="' + f + '">', SnapABug.logoLink() && (c += '<a href="' + SnapABug.getSnapABugSiteUrl() + '" target="_blank" style="text-indent:0px;border:none">'), c += '<img style="margin:0;border:none;padding:0;" src="' + SnapABug.getStaticBaseURL() + '/wbg/logo_chat_vertical.png" border="0" alt="powered by SnapEngage" width="16" height="138" />', SnapABug.logoLink() && (c += "</a>"), c += "</div>");
            c += '<div id="SnapABug_CallMe" style="' + l + '">' + SnapABug.getCallMeButton() + "</div>";
            document.getElementById("SnapABug_P").innerHTML = c;
            SnapABug.showForm(!1, 380, 400, 0, 0, 0, 0)
        },
        populateSubChatForm: function(a, c) {
            var d = SnapABug.getCSS(324, 8, 48, 42, "cursor:pointer;"), e = SnapABug.getCSS(0, 0, 380, 400);
            SnapABug.positionDragger(9, 7, 314, 61);
            var f = SnapABug.getCSS(29, 69, 320, 295, "font-family: arial, sans-serif; font-size: 12px; font-weight: normal; z-index: 2; background-color: #ffffff; color: #000000; overflow: hidden; border: 1px solid #8e8e8e; text-align: center;"),
            l = SnapABug.getCSS(352, 150, 16, 138), e = '<div style="left:0;top:0;z-index:1;"><img src="' + $ + '" style="' + e + '" galleryimg="no" width="380" height="400" /></div><div style="' + f + '"><div id="SnapABug_CM" style="position:relative;text-align:center;">' + a + "</div></div>";
            c && (e += '<div style="' + d + '" onclick="' + c + '" title="' + SnapABug.getMessage(10) + '"><img style="margin: 0; border: none; padding: 0; width: 48px; height: 42px;" src="' + SnapABug.getBlankImg() + '" width="48" height="42" alt="' + SnapABug.getMessage(10) +
            '" /></div>');
            e += this.soundNotificationDiv();
            SnapABug.isBranded() && (e += '<div style="' + l + '">', SnapABug.logoLink() && (e += '<a href="' + SnapABug.getSnapABugSiteUrl() + '" target="_blank" style="text-indent: 0px; border: none;">'), e += '<img style="margin: 0; border: none; padding: 0;" src="' + SnapABug.getStaticBaseURL() + '/wbg/logo_chat_vertical.png" border="0" alt="powered by SnapEngage" width="16" height="138" />', SnapABug.logoLink() && (e += "</a>"), e += "</div>");
            document.getElementById("SnapABug_P").innerHTML =
            e;
            SnapABug.showForm(!1, 380, 400, 19, 18, 304, 331)
        },
        populateCallMeForm: function(a, c) {
            function d() {
                var a = document.createElement("script");
                a.type = "text/javascript";
                a.src = SnapABug.getBaseURL() + "/js/i18nphonepicker/i18nphonepicker-min.js";
                a.onload = function() {
                    var a = document.createElement("link");
                    a.rel = "stylesheet";
                    a.type = "text/css";
                    a.href = SnapABug.getBaseURL() + "/js/i18nphonepicker/css/intlTelInput.css";
                    document.getElementsByTagName("head")[0].appendChild(a)
                };
                document.getElementsByTagName("head")[0].appendChild(a)
            }
            var e = SnapABug.getStaticBaseURL() + "/wbg/callme_phone_grey.png", l = SnapABug.getStaticBaseURL() + "/wbg/callme_phone_black.png", g = SnapABug.getStaticBaseURL() + "/wbg/callme_web_grey.png", n = SnapABug.getStaticBaseURL() + "/wbg/callme_web_black.png", p = "", k = ["31e46819-a83b-47a0-ab99-567671ec74d7", "timzon-snapabug-widget", "beaa0140-5a54-4c58-96e1-cc7971339d75", "a4c15bd0-c67e-4a5b-9a1f-e29ae0f6023b", "7fc26765-2ac7-4297-a8af-b19743176bbc"];
            window._1a058890_5788_4457_aa0e_3453f3f878a0 = SnapABug;
            a ||!SnapABug.isInArray(SnapABug.getWidgetId(),
            k)&&!SnapABug.getEnableInternationalPhoneCalls() || (SnapABug.setEnableInternationalPhoneCalls(!0), d());
            !a && f ? (p += '<div id="SnapABug_CMW" onclick="SnapABugChat.submitFormCallMe(true, true)" onmouseover="javascript: document.getElementById(&apos;SnapABug_CMW&apos;).style.backgroundImage=&apos;url(' + n + ')&apos;;document.getElementById(&apos;SnapABug_CMW&apos;).style.color=&apos;white&apos;;" onmouseout="javascript: if (document.getElementById(&apos;SnapABug_CMW&apos;)) {document.getElementById(&apos;SnapABug_CMW&apos;).style.backgroundImage=&apos;url(' +
            g + ')&apos;;document.getElementById(&apos;SnapABug_CMW&apos;).style.color=&apos;#4d4d4d&apos;;}" style="width:230px;height:86px;margin:20px auto 0 auto;background-image:url(' + g + '); background-repeat: no-repeat no-repeat;cursor:pointer;position:relative;color:#4d4d4d;"><div style="width:230px;position:absolute;bottom:5px;margin-top:3px;font-family:arial,sans-serif;font-size:12px;font-weight:normal;font-weight:bold;">Call from my computer</div></div>', SnapABug.isUSCanada() && (p += '<div id="SnapABug_CMP" onclick="SnapABugChat.populateCallMeForm(true);" onmouseover="javascript: document.getElementById(&apos;SnapABug_CMP&apos;).style.backgroundImage=&apos;url(' +
            l + ')&apos;;document.getElementById(&apos;SnapABug_CMP&apos;).style.color=&apos;white&apos;;" onmouseout="javascript: if (document.getElementById(&apos;SnapABug_CMP&apos;)) {document.getElementById(&apos;SnapABug_CMP&apos;).style.backgroundImage=&apos;url(' + e + ')&apos;;document.getElementById(&apos;SnapABug_CMP&apos;).style.color=&apos;#4d4d4d&apos;;}" style="width:230px;height:86px;margin:20px auto 0 auto;background-image:url(' + e + '); background-repeat: no-repeat no-repeat;cursor:pointer;position:relative;color:#4d4d4d;"><div style="width:230px;position:absolute;bottom:5px;margin-top:3px;font-family:arial,sans-serif;font-size:12px;font-weight:normal;color:#4d4d4d;font-weight:bold;">Call me on my phone</div></div>')) :
            (c || (c = ""), e = "(US and Canada only)", SnapABug.getEnableInternationalPhoneCalls() && (e = ""), p += '<div id="SnapABug_CMPFORM" style="width:230px;margin:20px auto;padding:10px 0px;border-radius:10px;-moz-border-radius:10px;-webkit-border-radius:10px;background-color:white;"><div id="SnapABug_phone" style="margin-top:6px;margin-bottom:6px;font-family:lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif;font-size:12px;font-weight:default;color:grey;">Enter your phone number</div><form name="SnapABug_Form" id="SnapABug_Form" accept-charset="UTF-8" style="margin:0px;" action="#"><input style="height:30px;margin-top:0px;margin-top:0px;font-family:lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif;font-size:13px;color:black;border:2px inset;' +
            (c ? "border-color:red;" : "") + '" type="text" id="phonenumber" name="phonenumber" value="' + c + '" size="16"/><br /><span id="SnapABug_CallMeSubmit"><input onclick="return SnapABugChat.submitFormCallMe(true);" style="width:auto;height:28px;margin:0;margin-top:6px;padding:3px 5px 3px 5px;font-size:12px;font-family:lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif;cursor:pointer;font-weight:bold;" type="submit" value="Call Me Now" /></span></form><div id="SnapABug_phone" style="margin-top:4px;font-family:lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif;font-size:12px;font-weight:default;color:grey;">' +
            e + "</div></div>");
            z && (p += '<div onclick="return SnapABugChat.submitFormCallMe(false);" style="color:grey;width:230px;height:28px;margin:20px auto;padding:10px 0px;font-size:12px;font-family:lucida grande,Helvetica Neue,Helvetica,Arial,sans-serif;text-align:left;cursor:pointer;">Return to chat</div>');
            this.populateSubChatForm(p, "SnapABugChat.submitFormCallMe(false);");
            a && (SnapABug.phonePicker.init("phonenumber", SnapABug.getVisitorCountry()), SnapABugChat.selectPhoneInputCountry(SnapABug.getVisitorCountry()),
            SnapABug.phonePicker.setCountryListWindowHeight(150));
            p = document.getElementById("SnapABug_CM");
            p.style.top = Math.round((290 - p.offsetHeight) / 2) + "px"
        },
        selectPhoneInputCountry: function(a) {
            if ("string" == typeof a || a instanceof String)
                a = a.toLowerCase();
            if (SnapABug.phonePicker) {
                var c = SnapABug.phonePicker.getCountryData();
                if (c.countries && a)
                    for (var d = 0; d < c.countries.length; d++)
                        a === c.countries[d].iso2 && SnapABug.phonePicker.selectCountry(a)
            }
        },
        isPhoneNumberValid: function() {
            var a = document.getElementById("SnapABug_Form");
            return a && a.phonenumber && a.phonenumber.value ? /^\+{0,1}[\d \(\)-.]+[xX]{0,1}[wW]*[\d \(\)\-]+$/.test(a.phonenumber.value) && 5 < a.phonenumber.value.length : !1
        },
        submitFormCallMe: function(a, c) {
            if (a)
                if (SnapABug.disableCallMe(), c)
                    window.open(SnapABug.getBaseURL() + "/voice/webphone.jsp?c=" + SnapABug.getCaseId(), "webphone", "left=20,top=20,width=310,height=260,toolbar=0,location=0,directories=0,status=0,menubar=0,scrollbars=0,resizable=1"), f ? SnapABugChat.formCallMeDone() : SnapABug.hideOverlay(!1);
                else {
                    var d = "",
                    e = document.getElementById("SnapABug_Form");
                    if (e && e.phonenumber && e.phonenumber.value)
                        if (SnapABugChat.isPhoneNumberValid())
                            d = e.phonenumber.value.replace(/[^+0-9xX]*/g, "");
                        else {
                            SnapABug.callbackCheckPhone(!1);
                            return 
                        }
                        "" != d && (SnapABug.setCallMePhone(d), SnapABug.sendCheckPhone(d), document.getElementById("SnapABug_CallMeSubmit").innerHTML = "Please wait...")
                } else 
                    SnapABug.enableCallMe(), f || SnapABugChat.isProactive() ? SnapABugChat.formCallMeDone() : SnapABug.hideOverlay(!1);
            return !1
        },
        showFormChat: function() {
            null ==
            SnapABug.getMobileUI() && this.populateChatForm();
            this.isProactive() && (z = K, this.setReadOnly(!1));
            "" != z ? this.displayHistory() : this.setReadOnly(!0);
            var a = document.SnapABug_Form;
            a && ("" != w ? (a.chat.value = w, w = "") : "" != l && "(none)" != l && ("" == z && this.appendTextToChat(SnapABug.getMessage(15), l + "<br />", !1), l = ""), D ? a.chat.value = "" != W ? W : unescape(SnapABug.getMessage(14)) : SnapABugChat.isMinimized() || a.chat.focus(), "" != this.getMessageCookie() && (D=!1, a.chat.value = this.getMessageCookie(), this.setMessageCookie("")))
        },
        displayHistory: function() {
            var a = document.getElementById("SnapABug_CL");
            try {
                a.innerHTML = z + this.createTranscriptLine(n, "", !0), this.scrollDown()
            } catch (c) {}
        },
        startPolling: function() {
            u && this.stopPolling();
            u = setInterval("SnapABugChat.pollChat()", 2E3)
        },
        stopPolling: function() {
            clearInterval(u)
        },
        keyDownHandler: function(a) {
            a = window.event || a;
            var c = a.keyCode || a.which;
            if (a && 13 == c) {
                if (a.ctrlKey) {
                    if (c = document.SnapABug_Form)
                        c.chat.value += "\n"
                } else 
                    A || (f || SnapABug.clearProactiveChatAutocloseTimeout(), this.submitChat());
                a.stopPropagation && a.stopPropagation();
                a.preventDefault && a.preventDefault();
                a.cancelBubble && (a.cancelBubble=!0);
                a.returnValue && (a.returnValue=!1);
                return !1
            }
            if (D) {
                if (c = document.SnapABug_Form)
                    c.chat.value = "";
                D=!1
            }
            f || SnapABug.setProactiveChatAutocloseTimeout();
            return !0
        },
        submitChat: function() {
            var a = document.SnapABug_Form;
            if (a) {
                var c = a.chat.value;
                D ? (a && (a.chat.value = ""), D=!1) : (c = SnapABug.deactivateHTML(c), "" != c && (this.appendTextToChat(SnapABug.getMessage(15), c, !1), this.isProactive() ? (SnapABug.sendCreateCase(SnapABug.getUserEmail(),
                c, !0), this.setReadOnly(!0)) : this.sendChatMessage(1, c), d = a.chat.value = ""));
                a.chat.focus()
            }
            return !1
        },
        setReadOnly: function(a) {
            A = a;
            try {
                var c = document.getElementById("SnapABug_CE");
                c && (A ? c.setAttribute("readOnly", "readonly") : c.removeAttribute("readOnly"))
            } catch (d) {}
        },
        callbackChatPosted: function(a) {},
        appendTextToChat: function(a, c, d) {
            this.setCaseIdCookie();
            c = c.replace(/\&\#41\;/g, ")");
            c = c.replace(/\&\#40\;/g, "(");
            c = c.replace(/\\n/g, "<br />");
            c = c.replace(/\n/g, "<br />");
            c = c.replace(/&lt; *br *\/*&gt;/g, "<br />");
            c = c.replace(/&lt; *BR *\/*&gt;/g, "<br />");
            null == SnapABug.getMobileUI() && (c = c.replace(/(\s|^):\-\)/g, ' <img src="' + k + '/img/emote/smile.png" alt="[smile]" width="16" height="16" />'), c = c.replace(/(\s|^);\-\)/g, ' <img src="' + k + '/img/emote/wink.png" alt="[wink]" width="16" height="16" />'), c = c.replace(/(\s|^):\-D/g, ' <img src="' + k + '/img/emote/grin.png" alt="[grin]" width="16" height="16" />'), c = c.replace(/(\s|^):\-\(/g, ' <img src="' + k + '/img/emote/sad.png" alt="[sad]" width="16" height="16" />'), c = c.replace(/(\s|^):\-o/g,
            ' <img src="' + k + '/img/emote/shock.png" alt="[shock]" width="16" height="16" />'), c = c.replace(/(\s|^):\)/g, ' <img src="' + k + '/img/emote/smile.png" alt="[smile]" width="16" height="16" />'), c = c.replace(/(\s|^);\)/g, ' <img src="' + k + '/img/emote/wink.png" alt="[wink]" width="16" height="16" />'), c = c.replace(/(\s|^):D/g, ' <img src="' + k + '/img/emote/grin.png" alt="[grin]" width="16" height="16" />'), c = c.replace(/(\s|^):\(/g, ' <img src="' + k + '/img/emote/sad.png" alt="[sad]" width="16" height="16" />'), c = c.replace(/(\s|^):o/g,
            ' <img src="' + k + '/img/emote/shock.png" alt="[shock]" width="16" height="16" />'));
            if (null != SnapABug.getMobileUI())
                SnapABug.getMobileUI().updateChat(c, a == SnapABug.getMessage(15), d);
            else {
                var e = document.getElementById("SnapABug_CL");
                z += this.createTranscriptLine(a, c, !1);
                B = a;
                !0 != d && e && (e.innerHTML = z + this.createTranscriptLine(n, "", !0), this.scrollDown())
            }
        },
        createTranscriptLine: function(a, c, d) {
            var e = "", f = "";
            z || (e += '<div style="height:7px;"></div>');
            if (!d && "" != q && a == n) {
                if ("" == x || x != n)
                    f += '<img src="' + q + '" style="float:left;margin-right:5px;padding:1px;border:1px solid #888;background-color:#fff;" width="53" height="53" />';
                v = q;
                x = n
            }
            if ("" != a && a != B) {
                var l = "color:#A1A1A3;";
                a == n && (l = H);
                f += '<span style="' + l + "font-weight:bold;" + SnapABug.getDirection() + '">' + a + ":</span> "
            }
            d || "" == q || a != n || (f += "<br />", q = "");
            var g = l = "";
            d && (l = 'id="SnapABug_Typing" ', g = "visibility:hidden;", c = '<img src="' + k + '/img/typing.gif" alt="[...]" width="24" height="8" />');
            a != B ? ("" != z && (e += '<div style="height:7px;"></div>'), e += "<div " + l + 'style="clear:left;line-height:normal;color:black;' + g + SnapABug.getDirection() + '">' + f + c + "</div>") : e += "<div " + l + 'style="line-height:normal;color:black;' +
            g + SnapABug.getDirection() + '">' + f + c + "</div>";
            return e
        },
        isTyping: function(a) {
            if (null != SnapABug.getMobileUI())
                a && SnapABug.getMobileUI().triggerIsTyping();
            else {
                var c = document.getElementById("SnapABug_Typing");
                c && (c.style.visibility = a ? "visible" : "hidden")
            }
        },
        scrollDown: function() {
            var a = document.getElementById("SnapABug_CL");
            a && (a.scrollTop = a.scrollHeight)
        },
        waitingForAgentMessage: function(a) {
            J || (a = a.replace(/\[\[/g, '<a href="#" onclick="SnapABugChat.cancelChat(); return false;">'), a = a.replace(/\]\]/g, "</a>"),
            a = a.replace(/\[/g, '<a href="#" onclick="SnapABugChat.cancelChat(); return false;">'), a = a.replace(/\]/g, "</a>"), this.appendTextToChat("", "<br />" + a, !1))
        },
        isVisitorTyping: function() {
            var a=!1, c = "", m = document.SnapABug_Form;
            m && m.chat && (c = m.chat.value, !D && c && d != c && (a=!0));
            d = c;
            return a
        },
        pollChat: function() {
            if (this.isChatViewCookieValid() && "" != f) {
                this.refreshChatViewCookie();
                if (0 == g) {
                    var a = "c=" + encodeURIComponent(f) + "&i=" + this.getPollIdCookie(), a = a + ("&t=" + (new Date).getTime());
                    if (0 == r%2) {
                        a += "&gt";
                        this.isVisitorTyping() &&
                        (a += "&st");
                        var c = document.SnapABug_Form;
                        c && SnapABug.isSneakPeek() && (SnapABugChat.introMsg || (a += "&tt=" + encodeURIComponent(c.chat.value)))
                    }
                    SnapABug.rpc(k + "/snapabug/ServiceChatPoll?" + a, 0, !0)
                }
                g++;
                3 < g && (g = 0);
                r++
            } else 
                this.stopPolling(), SnapABug.hideOverlay(!1), SnapABug.setChatStatus(!1)
        },
        callbackPolling: function(a, c, d) {
            1 != Y[c] && (Y[c] = 1, - 1 != c && this.setPollIdCookie(c), this.callbackMessage(!0, a, !1));
            1 == d ? this.isTyping(!0) : 2 == d && this.isTyping(!1);
            g = 0
        },
        sendChatMessage: function(a, c) {
            var d = "a=" + a + "&c=" + encodeURIComponent(f),
            e = new Date, d = d + ("&t=" + e.getTime());
            "" != c && (d += "&m=" + encodeURIComponent(c));
            d = k + "/snapabug/ServiceChatMessage?" + d;
            2E3 < d.length && (d = "a=" + a + "&c=" + encodeURIComponent(f), d += "&t=" + e.getTime(), d += "&m=" + encodeURIComponent("[truncated message] " + c), d = k + "/snapabug/ServiceChatMessage?" + d);
            SnapABug.rpc(d, 3, !0);
            1 == a && SnapABug.callCallback("ChatMessageSent", c)
        },
        callbackMessage: function(a, c, d, e) {
            if ("" != c)
                switch (c = decodeURIComponent(c).replace(/\%20/g, " "), c.substr(0, 4)) {
                case "/SCR":
                    d || this.takeScreenshot();
                    break;
                case "/BYE":
                    d || (this.appendTextToChat("", c.substr(5), !1), this.setReadOnly(!0), this.closeChat(), SnapABug.disableCallMe(), null != SnapABug.getMobileUI() && SnapABug.getMobileUI().disableTextarea());
                    break;
                case "/NIC":
                    n = c.substr(5);
                    if (null != SnapABug.getMobileUI())
                        SnapABug.getMobileUI().updateHeaderText(n);
                    else if (e = document.getElementById("SnapABug_AName"))
                        e.innerHTML = n;
                        break;
                    case "/GO2":
                        d || this.redirectBrowser(c.substr(5));
                        break;
                    case "/WAI":
                        this.setReadOnly(!0);
                        this.appendTextToChat("", c.substr(5), !1);
                        break;
                    case "/NFO":
                        null != SnapABug.getMobileUI() && SnapABug.getMobileUI().isInitAgentMsg(c.substr(5));
                        this.setReadOnly(!1);
                        this.appendTextToChat("", c.substr(5), !1);
                        break;
                    case "/CAL":
                        - 1 != c.substr(5).indexOf("1") ? SnapABug.enableCallMe() : SnapABug.disableCallMe();
                        break;
                    case "/BAN":
                        this.setBannedCookie();
                        SnapABug.allowChat(!1);
                        break;
                    case "/AVA":
                        q = c.substr(5);
                        null != SnapABug.getMobileUI() ? SnapABug.getMobileUI().updateAgentAvatar(q) : (e = document.getElementById("SnapABug_AAvatar")) ? (e.src = q, q = "") : q == v && (q = "");
                        break;
                    case "/SYS":
                        a = c.substr(5);
                        c = a.substr(0, 4);
                        "PURL" == c ? P = a.substr(5) : "PTXT" == c ? L = decodeURIComponent(a.substr(5)).replace(/\%20/g, " ") : "WAGT" == c ? SnapABug.isMobile() && "undefined" != typeof SNAPENGAGE_MOBILE || (e = decodeURIComponent(a.substr(5)).replace(/\%20/g, " "), a = e.substring(0, e.indexOf(",")), e = e.substring(e.indexOf(",") + 1), window.setTimeout('SnapABugChat.waitingForAgentMessage("' + e.replace(/\"/g, '\\"') + '");', 1E3 * a)) : "EMAI" == c ? (e = decodeURIComponent(a.substr(5)).replace(/\%20/g, " "), SnapABug.setUserEmail(e),
                        SnapABugChat.setUserEmailCookie(e)) : "REAS" == c ? (e = a.substr(5), SnapABugChat.setReassignmentTimerAfterRestartChat(e)) : "RERY" != c || e ? "SRVY" == c && (G = eval("(" + a.substr(5) + ")")) : SnapABugChat.updateHistoryCookieForLastCase(SnapABug.getCaseId());
                        break;
                    default:
                        this.setReadOnly(!1);
                        var f = n;
                        a ? J=!0 : f = SnapABug.getMessage(15);
                        this.appendTextToChat(f, c, e);
                        d || (this.newMessageNotification(), a && (SnapABugChat.isMinimized() && SnapABugChat.newMinimizedMessageReceived(), SnapABug.callCallback("ChatMessageReceived", n, c)))
                }
            g =
            0
        },
        callbackHistory: function(a, c) {
            if (a && 0 < a.length) {
                Y = {};
                for (var d = 0, e = 0; e < a.length / 3; e++) {
                    var f = a[3 * e + 2];
                    this.callbackMessage(a[3 * e], a[3 * e + 1], !0, !0);
                    - 1 != f && (Y[f] = 1, d = f)
                }
                J = c;
                this.displayHistory();
                this.setPollIdCookie(d)
            }
            g = 0
        },
        newMessageNotification: function() {
            document.title = this.getPageTitle(!0);
            window.setTimeout("document.title=SnapABugChat.getPageTitle(false);", 1E3);
            window.setTimeout("document.title=SnapABugChat.getPageTitle(true);", 1500);
            window.setTimeout("document.title=SnapABugChat.getPageTitle(false);",
            2500);
            window.setTimeout("document.title=SnapABugChat.getPageTitle(true);", 3E3);
            window.setTimeout("document.title=SnapABugChat.getPageTitle(false);", 4E3);
            window.setTimeout("document.title=SnapABugChat.getPageTitle(true);", 4500);
            window.setTimeout("document.title=SnapABugChat.getPageTitle(false);", 5500);
            SnapABug.getChatSoundAllowed() && this.playSound()
        },
        playSound: function() {
            var a = k + "/sounds/chatmsg.mp3", c = k + "/sounds/chatmsg.ogg";
            try {
                (SnapABug.isFirefox() ? new Audio(c) : new Audio(a)).play()
            } catch (d) {
                try {
                    var e =
                    document.getElementById("SnapABug_Snd");
                    e && (e.innerHTML = '<iframe src="' + k + '/sounds/chatmsg.html" allowtransparency="true" frameborder="0" />')
                } catch (f) {}
            }
        },
        getPageTitle: function(a) {
            return !0 == a ? n + " says..." : X ? X : document.domain
        },
        setQuestion: function(a) {
            l = a;
            "(none)" != a && (O = a)
        },
        cancelChat: function() {
            J ? SnapABugChat.closeChat(!0) : (this.sendChatMessage(8, ""), f = "");
            SnapABug.closeForm(!0);
            SnapABug.setDescription(O);
            SnapABug.allowChat(!1);
            SnapABug.startLink()
        },
        closeChat: function(a) {
            this.stopPolling();
            a && f &&
            this.sendChatMessage(2, "");
            G ? this.populateChatAgentSurvey() : P ? (a = "?agent=" + encodeURIComponent(n) + "&case=" + encodeURIComponent(SnapABug.getBaseURL() + "/viewcase?c=" + f), this.populateSubChatForm('<iframe src="' + P + a + '" frameborder="0" width="100%" height="100%" style="overflow:hidden"></iframe>', "SnapABug.closeForm();"), P = "") : L && (L = L.replace(/\{alias\}/g, encodeURIComponent(n)), L = L.replace(/\{case\}/g, encodeURIComponent(SnapABug.getBaseURL() + "/viewcase?c=" + f)), this.populateSubChatForm(L, "SnapABug.closeForm();"),
            L = "");
            f = "";
            this.deleteChatCookies();
            this.reInit()
        },
        redirectBrowser: function(a) {
            var c = document.SnapABug_Form;
            if (c && (w = c.chat.value, "" != w))
                try {
                    (new RegExp(/^\s+$/)).test(w) || this.setMessageCookie(w.replace(/^\s*/, "").replace(/\s*$/, ""))
            } catch (d) {}
            this.appendTextToChat("", "redirecting...", !1);
            c=!1;
            try {
                var e = document.location.hostname;
                if (0 == a.indexOf("http")&&-1 != a.indexOf("://")) {
                    var f = a.substring(a.indexOf("://") + 3);
                    - 1 != f.indexOf("/") && (f = f.substring(0, f.indexOf("/")), - 1 != f.indexOf(":") && (f = f.substring(0,
                    f.indexOf(":"))));
                    var l = SnapABug.getBaseURL() + "/chat?c=";
                    0 == a.indexOf(l) || 0 == a.indexOf(l.replace("http:", "https:")) ? c=!0 : "" == f || e == f || "" != SnapABug.getDomain()&&-1 != f.indexOf(SnapABug.getDomain()) || (a = l + SnapABug.getCaseId() + "&t=url&code=" + encodeURIComponent(a), c=!0, this.deleteChatCookies())
                }
            } catch (g) {}
            c && top && top.frames && 0 < top.frames.length ? setTimeout('top.location.href = "' + a + '"', 100) : setTimeout('window.location.href = "' + a + '"', 100)
        },
        takeScreenshot: function() {
            if (SnapABug.isScreenshotCapable()) {
                var a =
                document.SnapABug_Form;
                a && (w = a.chat.value);
                this.stopPolling();
                SnapABug.populateFormStatus(1);
                C=!0;
                SnapABug.startWebCapture()
            } else 
                this.sendChatMessage(5, "")
        },
        webCaptureDone: function(a) {
            C=!1;
            this.sendChatMessage(4, a);
            this.showFormChat();
            this.startPolling()
        },
        isWebCapInProgress: function() {
            return C
        },
        formCallMeDone: function() {
            this.showFormChat()
        },
        setCaseIdCookie: function() {
            f && this.setCookie("SnapABugChatSession", f, 16)
        },
        getCaseIdCookie: function() {
            return this.getCookie("SnapABugChatSession")
        },
        setNoProactiveChatCookie: function() {
            this.setCookie("SnapABugNoProactiveChat",
            "no", SnapABug.getNoProactiveChatDelay())
        },
        isNoProactiveChatCookie: function() {
            return "no" == this.getCookie("SnapABugNoProactiveChat") ||!this.isCookieEnabled()
        },
        setPollIdCookie: function(a) {
            "" != f && this.setCookie("SnapABugChatPoll" + f, a, 16)
        },
        getPollIdCookie: function() {
            var a = M;
            "" != f && (M = parseInt(this.getCookie("SnapABugChatPoll" + f)), isNaN(M) && (M = a));
            return M
        },
        setMessageCookie: function(a) {
            this.setCookie("SnapABugChatMessage", encodeURIComponent(a), 16)
        },
        getMessageCookie: function() {
            return decodeURIComponent(this.getCookie("SnapABugChatMessage"))
        },
        setChatViewCookie: function() {
            p = Math.random();
            this.setCookie("SnapABugChatView", p, 16)
        },
        refreshChatViewCookie: function() {
            this.setCookie("SnapABugChatView", p, 16)
        },
        isChatViewCookieValid: function() {
            var a = this.getCookie("SnapABugChatView");
            return p == a || "" == a?!0 : !1
        },
        setHistoryCookieAppendCase: function(a, c) {
            try {
                var d = this.getCookie("SnapABugHistory");
                c || (c = (new Date).getTime());
                "" == d && (d = "1#");
                var e = d.split("!");
                if (5 < e.length)
                    for (var d = d.split("#")[0] + "#", f = e.length - 5; f < e.length - 1; f++)
                        d += e[f] + "!";
                this.setCookie("SnapABugHistory",
                d + a + ":" + c + "!", 525600)
            } catch (l) {
                this.setCookie("SnapABugHistory", "0#", 525600)
            }
        },
        updateHistoryCookieForLastCase: function(a) {
            try {
                var c = this.getCookie("SnapABugHistory"), d = (new Date).getTime(), e = c.split("#"), f = e[1].split("!"), l = f[f.length - 2].split(":");
                l[0] == a && (l[1] = d, c = c.substring(0, c.length - 1), 2 == f.length ? (c = e[0] + "#", c += l[0] + ":" + l[1] + "!") : (c = c.substring(0, c.lastIndexOf("!")), c += "!" + l[0] + ":" + l[1] + "!"), this.setCookie("SnapABugHistory", c, 525600))
            } catch (g) {}
        },
        setHistoryCookieIncrementVisits: function() {
            try {
                var a =
                this.getCookie("SnapABugHistory");
                if ("" == a)
                    a = "1#";
                else {
                    var c = a.split("#"), d = parseInt(c[0]) + 1;
                    c[0] = d;
                    a = c[0] + "#" + c[1]
                }
                this.setCookie("SnapABugHistory", a, 525600)
            } catch (e) {
                this.setCookie("SnapABugHistory", "0#", 525600)
            }
        },
        getHistoryCookie: function() {
            return this.getCookie("SnapABugHistory")
        },
        getHistoryCookieLastCase: function() {
            var a = this.getCookie("SnapABugHistory").split("#")[1].split("!");
            return a[a.length - 2].split(":")
        },
        getTokensCookies: function() {
            var a = "", c = this.getCookie("hubspotutk");
            c && (a += "28#" +
            c + "!");
            return a
        },
        getVisitorIsFirstTimeVisiting: function() {
            try {
                var a = this.getCookie("SnapABugHistory").split("#");
                return 1 >= parseInt(a[0])?!0 : !1
            } catch (c) {
                return !0
            }
        },
        getVisitorNumberOfPreviousChats: function() {
            try {
                return (this.getCookie("SnapABugHistory").split("#")[1].match(/!/g) || []).length
            } catch (a) {
                return 0
            }
        },
        setVisitCookie: function() {
            SnapABugChat.getVisitCookie() || SnapABugChat.setCookie("SnapABugVisit", "0#" + Math.round((new Date).getTime() / 1E3), SnapABug.getVisitorCookieLifeTime())
        },
        getVisitCookie: function() {
            return SnapABugChat.getCookie("SnapABugVisit")
        },
        incrementVisitCookie: function() {
            if (SnapABugChat.getVisitCookie()) {
                var a = SnapABugChat.getVisitCookie().split("#"), a = parseInt(a[0]) + 1 + "#" + a[1];
                SnapABugChat.setCookie("SnapABugVisit", a, SnapABug.getVisitorCookieLifeTime())
            }
        },
        getVisitorTimeOnSite: function() {
            return SnapABugChat.getVisitCookie() ? Math.round((new Date).getTime() / 1E3) - SnapABugChat.getVisitCookie().split("#")[1] : 0
        },
        getVisitorTimeOnPage: function() {
            return ( + new Date - ha) / 1E3
        },
        getVisitorNumberOfPagesVisited: function() {
            return SnapABugChat.getVisitCookie() ?
            SnapABugChat.getVisitCookie().split("#")[0] : ""
        },
        setUserEmailCookie: function(a) {
            this.deleteCookie("SnapABugUserEmail");
            for (var c = "", d = 0; d < a.length; ++d)
                c += String.fromCharCode(6^a.charCodeAt(d));
            this.setCookie("SnapABugUserEmail", encodeURIComponent("#" + c), 525600)
        },
        getUserEmailCookie: function() {
            var a = decodeURIComponent(this.getCookie("SnapABugUserEmail")), c = "";
            if ("#" == a.charAt(0))
                for (var d = 1; d < a.length; d++)
                    c += String.fromCharCode(6^a.charCodeAt(d));
            else 
                c = a;
            return c
        },
        setBannedCookie: function() {
            this.setCookie("SnapABugBanned",
            "1", 43200)
        },
        getBannedCookie: function() {
            return this.getCookie("SnapABugBanned")
        },
        deleteChatCookies: function() {
            this.setCookie("SnapABugChatSession", "", 16);
            this.setCookie("SnapABugChatView", "", 16);
            this.setCookie("SnapABugChatMessage", "", 16);
            this.deleteCookie("SnapABugChatSession");
            this.deleteCookie("SnapABugChatView");
            this.deleteCookie("SnapABugChatMessage")
        },
        setCookie: function(a, c, d) {
            var e = "";
            0 < d && (e = new Date, e.setTime(e.getTime() + 6E4 * d), e = "; expires=" + e.toGMTString());
            d = "";
            SnapABug.getDomain() && (d =
            "; domain=" + SnapABug.getDomain());
            document.cookie = a + "=" + c + e + "; path=/;" + d
        },
        deleteCookie: function(a) {
            var c = "";
            SnapABug.getDomain() && (c = "; domain=" + SnapABug.getDomain());
            document.cookie = a + "=; expires=Thu, 01-Jan-70 00:00:01 GMT;; path=/;" + c
        },
        getCookie: function(a) {
            for (var c = document.cookie.split(";"), d = 0; d < c.length; d++) {
                for (var e = c[d]; " " == e.charAt(0);)
                    e = e.substring(1, e.length);
                if (0 == e.indexOf(a))
                    return e.substring(a.length + 1, e.length)
            }
            return ""
        },
        isCookieEnabled: function() {
            var a = navigator.cookieEnabled ?
            !0: !1;
            "undefined" != typeof navigator.cookieEnabled || a || (a = "" != SnapABugChat.getVisitCookie());
            return a
        },
        populateChatAgentSurvey: function() {
            if (G) {
                SnapABugChat.populateSubChatForm('<div id="chatSurvey" style="padding-top: 10px"> \t\t\t\t\t\t<div id="chatSurveyQuestion" style="padding: 5px;">How would you rate your chat?</div> \t\t\t\t\t\t<div id="chatSurveyRating"></div> \t\t\t\t\t\t<div id="chatSurveyRatingText" style="height: 10px;" ></div> \t\t\t\t\t\t<div id="chatSurveyCommentsContainer" style="padding: 5px; display: none;"> \t\t\t\t\t\t\t<textarea id="chatSurveyComments" rows="10" cols="25" maxlength="500" style="width: 100%; height: auto; resize: none; outline: none; border: 1px solid #999999 !important; box-shadow:none !important; font-family: arial,sans-serif; font-size: 12px; font-weight: normal; font-style: normal; line-height: normal; -webkit-box-sizing: border-box; -moz-box-sizing: border-box; box-sizing: border-box;">Any comments you wish to add?</textarea> \t\t\t\t\t\t\t<div style="padding-top: 5px;"> \t\t\t\t\t\t\t\t<div style="display: inline-block;"><input id="chatSurveySubmitButton" value="Submit" type="button" style="float: none; clear: none; height: 25px; width: inherit; margin: 0px 0px 0px 4px; cursor: pointer; outline: none; border: 1px solid rgb(160, 161, 160); background-image: -webkit-gradient(linear, 0% 0%, 0% 100%, from(rgb(255, 255, 255)), to(rgb(199, 201, 199))); padding: 3px 5px; width: auto; overflow: visible; border-top-left-radius: 10px; border-top-right-radius: 10px; border-bottom-right-radius: 10px; border-bottom-left-radius: 10px; color: rgb(81, 85, 93); font-size: 12px; font-family: \'lucida grande\', \'Helvetica Neue\', Helvetica, Arial, sans-serif; text-decoration: none; vertical-align: middle; font-weight: bold; display: inline-block;"></div> \t\t\t\t\t\t\t\t<div style="display: inline-block; padding-top: 5px;"><a id="chatSurveyCancelButton" style="color: #999; text-decoration: none;" href="#">or Cancel</a></div> \t\t\t\t\t\t\t</div> \t\t\t\t\t\t</div> \t\t\t\t\t</div> \t\t\t\t\t<div id="chatSurveyResults" style="padding-top: 75px; text-align: center; display: none;">Thank you for your feedback!</div>',
                "SnapABugChat.closeChatAgentSurvey();");
                var a = document.getElementById("SnapABug_CM");
                a && (a.parentNode.style.borderColor = "white");
                n && "agent" != n ? (a = SnapABug.getMessage(22), a = a.replace("{alias}", n), document.getElementById("chatSurveyQuestion").innerHTML = a) : document.getElementById("chatSurveyQuestion").innerHTML = SnapABug.getMessage(21);
                document.getElementById("chatSurveyComments").value = SnapABug.getMessage(23);
                document.getElementById("chatSurveySubmitButton").value = SnapABug.getMessage(24);
                document.getElementById("chatSurveyCancelButton").innerHTML =
                SnapABug.getMessage(25);
                document.getElementById("chatSurveyResults").innerHTML = SnapABug.getMessage(26);
                document.getElementById("chatSurveyComments").onfocus = function() {
                    this.value === SnapABug.getMessage(23) && (this.value = "")
                };
                document.getElementById("chatSurveySubmitButton").onclick = function() {
                    document.getElementById("chatSurvey").style.display = "none";
                    document.getElementById("chatSurveyResults").style.display = "";
                    G = null;
                    SnapABugChat.sendChatAgentSurveyRating()
                };
                document.getElementById("chatSurveyCancelButton").onclick =
                function() {
                    SnapABugChat.closeChatAgentSurvey()
                };
                setTimeout(function() {
                    SnapABugChat.closeChatAgentSurvey()
                }, 36E6);
                for (var a = "", c = G.ratings, d = 0; d < c.length; d++)
                    a = a + "<img id='" + c[d].id + "' alt='" + c[d].text + "' src='" + SnapABug.getBaseURL() + "/" + c[d].defaultImg + "' style='cursor: pointer;' onmouseover='SnapABugChat.mouseoverChatAgentSurveyRating(this.id)' onmouseout='SnapABugChat.mouseoutChatAgentSurveyRating(this.id)' onmousedown='SnapABugChat.setChatAgentSurveyRating(this.id)' >";
                document.getElementById("chatSurveyRating").innerHTML =
                a
            }
        },
        mouseoverChatAgentSurveyRating: function(a) {
            if (G)
                for (var c = G.ratings, d = 0; d < c.length; d++) {
                    if (0 == G.type || 2 == G.type)
                        document.getElementById(c[d].id).src = SnapABug.getBaseURL() + "/" + c[d].selectedImg, document.getElementById("chatSurveyRatingText").innerHTML = c[d].text;
                        if (c[d].id == a) {
                            document.getElementById(c[d].id).src = SnapABug.getBaseURL() + "/" + c[d].selectedImg;
                            document.getElementById("chatSurveyRatingText").innerHTML = c[d].text;
                            break
                        }
                }
        },
        mouseoutChatAgentSurveyRating: function(a) {
            if (G) {
                a = G.ratings;
                for (var c =
                a.length - 1; 0 <= c; c--)
                    document.getElementById(a[c].id).src = SnapABug.getBaseURL() + "/" + a[c].defaultImg;
                R ? SnapABugChat.mouseoverChatAgentSurveyRating(R.id) : document.getElementById("chatSurveyRatingText").innerHTML = ""
            }
        },
        setChatAgentSurveyRating: function(a) {
            if (G) {
                for (var c = G.ratings, d = 0; d < c.length; d++)
                    if (c[d].id == a) {
                        R = c[d];
                        SnapABugChat.sendChatAgentSurveyRating();
                        SnapABugChat.mouseoutChatAgentSurveyRating(R.id);
                        break
                    }
                document.getElementById("chatSurveyCommentsContainer").style.display = ""
            }
        },
        sendChatAgentSurveyRating: function() {
            if (R) {
                var a =
                SnapABug.getBaseURL() + "/chatjs/postChat/recordSurvey?publicCaseId=" + SnapABug.getCaseId() + "&rating=" + R.id, c = document.getElementById("chatSurveyComments").value;
                c.trim() != SnapABug.getMessage(23) && (a = a + "&comments=" + encodeURIComponent(c));
                SnapABug.rpc(a, 3, !0)
            }
        },
        callbackRecordSurveyAction: function() {},
        closeChatAgentSurvey: function() {
            G = null;
            SnapABug.closeForm()
        },
        isMinimized: function() {
            return SnapABugChat.getChatWindowCookieData("isMinimized")
        },
        minimizeToggle: function() {
            var a = SnapABugChat.isMinimized();
            if ("undefined" !== typeof a) {
                if (a)
                    return SnapABugChat.unminimizeChat(), SnapABug.sendTextToChat("Visitor has un-minimized the Chat Box."), !1;
                SnapABugChat.minimizeChat();
                SnapABug.sendTextToChat("Visitor has minimized the Chat Box.");
                return !0
            }
        },
        minimizeChat: function(a) {
            if ("undefined" !== SnapABugChat.isMinimized()) {
                SnapABugChat.setChatWindowCookieData("isMinimized", !0);
                var c = document.getElementById("SnapABug_WP");
                - 1 === c.className.indexOf("minimized") && (c.className += " minimized");
                document.getElementById("SnapABug_NMNi").innerHTML =
                SnapABugChat.getChatWindowCookieData("newMinimizedMessageCount");
                var d = SnapABugChat.minimizedDimensions;
                if ("undefined" === typeof a ||!a) {
                    var e = [parseInt(c.style.left, 10), parseInt(c.style.right, 10), parseInt(c.style.top, 10), parseInt(c.style.bottom, 10)];
                    for (a = 0; a < e.length; a++)
                        if (isNaN(e[a]) || 0 === e[a])
                            e[a] =- 1;
                    SnapABugChat.setChatWindowCookieData("chatBoxPosition", e)
                }
                c.style.width = d[2] + "px";
                c.style.height = d[3] + "px";
                c.style.bottom = c.style.right = c.style.top = c.style.left = "";
                c.style.right = "0";
                c.style.bottom =
                "0";
                if (0 !== d[0] || 0 !== d[1])
                    a = document.getElementById("SnapABug_P"), a.style.left = d[0] + "px", a.style.top = d[1] + "px";
                document.getElementById("SnapABug_DB").style.display = "none";
                d = [];
                e = c.getElementsByTagName("a");
                for (a = 0; a < e.length; a++)
                    d.push(e[a]);
                e = c.getElementsByTagName("textarea");
                for (a = 0; a < e.length; a++)
                    d.push(e[a]);
                if (SnapEngage.isIE())
                    for (c = c.getElementsByTagName("iframe"), a = 0; a < c.length; a++)
                        d.push(c[a]);
                for (a = 0; a < d.length; a++)
                    d[a].setAttribute("tabIndex", "-1")
            }
        },
        unminimizeChat: function() {
            if ("undefined" !==
            SnapABugChat.isMinimized()) {
                SnapABugChat.setChatWindowCookieData("isMinimized", !1);
                var a = document.getElementById("SnapABug_WP");
                a.className = a.className.replace(/(\s|)minimized(\s|)/, "");
                var c = document.getElementById("SnapABug_NMNi");
                c || (SnapABug.initMinimizeNotification(), c = document.getElementById("SnapABug_NMNi"));
                c.innerHTML = "";
                c = SnapABugChat.minimizedDimensions;
                document.getElementById("SnapABug_NMN").style.display = "none";
                SnapABugChat.setChatWindowCookieData("newMinimizedMessageCount", 0);
                a.style.width =
                SnapABug.formWidth + "px";
                a.style.height = SnapABug.formHeight + "px";
                if (0 !== c[0] || 0 !== c[1])
                    c = document.getElementById("SnapABug_P"), c.style.left = "0", c.style.top = "0";
                document.getElementById("SnapABug_DB").style.display = "block";
                for (var c = [], d = a.getElementsByTagName("a"), e = 0; e < d.length; e++)
                    null !== d[e].getAttribute("tabIndex") && c.push(d[e]);
                d = a.getElementsByTagName("textarea");
                for (e = 0; e < d.length; e++)
                    null !== d[e].getAttribute("tabIndex") && c.push(d[e]);
                if (SnapEngage.isIE())
                    for (a = a.getElementsByTagName("iframe"),
                    e = 0; e < a.length; e++)
                        null !== a[e].getAttribute("tabIndex") && c.push(a[e]);
                for (e = 0; e < c.length; e++)
                    c[e].removeAttribute("tabIndex");
                a = SnapABugChat.getChatWindowCookieData("chatBoxPosition");
                c=!1;
                c = 0 <= a[0] && 0 <= a[2]&&-1 === a[1]&&-1 === a[3];
                e=!1;
                c && ((e = a[0] + SnapABug.formWidth > SnapABug.getWindowWidth()) || (e = a[2] + SnapABug.formHeight > SnapABug.getWindowHeight()));
                if (!c || e)
                    a = SnapABug.getDefaultChatPosition();
                SnapABug.positionForm(a[0], a[1], a[2], a[3]);
                SnapABugChat.setChatWindowCookieData("chatBoxPosition", [a[0], a[1],
                a[2], a[3]])
            }
        },
        newMinimizedMessageReceived: function() {
            var a = document.getElementById("SnapABug_NMN"), c = document.getElementById("SnapABug_NMNi");
            a && (a.style.display = "block", a = SnapABugChat.getChatWindowCookieData("newMinimizedMessageCount"), c.innerHTML=++a, SnapABugChat.setChatWindowCookieData("newMinimizedMessageCount", a))
        },
        setPreviousChatWindowState: function() {
            if (0 !== SnapABugChat.getCookie("SnapABugChatWindow").split("|").length)
                if (SnapABugChat.isMinimized()) {
                    var a = document.getElementById("SnapABug_NMN");
                    if (a) {
                        var c = SnapABugChat.getChatWindowCookieData("newMinimizedMessageCount");
                        a.style.display = 0 < c ? "block" : "none"
                    }
                    SnapABugChat.minimizeChat(!0)
                } else (a = SnapABugChat.getChatWindowCookieData("chatBoxPosition")
                    ) && 3 < a.length && SnapABug.positionForm(a[0], a[1], a[2], a[3])
        },
        setChatWindowCookieData: function(a, c) {
            var d = SnapABugChat.getCookie("SnapABugChatWindow"), e = [];
            "" === d || 3 > d.split("|").length ? (e[0] = void 0, e[1] = 0, e[2] = [ - 1, 0, - 1, 0]) : e = d.split("|");
            if (a)
                switch (a) {
                case "isMinimized":
                    e[0] = c;
                    break;
                case "newMinimizedMessageCount":
                    e[1] =
                    c;
                    break;
                case "chatBoxPosition":
                    e[2] = c
                }
            SnapABugChat.setCookie("SnapABugChatWindow", e.join("|"), 30)
        },
        getChatWindowCookieData: function(a) {
            if (!a)
                return !1;
            var c = SnapABugChat.getCookie("SnapABugChatWindow").split("|");
            if ("" === c)
                SnapABugChat.setChatWindowCookieData(), SnapABugChat.getCookie("SnapABugChatWindow").split("|");
            else {
                switch (a) {
                case "isMinimized":
                    a = "undefined" === typeof c[0] || "undefined" === c[0] ? void 0 : "true" === c[0];
                    break;
                case "newMinimizedMessageCount":
                    a = c[1] ? parseInt(c[1], 10) : 0;
                    break;
                case "chatBoxPosition":
                    if (c[2])
                        for (a =
                        c[2].split(","), c = 0; c < a.length; c++)
                            a[c] = parseInt(a[c], 10);
                    else 
                        return SnapABug.getDefaultChatPosition();
                    break;
                default:
                    a = void 0
                }
                return a
            }
        }
    }
}();
if ("undefined" == typeof YAHOO ||!YAHOO)
    var YAHOO = {};
YAHOO.namespace = function() {
    var f = arguments, k = null, u, g, r;
    for (u = 0; u < f.length; u += 1)
        for (r = ("" + f[u]).split("."), k = YAHOO, g = "YAHOO" == r[0] ? 1 : 0; g < r.length; g += 1)
            k[r[g]] = k[r[g]] || {}, k = k[r[g]];
    return k
};
YAHOO.log = function(f, k, u) {
    var g = YAHOO.widget.Logger;
    return g && g.log ? g.log(f, k, u) : !1
};
YAHOO.register = function(f, k, u) {
    var g = YAHOO.env.modules, r, d, n;
    g[f] || (g[f] = {
        versions: [],
        builds: []
    });
    g = g[f];
    r = u.version;
    u = u.build;
    d = YAHOO.env.listeners;
    g.name = f;
    g.version = r;
    g.build = u;
    g.versions ? g.versions.push(r) : !1;
    g.builds ? g.builds.push(u) : !1;
    g.mainClass = k;
    for (n = 0; n < d.length; n += 1)
        d[n](g);
    k ? (k.VERSION = r, k.BUILD = u) : YAHOO.log("mainClass is undefined for module " + f, "warn")
};
YAHOO.env = YAHOO.env || {
    modules: [],
    listeners: []
};
YAHOO.env.getVersion = function(f) {
    return YAHOO.env.modules[f] || null
};
YAHOO.env.ua = function() {
    var f = function(f) {
        var d = 0;
        return parseFloat(f.replace(/\./g, function() {
            return 1 == d++?"" : "."
        }))
    }, k = {
        ie: 0,
        opera: 0,
        gecko: 0,
        webkit: 0,
        mobile: null,
        air: 0,
        caja: navigator.cajaVersion,
        secure: !1,
        os: null
    }, u = navigator && navigator.userAgent, g = window && window.location, g = g && g.href;
    k.secure = g && 0 === g.toLowerCase().indexOf("https");
    if (u) {
        /windows|win32/i.test(u) ? k.os = "windows" : /macintosh/i.test(u) && (k.os = "macintosh");
        /KHTML/.test(u) && (k.webkit = 1);
        if ((g = u.match(/AppleWebKit\/([^\s]*)/)) && g[1]) {
            k.webkit =
            f(g[1]);
            if (/ Mobile\//.test(u))
                k.mobile = "Apple";
            else if (g = u.match(/NokiaN[^\/]*/))
                k.mobile = g[0];
            if (g = u.match(/AdobeAIR\/([^\s]*)/))
                k.air = g[0]
        }
        if (!k.webkit)
            if ((g = u.match(/Opera[\s\/]([^\s]*)/)) && g[1]) {
                if (k.opera = f(g[1]), g = u.match(/Opera Mini[^;]*/))
                    k.mobile = g[0]
            } else if ((g = u.match(/MSIE\s([^;]*)/)) && g[1])
                k.ie = f(g[1]);
            else if (g = u.match(/Gecko\/([^\s]*)/))
                k.gecko = 1, (g = u.match(/rv:([^\s\)]*)/)) && g[1] && (k.gecko = f(g[1]))
    }
    return k
}();
(function() {
    YAHOO.namespace("util", "widget", "example");
    if ("undefined" !== typeof YAHOO_config) {
        var f = YAHOO_config.listener, k = YAHOO.env.listeners, u=!0, g;
        if (f) {
            for (g = 0; g < k.length; g++)
                if (k[g] == f) {
                    u=!1;
                    break
                }
            u && k.push(f)
        }
    }
})();
YAHOO.lang = YAHOO.lang || {};
(function() {
    var f = YAHOO.lang, k = Object.prototype, u = [], g = ["toString", "valueOf"], r = {
        isArray: function(d) {
            return "[object Array]" === k.toString.apply(d)
        },
        isBoolean: function(d) {
            return "boolean" === typeof d
        },
        isFunction: function(d) {
            return "function" === typeof d || "[object Function]" === k.toString.apply(d)
        },
        isNull: function(d) {
            return null === d
        },
        isNumber: function(d) {
            return "number" === typeof d && isFinite(d)
        },
        isObject: function(d) {
            return d && ("object" === typeof d || f.isFunction(d)) ||!1
        },
        isString: function(d) {
            return "string" ===
            typeof d
        },
        isUndefined: function(d) {
            return "undefined" === typeof d
        },
        _IEEnumFix: YAHOO.env.ua.ie ? function(d, n) {
            var x, q, v;
            for (x = 0; x < g.length; x += 1)
                q = g[x], v = n[q], f.isFunction(v) && v != k[q] && (d[q] = v)
        }
        : function() {},
        extend: function(d, g, x) {
            if (!g ||!d)
                throw Error("extend failed, please check that all dependencies are included.");
            var q = function() {}, v;
            q.prototype = g.prototype;
            d.prototype = new q;
            d.prototype.constructor = d;
            d.superclass = g.prototype;
            g.prototype.constructor == k.constructor && (g.prototype.constructor = g);
            if (x) {
                for (v in x)
                    f.hasOwnProperty(x,
                    v) && (d.prototype[v] = x[v]);
                f._IEEnumFix(d.prototype, x)
            }
        },
        augmentObject: function(d, g) {
            if (!g ||!d)
                throw Error("Absorb failed, verify dependencies.");
            var k = arguments, q, v = k[2];
            if (v&&!0 !== v)
                for (q = 2; q < k.length; q += 1)
                    d[k[q]] = g[k[q]];
            else {
                for (q in g)
                    !v && q in d || (d[q] = g[q]);
                f._IEEnumFix(d, g)
            }
        },
        augmentProto: function(d, g) {
            if (!g ||!d)
                throw Error("Augment failed, verify dependencies.");
            var k = [d.prototype, g.prototype], q;
            for (q = 2; q < arguments.length; q += 1)
                k.push(arguments[q]);
            f.augmentObject.apply(this, k)
        },
        dump: function(d,
        g) {
            var k, q, v = [];
            if (f.isObject(d)) {
                if (d instanceof Date || "nodeType"in d && "tagName"in d)
                    return d;
                if (f.isFunction(d))
                    return "f(){...}"
            } else 
                return d + "";
            g = f.isNumber(g) ? g : 3;
            if (f.isArray(d)) {
                v.push("[");
                k = 0;
                for (q = d.length; k < q; k += 1)
                    f.isObject(d[k]) ? v.push(0 < g ? f.dump(d[k], g - 1) : "{...}") : v.push(d[k]), v.push(", ");
                1 < v.length && v.pop();
                v.push("]")
            } else {
                v.push("{");
                for (k in d)
                    f.hasOwnProperty(d, k) && (v.push(k + " => "), f.isObject(d[k]) ? v.push(0 < g ? f.dump(d[k], g - 1) : "{...}") : v.push(d[k]), v.push(", ")); 1 < v.length && v.pop();
                    v.push("}")
            }
            return v.join("")
        }, substitute : function(d, g, k) {
            for (var q, v, r, u, C, z = [], w; ; ) {
                q = d.lastIndexOf("{"); if (0 > q)break; v = d.indexOf("}", q); if (q + 1 >= v)break; u = w = d.substring(q + 1, v); C = null; r = u.indexOf(" "); - 1 < r && (C = u.substring(r + 1), u = u.substring(0, r)); r = g[u]; k && (r = k(u, r, C)); f.isObject(r) ? f.isArray(r) ? r = f.dump(r, parseInt(C, 10)) : (C = C || "", u = C.indexOf("dump"), - 1 < u && (C = C.substring(4)), w = r.toString(), r = "[object Object]" === w||-1 < u ? f.dump(r, parseInt(C, 10)) : w) : f.isString(r) || f.isNumber(r) || (r = "~-" + z.length + "-~",
                z[z.length] = w); d = d.substring(0, q) + r + d.substring(v + 1)
            }
            for (q = z.length - 1; 0 <= q; q -= 1)d = d.replace(new RegExp("~-" + q + "-~"), "{" + z[q] + "}", "g"); return d
        }, trim : function(d) {
            try {
                return d.replace(/^\s+|\s+$/g, "")
            } catch (f) {
                return d
            }
        }, merge : function() {
            var d = {}, g = arguments, k = g.length, q; for (q = 0; q < k; q += 1)f.augmentObject(d, g[q], !0); return d
        }, later : function(d, g, k, q, r) {
            d = d || 0; g = g || {}; var B = k, A = q, C; f.isString(k) && (B = g[k]); if (!B)throw new TypeError("method undefined"); A&&!f.isArray(A) && (A = [q]); k = function() {
                B.apply(g, A ||
                u)
            }; C = r ? setInterval(k, d) : setTimeout(k, d); return {
                interval : r, cancel : function() {
                    this.interval ? clearInterval(C) : clearTimeout(C)
                }
            }
        }, isValue : function(d) {
            return f.isObject(d) || f.isString(d) || f.isNumber(d) || f.isBoolean(d)
        }
    }; f.hasOwnProperty = k.hasOwnProperty ? function(d, f) {
        return d && d.hasOwnProperty(f)
    } : function(d, g) {
        return !f.isUndefined(d[g]) && d.constructor.prototype[g] !== d[g]
    }; r.augmentObject(f, r, !0); YAHOO.util.Lang = f; f.augment = f.augmentProto; YAHOO.augment = f.augmentProto; YAHOO.extend = f.extend
})();
YAHOO.register("yahoo", YAHOO, {
    version : "2.8.2r1", build : "7"
});
YAHOO.util.Get = function() {
    var f = {}, k = 0, u = 0, g=!1, r = YAHOO.env.ua, d = YAHOO.lang, n = function(d, f, g) {
        d = (g || window).document.createElement(d); for (var k in f)f[k] && YAHOO.lang.hasOwnProperty(f, k) && d.setAttribute(k, f[k]); return d
    }, x = function(f, g, k) {
        f = {
            id : "yui__dyn_" + u++, type : "text/javascript", src : f
        }; k && d.augmentObject(f, k); return n("script", f, g)
    }, q = function(d, f) {
        return {
            tId : d.tId, win : d.win, data : d.data, nodes : d.nodes, msg : f, purge : function() {
                w(this.tId)
            }
        }
    }, v = function(g, k) {
        var p = f[k]; (p = d.isString(g) ? p.win.document.getElementById(g) :
        g) || B(k, "target node not found: " + g); return p
    }, B = function(d, g) {
        var k = f[d]; k.onFailure && k.onFailure.call(k.scope || k.win, q(k, g))
    }, A = function(d) {
        var g = f[d]; g.finished=!0; g.aborted ? B(d, "transaction " + d + " was aborted") : g.onSuccess && g.onSuccess.call(g.scope || g.win, q(g))
    }, C = function(d) {
        d = f[d]; d.onTimeout && d.onTimeout.call(d.scope || d, q(d))
    }, z = function(g, k) {
        var p = f[g]; p.timer && p.timer.cancel(); if (p.aborted)B(g, "transaction " + g + " was aborted"); else {
            k ? (p.url.shift(), p.varName && p.varName.shift()) : (p.url = d.isString(p.url) ?
            [p.url] : p.url, p.varName && (p.varName = d.isString(p.varName) ? [p.varName] : p.varName)); var q = p.win, w = q.document.getElementsByTagName("head")[0], H; if (0 === p.url.length)if ("script" === p.type && r.webkit && 420 > r.webkit&&!p.finalpass&&!p.varName) {
                var F = x(null, p.win, p.attributes); F.innerHTML = 'YAHOO.util.Get._finalize("' + g + '");'; p.nodes.push(F); w.appendChild(F)
            } else A(g); else {
                F = p.url[0]; if (!F)return p.url.shift(), z(g); p.timeout && (p.timer = d.later(p.timeout, p, C, g)); if ("script" === p.type)H = x(F, q, p.attributes); else {
                    H =
                    p.attributes; var D = {
                        id : "yui__dyn_" + u++, type : "text/css", rel : "stylesheet", href : F
                    }; H && d.augmentObject(D, H); H = n("link", D, q)
                }
                W(p.type, H, g, F, q, p.url.length); p.nodes.push(H); p.insertBefore ? (w = v(p.insertBefore, g)) && w.parentNode.insertBefore(H, w) : w.appendChild(H); (r.webkit || r.gecko) && "css" === p.type && z(g, F)
            }
        }
    }, w = function(d) {
        if (f[d]) {
            var g = f[d], k = g.nodes, n = k.length, q = g.win.document.getElementsByTagName("head")[0], r, u; g.insertBefore && (d = v(g.insertBefore, d)) && (q = d.parentNode); for (d = 0; d < n; d += 1) {
                r = k[d]; if (r.clearAttributes)r.clearAttributes();
                else for (u in r)delete r[u]; q.removeChild(r)
            }
            g.nodes = []
        }
    }, D = function(l, n, p) {
        var q = "q" + k++; p = p || {}; if (0 === k%YAHOO.util.Get.PURGE_THRESH&&!g) {
            g=!0; for (var r in f) {
                var u = f[r]; u.autopurge && u.finished && (w(u.tId), delete f[r])
            }
            g=!1
        }
        f[q] = d.merge(p, {
            tId : q, type : l, url : n, finished : !1, aborted : !1, nodes : []
        }); n = f[q]; n.win = n.win || window; n.scope = n.scope || n.win; n.autopurge = "autopurge"in n ? n.autopurge : "script" === l?!0 : !1; p.charset && (n.attributes = n.attributes || {}, n.attributes.charset = p.charset); d.later(0, n, z, q); return {
            tId : q
        }
    },
    W = function(g, k, p, n, q, u, v) {
        var x = v || z; if (r.ie)k.onreadystatechange = function() {
            var d = this.readyState; if ("loaded" === d || "complete" === d)k.onreadystatechange = null, x(p, n)
        }; else if (r.webkit) {
            if ("script" === g)if (420 <= r.webkit)k.addEventListener("load", function() {
                x(p, n)
            }); else {
                var w = f[p]; w.varName ? (g = YAHOO.util.Get.POLL_FREQ, w.maxattempts = YAHOO.util.Get.TIMEOUT / g, w.attempts = 0, w._cache = w.varName[0].split("."), w.timer = d.later(g, w, function(d) {
                    d = this._cache; var f = d.length, g = this.win, k; for (k = 0; k < f; k += 1)if (g = g[d[k]],
                    !g) {
                        this.attempts++; this.attempts++>this.maxattempts && (w.timer.cancel(), B(p, "Over retry limit, giving up")); return 
                    }
                    w.timer.cancel(); x(p, n)
                }, null, !0)) : d.later(YAHOO.util.Get.POLL_FREQ, null, x, [p, n])
            }
        } else k.onload = function() {
            x(p, n)
        }
    }; return {
        POLL_FREQ : 10, PURGE_THRESH : 20, TIMEOUT : 2E3, _finalize : function(f) {
            d.later(0, null, A, f)
        }, abort : function(g) {
            g = d.isString(g) ? g : g.tId; if (g = f[g])g.aborted=!0
        }, script : function(d, f) {
            return D("script", d, f)
        }, css : function(d, f) {
            return D("css", d, f)
        }
    }
}();
YAHOO.register("get", YAHOO.util.Get, {
    version : "2.8.2r1", build : "7"
}); var SnapEngage = SnapABug, SnapEngageChat = SnapABugChat; if (typeof SnapEngageNoInit == "undefined") {
    SnapEngage.setStartChat('Hello! How can I help you?'); SnapEngage.setDomain('jimdo.com'); if (typeof SnapEngage.setEnableInternationalPhoneCalls !== "undefined") {
        SnapEngage.setEnableInternationalPhoneCalls(true); 
    }
    if (typeof SnapEngage.messagesInit !== "undefined") {
        SnapEngage.messagesInit('{\"isSingle\":true,\"en\":{\"0\":\"Help\",\"1\":\"Your email:\",\"2\":\"Your question:\",\"3\":\"Include a snapshot of the current page\",\"4\":\"Send\",\"5\":\"Send a screen shot of what I am seeing on the web page so that the support team can provide a fast answer.\",\"6\":\"Preparing for snapshot...<br /><br />Please click \\"Run\\" or \\"Allow\\" to let our applet send a snapshot of the web page to the support team.\",\"7\":\"Thank you,<br />Your message has been sent.\",\"8\":\"Sorry, your message couldn\'t be sent.<br />We are investigating the issue.<br />Please try again later.\",\"9\":\"Uploading...\",\"10\":\"Close\",\"11\":\"You have already submitted this message.\",\"12\":\"Email us\",\"13\":\"Chat with us\",\"14\":\"Type your message here and press [enter] to send...\",\"15\":\"Me\",\"16\":\"How can we help?\",\"17\":\"Do any of these help?\",\"18\":\"Next\",\"19\":\"Back\",\"20\":\"None of these helped...\",\"21\":\"How would you rate your chat?\",\"22\":\"How would you rate your chat with {alias}?\",\"23\":\"Any comments you wish to add?\",\"24\":\"Submit\",\"25\":\"or Cancel\",\"26\":\"Thank you for your feedback!\",\"27\":\"Not helpful\",\"28\":\"Needs work\",\"29\":\"So-so\",\"30\":\"Helpful\",\"31\":\"Very helpful\",\"32\":\"Poor\",\"33\":\"Ok\",\"34\":\"Great\",\"35\":\"Just a moment...\",\"36\":\"Minimize\",\"37\":\"End Chat\",\"38\":\"Call from my computer \",\"39\":\"Call me on my phone\",\"40\":\"Return to chat\",\"41\":\"Call me\"}}'); 
    }
    SnapEngage.add("6ae0477d-5d2b-4417-874d-212734348eda", "en"); 
}
