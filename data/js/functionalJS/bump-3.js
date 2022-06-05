/* bump_cloud_build:smp-external-feedback@bbc.co.uk */
define("mp/modules/policies", [], function() {
    var e = {
        iplayer: {
            guidanceCleared: !1,
            siteID: "iPlayer",
            defaultUiClass: "iplayer",
            mediasetClass: "pc"
        },
        news: {
            ui: {
                colour: "#ff0000",
                alternateColour: "#ff0000"
            },
            guidanceCleared: !0,
            siteID: "News",
            defaultUiClass: "news",
            mediasetClass: "journalism",
            preferHtmlOnMobile: !0
        },
        sport: {
            ui: {
                colour: "#eed71c",
                foreColour: "#000000",
                alternateColour: "#0087ff",
                alternateForeColour: "#ffffff"
            },
            siteID: "Sport",
            defaultUiClass: "sport",
            guidanceCleared: !0,
            mediasetClass: "journalism"
        },
        background: {
            noTracking: !0,
            backgroundPlayer: !0,
            mediasetClass: "journalism",
            responsive: !0,
            loop: !0,
            muted: !0,
            autoplay: !0,
            superResponsive: !0,
            ui: {
                controls: {
                    enabled: !1
                },
                errorDialog: {
                    enabled: !1
                },
                buffer: {
                    enabled: !1
                },
                cta: {
                    enabled: !1
                },
                fullscreen: {
                    enabled: !1,
                    dblclick: !1
                },
                poster: {
                    availableWhenPaused: !0,
                    availableWhenSettingUp: !0
                },
                displayCover: !0
            }
        }
    };
    return e
});
define("mp/classes/errors/error", [], function() {
    function e(e, t, i) {
        var n = "en";
        if (e) {
            n = e.locale();
            if (e.overrideError)
                var a = e.overrideError[t]
        }
        if (window.embeddedMedia.language) {
            var o = window.embeddedMedia.language.t(t, n, a);
            if (o != t)
                return o
        }
        return i || "Unknown Error"
    }
    function t(t, i, n) {
        n || (n = {});
        this._c = t;
        this._s = n.severity || "critical";
        this._d = e(i, t + "", n.description);
        return this
    }
    t.prototype = {
        code: function() {
            return this._c
        },
        description: function() {
            return this._d
        },
        detail: function() {
            return ""
        },
        severity: function() {
            return this._s
        },
        responsibility: function() {
            return "client"
        }
    };
    return t
});
define("mp/classes/settings", ["jquery-1.9", "mp/modules/policies", "mp/classes/errors/error"], function(e, t, i) {
    function n(e) {
        e && e.getFullYear || (e = new Date);
        var t = e.getFullYear(), i = e.valueOf(), n = new Date(i);
        n.setMonth(2);
        n.setDate(31 - Math.floor(5 * t / 4 + 4)%7);
        n.setUTCHours(1);
        n.setUTCMinutes(0);
        n.setUTCSeconds(0);
        var a = new Date(i);
        a.setMonth(9);
        a.setDate(31 - Math.floor(5 * t / 4 + 1)%7);
        a.setUTCHours(1);
        a.setUTCMinutes(0);
        a.setUTCSeconds(0);
        return e > n && a > e
    }
    var a = {
        locale: {
            lang: "en",
            path: "//emp.bbci.co.uk/emp/translations/1.0.4/{lang}.xml"
        },
        colour: "#f54997",
        foreColour: "#ffffff",
        alternateColour: "#f54997",
        alternateForeColour: "#ffffff"
    }, o = {
        container: {},
        playlist: "",
        startTime: 0,
        volume: .7,
        overrideHoldingImage: "",
        product: "iplayer",
        today: new Date,
        guidance: {},
        mediator: {},
        mediaProfile: "",
        includeAppClickHandler: !0,
        plugins: {
            player: {
                width: "640px",
                height: "360px",
                poster: ""
            },
            contextMenu: {},
            debugWindow: {},
            toLoad: []
        },
        configVars: {},
        quality: "medium",
        uiClass: "",
        customIStatsURL: "",
        customDAXURL: "//sa.bbc.co.uk/bbc/bbc/s?",
        enableStatsReporting: !0,
        enableDaxReporting: !0,
        customRdotBaseURL: "http://r.bbci.co.uk/e",
        autoplay: !1,
        enableRdotReporting: !0,
        enableSonarReporting: !0,
        statsObject: {},
        embedPageURL: location.href,
        forceGuidance: "",
        aaampGuidance: "",
        appName: "",
        appType: "",
        counterName: "smp.player.page",
        siteID: "",
        significantTime: 10,
        playerType: "",
        daxSessionID: "",
        blacklistedPlugins: [],
        suppressType: [],
        suppressItemKind: [],
        sessionID: "",
        overrideError: {},
        overrideErrorLink: {},
        pageType: "",
        parentSection: 0,
        edition: "",
        uniqueID: 0,
        ui: {},
        webcastData: {},
        allowMainlineProfiles: !0
    }, r = function(t, i, n) {
        var a = "object" == typeof n;
        if (a) {
            "object" == typeof o[i] && e.extend(!0, t[i], n);
            "playlistObject" == i && (t[i] = n)
        } else 
            t[i] = n
    }, s = function(e) {
        return this.init(e)
    };
    s.prototype = {
        set: function() {
            var t = this;
            if (2 === arguments.length) {
                var i = arguments[0], a = arguments[1];
                r(t, i, a)
            } else 
                "object" == typeof arguments[0] && e.each(arguments[0], function(e, i) {
                    r(t, e, i)
                });
            (document.cookie + "").match(/bumpForceAlternate=true/) && (t.playerType = "alternate");
            for (i in t.webcastData)
                if (t.webcastData.hasOwnProperty(i)) {
                    var o = t.webcastData[i];
                    "[object Date]" === Object.prototype.toString.call(o) && (t.webcastData[i] = o.getTime() / 1e3)
                }
            t.isBSTime = n(t.today);
            return t
        },
        init: function(i) {
            var n = this;
            n.reset();
            "object" != typeof i && (i = {});
            var o = i.product;
            o && t[o] || (o = n.product);
            var s = t[o];
            e.each(s, function(e, t) {
                r(n, e, t)
            });
            n.ui = e.extend(!0, {}, a, s.ui);
            (document.cookie + "").match(/NO-SA=1/) && (n.noTracking=!0);
            return n.set(i)
        },
        reset: function() {
            var t = this, i = e.extend(!0, {}, o);
            e.each(i, function(e, i) {
                t[e] = i
            });
            return t
        },
        validate: function() {
            var e = this;
            if (!t[e.product]) {
                e.dispatch(6007);
                e.product = o.product
            }
            if (!e.container || void 0 === e.container.get(0) || 0 === e.container.length)throw "Error 6009 DOM element not found"
        },
        dispatch: function(t) {
            var n = this, a = e.Event("error");
            a.errorData = new i(t, n);
            n.trigger(a)
        },
        trigger: function(e) {
            var t = this;
            t.container && "function" == typeof t.container.trigger && t.container.trigger(e)
        },
        locale: function() {
            return this.ui.locale.lang
        }
    };
    return s
});
define("mp/modules/device", [], function() {
    function e() {
        for (var e = window.embeddedMedia.demi, t = e.ua, i = function() {
            var e = /mac/.test(t), i = {
                win: /windows/.test(t)&&!/windows phone/.test(t),
                mac: e&&!/like mac os x/.test(t),
                linux: /linux/.test(t),
                unix: /x11/.test(t),
                ios: /like mac os x/.test(t),
                android: /android/.test(t),
                blackberry: /blackberry/.test(t) || /playbook/.test(t),
                playbook: /playbook/.test(t),
                winphone: /windows phone/.test(t),
                kindleSilk: /silk/.test(t),
                macsafari: /safari/.test(t) && e&&!/chrome/.test(t)
            };
            for (var n in i)
                i[n] && (i.name = n);
            if (i.win) / nt 5\. / .test(t) ? i.win = "xp" : /nt 6\.0/.test(t) ? i.win = "vista" : /nt 6\.1/.test(t) ? i.win = "7" : /nt 6\.2/.test(t) && (i.win = "8");
            else if (i.mac) / os x 10.0 / .test(t) ? i.mac = "cheetah" : /os x 10.1/.test(t) ? i.mac = "puma" : /os x 10.2/.test(t) ? i.mac = "jaguar" : /os x 10.3/.test(t) ? i.mac = "panther" : /os x 10.4/.test(t) ? i.mac = "tiger" : /os x 10.5/.test(t) ? i.mac = "leopard" : /os x 10.6/.test(t) ? i.mac = "snow leopard" : /os x 10.7/.test(t) ? i.mac = "lion" : /os x 10.8/.test(t) && (i.mac = "mountain lion");
            else if (i.ios) / os 3 / .test(t) ? i.ios = "3" : /os 4/.test(t) ? i.ios = "4" : /os 5/.test(t) ? i.ios = "5" : /os 6/.test(t) && (i.ios = "6");
            else if (i.android) {
                t.match(/android[^\d]*(([\d|\.]+\d))/);
                i.androidVersion = parseFloat(RegExp.$1)
            } else 
                i.winphone && (/windows phone 8.0/.test(t) ? i.winphone = "windows phone 8.0" : /windows phone 8.1/.test(t) && (i.winphone = "windows phone 8.1"));
            return i
        }(), n = {
            connection: {
                mobile: e.isMobileNetwork
            },
            isTabletDevice: e.isTabletDevice,
            isMobileDevice: e.isMobileDevice,
            isMobileOrTablet: e.isMobileDevice || e.isTabletDevice || i.ios || i.android || i.blackberry || i.playbook || i.winphone || i.kindleSilk,
            hasTouchScreen: e.isMobileDevice || e.isTabletDevice || i.ios || i.android || i.blackberry || i.playbook || i.winphone || i.kindleSilk,
            hasStylus: e.hasStylus,
            os: i,
            hasHlsVodSupport: !1,
            supportsFlash: !0
        }, a = ["ios", "macsafari"], o = 0; a.length > o; o++)
            a[o] == n.os.name && (n.hasHlsVodSupport=!0);
        n.allowHlsVod = function() {
            n.hasHlsVodSupport || n.os.androidVersion >= 4.1 && (n.hasHlsVodSupport=!0)
        };
        n.useOpaque = /chrome/.test(t) || /trident/.test(t);
        n.waitForLoad = n.os.winphone || 8 === document.documentMode || 9 === document.documentMode;
        if (n.isMobileDevice)
            n.supportsFlash=!1;
        else {
            var r = ["ios", "android", "winphone", "kindleSilk", "playbook"];
            for (o = 0; r.length > o; o++)
                r[o] == n.os.name && (n.supportsFlash=!1)
        }
        return n
    }
    return {
        get: e
    }
});
define("mp/modules/html5", [], function() {
    function e(e) {
        var t = document.createElement(e);
        return !(!t ||!t.canPlayType)
    }
    return {
        media: {
            audio: e("audio"),
            video: e("video")
        },
        audio: {},
        video: {}
    }
});
define("mp/classes/settingsbuilder", ["jquery-1.9", "mp/modules/device"], function(e, t) {
    var i = {
        mediasets: {
            pc: {
                flash: {
                    desktop: "pc",
                    cellular: "mobile-cellular-baseline",
                    mobile: "mobile-phone-baseline",
                    tablet: "mobile-tablet-baseline"
                },
                html5: {
                    desktop: "apple-ipad-hls",
                    cellular: "apple-iphone4-ipad-hls-3g",
                    mobile: "apple-iphone4-hls",
                    tablet: "apple-ipad-hls"
                }
            },
            journalism: {
                flash: {
                    desktop: "journalism-pc",
                    cellular: "journalism-flash-3g",
                    mobile: "journalism-flash-3g",
                    tablet: "journalism-flash-3g"
                },
                html5: {
                    desktop: "journalism-http-tablet",
                    cellular: "journalism-http-3g",
                    mobile: "journalism-http-mobile",
                    tablet: "journalism-http-tablet"
                }
            }
        },
        mediaSet: function(e, i) {
            if (e.mediator.overrideMediaSet)
                return e.mediator.overrideMediaSet;
            var n = e.mediaProfile;
            "pc" != n && "journalism" != n && (n = e.mediasetClass);
            "pc" != n && (n = "journalism");
            e.mediasets = this.mediasets[n];
            var a = t.get(), o = e.mediasets[i || "flash"];
            return a.isMobileOrTablet ? a.connection.mobile ? o.cellular : a.isTabletDevice ? o.tablet : o.mobile : o.desktop
        },
        mediatorHref: function(t, i, n) {
            n || (n = "{protocol}://{host}/mediaselector/5/select/version/2.0/mediaset/{mediaset}/vpid/{id}");
            var a = {
                href: n,
                protocol: "https:" == location.protocol ? "https": "http",
                host: "open.live.bbc.co.uk",
                mediaset: i
            };
            e.extend(!0, a, t.mediator);
            var o = a.href;
            "https" != a.protocol && (a.protocol = "http");
            e.each(a, function(e, t) {
                o = o.replace("{" + e + "}", t)
            });
            return o
        }
    };
    return i
});
define("mp/classes/embedPopup", ["jquery-1.9"], function(e) {
    var t = {
        show: function(i) {
            t._embedBox && t.close(!0);
            var n = i._settings;
            window.embeddedMedia.opened();
            var a = {
                width: "100%",
                height: "100%",
                margin: "0px",
                position: "fixed",
                "z-index": 1e3,
                top: "0px",
                left: "0px",
                "background-color": "#000000",
                zoom: 1,
                filter: "alpha(opacity=50)",
                opacity: .5
            };
            t._background = e('<div id="embedBackground"></div>').css(a).appendTo(document.body);
            var o = {
                "-webkit-box-orient": "horizontal",
                "-webkit-box-pack": "center",
                "-webkit-box-align": "center",
                "-moz-box-orient": "horizontal",
                "-moz-box-pack": "center",
                "-moz-box-align": "center",
                display: "box",
                "box-orient": "horizontal",
                "box-pack": "center",
                "box-align": "center",
                position: "fixed",
                width: "400px",
                height: "160px",
                bottom: "0",
                left: "0",
                top: "0",
                right: "0",
                margin: "auto",
                "vertical-align": "middle",
                "text-align": "center",
                color: "white",
                "font-size": "22px",
                "z-index": 1001,
                "background-color": "#262626",
                "font-family": "Arial, sans-serif"
            };
            t._embedBox = e('<div id="embedBox" role="alertdialog" aria-labelledby="embedMediaTitle" aria-describedby="embedMediaDescription"></div>').css(o).appendTo(document.body);
            var r = n.locale(), s = window.embeddedMedia.language, l = s.t("embedTitle", r), d = s.t("embedDescription", r), c = s.t("embedTerms", r), p = s.t("embedClose", r), u = e('<span id="embedMediaTitle" style="position:absolute;left:0px;margin-left:10%;margin-top:6%">' + l + '</span><span id="embedMediaDescription" style="display:none" aria-hidden="true">' + d + "</span>");
            u.appendTo(t._embedBox);
            var m = {
                position: "absolute",
                width: "80%",
                height: "30px",
                top: "50%",
                left: 0,
                "margin-top": "-15px",
                "margin-left": "10%",
                "font-size": "14px",
                color: "#848484"
            }, f = encodeURIComponent(n.product), h = encodeURIComponent(i.playlist.title), g = n.externalEmbedUrl;
            if (g)
                for (var y = i.playlist.items, b = 0; y.length > b; b++) {
                    var v = y[b].kind;
                    if ("programme" == v || "radioProgramme" == v) {
                        g = g.replace("{vpid}", y[b].identifier);
                        break
                    }
                } else if (i.playlist.url) {
                var w = encodeURIComponent(i.playlist.url);
                g = "http://emp.bbc.co.uk/emp/embed/smpEmbed.html?playlist=" + w + "&title=" + h + "&product=" + f
            }
            if (g) {
                "en" != r && (g += "&lang=" + r);
                var k = '<iframe width="400" height="500" frameborder="0" src="' + g + '"></iframe>', _ = e('<input type="text" id="embedText" />').css(m).appendTo(t._embedBox);
                e("#embedText")[0].value = k
            } else 
                t.close();
            var x = {
                position: "absolute",
                bottom: 0,
                left: 0,
                "margin-bottom": "6%",
                "margin-left": "10%",
                "font-size": "0.6em",
                color: "#ffffff",
                "text-decoration": "none"
            }, P = e('<a target="_blank" href="http://www.bbc.co.uk/terms/additional_emp.shtml">' + c + "</a> ");
            P.css(x).appendTo(t._embedBox);
            _.click(function() {
                e(this).select()
            });
            setTimeout(function() {
                _.focus()
            }, 100);
            var T = {
                position: "absolute",
                right: "0px",
                "font-size": "0.8em",
                top: "0px",
                "margin-top": "10px",
                "margin-right": "10px",
                color: "#ffffff",
                "text-decoration": "none",
                cursor: "pointer"
            }, C = e('<a aria-label="' + p + '" href="#">x</a> ');
            C.css(T).appendTo(t._embedBox);
            C.click(function(e) {
                e.preventDefault && e.preventDefault();
                e.returnValue=!1;
                t.close()
            });
            t._tabList = [_, P, C];
            t._currentTab = 0;
            for (var S = function(e) {
                e = e || window.event;
                var i = e.keyCode || e.which;
                27 === i && t.close();
                if (9 == i || 13 == i) {
                    e.preventDefault && e.preventDefault();
                    e.returnValue=!1;
                    if (9 == i) {
                        var n = t._tabList.length - 1;
                        t._currentTab = e.shiftKey ? 0 === t._currentTab ? n : t._currentTab - 1 : t._currentTab === n ? 0 : t._currentTab + 1;
                        setTimeout(function() {
                            t._tabList[t._currentTab].focus()
                        }, 200)
                    } else 
                        this.click()
                }
            }, b = 0; t._tabList.length > b; b++)
                t._tabList[b].keydown(S)
        },
        close: function(e) {
            var t = this;
            e || window.embeddedMedia.closed();
            if (t._background) {
                t._background.remove();
                t._background = void 0
            }
            if (t._embedBox) {
                t._embedBox.remove();
                t._embedBox = void 0
            }
        }
    };
    return t
});
define("mp/classes/players/player", [], function() {
    function e() {
        this.includePoster=!0;
        this._volume = 1;
        return this
    }
    e.prototype = {
        name: function() {
            return this._name
        },
        setPoster: function(e, t) {
            t.poster(e)
        },
        uiDefaults: function(e, t) {
            var i = this._settings, n = i.ui;
            if ("iplayer" == i.product) {
                n.cta || (n.cta = {});
                n.cta.iplayer!==!1 && (n.cta.iplayer=!0)
            }("iplayer" == i.product || "sport" == i.product) && i.allowCasting && (n.chromecast || (n.chromecast = {
                enabled: !0
            }));
            if (!i.guidanceCleared) {
                n.guidance || (n.guidance = {});
                n.guidance.pinRequired!==!1 && (n.guidance.pinRequired=!0)
            }
            if (e) {
                n.controls || (n.controls = {
                    volumeSlider: !1
                });
                if (t) {
                    n.guidance || (n.guidance = {});
                    n.guidance.useFlashGuidance=!1
                }
                n.controls.popout=!1;
                n.controls.volumeSlider || (n.controls.volumeSlider=!1);
                i.volume = 1
            }
            try {
                var a = /[; ]ckps_smpSettings=([^\s;]*)/gim;
                a.test(" " + document.cookie);
                if (RegExp.$1) {
                    var o = JSON.parse(decodeURIComponent(RegExp.$1));
                    if (o.controls) {
                        n.controls || (n.controls = {});
                        n.controls.timeout = o.controls.timeout;
                        n.controls.sizeAdjust = o.controls.sizeAdjust;
                        n.controls.spaceControlsPlayback|=o.controls.spaceControlsPlayback
                    }
                    if (o.subtitles) {
                        n.subtitles || (n.subtitles = {});
                        n.subtitles.contrast = o.subtitles.contrast
                    }
                }
            } catch (r) {}
        },
        destroy: function() {
            this._container && this._container.empty()
        },
        call: function() {},
        play: function() {
            this.call("play")
        },
        stop: function() {
            this.call("stop")
        },
        suspend: function() {},
        pause: function() {
            this.call("pause")
        },
        muted: function(e) {
            "undefined" != typeof e && this.call("muted", e);
            return this._muted
        },
        volume: function(e) {
            "number" == typeof e && this.call("volume", e);
            return this._volume
        },
        duration: function() {
            return this._duration
        },
        ended: function() {
            return this._ended
        },
        seeking: function() {
            return this._seeking
        },
        paused: function() {
            return this._paused
        },
        kind: function() {
            return this._kind
        },
        currentTime: function(e) {
            "number" == typeof e && this.call("currentTime", e);
            return this._currentTime
        },
        setData: function() {},
        webcastData: function(e) {
            for (var t in e)
                if (e.hasOwnProperty(t)) {
                    var i = e[t];
                    "[object Date]" === Object.prototype.toString.call(i) && (e[t] = i.getTime() / 1e3)
                }
        },
        loadPlaylist: function() {},
        updateUiConfig: function() {}
    };
    return e
});
define("mp/classes/daxreporting", [], function() {
    function e(e) {
        var t = [i + r, "ns_type=hidden", "app_name=" + n, "app_type=" + a, "mediaplayer_name=BUMPV3", "mediaplayer_type=BUMPV3", "mediaplayer_version=" + window.embeddedMedia.version, s];
        e.mediaplayer_name && t.splice(4, 2);
        for (var o in e)
            e.hasOwnProperty(o) && t.push(encodeURIComponent(o) + "=" + encodeURIComponent(e[o]));
        return t.join("&")
    }
    function t(t) {
        if (o) {
            var i = e(t), n = new Image;
            n.src = i;
            return i
        }
    }
    var i = "//sa.bbc.co.uk/bbc/bbc/s?", n = "BUMPV3", a = "BUMPV3", o=!0, r = "name=unknown", s = "dax_session_id=unknown", l = {
        32: "spacebar",
        13: "enter",
        27: "esc"
    };
    return {
        init: function(e, t, l, d, c, p) {
            o = e;
            c && (n = c);
            p && (a = p);
            r = "name=" + encodeURIComponent(t);
            s = "dax_session_id=" + encodeURIComponent(l);
            i = d ? d : i
        },
        trackAction: function(e, i, n) {
            var a = {};
            if (!isNaN(i) && void 0 !== l[i]) {
                a.key = l[i];
                i = "key_press"
            }
            a.control_id = e;
            a.action_type = i;
            for (var o in n)
                n.hasOwnProperty(o) && (a[o] = n[o]);
            return t(a)
        }
    }
});
define("mp/classes/chromecastprovider", ["mp/classes/daxreporting"], function(e) {
    function t() {}
    var i = function(e) {
        this.currentSession = null;
        this.mediaSession = null;
        this.smp = e;
        this.isCasting=!1;
        this.isAvailable=!1;
        this.ccId = "";
        this.cancelRequest=!1;
        this.interrupted=!1;
        var t = document.createElement("script");
        t.type = "text/javascript";
        t.charset = "utf-8";
        t.async=!0;
        t.src = "https://www.gstatic.com/cv/js/sender/v1/cast_sender.js";
        document.getElementsByTagName("head")[0].appendChild(t);
        n.call(this)
    }, n = function() {
        var t = function() {
            this.dispatchToPlayer("onCastInitialised")
        }, i = function(e) {
            this.errorState(e)
        }, a = function(e) {
            this.currentSession = e;
            this.isAvailable=!0;
            this.dispatchToPlayer("ccInfo", {
                detail: "Receivers available"
            })
        }, o = function(t) {
            e.trackAction("ccAvailable", t);
            if ("available" == t) {
                this.isAvailable=!0;
                this.dispatchToPlayer("ccInfo", {
                    detail: "Receivers available"
                })
            } else {
                this.isAvailable=!1;
                this.isCasting=!1;
                this.dispatchToPlayer("ccInfo", {
                    detail: "Receivers not available"
                })
            }
        };
        if (chrome.cast && chrome.cast.isAvailable) {
            var r = "5E81F6DB";
            "sport" == this.smp._settings.product && (r = "F4ABD101");
            var s = new chrome.cast.SessionRequest(r), l = new chrome.cast.ApiConfig(s, a.bind(this), o.bind(this));
            chrome.cast.initialize(l, t.bind(this), i.bind(this))
        } else 
            setTimeout(n.bind(this), 1e3)
    };
    i.prototype = {
        requestCast: function(e) {
            if (this.currentSession)
                this.launchProg(e);
            else {
                var t = function(t) {
                    this.currentSession = t;
                    this.launchProg(e)
                }, i = function(e) {
                    this.errorState(e)
                };
                chrome.cast.requestSession(t.bind(this), i.bind(this))
            }
        },
        launchProg: function(e) {
            this.currentSession.addUpdateListener(this.onSessionUpdate.bind(this));
            this.ccId = this.currentSession.receiver.friendlyName;
            this.dispatchToPlayer("ccInfo", {
                state: "CONNECTING",
                id: this.ccId
            });
            for (var t = this.smp.playlist.holdingImageURL, i = this.smp.playlist.links || [], n = 0; i.length > n; n++)
                "holdingrecipe" === i[n].rel && (t = i[n].href);
            this.loadMedia(e, this.smp.playlist.title, this.smp.playlist.summary, t)
        },
        loadMedia: function(e, t, i, n) {
            if (null == this.currentSession)
                this.errorState("There is no active media playback");
            else {
                var a, o = function(e) {
                    this.interrupted=!1;
                    this.mediaSession = e;
                    this.mediaSession.addUpdateListener(this.onMediaStatusUpdate.bind(this))
                }, r = function(e) {
                    this.errorState(e)
                }, s = new chrome.cast.media.GenericMediaMetadata;
                n && (a = new chrome.cast.Image( - 1 === n.indexOf("{recipe}")&&-1 === n.indexOf("$recipe") && this.smp._settings.castPlaceholderImage ? this.smp._settings.castPlaceholderImage : n.replace("{recipe}", "1280x720").replace("$recipe", "1280x720")));
                a.height = 720;
                a.width = 1280;
                s.images = [a];
                s.title = t;
                for (var l = n.split("/"), d = l[l.length - 1].split(".")[0], c = "", p = this.smp.playlist.items, u = 0; p.length > u; u++)
                    if ("programme" === p[u].kind) {
                        c = p[u].group;
                        break
                    }
                var m = new chrome.cast.media.MediaInfo(e, "video/mp4");
                m.metadata = s;
                m.customData = {
                    base_image_url: "http://ichef.bbci.co.uk/programmeimages/" + d + "/",
                    secondary_title: i,
                    original_pid: c,
                    sender: "smp",
                    image_identifier: d
                };
                var f = new chrome.cast.media.LoadRequest(m);
                f.currentTime = this.smp.currentTime();
                f.currentTime >= this.smp.duration() - 5 && (f.currentTime = 0);
                this.currentSession.loadMedia(f, o.bind(this), r.bind(this))
            }
        },
        stopCast: function() {
            this.cancelRequest=!0;
            var e = function() {
                this.isCasting=!1;
                this.dispatchToPlayer("ccInfo", {
                    state: "STOPPED"
                })
            }, t = function(e) {
                this.isCasting=!1;
                this.errorState(e)
            };
            this.currentSession.stop(e.bind(this), t.bind(this))
        },
        stopMedia: function() {
            var e = function() {
                this.isCasting=!1;
                this.dispatchToPlayer("ccInfo", {
                    state: "STOPPED"
                })
            }, t = function(e) {
                this.errorState(e)
            };
            this.mediaSession.stop(new chrome.cast.media.StopRequest, e.bind(this), t.bind(this))
        },
        interrupt: function() {
            this.isCasting=!1;
            this.interrupted=!0;
            this.dispatchToPlayer("ccInfo", {
                state: "INTERRUPT"
            });
            this.stopMedia()
        },
        pause: function() {
            if (null == this.mediaSession)
                this.errorState("There is no active media playback");
            else {
                var e = function() {}, t = function() {
                    this.errorState("Unable to pause")
                };
                this.mediaSession.pause(new chrome.cast.media.PauseRequest, e.bind(this), t.bind(this))
            }
        },
        seekTo: function(e) {
            if (null == this.mediaSession)
                this.errorState("There is no active media playback");
            else {
                var i = function() {
                    this.errorState("Unable to seek")
                }, n = new chrome.cast.media.SeekRequest;
                n.currentTime = e;
                this.mediaSession.seek(n, t.bind(this), i.bind(this))
            }
        },
        play: function() {
            if (null == this.mediaSession)
                this.errorState("There is no active media playback");
            else {
                var e = function() {
                    this.errorState("Unable to play")
                };
                this.mediaSession.play(new chrome.cast.media.PlayRequest, t.bind(this), e.bind(this))
            }
        },
        setVolume: function(e) {
            if (null == this.mediaSession)
                this.errorState("There is no active media playback");
            else {
                var i = function() {
                    this.errorState("Unable to set volume")
                };
                0 === e ? this.currentSession.setReceiverMuted(!0, t.bind(this), i.bind(this)) : this.currentSession.setReceiverVolumeLevel(e, t.bind(this), i.bind(this))
            }
        },
        subtitlesOn: function() {
            var e = function() {
                this.errorState("Failed to turn subtitles on")
            };
            this.currentSession.sendMessage("urn:x-cast:uk.co.bbc.cast", "subtitlesOn", t.bind(this), e.bind(this))
        },
        subtitlesOff: function() {
            var e = function() {
                this.errorState("Failed to turn subtitles off")
            };
            this.currentSession.sendMessage("urn:x-cast:uk.co.bbc.cast", "subtitlesOff", t.bind(this), e.bind(this))
        },
        onMediaStatusUpdate: function(e) {
            if (e) {
                if (this.mediaSession&&!this.interrupted) {
                    this.isCasting=!0;
                    this.dispatchToPlayer("ccInfo", {
                        state: this.mediaSession.playerState,
                        currentTime: this.mediaSession.currentTime,
                        id: this.ccId,
                        subtitlesEnabled: this.mediaSession.customData.subtitlesEnabled,
                        floatVolume: this.currentSession.receiver.volume.level
                    })
                }
            } else if ("IDLE" == this.mediaSession.playerState&&!this.interrupted) {
                this.isCasting=!1;
                this.stopMedia()
            }
        },
        onSessionUpdate: function(e) {
            if (e) {
                if (this.currentSession&&!this.cancelRequest) {
                    var t = this.currentSession.receiver.volume.level;
                    this.currentSession.receiver.volume.muted && (t = 0);
                    this.mediaSession ? this.dispatchToPlayer("ccInfo", {
                        state: this.mediaSession.playerState,
                        currentTime: this.mediaSession.currentTime,
                        id: this.ccId,
                        subtitlesEnabled: this.mediaSession.customData.subtitlesEnabled,
                        floatVolume: t
                    }) : this.dispatchToPlayer("ccInfo", {
                        floatVolume: t
                    })
                }
            } else {
                this.stopCast();
                this.currentSession = null
            }
        },
        errorState: function(t) {
            e.trackAction("ccError", "error", {
                detail: t.description || t
            });
            this.dispatchToPlayer("ccInfo", {
                state: "ERROR",
                detail: t
            })
        },
        dispatchToPlayer: function(e, t) {
            if (t) {
                t.available = this.isAvailable;
                t.casting = this.isCasting
            }
            this.smp.call(e, t)
        }
    };
    return i
});
define("mp/classes/players/flash/smp-flash", ["jquery-1.9", "swfobject-2", "mp/modules/device", "mp/classes/settingsbuilder", "mp/classes/embedPopup", "mp/classes/players/player", "mp/classes/errors/error", "mp/classes/chromecastprovider"], function(e, t, i, n, a, o, r, s) {
    function l(e) {
        return e.getUTCFullYear() + "/" + d(e.getUTCMonth() + 1) + "/" + d(e.getUTCDate()) + " 00:00:00 UTC"
    }
    function d(e) {
        return 10 > e ? "0" + e : e
    }
    function c(e, t) {
        return function() {
            window.embeddedMedia.handle(e, t)
        }
    }
    function p() {
        var t = window.embeddedMedia, i = 0;
        return function(n, o) {
            var s = t.playerInstances[n];
            if (s && n == s._container.attr("id"))
                if (s._swf) {
                    var l = e.Event(o.type), d = o.data;
                    d && d.callType && (d = d.data);
                    if ("object" == typeof d) {
                        for (var p in d)
                            if ("type" != p) {
                                l[p] = d[p];
                                "currentDuration" == p && (l.duration = d[p])
                            }
                    } else 
                        l.eventData = d;
                        switch (o.type) {
                        case"ccRequest":
                            s.castProvider.requestCast(o.data.mediatorIdentifer);
                            break;
                        case"ccPlay":
                            s.castProvider.play();
                            break;
                        case"ccPause":
                            s.castProvider.pause();
                            break;
                        case"ccSeek":
                            s.castProvider.seekTo(o.data.seekTime);
                            break;
                        case"ccVolume":
                            s.castProvider.setVolume(o.data.floatVolume);
                            break;
                        case"ccCancel":
                            s.castProvider.stopMedia();
                            break;
                        case"ccInterrupt":
                            s.castProvider.interrupt();
                            break;
                        case"ccSubsOn":
                            s.castProvider.subtitlesOn();
                            break;
                        case"ccSubsOff":
                            s.castProvider.subtitlesOff();
                            break;
                        case"initialised":
                            if (20 > s._initAttempts) {
                                s._initAttempts++;
                                try {
                                    s.initialised()
                                } catch (u) {
                                    setTimeout(c(n, o), 100);
                                    return 
                                }
                            } else {
                                s.call("showError", window.embeddedMedia.language.t("6004", s._settings.locale()));
                                s._settings.dispatch(6004);
                                s._initAttempts = 0
                            }
                            break;
                        case"loadedmetadata":
                            s._duration = d.duration;
                            break;
                        case"canplay":
                            s._currentTime = d.resumeTime;
                            break;
                        case"timeupdate":
                            s._currentTime = d.currentTime;
                            s._duration = l.duration;
                            if (Math.abs(s._currentTime - s._lastSignificant) > s._settings.significantTime) {
                                var m = e.Event("significanttimeupdate");
                                for (var p in d)
                                    "type" != p && (m[p] = d[p]);
                                    s._lastSignificant = s._currentTime;
                                    m.duration = s._duration;
                                    s._container.trigger(m)
                                }
                                break;
                            case"seeked":
                                s._seeking=!1;
                                break;
                            case"seeking":
                                s._seeking=!0;
                                break;
                            case"playlistEnded":
                            case"ended":
                                s._playing=!1;
                                s._ended=!0;
                                break;
                            case"volumechange":
                                l.volume = d;
                                s._volume = d;
                                s._muted = 0 === d;
                                s._settings.set("volume", d);
                                break;
                            case"pause":
                                s._paused=!0;
                                break;
                            case"playlistLoaded":
                                s._seeking=!1;
                                s._paused=!0;
                                s._ended=!1;
                                s._playing=!1;
                                l.url = d.url;
                                s.playlist = d.playlist;
                                break;
                            case"playing":
                                s._playing=!0;
                                s._paused=!1;
                                s._ended=!1;
                                s._kind = l.kind;
                                break;
                            case"guidanceshow":
                                if (s._settings.guidanceCleared)
                                    s.call("updateGuidance", s._settings.guidance);
                                else {
                                    s._restoreFocus=!0;
                                    s._gCalled=!0;
                                    s._lockClicked = o.data.lockClicked;
                                    t.parentalGuidance.show(o.data, s._settings.ui.colour, s._settings.locale())
                                }
                                break;
                            case"linkRequest":
                                "managePIN" === d.type || "PGinfo" === d.type ? window.open("http://www.bbc.co.uk/guidance/help", "pg") : "playlistRedirect" == d.type && window.open(d.url, "_top");
                                break;
                            case"verify":
                                t.parentalGuidance.verify(d, function(e) {
                                    s.call("verified", e)
                                });
                                break;
                            case"setPIN":
                                t.parentalGuidance.setPIN(d.PIN);
                                break;
                            case"setGuidanceSkipped":
                                t.parentalGuidance.setAgeConfirmation(d.skipped);
                                break;
                            case"guidanceLocked":
                                t.parentalGuidance.setLocked(d.locked, !0);
                                break;
                            case"error":
                                o.data && (l.errorData = new r(o.data.code, s._settings, o.data));
                                break;
                            case"screenReaderAlert":
                                var f = d;
                                if ("string" != typeof f)
                                    return !1;
                                    var h = e("<div role='alert' style='width: 1px; position: absolute; left: -2500px; overflow: hidden;'><p>" + f + "</p></div>").appendTo(e("body"));
                                    h.hide();
                                    h.show();
                                    window.setTimeout(function() {
                                        h.remove()
                                    }, 2500);
                                    break;
                                case"focusOnFlash":
                                case"focusAfterFlash":
                                case"focusBeforeFlash":
                                    s[o.type]();
                                    break;
                                case"embed":
                                    if (s.playlist) {
                                        s._restoreFocus=!0;
                                        a.show(s)
                                    }
                                }
                                s._container.trigger(l)
                        } else 
                            i++<20 ? setTimeout(c(n, o), 100) : s._settings.dispatch(6e3)
        }
    }
    function u() {
        this.type = "flash";
        this._name = "smp-flash";
        this._volume = 0;
        this._currentTime = 0;
        this._lastSignificant = 0;
        this._duration = 0;
        this._kind = "";
        this._pluginsToLoad = [];
        this._initAttempts = 0;
        this.includePoster=!1;
        this.castProvider = null;
        return this
    }
    var m = u.prototype = new o;
    m.compatibility = function(e) {
        return t.hasFlashPlayerVersion(i.isMobileOrTablet ? "10.2" : "10.0") && (!e.backgroundPlayer || i.get().useOpaque)
    };
    m.embed = function(t, n) {
        var a = this;
        a._settings = t;
        a._container = t.container;
        for (var o = 0; t.plugins.toLoad.length > o; o++) {
            var r = t.plugins.toLoad[o];
            this.loadPlugin(r, r.data)
        }
        this.onLoad = n;
        var s = this._container.attr("id"), d = "//emp.bbci.co.uk/emp/SMPf/1.10.2/StandardMediaPlayerChromelessFlash.swf";
        "alternate" == t.playerType && (d = "//emp.bbci.co.uk/emp/SMPf/1.10.2/StandardMediaPlayerChromelessFlash.swf");
        var c = this.name() + "SWF" + s, p = {
            domId: s,
            jsCallbackMethod: "window.embeddedMedia.handle",
            forceGuidance: t.forceGuidance,
            today: l(t.today),
            isBSTime: t.isBSTime,
            waitOnPluginLoad: t.waitOnPluginLoad,
            uxHighlightColour: t.ui.colour,
            uxHighlightForeColour: t.ui.foreColour,
            customRdotBaseURL: t.customRdotBaseURL,
            enableRdotReporting: t.enableRdotReporting && "https:" != location.protocol,
            noTracking: t.noTracking,
            isTouchScreen: i.get().hasTouchScreen,
            embedReferer: document.referrer || "",
            embedPageUrl: t.embedPageURL,
            customDAXURL: t.customDAXURL,
            enableDaxReporting: t.enableDaxReporting,
            enableSonarReporting: t.enableSonarReporting,
            bumpVersion: window.embeddedMedia.version,
            comscoreURL: "//sa.bbc.co.uk/bbc/nkdata/s",
            daxSessionID: t.daxSessionID,
            appName: t.appName,
            appType: t.appType
        };
        if (t.ui.locale) {
            t.ui.locale.lang && (p.lang = t.ui.locale.lang);
            t.ui.locale.path && (p.localePath = t.ui.locale.path)
        }
        var u = "100%", m = "100%", f = "10.2", h = {
            quality: "high",
            bgcolor: "000000",
            allowFullScreen: !0,
            allowScriptAccess: "always"
        };
        i.get().useOpaque && (h.wmode = "opaque");
        var g = {}, y = {
            position: "relative",
            "z-index": "1",
            height: "100%",
            width: "100%",
            "padding-bottom": 0
        };
        t.superResponsive && (y["padding-bottom"] = "56.25%");
        var b, v, w, k, _;
        b = e("<div />").css(y);
        if (!t.backgroundPlayer) {
            var x = {
                "z-index": 1e6,
                position: "fixed",
                top: "-9999px",
                left: "100px",
                "background-color": "#ffffff",
                color: "#000000",
                "border-style": "solid",
                "border-width": "1px",
                padding: "8px 8px"
            }, P = "font-weight:bold; font-size : 0.9em; font-family: arial; color:black", T = t.locale();
            _ = e('<div id="' + s + 'jumpMediaPlayer"><a style="' + P + '" id="' + s + 'jumpMediaPlayerLink" href="">' + window.embeddedMedia.language.t("jumpMediaPlayer", T) + "</a></div>").css(x).appendTo(b);
            w = e('<div id="' + s + 'beforeFlash"><a style="' + P + '" id="' + s + 'beforeFlashLink" href="http://www.bbc.co.uk/faqs/online/mp_accessibility_help">' + window.embeddedMedia.language.t("mediaPlayerHelp", T) + "</a></div>").css(x).appendTo(b);
            var C = function() {
                e(this).css({
                    top: "100px"
                })
            }, S = function() {
                e(this).css({
                    top: "-9999px"
                })
            };
            _.focusin(C);
            _.focusout(S);
            w.focusin(C);
            w.focusout(S);
            _.keydown(function(e) {
                var t = e.keyCode || e.which;
                if (13 === t || 32 === t) {
                    e.preventDefault && e.preventDefault();
                    a.focusAfterFlash()
                }
            });
            w.keydown(function(e) {
                var t = e.keyCode || e.which;
                if (9 === t&&!e.shiftKey) {
                    e.preventDefault && e.preventDefault();
                    e.returnValue=!1;
                    setTimeout(function() {
                        if (a._swf) {
                            w.blur();
                            a._swf.focusOnFirst();
                            a._swf.tabIndex = 0;
                            a._swf.focus()
                        }
                    }, 200)
                }
            });
            a.focusBeforeFlash = function() {
                var e = document.getElementById(s + "beforeFlashLink");
                e && setTimeout(function() {
                    e.focus()
                }, 10)
            }
        }
        v = e("<div/>").css({
            position: "absolute",
            left: "0px",
            top: "0px",
            right: "0px",
            bottom: "0px"
        }).appendTo(b);
        e("<div/>").attr({
            id: c
        }).appendTo(v);
        if (!t.backgroundPlayer) {
            k = e('<div id="' + s + 'afterFlash" style="z-index: 1000000;position:fixed;top: 10px;left : -999px;background-color: #ffffff;color: #000000;border-style:solid;border-width:1px;padding: 8px 8px"><a id="' + s + 'returnToMediaPlayerLink" href="#afterFlash">' + window.embeddedMedia.language.t("outOfMediaPlayer", T) + "</a></div>").appendTo(b);
            k.keydown(function(e) {
                var t = e.keyCode || e.which;
                if (13 === t || 32 === t || 9 === t && e.shiftKey) {
                    e.preventDefault && e.preventDefault();
                    e.returnValue=!1;
                    setTimeout(function() {
                        if (a._swf) {
                            9 === t ? a._swf.focusOnLast() : a._swf.focusOnFirst();
                            a._swf.tabIndex = 0;
                            a._swf.focus()
                        }
                    }, 200)
                }
            })
        }
        a.focusAfterFlash = function() {
            var e = document.getElementById(s + "returnToMediaPlayerLink");
            e && setTimeout(function() {
                e.focus()
            }, 10)
        };
        a.focusOnFlash = function() {
            setTimeout(function() {
                if (a._swf) {
                    a._swf.tabIndex = 0;
                    a._swf.focus()
                }
            }, 10)
        };
        a.wrapper = b;
        a._container.empty().append(b);
        a.attachListeners();
        a.embeddingProperties = [d, c, u, m, f, !1, p, h, g, function(e) {
            if (e && e.success && e.ref) {
                a._swf = e.ref;
                a.onLoad(a, !0)
            } else 
                a._settings.dispatch(6e3)
        }
        ];
        a.embedIt()
    };
    m.embedIt = function() {
        t.embedSWF.apply(t, this.embeddingProperties)
    };
    m.attachListeners = function() {
        window.embeddedMedia.handle = p(this._settings.domid, this)
    };
    m.hide = function(e) {
        e ? this._swf.style.visibility = "hidden" : this.wrapper[0].style.left = "-9000px"
    };
    m.show = function(e) {
        this._swf.style.visibility = "visible";
        this.wrapper[0].style.left = "0px";
        e && this.play()
    };
    m.initialised = function() {
        var e = this, t = i.get();
        e.uiDefaults(t.isMobileOrTablet, t.os && t.os.blackberry);
        e.call("setUIOptions", e._settings.ui);
        e._volume = e._settings.volume;
        e.call("volume", e._volume);
        e._muted = e._settings.muted;
        e.call("muted", e._muted);
        for (var n = 0; e._pluginsToLoad.length > n; n++) {
            var a = e._pluginsToLoad[n];
            e.setData(a.data);
            a.swf && e.loadPluginWithCheck(a.swf)
        }
        e.ready=!0;
        e.call("updateGuidance", e._settings.guidance);
        e.loadPlaylist();
        var o = window.embeddedMedia, r=!1;
        o.openCallbacks.push(function() {
            e.hide();
            if (e._playing&&!e.paused()) {
                r=!0;
                e.pause()
            }
        });
        o.closedCallbacks.push(function() {
            e.show(r);
            r=!1;
            if (e._restoreFocus) {
                e._restoreFocus=!1;
                e.focusOnFlash();
                e._swf.focusOnFirst()
            }
        });
        var l = o.parentalGuidance;
        l.closedCallbacks.push(function(t) {
            e.call("guidanceClosed", t);
            if (e._gCalled) {
                e._gCalled=!1;
                if (t&&!e._lockClicked) {
                    var i = e._settings.startTime;
                    isNaN(i) && (i = 0);
                    e.currentTime(i)
                }
            }
        });
        l.updateCallbacks.push(function(t) {
            var i = e._settings;
            t.noPin = i.guidanceCleared;
            i.guidance = t;
            e.call("updateGuidance", t)
        });
        window.chrome && e._settings.allowCasting && (e.castProvider = new s(e))
    };
    m.loadPlaylist = function() {
        var e = this;
        if (e._swf) {
            var t = {}, a = e._settings;
            e._volume = a.volume;
            var o = n.mediaSet(a);
            t.playlistUrl = a.playlist;
            t.embedPageURL = a.embedPageURL;
            t.customIStatsURL = a.customIStatsURL;
            t.noTracking = a.noTracking;
            t.counterName = a.counterName;
            t.appName = a.appName;
            t.appType = a.appType;
            t.quality = a.quality;
            t.loop = a.loop;
            t.siteID = a.siteID;
            t.overrideHoldingImage = a.overrideHoldingImage;
            t.customDAXURL = a.customDAXURL;
            t.enableDaxReporting = a.enableDaxReporting;
            t.enableSonarReporting = a.enableSonarReporting;
            t.comscoreURL = "//sa.bbc.co.uk/bbc/nkdata/s";
            t.bumpVersion = window.embeddedMedia.version;
            if (a.statsObject&&!a.statsObject.deviceId) {
                var r = (document.cookie + "").match(/\bs1=.+?\b/gim);
                r && (a.statsObject.deviceId = r[0].replace("s1=", ""))
            }
            t.statsInfo = a.statsObject;
            t.daxSessionID = a.daxSessionID;
            t.suppressItemKind = a.suppressItemKind;
            t.startTime = a.startTime;
            t.sessionID = a.sessionID;
            t.unreliableTimecodes = a.unreliableTimecodes;
            t.pageType = a.pageType;
            t.parentSection = a.parentSection;
            t.edition = a.edition;
            t.uniqueId = a.uniqueID;
            a.sessionID = "";
            "number" == typeof a.bitrateFloor && (t.bitrateFloor = a.bitrateFloor);
            "number" == typeof a.bitrateCeiling && (t.bitrateCeiling = a.bitrateCeiling);
            "number" == typeof a.limitQuality ? t.limitQuality = a.limitQuality : a.lowBandwidth && (t.limitQuality = 1);
            t.autoPlay = a.autoplay;
            t.promptGuidanceWhenAutoplay = a.promptGuidanceWhenAutoplay;
            t.enableStageVideo = a.disableStageVideo ||!(i.os && "xp" == i.os.win && a.superResponsive);
            t.mediationUrl = n.mediatorHref(a, o);
            a.mediator.saml && (t.samlMediationUrl = n.mediatorHref(a, o, "https://{saml}/saml/mediaselector/5/select/version/2.0/mediaset/{mediaset}/vpid/{id}"));
            a.mediator.fck && (t.forceConnectionSupplier = a.mediator.fck);
            a.playlistObject ? e.call("setPlaylistObject", {
                playlist: a.playlistObject,
                options: t
            }) : e.call("setPlaylistOptions", t);
            var s = a.webcastData;
            s && (s.accurateStartTime || s.scheduledStartTime || s.accurateEndTime || s.scheduledEndTime) && e.webcastData(a.webcastData)
        }
    };
    m.call = function(e, t) {
        if (this._swf.call) {
            this.hasCalledSwf=!0;
            this._swf.call(e, t)
        } else 
            !this.hasCalledSwf || "setPlaylistOptions" != e && "setPlaylistObject" != e || this.embedIt()
    };
    m.loadPlugin = function(e, t) {
        if (e && e.swf)
            if (this.ready) {
                this.setData(t);
                this.loadPluginWithCheck(e.swf)
            } else 
                this._pluginsToLoad.push({
                    swf: e.swf,
                    data: t
                })
    };
    m.loadPluginWithCheck = function(e) {
        for (var t = this._settings.blacklistedPlugins, i = 0; t.length > i; i++)
            if (e.indexOf(t[i])>-1)
                return;
        this.call("loadPlugin", e)
    };
    m.updateUiConfig = function(e) {
        this.call("setUIOptions", e)
    };
    m.setData = function(e) {
        if (e)
            if (e instanceof Array)
                for (var t = 0; e.length > t; t++)
                    this.call("setData", e[t]);
            else 
                this.call("setData", e)
    };
    m.setPoster = function(e) {
        e && this.call("setPoster", e)
    };
    m.dispatchEvent = function(e, t) {
        e && this.call(e, t)
    };
    m.pauseAt = function(e) {
        if (e) {
            e = e instanceof Array ? e : [e];
            this.call("pauseAt", e)
        }
    };
    m.webcastData = function(e) {
        if (e) {
            o.prototype.webcastData.call(this, e);
            this.call("webcastData", e)
        }
    };
    m.suspend = function() {
        this.call("stop")
    };
    return u
});
define("mp/classes/players/app", ["jquery-1.9", "mp/classes/players/player"], function(e, t) {
    function i() {
        var t = this, i = t._settings;
        if (!t.hasPlaylist())
            return !0;
        var a = i.ui.colour + "", o = t.css;
        i.superResponsive && (o["padding-bottom"] = "56.25%");
        if (!t.wrapper) {
            t.wrapper = e("<div />").css(o);
            t.wrapperPopup = e("<div />").css(o)
        }
        if (!i.backgroundPlayer) {
            var r = "iplayer" === i.product ? 'url("//emp.bbci.co.uk/emp/assets/images/cta_play.png")': 'url("//emp.bbci.co.uk/emp/assets/images/cta_play_journalism.png")';
            i.ui.cta && void 0 !== i.ui.cta.custom && (r = 'url("' + i.ui.cta.custom + '")');
            var s = window.embeddedMedia.language.t("aaampPlay", i.locale()), l = e('<a id="cta"/>').attr({
                title: s
            }), d = i.ui[t.classStr] || i.ui.CTAcss;
            t.incGuidance=!0;
            if (d) {
                t.incGuidance=!1;
                l[0].className = d
            } else {
                l.css({
                    backgroundImage: r,
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                    backgroundColor: "#000000",
                    opacity: .75,
                    margin: 0,
                    width: "80px",
                    height: "80px",
                    display: "block",
                    position: "absolute",
                    bottom: "-0.5px"
                });
                t._container.on("mousedown touchstart mouseover", function() {
                    e(l).css({
                        backgroundColor: a,
                        opacity: 1
                    })
                });
                t._container.on("touchmove mouseup touchend mouseout", function() {
                    e(l).css({
                        backgroundColor: "#000000",
                        opacity: .75
                    })
                })
            }
            if (i.includeAppClickHandler) {
                var c, p=!1, u = function() {
                    p=!1
                }, s = function() {
                    if ("" === t.wrapper[0].style.display&&!p) {
                        t.dax && setTimeout(function() {
                            var n = e.Event("userPlay");
                            i.trigger(n);
                            t.dax.trackAction("play", "click", {
                                mediaplayer_name: t.daxname,
                                mediaplayer_type: "video"
                            })
                        }, 50);
                        t.play();
                        p=!0;
                        clearTimeout(c);
                        c = setTimeout(u, 100)
                    }
                };
                e(l).attr("tabIndex", 0).on("keyup", function(e) {
                    (32 == e.keyCode || 13 == e.keyCode) && s()
                });
                t._container.on("click", s)
            }
            t.wrapper.append(l);
            t._container.empty().append(t.wrapper);
            if (t.incGuidance) {
                var m = i.playlistObject && i.playlistObject.guidance || i.forceGuidance || i[t._guidance];
                if (m) {
                    t.guidanceWrapper = n(m);
                    t.wrapper.append(t.guidanceWrapper.div)
                }
            }
            return !1
        }
        t._container.empty().append(t.wrapper)
    }
    function n(e) {
        var t = document.createElement("div");
        t.id = "p_guidance";
        var i = document.createElement("div");
        i.id = "p_guidanceBox";
        var n = document.createElement("span");
        n.id = "redCirc";
        var a = document.createElement("span");
        a.id = "redCirc_c";
        n.appendChild(a);
        a.textContent = "G";
        a.setAttribute("aria-labelledby", "p_guidanceText");
        i.appendChild(n);
        var o = document.createElement("span");
        o.id = "p_guidanceText";
        o.textContent = e;
        i.appendChild(o);
        t.appendChild(i);
        var r = document.createElement("style"), s = document.getElementsByTagName("head")[0];
        s.appendChild(r);
        var l = r.sheet ? r.sheet: r.styleSheet;
        l.insertRule("#p_guidance {position:absolute;left: 81px;height:80px;bottom:0px;background-color: rgba(0, 0, 0, 0.75);margin-top: -40px;display:table; padding-right:30px;  font-family:helvetica, Arial, sans-serif;line-height:1}", 0);
        l.insertRule("#p_guidanceBox {color:#ffffff;font-size:14px;font-weight:bold;padding-left:5px;display:table-cell;vertical-align:middle;}", 1);
        l.insertRule("#redCirc_c {position:relative;left:3px;top:1px;line-height:normal}", 2);
        l.insertRule("#p_guidanceText {margin-left:5px;}", 3);
        l.insertRule("#redCirc {display:inline-block;margin:auto;border-radius:50%;height:18px;width:18px;background-color:red;}", 4);
        return {
            text: o,
            div: t
        }
    }
    function a(e) {
        this._name = e;
        return this
    }
    var o = {
        position: "relative",
        "z-index": 1,
        height: "100%",
        width: "100%",
        cursor: "pointer",
        "padding-bottom": 0
    }, r = a.prototype = new t;
    r.createCta = i;
    r.css = o;
    r.embed = function(e, t) {
        var i = this;
        i._settings = e;
        i._container = e.container;
        i.ctaNeedsCreating = i.createCta();
        i.extraEmbed && i.extraEmbed();
        t(i);
        i._container.trigger("initialised")
    };
    r.show = function(e) {
        this.wrapper.children().show();
        e && this.play()
    };
    r.hide = function() {
        this.wrapper.children().hide()
    };
    r.hasPlaylist = function() {
        var e = this._settings;
        return e.playlist || e.playlistObject
    };
    r.loadPlaylist = function() {
        var e = this, t = e._settings;
        if (e.ctaNeedsCreating)
            setTimeout(function() {
                e.ctaNeedsCreating = e.createCta()
            }, 100);
        else if (e.incGuidance) {
            var i = t.playlistObject && t.playlistObject.guidance || t.forceGuidance || t[e._guidance];
            if (i)
                if (e.guidanceWrapper) {
                    e.guidanceWrapper.div.style.display = "";
                    e.guidanceWrapper.text.textContent = i
                } else {
                    e.guidanceWrapper = n(i);
                    e.wrapper.append(e.guidanceWrapper.div)
                } else if (e.guidanceWrapper) {
                e.guidanceWrapper.div.style.display = "none";
                e.guidanceWrapper.text.textContent = ""
            }
        }
        t.autoplay && e.play();
        e.resetContainer && e.resetContainer()
    };
    return a
});
define("mp/classes/players/flash/smp-flash-noembed", ["mp/classes/players/app", "swfobject-2", "mp/modules/device"], function(e, t, i) {
    function n() {
        this._settings.delayEmbed=!1;
        this._settings.autoplay=!0;
        this._settings.promptGuidanceWhenAutoplay=!0;
        this._container.unbind("click");
        this._container.empty();
        this.reloadScope._loadOptions.call(this.reloadScope, this.reloadScope._loadOptionsArg, !0)
    }
    function a(e) {
        return !e.autoplay && e.delayEmbed===!0 && t.hasFlashPlayerVersion(i.isMobileOrTablet ? "10.2" : "10.0")
    }
    function o() {
        this.compatibility = a;
        this.daxname = "SMPFLASHNOEMBED";
        this._name = "smp-flash-noembed";
        return this
    }
    o.prototype = new e("smp-flash-noembed");
    o.prototype.play = n;
    return o
});
define("mp/classes/xd", [], function() {
    var e = function(e) {
        var t = this, i = document.getElementById(e);
        t._receiveKey = e;
        i && i.contentWindow && (t._postElement = i.contentWindow);
        t._listeners = {};
        var n = {};
        n[t._initXD] = {
            receiveKey: t._receiveKey
        };
        t.post(n);
        t.receive(function(e) {
            if (e.data) {
                var i = e.data.type, n = e.data.data, a = t._listeners[i];
                if (a)
                    for (var o = 0; a.length > o; o++) {
                        var r = a[o];
                        r.call(t, n)
                    }
                }
        })
    };
    e.prototype = {
        _initXD: "initXD",
        _receiveKey: "",
        _postElement: null,
        _listeners: {},
        post: function(e) {
            var t = this;
            e && t._postElement && t._postElement.postMessage && t._postElement.postMessage(JSON.stringify(e), "*");
            return t
        },
        receive: function(e) {
            var t = this, i = function(i) {
                if (0 === i.data.indexOf("{")) {
                    var n = {};
                    for (var a in i)
                        n[a] = "data" == a ? JSON.parse(i[a]) : i[a];
                    if (!n.data.receiveKey || n.data.receiveKey !== t._receiveKey)
                        return !1;
                    e(n)
                }
            };
            window.addEventListener ? window.addEventListener("message", i, !1) : window.attachEvent("onmessage", i);
            return t
        },
        on: function(e, t) {
            var i = this, n = function() {
                setTimeout(function(e) {
                    t.apply(i, e)
                }, 0, arguments)
            };
            this._listeners[e] || (this._listeners[e] = []);
            this._listeners[e].push(n);
            return i
        },
        fire: function(e, t) {
            this.post({
                receiveKey: this._receiveKey,
                type: e,
                data: t
            })
        }
    };
    return e
});
define("mp/classes/players/html5/smp-html5", ["jquery-1.9", "mp/modules/device", "mp/modules/html5", "mp/classes/xd", "mp/classes/errors/error", "mp/classes/settingsbuilder", "mp/classes/embedPopup", "mp/classes/players/player"], function(e, t, i, n, a, o, r, s) {
    function l() {
        this._name = "smphtml5";
        this.type = "html";
        this._currentTime = 0;
        this._duration = 0;
        this._lastSignificant = 0;
        this._kind = "";
        this.pluginsToLoad = [];
        this.includePoster=!1;
        return this
    }
    function d(e) {
        var i = [];
        e.uiClass && i.push(e.uiClass);
        e.defaultUiClass && i.push(e.defaultUiClass);
        t.get().hasTouchScreen && i.push("touch");
        return i.join(" ")
    }
    function c(e) {
        var i = {
            embedReferer: document.referrer || "",
            embedPageUrl: e.embedPageUrl,
            playlist: e.playlist,
            playlistObject: e.playlistObject,
            isMobileNetwork: t.get().connection.mobile,
            uiClass: d(e),
            ui: e.ui,
            mediatorHref: o.mediatorHref(e, o.mediaSet(e, "html5"), "{protocol}://{host}/mediaselector/5/select/version/2.0/vpid/{id}/format/json/mediaset/{mediaset}/jsfunc/{callback}"),
            forceGuidance: e.forceGuidance,
            guidance: e.guidance,
            iStatsDomain: e.customIStatsURL,
            enableStatsReporting: e.enableStatsReporting,
            enableDaxReporting: e.enableDaxReporting,
            noTracking: e.noTracking,
            daxDomain: e.customDAXURL,
            counterName: e.counterName,
            appName: e.appName,
            appType: e.appType,
            daxSessionID: e.daxSessionID,
            siteID: e.siteID,
            blockRestricted: !e.attemptRestrictedPlayback,
            startTime: e.startTime,
            isBSTime: e.isBSTime,
            loop: e.loop,
            autoplay: e.autoplay,
            overrideHoldingImage: e.overrideHoldingImage,
            rDotBaseURL: e.customRdotBaseURL,
            enableRdotReporting: e.enableRdotReporting && "https:" != location.protocol,
            suppressType: e.suppressType,
            suppressItemKind: e.suppressItemKind,
            bitrateFloor: e.bitrateFloor,
            bitrateCeiling: e.bitrateCeiling,
            supportLiveRewind: e.supportLiveRewindOnMobile,
            promptGuidanceWhenAutoplay: e.promptGuidanceWhenAutoplay,
            enableConvivaReporting: !1,
            statsObject: e.statsObject,
            pageType: e.pageType,
            parentSection: e.parentSection,
            edition: e.edition,
            uniqueId: e.uniqueID
        };
        e.mediator.saml && (i.samlMediatorHref = o.mediatorHref(e, o.mediaSet(e, "html5"), "https://{saml}/saml/mediaselector/5/select/version/2.0/vpid/{id}/format/json/mediaset/{mediaset}/jsfunc/{callback}"));
        return i
    }
    function p(t) {
        return function(i) {
            var n = i.type, o = i.properties, s = e.Event(n);
            for (var l in i)
                "type" != l && "_" != l && (s[l] = i[l]);
            for (l in o)
                "type" != l && (s[l] = o[l]);
            switch (n) {
            case"initialised":
                t.initialised();
                break;
            case"loadedmetadata":
                t._duration = o.duration;
                break;
            case"timeupdate":
                t._currentTime = o.currentTime;
                t._duration = o.duration;
                if (Math.abs(t._currentTime - t._lastSignificant) > t._settings.significantTime) {
                    var d = e.Event("significanttimeupdate");
                    for (var l in o)
                        "type" != l && (d[l] = o[l]);
                    t._lastSignificant = t._currentTime;
                    t._container.trigger(d)
                }
                break;
            case"volumechange":
                t._muted = o.muted;
                t._volume = o.volume;
                break;
            case"guidanceShow":
                t._gCalled=!0;
                t._lockClicked = o.lockClicked;
                window.embeddedMedia.parentalGuidance.show(i.properties, t._settings.ui.colour, t._settings.locale());
                if (t._fullScreen) {
                    t.wrapper.css(u);
                    t.wrapper[0].style.WebkitTransform = ""
                }
                break;
            case"guidanceLocked":
                window.embeddedMedia.parentalGuidance.setLocked(i.data, !0);
                break;
            case"seeked":
                t._seeking=!1;
                break;
            case"seeking":
                t._seeking=!0;
                break;
            case"ended":
                t._ended=!0;
                break;
            case"pause":
                t._paused=!0;
                break;
            case"playing":
                t._paused=!1;
                t._ended=!1;
                t._kind = s.kind;
                break;
            case"linkRequest":
                var c = i.data;
                "managePIN" === c.type || "PGinfo" === c.type ? window.open("http://www.bbc.co.uk/guidance/help", "pg") : "playlistRedirect" == c.type && window.open(c.url, "_top");
                break;
            case"embed":
                t.playlist && r.show(t);
                break;
            case"playlistLoaded":
                t._seeking=!1;
                t._paused=!0;
                t._ended=!1;
                t.playlist = i.playlist;
                break;
            case"error":
                s.errorData = new a(i.code, t._settings)
            }
            t._container.trigger(s)
        }
    }
    var u = {
        "border-bottom": 0,
        "z-index": 999,
        position: "relative",
        height: "100%",
        width: "100%",
        "padding-bottom": 0
    }, m = l.prototype = new s;
    m.compatibility = function(e) {
        return !e.backgroundPlayer && (t.get().os.ios || e.preferHtmlOnMobile) && t.get().isMobileOrTablet && i.media.video && i.media.audio
    };
    m.embed = function(i, a) {
        var o = this;
        o._settings = i;
        o._container = i.container;
        o._container.empty();
        var r = t.get();
        o.uiDefaults(r.isMobileOrTablet);
        i.insideIframe && (i.ui.fullscreen || (i.ui.fullscreen = {
            enabled: !1,
            dblclick: !1
        }));
        for (var s = 0; i.plugins.toLoad.length > s; s++) {
            var l = i.plugins.toLoad[s];
            o.loadPlugin(l, l.data)
        }
        i.attemptHlsPlaybackOnAndroid && r.allowHlsVod();
        r.hasHlsVodSupport || i.suppressType.push("hls");
        var d = "//emp.bbc.co.uk/emp/SMPj/1.5.2/iframe.html";
        "alternate" == i.playerType && (d = "//emp.bbc.co.uk/emp/SMPj/1.5.2/iframe.html");
        document.cookie.match(/html5Debug=true/) && (d = "//emp.bbc.co.uk/emp/SMPj/1.5.2/debug.html");
        - 1 !== location.host.indexOf("bbc.co.uk") && (document.domain = "bbc.co.uk");
        if ( - 1 !== location.host.indexOf("bbc.com")) {
            d = d.replace("bbc.co.uk", "bbc.com");
            document.domain = "bbc.com"
        }
        o.uniqueId = "X" + Math.random();
        var c = o._container;
        i.superResponsive && (u["padding-bottom"] = "56.25%");
        var m = o.wrapper = e("<div />").css(u).appendTo(c), f = e("<iframe />").attr({
            id: o.getId(),
            name: o.getId(),
            frameborder: 0,
            scrolling: "no",
            src: d
        }).css({
            position: "absolute",
            left: "0px",
            top: "0px",
            width: "100%",
            height: "100%"
        }).appendTo(m).load(function() {
            var t, i, a, r, s = e(this).attr("id"), l = document.body.style, d = document.documentElement.style;
            o._fullScreen=!1;
            var c = function() {
                o._fullScreen=!0;
                var n = document.getElementById("orb-banner");
                n && (n.style.visibility = "hidden");
                t = l.overflow;
                l.overflow = "hidden";
                i = d.overflow;
                d.overflow = "hidden";
                e(document.body).bind("touchmove.playerFullscreen", function(e) {
                    e.preventDefault()
                });
                var s = o.resizePlayer = function() {
                    var e = window.innerWidth, t = window.innerHeight;
                    m.css({
                        position: "fixed",
                        top: "0px",
                        left: "0px",
                        "z-index": "2147483600",
                        height: t + "px",
                        width: e + "px",
                        "padding-bottom": 0
                    });
                    m[0].style.WebkitTransform = "translate3d(0,0,10em)"
                }, c = function() {
                    s();
                    clearTimeout(r);
                    r = setTimeout(s, 200)
                };
                s();
                e(window).bind("resize.playerFullscreen orientationchange.playerFullscreen", c);
                if (!a) {
                    a = document.createElement("meta");
                    a.name = "viewport";
                    a.content = "width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0";
                    document.getElementsByTagName("head")[0].appendChild(a)
                }
            };
            o._pfs = function() {
                o._fullScreen=!1;
                o.resizePlayer = null;
                clearTimeout(r);
                var n = document.getElementById("orb-banner");
                n && (n.style.visibility = "visible");
                e(window).unbind("resize.playerFullscreen orientationchange.playerFullscreen");
                m.css(u);
                m[0].style.WebkitTransform = "";
                l.overflow = t;
                d.overflow = i;
                e(document.body).unbind("touchmove.playerFullscreen");
                if (a && a.parentNode) {
                    a.parentNode.removeChild(a);
                    a = null
                }
            };
            o.xd = new n(s);
            o.xd.on("jsapi", p(o)).on("playerFullscreenEnter", c).on("playerFullscreenExit", o._pfs);
            "object" != typeof window.JSON ? require(["http://emp.bbci.co.uk/emp/assets/bump-3/json.js"], function() {
                o.fireBootstrap()
            }) : o.fireBootstrap()
        });
        window.addEventListener && window.addEventListener("pageshow", function(e) {
            if (e.persisted) {
                f.attr("src", function(e, t) {
                    return t
                });
                o._fullScreen && o._pfs()
            }
        });
        a(o, !0)
    };
    m.fireBootstrap = function() {
        var e = this;
        window.embeddedMedia.playerInstances[e.uniqueId] = e;
        var t = {
            build: "SMP-BUMP",
            player: {
                vars: c(e._settings)
            },
            id: e.uniqueId,
            requireMap: window.bbcRequireMap
        };
        try {
            var i = document.getElementById(e.getId());
            i.contentWindow.bootstrap(t)
        } catch (n) {
            e.xd.fire("bootstrap", t)
        }
    };
    m.updateUiConfig = function(e) {
        this.call("updateUiConfig", e)
    };
    m.loadPlaylist = function() {
        var e = this;
        if (e.ready) {
            var t = e._settings;
            t.playlistObject ? e.call("loadPlaylist", [t.playlistObject, t.autoplay, t.overrideHoldingImage, t.startTime, t.promptGuidanceWhenAutoplay]) : e.call("loadPlaylist", [t.playlist, t.autoplay, t.overrideHoldingImage, t.startTime, t.promptGuidanceWhenAutoplay]);
            var i = e._settings.webcastData;
            i && (i.accurateStartTime || i.scheduledStartTime || i.accurateEndTime || i.scheduledEndTime) && e.webcastData(e._settings.webcastData)
        }
    };
    m.getId = function() {
        return this._name + "iframe" + this._settings.domid
    };
    m.call = function(e, t) {
        try {
            var i = document.getElementById(this.getId());
            t instanceof Array ? i.contentWindow.player[e].apply(i.contentWindow.player, t) : i.contentWindow.player[e](t)
        } catch (n) {
            this.xd.fire("jsapi", {
                call: e,
                params: t
            })
        }
    };
    m.loadPlugin = function(e, t) {
        e && e.html && (this.ready ? this.call("loadPlugin", [e.html, t]) : this.pluginsToLoad.push({
            url: e.html,
            data: t
        }))
    };
    m.setData = function(e) {
        if (e)
            if (e instanceof Array)
                for (var t = 0; e.length > t; t++)
                    this.call("setData", e[t]);
            else 
                this.call("setData", e)
    };
    m.webcastData = function(e) {
        if (e) {
            s.prototype.webcastData.call(this, e);
            this.setData({
                name: "SMP.webcastData",
                data: e
            })
        }
    };
    m.setPoster = function(e) {
        this.ready && e && this.call("setPoster", e)
    };
    m.dispatchEvent = function(e, t) {
        e && this.call("dispatch", {
            type: e,
            data: t
        }, !0)
    };
    m.pauseAt = function(e) {
        if (e) {
            e = e instanceof Array ? e : [e];
            this.call("pauseAt", {
                times: e
            })
        }
    };
    m.show = function(e) {
        this.wrapper[0].firstChild.style.width = "100%";
        this._settings.tryPlayAfterInHtml && e && this.play()
    };
    m.hide = function() {
        this.wrapper[0].firstChild.style.width = "0%"
    };
    m.getHandleJSAPI = function() {
        return p(this)
    };
    m.suspend = function() {
        this.call("pause");
        this.call("suspend")
    };
    m.initialised = function() {
        var e = this;
        e.ready=!0;
        for (var t = 0; e.pluginsToLoad.length > t; t++) {
            var i = e.pluginsToLoad[t];
            e.setData(i.data);
            e.call("loadPlugin", [i.url, i.data])
        }
        window.embeddedMedia.parentalGuidance.closedCallbacks.push(function(t) {
            e.call("guidanceClosed", [t]);
            e._fullScreen && e.resizePlayer && e.resizePlayer();
            setTimeout(function() {
                e.resizePlayer && e.resizePlayer()
            }, 500);
            e._gCalled && t&&!e._lockClicked && e.play();
            e._gCalled=!1
        });
        window.embeddedMedia.parentalGuidance.updateCallbacks.push(function(t) {
            e._settings.guidance = t;
            e.call("updateGuidance", t)
        });
        var n = e._settings.webcastData;
        n && (n.accurateStartTime || n.scheduledStartTime || n.accurateEndTime || n.scheduledEndTime) && e.webcastData(e._settings.webcastData)
    };
    return l
});
define("mp/classes/players/apps/smp-air", ["jquery-1.9", "mp/modules/device", "mp/classes/settingsbuilder", "mp/classes/players/app"], function(e, t, i, n) {
    function a(e, t) {
        return null != t ? "string" == typeof t && 0 === t.length ? "" : "&" + encodeURIComponent(e) + "=" + encodeURIComponent(t) : ""
    }
    function o(n) {
        var o = "?";
        n.playlist && "" != n.playlist && (o = "?playlistUrl=" + encodeURIComponent(n.playlist));
        var r = {};
        r.product = n.product;
        r.mediaSet = i.mediaSet(n);
        r.embedPageURL = n.embedPageURL;
        r.counterName = n.counterName;
        r.siteID = n.siteID;
        r.overrideHoldingImage = n.overrideHoldingImage;
        r.daxSessionID = n.daxSessionID;
        r.sessionID = n.sessionID;
        r.unreliableTimecodes = n.unreliableTimecodes;
        r.appName = n.appName;
        r.appType = n.appType;
        r.quality = n.quality;
        r.parentPID = n.statsObject.parentPID;
        r.parentPIDType = n.statsObject.parentPIDType;
        n.startTime > 0 && (r.startTime = n.startTime);
        r.pageType = n.pageType;
        r.parentSection = n.parentSection;
        r.edition = n.edition;
        r.uniqueId = n.uniqueID;
        n.noTracking && (r.noTracking=!0);
        n.sessionID = "";
        "number" == typeof n.limitQuality ? r.limitQuality = n.limitQuality : n.lowBandwidth && (r.limitQuality = 1);
        r.mediatorHost = n.mediator && n.mediator.host ? n.mediator.host : "open.live.bbc.co.uk";
        r.mediasetClass = n.mediaProfile && n.mediaProfile.length > 0 ? "pc" === n.mediaProfile ? "default" : n.mediaProfile : "pc" === n.mediasetClass ? "default" : n.mediasetClass;
        r.mediator = i.mediatorHref(n, r.mediaSet);
        n.mediator.saml && (r.samlMediationUrl = "https://" + n.mediator.saml + "/saml/mediaselector/5/select/version/2.0/mediaset/{mediaset}/vpid/{id}");
        n.mediator.fck && (r.forceConnectionSupplier = n.mediator.fck);
        r.forceGuidance = n.forceGuidance;
        r.uxHighlightColour = n.ui.colour.toString().replace("#", "0x");
        r.uxFc = n.ui.foreColour.toString().replace("#", "0x");
        r.uxAc = n.ui.alternateColour.toString().replace("#", "0x");
        r.uxAfc = n.ui.alternateForeColour.toString().replace("#", "0x");
        n.ui.locale && n.ui.locale.lang && (r.lang = n.ui.locale.lang);
        if (n.plugins) {
            n.plugins.smpAirPlugins && (r.pluginSet = n.plugins.smpAirPlugins);
            n.plugins.smpAirPluginData && (r.pd = JSON.stringify(n.plugins.smpAirPluginData))
        }
        t.get().os.kindleSilk && n.kindlePolicy ? r.policy = n.kindlePolicy : n.policy && (r.policy = n.policy);
        if (n.webcastData) {
            n.webcastData.accurateStartTime && (r.AST = n.webcastData.accurateStartTime);
            n.webcastData.scheduledEndTime && (r.SET = n.webcastData.scheduledEndTime);
            n.webcastData.accurateStartTime && (r.AET = n.webcastData.accurateEndTime)
        }
        r.enableBitrateSwitching=!n.disableBitrateSwitching;
        r.debug = n.debug;
        r.allowMainlineProfiles = n.allowMainlineProfiles;
        r.enableLiveRewind = n.ui.controls && "undefined" != typeof n.ui.controls.enableLiveRewind ? n.supportLiveRewindOnMobile && n.ui.controls.enableLiveRewind!==!1 : n.supportLiveRewindOnMobile;
        if (t.get().connection.mobile) {
            r.isCellular=!0;
            r.startingQuality = 10
        } else 
            r.startingQuality = 60;
        e.each(r, function(e, t) {
            r.hasOwnProperty(e) && (o += a(e, t))
        });
        e.each(n.configVars, function(e, t) {
            n.configVars.hasOwnProperty(e) && (o += a(e, t))
        });
        n.playlistObject && (o += "&pObj=" + encodeURIComponent(JSON.stringify(n.playlistObject)));
        return o
    }
    function r(e, t) {
        var i = "intent://play/" + o(t) + "#Intent;scheme=bbcmediaplayer;package=air.uk.co.bbc.android.mediaplayer;end";
        s(t, i)
    }
    function s(t, i) {
        if (t.insideIframe) {
            var n = e.Event("navigateTo");
            n.href = i;
            t.trigger(n)
        } else 
            window.location = i
    }
    function l() {
        return /Chrome/.test(navigator.userAgent)
    }
    function d() {
        return t.get().os.kindleSilk ? "http://www.amazon.co.uk/Media-Applications-Technologies-Ltd-Player/dp/B009SQRRCG/ref=sr_1_2?s=mobile-apps&ie=UTF8&qid=1392817067&sr=1-2" : "market://details?id=air.uk.co.bbc.android.mediaplayer"
    }
    function c(t, i, n) {
        var a = new Date - n;
        if (!(a > 600 || "true" == window.name)) {
            var o = document.createElement("style"), r = document.getElementsByTagName("head")[0];
            r.appendChild(o);
            var l = t._container.width(), c = o.sheet ? o.sheet: o.styleSheet;
            l > 300 ? c.insertRule("#marketPlacePopup{position:absolute;width:272px;height:93px;padding-top:16px; left:" + Math.floor((l - 272) / 2) + "px}", 0) : c.insertRule("#marketPlacePopup{position:fixed;width:272px;height:93px;left:10px;top:60px;padding:10px;margin-left:16px;z-index:100000;background-color:white;height:234px}", 0);
            c.insertRule("#marketPlacePopupHeader,#marketPlacePopupMessage{padding:8px 8px 0;color:#fff;font-size:1.4em;line-height:1.143;background:rgba(0,0,0,0.75)}", 1);
            c.insertRule("#marketPlacePopupHeader{font-size:1.8em;line-height:1.333}", 2);
            c.insertRule("#marketPlacePopupMessage{padding-bottom:6px}", 3);
            c.insertRule(".buttons{width:100%;display:block;margin-top:2px}", 4);
            c.insertRule("#marketPlacePopupInstallButton:hover,#marketPlacePopupCancelButton:hover,#marketPlacePopupInstallButton:focus,#marketPlacePopupCancelButton:focus{background:" + i.ui.colour + "}", 5);
            c.insertRule("#marketPlacePopupInstallButton,#marketPlacePopupCancelButton{width:49%;height:44px;padding:0 16px;line-height:44px;font-size:1.4em;text-transform:uppercase;color:#1c1c1c;background:#dcdcdc}", 6);
            c.insertRule("#marketPlacePopupCancelButton{float:right}", 7);
            var p = i.locale(), u = window.embeddedMedia.language, m = u.t("aaampTitle", p), f = u.t("aaampMsg", p), h = u.t("aaampInstall", p), g = u.t("aaampCancel", p);
            t._container.trigger(e.Event("showingAppInstallPrompt"));
            var y = e('<div id="marketPlacePopup"><div id="marketPlacePopupHeader" class="marketPlacePopupHeader">' + m + '</div><div id="marketPlacePopupMessage" class="marketPlacePopupMessage">' + f + '</div><div id="marketPlacePopupButton" class="buttons"><button id="marketPlacePopupInstallButton" class="install">' + h + '</button><button id="marketPlacePopupCancelButton" class="cancel">' + g + "</button></div></div>");
            t._container.unbind("mousedown touchstart mouseup touchend mousemove touchmove");
            t.wrapper[0].style.display = "none";
            if (l > 300) {
                t._container.append(t.wrapperPopup);
                t.wrapperPopup.append(y)
            } else 
                e(document.body).append(y);
            t.resetContainer = function() {
                document.getElementById("marketPlacePopup").parentNode.removeChild(document.getElementById("marketPlacePopup"));
                setTimeout(function() {
                    t._container.trigger(e.Event("hidingAppInstallPrompt"));
                    t.wrapper[0].style.display = ""
                }, 250)
            };
            e("#marketPlacePopupInstallButton").bind({
                "mouseup touchend": function() {
                    e("#marketPlacePopupInstallButton").unbind("mouseup touchend");
                    t.dax.trackAction("smpAirPrompt", "install");
                    t.resetContainer();
                    s(i, d());
                    return !1
                }
            });
            e("#marketPlacePopupCancelButton").bind({
                "mouseup touchend": function() {
                    e("#marketPlacePopupCancelButton").unbind("mouseup touchend");
                    t.dax.trackAction("smpAirPrompt", "cancel");
                    t.resetContainer();
                    return !1
                }
            })
        }
    }
    function p(t, i) {
        var n = e("iframe", t._container).length > 0;
        if (!n) {
            var a =+ new Date, r = e("<iframe />").css({
                display: "none"
            }).attr({
                id: "BBCMediaPlayer",
                src: "bbcmediaplayer://play/" + o(i)
            });
            t.timer = setTimeout(function() {
                clearTimeout(t.timer);
                r.remove();
                c(t, i, a)
            }, 500);
            t._container.append(r)
        }
    }
    function u() {
        var e = this, t = e._settings;
        l() ? r(e, t) : p(e, t);
        return !1
    }
    function m() {
        return t.get().os.android || t.get().os.kindleSilk
    }
    function f() {
        function t() {
            if (void 0 !== document.hidden)
                window.name = document.hidden;
            else 
                for (var e = ["webkit", "moz", "ms", "o"], t = 0; e.length > t; t++)
                    void 0 !== document[e[t] + "Hidden"] && (window.name = document[e[t] + "Hidden"])
        }
        var i = {
            hidden: "visibilitychange",
            mozHidden: "mozvisibilitychange",
            webkitHidden: "webkitvisibilitychange",
            msHidden: "msvisibilitychange",
            oHidden: "ovisibilitychange"
        };
        for (var n in i)
            if (i.hasOwnProperty(n) && n in document) {
                document.addEventListener(i[n], t);
                break
            }
        e(window).blur(function() {
            window.name = "true"
        });
        e(window).focus(function() {
            window.name = "false"
        })
    }
    function h() {
        this.daxname = "SMPAIR";
        this.classStr = "aaampCTAcss";
        this._guidance = "aaampGuidance";
        this.compatibility = m;
        return this
    }
    h.prototype = new n("aaamp", !0);
    h.prototype.play = u;
    h.prototype.destroy = function() {
        clearTimeout(this.timer)
    };
    h.prototype.extraEmbed = f;
    h.prototype.playlistOptions = o;
    h.prototype.getMarketPlaceURL = d;
    h.prototype.hasPlaylist = function() {
        var e = this._settings;
        return e.playlist || e.playlistObject
    };
    return h
});
define("mp/classes/players/apps/wmp", ["mp/modules/device", "mp/classes/players/app"], function(e, t) {
    function i() {
        this.daxname = "SMPWMP";
        this.classStr = "wmpCTAcss";
        this._guidance = "wmpGuidance";
        this.compatibility = a;
        return this
    }
    function n() {
        var e = this._settings;
        if (e.playlistObject) {
            var t = e.playlistObject.items;
            if (t)
                for (var i = t.length - 1; i >= 0; i--)("ident" == t[i].kind || "advert" == t[i].kind) 
                    && t.splice(i, 1);
            window.external.notify("echo:" + e.countername);
            window.external.notify(JSON.stringify(e.playlistObject))
        }
    }
    function a(t) {
        return t.requestWMP && e.get().os.winphone && window.external && "notify"in window.external
    }
    i.prototype = new t("wmp", !0);
    i.prototype.play = n;
    i.prototype.hasPlaylist = function() {
        return !!this._settings.playlistObject
    };
    return i
});
define("mp/classes/players/apps/smp-phantom", ["mp/classes/players/app", "mp/modules/device"], function(e, t) {
    function i(e) {
        return e.phantomPlayer===!0 || e.backgroundPlayer&&!t.get().useOpaque
    }
    function n() {
        this.compatibility = i;
        this.daxname = "SMPPHANTOM";
        this._name = "smp-phantom";
        return this
    }
    n.prototype = new e("smp-phantom");
    n.prototype.play = function() {};
    return n
});
define("mp/modules/players", ["mp/classes/players/flash/smp-flash", "mp/classes/players/flash/smp-flash-noembed", "mp/classes/players/html5/smp-html5", "mp/classes/players/apps/smp-air", "mp/classes/players/apps/wmp", "mp/classes/players/apps/smp-phantom"], function(e, t, i, n, a, o) {
    return [o, t, e, i, n, a]
});
define("mp/classes/decisionengine", ["swfobject-2", "mp/modules/device", "mp/modules/html5", "mp/modules/players"], function(e, t, i, n) {
    function a(e) {
        this._settings = e;
        return this
    }
    a.prototype = {
        embed: function(e) {
            this.player || this.decide();
            var t = this.player;
            t && t.embed(this._settings, e);
            return t
        },
        decide: function() {
            var e = this, i = document.cookie.match(/bbcBumpForcePlayer=(.*?)(?:;|$)/), a = i && i.length ? i[1]: "", o = this._settings, r = [];
            if (o.showUnsupportedMessage && "" === a)
                return e;
            var s = n.concat();
            if (o.overrideSortOrder) {
                for (var l = o.overrideSortOrder, d = 0; l.length > d; d++)
                    for (var c = l[d], p = 0; s.length > p; p++)
                        if (s[p]()._name == c) {
                            r.push(s[p]);
                            s.splice(p, 1);
                            break
                        }
                r = r.concat(s)
            } else if (o.preferHtmlOnMobile && t.get().isMobileOrTablet) {
                for (var u = [], m = 0; s.length > m; m++)
                    "html" == s[m]().type ? r.push(s[m]) : u.push(s[m]);
                r = r.concat(u)
            } else 
                r = s.concat();
            for (m = 0; r.length > m; m++) {
                var f = new r[m];
                if (a == f._name || "" === a && f.compatibility(o)) {
                    e.player = f;
                    e.name = f._name;
                    break
                }
            }
            return e.player
        },
        deviceData: function() {
            return {
                device: t.get(),
                html5: i,
                flash: {
                    flashPlayerVersion: e.getFlashPlayerVersion()
                }
            }
        }
    };
    return a
});
define("mp/classes/rdotreporting", [], function() {
    function e(e) {
        var t = new Image;
        t.src = e + "&cb=" + (new Date).getTime();
        return e
    }
    function t() {
        return [a, "mediaplayout", "-", "bumpv3", window.embeddedMedia.version, "-", "-"]
    }
    function i(e) {
        for (var i = t(), n = 0; e.length > n; n++)
            i.push(encodeURIComponent(e[n]));
        return i.join("/")
    }
    var n, a = "http://r.bbci.co.uk/e", o = {};
    return {
        init: function(e, t) {
            o = {};
            n = e && "https:" != location.protocol;
            t && "" !== t && (a = t.replace(/\/*$/g, ""))
        },
        trackError: function(t) {
            return n ? e(i(["error", t])) : void 0
        }
    }
});
define("mp/libs/securehash", ["mp/libs/securehash"], function() {
    var e = 0;
    return {
        generateHash: function(t) {
            function i(e) {
                return a(n(o(e)))
            }
            function n(e) {
                return s(l(r(e), 8 * e.length))
            }
            function a(t) {
                try {} catch (i) {
                    e = 0
                }
                for (var n, a = e ? "0123456789ABCDEF" : "0123456789abcdef", o = "", r = 0; t.length > r; r++) {
                    n = t.charCodeAt(r);
                    o += a.charAt(n>>>4 & 15) + a.charAt(15 & n)
                }
                return o
            }
            function o(e) {
                for (var t, i, n = "", a =- 1; ++a < e.length;) {
                    t = e.charCodeAt(a);
                    i = e.length > a + 1 ? e.charCodeAt(a + 1) : 0;
                    if (t >= 55296 && 56319 >= t && i >= 56320 && 57343 >= i) {
                        t = 65536 + ((1023 & t)<<10) + (1023 & i);
                        a++
                    }
                    127 >= t ? n += String.fromCharCode(t) : 2047 >= t ? n += String.fromCharCode(192 | t>>>6 & 31, 128 | 63 & t) : 65535 >= t ? n += String.fromCharCode(224 | t>>>12 & 15, 128 | t>>>6 & 63, 128 | 63 & t) : 2097151 >= t && (n += String.fromCharCode(240 | t>>>18 & 7, 128 | t>>>12 & 63, 128 | t>>>6 & 63, 128 | 63 & t))
                }
                return n
            }
            function r(e) {
                for (var t = Array(e.length>>2), i = 0; t.length > i; i++)
                    t[i] = 0;
                for (var i = 0; 8 * e.length > i; i += 8)
                    t[i>>5]|=(255 & e.charCodeAt(i / 8))<<24 - i%32;
                return t
            }
            function s(e) {
                for (var t = "", i = 0; 32 * e.length > i; i += 8)
                    t += String.fromCharCode(e[i>>5]>>>24 - i%32 & 255);
                return t
            }
            function l(e, t) {
                e[t>>5]|=128<<24 - t%32;
                e[(t + 64>>9<<4) + 15] = t;
                for (var i = Array(80), n = 1732584193, a =- 271733879, o =- 1732584194, r = 271733878, s =- 1009589776, l = 0; e.length > l; l += 16) {
                    for (var m = n, f = a, h = o, g = r, y = s, b = 0; 80 > b; b++) {
                        i[b] = 16 > b ? e[l + b] : u(i[b - 3]^i[b - 8]^i[b - 14]^i[b - 16], 1);
                        var v = p(p(u(n, 5), d(b, a, o, r)), p(p(s, i[b]), c(b)));
                        s = r;
                        r = o;
                        o = u(a, 30);
                        a = n;
                        n = v
                    }
                    n = p(n, m);
                    a = p(a, f);
                    o = p(o, h);
                    r = p(r, g);
                    s = p(s, y)
                }
                return Array(n, a, o, r, s)
            }
            function d(e, t, i, n) {
                return 20 > e ? t & i|~t & n : 40 > e ? t^i^n : 60 > e ? t & i | t & n | i & n : t^i^n
            }
            function c(e) {
                return 20 > e ? 1518500249 : 40 > e ? 1859775393 : 60 > e?-1894007588 : - 899497514
            }
            function p(e, t) {
                var i = (65535 & e) + (65535 & t), n = (e>>16) + (t>>16) + (i>>16);
                return n<<16 | 65535 & i
            }
            function u(e, t) {
                return e<<t | e>>>32 - t
            }
            return i(t)
        }
    }
});
define("mp/classes/parentalguidance", ["mp/libs/securehash"], function(e) {
    function t(e) {
        e.preventDefault()
    }
    function i(e) {
        return !e || 4 > e.length?!1 : /[^0-9]/.test(e)?!1 : !0
    }
    function n() {
        return "" !== B && "undefined" != typeof B
    }
    function a() {
        O = "true" === r("pgAgeConfirm");
        var e = r("pgPIN");
        null != e && "" !== e && (B = e);
        var t = r("pgUnlocked");
        U = "true" == t
    }
    function o(e) {
        R = [];
        O=!1;
        this.closedCallbacks = [];
        this.updateCallbacks = [];
        j = e || {
            trackAction: function() {}
        };
        this.guidance = {};
        a();
        this._isLocked = n()&&!U;
        A = window.embeddedMedia.language;
        this.updateGuidanceState();
        return this
    }
    function r(e) {
        var t = "ckps_" + e, i = document.cookie, n = i.indexOf(" " + t + "=");
        - 1 == n && (n = i.indexOf(t + "="));
        if ( - 1 == n)
            i = null;
        else {
            n = i.indexOf("=", n) + 1;
            var a = i.indexOf(";", n);
            - 1 == a && (a = i.length);
            i = unescape(i.substring(n, a))
        }
        return i
    }
    function s(e, t, i) {
        var n = "";
        if (!i) {
            var a = new Date;
            a.setTime(a.getTime() + 15552e6);
            n = "; expires=" + a.toGMTString()
        }
        var o = "ckps_", r = location.hostname;
        - 1 !== r.indexOf(".bbc.co.uk") && (r = ".bbc.co.uk");
        - 1 !== r.indexOf(".bbc.com") && (r = ".bbc.com");
        document.cookie = o + e + "=" + escape(t) + n + "; Domain=" + r + "; Path=/"
    }
    function l(e) {
        for (; e.lastChild;)
            e.removeChild(e.lastChild)
    }
    function d(e, t) {
        R.push(e);
        e.setAttribute("tabindex", "0");
        e.onkeydown = function(t) {
            t = t || window.event;
            var i = t.keyCode || t.which;
            if (9 == i || 13 == i) {
                t.preventDefault ? t.preventDefault() : t.returnValue=!1;
                if (9 == i) {
                    var n = R.length - 1;
                    L = t.shiftKey ? 0 === L ? n : L - 1 : L === n ? 0 : L + 1;
                    setTimeout(function() {
                        R[L].focus()
                    }, 200)
                } else 
                    setTimeout(function() {
                        e.click()
                    }, 200)
            }
        };
        if (t) {
            L = R.length - 1;
            e.focus()
        }
    }
    function c() {
        L = 0;
        R = [];
        d(M)
    }
    function p(e) {
        var t = "";
        if (O) {
            e.style.backgroundPosition = "center";
            t = A.t("pgAriaAgeConfirmCheckBox", H)
        } else {
            e.style.backgroundPosition = "-30px";
            t = A.t("pgAriaAgeConfirmCheckBoxUnchecked", H)
        }
        e.setAttribute("aria-label", t);
        e.textContent = e.innerText = t
    }
    function u(e, t, i) {
        var n = document.getElementById("pgErrorMessage");
        if (n)
            l(n);
        else {
            n = document.createElement("div");
            n.id = "pgErrorMessage";
            var a = n.style;
            a.width = "122%";
            a.backgroundColor = "#d13c2c";
            a.filter = "alpha(opacity=70)";
            a.opacity = .7;
            a.marginTop = "15px";
            a.marginBottom = "5px";
            a.left = "-25px";
            a.position = "relative"
        }
        i && e ? e.insertBefore(n, e.firstChild) : e.appendChild(n);
        var o = document.createElement("div"), r = o.style;
        r.fontSize = "14px";
        r.position = "relative";
        r.width = "90%";
        r.color = "#ffffff";
        r.fontSize = "11px";
        r.textAlign = "center";
        r.paddingTop = r.paddingBottom = r.left = "10px";
        o.innerHTML = t;
        n.appendChild(o)
    }
    function m(e) {
        if (S && I)
            return !1;
        S = document.createElement("div");
        S.setAttribute("tabindex", "0");
        S.id = "parentalGuidanceBackground";
        var t = S.style;
        t.position = "fixed";
        t.width = "5000px";
        t.height = "5000px";
        t.left = "-500px";
        t.top = "-500px";
        t.zIndex = "2147483600";
        t.backgroundColor = "#333333";
        t.filter = "alpha(opacity=85)";
        t.opacity = .85;
        document.body.appendChild(S);
        I = document.createElement("div");
        I.id = "guidancePopup";
        I.setAttribute("role", "alertdialog");
        I.setAttribute("aria-labelledby", "pgHeaderTitle");
        var i = I.style;
        i.position = "fixed";
        i.width = "290px";
        if (window.innerHeight > 350) {
            i.top = "50%";
            i.marginTop = "-175px"
        } else {
            i.top = "5px";
            i.marginTop = "0px"
        }
        i.marginLeft = "-145px";
        i.left = "50%";
        i.lineHeight = "normal";
        i.verticalAlign = "middle";
        i.color = "white";
        i.fontSize = "22px";
        i.zIndex = "2147483600";
        i.backgroundColor = "#141414";
        i.fontFamily = "Arial, sans-serif";
        i.textAlign = "left";
        document.body.appendChild(I);
        I.onkeydown = function(t) {
            t = t || window.event;
            var i = t.keyCode || t.which;
            27 === i && e.close(!1)
        };
        var n = document.createElement("div");
        n.id = "guidanceContainer";
        n.style.paddingRight = n.style.paddingLeft = "26px";
        I.appendChild(n);
        var a = document.createElement("div");
        a.id = "pgHeaderTitle";
        a.style.height = "42px";
        a.style.position = "relative";
        n.appendChild(a);
        var o = document.createElement("span");
        o.innerHTML = A.t("pgHeader", H);
        o.id = "pgHeader";
        o.style.position = "absolute";
        o.style.left = "0px";
        o.style.bottom = "0px";
        o.style.fontSize = "18px";
        o.style.color = "#ffffff";
        a.appendChild(o);
        E = document.createElement("div");
        E.id = "guidanceBody";
        E.style.position = "relative";
        E.style.width = "100%";
        n.appendChild(E);
        D = document.createElement("div");
        D.id = "guidanceFooter";
        var r = D.style;
        r.width = "100%";
        r.height = "42px";
        r.bottom = "0px";
        r.display = "table";
        r.textAlign = "center";
        I.appendChild(D);
        M = document.createElement("a");
        M.setAttribute("aria-label", A.t("pgClose", H));
        M.id = "pgExit";
        var s = M.style;
        M.title = A.t("pgClose", H);
        s.position = "absolute";
        s.right = s.top = "12px";
        s.cursor = "pointer";
        s.width = s.height = "15px";
        s.backgroundImage = "url(//emp.bbci.co.uk/emp/assets/bump-3/images/cross.png)";
        s.backgroundRepeat = "no-repeat";
        I.appendChild(M);
        var l=!1;
        setTimeout(function() {
            l=!0
        }, 500);
        M.onclick = function(t) {
            if (l) {
                t = t || window.event;
                t.preventDefault ? t.preventDefault() : t.returnValue=!1;
                j.trackAction(this.id, "click");
                G.userAction(this.id, "click");
                e.close(!1)
            }
        };
        return !0
    }
    function f(e) {
        var t = document.createElement("div"), i = t.style;
        i.marginTop = "-1px";
        i.position = "absolute";
        i.opacity = "0.2";
        i.filter = "alpha(opacity=20)";
        i.width = "100%";
        i.height = "1px";
        i.backgroundColor = "#ffffff";
        e.appendChild(t)
    }
    function h(e, t) {
        function i() {
            if (l) {
                j.trackAction(n.id, "click");
                G.userAction(n.id, "click");
                c()
            }
        }
        var n = document.createElement("a");
        n.id = t.id;
        n.innerHTML = t.text;
        n.setAttribute("aria-label", t.text + " button");
        var a = n.style;
        a.cursor = "pointer";
        try {
            a.display = "table-cell"
        } catch (o) {
            a.display = "block"
        }
        a.verticalAlign = "middle";
        a.width = a.height = "100%";
        a.fontSize = "18px";
        a.color = "#ffffff";
        D.appendChild(n);
        var r = function() {
            a.backgroundColor = F
        }, s = function() {
            a.backgroundColor = ""
        }, l=!1;
        setTimeout(function() {
            l=!0
        }, 500);
        var c = t.clickFunction;
        n.onclick = i;
        n.onmouseover = r;
        n.onmouseout = s;
        n.onfocus = r;
        n.onblur = s;
        d(n, t.focus);
        return n
    }
    function g(e, t, i) {
        l(D);
        f(D);
        var n = h(e, t);
        if (i) {
            n.style.width = "50%";
            n.style.borderRightColor = "#aaaaaa";
            n.style.borderRightWidth = "1px";
            n.styleborderRightStyle = "solid";
            h(e, i);
            var a = document.createElement("div");
            a.style.left = "50%";
            a.style.position = "absolute";
            a.style.opacity = "0.2";
            a.style.filter = "alpha(opacity=20)";
            a.style.width = "1px";
            a.style.height = D.style.height;
            a.style.backgroundColor = "#ffffff";
            D.appendChild(a)
        }
    }
    function y(e, t) {
        l(E);
        c();
        var i = document.createElement("div");
        i.id = "guidanceMessage";
        var n = i.style;
        n.marginTop = "15px";
        n.marginBottom = "5px";
        n.width = "100%";
        n.position = "relative";
        n.fontSize = "13px";
        E.appendChild(i);
        var a = document.createElement("div");
        a.id = "guidanceG";
        var o = a.style;
        o.display = "inline-block";
        o.margin = "auto";
        o.borderRadius = "50%";
        o.width = "15px";
        o.height = "15px";
        o.backgroundColor = "red";
        o.position = "relative";
        o.top = "-2px";
        o.fontSize = "12px";
        i.appendChild(a);
        var r = document.createElement("span");
        r.innerHTML = "G";
        var s = r.style;
        s.lineHeight = "normal";
        s.position = "relative";
        s.left = "2px";
        s.top = "1px";
        s.color = "#ffffff";
        a.appendChild(r);
        var m = document.createElement("span");
        m.id = "guidanceText";
        m.innerHTML = t;
        s = m.style;
        s.left = "0px";
        s.top = "16px";
        s.marginLeft = "5px";
        s.lineHeight = "16px";
        s.color = "#868686";
        i.appendChild(m);
        var f = document.createElement("div");
        f.id = "ageConfirm";
        var h = f.style;
        h.width = "100%";
        h.height = "57px";
        h.position = "relative";
        h.fontSize = "14px";
        h.color = "#ffffff";
        E.appendChild(f);
        var y = document.createElement("button");
        y.id = "ageConfirmCheck";
        var v = y.style;
        v.background = "url(//emp.bbci.co.uk/emp/assets/bump-3/images/tick.png) no-repeat center";
        v.backgroundPosition = "-30px";
        v.position = "absolute";
        v.left = "0px";
        v.top = "16px";
        v.height = v.width = "25px";
        v.verticalAlign = "middle";
        v.cursor = "pointer";
        v.backgroundColor = "#ffffff";
        v.borderWidth = "0";
        v.textIndent = "-9999px";
        v.whiteSpace = "nowrap";
        v.overflow = "hidden";
        f.appendChild(y);
        d(y, !0);
        p(y);
        var w = document.createElement("span");
        w.innerHTML = A.t("pgAgeConfirmation", H);
        w.style.position = "relative";
        w.style.top = "20px";
        w.style.marginLeft = "35px";
        f.appendChild(w);
        var k=!1;
        setTimeout(function() {
            k=!0
        }, 500);
        y.onclick = function() {
            if (k) {
                O=!O;
                p(y)
            }
        };
        var _ = function() {
            O ? b(e) : u(i, A.t("pgPasswordValidationError", H), !1)
        };
        g(e, {
            id: "pgAgeContinue",
            text: A.t("pgContinue", H),
            clickFunction: _
        })
    }
    function b(e) {
        l(E);
        c();
        var t = document.createElement("div");
        t.setAttribute("role", "alertdialog");
        t.setAttribute("aria-labelledby", "pgInfoHeader");
        t.setAttribute("aria-describedby", "pgInfo");
        t.style.marginBottom = "24px";
        E.appendChild(t);
        var i = document.createElement("div");
        i.id = "pgInfoHeader";
        i.innerHTML = A.t("pgTurnOnParentalGuidance", H);
        i.style.marginTop = "16px";
        i.style.height = "24px";
        i.style.fontSize = "14px";
        t.appendChild(i);
        if (N)
            var n = "pgInfo", a = "pgUnlock";
        else {
            var o = e._intent;
            if ("download" == o) {
                n = "pgInfoPlusDownload";
                a = "pgDownload"
            } else if ("play" == o) {
                n = "pgInfoPlusPlay";
                a = "toolTipPlay"
            } else {
                n = "pgInfo";
                a = o
            }
        }
        var r = '<a id="pgLink" target="_blank" style="color:#ffffff; text-decoration:none;" href="http://www.bbc.co.uk/guidance/help">' + A.t("pgHeader", H) + "</a> " + A.t(n, H), s = document.createElement("div");
        s.id = "pgInfo";
        s.innerHTML = r;
        s.style.fontSize = "13px";
        s.style.color = "#868686";
        t.appendChild(s);
        var p = s.getElementsByTagName("p");
        if (p)
            for (var u = 0; p.length > u; u++) {
                p[u].style.color = "#868686";
                p[u].style.fontSize = "13px"
            }
        var m = document.getElementById("pgLink"), f=!1;
        setTimeout(function() {
            f=!0
        }, 500);
        m.onclick = function() {
            return f
        };
        d(m, !0);
        var h = function() {
            if (f) {
                e.storeAgeConfirmation();
                k(e)
            }
        }, y = function() {
            if (f) {
                e.storeAgeConfirmation();
                e.close(!0)
            }
        }, b = {
            id: "pgTurnOn",
            text: A.t("pgTurnOn", H),
            clickFunction: h
        }, v = {
            id: "toolTipPlay",
            text: A.t(a, H),
            clickFunction: y
        };
        g(e, b, N ? null : v)
    }
    function v(e) {
        var t = document.createElement("div");
        t.id = "pinEntry";
        t.style.width = "100%";
        t.style.height = "40px";
        t.style.marginBottom = "25px";
        e.appendChild(t);
        var i = document.createElement("input");
        i.setAttribute("maxlength", 4);
        i.style.letterSpacing = "5px";
        i.style.display = "block";
        i.style.width = "100%";
        i.style.height = "100%";
        i.style.padding = "0px";
        i.style.resize = "none";
        i.style.color = "#868686";
        i.style.fontSize = "30px";
        i.style.textAlign = "center";
        i.setAttribute("type", "password");
        i.pattern = "[0-9]*";
        t.appendChild(i);
        return i
    }
    function w(e, t) {
        var i=!1, n = function(e) {
            e = e || window.event;
            var n = e.keyCode || e.which;
            13 === n && i && setTimeout(t, 300)
        };
        setTimeout(function() {
            i=!0
        }, 400);
        e.addEventListener ? e.addEventListener("keyup", n) : e.onkeyup = n
    }
    function k(e) {
        l(E);
        c();
        var t = document.createElement("div");
        t.id = "activatePIN";
        t.style.marginTop = "16px";
        t.style.fontSize = "14px";
        t.style.color = "#ffffff";
        E.appendChild(t);
        var n = document.createElement("span");
        n.innerHTML = A.t("pgEnterPIN", H);
        n.style.fontSize = "14px";
        n.style.color = "#ffffff";
        t.appendChild(n);
        var a = document.createElement("div");
        a.innerHTML = "<span>" + A.t("pgActivateInfo", H) + "</span>";
        a.style.marginTop = "10px";
        a.style.marginBottom = "24px";
        a.style.fontSize = "13px";
        a.style.color = "#868686";
        E.appendChild(a);
        var o = v(E);
        o.setAttribute("aria-label", A.t("pgAriaPINEntryDescription", H) + "." + A.t("pgAriaPINPrompt", H));
        d(o, !0);
        var r = function() {
            if (i(o.value)) {
                e.setPIN(o.value);
                e.setLocked(N);
                e.close(!N)
            } else {
                o.value = "";
                u(a, A.t("pgPINValidationError", H), !1);
                o.focus()
            }
        }, s = T(e._intent, "pgActivate");
        w(o, r);
        g(e, {
            id: "pgActivate",
            text: A.t(s, H),
            clickFunction: r
        })
    }
    function _(e, t) {
        var i = document.createElement("div");
        i.style.marginBottom = "20px";
        i.style.width = "100%";
        i.style.height = "15px";
        e.appendChild(i);
        var n = document.createElement("a");
        n.setAttribute("target", "_blank");
        n.setAttribute("href", "http://www.bbc.co.uk/guidance/help");
        n.innerHTML = t;
        n.style.color = "#ffffff";
        n.style.textDecoration = "underline";
        n.style.width = "100%";
        n.style.fontSize = "14px";
        n.style.position = "absolute";
        n.style.bottom = "0px";
        n.style.textAlign = "center";
        i.appendChild(n);
        return n
    }
    function x(e) {
        l(E);
        c();
        var t = document.createElement("div");
        t.id = "enterPIN";
        t.style.marginTop = "16px";
        t.style.paddingBottom = "16px";
        t.style.width = "100%";
        t.style.position = "relative";
        t.style.fontSize = "14px";
        t.style.color = "#ffffff";
        E.appendChild(t);
        var i = document.createElement("div");
        t.appendChild(i);
        var n = document.createElement("span");
        n.innerHTML = A.t("pgEnterPINPrompt", H);
        n.style.fontSize = "14px";
        n.style.color = "#ffffff";
        i.appendChild(n);
        var a = v(E);
        a.setAttribute("aria-label", A.t("pgAriaPINEntryDescription", H) + "." + A.t("pgEnterPINPrompt", H));
        d(a, !0);
        var o = _(E, A.t("pgManageYourPIN", H));
        d(o);
        o.onclick = function() {
            return r
        };
        var r=!1;
        setTimeout(function() {
            r=!0
        }, 500);
        var p = function() {
            if (r)
                if (e.verifyPIN(a.value)) {
                    N && s("pgUnlocked", "true");
                    e.setLocked(!1);
                    e.close(!0)
                } else {
                    a.value = "";
                    u(n, A.t("pgIncorrectPIN", H), !1);
                    a.focus()
                }
        };
        w(a, p);
        var m = T(e._intent, "pgUnlock");
        g(e, {
            id: "pgTurnOff",
            text: A.t(m, H),
            clickFunction: p
        })
    }
    function P() {
        V++;
        clearTimeout(q);
        q = setTimeout(C, 100)
    }
    function T(e, t) {
        return N ? t : "download" == e ? "pgDownload" : "play" == e ? "toolTipPlay" : e
    }
    function C() {
        if (S) {
            var e = S.style;
            e.left = e.top = "-500px";
            e.height = e.width = "5000px"
        }
        if (I) {
            var t = Math.floor(window.innerWidth / 2 - I.offsetWidth / 2);
            (0 > t || V > 0) && (t = 0);
            I.style.left = t + "px";
            var i = window.innerHeight - I.offsetHeight;
            if (window.innerHeight >= I.offsetHeight) {
                var n = Math.floor(i / 2);
                (V > 0 || 5 > n) && (n = 5)
            } else 
                n = 2;
            I.style.marginTop = I.style.marginLeft = "0px";
            I.style.top = n + "px"
        }
    }
    var S, I, E, D, M, A, L, R, O, N, B, U, j, H = "en", G = {
        userAction: function() {}
    }, F = "#f54897", z = function() {
        var e = {
            open: function() {},
            readyState: 4,
            status: 500
        };
        e.send = function() {
            e.onreadystatechange()
        };
        return e
    };
    z = window.XMLHttpRequest ? function() {
        return new XMLHttpRequest
    } : function() {
        return new ActiveXObject("MSXML2.XMLHTTP")
    };
    var q, V = 0;
    o.prototype = {
        show: function(e, i, o) {
            var r = this;
            H = o;
            F = i;
            a();
            if (e) {
                N = e.lockClicked;
                r._intent = e.intent || "play"
            }
            window.embeddedMedia.opened();
            var s = O && (!n() ||!r._isLocked);
            if (!s || N) {
                if (m(r)) {
                    n() ? x(r) : N && s ? b(r) : y(r, e.guidance);
                    document.body.addEventListener && document.body.addEventListener("touchmove", t);
                    V = 0;
                    if (window.addEventListener) {
                        window.addEventListener("resize", C);
                        window.addEventListener("orientationchange", P)
                    }
                }
            } else 
                r.close(!0)
        },
        close: function(e, i) {
            var n = this;
            document.body.removeEventListener && document.body.removeEventListener("touchmove", t);
            if (window.removeEventListener) {
                window.removeEventListener("resize", C);
                window.removeEventListener("orientationchange", P)
            }
            if (S) {
                S.parentNode.removeChild(S);
                S = void 0
            }
            if (I) {
                I.parentNode.removeChild(I);
                I = void 0
            }
            if (!i) {
                window.embeddedMedia.closed();
                for (var a = 0; n.closedCallbacks.length > a; a++)
                    n.closedCallbacks[a](e)
            }
        },
        setPIN: function(t) {
            B = e.generateHash("smpembed" + t);
            s("pgPIN", B)
        },
        setLocked: function(e, t) {
            if (t) {
                s("pgUnlocked", e ? "" : "true");
                U=!e
            }
            this._isLocked = e;
            this.updateGuidanceState()
        },
        verify: function(e, t) {
            var i = this.verifyPIN(e.verifyPIN);
            if (i) {
                this.setLocked(!1);
                e.lockClicked && s("pgUnlocked", "true")
            }
            t(i)
        },
        verifyPIN: function(t) {
            return e.generateHash("smpembed" + t) === B
        },
        setAgeConfirmation: function(e) {
            O = e;
            this.storeAgeConfirmation()
        },
        storeAgeConfirmation: function() {
            s("pgAgeConfirm", O);
            this.setLocked(!0)
        },
        updateGuidanceState: function() {
            var e = this.guidance;
            e.isGuidanceLocked = this._isLocked && n()&&!U;
            e.hasGuidancePIN = n();
            e.guidanceSkipped = O&&!n();
            e.guidancePINPassed=!this._isLocked && n();
            e.canPlay = e.guidanceSkipped || e.hasGuidancePIN && e.guidancePINPassed || U;
            for (var t = 0; this.updateCallbacks.length > t; t++)
                this.updateCallbacks[t](e)
        },
        setSonar: function(e) {
            G = e
        }
    };
    return o
});
define("mp/classes/sonarreporter", [], function() {
    function e(e) {
        return (e + "").replace(/-/gm, "")
    }
    function t(e) {
        var t = this;
        this._echo = null;
        this._player = e;
        this._playerVersion = "undefined";
        this._duration = 0;
        this._isLive=!1;
        this._smpLabels = {
            VPID: "unknown"
        };
        this._playerName = e.daxname;
        this._parentPID = "unknown";
        return t
    }
    function i(e, t) {
        var i = n(e, {});
        i = n(t, i);
        delete i.actionType;
        delete i.controlId;
        return i
    }
    function n(e, t) {
        for (var i in e)
            e.hasOwnProperty(i) && (t[i] = e[i]);
        return t
    }
    function a(e, t) {
        t || (t = {
            connections: []
        });
        e._isLive = t.live;
        var i = e._isLive ? s.MediaConsumptionMode.LIVE: s.MediaConsumptionMode.ON_DEMAND, n = "audio" == t.type ? s.AvType.AUDIO: s.AvType.VIDEO;
        e._smpLabels[f] = t.mediatorIdentifier ? t.mediatorIdentifier : "unknown";
        e.checkEpisodePID();
        var a = e._isLive ? e._smpLabels[f]: "", r = new o(e._parentPID, n, e._parentPIDType, e._smpLabels[f], a, i, s.RetrievalType.STREAM);
        e._isLive && r.setLength(0);
        var l = t.connections[0];
        if (l) {
            var d = l.supplier ? l.supplier: "undefined";
            r.setCDN(d);
            var c = l.encoding ? l.encoding: "undefined";
            r.setCodec(c)
        }
        e._sonarEcho.setMedia(r);
        e.updateSMPLabels(t)
    }
    var o, r, s, l, d, c = "//sa.bbc.co.uk/bbc/nkdata/s", p = "bbc_smp_lr", u = "bbc_smp_bv", m = "connection_type", f = "version_id";
    t.prototype = {
        initComplete: function() {
            var e = this, t = e._player._settings;
            e._parentPIDType = s.PIPsType.EPISODE;
            var i = e.setStatsData(), n = t.appName && "" !== t.appName ? t.appName: e._playerName, a = t.appType && "" !== t.appType ? t.appType: s.ApplicationType.WEB;
            e._sonarEcho = new r(n, a, i);
            e._smpLabels[m] = window.embeddedMedia.demi.isMobileNetwork ? "3g" : "wifi";
            delete e._sonarEcho.consumers[r.Consumers.RUM];
            e._sonarEcho.setPlayerName(e._playerName);
            e._sonarEcho.setCounterName(t.counterName);
            window.embeddedMedia.parentalGuidance.setSonar(e)
        },
        report: function(e, t) {
            var i = this, n = i._isLive ? 0: Math.round(t);
            i._sonarEcho["av" + e + "Event"](1e3 * n, i._smpLabels)
        },
        pause: function(e) {
            this.report("Pause", e)
        },
        end: function(e) {
            this.report("End", e)
        },
        seek: function(e) {
            this.report("Seek", e)
        },
        playResume: function(e) {
            this.report("Play", e)
        },
        buffering: function(e) {
            this.report("Buffer", e)
        },
        userAction: function(e, t, i) {
            var n = this;
            n._sonarEcho && n._sonarEcho.userActionEvent(e, t, i)
        },
        initPlayer: function(e) {
            var t = this;
            e.bind("playlistLoaded SonarPlayEvent SonarConnectionEvent SonarPauseEvent SonarResumeEvent SonarSeekEvent SonarEndEvent SonarUserActionEvent SonarPlayerVersionEvent SonarErrorEvent seeking volumechange uiinfo mediaItemChanged", {
                sonarReporter: t
            }, t.eventHandler);
            require(["//static.bbc.co.uk/nkdata/echoclient/2.0.0/sharedmodules/echo.js"], function() {
                require(["echo"], function(e) {
                    if (e) {
                        t._echo = e;
                        o = e.Media;
                        r = e.EchoClient;
                        s = e.Enums;
                        l = e.ConfigKeys;
                        d = e.Environment;
                        o && r && s && l && d && t.initComplete()
                    }
                })
            })
        },
        setPlayerName: function(t) {
            t && (this._playerName = e(t))
        },
        eventHandler: function(e) {
            var t = e.data.sonarReporter;
            if (t._sonarEcho) {
                var n = t._player.currentTime();
                switch (e.type) {
                case"playlistLoading":
                    t.end(n);
                    break;
                case"playlistLoaded":
                    t.setStatsData();
                    break;
                case"mediaItemChanged":
                    a(t, e.mediaItem);
                    break;
                case"SonarPlayEvent":
                    t.playResume(Math.min(n, 0));
                    if (!t._isLive && t._player.duration() > 0 && 1 > t._duration) {
                        t._duration = t._player.duration();
                        t._sonarEcho.setMediaLength(t._duration)
                    }
                    break;
                case"SonarPauseEvent":
                    t.pause(n);
                    break;
                case"SonarEndEvent":
                    t.end(n);
                    break;
                case"SonarSeekEvent":
                    t._player.paused() || t.playResume(n);
                    break;
                case"SonarResumeEvent":
                    t.playResume(n);
                    break;
                case"SonarUserActionEvent":
                    var o = i(t._smpLabels, e.properties);
                    t.userAction(e.actionType, e.controlId, o);
                    break;
                case"SonarPlayerVersionEvent":
                    if (t._playerVersion !== e.properties.version) {
                        t._playerVersion = e.properties.version;
                        t._sonarEcho.setPlayerVersion((t._playerVersion + "").replace(/[^\d\.]/gim, ""))
                    }
                    break;
                case"seeking":
                    t.seek(n);
                    break;
                case"volumechange":
                    t._sonarEcho.setPlayerVolume(e.muted ? 0 : e.volume);
                    break;
                case"uiinfo":
                    e.properties && (e.properties.subtitles ? t._sonarEcho.setPlayerIsSubtitled(!0) : e.properties.subtitles===!1 && t._sonarEcho.setPlayerIsSubtitled(!1))
                }
            }
        },
        volume: function(e) {
            var t = this;
            e && t._sonarEcho.setPlayerVolume(e)
        },
        updateSMPLabels: function(e) {
            var t = this;
            t._smpLabels[p] = e.rewindWindow > 0 && t._isLive ? "1" : "0"
        },
        mediaBitrate: function(e) {
            e && this._sonarEcho.setMediaBitrate(e)
        },
        setStatsData: function() {
            var e = this, t = {}, i = e._player._settings.statsObject;
            - 1 != c.indexOf("echochamber") && (i.trace = i.trace || "EMP_TESTING-" + Math.random());
            if (l) {
                t[l.COMSCORE.URL] = c;
                t[l.ECHO.TRACE] = i.trace
            }
            e._smpLabels[u] = window.embeddedMedia.version;
            if (i) {
                i.parentPID && (e._parentPID = i.parentPID);
                i.bbcIdentity && (e._smpLabels.bbc_identity = i.bbcIdentity)
            }
            e.checkParentPIDType();
            return t
        },
        checkEpisodePID: function() {
            var e = this, t = e._player._settings.statsObject;
            "unknown" === e._parentPID ? e._parentPID = e._smpLabels[f] : t && (t.parentPID || "unknown" == e._smpLabels[f] || (e._parentPID = e._smpLabels[f]))
        },
        checkParentPIDType: function() {
            var e = this, t = e._player._settings.statsObject;
            t.parentPIDType && ( - 1 != t.parentPIDType.search("episode") ? e._parentPIDType = s.PIPsType.EPISODE : - 1 != t.parentPIDType.search("clip") && (e._parentPIDType = s.PIPsType.CLIP))
        }
    };
    return t
});
define("mp/classes/language", [], function() {
    function e(e) {
        if (i[e])
            for (; i[e].length > 0;)
                i[e].pop()()
    }
    var t = {
        en: {
            pgHeader: "Parental Guidance Lock",
            pgAgeConfirmation: "I am aged 16 or over",
            pgClose: "Close pop up",
            pgContinue: "Continue",
            pgPasswordValidationError: "Please confirm your age before continuing",
            pgAriaAgeConfirmCheckBox: "Age confirmation check box. Checked.",
            pgAriaAgeConfirmCheckBoxUnchecked: "Age confirmation check box. Check to confirm you are sixteen or over.",
            pgTurnOnParentalGuidance: "Turn on the Parental Guidance Lock?",
            pgInfoPlusDownload: "lets you set up a PIN to control access to Guidance-labelled BBC content on your chosen browser.<p>To set up your PIN choose ‘Turn On’. To continue without Parental Guidance choose ‘Download’.</p>",
            pgInfoPlusPlay: "lets you set up a PIN to control access to Guidance-labelled BBC content on your chosen browser.<p>To set up your PIN choose ‘Turn On’. To continue without Parental Guidance choose ‘Play’.</p>",
            pgInfo: "lets you set up a PIN to control access to Guidance-labelled BBC content on your chosen browser.",
            pgTurnOn: "Turn On",
            toolTipPlay: "Play",
            pgEnterPIN: "Please enter a four-digit PIN",
            pgActivateInfo: "You will need your PIN to play Guidance-labelled content",
            pgPINValidationError: "Your PIN must be four digits",
            pgAriaPINEntryDescription: "PIN entry text field",
            pgAriaPINPrompt: "Please enter a valid four-digit PIN",
            pgEnterPINPrompt: "Please enter your four-digit PIN to continue",
            pgManageYourPIN: "Manage your PIN",
            pgUnlock: "Unlock",
            pgIncorrectPIN: "Incorrect PIN. Please try again",
            pgDownload: "Download",
            pgActivate: "Activate",
            6000: "Unable to embed player",
            6003: "Your device is not currently supported",
            6004: "Player failed to initialise",
            6010: "You need to install Flash Player to play this content.",
            downloadFlash: "Download Flash Player now",
            embedTitle: "Embed This Media",
            embedDescription: "Copy the code below to share this content on your own site. Press escape to close",
            embedTerms: "Terms and conditions",
            embedClose: "Close",
            aaampTitle: "BBC Media Player",
            aaampMsg: "To play BBC Programmes you need to install the BBC Media Player application from your app store",
            aaampInstall: "Install",
            aaampCancel: "cancel",
            aaampPlay: "click to play",
            jumpMediaPlayer: "Jump media player",
            mediaPlayerHelp: "Media player help",
            outOfMediaPlayer: "Out of media player. Press enter to return or tab to continue."
        }
    }, i = {}, n = {}, a = {}, o = 5e3, r = {
        "en-GB": "en",
        "en-gb": "en",
        "cy-GB": "cy",
        "cy-gb": "cy",
        "gd-GB": "gd",
        "gd-gb": "gd",
        "ga-GB": "ga",
        "ga-gb": "ga"
    }, s = {
        add: function(i, n) {
            clearTimeout(a[i]);
            t[i] = n;
            e(i)
        },
        load: function(s, l) {
            r[s] && (s = r[s]);
            if (s && /^[\-a-zA-Z]{1,8}$/.test(s)&&!t[s]) {
                if (l) {
                    i[s] || (i[s] = []);
                    i[s].push(l)
                }
                if (!n[s]) {
                    n[s]=!0;
                    var d = document.createElement("script");
                    d.type = "text/javascript";
                    d.charset = "utf-8";
                    d.async=!0;
                    d.src = "//emp.bbci.co.uk/emp/translations/1.0.4/language." + s + ".js";
                    document.getElementsByTagName("head")[0].appendChild(d);
                    clearTimeout(a[s]);
                    a[s] = setTimeout(function() {
                        e(s)
                    }, o)
                }
            } else 
                l && l()
        },
        t: function(e, i, n) {
            r[i] && (i = r[i]);
            if (n) {
                if ("string" == typeof n)
                    return n;
                if (n[i])
                    return n[i]
            }
            return t[i] && t[i][e] ? t[i][e] : t.en[e] || e
        }
    };
    return s
});
define("mp/modules/helpers", ["mp/classes/language"], function(e) {
    return {
        addEmbeddedMedia: function() {
            if (window.embeddedMedia) {
                window.embeddedMedia.playerInstances || (window.embeddedMedia.playerInstances = {});
                window.embeddedMedia.players || (window.embeddedMedia.players = []);
                if (!window.embeddedMedia.playerInitFunctions) {
                    window.embeddedMedia.playerInitFunctions = [];
                    window.embeddedMedia.playersToInit = 0
                }
            } else 
                window.embeddedMedia = {
                    players: [],
                    playerInstances: {},
                    playersToInit: 0,
                    playerInitFunctions: []
                };
            var t = window.embeddedMedia;
            if (!t.openCallbacks) {
                t.openCallbacks = [];
                t.closedCallbacks = [];
                t.opened = function() {
                    for (var e = 0; t.openCallbacks.length > e; e++)
                        t.openCallbacks[e]()
                };
                t.closed = function() {
                    for (var e = 0; t.closedCallbacks.length > e; e++)
                        t.closedCallbacks[e]()
                }
            }
            t.addPlayer || (t.addPlayer = function(e, i) {
                t.playerInstances[e] = i;
                t.players.push(i)
            });
            t.language || (t.language = e);
            if (!t.demi) {
                t.demi = {
                    loading: !0,
                    loadingNetwork: !0,
                    loadingScript: !0,
                    isTabletDevice: !1,
                    isMobileDevice: !1,
                    hasTouchScreen: !1,
                    hasStylus: !1,
                    callbacks: [],
                    ua: window.navigator.userAgent.toLowerCase()
                };
                if (window.orb && window.orb.fig && window.orb.fig.device) {
                    var i = window.orb.fig.device;
                    if (i.isMobile && i.isTablet) {
                        t.demi.isMobileDevice = i.isMobile();
                        t.demi.isTabletDevice = i.isTablet();
                        t.demi.hasTouchScreen = i.isMobile() || i.isTablet();
                        t.demi.loadingScript=!1
                    }
                }
            }
            return t
        }
    }
});
define("mp/classes/player", ["jquery-1.9", "mp/classes/settings", "mp/classes/decisionengine", "mp/classes/errors/error", "mp/modules/device", "mp/classes/rdotreporting", "mp/classes/daxreporting", "mp/classes/parentalguidance", "mp/classes/players/player", "mp/classes/sonarreporter", "mp/modules/helpers"], function(e, t, i, n, a, o, r, s, l, d, c) {
    function p(e, t) {
        if (e) {
            var i = 0;
            t && t.width && (i = t.width());
            (0 === i || isNaN(i)) && (i = 768);
            for (var n = k.length - 1; n > 0&&!(i / k[n] > .98); n--);
            var a = k[n] + "x" + _[n];
            return e.replace("{recipe}", a).replace("$recipe", a)
        }
    }
    function u() {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(e) {
            var t = Math.random() * new Date%16 | 0, i = "x" == e ? t: 3 & t | 8;
            return i.toString(16)
        }).toUpperCase()
    }
    function m(e) {
        var t = window.embeddedMedia.demi;
        if (!e ||!t.loadingNetwork&&!t.loadingScript) {
            t.loading=!1;
            for (var i = 0; t.callbacks.length > i; i++)
                t.callbacks[i]()
        }
    }
    function f(e, t) {
        var i = "https:" == location.protocol ? "https": "http", n = i + "://" + (e && e.mediator && e.mediator.host ? e.mediator.host : "open.live.bbc.co.uk") + "/wurfldemi/";
        g = setTimeout(m, I);
        h(n + "network.jsonp?callback=bumpDemiNetwork");
        t && h(n + "useragent.jsonp?callback=bumpDemiScript&ua=" + encodeURIComponent(window.navigator.userAgent))
    }
    function h(e) {
        var t = document.createElement("script");
        t.type = "text/javascript";
        t.charset = "utf-8";
        t.async=!0;
        t.src = e;
        document.getElementsByTagName("head")[0].appendChild(t)
    }
    var g, y, b = new RegExp("^([0-9\\.]+)%$", "gi"), v = new RegExp("^([0-9\\.]+)px$", "gi"), w = 16 / 9, k = [192, 320, 384, 432, 480, 512, 608, 640, 688, 720, 768, 800, 832, 976, 1024, 1376, 1920], _ = [108, 180, 216, 243, 270, 288, 342, 360, 387, 405, 432, 450, 468, 549, 576, 774, 1080], x = function(e, t) {
        var i;
        "number" == typeof e && (e = String(e));
        "string" == typeof e && e.length > 0 && (e.match(v) ? i = t ? e : e.replace("px", "") : e.match(b) ? i = e : isNaN(Number(e)) || (i = String(Number(e)) + (t ? "px" : "")));
        return i
    }, P = function(e, t, i) {
        t = t && "width" === t ? "width" : "height";
        var n, a = e._settings, o = a.container, r = "width" === t ? "height": "width", s = x(e["_" + r] || a.plugins.player[r]);
        void 0 !== s && (n = s.match(b) ? s : String(isNaN(s) ? o[t]() : Math.ceil("height" === r ? s * w : s / w)));
        return n + (i ? "px" : "")
    }, T = function(e, t, i, n, a) {
        i = i && "width" === i ? "width" : "height";
        if ("undefined" != typeof t) {
            t += "";
            var o = "width" === i ? "height": "width", r = e._settings, s = r.container, l = t, d = l.match(v), c = x(l);
            if (void 0 !== c) {
                e["_" + i] = l;
                r.plugins.player[i] = c
            } else {
                c = P(e, i);
                if (void 0 !== c) {
                    e["_" + i] = "0";
                    r.plugins.player[i] = c
                }
            }
            if (void 0 !== c) {
                n = n || null !== l.match(b)?!0 : !1;
                n || s.css(i, c + "px");
                if (!a && void 0 === e["_" + o]) {
                    var p = P(e, o, d);
                    T(e, p, o, n, !0)
                }
                S(e)
            }
        }
        return e["_" + i]
    }, C = function(t) {
        var i = {
            position: "relative",
            "z-index": "1",
            height: "100%",
            width: "100%",
            "padding-bottom": 0
        };
        t.superResponsive && (i["padding-bottom"] = "56.25%");
        if (!t.responsive&&!t.superResponsive) {
            t.container.css("width", x(t.plugins.player.width + "") + "px");
            t.container.css("height", x(t.plugins.player.height + "") + "px")
        }
        return e("<div />").css(i)
    }, S = function(e) {
        var t = e._settings, i = t.container, n = t.playlistObject && t.playlistObject.holdingImageURL || t.plugins.player.poster || t.overrideHoldingImage, a = p(n, i);
        if (a && e.player.includePoster) {
            var o = Math.abs(16 / 9 - i.width() / i.height()) < .2 ? "cover": "contain";
            i.css({
                "background-image": "url(" + a + ")",
                "background-repeat": "no-repeat",
                "background-position": "center center",
                "background-size": o,
                overflow: "hidden"
            })
        } else 
            i.css("background-image", "");
        "smphtml5" == e.name() && i.css("overflow", "")
    }, I = 1e4;
    window.bumpDemiNetwork = function(e) {
        clearTimeout(g);
        var t = c.addEmbeddedMedia().demi;
        t.isMobileNetwork = e && e.is_mobile_network;
        t.loadingNetwork=!1;
        m(!0)
    };
    window.bumpDemiScript = function(e) {
        clearTimeout(y);
        var t = c.addEmbeddedMedia().demi;
        if (e) {
            t.isTabletDevice = "tablet" === e.type;
            t.isMobileDevice = "mobile" === e.type;
            t.hasTouchScreen = "touchScreen" === e.navigation;
            t.hasStylus = "stylus" === e.navigation
        }
        t.loadingScript=!1;
        m(!0)
    };
    var E = function(n, a) {
        var p = c.addEmbeddedMedia();
        p.version = this.bumpVersion = "3.6.3";
        p.playersToInit++;
        if (!p.demiKickedOff) {
            p.demiKickedOff=!0;
            f(a, p.demi.loadingScript)
        }
        this.bumpRevision = "1573470";
        this.player = new l;
        this.player.fake=!0;
        this.supportsMarkers=!1;
        a.container = n;
        a.domid = n.attr("id");
        var m = this._settings = new t(a);
        m.daxSessionID && "" !== m.daxSessionID || (m.daxSessionID = u());
        o.init(m.enableRdotReporting, m.customRdotBaseURL);
        r.init(m.enableDaxReporting, m.counterName, m.daxSessionID, m.customDAXURL, m.appName, m.appType);
        p.parentalGuidance || (p.parentalGuidance = new s(r));
        m.guidance = p.parentalGuidance.guidance;
        var h = p.parentalGuidance;
        h.closedCallbacks.push(function() {
            var t = e.Event("guidanceClosed");
            t.guidance = m.guidance;
            m.container.trigger(t)
        });
        h.updateCallbacks.push(function(t) {
            var i = e.Event("guidanceUpdated");
            t.noPin = m.guidanceCleared;
            m.guidance = t;
            i.guidance = m.guidance;
            m.container.trigger(i)
        });
        this._decisionEngine = new i(m);
        m.enableSonarReporting&&!m.noTracking && "https:" != location.protocol && (this._sonarReporter = new d(this, h));
        return this
    };
    E.prototype = {
        showGuidance: function(e, t, i, n) {
            var a = {
                lockClicked: i
            };
            a.title = e ? e : "";
            a.guidance = t ? t : "";
            a.intent = n ? n : "play";
            var o = window.embeddedMedia.parentalGuidance, r = this._settings;
            r.guidanceCleared || o.show(a, r.ui.colour, r.locale())
        },
        lockGuidance: function() {
            this._settings.guidance.hasGuidancePIN===!0 ? window.embeddedMedia.parentalGuidance.setLocked(!0, !0) : this.unlockGuidance()
        },
        unlockGuidance: function() {
            this.showGuidance("", "", !0)
        },
        name: function() {
            return this._decisionEngine.name
        },
        decide: function() {
            this._decisionEngine.decide();
            if (this._settings.enableSonarReporting && this._sonarReporter) {
                this._sonarReporter.setPlayerName(this._decisionEngine.name);
                "https:" != location.protocol && "smphtml5" === this._decisionEngine.name && this._sonarReporter.initPlayer(this)
            }
            return this.name()
        },
        settings: function() {
            return this._settings.set.apply(this._settings, e.makeArray(arguments))
        },
        deviceData: function() {
            return this._decisionEngine.deviceData()
        },
        poster: function(e) {
            if ("string" == typeof e) {
                this._settings.set({
                    plugins: {
                        player: {
                            poster: e
                        }
                    }
                });
                S(this)
            }
            return this._settings.plugins.player.poster
        },
        play: function() {
            this.player.play()
        },
        autoplay: function(e) {
            "boolean" == typeof e && (this._settings.autoplay = e);
            return this._settings.autoplay
        },
        pause: function() {
            this.player.pause()
        },
        suspend: function() {
            this.player.suspend()
        },
        currentTime: function(e) {
            return this.player.currentTime(e)
        },
        duration: function() {
            return this.player.duration()
        },
        muted: function(e) {
            return this.player.muted(e)
        },
        loadPlaylist: function(e, t, i, n) {
            var a = this, o = a.player, r = a._settings;
            r.playlist = "";
            r.playlistObject = null;
            r.webcastData = {};
            r.statsObject.parentPID = void 0;
            r.statsObject.parentPIDType = void 0;
            e && ("string" != typeof e ? r.playlistObject = e : r.playlist = e);
            if ("object" == typeof t) {
                r.startTime = 0;
                a.settings(t)
            } else {
                a.autoplay(t);
                r.startTime = 0;
                r.overrideHoldingImage = i || r.overrideHoldingImage
            }
            o._settings = r;
            if (o.includePoster) {
                var s = function() {
                    a.poster(r.overrideHoldingImage)
                };
                r.autoplay ? setTimeout(s, 100) : s()
            }
            if (n) {
                n.startTime && (r.startTime = n.startTime);
                "boolean" == typeof n.promptGuidanceWhenAutoplay && (r.promptGuidanceWhenAutoplay = n.promptGuidanceWhenAutoplay)
            }
            o.loadPlaylist()
        },
        kind: function() {
            return this.player.kind()
        },
        stop: function() {
            return this.player.stop()
        },
        loadPlugin: function(e, t) {
            if (this.player.loadPlugin)
                this.player.loadPlugin(e, t);
            else if (e) {
                e.data = t;
                this._settings.plugins.toLoad.push(e)
            }
        },
        updateUiConfig: function(t) {
            var i = this._settings, n = i.locale();
            e.extend(i.ui, t);
            n != i.locale() && window.embeddedMedia.language.load(i.locale());
            this.player.updateUiConfig(t)
        },
        setData: function(e) {
            this._settings.plugins.smpAirPluginData = e;
            this.player.setData(e)
        },
        setPoster: function(e) {
            this.player.setPoster(e, this)
        },
        dispatchEvent: function(e, t) {
            this.player.dispatchEvent(e, t)
        },
        pauseAt: function(e) {
            this.player.pauseAt(e)
        },
        webcastData: function(e) {
            this._settings.webcastData = e;
            this.player.webcastData(e)
        },
        volume: function(e) {
            var t = this.player.volume(e);
            return t
        },
        paused: function() {
            return this.player.paused()
        },
        ended: function() {
            return this.player.ended()
        },
        seeking: function() {
            return this.player.seeking()
        },
        width: function(e) {
            return T(this, e, "width")
        },
        height: function(e) {
            return T(this, e, "height")
        },
        bind: function() {
            var t = this._settings.container;
            t.bind.apply(t, e.makeArray(arguments));
            return this
        },
        unbind: function() {
            var t = this._settings.container;
            t.unbind.apply(t, e.makeArray(arguments));
            return this
        },
        _loadOptions: function(t, s) {
            var l = this;
            this._loadOptionsArg = t;
            var d, c = l.player, p = l._settings;
            if ("string" == typeof t)
                d = p.playlist = t;
            else if ("object" == typeof t) {
                p.set(t);
                d = t.playlist
            }
            if (c.fake || s) {
                l._decisionEngine = new i(p);
                p.validate();
                l.decide();
                var u = l._decisionEngine.name;
                if (u) {
                    var m = function(e, t) {
                        window.embeddedMedia.addPlayer(p.domid, e);
                        l.player = e;
                        e.dax = r;
                        l.supportsMarkers=!!t;
                        S(l);
                        "smp-flash-noembed" == e.name() && (e.reloadScope = l)
                    }, f=!0, h = window.innerHeight;
                    isNaN(h) && (h = 1e3);
                    p.container.offset().top > h && (f=!1);
                    if (f || 1 == window.embeddedMedia.playersToInit || p.delayEmbed===!0 || s) {
                        l.addInitialisationListener();
                        l._decisionEngine.embed(m);
                        if (!p.responsive&&!p.superResponsive) {
                            p.container.css("width", x(p.plugins.player.width + "") + "px");
                            p.container.css("height", x(p.plugins.player.height + "") + "px")
                        }
                        window.embeddedMedia.playersToInit--
                    } else 
                        window.embeddedMedia.playerInitFunctions.push(function() {
                            l.addInitialisationListener();
                            l._decisionEngine.embed(m);
                            if (!p.responsive&&!p.superResponsive) {
                                p.container.css("width", x(p.plugins.player.width + "") + "px");
                                p.container.css("height", x(p.plugins.player.height + "") + "px")
                            }
                        });
                    if (window.embeddedMedia.playerInitFunctions.length > 0 && window.embeddedMedia.playerInitFunctions.length == window.embeddedMedia.playersToInit)
                        for (var g = 0; window.embeddedMedia.playerInitFunctions.length > g; g++) {
                            var y = 1e3 * (g + 2), b = g;
                            setTimeout(window.embeddedMedia.playerInitFunctions[b], y)
                        }
                    } else {
                    var v = e.Event("error");
                    S(l);
                    v.errorData = new n(!a.get().supportsFlash || p.showUnsupportedMessage ? 6003 : 6010, p);
                    p.ui.hideDefaultErrors || this.showErrorDialog(v.errorData);
                    p.container.trigger(v);
                    o.trackError(v.errorData.code())
                }
                return this
            }
            c.loadPlaylist(d, p.autoplay, p.overrideHoldingImage)
        },
        load: function(t) {
            var i = this, n = function() {
                window.embeddedMedia.language.load(i._settings.locale(), function() {
                    i._loadOptions(t)
                });
                return i
            }, o = function() {
                window.embeddedMedia.demi.loading ? window.embeddedMedia.demi.callbacks.push(n) : n();
                return i
            };
            if (a.get().waitForLoad) {
                if ("complete" === document.readyState || "interactive" === document.readyState)
                    return o();
                e(document).ready(o);
                return i
            }
            if (window.embeddedMedia.demi.loading) {
                window.embeddedMedia.demi.callbacks.push(n);
                return i
            }
            return n()
        },
        reset: function() {
            return this
        },
        addInitialisationListener: function() {
            var t = this, i = setTimeout(function() {
                var i = e.Event("error");
                i.errorData = new n(6004, t._settings);
                t._settings.container && t._settings.container.trigger(i)
            }, t._Timeout || 15e3);
            this._settings.container.bind("initialised", function() {
                clearTimeout(i)
            })
        },
        showErrorDialog: function(e) {
            var t = this._settings, i = t.container, n = t.ui.miniMode && t.ui.miniMode.enabled?!0 : !1;
            i.empty();
            var o = document.createElement("div");
            o.style.position = "absolute";
            o.style.top = o.style.left = o.style.bottom = o.style.right = "0px";
            var r = document.createElement("div"), s = r.style;
            s.position = "absolute";
            if (n) {
                s.width = "150px";
                s.height = "80px";
                s.fontSize = "16px"
            } else {
                s.width = "320px";
                s.height = "160px";
                s.fontSize = "18px";
                if (a.get().os.winphone)
                    s.top = s.left = s.marginLeft = s.marginTop = 0;
                else {
                    s.top = s.left = "50%";
                    s.marginLeft = "-160px";
                    s.marginTop = "-80px"
                }
            }
            s.verticalAlign = "middle";
            s.textAlign = "center";
            s.color = "white";
            s.backgroundColor = "#252525";
            var l = document.createElement("div"), d = l.style;
            d.position = "absolute";
            d.display = "block";
            d.top = d.left = d.right = n ? "8px" : "20px";
            l.innerHTML = e.description();
            if (6010 == e.code()&&!n) {
                var c = document.createElement("div"), p = c.style;
                p.position = "absolute";
                p.bottom = "20px";
                p.left = "10px";
                p.right = "10px";
                var u = window.embeddedMedia.language.t("downloadFlash", t.locale(), t.overrideErrorLink.linkTitle), m = t.overrideErrorLink.link || "http://get.adobe.com/flashplayer";
                c.innerHTML = '<a href="' + m + '" target="_blank" style="color:white;display:inline-block;font-size: 14px;background-color : black;text-decoration: none;padding:10px;">' + u + "</a>";
                r.appendChild(c)
            }
            r.appendChild(l);
            i.append(C(t).append(r))
        }
    };
    return E
});
define("bump-3", ["jquery-1.9", "mp/classes/player"], function(e, t) {
    var i = {};
    if ("function" != typeof e.fn.player) {
        var n = {
            getPlayer: function() {
                var t = e(e(this).get(0)), n = t.attr("id");
                return i[n]
            },
            player: function(n) {
                var a = e(e(this).get(0)), o = a.attr("id");
                if ("undefined" == typeof i[o] || i[o].settings().container[0] != a[0]) {
                    if (!o) {
                        var r = "bbcMediaPlayer", s = 0;
                        do 
                            o = r + s++;
                        while (document.getElementById(o));
                        a.attr("id", o)
                    }
                    i[o] = new t(a, n)
                } else 
                    i[o].settings(n);
                return i[o]
            }
        };
        e.fn.extend(n)
    }
    if ("function" != typeof e.players) {
        var a = {
            players: function() {
                return i
            }
        };
        e.extend(a)
    }
    return e
});
