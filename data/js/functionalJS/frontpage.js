/*
 RequireJS text 0.27.0 Copyright (c) 2010-2011, The Dojo Foundation All Rights Reserved.
 Available via the MIT or new BSD license.
 see: http://github.com/jrburke/requirejs for details
*/

/*!
 * Autolinker.js
 * 0.12.2
 *
 * Copyright(c) 2014 Gregory Jacobs <greg@greg-jacobs.com>
 * MIT Licensed. http://www.opensource.org/licenses/mit-license.php
 *
 * https://github.com/gregjacobs/Autolinker.js
 */

(function() {
    define("lodash", [], function() {
        return window._
    }), define("bs/util/pubsub", ["lodash"], function(e) {
        var t = function() {
            this.channelHandlers = {}
        };
        return t.prototype = {
            channels: {
                scores_update: "story_scores:update",
                scores_update_error: "story_scores:error",
                user_auth: "user_auth:success",
                scrolltop: "scrolltop",
                popular_stories_rendered: "popular_stories_rendered",
                new_stories_rendered: "new_stories_rendered"
            },
            on: function(e, t, n, r) {
                var i = this.channelHandlers[t];
                return i || (i = this.channelHandlers[t] = []), n && typeof n == "function" && (n._scope = r || null, i.push(n)), this
            },
            one: function(t, n, r) {
                var i = this, s = function() {
                    var o = e.toArray(arguments);
                    r.apply(this, o), i.off(t, n, s)
                };
                return this.on(t, n, s), this
            },
            off: function(t, n, r) {
                var i = this.channelHandlers[n];
                return i && (this.channelHandlers[n] = e.reject(i, function(e) {
                    return e === r
                })), this
            },
            fire: function(t) {
                var n = this.channelHandlers[t], r = e.toArray(arguments).slice(1);
                return n && e.each(n, function(e) {
                    e.apply(e._scope, r)
                }, this), this
            }
        }, new t
    }), define("bs/util/cookies", [], function() {
        var e = {
            set: function(e) {
                var t = [];
                typeof e == "object" && (typeof e.name == "string" && e.name.length > 0 && (typeof e.value == "string" || typeof e.value == "number" || typeof e.value == "boolean") && t.push($.trim(e.name) + "=" + encodeURIComponent(e.value)), e.expires instanceof Date && t.push("expires=" + e.expires.toGMTString()), typeof e.path == "string" && t.push("path=" + e.path), typeof e.domain == "string" && t.push("domain=" + e.domain), typeof e.secure == "boolean" && e.secure && t.push("secure"), document.cookie = t.join(";"))
            },
            get: function(t) {
                var n = e.getAll();
                return typeof n[t] != "undefined" ? n[t] : null
            },
            getAll: function() {
                var e = {}, t = document.cookie;
                if (t) {
                    var n = null, r = null, i = null, s = t.split(";"), o = s.length - 1;
                    if (o >= 0)
                        do 
                            i = s[o].split("="), n = $.trim(i[0]), r = i[1], n !== "expires" && typeof r != "undefined" && (e[n] = decodeURIComponent(r));
                    while (o--)
                    }
                return e
            }, remove: function(t) {
                var n = {};
                typeof t == "string" ? n.name = t : typeof t == "object" && $.extend(n, t), n.value = "", n.expires = new Date(0), e.set(n)
            }, expires: {
                oneYear: function() {
                    var e = 31536e6;
                    return new Date((new Date).valueOf() + e)
                }
            }
        };
        return e
    }), define("bs/util/validation", [], function() {
        return {
            isURL: function(e) {
                return /^https?:\/\/(((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:)*@)?(((\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]))|((([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?)(:\d*)?)(\/((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)+(\/(([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)*)*)?)?(\?((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|[\uE000-\uF8FF]|\/|\?)*)?(\#((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|\/|\?)*)?$/i.test(e)
            },
            isEmail: function(e) {
                return /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\ ".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA -Z\-0-9]+\.)+[a-zA-Z]{2,}))$/.test(e)
            }
        }
    }), define("jquery", [], function() {
        return jQuery
    }), define("bs/util/xhr", ["bs/util/cookies", "jquery", "lodash"], function(e, t, n) {
        var r = {
            dataType: "digg",
            converters: {
                "text digg": function(e) {
                    var n = t.parseJSON(e);
                    if (n.status !== "ok")
                        throw new Error("Bad Digg API Response: " + n.mesg);
                    return n
                }
            },
            traditional: !0
        };
        return {
            config: n.clone(r),
            dataProvider: function() {
                var i = function(e) {
                    this.config = n.extend(n.clone(r), e)
                };
                return i.prototype = {
                    isPolling: function() {
                        return !!this.pollingRef
                    },
                    request: function(r) {
                        var i = n.extend(n.clone(this.config), r);
                        return i.type === "POST" && (i.data = i.data || {}, i.data._xsrf = e.get("_xsrf")), t.ajax(i)
                    },
                    startPolling: function(e, n) {
                        var r = n.delayStart ||!1;
                        t.proxy(function() {
                            r ? r=!1 : this.request(n), this.pollingRef = setTimeout(t.proxy(arguments.callee, this), e)
                        }, this)()
                    },
                    stopPolling: function() {
                        this.pollingRef && (clearTimeout(this.pollingRef), this.pollingRef = null)
                    }
                }, i
            }()
        }
    }), define("bs/util/helpers", [], function() {
        var e = ["You've got a lot of work to do, slacker.", "C'mon!", 'Just "Mark as Read". You know you want to.'];
        return {
            transitionEndEventNames: {
                WebkitTransition: "webkitTransitionEnd",
                MozTransition: "transitionend",
                OTransition: "oTransitionEnd otransitionend",
                msTransition: "MSTransitionEnd",
                transition: "transitionend"
            },
            popWin: function(e, t, n) {
                return n.width && n.height && (n.left = window.screen.width / 2 - n.width / 2, n.top = window.screen.height / 2 - n.height / 2), n = $.param(n).replace(/&/g, ","), window.open(e, t, n)
            },
            createGlobalCallback: function(e) {
                var t = "cb" + (new Date).valueOf();
                return window[t] = function() {
                    e.apply(this, arguments);
                    try {
                        delete window[t]
                    } catch (n) {
                        window[t] = null
                    }
                }, t
            },
            listen: function(e, t, n) {
                if (t.addEventListener)
                    t.addEventListener(e, n, !1);
                else if (t.attachEvent) {
                    var r = t.attachEvent("on" + e, n);
                    return r
                }
            },
            unlisten: function(e, t, n) {
                if (t.addEventListener)
                    t.removeEventListener(e, n, !1);
                else if (t.detachEvent) {
                    var r = t.detachEvent("on" + e, n);
                    return r
                }
            },
            throttle: function(t, n) {
                var r = null;
                return function() {
                    var e = this, i = arguments;
                    clearTimeout(r), r = setTimeout(function() {
                        t.apply(e, i)
                    }, n)
                }
            },
            deparam: function(e) {
                var t, n = {};
                return e[0] === "?" && (e = e.substring(1)), t = e.split("&"), $.each(t, function(e, t) {
                    var r = t.split("=");
                    n[r[0]] = decodeURIComponent(r[1])
                }), n
            },
            param: function(e) {
                return $.param(e, !0)
            },
            makeExpandingArea: function(e) {
                var t = e.querySelector("textarea"), n = e.querySelector("span");
                t.addEventListener ? (t.addEventListener("input", function() {
                    n.textContent = t.value
                }, !1), n.textContent = t.value) : t.attachEvent && (t.attachEvent("onpropertychange", function() {
                    n.innerText = t.value
                }), n.innerText = t.value), e.className += " active"
            },
            enforceRangeLimits: function(e, t, n) {
                return e < t ? t : e > n ? n : e
            },
            getMaxUnreadMessage: function() {
                var t = Math.floor(Math.random() * e.length);
                return e[t]
            },
            isElementInViewport: function(e) {
                var t = e[0].getBoundingClientRect(), n = e.parents(".scrollable"), r;
                r = n[0] && n[0].getBoundingClientRect();
                if (!r)
                    return;
                return t.bottom >= r.bottom ? "bottom" : t.top <= r.top ? "top" : !0
            },
            chunkString: function(e, t) {
                var n = [], r = 0, i = t;
                if (e.length%t !== 0)
                    throw new Error("chunkString(): cannot chunk str argument based on chunkLen");
                while (i < e.length)
                    n.push(e.substr(r, t)), r += t, i += t;
                return n
            },
            getTweetStoryLink: function(e) {
                var t;
                e = e || {};
                if (!e.content_id&&!e.url)
                    throw new Error("Missing content_id and url arguments. One is required.");
                if (!e.title)
                    throw new Error("Missing title argument");
                return t = e.content_id ? "http://on.digg.com/" + e.content_id : e.url, ["https://twitter.com/intent/tweet?url=", encodeURIComponent(t), "&via=Digg&text=", encodeURIComponent(e.title)].join("")
            },
            getFacebookStoryShareLink: function(e) {
                var t;
                e = e || {};
                if (!e.app_id)
                    throw new Error("Missing Facebook App ID argument");
                if (!e.content_id&&!e.url)
                    throw new Error("Missing content_id and url arguments. One is required.");
                return t = e.content_id ? "http://on.digg.com/" + e.content_id : e.url, ["https://www.facebook.com/dialog/feed?display=popup&app_id=", e.app_id, "&link=", encodeURIComponent(t), "&redirect_uri=http://digg.com/fbshare"].join("")
            },
            getTweetURL: function(e) {
                e = e || {};
                if (!e.twitter_screen_name)
                    throw new Error("Missing Twitter screen name argument");
                if (!e.tweet_id)
                    throw new Error("Missing tweet_id argument");
                return ["http://www.twitter.com/", e.twitter_screen_name, "/status/", e.tweet_id].join("")
            }
        }
    }), define("bs/util/i18n", [], function() {
        var e = {
            Jan: "Jan",
            Feb: "Feb",
            Mar: "Mar",
            Apr: "Apr",
            May: "May",
            Jun: "Jun",
            Jul: "Jul",
            Aug: "Aug",
            Sep: "Sep",
            Oct: "Oct",
            Nov: "Nov",
            Dec: "Dec"
        };
        return {
            __: function(t, n) {
                return e[t] ? e[t] : t
            }
        }
    }), define("bs/util/format", ["bs/util/i18n"], function(e) {
        function t(e, t) {
            return t || (t = 1), e ? e : t
        }
        function n(t) {
            var n = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            return e.__(n[t])
        }
        function r(e) {
            var t = e.getHours(), n = e.getMinutes(), r = t >= 12 ? "pm": "am";
            t%=12, t = t ? t : 12, n = n < 10 ? "0" + n : n;
            var i = t + ":" + n + " " + r;
            return i
        }
        return {
            secondsToFuzzyTime: function(e, r) {
                var i = 6e4, s = 60 * i, o = 24 * s, u = 7 * o, a = o * 3652425 / 1e4, f = new Date(e * 1e3), l = new Date, c = l - f, h, p = r ? "": " ago";
                return c < i ? h = "just now" : c / i <= 45 ? h = t(Math.floor(c / i), 1) + "m" + p : c / s <= 16 ? h = t(Math.floor(c / s), 1) + "h" + p : c / o <= 6 ? h = t(Math.floor(c / o), 1) + "d" + p : c / u <= 6 ? h = t(Math.floor(c / u), 1) + "w" + p : c / a <= 1 ? h = [f.getDate(), n(f.getMonth())].join(" ") : h = [f.getDate(), n(f.getMonth()), f.getFullYear()].join(" "), h
            },
            secondsToDiggReaderTime: function(e) {
                var t, i = new Date(e * 1e3), s = i.getDate(), o = i.getMonth(), u = i.getFullYear(), a = new Date, f = a.getDate(), l = (new Date(new Date(a.getFullYear(), a.getMonth(), a.getDate() - 1))).getDate(), c = a.getMonth(), h = a.getFullYear();
                return s === f && o === c && u === h ? t = r(i) : s === l && o === c && u === h ? t = "Yesterday" : u === h ? t = n(o) + " " + s : t = n(o) + " " + s + " " + u, t
            },
            intWithCommas: function(e, t) {
                var n = (e + "").split("").reverse(), r = [], i, s, o = 0, u;
                for (i = 1, s = n.length; i <= s; i++)
                    i%3 === 0 ? (r.push(n[i - 1]), i !== s && (r.push(","), o++)) : r.push(n[i - 1]);
                return t && s > 4 ? (u = r.reverse().join("").split(",")[0], o === 1 ? u += "k" : o === 2 && (u += "m")) : u = r.reverse().join(""), u
            },
            truncate: function(e, t, n, r) {
                if (!e || e.length <= t)
                    return e;
                typeof r == "undefined" && (r = "…");
                var i;
                if (n) {
                    i = e.substr(0, t - 3).lastIndexOf(" ");
                    var s = e.substr(i - 1, 1);
                    while ([",", ";", ":", "(", "-", "–", "—", " "].indexOf(s)!==-1) {
                        i--;
                        if (i===-1)
                            break;
                        s = e.substr(i - 1, 1)
                    }
                }
                if (!n || i===-1)
                    i = t - 3;
                return e.substr(0, i) + r
            },
            slugify: function(e) {
                return e.toString().toLowerCase().replace(/\s+/g, "-").replace(/[^\w\-]+/g, "").replace(/\-\-+/g, "-").replace(/^-+/, "").replace(/-+$/, "")
            }
        }
    }), define("bs/widgets/dailydiggsub", ["bs/core", "bs/util/xhr", "jquery"], function(e, t, n) {
        function u(e) {
            var o = n(e.target).find(".form-dailydigg-input"), u = n.trim(o.val()), l = require("bs/core").user.isLoggedIn() ? "/api/settings/email.json": "/api/dailydiggemail.json", c = new t.dataProvider({
                url: l,
                type: "POST",
                data: {
                    email: u
                },
                success: a,
                error: f
            }, e), h;
            return i = e, r = n(e.target), s || (h = document.createElement("div"), h.className = "response", r.append(h), s=!0), c.request(), e.stopPropagation(), e.preventDefault(), !1
        }
        function a(e) {
            r.removeClass("error").addClass("success").find(".response").html("Thanks for subscribing!"), require("bs/core").pubsub.fire("dailydigg_subscribe_success", i)
        }
        function f(e) {
            r.addClass("error").find(".response").html("There was a problem submitting your email. Please try again.")
        }
        var r, i, s=!1, o=!1;
        return {
            init: function() {
                if (o)
                    return;
                o=!0, n(document).on("submit", ".form-dailydigg-sub", u)
            }
        }
    }), function() {
        var e = ["Msxml2.XMLHTTP", "Microsoft.XMLHTTP", "Msxml2.XMLHTTP.4.0"], t = /^\s*<\?xml(\s)+version=[\'\"](\d)*.(\d)*[\'\"](\s)*\?>/im, n = /<body[^>]*>\s*([\s\S]+)\s*<\/body>/im, r = typeof location != "undefined" && location.href, i = r && location.protocol && location.protocol.replace(/\:/, ""), s = r && location.hostname, o = r && (location.port || void 0), u = [];
        define("text", [], function() {
            var a, f, l;
            return typeof window != "undefined" && window.navigator && window.document ? f = function(e, t) {
                var n = a.createXhr();
                n.open("GET", e, !0), n.onreadystatechange = function() {
                    n.readyState === 4 && t(n.responseText)
                }, n.send(null)
            } : typeof process != "undefined" && process.versions && process.versions.node ? (l = require.nodeRequire("fs"), f = function(e, t) {
                t(l.readFileSync(e, "utf8"))
            }) : typeof Packages != "undefined" && (f = function(e, t) {
                var n = new java.io.File(e), r = java.lang.System.getProperty("line.separator"), n = new java.io.BufferedReader(new java.io.InputStreamReader(new java.io.FileInputStream(n), "utf-8")), i, s, o = "";
                try {
                    i = new java.lang.StringBuffer, (s = n.readLine()) && s.length() && s.charAt(0) === 65279 && (s = s.substring(1));
                    for (i.append(s); (s = n.readLine()) !== null;)
                        i.append(r), i.append(s);
                    o = String(i.toString())
                } finally {
                    n.close()
                }
                t(o)
            }), a = {
                version: "0.27.0",
                strip: function(e) {
                    if (e) {
                        var e = e.replace(t, ""), r = e.match(n);
                        r && (e = r[1])
                    } else 
                        e = "";
                    return e
                },
                jsEscape: function(e) {
                    return e.replace(/(['\\])/g, "\\$1").replace(/[\f]/g, "\\f").replace(/[\b]/g, "\\b").replace(/[\n]/g, "\\n").replace(/[\t]/g, "\\t").replace(/[\r]/g, "\\r")
                },
                createXhr: function() {
                    var t, n, r;
                    if (typeof XMLHttpRequest != "undefined")
                        return new XMLHttpRequest;
                    for (n = 0; n < 3; n++) {
                        r = e[n];
                        try {
                            t = new ActiveXObject(r)
                        } catch (i) {}
                        if (t) {
                            e = [r];
                            break
                        }
                    }
                    if (!t)
                        throw Error("createXhr(): XMLHttpRequest not available");
                    return t
                },
                get: f,
                parseName: function(e) {
                    var t=!1, n = e.indexOf("."), r = e.substring(0, n), e = e.substring(n + 1, e.length), n = e.indexOf("!");
                    return n!==-1 && (t = e.substring(n + 1, e.length), t = t === "strip", e = e.substring(0, n)), {
                        moduleName: r,
                        ext: e,
                        strip: t
                    }
                },
                xdRegExp: /^((\w+)\:)?\/\/([^\/\\]+)/,
                useXhr: function(e, t, n, r) {
                    var i = a.xdRegExp.exec(e), s;
                    return i ? (e = i[2], i = i[3], i = i.split(":"), s = i[1], i = i[0], (!e || e === t) && (!i || i === n) && (!s&&!i || s === r)) : !0
                },
                finishLoad: function(e, t, n, r, i) {
                    n = t ? a.strip(n) : n, i.isBuild && i.inlineText && (u[e] = n), r(n)
                },
                load: function(e, t, n, u) {
                    var f = a.parseName(e), l = f.moduleName + "." + f.ext, c = t.toUrl(l), h = u && u.text && u.text.useXhr || a.useXhr;
                    !r || h(c, i, s, o) ? a.get(c, function(t) {
                        a.finishLoad(e, f.strip, t, n, u)
                    }) : t([l], function(e) {
                        a.finishLoad(f.moduleName + "." + f.ext, f.strip, e, n, u)
                    })
                },
                write: function(e, t, n) {
                    if (t in u) {
                        var r = a.jsEscape(u[t]);
                        n.asModule(e + "!" + t, "define(function () { return '" + r + "';});\n")
                    }
                },
                writeFile: function(e, t, n, r, i) {
                    var t = a.parseName(t), s = t.moduleName + "." + t.ext, o = n.toUrl(t.moduleName + "." + t.ext) + ".js";
                    a.load(s, n, function() {
                        var t = function(e) {
                            return r(o, e)
                        };
                        t.asModule = function(e, t) {
                            return r.asModule(e, o, t)
                        }, a.write(e, s, t, i)
                    }, i)
                }
            }
        })
    }(), define("text!bs/templates/popover.html", [], function() {
        return '<div class="popover" role="dialog" aria-hidden="true">\n    <div class="arrow"></div>\n    <div class="btn-close">&#x2716;</div>\n    <div class="popover-content"><%= popoverContent %></div>\n</div>'
    }), define("text!bs/templates/onboard_firstdigg.html", [], function() {
        return '<div id="first-digg" class="first-digg">\n<% if (facebook && facebook.username && fb_enable_og) { %>\n    <div class="facebook">\n        <h5 class="hed">Success!</h5>\n        <p>Your diggs will appear in your Facebook timeline.<br/>\n           <a href="/settings">Go to Settings</a> to change this preference.</p>\n    </div>\n<% } else { %>\n    <div class="twitter">\n        <h5 class="hed">You dugg this story!</h5>\n        <p>Thanks for that.</p>\n    </div>\n<% } %>\n</div>\n'
    }), define("text!bs/templates/onboard_firstsave.html", [], function() {
        return '<h5 class="hed">Saved!</h5>\n<p>Saved stories are also synced to your<br/>iPhone, iPad, and/or Android device.</p>\n'
    }), define("text!bs/templates/modal_dailydigg.html", [], function() {
        return '<div id="modal-dailydiggsub" class="modal-dailydiggsub modal fade hide" role="dialog" aria-hidden="true">\n    <header class="modal-header">\n        <span class="modal-header-text">Subscribe to <b>The Daily Digg</b></span>\n        <div id="btn-savedstories-modal-close" class="btn-modal-close">✖</div>\n    </header>\n    <section class="modal-body">\n    \n        <p class="pitch">Welcome to Digg! Every day we send an email with a handful of our <b>top stories</b> and <b>videos</b> and none of the clutter, noise, or cats that usually distract you.</p>\n    <% if (campaignVersion < 4) { %>\n        <div class="kudos">\n        <% if (campaignVersion == 0) { %>\n            <blockquote class="tweet" cite="https://twitter.com/stevekovach/status/408209270900260864"><p>The @digg newsletter is the best email newsletter I get. It\'s also the only one.</p><footer><b>Steve Kovach</b>&nbsp;<a href="https://twitter.com/stevekovach/status/408209270900260864" target="_blank">@stevekovach&nbsp;&nbsp;Dec 4, 2013, 7:20am</a></footer></blockquote>\n        \n        <% } else if (campaignVersion == 1) { %>\n            <blockquote class="tweet" cite="https://twitter.com/mccarthyryanj/status/345188184633769984"><p>You should really sign up for the @digg email newsletter. It\'s great.</p><footer><b>Ryan McCarthy</b>&nbsp;<a href="https://twitter.com/mccarthyryanj/status/345188184633769984" target="_blank">@mccarthyryanj&nbsp;&nbsp;June 13, 2013, 7:37am</a></footer></blockquote>\n        \n        <% } else if (campaignVersion == 2) { %>\n            <blockquote class="tweet" cite="https://twitter.com/travors/status/393001208841252864"><p>The @digg newsletter is really good, in fact it\'s the only newsletter that always engages me.</p><footer><b>Dan Walsh</b>&nbsp;<a href="https://twitter.com/travors/status/393001208841252864" target="_blank">@travors&nbsp;&nbsp;Oct 13, 2013, 6:09am</a></footer></blockquote>\n            \n        <% } else if (campaignVersion == 3) { %>\n            <blockquote class="tweet" cite="https://twitter.com/BrianMueller333/status/350593250521919488"><p>Never thought I\'d look forward to receiving a daily newsletter from Digg, but here we are.</p><footer><b>Brian Mueller</b>&nbsp;<a href="https://twitter.com/BrianMueller333/status/350593250521919488" target="_blank">@BrianMueller333&nbsp;&nbsp;June 28, 2013, 5:35am</a></footer></blockquote>\n        \n        <% } %>\n        </div>\n    <% } %>\n        <form class="form-dailydigg-sub">\n            <label class="form-dailydigg-lbl">\n                <span class="lbl">Enter your email address</span>\n                <input class="form-dailydigg-input" type="email" placeholder="Enter your email address" value="<%= inputValue %>" required />\n            </label>\n            <button class="form-dailydigg-submit" type="submit">Subscribe</button>\n            <p class="calltoaction"><% if (email) { %>Hit the subscribe button<% } else { %>Enter your email, hit subscribe <% } %> and pat yourself on the back for making a great decision.</p>\n        </form>\n    </section>'
    }), define("bootstrap_modal", [], function() {}), define("bs/widgets/onboarding", ["require", "bs/core", "bs/util/cookies", "bs/util/xhr", "bs/util/format", "bs/util/helpers", "bs/widgets/dailydiggsub", "text!bs/templates/popover.html", "text!bs/templates/onboard_firstdigg.html", "text!bs/templates/onboard_firstsave.html", "text!bs/templates/modal_dailydigg.html", "bootstrap_modal", "lodash"], function(e, t, n, r, i, s, o, u, a, f, l, c, h) {
        function E(e, t) {
            return Math.floor(Math.random() * (t - e + 1)) + e
        }
        function S(e) {
            var t = $(e.target).closest(".modal"), n = $(e.target).hasClass("btn-modal-close"), r = $(e.target).hasClass("form-dailydigg-cancel"), i = $(e.target).is(".form-dailydigg-sub.success");
            if (t.length === 0 || n || r || i)
                i ? setTimeout(function() {
                    g.modal("hide").remove()
                }, 1500) : g.modal("hide").remove(), C.optOut(), $(document).off("click touchend", S), window.location.hash = "_"
        }
        function x(t) {
            var n = typeof t != "undefined" ? $(t.target).closest(".popover"): $("body").closest(".popover"), r = typeof t != "undefined" ? $(t.target).hasClass("btn-close"): !1, i = typeof t != "undefined" ? $(t.target).hasClass("modal"): !1, s = typeof t != "undefined" ? $(t.target).hasClass("modal-backdrop"): !1;
            if (i || s)
                return;
            if (n.length === 0 || r)
                $(y).removeClass("show-story-actions"), $(".popover:not(.nav-user .popover)").remove(), $(document).off("click touchend", x), e("bs/core").pubsub.off("story_actions", "sitenav_auth_displayed", x), e("bs/core").pubsub.off("story_actions", "savedstories_modal_displayed", x)
        }
        function T(t) {
            S(t), e("bs/core").pubsub.fire("dailydigg_tweet_campaign_success", w[b]), e("bs/core").pubsub.off("modal_dailydigg", "dailydigg_subscribe_success", T)
        }
        function N() {
            window.location.hash = "_", g.off("hidden.bs.modal")
        }
        var p = h.template(u), d = h.template(a), v = h.template(f), m = h.template(l), g, y = ".story-container", b, w = ["@stevekovach", "@mccarthyryanj", "@travors", "@BrianMueller333", "noTweet"], C = {
            dailyDiggModal: function(t) {
                var n = e("bs/core").user.isLoggedIn() ? e("bs/core").user.get("email"): !1, r = {
                    email: n ? n: !1,
                    inputValue: n ? n: "",
                    campaignVersion: E(0, 5)
                }, i = m(r);
                return t && t.preventDefault && t.preventDefault(), g = $(i).appendTo("body").modal({
                    keyboard: !0,
                    show: !0
                }), g.on("hidden.bs.modal", N), b = r.campaignVersion, o.init(), setTimeout(function() {
                    $(".form-dailydigg-input").focus()
                }, 10), $(document).on("click touchend", S), g.on("click", ".btn-modal-close", S), e("bs/core").pubsub.fire("dailydigg_tweet_campaign_view", w[b]), e("bs/core").pubsub.on("modal_dailydigg", "dailydigg_subscribe_success", T), !1
            },
            dailyDiggModalHash: function(t) {
                var n = C.dailyDiggModal, r;
                window.location.hash === "#dailydigg" && (r = e("bs/core"), r.user.isLoggedIn() ? C.dailyDiggModal(t) : (r.pubsub.one("daily_digg_modal", r.pubsub.channels.user_auth, n), $(function() {
                    C.dailyDiggModal(t)
                })))
            },
            firstDigg: function(t) {
                var n = e("bs/core").user.getUser(), r = $("<div></div>"), i = p({
                    popoverContent: d(n)
                });
                x(), r.replaceWith(i).appendTo(t.parents(".story-action")).fadeIn(160).parents(y).addClass("show-story-actions"), $(document).on("click touchend", x), e("bs/core").pubsub.on("story_actions", "sitenav_auth_displayed", x), e("bs/core").pubsub.on("story_actions", "savedstories_modal_displayed", x)
            },
            firstSave: function(t) {
                var n = $("<div></div>"), r = p({
                    popoverContent: v()
                });
                x(), n.replaceWith(r).appendTo("#nav-saved").fadeIn(160), t.parents(y).addClass("show-story-actions"), $(document).on("click touchend", x), e("bs/core").pubsub.on("story_actions", "sitenav_auth_displayed", x), e("bs/core").pubsub.on("story_actions", "savedstories_modal_displayed", x), e("bs/core").pubsub.fire("onboarding_firstsave")
            },
            optOut: function() {
                n.set({
                    name: "digg_onboarding_dailydiggsub",
                    value: 1,
                    expires: n.expires.oneYear()
                })
            }
        };
        return C
    }), define("text!bs/templates/storyaction_auth.html", [], function() {
        return '<div class="arrow"></div>\n<div class="btn-close">&#x2716;</div>\n<div class="popover-content">\n    <strong>Sign in</strong>\n    <p>Please sign in to <span id="auth-story-action"><%= action %></span> a story.</p>\n    \n    <a id="btn-facebook-auth" class="btn-service-auth btn-facebook-auth" href="#fb"><div class="auth-ico"></div>Sign in</a>\n    <a id="btn-twitter-auth" class="btn-service-auth btn-twitter-auth" href="#twitter"><div class="auth-ico"></div>Sign in</a>\n    <a id="btn-google-auth" class="btn-service-auth btn-google-auth" href="#google"><div class="auth-ico"></div>Sign in</a>\n</div>'
    }), define("bs/util/user", ["require", "bs/core", "bs/util/cookies", "bs/util/validation", "bs/util/xhr", "bs/util/helpers", "bs/widgets/onboarding", "text!bs/templates/storyaction_auth.html", "lodash"], function(e, t, n, r, i, s, o, u, a) {
        function g() {
            p || (p=!0, FB.init({
                appId: e("bs/core").config.FB_APP_ID,
                status: !0,
                cookie: !0,
                oauth: !0,
                xfbml: !0,
                version: "v2.0"
            }))
        }
        function y(e, t) {
            var n = t ? {
                url: "/session",
                type: "POST",
                data: {
                    action: "refresh"
                }
            }
            : {};
            c || (c=!0, v.request(n).success(function(t) {
                c=!1, e && typeof e == "function" && e(f)
            }).error(function(e) {
                c=!1
            }))
        }
        function b(e) {
            var t = typeof e != "undefined" ? $(e.target).closest(".popover"): $("body").closest(".popover"), n = typeof e != "undefined" ? $(e.target).hasClass("btn-close"): !1;
            if (t.length === 0 || n)
                h.$loginEl.fadeOut(160), $(document).off("click", b), $(".story-container").removeClass("show-story-actions")
        }
        function w() {
            e("bs/core").pubsub.one("site_nav_user_widget", e("bs/core").pubsub.channels.user_auth, function(e) {
                var t = (new Date).valueOf() / 1e3-~~h.get("date_signup") < 900, r = n.get("digg_onboarding_dailydiggsub"), i = document.location.hash;
                (i.match("test-onboarding") || t && r !== "1"&&!h.get("email_subscriber")) && o.dailyDiggModal()
            }), $(document).off("click", ".btn-facebook-auth", _).off("click", ".btn-twitter-auth", D).off("click", ".btn-google-auth", P), E()
        }
        function E() {
            var e=!!n.get("frontend.user");
            return e&&!f && y(), e
        }
        function S(t, n) {
            var r = s.createGlobalCallback(function(r) {
                t && typeof t == "function" ? t = s.createGlobalCallback(t) : n !== "link" && n !== "login" && (t || (t = s.createGlobalCallback(w)), n === "digg_deeper" && o.optOut()), O(r, t), e("bs/core").pubsub.fire("twitter_auth_success", n)
            });
            return s.popWin("/session/twitter?cb=" + r, "twitterSignin", {
                width: 590,
                height: 460
            }), !1
        }
        function x(t, n) {
            var r = s.createGlobalCallback(function(r) {
                t && typeof t == "function" ? t = s.createGlobalCallback(t) : n !== "link" && n !== "login" && (t || (t = s.createGlobalCallback(w))), T(r, t), e("bs/core").pubsub.fire("google_auth_success", n)
            });
            return s.popWin("/session/google?cb=" + r, "googleSignin", {
                width: 590,
                height: 560
            }), !1
        }
        function T(e, t) {
            e.oauth_refresh_token && (document.getElementById("oauth_refresh_token").value = e.oauth_refresh_token), document.getElementById("oauth_access_token").value = e.oauth_access_token, document.getElementById("google_uid").value = e.google_uid, document.getElementById("google_name").value = e.google_name, document.getElementById("google_email").value = e.google_email, document.getElementById("google_expiration_date").value = e.expiration_date, t && (document.getElementById("google_callback").value = t), document.getElementById("session_google").submit()
        }
        function N(t, n) {
            var r = s.createGlobalCallback(function(r) {
                t && typeof t == "function" && (t = s.createGlobalCallback(t)), C(r, t), e("bs/core").pubsub.fire("pocket_auth_success", n)
            });
            return s.popWin("/session/pocket?cb=" + r, "pocketAuth", {
                width: 960,
                height: 640
            }), !1
        }
        function C(e, t) {
            document.getElementById("pocket_token").value = e.pocket_token, document.getElementById("pocket_user_name").value = e.pocket_user_name, t && (document.getElementById("pocket_callback").value = t), document.getElementById("session_pocket").submit()
        }
        function k(t, n) {
            var r = s.createGlobalCallback(function(r) {
                t && typeof t == "function" && (t = s.createGlobalCallback(t)), L(r, t), e("bs/core").pubsub.fire("readability_auth_success", n)
            });
            return s.popWin("/session/readability?cb=" + r, "readabilityAuth", {
                width: 960,
                height: 640
            }), !1
        }
        function L(e, t) {
            document.getElementById("readability_token").value = e.readability_token, document.getElementById("readability_secret").value = e.readability_secret, document.getElementById("readability_username").value = e.readability_username, t && (document.getElementById("readability_callback").value = t), document.getElementById("session_readability").submit()
        }
        function A(e, t) {
            document.getElementById("instapaper_token").value = e.token, document.getElementById("instapaper_secret").value = e.secret, document.getElementById("instapaper_username").value = e.username, t && (document.getElementById("instapaper_callback").value = t), document.getElementById("session_instapaper").submit()
        }
        function O(e, t) {
            document.getElementById("twitter_key").value = e.key, document.getElementById("twitter_secret").value = e.secret, document.getElementById("twitter_screen_name").value = e.screen_name, document.getElementById("twitter_user_id").value = e.user_id, document.getElementById("twitter_callback").value = t, document.getElementById("session_twitter").submit()
        }
        function M(e, t) {
            var n = e.authResponse;
            document.getElementById("fb_access_token").value = n.accessToken, document.getElementById("fb_expires").value = n.expiresIn, t != null && (document.getElementById("facebook_callback").value = t), document.getElementById("session_facebook").submit()
        }
        function _(t) {
            var n = "Sign In";
            return t && (t.stopPropagation(), t.preventDefault()), e("bs/core").pubsub.fire("fb_auth_click", n), g(), FB.getLoginStatus(function(t) {
                t.authResponse ? (h.loginFacebook(t, s.createGlobalCallback(w)), e("bs/core").pubsub.fire("fb_auth_success", n)) : FB.login(function(t) {
                    t.authResponse && (h.loginFacebook(t, s.createGlobalCallback(w)), e("bs/core").pubsub.fire("fb_auth_success", n))
                }, {
                    scope: "email,publish_actions,publish_stream"
                })
            }), !1
        }
        function D(t) {
            var n = "Sign In";
            return t && (t.stopPropagation(), t.preventDefault()), e("bs/core").pubsub.fire("twitter_auth_click", n), h.initTwitterLogin(null, n), !1
        }
        function P(t) {
            var n = "Sign In";
            return t && (t.stopPropagation(), t.preventDefault()), e("bs/core").pubsub.fire("google_auth_click", n), h.initGoogleLogin(null, n), !1
        }
        var f = null, l = null, c=!1, h, p=!1, d = function(t) {
            c=!1, f = t.user, e("bs/core").pubsub.fire(e("bs/core").pubsub.channels.user_auth, f)
        }, v = new i.dataProvider({
            url: "/api/user.json",
            cache: !1,
            success: d
        }), m = a.template(u);
        return h = {
            loginEl: null,
            isLoggedIn: E,
            get: function(e) {
                return f && e in f ? f[e] : null
            },
            deepGet: function(e) {
                if (!f)
                    return null;
                var t = f, n=!0;
                return a.each(e.split("."), function(e) {
                    if (!(e in t))
                        return n=!1, !1;
                    t = t[e]
                }), n ? t : null
            },
            refresh: function(e) {
                E() && y(e)
            },
            getUser: function(t) {
                if (t && typeof t == "function") {
                    E() ? f ? t(f) : e("bs/core").pubsub.one("bs.user", e("bs/core").pubsub.channels.user_auth, t) : t(f);
                    return 
                }
                return f
            },
            updateAccessToken: function(e) {
                l = e
            },
            initFacebookLogin: function() {
                g(), h.isLoggedIn() && FB.getLoginStatus(function(e) {
                    var t;
                    e.authResponse && (t = new i.dataProvider({
                        url: "/api/user/token/update.json",
                        type: "POST",
                        data: {
                            access_token: e.authResponse.accessToken
                        }
                    }), h.updateAccessToken(e.authResponse.accessToken), t.request())
                })
            },
            loginFacebook: M,
            initTwitterLogin: S,
            initGoogleLogin: x,
            initPocketAuth: N,
            initReadabilityAuth: k,
            authenticateInstapaper: A,
            authorized: function(t, n, r) {
                return n = n || "Sign In", r = r || {}, r.storyActionEl = r.storyActionEl || ".story-action", r.twitterOnly ? function() {
                    var i = function(e) {
                        f.twitter && f.twitter.screen_name ? e.call(this, f) : s.call(this, e)
                    }, s = function(t) {
                        E() ? h.initTwitterLogin(a.bind(t, this), n) : (e("bs/core").pubsub.one("user", e("bs/core").pubsub.channels.user_auth, a.bind(function() {
                            y(a.bind(t, this))
                        }, this)), h.initTwitterLogin(null, n))
                    };
                    r.async && $("#session_twitter").attr("target", "session_iframe"), E() ? f ? i(t) : y(a.bind(i, this, t)) : s.call(this, t)
                } : (h.$loginEl ? h.$loginEl.empty().detach() : h.$loginEl = $('<div id="story-action-auth" class="popover story-action-auth" style="display:none"></div>'), function() {
                    var i = $(this), s = i.closest(r.storyActionEl), o = s.closest(".story-container");
                    return E() ? f ? t.call(this, f) : y(a.bind(t, this)) : (e("bs/core").pubsub.fire("signin_tooltip_displayed", n), e("bs/core").pubsub.fire("auth_tooltip_displayed"), e("bs/core").pubsub.one("savedstories", e("bs/core").pubsub.channels.user_auth, a.bind(function() {
                        $(document).off("click", b), h.$loginEl.fadeOut(160), y(a.bind(t, this))
                    }, this)), h.$loginEl.html(m({
                        action: n
                    })).appendTo(s).fadeIn(160), o.addClass("show-story-actions"), $(document).on("click", b), e("bs/core").pubsub.on("storyaction", "sitenav_auth_displayed", b)), !1
                })
            }
        }, window.fbAsyncInit = h.initFacebookLogin, h.isLoggedIn() || $(document).on("click", ".btn-facebook-auth", _).on("click", ".btn-twitter-auth", D).on("click", ".btn-google-auth", P), $(function() {
            $("#session-forms").load("/session/forms")
        }), h
    }), define("bs/util/media_queries_monitor", ["lodash"], function(e) {
        function t(t) {
            this.query = t.query, this.listeners = [], window.matchMedia ? (this.mql = window.matchMedia(this.query), this.mql.addListener(e.bind(this.checkMatches, this))) : this.mql = {
                matches: !1
            }
        }
        function n(e) {
            this.mediaQueries = {}
        }
        return t.prototype = {
            addListener: function(e) {
                this.listeners.push(e)
            },
            removeListener: function(t) {
                this.listeners = e.reject(this.listeners, function(e) {
                    return e === t
                })
            },
            checkMatches: function(t) {
                var n = t || this.mql;
                n.matches && e.each(this.listeners, function(e) {
                    e()
                })
            }
        }, n.prototype = {
            addQuery: function(n, r) {
                this.mediaQueries[n] || (this.mediaQueries[n] = new t({
                    query: n
                })), r && typeof r == "function" && this.mediaQueries[n].addListener(r), e.invoke(this.mediaQueries, "checkMatches")
            },
            removeQuery: function(e, t) {
                if (!this.mediaQueries[e])
                    throw new Error('"' + e + '" is not a currently monitored media query');
                this.mediaQueries[e].removeListener(t)
            }
        }, {
            factory: function(e) {
                return new n(e)
            }
        }
    }), define("modernizr", [], function() {
        return Modernizr
    }), define("bs/core", ["bs/util/pubsub", "bs/util/user", "bs/util/media_queries_monitor", "jquery", "lodash", "modernizr"], function(e, t, n, r, i, s) {
        var o = window.BS || {};
        return o.extend = r.extend, o.pubsub || o.extend(o, {
            pubsub: e
        }), o.user || (o.extend(o, {
            user: t
        }), window.fbAsyncInit || (window.fbAsyncInit = require("bs/user").initFacebookLogin)), o.mediaQueriesMonitor || o.extend(o, {
            mediaQueriesMonitor: n.factory()
        }), s.addTest("ipad", function() {
            return !!navigator.userAgent.match(/iPad/i)
        }), s.addTest("iphone", function() {
            return !!navigator.userAgent.match(/iPhone/i)
        }), s.addTest("ipod", function() {
            return !!navigator.userAgent.match(/iPod/i)
        }), s.addTest("appleios", function() {
            return s.ipad || s.ipod || s.iphone
        }), s.ipad && (o.support = {
            ipad_version_ready: new r.Deferred
        }, window.ondevicemotion = function(e) {
            if (navigator.platform.indexOf("iPad")!=-1) {
                var t = 1;
                e.acceleration && (t += window.devicePixelRatio), document.documentElement.className += " ipad-v" + t, o.support.ipad_version_ready.resolve()
            }
            window.ondevicemotion = null
        }), s.addTest("positionfixed", function() {
            var e = document.createElement("div"), t = e.cloneNode(!1), n=!1, r = document.body || function() {
                return n=!0, document.documentElement.appendChild(document.createElement("body"))
            }(), i = r.style.cssText;
            r.style.cssText = "padding:0;margin:0", e.style.cssText = "position:fixed;top:42px", r.appendChild(e), r.appendChild(t);
            var o = e.offsetTop !== t.offsetTop;
            return r.removeChild(e), r.removeChild(t), r.style.cssText = i, n && document.documentElement.removeChild(r), navigator.userAgent.match(/OS (\d)/i), o && (!s.appleios || s.appleios && RegExp.$1 >= 5)
        }), s.addTest("lastchild", function() {
            var e, t = ["#modernizr-last-child li{display:block;width:100px;height:100px;}", "#modernizr-last-child li:last-child{width:200px;}"], n = document.getElementsByTagName("head")[0] || function() {
                return document.documentElement.appendChild(document.createElement("head"))
            }(), r = document.body || function() {
                return document.documentElement.appendChild(document.createElement("body"))
            }(), i = document.createElement("ul"), s = document.createElement("li"), o = document.createElement("li"), u = document.createElement("style");
            return u.type = "text/css", u.styleSheet ? u.styleSheet.cssText = t.join("") : u.appendChild(document.createTextNode(t.join(""))), n.appendChild(u), i.id = "modernizr-last-child", i.appendChild(s), i.appendChild(o), r.appendChild(i), e = o.offsetWidth > s.offsetWidth, n.removeChild(u), r.removeChild(i), e
        }), function(e, t) {
            t.support.touch = "ontouchstart"in e
        }(this, this.jQuery), r.fn.k = function(e, t) {
            return t === undefined && (t = "38,38,40,40,37,39,37,39,66,65"), this.each(function() {
                var n = [];
                r(this).keydown(function(i) {
                    n.push(i.keyCode), n.toString().indexOf(t) >= 0 && (r(this).unbind("keydown", arguments.callee), e(i))
                })
            })
        }, o
    }), define("bs/util/storyactions2", ["require", "bs/core", "bs/util/cookies", "bs/util/xhr", "bs/util/format", "bs/util/helpers", "bs/widgets/onboarding", "lodash"], function(e, t, n, r, i, s, o, u) {
        function d(e) {
            var t = (new Date).valueOf() + 31536e6;
            $.extend(h, e), n.set({
                name: c,
                value: $.param(h),
                expires: new Date(t)
            })
        }
        function v(e) {
            var n = "http://api.digg.com/api/v3/redirect", r = $(this).closest(a), i = r.data("contentId"), s = t.user.get("user_id"), o = r.data("source"), u = {
                url: this.href
            };
            i && (u.content_id = i), s && (u.user_id = s), o && (u.source = o), navigator.mozApps && window.locationbar.visible===!1 && (u.platform = 1005, window.open([n, $.param(u)].join("?")));
            if (p === "mobile")
                u.platform = 1005, window.location.href = [n, $.param(u)].join("?");
            else {
                if (this.target !== "_blank")
                    return;
                window.open([n, $.param(u)].join("?"))
            }
            return !1
        }
        function m(e) {
            var t = {
                contentID: e.data("contentId"),
                diggUrlID: e.data("urlId"),
                category: e.data("category"),
                diggScore: e.data("diggScore"),
                tweets: e.data("tweets"),
                fbShares: e.data("fbShares"),
                diggs: e.data("diggs"),
                dugg: e.data("dugg"),
                read_later: e.data("read_later")
            };
            return t.diggScore = u.isString(t.diggScore) ? parseInt(t.diggScore.replace(/,/g, ""), 10) : t.diggScore, t.tweets = u.isString(t.tweets) ? parseInt(t.tweets.replace(/,/g, ""), 10) : t.tweets, t.fbShares = u.isString(t.fbShares) ? parseInt(t.fbShares.replace(/,/g, ""), 10) : t.fbShares, t.diggs = u.isString(t.diggs) ? parseInt(t.diggs.replace(/,/g, ""), 10) : t.diggs, t
        }
        function g(e, t) {
            e.data(t)
        }
        function y(e) {
            u.each(e, function(e, t) {
                b(t, e)
            })
        }
        function b(e, t, n) {
            var r = $(".story-" + e), s = $(".diggscore-" + e), o = $(".diggs-" + e), u = $(".tweets-" + e), a = $(".fb_shares-" + e), c = $(".digg-" + e), h = $(".save-" + e), p = s.parent(".story-score"), d = $(".diggs-label-" + e);
            g(r, {
                diggScore: t.digg_score,
                diggs: t.diggs,
                tweets: t.tweets,
                fbShares: t.fb_shares,
                dugg: t.dugg,
                read_later: t.read_later
            });
            if (!s ||!s.size())
                return;
            t.digg_score > 0 && p.hasClass("story-score-zero") ? p.removeClass("story-score-zero") : t.digg_score === 0 && p.addClass("story-score-zero"), d.length > 0 && t.diggs === 0 ? p.addClass("story-score-zero") : d.length > 0 && t.diggs === 1 ? (d.html("digg"), p.removeClass("story-score-zero")) : d.length > 0 && (d.html("diggs"), p.removeClass("story-score-zero")), o.html(i.intWithCommas(t.diggs)), u.html(i.intWithCommas(t.tweets)), a.html(i.intWithCommas(t.fb_shares)), s.html(i.intWithCommas(t.digg_score, !0)), !n && t.hasOwnProperty("dugg") && t.dugg && (c.html("Dugg"), r.addClass(f)), !n && t.hasOwnProperty("read_later") && t.read_later && (h.html("Saved"), r.addClass(l))
        }
        function w(e) {
            var t = $(this), n = t.parents(a), i = m(n), s = {
                content_id: i.contentID,
                digg_url_id: i.diggUrlID
            }, o = i.dugg ? "/api/digg/delete.json": "/api/digg/submit.json", f = new r.dataProvider({
                url: o,
                type: "POST",
                data: s,
                success: u.bind(E, this, t, i.contentID)
            });
            return f.request(), !1
        }
        function E(e, n, r) {
            var i = document.location.hash, s = $('.digg-story-el[data-content-id="' + n + '"]'), u = (new Date).valueOf() / 1e3-~~t.user.get("date_signup") < 900, a = r.data;
            b(n, a, !0), a.dugg ? (t.pubsub.fire("digg_success", e, a, n), s.addClass(f), $(".digg-" + n).html("Dugg")) : (t.pubsub.fire("digg_delete_success", e, a, n), s.removeClass(f), $(".digg-" + n).html("Digg"));
            if (i.match("test-onboarding") || a.dugg&&!h.dugg && u)
                o.firstDigg(e), d({
                    dugg: 1
                })
        }
        function S(e) {
            var t = $(this), n = t.parents(a), i = m(n), s = i.read_later ? "/api/story/readlater/delete.json": "/api/story/readlater/add.json", o = {
                content_id: i.contentID,
                digg_url_id: i.diggUrlID
            }, f = new r.dataProvider({
                url: s,
                type: "POST",
                data: o,
                success: u.bind(x, this, t, i.contentID)
            });
            return f.request(), !1
        }
        function x(e, n, r) {
            var i = document.location.hash, s = $('.digg-story-el[data-content-id="' + n + '"]'), u = (new Date).valueOf() / 1e3-~~t.user.get("date_signup") < 900, a = r.data;
            g(s, {
                read_later: a.read_later
            }), a.read_later ? (s.addClass(l), $(".save-" + n).html("Saved"), t.pubsub.fire("add_reading_list_success", e, a, n)) : (s.removeClass(l), $(".save-" + n).html("Save"), t.pubsub.fire("delete_reading_list_request", e, a, n));
            if (i.match("test-onboarding") || a.read_later&&!h.saved && u)
                o.firstSave(e), d({
                    saved: 1
                })
        }
        function T(e) {
            var n = $(this), r = n.hasClass("share-twitter") ? "twitter": "facebook";
            return e.preventDefault(), t.pubsub.fire("story_share", r, n), r === "twitter"?!1 : (s.popWin(this.href, null, {
                width: 550,
                height: 450
            }), !1)
        }
        function N(e) {
            var t = $(this), n = t.data("shareContentId"), r = t.data("shareId"), i = "share-details-" + n + "-" + r;
            $(".share-details-" + n).hide(), $("#" + i).show(), $(".story-trending-share-img-container-" + n).removeClass("current"), t.parent().addClass("current")
        }
        var a = ".digg-story-el", f = "story-acted-dugg", l = "story-acted-saved", c = "saflags", h = n.get(c), p = n.get("preferred_view") || "desktop";
        return h = h ? s.deparam(h) : {}, {
            submitDigg: w,
            addReadingList: S,
            renderStoryScores: y,
            redirectStoryLink: v,
            shareStory: T,
            showShareDetails: N
        }
    }), define("bs/util/storyscores", ["bs/core", "bs/util/xhr", "bs/util/cookies", "lodash"], function(e, t, n, r) {
        function d(e) {
            this.name = e, this.pollInterval = i, this.storyScores = {}, this.delta = {}, this.currentContentIDs = [], this.dataProvider = new t.dataProvider({
                url: "/api/story/scores.json?" + this.name,
                type: "POST",
                success: r.bind(c, this),
                error: h
            })
        }
        var i = 3e5, s = {}, o = {}, u = [], a = function(e, t, n) {
            if (!s[t] ||!r.isEqual(s[t], e))
                o[t] = e, s[t] = e
        }, f = function(t) {
            o = {}, r.each(t.data, a), e.pubsub.fire(e.pubsub.channels.scores_update, o)
        }, l = function(e, t, n) {
            if (!this.storyScores[t] ||!r.isEqual(this.storyScores[t], e))
                this.delta[t] = e, this.storyScores[t] = e
        }, c = function(t) {
            this.delta = {}, r.each(t.data, r.bind(l, this)), e.pubsub.fire(e.pubsub.channels.scores_update, this.delta)
        }, h = function(t) {
            e.pubsub.fire(e.pubsub.channels.scores_update_error, t)
        }, p = new t.dataProvider({
            url: "/api/story/scores.json",
            type: "POST",
            success: f,
            error: h
        });
        return d.prototype = {
            setPollInterval: function(e) {
                return this.pollInterval = e, this
            },
            register: function(t) {
                return Array.prototype.push.apply(this.currentContentIDs, t), this.currentContentIDs = r.uniq(this.currentContentIDs), this.dataProvider.isPolling() && this.dataProvider.stopPolling(), e.features.polling_scores ? e.user.isLoggedIn() && this.dataProvider.startPolling(this.pollInterval, {
                    data: {
                        content_ids: this.currentContentIDs.join(",")
                    }
                }) : e.user.isLoggedIn() && this.dataProvider.request({
                    data: {
                        content_ids: this.currentContentIDs.join(",")
                    }
                }), this
            },
            deregisterAll: function() {
                return this.currentContentIDs = [], this.storyScores = {}, this
            }
        }, {
            factory: function(e) {
                return new d(e)
            },
            register: function(t) {
                return Array.prototype.push.apply(u, t), u = r.uniq(u), p.isPolling() && p.stopPolling(), e.features.polling_scores && p.startPolling(i, {
                    data: {
                        content_ids: u.join(",")
                    }
                }), s
            },
            deregister: function(e) {
                u = r.difference(u, e)
            }
        }
    }), define("bs/util/tracking", ["bs/core", "jquery", "lodash"], function(e, t, n) {
        function r(e) {
            var t = e.match(/:\/\/(.[^\/]+)/);
            return t ? t[1] : null
        }
        function i(e) {
            var t;
            return e ? (t = e.split("-"), t[0].charAt(0).toUpperCase() + t[0].substring(1)) : ""
        }
        function s(e) {
            e.submit()
        }
        function o(e) {
            var i = e.closest(".stories-container"), s = e.closest(".digg-story-el").size() ? e.closest(".digg-story-el"): e.closest(".js--digg-story"), o = s.data("contenturl") || e.attr("href"), u = r(o), a = s.data("position") || 0, f = ["Tag", "Source", "Search"], l = s.find(".first-tag"), c = l.size() ? t.trim(l.text()): "Uncategorized", h = {
                section: i.data("tracking-section"),
                position: a,
                url: o,
                domain: u,
                primary_tag: c
            };
            return n.indexOf(f, h.category)>-1 && (h.is_archived=!0, h.label = i.data("tracking-label")), h
        }
        function u() {
            this.debug=!1
        }
        var a = function() {
            var e = this;
            t(document).on("click", ".story-link", function(n) {
                var r = t(this), i = "Story Click", s = o(r), u = [i, s.section, s.label || s.primary_tag, s.position];
                return e.trackEvent.apply(e, u)
            })
        }, f = function() {
            var n = this;
            e.pubsub.on("tracking", "digg_success", function(e) {
                var r = "Digg", i = o(t(e)), s = [r, i.section, i.label || i.primary_tag, i.position];
                return n.trackSocial.apply(n, ["digg", "digg", i.url]), n.trackEvent.apply(n, s)
            })
        }, l = function() {
            var n = this;
            e.pubsub.on("tracking", "add_reading_list_success", function(e) {
                var r = "Saved Story", i = o(t(e)), s = [r, i.section, i.label || i.primary_tag, i.position];
                return n.trackSocial.apply(n, ["digg", "save", i.URL]), n.trackEvent.apply(n, s)
            })
        }, c = function() {
            var t = this, n = "Daily Digg Email Submit";
            e.pubsub.on("tracking", "daily_digg_email_submit_start", function(e) {
                return t.trackEvent(n, "Email Address entered")
            }), e.pubsub.on("tracking", "daily_digg_email_submit_success", function(e) {
                return t.trackEvent(n, "Email Address submission complete")
            })
        }, h = function() {
            var n = this;
            e.pubsub.on("tracking", "story_share", function(e, t) {
                var r = e === "twitter" ? "Social: Twitter": "Social: Facebook", i = o(t), s = [r, "Share", i.section];
                return n.trackSocial.apply(n, [e, "share", i.url]), n.trackEvent.apply(n, s)
            }), t(document).on("click", ".btn-tweet-reply", function(e) {
                var r = t(this), i = "Social: Twitter", s = o(r);
                return n.trackSocial.apply(n, ["twitter", "reply", s.url]), n.trackEvent(i, "Reply", s.section)
            }).on("click", ".btn-tweet-retweet", function(e) {
                var r = t(this), i = "Social: Twitter", s = o(r);
                return n.trackSocial.apply(n, ["twitter", "retweet", s.url]), n.trackEvent(i, "Retweet", s.section)
            }).on("click", ".btn-tweet-favorite", function(e) {
                var r = t(this), i = "Social: Twitter", s = o(r);
                return n.trackSocial.apply(n, ["twitter", "favorite", s.url]), n.trackEvent(i, "Favorite", s.section, s.URL)
            })
        }, p = function() {
            var t = this;
            e.pubsub.on("tracking", "signin_tooltip_displayed", function(e) {
                var n = "Sign in attempt", r = "on " + e;
                return t.trackEvent(n, r)
            }), e.pubsub.on("tracking", "fb_auth_click", function(e) {
                var n = "Click Facebook Auth", r = "on " + e;
                return t.trackEvent(n, r)
            }), e.pubsub.on("tracking", "fb_auth_success", function(e) {
                var n = "Facebook Auth Complete", r = "on " + e;
                return t.trackEvent(n, r)
            }), e.pubsub.on("tracking", "twitter_auth_click", function(e) {
                var n = "Click Twitter Auth", r = "on " + e;
                return t.trackEvent(n, r)
            }), e.pubsub.on("tracking", "twitter_auth_success", function(e) {
                var n = "Twitter Auth Complete", r = "on " + e;
                return t.trackEvent(n, r)
            }), e.pubsub.on("tracking", "google_auth_click", function(e) {
                var n = "Click Google Auth", r = "on " + e;
                return t.trackEvent(n, r)
            }), e.pubsub.on("tracking", "google_auth_success", function(e) {
                var n = "Google Auth Complete", r = "on " + e;
                return t.trackEvent(n, r)
            }), e.pubsub.on("tracking", "dailydigg_signup", function(e) {
                var n = "Email Signup", r = "on " + e.source_action, i = "via " + e.network;
                return t.trackEvent(n, r, i)
            })
        }, d = function() {
            e.pubsub.on("tracking", "dailydigg_tweet_campaign_view", function(e) {
                return N.trackEvent("Daily Digg Signup Modal", "Tweet Campaign View", e)
            }), e.pubsub.on("tracking", "dailydigg_tweet_campaign_success", function(e) {
                return N.trackEvent("Daily Digg Signup Modal", "Tweet Campaign Success", e)
            })
        }, v = function() {
            var e = this;
            t(document).on("click", ".ftr-ios-link", function(t) {
                return e.trackEvent("Click Digg for iPhone", "Footer")
            })
        }, m = function() {
            var n = this;
            t(document).on("click", "#btn-hdr-submit-link", function(e) {
                return n.trackEvent("Submit a Link", "Footer")
            }), e.pubsub.on("tracking", "submit_link_pasted", function() {
                return n.trackEvent("Submit a Link", "Paste Link")
            }), e.pubsub.on("tracking", "submit_link_fb_auth", function() {
                return n.trackEvent("Submit a Link", "Auth with Facebook")
            })
        }, g = function() {
            var t = this;
            e.pubsub.on("tracking", "fb_enable_og_update", function(e) {
                var n = e ? "on": "off";
                return t.trackEvent("Toggle Timeline Sharing", "Turn " + n)
            })
        }, y = function() {
            var t = this;
            e.pubsub.on("tracking", "homepage_section_view", function(e) {
                var n = i(e);
                return t.trackEvent("Homepage Section View", n)
            })
        }, b = function() {
            var t = this;
            e.pubsub.on("tracking", "spreadgun", function(e) {
                return t.trackEvent("Konami!", e)
            })
        }, w = function() {
            function r(t, r) {
                var i = {
                    twitter: "Twitter",
                    fb: "Facebook",
                    gplus: "G+"
                };
                return e.trackEvent(n, t, i[r])
            }
            function i(e) {
                var t = /digg-social-(twitter|fb|gplus)/, n = t.exec(this.className)[1];
                return r("Header", n)
            }
            function s(e) {
                var n = t(this).data("type");
                return r("Footer", n)
            }
            var e = this, n = "Click on Social Button";
            t(document).on("click", ".digg-social-icon", i).on("click", ".footer-social-link", s)
        }, E = function() {
            var e = this;
            t(function() {
                t("#form-digg-search").on("submit", function(r) {
                    var i = t(this).find("input[name=q]").val();
                    n.delay(s, 400, this), e.trackEvent("Search", "Header", i)
                }), t("#search-page-form").on("submit", function(r) {
                    var i = t(this).find("input[name=q]").val();
                    n.delay(s, 400, this), e.trackEvent("Search", "Search Page", i)
                })
            })
        }, S = function() {
            var e = this;
            t(document).on("click", ".tag-list-item-link", function(n) {
                n.preventDefault(), setTimeout(function() {
                    window.location.href = n.target.href
                }, 400);
                var r = t("article"), i = o(r);
                return e.trackEvent("Tag Navigation Click", "From " + i.Section, "To " + n.target.text)
            }).on("click", ".story-tag-link", function(n) {
                n.preventDefault(), setTimeout(function() {
                    window.location.href = n.target.href
                }, 400);
                var r = t(this), i = t("body").hasClass("homepage");
                return i ? e.trackEvent("Archive Click", "Tag Click", r.attr("href")) : e.trackEvent("Pivot Click", "Source Click", r.attr("href"))
            }).on("click", ".story-source-link", function(n) {
                n.preventDefault(), setTimeout(function() {
                    window.location.href = n.target.href
                }, 400);
                var r = t(this), i = t("body").hasClass("homepage");
                return i ? e.trackEvent("Archive Click", "Source Click", r.attr("href")) : e.trackEvent("Pivot Click", "Source Click", r.attr("href"))
            })
        }, x = function() {
            var e = this, n;
            t(document).on("click", ".btn-more-stories", function(r) {
                var i = t(this), s = i.closest("[data-tracking-category]"), o = s.data("tracking-category"), u = s.data("tracking-action"), a=~~i.data("incrementor");
                return a += 1, i.data("incrementor", a), n = "Page " + a.toString(), e.trackEvent("Archive Pagination Depth", n, o + " " + u)
            })
        }, T = function() {
            var t = this, n = "Digg Deeper";
            e.pubsub.on("tracking", "digg_deeper:onboarding:start", function() {
                var e = [n, "Onboarding start", window.location.href];
                return t.trackEvent.apply(t, e)
            }), e.pubsub.on("tracking", "digg_deeper:onboarding:opt_in", function() {
                var e = [n, "Twitter alerts opt in", window.location.href];
                return t.trackEvent.apply(t, e)
            }), e.pubsub.on("tracking", "digg_deeper:onboarding:email_opt_in", function() {
                var e = [n, "Email alerts opt in", window.location.href];
                return t.trackEvent.apply(t, e)
            }), e.pubsub.on("tracking", "digg_deeper:onboarding:opt_out", function(e) {
                var r = [n, e + " opt out", window.location.href];
                return t.trackEvent.apply(t, r)
            }), e.pubsub.on("tracking", "digg_deeper:onboarding:success", function() {
                var e = [n, "Onboarding success", window.location.href];
                return t.trackEvent.apply(t, e)
            }), e.pubsub.on("tracking", "digg_deeper:stream:sortby_selected", function(e) {
                var r = [n, "Sortby selected", e.charAt(0).toUpperCase() + e.slice(1)];
                return t.trackEvent.apply(t, r)
            }), e.pubsub.on("tracking", "digg_deeper:stream:loaded", function(e, r) {
                var i = [n, "Stream page loaded", "Sort by: " + e.charAt(0).toUpperCase() + e.slice(1), r];
                return t.trackEvent.apply(t, i)
            }), e.pubsub.on("tracking", "digg_deeper:stream:sharer_click", function(e) {
                var r = [n, "Sharer click", e];
                return t.trackEvent.apply(t, r)
            }), e.pubsub.on("tracking", "digg_deeper:stream:settings_click", function(e) {
                var r = [n, "Settings click", e];
                return t.trackEvent.apply(t, r)
            })
        };
        u.prototype = {
            initialize: function() {
                a.call(this), f.call(this), l.call(this), h.call(this), w.call(this), y.call(this), p.call(this), v.call(this), g.call(this), m.call(this), b.call(this), c.call(this), S.call(this), x.call(this), E.call(this), d.call(this), T.call(this)
            },
            trackEvent: function() {
                var e = Array.prototype.slice.call(arguments).reverse();
                return e.push("event", "send"), window.ga.apply(null, e.reverse()), this.debug && console.log("tracking", e), !0
            },
            trackSocial: function() {
                var e = Array.prototype.slice.call(arguments).reverse();
                return e.push("social", "send"), window.ga.apply(null, e.reverse()), this.debug && console.log("tracking social", e), !0
            }
        };
        var N = new u;
        t(n.bind(N.initialize, N))
    }), define("bs/util/scrollmonitor", ["bs/core", "jquery", "lodash"], function(e, t, n) {
        function i(e, r) {
            var i = 20, s = 1e3 / i, o = n.bind(this.monitor, this);
            typeof e == "string" && (e = document.getElementById(e)), r || (r = {}), this._el = e, this._lastScrollTop = 0, this.direction = null, this.name = r.name || "scrollMonitor", r.pubsub && (this.pubsub = r.pubsub), this.updateScrollHeight(), this._monitor = n.throttle(o, s), t(this._el).on("scroll", this._monitor)
        }
        var r = function() {
            var r = 20;
            this.$win = t(window), this.$body = t(document.body), this.winHeight = this.$win.height(), this.scrollHeight = this.$body.height(), this.$win.scroll(n.throttle(n.bind(this.monitor, this), 1e3 / r))
        };
        return r.prototype = {
            updateScrollHeight: function() {
                this.scrollHeight = this.$body.height()
            },
            updateWinHeight: function() {
                this.winHeight = this.$win.height()
            },
            monitor: function() {
                var t = 30;
                e.pubsub.fire(e.pubsub.channels.scrolltop, this.$win.scrollTop()), this.winHeight + document.body.scrollTop + t >= this.scrollHeight && e.pubsub.fire("window_scroll_monitor:bottom")
            }
        }, e.scrollMonitor = new r, n.bindAll(e.scrollMonitor), i.prototype = {
            data: function() {
                var e;
                return this.direction = this._el.scrollTop - this._lastScrollTop < 0 ? "up" : "down", e = {
                    el: this._el,
                    scrollTop: this._el.scrollTop,
                    scrollHeight: this._el.scrollHeight,
                    direction: this.direction
                }, e.scrollBottom = e.scrollTop + e.scrollHeight, e
            },
            isBottom: function() {
                var e = 100;
                return this._el.offsetHeight + this._el.scrollTop + e >= this._el.scrollHeight
            },
            checkScrollDepth: function(e) {
                return this._el.offsetHeight + this._el.scrollTop >= this._el.scrollHeight * e
            },
            updateScrollHeight: function() {
                this.scrollHeight = this._el.scrollHeight
            },
            monitor: function() {
                this.pubsub && (this.pubsub.fire(this.name + ":scroll", this.data()), this.checkScrollDepth(.5) && this.pubsub.fire(this.name + ":scroll:50%", this.data()), this.isBottom() && this.pubsub.fire(this.name + ":scroll:bottom", this.data())), this._lastScrollTop = this._el.scrollTop
            },
            destroy: function() {
                t(this._el).off("scroll"), this._el = null
            }
        }, {
            factory: function(e, t) {
                return new i(e, t)
            }
        }
    }), define("bs/widgets/lazyimages", ["jquery", "bs/core", "bs/util/scrollmonitor"], function(e, t, n) {
        function u(t) {
            var n = e(t), r = e(t).data("src");
            r = r !== o ? r : o, n.attr("src", r)
        }
        function a(t) {
            var n = e(t), r = e(t).data("src");
            r = r !== o ? o : r, n.attr("src", r)
        }
        function f() {
            var e = s, t = c(), n = e * 2 + t, o = t - e * 2;
            for (var f = i.length - 1; f >= 0; f -= 1)
                if (i[f].offset < n && i[f].offset > o&&!i[f].loaded)
                    u(i[f].el), i[f].loaded=!0;
                else if (r.garbageCollection && i[f].offset > n || r.garbageCollection && i[f].offset < o && i[f].loaded)
                    a(i[f].el), i[f].loaded=!1
        }
        function l() {
            var e = document.documentElement.clientHeight || window.innerHeight;
            return s = e, e
        }
        function c() {
            var e = document.documentElement.scrollTop || document.body.scrollTop;
            return e
        }
        function h(e) {
            var t = 0;
            if (e.offsetParent)
                do 
                    t += e.offsetTop, e = e.offsetParent;
            while (e && e.nodeName !== "BODY");
            return t
        }
        function p() {
            var t = e(".lazy-image-img");
            for (var n = t.length - 1; n >= 0; n -= 1)
                e(t[n]).hasClass("need-offset") && (e(t[n]).removeClass("need-offset").addClass("got-offset"), i.push({
                    el: e(t[n]),
                    offset: h(t[n]),
                    loaded: !1
                }))
        }
        function d() {
            Modernizr.touch && e(window).on("touchend", function() {
                f()
            }), t.pubsub.on("scroll", t.pubsub.channels.scrolltop, f), e(window).on("resize", function() {
                l(), f()
            }), window.addEventListener && window.addEventListener("orientationchange", function() {
                l(), f()
            }, !1), t.pubsub.on("storystream", "render_stories", function() {
                p(), f()
            })
        }
        var r = {
            garbageCollection: !1
        }, i = [], s = document.documentElement.clientHeight || window.innerHeight, o = "data:image/gif;base64,R0lGODlhAQABAIAAAP///////yH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
        return {
            init: function(e) {
                r = e ? e : r, p(), f(), i.length > 0 && d()
            }
        }
    }), define("text!bs/templates/user_auth.html", [], function() {
        return '<div id="nav-user-auth" class="nav-user-auth">\n    <a id="btn-facebook-auth-topnav" class="btn-service-auth btn-facebook-auth" href="#fb">\n      <div class="auth-ico"></div>\n      Sign in with Facebook\n    </a>\n    <a id="btn-twitter-auth-topnav" class="btn-service-auth btn-twitter-auth" href="#twitter">\n      <div class="auth-ico"></div>\n      Sign in with Twitter\n    </a>\n    <a id="btn-google-auth-topnav" class="btn-service-auth btn-google-auth" href="#google">\n      <div class="auth-ico"></div>\n      Sign in with Google\n    </a>\n</div>'
    }), define("text!bs/templates/user_nav.html", [], function() {
        return '<div id="nav-user-nav" class="nav-user-nav">\n    <ul class="popover-list">\n        <li class="popover-list-item"><a class="popover-list-link" href="/settings">Settings</a></li>\n        <li class="popover-list-item"><a class="popover-list-link" href="/logout">Sign Out</a></li>\n    </ul>\n</div>'
    }), define("bs/widgets/usernav", ["bs/core", "text!bs/templates/popover.html", "text!bs/templates/user_auth.html", "text!bs/templates/user_nav.html", "lodash"], function(e, t, n, r, i) {
        function h(t) {
            var n = typeof t != "undefined" ? $(t.target).closest(".popover"): $("body").closest(".popover");
            l && n.length === 0 && ($(document).off("click touchend", h), e.pubsub.off("usernav", "auth_tooltip_displayed", h), s.removeClass("usernav-active"), o.removeClass(c), l=!1)
        }
        function p(t) {
            l ? h(t) : (s.addClass("usernav-active"), o.addClass(c), l=!0, $(document).on("click touchend", h), e.pubsub.on("usernav", "auth_tooltip_displayed", h), e.pubsub.fire("sitenav_auth_displayed"), e.pubsub.fire("hide_onboarding_popovers"));
            if ($(t.target).hasClass("nav-link"))
                return !1
        }
        function d() {
            o.addClass("widget-active").append(u({
                popoverContent: a()
            }))
        }
        function v(t) {
            o.addClass("widget-active").html('<a class="nav-link" href="#" title="' + t.name + '">' + t.name + "</a>" + u({
                popoverContent: f()
            })), s.addClass("authenticated"), h(), e.pubsub.off("site_nav_user_widget", e.pubsub.channels.user_auth, v)
        }
        function m() {
            var t = e.user.get("name");
            o.find(".nav-link").attr("title", t).html(t)
        }
        var s = $("#digg-header"), o = $("#nav-user"), u = i.template(t), a = i.template(n), f = i.template(r), l=!1, c = "show-popover";
        return {
            init: function() {
                e.user.isLoggedIn() ? $(m) : (d(), e.pubsub.one("site_nav_user_widget", e.pubsub.channels.user_auth, v)), $(document).on("click", "#nav-user", p), e.pubsub.on("usernav", "savedstories_modal_displayed", h), e.pubsub.on("usernav", "onboarding_firstsave", h)
            }
        }
    }), define("text!bs/templates/story_saved.html", [], function() {
        return '<article class="story-container story-<%= content.content_id %> digg-story-el" data-content-id="<%= content.content_id %>" data-contenturl="<%=content.url%>" data-position="<%=position%>" data-diggs="<%= diggs.count %>" data-tweets="<%= tweets.count %>" data-fb-shares="<%= fb_shares.count %>" data-digg-score="<%= digg_score %>">\n    <% if (typeof image != \'undefined\') { %>\n    <div class="story-image">\n        <a data-position="<%=position%>" data-content-id="<%= content.content_id %>" class="story-link" href="<%= content.url %>" target="<%=content.link_target%>">\n            <img class="story-image-img" src="<%= image.url %>" width="<%= image.width %>" height="<%= image.height %>" alt="" />\n        </a>\n    </div>\n    <% } %>\n    <div class="story-content">\n        <header class="story-header">\n            <div class="story-kicker"><%= content.kicker %></div>\n            <h2 class="story-title entry-title">\n                <a class="story-title-link story-link" href="<%= content.url %>" target="<%=content.link_target%>"><%= content.title ? content.title.replace(/(\\S+) (\\S+)$/, "$1&nbsp;$2") : "" %></a>\n            </h2>\n            <div class="story-meta">\n            <% if (content.domain_name != \'YouTube\') { %>\n                <span class="story-meta-item story-source">\n                    <% if (content.domain_name === \'Digg Top Stories\') { %>\n                        <a itemprop="publisher copyrightHolder sourceOrganization provider" class="story-meta-item-link story-source-link" href="/source/<%= content.domain %>"><%= content.domain %></a>\n\n                    <% } else { %>\n                        <%= content.domain_name %>\n\n                    <% } %>\n                </span>\n            <% } %>\n            <% if (content.tags && content.tags.length) { %>\n                <% for (var i = 0, j = content.tags.length; i < j; i++) { %>\n                    <span class="story-meta-item story-tag<% if (i === 0) { %> first-tag<% } %>">\n                        <a itemprop="keywords" class="story-meta-item-link story-tag-link" href="/tag/<%= content.tags[i].slug %>"><%= content.tags[i].display %></a>\n                    </span>\n                <% } %>\n            <% } %>\n            <% if (date !== null) { %><abbr class="published story-meta-item story-timestamp"><time><%= formattedDate %></time></abbr><% } %>\n            </div>\n        </header>\n    </div>\n    <ul class="story-actions">\n        <li class="story-action story-action-digg btn-story-action-container"><a class="target digg-<%= content.content_id %>" href="#">Digg</a></li>\n        <li class="story-action story-action-save btn-story-action-container"><a class="target save-<%= content.content_id %>" href="#">Save</a></li>\n        <li class="story-action story-action-share"><a class="target share-facebook" href="<%= fblink %>">Facebook</a></li>\n        <li class="story-action story-action-share"><a class="target share-twitter" href="<%= tweetlink %>">Twitter</a></li>\n    </ul>\n</article>\n'
    }), define("text!bs/templates/saved_nav.html", [], function() {
        return '<li id="saved-stories-container" class="nav-item nav-saved"><a class="nav-link" href="#savedstories" id="btn-savedstories">Saved <sup id="savedstories-new-count" class="savedstories-none"><%= savedStoryCount %></sup></a></li>'
    }), define("text!bs/templates/reader/feed_zero_saved.html", [], function() {
        return '<div class="zero-items-message">\n    <h2 class="zero-items-message-hed">You haven’t saved anything yet.</h2>\n    <p>Click the <span class="save-icon">SAVE ICON</span> to save an item here.</p>\n</div>\n'
    }), define("bs/widgets/savedstories2", ["bs/core", "bs/util/xhr", "bs/util/format", "bs/util/storyactions2", "bs/util/storyscores", "text!bs/templates/story_saved.html", "text!bs/templates/saved_nav.html", "text!bs/templates/reader/feed_zero_saved.html", "bootstrap_modal", "lodash"], function(e, t, n, r, i, s, o, u, a, f) {
        function E(t) {
            var r = [""], i = [];
            t.length === 0 ? b.html(y) : ($("#btn-savedstories-more").remove(), f.each(t, function(t, s) {
                t.view = "saved", t.position = s, t.content.link_to_publisher=!0, t.content.link_target = t.content.link_to_publisher ? "_blank" : "_self", t.content.title = t.content.title_alt ? t.content.title_alt : t.content.title, t.formattedDate = n.secondsToFuzzyTime(t.date), t.tweetlink = "https://twitter.com/intent/tweet?url=http://on.digg.com/" + t.content.content_id + "&via=Digg&text=" + encodeURIComponent(t.content.title), t.fblink = "https://www.facebook.com/dialog/feed?display=popup&app_id=" + e.config.FB_APP_ID + "&link=http://on.digg.com/" + t.content.content_id + "&redirect_uri=http://digg.com/fbshare", i.push(t.content.content_id), r.push(m(t))
            }), p !== "-1" && r.push('<div id="btn-savedstories-more" class="btn-action">Load more...</div>'), b.append(r.join("")).parents(".modal").removeClass("loading"), h = l.find(".modal-body").get(0).scrollHeight, e.pubsub.fire("savedstories_rendered", i)), b.parents(".modal").removeClass("loading")
        }
        function S() {
            var e = new t.dataProvider({
                url: "/api/readlater/stats.json",
                success: function(e) {
                    v = e.data["new"], $("#nav-user").before(g({
                        savedStoryCount: v
                    })), v > 0 && $("#savedstories-new-count").removeClass("savedstories-none")
                }
            });
            e.request()
        }
        function x() {
            var e = $("#savedstories-new-count"), n = new t.dataProvider({
                url: "/api/readlater/stats.json",
                success: function(t) {
                    v = t.data["new"], e.html(v), v > 0 && e.removeClass("savedstories-none")
                }
            });
            n.request()
        }
        function T(e) {
            var t = $("#savedstories-new-count"), n=~~t.html() || 0;
            n += e, n > 0 ? t.removeClass("savedstories-none") : t.addClass("savedstories-none"), t.html(n)
        }
        function N(e, t, n) {
            return $(".story-" + n).addClass("story-deleted").find(".save-" + n).text("Undo"), !1
        }
        function C(e, t, n) {
            return $(".story-" + n).removeClass("story-deleted").find(".save-" + n).text("Saved"), !1
        }
        var l, c, h, p, d, v = 0, m = f.template(s), g = f.template(o), y = f.template(u), b = $("#modal-savedstories-container"), w = new t.dataProvider({
            url: "/api/readlater.json",
            cache: !1,
            success: function(e) {
                c = e.data.feed, p = e.data.next_position, d = e.data.total_count, E(c)
            },
            error: function(e) {
                b.html('<div class="stories-error">There was a problem loading Saved stories. <br/>We\'re working on a fix. Wait a moment and try again.</div>')
            }
        });
        e.pubsub.on("saved_stories", "add_reading_list_success", function() {
            T(1)
        }), e.pubsub.on("saved_stories", "delete_reading_list_request", function() {
            T( - 1)
        });
        var k = {
            initialize: function(t) {
                e.user.isLoggedIn() ? x() : e.pubsub.one("savedstories", e.pubsub.channels.user_auth, S), l = $(t.elID).modal({
                    keyboard: !0,
                    show: !1
                }), e.pubsub.on("story_scores", "savedstories_rendered", function(e) {
                    var t = i.factory("saved_stories_" + (new Date).valueOf());
                    t.register(e)
                }), $(document).on("click touchend", "#btn-savedstories", k.show).on("click touchend", "#btn-savedstories-modal-close", k.hide).on("click touchend", ".btn-deleted-undo", C).on("click touchend", "#btn-savedstories-more", function() {
                    $(this).addClass("loading").html("<span class='loading-txt'>Loading ...</span>"), w.request({
                        data: {
                            position: p
                        }
                    })
                })
            },
            show: function() {
                var t = Math.floor($(window).height() * .8), n = l.find(".modal-body");
                return l.addClass("loading").css({
                    height: t + "px",
                    marginTop: "-" + Math.floor(t / 2) + "px"
                }).modal("show"), n.height(t - l.find(".modal-header").height() - 100 + "px"), w.request(), $(document).on("click touchend", k.hide), e.pubsub.on("savedstories_list", "delete_reading_list_request", N), e.pubsub.on("savedstories_list", "add_reading_list_success", C), e.pubsub.fire("savedstories_modal_displayed"), !1
            },
            hide: function(t) {
                var n = $(t.target).closest(l), r = $(t.target).attr("id") === "btn-savedstories-modal-close";
                if (n.length === 0 || r)
                    l.modal("hide"), b.html("").empty(), $(document).off("click touchend", k.hide), e.pubsub.off("savedstories_list", "delete_reading_list_request", N), e.pubsub.off("savedstories_list", "add_reading_list_success", C);
                return !1
            }
        };
        return k
    }), define("bs/util/interface", [], function() {
        function e(e) {
            this.members = e
        }
        return {
            factory: function(t) {
                return new e(t)
            },
            ensureImplements: function(e, t) {
                var n, r, i;
                for (n = 0, r = t.members.length; n < r; n++) {
                    i = t.members[n];
                    if (typeof e[i] == "undefined")
                        throw new Error("InterfaceError: missing member '" + i + "'")
                }
                return !0
            }
        }
    }), define("bs/util/touchit", ["bs/util/interface"], function(e) {
        function t(e, t, n, r) {
            if (e.addEventListener)
                return e.addEventListener(t, n, r), {
                    destroy: function() {
                        e.removeEventListener(t, n, r)
                    }
                };
            var i = function(e) {
                n.handleEvent(window.event, n)
            };
            return e.attachEvent("on" + t, i), {
                destroy: function() {
                    e.detachEvent("on" + t, i)
                }
            }
        }
        function i(e, n, i) {
            this.events = [], this.touchEvents = [], this.element = e, this.handler = n, this.useCapture = i, this.addClassTimer = null, this.removeClassTimer = null, r ? this.events.push(t(e, "touchstart", this, this.useCapture)) : this.events.push(t(e, "click", this, this.useCapture)), document.addEventListener("click", this.clickbuster.onClick, !0), this.clickbuster.coordinates = []
        }
        var n = "touching", r = "ontouchstart"in window;
        return i.prototype = {
            destroy: function() {
                for (var e = this.events.length - 1; e >= 0; e -= 1)
                    this.events[e].destroy();
                this.events = this.touchEvents = this.element = this.handler = this.fastButton = null
            },
            handleEvent: function(e) {
                switch (e.type) {
                case"touchstart":
                    this.onTouchStart(e);
                    break;
                case"touchmove":
                    this.onTouchMove(e);
                    break;
                case"touchend":
                    this.onClick(e);
                    break;
                case"click":
                    this.onClick(e)
                }
            },
            onTouchStart: function(e) {
                var r = this.element, i = n;
                this.touchEvents.push(t(this.element, "touchend", this, this.useCapture)), this.touchEvents.push(t(document.body, "touchmove", this, this.useCapture)), this.startX = e.touches[0].clientX, this.startY = e.touches[0].clientY, this.addClassTimer != null && clearTimeout(this.addClassTimer), this.removeClassTimer != null && clearTimeout(this.removeClassTimer), this.hasClass(r, i) || (this.addClassTimer = setTimeout(function() {
                    r.className = r.className + " " + i
                }, 10), this.removeClassTimer = setTimeout(function() {
                    r.className = r.className.replace(new RegExp("\\b" + i + "\\b"), "")
                }, 310))
            },
            onTouchMove: function(e) {
                if (Math.abs(e.touches[0].clientX - this.startX) > 10 || Math.abs(e.touches[0].clientY - this.startY) > 10)
                    this.removeClass(this.element, n), this.reset()
            },
            onClick: function(e) {
                e.stopPropagation();
                var t = this.handler.call(this.element, e);
                return e.type === "touchend" && this.clickbuster.preventGhostClick(this.startX, this.startY), this.reset(), t
            },
            addClass: function(e, t) {
                e.className = e.className + " " + t
            },
            removeClass: function(e, t) {
                e.className = e.className.replace(new RegExp("\\b" + t + "\\b"), "")
            },
            hasClass: function(e, t) {
                return e.className.indexOf(t)>-1?!0 : !1
            },
            reset: function() {
                clearTimeout(this.addClassTimer), this.addClassTimer = null, clearTimeout(this.removeClassTimer), this.removeClassTimer = null, setTimeout(this.removeClass(this.element, n), 200);
                for (var e = this.touchEvents.length - 1; e >= 0; e -= 1)
                    this.touchEvents[e].destroy();
                this.touchEvents = []
            },
            clickbuster: {
                preventGhostClick: function(e, t) {
                    this.coordinates.push(e, t), setTimeout(this.pop, 2500)
                },
                pop: function() {
                    typeof this.coordinates != "undefined" && this.coordinates.splice(0, 2)
                },
                onClick: function(e) {
                    if (typeof this.coordinates != "undefined")
                        for (var t = 0; t < this.coordinates.length; t += 2) {
                            var n = this.coordinates[t], r = this.coordinates[t + 1];
                            Math.abs(e.clientX - n) < 25 && Math.abs(e.clientY - r) < 25 && (alert("ghost click"), e.stopPropagation(), e.preventDefault())
                        }
                }
            }
        }, {
            factory: function(e, t, n) {
                return new i(e, t, n)
            }
        }
    }), define("bs/widgets/headersearch2", ["bs/core", "bs/util/helpers", "bs/util/touchit", "jquery"], function(e, t, n, r) {
        function d(e) {
            return Modernizr.csstransitions ? (u.on(p, v), i.addClass(f)) : v(), a.focus(), !1
        }
        function v(e) {
            r(document).on("click touchend", m), Modernizr.csstransitions ? (u.off(p, v), i.removeClass(f).addClass(l)) : i.addClass(l), h=!0
        }
        function m(e) {
            Modernizr.csstransitions && e.target.name !== "q" && h ? (r(document).off("click touchend", m), u.on(p, g), i.removeClass(l).addClass(c)) : h && e.target.name !== "q" && g(e)
        }
        function g(e) {
            Modernizr.csstransitions ? (u.off(p, g), i.removeClass(c)) : i.removeClass(l), h=!1, a.blur(), r(document).off("click touchend", m)
        }
        function y(e) {
            Modernizr.csstransitions && u.off(p, g), i.removeClass(c), i.removeClass(l), h=!1, a.blur(), r(document).off("click touchend", m)
        }
        var i = r("#digg-header"), s = document.getElementById("nav-search"), o = r(s), u = r("#form-digg-search"), a = r("#form-digg-search-input"), f = "nav-search-expand", l = "nav-search-active", c = "nav-search-collapse", h=!1, p = t.transitionEndEventNames[Modernizr.prefixed("transition")];
        return {
            init: function() {
                document.addEventListener ? n.factory(s, d) : o.on("click", d), e.pubsub.on("mobileheader", "mobile_nav_collapse", y)
            }
        }
    }), define("bs/widgets/mobileheader", ["jquery", "bs/core", "bs/util/touchit", "lodash"], function(e, t, n, r) {
        function a(n) {
            var r = "mobile-nav-active";
            u && (f(), u=!1), s.hasClass(r) ? (s.removeClass(r), e("body").removeClass("freeze-scroll"), t.pubsub.fire("mobile_nav_collapse")) : (s.addClass(r), e("body").addClass("freeze-scroll"))
        }
        function f() {
            var e = l(), t = s.height(), n = e - t;
            window.matchMedia && window.matchMedia("(max-width:879px)").matches ? o.css("height", n) : window.matchMedia && window.matchMedia("(min-width:880px)").matches && o.css("height", 55)
        }
        function l() {
            return document.documentElement.clientHeight
        }
        function c() {
            f(), u=!1
        }
        function h(e) {
            var t = document.getElementById("digg-header");
            if (e && e.target === t)
                return e.stopPropagation(), e.preventDefault(), !1
        }
        var i = document.getElementById("digg-nav-toggle"), s = e("#digg-header"), o = s.find(".digg-nav-wrap"), u=!0;
        return {
            init: function() {
                e(window).on("resize orientationchange", r.throttle(c, 100)), e(document).on("touchstart", h), document.addEventListener ? n.factory(i, a) : e(i).on("click", a)
            }
        }
    }), define("bs/widgets/promobanner", ["require", "bs/core", "bs/util/cookies"], function(e, t, n) {
        function s() {
            var e = $(this).parents(i).data("campaign");
            n.set({
                name: e,
                value: 1,
                expires: new Date((new Date).valueOf() + 31536e6),
                path: "/"
            }), $("body").addClass("promo-collapse"), i.off("click touchend", ".promo-close-btn", s)
        }
        var r=!1, i = $("#site-promo-banner"), o = {
            init: function() {
                i.length > 0&&!r && i.on("click touchend", ".promo-close-btn", s)
            }
        };
        return o
    }), define("text!bs/templates/story.html", [], function() {
        return '<article class="story-container story-<%= content.content_id %> digg-story-el" data-content-id="<%= content.content_id %>" data-contenturl="<%=content.url%>" data-position="<%=position%>" data-diggs="<%= diggs.count %>" data-tweets="<%= tweets.count %>" data-fb-shares="<%= fb_shares.count %>" data-digg-score="<%= digg_score %>">\n    <% if (typeof image != \'undefined\') { %>\n    <div class="story-image">\n        <a data-position="<%=position%>" data-content-id="<%= content.content_id %>" class="story-link" href="<%= content.url %>" target="<%=content.link_target%>">\n            <img class="story-image-img" src="<%= image.url %>" width="<%= image.width %>" height="<%= image.height %>" alt="" />\n        </a>\n    </div>\n    <% } %>\n    <div class="story-content">\n        <header class="story-header">\n            <div class="story-kicker"><%= content.kicker %></div>\n            <h2 class="story-title entry-title">\n                <a class="story-title-link story-link" href="<%= content.url %>" target="<%=content.link_target%>"><%= content.title ? content.title.replace(/(\\S+) (\\S+)$/, "$1&nbsp;$2") : "" %></a>\n            </h2>\n            <div class="story-meta">\n            <% if (typeof showDiggsNotDiggScore !== \'undefined\' && showDiggsNotDiggScore) { %>\n                <div class="story-score<% if (diggs.count == 0) { %> story-score-zero<% } %>">\n                    <div class="story-score-diggs-not-diggscore"><span class="count diggs-<%= content.content_id %>"><%= diggs.count %></span>&nbsp;<span class="label diggs-label-<%= content.content_id %>"><% if (diggs.count == 1) { %>digg<% } else { %>diggs<% } %></span></div>\n                    <div class="story-score-diggscore hide-diggscore diggscore-<%= content.content_id %>"><%= digg_score %></div>\n            <% } else { %>\n                <div class="story-score<% if (digg_score == 0) { %> story-score-zero<% } %>">\n                    <div class="story-score-diggscore diggscore-<%= content.content_id %>"><%= digg_score %></div>\n            <% } %>\n                    <div class="story-score-details">\n                        <div class="arrow"></div>\n                        <ul class="story-score-details-list">\n                            <li class="story-score-detail story-score-diggs"><span class="label">Diggs:</span> <span class="count diggs-<%= content.content_id %>"><%= diggs.count %></span></li>\n                            <li class="story-score-detail story-score-twitter"><span class="label">Tweets:</span> <span class="count tweets-<%= content.content_id %>"><%= tweets.count %></span></li>\n                            <li class="story-score-detail story-score-facebook"><span class="label">Facebook Shares:</span> <span class="count fb_shares-<%= content.content_id %>"><%= fb_shares.count %></span></li>\n                        </ul>\n                    </div>\n                </div>\n            <% if (content.domain_name != \'YouTube\') { %>\n                <span class="story-meta-item story-source">\n                    <a itemprop="publisher copyrightHolder sourceOrganization provider" class="story-meta-item-link story-source-link" href="/source/<%= content.domain %>"><%= content.domain_name %></a>\n                </span>\n            <% } %>\n            <% if (content.tags && content.tags.length) { %>\n                <% for (var i = 0, j = content.tags.length; i < j; i++) { %>\n                    <span class="story-meta-item story-tag<% if (i === 0) { %> first-tag<% } %>">\n                        <a itemprop="keywords" class="story-meta-item-link story-tag-link" href="/tag/<%= content.tags[i].slug %>"><%= content.tags[i].display %></a>\n                    </span>\n                <% } %>\n            <% } %>\n            <% if (date !== null) { %><abbr class="published story-meta-item story-timestamp"><time><%= formattedDate %></time></abbr><% } %>\n            </div>\n        </header>\n    </div>\n    <ul class="story-actions">\n        <li class="story-action story-action-digg btn-story-action-container"><a class="target digg-<%= content.content_id %>" href="#">Digg</a></li>\n        <li class="story-action story-action-save btn-story-action-container"><a class="target save-<%= content.content_id %>" href="#">Save</a></li>\n        <li class="story-action story-action-share"><a class="target share-facebook" href="<%= fblink %>">Facebook</a></li>\n        <li class="story-action story-action-share"><a class="target share-twitter" href="<%= tweetlink %>">Twitter</a></li>\n    </ul>\n</article>\n'
    }), define("bs/widgets/popularvideos", ["bs/core", "bs/util/xhr", "bs/util/format", "bs/util/storyscores", "text!bs/templates/story.html", "lodash"], function(e, t, n, r, i, s) {
        function d() {
            o.addClass("loaded"), u.html("<div class='stories-error'>There was problem loading the Most Dugg Videos. Wait a moment and try again.</div>")
        }
        function v(t) {
            var r = t.data.feed, i = [], f = "";
            s.each(r, function(t, r) {
                t.position = r, t.content.link_target = t.content.link_to_publisher ? "_blank" : "_self", t.content.title = t.content.title_alt ? t.content.title_alt : t.content.title, t.image = s.findWhere(t.content.media.images, {
                    blurred: !1,
                    type: "large_thumb"
                }), r === 0 && (f = s.findWhere(t.content.media.images, {
                    blurred: !0,
                    type: "marquee_standard"
                })), t.formattedDate = n.secondsToFuzzyTime(t.date), t.diggs.count = n.intWithCommas(t.diggs.count), t.tweets.count = n.intWithCommas(t.tweets.count), t.fb_shares.count = n.intWithCommas(t.fb_shares.count), t.digg_score = n.intWithCommas(t.digg_score), t.showDiggsNotDiggScore=!0, t.tweetlink = "https://twitter.com/intent/tweet?url=http://on.digg.com/" + t.content.content_id + "&via=Digg&text=" + encodeURIComponent(t.content.title), t.fblink = "https://www.facebook.com/dialog/feed?display=popup&app_id=" + e.config.FB_APP_ID + "&link=http://on.digg.com/" + t.content.content_id + "&redirect_uri=http://digg.com/fbshare", i.push(a(t))
            }), o.addClass("loaded").css("background-image", "url(" + f.url + ")"), u.empty().append(i.join(""))
        }
        var o = $("#popular-videos"), u = o.find(".content"), a = s.template(i), f = r.factory("popular_videos"), l = [], c=!1, h = {
            count: 5,
            exclude_sponsored: !0
        }, p = new t.dataProvider({
            url: "/api/video/popular.json",
            data: h,
            success: function(e) {
                v(e), l = s.map(e.data.feed, function(e) {
                    return e.content.content_id
                }), l.length ? f.register(l) : d()
            },
            error: function(e) {
                d()
            }
        });
        return {
            init: function() {
                if (c)
                    return;
                c=!0, p.request()
            }
        }
    }), define("bs/widgets/popularstories", ["bs/core", "bs/util/xhr", "bs/util/format", "bs/util/storyscores", "text!bs/templates/story.html", "lodash"], function(e, t, n, r, i, s) {
        function d() {
            o.addClass("loaded"), u.html('<div class="stories-error">There was problem loading the Most Dugg Stories. Wait a moment and try again.</div>')
        }
        function v(t) {
            var r = t.data.feed, i = [];
            s.each(r, function(t, r) {
                t.position = r, t.content.link_target = t.content.link_to_publisher ? "_blank" : "_self", t.content.title = t.content.title_alt ? t.content.title_alt : t.content.title, t.image = s.findWhere(t.content.media.images, {
                    blurred: !1,
                    type: "large_thumb"
                }), t.formattedDate = n.secondsToFuzzyTime(t.date), t.diggs.count = n.intWithCommas(t.diggs.count), t.tweets.count = n.intWithCommas(t.tweets.count), t.fb_shares.count = n.intWithCommas(t.fb_shares.count), t.digg_score = n.intWithCommas(t.digg_score), t.showDiggsNotDiggScore=!0, t.tweetlink = "https://twitter.com/intent/tweet?url=http://on.digg.com/" + t.content.content_id + "&via=Digg&text=" + encodeURIComponent(t.content.title), t.fblink = "https://www.facebook.com/dialog/feed?display=popup&app_id=" + e.config.FB_APP_ID + "&link=http://on.digg.com/" + t.content.content_id + "&redirect_uri=http://digg.com/fbshare", i.push(a(t))
            }), o.addClass("loaded"), u.empty().append(i.join(""))
        }
        var o = $("#popular-stories"), u = o.find(".content"), a = s.template(i), f = {
            count: 10,
            exclude_sponsored: !0,
            exclude_videos: !0
        }, l = new t.dataProvider({
            url: "/api/news/popular.json",
            data: f,
            success: function(e) {
                h = s.map(e.data.feed, function(e) {
                    return e.content.content_id
                }), h.length ? (c.register(h), v(e)) : d()
            },
            error: function(e, t, n) {
                d()
            }
        }), c = r.factory("popular_stories"), h = [], p=!1;
        return {
            init: function() {
                if (p)
                    return;
                p=!0, l.request()
            }
        }
    }), define("bs/widgets/upcomingstories", ["bs/core", "bs/util/xhr", "bs/util/format", "bs/util/storyscores", "text!bs/templates/story.html", "lodash"], function(e, t, n, r, i, s) {
        function d() {
            o.addClass("loaded"), u.html("<div class='stories-error'>There was problem loading the Upcoming Stories. Wait a moment and try again.</div>")
        }
        function v(t) {
            var r = t.data.feed, i = [];
            s.each(r, function(t, r) {
                t.position = r, t.content.link_target = t.content.link_to_publisher ? "_blank" : "_self", t.content.title = t.content.title_alt ? t.content.title_alt : t.content.title, t.formattedDate = n.secondsToFuzzyTime(t.date), t.diggs.count = n.intWithCommas(t.diggs.count), t.tweets.count = n.intWithCommas(t.tweets.count), t.fb_shares.count = n.intWithCommas(t.fb_shares.count), t.digg_score = n.intWithCommas(t.digg_score), t.showDiggsNotDiggScore=!0, t.tweetlink = "https://twitter.com/intent/tweet?url=http://on.digg.com/" + t.content.content_id + "&via=Digg&text=" + encodeURIComponent(t.content.title), t.fblink = "https://www.facebook.com/dialog/feed?display=popup&app_id=" + e.config.FB_APP_ID + "&link=http://on.digg.com/" + t.content.content_id + "&redirect_uri=http://digg.com/fbshare", i.push(a(t))
            }), o.addClass("loaded"), u.empty().append(i.join(""))
        }
        var o = $("#upcoming-stories"), u = o.find(".content"), a = s.template(i), f = r.factory("upcoming_stories"), l = [], c=!1, h = {
            count: 15
        }, p = new t.dataProvider({
            url: "/api/news/upcoming.json",
            data: h,
            success: function(e) {
                l = s.map(e.data.feed, function(e) {
                    return e.content.content_id
                }), l.length ? (f.register(l), v(e)) : d()
            },
            error: function(e) {
                d()
            }
        });
        return {
            init: function() {
                if (c)
                    return;
                c=!0, p.request()
            }
        }
    }), define("bs/widgets/footer", ["bs/core", "bs/util/xhr", "jquery"], function(e, t, n) {
        function r() {
            var r = n.trim(n("#digg-reader-email-form-input").val()), i = "/api/promo/signup.json", s = new t.dataProvider({
                url: i,
                type: "POST",
                data: {
                    promo: "reader_android",
                    email: r
                },
                success: function(t) {
                    n("#digg-reader-email-form").hide(), n("#digg-reader-email-error").hide(), n("#digg-reader-email-success").html("Thanks!").show(), e.pubsub.fire("save_digg_reader_submit_complete")
                },
                error: function(e) {
                    n("#digg-reader-email-error").html("There was a problem submitting your email.").show()
                }
            });
            return s.request(), !1
        }
        function i() {
            var r = n.trim(n("#daily-digg-email-input").val()), i = "/api/dailydiggemail.json", s = new t.dataProvider({
                url: i,
                type: "POST",
                data: {
                    email: r
                },
                success: function(t) {
                    n("#daily-digg-email-form").hide(), n("#daily-digg-email-error").hide(), n("#daily-digg-email-success").html("Thanks for subscribing!").show(), e.pubsub.fire("save_daily_digg_submit_complete")
                },
                error: function(e) {
                    n("#daily-digg-email-error").html("There was a problem submitting your email. Please try again.").show()
                }
            });
            return s.request(), !1
        }
        n(function() {
            Modernizr.inputtypes.email && n("#daily-digg-email-input").clone().attr("type", "email").insertAfter("#daily-digg-email-input").prev().remove(), n("#footer-copyright-year").html((new Date).getFullYear()), n("#daily-digg-email-input").on("keyup", function(e) {
                e.keyCode == 13 ? i() : n("#daily-digg-email-error").html("&nbsp;")
            }), n("#daily-digg-email-submit-btn").on("click", function(e) {
                i()
            }), n("#digg-reader-email-form-input").on("keyup", function(e) {
                return e.keyCode == 13 ? r() : n("#digg-reader-email-error").html("&nbsp;"), e.stopPropagation(), e.preventDefault(), !1
            }), n("#reader-promo-submit-btn").on("click", function(e) {
                r()
            }), n("#daily-digg-email-form").addClass("ready"), n("#digg-reader-email-form").addClass("ready")
        })
    }), define("bs/util/storyactions_v3", ["require", "bs/core", "bs/util/cookies", "bs/util/xhr", "bs/util/format", "bs/util/helpers", "bs/widgets/onboarding", "lodash"], function(e, t, n, r, i, s, o, u) {
        function d(e) {
            var t = (new Date).valueOf() + 31536e6;
            $.extend(h, e), n.set({
                name: c,
                value: $.param(h),
                expires: new Date(t)
            })
        }
        function v(e) {
            var n = "http://api.digg.com/api/v3/redirect", r = $(this).closest(a), i = r.data("contentId"), s = t.user.get("user_id"), o = r.data("source"), u = {
                url: this.href
            };
            i && (u.content_id = i), s && (u.user_id = s), o && (u.source = o), navigator.mozApps && window.locationbar.visible===!1 && (u.platform = 1005, window.open([n, $.param(u)].join("?")));
            if (p === "mobile")
                u.platform = 1005, window.location.href = [n, $.param(u)].join("?");
            else {
                if (this.target !== "_blank")
                    return;
                window.open([n, $.param(u)].join("?"))
            }
            return !1
        }
        function m(e) {
            var t = {
                contentID: e.data("contentId"),
                diggUrlID: e.data("urlId"),
                category: e.data("category"),
                diggScore: e.data("diggScore"),
                tweets: e.data("tweets"),
                fbShares: e.data("fbShares"),
                diggs: e.data("diggs"),
                dugg: e.data("dugg") || e.hasClass(f),
                read_later: e.data("read_later") || e.hasClass(l)
            };
            return t.diggScore = u.isString(t.diggScore) ? parseInt(t.diggScore.replace(/,/g, ""), 10) : t.diggScore, t.tweets = u.isString(t.tweets) ? parseInt(t.tweets.replace(/,/g, ""), 10) : t.tweets, t.fbShares = u.isString(t.fbShares) ? parseInt(t.fbShares.replace(/,/g, ""), 10) : t.fbShares, t.diggs = u.isString(t.diggs) ? parseInt(t.diggs.replace(/,/g, ""), 10) : t.diggs, t
        }
        function g(e, t) {
            e.data(t)
        }
        function y(e) {
            u.each(e, function(e, t) {
                b(t, e)
            })
        }
        function b(e, t, n) {
            var r = $(".story-" + e), s = $(".diggscore-" + e), o = $(".diggs-" + e), u = $(".tweets-" + e), a = $(".fb_shares-" + e), c = $(".digg-" + e), h = $(".save-" + e), p = s.parent(".story-score"), d = $(".diggs-label-" + e);
            g(r, {
                diggScore: t.digg_score,
                diggs: t.diggs,
                tweets: t.tweets,
                fbShares: t.fb_shares,
                dugg: t.dugg,
                read_later: t.read_later
            });
            if (!s ||!s.size())
                return;
            t.digg_score > 0 && p.hasClass("story-score-zero") ? p.removeClass("story-score-zero") : t.digg_score === 0 && p.addClass("story-score-zero"), d.length > 0 && t.diggs === 0 ? p.addClass("story-score-zero") : d.length > 0 && t.diggs === 1 ? (d.html("digg"), p.removeClass("story-score-zero")) : d.length > 0 && (d.html("diggs"), p.removeClass("story-score-zero")), o.html(i.intWithCommas(t.diggs)), u.html(i.intWithCommas(t.tweets)), a.html(i.intWithCommas(t.fb_shares)), s.html(i.intWithCommas(t.digg_score, !0)), !n && t.hasOwnProperty("dugg") && t.dugg && (c.html("Dugg"), r.addClass(f)), !n && t.hasOwnProperty("read_later") && t.read_later && (h.html("Saved"), r.addClass(l))
        }
        function w(e) {
            var t = $(this), n = t.parents(a), i = m(n), s = {
                content_id: i.contentID,
                digg_url_id: i.diggUrlID
            }, o = i.dugg ? "/api/digg/delete.json": "/api/digg/submit.json", f = new r.dataProvider({
                url: o,
                type: "POST",
                data: s,
                success: u.bind(E, this, t, i.contentID)
            });
            return f.request(), !1
        }
        function E(e, n, r) {
            var i = document.location.hash, s = $('.js--digg-story[data-content-id="' + n + '"]'), u = (new Date).valueOf() / 1e3-~~t.user.get("date_signup") < 900, a = r.data;
            b(n, a, !0), a.dugg ? (t.pubsub.fire("digg_success", e, a, n), s.addClass(f), s.find(".js--digg-label").html("Dugg")) : (t.pubsub.fire("digg_delete_success", e, a, n), s.removeClass(f), s.find(".js--digg-label").html("Digg"));
            if (i.match("test-onboarding") || a.dugg&&!h.dugg && u)
                o.firstDigg(e), d({
                    dugg: 1
                })
        }
        function S(e) {
            var t = $(this), n = t.parents(a), i = m(n), s = i.read_later ? "/api/story/readlater/delete.json": "/api/story/readlater/add.json", o = {
                content_id: i.contentID,
                digg_url_id: i.diggUrlID
            }, f = new r.dataProvider({
                url: s,
                type: "POST",
                data: o,
                success: u.bind(x, this, t, i.contentID)
            });
            return f.request(), !1
        }
        function x(e, n, r) {
            var i = document.location.hash, s = $('.js--digg-story[data-content-id="' + n + '"]'), u = (new Date).valueOf() / 1e3-~~t.user.get("date_signup") < 900, a = r.data;
            g(s, {
                read_later: a.read_later
            }), a.read_later ? (s.addClass(l), s.find(".js--save-label").html("Saved"), t.pubsub.fire("add_reading_list_success", e, a, n)) : (s.removeClass(l), s.find(".js--save-label").html("Save"), t.pubsub.fire("delete_reading_list_request", e, a, n));
            if (i.match("test-onboarding") || a.read_later&&!h.saved && u)
                o.firstSave(e), d({
                    saved: 1
                })
        }
        function T(e) {
            var n = $(this), r = n.hasClass("share-twitter") ? "twitter": "facebook";
            return e.preventDefault(), t.pubsub.fire("story_share", r, n), r === "twitter"?!1 : (s.popWin(this.href, null, {
                width: 550,
                height: 450
            }), !1)
        }
        function N(e) {
            var t = $(this), n = t.data("shareContentId"), r = t.data("shareId"), i = "share-details-" + n + "-" + r;
            $(".share-details-" + n).hide(), $("#" + i).show(), $(".story-trending-share-img-container-" + n).removeClass("current"), t.parent().addClass("current")
        }
        var a = ".js--digg-story", f = "is--dugg", l = "is--saved", c = "saflags", h = n.get(c), p = n.get("preferred_view") || "desktop";
        return h = h ? s.deparam(h) : {}, {
            submitDigg: w,
            addReadingList: S,
            renderStoryScores: y,
            redirectStoryLink: v,
            shareStory: T,
            showShareDetails: N
        }
    }), define("text!bs/templates/deeper/story_modal.html", [], function() {
        return '<div class="js--modal--digg-deeper-story modal--digg-deeper modal fade hide" role="dialog" aria-hidden="true">\n\n    <div class="js--btn-modal-close modal--digg-deeper__close">✖</div>\n\n    <%= storyDetailsHTML %><!-- will be article.digg-story element -->\n\n    <div class="modal--digg-deeper__shared-by">Shared by <%= sharerCount %> people</div>\n\n    <div class="modal--digg-deeper_sharers">\n        <%= sharersHTML %>\n\n        <div class="ico-loading"></div>\n    </div>\n\n</div>\n'
    }), define("text!bs/templates/deeper/story.html", [], function() {
        return '<% \n/**\n * Template for a story living in the Digg Deeper stream\n */\n%>\n\n<article \n    class="digg-story--deeper\n        js--digg-story\n        js--story-<%= content.content_id %>\n        <%= dugg ? \'is--dugg\' : \'\' %>\n        <%= read_later ? \'is--saved\' : \'\' %>\n    " \n    data-content-id="<%= content.content_id %>"\n    data-contenturl="<%= content.url %>"\n    data-position="<%= position %>"\n    data-source="digg_deeper"\n>\n    <% if ( ! tweets.hideSharers) { %>\n        <% if (feed === \'fof\') { %>\n            <div class="digg-story--deeper__sharers">\n                <% var sharer, tweet; %>\n                <% for (var s = 0, lenTweets = friends.data.length > maxSharersDisplayed ? maxSharersDisplayed : friends.data.length; s < lenTweets; s++) { %>\n                    <% sharer = friends.data[s]; %>\n                        <figure class="digg-story--deeper__sharer-img-container js--digg-deeper__sharer-img-container">\n                            <a href="/u/<%= sharer.user_id %>/deeper" class="digg-story--deeper__sharer-link">\n                                <img class="digg-story--deeper__sharer-img js--digg-deeper-sharer-img" src="<%= sharer.profile_image_url %>" alt="Image of <%= sharer.name %>">\n                            </a>\n                            <figcaption class="digg-story--deeper__sharer-name">\n                                <!-- <%= sharer.name %> -->\n                            </figcaption>\n                        </figure>\n                        <% if (lenTweets === 1) { %>\n                            <span class="digg-story--deeper__single-sharer-label">Top story for <%= sharer.name %></span>\n                        <% } %>\n                <% } %>\n\n                <% if (tweets.count > maxSharersDisplayed) { %>\n                    <span class="<%= ~~tweets.count - maxSharersDisplayed > 10 ? \'digg-story--deeper__sharers-circle--sml\' : \'digg-story--deeper__sharers-circle\' %>">\n                        <span class="digg-story--deeper__sharers-total">+<%= ~~tweets.count - maxSharersDisplayed %></span>\n                    </span>\n                <% } %>\n            </div>\n        <% } else { %>\n            <div class="digg-story--deeper__sharers">\n                <% var sharer, tweet; %>\n                <% for (var s = 0, lenTweets = tweets.data.length > maxSharersDisplayed ? maxSharersDisplayed : tweets.data.length; s < lenTweets; s++) { %>\n                    <% tweet = tweets.data[s]; %>\n                    <% sharer = tweet.user; %>\n                        <figure class="digg-story--deeper__sharer-img-container js--digg-deeper__sharer-img-container">\n                            <span class="digg-story--deeper__sharer-link js--btn-show-digg-story-modal" data-screenname="<%= sharer.screen_name %>">\n                                <img class="digg-story--deeper__sharer-img" src="<%= sharer.profile_image_url %>" alt="Image of <%= sharer.name %>">\n                            </span>\n                            <% if ( ! tweets.hideTweetDetails) { %>\n                                <figcaption class="digg-story--deeper__sharer-tweet js--digg-deeper__sharer-details">\n                                    <img class="digg-story--deeper__sharer-img-detail" src="<%= sharer.profile_image_url %>" alt="Image of <%= sharer.name %>">\n                                    <div class="digg-story--deeper__sharer-tweet-details-label">From Twitter</div>\n                                    <span class="digg-story--deeper__sharer-tweet-name"><%= sharer.name %></span> <span class="digg-story--deeper__sharer-tweet-screenname">@<%= sharer.screen_name %></span> <a class="digg-story--deeper__sharer-tweet-link" href="<%= tweet.url %>" target="_blank"><%= tweet.fuzzyTimeStamp %></a>\n                                    <div class="digg-story--deeper__sharer-tweet-text"><%= tweet.text %></div>\n                                </figcaption>\n                            <% } %>\n                        </figure>\n                        <% if (lenTweets === 1) { %>\n                            <span class="digg-story--deeper__single-sharer-label"><%= sharer.name %></span>\n                        <% } %>\n                <% } %>\n\n                <% if (tweets.count > maxSharersDisplayed) { %>\n                    <span class="<%= ~~tweets.count - maxSharersDisplayed > 10 ? \'digg-story--deeper__sharers-circle--sml\' : \'digg-story--deeper__sharers-circle\' %>  js--btn-show-digg-story-modal">\n                        <span class="digg-story--deeper__sharers-total">+<%= ~~tweets.count - maxSharersDisplayed %></span>\n                    </span>\n                <% } %>\n            </div>\n        <% } %>\n    <% } %>\n\n    <div class="digg-story--deeper__content">\n        <header class="digg-story--deeper__header">\n            <h2 class="digg-story--deeper__title">\n                <a \n                    class="digg-story--deeper__title-link js--digg-story__link"\n                    href="<%= content.url %>"\n                    target="<%= content.link_target %>"\n                >\n                    <%= content.title %></a>\n            </h2>\n\n            <div class="digg-story--deeper__metadata-container">\n                <% if (showDiversityScore && digg_rank_score_display && digg_rank_score_display !== 0) { %>\n                    <span class="digg-story--deeper__metadata-score">\n                        <%= digg_rank_score_display %> \n                    </span>\n                <% } %>\n\n                <span class="digg-story--deeper__metadata--source">\n                    <a \n                        itemprop="publisher copyrightHolder sourceOrganization provider"\n                        class="digg-story--deeper__metadata--link js--digg-story__link"\n                        href="<%= content.url %>"\n                        target="<%= content.link_target %>"\n                    >\n                        <%= content.domain %>\n                    </a>\n                </span>\n\n                <% if (date !== null) { %>\n                    <abbr class="published digg-story--deeper__metadata--timestamp">\n                       <time><%= formattedDate %></time>\n                    </abbr>\n                <% } %>\n\n                <ul class="digg-story--deeper__actions--metadata-list">\n                    <li class="digg-story--deeper__action--digg js--digg-story-action--digg">\n                        <span class="js--digg-label digg-story--deeper__action-label"><%= dugg ? \'Dugg\' : \'Digg\' %></span>\n                    </li>\n                    <li class="digg-story--deeper__action--save js--digg-story-action--save">\n                        <span class="js--save-label digg-story--deeper__action-label"><%= read_later ? \'Saved\' : \'Save\' %></span>\n                    </li>\n                    <li class="digg-story--deeper__action--share js--digg-story-action--share">\n                        <div class="digg-story--deeper__share-tooltip">\n                            <ul class="digg-story--deeper__share-action-list">\n                                <li class="digg-story--deeper__share-action-item"><a href="<%= tweetlink %>" class="digg-story--deeper__share-link js--digg-story-action--tweet">Share to Twitter</a></li>\n                                <li class="digg-story--deeper__share-action-item"><a href="<%= fblink %>" class="digg-story--deeper__share-link js--digg-story-action--facebook">Share to Facebook</a></li>\n                            </ul>\n                        </div>\n                        \n                        <span class="js--share-label digg-story--deeper__action-label">Share</span>\n                    </li>\n\n                    <% if ( ! tweets.hideSharers && ! tweets.hideTweetDetails) { %>\n                        <li class="digg-story--deeper__action js--btn-show-digg-story-modal">\n                            <span class="digg-story--deeper__action-label">View Tweets</span>\n                        </li>\n                    <% } %>\n\n                </ul>\n            </div>\n        </header>\n\n        <% if (typeof noContentPreview === \'undefined\' || ! noContentPreview) { %>\n            <div class="digg-story--deeper__content-preview">\n                <% if (content.video) { %>\n                    <%= content.video %>\n                <% } else if (content.media.images.length && content.media.images[0].url && content.media.images[0].url.length && content.media.images[0].url.indexOf(\'http\') > -1) { %>\n                    <figure class="digg-story--deeper__img-container"><a class="digg-story--deeper__img-link js--digg-story__link"\n                            href="<%= content.url %>"\n                            target="<%= content.link_target %>"><img src="<%= content.media.images[0].url %>" alt="" class="digg-story--deeper__img js--digg-story__img"></a></figure>\n                <% } %>\n\n                <% if (content.description) { %>\n                    <p class="digg-story--deeper__description">\n                        <%= content.description %>\n                    </p>\n                <% } %>\n            </div>\n        <% } %>\n    </div>\n\n</article>\n'
    }), define("text!bs/templates/deeper/story_sharer_tweet.html", [], function() {
        return '<div class="digg-story--deeper__sharer story-embedded-share">\n    <a href="https://twitter.com/<%= sharer.screen_name %>" target="_blank" class="digg-story--deeper__sharer-link" data-screenname="<%= sharer.screen_name %>">\n        <img class="digg-story--deeper__sharer-img" src="<%= sharer.profile_image_url %>" alt="Image of <%= sharer.name %>" title="<%= sharer.name %>">\n    </a>\n    <div class="digg-story--deeper__sharer__body">\n        <div class="digg-story--deeper__sharer__user-info">\n            <a href="https://twitter.com/<%= sharer.screen_name %>" target="_blank" class="digg-story--deeper__sharer__name-link">\n                <%= sharer.name %>\n            </a>\n            <a href="https://twitter.com/<%= sharer.screen_name %>" target="_blank" class="digg-story--deeper__sharer__meta-link">\n                @<%= sharer.screen_name %>\n            </a>\n            <span class="digg-story--deeper__sharer__meta-text">&middot;</span>\n            <a href="<%= tweet.url %>" target="_blank" class="digg-story--deeper__sharer__meta-link">\n                <%= tweet.fuzzyTimeStamp.replace(\' ago\', \'\') %>\n            </a>\n        </div>\n\n        <div class="digg-story--deeper__sharer__text">\n            <%= tweet.text %>\n        </div>\n\n        <ul class="horizontal-list story-embedded-tweet-actions">\n            <li><a class="btn-tweet-reply digg-story--deeper__sharer__tweet-action-link" href="https://twitter.com/intent/tweet?in_reply_to=<%= tweet.id %>">Reply</a></li>\n            <li><a class="btn-tweet-retweet digg-story--deeper__sharer__tweet-action-link" href="https://twitter.com/intent/retweet?tweet_id=<%= tweet.id %>">Retweet</a></li>\n            <li><a class="btn-tweet-favorite digg-story--deeper__sharer__tweet-action-link" href="https://twitter.com/intent/favorite?tweet_id=<%= tweet.id %>">Favorite</a></li>\n        </ul>\n    </div>\n</div>\n'
    }), function(e, t) {
        typeof define == "function" && define.amd ? define("autolinker", t) : typeof exports != "undefined" ? module.exports = t() : e.Autolinker = t()
    }(this, function() {
        var e = function(t) {
            e.Util.assign(this, t)
        };
        return e.prototype = {
            constructor: e,
            urls: !0,
            email: !0,
            twitter: !0,
            newWindow: !0,
            stripPrefix: !0,
            className: "",
            htmlCharacterEntitiesRegex: /(&nbsp;|&#160;|&lt;|&#60;|&gt;|&#62;)/gi,
            matcherRegex: function() {
                var e = /(^|[^\w])@(\w{1,15})/, t = /(?:[\-;:&=\+\$,\w\.]+@)/, n = /(?:[A-Za-z]{3,9}:(?:\/\/)?)/, r = /(?:www\.)/, i = /[A-Za-z0-9\.\-]*[A-Za-z0-9\-]/, s = /\.(?:international|construction|contractors|enterprises|photography|productions|foundation|immobilien|industries|management|properties|technology|christmas|community|directory|education|equipment|institute|marketing|solutions|vacations|bargains|boutique|builders|catering|cleaning|clothing|computer|democrat|diamonds|graphics|holdings|lighting|partners|plumbing|supplies|training|ventures|academy|careers|company|cruises|domains|exposed|flights|florist|gallery|guitars|holiday|kitchen|neustar|okinawa|recipes|rentals|reviews|shiksha|singles|support|systems|agency|berlin|camera|center|coffee|condos|dating|estate|events|expert|futbol|kaufen|luxury|maison|monash|museum|nagoya|photos|repair|report|social|supply|tattoo|tienda|travel|viajes|villas|vision|voting|voyage|actor|build|cards|cheap|codes|dance|email|glass|house|mango|ninja|parts|photo|shoes|solar|today|tokyo|tools|watch|works|aero|arpa|asia|best|bike|blue|buzz|camp|club|cool|coop|farm|fish|gift|guru|info|jobs|kiwi|kred|land|limo|link|menu|mobi|moda|name|pics|pink|post|qpon|rich|ruhr|sexy|tips|vote|voto|wang|wien|wiki|zone|bar|bid|biz|cab|cat|ceo|com|edu|gov|int|kim|mil|net|onl|org|pro|pub|red|tel|uno|wed|xxx|xyz|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cu|cv|cw|cx|cy|cz|de|dj|dk|dm|do|dz|ec|ee|eg|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)\b/, o = /(?:[\-A-Za-z0-9+&@#\/%?=~_()|!:,.;]*[\-A-Za-z0-9+&@#\/%=~_()|])?/;
                return new RegExp(["(", e.source, ")", "|", "(", t.source, i.source, s.source, ")", "|", "(", "(?:", "(?:", n.source, i.source, ")", "|", "(?:", "(.?//)?", r.source, i.source, ")", "|", "(?:", "(.?//)?", i.source, s.source, ")", ")", o.source, ")"].join(""), "gi")
            }(),
            invalidProtocolRelMatchRegex: /^[\w]\/\//,
            charBeforeProtocolRelMatchRegex: /^(.)?\/\//,
            link: function(t) {
                var n = this, r = this.getHtmlParser(), i = this.htmlCharacterEntitiesRegex, s = 0, o = [];
                return r.parse(t, {
                    processHtmlNode: function(e, t, n) {
                        t === "a" && (n ? s = Math.max(s - 1, 0) : s++), o.push(e)
                    },
                    processTextNode: function(t) {
                        if (s === 0) {
                            var r = e.Util.splitAndCapture(t, i);
                            for (var u = 0, a = r.length; u < a; u++) {
                                var f = r[u], l = n.processTextNode(f);
                                o.push(l)
                            }
                        } else 
                            o.push(t)
                    }
                }), o.join("")
            },
            getHtmlParser: function() {
                var t = this.htmlParser;
                return t || (t = this.htmlParser = new e.HtmlParser), t
            },
            getTagBuilder: function() {
                var t = this.tagBuilder;
                return t || (t = this.tagBuilder = new e.AnchorTagBuilder({
                    newWindow: this.newWindow,
                    truncate: this.truncate,
                    className: this.className
                })), t
            },
            processTextNode: function(t) {
                var n = this, r = this.charBeforeProtocolRelMatchRegex;
                return t.replace(this.matcherRegex, function(t, i, s, o, u, a, f, l) {
                    var c = i, h = s, p = o, d = u, v = a, m = f || l, g = "", y = "", b;
                    if (!n.isValidMatch(c, d, v, m))
                        return t;
                    n.matchHasUnbalancedClosingParen(t) && (t = t.substr(0, t.length - 1), y = ")");
                    if (d)
                        b = new e.match.Email({
                            matchedText: t,
                            email: d
                        });
                    else if (c)
                        h && (g = h, t = t.slice(1)), b = new e.match.Twitter({
                            matchedText: t,
                            twitterHandle: p
                        });
                    else {
                        if (m) {
                            var w = m.match(r)[1] || "";
                            w && (g = w, t = t.slice(1))
                        }
                        b = new e.match.Url({
                            matchedText: t,
                            url: t,
                            protocolRelativeMatch: m,
                            stripPrefix: n.stripPrefix
                        })
                    }
                    var E = n.createMatchReturnVal(b, t);
                    return g + E + y
                })
            },
            isValidMatch: function(e, t, n, r) {
                return e&&!this.twitter || t&&!this.email || n&&!this.urls || n && n.indexOf(".")===-1 || n && /^[A-Za-z]{3,9}:/.test(n)&&!/:.*?[A-Za-z]/.test(n) || r && this.invalidProtocolRelMatchRegex.test(r)?!1 : !0
            },
            matchHasUnbalancedClosingParen: function(e) {
                var t = e.charAt(e.length - 1);
                if (t === ")") {
                    var n = e.match(/\(/g), r = e.match(/\)/g), i = n && n.length || 0, s = r && r.length || 0;
                    if (i < s)
                        return !0
                }
                return !1
            },
            createMatchReturnVal: function(t, n) {
                var r;
                this.replaceFn && (r = this.replaceFn.call(this, this, t));
                if (typeof r == "string")
                    return r;
                if (r===!1)
                    return n;
                if (r instanceof e.HtmlTag)
                    return r.toString();
                var i = this.getTagBuilder(), s = i.build(t);
                return s.toString()
            }
        }, e.link = function(t, n) {
            var r = new e(n);
            return r.link(t)
        }, e.match = {}, e.Util = {
            abstractMethod: function() {
                throw "abstract"
            },
            assign: function(e, t) {
                for (var n in t)
                    t.hasOwnProperty(n) && (e[n] = t[n]);
                return e
            },
            extend: function(t, n) {
                var r = t.prototype, i = function() {};
                i.prototype = r;
                var s;
                n.hasOwnProperty("constructor") ? s = n.constructor : s = function() {
                    r.constructor.apply(this, arguments)
                };
                var o = s.prototype = new i;
                return o.constructor = s, o.superclass = r, delete n.constructor, e.Util.assign(o, n), s
            },
            ellipsis: function(e, t, n) {
                return e.length > t && (n = n == null ? ".." : n, e = e.substring(0, t - n.length) + n), e
            },
            indexOf: function(e, t) {
                if (Array.prototype.indexOf)
                    return e.indexOf(t);
                for (var n = 0, r = e.length; n < r; n++)
                    if (e[n] === t)
                        return n;
                return - 1
            },
            splitAndCapture: function(e, t) {
                if (!t.global)
                    throw new Error("`splitRegex` must have the 'g' flag set");
                var n = [], r = 0, i;
                while (i = t.exec(e))
                    n.push(e.substring(r, i.index)), n.push(i[0]), r = i.index + i[0].length;
                return n.push(e.substring(r)), n
            }
        }, e.HtmlParser = e.Util.extend(Object, {
            htmlRegex: function() {
                var e = /[0-9a-zA-Z:]+/, t = /[^\s\0"'>\/=\x01-\x1F\x7F]+/, n = /(?:".*?"|'.*?'|[^'"=<>`\s]+)/, r = t.source + "(?:\\s*=\\s*" + n.source + ")?";
                return new RegExp(["<(?:!|(/))?", "(" + e.source + ")", "(?:", "\\s+", "(?:", r, "|", n.source + ")", ")*", "\\s*/?", ">"].join(""), "g")
            }(),
            parse: function(e, t) {
                t = t || {};
                var n = t.processHtmlNode || function() {}, r = t.processTextNode || function() {}, i = this.htmlRegex, s, o = 0;
                while ((s = i.exec(e)) !== null) {
                    var u = s[0], a = s[2], f=!!s[1], l = e.substring(o, s.index);
                    l && r(l), n(u, a, f), o = s.index + u.length
                }
                if (o < e.length) {
                    var c = e.substring(o);
                    c && r(c)
                }
            }
        }), e.HtmlTag = e.Util.extend(Object, {
            whitespaceRegex: /\s+/,
            constructor: function(t) {
                e.Util.assign(this, t), this.innerHtml = this.innerHtml || this.innerHTML
            },
            setTagName: function(e) {
                return this.tagName = e, this
            },
            getTagName: function() {
                return this.tagName || ""
            },
            setAttr: function(e, t) {
                var n = this.getAttrs();
                return n[e] = t, this
            },
            getAttr: function(e) {
                return this.getAttrs()[e]
            },
            setAttrs: function(t) {
                var n = this.getAttrs();
                return e.Util.assign(n, t), this
            },
            getAttrs: function() {
                return this.attrs || (this.attrs = {})
            },
            setClass: function(e) {
                return this.setAttr("class", e)
            },
            addClass: function(t) {
                var n = this.getClass(), r = this.whitespaceRegex, i = e.Util.indexOf, s = n ? n.split(r): [], o = t.split(r), u;
                while (u = o.shift())
                    i(s, u)===-1 && s.push(u);
                return this.getAttrs()["class"] = s.join(" "), this
            },
            removeClass: function(t) {
                var n = this.getClass(), r = this.whitespaceRegex, i = e.Util.indexOf, s = n ? n.split(r): [], o = t.split(r), u;
                while (s.length && (u = o.shift())) {
                    var a = i(s, u);
                    a!==-1 && s.splice(a, 1)
                }
                return this.getAttrs()["class"] = s.join(" "), this
            },
            getClass: function() {
                return this.getAttrs()["class"] || ""
            },
            hasClass: function(e) {
                return (" " + this.getClass() + " ").indexOf(" " + e + " ")!==-1
            },
            setInnerHtml: function(e) {
                return this.innerHtml = e, this
            },
            getInnerHtml: function() {
                return this.innerHtml || ""
            },
            toString: function() {
                var e = this.getTagName(), t = this.buildAttrsStr();
                return t = t ? " " + t : "", ["<", e, t, ">", this.getInnerHtml(), "</", e, ">"].join("")
            },
            buildAttrsStr: function() {
                if (!this.attrs)
                    return "";
                var e = this.getAttrs(), t = [];
                for (var n in e)
                    e.hasOwnProperty(n) && t.push(n + '="' + e[n] + '"');
                return t.join(" ")
            }
        }), e.AnchorTagBuilder = e.Util.extend(Object, {
            constructor: function(t) {
                e.Util.assign(this, t)
            },
            build: function(t) {
                var n = new e.HtmlTag({
                    tagName: "a",
                    attrs: this.createAttrs(t.getType(), t.getAnchorHref()),
                    innerHtml: this.processAnchorText(t.getAnchorText())
                });
                return n
            },
            createAttrs: function(e, t) {
                var n = {
                    href: t
                }, r = this.createCssClass(e);
                return r && (n["class"] = r), this.newWindow && (n.target = "_blank"), n
            },
            createCssClass: function(e) {
                var t = this.className;
                return t ? t + " " + t + "-" + e : ""
            },
            processAnchorText: function(e) {
                return e = this.doTruncate(e), e
            },
            doTruncate: function(t) {
                return e.Util.ellipsis(t, this.truncate || Number.POSITIVE_INFINITY)
            }
        }), e.match.Match = e.Util.extend(Object, {
            constructor: function(t) {
                e.Util.assign(this, t)
            },
            getType: e.Util.abstractMethod,
            getMatchedText: function() {
                return this.matchedText
            },
            getAnchorHref: e.Util.abstractMethod,
            getAnchorText: e.Util.abstractMethod
        }), e.match.Email = e.Util.extend(e.match.Match, {
            getType: function() {
                return "email"
            },
            getEmail: function() {
                return this.email
            },
            getAnchorHref: function() {
                return "mailto:" + this.email
            },
            getAnchorText: function() {
                return this.email
            }
        }), e.match.Twitter = e.Util.extend(e.match.Match, {
            getType: function() {
                return "twitter"
            },
            getTwitterHandle: function() {
                return this.twitterHandle
            },
            getAnchorHref: function() {
                return "https://twitter.com/" + this.twitterHandle
            },
            getAnchorText: function() {
                return "@" + this.twitterHandle
            }
        }), e.match.Url = e.Util.extend(e.match.Match, {
            urlPrefixRegex: /^(https?:\/\/)?(www\.)?/i,
            protocolRelativeRegex: /^\/\//,
            checkForProtocolRegex: /^[A-Za-z]{3,9}:/,
            getType: function() {
                return "url"
            },
            getUrl: function() {
                var e = this.url;
                return !this.protocolRelativeMatch&&!this.checkForProtocolRegex.test(e) && (e = this.url = "http://" + e), e
            },
            getAnchorHref: function() {
                var e = this.getUrl();
                return e.replace(/&amp;/g, "&")
            },
            getAnchorText: function() {
                var e = this.getUrl();
                return this.protocolRelativeMatch && (e = this.stripProtocolRelativePrefix(e)), this.stripPrefix && (e = this.stripUrlPrefix(e)), e = this.removeTrailingSlash(e), e
            },
            stripUrlPrefix: function(e) {
                return e.replace(this.urlPrefixRegex, "")
            },
            stripProtocolRelativePrefix: function(e) {
                return e.replace(this.protocolRelativeRegex, "")
            },
            removeTrailingSlash: function(e) {
                return e.charAt(e.length - 1) === "/" && (e = e.slice(0, - 1)), e
            }
        }), e
    }), define("bs/widgets/digg_deeper_story_modal", ["bs/core", "bs/util/xhr", "bs/util/format", "bs/util/helpers", "text!bs/templates/deeper/story_modal.html", "text!bs/templates/deeper/story.html", "text!bs/templates/deeper/story_sharer_tweet.html", "autolinker", "lodash", "bootstrap_modal", "jquery"], function(e, t, n, r, i, s, o, u, a, f, l) {
        function y(e, t) {
            var n = {
                content_id: e,
                full_text: !0
            };
            m.request({
                data: n,
                success: function(e) {
                    if (!e ||!e.data ||!e.data.feed ||!e.data.feed.length)
                        return console.error("unexpected response format:", e), t("failed");
                    t(null, e.data.feed[0])
                },
                error: function(e, n, r) {
                    console.error("failed to get full alert:", n, r), t(r)
                }
            })
        }
        function b(e, t) {
            c.removeClass("is--loading");
            if (e)
                return;
            var n = t.tweets.data.splice(10);
            n = w(n), l(E(n)).appendTo(l(".modal--digg-deeper_sharers"))
        }
        function w(e) {
            return a.each(e, function(e) {
                e.url = r.getTweetURL({
                    twitter_screen_name: e.user.screen_name,
                    tweet_id: e.id
                }), e.fuzzyTimeStamp = n.secondsToFuzzyTime(e.date), e.text = u.link(e.text)
            }), e
        }
        function E(e) {
            return a.map(e, function(e) {
                return v({
                    tweet: e,
                    sharer: e.user
                })
            }).join("\n")
        }
        function S(t, i, s) {
            return t.position = i, t.formattedDate = n.secondsToFuzzyTime(t.date), t.content.title || (t.content.title = t.tweets.data[0].text), t.content.link_target = "_blank", t.fblink = r.getFacebookStoryShareLink({
                app_id: e.config.FB_APP_ID,
                url: t.content.url
            }), t.tweetlink = r.getTweetStoryLink({
                url: t.content.url,
                title: t.content.title || t.tweets[0].text
            }), !t.content.media.videos.length && t.content.media.images.length && (t.content.description = n.truncate(t.content.description, g, !0)), t.tweets.hideSharers=!0, t.noContentPreview=!s, d(t)
        }
        var c, h = {}, p = a.template(i), d = a.template(s), v = a.template(o), m = new t.dataProvider({
            url: "/api/news/alert.json",
            cache: !1
        }), g = 135, x = {
            saveStoryDetails: function(e) {
                a.each(e, function(e) {
                    e.tweets.data = w(e.tweets.data), h[e.story_id] = e
                })
            },
            show: function(e, t) {
                var n = l(e.target).closest(".digg-story-el, .digg-story--deeper"), r = n.data("content-id"), i = n.data("position"), s = h[r];
                if (!s)
                    return;
                var o = p({
                    storyDetailsHTML: S(s, i, t),
                    sharersHTML: E(s.tweets.data),
                    sharerCount: s.tweets.count
                });
                return c = l(o).appendTo("body").modal({
                    keyboard: !0,
                    show: !0
                }), l(document).on("click touchend", x.hide), s.tweets.count > 10 && (c.addClass("is--loading"), y(r, b)), !1
            },
            hide: function(e) {
                var t = l(e.target).hasClass("js--btn-modal-close"), n = l(e.target).closest(".js--modal--digg-deeper-story").length > 0;
                if (t ||!n)
                    l(".js--modal--digg-deeper-story").modal("hide").remove(), l(document).off("click touchend", x.hide)
            }
        };
        return x
    }), define("text!bs/templates/deeper/stream_container_frontpage.html", [], function() {
        return '<header class="widget-hdr">\n    <h3 class="widget-hdr__title"><a class="widget-hdr__link" href="/deeper">Digg Deeper</a></h3>\n    <a class="widget-hdr__link--view-all" href="/deeper">View All</a>\n    <span id="alerts-sortby">\n        <select class="digg-deeper__sort-select js--digg-deeper-sort-select" dir="rtl">\n            <option value="score">Sort by: Most Shared</option>\n            <option value="date">Sort by: Time</option>\n        </select>\n    </span>\n</header>\n<div \n    id="digg-deeper" \n    class="stories-container" \n    data-tracking-section="Frontpage: Digg Deeper"\n>\n    <div class="ico-loading alerts-first-page-loading"></div>\n</div>\n'
    }), define("text!bs/templates/alerts/onboarding.html", [], function() {
        return '<div class="btn-alerts-close btn-alerts-opt-out">✖</div>\n<h3 class="alerts-hdr">Digg Deeper</h3>\n<p class="alerts-desc">Digg now shows you the <span class="nowrap">most-shared</span> stories from your Twitter friends.</p>\n<p><a href="http://blog.digg.com/post/91454524841/digg-deeper" target="_blank">Learn how it works</a></p>\n<p><button class="btn-primary btn-enable-alerts"><%= isUserLoggedIn && hasTwitter ? \'Turn On\' : \'Sign In with Twitter\' %></button></p>\n'
    }), define("text!bs/templates/alerts/email_onboarding.html", [], function() {
        return '<p class="alerts-desc first-el">You\'ll see your most-shared stories from Twitter here in a moment.</p>\n<p class="alerts-desc">Want to also get email alerts for these stories?</p>\n<p><input type="email" id="input-alerts-email" placeholder="you@domain.com" value="<%= email %>">\n<button class="btn-primary btn-enable-alerts-email">Enter</button></p>\n<p class="btn-alerts-email-skip">Skip this step</p>\n'
    }), define("text!bs/templates/alerts/email_onboarding_details.html", [], function() {
        return '<div class="email-details-container">\n\n    <p class="alerts-desc first-el">What emails would you like to receive?</p>\n\n    <div class="email-details">\n        <p><label><input type="radio" name="email-setting" value="alerts">Real-time alerts<br>\n            <span class="note">Up to 5 emails/day</span></label>\n        </p>\n        <p><label><input type="radio" name="email-setting" value="digest">Daily email summary</label></p>\n        <p><label><input type="radio" name="email-setting" value="both" checked>Both</label></p>\n    </div>\n\n    <p><button class="btn-primary btn-alerts-email-details-continue">Continue</button></p>\n\n    <input type="hidden" id="input-alerts-email" value="<%= email %>">\n\n</div>\n'
    }), define("bs/widgets/alerts", ["bs/core", "bs/util/xhr", "bs/util/format", "bs/util/cookies", "bs/util/helpers", "bs/util/scrollmonitor", "bs/util/storyactions_v3", "bs/widgets/digg_deeper_story_modal", "text!bs/templates/deeper/stream_container_frontpage.html", "text!bs/templates/deeper/story.html", "text!bs/templates/alerts/onboarding.html", "text!bs/templates/alerts/email_onboarding.html", "text!bs/templates/alerts/email_onboarding_details.html", "lodash", "jquery"], function(e, t, n, r, i, s, o, u, a, f, l, c, h, p, d) {
        function L(t, r) {
            return t.position = r, t.formattedDate = n.secondsToFuzzyTime(t.date), t.content.title || (t.content.title = t.tweets.data[0].text), t.content.displayTitle = n.truncate(t.content.title, g), t.content.link_target = "_blank", t.fblink = i.getFacebookStoryShareLink({
                app_id: e.config.FB_APP_ID,
                url: t.content.url
            }), t.tweetlink = i.getTweetStoryLink({
                url: t.content.url,
                title: t.content.title || t.tweets[0].text
            }), p.each(t.tweets.data, function(e) {
                e.url = i.getTweetURL({
                    twitter_screen_name: e.user.screen_name,
                    tweet_id: e.id
                }), e.fuzzyTimeStamp = n.secondsToFuzzyTime(e.date)
            }), t.noContentPreview=!0, t.maxSharersDisplayed = 6, t.feed = "digg_deeper", t.showDiversityScore=!1, y(t)
        }
        function A(e) {
            var t = d(this), n = this.scrollTop, r = this.scrollHeight, i = t.height(), s = e.type == "DOMMouseScroll" ? e.originalEvent.detail*-40: e.originalEvent.wheelDelta, o = s > 0, u = function() {
                return e.stopPropagation(), e.preventDefault(), e.returnValue=!1, !1
            };
            if (t.hasClass("no-alerts-pagination"))
                return !0;
            if (!o&&-s > r - i - n)
                return t.scrollTop(r), u();
            if (o && s > n)
                return t.scrollTop(0), u()
        }
        function O(e) {
            e = e || {};
            if (!e.id)
                throw new Error("Missing id selector for Alerts widget");
            this.id = e.id, this.el = document.getElementById(this.id), this.$el = d(this.el), e.elHeight && this.setElHeight(e.elHeight)
        }
        var v = "We're working on finding the best stories from your Twitter stream. While you're waiting, check out some of our most dugg stories.", m = "twitter-news-optout", g = 75, y = p.template(f), b = 0, w, E, S = require("bs/core").user, x, T = new t.dataProvider({
            url: "/api/news/alerts/top.json",
            cache: !1
        }), N, C = new t.dataProvider({
            url: "/api/news/popular.json",
            data: {
                count: 10,
                exclude_sponsored: 10,
                exclude_videos: 10
            },
            success: function(e) {
                N = e.data.feed
            },
            error: function(e, t, n) {}
        }), k = C.request();
        return O.prototype = {
            alertsOnboardingTemplate: p.template(l),
            emailOnboardingTemplate: p.template(c),
            emailOnboardingDetailsTemplate: p.template(h),
            init: function(t) {
                var n = r.get(m);
                if (!this.el)
                    return;
                e.mediaQueriesMonitor.addQuery("(max-width: 480px)", p.bind(this.handlePhoneLayout, this)), !S.isLoggedIn()&&!n ||!S.deepGet("apps.digg_deeper.enabled") ? (e.pubsub.fire("digg_deeper:onboarding:start"), this.renderOnboarding(), S.isLoggedIn() || e.pubsub.one("site_nav_user_widget", e.pubsub.channels.user_auth, p.bind(function() {
                    S.deepGet("apps.digg_deeper.enabled") ? this.renderDiggDeeper() : this.renderOnboarding()
                }, this))) : S.deepGet("apps.digg_deeper.enabled") && this.renderDiggDeeper()
            },
            handlePhoneLayout: function() {},
            renderDiggDeeper: function() {
                this.el.className = "alerts-enabled", this.el.innerHTML = "", x = x || S.get("deeper_sortby") || "score", this.renderStreamContainer(), this.setStreamElHeight(), this.$el.find("#digg-deeper").show(1), this.fetch(), this.$el.on("click", ".digg-story--deeper__sharer-link", p.bind(this.handleStorySharerClick, this)).on("click", ".js--btn-show-digg-story-modal", p.bind(this.showDeeperModal, this)).on("click", ".ico-settings", p.bind(this.handleSettingsClick, this)).on("click", ".js--digg-story-action--tweet, .js--digg-story-action--facebook", o.shareStory).on("mouseover", ".js--digg-deeper__sharer-img-container", p.bind(this.positionTooltip, this)), d(document).on("click", ".js--digg-story__link", o.redirectStoryLink).on("click", ".js--digg-story-action--digg", e.user.authorized(o.submitDigg, "Digg")).on("click", ".js--digg-story-action--save", e.user.authorized(o.addReadingList, "Save"))
            },
            positionTooltip: function(e) {
                var t = d(e.currentTarget), n = t.find(".js--digg-deeper__sharer-details"), r = t.position();
                n.css({
                    top: r.top
                })
            },
            handleStorySharerClick: function(t) {
                var n = d(t.currentTarget).data("screenname") || "Unknown";
                e.pubsub.fire("digg_deeper:stream:sharer_click", n)
            },
            showDeeperModal: function(e) {
                e.stopPropagation(), e.preventDefault(), u.show(e, !0)
            },
            handleSettingsClick: function(t) {
                e.pubsub.fire("digg_deeper:stream:settings_click")
            },
            setElHeight: function(e) {
                var t = parseInt(this.$el.css("padding-top"), 10), n = parseInt(this.$el.css("padding-bottom"), 10);
                this.$el.height(e - t - n), this.$el.hasClass("alerts-enabled") && this.setStreamElHeight()
            },
            fetch: function() {
                var e;
                e = {
                    sortby: x,
                    full_text: !0
                }, E != null && (e.position = E), T.request({
                    data: e,
                    success: p.bind(this.checkFeedResponse, this),
                    error: p.bind(this.handleStreamRequestError, this)
                })
            },
            getNextPage: function(t) {
                E !== "-1" && w !== E && (this.$el.find("#digg-deeper").append('<div class="ico-loading alerts-page-loading"></div>'), w = E, this.fetch(), e.pubsub.fire("loading_in_progress"))
            },
            handleStreamRequestError: function(e, t, n) {
                this.renderImportFailed(v)
            },
            checkFeedResponse: function(t, n, r) {
                var i = t.data.polling.status, s=~~t.data.polling.interval * 1e3, o = t.status_message || v;
                i === 0 ? (E = t.data.next_position, b++, e.pubsub.fire("digg_deeper:stream:loaded", x, b), e.pubsub.fire("loading_in_done"), u.saveStoryDetails(t.data.feed), this.renderStories(t)) : i===-1 && w == null ? this.renderImportFailed(o) : i === 1 && setTimeout(p.bind(this.fetch, this), s)
            },
            renderImportFailed: function(e) {
                var t = this.$el.find("#digg-deeper");
                this.$el.find(".ico-loading").remove(), t.hide(), t.append('<div class="alerts-message">' + e + "</div>"), this.renderMostDuggStories(5), this.$el.find("#digg-deeper").fadeIn(800)
            },
            renderStreamContainer: function() {
                document.getElementById("digg-alerts") || (this.el.className = "alerts-enabled", this.el.innerHTML = a, d(".js--digg-deeper-sort-select").find("option[value=" + x + "]").attr("selected", "selected"), this.$el.on("change", ".js--digg-deeper-sort-select", p.bind(this.selectSort, this)).on("mouseenter", ".more-shares", p.bind(this.showMoreSharersTooltip, this)).on("mouseleave", ".more-shares", p.bind(this.hideMoreSharersTooltip, this)))
            },
            selectSort: function(t) {
                x = d(".js--digg-deeper-sort-select").val(), b = 0, E = null, w = null, this.$el.find("#digg-deeper").html('<div class="ico-loading alerts-first-page-loading"></div>'), e.pubsub.fire("digg_deeper:stream:sortby_selected", x), this.fetch()
            },
            setStreamElHeight: function() {
                var e = d(".no-marquee").size() ? 5: 15;
                this.$el.find("#digg-deeper").height(this.$el.height() - this.$el.find(".widget-hdr").outerHeight(!0) + e)
            },
            renderStories: function(t) {
                var n = t.data.feed, r = t.status_message, i = p.map(n, L), o = this.$el.find(".story-container").size(), u = this.$el.find("#digg-deeper");
                this.scrollPanel || (this.scrollPanel = u[0], d(document).on("DOMMouseScroll mousewheel", "#digg-deeper", A)), u.find(".ico-loading").remove(), o ? u.append(i.join("")) : (u.hide(), u.append(i.join("")).fadeIn(800)), r && w == null && (u.append('<div class="alerts-message">' + r + "</div>"), k.done(p.bind(this.renderMostDuggStories, this, 5))), this.scrollmonitor || (this.scrollmonitor = s.factory(this.scrollPanel, {
                    name: "digg_deeper",
                    pubsub: e.pubsub
                }), e.pubsub.on("digg_deeper", "digg_deeper:scroll:bottom", p.bind(this.getNextPage, this)))
            },
            showMoreSharersTooltip: function(e) {
                var t = e.currentTarget;
                d(t).addClass("tooltip-show"), this.hideMoreSharersProxy = p.bind(this.hideMoreSharersTooltip, this), this.$el.on("mouseleave", ".btn-show-more-shares", this.hideMoreSharersProxy)
            },
            hideMoreSharersTooltip: function(e) {
                var t = e.currentTarget;
                d(t).removeClass("tooltip-show")
            },
            renderMostDuggStories: function(t) {
                var r = [], s = 0, o;
                for (; s < t; s++)
                    o = N[s], o.position = s, o.formattedDate = n.secondsToFuzzyTime(o.date), o.content.link_target = "_blank", o.fblink = i.getFacebookStoryShareLink({
                        app_id: e.config.FB_APP_ID,
                        content_id: o.content.content_id
                    }), o.tweetlink = i.getTweetStoryLink({
                        content_id: o.content.content_id,
                        title: o.content.title || o.tweets[0].text
                    }), o.feed = "most-dugg", o.maxSharersDisplayed = 1, o.noContentPreview=!0, o.showDiversityScore=!1, o.tweets.count = 1, o.tweets.hideTweetDetails=!0, o.tweets.data = [{
                        user: {
                            name: "Top story on Digg",
                            profile_image_url: "https://pbs.twimg.com/profile_images/529093119665397760/W5XVfjAs_400x400.jpeg",
                            screen_name: "digg",
                            user_id: "15163466"
                        }
                    }
                    ], r.push(y(o));
                this.$el.find("#digg-deeper").append(r.join(""))
            },
            renderOnboarding: function() {
                var t = e.user.get("twitter"), n = p.bind(this.beginOnboarding, this);
                this.el.className = "onboarding", this.el.innerHTML = this.alertsOnboardingTemplate({
                    isUserLoggedIn: e.user.isLoggedIn(),
                    hasTwitter: t && t.username
                }), d(this.el).on("click", ".btn-enable-alerts", S.authorized(n, "digg_deeper", {
                    twitterOnly: !0
                })).on("click", ".btn-alerts-opt-out", p.bind(this.optOut, this, "Twitter alerts"))
            },
            optOut: function(t) {
                e.pubsub.fire("digg_deeper:onboarding:opt_out", t), r.set({
                    name: m,
                    value: 1,
                    expires: r.expires.oneYear(),
                    path: "/"
                }), p.defer(function() {
                    window.location.reload(!0)
                })
            },
            beginOnboarding: function() {
                S.deepGet("apps.digg_deeper.enabled") ? this.$el.find("#digg-deeper").size() || this.renderDiggDeeper() : this.enableTwitterAlerts()
            },
            renderEmailOnboarding: function() {
                this.el.innerHTML = this.emailOnboardingTemplate({
                    email: e.user.get("email") || ""
                }), d("#input-alerts-email").focus(), d(this.el).on("click", ".btn-enable-alerts-email", p.bind(this.alertsEmailDetails, this)).on("click", ".btn-alerts-email-skip", p.bind(this.renderOnboardingSuccess, this, {
                    emailOptOut: !0
                })).on("keyup", "#input-alerts-email", p.bind(this.alertsEmailDetailsProxy, this))
            },
            renderEmailDetailsOnboarding: function(e) {
                this.el.innerHTML = this.emailOnboardingDetailsTemplate({
                    email: e
                }), d("input[name=email-setting]").last().focus(), d(this.el).on("keyup", "input[name=email-setting]", p.bind(this.alertsEmailProxy, this)).on("click", ".btn-alerts-email-details-continue", p.bind(this.enableAlertsEmail, this))
            },
            alertsEmailDetailsProxy: function(e) {
                e.keyCode === 13 && (e.stopPropagation(), e.preventDefault(), this.alertsEmailDetails())
            },
            alertsEmailDetails: function(e) {
                var t = d("#input-alerts-email"), n = t.val();
                this.renderEmailDetailsOnboarding(n)
            },
            alertsEmailProxy: function(e) {
                e.keyCode === 13 && (e.stopPropagation(), e.preventDefault(), this.enableAlertsEmail())
            },
            enableAlertsEmail: function() {
                var n = d("#input-alerts-email"), r = n.val(), i = d('input[name="email-setting"]:checked').val(), s = 0, o=!1;
                if (i === "both" || i === "digest")
                    o=!0;
                if (i === "both" || i === "alerts")
                    s = 2;
                var u = new t.dataProvider({
                    url: "/api/settings/email.json",
                    type: "POST",
                    success: p.bind(this.renderOnboardingSuccess, this),
                    error: p.bind(this.enableTwitterAlertsError),
                    data: {
                        source: "onboarding",
                        email: r,
                        "apps.digg_deeper.alerts.email.frequency": s,
                        "apps.digg_deeper.digest.email.enabled": o
                    }
                });
                r ? (e.pubsub.fire("digg_deeper:onboarding:email_opt_in"), u.request()) : n.addClass("input-error")
            },
            renderOnboardingSuccess: function(t) {
                if (this.onboarded)
                    return;
                this.onboarded=!0, t && t.emailOptOut && e.pubsub.fire("digg_deeper:onboarding:opt_out", "Email alerts"), e.pubsub.fire("digg_deeper:onboarding:success"), r.set({
                    name: m,
                    value: 1,
                    expires: r.expires.oneYear(),
                    path: "/"
                }), this.el.innerHTML = '<div class="ico-loading onboarding-success"></div><p class="alerts-desc">Almost there...</p>', this.renderDiggDeeper()
            },
            enableTwitterAlertsSuccess: function() {
                S.deepGet("apps.digg_deeper.enabled") || setTimeout(p.bind(this.renderEmailOnboarding, this), 3500)
            },
            enableTwitterAlertsError: function() {
                this.el.className = "alerts-enabled", this.el.innerHTML = "<p class=\"alerts-desc\">Oops! We're having trouble Digging Deeper. Why don't you refresh the page and try again.</p>"
            },
            enableTwitterAlerts: function() {
                var n = new t.dataProvider({
                    url: "/api/settings/digg_deeper.json",
                    type: "POST",
                    success: p.bind(this.enableTwitterAlertsSuccess, this),
                    error: p.bind(this.enableTwitterAlertsError, this),
                    data: {
                        source: "onboarding",
                        enabled: !0
                    }
                });
                n.request(), e.pubsub.fire("digg_deeper:onboarding:opt_in"), this.el.innerHTML = '<div class="ico-loading onboarding-success"></div><p class="alerts-desc">Here we go!<br>Compiling stories from your Twitter timeline...</p>'
            }
        }, {
            factory: function(e) {
                return new O(e)
            }
        }
    }), require(["bs/core", "bs/util/storyactions2", "bs/util/storyscores", "bs/util/tracking", "bs/util/cookies", "bs/widgets/lazyimages", "bs/widgets/usernav", "bs/widgets/savedstories2", "bs/widgets/headersearch2", "bs/widgets/mobileheader", "bs/widgets/promobanner", "bs/widgets/popularvideos", "bs/widgets/popularstories", "bs/widgets/upcomingstories", "bs/widgets/footer", "bs/widgets/onboarding", "bs/widgets/alerts", "lodash"], function(e, t, n, r, i, s, o, u, a, f, l, c, h, p, d, v, m, g) {
        function y(t, n) {
            var r = document.getElementById("top-stories").getBoundingClientRect(), i = 100;
            if (n || t >= r.bottom - i)
                c.init(), h.init(), p.init(), $(".ftr-ios-hero-img").addClass("lazy-loaded"), e.pubsub.off("scroll", e.pubsub.channels.scrolltop, y)
        }
        function b() {
            var e = document.getElementById("popular-videos").getBoundingClientRect(), t = $(document).scrollTop();
            y(t, !0), t < e.top && $("html,body").animate({
                scrollTop: e.bottom - 25
            }, 1e3)
        }
        function S() {
            function s() {
                var e = [{
                    src: "/static/audio/rickLoop.mp3",
                    type: "audio/mp3"
                }, {
                    src: "/static/audio/rickLoop.ogg",
                    type: "audio/ogg"
                }
                ];
                r = document.createElement("audio");
                if (!r.canPlayType)
                    return;
                g.each(e, function(e, t) {
                    var n = document.createElement("source");
                    n.setAttribute("src", e.src), n.setAttribute("type", e.type), r.appendChild(n)
                }), $("body").append(r), r.volume = 0, r.addEventListener("ended", function() {
                    u()
                }, !1), r.play(), o()
            }
            function o() {
                if (i >= 1)
                    return;
                r.volume = i, i += .1, setTimeout(o, 500)
            }
            function u() {
                var e;
                for (var t = 0; t < 7; t++)
                    e = $(n[t]), e.attr("src", e.attr("data-src"))
            }
            function a() {
                var e, r;
                if (!t.length)
                    return;
                r = t.pop(), e = $(n[r - 1]), e.addClass("img-rr"), e.css({
                    opacity: 0
                }), e.attr("data-src", e.attr("src")), e.attr("src", "/static/images/rr/" + r + "-png.jpg"), e.fadeTo(1e3, 1), r === 3 ? (setTimeout(a, 5e3), s()) : a()
            }
            function f() {
                var e = [], n;
                g.each(t, function(t, n) {
                    e[n] = {
                        img: new Image,
                        done: new $.Deferred
                    }, e[n].img.onload = function() {
                        e[n].done.resolve()
                    }, e[n].src = "/static/images/rr/" + n + "-png.jpg"
                }), n = g.map(t, function(e, t) {
                    return e.done
                }), $.when.apply(this, n).then(a)
            }
            var t = [2, 3, 1, 4, 5, 6, 7].reverse(), n = $(".story-image-img"), r, i = 0;
            f(), e.pubsub.fire("spreadgun", "http://digg.com/")
        }
        var w, E = {
            init: function() {
                var r = $("#marquee").find("article"), i = $(".no-marquee"), c = {
                    id: "digg-alerts-container"
                };
                window.location.hash === "#upcoming" && b(), s.init(), o.init(), a.init(), l.init(), f.init(), r.size() ? c.elHeight = r.outerHeight() : i.size() && (c.elHeight = i.find(".col-2").outerHeight()), w = m.factory(c), e.user.getUser(g.bind(w.init, w));
                var h = function() {
                    var e = r.size() ? r.outerHeight(): i.find(".col-2").outerHeight();
                    w.setElHeight(e)
                };
                $(window).on("resize orientationchange", g.throttle(h, 100)), e.features.edit_mode && $("#edit-mode-label-container").insertBefore(".site-header-container"), e.features.saved_stories && u.initialize({
                    elID: "#modal-savedstories"
                });
                var p = $(".story-container"), d = g.map(p, function(e) {
                    return $(e).data("contentId")
                }), E = n.factory("top_stories");
                d.length && E.register(d), e.pubsub.on("score_updates", e.pubsub.channels.scores_update, function(e) {
                    $(function() {
                        t.renderStoryScores(e)
                    })
                }), $(document).on("click", ".launch-dailydigg-modal", v.dailyDiggModal).on("click", ".story-action-digg > .target", e.user.authorized(t.submitDigg, "Digg")).on("click", ".story-action-save > .target", e.user.authorized(t.addReadingList, "Save")).on("click", ".story-action-share > .target", t.shareStory).on("click", ".story-link", t.redirectStoryLink), v.dailyDiggModalHash(), $(window).on("hashchange", v.dailyDiggModalHash), $(function() {
                    e.pubsub.on("scroll", e.pubsub.channels.scrolltop, y), y($(document).scrollTop())
                })
            }
        };
        $.fn.k = function(e, t) {
            return t === undefined && (t = "38,38,40,40,37,39,37,39,66,65"), this.each(function() {
                var n = [];
                $(this).keydown(function(r) {
                    n.push(r.keyCode), n.toString().indexOf(t) >= 0 && ($(this).unbind("keydown", arguments.callee), e(r))
                })
            })
        }, $(window).k(S), E.init()
    }), define("bs/frontpage", function() {})
})();
