_DM.define("DM_SmartAlertStrategist", function() {
    function g(b) {
        var a = store.get(h.storeKey)[b];
        if (a === void 0 || a === "null") {
            a = null
        }
        return a
    }
    function i(c, a) {
        var b = store.get(h.storeKey), d = b || {};
        d[c] = a;
        store.set(h.storeKey, d);
        return store.get(h.storeKey)
    }
    var k = {
        customer_survey: ["customer_survey"],
        welcome_tour: ["tour_intro", "tour_already_registered", "tour_follow-categories", "tour_facebook-connect", "tour_follow-friends"]
    }, l = null, j = [], h = {
        storeKey: "DM_SmartAlertStrategist",
        add: function(d, a, c, b, f) {
            var e = {
                type: d,
                withClose: typeof b == "undefined" ? true: b,
                callback: a,
                capping: c ? c: null,
                version: f ? f: null
            };
            if (l) {
                var n = j || [];
                n[n.length] = e;
                j = n
            } else {
                l = e
            }
            PubSub.publish("dm:smartalert:added")
        },
        definitiveClose: function(a) {
            PubSub.publish("dm:metaheader:closed");
            DM_SmartAlertStrategist.setSmartAlertToHide(a)
        },
        setSmartAlertToHide: function(c) {
            var b = h.getSmartAlertKey(c), a = g(b) || {};
            a.hide = true;
            i(b, a)
        },
        canBeDisplayed: function(b) {
            if (h.without_smartalert) {
                return false
            }
            if (b.type == "registration_confirmation") {
                return true
            }
            var a = h.getSmartAlertKey(b.type);
            if (g(a) && g(a).hide) {
                return false
            }
            if (b.capping && g(a) && g(a).displayed && g(a).displayed >= b.capping) {
                return false
            }
            if ($(".sd_metaheader").length > 0) {
                return false
            }
            if (getCookie("hideSuperHeader")) {
                return false
            }
            var c = getCookie(DM_LoginCookie) ? $.parseJSON(decodeURIComponent(getCookie(DM_LoginCookie))).login: "";
            if (getCookie("sid") && (store.get("welcome") == 1 || store.get(c + "_welcometour"))) {
                if (k.welcome_tour.indexOf(b.type)==-1) {
                    l = j[0] || null;
                    return false
                }
            }
            return true
        },
        getSmartAlert: function() {
            var a = l;
            if (!a ||!h.canBeDisplayed(a)) {
                return false
            }
            return a
        },
        getSmartAlertKey: function(a) {
            var b = a;
            if (getCookie(DM_LoginCookie) && getCookie("sid")) {
                var c = $.parseJSON(decodeURIComponent(getCookie(DM_LoginCookie))).login;
                b = c + "_" + a
            }
            return b
        },
        onDisplayed: function(c, b) {
            if (c) {
                var a = g(b) || {};
                if (a.displayed) {
                    a.displayed += 1
                } else {
                    a.displayed = 1
                }
                i(b, a)
            }
        },
        __onDomReady: function() {
            if (h.serverside_smartalert) {
                DM_JS.load("/js/views/shared/smartalert/" + h.serverside_smartalert.type + ".js", h.serverside_smartalert.label)
            }
        },
        __initialize: function() {
            var a = store.get(h.storeKey);
            if (!a) {
                store.set(h.storeKey, {})
            }
            PubSub.subscribe("dm:smartalert:displayed", h.onDisplayed)
        }
    };
    return h
}, true);
_DM.define("Sd_SmartAlert", function() {
    var c = false, d = {
        slideSpeed: 400,
        create: function(j) {
            var l = j.type, a = j.withClose, b = j.callback, k = j.capping, m = j.version, n = {
                request: document.location.pathname,
                type: l,
                withClose: a,
                version: m,
                callback: function(e) {
                    $("#topwrapper").before(e);
                    $smartalert = $(".sd_metaheader");
                    $smartalert.hide();
                    PubSub.publish("dm:smartalert:displaying");
                    $smartalert.slideDown(d.slideSpeed, function() {
                        var f = DM_SmartAlertStrategist.getSmartAlertKey(l);
                        PubSub.publish("dm:smartalert:displayed", k, f);
                        $smartalert.find(".js-close-smartalert").on("click", d.closeSmartAlert);
                        b.call(this)
                    })
                }
            };
            DM_Controller.call("Shared_Metaheader", n)
        },
        onSmartAlertAdded: function() {
            var a = DM_SmartAlertStrategist.getSmartAlert();
            if (a&&!c) {
                c = true;
                d.create(a)
            }
        },
        closeSmartAlert: function() {
            $smartalert = $(".sd_metaheader");
            if ($smartalert) {
                setCookie("hideSuperHeader", 1, 7);
                PubSub.publish("dm:smartalert:closing");
                $(".sd_metaheader").slideUp(400, function() {
                    PubSub.publish("dm:smartalert:closed")
                })
            }
        },
        __initialize: function() {
            PubSub.subscribe("dm:smartalert:added", d.onSmartAlertAdded)
        }
    };
    return d
});
