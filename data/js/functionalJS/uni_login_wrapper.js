var passport = passport || window.passport || {};
passport._modulePool = passport._modulePool || {};
passport._define = passport._define || function(a, b) {
    passport._modulePool[a] = b && b()
};
passport._getModule = passport._getModule || function(a) {
    return passport._modulePool[a]
};
passport.pop = passport.pop || {};
passport.pop.insertScript = passport.pop.insertScript || function(b, a) {
    var e = document, c = e.createElement("SCRIPT");
    c.type = "text/javascript";
    c.charset = "UTF-8";
    if (c.readyState) {
        c.onreadystatechange = function() {
            if (c.readyState == "loaded" || c.readyState == "complete") {
                c.onreadystatechange = null;
                a && a()
            }
        }
    } else {
        c.onload = function() {
            a && a()
        }
    }
    c.src = b;
    e.getElementsByTagName("head")[0].appendChild(c)
};
passport.pop.init = passport.pop.init || function(f) {
    var h = {
        "http:": "http://passport.bdimg.com",
        "https:": "https://passport.baidu.com"
    };
    var a = {
        uni_login: "/passApi/js/uni_login_ab85ab4c.js",
        uni_login_tangram: "/passApi/js/uni_login_tangram_c901a356.js",
        uni_loginPad: "/passApi/js/uni_loginPad_6748c2a0.js",
        uni_loginPad_tangram: "/passApi/js/uni_loginPad_tangram_25f407fd.js",
        uni_loginWap: "/passApi/js/uni_loginWap_bb44a491.js",
        uni_loginWap_tangram: "/passApi/js/uni_loginWap_bb44a491.js",
        uni_accConnect: "/passApi/js/uni_accConnect_ac23edab.js",
        uni_accConnect_tangram: "/passApi/js/uni_accConnect_tangram_cfde8456.js",
        uni_multiBind: "/passApi/js/uni_multiBind_32f7258c.js",
        uni_multiBind_tangram: "/passApi/js/uni_multiBind_tangram_b5bf3fda.js",
        uni_multiUnbind: "/passApi/js/uni_multiUnbind_72856ad3.js",
        uni_multiUnbind_tangram: "/passApi/js/uni_multiUnbind_tangram_f466292b.js",
        uni_changeUser: "/passApi/js/uni_changeUser_ff79a0bc.js",
        uni_changeUser_tangram: "/passApi/js/uni_changeUser_tangram_59545c15.js",
        uni_loginMultichoice: "/passApi/js/uni_loginMultichoice_04d76963.js",
        uni_loginMultichoice_tangram: "/passApi/js/uni_loginMultichoice_tangram_453f9c45.js",
        uni_confirmWidget: "/passApi/js/uni_confirmWidget_ab5c02c9.js",
        uni_confirmWidget_tangram: "/passApi/js/uni_confirmWidget_tangram_57811d40.js"
    }, k = {
        login: "/passApi/css/uni_login_new_68d7802a.css",
        loginWap: "/passApi/css/uni_loginWap_dbb3b171.css",
        accConnect: "/passApi/css/uni_accConnect_ce05f16d.css",
        loginPad: "/passApi/css/uni_loginPad_e0bce2b6.css",
        multiBind: "/passApi/css/uni_multiBind_61bfd55d.css",
        multiUnbind: "/passApi/css/uni_multiUnbind_c6fc910f.css",
        changeUser: "/passApi/css/uni_changeUser_9b5d4672.css",
        loginMultichoice: "/passApi/css/uni_loginMultichoice_5567bb0f.css",
        confirmWidget: "/passApi/css/uni_confirmWidget_9dc02704.css"
    };
    var b = h[(window.location ? window.location.protocol.toLowerCase() : document.location.protocol.toLowerCase())];
    var f = f || {};
    f.type = f.type || "login";
    var g = document, i, f = f || {}, j = "_PassUni" + new Date().getTime();
    var e = b + k[f.type];
    if (f.cssUrlWrapper) {
        e = cssUrlWrapper.join("uni_login.css")
    }
    i = {
        show: function() {
            i.loadPass(f.apiOpt);
            i.willShow = true;
            return i
        },
        loadPass: function() {
            var d = g.createElement("link");
            d.rel = "stylesheet";
            d.type = "text/css";
            d.href = e;
            d.disabled = false;
            g.getElementsByTagName("head")[0].appendChild(d);
            i.show = function() {
                i.willShow = true;
                return i
            };
            if (f.plugCss) {
                d = g.createElement("link");
                d.rel = "stylesheet";
                d.type = "text/css";
                d.href = f.plugCss;
                d.disabled = false;
                g.getElementsByTagName("head")[0].appendChild(d)
            }
            i.passCallback();
            delete i.loadPass
        },
        passCallback: function() {
            if (i.components.length > 0) {
                passport.pop.insertScript(i.components.shift(), i.passCallback)
            } else {
                passport.pop[f.type](f, i, function() {
                    if (i.willShow) {
                        i.show()
                    }
                    if (f && f.onRender) {
                        f.onRender()
                    }
                });
                delete i.passCallback;
                delete i.components
            }
        },
        components: []
    };
    var c = "uni_" + f.type;
    if (f.tangram) {
        c += "_tangram"
    }
    if (f.apiOpt && f.apiOpt.product == "ik" && Math.random() < 0.3) {
        i.components.push(b + "/passApi/js/uni/passhunt_2fb2d301.js");
        f.hunter = true
    }
    i.components.push(b + a[c]);
    if (f.cache) {
        i.loadPass(f.apiOpt)
    }
    if (f.id) {
        if (g.getElementById(f.id)) {
            g.getElementById(f.id).onclick = function() {
                i.show()
            }
        }
    }
    return i
};
