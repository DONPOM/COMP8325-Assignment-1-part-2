(function(m) {
    var window = this;
    var yV = function(a, b) {
        if ("FORM" == a.tagName)
            for (var c = a.elements, d = 0; a = c[d]; d++)
                yV(a, b);
        else 
            1 == b && a.blur(), a.disabled = b
    };
    var gka = function(a) {
        m.Mk(a);
        a = m.L(a.currentTarget, "feed-item-collapsed-container");
        m.Dl(a, "collapsed", "expanded");
        m.z("yt-dom-content-change", zV)
    };
    var hka = function(a) {
        m.Mk(a);
        a = m.L(a.currentTarget, "feed-item-collapsed-container");
        m.Dl(a, "expanded", "collapsed");
        m.z("yt-dom-content-change", zV)
    };
    var ika = function() {
        "DELAYED_EMBED"in m.Za && (0, m.y)(m.u("DELAYED_EMBED"), function(a) {
            m.Lf(a.container, a.swf_config)
        })
    };
    var jka = function(a, b) {
        function c() {
            var a = window.document.forms["i18n-local-languages-feed-promo-form"];
            a.hl.value = d;
            a.submit()
        }
        var d = m.I(b.currentTarget, "lang-code"), e = m.F("i18n-local-languages-feed-promo"), f = m.I(e, "promo-gl"), e = m.I(e, "current-lang");
        m.al("i18n-promo-select", m.qc({
            gl: f,
            lang: d,
            current: e
        }));
        (new m.nB(a)).write("write_accept", {
            R: c,
            onError: c
        })
    };
    var kka = function(a) {
        var b = m.F("i18n-local-languages-feed-promo"), c = m.I(b, "promo-gl"), b = m.I(b, "current-lang"), c = m.qc({
            gl: c,
            current: b
        });
        m.Pj("i18n-promo-dismiss", c, void 0);
        m.N(AV);
        AV = [];
        (new m.nB(a)).write("write_dismiss")
    };
    var BV = function(a, b) {
        m.t.call(this);
        this.j = a;
        this.D = b;
        this.g = null;
        this.k = m.F("page");
        this.g = m.J(m.F("premium-yva-close"), "click", (0, m.p)(this.Vi, this));
        var c = this.j;
        m.Ef.getInstance().get("HIDDEN_MASTHEAD_ID") == c || m.C(this.k, "masthead-ad-expanded");
        this.C = m.J(window, "message", (0, m.p)(this.fr, this))
    };
    var CV = function(a) {
        var b = m.Ef.getInstance();
        b.set("HIDDEN_MASTHEAD_ID", a);
        b.save()
    };
    var DV = function(a) {
        if (a && a.data && "VideoMasthead" == a.data.source)
            if ("ClickToWatch" == a.data.event) {
                if (a = a.data, a.clickThroughUrl) {
                    var b = m.cl(a.clickThroughUrl), b = m.Mi(b.adurl), c = {};
                    c.adpings = m.dl(a.pings);
                    m.Zd(b, c);
                    m.be(a.clickThroughUrl)
                }
            } else 
                "CloseAd" == a.data.event && BV.prototype.Vi()
        };
    var lka = function(a) {
        a = m.L(a.target, "multi-question-shelf-feedback");
        EV(a);
        m.pd(a, "ui-action-swap-shelf") && FV(a)
    };
    var mka = function(a) {
        for (var b = m.L(a.target, "single-question-shelf"), c = [], d = m.G("single-question-shelf-answers", b).getElementsByTagName("input"), e = 0; e < d.length; e++)
            d[e].checked && c.push(d[e].value);
        if (0 < c.length) {
            c && m.R("/feed_change_ajax?action_give_feedback=1", {
                method: "POST",
                V: {
                    session_token: m.Kl(),
                    feedback_tokens: c,
                    should_merge: 1
                }
            });
            if (b = m.L(b, "feed-item-dismissal-notices"))
                b = m.G("feed-item-dismissal-activate-question", b), m.P(b);
            a = m.L(a.target, "single-question-shelf-feedback");
            m.pd(a, "ui-action-swap-shelf") &&
            FV(a)
        }
    };
    var nka = function(a) {
        a = m.L(a.target, "question-shelf-dismissal");
        m.pd(a, "feedback-token") && EV(a);
        m.pd(a, "action") && "hide" == m.I(a, "action").toLowerCase() && (a = m.L(a, "feed-item-container"), m.P(a), m.z("yt-dom-content-change", a))
    };
    var FV = function(a) {
        var b = m.I(a, "ui-action-swap-shelf"), c = m.L(a, "swappable-shelf"), d = m.L(a, "feed-item-main"), b = m.G(b, d);
        m.K(b);
        m.P(c);
        m.pd(a, "final-feedback") && "true" == m.I(a, "final-feedback").toLowerCase() && (a = m.G("yt-close-img", d), m.el(a, "feedback-token"));
        m.nC.getInstance().sc();
        m.z("yt-dom-content-change", b)
    };
    var EV = function(a) {
        m.pd(a, "feedback-token") && (a = {
            session_token: m.Kl(),
            feedback_tokens: [m.I(a, "feedback-token")]
        }, m.R("/feed_change_ajax?action_give_feedback=1", {
            method: "POST",
            V: a
        }))
    };
    var oka = function() {
        m.N(GV);
        m.nb(HV);
        (0, m.y)(IV, function(a) {
            m.bb(a)
        })
    };
    var pka = function(a) {
        a.preventDefault();
        a = m.L(a.currentTarget, "branded-page-related-channels-item");
        JV(a, !0)
    };
    var KV = function(a) {
        m.st(a, {
            duration: .3,
            ka: function() {
                m.ri(a)
            }
        })
    };
    var qka = function(a) {
        var b = m.M("branded-page-related-channels-item"), c = m.Ja(b, function(b) {
            return m.I(b, "external-id") == a
        });
        c && IV.push(m.v(function() {
            JV(c, !1)
        }, 2E3))
    };
    var JV = function(a, b) {
        var c=!!m.G("branded-page-related-channels-featured-badge", a), d = m.I(a, "external-id"), e = m.qi(a, "ul"), e = m.M("branded-page-related-channels-item", e), e = (0, m.sh)(e, function(a) {
            return m.I(a, "external-id")
        }), f = {};
        c && (f.featured = 1);
        m.R("/guide_ajax?action_reload_channel=1", {
            method: "POST",
            Z: f,
            V: {
                session_token: m.Kl(),
                dismissed_id: d,
                shown_ids: e.join(),
                dismiss_old_channel: b
            },
            R: function(b, c) {
                var d = c.new_suggested_html;
                d ? (d = m.Uk(d), m.ul(d, a), m.z("yt-dom-content-change", d), m.rt(d, 0, 1, {
                    duration: .3
                })) :
                KV(a)
            },
            onError: function() {
                KV(a)
            }
        })
    };
    var rka = function(a) {
        if (a = m.L(a.target, "yt-uix-toggle-menu-item"))
            if (a = m.L(a, "yt-uix-toggle-menu-item-container")) {
                var b = m.tl(a) || m.sl(a);
                b && (m.P(a), m.K(b))
            }
    };
    var LV = function(a, b) {
        this.J = "session_token=" + a;
        (this.g = "https://" + window.document.location.hostname) && "/" != this.g.charAt(this.g.length - 1) && (this.g += "/");
        this.k = b;
        this.C = null;
        this.H = [];
        this.D = [];
        this.L = {}
    };
    var ska = function(a) {
        a = a.getServiceInfo();
        for (var b in a) {
            var c = m.F(b + "-connected");
            if (c) {
                var d = a[b], e = m.F(b + "-not-connected"), f = m.F(b + "-display-name"), h = d.is_connected;
                m.wf(c, h);
                m.wf(e, !h);
                m.pl(f, d.connected_as || "")
            }
        }
    };
    var tka = function(a) {
        a == m.F("browse-items-primary") && (m.nC.getInstance().sc(), m.hB())
    };
    var uka = function() {
        m.R("/feed_ajax", {
            method: "POST",
            Z: {
                action_pause_watch_history: 1
            },
            V: {
                session_token: m.Kl()
            },
            R: MV,
            onError: NV
        })
    };
    var vka = function() {
        m.R("/feed_ajax", {
            method: "POST",
            Z: {
                action_resume_watch_history: 1
            },
            V: {
                session_token: m.Kl()
            },
            R: MV,
            onError: NV
        })
    };
    var MV = function(a, b) {
        OV(b) || m.wk(PV, QV)
    };
    var wka = function() {
        var a = {
            action_clear_watch_history: 1
        };
        a.clear_dialog_shown = RV;
        m.R("/feed_ajax", {
            method: "POST",
            Z: a,
            V: {
                session_token: m.Kl()
            },
            R: SV,
            onError: NV
        })
    };
    var xka = function() {
        m.R("/feed_ajax", {
            method: "POST",
            Z: {
                action_pause_search_history: 1
            },
            V: {
                session_token: m.Kl()
            },
            R: TV,
            onError: NV
        })
    };
    var yka = function() {
        m.R("/feed_ajax", {
            method: "POST",
            Z: {
                action_resume_search_history: 1
            },
            V: {
                session_token: m.Kl()
            },
            R: TV,
            onError: NV
        })
    };
    var TV = function(a, b) {
        OV(b) || m.wk(UV, VV)
    };
    var zka = function() {
        var a = {
            action_clear_search_history: 1
        };
        a.clear_dialog_shown = WV;
        m.R("/feed_ajax", {
            method: "POST",
            Z: a,
            V: {
                session_token: m.Kl()
            },
            R: SV,
            onError: NV
        })
    };
    var SV = function(a, b) {
        OV(b.errors) || Aka()
    };
    var NV = function(a, b) {
        OV(b)
    };
    var OV = function(a) {
        return a && a.errors && a.errors.length ? ((0, m.y)(a.errors, function(a) {
            m.gs(a, XV, YV)
        }), !0) : !1
    };
    var Aka = function() {
        var a = m.M("feed-item-container");
        (0, m.y)(a, function(a) {
            m.ri(a)
        })
    };
    var ZV = function() {
        return m.M("feed-item-checkbox", $V)
    };
    var Bka = function() {
        var a = m.I(aW, "video-ids") || "";
        if (bW)
            m.Br(bW, a);
        else {
            var b = m.G("feed-add-to-widget"), c = m.I(aW, "hide-watch-later") || "";
            bW = new m.Ar({
                videoIds: a,
                Su: "true" == c.toLowerCase()
            });
            bW.oa(b)
        }
    };
    var Cka = function(a) {
        cW(a.currentTarget.checked)
    };
    var Dka = function(a) {
        a = "false" != m.I(a.currentTarget, "enabled");
        m.wp(dW, a);
        cW(a)
    };
    var cW = function(a) {
        (0, m.y)(ZV(), function(b) {
            m.wp(b, a)
        });
        eW()
    };
    var Eka = function() {
        eW()
    };
    var eW = function() {
        var a = [], b=!0, c = ZV();
        (0, m.y)(c, function(c) {
            c.checked && (a.push(c.value), b=!1)
        });
        m.fl(aW, "video-ids", a.join(","));
        yV(aW, b);
        yV(fW, b);
        gW && yV(gW, 0 == c.length)
    };
    var Fka = function() {
        var a = m.I(aW, "video-ids"), b = {}, c = "";
        switch (hW) {
        case "history":
            b.action_remove_history_entries = 1;
            c = "/feed_ajax";
            break;
        case "watch_later":
            b.action_remove_watch_later_entries = 1;
            c = "/feed_change_ajax";
            break;
        default:
            throw Error("Unsupported feed name: " + hW);
        }
        m.R(c, {
            method: "POST",
            Z: b,
            V: {
                session_token: m.Kl(),
                video_ids: a
            },
            R: Gka,
            onError: iW
        })
    };
    var Hka = function() {
        m.R("/feed_change_ajax", {
            method: "POST",
            Z: {
                action_clear_watched_from_watch_later: 1
            },
            V: {
                session_token: m.Kl()
            },
            R: Ika,
            onError: iW
        })
    };
    var Ika = function(a, b) {
        jW(m.hb("WATCHED_VIDEOS_REMOVED"));
        var c = 0;
        (0, m.y)(m.M("watched", $V), function(a) {
            m.ri(m.qi(a, "LI"));
            c++
        });
        kW(b.video_count);
        lW()
    };
    var Jka = function(a, b) {
        m.ri(m.qi(a, "LI"));
        kW(b)
    };
    var Gka = function(a, b) {
        (0, m.y)(ZV(), function(a) {
            a.checked && m.ri(m.qi(a, "LI"))
        });
        kW(b.video_count)
    };
    var kW = function(a) {
        mW && (mW.innerHTML = a);
        eW();
        m.wp(dW, !1);
        m.z("yt-dom-content-change", $V)
    };
    var lW = function() {
        var a = m.M("watched", $V).length;
        nW && yV(nW, 0 == a)
    };
    var iW = function() {
        m.P(oW);
        m.K(pW)
    };
    var Kka = function() {
        m.R("/feed_ajax", {
            method: "POST",
            Z: {
                action_pause_watch_history: 1
            },
            V: {
                session_token: m.Kl()
            },
            R: Lka,
            onError: iW
        })
    };
    var Lka = function() {
        qW(m.hb("HISTORY_FEED_PAUSED"))
    };
    var Mka = function() {
        m.R("/feed_ajax", {
            method: "POST",
            Z: {
                action_resume_watch_history: 1
            },
            V: {
                session_token: m.Kl()
            },
            R: Nka,
            onError: iW
        })
    };
    var Nka = function() {
        qW(m.hb("HISTORY_FEED_RESUMED"))
    };
    var qW = function(a) {
        jW(a);
        m.wk(m.F("watch-history-pause-container"), m.F("watch-history-resume-container"))
    };
    var jW = function(a) {
        m.G("yt-alert-content", oW).innerHTML = a;
        m.K(oW);
        m.P(pW)
    };
    var Oka = function() {
        var a = {
            action_clear_watch_history: 1
        };
        a.clear_dialog_shown = rW;
        m.R("/feed_ajax", {
            method: "POST",
            Z: a,
            V: {
                session_token: m.Kl()
            },
            R: Pka,
            onError: iW
        })
    };
    var Pka = function() {
        jW(m.hb("HISTORY_FEED_CLEARED"));
        var a = ZV();
        0 != a.length && (m.qi(a[0], "UL").innerHTML = "", eW())
    };
    var sW = function() {
        m.sr.call(this)
    };
    var tW = function(a) {
        m.pd(a, "feedback-token") && (a = {
            session_token: m.Kl(),
            feedback_tokens: [m.I(a, "feedback-token")]
        }, m.R("/feed_change_ajax?action_give_feedback=1", {
            method: "POST",
            V: a
        }))
    };
    var uW = function() {
        m.Uc.call(this, "www/feed", ["www/common"], ["feed", "index", "results"], Qka, Rka)
    };
    var Qka = function() {
        m.$q(m.nC);
        m.nC.getInstance().sc();
        ika();
        "MASTHEAD_ENCRYPTED_ID"in m.Za && (window.masthead = new BV(m.u("MASTHEAD_ENCRYPTED_ID"), m.u("MASTHEAD_IS_BRANDED")), m.J(window, "message", DV));
        if ("PYV_IFRAME_ID"in m.Za) {
            var a = m.u("PYV_IFRAME_CONTENT"), b = m.F(m.u("PYV_IFRAME_ID"));
            b && (b = m.ql(b), b.open(), b.write("<!DOCTYPE html><html><head></head><body>" + a + "</body></html>"), m.E || b.close())
        }
        if (a = m.F("i18n-local-languages-feed-promo"))
            if (b = m.I(a, "column-name")) {
                var c = {
                    gl: m.I(a, "promo-gl")
                }, c = m.qc(c);
                m.Pj("i18n-promo-show", c, void 0);
                c = m.q(kka, b);
                AV.push(m.O(null, "click", m.q(jka, b), "i18n-local-languages-feed-promo-language-option"), m.O(a, "click", c, "i18n-local-languages-feed-promo-close"))
            }
        if (a = m.F("page"))
            vW.push(m.O(a, "click", m.q(lka), "multi-question-shelf-feedback")), vW.push(m.O(a, "click", m.q(mka), "single-question-shelf-feedback")), vW.push(m.O(a, "click", m.q(nka), "question-shelf-dismissal"));
        GV = m.O(window.document.body, "click", pka, "branded-page-related-channels-item-close");
        HV.push(m.x("subscription-subscribe-loaded",
        qka));
        (a = m.F("feed")) || (a = m.F("browse-items-primary"));
        wW.push(m.x("yt-uix-load-more-success", tka));
        b = m.G("individual-feed");
        c=!1;
        b && (m.I(b, "feed-name"), c = "user" == m.I(b, "feed-type"));
        m.DD(a, c);
        m.u("INIT_AUTOSHARE") && (a = new LV(m.Kl()), a.addServiceChangedCallback(ska), (b = m.F("facebook-connect-button")) && a.registerConnectDialogLauncher(b, "facebook", !0), (b = m.F("twitter-connect-button")) && a.registerConnectDialogLauncher(b, "twitter", !0), (b = m.F("orkut-connect-button")) && a.registerConnectDialogLauncher(b,
        "orkut", !0), window.autoshare = a);
        m.hB();
        m.$D();
        (a = zV = m.F("feed")) || (a = zV = m.F("browse-items-primary"));
        xW.push(m.O(a, "click", gka, "feed-item-expander-button"));
        xW.push(m.O(a, "click", hka, "feed-item-collapser-button"));
        yW || (yW=!0, zW.push(m.O(window.document.documentElement, "click", rka, "yt-uix-toggle-menu-item")));
        m.A(window.document.body, "exp-innertube-history") ? (YV = m.F("feed-action-alerts"), XV = m.G("ajax-alert", YV), PV = m.F("watch-history-pause-button"), QV = m.F("watch-history-resume-button"), UV = m.F("search-history-pause-button"),
        VV = m.F("search-history-resume-button"), AW.push(m.J(m.F("watch-history-clear-confirm-button"), "click", wka)), AW.push(m.J(PV, "click", uka)), AW.push(m.J(QV, "click", vka)), AW.push(m.J(m.F("search-history-clear-confirm-button"), "click", zka)), AW.push(m.J(UV, "click", xka)), AW.push(m.J(VV, "click", yka)), m.u("SHOULD_SHOW_CLEAR_WATCH_HISTORY") && m.F("watch-history-clear-all-button") && m.kq.getInstance().show(m.F("watch-history-clear-all-button")), m.u("SHOULD_SHOW_CLEAR_SEARCH_HISTORY") && m.F("search-history-clear-button") &&
        m.kq.getInstance().show(m.F("search-history-clear-button")), WV = RV = 0) : (a = m.G("individual-feed")) && "history" == m.I(a, "feed-name") && (a = m.G("individual-feed"), a = m.I(a, "feed-name"), (b = m.F("feed")) || (b = m.F("browse-items-primary")), hW = a, $V = b, dW = m.F("feed-item-select-all-checkbox"), oW = m.F("managed-feed-success"), pW = m.F("managed-feed-error"), fW = m.F("feed-item-remove-videos"), gW = m.F("managed-toolbar-play-all-button"), mW = m.F("feed-title-count"), BW = m.G("branded-page-v2-primary-col-header-container"), aW = m.G("addto-button",
        BW), CW.push(m.J(dW, "click", Cka)), CW.push(m.O(window.document.body, "click", Dka, "feed-item-selection-action")), CW.push(m.O(window.document.body, "click", Eka, "feed-item-checkbox")), CW.push(m.J(fW, "click", Fka)), CW.push(m.J(aW, "click", Bka)), "watch_later" == a && (nW = m.F("watch-later-clear-watched-button"), CW.push(m.J(nW, "click", Hka)), DW.push(m.x("WATCH_LATER_VIDEO_REMOVED", Jka)), lW()), "history" == a && (CW.push(m.J(m.F("watch-history-pause-button"), "click", Kka)), CW.push(m.J(m.F("watch-history-resume-button"), "click",
        Mka)), CW.push(m.O(window.document.body, "click", Oka, "watch-history-clear-button")), m.u("SHOULD_SHOW_CLEAR_WATCH_HISTORY") && m.F("watch-history-clear-all-button") && m.kq.getInstance().show(m.F("watch-history-clear-all-button")), rW = 0));
        m.u("LOAD_USER_INTERESTS") && (a = m.G("recommended-interests"), (new sW).oa(a))
    };
    var Rka = function() {
        m.br(m.nC);
        m.Xa("DELAYED_EMBED", []);
        window.masthead && (window.masthead.dispose(), m.Pk(window, "message", DV));
        m.N(AV);
        AV = [];
        m.N(vW);
        vW.length = 0;
        oka();
        m.nb(wW);
        wW.length = 0;
        m.KD();
        m.nb(m.bE);
        m.bE.length = 0;
        m.N(xW);
        xW.length = 0;
        yW=!1;
        m.N(zW);
        zW.length = 0;
        m.N(CW);
        CW.length = 0;
        m.nb(DW);
        DW.length = 0;
        oW = pW = dW = mW = gW = nW = fW = aW = BW = $V = null;
        bW && (bW.dispose(), bW = null);
        m.N(AW);
        AW.length = 0;
        VV = UV = QV = PV = YV = null;
        m.eC()
    };
    var zV;
    var xW = [];
    var AV = [];
    m.s(BV, m.t);
    var Ska = /^https?:\/\/(ad.doubleclick|s0.2mdn).net$/;
    m.g = BV.prototype;
    m.g.fr = function(a) {
        a && a.origin && Ska.test(a.origin) && "creative2yt_requestClose" == a.data && this.Sp()
    };
    m.g.N = function() {
        BV.K.N.call(this);
        m.N(this.g);
        m.N(this.C);
        this.C = this.g = null
    };
    m.g.Sp = function() {
        m.P("ad_creative_1");
        m.z("ads-masthead-hide");
        m.z("yt-dom-content-change");
        this.D && m.P("ad_creative_collapse_btn_1");
        m.K("ad_creative_expand_btn_1");
        m.D(this.k, "masthead-ad-expanded");
        CV(this.j);
        m.Pj("homepage_collapse_masthead_ad", void 0, void 0)
    };
    m.g.Vi = function() {
        m.K("ad_creative_expand_btn_1");
        m.ri(m.F("premium-yva"));
        m.ri(m.F("video-masthead"));
        m.z("yt-dom-content-change");
        CV(this.j)
    };
    m.g.rA = function() {
        m.Dl(window.document.getElementById("premium-yva"), "premium-yva-unexpanded", "premium-yva-expanded")
    };
    m.g.sA = function() {
        m.Dl(window.document.getElementById("premium-yva"), "premium-yva-expanded", "premium-yva-unexpanded")
    };
    m.g.gA = function() {
        m.P("premium-yva");
        m.D(m.F("premium-yva"), "premium-yva-unexpanded")
    };
    m.g.wA = function() {
        var a = m.Ef.getInstance();
        a.set("HIDDEN_MASTHEAD_ID", !1);
        a.save();
        m.Pj("homepage_expand_masthead_ad", void 0, void 0);
        m.ae(window.document.location.href)
    };
    m.l("yt.flash.embed", m.Lf, void 0);
    m.l("yt.www.ads.pyv.pyvHomeAfcCallback", m.mC, void 0);
    m.l("yt.www.ads.MastheadAd", BV, void 0);
    BV.prototype.autoCollapsePremiumYva = BV.prototype.gA;
    BV.prototype.collapse_ad = BV.prototype.Sp;
    BV.prototype.expand_ad = BV.prototype.wA;
    BV.prototype.userCollapsePremiumYva = BV.prototype.Vi;
    BV.prototype.userExpandPremiumYva = BV.prototype.rA;
    BV.prototype.userUnexpandPremiumYva = BV.prototype.sA;
    var vW = [];
    var GV, HV = [], IV = [];
    var yW=!1, zW = [];
    m.l("yt.sharing.AutoShare", LV, void 0);
    LV.prototype.Ca = function(a, b, c, d) {
        m.J(a, "click", (0, m.p)(this.S, this));
        if (a.id)
            this.L[a.id] = {
                serviceName: b,
                connectOnly: c
            }, d && (this.L[a.id].connectOnlyCallback = d);
        else 
            throw "Connect dialog launch buttons must have an id.";
    };
    LV.prototype.registerConnectDialogLauncher = LV.prototype.Ca;
    LV.prototype.S = function(a) {
        if (a = this.L[a.currentTarget.id]) {
            var b = a.connectOnly;
            a.connectOnlyCallback && (b = (0, a.connectOnlyCallback)());
            this.F(a.serviceName, b)
        }
    };
    LV.prototype.handleConnectService = LV.prototype.S;
    LV.prototype.ja = function() {
        this.j()
    };
    LV.prototype.doOnLoad = LV.prototype.ja;
    LV.prototype.xa = function(a) {
        this.H.push(a)
    };
    LV.prototype.addServiceChangedCallback = LV.prototype.xa;
    LV.prototype.U = function(a) {
        this.D.push(a)
    };
    LV.prototype.addCanConnectCallback = LV.prototype.U;
    LV.prototype.lb = function() {
        return this.k
    };
    LV.prototype.getServiceInfo = LV.prototype.lb;
    LV.prototype.F = function(a, b) {
        for (var c in this.D)
            if (!(0, this.D[c])(this, a, b))
                return;
        m.R(this.g + "autoshare", {
            Z: {
                action_ajax_stats_ping: "1",
                stat: "connect_has_google",
                service: a
            }
        });
        c = this.g + "autoshare?action_popup_auth=1&service=" + a + "&connect_only=" + (b ? "True" : "False") + "&root_url=" + (0, window.encodeURIComponent)(this.g);
        "facebook" == a && (c += "&permissions=" + (0, window.encodeURIComponent)("read_stream,offline_access,manage_pages,publish_stream"));
        this.T(c, {
            height: 500,
            width: 860
        })
    };
    LV.prototype.connectService = LV.prototype.F;
    LV.prototype.X = function(a, b) {
        var c = (0, m.p)(function(a, c) {
            this.k = m.uk(c.html_content);
            this.j();
            b && b()
        }, this), d = (0, m.p)(function() {
            b && b();
            this.j()
        }, this), e = {
            action_ajax_connect_service: 1
        };
        e.return_url = a;
        m.R(this.g + "autoshare?ajax_connect_service", {
            format: "XML",
            method: "POST",
            Rb: m.qc(e) + "&" + this.J,
            R: c,
            onError: d
        })
    };
    LV.prototype.connectServiceDone = LV.prototype.X;
    LV.prototype.ba = function(a) {
        this.O(a)
    };
    LV.prototype.disconnectService = LV.prototype.ba;
    LV.prototype.qa = function(a, b) {
        var c = (0, m.p)(function(a, b) {
            this.k = m.uk(b.html_content);
            this.j()
        }, this), d = (0, m.p)(function() {
            this.j()
        }, this), e = {
            action_ajax_set_connect_only: 1
        };
        e.service = a;
        e.connect_only = b ? "True" : "False";
        m.R(this.g + "autoshare?ajax_set_connect_only", {
            format: "XML",
            method: "POST",
            Rb: m.qc(e) + "&" + this.J,
            R: c,
            onError: d
        })
    };
    LV.prototype.setConnectOnly = LV.prototype.qa;
    LV.prototype.O = function(a) {
        var b = (0, m.p)(function(a, b) {
            this.k = m.uk(b.html_content);
            this.j()
        }, this), c = (0, m.p)(function() {
            this.j()
        }, this), d = {
            action_ajax_disconnect_service: 1
        };
        d.service = a;
        m.R(this.g + "autoshare?ajax_disconnect_service", {
            format: "XML",
            method: "POST",
            Rb: m.qc(d) + "&" + this.J,
            R: b,
            onError: c
        })
    };
    LV.prototype.j = function() {
        for (var a in this.H)(0, this.H[a])(this)
    };
    LV.prototype.T = function(a, b) {
        if (this.C)
            try {
                this.C.close()
        } catch (c) {
            this.C = null
        }
        this.C=!m.Nj(a, b)
    };
    LV.prototype.Ba = function() {
        this.F("facebook", !this.k.facebook.is_autosharing)
    };
    var wW = [];
    var RV, WV, YV, XV, PV, QV, UV, VV, AW = [];
    var hW, rW, $V, BW, aW, fW, nW, gW, mW, dW, pW, oW, bW, CW = [], DW = [];
    m.s(sW, m.sr);
    sW.prototype.W = function() {
        sW.K.W.call(this);
        m.tr(this, "click", "user-interests-recommendation-feedback", this.k)
    };
    sW.prototype.k = function(a) {
        var b = m.L(a.target, "user-interests-recommendation-feedback");
        tW(b);
        if (b = m.L(a.target, "recommendation"))
            a = m.tl(b), m.P(b), a ? (b = m.G("user-interests-recommendation-impression", a), tW(b), m.K(a)) : m.K(this.P("no-recommendations"))
    };
    m.l("yt.www.feed.managedfeed.onHideClearDialog", function() {
        rW++
    }, void 0);
    m.l("yt.www.feed.history.onHideClearWatchHistoryDialog", function() {
        RV++
    }, void 0);
    m.l("yt.www.feed.history.onHideClearSearchHistoryDialog", function() {
        WV++
    }, void 0);
    m.s(uW, m.Uc);
    uW.prototype.enable = function() {
        uW.K.enable.call(this)
    };
    uW.prototype.disable = function() {
        uW.K.disable.call(this)
    };
    m.Wc(new uW);
})(_yt_www);

