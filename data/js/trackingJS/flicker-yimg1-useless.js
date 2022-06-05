(function() {
    var m = "refresh-sihp", r = false, c = false, f, j, b, t, u, h, k, v, s, l, q = false, a = false, d = false, i = false, p = 1000, e = 20, n, g = {}, o;
    YUI.add(m, function(A) {
        var S = 1, R = 2, N = A.flickrAPI, z, G = 15, W, V, H = false, T, J = this;
        function Q(aa) {
            var Y;
            V = A.Cookie.get("liqpw");
            if (V) {
                p = V - (320 + 20);
                p = Math.min(p, 1310 - (320 + 20));
                p = Math.max(p, 970 - (320 + 20))
            } else {
                p = 1024 - (320 + 20)
            }
            f = A.one("#activityFeed");
            j = A.one("#see-more-button");
            b = A.one("#loading-cards");
            k = YUI.uniquePhotoIds.length;
            if (YUI.is_recoFeed) {
                H = true
            }
            if (!A.Lang.isObject(aa)) {
                console.warn("[" + m + "]", "No config");
                return 
            }
            T = aa.container;
            if (A.Lang.isString(T)) {
                T = A.one(T)
            }
            if (!A.Lang.isObject(T) ||!T.getDOMNode) {
                console.warn("[" + m + "]", "No container node");
                return 
            }
            W = A.on("scroll", A.betterThrottle(D, e, J));
            j.on("click", K);
            T.detach("click");
            T.on("click", w);
            T.on("mouseover", B);
            s = T.one("#no-friends-close");
            if (s) {
                s.on("click", function(ac) {
                    var ab = new Date();
                    ab.setTime(ab.getTime() + (1000 * 24 * 60 * 60 * 1000));
                    ac.halt();
                    A.Cookie.set("rfrsh_nofriends_dismiss", 1, {
                        domain: ".flickr.com",
                        path: "/",
                        expires: ab
                    });
                    A.one("#no-friends").remove();
                    A.fire("nofriends:close")
                })
            }
            l = T.one("#photobooks-banner-close");
            if (l) {
                l.on("click", function(ac) {
                    var ab = new Date();
                    ab.setTime(ab.getTime() + (1000 * 24 * 60 * 60 * 1000));
                    ac.halt();
                    A.Cookie.set("rfrsh_photobooks_dismiss", 1, {
                        domain: ".flickr.com",
                        path: "/",
                        expires: ab
                    });
                    A.one("#photobooks-banner").remove();
                    A.fire("photobooksbanner:close")
                })
            }
            aa.commentIconSelector = ".card-single .il_comments, .card-bundle .il_comments";
            aa.photoContainerSelector = ".imgContainer";
            aa.photoNodeSelector = ".imgWrapper";
            aa.commentButtonNodeSelector = ".il_comments";
            aa.commentContainerSelector = ".card";
            o = new A.refreshSIHPComment(aa);
            aa.dismissCommentEvent = o.getDismissEvent();
            A.refreshSIHPFavorite.init(aa);
            A.refreshSIHPShare.init(aa);
            A.refreshSIHPMuting.init(aa);
            if (A.refreshSIHPKeyboard && A.refreshSIHPKeyboard.init) {
                A.refreshSIHPKeyboard.init(aa)
            }
            z = A.refreshSIHPSidebar;
            if (!A.config.flickr.flags.enable_hermes_sihp_deferred_sidebar) {
                z.init(aa)
            }
            if (A.config.flickr.flags.enable_refresh_sihp_feed_ads) {
                A.refreshSIHPFeedAds.init(aa)
            }
            D();
            v = T.all("img.defer");
            A.deferImages(v, {
                threshold: A.DOM.winHeight()
            });
            if (A.config.flickr.flags.enable_sihp_viewcount) {
                v = v.filter(":not(.buddyicon)");
                v = v.concat(A.all(".card:first-child img:not(.buddyicon)"));
                A.ViewCount.onVisible(v)
            }
            if (A.config.flickr.flags.enable_keyboard_shortcut_legend && A.keyboardShortcutLegend && A.keyboardShortcutLegend.register) {
                A.keyboardShortcutLegend.register({
                    keyCodes: [32],
                    key: "space",
                    description: A.transjax.get("refresh-sihp-transjax", "scroll_down"),
                    group: A.transjax.get("keyboard-shortcut-legend", "navigation"),
                    order: 10
                });
                A.keyboardShortcutLegend.register({
                    keyCodes: [32, 16],
                    key: "shift + space",
                    description: A.transjax.get("refresh-sihp-transjax", "scroll_up"),
                    group: A.transjax.get("keyboard-shortcut-legend", "navigation"),
                    order: 10
                });
                A.keyboardShortcutLegend.register({
                    keyCodes: [74],
                    key: "J",
                    description: A.transjax.get("refresh-sihp-transjax", "scroll_to_next_row"),
                    group: A.transjax.get("keyboard-shortcut-legend", "navigation"),
                    order: 0
                });
                A.keyboardShortcutLegend.register({
                    keyCodes: [75],
                    key: "K",
                    description: A.transjax.get("refresh-sihp-transjax", "scroll_to_prev_row"),
                    group: A.transjax.get("keyboard-shortcut-legend", "navigation"),
                    order: 0
                })
            }
            if (!window.F) {
                window.F = {}
            }
            F = window.F;
            F.stewart_event = function(ad, ab, ac) {
                if (ab === "videoEnded"&&!n) {
                    P(ad)
                }
            };
            F.set_hd_cookie = function(ac) {
                n = true;
                window.setTimeout(function() {
                    n = false
                }, 500);
                if (ac !== 1 && ac !== 0) {
                    return 
                }
                var ab = new Date();
                ab.setTime(ab.getTime() + (1000 * 24 * 60 * 60 * 1000));
                A.Cookie.set("likes_hd", ac, {
                    domain: ".flickr.com",
                    path: "/",
                    expires: ab
                })
            };
            F.get_hd_cookie = function() {
                return A.Cookie.get("likes_hd")
            }
        }
        function P(ac) {
            var aa, Y;
            aa = g[ac].node;
            aa.setHTML(g[ac].code);
            Y = g[ac].sessionPhotos;
            if (Y) {
                Y.removeClass("hidden")
            }
            try {
                delete g[ac]
            } catch (ab) {}
        }
        function x(ad, ab) {
            var Y, aa, ac, ag, af, ah = A.Node.create('<span class="html5-balls"><span class="blueball"></span><span class="pinkball"></span></span>'), ae;
            ac = ad.ancestor(".card-contacted");
            Y = "flickr.contacts.add";
            aa = {
                user_id: ab,
                friend: 0,
                family: 0,
                silent: 1
            };
            ag = {
                success: function(ai) {
                    ah.remove();
                    if (ac) {
                        ac.append('<span class="added">' + A.transjax.get("refresh-sihp", "added_contact") + "</span>")
                    } else {
                        af.append('<span class="added">' + A.transjax.get("refresh-sihp", "added_contact") + "</span>")
                    }
                },
                failure: function(ai) {
                    ah.remove();
                    ad.show()
                }
            };
            if (ac) {
                ad.setStyle("visibility", "hidden");
                ac.append(ah)
            } else {
                af = ad.ancestor();
                af.append(ah);
                ad.hide()
            }
            N.callMethod(Y, aa, ag)
        }
        function L(aa) {
            var Y = aa.el.ancestor(".card-groupinvite");
            var ab = {
                photo_id: aa.photoId,
                group_id: aa.groupNsid,
                user_id: A.config.flickr.user.nsid
            };
            var ac = ["flickr", "groups", "invite", "photo"];
            ac.push(aa.accept ? "accept" : "decline");
            Y.addClass("card-state-loading");
            N.callMethod(ac.join("."), ab, {
                success: function(ad) {
                    Y.removeClass("card-state-loading");
                    Y.addClass("card-state-complete" + (aa.accept ? "" : "-declined"))
                },
                failure: function(ad) {
                    Y.removeClass("card-state-loading")
                }
            })
        }
        function E(aa, ab) {
            var Y = A.Node.create("<img/>");
            Y.on("load", ab);
            Y.setAttribute("src", aa)
        }
        function X(ad) {
            var aa = ad.ancestor(".imgContainer"), ag = aa.one(".main-session-photo"), af = aa.one(".playButton"), ak = aa.one(".titleText"), ai = A.Lang.trim(ad.getData("title")) || "Untitled", ah = "", Y, ae, aj, ab, ac = A.Node.create("<img/>");
            aa.one(".imgWrapper > a").set("href", ad.getData("link"));
            aa.one(".activity-item-date").set("text", ad.getData("time_ago"));
            if (ad.getData("is_fav") == "1") {
                ah = "is_fav"
            }
            aa.one(".il_fav").setAttribute("class", "il_fav " + ah + " " + ad.getData("fav_class")).one(".li_counter").set("text", ad.getData("fav_count"));
            aa.one(".il_comments").setAttribute("class", "il_comments " + ad.getData("comment_class")).one(".li_counter").set("text", ad.getData("comment_count"));
            aj = ad.getData("is_video_class");
            ae = aa.one(".imgWrapper");
            if (aj) {
                ae.addClass("is_video");
                ae.setAttribute("data-photo-media", "video")
            } else {
                ae.removeClass("is_video");
                ae.setAttribute("data-photo-media", "photo")
            }
            ae.setAttribute("data-photo-id", ad.getData("photo-id"));
            ae.setAttribute("data-comments-total-count", ad.getData("comment_count"));
            ac.setAttribute("alt", ai);
            ac.setAttribute("width", ad.getData("width"));
            ac.setAttribute("height", ad.getData("height"));
            ab = ad.getData("secret");
            if (ab) {
                af.setAttribute("data-secret", ab)
            } else {
                af.setAttribute("data-secret", "")
            }
            ak.set("text", ai);
            ac.setStyles({
                marginTop: ad.getData("mtop"),
                marginLeft: ad.getData("mleft"),
                width: ad.getData("width") + "px",
                height: ad.getData("height") + "px"
            });
            ac.addClass("main-session-photo");
            ac.setAttribute("src", ad.getAttribute("src"));
            ag.replace(ac);
            if (!ad.hasClass("is-preloaded")&&!ad.hasClass("is-loading")) {
                aa.addClass("is-loading");
                Y = ad.getData("img-src");
                E(Y, function() {
                    ad.addClass("is-preloaded");
                    aa.removeClass("is-loading");
                    ac.setAttribute("src", Y)
                });
                aa.one(".sessionPhotos").all(".session-img").each(function(al) {
                    al.addClass("is-loading");
                    E(al.getData("img-src"), function() {
                        al.addClass("is-preloaded");
                        al.removeClass("is-loading")
                    })
                })
            } else {
                ac.setAttribute("src", ad.getData("img-src"))
            }
        }
        function I(af, ab, ad) {
            var ae, ac, aa, Y = "/apps/video/stewart.swf.v" + A.config.flickr.video_player_version;
            af.setStyles({
                height: af.getComputedStyle("height"),
                width: af.getComputedStyle("width"),
                display: "block"
            });
            ac = "stewart_swf" + af.getData("photo-id");
            af.setAttribute("id", ac);
            ae = new deconcept.SWFObject(Y, ac + "-swf", "100%", "100%", 9, "#000000");
            ae.addParam("allowFullScreen", "true");
            ae.addParam("allowScriptAccess", "always");
            ae.addParam("bgcolor", "#000000");
            ae.addParam("wmode", "opaque");
            ae.addVariable("div_id", ac);
            ae.addVariable("intl_lang", A.config.flickr.lang);
            ae.addVariable("onsite", "true");
            ae.addVariable("intl_lang", A.config.flickr.lang);
            ae.addVariable("photo_secret", ab);
            ae.addVariable("photo_id", af.getData("photo-id"));
            ae.addVariable("flickr_noAutoPlay", false);
            if (ad) {
                aa = af.get("parentNode").one(".sessionPhotos");
                aa.addClass("hidden")
            }
            g[ac] = {
                node: af,
                code: af.getHTML(),
                sessionPhotos: aa
            };
            ae.write(ac);
            af.append("<span class='closeVideo' data-id='" + ac + "'>×</span>")
        }
        function w(ac) {
            var ab = ac.target, Y, aa;
            if (ab.hasClass("reco-add-button")) {
                ac.halt();
                x(ab, ab.getAttribute("data-user-nsid"))
            } else {
                if (ab.hasClass("reco-groupinvite-button")) {
                    ac.halt();
                    L({
                        accept: !!ab.hasClass("reco-cta-accept"),
                        el: ab,
                        photoId: ab.getAttribute("data-photo-id"),
                        groupNsid: ab.getAttribute("data-group-nsid")
                    })
                } else {
                    if (ab.hasClass("playButton")) {
                        ac.halt();
                        Y = ab.getData("secret");
                        aa = ab.get("parentNode");
                        if (ab.hasClass("multi-photo")) {
                            I(aa, Y, true)
                        } else {
                            I(aa, Y)
                        }
                    } else {
                        if (ab.hasClass("closeVideo")) {
                            P(ab.getData("id"))
                        }
                    }
                }
            }
        }
        function B(aa) {
            var Y;
            if (h) {
                window.clearTimeout(h);
                h = false
            }
            if (!u) {
                Y = aa.target;
                if (Y.hasClass("session-img")) {
                    h = window.setTimeout(function() {
                        X(Y)
                    }, 150)
                }
            }
        }
        function U() {
            j.ancestor().removeClass("hidden");
            b.addClass("hidden")
        }
        function K(Y) {
            Y.halt();
            j.ancestor().addClass("hidden");
            b.removeClass("hidden");
            Z();
            c = false
        }
        function D(ae) {
            var ab = A.DOM.winHeight(), ad = ab * 1.5, ac, aa, Y;
            window.clearTimeout(t);
            u = true;
            t = window.setTimeout(function() {
                u = false
            }, e * 3);
            if (document.documentElement.scrollTop) {
                Y = document.documentElement.scrollTop
            } else {
                Y = document.body.scrollTop
            }
            if (!H) {
                ac = document.body.offsetHeight;
                aa = document.documentElement.clientHeight + ab;
                if (ac <= Y + aa + ad) {
                    O()
                }
            }
            if (z) {
                z.scrollHandler(ae, Y)
            }
        }
        function Z() {
            var Y = "", ab, aa, ac = 0;
            b.removeClass("hidden");
            if (YUI.storedActionCards.length < 5&&!i) {
                q = false;
                M()
            } else {
                q = true
            }
            if (YUI.storedPhotosFromContactsCards.length < 5&&!d) {
                a = false;
                y()
            } else {
                a = true
            }
            if (q && a) {
                r = false;
                for (ab = 0; ab < 5; ab++) {
                    if (YUI.storedActionCards.length && YUI.storedPhotosFromContactsCards.length) {
                        if ((YUI.storedActionCards[0].timestamp * 1) > (YUI.storedPhotosFromContactsCards[0].timestamp * 1)) {
                            Y += YUI.storedActionCards.shift().html
                        } else {
                            Y += YUI.storedPhotosFromContactsCards.shift().html
                        }
                    } else {
                        if (!i) {
                            Y += YUI.storedActionCards.shift().html
                        } else {
                            if (!d) {
                                Y += YUI.storedPhotosFromContactsCards.shift().html
                            } else {}
                        }
                    }
                    k++;
                    ac++;
                    if (k == 100) {
                        c = true;
                        k = 0;
                        U();
                        break
                    }
                }
            }
            if (Y) {
                var ad = f.append(Y);
                if (o) {
                    o.loadCommentsCards(ad)
                } else {
                    console.warn("[" + m + "]", "comment module not loaded correctly")
                }
                if (z) {
                    z.updateConfig()
                }
                v = T.all("img.defer");
                A.deferImages(v, {
                    threshold: A.DOM.winHeight()
                });
                if (A.config.flickr.flags.enable_sihp_viewcount) {
                    v = v.filter(":not(.buddyicon)");
                    A.ViewCount.onVisible(v)
                }
                A.fire("SIHP:appended", {
                    count: ac
                })
            }
            if (i && d) {
                C()
            }
        }
        function y() {
            var ab, Y, aa;
            r = true;
            ab = "flickr.refresh.activity.getPhotoFeed";
            Y = {
                lWidth: p,
                page: R,
                per_page: G,
                $id_collection: YUI.uniquePhotoIds
            };
            aa = {
                success: function(ae) {
                    var af, ac, ad;
                    if (ae && ae.data && ae.data.cards && ae.data.cards.length) {
                        for (af = 0, ac = ae.data.cards.length; af < ac; af++) {
                            ad = ae.data.cards[af].id;
                            if (YUI.photosFromFriendsIds[0] == ad) {
                                d = true
                            }
                            if (A.Array.indexOf(YUI.uniquePhotoIds, ad)===-1 && A.Array.indexOf(YUI.photosFromFriendsIds, ad)===-1) {
                                YUI.uniquePhotoIds.push(ad);
                                YUI.storedPhotosFromContactsCards.push(ae.data.cards[af]);
                                YUI.photosFromFriendsIds.push(ad)
                            }
                        }
                    } else {
                        d = true
                    }
                    Z()
                },
                failure: function(ac) {
                    d = true;
                    r = false
                }
            };
            N.callMethod(ab, Y, aa);
            R++
        }
        function M() {
            var ab, Y, aa;
            r = true;
            ab = "flickr.refresh.activity.getActivityFeed";
            Y = {
                lWidth: p,
                page: S,
                per_page: G
            };
            aa = {
                success: function(ae) {
                    var ac = "", af, ad;
                    if (ae && ae.data && ae.data.cards && ae.data.cards.length) {
                        for (af = 0, ad = ae.data.cards.length; af < ad; af++) {
                            if (A.Array.indexOf(YUI.uniquePhotoIds, ae.data.cards[af].id)===-1) {
                                YUI.uniquePhotoIds.push(ae.data.cards[af].id);
                                YUI.storedActionCards.push(ae.data.cards[af])
                            }
                        }
                    } else {
                        i = true
                    }
                    Z()
                },
                failure: function(ac) {
                    r = false;
                    i = true
                }
            };
            N.callMethod(ab, Y, aa);
            S++
        }
        function C() {
            if (b) {
                b.addClass("hidden")
            }
        }
        function O() {
            if (!r&&!c) {
                r = true;
                if (A.shareThisV3Menu.isInitialized()) {
                    A.shareThisV3Menu.hide();
                    A.shareThisV3Menu.clear()
                }
                Z()
            }
        }
        A.refreshSIHP = {
            init: Q
        }
    }, "0.0.1", {
        requires: F.config.modules[m].requires || [],
        optional: F.config.modules[m].optional || []
    })
}());
function udm_(J) {
    var I = "comScore=", H = document, G = H.cookie, F = "", E = "indexOf", D = "substring", C = "length", B = 2048, A, z = "&ns_", y = "&", x, w, v, u, t = window, s = t.encodeURIComponent || escape;
    if (G[E](I) + 1) {
        for (v = 0, w = G.split(";"), u = w[C]; v < u; v++) {
            x = w[v][E](I), x + 1 && (F = y + unescape(w[v][D](x + I[C])))
        }
    }
    J += z + "_t=" + + (new Date) + z + "c=" + (H.characterSet || H.defaultCharset || "") + F, J[C] > B && J[E](y) > 0 && (A = J[D](0, B - 8).lastIndexOf(y), J = (J[D](0, A) + z + "cut=" + s(J[D](A + 1)))[D](0, B)), H.images ? (x = new Image, t.ns_p || (ns_p = x), x.src = J) : H.write("<", "p", "><", 'img src="', J, '" height="1" width="1" alt="*"', "><", "/p", ">")
}
typeof _comscore == "undefined" && (_comscore = []), function() {
    var v = "length", u = self, t = u.encodeURIComponent ? encodeURIComponent: escape, s = ".scorecardresearch.com", r = Math, q = "script", p = "width", o = /c2=(\d*)&/, n, m = function(D) {
        if (!!D) {
            var C, B = [], A, z = 0, y, x, w = "";
            for (var d in D) {
                A = typeof D[d];
                if (A == "string" || A == "number") {
                    B[B[v]] = d + "=" + t(D[d]), d == "c2" ? w = D[d] : d == "c1" && (z = 1)
                }
            }
            if (B[v] <= 0 || w == "") {
                return 
            }
            x = D.options || {}, x.d = x.d || document;
            if (typeof x.url_append == "string") {
                y = x.url_append.replace(/&amp;/, "&").split("&");
                for (var d = 0, c = y[v], a; d < c; d++) {
                    a = y[d].split("="), a[v] == 2 && (B[B[v]] = a[0] + "=" + t(a[1]))
                }
            }
            C = ["http", x.d.URL.charAt(4) == "s" ? "s://sb": "://b", s, "/p?", z ? "": "c1=2&", B.join("&").replace(/&$/, "")], udm_(C.join(""))
        }
    }, l = function(a) {
        a = a || _comscore;
        for (var f = 0, e = a[v]; f < e; f++) {
            m(a[f])
        }
        a = _comscore = []
    };
    l(), (n = u.COMSCORE) ? (n.purge = l, n.beacon = m) : COMSCORE = {
        purge: l,
        beacon: m
    }
}();
