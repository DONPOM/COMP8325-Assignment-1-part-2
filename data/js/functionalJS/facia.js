define("facia/modules/onwards/geo-most-popular-front", ["qwery", "common/utils/$", "common/utils/config", "common/utils/mediator", "common/modules/analytics/register", "common/modules/component"], function(t, e, n, o, a, i) {
    function s() {
        a.begin("most-popular")
    }
    return i.define(s), s.prototype.endpoint = "/most-read-geo.json", s.prototype.isNetworkFront = "Network Front" === n.page.contentType, s.prototype.manipulationType = "html", s.prototype.prerender = function() {
        this.elem = t(".headline-list", this.elem)[0]
    }, s.prototype.go = function() {
        var e = this.isNetworkFront ? ".js-tab-1": ".js-tab-2";
        this.parent = t(".js-popular-trails")[0], this.parent && (this.tab = t(e, this.parent)[0], this.fetch(this.tab, "html"))
    }, s.prototype.ready = function() {
        this.isNetworkFront && (e(".js-tabs-content", this.parent).addClass("tabs__content--no-border"), e(".js-tabs", this.parent).addClass("u-h")), a.end("most-popular"), o.emit("modules:geomostpopular:ready")
    }, s
}), define("facia/modules/ui/container-toggle", ["bean", "bonzo", "common/utils/$", "common/modules/userPrefs"], function(t, e, n, o) {
    return function(a) {
        var i = e(a), s = e(e.create('<button class="fc-container__toggle" data-link-name="Show"><i class="i i-arrow-grey-large"></i><span class="fc-container__toggle__text">Hide</span></button>')), r = "container-states", d = {
            hidden: "Show",
            displayed: "Hide"
        }, l = "displayed", c = function(t, e) {
            var n = o.get(r), a = t.attr("data-id");
            "displayed" === e ? delete n[a] : (n || (n = {}), n[a] = "closed"), o.set(r, n)
        }, u = function(e) {
            var n = o.get(r);
            n && n[e.attr("data-id")] && t.fire(s[0], "click")
        };
        o.remove("front-trailblocks"), this.addToggle = function() {
            n(".js-container__header", i[0]).append(s), i.removeClass("js-container--toggle").addClass("fc-container--has-toggle"), t.on(s[0], "click", function() {
                l = "displayed" === l ? "hidden" : "displayed", i["displayed" === l ? "removeClass": "addClass"]("fc-container--rolled-up"), s.attr("data-link-name", d["displayed" === l ? "hidden": "displayed"]), n(".fc-container__toggle__text", s[0]).text(d[l]), n(".ad-slot--paid-for-badge", a).css("display", "hidden" === l ? "none" : "block"), c(i, l)
            }), u(i)
        }
    }
}), define("text!facia/views/button-show-more.html", [], function() {
    return '<button class="button button--medium button--show-more button--primary collection__show-more"\n        data-test-id="show-more"\n        data-link-name="{{type}}">\n    <i class="i i-plus-white"></i><span class="js-button-text">{{type}}</span>\n</button>\n'
}), define("facia/modules/ui/container-show-more", ["bean", "bonzo", "qwery", "common/utils/$", "common/utils/template", "common/modules/userPrefs", "text!facia/views/button-show-more.html"], function(t, e, n, o, a, i, s) {
    return function(r) {
        function d() {
            l.toggleClass(m, "displayed" === p), p = "hidden" === p ? "displayed" : "hidden", y.changeButtonText(), y.changeButtonState(), y.updatePref(l, p)
        }
        var l = e(r), c = n(".js-hide", l).length > 0, u = n(".js-hide-on-mobile", l).length > 0, m = "fc-show-more--hidden", h = "js-button-text", f = null, p = "hidden", g = "section-states", w = {}, y = this;
        this.addShowMoreButton = function() {
            var e = this.getContainerTitle();
            w = {
                hidden: "More " + e,
                displayed: "Less " + e
            }, f = o.create(a(s, {
                type: w[p]
            })), (u || c) && (c || l.addClass("fc-show-more--mobile-only"), l.addClass(m).append(f).removeClass("js-container--fc-show-more"), t.on(f[0], "click", d)), this.readPrefs(l)
        }, this.getContainerTitle = function() {
            return l.data("title") || ""
        }, this.changeButtonText = function() {
            o("." + h, f).text("hidden" === p ? w[p] : w[p].split(" ")[0])
        }, this.changeButtonState = function() {
            f.attr("data-link-name", w[p]).toggleClass("button--primary", "displayed" !== p).toggleClass("button--tertiary", "displayed" === p), o(".i", f).toggleClass("i-plus-white", "displayed" !== p).toggleClass("i-minus-blue", "displayed" === p)
        }, this.updatePref = function(t, e) {
            var n = i.get(g, {
                type: "session"
            }), o = t.attr("data-id");
            "displayed" !== e ? delete n[o] : (n || (n = {}), n[o] = "more"), i.set(g, n, {
                type: "session"
            })
        }, this.readPrefs = function(e) {
            var n = i.get(g, {
                type: "session"
            });
            n && n[e.attr("data-id")] && t.fire(f[0], "click")
        }
    }
}), define("facia/modules/ui/football-snaps", ["bonzo"], function(t) {
    var e = {
        resizeIfPresent: function(e) {
            if (e) {
                var n = t(e);
                n.css("height", n.parent().css("height"))
            }
        }
    };
    return e
}), define("facia/modules/ui/snaps", ["bonzo", "lodash/functions/debounce", "common/utils/$", "common/utils/ajax", "common/utils/mediator", "common/utils/template", "common/utils/to-array", "common/modules/ui/relativedates", "facia/modules/ui/football-snaps"], function(t, e, n, o, a, i, s, r, d) {
    function l() {
        var t = s(n(".js-snappable.js-snap")).filter(function(t) {
            var e = t.getAttribute("data-snap-type");
            return e && p.indexOf(e)>-1
        }).filter(function(t) {
            return t.getAttribute("data-snap-uri")
        });
        t.forEach(f), t.length && a.on("window:resize", e(function() {
            t.forEach(function(t) {
                c(t, !0)
            })
        }, 200))
    }
    function c(t, e) {
        u(t, e), n(t).hasClass("facia-snap--football") && d.resizeIfPresent(t)
    }
    function u(e, n) {
        var o = e.offsetWidth, a = t(e), i = "facia-snap-point--";
        [{
            width: 0,
            name: "tiny"
        }, {
            width: 180,
            name: "mini"
        }, {
            width: 220,
            name: "small"
        }, {
            width: 300,
            name: "medium"
        }, {
            width: 700,
            name: "large"
        }, {
            width: 940,
            name: "huge"
        }
        ].map(function(t, e, a) {
            var i = o >= t.width && (a[e + 1] ? o < a[e + 1].width : !0);
            return t.action = i ? "addClass" : n ? "removeClass" : !1, t
        }).filter(function(t) {
            return t.action
        }).forEach(function(t) {
            a[t.action](i + t.name)
        })
    }
    function m(e) {
        var n = t(e).offset(), o = Math.ceil((n.width || 0) / 2), a = 400;
        t(e).html(i('<div style="height:{{height}}px; overflow:hidden; width: 100%;"><iframe src="{{src}}" style="height:{{height}}px; width: 100%; border: none;"></iframe></div>', {
            src: e.getAttribute("data-snap-uri"),
            height: Math.min(Math.max(n.height || 0, o), a)
        }))
    }
    function h(e, a) {
        o({
            url: e.getAttribute("data-snap-uri"),
            type: a ? "json": "html",
            crossOrigin: !0
        }).then(function(o) {
            n.create(a ? o.html : o).each(function(n) {
                t(e).html(n)
            }), r.init(e)
        })
    }
    function f(e) {
        switch (t(e).addClass("facia-snap-embed"), c(e), e.getAttribute("data-snap-type")) {
        case"document":
            m(e);
            break;
        case"fragment":
            h(e);
            break;
        case"json.html":
            h(e, !0)
        }
    }
    var p = ["document", "fragment", "json.html"];
    return {
        init: l
    }
}), define("bootstraps/facia", ["bonzo", "qwery", "common/utils/$", "common/utils/ajax", "common/utils/config", "common/utils/detect", "common/utils/mediator", "common/utils/storage", "common/utils/to-array", "facia/modules/onwards/geo-most-popular-front", "facia/modules/ui/container-toggle", "facia/modules/ui/container-show-more", "facia/modules/ui/snaps"], function(t, e, n, o, a, i, s, r, d, l, c, u, m) {
    var h = {
        showSnaps: function() {
            m.init(), s.on("modules:container:rendered", m.init)
        },
        showContainerShowMore: function() {
            var t = function() {
                var t = document;
                n(".js-container--fc-show-more", t).each(function(t) {
                    new u(t).addShowMoreButton()
                })
            };
            s.addListeners({
                "modules:container:rendered": t,
                "page:front:ready": t
            })
        },
        showContainerToggle: function() {
            var t = document, e = function() {
                n(".js-container--toggle", t).each(function(t) {
                    new c(t).addToggle()
                })
            };
            s.addListeners({
                "page:front:ready": e,
                "ui:container-toggle:add": e,
                "modules:geomostpopular:ready": e
            }), s.on(/page:front:ready|ui:container-toggle:add|modules:geomostpopular:ready/, function() {
                n(".js-container--toggle", t).each(function(t) {
                    new c(t).addToggle()
                })
            })
        },
        upgradeMostPopularToGeo: function() {
            a.switches.geoMostPopular && (new l).go()
        }
    }, f = function() {
        this.initialised || (this.initialised=!0, h.showSnaps(), h.showContainerShowMore(), h.showContainerToggle(), h.upgradeMostPopularToGeo()), s.emit("page:front:ready")
    };
    return {
        init: f
    }
});
//# sourceMappingURL=facia.js.map
