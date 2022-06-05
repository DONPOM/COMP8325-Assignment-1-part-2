define("common/modules/commercial/ad-block-test", ["lodash/functions/once", "common/utils/$", "common/utils/$css", "common/modules/analytics/beacon"], function(e, n, i, o) {
    return e(function() {
        var e=!1, t = n.create('<div class="ad_unit"></div>').appendTo(document.body);
        return "none" === i(t, "display") && (e=!0, o.counts("ads-blocked")), t.remove(), e
    })
}), define("common/modules/commercial/article-aside-adverts", ["qwery", "lodash/objects/defaults", "lodash/functions/once", "common/utils/$", "common/utils/$css", "common/utils/config", "common/modules/commercial/create-ad-slot"], function(e, n, i, o, t, a, s) {
    function c(e) {
        var i, c, r = n(e || {}, {
            columnSelector: ".content__secondary-column",
            adSlotContainerSelector: ".js-mpu-ad-slot"
        }), l = o(r.columnSelector), m = l.length && "none" === t(l, "display");
        return a.switches.standardAdverts && /Article|LiveBlog/.test(a.page.contentType)&&!m ? (i = "Article" === a.page.contentType ? o(".js-content-main-column") : !1, c=!i.length || i.dim().height >= 600 ? "right" : "right-small", o(r.adSlotContainerSelector).append(s(c, "mpu-banner-ad"))) : !1
    }
    return {
        init: i(c)
    }
}), define("common/modules/article/spacefinder", ["qwery", "common/utils/_"], function(e, n) {
    function i(e) {
        return {
            top: e.offsetTop,
            bottom: e.offsetTop + e.offsetHeight,
            element: e
        }
    }
    function o(o, t, s) {
        var c = n(o).filter(function(e) {
            return e.top >= t.minAbove && e.top + t.minBelow <= s
        });
        return n(t.selectors).forOwn(function(o, t) {
            var s = n(e(a + t)).map(i);
            c = c.filter(function(e) {
                return r(e, s, o)
            })
        }), c.valueOf()
    }
    function t(t) {
        t = t || s;
        var c = e(a)[0].offsetHeight, r = n(e(a + " > *")).map(i), l = r.filter(function(e) {
            return "p" === e.element.tagName.toLowerCase()
        }), m = o(l, t, c);
        return m.length ? m[0].element : void 0
    }
    var a = ".js-article__body", s = {
        minAbove: 250,
        minBelow: 300,
        selectors: {
            " > h2": {
                minAbove: 0,
                minBelow: 250
            },
            " > *:not(p):not(h2)": {
                minAbove: 25,
                minBelow: 250
            }
        }
    }, c = function(e, n, i) {
        var o = e.top - n.bottom >= i.minAbove, t = n.top - e.bottom >= i.minBelow;
        return o || t
    }, r = function(e, i, o) {
        return n(i).every(function(n) {
            return c(e, n, o)
        }).valueOf()
    };
    return {
        getParaWithSpace: t,
        _testElem: c,
        _testElems: r
    }
}), define("common/modules/commercial/article-body-adverts", ["lodash/functions/once", "lodash/objects/cloneDeep", "common/utils/$", "common/utils/config", "common/utils/detect", "common/modules/article/spacefinder", "common/modules/commercial/create-ad-slot"], function(e, n, i, o, t, a, s) {
    var c = [], r = [["inline1", "inline"], ["inline2", "inline"]], l = function(e) {
        if (e) {
            var n = r[c.length], o = i.create(s(n[0], n[1])).insertBefore(e);
            c.push(o)
        }
    }, m = function() {
        var e, i, s;
        return !o.switches.standardAdverts || "Article" !== o.page.contentType || o.page.isLiveBlog?!1 : (e = t.getBreakpoint(), i = {
            minAbove: t.isBreakpoint({
                max: "tablet"
            }) ? 300: 700,
            minBelow: 300,
            selectors: {
                " > h2": {
                    minAbove: "mobile" === e ? 20: 0,
                    minBelow: 250
                },
                " > *:not(p):not(h2)": {
                    minAbove: 35,
                    minBelow: 400
                },
                " .ad-slot": {
                    minAbove: 500,
                    minBelow: 500
                }
            }
        }, s = n(i), s.minAbove = 300, "foundation-features" === o.page.sponsorshipType && o.page.isInappropriateForSponsorship===!1 && (r.unshift(["fobadge", ["im", "paid-for-badge"]]), l(a.getParaWithSpace(s))), o.page.hasInlineMerchandise && (r.unshift(["im", "im"]), l(a.getParaWithSpace(s))), l(a.getParaWithSpace(i)), void(t.isBreakpoint({
            max: "tablet"
        }) && l(a.getParaWithSpace(i))))
    };
    return {
        init: e(m),
        destroy: function() {
            c.forEach(function(e) {
                e.remove()
            })
        }
    }
}), define("common/modules/commercial/front-commercial-components", ["bonzo", "lodash/functions/once", "common/utils/$", "common/utils/config", "common/modules/commercial/create-ad-slot"], function(e, n, i, o, t) {
    function a() {
        if (!o.switches.commercialComponents ||!o.page.isFront || o.page.hasPageSkin)
            return !1;
        var n, a = e(t("merchandising-high", "commercial-component-high")), s = i(".fc-container");
        return s.length >= 2 ? (n = 0, s.length >= 4 && (n = "Network Front" === o.page.contentType ? 3 : 2), a.insertAfter(s[n])) : void 0
    }
    return {
        init: n(a)
    }
}), define("common/modules/commercial/loader", ["bean", "bonzo", "raven", "lodash/collections/map", "lodash/collections/size", "lodash/objects/defaults", "lodash/objects/isArray", "lodash/objects/pick", "common/utils/$", "common/utils/_", "common/utils/config", "common/utils/mediator", "common/modules/component", "common/modules/lazyload", "common/modules/ui/tabs"], function(e, n, i, o, t, a, s, c, r, l, m, d, u, p, g) {
    var h = function(e) {
        return l(e).pairs().map(function(e) {
            var n = e[0], i = s(e[1]) ? e[1]: [e[1]];
            return o(i, function(e) {
                return [n, "=", encodeURIComponent(e)].join("")
            }).join("&")
        }).join("&")
    }, f = function() {
        var e = m.page.keywordIds ? o(m.page.keywordIds.split(","), function(e) {
            return e.split("/").pop()
        }): m.page.pageId.split("/").pop();
        return {
            k: e
        }
    }, v = function(e, n) {
        var i = c(a(n || {}, f()), function(e) {
            return s(e) ? e.length : e
        }), o = t(i) ? "?" + h(i): "";
        return [m.page.ajaxUrl, "/commercial/", e, ".json", o].join("")
    }, b = function(e) {
        var n = a(e || {}, {
            capi: [],
            capiAboutLinkUrl: "",
            capiKeywords: "",
            capiLinkUrl: "",
            capiTitle: "",
            components: [],
            jobIds: "",
            logo: "",
            oastoken: ""
        }), i = m.page.section, o = n.jobIds ? n.jobIds.split(","): [];
        this.oastoken = n.oastoken, this.components = {
            bestbuy: v("money/bestbuys"),
            bestbuyHigh: v("money/bestbuys-high"),
            book: v("books/book", {
                t: m.page.isbn
            }),
            books: v("books/bestsellers"),
            booksMedium: v("books/bestsellers-medium"),
            booksHigh: v("books/bestsellers-high"),
            jobs: v("jobs", {
                t: o
            }),
            jobsHigh: v("jobs-high"),
            masterclasses: v("masterclasses"),
            masterclassesHigh: v("masterclasses-high"),
            soulmates: v("soulmates/mixed"),
            soulmatesHigh: v("soulmates/mixed-high"),
            travel: v("travel/offers", {
                s: i
            }),
            travelHigh: v("travel/offers-high", {
                s: i
            }),
            multi: v("multi", {
                c: n.components
            }),
            capiSingle: v("capi-single", a(e, {
                s: i
            })),
            capiSingleMerch: v("capi-single-merch", a(e, {
                s: i
            })),
            capi: v("capi", a(e, {
                s: i,
                t: n.capi,
                k: n.capiKeywords.split(","),
                l: n.logo,
                ct: n.capiTitle,
                cl: n.capiLinkUrl,
                cal: n.capiAboutLinkUrl
            }))
        }
    };
    return u.define(b), b.prototype.postLoadEvents = {
        bestbuy: function(e) {
            (new g).init(e)
        }
    }, b.prototype.load = function(e, n) {
        return new p({
            url: this.components[e],
            container: n,
            beforeInsert: function(e) {
                return e ? e.replace(/%OASToken%/g, this.oastoken).replace(/%OmnitureToken%/g, "") : e
            }.bind(this),
            success: function() {
                this.postLoadEvents[e] && this.postLoadEvents[e](n), d.emit("modules:commercial:loader:loaded")
            }.bind(this)
        }).load(), this
    }, b.prototype.init = function(e, n) {
        return void 0 === this.components[e] ? (i.captureMessage("Unknown commercial component: " + e), !1) : this.load(e, n)
    }, b
}), define("common/modules/commercial/slice-adverts", ["bonzo", "qwery", "lodash/collections/contains", "lodash/functions/once", "lodash/objects/defaults", "common/utils/$", "common/utils/_", "common/utils/config", "common/utils/template", "common/modules/commercial/create-ad-slot", "common/modules/userPrefs"], function(e, n, i, o, t, a, s, c, r, l, m) {
    var d = ["inline1", "inline2"], u = function(o) {
        if (!c.switches.standardAdverts)
            return !1;
        for (var r, u, p, g, h = t(o || {}, {
            containerSelector: ".fc-container",
            sliceSelector: ".js-fc-slice-mpu-candidate"
        }), f = n(h.containerSelector), v = 0, b = [], _ = 1, k = m.get("container-states"); v < f.length;)
            r = f[v], u = e(r).data("id"), p = a(h.sliceSelector, r), g = i(["uk", "us", "au"], c.page.pageId) && 0 === v, !p.length || g || k && "closed" === k[u] ? v++ : (b.push(p.first()), v += _ + 1);
        return s(b).slice(0, d.length).forEach(function(n, i) {
            var o = d[i], t = e(l(o, "container-inline")).addClass("ad-slot--mobile"), s = e(l(o, "container-inline")).addClass("ad-slot--not-mobile");
            n.removeClass("fc-slice__item--no-mpu").append(s), t.insertAfter(a.ancestor(n[0], "fc-container"))
        }).valueOf(), b
    };
    return {
        init: o(u)
    }
}), define("common/modules/commercial/outbrain", ["common/utils/$", "common/utils/config", "common/utils/detect"], function(e, n, i) {
    function o() {
        if (n.switches.outbrain) {
            var o = {
                mobile: "MB_2",
                tablet: "MB_1",
                desktop: "AR_11",
                wide: "AR_11"
            };
            return e(".OUTBRAIN").first().attr("data-widget-id", o[i.getBreakpoint()]), require(["js!" + t + "!exports=outbrain"])
        }
    }
    var t = "//widgets.outbrain.com/outbrain.js";
    return {
        load: o
    }
}), define("common/modules/commercial/tags/amaa", ["common/utils/config"], function(e) {
    function n() {
        return e.switches.amaa ? require(["js!" + i + "!exports=superT"]) : void 0
    }
    var i = "//c.supert.ag/the-guardian/the-guardian/supertag-async.js";
    return {
        load: n
    }
}), define("common/modules/commercial/tags/effective-measure", ["common/utils/config"], function(e) {
    function n() {
        return e.switches.effectiveMeasure ? require(["js!" + i + "!exports=_EMeasure"]) : void 0
    }
    var i = ("https:" === document.location.protocol ? "https://au-ssl" : "http://au-cdn") + ".effectivemeasure.net/em.js";
    return {
        load: n
    }
}), define("common/modules/commercial/tags/imrworldwide", ["common/utils/config"], function(e) {
    function n() {
        if (e.switches.imrWorldwide) {
            var n = new Image;
            return n.src = ["//secure-uk.imrworldwide.com/cgi-bin/m?ci=uk-305078h&cg=0&cc=1&ts=compact", "&si=", encodeURI(window.location.href), "&rp=", encodeURI(document.referrer), "&rnd=", (new Date).getTime()].join(""), n
        }
    }
    return {
        load: n
    }
}), define("common/modules/commercial/tags/media-math", ["lodash/collections/contains", "lodash/objects/defaults", "common/utils/_", "common/utils/config", "common/utils/url"], function(e, n, i, o, t) {
    var a = "//pixel.mathtag.com/event/img?mt_id=328671&mt_adid=114751", s = function(n) {
        return i(n.split("?").pop().split("&")).filter(function(n) {
            return e(["q", "p", "as_q", "as_epq", "as_oq", "query", "search", "wd", "ix"], n.split("=").shift())
        }).map(function(e) {
            return decodeURIComponent(e.split("=").pop().replace(/\\+/g, " "))
        }).shift()
    };
    return {
        load: function(e) {
            var n, i, c, r = (e || {}).referrer || document.referrer;
            return o.switches.mediaMath ? (n = o.page, i = {
                v1: (n.host ? n.host : "") + "/" + n.pageId,
                v2: n.section,
                v3: s(r),
                v4: r,
                v5: n.keywords ? n.keywords.replace(/,/g, "|"): "",
                v6: n.contentType ? n.contentType.toLowerCase(): ""
            }, c = new Image, c.src = a + "&" + t.constructQuery(i), c) : !1
        }
    }
}), define("common/modules/commercial/tags/remarketing", ["common/utils/config"], function(e) {
    function n() {
        return e.switches.remarketing ? require(["js!" + i + "!exports=google_trackConversion"], function() {
            window.google_trackConversion({
                google_conversion_id: 971225648,
                google_custom_params: window.google_tag_params,
                google_remarketing_only: !0
            })
        }) : void 0
    }
    var i = "//www.googleadservices.com/pagead/conversion_async.js";
    return {
        load: n
    }
}), define("common/modules/commercial/tags/krux", ["common/utils/config"], function(e) {
    function n() {
        return e.switches.krux ? require(["js!//cdn.krxd.net/controltag?confid=JVZiE3vn"]) : void 0
    }
    return {
        load: n
    }
}), define("common/modules/commercial/tags/container", ["lodash/objects/defaults", "common/utils/config", "common/modules/commercial/outbrain", "common/modules/commercial/tags/amaa", "common/modules/commercial/tags/audience-science", "common/modules/commercial/tags/audience-science-gateway", "common/modules/commercial/tags/criteo", "common/modules/commercial/tags/effective-measure", "common/modules/commercial/tags/imrworldwide", "common/modules/commercial/tags/media-math", "common/modules/commercial/tags/remarketing", "common/modules/commercial/tags/krux"], function(e, n, i, o, t, a, s, c, r, l, m, d) {
    function u() {
        if ("Identity" === n.page.contentType || "identity" === n.page.section)
            return !1;
        switch (n.page.edition.toLowerCase()) {
        case"au":
            o.load(), c.load();
            break;
        default:
            t.load(), a.load()
        }
        l.load(), s.load(), r.load(), m.load(), i.load(), d.load()
    }
    return {
        init: u
    }
}), define("text!common/views/commercial/creatives/branded-component-jobs.html", [], function() {
    return '<div class="creative--branded-component creative--jobs" data-link-name="creative | branded component | jobs">\n    <div class="creative__inner">\n        <a href="{{clickMacro}}http://jobs.theguardian.com/" data-link-name="logo link">\n            <i class="creative__marque i i-marque-36-333"></i>\n        </a>\n        <h3 class="creative__marque-title">Jobs</h3>\n        <img src="{{imgUrl}}" width="198" height="137" />\n        <p class="creative__title">Finding quality jobs just got easier</p>\n        <p class="creative__text">Introducing the next-generation Guardian Jobs website, designed to work whatever your device</p>\n        <a class="button button--primary" href="{{clickMacro}}http://jobs.theguardian.com/" data-link-name="link">\n            Browse jobs<i class="i i-arrow-white-right i-right"></i>\n        </a>\n    </div>\n</div>\n'
}), define("text!common/views/commercial/creatives/branded-component-membership.html", [], function() {
    return '<div class="creative--branded-component creative--membership" data-link-name="creative | branded component | membership">\n    <div class="creative__inner">\n        <a href="{{clickMacro}}https://membership.theguardian.com/" data-link-name="logo link">\n            <i class="creative__marque i i-marque-36-333"></i>\n        </a>\n        <h3 class="creative__marque-title">Membership</h3>\n        <p class="creative__title">The open exchange of ideas and opinions has the power to change the world</p>\n        <p class="creative__text">Join the community for those who share this belief</p>\n        <a class="button" href="{{clickMacro}}https://membership.theguardian.com/" data-link-name="link">\n            Find out more<i class="i i-arrow-white-right i-right"></i>\n        </a>\n    </div>\n</div>\n'
}), define("text!common/views/commercial/creatives/branded-component-soulmates.html", [], function() {
    return '<div class="creative--branded-component creative--branded-component-soulmates" data-link-name="creative | branded component | soulmates">\n    <div class="creative__inner">\n        <p class="creative__title">Meet someone <em>worth</em> meeting</p>\n        <p class="creative__text">Thousands of new members sign up to Soulmates every month</p>\n        <div class="creative__link-wrapper">\n            <a class="creative__link" href="{{clickMacro}}https://soulmates.theguardian.com/" data-link-name="link">\n                Join now\n                <span class="creative__link__button button button--small">\n                    <span class="i i-arrow-white-right" />\n                </span>\n            </a>\n        </div>\n        <div class="i--commercial i-soulmates-title-background soulmates-logo">\n            <span class="u-h">The Guardian</span>\n            <h4 class="soulmates-logo__title">\n                <a href="{{clickMacro}}https://soulmates.theguardian.com/" data-link-name="logo link">\n                    <span class="i i--commercial i-soulmates-logo"></span>\n                    <span class="u-h">Soulmates</span>\n                </a>\n            </h4>\n        </div>\n    </div>\n</div>\n'
}), define("common/modules/commercial/creatives/branded-component", ["qwery", "common/utils/$", "common/utils/config", "common/utils/template", "text!common/views/commercial/creatives/branded-component-jobs.html", "text!common/views/commercial/creatives/branded-component-membership.html", "text!common/views/commercial/creatives/branded-component-soulmates.html"], function(e, n, i, o, t, a, s) {
    var c = {
        jobs: {
            template: t,
            config: {
                imgUrl: i.images.commercial.brandedComponentJobs
            }
        },
        membership: {
            template: a,
            config: {}
        },
        soulmates: {
            template: s,
            config: {}
        }
    }, r = function(e, n) {
        this.$adSlot = e, this.params = n
    };
    return r.prototype.create = function() {
        var e = c[this.params.type], t = n(".js-secondary-column");
        return !e || "none" === t.css("display") || t.dim().height < 1600 || "football" === i.page.section?!1 : (e.config.clickMacro = this.params.clickMacro, void n.create(o(e.template, e.config)).appendTo(t))
    }, r
}), define("common/modules/commercial/creatives/commercial-component", ["bean", "bonzo", "raven", "lodash/collections/map", "lodash/collections/size", "lodash/objects/defaults", "lodash/objects/isArray", "lodash/objects/pick", "common/utils/$", "common/utils/_", "common/utils/config", "common/utils/mediator", "common/modules/component", "common/modules/lazyload", "common/modules/ui/tabs"], function(e, n, i, o, t, a, s, c, r, l, m, d, u, p, g) {
    var h = function(e) {
        return l(e).pairs().map(function(e) {
            var n = e[0], i = s(e[1]) ? e[1]: [e[1]];
            return o(i, function(e) {
                return [n, "=", encodeURIComponent(e)].join("")
            }).join("&")
        }).join("&")
    }, f = function() {
        var e = m.page.keywordIds ? o(m.page.keywordIds.split(","), function(e) {
            return e.split("/").pop()
        }): m.page.pageId.split("/").pop();
        return {
            k: e
        }
    }, v = function(e, n) {
        var i = c(a(n || {}, f()), function(e) {
            return s(e) ? e.length : e
        }), o = t(i) ? "?" + h(i): "";
        return [m.page.ajaxUrl, "/commercial/", e, ".json", o].join("")
    }, b = function(e, n) {
        var i = m.page.section, o = n.jobIds ? n.jobIds.split(","): [];
        this.params = n, this.$adSlot = e, this.type = n.type, this.components = {
            bestbuy: v("money/bestbuys"),
            bestbuyHigh: v("money/bestbuys-high"),
            book: v("books/book", {
                t: m.page.isbn
            }),
            books: v("books/bestsellers"),
            booksMedium: v("books/bestsellers-medium"),
            booksHigh: v("books/bestsellers-high"),
            jobs: v("jobs", {
                t: o
            }),
            jobsHigh: v("jobs-high"),
            masterclasses: v("masterclasses"),
            masterclassesHigh: v("masterclasses-high"),
            soulmates: v("soulmates/mixed"),
            soulmatesHigh: v("soulmates/mixed-high"),
            travel: v("travel/offers", {
                s: i
            }),
            travelHigh: v("travel/offers-high", {
                s: i
            }),
            multi: v("multi", {
                c: n.components
            }),
            capiSingle: v("capi-single", a(n, {
                s: i
            })),
            capiSingleMerch: v("capi-single-merch", a(n, {
                s: i
            })),
            capi: v("capi", a(n, {
                s: i
            }))
        }
    };
    return b.prototype.postLoadEvents = {
        bestbuy: function(e) {
            (new g).init(e)
        }
    }, b.prototype.load = function() {
        return new p({
            url: this.components[this.type],
            container: this.$adSlot,
            success: function() {
                this.postLoadEvents[this.type] && this.postLoadEvents[this.type](this.$adSlot), d.emit("modules:commercial:creatives:commercial-component:loaded")
            }.bind(this)
        }).load(), this
    }, b.prototype.create = function() {
        return void 0 === this.components[this.type] ? (i.captureMessage("Unknown commercial component: " + name), !1) : this.load()
    }, b
}), define("text!common/views/commercial/creatives/expandable.html", [], function() {
    return '<div class="creative--expandable">\n    <div class="ad-slot__label adFullWidth__label facia-container--layout-front" data-test-id="ad-slot-label">\n        <div class="facia-container__inner">Advertisement</div>\n    </div>\n    <div class="ad-exp--expand l-side-margins">\n        <div class="facia-container__inner facia-container__inner--full-span">\n            <button class="ad-exp__close-button">\n                <i class="i i-close-icon-white-small"></i>\n            </button>\n            <a href="{{clickMacro}}{{link}}" target="_new">\n                <div class="ad-exp-collapse__slide slide-1" style="background-image: url(\'{{slide1}}\');">\n                    <div class="ad-exp__cta" style="background-image: url(\'{{cta}}\');"></div>\n                    <div class="ad-exp__logo" style="background-image: url(\'{{logo}}\');"></div>\n                </div>\n                <div class="ad-exp-collapse__slide slide-2" style="background-image: url(\'{{slide2}}\');">\n                    <div class="ad-exp__text-1 mobile-only" style="background-image: url(\'{{textMobile}}\');"></div>\n                    <div class="ad-exp__text-1 hide-until-tablet" style="background-image: url(\'{{text}}\');"></div>\n                </div>\n            </a>\n        </div>\n    </div>\n    <div class="ad-slot__label--wrapper facia-container__inner facia-container__inner--full-span">\n        <div class="ad-slot__label"></div>\n    </div>\n</div>\n'
}), define("common/modules/commercial/creatives/expandable", ["bean", "bonzo", "common/utils/$", "common/utils/mediator", "common/utils/storage", "common/utils/template", "text!common/views/commercial/creatives/expandable.html"], function(e, n, i, o, t, a, s) {
    var c = function(e, i) {
        this.$adSlot = e, this.params = i, this.isClosed=!0, this.closedHeight = Math.min(n.viewport().height / 3, 300), this.openedHeight = Math.min(2 * n.viewport().height / 3, 600)
    };
    return c.prototype.listener = function() {
        if (window.pageYOffset + n.viewport().height > this.$ad.offset().top + this.openedHeight) {
            var e = 6048e5;
            return t.local.set("gu.commercial.expandable.an-expandable", !0, {
                expires: Date.now() + e
            }), this.$button.toggleClass("button-spin"), this.$ad.css("height", this.openedHeight), this.isClosed=!1, !0
        }
    }, c.prototype.create = function() {
        var n = i.create(a(s, this.params));
        this.$ad = i(".ad-exp--expand", n).css("height", this.closedHeight), this.$button = i(".ad-exp__close-button", n), i(".ad-exp-collapse__slide", n).css("height", this.closedHeight), this.params.trackingPixel && this.$adSlot.before('<img src="' + this.params.trackingPixel + this.params.cacheBuster + '" class="creative__tracking-pixel" height="1px" width="1px"/>'), n.appendTo(this.$adSlot), t.local.get("gu.commercial.expandable.an-expandable") || o.on("window:scroll", this.listener.bind(this)), e.on(this.$button[0], "click", function() {
            this.$button.toggleClass("button-spin"), this.$ad.css("height", this.isClosed ? this.openedHeight : this.closedHeight), this.isClosed=!this.isClosed
        }.bind(this))
    }, c
}), define("text!common/views/commercial/creatives/fluid250.html", [], function() {
    return '<div class="ad-slot__label creative--fluid250__label fc-container--layout-front" data-test-id="ad-slot-label">\n    <div class="fc-container__inner">Advertisement</div>\n</div>\n<div class="creative--fluid250 l-side-margins hide-until-tablet" style="\n        background-color: {{backgroundColor}};\n        background-image: url({{backgroundImage}});\n        background-position: {{backgroundPosition}};\n        background-repeat: {{backgroundRepeat}};\n    ">\n    <a href="{{clickMacro}}{{link}}" target="_blank">\n        <div class="gs-container">\n            <div class="fluid250_layer fluid250_layer1" style="\n                background-image: url({{layerOneBGImage}});\n                background-position: {{layerOneBGPosition}};\n            "></div>\n            <div class="fluid250_layer fluid250_layer2" style="\n                background-image: url({{layerTwoBGImage}});\n                background-position: {{layerTwoBGPosition}};\n            "></div>\n            <div class="fluid250_layer fluid250_layer3" style="\n                background-image: url({{layerThreeBGImage}});\n                background-position: {{layerThreeBGPosition}};\n            "></div>\n        </div>\n    </a>\n</div>\n<div class="creative--fluid250 l-side-margins mobile-only" style="\n        background-color: {{backgroundColor}};\n        background-image: url({{backgroundImageM}});\n        background-position: {{backgroundPositionM}};\n        background-repeat: {{backgroundRepeatM}};\n    ">\n    <a href="{{link}}" target="_blank">\n        <div class="gs-container">\n            <div class="fluid250_layer fluid250_layer1" style="\n                background-image: url({{layerOneBGImageM}});\n                background-position: {{layerOneBGPositionM}};\n            "></div>\n            <div class="fluid250_layer fluid250_layer2" style="\n                background-image: url({{layerTwoBGImageM}});\n                background-position: {{layerTwoBGPositionM}};\n            "></div>\n            <div class="fluid250_layer fluid250_layer3" style="\n                background-image: url({{layerThreeBGImageM}});\n                background-position: {{layerThreeBGPositionM}};\n            "></div>\n        </div>\n    </a>\n</div>\n'
}), define("common/modules/commercial/creatives/fluid250", ["bean", "bonzo", "common/utils/$", "common/utils/mediator", "common/utils/storage", "common/utils/template", "text!common/views/commercial/creatives/fluid250.html"], function(e, n, i, o, t, a, s) {
    var c = function(e, n) {
        this.$adSlot = e, this.params = n
    };
    return c.prototype.create = function() {
        i.create(a(s, this.params)).appendTo(this.$adSlot), this.params.trackingPixel && this.$adSlot.before('<img src="' + this.params.trackingPixel + this.params.cacheBuster + '" class="creative__tracking-pixel" height="1px" width="1px"/>'), this.$adSlot.addClass("ad-slot__fluid250")
    }, c
}), define("text!common/views/commercial/creatives/fluid250GoogleAndroid.html", [], function() {
    return '<style type="text/css">\n.creative--fluid250 .fluid250_layer.fluid250_layer1 {\n    background-size: 90%;\n    width: 90%;\n    top: 10px;\n}\n@media (min-width: 480px) {\n    .creative--fluid250 .fluid250_layer.fluid250_layer1 {\n        background-size: 64%;\n        top: 0;\n    }\n}\n@media (min-width: 740px){\n    .creative--fluid250 .fluid250_layer.fluid250_layer1 {\n        background-size: 90%;\n        width: 100%;\n    }\n    .creative--fluid250 .fluid250_layer.fluid250_layer2 {\n        width: 25%;\n    }\n    .creative--fluid250 .fluid250_layer.fluid250_layer3 {\n        background-size: 795px;\n        width: 75%;\n    }\n    .creative--fluid250.hide-until-tablet {\n        background-image: url({{backgroundImageOffset}});\n    }\n}\n\n@media (min-width: 980px){\n    .creative--fluid250 .fluid250_layer.fluid250_layer1 {\n        background-size: contain;\n    }\n    .creative--fluid250 .fluid250_layer.fluid250_layer2 {\n        width: 20%;\n    }\n    .creative--fluid250 .fluid250_layer.fluid250_layer3 {\n    background-size: contain;\n    width: 80%;\n    }\n}\n\n@media (min-width: 1140px){\n    .creative--fluid250.hide-until-tablet {\n        background-image: url({{backgroundImage}});\n    }\n}\n</style>\n<div class="ad-slot__label creative--fluid250__label fc-container--layout-front" data-test-id="ad-slot-label">\n    <div class="fc-container__inner">Advertisement</div>\n</div>\n<div class="creative--fluid250 l-side-margins hide-until-tablet" style="\n        background-color: #ffffff;\n        background-position: 50% 0;\n        background-repeat: no-repeat;\n    ">\n    <a href="{{clickMacro}}{{link}}" target="_blank">\n        <div class="gs-container">\n            <div class="fluid250_layer fluid250_layer1" style="\n                background-image: url({{textImage}});\n                background-position: 50% 0;\n                top: 40px;\n                height: 52px;\n                margin-left: 0;\n            "></div>\n            <div class="fluid250_layer fluid250_layer2" style="\n                background-image: url({{logoImage}});\n                background-position: 100% 100%;\n                background-size: contain;\n                bottom: 0;\n                left: initial;\n                right: 0;\n                top: inherit;\n            "></div>\n            <div class="fluid250_layer fluid250_layer3" style="\n                background-image: url({{peopleImage}});\n                background-position: 100% 100%;\n            "></div>\n        </div>\n    </a>\n</div>\n<div class="creative--fluid250 l-side-margins mobile-only" style="\n        background-color: #ffffff;\n    ">\n    <a href="{{link}}" target="_blank">\n        <div class="gs-container">\n            <div class="fluid250_layer fluid250_layer1" style="\n                background-image: url({{textImage}});\n                background-position: 50% 0;\n                height: 48px;\n                margin-left: 5%;\n                z-index: 10;\n            "></div>\n            <div class="fluid250_layer fluid250_layer2" style="\n                background-image: url({{logoImage}});\n                background-position: 100% 100%;\n                background-repeat: no-repeat;\n                background-size: contain;\n                width: 30%;\n                height: 90px;\n                z-index: 10;\n                right: 0;\n                left: initial;\n                top: initial;\n                bottom: 0;\n            "></div>\n            <div class="fluid250_layer fluid250_layer3" style="\n                background-image: url({{peopleImage}});\n                background-position: 100% 100%;\n                background-repeat: no-repeat;\n                background-size: 115%;\n                width: 60%;\n                z-index: 10;\n            "></div>\n        </div>\n    </a>\n</div>\n'
}), define("common/modules/commercial/creatives/fluid250GoogleAndroid", ["bean", "bonzo", "common/utils/$", "common/utils/mediator", "common/utils/storage", "common/utils/template", "text!common/views/commercial/creatives/fluid250GoogleAndroid.html"], function(e, n, i, o, t, a, s) {
    var c = function(e, n) {
        this.$adSlot = e, this.params = n
    };
    return c.prototype.create = function() {
        i.create(a(s, this.params)).appendTo(this.$adSlot), this.params.trackingPixel && this.$adSlot.before('<img src="' + this.params.trackingPixel + this.params.cacheBuster + '" class="creative__tracking-pixel" height="1px" width="1px"/>'), this.$adSlot.addClass("ad-slot__fluid250")
    }, c
}), define("text!common/views/commercial/creatives/scrollable-mpu.html", [], function() {
    return '<a class="creative--scrollable-mpu"\n    href="{{clickMacro}}{{destination}}"\n    style="background-image: url(\'{{image}}\');"\n    target="_new"></a>\n{{trackingPixelImg}}\n'
}), define("common/modules/commercial/creatives/scrollable-mpu", ["common/utils/$", "common/utils/detect", "common/utils/mediator", "common/utils/template", "text!common/views/commercial/creatives/scrollable-mpu.html"], function(e, n, i, o, t) {
    var a = function(e, n) {
        this.$adSlot = e, this.params = n
    };
    return a.hasBgAttachmentFixed=!n.isIOS()&&!n.isAndroid(), a.prototype.updateBgPosition = function() {
        this.$scrollableMpu.css("background-position", this.$scrollableMpu.offset().left + "px 100%")
    }, a.prototype.create = function() {
        var n = {
            clickMacro: this.params.clickMacro,
            destination: this.params.destination,
            image: a.hasBgAttachmentFixed ? this.params.image: this.params.staticImage,
            trackingPixelImg: this.params.trackingPixel ? '<img src="' + this.params.trackingPixel + '" width="1" height="1" />': ""
        };
        this.$scrollableMpu = e.create(o(t, n)).appendTo(this.$adSlot), a.hasBgAttachmentFixed && (this.$scrollableMpu.css("background-attachment", "fixed"), this.updateBgPosition(), i.on("window:resize", this.updateBgPosition.bind(this)))
    }, a
}), define("text!common/views/commercial/creatives/ad-feature-mpu.html", [], function() {
    return '<div class="ad-slot--sponsored" data-link-name="creative | ad feature mpu">\n    <a class="ad-slot--sponsored__article" href="{{clickMacro}}{{articleUrl}}">\n        <img class="ad-slot--sponsored__image" src="{{articleImage}}" />\n        <span class="ad-slot--sponsored__text">{{articleHeaderText}}</span>\n    </a>\n    <p class="ad-slot--sponsored__label">Brought to you by:</p>\n    <a class="ad-slot--sponsored__link" href="{{clickMacro}}{{logoUrl}}">\n        <img class="ad-slot--sponsored__logo" src="{{logoImage}}" />\n    </a>\n    <a href="{{clickMacro}}{{infoUrl}}" class="ad-slot--sponsored__help">About this content</a>\n</div>\n'
}), define("text!common/views/commercial/creatives/ad-feature-mpu-large.html", [], function() {
    return '<div class="creative--ad-feature-mpu-large" data-link-name="creative | ad feature mpu large">\n    <a class="creative__article" href="{{clickMacro}}{{articleUrl}}" data-link-name="article">\n        <img class="creative__article__image" src="{{articleImage}}">\n        <h2 class="creative__article__title">{{articleHeaderText}}</h2>\n    </a>\n    <p class="creative__logo-text">Brought to you by:</p>\n    <a class="creative__logo" href="{{clickMacro}}{{logoUrl}}" data-link-name="logo">\n        <img src="{{logoImage}}" />\n    </a>\n    <a class="creative__logo-info" href="{{clickMacro}}{{infoUrl}}" data-link-name="info">About this content</a>\n</div>\n'
}), define("text!common/views/commercial/creatives/ad-single-manual.html", [], function() {
    return '<div class="commercial commercial--dfp commercial--dfp-single {{toneClass}}" role="complementary" data-link-name="creative | ad single manual | {{omnitureId}}">\n    <div class="facia-container__inner--commercial">\n        <div class="container__header">\n            <div class="container__header__inner">\n                <h3 class="commercial__title">\n                    <a href="{{clickMacro}}{{baseUrl}}"\n                       data-link-name="merchandising-title">\n                        <i class="i i-marque-54"></i><i class="i i-guardian-logo-commercial"></i>\n                        <span class="u-h">The Guardian</span> <span class="commercial__title__logo">{{title}}</span>\n                    </a>\n                </h3>\n            </div>\n        </div>\n        <div class="container__body">\n            <div class="lineitems">\n                <div class="lineitem lineitem--dfp-single">\n                    <div class="lineitem--dfp-single__offer">\n                        <a class="lineitem__link"\n                           href="{{clickMacro}}{{offerUrl}}"\n                           data-link-name="merchandising-{{offerTitle}}-link">\n                            <img class="lineitem__image"\n                                 src="{{offerImage}}" alt="{{offerTitle}}">\n                            <h4 class="lineitem__title">{{offerTitle}}</h4>\n                            <p class="lineitem__desc">{{offerText}}</p>\n                        </a>\n                        <a class="lineitem__cta button button--tertiary button--small"\n                           href="{{clickMacro}}{{offerUrl}}"\n                           data-link-name="merchandising-{{offerTitle}}-link">\n                            {{offerLinkText}}<i class="i i-arrow-black-right i-right"></i>\n                        </a>\n                    </div>\n                    <div class="lineitem--dfp-single__cta hide-on-mobile {{showCtaLink}}">\n                        <a class="commercial__cta button button--primary button--large"\n                           href="{{clickMacro}}{{seeMoreUrl}}"\n                           data-link-name="merchandising-viewall">\n                            {{viewAllText}}<i class="i i-arrow-white-right i-right"></i>\n                        </a>\n                    </div>\n                </div>\n            </div>\n        </div>\n    </div>\n</div>\n'
}), define("common/modules/commercial/creatives/template", ["common/utils/$", "common/utils/template", "text!common/views/commercial/creatives/ad-feature-mpu.html", "text!common/views/commercial/creatives/ad-feature-mpu-large.html", "text!common/views/commercial/creatives/ad-single-manual.html"], function(e, n, i, o, t) {
    var a = {
        "ad-feature-mpu": i,
        "ad-feature-mpu-large": o,
        "ad-single-manual": t
    }, s = function(e, n) {
        this.$adSlot = e, this.params = n
    };
    return s.prototype.create = function() {
        var i = n(a[this.params.creative], this.params);
        e.create(i).appendTo(this.$adSlot)
    }, s
}), define("bootstraps/commercial", ["bonzo", "qwery", "lodash/collections/forEach", "common/utils/$", "common/utils/config", "common/utils/mediator", "common/modules/commercial/ad-block-test", "common/modules/commercial/article-aside-adverts", "common/modules/commercial/article-body-adverts", "common/modules/commercial/badges", "common/modules/commercial/dfp", "common/modules/commercial/front-commercial-components", "common/modules/commercial/loader", "common/modules/commercial/slice-adverts", "common/modules/commercial/tags/container", "common/modules/userPrefs", "common/modules/commercial/creatives/branded-component", "common/modules/commercial/creatives/commercial-component", "common/modules/commercial/creatives/expandable", "common/modules/commercial/creatives/fluid250", "common/modules/commercial/creatives/fluid250GoogleAndroid", "common/modules/commercial/creatives/scrollable-mpu", "common/modules/commercial/creatives/template"], function(e, n, i, o, t, a, s, c, r, l, m, d, u, p, g, h) {
    var f = {
        commercialLoaderHelper: function() {
            i([["commercial-component", "merchandising"], ["commercial-component-high", "merchandising-high"]], function(i) {
                var o, t, a = new RegExp("^#" + i[0] + "=(.*)$").exec(window.location.hash), s = n('[data-name="' + i[1] + '"]').shift();
                a && s && (e(s).removeClass("ad-slot--dfp"), o = new u, t = {}, t[a[1]] = function() {
                    s.style.display = "block"
                }, o.postLoadEvents = t, o.init(a[1], s))
            })
        },
        tagContainer: function() {
            g.init()
        },
        articleAsideAdverts: function() {
            c.init()
        },
        articleBodyAdverts: function() {
            r.init()
        },
        sliceAdverts: function() {
            p.init()
        },
        frontCommercialComponents: function() {
            d.init()
        },
        badges: function() {
            l.init()
        },
        dfp: function() {
            m.init()
        },
        adBlockTest: function() {
            s()
        }
    }, v = function() {
        f.adBlockTest(), h.isOff("adverts") || t.page.shouldHideAdverts || t.page.isSSL && "admin" !== t.page.section || "#noads" === window.location.hash || (f.commercialLoaderHelper(), f.tagContainer(), f.articleAsideAdverts(), f.articleBodyAdverts(), f.sliceAdverts(), f.frontCommercialComponents(), f.badges(), f.dfp()), a.emit("page:commercial:ready")
    };
    return {
        init: v
    }
});
//# sourceMappingURL=commercial.js.map
