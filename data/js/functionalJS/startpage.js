window[jimdoData.jQuery](function(a) {
    function b() {
        var a = n.height(), c = o.height(), d = (p.width(), p.height()), e = d - a - c;
        return q.css("height", e), b
    }
    function c() {
        l =+ new Date, i.animate("next"), j = setTimeout(c, r)
    }
    function d() {
        var a =+ new Date;
        clearTimeout(j), k = r - (a - l), h.fadeIn(500)
    }
    function e() {
        j = setTimeout(c, k), k = null, h.fadeOut(500)
    }
    function f() {
        k = r
    }
    function g() {
        return "undefined" != typeof Hammer && a.isFunction(Hammer) && Hammer.VERSION && (Hammer(s[0]).on("swipeleft", function() {
            s.data("superslides").animate("next")
        }), Hammer(s[0]).on("swiperight", function() {
            s.data("superslides").animate("prev")
        }), m.off("loaded.Hammer")), g
    }
    var h, i, j, k, l, m = a(document), n = a("#header"), o = a("#panel").find(".panel-fixing"), p = a(window), q = a("#startpage-header"), r = 5e3, s = a("#slides-testimonials");
    s.superslides({
        inherit_width_from: ".tp-cm-jimmscroll-testimonial",
        inherit_height_from: ".tp-cm-jimmscroll-testimonial"
    }).on({
        mouseenter: d,
        mouseleave: e
    }), h = s.find(".slides-pagination").on("click", "a", f), i = s.data("superslides"), j = setTimeout(c, r), a(".tp-cm-jimscroll-videolink").magnificPopup({
        disableOn: 0,
        type: "iframe",
        mainClass: "mfp-fade",
        removalDelay: 160,
        preloader: !1,
        fixedContentPos: !0
    }), p.on("resize", b()), m.on("loaded.Hammer", g()), a(".tp-cm-header-more").on("click", function() {
        a("html, body").animate({
            scrollTop: a("#tp-cm-jimscroll").offset().top
        }, 500)
    })
});
