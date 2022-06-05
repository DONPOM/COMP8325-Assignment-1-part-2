(function(a) {
    var f = {
        interval: 250,
        scrollTime: 500,
        yPos: 400
    };
    a.modules.scrolltotop = function(g) {
        var b = a(window), c = a(".cc-FloatingButtonBarContainer-button-scroll"), d=!1, e = a.extend({}, f, g);
        b.on("scroll", a._.throttle(function() {
            if (!d) {
                var a = b.scrollTop() > e.yPos;
                c.toggleClass("cc-FloatingButtonBarContainer-button-scroll-show", a)
            }
        }, e.interval));
        c.on("click", function(b) {
            b && b.preventDefault();
            c.toggleClass("cc-FloatingButtonBarContainer-button-scroll-show", !1);
            d=!0;
            a("html,body").stop().animate({
                scrollTop: 0
            },
            e.scrollTime, "easeOutCirc", function() {
                d=!1
            })
        })
    }
})(jimdoGen002);
