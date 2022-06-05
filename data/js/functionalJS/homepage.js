$(function() {
    function b(c) {
        if (typeof _gaq != "undefined") {
            _gaq.push(["_trackPageview", c])
        }
    }
    $(".hp-pro").click(function() {
        b("/tracker/home/pro-button")
    });
    $(".hp-start").click(function() {
        b("/tracker/home/pro-bottom")
    });
    $(".hp-basic").click(function() {
        b("/tracker/home/gtc-bottom")
    });
    $(".pro-trial-start").on("click", function() {
        b("/tracker/modal/pro-button")
    });
    if ($(window).width() > 960) {
        $(window).scroll(function() {
            a()
        })
    }
    function a() {
        scrollPos = $(this).scrollTop();
        $(".page-title h1").css({
            top: - (scrollPos / 1.5) + "px",
            opacity: 1 - (scrollPos / 150)
        });
        $(".page-title .header-subhead").css({
            top: (scrollPos / 2) + "px",
            opacity: 1 - (scrollPos / 250)
        });
        $(".page-title .signin-buttons, .started-button").css({
            top: (scrollPos / 2.5) + "px",
            opacity: 1 - (scrollPos / 350)
        });
        $(".logos").css({
            bottom: 80 + (scrollPos / 2.5) + "px",
            opacity: 0.5 - (scrollPos / 750)
        });
        $(".godown").css({
            bottom: (scrollPos / 2.5) + "px",
            opacity: 1 - (scrollPos / 500)
        })
    }
    $(".godown").hover(function() {
        $(this).addClass("bounce")
    }, function() {
        $(this).removeClass("bounce")
    });
    $(".nav-tabs a").click(function(c) {
        c.preventDefault();
        $(this).tab("show")
    });
    $("header a[href*=#]:not([href=#])").click(function() {
        if (location.pathname.replace(/^\//, "") == this.pathname.replace(/^\//, "") && location.hostname == this.hostname) {
            var c = $(this.hash);
            c = c.length ? c : $("[name=" + this.hash.slice(1) + "]");
            if (c.length) {
                $("html,body").animate({
                    scrollTop: c.offset().top
                }, 1000);
                return false
            }
        }
    })
});
