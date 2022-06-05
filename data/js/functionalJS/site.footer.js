$(function() {
    $(window).scroll(function() {
        if ($(window).width() > 768) {
            if ($(window).scrollTop() <= 101) {
                $(".topbar").removeClass("alt")
            } else {
                $(".topbar").addClass("alt")
            }
        }
    });
    $(".mobile-menu").click(function(a) {
        a.preventDefault();
        $("body > header .nav-left").toggleClass("nav-toggle")
    });
    $(".logoutAnchor").on("click", function(a) {
        a.preventDefault();
        $.get("/logout", function() {
            window.close()
        })
    });
    $(document).mouseup(function(a) {
        var b = $(".drop-user");
        if (b.has(a.target).length === 0) {
            $("a.user").removeClass("nav-highlight");
            b.hide()
        }
    });
    $(".nav-user-auth").click(function(a) {
        a.preventDefault();
        $(".drop-user").toggle()
    });
    $(".header-drop-inner a").click(function() {
        $(".header-drop").hide()
    });
    if (document.location.pathname.indexOf("/get/")!==-1 || document.location.pathname.indexOf("/analytics")!==-1) {
        $("body").append('<img src="//ib.adnxs.com/seg?add=1468098&t=2" width="1" height="1"/>')
    }
});
function ToggleNavGtc(a) {
    if (a) {
        $("#nav-gtc-li").hide();
        $("#nav-options-create-acct").hide();
        $("#nav-options-gtc").show();
        $(".drop-user").addClass("drop-auth")
    } else {
        $("#nav-gtc-li").show();
        $("#nav-options-create-acct").show();
        $("#nav-options-gtc").hide()
    }
    $(".topnav, .topnav-options").fadeTo(1000, 1)
}
var doCrazyegg = (typeof(doCrazyegg) === "undefined") ? (Math.floor(Math.random() * (175 - 1 + 1)) === 1): doCrazyegg;
if (doCrazyegg) {
    setTimeout(function() {
        var d = document.createElement("script");
        var c = document.getElementsByTagName("script")[0];
        d.src = document.location.protocol + "//dnn506yrbagrg.cloudfront.net/pages/scripts/0012/0247.js?" + Math.floor(new Date().getTime() / 3600000);
        d.async = true;
        d.type = "text/javascript";
        c.parentNode.insertBefore(d, c)
    }, 1)
};
