function LIConnect(e) {
    "undefined" != typeof e && (this.config = e, "" == this.config.login_source && (this.config.login_source = "li-connect"), this.config.connectRoute = void 0 === e.connectRoute ? "connect" : e.connectRoute, this.config.connectUrl = "https://" + window.location.host + "/liconnect/" + this.config.connectRoute + "?login_source=" + this.config.login_source + "&submit_url=" + decodeURIComponent(document.location.protocol + "//" + document.location.host + document.location.pathname), this.bindEvents(), slideshare_object.registerMessageReceiver($.proxy(this.messageReceiver, this)))
}
function CarouselModalPlayer(e) {
    this.init(), this.config = e
}
function ModalShare(e) {
    this.config = e, e && this.init()
}
function ListPageModalShare(e) {
    this.config = e, e && this.init()
}
function MobileModalShare(e) {
    this.config = e, this.init()
}
function MobilePromo(e) {
    "object" == typeof e && (this.config = e, this.init())
}
LIConnect.prototype.openAuthDialog = function(e) {
    void 0 === e && (e = this.config.connectUrl), window.liChildWindow = window.open("https://www.linkedin.com/uas/oauth2/authorization?response_type=code&client_id=y4wa9oe4c6nu&scope=&state=" + this.config.state + "&redirect_uri=" + encodeURIComponent(e), "_blank", "height=400,width=600,status=yes,toolbar=no,menubar=no,location=no,scrollbars=yes"), (isBrowserMSIE() || navigator.userAgent.match(/iPhone|iPad|iPod|Android|BlackBerry|IEMobile|webOS|Mobile|Windows Phone/i)) && (timer = setInterval(this.checkChild, 500))
}, LIConnect.prototype.bindEvents = function() {
    var e = this;
    $(this.config.connectButtonClass).on("click", function() {
        window._gaq = _gaq || [], _gaq.push(["_trackEvent", "Login", "linkedin", e.config.login_source + "_LiconnectClicked"]), e.openAuthDialog()
    })
}, LIConnect.prototype.checkChild = function() {
    if (0 != window.liChildWindow.closed) {
        clearInterval(timer);
        var e = window.location != window.parent.location;
        e ? $.postMessage("modalLoginSubmit", $("#target_url").val(), parent) : (window._gaq = _gaq || [], _gaq.push(["_trackEvent", "Login", "linkedin", "login_successful"]), window.location.reload())
    }
}, LIConnect.prototype.uploadToLiCallback = function(e) {
    var t = $("#li_upload_" + e);
    t.removeClass("j-linkedin-connect"), t.addClass("added"), $("#li_upload_" + e + " .flat-button-text").text("Added to profile"), slideshare_object.ga("Treasury", "linkedin", "connect_and_upload_success")
}, LIConnect.prototype.messageReceiver = function(e) {
    switch (e) {
    case"liconnectSubmit":
        slideshare_object.ga("Login", "linkedin", "login_successful"), window.location.reload();
        break;
    case"liconnectSubmitVisualResume":
        window.location = "/professional-journey/show";
        break;
    case"liconnectModalSubmit":
        slideshare_object.ga("Login", "linkedin", "modal_login_successful"), $.postMessage("modalLoginSubmit", $("#target_url").val(), parent);
        break;
    case"liUploadSubmit":
        this.uploadToLiCallback(this.config.slideshow_id), bootbox.alert("Your SlideShare was successfully added to your LinkedIn profile.");
        break;
    case"liNewUserUpload":
        $("#j-linkedinConnectedMessage").modal("show"), this.uploadToLiCallback(this.config.slideshow_id)
    }
}, CarouselModalPlayer.prototype.init = function() {
    this.$container = $("#carousel-modal-player"), this.$modalHeader = this.$container.find(".carousel-modal-header .carousel-modal-title"), this.$modalBody = this.$container.find(".carousel-modal-body"), this.$modalFooter = this.$container.find(".carousel-modal-footer a"), this.populateSlideshowsLists(), this.bindEvents()
}, CarouselModalPlayer.prototype.populateSlideshowsLists = function() {
    this.slideshows = {}, this.slideshows_ordered = [];
    var e = this;
    $(".iso_slideshow").each(function(t) {
        var i = $(this).find(".iso_slideshow_link"), o = $(this).find(".j-author_text").text(), a = $(this).find(".j-author_text").attr("href"), s = i.attr("href"), n = i.data("ppt-id"), r = i.data("title");
        e.slideshows[n] = {
            index: t,
            title: r,
            url: s,
            authorName: o,
            authorUrl: a,
            container: $(this)
        }, e.slideshows_ordered.push(n)
    }), this.currentSlideshowIndex = 0
}, CarouselModalPlayer.prototype.getEmbedCode = function(e) {
    return '<iframe src="/slideshow/embed_code/' + e + '?rel=0" width="708" height="100%" frameborder="0" marginwidth="0" marginheight="0" scrolling="no"style="border:1px solid #CCC;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe>'
}, CarouselModalPlayer.prototype.updatePopup = function(e) {
    this.$modalBody.html(this.getEmbedCode(e)), this.$modalHeader.attr("href", this.slideshows[e].url), this.$modalHeader.html(this.slideshows[e].title), this.$modalFooter.attr("href", this.slideshows[e].authorUrl), this.$modalFooter.html(this.slideshows[e].authorName)
}, CarouselModalPlayer.prototype.show = function(e) {
    $("body").addClass("carousel-modal-player-visible"), this.updatePopup(this.slideshows_ordered[e]), this.$container.fadeIn(), this.currentSlideshowIndex = e, this.slideshows[this.slideshows_ordered[e]].container.trigger("modal_shown")
}, CarouselModalPlayer.prototype.showBySsid = function(e) {
    this.show(this.slideshows[e].index)
}, CarouselModalPlayer.prototype.hide = function() {
    $("body").removeClass("carousel-modal-player-visible"), slideshare_object && slideshare_object.user && slideshare_object.user.loggedin && slideshare_object.favorites && slideshare_object.favorites.user_favorites && (slideshare_object.favorites_status_update && slideshare_object.favorites_status_update(), slideshare_object.favoritesStatusUpdate && slideshare_object.favoritesStatusUpdate()), this.$container.fadeOut(), this.$modalBody.empty()
}, CarouselModalPlayer.prototype.next = function() {
    this.show((this.currentSlideshowIndex + 1)%this.slideshows_ordered.length)
}, CarouselModalPlayer.prototype.prev = function() {
    this.show((this.currentSlideshowIndex + this.slideshows_ordered.length - 1)%this.slideshows_ordered.length)
}, CarouselModalPlayer.prototype.bindSlideshowLinks = function() {
    var e = this;
    $("a.iso_slideshow_link").on("click", function(t) {
        /((iPod|iPhone|iPad).*AppleWebKit|Android|BlackBerry|Symbian|Mobile|Windows Phone|Opera Mobi)/i.test(navigator.userAgent) || (t.preventDefault(), t.stopPropagation(), e.showBySsid($(this).data("ppt-id")))
    })
}, CarouselModalPlayer.prototype.bindEvents = function() {
    var e = this;
    this.bindSlideshowLinks(), this.$container.on("click", function() {
        return e.hide(), !1
    }), this.$container.on("click", ".close", function() {
        return e.hide(), !1
    }), this.$modalHeader.on("click", function(e) {
        e.stopPropagation()
    }), this.$modalFooter.on("click", function(e) {
        e.stopPropagation()
    }), this.$container.on("click", ".carousel-modal-prev", function() {
        return e.prev(), !1
    }), this.$container.on("click", ".carousel-modal-next", function() {
        return e.next(), !1
    })
}, ModalShare.prototype.init = function() {
    this.$container = $(this.config.el), this.$shareBtn = $(this.config.btnShare), this.$shareEmailMsg = this.$container.find(".j-share-email-msg"), this.$addMsgButton = this.$container.find(".j-add-msg"), this.$emailSentButton = this.$container.find(".j-email-sent"), this.$shareEmailSend = this.$container.find("#share-email-send"), this.$shareEmailForm = this.$container.find(".j-share-email-form"), this.$linkURLInput = this.$container.find(".j-share-link-url"), this.$close = this.$container.find(".j-modal-close"), this.$contentWrapper = this.$container.find(".modal-content-wrapper"), this.gaTrackCategory = this.$container.data("ga-track-category"), this.gaTrackAction = this.$container.data("ga-track-action"), this.createShareHash(), slideshare_object && slideshare_object.user && slideshare_object.user.loggedin===!0 && null !== slideshare_object.user.login && this.$container.find(".j-share-email-name").val(slideshare_object.user.login), this.bindModalToShareButtons(), this.bindModalEvents(), this.bindCustomModalEvents(), this.adjustShareModalTop()
}, ModalShare.prototype.get_viral_share_url = function(e) {
    $elem = $("li." + e + " a");
    var t = this.data.ssLink;
    return $elem.length > 0 && (t = $.param($elem.data()), !t && (t = this.data.ssLink)), this.SHARE_HASH[e] + t
}, ModalShare.prototype.get_viral_share_urls = function() {
    return {
        facebook: this.get_viral_share_url("facebook"),
        linkedin: this.get_viral_share_url("linkedin"),
        twitter: this.get_viral_share_url("twitter"),
        wordpress: this.get_viral_share_url("wordpress"),
        pinterest: this.get_viral_share_url("pinterest")
    }
}, ModalShare.prototype.sharePopup = function(e) {
    var t = 626, i = 436, o = this.data.social[e];
    if (void 0 !== o) {
        if ("facebook" === e) {
            if (window.FB)
                return this.trackShareClick(e), void this.postToFacebook()
        } else if ("linkedin" === e && window.IN && window.IN.UI)
            return IN.UI.Share().params({
                url: this.data.ssLink
            }).place(), this.trackShareClick(e), void this.saveShare(e);
        window.open(o, "shareDialog", "width=" + t + ",height=" + i), this.trackShareClick(e), this.saveShare(e)
    }
}, ModalShare.prototype.show = function() {
    this.reset(), this.$container.show(), this.shareGATracking("modalOpen")
}, ModalShare.prototype.close = function() {}, ModalShare.prototype.reset = function() {
    this.$container.find(".j-embed-use-ssl-cbox").prop("checked", !1), $(".j-modal-popup").scrollTop(0), this.$close.css("top", "8px"), this.clearEmailFields(), this.$shareEmailForm.show(), this.$emailSentButton.hide(), this.collapseAllSections()
}, ModalShare.prototype.updateEmbed = function() {
    var e = this.$container.find(".j-share-embed-link"), t = e.val(), i = this.$container.find(".j-embed-size-picker").val(), o = this.$container.find(".j-embed-size-picker :selected").data("width") || this.config.embed_sizes.presets[i].size.width, a = this.$container.find(".j-embed-size-picker :selected").data("height") || this.config.embed_sizes.presets[i].size.height, s = t.match(/src="([^"]+)"/)[1].split("?")[0], n = this.$container.find(".j-embed-start-picker").prop("selectedIndex") + 1 || 1;
    s += n > 1 ? "?startSlide=" + n : "", t = t.replace(/src="[^"]+"/, 'src="' + s + '"'), t = t.replace("http:", ""), t = t.replace("https:", ""), t = t.replace(/height="\d+"/, 'height="' + a + '"'), t = t.replace(/width="\d+"/, 'width="' + o + '"'), e.val(t), this.shareGATracking("share_embed", "updateembed")
}, ModalShare.prototype.clearEmailErrorAlerts = function() {
    $(".error").removeClass("error")
}, ModalShare.prototype.clearEmailFields = function() {
    this.$container.find(".j-email-clear").val(""), this.$shareEmailSend.prop("disabled", !1), this.clearEmailErrorAlerts()
}, ModalShare.prototype.sendShareEmail = function() {
    this.shareGATracking("share_email", "send");
    var e = this, t = this.$container.find(".j-share-email-to").val(), i = this.$container.find(".j-share-email-name").val(), o = this.$shareEmailMsg.val(), a=!1;
    e.$shareEmailSend.val(""), e.$shareEmailSend.addClass("sending"), e.$shareEmailSend.prop("disabled", !0), e.clearEmailErrorAlerts(), a || (a=!0, $.post("/list/email_share", {
        data: {
            id: this.data.ssId,
            name: i,
            recipients: t,
            msg: o
        },
        dataType: "json"
    }).done(function(t) {
        if (0 === t.respCode)
            e.clearEmailFields(), e.$shareEmailForm.slideUp(400), e.collapseAllSections(), e.$container.find("#email-sent").slideDown(400), setTimeout(function() {
                e.$container.find("#email-sent").slideUp(400), e.$shareEmailForm.slideDown(400)
            }, 2e3), e.$contentWrapper.removeClass("email-expanded");
        else {
            var i = e.$container.find(".j-email-flash");
            switch (i.text(t.resp).show(), t.respCode) {
            case 1:
                e.$container.find(".j-share-email-to").addClass("error");
                break;
            case 2:
                e.$container.find(".j-share-email-name").addClass("error")
            }
            setTimeout(function() {
                i.fadeOut(400)
            }, 5e3)
        }
        a=!1, e.$shareEmailSend.val("Send"), e.$shareEmailSend.removeClass("sending"), e.$shareEmailSend.prop("disabled", !1)
    }).fail(function() {
        a=!1, e.$shareEmailSend.val("Send"), e.$shareEmailSend.removeClass("sending"), e.$shareEmailSend.prop("disabled", !1)
    }))
}, ModalShare.prototype.shareGATracking = function(e, t) {
    var i = window._gaq || [];
    e = this.gaTrackAction ? this.gaTrackAction + "_" + e : e, i.push(["_trackEvent", this.gaTrackCategory, e, t])
}, ModalShare.prototype.renderSlideStartOptions = function() {
    for (var e = "", t = 0; t < this.data.totalSlides; t++) {
        var i = "slide " + (t + 1);
        e += "<option ", 0 === t && (e += ' selected="selected"'), e += ">" + i + "</option>"
    }
    var o = this.$container.find(".j-embed-start-picker, .j-share-start-picker");
    o.empty(), o.append(e), this.updateStartSlide()
}, ModalShare.prototype.renderDefaultEmbedOptions = function() {
    var e = {
        0: {
            small: [344, 292],
            "default": [429, 357],
            large: [514, 422],
            xlarge: [599, 487]
        },
        1: {
            small: [386, 410],
            "default": [481, 512],
            large: [576, 614],
            xlarge: [672, 716]
        }
    }, t = e[this.data.ssType%2], i = "";
    for (var o in t) {
        var a = t[o][0] + " x " + t[o][1];
        i += '<option value="' + o + '" data-width="' + t[o][0] + '" data-height="' + t[o][1] + '"', "default" === o && (i += " selected"), i += ">" + a + "</option>"
    }
    var s = this.$container.find(".j-embed-size-picker");
    s.empty(), s.append(i), this.$container.find(".j-embed-start-picker").prop("selecteIndex", 1), this.$container.find(".j-share-embed-wp").val(this.data.social.wordpress), this.$container.find(".j-share-embed-link").text(this.data.embedCode)
}, ModalShare.prototype.toggleAddMessageHeight = function(e) {
    this.$shareEmailMsg.toggleClass("msg-expanded", e)
}, ModalShare.prototype.collapseAllSections = function() {
    $(".j-share-expand").slideUp(), $("#share-embed-link").scrollTop(0), $(".j-triggered").removeClass("j-triggered"), this.toggleAddMessageHeight(!1)
}, ModalShare.prototype.triggerExpand = function(e) {
    this.collapseAllSections(), $(e).addClass("j-triggered"), $(e).siblings(".j-share-expand").slideDown(400, this.adjustShareModalTop), this.$contentWrapper.removeClass("email-expanded embed-expanded"), $(e).hasClass("j-share-email-to") ? this.$contentWrapper.addClass("email-expanded") : $(e).hasClass("j-share-embed-link") && this.$contentWrapper.addClass("embed-expanded")
}, ModalShare.prototype.getNewSlideURL = function(e, t) {
    var i = document.createElement("a");
    i.href = e;
    var o = i.pathname.split("/");
    o[0] || o.shift();
    for (var a = o.length, s = o[a - 1]; !s && a > 0;)
        o.pop(), a--, s = o[a - 1];
    return parseInt(s, 10) && a > 2 && o.pop(), t > 1 && o.push(t), i.pathname = o.join("/"), i.href
}, ModalShare.prototype.getSlideImageURL = function(e) {
    return $('#svPlayerId .slide[data-index="' + e + '"] .slide_image').data("full")
}, ModalShare.prototype.bindModalEvents = function() {
    var e = this;
    this.$close.bind("click", function() {
        e.close()
    }), this.$container.find(".j-share-start-picker").change(function() {
        e.updateStartSlide(), e.shareGATracking("share_social", $(this).data("ga"))
    }), this.$shareEmailForm.submit(function(t) {
        t.preventDefault(), t.stopPropagation(), e.sendShareEmail()
    }), this.$shareEmailMsg.click(function() {
        e.toggleAddMessageHeight(!0)
    }), this.$container.find(".j-share-email-to, .j-share-email-name").click(function() {
        e.toggleAddMessageHeight(!1)
    }), this.$container.find(".j-share-social-list li").click(function() {
        e.sharePopup($(this).data("network"))
    }), this.$container.find(".j-update-embed").change(function() {
        e.updateEmbed()
    }), this.$container.find(".j-share-embed-link, .j-share-embed-wp, .j-last-screen-ssurl").bind("click", function() {
        this.select()
    }), this.$container.find(".j-share-link-url").click(function() {
        this.select(), e.shareGATracking("share_social", $(this).data("ga"))
    }), this.$container.find(".j-share-email-form input, .j-share-email-form textarea, .j-add-msg, .j-send-another-email").click(function() {
        "send" !== $(this).data("ga") && e.shareGATracking("share_email", $(this).data("ga"))
    }), this.$container.find(".j-embed-use-ssl-cbox, .j-embed-related-cbox, .j-embed-size-picker, .j-embed-start-picker").click(function() {
        e.shareGATracking("share_embed", $(this).data("ga"))
    }), this.$container.find(".j-share-expand-trigger").click(function() {
        $(this).siblings(".j-share-expand:visible").length > 0 || e.triggerExpand(this)
    }), $(window).resize(function() {
        e.adjustShareModalTop()
    }), $(document).keydown(function(t) {
        27 === t.which && (t.preventDefault(), e.close())
    }), $(".j-modal-popup").on("scroll", function() {
        e.$close.css("top", 8 + $(this).scrollTop() + "px")
    })
}, ModalShare.prototype.bindCustomModalEvents = function() {}, ModalShare.prototype.createShareHash = function() {
    this.SHARE_HASH = {
        facebook: "https://www.facebook.com/sharer/sharer.php?u=",
        linkedin: "https://www.linkedin.com/cws/share?",
        twitter: "https://twitter.com/share?",
        wordpress: "",
        pinterest: "https://pinterest.com/pin/create/button/?"
    }
}, ModalShare.prototype.getSSId = function(e) {
    return e.length ? e.data("slideshowid") : slideshare_object.modal_player.get_ss_id()
}, ModalShare.prototype.bindModalToShareButtons = function() {}, ModalShare.prototype.trackShareClick = function() {}, ModalShare.prototype.saveShare = function(e) {
    $.post("/share/viralshare", {
        data: {
            id: this.data.ssId,
            network: e
        }
    })
}, ModalShare.prototype.loadFBWidget = function() {
    !function(e, t, i) {
        var o, a = e.getElementsByTagName(t)[0];
        e.getElementById(i) || (o = e.createElement(t), o.id = i, o.src = "//connect.facebook.net/en_US/all.js", a.parentNode.insertBefore(o, a))
    }(document, "script", "facebook-jssdk");
    var e = this;
    !function() {
        function t() {
            e.shareGATracking("share_social", "fb_success_callback")
        }
        var i = {
            appId: 2490221586,
            status: !0,
            cookie: !0,
            xfbml: !0,
            oauth: !0,
            channelUrl: 0 / 0
        };
        "object" == typeof FB ? window.setTimeout(function() {
            FB.XFBML.parse(), FB.Event.subscribe("edge.create", t)
        }, 50) : window.fbAsyncInit = function() {
            FB.init(i), FB.Event.subscribe("edge.create", t)
        }
    }()
}, ModalShare.prototype.showLikeUsModal = function() {}, ModalShare.prototype.postToFacebook = function() {
    var e = this;
    if (window.FB) {
        var t = function(t) {
            t && t.post_id && e.saveShare("facebook")
        };
        FB.ui({
            method: "feed",
            display: "popup",
            link: this.data.ssLink
        }, t)
    }
}, ModalShare.prototype.adjustShareModalTop = function() {}, ModalShare.prototype.updateStartSlide = function() {}, ListPageModalShare.prototype = new ModalShare, ListPageModalShare.prototype.bindCustomModalEvents = function() {
    this.$container.find(".j-modal-content").click(function(e) {
        e.stopPropagation()
    })
}, ListPageModalShare.prototype.bindModalToShareButtons = function() {
    var e = this, t=!1, i = this.$container.find(".j-modal-popup");
    $(this.config.parentEl).on("click", this.config.btnShare, function() {
        var o = $(this).closest(".iso_slideshow").length > 0 ? $(this).closest(".iso_slideshow"): $("#modal_player");
        i.hide(), e.$container.fadeIn(200), data = {
            embedCode: o.data("embedcode"),
            totalSlides: o.data("totalslides"),
            ssLink: o.data("sslink"),
            title: o.data("title"),
            ssType: o.data("sstype"),
            ssId: o.data("slideshowid"),
            social: {
                linkedin: o.data("linkedinshareurl"),
                facebook: o.data("facebookshareurl"),
                twitter: o.data("twittershareurl"),
                pinterest: o.data("pinterestshareurl"),
                wordpress: o.data("wordpressshareurl"),
                email: "mailto:?subject=Check out this Slideshare " + o.data("sstypeasstring") + "&body=" + o.data("sslink")
            }
        }, e.data = data, i.show(), e.show(), e.adjustShareModalTop(), e.$linkURLInput.val(e.data.ssLink), e.renderDefaultEmbedOptions(), e.renderSlideStartOptions(), t || ($(".iso_slideshow_link").attr("title", ""), t=!0)
    })
}, ListPageModalShare.prototype.trackShareClick = function(e) {
    this.shareGATracking("share_social", e)
}, ListPageModalShare.prototype.close = function() {
    this.reset(), this.$container.hide(), this.shareGATracking("modalClose")
}, ListPageModalShare.prototype.adjustShareModalTop = function() {
    var e = $("#homepage-modal-share, #new-list-modal-share"), t = e.find(".j-modal-popup");
    e.length > 0 && t.length > 0 && t.animate({
        top: (e.height() - t.height()) / 2 + "px"
    }, 400)
}, MobileModalShare.prototype = {}, MobileModalShare.prototype.IOS4REGEX = /["iPhone"|"iPad"].+i?OS [4|3].+Safari/, MobileModalShare.prototype.isIpadAndChrome = function() {
    return /ipad.*crios|crios.*ipad|ipad.*chrome|chrome.*ipad/i.test(navigator.userAgent)
}, MobileModalShare.prototype.init = function() {
    this.$modalShare = $(this.config.el), this.$overlay = this.$modalShare.find(".share-overlay"), this.$shareList = this.$modalShare.find(".mobile-share-list"), this.$linkShare = this.$modalShare.find(".link-share"), this.gaCategory = this.$modalShare.data("ga"), this.bindModalEvents(), this.bindModalToShareBtns(), this.populateShareLinks(), this.applyAppropriateOrientationCSS()
}, MobileModalShare.prototype.getDeviceOrientation = function() {
    var e;
    if (window.orientation)
        e = 0 === window.orientation ? "portrait" : "landscape";
    else {
        var t = window.innerWidth || screen.width, i = window.innerHeight || screen.height;
        e = t > i ? "landscape" : "portrait"
    }
    return e
}, MobileModalShare.prototype.open = function() {
    this.gaTracking("open"), this.$modalShare.fadeIn(), $("body").addClass("no_scroll")
}, MobileModalShare.prototype.close = function() {
    this.gaTracking("close"), this.$modalShare.fadeOut(), $("body").removeClass("no_scroll")
}, MobileModalShare.prototype.populateShareLinks = function() {
    this.isIpadAndChrome() || $.each(this.$shareList.find("li > a"), function() {
        $(this).attr("href", $(this).data("url"))
    })
}, MobileModalShare.prototype.gaTracking = function(e, t) {
    if (t) {
        var i = window._gaq || [];
        e = "mobile_" + e, i.push(["_trackEvent", this.gaCategory, e, t])
    }
}, MobileModalShare.prototype.repositionShareModal = function() {
    this.$overlay.css("position", "static")
}, MobileModalShare.prototype.applyAppropriateOrientationCSS = function() {
    this.orientation = this.getDeviceOrientation(), "portrait" === this.orientation ? this.$modalShare.removeClass("landscape") : this.$modalShare.addClass("landscape")
}, MobileModalShare.prototype.bindModalEvents = function() {
    var e = this;
    this.$shareList.on("click", "li", function() {
        e.gaTracking("social", $(this).data("ga"))
    }), "undefined" == typeof $mainNav && this.isIpadAndChrome() && this.$shareList.on("click", "a", function() {
        window.open($(this).data("url"), "_blank")
    }), this.$modalShare.find(".btn, .share-overlay, .j-close").click(function() {
        e.close()
    }), $(window).on("resize orientationchange", function() {
        e.applyAppropriateOrientationCSS()
    })
}, MobileModalShare.prototype.bindModalToShareBtns = function() {
    var e = this;
    $(this.config.parentEl).on("click", this.config.shareBtn, function() {
        if (e.IOS4REGEX.test(navigator.userAgent) && e.repositionShareModal(), window.mobilePlayer)
            return void e.open();
        var t = $(this).closest(".iso_slideshow").data("slideshowid"), i = $(this).closest(".iso_slideshow");
        if (t) {
            data = {
                ssLink: i.data("sslink"),
                social: {
                    linkedin: i.data("linkedinshareurl"),
                    facebook: i.data("facebookshareurl"),
                    twitter: i.data("twittershareurl"),
                    sms: i.data("smsshareurl"),
                    whatsapp: i.data("whatsappshareurl"),
                    googleplus: i.data("googleshareurl"),
                    email: i.data("emailshareurl"),
                    pinterest: i.data("pinterestshareurl")
                }
            };
            var o = data.social.linkedin, a = data.social.facebook, s = data.social.twitter, n = data.social.email, r = data.social.sms, l = data.social.whatsapp, d = data.social.googleplus, h = data.social.pinterest;
            e.$linkShare.attr("href", data.ssLink), e.$linkShare.html(data.ssLink), e.$shareList.find(".li-share").attr("href", o), e.$shareList.find(".fb-share").attr("href", a), e.$shareList.find(".tw-share").attr("href", s), e.$shareList.find(".whatsApp-share").attr("href", l), e.$shareList.find("#sharebyemail").attr("href", n), e.$shareList.find("#sharebysms").attr("href", r), e.$shareList.find(".googleplus-share").attr("href", d), e.$shareList.find(".pinterest-share").attr("href", h), e.open()
        }
    })
}, MobilePromo.prototype.init = function() {
    this.promoName = this.config.promoName, this.previouslyClosed() || (this.currentPage = this.config.currentPage, this.$promo = $(this.config.promoSelector), this.$download = this.$promo.find(this.config.downloadSelector), this.config.lazyloadClass && (this.lazyloadImages = this.config.lazyloadClass), this.config.closedSelector && (this.$close = this.$promo.find(this.config.closedSelector)), this.config.moveAfterSelector && (this.$moveAfter = this.config.moveAfterSelector), this.config.variableContent && (this.variableContent = this.config.variableContent), this.config.navbarSelector && (this.$navbar = $(this.config.navbarSelector)), this.config.contentSelector && (this.$content = $(this.config.contentSelector)), this.config.launchOnInit && this.launchPromo())
}, MobilePromo.prototype.launchPromo = function(e, t) {
    if (this.appropriateContent = e, this.afterCloseAction = t, this.previouslyClosed())
        return void(this.afterCloseAction && (window.location.href = this.afterCloseAction));
    this.$moveAfter && this.relocatePromo(), e && this.variableContent && this.initContent(), this.initActions(), this.$promo.show(), this.lazyloadImages && mobile_util.lazyloadImages(this.lazyloadImages, 999);
    var i = this.$promo.outerHeight();
    this.$content && this.$content.css("margin-top", parseInt(this.$content.css("margin-top"), 10) + i), this.$navbar && this.$navbar.css("top", parseInt(this.$navbar.css("top"), 10) + i)
}, MobilePromo.prototype.relocatePromo = function() {
    this.$moveAfter.after(this.$promo)
}, MobilePromo.prototype.previouslyClosed = function() {
    return mobile_util.hasCookie(this.promoName + "_closed")
}, MobilePromo.prototype.initContent = function() {
    var e = this.variableContent[this.appropriateContent];
    if ("object" == typeof e)
        for (var t = 0; t < e.length; t++) {
            var i = $(e[t].selector);
            e[t].attr ? i.attr(e[t].attr, i.data(e[t].dataAttr)) : i.text(i.data(e[t].dataAttr))
        }
}, MobilePromo.prototype.initActions = function() {
    var e = this;
    mobile_util.trackGAEvent(this.currentPage, this.promoName, "loaded"), this.$close && this.$close.on("click", function() {
        e.closePromo(), mobile_util.trackGAEvent(e.currentPage, e.promoName, "closed"), e.afterCloseAction && (window.location.href = e.afterCloseAction)
    }), this.$download.on("click", function() {
        e.closePromo(), mobile_util.trackGAEvent(e.currentPage, e.promoName, "clicked")
    })
}, MobilePromo.prototype.closePromo = function() {
    this.$promo.hide();
    var e = this.$promo.outerHeight();
    this.$content && this.$content.css("margin-top", parseInt(this.$content.css("margin-top"), 10) - e), this.$navbar && this.$navbar.css("top", parseInt(this.$navbar.css("top"), 10) - e), mobile_util.setCookie(this.promoName + "_closed")
};
