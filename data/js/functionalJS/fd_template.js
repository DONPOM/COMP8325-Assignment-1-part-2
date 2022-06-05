var fnTmplShowUpdateBasket = function(sBasket) {
    $(".basket-popup").html(sBasket);
    var nTot = $(".basket-popup #basket_total_in").val();
    var nPounds = $(".basket-popup #basket_pounds_in").val();
    var nPence = $(".basket-popup #basket_pence_in").val();
    $(document).off("click", ".fd-fav-basket .basket-btn");
    $(document).off("click", ".pinned-fav-basket .basket-btn");
    $(document).off("click", ".fd-header-xs .basket-btn");
    if (nTot == 0) {
        $("#basket_pounds").text(nPounds);
        $("#basket_pence").text(nPence);
        $("#basket_value").show();
        $("._basket-total-wrapper").show();
        $("._basket-total-wrapper").hide();
    } else {
        $(".basket-total").text(nTot);
        $("#basket_pounds").text(nPounds);
        $("#basket_pence").text(nPence);
        $("#basket_value").show();
        $("._basket-total-wrapper").show();
        $(document).on("click", ".fd-fav-basket .basket-btn", function() {
            $(window).scrollTop(0);
            $(".basket-popup").slideToggle("4000", "swing");
        });
        $(document).on("click", ".pinned-fav-basket .basket-btn", function() {
            $(window).scrollTop(0);
            $(".basket-popup").slideToggle("4000", "swing");
        });
        $(document).on("click", ".fd-header-xs .basket-btn", function() {
            $(window).scrollTop(0);
            $(".basket-popup").slideToggle("4000", "swing");
        });
    };
}
$(document).ready(function() {
    var iCutOffHr = 14;
    var bFavDelLocked = false;
    var fnTmplMenuArrows = function() {
        var iViewportWidth = $(window).width();
        if (iViewportWidth > 640) {
            $(".fd-navbar-default ul li .arrowbox").each(function() {
                var paddingVal = parseInt($(this).parent().css("padding-left")) + parseInt($(this).parent().css("padding-right"));
                $(this).width($(this).parent().width() + paddingVal);
            });
            $(".fd-navbar-default ul li .arrowbox img").each(function() {
                $(this).width($(this).parent().width());
                $(this).css("height", "20px");
            });
            $(".fd-navbar-default ul li").hover(function() {
                $(".arrowbox", this).show();
                $("#menu_panel_id").val($(this).attr("data-menu-id"));
                var oPos = $(this).offset();
                $("#menu_panel_left").val(oPos.left);
                if ($(this).attr("data-menu-type") == "menu") {
                    fnTmplMenuPanelShow();
                }
            }, function() {
                fnTmplMenuPanelHide();
            });
            $(".menu_panel").hover(function() {
                fnTmplMenuPanelShow();
            }, function() {
                fnTmplMenuPanelHide();
            });
        } else {
            $(".fd-navbar-default ul li").hover(function() {
                $(".arrowbox", this).hide();
            });
        };
        $(".pinned-menu button").hover(function() {
            $(".arrowbox", this).show();
        }, function() {
            $(".arrowbox", this).hide();
        });
        $(".fd-header-xs .menu-wrapper button").hover(function() {
            $(".arrowbox", this).show();
        }, function() {
            $(".arrowbox", this).hide();
        });
        $(window).trigger("scroll");
    };
    var fnTmplMenuPanelShow = function() {
        var nMenuId = $("#menu_panel_id").val();
        $("#menu_panel_" + nMenuId).show();
        var iPanelWidth = Number($("#menu_panel_" + nMenuId + " .menu_panel_inner").width());
        var iViewportWidth = Number($(window).width());
        var iMenuLeft = Number($("#menu_panel_left").val());
        if ((iMenuLeft + iPanelWidth) > iViewportWidth) {
            $("#menu_panel_" + nMenuId).css("left", (iViewportWidth - iPanelWidth) + "px");
        } else {
            $("#menu_panel_" + nMenuId).css("left", iMenuLeft + "px");
        }
        $("#menu_panel_" + nMenuId).show();
        $(".fd-navbar-default ul li[data-menu-id='" + nMenuId + "'] .arrowbox").show();
        $(".fd-navbar-default ul li[data-menu-id='" + nMenuId + "']").addClass("sticky");
    }
    var fnTmplMenuPanelHide = function() {
        $(".menu_panel").hide();
        var nMenuId = $("#menu_panel_id").val();
        $(".fd-navbar-default ul li[data-menu-id='" + nMenuId + "'] .arrowbox").hide();
        $(".fd-navbar-default ul li[data-menu-id='" + nMenuId + "']").removeClass("sticky");
    }
    var fnTmplUpdateFavourites = function() {
        $.ajax({
            type: "POST",
            url: "/costumes/AJAX_favourites",
            data: "",
            dataType: "text",
            success: function(sResponse) {
                $(".favs-popup").html(sResponse);
                var nTot = $(".favs-popup #favs_total_in").val();
                if (nTot == 0) {
                    $(".favs-popup").hide();
                    $(".favs-btn").addClass("no-favs");
                    $(document).off("click", ".favs-btn");
                    $(document).off("click", ".favs-popup .header button.close");
                } else {
                    $("._favs_total").text(nTot);
                    $(".favs-btn").removeClass("no-favs");
                    $(document).off("click", ".favs-btn");
                    $(document).off("click", ".favs-popup .header button.close");
                    $(document).on("click", ".favs-btn", function() {
                        $(window).scrollTop(0);
                        $(".favs-popup").slideToggle("4000", "swing");
                    });
                    $(document).on("click", ".favs-popup .header button.close", function() {
                        $(".favs-popup").slideToggle("4000", "swing");
                    });
                };
            },
            error: function(response, textStatus, errorThrown) {}
        });
    }
    var fnTmplUpdateBasket = function() {
        $.ajax({
            type: "POST",
            url: "/costumes/AJAX_basket_panel",
            data: "",
            dataType: "text",
            success: function(sResponse) {
                fnTmplShowUpdateBasket(sResponse)
            },
            error: function(response, textStatus, errorThrown) {}
        });
    }
    var fnTmplShowBlogFeed = function() {
        $.ajax({
            url: '//www.google.com/jsapi',
            dataType: 'script',
            cache: true,
            success: function() {
                google.load("feeds", "1", {
                    'callback': function() {
                        var feed = new google.feeds.Feed("http://blog.fancydress.com/feed/");
                        feed.setNumEntries(2)
                        feed.load(function(result) {
                            if (!result.error) {
                                $("div#footer_blog_entries").html("");
                                for (var i = 0; i < result.feed.entries.length; i++) {
                                    var oEntry = result.feed.entries[i];
                                    var sEntry = $("#__footer_blog_prototype").html();
                                    sEntry = sEntry.replace(/__TITLE__/g, oEntry.title);
                                    sEntry = sEntry.replace(/__LINK__/g, oEntry.link);
                                    sEntry = sEntry.replace(/__SNIPPET__/g, oEntry.contentSnippet);
                                    $("div#footer_blog_entries").append(sEntry);
                                }
                            }
                        });
                    }
                });
            }
        });
    }
    var fnCountDown = function() {
        var dNow = new Date();
        var dThen = new Date(dNow.getFullYear(), dNow.getMonth(), dNow.getDate(), iCutOffHr, 0, 0);
        var sMsg = "";
        var iDayOffset = 0;
        if (dNow.getHours() < iCutOffHr) {
            switch (dNow.getDay()) {
            case 5:
                iDayOffset = 0;
                sMsg = "SAME DAY DISPATCH";
                break;
            case 6:
                iDayOffset = 48;
                sMsg = "MONDAY DISPATCH";
                break;
            case 0:
                iDayOffset = 24;
                sMsg = "MONDAY DISPATCH";
                break;
            default:
                iDayOffset = 0;
                sMsg = "NEXT DAY DELIVERY";
            }
        } else {
            switch (dNow.getDay()) {
            case 5:
                iDayOffset = 48;
                sMsg = "MONDAY DISPATCH";
                break;
            case 6:
                iDayOffset = 24;
                sMsg = "MONDAY DISPATCH";
                break;
            case 7:
                iDayOffset = 0;
                sMsg = "MONDAY DISPATCH";
                break;
            default:
                iDayOffset = 0;
                sMsg = "NEXT DAY DISPATCH";
            }
        }
        var dDiff = new Date(dThen - dNow);
        var sDiffHrs = dDiff.getUTCHours() + iDayOffset;
        var sDiffMins = dDiff.getUTCMinutes();
        if (sDiffHrs == 0 && sDiffMins == 0) {
            sDiffHrs = 24;
        }
        if (sDiffHrs > 1) {
            $("._cd_hrs").text(sDiffHrs + " hrs");
        } else if (sDiffHrs == 1) {
            $("._cd_hrs").text("1 hr");
        } else {
            $("._cd_hrs").text("");
        }
        if (sDiffMins > 1) {
            $("._cd_mins").text(sDiffMins + " mins");
        } else if (sDiffMins == 1) {
            $("._cd_mins").text("1 min");
        } else {
            $("._cd_mins").text("");
        }
        $("._cd_msg").html(sMsg);
        setTimeout(function() {
            fnCountDown()
        }, 900);
    };
    $(".fd-navbar-default ul li .arrowbox").hide();
    $('.pinned-menu button .arrowbox').hide();
    $('.fd-header-xs .menu-wrapper button .arrowbox').hide();
    $(".no-favs-msg").hide();
    $(".favs-popup").hide();
    $(".basket-popup").hide();
    $(".menu_panel").hide();
    $(".product .favourite").tooltip({
        delay: {
            show: 500,
            hide: 500
        },
        html: 'true'
    });
    $("#newsletter_fnames").popover({
        content: "You must supply a first name",
        placement: "bottom",
        trigger: "manual",
        html: true
    });
    $("#newsletter_sname").popover({
        content: "You must supply a last name",
        placement: "bottom",
        trigger: "manual",
        html: true
    });
    $("#newsletter_email").popover({
        content: "You must supply a valid email address",
        placement: "bottom",
        trigger: "manual",
        html: true
    });
    $('.pinned-menu .pinned-search-input').hide();
    $('.pinned-menu .close-search').hide();
    $(window).resize(fnTmplMenuArrows);
    $(document).on("hover", ".no-favs", function() {
        $(".no-favs-msg").slideToggle("4000", "swing");
    });
    $(document).on("click", ".basket-popup .header button.close", function() {
        $(".basket-popup").slideToggle("4000", "swing");
    });
    $("._search_form").submit(function() {
        $("._search-btn-bg").css("background-position", "-100% -100%");
        $("._search-btn-icon").hide();
        $("._search-btn-wait").show();
    });
    $("#newsletter_signup_email").focus(function() {
        $('#newsletter_signup_submit').popover("hide");
    });
    $("form#newsletter_signup").submit(function() {
        $("#newsletter_signup_submit").focus();
        var sEmail = $("#newsletter_signup_email").val();
        if (validateEmail(sEmail) == true) {
            $("#newsletter_email").val(sEmail);
            $("#dialog_newsletter_thankyou").modal("show");
            $("#newsletter_signup_email").val("");
        } else {
            $('#newsletter_signup_submit').popover("show");
        }
        return false;
    });
    $("#newsletter_enrol input").focus(function() {
        $(this).popover("hide");
    });
    $("#newsletter_enrol").submit(function() {
        var bErrors = false;
        var sFNames = $("#newsletter_fnames").val();
        var sSName = $("#newsletter_sname").val();
        var sEmail = $("#newsletter_email").val();
        var sGender = $("#newsletter_enrol input[type='radio']:checked").val();
        if (sFNames.length == 0) {
            $("#newsletter_fnames").popover("show");
            bErrors = true;
        }
        if (sSName.length == 0) {
            $("#newsletter_sname").popover("show");
            bErrors = true;
        }
        if (validateEmail(sEmail) != true) {
            $("#newsletter_email").popover("show");
            bErrors = true;
        }
        if (bErrors == false) {
            var sQryStr = "";
            sQryStr = sQryStr + "fnames=" + sFNames;
            sQryStr = sQryStr + "&sname=" + sSName;
            sQryStr = sQryStr + "&email=" + sEmail;
            sQryStr = sQryStr + "&gender=" + sGender;
            $.ajax({
                type: "POST",
                url: "/costumes/AJAX_newsletter_enrol",
                data: sQryStr,
                dataType: "text",
                success: function(sResponse) {
                    $("#newsletter_enrol").hide("slide");
                    $("#newsletter_enrol_success").show("slide");
                },
                error: function(response, textStatus, errorThrown) {
                    alert("Oops something went wrong: " + textStatus);
                }
            });
        }
        return false;
    });
    $(document).on("click", ".product .favourite", function() {
        var sQryStr = "product_id=" + $(this).attr("data-product-id");
        $.ajax({
            context: this,
            type: "POST",
            url: "/costumes/AJAX_add_to_favourites",
            data: sQryStr,
            success: function(sResponse) {
                var aResponse = sResponse.split("|");
                if (aResponse[0] == "OK") {
                    if (aResponse[1] == "Remove") {
                        $(this).find(".fav-active").hide();
                        $(this).find(".fav-inactive").show();
                    } else {
                        $(this).children("img.fav-active").show();
                        $(this).children("img.fav-inactive").hide();
                    };
                    fnTmplUpdateFavourites();
                } else {
                    alert("Oops something went wrong: " + aResponse[1]);
                };
            },
            error: function(response, textStatus, errorThrown) {
                alert("Oops something went wrong: " + textStatus);
            }
        });
    });
    $(document).on("click", "button.delete-favourite", function() {
        if (bFavDelLocked == true)
            return false;
        bFavDelLocked = true;
        var sPID = $(this).attr("data-product-id")
        var sQryStr = "product_id=" + sPID;
        $.ajax({
            context: this,
            type: "POST",
            url: "/costumes/AJAX_add_to_favourites",
            data: sQryStr,
            success: function(sResponse) {
                var aResponse = sResponse.split("|");
                if (aResponse[0] == "OK") {
                    fnTmplUpdateFavourites();
                    $(".product .favourite[data-product-id='" + sPID + "'] img.fav-active").hide();
                    $(".product .favourite[data-product-id='" + sPID + "'] img.fav-inactive").show();
                    bFavDelLocked = false;
                } else {
                    alert("Oops something went wrong: " + aResponse[1]);
                    bFavDelLocked = false;
                };
            },
            error: function(response, textStatus, errorThrown) {
                alert("Oops something went wrong: " + textStatus);
                bFavDelLocked = false;
            }
        });
    });
    $(document).on("change", "select.fav_sizes", function() {
        var sQryStr = "product_id=" + $(this).attr("data-pid");
        sQryStr = sQryStr + "&cur_size_sku=" + $(this).attr("data-pid");
        sQryStr = sQryStr + "&size_sku=" + $(this).val();
        $.ajax({
            context: this,
            type: "POST",
            url: "/costumes/AJAX_add_size_to_favourites",
            data: sQryStr,
            success: function(sResponse) {
                var aResponse = sResponse.split("|");
                if (aResponse[0] == "OK") {
                    fnTmplUpdateFavourites();
                } else {
                    alert("Oops something went wrong: " + aResponse[1]);
                };
            },
            error: function(response, textStatus, errorThrown) {
                alert("Oops something went wrong: " + textStatus);
            }
        });
    });
    $(document).on("click", "button#add_favs", function() {
        $(document).off("click", "button#add_favs");
        var sQryStr = "";
        var nCnt = 0;
        $(this).parent().find(".summary-product").each(function() {
            if ($(this).find(".fav_sizes").attr("in_stock") == "true") {
                var sSKU = $(this).find(".fav_sizes").attr("data-cur-size");
                var sPID = $(this).attr("data-pid");
                if (sSKU == "00000000") {
                    sQryStr = sQryStr + "&pids." + nCnt + "=" + sPID;
                } else {
                    sQryStr = sQryStr + "&pids." + nCnt + "=" + sPID + "-" + sSKU;
                }
                sQryStr = sQryStr + "&qtys." + nCnt + "=1";
                nCnt++
            }
        });
        sQryStr = sQryStr + "&mode=favs";
        $.ajax({
            type: "POST",
            url: "/costumes/AJAX_add_multi_to_basket",
            data: sQryStr,
            dataType: "text",
            success: function(sResponse) {
                var aResult = sResponse.split("|");
                if (aResult[0] == "OK") {
                    window.location = "/costumes/basket";
                } else {
                    alert("Sorry, we had a problem: " + aResult[1]);
                }
            },
            error: function(response, textStatus, errorThrown) {
                alert("Oops something went wrong: " + textStatus);
            }
        });
    });
    $(window).scroll(function() {
        if ($("#allow_pinned_header").val() == "true") {
            if ($(this).scrollTop() > 180 && $(window).width() > 970) {
                $(".pinned-menu-wrapper").slideDown();
            } else {
                $(".pinned-menu-wrapper").slideUp();
            };
        };
    });
    $(document).on("click", ".pinned-menu .pinned-search-btn", function() {
        if ($(".pinned-menu .pinned-search-input:visible").length == 0) {
            $(".pinned-menu .pinned-search-input").show();
            $(".pinned-menu .pinned-search-input").animate({
                width: "540px",
                padding: "6px 12px"
            }, 500, "swing");
            $(this).animate({
                left: "540px",
            }, 500, "swing", function() {
                $(".close-search").show();
                $(".pinned-benefits-button").addClass("disabled");
            });
            $(".pinned-menu .pinned-search-input").focus();
        } else {
            $("#fd-search").val($("#fd-pinned-search").val());
            $("#search_form").submit();
        }
        return false;
    });
    $(document).on("keypress", ".pinned-menu .pinned-search-input", function(event) {
        if (event.which == 13) {
            $(".pinned-menu .pinned-search-btn").trigger("click");
            event.preventDefault();
        }
    });
    $(document).on("click", ".close-search", function() {
        $(this).hide();
        $(".pinned-menu .pinned-search-input").animate({
            width: "0px",
            padding: "0px"
        }, 500, "swing");
        $(".pinned-menu .pinned-search-btn").animate({
            left: "0px",
        }, 500, "swing", function() {
            $(".pinned-menu .pinned-search-input").hide();
            $(".pinned-benefits-button").removeClass("disabled");
        });
    });
    $(".next-day").hover(function() {
        $('#icon-delivery').attr('src', sStaticRoot + '/resources/images/delivery-anim.gif');
        $(this).children('p').stop().animate({
            "color": "#39B3E6"
        }, 100);
    }, function() {
        $('#icon-delivery').attr('src', sStaticRoot + '/resources/images/delivery.png');
        $(this).children('p').stop().animate({
            "color": "#008CC7"
        }, 200);
    });
    $(".lowest-price").hover(function() {
        $('#icon-piggybank').attr('src', sStaticRoot + '/resources/images/piggybank-anim.gif');
        $(this).children('p').stop().animate({
            "color": "#39B3E6"
        }, 100);
    }, function() {
        $('#icon-piggybank').attr('src', sStaticRoot + '/resources/images/piggybank.png');
        $(this).children('p').stop().animate({
            "color": "#008CC7"
        }, 200);
    });
    $(".largest-selection").hover(function() {
        $('#icon-hangers').attr('src', sStaticRoot + '/resources/images/hangers02-anim.gif');
        $(this).children('p').stop().animate({
            "color": "#39B3E6"
        }, 100);
    }, function() {
        $('#icon-hangers').attr('src', sStaticRoot + '/resources/images/hangers02.png');
        $(this).children('p').stop().animate({
            "color": "#008CC7"
        }, 200);
    });
    $(".star-supplier").hover(function() {
        $('#icon-star').attr('src', sStaticRoot + '/resources/images/star03-anim.gif');
        $(this).children('p').stop().animate({
            "color": "#39B3E6"
        }, 100);
    }, function() {
        $('#icon-star').attr('src', sStaticRoot + '/resources/images/star03.png');
        $(this).children('p').stop().animate({
            "color": "#008CC7"
        }, 200);
    });
    $(".free-delivery").hover(function() {
        $('._cd_stopwatch').attr('src', sStaticRoot + '/resources/images/stopwatch-anim.gif');
        $(this).children('p').stop().animate({
            "color": "#39B3E6"
        }, 100);
    }, function() {
        $('._cd_stopwatch').attr('src', sStaticRoot + '/resources/images/stopwatch03.png');
        $(this).children('p').stop().animate({
            "color": "#717171"
        }, 200);
    });
    fnTmplMenuArrows();
    fnTmplUpdateFavourites();
    fnTmplUpdateBasket();
    fnTmplShowBlogFeed();
    fnCountDown();
    $(".ios-input-fix").css("-webkit-appearance", "none");
    $(".ios-input-fix").css("-webkit-border-radius", "0px");
    $(".ios-input-fix").css("border-radius", "3px 0 0 3px");
});
