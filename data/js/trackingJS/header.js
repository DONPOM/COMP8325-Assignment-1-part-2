var $j = jQuery.noConflict();
var changing_thumbs = new Array();
var disableFlipbook = "false";
var largeVersionMinimumScreenSize = 1900;
var screensize = window.screen.width;
var BrowserDetect = {
    init: function() {
        this.browser = this.searchString(this.dataBrowser) || "Other";
        this.version = this.searchVersion(navigator.userAgent) || this.searchVersion(navigator.appVersion) || "Unknown"
    },
    searchString: function(c) {
        for (var a = 0; a < c.length; a++) {
            var b = c[a].string;
            this.versionSearchString = c[a].subString;
            if (b.indexOf(c[a].subString)!=-1) {
                return c[a].identity
            }
        }
    },
    searchVersion: function(b) {
        var a = b.indexOf(this.versionSearchString);
        if (a==-1) {
            return 
        }
        return parseFloat(b.substring(a + this.versionSearchString.length + 1))
    },
    dataBrowser: [{
        string: navigator.userAgent,
        subString: "Chrome",
        identity: "Chrome"
    }, {
        string: navigator.userAgent,
        subString: "MSIE",
        identity: "Explorer"
    }, {
        string: navigator.userAgent,
        subString: "Firefox",
        identity: "Firefox"
    }, {
        string: navigator.userAgent,
        subString: "Safari",
        identity: "Safari"
    }, {
        string: navigator.userAgent,
        subString: "Opera",
        identity: "Opera"
    }
    ]
};
BrowserDetect.init();
function Performance(b) {
    var a = this;
    a.listener = b;
    a.domInteractive = false;
    a.domContentLoadedEventStart = false;
    a.domContentLoadedEventEnd = false;
    a.domComplete = false;
    a.loadEventStart = false;
    a.loadEventEnd = false;
    a.isset = function(c) {
        if (typeof c != "undefined") {
            return true
        }
        return false
    };
    a.test = function() {
        if (!a.domInteractive && performance.timing.domInteractive > 0) {
            a.domInteractive = true;
            var c = performance.timing.domInteractive - performance.timing.navigationStart;
            if (a.isset(a.listener) && a.listener.domInteractive) {
                a.listener.domInteractive(c)
            }
        }
        if (!a.domContentLoadedEventStart && performance.timing.domContentLoadedEventStart > 0) {
            a.domContentLoadedEventStart = true;
            var c = performance.timing.domContentLoadedEventStart - performance.timing.navigationStart;
            if (a.isset(a.listener) && a.listener.domContentLoadedEventStart) {
                a.listener.domContentLoadedEventStart(c)
            }
        }
        if (!a.domContentLoadedEventEnd && performance.timing.domContentLoadedEventEnd > 0) {
            a.domContentLoadedEventEnd = true;
            var c = performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart;
            if (a.isset(a.listener) && a.listener.domContentLoadedEventEnd) {
                a.listener.domContentLoadedEventEnd(c)
            }
        }
        if (!a.domComplete && performance.timing.domComplete > 0) {
            a.domComplete = true;
            var c = performance.timing.domComplete - performance.timing.navigationStart;
            if (a.isset(a.listener) && a.listener.domComplete) {
                a.listener.domComplete(c)
            }
        }
        if (!a.loadEventStart && performance.timing.loadEventStart > 0) {
            a.loadEventStart = true;
            var c = performance.timing.loadEventStart - performance.timing.navigationStart;
            if (a.isset(a.listener) && a.listener.loadEventStart) {
                a.listener.loadEventStart(c)
            }
        }
        if (!a.loadEventEnd && performance.timing.loadEventEnd > 0) {
            a.loadEventEnd = true;
            var c = performance.timing.loadEventEnd - performance.timing.navigationStart;
            if (a.isset(a.listener) && a.listener.loadEventEnd) {
                a.listener.loadEventEnd(c)
            }
        }
        if (!(a.domInteractive && a.domInteractive && a.domContentLoadedEventStart && a.domContentLoadedEventEnd && a.domComplete && a.loadEventStart)) {
            setTimeout(function() {
                a.test()
            }, 250)
        }
    };
    setTimeout(function() {
        a.test()
    }, 250)
}
function isIpad() {
    return (navigator.platform.indexOf("iPad")!=-1)
}
function toggleClickEvent() {
    var a;
    (isIpad()) ? a = "touchstart" : a = "click";
    return a
}
function toggle(c, a) {
    var b = document.getElementById(c);
    if (b.style.display == "none") {
        b.style.display = a;
        b.focus()
    } else {
        b.style.display = "none"
    }
}
var stopChangingThumbs = false;
$j(window).on("beforeunload", function() {
    stopChangingThumbs = true
});
function changeThumb(d, a, b, e, f) {
    if (changing_thumbs[a]) {
        if (f.indexOf("{i}")!==-1) {
            changeThumbPreload(a, f.replace("{i}", b))
        } else {
            if (f.indexOf("{index}")!==-1) {
                changeThumbPreload(a, f.replace("{index}", b))
            } else {
                var c = ($j("#" + a).width() == 240);
                if (f.indexOf("cdn1a.image.pornhub.phncdn.com")>-1 || f.indexOf("cdn1b.image.pornhub.phncdn.com")>-1 || f.indexOf("cdn1.image.keezmovies.phncdn.com")>-1 || f.indexOf("cdn1.videothumbs.xtube.com")>-1) {
                    imgBase = "";
                    f = c ? f.replace(/160x120/gi, "240x180") : f
                } else {
                    imgBase = c && d > 2400000 ? "medium" : "small"
                }
                if (f.indexOf(".jpg", f.length - 4)!==-1) {
                    lastIndexOfDash = f.lastIndexOf("/");
                    f = f.substring(0, lastIndexOfDash + 1);
                    changeThumbPreload(a, f + b + ".jpg")
                } else {
                    changeThumbPreload(a, f + imgBase + b + ".jpg")
                }
            }
        }
        b = b%e + 1;
        changing_thumbs[a] = setTimeout("changeThumb(" + d + ", '" + a + "', '" + b + "', '" + e + "', '" + f + "' )", 600)
    }
}
function changeThumbPreload(b, a) {
    if (!stopChangingThumbs) {
        $j(new Image()).attr("src", a).on("load", function() {
            $j("#" + b).attr("src", a)
        })
    }
}
function startThumbChange(b, a, c, d) {
    if (disableFlipbook == "false") {
        changing_thumbs[a] = true;
        changeThumb(b, a, 1, c, d)
    }
}
function endThumbChange(a) {
    if (disableFlipbook == "false") {
        var b = parseInt(screensize) < largeVersionMinimumScreenSize ? "smallthumb": "mediumthumb";
        $this = $j("#" + a);
        $this.attr("src", $this.data(b));
        clearTimeout(changing_thumbs[a])
    }
}
function setCookieAdvanced(b, g, c, j, e, a) {
    if (c) {
        var f = new Date();
        f.setTime(f.getTime());
        c = c * 1000 * 60 * 60 * 24;
        var h = new Date(f.getTime() + c)
    }
    var d = getDomain();
    document.cookie = b + "=" + escape(g) + (c ? ";expires=" + h.toGMTString() : "") + (j ? ";path=" + j : ";path=/") + (e ? ";domain=" + e : ";domain=" + d) + (a ? ";secure" : "")
}
function getCookieAdvanced(a) {
    var f = document.cookie.split(";");
    var b = "";
    var d = "";
    var e = "";
    var c = false;
    for (i = 0; i < f.length; i++) {
        b = f[i].split("=");
        d = b[0].replace(/^\s+|\s+$/g, "");
        if (d == a) {
            c = true;
            if (b.length > 1) {
                e = unescape(b[1].replace(/^\s+|\s+$/g, ""))
            }
            return e;
            break
        }
        b = null;
        d = ""
    }
    if (!c) {
        return null
    }
}
function deleteCookieAdvanced(a, d, c) {
    if (getCookieAdvanced(a)) {
        var b = getDomain();
        document.cookie = a + "=" + (d ? ";path=" + d : ";path=/") + (c ? ";domain=" + c : ";domain=" + b) + ";expires=Thu, 01-Jan-1970 00:00:01 GMT"
    }
}
function getDomain() {
    var c = document.domain;
    var b = c.toString().split(".".toString());
    var a = b[b.length - 2] + "." + b[b.length - 1];
    return a
}
function ajaxPost(b, a) {
    $j.post(b, function(c) {
        if (a) {
            $j(a).html(c)
        }
    });
    return false
}
function ajaxPostNew(b, a, c) {
    if (a) {
        $j(a).children().css({
            visibility: "hidden"
        });
        $j(a).find(".spinner").css({
            visibility: "visible"
        })
    }
    $j.post(b, function(e) {
        if (c) {
            location.reload()
        } else {
            if (a) {
                var d = $j(a);
                d.after($j(e));
                d.remove()
            }
        }
    });
    return false
}
var yesNoModal = {
    show: function(c, a, b) {
        $j("body").off(toggleClickEvent(), ".mobileFriedly.yesBtn").on(toggleClickEvent(), ".mobileFriedly.yesBtn", function(d) {
            d.preventDefault();
            modal.hide();
            a()
        });
        $j("body").off(toggleClickEvent(), ".mobileFriedly.noBtn, .mobileFriedly.modal-close").on(toggleClickEvent(), ".mobileFriedly.noBtn, .mobileFriedly.modal-close", function(d) {
            d.preventDefault();
            modal.hide();
            if (typeof b === "function") {
                b()
            }
        });
        modal.show("yesNoModal");
        $j("#yesNoModal").find(".text").html(c)
    }
};
var infoModal = {
    show: function(b, d, c, a) {
        $j("body").off(toggleClickEvent(), ".mobileFriedly.okBtn, .mobileFriedly.modal-close").on(toggleClickEvent(), ".mobileFriedly.okBtn, .mobileFriedly.modal-close", function(f) {
            f.preventDefault();
            modal.hide()
        });
        modal.show("infoModal");
        elements = [{
            selector: ".modal-title",
            value: b
        }, {
            selector: ".text",
            value: d
        }, {
            selector: ".success-text",
            value: c
        }, {
            selector: ".error-text",
            value: a
        }
        ];
        for (i = 0; i < elements.length; i++) {
            element = $j("#infoModal").find(elements[i].selector);
            if (elements[i].value) {
                element.html(elements[i].value);
                element.show()
            } else {
                element.hide()
            }
        }
    }
};
var searchTypes = [{
    name: "videos",
    autocompleteUrl: "/video/search_autocomplete",
    submitUrl: searchUrlVideo,
    testTypePattern: "video"
}, {
    name: "photos",
    autocompleteUrl: "/video/search_autocomplete",
    submitUrl: searchUrlPhoto,
    testTypePattern: "albums"
}, {
    name: "members",
    autocompleteUrl: null,
    submitUrl: searchUrlMember,
    testTypePattern: "user"
}, {
    name: "pornstars",
    autocompleteUrl: "/pornstars/search_autocomplete",
    submitUrl: searchUrlPornstar,
    testTypePattern: "pornstars"
}, {
    name: "gifs",
    autocompleteUrl: null,
    submitUrl: searchUrlGifs,
    testTypePattern: "gifs"
}
];
function getSearchType(a) {
    if (!a) {
        a = $j("#searchTypeSelected").data("search-type")
    }
    for (i = 0; i < searchTypes.length; i++) {
        if (searchTypes[i].name == a) {
            return searchTypes[i]
        }
    }
}
function setSearchType(a) {
    if (typeof a == "string") {
        a = getSearchType(a)
    }
    $j("#searchTypeSelected").data("search-type", a.name);
    $j("#searchTypeSelected i").attr("class", a.name);
    $j("#searchTypes").hide();
    $j("input#searchInput").focus();
    for (i = 0; i < searchTypes.length; i++) {
        if (searchTypes[i].name != a.name) {}
    }
}
_currentMenuId = "";
function showMenu(a) {
    if ($j("#" + a).hasClass("_selected")) {
        $j("#" + a).children(".wideDropdown").show();
        if (_currentMenuId != "" && _currentMenuId != a) {
            $j("#" + _currentMenuId).children(".wideDropdown").hide()
        }
        _currentMenuId = a
    }
}
function hideMenu(a) {
    if ($j("#headerMainMenu > ._selected").length == 0) {
        if (!($j("#" + a).hasClass("_selected"))) {
            $j("#" + a).children(".wideDropdown").hide()
        }
    }
}
$j(document).ready(function() {
    if (navigator.userAgent.indexOf("PLAYSTATION")>-1) {
        $j("#headerLoginLink").attr("href", loginPlaystation)
    }
    $j(".wideDropdown").each(function() {
        $j(".menu." + $j(this).data("submenu-type")).append($j(".wideDropdown." + $j(this).data("submenu-type")))
    });
    $j("#headerMainMenu > li").on({
        mouseenter: function() {
            $j(this).addClass("_selected");
            setTimeout('showMenu("' + this.id + '")', 250)
        },
        mouseleave: function() {
            $j(this).removeClass("_selected");
            setTimeout('hideMenu("' + this.id + '")', 250)
        }
    });
    $j(".menu").on({
        mouseenter: function() {
            $j(this).addClass("hovered")
        },
        mouseleave: function() {
            $j(this).removeClass("hovered")
        }
    });
    $j(".wideDropdown").on({
        mouseenter: function() {
            $j(this).prev().addClass("hovered")
        },
        mouseleave: function() {
            $j(this).prev().removeClass("hovered")
        }
    });
    $j(".menu.community, .menu.community *").on({
        mouseenter: function() {
            $j("#headerMenuContainer").addClass("hacked")
        },
        mouseleave: function() {
            $j("#headerMenuContainer").removeClass("hacked")
        }
    });
    for (i = 0; i < searchTypes.length; i++) {
        $j("#searchTypes").append('<li class="' + searchTypes[i].name + '" data-search-type="' + searchTypes[i].name + '"><i class="' + searchTypes[i].name + '"></i><span>' + searchTypes[i].name + "</span></li>")
    }
    $j("#searchTypeWrapper #searchTypes li.gifs span").replaceWith("<span>GIFs</span>");
    makeDropdown("#languageWrapper");
    makeDropdown(".languageWrapper");
    makeDropdown("#searchTypeWrapper");
    makeDropdown("#notificationIcons", "'.notificationIcon[data-type=\"' + $j('#notificationBox').attr('type') + '\"] > span'");
    if (isLogged) {
        makeDropdown("#profileMenuWrapper")
    }
    var a = searchTypes[0];
    var b = $j(location).attr("pathname").split("/")[1];
    if (b) {
        for (i = 0; i < searchTypes.length; i++) {
            if (searchTypes[i].testTypePattern == b) {
                a = searchTypes[i]
            }
        }
    }
    setSearchType(a);
    $j("ul#searchTypes li:nth-child(n + 2)").click(function() {
        setSearchType($j(this).data("search-type"))
    });
    (focusSearchInput) ? $j("input#searchInput").focus() : $j("input#searchInput").blur();
    $j("input#searchInput").on("keyup", function(g) {
        var f = RegExp(/[\<\>]/gi), h = $j("input#searchInput"), d = h.val().match(f), c = h.val().replace(f, "");
        if (d) {
            h.val(c)
        }
    });
    $j("input#searchInput").keypress(function(d) {
        var c = d.charCode || d.keyCode || d.altKey;
        if (c == 60 || c == 62) {
            return false
        }
    });
    $j("input#searchInput").autocomplete({
        minLength: 2,
        delay: 400,
        appendTo: "#searchBarContainer",
        source: function(g, e) {
            if (!getSearchType().autocompleteUrl) {
                return false
            }
            var f = g.term.toLowerCase().substr(0, 24), d = this.element, j = getSearchType()["name"], c = d.data("autocomplete-cache") || {}, h = false;
            if (typeof c[j] == "undefined") {
                c[j] = new Object()
            }
            $j.each(c[j], function(k, l) {
                if (f == k) {
                    e(l);
                    h = true;
                    return 
                }
            });
            if (h) {
                return 
            }
            $j.ajax({
                url: getSearchType().autocompleteUrl + "?pornstars=true",
                dataType: "json",
                data: {
                    search_rnd: d.attr("data-searchRnd"),
                    search_hash: d.attr("data-searchHash"),
                    orientation: d.attr("data-orientation"),
                    q: g.term,
                    alt: d.attr("data-alt")
                },
                success: function(l) {
                    if (l instanceof Array && l.length) {
                        l = $j.map(l, function(m) {
                            return {
                                label: m,
                                value: m
                            }
                        })
                    } else {
                        var k = [];
                        if (l.queries instanceof Array && l.queries.length) {
                            k = k.concat(new Array({
                                label: j.capitalize() + " results",
                                value: null
                            })).concat($j.map(l.queries, function(m) {
                                return {
                                    label: m,
                                    value: m
                                }
                            }))
                        }
                        if (l.pornstars instanceof Array && l.pornstars.length) {
                            k = k.concat(new Array({
                                label: "Pornstar profiles",
                                value: null
                            })).concat($j.map(l.pornstars, function(m) {
                                return {
                                    label: m.name + " - Rank: " + m.rank + "(" + m.slug + ")",
                                    value: m.name
                                }
                            }))
                        }
                        l = k
                    }
                    c[j][f] = l;
                    d.data("autocomplete-cache", c);
                    e(l)
                }
            })
        },
        select: function(d, e) {
            var c = d;
            while (c.originalEvent !== undefined) {
                c = c.originalEvent
            }
            if (c.type == "keydown") {
                $j("#btnSearch").trigger("click")
            }
        },
        open: function() {
            $j(".ui-autocomplete").css("z-index", 9999);
            var c = "query";
            $j(".ui-autocomplete .ui-menu-item a").each(function(f) {
                var j = $j(this).html();
                var h = $j("#searchInput").val();
                if (j == getSearchType()["name"].capitalize() + " results" || j == "Pornstar profiles") {
                    $j(this).addClass("label").on("click", function() {
                        return false
                    });
                    if (j == "Pornstar profiles") {
                        c = "pornstar"
                    }
                } else {
                    if (c == "query") {
                        $j(this).addClass("query")
                    } else {
                        if (c == "pornstar") {
                            var d = /\(.+?\)/.exec(j)[0];
                            j = j.replace(d, "");
                            $j(this).on("click", function() {
                                location.href = "/pornstar/" + d.substr(1, d.length - 2)
                            });
                            j = j.replace(" - Rank: ", "&nbsp;&nbsp;-&nbsp;&nbsp;Rank: ");
                            var g = /Rank: \d+/.exec(j)[0];
                            j = j.replace(g, '<span class="rank">' + g + "</span>")
                        }
                    }
                    var e = j.search(RegExp("" + h, "i"));
                    if (e >= 0) {
                        j = j.slice(0, e) + '<span class="soughtValue">' + h + "</span>" + j.slice(e + h.length)
                    }
                }
                $j(this).html(j)
            })
        },
        messages: {
            noResults: function() {
                return ""
            },
            results: function() {
                return ""
            }
        }
    });
    $j("input#searchInput").on("keyup", function(c) {
        input = $j("input#searchInput").val();
        if (c.keyCode == "13") {
            if (input == "hd" || input == "HD") {
                window.location = "/hd"
            } else {
                $j("#btnSearch").trigger("click")
            }
        }
    });
    $j(document).on("click", ".ui-menu-item a.query", function() {
        $j("#btnSearch").trigger("click")
    });
    $j("#btnSearch").on("click", function() {
        var c = $j("input#searchInput").val();
        if (c == "hd" || c == "HD") {
            window.location = "/hd"
        } else {
            $j("#search_form").submit()
        }
    });
    $j("form#search_form").on("submit", function(f) {
        var d = $j("input#searchInput").val().replace(/^[ \t]+|[ \t]+$/gi, ""), c = $j("input#searchInput").attr("data-val");
        if (d == "" || d == c) {
            f.preventDefault();
            return false
        } else {
            d = d.toLowerCase();
            d = encodeURIComponent(d);
            d = d.replace(/%20/g, "+");
            var g = getSearchType().submitUrl + d;
            $j(location).attr("href", g);
            return false
        }
    });
    if (isLogged) {
        $j("#notificationIcons .notificationIcon").on("click", function(j) {
            var f = $j(this), k = f.data("title"), c = f.data("type"), h = f.offset().left, g = $j("#notificationBox").width() / 2, d = f.width() / 2;
            $j("#notificationBox .titleContainer").text(k);
            $j("#notificationBox").css({
                left: Math.round(h - g + d)
            });
            $j("#notificationBox").data("type", c);
            $j("#notificationBox").attr("type", c);
            $j("ul#modelNotificationList > li, .handle3, .track3").hide();
            $j("#notificationBox #loadingDiv").show();
            setTimeout(function() {
                insertNotificationContent()
            }, 100)
        });
        $j("#notificationBox").on("click", ".subscribeConfirm label", function() {
            checkbox = $j(this).siblings("input");
            fakeCheckbox = $j(this).find(".fakeCheckBox");
            if (checkbox.prop("checked")) {
                fakeCheckbox.removeClass("checked")
            } else {
                fakeCheckbox.addClass("checked")
            }
        });
        $j("#notificationBox").on("click", ".friendRequestItem .btnFlag", function() {
            var c = this;
            yesNoModal.show("Are you sure you want to report as spam and reject this request?", function() {
                reportSpam($j(c).data("userid"))
            })
        })
    }
    $j("#languages li a, .languages li a").on("click", function() {
        if (location.hash != this.hash) {
            location.href = (this.href.indexOf("#")==-1 ? this.href : this.href.substring(0, this.href.indexOf("#"))) + location.hash;
            return false
        }
    });
    $j(document).on({
        mouseenter: function() {
            $this = $j(this);
            $j("#tooltip").html("<div class='tooltip-sub'>" + $this.data("title") + " </div></div><div class='arrow-down'></div>").show().addClass($this.attr("type"));
            $j("#tooltip").position({
                my: "center bottom-10",
                at: "center top",
                collision: "fit none",
                of: $this
            });
            $j(".arrow-down").position({
                my: "center bottom",
                at: "center top",
                collision: "none none",
                of: $this
            })
        },
        mouseleave: function() {
            $j("#tooltip").hide()
        }
    }, ".tooltipTrig")
});
