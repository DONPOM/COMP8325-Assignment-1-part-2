function getInnerHeight() {
    return self.innerHeight ? self.innerHeight : document.documentElement && document.documentElement.clientHeight ? document.documentElement.clientHeight : document.body ? document.body.clientHeight : void 0
}
function showTip(e, t, i, n, o, s, r) {
    var a = jQuery;
    n || (n = tipsShown, tipsShown++);
    var l = "string" == typeof t ? a("#" + t): a(t);
    if (1 == l.length) {
        1 == o ? s = s || 200 : s || (s = 400);
        var c = s + 11 + 11, u = a(document.body).width(), d = l.offset(), h = {
            height: l.height(),
            width: l.width()
        }, p = d.top + h.height + 12, f = d.left + h.width / 2 - c / 2;
        20 > f ? f = 20 : f + c > u - 20 && (f = u - 20 - c);
        var m = d.left + h.width / 2 - 13 - f;
        a("#tips").prepend("<div id='tip" + n + "' class='tooltip' style='display:none'>" + "<div class='carrot' style='left:" + m + "px'></div>" + "<a class='close'></a>" + "<div class='content' style='width:" + s + "px'>" + e + "</div>" + "</div>");
        var g = a("#tip" + n).outerHeight();
        p + g > a(window).height() && (p = d.top - g - 10, a("#tip" + n + " .carrot").remove()), r && (tooltipCallbacks["tip" + n] = r), a("#tip" + n).css({
            position: "absolute",
            top: p + "px",
            left: f + "px"
        }).on("mouseup", function(e) {
            e.stop()
        }), a("#tip" + n + " a.close").on("click", function() {
            hideTip("tip" + n, 1)
        }), 1 == o ? a("#tip" + n).show() : 2 == o ? a("#tip" + n).fadeIn() : setTimeout(function() {
            a("#tip" + n).css({
                position: "absolute",
                left: f + "px",
                top: getInnerHeight() / 2 + h.height + "px"
            }), a("#tip" + n).show(), a("#tip" + n).animate({
                top: p + "px"
            }, 1e3, "easeOutBounce")
        }, 500)
    }
}
function hideTip(e, t) {
    var i = jQuery;
    if (e && i("#" + e).length) {
        var n = tooltipCallbacks[e];
        n && delete tooltipCallbacks[e], 1 == t ? (i("#" + e).hide(), i("#tips").remove("#" + e), n && n()) : i("#" + e).fadeOut(function() {
            i("#" + e).length && i("#" + e).parent() == i("#tips") && (i("#tips").remove(i("#" + e)), n && n())
        })
    }
}
function hideAllTips() {
    for (var e = jQuery, t = 0; t < e("#tips").children().length; t++)
        hideTip(e("#tips").children()[t].id)
}
function done() {
    return !1
}
function isFlashInstalled() {
    function e(e) {
        return e = e.match(/[\d]+/g), e.length = 3, e.join(".")
    }
    if (void 0 !== hasFlash)
        return hasFlash;
    var t=!1, i = "";
    if (navigator.plugins && navigator.plugins.length) {
        var n = navigator.plugins["Shockwave Flash"];
        n && (t=!0, n.description && (i = e(n.description))), navigator.plugins["Shockwave Flash 2.0"] && (t=!0, i = "2.0.0.11")
    } else if (navigator.mimeTypes && navigator.mimeTypes.length) {
        var o = navigator.mimeTypes["application/x-shockwave-flash"];
        (t = o && o.enabledPlugin) && (i = e(o.enabledPlugin.description))
    } else 
        try {
            var s = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"), t=!0, i = e(s.GetVariable("$version"))
    } catch (r) {
        try {
            s = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"), t=!0, i = "6.0.21"
        } catch (a) {
            try {
                s = new ActiveXObject("ShockwaveFlash.ShockwaveFlash"), t=!0, i = e(s.GetVariable("$version"))
            } catch (l) {}
        }
    }
    return hasFlash=!!t
}
function showFlashWarning(e) {
    var t = _W.tl("Adobe Flash Player is required in order to use this feature."), i = e ? e: t;
    showError('<div style="text-align:center;">' + i + "<br/><br/><a target='_new' href='http://www.adobe.com/go/getflashplayer'><img style='border:0;' onClick='Effect.Fade(\"error2\");' src='http://" + editorStatic + "/weebly/images/get_flash.jpg'><a/></div>")
}
function checkContentEditable() {
    var e = new Element("div");
    "undefined" == typeof e.contentEditable && showError('<div style="text-align:center;">' + _W.tl("You are using an outdated browser. In order to edit text in the Weebly Editor you need to upgrade to a modern web browser.") + "<br/><br/><div style='width:350px; margin:10px auto; background:#FEEFDA;'><a href='http://www.firefox.com' target='_blank' style='margin-right:15px;'><img src='http://www.ie6nomore.com/files/theme/ie6nomore-firefox.jpg' style='border: none;' alt='Get Firefox 3.5'/></a><a href='http://www.browserforthebetter.com/download.html' target='_blank' style='margin-right:15px;'><img src='http://www.ie6nomore.com/files/theme/ie6nomore-ie8.jpg' style='border: none;' alt='Get Internet Explorer 8'/></a><a href='http://www.apple.com/safari/download/' target='_blank' style='margin-right:15px;'><img src='http://www.ie6nomore.com/files/theme/ie6nomore-safari.jpg' style='border: none;' alt='Get Safari 4'/></a><a href='http://www.google.com/chrome' target='_blank'><img src='http://www.ie6nomore.com/files/theme/ie6nomore-chrome.jpg' style='border: none;' alt='Get Google Chrome'/></a></div></div>")
}
function showEvent(e) {
    hasDoneUserEvent(e) || (EventTooltip.show(e), userEvents[e] = 1, _markUserEventDone(e))
}
function markUserEventDone(e) {
    userEvents[e] = 1, _markUserEventDone(e)
}
function _markUserEventDone(e) {
    preventAJAX() || (new Ajax.Request(ajax, {
        parameters: {
            pos: "doevent",
            event: e,
            cookie: document.cookie
        }
    }), fireTrackingEvent("Event", e))
}
function fireTrackingEvent(e, t, i, n) {
    try {
        if (n = n ? n : 0, !doTrackingEvent(e, t, i, n))
            throw "Did not process tracking event";
        if (Weebly.trackingArray.length > 0) {
            for (var o = 0; o < Weebly.trackingArray.length; o++) {
                var s = Weebly.trackingArray[o];
                doTrackingEvent(s.category, s.action, s.optional_label, s.optional_value)
            }
            Weebly.trackingArray = Array()
        }
    } catch (r) {
        Weebly.trackingArray.push({
            category: e,
            action: t,
            optional_label: i,
            optional_value: n
        })
    }
}
function doTrackingEvent(e, t, i, n) {
    try {
        "object" == typeof pageTracker ? pageTracker._trackEvent(e, t, i, parseInt(n)) : _gaq.push(["_trackEvent", e, t, i, parseInt(n)]), mpmetrics.track(e, {
            type: t
        })
    } catch (o) {
        return !1
    }
    return !0
}
function fireTransactionEvent(e, t, i, n, o) {
    try {
        "object" == typeof pageTracker ? (pageTracker._addTrans(e, n, i, "", "", "", "", ""), pageTracker._addItem(e, t, t, o, i, "1"), pageTracker._trackTrans()) : (_gaq.push(["_addTrans", e, n, i, "", "", "", "", ""]), _gaq.push(["_addItem", e, t, t, o, i, "1"]), _gaq.push(["_trackTrans"])), mpmetrics.track("Purchase " + o)
    } catch (s) {}
}
function showError(e, t, i) {
    "string" == typeof e ? ($("red-error-title").update(_W.tl("Oops!")), $("red-error-text").update(e)) : ($("red-error-title").update(e.title), $("red-error-text").update(e.message)), errorDialog || (errorDialog = new Weebly.Dialog($("red-error"), {
        inner_class: "weebly-dialog-inner-red",
        zIndex: 9999,
        modal: 1
    })), errorDialog.open(), i || fireTrackingEvent("WeeblyError", "Error", e);
    var n = {
        pos: "oopserror",
        message: e
    };
    return "object" == typeof t && (n.ajax_request = t.request.body.match(/^.*?&cookie/)[0], n.ajax_response = t.responseText, n.ajax_status = t.status, n.server = t.getHeader("X-Host"), t.request.times && (n.ajax_start = t.request.times.start, n.ajax_initialized = t.request.times.initialized, n.ajax_sent = t.request.times.sent, n.ajax_response_start = t.request.times.response, n.ajax_complete = (new Date).getTime() - t.request.times.start, n.current_upload = t.request.concurrentUpload ? 1 : 0)), n.cookie = document.cookie, new Ajax.Request(ajax, {
        parameters: n,
        bgRequest: !0
    }), errorDialog
}
function hideError() {
    errorDialog && errorDialog.close()
}
function showRetriableError(e, t, i) {
    "object" == typeof retriableErrorDialog && retriableErrorDialog.close();
    var n = "undefined" != typeof isClientSite ? isClientSite: !1, o = "undefined" != typeof removeSupportEmail ? removeSupportEmail: !1, s = o ? "": "<br/><br/>" + _W.tl("If you're still having trouble, please contact support at support@weebly.com.");
    s += "<br/><br/>", s += n ? "" : "<a href='#' onclick='window.location.href=\"http://" + configSiteName + "/\"' class='big-grey-button'><span class='button-inner'>" + _W.tl("Exit") + "</span></a>&nbsp;&nbsp;", s += "<a href='#' onclick='retryRetriableError();' class='big-blue-button'><span class='button-inner'>" + _W.tl("Retry") + "</span></a>", "string" == typeof e ? ($("red-error-title").update(_W.tl("Oops!")), $("red-error-text").update(e + s), fireTrackingEvent("WeeblyError", "Retriable Error", e)) : ($("red-error-title").update(e.title), $("red-error-text").update(e.message + s), fireTrackingEvent("WeeblyError", "Retriable Error", e.message)), retriableErrorDialog = new Weebly.Dialog($("red-error"), {
        inner_class: "weebly-dialog-inner-red",
        modal: 1,
        showClose: i
    }), retriableErrorDialog.open();
    try {
        var r = {
            pos: "oopserror",
            message: "string" == typeof e ? e: e.message
        };
        "object" != typeof t || t.body.match(/oopserror/) || (r.ajax_request = t.body.match(/^.*?&cookie/)[0], r.ajax_response = t.transport.responseText, r.ajax_status = t.transport.status, r.server = t.getHeader("X-Host"), t.times && (r.ajax_start = t.times.start, r.ajax_initialized = t.times.initialized, r.ajax_sent = t.times.sent, r.ajax_response_start = t.times.response, r.ajax_complete = (new Date).getTime() - t.times.start, r.current_upload = 0), r.cookie = document.cookie, new Ajax.Request("/weebly/getElements.php", {
            parameters: r,
            bgRequest: !0
        }), retriableAjax = t)
    } catch (a) {}
    return retriableErrorDialog
}
function retryRetriableError() {
    hideRetriableError(), retriableAjax.retry(!0)
}
function hideRetriableError() {
    retriableErrorDialog && retriableErrorDialog.close()
}
function showWarning(e, t) {
    return t = Object.extend({
        width: "585px",
        titleFontSize: "36px",
        showClose: !0,
        closeFunction: null
    }, t), $("orange-error").setStyle({
        width: t.width
    }), $("orange-error-title").setStyle({
        fontSize: t.titleFontSize
    }), "string" == typeof e ? ($("orange-error-title").update(_W.tl("Oops!")), $("orange-error-text").update(e), fireTrackingEvent("WeeblyError", "Warning", e)) : ($("orange-error-title").update(e.title), $("orange-error-text").update(e.message), fireTrackingEvent("WeeblyError", "Warning", e.message)), t.okButton ? $("orange-error-button").show().stopObserving("click").observe("click", function() {
        warningDialog.close()
    }) : $("orange-error-button").hide(), warningDialog = new Weebly.Dialog($("orange-error"), {
        inner_class: "weebly-dialog-inner-orange",
        modal: 1,
        showClose: t.showClose,
        closeFunction: t.closeFunction
    }), warningDialog.open(), warningDialog
}
function showDeniedAction(e) {
    jQuery("orange-error-title").setStyle({
        fontSize: "30px"
    }).update("Text Not Allowed"), jQuery("orange-error-text").setStyle({
        marginTop: "25px",
        marginBottom: "25px"
    });
    var t = new Element("a", {
        "class": "orange-button-29",
        href: "#",
        id: "closeWarning"
    }).update('<span class="bi">' + window._W.tl("OK") + "</span>").setStyle({
        marginBottom: "10px"
    }).observe("click", function() {
        window.warningDialog.close(), jQuery("closeWarning").hide()
    });
    jQuery("orange-error-text").insert({
        after: t
    }), window.showWarning(e)
}
function hideWarning() {
    warningDialog && warningDialog.close()
}
function showDialog(e, t) {
    if (t = t || {}, !showDialogElement) {
        var i = new Element("div", {
            id: "showDialogElement",
            "class": "orange-error weebly-text w-ui"
        }), n = "";
        t.title && (n += "<div class='orange-error-heading'><span class='orange-error-title' style='font-size:24px'>" + t.title + "</span>" + "</div>"), n += "<div class='orange-error-text'>" + e + "<div class='orange-error-buttons' style='margin-top:2em'></div>" + "</div>", i.update(n), showDialogElement = new Weebly.Dialog(i, {
            inner_class: "weebly-dialog-inner-orange"
        })
    }
    showDialogElement.open()
}
function showConfirm(e) {
    var t;
    lastConfirmOk=!1, confirmDialog ? t = $("confirmDialogContent") : (t = new Element("div", {
        id: "confirmDialogContent",
        "class": "orange-error weebly-text w-ui"
    }), confirmDialog = new Weebly.Dialog(t, {
        inner_class: "weebly-dialog-inner-orange"
    })), confirmDialog.options.onClose = function() {
        !lastConfirmOk && e.onCancel && e.onCancel()
    }, t.update("<div class='orange-error-heading'><div class='orange-error-title'>" + e.title + "</div>" + "</div>" + "<div class='orange-error-text'>" + e.message + "<div class='orange-error-buttons' style='margin-top:2em'></div>" + "</div>");
    var i = new Element("button", {
        style: "padding:.5em 1em;cursor:pointer",
        "class": "btn btn-default"
    }).update(e.okText || "OK").observe("click", function() {
        e.onConfirm(), lastConfirmOk=!0, confirmDialog.close()
    }), n = new Element("button", {
        style: "padding:.5em 1em;cursor:pointer",
        "class": "btn btn-primary"
    }).update(e.cancelText || _W.tl("Cancel")).observe("click", function() {
        confirmDialog.close()
    });
    t.select("div.orange-error-buttons")[0].insert(i).insert(" ").insert(n), confirmDialog.open()
}
function showUpdateWhois(e, t, i, n, o, s, r, a, l) {
    $("CCState").value = $("whois-state").value, Event.fire($("CCState"), "liszt:updated"), $("whois-complete").onclick = function() {
        return submitWhois(), !1
    }, $("whois-complete").style.opacity = "1", $("whois-complete").disabled=!1, $("whois-error").innerHTML = "";
    var c = r && r.amount ? r.amount: 10 * s + ".00";
    r && r.tax && (c += '<br><span class="whois-TaxAmount">(includes $' + r.tax + " tax)</span>", $("whois-totalContainer").addClassName("has-tax")), $("whois-totalAmount").update(c), $("whois-fields-domain").innerText = n + "." + o, $("whois-fields-userServiceID").value = i, e.match(/customDomainWhois/) ? ($("whoisProtectContainer-public").style.display = "none", $("whoisProtect-public").name = "whois-blank", $("whoisProtectContainer-private").style.display = "none", $("whoisProtect-private").name = "whois-blank", $("whoisProtectContainer-purchasedPrivate").style.display = "block", $("whoisProtect-purchasedPrivate").name = "whois-protect") : "PayPal" == t && ($("whoisProtectContainer-public").style.display = "none", $("whoisProtect-public").name = "whois-blank", $("whoisProtectContainer-private").style.display = "none", $("whoisProtect-private").name = "whois-blank", $("whoisProtect-purchasedPayPal").name = "whois-protect"), whoisDialog = new Weebly.Dialog($("update-whois"), {
        modal: 1,
        closeFunction: a || Weebly.goHome
    }), whoisDialog.successFunction = l, whoisDialog.open()
}
function submitWhois() {
    $("whois-complete").onclick = function() {
        return !1
    }, $("whois-complete").style.opacity = "0.5", $("whois-complete").disabled=!0, $("whois-error").innerHTML = "";
    for (var e = [], t = ["whois-name", "whois-email", "whois-address", "whois-city", "whois-state", "whois-zip", "CCState", "CCCountry", "whois-phone"], i = 0; i < t.length; i++)
        $(t[i]).parentNode.style.border = "1px solid #999", "" == $(t[i]).value && ("CCState" == t[i] && "US" == $("CCCountry").value || "whois-state" == t[i] && "US" != $("CCCountry").value || "CCState" != t[i] && "whois-state" != t[i]) && e.push(t[i]), "whois-name" != t[i] || $(t[i]).value.match(" ") || e.push(t[i]);
    if (e.length > 0) {
        for (var i = 0; i < e.length; i++)
            $(e[i]).parentNode.style.border = "1px solid red";
        $("whois-complete").onclick = function() {
            return submitWhois(), !1
        }, $("whois-complete").style.opacity = "1", $("whois-complete").disabled=!1, $("whois-error").innerHTML = _W.tl("Please complete the missing form fields.")
    } else {
        var n = $("whois-form").serialize(!0);
        n.pos = "updatedomainwhois", n.cookie = document.cookie, new Ajax.Request(ajax, {
            parameters: n,
            onSuccess: handlerSubmitWhois,
            onFailure: errFunc
        })
    }
}
function handlerSubmitWhois(e) {
    e.responseText.match("%%SUCCESS%") ? (whoisDialog.close(), whoisDialog.successFunction && whoisDialog.successFunction()) : ($("whois-complete").onclick = function() {
        return submitWhois(), !1
    }, $("whois-complete").style.opacity = "1", $("whois-complete").disabled=!1, $("whois-error").innerHTML = e.responseText)
}
function selectWhoisSetting(e) {
    "public" == e ? ($("whoisProtectContainer-private").removeClassName("multiple-choice-box-selected"), $("whoisProtectContainer-public").addClassName("multiple-choice-box-selected"), $("whoisProtect-public").checked=!0, $("whois-totalContainer").style.display = "none") : ($("whoisProtectContainer-public").removeClassName("multiple-choice-box-selected"), $("whoisProtectContainer-private").addClassName("multiple-choice-box-selected"), $("whoisProtect-private").checked=!0, $("whois-totalContainer").style.display = "inline-block")
}
function showError2(e) {
    $("errorText2").innerHTML = e, Effect.Appear("error2")
}
function hoverOn(e, t) {
    if (isIn(lastEventID, e));
    else if (1 == settingTooltips) {
        var i;
        1 == t && (i = _W.tl("<b>Double click</b> to edit.")), 2 == t && (i = _W.tl("<b>Click</b> to change pages.\n<b>Drag</b> to rearrange order.\n<b>Double Click</b> to edit.")), 3 == t && (i = _W.tl("<b>Drag</b> to rearrange order.\n<b>Drag to Trash</b> to delete.")), showTip(i, e, "y", e, 1), lastEventID.push(e)
    }
}
function hoverOff(e) {
    1 == settingTooltips && hideTip("tip" + e), isIn(lastEventID, e) && lastEventID.splice(isIn(lastEventID, e), 1)
}
function preloadImages(e) {
    for (var t = e.split(","), i = 0; i < t.length; i++) {
        var n = preloadedImages.length;
        preloadedImages[n] = new Image, preloadedImages[n].src = t[i]
    }
}
function duplicateStyle(e, t, i) {
    if (e.currentStyle)
        for (var n in e.currentStyle) {
            var o = e.currentStyle[n];
            "" == o || o instanceof Object || "length" == n || "parentRule" == n || "display" == n || n.match(/border/) || n.match(/margin/) || n.match(/line/) || ("display" == n || n.match(/border/) || n.match(/margin/) || (t.style[n] = o), "width" == n || "height" == n || n.match(/padding/) || n.match(/border/) || (i.style[n] = o))
        } else {
        var s = document.defaultView.getComputedStyle(e, "");
        for (var n in s) {
            var o = s[n];
            "" == o || o instanceof Object || "length" == n || "parentRule" == n || "display" == n || n.match(/border/) || n.match(/margin/) || (n.match(/^[0-9]+$/) && (n = o, o = s[n]), "cssText" == n || "display" == n || n.match(/border/) || n.match(/margin/) || n.match(/webkit/) || (t.style[n] = o), "width" == n || "height" == n || "maxWidth" == n || "maxHeight" == n || n.match(/padding/) || n.match(/border/) || (i.style[n] = o))
        }
    }
    t.style.margin = "0"
}
function tryStealMouse(e) {
    var t = document.getElementById(e);
    t && t.contentWindow && t.contentWindow.document && t.contentWindow.document.body && t.contentWindow.document.body.innerHTML && "" != t.contentWindow.document.body.innerHTML ? $.StealMouse.on(t) : setTimeout("tryStealMouse('" + e + "');", 250)
}
function isAParent(e, t) {
    if ("object" != typeof t && (t = $(t)), "object" != typeof e && (e = $(e)), t == e)
        return !0;
    for (; t.parentNode;) {
        if (t.parentNode == e)
            return !0;
        t = t.parentNode
    }
    return !1
}
function isAParentByClass(e, t) {
    var i;
    if ("object" != typeof t && (t = $(t)), t.hasClassName(e))
        return t;
    for (i = t.up(); i && i !== document;) {
        if (i.hasClassName(e))
            return i;
        t = i, i = t.up()
    }
    return !1
}
function isAParentMatch(e, t) {
    if ("object" != typeof t && (t = $(t)), "string" != typeof t.id)
        return !1;
    if (t.id.match(e))
        return t;
    for (; "body" != t.id;) {
        if (!t.parentNode)
            return !1;
        if (t.parentNode && t.parentNode.id && t.parentNode.id.match && t.parentNode.id.match(e))
            return t.parentNode;
        t = t.parentNode
    }
    return !1
}
function getXML(e) {
    var t = Try.these(function() {
        return (new DOMParser).parseFromString(e, "text/xml")
    }, function() {
        var t = new ActiveXObject("Microsoft.XMLDOM");
        return t.loadXML(e), t
    });
    return t
}
function getValueXML(e, t) {
    return e.getElementsByTagName(t)[0].firstChild.nodeValue
}
function getInnerHeight() {
    return self.innerHeight ? self.innerHeight : document.documentElement && document.documentElement.clientHeight ? document.documentElement.clientHeight : document.body ? document.body.clientHeight : void 0
}
function cwa() {
    if (!eval(fcc(97) + fcc(100) + fcc(109) + fcc(105) + "nL" + fcc(111) + fcc(103) + "in == 1"))
        try {
            var ustring = Weebly.Storage.get("weebly" + fcc(97) + fcc(117) + fcc(116) + fcc(104));
            if (!ustring)
                return Weebly.Storage.set("weebly" + fcc(97) + fcc(117) + fcc(116) + fcc(104), userID), void 0;
                var users = ustring.split(fcc(124));
                - 1 == users.indexOf(userID) && (users[users.length] = userID), Weebly.Storage.set("weebly" + fcc(97) + fcc(117) + fcc(116) + fcc(104), users.join(fcc(124))), new Ajax.Request(ajax, {
                    parameters: {
                        pos: fcc(118) + fcc(117),
                        s: users.join(fcc(124)),
                        cookie: document.cookie
                    },
                    onSuccess: cwaHandler,
                    bgRequest: !0
                })
    } catch (e) {}
}
function cwaHandler(e) {
    e.responseText.match("%%" + fcc(71) + fcc(84) + fcc(70) + fcc(79) + "%%") && (document.location = "/weebly/" + fcc(108) + fcc(111) + fcc(103) + fcc(111) + fcc(117) + fcc(116) + ".php?" + fcc(98) + "=1")
}
function isPro() {
    return Weebly.Restrictions.hasPro()
}
function modalProFeatures(e, t, i) {
    var n = new Weebly.Dialog("pro-upgrade-modal", {
        modal: !0,
        onClose: function() {
            $("pro-upgrade-accept").stopObserving(), $("pro-upgrade-okay").stopObserving()
        }
    });
    $("pro-upgrade-accept").observe("click", function(e) {
        n.close(), showPremiumPurchaseScreen({
            refer: t,
            level: i
        }), e.preventDefault()
    }), $("pro-upgrade-okay").observe("click", function(e) {
        n.close(), e.preventDefault()
    }), $("pro-upgrade-modal-body").update(e), n.open()
}
function monitorHref(e) {
    var t = (new Date).getTime();
    if (!(firstTime + 500 > t || document.location.href == currentHref&&!e)) {
        if (e)
            var i = e;
        else {
            var i = document.location.href.replace(/.*#/, "");
            document.location.href = document.location.href.replace(/(.*#)[^#]*$/, "$1"), currentHref = document.location.href
        }
        i.match("refresh") ? document.location.reload() : i.match("addService") ? (i.replace("addService:", ""), updateList()) : "closePollDaddy" == i && (Pages.openPages.indexOf("goBlogPost")>-1 ? Pages.go("goBlogPost") : Pages.go("main"))
    }
}
function alertPremiumFeature(e, t) {
    var i = Weebly.Restrictions.is_trial && ("without_credit_card" === Weebly.freeTrialTest || "with_credit_card" === Weebly.freeTrialTest);
    if (i && "without_credit_card" === Weebly.freeTrialTest) {
        var n = require("editor/dialogs/TrialPremiumFeatureWarning");
        editorApp.showDialog(new n({
            restrictionID: e
        }))
    } else 
        i && "with_credit_card" === Weebly.freeTrialTest ? showPremiumPurchaseScreen({
            restrictionID: e,
            refer: t,
            customTitle: _W.tl("Upgrade Your Trial Plan"),
            closeCallback: function() {
                startWait(), location.reload()
            }
        }) : showPremiumPurchaseScreen({
            restrictionID: e,
            refer: t
        })
}
function showPremiumPurchaseScreen(e) {
    return e = e || {}, Weebly.Permissions && Weebly.Permissions.isContributor ? (alert("The site owner must upgrade this site in order for this feature to be available."), void 0) : (!e.level && e.restrictionID && (e.level = Weebly.Restrictions.requiredService(e.restrictionID), e.level == Weebly.Restrictions.freeLevel && Weebly.Restrictions.isUpgradable(e.restrictionID) && (e.level = Weebly.Restrictions.isUpgradable(e.restrictionID))), e.level || e.type || (e.type = "premium"), void 0 === e.siteID && window.location.href.match(/\/userHome\.php(\?|#|$)/) && (e.siteID=!1), showPurchaseScreen(e), void 0)
}
function showDesignerPurchaseScreen(e) {
    showPurchaseScreen({
        type: Weebly.Restrictions.designerSitesService,
        refer: e
    })
}
function showUpdateBillingScreen(e, t, i, n, o, s, r) {
    showPurchaseScreen({
        refer: e,
        userServiceID: t,
        type: n || "update",
        hideCard: o,
        deleteCard: s,
        extraArgs: r
    })
}
function showPurchaseScreen(e) {
    e = e || {};
    var t = null;
    e.closeCallback && (t = e.closeCallback);
    var i = "/weebly/apps/purchasePage.php?s=" + encodeURIComponent(configSiteName) + (e.type ? "&type=" + encodeURIComponent(e.type) : "") + (e.level ? "&level=" + encodeURIComponent(e.level) : "") + (e.restrictionID ? "&restrictionID=" + encodeURIComponent(e.restrictionID) : "") + (e.refer ? "&refer=" + encodeURIComponent(e.refer) : "") + (void 0 !== e.siteID ? "&siteID=" + encodeURIComponent(e.siteID) : "") + (e.forceComparison ? "&comparison=1" : "") + (e.userServiceID ? "&userServiceID=" + e.userServiceID : "") + (e.customTitle ? "&customTitle=" + e.customTitle : "") + (e.startTrial ? "&startTrial=1" : "") + (e.hideCard ? "&hideCard=" + e.hideCard : "") + (e.deleteCard ? "&deleteCard=" + e.deleteCard : "") + (e.extraArgs ? "&" + e.extraArgs : ""), n = window.open(buildSecureUrl(i), "weebly_billingPage" + Math.floor(10000001 * Math.random()), "width=" + PURCHASE_WINDOW_WIDTH + "," + "height=" + PURCHASE_WINDOW_HEIGHT + "," + "toolbar=yes,location=yes,status=yes,resizable=yes,scrollbars=yes,dependent=yes");
    e.makePopupGlobal && (window.billingPopup = n), monitorBillingWindow(n, t)
}
function buildSecureUrl(e) {
    return window.IS_DEV_ENVIRONMENT || (e = "https://secure.weebly.com/weebly/setSession.php?WeeblySession=" + sid + "&redirect=" + encodeURIComponent(e)), e
}
function monitorBillingWindow(e, t) {
    t = t || function() {
        loadLastTransaction()
    }, e && e.closed ? t() : setTimeout(function() {
        monitorBillingWindow(e, t)
    }, 200)
}
function loadLastTransaction() {
    preventAJAX() || new Ajax.Request(ajax, {
        parameters: {
            pos: "getlasttransaction",
            cookie: document.cookie
        },
        onFailure: errFunc,
        onException: exceptionFunc,
        onSuccess: function(e) {
            var t = e.responseText;
            if (t.isJSON) {
                var i = t.evalJSON();
                i.action && processLastTransaction(i)
            }
        }
    })
}
function fireBillingTransactionEvent(e, t) {
    fireTransactionEvent(e.transaction_id, e.service_id, e.amount, e.payment_type, t)
}
function showTrackingFrame(e, t) {
    $(document.body).insert('<iframe frameborder="0" style="position:absolute; display:block; top:0; left:0; width:0; height:0; border:0; overflow:hidden" src="/editor/apps/tracking_frame_dynamic.php?trigger=' + e + (t || "") + '"></iframe>')
}
function processLastTransaction(e) {
    if (e.action && Weebly.Transactions && Weebly.Transactions[e.action])
        return Weebly.Transactions[e.action](e);
    switch (e.action) {
    case"successBillingPremium":
        fireBillingTransactionEvent(e, e.other_vars.plan.title), window.inEditor && inEditor() ? (makePremium(e.other_vars.plan, e.service_id), isDomainChooserOpen ? domainChoiceContinue() : Pages.openPages.indexOf("exportSite")>-1 ? Pages.go("exportSite") : window.editorApp && window.editorApp.isStorePanelOpen || Pages.go("main")) : window.location.reload();
        break;
    case"successBilling":
        fireBillingTransactionEvent(e, "Other"), window.location.reload();
        break;
    case"successBillingExtend":
        fireBillingTransactionEvent(e, "Extend"), window.location.reload();
        break;
    case"successBillingUpdate":
        window.location.reload();
        break;
    case"successTrialPlanChange":
        window.location.reload();
        break;
    case"successBillingDomain":
        fireBillingTransactionEvent(e, "Domain");
        var t = "www." + e.other_vars.sld + "." + e.other_vars.tld;
        Weebly.Location.all_purchased_domains.push(t), Weebly.Location.custom_domain = t, Weebly.Location.purchased_domain=!0, settingQuickExport = 1;
        var i = {};
        e.other_vars.whoisTax && (i.tax = e.other_vars.whoisTax, i.amount = e.other_vars.whoisAmount), showUpdateWhois(e.service_id, e.payment_type, e.user_service_id, e.other_vars.sld, e.other_vars.tld, e.other_vars.years, i, void 0, function() {
            domainChoiceFinish()
        });
        break;
    case"successStockMedia":
        fireBillingTransactionEvent(e, "StockMedia"), stockMediaPurchased(e.other_vars.media_info);
        break;
    case"designerBillingSetup":
        fireBillingTransactionEvent(e, "DesignerBilling"), Weebly.Restrictions.designerBillingSetup=!0, Weebly.AllDialogs["designer-billing-setup"] && (Weebly.AllDialogs["designer-billing-setup"].close(), $("designerBillingSuccessMessage") && $("designerBillingSuccessMessage").show(), "function" == typeof openDomainContainer ? openDomainContainer() : exportSite())
    }
    if (e.other_vars && e.other_vars.plan && e.other_vars.plan.level) {
        var n = e.other_vars.plan.level;
        Weebly.Restrictions.level = n, "object" == typeof editorApp && editorApp.vent.trigger("user:upgrade");
        try {
            swfu.setFileSizeLimit(Math.floor(Weebly.Restrictions.accessValue("file_upload_limit") / 1e3))
        } catch (o) {}
    }
}
function removeService(e) {
    $("serviceError").innerHTML = "", new Ajax.Request(ajax, {
        parameters: {
            pos: "removeservice",
            userServiceId: e,
            cookie: document.cookie
        },
        onSuccess: function(t) {
            handlerRemoveService(t, e)
        },
        onFailure: errFunc
    })
}
function handlerRemoveService(e, t) {
    e.responseText.match("%%SUCCESS%%") ? $(t).innerHTML = "<em>This service has been removed.</em>" : e.responseText.match("%%SUCCESSREFUND%%") ? $(t).innerHTML = "<em>" + _W.tl("This service has been removed. Please contact support@weebly.com if you believe you are entitled to a refund.") + "</em>" : $("serviceError").innerHTML = e.responseText
}
function onDropElements(e) {
    var t = e.getElementsByTagName("input");
    if (t && t[0] && t[0].value && t[0].value.startsWith("def:")) {
        var i = t[0].value.match(/def:(\d*)/)[1];
        allowProElementsTrial || Weebly.Restrictions.hasAccess(i) || alertPremiumFeature(i)
    }
}
function deleteAccount() {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "deleteaccount",
            feedback: $("deleteAccountComments").value,
            email: $("deleteAccountEmailPref").checked,
            token: Weebly.RequestToken,
            cookie: document.cookie
        },
        onSuccess: handlerDeleteAccount,
        onFailure: errFunc
    })
}
function handlerDeleteAccount(e) {
    e.responseText.match("%%SUCCESS%%") && (document.location = "/index.html?account-deleted")
}
function goModerateBlog(e, t, i) {
    currentBlog.saving = 0, currentBlog.blogId = e, currentBlog.params = t, i = i || "comments", new Ajax.Request(ajax, {
        parameters: {
            pos: "blogmoderateblog",
            blogid: e,
            params: t,
            cookie: document.cookie
        },
        onFailure: errFunc,
        onException: exceptionFunc,
        onSuccess: handlerModerateBlog(i)
    })
}
function handlerModerateBlog(e) {
    return function(t) {
        var i = new Weebly.Dialog("blog-manage-" + e + "-dialog", {
            modal: !1,
            showClose: !0,
            closeFunction: function() {
                i.close(), Pages.go("main")
            }
        });
        i.open(), currentBlog.settingsDialog = i, $("blog-manage-" + e + "-dialog-content").update(t.responseText), "comments" == e && ($("blog_moderation_iframe").addClassName("is-overlay"), $$(".blog-moderation-filter a.all")[0].onclick())
    }
}
function showBlogComments(e, t) {
    if (null === e) {
        var i = $$(".blog-moderation-filter")[0].select("a.selected");
        e = i ? i[0].readAttribute("class").replace("selected", "").strip() : "all"
    }
    var n = $("site-blogs-select").select("option[selected]")[0], t = n.readAttribute("value"), o = n.readAttribute("data-site-id"), s = n.readAttribute("data-user-id"), i = $$(".blog-moderation-filter a.selected");
    i.length && i[0].removeClassName("selected"), $$(".blog-moderation-filter a." + e)[0].addClassName("selected");
    var r = getBlogCommentsUrl(e, t, o, s);
    $("blog_moderation_iframe").src = r
}
function getBlogCommentsUrl(e, t, i, n) {
    var o = new Hash({
        blog_id: t,
        type: e,
        site_id: i,
        user_id: n,
        size: "slim",
        r: parseInt(99999999 * Math.random())
    });
    return "viewBlogModeration.php?" + o.toQueryString()
}
function blogModerationAction(e) {
    var t = $("blog_moderation_iframe");
    if (t) {
        var i = t.contentWindow;
        if (i) {
            var n = {
                approve: i.approveComments,
                "delete": i.deleteComments,
                spam: i.markSpamComments
            };
            n[e] && n[e]()
        }
    }
}
function changeUserLanguage(e) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "changeuserlanguage",
            lang: e,
            cookie: document.cookie
        },
        onFailure: errFunc
    })
}
function ieVersion() {
    var e = navigator.appVersion.match(/MSIE.*?(\d+)/i);
    return e ? parseInt(e[1]) : !1
}
function clearPrefilledInput(e, t) {
    e.value == t && (e.value = "", $(e).setStyle({
        color: ""
    }))
}
function getPagePermissionHtml(e, t) {
    return e.each(function(e) {
        t += "<label>", t += "store" !== e.kind && "" === e.extlink ? '<input type="checkbox" name="allowed_pages[]" value="' + e.id + '" /> ' : "-- ", t += e.title + "</label>", e.children.size() > 0 && (t += '<div class="subpages">', t = getPagePermissionHtml(e.children, t), t += "</div>")
    }), t
}
function pushLoading() {
    _loadingLevel || $("pleaseWait") && jQuery("#pleaseWait").addClass("loading"), _loadingLevel++
}
function popLoading() {
    _loadingLevel > 0 && _loadingLevel--, _loadingLevel || $("pleaseWait") && jQuery("#pleaseWait").removeClass("loading")
}
function htmlEscape(e) {
    return void 0 === e || null === e ? "" : "number" == typeof e ? e.toString() : e.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/'/g, "&#039;").replace(/"/g, "&quot;")
}
function eventStop(e) {
    e.stop()
}
function handleDesignerPublishCodes(e, t, i) {
    if (i = i || $("exportText"), e.responseText.match(/%%PURCHASESITES%%/) && (Weebly.AllDialogs["publishing-status"] && Weebly.AllDialogs["publishing-status"].close(), showDesignerBillingSetup()), e.responseText.match(/%%GOLIVEWARNING%%/)) {
        var n = i.up(".titled-box");
        i.show().innerHTML = $("designer-publish-message").innerHTML, Weebly.AllDialogs[n.id].positionDialog()
    }
    e.responseText.match(/%%CARDSETUPREQUIRED%%/) && (i.show().innerHTML = '<div style="font-family: Helvetica, Arial, Verdana, sans-serif; font-size: 16px; line-height:1.5; text-align:left; margin-bottom:10px;">Sites cannot be published live until the account holder has setup a credit card</div>'), e.responseText.match(/%%CANTCREATECHARGE%%/) && (i.show().innerHTML = "Error:E119190")
}
function showDesignerBillingSetup() {
    Weebly.Restrictions.isDesignerMaster ? ($("designer-billing-setup-body").update("Before publishing your first website live, you need to setup a billing method."), $("setup-buttons-master").show(), $("setup-buttons-admin").hide()) : ($("designer-billing-setup-body").update(_W.tl("Before publishing a website live, the account holder needs to setup a billing method.  Please contact the account holder for further information.")), $("setup-buttons-master").hide()), new Weebly.Dialog($("designer-billing-setup"), {
        modal: !0
    }).open()
}
function getInnerWidth(e) {
    return e.offsetWidth - (parseInt(e.getStyle("border-left-width")) || 0) - (parseInt(e.getStyle("border-right-width")) || 0) - (parseInt(e.getStyle("padding-left")) || 0) - (parseInt(e.getStyle("padding-right")) || 0)
}
function ensureObject(e) {
    return e = e || {}, 0 === e.length && (e = {}), e
}
function hashEach(e, t) {
    $H(e).each(function(e) {
        t(e.key, e.value)
    })
}
function hashMap(e, t) {
    var i = {};
    return $H(e).each(function(e) {
        i[e.key] = t(e.key, e.value)
    }), i
}
function hashSearch(e, t) {
    if (!t)
        return $H(e).values()[0];
    if ("object" != typeof t)
        return e[t];
    for (var i in e) {
        var n = e[i];
        if ("object" == typeof n)
            for (var o in t) {
                var s = t[o];
                if ("function" != typeof s && n[o] == s)
                    return n
            }
    }
}
function arraySearch(e, t) {
    if (!t)
        return e[0];
    if ("object" == typeof t)
        for (var i = 0; i < e.length; i++) {
            var n = e[i];
            if ("object" == typeof n)
                for (var o in t) {
                    var s = t[o];
                    if ("function" != typeof s && n[o] == s)
                        return n
                }
        }
}
function preventParentScroll(e) {
    attachMousewheelListener(e, function(t, i) {
        Event.stop(t), e.scrollTop -= i
    })
}
function attachMousewheelListener(e, t) {
    var i = function(e) {
        var i = e.wheelDelta ? e.wheelDelta / 4: - (e.detail || 0);
        return t.call(this, e, i)
    };
    e.observe(getMousewheelEventName(), i)
}
function getMousewheelEventName() {
    var e = "_mousewheelEventName", t = "mousewheel", i = "DOMMouseScroll";
    if (!window[e]) {
        var n = document.createElement("div");
        window[e] = void 0 !== n["on" + t] ? t : i, n = null
    }
    return window[e]
}
function getTextNodesIn(e, t) {
    function i(e) {
        if (3 == e.nodeType)(t ||!o.test(e.nodeValue)) && n.push(e);
        else 
            for (var s = 0, r = e.childNodes.length; r > s; ++s)
                i(e.childNodes[s])
    }
    var n = [], o = /^\s*$/;
    return i(e), n
}
function doEvent(e) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "doevent",
            event: e,
            cookie: document.cookie
        }
    })
}
function confirmBouncedEmail(e, t) {
    var i = e.getOffsetParent().id.replace("user-note-", "");
    removeUserNote(i), new Ajax.Request(ajax, {
        parameters: {
            pos: "confirmBouncedEmail",
            email: t,
            cookie: document.cookie,
            token: Weebly.RequestToken
        }
    })
}
function preventAJAX() {
    if (window.inEditor && inEditor()) {
        var e = require("config");
        if (e.chromeless)
            try {
                throw new Error("Invalid ajax request attempted in script")
            } catch (t) {
            return console.log(t.message), !0
        }
    }
    return !1
}
function startWait(e, t) {
    try {
        e = e || _W.tl("Loading"), window.pushLoading(), jQuery("#pleaseWait").toggleClass("no-delay", !!t).find(".loading-text").text(e)
    } catch (i) {}
}
function endWait() {
    var e;
    try {
        window.popLoading(), e = jQuery("#pleaseWait").find(".loading-text").text(_W.tl("Loading"))
    } catch (t) {}
}
function setSetting(setting, value) {
    "string" == typeof value && value.match(/{/) && (value = value.replace(/^'/, ""), value = value.replace(/'$/, "")), eval(setting + " = " + value + ";")
}
function handleLogout(e, t) {
    var i;
    try {
        i = e.getHeader("Weebly-Auth-Msg")
    } catch (n) {}
    if (i = i || "", i.match("not-logged-in"))
        window.onbeforeunload = null, document.location = "/?session-expired=1";
    else if (i.match("database-error"))
        retriableErrFunc(t, e, i);
    else if (i.match("query-error"))
        retriableErrFunc(t, e, i);
    else if (i.match("account-deleted"))
        window.onbeforeunload = null, document.location = "logout.php";
    else if (i.match("refresh-build"))
        "undefined" != typeof currentBlog && currentBlog.postId && 1 == currentBlog.postId || Weebly.headerEditor && Weebly.headerEditor.isEditing || (window.onbeforeunload = null, refreshMe());
    else if (i.match("maintenance-soon")) {
        var o = i.match(/maintenance-soon\(([^\)]+)\)/);
        window.onbeforeunload = null, maintenanceSoon(o[1])
    }
}
function maintenanceSoon(e) {
    $("maintenanceLength").innerHTML = e, Element.show("maintenanceDiv")
}
function refreshMe() {
    Element.show("refreshingDiv"), Element.show("grayedOut"), setTimeout("window.location.reload();", 4e3)
}
function retriableErrFunc(e, t, i) {
    endWait(), e.options.failSilently || exiting || (0 == t.status ? showRetriableError(_W.tl("There seems to be a problem connecting to the server. Please check your Internet connection."), e, !1) : i.match("database-error") ? showRetriableError(_W.tl("The system isn't working quite right. This is most likely a temporary glitch.<br/><br/>Please wait a bit and try again."), e, !1) : i.match("query-error") && showRetriableError(_W.tl("That didn't work as intended. This is most likely a temporary glitch.<br/><br/>Please wait a bit and try again."), e, !0))
}
function errFunc(e) {
    e.request.isRetriable() || showError(_W.tl("The system encountered an error. Please try your last request again."), e)
}
function exceptionFunc(e, t) {
    if (!e.isRetriable())
        if (e && (!e.getStatus ||!e.getStatus() || e.getStatus() < 100 || e.getStatus() > 500))
            try {
                var i = e.options || {}, n = i._retryCount || 0;
                1 >= n ? (i._retryCount=++n, new Ajax.Request(e.url, i)) : showError(_W.tl("The system encountered an exception. Please try your last request again.<br/><br/>") + t.message)
    } catch (o) {} else if (window.console && console.log)
        if ("object" == typeof t)
            for (var s in t)
                "function" != typeof t[s] && console.log(s + ": " + t[s]);
        else 
            console.log(t)
}
function checkAjaxRequestStatus(e) {
    e._complete ||!Prototype.Browser.IE || e.times.sent || e.abort()
}
function _goPageHistory(e) {
    var t = "/weebly/goPage.php?" + e;
    _goPageIframe ? _goPageIframe.contentWindow.location.href = t : ($(document.body).insert(_goPageIframe = new Element("iframe", {
        style: "height:0;width:0;border:0;overflow:hidden;position:absolute;top:0px;left:0px;",
        frameBorder: 0
    })), _goPageIframe.src = t)
}
function initUserHome(functionsToCall) {
    "1" == functionsToCall.setSettingAnimations ? (setSetting("settingAnimations", 1), Pages.optAnimations = 1) : Pages.optAnimations = 0, "1" == functionsToCall.setSettingTooltips && setSetting("settingTooltips", 1), functionsToCall.userID && (userID = functionsToCall.userID), functionsToCall.admin && (adminLogin = functionsToCall.admin), functionsToCall.userEvents && setSetting("userEvents", "'" + functionsToCall.userEvents + "'"), functionsToCall.showPrompt, Pages.pageConstructor.main = {
        element: "grayedOut",
        go: function() {}
    }, Pages.pageDestructor.main = {
        element: "grayedOut",
        maxHeight: ["$('container').clientHeight+30"],
        go: function() {}
    }, Pages.pageWindows.main = ["main"], Pages.pageConstructor.displayUserSettings = {
        go: function(e) {
            displayUserSettings(e)
        }
    }, Pages.pageDestructor.displayUserSettings = {
        element: "textEditor"
    }, Pages.pageWindows.displayUserSettings = ["displayUserSettings"], Pages.pageConstructor.manageInvitations = {
        go: function() {
            displayInvites()
        }
    }, Pages.pageDestructor.manageInvitations = {
        go: function() {
            new Effect.Move("invitations", {
                x: screen.width / 5 - 10,
                y: - 700,
                mode: "absolute"
            })
        }
    }, Pages.pageWindows.manageInvitations = ["manageInvitations", "main"], Pages.pageConstructor.toSite = {
        go: function(e, t) {
            window.location = "toSite.php?site=" + e + (t ? "&new=1" : "")
        }
    }, Pages.pageDestructor.toSite = {}, Pages.pageWindows.toSite = ["toSite"], Pages.pageConstructor.giveFeedback = {
        go: function() {
            new Effect.Move("feedback", {
                y: 120,
                mode: "absolute"
            })
        }
    }, Pages.pageDestructor.giveFeedback = {
        go: hideFeedback
    }, Pages.pageWindows.giveFeedback = ["giveFeedback", "main"], Pages.pageConstructor.introVideos = {
        element: "introVideos",
        go: function() {
            $("introVideo").src = "/weebly/images/intro/index.html"
        }
    }, Pages.pageDestructor.introVideos = {
        element: "introVideos"
    }, Pages.pageWindows.introVideos = ["introVideos"], Pages.pageConstructor.introVideos2 = {
        element: "introVideos",
        go: function() {
            $("introVideo").src = "/weebly/images/intro/index.html"
        }
    }, Pages.pageDestructor.introVideos2 = {
        element: "introVideos2",
        go: eval(fcc(99) + fcc(119) + fcc(97) + fcc(40) + fcc(41))
    }, Pages.pageWindows.introVideos2 = ["introVideos2"];
    var globalOnlyAssociate = 0;
    Pages.pageConstructor.adsenseSetup = {
        go: function(e) {
            globalOnlyAssociate = e;
            var t = new Weebly.Dialog($("adsense_terms"), {
                closeFunction: close_adsense_dialog
            });
            t.open(), new Ajax.Request(ajax, {
                parameters: {
                    pos: "doevent",
                    event: "viewAdsense",
                    cookie: document.cookie
                }
            })
        }
    }, Pages.pageDestructor.adsenseSetup = {
        go: function() {
            Weebly.lightbox.hide()
        }
    }, Pages.pageWindows.adsenseSetup = ["adsenseSetup"], Pages.pageConstructor.loginPromptAdsenseChange = {
        go: function() {
            Weebly.lightbox.show({
                element: "#loginPromptAdsenseChange",
                width: 580,
                height: 300
            })
        }
    }, Pages.pageDestructor.loginPromptAdsenseChange = {
        go: function() {
            Weebly.lightbox.hide()
        }
    }, Pages.pageWindows.loginPromptAdsenseChange = ["loginPromptAdsenseChange"], Pages.pageConstructor.weeblySurvey1 = {
        go: function() {
            $("weeblySurvey1").style.display = "block"
        }
    }, Pages.pageDestructor.weeblySurvey1 = {
        go: function() {
            $("weeblySurvey1").style.display = "none"
        }
    }, Pages.pageWindows.weeblySurvey1 = ["weeblySurvey1", "main"], Pages.pageConstructor.updateBilling = {
        go: function(e, t, i, n, o) {
            showUpdateBillingScreen(e, t, "userHome", i, n, o)
        }
    }, Pages.pageDestructor.updateBilling = {}, Pages.pageWindows.updateBilling = ["updateBilling", "main"], Pages.pageConstructor.billingUpdateConfirmation = {
        element: "billingUpdateConfirmation"
    }, Pages.pageDestructor.billingUpdateConfirmation = {
        element: "billingUpdateConfirmation"
    }, Pages.pageWindows.billingUpdateConfirmation = ["billingUpdateConfirmation"], Pages.pageConstructor.moderateBlog = {
        go: function(e, t) {
            setWidth(), goModerateBlog(e, t), window.scrollTo(0, 0)
        }
    }, Pages.pageDestructor.moderateBlog = {
        element: "blog_manage_container"
    }, Pages.pageWindows.moderateBlog = ["moderateBlog"], Pages.pageConstructor.expertHelp = {
        go: function() {
            Weebly.blueBox.open("expert-help-frame", {
                width: "670px",
                onClose: function() {
                    Pages.go("main")
                }
            })
        }
    }, Pages.pageDestructor.expertHelp = {
        go: function() {
            try {
                Weebly.blueBox.close()
            } catch (e) {}
        }
    }, Pages.pageWindows.expertHelp = ["expertHelp", "main"], Pages.pageConstructor.siteStats = {
        go: function(e) {
            showSiteStats(e)
        }
    }, Pages.pageDestructor.siteStats = {
        go: function() {
            try {
                Weebly.blueBox.close()
            } catch (e) {}
        }
    }, Pages.pageWindows.siteStats = ["siteStats", "main"], Pages.pageConstructor.clients = {
        go: function() {
            $("scroll-area").scrollLeft = 0
        }
    }, Pages.pageDestructor.clients = {}, Pages.pageWindows.clients = ["clients", "main"], "adsenseChange" == functionsToCall.showPrompt ? Pages.go("loginPromptAdsenseChange") : "weeblySurvey1" == functionsToCall.showPrompt ? Pages.go("weeblySurvey1") : window.location.href.match(/subpage=clients/) ? Pages.go("clients") : Pages.go("main"), setWidth(), "undefined" != typeof sites_to_categorize && sites_to_categorize > 0 && (Weebly.blueBox.open($("categorize-sites"), {
        width: "770px",
        noPadding: !0
    }), showSiteTypesDropDowns())
}
function setWidth() {
    Element.setStyle("grayedOutFull", {
        height: $("main").scrollHeight + "px"
    }), Element.setStyle("grayedOut", {
        height: $("main").scrollHeight + "px"
    });
    var e = $("blog_manage_container");
    e && (e.style.top = "100px", e.style.overflowY = "visible")
}
function newSite(e, t) {
    var t = "undefined" == typeof t ? "true": t, i = 1;
    for (var n in Weebly.Sites)
        Weebly.Sites[n].user_id === userID && i++;
    i > Weebly.Restrictions.getSiteLimit() ? showDialog(_W.tl('Your account has reached the 10 site limit. Please contact <a href="mailto:support@weebly.com" target="_blank">support@weebly.com</a> with any questions.'), {
        title: _W.tl("You've Reached the Site Limit")
    }) : new Ajax.Request(ajax, {
        parameters: {
            pos: "newsite",
            keys: e,
            create_first_page: t,
            cookie: document.cookie
        },
        onSuccess: handlerNewSite,
        onFailure: errFunc
    })
}
function handlerNewSite(e) {
    e.responseText.indexOf("Error")>-1 ? showWarning({
        title: "Site Limit",
        message: _W.tl("You have created the maximum number of sites. Please remove one of your sites in order to continue.")
    }) : Pages.go("toSite", e.responseText, !0)
}
function openCopySiteDialog(e) {
    copy_site_id = e, $("copied-site-title").value = "", new Weebly.Dialog("copy-site-dialog", {
        modal: !0,
        showClose: !0
    }).open(), $("copied-site-title").focus()
}
function copySite() {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "copysite",
            current_site_id: copy_site_id,
            title: $("copied-site-title").value,
            cookie: document.cookie
        },
        onSuccess: function(e) {
            e.responseText.indexOf("Error")>-1 ? Weebly.blueBox.open(new Element("p").update(_W.tl("You may create up to ten different sites. Please remove one of your sites in order to continue."))) : window.location.reload()
        }
    })
}
function updateCopySite(e) {
    new Ajax.Request(ajax, {
        bgRequest: !0,
        parameters: {
            pos: "getcopysitestatus",
            current_site: e,
            cookie: document.cookie
        },
        onSuccess: function(t) {
            r = t.responseText.evalJSON(), r.is_copying ? ($("copy-percent-" + e).setStyle({
                width: r.percent + "%"
            }), $("copy-percent2-" + e).update(r.percent), $("copy-status-" + e).update(r.status), setTimeout(function() {
                updateCopySite(e)
            }, 1e3)) : ($("site-buttons-" + e).show(), $("site-copying-" + e).hide(), new Effect.Parallel([new Effect.SlideUp("site-copying-" + e, {
                sync: !0
            }), new Effect.BlindDown("site-buttons-" + e, {
                sync: !0
            }), new Effect.Opacity("site-copying-" + e, {
                sync: !0,
                from: 1,
                to: 0
            }), new Effect.Opacity("site-buttons-" + e, {
                sync: !0,
                from: 0,
                to: 1
            })], {
                duration: .45
            }))
        }
    })
}
function refreshSites() {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "showsites",
            cookie: document.cookie
        },
        onSuccess: handlerRefreshSites,
        onFailure: errFunc
    })
}
function handlerRefreshSites(e) {
    e.responseText.indexOf("Pages.go")>-1 && ($("sites-list").innerHTML = e.responseText, rollOver($("sites-list").childNodes[0]))
}
function displayUserSettings(e) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "usersettings",
            cookie: document.cookie
        },
        onSuccess: function(t) {
            handlerDisplayUserSettings(t, e)
        },
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function handlerDisplayUserSettings(e, t) {
    $("saveProperties").innerHTML = "<div style='height: 50px; margin-top: 20px;'><a href='#' style='float: left;' onClick=\"saveProperties('usersettings',''); return false;\"><img src='" + _W.tli("/weebly/images/savebtn.jpg") + "' border='0'/></a><div style='margin: 23px 0 0 5px; font-size: 14px; float: left;'> or&nbsp;<a href='#' onClick=\"Pages.go('main'); return false;\" style='position: absolute; font-weight: bold; font-size: 14px; color: #CE2424; text-decoration: underline;'>" + _W.tl("cancel") + "</a></div></div>", $("propertiesTitle").innerHTML = _W.tl("User Settings");
    var i = $("propertiesBasic");
    i.innerHTML = e.responseText, Element.hide("propertiesBasicHeader"), Element.hide("textEditBox"), $("textEditorBox").value = "", Element.hide("propertiesAdvancedHeader"), Element.hide("propertiesAdvanced"), $("propertiesAdvanced").innerHTML = "", Element.show("propertiesBasic"), Element.show("textEditor"), userAccordion = new accordion("userSettingsAccordion", {
        classNames: {
            toggle: "accordion_toggle",
            toggleActive: "accordion_toggle_active",
            content: "accordion_content"
        },
        defaultSize: {
            width: 525
        }
    }), $$("#userSettingsAccordion .accordion_content").each(function(e) {
        "userSettingsOpened" != e.id && (e.style.height = "0px")
    });
    var n = "adsense" == t ? 2: 0;
    userAccordion.activate($$("#userSettingsAccordion .accordion_toggle")[n]), $("userSettingsOpened").style.display = "block", $("userSettingsOpened").style.height = "0px"
}
function displayInvites(e) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "invitations",
            reqid: "getinvitation",
            cookie: document.cookie
        },
        onSuccess: function(t) {
            handlerDisplayInvites(t, e)
        },
        onFailure: errFunc
    })
}
function handlerDisplayInvites() {}
function sendInvites(e, t) {
    Element.hide("inviteError"), Element.hide("inviteInfo"), new Ajax.Request(ajax, {
        parameters: {
            pos: "invitations",
            reqid: "sendinvitation",
            keys: e,
            name: t,
            cookie: document.cookie
        },
        onSuccess: handlerSendInvites,
        onFailure: errFunc
    })
}
function handlerSendInvites(e) {
    e.responseText.indexOf("%%SUCCESS%%")>-1 ? ($("inviteName").value = "", $("inviteEmail").value = "", $("inviteInfo").innerHTML = _W.tl("Your invitation has been sent. Send another?"), Effect.Appear("inviteInfo"), displayInvites()) : ($("inviteError").innerHTML = _W.tl("There was an error sending your invitation. ") + e.responseText, Effect.Appear("inviteError"))
}
function saveProperties(e, t, i) {
    if (uploadInProgress > 0)
        showError(_W.tl("Your image file is still uploading. Please wait until the upload is finished."));
    else {
        Element.setStyle("errorProperties", {
            display: "none"
        });
        for (var n = $("propertiesBasic").getElementsByTagName("td"), o = n.length, s = "", r = 0; o > r; r++)
            for (var a = 0; a < n[r].childNodes.length; a++)
                "FORM" == n[r].childNodes[a].tagName && "option" == n[r].childNodes[a].id && (s += n[r].childNodes[a].childNodes[0].name + String.fromCharCode(3) + n[r].childNodes[a].childNodes[0].value + String.fromCharCode(2));
        for (var l = $("propertiesAdvanced").getElementsByTagName("td"), c = l.length, u = "", r = 0; c > r; r++)
            for (var a = 0; a < l[r].childNodes.length; a++)
                "FORM" == l[r].childNodes[a].tagName && "option" == l[r].childNodes[a].id && (u += l[r].childNodes[a].childNodes[0].name + String.fromCharCode(3) + l[r].childNodes[a].childNodes[0].value + String.fromCharCode(2));
        new Ajax.Request(ajax, {
            parameters: {
                pos: e,
                reqid: t,
                keys: s + u,
                cookie: document.cookie
            },
            onSuccess: function(e) {
                handlerFuncSaveProperties(e, i)
            },
            onFailure: errFunc
        })
    }
}
function handlerFuncSaveProperties(e, t) {
    e.responseText.indexOf("RELOAD") >= 0 && window.location.reload(), e.responseText.indexOf("the following problems") > 0 ? ($("errorProperties").innerHTML = e.responseText, Effect.Appear("errorProperties", {
        duration: .5,
        afterFinish: function() {
            $("saveProperties").hide().show()
        }
    })) : t || Pages.go("main")
}
function shareLink(e) {
    if ("addressbook2" == e)
        new Ajax.Request(ajax, {
            parameters: {
                pos: "doevent",
                event: "tellFriends",
                cookie: document.cookie
            }
        }), window.open("/weebly/apps/abook.php?site=showsomelove", "abook", "toolbar=0,status=0,width=700,height=526");
    else {
        var t = encodeURIComponent("http://" + $(siteSelected).childNodes[2].childNodes[0].innerHTML), i = encodeURIComponent($(siteSelected).childNodes[1].childNodes[0].innerHTML);
        "facebook" == e ? window.open("http://www.facebook.com/sharer.php?u=" + t + "&t=" + i, "sharer", "toolbar=0,status=0,width=700,height=436") : "stumbleupon" == e ? window.open("http://www.stumbleupon.com/submit?url=" + t + "&title=" + i, "stumbleupon", "toolbar=0,status=0,width=826,height=536") : "addressbook" == e && window.open("/weebly/apps/abook.php?site=" + siteSelected, "abook", "toolbar=0,status=0,width=700,height=426")
    }
}
function showDeletePrompt(e, t) {
    t.appendChild($("showPromptDelete")), $("promptUserDelete").innerHTML = '<h2 style="font-size: 16px;">' + _W.tl("Confirm Deletion") + '</h2><br/><p style="font-size: 12px;">' + _W.tl("This action is permanent -- if you delete this site, there is no way to retrieve it.") + '</p><br/><p style="font-size: 12px;"><a href="#" onClick="deleteSite(' + "'" + e + "'" + "); Effect.SlideUp(" + "'showPromptDelete'" + '); return false;" style="color: red; margin-right: 15px;"><img src="images/page_cross.gif" /><font style="position: relative; top:-3px; margin-left: 2px;">' + _W.tl("Yes, delete this site") + '</font></a><a href="#" onClick="Effect.SlideUp(' + "'showPromptDelete'" + '); return false;"><img src="images/page_next.gif" /><font style="position: relative; top:-3px; margin-left: 2px;">' + _W.tl("No, keep this site") + "</font></a></p>", Effect.SlideDown("showPromptDelete")
}
function showMoveSiteService(e, t) {
    t = t || {};
    var i = $("move-site-service-prompt").select(".radio-box");
    i.each(function(e) {
        e.stopObserving(), e.observe("click", function() {
            i.invoke("removeClassName", "radio-box-selected"), e.addClassName("radio-box-selected"), e.down("input").checked=!0
        })
    }), i.invoke("removeClassName", "radio-box-selected"), $("move-site-service-new").addClassName("radio-box-selected"), $("move-new-site-input").checked=!0, t.title && $("move-site-service-title").update(t.title), $("move-site-service-none").show(), t.hide_none && $("move-site-service-none").hide();
    var n = t.level;
    e ? (n = Weebly.Sites[e].level, $("move-site-service-site-description").show(), $$(".move-service-initial-service").invoke("update", Weebly.Restrictions.levelToName(n))) : $("move-site-service-site-description").hide(), $("move-site-service-current").removeClassName("error");
    var o = function() {
        var e = $("move-site-service-current-sites").value, t = Weebly.Sites[e].level;
        t > 0 && t != n ? ($("move-site-service-current").addClassName("error"), $$(".move-service-new-service").invoke("update", Weebly.Restrictions.levelToName(t))) : $("move-site-service-current").removeClassName("error")
    };
    if (Weebly.Sites) {
        var s = "", r = $("move-site-service-current-sites");
        $H(Weebly.Sites).values().each(function(t) {
            t.user_id == userID && t.site_id != e && (s += '<option value="' + t.site_id + '">' + t.site_title.truncate(20) + "</option>")
        }), s ? ($("move-site-service-current").show(), r.update(s), $("move-site-service-current-sites_chzn") ? Event.fire(r, "liszt:updated") : new Chosen(r), o(), r.stopObserving(), r.observe("change", o)) : $("move-site-service-current").hide()
    }
    var a = $("move-site-service-button");
    a.stopObserving("click"), a.observe("click", function() {
        new Ajax.Request(ajax, {
            parameters: Object.extend({
                pos: "movesiteservice",
                source_site_id: e,
                target_site_id: r.value,
                user_service_id: t.user_service_id,
                delete_source_site: !!t.delete_site
            }, $("move-site-service-options").serialize(!0)),
            onSuccess: function(e) {
                var t = e.responseText.evalJSON();
                t.success ? window.location.reload() : alert(t.reason)
            }
        })
    });
    var l = new Weebly.Dialog("move-site-service-prompt", {
        modal: !0
    });
    l.open()
}
function deleteSite(e) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "deletesite",
            reqid: e,
            cookie: document.cookie
        },
        onSuccess: handlerDeleteSite,
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function handlerDeleteSite(e) {
    e.responseText.indexOf("%%SUCCESS%%")>-1 ? document.location = "userHome.php" : showErrorMessage(_W.tl("There was an error removing your site. Please try again."))
}
function removeSiteAccess(e) {
    var t = Weebly.Sites[e];
    new Ajax.Request(ajax, {
        parameters: {
            pos: "removesiteaccess",
            contributor_site_id: e,
            site_user_id: t.user_id
        },
        onSuccess: function() {
            window.location.reload()
        }
    })
}
function removeAdSense() {
    $("adsenseError") && $("adsenseError").parentNode.removeChild($("adsenseError")), new Ajax.Request(ajax, {
        parameters: {
            pos: "removeadsenseaccount",
            cookie: document.cookie
        },
        onSuccess: handlerRemoveAdSense,
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function handlerRemoveAdSense() {
    window.location.reload()
}
function adsenseHideNew() {
    $("asnew_2").checked=!0, $("asexisting_2").checked=!0, $("adsense_newUser").style.display = "none", $("adsense_existingUser").style.display = "block", $("as_action").style.display = "none"
}
function adsenseShowNew() {
    $("adsense_newUser").style.display = "block", $("adsense_existingUser").style.display = "none", $("asnew_1").checked=!0, $("asexisting_1").checked=!0
}
function removeUserNote(e) {
    $("user-note-" + e).hide(), new Ajax.Request(ajax, {
        parameters: {
            pos: "deleteUserNote",
            note_id: e,
            cookie: document.cookie
        },
        onFailure: errFunc,
        onException: exceptionFunc
    });
    var t = 0;
    $$("div.notification").each(function(e) {
        e.visible() && t++
    }), 0 == t && $("notifications-container").hide()
}
function clickHandler(e) {
    var t;
    if (!e)
        var e = window.event;
    e.target ? t = e.target : e.srcElement && (t = e.srcElement), 3 == t.nodeType && (t = t.parentNode), isAParent("weeblyLightbox", t)&&!isAParent("weeblyLightboxInside", t) && ($("weeblyLightboxInside").down("#adsense_form") ? Pages.go("displayUserSettings", "adsense") : Pages.go("main"))
}
function removeExternalAccount(e) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "removeexternalaccount",
            type: e,
            cookie: document.cookie
        },
        onSuccess: function(t) {
            handlerRemoveExternalAccount(t, e)
        },
        onFailure: errFunc
    })
}
function handlerRemoveExternalAccount() {
    window.location.reload()
}
function getFlashVersion() {
    try {
        try {
            var e = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");
            try {
                e.AllowScriptAccess = "always"
            } catch (t) {
                return "6,0,0"
            }
        } catch (t) {}
        return new ActiveXObject("ShockwaveFlash.ShockwaveFlash").GetVariable("$version").replace(/\D+/g, ",").match(/^,?(.+),?$/)[1]
    } catch (t) {
        try {
            if (navigator.mimeTypes["application/x-shockwave-flash"].enabledPlugin)
                return (navigator.plugins["Shockwave Flash 2.0"] || navigator.plugins["Shockwave Flash"]).description.replace(/\D+/g, ",").match(/^,?(.+),?$/)[1]
        } catch (t) {}
    }
    return "0,0,0"
}
function saveFlashVersion() {
    var e = getFlashVersion();
    new Ajax.Request(ajax, {
        parameters: {
            pos: "saveflashversion",
            version: e,
            cookie: document.cookie
        },
        bgRequest: !0
    })
}
function slideSiteAreaOpen(e, t, i, n) {
    hideMore(), siteAreaOpen=!0, activeSite = e, showSiteAreaButtons(), Weebly.Sites[e].permissions&&!Weebly.Sites[activeSite].permissions.include("edit_site") ? $("edit-site-subnav").hide() : ($("edit-site-subnav").show(), $("edit-site-subnav").href = "toSite.php?site=" + e);
    var o = Weebly.Sites[e].site_title.length > 50 ? Weebly.Sites[e].site_title.substr(0, 50) + "...": Weebly.Sites[e].site_title;
    i && "Not Published" != i ? $("site-data-area-title").update().insert(new Element("a", {
        href: "http://" + i,
        target: "_blank",
        title: _W.tl("Visit Site")
    }).update(o)) : $("site-data-area-title").update(o), $$(".client-site-name").invoke("update", o), $("fb-likes") && $("fb-likes").hide(), new Effect.Tween($("scroll-area"), 0, 960, {
        duration: .45,
        afterFinish: function() {
            n && ($("scroll-area").setStyle({
                height: "1063px"
            }), $(parent.document.getElementById("scroll-area")).setStyle({
                height: "1070px"
            }))
        }
    }, function(e) {
        $("scroll-area").scrollLeft = e
    }), n ? changeClientSiteTab(t) : changeSiteTab(t)
}
function showSiteAreaButtons() {
    var e = Weebly.Sites[activeSite], t = $("tab-button-stats").show(), i = $("tab-button-blog_moderation").show(), n = $("tab-button-form_data").show();
    e.permissions && (e.permissions.include("view_stats") || t.hide(), e.permissions.include("moderate_comments") || i.hide(), e.permissions.include("manage_form_entries") || n.hide())
}
function slideSiteAreaClosed(e) {
    siteAreaOpen=!1, activeSite = "", e && ($("scroll-area").setStyle({
        height: "603px"
    }), $(parent.document.getElementById("scroll-area")).setStyle({
        height: "603px"
    })), new Effect.Tween($("scroll-area"), 960, 0, {
        duration: .45,
        afterFinish: function() {
            $("fb-likes") && $("fb-likes").show()
        }
    }, function(e) {
        $("scroll-area").scrollLeft = e
    })
}
function changeSiteTab(e) {
    $$(".site-tab").invoke("hide"), $("site-data-box") && $("site-data-box").show(), $(e + "-" + activeSite) ? $(e + "-" + activeSite).show() : $(e) && $(e).show(), $("subnav-buttons").select(".btn").invoke("removeClassName", "btn-inverse");
    var t = $("tab-button-" + e);
    t && t.addClassName("btn-inverse"), "stats" == e && loadSiteStats(activeSite, Weebly.Sites[activeSite].user_id), "form_data" == e && loadSiteForms(activeSite, Weebly.Sites[activeSite].user_id), "blog_moderation" == e && loadSiteBlogs(activeSite, Weebly.Sites[activeSite].user_id)
}
function changeClientSiteTab(e) {
    changeSiteTab(e), $$(".client-site-tab-active").invoke("removeClassName", "client-site-tab-active"), $(e + "-tab").addClassName("client-site-tab-active")
}
function showMore(e, t, i) {
    displayMoreLinks(e);
    var n = $("more-container");
    if ($("more-link-blog").onclick = function() {
        slideSiteAreaOpen(e, "blog_moderation", t)
    }, $("more-link-form").onclick = function() {
        slideSiteAreaOpen(e, "form_data", t)
    }, $("more-link-copy").onclick = function() {
        openCopySiteDialog(e)
    }, Weebly.Sites[e].permissions)
        $("more-link-delete").onclick = function() {
            return showConfirmationPopup("This action is permanent -- if you remove your access to this site, you will need to be re-invited by the site owner.", function() {
                removeSiteAccess(e)
            }, {
                title: "Remove access to " + i + "?",
                confirmButtonText: "Remove access",
                cancelButtonText: "cancel"
            }), !1
        };
    else {
        if ($("more-link-designer-move")) {
            var o = Weebly.Sites[e];
            if (0 == o.site_version || o.custom_domain) {
                var s = jQuery("#more-button-" + e).closest(".site").data("commerce");
                $("more-link-designer-move").onclick = function() {
                    new Weebly.Dialog("move-to-designer-site", {
                        modal: !0
                    }).open(), $("move-to-designer-site-form")["move-designer-site-id"].value = e, s && jQuery("#move-to-designer-site").find(".commerce-warning").show()
                }
            } else 
                $("more-link-designer-move").onclick = function() {
                    new Weebly.Dialog("move-to-designer-site-alert", {
                        modal: !0
                    }).open()
                }
        }
        $("more-link-delete").onclick = function() {
            return !Weebly.Sites[e].hasTrial && Weebly.Restrictions.userLevel == Weebly.Restrictions.freeLevel && Weebly.Sites[e].level > Weebly.Restrictions.freeLevel ? showMoveSiteService(e) : showConfirmationPopup("This action is permanent -- if you delete this site, there is no way to retrieve it.", function() {
                deleteSite(e)
            }, {
                title: "Delete " + i + "?",
                confirmButtonText: "Delete site",
                cancelButtonText: "cancel"
            }), !1
        }
    }
    var r = safeCumulativeOffset($("more-button-" + e)), a = safeCumulativeOffset($("scroll-area"));
    n.style.left = r.left - a.left + 1 + "px", n.style.top = r.top - a.top + 28 + "px", n.show(), document.observe("mousedown", hideMore)
}
function displayMoreLinks(e) {
    var t = Weebly.Sites[e], i = $("more-link-blog").up().show(), n = $("more-link-form").up().show(), o = $("more-link-copy").up().show();
    $("more-link-delete").up().show();
    var s = $("more-link-designer-move");
    s && s.up().show(), t.permissions ? ($("more-link-delete").update('<span class="icon w-icon-remove-person"></span><span>' + _W.tl("Remove My Access") + "</span>"), o.hide(), s && s.up().hide(), t.permissions.include("moderate_comments") || i.hide(), t.permissions.include("manage_form_entries") || n.hide()) : $("more-link-delete").update('<span class="icon w-icon-delete"></span><span>' + _W.tl("Delete Site") + "</span>")
}
function showClientMore(e, t, i) {
    var n = $("more-container");
    $("more-link-blog").onclick = function() {
        slideSiteAreaOpen(e, "blog_moderation", t, !0)
    }, $("more-link-form").onclick = function() {
        slideSiteAreaOpen(e, "form_data", t, !0)
    }, $("more-link-stats").onclick = function() {
        slideSiteAreaOpen(e, "stats", t, !0)
    }, $("more-link-delete").onclick = function() {
        return showConfirmationPopup("This action is permanent -- if you delete this site, there is no way to retrieve it.", function() {
            deleteClientSite(e)
        }, {
            title: "Delete " + i + "?",
            confirmButtonText: "Delete site",
            cancelButtonText: "cancel"
        }), $("weebly-bluebox-container").setStyle({
            top: "203px"
        }), $("weebly-bluebox-bordercontainer").setStyle({
            top: "182px"
        }), !1
    };
    var o = $("more-button-" + e).cumulativeOffset().top > 350?-130 : 28;
    n.clonePosition($("more-button-" + e), {
        setWidth: !1,
        setHeight: !1,
        offsetTop: o,
        offsetLeft: - 43
    }), $("more-container").show(), document.observe("mousedown", hideMore)
}
function hideMore(e) {
    e && Event.element(e).up(".site-buttons-more") || ($$(".site-buttons-more").invoke("hide"), document.stopObserving("mousedown", hideMore))
}
function moveToDesignerSite() {
    new Ajax.Request(ajax, {
        parameters: Object.extend({
            pos: "movesitetodesigner",
            cookie: document.cookie
        }, $("move-to-designer-site-form").serialize(!0)),
        onSuccess: function(e) {
            e.responseText.match(/%%SUCCESS%%/) ? window.location.href = "userHome.php?page=developer&subpage=site&site_id=" + $("move-to-designer-site-form")["move-designer-site-id"].value : showWarning("There was an error moving your site to the Designer Platform")
        }
    })
}
function slideStatsDetailOpen(e) {
    $("stats-detail-area").morph("top:500px"), loadStatsDetail(e)
}
function slideStatsDetailClosed() {
    $("stats-detail-area").morph("top:1038px")
}
function loadSiteStats(e, t) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "sitestats",
            site_id: e,
            user_id: t
        },
        onSuccess: function(t) {
            $("stats-" + e).update(t.responseText)
        }
    })
}
function loadStatsDetail(e) {
    var t = "Top Pages", i = "Views", n = "Page", o = "statstoppages";
    switch (e) {
    case"top_pages":
        t = "Top Pages", i = "Views", n = "Page", o = "statstoppages";
        break;
    case"top_referring":
        t = "Referring Sites", i = "Views", n = "Referrer", o = "statstopreferring";
        break;
    case"top_searches":
        t = "Search Terms", i = "Views", n = "Keyword", o = "statstopsearches"
    }
    $("detail-heading-title").update(t), $("detail-key-heading").update(i), $("detail-value-heading").update(n);
    var s = {
        pos: o,
        site_id: activeSite,
        user_id: Weebly.Sites[activeSite].user_id,
        cookie: document.cookie
    };
    new Ajax.Request(ajax, {
        parameters: s,
        onSuccess: updateStatsDetail
    }), $("detail-body").update("Loading...")
}
function updateStatsDetail(e) {
    var t = e.responseText.evalJSON(), i = new Template('<div class="detail-row"><div class="detail-key">#{pageviews}</div><div class="detail-value">#{value}</div></div>');
    $("detail-body").update(""), t.each(function(e) {
        $("detail-body").insert({
            bottom: i.evaluate(e)
        })
    })
}
function loadSiteForms(e, t) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "getuserforms",
            site_id: e,
            user_id: t,
            cookie: document.cookie
        },
        onSuccess: function(e) {
            var t = e.responseText.evalJSON(), i = "";
            t.size() > 0 ? (hideNoSiteForms(), $("form-data-iframe").show(), $("site-form-controls").show(), siteForms = new Hash, t.each(function(e) {
                siteForms.set(e.ucfid, e);
                var t = e.deleted ? e.name + " [deleted]": e.name;
                i += '<option value="' + e.ucfid + '">' + t + "</option>"
            })) : (showNoSiteForms(), $("form-data-iframe").hide(), $("site-form-controls").hide()), $("site-forms-select").update(i), $("site-forms-select").value && showSiteFormData($("site-forms-select").value)
        }
    })
}
function showNoSiteForms() {
    var e = document.getElementById("no-site-form-video");
    e && (e.src = e.getAttribute("data-src")), $("no-site-forms").show()
}
function hideNoSiteForms() {
    var e = document.getElementById("no-site-form-video");
    e && (e.src = ""), $("no-site-forms").hide()
}
function showSiteFormData(e) {
    var t = siteForms.get(e);
    $("form-data-iframe").show(), $("form-data-iframe").src = "viewFormData.php?ucfid=" + t.ucfid + "&user_id=" + t.user + "&hash=" + t.hash + "&site_id=" + activeSite, $("site-forms-entries").update(t.total), $("site-forms-view").href = t.link, $("site-forms-export").href = "/weebly/exportFormData.php?ucfid=" + t.ucfid + "&user_id=" + t.user + "&hash=" + t.hash
}
function loadSiteBlogs(e, t) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "getsiteblogs",
            site_id: e,
            user_id: t,
            cookie: document.cookie
        },
        onSuccess: function(i) {
            var n = i.responseText.evalJSON(), o = "";
            if (n.size() > 0) {
                $("no-site-blogs").hide(), $("blog_moderation_iframe").show(), $("blog-moderation-controls").show();
                var s = [];
                n.each(function(e) {
                    o += '<option value="' + e.blog_id + '" data-user-id="' + e.user_id + '" data-site-id="' + e.site_id + '">' + e.title + "</option>", s.push(e.blog_id)
                }), s.length > 1 && (o = '<option value="' + s.join(",") + '" data-user-id="' + t + '" data-site-id="' + e + '">' + _W.tl("All Blogs") + "</option>" + o)
            } else 
                $("blog-moderation-controls").hide(), $("blog_moderation_iframe").hide(), hideCommentsLoading(), $("no-site-blogs").show();
            $("site-blogs-select").update(o), $("site-blogs-select").value && showBlogComments("all", $("site-blogs-select").value)
        }
    })
}
function showBlogComments(e) {
    if (null === e) {
        var t = $$(".blog-moderation-filter")[0].select("a.selected");
        e = t ? t[0].readAttribute("class").replace("selected", "").strip() : "all"
    }
    var i = $("site-blogs-select").select("option[selected]")[0], n = i.readAttribute("value"), o = i.readAttribute("data-site-id"), s = i.readAttribute("data-user-id"), t = $$(".blog-moderation-filter a.selected");
    t.length && t[0].removeClassName("selected"), $$(".blog-moderation-filter a." + e)[0].addClassName("selected");
    var r = getBlogCommentsUrl(e, n, o, s);
    $("blog_moderation_iframe").src = r
}
function blogModerationAction(e) {
    var t = $("blog_moderation_iframe");
    if (t) {
        var i = t.contentWindow;
        if (i) {
            var n = {
                approve: i.approveComments,
                "delete": i.deleteComments,
                spam: i.markSpamComments
            };
            n[e] && n[e]()
        }
    }
}
function getBlogCommentsUrl(e, t, i, n, o) {
    var s = $("blog_moderation_iframe"), r = s.hasClassName("is-overlay") ? "slim": "normal", a = new Hash({
        blog_id: t,
        type: e,
        site_id: i,
        user_id: n,
        size: o || r,
        r: parseInt(99999999 * Math.random())
    });
    return "viewBlogModeration.php?" + a.toQueryString()
}
function openAccountChange(e) {
    activeSetting = e, $$(".account-change").invoke("hide"), $("account-change-" + e).show(), $("account-change-error").hide(), new Weebly.Dialog("account-change-container", {
        modal: !0
    }).open()
}
function showAccountError(e) {
    $("account-change-error").update(e), $("account-change-error").show()
}
function changeUsername() {
    $("account-change-error").hide();
    var e = {
        pos: "changeusername",
        username: $("new-username").value,
        token: Weebly.RequestToken,
        cookie: document.cookie
    };
    new Ajax.Request(ajax, {
        parameters: e,
        onSuccess: function(e) {
            var t = e.responseText;
            t.match("%%SUCCESS%%") ? ($("current-username").update($("new-username").value), Weebly.AllDialogs["account-change-container"].close()) : showAccountError(t)
        },
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function changeEmail() {
    $("account-change-error").hide();
    var e = {
        pos: "changeuseremail",
        email: $("new-email").value,
        token: Weebly.RequestToken,
        cookie: document.cookie
    };
    new Ajax.Request(ajax, {
        parameters: e,
        onSuccess: function(e) {
            var t = e.responseText;
            t.match("%%SUCCESS%%") ? ($("current-email").update($("new-email").value), Weebly.AllDialogs["account-change-container"].close()) : showAccountError(t)
        },
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function changeFullName() {
    $("account-change-error").hide();
    var e = {
        pos: "changeuserfullname",
        fullname: $("new-fullname").value,
        token: Weebly.RequestToken,
        cookie: document.cookie
    };
    new Ajax.Request(ajax, {
        parameters: e,
        onSuccess: function(e) {
            var t = e.responseText;
            t.match("%%SUCCESS%%") ? ($("current-fullname").update($("new-fullname").value), Weebly.AllDialogs["account-change-container"].close()) : showAccountError(t)
        },
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function changePassword(e) {
    $("account-change-error").hide();
    var t = {
        pos: "changeuserpassword",
        new_password: $("new-password").value,
        repeat_password: $("repeat-password").value,
        token: Weebly.RequestToken,
        cookie: document.cookie
    };
    t.new_password == t.repeat_password && t.new_password.length >= 6 ? new Ajax.Request(ajax, {
        parameters: t,
        onSuccess: function(t) {
            var i = t.responseText;
            i.match("%%SUCCESS%%") ? e ? window.location.href = e : Weebly.AllDialogs["account-change-container"].close() : showAccountError(i)
        },
        onFailure: errFunc,
        onException: exceptionFunc
    }) : showAccountError(_W.tl("New password must match repeat password and be at least 6 characters long"))
}
function showRemoveAlert(e, t) {
    $("remove-setting-confirm-text").update(e), $("remove-setting-confirm-button").onclick = "adsense" == t ? function() {
        return removeAdSense(), !1
    } : function() {
        return removeExternalAccount(t), !1
    }, openAccountChange("remove")
}
function showConfirmationPopup(e, t, i) {
    i = Object.extend({
        title: "Confirm Change",
        confirmButtonText: "Yes",
        cancelButtonText: "No"
    }, i || {}), $("confirmation-popup-error").update(""), $("confirmation-popup-text").update(e), $("confirmation-popup-title").update(i.title.truncate(35)), $("confirmation-popup-confirm-button").update(i.confirmButtonText), $("confirmation-popup-cancel-button").update(i.cancelButtonText), $("confirmation-popup-confirm-button").onclick = t, new Weebly.Dialog("confirmation-popup", {
        modal: !0
    }).open()
}
function disableAutoRenew(e) {
    var t = {
        pos: "disableautorenew",
        user_service_id: e,
        cookie: document.cookie
    };
    new Ajax.Request(ajax, {
        parameters: t,
        onSuccess: function(e) {
            e.responseText.match("%%SUCCESS%%") ? window.location.reload() : $("confirmation-popup-error").update("There was an error removing auto-renew.  Please try again.")
        },
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function enableAutoRenew(e) {
    var t = {
        pos: "enableautorenew",
        user_service_id: e,
        cookie: document.cookie
    };
    new Ajax.Request(ajax, {
        parameters: t,
        onSuccess: function(e) {
            e.responseText.match("%%SUCCESS%%") ? window.location.reload() : $("confirmation-popup-error").update("There was an error adding auto-renew.  Please try again.")
        },
        onFailure: errFunc,
        onException: exceptionFunc
    })
}
function hideCommentsLoading() {
    var e = $("comments-loading");
    e && e.hide()
}
function disableDeveloper() {
    confirm("Are you sure you want to disable the Designer Platform for this account?") && new Ajax.Request(ajax, {
        parameters: {
            pos: "disabledeveloper",
            cookie: document.cookie
        },
        onSuccess: function() {
            window.location.reload()
        }
    })
}
function enableDeveloper() {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "enabledeveloper",
            cookie: document.cookie
        },
        onSuccess: function(e) {
            e.responseText.match(/%%ENABLED%%/) ? window.location.reload() : e.responseText.match(/%%CREATE%%/) && new Weebly.Dialog("setup-designer", {
                modal: !0
            }).open()
        }
    })
}
function openEducationForm() {
    $("education-form").show(), Weebly.blueBox.open("education-form-container", {
        width: "575px"
    })
}
function saveEducationForm() {
    function e(e) {
        return e.replace(/</g, "&lt;").replace(/>/g, "&gt;")
    }
    var t = {};
    $("education-form").select("input,select").each(function(e) {
        t[e.id.match(/ed-(\w+)-input/)[1]] = e.value
    });
    var i = {};
    for (name in t)
        name.match(/(school|district|state|country)/) ? i["ed_" + name] = t[name] : i[name] = t[name];
    new Ajax.Request("getElements.php", {
        parameters: {
            pos: "edusersettings",
            settings: Object.toJSON(i)
        },
        onSuccess: function() {
            $("ed-full-name") ? $("ed-full-name").innerHTML = e((t.honorific + " " + (t.fname + " " + t.lname).strip()).strip()) : ($("ed-fname").innerHTML = e(t.fname), $("ed-lname").innerHTML = e(t.lname));
            for (name in t)
                "fname" != name && "lname" != name && "honorific" != name && ($("ed-" + name).innerHTML = e(t[name]));
            "United States" == t.country ? $("ed-state-row").show() : $("ed-state-row").hide(), Weebly.blueBox.close()
        }
    })
}
function edCountryChange(e) {
    $("ed-state-input-row").style.visibility = "United States" == e.value ? "visible" : "hidden"
}
function storeScrollAreaHeight() {
    originalScrollAreaHeight || (originalScrollAreaHeight = $("scroll-area").getHeight())
}
function resetScrollAreaHeight() {
    originalScrollAreaHeight && $("scroll-area").setStyle({
        height: originalScrollAreaHeight + "px"
    })
}
function safeCumulativeOffset(e) {
    if (e.getBoundingClientRect && "BODY" != e.nodeName && "HTML" != e.nodeName) {
        var t = e.getBoundingClientRect(), i = document.body, n = document.documentElement, o = n.clientTop || i.clientTop || 0, s = n.clientLeft || i.clientLeft || 0, r = window.pageYOffset || n.scrollTop || i.scrollTop, a = window.pageXOffset || n.scrollLeft || i.scrollLeft, l = t.top + r - o, c = t.left + a - s, u = [c, l];
        return u.left = c, u.top = l, u
    }
    return $(e).cumulativeOffset()
}
function disableAddon(e, t) {
    confirmDisableEmailForwarding(e, t, {
        afterConfirm: function() {
            window.location.reload()
        }
    })
}
function toggleDesignerSites() {
    var e = $("designer-site-list"), t = $("designer-view-sites-button");
    e && t && (e.visible() ? (Effect.SlideUp("designer-site-list"), t.update(_W.tl("View sites"))) : (Effect.SlideDown("designer-site-list"), t.update(_W.tl("Hide sites"))))
}
function initBillingHelpIcons() {
    $$(".billing-help-icon").each(function(e) {
        e.observe("mouseover", function() {
            showBillingHelp(e)
        }), e.observe("mouseout", function() {
            hideBillingHelp(e)
        })
    })
}
function showBillingHelp(e) {
    var t = e.getAttribute("data-help-text"), i = new Element("div", {
        id: "designer-help-popup"
    }), n = e.cumulativeOffset(), o = n.top + 20, s = n.left - 250;
    i.update(t), i.setStyle({
        top: o + "px",
        left: s + "px"
    }), $(document.body).insert({
        bottom: i
    })
}
function hideBillingHelp() {
    var e = $("designer-help-popup");
    e && (e.hide(), e.remove())
}
function showVideoDialog() {
    var e, t = document.getElementById("sys-video"), i = "//www.youtube.com/embed/mM9uhe27YbA?rel=0&enablejsapi=1&wmode=opaque";
    window.addEventListener("message", function(i) {
        i.source === t.contentWindow && (e && (e = clearInterval(e)), t.contentWindow.postMessage('{"event":"command","func":"playVideo","args":""}', "*"))
    }), t.src = i, new Weebly.Dialog("userhome-sys-video", {
        modal: !0,
        showClose: !0,
        onOpen: function() {
            e = setInterval(function() {
                t && t.contentWindow && t.contentWindow.postMessage('{"event":"listening","id":"sys-video"}', "*")
            }, 250)
        },
        onClose: function() {
            t.remove(), window.location.hash = ""
        }
    }).open()
}
function showExpiredTrial(e) {
    homeApp.billing.showTrialExpired(e)
}
function domainChoiceSelect(e) {
    var t = Array("domainSubdomain", "domainNewDomain", "domainExistingDomain");
    if (e != domainChoiceSelected) {
        for (var i, n = 0; n < t.length; n++)
            if (i = $(t[n])) {
                if (t[n] == e) {
                    $(t[n]).addClassName("multiple-choice-box-selected");
                    var o = $(t[n]).down("input[type=radio]");
                    setTimeout(function() {
                        o.checked=!0
                    }, 10)
                } else 
                    $(t[n]).removeClassName("multiple-choice-box-selected"), $(t[n]).down("input[type=radio]").checked=!1;
                    (t[n] == e || "none" == $(t[n] + "-2").style.display) && (t.splice(n, 1), n--)
            }
        domainChoiceSelected = e
    }
}
function domainChoiceReset() {
    isDomainChooserOnboarding = isDomainChooserOnboarding||!!window.location.pathname.match("newSite.php"), isDomainChooserOnboarding ? (jQuery("#reserve-domain-now").show(), jQuery("#existingDomainSubtitle").text(_W.tl("We'll help you connect your domain when you're ready to publish"))) : (jQuery("#reserve-domain-now").hide(), Weebly.Restrictions.hasAccess("existing_domain") ? jQuery("#existingDomainSubtitle").text(_W.tl("We'll help you connect your domain in the next step.")) : jQuery("#existingDomainSubtitle").text(_W.tl("Choose a plan and connect your domain in the next step"))), jQuery("#domainChoiceSkip").show();
    var e;
    Array("domainSubdomain", "domainNewDomain", "domainExistingDomain"), $$(".multiple-choice-box-selected").invoke("removeClassName", "multiple-choice-box-selected"), (e = $("domainSubdomain-2")) && (e.style.display = "block"), (e = $("domainNewDomain-2")) && (e.style.display = "block", $("domain_sld").value = "", $("domainStatus").innerHTML = ""), (e = $("domainExistingDomain-2")) && ($("weeblyExistingDomain").value = ""), domainChoiceSelected = "";
    var t = Weebly.Location && Weebly.Location.weebly_subdomain ? Weebly.Location.weebly_subdomain: "";
    t = t.match(/^[\d]+$/) ? "" : t, $("domainChoiceTitle") && $("domainChoiceTitle").update("string" == typeof defaultDomainChoiceTitle ? defaultDomainChoiceTitle : _W.tl("Choose Your Website Domain")), Weebly.Restrictions && Weebly.Restrictions.hasAccess("domain_buying") && ($("domainExistingTitleDefault").show(), $("domainExistingTitleClient").hide(), $("clientDomainHelp").hide()), $("designerDomainSetupMessage").hide(), Weebly.Restrictions && Weebly.Restrictions.isClientSite && ($("domainChoiceTitle").update("Setup a Client Domain"), $("domainExistingTitleDefault").hide(), $("domainExistingTitleClient").show(), $("designerDomainSetupMessage").show(), $("clientDomainHelp").show()), $("chooseDomainDiv").style.display = "block", $("domainWrapper").style.display = "block", $("domainRegisterComplete").style.display = "none", $("domainExistingException").style.display = "none", $("clientDomainExistingException").style.display = "none", $("weeblyDomain").value = t, $("weeblyExistingDomain") && ($("weeblyExistingDomain").value = Weebly.Location && Weebly.Location.custom_domain&&!Weebly.Location.purchased_domain ? Weebly.Location.custom_domain : "www.example.com", $("existingDomainInvalid").hide()), $("domainChoiceError").innerHTML = "", $("domainChoiceError").style.display = "none", $("weeblyDomainStatus").innerHTML = "", $("finalDomainName").update(""), $("weeblyExistingDomain") && "www.example.com" != $("weeblyExistingDomain").value ? (domainChoiceSelect("domainExistingDomain"), $("weeblyExistingDomain").setStyle({
        color: ""
    })) : $("weeblyDomain").value&&!Weebly.Location.purchased_domain ? domainChoiceSelect("domainSubdomain") : domainChoiceSelect(void 0)
}
function domainChoiceContinue() {
    if ($("domainChoiceError").display = "none", "domainSubdomain" == domainChoiceSelected)
        new Ajax.Request(ajax, {
            parameters: {
                pos: "savedomain",
                type: "subdomain",
                domain: $("weeblyDomain").value,
                cookie: document.cookie
            },
            onSuccess: handlerSaveSubdomain,
            onFailure: errFunc
        });
    else if ("domainNewDomain" == domainChoiceSelected)
        if ($("finalDomainName").innerHTML) {
            if (Weebly.Permissions&&!Weebly.Permissions.allow_purchases)
                return alert("Only the site owner is able to purchase a domain name for this site"), !1;
                showPurchaseScreen({
                    refer: domainPurchaseRefer,
                    type: "domain",
                    extraArgs: "domain=" + $("finalDomainName").innerHTML
                })
        } else 
            $("domainChoiceError").innerHTML = "Please select an available domain.", $("domainChoiceError").style.display = "block";
        else if ("domainExistingDomain" == domainChoiceSelected)
        if ($("weeblyExistingDomain").value.match(/[a-zA-Z\-0-9]+[\.]+/)) {
            var e = $("weeblyExistingDomain").value;
            isDomainChooserOnboarding ? (Weebly.Location && (Weebly.Location.custom_domain = e, Weebly.Location.purchased_domain=!1), domainChoiceFinish()) : Weebly.Restrictions&&!Weebly.Restrictions.hasAccess("existing_domain")&&-1 == Weebly.Location.all_purchased_domains.indexOf("www." + e.toLowerCase().replace(/^www\./, "")) ? editorApp.vent.trigger("billing:upgrade:window", {
                restrictionID: "existing_domain",
                restrictionType: "feature"
            }) : new Ajax.Request(ajax, {
                parameters: {
                    pos: "savedomain",
                    continue_not_configured: "0",
                    type: "domain",
                    domain: e,
                    cookie: document.cookie
                },
                onSuccess: handlerSaveExistingDomain,
                onFailure: errFunc
            })
        } else 
            $("existingDomainInvalid").show()
    }
function handlerSaveExistingDomain(e) {
    if (e.responseText.indexOf("the following problems") > 0)
        $("domainChoiceError").innerHTML = e.responseText, $("domainChoiceError").style.display = "block";
    else if (e.responseText.indexOf("%%accessdenied%%")>-1)
        $("domainChoiceError").innerHTML = _W.tl("Please upgrade to use an existing domain name."), $("domainChoiceError").style.display = "block";
    else if (e.responseText.indexOf("%%notconfigured%%")>-1)
        if (isDomainChooserOnboarding)
            domainExistingExceptionContinue();
        else {
            e.responseText = e.responseText.replace(/%%notconfigured%%/, "");
            var t = e.responseText.split("|");
            $("domainExistingInfo").innerHTML = "<a href='http://" + t[1] + ".weebly.com/' target='_blank'>http://" + t[1] + ".weebly.com</a>", Weebly.Restrictions && Weebly.Restrictions.isClientSite ? ($("client-registrar-email").value = existingClientDomainEmailText.replace("%%DOMAIN_NAME%%", $("weeblyExistingDomain").value), $("domainChoiceTitle").update("Setup a Client Domain"), $("clientDomainExistingException").style.display = "block", $("chooseDomainDiv").style.display = "none") : ($("registrar-email").value = existingDomainEmailText.replace("%%DOMAIN_NAME%%", $("weeblyExistingDomain").value), $("domainChoiceTitle").update(_W.tl("Setup a Domain You Already Own")), $("domainExistingException").style.display = "block", $("chooseDomainDiv").style.display = "none"), domainNextStep = domainNextStep ? domainNextStep : "displaySiteSettings"
        } else 
            Weebly.Location.custom_domain = $("weeblyExistingDomain").value, Weebly.Location.purchased_domain=!1, domainChoiceFinish()
}
function showDomainInstructions(e) {
    Pages.go("domainMenu"), $("weeblyExistingDomain").value = e, $("registrar-email").value = existingDomainEmailText.replace("%%DOMAIN_NAME%%", $("weeblyExistingDomain").value), $("domainExistingException").style.display = "block", $("chooseDomainDiv").style.display = "none", domainDialog.positionDialog()
}
function domainExistingExceptionContinue() {
    var e = $("weeblyExistingDomain").value;
    return new Ajax.Request(ajax, {
        parameters: {
            pos: "savedomain",
            continue_not_configured: "1",
            type: "domain",
            domain: e,
            cookie: document.cookie
        },
        onSuccess: function() {
            showTrackingFrame("connected_external_domain"), Weebly.Location.custom_domain = e, Weebly.Location.purchased_domain=!1, domainChoiceFinish()
        },
        onFailure: errFunc
    }), !1
}
function handlerSaveSubdomain(e) {
    e.responseText.indexOf("the following problems") > 0 ? ($("domainChoiceError").innerHTML = e.responseText, $("domainChoiceError").style.display = "block") : (Weebly.Location && (Weebly.Location.weebly_subdomain = $("weeblyDomain").value, Weebly.Location.custom_domain = "", Weebly.Location.purchased_domain=!1), domainChoiceFinish())
}
function openCompetitorWindow() {
    "$217.82" == $("competitorTotal").innerHTML ? window.open("http://" + editorStatic + "/weebly/images/godaddy1year.png", "godaddy1", "toolbar=0,status=0,width=746,height=549") : window.open("http://" + editorStatic + "/weebly/images/godaddy2years.png", "godaddy2", "toolbar=0,status=0,width=820,height=562")
}
function completeDomainPurchase(e) {
    $("finalDomainNameConfirmation").innerHTML = e, $("finalDomainNameComplete").innerHTML = $("finalDomainName").innerHTML, $("domainRegisterComplete").style.display = "block", $("domainWrapper").style.display = "block", $("chooseDomain").style.width = "566px"
}
function continueFromPublish() {
    if ("" != $("publish_domain_sld").value) {
        if (!$("domainStatus").innerHTML.match(/Not Available/)&&!$("domainStatus").innerHTML.match(/Checking/) && "" != $("domainStatus").innerHTML) {
            var e = filterDomain($("publish_domain_sld").value);
            $("finalDomainName").innerHTML = e + "." + $("publish_domain_tld").value;
            var t = window.open(buildSecureUrl("/weebly/apps/purchasePage.php?type=domain&s=" + configSiteName + "&domain=" + $("finalDomainName").innerHTML), "weebly_billingPage", "height=700,width=960,toolbar=yes,location=yes,status=yes,resizable=yes,scrollbars=yes,dependent=yes");
            monitorBillingWindow(t)
        }
    } else 
        $("domainStatus").innerHTML = "Please choose a Domain"
}
function domainChoiceFinish() {
    if ("publish" == domainNextStep)
        settingQuickExport = 1, Pages.go("doExport");
    else if ("blogpostpublish" == domainNextStep)
        settingQuickExport = 1, currentBlog = jQuery.extend(!0, {}, domainNextBlog), goPublishPost(domainNextBlog.postId);
    else if ("editor" == domainNextStep) {
        var e = $("pro-referral-dialog");
        if (e) {
            domainDialog && domainDialog.close();
            var t = new Weebly.Dialog(e, {
                showClose: !1
            });
            t.open()
        } else 
            window.location = "main.php"
    } else 
        "categorizeSite" === domainNextStep ? Pages.go("categorizeSite") : "planningModal" === domainNextStep ? Pages.go("planningModal") : "displaySiteSettings" === domainNextStep ? (require("editor/app").settings.refreshSettings(), domainDialog.close(), Pages.currentNavPage = "none", showTrackingFrame("setup_custom_domain")) : Pages.go("main")
}
function onTextChange(e, t) {
    var i = $(e);
    if ("" == i.value)
        return isUserTyping = 0, $("domainStatus").innerHTML = "", void 0;
    if (t || 0 == isUserTyping)
        if (t)
            if (i && i.value && i.value == t) {
                isUserTyping = 0;
                var n = i.up(".multiple-choice-box");
                n || (n = $("exportText")), showDomainCheckChecking(n), "domain_sld" == e || "publish_domain_sld" == e ? checkDomain() : checkWeeblyDomain()
            } else 
                setTimeout(function() {
                    onTextChange(e, i.value)
                }, 700);
        else 
            isUserTyping = e, setTimeout(function() {
                onTextChange(e, i.value)
            }, 700)
}
function checkWeeblyDomain() {
    var e = $("weeblyDomain").value, t = new Array;
    if ("" != e) {
        t.push(new RegExp("^.*:\\/\\/", "")), t.push(new RegExp("^.*@", "")), t.push(new RegExp("^\\-+", "")), t.push(new RegExp("\\+$", "")), t.push(new RegExp("[^a-zA-Z0-9-]", "g"));
        for (var i = e, n = 0; n < t.length; n++)
            e = e.replace(t[n], "");
        e = e.substr(0, 40), i != e && ($("weeblyDomain").value = e), new Ajax.Request(ajax, {
            parameters: {
                pos: "checkWeeblyDomain",
                weeblySubDomain: e,
                cookie: document.cookie
            },
            onSuccess: handlerCheckWeeblyDomain,
            onFailure: errFunc,
            bgRequest: !0,
            onException: exceptionFunc
        })
    }
}
function handlerCheckWeeblyDomain(e) {
    e.responseText.match(/%%SUCCESS%%/) ? ($("domainSubdomain").down(".domain-check-success").show(), $("domainSubdomain").down(".domain-check-taken").hide(), $("domainSubdomain").down(".domain-check-checking").hide()) : ($("domainSubdomain").down(".domain-check-success").hide(), $("domainSubdomain").down(".domain-check-taken").show(), $("domainSubdomain").down(".domain-check-checking").hide())
}
function genericCheckDomain(e) {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "checkdomain",
            domainname: e.sld,
            domaintld: e.tld,
            cookie: document.cookie
        },
        onSuccess: function(t) {
            var i = t.responseText;
            i.match(/%%SUCCESS%%/) ? e.available(t) : i.match(/%%FAILURE%%/) && e.unavailable(t)
        },
        onFailure: errFunc,
        onException: exceptionFunc,
        bgRequest: !0
    })
}
function checkDomain() {
    if ($("publish_domain_sld"))
        var e = $("publish_domain_sld"), t = $("publish_domain_tld");
    else 
        var e = $("domain_sld"), t = $("domain_tld");
    if (domainName = e.value, domainNameTld = t.value, "" != domainName) {
        var i = domainName;
        domainName = filterDomain(domainName), i != domainName && (e.value = domainName), $("domainStatus").innerHTML = '<a style="color: grey;">' + _W.tl("Checking ..") + "</a>", $("finalDomainName").innerHTML = domainName + "." + domainNameTld, new Ajax.Request(ajax, {
            parameters: {
                pos: "checkdomain",
                domainname: domainName,
                domaintld: domainNameTld,
                cookie: document.cookie
            },
            onSuccess: handlerCheckDomain,
            onFailure: errFunc,
            bgRequest: !0,
            onException: exceptionFunc
        })
    }
}
function filterDomain(e) {
    var t = new Array;
    if ("" != e) {
        t.push(new RegExp("\\.com", "")), t.push(new RegExp("\\.net", "")), t.push(new RegExp("\\.org", "")), t.push(new RegExp("^.*:\\/\\/", "")), t.push(new RegExp("^.*@", "")), t.push(new RegExp("www\\.", "")), t.push(new RegExp("^\\-+", "")), t.push(new RegExp("\\+$", "")), t.push(new RegExp("[^a-zA-Z0-9-]", "g"));
        for (var i = 0; i < t.length; i++)
            e = e.replace(t[i], "");
        return e.substr(0, 63)
    }
    return ""
}
function handlerCheckDomain(e) {
    var t = $("publish_domain_sld") ? $("exportText"): $("domainNewDomain");
    e.responseText.match(/%%SUCCESS%%/) ? (t.down(".domain-check-success").show(), t.down(".domain-check-taken").hide(), t.down(".domain-check-checking").hide(), $("domainStatus").innerHTML = '<a style="color: green;">' + _W.tl("Available") + "</a>", $("finalDomainName").innerHTML = domainName + "." + domainNameTld) : e.responseText.match(/%%FAILURE%%/) && (t.down(".domain-check-success").hide(), t.down(".domain-check-checking").hide(), t.down(".domain-check-taken").show(), $("domainStatus").innerHTML = '<a style="color: red;">' + _W.tl("Not Available") + "</a>", $("finalDomainName").update(""))
}
function showDomainCheckChecking(e) {
    e.down(".domain-check-success").hide(), e.down(".domain-check-taken").hide(), e.down(".domain-check-checking").show()
}
function showFeedback() {
    new Effect.Move("feedback", {
        y: 100,
        mode: "absolute"
    })
}
function hideFeedback() {
    new Effect.Move("feedback", {
        y: - 700,
        mode: "absolute"
    })
}
function giveFeedback() {
    new Ajax.Request(ajax, {
        parameters: {
            pos: "givefeedback",
            feedback: $("feedbackText").value,
            referral: $("referralText").value,
            cookie: document.cookie
        },
        onFailure: errFunc
    }), $("feedbackText").value = "", $("referralText").value = "", feedbackInit()
}
function feedbackInit() {
    new Effect.Move("feedback", {
        y: - 700,
        mode: "absolute"
    })
}
function flash_ready() {
    Weebly.Storage.flash.onready()
}
function close_adsense_dialog() {
    return Pages.go("main"), Weebly.AllDialogs.adsense_terms.close(), !1
}
function close_adsense_confirmation_dialog() {
    return Weebly.AllDialogs.adsense_confirmation.close(), !1
}
function open_adsense_confirmation_dialog() {
    var e = new Weebly.Dialog($("adsense_confirmation"), {
        modal: !0
    });
    e.open()
}
function getParameterByName(e) {
    e = e.replace(/[\[]/, "\\[").replace(/[\]]/, "\\]");
    var t = "[\\?&]" + e + "=([^&#]*)", i = new RegExp(t), n = i.exec(window.location.search);
    return null == n ? "" : decodeURIComponent(n[1].replace(/\+/g, " "))
}
var _wAMD;
if (function() {
    if (!_wAMD ||!_wAMD.requirejs) {
        _wAMD ? t = _wAMD : _wAMD = {};
        var e, t, i;
        !function(n) {
            function o(e, t) {
                return w.call(e, t)
            }
            function s(e, t) {
                var i, n, o, s, r, a, l, c, u, d, h = t && t.split("/"), p = y.map, f = p && p["*"] || {};
                if (e && "." === e.charAt(0))
                    if (t) {
                        for (h = h.slice(0, h.length - 1), e = h.concat(e.split("/")), c = 0; c < e.length; c += 1)
                            if (d = e[c], "." === d)
                                e.splice(c, 1), c -= 1;
                            else if (".." === d) {
                                if (1 === c && (".." === e[2] || ".." === e[0]))
                                    break;
                                    c > 0 && (e.splice(c - 1, 2), c -= 2)
                            }
                            e = e.join("/")
                    } else 
                        0 === e.indexOf("./") && (e = e.substring(2));
                if ((h || f) && p) {
                    for (i = e.split("/"), c = i.length; c > 0; c -= 1) {
                        if (n = i.slice(0, c).join("/"), h)
                            for (u = h.length; u > 0; u -= 1)
                                if (o = p[h.slice(0, u).join("/")], o && (o = o[n])) {
                                    s = o, r = c;
                                    break
                                }
                        if (s)
                            break;
                        !a && f && f[n] && (a = f[n], l = c)
                    }
                    !s && a && (s = a, r = l), s && (i.splice(0, r, s), e = i.join("/"))
                }
                return e
            }
            function r(e, t) {
                return function() {
                    return p.apply(n, _.call(arguments, 0).concat([e, t]))
                }
            }
            function a(e) {
                return function(t) {
                    return s(t, e)
                }
            }
            function l(e) {
                return function(t) {
                    g[e] = t
                }
            }
            function c(e) {
                if (o(v, e)) {
                    var t = v[e];
                    delete v[e], b[e]=!0, h.apply(n, t)
                }
                if (!o(g, e)&&!o(b, e))
                    throw new Error("No " + e);
                return g[e]
            }
            function u(e) {
                var t, i = e ? e.indexOf("!"): - 1;
                return i>-1 && (t = e.substring(0, i), e = e.substring(i + 1, e.length)), [t, e]
            }
            function d(e) {
                return function() {
                    return y && y.config && y.config[e] || {}
                }
            }
            var h, p, f, m, g = {}, v = {}, y = {}, b = {}, w = Object.prototype.hasOwnProperty, _ = [].slice;
            f = function(e, t) {
                var i, n = u(e), o = n[0];
                return e = n[1], o && (o = s(o, t), i = c(o)), o ? e = i && i.normalize ? i.normalize(e, a(t)) : s(e, t) : (e = s(e, t), n = u(e), o = n[0], e = n[1], o && (i = c(o))), {
                    f: o ? o + "!" + e: e,
                    n: e,
                    pr: o,
                    p: i
                }
            }, m = {
                require: function(e) {
                    return r(e)
                },
                exports: function(e) {
                    var t = g[e];
                    return "undefined" != typeof t ? t : g[e] = {}
                },
                module: function(e) {
                    return {
                        id: e,
                        uri: "",
                        exports: g[e],
                        config: d(e)
                    }
                }
            }, h = function(e, t, i, s) {
                var a, u, d, h, p, y, w = [];
                if (s = s || e, "function" == typeof i) {
                    for (t=!t.length && i.length ? ["require", "exports", "module"] : t, p = 0; p < t.length; p += 1)
                        if (h = f(t[p], s), u = h.f, "require" === u)
                            w[p] = m.require(e);
                        else if ("exports" === u)
                            w[p] = m.exports(e), y=!0;
                        else if ("module" === u)
                            a = w[p] = m.module(e);
                        else if (o(g, u) || o(v, u) || o(b, u))
                            w[p] = c(u);
                        else {
                            if (!h.p)
                                throw new Error(e + " missing " + u);
                                h.p.load(h.n, r(s, !0), l(u), {}), w[p] = g[u]
                        }
                    d = i.apply(g[e], w), e && (a && a.exports !== n && a.exports !== g[e] ? g[e] = a.exports : d === n && y || (g[e] = d))
                } else 
                    e && (g[e] = i)
            }, e = t = p = function(e, t, i, o, s) {
                return "string" == typeof e ? m[e] ? m[e](t) : c(f(e, t).f) : (e.splice || (y = e, t.splice ? (e = t, t = i, i = null) : e = n), t = t || function() {}, "function" == typeof i && (i = o, o = s), o ? h(n, e, t, i) : setTimeout(function() {
                    h(n, e, t, i)
                }, 4), p)
            }, p.config = function(e) {
                return y = e, y.deps && p(y.deps, y.callback), p
            }, i = function(e, t, i) {
                t.splice || (i = t, t = []), o(g, e) || o(v, e) || (v[e] = [e, t, i])
            }, i.amd = {
                jQuery: !0
            }
        }(), _wAMD.requirejs = e, _wAMD.require = t, _wAMD.define = i
    }
}(), _wAMD.define("vendor/almond", function() {}), function() {
    function e(e, t) {
        s > 0 && 0 >= r ? e() : t ? a.unshift(e) : a.push(e)
    }
    function t() {
        var e = Array.prototype.slice.apply(arguments);
        if (1 === e.length && "string" == typeof e[0]);
        else {
            var t;
            "function" == typeof e[e.length - 1] && (t = e.pop()), s++, r++, e.push(function() {
                var e;
                if (t && (e = t.apply(this, arguments)), r--, s > 0 && 0 >= r)
                    for (; a.length;)
                        a.shift()();
                return e
            })
        }
        return p.apply(this, e)
    }
    function i() {
        window.define = l, window.require = c, window.requirejs = u
    }
    function n() {
        Weebly.jQuery.noConflict(!0), window.jQuery = window.jQuery || Weebly.jQuery
    }
    function o() {
        i(), n()
    }
    _W = Weebly = window.Weebly || {}, _W.tl = _W.tl || function(e) {
        return e
    }, Weebly.ns = function(e, t) {
        for (var t = t || window, i = e.split("."), n = 0, o = i.length; o > n; n++) {
            var s = i[n];
            "string" != typeof s ? t = s : (t[s] || (t[s] = {}), t = t[s])
        }
        return t
    };
    var s = 0, r = 0, a = [];
    Weebly.ready = e;
    var l = window.define, c = window.require, u = window.requirejs, d = window._wAMD || {}, h = d.define || window.define, p = d.require || window.require, f = d.requirejs || window.requirejs;
    window.define = d.define = h, window.require = d.require = t, window.requirejs = d.requirejs = f, Weebly.relinquishAMD = i, Weebly.jQuery = jQuery, Weebly.relinquishJQuery = n;
    var m = jQuery.fn.ready;
    jQuery.fn.ready = function() {
        var t = this, i = arguments;
        return e(function() {
            m.apply(t, i)
        }), t
    }, Weebly.relinquish = o
}(), window.console || (window.console = {}), window.console.log || (window.console.log = function() {}, window.console.warn = window.console.log, window.console.info = window.console.log, window.console.error = window.console.log), _wAMD.define("init", function() {}), _wAMD.define("jquery", [], function() {
    return Weebly.jQuery
}), function(e) {
    !function(e, t) {
        "object" == typeof exports && exports ? module.exports = t : "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("public_legacy/mustache", t) : e.Mustache = t
    }(this, function() {
        function e(e, t) {
            return y.call(e, t)
        }
        function t(t) {
            return !e(f, t)
        }
        function i(e) {
            return e.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, "\\$&")
        }
        function n(e) {
            return String(e).replace(/[&<>"'\/]/g, function(e) {
                return _[e]
            })
        }
        function o(e) {
            this.string = e, this.tail = e, this.pos = 0
        }
        function s(e, t) {
            this.view = e, this.parent = t, this._cache = {}
        }
        function r() {
            this.clearCache()
        }
        function a(e, t, i, n) {
            for (var o, s, r, l = "", c = 0, u = e.length; u > c; ++c)
                switch (o = e[c], s = o[1], o[0]) {
                case"#":
                    if (r = i.lookup(s), "object" == typeof r)
                        if (w(r))
                            for (var h = 0, p = r.length; p > h; ++h)
                                l += a(o[4], t, i.push(r[h]), n);
                        else 
                            r && (l += a(o[4], t, i.push(r), n));
                else if ("function" == typeof r) {
                    var f = null == n ? null: n.slice(o[3], o[5]);
                    r = r.call(i.view, f, function(e) {
                        return t.render(e, i)
                    }), null != r && (l += r)
                } else 
                    r && (l += a(o[4], t, i, n));
                    break;
                case"^":
                    r = i.lookup(s), (!r || w(r) && 0 === r.length) && (l += a(o[4], t, i, n));
                    break;
                case">":
                    r = t.getPartial(s), "function" == typeof r && (l += r(i));
                    break;
                case"&":
                    r = i.lookup(s), null != r && (l += r);
                    break;
                case"name":
                    r = i.lookup(s), null != r && (l += d.escape(r));
                    break;
                case"text":
                    l += s
            }
            return l
        }
        function l(e) {
            for (var t, i = [], n = i, o = [], s = 0, r = e.length; r > s; ++s)
                switch (t = e[s], t[0]) {
                case"#":
                case"^":
                    o.push(t), n.push(t), n = t[4] = [];
                    break;
                case"/":
                    var a = o.pop();
                    a[5] = t[2], n = o.length > 0 ? o[o.length - 1][4] : i;
                    break;
                default:
                    n.push(t)
                }
            return i
        }
        function c(e) {
            for (var t, i, n = [], o = 0, s = e.length; s > o; ++o)
                t = e[o], t && ("text" === t[0] && i && "text" === i[0] ? (i[1] += t[1], i[3] = t[3]) : (i = t, n.push(t)));
            return n
        }
        function u(e) {
            return [new RegExp(i(e[0]) + "\\s*"), new RegExp("\\s*" + i(e[1]))]
        }
        var d = {};
        d.name = "mustache.js", d.version = "0.7.2", d.tags = ["{{", "}}"], d.Scanner = o, d.Context = s, d.Writer = r;
        var h = /\s*/, p = /\s+/, f = /\S/, m = /\s*=/, g = /\s*\}/, v = /#|\^|\/|>|\{|&|=|!/, y = RegExp.prototype.test, b = Object.prototype.toString, w = Array.isArray || function(e) {
            return "[object Array]" === b.call(e)
        }, _ = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
            "/": "&#x2F;"
        };
        d.escape = n, o.prototype.eos = function() {
            return "" === this.tail
        }, o.prototype.scan = function(e) {
            var t = this.tail.match(e);
            return t && 0 === t.index ? (this.tail = this.tail.substring(t[0].length), this.pos += t[0].length, t[0]) : ""
        }, o.prototype.scanUntil = function(e) {
            var t, i = this.tail.search(e);
            switch (i) {
            case - 1:
                t = this.tail, this.pos += this.tail.length, this.tail = "";
                break;
            case 0:
                t = "";
                break;
            default:
                t = this.tail.substring(0, i), this.tail = this.tail.substring(i), this.pos += i
            }
            return t
        }, s.make = function(e) {
            return e instanceof s ? e : new s(e)
        }, s.prototype.push = function(e) {
            return new s(e, this)
        }, s.prototype.lookup = function(e) {
            var t = this._cache[e];
            if (!t) {
                if ("." == e)
                    t = this.view;
                else 
                    for (var i = this; i;) {
                        if (e.indexOf(".") > 0) {
                            t = i.view;
                            for (var n = e.split("."), o = 0; t && o < n.length;)
                                t = t[n[o++]]
                        } else 
                            t = i.view[e];
                            if (null != t)
                                break;
                                i = i.parent
                    }
                this._cache[e] = t
            }
            return "function" == typeof t && (t = t.call(this.view)), t
        }, r.prototype.clearCache = function() {
            this._cache = {}, this._partialCache = {}
        }, r.prototype.compile = function(e, t) {
            var i = this._cache[e];
            if (!i) {
                var n = d.parse(e, t);
                i = this._cache[e] = this.compileTokens(n, e)
            }
            return i
        }, r.prototype.compilePartial = function(e, t, i) {
            var n = this.compile(t, i);
            return this._partialCache[e] = n, n
        }, r.prototype.getPartial = function(e) {
            return e in this._partialCache ||!this._loadPartial || this.compilePartial(e, this._loadPartial(e)), this._partialCache[e]
        }, r.prototype.compileTokens = function(e, t) {
            var i = this;
            return function(n, o) {
                if (o)
                    if ("function" == typeof o)
                        i._loadPartial = o;
                    else 
                        for (var r in o)
                            i.compilePartial(r, o[r]);
                return a(e, i, s.make(n), t)
            }
        }, r.prototype.render = function(e, t, i) {
            return this.compile(e)(t, i)
        }, d.parse = function(e, n) {
            function s() {
                if (S&&!M)
                    for (; x.length;)
                        delete k[x.pop()];
                else 
                    x = [];
                S=!1, M=!1
            }
            if (e = e || "", n = n || d.tags, "string" == typeof n && (n = n.split(p)), 2 !== n.length)
                throw new Error("Invalid tags: " + n.join(", "));
            for (var r, a, f, y, b, w = u(n), _ = new o(e), C = [], k = [], x = [], S=!1, M=!1; !_.eos();) {
                if (r = _.pos, f = _.scanUntil(w[0]))
                    for (var T = 0, D = f.length; D > T; ++T)
                        y = f.charAt(T), t(y) ? x.push(k.length) : M=!0, k.push(["text", y, r, r + 1]), r += 1, "\n" == y && s();
                if (!_.scan(w[0]))
                    break;
                if (S=!0, a = _.scan(v) || "name", _.scan(h), "=" === a ? (f = _.scanUntil(m), _.scan(m), _.scanUntil(w[1])) : "{" === a ? (f = _.scanUntil(new RegExp("\\s*" + i("}" + n[1]))), _.scan(g), _.scanUntil(w[1]), a = "&") : f = _.scanUntil(w[1]), !_.scan(w[1]))
                    throw new Error("Unclosed tag at " + _.pos);
                if (b = [a, f, r, _.pos], k.push(b), "#" === a || "^" === a)
                    C.push(b);
                else if ("/" === a) {
                    if (0 === C.length)
                        throw new Error('Unopened section "' + f + '" at ' + r);
                    var E = C.pop();
                    if (E[1] !== f)
                        throw new Error('Unclosed section "' + E[1] + '" at ' + r)
                    } else if ("name" === a || "{" === a || "&" === a)
                    M=!0;
                else if ("=" === a) {
                    if (n = f.split(p), 2 !== n.length)
                        throw new Error("Invalid tags at " + r + ": " + n.join(", "));
                    w = u(n)
                }
            }
            var E = C.pop();
            if (E)
                throw new Error('Unclosed section "' + E[1] + '" at ' + _.pos);
            return k = c(k), l(k)
        };
        var C = new r;
        return d.clearCache = function() {
            return C.clearCache()
        }, d.compile = function(e, t) {
            return C.compile(e, t)
        }, d.compilePartial = function(e, t, i) {
            return C.compilePartial(e, t, i)
        }, d.compileTokens = function(e, t) {
            return C.compileTokens(e, t)
        }, d.render = function(e, t, i) {
            return C.render(e, t, i)
        }, d.to_html = function(e, t, i, n) {
            var o = d.render(e, t, i);
            return "function" != typeof n ? o : (n(o), void 0)
        }, d
    }()), e.mustache = function(e, t, i) {
        return Mustache.render(e, t, i)
    }, e.fn.mustache = function(t, i) {
        return e(this).map(function(n, o) {
            var s = e.trim(e(o).html()), r = e.mustache(s, t, i);
            return e(r).get()
        })
    }
}(jQuery), function(e) {
    function t(e, t, i) {
        switch (arguments.length) {
        case 2:
            return null != e ? e : t;
        case 3:
            return null != e ? e : null != t ? t : i;
        default:
            throw new Error("Implement me")
        }
    }
    function i() {
        return {
            empty: !1,
            unusedTokens: [],
            unusedInput: [],
            overflow: - 2,
            charsLeftOver: 0,
            nullInput: !1,
            invalidMonth: null,
            invalidFormat: !1,
            userInvalidated: !1,
            iso: !1
        }
    }
    function n(e) {
        vt.suppressDeprecationWarnings===!1 && "undefined" != typeof console && console.warn && console.warn("Deprecation warning: " + e)
    }
    function o(e, t) {
        var i=!0;
        return d(function() {
            return i && (n(e), i=!1), t.apply(this, arguments)
        }, t)
    }
    function s(e, t) {
        pi[e] || (n(t), pi[e]=!0)
    }
    function r(e, t) {
        return function(i) {
            return f(e.call(this, i), t)
        }
    }
    function a(e, t) {
        return function(i) {
            return this.localeData().ordinal(e.call(this, i), t)
        }
    }
    function l() {}
    function c(e, t) {
        t!==!1 && A(e), h(this, e), this._d = new Date( + e._d)
    }
    function u(e) {
        var t = k(e), i = t.year || 0, n = t.quarter || 0, o = t.month || 0, s = t.week || 0, r = t.day || 0, a = t.hour || 0, l = t.minute || 0, c = t.second || 0, u = t.millisecond || 0;
        this._milliseconds =+ u + 1e3 * c + 6e4 * l + 36e5 * a, this._days =+ r + 7 * s, this._months =+ o + 3 * n + 12 * i, this._data = {}, this._locale = vt.localeData(), this._bubble()
    }
    function d(e, t) {
        for (var i in t)
            t.hasOwnProperty(i) && (e[i] = t[i]);
        return t.hasOwnProperty("toString") && (e.toString = t.toString), t.hasOwnProperty("valueOf") && (e.valueOf = t.valueOf), e
    }
    function h(e, t) {
        var i, n, o;
        if ("undefined" != typeof t._isAMomentObject && (e._isAMomentObject = t._isAMomentObject), "undefined" != typeof t._i && (e._i = t._i), "undefined" != typeof t._f && (e._f = t._f), "undefined" != typeof t._l && (e._l = t._l), "undefined" != typeof t._strict && (e._strict = t._strict), "undefined" != typeof t._tzm && (e._tzm = t._tzm), "undefined" != typeof t._isUTC && (e._isUTC = t._isUTC), "undefined" != typeof t._offset && (e._offset = t._offset), "undefined" != typeof t._pf && (e._pf = t._pf), "undefined" != typeof t._locale && (e._locale = t._locale), Pt.length > 0)
            for (i in Pt)
                n = Pt[i], o = t[n], "undefined" != typeof o && (e[n] = o);
        return e
    }
    function p(e) {
        return 0 > e ? Math.ceil(e) : Math.floor(e)
    }
    function f(e, t, i) {
        for (var n = "" + Math.abs(e), o = e >= 0; n.length < t;)
            n = "0" + n;
        return (o ? i ? "+" : "" : "-") + n
    }
    function m(e, t) {
        var i = {
            milliseconds: 0,
            months: 0
        };
        return i.months = t.month() - e.month() + 12 * (t.year() - e.year()), e.clone().add(i.months, "M").isAfter(t)&&--i.months, i.milliseconds =+ t-+e.clone().add(i.months, "M"), i
    }
    function g(e, t) {
        var i;
        return t = I(t, e), e.isBefore(t) ? i = m(e, t) : (i = m(t, e), i.milliseconds =- i.milliseconds, i.months =- i.months), i
    }
    function v(e, t) {
        return function(i, n) {
            var o, r;
            return null === n || isNaN( + n) || (s(t, "moment()." + t + "(period, number) is deprecated. Please use moment()." + t + "(number, period)."), r = i, i = n, n = r), i = "string" == typeof i?+i : i, o = vt.duration(i, n), y(this, o, e), this
        }
    }
    function y(e, t, i, n) {
        var o = t._milliseconds, s = t._days, r = t._months;
        n = null == n?!0 : n, o && e._d.setTime( + e._d + o * i), s && dt(e, "Date", ut(e, "Date") + s * i), r && ct(e, ut(e, "Month") + r * i), n && vt.updateOffset(e, s || r)
    }
    function b(e) {
        return "[object Array]" === Object.prototype.toString.call(e)
    }
    function w(e) {
        return "[object Date]" === Object.prototype.toString.call(e) || e instanceof Date
    }
    function _(e, t, i) {
        var n, o = Math.min(e.length, t.length), s = Math.abs(e.length - t.length), r = 0;
        for (n = 0; o > n; n++)(i && e[n] !== t[n] ||!i && S(e[n]) !== S(t[n])
            ) && r++;
        return r + s
    }
    function C(e) {
        if (e) {
            var t = e.toLowerCase().replace(/(.)s$/, "$1");
            e = ri[e] || ai[t] || t
        }
        return e
    }
    function k(e) {
        var t, i, n = {};
        for (i in e)
            e.hasOwnProperty(i) && (t = C(i), t && (n[t] = e[i]));
        return n
    }
    function x(t) {
        var i, n;
        if (0 === t.indexOf("week"))
            i = 7, n = "day";
        else {
            if (0 !== t.indexOf("month"))
                return;
            i = 12, n = "month"
        }
        vt[t] = function(o, s) {
            var r, a, l = vt._locale[t], c = [];
            if ("number" == typeof o && (s = o, o = e), a = function(e) {
                var t = vt().utc().set(n, e);
                return l.call(vt._locale, t, o || "")
            }, null != s)
                return a(s);
            for (r = 0; i > r; r++)
                c.push(a(r));
            return c
        }
    }
    function S(e) {
        var t =+ e, i = 0;
        return 0 !== t && isFinite(t) && (i = t >= 0 ? Math.floor(t) : Math.ceil(t)), i
    }
    function M(e, t) {
        return new Date(Date.UTC(e, t + 1, 0)).getUTCDate()
    }
    function T(e, t, i) {
        return st(vt([e, 11, 31 + t - i]), t, i).week
    }
    function D(e) {
        return E(e) ? 366 : 365
    }
    function E(e) {
        return 0 === e%4 && 0 !== e%100 || 0 === e%400
    }
    function A(e) {
        var t;
        e._a&&-2 === e._pf.overflow && (t = e._a[xt] < 0 || e._a[xt] > 11 ? xt : e._a[St] < 1 || e._a[St] > M(e._a[kt], e._a[xt]) ? St : e._a[Mt] < 0 || e._a[Mt] > 23 ? Mt : e._a[Tt] < 0 || e._a[Tt] > 59 ? Tt : e._a[Dt] < 0 || e._a[Dt] > 59 ? Dt : e._a[Et] < 0 || e._a[Et] > 999 ? Et : - 1, e._pf._overflowDayOfYear && (kt > t || t > St) && (t = St), e._pf.overflow = t)
    }
    function P(e) {
        return null == e._isValid && (e._isValid=!isNaN(e._d.getTime()) && e._pf.overflow < 0&&!e._pf.empty&&!e._pf.invalidMonth&&!e._pf.nullInput&&!e._pf.invalidFormat&&!e._pf.userInvalidated, e._strict && (e._isValid = e._isValid && 0 === e._pf.charsLeftOver && 0 === e._pf.unusedTokens.length)), e._isValid
    }
    function R(e) {
        return e ? e.toLowerCase().replace("_", "-") : e
    }
    function F(e) {
        for (var t, i, n, o, s = 0; s < e.length;) {
            for (o = R(e[s]).split("-"), t = o.length, i = R(e[s + 1]), i = i ? i.split("-") : null; t > 0;) {
                if (n = O(o.slice(0, t).join("-")))
                    return n;
                if (i && i.length >= t && _(o, i, !0) >= t - 1)
                    break;
                t--
            }
            s++
        }
        return null
    }
    function O(e) {
        var t = null;
        if (!At[e] && Rt)
            try {
                t = vt.locale(), require("./locale/" + e), vt.locale(t)
        } catch (i) {}
        return At[e]
    }
    function I(e, t) {
        return t._isUTC ? vt(e).zone(t._offset || 0) : vt(e).local()
    }
    function W(e) {
        return e.match(/\[[\s\S]/) ? e.replace(/^\[|\]$/g, "") : e.replace(/\\/g, "")
    }
    function N(e) {
        var t, i, n = e.match(Wt);
        for (t = 0, i = n.length; i > t; t++)
            n[t] = hi[n[t]] ? hi[n[t]] : W(n[t]);
        return function(o) {
            var s = "";
            for (t = 0; i > t; t++)
                s += n[t]instanceof Function ? n[t].call(o, e) : n[t];
            return s
        }
    }
    function z(e, t) {
        return e.isValid() ? (t = L(t, e.localeData()), li[t] || (li[t] = N(t)), li[t](e)) : e.localeData().invalidDate()
    }
    function L(e, t) {
        function i(e) {
            return t.longDateFormat(e) || e
        }
        var n = 5;
        for (Nt.lastIndex = 0; n >= 0 && Nt.test(e);)
            e = e.replace(Nt, i), Nt.lastIndex = 0, n -= 1;
        return e
    }
    function B(e, t) {
        var i, n = t._strict;
        switch (e) {
        case"Q":
            return Gt;
        case"DDDD":
            return Jt;
        case"YYYY":
        case"GGGG":
        case"gggg":
            return n ? Qt : Bt;
        case"Y":
        case"G":
        case"g":
            return Zt;
        case"YYYYYY":
        case"YYYYY":
        case"GGGGG":
        case"ggggg":
            return n ? Kt : $t;
        case"S":
            if (n)
                return Gt;
        case"SS":
            if (n)
                return Xt;
        case"SSS":
            if (n)
                return Jt;
        case"DDD":
            return Lt;
        case"MMM":
        case"MMMM":
        case"dd":
        case"ddd":
        case"dddd":
            return Ht;
        case"a":
        case"A":
            return t._locale._meridiemParse;
        case"X":
            return Ut;
        case"Z":
        case"ZZ":
            return qt;
        case"T":
            return Vt;
        case"SSSS":
            return jt;
        case"MM":
        case"DD":
        case"YY":
        case"GG":
        case"gg":
        case"HH":
        case"hh":
        case"mm":
        case"ss":
        case"ww":
        case"WW":
            return n ? Xt : zt;
        case"M":
        case"D":
        case"d":
        case"H":
        case"h":
        case"m":
        case"s":
        case"w":
        case"W":
        case"e":
        case"E":
            return zt;
        case"Do":
            return Yt;
        default:
            return i = new RegExp(X(G(e.replace("\\", "")), "i"))
        }
    }
    function $(e) {
        e = e || "";
        var t = e.match(qt) || [], i = t[t.length - 1] || [], n = (i + "").match(oi) || ["-", 0, 0], o =+ (60 * n[1]) + S(n[2]);
        return "+" === n[0]?-o : o
    }
    function j(e, t, i) {
        var n, o = i._a;
        switch (e) {
        case"Q":
            null != t && (o[xt] = 3 * (S(t) - 1));
            break;
        case"M":
        case"MM":
            null != t && (o[xt] = S(t) - 1);
            break;
        case"MMM":
        case"MMMM":
            n = i._locale.monthsParse(t), null != n ? o[xt] = n : i._pf.invalidMonth = t;
            break;
        case"D":
        case"DD":
            null != t && (o[St] = S(t));
            break;
        case"Do":
            null != t && (o[St] = S(parseInt(t, 10)));
            break;
        case"DDD":
        case"DDDD":
            null != t && (i._dayOfYear = S(t));
            break;
        case"YY":
            o[kt] = vt.parseTwoDigitYear(t);
            break;
        case"YYYY":
        case"YYYYY":
        case"YYYYYY":
            o[kt] = S(t);
            break;
        case"a":
        case"A":
            i._isPm = i._locale.isPM(t);
            break;
        case"H":
        case"HH":
        case"h":
        case"hh":
            o[Mt] = S(t);
            break;
        case"m":
        case"mm":
            o[Tt] = S(t);
            break;
        case"s":
        case"ss":
            o[Dt] = S(t);
            break;
        case"S":
        case"SS":
        case"SSS":
        case"SSSS":
            o[Et] = S(1e3 * ("0." + t));
            break;
        case"X":
            i._d = new Date(1e3 * parseFloat(t));
            break;
        case"Z":
        case"ZZ":
            i._useUTC=!0, i._tzm = $(t);
            break;
        case"dd":
        case"ddd":
        case"dddd":
            n = i._locale.weekdaysParse(t), null != n ? (i._w = i._w || {}, i._w.d = n) : i._pf.invalidWeekday = t;
            break;
        case"w":
        case"ww":
        case"W":
        case"WW":
        case"d":
        case"e":
        case"E":
            e = e.substr(0, 1);
        case"gggg":
        case"GGGG":
        case"GGGGG":
            e = e.substr(0, 2), t && (i._w = i._w || {}, i._w[e] = S(t));
            break;
        case"gg":
        case"GG":
            i._w = i._w || {}, i._w[e] = vt.parseTwoDigitYear(t)
        }
    }
    function H(e) {
        var i, n, o, s, r, a, l;
        i = e._w, null != i.GG || null != i.W || null != i.E ? (r = 1, a = 4, n = t(i.GG, e._a[kt], st(vt(), 1, 4).year), o = t(i.W, 1), s = t(i.E, 1)) : (r = e._locale._week.dow, a = e._locale._week.doy, n = t(i.gg, e._a[kt], st(vt(), r, a).year), o = t(i.w, 1), null != i.d ? (s = i.d, r > s&&++o) : s = null != i.e ? i.e + r : r), l = rt(n, o, s, a, r), e._a[kt] = l.year, e._dayOfYear = l.dayOfYear
    }
    function q(e) {
        var i, n, o, s, r = [];
        if (!e._d) {
            for (o = U(e), e._w && null == e._a[St] && null == e._a[xt] && H(e), e._dayOfYear && (s = t(e._a[kt], o[kt]), e._dayOfYear > D(s) && (e._pf._overflowDayOfYear=!0), n = tt(s, 0, e._dayOfYear), e._a[xt] = n.getUTCMonth(), e._a[St] = n.getUTCDate()), i = 0; 3 > i && null == e._a[i]; ++i)
                e._a[i] = r[i] = o[i];
            for (; 7 > i; i++)
                e._a[i] = r[i] = null == e._a[i] ? 2 === i ? 1 : 0 : e._a[i];
            e._d = (e._useUTC ? tt : et).apply(null, r), null != e._tzm && e._d.setUTCMinutes(e._d.getUTCMinutes() + e._tzm)
        }
    }
    function V(e) {
        var t;
        e._d || (t = k(e._i), e._a = [t.year, t.month, t.day, t.hour, t.minute, t.second, t.millisecond], q(e))
    }
    function U(e) {
        var t = new Date;
        return e._useUTC ? [t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate()] : [t.getFullYear(), t.getMonth(), t.getDate()]
    }
    function Y(e) {
        if (e._f === vt.ISO_8601)
            return Q(e), void 0;
        e._a = [], e._pf.empty=!0;
        var t, i, n, o, s, r = "" + e._i, a = r.length, l = 0;
        for (n = L(e._f, e._locale).match(Wt) || [], t = 0; t < n.length; t++)
            o = n[t], i = (r.match(B(o, e)) || [])[0], i && (s = r.substr(0, r.indexOf(i)), s.length > 0 && e._pf.unusedInput.push(s), r = r.slice(r.indexOf(i) + i.length), l += i.length), hi[o] ? (i ? e._pf.empty=!1 : e._pf.unusedTokens.push(o), j(o, i, e)) : e._strict&&!i && e._pf.unusedTokens.push(o);
        e._pf.charsLeftOver = a - l, r.length > 0 && e._pf.unusedInput.push(r), e._isPm && e._a[Mt] < 12 && (e._a[Mt] += 12), e._isPm===!1 && 12 === e._a[Mt] && (e._a[Mt] = 0), q(e), A(e)
    }
    function G(e) {
        return e.replace(/\\(\[)|\\(\])|\[([^\]\[]*)\]|\\(.)/g, function(e, t, i, n, o) {
            return t || i || n || o
        })
    }
    function X(e) {
        return e.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&")
    }
    function J(e) {
        var t, n, o, s, r;
        if (0 === e._f.length)
            return e._pf.invalidFormat=!0, e._d = new Date(0 / 0), void 0;
        for (s = 0; s < e._f.length; s++)
            r = 0, t = h({}, e), t._pf = i(), t._f = e._f[s], Y(t), P(t) && (r += t._pf.charsLeftOver, r += 10 * t._pf.unusedTokens.length, t._pf.score = r, (null == o || o > r) && (o = r, n = t));
        d(e, n || t)
    }
    function Q(e) {
        var t, i, n = e._i, o = ei.exec(n);
        if (o) {
            for (e._pf.iso=!0, t = 0, i = ii.length; i > t; t++)
                if (ii[t][1].exec(n)) {
                    e._f = ii[t][0] + (o[6] || " ");
                    break
                }
            for (t = 0, i = ni.length; i > t; t++)
                if (ni[t][1].exec(n)) {
                    e._f += ni[t][0];
                    break
                }
            n.match(qt) && (e._f += "Z"), Y(e)
        } else 
            e._isValid=!1
    }
    function K(e) {
        Q(e), e._isValid===!1 && (delete e._isValid, vt.createFromInputFallback(e))
    }
    function Z(t) {
        var i, n = t._i;
        n === e ? t._d = new Date : w(n) ? t._d = new Date( + n) : null !== (i = Ft.exec(n)) ? t._d = new Date( + i[1]) : "string" == typeof n ? K(t) : b(n) ? (t._a = n.slice(0), q(t)) : "object" == typeof n ? V(t) : "number" == typeof n ? t._d = new Date(n) : vt.createFromInputFallback(t)
    }
    function et(e, t, i, n, o, s, r) {
        var a = new Date(e, t, i, n, o, s, r);
        return 1970 > e && a.setFullYear(e), a
    }
    function tt(e) {
        var t = new Date(Date.UTC.apply(null, arguments));
        return 1970 > e && t.setUTCFullYear(e), t
    }
    function it(e, t) {
        if ("string" == typeof e)
            if (isNaN(e)) {
                if (e = t.weekdaysParse(e), "number" != typeof e)
                    return null
            } else 
                e = parseInt(e, 10);
        return e
    }
    function nt(e, t, i, n, o) {
        return o.relativeTime(t || 1, !!i, e, n)
    }
    function ot(e, t, i) {
        var n = vt.duration(e).abs(), o = Ct(n.as("s")), s = Ct(n.as("m")), r = Ct(n.as("h")), a = Ct(n.as("d")), l = Ct(n.as("M")), c = Ct(n.as("y")), u = o < ci.s && ["s", o] || 1 === s && ["m"] || s < ci.m && ["mm", s] || 1 === r && ["h"] || r < ci.h && ["hh", r] || 1 === a && ["d"] || a < ci.d && ["dd", a] || 1 === l && ["M"] || l < ci.M && ["MM", l] || 1 === c && ["y"] || ["yy", c];
        return u[2] = t, u[3] =+ e > 0, u[4] = i, nt.apply({}, u)
    }
    function st(e, t, i) {
        var n, o = i - t, s = i - e.day();
        return s > o && (s -= 7), o - 7 > s && (s += 7), n = vt(e).add(s, "d"), {
            week: Math.ceil(n.dayOfYear() / 7),
            year: n.year()
        }
    }
    function rt(e, t, i, n, o) {
        var s, r, a = tt(e, 0, 1).getUTCDay();
        return a = 0 === a ? 7 : a, i = null != i ? i : o, s = o - a + (a > n ? 7 : 0) - (o > a ? 7 : 0), r = 7 * (t - 1) + (i - o) + s + 1, {
            year: r > 0 ? e: e - 1,
            dayOfYear: r > 0 ? r: D(e - 1) + r
        }
    }
    function at(t) {
        var i = t._i, n = t._f;
        return t._locale = t._locale || vt.localeData(t._l), null === i || n === e && "" === i ? vt.invalid({
            nullInput: !0
        }) : ("string" == typeof i && (t._i = i = t._locale.preparse(i)), vt.isMoment(i) ? new c(i, !0) : (n ? b(n) ? J(t) : Y(t) : Z(t), new c(t)))
    }
    function lt(e, t) {
        var i, n;
        if (1 === t.length && b(t[0]) && (t = t[0]), !t.length)
            return vt();
        for (i = t[0], n = 1; n < t.length; ++n)
            t[n][e](i) && (i = t[n]);
        return i
    }
    function ct(e, t) {
        var i;
        return "string" == typeof t && (t = e.localeData().monthsParse(t), "number" != typeof t) ? e : (i = Math.min(e.date(), M(e.year(), t)), e._d["set" + (e._isUTC ? "UTC" : "") + "Month"](t, i), e)
    }
    function ut(e, t) {
        return e._d["get" + (e._isUTC ? "UTC" : "") + t]()
    }
    function dt(e, t, i) {
        return "Month" === t ? ct(e, i) : e._d["set" + (e._isUTC ? "UTC" : "") + t](i)
    }
    function ht(e, t) {
        return function(i) {
            return null != i ? (dt(this, e, i), vt.updateOffset(this, t), this) : ut(this, e)
        }
    }
    function pt(e) {
        return 400 * e / 146097
    }
    function ft(e) {
        return 146097 * e / 400
    }
    function mt(e) {
        vt.duration.fn[e] = function() {
            return this._data[e]
        }
    }
    function gt(e) {
        "undefined" == typeof ender && (yt = _t.moment, _t.moment = e ? o("Accessing Moment through the global scope is deprecated, and will be removed in an upcoming release.", vt) : vt)
    }
    for (var vt, yt, bt, wt = "2.8.1", _t = "undefined" != typeof global ? global : this, Ct = Math.round, kt = 0, xt = 1, St = 2, Mt = 3, Tt = 4, Dt = 5, Et = 6, At = {}, Pt = [], Rt = "undefined" != typeof module && module.exports, Ft = /^\/?Date\((\-?\d+)/i, Ot = /(\-)?(?:(\d*)\.)?(\d+)\:(\d+)(?:\:(\d+)\.?(\d{3})?)?/, It = /^(-)?P(?:(?:([0-9,.]*)Y)?(?:([0-9,.]*)M)?(?:([0-9,.]*)D)?(?:T(?:([0-9,.]*)H)?(?:([0-9,.]*)M)?(?:([0-9,.]*)S)?)?|([0-9,.]*)W)$/, Wt = /(\[[^\[]*\])|(\\)?(Mo|MM?M?M?|Do|DDDo|DD?D?D?|ddd?d?|do?|w[o|w]?|W[o|W]?|Q|YYYYYY|YYYYY|YYYY|YY|gg(ggg?)?|GG(GGG?)?|e|E|a|A|hh?|HH?|mm?|ss?|S{1,4}|X|zz?|ZZ?|.)/g, Nt = /(\[[^\[]*\])|(\\)?(LT|LL?L?L?|l{1,4})/g, zt = /\d\d?/, Lt = /\d{1,3}/, Bt = /\d{1,4}/, $t = /[+\-]?\d{1,6}/, jt = /\d+/, Ht = /[0-9]*['a-z\u00A0-\u05FF\u0700-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]+|[\u0600-\u06FF\/]+(\s*?[\u0600-\u06FF]+){1,2}/i, qt = /Z|[\+\-]\d\d:?\d\d/gi, Vt = /T/i, Ut = /[\+\-]?\d+(\.\d{1,3})?/, Yt = /\d{1,2}/, Gt = /\d/, Xt = /\d\d/, Jt = /\d{3}/, Qt = /\d{4}/, Kt = /[+-]?\d{6}/, Zt = /[+-]?\d+/, ei = /^\s*(?:[+-]\d{6}|\d{4})-(?:(\d\d-\d\d)|(W\d\d$)|(W\d\d-\d)|(\d\d\d))((T| )(\d\d(:\d\d(:\d\d(\.\d+)?)?)?)?([\+\-]\d\d(?::?\d\d)?|\s*Z)?)?$/, ti = "YYYY-MM-DDTHH:mm:ssZ", ii = [["YYYYYY-MM-DD", /[+-]\d{6}-\d{2}-\d{2}/], ["YYYY-MM-DD", /\d{4}-\d{2}-\d{2}/], ["GGGG-[W]WW-E", /\d{4}-W\d{2}-\d/], ["GGGG-[W]WW", /\d{4}-W\d{2}/], ["YYYY-DDD", /\d{4}-\d{3}/]], ni = [["HH:mm:ss.SSSS", /(T| )\d\d:\d\d:\d\d\.\d+/], ["HH:mm:ss", /(T| )\d\d:\d\d:\d\d/], ["HH:mm", /(T| )\d\d:\d\d/], ["HH", /(T| )\d\d/]], oi = /([\+\-]|\d\d)/gi, si = ("Date|Hours|Minutes|Seconds|Milliseconds".split("|")
        , {
        Milliseconds: 1, Seconds : 1e3, Minutes : 6e4, Hours : 36e5, Days : 864e5, Months : 2592e6, Years : 31536e6
    }), ri = {
        ms: "millisecond",
        s: "second",
        m: "minute",
        h: "hour",
        d: "day",
        D: "date",
        w: "week",
        W: "isoWeek",
        M: "month",
        Q: "quarter",
        y: "year",
        DDD: "dayOfYear",
        e: "weekday",
        E: "isoWeekday",
        gg: "weekYear",
        GG: "isoWeekYear"
    }, ai = {
        dayofyear: "dayOfYear",
        isoweekday: "isoWeekday",
        isoweek: "isoWeek",
        weekyear: "weekYear",
        isoweekyear: "isoWeekYear"
    }, li = {}, ci = {
        s: 45,
        m: 45,
        h: 22,
        d: 26,
        M: 11
    }, ui = "DDD w W M D d".split(" "), di = "M D H h m s w W".split(" "), hi = {
        M: function() {
            return this.month() + 1
        },
        MMM: function(e) {
            return this.localeData().monthsShort(this, e)
        },
        MMMM: function(e) {
            return this.localeData().months(this, e)
        },
        D: function() {
            return this.date()
        },
        DDD: function() {
            return this.dayOfYear()
        },
        d: function() {
            return this.day()
        },
        dd: function(e) {
            return this.localeData().weekdaysMin(this, e)
        },
        ddd: function(e) {
            return this.localeData().weekdaysShort(this, e)
        },
        dddd: function(e) {
            return this.localeData().weekdays(this, e)
        },
        w: function() {
            return this.week()
        },
        W: function() {
            return this.isoWeek()
        },
        YY: function() {
            return f(this.year()%100, 2)
        },
        YYYY: function() {
            return f(this.year(), 4)
        },
        YYYYY: function() {
            return f(this.year(), 5)
        },
        YYYYYY: function() {
            var e = this.year(), t = e >= 0 ? "+": "-";
            return t + f(Math.abs(e), 6)
        },
        gg: function() {
            return f(this.weekYear()%100, 2)
        },
        gggg: function() {
            return f(this.weekYear(), 4)
        },
        ggggg: function() {
            return f(this.weekYear(), 5)
        },
        GG: function() {
            return f(this.isoWeekYear()%100, 2)
        },
        GGGG: function() {
            return f(this.isoWeekYear(), 4)
        },
        GGGGG: function() {
            return f(this.isoWeekYear(), 5)
        },
        e: function() {
            return this.weekday()
        },
        E: function() {
            return this.isoWeekday()
        },
        a: function() {
            return this.localeData().meridiem(this.hours(), this.minutes(), !0)
        },
        A: function() {
            return this.localeData().meridiem(this.hours(), this.minutes(), !1)
        },
        H: function() {
            return this.hours()
        },
        h: function() {
            return this.hours()%12 || 12
        },
        m: function() {
            return this.minutes()
        },
        s: function() {
            return this.seconds()
        },
        S: function() {
            return S(this.milliseconds() / 100)
        },
        SS: function() {
            return f(S(this.milliseconds() / 10), 2)
        },
        SSS: function() {
            return f(this.milliseconds(), 3)
        },
        SSSS: function() {
            return f(this.milliseconds(), 3)
        },
        Z: function() {
            var e =- this.zone(), t = "+";
            return 0 > e && (e =- e, t = "-"), t + f(S(e / 60), 2) + ":" + f(S(e)%60, 2)
        },
        ZZ: function() {
            var e =- this.zone(), t = "+";
            return 0 > e && (e =- e, t = "-"), t + f(S(e / 60), 2) + f(S(e)%60, 2)
        },
        z: function() {
            return this.zoneAbbr()
        },
        zz: function() {
            return this.zoneName()
        },
        X: function() {
            return this.unix()
        },
        Q: function() {
            return this.quarter()
        }
    }, pi = {}, fi = ["months", "monthsShort", "weekdays", "weekdaysShort", "weekdaysMin"];
    ui.length;
    )bt = ui.pop(), hi[bt + "o"] = a(hi[bt], bt);
    for (; di.length;)
        bt = di.pop(), hi[bt + bt] = r(hi[bt], 2);
    hi.DDDD = r(hi.DDD, 3), d(l.prototype, {
        set: function(e) {
            var t, i;
            for (i in e)
                t = e[i], "function" == typeof t ? this[i] = t : this["_" + i] = t
        },
        _months: "January_February_March_April_May_June_July_August_September_October_November_December".split("_"),
        months: function(e) {
            return this._months[e.month()]
        },
        _monthsShort: "Jan_Feb_Mar_Apr_May_Jun_Jul_Aug_Sep_Oct_Nov_Dec".split("_"),
        monthsShort: function(e) {
            return this._monthsShort[e.month()]
        },
        monthsParse: function(e) {
            var t, i, n;
            for (this._monthsParse || (this._monthsParse = []), t = 0; 12 > t; t++)
                if (this._monthsParse[t] || (i = vt.utc([2e3, t]), n = "^" + this.months(i, "") + "|^" + this.monthsShort(i, ""), this._monthsParse[t] = new RegExp(n.replace(".", ""), "i")), this._monthsParse[t].test(e))
                    return t
        },
        _weekdays: "Sunday_Monday_Tuesday_Wednesday_Thursday_Friday_Saturday".split("_"),
        weekdays: function(e) {
            return this._weekdays[e.day()]
        },
        _weekdaysShort: "Sun_Mon_Tue_Wed_Thu_Fri_Sat".split("_"),
        weekdaysShort: function(e) {
            return this._weekdaysShort[e.day()]
        },
        _weekdaysMin: "Su_Mo_Tu_We_Th_Fr_Sa".split("_"),
        weekdaysMin: function(e) {
            return this._weekdaysMin[e.day()]
        },
        weekdaysParse: function(e) {
            var t, i, n;
            for (this._weekdaysParse || (this._weekdaysParse = []), t = 0; 7 > t; t++)
                if (this._weekdaysParse[t] || (i = vt([2e3, 1]).day(t), n = "^" + this.weekdays(i, "") + "|^" + this.weekdaysShort(i, "") + "|^" + this.weekdaysMin(i, ""), this._weekdaysParse[t] = new RegExp(n.replace(".", ""), "i")), this._weekdaysParse[t].test(e))
                    return t
        },
        _longDateFormat: {
            LT: "h:mm A",
            L: "MM/DD/YYYY",
            LL: "MMMM D, YYYY",
            LLL: "MMMM D, YYYY LT",
            LLLL: "dddd, MMMM D, YYYY LT"
        },
        longDateFormat: function(e) {
            var t = this._longDateFormat[e];
            return !t && this._longDateFormat[e.toUpperCase()] && (t = this._longDateFormat[e.toUpperCase()].replace(/MMMM|MM|DD|dddd/g, function(e) {
                return e.slice(1)
            }), this._longDateFormat[e] = t), t
        },
        isPM: function(e) {
            return "p" === (e + "").toLowerCase().charAt(0)
        },
        _meridiemParse: /[ap]\.?m?\.?/i,
        meridiem: function(e, t, i) {
            return e > 11 ? i ? "pm" : "PM" : i ? "am" : "AM"
        },
        _calendar: {
            sameDay: "[Today at] LT",
            nextDay: "[Tomorrow at] LT",
            nextWeek: "dddd [at] LT",
            lastDay: "[Yesterday at] LT",
            lastWeek: "[Last] dddd [at] LT",
            sameElse: "L"
        },
        calendar: function(e, t) {
            var i = this._calendar[e];
            return "function" == typeof i ? i.apply(t) : i
        },
        _relativeTime: {
            future: "in %s",
            past: "%s ago",
            s: "a few seconds",
            m: "a minute",
            mm: "%d minutes",
            h: "an hour",
            hh: "%d hours",
            d: "a day",
            dd: "%d days",
            M: "a month",
            MM: "%d months",
            y: "a year",
            yy: "%d years"
        },
        relativeTime: function(e, t, i, n) {
            var o = this._relativeTime[i];
            return "function" == typeof o ? o(e, t, i, n) : o.replace(/%d/i, e)
        },
        pastFuture: function(e, t) {
            var i = this._relativeTime[e > 0 ? "future": "past"];
            return "function" == typeof i ? i(t) : i.replace(/%s/i, t)
        },
        ordinal: function(e) {
            return this._ordinal.replace("%d", e)
        },
        _ordinal: "%d",
        preparse: function(e) {
            return e
        },
        postformat: function(e) {
            return e
        },
        week: function(e) {
            return st(e, this._week.dow, this._week.doy).week
        },
        _week: {
            dow: 0,
            doy: 6
        },
        _invalidDate: "Invalid date",
        invalidDate: function() {
            return this._invalidDate
        }
    }), vt = function(t, n, o, s) {
        var r;
        return "boolean" == typeof o && (s = o, o = e), r = {}, r._isAMomentObject=!0, r._i = t, r._f = n, r._l = o, r._strict = s, r._isUTC=!1, r._pf = i(), at(r)
    }, vt.suppressDeprecationWarnings=!1, vt.createFromInputFallback = o("moment construction falls back to js Date. This is discouraged and will be removed in upcoming major release. Please refer to https://github.com/moment/moment/issues/1407 for more info.", function(e) {
        e._d = new Date(e._i)
    }), vt.min = function() {
        var e = [].slice.call(arguments, 0);
        return lt("isBefore", e)
    }, vt.max = function() {
        var e = [].slice.call(arguments, 0);
        return lt("isAfter", e)
    }, vt.utc = function(t, n, o, s) {
        var r;
        return "boolean" == typeof o && (s = o, o = e), r = {}, r._isAMomentObject=!0, r._useUTC=!0, r._isUTC=!0, r._l = o, r._i = t, r._f = n, r._strict = s, r._pf = i(), at(r).utc()
    }, vt.unix = function(e) {
        return vt(1e3 * e)
    }, vt.duration = function(e, t) {
        var i, n, o, s, r = e, a = null;
        return vt.isDuration(e) ? r = {
            ms: e._milliseconds,
            d: e._days,
            M: e._months
        } : "number" == typeof e ? (r = {}, t ? r[t] = e : r.milliseconds = e) : (a = Ot.exec(e)) ? (i = "-" === a[1]?-1 : 1, r = {
            y : 0, d : S(a[St]) * i, h : S(a[Mt]) * i, m : S(a[Tt]) * i, s : S(a[Dt]) * i, ms : S(a[Et]) * i
        }) : (a = It.exec(e)) ? (i = "-" === a[1]?-1 : 1, o = function(e) {
            var t = e && parseFloat(e.replace(",", "."));
            return (isNaN(t) ? 0 : t) * i
        }, r = {
            y:
            o(a[2]),
            M:
            o(a[3]),
            d:
            o(a[4]),
            h:
            o(a[5]),
            m:
            o(a[6]),
            s:
            o(a[7]),
            w:
            o(a[8])
        }) : "object" == typeof r && ("from"in r || "to"in r) && (s = g(vt(r.from), vt(r.to)), r = {}, r.ms = s.milliseconds, r.M = s.months), n = new u(r), vt.isDuration(e) && e.hasOwnProperty("_locale") && (n._locale = e._locale), n
    }, vt.version = wt, vt.defaultFormat = ti, vt.ISO_8601 = function() {}, vt.momentProperties = Pt, vt.updateOffset = function() {}, vt.relativeTimeThreshold = function(t, i) {
        return ci[t] === e?!1 : i === e ? ci[t] : (ci[t] = i, !0)
    }, vt.lang = o("moment.lang is deprecated. Use moment.locale instead.", function(e, t) {
        return vt.locale(e, t)
    }), vt.locale = function(e, t) {
        var i;
        return e && (i = "undefined" != typeof t ? vt.defineLocale(e, t) : vt.localeData(e), i && (vt.duration._locale = vt._locale = i)), vt._locale._abbr
    }, vt.defineLocale = function(e, t) {
        return null !== t ? (t.abbr = e, At[e] || (At[e] = new l), At[e].set(t), vt.locale(e), At[e]) : (delete At[e], null)
    }, vt.langData = o("moment.langData is deprecated. Use moment.localeData instead.", function(e) {
        return vt.localeData(e)
    }), vt.localeData = function(e) {
        var t;
        if (e && e._locale && e._locale._abbr && (e = e._locale._abbr), !e)
            return vt._locale;
        if (!b(e)) {
            if (t = O(e))
                return t;
            e = [e]
        }
        return F(e)
    }, vt.isMoment = function(e) {
        return e instanceof c || null != e && e.hasOwnProperty("_isAMomentObject")
    }, vt.isDuration = function(e) {
        return e instanceof u
    };
    for (bt = fi.length - 1; bt >= 0; --bt)
        x(fi[bt]);
    vt.normalizeUnits = function(e) {
        return C(e)
    }, vt.invalid = function(e) {
        var t = vt.utc(0 / 0);
        return null != e ? d(t._pf, e) : t._pf.userInvalidated=!0, t
    }, vt.parseZone = function() {
        return vt.apply(null, arguments).parseZone()
    }, vt.parseTwoDigitYear = function(e) {
        return S(e) + (S(e) > 68 ? 1900 : 2e3)
    }, d(vt.fn = c.prototype, {
        clone: function() {
            return vt(this)
        },
        valueOf: function() {
            return + this._d + 6e4 * (this._offset || 0)
        },
        unix: function() {
            return Math.floor( + this / 1e3)
        },
        toString: function() {
            return this.clone().locale("en").format("ddd MMM DD YYYY HH:mm:ss [GMT]ZZ")
        },
        toDate: function() {
            return this._offset ? new Date( + this) : this._d
        },
        toISOString: function() {
            var e = vt(this).utc();
            return 0 < e.year() && e.year() <= 9999 ? z(e, "YYYY-MM-DD[T]HH:mm:ss.SSS[Z]") : z(e, "YYYYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
        },
        toArray: function() {
            var e = this;
            return [e.year(), e.month(), e.date(), e.hours(), e.minutes(), e.seconds(), e.milliseconds()]
        },
        isValid: function() {
            return P(this)
        },
        isDSTShifted: function() {
            return this._a ? this.isValid() && _(this._a, (this._isUTC ? vt.utc(this._a) : vt(this._a)).toArray()) > 0 : !1
        },
        parsingFlags: function() {
            return d({}, this._pf)
        },
        invalidAt: function() {
            return this._pf.overflow
        },
        utc: function(e) {
            return this.zone(0, e)
        },
        local: function(e) {
            return this._isUTC && (this.zone(0, e), this._isUTC=!1, e && this.add(this._d.getTimezoneOffset(), "m")), this
        },
        format: function(e) {
            var t = z(this, e || vt.defaultFormat);
            return this.localeData().postformat(t)
        },
        add: v(1, "add"),
        subtract: v( - 1, "subtract"),
        diff: function(e, t, i) {
            var n, o, s = I(e, this), r = 6e4 * (this.zone() - s.zone());
            return t = C(t), "year" === t || "month" === t ? (n = 432e5 * (this.daysInMonth() + s.daysInMonth()), o = 12 * (this.year() - s.year()) + (this.month() - s.month()), o += (this - vt(this).startOf("month") - (s - vt(s).startOf("month"))) / n, o -= 6e4 * (this.zone() - vt(this).startOf("month").zone() - (s.zone() - vt(s).startOf("month").zone())) / n, "year" === t && (o/=12)) : (n = this - s, o = "second" === t ? n / 1e3 : "minute" === t ? n / 6e4 : "hour" === t ? n / 36e5 : "day" === t ? (n - r) / 864e5 : "week" === t ? (n - r) / 6048e5 : n), i ? o : p(o)
        },
        from: function(e, t) {
            return vt.duration({
                to: this,
                from: e
            }).locale(this.locale()).humanize(!t)
        },
        fromNow: function(e) {
            return this.from(vt(), e)
        },
        calendar: function(e) {
            var t = e || vt(), i = I(t, this).startOf("day"), n = this.diff(i, "days", !0), o =- 6 > n ? "sameElse" : - 1 > n ? "lastWeek" : 0 > n ? "lastDay" : 1 > n ? "sameDay" : 2 > n ? "nextDay" : 7 > n ? "nextWeek" : "sameElse";
            return this.format(this.localeData().calendar(o, this))
        },
        isLeapYear: function() {
            return E(this.year())
        },
        isDST: function() {
            return this.zone() < this.clone().month(0).zone() || this.zone() < this.clone().month(5).zone()
        },
        day: function(e) {
            var t = this._isUTC ? this._d.getUTCDay(): this._d.getDay();
            return null != e ? (e = it(e, this.localeData()), this.add(e - t, "d")) : t
        },
        month: ht("Month", !0),
        startOf: function(e) {
            switch (e = C(e)) {
            case"year":
                this.month(0);
            case"quarter":
            case"month":
                this.date(1);
            case"week":
            case"isoWeek":
            case"day":
                this.hours(0);
            case"hour":
                this.minutes(0);
            case"minute":
                this.seconds(0);
            case"second":
                this.milliseconds(0)
            }
            return "week" === e ? this.weekday(0) : "isoWeek" === e && this.isoWeekday(1), "quarter" === e && this.month(3 * Math.floor(this.month() / 3)), this
        },
        endOf: function(e) {
            return e = C(e), this.startOf(e).add(1, "isoWeek" === e ? "week" : e).subtract(1, "ms")
        },
        isAfter: function(e, t) {
            return t = "undefined" != typeof t ? t : "millisecond", + this.clone().startOf(t)>+vt(e).startOf(t)
        },
        isBefore: function(e, t) {
            return t = "undefined" != typeof t ? t : "millisecond", + this.clone().startOf(t)<+vt(e).startOf(t)
        },
        isSame: function(e, t) {
            return t = t || "ms", + this.clone().startOf(t) ===+ I(e, this).startOf(t)
        },
        min: o("moment().min is deprecated, use moment.min instead. https://github.com/moment/moment/issues/1548", function(e) {
            return e = vt.apply(null, arguments), this > e ? this : e
        }),
        max: o("moment().max is deprecated, use moment.max instead. https://github.com/moment/moment/issues/1548", function(e) {
            return e = vt.apply(null, arguments), e > this ? this : e
        }),
        zone: function(e, t) {
            var i, n = this._offset || 0;
            return null == e ? this._isUTC ? n : this._d.getTimezoneOffset() : ("string" == typeof e && (e = $(e)), Math.abs(e) < 16 && (e = 60 * e), !this._isUTC && t && (i = this._d.getTimezoneOffset()), this._offset = e, this._isUTC=!0, null != i && this.subtract(i, "m"), n !== e && (!t || this._changeInProgress ? y(this, vt.duration(n - e, "m"), 1, !1) : this._changeInProgress || (this._changeInProgress=!0, vt.updateOffset(this, !0), this._changeInProgress = null)), this)
        },
        zoneAbbr: function() {
            return this._isUTC ? "UTC" : ""
        },
        zoneName: function() {
            return this._isUTC ? "Coordinated Universal Time" : ""
        },
        parseZone: function() {
            return this._tzm ? this.zone(this._tzm) : "string" == typeof this._i && this.zone(this._i), this
        },
        hasAlignedHourOffset: function(e) {
            return e = e ? vt(e).zone() : 0, 0 === (this.zone() - e)%60
        },
        daysInMonth: function() {
            return M(this.year(), this.month())
        },
        dayOfYear: function(e) {
            var t = Ct((vt(this).startOf("day") - vt(this).startOf("year")) / 864e5) + 1;
            return null == e ? t : this.add(e - t, "d")
        },
        quarter: function(e) {
            return null == e ? Math.ceil((this.month() + 1) / 3) : this.month(3 * (e - 1) + this.month()%3)
        },
        weekYear: function(e) {
            var t = st(this, this.localeData()._week.dow, this.localeData()._week.doy).year;
            return null == e ? t : this.add(e - t, "y")
        },
        isoWeekYear: function(e) {
            var t = st(this, 1, 4).year;
            return null == e ? t : this.add(e - t, "y")
        },
        week: function(e) {
            var t = this.localeData().week(this);
            return null == e ? t : this.add(7 * (e - t), "d")
        },
        isoWeek: function(e) {
            var t = st(this, 1, 4).week;
            return null == e ? t : this.add(7 * (e - t), "d")
        },
        weekday: function(e) {
            var t = (this.day() + 7 - this.localeData()._week.dow)%7;
            return null == e ? t : this.add(e - t, "d")
        },
        isoWeekday: function(e) {
            return null == e ? this.day() || 7 : this.day(this.day()%7 ? e : e - 7)
        },
        isoWeeksInYear: function() {
            return T(this.year(), 1, 4)
        },
        weeksInYear: function() {
            var e = this.localeData()._week;
            return T(this.year(), e.dow, e.doy)
        },
        get: function(e) {
            return e = C(e), this[e]()
        },
        set: function(e, t) {
            return e = C(e), "function" == typeof this[e] && this[e](t), this
        },
        locale: function(t) {
            return t === e ? this._locale._abbr : (this._locale = vt.localeData(t), this)
        },
        lang: o("moment().lang() is deprecated. Use moment().localeData() instead.", function(t) {
            return t === e ? this.localeData() : (this._locale = vt.localeData(t), this)
        }),
        localeData: function() {
            return this._locale
        }
    }), vt.fn.millisecond = vt.fn.milliseconds = ht("Milliseconds", !1), vt.fn.second = vt.fn.seconds = ht("Seconds", !1), vt.fn.minute = vt.fn.minutes = ht("Minutes", !1), vt.fn.hour = vt.fn.hours = ht("Hours", !0), vt.fn.date = ht("Date", !0), vt.fn.dates = o("dates accessor is deprecated. Use date instead.", ht("Date", !0)), vt.fn.year = ht("FullYear", !0), vt.fn.years = o("years accessor is deprecated. Use year instead.", ht("FullYear", !0)), vt.fn.days = vt.fn.day, vt.fn.months = vt.fn.month, vt.fn.weeks = vt.fn.week, vt.fn.isoWeeks = vt.fn.isoWeek, vt.fn.quarters = vt.fn.quarter, vt.fn.toJSON = vt.fn.toISOString, d(vt.duration.fn = u.prototype, {
        _bubble: function() {
            var e, t, i, n = this._milliseconds, o = this._days, s = this._months, r = this._data, a = 0;
            r.milliseconds = n%1e3, e = p(n / 1e3), r.seconds = e%60, t = p(e / 60), r.minutes = t%60, i = p(t / 60), r.hours = i%24, o += p(i / 24), a = p(pt(o)), o -= p(ft(a)), s += p(o / 30), o%=30, a += p(s / 12), s%=12, r.days = o, r.months = s, r.years = a
        },
        abs: function() {
            return this._milliseconds = Math.abs(this._milliseconds), this._days = Math.abs(this._days), this._months = Math.abs(this._months), this._data.milliseconds = Math.abs(this._data.milliseconds), this._data.seconds = Math.abs(this._data.seconds), this._data.minutes = Math.abs(this._data.minutes), this._data.hours = Math.abs(this._data.hours), this._data.months = Math.abs(this._data.months), this._data.years = Math.abs(this._data.years), this
        },
        weeks: function() {
            return p(this.days() / 7)
        },
        valueOf: function() {
            return this._milliseconds + 864e5 * this._days + 2592e6 * (this._months%12) + 31536e6 * S(this._months / 12)
        },
        humanize: function(e) {
            var t = ot(this, !e, this.localeData());
            return e && (t = this.localeData().pastFuture( + this, t)), this.localeData().postformat(t)
        },
        add: function(e, t) {
            var i = vt.duration(e, t);
            return this._milliseconds += i._milliseconds, this._days += i._days, this._months += i._months, this._bubble(), this
        },
        subtract: function(e, t) {
            var i = vt.duration(e, t);
            return this._milliseconds -= i._milliseconds, this._days -= i._days, this._months -= i._months, this._bubble(), this
        },
        get: function(e) {
            return e = C(e), this[e.toLowerCase() + "s"]()
        },
        as: function(e) {
            var t, i;
            if (e = C(e), t = this._days + this._milliseconds / 864e5, "month" === e || "year" === e)
                return i = this._months + 12 * pt(t), "month" === e ? i : i / 12;
            switch (t += ft(this._months / 12), e) {
            case"week":
                return t / 7;
            case"day":
                return t;
            case"hour":
                return 24 * t;
            case"minute":
                return 60 * 24 * t;
            case"second":
                return 60 * 60 * 24 * t;
            case"millisecond":
                return 1e3 * 60 * 60 * 24 * t;
            default:
                throw new Error("Unknown unit " + e)
            }
        },
        lang: vt.fn.lang,
        locale: vt.fn.locale,
        toIsoString: o("toIsoString() is deprecated. Please use toISOString() instead (notice the capitals)", function() {
            return this.toISOString()
        }),
        toISOString: function() {
            var e = Math.abs(this.years()), t = Math.abs(this.months()), i = Math.abs(this.days()), n = Math.abs(this.hours()), o = Math.abs(this.minutes()), s = Math.abs(this.seconds() + this.milliseconds() / 1e3);
            return this.asSeconds() ? (this.asSeconds() < 0 ? "-" : "") + "P" + (e ? e + "Y" : "") + (t ? t + "M" : "") + (i ? i + "D" : "") + (n || o || s ? "T" : "") + (n ? n + "H" : "") + (o ? o + "M" : "") + (s ? s + "S" : "") : "P0D"
        },
        localeData: function() {
            return this._locale
        }
    });
    for (bt in si)
        si.hasOwnProperty(bt) && mt(bt.toLowerCase());
    vt.duration.fn.asMilliseconds = function() {
        return this.as("ms")
    }, vt.duration.fn.asSeconds = function() {
        return this.as("s")
    }, vt.duration.fn.asMinutes = function() {
        return this.as("m")
    }, vt.duration.fn.asHours = function() {
        return this.as("h")
    }, vt.duration.fn.asDays = function() {
        return this.as("d")
    }, vt.duration.fn.asWeeks = function() {
        return this.as("weeks")
    }, vt.duration.fn.asMonths = function() {
        return this.as("M")
    }, vt.duration.fn.asYears = function() {
        return this.as("y")
    }, vt.locale("en", {
        ordinal: function(e) {
            var t = e%10, i = 1 === S(e%100 / 10) ? "th": 1 === t ? "st": 2 === t ? "nd": 3 === t ? "rd": "th";
            return e + i
        }
    }), Rt ? module.exports = vt : "function" == typeof _wAMD.define && _wAMD.define.amd ? (_wAMD.define("moment", ["require", "exports", "module"], function(e, t, i) {
        return i.config && i.config() && i.config().noGlobal===!0 && (_t.moment = yt), vt
    }), gt(!0)) : gt()
}.call(this), _wAMD.define("mustache", ["jquery", "public_legacy/mustache", "moment"], function(e, t, i) {
    window.Mustache = t;
    var n = {
        date: "MMMM D, YYYY",
        shortDate: "M/D/YY",
        longDate: "M/D/YYYY",
        time: "h:mm a"
    }, o = function(e) {
        return function(t, o) {
            var s = o(t), r = i.unix(s);
            return r.format(n[e])
        }
    }, s = {
        buildTime: "undefined" != typeof buildTime ? buildTime: null,
        ASSETS_BASE: "undefined" != typeof ASSETS_BASE ? ASSETS_BASE: null,
        tl: function() {
            return function(e, t) {
                return t(_W.tl(e))
            }
        },
        esc_attr: function() {
            return function(e, t) {
                return _.escape(t(e))
            }
        },
        fmt: {
            date: function() {
                return o("date")
            },
            shortDate: function() {
                return o("shortDate")
            },
            longDate: function() {
                return o("longDate")
            },
            time: function() {
                return o("time")
            }
        }
    };
    t.origRender = t.render, t.render = function(i, n) {
        return e.extend(n, s), t.origRender.apply(this, arguments)
    };
    var r = t.Writer.prototype;
    return r.origCompile = r.compile, r.compile = function(t) {
        var i = r.origCompile.apply(this, [t]);
        return function() {
            var t = Array.prototype.slice.call(arguments);
            return t[0] = e.extend(t[0] || {}, s), i.apply(this, t)
        }
    }, t
}), function() {
    var e = this, t = e._, i = {}, n = Array.prototype, o = Object.prototype, s = Function.prototype, r = n.push, a = n.slice, l = n.concat, c = o.toString, u = o.hasOwnProperty, d = n.forEach, h = n.map, p = n.reduce, f = n.reduceRight, m = n.filter, g = n.every, v = n.some, y = n.indexOf, b = n.lastIndexOf, w = Array.isArray, _ = Object.keys, C = s.bind, k = function(e) {
        return e instanceof k ? e : this instanceof k ? (this._wrapped = e, void 0) : new k(e)
    };
    "undefined" != typeof exports ? ("undefined" != typeof module && module.exports && (exports = module.exports = k), exports._ = k) : e._ = k, k.VERSION = "1.6.0";
    var x = k.each = k.forEach = function(e, t, n) {
        if (null == e)
            return e;
        if (d && e.forEach === d)
            e.forEach(t, n);
        else if (e.length ===+ e.length) {
            for (var o = 0, s = e.length; s > o; o++)
                if (t.call(n, e[o], o, e) === i)
                    return 
        } else 
            for (var r = k.keys(e), o = 0, s = r.length; s > o; o++)
                if (t.call(n, e[r[o]], r[o], e) === i)
                    return;
        return e
    };
    k.map = k.collect = function(e, t, i) {
        var n = [];
        return null == e ? n : h && e.map === h ? e.map(t, i) : (x(e, function(e, o, s) {
            n.push(t.call(i, e, o, s))
        }), n)
    };
    var S = "Reduce of empty array with no initial value";
    k.reduce = k.foldl = k.inject = function(e, t, i, n) {
        var o = arguments.length > 2;
        if (null == e && (e = []), p && e.reduce === p)
            return n && (t = k.bind(t, n)), o ? e.reduce(t, i) : e.reduce(t);
        if (x(e, function(e, s, r) {
            o ? i = t.call(n, i, e, s, r) : (i = e, o=!0)
        }), !o)
            throw new TypeError(S);
        return i
    }, k.reduceRight = k.foldr = function(e, t, i, n) {
        var o = arguments.length > 2;
        if (null == e && (e = []), f && e.reduceRight === f)
            return n && (t = k.bind(t, n)), o ? e.reduceRight(t, i) : e.reduceRight(t);
        var s = e.length;
        if (s!==+s) {
            var r = k.keys(e);
            s = r.length
        }
        if (x(e, function(a, l, c) {
            l = r ? r[--s] : --s, o ? i = t.call(n, i, e[l], l, c) : (i = e[l], o=!0)
        }), !o)
            throw new TypeError(S);
        return i
    }, k.find = k.detect = function(e, t, i) {
        var n;
        return M(e, function(e, o, s) {
            return t.call(i, e, o, s) ? (n = e, !0) : void 0
        }), n
    }, k.filter = k.select = function(e, t, i) {
        var n = [];
        return null == e ? n : m && e.filter === m ? e.filter(t, i) : (x(e, function(e, o, s) {
            t.call(i, e, o, s) && n.push(e)
        }), n)
    }, k.reject = function(e, t, i) {
        return k.filter(e, function(e, n, o) {
            return !t.call(i, e, n, o)
        }, i)
    }, k.every = k.all = function(e, t, n) {
        t || (t = k.identity);
        var o=!0;
        return null == e ? o : g && e.every === g ? e.every(t, n) : (x(e, function(e, s, r) {
            return (o = o && t.call(n, e, s, r)) ? void 0 : i
        }), !!o)
    };
    var M = k.some = k.any = function(e, t, n) {
        t || (t = k.identity);
        var o=!1;
        return null == e ? o : v && e.some === v ? e.some(t, n) : (x(e, function(e, s, r) {
            return o || (o = t.call(n, e, s, r)) ? i : void 0
        }), !!o)
    };
    k.contains = k.include = function(e, t) {
        return null == e?!1 : y && e.indexOf === y?-1 != e.indexOf(t) : M(e, function(e) {
            return e === t
        })
    }, k.invoke = function(e, t) {
        var i = a.call(arguments, 2), n = k.isFunction(t);
        return k.map(e, function(e) {
            return (n ? t : e[t]).apply(e, i)
        })
    }, k.pluck = function(e, t) {
        return k.map(e, k.property(t))
    }, k.where = function(e, t) {
        return k.filter(e, k.matches(t))
    }, k.findWhere = function(e, t) {
        return k.find(e, k.matches(t))
    }, k.max = function(e, t, i) {
        if (!t && k.isArray(e) && e[0] ===+ e[0] && e.length < 65535)
            return Math.max.apply(Math, e);
        var n =- 1 / 0, o =- 1 / 0;
        return x(e, function(e, s, r) {
            var a = t ? t.call(i, e, s, r): e;
            a > o && (n = e, o = a)
        }), n
    }, k.min = function(e, t, i) {
        if (!t && k.isArray(e) && e[0] ===+ e[0] && e.length < 65535)
            return Math.min.apply(Math, e);
        var n = 1 / 0, o = 1 / 0;
        return x(e, function(e, s, r) {
            var a = t ? t.call(i, e, s, r): e;
            o > a && (n = e, o = a)
        }), n
    }, k.shuffle = function(e) {
        var t, i = 0, n = [];
        return x(e, function(e) {
            t = k.random(i++), n[i - 1] = n[t], n[t] = e
        }), n
    }, k.sample = function(e, t, i) {
        return null == t || i ? (e.length!==+e.length && (e = k.values(e)), e[k.random(e.length - 1)]) : k.shuffle(e).slice(0, Math.max(0, t))
    };
    var T = function(e) {
        return null == e ? k.identity : k.isFunction(e) ? e : k.property(e)
    };
    k.sortBy = function(e, t, i) {
        return t = T(t), k.pluck(k.map(e, function(e, n, o) {
            return {
                value: e,
                index: n,
                criteria: t.call(i, e, n, o)
            }
        }).sort(function(e, t) {
            var i = e.criteria, n = t.criteria;
            if (i !== n) {
                if (i > n || void 0 === i)
                    return 1;
                if (n > i || void 0 === n)
                    return - 1
            }
            return e.index - t.index
        }), "value")
    };
    var D = function(e) {
        return function(t, i, n) {
            var o = {};
            return i = T(i), x(t, function(s, r) {
                var a = i.call(n, s, r, t);
                e(o, a, s)
            }), o
        }
    };
    k.groupBy = D(function(e, t, i) {
        k.has(e, t) ? e[t].push(i) : e[t] = [i]
    }), k.indexBy = D(function(e, t, i) {
        e[t] = i
    }), k.countBy = D(function(e, t) {
        k.has(e, t) ? e[t]++ : e[t] = 1
    }), k.sortedIndex = function(e, t, i, n) {
        i = T(i);
        for (var o = i.call(n, t), s = 0, r = e.length; r > s;) {
            var a = s + r>>>1;
            i.call(n, e[a]) < o ? s = a + 1 : r = a
        }
        return s
    }, k.toArray = function(e) {
        return e ? k.isArray(e) ? a.call(e) : e.length ===+ e.length ? k.map(e, k.identity) : k.values(e) : []
    }, k.size = function(e) {
        return null == e ? 0 : e.length ===+ e.length ? e.length : k.keys(e).length
    }, k.first = k.head = k.take = function(e, t, i) {
        return null == e ? void 0 : null == t || i ? e[0] : 0 > t ? [] : a.call(e, 0, t)
    }, k.initial = function(e, t, i) {
        return a.call(e, 0, e.length - (null == t || i ? 1 : t))
    }, k.last = function(e, t, i) {
        return null == e ? void 0 : null == t || i ? e[e.length - 1] : a.call(e, Math.max(e.length - t, 0))
    }, k.rest = k.tail = k.drop = function(e, t, i) {
        return a.call(e, null == t || i ? 1 : t)
    }, k.compact = function(e) {
        return k.filter(e, k.identity)
    };
    var E = function(e, t, i, n) {
        if (t && k.every(e, k.isArray))
            return l.apply(n, e);
        for (var o = 0, s = e.length; s > o; o++) {
            var a = e[o];
            k.isArray(a) || k.isArguments(a) ? t ? r.apply(n, a) : E(a, t, i, n) : i || n.push(a)
        }
        return n
    };
    k.flatten = function(e, t) {
        return E(e, t, !1, [])
    }, k.without = function(e) {
        return k.difference(e, a.call(arguments, 1))
    }, k.partition = function(e, t, i) {
        t = T(t);
        var n = [], o = [];
        return x(e, function(e) {
            (t.call(i, e) ? n : o).push(e)
        }), [n, o]
    }, k.uniq = k.unique = function(e, t, i, n) {
        k.isFunction(t) && (n = i, i = t, t=!1);
        var o = i ? k.map(e, i, n): e, s = [], r = [];
        return x(o, function(i, n) {
            (t ? n && r[r.length - 1] === i : k.contains(r, i)) || (r.push(i), s.push(e[n]))
        }), s
    }, k.union = function() {
        return k.uniq(E(arguments, !0, !0, []))
    }, k.intersection = function(e) {
        var t = a.call(arguments, 1);
        return k.filter(k.uniq(e), function(e) {
            return k.every(t, function(t) {
                return k.contains(t, e)
            })
        })
    }, k.difference = function(e) {
        var t = E(a.call(arguments, 1), !0, !0, []);
        return k.filter(e, function(e) {
            return !k.contains(t, e)
        })
    }, k.zip = function() {
        for (var e = k.max(k.pluck(arguments, "length").concat(0)), t = new Array(e), i = 0; e > i; i++)
            t[i] = k.pluck(arguments, "" + i);
        return t
    }, k.object = function(e, t) {
        if (null == e)
            return {};
        for (var i = {}, n = 0, o = e.length; o > n; n++)
            t ? i[e[n]] = t[n] : i[e[n][0]] = e[n][1];
        return i
    }, k.indexOf = function(e, t, i) {
        if (null == e)
            return - 1;
        var n = 0, o = e.length;
        if (i) {
            if ("number" != typeof i)
                return n = k.sortedIndex(e, t), e[n] === t ? n : - 1;
            n = 0 > i ? Math.max(0, o + i) : i
        }
        if (y && e.indexOf === y)
            return e.indexOf(t, i);
        for (; o > n; n++)
            if (e[n] === t)
                return n;
        return - 1
    }, k.lastIndexOf = function(e, t, i) {
        if (null == e)
            return - 1;
        var n = null != i;
        if (b && e.lastIndexOf === b)
            return n ? e.lastIndexOf(t, i) : e.lastIndexOf(t);
        for (var o = n ? i : e.length; o--;)
            if (e[o] === t)
                return o;
        return - 1
    }, k.range = function(e, t, i) {
        arguments.length <= 1 && (t = e || 0, e = 0), i = arguments[2] || 1;
        for (var n = Math.max(Math.ceil((t - e) / i), 0), o = 0, s = new Array(n); n > o;)
            s[o++] = e, e += i;
        return s
    };
    var A = function() {};
    k.bind = function(e, t) {
        var i, n;
        if (C && e.bind === C)
            return C.apply(e, a.call(arguments, 1));
        if (!k.isFunction(e))
            throw new TypeError;
        return i = a.call(arguments, 2), n = function() {
            if (!(this instanceof n))
                return e.apply(t, i.concat(a.call(arguments)));
            A.prototype = e.prototype;
            var o = new A;
            A.prototype = null;
            var s = e.apply(o, i.concat(a.call(arguments)));
            return Object(s) === s ? s : o
        }
    }, k.partial = function(e) {
        var t = a.call(arguments, 1);
        return function() {
            for (var i = 0, n = t.slice(), o = 0, s = n.length; s > o; o++)
                n[o] === k && (n[o] = arguments[i++]);
            for (; i < arguments.length;)
                n.push(arguments[i++]);
            return e.apply(this, n)
        }
    }, k.bindAll = function(e) {
        var t = a.call(arguments, 1);
        if (0 === t.length)
            throw new Error("bindAll must be passed function names");
        return x(t, function(t) {
            e[t] = k.bind(e[t], e)
        }), e
    }, k.memoize = function(e, t) {
        var i = {};
        return t || (t = k.identity), function() {
            var n = t.apply(this, arguments);
            return k.has(i, n) ? i[n] : i[n] = e.apply(this, arguments)
        }
    }, k.delay = function(e, t) {
        var i = a.call(arguments, 2);
        return setTimeout(function() {
            return e.apply(null, i)
        }, t)
    }, k.defer = function(e) {
        return k.delay.apply(k, [e, 1].concat(a.call(arguments, 1)))
    }, k.throttle = function(e, t, i) {
        var n, o, s, r = null, a = 0;
        i || (i = {});
        var l = function() {
            a = i.leading===!1 ? 0 : k.now(), r = null, s = e.apply(n, o), n = o = null
        };
        return function() {
            var c = k.now();
            a || i.leading!==!1 || (a = c);
            var u = t - (c - a);
            return n = this, o = arguments, 0 >= u ? (clearTimeout(r), r = null, a = c, s = e.apply(n, o), n = o = null) : r || i.trailing===!1 || (r = setTimeout(l, u)), s
        }
    }, k.debounce = function(e, t, i) {
        var n, o, s, r, a, l = function() {
            var c = k.now() - r;
            t > c ? n = setTimeout(l, t - c) : (n = null, i || (a = e.apply(s, o), s = o = null))
        };
        return function() {
            s = this, o = arguments, r = k.now();
            var c = i&&!n;
            return n || (n = setTimeout(l, t)), c && (a = e.apply(s, o), s = o = null), a
        }
    }, k.once = function(e) {
        var t, i=!1;
        return function() {
            return i ? t : (i=!0, t = e.apply(this, arguments), e = null, t)
        }
    }, k.wrap = function(e, t) {
        return k.partial(t, e)
    }, k.compose = function() {
        var e = arguments;
        return function() {
            for (var t = arguments, i = e.length - 1; i >= 0; i--)
                t = [e[i].apply(this, t)];
            return t[0]
        }
    }, k.after = function(e, t) {
        return function() {
            return --e < 1 ? t.apply(this, arguments) : void 0
        }
    }, k.keys = function(e) {
        if (!k.isObject(e))
            return [];
        if (_)
            return _(e);
        var t = [];
        for (var i in e)
            k.has(e, i) && t.push(i);
        return t
    }, k.values = function(e) {
        for (var t = k.keys(e), i = t.length, n = new Array(i), o = 0; i > o; o++)
            n[o] = e[t[o]];
        return n
    }, k.pairs = function(e) {
        for (var t = k.keys(e), i = t.length, n = new Array(i), o = 0; i > o; o++)
            n[o] = [t[o], e[t[o]]];
        return n
    }, k.invert = function(e) {
        for (var t = {}, i = k.keys(e), n = 0, o = i.length; o > n; n++)
            t[e[i[n]]] = i[n];
        return t
    }, k.functions = k.methods = function(e) {
        var t = [];
        for (var i in e)
            k.isFunction(e[i]) && t.push(i);
        return t.sort()
    }, k.extend = function(e) {
        return x(a.call(arguments, 1), function(t) {
            if (t)
                for (var i in t)
                    e[i] = t[i]
        }), e
    }, k.pick = function(e) {
        var t = {}, i = l.apply(n, a.call(arguments, 1));
        return x(i, function(i) {
            i in e && (t[i] = e[i])
        }), t
    }, k.omit = function(e) {
        var t = {}, i = l.apply(n, a.call(arguments, 1));
        for (var o in e)
            k.contains(i, o) || (t[o] = e[o]);
        return t
    }, k.defaults = function(e) {
        return x(a.call(arguments, 1), function(t) {
            if (t)
                for (var i in t)
                    void 0 === e[i] && (e[i] = t[i])
        }), e
    }, k.clone = function(e) {
        return k.isObject(e) ? k.isArray(e) ? e.slice() : k.extend({}, e) : e
    }, k.tap = function(e, t) {
        return t(e), e
    };
    var P = function(e, t, i, n) {
        if (e === t)
            return 0 !== e || 1 / e == 1 / t;
        if (null == e || null == t)
            return e === t;
        e instanceof k && (e = e._wrapped), t instanceof k && (t = t._wrapped);
        var o = c.call(e);
        if (o != c.call(t))
            return !1;
        switch (o) {
        case"[object String]":
            return e == String(t);
        case"[object Number]":
            return e!=+e ? t!=+t : 0 == e ? 1 / e == 1 / t : e ==+ t;
        case"[object Date]":
        case"[object Boolean]":
            return + e ==+ t;
        case"[object RegExp]":
            return e.source == t.source && e.global == t.global && e.multiline == t.multiline && e.ignoreCase == t.ignoreCase
        }
        if ("object" != typeof e || "object" != typeof t)
            return !1;
        for (var s = i.length; s--;)
            if (i[s] == e)
                return n[s] == t;
        var r = e.constructor, a = t.constructor;
        if (r !== a&&!(k.isFunction(r) && r instanceof r && k.isFunction(a) && a instanceof a) && "constructor"in e && "constructor"in t)
            return !1;
        i.push(e), n.push(t);
        var l = 0, u=!0;
        if ("[object Array]" == o) {
            if (l = e.length, u = l == t.length)
                for (; l--&&(u = P(e[l], t[l], i, n)););
        } else {
            for (var d in e)
                if (k.has(e, d) && (l++, !(u = k.has(t, d) && P(e[d], t[d], i, n))))
                    break;
            if (u) {
                for (d in t)
                    if (k.has(t, d)&&!l--)
                        break;
                u=!l
            }
        }
        return i.pop(), n.pop(), u
    };
    k.isEqual = function(e, t) {
        return P(e, t, [], [])
    }, k.isEmpty = function(e) {
        if (null == e)
            return !0;
        if (k.isArray(e) || k.isString(e))
            return 0 === e.length;
        for (var t in e)
            if (k.has(e, t))
                return !1;
        return !0
    }, k.isElement = function(e) {
        return !(!e || 1 !== e.nodeType)
    }, k.isArray = w || function(e) {
        return "[object Array]" == c.call(e)
    }, k.isObject = function(e) {
        return e === Object(e)
    }, x(["Arguments", "Function", "String", "Number", "Date", "RegExp"], function(e) {
        k["is" + e] = function(t) {
            return c.call(t) == "[object " + e + "]"
        }
    }), k.isArguments(arguments) || (k.isArguments = function(e) {
        return !(!e ||!k.has(e, "callee"))
    }), "function" != typeof/./ && (k.isFunction = function(e) {
        return "function" == typeof e
    }), k.isFinite = function(e) {
        return isFinite(e)&&!isNaN(parseFloat(e))
    }, k.isNaN = function(e) {
        return k.isNumber(e) && e!=+e
    }, k.isBoolean = function(e) {
        return e===!0 || e===!1 || "[object Boolean]" == c.call(e)
    }, k.isNull = function(e) {
        return null === e
    }, k.isUndefined = function(e) {
        return void 0 === e
    }, k.has = function(e, t) {
        return u.call(e, t)
    }, k.noConflict = function() {
        return e._ = t, this
    }, k.identity = function(e) {
        return e
    }, k.constant = function(e) {
        return function() {
            return e
        }
    }, k.property = function(e) {
        return function(t) {
            return t[e]
        }
    }, k.matches = function(e) {
        return function(t) {
            if (t === e)
                return !0;
            for (var i in e)
                if (e[i] !== t[i])
                    return !1;
            return !0
        }
    }, k.times = function(e, t, i) {
        for (var n = Array(Math.max(0, e)), o = 0; e > o; o++)
            n[o] = t.call(i, o);
        return n
    }, k.random = function(e, t) {
        return null == t && (t = e, e = 0), e + Math.floor(Math.random() * (t - e + 1))
    }, k.now = Date.now || function() {
        return (new Date).getTime()
    };
    var R = {
        escape: {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;"
        }
    };
    R.unescape = k.invert(R.escape);
    var F = {
        escape: new RegExp("[" + k.keys(R.escape).join("") + "]", "g"),
        unescape: new RegExp("(" + k.keys(R.unescape).join("|") + ")", "g")
    };
    k.each(["escape", "unescape"], function(e) {
        k[e] = function(t) {
            return null == t ? "" : ("" + t).replace(F[e], function(t) {
                return R[e][t]
            })
        }
    }), k.result = function(e, t) {
        if (null == e)
            return void 0;
        var i = e[t];
        return k.isFunction(i) ? i.call(e) : i
    }, k.mixin = function(e) {
        x(k.functions(e), function(t) {
            var i = k[t] = e[t];
            k.prototype[t] = function() {
                var e = [this._wrapped];
                return r.apply(e, arguments), z.call(this, i.apply(k, e))
            }
        })
    };
    var O = 0;
    k.uniqueId = function(e) {
        var t=++O + "";
        return e ? e + t : t
    }, k.templateSettings = {
        evaluate: /<%([\s\S]+?)%>/g,
        interpolate: /<%=([\s\S]+?)%>/g,
        escape: /<%-([\s\S]+?)%>/g
    };
    var I = /(.)^/, W = {
        "'": "'",
        "\\": "\\",
        "\r": "r",
        "\n": "n",
        "	": "t",
        "\u2028": "u2028",
        "\u2029": "u2029"
    }, N = /\\|'|\r|\n|\t|\u2028|\u2029/g;
    k.template = function(e, t, i) {
        var n;
        i = k.defaults({}, i, k.templateSettings);
        var o = new RegExp([(i.escape || I).source, (i.interpolate || I).source, (i.evaluate || I).source].join("|") + "|$", "g"), s = 0, r = "__p+='";
        e.replace(o, function(t, i, n, o, a) {
            return r += e.slice(s, a).replace(N, function(e) {
                return "\\" + W[e]
            }), i && (r += "'+\n((__t=(" + i + "))==null?'':_.escape(__t))+\n'"), n && (r += "'+\n((__t=(" + n + "))==null?'':__t)+\n'"), o && (r += "';\n" + o + "\n__p+='"), s = a + t.length, t
        }), r += "';\n", i.variable || (r = "with(obj||{}){\n" + r + "}\n"), r = "var __t,__p='',__j=Array.prototype.join,print=function(){__p+=__j.call(arguments,'');};\n" + r + "return __p;\n";
        try {
            n = new Function(i.variable || "obj", "_", r)
        } catch (a) {
            throw a.source = r, a
        }
        if (t)
            return n(t, k);
        var l = function(e) {
            return n.call(this, e, k)
        };
        return l.source = "function(" + (i.variable || "obj") + "){\n" + r + "}", l
    }, k.chain = function(e) {
        return k(e).chain()
    };
    var z = function(e) {
        return this._chain ? k(e).chain() : e
    };
    k.mixin(k), x(["pop", "push", "reverse", "shift", "sort", "splice", "unshift"], function(e) {
        var t = n[e];
        k.prototype[e] = function() {
            var i = this._wrapped;
            return t.apply(i, arguments), "shift" != e && "splice" != e || 0 !== i.length || delete i[0], z.call(this, i)
        }
    }), x(["concat", "join", "slice"], function(e) {
        var t = n[e];
        k.prototype[e] = function() {
            return z.call(this, t.apply(this._wrapped, arguments))
        }
    }), k.extend(k.prototype, {
        chain: function() {
            return this._chain=!0, this
        },
        value: function() {
            return this._wrapped
        }
    }), "function" == typeof _wAMD.define && _wAMD.define.amd && _wAMD.define("underscore", [], function() {
        return k
    })
}.call(this), function(e, t) {
    if ("function" == typeof _wAMD.define && _wAMD.define.amd)
        _wAMD.define("backbone", ["underscore", "jquery", "exports"], function(i, n, o) {
            e.Backbone = t(e, o, i, n)
        });
    else if ("undefined" != typeof exports) {
        var i = require("underscore");
        t(e, exports, i)
    } else 
        e.Backbone = t(e, {}, e._, e.jQuery || e.Zepto || e.ender || e.$)
}(this, function(e, t, i, n) {
    var o = e.Backbone, s = [];
    s.push;
    var r = s.slice;
    s.splice, t.VERSION = "1.1.2", t.$ = n, t.noConflict = function() {
        return e.Backbone = o, this
    }, t.emulateHTTP=!1, t.emulateJSON=!1;
    var a = t.Events = {
        on: function(e, t, i) {
            if (!c(this, "on", e, [t, i]) ||!t)
                return this;
            this._events || (this._events = {});
            var n = this._events[e] || (this._events[e] = []);
            return n.push({
                callback: t,
                context: i,
                ctx: i || this
            }), this
        },
        once: function(e, t, n) {
            if (!c(this, "once", e, [t, n]) ||!t)
                return this;
            var o = this, s = i.once(function() {
                o.off(e, s), t.apply(this, arguments)
            });
            return s._callback = t, this.on(e, s, n)
        },
        off: function(e, t, n) {
            var o, s, r, a, l, u, d, h;
            if (!this._events ||!c(this, "off", e, [t, n]))
                return this;
            if (!e&&!t&&!n)
                return this._events = void 0, this;
            for (a = e ? [e] : i.keys(this._events), l = 0, u = a.length; u > l; l++)
                if (e = a[l], r = this._events[e]) {
                    if (this._events[e] = o = [], t || n)
                        for (d = 0, h = r.length; h > d; d++)
                            s = r[d], (t && t !== s.callback && t !== s.callback._callback || n && n !== s.context) && o.push(s);
                            o.length || delete this._events[e]
                }
            return this
        },
        trigger: function(e) {
            if (!this._events)
                return this;
            var t = r.call(arguments, 1);
            if (!c(this, "trigger", e, t))
                return this;
            var i = this._events[e], n = this._events.all;
            return i && u(i, t), n && u(n, arguments), this
        },
        stopListening: function(e, t, n) {
            var o = this._listeningTo;
            if (!o)
                return this;
            var s=!t&&!n;
            n || "object" != typeof t || (n = this), e && ((o = {})[e._listenId] = e);
            for (var r in o)
                e = o[r], e.off(t, n, this), (s || i.isEmpty(e._events)) && delete this._listeningTo[r];
            return this
        }
    }, l = /\s+/, c = function(e, t, i, n) {
        if (!i)
            return !0;
        if ("object" == typeof i) {
            for (var o in i)
                e[t].apply(e, [o, i[o]].concat(n));
            return !1
        }
        if (l.test(i)) {
            for (var s = i.split(l), r = 0, a = s.length; a > r; r++)
                e[t].apply(e, [s[r]].concat(n));
            return !1
        }
        return !0
    }, u = function(e, t) {
        var i, n =- 1, o = e.length, s = t[0], r = t[1], a = t[2];
        switch (t.length) {
        case 0:
            for (; ++n < o;)(i = e[n])
                .callback.call(i.ctx);
            return;
        case 1:
            for (; ++n < o;)(i = e[n])
                .callback.call(i.ctx, s);
            return;
        case 2:
            for (; ++n < o;)(i = e[n])
                .callback.call(i.ctx, s, r);
            return;
        case 3:
            for (; ++n < o;)(i = e[n])
                .callback.call(i.ctx, s, r, a);
            return;
        default:
            for (; ++n < o;)(i = e[n])
                .callback.apply(i.ctx, t);
            return 
        }
    }, d = {
        listenTo: "on",
        listenToOnce: "once"
    };
    i.each(d, function(e, t) {
        a[t] = function(t, n, o) {
            var s = this._listeningTo || (this._listeningTo = {}), r = t._listenId || (t._listenId = i.uniqueId("l"));
            return s[r] = t, o || "object" != typeof n || (o = this), t[e](n, o, this), this
        }
    }), a.bind = a.on, a.unbind = a.off, i.extend(t, a);
    var h = t.Model = function(e, t) {
        var n = e || {};
        t || (t = {}), this.cid = i.uniqueId("c"), this.attributes = {}, t.collection && (this.collection = t.collection), t.parse && (n = this.parse(n, t) || {}), n = i.defaults({}, n, i.result(this, "defaults")), this.set(n, t), this.changed = {}, this.initialize.apply(this, arguments)
    };
    i.extend(h.prototype, a, {
        changed: null,
        validationError: null,
        idAttribute: "id",
        initialize: function() {},
        toJSON: function() {
            return i.clone(this.attributes)
        },
        sync: function() {
            return t.sync.apply(this, arguments)
        },
        get: function(e) {
            return this.attributes[e]
        },
        escape: function(e) {
            return i.escape(this.get(e))
        },
        has: function(e) {
            return null != this.get(e)
        },
        set: function(e, t, n) {
            var o, s, r, a, l, c, u, d;
            if (null == e)
                return this;
            if ("object" == typeof e ? (s = e, n = t) : (s = {})[e] = t, n || (n = {}), !this._validate(s, n))
                return !1;
            r = n.unset, l = n.silent, a = [], c = this._changing, this._changing=!0, c || (this._previousAttributes = i.clone(this.attributes), this.changed = {}), d = this.attributes, u = this._previousAttributes, this.idAttribute in s && (this.id = s[this.idAttribute]);
            for (o in s)
                t = s[o], i.isEqual(d[o], t) || a.push(o), i.isEqual(u[o], t) ? delete this.changed[o] : this.changed[o] = t, r ? delete d[o] : d[o] = t;
            if (!l) {
                a.length && (this._pending = n);
                for (var h = 0, p = a.length; p > h; h++)
                    this.trigger("change:" + a[h], this, d[a[h]], n)
            }
            if (c)
                return this;
            if (!l)
                for (; this._pending;)
                    n = this._pending, this._pending=!1, this.trigger("change", this, n);
            return this._pending=!1, this._changing=!1, this
        },
        unset: function(e, t) {
            return this.set(e, void 0, i.extend({}, t, {
                unset : !0
            }))
        },
        clear: function(e) {
            var t = {};
            for (var n in this.attributes)
                t[n] = void 0;
            return this.set(t, i.extend({}, e, {
                unset: !0
            }))
        },
        hasChanged: function(e) {
            return null == e?!i.isEmpty(this.changed) : i.has(this.changed, e)
        },
        changedAttributes: function(e) {
            if (!e)
                return this.hasChanged() ? i.clone(this.changed) : !1;
            var t, n=!1, o = this._changing ? this._previousAttributes : this.attributes;
            for (var s in e)
                i.isEqual(o[s], t = e[s]) || ((n || (n = {}))[s] = t);
            return n
        },
        previous: function(e) {
            return null != e && this._previousAttributes ? this._previousAttributes[e] : null
        },
        previousAttributes: function() {
            return i.clone(this._previousAttributes)
        },
        fetch: function(e) {
            e = e ? i.clone(e) : {}, void 0 === e.parse && (e.parse=!0);
            var t = this, n = e.success;
            return e.success = function(i) {
                return t.set(t.parse(i, e), e) ? (n && n(t, i, e), t.trigger("sync", t, i, e), void 0) : !1
            }, N(this, e), this.sync("read", this, e)
        },
        save: function(e, t, n) {
            var o, s, r, a = this.attributes;
            if (null == e || "object" == typeof e ? (o = e, n = t) : (o = {})[e] = t, n = i.extend({
                validate: !0
            }, n), o&&!n.wait) {
                if (!this.set(o, n))
                    return !1
            } else if (!this._validate(o, n))
                return !1;
            o && n.wait && (this.attributes = i.extend({}, a, o)), void 0 === n.parse && (n.parse=!0);
            var l = this, c = n.success;
            return n.success = function(e) {
                l.attributes = a;
                var t = l.parse(e, n);
                return n.wait && (t = i.extend(o || {}, t)), i.isObject(t)&&!l.set(t, n)?!1 : (c && c(l, e, n), l.trigger("sync", l, e, n), void 0)
            }, N(this, n), s = this.isNew() ? "create" : n.patch ? "patch" : "update", "patch" === s && (n.attrs = o), r = this.sync(s, this, n), o && n.wait && (this.attributes = a), r
        },
        destroy: function(e) {
            e = e ? i.clone(e) : {};
            var t = this, n = e.success, o = function() {
                t.trigger("destroy", t, t.collection, e)
            };
            if (e.success = function(i) {
                (e.wait || t.isNew()) && o(), n && n(t, i, e), t.isNew() || t.trigger("sync", t, i, e)
            }, this.isNew())
                return e.success(), !1;
            N(this, e);
            var s = this.sync("delete", this, e);
            return e.wait || o(), s
        },
        url: function() {
            var e = i.result(this, "urlRoot") || i.result(this.collection, "url") || W();
            return this.isNew() ? e : e.replace(/([^\/])$/, "$1/") + encodeURIComponent(this.id)
        },
        parse: function(e) {
            return e
        },
        clone: function() {
            return new this.constructor(this.attributes)
        },
        isNew: function() {
            return !this.has(this.idAttribute)
        },
        isValid: function(e) {
            return this._validate({}, i.extend(e || {}, {
                validate: !0
            }))
        },
        _validate: function(e, t) {
            if (!t.validate ||!this.validate)
                return !0;
            e = i.extend({}, this.attributes, e);
            var n = this.validationError = this.validate(e, t) || null;
            return n ? (this.trigger("invalid", this, n, i.extend(t, {
                validationError: n
            })), !1) : !0
        }
    });
    var p = ["keys", "values", "pairs", "invert", "pick", "omit"];
    i.each(p, function(e) {
        h.prototype[e] = function() {
            var t = r.call(arguments);
            return t.unshift(this.attributes), i[e].apply(i, t)
        }
    });
    var f = t.Collection = function(e, t) {
        t || (t = {}), t.model && (this.model = t.model), void 0 !== t.comparator && (this.comparator = t.comparator), this._reset(), this.initialize.apply(this, arguments), e && this.reset(e, i.extend({
            silent: !0
        }, t))
    }, m = {
        add: !0,
        remove: !0,
        merge: !0
    }, g = {
        add: !0,
        remove: !1
    };
    i.extend(f.prototype, a, {
        model: h,
        initialize: function() {},
        toJSON: function(e) {
            return this.map(function(t) {
                return t.toJSON(e)
            })
        },
        sync: function() {
            return t.sync.apply(this, arguments)
        },
        add: function(e, t) {
            return this.set(e, i.extend({
                merge: !1
            }, t, g))
        },
        remove: function(e, t) {
            var n=!i.isArray(e);
            e = n ? [e] : i.clone(e), t || (t = {});
            var o, s, r, a;
            for (o = 0, s = e.length; s > o; o++)
                a = e[o] = this.get(e[o]), a && (delete this._byId[a.id], delete this._byId[a.cid], r = this.indexOf(a), this.models.splice(r, 1), this.length--, t.silent || (t.index = r, a.trigger("remove", a, this, t)), this._removeReference(a, t));
            return n ? e[0] : e
        },
        set: function(e, t) {
            t = i.defaults({}, t, m), t.parse && (e = this.parse(e, t));
            var n=!i.isArray(e);
            e = n ? e ? [e] : [] : i.clone(e);
            var o, s, r, a, l, c, u, d = t.at, p = this.model, f = this.comparator && null == d && t.sort!==!1, g = i.isString(this.comparator) ? this.comparator: null, v = [], y = [], b = {}, w = t.add, _ = t.merge, C = t.remove, k=!f && w && C ? [] : !1;
            for (o = 0, s = e.length; s > o; o++) {
                if (l = e[o] || {}, r = l instanceof h ? a = l : l[p.prototype.idAttribute || "id"], c = this.get(r))
                    C && (b[c.cid]=!0), _ && (l = l === a ? a.attributes : l, t.parse && (l = c.parse(l, t)), c.set(l, t), f&&!u && c.hasChanged(g) && (u=!0)), e[o] = c;
                else if (w) {
                    if (a = e[o] = this._prepareModel(l, t), !a)
                        continue;
                    v.push(a), this._addReference(a, t)
                }
                a = c || a, !k ||!a.isNew() && b[a.id] || k.push(a), b[a.id]=!0
            }
            if (C) {
                for (o = 0, s = this.length; s > o; ++o)
                    b[(a = this.models[o]).cid] || y.push(a);
                y.length && this.remove(y, t)
            }
            if (v.length || k && k.length)
                if (f && (u=!0), this.length += v.length, null != d)
                    for (o = 0, s = v.length; s > o; o++)
                        this.models.splice(d + o, 0, v[o]);
                else {
                    k && (this.models.length = 0);
                    var x = k || v;
                    for (o = 0, s = x.length; s > o; o++)
                        this.models.push(x[o])
                }
            if (u && this.sort({
                silent: !0
            }), !t.silent) {
                for (o = 0, s = v.length; s > o; o++)(a = v[o])
                    .trigger("add", a, this, t);
                (u || k && k.length) && this.trigger("sort", this, t)
            }
            return n ? e[0] : e
        },
        reset: function(e, t) {
            t || (t = {});
            for (var n = 0, o = this.models.length; o > n; n++)
                this._removeReference(this.models[n], t);
            return t.previousModels = this.models, this._reset(), e = this.add(e, i.extend({
                silent: !0
            }, t)), t.silent || this.trigger("reset", this, t), e
        },
        push: function(e, t) {
            return this.add(e, i.extend({
                at: this.length
            }, t))
        },
        pop: function(e) {
            var t = this.at(this.length - 1);
            return this.remove(t, e), t
        },
        unshift: function(e, t) {
            return this.add(e, i.extend({
                at: 0
            }, t))
        },
        shift: function(e) {
            var t = this.at(0);
            return this.remove(t, e), t
        },
        slice: function() {
            return r.apply(this.models, arguments)
        },
        get: function(e) {
            return null == e ? void 0 : this._byId[e] || this._byId[e.id] || this._byId[e.cid]
        },
        at: function(e) {
            return this.models[e]
        },
        where: function(e, t) {
            return i.isEmpty(e) ? t ? void 0 : [] : this[t ? "find": "filter"](function(t) {
                for (var i in e)
                    if (e[i] !== t.get(i))
                        return !1;
                return !0
            })
        },
        findWhere: function(e) {
            return this.where(e, !0)
        },
        sort: function(e) {
            if (!this.comparator)
                throw new Error("Cannot sort a set without a comparator");
            return e || (e = {}), i.isString(this.comparator) || 1 === this.comparator.length ? this.models = this.sortBy(this.comparator, this) : this.models.sort(i.bind(this.comparator, this)), e.silent || this.trigger("sort", this, e), this
        },
        pluck: function(e) {
            return i.invoke(this.models, "get", e)
        },
        fetch: function(e) {
            e = e ? i.clone(e) : {}, void 0 === e.parse && (e.parse=!0);
            var t = e.success, n = this;
            return e.success = function(i) {
                var o = e.reset ? "reset": "set";
                n[o](i, e), t && t(n, i, e), n.trigger("sync", n, i, e)
            }, N(this, e), this.sync("read", this, e)
        },
        create: function(e, t) {
            if (t = t ? i.clone(t) : {}, !(e = this._prepareModel(e, t)))
                return !1;
            t.wait || this.add(e, t);
            var n = this, o = t.success;
            return t.success = function(e, i) {
                t.wait && n.add(e, t), o && o(e, i, t)
            }, e.save(null, t), e
        },
        parse: function(e) {
            return e
        },
        clone: function() {
            return new this.constructor(this.models)
        },
        _reset: function() {
            this.length = 0, this.models = [], this._byId = {}
        },
        _prepareModel: function(e, t) {
            if (e instanceof h)
                return e;
            t = t ? i.clone(t) : {}, t.collection = this;
            var n = new this.model(e, t);
            return n.validationError ? (this.trigger("invalid", this, n.validationError, t), !1) : n
        },
        _addReference: function(e) {
            this._byId[e.cid] = e, null != e.id && (this._byId[e.id] = e), e.collection || (e.collection = this), e.on("all", this._onModelEvent, this)
        },
        _removeReference: function(e) {
            this === e.collection && delete e.collection, e.off("all", this._onModelEvent, this)
        },
        _onModelEvent: function(e, t, i, n) {
            ("add" !== e && "remove" !== e || i === this) && ("destroy" === e && this.remove(t, n), t && e === "change:" + t.idAttribute && (delete this._byId[t.previous(t.idAttribute)], null != t.id && (this._byId[t.id] = t)), this.trigger.apply(this, arguments))
        }
    });
    var v = ["forEach", "each", "map", "collect", "reduce", "foldl", "inject", "reduceRight", "foldr", "find", "detect", "filter", "select", "reject", "every", "all", "some", "any", "include", "contains", "invoke", "max", "min", "toArray", "size", "first", "head", "take", "initial", "rest", "tail", "drop", "last", "without", "difference", "indexOf", "shuffle", "lastIndexOf", "isEmpty", "chain", "sample"];
    i.each(v, function(e) {
        f.prototype[e] = function() {
            var t = r.call(arguments);
            return t.unshift(this.models), i[e].apply(i, t)
        }
    });
    var y = ["groupBy", "countBy", "sortBy", "indexBy"];
    i.each(y, function(e) {
        f.prototype[e] = function(t, n) {
            var o = i.isFunction(t) ? t: function(e) {
                return e.get(t)
            };
            return i[e](this.models, o, n)
        }
    });
    var b = t.View = function(e) {
        this.cid = i.uniqueId("view"), e || (e = {}), i.extend(this, i.pick(e, _)), this._ensureElement(), this.initialize.apply(this, arguments), this.delegateEvents()
    }, w = /^(\S+)\s*(.*)$/, _ = ["model", "collection", "el", "id", "attributes", "className", "tagName", "events"];
    i.extend(b.prototype, a, {
        tagName: "div",
        $: function(e) {
            return this.$el.find(e)
        },
        initialize: function() {},
        render: function() {
            return this
        },
        remove: function() {
            return this.$el.remove(), this.stopListening(), this
        },
        setElement: function(e, i) {
            return this.$el && this.undelegateEvents(), this.$el = e instanceof t.$ ? e : t.$(e), this.el = this.$el[0], i!==!1 && this.delegateEvents(), this
        },
        delegateEvents: function(e) {
            if (!e&&!(e = i.result(this, "events")))
                return this;
            this.undelegateEvents();
            for (var t in e) {
                var n = e[t];
                if (i.isFunction(n) || (n = this[e[t]]), n) {
                    var o = t.match(w), s = o[1], r = o[2];
                    n = i.bind(n, this), s += ".delegateEvents" + this.cid, "" === r ? this.$el.on(s, n) : this.$el.on(s, r, n)
                }
            }
            return this
        },
        undelegateEvents: function() {
            return this.$el.off(".delegateEvents" + this.cid), this
        },
        _ensureElement: function() {
            if (this.el)
                this.setElement(i.result(this, "el"), !1);
            else {
                var e = i.extend({}, i.result(this, "attributes"));
                this.id && (e.id = i.result(this, "id")), this.className && (e["class"] = i.result(this, "className"));
                var n = t.$("<" + i.result(this, "tagName") + ">").attr(e);
                this.setElement(n, !1)
            }
        }
    }), t.sync = function(e, n, o) {
        var s = k[e];
        i.defaults(o || (o = {}), {
            emulateHTTP: t.emulateHTTP,
            emulateJSON: t.emulateJSON
        });
        var r = {
            type: s,
            dataType: "json"
        };
        if (o.url || (r.url = i.result(n, "url") || W()), null != o.data ||!n || "create" !== e && "update" !== e && "patch" !== e || (r.contentType = "application/json", r.data = JSON.stringify(o.attrs || n.toJSON(o))), o.emulateJSON && (r.contentType = "application/x-www-form-urlencoded", r.data = r.data ? {
            model: r.data
        } : {}), o.emulateHTTP && ("PUT" === s || "DELETE" === s || "PATCH" === s)) {
            r.type = "POST", o.emulateJSON && (r.data._method = s);
            var a = o.beforeSend;
            o.beforeSend = function(e) {
                return e.setRequestHeader("X-HTTP-Method-Override", s), a ? a.apply(this, arguments) : void 0
            }
        }
        "GET" === r.type || o.emulateJSON || (r.processData=!1), "PATCH" === r.type && C && (r.xhr = function() {
            return new ActiveXObject("Microsoft.XMLHTTP")
        });
        var l = o.xhr = t.ajax(i.extend(r, o));
        return n.trigger("request", n, l, o), l
    };
    var C=!("undefined" == typeof window ||!window.ActiveXObject || window.XMLHttpRequest && (new XMLHttpRequest).dispatchEvent), k = {
        create: "POST",
        update: "PUT",
        patch: "PATCH",
        "delete": "DELETE",
        read: "GET"
    };
    t.ajax = function() {
        return t.$.ajax.apply(t.$, arguments)
    };
    var x = t.Router = function(e) {
        e || (e = {}), e.routes && (this.routes = e.routes), this._bindRoutes(), this.initialize.apply(this, arguments)
    }, S = /\((.*?)\)/g, M = /(\(\?)?:\w+/g, T = /\*\w+/g, D = /[\-{}\[\]+?.,\\\^$|#\s]/g;
    i.extend(x.prototype, a, {
        initialize: function() {},
        route: function(e, n, o) {
            i.isRegExp(e) || (e = this._routeToRegExp(e)), i.isFunction(n) && (o = n, n = ""), o || (o = this[n]);
            var s = this;
            return t.history.route(e, function(i) {
                var r = s._extractParameters(e, i);
                s.execute(o, r), s.trigger.apply(s, ["route:" + n].concat(r)), s.trigger("route", n, r), t.history.trigger("route", s, n, r)
            }), this
        },
        execute: function(e, t) {
            e && e.apply(this, t)
        },
        navigate: function(e, i) {
            return t.history.navigate(e, i), this
        },
        _bindRoutes: function() {
            if (this.routes) {
                this.routes = i.result(this, "routes");
                for (var e, t = i.keys(this.routes); null != (e = t.pop());)
                    this.route(e, this.routes[e])
            }
        },
        _routeToRegExp: function(e) {
            return e = e.replace(D, "\\$&").replace(S, "(?:$1)?").replace(M, function(e, t) {
                return t ? e : "([^/?]+)"
            }).replace(T, "([^?]*?)"), new RegExp("^" + e + "(?:\\?([\\s\\S]*))?$")
        },
        _extractParameters: function(e, t) {
            var n = e.exec(t).slice(1);
            return i.map(n, function(e, t) {
                return t === n.length - 1 ? e || null : e ? decodeURIComponent(e) : null
            })
        }
    });
    var E = t.History = function() {
        this.handlers = [], i.bindAll(this, "checkUrl"), "undefined" != typeof window && (this.location = window.location, this.history = window.history)
    }, A = /^[#\/]|\s+$/g, P = /^\/+|\/+$/g, R = /msie [\w.]+/, F = /\/$/, O = /#.*$/;
    E.started=!1, i.extend(E.prototype, a, {
        interval: 50,
        atRoot: function() {
            return this.location.pathname.replace(/[^\/]$/, "$&/") === this.root
        },
        getHash: function(e) {
            var t = (e || this).location.href.match(/#(.*)$/);
            return t ? t[1] : ""
        },
        getFragment: function(e, t) {
            if (null == e)
                if (this._hasPushState ||!this._wantsHashChange || t) {
                    e = decodeURI(this.location.pathname + this.location.search);
                    var i = this.root.replace(F, "");
                    e.indexOf(i) || (e = e.slice(i.length))
                } else 
                    e = this.getHash();
            return e.replace(A, "")
        },
        start: function(e) {
            if (E.started)
                throw new Error("Backbone.history has already been started");
            E.started=!0, this.options = i.extend({
                root: "/"
            }, this.options, e), this.root = this.options.root, this._wantsHashChange = this.options.hashChange!==!1, this._wantsPushState=!!this.options.pushState, this._hasPushState=!!(this.options.pushState && this.history && this.history.pushState);
            var n = this.getFragment(), o = document.documentMode, s = R.exec(navigator.userAgent.toLowerCase()) && (!o || 7 >= o);
            if (this.root = ("/" + this.root + "/").replace(P, "/"), s && this._wantsHashChange) {
                var r = t.$('<iframe src="javascript:0" tabindex="-1">');
                this.iframe = r.hide().appendTo("body")[0].contentWindow, this.navigate(n)
            }
            this._hasPushState ? t.$(window).on("popstate", this.checkUrl) : this._wantsHashChange && "onhashchange"in window&&!s ? t.$(window).on("hashchange", this.checkUrl) : this._wantsHashChange && (this._checkUrlInterval = setInterval(this.checkUrl, this.interval)), this.fragment = n;
            var a = this.location;
            if (this._wantsHashChange && this._wantsPushState) {
                if (!this._hasPushState&&!this.atRoot())
                    return this.fragment = this.getFragment(null, !0), this.location.replace(this.root + "#" + this.fragment), !0;
                this._hasPushState && this.atRoot() && a.hash && (this.fragment = this.getHash().replace(A, ""), this.history.replaceState({}, document.title, this.root + this.fragment))
            }
            return this.options.silent ? void 0 : this.loadUrl()
        },
        stop: function() {
            t.$(window).off("popstate", this.checkUrl).off("hashchange", this.checkUrl), this._checkUrlInterval && clearInterval(this._checkUrlInterval), E.started=!1
        },
        route: function(e, t) {
            this.handlers.unshift({
                route: e,
                callback: t
            })
        },
        checkUrl: function() {
            var e = this.getFragment();
            return e === this.fragment && this.iframe && (e = this.getFragment(this.getHash(this.iframe))), e === this.fragment?!1 : (this.iframe && this.navigate(e), this.loadUrl(), void 0)
        },
        loadUrl: function(e) {
            return e = this.fragment = this.getFragment(e), i.any(this.handlers, function(t) {
                return t.route.test(e) ? (t.callback(e), !0) : void 0
            })
        },
        navigate: function(e, t) {
            if (!E.started)
                return !1;
            t && t!==!0 || (t = {
                trigger: !!t
            });
            var i = this.root + (e = this.getFragment(e || ""));
            if (e = e.replace(O, ""), this.fragment !== e) {
                if (this.fragment = e, "" === e && "/" !== i && (i = i.slice(0, - 1)), this._hasPushState)
                    this.history[t.replace ? "replaceState": "pushState"]({}, document.title, i);
                else {
                    if (!this._wantsHashChange)
                        return this.location.assign(i);
                    this._updateHash(this.location, e, t.replace), this.iframe && e !== this.getFragment(this.getHash(this.iframe)) && (t.replace || this.iframe.document.open().close(), this._updateHash(this.iframe.location, e, t.replace))
                }
                return t.trigger ? this.loadUrl(e) : void 0
            }
        },
        _updateHash: function(e, t, i) {
            if (i) {
                var n = e.href.replace(/(javascript:|#).*$/, "");
                e.replace(n + "#" + t)
            } else 
                e.hash = "#" + t
        }
    }), t.history = new E;
    var I = function(e, t) {
        var n, o = this;
        n = e && i.has(e, "constructor") ? e.constructor : function() {
            return o.apply(this, arguments)
        }, i.extend(n, o, t);
        var s = function() {
            this.constructor = n
        };
        return s.prototype = o.prototype, n.prototype = new s, e && i.extend(n.prototype, e), n.__super__ = o.prototype, n
    };
    h.extend = f.extend = x.extend = b.extend = E.extend = I;
    var W = function() {
        throw new Error('A "url" property or function must be specified')
    }, N = function(e, t) {
        var i = t.error;
        t.error = function(n) {
            i && i(e, n, t), e.trigger("error", e, n, t)
        }
    };
    return t
}), function(e) {
    "object" == typeof exports ? module.exports = e(require("backbone"), require("underscore")) : "function" == typeof _wAMD.define && _wAMD.define.amd && _wAMD.define("backbone-validation", ["backbone", "underscore"], e)
}(function(e, t) {
    return e.Validation = function(t) {
        var i = {
            forceUpdate: !1,
            selector: "name",
            labelFormatter: "sentenceCase",
            valid: Function.prototype,
            invalid: Function.prototype
        }, n = {
            formatLabel: function(e, t) {
                return c[i.labelFormatter](e, t)
            },
            format: function() {
                var e = Array.prototype.slice.call(arguments), t = e.shift();
                return t.replace(/\{(\d+)\}/g, function(t, i) {
                    return "undefined" != typeof e[i] ? e[i] : t
                })
            }
        }, o = function(i, n, s) {
            return n = n || {}, s = s || "", t.each(i, function(t, r) {
                i.hasOwnProperty(r) && (!t || "object" != typeof t || t instanceof Array || t instanceof Date || t instanceof RegExp || t instanceof e.Model || t instanceof e.Collection ? n[s + r] = t : o(t, n, s + r + "."))
            }), n
        }, s = function() {
            var e = function(e) {
                return t.reduce(t.keys(t.result(e, "validation") || {}), function(e, t) {
                    return e[t] = void 0, e
                }, {})
            }, s = function(e, i) {
                var n = e.validation ? t.result(e, "validation")[i] || {}
                : {};
                return (t.isFunction(n) || t.isString(n)) && (n = {
                    fn: n
                }), t.isArray(n) || (n = [n]), t.reduce(n, function(e, i) {
                    return t.each(t.without(t.keys(i), "msg"), function(t) {
                        e.push({
                            fn: u[t],
                            val: i[t],
                            msg: i.msg
                        })
                    }), e
                }, [])
            }, a = function(e, i, o, r) {
                return t.reduce(s(e, i), function(s, a) {
                    var l = t.extend({}, n, u), c = a.fn.call(l, o, i, a.val, e, r);
                    return c===!1 || s===!1?!1 : c&&!s ? t.result(a, "msg") || c : s
                }, "")
            }, l = function(e, i) {
                var n, s = {}, r=!0, l = t.clone(i), c = o(i);
                return t.each(c, function(t, i) {
                    n = a(e, i, t, l), n && (s[i] = n, r=!1)
                }), {
                    invalidAttrs: s,
                    isValid: r
                }
            }, c = function(i, n) {
                return {
                    preValidate: function(e, i) {
                        var n, o = this, s = {};
                        return t.isObject(e) ? (t.each(e, function(e, t) {
                            n = o.preValidate(t, e), n && (s[t] = n)
                        }), t.isEmpty(s) ? void 0 : s) : a(this, e, i, t.extend({}, this.attributes))
                    },
                    isValid: function(e) {
                        var i = o(this.attributes);
                        return t.isString(e)?!a(this, e, i[e], t.extend({}, this.attributes)) : t.isArray(e) ? t.reduce(e, function(e, n) {
                            return e&&!a(this, n, i[n], t.extend({}, this.attributes))
                        }, !0, this) : (e===!0 && this.validate(), this.validation ? this._isValid : !0)
                    },
                    validate: function(s, r) {
                        var a = this, c=!s, u = t.extend({}, n, r), d = e(a), h = t.extend({}, d, a.attributes, s), p = o(s || h), f = l(a, h);
                        return a._isValid = f.isValid, t.each(d, function(e, t) {
                            var n = f.invalidAttrs.hasOwnProperty(t);
                            n || u.valid(i, t, u.selector)
                        }), t.each(d, function(e, t) {
                            var n = f.invalidAttrs.hasOwnProperty(t), o = p.hasOwnProperty(t);
                            n && (o || c) && u.invalid(i, t, f.invalidAttrs[t], u.selector)
                        }), t.defer(function() {
                            a.trigger("validated", a._isValid, a, f.invalidAttrs), a.trigger("validated:" + (a._isValid ? "valid" : "invalid"), a, f.invalidAttrs)
                        }), !u.forceUpdate && t.intersection(t.keys(f.invalidAttrs), t.keys(p)).length > 0 ? f.invalidAttrs : void 0
                    }
                }
            }, d = function(e, i, n) {
                t.extend(i, c(e, n))
            }, h = function(e) {
                delete e.validate, delete e.preValidate, delete e.isValid
            }, p = function(e) {
                d(this.view, e, this.options)
            }, f = function(e) {
                h(e)
            };
            return {
                version: "0.9.1",
                configure: function(e) {
                    t.extend(i, e)
                },
                bind: function(e, n) {
                    n = t.extend({}, i, r, n);
                    var o = n.model || e.model, s = n.collection || e.collection;
                    if ("undefined" == typeof o && "undefined" == typeof s)
                        throw "Before you execute the binding your view must have a model or a collection.\nSee http://thedersen.com/projects/backbone-validation/#using-form-model-validation for more information.";
                    o ? d(e, o, n) : s && (s.each(function(t) {
                        d(e, t, n)
                    }), s.bind("add", p, {
                        view: e,
                        options: n
                    }), s.bind("remove", f))
                },
                unbind: function(e, i) {
                    i = t.extend({}, i);
                    var n = i.model || e.model, o = i.collection || e.collection;
                    n ? h(n) : o && (o.each(function(e) {
                        h(e)
                    }), o.unbind("add", p), o.unbind("remove", f))
                },
                mixin: c(null, i)
            }
        }(), r = s.callbacks = {
            valid: function(e, t, i) {
                e.$("[" + i + '~="' + t + '"]').removeClass("invalid").removeAttr("data-error")
            },
            invalid: function(e, t, i, n) {
                e.$("[" + n + '~="' + t + '"]').addClass("invalid").attr("data-error", i)
            }
        }, a = s.patterns = {
            digits: /^\d+$/,
            number: /^-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$/,
            email: /^((([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+(\.([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+)*)|((\x22)((((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(([\x01-\x08\x0b\x0c\x0e-\x1f\x7f]|\x21|[\x23-\x5b]|[\x5d-\x7e]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(\\([\x01-\x09\x0b\x0c\x0d-\x7f]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]))))*(((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(\x22)))@((([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))$/i,
            url: /^(?:(?:https?|ftp):\/\/)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:\/[^\s]*)?$/i,
            domain: /^(?:[a-zA-Z0-9]+(?:\-*[a-zA-Z0-9])*\.)+[a-zA-Z]{2,6}$/
        }, l = s.messages = {
            required: "{0} is required",
            acceptance: "{0} must be accepted",
            min: "{0} must be greater than or equal to {1}",
            max: "{0} must be less than or equal to {1}",
            range: "{0} must be between {1} and {2}",
            length: "{0} must be {1} characters",
            minLength: "{0} must be at least {1} characters",
            maxLength: "{0} must be at most {1} characters",
            rangeLength: "{0} must be between {1} and {2} characters",
            oneOf: "{0} must be one of: {1}",
            equalTo: "{0} must be the same as {1}",
            digits: "{0} must only contain digits",
            number: "{0} must be a number",
            "int": "{0} must be a whole number",
            email: "{0} must be a valid email",
            url: "{0} must be a valid url",
            domain: "{0} must be a valid domain",
            inlinePattern: "{0} is invalid"
        }, c = s.labelFormatters = {
            none: function(e) {
                return e
            },
            sentenceCase: function(e) {
                return e.replace(/(?:^\w|[A-Z]|\b\w)/g, function(e, t) {
                    return 0 === t ? e.toUpperCase() : " " + e.toLowerCase()
                }).replace(/_/g, " ")
            },
            label: function(e, t) {
                return t.labels && t.labels[e] || c.sentenceCase(e, t)
            }
        }, u = s.validators = function() {
            var e = String.prototype.trim ? function(e) {
                return null === e ? "" : String.prototype.trim.call(e)
            }
            : function(e) {
                var t = /^\s+/, i = /\s+$/;
                return null === e ? "" : e.toString().replace(t, "").replace(i, "")
            }, i = function(e) {
                return t.isNumber(e) || t.isString(e) && e.match(a.number)
            }, n = function(i) {
                return !(t.isNull(i) || t.isUndefined(i) || t.isString(i) && "" === e(i) || t.isArray(i) && t.isEmpty(i))
            };
            return {
                fn: function(e, i, n, o, s) {
                    return t.isString(n) && (n = o[n]), n.call(o, e, i, s)
                },
                required: function(e, i, o, s, r) {
                    var a = t.isFunction(o) ? o.call(s, e, i, r): o;
                    return a || n(e) ? a&&!n(e) ? this.format(l.required, this.formatLabel(i, s)) : void 0 : !1
                },
                acceptance: function(e, i, n, o) {
                    return "true" === e || t.isBoolean(e) && e!==!1 ? void 0 : this.format(l.acceptance, this.formatLabel(i, o))
                },
                min: function(e, t, n, o) {
                    return !i(e) || n > e ? this.format(l.min, this.formatLabel(t, o), n) : void 0
                },
                max: function(e, t, n, o) {
                    return !i(e) || e > n ? this.format(l.max, this.formatLabel(t, o), n) : void 0
                },
                range: function(e, t, n, o) {
                    return !i(e) || e < n[0] || e > n[1] ? this.format(l.range, this.formatLabel(t, o), n[0], n[1]) : void 0
                },
                length: function(e, i, n, o) {
                    return t.isString(e) && e.length === n ? void 0 : this.format(l.length, this.formatLabel(i, o), n)
                },
                minLength: function(e, i, n, o) {
                    return !t.isString(e) || e.length < n ? this.format(l.minLength, this.formatLabel(i, o), n) : void 0
                },
                maxLength: function(e, i, n, o) {
                    return !t.isString(e) || e.length > n ? this.format(l.maxLength, this.formatLabel(i, o), n) : void 0
                },
                rangeLength: function(e, i, n, o) {
                    return !t.isString(e) || e.length < n[0] || e.length > n[1] ? this.format(l.rangeLength, this.formatLabel(i, o), n[0], n[1]) : void 0
                },
                oneOf: function(e, i, n, o) {
                    return t.include(n, e) ? void 0 : this.format(l.oneOf, this.formatLabel(i, o), n.join(", "))
                },
                equalTo: function(e, t, i, n, o) {
                    return e !== o[i] ? this.format(l.equalTo, this.formatLabel(t, n), this.formatLabel(i, n)) : void 0
                },
                pattern: function(e, t, i, o) {
                    return n(e) && e.toString().match(a[i] || i) ? void 0 : this.format(l[i] || l.inlinePattern, this.formatLabel(t, o), i)
                },
                url: function(e, t, i, n) {
                    return a.url.test(e) ? void 0 : this.format(l.url, this.formatLabel(t, n))
                },
                domain: function(e, t, i, n) {
                    return a.domain.test(e) ? void 0 : this.format(l.domain, this.formatLabel(t, n))
                },
                email: function(e, t, i, n) {
                    return a.email.test(e) ? void 0 : this.format(l.email, this.formatLabel(t, n))
                },
                number: function(e, t, i, n) {
                    return isNaN(parseFloat(e)) ||!isFinite(e) ? this.format(l.number, this.formatLabel(t, n)) : void 0
                },
                "int": function(e, t, i, n) {
                    return 0 !== e%1 ? this.format(l.int, this.formatLabel(t, n)) : void 0
                }
            }
        }();
        return t.each(u, function(e, i) {
            u[i] = t.bind(u[i], t.extend({}, n, u))
        }), s
    }(t), e.Validation
}), jsonrpc = {}, jsonrpc.CallStack = function(e, t, i, n) {
    this._counter = 0, this._enterFn = e, this._exitFn = i, this._enterScope = t, this._exitScope = n
}, jsonrpc.CallStack.prototype = {
    enter: function() {
        this._counter = this._counter < 0 ? 1 : this._counter + 1, 1 === this._counter && this._enterFn.apply(this._enterScope, arguments)
    },
    exit: function() {
        this._counter -= 1, 0 === this._counter && this._exitFn.apply(this._exitScope, arguments)
    }
}, jsonrpc.DelayedTask = function(e, t, i) {
    this._fn = e || function() {}, this._scope = t || void 0, this._args = i || [], this._id = null
}, jsonrpc.Observable = function() {
    this._listeners = []
}, jsonrpc.Observable.prototype = {
    bind: function(e, t) {
        var i = {
            fn: e,
            scope: t || this
        };
        return this._listeners.push(i), i
    },
    unbind: function(e) {
        var t = this._listeners.indexOf(e);
        - 1 !== t && this._listeners.splice(t, 1)
    },
    trigger: function() {
        var e;
        for (e = 0; e < this._listeners.length; e += 1)
            this._listeners[e].fn.apply(this._listeners[e].scope, arguments)
    }
}, jsonrpc.DelayedTask.prototype = {
    delay: function(e, t, i, n) {
        var o = this;
        this._fn = t || this._fn, this._scope = i || this._scope, this._args = n || this._args, this.cancel(), this._id = window.setInterval(function() {
            window.clearInterval(o._id), o._id = null, o._fn.apply(o._scope, o._args)
        }, e)
    },
    cancel: function() {
        this._id && (window.clearInterval(this._id), this._id = null)
    }
}, jsonrpc.JsonRpc = function(e) {
    this._url = e, this.loading = new jsonrpc.Observable, this.loaded = new jsonrpc.Observable, this.unhandledFailure = new jsonrpc.Observable, this._loadingState = new jsonrpc.CallStack(this.loading.trigger, this.loading, this.loaded.trigger, this.loaded), this._requests = [], this._batchingMilliseconds = 10, this._delayedTask = new jsonrpc.DelayedTask
}, jsonrpc.JsonRpc.prototype = {
    setBatchingMilliseconds: function(e) {
        this._batchingMilliseconds = e
    },
    call: function() {
        var e = this._getParams.apply(this, arguments);
        this._loadingState.enter(), this._requests.push(e), this._batchingMilliseconds ? this._delayedTask.delay(this._batchingMilliseconds, this._sendRequests, this) : this._sendRequests()
    },
    _sendRequests: function() {
        function e(e, o, s) {
            var r;
            if (e)
                r = i._isArray(o) ? o : [o];
            else 
                for (r = [], t = 0; t < n.length; t += 1)
                    r[t] = {
                        id: t,
                        error: {
                            message: o
                        }
                    };
            i._handleResponses(n, r, s)
        }
        var t, i = this, n = this._requests, o = [], s = [];
        for (i._requests = [], t = 0; t < n.length; t += 1)
            n[t].request.id = t, n[t].secure ? s.push(n[t].request) : o.push(n[t].request);
        o.length > 0 && (1 === o.length && (o = o[0]), i._doJsonPost(i._url, o, e)), s.length > 0 && (1 === s.length && (s = s[0]), i._doJsonpGet(i._url, s, e))
    },
    _handleResponses: function(e, t, i) {
        var n, o, s;
        for (n = 0; n < t.length; n += 1)
            o = t[n], s = e[o.id], this._handleResponse(s, o, i)
    },
    _handleResponse: function(e, t, i) {
        var n=!t.error, o = n ? t.result : t.error.message;
        this._loadingState.exit(), n ? e.success.call(e.scope, o, i) : e.failure.call(e.scope, o, i), e.callback.call(e.scope, n, o, i)
    },
    _getParams: function() {
        var e = this, t = Array.prototype.slice.call(arguments), i = {
            request: {
                jsonrpc: "2.0",
                method: t.shift()
            }
        };
        for (i.request.params = []; t.length > 1&&!this._isFunction(t[0]);)
            i.request.params.push(t.shift());
        return this._isFunction(t[0]) ? (i.success = t[0], i.scope = t[1]) : (i.success = t[0].success, i.failure = t[0].failure, i.callback = t[0].callback, i.scope = t[0].scope, i.secure=!!t[0].secure), i.success = i.success || function() {}, i.failure = i.failure || function() {
            e.unhandledFailure.trigger.apply(e.unhandledFailure, arguments)
        }, i.callback = i.callback || function() {}, i
    },
    _isArray: function(e) {
        return "[object Array]" === Object.prototype.toString.apply(e)
    },
    _isFunction: function(e) {
        return "[object Function]" === Object.prototype.toString.apply(e)
    },
    _beforeSend: function() {},
    _doJsonPost: function(e, t, i) {
        var n = Weebly.jQuery.ajax({
            type: "POST",
            url: e,
            cache: !1,
            contentType: "application/json; charset=UTF-8",
            dataType: "json",
            beforeSend: this._beforeSend,
            data: JSON.stringify(t),
            headers: {
                "x-wtok": Weebly.RequestToken
            }
        }).done(function(e) {
            var t = n.getResponseHeader("Content-Type");
            200 !== n.status ? i(!1, 'Expected HTTP response "200 OK", found "' + n.status + " " + n.statusText + '"', n) : 0 !== t.indexOf("application/json") ? i(!1, 'Expected JSON encoded response, found "' + t + '"', n) : i(!0, e, n)
        })
    },
    _doJsonpGet: function(e, t, i) {
        e = "https://" + _W.securePrefix + e, Weebly.jQuery.ajax({
            type: "GET",
            url: e,
            cache: !1,
            jsonpCallback: "WJsonp",
            dataType: "jsonp",
            beforeSend: this._beforeSend,
            data: t,
            headers: {
                "x-wtok": Weebly.RequestToken
            }
        }).done(function(e) {
            i(!0, e)
        })
    }
}, _wAMD.define("vendor/jsonrpc", function() {}), Weebly.Restrictions = {
    settings: {},
    services: [],
    source: "main",
    level: 0,
    freeLevel: 0,
    defaultSource: "main",
    disabledLevel: 1e7,
    defaultLevel: "all",
    starterLevel: 5,
    proLevel: 10,
    businessLevel: 15,
    designerSitesService: "Weebly.designerSites",
    unlimitedValue: "-1",
    hasAccess: function(e) {
        var t = Weebly.Restrictions, i = t.getMinimumLevel(e);
        return t.level >= i
    },
    hasService: function(e) {
        var t = Weebly.Restrictions;
        return t.level == e?!0 : - 1 != t.services.indexOf(e)
    },
    getPlans: function() {
        var e = Weebly.Restrictions, t = [e.freeLevel, e.starterLevel, e.proLevel, e.businessLevel], i = [];
        for (index in t)
            planid = t[index], i.push({
                plan: planid,
                name: e.levelToName(planid)
            });
        return i
    },
    accessValue: function(e) {
        var t = Weebly.Restrictions;
        if (!t.settings[e])
            return "";
        for (var i = "", n = t.level; n >= t.freeLevel; n--)
            if (i = t.settings[e][source] && "undefined" != typeof t.settings[e][source][n] ? t.settings[e][source][n] : t.settings[e][t.defaultSource][n], "undefined" != typeof i)
                return i;
        return ""
    },
    isUnlimited: function(e) {
        var t = Weebly.Restrictions, i = t.accessValue(e);
        return i == t.unlimitedValue
    },
    featureEnabled: function(e) {
        var t = Weebly.Restrictions, i = t.getMinimumLevel(e);
        return i != t.disabledLevel
    },
    isUpgradable: function(e) {
        var t = Weebly.Restrictions;
        if (!t.settings[e])
            return !1;
        var i = t.settings[e][source] ? source: t.defaultSource, n=!1;
        return $H(t.settings[e][i]).keys().each(function(e) {
            !n&&!isNaN(parseFloat(e)) && e > t.level && (n = e)
        }), n
    },
    getMinimumLevel: function(e) {
        var t = Weebly.Restrictions;
        if (!t.settings[e])
            return t.freeLevel;
        var i = t.settings[e][source] ? source: t.defaultSource, n = t.settings[e][i].minimum_level;
        return n || t.freeLevel
    },
    requiredService: function(e) {
        var t = Weebly.Restrictions;
        return t.getMinimumLevel(e)
    },
    requiresUpgrade: function(e) {
        var t = Weebly.Restrictions, i = t.getMinimumLevel(e);
        return i && i != t.freeLevel
    },
    addService: function(e) {
        var t = Weebly.Restrictions;
        t.services.push(e)
    },
    removeService: function(e) {
        var t = Weebly.Restrictions;
        t.services = t.services.without(e)
    },
    hasNewElementAccess: function() {
        var e = Weebly.Restrictions, t = $$("#secondlist .outside_top form");
        if (1 == t.size()) {
            var i = t[0].idfield.value.replace(/[^\d]/g, "");
            return e.hasAccess(i)
        }
        return !0
    },
    getSiteLimit: function() {
        var e = Weebly.Restrictions;
        return e.siteLimit && e.siteLimit > 0 ? e.siteLimit : e.accessValue("site_limit")
    },
    levelToName: function(e) {
        var t = Weebly.Restrictions;
        switch ("string" == typeof e && (e = parseInt(e)), e) {
        case t.freeLevel:
            return _W.tl("Free");
        case t.proLevel:
            return _W.tl("Pro");
        case t.starterLevel:
            return _W.tl("Starter");
        case t.businessLevel:
            return _W.tl("Business");
        default:
            return ""
        }
    },
    getNextLevel: function(e) {
        var t = Weebly.Restrictions;
        switch (e) {
        case t.freeLevel:
            return t.starterLevel;
        case t.starterLevel:
            return t.proLevel;
        case t.proLevel:
            return t.businessLevel;
        case t.businessLevel:
        default:
            return !1
        }
    },
    hasPro: function() {
        var e = Weebly.Restrictions;
        return e.level >= e.proLevel
    },
    hasBusiness: function() {
        var e = Weebly.Restrictions;
        return e.level >= e.businessLevel
    }
}, _wAMD.define("legacy/weebly_restrictions", function(e) {
    return function() {
        var t;
        return t || e.Weebly.Restrictions
    }
}(this)), "undefined" == typeof Effect)throw "accordion.js requires including script.aculo.us' effects.js library!";
var accordion = Class.create();
accordion.prototype = {
    showAccordion: null,
    currentAccordion: null,
    duration: null,
    effects: [],
    animating: !1,
    initialize: function(e, t) {
        if (!$(e))
            throw e + " doesn't exist!";
        this.options = Object.extend({
            resizeSpeed: 8,
            classNames: {
                toggle: "accordion_toggle",
                toggleActive: "accordion_toggle_active",
                content: "accordion_content"
            },
            defaultSize: {
                height: null,
                width: null
            },
            direction: "vertical",
            onEvent: "click"
        }, t || {}), this.duration = .15 * (11 - this.options.resizeSpeed);
        var i = $$("#" + e + " ." + this.options.classNames.toggle);
        i.each(function(e) {
            if (Event.observe(e, this.options.onEvent, this.activate.bind(this, e), !1), "click" == this.options.onEvent && (e.onclick = function() {
                return !1
            }), "horizontal" == this.options.direction)
                var t = $({
                    width: "0px",
                    display: "none"
                });
            else 
                var t = $({
                    height: "0px",
                    display: "none"
                });
            this.currentAccordion = $(e.next(0)).setStyle(t)
        }.bind(this))
    },
    activate: function(e) {
        return this.animating?!1 : (this.effects = [], this.currentAccordion = $(e.next(0)), this.currentAccordion.setStyle({
            display: "block"
        }), this.currentAccordion.previous(0).addClassName(this.options.classNames.toggleActive), this.scaling = "horizontal" == this.options.direction ? $({
            scaleX: !0,
            scaleY: !1
        }) : $({
            scaleX: !1,
            scaleY: !0
        }), this.currentAccordion == this.showAccordion ? this.deactivate() : this._handleAccordion(), void 0)
    },
    deactivate: function() {
        var e = $({
            duration: this.duration,
            scaleContent: !1,
            scaleX: this.scaling.scaleX,
            scaleY: this.scaling.scaleY,
            transition: Effect.Transitions.sinoidal,
            queue: {
                position: "end",
                scope: "accordionAnimation"
            },
            scaleMode: {
                originalHeight: this.options.defaultSize.height ? this.options.defaultSize.height: this.currentAccordion.scrollHeight,
                originalWidth: this.options.defaultSize.width ? this.options.defaultSize.width: this.currentAccordion.scrollWidth
            },
            afterFinish: function() {
                this.showAccordion.setStyle({
                    height: "auto",
                    display: "none"
                }), this.showAccordion = null, this.animating=!1
            }.bind(this)
        });
        this.showAccordion.previous(0).removeClassName(this.options.classNames.toggleActive), new Effect.Scale(this.showAccordion, 0, e)
    },
    _handleAccordion: function() {
        var e = $({
            sync: !0,
            scaleFrom: 0,
            scaleContent: !1,
            scaleX: this.scaling.scaleX,
            scaleY: this.scaling.scaleY,
            transition: Effect.Transitions.sinoidal,
            scaleMode: {
                originalHeight: this.options.defaultSize.height ? this.options.defaultSize.height: this.currentAccordion.scrollHeight,
                originalWidth: this.options.defaultSize.width ? this.options.defaultSize.width: this.currentAccordion.scrollWidth
            }
        });
        this.effects.push(new Effect.Scale(this.currentAccordion, 100, e)), this.showAccordion && (this.showAccordion.previous(0).removeClassName(this.options.classNames.toggleActive), e = $({
            sync: !0,
            scaleContent: !1,
            scaleX: this.scaling.scaleX,
            scaleY: this.scaling.scaleY,
            transition: Effect.Transitions.sinoidal
        }), this.effects.push(new Effect.Scale(this.showAccordion, 0, e))), new Effect.Parallel(this.effects, {
            duration: this.duration,
            queue: {
                position: "end",
                scope: "accordionAnimation"
            },
            beforeStart: function() {
                this.animating=!0
            }.bind(this),
            afterFinish: function() {
                this.showAccordion && this.showAccordion.setStyle({
                    display: "none"
                }), $(this.currentAccordion).setStyle({
                    height: "auto"
                }), this.showAccordion = this.currentAccordion, this.animating=!1
            }.bind(this)
        })
    }
}, _wAMD.define("legacy/accordion", function() {}), window.Weebly || (Weebly = {}), Weebly.AllDialogs = {}, Weebly.Dialog = function(e, t) {
    function i() {
        a.height(document.body.offsetHeight);
        var e = r(window).height();
        e > document.body.offsetHeight && a.height(e)
    }
    function n(i, n) {
        e = i, n || c.empty(), c.append(e), e = c.children().first(), l.show(), s(), t.onOpen && t.onOpen.call(l, e)
    }
    function o() {
        var e = r("<div>"), i = t.outerClass || "";
        e.addClass(t.inner_class || "weebly-dialog-inner");
        var n = document.createElement("div");
        if (n.innerHTML = "<div class='weebly-dialog " + i + "'>" + "  <div class='weebly-dialog-border'>" + "    <div class='weebly-dialog-content'>" + "    </div>" + "  </div>" + "</div>", l = r(n.firstChild), l.find("div.weebly-dialog-content").first().append(e), t.showClose!==!1) {
            "function" != typeof showClose && (showClose = function() {
                d()
            });
            var o = r('<div class="weebly-dialog-close">').on("click", "function" == typeof t.closeFunction ? t.closeFunction : function() {
                d(!0)
            });
            e.append(o)
        }
        c = r("<div>"), e.append(c), t.iframe ? c.css("line-height", 0) : (t.width && c.width(t.width), t.height && c.height(t.height)), l.css("position", "absolute"), l.css("left", "-10000px"), l.css("z-index", t.zIndex || 999), r(document.body).append(l)
    }
    function s() {
        if (l) {
            var n = e[0].scrollHeight + 29, o = r(window).height(), s = l[0].scrollHeight;
            s = Math.max(n, s), l.css("top", r(window).scrollTop() + Math.max(10, Math.round(o / 2 - s / 2)) + "px"), l.css("left", Math.round(r(document.body).width() / 2 - l.width() / 2) + "px"), !t.noScrollbars && s > o - 20 ? (e.height(e.scrollHeight - (s - (o - 20))), t.iframe || e.css("overflow", "auto")) : t.iframe ? t.height && (e.height = t.height) : (e.css("height", ""), e.css("overflow", "")), a && a.is(":visible") && i()
        }
    }
    var r = jQuery;
    t || "object" != typeof e || e.nodeName ? (e = "string" == typeof e ? r("#" + e) : r(e), t = t || {}, Weebly.AllDialogs[e.attr("id")] = this) : (t = e, e = null), this.options = t;
    var a, l, c;
    this.setOption = function(e, t) {
        void 0 === t ? delete this.options[e] : this.options[e] = t
    }, this.open = function(s, c) {
        if (s && (t.url = s), c && (t.parameters = c), t.modal)
            if (a)
                a.show();
            else {
                a = r('<div class="weebly-blackout">').attr("id", t.blackoutID || ""), r(document.body).append(a), a.css("z-index", t.zIndex || 999);
                var u = void 0 !== t.blackoutOpacity ? t.blackoutOpacity : .5;
                a.css("opacity", u), i()
            }
        if (l || o(), t.modal && r(document.body).append(l), t.iframe) {
            var d = "";
            t.parameters && (d = r.param(t.parameters), d =- 1 == t.url.indexOf("?") ? "?" + d : "&" + d);
            var h = {};
            h.border = 0, t.width && (h.width = t.width + "px"), t.height && (h.height = t.width + "px"), n(r('<iframe src="' + t.url + d + '" frameborder="0">').css(h))
        } else 
            t.url ? new r.ajax(t.url, {
                type: t.method || "get",
                data: t.parameters,
                success: function(e) {
                    n(e)
                }
            }) : n(e.show(), !0)
    }, this.update = function(t) {
        e = t, c.empty().append(e), s()
    }, this.positionDialog = s;
    var u;
    r(window).on("resize", function() {
        !u && e && e.offsetWidth && (u=!0, s(), u=!1)
    });
    var d = this.close = function(e) {
        l && (t.onClose && t.onClose.call(l, e)===!1 || (a && a.hide(), l.hide()))
    };
    this.destroy = function() {
        a && a.remove(), c && c.remove()
    }
}, !window.getInnerHeight, _wAMD.define("legacy/weebly_dialog", function(e) {
    return function() {
        var t;
        return t || e.Weebly.Dialog
    }
}(this)), Ajax.JSONRequest = Class.create(Ajax.Base, function() {
    var e = 0, t = document.getElementsByTagName("head")[0];
    return {
        initialize: function($super, e, t) {
            $super(t), this.options.url = e, this.options.callbackParamName = this.options.callbackParamName || "callback", this.options.timeout = this.options.timeout || 10, this.options.invokeImmediately = Object.isUndefined(this.options.invokeImmediately)?!0 : this.options.invokeImmediately, this.options.invokeImmediately && this.request()
        },
        _cleanup: function() {
            this.timeout && (clearTimeout(this.timeout), this.timeout = null), this.transport && Object.isElement(this.transport) && this.transport.remove()
        },
        request: function() {
            var i = new Ajax.JSONResponse(this), n = this.options.callbackParamName, o = "_prototypeJSONPCallback_" + e++, s = function() {
                Object.isFunction(this.options.onComplete) && this.options.onComplete.call(this, i)
            }.bind(this);
            this.options.parameters[n] = o;
            var r = this.options.url + ((this.options.url.include("?") ? "&" : "?") + Object.toQueryString(this.options.parameters));
            window[o] = function(e) {
                this._cleanup(), window[o] = void 0, Object.isFunction(this.options.onSuccess) && (i.status = 200, i.statusText = "OK", i.setResponseContent(e), this.options.onSuccess.call(this, i)), s()
            }.bind(this), this.transport = new Element("script", {
                type: "text/javascript",
                src: r
            }), Object.isFunction(this.options.onCreate) && this.options.onCreate.call(this, i), t.appendChild(this.transport), this.timeout = setTimeout(function() {
                this._cleanup(), window[o] = Prototype.emptyFunction, Object.isFunction(this.options.onFailure) && (i.status = 504, i.statusText = "Gateway Timeout", this.options.onFailure.call(this, i)), s()
            }.bind(this), 1e3 * this.options.timeout)
        },
        toString: function() {
            return "[object Ajax.JSONRequest]"
        }
    }
}()), Ajax.JSONResponse = Class.create({
    initialize: function(e) {
        this.request = e
    },
    request: void 0,
    status: 0,
    statusText: "",
    responseJSON: void 0,
    responseText: void 0,
    setResponseContent: function(e) {
        this.responseJSON = e,
        this.responseText = Object.toJSON(e)
    }, getTransport : function() {
        return this.request ? this.request.transport : void 0
    }, toString: function() {
        return "[object Ajax.JSONResponse]"
    }
}), _wAMD.define("legacy/jsonp", function() {});
var tipsShown = 20, tooltipCallbacks = {};
_wAMD.define("legacy/weebly_tooltip", function() {}), Effect.Transitions.Elastic = function(e) {
    return - 1 * Math.pow(4, - 8 * e) * Math.sin((6 * e - 1) * 2 * Math.PI / 2) + 1
}, Effect.Transitions.SwingFromTo = function(e) {
    var t = 1.70158;
    return (e/=.5) < 1 ? .5 * e * e * (((t*=1.525) + 1) * e - t) : .5 * ((e -= 2) * e * (((t*=1.525) + 1) * e + t) + 2)
}, Effect.Transitions.SwingFrom = function(e) {
    var t = 1.70158;
    return e * e * ((t + 1) * e - t)
}, Effect.Transitions.SwingTo = function(e) {
    var t = 1.70158;
    return (e -= 1) * e * ((t + 1) * e + t) + 1
}, Effect.Transitions.Bounce = function(e) {
    return 1 / 2.75 > e ? 7.5625 * e * e : 2 / 2.75 > e ? 7.5625 * (e -= 1.5 / 2.75) * e + .75 : 2.5 / 2.75 > e ? 7.5625 * (e -= 2.25 / 2.75) * e + .9375 : 7.5625 * (e -= 2.625 / 2.75) * e + .984375
}, _wAMD.define("legacy/transitions", function() {}), function(e) {
    e.fn.up = function(e) {
        return this.eq(0).parent().closest(e || "*")
    }, e.fn.down = function(e) {
        return e ? this.eq(0).find(e || "*").eq(0) : this.eq(0).children(":first")
    };
    var t = 1;
    e.fn.identify = function() {
        var i = this.attr("id");
        if (!i && this.length) {
            do 
                i = "anonymous_element_" + t++;
            while (e("#" + i).length);
            this.attr("id", i)
        }
        return i
    }, e.fn.placeholder = function() {
        if (!("placeholder"in document.createElement("input"))) {
            var t, i;
            this.each(function(n, o) {
                i = o.getAttribute("placeholder"), i && "input" === o.nodeName.toLowerCase() && (t = e(o).on({
                    focus: function() {
                        o.value === i && (e(o).removeClass("wsite-placeholder"), o.value = "")
                    },
                    blur: function() {
                        o.value.length || (o.value = i, o.className += " wsite-placeholder")
                    }
                }), o.className += " wsite-placeholder", o.value = i)
            })
        }
    }, e.extend({
        isValidSelector: function(t) {
            try {
                e(t)
            } catch (i) {
                return !1
            }
            return !0
        }
    }), document.observe || (document.observe = function(t, i) {
        "dom:loaded" == t && e(document).ready(i)
    })
}(Weebly.jQuery), Weebly.evalJSON = function(json) {
    var cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g;
    cx.test(json) && (json = json.replace(cx, function(e) {
        return "\\u" + ("0000" + e.charCodeAt(0).toString(16)).slice( - 4)
    }));
    try {
        return eval("(" + json + ")")
    } catch (e) {}
    throw new SyntaxError("Badly formed JSON string: " + json)
}, _wAMD.define("legacy/jquery_utils", function() {}), Weebly = window.Weebly || {};
var currentBlog = {
    postId: 0,
    newPost: 0,
    title: "",
    categories: "",
    skipToComments: "",
    saving: 0,
    permalink: "",
    comments: "",
    date: ""
}, preloadedImages = Array(), lastEventID;
lastEventID = Array();
var hasFlash = void 0;
!function() {
    var e, t = {
        first_tip: "x-elements",
        selectTheme: "x-pages",
        tab_pages: "x-pages",
        secondPage: "x-pages-subpages",
        updatePages: "x-publish",
        addElement: "x-design"
    }, i = {};
    for (var n in t)
        e = t[n], i[e] = n;
    window.hasDoneUserEvent = function(e) {
        return !!(isClientSite&&!isMaster || userEvents[e] || t[e] && userEvents[t[e]] || i[e] && userEvents[i[e]])
    }
}(), Weebly.trackingArray = Array();
var errorDialog, retriableErrorDialog, retriableAjax, warningDialog, showDialogElement, confirmDialog, lastConfirmOk=!1, whoisDialog;
shortcut = {
    all_shortcuts: {},
    add: function(e, t, i) {
        var n = {
            type: "keydown",
            propagate: !1,
            disable_in_input: !1,
            target: document,
            keycode: !1
        };
        if (i)
            for (var o in n)
                "undefined" == typeof i[o] && (i[o] = n[o]);
        else 
            i = n;
        var s = i.target;
        "string" == typeof i.target && (s = document.getElementById(i.target)), e = e.toLowerCase();
        var r = function(n) {
            if (n = n || window.event, i.disable_in_input) {
                var o;
                if (n.target ? o = n.target : n.srcElement && (o = n.srcElement), 3 == o.nodeType && (o = o.parentNode), "INPUT" == o.tagName || "TEXTAREA" == o.tagName)
                    return 
            }
            n.keyCode ? code = n.keyCode : n.which && (code = n.which);
            var s = String.fromCharCode(code).toLowerCase();
            188 == code && (s = ","), 190 == code && (s = ".");
            var r = e.split("+"), a = 0, l = {
                "`": "~",
                1: "!",
                2: "@",
                3: "#",
                4: "$",
                5: "%",
                6: "^",
                7: "&",
                8: "*",
                9: "(",
                0: ")",
                "-": "_",
                "=": "+",
                ";": ":",
                "'": '"',
                ",": "<",
                ".": ">",
                "/": "?",
                "\\": "|"
            }, c = {
                esc: 27,
                escape: 27,
                tab: 9,
                space: 32,
                "return": 13,
                enter: 13,
                backspace: 8,
                scrolllock: 145,
                scroll_lock: 145,
                scroll: 145,
                capslock: 20,
                caps_lock: 20,
                caps: 20,
                numlock: 144,
                num_lock: 144,
                num: 144,
                pause: 19,
                "break": 19,
                insert: 45,
                home: 36,
                "delete": 46,
                end: 35,
                pageup: 33,
                page_up: 33,
                pu: 33,
                pagedown: 34,
                page_down: 34,
                pd: 34,
                left: 37,
                up: 38,
                right: 39,
                down: 40,
                f1: 112,
                f2: 113,
                f3: 114,
                f4: 115,
                f5: 116,
                f6: 117,
                f7: 118,
                f8: 119,
                f9: 120,
                f10: 121,
                f11: 122,
                f12: 123
            }, u = {
                shift: {
                    wanted: !1,
                    pressed: !1
                },
                ctrl: {
                    wanted: !1,
                    pressed: !1
                },
                alt: {
                    wanted: !1,
                    pressed: !1
                },
                meta: {
                    wanted: !1,
                    pressed: !1
                }
            };
            n.ctrlKey && (u.ctrl.pressed=!0), n.shiftKey && (u.shift.pressed=!0), n.altKey && (u.alt.pressed=!0), n.metaKey && (u.meta.pressed=!0);
            for (var d = 0; k = r[d], d < r.length; d++)
                "ctrl" == k || "control" == k ? (a++, u.ctrl.wanted=!0) : "shift" == k ? (a++, u.shift.wanted=!0) : "alt" == k ? (a++, u.alt.wanted=!0) : "meta" == k ? (a++, u.meta.wanted=!0) : k.length > 1 ? c[k] == code && a++ : i.keycode ? i.keycode == code && a++ : s == k ? a++ : l[s] && n.shiftKey && (s = l[s], s == k && a++);
            return a == r.length && u.ctrl.pressed == u.ctrl.wanted && u.shift.pressed == u.shift.wanted && u.alt.pressed == u.alt.wanted && u.meta.pressed == u.meta.wanted ? (t(n), Event.stop(n), !1) : !0
        };
        this.all_shortcuts[e] = {
            callback: r,
            target: s,
            event: i.type
        }, s.addEventListener ? s.addEventListener(i.type, r, !1) : s.attachEvent ? s.attachEvent("on" + i.type, r) : s["on" + i.type] = r
    },
    remove: function(e) {
        e = e.toLowerCase();
        var t = this.all_shortcuts[e];
        if (delete this.all_shortcuts[e], t) {
            var i = t.event, n = t.target, o = t.callback;
            n.detachEvent ? n.detachEvent("on" + i, o) : n.removeEventListener ? n.removeEventListener(i, o, !1) : n["on" + i]=!1
        }
    }
}, Weebly.keyTracker = {}, window.disableBackspaceExit=!0, document.observe("keydown", function(e) {
    if (!Event.element(e).hasClassName("editable-text")) {
        e.keyCode ? code = e.keyCode : e.which && (code = e.which);
        var t, i = String.fromCharCode(code).toLowerCase();
        if (e.target ? t = e.target : e.srcElement && (t = e.srcElement), 3 == t.nodeType && (t = t.parentNode), disableBackspaceExit && 8 == code && "INPUT" != t.tagName && "TEXTAREA" != t.tagName)
            return e.keyCode = 0, event.keyCode = 0, e.returnValue=!1, event.returnValue=!1, Event.stop(e), !1;
        Weebly.keyTracker[i] = 1, Weebly.keyTracker.p && Weebly.keyTracker.s && Weebly.keyTracker.u && showAbout()
    }
}), document.observe("keyup", function(e) {
    if (!Event.element(e).hasClassName("editable-text")) {
        e.keyCode ? code = e.keyCode : e.which && (code = e.which);
        var t = String.fromCharCode(code).toLowerCase();
        Weebly.keyTracker[t] = 0
    }
}), $.StealMouse = Class.create(), $.StealMouse.__class_name = "$.StealMouse", $.StealMouse.prototype.__class_name = "$.StealMouse", $_StealMouse = $.StealMouse, Object.extend($.StealMouse, {
    on: function(e) {
        e.__steal_mouseup = function(t) {
            $.StealMouse._sendMouseEvent(t, "mouseup", e)
        }, Event.observe(e.contentWindow.document, "mouseup", e.__steal_mouseup), e.__steal_mousedown = function(t) {
            $.StealMouse._sendMouseEvent(t, "mousedown", e)
        }, Event.observe(e.contentWindow.document, "mousedown", e.__steal_mousedown), e.__steal_mousemove = function(t) {
            $.StealMouse._sendMouseEvent(t, "mousemove", e)
        }
    },
    off: function() {
        Event.stopObserving(ifr.contentWindow.document, "mouseup", ifr.__steal_mouseup), Event.stopObserving(ifr.contentWindow.document, "mousedown", ifr.__steal_mousedown), Event.stopObserving(ifr.contentWindow.document, "mousemove", ifr.__steal_mousemove)
    },
    _sendMouseEvent: function(e, t, i) {
        var n = Position.cumulativeOffset(i), o = [0, 0], s = {
            x: n[0] + o[0],
            y: n[1] + o[1]
        };
        if (document.createEvent) {
            var r = document.createEvent("MouseEvents");
            r.initMouseEvent(t, !0, !1, window, e.detail, e.screenX, e.screenY, e.clientX + s.x, e.clientY + s.y, e.ctrlKey, e.altKey, e.shiftKey, e.metaKey, e.button, null), i.dispatchEvent(r)
        } else {
            var r = document.createEventObject();
            r.detail = e.detail, r.screenX = e.screenX, r.screenY = e.screenY, r.clientX = e.clientX + s.x, r.clientY = e.clientY + s.y, r.ctrlKey = e.ctrlKey, r.altKey = e.altKey, r.shiftKey = e.shiftKey, r.metaKey = e.metaKey, r.button = e.button, r.relatedTarget = e.relatedTarget, r.target = i, r.srcElement = i
        }
    }
});
var toJsonString;
!function() {
    function e(e) {
        return /[\x00-\x19\'\\]/.test(e) ? e.replace(/([\\'])/g, "\\$1").replace(/\r/g, "\\r").replace(/\n/g, "\\n").replace(/\t/g, "\\t").replace(/[\x00-\x19]/g, "") : e
    }
    toJsonString = function(t) {
        var i;
        switch (typeof t) {
        case"string":
            return "'" + e(t) + "'";
        case"number":
            return String(t);
        case"object":
            if (t) {
                var n = [];
                if (t.constructor == Array) {
                    for (var o = 0; o < t.length; o++) {
                        var s = toJsonString(t[o]);
                        s != i && (n[n.length] = s)
                    }
                    return "[" + n.join(",") + "]"
                }
                if (t.constructor == Date)
                    return "new Date(" + t.getTime() + ")";
                for (var r in t) {
                    var s = toJsonString(t[r]);
                    s != i && (n[n.length] = (/^[A-Za-z_]\w*$/.test(r) ? "'" + r + "'" + ":" : "'" + e(r) + "':") + s)
                }
                return "{" + n.join(",") + "}"
            }
            return "null";
        case"boolean":
            return String(t);
        case"function":
            return;
        case"undefined":
            return "null"
        }
    }
}(), Weebly.on = {
    textIsChanging: 0,
    currentTextElement: null,
    currentTextCallBack: null,
    textChange: function(e, t, i, n) {
        e = $(e), i || (i = 1500), e != Weebly.on.currentTextElement && "function" == typeof Weebly.on.currentTextElement && Weebly.on.currentTextCallBack(), (Weebly.on.currentTextElement != e || "undefined" != typeof n) && ("undefined" == typeof n ? (Weebly.on.currentTextElement = e, Weebly.on.currentTextCallBack = t, Weebly.on.myFunction = Weebly.on.textChange.bind(Weebly.on.textChange, e, t, i, e.value), setTimeout("Weebly.on.myFunction();", i)) : e && e.value && e.value == n || "" == e.value && "" == n ? (Weebly.on.currentTextCallBack(e.value), Weebly.on.currentTextElement = null, Weebly.on.currentTextCallBack = null) : (Weebly.on.myFunction = Weebly.on.textChange.bind(Weebly.on.textChange, e, t, i, e.value), setTimeout("Weebly.on.myFunction();", i)))
    }
}, Weebly.lightbox = {
    element: "",
    onHide: function() {},
    show: function(e) {
        if (e && e.element) {
            if (this.elementNode = $$(e.element)[0], this.element = e.element, e.button && e.button.onClick) {
                e.button.image = e.button.image ? e.button.image : "http://" + editorStatic + "/weebly/images/accept-button.jpg";
                var t = document.createElement("DIV");
                t.innerHTML = "<div style='margin-right: 20px; text-align: right;'><img src='http://" + editorStatic + "/weebly/images/spinner.gif' id='lightbox_spinner' style='position: relative; top: -10px; left: -5px; display: none;'/><a href='#' onmousedown='" + '$("focusMe").focus(); return false;' + "' onclick='" + e.button.onClick + "; return false;'><img src='" + e.button.image + "' style='border: 0;' id='lightbox_submitbtn'/></a></div>", $("weeblyLightboxButton").appendChild(t)
            }
            e.width || (e.width = this.elementNode.getWidth() + 20), e.height || (e.height = this.elementNode.getHeight() + $("weeblyLightboxButton").getHeight() + 50), $("weeblyLightboxContent").childNodes.length > 0 && $("weeblyLightboxContent").childNodes[0] && ($("weeblyLightboxContent").childNodes[0].style.display = "none", document.body.appendChild($("weeblyLightboxContent").childNodes[0])), e.padding = e.padding ? e.padding : 10, $("weeblyLightboxContent").style.padding = e.padding + "px", Weebly.lightbox.onHide = e.onHide ? e.onHide : function() {}, $("grayedOutFull").style.display = "block", $("weeblyLightbox").style.display = "block", $("weeblyLightboxClose") && $("weeblyLightboxClose").show(), e.options && e.options.hideClose && $("weeblyLightboxClose").hide();
            var i = (getInnerHeight() - e.height) / 2;
            i = i > 0 ? i : 10, e.animate ? new Effect.Morph($("weeblyLightboxInside"), {
                style: {
                    width: e.width + "px",
                    height: e.height + "px",
                    marginTop: i + "px"
                },
                afterFinish: function() {
                    Weebly.lightbox.insertContent()
                }
            }) : ($("weeblyLightboxInside").style.marginTop = i + "px", $("weeblyLightboxInside").style.width = e.width + "px", navigator.appVersion.match("MSIE 6") ? $("weeblyLightboxInside").style.height = e.height + "px" : $("weeblyLightboxInside").style.minHeight = e.height + "px", Weebly.lightbox.insertContent()), "function" == typeof e.onFinish && e.onFinish()
        }
    },
    insertContent: function() {
        $("weeblyLightboxContent").appendChild(Weebly.lightbox.elementNode), Weebly.lightbox.elementNode.style.display = "block"
    },
    hide: function() {
        Weebly.lightbox.onHide(), $("grayedOutFull").style.display = "none", $("weeblyLightbox").style.display = "none", $("weeblyLightboxContent").childNodes.length > 0 && $("weeblyLightboxContent").childNodes[0] && ($("weeblyLightboxContent").childNodes[0].style.display = "none", document.body.appendChild($("weeblyLightboxContent").childNodes[0])), $("weeblyLightboxButton").innerHTML = ""
    }
};
var currentHref = document.location.href.replace(/#.*$/, ""), firstTime = (new Date).getTime(), publishAfterPro=!1;
"undefined" != typeof doMonitorHref && document.observe("dom:loaded", function() {
    setInterval("monitorHref()", 200)
}), eval("var " + String.fromCharCode(102) + String.fromCharCode(99) + String.fromCharCode(99) + " = function(num) { return String.fromCharCode(num); };");
var ElementExtensions = {
    center: function(e, t, i) {
        e = $(e);
        var n = e.getDimensions(), o = document.viewport.getDimensions(), s = document.viewport.getScrollOffsets(), r = o.width / 2 + s.left - n.width / 2, a = o.height / 2 + s.top - n.height / 2;
        return t && t > r && (r = parseInt(t)), i && i > a && (a = parseInt(i)), e.setStyle({
            position: "absolute",
            top: Math.floor(a) + "px",
            left: Math.floor(r) + "px"
        }), e
    },
    getInnerText: function(e) {
        return Weebly.jQuery(e).text()
    },
    setInnerText: function(e, t) {
        return Weebly.jQuery(e).text(t)
    }
};
Element.addMethods(ElementExtensions), String.prototype.safeReplace = function(e, t) {
    return this.replace(e, (t + "").replace(/\$/g, "$$$$"))
}, _loadingLevel = 0, _wAMD.define("legacy/weebly_utils", ["legacy/weebly_restrictions"], function() {}), window.Weebly = window.Weebly || {};
var settingAnimations = 0, settingQuickExport = 0, settingTooltips = 0, uploadInProgress = 0, exiting = 0, ajax = "/weebly/getElements.php", ajaxStatusCheckTimeout = 4e3, ajaxStatusTimeoutGrowthFactor = 1;
Weebly.ajaxLog = [], window.onbeforeunload = function() {
    exiting = 1
};
var myGlobalHandlers = {
    onCreate: function(e, t) {
        if (t.request.times = {
            start: (new Date).getTime()
        }, window.swfu && swfu.currentUpload && $("upload" + swfu.currentUpload) && (t.request.concurrentUpload=!0), e.options.bgRequest || startWait(), e.parameters && e.parameters.pos && (Weebly.ajaxLog.push(e.parameters.pos), Weebly.ajaxLog.size() > 10 && Weebly.ajaxLog.shift()), e.options.requestHeaders = e.options.requestHeaders || {}, "undefined" != typeof currentSite && (e.options.requestHeaders["Weebly-Site-ID"] = currentSite), !e.options.requestHeaders["x-ajax-request-id"]) {
            var i = (new Date).getTime() + "" + Math.floor(999 * Math.random());
            e.options.requestHeaders["x-ajax-request-id"] = i
        }
        e.options.isRetry && e.options.previouslyAborted && (e.options.requestHeaders["x-ajax-abort-retry"] = ajaxStatusCheckTimeout), e.options.isRetry || Prototype.Browser.IE6 || setTimeout(function() {
            checkAjaxRequestStatus(e)
        }, ajaxStatusCheckTimeout)
    },
    onLoading: function(e, t) {
        t.request.times && t.request.times.start && (t.request.times.initialized = (new Date).getTime() - t.request.times.start)
    },
    onLoaded: function(e, t) {
        t.request.times && t.request.times.start && (t.request.times.sent = (new Date).getTime() - t.request.times.start, e.options.isRetry && t.request.times.sent > ajaxStatusCheckTimeout / 2 && (ajaxStatusCheckTimeout += ajaxStatusCheckTimeout * ajaxStatusTimeoutGrowthFactor, ajaxStatusTimeoutGrowthFactor = .9 * ajaxStatusTimeoutGrowthFactor))
    },
    onInteractive: function(e, t) {
        t.request.times && t.request.times.start&&!t.request.times.response && (t.request.times.response = (new Date).getTime() - t.request.times.start)
    },
    onComplete: function(e, t) {
        return 0 != t.status || e.aborted ? (t.request.times && t.request.times.start && (t.request.times.end = (new Date).getTime(), t.request.times.complete = t.request.times.end - t.request.times.start), e.options.bgRequest || endWait(), e.isRetriable() ? e.retry() : handleLogout(t, e), void 0) : (retriableErrFunc(e, t), !1)
    },
    onException: endWait
};
Ajax.Responders.register(myGlobalHandlers), Ajax.Request.addMethods({
    abort: function() {
        if (!this._complete) {
            this.transport.onreadystatechange = Prototype.emptyFunction, this.transport.abort(), this._complete=!0, this.aborted=!0;
            var e = new Ajax.Response(this);
            ["Abort", "Complete"].each(function(t) {
                try {
                    (this.options["on" + t] || Prototype.emptyFunction)(e, e.headerJSON), Ajax.Responders.dispatch("on" + t, this, e, e.headerJSON)
                } catch (i) {
                    this.dispatchException(i)
                }
            }, this)
        }
    },
    retry: function(e) {
        if (e || this.isRetriable()) {
            var t = this.options || {};
            t._retryCount = (t._retryCount || 0) + 1, t.isRetry=!0, t.previouslyAborted = this.aborted?!0 : !1, new Ajax.Request(this.url, t)
        }
    },
    isRetriable: function() {
        if (!this._complete)
            return !1;
        var e = this.getStatus(), t = this.options || {}, i = t.maxRetries || 1, n = t._retryCount || 0;
        return (408 == e || e > 1e4 || this.aborted) && i > n
    }
}), _wAMD.define("legacy/weebly_utils_ajax", function() {});
var noJump = 0, Pages = {
    Version: "0.5",
    Author: "David Rusenko",
    Company: "Weebly, Inc.",
    openPages: new Array("main"),
    currentNavPage: "load",
    optConfirmLoad: 0,
    optAnimations: 1,
    history: [],
    lastPage: "",
    load: function() {},
    pageConstructor: {
        main: {}
    },
    pageDestructor: {
        main: {}
    },
    pageWindows: {
        main: ["main"]
    },
    currentVar1: "empty",
    currentVar2: "empty",
    currentVar3: "empty",
    currentVar4: "empty",
    currentVar5: "empty",
    go: function() {
        var e = this, t = arguments;
        if (Weebly.Commerce && window.editorApp && editorApp.isStorePanelOpen) {
            var i = require("backbone-all");
            i.preventClose(function() {
                e._go.apply(e, t)
            })
        } else 
            e._go.apply(e, t)
    },
    _go: function(e, t, i, n, o, s) {
        if (window.editorApp && editorApp.vent.trigger("editor:pages:go", arguments), Pages.updateHistory(e), Weebly.Elements && Weebly.Elements.unselectElement(), !Weebly.BlogEditor || Weebly.BlogEditor.tryPageSwitch(arguments)) {
            if ("saveBlogPost" !== e && Weebly.BlogEditor && Weebly.BlogEditor.isEditing&&!Weebly.BlogEditor.isSaving && Weebly.mobile.isMobileEditor())
                return Weebly.mobile.isMobileEditor(!1), Pages.go("mobileView"), void 0;
            var r = "undefined" == typeof t || "" == t || "undefined" == t ? 0: 1, a = "undefined" == typeof i || "" == i || "undefined" == i ? 0: 1, l = "undefined" == typeof n || "" == n || "undefined" == n ? 0: 1, c = "undefined" == typeof o || "" == o || "undefined" == o ? 0: 1, u = "undefined" == typeof s || "" == s || "undefined" == s ? 0: 1;
            if (e != Pages.currentNavPage || t != Pages.currentVar1 && r || i != Pages.currentVar2 && a || n != Pages.currentVar3 && l || o != Pages.currentVar4 && c || s != Pages.currentVar5 && u) {
                Pages.handleMain(e), Pages.closeAll(e), 1 == noJump ? noJump = 0 : window.scrollTo(0, 0);
                var d = Pages.pageConstructor[e];
                d && "main" != e && (d.element && Pages.showElement(d.element), d.go && d.go(t, i, n, o, s)), window.litePageChange || Pages.optConfirmLoad || Pages.confirmLoad(e, t, i, n, o, s), Pages.currentNavPage = e, Pages.currentVar1 = "undefined" == typeof t || "" == t || "undefined" == t ? "empty" : t, Pages.currentVar2 = "undefined" == typeof i || "" == i || "undefined" == i ? "empty" : i, Pages.currentVar3 = "undefined" == typeof n || "" == i || "undefined" == n ? "empty" : n, Pages.currentVar4 = "undefined" == typeof o || "" == i || "undefined" == o ? "empty" : o, Pages.currentVar5 = "undefined" == typeof s || "" == i || "undefined" == s ? "empty" : s, Weebly.Events && Weebly.Events.fireEvent("navPageChange", e)
            }
            - 1 == Pages.openPages.indexOf(e) && Pages.openPages.push(e)
        }
    },
    showElement: function(e) {
        Pages.optAnimations ? setTimeout("Effect.Appear('" + e + "');", 1e3) : Element.show(e)
    },
    handleMain: function(e) {
        if (Pages.pageWindows[e].indexOf("main")>-1)
            Pages.pageConstructor.main && (Pages.pageConstructor.main.element && (Pages.optAnimations ? setTimeout("Effect.Fade('" + Pages.pageConstructor.main.element + "')", 1e3) : Element.hide(Pages.pageConstructor.main.element)), Pages.pageConstructor.main.go && Pages.pageConstructor.main.go()), - 1 == Pages.openPages.indexOf("main") && Pages.openPages.push("main");
        else {
            if (Pages.pageDestructor.main) {
                if (Pages.pageDestructor.main.element) {
                    var t = "100%";
                    Element.setStyle(Pages.pageDestructor.main.element, {
                        height: t
                    }), Pages.optAnimations ? Effect.Appear(Pages.pageDestructor.main.element) : Element.show(Pages.pageDestructor.main.element)
                }
                Pages.pageDestructor.main.go && Pages.pageDestructor.main.go()
            }
            - 1 == Pages.openPages.indexOf("main") && Pages.openPages.push("main")
        }
        "main" == e && "visible" == $("grayedOut").style.visibility && Element.hide("grayedOut")
    },
    confirmLoad: function(e, t, i, n, o, s) {
        - 1 == Pages.openPages.indexOf(e) && Pages.openPages.push(e), "function" == typeof t && (t = ""), _goPageHistory("page=" + e + "&1=" + encodeURIComponent(t) + "&2=" + encodeURIComponent(i) + "&3=" + encodeURIComponent(n) + "&4=" + encodeURIComponent(o) + "&5=" + encodeURIComponent(s))
    },
    closeAll: function(e) {
        for (j = Pages.openPages.length - 1; j >= 0; j--) {
            var t = Pages.pageWindows[e].indexOf(Pages.openPages[j]);
            if ( - 1 == t && "main" != Pages.openPages[j]) {
                var i = Pages.pageDestructor[Pages.openPages[j]];
                i && (i.element && (Pages.optAnimations ? Effect.Fade($(i.element)) : Element.hide($(i.element))), i.go && i.go()), Pages.openPages.splice(j, 1)
            }
        }
    },
    returnToEditor: function() {
        Pages.openPages.indexOf("goBlogPost")>-1 ? Pages.go("goBlogPost") : Pages.go("main")
    },
    updateHistory: function(e) {
        e != Pages.lastPage && (Pages.history.push(e), Pages.history.size() > 10 && Pages.history.shift()), Pages.lastPage = e
    },
    goToPreviousPage: function() {
        if (Pages.history.size() > 1) {
            Pages.history.pop();
            var e = Pages.history.pop();
            Pages.go(e)
        } else 
            Pages.go("editMenu")
    }
}, _goPageIframe;
_wAMD.define("legacy/weebly_pages", function() {}), Prototype.Browser.IE6 = Prototype.Browser.IE && 6 == parseInt(navigator.userAgent.substring(navigator.userAgent.indexOf("MSIE") + 5));
var copy_site_id = null;
document.observe("dom:loaded", function() {
    "object" == typeof Weebly.CopyingSites && Weebly.CopyingSites.each(function(e) {
        updateCopySite(e)
    })
}), Event.observe(document, "mousedown", clickHandler), Weebly.blueBox = {
    open: function(e, t) {
        Weebly.blueBox.callbacks = {}, t = t ? t : {}, "object" == typeof t && "function" == typeof t.onLoad && (Weebly.blueBox.callbacks.onLoad = t.onLoad), "object" == typeof t && "function" == typeof t.onClose && (Weebly.blueBox.callbacks.onClose = t.onClose), $("weebly-bluebox").setStyle({
            height: $(document.body).getHeight() + "px"
        }), $("weebly-bluebox").show(), $(e) && $("weebly-bluebox-content").insert({
            top: $(e).show()
        }), $("weebly-bluebox-bg").onclick = Weebly.blueBox.close, $("weebly-bluebox").down(".weebly-bluebox-close").onclick = Weebly.blueBox.close, $("weebly-bluebox-content").setStyle({
            padding: ""
        }), t.noPadding && $("weebly-bluebox-content").setStyle({
            padding: "0px"
        });
        var i = $("weebly-bluebox-container").getHeight(), n = document.viewport.getHeight() / 2 - i / 2 + document.viewport.getScrollOffsets()[1];
        n = 10 > n ? 10 : n, $("weebly-bluebox-container").setStyle({
            top: n + "px"
        }), $("weebly-bluebox-bordercontainer").setStyle({
            height: i + 26 + "px",
            top: n - 13 + "px",
            position: "absolute"
        }), $("weebly-bluebox-border").setStyle({
            height: i + 26 + "px",
            opacity: "0.5"
        }), t.width && ($("weebly-bluebox-container").setStyle({
            width: t.width
        }), $("weebly-bluebox-border").setStyle({
            width: t.width.replace("px", "") - - 40 + "px"
        })), "function" == typeof Weebly.blueBox.callbacks.onLoad && Weebly.blueBox.callbacks.onLoad()
    },
    close: function() {
        document.body.appendChild($("weebly-bluebox-content").childNodes[0].hide()), $("weebly-bluebox-content").update(""), $("weebly-bluebox").hide(), "function" == typeof Weebly.blueBox.callbacks.onClose && Weebly.blueBox.callbacks.onClose()
    }
};
var adminLogin = 0, activeSite = window.activeSite || "", siteAreaOpen=!1;
activeSetting = "", originalScrollAreaHeight=!1, _wAMD.define("legacy/weebly_user_home", function() {}), jQuery(document).on("click", "#error", function() {
    new Effect.Fade("error")
}).on("click", "#tip11", function() {
    Effect.Fade("tip11")
}), _wAMD.define("legacy/weebly_user_home_style", function() {}), Weebly = window.Weebly || {}, Weebly.DropDowns = {
    Version: "0.5",
    Author: "David Rusenko",
    Company: "Weebly, Inc.",
    activeMenu: null,
    dropdownsRef: [],
    dropdownsArray: [],
    init: function() {
        Event.observe(document, "click", Weebly.DropDowns.handleClick)
    },
    register: function(e) {
        0 == this.dropdownsArray.length && this.init(), this.dropdownsRef[e.id] = e, this.dropdownsArray.push(e)
    },
    destroy: function(e) {
        if (e.button && (e.button.pare, e.button.onclick = null), e.element && e.element.childNodes.length > 0)
            for (var t = 0; t < e.element.childNodes.length; t++)
                e.element.childNodes[t].onclick = null;
        e.element && e.element.parentNode && "weeblyDropDown" == e.element.parentNode.className && e.element.parentNode.parentNode.removeChild(e.element.parentNode);
        for (var t = 0; t < Weebly.DropDowns.dropdownsArray.length; t++)
            Weebly.DropDowns.dropdownsArray[t] == e && Weebly.DropDowns.dropdownsArray.splice(t, 1);
        delete Weebly.DropDowns.dropdownsRef[e.id], delete e
    },
    setValue: function(e, t, i) {
        var n = Weebly.DropDowns.dropdownsRef[e];
        n.open(), n.options.rowFunction ? n.close($(e + "-" + t), i) : n.close(t, i)
    },
    handleClick: function(e) {
        var t;
        e.target ? t = e.target : e.srcElement && (t = e.srcElement), 3 == t.nodeType && (t = t.parentNode)
    },
    refreshChildren: function(e) {
        $H(this.dropdownsRef).each(function(t) {
            "object" == typeof t.value && $(t.value.element).descendantOf(e) && t.value.adjustScroll()
        })
    }
}, Weebly.DropDown = Class.create(), Weebly.DropDown.prototype = {
    isOpen: !1,
    initialize: function(e) {
        e = $(e), this.options = Object.extend({
            width: "95",
            height: "21",
            openWidth: "195",
            openHeight: null,
            border: "1px solid #9f9f9f",
            marginRight: 0,
            background: "#FFFFFF url(http://" + editorStatic + "/weebly/images/dropdown_background.jpg) repeat-x",
            overflowY: "scroll",
            dropDownButton: {
                src: "http://" + editorStatic + "/weebly/images/dropdown_arrow.gif",
                width: "21",
                height: "18"
            },
            containerClass: "",
            fieldName: e.name,
            rowFunction: null,
            rowCompareFunction: null,
            generateValueFunction: null,
            generateContentsFunction: null,
            rowMargin: "3px 0 2px 0",
            type: null,
            onClose: function() {},
            onOpen: function() {},
            updateFunction: function() {},
            noInitialUpdate: !0,
            rowHoverColor: "#B5D2F0",
            noRefresh: null,
            scaleBy: 20,
            zIndex: 20,
            availableValues: []
        }, arguments[1] || {}), this.id = Math.floor(10000001 * Math.random());
        var t = this.id;
        this.formEl = e, this.values = [], "YesNo" == this.options.type && (this.options.rowFunction = function(e) {
            return e > 1 ? void 0 : 0 == e%2 ? ["Yes", "Yes"] : ["No", "No"]
        }), this.options.availableValues.size() > 0 && (this.options.rowFunction = function(e) {
            return "object" == typeof this.availableValues[e] ? this.availableValues[e] : !1
        });
        var i = document.createElement("DIV");
        i.className = "weeblyDropDown", i.style.position = "relative", i.style.marginRight = this.options.marginRight + "px", i.style.width = this.options.width + "px", this.container = $(i);
        var n = document.createElement("DIV");
        n.id = this.id + "_dropdownButton", n.className = "dropdownButton", n.onclick = function() {
            Weebly.DropDowns.dropdownsRef[t].open()
        }, n.style.position = "absolute", n.style.width = this.options.dropDownButton.width + "px", n.style.height = this.options.dropDownButton.height + "px", n.style.cursor = "pointer", n.style.zIndex = this.options.zIndex - - 1, n.style.top = "3px", n.style.right = 0, n.style.background = "transparent url('" + this.options.dropDownButton.src + "') no-repeat top left", this.button = n;
        var o = document.createElement("DIV");
        if (o.id = this.id + "_dropdownContainer", o.style.width = this.options.width + "px", o.style.height = Prototype.Browser.IE ? this.options.height - - 0 + "px" : this.options.height + "px", o.style.position = "absolute", o.style.border = this.options.border, o.style.background = this.options.background, o.style.zIndex = this.options.zIndex, o.style.overflowX = "hidden", o.style.overflowY = "hidden", this.options.containerClass && $(o).addClassName(this.options.containerClass), this.element = $(o), i.appendChild(n), i.appendChild(o), e.parentNode.insertBefore(i, e.nextSibling), this.options.rowFunction) {
            for (var s, r = 0; (s = this.options.rowFunction(r))&&!(r > 1e3);) {
                var a = document.createElement("DIV");
                a.id = this.id + "-" + r, t = this.id, this.options.noRefresh, a.onclick = function() {
                    Weebly.DropDowns.dropdownsRef[t].close(this)
                };
                var l = this.options.rowHoverColor;
                a.onmouseover = function() {
                    this.style.background = l
                }, a.onmouseout = function() {
                    this.style.background = "none"
                }, a.style.height = this.options.height + "px", a.style.overflow = "hidden", a.style.cursor = "pointer", a.style.paddingLeft = "8px", a.innerHTML = "<textarea style='display: none;' name='val'>" + s[0] + "</textarea><div style='line-height:" + this.options.height + "px;'>" + s[1] + "</div>", s[2] && this.element.appendChild(s[2]), this.element.appendChild(a), e.value && (this.options.rowCompareFunction ? this.options.rowCompareFunction(e.value, s) : e.value == s[0]) && (this.element.scrollTop = 0, this.element.scrollTop = this.getScrollHeight(r), this.lastEl = a, this.scrollMe(this.getScrollHeight(r)), this.options.noInitialUpdate || this.options.updateFunction(e.value)), this.values[r] = s[0], r++
            }
            this.options.openHeight || (this.options.openHeight = r * this.options.height), e.value && this.lastEl || (this.lastEl = o.firstChild)
        } else if ("function" == typeof this.options.generateValueFunction && "function" == typeof this.options.generateContentsFunction) {
            var a = document.createElement("DIV");
            a.id = this.id + "-value", a.innerHTML = this.options.generateValueFunction(this), a.style.width = this.options.openWidth - 10 + "px", a.style.height = this.options.height + "px", a.style.cursor = "pointer";
            var l = this.options.rowHoverColor;
            a.onmouseover = function() {
                this.style.background = l
            }, a.onmouseout = function() {
                this.style.background = "none"
            }, t = this.id, a.onclick = function() {
                Weebly.DropDowns.dropdownsRef[t].close(this)
            }, this.element.appendChild(a);
            var c = document.createElement("DIV");
            c.id = this.id + "-contents", c.style.width = this.options.openWidth - 10 + "px", c.style.display = "none", this.element.appendChild(c)
        }
        Weebly.DropDowns.register(this)
    },
    open: function() {
        var e = this.element.scrollTop;
        if (this.element.scrollTop = 0, this.checkGone(), 1 != this.isOpen) {
            if (Weebly.DropDowns.activeMenu && Weebly.DropDowns.dropdownsRef[Weebly.DropDowns.activeMenu].closeMe(), "function" == typeof this.options.generateValueFunction && "function" == typeof this.options.generateContentsFunction) {
                var t = $(this.id + "-contents");
                if (t.innerHTML = this.options.generateContentsFunction(this), t.style.display = "block", $(this.id + "-value").style.display = "none", t.innerHTML.match("-currentcolor")) {
                    var i = this.id + "-currentcolor";
                    new Control.ColorPicker(this.id + "-currentcolor", {
                        IMAGE_BASE: "http://" + editorStatic + "/weebly/images/colorpicker/",
                        swatch: this.id + "-colorpicker",
                        onClose: function() {
                            setThemeColor(i)
                        }
                    })
                }
            }
            this.adjustLeft = 0, this.resizeWidth(this.options.openWidth), this.resizeHeight(this.options.openHeight), this.element.style.overflowY = this.options.overflowY, this.element.style.zIndex = this.options.zIndex - - 2, this.container.style.zIndex = this.options.zIndex - - 5, this.element.scrollTop = e, this.button.style.display = "none", this.isOpen=!0, Weebly.DropDowns.activeMenu = this.id, this.options.onOpen(this)
        }
    },
    close: function(e, t) {
        if (this.checkGone(), 0 == this.isOpen)
            return this.open(), void 0;
        if (Prototype.Browser.IE ? this.resizeHeight(this.options.height - - 2) : this.resizeHeight(this.options.height), this.element.style.overflowY = "hidden", this.options.rowFunction) {
            var i = e.id.replace(/^[0-9]+\-([0-9]+)$/, "$1");
            this.scrollMe(this.getScrollHeight(i)), this.lastEl = e
        } else if ("function" == typeof this.options.generateValueFunction && "function" == typeof this.options.generateContentsFunction) {
            $(this.id + "-contents").style.display = "none", $(this.id + "-contents").innerHTML = "", "undefined" != typeof e && (this.formEl.value = e, $(this.id + "-value").innerHTML = this.options.generateValueFunction(this)), $(this.id + "-value").style.display = "block";
            var n = $(this.id + "-contents");
            n.innerHTML.match("-currentcolor") && "none" != $("colorpicker").style.display && (Control.ColorPicker.activeColorPicker.isOpen=!1, $("colorpicker").style.display = "none")
        }
        this.element.style.zIndex = this.options.zIndex, this.container.style.zIndex = this.options.zIndex, this.button.style.display = "block", t || (this.options.rowFunction ? (this.formEl.value = e.firstChild.value, this.options.updateFunction(this.formEl.value)) : "function" == typeof this.options.generateValueFunction && "function" == typeof this.options.generateContentsFunction && (this.formEl.value = e, this.options.updateFunction(e)), this.options.onClose(this.options.noRefresh)), this.isOpen=!1, Weebly.DropDowns.activeMenu = null, this.adjustScroll(), this.resizeWidth(this.options.width)
    },
    changeOptions: function(e) {
        this.element.update(""), this.values = [];
        var t = 0;
        e.each(function(e) {
            var i = new Element("div", {
                id: this.id + "-" + t
            });
            i.onclick = function() {
                this.close(i)
            }.bind(this);
            var n = this.options.rowHoverColor;
            i.onmouseover = function() {
                this.style.background = n
            }, i.onmouseout = function() {
                this.style.background = "none"
            }, i.setStyle({
                height: this.options.height + "px",
                overflow: "hidden",
                cursor: "pointer",
                paddingLeft: "8px"
            }), i.update("<textarea style='display: none;' name='val'>" + e[0] + "</textarea><div style='line-height:" + this.options.height + "px;'>" + e[1] + "</div>"), this.element.appendChild(i), this.values[t] = e[0], 0 == t && (this.setValue(e[0]), this.formEl.value = e[0]), t++
        }, this), this.options.openHeight = t * this.options.height
    },
    setValue: function(e) {
        var t = 0;
        this.scrollMe(this.getScrollHeight(t)), this.lastEl = $(this.id + "-" + t), this.values.each(function(i) {
            i == e && (this.scrollMe(this.getScrollHeight(t)), this.lastEl = $(this.id + "-" + t)), t++
        }.bind(this))
    },
    getValue: function() {
        return this.formEl.value
    },
    closeMe: function() {
        this.close(this.lastEl, 1)
    },
    adjustScroll: function() {
        if (this.options.rowFunction && this.lastEl) {
            var e = this.lastEl.id.replace(/^[0-9]+\-([0-9]+)$/, "$1");
            this.scrollMe(this.getScrollHeight(e))
        }
    },
    getScrollHeight: function(e) {
        var t = this.element, i = t.select("> div:not(.font-group-heading):eq(" + e + ")")[0];
        return i.cumulativeOffset().top - t.cumulativeOffset().top
    },
    checkGone: function() {
        this.element && this.button || Weebly.DropDowns.destroy(this)
    },
    resizeWidth: function(e) {
        this.options.scaleBy;
        var t = this.element;
        Element.getStyle(t, "width").replace(/px/, ""), t.style.width = e + "px";
        var i = Position.cumulativeOffset(t);
        t.style.marginLeft = i[0] - - e > document.body.clientWidth && e > this.options.width ? "-" + (e - this.options.width) + "px" : "0px"
    },
    resizeHeight: function(e) {
        this.options.scaleBy;
        var t = this.element;
        Element.getStyle(t, "height").replace(/px/, ""), t.style.height = e + "px", "auto" == this.options.overflowY && (t.style.height = "")
    },
    scrollMe: function(e) {
        Prototype.Browser.IE;
        var t = this.element;
        $(t.id).scrollTop = e
    }
}, _wAMD.define("legacy/weebly_dropdowns", function(e) {
    return function() {
        var t;
        return t || e.Weebly.DropDowns
    }
}(this));
var domainChoiceSelected, domainPurchaseRefer = "", domainNextStep, domainNextBlog, isDomainChooserOpen=!1, isDomainChooserOnboarding=!1, isUserTyping = 0;
_wAMD.define("legacy/weebly_domain_utils", ["legacy/weebly_restrictions", "legacy/weebly_dialog"], function() {}), function() {
    var e = {
        HTMLEvents: /^(?:load|unload|abort|error|select|change|submit|reset|focus|blur|resize|scroll)$/,
        MouseEvents: /^(?:click|mouse(?:down|up|over|move|out))$/
    }, t = {
        pointerX: 0,
        pointerY: 0,
        button: 0,
        ctrlKey: !1,
        altKey: !1,
        shiftKey: !1,
        metaKey: !1,
        bubbles: !0,
        cancelable: !0
    };
    Event.simulate = function(i, n) {
        var o, s = Object.extend(t, arguments[2] || {}), r = null;
        i = $(i);
        for (var a in e)
            if (e[a].test(n)) {
                r = a;
                break
            }
        if (!r)
            throw new SyntaxError("Only HTMLEvents and MouseEvents interfaces are supported");
        return document.createEvent ? (o = document.createEvent(r), "HTMLEvents" == r ? o.initEvent(n, s.bubbles, s.cancelable) : o.initMouseEvent(n, s.bubbles, s.cancelable, document.defaultView, s.button, s.pointerX, s.pointerY, s.pointerX, s.pointerY, s.ctrlKey, s.altKey, s.shiftKey, s.metaKey, s.button, i), i.dispatchEvent(o)) : (s.clientX = s.pointerX, s.clientY = s.pointerY, o = Object.extend(document.createEventObject(), s), i.fireEvent("on" + n, o)), i
    }, Element.addMethods({
        simulate: Event.simulate
    })
}(), _wAMD.define("legacy/event.simulate", function() {}), _wAMD.define("legacy/weebly_feedback", function() {}), Weebly = window.Weebly || {}, Weebly.Storage = {
    Version: "0.7",
    Author: "David Rusenko",
    Company: "Weebly, Inc.",
    modules: ["flash", "localStorage", "cookie"],
    onReady: function() {},
    onReadyCalled: !1,
    init: function() {
        var e = jQuery;
        e.each(this.modules, function(e, t) {
            Weebly.Storage[t].init()
        }), setTimeout("if (!Weebly.Storage.onReadyCalled) { Weebly.Storage.onReadyCalled = true; Weebly.Storage.onReady() }", 2e3)
    },
    get: function(e) {
        for (var t = 0; t < this.modules.length; t++) {
            if (r = Weebly.Storage[this.modules[t]].get(e), "undefined" != typeof r && r!==!1)
                return r;
            r===!1 && (this.modules.splice(t, 1), t--)
        }
    },
    set: function(e, t) {
        var i = jQuery;
        i.each(this.modules, function(i, n) {
            Weebly.Storage[n].set(e, t)
        })
    }
}, Weebly.Storage.cookie = {
    init: function() {},
    get: function(e) {
        var t = document.cookie.match(e + "s?=s?([^;]+);");
        return t = t ? t : document.cookie.match(e + "s?=s?([^;]+)$"), t && t[1] ? t[1] : void 0
    },
    set: function(e, t) {
        document.cookie = e + "=" + t + "; expires=Fri, 2 Oct 2020 23:59:59 UTC; path=/"
    }
}, Weebly.Storage.localStorage = {
    init: function() {},
    get: function(e) {
        try {
            var t = localStorage[e];
            t && "undefined" != typeof t.value && (t = t.value)
        } catch (i) {
            return !1
        }
        return t
    },
    set: function(e, t) {
        try {
            localStorage[e] = t
        } catch (i) {}
    }
}, Weebly.Storage.flash = {
    init: function() {},
    onready: function() {
        this.cookie_id = "flashconnector", this.flash_cookie_ready=!1, this.flash_cookie_able=!1, this.flash_cookie = null, this.flash_alert=!1, document.getElementById && document.getElementById(this.cookie_id) && this.get_movie(this.cookie_id) && (this.flash_cookie_ready=!0, this.flash_cookie_able = this.flash_cookie.f_cookie_able(), Weebly.Storage.onReadyCalled || (Weebly.Storage.onReadyCalled=!0, Weebly.Storage.onReady()))
    },
    is_able: function() {
        return this.flash_alert || this.flash_cookie_ready && this.flash_cookie_able || (this.flash_alert=!0), this.flash_cookie_ready && this.flash_cookie_able
    },
    del: function(e) {
        this.is_able() && this.flash_cookie.f_delete_cookie(e)
    },
    get: function(e) {
        if (!this.is_able())
            return !1;
        try {
            var t = this.flash_cookie.f_get_cookie(e);
            t = "null" == t ? "" : t, t = "undefined" == typeof t?!1 : t
        } catch (i) {
            return !1
        }
        return t
    },
    set: function(e, t) {
        this.is_able() && this.flash_cookie.f_set_cookie(e, t)
    },
    get_movie: function() {
        return this.flash_cookie =- 1 != navigator.appName.indexOf("Microsoft") ? window[this.cookie_id] : document[this.cookie_id], this.flash_cookie?!0 : !1
    }
}, Weebly.Storage.init(), _wAMD.define("legacy/weebly_storage", function(e) {
    return function() {
        var t;
        return t || e.Weebly.Storage
    }
}(this)), document.observe("dom:loaded", function() {
    $("adsense_terms-cancel") && $("adsense_terms-cancel").observe("click", close_adsense_dialog), $("adsense_terms-close") && $("adsense_terms-close").observe("click", close_adsense_confirmation_dialog), "1" == getParameterByName("ac") && open_adsense_confirmation_dialog()
}), _wAMD.define("legacy/weebly_adsense", ["legacy/weebly_dialog"], function() {}), function(e) {
    "object" == typeof exports ? module.exports = e(module.exports, require("underscore"), require("backbone")) : "function" == typeof define && define.amd ? (_wAMD.define("window", [], function() {
        return window
    }), _wAMD.define("backgrid", ["window", "underscore", "backbone"], e)) : e(this, this._, this.Backbone)
}(function(e, t, i) {
    function n(e, t, i) {
        var n = t - (e + "").length;
        n = 0 > n ? 0 : n;
        for (var o = "", s = 0; n > s; s++)
            o += i;
        return o + e
    }
    var o = "	\n\f\r   ᠎             　\u2028\u2029﻿";
    if (!String.prototype.trim || o.trim()) {
        o = "[" + o + "]";
        var s = new RegExp("^" + o + o + "*"), r = new RegExp(o + o + "*$");
        String.prototype.trim = function() {
            if (void 0 === this || null === this)
                throw new TypeError("can't convert " + this + " to object");
            return String(this).replace(s, "").replace(r, "")
        }
    }
    var a = i.$, l = e.Backgrid = {
        Extension: {},
        resolveNameToClass: function(e, i) {
            if (t.isString(e)) {
                var n = t.map(e.split("-"), function(e) {
                    return e.slice(0, 1).toUpperCase() + e.slice(1)
                }).join("") + i, o = l[n] || l.Extension[n];
                if (t.isUndefined(o))
                    throw new ReferenceError("Class '" + n + "' not found");
                return o
            }
            return e
        },
        callByNeed: function() {
            var e = arguments[0];
            if (!t.isFunction(e))
                return e;
            var i = arguments[1], n = [].slice.call(arguments, 2);
            return e.apply(i, n + "" ? n : [])
        }
    };
    t.extend(l, i.Events);
    var c = l.Command = function(e) {
        t.extend(this, {
            altKey: !!e.altKey,
            "char": e["char"],
            charCode: e.charCode,
            ctrlKey: !!e.ctrlKey,
            key: e.key,
            keyCode: e.keyCode,
            locale: e.locale,
            location: e.location,
            metaKey: !!e.metaKey,
            repeat: !!e.repeat,
            shiftKey: !!e.shiftKey,
            which: e.which
        })
    };
    t.extend(c.prototype, {
        moveUp: function() {
            return 38 == this.keyCode
        },
        moveDown: function() {
            return 40 === this.keyCode
        },
        moveLeft: function() {
            return this.shiftKey && 9 === this.keyCode
        },
        moveRight: function() {
            return !this.shiftKey && 9 === this.keyCode
        },
        save: function() {
            return 13 === this.keyCode
        },
        cancel: function() {
            return 27 === this.keyCode
        },
        passThru: function() {
            return !(this.moveUp() || this.moveDown() || this.moveLeft() || this.moveRight() || this.save() || this.cancel())
        }
    });
    var u = l.CellFormatter = function() {};
    t.extend(u.prototype, {
        fromRaw: function(e) {
            return e
        },
        toRaw: function(e) {
            return e
        }
    });
    var d = l.NumberFormatter = function(e) {
        if (t.extend(this, this.defaults, e || {}), this.decimals < 0 || this.decimals > 20)
            throw new RangeError("decimals must be between 0 and 20")
    };
    d.prototype = new u, t.extend(d.prototype, {
        defaults: {
            decimals: 2,
            decimalSeparator: ".",
            orderSeparator: ","
        },
        HUMANIZED_NUM_RE: /(\d)(?=(?:\d{3})+$)/g,
        fromRaw: function(e) {
            if (t.isNull(e) || t.isUndefined(e))
                return "";
            e = e.toFixed(~~this.decimals);
            var i = e.split("."), n = i[0], o = i[1] ? (this.decimalSeparator || ".") + i[1]: "";
            return n.replace(this.HUMANIZED_NUM_RE, "$1" + this.orderSeparator) + o
        },
        toRaw: function(e) {
            if (e = e.trim(), "" === e)
                return null;
            for (var i = "", n = e.split(this.orderSeparator), o = 0; o < n.length; o++)
                i += n[o];
            var s = i.split(this.decimalSeparator);
            i = "";
            for (var o = 0; o < s.length; o++)
                i = i + s[o] + ".";
            "." === i[i.length - 1] && (i = i.slice(0, i.length - 1));
            var r = 1 * (1 * i).toFixed(~~this.decimals);
            return t.isNumber(r)&&!t.isNaN(r) ? r : void 0
        }
    });
    var h = l.PercentFormatter = function() {
        l.NumberFormatter.apply(this, arguments)
    };
    h.prototype = new l.NumberFormatter, t.extend(h.prototype, {
        defaults: t.extend({}, d.prototype.defaults, {
            multiplier: 1,
            symbol: "%"
        }),
        fromRaw: function(e) {
            var t = [].slice.call(arguments, 1);
            return t.unshift(e * this.multiplier), (d.prototype.fromRaw.apply(this, t) || "0") + this.symbol
        },
        toRaw: function(e) {
            var i = e.split(this.symbol);
            if (i && i[0] && "" === i[1] || null == i[1]) {
                var n = d.prototype.toRaw.call(this, i[0]);
                return t.isUndefined(n) ? n : n / this.multiplier
            }
        }
    });
    var p = l.DatetimeFormatter = function(e) {
        if (t.extend(this, this.defaults, e || {}), !this.includeDate&&!this.includeTime)
            throw new Error("Either includeDate or includeTime must be true")
    };
    p.prototype = new u, t.extend(p.prototype, {
        defaults: {
            includeDate: !0,
            includeTime: !0,
            includeMilli: !1
        },
        DATE_RE: /^([+\-]?\d{4})-(\d{2})-(\d{2})$/,
        TIME_RE: /^(\d{2}):(\d{2}):(\d{2})(\.(\d{3}))?$/,
        ISO_SPLITTER_RE: /T|Z| +/,
        _convert: function(e, i) {
            if ("" === (e + "").trim())
                return null;
            var o, s = null;
            if (t.isNumber(e)) {
                var r = new Date(e);
                o = n(r.getUTCFullYear(), 4, 0) + "-" + n(r.getUTCMonth() + 1, 2, 0) + "-" + n(r.getUTCDate(), 2, 0), s = n(r.getUTCHours(), 2, 0) + ":" + n(r.getUTCMinutes(), 2, 0) + ":" + n(r.getUTCSeconds(), 2, 0)
            } else {
                e = e.trim();
                var a = e.split(this.ISO_SPLITTER_RE) || [];
                o = this.DATE_RE.test(a[0]) ? a[0] : "", s = o && a[1] ? a[1] : this.TIME_RE.test(a[0]) ? a[0] : ""
            }
            var l = this.DATE_RE.exec(o) || [], c = this.TIME_RE.exec(s) || [];
            if (i) {
                if (this.includeDate && t.isUndefined(l[0]))
                    return;
                if (this.includeTime && t.isUndefined(c[0]))
                    return;
                if (!this.includeDate && o)
                    return;
                if (!this.includeTime && s)
                    return 
            }
            var r = new Date(Date.UTC(1 * l[1] || 0, 1 * l[2] - 1 || 0, 1 * l[3] || 0, 1 * c[1] || null, 1 * c[2] || null, 1 * c[3] || null, 1 * c[5] || null)), u = "";
            return this.includeDate && (u = n(r.getUTCFullYear(), 4, 0) + "-" + n(r.getUTCMonth() + 1, 2, 0) + "-" + n(r.getUTCDate(), 2, 0)), this.includeTime && (u = u + (this.includeDate ? "T" : "") + n(r.getUTCHours(), 2, 0) + ":" + n(r.getUTCMinutes(), 2, 0) + ":" + n(r.getUTCSeconds(), 2, 0), this.includeMilli && (u = u + "." + n(r.getUTCMilliseconds(), 3, 0))), this.includeDate && this.includeTime && (u += "Z"), u
        },
        fromRaw: function(e) {
            return t.isNull(e) || t.isUndefined(e) ? "" : this._convert(e)
        },
        toRaw: function(e) {
            return this._convert(e, !0)
        }
    });
    var f = l.StringFormatter = function() {};
    f.prototype = new u, t.extend(f.prototype, {
        fromRaw: function(e) {
            return t.isUndefined(e) || t.isNull(e) ? "" : e + ""
        }
    });
    var m = l.EmailFormatter = function() {};
    m.prototype = new u, t.extend(m.prototype, {
        toRaw: function(e) {
            var i = e.trim().split("@");
            return 2 === i.length && t.all(i) ? e : void 0
        }
    });
    var g = l.SelectFormatter = function() {};
    g.prototype = new u, t.extend(g.prototype, {
        fromRaw: function(e) {
            return t.isArray(e) ? e : null != e ? [e] : []
        }
    });
    var v = l.CellEditor = i.View.extend({
        initialize: function(e) {
            this.formatter = e.formatter, this.column = e.column, this.column instanceof D || (this.column = new D(this.column)), this.listenTo(this.model, "backgrid:editing", this.postRender)
        },
        postRender: function(e, t) {
            return (null == t || t.get("name") == this.column.get("name")) && this.$el.select(), this
        }
    }), y = l.InputCellEditor = v.extend({
        tagName: "input",
        attributes: {
            type: "text"
        },
        events: {
            blur: "saveOrCancel",
            keydown: "saveOrCancel"
        },
        initialize: function(e) {
            y.__super__.initialize.apply(this, arguments), e.placeholder && this.$el.attr("placeholder", e.placeholder)
        },
        render: function() {
            var e = this.model;
            return this.$el.val(this.formatter.fromRaw(e.get(this.column.get("name")), e)), this
        },
        saveOrCancel: function(e) {
            var i = this.formatter, n = this.model, o = this.column, s = new c(e), r = "blur" === e.type;
            if (s.moveUp() || s.moveDown() || s.moveLeft() || s.moveRight() || s.save() || r) {
                e.preventDefault(), e.stopPropagation();
                var a = this.$el.val(), l = i.toRaw(a, n);
                t.isUndefined(l) ? n.trigger("backgrid:error", n, o, a) : (n.set(o.get("name"), l), n.trigger("backgrid:edited", n, o, s))
            } else 
                s.cancel() && (e.stopPropagation(), n.trigger("backgrid:edited", n, o, s))
        },
        postRender: function(e, t) {
            if (null == t || t.get("name") == this.column.get("name"))
                if ("right" === this.$el.css("text-align")) {
                    var i = this.$el.val();
                    this.$el.select().val(null).val(i)
                } else 
                    this.$el.select();
            return this
        }
    }), b = l.Cell = i.View.extend({
        tagName: "td",
        formatter: u,
        editor: y,
        events: {
            click: "enterEditMode"
        },
        initialize: function(e) {
            this.column = e.column, this.column instanceof D || (this.column = new D(this.column));
            var i = this.column, n = this.model, o = this.$el, s = l.resolveNameToClass(i.get("formatter") || this.formatter, "Formatter");
            t.isFunction(s.fromRaw) || t.isFunction(s.toRaw) || (s = new s), this.formatter = s, this.editor = l.resolveNameToClass(this.editor, "CellEditor"), this.listenTo(n, "change:" + i.get("name"), function() {
                o.hasClass("editor") || this.render()
            }), this.listenTo(n, "backgrid:error", this.renderError), this.listenTo(i, "change:editable change:sortable change:renderable", function(e) {
                var t = e.changedAttributes();
                for (var i in t)
                    t.hasOwnProperty(i) && o.toggleClass(i, t[i])
            }), l.callByNeed(i.editable(), i, n) && o.addClass("editable"), l.callByNeed(i.sortable(), i, n) && o.addClass("sortable"), l.callByNeed(i.renderable(), i, n) && o.addClass("renderable")
        },
        render: function() {
            this.$el.empty();
            var e = this.model;
            return this.$el.text(this.formatter.fromRaw(e.get(this.column.get("name")), e)), this.delegateEvents(), this
        },
        enterEditMode: function() {
            var e = this.model, t = this.column, i = l.callByNeed(t.editable(), t, e);
            i && (this.currentEditor = new this.editor({
                column: this.column,
                model: this.model,
                formatter: this.formatter
            }), e.trigger("backgrid:edit", e, t, this, this.currentEditor), this.undelegateEvents(), this.$el.empty(), this.$el.append(this.currentEditor.$el), this.currentEditor.render(), this.$el.addClass("editor"), e.trigger("backgrid:editing", e, t, this, this.currentEditor))
        },
        renderError: function(e, t) {
            (null == t || t.get("name") == this.column.get("name")) && this.$el.addClass("error")
        },
        exitEditMode: function() {
            this.$el.removeClass("error"), this.currentEditor.remove(), this.stopListening(this.currentEditor), delete this.currentEditor, this.$el.removeClass("editor"), this.render()
        },
        remove: function() {
            return this.currentEditor && (this.currentEditor.remove.apply(this.currentEditor, arguments), delete this.currentEditor), b.__super__.remove.apply(this, arguments)
        }
    }), w = l.StringCell = b.extend({
        className: "string-cell",
        formatter: f
    }), _ = l.UriCell = b.extend({
        className: "uri-cell",
        title: null,
        target: "_blank",
        initialize: function(e) {
            _.__super__.initialize.apply(this, arguments), this.title = e.title || this.title, this.target = e.target || this.target
        },
        render: function() {
            this.$el.empty();
            var e = this.model.get(this.column.get("name")), t = this.formatter.fromRaw(e, this.model);
            return this.$el.append(a("<a>", {
                tabIndex: - 1,
                href: e,
                title: this.title || t,
                target: this.target
            }).text(t)), this.delegateEvents(), this
        }
    });
    l.EmailCell = w.extend({
        className: "email-cell",
        formatter: m,
        render: function() {
            this.$el.empty();
            var e = this.model, t = this.formatter.fromRaw(e.get(this.column.get("name")), e);
            return this.$el.append(a("<a>", {
                tabIndex: - 1,
                href: "mailto:" + t,
                title: t
            }).text(t)), this.delegateEvents(), this
        }
    });
    var C = l.NumberCell = b.extend({
        className: "number-cell",
        decimals: d.prototype.defaults.decimals,
        decimalSeparator: d.prototype.defaults.decimalSeparator,
        orderSeparator: d.prototype.defaults.orderSeparator,
        formatter: d,
        initialize: function() {
            C.__super__.initialize.apply(this, arguments);
            var e = this.formatter;
            e.decimals = this.decimals, e.decimalSeparator = this.decimalSeparator, e.orderSeparator = this.orderSeparator
        }
    });
    l.IntegerCell = C.extend({
        className: "integer-cell",
        decimals: 0
    });
    var k = l.PercentCell = C.extend({
        className: "percent-cell",
        multiplier: h.prototype.defaults.multiplier,
        symbol: h.prototype.defaults.symbol,
        formatter: h,
        initialize: function() {
            k.__super__.initialize.apply(this, arguments);
            var e = this.formatter;
            e.multiplier = this.multiplier, e.symbol = this.symbol
        }
    }), x = l.DatetimeCell = b.extend({
        className: "datetime-cell",
        includeDate: p.prototype.defaults.includeDate,
        includeTime: p.prototype.defaults.includeTime,
        includeMilli: p.prototype.defaults.includeMilli,
        formatter: p,
        initialize: function() {
            x.__super__.initialize.apply(this, arguments);
            var e = this.formatter;
            e.includeDate = this.includeDate, e.includeTime = this.includeTime, e.includeMilli = this.includeMilli;
            var i = this.includeDate ? "YYYY-MM-DD": "";
            i += this.includeDate && this.includeTime ? "T" : "", i += this.includeTime ? "HH:mm:ss" : "", i += this.includeTime && this.includeMilli ? ".SSS" : "", this.editor = this.editor.extend({
                attributes: t.extend({}, this.editor.prototype.attributes, this.editor.attributes, {
                    placeholder: i
                })
            })
        }
    });
    l.DateCell = x.extend({
        className: "date-cell",
        includeTime: !1
    }), l.TimeCell = x.extend({
        className: "time-cell",
        includeDate: !1
    });
    var S = l.BooleanCellEditor = v.extend({
        tagName: "input",
        attributes: {
            tabIndex: - 1,
            type: "checkbox"
        },
        events: {
            mousedown: function() {
                this.mouseDown=!0
            },
            blur: "enterOrExitEditMode",
            mouseup: function() {
                this.mouseDown=!1
            },
            change: "saveOrCancel",
            keydown: "saveOrCancel"
        },
        render: function() {
            var e = this.model, t = this.formatter.fromRaw(e.get(this.column.get("name")), e);
            return this.$el.prop("checked", t), this
        },
        enterOrExitEditMode: function(e) {
            if (!this.mouseDown) {
                var t = this.model;
                t.trigger("backgrid:edited", t, this.column, new c(e))
            }
        },
        saveOrCancel: function(e) {
            var t = this.model, i = this.column, n = this.formatter, o = new c(e);
            if (o.passThru() && "change" != e.type)
                return !0;
            o.cancel() && (e.stopPropagation(), t.trigger("backgrid:edited", t, i, o));
            var s = this.$el;
            if (o.save() || o.moveLeft() || o.moveRight() || o.moveUp() || o.moveDown()) {
                e.preventDefault(), e.stopPropagation();
                var r = n.toRaw(s.prop("checked"), t);
                t.set(i.get("name"), r), t.trigger("backgrid:edited", t, i, o)
            } else if ("change" == e.type) {
                var r = n.toRaw(s.prop("checked"), t);
                t.set(i.get("name"), r), s.focus()
            }
        }
    });
    l.BooleanCell = b.extend({
        className: "boolean-cell",
        editor: S,
        events: {
            click: "enterEditMode"
        },
        render: function() {
            this.$el.empty();
            var e = this.model, t = this.column, i = l.callByNeed(t.editable(), t, e);
            return this.$el.append(a("<input>", {
                tabIndex: - 1,
                type: "checkbox",
                checked: this.formatter.fromRaw(e.get(t.get("name")), e),
                disabled: !i
            })), this.delegateEvents(), this
        }
    });
    var M = l.SelectCellEditor = v.extend({
        tagName: "select",
        events: {
            change: "save",
            blur: "close",
            keydown: "close"
        },
        template: t.template('<option value="<%- value %>" <%= selected ? \'selected="selected"\' : "" %>><%- text %></option>', null, {
            variable: null
        }),
        setOptionValues: function(e) {
            this.optionValues = e, this.optionValues = t.result(this, "optionValues")
        },
        setMultiple: function(e) {
            this.multiple = e, this.$el.prop("multiple", e)
        },
        _renderOptions: function(e, i) {
            for (var n = "", o = 0; o < e.length; o++)
                n += this.template({
                    text: e[o][0],
                    value: e[o][1],
                    selected: t.indexOf(i, e[o][1])>-1
                });
            return n
        },
        render: function() {
            this.$el.empty();
            var e = t.result(this, "optionValues"), i = this.model, n = this.formatter.fromRaw(i.get(this.column.get("name")), i);
            if (!t.isArray(e))
                throw new TypeError("optionValues must be an array");
            for (var o = null, s = null, o = null, r = null, l = null, c = 0; c < e.length; c++) {
                var o = e[c];
                if (t.isArray(o))
                    s = o[0], o = o[1], this.$el.append(this.template({
                        text: s,
                        value: o,
                        selected: t.indexOf(n, o)>-1
                    }));
                else {
                    if (!t.isObject(o))
                        throw new TypeError("optionValues elements must be a name-value pair or an object hash of { name: 'optgroup label', value: [option name-value pairs] }");
                    r = o.name, l = a("<optgroup></optgroup>", {
                        label: r
                    }), l.append(this._renderOptions.call(this, o.values, n)), this.$el.append(l)
                }
            }
            return this.delegateEvents(), this
        },
        save: function() {
            var e = this.model, t = this.column;
            e.set(t.get("name"), this.formatter.toRaw(this.$el.val(), e))
        },
        close: function(e) {
            var t = this.model, i = this.column, n = new c(e);
            n.cancel() ? (e.stopPropagation(), t.trigger("backgrid:edited", t, i, new c(e))) : (n.save() || n.moveLeft() || n.moveRight() || n.moveUp() || n.moveDown() || "blur" == e.type) && (e.preventDefault(), e.stopPropagation(), this.save(e), t.trigger("backgrid:edited", t, i, new c(e)))
        }
    }), T = l.SelectCell = b.extend({
        className: "select-cell",
        editor: M,
        multiple: !1,
        formatter: g,
        optionValues: void 0,
        delimiter: ", ",
        initialize: function() {
            T.__super__.initialize.apply(this,
            arguments),
            this.listenTo(this.model,
            "backgrid:edit",
            function(e,
            t,
            i,
            n) {
                t.get("name") == this.column.get("name") && (n.setOptionValues(this.optionValues),
                n.setMultiple(this.multiple))
            })
        }, render : function() {
            this.$el.empty();
            var e = t.result(this, "optionValues"), i = this.model, n = this.formatter.fromRaw(i.get(this.column.get("name")), i), o = [];
            try {
                if (!t.isArray(e) || t.isEmpty(e))
                    throw new TypeError;
                for (var s = 0; s < n.length; s++)
                    for (var r = n[s], a = 0; a < e.length; a++) {
                        var l = e[a];
                        if (t.isArray(l)) {
                            var c = l[0], l = l[1];
                            l == r && o.push(c)
                        } else {
                            if (!t.isObject(l))
                                throw new TypeError;
                                for (var u = l.values, d = 0; d < u.length; d++) {
                                    var h = u[d];
                                    h[1] == r && o.push(h[0])
                                }
                            }
                    }
                this.$el.append(o.join(this.delimiter))
            } catch (p) {
                if (p instanceof TypeError)
                    throw new TypeError("'optionValues' must be of type {Array.<Array>|Array.<{name: string, values: Array.<Array>}>}");
                throw p
            }
            return this.delegateEvents(), this
        }
    }), D = l.Column = i.Model.extend({
        defaults: {
            name: void 0,
            label: void 0,
            sortable: !0,
            editable: !0,
            renderable: !0,
            formatter: void 0,
            sortType: "cycle",
            sortValue: void 0,
            direction: null,
            cell: void 0,
            headerCell: void 0,
            headerClass: void 0
        },
        initialize: function() {
            this.has("label") || this.set({
                label: this.get("name")
            }, {
                silent: !0
            });
            var e = l.resolveNameToClass(this.get("headerCell"), "HeaderCell"), t = l.resolveNameToClass(this.get("cell"), "Cell");
            this.set({
                cell: t,
                headerCell: e
            }, {
                silent: !0
            })
        },
        sortValue: function() {
            var e = this.get("sortValue");
            return t.isString(e) ? this[e] : t.isFunction(e) ? e : function(e, t) {
                return e.get(t)
            }
        }
    });
    t.each(["sortable", "renderable", "editable"], function(e) {
        D.prototype[e] = function() {
            var i = this.get(e);
            return t.isString(i) ? this[i] : t.isFunction(i) ? i : !!i
        }
    });
    var E = l.Columns = i.Collection.extend({
        model: D
    }), A = l.Row = i.View.extend({
        tagName: "tr",
        initialize: function(e) {
            var t = this.columns = e.columns;
            t instanceof i.Collection || (t = this.columns = new E(t));
            for (var n = this.cells = [], o = 0; o < t.length; o++)
                n.push(this.makeCell(t.at(o), e));
            this.listenTo(t, "add", function(t, i) {
                var o = i.indexOf(t), s = this.makeCell(t, e);
                n.splice(o, 0, s);
                var r = this.$el;
                0 === o ? r.prepend(s.render().$el) : o === i.length - 1 ? r.append(s.render().$el) : r.children().eq(o).before(s.render().$el)
            }), this.listenTo(t, "remove", function(e, t, i) {
                n[i.index].remove(), n.splice(i.index, 1)
            })
        },
        makeCell: function(e) {
            return new (e.get("cell"))({
                column: e,
                model: this.model
            })
        },
        render: function() {
            this.$el.empty();
            for (var e = document.createDocumentFragment(), t = 0; t < this.cells.length; t++)
                e.appendChild(this.cells[t].render().el);
            return this.el.appendChild(e), this.delegateEvents(), this
        },
        remove: function() {
            for (var e = 0; e < this.cells.length; e++) {
                var t = this.cells[e];
                t.remove.apply(t, arguments)
            }
            return i.View.prototype.remove.apply(this, arguments)
        }
    }), P = l.EmptyRow = i.View.extend({
        tagName: "tr",
        emptyText: null,
        initialize: function(e) {
            this.emptyText = e.emptyText, this.columns = e.columns
        },
        render: function() {
            this.$el.empty();
            var e = document.createElement("td");
            return e.setAttribute("colspan", this.columns.length), e.appendChild(document.createTextNode(t.result(this, "emptyText"))), this.el.className = "empty", this.el.appendChild(e), this
        }
    }), R = l.HeaderCell = i.View.extend({
        tagName: "th",
        events: {
            "click a": "onClick"
        },
        initialize: function(e) {
            this.column = e.column, this.column instanceof D || (this.column = new D(this.column));
            var t = this.column, i = this.collection, n = this.$el;
            this.listenTo(t, "change:editable change:sortable change:renderable", function(e) {
                var t = e.changedAttributes();
                for (var i in t)
                    t.hasOwnProperty(i) && n.toggleClass(i, t[i])
            }), this.listenTo(t, "change:direction", this.setCellDirection), this.listenTo(t, "change:name change:label", this.render), l.callByNeed(t.editable(), t, i) && n.addClass("editable"), l.callByNeed(t.sortable(), t, i) && n.addClass("sortable"), l.callByNeed(t.renderable(), t, i) && n.addClass("renderable"), this.listenTo(i.fullCollection || i, "sort", this.removeCellDirection)
        },
        removeCellDirection: function() {
            this.$el.removeClass("ascending").removeClass("descending"), this.column.set("direction", null)
        },
        setCellDirection: function(e, t) {
            this.$el.removeClass("ascending").removeClass("descending"), e.cid == this.column.cid && this.$el.addClass(t)
        },
        onClick: function(e) {
            function t(e, t) {
                "ascending" === n.get("direction") ? o.trigger(s, t, "descending") : "descending" === n.get("direction") ? o.trigger(s, t, null) : o.trigger(s, t, "ascending")
            }
            function i(e, t) {
                "ascending" === n.get("direction") ? o.trigger(s, t, "descending") : o.trigger(s, t, "ascending")
            }
            e.preventDefault();
            var n = this.column, o = this.collection, s = "backgrid:sort", r = l.callByNeed(n.sortable(), n, this.collection);
            if (r) {
                var a = n.get("sortType");
                "toggle" === a ? i(this, n) : t(this, n)
            }
        },
        render: function() {
            this.$el.empty();
            var e, t = this.column, i = l.callByNeed(t.sortable(), t, this.collection);
            return e = i ? a("<a>").text(t.get("label")).append("<b class='sort-caret'></b>") : document.createTextNode(t.get("label")), this.$el.append(e), this.$el.addClass(t.get("name")), this.$el.addClass(t.get("direction")), this.$el.addClass(t.get("headerClass")), this.delegateEvents(), this
        }
    });
    l.HeaderRow = l.Row.extend({
        requiredOptions: ["columns", "collection"],
        initialize: function() {
            l.Row.prototype.initialize.apply(this, arguments)
        },
        makeCell: function(e, t) {
            var i = e.get("headerCell") || t.headerCell || R;
            return i = new i({
                column: e,
                collection: this.collection
            })
        }
    });
    var F = l.Header = i.View.extend({
        tagName: "thead",
        initialize: function(e) {
            this.columns = e.columns, this.columns instanceof i.Collection || (this.columns = new E(this.columns)), this.row = new l.HeaderRow({
                columns: this.columns,
                collection: this.collection
            })
        },
        render: function() {
            return this.$el.append(this.row.render().$el), this.delegateEvents(), this
        },
        remove: function() {
            return this.row.remove.apply(this.row, arguments), i.View.prototype.remove.apply(this, arguments)
        }
    }), O = l.Body = i.View.extend({
        tagName: "tbody",
        initialize: function(e) {
            this.columns = e.columns, this.columns instanceof i.Collection || (this.columns = new E(this.columns)), this.row = e.row || A, this.rows = this.collection.map(function(e) {
                var t = new this.row({
                    columns: this.columns,
                    model: e
                });
                return t
            }, this), this.emptyText = e.emptyText, this._unshiftEmptyRowMayBe();
            var t = this.collection;
            this.listenTo(t, "add", this.insertRow), this.listenTo(t, "remove", this.removeRow), this.listenTo(t, "sort", this.refresh), this.listenTo(t, "reset", this.refresh), this.listenTo(t, "backgrid:sort", this.sort), this.listenTo(t, "backgrid:edited", this.moveToNextCell)
        },
        _unshiftEmptyRowMayBe: function() {
            0 === this.rows.length && null != this.emptyText && this.rows.unshift(new P({
                emptyText: this.emptyText,
                columns: this.columns
            }))
        },
        insertRow: function(e, t, n) {
            if (this.rows[0]instanceof P && this.rows.pop().remove(), !(t instanceof i.Collection || n))
                return this.collection.add(e, n = t), void 0;
            var o = new this.row({
                columns: this.columns,
                model: e
            }), s = t.indexOf(e);
            this.rows.splice(s, 0, o);
            var r = this.$el, a = r.children(), l = o.render().$el;
            return s >= a.length ? r.append(l) : a.eq(s).before(l), this
        },
        removeRow: function(e, i, n) {
            return n ? ((t.isUndefined(n.render) || n.render) && this.rows[n.index].remove(), this.rows.splice(n.index, 1), this._unshiftEmptyRowMayBe(), this) : (this.collection.remove(e, n = i), this._unshiftEmptyRowMayBe(), void 0)
        },
        refresh: function() {
            for (var e = 0; e < this.rows.length; e++)
                this.rows[e].remove();
            return this.rows = this.collection.map(function(e) {
                var t = new this.row({
                    columns: this.columns,
                    model: e
                });
                return t
            }, this), this._unshiftEmptyRowMayBe(), this.render(), this.collection.trigger("backgrid:refresh", this), this
        },
        render: function() {
            this.$el.empty();
            for (var e = document.createDocumentFragment(), t = 0; t < this.rows.length; t++) {
                var i = this.rows[t];
                e.appendChild(i.render().el)
            }
            return this.el.appendChild(e), this.delegateEvents(), this
        },
        remove: function() {
            for (var e = 0; e < this.rows.length; e++) {
                var t = this.rows[e];
                t.remove.apply(t, arguments)
            }
            return i.View.prototype.remove.apply(this, arguments)
        },
        sort: function(e, n) {
            if (!t.contains(["ascending", "descending", null], n))
                throw new RangeError('direction must be one of "ascending", "descending" or `null`');
            t.isString(e) && (e = this.columns.findWhere({
                name: e
            }));
            var o, s = this.collection;
            o = "ascending" === n?-1 : "descending" === n ? 1 : null;
            var r = this.makeComparator(e.get("name"), o, o ? e.sortValue() : function(e) {
                return 1 * e.cid.replace("c", "")
            });
            return i.PageableCollection && s instanceof i.PageableCollection ? (s.setSorting(o && e.get("name"), o, {
                sortValue: e.sortValue()
            }), s.fullCollection ? (null == s.fullCollection.comparator && (s.fullCollection.comparator = r), s.fullCollection.sort(), s.trigger("backgrid:sorted", e, n, s)) : s.fetch({
                reset: !0,
                success: function() {
                    s.trigger("sort"), e.set("direction", n), s.trigger("backgrid:sorted", e, n, s)
                }
            })) : (s.comparator = r, s.sort(), s.trigger("backgrid:sorted", e, n, s)), e.set("direction", n), this
        },
        makeComparator: function(e, t, i) {
            return function(n, o) {
                var s, r = i(n, e), a = i(o, e);
                return 1 === t && (s = r, r = a, a = s), r === a ? 0 : a > r?-1 : 1
            }
        },
        moveToNextCell: function(e, t, i) {
            var n, o, s, r, a, c = this.collection.indexOf(e), u = this.columns.indexOf(t);
            if (this.rows[c].cells[u].exitEditMode(), i.moveUp() || i.moveDown() || i.moveLeft() || i.moveRight() || i.save()) {
                var d = this.columns.length, h = d * this.collection.length;
                if (i.moveUp() || i.moveDown()) {
                    r = c + (i.moveUp()?-1 : 1);
                    var p = this.rows[r];
                    p ? (n = p.cells[u], l.callByNeed(n.column.editable(), n.column, e) && (n.enterEditMode(), e.trigger("backgrid:next", r, u, !1))) : e.trigger("backgrid:next", r, u, !0)
                } else if (i.moveLeft() || i.moveRight()) {
                    for (var f = i.moveRight(), m = c * d + u + (f ? 1 : - 1); m >= 0 && h > m; f ? m++ : m--)
                        if (r=~~(m / d), a = m - r * d, n = this.rows[r].cells[a], o = l.callByNeed(n.column.renderable(), n.column, n.model), s = l.callByNeed(n.column.editable(), n.column, e), o && s) {
                            n.enterEditMode(), e.trigger("backgrid:next", r, a, !1);
                            break
                        }
                    m == h && e.trigger("backgrid:next", ~~(m / d), m - r * d, !0)
                }
            }
            return this
        }
    });
    return l.Footer = i.View.extend({
        tagName: "tfoot",
        initialize: function(e) {
            this.columns = e.columns, this.columns instanceof i.Collection || (this.columns = new l.Columns(this.columns))
        }
    }), l.Grid = i.View.extend({
        tagName: "table",
        className: "backgrid",
        header: F,
        body: O,
        footer: null,
        initialize: function(e) {
            e.columns instanceof i.Collection || (e.columns = new E(e.columns)), this.columns = e.columns;
            var n = t.omit(e, ["el", "id", "attributes", "className", "tagName", "events"]);
            this.body = e.body || this.body, this.body = new this.body(n), this.header = e.header || this.header, this.header && (this.header = new this.header(n)), this.footer = e.footer || this.footer, this.footer && (this.footer = new this.footer(n)), this.listenTo(this.columns, "reset", function() {
                this.header && (this.header = new (this.header.remove().constructor)(n)), this.body = new (this.body.remove().constructor)(n), this.footer && (this.footer = new (this.footer.remove().constructor)(n)), this.render()
            })
        },
        insertRow: function() {
            return this.body.insertRow.apply(this.body, arguments), this
        },
        removeRow: function() {
            return this.body.removeRow.apply(this.body, arguments), this
        },
        insertColumn: function() {
            return this.columns.add.apply(this.columns, arguments), this
        },
        removeColumn: function() {
            return this.columns.remove.apply(this.columns, arguments), this
        },
        sort: function() {
            return this.body.sort.apply(this.body, arguments), this
        },
        render: function() {
            return this.$el.empty(), this.header && this.$el.append(this.header.render().$el), this.footer && this.$el.append(this.footer.render().$el), this.$el.append(this.body.render().$el), this.delegateEvents(), this.trigger("backgrid:rendered", this), this
        },
        remove: function() {
            return this.header && this.header.remove.apply(this.header, arguments), this.body.remove.apply(this.body, arguments), this.footer && this.footer.remove.apply(this.footer, arguments), i.View.prototype.remove.apply(this, arguments)
        }
    }), l
});
var lunr = function(e) {
    var t = new lunr.Index;
    return t.pipeline.add(lunr.stopWordFilter, lunr.stemmer), e && e.call(t, t), t
};
lunr.version = "0.4.4", "undefined" != typeof module && (module.exports = lunr), lunr.utils = {}, lunr.utils.warn = function(e) {
    return function(t) {
        e.console && console.warn && console.warn(t)
    }
}(this), lunr.utils.zeroFillArray = function() {
    var e = [0];
    return function(t) {
        for (; e.length < t;)
            e = e.concat(e);
        return e.slice(0, t)
    }
}(), lunr.EventEmitter = function() {
    this.events = {}
}, lunr.EventEmitter.prototype.addListener = function() {
    var e = Array.prototype.slice.call(arguments), t = e.pop(), i = e;
    if ("function" != typeof t)
        throw new TypeError("last argument must be a function");
    i.forEach(function(e) {
        this.hasHandler(e) || (this.events[e] = []), this.events[e].push(t)
    }, this)
}, lunr.EventEmitter.prototype.removeListener = function(e, t) {
    if (this.hasHandler(e)) {
        var i = this.events[e].indexOf(t);
        this.events[e].splice(i, 1), this.events[e].length || delete this.events[e]
    }
}, lunr.EventEmitter.prototype.emit = function(e) {
    if (this.hasHandler(e)) {
        var t = Array.prototype.slice.call(arguments, 1);
        this.events[e].forEach(function(e) {
            e.apply(void 0, t)
        })
    }
}, lunr.EventEmitter.prototype.hasHandler = function(e) {
    return e in this.events
}, lunr.tokenizer = function(e) {
    if (!arguments.length || null == e || void 0 == e)
        return [];
    if (Array.isArray(e))
        return e.map(function(e) {
            return e.toLowerCase()
        });
    for (var t = e.toString().replace(/^\s+/, ""), i = t.length - 1; i >= 0; i--)
        if (/\S/.test(t.charAt(i))) {
            t = t.substring(0, i + 1);
            break
        }
    return t.split(/\s+/).map(function(e) {
        return e.replace(/^\W+/, "").replace(/\W+$/, "").toLowerCase()
    })
}, lunr.Pipeline = function() {
    this._stack = []
}, lunr.Pipeline.registeredFunctions = {}, lunr.Pipeline.registerFunction = function(e, t) {
    t in this.registeredFunctions && lunr.utils.warn("Overwriting existing registered function: " + t), e.label = t, lunr.Pipeline.registeredFunctions[e.label] = e
}, lunr.Pipeline.warnIfFunctionNotRegistered = function(e) {
    var t = e.label && e.label in this.registeredFunctions;
    t || lunr.utils.warn("Function is not registered with pipeline. This may cause problems when serialising the index.\n", e)
}, lunr.Pipeline.load = function(e) {
    var t = new lunr.Pipeline;
    return e.forEach(function(e) {
        var i = lunr.Pipeline.registeredFunctions[e];
        if (!i)
            throw new Error("Cannot load un-registered function: " + e);
        t.add(i)
    }), t
}, lunr.Pipeline.prototype.add = function() {
    var e = Array.prototype.slice.call(arguments);
    e.forEach(function(e) {
        lunr.Pipeline.warnIfFunctionNotRegistered(e), this._stack.push(e)
    }, this)
}, lunr.Pipeline.prototype.after = function(e, t) {
    lunr.Pipeline.warnIfFunctionNotRegistered(t);
    var i = this._stack.indexOf(e) + 1;
    this._stack.splice(i, 0, t)
}, lunr.Pipeline.prototype.before = function(e, t) {
    lunr.Pipeline.warnIfFunctionNotRegistered(t);
    var i = this._stack.indexOf(e);
    this._stack.splice(i, 0, t)
}, lunr.Pipeline.prototype.remove = function(e) {
    var t = this._stack.indexOf(e);
    this._stack.splice(t, 1)
}, lunr.Pipeline.prototype.run = function(e) {
    for (var t = [], i = e.length, n = this._stack.length, o = 0; i > o; o++) {
        for (var s = e[o], r = 0; n > r && (s = this._stack[r](s, o, e), void 0 !== s); r++);
        void 0 !== s && t.push(s)
    }
    return t
}, lunr.Pipeline.prototype.toJSON = function() {
    return this._stack.map(function(e) {
        return lunr.Pipeline.warnIfFunctionNotRegistered(e), e.label
    })
}, lunr.Vector = function(e) {
    this.elements = e
}, lunr.Vector.prototype.magnitude = function() {
    if (this._magnitude)
        return this._magnitude;
    for (var e, t = 0, i = this.elements, n = i.length, o = 0; n > o; o++)
        e = i[o], t += e * e;
    return this._magnitude = Math.sqrt(t)
}, lunr.Vector.prototype.dot = function(e) {
    for (var t = this.elements, i = e.elements, n = t.length, o = 0, s = 0; n > s; s++)
        o += t[s] * i[s];
    return o
}, lunr.Vector.prototype.similarity = function(e) {
    return this.dot(e) / (this.magnitude() * e.magnitude())
}, lunr.Vector.prototype.toArray = function() {
    return this.elements
}, lunr.SortedSet = function() {
    this.length = 0, this.elements = []
}, lunr.SortedSet.load = function(e) {
    var t = new this;
    return t.elements = e, t.length = e.length, t
}, lunr.SortedSet.prototype.add = function() {
    Array.prototype.slice.call(arguments).forEach(function(e) {
        ~this.indexOf(e) || this.elements.splice(this.locationFor(e), 0, e)
    }, this), this.length = this.elements.length
}, lunr.SortedSet.prototype.toArray = function() {
    return this.elements.slice()
}, lunr.SortedSet.prototype.map = function(e, t) {
    return this.elements.map(e, t)
}, lunr.SortedSet.prototype.forEach = function(e, t) {
    return this.elements.forEach(e, t)
}, lunr.SortedSet.prototype.indexOf = function(e, t, i) {
    var t = t || 0, i = i || this.elements.length, n = i - t, o = t + Math.floor(n / 2), s = this.elements[o];
    return 1 >= n ? s === e ? o : - 1 : e > s ? this.indexOf(e, o, i) : s > e ? this.indexOf(e, t, o) : s === e ? o : void 0
}, lunr.SortedSet.prototype.locationFor = function(e, t, i) {
    var t = t || 0, i = i || this.elements.length, n = i - t, o = t + Math.floor(n / 2), s = this.elements[o];
    if (1 >= n) {
        if (s > e)
            return o;
        if (e > s)
            return o + 1
    }
    return e > s ? this.locationFor(e, o, i) : s > e ? this.locationFor(e, t, o) : void 0
}, lunr.SortedSet.prototype.intersect = function(e) {
    for (var t = new lunr.SortedSet, i = 0, n = 0, o = this.length, s = e.length, r = this.elements, a = e.elements; ;) {
        if (i > o - 1 || n > s - 1)
            break;
        r[i] !== a[n] ? r[i] < a[n] ? i++ : r[i] > a[n] && n++ : (t.add(r[i]), i++, n++)
    }
    return t
}, lunr.SortedSet.prototype.clone = function() {
    var e = new lunr.SortedSet;
    return e.elements = this.toArray(), e.length = e.elements.length, e
}, lunr.SortedSet.prototype.union = function(e) {
    var t, i, n;
    return this.length >= e.length ? (t = this, i = e) : (t = e, i = this), n = t.clone(), n.add.apply(n, i.toArray()), n
}, lunr.SortedSet.prototype.toJSON = function() {
    return this.toArray()
}, lunr.Index = function() {
    this._fields = [], this._ref = "id", this.pipeline = new lunr.Pipeline, this.documentStore = new lunr.Store, this.tokenStore = new lunr.TokenStore, this.corpusTokens = new lunr.SortedSet, this.eventEmitter = new lunr.EventEmitter, this._idfCache = {}, this.on("add", "remove", "update", function() {
        this._idfCache = {}
    }.bind(this))
}, lunr.Index.prototype.on = function() {
    var e = Array.prototype.slice.call(arguments);
    return this.eventEmitter.addListener.apply(this.eventEmitter, e)
}, lunr.Index.prototype.off = function(e, t) {
    return this.eventEmitter.removeListener(e, t)
}, lunr.Index.load = function(e) {
    e.version !== lunr.version && lunr.utils.warn("version mismatch: current " + lunr.version + " importing " + e.version);
    var t = new this;
    return t._fields = e.fields, t._ref = e.ref, t.documentStore = lunr.Store.load(e.documentStore), t.tokenStore = lunr.TokenStore.load(e.tokenStore), t.corpusTokens = lunr.SortedSet.load(e.corpusTokens), t.pipeline = lunr.Pipeline.load(e.pipeline), t
}, lunr.Index.prototype.field = function(e, t) {
    var t = t || {}, i = {
        name: e,
        boost: t.boost || 1
    };
    return this._fields.push(i), this
}, lunr.Index.prototype.ref = function(e) {
    return this._ref = e, this
}, lunr.Index.prototype.add = function(e, t) {
    var i = {}, n = new lunr.SortedSet, o = e[this._ref], t = void 0 === t?!0 : t;
    this._fields.forEach(function(t) {
        var o = this.pipeline.run(lunr.tokenizer(e[t.name]));
        i[t.name] = o, lunr.SortedSet.prototype.add.apply(n, o)
    }, this), this.documentStore.set(o, n), lunr.SortedSet.prototype.add.apply(this.corpusTokens, n.toArray());
    for (var s = 0; s < n.length; s++) {
        var r = n.elements[s], a = this._fields.reduce(function(e, t) {
            var n = i[t.name].length;
            if (!n)
                return e;
            var o = i[t.name].filter(function(e) {
                return e === r
            }).length;
            return e + o / n * t.boost
        }, 0);
        this.tokenStore.add(r, {
            ref: o,
            tf: a
        })
    }
    t && this.eventEmitter.emit("add", e, this)
}, lunr.Index.prototype.remove = function(e, t) {
    var i = e[this._ref], t = void 0 === t?!0 : t;
    if (this.documentStore.has(i)) {
        var n = this.documentStore.get(i);
        this.documentStore.remove(i), n.forEach(function(e) {
            this.tokenStore.remove(e, i)
        }, this), t && this.eventEmitter.emit("remove", e, this)
    }
}, lunr.Index.prototype.update = function(e, t) {
    var t = void 0 === t?!0 : t;
    this.remove(e, !1), this.add(e, !1), t && this.eventEmitter.emit("update", e, this)
}, lunr.Index.prototype.idf = function(e) {
    var t = "@" + e;
    if (Object.prototype.hasOwnProperty(this._idfCache, t))
        return this._idfCache[t];
    var i = this.tokenStore.count(e), n = 1;
    return i > 0 && (n = 1 + Math.log(this.tokenStore.length / i)), this._idfCache[t] = n
}, lunr.Index.prototype.search = function(e) {
    var t = this.pipeline.run(lunr.tokenizer(e)), i = lunr.utils.zeroFillArray(this.corpusTokens.length), n = [], o = this._fields.reduce(function(e, t) {
        return e + t.boost
    }, 0), s = t.some(function(e) {
        return this.tokenStore.has(e)
    }, this);
    if (!s)
        return [];
    t.forEach(function(e, t, s) {
        var r = 1 / s.length * this._fields.length * o, a = this, l = this.tokenStore.expand(e).reduce(function(t, n) {
            var o = a.corpusTokens.indexOf(n), s = a.idf(n), l = 1, c = new lunr.SortedSet;
            if (n !== e) {
                var u = Math.max(3, n.length - e.length);
                l = 1 / Math.log(u)
            }
            return o>-1 && (i[o] = r * s * l), Object.keys(a.tokenStore.get(n)).forEach(function(e) {
                c.add(e)
            }), t.union(c)
        }, new lunr.SortedSet);
        n.push(l)
    }, this);
    var r = n.reduce(function(e, t) {
        return e.intersect(t)
    }), a = new lunr.Vector(i);
    return r.map(function(e) {
        return {
            ref: e,
            score: a.similarity(this.documentVector(e))
        }
    }, this).sort(function(e, t) {
        return t.score - e.score
    })
}, lunr.Index.prototype.documentVector = function(e) {
    for (var t = this.documentStore.get(e), i = t.length, n = lunr.utils.zeroFillArray(this.corpusTokens.length), o = 0; i > o; o++) {
        var s = t.elements[o], r = this.tokenStore.get(s)[e].tf, a = this.idf(s);
        n[this.corpusTokens.indexOf(s)] = r * a
    }
    return new lunr.Vector(n)
}, lunr.Index.prototype.toJSON = function() {
    return {
        version: lunr.version,
        fields: this._fields,
        ref: this._ref,
        documentStore: this.documentStore.toJSON(),
        tokenStore: this.tokenStore.toJSON(),
        corpusTokens: this.corpusTokens.toJSON(),
        pipeline: this.pipeline.toJSON()
    }
}, lunr.Store = function() {
    this.store = {}, this.length = 0
}, lunr.Store.load = function(e) {
    var t = new this;
    return t.length = e.length, t.store = Object.keys(e.store).reduce(function(t, i) {
        return t[i] = lunr.SortedSet.load(e.store[i]), t
    }, {}), t
}, lunr.Store.prototype.set = function(e, t) {
    this.store[e] = t, this.length = Object.keys(this.store).length
}, lunr.Store.prototype.get = function(e) {
    return this.store[e]
}, lunr.Store.prototype.has = function(e) {
    return e in this.store
}, lunr.Store.prototype.remove = function(e) {
    this.has(e) && (delete this.store[e], this.length--)
}, lunr.Store.prototype.toJSON = function() {
    return {
        store: this.store,
        length: this.length
    }
}, lunr.stemmer = function() {
    var e = {
        ational: "ate",
        tional: "tion",
        enci: "ence",
        anci: "ance",
        izer: "ize",
        bli: "ble",
        alli: "al",
        entli: "ent",
        eli: "e",
        ousli: "ous",
        ization: "ize",
        ation: "ate",
        ator: "ate",
        alism: "al",
        iveness: "ive",
        fulness: "ful",
        ousness: "ous",
        aliti: "al",
        iviti: "ive",
        biliti: "ble",
        logi: "log"
    }, t = {
        icate: "ic",
        ative: "",
        alize: "al",
        iciti: "ic",
        ical: "ic",
        ful: "",
        ness: ""
    }, i = "[^aeiou]", n = "[aeiouy]", o = i + "[^aeiouy]*", s = n + "[aeiou]*", r = "^(" + o + ")?" + s + o, a = "^(" + o + ")?" + s + o + "(" + s + ")?$", l = "^(" + o + ")?" + s + o + s + o, c = "^(" + o + ")?" + n;
    return function(i) {
        var s, u, d, h, p, f, m;
        if (i.length < 3)
            return i;
        if (d = i.substr(0, 1), "y" == d && (i = d.toUpperCase() + i.substr(1)), h = /^(.+?)(ss|i)es$/, p = /^(.+?)([^s])s$/, h.test(i) ? i = i.replace(h, "$1$2") : p.test(i) && (i = i.replace(p, "$1$2")), h = /^(.+?)eed$/, p = /^(.+?)(ed|ing)$/, h.test(i)) {
            var g = h.exec(i);
            h = new RegExp(r), h.test(g[1]) && (h = /.$/, i = i.replace(h, ""))
        } else if (p.test(i)) {
            var g = p.exec(i);
            s = g[1], p = new RegExp(c), p.test(s) && (i = s, p = /(at|bl|iz)$/, f = new RegExp("([^aeiouylsz])\\1$"), m = new RegExp("^" + o + n + "[^aeiouwxy]$"), p.test(i) ? i += "e" : f.test(i) ? (h = /.$/, i = i.replace(h, "")) : m.test(i) && (i += "e"))
        }
        if (h = /^(.+?)y$/, h.test(i)) {
            var g = h.exec(i);
            s = g[1], h = new RegExp(c), h.test(s) && (i = s + "i")
        }
        if (h = /^(.+?)(ational|tional|enci|anci|izer|bli|alli|entli|eli|ousli|ization|ation|ator|alism|iveness|fulness|ousness|aliti|iviti|biliti|logi)$/, h.test(i)) {
            var g = h.exec(i);
            s = g[1], u = g[2], h = new RegExp(r), h.test(s) && (i = s + e[u])
        }
        if (h = /^(.+?)(icate|ative|alize|iciti|ical|ful|ness)$/, h.test(i)) {
            var g = h.exec(i);
            s = g[1], u = g[2], h = new RegExp(r), h.test(s) && (i = s + t[u])
        }
        if (h = /^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)$/, p = /^(.+?)(s|t)(ion)$/, h.test(i)) {
            var g = h.exec(i);
            s = g[1], h = new RegExp(l), h.test(s) && (i = s)
        } else if (p.test(i)) {
            var g = p.exec(i);
            s = g[1] + g[2], p = new RegExp(l), p.test(s) && (i = s)
        }
        if (h = /^(.+?)e$/, h.test(i)) {
            var g = h.exec(i);
            s = g[1], h = new RegExp(l), p = new RegExp(a), f = new RegExp("^" + o + n + "[^aeiouwxy]$"), (h.test(s) || p.test(s)&&!f.test(s)) && (i = s)
        }
        return h = /ll$/, p = new RegExp(l), h.test(i) && p.test(i) && (h = /.$/, i = i.replace(h, "")), "y" == d && (i = d.toLowerCase() + i.substr(1)), i
    }
}(), lunr.Pipeline.registerFunction(lunr.stemmer, "stemmer"), lunr.stopWordFilter = function(e) {
    return - 1 === lunr.stopWordFilter.stopWords.indexOf(e) ? e : void 0
}, lunr.stopWordFilter.stopWords = new lunr.SortedSet, lunr.stopWordFilter.stopWords.length = 119, lunr.stopWordFilter.stopWords.elements = ["", "a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"], lunr.Pipeline.registerFunction(lunr.stopWordFilter, "stopWordFilter"), lunr.TokenStore = function() {
    this.root = {
        docs: {}
    }, this.length = 0
}, lunr.TokenStore.load = function(e) {
    var t = new this;
    return t.root = e.root, t.length = e.length, t
}, lunr.TokenStore.prototype.add = function(e, t, i) {
    var i = i || this.root, n = e[0], o = e.slice(1);
    return n in i || (i[n] = {
        docs: {}
    }), 0 === o.length ? (i[n].docs[t.ref] = t, this.length += 1, void 0) : this.add(o, t, i[n])
}, lunr.TokenStore.prototype.has = function(e) {
    if (!e)
        return !1;
    for (var t = this.root, i = 0; i < e.length; i++) {
        if (!t[e[i]])
            return !1;
        t = t[e[i]]
    }
    return !0
}, lunr.TokenStore.prototype.getNode = function(e) {
    if (!e)
        return {};
    for (var t = this.root, i = 0; i < e.length; i++) {
        if (!t[e[i]])
            return {};
        t = t[e[i]]
    }
    return t
}, lunr.TokenStore.prototype.get = function(e, t) {
    return this.getNode(e, t).docs || {}
}, lunr.TokenStore.prototype.count = function(e, t) {
    return Object.keys(this.get(e, t)).length
}, lunr.TokenStore.prototype.remove = function(e, t) {
    if (e) {
        for (var i = this.root, n = 0; n < e.length; n++) {
            if (!(e[n]in i))
                return;
            i = i[e[n]]
        }
        delete i.docs[t]
    }
}, lunr.TokenStore.prototype.expand = function(e, t) {
    var i = this.getNode(e), n = i.docs || {}, t = t || [];
    return Object.keys(n).length && t.push(e), Object.keys(i).forEach(function(i) {
        "docs" !== i && t.concat(this.expand(e + i, t))
    }, this), t
}, lunr.TokenStore.prototype.toJSON = function() {
    return {
        root: this.root,
        length: this.length
    }
}, _wAMD.define("lunr", function() {}), function(e, t) {
    "object" == typeof exports ? function() {
        var e;
        try {
            e = require("lunr")
        } catch (i) {}
        module.exports = t(require("underscore"), require("backbone"), require("backgrid"), e)
    }() : "function" == typeof define && define.amd ? _wAMD.define("backgrid-filter", ["underscore", "backbone", "backgrid", "lunr"], t) : t(e._, e.Backbone, e.Backgrid, e.lunr)
}(this, function(e, t, i, n) {
    var o = i.Extension.ServerSideFilter = t.View.extend({
        tagName: "form",
        className: "backgrid-filter form-search",
        template: e.template('<span class="search">&nbsp;</span><input type="search" <% if (placeholder) { %> placeholder="<%- placeholder %>" <% } %> name="<%- name %>" /><a class="clear" data-backgrid-action="clear" href="#">&times;</a>', null, {
            variable: null
        }),
        events: {
            "keyup input[type=search]": "showClearButtonMaybe",
            "click a[data-backgrid-action=clear]": "clear",
            submit: "search"
        },
        name: "q",
        placeholder: null,
        initialize: function(e) {
            o.__super__.initialize.apply(this, arguments), this.name = e.name || this.name, this.placeholder = e.placeholder || this.placeholder, this.template = e.template || this.template;
            var i = this.collection, n = this;
            t.PageableCollection && i instanceof t.PageableCollection && "server" == i.mode && (i.queryParams[this.name] = function() {
                return n.searchBox().val() || null
            })
        },
        showClearButtonMaybe: function() {
            var e = this.clearButton(), t = this.searchBox().val();
            t ? e.show() : e.hide()
        },
        searchBox: function() {
            return this.$el.find("input[type=search]")
        },
        clearButton: function() {
            return this.$el.find("a[data-backgrid-action=clear]")
        },
        search: function(i) {
            i && i.preventDefault();
            var n = this.collection, o = {}, s = this.searchBox().val(), r = {
                property: this.name,
                value: "%" + s + "%",
                comparison: "LIKE"
            };
            if (e.isUndefined(n.fetchOptions) && (n.fetchOptions = {}), (e.isUndefined(n.fetchOptions.filter) ||!e.isArray(n.fetchOptions.filter)) && (n.fetchOptions.filter = []), s) {
                o[this.name] = s;
                var a=!1;
                e.each(n.fetchOptions.filter, function(e, t) {
                    e.property == r.property && (n.fetchOptions.filter[t] = r, a=!0)
                }), a || n.fetchOptions.filter.push(r)
            } else {
                var l = [], c=!1;
                e.each(n.fetchOptions.filter, function(e) {
                    e.property == r.property ? c=!0 : l.push(e)
                }), c && (n.fetchOptions.filter = l)
            }
            t.PageableCollection && n instanceof t.PageableCollection ? n.getFirstPage({
                data: o,
                reset: !0,
                fetch: !0
            }) : n.fetch({
                data: o,
                reset: !0
            })
        },
        clear: function(e) {
            e && e.preventDefault(), this.searchBox().val(null), this.showClearButtonMaybe();
            var i = this.collection;
            t.PageableCollection && i instanceof t.PageableCollection ? i.getFirstPage({
                reset: !0,
                fetch: !0
            }) : i.fetch({
                reset: !0
            })
        },
        render: function() {
            return this.$el.empty().append(this.template({
                name: this.name,
                placeholder: this.placeholder,
                value: this.value
            })), this.showClearButtonMaybe(), this.delegateEvents(), this
        }
    }), s = i.Extension.ClientSideFilter = o.extend({
        events: e.extend({}, o.prototype.events, {
            "click a[data-backgrid-action=clear]": function(e) {
                e.preventDefault(), this.clear()
            },
            "keydown input[type=search]": "search",
            submit: function(e) {
                e.preventDefault(), this.search()
            }
        }),
        fields: null,
        wait: 149,
        initialize: function(t) {
            s.__super__.initialize.apply(this, arguments), this.fields = t.fields || this.fields, this.wait = t.wait || this.wait, this._debounceMethods(["search", "clear"]);
            var i = this.collection = this.collection.fullCollection || this.collection, n = this.shadowCollection = i.clone();
            this.listenTo(i, "add", function(e, t, i) {
                n.add(e, i)
            }), this.listenTo(i, "remove", function(e, t, i) {
                n.remove(e, i)
            }), this.listenTo(i, "sort", function(e) {
                this.searchBox().val() || n.reset(e.models)
            }), this.listenTo(i, "reset", function(t, i) {
                i = e.extend({
                    reindex: !0
                }, i || {}), i.reindex && null == i.from && null == i.to && n.reset(t.models)
            })
        },
        _debounceMethods: function(t) {
            e.isString(t) && (t = [t]), this.undelegateEvents();
            for (var i = 0, n = t.length; n > i; i++) {
                var o = t[i], s = this[o];
                this[o] = e.debounce(s, this.wait)
            }
            this.delegateEvents()
        },
        makeRegExp: function(e) {
            return new RegExp(e.trim().split(/\s+/).join("|"), "i")
        },
        makeMatcher: function(e) {
            var t = this.makeRegExp(e);
            return function(e) {
                for (var i = this.fields || e.keys(), n = 0, o = i.length; o > n; n++)
                    if (t.test(e.get(i[n]) + ""))
                        return !0;
                return !1
            }
        },
        search: function() {
            var t = e.bind(this.makeMatcher(this.searchBox().val()), this), i = this.collection;
            i.pageableCollection && i.pageableCollection.getFirstPage({
                silent: !0
            }), i.reset(this.shadowCollection.filter(t), {
                reindex: !1
            })
        },
        clear: function() {
            this.searchBox().val(null), this.showClearButtonMaybe();
            var e = this.collection;
            e.pageableCollection && e.pageableCollection.getFirstPage({
                silent: !0
            }), e.reset(this.shadowCollection.models, {
                reindex: !1
            })
        }
    }), r = i.Extension.LunrFilter = s.extend({
        ref: "id",
        fields: null,
        initialize: function(e) {
            r.__super__.initialize.apply(this, arguments), this.ref = e.ref || this.ref;
            var t = this.collection = this.collection.fullCollection || this.collection;
            this.listenTo(t, "add", this.addToIndex), this.listenTo(t, "remove", this.removeFromIndex), this.listenTo(t, "reset", this.resetIndex), this.listenTo(t, "change", this.updateIndex), this.resetIndex(t)
        },
        resetIndex: function(t, i) {
            if (i = e.extend({
                reindex: !0
            }, i || {}), i.reindex) {
                var o = this;
                this.index = n(function() {
                    e.each(o.fields, function(e, t) {
                        this.field(t, e), this.ref(o.ref)
                    }, this)
                }), t.each(function(e) {
                    this.addToIndex(e)
                }, this)
            }
        },
        addToIndex: function(e) {
            var t = this.index, i = e.toJSON();
            t.documentStore.has(i[this.ref]) ? t.update(i) : t.add(i)
        },
        removeFromIndex: function(e) {
            var t = this.index, i = e.toJSON();
            t.documentStore.has(i[this.ref]) && t.remove(i)
        },
        updateIndex: function(t) {
            var i = t.changedAttributes();
            i&&!e.isEmpty(e.intersection(e.keys(this.fields), e.keys(i))) && this.index.update(t.toJSON())
        },
        search: function() {
            var e = this.collection;
            if (!this.searchBox().val())
                return e.reset(this.shadowCollection.models, {
                    reindex: !1
                }), void 0;
            for (var t = this.index.search(this.searchBox().val()), i = [], n = 0; n < t.length; n++) {
                var o = t[n];
                i.push(this.shadowCollection.get(o.ref))
            }
            e.pageableCollection && e.pageableCollection.getFirstPage({
                silent: !0
            }), e.reset(i, {
                reindex: !1
            })
        }
    })
}), function(e) {
    if ("object" == typeof exports)
        module.exports = e(require("underscore"), require("backbone"));
    else if ("function" == typeof define && define.amd)
        _wAMD.define("backbone-pageable", ["underscore", "backbone"], e);
    else if ("undefined" != typeof _ && "undefined" != typeof Backbone) {
        var t = Backbone.PageableCollection, i = e(_, Backbone);
        Backbone.PageableCollection.noConflict = function() {
            return Backbone.PageableCollection = t, i
        }
    }
}(function(e, t) {
    function i(t, i) {
        if (!e.isNumber(t) || e.isNaN(t) ||!e.isFinite(t)||~~t !== t)
            throw new TypeError("`" + i + "` must be a finite integer");
        return t
    }
    function n(e) {
        for (var t, i, n, o, s = {}, r = decodeURIComponent, a = e.split("&"), l = 0, c = a.length; c > l; l++) {
            var u = a[l];
            t = u.split("="), i = t[0], n = t[1] ||!0, i = r(i), n = r(n), o = s[i], f(o) ? o.push(n) : s[i] = o ? [o, n] : n
        }
        return s
    }
    function o(e, t, i) {
        var n = e._events[t];
        if (n && n.length) {
            var o = n[n.length - 1], s = o.callback;
            o.callback = function() {
                try {
                    s.apply(this, arguments), i()
                } catch (e) {
                    throw e
                } finally {
                    o.callback = s
                }
            }
        } else 
            i()
    }
    var s = e.extend, r = e.omit, a = e.clone, l = e.each, c = e.pick, u = e.contains, d = e.isEmpty, h = e.pairs, p = e.invert, f = e.isArray, m = e.isFunction, g = e.isObject, v = e.keys, y = e.isUndefined, b = e.result, w = Math.ceil, _ = Math.floor, C = Math.max, k = t.Collection.prototype, x = /[\s'"]/g, S = /[<>\s'"]/g, M = t.PageableCollection = t.Collection.extend({
        state: {
            firstPage: 1,
            lastPage: null,
            currentPage: null,
            pageSize: 25,
            totalPages: null,
            totalRecords: null,
            sortKey: null,
            order: - 1
        },
        mode: "server",
        queryParams: {
            currentPage: "page",
            pageSize: "per_page",
            totalPages: "total_pages",
            totalRecords: "total_entries",
            sortKey: "sort_by",
            order: "order",
            directions: {
                "-1": "asc",
                1: "desc"
            }
        },
        constructor: function(e, t) {
            k.constructor.apply(this, arguments), t = t || {};
            var i = this.mode = t.mode || this.mode || T.mode, n = s({}, T.queryParams, this.queryParams, t.queryParams || {});
            n.directions = s({}, T.queryParams.directions, this.queryParams.directions, n.directions || {}), this.queryParams = n;
            var o = this.state = s({}, T.state, this.state, t.state || {});
            o.currentPage = null == o.currentPage ? o.firstPage : o.currentPage, f(e) || (e = e ? [e] : []), "server" == i || null != o.totalRecords || d(e) || (o.totalRecords = e.length), this.switchMode(i, s({
                fetch: !1,
                resetState: !1,
                models: e
            }, t));
            var r = t.comparator;
            if (o.sortKey&&!r && this.setSorting(o.sortKey, o.order, t), "server" != i) {
                var l = this.fullCollection;
                r && t.full && (this.comparator = null, l.comparator = r), t.full && l.sort(), e&&!d(e) && (this.reset([].slice.call(e), s({
                    silent: !0
                }, t)), this.getPage(o.currentPage), e.splice.apply(e, [0, e.length].concat(this.models)))
            }
            this._initState = a(this.state)
        },
        _makeFullCollection: function(e, i) {
            var n, o, s, r = ["url", "model", "sync", "comparator"], a = this.constructor.prototype, l = {};
            for (n = 0, o = r.length; o > n; n++)
                s = r[n], y(a[s]) || (l[s] = a[s]);
            var c = new (t.Collection.extend(l))(e, i);
            for (n = 0, o = r.length; o > n; n++)
                s = r[n], this[s] !== a[s] && (c[s] = this[s]);
            return c
        },
        _makeCollectionEventHandler: function(e, t) {
            return function(i, n, r, c) {
                var u = e._handlers;
                l(v(u), function(i) {
                    var n = u[i];
                    e.off(i, n), t.off(i, n)
                });
                var d = a(e.state), h = d.firstPage, p = 0 === h ? d.currentPage: d.currentPage - 1, f = d.pageSize, m = p * f, g = m + f;
                if ("add" == i) {
                    var b, _, C, k, c = c || {};
                    if (r == t)
                        _ = t.indexOf(n), _ >= m && g > _ && (k = e, b = C = _ - m);
                    else {
                        b = e.indexOf(n), _ = m + b, k = t;
                        var C = y(c.at) ? _: c.at + m
                    }
                    if (c.onRemove || (++d.totalRecords, delete c.onRemove), e.state = e._checkState(d), k) {
                        k.add(n, s({}, c || {}, {
                            at: C
                        }));
                        var x = b >= f ? n: !y(c.at) && g > C && e.length > f ? e.at(f): null;
                        x && o(r, i, function() {
                            e.remove(x, {
                                onAdd: !0
                            })
                        })
                    }
                }
                if ("remove" == i)
                    if (c.onAdd)
                        delete c.onAdd;
                    else {
                        if (--d.totalRecords) {
                            var S = d.totalPages = w(d.totalRecords / f);
                            d.lastPage = 0 === h ? S - 1 : S || h, d.currentPage > S && (d.currentPage = d.lastPage)
                        } else 
                            d.totalRecords = null, d.totalPages = null;
                            e.state = e._checkState(d);
                            var M, T = c.index;
                            r == e ? ((M = t.at(g)) && o(e, i, function() {
                                e.push(M, {
                                    onRemove: !0
                                })
                            }), t.remove(n)) : T >= m && g > T && ((M = t.at(g - 1)) && o(e, i, function() {
                                e.push(M, {
                                    onRemove: !0
                                })
                            }), e.remove(n))
                    }
                if ("reset" == i)
                    if (c = r, r = n, r == e && null == c.from && null == c.to) {
                        var D = t.models.slice(0, m), E = t.models.slice(m + e.models.length);
                        t.reset(D.concat(e.models).concat(E), c)
                    } else 
                        r == t && ((d.totalRecords = t.models.length) || (d.totalRecords = null, d.totalPages = null), "client" == e.mode && (d.lastPage = d.currentPage = d.firstPage), e.state = e._checkState(d), e.reset(t.models.slice(m, g), s({}, c, {
                            parse: !1
                        })));
                "sort" == i && (c = r, r = n, r === t && e.reset(t.models.slice(m, g), s({}, c, {
                    parse: !1
                }))), l(v(u), function(i) {
                    var n = u[i];
                    l([e, t], function(e) {
                        e.on(i, n);
                        var t = e._events[i] || [];
                        t.unshift(t.pop())
                    })
                })
            }
        },
        _checkState: function(e) {
            var t = this.mode, n = this.links, o = e.totalRecords, s = e.pageSize, r = e.currentPage, a = e.firstPage, l = e.totalPages;
            if (null != o && null != s && null != r && null != a && ("infinite" == t ? n : !0)) {
                if (o = i(o, "totalRecords"), s = i(s, "pageSize"), r = i(r, "currentPage"), a = i(a, "firstPage"), 1 > s)
                    throw new RangeError("`pageSize` must be >= 1");
                if (l = e.totalPages = w(o / s), 0 > a || a > 1)
                    throw new RangeError("`firstPage must be 0 or 1`");
                if (e.lastPage = 0 === a ? C(0, l - 1) : l || a, "infinite" == t) {
                    if (!n[r + ""])
                        throw new RangeError("No link found for page " + r)
                    } else if (a > r || l > 0 && (a ? r > l : r >= l))
                    throw new RangeError("`currentPage` must be firstPage <= currentPage " + (a ? ">" : ">=") + " totalPages if " + a + "-based. Got " + r + ".")
            }
            return e
        },
        setPageSize: function(e, t) {
            e = i(e, "pageSize"), t = t || {
                first: !1
            };
            var n = this.state, o = w(n.totalRecords / e), a = o ? C(n.firstPage, _(o * (n.firstPage ? n.currentPage : n.currentPage + 1) / n.totalPages)): n.firstPage;
            return n = this.state = this._checkState(s({}, n, {
                pageSize: e,
                currentPage: t.first ? n.firstPage: a,
                totalPages: o
            })), this.getPage(n.currentPage, r(t, ["first"]))
        },
        switchMode: function(t, i) {
            if (!u(["server", "client", "infinite"], t))
                throw new TypeError('`mode` must be one of "server", "client" or "infinite"');
            i = i || {
                fetch: !0,
                resetState: !0
            };
            var n = this.state = i.resetState ? a(this._initState): this._checkState(s({}, this.state));
            this.mode = t;
            var o, c = this, d = this.fullCollection, h = this._handlers = this._handlers || {};
            if ("server" == t || d)
                "server" == t && d && (l(v(h), function(e) {
                    o = h[e], c.off(e, o), d.off(e, o)
                }), delete this._handlers, this._fullComparator = d.comparator, delete this.fullCollection);
            else {
                d = this._makeFullCollection(i.models || [], i), d.pageableCollection = this, this.fullCollection = d;
                var p = this._makeCollectionEventHandler(this, d);
                l(["add", "remove", "reset", "sort"], function(t) {
                    h[t] = o = e.bind(p, {}, t), c.on(t, o), d.on(t, o)
                }), d.comparator = this._fullComparator
            }
            if ("infinite" == t)
                for (var f = this.links = {}, m = n.firstPage, g = w(n.totalRecords / n.pageSize), y = 0 === m ? C(0, g - 1) : g || m, b = n.firstPage; y >= b; b++)
                    f[b] = this.url;
            else 
                this.links && delete this.links;
            return i.fetch ? this.fetch(r(i, "fetch", "resetState")) : this
        },
        hasPrevious: function() {
            var e = this.state, t = e.currentPage;
            return "infinite" != this.mode ? t > e.firstPage : !!this.links[t - 1]
        },
        hasNext: function() {
            var e = this.state, t = this.state.currentPage;
            return "infinite" != this.mode ? t < e.lastPage : !!this.links[t + 1]
        },
        getFirstPage: function(e) {
            return this.getPage("first", e)
        },
        getPreviousPage: function(e) {
            return this.getPage("prev", e)
        },
        getNextPage: function(e) {
            return this.getPage("next", e)
        },
        getLastPage: function(e) {
            return this.getPage("last", e)
        },
        getPage: function(e, t) {
            var n = this.mode, o = this.fullCollection;
            t = t || {
                fetch: !1
            };
            var a = this.state, l = a.firstPage, c = a.currentPage, u = a.lastPage, h = a.pageSize, p = e;
            switch (e) {
            case"first":
                p = l;
                break;
            case"prev":
                p = c - 1;
                break;
            case"next":
                p = c + 1;
                break;
            case"last":
                p = u;
                break;
            default:
                p = i(e, "index")
            }
            this.state = this._checkState(s({}, a, {
                currentPage: p
            })), t.from = c, t.to = p;
            var f = (0 === l ? p : p - 1) * h, m = o && o.length ? o.models.slice(f, f + h): [];
            return "client" != n && ("infinite" != n || d(m)) || t.fetch ? ("infinite" == n && (t.url = this.links[p]), this.fetch(r(t, "fetch"))) : (this.reset(m, r(t, "fetch")), this)
        },
        getPageByOffset: function(e, t) {
            if (0 > e)
                throw new RangeError("`offset must be > 0`");
            e = i(e);
            var n = _(e / this.state.pageSize);
            return 0 !== this.state.firstPage && n++, n > this.state.lastPage && (n = this.state.lastPage), this.getPage(n, t)
        },
        sync: function(e, i, n) {
            var o = this;
            if ("infinite" == o.mode) {
                var r = n.success, a = o.state.currentPage;
                n.success = function(e, t, i) {
                    var l = o.links, c = o.parseLinks(e, s({
                        xhr: i
                    }, n));
                    c.first && (l[o.state.firstPage] = c.first), c.prev && (l[a - 1] = c.prev), c.next && (l[a + 1] = c.next), r && r(e, t, i)
                }
            }
            return (k.sync || t.sync).call(o, e, i, n)
        },
        parseLinks: function(e, t) {
            var i = {}, n = t.xhr.getResponseHeader("Link");
            if (n) {
                var o = ["first", "prev", "next"];
                l(n.split(","), function(e) {
                    var t = e.split(";"), n = t[0].replace(S, ""), s = t.slice(1);
                    l(s, function(e) {
                        var t = e.split("="), s = t[0].replace(x, ""), r = t[1].replace(x, "");
                        "rel" == s && u(o, r) && (i[r] = n)
                    })
                })
            }
            return i
        },
        parse: function(e, t) {
            var i = this.parseState(e, a(this.queryParams), a(this.state), t);
            return i && (this.state = this._checkState(s({}, this.state, i))), this.parseRecords(e, t)
        },
        parseState: function(t, i, n) {
            if (t && 2 === t.length && g(t[0]) && f(t[1])) {
                var o = a(n), s = t[0];
                return l(h(r(i, "directions")), function(t) {
                    var i = t[0], n = t[1], r = s[n];
                    y(r) || e.isNull(r) || (o[i] = s[n])
                }), s.order && (o.order = 1 * p(i.directions)[s.order]), o
            }
        },
        parseRecords: function(e) {
            return e && 2 === e.length && g(e[0]) && f(e[1]) ? e[1] : e
        },
        fetch: function(e) {
            e = e || {};
            var t = this._checkState(this.state), i = this.mode;
            "infinite" != i || e.url || (e.url = this.links[t.currentPage]);
            var o = e.data || {}, l = b(e, "url") || b(this, "url") || "", u = l.indexOf("?");
            - 1 != u && (s(o, n(l.slice(u + 1))), l = l.slice(0, u)), e.url = l, e.data = o;
            var d, p, f, g, w = "client" == this.mode ? c(this.queryParams, "sortKey", "order"): r(c(this.queryParams, v(T.queryParams)), "directions"), _ = h(w), C = a(this);
            for (d = 0; d < _.length; d++)
                p = _[d], f = p[0], g = p[1], g = m(g) ? g.call(C) : g, null != t[f] && null != g && (o[g] = t[f]);
            t.sortKey && t.order ? o[w.order] = this.queryParams.directions[t.order + ""] : t.sortKey || delete o[w.order];
            var x = h(r(this.queryParams, v(T.queryParams)));
            for (d = 0; d < x.length; d++)
                p = x[d], g = p[1], g = m(g) ? g.call(C) : g, null != g && (o[p[0]] = g);
            if ("server" != i) {
                var S = this, M = this.fullCollection, D = e.success;
                return e.success = function(t, n, o) {
                    o = o || {}, y(e.silent) ? delete o.silent : o.silent = e.silent;
                    var r = t.models;
                    "client" == i ? M.reset(r, o) : (M.add(r, s({
                        at: M.length
                    }, s(o, {
                        parse: !1
                    }))), S.trigger("reset", S, o)), D && D(t, n, o)
                }, k.fetch.call(this, s({}, e, {
                    silent: !0
                }))
            }
            return k.fetch.call(this, e)
        },
        _makeComparator: function(e, t, i) {
            var n = this.state;
            return e = e || n.sortKey, t = t || n.order, e && t ? (i || (i = function(e, t) {
                return e.get(t)
            }), function(n, o) {
                var s, r = i(n, e), a = i(o, e);
                return 1 === t && (s = r, r = a, a = s), r === a ? 0 : a > r?-1 : 1
            }) : void 0
        },
        setSorting: function(t, i, n) {
            var o = this.state;
            o.sortKey = t, o.order = i = i || o.order;
            var r = this.fullCollection, a=!1, l=!1;
            t || (a = l=!0);
            var c = this.mode;
            n = s({
                side: "client" == c ? c: "server",
                full: !0
            }, n);
            var u = this._makeComparator(t, i, n.sortValue), d = n.full, h = n.side;
            return "client" == h ? d ? (r && (r.comparator = u), a=!0) : (this.comparator = u, l=!0) : "server" != h || d || (this.comparator = u), a && (this.comparator = null), l && r && (r.comparator = null), this.fetchOptions = e.extend({}, this.fetchOptions, {
                sort: [{
                    property: t,
                    direction: i > 0 ? "ASC": "DESC"
                }
                ]
            }), this
        }
    }), T = M.prototype;
    return M
}), function(e, t) {
    "object" == typeof exports ? module.exports = t(require("underscore"), require("backbone"), require("backgrid"), require("backbone-pageable")) : "function" == typeof define && define.amd ? _wAMD.define("backgrid-paginator", ["underscore", "backbone", "backgrid", "backbone-pageable"], t) : t(e._, e.Backbone, e.Backgrid)
}(this, function(e, t, i) {
    var n = i.Extension.PageHandle = t.View.extend({
        tagName: "li",
        events: {
            "click a": "changePage"
        },
        title: e.template("Page <%- label %>", null, {
            variable: null
        }),
        isRewind: !1,
        isBack: !1,
        isForward: !1,
        isFastForward: !1,
        initialize: function(t) {
            var i = this.collection, n = i.state, o = n.currentPage, s = n.firstPage, r = n.lastPage;
            e.extend(this, e.pick(t, ["isRewind", "isBack", "isForward", "isFastForward"]));
            var a;
            this.isRewind ? a = s : this.isBack ? a = Math.max(s, o - 1) : this.isForward ? a = Math.min(r, o + 1) : this.isFastForward ? a = r : (a =+ t.pageIndex, a = s ? a + 1 : a), this.pageIndex = a, this.label = (t.label || (s ? a : a + 1)) + "";
            var l = t.title || this.title;
            this.title = e.isFunction(l) ? l({
                label: this.label
            }) : l
        },
        render: function() {
            this.$el.empty();
            var e = document.createElement("a");
            e.href = "#", this.title && (e.title = this.title), e.innerHTML = this.label, this.el.appendChild(e);
            var t = this.collection, i = t.state, n = i.currentPage, o = this.pageIndex;
            return this.isRewind && n == i.firstPage || this.isBack&&!t.hasPrevious() || this.isForward&&!t.hasNext() || this.isFastForward && (n == i.lastPage || i.totalPages < 1) ? this.$el.addClass("disabled") : this.isRewind || this.isBack || this.isForward || this.isFastForward || i.currentPage != o || this.$el.addClass("active"), this.delegateEvents(), this
        },
        changePage: function(e) {
            e.preventDefault();
            var t = this.$el, i = this.collection;
            return t.hasClass("active") || t.hasClass("disabled") || (this.isRewind ? i.getFirstPage() : this.isBack ? i.getPreviousPage() : this.isForward ? i.getNextPage() : this.isFastForward ? i.getLastPage() : i.getPage(this.pageIndex, {
                reset: !0
            })), this
        }
    }), o = i.Extension.Paginator = t.View.extend({
        className: "backgrid-paginator",
        windowSize: 10,
        slideScale: .5,
        controls: {
            rewind: {
                label: "《",
                title: "First"
            },
            back: {
                label: "〈",
                title: "Previous"
            },
            forward: {
                label: "〉",
                title: "Next"
            },
            fastForward: {
                label: "》",
                title: "Last"
            }
        },
        renderIndexedPageHandles: !0,
        pageHandle: n,
        goBackFirstOnSort: !0,
        initialize: function(t) {
            var i = this;
            i.controls = e.defaults(t.controls || {}, i.controls, o.prototype.controls), e.extend(i, e.pick(t || {}, "windowSize", "pageHandle", "slideScale", "goBackFirstOnSort", "renderIndexedPageHandles"));
            var n = i.collection;
            i.listenTo(n, "add", i.render), i.listenTo(n, "remove", i.render), i.listenTo(n, "reset", i.render), i.listenTo(n, "backgrid:sorted", function() {
                i.goBackFirstOnSort && n.getFirstPage({
                    reset: !0
                })
            })
        },
        slideMaybe: function(e, t, i, n) {
            return Math.round(i%n / n)
        },
        slideThisMuch: function(e, t, i, n, o) {
            return ~~(n * o)
        },
        _calculateWindow: function() {
            var e = this.collection, t = e.state, i = t.firstPage, n =+ t.lastPage;
            n = Math.max(0, i ? n - 1 : n);
            var o = Math.max(t.currentPage, t.firstPage);
            o = i ? o - 1 : o;
            var s = this.windowSize, r = this.slideScale, a = Math.floor(o / s) * s;
            o <= n - this.slideThisMuch() && (a += this.slideMaybe(i, n, o, s, r) * this.slideThisMuch(i, n, o, s, r));
            var l = Math.min(n + 1, a + s);
            return [a, l]
        },
        makeHandles: function() {
            var t = [], i = this.collection, n = this._calculateWindow(), o = n[0], s = n[1];
            if (this.renderIndexedPageHandles)
                for (var r = o; s > r; r++)
                    t.push(new this.pageHandle({
                        collection: i,
                        pageIndex: r
                    }));
            var a = this.controls;
            return e.each(["back", "rewind", "forward", "fastForward"], function(e) {
                var n = a[e];
                if (n) {
                    var o = {
                        collection: i,
                        title: n.title,
                        label: n.label
                    };
                    o["is" + e.slice(0, 1).toUpperCase() + e.slice(1)]=!0;
                    var s = new this.pageHandle(o);
                    "rewind" == e || "back" == e ? t.unshift(s) : t.push(s)
                }
            }, this), t
        },
        render: function() {
            if (this.$el.empty(), this.handles)
                for (var e = 0, t = this.handles.length; t > e; e++)
                    this.handles[e].remove();
            for (var i = this.handles = this.makeHandles(), n = document.createElement("ul"), e = 0; e < i.length; e++)
                n.appendChild(i[e].render().el);
            return this.el.appendChild(n), this
        }
    })
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-affix-cell", ["backbone", "backgrid"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    var i = e.$, n = t.Extension.AffixCellEditor = t.CellEditor.extend({
        tagName: "div",
        events: {
            "blur input": "saveOrCancel",
            "keydown input": "saveOrCancel"
        },
        initialize: function() {
            this.unitContainer = i("<span/>", {
                "class": "unit"
            }), this.valueInput = i("<input/>", {
                "class": "value",
                type: "text"
            }), t.CellEditor.prototype.initialize.apply(this, arguments)
        },
        render: function() {
            var e = this.formatter.fromRaw(this.model.get(this.column.get("name")));
            return this.$el.append(this.unitContainer, this.valueInput), this.valueInput.val(e), this.unitContainer.html(this.unit), this.$el.addClass(this.affix), this
        },
        postRender: function() {
            var e = {}, t =- 1 !== _.indexOf(["suffix", "postfix"], this.affix) ? "paddingRight" : "paddingLeft";
            e[t] = this.unitContainer.width() + 15, this.valueInput.css(e).select()
        },
        saveOrCancel: function(e) {
            var i = new t.Command(e), n = "blur" === e.type || "focusout" === e.type;
            if (i.moveUp() || i.moveDown() || i.moveLeft() || i.moveRight() || i.save() || n) {
                var o = this.valueInput.val(), s = this.formatter.toRaw(o);
                e.preventDefault(), e.stopPropagation(), _.isUndefined(s) ? this.model.trigger("backgrid:error", this.model, this.column, o) : (this.model.set(this.column.get("name"), s), this.model.trigger("backgrid:edited", this.model, this.column, i))
            } else 
                i.cancel() && (e.stopPropagation(), this.model.trigger("backgrid:edited", this.model, this.column, i))
        },
        setAffix: function(e) {
            this.affix = e
        },
        setUnit: function(e) {
            this.unit = e
        }
    });
    t.Extension.AffixCell = t.Cell.extend({
        className: "affix-cell",
        editor: n,
        formatter: {
            fromRaw: function(e) {
                return e
            },
            toRaw: function(e) {
                return e
            }
        },
        affix: "prefix",
        unit: void 0,
        initialize: function() {
            t.Cell.prototype.initialize.apply(this,
            arguments),
            this.listenTo(this.model,
            "backgrid:edit",
            function(e,
            t,
            i,
            n) {
                t.get("name") === this.column.get("name") && (n.setAffix(this.affix),
                n.setUnit(this.unit))
            })
        }, render : function() {
            var e, t;
            return e = i("<span/>", {
                "class": "unit",
                html: this.unit
            }), t = i("<span/>", {
                "class": "value",
                html: this.formatter.fromRaw(this.model.get(this.column.get("name")))
            }), this.$el.addClass(this.affix), this.$el.append(e, t), this.delegateEvents(), this
        }
    })
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-virtual-cell", ["backbone", "backgrid"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    e.$, t.Extension.VirtualCell = t.Cell.extend({
        className: "virtual-cell",
        tagName: "td",
        view: e.View.extend({}),
        initialize: function(e) {
            this.column = e.column, this.column instanceof t.Column || (this.column = new t.Column(this.column));
            var i = this.column, n = this.$el;
            this.listenTo(i, "change:renderable", function(e, t) {
                n.toggleClass("renderable", t)
            }), i.get("renderable") && n.addClass("renderable")
        },
        render: function() {
            var e = new this.view({
                model: this.model,
                el: this.el
            });
            return e.render(), this.delegateEvents(), this
        }
    })
}), function(e, t, i) {
    function n(e, t, i) {
        for (var n = [], o = 0; o < e.length; o++) {
            var s = tinycolor(e[o]), r = s.toHsl().l < .5 ? "sp-thumb-el sp-thumb-dark": "sp-thumb-el sp-thumb-light";
            r += tinycolor.equals(t, e[o]) ? " sp-thumb-active" : "";
            var a = g ? "background-color:" + s.toRgbString(): "filter:" + s.toFilter();
            n.push('<span title="' + s.toRgbString() + '" data-color="' + s.toRgbString() + '" class="' + r + '"><span class="sp-thumb-inner" style="' + a + ';" /></span>')
        }
        return "<div class='sp-cf " + i + "'>" + n.join("") + "</div>"
    }
    function o() {
        for (var e = 0; e < f.length; e++)
            f[e] && f[e].hide()
    }
    function s(e, i) {
        var n = t.extend({}, p, e);
        return n.callbacks = {
            move: u(n.move, i),
            change: u(n.change, i),
            show: u(n.show, i),
            hide: u(n.hide, i),
            beforeShow: u(n.beforeShow, i)
        }, n
    }
    function r(r, l) {
        function u() {
            kt.toggleClass("sp-flat", U), kt.toggleClass("sp-input-disabled", !V.showInput), kt.toggleClass("sp-alpha-enabled", V.showAlpha), kt.toggleClass("sp-buttons-disabled", !V.showButtons || U), kt.toggleClass("sp-palette-disabled", !V.showPalette), kt.toggleClass("sp-palette-only", V.showPaletteOnly), kt.toggleClass("sp-initial-disabled", !V.showInitial), kt.addClass(V.className), kt.attr("id", V.id), z()
        }
        function p() {
            function i(e) {
                return e.data && e.data.ignore ? (A(t(this).data("color")), O()) : (A(t(this).data("color")), N(!0), O(), X && D()), !1
            }
            if (m && kt.find("*:not(input)").attr("unselectable", "on"), u(), $t && _t.hide().after(jt), U ? _t.after(kt).hide() : t(wt).append(kt), Q && e.localStorage) {
                try {
                    var n = e.localStorage[Q].split(",#");
                    n.length > 1 && (delete e.localStorage[Q], t.each(n, function(e, t) {
                        b(t)
                    }))
                } catch (o) {}
                try {
                    gt = e.localStorage[Q].split(";")
                } catch (o) {}
            }
            Ht.bind("click.spectrum touchstart.spectrum", function(e) {
                Ct || M(), e.stopPropagation(), t(e.target).is("input") || e.preventDefault()
            }), (_t.is(":disabled") || V.disabled===!0) && j(), kt.click(c), Pt.change(S), Pt.bind("paste", function() {
                setTimeout(S, 1)
            }), Pt.keydown(function(e) {
                13 == e.keyCode && (S(), e.preventDefault())
            }), It.text(V.cancelText), It.bind("click.spectrum", function(e) {
                e.stopPropagation(), e.preventDefault(), D("cancel")
            }), Wt.text(V.chooseText), Wt.bind("click.spectrum", function(e) {
                e.stopPropagation(), e.preventDefault(), R() && (N(!0), D())
            }), Nt.bind("click.spectrum", function(e) {
                e.stopPropagation(), e.preventDefault(), q()
            }), zt.bind("click.spectrum", function(e) {
                e.stopPropagation(), e.preventDefault(), H()
            }), d(Et, function(e, t, i) {
                pt = e / at, i.shiftKey && (pt = Math.round(10 * pt) / 10), O()
            }), d(Mt, function(e, t) {
                ut = parseFloat(t / st), O()
            }, k, x), d(xt, function(e, t) {
                dt = parseFloat(e / it), ht = parseFloat((nt - t) / nt), O()
            }, k, x), Vt ? (A(Vt), I(), Gt = Yt || tinycolor(Vt).format, b(Vt)) : I(), U && T();
            var s = m ? "mousedown.spectrum": "click.spectrum touchstart.spectrum";
            Ft.delegate(".sp-thumb-el", s, i), Ot.delegate(".sp-thumb-el:nth-child(1)", s, {
                ignore: !0
            }, i)
        }
        function b(i) {
            if (J) {
                var n = tinycolor(i).toRgbString();
                if ( - 1 === t.inArray(n, gt))
                    for (gt.push(n); gt.length > vt;)
                        gt.shift();
                if (Q && e.localStorage)
                    try {
                        e.localStorage[Q] = gt.join(";")
                    } catch (o) {}
            }
        }
        function w() {
            var e, t = [], i = gt, n = {};
            if (V.showPalette) {
                for (var o = 0; o < mt.length; o++)
                    for (var s = 0; s < mt[o].length; s++)
                        e = tinycolor(mt[o][s]).toRgbString(), n[e]=!0;
                for (o = 0; o < i.length; o++)
                    e = tinycolor(i[o]).toRgbString(), n.hasOwnProperty(e) || (t.push(i[o]), n[e]=!0)
            }
            return t.reverse().slice(0, V.maxSelectionSize)
        }
        function _() {
            var e = P(), i = t.map(mt, function(t, i) {
                return n(t, e, "sp-palette-row sp-palette-row-" + i)
            });
            gt && i.push(n(w(), e, "sp-palette-row sp-palette-row-selection")), Ft.html(i.join(""))
        }
        function C() {
            if (V.showInitial) {
                var e = Ut, t = P();
                Ot.html(n([e, t], t, "sp-palette-row-initial"))
            }
        }
        function k() {
            (0 === nt || 0 === it || 0 === st) && z(), kt.addClass(yt)
        }
        function x() {
            kt.removeClass(yt)
        }
        function S() {
            var e = tinycolor(Pt.val());
            e.ok ? A(e) && Z.move(e) : Pt.addClass("sp-validation-error")
        }
        function M() {
            tt ? D() : T()
        }
        function T() {
            var i = t.Event("beforeShow.spectrum");
            return tt ? (z(), void 0) : (_t.trigger(i, [P()]), Z.beforeShow(P())===!1 || i.isDefaultPrevented() || (o(), tt=!0, t(bt).bind("click.spectrum", D), t(e).bind("resize.spectrum", et), jt.addClass("sp-active"), kt.removeClass("sp-hidden"), V.autoExpand && (F() ? H() : q()), V.showPalette && _(), z(), I(), Ut = P(), C(), Z.show(Ut), _t.trigger("show.spectrum", [Ut])), void 0)
        }
        function D(i) {
            if ((!i || "click" != i.type || 2 != i.button) && tt&&!U) {
                tt=!1, t(bt).unbind("click.spectrum", D), t(e).unbind("resize.spectrum", et), jt.removeClass("sp-active"), kt.addClass("sp-hidden");
                var n=!tinycolor.equals(P(), Ut);
                n && (Xt && "cancel" !== i ? N(!0) : E()), Z.hide(P()), V.dontHideBoundElement || _t.trigger("hide.spectrum", [P()])
            }
        }
        function E() {
            A(Ut, !0)
        }
        function A(e, t) {
            if (tinycolor.equals(e, P()))
                return !1;
            var i = tinycolor(e), n = i.toHsv();
            return ut = n.h, dt = n.s, ht = n.v, pt = n.a, I(), i.ok&&!t && (Gt = Yt || i.format), !0
        }
        function P() {
            return tinycolor.fromRatio({
                h: ut,
                s: dt,
                v: ht,
                a: Math.round(100 * pt) / 100
            })
        }
        function R() {
            return !Pt.hasClass("sp-validation-error")
        }
        function F(e) {
            var t=!1, e = e || P();
            e: for (var i = 0; i < mt.length; i++)
                for (var n = 0; n < mt[i].length; n++)
                    if (t = tinycolor.equals(e, mt[i][n]))
                        break e;
            return t
        }
        function O() {
            I(), Z.move(P()), _t.trigger("move.spectrum", [P()])
        }
        function I() {
            Pt.removeClass("sp-validation-error"), W();
            var e = tinycolor({
                h: ut,
                s: "1.0",
                v: "1.0"
            });
            xt.css("background-color", e.toHexString());
            var t = Gt;
            1 > pt && ("hex" === t || "name" === t) && (t = "rgb");
            var i = P(), n = i.toHexString(), o = i.toRgbString();
            if (g || 1 === i.alpha ? qt.css("background-color", o) : (qt.css("background-color", "transparent"), qt.css("filter", i.toFilter())), V.showAlpha) {
                var s = i.toRgb();
                s.a = 0;
                var r = tinycolor(s).toRgbString(), a = "linear-gradient(left, " + r + ", " + n + ")";
                m ? Dt.css("filter", tinycolor(r).toFilter({
                    gradientType: 1
                }, n)) : (Dt.css("background", "-webkit-" + a), Dt.css("background", "-moz-" + a), Dt.css("background", "-ms-" + a), Dt.css("background", a))
            }
            V.showInput && (1 > pt && ("hex" === t || "name" === t) && (t = "rgb"), Pt.val(i.toString(t))), Lt.css("background-color", i.toString(t)), V.showPalette && _(), C()
        }
        function W() {
            var e = dt, t = ht, i = e * it, n = nt - t * nt;
            i = Math.max( - ot, Math.min(it - ot, i - ot)), n = Math.max( - ot, Math.min(nt - ot, n - ot)), St.css({
                top: n,
                left: i
            });
            var o = pt * at;
            At.css({
                left: o - lt / 2
            });
            var s = ut * st;
            Tt.css({
                top: s - ct
            })
        }
        function N(e) {
            var t = P();
            Bt && _t.val(t.toString(Gt)).change();
            var i=!tinycolor.equals(t, Ut);
            Ut = t, b(t), e && i && (Z.change(t), _t.trigger("change.spectrum", [t]))
        }
        function z() {
            it = xt.width(), nt = xt.height(), ot = St.height(), rt = Mt.width(), st = Mt.height(), ct = Tt.height(), at = Et.width(), lt = At.width(), U || kt.offset(a(kt, Ht, G, Y)), W()
        }
        function L() {
            _t.show(), Ht.unbind("click.spectrum touchstart.spectrum"), kt.remove(), jt.remove(), f[Jt.id] = null
        }
        function B(e, n) {
            return e === i ? t.extend({}, V) : n === i ? V[e] : (V[e] = n, u(), void 0)
        }
        function $() {
            Ct=!1, _t.attr("disabled", !1), Ht.removeClass("sp-disabled")
        }
        function j() {
            D(), Ct=!0, _t.attr("disabled", !0), Ht.addClass("sp-disabled")
        }
        function H() {
            Rt.hide(), Nt.show(), kt.find(".sp-palette-input-container").show().find(".sp-custom-el").toggle(!F()), z()
        }
        function q() {
            Rt.show(), kt.find(".sp-palette-input-container").hide(), z()
        }
        var V = s(l, r), U = V.flat, Y = V.containerOffsetY, G = V.containerOffsetX, X = V.clickPaletteToClose, J = V.showSelectionPalette, Q = V.localStorageKey, K = V.theme, Z = V.callbacks, et = h(z, 10), tt=!1, it = 0, nt = 0, ot = 0, st = 0, rt = 0, at = 0, lt = 0, ct = 0, ut = 0, dt = 0, ht = 0, pt = 1, ft = V.palette.slice(0), mt = t.isArray(ft[0]) ? ft : [ft], gt = V.selectionPalette.slice(0), vt = V.maxSelectionSize, yt = "sp-dragging", bt = r.ownerDocument, wt = bt.body, _t = t(r), Ct=!1, kt = t(y, bt).addClass(K), xt = kt.find(".sp-color"), St = kt.find(".sp-dragger"), Mt = kt.find(".sp-hue"), Tt = kt.find(".sp-slider"), Dt = kt.find(".sp-alpha-inner"), Et = kt.find(".sp-alpha"), At = kt.find(".sp-alpha-handle"), Pt = kt.find(".sp-input"), Rt = kt.find(".sp-picker-container"), Ft = kt.find(".sp-palette"), Ot = kt.find(".sp-initial"), It = kt.find(".sp-cancel"), Wt = kt.find(".sp-choose"), Nt = kt.find(".sp-expand"), zt = kt.find(".sp-collapse"), Lt = kt.find(".sp-custom-inner"), Bt = _t.is("input"), $t = Bt&&!U, jt = $t ? t(v).addClass(K) : t([]), Ht = $t ? jt : _t, qt = jt.find(".sp-preview-inner"), Vt = V.color || Bt && _t.val(), Ut=!1, Yt = V.preferredFormat, Gt = Yt, Xt=!V.showButtons || V.clickoutFiresChange;
        p();
        var Jt = {
            show: T,
            hide: D,
            toggle: M,
            reflow: z,
            option: B,
            enable: $,
            disable: j,
            collapse: H,
            expand: q,
            set: function(e) {
                A(e), N()
            },
            get: P,
            destroy: L,
            container: kt,
            isPaletteColor: function(e) {
                return F(e)
            }
        };
        return Jt.id = f.push(Jt) - 1, Jt
    }
    function a(e, i, n, o) {
        var s = 0, r = function() {
            var i = 0;
            return e.children(":visible").each(function() {
                i += t(this).outerWidth()
            }), i
        }(), a = e.outerHeight(), l = i.outerHeight(), c = i.outerWidth(), u = e[0].ownerDocument, d = u.documentElement, h = d.clientWidth + t(u).scrollLeft(), p = d.clientHeight + t(u).scrollTop(), f = i.offset();
        return f.top += l, f.left -= Math.min(f.left, f.left + r > h && h > r ? Math.abs(f.left + r - h - (c + f.left - h)) : 0), f.left += n, f.top -= Math.min(f.top, f.top + a > p && p > a ? Math.abs(a + l - s) : s), f.top += o, f
    }
    function l() {}
    function c(e) {
        e.stopPropagation()
    }
    function u(e, t) {
        var i = Array.prototype.slice, n = i.call(arguments, 2);
        return function() {
            return e.apply(t, n.concat(i.call(arguments)))
        }
    }
    function d(i, n, o, s) {
        function r(e) {
            e.stopPropagation && e.stopPropagation(), e.preventDefault && e.preventDefault(), e.returnValue=!1
        }
        function a(e) {
            if (d) {
                if (m && document.documentMode < 9&&!e.button)
                    return c();
                var t = e.originalEvent.touches, o = t ? t[0].pageX: e.pageX, s = t ? t[0].pageY: e.pageY, a = Math.max(0, Math.min(o - h.left, f)), l = Math.max(0, Math.min(s - h.top, p));
                g && r(e), n.apply(i, [a, l, e])
            }
        }
        function l(e) {
            var n = e.which ? 3 == e.which: 2 == e.button;
            e.originalEvent.touches, n || d || o.apply(i, arguments)!==!1 && (d=!0, p = t(i).height(), f = t(i).width(), h = t(i).offset(), t(u).bind(v), t(u.body).addClass("sp-dragging"), g || a(e), r(e))
        }
        function c() {
            d && (t(u).unbind(v), t(u.body).removeClass("sp-dragging"), s.apply(i, arguments)), d=!1
        }
        n = n || function() {}, o = o || function() {}, s = s || function() {};
        var u = i.ownerDocument || document, d=!1, h = {}, p = 0, f = 0, g = "ontouchstart"in e, v = {};
        v.selectstart = r, v.dragstart = r, v[g ? "touchmove": "mousemove"] = a, v[g ? "touchend": "mouseup"] = c, t(i).bind(g ? "touchstart" : "mousedown", l)
    }
    function h(e, t, i) {
        var n;
        return function() {
            var o = this, s = arguments, r = function() {
                n = null, e.apply(o, s)
            };
            i && clearTimeout(n), (i ||!n) && (n = setTimeout(r, t))
        }
    }
    var p = {
        beforeShow: l,
        move: l,
        change: l,
        show: l,
        hide: l,
        autoExpand: !1,
        color: !1,
        flat: !1,
        showInput: !1,
        showButtons: !0,
        containerOffsetX: 0,
        containerOffsetY: 0,
        clickoutFiresChange: !1,
        clickPaletteToClose: !0,
        showInitial: !1,
        showPalette: !1,
        showPaletteOnly: !1,
        showSelectionPalette: !0,
        dontHideBoundElement: !1,
        localStorageKey: !1,
        maxSelectionSize: 7,
        cancelText: "cancel",
        chooseText: "choose",
        preferredFormat: !1,
        id: "",
        className: "",
        showAlpha: !1,
        theme: "sp-light",
        palette: ["fff", "000"],
        selectionPalette: [],
        disabled: !1
    }, f = [], m=!!/msie/i.exec(e.navigator.userAgent), g = function() {
        function e(e, t) {
            return !!~("" + e).indexOf(t)
        }
        var t = document.createElement("div"), i = t.style;
        return i.cssText = "background-color:rgba(0,0,0,.5)", e(i.backgroundColor, "rgba") || e(i.backgroundColor, "hsla")
    }(), v = ["<div class='sp-replacer'>", "<div class='sp-preview'><div class='sp-preview-inner'></div></div>", "<div class='sp-dd'>&#9660;</div>", "</div>"].join(""), y = function() {
        var e = "";
        if (m)
            for (var t = 1; 6 >= t; t++)
                e += "<div class='sp-" + t + "'></div>";
        return ["<div class='sp-container sp-hidden'>", "<div class='sp-palette-container'>", "<div class='sp-palette sp-thumb sp-cf'></div>", "<div class='sp-palette-input-container'>", "<span class='sp-custom-el' type='text' spellcheck='false'>", "<span class='sp-custom-inner'></span>", "</span>", "<span class='sp-expand'><span class='sp-expand-inner'></span></span>", "</div>", "</div>", "<div class='sp-picker-container'>", "<div class='sp-top sp-cf'>", "<div class='sp-fill'></div>", "<div class='sp-top-inner'>", "<div class='sp-color-outer'>", "<div class='sp-color'>", "<div class='sp-sat'>", "<div class='sp-val'>", "<div class='sp-dragger'></div>", "</div>", "</div>", "</div>", "</div>", "<div class='sp-hue-outer'>", "<div class='sp-hue'>", "<div class='sp-slider'></div>", e, "</div>", "</div>", "</div>", "<div class='sp-alpha'><div class='sp-alpha-inner'><div class='sp-alpha-handle'></div></div></div>", "</div>", "<div class='sp-input-container sp-cf'>", "<span class='sp-custom-el' type='text' spellcheck='false'>", "<span class='sp-custom-inner'></span>", "</span>", "<input class='sp-input' type='text' spellcheck='false'  />", "<span class='sp-collapse'><span class='sp-collapse-inner'></span></span>", "</div>", "<div class='sp-initial sp-thumb sp-cf'></div>", "<div class='sp-button-container sp-cf'>", "<a class='sp-cancel' href='#'></a>", "<button class='sp-choose'></button>", "</div>", "</div>", "</div>"].join("")
    }(), b = "spectrum.id";
    t.fn.spectrum = function(e) {
        if ("string" == typeof e) {
            var i = this, n = Array.prototype.slice.call(arguments, 1);
            return this.each(function() {
                var o = f[t(this).data(b)];
                if (o) {
                    var s = o[e];
                    if (!s)
                        throw new Error("Spectrum: no such method: '" + e + "'");
                    "get" == e ? i = o.get() : "container" == e ? i = o.container : "option" == e ? i = o.option.apply(o, n) : "destroy" == e ? (o.destroy(), t(this).removeData(b)) : "isPaletteColor" == e ? i = o.isPaletteColor.apply(o, n) : s.apply(o, n)
                }
            }), i
        }
        return this.spectrum("destroy").each(function() {
            var i = r(this, e);
            t(this).data(b, i.id)
        })
    }, t.fn.spectrum.load=!0, t.fn.spectrum.loadOpts = {}, t.fn.spectrum.draggable = d, t.fn.spectrum.defaults = p, t.spectrum = {}, t.spectrum.localization = {}, t.spectrum.palettes = {}, t.fn.spectrum.processNativeColorInputs = function() {
        var e = t("<input type='color' value='!' />")[0], i = "color" === e.type && "!" != e.value;
        i || t("input[type=color]").spectrum({
            preferredFormat: "hex6"
        })
    }, function(e) {
        function t(e) {
            if ("object" == typeof e && e.hasOwnProperty("_tc_id"))
                return e;
            var n = i(e), s = n.r, a = n.g, c = n.b, u = S(n.a), d = n.format;
            return {
                ok: n.ok,
                format: d,
                _tc_id: b++,
                alpha: u,
                toHsv: function() {
                    var e = r(s, a, c);
                    return {
                        h: e.h,
                        s: e.s,
                        v: e.v,
                        a: u
                    }
                },
                toHsvString: function() {
                    var e = r(s, a, c), t = _(360 * e.h), i = _(100 * e.s), n = _(100 * e.v);
                    return 1 == u ? "hsv(" + t + ", " + i + "%, " + n + "%)" : "hsva(" + t + ", " + i + "%, " + n + "%, " + u + ")"
                },
                toHsl: function() {
                    var e = o(s, a, c);
                    return {
                        h: e.h,
                        s: e.s,
                        l: e.l,
                        a: u
                    }
                },
                toHslString: function() {
                    var e = o(s, a, c), t = _(360 * e.h), i = _(100 * e.s), n = _(100 * e.l);
                    return 1 == u ? "hsl(" + t + ", " + i + "%, " + n + "%)" : "hsla(" + t + ", " + i + "%, " + n + "%, " + u + ")"
                },
                toHex: function() {
                    return l(s, a, c)
                },
                toHexString: function(e) {
                    return "#" + l(s, a, c, e)
                },
                toRgb: function() {
                    return {
                        r: _(s),
                        g: _(a),
                        b: _(c),
                        a: u
                    }
                },
                toRgbString: function() {
                    return 1 == u ? "rgb(" + _(s) + ", " + _(a) + ", " + _(c) + ")" : "rgba(" + _(s) + ", " + _(a) + ", " + _(c) + ", " + u + ")"
                },
                toName: function() {
                    return T[l(s, a, c)] ||!1
                },
                toFilter: function(e, i) {
                    var n = l(s, a, c, !0), o = n, r = Math.round(255 * S(u)).toString(16), d = r, h = e && e.gradientType ? "GradientType = 1, ": "";
                    if (i) {
                        var p = t(i);
                        o = p.toHex(), d = Math.round(255 * S(p.alpha)).toString(16)
                    }
                    return "progid:DXImageTransform.Microsoft.gradient(" + h + "startColorstr=#" + m(r) + n + ",endColorstr=#" + m(d) + o + ")"
                },
                toString: function(e) {
                    e = e || this.format;
                    var t=!1;
                    return "rgb" === e && (t = this.toRgbString()), "hex" === e && (t = this.toHexString()), "hex6" === e && (t = this.toHexString(!0)), "name" === e && (t = this.toName()), "hsl" === e && (t = this.toHslString()), "hsv" === e && (t = this.toHsvString()), t || this.toHexString(!0)
                }
            }
        }
        function i(e) {
            var t = {
                r: 0,
                g: 0,
                b: 0
            }, i = 1, o=!1, r=!1;
            return "string" == typeof e && (e = g(e)), "object" == typeof e && (e.hasOwnProperty("r") && e.hasOwnProperty("g") && e.hasOwnProperty("b") ? (t = n(e.r, e.g, e.b), o=!0, r = "rgb") : e.hasOwnProperty("h") && e.hasOwnProperty("s") && e.hasOwnProperty("v") ? (t = a(e.h, e.s, e.v), o=!0, r = "hsv") : e.hasOwnProperty("h") && e.hasOwnProperty("s") && e.hasOwnProperty("l") && (t = s(e.h, e.s, e.l), o=!0, r = "hsl"), e.hasOwnProperty("a") && (i = e.a)), t.r = C(255, k(t.r, 0)), t.g = C(255, k(t.g, 0)), t.b = C(255, k(t.b, 0)), t.r < 1 && (t.r = _(t.r)), t.g < 1 && (t.g = _(t.g)), t.b < 1 && (t.b = _(t.b)), {
                ok: o,
                format: e && e.format || r,
                r: t.r,
                g: t.g,
                b: t.b,
                a: i
            }
        }
        function n(e, t, i) {
            return {
                r: 255 * u(e, 255),
                g: 255 * u(t, 255),
                b: 255 * u(i, 255)
            }
        }
        function o(e, t, i) {
            e = u(e, 255), t = u(t, 255), i = u(i, 255);
            var n, o, s = k(e, t, i), r = C(e, t, i), a = (s + r) / 2;
            if (s == r)
                n = o = 0;
            else {
                var l = s - r;
                switch (o = a > .5 ? l / (2 - s - r) : l / (s + r), s) {
                case e:
                    n = (t - i) / l + (i > t ? 6 : 0);
                    break;
                case t:
                    n = (i - e) / l + 2;
                    break;
                case i:
                    n = (e - t) / l + 4
                }
                n/=6
            }
            return {
                h: n,
                s: o,
                l: a
            }
        }
        function s(e, t, i) {
            function n(e, t, i) {
                return 0 > i && (i += 1), i > 1 && (i -= 1), 1 / 6 > i ? e + 6 * (t - e) * i : .5 > i ? t : 2 / 3 > i ? e + 6 * (t - e) * (2 / 3 - i) : e
            }
            var o, s, r;
            if (e = u(e, 360), t = u(t, 100), i = u(i, 100), 0 === t)
                o = s = r = i;
            else {
                var a = .5 > i ? i * (1 + t): i + t - i * t, l = 2 * i - a;
                o = n(l, a, e + 1 / 3), s = n(l, a, e), r = n(l, a, e - 1 / 3)
            }
            return {
                r: 255 * o,
                g: 255 * s,
                b: 255 * r
            }
        }
        function r(e, t, i) {
            e = u(e, 255), t = u(t, 255), i = u(i, 255);
            var n, o, s = k(e, t, i), r = C(e, t, i), a = s, l = s - r;
            if (o = 0 === s ? 0 : l / s, s == r)
                n = 0;
            else {
                switch (s) {
                case e:
                    n = (t - i) / l + (i > t ? 6 : 0);
                    break;
                case t:
                    n = (i - e) / l + 2;
                    break;
                case i:
                    n = (e - t) / l + 4
                }
                n/=6
            }
            return {
                h: n,
                s: o,
                v: a
            }
        }
        function a(e, t, i) {
            e = 6 * u(e, 360), t = u(t, 100), i = u(i, 100);
            var n = w.floor(e), o = e - n, s = i * (1 - t), r = i * (1 - o * t), a = i * (1 - (1 - o) * t), l = n%6, c = [i, r, s, s, a, i][l], d = [a, i, i, r, s, s][l], h = [s, s, a, i, i, r][l];
            return {
                r: 255 * c,
                g: 255 * d,
                b: 255 * h
            }
        }
        function l(e, t, i, n) {
            var o = [m(_(e).toString(16)), m(_(t).toString(16)), m(_(i).toString(16))];
            return n || o[0].charAt(0) != o[0].charAt(1) || o[1].charAt(0) != o[1].charAt(1) || o[2].charAt(0) != o[2].charAt(1) ? o.join("") : o[0].charAt(0) + o[1].charAt(0) + o[2].charAt(0)
        }
        function c(e) {
            var t = {};
            for (var i in e)
                e.hasOwnProperty(i) && (t[e[i]] = i);
            return t
        }
        function u(e, t) {
            p(e) && (e = "100%");
            var i = f(e);
            return e = C(t, k(0, S(e))), i && (e*=t / 100), w.abs(e - t) < 1e-6 ? 1 : e >= 1 ? e%t / S(t) : e
        }
        function d(e) {
            return C(1, k(0, e))
        }
        function h(e) {
            return parseInt(e, 16)
        }
        function p(e) {
            return "string" == typeof e&&-1 != e.indexOf(".") && 1 === S(e)
        }
        function f(e) {
            return "string" == typeof e&&-1 != e.indexOf("%")
        }
        function m(e) {
            return 1 == e.length ? "0" + e : "" + e
        }
        function g(e) {
            e = e.replace(v, "").replace(y, "").toLowerCase();
            var t=!1;
            if (M[e])
                e = M[e], t=!0;
            else if ("transparent" == e)
                return {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 0
                };
            var i;
            return (i = D.rgb.exec(e)) ? {
                r: i[1],
                g: i[2],
                b: i[3]
            } : (i = D.rgba.exec(e)) ? {
                r: i[1],
                g: i[2],
                b: i[3],
                a: i[4]
            } : (i = D.hsl.exec(e)) ? {
                h: i[1],
                s: i[2],
                l: i[3]
            } : (i = D.hsla.exec(e)) ? {
                h: i[1],
                s: i[2],
                l: i[3],
                a: i[4]
            } : (i = D.hsv.exec(e)) ? {
                h: i[1],
                s: i[2],
                v: i[3]
            } : (i = D.hex6.exec(e)) ? {
                r: h(i[1]),
                g: h(i[2]),
                b: h(i[3]),
                format: t ? "name": "hex"
            } : (i = D.hex3.exec(e)) ? {
                r: h(i[1] + "" + i[1]),
                g: h(i[2] + "" + i[2]),
                b: h(i[3] + "" + i[3]),
                format: t ? "name": "hex"
            } : !1
        }
        var v = /^[\s,#]+/, y = /\s+$/, b = 0, w = Math, _ = w.round, C = w.min, k = w.max, x = w.random, S = e.parseFloat;
        t.fromRatio = function(e) {
            if ("object" == typeof e)
                for (var i in e)
                    1 === e[i] && (e[i] = "1.0");
            return t(e)
        }, t.equals = function(e, i) {
            return e && i ? t(e).toRgbString() == t(i).toRgbString() : !1
        }, t.random = function() {
            return t.fromRatio({
                r: x(),
                g: x(),
                b: x()
            })
        }, t.desaturate = function(e, i) {
            var n = t(e).toHsl();
            return n.s -= (i || 10) / 100, n.s = d(n.s), t(n)
        }, t.saturate = function(e, i) {
            var n = t(e).toHsl();
            return n.s += (i || 10) / 100, n.s = d(n.s), t(n)
        }, t.greyscale = function(e) {
            return t.desaturate(e, 100)
        }, t.lighten = function(e, i) {
            var n = t(e).toHsl();
            return n.l += (i || 10) / 100, n.l = d(n.l), t(n)
        }, t.darken = function(e, i) {
            var n = t(e).toHsl();
            return n.l -= (i || 10) / 100, n.l = d(n.l), t(n)
        }, t.complement = function(e) {
            var i = t(e).toHsl();
            return i.h = (i.h + .5)%1, t(i)
        }, t.triad = function(e) {
            var i = t(e).toHsl(), n = 360 * i.h;
            return [t(e), t({
                h: (n + 120)%360,
                s: i.s,
                l: i.l
            }), t({
                h: (n + 240)%360,
                s: i.s,
                l: i.l
            })]
        }, t.tetrad = function(e) {
            var i = t(e).toHsl(), n = 360 * i.h;
            return [t(e), t({
                h: (n + 90)%360,
                s: i.s,
                l: i.l
            }), t({
                h: (n + 180)%360,
                s: i.s,
                l: i.l
            }), t({
                h: (n + 270)%360,
                s: i.s,
                l: i.l
            })]
        }, t.splitcomplement = function(e) {
            var i = t(e).toHsl(), n = 360 * i.h;
            return [t(e), t({
                h: (n + 72)%360,
                s: i.s,
                l: i.l
            }), t({
                h: (n + 216)%360,
                s: i.s,
                l: i.l
            })]
        }, t.analogous = function(e, i, n) {
            i = i || 6, n = n || 30;
            var o = t(e).toHsl(), s = 360 / n, r = [t(e)];
            for (o.h*=360, o.h = (o.h - (s * i>>1) + 720)%360; --i;)
                o.h = (o.h + s)%360, r.push(t(o));
            return r
        }, t.monochromatic = function(e, i) {
            i = i || 6;
            for (var n = t(e).toHsv(), o = n.h, s = n.s, r = n.v, a = [], l = 1 / i; i--;)
                a.push(t({
                    h: o,
                    s: s,
                    v: r
                })), r = (r + l)%1;
            return a
        }, t.readable = function(e, i) {
            var n = t(e).toRgb(), o = t(i).toRgb();
            return (o.r - n.r) * (o.r - n.r) + (o.g - n.g) * (o.g - n.g) + (o.b - n.b) * (o.b - n.b) > 10404
        };
        var M = t.names = {
            aliceblue: "f0f8ff",
            antiquewhite: "faebd7",
            aqua: "0ff",
            aquamarine: "7fffd4",
            azure: "f0ffff",
            beige: "f5f5dc",
            bisque: "ffe4c4",
            black: "000",
            blanchedalmond: "ffebcd",
            blue: "00f",
            blueviolet: "8a2be2",
            brown: "a52a2a",
            burlywood: "deb887",
            burntsienna: "ea7e5d",
            cadetblue: "5f9ea0",
            chartreuse: "7fff00",
            chocolate: "d2691e",
            coral: "ff7f50",
            cornflowerblue: "6495ed",
            cornsilk: "fff8dc",
            crimson: "dc143c",
            cyan: "0ff",
            darkblue: "00008b",
            darkcyan: "008b8b",
            darkgoldenrod: "b8860b",
            darkgray: "a9a9a9",
            darkgreen: "006400",
            darkgrey: "a9a9a9",
            darkkhaki: "bdb76b",
            darkmagenta: "8b008b",
            darkolivegreen: "556b2f",
            darkorange: "ff8c00",
            darkorchid: "9932cc",
            darkred: "8b0000",
            darksalmon: "e9967a",
            darkseagreen: "8fbc8f",
            darkslateblue: "483d8b",
            darkslategray: "2f4f4f",
            darkslategrey: "2f4f4f",
            darkturquoise: "00ced1",
            darkviolet: "9400d3",
            deeppink: "ff1493",
            deepskyblue: "00bfff",
            dimgray: "696969",
            dimgrey: "696969",
            dodgerblue: "1e90ff",
            firebrick: "b22222",
            floralwhite: "fffaf0",
            forestgreen: "228b22",
            fuchsia: "f0f",
            gainsboro: "dcdcdc",
            ghostwhite: "f8f8ff",
            gold: "ffd700",
            goldenrod: "daa520",
            gray: "808080",
            green: "008000",
            greenyellow: "adff2f",
            grey: "808080",
            honeydew: "f0fff0",
            hotpink: "ff69b4",
            indianred: "cd5c5c",
            indigo: "4b0082",
            ivory: "fffff0",
            khaki: "f0e68c",
            lavender: "e6e6fa",
            lavenderblush: "fff0f5",
            lawngreen: "7cfc00",
            lemonchiffon: "fffacd",
            lightblue: "add8e6",
            lightcoral: "f08080",
            lightcyan: "e0ffff",
            lightgoldenrodyellow: "fafad2",
            lightgray: "d3d3d3",
            lightgreen: "90ee90",
            lightgrey: "d3d3d3",
            lightpink: "ffb6c1",
            lightsalmon: "ffa07a",
            lightseagreen: "20b2aa",
            lightskyblue: "87cefa",
            lightslategray: "789",
            lightslategrey: "789",
            lightsteelblue: "b0c4de",
            lightyellow: "ffffe0",
            lime: "0f0",
            limegreen: "32cd32",
            linen: "faf0e6",
            magenta: "f0f",
            maroon: "800000",
            mediumaquamarine: "66cdaa",
            mediumblue: "0000cd",
            mediumorchid: "ba55d3",
            mediumpurple: "9370db",
            mediumseagreen: "3cb371",
            mediumslateblue: "7b68ee",
            mediumspringgreen: "00fa9a",
            mediumturquoise: "48d1cc",
            mediumvioletred: "c71585",
            midnightblue: "191970",
            mintcream: "f5fffa",
            mistyrose: "ffe4e1",
            moccasin: "ffe4b5",
            navajowhite: "ffdead",
            navy: "000080",
            oldlace: "fdf5e6",
            olive: "808000",
            olivedrab: "6b8e23",
            orange: "ffa500",
            orangered: "ff4500",
            orchid: "da70d6",
            palegoldenrod: "eee8aa",
            palegreen: "98fb98",
            paleturquoise: "afeeee",
            palevioletred: "db7093",
            papayawhip: "ffefd5",
            peachpuff: "ffdab9",
            peru: "cd853f",
            pink: "ffc0cb",
            plum: "dda0dd",
            powderblue: "b0e0e6",
            purple: "800080",
            red: "f00",
            rosybrown: "bc8f8f",
            royalblue: "4169e1",
            saddlebrown: "8b4513",
            salmon: "fa8072",
            sandybrown: "f4a460",
            seagreen: "2e8b57",
            seashell: "fff5ee",
            sienna: "a0522d",
            silver: "c0c0c0",
            skyblue: "87ceeb",
            slateblue: "6a5acd",
            slategray: "708090",
            slategrey: "708090",
            snow: "fffafa",
            springgreen: "00ff7f",
            steelblue: "4682b4",
            tan: "d2b48c",
            teal: "008080",
            thistle: "d8bfd8",
            tomato: "ff6347",
            turquoise: "40e0d0",
            violet: "ee82ee",
            wheat: "f5deb3",
            white: "fff",
            whitesmoke: "f5f5f5",
            yellow: "ff0",
            yellowgreen: "9acd32"
        }, T = t.hexNames = c(M), D = function() {
            var e = "[-\\+]?\\d+%?", t = "[-\\+]?\\d*\\.\\d+%?", i = "(?:" + t + ")|(?:" + e + ")", n = "[\\s|\\(]+(" + i + ")[,|\\s]+(" + i + ")[,|\\s]+(" + i + ")\\s*\\)?", o = "[\\s|\\(]+(" + i + ")[,|\\s]+(" + i + ")[,|\\s]+(" + i + ")[,|\\s]+(" + i + ")\\s*\\)?";
            return {
                rgb: new RegExp("rgb" + n),
                rgba: new RegExp("rgba" + o),
                hsl: new RegExp("hsl" + n),
                hsla: new RegExp("hsla" + o),
                hsv: new RegExp("hsv" + n),
                hex3: /^([0-9a-fA-F]{1})([0-9a-fA-F]{1})([0-9a-fA-F]{1})$/,
                hex6: /^([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/
            }
        }();
        e.tinycolor = t
    }(this), t(function() {
        t.fn.spectrum.load && t.fn.spectrum.processNativeColorInputs()
    }), t.ui && t.ui.dialog && t.widget("ui.dialog", t.ui.dialog, {
        _allowInteraction: function(e) {
            return t(e.target).closest(".sp-container").length?!0 : this._super(e)
        }
    })
}(window, jQuery), _wAMD.define("vendor/spectrum", function() {}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-spectrum-cell", ["backbone", "backgrid", "vendor/spectrum"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    var i = e.$, n = t.Extension.SpectrumCellEditor = t.InputCellEditor.extend({
        tagName: "div",
        events: {
            click: "handleClick",
            "blur .color-name": "openOrSave",
            "keydown .color-name": "saveOrCancel"
        },
        attributes: {},
        initialize: function() {
            this.colorSwatch = i("<span/>", {
                "class": "color-swatch"
            }), this.colorName = i("<input/>", {
                "class": "color-name",
                name: "color-name",
                type: "text",
                value: ""
            }), this.$el.append(this.colorSwatch, this.colorName), this.namespace = ".spectrum-cell-" + Math.floor(10000001 * Math.random()), i(document).on("keydown" + this.namespace, _.bind(this.handleDocumentKeydown, this)), t.InputCellEditor.prototype.initialize.apply(this, arguments)
        },
        render: function() {
            var e = this.formatter.fromRaw(this.model.get(this.column.get("name"))), t = this;
            return this.colorName.val(e.name), this.colorSwatch.css("background-color", e.value), this.colorSwatch.spectrum({
                color: e.value,
                containerOffsetY: 5,
                containerOffsetX: - 5,
                showInput: !0,
                showPalette: !0,
                chooseText: "Ok",
                theme: "spectrum-cell-picker sp-w",
                clickoutFiresChange: !0,
                clickPaletteToClose: !1,
                preferredFormat: "hex6",
                showSelectionPalette: !1,
                dontHideBoundElement: !0,
                palette: [["#d5d5d5", "#a1a1a1", "#818181", "#626262", "#515151", "#3f3f3f", "#2a2a2a"], ["#f8a9a9", "#e97676", "#e05c5c", "#da4444", "#c23b3b", "#a82e2e", "#8d2424"], ["#f8c7a9", "#e9a476", "#e0915c", "#da8044", "#c2743b", "#a85f2e", "#8d5024"], ["#f8eaa9", "#e9cf76", "#e0bf5c", "#dab844", "#c2a43b", "#a88d2e", "#8d7824"], ["#c9f8a9", "#a9e976", "#91e05c", "#81c94c", "#6cb83a", "#5fa233", "#508d24"], ["#a9f8dd", "#76e9c7", "#5ce0b6", "#4cc9a4", "#3ab890", "#33a27f", "#248d6c"], ["#a9e4f8", "#76cae9", "#5cbce0", "#4caac9", "#3a96b8", "#3387a2", "#24678d"], ["#b5aaf6", "#9789e8", "#8273da", "#7161d0", "#6555c2", "#5848b7", "#5040ae"], ["#d8aaf6", "#c389e8", "#b073da", "#a561d0", "#9555c2", "#8c48b7", "#8640ae"], ["#f6aaf0", "#e889e0", "#da73d2", "#d061c7", "#c255b9", "#b748ae", "#ae40a5"]],
                beforeShow: function() {
                    t.colorSwatch.spectrum("collapse")
                },
                show: function() {
                    t.renderOverlay()
                },
                hide: function() {
                    t.destroyOverlay()
                },
                move: function(e) {
                    t.colorSwatch.css("background-color", e.toHexString())
                },
                change: function(e) {
                    t.colorSwatch.css("background-color", e.toHexString())
                }
            }), this
        },
        renderOverlay: function() {
            this.overlay = i("<div/>", {
                id: "spectrum-cell-overlay"
            }), this.overlay.on("click" + this.namespace, _.bind(function(e) {
                e.stopPropagation(), this.refocus()
            }, this)), i(document.body).append(this.overlay)
        },
        destroyOverlay: function() {
            this.overlay && this.overlay.off().remove()
        },
        refocus: function() {
            this.colorSwatch.spectrum("hide"), this.colorName.select()
        },
        postRender: function() {
            this.colorName.select()
        },
        setCell: function(e) {
            this.cell = e
        },
        openOrSave: function(e) {
            this.colorSwatch.is(":hover") || this.saveOrCancel(e)
        },
        saveOrCancel: function(e) {
            var i = new t.Command(e), n = i.moveUp() || i.moveDown() || i.moveLeft() || i.moveRight() || i.save(), o = "blur" === e.type || "focusout" === e.type;
            n || o ? this.save(e) : i.cancel() && this.cancel(e)
        },
        cancel: function(e) {
            var i = new t.Command(e), n = this.model, o = this.column;
            e.stopPropagation(), n.trigger("backgrid:edited", n, o, i)
        },
        save: function(e) {
            var i, n, o, s = this.model, r = this.column, a = this.formatter, l = new t.Command(e);
            e.preventDefault(), e.stopPropagation(), i = this.colorName.val(), n = this.colorSwatch.spectrum("get").toHexString(), o = a.toRaw(i, n), _.isUndefined(o) ? s.trigger("backgrid:error", s, r, i, n) : (s.set(r.get("name"), o), s.trigger("backgrid:edited", s, r, l))
        },
        remove: function() {
            i(document).off(this.namespace), this.colorSwatch.spectrum("destroy"), e.View.prototype.remove.apply(this, arguments)
        },
        handleClick: function(e) {
            e.stopPropagation(), this.colorSwatch.spectrum("hide")
        },
        handleDocumentClick: function() {
            this.saveOrCancel(i.Event("blur"))
        },
        handleDocumentKeydown: function(e) {
            var t = e.keyCode || e.which;
            (9 === t || 27 === t) && (e.preventDefault(), this.refocus())
        }
    });
    t.Extension.SpectrumCell = t.Cell.extend({
        className: "color-picker-cell",
        editor: n,
        formatter: {
            fromRaw: function(e) {
                var t = {}, i = [];
                return _.isNull(e) || _.isUndefined(e) ? t = {
                    name: "—",
                    value: "#000000"
                } : (i = e.match(/(.*)<(.*)>$/), i && i.length ? (t.name = i[1], t.value = i[2]) : (t.name = e, t.value = "#000000")), t
            },
            toRaw: function(e, t) {
                return "" === t && "" === e ? void 0 : e + "<" + t + ">"
            }
        },
        initialize: function() {
            t.Cell.prototype.initialize.apply(this, arguments), this.listenTo(this.model, "backgrid:edit", function(e, t, i, n) {
                t.get("name") == this.column.get("name") && n.setCell(i)
            })
        },
        render: function() {
            var e = this.formatter.fromRaw(this.model.get(this.column.get("name"))), t = i("<span/>", {
                "class": "color-swatch",
                style: "background-color: " + e.value
            }), n = i("<span/>", {
                "class": "color-name",
                text: e.name
            });
            return this.$el.empty().append(t, n), this.delegateEvents(), this
        }
    })
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-placeholder-cell", ["backbone", "backgrid"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    var i = e.$;
    t.Extension.PlaceholderCell = t.Cell.extend({
        className: "placeholder-cell",
        showPlaceholder: !0,
        placeholderTemplate: '<div class="cell-placeholder"></div>',
        emptyText: "",
        render: function() {
            if (t.Cell.prototype.render.apply(this), this.showPlaceholder) {
                var e = this.$el.text() === this.emptyText, n = i(this.placeholderTemplate);
                e ? (this.$el.html(""), n.show()) : n.hide(), this.$el.append(n)
            }
            return this
        }
    })
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-typeahead-cell", ["backbone", "backgrid", "components/backgrid/backgrid-placeholder-cell"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    var i = e.$, n = {
        suggestionLimit: 8,
        minimumInputLength: 0,
        showPlaceholder: !0
    }, o = {
        TAB: 9,
        ENTER: 13,
        ESC: 27,
        LEFT: 37,
        UP: 38,
        RIGHT: 39,
        DOWN: 40,
        PAGE_UP: 33,
        PAGE_DOWN: 34,
        HOME: 36,
        END: 35,
        BACKSPACE: 8
    }, s = function(e) {
        e.preventDefault(), e.stopPropagation()
    }, r = function(e) {
        var t, n, o = e, s = function(e) {
            return "" + e.text
        };
        i.isArray(o) && (n = o, o = {
            results: n
        }), i.isFunction(o)===!1 && (n = o, o = function() {
            return n
        });
        var r = o();
        return r.text && (s = r.text, i.isFunction(s) || (t = r.text, s = function(e) {
            return e[t]
        })), function(e) {
            e = i.extend({}, {
                matcher: function(e, t) {
                    return 0 === ("" + t).toUpperCase().indexOf(("" + e).toUpperCase())
                }
            }, e);
            var t, n = e.term, r = {
                results: []
            };
            return "" === n ? (e.callback(o()), void 0) : (t = function(o, r) {
                var a, l;
                if (o = o[0], o.children) {
                    a = {};
                    for (l in o)
                        o.hasOwnProperty(l) && (a[l] = o[l]);
                    a.children = [], i(o.children).each2(function(e, i) {
                        t(i, a.children)
                    }), (a.children.length || e.matcher(n, s(a), o)) && r.push(a)
                } else 
                    e.matcher(n, s(o), o) && r.push(o)
            }, i(o().results).each2(function(e, i) {
                t(i, r.results)
            }), n = n.toUpperCase(), r.results = _.sortBy(r.results, function(e) {
                return e.text.toUpperCase().indexOf(n)
            }), e.callback(r), void 0)
        }
    }, a = t.Extension.TypaheadCellEditor = t.CellEditor.extend({
        tagName: "input",
        attributes: {
            type: "text",
            "class": "typeahead-cell-input"
        },
        events: {
            keydown: "onKeydown",
            keyup: "onKeyup",
            blur: "selectOrSave"
        },
        options: {},
        initialize: function() {
            this.hint = this.createHintContainer(), this.results = this.createResultsContainer(), this.resultsChoices = this.results.children(".typeahead-cell-choices"), this.results.on("click", ".typeahead-cell-result", _.bind(this.clickSelectChoice, this)).on("hover", ".typeahead-cell-result", _.bind(this.onHover, this)), t.CellEditor.prototype.initialize.apply(this, arguments)
        },
        render: function() {
            this.searcher = r(this.options.data), this.$el.after(this.results), this.$el.after(this.hint), this.$el.val(this.formatter.fromRaw(this.model.get(this.column.get("name")))), this.$el.select(), t.CellEditor.prototype.render.apply(this, arguments)
        },
        postRender: function() {
            return this.positionResults(), this.positionHint(), this.$el.focus(), this
        },
        setOptions: function(e) {
            this.options = _.extend({}, n, e)
        },
        save: function(e) {
            var t = this.formatter.toRaw(this.$el.val());
            _.isUndefined(t) ? this.model.trigger("backgrid:error", this.model, this.column, t) : (this.addOption(t), this.model.set(this.column.get("name"), t), this.closeCell(e))
        },
        addOption: function(e) {
            var t = [];
            t = _.map(this.options.data, function(e) {
                return e.text
            }), - 1 === _.indexOf(t, e) && this.options.data.push({
                id: t.length,
                text: e
            })
        },
        closeCell: function(e) {
            this.model.trigger("backgrid:edited", this.model, this.column, new t.Command(e))
        },
        remove: function() {
            return t.CellEditor.prototype.remove.apply(this, arguments)
        },
        clickSelectChoice: function(e) {
            var t = this.resultsChoices.children().index(e.currentTarget);
            this.highlightSet(t), this.$el.val(e.currentTarget.getAttribute("data-result")), this.closeResults(), this.$el.focus(), s(e)
        },
        onHover: function(e) {
            var t = this.resultsChoices.children().index(e.target);
            this.highlight(t)
        },
        onKeydown: function(e) {
            switch (this.ignoreKeyUp=!1, e.which) {
            case o.UP:
            case o.DOWN:
                this.shouldSave(e), s(e);
                break;
            case o.ENTER:
            case o.TAB:
                this.completeOrSave(e), this.ignoreKeyUp=!0, s(e)
            }
        },
        onKeyup: function(e) {
            if (this.ignoreKeyUp===!1)
                switch (e.which) {
                case o.UP:
                case o.DOWN:
                    this.moveHighlight(e.which === o.UP?-1 : 1), s(e);
                    break;
                case o.ENTER:
                    this.closeOrSave(e);
                    break;
                case o.ESC:
                    this.closeCell(e);
                    break;
                default:
                    e.metaKey || this.updateResults()
                }
            this.ignoreKeyUp=!1
        },
        selectOrSave: function(e) {
            this.isOpened() && this.resultsChoices.is(":hover") ? s(e) : this.save(e)
        },
        completeOrSave: function(e) {
            if (this.isOpened()) {
                var t = this.getHighlightIndex();
                this.highlightSet(t), this.closeResults()
            } else 
                this.save(e)
        },
        closeOrSave: function(e) {
            var t = this.getHighlight();
            0 === t.length ? this.save(e) : this.closeResults()
        },
        shouldSave: function(e) {
            this.isOpened() || this.save(e)
        },
        moveHighlight: function(e) {
            var t = (this.getHighlight(), this.getHighlightIndex());
            this.setPreviousValue(), this.highlightSet(t + e)
        },
        getHighlight: function() {
            return this.resultsChoices.children(".typeahead-cell-selected")
        },
        getHighlightIndex: function() {
            var e = this.getHighlight();
            return e ? this.resultsChoices.children().index(e) : 0
        },
        clearHighlight: function() {
            this.getHighlight().removeClass("typeahead-cell-selected")
        },
        highlight: function(e) {
            var t = this.resultsChoices.children();
            this.clearHighlight(), t.eq(e).addClass("typeahead-cell-selected")
        },
        highlightSet: function(e) {
            var t, i = this.resultsChoices.children();
            if (0 !== arguments.length && 0 !== i.length) {
                if (e >= i.length && (e = i.length - 1), 0 > e)
                    return this.$el.val(this.previousValue), this.updateResults(), this.clearHighlight(), this.clearPreviousValue(), void 0;
                this.clearHint(), this.clearHighlight(), t = i.eq(e).addClass("typeahead-cell-selected"), this.$el.val(t.attr("data-result"))
            }
        },
        isOpened: function() {
            return this.results.hasClass("typeahead-cell-opened")
        },
        openResults: function() {
            this.results.addClass("typeahead-cell-opened")
        },
        closeResults: function() {
            this.results.removeClass("typeahead-cell-opened"), this.clearHighlight(), this.clearHint()
        },
        updateResults: function() {
            var e = this, t = this.$el.val();
            t.length > this.options.minimumInputLength ? this.searcher({
                term: t,
                callback: function(t) {
                    t = i.map(t.results, function(e) {
                        return e.text
                    }), e.populateHint(t[0]), e.populateResults(t)
                }
            }) : (this.closeResults(), this.clearHint())
        },
        positionResults: function() {
            var e = this.$el.outerWidth();
            this.results.css({
                width: e
            })
        },
        positionHint: function() {
            var e = this.$el.outerWidth(), t = this.$el.position(), i = t.top, n = t.left;
            this.hint.css({
                width: e,
                top: i,
                left: n
            })
        },
        clearHint: function() {
            this.hint.val("")
        },
        setPreviousValue: function() {
            this.previousValue || (this.previousValue = this.$el.val())
        },
        clearPreviousValue: function() {
            this.previousValue = null
        },
        populateHint: function(e) {
            if (!e)
                return this.clearHint(), void 0;
            var t = this.$el.val();
            0 === e.toUpperCase().indexOf(t.toUpperCase()) ? (e = e.replace(new RegExp("^(.{0}).{" + t.length + "}"), "$1" + t), this.hint.val(e)) : this.clearHint()
        },
        populateResults: function(e) {
            var t, n = (e.length > this.options.suggestionLimit ? 8 : e.length, this.$el.val());
            if (this.openResults(), this.resultsChoices.empty(), e[0] && e[0].length === n.length)
                return this.closeResults(), void 0;
            if (0 !== e.length)
                for (var o = 0; o < e.length; o++)
                    t = i(document.createElement("li")), t.attr({
                        "class": "typeahead-cell-result",
                        "data-result": e[o]
                    }), t.text(e[o]), this.resultsChoices.append(t);
            if (0 === e.length || e[0].length !== n.length) {
                var s = i(document.createElement("label"));
                s.attr({
                    "class": "typeahead-cell-new"
                }), s.text(_W.tl("New Choice")), t = i(document.createElement("li")), t.attr({
                    "class": "typeahead-cell-result",
                    "data-result": n
                }), t.text(n), t.prepend(s), this.resultsChoices.append(t)
            }
            this.highlight(0)
        },
        createResultsContainer: function() {
            var e = i(document.createElement("div")).attr({
                "class": "typeahead-cell-results"
            }).html(['<ul class="typeahead-cell-choices">', "</ul>"].join(""));
            return e
        },
        createHintContainer: function() {
            var e = i(document.createElement("input")).attr({
                type: "text",
                "class": "typeahead-cell-hint",
                autocomplete: "off",
                spellcheck: "off",
                disabled: "disabled"
            });
            return e
        }
    });
    t.Extension.TypeaheadCell = t.Extension.PlaceholderCell.extend({
        className: "typeahead-cell",
        editor: a,
        formatter: {
            fromRaw: function(e) {
                return _.isNull(e) || _.isUndefined(e) ? "" : e
            },
            toRaw: function(e) {
                return "" === e ? void 0 : e
            }
        },
        placeholderTemplate: '<div class="typeahead-cell-placeholder cell-placeholder"></div>',
        initialize: function(e) {
            t.Cell.prototype.initialize.apply(this, arguments), this.datalist = e.datalist || this.datalist, this.listenTo(this.model, "backgrid:edit", function(e, t, i, n) {
                t.get("name") == this.column.get("name") && n.setOptions(this.options)
            })
        }
    })
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-html-cell", ["backbone", "backgrid"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    e.$, t.Extension.HtmlCell = t.Cell.extend({
        className: "html-cell",
        initialize: function() {
            t.Cell.prototype.initialize.apply(this, arguments)
        },
        render: function() {
            this.$el.empty();
            var e = this.model.get(this.column.get("name")), t = this.formatter.fromRaw(e, this.model);
            return this.$el.append(t), this.delegateEvents(), this
        }
    })
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-validate-formatter", ["backbone", "backgrid"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    var i = String.prototype.trim ? function(e) {
        return null === e ? "" : String.prototype.trim.call(e)
    }
    : function(e) {
        var t = /^\s+/, i = /\s+$/;
        return null === e ? "" : e.toString().replace(t, "").replace(i, "")
    }, n = function(e) {
        return _.isNumber(e) || _.isString(e) && e.match(s.number)
    }, o = function(e) {
        return !(_.isNull(e) || _.isUndefined(e) || _.isString(e) && "" === i(e) || _.isArray(e) && _.isEmpty(e))
    }, s = {
        digits: /^\d+$/,
        number: /^-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$/,
        email: /^((([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+(\.([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+)*)|((\x22)((((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(([\x01-\x08\x0b\x0c\x0e-\x1f\x7f]|\x21|[\x23-\x5b]|[\x5d-\x7e]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(\\([\x01-\x09\x0b\x0c\x0d-\x7f]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]))))*(((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(\x22)))@((([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))$/i,
        url: /^(https?|ftp):\/\/(((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:)*@)?(((\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]))|((([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?)(:\d*)?)(\/((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)+(\/(([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)*)*)?)?(\?((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|[\uE000-\uF8FF]|\/|\?)*)?(\#((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|\/|\?)*)?$/i
    }, r = {
        required: function(e, t, i) {
            var n = _.isFunction(t) ? t.call(e, t, i): t;
            return n&&!o(e)?!1 : void 0
        },
        acceptance: function(e) {
            return "true" === e || _.isBoolean(e) && e!==!1 ? void 0 : !1
        },
        min: function(e, t) {
            return e = parseInt(e), t = parseInt(t), !n(e) || t > e?!1 : void 0
        },
        max: function(e, t) {
            return e = parseInt(e), t = parseInt(t), !n(e) || e > t?!1 : void 0
        },
        range: function(e, t) {
            return !n(e) || e < t[0] || e > t[1]?!1 : void 0
        },
        length: function(e, t) {
            return o(e) && i(e).length === t ? void 0 : !1
        },
        minLength: function(e, t) {
            return !o(e) || i(e).length < t?!1 : void 0
        },
        maxLength: function(e, t) {
            return !o(e) || i(e).length > t?!1 : void 0
        },
        rangeLength: function(e, t) {
            return !o(e) || i(e).length < t[0] || i(e).length > t[1]?!1 : void 0
        },
        oneOf: function(e, t) {
            return _.include(t, e) ? void 0 : !1
        },
        equalTo: function(e, t, i) {
            return e !== i[t]?!1 : void 0
        },
        pattern: function(e, t) {
            return o(e) && e.toString().match(s[t] || t) ? void 0 : !1
        }
    }, a = {
        fromRaw: function(e) {
            return e
        },
        toRaw: function(e) {
            return e
        }
    };
    t.Extension.ValidateFormatter = function(e) {
        var e = e || {}, t = e.validator || {}, i = e.formatter || a;
        return {
            fromRaw: function(e) {
                return i.fromRaw(e)
            },
            toRaw: function(e) {
                var n=!0;
                return _.each(t, function(t, i) {
                    n!==!1 && (n = r[i](e, t)===!1?!1 : !0)
                }), n ? i.toRaw(e) : void 0
            }
        }
    }
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-emptytext-numberformatter", ["backbone", "backgrid"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    e.$;
    var i = t.Extension.EmptyTextNumberFormatter = function(e) {
        if (e = e ? _.clone(e) : {}, _.extend(this, this.defaults, e), this.decimals < 0 || this.decimals > 20)
            throw new RangeError("decimals must be between 0 and 20")
    };
    i.prototype = new t.NumberFormatter, _.extend(i.prototype, {
        emptyText: "—",
        fromRaw: function(e) {
            if (_.isNull(e) || _.isUndefined(e))
                return this.emptyText;
            e = e.toFixed(~~this.decimals);
            var t = e.split("."), i = t[0], n = t[1] ? (this.decimalSeparator || ".") + t[1]: "";
            return i.replace(this.HUMANIZED_NUM_RE, "$1" + this.orderSeparator) + n
        },
        toRaw: function(e) {
            var t = "";
            if (e === this.emptyText || "" === e)
                return null;
            for (var i = e.trim().split(this.orderSeparator), n = 0; n < i.length; n++)
                t += i[n];
            var o = t.split(this.decimalSeparator);
            t = "";
            for (var n = 0; n < o.length; n++)
                t = t + o[n] + ".";
            "." === t[t.length - 1] && (t = t.slice(0, t.length - 1));
            var s = 1 * (1 * t).toFixed(~~this.decimals);
            return _.isNumber(s)&&!_.isNaN(s) ? s : void 0
        }
    })
}), function(e) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("components/backgrid/backgrid-emptytext-stringformatter", ["backbone", "backgrid"], e) : "undefined" != typeof Backbone && "undefined" != typeof Backgrid && e(Backbone, Backgrid)
}(function(e, t) {
    e.$;
    var i = t.Extension.EmptyTextStringFormatter = {};
    _.extend(i, {
        emptyText: "—",
        fromRaw: function(e) {
            return _.isNull(e) || _.isUndefined(e) ? this.emptyText : e
        },
        toRaw: function(e) {
            return e === this.emptyText || "" === e ? null : e
        }
    })
}), _wAMD.define("backgrid-all", ["backgrid", "backgrid-filter", "backgrid-paginator", "components/backgrid/backgrid-affix-cell", "components/backgrid/backgrid-virtual-cell", "components/backgrid/backgrid-spectrum-cell", "components/backgrid/backgrid-typeahead-cell", "components/backgrid/backgrid-html-cell", "components/backgrid/backgrid-validate-formatter", "components/backgrid/backgrid-emptytext-numberformatter", "components/backgrid/backgrid-emptytext-stringformatter"], function(e) {
    return e
}), function(e, t) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("backbone-relational", ["exports", "backbone", "underscore"], t) : "undefined" != typeof exports ? t(exports, require("backbone"), require("underscore")) : t(e, e.Backbone, e._)
}(this, function(e, t, i) {
    t.Relational = {
        showWarnings: !0
    }, t.Semaphore = {
        _class: "Backbone.Semaphore",
        _permitsAvailable: null,
        _permitsUsed: 0,
        acquire: function() {
            if (this._permitsAvailable && this._permitsUsed >= this._permitsAvailable)
                throw new Error("Max permits acquired");
            this._permitsUsed++
        },
        release: function() {
            if (0 === this._permitsUsed)
                throw new Error("All permits released");
            this._permitsUsed--
        },
        isLocked: function() {
            return this._permitsUsed > 0
        },
        setAvailablePermits: function(e) {
            if (this._permitsUsed > e)
                throw new Error("Available permits cannot be less than used permits");
            this._permitsAvailable = e
        }
    }, t.BlockingQueue = function() {
        this._queue = []
    }, i.extend(t.BlockingQueue.prototype, t.Semaphore, {
        _class: "Backbone.BlockingQueue",
        _queue: null,
        add: function(e) {
            this.isBlocked() ? this._queue.push(e) : e()
        },
        process: function() {
            var e = this._queue;
            for (this._queue = []; e && e.length;)
                e.shift()()
        },
        block: function() {
            this.acquire()
        },
        unblock: function() {
            this.release(), this.isBlocked() || this.process()
        },
        isBlocked: function() {
            return this.isLocked()
        }
    }), t.Relational.eventQueue = new t.BlockingQueue, t.Store = function() {
        this._collections = [], this._reverseRelations = [], this._orphanRelations = [], this._subModels = [], this._modelScopes = [e]
    }, i.extend(t.Store.prototype, t.Events, {
        _class: "Backbone.Store",
        initializeRelation: function(e, n, o) {
            var s = i.isString(n.type) ? t[n.type] || this.getObjectByName(n.type): n.type;
            s && s.prototype instanceof t.Relation ? new s(e, n, o) : t.Relational.showWarnings && "undefined" != typeof console && console.warn("Relation=%o; missing or invalid relation type!", n)
        },
        addModelScope: function(e) {
            this._modelScopes.push(e)
        },
        removeModelScope: function(e) {
            this._modelScopes = i.without(this._modelScopes, e)
        },
        addSubModels: function(e, t) {
            this._subModels.push({
                superModelType: t,
                subModels: e
            })
        },
        setupSuperModel: function(e) {
            i.find(this._subModels, function(t) {
                return i.each(t.subModels || [], function(i, n) {
                    var o = this.getObjectByName(i);
                    return e === o ? (t.superModelType._subModels[n] = e, e._superModel = t.superModelType, e._subModelTypeValue = n, e._subModelTypeAttribute = t.superModelType.prototype.subModelTypeAttribute, !0) : void 0
                }, this)
            }, this)
        },
        addReverseRelation: function(e) {
            var t = i.any(this._reverseRelations, function(t) {
                return i.all(e || [], function(e, i) {
                    return e === t[i]
                })
            });
            !t && e.model && e.type && (this._reverseRelations.push(e), this._addRelation(e.model, e), this.retroFitRelation(e))
        },
        addOrphanRelation: function(e) {
            var t = i.any(this._orphanRelations, function(t) {
                return i.all(e || [], function(e, i) {
                    return e === t[i]
                })
            });
            !t && e.model && e.type && this._orphanRelations.push(e)
        },
        processOrphanRelations: function() {
            i.each(this._orphanRelations.slice(0), function(e) {
                var n = t.Relational.store.getObjectByName(e.relatedModel);
                n && (this.initializeRelation(null, e), this._orphanRelations = i.without(this._orphanRelations, e))
            }, this)
        },
        _addRelation: function(e, t) {
            e.prototype.relations || (e.prototype.relations = []), e.prototype.relations.push(t), i.each(e._subModels || [], function(e) {
                this._addRelation(e, t)
            }, this)
        },
        retroFitRelation: function(e) {
            var t = this.getCollection(e.model, !1);
            t && t.each(function(t) {
                t instanceof e.model && new e.type(t, e)
            }, this)
        },
        getCollection: function(e, n) {
            e instanceof t.RelationalModel && (e = e.constructor);
            for (var o = e; o._superModel;)
                o = o._superModel;
            var s = i.find(this._collections, function(e) {
                return e.model === o
            });
            return s || n===!1 || (s = this._createCollection(o)), s
        },
        getObjectByName: function(e) {
            var t = e.split("."), n = null;
            return i.find(this._modelScopes, function(e) {
                return n = i.reduce(t || [], function(e, t) {
                    return e ? e[t] : void 0
                }, e), n && n !== e?!0 : void 0
            }, this), n
        },
        _createCollection: function(e) {
            var i;
            return e instanceof t.RelationalModel && (e = e.constructor), e.prototype instanceof t.RelationalModel && (i = new t.Collection, i.model = e, this._collections.push(i)), i
        },
        resolveIdForItem: function(e, n) {
            var o = i.isString(n) || i.isNumber(n) ? n: null;
            return null === o && (n instanceof t.RelationalModel ? o = n.id : i.isObject(n) && (o = e.prototype._getNewId(n))), o || 0 === o || (o = null), o
        },
        find: function(e, t) {
            var i = this.resolveIdForItem(e, t), n = this.getCollection(e);
            if (n) {
                var o = n.get(i);
                if (o instanceof e)
                    return o
            }
            return null
        },
        register: function(e) {
            var t = this.getCollection(e);
            if (t) {
                var i = e.collection;
                t.add(e), e.collection = i
            }
        },
        checkId: function(e, i) {
            var n = this.getCollection(e), o = n && n.get(i);
            if (o && e !== o)
                throw t.Relational.showWarnings && "undefined" != typeof console && console.warn("Duplicate id! Old RelationalModel=%o, new RelationalModel=%o", o, e), new Error("Cannot instantiate more than one Backbone.RelationalModel with the same id per type!")
        },
        update: function(e) {
            var t = this.getCollection(e);
            t.contains(e) || this.register(e), t._onModelEvent("change:" + e.idAttribute, e, t), e.trigger("relational:change:id", e, t)
        },
        unregister: function(e) {
            var n, o;
            e instanceof t.Model ? (n = this.getCollection(e), o = [e]) : e instanceof t.Collection ? (n = this.getCollection(e.model), o = i.clone(e.models)) : (n = this.getCollection(e), o = i.clone(n.models)), i.each(o, function(e) {
                this.stopListening(e), i.invoke(e.getRelations(), "stopListening")
            }, this), i.contains(this._collections, e) ? n.reset([]) : i.each(o, function(e) {
                n.get(e) ? n.remove(e) : n.trigger("relational:remove", e, n)
            }, this)
        },
        reset: function() {
            this.stopListening(), i.each(this._collections, function(e) {
                this.unregister(e)
            }, this), this._collections = [], this._subModels = [], this._modelScopes = [e]
        }
    }), t.Relational.store = new t.Store, t.Relation = function(e, n, o) {
        if (this.instance = e, n = i.isObject(n) ? n : {}, this.reverseRelation = i.defaults(n.reverseRelation || {}, this.options.reverseRelation), this.options = i.defaults(n, this.options, t.Relation.prototype.options), this.reverseRelation.type = i.isString(this.reverseRelation.type) ? t[this.reverseRelation.type] || t.Relational.store.getObjectByName(this.reverseRelation.type) : this.reverseRelation.type, this.key = this.options.key, this.keySource = this.options.keySource || this.key, this.keyDestination = this.options.keyDestination || this.keySource || this.key, this.model = this.options.model || this.instance.constructor, this.relatedModel = this.options.relatedModel, !i.isFunction(this.relatedModel) || this.relatedModel.prototype instanceof t.RelationalModel || (this.relatedModel = i.result(this, "relatedModel")), i.isString(this.relatedModel) && (this.relatedModel = t.Relational.store.getObjectByName(this.relatedModel)), this.checkPreconditions() && (!this.options.isAutoRelation && this.reverseRelation.type && this.reverseRelation.key && t.Relational.store.addReverseRelation(i.defaults({
            isAutoRelation: !0,
            model: this.relatedModel,
            relatedModel: this.model,
            reverseRelation: this.options
        }, this.reverseRelation)), e)) {
            var s = this.keySource;
            s !== this.key && "object" == typeof this.instance.get(this.key) && (s = this.key), this.setKeyContents(this.instance.get(s)), this.relatedCollection = t.Relational.store.getCollection(this.relatedModel), this.keySource !== this.key && delete this.instance.attributes[this.keySource], this.instance._relations[this.key] = this, this.initialize(o), this.options.autoFetch && this.instance.fetchRelated(this.key, i.isObject(this.options.autoFetch) ? this.options.autoFetch : {}), this.listenTo(this.instance, "destroy", this.destroy).listenTo(this.relatedCollection, "relational:add relational:change:id", this.tryAddRelated).listenTo(this.relatedCollection, "relational:remove", this.removeRelated)
        }
    }, t.Relation.extend = t.Model.extend, i.extend(t.Relation.prototype, t.Events, t.Semaphore, {
        _class: "Backbone.Relation",
        options: {
            createModels: !0,
            includeInJSON: !0,
            isAutoRelation: !1,
            autoFetch: !1,
            parse: !1
        },
        instance: null,
        key: null,
        keyContents: null,
        relatedModel: null,
        relatedCollection: null,
        reverseRelation: null,
        related: null,
        checkPreconditions: function() {
            var e = this.instance, n = this.key, o = this.model, s = this.relatedModel, r = t.Relational.showWarnings && "undefined" != typeof console;
            if (!o ||!n ||!s)
                return r && console.warn("Relation=%o: missing model, key or relatedModel (%o, %o, %o).", this, o, n, s), !1;
            if (!(o.prototype instanceof t.RelationalModel))
                return r && console.warn("Relation=%o: model does not inherit from Backbone.RelationalModel (%o).", this, e), !1;
            if (!(s.prototype instanceof t.RelationalModel))
                return r && console.warn("Relation=%o: relatedModel does not inherit from Backbone.RelationalModel (%o).", this, s), !1;
            if (this instanceof t.HasMany && this.reverseRelation.type === t.HasMany)
                return r && console.warn("Relation=%o: relation is a HasMany, and the reverseRelation is HasMany as well.", this), !1;
            if (e && i.keys(e._relations).length) {
                var a = i.find(e._relations, function(e) {
                    return e.key === n
                }, this);
                if (a)
                    return r && console.warn("Cannot create relation=%o on %o for model=%o: already taken by relation=%o.", this, n, e, a), !1
            }
            return !0
        },
        setRelated: function(e) {
            this.related = e, this.instance.attributes[this.key] = e
        },
        _isReverseRelation: function(e) {
            return e.instance instanceof this.relatedModel && this.reverseRelation.key === e.key && this.key === e.reverseRelation.key
        },
        getReverseRelations: function(e) {
            var t = [], n = i.isUndefined(e) ? this.related && (this.related.models || [this.related]): [e];
            return i.each(n || [], function(e) {
                i.each(e.getRelations() || [], function(e) {
                    this._isReverseRelation(e) && t.push(e)
                }, this)
            }, this), t
        },
        destroy: function() {
            this.stopListening(), this instanceof t.HasOne ? this.setRelated(null) : this instanceof t.HasMany && this.setRelated(this._prepareCollection()), i.each(this.getReverseRelations(), function(e) {
                e.removeRelated(this.instance)
            }, this)
        }
    }), t.HasOne = t.Relation.extend({
        _class: "Backbone.HasOne",
        options: {
            reverseRelation: {
                type: "HasMany"
            }
        },
        initialize: function(e) {
            this.listenTo(this.instance, "relational:change:" + this.key, this.onChange);
            var t = this.findRelated(e);
            this.setRelated(t), i.each(this.getReverseRelations(), function(t) {
                t.addRelated(this.instance, e)
            }, this)
        },
        findRelated: function(e) {
            var t = null;
            if (e = i.defaults({
                parse: this.options.parse
            }, e), this.keyContents instanceof this.relatedModel)
                t = this.keyContents;
            else if (this.keyContents || 0 === this.keyContents) {
                var n = i.defaults({
                    create: this.options.createModels
                }, e);
                t = this.relatedModel.findOrCreate(this.keyContents, n)
            }
            return t && (this.keyId = null), t
        },
        setKeyContents: function(e) {
            this.keyContents = e, this.keyId = t.Relational.store.resolveIdForItem(this.relatedModel, this.keyContents)
        },
        onChange: function(e, n, o) {
            if (!this.isLocked()) {
                this.acquire(), o = o ? i.clone(o) : {};
                var s = i.isUndefined(o.__related), r = s ? this.related: o.__related;
                if (s) {
                    this.setKeyContents(n);
                    var a = this.findRelated(o);
                    this.setRelated(a)
                }
                if (r && this.related !== r && i.each(this.getReverseRelations(r), function(e) {
                    e.removeRelated(this.instance, null, o)
                }, this), i.each(this.getReverseRelations(), function(e) {
                    e.addRelated(this.instance, o)
                }, this), !o.silent && this.related !== r) {
                    var l = this;
                    this.changed=!0, t.Relational.eventQueue.add(function() {
                        l.instance.trigger("change:" + l.key, l.instance, l.related, o, !0), l.changed=!1
                    })
                }
                this.release()
            }
        },
        tryAddRelated: function(e, t, i) {
            !this.keyId && 0 !== this.keyId || e.id !== this.keyId || (this.addRelated(e, i), this.keyId = null)
        },
        addRelated: function(e, t) {
            var n = this;
            e.queue(function() {
                if (e !== n.related) {
                    var o = n.related || null;
                    n.setRelated(e), n.onChange(n.instance, e, i.defaults({
                        __related: o
                    }, t))
                }
            })
        },
        removeRelated: function(e, t, n) {
            if (this.related && e === this.related) {
                var o = this.related || null;
                this.setRelated(null), this.onChange(this.instance, e, i.defaults({
                    __related: o
                }, n))
            }
        }
    }), t.HasMany = t.Relation.extend({
        _class: "Backbone.HasMany",
        collectionType: null,
        options: {
            reverseRelation: {
                type: "HasOne"
            },
            collectionType: t.Collection,
            collectionKey: !0,
            collectionOptions: {}
        },
        initialize: function(e) {
            if (this.listenTo(this.instance, "relational:change:" + this.key, this.onChange), this.collectionType = this.options.collectionType, !i.isFunction(this.collectionType) || this.collectionType === t.Collection || this.collectionType.prototype instanceof t.Collection || (this.collectionType = i.result(this, "collectionType")), i.isString(this.collectionType)) {
                var n = this.collectionType;
                this.collectionType = t.Relational.store.getObjectByName(this.collectionType)
            }
            if (this.collectionType || (n && t.Relational.showWarnings && "undefined" != typeof console && console.warn("`collectionType` of " + n + " not found, using Backbone.Collection in its place."), this.collectionType = t.Collection), this.collectionType !== t.Collection&&!(this.collectionType.prototype instanceof t.Collection))
                throw new Error("`collectionType` must inherit from Backbone.Collection");
            var o = this.findRelated(e);
            this.setRelated(o)
        },
        _prepareCollection: function(e) {
            if (this.related && this.stopListening(this.related), !(e && e instanceof t.Collection)) {
                var n = i.isFunction(this.options.collectionOptions) ? this.options.collectionOptions(this.instance): this.options.collectionOptions;
                e = new this.collectionType(null, n)
            }
            if (e.model = this.relatedModel, this.options.collectionKey) {
                var o = this.options.collectionKey===!0 ? this.options.reverseRelation.key: this.options.collectionKey;
                e[o] && e[o] !== this.instance ? t.Relational.showWarnings && "undefined" != typeof console && console.warn("Relation=%o; collectionKey=%s already exists on collection=%o", this, o, this.options.collectionKey) : o && (e[o] = this.instance)
            }
            return this.listenTo(e, "relational:add", this.handleAddition).listenTo(e, "relational:remove", this.handleRemoval).listenTo(e, "relational:reset", this.handleReset), e
        },
        findRelated: function(e) {
            var n = null;
            if (e = i.defaults({
                parse: this.options.parse
            }, e), this.keyContents instanceof t.Collection)
                this._prepareCollection(this.keyContents), n = this.keyContents;
            else {
                var o = [];
                i.each(this.keyContents, function(t) {
                    if (t instanceof this.relatedModel)
                        var n = t;
                    else 
                        n = this.relatedModel.findOrCreate(t, i.extend({
                            merge: !0
                        }, e, {
                            create: this.options.createModels
                        }));
                    n && o.push(n)
                }, this), n = this.related instanceof t.Collection ? this.related : this._prepareCollection(), n.set(o, i.defaults({
                    merge: !1,
                    parse: !1
                }, e))
            }
            return this.keyIds = i.difference(this.keyIds, i.pluck(n.models, "id")), n
        },
        setKeyContents: function(e) {
            this.keyContents = e instanceof t.Collection ? e : null, this.keyIds = [], this.keyContents ||!e && 0 !== e || (this.keyContents = i.isArray(e) ? e : [e], i.each(this.keyContents, function(e) {
                var i = t.Relational.store.resolveIdForItem(this.relatedModel, e);
                (i || 0 === i) && this.keyIds.push(i)
            }, this))
        },
        onChange: function(e, n, o) {
            o = o ? i.clone(o) : {}, this.setKeyContents(n), this.changed=!1;
            var s = this.findRelated(o);
            if (this.setRelated(s), !o.silent) {
                var r = this;
                t.Relational.eventQueue.add(function() {
                    r.changed && (r.instance.trigger("change:" + r.key, r.instance, r.related, o, !0), r.changed=!1)
                })
            }
        },
        handleAddition: function(e, n, o) {
            o = o ? i.clone(o) : {}, this.changed=!0, i.each(this.getReverseRelations(e), function(e) {
                e.addRelated(this.instance, o)
            }, this);
            var s = this;
            !o.silent && t.Relational.eventQueue.add(function() {
                s.instance.trigger("add:" + s.key, e, s.related, o)
            })
        },
        handleRemoval: function(e, n, o) {
            o = o ? i.clone(o) : {}, this.changed=!0, i.each(this.getReverseRelations(e), function(e) {
                e.removeRelated(this.instance, null, o)
            }, this);
            var s = this;
            !o.silent && t.Relational.eventQueue.add(function() {
                s.instance.trigger("remove:" + s.key, e, s.related, o)
            })
        },
        handleReset: function(e, n) {
            var o = this;
            n = n ? i.clone(n) : {}, !n.silent && t.Relational.eventQueue.add(function() {
                o.instance.trigger("reset:" + o.key, o.related, n)
            })
        },
        tryAddRelated: function(e, t, n) {
            var o = i.contains(this.keyIds, e.id);
            o && (this.addRelated(e, n), this.keyIds = i.without(this.keyIds, e.id))
        },
        addRelated: function(e, t) {
            var n = this;
            e.queue(function() {
                n.related&&!n.related.get(e) && n.related.add(e, i.defaults({
                    parse: !1
                }, t))
            })
        },
        removeRelated: function(e, t, i) {
            this.related.get(e) && this.related.remove(e, i)
        }
    }), t.RelationalModel = t.Model.extend({
        _class: "Backbone.RelationalModel",
        relations: null,
        _relations: null,
        _isInitialized: !1,
        _deferProcessing: !1,
        _queue: null,
        _attributeChangeFired: !1,
        subModelTypeAttribute: "type",
        subModelTypes: null,
        keydefs: {},
        compoundKeyDelim: "-",
        constructor: function(e, n) {
            if (n && n.collection) {
                var o = this, s = this.collection = n.collection;
                delete n.collection, this._deferProcessing=!0;
                var r = function(e) {
                    e === o && (o._deferProcessing=!1, o.processQueue(), s.off("relational:add", r))
                };
                s.on("relational:add", r), i.defer(function() {
                    r(o)
                })
            }
            t.Relational.store.processOrphanRelations(), t.Relational.store.listenTo(this, "relational:unregister", t.Relational.store.unregister), this._queue = new t.BlockingQueue, this._queue.block(), t.Relational.eventQueue.block();
            try {
                t.Model.apply(this, arguments)
            } finally {
                t.Relational.eventQueue.unblock()
            }
            this.createSavepoint(), this.on("sync", this.createSavepoint, this)
        },
        createSavepoint: function() {
            return this.savepoint = JSON.parse(JSON.stringify(this)), this
        },
        rollback: function(e) {
            e = e || {}, this.set(this.savepoint), this.changed = [], e.silent || this.trigger("rollback", this, e)
        },
        commit: function(e) {
            e = e || {};
            var n = this, o = n.isNew(), s = [], r = [];
            return (o || n.hasChangedDeepSinceSavepoint()) && (o || i.each(n.relations, function(t) {
                if (t.includeInJSON && (s.push(t), t.includeInJSON=!1, !t.isAutoRelation)) {
                    var i = n.get(t.key);
                    r.push(i.commit(e))
                }
            }), r.push(n.save(null, e)), i.each(s, function(e) {
                e.includeInJSON=!0
            }), e.silent || n.trigger("commit", n, e)), t.$.when.apply(t.$, r)
        },
        hasChangedDeepSinceSavepoint: function() {
            return !i.isEqual(this.toJSON(), this.savepoint)
        },
        trigger: function(e) {
            if (e.length > 5 && 0 === e.indexOf("change")) {
                var i = this, n = arguments;
                t.Relational.eventQueue.add(function() {
                    if (i._isInitialized) {
                        var o=!0;
                        if ("change" === e)
                            o = i.hasChanged() || i._attributeChangeFired, i._attributeChangeFired=!1;
                        else {
                            var s = e.slice(7), r = i.getRelation(s);
                            r ? (o = n[4]===!0, o ? i.changed[s] = n[2] : r.changed || delete i.changed[s]) : o && (i._attributeChangeFired=!0)
                        }
                        o && t.Model.prototype.trigger.apply(i, n)
                    }
                })
            } else 
                "destroy" === e ? (t.Model.prototype.trigger.apply(this, arguments), t.Relational.store.unregister(this)) : t.Model.prototype.trigger.apply(this, arguments);
            return this
        },
        initializeRelations: function(e) {
            this.acquire(), this._relations = {}, i.each(this.relations || [], function(i) {
                t.Relational.store.initializeRelation(this, i, e)
            }, this), this._isInitialized=!0, this.release(), this.processQueue()
        },
        updateRelations: function(e, t) {
            this._isInitialized&&!this.isLocked() && i.each(this._relations, function(i) {
                if (!e || i.keySource in e || i.key in e) {
                    var n = this.attributes[i.keySource] || this.attributes[i.key], o = e && (e[i.keySource] || e[i.key]);
                    (i.related !== n || null === n && null === o) && this.trigger("relational:change:" + i.key, this, n, t || {})
                }
                i.keySource !== i.key && delete this.attributes[i.keySource]
            }, this)
        },
        queue: function(e) {
            this._queue.add(e)
        },
        processQueue: function() {
            this._isInitialized&&!this._deferProcessing && this._queue.isBlocked() && this._queue.unblock()
        },
        getRelation: function(e) {
            return this._relations[e]
        },
        getRelations: function() {
            return i.values(this._relations)
        },
        fetchRelated: function(e, n, o) {
            n = i.extend({
                update: !0,
                remove: !1
            }, n);
            var s, r, a = [], l = this.getRelation(e), c = l && (l.keyIds && l.keyIds.slice(0) || (l.keyId || 0 === l.keyId ? [l.keyId] : []));
            if (o && (s = l.related instanceof t.Collection ? l.related.models : [l.related], i.each(s, function(e) {
                (e.id || 0 === e.id) && c.push(e.id)
            })), c && c.length) {
                var u = [];
                if (s = i.map(c, function(e) {
                    var t = l.relatedModel.findModel(e);
                    if (!t) {
                        var i = {};
                        i[l.relatedModel.prototype.idAttribute] = e, t = l.relatedModel.findOrCreate(i, n), u.push(t)
                    }
                    return t
                }, this), l.related instanceof t.Collection && i.isFunction(l.related.url) && (r = l.related.url(s)), r && r !== l.related.url()) {
                    var d = i.defaults({
                        error: function() {
                            var e = arguments;
                            i.each(u, function(t) {
                                t.trigger("destroy", t, t.collection, n), n.error && n.error.apply(t, e)
                            })
                        },
                        url: r
                    }, n);
                    a = [l.related.fetch(d)]
                } else 
                    a = i.map(s, function(e) {
                        var t = i.defaults({
                            error: function() {
                                i.contains(u, e) && (e.trigger("destroy", e, e.collection, n), n.error && n.error.apply(e, arguments))
                            }
                        }, n);
                        return e.fetch(t)
                    }, this)
            }
            return a
        },
        get: function(e) {
            var n = t.Model.prototype.get.call(this, e);
            if (!this.dotNotation||-1 === e.indexOf("."))
                return n;
            var o = e.split("."), s = i.reduce(o, function(e, n) {
                if (i.isNull(e) || i.isUndefined(e))
                    return void 0;
                if (e instanceof t.Model)
                    return t.Model.prototype.get.call(e, n);
                if (e instanceof t.Collection)
                    return t.Collection.prototype.at.call(e, n);
                throw new Error("Attribute must be an instanceof Backbone.Model or Backbone.Collection. Is: " + e + ", currentSplit: " + n)
            }, this);
            if (void 0 !== n && void 0 !== s)
                throw new Error("Ambiguous result for '" + e + "'. direct result: " + n + ", dotNotation: " + s);
            return n || s
        },
        _getNewId: function(e) {
            var t, i = this, n = null, o = "", s=!1, r = i.keydefs.PRIMARY;
            return r && (r.forEach(function(n) {
                t = e[n] || i.attributes && i.attributes[n], t || 0 === t ? o += (o ? i.compoundKeyDelim : "") + (e[n] || i.attributes && i.attributes[n]) : s=!0
            }), s || (n = o)), n || (n = e && i.idAttribute in e && e[i.idAttribute]), n
        },
        set: function(e, n, o) {
            t.Relational.eventQueue.block();
            var s;
            i.isObject(e) || null == e ? (s = e, o = n) : (s = {}, s[e] = n);
            try {
                var r = this.id, a = this._getNewId(s);
                t.Relational.store.checkId(this, a);
                var l = t.Model.prototype.set.apply(this, arguments);
                a && (this.id = a), this._isInitialized || this.isLocked() ? a && a !== r && t.Relational.store.update(this) : (this.constructor.initializeModelHierarchy(), (a || 0 === a) && t.Relational.store.register(this), this.initializeRelations(o)), s && this.updateRelations(s, o)
            } finally {
                t.Relational.eventQueue.unblock()
            }
            return l
        },
        clone: function() {
            var e = i.clone(this.attributes);
            return i.isUndefined(e[this.idAttribute]) || (e[this.idAttribute] = null), i.each(this.getRelations(), function(t) {
                delete e[t.key]
            }), new this.constructor(e)
        },
        toJSON: function(e) {
            if (this.isLocked())
                return this.id;
            this.acquire();
            var n = t.Model.prototype.toJSON.call(this, e);
            return !this.constructor._superModel || this.constructor._subModelTypeAttribute in n || (n[this.constructor._subModelTypeAttribute] = this.constructor._subModelTypeValue), i.each(this._relations, function(o) {
                var s = n[o.key], r = o.options.includeInJSON, a = null;
                r===!0 ? s && i.isFunction(s.toJSON) && (a = s.toJSON(e)) : i.isString(r) ? (s instanceof t.Collection ? a = s.pluck(r) : s instanceof t.Model && (a = s.get(r)), r === o.relatedModel.prototype.idAttribute && (o instanceof t.HasMany ? a = a.concat(o.keyIds) : o instanceof t.HasOne && (a = a || o.keyId, a || i.isObject(o.keyContents) || (a = o.keyContents || null)))) : i.isArray(r) ? s instanceof t.Collection ? (a = [], s.each(function(e) {
                    var t = {};
                    i.each(r, function(i) {
                        t[i] = e.get(i)
                    }), a.push(t)
                })) : s instanceof t.Model && (a = {}, i.each(r, function(e) {
                    a[e] = s.get(e)
                })) : delete n[o.key], r && (n[o.keyDestination] = a), o.keyDestination !== o.key && delete n[o.key]
            }), this.release(), n
        }
    }, {
        setup: function() {
            return this.prototype.relations = (this.prototype.relations || []).slice(0), this._subModels = {}, this._superModel = null, this.prototype.hasOwnProperty("subModelTypes") ? t.Relational.store.addSubModels(this.prototype.subModelTypes, this) : this.prototype.subModelTypes = null, i.each(this.prototype.relations || [], function(e) {
                if (e.model || (e.model = this), e.reverseRelation && e.model === this) {
                    var n=!0;
                    if (i.isString(e.relatedModel)) {
                        var o = t.Relational.store.getObjectByName(e.relatedModel);
                        n = o && o.prototype instanceof t.RelationalModel
                    }
                    n ? t.Relational.store.initializeRelation(null, e) : i.isString(e.relatedModel) && t.Relational.store.addOrphanRelation(e)
                }
            }, this), this
        },
        build: function(e, t) {
            this.initializeModelHierarchy();
            var i = this._findSubModelType(this, e) || this;
            return new i(e, t)
        },
        _findSubModelType: function(e, t) {
            if (e._subModels && e.prototype.subModelTypeAttribute in t) {
                var i = t[e.prototype.subModelTypeAttribute], n = e._subModels[i];
                if (n)
                    return n;
                for (i in e._subModels)
                    if (n = this._findSubModelType(e._subModels[i], t))
                        return n
            }
            return null
        },
        initializeModelHierarchy: function() {
            if (this.inheritRelations(), this.prototype.subModelTypes) {
                var e = i.keys(this._subModels), n = i.omit(this.prototype.subModelTypes, e);
                i.each(n, function(e) {
                    var i = t.Relational.store.getObjectByName(e);
                    i && i.initializeModelHierarchy()
                })
            }
        },
        inheritRelations: function() {
            if (i.isUndefined(this._superModel) || i.isNull(this._superModel))
                if (t.Relational.store.setupSuperModel(this), this._superModel) {
                    if (this._superModel.inheritRelations(), this._superModel.prototype.relations) {
                        var e = i.filter(this._superModel.prototype.relations || [], function(e) {
                            return !i.any(this.prototype.relations || [], function(t) {
                                return e.relatedModel === t.relatedModel && e.key === t.key
                            }, this)
                        }, this);
                        this.prototype.relations = e.concat(this.prototype.relations)
                    }
                } else 
                    this._superModel=!1
            },
        findOrCreate: function(e, t) {
            t || (t = {});
            var n = i.isObject(e) && t.parse && this.prototype.parse ? this.prototype.parse(i.clone(e)): e, o = this.findModel(n);
            return i.isObject(e) && (o && t.merge!==!1 ? (delete t.collection, delete t.url, o.set(n, t)) : o || t.create===!1 || (o = this.build(e, t))), o
        },
        find: function(e, t) {
            return t || (t = {}), t.create=!1, this.findOrCreate(e, t)
        },
        findModel: function(e) {
            return t.Relational.store.find(this, e)
        }
    }), i.extend(t.RelationalModel.prototype, t.Semaphore);
    var n = t.Collection.prototype.__initialize = t.Collection.prototype.initialize;
    t.Collection.prototype.initialize = function() {
        n.apply(this, arguments), this.createSavepoint(), this.on("sync", this.createSavepoint, this)
    }, t.Collection.prototype.createSavepoint = function() {
        return this._added = [], this._removed = [], this
    }, t.Collection.prototype.rollback = function(e) {
        return e = e || {}, this.add(this._removed, {
            silent: !0
        }), i.each(this.models, function(e) {
            e.rollback()
        }), this.remove(this._added, {
            silent: !0
        }), this.createSavepoint(), e.silent || this.trigger("rollback", this, e), this
    }, t.Collection.prototype.hasChangedDeepSinceSavepoint = function() {
        return this._removed.length || this._added.length?!0 : !!i.find(this.models, function(e) {
            return e.hasChangedDeepSinceSavepoint()
        })
    }, t.Collection.prototype.commit = function(e) {
        var n = [];
        return this.hasChangedDeepSinceSavepoint() && (e = e || {}, i.each(this._removed, function(e) {
            n.push(e.destroy())
        }), i.each(this.models, function(e) {
            n.push(e.commit())
        }), this.createSavepoint(), e.silent || this.trigger("commit", this, e)), t.$.when.apply(t.$, n)
    }, t.Collection.prototype._class = "Backbone.Collection", t.Collection.prototype.__prepareModel = t.Collection.prototype._prepareModel, t.Collection.prototype._prepareModel = function(e, n) {
        var o;
        return e instanceof t.Model ? (e.collection || (e.collection = this), o = e) : (n = n ? i.clone(n) : {}, n.collection = this, o = "undefined" != typeof this.model.findOrCreate ? this.model.findOrCreate(e, n) : new this.model(e, n), o && o.validationError && (this.trigger("invalid", this, e, n), o=!1)), o
    };
    var o = t.Collection.prototype.__set = t.Collection.prototype.set;
    t.Collection.prototype.set = function(e, n) {
        if (!(this.model.prototype instanceof t.RelationalModel))
            return o.apply(this, arguments);
        n && n.parse && (e = this.parse(e, n));
        var s=!i.isArray(e), r = [], a = [];
        e = s ? e ? [e] : [] : i.clone(e), i.each(e, function(e) {
            e instanceof t.Model || (e = t.Collection.prototype._prepareModel.call(this, e, n)), e && (a.push(e), this.get(e) || this.get(e.cid) ? null != e.id && (this._byId[e.id] = e) : r.push(e))
        }, this), a = s ? a.length ? a[0] : null : a;
        var l = o.call(this, a, i.defaults({
            parse: !1
        }, n));
        return i.each(r, function(e) {
            (this.get(e) || this.get(e.cid)) && this.trigger("relational:add", e, this, n)
        }, this), l
    };
    var s = t.Collection.prototype.__get = t.Collection.prototype.get;
    t.Collection.prototype.get = function(e) {
        if (!(this.model.prototype instanceof t.RelationalModel))
            return s.apply(this, arguments);
        if (null == e)
            return void 0;
        var i = t.Relational.store.resolveIdForItem(this.model, e);
        return this._byId[null != i ? i: e.cid || e]
    };
    var r = t.Collection.prototype.__add = t.Collection.prototype.add;
    t.Collection.prototype.add = function(e, n) {
        if (!(this.model.prototype instanceof t.RelationalModel))
            return r.apply(this, arguments);
        var o = r.call(this, e, i.defaults(n || {}, {
            add: !0,
            merge: !1,
            remove: !1
        }));
        return this._added = this._added.concat(o), o
    };
    var a = t.Collection.prototype.__remove = t.Collection.prototype.remove;
    t.Collection.prototype.remove = function(e, n) {
        if (!(this.model.prototype instanceof t.RelationalModel))
            return this._removed = this._removed.concat(e), a.apply(this, arguments);
        var o=!i.isArray(e), s = [];
        e = o ? e ? [e] : [] : i.clone(e), n || (n = {}), i.each(e, function(e) {
            e = this.get(e) || e && this.get(e.cid), e && s.push(e)
        }, this);
        var r = a.call(this, o ? s.length ? s[0] : null : s, n);
        return i.each(s, function(e) {
            this.trigger("relational:remove", e, this, n)
        }, this), this._removed = this._removed.concat(s), r
    };
    var l = t.Collection.prototype.__reset = t.Collection.prototype.reset;
    t.Collection.prototype.reset = function(e, n) {
        n = i.extend({
            merge: !0
        }, n);
        var o = l.call(this, e, n);
        return this._added = [], this._removed = [], this.model.prototype instanceof t.RelationalModel && this.trigger("relational:reset", this, n), this.createSavepoint(), o
    };
    var c = t.Collection.prototype.__sort = t.Collection.prototype.sort;
    t.Collection.prototype.sort = function(e) {
        var i = c.call(this, e);
        return this.model.prototype instanceof t.RelationalModel && this.trigger("relational:reset", this, e), i
    };
    var u = t.Collection.prototype.__trigger = t.Collection.prototype.trigger;
    t.Collection.prototype.trigger = function(e) {
        if (!(this.model.prototype instanceof t.RelationalModel))
            return u.apply(this, arguments);
        if ("add" === e || "remove" === e || "reset" === e || "sort" === e) {
            var n = this, o = arguments;
            i.isObject(o[3]) && (o = i.toArray(o), o[3] = i.clone(o[3])), t.Relational.eventQueue.add(function() {
                u.apply(n, o)
            })
        } else 
            u.apply(this, arguments);
        return this
    }, t.RelationalModel.extend = function() {
        var e = t.Model.extend.apply(this, arguments);
        return e.setup(this), e
    }
}), function(e, t) {
    if ("object" == typeof exports) {
        var i = require("underscore"), n = require("backbone");
        module.exports = t(i, n)
    } else 
        "function" == typeof _wAMD.define && _wAMD.define.amd && _wAMD.define("backbone.wreqr", ["underscore", "backbone"], t)
}(this, function(e, t) {
    return t.Wreqr = function(e, t, i) {
        var n = {};
        return n.Handlers = function(e, t) {
            var i = function(e) {
                this.options = e, this._wreqrHandlers = {}, t.isFunction(this.initialize) && this.initialize(e)
            };
            return i.extend = e.Model.extend, t.extend(i.prototype, e.Events, {
                setHandlers: function(e) {
                    t.each(e, function(e, i) {
                        var n = null;
                        t.isObject(e)&&!t.isFunction(e) && (n = e.context, e = e.callback), this.setHandler(i, e, n)
                    }, this)
                },
                setHandler: function(e, t, i) {
                    var n = {
                        callback: t,
                        context: i
                    };
                    this._wreqrHandlers[e] = n, this.trigger("handler:add", e, t, i)
                },
                hasHandler: function(e) {
                    return !!this._wreqrHandlers[e]
                },
                getHandler: function(e) {
                    var t = this._wreqrHandlers[e];
                    if (!t)
                        throw new Error("Handler not found for '" + e + "'");
                    return function() {
                        var e = Array.prototype.slice.apply(arguments);
                        return t.callback.apply(t.context, e)
                    }
                },
                removeHandler: function(e) {
                    delete this._wreqrHandlers[e]
                },
                removeAllHandlers: function() {
                    this._wreqrHandlers = {}
                }
            }), i
        }(e, i), n.CommandStorage = function() {
            var t = function(e) {
                this.options = e, this._commands = {}, i.isFunction(this.initialize) && this.initialize(e)
            };
            return i.extend(t.prototype, e.Events, {
                getCommands: function(e) {
                    var t = this._commands[e];
                    return t || (t = {
                        command: e,
                        instances: []
                    }, this._commands[e] = t), t
                },
                addCommand: function(e, t) {
                    var i = this.getCommands(e);
                    i.instances.push(t)
                },
                clearCommands: function(e) {
                    var t = this.getCommands(e);
                    t.instances = []
                }
            }), t
        }(), n.Commands = function(e) {
            return e.Handlers.extend({
                storageType: e.CommandStorage,
                constructor: function(t) {
                    this.options = t || {}, this._initializeStorage(this.options), this.on("handler:add", this._executeCommands, this);
                    var i = Array.prototype.slice.call(arguments);
                    e.Handlers.prototype.constructor.apply(this, i)
                },
                execute: function(e, t) {
                    e = arguments[0], t = Array.prototype.slice.call(arguments, 1), this.hasHandler(e) ? this.getHandler(e).apply(this, t) : this.storage.addCommand(e, t)
                },
                _executeCommands: function(e, t, n) {
                    var o = this.storage.getCommands(e);
                    i.each(o.instances, function(e) {
                        t.apply(n, e)
                    }), this.storage.clearCommands(e)
                },
                _initializeStorage: function(e) {
                    var t, n = e.storageType || this.storageType;
                    t = i.isFunction(n) ? new n : n, this.storage = t
                }
            })
        }(n), n.RequestResponse = function(e) {
            return e.Handlers.extend({
                request: function() {
                    var e = arguments[0], t = Array.prototype.slice.call(arguments, 1);
                    return this.getHandler(e).apply(this, t)
                }
            })
        }(n), n.EventAggregator = function(e, t) {
            var i = function() {};
            return i.extend = e.Model.extend, t.extend(i.prototype, e.Events), i
        }(e, i), n
    }(t, t.Marionette, e), t.Wreqr
}), function(e, t) {
    if ("object" == typeof exports) {
        var i = require("underscore"), n = require("backbone");
        module.exports = t(i, n)
    } else 
        "function" == typeof _wAMD.define && _wAMD.define.amd && _wAMD.define("backbone.babysitter", ["underscore", "backbone"], t)
}(this, function(e, t) {
    "option strict";
    return t.ChildViewContainer = function(e, t) {
        var i = function(e) {
            this._views = {}, this._indexByModel = {}, this._indexByCustom = {}, this._updateLength(), t.each(e, this.add, this)
        };
        t.extend(i.prototype, {
            add: function(e, t) {
                var i = e.cid;
                this._views[i] = e, e.model && (this._indexByModel[e.model.cid] = i), t && (this._indexByCustom[t] = i), this._updateLength()
            },
            findByModel: function(e) {
                return this.findByModelCid(e.cid)
            },
            findByModelCid: function(e) {
                var t = this._indexByModel[e];
                return this.findByCid(t)
            },
            findByCustom: function(e) {
                var t = this._indexByCustom[e];
                return this.findByCid(t)
            },
            findByIndex: function(e) {
                return t.values(this._views)[e]
            },
            findByCid: function(e) {
                return this._views[e]
            },
            remove: function(e) {
                var i = e.cid;
                e.model && delete this._indexByModel[e.model.cid], t.any(this._indexByCustom, function(e, t) {
                    return e === i ? (delete this._indexByCustom[t], !0) : void 0
                }, this), delete this._views[i], this._updateLength()
            },
            call: function(e) {
                this.apply(e, t.tail(arguments))
            },
            apply: function(e, i) {
                t.each(this._views, function(n) {
                    t.isFunction(n[e]) && n[e].apply(n, i || [])
                })
            },
            _updateLength: function() {
                this.length = t.size(this._views)
            }
        });
        var n = ["forEach", "each", "map", "find", "detect", "filter", "select", "reject", "every", "all", "some", "any", "include", "contains", "invoke", "toArray", "first", "initial", "rest", "last", "without", "isEmpty", "pluck"];
        return t.each(n, function(e) {
            i.prototype[e] = function() {
                var i = t.values(this._views), n = [i].concat(t.toArray(arguments));
                return t[e].apply(t, n)
            }
        }), i
    }(t, e), t.ChildViewContainer
}), function(e, t) {
    if ("object" == typeof exports) {
        var i = require("underscore"), n = require("backbone"), o = require("backbone.wreqr"), s = require("backbone.babysitter");
        module.exports = t(i, n, o, s)
    } else 
        "function" == typeof _wAMD.define && _wAMD.define.amd && _wAMD.define("backbone-marionette", ["underscore", "backbone", "backbone.wreqr", "backbone.babysitter"], t)
}(this, function(e, t) {
    return function(e, t, i) {
        function n(e) {
            return r.call(e)
        }
        function o(e, t) {
            var i = new Error(e);
            throw i.name = t || "Error", i
        }
        var s = {};
        t.Marionette = s, s.$ = t.$;
        var r = Array.prototype.slice;
        return s.extend = t.Model.extend, s.getOption = function(e, t) {
            if (e && t) {
                var i;
                return i = e.options && t in e.options && void 0 !== e.options[t] ? e.options[t] : e[t]
            }
        }, s.normalizeMethods = function(e) {
            var t, n = {};
            return i.each(e, function(e, o) {
                t = e, i.isFunction(t) || (t = this[t]), t && (n[o] = t)
            }, this), n
        }, s.triggerMethod = function() {
            function e(e, t, i) {
                return i.toUpperCase()
            }
            var t = /(^|:)(\w)/gi, n = function(n) {
                var o = "on" + n.replace(t, e), s = this[o];
                return i.isFunction(this.trigger) && this.trigger.apply(this, arguments), i.isFunction(s) ? s.apply(this, i.tail(arguments)) : void 0
            };
            return n
        }(), s.MonitorDOMRefresh = function(e) {
            function t(e) {
                e._isShown=!0, o(e)
            }
            function n(e) {
                e._isRendered=!0, o(e)
            }
            function o(e) {
                e._isShown && e._isRendered && s(e) && i.isFunction(e.triggerMethod) && e.triggerMethod("dom:refresh")
            }
            function s(t) {
                return e.contains(t.el)
            }
            return function(e) {
                e.listenTo(e, "show", function() {
                    t(e)
                }), e.listenTo(e, "render", function() {
                    n(e)
                })
            }
        }(document.documentElement), function(e) {
            function t(e, t, n, s) {
                var r = s.split(/\s+/);
                i.each(r, function(i) {
                    var s = e[i];
                    s || o("Method '" + i + "' was configured as an event handler, but does not exist."), e.listenTo(t, n, s, e)
                })
            }
            function n(e, t, i, n) {
                e.listenTo(t, i, n, e)
            }
            function s(e, t, n, o) {
                var s = o.split(/\s+/);
                i.each(s, function(i) {
                    var o = e[i];
                    e.stopListening(t, n, o, e)
                })
            }
            function r(e, t, i, n) {
                e.stopListening(t, i, n, e)
            }
            function a(e, t, n, o, s) {
                t && n && (i.isFunction(n) && (n = n.call(e)), i.each(n, function(n, r) {
                    i.isFunction(n) ? o(e, t, r, n) : s(e, t, r, n)
                }))
            }
            e.bindEntityEvents = function(e, i, o) {
                a(e, i, o, n, t)
            }, e.unbindEntityEvents = function(e, t, i) {
                a(e, t, i, r, s)
            }
        }(s), s.Callbacks = function() {
            this._deferred = s.$.Deferred(), this._callbacks = []
        }, i.extend(s.Callbacks.prototype, {
            add: function(e, t) {
                this._callbacks.push({
                    cb: e,
                    ctx: t
                }), this._deferred.done(function(i, n) {
                    t && (i = t), e.call(i, n)
                })
            },
            run: function(e, t) {
                this._deferred.resolve(t, e)
            },
            reset: function() {
                var e = this._callbacks;
                this._deferred = s.$.Deferred(), this._callbacks = [], i.each(e, function(e) {
                    this.add(e.cb, e.ctx)
                }, this)
            }
        }), s.Controller = function(e) {
            this.triggerMethod = s.triggerMethod, this.options = e || {}, i.isFunction(this.initialize) && this.initialize(this.options)
        }, s.Controller.extend = s.extend, i.extend(s.Controller.prototype, t.Events, {
            close: function() {
                this.stopListening(), this.triggerMethod("close"), this.unbind()
            }
        }), s.Region = function(e) {
            if (this.options = e || {}, this.el = s.getOption(this, "el"), !this.el) {
                var t = new Error("An 'el' must be specified for a region.");
                throw t.name = "NoElError", t
            }
            if (this.initialize) {
                var i = Array.prototype.slice.apply(arguments);
                this.initialize.apply(this, i)
            }
        }, i.extend(s.Region, {
            buildRegion: function(e, t) {
                var n = "string" == typeof e, o = "string" == typeof e.selector, s = "undefined" == typeof e.regionType, r = "function" == typeof e;
                if (!r&&!n&&!o)
                    throw new Error("Region must be specified as a Region type, a selector string or an object with selector property");
                var a, l;
                n && (a = e), e.selector && (a = e.selector, delete e.selector), r && (l = e), !r && s && (l = t), e.regionType && (l = e.regionType, delete e.regionType), (n || r) && (e = {}), e.el = a;
                var c = new l(e);
                return e.parentEl && (c.getEl = function(t) {
                    var n = e.parentEl;
                    return i.isFunction(n) && (n = n()), n.find(t)
                }), c
            }
        }), i.extend(s.Region.prototype, t.Events, {
            show: function(e) {
                this.ensureEl();
                var t = e.isClosed || i.isUndefined(e.$el), n = e !== this.currentView;
                n && this.close(), e.render(), (n || t) && this.open(e), this.currentView = e, s.triggerMethod.call(this, "show", e), s.triggerMethod.call(e, "show")
            },
            ensureEl: function() {
                this.$el && 0 !== this.$el.length || (this.$el = this.getEl(this.el))
            },
            getEl: function(e) {
                return s.$(e)
            },
            open: function(e) {
                this.$el.empty().append(e.el)
            },
            close: function() {
                var e = this.currentView;
                e&&!e.isClosed && (e.close ? e.close() : e.remove && e.remove(), s.triggerMethod.call(this, "close", e), delete this.currentView)
            },
            attachView: function(e) {
                this.currentView = e
            },
            reset: function() {
                this.close(), delete this.$el
            }
        }), s.Region.extend = s.extend, s.RegionManager = function(e) {
            var t = e.Controller.extend({
                constructor: function(t) {
                    this._regions = {}, e.Controller.prototype.constructor.call(this, t)
                },
                addRegions: function(e, t) {
                    var n = {};
                    return i.each(e, function(e, o) {
                        "string" == typeof e && (e = {
                            selector: e
                        }), e.selector && (e = i.defaults({}, e, t));
                        var s = this.addRegion(o, e);
                        n[o] = s
                    }, this), n
                },
                addRegion: function(t, n) {
                    var o, s = i.isObject(n), r = i.isString(n), a=!!n.selector;
                    return o = r || s && a ? e.Region.buildRegion(n, e.Region) : i.isFunction(n) ? e.Region.buildRegion(n, e.Region) : n, this._store(t, o), this.triggerMethod("region:add", t, o), o
                },
                get: function(e) {
                    return this._regions[e]
                },
                removeRegion: function(e) {
                    var t = this._regions[e];
                    this._remove(e, t)
                },
                removeRegions: function() {
                    i.each(this._regions, function(e, t) {
                        this._remove(t, e)
                    }, this)
                },
                closeRegions: function() {
                    i.each(this._regions, function(e) {
                        e.close()
                    }, this)
                },
                close: function() {
                    this.removeRegions();
                    var t = Array.prototype.slice.call(arguments);
                    e.Controller.prototype.close.apply(this, t)
                },
                _store: function(e, t) {
                    this._regions[e] = t, this._setLength()
                },
                _remove: function(e, t) {
                    t.close(), delete this._regions[e], this._setLength(), this.triggerMethod("region:remove", e, t)
                },
                _setLength: function() {
                    this.length = i.size(this._regions)
                }
            }), n = ["forEach", "each", "map", "find", "detect", "filter", "select", "reject", "every", "all", "some", "any", "include", "contains", "invoke", "toArray", "first", "initial", "rest", "last", "without", "isEmpty", "pluck"];
            return i.each(n, function(e) {
                t.prototype[e] = function() {
                    var t = i.values(this._regions), n = [t].concat(i.toArray(arguments));
                    return i[e].apply(i, n)
                }
            }), t
        }(s), s.TemplateCache = function(e) {
            this.templateId = e
        }, i.extend(s.TemplateCache, {
            templateCaches: {},
            get: function(e) {
                var t = this.templateCaches[e];
                return t || (t = new s.TemplateCache(e), this.templateCaches[e] = t), t.load()
            },
            clear: function() {
                var e, t = n(arguments), i = t.length;
                if (i > 0)
                    for (e = 0; i > e; e++)
                        delete this.templateCaches[t[e]];
                else 
                    this.templateCaches = {}
            }
        }), i.extend(s.TemplateCache.prototype, {
            load: function() {
                if (this.compiledTemplate)
                    return this.compiledTemplate;
                var e = this.loadTemplate(this.templateId);
                return this.compiledTemplate = this.compileTemplate(e), this.compiledTemplate
            },
            loadTemplate: function(e) {
                var t = s.$(e).html();
                return t && 0 !== t.length || o("Could not find template: '" + e + "'", "NoTemplateError"), t
            },
            compileTemplate: function(e) {
                return i.template(e)
            }
        }), s.Renderer = {
            render: function(e, t) {
                if (!e) {
                    var i = new Error("Cannot render the template since it's false, null or undefined.");
                    throw i.name = "TemplateNotFoundError", i
                }
                var n;
                return n = "function" == typeof e ? e : s.TemplateCache.get(e), n(t)
            }
        }, s.View = t.View.extend({
            constructor: function(e) {
                i.bindAll(this, "render");
                var n = Array.prototype.slice.apply(arguments);
                this.options = i.extend({}, i.result(this, "options"), i.isFunction(e) ? e.call(this) : e), this.events = this.normalizeUIKeys(i.result(this, "events")), t.View.prototype.constructor.apply(this, n), s.MonitorDOMRefresh(this), this.listenTo(this, "show", this.onShowCalled, this)
            },
            triggerMethod: s.triggerMethod,
            normalizeMethods: s.normalizeMethods,
            getTemplate: function() {
                return s.getOption(this, "template")
            },
            mixinTemplateHelpers: function(e) {
                e = e || {};
                var t = s.getOption(this, "templateHelpers");
                return i.isFunction(t) && (t = t.call(this)), i.extend(e, t)
            },
            normalizeUIKeys: function(e) {
                return "undefined" != typeof e ? (i.each(i.keys(e), function(t) {
                    var i = t.split("@ui.");
                    2 === i.length && (e[i[0] + this.ui[i[1]]] = e[t], delete e[t])
                }, this), e) : void 0
            },
            configureTriggers: function() {
                if (this.triggers) {
                    var e = {}, t = this.normalizeUIKeys(i.result(this, "triggers"));
                    return i.each(t, function(t, n) {
                        var o = i.isObject(t), s = o ? t.event: t;
                        e[n] = function(e) {
                            if (e) {
                                var i = e.preventDefault, n = e.stopPropagation, r = o ? t.preventDefault: i, a = o ? t.stopPropagation: n;
                                r && i && i.apply(e), a && n && n.apply(e)
                            }
                            var l = {
                                view: this,
                                model: this.model,
                                collection: this.collection
                            };
                            this.triggerMethod(s, l)
                        }
                    }, this), e
                }
            },
            delegateEvents: function(e) {
                this._delegateDOMEvents(e), s.bindEntityEvents(this, this.model, s.getOption(this, "modelEvents")), s.bindEntityEvents(this, this.collection, s.getOption(this, "collectionEvents"))
            },
            _delegateDOMEvents: function(e) {
                e = e || this.events, i.isFunction(e) && (e = e.call(this));
                var n = {}, o = this.configureTriggers();
                i.extend(n, e, o), t.View.prototype.delegateEvents.call(this, n)
            },
            undelegateEvents: function() {
                var e = Array.prototype.slice.call(arguments);
                t.View.prototype.undelegateEvents.apply(this, e), s.unbindEntityEvents(this, this.model, s.getOption(this, "modelEvents")), s.unbindEntityEvents(this, this.collection, s.getOption(this, "collectionEvents"))
            },
            onShowCalled: function() {},
            close: function() {
                if (!this.isClosed) {
                    var e = this.triggerMethod("before:close");
                    e!==!1 && (this.isClosed=!0, this.triggerMethod("close"), this.unbindUIElements(), this.remove())
                }
            },
            bindUIElements: function() {
                if (this.ui) {
                    this._uiBindings || (this._uiBindings = this.ui);
                    var e = i.result(this, "_uiBindings");
                    this.ui = {}, i.each(i.keys(e), function(t) {
                        var i = e[t];
                        this.ui[t] = this.$(i)
                    }, this)
                }
            },
            unbindUIElements: function() {
                this.ui && this._uiBindings && (i.each(this.ui, function(e, t) {
                    delete this.ui[t]
                }, this), this.ui = this._uiBindings, delete this._uiBindings)
            }
        }), s.ItemView = s.View.extend({
            constructor: function() {
                s.View.prototype.constructor.apply(this, n(arguments))
            },
            serializeData: function() {
                var e = {};
                return this.model ? e = this.model.toJSON() : this.collection && (e = {
                    items: this.collection.toJSON()
                }), e
            },
            render: function() {
                this.isClosed=!1, this.triggerMethod("before:render", this), this.triggerMethod("item:before:render", this);
                var e = this.serializeData();
                e = this.mixinTemplateHelpers(e);
                var t = this.getTemplate(), i = s.Renderer.render(t, e);
                return this.$el.html(i), this.bindUIElements(), this.triggerMethod("render", this), this.triggerMethod("item:rendered", this), this
            },
            close: function() {
                this.isClosed || (this.triggerMethod("item:before:close"), s.View.prototype.close.apply(this, n(arguments)), this.triggerMethod("item:closed"))
            }
        }), s.CollectionView = s.View.extend({
            itemViewEventPrefix: "itemview",
            constructor: function() {
                this._initChildViewStorage(), s.View.prototype.constructor.apply(this, n(arguments)), this._initialEvents(), this.initRenderBuffer()
            },
            initRenderBuffer: function() {
                this.elBuffer = document.createDocumentFragment(), this._bufferedChildren = []
            },
            startBuffering: function() {
                this.initRenderBuffer(), this.isBuffering=!0
            },
            endBuffering: function() {
                this.isBuffering=!1, this.appendBuffer(this, this.elBuffer), this._triggerShowBufferedChildren(), this.initRenderBuffer()
            },
            _triggerShowBufferedChildren: function() {
                this._isShown && (i.each(this._bufferedChildren, function(e) {
                    s.triggerMethod.call(e, "show")
                }), this._bufferedChildren = [])
            },
            _initialEvents: function() {
                this.collection && (this.listenTo(this.collection, "add", this.addChildView, this), this.listenTo(this.collection, "remove", this.removeItemView, this), this.listenTo(this.collection, "reset", this.render, this))
            },
            addChildView: function(e) {
                this.closeEmptyView();
                var t = this.getItemView(e), i = this.collection.indexOf(e);
                this.addItemView(e, t, i)
            },
            onShowCalled: function() {
                this.children.each(function(e) {
                    s.triggerMethod.call(e, "show")
                })
            },
            triggerBeforeRender: function() {
                this.triggerMethod("before:render", this), this.triggerMethod("collection:before:render", this)
            },
            triggerRendered: function() {
                this.triggerMethod("render", this), this.triggerMethod("collection:rendered", this)
            },
            render: function() {
                return this.isClosed=!1, this.triggerBeforeRender(), this._renderChildren(), this.triggerRendered(), this
            },
            _renderChildren: function() {
                this.startBuffering(), this.closeEmptyView(), this.closeChildren(), this.isEmpty(this.collection) ? this.showEmptyView() : this.showCollection(), this.endBuffering()
            },
            showCollection: function() {
                var e;
                this.collection.each(function(t, i) {
                    e = this.getItemView(t), this.addItemView(t, e, i)
                }, this)
            },
            showEmptyView: function() {
                var e = this.getEmptyView();
                if (e&&!this._showingEmptyView) {
                    this._showingEmptyView=!0;
                    var i = new t.Model;
                    this.addItemView(i, e, 0)
                }
            },
            closeEmptyView: function() {
                this._showingEmptyView && (this.closeChildren(), delete this._showingEmptyView)
            },
            getEmptyView: function() {
                return s.getOption(this, "emptyView")
            },
            getItemView: function() {
                var e = s.getOption(this, "itemView");
                return e || o("An `itemView` must be specified", "NoItemViewError"), e
            },
            addItemView: function(e, t, n) {
                var o = s.getOption(this, "itemViewOptions");
                i.isFunction(o) && (o = o.call(this, e, n));
                var r = this.buildItemView(e, t, o);
                return this.addChildViewEventForwarding(r), this.triggerMethod("before:item:added", r), this.children.add(r), this.renderItemView(r, n), this._isShown&&!this.isBuffering && s.triggerMethod.call(r, "show"), this.triggerMethod("after:item:added", r), r
            },
            addChildViewEventForwarding: function(e) {
                var t = s.getOption(this, "itemViewEventPrefix");
                this.listenTo(e, "all", function() {
                    var o = n(arguments), r = o[0], a = this.normalizeMethods(this.getItemEvents());
                    o[0] = t + ":" + r, o.splice(1, 0, e), "undefined" != typeof a && i.isFunction(a[r]) && a[r].apply(this, o), s.triggerMethod.apply(this, o)
                }, this)
            },
            getItemEvents: function() {
                return i.isFunction(this.itemEvents) ? this.itemEvents.call(this) : this.itemEvents
            },
            renderItemView: function(e, t) {
                e.render(), this.appendHtml(this, e, t)
            },
            buildItemView: function(e, t, n) {
                var o = i.extend({
                    model: e
                }, n);
                return new t(o)
            },
            removeItemView: function(e) {
                var t = this.children.findByModel(e);
                this.removeChildView(t), this.checkEmpty()
            },
            removeChildView: function(e) {
                e && (this.stopListening(e), e.close ? e.close() : e.remove && e.remove(), this.children.remove(e)), this.triggerMethod("item:removed", e)
            },
            isEmpty: function() {
                return !this.collection || 0 === this.collection.length
            },
            checkEmpty: function() {
                this.isEmpty(this.collection) && this.showEmptyView()
            },
            appendBuffer: function(e, t) {
                e.$el.append(t)
            },
            appendHtml: function(e, t) {
                e.isBuffering ? (e.elBuffer.appendChild(t.el), e._bufferedChildren.push(t)) : e.$el.append(t.el)
            },
            _initChildViewStorage: function() {
                this.children = new t.ChildViewContainer
            },
            close: function() {
                this.isClosed || (this.triggerMethod("collection:before:close"), this.closeChildren(), this.triggerMethod("collection:closed"), s.View.prototype.close.apply(this, n(arguments)))
            },
            closeChildren: function() {
                this.children.each(function(e) {
                    this.removeChildView(e)
                }, this), this.checkEmpty()
            }
        }), s.CompositeView = s.CollectionView.extend({
            constructor: function() {
                s.CollectionView.prototype.constructor.apply(this, n(arguments))
            },
            _initialEvents: function() {
                this.once("render", function() {
                    this.collection && (this.listenTo(this.collection, "add", this.addChildView, this), this.listenTo(this.collection, "remove", this.removeItemView, this), this.listenTo(this.collection, "reset", this._renderChildren, this))
                })
            },
            getItemView: function() {
                var e = s.getOption(this, "itemView") || this.constructor;
                return e || o("An `itemView` must be specified", "NoItemViewError"), e
            },
            serializeData: function() {
                var e = {};
                return this.model && (e = this.model.toJSON()), e
            },
            render: function() {
                this.isRendered=!0, this.isClosed=!1, this.resetItemViewContainer(), this.triggerBeforeRender();
                var e = this.renderModel();
                return this.$el.html(e), this.bindUIElements(), this.triggerMethod("composite:model:rendered"), this._renderChildren(), this.triggerMethod("composite:rendered"), this.triggerRendered(), this
            },
            _renderChildren: function() {
                this.isRendered && (this.triggerMethod("composite:collection:before:render"), s.CollectionView.prototype._renderChildren.call(this), this.triggerMethod("composite:collection:rendered"))
            },
            renderModel: function() {
                var e = {};
                e = this.serializeData(), e = this.mixinTemplateHelpers(e);
                var t = this.getTemplate();
                return s.Renderer.render(t, e)
            },
            appendBuffer: function(e, t) {
                var i = this.getItemViewContainer(e);
                i.append(t)
            },
            appendHtml: function(e, t) {
                if (e.isBuffering)
                    e.elBuffer.appendChild(t.el), e._bufferedChildren.push(t);
                else {
                    var i = this.getItemViewContainer(e);
                    i.append(t.el)
                }
            },
            getItemViewContainer: function(e) {
                if ("$itemViewContainer"in e)
                    return e.$itemViewContainer;
                var t, n = s.getOption(e, "itemViewContainer");
                if (n) {
                    var r = i.isFunction(n) ? n.call(this): n;
                    t = e.$(r), t.length <= 0 && o("The specified `itemViewContainer` was not found: " + e.itemViewContainer, "ItemViewContainerMissingError")
                } else 
                    t = e.$el;
                return e.$itemViewContainer = t, t
            },
            resetItemViewContainer: function() {
                this.$itemViewContainer && delete this.$itemViewContainer
            }
        }), s.Layout = s.ItemView.extend({
            regionType: s.Region,
            constructor: function(e) {
                e = e || {}, this._firstRender=!0, this._initializeRegions(e), s.ItemView.prototype.constructor.call(this, e)
            },
            render: function() {
                this.isClosed && this._initializeRegions(), this._firstRender ? this._firstRender=!1 : this.isClosed || this._reInitializeRegions();
                var e = Array.prototype.slice.apply(arguments), t = s.ItemView.prototype.render.apply(this, e);
                return t
            },
            close: function() {
                if (!this.isClosed) {
                    this.regionManager.close();
                    var e = Array.prototype.slice.apply(arguments);
                    s.ItemView.prototype.close.apply(this, e)
                }
            },
            addRegion: function(e, t) {
                var i = {};
                return i[e] = t, this._buildRegions(i)[e]
            },
            addRegions: function(e) {
                return this.regions = i.extend({}, this.regions, e), this._buildRegions(e)
            },
            removeRegion: function(e) {
                return delete this.regions[e], this.regionManager.removeRegion(e)
            },
            _buildRegions: function(e) {
                var t = this, i = {
                    regionType: s.getOption(this, "regionType"),
                    parentEl: function() {
                        return t.$el
                    }
                };
                return this.regionManager.addRegions(e, i)
            },
            _initializeRegions: function(e) {
                var t;
                this._initRegionManager(), t = i.isFunction(this.regions) ? this.regions(e) : this.regions || {}, this.addRegions(t)
            },
            _reInitializeRegions: function() {
                this.regionManager.closeRegions(), this.regionManager.each(function(e) {
                    e.reset()
                })
            },
            _initRegionManager: function() {
                this.regionManager = new s.RegionManager, this.listenTo(this.regionManager, "region:add", function(e, t) {
                    this[e] = t, this.trigger("region:add", e, t)
                }), this.listenTo(this.regionManager, "region:remove", function(e, t) {
                    delete this[e], this.trigger("region:remove", e, t)
                })
            }
        }), s.AppRouter = t.Router.extend({
            constructor: function(e) {
                t.Router.prototype.constructor.apply(this, n(arguments)), this.options = e || {};
                var i = s.getOption(this, "appRoutes"), o = this._getController();
                this.processAppRoutes(o, i)
            },
            appRoute: function(e, t) {
                var i = this._getController();
                this._addAppRoute(i, e, t)
            },
            processAppRoutes: function(e, t) {
                if (t) {
                    var n = i.keys(t).reverse();
                    i.each(n, function(i) {
                        this._addAppRoute(e, i, t[i])
                    }, this)
                }
            },
            _getController: function() {
                return s.getOption(this, "controller")
            },
            _addAppRoute: function(e, t, n) {
                var o = e[n];
                if (!o)
                    throw new Error("Method '" + n + "' was not found on the controller");
                this.route(t, n, i.bind(o, e))
            }
        }), s.Application = function(e) {
            this._initRegionManager(), this._initCallbacks = new s.Callbacks, this.vent = new t.Wreqr.EventAggregator, this.commands = new t.Wreqr.Commands, this.reqres = new t.Wreqr.RequestResponse, this.submodules = {}, i.extend(this, e), this.triggerMethod = s.triggerMethod
        }, i.extend(s.Application.prototype, t.Events, {
            execute: function() {
                var e = Array.prototype.slice.apply(arguments);
                this.commands.execute.apply(this.commands, e)
            },
            request: function() {
                var e = Array.prototype.slice.apply(arguments);
                return this.reqres.request.apply(this.reqres, e)
            },
            addInitializer: function(e) {
                this._initCallbacks.add(e)
            },
            start: function(e) {
                this.triggerMethod("initialize:before", e), this._initCallbacks.run(e, this), this.triggerMethod("initialize:after", e), this.triggerMethod("start", e)
            },
            addRegions: function(e) {
                return this._regionManager.addRegions(e)
            },
            closeRegions: function() {
                this._regionManager.closeRegions()
            },
            removeRegion: function(e) {
                this._regionManager.removeRegion(e)
            },
            getRegion: function(e) {
                return this._regionManager.get(e)
            },
            module: function(e,
            t) {
                var i = s.Module;
                t && (i = t.moduleClass || i);
                var o = n(arguments);
                return o.unshift(this),
                i.create.apply(i,
                o)
            }, _initRegionManager : function() {
                this._regionManager = new s.RegionManager, this.listenTo(this._regionManager, "region:add", function(e, t) {
                    this[e] = t
                }), this.listenTo(this._regionManager, "region:remove", function(e) {
                    delete this[e]
                })
            }
        }), s.Application.extend = s.extend, s.Module = function(e, t, n) {
            this.moduleName = e, this.options = i.extend({}, this.options, n), this.initialize = n.initialize || this.initialize, this.submodules = {}, this._setupInitializersAndFinalizers(), this.app = t, this.startWithParent=!0, this.triggerMethod = s.triggerMethod, i.isFunction(this.initialize) && this.initialize(this.options, e, t)
        }, s.Module.extend = s.extend, i.extend(s.Module.prototype, t.Events, {
            initialize: function() {},
            addInitializer: function(e) {
                this._initializerCallbacks.add(e)
            },
            addFinalizer: function(e) {
                this._finalizerCallbacks.add(e)
            },
            start: function(e) {
                this._isInitialized || (i.each(this.submodules, function(t) {
                    t.startWithParent && t.start(e)
                }), this.triggerMethod("before:start", e), this._initializerCallbacks.run(e, this), this._isInitialized=!0, this.triggerMethod("start", e))
            },
            stop: function() {
                this._isInitialized && (this._isInitialized=!1, s.triggerMethod.call(this, "before:stop"), i.each(this.submodules, function(e) {
                    e.stop()
                }), this._finalizerCallbacks.run(void 0, this), this._initializerCallbacks.reset(), this._finalizerCallbacks.reset(), s.triggerMethod.call(this, "stop"))
            },
            addDefinition: function(e, t) {
                this._runModuleDefinition(e, t)
            },
            _runModuleDefinition: function(e, n) {
                if (e) {
                    var o = i.flatten([this, this.app, t, s, s.$, i, n]);
                    e.apply(this, o)
                }
            },
            _setupInitializersAndFinalizers: function() {
                this._initializerCallbacks = new s.Callbacks, this._finalizerCallbacks = new s.Callbacks
            }
        }), i.extend(s.Module, {
            create: function(e, t, o) {
                var s = e, r = n(arguments);
                r.splice(0, 3), t = t.split(".");
                var a = t.length, l = [];
                return l[a - 1] = o, i.each(t, function(t, i) {
                    var n = s;
                    s = this._getModule(n, t, e, o), this._addModuleDefinition(n, s, l[i], r)
                }, this), s
            },
            _getModule: function(e, t, n, o) {
                var r = s.Module, a = i.extend({}, o);
                o && (r = o.moduleClass || r);
                var l = e[t];
                return l || (l = new r(t, n, a), e[t] = l, e.submodules[t] = l), l
            },
            _addModuleDefinition: function(e, t, n, o) {
                var s, r;
                i.isFunction(n) ? (s = n, r=!0) : i.isObject(n) ? (s = n.define, r = "undefined" != typeof n.startWithParent ? n.startWithParent : !0) : r=!0, s && t.addDefinition(s, o), t.startWithParent = t.startWithParent && r, t.startWithParent&&!t.startWithParentIsConfigured && (t.startWithParentIsConfigured=!0, e.addInitializer(function(e) {
                    t.startWithParent && t.start(e)
                }))
            }
        }), s
    }(this, t, e), t.Marionette
}), _wAMD.define("util/backbone-prevent", ["jquery", "underscore", "backbone"], function(e, t, i) {
    i.preventClose = function(n) {
        var o = this;
        if (!this.defer) {
            if (this.defer = e.Deferred(), i.preventClose.preventer) {
                var s = i.preventClose.preventer();
                s && s.promise ? this.defer = s : s ? this.defer.resolve() : this.defer.reject(), this.defer.fail(function() {
                    delete i.preventClose.preventer
                })
            } else 
                this.defer.reject();
            this.defer.always(function() {
                t.defer(function() {
                    delete o.defer
                })
            })
        }
        this.defer.fail(function() {
            n()
        })
    }, i.preventClose.ifModelChanged = function(i) {
        if (!i.hasChangedDeepSinceSavepoint())
            return !1;
        var n = e.Deferred();
        return t.defer(function() {
            confirm("Are you sure? You'll lose your current changes.") ? n.reject() : n.resolve()
        }), n.promise()
    };
    var n = i.View.prototype.constructor;
    i.View.prototype.constructor = function() {
        n.apply(this, arguments), this.preventClose && (i.preventClose.preventer = t.bind(this.preventClose, this))
    }
}), function(e, t) {
    "function" == typeof _wAMD.define && _wAMD.define.amd ? _wAMD.define("util/backbone-rpc", ["underscore", "backbone"], t) : t(e._, e.Backbone)
}(this, function(e, t) {
    var i = t.sync;
    e.extend(t, {
        sync: function(n, o, s) {
            if (!o.rpc)
                return i.apply(t, arguments);
            var r;
            switch (n) {
            case"delete":
                n = "destroy";
            case"create":
            case"update":
            case"patch":
                r = s.attrs || o.toJSON(s);
                break;
            case"read":
                if (o.rpc && "read" == n) {
                    s = e.extend({}, e.omit(s, "data"), s.data), r = e.extend({
                        filter: []
                    }, s);
                    var a = o.keydefs || o.model.prototype.keydefs;
                    if (a.PRIMARY)
                        for (var l = 0; l < a.PRIMARY.length; l++) {
                            var c = a.PRIMARY[l];
                            void 0 == o.get(c) || e.findWhere(r.filter, {
                                property: c
                            }) || r.filter.push({
                                property: c,
                                value: o.get(c)
                            })
                        }
                    }
                break;
            default:
                console.warn(n + " params not defined for this model ", o), r = s.attrs || o.toJSON()
            }
            if (o.rpc[n]) {
                var u = o.rpc[n](r);
                return u.done(function(e) {
                    return e.success===!1 ? s.error(e) : s.success(e)
                }), u.fail(function(e) {
                    return s.error(e)
                }), u
            }
            console.warn(n + " not defined for this model ", o)
        }
    })
}), function() {
    Weebly.setup_rpc = function(e) {
        var t = Weebly.ns(e.namespace), i = new jsonrpc.JsonRpc(e.url), n = function(e, t, n) {
            return function() {
                var o = Array.prototype.slice.call(arguments);
                if (o.length < t)
                    return console.error("Wrong number of args for " + e), void 0;
                n=!!n, o.unshift(e);
                var s = Weebly.jQuery.Deferred();
                return o.push({
                    success: s.resolve,
                    failure: s.reject,
                    scope: s,
                    secure: n
                }), i.call.apply(i, o), s
            }
        };
        for (var o in e.actions)
            if (e.actions.hasOwnProperty(o))
                for (var s = Weebly.ns(o, t), r = e.actions[o] || [], a = 0; a < r.length; a++) {
                    var l = r[a];
                    s[l.name] = n(o + "::" + l.name, l.len, l.secure, l.multiple)
                }
    }, Weebly.setup_model_rpc = function(e) {
        var t = Weebly.ns(e.rpc_namespace), i = Weebly.ns(e.model_namespace), n = Weebly.ns(e.collection_namespace), o = Weebly.ns(e.bootstrap_namespace);
        _.extend(i, e.models), _.extend(n, e.collections), _.extend(o, e.bootstrap);
        for (var s in n)
            t[s] && (n[s].rpc = t[s]);
        for (var s in i)
            if (t[s]) {
                i[s].rpc = t[s];
                var r = i[s].relations;
                if (r)
                    for (var a = 0; a < r.length; a++) {
                        var l = r[a];
                        n[l.relatedModel] && (l.collectionType = l.relatedModel + "Collection")
                    }
                }
        }
}(), _wAMD.define("backbone-all", ["backbone", "backbone-pageable", "backbone-validation", "backbone-relational", "backbone-marionette", "util/backbone-prevent", "util/backbone-rpc"], function(e) {
    var t = {}, i = e.RelationalModel.extend;
    return e.Relational.store.addModelScope(t), e.RelationalModel.extend = function(e) {
        var n = i.apply(this, arguments), o = e._class;
        return o && (o = o.split("."), o = o[o.length - 1], t[o] = n), n
    }, e
}), _wAMD.define("userhome/promo-overlay", ["backbone-all", "jquery"], function(e, t) {
    var i = e.Marionette.ItemView.extend({
        el: t("#promo-overlay"),
        ui: {
            form: "form",
            thanks: ".thanks",
            submit: "a.submit"
        },
        events: {
            "click @ui.submit": "handleSubmitClick",
            "submit @ui.form": "handleFormSubmit"
        },
        surveyUrl: "/survey.php?s=4",
        response: !1,
        ajax: "&ajax=1",
        initialize: function() {
            var e = this;
            e.bindUIElements(), e.el && (e.el.show(), new Weebly.Dialog("promo-overlay", {
                modal: !0,
                onClose: function() {
                    e.handleOverlayClose()
                }
            }).open())
        },
        handleOverlayClose: function() {
            var e = this;
            e.response || t.post(e.surveyUrl, "how_did_you_hear=closed" + e.ajax)
        },
        handleSubmitClick: function() {
            return this.ui.form.submit(), !1
        },
        handleFormSubmit: function() {
            var e = this;
            return t.post(e.surveyUrl, e.ui.form.serialize() + e.ajax, function() {
                e.ui.thanks.show(), e.ui.form.remove(), e.response=!0
            }), !1
        }
    });
    return i
}), _wAMD.define("jquery-ui/core", ["jquery"], function(e) {
    !function(e, t) {
        function i(t, i) {
            var o, s, r, a = t.nodeName.toLowerCase();
            return "area" === a ? (o = t.parentNode, s = o.name, t.href && s && "map" === o.nodeName.toLowerCase() ? (r = e("img[usemap=#" + s + "]")[0], !!r && n(r)) : !1) : (/input|select|textarea|button|object/.test(a)?!t.disabled : "a" === a ? t.href || i : i) && n(t)
        }
        function n(t) {
            return e.expr.filters.visible(t)&&!e(t).parents().addBack().filter(function() {
                return "hidden" === e.css(this, "visibility")
            }).length
        }
        var o = 0, s = /^ui-id-\d+$/;
        e.ui = e.ui || {}, e.extend(e.ui, {
            version: "1.10.2",
            keyCode: {
                BACKSPACE: 8,
                COMMA: 188,
                DELETE: 46,
                DOWN: 40,
                END: 35,
                ENTER: 13,
                ESCAPE: 27,
                HOME: 36,
                LEFT: 37,
                NUMPAD_ADD: 107,
                NUMPAD_DECIMAL: 110,
                NUMPAD_DIVIDE: 111,
                NUMPAD_ENTER: 108,
                NUMPAD_MULTIPLY: 106,
                NUMPAD_SUBTRACT: 109,
                PAGE_DOWN: 34,
                PAGE_UP: 33,
                PERIOD: 190,
                RIGHT: 39,
                SPACE: 32,
                TAB: 9,
                UP: 38
            }
        }), e.fn.extend({
            focus: function(t) {
                return function(i, n) {
                    return "number" == typeof i ? this.each(function() {
                        var t = this;
                        setTimeout(function() {
                            e(t).focus(), n && n.call(t)
                        }, i)
                    }) : t.apply(this, arguments)
                }
            }(e.fn.focus),
            scrollParent: function() {
                var t;
                return t = e.ui.ie && /(static|relative)/.test(this.css("position")) || /absolute/.test(this.css("position")) ? this.parents().filter(function() {
                    return /(relative|absolute|fixed)/.test(e.css(this, "position")) && /(auto|scroll)/.test(e.css(this, "overflow") + e.css(this, "overflow-y") + e.css(this, "overflow-x"))
                }).eq(0) : this.parents().filter(function() {
                    return /(auto|scroll)/.test(e.css(this, "overflow") + e.css(this, "overflow-y") + e.css(this, "overflow-x"))
                }).eq(0), /fixed/.test(this.css("position")) ||!t.length ? e(document) : t
            },
            zIndex: function(i) {
                if (i !== t)
                    return this.css("zIndex", i);
                if (this.length)
                    for (var n, o, s = e(this[0]); s.length && s[0] !== document;) {
                        if (n = s.css("position"), ("absolute" === n || "relative" === n || "fixed" === n) && (o = parseInt(s.css("zIndex"), 10), !isNaN(o) && 0 !== o))
                            return o;
                            s = s.parent()
                    }
                return 0
            },
            uniqueId: function() {
                return this.each(function() {
                    this.id || (this.id = "ui-id-" + ++o)
                })
            },
            removeUniqueId: function() {
                return this.each(function() {
                    s.test(this.id) && e(this).removeAttr("id")
                })
            }
        }), e.extend(e.expr[":"], {
            data: e.expr.createPseudo ? e.expr.createPseudo(function(t) {
                return function(i) {
                    return !!e.data(i, t)
                }
            }): function(t, i, n) {
                return !!e.data(t, n[3])
            },
            focusable: function(t) {
                return i(t, !isNaN(e.attr(t, "tabindex")))
            },
            tabbable: function(t) {
                var n = e.attr(t, "tabindex"), o = isNaN(n);
                return (o || n >= 0) && i(t, !o)
            }
        }), e("<a>").outerWidth(1).jquery || e.each(["Width", "Height"], function(i, n) {
            function o(t, i, n, o) {
                return e.each(s, function() {
                    i -= parseFloat(e.css(t, "padding" + this)) || 0, n && (i -= parseFloat(e.css(t, "border" + this + "Width")) || 0), o && (i -= parseFloat(e.css(t, "margin" + this)) || 0)
                }), i
            }
            var s = "Width" === n ? ["Left", "Right"]: ["Top", "Bottom"], r = n.toLowerCase(), a = {
                innerWidth: e.fn.innerWidth,
                innerHeight: e.fn.innerHeight,
                outerWidth: e.fn.outerWidth,
                outerHeight: e.fn.outerHeight
            };
            e.fn["inner" + n] = function(i) {
                return i === t ? a["inner" + n].call(this) : this.each(function() {
                    e(this).css(r, o(this, i) + "px")
                })
            }, e.fn["outer" + n] = function(t, i) {
                return "number" != typeof t ? a["outer" + n].call(this, t) : this.each(function() {
                    e(this).css(r, o(this, t, !0, i) + "px")
                })
            }
        }), e.fn.addBack || (e.fn.addBack = function(e) {
            return this.add(null == e ? this.prevObject : this.prevObject.filter(e))
        }), e("<a>").data("a-b", "a").removeData("a-b").data("a-b") && (e.fn.removeData = function(t) {
            return function(i) {
                return arguments.length ? t.call(this, e.camelCase(i)) : t.call(this)
            }
        }(e.fn.removeData)), e.ui.ie=!!/msie [\w.]+/.exec(navigator.userAgent.toLowerCase()), e.support.selectstart = "onselectstart"in document.createElement("div"), e.fn.extend({
            disableSelection: function() {
                return this.bind((e.support.selectstart ? "selectstart" : "mousedown") + ".ui-disableSelection", function(e) {
                    e.preventDefault()
                })
            },
            enableSelection: function() {
                return this.unbind(".ui-disableSelection")
            }
        }), e.extend(e.ui, {
            plugin: {
                add: function(t, i, n) {
                    var o, s = e.ui[t].prototype;
                    for (o in n)
                        s.plugins[o] = s.plugins[o] || [], s.plugins[o].push([i, n[o]])
                },
                call: function(e, t, i) {
                    var n, o = e.plugins[t];
                    if (o && e.element[0].parentNode && 11 !== e.element[0].parentNode.nodeType)
                        for (n = 0; n < o.length; n++)
                            e.options[o[n][0]] && o[n][1].apply(e.element, i)
                }
            },
            hasScroll: function(t, i) {
                if ("hidden" === e(t).css("overflow"))
                    return !1;
                var n = i && "left" === i ? "scrollLeft": "scrollTop", o=!1;
                return t[n] > 0?!0 : (t[n] = 1, o = t[n] > 0, t[n] = 0, o)
            }
        })
    }(e)
}), _wAMD.define("jquery-ui/widget", ["jquery"], function(e) {
    !function(e, t) {
        var i = 0, n = Array.prototype.slice, o = e.cleanData;
        e.cleanData = function(t) {
            for (var i, n = 0; null != (i = t[n]); n++)
                try {
                    e(i).triggerHandler("remove")
            } catch (s) {}
            o(t)
        }, e.widget = function(t, i, n) {
            var o, s, r, a, l = {}, c = t.split(".")[0];
            t = t.split(".")[1], o = c + "-" + t, n || (n = i, i = e.Widget), e.expr[":"][o.toLowerCase()] = function(t) {
                return !!e.data(t, o)
            }, e[c] = e[c] || {}, s = e[c][t], r = e[c][t] = function(e, t) {
                return this._createWidget ? (arguments.length && this._createWidget(e, t), void 0) : new r(e, t)
            }, e.extend(r, s, {
                version: n.version,
                _proto: e.extend({}, n),
                _childConstructors: []
            }), a = new i, a.options = e.widget.extend({}, a.options), e.each(n, function(t, n) {
                return e.isFunction(n) ? (l[t] = function() {
                    var e = function() {
                        return i.prototype[t].apply(this, arguments)
                    }, o = function(e) {
                        return i.prototype[t].apply(this, e)
                    };
                    return function() {
                        var t, i = this._super, s = this._superApply;
                        return this._super = e, this._superApply = o, t = n.apply(this, arguments), this._super = i, this._superApply = s, t
                    }
                }(), void 0) : (l[t] = n, void 0)
            }), r.prototype = e.widget.extend(a, {
                widgetEventPrefix: s ? a.widgetEventPrefix: t
            }, l, {
                constructor: r,
                namespace: c,
                widgetName: t,
                widgetFullName: o
            }), s ? (e.each(s._childConstructors, function(t, i) {
                var n = i.prototype;
                e.widget(n.namespace + "." + n.widgetName, r, i._proto)
            }), delete s._childConstructors) : i._childConstructors.push(r), e.widget.bridge(t, r)
        }, e.widget.extend = function(i) {
            for (var o, s, r = n.call(arguments, 1), a = 0, l = r.length; l > a; a++)
                for (o in r[a])
                    s = r[a][o], r[a].hasOwnProperty(o) && s !== t && (i[o] = e.isPlainObject(s) ? e.isPlainObject(i[o]) ? e.widget.extend({}, i[o], s) : e.widget.extend({}, s) : s);
            return i
        }, e.widget.bridge = function(i, o) {
            var s = o.prototype.widgetFullName || i;
            e.fn[i] = function(r) {
                var a = "string" == typeof r, l = n.call(arguments, 1), c = this;
                return r=!a && l.length ? e.widget.extend.apply(null, [r].concat(l)) : r, a ? this.each(function() {
                    var n, o = e.data(this, s);
                    return o ? e.isFunction(o[r]) && "_" !== r.charAt(0) ? (n = o[r].apply(o, l), n !== o && n !== t ? (c = n && n.jquery ? c.pushStack(n.get()) : n, !1) : void 0) : e.error("no such method '" + r + "' for " + i + " widget instance") : e.error("cannot call methods on " + i + " prior to initialization; " + "attempted to call method '" + r + "'")
                }) : this.each(function() {
                    var t = e.data(this, s);
                    t ? t.option(r || {})._init() : e.data(this, s, new o(r, this))
                }), c
            }
        }, e.Widget = function() {}, e.Widget._childConstructors = [], e.Widget.prototype = {
            widgetName: "widget",
            widgetEventPrefix: "",
            defaultElement: "<div>",
            options: {
                disabled: !1,
                create: null
            },
            _createWidget: function(t, n) {
                n = e(n || this.defaultElement || this)[0], this.element = e(n), this.uuid = i++, this.eventNamespace = "." + this.widgetName + this.uuid, this.options = e.widget.extend({}, this.options, this._getCreateOptions(), t), this.bindings = e(), this.hoverable = e(), this.focusable = e(), n !== this && (e.data(n, this.widgetFullName, this), this._on(!0, this.element, {
                    remove: function(e) {
                        e.target === n && this.destroy()
                    }
                }), this.document = e(n.style ? n.ownerDocument : n.document || n), this.window = e(this.document[0].defaultView || this.document[0].parentWindow)), this._create(), this._trigger("create", null, this._getCreateEventData()), this._init()
            },
            _getCreateOptions: e.noop,
            _getCreateEventData: e.noop,
            _create: e.noop,
            _init: e.noop,
            destroy: function() {
                this._destroy(), this.element.unbind(this.eventNamespace).removeData(this.widgetName).removeData(this.widgetFullName).removeData(e.camelCase(this.widgetFullName)), this.widget().unbind(this.eventNamespace).removeAttr("aria-disabled").removeClass(this.widgetFullName + "-disabled " + "ui-state-disabled"), this.bindings.unbind(this.eventNamespace), this.hoverable.removeClass("ui-state-hover"), this.focusable.removeClass("ui-state-focus")
            },
            _destroy: e.noop,
            widget: function() {
                return this.element
            },
            option: function(i, n) {
                var o, s, r, a = i;
                if (0 === arguments.length)
                    return e.widget.extend({}, this.options);
                if ("string" == typeof i)
                    if (a = {}, o = i.split("."), i = o.shift(), o.length) {
                        for (s = a[i] = e.widget.extend({}, this.options[i]), r = 0; r < o.length - 1; r++)
                            s[o[r]] = s[o[r]] || {}, s = s[o[r]];
                            if (i = o.pop(), n === t)
                                return s[i] === t ? null : s[i];
                                s[i] = n
                    } else {
                        if (n === t)
                            return this.options[i] === t ? null : this.options[i];
                            a[i] = n
                    }
                return this._setOptions(a), this
            },
            _setOptions: function(e) {
                var t;
                for (t in e)
                    this._setOption(t, e[t]);
                return this
            },
            _setOption: function(e, t) {
                return this.options[e] = t, "disabled" === e && (this.widget().toggleClass(this.widgetFullName + "-disabled ui-state-disabled", !!t).attr("aria-disabled", t), this.hoverable.removeClass("ui-state-hover"), this.focusable.removeClass("ui-state-focus")), this
            },
            enable: function() {
                return this._setOption("disabled", !1)
            },
            disable: function() {
                return this._setOption("disabled", !0)
            },
            _on: function(t, i, n) {
                var o, s = this;
                "boolean" != typeof t && (n = i, i = t, t=!1), n ? (i = o = e(i), this.bindings = this.bindings.add(i)) : (n = i, i = this.element, o = this.widget()), e.each(n, function(n, r) {
                    function a() {
                        return t || s.options.disabled!==!0&&!e(this).hasClass("ui-state-disabled") ? ("string" == typeof r ? s[r] : r).apply(s, arguments) : void 0
                    }
                    "string" != typeof r && (a.guid = r.guid = r.guid || a.guid || e.guid++);
                    var l = n.match(/^(\w+)\s*(.*)$/), c = l[1] + s.eventNamespace, u = l[2];
                    u ? o.delegate(u, c, a) : i.bind(c, a)
                })
            },
            _off: function(e, t) {
                t = (t || "").split(" ").join(this.eventNamespace + " ") + this.eventNamespace, e.unbind(t).undelegate(t)
            },
            _delay: function(e, t) {
                function i() {
                    return ("string" == typeof e ? n[e] : e).apply(n, arguments)
                }
                var n = this;
                return setTimeout(i, t || 0)
            },
            _hoverable: function(t) {
                this.hoverable = this.hoverable.add(t), this._on(t, {
                    mouseenter: function(t) {
                        e(t.currentTarget).addClass("ui-state-hover")
                    },
                    mouseleave: function(t) {
                        e(t.currentTarget).removeClass("ui-state-hover")
                    }
                })
            },
            _focusable: function(t) {
                this.focusable = this.focusable.add(t), this._on(t, {
                    focusin: function(t) {
                        e(t.currentTarget).addClass("ui-state-focus")
                    },
                    focusout: function(t) {
                        e(t.currentTarget).removeClass("ui-state-focus")
                    }
                })
            },
            _trigger: function(t, i, n) {
                var o, s, r = this.options[t];
                if (n = n || {}, i = e.Event(i), i.type = (t === this.widgetEventPrefix ? t : this.widgetEventPrefix + t).toLowerCase(), i.target = this.element[0], s = i.originalEvent)
                    for (o in s)
                        o in i || (i[o] = s[o]);
                return this.element.trigger(i, n), !(e.isFunction(r) && r.apply(this.element[0], [i].concat(n))===!1 || i.isDefaultPrevented())
            }
        }, e.each({
            show: "fadeIn",
            hide: "fadeOut"
        }, function(t, i) {
            e.Widget.prototype["_" + t] = function(n, o, s) {
                "string" == typeof o && (o = {
                    effect: o
                });
                var r, a = o ? o===!0 || "number" == typeof o ? i: o.effect || i: t;
                o = o || {}, "number" == typeof o && (o = {
                    duration: o
                }), r=!e.isEmptyObject(o), o.complete = s, o.delay && n.delay(o.delay), r && e.effects && e.effects.effect[a] ? n[t](o) : a !== t && n[a] ? n[a](o.duration, o.easing, s) : n.queue(function(i) {
                    e(this)[t](), s && s.call(n[0]), i()
                })
            }
        })
    }(e)
}), _wAMD.define("jquery-ui/button", ["jquery", "./core", "./widget"], function(e) {
    !function(e) {
        var t, i, n, o, s = "ui-button ui-widget ui-state-default ui-corner-all", r = "ui-state-hover ui-state-active ", a = "ui-button-icons-only ui-button-icon-only ui-button-text-icons ui-button-text-icon-primary ui-button-text-icon-secondary ui-button-text-only", l = function() {
            var t = e(this).find(":ui-button");
            setTimeout(function() {
                t.button("refresh")
            }, 1)
        }, c = function(t) {
            var i = t.name, n = t.form, o = e([]);
            return i && (i = i.replace(/'/g, "\\'"), o = n ? e(n).find("[name='" + i + "']") : e("[name='" + i + "']", t.ownerDocument).filter(function() {
                return !this.form
            })), o
        };
        e.widget("ui.button", {
            version: "1.10.2",
            defaultElement: "<button>",
            options: {
                disabled: null,
                text: !0,
                label: null,
                icons: {
                    primary: null,
                    secondary: null
                }
            },
            _create: function() {
                this.element.closest("form").unbind("reset" + this.eventNamespace).bind("reset" + this.eventNamespace, l), "boolean" != typeof this.options.disabled ? this.options.disabled=!!this.element.prop("disabled") : this.element.prop("disabled", this.options.disabled), this._determineButtonType(), this.hasTitle=!!this.buttonElement.attr("title");
                var r = this, a = this.options, u = "checkbox" === this.type || "radio" === this.type, d = u ? "": "ui-state-active", h = "ui-state-focus";
                null === a.label && (a.label = "input" === this.type ? this.buttonElement.val() : this.buttonElement.html()), this._hoverable(this.buttonElement), this.buttonElement.addClass(s).attr("role", "button").bind("mouseenter" + this.eventNamespace, function() {
                    a.disabled || this === t && e(this).addClass("ui-state-active")
                }).bind("mouseleave" + this.eventNamespace, function() {
                    a.disabled || e(this).removeClass(d)
                }).bind("click" + this.eventNamespace, function(e) {
                    a.disabled && (e.preventDefault(), e.stopImmediatePropagation())
                }), this.element.bind("focus" + this.eventNamespace, function() {
                    r.buttonElement.addClass(h)
                }).bind("blur" + this.eventNamespace, function() {
                    r.buttonElement.removeClass(h)
                }), u && (this.element.bind("change" + this.eventNamespace, function() {
                    o || r.refresh()
                }), this.buttonElement.bind("mousedown" + this.eventNamespace, function(e) {
                    a.disabled || (o=!1, i = e.pageX, n = e.pageY)
                }).bind("mouseup" + this.eventNamespace, function(e) {
                    a.disabled || (i !== e.pageX || n !== e.pageY) && (o=!0)
                })), "checkbox" === this.type ? this.buttonElement.bind("click" + this.eventNamespace, function() {
                    return a.disabled || o?!1 : void 0
                }) : "radio" === this.type ? this.buttonElement.bind("click" + this.eventNamespace, function() {
                    if (a.disabled || o)
                        return !1;
                    e(this).addClass("ui-state-active"), r.buttonElement.attr("aria-pressed", "true");
                    var t = r.element[0];
                    c(t).not(t).map(function() {
                        return e(this).button("widget")[0]
                    }).removeClass("ui-state-active").attr("aria-pressed", "false")
                }) : (this.buttonElement.bind("mousedown" + this.eventNamespace, function() {
                    return a.disabled?!1 : (e(this).addClass("ui-state-active"), t = this, r.document.one("mouseup", function() {
                        t = null
                    }), void 0)
                }).bind("mouseup" + this.eventNamespace, function() {
                    return a.disabled?!1 : (e(this).removeClass("ui-state-active"), void 0)
                }).bind("keydown" + this.eventNamespace, function(t) {
                    return a.disabled?!1 : ((t.keyCode === e.ui.keyCode.SPACE || t.keyCode === e.ui.keyCode.ENTER) && e(this).addClass("ui-state-active"), void 0)
                }).bind("keyup" + this.eventNamespace + " blur" + this.eventNamespace, function() {
                    e(this).removeClass("ui-state-active")
                }), this.buttonElement.is("a") && this.buttonElement.keyup(function(t) {
                    t.keyCode === e.ui.keyCode.SPACE && e(this).click()
                })), this._setOption("disabled", a.disabled), this._resetButton()
            },
            _determineButtonType: function() {
                var e, t, i;
                this.type = this.element.is("[type=checkbox]") ? "checkbox" : this.element.is("[type=radio]") ? "radio" : this.element.is("input") ? "input" : "button", "checkbox" === this.type || "radio" === this.type ? (e = this.element.parents().last(), t = "label[for='" + this.element.attr("id") + "']", this.buttonElement = e.find(t), this.buttonElement.length || (e = e.length ? e.siblings() : this.element.siblings(), this.buttonElement = e.filter(t), this.buttonElement.length || (this.buttonElement = e.find(t))), this.element.addClass("ui-helper-hidden-accessible"), i = this.element.is(":checked"), i && this.buttonElement.addClass("ui-state-active"), this.buttonElement.prop("aria-pressed", i)) : this.buttonElement = this.element
            },
            widget: function() {
                return this.buttonElement
            },
            _destroy: function() {
                this.element.removeClass("ui-helper-hidden-accessible"), this.buttonElement.removeClass(s + " " + r + " " + a).removeAttr("role").removeAttr("aria-pressed").html(this.buttonElement.find(".ui-button-text").html()), this.hasTitle || this.buttonElement.removeAttr("title")
            },
            _setOption: function(e, t) {
                return this._super(e, t), "disabled" === e ? (t ? this.element.prop("disabled", !0) : this.element.prop("disabled", !1), void 0) : (this._resetButton(), void 0)
            },
            refresh: function() {
                var t = this.element.is("input, button") ? this.element.is(":disabled"): this.element.hasClass("ui-button-disabled");
                t !== this.options.disabled && this._setOption("disabled", t), "radio" === this.type ? c(this.element[0]).each(function() {
                    e(this).is(":checked") ? e(this).button("widget").addClass("ui-state-active").attr("aria-pressed", "true") : e(this).button("widget").removeClass("ui-state-active").attr("aria-pressed", "false")
                }) : "checkbox" === this.type && (this.element.is(":checked") ? this.buttonElement.addClass("ui-state-active").attr("aria-pressed", "true") : this.buttonElement.removeClass("ui-state-active").attr("aria-pressed", "false"))
            },
            _resetButton: function() {
                if ("input" === this.type)
                    return this.options.label && this.element.val(this.options.label), void 0;
                var t = this.buttonElement.removeClass(a), i = e("<span></span>", this.document[0]).addClass("ui-button-text").html(this.options.label).appendTo(t.empty()).text(), n = this.options.icons, o = n.primary && n.secondary, s = [];
                n.primary || n.secondary ? (this.options.text && s.push("ui-button-text-icon" + (o ? "s" : n.primary ? "-primary" : "-secondary")), n.primary && t.prepend("<span class='ui-button-icon-primary ui-icon " + n.primary + "'></span>"), n.secondary && t.append("<span class='ui-button-icon-secondary ui-icon " + n.secondary + "'></span>"), this.options.text || (s.push(o ? "ui-button-icons-only" : "ui-button-icon-only"), this.hasTitle || t.attr("title", e.trim(i)))) : s.push("ui-button-text-only"), t.addClass(s.join(" "))
            }
        }), e.widget("ui.buttonset", {
            version: "1.10.2",
            options: {
                items: "button, input[type=button], input[type=submit], input[type=reset], input[type=checkbox], input[type=radio], a, :data(ui-button)"
            },
            _create: function() {
                this.element.addClass("ui-buttonset")
            },
            _init: function() {
                this.refresh()
            },
            _setOption: function(e, t) {
                "disabled" === e && this.buttons.button("option", e, t), this._super(e, t)
            },
            refresh: function() {
                var t = "rtl" === this.element.css("direction");
                this.buttons = this.element.find(this.options.items).filter(":ui-button").button("refresh").end().not(":ui-button").button().end().map(function() {
                    return e(this).button("widget")[0]
                }).removeClass("ui-corner-all ui-corner-left ui-corner-right").filter(":first").addClass(t ? "ui-corner-right" : "ui-corner-left").end().filter(":last").addClass(t ? "ui-corner-left" : "ui-corner-right").end().end()
            },
            _destroy: function() {
                this.element.removeClass("ui-buttonset"), this.buttons.map(function() {
                    return e(this).button("widget")[0]
                }).removeClass("ui-corner-left ui-corner-right").end().button("destroy")
            }
        })
    }(e)
}), _wAMD.define("jquery-ui/mouse", ["jquery", "./widget"], function(e) {
    !function(e) {
        var t=!1;
        e(document).mouseup(function() {
            t=!1
        }), e.widget("ui.mouse", {
            version: "1.10.2",
            options: {
                cancel: "input,textarea,button,select,option",
                distance: 1,
                delay: 0
            },
            _mouseInit: function() {
                var t = this;
                this.element.bind("mousedown." + this.widgetName, function(e) {
                    return t._mouseDown(e)
                }).bind("click." + this.widgetName, function(i) {
                    return !0 === e.data(i.target, t.widgetName + ".preventClickEvent") ? (e.removeData(i.target, t.widgetName + ".preventClickEvent"), i.stopImmediatePropagation(), !1) : void 0
                }), this.started=!1
            },
            _mouseDestroy: function() {
                this.element.unbind("." + this.widgetName), this._mouseMoveDelegate && e(document).unbind("mousemove." + this.widgetName, this._mouseMoveDelegate).unbind("mouseup." + this.widgetName, this._mouseUpDelegate)
            },
            _mouseDown: function(i) {
                if (!t) {
                    this._mouseStarted && this._mouseUp(i), this._mouseDownEvent = i;
                    var n = this, o = 1 === i.which, s = "string" == typeof this.options.cancel && i.target.nodeName ? e(i.target).closest(this.options.cancel).length: !1;
                    return o&&!s && this._mouseCapture(i) ? (this.mouseDelayMet=!this.options.delay, this.mouseDelayMet || (this._mouseDelayTimer = setTimeout(function() {
                        n.mouseDelayMet=!0
                    }, this.options.delay)), this._mouseDistanceMet(i) && this._mouseDelayMet(i) && (this._mouseStarted = this._mouseStart(i)!==!1, !this._mouseStarted) ? (i.preventDefault(), !0) : (!0 === e.data(i.target, this.widgetName + ".preventClickEvent") && e.removeData(i.target, this.widgetName + ".preventClickEvent"), this._mouseMoveDelegate = function(e) {
                        return n._mouseMove(e)
                    }, this._mouseUpDelegate = function(e) {
                        return n._mouseUp(e)
                    }, e(document).bind("mousemove." + this.widgetName, this._mouseMoveDelegate).bind("mouseup." + this.widgetName, this._mouseUpDelegate), i.preventDefault(), t=!0, !0)) : !0
                }
            },
            _mouseMove: function(t) {
                return e.ui.ie && (!document.documentMode || document.documentMode < 9)&&!t.button ? this._mouseUp(t) : this._mouseStarted ? (this._mouseDrag(t), t.preventDefault()) : (this._mouseDistanceMet(t) && this._mouseDelayMet(t) && (this._mouseStarted = this._mouseStart(this._mouseDownEvent, t)!==!1, this._mouseStarted ? this._mouseDrag(t) : this._mouseUp(t)), !this._mouseStarted)
            },
            _mouseUp: function(t) {
                return e(document).unbind("mousemove." + this.widgetName, this._mouseMoveDelegate).unbind("mouseup." + this.widgetName, this._mouseUpDelegate), this._mouseStarted && (this._mouseStarted=!1, t.target === this._mouseDownEvent.target && e.data(t.target, this.widgetName + ".preventClickEvent", !0), this._mouseStop(t)), !1
            },
            _mouseDistanceMet: function(e) {
                return Math.max(Math.abs(this._mouseDownEvent.pageX - e.pageX), Math.abs(this._mouseDownEvent.pageY - e.pageY)) >= this.options.distance
            },
            _mouseDelayMet: function() {
                return this.mouseDelayMet
            },
            _mouseStart: function() {},
            _mouseDrag: function() {},
            _mouseStop: function() {},
            _mouseCapture: function() {
                return !0
            }
        })
    }(e)
}), _wAMD.define("jquery-ui/draggable", ["jquery", "./core", "./mouse", "./widget"], function(e) {
    !function(e) {
        e.widget("ui.draggable", e.ui.mouse, {
            version: "1.10.2",
            widgetEventPrefix: "drag",
            options: {
                addClasses: !0,
                appendTo: "parent",
                axis: !1,
                connectToSortable: !1,
                containment: !1,
                cursor: "auto",
                cursorAt: !1,
                grid: !1,
                handle: !1,
                helper: "original",
                iframeFix: !1,
                opacity: !1,
                refreshPositions: !1,
                revert: !1,
                revertDuration: 500,
                scope: "default",
                scroll: !0,
                scrollSensitivity: 20,
                scrollSpeed: 20,
                snap: !1,
                snapMode: "both",
                snapTolerance: 20,
                stack: !1,
                zIndex: !1,
                drag: null,
                start: null,
                stop: null
            },
            _create: function() {
                "original" !== this.options.helper || /^(?:r|a|f)/.test(this.element.css("position")) || (this.element[0].style.position = "relative"), this.options.addClasses && this.element.addClass("ui-draggable"), this.options.disabled && this.element.addClass("ui-draggable-disabled"), this._mouseInit()
            },
            _destroy: function() {
                this.element.removeClass("ui-draggable ui-draggable-dragging ui-draggable-disabled"), this._mouseDestroy()
            },
            _mouseCapture: function(t) {
                var i = this.options;
                return this.helper || i.disabled || e(t.target).closest(".ui-resizable-handle").length > 0?!1 : (this.handle = this._getHandle(t), this.handle ? (e(i.iframeFix===!0 ? "iframe" : i.iframeFix).each(function() {
                    e("<div class='ui-draggable-iframeFix' style='background: #fff;'></div>").css({
                        width: this.offsetWidth + "px",
                        height: this.offsetHeight + "px",
                        position: "absolute",
                        opacity: "0.001",
                        zIndex: 1e3
                    }).css(e(this).offset()).appendTo("body")
                }), !0) : !1)
            },
            _mouseStart: function(t) {
                var i = this.options;
                return this.helper = this._createHelper(t), this.helper.addClass("ui-draggable-dragging"), this._cacheHelperProportions(), e.ui.ddmanager && (e.ui.ddmanager.current = this), this._cacheMargins(), this.cssPosition = this.helper.css("position"), this.scrollParent = this.helper.scrollParent(), this.offset = this.positionAbs = this.element.offset(), this.offset = {
                    top: this.offset.top - this.margins.top,
                    left: this.offset.left - this.margins.left
                }, e.extend(this.offset, {
                    click: {
                        left: t.pageX - this.offset.left,
                        top: t.pageY - this.offset.top
                    },
                    parent: this._getParentOffset(),
                    relative: this._getRelativeOffset()
                }), this.originalPosition = this.position = this._generatePosition(t), this.originalPageX = t.pageX, this.originalPageY = t.pageY, i.cursorAt && this._adjustOffsetFromHelper(i.cursorAt), i.containment && this._setContainment(), this._trigger("start", t)===!1 ? (this._clear(), !1) : (this._cacheHelperProportions(), e.ui.ddmanager&&!i.dropBehaviour && e.ui.ddmanager.prepareOffsets(this, t), this._mouseDrag(t, !0), e.ui.ddmanager && e.ui.ddmanager.dragStart(this, t), !0)
            },
            _mouseDrag: function(t, i) {
                if (this.position = this._generatePosition(t), this.positionAbs = this._convertPositionTo("absolute"), !i) {
                    var n = this._uiHash();
                    if (this._trigger("drag", t, n)===!1)
                        return this._mouseUp({}), !1;
                    this.position = n.position
                }
                return this.options.axis && "y" === this.options.axis || (this.helper[0].style.left = this.position.left + "px"), this.options.axis && "x" === this.options.axis || (this.helper[0].style.top = this.position.top + "px"), e.ui.ddmanager && e.ui.ddmanager.drag(this, t), !1
            },
            _mouseStop: function(t) {
                var i, n = this, o=!1, s=!1;
                for (e.ui.ddmanager&&!this.options.dropBehaviour && (s = e.ui.ddmanager.drop(this, t)), this.dropped && (s = this.dropped, this.dropped=!1), i = this.element[0]; i && (i = i.parentNode);)
                    i === document && (o=!0);
                return o || "original" !== this.options.helper ? ("invalid" === this.options.revert&&!s || "valid" === this.options.revert && s || this.options.revert===!0 || e.isFunction(this.options.revert) && this.options.revert.call(this.element, s) ? e(this.helper).animate(this.originalPosition, parseInt(this.options.revertDuration, 10), function() {
                    n._trigger("stop", t)!==!1 && n._clear()
                }) : this._trigger("stop", t)!==!1 && this._clear(), !1) : !1
            },
            _mouseUp: function(t) {
                return e("div.ui-draggable-iframeFix").each(function() {
                    this.parentNode.removeChild(this)
                }), e.ui.ddmanager && e.ui.ddmanager.dragStop(this, t), e.ui.mouse.prototype._mouseUp.call(this, t)
            },
            cancel: function() {
                return this.helper.is(".ui-draggable-dragging") ? this._mouseUp({}) : this._clear(), this
            },
            _getHandle: function(t) {
                return this.options.handle?!!e(t.target).closest(this.element.find(this.options.handle)).length : !0
            },
            _createHelper: function(t) {
                var i = this.options, n = e.isFunction(i.helper) ? e(i.helper.apply(this.element[0], [t])): "clone" === i.helper ? this.element.clone().removeAttr("id"): this.element;
                return n.parents("body").length || n.appendTo("parent" === i.appendTo ? this.element[0].parentNode : i.appendTo), n[0] === this.element[0] || /(fixed|absolute)/.test(n.css("position")) || n.css("position", "absolute"), n
            },
            _adjustOffsetFromHelper: function(t) {
                "string" == typeof t && (t = t.split(" ")), e.isArray(t) && (t = {
                    left: + t[0],
                    top: + t[1] || 0
                }), "left"in t && (this.offset.click.left = t.left + this.margins.left), "right"in t && (this.offset.click.left = this.helperProportions.width - t.right + this.margins.left), "top"in t && (this.offset.click.top = t.top + this.margins.top), "bottom"in t && (this.offset.click.top = this.helperProportions.height - t.bottom + this.margins.top)
            },
            _getParentOffset: function() {
                this.offsetParent = this.helper.offsetParent();
                var t = this.offsetParent.offset();
                return "absolute" === this.cssPosition && this.scrollParent[0] !== document && e.contains(this.scrollParent[0], this.offsetParent[0]) && (t.left += this.scrollParent.scrollLeft(), t.top += this.scrollParent.scrollTop()), (this.offsetParent[0] === document.body || this.offsetParent[0].tagName && "html" === this.offsetParent[0].tagName.toLowerCase() && e.ui.ie) && (t = {
                    top: 0,
                    left: 0
                }), {
                    top: t.top + (parseInt(this.offsetParent.css("borderTopWidth"), 10) || 0),
                    left: t.left + (parseInt(this.offsetParent.css("borderLeftWidth"), 10) || 0)
                }
            },
            _getRelativeOffset: function() {
                if ("relative" === this.cssPosition) {
                    var e = this.element.position();
                    return {
                        top: e.top - (parseInt(this.helper.css("top"), 10) || 0) + this.scrollParent.scrollTop(),
                        left: e.left - (parseInt(this.helper.css("left"), 10) || 0) + this.scrollParent.scrollLeft()
                    }
                }
                return {
                    top: 0,
                    left: 0
                }
            },
            _cacheMargins: function() {
                this.margins = {
                    left: parseInt(this.element.css("marginLeft"), 10) || 0,
                    top: parseInt(this.element.css("marginTop"), 10) || 0,
                    right: parseInt(this.element.css("marginRight"), 10) || 0,
                    bottom: parseInt(this.element.css("marginBottom"), 10) || 0
                }
            },
            _cacheHelperProportions: function() {
                this.helperProportions = {
                    width: this.helper.outerWidth(),
                    height: this.helper.outerHeight()
                }
            },
            _setContainment: function() {
                var t, i, n, o = this.options;
                if ("parent" === o.containment && (o.containment = this.helper[0].parentNode), ("document" === o.containment || "window" === o.containment) && (this.containment = ["document" === o.containment ? 0: e(window).scrollLeft() - this.offset.relative.left - this.offset.parent.left, "document" === o.containment ? 0: e(window).scrollTop() - this.offset.relative.top - this.offset.parent.top, ("document" === o.containment ? 0 : e(window).scrollLeft()) + e("document" === o.containment ? document : window).width() - this.helperProportions.width - this.margins.left, ("document" === o.containment ? 0 : e(window).scrollTop()) + (e("document" === o.containment ? document : window).height() || document.body.parentNode.scrollHeight) - this.helperProportions.height - this.margins.top]), /^(document|window|parent)$/.test(o.containment) || o.containment.constructor === Array)
                    o.containment.constructor === Array && (this.containment = o.containment);
                else {
                    if (i = e(o.containment), n = i[0], !n)
                        return;
                    t = "hidden" !== e(n).css("overflow"), this.containment = [(parseInt(e(n).css("borderLeftWidth"), 10) || 0) + (parseInt(e(n).css("paddingLeft"), 10) || 0), (parseInt(e(n).css("borderTopWidth"), 10) || 0) + (parseInt(e(n).css("paddingTop"), 10) || 0), (t ? Math.max(n.scrollWidth, n.offsetWidth) : n.offsetWidth) - (parseInt(e(n).css("borderRightWidth"), 10) || 0) - (parseInt(e(n).css("paddingRight"), 10) || 0) - this.helperProportions.width - this.margins.left - this.margins.right, (t ? Math.max(n.scrollHeight, n.offsetHeight) : n.offsetHeight) - (parseInt(e(n).css("borderBottomWidth"), 10) || 0) - (parseInt(e(n).css("paddingBottom"), 10) || 0) - this.helperProportions.height - this.margins.top - this.margins.bottom], this.relative_container = i
                }
            },
            _convertPositionTo: function(t, i) {
                i || (i = this.position);
                var n = "absolute" === t ? 1: - 1, o = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && e.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent, s = /(html|body)/i.test(o[0].tagName);
                return {
                    top: i.top + this.offset.relative.top * n + this.offset.parent.top * n - ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : s ? 0 : o.scrollTop()) * n,
                    left: i.left + this.offset.relative.left * n + this.offset.parent.left * n - ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft(): s ? 0: o.scrollLeft()) * n
                }
            },
            _generatePosition: function(t) {
                var i, n, o, s, r = this.options, a = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && e.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent, l = /(html|body)/i.test(a[0].tagName), c = t.pageX, u = t.pageY;
                return this.originalPosition && (this.containment && (this.relative_container ? (n = this.relative_container.offset(), i = [this.containment[0] + n.left, this.containment[1] + n.top, this.containment[2] + n.left, this.containment[3] + n.top]) : i = this.containment, t.pageX - this.offset.click.left < i[0] && (c = i[0] + this.offset.click.left), t.pageY - this.offset.click.top < i[1] && (u = i[1] + this.offset.click.top), t.pageX - this.offset.click.left > i[2] && (c = i[2] + this.offset.click.left), t.pageY - this.offset.click.top > i[3] && (u = i[3] + this.offset.click.top)), r.grid && (o = r.grid[1] ? this.originalPageY + Math.round((u - this.originalPageY) / r.grid[1]) * r.grid[1] : this.originalPageY, u = i ? o - this.offset.click.top >= i[1] || o - this.offset.click.top > i[3] ? o : o - this.offset.click.top >= i[1] ? o - r.grid[1] : o + r.grid[1] : o, s = r.grid[0] ? this.originalPageX + Math.round((c - this.originalPageX) / r.grid[0]) * r.grid[0] : this.originalPageX, c = i ? s - this.offset.click.left >= i[0] || s - this.offset.click.left > i[2] ? s : s - this.offset.click.left >= i[0] ? s - r.grid[0] : s + r.grid[0] : s)), {
                    top: u - this.offset.click.top - this.offset.relative.top - this.offset.parent.top + ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : l ? 0 : a.scrollTop()),
                    left: c - this.offset.click.left - this.offset.relative.left - this.offset.parent.left + ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft(): l ? 0: a.scrollLeft())
                }
            },
            _clear: function() {
                this.helper.removeClass("ui-draggable-dragging"), this.helper[0] === this.element[0] || this.cancelHelperRemoval || this.helper.remove(), this.helper = null, this.cancelHelperRemoval=!1
            },
            _trigger: function(t, i, n) {
                return n = n || this._uiHash(), e.ui.plugin.call(this, t, [i, n]), "drag" === t && (this.positionAbs = this._convertPositionTo("absolute")), e.Widget.prototype._trigger.call(this, t, i, n)
            },
            plugins: {},
            _uiHash: function() {
                return {
                    helper: this.helper,
                    position: this.position,
                    originalPosition: this.originalPosition,
                    offset: this.positionAbs
                }
            }
        }), e.ui.plugin.add("draggable", "connectToSortable", {
            start: function(t, i) {
                var n = e(this).data("ui-draggable"), o = n.options, s = e.extend({}, i, {
                    item: n.element
                });
                n.sortables = [], e(o.connectToSortable).each(function() {
                    var i = e.data(this, "ui-sortable");
                    i&&!i.options.disabled && (n.sortables.push({
                        instance: i,
                        shouldRevert: i.options.revert
                    }), i.refreshPositions(), i._trigger("activate", t, s))
                })
            },
            stop: function(t, i) {
                var n = e(this).data("ui-draggable"), o = e.extend({}, i, {
                    item: n.element
                });
                e.each(n.sortables, function() {
                    this.instance.isOver ? (this.instance.isOver = 0, n.cancelHelperRemoval=!0, this.instance.cancelHelperRemoval=!1, this.shouldRevert && (this.instance.options.revert = this.shouldRevert), this.instance._mouseStop(t), this.instance.options.helper = this.instance.options._helper, "original" === n.options.helper && this.instance.currentItem.css({
                        top: "auto",
                        left: "auto"
                    })) : (this.instance.cancelHelperRemoval=!1, this.instance._trigger("deactivate", t, o))
                })
            },
            drag: function(t, i) {
                var n = e(this).data("ui-draggable"), o = this;
                e.each(n.sortables, function() {
                    var s=!1, r = this;
                    this.instance.positionAbs = n.positionAbs, this.instance.helperProportions = n.helperProportions, this.instance.offset.click = n.offset.click, this.instance._intersectsWith(this.instance.containerCache) && (s=!0, e.each(n.sortables, function() {
                        return this.instance.positionAbs = n.positionAbs, this.instance.helperProportions = n.helperProportions, this.instance.offset.click = n.offset.click, this !== r && this.instance._intersectsWith(this.instance.containerCache) && e.contains(r.instance.element[0], this.instance.element[0]) && (s=!1), s
                    })), s ? (this.instance.isOver || (this.instance.isOver = 1, this.instance.currentItem = e(o).clone().removeAttr("id").appendTo(this.instance.element).data("ui-sortable-item", !0), this.instance.options._helper = this.instance.options.helper, this.instance.options.helper = function() {
                        return i.helper[0]
                    }, t.target = this.instance.currentItem[0], this.instance._mouseCapture(t, !0), this.instance._mouseStart(t, !0, !0), this.instance.offset.click.top = n.offset.click.top, this.instance.offset.click.left = n.offset.click.left, this.instance.offset.parent.left -= n.offset.parent.left - this.instance.offset.parent.left, this.instance.offset.parent.top -= n.offset.parent.top - this.instance.offset.parent.top, n._trigger("toSortable", t), n.dropped = this.instance.element, n.currentItem = n.element, this.instance.fromOutside = n), this.instance.currentItem && this.instance._mouseDrag(t)) : this.instance.isOver && (this.instance.isOver = 0, this.instance.cancelHelperRemoval=!0, this.instance.options.revert=!1, this.instance._trigger("out", t, this.instance._uiHash(this.instance)), this.instance._mouseStop(t, !0), this.instance.options.helper = this.instance.options._helper, this.instance.currentItem.remove(), this.instance.placeholder && this.instance.placeholder.remove(), n._trigger("fromSortable", t), n.dropped=!1)
                })
            }
        }), e.ui.plugin.add("draggable", "cursor", {
            start: function() {
                var t = e("body"), i = e(this).data("ui-draggable").options;
                t.css("cursor") && (i._cursor = t.css("cursor")), t.css("cursor", i.cursor)
            },
            stop: function() {
                var t = e(this).data("ui-draggable").options;
                t._cursor && e("body").css("cursor", t._cursor)
            }
        }), e.ui.plugin.add("draggable", "opacity", {
            start: function(t, i) {
                var n = e(i.helper), o = e(this).data("ui-draggable").options;
                n.css("opacity") && (o._opacity = n.css("opacity")), n.css("opacity", o.opacity)
            },
            stop: function(t, i) {
                var n = e(this).data("ui-draggable").options;
                n._opacity && e(i.helper).css("opacity", n._opacity)
            }
        }), e.ui.plugin.add("draggable", "scroll", {
            start: function() {
                var t = e(this).data("ui-draggable");
                t.scrollParent[0] !== document && "HTML" !== t.scrollParent[0].tagName && (t.overflowOffset = t.scrollParent.offset())
            },
            drag: function(t) {
                var i = e(this).data("ui-draggable"), n = i.options, o=!1;
                i.scrollParent[0] !== document && "HTML" !== i.scrollParent[0].tagName ? (n.axis && "x" === n.axis || (i.overflowOffset.top + i.scrollParent[0].offsetHeight - t.pageY < n.scrollSensitivity ? i.scrollParent[0].scrollTop = o = i.scrollParent[0].scrollTop + n.scrollSpeed : t.pageY - i.overflowOffset.top < n.scrollSensitivity && (i.scrollParent[0].scrollTop = o = i.scrollParent[0].scrollTop - n.scrollSpeed)), n.axis && "y" === n.axis || (i.overflowOffset.left + i.scrollParent[0].offsetWidth - t.pageX < n.scrollSensitivity ? i.scrollParent[0].scrollLeft = o = i.scrollParent[0].scrollLeft + n.scrollSpeed : t.pageX - i.overflowOffset.left < n.scrollSensitivity && (i.scrollParent[0].scrollLeft = o = i.scrollParent[0].scrollLeft - n.scrollSpeed))) : (n.axis && "x" === n.axis || (t.pageY - e(document).scrollTop() < n.scrollSensitivity ? o = e(document).scrollTop(e(document).scrollTop() - n.scrollSpeed) : e(window).height() - (t.pageY - e(document).scrollTop()) < n.scrollSensitivity && (o = e(document).scrollTop(e(document).scrollTop() + n.scrollSpeed))), n.axis && "y" === n.axis || (t.pageX - e(document).scrollLeft() < n.scrollSensitivity ? o = e(document).scrollLeft(e(document).scrollLeft() - n.scrollSpeed) : e(window).width() - (t.pageX - e(document).scrollLeft()) < n.scrollSensitivity && (o = e(document).scrollLeft(e(document).scrollLeft() + n.scrollSpeed)))), o!==!1 && e.ui.ddmanager&&!n.dropBehaviour && e.ui.ddmanager.prepareOffsets(i, t)
            }
        }), e.ui.plugin.add("draggable", "snap", {
            start: function() {
                var t = e(this).data("ui-draggable"), i = t.options;
                t.snapElements = [], e(i.snap.constructor !== String ? i.snap.items || ":data(ui-draggable)" : i.snap).each(function() {
                    var i = e(this), n = i.offset();
                    this !== t.element[0] && t.snapElements.push({
                        item: this,
                        width: i.outerWidth(),
                        height: i.outerHeight(),
                        top: n.top,
                        left: n.left
                    })
                })
            },
            drag: function(t, i) {
                var n, o, s, r, a, l, c, u, d, h, p = e(this).data("ui-draggable"), f = p.options, m = f.snapTolerance, g = i.offset.left, v = g + p.helperProportions.width, y = i.offset.top, b = y + p.helperProportions.height;
                for (d = p.snapElements.length - 1; d >= 0; d--)
                    a = p.snapElements[d].left, l = a + p.snapElements[d].width, c = p.snapElements[d].top, u = c + p.snapElements[d].height, g > a - m && l + m > g && y > c - m && u + m > y || g > a - m && l + m > g && b > c - m && u + m > b || v > a - m && l + m > v && y > c - m && u + m > y || v > a - m && l + m > v && b > c - m && u + m > b ? ("inner" !== f.snapMode && (n = Math.abs(c - b) <= m, o = Math.abs(u - y) <= m, s = Math.abs(a - v) <= m, r = Math.abs(l - g) <= m, n && (i.position.top = p._convertPositionTo("relative", {
                        top: c - p.helperProportions.height,
                        left: 0
                    }).top - p.margins.top), o && (i.position.top = p._convertPositionTo("relative", {
                        top: u,
                        left: 0
                    }).top - p.margins.top), s && (i.position.left = p._convertPositionTo("relative", {
                        top: 0,
                        left: a - p.helperProportions.width
                    }).left - p.margins.left), r && (i.position.left = p._convertPositionTo("relative", {
                        top: 0,
                        left: l
                    }).left - p.margins.left)), h = n || o || s || r, "outer" !== f.snapMode && (n = Math.abs(c - y) <= m, o = Math.abs(u - b) <= m, s = Math.abs(a - g) <= m, r = Math.abs(l - v) <= m, n && (i.position.top = p._convertPositionTo("relative", {
                        top: c,
                        left: 0
                    }).top - p.margins.top), o && (i.position.top = p._convertPositionTo("relative", {
                        top: u - p.helperProportions.height,
                        left: 0
                    }).top - p.margins.top), s && (i.position.left = p._convertPositionTo("relative", {
                        top: 0,
                        left: a
                    }).left - p.margins.left), r && (i.position.left = p._convertPositionTo("relative", {
                        top: 0,
                        left: l - p.helperProportions.width
                    }).left - p.margins.left)), !p.snapElements[d].snapping && (n || o || s || r || h) && p.options.snap.snap && p.options.snap.snap.call(p.element, t, e.extend(p._uiHash(), {
                        snapItem: p.snapElements[d].item
                    })), p.snapElements[d].snapping = n || o || s || r || h) : (p.snapElements[d].snapping && p.options.snap.release && p.options.snap.release.call(p.element, t, e.extend(p._uiHash(), {
                        snapItem: p.snapElements[d].item
                    })), p.snapElements[d].snapping=!1)
            }
        }), e.ui.plugin.add("draggable", "stack", {
            start: function() {
                var t, i = this.data("ui-draggable").options, n = e.makeArray(e(i.stack)).sort(function(t, i) {
                    return (parseInt(e(t).css("zIndex"), 10) || 0) - (parseInt(e(i).css("zIndex"), 10) || 0)
                });
                n.length && (t = parseInt(e(n[0]).css("zIndex"), 10) || 0, e(n).each(function(i) {
                    e(this).css("zIndex", t + i)
                }), this.css("zIndex", t + n.length))
            }
        }), e.ui.plugin.add("draggable", "zIndex", {
            start: function(t, i) {
                var n = e(i.helper), o = e(this).data("ui-draggable").options;
                n.css("zIndex") && (o._zIndex = n.css("zIndex")), n.css("zIndex", o.zIndex)
            },
            stop: function(t, i) {
                var n = e(this).data("ui-draggable").options;
                n._zIndex && e(i.helper).css("zIndex", n._zIndex)
            }
        })
    }(e)
}), _wAMD.define("jquery-ui/position", ["jquery"], function(e) {
    !function(e, t) {
        function i(e, t, i) {
            return [parseFloat(e[0]) * (p.test(e[0]) ? t / 100 : 1), parseFloat(e[1]) * (p.test(e[1]) ? i / 100 : 1)]
        }
        function n(t, i) {
            return parseInt(e.css(t, i), 10) || 0
        }
        function o(t) {
            var i = t[0];
            return 9 === i.nodeType ? {
                width: t.width(),
                height: t.height(),
                offset: {
                    top: 0,
                    left: 0
                }
            } : e.isWindow(i) ? {
                width: t.width(),
                height: t.height(),
                offset: {
                    top: t.scrollTop(),
                    left: t.scrollLeft()
                }
            } : i.preventDefault ? {
                width: 0,
                height: 0,
                offset: {
                    top: i.pageY,
                    left: i.pageX
                }
            } : {
                width: t.outerWidth(),
                height: t.outerHeight(),
                offset: t.offset()
            }
        }
        e.ui = e.ui || {};
        var s, r = Math.max, a = Math.abs, l = Math.round, c = /left|center|right/, u = /top|center|bottom/, d = /[\+\-]\d+(\.[\d]+)?%?/, h = /^\w+/, p = /%$/, f = e.fn.position;
        e.position = {
            scrollbarWidth: function() {
                if (s !== t)
                    return s;
                var i, n, o = e("<div style='display:block;width:50px;height:50px;overflow:hidden;'><div style='height:100px;width:auto;'></div></div>"), r = o.children()[0];
                return e("body").append(o), i = r.offsetWidth, o.css("overflow", "scroll"), n = r.offsetWidth, i === n && (n = o[0].clientWidth), o.remove(), s = i - n
            },
            getScrollInfo: function(t) {
                var i = t.isWindow ? "": t.element.css("overflow-x"), n = t.isWindow ? "": t.element.css("overflow-y"), o = "scroll" === i || "auto" === i && t.width < t.element[0].scrollWidth, s = "scroll" === n || "auto" === n && t.height < t.element[0].scrollHeight;
                return {
                    width: s ? e.position.scrollbarWidth(): 0,
                    height: o ? e.position.scrollbarWidth(): 0
                }
            },
            getWithinInfo: function(t) {
                var i = e(t || window), n = e.isWindow(i[0]);
                return {
                    element: i,
                    isWindow: n,
                    offset: i.offset() || {
                        left: 0,
                        top: 0
                    },
                    scrollLeft: i.scrollLeft(),
                    scrollTop: i.scrollTop(),
                    width: n ? i.width(): i.outerWidth(),
                    height: n ? i.height(): i.outerHeight()
                }
            }
        }, e.fn.position = function(t) {
            if (!t ||!t.of)
                return f.apply(this, arguments);
            t = e.extend({}, t);
            var s, p, m, g, v, y, b = e(t.of), w = e.position.getWithinInfo(t.within), _ = e.position.getScrollInfo(w), C = (t.collision || "flip").split(" "), k = {};
            return y = o(b), b[0].preventDefault && (t.at = "left top"), p = y.width, m = y.height, g = y.offset, v = e.extend({}, g), e.each(["my", "at"], function() {
                var e, i, n = (t[this] || "").split(" ");
                1 === n.length && (n = c.test(n[0]) ? n.concat(["center"]) : u.test(n[0]) ? ["center"].concat(n) : ["center", "center"]), n[0] = c.test(n[0]) ? n[0] : "center", n[1] = u.test(n[1]) ? n[1] : "center", e = d.exec(n[0]), i = d.exec(n[1]), k[this] = [e ? e[0]: 0, i ? i[0]: 0], t[this] = [h.exec(n[0])[0], h.exec(n[1])[0]]
            }), 1 === C.length && (C[1] = C[0]), "right" === t.at[0] ? v.left += p : "center" === t.at[0] && (v.left += p / 2), "bottom" === t.at[1] ? v.top += m : "center" === t.at[1] && (v.top += m / 2), s = i(k.at, p, m), v.left += s[0], v.top += s[1], this.each(function() {
                var o, c, u = e(this), d = u.outerWidth(), h = u.outerHeight(), f = n(this, "marginLeft"), y = n(this, "marginTop"), x = d + f + n(this, "marginRight") + _.width, S = h + y + n(this, "marginBottom") + _.height, M = e.extend({}, v), T = i(k.my, u.outerWidth(), u.outerHeight());
                "right" === t.my[0] ? M.left -= d : "center" === t.my[0] && (M.left -= d / 2), "bottom" === t.my[1] ? M.top -= h : "center" === t.my[1] && (M.top -= h / 2), M.left += T[0], M.top += T[1], e.support.offsetFractions || (M.left = l(M.left), M.top = l(M.top)), o = {
                    marginLeft: f,
                    marginTop: y
                }, e.each(["left", "top"], function(i, n) {
                    e.ui.position[C[i]] && e.ui.position[C[i]][n](M, {
                        targetWidth: p,
                        targetHeight: m,
                        elemWidth: d,
                        elemHeight: h,
                        collisionPosition: o,
                        collisionWidth: x,
                        collisionHeight: S,
                        offset: [s[0] + T[0], s[1] + T[1]],
                        my: t.my,
                        at: t.at,
                        within: w,
                        elem: u
                    })
                }), t.using && (c = function(e) {
                    var i = g.left - M.left, n = i + p - d, o = g.top - M.top, s = o + m - h, l = {
                        target: {
                            element: b,
                            left: g.left,
                            top: g.top,
                            width: p,
                            height: m
                        },
                        element: {
                            element: u,
                            left: M.left,
                            top: M.top,
                            width: d,
                            height: h
                        },
                        horizontal: 0 > n ? "left": i > 0 ? "right": "center",
                        vertical: 0 > s ? "top": o > 0 ? "bottom": "middle"
                    };
                    d > p && a(i + n) < p && (l.horizontal = "center"), h > m && a(o + s) < m && (l.vertical = "middle"), l.important = r(a(i), a(n)) > r(a(o), a(s)) ? "horizontal" : "vertical", t.using.call(this, e, l)
                }), u.offset(e.extend(M, {
                    using: c
                }))
            })
        }, e.ui.position = {
            fit: {
                left: function(e, t) {
                    var i, n = t.within, o = n.isWindow ? n.scrollLeft: n.offset.left, s = n.width, a = e.left - t.collisionPosition.marginLeft, l = o - a, c = a + t.collisionWidth - s - o;
                    t.collisionWidth > s ? l > 0 && 0 >= c ? (i = e.left + l + t.collisionWidth - s - o, e.left += l - i) : e.left = c > 0 && 0 >= l ? o : l > c ? o + s - t.collisionWidth : o : l > 0 ? e.left += l : c > 0 ? e.left -= c : e.left = r(e.left - a, e.left)
                },
                top: function(e, t) {
                    var i, n = t.within, o = n.isWindow ? n.scrollTop: n.offset.top, s = t.within.height, a = e.top - t.collisionPosition.marginTop, l = o - a, c = a + t.collisionHeight - s - o;
                    t.collisionHeight > s ? l > 0 && 0 >= c ? (i = e.top + l + t.collisionHeight - s - o, e.top += l - i) : e.top = c > 0 && 0 >= l ? o : l > c ? o + s - t.collisionHeight : o : l > 0 ? e.top += l : c > 0 ? e.top -= c : e.top = r(e.top - a, e.top)
                }
            },
            flip: {
                left: function(e, t) {
                    var i, n, o = t.within, s = o.offset.left + o.scrollLeft, r = o.width, l = o.isWindow ? o.scrollLeft: o.offset.left, c = e.left - t.collisionPosition.marginLeft, u = c - l, d = c + t.collisionWidth - r - l, h = "left" === t.my[0]?-t.elemWidth : "right" === t.my[0] ? t.elemWidth : 0, p = "left" === t.at[0] ? t.targetWidth : "right" === t.at[0]?-t.targetWidth : 0, f =- 2 * t.offset[0];
                    0 > u ? (i = e.left + h + p + f + t.collisionWidth - r - s, (0 > i || i < a(u)) && (e.left += h + p + f)) : d > 0 && (n = e.left - t.collisionPosition.marginLeft + h + p + f - l, (n > 0 || a(n) < d) && (e.left += h + p + f))
                },
                top: function(e, t) {
                    var i, n, o = t.within, s = o.offset.top + o.scrollTop, r = o.height, l = o.isWindow ? o.scrollTop: o.offset.top, c = e.top - t.collisionPosition.marginTop, u = c - l, d = c + t.collisionHeight - r - l, h = "top" === t.my[1], p = h?-t.elemHeight : "bottom" === t.my[1] ? t.elemHeight : 0, f = "top" === t.at[1] ? t.targetHeight : "bottom" === t.at[1]?-t.targetHeight : 0, m =- 2 * t.offset[1];
                    0 > u ? (n = e.top + p + f + m + t.collisionHeight - r - s, e.top + p + f + m > u && (0 > n || n < a(u)) && (e.top += p + f + m)) : d > 0 && (i = e.top - t.collisionPosition.marginTop + p + f + m - l, e.top + p + f + m > d && (i > 0 || a(i) < d) && (e.top += p + f + m))
                }
            },
            flipfit: {
                left: function() {
                    e.ui.position.flip.left.apply(this, arguments), e.ui.position.fit.left.apply(this, arguments)
                },
                top: function() {
                    e.ui.position.flip.top.apply(this, arguments), e.ui.position.fit.top.apply(this, arguments)
                }
            }
        }, function() {
            var t, i, n, o, s, r = document.getElementsByTagName("body")[0], a = document.createElement("div");
            t = document.createElement(r ? "div" : "body"), n = {
                visibility: "hidden",
                width: 0,
                height: 0,
                border: 0,
                margin: 0,
                background: "none"
            }, r && e.extend(n, {
                position: "absolute",
                left: "-1000px",
                top: "-1000px"
            });
            for (s in n)
                t.style[s] = n[s];
            t.appendChild(a), i = r || document.documentElement, i.insertBefore(t, i.firstChild), a.style.cssText = "position: absolute; left: 10.7432222px;", o = e(a).offset().left, e.support.offsetFractions = o > 10 && 11 > o, t.innerHTML = "", i.removeChild(t)
        }()
    }(e)
}), _wAMD.define("jquery-ui/resizable", ["jquery", "./core", "./mouse", "./widget"], function(e) {
    !function(e) {
        function t(e) {
            return parseInt(e, 10) || 0
        }
        function i(e) {
            return !isNaN(parseInt(e, 10))
        }
        e.widget("ui.resizable", e.ui.mouse, {
            version: "1.10.2",
            widgetEventPrefix: "resize",
            options: {
                alsoResize: !1,
                animate: !1,
                animateDuration: "slow",
                animateEasing: "swing",
                aspectRatio: !1,
                autoHide: !1,
                containment: !1,
                ghost: !1,
                grid: !1,
                handles: "e,s,se",
                helper: !1,
                maxHeight: null,
                maxWidth: null,
                minHeight: 10,
                minWidth: 10,
                zIndex: 90,
                resize: null,
                start: null,
                stop: null
            },
            _create: function() {
                var t, i, n, o, s, r = this, a = this.options;
                if (this.element.addClass("ui-resizable"), e.extend(this, {
                    _aspectRatio: !!a.aspectRatio,
                    aspectRatio: a.aspectRatio,
                    originalElement: this.element,
                    _proportionallyResizeElements: [],
                    _helper: a.helper || a.ghost || a.animate ? a.helper || "ui-resizable-helper": null
                }), this.element[0].nodeName.match(/canvas|textarea|input|select|button|img/i) && (this.element.wrap(e("<div class='ui-wrapper' style='overflow: hidden;'></div>").css({
                    position: this.element.css("position"),
                    width: this.element.outerWidth(),
                    height: this.element.outerHeight(),
                    top: this.element.css("top"),
                    left: this.element.css("left")
                })), this.element = this.element.parent().data("ui-resizable", this.element.data("ui-resizable")), this.elementIsWrapper=!0, this.element.css({
                    marginLeft: this.originalElement.css("marginLeft"),
                    marginTop: this.originalElement.css("marginTop"),
                    marginRight: this.originalElement.css("marginRight"),
                    marginBottom: this.originalElement.css("marginBottom")
                }), this.originalElement.css({
                    marginLeft: 0,
                    marginTop: 0,
                    marginRight: 0,
                    marginBottom: 0
                }), this.originalResizeStyle = this.originalElement.css("resize"), this.originalElement.css("resize", "none"), this._proportionallyResizeElements.push(this.originalElement.css({
                    position: "static",
                    zoom: 1,
                    display: "block"
                })), this.originalElement.css({
                    margin: this.originalElement.css("margin")
                }), this._proportionallyResize()), this.handles = a.handles || (e(".ui-resizable-handle", this.element).length ? {
                    n: ".ui-resizable-n",
                    e: ".ui-resizable-e",
                    s: ".ui-resizable-s",
                    w: ".ui-resizable-w",
                    se: ".ui-resizable-se",
                    sw: ".ui-resizable-sw",
                    ne: ".ui-resizable-ne",
                    nw: ".ui-resizable-nw"
                } : "e,s,se"), this.handles.constructor === String)
                    for ("all" === this.handles && (this.handles = "n,e,s,w,se,sw,ne,nw"), t = this.handles.split(","), this.handles = {}, i = 0; i < t.length; i++)
                        n = e.trim(t[i]), s = "ui-resizable-" + n, o = e("<div class='ui-resizable-handle " + s + "'></div>"), o.css({
                            zIndex: a.zIndex
                        }), "se" === n && o.addClass("ui-icon ui-icon-gripsmall-diagonal-se"), this.handles[n] = ".ui-resizable-" + n, this.element.append(o);
                this._renderAxis = function(t) {
                    var i, n, o, s;
                    t = t || this.element;
                    for (i in this.handles)
                        this.handles[i].constructor === String && (this.handles[i] = e(this.handles[i], this.element).show()), this.elementIsWrapper && this.originalElement[0].nodeName.match(/textarea|input|select|button/i) && (n = e(this.handles[i], this.element), s = /sw|ne|nw|se|n|s/.test(i) ? n.outerHeight() : n.outerWidth(), o = ["padding", /ne|nw|n/.test(i) ? "Top": /se|sw|s/.test(i) ? "Bottom": /^e$/.test(i) ? "Right": "Left"].join(""), t.css(o, s), this._proportionallyResize()), e(this.handles[i]).length
                }, this._renderAxis(this.element), this._handles = e(".ui-resizable-handle", this.element).disableSelection(), this._handles.mouseover(function() {
                    r.resizing || (this.className && (o = this.className.match(/ui-resizable-(se|sw|ne|nw|n|e|s|w)/i)), r.axis = o && o[1] ? o[1] : "se")
                }), a.autoHide && (this._handles.hide(), e(this.element).addClass("ui-resizable-autohide").mouseenter(function() {
                    a.disabled || (e(this).removeClass("ui-resizable-autohide"), r._handles.show())
                }).mouseleave(function() {
                    a.disabled || r.resizing || (e(this).addClass("ui-resizable-autohide"), r._handles.hide())
                })), this._mouseInit()
            },
            _destroy: function() {
                this._mouseDestroy();
                var t, i = function(t) {
                    e(t).removeClass("ui-resizable ui-resizable-disabled ui-resizable-resizing").removeData("resizable").removeData("ui-resizable").unbind(".resizable").find(".ui-resizable-handle").remove()
                };
                return this.elementIsWrapper && (i(this.element), t = this.element, this.originalElement.css({
                    position: t.css("position"),
                    width: t.outerWidth(),
                    height: t.outerHeight(),
                    top: t.css("top"),
                    left: t.css("left")
                }).insertAfter(t), t.remove()), this.originalElement.css("resize", this.originalResizeStyle), i(this.originalElement), this
            },
            _mouseCapture: function(t) {
                var i, n, o=!1;
                for (i in this.handles)
                    n = e(this.handles[i])[0], (n === t.target || e.contains(n, t.target)) && (o=!0);
                return !this.options.disabled && o
            },
            _mouseStart: function(i) {
                var n, o, s, r = this.options, a = this.element.position(), l = this.element;
                return this.resizing=!0, /absolute/.test(l.css("position")) ? l.css({
                    position: "absolute",
                    top: l.css("top"),
                    left: l.css("left")
                }) : l.is(".ui-draggable") && l.css({
                    position: "absolute",
                    top: a.top,
                    left: a.left
                }), this._renderProxy(), n = t(this.helper.css("left")), o = t(this.helper.css("top")), r.containment && (n += e(r.containment).scrollLeft() || 0, o += e(r.containment).scrollTop() || 0), this.offset = this.helper.offset(), this.position = {
                    left: n,
                    top: o
                }, this.size = this._helper ? {
                    width: l.outerWidth(),
                    height: l.outerHeight()
                } : {
                    width: l.width(),
                    height: l.height()
                }, this.originalSize = this._helper ? {
                    width: l.outerWidth(),
                    height: l.outerHeight()
                } : {
                    width: l.width(),
                    height: l.height()
                }, this.originalPosition = {
                    left: n,
                    top: o
                }, this.sizeDiff = {
                    width: l.outerWidth() - l.width(),
                    height: l.outerHeight() - l.height()
                }, this.originalMousePosition = {
                    left: i.pageX,
                    top: i.pageY
                }, this.aspectRatio = "number" == typeof r.aspectRatio ? r.aspectRatio : this.originalSize.width / this.originalSize.height || 1, s = e(".ui-resizable-" + this.axis).css("cursor"), e("body").css("cursor", "auto" === s ? this.axis + "-resize" : s), l.addClass("ui-resizable-resizing"), this._propagate("start", i), !0
            },
            _mouseDrag: function(t) {
                var i, n = this.helper, o = {}, s = this.originalMousePosition, r = this.axis, a = this.position.top, l = this.position.left, c = this.size.width, u = this.size.height, d = t.pageX - s.left || 0, h = t.pageY - s.top || 0, p = this._change[r];
                return p ? (i = p.apply(this, [t, d, h]), this._updateVirtualBoundaries(t.shiftKey), (this._aspectRatio || t.shiftKey) && (i = this._updateRatio(i, t)), i = this._respectSize(i, t), this._updateCache(i), this._propagate("resize", t), this.position.top !== a && (o.top = this.position.top + "px"), this.position.left !== l && (o.left = this.position.left + "px"), this.size.width !== c && (o.width = this.size.width + "px"), this.size.height !== u && (o.height = this.size.height + "px"), n.css(o), !this._helper && this._proportionallyResizeElements.length && this._proportionallyResize(), e.isEmptyObject(o) || this._trigger("resize", t, this.ui()), !1) : !1
            },
            _mouseStop: function(t) {
                this.resizing=!1;
                var i, n, o, s, r, a, l, c = this.options, u = this;
                return this._helper && (i = this._proportionallyResizeElements, n = i.length && /textarea/i.test(i[0].nodeName), o = n && e.ui.hasScroll(i[0], "left") ? 0 : u.sizeDiff.height, s = n ? 0 : u.sizeDiff.width, r = {
                    width: u.helper.width() - s,
                    height: u.helper.height() - o
                }, a = parseInt(u.element.css("left"), 10) + (u.position.left - u.originalPosition.left) || null, l = parseInt(u.element.css("top"), 10) + (u.position.top - u.originalPosition.top) || null, c.animate || this.element.css(e.extend(r, {
                    top: l,
                    left: a
                })), u.helper.height(u.size.height), u.helper.width(u.size.width), this._helper&&!c.animate && this._proportionallyResize()), e("body").css("cursor", "auto"), this.element.removeClass("ui-resizable-resizing"), this._propagate("stop", t), this._helper && this.helper.remove(), !1
            },
            _updateVirtualBoundaries: function(e) {
                var t, n, o, s, r, a = this.options;
                r = {
                    minWidth: i(a.minWidth) ? a.minWidth: 0,
                    maxWidth: i(a.maxWidth) ? a.maxWidth: 1 / 0,
                    minHeight: i(a.minHeight) ? a.minHeight: 0,
                    maxHeight: i(a.maxHeight) ? a.maxHeight: 1 / 0
                }, (this._aspectRatio || e) && (t = r.minHeight * this.aspectRatio, o = r.minWidth / this.aspectRatio, n = r.maxHeight * this.aspectRatio, s = r.maxWidth / this.aspectRatio, t > r.minWidth && (r.minWidth = t), o > r.minHeight && (r.minHeight = o), n < r.maxWidth && (r.maxWidth = n), s < r.maxHeight && (r.maxHeight = s)), this._vBoundaries = r
            },
            _updateCache: function(e) {
                this.offset = this.helper.offset(), i(e.left) && (this.position.left = e.left), i(e.top) && (this.position.top = e.top), i(e.height) && (this.size.height = e.height), i(e.width) && (this.size.width = e.width)
            },
            _updateRatio: function(e) {
                var t = this.position, n = this.size, o = this.axis;
                return i(e.height) ? e.width = e.height * this.aspectRatio : i(e.width) && (e.height = e.width / this.aspectRatio), "sw" === o && (e.left = t.left + (n.width - e.width), e.top = null), "nw" === o && (e.top = t.top + (n.height - e.height), e.left = t.left + (n.width - e.width)), e
            },
            _respectSize: function(e) {
                var t = this._vBoundaries, n = this.axis, o = i(e.width) && t.maxWidth && t.maxWidth < e.width, s = i(e.height) && t.maxHeight && t.maxHeight < e.height, r = i(e.width) && t.minWidth && t.minWidth > e.width, a = i(e.height) && t.minHeight && t.minHeight > e.height, l = this.originalPosition.left + this.originalSize.width, c = this.position.top + this.size.height, u = /sw|nw|w/.test(n), d = /nw|ne|n/.test(n);
                return r && (e.width = t.minWidth), a && (e.height = t.minHeight), o && (e.width = t.maxWidth), s && (e.height = t.maxHeight), r && u && (e.left = l - t.minWidth), o && u && (e.left = l - t.maxWidth), a && d && (e.top = c - t.minHeight), s && d && (e.top = c - t.maxHeight), e.width || e.height || e.left ||!e.top ? e.width || e.height || e.top ||!e.left || (e.left = null) : e.top = null, e
            },
            _proportionallyResize: function() {
                if (this._proportionallyResizeElements.length) {
                    var e, t, i, n, o, s = this.helper || this.element;
                    for (e = 0; e < this._proportionallyResizeElements.length; e++) {
                        if (o = this._proportionallyResizeElements[e], !this.borderDif)
                            for (this.borderDif = [], i = [o.css("borderTopWidth"), o.css("borderRightWidth"), o.css("borderBottomWidth"), o.css("borderLeftWidth")], n = [o.css("paddingTop"), o.css("paddingRight"), o.css("paddingBottom"), o.css("paddingLeft")], t = 0; t < i.length; t++)
                                this.borderDif[t] = (parseInt(i[t], 10) || 0) + (parseInt(n[t], 10) || 0);
                        o.css({
                            height: s.height() - this.borderDif[0] - this.borderDif[2] || 0,
                            width: s.width() - this.borderDif[1] - this.borderDif[3] || 0
                        })
                    }
                }
            },
            _renderProxy: function() {
                var t = this.element, i = this.options;
                this.elementOffset = t.offset(), this._helper ? (this.helper = this.helper || e("<div style='overflow:hidden;'></div>"), this.helper.addClass(this._helper).css({
                    width: this.element.outerWidth() - 1,
                    height: this.element.outerHeight() - 1,
                    position: "absolute",
                    left: this.elementOffset.left + "px",
                    top: this.elementOffset.top + "px",
                    zIndex: ++i.zIndex
                }), this.helper.appendTo("body").disableSelection()) : this.helper = this.element
            },
            _change: {
                e: function(e, t) {
                    return {
                        width: this.originalSize.width + t
                    }
                },
                w: function(e, t) {
                    var i = this.originalSize, n = this.originalPosition;
                    return {
                        left: n.left + t,
                        width: i.width - t
                    }
                },
                n: function(e, t, i) {
                    var n = this.originalSize, o = this.originalPosition;
                    return {
                        top: o.top + i,
                        height: n.height - i
                    }
                },
                s: function(e, t, i) {
                    return {
                        height: this.originalSize.height + i
                    }
                },
                se: function(t, i, n) {
                    return e.extend(this._change.s.apply(this, arguments), this._change.e.apply(this, [t, i, n]))
                },
                sw: function(t, i, n) {
                    return e.extend(this._change.s.apply(this, arguments), this._change.w.apply(this, [t, i, n]))
                },
                ne: function(t, i, n) {
                    return e.extend(this._change.n.apply(this, arguments), this._change.e.apply(this, [t, i, n]))
                },
                nw: function(t, i, n) {
                    return e.extend(this._change.n.apply(this, arguments), this._change.w.apply(this, [t, i, n]))
                }
            },
            _propagate: function(t, i) {
                e.ui.plugin.call(this, t, [i, this.ui()]), "resize" !== t && this._trigger(t, i, this.ui())
            },
            plugins: {},
            ui: function() {
                return {
                    originalElement: this.originalElement,
                    element: this.element,
                    helper: this.helper,
                    position: this.position,
                    size: this.size,
                    originalSize: this.originalSize,
                    originalPosition: this.originalPosition
                }
            }
        }), e.ui.plugin.add("resizable", "animate", {
            stop: function(t) {
                var i = e(this).data("ui-resizable"), n = i.options, o = i._proportionallyResizeElements, s = o.length && /textarea/i.test(o[0].nodeName), r = s && e.ui.hasScroll(o[0], "left") ? 0: i.sizeDiff.height, a = s ? 0: i.sizeDiff.width, l = {
                    width: i.size.width - a,
                    height: i.size.height - r
                }, c = parseInt(i.element.css("left"), 10) + (i.position.left - i.originalPosition.left) || null, u = parseInt(i.element.css("top"), 10) + (i.position.top - i.originalPosition.top) || null;
                i.element.animate(e.extend(l, u && c ? {
                    top: u,
                    left: c
                } : {}), {
                    duration: n.animateDuration,
                    easing: n.animateEasing,
                    step: function() {
                        var n = {
                            width: parseInt(i.element.css("width"), 10),
                            height: parseInt(i.element.css("height"), 10),
                            top: parseInt(i.element.css("top"), 10),
                            left: parseInt(i.element.css("left"), 10)
                        };
                        o && o.length && e(o[0]).css({
                            width: n.width,
                            height: n.height
                        }), i._updateCache(n), i._propagate("resize", t)
                    }
                })
            }
        }), e.ui.plugin.add("resizable", "containment", {
            start: function() {
                var i, n, o, s, r, a, l, c = e(this).data("ui-resizable"), u = c.options, d = c.element, h = u.containment, p = h instanceof e ? h.get(0): /parent/.test(h) ? d.parent().get(0): h;
                p && (c.containerElement = e(p), /document/.test(h) || h === document ? (c.containerOffset = {
                    left: 0,
                    top: 0
                }, c.containerPosition = {
                    left: 0,
                    top: 0
                }, c.parentData = {
                    element: e(document),
                    left: 0,
                    top: 0,
                    width: e(document).width(),
                    height: e(document).height() || document.body.parentNode.scrollHeight
                }) : (i = e(p), n = [], e(["Top", "Right", "Left", "Bottom"]).each(function(e, o) {
                    n[e] = t(i.css("padding" + o))
                }), c.containerOffset = i.offset(), c.containerPosition = i.position(), c.containerSize = {
                    height: i.innerHeight() - n[3],
                    width: i.innerWidth() - n[1]
                }, o = c.containerOffset, s = c.containerSize.height, r = c.containerSize.width, a = e.ui.hasScroll(p, "left") ? p.scrollWidth : r, l = e.ui.hasScroll(p) ? p.scrollHeight : s, c.parentData = {
                    element: p,
                    left: o.left,
                    top: o.top,
                    width: a,
                    height: l
                }))
            },
            resize: function(t) {
                var i, n, o, s, r = e(this).data("ui-resizable"), a = r.options, l = r.containerOffset, c = r.position, u = r._aspectRatio || t.shiftKey, d = {
                    top: 0,
                    left: 0
                }, h = r.containerElement;
                h[0] !== document && /static/.test(h.css("position")) && (d = l), c.left < (r._helper ? l.left : 0) && (r.size.width = r.size.width + (r._helper ? r.position.left - l.left : r.position.left - d.left), u && (r.size.height = r.size.width / r.aspectRatio), r.position.left = a.helper ? l.left : 0), c.top < (r._helper ? l.top : 0) && (r.size.height = r.size.height + (r._helper ? r.position.top - l.top : r.position.top), u && (r.size.width = r.size.height * r.aspectRatio), r.position.top = r._helper ? l.top : 0), r.offset.left = r.parentData.left + r.position.left, r.offset.top = r.parentData.top + r.position.top, i = Math.abs((r._helper ? r.offset.left - d.left : r.offset.left - d.left) + r.sizeDiff.width), n = Math.abs((r._helper ? r.offset.top - d.top : r.offset.top - l.top) + r.sizeDiff.height), o = r.containerElement.get(0) === r.element.parent().get(0), s = /relative|absolute/.test(r.containerElement.css("position")), o && s && (i -= r.parentData.left), i + r.size.width >= r.parentData.width && (r.size.width = r.parentData.width - i, u && (r.size.height = r.size.width / r.aspectRatio)), n + r.size.height >= r.parentData.height && (r.size.height = r.parentData.height - n, u && (r.size.width = r.size.height * r.aspectRatio))
            },
            stop: function() {
                var t = e(this).data("ui-resizable"), i = t.options, n = t.containerOffset, o = t.containerPosition, s = t.containerElement, r = e(t.helper), a = r.offset(), l = r.outerWidth() - t.sizeDiff.width, c = r.outerHeight() - t.sizeDiff.height;
                t._helper&&!i.animate && /relative/.test(s.css("position")) && e(this).css({
                    left: a.left - o.left - n.left,
                    width: l,
                    height: c
                }), t._helper&&!i.animate && /static/.test(s.css("position")) && e(this).css({
                    left: a.left - o.left - n.left,
                    width: l,
                    height: c
                })
            }
        }), e.ui.plugin.add("resizable", "alsoResize", {
            start: function() {
                var t = e(this).data("ui-resizable"), i = t.options, n = function(t) {
                    e(t).each(function() {
                        var t = e(this);
                        t.data("ui-resizable-alsoresize", {
                            width: parseInt(t.width(), 10),
                            height: parseInt(t.height(), 10),
                            left: parseInt(t.css("left"), 10),
                            top: parseInt(t.css("top"), 10)
                        })
                    })
                };
                "object" != typeof i.alsoResize || i.alsoResize.parentNode ? n(i.alsoResize) : i.alsoResize.length ? (i.alsoResize = i.alsoResize[0], n(i.alsoResize)) : e.each(i.alsoResize, function(e) {
                    n(e)
                })
            },
            resize: function(t, i) {
                var n = e(this).data("ui-resizable"), o = n.options, s = n.originalSize, r = n.originalPosition, a = {
                    height: n.size.height - s.height || 0,
                    width: n.size.width - s.width || 0,
                    top: n.position.top - r.top || 0,
                    left: n.position.left - r.left || 0
                }, l = function(t, n) {
                    e(t).each(function() {
                        var t = e(this), o = e(this).data("ui-resizable-alsoresize"), s = {}, r = n && n.length ? n: t.parents(i.originalElement[0]).length ? ["width", "height"]: ["width", "height", "top", "left"];
                        e.each(r, function(e, t) {
                            var i = (o[t] || 0) + (a[t] || 0);
                            i && i >= 0 && (s[t] = i || null)
                        }), t.css(s)
                    })
                };
                "object" != typeof o.alsoResize || o.alsoResize.nodeType ? l(o.alsoResize) : e.each(o.alsoResize, function(e, t) {
                    l(e, t)
                })
            },
            stop: function() {
                e(this).removeData("resizable-alsoresize")
            }
        }), e.ui.plugin.add("resizable", "ghost", {
            start: function() {
                var t = e(this).data("ui-resizable"), i = t.options, n = t.size;
                t.ghost = t.originalElement.clone(), t.ghost.css({
                    opacity: .25,
                    display: "block",
                    position: "relative",
                    height: n.height,
                    width: n.width,
                    margin: 0,
                    left: 0,
                    top: 0
                }).addClass("ui-resizable-ghost").addClass("string" == typeof i.ghost ? i.ghost : ""), t.ghost.appendTo(t.helper)
            },
            resize: function() {
                var t = e(this).data("ui-resizable");
                t.ghost && t.ghost.css({
                    position: "relative",
                    height: t.size.height,
                    width: t.size.width
                })
            },
            stop: function() {
                var t = e(this).data("ui-resizable");
                t.ghost && t.helper && t.helper.get(0).removeChild(t.ghost.get(0))
            }
        }), e.ui.plugin.add("resizable", "grid", {
            resize: function() {
                var t = e(this).data("ui-resizable"), i = t.options, n = t.size, o = t.originalSize, s = t.originalPosition, r = t.axis, a = "number" == typeof i.grid ? [i.grid, i.grid]: i.grid, l = a[0] || 1, c = a[1] || 1, u = Math.round((n.width - o.width) / l) * l, d = Math.round((n.height - o.height) / c) * c, h = o.width + u, p = o.height + d, f = i.maxWidth && i.maxWidth < h, m = i.maxHeight && i.maxHeight < p, g = i.minWidth && i.minWidth > h, v = i.minHeight && i.minHeight > p;
                i.grid = a, g && (h += l), v && (p += c), f && (h -= l), m && (p -= c), /^(se|s|e)$/.test(r) ? (t.size.width = h, t.size.height = p) : /^(ne)$/.test(r) ? (t.size.width = h, t.size.height = p, t.position.top = s.top - d) : /^(sw)$/.test(r) ? (t.size.width = h, t.size.height = p, t.position.left = s.left - u) : (t.size.width = h, t.size.height = p, t.position.top = s.top - d, t.position.left = s.left - u)
            }
        })
    }(e)
}), _wAMD.define("jquery-ui/dialog", ["jquery", "./core", "./widget", "./button", "./draggable", "./mouse", "./position", "./resizable"], function(e) {
    !function(e) {
        var t = {
            buttons: !0,
            height: !0,
            maxHeight: !0,
            maxWidth: !0,
            minHeight: !0,
            minWidth: !0,
            width: !0
        }, i = {
            maxHeight: !0,
            maxWidth: !0,
            minHeight: !0,
            minWidth: !0
        };
        e.widget("ui.dialog", {
            version: "1.10.2",
            options: {
                appendTo: "body",
                autoOpen: !0,
                buttons: [],
                closeOnEscape: !0,
                closeText: "close",
                dialogClass: "",
                draggable: !0,
                hide: null,
                height: "auto",
                maxHeight: null,
                maxWidth: null,
                minHeight: 150,
                minWidth: 150,
                modal: !1,
                position: {
                    my: "center",
                    at: "center",
                    of: window,
                    collision: "fit",
                    using: function(t) {
                        var i = e(this).css(t).offset().top;
                        0 > i && e(this).css("top", t.top - i)
                    }
                },
                resizable: !0,
                show: null,
                title: null,
                width: 300,
                beforeClose: null,
                close: null,
                drag: null,
                dragStart: null,
                dragStop: null,
                focus: null,
                open: null,
                resize: null,
                resizeStart: null,
                resizeStop: null
            },
            _create: function() {
                this.originalCss = {
                    display: this.element[0].style.display,
                    width: this.element[0].style.width,
                    minHeight: this.element[0].style.minHeight,
                    maxHeight: this.element[0].style.maxHeight,
                    height: this.element[0].style.height
                }, this.originalPosition = {
                    parent: this.element.parent(),
                    index: this.element.parent().children().index(this.element)
                }, this.originalTitle = this.element.attr("title"), this.options.title = this.options.title || this.originalTitle, this._createWrapper(), this.element.show().removeAttr("title").addClass("ui-dialog-content ui-widget-content").appendTo(this.uiDialog), this._createTitlebar(), this._createButtonPane(), this.options.draggable && e.fn.draggable && this._makeDraggable(), this.options.resizable && e.fn.resizable && this._makeResizable(), this._isOpen=!1
            },
            _init: function() {
                this.options.autoOpen && this.open()
            },
            _appendTo: function() {
                var t = this.options.appendTo;
                return t && (t.jquery || t.nodeType) ? e(t) : this.document.find(t || "body").eq(0)
            },
            _destroy: function() {
                var e, t = this.originalPosition;
                this._destroyOverlay(), this.element.removeUniqueId().removeClass("ui-dialog-content ui-widget-content").css(this.originalCss).detach(), this.uiDialog.stop(!0, !0).remove(), this.originalTitle && this.element.attr("title", this.originalTitle), e = t.parent.children().eq(t.index), e.length && e[0] !== this.element[0] ? e.before(this.element) : t.parent.append(this.element)
            },
            widget: function() {
                return this.uiDialog
            },
            disable: e.noop,
            enable: e.noop,
            close: function(t) {
                var i = this;
                this._isOpen && this._trigger("beforeClose", t)!==!1 && (this._isOpen=!1, this._destroyOverlay(), this.opener.filter(":focusable").focus().length || e(this.document[0].activeElement).blur(), this._hide(this.uiDialog, this.options.hide, function() {
                    i._trigger("close", t)
                }))
            },
            isOpen: function() {
                return this._isOpen
            },
            moveToTop: function() {
                this._moveToTop()
            },
            _moveToTop: function(e, t) {
                var i=!!this.uiDialog.nextAll(":visible").insertBefore(this.uiDialog).length;
                return i&&!t && this._trigger("focus", e), i
            },
            open: function() {
                var t = this;
                return this._isOpen ? (this._moveToTop() && this._focusTabbable(), void 0) : (this._isOpen=!0, this.opener = e(this.document[0].activeElement), this._size(), this._position(), this._createOverlay(), this._moveToTop(null, !0), this._show(this.uiDialog, this.options.show, function() {
                    t._focusTabbable(), t._trigger("focus")
                }), this._trigger("open"), void 0)
            },
            _focusTabbable: function() {
                var e = this.element.find("[autofocus]");
                e.length || (e = this.element.find(":tabbable")), e.length || (e = this.uiDialogButtonPane.find(":tabbable")), e.length || (e = this.uiDialogTitlebarClose.filter(":tabbable")), e.length || (e = this.uiDialog), e.eq(0).focus()
            },
            _keepFocus: function(t) {
                function i() {
                    var t = this.document[0].activeElement, i = this.uiDialog[0] === t || e.contains(this.uiDialog[0], t);
                    i || this._focusTabbable()
                }
                t.preventDefault(), i.call(this), this._delay(i)
            },
            _createWrapper: function() {
                this.uiDialog = e("<div>").addClass("ui-dialog ui-widget ui-widget-content ui-corner-all ui-front " + this.options.dialogClass).hide().attr({
                    tabIndex: - 1,
                    role: "dialog"
                }).appendTo(this._appendTo()), this._on(this.uiDialog, {
                    keydown: function(t) {
                        if (this.options.closeOnEscape&&!t.isDefaultPrevented() && t.keyCode && t.keyCode === e.ui.keyCode.ESCAPE)
                            return t.preventDefault(), this.close(t), void 0;
                        if (t.keyCode === e.ui.keyCode.TAB) {
                            var i = this.uiDialog.find(":tabbable"), n = i.filter(":first"), o = i.filter(":last");
                            t.target !== o[0] && t.target !== this.uiDialog[0] || t.shiftKey ? t.target !== n[0] && t.target !== this.uiDialog[0] ||!t.shiftKey || (o.focus(1), t.preventDefault()) : (n.focus(1), t.preventDefault())
                        }
                    },
                    mousedown: function(e) {
                        this._moveToTop(e) && this._focusTabbable()
                    }
                }), this.element.find("[aria-describedby]").length || this.uiDialog.attr({
                    "aria-describedby": this.element.uniqueId().attr("id")
                })
            },
            _createTitlebar: function() {
                var t;
                this.uiDialogTitlebar = e("<div>").addClass("ui-dialog-titlebar ui-widget-header ui-corner-all ui-helper-clearfix").prependTo(this.uiDialog), this._on(this.uiDialogTitlebar, {
                    mousedown: function(t) {
                        e(t.target).closest(".ui-dialog-titlebar-close") || this.uiDialog.focus()
                    }
                }), this.uiDialogTitlebarClose = e("<button></button>").button({
                    label: this.options.closeText,
                    icons: {
                        primary: "ui-icon-closethick"
                    },
                    text: !1
                }).addClass("ui-dialog-titlebar-close").appendTo(this.uiDialogTitlebar), this._on(this.uiDialogTitlebarClose, {
                    click: function(e) {
                        e.preventDefault(), this.close(e)
                    }
                }), t = e("<span>").uniqueId().addClass("ui-dialog-title").prependTo(this.uiDialogTitlebar), this._title(t), this.uiDialog.attr({
                    "aria-labelledby": t.attr("id")
                })
            },
            _title: function(e) {
                this.options.title || e.html("&#160;"), e.text(this.options.title)
            },
            _createButtonPane: function() {
                this.uiDialogButtonPane = e("<div>").addClass("ui-dialog-buttonpane ui-widget-content ui-helper-clearfix"), this.uiButtonSet = e("<div>").addClass("ui-dialog-buttonset").appendTo(this.uiDialogButtonPane), this._createButtons()
            },
            _createButtons: function() {
                var t = this, i = this.options.buttons;
                return this.uiDialogButtonPane.remove(), this.uiButtonSet.empty(), e.isEmptyObject(i) || e.isArray(i)&&!i.length ? (this.uiDialog.removeClass("ui-dialog-buttons"), void 0) : (e.each(i, function(i, n) {
                    var o, s;
                    n = e.isFunction(n) ? {
                        click: n,
                        text: i
                    } : n, n = e.extend({
                        type: "button"
                    }, n), o = n.click, n.click = function() {
                        o.apply(t.element[0], arguments)
                    }, s = {
                        icons: n.icons,
                        text: n.showText
                    }, delete n.icons, delete n.showText, e("<button></button>", n).button(s).appendTo(t.uiButtonSet)
                }), this.uiDialog.addClass("ui-dialog-buttons"), this.uiDialogButtonPane.appendTo(this.uiDialog), void 0)
            },
            _makeDraggable: function() {
                function t(e) {
                    return {
                        position: e.position,
                        offset: e.offset
                    }
                }
                var i = this, n = this.options;
                this.uiDialog.draggable({
                    cancel: ".ui-dialog-content, .ui-dialog-titlebar-close",
                    handle: ".ui-dialog-titlebar",
                    containment: "document",
                    start: function(n, o) {
                        e(this).addClass("ui-dialog-dragging"), i._blockFrames(), i._trigger("dragStart", n, t(o))
                    },
                    drag: function(e, n) {
                        i._trigger("drag", e, t(n))
                    },
                    stop: function(o, s) {
                        n.position = [s.position.left - i.document.scrollLeft(), s.position.top - i.document.scrollTop()], e(this).removeClass("ui-dialog-dragging"), i._unblockFrames(), i._trigger("dragStop", o, t(s))
                    }
                })
            },
            _makeResizable: function() {
                function t(e) {
                    return {
                        originalPosition: e.originalPosition,
                        originalSize: e.originalSize,
                        position: e.position,
                        size: e.size
                    }
                }
                var i = this, n = this.options, o = n.resizable, s = this.uiDialog.css("position"), r = "string" == typeof o ? o: "n,e,s,w,se,sw,ne,nw";
                this.uiDialog.resizable({
                    cancel: ".ui-dialog-content",
                    containment: "document",
                    alsoResize: this.element,
                    maxWidth: n.maxWidth,
                    maxHeight: n.maxHeight,
                    minWidth: n.minWidth,
                    minHeight: this._minHeight(),
                    handles: r,
                    start: function(n, o) {
                        e(this).addClass("ui-dialog-resizing"), i._blockFrames(), i._trigger("resizeStart", n, t(o))
                    },
                    resize: function(e, n) {
                        i._trigger("resize", e, t(n))
                    },
                    stop: function(o, s) {
                        n.height = e(this).height(), n.width = e(this).width(), e(this).removeClass("ui-dialog-resizing"), i._unblockFrames(), i._trigger("resizeStop", o, t(s))
                    }
                }).css("position", s)
            },
            _minHeight: function() {
                var e = this.options;
                return "auto" === e.height ? e.minHeight : Math.min(e.minHeight, e.height)
            },
            _position: function() {
                var e = this.uiDialog.is(":visible");
                e || this.uiDialog.show(), this.uiDialog.position(this.options.position), e || this.uiDialog.hide()
            },
            _setOptions: function(n) {
                var o = this, s=!1, r = {};
                e.each(n, function(e, n) {
                    o._setOption(e, n), e in t && (s=!0), e in i && (r[e] = n)
                }), s && (this._size(), this._position()), this.uiDialog.is(":data(ui-resizable)") && this.uiDialog.resizable("option", r)
            },
            _setOption: function(e, t) {
                var i, n, o = this.uiDialog;
                "dialogClass" === e && o.removeClass(this.options.dialogClass).addClass(t), "disabled" !== e && (this._super(e, t), "appendTo" === e && this.uiDialog.appendTo(this._appendTo()), "buttons" === e && this._createButtons(), "closeText" === e && this.uiDialogTitlebarClose.button({
                    label: "" + t
                }), "draggable" === e && (i = o.is(":data(ui-draggable)"), i&&!t && o.draggable("destroy"), !i && t && this._makeDraggable()), "position" === e && this._position(), "resizable" === e && (n = o.is(":data(ui-resizable)"), n&&!t && o.resizable("destroy"), n && "string" == typeof t && o.resizable("option", "handles", t), n || t===!1 || this._makeResizable()), "title" === e && this._title(this.uiDialogTitlebar.find(".ui-dialog-title")))
            },
            _size: function() {
                var e, t, i, n = this.options;
                this.element.show().css({
                    width: "auto",
                    minHeight: 0,
                    maxHeight: "none",
                    height: 0
                }), n.minWidth > n.width && (n.width = n.minWidth), e = this.uiDialog.css({
                    height: "auto",
                    width: n.width
                }).outerHeight(), t = Math.max(0, n.minHeight - e), i = "number" == typeof n.maxHeight ? Math.max(0, n.maxHeight - e) : "none", "auto" === n.height ? this.element.css({
                    minHeight: t,
                    maxHeight: i,
                    height: "auto"
                }) : this.element.height(Math.max(0, n.height - e)), this.uiDialog.is(":data(ui-resizable)") && this.uiDialog.resizable("option", "minHeight", this._minHeight())
            },
            _blockFrames: function() {
                this.iframeBlocks = this.document.find("iframe").map(function() {
                    var t = e(this);
                    return e("<div>").css({
                        position: "absolute",
                        width: t.outerWidth(),
                        height: t.outerHeight()
                    }).appendTo(t.parent()).offset(t.offset())[0]
                })
            },
            _unblockFrames: function() {
                this.iframeBlocks && (this.iframeBlocks.remove(), delete this.iframeBlocks)
            },
            _allowInteraction: function(t) {
                return e(t.target).closest(".ui-dialog").length?!0 : !!e(t.target).closest(".ui-datepicker").length
            },
            _createOverlay: function() {
                if (this.options.modal) {
                    var t = this, i = this.widgetFullName;
                    e.ui.dialog.overlayInstances || this._delay(function() {
                        e.ui.dialog.overlayInstances && this.document.bind("focusin.dialog", function(n) {
                            t._allowInteraction(n) || (n.preventDefault(), e(".ui-dialog:visible:last .ui-dialog-content").data(i)._focusTabbable())
                        })
                    }), this.overlay = e("<div>").addClass("ui-widget-overlay ui-front").appendTo(this._appendTo()), this._on(this.overlay, {
                        mousedown: "_keepFocus"
                    }), e.ui.dialog.overlayInstances++
                }
            },
            _destroyOverlay: function() {
                this.options.modal && this.overlay && (e.ui.dialog.overlayInstances--, e.ui.dialog.overlayInstances || this.document.unbind("focusin.dialog"), this.overlay.remove(), this.overlay = null)
            }
        }), e.ui.dialog.overlayInstances = 0, e.uiBackCompat!==!1 && e.widget("ui.dialog", e.ui.dialog, {
            _position: function() {
                var t, i = this.options.position, n = [], o = [0, 0];
                i ? (("string" == typeof i || "object" == typeof i && "0"in i) && (n = i.split ? i.split(" ") : [i[0], i[1]], 1 === n.length && (n[1] = n[0]), e.each(["left", "top"], function(e, t) {
                    + n[e] === n[e] && (o[e] = n[e], n[e] = t)
                }), i = {
                    my: n[0] + (o[0] < 0 ? o[0] : "+" + o[0]) + " " + n[1] + (o[1] < 0 ? o[1] : "+" + o[1]),
                    at: n.join(" ")
                }), i = e.extend({}, e.ui.dialog.prototype.options.position, i)) : i = e.ui.dialog.prototype.options.position, t = this.uiDialog.is(":visible"), t || this.uiDialog.show(), this.uiDialog.position(i), t || this.uiDialog.hide()
            }
        })
    }(e)
}), _wAMD.define("components/regions/Dialog", ["jquery", "backbone-all", "underscore", "jquery-ui/dialog"], function(e, t, i) {
    return t.Marionette.Region.extend({
        defaults: {
            title: "Default Title",
            closeOnEscape: !0,
            draggable: !1,
            modal: !0,
            resizable: !1,
            position: {
                my: "center",
                at: "center",
                of: window
            }
        },
        constructor: function() {
            i.extend(this, t.Events)
        },
        el: "#dialog-region",
        dialogClass: "popup-modal",
        onShow: function(e) {
            i.extend({}, e.dialog), this.view = e, this.dialogOptions = i.result(e, "dialog"), this.$el.dialog(this.getDefaultOptions(this.dialogOptions, e)), this.updateDialogCss(e), this.namespace = ".dialog-" + Math.floor(1e6 * Math.random() + 1), this.setupBindings(e), i.result(this.view.options, "onShow")
        },
        onClose: function() {
            i.result(this.view.options, "onClose"), this.stopListening(), e("body").off("keydown" + this.namespace), e("body").off("click" + this.namespace), e(window).off("resize" + this.namespace)
        },
        updateDialogCss: function(t) {
            var i = this.$el.find(".dialog-body");
            if (i.length) {
                var n = i.height(), o = i.parent().height(), s = e(window).height() - 170;
                n > o && s > o && (s = o - 30), this.$el.addClass("hasBody"), i.css({
                    maxHeight: s + "px"
                })
            } else 
                this.$el.removeClass("hasBody");
            if (t && this.dialogOptions && this.dialogOptions.size) {
                var r = this.dialogOptions.size.split("x");
                if (2 == r.length) {
                    var a = this.$el.parent(), l = r[0] + "px", c = r[1] + "px";
                    a.css({
                        width: l,
                        minWidth: l,
                        maxWidth: l,
                        height: c,
                        minHeight: c,
                        maxHeight: c,
                        left: (e(window).width() - r[0]) / 2,
                        top: (e(window).height() - r[1]) / 2
                    })
                }
            }
        },
        setupBindings: function(t) {
            this.listenTo(t, "render", function() {
                this.updateView(t), this.updateDialogCss(t)
            }), this.previousView ? this.listenTo(t, "dialog:close", this.restoreMainView) : this.listenTo(t, "dialog:close", this.closeDialog), this.$el.dialog("option", {
                close: i.bind(this.closeDialog, this)
            }), this.listenTo(t, "dialog:update:buttons", this.createFooter), this.listenTo(t, "dialog:update:title", this.updateDialogTitle), this.listenTo(t, "dialog:reposition", this.reposition), e("body").on("keydown" + this.namespace, i.bind(this.handleKeydown, this)), e("body").on("click" + this.namespace, ".ui-widget-overlay", i.bind(this.handleBackgroundClick, this));
            var n = i.bind(this.reposition, this);
            e(window).on("resize" + this.namespace, i.throttle(n, 100))
        },
        reposition: function() {
            var e = this;
            e.$el.dialog({
                position: e.defaults.position
            })
        },
        closeDialog: function() {
            this.close(), this.$el.dialog("destroy")
        },
        cancelDialog: function(e) {
            e && e.preventDefault(), "function" == typeof this.view.dialogCancel ? this.view.dialogCancel() : (this.closeDialog(), this.previousView && (this.previousView.close(), this.previousView = null))
        },
        handleKeydown: function(e) {
            27 == e.which && this.$el.dialog("option", "closeOnEscape") && (e && e.preventDefault(), this.cancelDialog(e))
        },
        handleBackgroundClick: function() {
            this.dialogOptions && (this.dialogOptions.cancelAll || this.dialogOptions.cancelBg) && this.cancelDialog()
        },
        getDefaultOptions: function(e, t) {
            return this.updateView(t), e = e || {}, e.buttons && this.updateView(t), e.dialogClass = i.compact(["w-ui", e.dialogClass, this.dialogClass]).join(" "), e = i.defaults(e, this.defaults), e = i.omit(e, "buttons")
        },
        updateView: function(e) {
            if (e.dialog) {
                var t = i.isFunction(e.dialog) ? e.dialog(): e.dialog;
                (t.cancelX || t.cancelAll) && this.createCancelX(e), t.cancelXThin && this.createCancelXThin(e), t.buttons && this.createFooter(t.buttons, e), e.bindUIElements()
            }
        },
        createCancelX: function(e) {
            e.$el.find("a.close-x").length || (e.$el.prepend('<a autofocus href="#"></a>'), e.$el.prepend('<a href="#" class="close-x w-icon-delete"></a>'), e.events = e.events || {}, e.events["click a.close-x"] = i.bind(this.cancelDialog, this), e.delegateEvents())
        },
        createCancelXThin: function(e) {
            e.$el.find("a.close-x").length || (e.$el.prepend('<a autofocus href="#"></a>'), e.$el.prepend('<a href="#" class="close-x close-x-thin">&times;</a>'), e.events = e.events || {}, e.events["click a.close-x"] = i.bind(this.cancelDialog, this), e.delegateEvents())
        },
        createFooter: function(t, i) {
            var n, i = i || this.currentView;
            n = this.$el.find(".dialog-footer").length ? this.$el.find(".dialog-footer").empty() : e('<div class="dialog-footer" />'), n = this.createButtonFooter(n, t), n.append(i.$el.find(".footer")), i.$el.append(n)
        },
        createButtonFooter: function(e, t) {
            return i.each(t, function(t) {
                t.group && t.buttons ? e.append(this.createButtonGroup(t)) : (t.html || t.text) && e.append(this.createButton(t))
            }, this), e
        },
        createButtonGroup: function(t) {
            var n = e('<div class="' + t.group + '"/>');
            return i.each(t.buttons, function(e) {
                n.append(this.createButton(e))
            }, this), n
        },
        createButton: function(t) {
            var i = t.type || "button", n = e(document.createElement(i));
            return t.text ? n.text(t.text) : t.html && n.html(t.html), n.addClass(t.className || ""), n
        },
        updateDialogTitle: function(e) {
            this.$el.dialog("option", "title", e)
        },
        showSecondaryDialog: function(e) {
            this.ensureEl();
            var n = e.isClosed || i.isUndefined(e.$el), o = e !== this.currentView;
            o && (1 == this.$el.find(">div").length ? (this.previousView = this.currentView, this.$el.find(">div").hide()) : this.close()), e.render(), (o || n) && this.$el.append(e.el), this.currentView = e, t.Marionette.triggerMethod.call(this, "show", e), t.Marionette.triggerMethod.call(e, "show")
        },
        restoreMainView: function() {
            this.previousView && (this.currentView.close(), this.$el.find(">div").show(), t.Marionette.triggerMethod.call(this, "show", this.previousView), t.Marionette.triggerMethod.call(this.previousView, "reloadDialog"), t.Marionette.triggerMethod.call(this.previousView, "show"), this.currentView = this.previousView, this.previousView = null)
        }
    })
}), _wAMD.define("components/regions/FullscreenModal", ["jquery", "backbone-all", "components/regions/Dialog"], function(e, t, i) {
    var n = i.prototype;
    return i.extend({
        dialogClass: "fullscreen-modal",
        updateDialogCss: function() {
            e("body").css("height", "100%").css("overflow", "hidden")
        },
        closeDialog: function() {
            e("body").css("height", "").css("overflow", ""), this.close(), this.$el.dialog("close")
        },
        getDefaultOptions: function(e) {
            return e = n.getDefaultOptions.apply(this, arguments), e = _.defaults(e, {
                hide: 400,
                show: 400
            })
        },
        createCancelX: function(e) {
            e.$el.find("a.btn-x").length || (e.$el.prepend('<a autofocus href="#"></a>'), e.$el.prepend('<a href="#" class="btn-x">&times;</a>'), e.events["click a.btn-x"] = _.bind(this.cancelDialog, this), e.delegateEvents())
        },
        createFooter: function() {},
        updateDialogTitle: function() {}
    })
}), _wAMD.define("components/layouts/FullscreenModalView", ["backbone-all", "mustache", "components/regions/FullscreenModal"], function(e, t, i) {
    var n = e.Marionette.Layout.extend({
        template: t.compile('<div class="fullScreenModal"></div>'),
        onRender: function() {
            this.addRegions({
                fullscreenModal: i.extend({
                    el: this.$(".fullScreenModal")
                })
            })
        }
    });
    return n
}), _wAMD.define("components/layouts/DialogView", ["backbone-all", "mustache", "components/regions/Dialog"], function(e, t, i) {
    var n = e.Marionette.Layout.extend({
        template: t.compile('<div class="dialog"></div>'),
        onRender: function() {
            this.addRegions({
                dialog: i.extend({
                    el: this.$(".dialog")
                })
            })
        }
    });
    return n
}), _wAMD.define("userhome/app", ["jquery", "backbone-all", "userhome/promo-overlay", "components/layouts/FullscreenModalView", "components/layouts/DialogView"], function(e, t, i, n, o) {
    var s = new t.Marionette.Application;
    s.on("start", function() {
        window.location.href.match(/#video/) && showVideoDialog()
    }), s.addInitializer(function() {
        new i
    }), window.homeApp = s;
    var r;
    s.showFullscreenModal = function(t) {
        r || (r = new n({
            el: e("<div>").appendTo("body")
        }), r.render()), r.fullscreenModal.show(t)
    };
    var a;
    return s.showDialog = function(t) {
        return a || (a = new o({
            el: e("<div>").appendTo("body")
        }), a.render()), a.dialog.show(t), t
    }, s
}), _wAMD.define("text", {}), _wAMD.define("tpl", ["text"], function(e) {
    var t = {};
    return {
        load: function(i, n, o, s) {
            if (s.isBuild && s.inlineTpl===!1)
                return o();
            var r = function(e) {
                s.isBuild && (t[i] = e), o(e)
            };
            e.get(n.toUrl("templates/" + i + ".tpl"), r, o.error)
        },
        write: function(i, n, o) {
            if (t.hasOwnProperty(n)) {
                var s = e.jsEscape(t[n]);
                o.asModule(i + "!" + n, "define(function () { return '" + s + "'; });\n")
            }
        }
    }
}), _wAMD.define("tpl!userhome/billing/trial-expired", [], function() {
    return '<span class="w-icon-arrow-left" style="display:none;"></span>\n<div id="showExpiredTrialPlans" style="width: 100%; margin: 0 auto; position: relative;">\n	<div id="modalContent" style="padding: 44px 110px 0px;">\n		<div class="title">{{#tl}}Your Free Trial has Expired{{/tl}}</div>\n		<div class="subtitle">{{#tl}}Please upgrade to continue.{{/tl}}</div>\n		<hr/>\n		<div class="header">{{#tl}}Select a plan:{{/tl}}</div>\n		<div class="planChoiceContainer">\n			<div plan="starter" class="planChoice">\n				<div class="planChoiceHeader">{{#tl}}Starter{{/tl}}</div>\n				<p class="planChoiceContent">{{#tl}}Connect your own domain, get expanded site stats, and premium support.{{/tl}}</p>\n				<div class="planChoiceHeader">$4/mo</div>\n			</div>\n			<div plan="pro" class="planChoice">\n				<div class="planChoiceHeader">{{#tl}}Pro{{/tl}}</div>\n				<p class="planChoiceContent">{{#tl}}Professional multimedia features, powerful site search, and password protection.{{/tl}}</p>\n				<div class="planChoiceHeader">$7/mo</div>\n			</div>\n			<div plan="business" class="planChoice">\n				<div class="planChoiceHeader">{{#tl}}Business{{/tl}}</div>\n				<p class="planChoiceContent">{{#tl}}Fully integrated eCommerce and all available features. The complete package.{{/tl}}</p>\n				<div class="planChoiceHeader">$20/mo</div>\n			</div>\n		</div>\n		<div class="btn-flat btn-flat-outline btn-flat-light" id="compareLink">Compare Plans</div>\n	</div>\n</div>\n<div id="showExpiredTrialComparison" style="display:none; width: 100%; margin: 0 auto; position: relative;">\n	<div id="modalContent" style="padding: 44px 110px 0px;">\n\n		<div class="title">{{#tl}}Compare Plans{{/tl}}</div>\n		<div class="subtitle">{{#tl}}30 day money back guarantee{{/tl}}</div>\n		<hr/>\n\n		<table cellspacing="0" cellpadding="0" id="planComparison">\n			<tr>\n				<th></th>\n				<th class="planHeader">\n					<div class="planTitle">{{#tl}}Starter{{/tl}}</div>\n					<div class="subtitle">$4/mo</div>\n					<div plan="starter" class="btn-flat btn-flat-outline btn-flat-light">{{#tl}}Choose{{/tl}}</div>\n				</th>\n				<th class="planHeader">\n					<div class="planTitle">{{#tl}}Pro{{/tl}}</div>\n					<div class="subtitle">$7/mo</div>\n					<div plan="pro" class="btn-flat btn-flat-outline btn-flat-light">{{#tl}}Choose{{/tl}}</div>\n				</th>\n				<th class="planHeader">\n					<div class="planTitle">{{#tl}}Business{{/tl}}</div>\n					<div class="subtitle">$20/mo</div>\n					<div plan="business" class="btn-flat btn-flat-outline btn-flat-light">{{#tl}}Choose{{/tl}}</div>\n				</th>\n			</tr>\n			<tr>\n				<td>{{#tl}}Connect your domain{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Connect a domain to publish your site to a domain you already own (example.com) instead of a free subdomain of weebly.com.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Expanded stats{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Learn more about your site visitors with expanded stats. Understand which pages are most popular, how people find your site, and which sites refer you traffic.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Premium support{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}We aim to provide everyone with amazing support, but Premium users receive the fastest responses from the most qualified members of our team.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Customizable footer{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}With the customizable footer, you have the flexibility to include your site menu, social icons, a contact form, logo, and more at the bottom of your site.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Remove Weebly link{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Remove the \'Create a free website with Weebly\' footer link from the bottom of your site with one click.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Your own favicon{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Give your site a more professional look by uploading your own favicon. Visitors will see this icon in their address bar and browser tab.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Header slideshow{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Add a header slideshow to showcase what’s important in an interactive and professional way. Supports various transition effects, captions, and linking.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}HD video &amp; audio players{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Display your videos with stunning clarity in our unbranded HD video player. Our HTML5 players ensure smooth viewing on web, mobile, and iPad.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Password protect pages{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Enable password protection to keep specified areas of your website hidden from view. Only those with the password are able to gain access.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Site search{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Site search helps your visitors quickly find the information they need. Search results display beautifully right within the theme of your site.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}SSL security{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}The Business plan includes a free SSL certificate for your domain (normally $69/year). SSL ensures that visitors navigate your site over a secured 128-bit encrypted connection, displaying a lock in the address bar. If using eCommerce, it enables customers to checkout directly on your domain.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Membership{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Build a membership website that lets people log in to access special content. You can add members, manage them in groups and control which pages they have access to.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td>{{#up_to}}20{{/up_to}}</td>\n				<td>{{#tl}}Unlimited{{/tl}}</td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Member registration{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Add a Register button to your website and allow site visitors to sign up to become members. You can automatically approve new members or manually review each one, and get alerted when a new member joins.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Google Advertising credit{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}We\'ve partnered with Google to provide $100 of free advertising (after spending $25) to new Google Adwords customers. Coupon will be emailed after purchase and is only available to customers in the U.S. and Canada.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td>$100</td>\n				<td>$100</td>\n				<td>$100</td>\n			</tr>\n			<tr class="breaker">\n				<td>eCommerce</td>\n				<td></td>\n				<td></td>\n				<td></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Number of products{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Sell an unlimited number of products with the Business plan.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td>{{starter_product_limit}}</td>\n				<td>{{pro_product_limit}}</td>\n				<td>{{business_product_limit}}</td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Weebly transaction fee{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}The Weebly transaction fee is charged on top of the standard processing fees charged by PayPal, Stripe, etc. Choose the Business plan to remove the Weebly transaction fee.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td>{{starter_transaction_fee}}%</td>\n				<td>{{pro_transaction_fee}}%</td>\n				<td>{{business_transaction_fee}}%</td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Integrated shopping cart{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}The shopping cart integrates directly within your site and theme, providing a seamless checkout experience that works everywhere -- including mobile and tablets. With the Business plan, customers checkout on your domain instead of checkout.weebly.com.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td>checkout.weebly.com</td>\n				<td>checkout.weebly.com</td>\n				<td>On your domain</td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Digital goods{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Sell digital goods like ZIP files, songs, and eBooks. Customers receive a secure link to download the files immediately after purchase and by email. You control how many times files can be downloaded or how many days they are available for download.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Inventory management{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Track your inventory so that you always know how many items are left in stock. Customers are automatically shown when a product is \'out of stock\'.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Shipping &amp; tax calculator{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Flexible shipping options allow you to offer \'free shipping\', set rates based on price or weight, and define carriers and delivery speed. Fine-grained tax controls calculate and add taxes to orders with full international capabilities.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n			<tr>\n				<td>{{#tl}}Coupon codes{{/tl}} <span tooltip="{{#esc_attr}}{{#tl}}Offer coupon codes to drive sales, encourage repeat buying behavior, and attract new customers to your store. Flexible options including $ discount, % discount, and the ability to define which products are eligible by offer.{{/tl}}{{/esc_attr}}" class="w-icon-question"></span></td>\n				<td></td>\n				<td></td>\n				<td><span class="w-icon-check-mark"></span></td>\n			</tr>\n		</table>\n\n		<div id="planTooltip"></div>\n	</div>\n</div>\n\n'
}), _wAMD.define("jquery-ui/effect", ["jquery"], function(e) {
    !function(t, i) {
        var n = "ui-effects-";
        t.effects = {
            effect: {}
        }, function(e, t) {
            function i(e, t, i) {
                var n = d[t.type] || {};
                return null == e ? i ||!t.def ? null : t.def : (e = n.floor?~~e : parseFloat(e), isNaN(e) ? t.def : n.mod ? (e + n.mod)%n.mod : 0 > e ? 0 : n.max < e ? n.max : e)
            }
            function n(t) {
                var i = c(), n = i._rgba = [];
                return t = t.toLowerCase(), f(l, function(e, o) {
                    var s, r = o.re.exec(t), a = r && o.parse(r), l = o.space || "rgba";
                    return a ? (s = i[l](a), i[u[l].cache] = s[u[l].cache], n = i._rgba = s._rgba, !1) : void 0
                }), n.length ? ("0,0,0,0" === n.join() && e.extend(n, s.transparent), i) : s[t]
            }
            function o(e, t, i) {
                return i = (i + 1)%1, 1 > 6 * i ? e + 6 * (t - e) * i : 1 > 2 * i ? t : 2 > 3 * i ? e + 6 * (t - e) * (2 / 3 - i) : e
            }
            var s, r = "backgroundColor borderBottomColor borderLeftColor borderRightColor borderTopColor color columnRuleColor outlineColor textDecorationColor textEmphasisColor", a = /^([\-+])=\s*(\d+\.?\d*)/, l = [{
                re: /rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,
                parse: function(e) {
                    return [e[1], e[2], e[3], e[4]]
                }
            }, {
                re: /rgba?\(\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,
                parse: function(e) {
                    return [2.55 * e[1], 2.55 * e[2], 2.55 * e[3], e[4]]
                }
            }, {
                re: /#([a-f0-9]{2})([a-f0-9]{2})([a-f0-9]{2})/,
                parse: function(e) {
                    return [parseInt(e[1], 16), parseInt(e[2], 16), parseInt(e[3], 16)]
                }
            }, {
                re: /#([a-f0-9])([a-f0-9])([a-f0-9])/,
                parse: function(e) {
                    return [parseInt(e[1] + e[1], 16), parseInt(e[2] + e[2], 16), parseInt(e[3] + e[3], 16)]
                }
            }, {
                re: /hsla?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,
                space: "hsla",
                parse: function(e) {
                    return [e[1], e[2] / 100, e[3] / 100, e[4]]
                }
            }
            ], c = e.Color = function(t, i, n, o) {
                return new e.Color.fn.parse(t, i, n, o)
            }, u = {
                rgba: {
                    props: {
                        red: {
                            idx: 0,
                            type: "byte"
                        },
                        green: {
                            idx: 1,
                            type: "byte"
                        },
                        blue: {
                            idx: 2,
                            type: "byte"
                        }
                    }
                },
                hsla: {
                    props: {
                        hue: {
                            idx: 0,
                            type: "degrees"
                        },
                        saturation: {
                            idx: 1,
                            type: "percent"
                        },
                        lightness: {
                            idx: 2,
                            type: "percent"
                        }
                    }
                }
            }, d = {
                "byte": {
                    floor: !0,
                    max: 255
                },
                percent: {
                    max: 1
                },
                degrees: {
                    mod: 360,
                    floor: !0
                }
            }, h = c.support = {}, p = e("<p>")[0], f = e.each;
            p.style.cssText = "background-color:rgba(1,1,1,.5)", h.rgba = p.style.backgroundColor.indexOf("rgba")>-1, f(u, function(e, t) {
                t.cache = "_" + e, t.props.alpha = {
                    idx: 3,
                    type: "percent",
                    def: 1
                }
            }), c.fn = e.extend(c.prototype, {
                parse: function(o, r, a, l) {
                    if (o === t)
                        return this._rgba = [null, null, null, null], this;
                    (o.jquery || o.nodeType) && (o = e(o).css(r), r = t);
                    var d = this, h = e.type(o), p = this._rgba = [];
                    return r !== t && (o = [o, r, a, l], h = "array"), "string" === h ? this.parse(n(o) || s._default) : "array" === h ? (f(u.rgba.props, function(e, t) {
                        p[t.idx] = i(o[t.idx], t)
                    }), this) : "object" === h ? (o instanceof c ? f(u, function(e, t) {
                        o[t.cache] && (d[t.cache] = o[t.cache].slice())
                    }) : f(u, function(t, n) {
                        var s = n.cache;
                        f(n.props, function(e, t) {
                            if (!d[s] && n.to) {
                                if ("alpha" === e || null == o[e])
                                    return;
                                d[s] = n.to(d._rgba)
                            }
                            d[s][t.idx] = i(o[e], t, !0)
                        }), d[s] && e.inArray(null, d[s].slice(0, 3)) < 0 && (d[s][3] = 1, n.from && (d._rgba = n.from(d[s])))
                    }), this) : void 0
                },
                is: function(e) {
                    var t = c(e), i=!0, n = this;
                    return f(u, function(e, o) {
                        var s, r = t[o.cache];
                        return r && (s = n[o.cache] || o.to && o.to(n._rgba) || [], f(o.props, function(e, t) {
                            return null != r[t.idx] ? i = r[t.idx] === s[t.idx] : void 0
                        })), i
                    }), i
                },
                _space: function() {
                    var e = [], t = this;
                    return f(u, function(i, n) {
                        t[n.cache] && e.push(i)
                    }), e.pop()
                },
                transition: function(e, t) {
                    var n = c(e), o = n._space(), s = u[o], r = 0 === this.alpha() ? c("transparent"): this, a = r[s.cache] || s.to(r._rgba), l = a.slice();
                    return n = n[s.cache], f(s.props, function(e, o) {
                        var s = o.idx, r = a[s], c = n[s], u = d[o.type] || {};
                        null !== c && (null === r ? l[s] = c : (u.mod && (c - r > u.mod / 2 ? r += u.mod : r - c > u.mod / 2 && (r -= u.mod)), l[s] = i((c - r) * t + r, o)))
                    }), this[o](l)
                },
                blend: function(t) {
                    if (1 === this._rgba[3])
                        return this;
                    var i = this._rgba.slice(), n = i.pop(), o = c(t)._rgba;
                    return c(e.map(i, function(e, t) {
                        return (1 - n) * o[t] + n * e
                    }))
                },
                toRgbaString: function() {
                    var t = "rgba(", i = e.map(this._rgba, function(e, t) {
                        return null == e ? t > 2 ? 1 : 0 : e
                    });
                    return 1 === i[3] && (i.pop(), t = "rgb("), t + i.join() + ")"
                },
                toHslaString: function() {
                    var t = "hsla(", i = e.map(this.hsla(), function(e, t) {
                        return null == e && (e = t > 2 ? 1 : 0), t && 3 > t && (e = Math.round(100 * e) + "%"), e
                    });
                    return 1 === i[3] && (i.pop(), t = "hsl("), t + i.join() + ")"
                },
                toHexString: function(t) {
                    var i = this._rgba.slice(), n = i.pop();
                    return t && i.push(~~(255 * n)), "#" + e.map(i, function(e) {
                        return e = (e || 0).toString(16), 1 === e.length ? "0" + e : e
                    }).join("")
                },
                toString: function() {
                    return 0 === this._rgba[3] ? "transparent" : this.toRgbaString()
                }
            }), c.fn.parse.prototype = c.fn, u.hsla.to = function(e) {
                if (null == e[0] || null == e[1] || null == e[2])
                    return [null, null, null, e[3]];
                var t, i, n = e[0] / 255, o = e[1] / 255, s = e[2] / 255, r = e[3], a = Math.max(n, o, s), l = Math.min(n, o, s), c = a - l, u = a + l, d = .5 * u;
                return t = l === a ? 0 : n === a ? 60 * (o - s) / c + 360 : o === a ? 60 * (s - n) / c + 120 : 60 * (n - o) / c + 240, i = 0 === c ? 0 : .5 >= d ? c / u : c / (2 - u), [Math.round(t)%360, i, d, null == r ? 1: r]
            }, u.hsla.from = function(e) {
                if (null == e[0] || null == e[1] || null == e[2])
                    return [null, null, null, e[3]];
                var t = e[0] / 360, i = e[1], n = e[2], s = e[3], r = .5 >= n ? n * (1 + i): n + i - n * i, a = 2 * n - r;
                return [Math.round(255 * o(a, r, t + 1 / 3)), Math.round(255 * o(a, r, t)), Math.round(255 * o(a, r, t - 1 / 3)), s]
            }, f(u, function(n, o) {
                var s = o.props, r = o.cache, l = o.to, u = o.from;
                c.fn[n] = function(n) {
                    if (l&&!this[r] && (this[r] = l(this._rgba)), n === t)
                        return this[r].slice();
                    var o, a = e.type(n), d = "array" === a || "object" === a ? n: arguments, h = this[r].slice();
                    return f(s, function(e, t) {
                        var n = d["object" === a ? e: t.idx];
                        null == n && (n = h[t.idx]), h[t.idx] = i(n, t)
                    }), u ? (o = c(u(h)), o[r] = h, o) : c(h)
                }, f(s, function(t, i) {
                    c.fn[t] || (c.fn[t] = function(o) {
                        var s, r = e.type(o), l = "alpha" === t ? this._hsla ? "hsla": "rgba": n, c = this[l](), u = c[i.idx];
                        return "undefined" === r ? u : ("function" === r && (o = o.call(this, u), r = e.type(o)), null == o && i.empty ? this : ("string" === r && (s = a.exec(o), s && (o = u + parseFloat(s[2]) * ("+" === s[1] ? 1 : - 1))), c[i.idx] = o, this[l](c)))
                    })
                })
            }), c.hook = function(t) {
                var i = t.split(" ");
                f(i, function(t, i) {
                    e.cssHooks[i] = {
                        set: function(t, o) {
                            var s, r, a = "";
                            if ("transparent" !== o && ("string" !== e.type(o) || (s = n(o)))) {
                                if (o = c(s || o), !h.rgba && 1 !== o._rgba[3]) {
                                    for (r = "backgroundColor" === i ? t.parentNode : t; ("" === a || "transparent" === a) && r && r.style;)
                                        try {
                                            a = e.css(r, "backgroundColor"), r = r.parentNode
                                    } catch (l) {}
                                    o = o.blend(a && "transparent" !== a ? a : "_default")
                                }
                                o = o.toRgbaString()
                            }
                            try {
                                t.style[i] = o
                            } catch (l) {}
                        }
                    }, e.fx.step[i] = function(t) {
                        t.colorInit || (t.start = c(t.elem, i), t.end = c(t.end), t.colorInit=!0), e.cssHooks[i].set(t.elem, t.start.transition(t.end, t.pos))
                    }
                })
            }, c.hook(r), e.cssHooks.borderColor = {
                expand: function(e) {
                    var t = {};
                    return f(["Top", "Right", "Bottom", "Left"], function(i, n) {
                        t["border" + n + "Color"] = e
                    }), t
                }
            }, s = e.Color.names = {
                aqua: "#00ffff",
                black: "#000000",
                blue: "#0000ff",
                fuchsia: "#ff00ff",
                gray: "#808080",
                green: "#008000",
                lime: "#00ff00",
                maroon: "#800000",
                navy: "#000080",
                olive: "#808000",
                purple: "#800080",
                red: "#ff0000",
                silver: "#c0c0c0",
                teal: "#008080",
                white: "#ffffff",
                yellow: "#ffff00",
                transparent: [null, null, null, 0],
                _default: "#ffffff"
            }
        }(e), function() {
            function n(e) {
                var i, n, o = e.ownerDocument.defaultView ? e.ownerDocument.defaultView.getComputedStyle(e, null): e.currentStyle, s = {};
                if (o && o.length && o[0] && o[o[0]])
                    for (n = o.length; n--;)
                        i = o[n], "string" == typeof o[i] && (s[t.camelCase(i)] = o[i]);
                else 
                    for (i in o)
                        "string" == typeof o[i] && (s[i] = o[i]);
                return s
            }
            function o(e, i) {
                var n, o, s = {};
                for (n in i)
                    o = i[n], e[n] !== o && (r[n] || (t.fx.step[n] ||!isNaN(parseFloat(o))) && (s[n] = o));
                return s
            }
            var s = ["add", "remove", "toggle"], r = {
                border: 1,
                borderBottom: 1,
                borderColor: 1,
                borderLeft: 1,
                borderRight: 1,
                borderTop: 1,
                borderWidth: 1,
                margin: 1,
                padding: 1
            };
            t.each(["borderLeftStyle", "borderRightStyle", "borderBottomStyle", "borderTopStyle"], function(i, n) {
                t.fx.step[n] = function(t) {
                    ("none" !== t.end&&!t.setAttr || 1 === t.pos&&!t.setAttr) && (e.style(t.elem, n, t.end), t.setAttr=!0)
                }
            }), t.fn.addBack || (t.fn.addBack = function(e) {
                return this.add(null == e ? this.prevObject : this.prevObject.filter(e))
            }), t.effects.animateClass = function(e, i, r, a) {
                var l = t.speed(i, r, a);
                return this.queue(function() {
                    var i, r = t(this), a = r.attr("class") || "", c = l.children ? r.find("*").addBack(): r;
                    c = c.map(function() {
                        var e = t(this);
                        return {
                            el: e,
                            start: n(this)
                        }
                    }), i = function() {
                        t.each(s, function(t, i) {
                            e[i] && r[i + "Class"](e[i])
                        })
                    }, i(), c = c.map(function() {
                        return this.end = n(this.el[0]), this.diff = o(this.start, this.end), this
                    }), r.attr("class", a), c = c.map(function() {
                        var e = this, i = t.Deferred(), n = t.extend({}, l, {
                            queue: !1,
                            complete: function() {
                                i.resolve(e)
                            }
                        });
                        return this.el.animate(this.diff, n), i.promise()
                    }), t.when.apply(t, c.get()).done(function() {
                        i(), t.each(arguments, function() {
                            var e = this.el;
                            t.each(this.diff, function(t) {
                                e.css(t, "")
                            })
                        }), l.complete.call(r[0])
                    })
                })
            }, t.fn.extend({
                addClass: function(e) {
                    return function(i, n, o, s) {
                        return n ? t.effects.animateClass.call(this, {
                            add: i
                        }, n, o, s) : e.apply(this, arguments)
                    }
                }(t.fn.addClass),
                removeClass: function(e) {
                    return function(i, n, o, s) {
                        return arguments.length > 1 ? t.effects.animateClass.call(this, {
                            remove: i
                        }, n, o, s) : e.apply(this, arguments)
                    }
                }(t.fn.removeClass),
                toggleClass: function(e) {
                    return function(n, o, s, r, a) {
                        return "boolean" == typeof o || o === i ? s ? t.effects.animateClass.call(this, o ? {
                            add: n
                        } : {
                            remove: n
                        }, s, r, a) : e.apply(this, arguments) : t.effects.animateClass.call(this, {
                            toggle: n
                        }, o, s, r)
                    }
                }(t.fn.toggleClass),
                switchClass: function(e, i, n, o, s) {
                    return t.effects.animateClass.call(this, {
                        add: i,
                        remove: e
                    }, n, o, s)
                }
            })
        }(), function() {
            function e(e, i, n, o) {
                return t.isPlainObject(e) && (i = e, e = e.effect), e = {
                    effect: e
                }, null == i && (i = {}), t.isFunction(i) && (o = i, n = null, i = {}), ("number" == typeof i || t.fx.speeds[i]) && (o = n, n = i, i = {}), t.isFunction(n) && (o = n, n = null), i && t.extend(e, i), n = n || i.duration, e.duration = t.fx.off ? 0 : "number" == typeof n ? n : n in t.fx.speeds ? t.fx.speeds[n] : t.fx.speeds._default, e.complete = o || i.complete, e
            }
            function o(e) {
                return !e || "number" == typeof e || t.fx.speeds[e]?!0 : "string" != typeof e || t.effects.effect[e] ? t.isFunction(e)?!0 : "object" != typeof e || e.effect?!1 : !0 : !0
            }
            t.extend(t.effects, {
                version: "1.10.2",
                save: function(e, t) {
                    for (var i = 0; i < t.length; i++)
                        null !== t[i] && e.data(n + t[i], e[0].style[t[i]])
                },
                restore: function(e, t) {
                    var o, s;
                    for (s = 0; s < t.length; s++)
                        null !== t[s] && (o = e.data(n + t[s]), o === i && (o = ""), e.css(t[s], o))
                },
                setMode: function(e, t) {
                    return "toggle" === t && (t = e.is(":hidden") ? "show" : "hide"), t
                },
                getBaseline: function(e, t) {
                    var i, n;
                    switch (e[0]) {
                    case"top":
                        i = 0;
                        break;
                    case"middle":
                        i = .5;
                        break;
                    case"bottom":
                        i = 1;
                        break;
                    default:
                        i = e[0] / t.height
                    }
                    switch (e[1]) {
                    case"left":
                        n = 0;
                        break;
                    case"center":
                        n = .5;
                        break;
                    case"right":
                        n = 1;
                        break;
                    default:
                        n = e[1] / t.width
                    }
                    return {
                        x: n,
                        y: i
                    }
                },
                createWrapper: function(e) {
                    if (e.parent().is(".ui-effects-wrapper"))
                        return e.parent();
                    var i = {
                        width: e.outerWidth(!0),
                        height: e.outerHeight(!0),
                        "float": e.css("float")
                    }, n = t("<div></div>").addClass("ui-effects-wrapper").css({
                        fontSize: "100%",
                        background: "transparent",
                        border: "none",
                        margin: 0,
                        padding: 0
                    }), o = {
                        width: e.width(),
                        height: e.height()
                    }, s = document.activeElement;
                    try {
                        s.id
                    } catch (r) {
                        s = document.body
                    }
                    return e.wrap(n), (e[0] === s || t.contains(e[0], s)) && t(s).focus(), n = e.parent(), "static" === e.css("position") ? (n.css({
                        position: "relative"
                    }), e.css({
                        position: "relative"
                    })) : (t.extend(i, {
                        position: e.css("position"),
                        zIndex: e.css("z-index")
                    }), t.each(["top", "left", "bottom", "right"], function(t, n) {
                        i[n] = e.css(n), isNaN(parseInt(i[n], 10)) && (i[n] = "auto")
                    }), e.css({
                        position: "relative",
                        top: 0,
                        left: 0,
                        right: "auto",
                        bottom: "auto"
                    })), e.css(o), n.css(i).show()
                },
                removeWrapper: function(e) {
                    var i = document.activeElement;
                    return e.parent().is(".ui-effects-wrapper") && (e.parent().replaceWith(e), (e[0] === i || t.contains(e[0], i)) && t(i).focus()), e
                },
                setTransition: function(e, i, n, o) {
                    return o = o || {}, t.each(i, function(t, i) {
                        var s = e.cssUnit(i);
                        s[0] > 0 && (o[i] = s[0] * n + s[1])
                    }), o
                }
            }), t.fn.extend({
                effect: function() {
                    function i(e) {
                        function i() {
                            t.isFunction(s) && s.call(o[0]), t.isFunction(e) && e()
                        }
                        var o = t(this), s = n.complete, a = n.mode;
                        (o.is(":hidden") ? "hide" === a : "show" === a) ? (o[a](), i()) : r.call(o[0], n, i)
                    }
                    var n = e.apply(this, arguments), o = n.mode, s = n.queue, r = t.effects.effect[n.effect];
                    return t.fx.off ||!r ? o ? this[o](n.duration, n.complete) : this.each(function() {
                        n.complete && n.complete.call(this)
                    }) : s===!1 ? this.each(i) : this.queue(s || "fx", i)
                },
                show: function(t) {
                    return function(i) {
                        if (o(i))
                            return t.apply(this, arguments);
                        var n = e.apply(this, arguments);
                        return n.mode = "show", this.effect.call(this, n)
                    }
                }(t.fn.show),
                hide: function(t) {
                    return function(i) {
                        if (o(i))
                            return t.apply(this, arguments);
                        var n = e.apply(this, arguments);
                        return n.mode = "hide", this.effect.call(this, n)
                    }
                }(t.fn.hide),
                toggle: function(t) {
                    return function(i) {
                        if (o(i) || "boolean" == typeof i)
                            return t.apply(this, arguments);
                        var n = e.apply(this, arguments);
                        return n.mode = "toggle", this.effect.call(this, n)
                    }
                }(t.fn.toggle),
                cssUnit: function(e) {
                    var i = this.css(e), n = [];
                    return t.each(["em", "px", "%", "pt"], function(e, t) {
                        i.indexOf(t) > 0 && (n = [parseFloat(i), t])
                    }), n
                }
            })
        }(), function() {
            var e = {};
            t.each(["Quad", "Cubic", "Quart", "Quint", "Expo"], function(t, i) {
                e[i] = function(e) {
                    return Math.pow(e, t + 2)
                }
            }), t.extend(e, {
                Sine: function(e) {
                    return 1 - Math.cos(e * Math.PI / 2)
                },
                Circ: function(e) {
                    return 1 - Math.sqrt(1 - e * e)
                },
                Elastic: function(e) {
                    return 0 === e || 1 === e ? e : - Math.pow(2, 8 * (e - 1)) * Math.sin((80 * (e - 1) - 7.5) * Math.PI / 15)
                },
                Back: function(e) {
                    return e * e * (3 * e - 2)
                },
                Bounce: function(e) {
                    for (var t, i = 4; e < ((t = Math.pow(2, --i)) - 1) / 11;);
                    return 1 / Math.pow(4, 3 - i) - 7.5625 * Math.pow((3 * t - 2) / 22 - e, 2)
                }
            }), t.each(e, function(e, i) {
                t.easing["easeIn" + e] = i, t.easing["easeOut" + e] = function(e) {
                    return 1 - i(1 - e)
                }, t.easing["easeInOut" + e] = function(e) {
                    return .5 > e ? i(2 * e) / 2 : 1 - i( - 2 * e + 2) / 2
                }
            })
        }()
    }(e)
}), _wAMD.define("jquery-ui/effect-slide", ["jquery", "./effect"], function(e) {
    !function(e) {
        e.effects.effect.slide = function(t, i) {
            var n, o = e(this), s = ["position", "top", "bottom", "left", "right", "width", "height"], r = e.effects.setMode(o, t.mode || "show"), a = "show" === r, l = t.direction || "left", c = "up" === l || "down" === l ? "top": "left", u = "up" === l || "left" === l, d = {};
            e.effects.save(o, s), o.show(), n = t.distance || o["top" === c ? "outerHeight": "outerWidth"](!0), e.effects.createWrapper(o).css({
                overflow: "hidden"
            }), a && o.css(c, u ? isNaN(n) ? "-" + n : - n : n), d[c] = (a ? u ? "+=" : "-=" : u ? "-=" : "+=") + n, o.animate(d, {
                queue: !1,
                duration: t.duration,
                easing: t.easing,
                complete: function() {
                    "hide" === r && o.hide(), e.effects.restore(o, s), e.effects.removeWrapper(o), i()
                }
            })
        }
    }(e)
}), _wAMD.define("userhome/billing/TrialExpiredView", ["jquery", "backbone-all", "mustache", "tpl!userhome/billing/trial-expired", "jquery-ui/effect-slide"], function(e, t, i, n) {
    var o = t.Marionette.ItemView.extend({
        id: "showExpiredTrial",
        dialog: {
            dialogClass: "trial-expired-dialog",
            cancelX: !0
        },
        events: {
            "click @ui.planChoices": "selectPlan",
            "click @ui.planButtons": "selectPlan",
            "click @ui.compareLink": "showComparison",
            "click @ui.backArrow": "showPlans",
            "mouseover @ui.questionIcons": "showTooltip",
            "mouseout @ui.questionIcons": "hideTooltip",
            "mouseover @ui.planTableCells": "highlightColumn",
            "mouseover @ui.planTableHeaders": "highlightColumn",
            "mouseout @ui.comparisonTable": "clearColumnHighlight"
        },
        template: i.compile(n),
        templateHelpers: function() {
            return {
                up_to: function() {
                    return function(e, t) {
                        return _W.tl("Up to %i").replace("%i", t(e))
                    }
                }
            }
        },
        ui: {
            planTooltip: "#planTooltip",
            planTableHeaders: "#planComparison th",
            planTableCells: "#planComparison td",
            trialPlans: "#showExpiredTrialPlans",
            trialPlanComparison: "#showExpiredTrialComparison",
            backArrow: ".w-icon-arrow-left",
            planChoices: ".planChoice",
            planButtons: ".planHeader .btn-flat",
            questionIcons: ".w-icon-question",
            comparisonTable: "#planComparison",
            compareLink: "#compareLink"
        },
        initialize: function(e) {
            var i = this;
            i.siteId = e.siteId, i.setupTransactionHandler(), i.model = new t.Model({
                legacy_commerce: "",
                mention_existingDomains: "",
                business_product_limit: "",
                pro_product_limit: "",
                starter_product_limit: "",
                business_transaction_fee: "",
                pro_transaction_fee: "",
                starter_transaction_fee: ""
            }), Weebly.Userhome.RPC.Billing.getPlanSpecifics().done(function(e) {
                i.model.set(e), i.render()
            })
        },
        onClose: function() {
            this.removeTransactionHandler()
        },
        highlightColumn: function(t) {
            var i = e(t.currentTarget), n = i.parents("tr").find(i.get(0).tagName).index(i);
            n > 0 && (this.clearColumnHighlight(), this.ui.comparisonTable.find("tr th:nth-child(" + (n + 1) + ")").addClass("highlight"), this.ui.comparisonTable.find("tr td:nth-child(" + (n + 1) + ")").addClass("highlight"))
        },
        clearColumnHighlight: function() {
            this.ui.planTableHeaders.removeClass("highlight"), this.ui.planTableCells.removeClass("highlight")
        },
        cancel: function() {
            this.trigger("dialog:close")
        },
        selectPlan: function(t) {
            var i = this, n = e(t.currentTarget), o = Weebly.Restrictions.starterLevel;
            switch (i.ui.planChoices.removeClass("selected"), n.addClass("selected"), n.attr("plan")) {
            case"pro":
                o = Weebly.Restrictions.proLevel;
                break;
            case"business":
                o = Weebly.Restrictions.businessLevel
            }
            showPremiumPurchaseScreen({
                siteID: i.siteId,
                refer: "site_row",
                level: o
            })
        },
        showComparison: function() {
            var e = this;
            e.ui.trialPlans.toggle("slide", {
                direction: "left"
            }, 500, function() {
                e.ui.trialPlanComparison.toggle("slide", {
                    direction: "right"
                }, 500)
            }), e.ui.backArrow.show()
        },
        showPlans: function() {
            var e = this;
            e.ui.trialPlanComparison.toggle("slide", {
                direction: "right"
            }, 500, function() {
                e.ui.trialPlans.toggle("slide", {
                    direction: "left"
                }, 500)
            }), e.ui.backArrow.hide()
        },
        showTooltip: function(t) {
            var i = e(t.currentTarget), n = i.position();
            i.parent().addClass("activeTooltip"), this.ui.planTooltip.html(i.attr("tooltip")), this.ui.planTooltip.css({
                left: n.left + 43,
                top: n.top - this.ui.planTooltip.height() / 2 - 15
            }), this.ui.planTooltip.show()
        },
        hideTooltip: function() {
            this.ui.planTooltip.hide(), this.ui.planTableCells.removeClass("activeTooltip")
        },
        setupTransactionHandler: function() {
            Weebly.Transactions || (Weebly.Transactions = {}), Weebly.Transactions.successBillingPremium = _.bind(this.handleBillingUpdate, this), Weebly.Transactions.successBillingExtend = _.bind(this.handleBillingUpdate, this)
        },
        removeTransactionHandler: function() {
            delete Weebly.Transactions.successBillingPremium, delete Weebly.Transactions.successBillingExtend
        },
        handleBillingUpdate: function() {
            window.location.href = "toSite.php?site=" + this.siteId
        }
    });
    return o
}), _wAMD.define("userhome/billing/module", ["userhome/app", "./TrialExpiredView"], function(e, t) {
    e.module("billing", function(e, i) {
        e.showTrialExpired = function(e) {
            i.showFullscreenModal(new t({
                siteId: e
            }))
        }
    })
}), function(e) {
    function t(t) {
        return e(t).each(function(t, i) {
            var n = e(i), o = n.data("wLoadButton.opt");
            o && n.addClass(o.loadingClass), n.attr({
                disabled: "disabled"
            })
        })
    }
    function i(t) {
        return e(t).each(function(t, i) {
            var n = e(i), o = n.data("wLoadButton.opt") || {
                timeout: 0
            };
            setTimeout(function() {
                n.removeAttr("disabled"), o && (o.loadingClass && n.removeClass(o.loadingClass), o.successIconClass && n.find(o.successIconClass).show(), o.startClass && n.removeClass(o.startClass), o.endText && n.find(".text").text(o.endText))
            }, o.timeout)
        })
    }
    function n(t) {
        return e(t).each(function(t, i) {
            var n = e(i), o = n.data("wLoadButton.opt"), s = n.data("wLoadButton.original");
            n.removeAttr("disabled"), o && o.successIconClass && n.find(o.successIconClass).hide(), s && (s.hasStartClass && n.addClass(o.startClass), s.text && n.find(".text").text(s.text))
        })
    }
    "undefined" == typeof e.fn.wLoadButton && (e.fn.wLoadButton = function(o) {
        var s = this;
        if ("startWait" == o)
            return t(s);
        if ("endWait" == o)
            return i(s);
        if ("reset" == o)
            return n(s);
        var r;
        return o = e.extend({}, e.fn.wLoadButton.defaults, o), this.each(function(t, i) {
            $el = e(i), $el.data("wLoadButton.opt", o), ($el.hasClass("btn") || $el.hasClass("btn-com") || $el.hasClass("btn-flat")) && ($el.data("wLoadButton.original", {
                text: $el.find(".text").text(),
                hasStartClass: $el.hasClass(o.startClass)
            }), $el.find(o.successIconClass).hide(), 0 == $el.find("." + o.indicatorClass).length && (r = e("<span></span>").addClass(o.indicatorClass), $el.append(r)))
        })
    }, e.fn.wLoadButton.defaults = {
        loadingClass: "btn-loading",
        indicatorClass: "spinner",
        successIconClass: ".icon",
        startClass: "btn-success",
        timeout: 0,
        endText: !1
    })
}(window.Weebly && window.Weebly.jQuery || window.jQuery), _wAMD.define("util/ui/wLoadButton", function() {}), _wAMD.define("tpl!userhome/domains/google/unlink", [], function() {
    return "{{#tl}}Remove Weebly's ability to manage {{domain}}?{{/tl}}"
}), _wAMD.define("userhome/domains/views/google/Unlink", ["jquery", "backbone-all", "mustache", "userhome/app", "util/ui/wLoadButton", "tpl!userhome/domains/google/unlink"], function(e, t, i, n, o, s) {
    var r = t.Marionette.ItemView.extend({
        template: i.compile(s),
        dialog: {
            dialogClass: "domains-dialog",
            title: _W.tl("Unlink Domain"),
            buttons: [{
                group: "pull-right",
                buttons: [{
                    text: _W.tl("Cancel"),
                    className: "btn-com close-dialog"
                }, {
                    text: _W.tl("Unlink"),
                    className: "btn-com btn-danger unlink-domain-btn"
                }
                ]
            }
            ]
        },
        events: {
            "click .unlink-domain-btn": "unlinkDomain",
            "click .close-dialog": "closeDialog"
        },
        unlinkDomain: function() {
            var e = this.$el.find(".unlink-domain-btn").wLoadButton();
            e.wLoadButton("startWait"), Weebly.Userhome.RPC.Domains.unlinkDomain(this.model.get("domain")).done(function() {
                window.location.reload()
            })
        },
        closeDialog: function() {
            this.trigger("dialog:close")
        }
    });
    return r
}), _wAMD.define("tpl!userhome/domains/google/add-subdomain", [], function() {
    return '<div class="add-subdomain dialog-body">\n	<p>{{#tl}}Subdomain Name:{{/tl}}</p>\n	<input type="text" name="subdomain" id="subdomain-input" />\n	<span>.{{domain}}</span>\n	<p id="subdomain-error"  style="display:none;">You need to enter a valid subdomain name.</p>\n</div>\n'
}), _wAMD.define("userhome/domains/views/google/AddSubdomain", ["jquery", "backbone-all", "mustache", "userhome/app", "util/ui/wLoadButton", "tpl!userhome/domains/google/add-subdomain"], function(e, t, i, n, o, s) {
    var r = t.Marionette.ItemView.extend({
        template: i.compile(s),
        ui: {
            subdomain: "#subdomain-input",
            button: ".add-subdomain-btn",
            error: "#subdomain-error"
        },
        dialog: {
            dialogClass: "domains-dialog",
            title: _W.tl("Add Subdomain"),
            buttons: [{
                group: "pull-right",
                buttons: [{
                    text: _W.tl("Cancel"),
                    className: "btn-com close-dialog"
                }, {
                    text: _W.tl("Add"),
                    className: "btn-com btn-success add-subdomain-btn"
                }
                ]
            }
            ]
        },
        events: {
            "click .add-subdomain-btn": "addSubdomain",
            "click .close-dialog": "closeDialog"
        },
        addSubdomain: function() {
            var e = this.ui.subdomain.val();
            if (/^[0-9a-z][0-9a-z-]{1,18}[0-9a-z]$/i.test(e)) {
                var t = this.ui.button.wLoadButton();
                t.wLoadButton("startWait"), Weebly.Userhome.RPC.Domains.setupSubdomain(this.model.get("domain"), e).done(function(e) {
                    window.location.href = e
                })
            } else 
                this.ui.error.show()
        },
        closeDialog: function() {
            this.trigger("dialog:close")
        }
    });
    return r
}), _wAMD.define("tpl!userhome/domains/google/unlink-subdomain", [], function() {
    return "{{#tl}}Remove Weebly's ability to manage subdomain {{subdomain}}.{{domain}}?{{/tl}}\n"
}), _wAMD.define("userhome/domains/views/google/UnlinkSubdomain", ["jquery", "backbone-all", "mustache", "userhome/app", "util/ui/wLoadButton", "tpl!userhome/domains/google/unlink-subdomain"], function(e, t, i, n, o, s) {
    var r = t.Marionette.ItemView.extend({
        template: i.compile(s),
        dialog: {
            dialogClass: "domains-dialog",
            title: _W.tl("Unlink Subdomain"),
            buttons: [{
                group: "pull-right",
                buttons: [{
                    text: _W.tl("Cancel"),
                    className: "btn-com close-dialog"
                }, {
                    text: _W.tl("Unlink"),
                    className: "btn-com btn-danger unlink-subdomain-btn"
                }
                ]
            }
            ]
        },
        events: {
            "click .unlink-subdomain-btn": "unlinkSubdomain",
            "click .close-dialog": "closeDialog"
        },
        unlinkSubdomain: function() {
            var e = this.$el.find(".unlink-subdomain-btn").wLoadButton();
            e.wLoadButton("startWait"), Weebly.Userhome.RPC.Domains.unlinkSubdomain(this.model.get("domain"), this.model.get("subdomain")).done(function() {
                window.location.reload()
            })
        },
        closeDialog: function() {
            this.trigger("dialog:close")
        }
    });
    return r
}), _wAMD.define("userhome/domains/module", ["userhome/app", "./views/google/Unlink", "./views/google/AddSubdomain", "./views/google/UnlinkSubdomain"], function(e, t, i, n) {
    e.module("domains", function(e, o, s) {
        e.on("start", function() {}), e.unlinkGoogleDomain = function(e) {
            o.showDialog(new t({
                model: new s.Model({
                    domain: e
                })
            }))
        }, e.addGoogleSubdomain = function(e) {
            o.showDialog(new i({
                model: new s.Model({
                    domain: e
                })
            }))
        }, e.unlinkGoogleSubdomain = function(e, t) {
            o.showDialog(new n({
                model: new s.Model({
                    domain: e,
                    subdomain: t
                })
            }))
        }
    })
}), _wAMD.require(["jquery", "mustache", "underscore", "backbone", "backbone-validation", "vendor/jsonrpc", "legacy/weebly_restrictions", "legacy/accordion", "legacy/weebly_dialog", "legacy/jsonp", "legacy/weebly_tooltip", "legacy/transitions", "legacy/jquery_utils", "legacy/weebly_utils", "legacy/weebly_utils_ajax", "legacy/weebly_pages", "legacy/weebly_user_home", "legacy/weebly_user_home_style", "legacy/weebly_dropdowns", "legacy/weebly_domain_utils", "legacy/event.simulate", "legacy/weebly_feedback", "legacy/weebly_storage", "legacy/weebly_adsense", "backgrid-all"], function() {
    _wAMD.require(["jquery", "userhome/app", "userhome/billing/module", "userhome/domains/module"], function(e, t) {
        "function" == typeof initUserHomeRPC && initUserHomeRPC(), e(function() {
            t.start()
        })
    })
}), _wAMD.define("userhome/main", function() {});
