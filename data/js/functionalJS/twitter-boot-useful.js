define("app/ui/dialogs/age_gate_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "core/i18n", "core/utils", "app/utils/cookie"], function(module, require, exports) {
    function ageGateDialog() {
        this.defaultAttrs({
            submitButtonSelector: "#age-gate-dialog-submit-button",
            monthSelector: "#age-gate-month",
            daySelector: "#age-gate-day",
            yearSelector: "#age-gate-year",
            requiredSelector: ".age-gate-error",
            ageGateCookieKey: "age-gated"
        }), this.setUpDialog = function(a, b) {
            b.ageGated ? this.showError() : (this.reset(), this.ageData = b, this.setDateOption(), this.open())
        }, this.fillYearSelect = function() {
            var a = (new Date).getFullYear();
            for (var b = a; b > a - 100; b--)
                this.yearSelect.options.add(new Option(b, b));
            this.yearSelect.options.add(new Option(_('Before {{year}}', {
                year: b
            }), b))
        }, this.month = {
            1: _('January'),
            2: _('February'),
            3: _('March'),
            4: _('April'),
            5: _('May'),
            6: _('June'),
            7: _('July'),
            8: _('August'),
            9: _('September'),
            10: _('October'),
            11: _('November'),
            12: _('December')
        }, this.fillMonthSelect = function() {
            for (var a = 1; a <= 12; a++)
                this.monthSelect.options.add(new Option(this.month[a], a))
        }, this.fillDaySelect = function(a, b) {
            a || (a = 1), b || (b = 31);
            for (var c = a; c <= b; c++)
                this.daySelect.options.add(new Option(c, c))
        }, this.resetSelect = function(a, b) {
            a.options.length = 0, a.options.add(new Option(b, - 1, !0, !0)), a.options[0].disabled=!0
        }, this.reset = function() {
            this.resetSelect(this.yearSelect, _('Year')), this.resetSelect(this.monthSelect, _('Month')), this.resetSelect(this.daySelect, _('Day')), this.hideRequired()
        }, this.setDateOption = function() {
            this.fillYearSelect(), this.fillMonthSelect(), this.fillDaySelect()
        }, this.showRequired = function() {
            this.select("requiredSelector").removeClass("hidden")
        }, this.hideRequired = function() {
            this.select("requiredSelector").addClass("hidden")
        }, this.getDateFromSelect = function() {
            var a = {};
            return a.year = this.yearSelect.options[this.yearSelect.selectedIndex].value, a.month = this.monthSelect.options[this.monthSelect.selectedIndex].value, a.day = this.daySelect.options[this.daySelect.selectedIndex].value, a
        }, this.getNumberOfDaysGivenMonthYear = function(a, b) {
            return b < 0 && (b = 2012), (new Date(b, a, 0)).getDate()
        }, this.validateDate = function() {
            var a = this.getDateFromSelect(), b = 0 < a.year && 0 < a.month && a.month <= 12 && 0 < a.day && a.day <= this.getNumberOfDaysGivenMonthYear(a.month, a.year);
            return b ? new Date(a.year, a.month - 1, a.day) : (a.year==-1 || a.month==-1 || a.day==-1 ? this.showRequired() : this.trigger("uiShowMessage", {
                message: _('Invalid date. Please input your date of birth.')
            }), null)
        }, this.updateDays = function() {
            var a, b = this.getDateFromSelect();
            this.hideRequired();
            if (b.month < 1 || b.month > 12)
                return;
            this.resetSelect(this.daySelect, "Day"), a = this.getNumberOfDaysGivenMonthYear(b.month, b.year), this.fillDaySelect(1, a), b.day > 0 && b.day <= a && (this.daySelect.selectedIndex = b.day)
        }, this.showError = function() {
            this.trigger("uiShowMessage", {
                message: _('You do not meet the minimum age requirements to follow this account.')
            })
        }, this.setAgeGateCookie = function() {
            cookie(this.attr.ageGateCookieKey, (new Date).getTime())
        }, this.submitAgeGating = function() {
            var a = this.validateDate();
            a && (a.getTime() <= this.ageData.minBirthDate ? this.trigger("uiAgeGatePassed", utils.merge(this.ageData.originalData, {
                passedAgeGating: !0
            })) : (this.showError(), this.setAgeGateCookie()), this.close())
        }, this.after("initialize", function() {
            this.monthSelect = this.select("monthSelector")[0], this.daySelect = this.select("daySelector")[0], this.yearSelect = this.select("yearSelector")[0], this.on(document, "uiNeedsAgeGateDialog", this.setUpDialog), this.on("click", {
                submitButtonSelector: this.submitAgeGating
            }), this.on("change", {
                monthSelector: this.updateDays,
                yearSelector: this.updateDays,
                daySelector: this.hideRequired
            })
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), _ = require("core/i18n"), utils = require("core/utils"), cookie = require("app/utils/cookie"), AgeGateDialog = defineComponent(ageGateDialog, withDialog);
    module.exports = AgeGateDialog
});
define("app/ui/dialogs/block_or_report", ["module", "require", "exports", "core/component", "core/i18n", "app/ui/with_dialog"], function(module, require, exports) {
    function blockOrReportDialog() {
        this.defaultAttrs({
            blockUserLabelSelector: ".block-user-label",
            blockUserTextSelector: ".block-user-text",
            blockButtonSelector: ".report-tweet-block-button",
            reportButtonSelector: ".report-tweet-report-button",
            nextButtonSelector: ".report-tweet-next-button",
            harassmentIframeSelector: "#report-harassment-frame",
            harassmentNextSelector: ".harassment-next-button",
            harassmentBackSelector: ".harassment-back-button",
            harassmentDoneSelector: ".harassment-done-button",
            reportTypeSelector: "input[name=report_type]",
            abusiveTypeSelector: "input[name=abuse_type]",
            selectedReportTypeSelector: "input[name=report_type]:checked",
            selectedAnnoyingSelector: "input[value=annoying]",
            selectedAbusiveTypeSelector: "input[name=abuse_type]:checked",
            blockInputSelector: "input[name=block_user]",
            alsoReportInputSelector: "input[name=also_report]",
            flagMediaSelector: ".flag-media",
            flagMediaTextSelector: ".flag-media-desc",
            spamTextSelector: ".spam-desc",
            abuseTextSelector: ".abuse-desc",
            annoyingTextSelector: ".annoying-desc",
            impersonationLinkSelector: ".report-impersonation-link",
            privateInformationLinkSelector: ".report-private-information-link",
            harassmentLinkSelector: ".report-harassment-link",
            selfHarmLinkSelector: ".report-self-harm-link",
            adsLinkSelector: ".report-ads-link",
            twitterRulesLinkSelector: ".twitter-rules-link",
            reportFormSelector: ".report-form",
            reportControlSelector: "#report-control",
            reportTitleSelector: ".report-title",
            harassmentTitleSelector: ".harassment-title",
            harassmentFormSelector: ".harassment-form",
            reporterUserIdSelector: "#current-user-id",
            abuseLinks: {
                impersonation: "https://support.twitter.com/forms/impersonation?{{q}}",
                privateInformation: "https://support.twitter.com/forms/private_information?{{q}}",
                harassment: "https://support.twitter.com/forms/abusiveuser?{{q}}",
                selfHarm: "https://support.twitter.com/forms/general?subtopic=self_harm&{{q}}",
                ads: "https://support.twitter.com/forms/ads?{{q}}"
            }
        }), this.setUpDialog = function(a, b) {
            this.reset(), this.eventData = b, this.setAlreadyBlocking(), this.setTriggeredTarget(), this.setScreenNameText(), this.setDescriptionText(), this.setAbuseReportLinks(), this.setFlagMedia(), this.showBlockOrReport(), this.setReportHarassmentContext(), this.setReportAds(), this.open()
        }, this.reset = function() {
            this.isComplete=!1, this.$node.addClass("block-selected"), this.select("blockInputSelector").attr("checked", !0), this.select("blockButtonSelector").attr("disabled", !1), this.resetReportSection()
        }, this.resetReportSection = function() {
            this.$node.removeClass("abuse-selected"), this.$node.removeClass("also-report-selected"), this.select("selectedAnnoyingSelector").attr("checked", !0), this.select("alsoReportInputSelector").attr("checked", !1), this.select("selectedAbusiveTypeSelector").attr("checked", !1), this.select("reportButtonSelector").attr("disabled", !0), this.select("nextButtonSelector").attr("disabled", !0), this.select("harassmentDoneSelector").hide()
        }, this.setAlreadyBlocking = function() {
            this.eventData.viewerBlocksAuthor ? (this.$node.addClass("already-blocking"), this.select("blockInputSelector").attr("checked", !1), this.select("alsoReportInputSelector").attr("checked", !0), this.select("reportButtonSelector").attr("disabled", !1), this.updateState()) : this.$node.removeClass("already-blocking")
        }, this.setTriggeredTarget = function() {
            this.eventData.target === "user" ? this.$node.addClass("report-user") : this.$node.removeClass("report-user")
        }, this.setReportHarassmentContext = function() {
            var a = this.select("reporterUserIdSelector").val(), b = {
                baseUrl: "/safety/report_story",
                reporterId: a,
                reportedId: this.eventData.userId,
                reportedTweetId: "",
                source: ""
            }, c = "{{baseUrl}}?source={{source}}&reported_user_id={{reportedId}}&reporter_user_id={{reporterId}}";
            if (this.$node.hasClass("report-user")) {
                b.source = "reportprofile";
                var d = _(c, b);
                this.select("harassmentIframeSelector").attr("src", d)
            } else {
                b.reportedTweetId = this.eventData.tweetId, b.source = "reporttweet";
                var e = c + "&reported_tweet_id={{reportedTweetId}}", f = _(e, b);
                this.select("harassmentIframeSelector").attr("src", f)
            }
        }, this.setScreenNameText = function() {
            var a = this.select("blockUserLabelSelector");
            a.text(_('Block @{{screenName}}', this.eventData)), this.select("blockUserTextSelector").text(_('@{{screenName}} will no longer be able to follow or message you. ', this.eventData))
        }, this.isTriggeredByUser = function(a, b) {
            return b.target === "user"
        }, this.setDescriptionText = function(a, b) {
            var c = this.select("spamTextSelector"), d = this.select("abuseTextSelector"), e = this.select("annoyingTextSelector");
            this.$node.hasClass("report-user") ? (e.text(_('This account is annoying')), c.text(_('This is a spam account')), d.text(_('account'))) : (e.text(_('This Tweet is annoying')), c.text(_('This Tweet is spam')), d.text(_('Tweet')))
        }, this.setFlagMedia = function() {
            var a = this.select("flagMediaSelector");
            this.eventData.mediaType ? (this.eventData.mediaType == "image" ? this.select("flagMediaTextSelector").text(_('This Tweet contains a possibly sensitive image')) : this.eventData.mediaType == "video" ? this.select("flagMediaTextSelector").text(_('This Tweet contains a possibly sensitive video')) : this.select("flagMediaTextSelector").text(_('This Tweet contains possibly sensitive media')), a.show()) : a.hide()
        }, this.setReportAds = function() {
            var a = "for-promoted-tweet";
            this.eventData.disclosureType !== "promoted" ? this.$node.removeClass(a) : this.$node.addClass(a)
        }, this.setAbuseReportLinks = function() {
            var a = "reported_username={{screenName}}&reported_tweet_id={{tweetId}}", b = {
                q: _(a, this.eventData)
            }, c = this.attr.abuseLinks;
            this.select("impersonationLinkSelector").attr("data-form-link", _(c.impersonation, b)), this.select("privateInformationLinkSelector").attr("data-form-link", _(c.privateInformation, b)), this.select("harassmentLinkSelector").attr("data-form-link", _(c.harassment, b)), this.select("selfHarmLinkSelector").attr("data-form-link", _(c.selfHarm, b)), this.select("adsLinkSelector").attr("data-form-link", _(c.ads, b))
        }, this.updateState = function() {
            function a(a, b) {
                var c = b ? "addClass": "removeClass";
                this.$node[c](a)
            }
            function b(a, b) {
                this.select(a).attr("disabled", !b)
            }
            var c = this.blockUserIsChecked(), d = this.alsoReportIsChecked(), e = this.reportType() == "abusive", f = this.select("abusiveTypeSelector").is(":checked"), g = [["block-selected", c], ["also-report-selected", d], ["abuse-selected", e]];
            g.forEach(function(b) {
                a.call(this, b[0], b[1])
            }, this);
            var h = [["blockButtonSelector", c], ["reportButtonSelector", d], ["nextButtonSelector", f]];
            h.forEach(function(a) {
                b.call(this, a[0], a[1])
            }, this), d || this.resetReportSection()
        }, this.reportType = function() {
            return this.select("selectedReportTypeSelector").val()
        }, this.abusiveType = function() {
            return this.select("selectedAbusiveTypeSelector").val()
        }, this.blockUserIsChecked = function() {
            return this.select("blockInputSelector").attr("checked") == "checked"
        }, this.alsoReportIsChecked = function() {
            return this.select("alsoReportInputSelector").attr("checked") == "checked" || this.select("alsoReportInputSelector").length == 0
        }, this.submitReport = function() {
            var a = this.$node.hasClass("report-user") ? "uiReportUserAction": "uiDidBlockOrReport", b = this.alsoReportIsChecked() ? this.reportType(): "";
            this.trigger(a, {
                eventData: this.eventData,
                tweetId: this.eventData.tweetId,
                userId: this.eventData.userId,
                screenName: this.eventData.screenName,
                reportType: b,
                blockUser: this.blockUserIsChecked(),
                impressionId: this.eventData.impressionId,
                disclosureType: this.eventData.disclosureType
            }), this.blockUserIsChecked() && (this.trigger("uiDidTriggerBlockingAction", {
                userId: this.eventData.userId
            }), this.$node.addClass("already-blocking"), this.select("blockInputSelector").attr("checked", !1))
        }, this.showBlockOrReport = function() {
            this.$node.removeClass("abuse-dialog")
        }, this.showHarassmentReport = function() {
            this.$node.addClass("abuse-dialog"), this.select("harassmentBackSelector").show(), this.select("harassmentNextSelector").show()
        }, this.blockOrReport = function() {
            this.submitReport(), this.completeReport()
        }, this.reportAbuseNext = function() {
            this.submitReport();
            if (this.abusiveType()) {
                var a = this.select("selectedAbusiveTypeSelector").attr("data-form-link");
                this.abusiveType() == "harassment" && $(this.attr.selectedAbusiveTypeSelector).hasClass("report-flow-v2") ? (this.reportAbuseToSupport(), this.showHarassmentReport()) : (this.reportAbuseToSupport(), window.open(a, "_blank"), this.completeReport())
            }
        }, this.reportAbuseToSupport = function(a, b) {
            this.trigger("uiDidBlockOrReportToSupport", {
                eventData: this.eventData,
                abuseType: this.abusiveType(),
                userId: this.eventData.userId
            })
        }, this.cancelReport = function() {
            this.trigger("uiDidCancelBlockOrReport", this.eventData)
        }, this.completeReport = function() {
            this.isComplete=!0, this.close()
        }, this.around("close", function(a) {
            this.isOpen()&&!this.isComplete && this.cancelReport(), a()
        }), this.twitterRulesClick = function() {
            this.trigger("uiDidOpenTwitterRulesLink", {
                eventData: this.eventData,
                userId: this.eventData.userId
            })
        }, this.harassmentFrameNext = function() {
            this.select("harassmentIframeSelector").contents().find("#report_webview_form").submit()
        }, this.harassmentFrameBack = function() {
            var a = this.select("harassmentIframeSelector").contents().get(0).location.href.indexOf("next_view") < 0;
            a ? this.showBlockOrReport() : this.select("harassmentIframeSelector").get(0).contentWindow.history.back()
        }, this.checkContentAndUpdate = function() {
            var a = this.select("harassmentIframeSelector").contents().get(0).location.href.indexOf("/safety/report_story_complete") >= 0;
            a ? (this.select("harassmentBackSelector").hide(), this.select("harassmentNextSelector").hide(), this.select("harassmentDoneSelector").show()) : this.select("harassmentDoneSelector").hide()
        }, this.after("initialize", function() {
            this.isComplete=!1, this.on(document, "uiNeedsBlockOrReportDialog", this.setUpDialog);
            var a = this;
            this.select("harassmentIframeSelector").load(function() {
                a.checkContentAndUpdate()
            }), this.on("click", {
                blockButtonSelector: this.blockOrReport,
                reportButtonSelector: this.blockOrReport,
                nextButtonSelector: this.reportAbuseNext,
                twitterRulesLinkSelector: this.twitterRulesClick,
                harassmentNextSelector: this.harassmentFrameNext,
                harassmentBackSelector: this.harassmentFrameBack,
                harassmentDoneSelector: this.close
            }), this.on("change", {
                blockInputSelector: this.updateState,
                alsoReportInputSelector: this.updateState,
                reportTypeSelector: this.updateState,
                abusiveTypeSelector: this.updateState
            })
        })
    }
    var defineComponent = require("core/component"), _ = require("core/i18n"), withDialog = require("app/ui/with_dialog"), BlockOrReportDialog = defineComponent(withDialog, blockOrReportDialog);
    module.exports = BlockOrReportDialog
});
define("app/ui/dialogs/with_modal_tweet", ["module", "require", "exports", "app/data/user_info"], function(module, require, exports) {
    function withModalTweet() {
        this.defaultAttrs({
            modalTweetContainerSelector: ".modal-container",
            modalTweetDoGearSelector: ".dogear",
            modalTweetLoadingClass: "tweet-loading",
            modalTweetSelector: ".modal-tweet",
            modalTweetShowingClass: "tweet-showing",
            modalTweetTimeSelector: ".time",
            modalTweetTextSelector: ".js-tweet-text"
        }), this.modalTweetOptions = {}, this.turnOnShowingTweetClass = function() {
            this.$node.addClass(this.attr.modalTweetShowingClass)
        }, this.turnOffShowingTweetClass = function() {
            this.$node.removeClass(this.attr.modalTweetShowingClass)
        }, this.turnOnLoadingTweetClass = function() {
            this.$node.addClass(this.attr.modalTweetLoadingClass)
        }, this.turnOffLoadingTweetClass = function() {
            this.$node.removeClass(this.attr.modalTweetLoadingClass)
        }, this.requestForTweet = function(a, b) {
            this.displayTweet(b.tweetId, b.tweetOptions)
        }, this.displayTweet = function(a, b) {
            this.modalTweetOptions = b, this.retrieveTweet(a, b)
        }, this.retrieveTweet = function(a, b) {
            this.turnOnLoadingTweetClass(), this.trigger("uiGetTweet", {
                id: a,
                modal: b.modal
            }), this.modalTweetOptions = b
        }, this.gotTweet = function(a, b) {
            this.turnOffLoadingTweetClass(), b.id && b.tweet_html && (this.addTweet(b.tweet_html), this.modifyTweetOnRender(this.modalTweetOptions))
        }, this.modifyTweetOnRender = function(a) {
            a && a.userOnly && (this.$node.find(this.attr.modalTweetTimeSelector).remove(), this.$node.find(this.attr.modalTweetDoGearSelector).remove(), this.$node.find(this.attr.modalTweetTextSelector).remove()), a && a.disableLinks && (this.select("modalTweetSelector").addClass("disabled-links"), this.$node.find("a").attr("tabindex", - 1).click(function(a) {
                a.preventDefault(), a.stopPropagation()
            }))
        }, this.addTweet = function(a) {
            this.select("modalTweetSelector").empty().append(a), this.turnOnShowingTweetClass()
        }, this.removeTweet = function() {
            this.turnOffLoadingTweetClass(), this.turnOffShowingTweetClass(), this.select("modalTweetSelector").empty()
        }, this.after("initialize", function() {
            this.on("uiWantsTweet", this.requestForTweet), this.on(document, "dataGotTweet", this.gotTweet), this.on(document, "uiDialogClosed dataFailedToGetTweet", this.removeTweet)
        })
    }
    var userInfo = require("app/data/user_info");
    module.exports = withModalTweet
});
define("app/ui/dialogs/block_user_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/ui/dialogs/with_modal_tweet"], function(module, require, exports) {
    function blockUserDialog() {
        this.defaults = {
            cancelSelector: ".cancel-action",
            blockSelector: ".block-action"
        }, this.openBlockUser = function(a, b) {
            var c = {
                userOnly: !0,
                modal: "block"
            };
            this.attr.sourceEventData = b, this.displayTweet(b.tweetId, c), this.open()
        }, this.blockUser = function() {
            this.trigger("uiDidBlockUser", {
                sourceEventData: this.attr.sourceEventData
            }), this.close()
        }, this.after("initialize", function() {
            this.on("click", {
                cancelSelector: this.close,
                blockSelector: this.blockUser
            }), this.on(document, "uiOpenBlockUserDialog", this.openBlockUser)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withModalTweet = require("app/ui/dialogs/with_modal_tweet");
    module.exports = defineComponent(blockUserDialog, withDialog, withModalTweet)
});
define("app/ui/with_position", ["module", "require", "exports"], function(module, require, exports) {
    function Position() {
        this.adjacent = function(a, b, c) {
            var d, e;
            c = c || {}, d = e = b.offset(), e.gravity = c.gravity, e.weight = c.weight;
            var f = {
                height: b.outerHeight(),
                width: b.outerWidth()
            }, g = {
                height: a.outerHeight(),
                width: a.outerWidth()
            }, h = {
                height: $(window).height(),
                width: $(window).width()
            }, i = {
                height: $("body").height(),
                width: $("body").width()
            };
            return e.gravity || (e.gravity = "vertical"), "vertical,vertical-north,north,south".indexOf(e.gravity)!=-1 && ("right,left,center".indexOf(e.weight)==-1 && (e.weight = d.left + g.width > h.width ? "right" : "left"), e.gravity == "vertical-north" ? e.gravity = d.top - g.height < $(window).scrollTop() ? "north" : "south" : e.gravity == "vertical" && (e.gravity = d.top + g.height > $(window).scrollTop() + h.height ? "south" : "north"), c.position == "relative" && (d = {
                left: 0,
                top: 0
            }, e.left = 0), e.weight == "right" ? e.left = d.left - g.width + f.width : e.weight == "center" && (e.left = d.left - (g.width - f.width) / 2), e.top = e.gravity == "north" ? d.top + f.height : d.top - g.height), "horizontal,east,west".indexOf(e.gravity)!=-1 && ("top,bottom,center".indexOf(e.weight)==-1 && (d.top - g.height / 2 < 0 ? e.weight = "top" : d.top + g.height / 2 > Math.max(h.height, i.height) ? e.weight = "bottom" : e.weight = "center"), e.gravity == "horizontal" && (e.gravity = d.left + f.width / 2 > h.width / 2 ? "east" : "west"), c.position == "relative" && (d = {
                left: 0,
                top: 0
            }, e.top = 0), e.weight == "center" ? e.top = d.top + f.height / 2 - g.height / 2 : e.weight == "bottom" && (e.top = d.top - g.height + f.height), e.left = e.gravity == "west" ? d.left + f.width : d.left - g.width), e
        }
    }
    module.exports = Position
});
define("app/ui/commerce/with_email_confirmation", ["module", "require", "exports", "core/i18n"], function(module, require, exports) {
    function withEmailConfirmation() {
        this.defaultAttrs({
            emailNotConfirmedClass: "email-not-confirmed",
            confirmDialogOptionData: {
                titleText: _('Verify email address'),
                bodyText: _('In order to proceed, you must first confirm your email address in your account settings.'),
                cancelText: _('Cancel'),
                submitText: _('Go to settings'),
                action: "CommerceUserWantToVerifyEmail"
            }
        }), this.isEmailConfirmedOrHandle = function(a) {
            var b = $(a).closest(this.attr.commerceCardSelector);
            return b.hasClass(this.attr.emailNotConfirmedClass) ? (this.trigger("uiOpenConfirmDialog", this.attr.confirmDialogOptionData), !1) : !0
        }, this.goToVerifyEmail = function() {
            window.location.href = "/settings/account"
        }, this.after("initialize", function() {
            this.on("uiCommerceUserWantToVerifyEmailConfirm", this.goToVerifyEmail)
        })
    }
    var _ = require("core/i18n");
    module.exports = withEmailConfirmation
});
define("app/utils/with_iframe_height_adjuster", ["module", "require", "exports"], function(module, require, exports) {
    function withIframeHeightAdjuster() {
        this.fitIframeHeight = function(a, b) {
            if (!a ||!b)
                return;
            var c = b.iframeSelector || this.attr.iframeSelector, d = {};
            a.originalEvent && (a = a.originalEvent);
            if (a.origin === window.location.protocol + "//" + window.location.host) {
                if (typeof a.data == "string" && a.data != "undefined" && a.data != "[object Object]")
                    try {
                        d = JSON.parse(a.data)
                    } catch (e) {
                    console.log("Failed to JSON.parse: " + a.data)
                }
                $.isFunction(b.isQualified) && b.isQualified(d) && (this.$node.find(c).css("height", d.height + "px"), $.isFunction(b.additonalTreatment) && b.additonalTreatment(this.$node.find(c)))
            }
        }
    }
    module.exports = withIframeHeightAdjuster
});
define("app/ui/dialogs/buy_now_dialog", ["module", "require", "exports", "core/component", "core/utils", "app/ui/with_dialog", "app/ui/with_position", "app/ui/commerce/with_email_confirmation", "app/utils/with_iframe_height_adjuster", "app/data/with_card_metadata"], function(module, require, exports) {
    function buyNowConfirmDialog() {
        this.defaultAttrs({
            buyNowDialogIframeSelector: ".buy-now-card-iframe",
            buyNowActionSelector: ".BuyNowCard-buyNowAction",
            buyNowTweetSelector: ".tweet, .ProfileTweet",
            commerceCardSelector: ".BuyNowCard",
            top: 47,
            defaultHeight: "400px"
        }), this.openDialog = function(a, b) {
            var c, d, e;
            b && b.el && this.isEmailConfirmedOrHandle(b.el) && (e = $(b.el).closest(this.attr.buyNowTweetSelector), this.promotedMetadata = this.extractPromotedMetadata(e), d = e.data("tweet-id"), this.trigger("uiBuyNowCardClicked", utils.merge({
                tweetId: d
            }, this.promotedMetadata)), this.scribingData = this.getCardDataFromTweet(e), c = "/i/pay/status/" + d, this.select("buyNowDialogIframeSelector").attr("src", c), this.open())
        }, this.closeDialog = function(a, b) {
            this.purchaseSuccess || this.trigger("uiCommercePurchaseFlowAborted", this.scribingData), this.purchaseSuccess=!1, this.select("buyNowDialogIframeSelector").attr("src", "about:blank"), this.select("buyNowDialogIframeSelector").css("height", this.attr.defaultHeight)
        }, this.adjustIframeHeight = function(a) {
            var b = {
                iframeSelector: this.attr.buyNowDialogIframeSelector,
                isQualified: function(a) {
                    return a.name && a.name === "buy_now" && a.height
                }
            };
            this.fitIframeHeight(a, b)
        }, this.showSuccessMessage = function(a, b) {
            this.purchaseSuccess=!0, this.trigger("uiCloseDialog"), this.trigger("uiShowMessage", {
                message: b.message
            }), this.trigger("uiCommerceBuyNowSuccess", this.promotedMetadata)
        }, this.extractPromotedMetadata = function(a) {
            return {
                impressionId: a.data("impression-id"),
                disclosureType: a.data("disclosure-type")
            }
        }, this.passBootData = function(a, b) {
            var c = this.scribingData;
            this.select("buyNowDialogIframeSelector").each(function() {
                this.contentWindow.$("body").trigger("uiBuyNowCardDataBoot", c)
            })
        }, this.after("initialize", function() {
            this.purchaseSuccess=!1, this.on(window, "message", this.adjustIframeHeight), this.on("uiBuyNowWaitCardData", this.passBootData), this.on("uiDialogClosed", this.closeDialog), this.on("uiCommerceShowSuccessMessage", this.showSuccessMessage), this.on(document, "click", {
                buyNowActionSelector: this.openDialog
            })
        }), this.before("teardown", function() {
            this.off(window, "message", this.adjustIframeHeight)
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), withDialog = require("app/ui/with_dialog"), withPosition = require("app/ui/with_position"), withEmailConfirmation = require("app/ui/commerce/with_email_confirmation"), withIframeHeightAdjuster = require("app/utils/with_iframe_height_adjuster"), withCardMetadata = require("app/data/with_card_metadata"), BuyNowConfirmDialog = defineComponent(buyNowConfirmDialog, withDialog, withPosition, withIframeHeightAdjuster, withEmailConfirmation, withCardMetadata);
    module.exports = BuyNowConfirmDialog
});
define("app/data/commerce/buy_now_dialog_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe", "app/data/with_card_metadata"], function(module, require, exports) {
    function buyNowDialogScribe() {
        this.scribePurchaseFlowAborted = function(a, b) {
            var c = {
                items: [this.getCard2Item(b)]
            };
            this.scribe({
                page: "buy_now",
                component: "product_detail",
                action: "exit"
            }, c)
        }, this.after("initialize", function() {
            this.on("uiCommercePurchaseFlowAborted", this.scribePurchaseFlowAborted)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), withCardMetadata = require("app/data/with_card_metadata");
    module.exports = defineComponent(buyNowDialogScribe, withCardMetadata, withScribe)
});
define("app/ui/dialogs/captcha_challenge_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog"], function(module, require, exports) {
    function captchaChallengeDialog() {
        this.defaultAttrs({
            challengeSelector: "#recaptcha_challenge_field",
            responseSelector: "#recaptcha_response_field",
            submitSelector: "#recaptcha_submit",
            captchaName: "Captcha",
            bodyTextSelector: ".modal-text",
            captchaFormSelector: "#captcha-challenge-form",
            recaptchaPublicKey: "6LfbTAAAAAAAAE0hk8Vnfd1THHnn9lJuow6fgulO"
        }), this.refreshCaptchaDialog = function(a, b) {
            this.reloadCaptcha(), this.open()
        }, this.reloadCaptcha = function() {
            Recaptcha && Recaptcha.reload()
        }, this.submitCaptcha = function() {
            var a = this.select("challengeSelector").val(), b = this.select("responseSelector").val();
            this.trigger("uiSubmitSpamChallenge", {
                challenge: a,
                response: b,
                challengeName: this.attr.captchaName
            })
        }, this.getRecaptcha = function(a, b) {
            this.isRecaptchaDefined() && Recaptcha.create(this.attr.recaptchaPublicKey, "captcha-challenge-form", {
                theme: "clean",
                callback: function() {
                    Recaptcha.focus_response_field(), a && b.select("bodyTextSelector").text(a), b.open()
                }
            })
        }, this.isRecaptchaDefined = function() {
            return typeof Recaptcha != "undefined"
        }, this.openCaptchaDialog = function(a, b) {
            this.getRecaptcha(b.message, this)
        }, this.after("initialize", function() {
            this.on("click", {
                submitSelector: this.submitCaptcha
            }), this.on(document, "uiOpenSpamCaptchaChallenge", this.openCaptchaDialog), this.on(document, "uiCloseSpamCaptchaChallenge", this.blurAndCloseImmediately), this.on(document, "uiRefreshSpamCaptchaChallenge", this.refreshCaptchaDialog)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog");
    module.exports = defineComponent(captchaChallengeDialog, withDialog)
});
define("app/utils/with_event_params", ["module", "require", "exports", "core/utils", "core/parameterize"], function(module, require, exports) {
    function withEventParams() {
        this.rewriteEventName = function(a) {
            var b = [];
            for (var c = 1; c < arguments.length; c++)
                b[c - 1] = arguments[c];
            var d = typeof b[0] == "string" || b[0].defaultBehavior ? 0: 1, e = b[d], f = e.type || e;
            b[d] = parameterize(f, this.attr.eventParams, !0), e.type && (e.type = b[d], b[d] = e), a.apply(this, b)
        }, this.around("on", this.rewriteEventName), this.around("off", this.rewriteEventName), this.around("trigger", this.rewriteEventName)
    }
    var util = require("core/utils"), parameterize = require("core/parameterize");
    module.exports = withEventParams
});
define("template", ["module"], function(a) {
    var b = {};
    b.swift_tweet_box = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<form class="t1-form tweet-form\n        '), d.s(d.f("condensed_tweet_box", a, b, 1), a, b, 0, 64, 73, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("condensed")
            }), a.pop()), d.b("\n" + c), d.b("        "), d.s(d.f("expanded_one_line_tweet_box", a, b, 1), a, b, 0, 138, 148, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("is-oneLine")
            }), a.pop()), d.b('"\n      method="post"\n      target="tweet-post-iframe"\n'), d.s(d.f("screen_names_for_tweet_box", a, b, 1), a, b, 0, 272, 410, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('        data-default-text="'), c.b(c.v(c.f("screen_names_for_tweet_box", a, b, 0))), c.b(' "\n        data-condensed-text="Reply to '), c.b(c.v(c.f("screen_names_for_tweet_box", a, b, 0))), c.b(' "\n')
            }), a.pop()), d.s(d.f("condensed_tweet_box_text", a, b, 1), a, b, 0, 477, 543, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('        data-condensed-text="'), c.b(c.v(c.f("condensed_tweet_box_text", a, b, 0))), c.b('"\n')
            }), a.pop()), d.b('      action="//'), d.b(d.v(d.f("upload_domain", a, b, 0))), d.b('/i/tweet/create_with_media.iframe"\n      enctype="multipart/form-data">\n  <input type="hidden" name="post_authenticity_token" class="auth-token">\n  <input type="hidden" name="iframe_callback" class="iframe-callback">\n  <input type="hidden" name="in_reply_to_status_id" class="in-reply-to-status-id">\n  <input type="hidden" name="impression_id" class="impression-id">\n  <input type="hidden" name="earned" class="earned">\n  <input type="hidden" name="page_context" class="page-context">\n'), d.s(d.f("media_tagging_compose_enabled", a, b, 1), a, b, 0, 1127, 1195, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('  <input type="hidden" name="tagged_users" class="tagged-users">\n')
            }), a.pop()), d.b("\n" + c), d.s(d.f("current_user", a, b, 1), a, b, 0, 1250, 1353, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('    <img class="inline-reply-user-image avatar size32" src="'), c.b(c.v(c.f("profile_image_absolute_url", a, b, 0))), c.b('" alt="">\n')
            }), a.pop()), d.b('  <span class="inline-reply-caret">\n    <span class="caret-inner"></span>\n  </span>\n\n  <div class="tweet-content">\n    <span class="icon add-photo-icon hidden"></span>\n'), d.b('    <span class="icon nav-tweet hidden"></span>\n    <span class="tweet-drag-help tweet-drag-photo-here hidden">Drag photo here</span>\n    <span class="visuallyhidden" id="'), d.b(d.v(d.f("tweet_box_id", a, b, 0))), d.b('-label">Tweet text</span>\n\n'), d.b('    <div aria-labelledby="'), d.b(d.v(d.f("tweet_box_id", a, b, 0))), d.b('-label" id="'), d.b(d.v(d.f("tweet_box_id", a, b, 0))), d.b('" class="tweet-box rich-editor '), d.s(d.f("use_user_colors", a, b, 1), a, b, 0, 2049, 2148, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("u-textUserColor u-borderUserColorLight u-borderUserColorLightFocus u-boxShadowUserColorLighterFocus")
            }), a.pop()), d.b('" contenteditable="true" spellcheck="true" role="textbox"\n      aria-multiline="true">'), d.b(d.rp("<prefilled_tweetbox_content0", a, b, "")), d.b('</div>\n    <div class="rich-normalizer"></div>\n\n'), d.b(d.rp("<swift_typeahead_dropdown1", a, b, "    ")), d.b("\n" + c), d.b('    <textarea aria-hidden="true" class="tweet-box-shadow" name="status"></textarea>\n\n'), d.b('    <div class="thumbnail-container">\n      <div class="previews"></div>\n      <div class="preview template">\n'), d.b(d.rp("<dismiss_button2", a, b, "        ")), d.b('        <span class="filename"></span>\n      </div>\n      <div class="preview-message">\n'), d.s(d.f("media_tagging_compose_enabled", a, b, 1), a, b, 0, 2823, 3144, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('          <button type="button" class="start-tagging js-open-user-select no-users u-borderUserColorLight u-textUserColor" disabled>\n            <span class="Icon Icon--me Icon--smallest"></span>\n            <span class="tagged-users">\n              Who\'s in these photos?\n            </span>\n          </button>\n')
            }), a.pop()), d.b("      </div>\n    </div>\n"), d.s(d.f("media_tagging_compose_enabled", a, b, 1), a, b, 0, 3241, 4038, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('      <div class="photo-tagging-container user-select-container hidden">\n        <div class="tagging-dropdown dropdown-menu">\n          <div class="dropdown-caret center">\n            <div class="caret-outer"></div>\n            <div class="caret-inner"></div>\n          </div>\n          <div class="photo-tagging-controls user-select-controls">\n            <label class="t1-label">\n              <span class="Icon Icon--search nav-search"></span>\n              <span class="u-hiddenVisually">Users in this photo</span>\n              <input class="js-initial-focus" type="text" placeholder="Search and tag up to 10 people">\n            </label>\n          </div>\n          <div class="typeahead-container">\n'), c.b(c.rp("<swift_typeahead_dropdown3", a, b, "            ")), c.b("          </div>\n        </div>\n      </div>\n")
            }), a.pop()), d.b("  </div>\n\n"), d.s(d.f("tweetbox_skip_button", a, b, 1), a, b, 0, 4110, 4196, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('    <button type="button" class="btn tweet-skip-button u-floatRight">Skip</button>\n')
            }), a.pop()), d.b('  <div class="toolbar js-toolbar">\n    <div class="tweet-box-extras">\n\n'), d.s(d.f("include_image_picker", a, b, 1), a, b, 0, 4324, 4363, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.rp("<media/media_picker4", a, b, "        "))
            }), a.pop()), d.b("\n" + c), d.s(d.f("content_view", a, b, 1), a, b, 0, 4413, 4552, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.s(c.f("include_lifeline_alert_status", a, b, 1), a, b, 0, 4456, 4511, "{{ }}") && (c.rs(a, b, function(a, b, c) {
                    c.b(c.rp("<lifeline/lifeline_alert_status5", a, b, "          "))
                }), a.pop())
            }), a.pop()), d.b("\n" + c), d.s(d.f("include_geo_picker", a, b, 1), a, b, 0, 4600, 4635, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.rp("<geo/geo_picker6", a, b, "        "))
            }), a.pop()), d.b("\n" + c), d.b('    </div>\n    <div class="tweet-button">\n      <span class="spinner"></span>\n      <span class="tweet-counter">140</span>\n'), d.b(d.rp("<tweet_button7", a, b, "      ")), d.b("    </div>\n  </div>\n</form>\n"), d.fl()
        },
        partials: {
            "<prefilled_tweetbox_content0": {
                name: "prefilled_tweetbox_content",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_dropdown1": {
                name: "swift_typeahead_dropdown",
                partials: {},
                subs: {}
            },
            "<dismiss_button2": {
                name: "dismiss_button",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_dropdown3": {
                name: "swift_typeahead_dropdown",
                partials: {},
                subs: {}
            },
            "<media/media_picker4": {
                name: "media/media_picker",
                partials: {},
                subs: {}
            },
            "<lifeline/lifeline_alert_status5": {
                name: "lifeline/lifeline_alert_status",
                partials: {},
                subs: {}
            },
            "<geo/geo_picker6": {
                name: "geo/geo_picker",
                partials: {},
                subs: {}
            },
            "<tweet_button7": {
                name: "tweet_button",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b.dismiss_button = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<button type="button" class="dismiss js-dismiss" tabindex="-1">\n  <span class="icon dismiss-white">\n    <span class="visuallyhidden">\n      Dismiss\n    </span>\n  </span>\n</button>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b.swift_dm_box = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<form class="t1-form dm-tweetbox tweet-form"\n    method="post"\n    target="dm-post-iframe"\n    action="//'), d.b(d.v(d.f("media_upload_domain", a, b, 0))), d.b('/i/media/upload.iframe"\n    enctype="multipart/form-data">\n  <div class="tweet-content">\n    <input type="hidden" name="authenticity_token" class="auth-token">\n    <span class="tweet-drag-help tweet-drag-photo-here hidden">Drag photo here</span>\n    <span class="visuallyhidden" id="'), d.b(d.v(d.f("tweet_box_id", a, b, 0))), d.b('-label">Direct message text</span>\n    <div class="modal-tweet u-borderUserColorLighter"></div>\n    <div id="'), d.b(d.v(d.f("tweet_box_id", a, b, 0))), d.b('" aria-labelledby="'), d.b(d.v(d.f("tweet_box_id", a, b, 0))), d.b('-label" class="tweet-box rich-editor" contenteditable="true" spellcheck="true" role="textbox" aria-multiline="true"></div>\n    <div class="rich-normalizer"></div>\n\n'), d.b('    <div class="thumbnail-container">\n      <div class="previews"></div>\n      <div class="preview template">\n'), d.b(d.rp("<dismiss_button0", a, b, "        ")), d.b('        <span class="filename"></span>\n      </div>\n    </div>\n  </div>\n  <div class="toolbar js-toolbar">\n    <div class="tweet-box-extras">\n'), d.b(d.rp("<media/media_picker1", a, b, "      ")), d.b('    </div>\n    <div class="tweet-button">\n      <span class="spinner"></span>\n      <span class="tweet-counter">140</span>\n      <button class="btn tweet-action primary-btn messaging tweet-btn disabled" type="button">\n        <span class="button-text messaging-text">\n          <span class="Icon Icon--dm"></span>\n          Send message\n        </span>\n      </button>\n    </div>\n  </div>\n  <div class="dm-delete-confirm js-dm-delete-confirm">\n'), d.b(d.rp("<direct_message_delete_confirm2", a, b, "    ")), d.b('  </div>\n  <div class="dm-spam-confirm js-dm-spam-confirm">\n'), d.b(d.rp("<direct_message_spam_confirm3", a, b, "    ")), d.b('  </div>\n  <div class="dm-abuse-confirm js-dm-abuse-confirm">\n'), d.b(d.rp("<direct_message_abuse_confirm4", a, b, "    ")), d.b("  </div>\n</form>\n"), d.fl()
        },
        partials: {
            "<dismiss_button0": {
                name: "dismiss_button",
                partials: {},
                subs: {}
            },
            "<media/media_picker1": {
                name: "media/media_picker",
                partials: {},
                subs: {}
            },
            "<direct_message_delete_confirm2": {
                name: "direct_message_delete_confirm",
                partials: {},
                subs: {}
            },
            "<direct_message_spam_confirm3": {
                name: "direct_message_spam_confirm",
                partials: {},
                subs: {}
            },
            "<direct_message_abuse_confirm4": {
                name: "direct_message_abuse_confirm",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b.tweet_button = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<button class="btn primary-btn tweet-action'), d.s(d.f("enable_tweet_button", a, b, 1), a, b, 1, 0, 0, "") || d.b(" disabled"), d.b(' tweet-btn js-tweet-btn" type="button" '), d.s(d.f("enable_tweet_button", a, b, 1), a, b, 1, 0, 0, "") || d.b("disabled"), d.b('>\n  <span class="button-text tweeting-text">\n    <span class="Icon Icon--tweet"></span>\n    '), d.s(d.f("second_session_profile", a, b, 1), a, b, 0, 314, 352, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("Tweet & continue")
            }), a.pop()), d.s(d.f("second_session_profile", a, b, 1), a, b, 1, 0, 0, "") || (d.s(d.f("first_tweet_button_text", a, b, 1), a, b, 0, 456, 509, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.v(c.f("first_tweet_button_text", a, b, 0)))
            }), a.pop()), d.s(d.f("first_tweet_button_text", a, b, 1), a, b, 1, 0, 0, "") || d.b("Tweet")), d.b("\n" + c), d.b('  </span>\n  <span class="button-text messaging-text">\n    <span class="Icon Icon--dm Icon--small"></span>\n    Send message\n  </span>\n</button>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b.verified_account_badge = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.s(d.f("large_verified_badge_with_border", a, b, 1), a, b, 0, 37, 331, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('<span class="icon verified verified-large-border"><span class="visuallyhidden">Verified account</span></span>')
            }), a.pop()), d.s(d.f("large_verified_badge_with_border", a, b, 1), a, b, 1, 0, 0, "") || (d.b('<span class="Icon Icon--verified Icon--small">'), d.b('<span class="u-hiddenVisually">Verified account</span>'), d.b("</span>")), d.b("\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b.user_css_color = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<style id="user-style-'), d.b(d.v(d.f("style_namespace", a, b, 0))), d.b('">\n'), d.b("\n" + c), d.s(d.f("user_color", a, b, 1), a, b, 0, 320, 941, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  a,\n  a:hover,\n  a:focus,\n  a:active {\n    color: #"), c.b(c.v(c.f("user_color", a, b, 0))), c.b(";\n  }\n\n  .u-textUserColor,\n  .u-textUserColorHover:hover,\n  .u-textUserColorHover:focus {\n    color: #"), c.b(c.v(c.f("user_color", a, b, 0))), c.b(" !important;\n  }\n\n  .u-borderUserColor,\n  .u-borderUserColorHover:hover,\n  .u-borderUserColorHover:focus {\n    border-color: #"), c.b(c.v(c.f("user_color", a, b, 0))), c.b(" !important;\n  }\n\n  .u-bgUserColor,\n  .u-bgUserColorHover:hover,\n  .u-bgUserColorHover:focus {\n    background-color: #"), c.b(c.v(c.f("user_color", a, b, 0))), c.b(" !important;\n  }\n\n  .u-boxShadowInsetUserColorHover:hover,\n  .u-boxShadowInsetUserColorHover:focus {\n    box-shadow: inset 0 0 0 5px #"), c.b(c.v(c.f("user_color", a, b, 0))), c.b(" !important;\n  }\n\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("user_color_light", a, b, 1), a, b, 0, 1031, 1391, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  .u-textUserColorLight {\n    color: #"), c.b(c.v(c.f("user_color_light", a, b, 0))), c.b(" !important;\n  }\n\n  .u-borderUserColorLight,\n  .u-borderUserColorLightFocus:focus,\n  .u-borderUserColorLightHover:hover,\n  .u-borderUserColorLightHover:focus {\n    border-color: #"), c.b(c.v(c.f("user_color_light", a, b, 0))), c.b(" !important;\n  }\n\n  .u-bgUserColorLight {\n    background-color: #"), c.b(c.v(c.f("user_color_light", a, b, 0))), c.b(" !important;\n  }\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("rgba_user_color_lighter", a, b, 1), a, b, 0, 1501, 1651, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  .u-boxShadowUserColorLighterFocus:focus {\n    box-shadow: 0 0 8px rgba(0, 0, 0, 0.05), inset 0 1px 2px "), c.b(c.v(c.f("rgba_user_color_lighter", a, b, 0))), c.b(" !important;\n  }\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("user_color_lightest", a, b, 1), a, b, 0, 1760, 2024, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  .u-textUserColorLightest {\n    color: #"), c.b(c.v(c.f("user_color_lightest", a, b, 0))), c.b(" !important;\n  }\n\n  .u-borderUserColorLightest {\n    border-color: #"), c.b(c.v(c.f("user_color_lightest", a, b, 0))), c.b(" !important;\n  }\n\n  .u-bgUserColorLightest {\n    background-color: #"), c.b(c.v(c.f("user_color_lightest", a, b, 0))), c.b(" !important;\n  }\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("user_color_lighter", a, b, 1), a, b, 0, 2127, 2385, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  .u-textUserColorLighter {\n    color: #"), c.b(c.v(c.f("user_color_lighter", a, b, 0))), c.b(" !important;\n  }\n\n  .u-borderUserColorLighter {\n    border-color: #"), c.b(c.v(c.f("user_color_lighter", a, b, 0))), c.b(" !important;\n  }\n\n  .u-bgUserColorLighter {\n    background-color: #"), c.b(c.v(c.f("user_color_lighter", a, b, 0))), c.b(" !important;\n  }\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("user_color_dark", a, b, 1), a, b, 0, 2481, 2658, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  .u-bgUserColorDarkHover:hover {\n    background-color: #"), c.b(c.v(c.f("user_color_dark", a, b, 0))), c.b(" !important;\n  }\n\n  .u-borderUserColorDark {\n    border-color: #"), c.b(c.v(c.f("user_color_dark", a, b, 0))), c.b(" !important;\n  }\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("user_color_darker", a, b, 1), a, b, 0, 2755, 2855, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  .u-bgUserColorDarkerActive:active {\n    background-color: #"), c.b(c.v(c.f("user_color_darker", a, b, 0))), c.b(" !important;\n  }\n")
            }), a.pop()), d.b("\n"), d.b("\n" + c), d.b("\n"), d.b("\n" + c), d.b("\n" + c), d.b("a,\n.btn-link,\n.btn-link:focus,\n.icon-btn,\n"), d.b(".pretty-link b,\n.pretty-link:hover s,\n.pretty-link:hover b,\n.pretty-link:focus s,\n.pretty-link:focus b,\n/* Account Group */\n.metadata a:hover,\n.metadata a:focus,\n"), d.b(".account-group:hover .fullname,\n.account-group:focus .fullname,\n.account-summary:focus .fullname,\n"), d.b(".message .message-text a,\n.stats a strong,\n.plain-btn:hover,\n.plain-btn:focus,\n.dropdown.open .user-dropdown.plain-btn,\n.open > .plain-btn,\n#global-actions .new:before,\n.module .list-link:hover,\n.module .list-link:focus,\n"), d.b(".UserCompletion-step:hover,\n"), d.b(".stats a:hover,\n.stats a:hover strong,\n.stats a:focus,\n.stats a:focus strong,\n"), d.b(".profile-modal-header .fullname a:hover,\n.profile-modal-header .username a:hover,\n.profile-modal-header .fullname a:focus,\n.profile-modal-header .username a:focus,\n"), d.b(".story-article:hover .metadata,\n.story-article .metadata a:focus,\n"), d.b(".find-friends-sources li:hover .source,\n\n"), d.b("\n" + c), d.b(".stream-item a:hover .fullname,\n.stream-item a:focus .fullname,\n"), d.b(".stream-item .view-all-supplements:hover,\n.stream-item .view-all-supplements:focus,\n"), d.b(".tweet .time a:hover,\n.tweet .time a:focus,\n.tweet .details.with-icn b,\n.tweet .details.with-icn .Icon,\n.tweet .tweet-geo-text a:hover,\n"), d.b(".stream-item:hover .original-tweet .expand-action-wrapper,\n.stream-item .original-tweet.focus .expand-action-wrapper,\n.opened-tweet.original-tweet  .expand-action-wrapper,\n"), d.b(".stream-item:hover .original-tweet .details b,\n.stream-item .original-tweet.focus .details b,\n.stream-item.open .original-tweet .details b,\n"), d.b(".simple-tweet:hover .details b,\n.simple-tweet.focus .details b,\n.simple-tweet.open .details b,\n.simple-tweet:hover .details .expand-action-wrapper,\n.simple-tweet.focus .details .expand-action-wrapper,\n.simple-tweet.open .details .collapse-action-wrapper,\n.simple-tweet:hover .details .simple-details-link,\n.simple-tweet.focus .details .simple-details-link,\n"), d.b(".client-and-actions a:hover,\n.client-and-actions a:focus,\n"), d.b(".dismiss-promoted:hover b,\n"), d.b(".tweet .context .pretty-link:hover s,\n.tweet .context .pretty-link:hover b,\n.tweet .context .pretty-link:focus s,\n.tweet .context .pretty-link:focus b,\n"), d.b(".list .username a:hover,\n.list .username a:focus,\n.list-membership-container .create-a-list,\n.list-membership-container .create-a-list:hover,\n\n"), d.b(".story-header:hover .view-tweets,\n.card .list-details a:hover,\n.card .list-details a:focus,\n.card .card-body:hover .attribution,\n.card .card-body .attribution:focus,\n.events-card .card-body:hover .attribution,\n.events-card .card-body .attribution:focus,\n.new-tweets-bar,\n.onebox .soccer ul.ticker a:hover,\n.onebox .soccer ul.ticker a:focus,\n\n"), d.b(".discover-item-actions a,\n\n"), d.b(".remove-background-btn,\n\n"), d.b(".stream-item-activity-me .latest-tweet .tweet-row a:hover,\n.stream-item-activity-me .latest-tweet .tweet-row a:focus,\n.stream-item-activity-me .latest-tweet .tweet-row a:hover b,\n.stream-item-activity-me .latest-tweet .tweet-row a:focus b {\n"), d.s(d.f("profile_link_color", a, b, 1), a, b, 0, 7286, 7325, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    color: #"), c.b(c.v(c.f("profile_link_color", a, b, 0))), c.b(";\n")
            }), a.pop()), d.s(d.f("profile_link_color", a, b, 1), a, b, 1, 0, 0, "") || d.b("    color: #0084B4;\n"), d.b("}\n\n"), d.s(d.f("profile_link_color", a, b, 1), a, b, 0, 7537, 11947, "{{ }}") && (d.rs(a, b, function(a, b, d) {
                d.b("  #global-actions > li > a {\n    border-bottom-color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n  #global-actions > li:hover > a,\n  #global-actions > li > a:focus,\n  .nav.right-actions > li > a:hover,\n  .nav.right-actions > li > button:hover,\n  .nav.right-actions > li > a:focus,\n  .nav.right-actions > li > button:focus {\n    color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n"), d.s(d.f("web_notifications_badge_on", a, b, 1), a, b, 0, 7923, 8047, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                    c.b("  /* Surpress the new connect glow if in experiment. */\n   #global-actions .people.new:before {\n     content: none;\n   }\n")
                }), a.pop()), d.b("\n" + c), d.b("  /* hover state for photo select button*/\n  .photo-selector:not(.disabled):hover .btn,\n  .icon-btn:hover,\n  .icon-btn:active,\n  .icon-btn.active,\n  .icon-btn.enabled {\n    border-color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n    border-color: rgba("), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(","), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(","), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(",.5);\n    color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n  /* hover state for photo select button*/\n  .photo-selector:not(.disabled):hover .btn,\n  .icon-btn:hover {\n    background-image: linear-gradient(rgba(255,255,255,0), rgba("), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(","), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(","), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(",.1));\n    filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00FFFFFF', endColorstr='#19"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b("'); /* IE8-9 */\n  }\n\n  .icon-btn.disabled,\n  .icon-btn.disabled:hover,\n  .icon-btn[disabled],\n  .icon-btn[aria-disabled=true] {\n    color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n  /* tweet-btn can have primary-btn class as well so the following rules ensure that the correct background color is applied */\n  .tweet-btn,\n  .tweet-btn:focus {\n    background-color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n    background: rgba("), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(","), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(","), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(",.8);\n  }\n\n  .tweet-btn:hover,\n  .tweet-btn:active,\n  .tweet-btn.active {\n    background-color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n  .tweet-btn.btn.disabled,\n  .tweet-btn.btn.disabled:hover,\n  .tweet-btn.btn[disabled],\n  .tweet-btn.btn[aria-disabled=true] {\n    background: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n  .btn:focus,\n  .btn.focus,\n  .Button:focus {\n    box-shadow:\n      0 0 0 1px #fff,\n      0 0 0 3px rgba("), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(", 0.5);\n  }\n\n  .selected-stream-item:focus {\n    box-shadow: 0 0 0 3px rgba("), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(', 0.5);\n  }\n\n  /* highlighting when navigate through timeline stream table view with j & k. */\n  .js-navigable-stream.stream-table-view .selected-stream-item[tabindex="-1"]:focus {\n    outline: 3px solid rgba('), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(", 0.5) !important;\n  }\n\n  /* box-shadow does not work with table tr element */\n  .js-navigable-stream.stream-table-view .selected-stream-item:focus {\n    box-shadow: none;\n  }\n\n  /**\n   * 1. Bring actionable tweet to top when active to ensure border\n   *    highlighting wraps entire tweet. Value must be at least at if not\n   *    higher than the z-index value of ProfileHeading to ensure first\n   *    tweet in timeline receives border on all four sides.\n   *    NOTE: z-index should be 2 to avoid conflicts with .ProfileCanopy and .ProfileCard-avatarLink\n   */\n\n  .ProfileTweet.is-actionable,\n  .js-stream-item.is-selected:focus .ProfileTweet,\n  .js-stream-item.is-selected:focus .ProfileCard,\n  .QuoteTweet:hover,\n  .QuoteTweet:focus,\n  .QuoteTweet:active {\n    border-color: rgba("), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(", 0.5);\n    z-index: 2; /* 1 */\n  }\n\n  .global-dm-nav.new.with-count .dm-new {\n    background: #fff;\n  }\n\n  .global-dm-nav.new.with-count .dm-new .count-inner {\n    background: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n  .global-nav .people .count .count-inner {\n    background: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n\n  .dropdown-menu li > a:hover,\n  .dropdown-menu li > a:focus,\n  .dropdown-menu .dropdown-link:hover,\n  .dropdown-menu .dropdown-link:focus,\n  .dropdown-menu .dropdown-link.is-focused,\n  .dropdown-menu li:hover .dropdown-link,\n  .dropdown-menu li:focus .dropdown-link,\n  .dropdown-menu .typeahead-recent-search-item.selected,\n  .dropdown-menu .typeahead-saved-search-item.selected,\n  .dropdown-menu .selected a,\n  .dropdown-menu .dropdown-link.selected {\n    background-color: #"), d.b(d.v(d.f("profile_link_color", a, b, 0))), d.b(";\n  }\n")
            }), a.pop()), d.b("\n" + c), d.b("/* give tweet boxes 10% of the users link color as background */\n.home-tweet-box,\n.dm-tweetbox,\n.WebToast-box--altColor,\n.content-main .conversations-enabled .expansion-container .inline-reply-tweetbox {\n  background-color: #"), d.b(d.v(d.f("user_color_lightest", a, b, 0))), d.b(";\n}\n\n.conversations-enabled .inline-reply-caret .caret-inner {\n  border-bottom-color: #"), d.b(d.v(d.f("user_color_lightest", a, b, 0))), d.b(";\n}\n.top-timeline-tweetbox .timeline-tweet-box .tweet-form.condensed .tweet-box {\n  color: #"), d.b(d.v(d.f("user_color", a, b, 0))), d.b(";\n}\n/* give tweet box containers an outline using the users link color */\n.home-tweet-box,\n.top-timeline-tweetbox .timeline-tweet-box {\n  border-color: #"), d.b(d.v(d.f("user_color_lighter", a, b, 0))), d.b(' !important;\n}\n/* give tweet boxes an outline using the users link color */\n.tweet-box[contenteditable="true"] {\n  border-color: '), d.b(d.v(d.f("rgba_user_color_lighter", a, b, 0))), d.b(';\n}\n\ninput:focus,\ntextarea:focus,\ndiv[contenteditable="true"]:focus,\ndiv[contenteditable="true"].fake-focus {\n  border-color: #'), d.b(d.v(d.f("profile_pretty_link_color", a, b, 0))), d.b(";\n  box-shadow: inset 0 0 0 1px rgba("), d.b(d.v(d.d("profile_link_color_rgb.r", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.g", a, b, 0))), d.b(", "), d.b(d.v(d.d("profile_link_color_rgb.b", a, b, 0))), d.b(', 0.7);\n}\n\n.currently-dragging.modal-enabled .modal .tweet-box,\n.currently-dragging:not(.modal-enabled) .tweet-content .tweet-box,\n.tweet-box[contenteditable="true"]:focus {\n  border-color: #'), d.b(d.v(d.f("user_color_light", a, b, 0))), d.b(";\n  box-shadow: none;\n}\n\n"), d.b("\n" + c), d.b("s,\n.pretty-link:hover s,\n.pretty-link:focus s,\n.stream-item-activity-me .latest-tweet .tweet-row a:hover s,\n.stream-item-activity-me .latest-tweet .tweet-row a:focus s {\n"), d.s(d.f("profile_pretty_link_color", a, b, 1), a, b, 0, 13575, 13621, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    color: #"), c.b(c.v(c.f("profile_pretty_link_color", a, b, 0))), c.b(";\n")
            }), a.pop()), d.s(d.f("profile_pretty_link_color", a, b, 1), a, b, 1, 0, 0, "") || d.b("    color: #66B5D2;\n"), d.b("}\n\n"), d.b(".vellip,\n.vellip:before,\n.vellip:after,\n.conversation-module > li:after,\n.conversation-module > li:before {\n"), d.s(d.f("profile_pretty_link_color", a, b, 1), a, b, 0, 13969, 14026, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    background-color: #"), c.b(c.v(c.f("profile_pretty_link_color", a, b, 0))), c.b(";\n")
            }), a.pop()), d.s(d.f("profile_pretty_link_color", a, b, 1), a, b, 1, 0, 0, "") || d.b("    background-color: #66B5D2;\n"), d.b("}\n\n"), d.b(".tweet .sm-reply,\n.tweet .sm-rt,\n.tweet .sm-fav,\n.tweet .sm-image,\n.tweet .sm-video,\n.tweet .sm-audio,\n.tweet .sm-geo,\n.tweet .sm-in,\n.tweet .sm-trash,\n.tweet .sm-more,\n.tweet .sm-page,\n.tweet .sm-embed,\n.tweet .sm-summary,\n.tweet .sm-chat,\n"), d.b(".timelines-navigation .active .profile-nav-icon,\n.timelines-navigation .profile-nav-icon:hover,\n.timelines-navigation .profile-nav-link:focus .profile-nav-icon,\n"), d.b(".sm-top-tweet,\n"), d.b(".metadata a.tweet-geo-text:hover .sm-geo,\n"), d.b(".discover-item .js-action-card-search:hover .sm-search,\n.discover-item .js-action-card-search:focus .sm-search {\n"), d.s(d.f("profile_link_color", a, b, 1), a, b, 0, 15076, 15126, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    background-color: #"), c.b(c.v(c.f("profile_link_color", a, b, 0))), c.b(";\n")
            }), a.pop()), d.s(d.f("profile_link_color", a, b, 1), a, b, 1, 0, 0, "") || d.b("    background-color: #6684B4;\n"), d.b("}\n\n"), d.s(d.f("mini_profile_banner_url_for_web", a, b, 1), a, b, 0, 15272, 15393, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(".enhanced-mini-profile .mini-profile .profile-summary {\n  background-image: url("), c.b(c.v(c.f("mini_profile_banner_url_for_web", a, b, 0))), c.b(");\n}\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("profile_banner_url_for_web", a, b, 1), a, b, 0, 15462, 15592, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(".wrapper-profile .profile-card.profile-header .profile-header-inner {\n  background-image: url("), c.b(c.v(c.f("profile_banner_url_for_web", a, b, 0))), c.b(");\n}\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("profile_link_color_rgb", a, b, 1), a, b, 0, 15652, 16047, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  #global-tweet-dialog .modal-header {\n    border-bottom: solid 1px rgba("), c.b(c.v(c.d("profile_link_color_rgb.r", a, b, 0))), c.b(", "), c.b(c.v(c.d("profile_link_color_rgb.g", a, b, 0))), c.b(", "), c.b(c.v(c.d("profile_link_color_rgb.b", a, b, 0))), c.b(", .25);\n  }\n\n  #global-tweet-dialog .modal-tweet-form-container {\n    background-color: #"), c.b(c.v(c.f("profile_link_color", a, b, 0))), c.b(";\n    background: rgba("), c.b(c.v(c.d("profile_link_color_rgb.r", a, b, 0))), c.b(", "), c.b(c.v(c.d("profile_link_color_rgb.g", a, b, 0))), c.b(", "), c.b(c.v(c.d("profile_link_color_rgb.b", a, b, 0))), c.b(", .1);\n  }\n")
            }), a.pop()), d.b("\n" + c), d.s(d.f("user_color_lightest", a, b, 1), a, b, 0, 16100, 16180, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  .inline-reply-tweetbox {\n    background-color: #"), c.b(c.v(c.f("user_color_lightest", a, b, 0))), c.b(";\n  }\n")
            }), a.pop()), d.b("\n" + c), d.b("</style>\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b.screen_reader_kb_shortcuts_message = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div id="kb-shortcuts-msg" class="visuallyhidden">\n  <h2>Keyboard Shortcuts</h2>\n  <p>\n    Keyboard shortcuts are available for common actions and site navigation.\n'), d.b('    <button id="show-shortcuts-btn" type="button" tabindex="-1">View Keyboard Shortcuts</button>\n    <button id="dismiss-shortcuts-btn" type="button" tabindex="-1">Dismiss this message</button>\n  </p>\n</div>'), d.fl()
        },
        partials: {},
        subs: {}
    }), b.swift_typeahead_dropdown = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b("\n" + c), d.b('<div role="listbox" class="dropdown-menu typeahead">\n  <div aria-hidden="true" class="dropdown-caret">\n    <div class="caret-outer"></div>\n    <div class="caret-inner"></div>\n  </div>\n  <div role="presentation" class="dropdown-inner js-typeahead-results">\n'), d.s(d.f("typeahead_recent_searches", a, b, 1), a, b, 0, 369, 407, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.rp("<swift_recent_searches0", a, b, "      "))
            }), a.pop()), d.b(d.rp("<swift_saved_searches1", a, b, "    ")), d.s(d.f("typeahead_concierge", a, b, 1), a, b, 0, 496, 538, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.rp("<swift_typeahead_concierge2", a, b, "      "))
            }), a.pop()), d.b(d.rp("<swift_typeahead_topics3", a, b, "    ")), d.b(d.rp("<swift_typeahead_accounts4", a, b, "    ")), d.s(d.f("typeahead_contacts", a, b, 1), a, b, 0, 656, 697, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.rp("<swift_typeahead_contacts5", a, b, "      "))
            }), a.pop()), d.b(d.rp("<swift_typeahead_trend_locations6", a, b, "    ")), d.b(d.rp("<swift_typeahead_select_users7", a, b, "    ")), d.b("  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<swift_recent_searches0": {
                name: "swift_recent_searches",
                partials: {},
                subs: {}
            },
            "<swift_saved_searches1": {
                name: "swift_saved_searches",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_concierge2": {
                name: "swift_typeahead_concierge",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_topics3": {
                name: "swift_typeahead_topics",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_accounts4": {
                name: "swift_typeahead_accounts",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_contacts5": {
                name: "swift_typeahead_contacts",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_trend_locations6": {
                name: "swift_typeahead_trend_locations",
                partials: {},
                subs: {}
            },
            "<swift_typeahead_select_users7": {
                name: "swift_typeahead_select_users",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b.prefilled_tweetbox_content = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.s(d.f("screen_names_for_tweet_box", a, b, 1), a, b, 0, 31, 197, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.s(c.f("condensed_tweet_box", a, b, 1), a, b, 1, 0, 0, "") || (c.b(c.v(c.f("screen_names_for_tweet_box", a, b, 0))), c.b(" ")), c.s(c.f("condensed_tweet_box", a, b, 1), a, b, 0, 134, 173, "{{ }}") && (c.rs(a, b, function(a, b, c) {
                    c.b("Reply to "), c.b(c.v(c.f("screen_names_for_tweet_box", a, b, 0)))
                }), a.pop())
            }), a.pop()), d.s(d.f("second_session_profile_prefilled_text", a, b, 1), a, b, 0, 270, 311, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.v(c.f("second_session_profile_prefilled_text", a, b, 0)))
            }), a.pop()), d.s(d.f("second_session_profile_prefilled_text", a, b, 1), a, b, 1, 0, 0, "") || d.s(d.f("first_tweet_prefilled_text", a, b, 1), a, b, 0, 426, 456, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(c.v(c.f("first_tweet_prefilled_text", a, b, 0)))
            }), a.pop()), d.fl()
        },
        partials: {},
        subs: {}
    }), b.swift_typeahead_accounts = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b("\n" + c), d.b('<ul role="presentation" class="typeahead-items typeahead-accounts js-typeahead-accounts">\n'), d.b('  <li role="presentation" data-user-id="" data-user-screenname="" data-remote="true" data-score="" class="typeahead-item typeahead-account-item js-selectable">\n'), d.b('    <a role="option" class="js-nav" data-query-source="typeahead_click" data-search-query="" data-ds="account">\n      <img class="avatar size24" alt="">\n      <span class="typeahead-user-item-info">\n        <span class="fullname"></span>\n        <span class="js-verified hidden">'), d.b(d.rp("<verified_account_badge0", a, b, "")), d.b('</span>\n        <span class="username"><s>@</s><b></b></span>\n      </span>\n    </a>\n  </li>\n  <li role="presentation" class="js-selectable typeahead-accounts-shortcut js-shortcut"><a role="option" class="js-nav" href="" data-search-query="" data-query-source="typeahead_click" data-shortcut="true" data-ds="account_search"></a></li>\n</ul>\n'), d.fl()
        },
        partials: {
            "<verified_account_badge0": {
                name: "verified_account_badge",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b.swift_typeahead_topics = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<ul role="presentation" class="typeahead-items typeahead-topics'), d.s(d.f("typeahead_topic_social_context", a, b, 1), a, b, 0, 98, 113, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(" social-context")
            }), a.pop()), d.b('">\n'), d.b('  <li role="presentation" class="typeahead-item typeahead-topic-item">\n    <a role="option" class="js-nav" href="" data-search-query="" data-query-source="typeahead_click" data-ds="topics" tabindex="-1">\n'), d.s(d.f("typeahead_topic_social_context", a, b, 1), a, b, 0, 490, 550, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('        <div class="typeahead-social-context"></div>\n')
            }), a.pop()), d.b("    </a>\n  </li>\n</ul>\n\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b.create_custom_timeline_button = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<button type="button" class="btn-link js-create-custom-timeline-button" data-element-term="create_custom_timeline_button">Create new collection</button>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["start/wtf_follow_button"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.s(d.f("num_users_selected", a, b, 1), a, b, 0, 23, 71, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  Follow "), c.b(c.v(c.f("num_users_selected", a, b, 0))), c.b(" &amp; continue\n")
            }), a.pop()), d.s(d.f("num_users_selected", a, b, 1), a, b, 1, 0, 0, "") || d.b("  Continue\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["start/follow_context"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<li class="StartFollowContext-item '), d.s(d.f("user", a, b, 1), a, b, 0, 44, 53, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("is-active")
            }), a.pop()), d.b('">\n'), d.s(d.f("user", a, b, 1), a, b, 0, 76, 157, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('    <img src="'), c.b(c.v(c.f("profile_url", a, b, 0))), c.b('" title="'), c.b(c.v(c.f("screen_name", a, b, 0))), c.b('" alt="'), c.b(c.v(c.f("screen_name", a, b, 0))), c.b('">\n')
            }), a.pop()), d.b("</li>"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["highline/user_location"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.s(d.f("place_id", a, b, 1), a, b, 0, 13, 103, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('  <a href="/search?q=place%3A'), c.b(c.v(c.f("place_id", a, b, 0))), c.b('" data-place-id="'), c.b(c.v(c.f("place_id", a, b, 0))), c.b('">'), c.b(c.v(c.f("location", a, b, 0))), c.b("</a>\n")
            }), a.pop()), d.s(d.f("place_id", a, b, 1), a, b, 1, 0, 0, "") || (d.b("  "), d.b(d.v(d.f("location", a, b, 0))), d.b("\n" + c)), d.fl()
        },
        partials: {},
        subs: {}
    }), b["spoonbill/retweet"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast is-actionable" data-redirect-to="notifications">\n  <div class="WebToast-box u-cf">\n'), d.b(d.rp("<spoonbill/notification_box_close0", a, b, "    ")), d.b("\n" + c), d.b('    <div class="WebToast-header u-cf">\n      <div class="WebToast-imageBox u-floatLeft">\n        <span class="Icon Icon--retweeted Icon--small u-floatRight"></span>\n      </div>\n      <div class="WebToast-contentBox">\n        Retweeted\n      </div>\n    </div>\n\n    <div class="u-cf">\n      <div class="WebToast-imageBox u-floatLeft">\n        <a class="js-user-profile-link js-action-profile-avatar" tabindex="-1" href="/'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('" data-user-id="'), d.b(d.v(d.f("user_id", a, b, 0))), d.b('">\n          <img class="WebToast-avatar" width="32" height="32" src="'), d.b(d.v(d.f("user_avatar", a, b, 0))), d.b('" alt="">\n        </a>\n      </div>\n      <div class="WebToast-contentBox">\n'), d.b(d.rp("<spoonbill/user_identity_fullname_link1", a, b, "        ")), d.b("        retweeted\n"), d.b(d.rp("<spoonbill/tweet_excerpt2", a, b, "        ")), d.b("      </div>\n    </div>\n  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<spoonbill/notification_box_close0": {
                name: "spoonbill/notification_box_close",
                partials: {},
                subs: {}
            },
            "<spoonbill/user_identity_fullname_link1": {
                name: "spoonbill/user_identity_fullname_link",
                partials: {},
                subs: {}
            },
            "<spoonbill/tweet_excerpt2": {
                name: "spoonbill/tweet_excerpt",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["spoonbill/follow_btn"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="user-actions '), d.b(d.v(d.f("follow_state", a, b, 0))), d.b('" data-user-id="'), d.b(d.v(d.f("user_id", a, b, 0))), d.b('" data-screen-name="'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('" data-name="'), d.b(d.v(d.f("user_name", a, b, 0))), d.b('" data-protected="'), d.b(d.v(d.f("user_protected", a, b, 0))), d.b('" data-item-type="user">\n  <button class="Button WebToast-followButton">\n    <span class="follow-text button-text">\n      <span class="Icon Icon--follow Icon--medium"></span>\n      <span class="u-hiddenVisually">Follow @'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('</span>\n    </span>\n    <span class="following-text button-text">\n      <span class="Icon Icon--following Icon--medium"></span>\n      <span class="u-hiddenVisually">Following @'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('</span>\n    </span>\n    <span class="unfollow-text button-text">\n      <span class="Icon Icon--unfollow Icon--medium"></span>\n      <span class="u-hiddenVisually">Unfollow @'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('</span>\n    </span>\n    <span class="pending-text button-text">\n      Pending\n      <span class="u-hiddenVisually">Pending @'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('</span>\n    </span>\n    <span class="blocked-text button-text">\n      Blocked\n      <span class="u-hiddenVisually">Blocked @'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b("</span>\n    </span>\n  </button>\n</div>\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["spoonbill/notification_box_close"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast-close">\n  <button class="WebToast-closeButton" type="button">\n    <span class="Icon Icon--close"></span>\n    <span class="u-hiddenVisually">Close</span>\n  </button>\n</div>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["spoonbill/tweet_excerpt"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast-tweetExcerpt u-dir u-textBreak" dir="ltr">\n  '), d.b(d.t(d.f("tweet_html", a, b, 0))), d.b("\n" + c), d.b("</div>\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["spoonbill/tweet"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="u-cf u-textBreak">\n'), d.b(d.rp("<spoonbill/user_identity0", a, b, "  ")), d.b("\n" + c), d.b('  <div class="WebToast-contentBox">\n    <div class="WebToast-tweetExcerpt tweet-text u-dir" dir="ltr">'), d.b(d.t(d.f("tweet_html", a, b, 0))), d.b("</div>\n  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<spoonbill/user_identity0": {
                name: "spoonbill/user_identity",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["spoonbill/follow"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast">\n  <div class="WebToast-box u-cf">\n'), d.b(d.rp("<spoonbill/notification_box_close0", a, b, "    ")), d.b("\n" + c), d.b('    <div class="WebToast-header u-cf">\n      <div class="WebToast-imageBox u-floatLeft">\n        <span class="Icon Icon--follower Icon--small u-floatRight"></span>\n      </div>\n      <div class="WebToast-contentBox">\n        New follower\n      </div>\n    </div>\n\n    <div class="u-cf">\n'), d.s(d.f("show_follow_button", a, b, 1), a, b, 0, 415, 488, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('        <div class="u-floatRight">'), c.b(c.rp("<spoonbill/follow_btn1", a, b, "")), c.b("</div>\n")
            }), a.pop()), d.b(d.rp("<spoonbill/user_identity2", a, b, "      ")), d.b("    </div>\n  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<spoonbill/notification_box_close0": {
                name: "spoonbill/notification_box_close",
                partials: {},
                subs: {}
            },
            "<spoonbill/follow_btn1": {
                name: "spoonbill/follow_btn",
                partials: {},
                subs: {}
            },
            "<spoonbill/user_identity2": {
                name: "spoonbill/user_identity",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["spoonbill/user_identity"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="'), d.s(d.f("with_block_username", a, b, 1), a, b, 1, 0, 0, "") || d.b("u-inline"), d.b('">\n  <a class="WebToast-accountLink u-linkComplex js-action-profile-name js-user-profile-link" href="/'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('" data-user-id="'), d.b(d.v(d.f("user_id", a, b, 0))), d.b('">\n    <span class="WebToast-imageBox u-floatLeft">\n      <img class="WebToast-avatar" width="32" height="32" src="'), d.b(d.v(d.f("user_avatar", a, b, 0))), d.b('" alt="">\n    </span>\n\n    <span class="WebToast-contentBox u-block">\n      <b class="WebToast-fullname u-linkComplex-target '), d.s(d.f("with_block_username", a, b, 1), a, b, 0, 496, 510, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("u-block u-nbfc")
            }), a.pop()), d.b('">'), d.b(d.v(d.f("user_name", a, b, 0))), d.b("</b>\n      "), d.s(d.f("with_block_username", a, b, 1), a, b, 1, 0, 0, "") || d.b("<span>&rlm;</span>"), d.b("\n" + c), d.b('      <span class="WebToast-username u-dir" dir="ltr">\n        <span class="at">@</span>'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b("\n" + c), d.b("      </span>\n    </span>\n  </a>\n</div>\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["spoonbill/favorite"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast is-actionable" data-redirect-to="notifications">\n  <div class="WebToast-box u-cf">\n'), d.b(d.rp("<spoonbill/notification_box_close0", a, b, "    ")), d.b("\n" + c), d.b('    <div class="WebToast-header u-cf">\n      <div class="WebToast-imageBox u-floatLeft">\n        <span class="Icon Icon--favorited Icon--small u-floatRight"></span>\n      </div>\n      <div class="WebToast-contentBox">\n        Favorited\n      </div>\n    </div>\n\n    <div class="u-cf">\n      <div class="WebToast-imageBox u-floatLeft">\n        <a class="js-user-profile-link js-action-profile-avatar" tabindex="-1" href="/'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('" data-user-id="'), d.b(d.v(d.f("user_id", a, b, 0))), d.b('">\n          <img class="WebToast-avatar" width="32" height="32" src="'), d.b(d.v(d.f("user_avatar", a, b, 0))), d.b('" alt="">\n        </a>\n      </div>\n      <div class="WebToast-contentBox">\n'), d.b(d.rp("<spoonbill/user_identity_fullname_link1", a, b, "        ")), d.b("        favorited\n"), d.b(d.rp("<spoonbill/tweet_excerpt2", a, b, "        ")), d.b("      </div>\n    </div>\n  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<spoonbill/notification_box_close0": {
                name: "spoonbill/notification_box_close",
                partials: {},
                subs: {}
            },
            "<spoonbill/user_identity_fullname_link1": {
                name: "spoonbill/user_identity_fullname_link",
                partials: {},
                subs: {}
            },
            "<spoonbill/tweet_excerpt2": {
                name: "spoonbill/tweet_excerpt",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["spoonbill/dm"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast dm-notification" data-item-id="'), d.b(d.v(d.f("tweet_id", a, b, 0))), d.b('" data-tweet-id="'), d.b(d.v(d.f("tweet_id", a, b, 0))), d.b('" data-screen-name="'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('" data-name="'), d.b(d.v(d.f("user_name", a, b, 0))), d.b('" data-user-id="'), d.b(d.v(d.f("user_id", a, b, 0))), d.b('" data-retweeter="'), d.b(d.v(d.f("retweeter", a, b, 0))), d.b('" data-mentions="'), d.b(d.v(d.f("mentions", a, b, 0))), d.b('" data-item-type="tweet">\n  <div class="WebToast-box">\n'), d.b(d.rp("<spoonbill/notification_box_close0", a, b, "    ")), d.b("\n" + c), d.b('    <div class="WebToast-header u-cf">\n      <div class="WebToast-imageBox u-floatLeft">\n        <span class="Icon Icon--message Icon--small u-floatRight"></span>\n      </div>\n      <div class="WebToast-contentBox">\n        Direct Message\n      </div>\n    </div>\n'), d.b(d.rp("<spoonbill/tweet1", a, b, "    ")), d.b('  </div>\n\n  <div class="WebToast-line"></div>\n\n  <div class="WebToast-box WebToast-box--altColor js-inlineReply">\n'), d.b("  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<spoonbill/notification_box_close0": {
                name: "spoonbill/notification_box_close",
                partials: {},
                subs: {}
            },
            "<spoonbill/tweet1": {
                name: "spoonbill/tweet",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["spoonbill/mention"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast original-tweet" data-item-id="'), d.b(d.v(d.f("tweet_id", a, b, 0))), d.b('" data-tweet-id="'), d.b(d.v(d.f("tweet_id", a, b, 0))), d.b('" data-screen-name="'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('" data-name="'), d.b(d.v(d.f("user_name", a, b, 0))), d.b('" data-user-id="'), d.b(d.v(d.f("user_id", a, b, 0))), d.b('" data-retweeter="'), d.b(d.v(d.f("retweeter", a, b, 0))), d.b('" data-mentions="'), d.b(d.v(d.f("mentions", a, b, 0))), d.b('" data-item-type="tweet">\n  <div class="WebToast-box">\n'), d.b(d.rp("<spoonbill/notification_box_close0", a, b, "    ")), d.b("\n" + c), d.b(d.rp("<spoonbill/tweet1", a, b, "    ")), d.b('  </div>\n\n  <div class="WebToast-line"></div>\n\n'), d.b(d.rp("<spoonbill/tweet_actions2", a, b, "  ")), d.b("\n" + c), d.b('  <div class="WebToast-line"></div>\n\n  <div class="WebToast-box WebToast-box--altColor js-inlineReply">\n'), d.b("  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<spoonbill/notification_box_close0": {
                name: "spoonbill/notification_box_close",
                partials: {},
                subs: {}
            },
            "<spoonbill/tweet1": {
                name: "spoonbill/tweet",
                partials: {},
                subs: {}
            },
            "<spoonbill/tweet_actions2": {
                name: "spoonbill/tweet_actions",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["spoonbill/tweet_actions"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="WebToast-actionBar">\n  <ul class="Arrange Arrange--equal">\n    <li class="Arrange-sizeFill">\n      <button class="WebToast-action WebToast-action--reply js-actionReply" type="button">\n        <span class="Icon Icon--reply u-block"></span>\n        <span class="u-hiddenVisually">Reply</span>\n      </button>\n    </li>\n    <li class="Arrange-sizeFill">\n      <button class="WebToast-action WebToast-action--retweet js-actionInlineRetweet" type="button">\n        <span class="Icon Icon--retweet u-block"></span>\n        <span class="u-hiddenVisually">Retweet</span>\n      </button>\n    </li>\n    <li class="Arrange-sizeFill">\n      <button class="WebToast-action WebToast-action--favorite js-actionFavorite" type="button">\n        <span class="Icon Icon--favorite u-block"></span>\n        <span class="u-hiddenVisually">Favorite</span>\n      </button>\n    </li>\n  </ul>\n</div>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["spoonbill/user_identity_fullname_link"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<a class="WebToast-accountLink js-user-profile-link js-action-profile-name" href="/'), d.b(d.v(d.f("user_screen_name", a, b, 0))), d.b('" data-user-id="'), d.b(d.v(d.f("user_id", a, b, 0))), d.b('">'), d.b('<b class="WebToast-fullname">'), d.b(d.v(d.f("user_name", a, b, 0))), d.b("</b>"), d.b("</a>\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["geo/geo_picker"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="geo-picker">\n  <button class="btn js-geo-search-trigger geo-picker-btn icon-btn js-tooltip" type="button" '), d.s(d.f("is_advanced", a, b, 1), a, b, 1, 0, 0, "") || d.b('tabindex="-1" '), d.b('title="Add location">\n    <span class="Icon Icon--geo"></span>\n    <span class="text geo-status">Add location</span>\n  </button>\n  <span class="dropdown-container"></span>\n  <input type="hidden" name="place_id">\n</div>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["geo/dropdown_search_results"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.s(d.f("places", a, b, 1), a, b, 0, 11, 170, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('  <li class="dropdown-link GeoSearch-focusable GeoSearch-result GeoSearch-result--search" data-result-type="search" data-place-id="'), c.b(c.v(c.f("id", a, b, 0))), c.b('">'), c.b(c.v(c.f("full_name", a, b, 0))), c.b("</li>\n")
            }), a.pop()), d.s(d.f("places", a, b, 1), a, b, 1, 0, 0, "") || d.b('  <li class="dropdown-link GeoSearch-focusable GeoSearch-result GeoSearch-result--search GeoSearch-noResults">No search results</li>\n'), d.b('<li class="GeoSearch-result--search dropdown-divider"></li>'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["geo/disabled_dropdown"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="GeoSearch-dropdownMenu dropdown-menu" tabindex="-1">\n  <div class="dropdown-caret">\n    <span class="caret-outer"></span>\n    <span class="caret-inner"></span>\n  </div>\n  <ul>\n    <li class="geo-not-enabled-yet">\n      <h2>Add a location to your Tweets</h2>\n      <p>\n        When you tweet with a location, Twitter stores that location.&#32;\n        You can switch location on/off before each Tweet and always have the option to delete your location history.\n        <a href="http://support.twitter.com/forums/26810/entries/78525" target="_blank">Learn more</a>\n      </p>\n      <div>\n        <button type="button" class="geo-turn-on btn primary-btn">Turn location on</button>\n        <button type="button" class="geo-not-now btn-link">Not now</button>\n      </div>\n    </li>\n  </ul>\n</div>'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["geo/dropdown_places"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.s(d.f("places", a, b, 1), a, b, 0, 11, 242, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('  <li class="dropdown-link GeoSearch-focusable GeoSearch-result GeoSearch-result--place" data-result-type="place" data-place-id="'), c.b(c.v(c.f("id", a, b, 0))), c.b('">'), c.s(c.f("is_first_place", a, b, 1), a, b, 0, 168, 204, "{{ }}") && (c.rs(a, b, function(a, b, c) {
                    c.b('<span class="icon checkmark"></span>')
                }), a.pop()), c.b(c.v(c.f("full_name", a, b, 0))), c.b("</li>\n")
            }), a.pop()), d.b('<li class="GeoSearch-result--place dropdown-divider"></li>'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["geo/enabled_dropdown"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="GeoSearch-dropdownMenu dropdown-menu" tabindex="-1">\n  <div class="dropdown-caret">\n    <span class="caret-outer"></span>\n    <span class="caret-inner"></span>\n  </div>\n  <ul>\n    <li class="geo-query-location">\n      <input class="GeoSearch-queryInput" type="text" autocomplete="off" placeholder="Search for a neighborhood or city">\n      <span class="icon generic-search"></span>\n    </li>\n    <li class="geo-dropdown-status"></li>\n    <li class="dropdown-link geo-turn-off-item GeoSearch-focusable">\n      <span class="icon close"></span>Turn off location\n    </li>\n  </ul>\n</div>'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["media/media_picker"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="photo-selector">\n  <button aria-hidden="true" class="btn icon-btn" type="button" tabindex="-1">\n    <span class="tweet-camera Icon Icon--camera"></span>\n    <span class="text add-photo-label">Add photo</span>\n  </button>\n  <div class="image-selector">\n    <input type="hidden" name="media_data_empty" class="file-data">\n    <div class="multi-photo-data-container hidden">\n    </div>\n    <label class="t1-label">\n      <span class="visuallyhidden">Add Photo</span>\n      <input type="file" name="media_empty" class="file-input js-tooltip" tabindex="-1" title="Add Photo">\n    </label>\n    <div class="swf-container"></div>\n  </div>\n</div>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["dialogs/close_button"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<button type="button" class="modal-btn modal-close js-close">\n  <span class="Icon Icon--close Icon--medium">\n    <span class="visuallyhidden">Close</span>\n  </span>\n</button>\n'), d.fl()
        },
        partials: {},
        subs: {}
    }), b["dialogs/keyboard_shortcuts_dialog"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div id="keyboard-shortcut-dialog" class="modal-container">\n  <div class="close-modal-background-target"></div>\n  <div class="modal modal-large draggable">\n    <div class="modal-content">\n'), d.b(d.rp("<dialogs/close_button0", a, b, "      ")), d.b("\n" + c), d.b('      <div class="modal-header">\n        <h3 class="modal-title">Keyboard shortcuts</h3>\n      </div>\n\n'), d.b('      <div class="modal-body">\n\n        <div class="keyboard-shortcuts clearfix" id="keyboard-shortcut-menu">\n          <p class="visuallyhidden">\n            Note: To use these shortcuts, users of screen readers may need to toggle off the virtual navigation.\n          </p>\n'), d.s(d.f("logged_in", a, b, 1), a, b, 0, 687, 7730, "{{ }}") && (d.rs(a, b, function(a, b, d) {
                d.b('          <table class="modal-table" summary="Shortcuts for common actions.">\n            <thead>\n              <tr>\n                <th colspan="2">Actions</th>\n              </tr>\n            </thead>\n            <tbody>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">n</b>\n                </td>\n                <td class="shortcut-label">New Tweet</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">f</b>\n                </td>\n                <td class="shortcut-label">Favorite</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">r</b>\n                </td>\n                <td class="shortcut-label">Reply</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">t</b>\n                </td>\n                <td class="shortcut-label">Retweet</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">m</b>\n                </td>\n                <td class="shortcut-label">Direct message</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">u</b>\n                </td>\n                <td class="shortcut-label">Mute User</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">b</b>\n                </td>\n                <td class="shortcut-label">Block User</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">Enter</b>\n                </td>\n                <td class="shortcut-label">Open Tweet details</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">l</b>\n                </td>\n                <td class="shortcut-label">Close all open Tweets</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">/</b>\n                </td>\n                <td class="shortcut-label">Search</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">'), d.s(d.f("is_mac", a, b, 1), a, b, 0, 3026, 3029, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                    c.b("Cmd")
                }), a.pop()), d.s(d.f("is_mac", a, b, 1), a, b, 1, 0, 0, "") || d.b("Ctrl"), d.b('</b>\n                  <b class="sc-key">Enter</b>\n                </td>\n                <td class="shortcut-label">Send Tweet</td>\n              </tr>\n            </tbody>\n          </table>\n          <table class="modal-table" summary="Shortcuts for navigating between items in timelines.">\n            <thead>\n              <tr>\n                <th colspan="2">Navigation</th>\n              </tr>\n            </thead>\n            <tbody>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">?</b>\n                </td>\n                <td class="shortcut-label">This menu</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">j</b>\n                </td>\n                <td class="shortcut-label">Next Tweet</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">k</b>\n                </td>\n                <td class="shortcut-label">Previous Tweet</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">Space</b>\n                </td>\n                <td class="shortcut-label">Page down</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">.</b>\n                </td>\n                <td class="shortcut-label">Load new Tweets</td>\n              </tr>\n            </tbody>\n          </table>\n          <table class="modal-table" summary="Shortcuts for navigating to different timelines or pages.">\n            <thead>\n              <tr>\n                <th colspan="2">Timelines</th>\n              </tr>\n            </thead>\n            <tbody>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">h</b>\n                </td>\n                <td class="shortcut-label">Home</td>\n              </tr>\n\n'), d.s(d.f("enable_activity", a, b, 1), a, b, 0, 5064, 5786, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                    c.b('                <tr>\n                  <td class="shortcut">\n                    <b class="sc-key">g</b> <b class="sc-key">n</b>\n                  </td>\n                  <td class="shortcut-label">Notifications</td>\n                </tr>\n                <tr>\n                  <td class="shortcut">\n                    <b class="sc-key">g</b> <b class="sc-key">a</b>\n                  </td>\n                  <td class="shortcut-label">Activity</td>\n                </tr>\n                <tr>\n                  <td class="shortcut">\n                    <b class="sc-key">g</b> <b class="sc-key">r</b>\n                  </td>\n                  <td class="shortcut-label">Mentions</td>\n                </tr>\n')
                }), a.pop()), d.s(d.f("enable_activity", a, b, 1), a, b, 1, 0, 0, "") || d.b('                  <tr>\n                    <td class="shortcut">\n                      <b class="sc-key">g</b> <b class="sc-key">r</b>\n                    </td>\n                    <td class="shortcut-label">Mentions</td>\n                  </tr>\n'), d.b("\n" + c), d.b('              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">d</b>\n                </td>\n                <td class="shortcut-label">Discover</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">p</b>\n                </td>\n                <td class="shortcut-label">Profile</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">f</b>\n                </td>\n                <td class="shortcut-label">Favorites</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">l</b>\n                </td>\n                <td class="shortcut-label">Lists</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">m</b>\n                </td>\n                <td class="shortcut-label">Messages</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">s</b>\n                </td>\n                <td class="shortcut-label">Settings</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">u</b>\n                </td>\n                <td class="shortcut-label">Go to user...</td>\n              </tr>\n            </tbody>\n          </table>\n')
            }), a.pop()), d.s(d.f("logged_in", a, b, 1), a, b, 1, 0, 0, "") || d.b('          <table class="modal-table">\n            <tbody>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">Enter</b>\n                </td>\n                <td class="shortcut-label">Open Tweet details</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">g</b> <b class="sc-key">u</b>\n                </td>\n                <td class="shortcut-label">Go to user...</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">?</b>\n                </td>\n                <td class="shortcut-label">This menu</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">j</b>\n                </td>\n                <td class="shortcut-label">Next Tweet</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">k</b>\n                </td>\n                <td class="shortcut-label">Previous Tweet</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">Space</b>\n                </td>\n                <td class="shortcut-label">Page down</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">/</b>\n                </td>\n                <td class="shortcut-label">Search</td>\n              </tr>\n              <tr>\n                <td class="shortcut">\n                  <b class="sc-key">.</b>\n                </td>\n                <td class="shortcut-label">Load new Tweets</td>\n              </tr>\n            </tbody>\n          </table>\n'), d.b("        </div>\n      </div>\n    </div>\n  </div>\n</div>\n\n\n"), d.fl()
        },
        partials: {
            "<dialogs/close_button0": {
                name: "dialogs/close_button",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["dialogs/create_or_edit_custom_timeline_dialog"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="close-modal-background-target"></div>\n<div class="modal modal-medium draggable">\n  <div class="modal-content">\n'), d.b(d.rp("<dialogs/close_button0", a, b, "    ")), d.b('    <div class="modal-header">\n      <h3 class="modal-title">\n        '), d.s(d.f("is_create_dialog", a, b, 1), a, b, 0, 244, 267, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("Create a new collection")
            }), a.pop()), d.b("\n" + c), d.b("        "), d.s(d.f("is_create_dialog", a, b, 1), a, b, 1, 0, 0, "") || d.b("Edit collection"), d.b("\n" + c), d.b('      </h3>\n    </div>\n\n    <div class="modal-body">\n      <div class="custom-timeline-editor">\n        '), d.s(d.f("is_create_dialog", a, b, 1), a, b, 0, 480, 671, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('<p class="explanation">\n          Collections are public sets of Tweets that you control.\n          <br />\n          Create a collection, add Tweets, and share it with the world.\n        </p>')
            }), a.pop()), d.b("\n"), d.b("\n" + c), d.b('        <div class="field custom-timeline-name-field">\n          <label class="t1-label" for="custom-timeline-name">Collection name</label>\n          <input type="text" class="text" name="name" id="custom-timeline-name" value="'), d.b(d.v(d.f("timeline_name", a, b, 0))), d.b('" aria-describedby="custom-timeline-name-help" />\n          <p class="help-text">\n            <span id="custom-timeline-name-help">25 characters</span>\n            <span class="custom-timeline-name-count">25</span>\n          </p>\n        </div>\n        <hr/>\n\n        <div class="field custom-timeline-description-field">\n          <label class="t1-label" for="custom-timeline-description">Description</label>\n          <textarea name="custom-timeline-description" id="custom-timeline-description" aria-describedby="custom-timeline-description-help">'), d.b(d.v(d.f("timeline_description", a, b, 0))), d.b('</textarea>\n          <p class="help-text">\n            <span id="custom-timeline-description-help">160 characters, optional</span>\n            <span class="custom-timeline-description-count">160</span>\n          </p>\n        </div>\n        <hr/>\n\n        <fieldset class="field">\n          <legend class="t1-legend">Collection order</legend>\n          <div class="options">\n            <label class="t1-label u-block">\n              <input class="radio u-inlineBlock" type="radio" name="'), d.b(d.v(d.f("id_prefix", a, b, 0))), d.b('-timeline-order" value="curation_reverse_chron"'), d.s(d.f("curation_reverse_chron", a, b, 1), a, b, 0, 2087, 2105, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(' checked="checked"')
            }), a.pop()), d.b(' />\n              <b>Recently added first</b>\n            </label>\n            <label class="t1-label u-block">\n              <input class="radio u-inlineBlock" type="radio" name="'), d.b(d.v(d.f("id_prefix", a, b, 0))), d.b('-timeline-order" value="tweet_chron"'), d.s(d.f("tweet_chron", a, b, 1), a, b, 0, 2377, 2395, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(' checked="checked"')
            }), a.pop()), d.b(' />\n              <b>Oldest first</b>\n            </label>\n            <label class="t1-label u-block">\n              <input class="radio u-inlineBlock" type="radio" name="'), d.b(d.v(d.f("id_prefix", a, b, 0))), d.b('-timeline-order" value="tweet_reverse_chron"'), d.s(d.f("tweet_reverse_chron", a, b, 1), a, b, 0, 2664, 2682, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b(' checked="checked"')
            }), a.pop()), d.b(' />\n              <b>Newest first</b>\n            </label>\n          </div>\n        </fieldset>\n        <hr/>\n\n        <div class="custom-timeline-editor-save">\n          <button type="button" class="btn btn-primary update-custom-timeline-button" data-custom-timeline-id="'), d.b(d.v(d.f("id", a, b, 0))), d.b('" disabled>\n            '), d.s(d.f("is_create_dialog", a, b, 1), a, b, 0, 3029, 3046, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("Create collection")
            }), a.pop()), d.b("\n" + c), d.b("            "), d.s(d.f("is_create_dialog", a, b, 1), a, b, 1, 0, 0, "") || d.b("Save collection"), d.b("\n" + c), d.b("          </button>\n        </div>\n      </div>\n    </div>\n  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<dialogs/close_button0": {
                name: "dialogs/close_button",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["dialogs/curate_dialog"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="close-modal-background-target"></div>\n<div class="modal modal-medium draggable">\n  <div class="modal-content">\n'), d.b(d.rp("<dialogs/close_button0", a, b, "    ")), d.b('    <div class="modal-header">\n      <h3 class="modal-title">Add Tweet to collection</h3>\n    </div>\n    <div class="modal-body">\n      <div class="timeline-selector"></div>\n    </div>\n    <div class="modal-footer">\n'), d.b(d.rp("<create_custom_timeline_button1", a, b, "      ")), d.b("\n" + c), d.b('      <button type="button" class="btn js-close">Cancel</button>\n      <button type="button" class="btn primary-btn modal-submit js-submit">Save changes</button>\n    </div>\n  </div>\n</div>\n'), d.fl()
        },
        partials: {
            "<dialogs/close_button0": {
                name: "dialogs/close_button",
                partials: {},
                subs: {}
            },
            "<create_custom_timeline_button1": {
                name: "create_custom_timeline_button",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["dialogs/home_nav_headlined_popover"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b("<h2>Your timeline is ready!</h2>\n<p>Click Home whenever you want to go to your timeline</p>"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["dialogs/confirm_dialog"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div id="confirm_dialog" class="modal-container">\n  <div class="close-modal-background-target"></div>\n  <div class="modal draggable">\n    <div class="modal-content">\n'), d.b(d.rp("<dialogs/close_button0", a, b, "      ")), d.b('      <div class="modal-header">\n        <h3 class="modal-title">'), d.b(d.v(d.f("title_text", a, b, 0))), d.b("</h3>\n      </div>\n"), d.s(d.f("body_text", a, b, 1), a, b, 0, 316, 426, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('        <div class="modal-body">\n          <p class="modal-body-text">'), c.b(c.v(c.f("body_text", a, b, 0))), c.b("</p>\n        </div>\n")
            }), a.pop()), d.b('      <div class="modal-footer">\n        <button class="btn js-close" id="confirm_dialog_cancel_button">'), d.b(d.v(d.f("cancel_text", a, b, 0))), d.b('</button>\n        <button id="confirm_dialog_submit_button" class="btn primary-btn modal-submit">'), d.b(d.v(d.f("submit_text", a, b, 0))), d.b("</button>\n      </div>\n    </div>\n  </div>\n</div>\n"), d.fl()
        },
        partials: {
            "<dialogs/close_button0": {
                name: "dialogs/close_button",
                partials: {},
                subs: {}
            }
        },
        subs: {}
    }), b["dialogs/home_nav_follow_popover"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.s(d.f("is_first_follow", a, b, 1), a, b, 0, 20, 748, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("  <p>\n"), c.s(c.f("total_num_one", a, b, 1), a, b, 0, 49, 125, "{{ }}") && (c.rs(a, b, function(a, b, c) {
                    c.b("      Tweets from <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong> are now in your timeline.\n")
                }), a.pop()), c.s(c.f("total_num_two", a, b, 1), a, b, 0, 166, 273, "{{ }}") && (c.rs(a, b, function(a, b, c) {
                    c.b("      Tweets from <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong> and <strong>"), c.b(c.v(c.f("user2", a, b, 0))), c.b("</strong> are now in your timeline.\n")
                }), a.pop()), c.s(c.f("total_num_three", a, b, 1), a, b, 0, 316, 439, "{{ }}") && (c.rs(a, b, function(a, b, c) {
                    c.b("      Tweets from <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong>, <strong>"), c.b(c.v(c.f("user2", a, b, 0))), c.b("</strong> and 1 other person are now in your timeline.\n")
                }), a.pop()), c.s(c.f("total_num_other", a, b, 1), a, b, 0, 484, 614, "{{ }}") && (c.rs(a, b, function(a, b, c) {
                    c.b("      Tweets from <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong>, <strong>"), c.b(c.v(c.f("user2", a, b, 0))), c.b("</strong> and "), c.b(c.v(c.f("num_others", a, b, 0))), c.b(" others are now in your timeline.\n")
                }), a.pop()), c.b("  </p>\n  <p>\n    <small>When you're finished following people, click Home to go to your timeline.</small>\n  </p>\n")
            }), a.pop()), d.s(d.f("is_first_follow", a, b, 1), a, b, 1, 0, 0, "") || (d.s(d.f("total_num_one", a, b, 1), a, b, 0, 810, 857, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    You followed <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong>\n")
            }), a.pop()), d.s(d.f("total_num_two", a, b, 1), a, b, 0, 896, 974, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    You followed <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong> and <strong>"), c.b(c.v(c.f("user2", a, b, 0))), c.b("</strong>\n")
            }), a.pop()), d.s(d.f("total_num_three", a, b, 1), a, b, 0, 1015, 1109, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    You followed <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong>, <strong>"), c.b(c.v(c.f("user2", a, b, 0))), c.b("</strong> and 1 other person\n")
            }), a.pop()), d.s(d.f("total_num_other", a, b, 1), a, b, 0, 1152, 1253, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b("    You followed <strong>"), c.b(c.v(c.f("user1", a, b, 0))), c.b("</strong>, <strong>"), c.b(c.v(c.f("user2", a, b, 0))), c.b("</strong> and "), c.b(c.v(c.f("num_others", a, b, 0))), c.b(" others\n")
            }), a.pop())), d.fl()
        },
        partials: {},
        subs: {}
    }), b["highline/components/iframe_async_image_upload"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="IframeAsyncImageUpload u-hidden">\n  <iframe class="IframeAsyncImageUpload-iframe" name="IframeAsyncImageUpload-iframe--'), d.b(d.v(d.f("upload_id", a, b, 0))), d.b('" src="about:blank;"></iframe>\n  <form class="IframeAsyncImageUpload-form" enctype="multipart/form-data" method="post" action="'), d.b(d.v(d.f("form_action", a, b, 0))), d.b('">\n    <input type="hidden" name="authenticity_token" value="'), d.b(d.v(d.f("auth_token", a, b, 0))), d.b('">\n    <input type="hidden" name="origin" value="'), d.b(d.v(d.f("origin", a, b, 0))), d.b('">\n    <input type="hidden" name="iframe_upload" value="true">\n    <input type="hidden" name="upload_id" value="'), d.b(d.v(d.f("upload_id", a, b, 0))), d.b('">\n    <input type="hidden" name="fileName" value="'), d.b(d.v(d.f("file_name", a, b, 0))), d.b('">\n'), d.s(d.f("file_data", a, b, 1), a, b, 0, 630, 701, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('      <input type="hidden" name="fileData" value="'), c.b(c.v(c.f("file_data", a, b, 0))), c.b('">\n')
            }), a.pop()), d.s(d.f("page_context", a, b, 1), a, b, 0, 737, 815, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('      <input type="hidden" name="page_context" value="'), c.b(c.v(c.f("page_context", a, b, 0))), c.b('">\n')
            }), a.pop()), d.s(d.f("section_context", a, b, 1), a, b, 0, 857, 941, "{{ }}") && (d.rs(a, b, function(a, b, c) {
                c.b('      <input type="hidden" name="section_context" value="'), c.b(c.v(c.f("section_context", a, b, 0))), c.b('">\n')
            }), a.pop()), d.b("  </form>\n</div>"), d.fl()
        },
        partials: {},
        subs: {}
    }), b["highline/components/color_picker"] = new Hogan.Template({
        code: function(a, b, c) {
            var d = this;
            return d.b(c = c || ""), d.b('<div class="ColorPicker dropdown-menu" tabindex="-1" role="dialog" aria-labelledby="color-picker">'), d.b("<fieldset>"), d.b('<legend id="color-picker" class="u-hiddenVisually">Color Picker</legend>'), d.b('<div class="dropdown-caret">'), d.b('<span class="caret-outer"></span>'), d.b('<span class="caret-inner"></span>'), d.b("</div>"), d.b('<div class="ColorPicker-colorList">'), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#fa743e" style="background-color: #fa743e">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">orange</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#ffcc4d" style="background-color: #ffcc4d">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">yellow</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#94d487" style="background-color: #94d487">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">lime</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#4a913c" style="background-color: #4a913c">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">green</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#89c9fa" style="background-color: #89c9fa">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">blue</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#3b94d9" style="background-color: #3b94d9">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">navy</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#abb8c2" style="background-color: #abb8c2">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">grey</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#dd2e44" style="background-color: #dd2e44">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">red</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#f5abb5" style="background-color: #f5abb5">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">pink</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-color" data-color="#9266cc" style="background-color: #9266cc">'), d.b('<label class="Icon Icon--smallest">'), d.b('<span class="u-hiddenVisually">purple</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-transparent">'), d.b('<div class="ColorPicker-item ColorPicker-more" style="background-color: #f0f0f0">'), d.b('<label class="Icon Icon--smallest Icon--add">'), d.b('<span class="u-hiddenVisually">more</span>'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b("</div>"), d.b("</div>"), d.b('<div class="ColorPicker-item ColorPicker-hex" data-color="#f0f0f0" data-original-color="#f0f0f0" style="background-color: #f0f0f0">'), d.b('<label class="Icon Icon--smallest Icon--discover">'), d.b('<input class="u-hiddenVisually" type="radio" name="ColorPicker-colorGroup">'), d.b("</label>"), d.b('<input type="text" class="ColorPicker-hexInput" aria-labelledby="js-userColorButton" placeholder="" maxlength="6">'), d.b("</div>"), d.b("</div>"), d.b("</fieldset>"), d.b("</div>\n"), d.fl()
        },
        partials: {},
        subs: {}
    }), a.exports = b
})
define("app/ui/dialogs/confirm_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/utils/with_event_params", "template"], function(module, require, exports) {
    function confirmDialog() {
        this.defaultAttrs({
            confirmDialogSelector: "#confirm_dialog",
            cancelSelector: "#confirm_dialog_cancel_button",
            submitSelector: "#confirm_dialog_submit_button",
            modalSizes: {
                small: "modal-small",
                medium: "modal-medium",
                large: "modal-large"
            }
        }), this.openWithOptions = function(a, b) {
            if (this.openState)
                return;
            this.attr.eventParams = {
                action: b.action
            }, this.attr.top = b.top, this.eventNode = a.target;
            var c = {
                title_text: b.titleText,
                body_text: b.bodyText,
                cancel_text: b.cancelText,
                submit_text: b.submitText
            }, d = template["dialogs/confirm_dialog"].render(c, template);
            $("#confirm_dialog").length ? $("#confirm_dialog").replaceWith(d) : $("body").append(d), this.$dialogContainer = $("#confirm_dialog"), this.$dialog = this.$dialogContainer.find(this.attr.modalSelector);
            var e = this.attr.modalSizes[b.modalSize];
            typeof e != "undefined" && this.$dialog.addClass(e), this.open()
        }, this.submit = function(a, b) {
            this.trigger(this.eventNode, "ui{{action}}Confirm"), this.close()
        }, this.cancel = function(a, b) {
            this.trigger(this.eventNode, "ui{{action}}Cancel"), a.type == "click" && this.close()
        }, this.after("initialize", function() {
            this.on(document, "uiOpenConfirmDialog", this.openWithOptions), this.on("click", {
                submitSelector: this.submit,
                cancelSelector: this.cancel
            }), this.on("uiDialogCloseRequested", {
                confirmDialogSelector: this.cancel
            })
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withEventParams = require("app/utils/with_event_params"), template = require("template");
    module.exports = defineComponent(confirmDialog, withDialog, withEventParams)
});
define("app/ui/connect_badge", ["module", "require", "exports", "core/component", "app/utils/storage/core"], function(module, require, exports) {
    function ConnectBadge() {
        this.defaultAttrs({
            keepBadgeStorageKey: "keep_badge_until",
            keepBadgeInterval: 1e3
        }), this.processNewNotifications = function(a, b) {
            b && b.b && b.b.status == "ok" && b.b.response && this.possiblyUpdateBadgeAndHighlighting(b.b.response)
        }, this.possiblyUpdateBadgeAndHighlighting = function(a) {
            a.show_badge_highlighting && a.count >= 0 && (this.trigger("uiUpdateConnectBadge", a.count), a.timestamp >= 0 && (a.count > 0 || this.pastStorageTime()) && this.trigger("uiUpdateActivityHighlighting", a.timestamp), this.updateStorageTime())
        }, this.pastStorageTime = function() {
            var a = (new Date).getTime(), b = this.storage.getItem(this.attr.keepBadgeStorageKey) || {
                until: 0
            };
            return b.until < a
        }, this.updateStorageTime = function() {
            var a = (new Date).getTime(), b = this.storage.getItem(this.attr.keepBadgeStorageKey) || {
                until: 0
            };
            this.storage.setItem(this.attr.keepBadgeStorageKey, {
                until: a + this.attr.keepBadgeInterval
            })
        }, this.after("initialize", function() {
            this.storage = new Storage("connect_badge"), this.updateStorageTime(0), this.on(document, "dataNotificationsReceived", this.processNewNotifications)
        })
    }
    var defineComponent = require("core/component"), Storage = require("app/utils/storage/core");
    module.exports = defineComponent(ConnectBadge)
});
define("app/utils/caret", ["module", "require", "exports"], function(module, require, exports) {
    var caret = {
        getPosition: function(a) {
            try {
                if (document.selection) {
                    var b = document.selection.createRange();
                    return b.moveStart("character", - a.value.length), b.text.length
                }
                if (typeof a.selectionStart == "number")
                    return a.selectionStart
            } catch (c) {}
            return 0
        },
        setPosition: function(a, b) {
            try {
                if (document.selection) {
                    var c = a.createTextRange();
                    c.collapse(!0), c.moveEnd("character", b), c.moveStart("character", b), c.select()
                } else 
                    typeof a.selectionStart == "number" && (a.selectionStart = b, a.selectionEnd = b)
            } catch (d) {}
        },
        getSelection: function() {
            return window.getSelection ? window.getSelection().toString() : document.selection.createRange().text
        }
    };
    module.exports = caret
});
define("app/ui/with_text_editor", ["module", "require", "exports", "app/utils/caret"], function(module, require, exports) {
    function withTextEditor() {
        this.caret = caret, this.defaultAttrs({
            textSelector: "textarea.tweet-box"
        }), this.hasFocus = function() {
            return document.activeElement === this.$text[0]
        }, this.getVisibleText = function() {
            return this.cleanedText || ""
        }, this.setVisibleText = function(a) {
            this.detectUpdatedText(!1, a)
        }, this.addHiddenText = function(a) {
            return a
        }, this.cleanText = function(a) {
            return a
        }, this.detectUpdatedText = function(a, b) {
            b === undefined ? b = this.$text.val() : this.$text.val(b);
            if (b !== this.prevText || a)
                this.prevText = b, this.updateCleanedTextAndOffset(this.cleanText(b), caret.getPosition(this.$text[0]))
        }, this.updateCleanedTextAndOffset = function(a, b) {
            this.cleanedText = a, this.trigger("uiTextChanged", {
                text: a,
                position: b
            })
        }, this.ignoreDuringFakeFocus = function(a) {
            var b = this;
            return function() {
                if (!b.$text.hasClass("fake-focus"))
                    return a.apply(this, arguments)
            }
        }, this.setSelection = function(a) {}, this.setCursorPosition = function(a) {
            a === undefined && (a = this.attr.cursorPosition), a === undefined && (a = this.$text.val().length), caret.setPosition(this.$text.get(0), a)
        }, this.changeTextAndPosition = function(a, b) {
            this.setVisibleText(b.text), this.setCursorPosition(b.position)
        }, this.focus = function() {
            this.hasFocus() || this.$text.focus()
        }, this.initTextNode = function() {
            this.$text = this.select("textSelector"), this.detectUpdatedText()
        }, this.after("initialize", function() {
            this.on("uiChangeTextAndPosition", this.changeTextAndPosition)
        })
    }
    var caret = require("app/utils/caret");
    module.exports = withTextEditor
});
/*! twitter-text-js 1.10.0 (c) 2012 Twitter, Inc. http://www.apache.org/licenses/LICENSE-2.0 */
provide("lib/twitter-text", function(a) {
    var b = {}; /*!
     * twitter-text-js 1.10.0
     *
     * Copyright 2012 Twitter, Inc.
     *
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this work except in compliance with the License.
     * You may obtain a copy of the License at:
     *
     *    http://www.apache.org/licenses/LICENSE-2.0
     */
    (function() {
        function c(a, c) {
            return c = c || "", typeof a != "string" && (a.global && c.indexOf("g") < 0 && (c += "g"), a.ignoreCase && c.indexOf("i") < 0 && (c += "i"), a.multiline && c.indexOf("m") < 0 && (c += "m"), a = a.source), new RegExp(a.replace(/#\{(\w+)\}/g, function(a, c) {
                var d = b.txt.regexen[c] || "";
                return typeof d != "string" && (d = d.source), d
            }), c)
        }
        function d(a, b) {
            return a.replace(/#\{(\w+)\}/g, function(a, c) {
                return b[c] || ""
            })
        }
        function e(a, b, c) {
            var d = String.fromCharCode(b);
            return c !== b && (d += "-" + String.fromCharCode(c)), a.push(d), a
        }
        function q(a) {
            var b = {};
            for (var c in a)
                a.hasOwnProperty(c) && (b[c] = a[c]);
            return b
        }
        function u(a, b, c) {
            return c?!a || a.match(b) && RegExp["$&"] === a : typeof a == "string" && a.match(b) && RegExp["$&"] === a
        }
        b.txt = {}, b.txt.regexen = {};
        var a = {
            "&": "&amp;",
            ">": "&gt;",
            "<": "&lt;",
            '"': "&quot;",
            "'": "&#39;"
        };
        b.txt.htmlEscape = function(b) {
            return b && b.replace(/[&"'><]/g, function(b) {
                return a[b]
            })
        }, b.txt.regexSupplant = c, b.txt.stringSupplant = d, b.txt.addCharsToCharClass = e;
        var f = String.fromCharCode, g = [f(32), f(133), f(160), f(5760), f(6158), f(8232), f(8233), f(8239), f(8287), f(12288)];
        e(g, 9, 13), e(g, 8192, 8202);
        var h = [f(65534), f(65279), f(65535)];
        e(h, 8234, 8238), b.txt.regexen.spaces_group = c(g.join("")), b.txt.regexen.spaces = c("[" + g.join("") + "]"), b.txt.regexen.invalid_chars_group = c(h.join("")), b.txt.regexen.punct = /\!'#%&'\(\)*\+,\\\-\.\/:;<=>\?@\[\]\^_{|}~\$/, b.txt.regexen.rtl_chars = /[\u0600-\u06FF]|[\u0750-\u077F]|[\u0590-\u05FF]|[\uFE70-\uFEFF]/gm, b.txt.regexen.non_bmp_code_pairs = /[\uD800-\uDBFF][\uDC00-\uDFFF]/gm;
        var i = [];
        e(i, 1024, 1279), e(i, 1280, 1319), e(i, 11744, 11775), e(i, 42560, 42655), e(i, 1425, 1471), e(i, 1473, 1474), e(i, 1476, 1477), e(i, 1479, 1479), e(i, 1488, 1514), e(i, 1520, 1524), e(i, 64274, 64296), e(i, 64298, 64310), e(i, 64312, 64316), e(i, 64318, 64318), e(i, 64320, 64321), e(i, 64323, 64324), e(i, 64326, 64335), e(i, 1552, 1562), e(i, 1568, 1631), e(i, 1646, 1747), e(i, 1749, 1756), e(i, 1758, 1768), e(i, 1770, 1775), e(i, 1786, 1788), e(i, 1791, 1791), e(i, 1872, 1919), e(i, 2208, 2208), e(i, 2210, 2220), e(i, 2276, 2302), e(i, 64336, 64433), e(i, 64467, 64829), e(i, 64848, 64911), e(i, 64914, 64967), e(i, 65008, 65019), e(i, 65136, 65140), e(i, 65142, 65276), e(i, 8204, 8204), e(i, 3585, 3642), e(i, 3648, 3662), e(i, 4352, 4607), e(i, 12592, 12677), e(i, 43360, 43391), e(i, 44032, 55215), e(i, 55216, 55295), e(i, 65441, 65500), e(i, 12449, 12538), e(i, 12540, 12542), e(i, 65382, 65439), e(i, 65392, 65392), e(i, 65296, 65305), e(i, 65313, 65338), e(i, 65345, 65370), e(i, 12353, 12438), e(i, 12441, 12446), e(i, 13312, 19903), e(i, 19968, 40959), e(i, 173824, 177983), e(i, 177984, 178207), e(i, 194560, 195103), e(i, 12291, 12291), e(i, 12293, 12293), e(i, 12347, 12347), b.txt.regexen.nonLatinHashtagChars = c(i.join(""));
        var j = [];
        e(j, 192, 214), e(j, 216, 246), e(j, 248, 255), e(j, 256, 591), e(j, 595, 596), e(j, 598, 599), e(j, 601, 601), e(j, 603, 603), e(j, 611, 611), e(j, 616, 616), e(j, 623, 623), e(j, 626, 626), e(j, 649, 649), e(j, 651, 651), e(j, 699, 699), e(j, 768, 879), e(j, 7680, 7935), b.txt.regexen.latinAccentChars = c(j.join("")), b.txt.regexen.hashSigns = /[#＃]/, b.txt.regexen.hashtagAlpha = c(/[a-z_#{latinAccentChars}#{nonLatinHashtagChars}]/i), b.txt.regexen.hashtagAlphaNumeric = c(/[a-z0-9_#{latinAccentChars}#{nonLatinHashtagChars}]/i), b.txt.regexen.endHashtagMatch = c(/^(?:#{hashSigns}|:\/\/)/), b.txt.regexen.hashtagBoundary = c(/(?:^|$|[^&a-z0-9_#{latinAccentChars}#{nonLatinHashtagChars}])/), b.txt.regexen.validHashtag = c(/(#{hashtagBoundary})(#{hashSigns})(#{hashtagAlphaNumeric}*#{hashtagAlpha}#{hashtagAlphaNumeric}*)/gi), b.txt.regexen.validMentionPrecedingChars = /(?:^|[^a-zA-Z0-9_!#$%&*@＠]|(?:rt|RT|rT|Rt):?)/, b.txt.regexen.atSigns = /[@＠]/, b.txt.regexen.validMentionOrList = c("(#{validMentionPrecedingChars})(#{atSigns})([a-zA-Z0-9_]{1,20})(/[a-zA-Z][a-zA-Z0-9_-]{0,24})?", "g"), b.txt.regexen.validReply = c(/^(?:#{spaces})*#{atSigns}([a-zA-Z0-9_]{1,20})/), b.txt.regexen.endMentionMatch = c(/^(?:#{atSigns}|[#{latinAccentChars}]|:\/\/)/), b.txt.regexen.validUrlPrecedingChars = c(/(?:[^A-Za-z0-9@＠$#＃#{invalid_chars_group}]|^)/), b.txt.regexen.invalidUrlWithoutProtocolPrecedingChars = /[-_.\/]$/, b.txt.regexen.invalidDomainChars = d("#{punct}#{spaces_group}#{invalid_chars_group}", b.txt.regexen), b.txt.regexen.validDomainChars = c(/[^#{invalidDomainChars}]/), b.txt.regexen.validSubdomain = c(/(?:(?:#{validDomainChars}(?:[_-]|#{validDomainChars})*)?#{validDomainChars}\.)/), b.txt.regexen.validDomainName = c(/(?:(?:#{validDomainChars}(?:-|#{validDomainChars})*)?#{validDomainChars}\.)/), b.txt.regexen.validGTLD = c(RegExp("(?:(?:abogado|academy|accountants|active|actor|aero|agency|airforce|allfinanz|alsace|archi|army|arpa|asia|associates|attorney|auction|audio|autos|axa|band|bar|bargains|bayern|beer|berlin|best|bid|bike|bio|biz|black|blackfriday|blue|bmw|bnpparibas|boo|boutique|brussels|budapest|build|builders|business|buzz|bzh|cab|cal|camera|camp|cancerresearch|capetown|capital|caravan|cards|care|career|careers|casa|cash|cat|catering|center|ceo|cern|channel|cheap|christmas|chrome|church|citic|city|claims|cleaning|click|clinic|clothing|club|codes|coffee|college|cologne|com|community|company|computer|condos|construction|consulting|contractors|cooking|cool|coop|country|credit|creditcard|crs|cruises|cuisinella|cymru|dad|dance|dating|day|deals|degree|democrat|dental|dentist|desi|diamonds|diet|digital|direct|directory|discount|dnp|domains|durban|dvag|eat|edu|education|email|emerck|engineer|engineering|enterprises|equipment|esq|estate|eus|events|exchange|expert|exposed|fail|farm|feedback|finance|financial|fish|fishing|fitness|flights|florist|flsmidth|fly|foo|forsale|foundation|frl|frogans|fund|furniture|futbol|gal|gallery|gbiz|gent|gift|gifts|gives|glass|gle|global|globo|gmail|gmo|gmx|google|gop|gov|graphics|gratis|green|gripe|guide|guitars|guru|hamburg|haus|healthcare|help|here|hiphop|hiv|holdings|holiday|homes|horse|host|hosting|house|how|ibm|immo|immobilien|industries|info|ing|ink|institute|insure|int|international|investments|jetzt|jobs|joburg|juegos|kaufen|kim|kitchen|kiwi|koeln|krd|kred|lacaixa|land|lawyer|lease|lgbt|life|lighting|limited|limo|link|loans|london|lotto|ltda|luxe|luxury|maison|management|mango|market|marketing|media|meet|melbourne|meme|menu|miami|mil|mini|mobi|moda|moe|monash|mortgage|moscow|motorcycles|mov|museum|nagoya|name|navy|net|network|neustar|new|nexus|ngo|nhk|ninja|nra|nrw|nyc|okinawa|ong|onl|ooo|org|organic|otsuka|ovh|paris|partners|parts|pharmacy|photo|photography|photos|physio|pics|pictures|pink|pizza|place|plumbing|pohl|poker|post|praxi|press|pro|prod|productions|prof|properties|property|pub|qpon|quebec|realtor|recipes|red|rehab|reise|reisen|ren|rentals|repair|report|republican|rest|restaurant|reviews|rich|rio|rip|rocks|rodeo|rsvp|ruhr|ryukyu|saarland|sarl|sca|scb|schmidt|schule|scot|services|sexy|shiksha|shoes|singles|social|software|sohu|solar|solutions|soy|space|spiegel|supplies|supply|support|surf|surgery|suzuki|systems|taipei|tatar|tattoo|tax|technology|tel|tienda|tips|tirol|today|tokyo|tools|top|town|toys|trade|training|travel|tui|university|uno|uol|vacations|vegas|ventures|vermögensberater|vermögensberatung|versicherung|vet|viajes|villas|vision|vlaanderen|vodka|vote|voting|voto|voyage|wales|wang|watch|webcam|website|wed|wedding|whoswho|wien|wiki|williamhill|wme|work|works|world|wtc|wtf|xxx|xyz|yachts|yandex|yoga|yokohama|youtube|zip|zone|дети|москва|онлайн|орг|рус|сайт|بازار|شبكة|موقع|संगठन|みんな|世界|中信|中文网|企业|佛山|公司|公益|商城|商标|在线|广东|我爱你|手机|政务|机构|游戏|移动|组织机构|网址|网络|集团|삼성)(?=[^0-9a-zA-Z@]|$))")), b.txt.regexen.validCCTLD = c(RegExp("(?:(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bl|bm|bn|bo|bq|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cu|cv|cw|cx|cy|cz|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mf|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|um|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw|бел|мкд|мон|рф|срб|укр|қаз|الاردن|الجزائر|السعودية|المغرب|امارات|ایران|بھارت|تونس|سودان|سورية|عراق|عمان|فلسطين|قطر|مصر|مليسيا|پاکستان|भारत|বাংলা|ভারত|ਭਾਰਤ|ભારત|இந்தியா|இலங்கை|சிங்கப்பூர்|భారత్|ලංකා|ไทย|გე|中国|中國|台湾|台灣|新加坡|香港|한국)(?=[^0-9a-zA-Z@]|$))")), b.txt.regexen.validPunycode = c(/(?:xn--[0-9a-z]+)/), b.txt.regexen.validSpecialCCTLD = c(RegExp("(?:(?:co|tv)(?=[^0-9a-zA-Z@]|$))")), b.txt.regexen.validDomain = c(/(?:#{validSubdomain}*#{validDomainName}(?:#{validGTLD}|#{validCCTLD}|#{validPunycode}))/), b.txt.regexen.validAsciiDomain = c(/(?:(?:[\-a-z0-9#{latinAccentChars}]+)\.)+(?:#{validGTLD}|#{validCCTLD}|#{validPunycode})/gi), b.txt.regexen.invalidShortDomain = c(/^#{validDomainName}#{validCCTLD}$/i), b.txt.regexen.validSpecialShortDomain = c(/^#{validDomainName}#{validSpecialCCTLD}$/i), b.txt.regexen.validPortNumber = c(/[0-9]+/), b.txt.regexen.validGeneralUrlPathChars = c(/[a-z0-9!\*';:=\+,\.\$\/%#\[\]\-_~@|&#{latinAccentChars}]/i), b.txt.regexen.validUrlBalancedParens = c("\\((?:#{validGeneralUrlPathChars}+|(?:#{validGeneralUrlPathChars}*\\(#{validGeneralUrlPathChars}+\\)#{validGeneralUrlPathChars}*))\\)", "i"), b.txt.regexen.validUrlPathEndingChars = c(/[\+\-a-z0-9=_#\/#{latinAccentChars}]|(?:#{validUrlBalancedParens})/i), b.txt.regexen.validUrlPath = c("(?:(?:#{validGeneralUrlPathChars}*(?:#{validUrlBalancedParens}#{validGeneralUrlPathChars}*)*#{validUrlPathEndingChars})|(?:@#{validGeneralUrlPathChars}+/))", "i"), b.txt.regexen.validUrlQueryChars = /[a-z0-9!?\*'@\(\);:&=\+\$\/%#\[\]\-_\.,~|]/i, b.txt.regexen.validUrlQueryEndingChars = /[a-z0-9_&=#\/]/i, b.txt.regexen.extractUrl = c("((#{validUrlPrecedingChars})((https?:\\/\\/)?(#{validDomain})(?::(#{validPortNumber}))?(\\/#{validUrlPath}*)?(\\?#{validUrlQueryChars}*#{validUrlQueryEndingChars})?))", "gi"), b.txt.regexen.validTcoUrl = /^https?:\/\/t\.co\/[a-z0-9]+/i, b.txt.regexen.urlHasProtocol = /^https?:\/\//i, b.txt.regexen.urlHasHttps = /^https:\/\//i, b.txt.regexen.cashtag = /[a-z]{1,6}(?:[._][a-z]{1,2})?/i, b.txt.regexen.validCashtag = c("(^|#{spaces})(\\$)(#{cashtag})(?=$|\\s|[#{punct}])", "gi"), b.txt.regexen.validateUrlUnreserved = /[a-z0-9\-._~]/i, b.txt.regexen.validateUrlPctEncoded = /(?:%[0-9a-f]{2})/i, b.txt.regexen.validateUrlSubDelims = /[!$&'()*+,;=]/i, b.txt.regexen.validateUrlPchar = c("(?:#{validateUrlUnreserved}|#{validateUrlPctEncoded}|#{validateUrlSubDelims}|[:|@])", "i"), b.txt.regexen.validateUrlScheme = /(?:[a-z][a-z0-9+\-.]*)/i, b.txt.regexen.validateUrlUserinfo = c("(?:#{validateUrlUnreserved}|#{validateUrlPctEncoded}|#{validateUrlSubDelims}|:)*", "i"), b.txt.regexen.validateUrlDecOctet = /(?:[0-9]|(?:[1-9][0-9])|(?:1[0-9]{2})|(?:2[0-4][0-9])|(?:25[0-5]))/i, b.txt.regexen.validateUrlIpv4 = c(/(?:#{validateUrlDecOctet}(?:\.#{validateUrlDecOctet}){3})/i), b.txt.regexen.validateUrlIpv6 = /(?:\[[a-f0-9:\.]+\])/i, b.txt.regexen.validateUrlIp = c("(?:#{validateUrlIpv4}|#{validateUrlIpv6})", "i"), b.txt.regexen.validateUrlSubDomainSegment = /(?:[a-z0-9](?:[a-z0-9_\-]*[a-z0-9])?)/i, b.txt.regexen.validateUrlDomainSegment = /(?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)/i, b.txt.regexen.validateUrlDomainTld = /(?:[a-z](?:[a-z0-9\-]*[a-z0-9])?)/i, b.txt.regexen.validateUrlDomain = c(/(?:(?:#{validateUrlSubDomainSegment]}\.)*(?:#{validateUrlDomainSegment]}\.)#{validateUrlDomainTld})/i), b.txt.regexen.validateUrlHost = c("(?:#{validateUrlIp}|#{validateUrlDomain})", "i"), b.txt.regexen.validateUrlUnicodeSubDomainSegment = /(?:(?:[a-z0-9]|[^\u0000-\u007f])(?:(?:[a-z0-9_\-]|[^\u0000-\u007f])*(?:[a-z0-9]|[^\u0000-\u007f]))?)/i, b.txt.regexen.validateUrlUnicodeDomainSegment = /(?:(?:[a-z0-9]|[^\u0000-\u007f])(?:(?:[a-z0-9\-]|[^\u0000-\u007f])*(?:[a-z0-9]|[^\u0000-\u007f]))?)/i, b.txt.regexen.validateUrlUnicodeDomainTld = /(?:(?:[a-z]|[^\u0000-\u007f])(?:(?:[a-z0-9\-]|[^\u0000-\u007f])*(?:[a-z0-9]|[^\u0000-\u007f]))?)/i, b.txt.regexen.validateUrlUnicodeDomain = c(/(?:(?:#{validateUrlUnicodeSubDomainSegment}\.)*(?:#{validateUrlUnicodeDomainSegment}\.)#{validateUrlUnicodeDomainTld})/i), b.txt.regexen.validateUrlUnicodeHost = c("(?:#{validateUrlIp}|#{validateUrlUnicodeDomain})", "i"), b.txt.regexen.validateUrlPort = /[0-9]{1,5}/, b.txt.regexen.validateUrlUnicodeAuthority = c("(?:(#{validateUrlUserinfo})@)?(#{validateUrlUnicodeHost})(?::(#{validateUrlPort}))?", "i"), b.txt.regexen.validateUrlAuthority = c("(?:(#{validateUrlUserinfo})@)?(#{validateUrlHost})(?::(#{validateUrlPort}))?", "i"), b.txt.regexen.validateUrlPath = c(/(\/#{validateUrlPchar}*)*/i), b.txt.regexen.validateUrlQuery = c(/(#{validateUrlPchar}|\/|\?)*/i), b.txt.regexen.validateUrlFragment = c(/(#{validateUrlPchar}|\/|\?)*/i), b.txt.regexen.validateUrlUnencoded = c("^(?:([^:/?#]+):\\/\\/)?([^/?#]*)([^?#]*)(?:\\?([^#]*))?(?:#(.*))?$", "i");
        var k = "tweet-url list-slug", l = "tweet-url username", m = "tweet-url hashtag", n = "tweet-url cashtag", o = {
            urlClass: !0,
            listClass: !0,
            usernameClass: !0,
            hashtagClass: !0,
            cashtagClass: !0,
            usernameUrlBase: !0,
            listUrlBase: !0,
            hashtagUrlBase: !0,
            cashtagUrlBase: !0,
            usernameUrlBlock: !0,
            listUrlBlock: !0,
            hashtagUrlBlock: !0,
            linkUrlBlock: !0,
            usernameIncludeSymbol: !0,
            suppressLists: !0,
            suppressNoFollow: !0,
            targetBlank: !0,
            suppressDataScreenName: !0,
            urlEntities: !0,
            symbolTag: !0,
            textWithSymbolTag: !0,
            urlTarget: !0,
            invisibleTagAttrs: !0,
            linkAttributeBlock: !0,
            linkTextBlock: !0,
            htmlEscapeNonEntities: !0
        }, p = {
            disabled: !0,
            readonly: !0,
            multiple: !0,
            checked: !0
        };
        b.txt.tagAttrs = function(a) {
            var c = "";
            for (var d in a) {
                var e = a[d];
                p[d] && (e = e ? d : null);
                if (e == null)
                    continue;
                c += " " + b.txt.htmlEscape(d) + '="' + b.txt.htmlEscape(e.toString()) + '"'
            }
            return c
        }, b.txt.linkToText = function(a, c, e, f) {
            f.suppressNoFollow || (e.rel = "nofollow"), f.linkAttributeBlock && f.linkAttributeBlock(a, e), f.linkTextBlock && (c = f.linkTextBlock(a, c));
            var g = {
                text: c,
                attr: b.txt.tagAttrs(e)
            };
            return d("<a#{attr}>#{text}</a>", g)
        }, b.txt.linkToTextWithSymbol = function(a, c, d, e, f) {
            var g = f.symbolTag ? "<" + f.symbolTag + ">" + c + "</" + f.symbolTag + ">": c;
            d = b.txt.htmlEscape(d);
            var h = f.textWithSymbolTag ? "<" + f.textWithSymbolTag + ">" + d + "</" + f.textWithSymbolTag + ">": d;
            return f.usernameIncludeSymbol ||!c.match(b.txt.regexen.atSigns) ? b.txt.linkToText(a, g + h, e, f) : g + b.txt.linkToText(a, h, e, f)
        }, b.txt.linkToHashtag = function(a, c, d) {
            var e = c.substring(a.indices[0], a.indices[0] + 1), f = b.txt.htmlEscape(a.hashtag), g = q(d.htmlAttrs || {});
            return g.href = d.hashtagUrlBase + f, g.title = "#" + f, g["class"] = d.hashtagClass, f.charAt(0).match(b.txt.regexen.rtl_chars) && (g["class"] += " rtl"), d.targetBlank && (g.target = "_blank"), b.txt.linkToTextWithSymbol(a, e, f, g, d)
        }, b.txt.linkToCashtag = function(a, c, d) {
            var e = b.txt.htmlEscape(a.cashtag), f = q(d.htmlAttrs || {});
            return f.href = d.cashtagUrlBase + e, f.title = "$" + e, f["class"] = d.cashtagClass, d.targetBlank && (f.target = "_blank"), b.txt.linkToTextWithSymbol(a, "$", e, f, d)
        }, b.txt.linkToMentionAndList = function(a, c, d) {
            var e = c.substring(a.indices[0], a.indices[0] + 1), f = b.txt.htmlEscape(a.screenName), g = b.txt.htmlEscape(a.listSlug), h = a.listSlug&&!d.suppressLists, i = q(d.htmlAttrs || {});
            return i["class"] = h ? d.listClass : d.usernameClass, i.href = h ? d.listUrlBase + f + g : d.usernameUrlBase + f, !h&&!d.suppressDataScreenName && (i["data-screen-name"] = f), d.targetBlank && (i.target = "_blank"), b.txt.linkToTextWithSymbol(a, e, h ? f + g : f, i, d)
        }, b.txt.linkToUrl = function(a, c, d) {
            var e = a.url, f = e, g = b.txt.htmlEscape(f), h = d.urlEntities && d.urlEntities[e] || a;
            h.display_url && (g = b.txt.linkTextWithEntity(h, d));
            var i = q(d.htmlAttrs || {});
            return e.match(b.txt.regexen.urlHasProtocol) || (e = "http://" + e), i.href = e, d.targetBlank && (i.target = "_blank"), d.urlClass && (i["class"] = d.urlClass), d.urlTarget && (i.target = d.urlTarget), !d.title && h.display_url && (i.title = h.expanded_url), b.txt.linkToText(a, g, i, d)
        }, b.txt.linkTextWithEntity = function(a, c) {
            var e = a.display_url, f = a.expanded_url, g = e.replace(/…/g, "");
            if (f.indexOf(g)!=-1) {
                var h = f.indexOf(g), i = {
                    displayUrlSansEllipses: g,
                    beforeDisplayUrl: f.substr(0, h),
                    afterDisplayUrl: f.substr(h + g.length),
                    precedingEllipsis: e.match(/^…/) ? "…": "",
                    followingEllipsis: e.match(/…$/) ? "…": ""
                };
                for (var j in i)
                    i.hasOwnProperty(j) && (i[j] = b.txt.htmlEscape(i[j]));
                return i.invisible = c.invisibleTagAttrs, d("<span class='tco-ellipsis'>#{precedingEllipsis}<span #{invisible}>&nbsp;</span></span><span #{invisible}>#{beforeDisplayUrl}</span><span class='js-display-url'>#{displayUrlSansEllipses}</span><span #{invisible}>#{afterDisplayUrl}</span><span class='tco-ellipsis'><span #{invisible}>&nbsp;</span>#{followingEllipsis}</span>", i)
            }
            return e
        }, b.txt.autoLinkEntities = function(a, c, d) {
            d = q(d || {}), d.hashtagClass = d.hashtagClass || m, d.hashtagUrlBase = d.hashtagUrlBase || "https://twitter.com/#!/search?q=%23", d.cashtagClass = d.cashtagClass || n, d.cashtagUrlBase = d.cashtagUrlBase || "https://twitter.com/#!/search?q=%24", d.listClass = d.listClass || k, d.usernameClass = d.usernameClass || l, d.usernameUrlBase = d.usernameUrlBase || "https://twitter.com/", d.listUrlBase = d.listUrlBase || "https://twitter.com/", d.htmlAttrs = b.txt.extractHtmlAttrsFromOptions(d), d.invisibleTagAttrs = d.invisibleTagAttrs || "style='position:absolute;left:-9999px;'";
            var e, f, g;
            if (d.urlEntities) {
                e = {};
                for (f = 0, g = d.urlEntities.length; f < g; f++)
                    e[d.urlEntities[f].url] = d.urlEntities[f];
                d.urlEntities = e
            }
            var h = "", i = 0;
            c.sort(function(a, b) {
                return a.indices[0] - b.indices[0]
            });
            var j = d.htmlEscapeNonEntities ? b.txt.htmlEscape: function(a) {
                return a
            };
            for (var f = 0; f < c.length; f++) {
                var o = c[f];
                h += j(a.substring(i, o.indices[0])), o.url ? h += b.txt.linkToUrl(o, a, d) : o.hashtag ? h += b.txt.linkToHashtag(o, a, d) : o.screenName ? h += b.txt.linkToMentionAndList(o, a, d) : o.cashtag && (h += b.txt.linkToCashtag(o, a, d)), i = o.indices[1]
            }
            return h += j(a.substring(i, a.length)), h
        }, b.txt.autoLinkWithJSON = function(a, c, d) {
            if (c.user_mentions)
                for (var e = 0; e < c.user_mentions.length; e++)
                    c.user_mentions[e].screenName = c.user_mentions[e].screen_name;
            if (c.hashtags)
                for (var e = 0; e < c.hashtags.length; e++)
                    c.hashtags[e].hashtag = c.hashtags[e].text;
            if (c.symbols)
                for (var e = 0; e < c.symbols.length; e++)
                    c.symbols[e].cashtag = c.symbols[e].text;
            var f = [];
            for (var g in c)
                f = f.concat(c[g]);
            return b.txt.modifyIndicesFromUnicodeToUTF16(a, f), b.txt.autoLinkEntities(a, f, d)
        }, b.txt.extractHtmlAttrsFromOptions = function(a) {
            var b = {};
            for (var c in a) {
                var d = a[c];
                if (o[c])
                    continue;
                p[c] && (d = d ? c : null);
                if (d == null)
                    continue;
                b[c] = d
            }
            return b
        }, b.txt.autoLink = function(a, c) {
            var d = b.txt.extractEntitiesWithIndices(a, {
                extractUrlsWithoutProtocol: !1
            });
            return b.txt.autoLinkEntities(a, d, c)
        }, b.txt.autoLinkUsernamesOrLists = function(a, c) {
            var d = b.txt.extractMentionsOrListsWithIndices(a);
            return b.txt.autoLinkEntities(a, d, c)
        }, b.txt.autoLinkHashtags = function(a, c) {
            var d = b.txt.extractHashtagsWithIndices(a);
            return b.txt.autoLinkEntities(a, d, c)
        }, b.txt.autoLinkCashtags = function(a, c) {
            var d = b.txt.extractCashtagsWithIndices(a);
            return b.txt.autoLinkEntities(a, d, c)
        }, b.txt.autoLinkUrlsCustom = function(a, c) {
            var d = b.txt.extractUrlsWithIndices(a, {
                extractUrlsWithoutProtocol: !1
            });
            return b.txt.autoLinkEntities(a, d, c)
        }, b.txt.removeOverlappingEntities = function(a) {
            a.sort(function(a, b) {
                return a.indices[0] - b.indices[0]
            });
            var b = a[0];
            for (var c = 1; c < a.length; c++)
                b.indices[1] > a[c].indices[0] ? (a.splice(c, 1), c--) : b = a[c]
        }, b.txt.extractEntitiesWithIndices = function(a, c) {
            var d = b.txt.extractUrlsWithIndices(a, c).concat(b.txt.extractMentionsOrListsWithIndices(a)).concat(b.txt.extractHashtagsWithIndices(a, {
                checkUrlOverlap: !1
            })).concat(b.txt.extractCashtagsWithIndices(a));
            return d.length == 0 ? [] : (b.txt.removeOverlappingEntities(d), d)
        }, b.txt.extractMentions = function(a) {
            var c = [], d = b.txt.extractMentionsWithIndices(a);
            for (var e = 0; e < d.length; e++) {
                var f = d[e].screenName;
                c.push(f)
            }
            return c
        }, b.txt.extractMentionsWithIndices = function(a) {
            var c = [], d, e = b.txt.extractMentionsOrListsWithIndices(a);
            for (var f = 0; f < e.length; f++)
                d = e[f], d.listSlug == "" && c.push({
                    screenName: d.screenName,
                    indices: d.indices
                });
            return c
        }, b.txt.extractMentionsOrListsWithIndices = function(a) {
            if (!a ||!a.match(b.txt.regexen.atSigns))
                return [];
            var c = [], d;
            return a.replace(b.txt.regexen.validMentionOrList, function(a, d, e, f, g, h, i) {
                var j = i.slice(h + a.length);
                if (!j.match(b.txt.regexen.endMentionMatch)) {
                    g = g || "";
                    var k = h + d.length, l = k + f.length + g.length + 1;
                    c.push({
                        screenName: f,
                        listSlug: g,
                        indices: [k, l]
                    })
                }
            }), c
        }, b.txt.extractReplies = function(a) {
            if (!a)
                return null;
            var c = a.match(b.txt.regexen.validReply);
            return !c || RegExp.rightContext.match(b.txt.regexen.endMentionMatch) ? null : c[1]
        }, b.txt.extractUrls = function(a, c) {
            var d = [], e = b.txt.extractUrlsWithIndices(a, c);
            for (var f = 0; f < e.length; f++)
                d.push(e[f].url);
            return d
        }, b.txt.extractUrlsWithIndices = function(a, c) {
            c || (c = {
                extractUrlsWithoutProtocol: !0
            });
            if (!a || (c.extractUrlsWithoutProtocol?!a.match(/\./) : !a.match(/:/))
                )return [];
            var d = [];
            while (b.txt.regexen.extractUrl.exec(a)) {
                var e = RegExp.$2, f = RegExp.$3, g = RegExp.$4, h = RegExp.$5, i = RegExp.$7, j = b.txt.regexen.extractUrl.lastIndex, k = j - f.length;
                if (!g) {
                    if (!c.extractUrlsWithoutProtocol || e.match(b.txt.regexen.invalidUrlWithoutProtocolPrecedingChars))
                        continue;
                    var l = null, m = 0;
                    h.replace(b.txt.regexen.validAsciiDomain, function(a) {
                        var c = h.indexOf(a, m);
                        m = c + a.length, l = {
                            url: a,
                            indices: [k + c, k + m]
                        }, (i || a.match(b.txt.regexen.validSpecialShortDomain) ||!a.match(b.txt.regexen.invalidShortDomain)) && d.push(l)
                    });
                    if (l == null)
                        continue;
                    i && (l.url = f.replace(h, l.url), l.indices[1] = j)
                } else 
                    f.match(b.txt.regexen.validTcoUrl) && (f = RegExp.lastMatch, j = k + f.length), d.push({
                        url: f,
                        indices: [k, j]
                    })
            }
            return d
        }, b.txt.extractHashtags = function(a) {
            var c = [], d = b.txt.extractHashtagsWithIndices(a);
            for (var e = 0; e < d.length; e++)
                c.push(d[e].hashtag);
            return c
        }, b.txt.extractHashtagsWithIndices = function(a, c) {
            c || (c = {
                checkUrlOverlap: !0
            });
            if (!a ||!a.match(b.txt.regexen.hashSigns))
                return [];
            var d = [];
            a.replace(b.txt.regexen.validHashtag, function(a, c, e, f, g, h) {
                var i = h.slice(g + a.length);
                if (i.match(b.txt.regexen.endHashtagMatch))
                    return;
                var j = g + c.length, k = j + f.length + 1;
                d.push({
                    hashtag: f,
                    indices: [j, k]
                })
            });
            if (c.checkUrlOverlap) {
                var e = b.txt.extractUrlsWithIndices(a);
                if (e.length > 0) {
                    var f = d.concat(e);
                    b.txt.removeOverlappingEntities(f), d = [];
                    for (var g = 0; g < f.length; g++)
                        f[g].hashtag && d.push(f[g])
                    }
            }
            return d
        }, b.txt.extractCashtags = function(a) {
            var c = [], d = b.txt.extractCashtagsWithIndices(a);
            for (var e = 0; e < d.length; e++)
                c.push(d[e].cashtag);
            return c
        }, b.txt.extractCashtagsWithIndices = function(a) {
            if (!a || a.indexOf("$")==-1)
                return [];
            var c = [];
            return a.replace(b.txt.regexen.validCashtag, function(a, b, d, e, f, g) {
                var h = f + b.length, i = h + e.length + 1;
                c.push({
                    cashtag: e,
                    indices: [h, i]
                })
            }), c
        }, b.txt.modifyIndicesFromUnicodeToUTF16 = function(a, c) {
            b.txt.convertUnicodeIndices(a, c, !1)
        }, b.txt.modifyIndicesFromUTF16ToUnicode = function(a, c) {
            b.txt.convertUnicodeIndices(a, c, !0)
        }, b.txt.getUnicodeTextLength = function(a) {
            return a.replace(b.txt.regexen.non_bmp_code_pairs, " ").length
        }, b.txt.convertUnicodeIndices = function(a, b, c) {
            if (b.length == 0)
                return;
            var d = 0, e = 0;
            b.sort(function(a, b) {
                return a.indices[0] - b.indices[0]
            });
            var f = 0, g = b[0];
            while (d < a.length) {
                if (g.indices[0] == (c ? d : e)) {
                    var h = g.indices[1] - g.indices[0];
                    g.indices[0] = c ? e : d, g.indices[1] = g.indices[0] + h, f++;
                    if (f == b.length)
                        break;
                    g = b[f]
                }
                var i = a.charCodeAt(d);
                55296 <= i && i <= 56319 && d < a.length - 1 && (i = a.charCodeAt(d + 1), 56320 <= i && i <= 57343 && d++), e++, d++
            }
        }, b.txt.splitTags = function(a) {
            var b = a.split("<"), c, d = [], e;
            for (var f = 0; f < b.length; f += 1) {
                e = b[f];
                if (!e)
                    d.push("");
                else {
                    c = e.split(">");
                    for (var g = 0; g < c.length; g += 1)
                        d.push(c[g])
                    }
            }
            return d
        }, b.txt.hitHighlight = function(a, c, d) {
            var e = "em";
            c = c || [], d = d || {};
            if (c.length === 0)
                return a;
            var f = d.tag || e, g = ["<" + f + ">", "</" + f + ">"], h = b.txt.splitTags(a), i, j, k = "", l = 0, m = h[0], n = 0, o = 0, p=!1, q = m, r = [], s, t, u, v, w;
            for (i = 0; i < c.length; i += 1)
                for (j = 0; j < c[i].length; j += 1)
                    r.push(c[i][j]);
            for (s = 0; s < r.length; s += 1) {
                t = r[s], u = g[s%2], v=!1;
                while (m != null && t >= n + m.length)
                    k += q.slice(o), p && t === n + q.length && (k += u, v=!0), h[l + 1] && (k += "<" + h[l + 1] + ">"), n += q.length, o = 0, l += 2, m = h[l], q = m, p=!1;
                !v && m != null ? (w = t - n, k += q.slice(o, w) + u, o = w, s%2 === 0 ? p=!0 : p=!1) : v || (v=!0, k += u)
            }
            if (m != null) {
                o < q.length && (k += q.slice(o));
                for (s = l + 1; s < h.length; s += 1)
                    k += s%2 === 0 ? h[s] : "<" + h[s] + ">"
            }
            return k
        };
        var r = 140, s = [f(65534), f(65279), f(65535), f(8234), f(8235), f(8236), f(8237), f(8238)];
        b.txt.getTweetLength = function(a, c) {
            c || (c = {
                short_url_length: 22,
                short_url_length_https: 23
            });
            var d = b.txt.getUnicodeTextLength(a), e = b.txt.extractUrlsWithIndices(a);
            b.txt.modifyIndicesFromUTF16ToUnicode(a, e);
            for (var f = 0; f < e.length; f++)
                d += e[f].indices[0] - e[f].indices[1], e[f].url.toLowerCase().match(b.txt.regexen.urlHasHttps) ? d += c.short_url_length_https : d += c.short_url_length;
            return d
        }, b.txt.isInvalidTweet = function(a) {
            if (!a)
                return "empty";
            if (b.txt.getTweetLength(a) > r)
                return "too_long";
            for (var c = 0; c < s.length; c++)
                if (a.indexOf(s[c]) >= 0)
                    return "invalid_characters";
            return !1
        }, b.txt.isValidTweetText = function(a) {
            return !b.txt.isInvalidTweet(a)
        }, b.txt.isValidUsername = function(a) {
            if (!a)
                return !1;
            var c = b.txt.extractMentions(a);
            return c.length === 1 && c[0] === a.slice(1)
        };
        var t = c(/^#{validMentionOrList}$/);
        b.txt.isValidList = function(a) {
            var b = a.match(t);
            return !!b && b[1] == ""&&!!b[4]
        }, b.txt.isValidHashtag = function(a) {
            if (!a)
                return !1;
            var c = b.txt.extractHashtags(a);
            return c.length === 1 && c[0] === a.slice(1)
        }, b.txt.isValidUrl = function(a, c, d) {
            c == null && (c=!0), d == null && (d=!0);
            if (!a)
                return !1;
            var e = a.match(b.txt.regexen.validateUrlUnencoded);
            if (!e || e[0] !== a)
                return !1;
            var f = e[1], g = e[2], h = e[3], i = e[4], j = e[5];
            return (!d || u(f, b.txt.regexen.validateUrlScheme) && f.match(/^https?$/i)) && u(h, b.txt.regexen.validateUrlPath) && u(i, b.txt.regexen.validateUrlQuery, !0) && u(j, b.txt.regexen.validateUrlFragment, !0) ? c && u(g, b.txt.regexen.validateUrlUnicodeAuthority) ||!c && u(g, b.txt.regexen.validateUrlAuthority) : !1
        }, typeof module != "undefined" && module.exports && (module.exports = b.txt)
    })(), a(b.txt)
})
define("app/ui/with_character_counter", ["module", "require", "exports", "lib/twitter-text"], function(module, require, exports) {
    function withCharacterCounter() {
        var a = 23, b = 24;
        this.defaultAttrs({
            counterSelector: ".tweet-counter",
            maxLength: 140,
            superwarnLength: 130,
            warnLength: 120,
            superwarnClass: "superwarn",
            warnClass: "warn",
            maxReachedClass: "max-reached",
            dmOnly: !1
        }), this.updateCounter = function() {
            var a = this.getLength(), b = a >= this.attr.warnLength && a < this.attr.superwarnLength, c = a >= this.attr.superwarnLength, d = this.maxReached(), e = this.attr.maxLength - a;
            this.$counter.html(e).toggleClass(this.attr.warnClass, b).toggleClass(this.attr.superwarnClass, c).toggleClass(this.attr.maxReachedClass, d), (b || c) && this.trigger("uiCharCountWarningVisible", {
                charCount: e
            })
        }, this.getLength = function(a) {
            return a = this.addHiddenText(a === undefined ? this.getVisibleText() : a), a !== this.prevCounterText && (this.prevCounterText = a, this.prevCounterLength = twitterText.getTweetLength(a)), this.prevCounterLength + this.getAttachmentLength()
        }, this.getAttachmentLength = function() {
            return this.hasAttachment && this.hasAttachment() ? this.attr.dmOnly ? b : a : 0
        }, this.maxReached = function() {
            return this.getLength() > this.attr.maxLength
        }, this.after("initialize", function() {
            this.$counter = this.select("counterSelector"), this.on("uiTextChanged uiAttachmentChanged", this.updateCounter), this.updateCounter()
        })
    }
    var twitterText = require("lib/twitter-text");
    module.exports = withCharacterCounter
});
define("app/ui/with_text_polling", ["module", "require", "exports"], function(module, require, exports) {
    function withTextPolling() {
        this.defaultAttrs({
            pollIntervalInMs: 100
        }), this.pollUpdatedText = function() {
            this.detectUpdatedText(), this.hasFocus() || this.stopPollingUpdatedText()
        }, this.startPollingUpdatedText = function() {
            this.detectUpdatedText(), this.pollUpdatedTextId === undefined && (this.pollUpdatedTextId = setInterval(this.pollUpdatedText.bind(this), this.attr.pollIntervalInMs))
        }, this.stopPollingUpdatedText = function() {
            this.pollUpdatedTextId !== undefined && (clearInterval(this.pollUpdatedTextId), delete this.pollUpdatedTextId)
        }, this.after("initialize", function() {
            this.on(this.$text, "focus", this.startPollingUpdatedText), this.hasFocus() && this.startPollingUpdatedText()
        }), this.before("teardown", function() {
            this.stopPollingUpdatedText()
        })
    }
    module.exports = withTextPolling
});
define("app/ui/updating_text_counter", ["module", "require", "exports", "core/component", "app/ui/with_text_editor", "app/ui/with_character_counter", "app/ui/with_text_polling"], function(module, require, exports) {
    function updatingTextCounter() {
        this.after("initialize", function() {
            this.initTextNode()
        })
    }
    var defineComponent = require("core/component"), withTextEditor = require("app/ui/with_text_editor"), withCharacterCounter = require("app/ui/with_character_counter"), withTextPolling = require("app/ui/with_text_polling"), UpdatingTextCounter = defineComponent(updatingTextCounter, withTextEditor, withCharacterCounter, withTextPolling);
    module.exports = UpdatingTextCounter
});
define("app/ui/dialogs/with_create_or_edit_custom_timeline_dialog", ["module", "require", "exports"], function(module, require, exports) {
    module.exports = function() {
        var a = /^\s*$/;
        this.defaultAttrs({
            top: 90,
            saveSelector: ".update-custom-timeline-button",
            nameSelector: "[name=name]",
            descriptionSelector: "[name=custom-timeline-description]",
            dialogTitleSelector: ".modal-title",
            maxReachedSelector: ".max-reached"
        }), this.params = function() {
            return {
                name: this.$name.val(),
                description: this.$description.val(),
                order: this.select("orderSelector").val()
            }
        }, this.saveCustomTimeline = function(a, b) {
            this.disableSaveButton(), this.trigger(this.attr.createOrUpdateEventName, this.params())
        }, this.disableSaveButton = function() {
            this.$saveButton.attr("disabled", !0)
        }, this.toggleSaveButton = function(b, c) {
            var d = a.test(this.$name.val()), e = this.select("maxReachedSelector").length > 0;
            this.$saveButton.attr("disabled", d || e)
        }, this.after("open", function() {
            this.$saveButton = this.select("saveSelector"), this.$name = this.select("nameSelector"), this.$description = this.select("descriptionSelector"), this.toggleSaveButton()
        }), this.after("initialize", function() {
            this.on(document, "dataCustomTimelineUpdated", this.close), this.on("click", {
                saveSelector: this.saveCustomTimeline
            }), this.on("input", {
                nameSelector: this.toggleSaveButton
            }), this.on("uiTextChanged", this.toggleSaveButton)
        })
    }
});
define("app/ui/dialogs/create_custom_timeline_dialog", ["module", "require", "exports", "core/component", "app/ui/updating_text_counter", "app/ui/with_dialog", "app/ui/dialogs/with_create_or_edit_custom_timeline_dialog", "template"], function(module, require, exports) {
    function createCustomTimelineDialog() {
        this.defaultAttrs({
            createOrUpdateEventName: "uiCreateCustomTimeline",
            orderSelector: "[name=create-timeline-order]:checked"
        }), this.around("params", function(a) {
            var b = a();
            return b.tweetId = this.tweetId, b.impressionId = this.impressionId, b
        }), this.openDialog = function(a, b) {
            this.tweetId = b.tweetId, this.impressionId = b.impressionId;
            var c = {
                is_create_dialog: !0,
                id_prefix: "create",
                curation_reverse_chron: !0
            }, d = template["dialogs/create_or_edit_custom_timeline_dialog"].render(c, template);
            this.$node.html(d), this.$dialog = this.$dialogContainer.find(this.attr.modalSelector), this.open()
        }, this.after("open", function() {
            UpdatingTextCounter.attachTo(".custom-timeline-name-field", {
                maxLength: 25,
                superwarnLength: 20,
                warnLength: 15,
                textSelector: "#custom-timeline-name",
                counterSelector: ".custom-timeline-name-count"
            }), UpdatingTextCounter.attachTo(".custom-timeline-description-field", {
                maxLength: 160,
                superwarnLength: 150,
                warnLength: 140,
                textSelector: "#custom-timeline-description",
                counterSelector: ".custom-timeline-description-count"
            })
        }), this.after("initialize", function() {
            this.on(document, "uiOpenCreateCustomTimelineDialog", this.openDialog)
        })
    }
    var defineComponent = require("core/component"), UpdatingTextCounter = require("app/ui/updating_text_counter"), withDialog = require("app/ui/with_dialog"), withCreateOrEditCustomTimelineDialog = require("app/ui/dialogs/with_create_or_edit_custom_timeline_dialog"), template = require("template");
    module.exports = defineComponent(withDialog, withCreateOrEditCustomTimelineDialog, createCustomTimelineDialog)
});
define("app/ui/with_custom_timeline_create_button", ["module", "require", "exports"], function(module, require, exports) {
    module.exports = function() {
        this.defaultAttrs({
            createCustomTimelineSelector: ".js-create-custom-timeline-button"
        }), this.openCreateDialog = function() {
            this.trigger("uiOpenCreateCustomTimelineDialog", {
                tweetId: this.tweetId,
                impressionId: this.impressionId
            })
        }, this.after("initialize", function() {
            this.on("click", {
                createCustomTimelineSelector: this.openCreateDialog
            })
        })
    }
});
define("app/ui/dialogs/curate_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/ui/with_custom_timeline_create_button", "app/data/with_scribe", "core/i18n", "template"], function(module, require, exports) {
    function curateDialog() {
        this.defaultAttrs({
            top: 90,
            closeOnOtherDialogOpened: !0,
            timelineSelectorSelector: ".timeline-selector",
            wasMemberSelector: ".timeline-item .was-member",
            wasNotMemberSelector: ".timeline-item .was-not-member",
            submitSelector: ".js-submit"
        }), this.openDialog = function(a, b) {
            var c = template["dialogs/curate_dialog"].render({}, template);
            this.$node.html(c), this.$dialog = this.$dialogContainer.find(this.attr.modalSelector), this.tweetId = b.tweetId, this.impressionId = b.impressionId, this.trigger("uiNeedsFullCustomTimelines", {
                tweetId: b.tweetId
            }), this.scribe({
                component: "curate_dialog",
                action: "open"
            }, b), this.open()
        }, this.renderFullTimelinesList = function(a, b) {
            this.select("timelineSelectorSelector").html(b.html)
        }, this.submitChanges = function() {
            function a(a) {
                return a.map(function() {
                    return $(this).attr("data-timeline-id")
                }).get()
            }
            this.select("submitSelector").attr("disabled", !0);
            var b = this.select("wasMemberSelector").filter(":not(:checked)"), c = this.select("wasNotMemberSelector").filter(":checked"), d = {
                removed: a(b),
                added: a(c),
                tweetId: this.tweetId,
                impressionId: this.impressionId
            };
            this.trigger("uiCurateTweets", d)
        }, this.after("initialize", function() {
            this.on(document, "uiOpenCurateDialog", this.openDialog), this.on(document, "dataGotFullCustomTimelines", this.renderFullTimelinesList), this.on(document, "dataTweetsCurated", this.close), this.on("click", {
                submitSelector: this.submitChanges
            })
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withCustomTimelineCreateButton = require("app/ui/with_custom_timeline_create_button"), withScribe = require("app/data/with_scribe"), _ = require("core/i18n"), template = require("template");
    module.exports = defineComponent(withDialog, curateDialog, withCustomTimelineCreateButton, withScribe)
});
define("app/ui/dashboard", ["module", "require", "exports", "core/component", "core/utils", "core/clock"], function(module, require, exports) {
    function dashboard() {
        this.defaultAttrs({
            fixedClass: "fixed-dashboard",
            padding: 5
        }), this.fixDashboard = function() {
            var a = this.$node;
            this.setWtfModule(a);
            var b = a.height(), c = $(document).scrollTop(), d = this.fixWholeDashboard(a, this.windowHeight, b, c);
            if (!d) {
                var e = this.$wtfModule.position().top;
                this.fixPartialDashboard(a, this.windowHeight, b, c, e)
            }
        }, this.fixWholeDashboard = function(a, b, c, d) {
            var e = this.initialDashboardTop + c;
            return e <= b ? (this.fixDashboardElement(a, ""), !0) : (d < 1 && this.unfixDashboard(a), !1)
        }, this.fixPartialDashboard = function(a, b, c, d, e) {
            if (d > e && this.fixedDashboard)
                return;
            var f = this.topBarHeight + d + this.attr.padding, g = (e - this.topBarHeight - this.attr.padding)*-1 + this.initialDashboardTop, h = this.initialDashboardTop + c + g, i = h <= b, j = f >= e;
            j && i ? this.fixedDashboard || this.fixDashboardElement(a, g) : this.unfixDashboard(a)
        }, this.setWtfModule = function(a) {
            this.$wtfModule = this.$wtfModule || a.find(".js-wtf-module")
        }, this.setWindowHeight = function() {
            this.windowHeight = $(window).height()
        }, this.setTopBarHeight = function() {
            var a = $(".js-topbar");
            this.topBarHeight = a.height()
        }, this.setWindowHeightAndFixDashboard = function() {
            this.setWindowHeight(), this.fixDashboard()
        }, this.fixDashboardElement = function(a, b) {
            a.addClass(this.attr.fixedClass).css("top", b), this.fixedDashboard=!0
        }, this.unfixDashboard = function(a) {
            a.removeClass(this.attr.fixedClass).css("top", ""), this.fixedDashboard=!1
        }, this.after("initialize", function(a, b) {
            b.freezeDashboard && (this.fixedDashboard=!1, this.initialDashboardTop = $(a).position().top, this.setWindowHeight(), this.setTopBarHeight(), this.fixDashboard(), clock.setTimeoutEvent("uiDashboardShouldCheckHeight", 1e3), this.on(document, "uiDashboardShouldCheckHeight", this.fixDashboard), this.on(window, "resize", utils.debounce(this.setWindowHeightAndFixDashboard.bind(this), 100)), this.on(window, "scroll", utils.debounce(this.fixDashboard.bind(this), 15)))
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), clock = require("core/clock");
    module.exports = defineComponent(dashboard)
});
define("app/ui/dialogs/delete_tweet_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/ui/dialogs/with_modal_tweet"], function(module, require, exports) {
    function deleteTweetDialog() {
        this.defaultAttrs({
            cancelSelector: ".cancel-action",
            deleteSelector: ".delete-action"
        }), this.openDeleteTweet = function(a, b) {
            this.attr.sourceEventData = b, this.displayTweet(b.tweetId, {
                modal: "delete"
            }), this.id = b.id, this.open()
        }, this.deleteTweet = function() {
            this.trigger("uiDidDeleteTweet", {
                id: this.id,
                sourceEventData: this.attr.sourceEventData
            })
        }, this.deleteTweetSuccess = function(a, b) {
            this.trigger("uiDidDeleteTweetSuccess", this.attr.sourceEventData), this.close()
        }, this.restoreFocusToTweet = function(a) {
            $(a.target).is(this.$dialog) && this.activeEl && this.trigger($(this.activeEl).closest(".tweet"), "uiShouldAddFocusStyle")
        }, this.after("initialize", function() {
            this.on("click", {
                cancelSelector: this.close,
                deleteSelector: this.deleteTweet
            }), this.on(document, "uiOpenDeleteDialog", this.openDeleteTweet), this.on(document, "dataDidDeleteTweet", this.deleteTweetSuccess), this.on(document, "uiDialogRestorePreviousFocus", this.restoreFocusToTweet), this.on(document, "uiCloseDeleteTweetDialog", this.close)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withModalTweet = require("app/ui/dialogs/with_modal_tweet"), DeleteTweetDialog = defineComponent(deleteTweetDialog, withDialog, withModalTweet);
    module.exports = DeleteTweetDialog
});
define("app/data/sms/device_verification", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function deviceVerification() {
        this.submitDeviceDownload = function(a, b) {
            this.post({
                url: "/i/devices/download",
                data: {
                    country_code: b.countryCode,
                    phone_number: b.deviceNumber
                },
                success: "dataDidSubmitDownloadLinkRequest",
                error: "dataFailedToSubmitDownloadLinkRequest"
            })
        }, this.resendDeviceDownload = function(a, b) {
            this.post({
                url: "/i/devices/download",
                data: {
                    device_id: b.deviceId,
                    resend: !0
                },
                success: "dataDidResendDownloadLinkRequest",
                error: "dataFailedToResendDownloadLinkRequest"
            })
        }, this.submitDeviceVerification = function(a, b) {
            var c = function(a) {
                a.success ? this.trigger("dataDidVerifyDevice", a) : this.trigger("dataFailedToVerifyDevice", a)
            };
            this.post({
                url: "/i/devices/verify",
                data: {
                    device_id: b.deviceId,
                    pin: b.pin
                },
                success: c.bind(this),
                error: "dataFailedToVerifyDevice"
            })
        }, this.after("initialize", function() {
            this.on(document, "uiDidSubmitDownloadLinkRequest", this.submitDeviceDownload), this.on(document, "uiDidResendDownloadLinkRequest", this.resendDeviceDownload), this.on(document, "uiDidSubmitDeviceVerification", this.submitDeviceVerification)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), DeviceVerification = defineComponent(deviceVerification, withData);
    module.exports = DeviceVerification
});
define("app/ui/sms/with_sms_country_selector", ["module", "require", "exports"], function(module, require, exports) {
    function withSmsCountrySelector() {
        this.defaultAttrs({
            smsCountryCodeSelector: "#sms-country-code",
            smsCountryCodePreviewSelector: "#sms-country-code-preview"
        }), this.updateSmsCountryCode = function() {
            this.$smsCountryCodePreview.text("+" + this.$smsCountryCode.val())
        }, this.after("initialize", function() {
            this.$smsCountryCode = this.select("smsCountryCodeSelector"), this.$smsCountryCodePreview = this.select("smsCountryCodePreviewSelector"), this.on(this.$smsCountryCode, "change", this.updateSmsCountryCode)
        })
    }
    module.exports = withSmsCountrySelector
});
define("app/ui/with_loading_indicator", ["module", "require", "exports"], function(module, require, exports) {
    function withLoadingIndicator() {
        this.defaultAttrs({
            spinnerContainer: "body",
            spinnerClass: "pushing-state"
        }), this.showSpinner = function(a, b) {
            this.select("spinnerContainer").addClass(this.attr.spinnerClass)
        }, this.hideSpinner = function(a, b) {
            this.select("spinnerContainer").removeClass(this.attr.spinnerClass)
        }
    }
    module.exports = withLoadingIndicator
});
define("app/ui/sms/device_verification_banner", ["module", "require", "exports", "core/component", "core/i18n", "app/ui/sms/with_sms_country_selector", "app/ui/with_loading_indicator"], function(module, require, exports) {
    function deviceVerificationBanner() {
        var a = _('Your device has been verified!'), b = _('The PIN you entered was not valid. Please try again.'), c = _('A new PIN was sent to your device.'), d = _('There was an error resending your PIN.');
        this.defaultAttrs({
            bannerSelector: "#device-verification-banner",
            idSelector: "#device-verification-id",
            pinSelector: "#device-verification-pin",
            submitSelector: "#device-verification-submit",
            resendSelector: "#device-verification-resend",
            formElementsSelector: "#device-verification-pin, #device-verification-submit",
            replaceBannerHTML: !1
        }), this.deviceId = function() {
            return this.select("idSelector").val()
        }, this.pin = function() {
            return this.select("pinSelector").val()
        }, this.dataStart = function() {
            this.select("formElementsSelector").attr("disabled", !0), this.trigger("uiShowGlobalLoadingIndicator")
        }, this.dataEnd = function() {
            this.select("formElementsSelector").attr("disabled", !1), this.trigger("uiHideGlobalLoadingIndicator")
        }, this.showVerificationBanner = function(a, b) {
            this.attr.replaceBannerHTML ? this.$node.html(b.banner_html) : this.$node.prepend($(b.banner_html))
        }, this.updateSubmit = function(a, b) {
            this.select("submitSelector").attr("disabled", this.select("pinSelector").val() == "")
        }, this.submitVerification = function() {
            this.dataStart(), this.trigger("uiDidSubmitDeviceVerification", {
                deviceId: this.deviceId(),
                pin: this.pin()
            })
        }, this.deviceVerificationSuccess = function(b, c) {
            this.dataEnd(), this.trigger("uiShowMessage", {
                message: a
            }), this.select("bannerSelector").hide()
        }, this.deviceVerificationFailure = function(a, c) {
            this.dataEnd(), this.trigger("uiShowMessage", {
                message: b
            })
        }, this.resendVerification = function() {
            this.dataStart(), this.trigger("uiDidResendDownloadLinkRequest", {
                deviceId: this.deviceId()
            })
        }, this.resendSuccess = function() {
            this.dataEnd(), this.trigger("uiShowMessage", {
                message: c
            })
        }, this.resendFailure = function() {
            this.dataEnd(), this.trigger("uiShowMessage", {
                message: d
            })
        }, this.after("initialize", function() {
            this.on(document, "dataDidSubmitDownloadLinkRequest", this.showVerificationBanner), this.on(document, "dataDidVerifyDevice", this.deviceVerificationSuccess), this.on(document, "dataFailedToVerifyDevice", this.deviceVerificationFailure), this.on(document, "dataDidResendDownloadLinkRequest", this.resendSuccess), this.on(document, "dataFailedToResendDownloadLinkRequest", this.resendFailure), this.on("click", {
                submitSelector: this.submitVerification,
                resendSelector: this.resendVerification
            }), this.on("keyup", {
                pinSelector: this.updateSubmit
            })
        })
    }
    var defineComponent = require("core/component"), _ = require("core/i18n"), withSmsCountrySelector = require("app/ui/sms/with_sms_country_selector"), withLoadingIndicator = require("app/ui/with_loading_indicator"), DeviceVerificationBanner = defineComponent(deviceVerificationBanner);
    module.exports = DeviceVerificationBanner
});
define("app/ui/direct_message_conversation_actions", ["module", "require", "exports", "core/component", "core/i18n"], function(module, require, exports) {
    function conversationActions() {
        this.defaultAttrs({
            deleteSelector: ".js-actionDeleteConversation",
            viewParticipantsSelector: ".js-actionViewParticipants",
            conversationIdSelector: "[data-conversation-id]",
            notificationsEnableTextSelector: ".notifications-enable-text",
            notificationsDisableTextSelector: ".notifications-disable-text"
        }), this.getConversationId = function() {
            return this.select("conversationIdSelector").attr("data-conversation-id")
        }, this.confirmDelete = function(a) {
            this.trigger("uiOpenConfirmDialog", {
                action: "DeleteConversation",
                titleText: _('Delete Conversation'),
                bodyText: _('Are you sure you want to delete this conversation?'),
                cancelText: _('Cancel'),
                submitText: _('Delete')
            })
        }, this.deleteConversation = function(a) {
            this.trigger("uiDeleteConversation", {
                conversation_id: this.getConversationId()
            })
        }, this.viewParticipants = function(a) {
            this.trigger("uiViewParticipants", {
                conversation_id: this.getConversationId()
            })
        }, this.enableNotifications = function() {
            this.select("notificationsEnableTextSelector").addClass("u-hidden"), this.select("notificationsDisableTextSelector").removeClass("u-hidden")
        }, this.disableNotifications = function() {
            this.select("notificationsDisableTextSelector").addClass("u-hidden"), this.select("notificationsEnableTextSelector").removeClass("u-hidden")
        }, this.updateNotificationsState = function(a, b) {
            switch (a.type) {
            case"dataDMConversationResult":
                b.conversation_actions && this.getConversationId() == b.conversation_actions.conversation_id && (b.conversation_actions.notifications ? this.enableNotifications() : this.disableNotifications());
                break;
            case"dataNotificationsEnabled":
                this.getConversationId() == b.sourceEventData.conversation_id && this.enableNotifications();
                break;
            case"dataNotificationsDisabled":
                this.getConversationId() == b.sourceEventData.conversation_id && this.disableNotifications()
            }
        }, this.handleNotificationsChangeFailure = function(a, b) {
            this.getConversationId() == b.sourceEventData.conversation_id && (a.type == "dataNotificationsEnableFailed" ? this.disableNotifications() : this.enableNotifications())
        }, this.toggleNotifications = function(a, b) {
            var c = this.select("notificationsEnableTextSelector"), d=!c.hasClass("u-hidden");
            d ? this.enableNotifications() : this.disableNotifications(), this.trigger("uiCloseDropdowns"), this.trigger("uiToggleNotifications", {
                conversation_id: this.getConversationId(),
                enable_notifications: d
            })
        }, this.after("initialize", function() {
            this.on("uiDeleteConversationConfirm", this.deleteConversation), this.on("click", {
                deleteSelector: this.confirmDelete,
                viewParticipantsSelector: this.viewParticipants,
                notificationsEnableTextSelector: this.toggleNotifications,
                notificationsDisableTextSelector: this.toggleNotifications
            }), this.on(document, "dataDMConversationResult dataNotificationsEnabled dataNotificationsDisabled", this.updateNotificationsState), this.on(document, "dataNotificationsEnableFailed dataNotificationsDisableFailed", this.handleNotificationsChangeFailure)
        })
    }
    var defineComponent = require("core/component"), _ = require("core/i18n"), ConversationActions = defineComponent(conversationActions);
    module.exports = ConversationActions
});
define("app/data/dm/delete_conversation", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function deleteConversation() {
        this.deleteConversation = function(a, b) {
            this.destroy({
                url: "/messages/with/conversation",
                data: {
                    id: b.conversation_id
                },
                eventData: b,
                success: this.success.bind(this, b.conversation_id),
                error: this.failure.bind(this, b.conversation_id)
            })
        }, this.success = function(a, b) {
            this.trigger("dataConversationDeleted", {
                conversation_id: a
            }), this.trigger("dataDMConversationListResult", b)
        }, this.failure = function(a, b) {
            this.trigger("dataConversationDeleteFailed", {
                conversation_id: a
            }), this.trigger("dataDMError", b)
        }, this.after("initialize", function() {
            this.on("uiDeleteConversation", this.deleteConversation)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), DeleteConversation = defineComponent(deleteConversation, withData);
    module.exports = DeleteConversation
});
define("app/ui/direct_message_actions", ["module", "require", "exports", "core/component", "app/ui/with_dropdownmenu"], function(module, require, exports) {
    function dmActionsDropdown() {
        this.defaultAttrs({
            itemType: "user",
            dialogSelector: "#dm_dialog",
            viewConversationList: "#dm_dialog_conversation_list",
            viewConversation: "#dm_dialog_conversation",
            tweetBoxSelector: ".dm-tweetbox",
            deleteSelector: ".dm-delete",
            deleteConfirmSelector: ".dm-deleting .js-prompt-ok",
            deleteCancelSelector: ".dm-deleting .js-prompt-cancel",
            reportSelector: ".dm-report",
            spamSelector: ".dm-spam",
            spamConfirmSelector: ".dm-reporting-spam .js-prompt-ok",
            spamCancelSelector: ".dm-reporting-spam .js-prompt-cancel",
            spamDialogSelector: ".dm-spam-confirm",
            abuseSelector: ".dm-abuse",
            abuseConfirmSelector: ".dm-reporting-abuse .js-prompt-ok",
            abuseCancelSelector: ".dm-reporting-abuse .js-prompt-cancel"
        }), this.deleteMessage = function(a, b) {
            a.preventDefault(), this.clearActions(), this.select("tweetBoxSelector").addClass("dm-deleting"), $(a.target).closest(".dm").addClass("marked-for-deletion"), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.deleteConfirm = function(a, b) {
            var c = this.select("viewConversation").find(".marked-for-deletion"), d = this.select("viewConversation").find("div[data-thread-id]").attr("data-thread-id");
            this.trigger("uiDMDialogDeleteMessage", {
                id: c.attr("data-message-id"),
                screen_name: d.replace(/^@/, "")
            }), this.select("viewConversationList").addClass("needs-refresh"), this.select("tweetBoxSelector").removeClass("dm-deleting")
        }, this.deleteCancel = function(a, b) {
            this.select("tweetBoxSelector").removeClass("dm-deleting"), this.select("viewConversation").find(".marked-for-deletion").removeClass("marked-for-deletion")
        }, this.reportMenu = function(a, b) {
            $(a.target).closest(".dm").addClass("active-menu")
        }, this.reportMessage = function(a, b) {
            this.trigger("uiForceDropdownClosed"), this.handleMenuClose(), this.clearActions();
            var c = $(a.target).hasClass("dm-spam") ? "spam": "abuse";
            this.select("tweetBoxSelector").addClass("dm-reporting-" + c), $(a.target).closest(".dm").addClass("marked-for-" + c), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.reportConfirm = function(a, b) {
            var c = this.select("viewConversation"), d = $(a.target).closest(this.attr.spamDialogSelector).length ? "spam": "abuse";
            this.trigger("uiDMDialogReportDM", {
                id: c.find(".marked-for-" + d).attr("data-message-id"),
                screen_name: c.find(".dm-convo").attr("data-thread-id"),
                report_as: d
            }), this.select("viewConversationList").addClass("needs-refresh"), this.select("tweetBoxSelector").removeClass("dm-reporting-" + d)
        }, this.reportCancel = function(a, b) {
            this.select("tweetBoxSelector").removeClass("dm-reporting-spam").removeClass("dm-reporting-abuse"), this.select("viewConversation").find(".marked-for-spam").removeClass("marked-for-spam"), this.select("viewConversation").find(".marked-for-abuse").removeClass("marked-for-abuse")
        }, this.handleMenuClose = function(a, b) {
            var c = this.$node.find(".dm-convo .dm");
            c.length && (c.removeClass("active-menu"), document.body.offsetHeight)
        }, this.clearActions = function(a, b) {
            this.deleteCancel(), this.reportCancel()
        }, this.after("initialize", function() {
            this.$dialog = this.select("dialogSelector"), this.on(document, "uiDMDialogClearActions", this.clearActions), this.on(document, "uiDropdownCloseRequested", this.handleMenuClose), this.on("click", {
                deleteSelector: this.deleteMessage,
                deleteConfirmSelector: this.deleteConfirm,
                deleteCancelSelector: this.deleteCancel,
                reportSelector: this.reportMenu,
                spamSelector: this.reportMessage,
                abuseSelector: this.reportMessage,
                spamConfirmSelector: this.reportConfirm,
                spamCancelSelector: this.reportCancel,
                abuseConfirmSelector: this.reportConfirm,
                abuseCancelSelector: this.reportCancel
            })
        })
    }
    var defineComponent = require("core/component"), withDropdownMenu = require("app/ui/with_dropdownmenu"), DMActionsDropdown = defineComponent(dmActionsDropdown, withDropdownMenu);
    module.exports = DMActionsDropdown
});
define("app/ui/direct_message_compose_with_intent", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function directMessageComposeWithIntent() {
        this.composeNewDMWithTweet = function(a, b) {
            this.trigger("uiOpenNewDM"), this.trigger("uiComposeWithTweet", b)
        }, this.after("initialize", function() {
            this.on("uiComposeNewDMWithTweet", this.composeNewDMWithTweet)
        })
    }
    var defineComponent = require("core/component"), DirectMessageComposeWithIntent = defineComponent(directMessageComposeWithIntent);
    module.exports = DirectMessageComposeWithIntent
});
define("app/utils/string", ["module", "require", "exports"], function(module, require, exports) {
    function isWhitespaceChar(a) {
        var b = a.charCodeAt(0);
        return b <= 32?!0 : !1
    }
    function subtractOne(a, b) {
        if (a == "")
            return "";
        var c = a.split(""), d = c.pop();
        return a = c.join(""), d == "0" ? subtractOne(a, !0) + "9" : (d -= 1, a == "" && d == 0 ? b ? "" : "0" : a + d)
    }
    function isDigitChar(a) {
        var b = a.charCodeAt(0);
        return b >= 48 && b <= 57?!0 : !1
    }
    function compareRight(a, b) {
        var c = 0, d = 0, e = 0, f, g;
        for (; ; d++, e++) {
            f = a.charAt(d), g = b.charAt(e);
            if (!isDigitChar(f)&&!isDigitChar(g))
                return c;
            if (!isDigitChar(f))
                return - 1;
            if (!isDigitChar(g))
                return 1;
            f < g ? c === 0 && (c =- 1) : f > g && c === 0 && (c = 1)
        }
    }
    var utils = {
        compare: function(a, b) {
            var c = 0, d = 0, e, f, g, h, i;
            if (a === b)
                return 0;
            typeof a == "number" && (a = a.toString()), typeof b == "number" && (b = b.toString());
            for (; ;) {
                if (c > 100)
                    return;
                e = f = 0, g = a.charAt(c), h = b.charAt(d);
                while (isWhitespaceChar(g) || g == "0")
                    g == "0" ? e++ : e = 0, g = a.charAt(++c);
                while (isWhitespaceChar(h) || h == "0")
                    h == "0" ? f++ : f = 0, h = b.charAt(++d);
                if (isDigitChar(g) && isDigitChar(h) && (i = compareRight(a.substring(c), b.substring(d))) != 0)
                    return i;
                if (g == 0 && h == 0)
                    return e - f;
                if (g < h)
                    return - 1;
                if (g > h)
                    return 1;
                ++c, ++d
            }
        },
        wordAtPosition: function(a, b, c, d) {
            c = c || /[^\s]+/g, d = d || 0;
            var e = null;
            return a.replace(c, function() {
                var a = arguments[d], c = arguments[arguments.length - 2];
                c <= b && c + a.length >= b && (e = a)
            }), e
        },
        parseBigInt: function(a) {
            return isNaN(Number(a)) ? NaN : a.toString()
        },
        subtractOne: subtractOne
    };
    module.exports = utils
});
define("app/data/direct_messages", ["module", "require", "exports", "core/i18n", "core/component", "app/utils/storage/core", "app/utils/string", "app/data/with_auth_token", "app/data/with_data"], function(module, require, exports) {
    function directMessages() {
        var a = 0, b;
        this.IFRAME_TIMEOUT = 12e4, this.defaultAttrs({
            noShowError: !0
        }), this.pollConversationList = function(a, b) {
            this.requestConversationList(null, {
                since_id: this.lastMessageId
            })
        }, this.requestConversationList = function(a, b) {
            var c = function(a) {
                a.is_empty_state ? this.trigger("dataDMConversationEmptyState", a) : this.trigger("dataDMConversationListResult", a)
            }.bind(this);
            this.get({
                url: "/messages",
                data: b,
                eventData: b,
                success: c,
                error: "dataDMError"
            })
        }, this.requestConversation = function(a, b) {
            var b = b || {}, c = b.conversation_id || b.screen_name;
            if (!c)
                return;
            this.get({
                url: "/messages/with/conversation",
                data: {
                    id: c,
                    recipient: b.recipient
                },
                eventData: b,
                success: b.msgAction ? "dataDM" + b.msgAction + "Success": "dataDMConversationResult",
                error: "dataDMError"
            })
        }, this.submitDMForm = function(a) {
            b = b || a.attr("action"), a.find(".auth-token").val(this.getAuthToken()), a.find(".file-data").attr("value") && a.removeAttr("encType"), a.attr("action", b + "?origin=" + encodeURIComponent(location.origin)), a.submit()
        }, this.isValidUpload = function(a) {
            var c = a.msgEvent, d = c, e = a.tweetBoxData;
            c.originalEvent && (c = c.originalEvent);
            var f = c.origin && b.indexOf(c.origin.replace(/^https?:/, "")) === 0;
            if (!f)
                return !1;
            try {
                d = JSON.parse(c.data)
            } catch (g) {}
            return this.uploadTimedOut || clearTimeout(a.tweetTimeout), d.tweetboxId = e.tweetboxId, d.media_id_string ? this.trigger("dataDMMediaUploadSuccess", $.extend(e, {
                media_id: d.media_id_string
            })) : this.trigger("dataDMError", d), !0
        }, this.uploadMedia = function(a, b) {
            this.uploadTimedOut=!1;
            var c = setTimeout(function() {
                this.uploadTimedOut=!0, this.off(window, "message", d), this.trigger("dataDMError", {
                    error: _('Uploading a photo timed out.'),
                    tweetboxId: b.tweetboxId
                })
            }.bind(this), this.IFRAME_TIMEOUT), d = function(a) {
                var e = this.isValidUpload({
                    msgEvent: a,
                    tweetTimeout: c,
                    tweetBoxData: b
                });
                e && this.off(window, "message", d)
            }.bind(this);
            this.on(window, "message", d), this.submitDMForm($("#" + b.tweetboxId))
        }, this.sendMessage = function(a, b) {
            var c = function(c) {
                c && c.retry && a.type != "dataDMDialogSendMessageRetry" ? setTimeout(function() {
                    this.trigger("dataDMDialogSendMessageRetry", b)
                }.bind(this), c.delay) : this.trigger("dataDMSendSuccess", c)
            };
            this.post({
                url: "/i/direct_messages/new",
                data: b,
                eventData: b,
                success: c.bind(this),
                error: "dataDMError"
            })
        }, this.deleteMessage = function(a, b) {
            var c = function(a) {
                this.trigger("dataDMError", a), this.trigger("dataDMDeleteFailed", b)
            };
            this.post({
                url: "/i/direct_messages/destroy",
                data: b,
                eventData: b,
                success: "dataDMDeleteSuccess",
                error: c.bind(this)
            })
        }, this.reportDM = function(a, b) {
            b.report_as != "not_spam" && this.trigger("uiDMDialogDeleteMessage", b), this.post({
                url: "/i/direct_messages/report",
                data: b,
                eventData: b,
                success: "dataDMConversationResult",
                error: "dataDMError"
            })
        }, this.triggerUnreadCount = function(b, c) {
            a = c.msgCount, a > 0 ? this.trigger("dataUserHasUnreadDMsWithCount", {
                msgCount: a
            }) : this.trigger("dataUserHasNoUnreadDMsWithCount")
        }, this.dispatchUnreadNotification = function(a, b) {
            if (!b ||!b.d)
                return;
            var c = b.d;
            c.status === "ok" && c.response != null && this.triggerUnreadCount(null, {
                msgCount: c.response
            })
        }, this.markDMsAsRead = function(a, b) {
            if (!b.last_message_id)
                throw new Error("Require last_message_id to mark a DM as read");
            var c = {
                last_message_id: b.last_message_id
            };
            b.recipient_id && (c.recipient_id = b.recipient_id), b.recipient_name && (c.recipient_name = b.recipient_name), b.retry === undefined && (b.retry=!0);
            var d, e;
            b.from_notification ? (c.get_count=!0, d = function(a) {
                "msgCount"in a && this.trigger("uiReadStateChanged", a)
            }, e = $.noop) : (d = function(a) {
                a.text != "success" ? e.call(this, a) : (a.lastMessageId = b.last_message_id, this.trigger("dataDMReadSuccess", a))
            }, e = function(a) {
                b.retry ? (b.retry=!1, this.trigger("dataMarkDMsAsRead", b)) : this.trigger("dataDMReadError", a)
            }), this.post({
                url: "/i/messages/mark_read",
                data: c,
                eventData: b,
                success: d.bind(this),
                error: e.bind(this)
            })
        }, this.checkForEnvelope = function(b, c) {
            c && c.section == "profile" && this.triggerUnreadCount(null, {
                msgCount: a
            })
        }, this.possiblyOpenDMDialog = function(a, b) {
            var c = this.attr.dm_options;
            if (c && c.show_dm_dialog) {
                var d = c.recipient, e = c.recipient_id, f = c.recipient_name;
                $(document).trigger("uiNeedsDMDialog", {
                    id: e,
                    screen_name: d,
                    name: f,
                    fromInitData: !0
                })
            }
        }, this.after("initialize", function() {
            this.on("uiNeedsDMConversationList", this.requestConversationList), this.on("uiNeedsDMConversation", this.requestConversation), this.on("uiDMDialogSendMessage dataDMDialogSendMessageRetry", this.sendMessage), this.on("uiSendDMWithMedia", this.uploadMedia), this.on("uiDMDialogDeleteMessage", this.deleteMessage), this.on("uiDMDialogReportDM", this.reportDM), this.on("dataRefreshDMs", this.pollConversationList), this.on("dataMarkDMsAsRead", this.markDMsAsRead), this.on("uiPageChanged", this.checkForEnvelope), this.on("uiSwiftLoaded", this.possiblyOpenDMDialog), this.on("dataNotificationsReceived", this.dispatchUnreadNotification), this.on("uiReadStateChanged", this.triggerUnreadCount), this.lastMessageId = $("#dm_dialog_conversation_list .DMInboxItem").first().attr("data-last-message-id") || 0, this.storage = new Storage("DM")
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component"), Storage = require("app/utils/storage/core"), StringUtils = require("app/utils/string"), withAuthToken = require("app/data/with_auth_token"), withData = require("app/data/with_data"), DirectMessages = defineComponent(directMessages, withData, withAuthToken);
    module.exports = DirectMessages
});
define("app/ui/with_dynamic_stylesheet", ["module", "require", "exports"], function(module, require, exports) {
    function withDynamicStylesheet() {
        this.createStyleSheet = function(a) {
            var b = document.createElement("style");
            return b.type = "text/css", a && (b.id = a), document.getElementsByTagName("head")[0].appendChild(b), b.sheet || new ShimStyleSheet
        }, this.rules = function(a) {
            return a.cssRules ? a.cssRules : a.rules
        }, this.insertCSSRule = function(a, b) {
            return a.insertRule(b, this.rules(a).length)
        }, this.removeCSSRule = function(a, b) {
            a.deleteRule(b)
        }
    }
    module.exports = withDynamicStylesheet;
    var ShimStyleSheet = function() {
        function a() {
            this.cssRules = []
        }
        return a.prototype.insertRule = function(a, b) {
            return this.cssRules[b] = {
                cssText: a
            }, b
        }, a.prototype.deleteRule = function(a) {
            delete this.cssRules[a], this.cssRules = this.cssRules.filter(function(a) {
                a != null
            })
        }, a
    }()
});
define("app/ui/with_item_actions", ["module", "require", "exports", "core/compose", "app/data/user_info", "core/utils", "app/data/with_card_metadata", "app/ui/with_interaction_data"], function(module, require, exports) {
    function withItemActions() {
        compose.mixin(this, [withInteractionData, withCardMetadata]), this.defaultAttrs({
            profileHoversEnabled: !1,
            pageContainer: "#doc",
            nestedContainerSelector: ".js-stream-item .in-reply-to, .js-expansion-container",
            showWithScreenNameSelector: ".show-popup-with-screen-name, .twitter-atreply",
            showWithIdSelector: ".show-popup-with-id, .js-user-profile-link",
            searchtagSelector: ".twitter-hashtag, .twitter-cashtag",
            cashtagSelector: ".twitter-cashtag",
            geoPivotSelector: ".js-geo-pivot-link",
            itemLinkSelector: ".twitter-timeline-link",
            cardInteractionLinkSelector: ".js-card2-interaction-link",
            cardExternalLinkSelector: ".js-card2-external-link",
            viewMoreItemSelector: ".view-more-container",
            inSnapbackExperiment: !1,
            snapbackSkipProfilePopup: !1
        }), this.showProfilePopupWithScreenName = function(a, b) {
            var c = $(a.target).closest(this.attr.showWithScreenNameSelector).text();
            c[0] === "@" && (c = c.substring(1));
            var d = {
                screenName: c
            }, e = this.getCardDataFromTweet($(a.target));
            b = utils.merge(this.interactionData(a, d), e), this.showProfile(a, b)
        }, this.showProfilePopupWithId = function(a, b) {
            var c = this.getCardDataFromTweet($(a.target));
            b = utils.merge(this.interactionDataWithCard(a), c), this.showProfile(a, b)
        }, this.showProfile = function(a, b) {
            if (this.ignoreProfilePopup(a)) {
                a.preventDefault();
                return 
            }
            this.attr.profileHoversEnabled && a.type == "mouseover" ? this.trigger(a.target, "uiShowProfileHover", b) : this.skipProfilePopup(a) || this.modifierKey(a) ? this.trigger(a.target, "uiShowProfileNewWindow", b) : (a.preventDefault(), this.trigger("uiShowProfilePopup", b))
        }, this.hideHover = function(a, b) {
            var c = this.getCardDataFromTweet($(a.target));
            b = utils.merge(this.interactionData(a), c), this.trigger("uiHideProfileHover", b)
        }, this.searchtagClick = function(a, b) {
            var c = $(a.target), d = c.closest(this.attr.searchtagSelector), e = d.is(this.attr.cashtagSelector) ? "uiCashtagClick": "uiHashtagClick", f = {
                query: d.text()
            };
            this.trigger(e, this.interactionData(a, f))
        }, this.isCardUrl = function(a) {
            var b = /^(https?:\/\/)?cards(-staging|-beta)?.twitter.com/;
            return b.test(a)
        }, this.geoPivotClick = function(a, b) {
            var c = {
                placeId: $(b.el).data("place-id")
            };
            this.trigger("uiGeoPivotClick", this.interactionData(a, c))
        }, this.itemLinkClick = function(a, b) {
            function c(a) {
                return !!a.parents(".permalink").length
            }
            var d = $(a.target).closest(this.attr.itemLinkSelector), e, f = {
                url: d.attr("data-expanded-url") || d.attr("href"),
                tcoUrl: d.attr("href"),
                text: d.text()
            };
            f = utils.merge(f, this.getCardDataFromTweet($(a.target))), (f.cardName === "promotion" || f.cardType === "promotions") && this.isCardUrl(f.url)&&!c($(a.target)) && (a.preventDefault(), e = d.parents(".stream-item"), this.trigger(e, "uiPromotionCardUrlClick")), this.trigger("uiItemLinkClick", this.interactionData(a, f))
        }, this.cardLinkClick = function(a, b, c) {
            var d = $(b.target).closest(this.attr.cardLinkSelector), e = this.getCardDataFromTweet($(b.target));
            this.trigger(a, this.interactionDataWithCard(b, e))
        }, this.getUserIdFromElement = function(a) {
            return a.length ? a.data("user-id") : null
        }, this.itemSelected = function(a, b) {
            var c = this.getCardDataFromTweet($(a.target));
            b.organicExpansion && this.trigger("uiItemSelected", utils.merge(this.interactionData(a), c))
        }, this.itemDeselected = function(a, b) {
            var c = this.getCardDataFromTweet($(a.target));
            this.trigger("uiItemDeselected", utils.merge(this.interactionData(a), c))
        }, this.isNested = function() {
            return this.$node.closest(this.attr.nestedContainerSelector).length
        }, this.ignoreProfilePopup = function(a) {
            return $(a.target).parents(".QuoteTweet").length
        }, this.skipProfilePopup = function(a) {
            var b = $(a.target), c = b.closest(".media-tagging-block").length > 0, d = b.closest("#profile_popup").length > 0, e = b.closest(".twitter-atreply").length > 0;
            return c?!1 : this.attr.snapbackExperimentOptions?!this.attr.snapbackExperimentOptions.profilePopup : userInfo.getDecider("disable_profile_popup") || this.attr.profileHoversEnabled || this.attr.disableProfilePopup || d&&!e
        }, this.modifierKey = function(a) {
            if (a.shiftKey || a.ctrlKey || a.metaKey || a.which > 1)
                return !0
        }, this.removeTweetsFromUser = function(a, b) {
            var c = this.$node.find("[data-user-id=" + b.userId + "].js-stream-tweet"), d = this;
            $.each(c, function(a, b) {
                var c = $(b).attr("data-tweet-id");
                d.removeEngagementsOnTweet(c)
            }), c.parent().remove(), this.trigger("uiRemovedSomeTweets")
        }, this.navigateToViewMoreURL = function(a) {
            var b = $(a.target), c;
            b.find(this.attr.viewMoreItemSelector).length && (c = b.find(".view-more-link"), this.trigger(c, "uiNavigate", {
                href: c.attr("href")
            }))
        }, this.removeEngagementsOnTweet = function(a) {
            var b = this.$node.find("[data-item-id=" + a + "]");
            b.remove()
        }, this.after("initialize", function() {
            this.isNested() || (this.on("click", {
                showWithScreenNameSelector: this.showProfilePopupWithScreenName,
                showWithIdSelector: this.showProfilePopupWithId,
                searchtagSelector: this.searchtagClick,
                geoPivotSelector: this.geoPivotClick,
                itemLinkSelector: this.itemLinkClick,
                cardExternalLinkSelector: this.cardLinkClick.bind(this, "uiCardExternalLinkClick"),
                cardInteractionLinkSelector: this.cardLinkClick.bind(this, "uiCardInteractionLinkClick")
            }), this.attr.profileHoversEnabled && (this.on("mouseover", {
                showWithScreenNameSelector: this.showProfilePopupWithScreenName,
                showWithIdSelector: this.showProfilePopupWithId
            }), this.on("mouseout", {
                showWithScreenNameSelector: this.hideHover,
                showWithIdSelector: this.hideHover
            })), this.on("uiHasExpandedTweet uiHasClickedTweet", this.itemSelected), this.on("uiHasCollapsedTweet", this.itemDeselected), this.on("uiRemoveTweetsFromUser", this.removeTweetsFromUser), this.on("uiShortcutEnter", this.navigateToViewMoreURL))
        })
    }
    var compose = require("core/compose"), userInfo = require("app/data/user_info"), utils = require("core/utils"), withCardMetadata = require("app/data/with_card_metadata"), withInteractionData = require("app/ui/with_interaction_data");
    module.exports = withItemActions
});
define("app/ui/with_timestamp_updating", ["module", "require", "exports", "core/i18n", "lib/twitter_cldr"], function(module, require, exports) {
    function withTimestampUpdating() {
        this.defaultAttrs({
            timestampSelector: ".tweet-timestamp",
            relativeTimestampSelector: ".js-relative-timestamp",
            relativeTimestampClass: "js-relative-timestamp"
        }), this.currentTimeSecs = function() {
            return new Date / 1e3
        }, this.isMuricanLocale = function() {
            var a = ["en"];
            return $.inArray($("html").attr("lang"), a) >= 0
        }, this.updateAccessibleShortTimestamp = function(a, b) {
            var c = this.currentTimeSecs(), d = Math.floor(c - b);
            if (d < 3)
                return;
            var e = TimespanFormatter.format(b - c), f = a.closest(this.attr.timestampSelector), g = f.find(".u-hiddenVisually");
            g.length || (f.find(this.attr.relativeTimestampSelector).removeAttr("data-aria-label-part").attr("aria-hidden", "true"), g = $(HIDDEN_TIMESTAMP), f.append(g)), g.text(e)
        }, this.updateTimestamps = function() {
            this.select("relativeTimestampSelector").each(function(a, b) {
                var c = $(b), d = c.data("time"), e = this.isMuricanLocale() ? "abbreviated": "short", f = this.formatTimestamp(d, e);
                f.relative || c.removeClass(this.attr.relativeTimestampClass), c.text(f.text), this.updateAccessibleShortTimestamp(c, d)
            }.bind(this))
        }, this.formatTimestamp = function(a, b) {
            var c = this.currentTimeSecs(), d = Math.floor(c - a), e;
            return d < 3 ? e = NOW_TEXT : e = TimespanFormatter.format(d, {
                direction: "none",
                type: b
            }), {
                text: e,
                relative: !0
            }
        }, this.after("initialize", function() {
            this.on(document, "uiWantsToRefreshTimestamps uiPageChanged", this.updateTimestamps)
        })
    }
    var _ = require("core/i18n"), TwitterCldr = require("lib/twitter_cldr"), TimespanFormatter = new TwitterCldr.TimespanFormatter, NOW_TEXT = _('now'), HIDDEN_TIMESTAMP = '<span class="u-hiddenVisually" data-aria-label-part="last"></span>';
    module.exports = withTimestampUpdating
});
define("app/ui/with_tweetbox_initialization", ["module", "require", "exports", "core/utils"], function(module, require, exports) {
    var utils = require("core/utils");
    module.exports = function() {
        this.defaultAttrs({
            tweetFormSelector: ".tweet-form"
        }), this.initTweetbox = function(a) {
            var b = this.select("tweetFormSelector"), c = this.attr.preexpandTweetbox ||!1;
            this.trigger(b, "uiInitTweetbox", utils.merge({
                draftTweetId: this.attr.draftTweetId,
                condensable: !c,
                preexpandTweetbox: c,
                suppressSuccessMessage: this.attr.suppressSuccessMessage
            }, {
                eventData: this.attr.eventData
            }, a))
        }
    }
});
define("app/ui/user_actions_dropdown", ["module", "require", "exports", "core/component", "app/ui/with_dropdownmenu"], function(module, require, exports) {
    function userActionsDropdown() {
        this.defaultAttrs({
            dropdownThresholdSelector: ".dropdown-threshold"
        }), this.scrollIntoView = function() {
            var a = this.$node.closest(this.attr.dropdownThresholdSelector), b, c;
            a.length && (b = this.$node.find(this.attr.dropDownSelector), c = b.offset().top + b.outerHeight() - (a.offset().top + a.height()), c > 0 && a.animate({
                scrollTop: a.scrollTop() + c
            }))
        }, this.after("initialize", function() {
            this.on("uiDropdownOpened", this.scrollIntoView), this.on(document, "uiHasInjectedNewTimeline uiHasInjectedOldTimelineItems uiHasInjectedRangeTimelineItems", this.applyARIAAttrs)
        })
    }
    var defineComponent = require("core/component"), withDropdownMenu = require("app/ui/with_dropdownmenu"), UserActionsDropdown = defineComponent(withDropdownMenu, userActionsDropdown);
    module.exports = UserActionsDropdown
});
define("app/utils/user_dom_data", ["module", "require", "exports"], function(module, require, exports) {
    var userDomData = {
        getId: function(a) {
            return a.attr("data-user-id")
        },
        getName: function(a) {
            return a.data("name")
        },
        getScreenName: function(a) {
            return a.data("screen-name")
        }
    };
    module.exports = userDomData
});
define("app/ui/with_user_actions", ["module", "require", "exports", "core/compose", "core/i18n", "core/utils", "app/ui/with_interaction_data", "app/ui/user_actions_dropdown", "app/utils/string", "app/utils/user_dom_data"], function(module, require, exports) {
    function withUserActions() {
        compose.mixin(this, [withInteractionData]), this.defaultAttrs({
            followButtonSelector: ".follow-button, .follow-link, .WebToast-followButton",
            topicLinkSelector: ".user-topic-link",
            userInfoSelector: ".user-actions",
            dropdownSelector: ".user-actions .dropdown",
            dropdownItemSelector: ".user-actions .dropdown.open .dropdown-menu li",
            blockOrReportButtonSelector: ".block-or-report-text",
            reportButtonSelector: ".report-text",
            unmuteButtonSelector: ".unmute-button",
            muteButtonSelector: ".mute-button",
            followStates: ["not-following", "following", "blocked", "pending"],
            userActionClassesToEvents: {
                "mention-text": ["uiMentionAction", "mentionUser"],
                "dm-text": ["uiDmAction", "dmUser"],
                "list-text": ["uiListAction"],
                "block-text": ["uiBlockAction"],
                "unblock-text": ["uiUnblockAction"],
                "hide-suggestion-text": ["uiHideSuggestionAction"],
                "retweet-on-text": ["uiRetweetOnAction"],
                "retweet-off-text": ["uiRetweetOffAction"],
                "device-notifications-on-text": ["uiDeviceNotificationsOnAction", "deviceNotificationsOn"],
                "device-notifications-off-text": ["uiDeviceNotificationsOffAction"],
                "embed-profile": ["uiEmbedProfileAction", "redirectToEmbedProfile"]
            }
        }), this.getClassNameFromList = function(a, b) {
            var c = b.filter(function(b) {
                return a.hasClass(b)
            });
            return c[0]
        }, this.getUserActionEventNameAndMethod = function(a) {
            var b = this.getClassNameFromList(a.closest("li"), Object.keys(this.attr.userActionClassesToEvents));
            return this.attr.userActionClassesToEvents[b]
        }, this.getFollowState = function(a) {
            return this.getClassNameFromList(a, this.attr.followStates)
        }, this.getInfoElementFromEvent = function(a) {
            var b = $(a.target);
            return b.closest(this.attr.userInfoSelector)
        }, this.findInfoElementForUser = function(a) {
            var b = this.attr.userInfoSelector + "[data-user-id=" + StringUtils.parseBigInt(a) + "]";
            return this.$node.find(b)
        }, this.getEventName = function(a) {
            var b = {
                "not-following": "uiFollowAction",
                following: "uiUnfollowAction",
                blocked: "uiUnblockAction",
                pending: "uiCancelFollowRequestAction"
            };
            return b[a]
        }, this.addCancelHoverStyleClass = function(a) {
            a.addClass("cancel-hover-style"), a.one("mouseleave", function() {
                a.removeClass("cancel-hover-style")
            })
        }, this.handleFollowButtonClick = function(a) {
            a.stopPropagation(), this.trigger($(a.target), "uiCloseDropdowns");
            var b = this.getInfoElementFromEvent(a), c = $(a.target).closest(this.attr.followButtonSelector);
            this.addCancelHoverStyleClass(c);
            var d = this.getFollowState(b);
            d == "not-following" && b.data("protected") && this.trigger("uiShowMessage", {
                message: _('A follow request has been sent to @{{screen_name}} and is pending their approval.', {
                    screen_name: b.data("screen-name")
                })
            });
            var e = this.getEventName(d), f = {
                originalFollowState: d,
                inject_tweet: d == "not-following" && this.attr.injectTweetOnFollow
            };
            this.trigger(e, this.interactionData(a, f))
        }, this.handleTopicLinkClick = function(a) {
            var b = this.interactionData(a);
            this.trigger("uiUserTopicClickAction", b)
        }, this.handleLoggedOutFollowButtonClick = function(a) {
            a.stopPropagation(), this.trigger($(a.target), "uiCloseDropdowns"), this.trigger("uiOpenSigninOrSignupDialog", {
                signUpOnly: !0,
                screenName: this.getInfoElementFromEvent(a).data("screen-name")
            }), this.trigger("uiLoggedOutFollowAttempt", this.interactionData(a))
        }, this.handleBlockOrReportClick = function(a) {
            var b = this.getInfoElementFromEvent(a), c = b.data("screen-name"), d = b.data("user-id").toString(), e = this.getFollowState(b), f = utils.merge(this.interactionData(a), {
                screenName: c,
                userId: d,
                target: "user",
                viewerFollowsAuthor: e === "following",
                viewerBlocksAuthor: e === "blocked"
            });
            this.trigger("uiNeedsBlockOrReportDialog", f)
        }, this.handleUserAction = function(a) {
            var b = $(a.target), c = this.getUserActionEventNameAndMethod(b);
            if (!c)
                return;
            a.stopPropagation();
            var d = this.getInfoElementFromEvent(a), e = c[0], f = c[1], g = this.getFollowState(d), h = {
                originalFollowState: g
            };
            f && (h = this[f](d, e, h)), h && this.trigger(e, this.interactionData(a, h)), this.trigger(b, "uiCloseDropdowns")
        }, this.deviceNotificationsOn = function(a, b, c) {
            return this.attr.deviceEnabled ? c : (this.attr.smsDeviceVerified || this.attr.hasPushDevice ? this.trigger("uiOpenConfirmDialog", {
                titleText: _('Enable mobile notifications for Tweets'),
                bodyText: _('Before you can receive mobile notifications for @{{screenName}}\'s Tweets, you need to enable the Tweet notification setting.', {
                    screenName: a.data("screen-name")
                }),
                cancelText: _('Close'),
                submitText: _('Enable Tweet notifications'),
                action: this.attr.hasPushDevice ? "ShowPushTweetsNotifications": "ShowMobileNotifications",
                top: this.attr.top
            }) : this.trigger("uiOpenConfirmDialog", {
                titleText: _('Setup mobile notifications'),
                bodyText: _('Before you can receive mobile notifications for @{{screenName}}\'s Tweets, you need to set up your phone.', {
                    screenName: a.data("screen-name")
                }),
                cancelText: _('Cancel'),
                submitText: _('Set up phone'),
                action: "ShowMobileNotifications",
                top: this.attr.top
            }), !1)
        }, this.redirectToMobileNotifications = function() {
            window.location = "/settings/devices"
        }, this.redirectToPushNotificationsHelp = function() {
            window.location = "//support.twitter.com/articles/20169887"
        }, this.redirectToEmbedProfile = function(a, b, c) {
            return this.trigger("uiNavigate", {
                href: "/settings/widgets/new/user?screen_name=" + a.data("screen-name")
            }), !0
        }, this.mentionUser = function(a, b, c) {
            this.trigger("uiOpenTweetDialog", {
                screenName: a.data("screen-name"),
                title: _('Tweet to {{name}}', {
                    name: a.data("name")
                })
            })
        }, this.dmUser = function(a, b, c) {
            return this.trigger("uiNeedsDMDialog", {
                id: userDomDataUtil.getId(a),
                screen_name: userDomDataUtil.getScreenName(a),
                name: userDomDataUtil.getName(a)
            }), c
        }, this.hideSuggestion = function(a, b, c) {
            return utils.merge(c, {
                feedbackToken: a.data("feedback-token")
            })
        }, this.followStateChange = function(a, b) {
            this.updateFollowState(b.userId, b.newState), b.fromShortcut && (b.newState === "not-following" ? this.trigger("uiShowMessage", {
                message: _('You have unblocked {{username}}', {
                    username: b.username
                })
            }) : b.newState === "blocked" && this.trigger("uiUpdateAfterBlock", {
                userId: b.userId
            }))
        }, this.updateFollowState = function(a, b) {
            var c = this.findInfoElementForUser(a), d = this.getFollowState(c);
            d && c.removeClass(d), c.addClass(b), b === "blocked" ? this.trigger("uiUpdateAfterBlock", {
                userId: a
            }) : b === "not-following" && this.trigger("uiShowProfileTweets")
        }, this.follow = function(a, b) {
            var c = this.findInfoElementForUser(b.userId), d = c.data("protected") ? "pending": "following";
            this.updateFollowState(b.userId, d), c.addClass("including")
        }, this.unfollow = function(a, b) {
            var c = this.findInfoElementForUser(b.userId);
            this.updateFollowState(b.userId, "not-following"), c.removeClass("including notifying")
        }, this.cancel = function(a, b) {
            var c = this.findInfoElementForUser(b.userId);
            this.updateFollowState(b.userId, "not-following")
        }, this.block = function(a, b) {
            var c = this.findInfoElementForUser(b.userId);
            this.trigger("jsClearReinjectionCookiesForUser", {
                userId: b.userId
            }), this.updateFollowState(b.userId, "blocked"), c.removeClass("including notifying")
        }, this.unblock = function(a, b) {
            this.updateFollowState(b.userId, "not-following")
        }, this.retweetsOn = function(a, b) {
            var c = this.findInfoElementForUser(b.userId);
            c.addClass("including")
        }, this.retweetsOff = function(a, b) {
            var c = this.findInfoElementForUser(b.userId);
            c.removeClass("including")
        }, this.notificationsOn = function(a, b) {
            var c = this.findInfoElementForUser(b.userId), d = c.find(this.attr.deviceFollowCheckboxSelector);
            c.addClass("notifying"), d.prop("checked", !0)
        }, this.notificationsOff = function(a, b) {
            var c = this.findInfoElementForUser(b.userId), d = c.find(this.attr.deviceFollowCheckboxSelector);
            c.removeClass("notifying"), d.prop("checked", !1)
        }, this.blockUserConfirmed = function(a, b) {
            a.stopImmediatePropagation(), this.trigger("uiBlockAction", b.sourceEventData)
        }, this.preventFocus = function(a, b) {
            a.preventDefault()
        }, this.handleMuteButtonClick = function(a) {
            return function(b, c) {
                c = this.interactionData(b), c.scribeContext = {
                    element: "muted_button"
                }, this.trigger(b.target, a, c)
            }
        }, this.updateMuteButtonState = function(a, b) {
            var c = this.$node.find(this.attr.userInfoSelector + "[data-user-id=" + b.userId + "]"), d = c.find(this.attr.unmuteButtonSelector), e = c.find(this.attr.muteButtonSelector), f = d.is(":focus"), g = e.is(":focus");
            c[b.modOp](b.modClass), e.removeClass("first-load"), f ? e.focus() : g && d.focus()
        }, this.after("initialize", function() {
            UserActionsDropdown.attachTo(this.$node);
            if (!this.attr.loggedIn) {
                this.on("click", {
                    followButtonSelector: this.handleLoggedOutFollowButtonClick
                });
                return 
            }
            this.on("click", {
                followButtonSelector: this.handleFollowButtonClick,
                topicLinkSelector: this.handleTopicLinkClick,
                dropdownItemSelector: this.handleUserAction,
                unmuteButtonSelector: this.handleMuteButtonClick("uiDidUnmuteUser"),
                muteButtonSelector: this.handleMuteButtonClick("uiDidMuteUser"),
                blockOrReportButtonSelector: this.handleBlockOrReportClick,
                reportButtonSelector: this.handleBlockOrReportClick
            }), this.on("mousedown", {
                unmuteButtonSelector: this.preventFocus,
                muteButtonSelector: this.preventFocus
            }), this.on(document, "uiFollowStateChange dataFollowStateChange dataBulkFollowStateChange", this.followStateChange), this.on(document, "uiFollowAction", this.follow), this.on(document, "uiUnfollowAction", this.unfollow), this.on(document, "uiCancelFollowRequestAction", this.cancel), this.on(document, "uiUnblockAction", this.unblock), this.on(document, "uiRetweetOnAction dataRetweetOnAction", this.retweetsOn), this.on(document, "uiRetweetOffAction dataRetweetOffAction", this.retweetsOff), this.on(document, "uiDeviceNotificationsOnAction dataDeviceNotificationsOnAction", this.notificationsOn), this.on(document, "uiDeviceNotificationsOffAction dataDeviceNotificationsOffAction", this.notificationsOff), this.on(document, "uiShowMobileNotificationsConfirm", this.redirectToMobileNotifications), this.on(document, "uiShowPushTweetsNotificationsConfirm", this.redirectToPushNotificationsHelp), this.on(document, "uiUpdateMuteButtonState", this.updateMuteButtonState), this.on(document, "uiBlockAction", this.block), this.on(document, "uiDidBlockUser", this.blockUserConfirmed), this.on(document, "uiDidTriggerBlockingAction", this.block)
        })
    }
    var compose = require("core/compose"), _ = require("core/i18n"), utils = require("core/utils"), withInteractionData = require("app/ui/with_interaction_data"), UserActionsDropdown = require("app/ui/user_actions_dropdown"), StringUtils = require("app/utils/string"), userDomDataUtil = require("app/utils/user_dom_data");
    module.exports = withUserActions
});
define("app/ui/direct_message_dialog", ["module", "require", "exports", "core/component", "app/utils/string", "app/ui/with_dialog", "app/ui/with_dynamic_stylesheet", "app/ui/with_item_actions", "app/ui/with_timestamp_updating", "app/ui/with_tweetbox_initialization", "app/ui/with_user_actions"], function(module, require, exports) {
    function directMessageDialog() {
        this.defaultAttrs({
            itemType: "user",
            dialogSelector: "#dm_dialog",
            classConversationList: "dm-conversation-list",
            classConversation: "dm-conversation",
            classNew: "dm-new",
            classViewParticipants: "dm-view-participants",
            viewConversationList: "#dm_dialog_conversation_list",
            viewConversation: "#dm_dialog_conversation",
            viewNew: "#dm_dialog_new",
            viewConversationParticipants: "#dm_dialog_view_participants",
            linksForConversationListView: "#dm_dialog_new h3 a, #dm_dialog_conversation h3 a, #dm_dialog_view_participants h3 a",
            linksForConversationView: ".DMInboxItem, .dm-view-participants-done",
            linksForNewView: ".dm-new-button",
            autocompleteParticipantSelector: ".dm-to.user-select-container",
            toFieldSelector: ".dm-to-input",
            contentSelector: ".twttr-dialog-content",
            tweetBoxSelector: ".dm-tweetbox",
            newConversationEditor: "#tweet-box-dm-new-conversation",
            conversationActionsSelector: ".dm-conversation-actions",
            viewParticipantsDoneSelector: ".dm-view-participants-done",
            conversationListSelector: "#dm_dialog_conversation_list .DMInbox",
            errorContainerSelector: ".js-dm-error",
            errorTextSelector: ".dm-error-text",
            errorCloseSelector: ".js-dismiss",
            errorReloadSelector: ".js-reload",
            markAllReadSelector: ".mark-all-read",
            markReadConfirmSelector: ".mark-read-confirm .js-prompt-ok",
            markReadCancelSelector: ".mark-read-confirm .js-prompt-cancel",
            mediaPreviewSelector: ".DMInboxItem-media"
        }), this.usingConversationService = function() {
            return this.attr.deciders && this.attr.deciders.dm_convo
        }, this.generateConversationId = function(a, b) {
            if (this.usingConversationService() && a && b) {
                var c = [a, b];
                return (StringUtils.compare(c[0], c[1]) < 0 ? c : c.reverse()).join("-")
            }
        }, this.openDialog = function(a, b) {
            b && b.fromInitData && this.on("uiDialogClosed", function() {
                this.isDialogNavigation || this.trigger("uiNavigate", {
                    href: "/"
                })
            }), this.trigger("dataRefreshDMs"), b && b.screen_name ? (this.triggerEventOnUserSelector("uiResetSelectedUsers"), this.setRecipientForNewConversationRender(b)) : this.renderConversationListView(), this.open()
        }, this.before("openDialog", function(a, b) {
            if (b && b.id) {
                var c = this.generateConversationId(this.attr.userId, b.id);
                c && (b.conversation_id = c)
            }
        }), this.closeDialog = function() {
            this.isDialogNavigation=!1, this.close()
        }, this.renderConversationListView = function(a, b) {
            a && a.preventDefault(), this.renderView(this.attr.classConversationList);
            var c = this.select("viewConversationList");
            this.usingConversationService() && (this.unreadCount = 0, this.trigger("uiReadStateChanged", {
                msgCount: this.unreadCount
            }));
            if (c.hasClass("needs-refresh")) {
                c.removeClass("needs-refresh"), this.trigger("uiNeedsDMConversationList", {
                    since_id: 0
                });
                return 
            }
            this.trigger("uiDMDialogOpenedConversationList")
        }, this.possiblyRenderConversation = function(a, b) {
            var c = b.items;
            c.length && this.renderConversationView(null, c[0])
        }, this.before("possiblyRenderConversation", function(a, b) {
            var c = b.items;
            if (c.length != 1)
                return;
            var d = this.generateConversationId(this.attr.userId, c[0].id);
            d && (b.items[0].conversation_id = d)
        }), this.handleProfileLinkClick = function() {
            this.isDialogNavigation || (this.isDialogNavigation = this.$dialog.is(":visible"), this.closeImmediately())
        }, this.renderConversationView = function(a, b) {
            if (a) {
                var c = $(a.target);
                if (this.isProfileLink(c)) {
                    this.handleProfileLinkClick();
                    return 
                }
                if (c.hasClass("DMInboxItem-media"))
                    return 
            }
            this.dmRecipientSelected=!0, this.triggerUiDMRecipientChanged(), this.trigger("uiDMDialogClearActions"), a && a.preventDefault();
            var b = b || {}, d = b.screen_name || $(b.el).attr("data-thread-id"), e = b.name || $(b.el).find(".fullname").html(), f = b.conversation_id || $(b.el).attr("data-thread-id") || d;
            this.lastMessageIdInConversation = $(b.el).attr("data-last-message-id"), this.isConversationUnread = $(b.el).hasClass("is-unread"), this.trigger("uiPrepareUpdateConversationName", {
                is_oto: b.is_oto
            }), this.$node.find(".dm_dialog_real_name").html(e), this.select("viewConversation").find(this.attr.contentSelector).empty(), b.preventDMBoxReset || (this.renderView(this.attr.classConversation), this.trigger("uiDMDialogOpenedConversation", {
                recipient: d
            }));
            var g = conversationCache[d] || {}, h = g.data && g.lastMessageId == this.lastMessageIdInConversation;
            h ? this.updateConversation(null, g.data) : this.trigger("uiNeedsDMConversation", {
                screen_name: d,
                recipient: b.name,
                conversation_id: f,
                fromInitData: b.fromInitData
            }), b.preventDMBoxReset || (this.resetDMBox(), this.select("newConversationEditor").focus())
        }, this.renderNewView = function(a, b) {
            a && a.preventDefault(), this.renderView(this.attr.classNew), this.trigger("uiDMDialogOpenedNewConversation"), this.resetDMBox(), this.resetUserSelection(), this.isOpen() || (this.trigger("dataRefreshDMs"), this.open()), b && b.recipient && this.triggerEventOnUserSelector("uiShowSelectedUsers")
        }, this.renderViewParticipants = function(a, b) {
            a && a.preventDefault(), this.trigger("uiNeedsConversationParticipants", {
                conversation_id: b.conversation_id
            }), this.renderView(this.attr.classViewParticipants)
        }, this.renderView = function(a) {
            this.hideError(), this.markReadCancel(), this.$dialogContainer.removeClass(this.viewClasses).addClass(a), this.trigger(this.$dialog, "uiDialogContentChanged"), this.setScribeComponent(a)
        }, this.setScribeComponent = function(a) {
            this.attr.eventData || (this.attr.eventData = {});
            var b;
            switch (a) {
            case this.attr.classNew:
                b = "dm_new_conversation_dialog";
                break;
            case this.attr.classConversation:
                b = "dm_existing_conversation_dialog";
                break;
            case this.attr.classConversationList:
                b = "dm_conversation_list_dialog";
                break;
            case this.attr.classViewParticipants:
                b = "dm_view_participants_dialog"
            }
            this.attr.eventData.scribeContext = {
                component: b
            }
        }, this.clearConversationList = function(a, b) {
            this.select("conversationListSelector").empty()
        }, this.updateConversationList = function(a, b) {
            var c = this.select("viewConversationList"), d = c.find("li");
            b.sourceEventData.since_id ? d.filter(function() {
                return $.inArray($(this).attr("data-thread-id"), b.threads)>-1
            }).remove() : d.remove();
            var e = b.html || "";
            c.find(this.attr.contentSelector + " ul").prepend(e);
            var f = c.find(".dm-no-messages");
            c.find("li").length == 0 ? (f.show(), this.select("markAllReadSelector").addClass("disabled").attr("disabled", !0)) : (f.hide(), this.select("markAllReadSelector").removeClass("disabled").attr("disabled", !1)), this.trigger("uiResetDMPoll"), this.latestMessageId = b.last_message_id||-1
        }, this.updateViewParticipants = function(a, b) {
            a && a.preventDefault(), this.$dialogContainer.hasClass(this.attr.classViewParticipants) || this.renderViewParticipants(null, b);
            var c = this.select("viewConversationParticipants").find(this.attr.contentSelector);
            c.html(b.html), this.select("viewParticipantsDoneSelector").attr("data-thread-id", b.conversation_id), this.trigger("uiDMDialogOpenedViewParticipants")
        }, this.updateConversation = function(a, b) {
            a && a.preventDefault();
            var c = a && a.type, d = b.recipient.screen_name, e = (conversationCache[d] || {}).forceRetryMarkAsRead;
            conversationCache[d] = {
                data: b,
                lastMessageId: - 1,
                forceRetryMarkAsRead: !!e
            }, b.recipient && this.$node.find(".dm_dialog_real_name").html(b.recipient.name).data("raw-name", b.recipient.raw_name), this.trigger("uiPrepareUpdateConversationName", {
                is_oto: b.is_oto
            }), b.conversation_actions && this.select("conversationActionsSelector").html(b.conversation_actions.html), (c == "dataDMConversationResult" || c == "dataDMSendSuccess")&&!this.$dialogContainer.hasClass(this.attr.classConversation) && this.renderConversationView(null, {
                screen_name: d,
                conversation_id: b.conversation_id || d,
                name: b.recipient.name,
                is_oto: b.is_oto
            });
            var f = this.select("viewConversation").find(this.attr.contentSelector);
            f.html(b.html), b.sourceEventData && b.sourceEventData.fromInitData && this.resetDMBox(), this.trigger("uiResetDMPoll");
            if (b.is_oto && b.is_empty) {
                c === "dataDMDeleteSuccess" ? (this.trigger("dataRefreshDMs"), this.renderConversationListView()) : this.setRecipientForNewConversationRender(b.recipient);
                return 
            }
            var g = f.find(".dm-convo");
            if (g.length) {
                var h = g.find(".dm").last().attr("data-message-id");
                conversationCache[d].lastMessageId = h;
                if (conversationCache[d].forceRetryMarkAsRead || c == "dataDMConversationResult" && (!b.sourceEventData ||!b.sourceEventData.msgAction))
                    conversationCache[d].forceRetryMarkAsRead=!1, g.find(".dm[data-unread=true]").length && (conversationCache[d].forceRetryMarkAsRead = e=!0), this.initMsgRead({
                        messageId: h,
                        recipientId: b.recipient.id_str,
                        recipientName: d
                    }, e);
                g.scrollTop(g[0].scrollHeight)
            }
        }, this.initMsgRead = function(a, b) {
            var c = this.lastMessageIdInConversation && StringUtils.compare(this.lastMessageIdInConversation, a.messageId)==-1;
            if (b || this.isConversationUnread || c) {
                this.markMessages(null, a), this.select("viewConversationList").find(".DMInboxItem.is-unread[data-last-message-id=" + a.messageId + "]").removeClass("is-unread");
                if (!this.usingConversationService()) {
                    var d = this.select("viewConversation").find(".js-dm-item[data-unread=true]").length;
                    this.unreadCount -= d, this.trigger("uiReadStateChanged", {
                        msgCount: this.unreadCount
                    })
                }
            }
        }, this.sendMessage = function(a, b) {
            var c = this.$dialogContainer.hasClass(this.attr.classConversation), d = this.select("viewConversation"), e = d.find("div[data-thread-id]"), f = c && e.length, g, h = b.recipient;
            if (!h && f) {
                h = e.attr("data-thread-id");
                if (!h) {
                    setTimeout(function() {
                        this.sendMessage(a, b)
                    }.bind(this), 100);
                    return 
                }
            } else 
                h || (h = "", g = this.select("viewNew").find("input[type=text]").val().trim());
            var i = {
                tweetboxId: b.tweetboxId,
                screen_name: (g || "").replace(/^@/, ""),
                conversation_id: h,
                text: b.tweetData.status
            };
            f && (this.retryRefresh = 0, i.lastMsgId = d.find(".dm-convo .dm:last").attr("data-message-id")), b.media_id && (i.media_id = b.media_id), b.tweetData && b.tweetData.tweet_id && (i.tweet_id = b.tweetData.tweet_id, this.trigger("uiDMDialogSendMessageWithTweet", {
                conversation_id: i.conversation_id || i.screen_name,
                status_id: i.tweet_id
            })), this.trigger("uiDMDialogSendMessage", i), this.resetDMBox(), this.trigger("uiDMInboxWantsRefreshed")
        }, this.inboxNeedsRefresh = function(a, b) {
            this.select("viewConversationList").addClass("needs-refresh")
        }, this.markMessages = function(a, b) {
            this.trigger("dataMarkDMsAsRead", {
                last_message_id: b.messageId || this.latestMessageId,
                recipient_id: b.recipientId,
                recipient_name: b.recipientName
            })
        }, this.markAllMessages = function(a, b) {
            this.markMessages(a, b), this.trigger("uiDMDialogMarkMessage"), this.trigger("uiReadStateChanged", {
                msgCount: 0
            })
        }, this.updateReadState = function(a, b) {
            (!b || b.text != "success") && this.handleReadStateFailure(a, b), this.trigger("uiDMInboxWantsRefreshed"), this.$dialogContainer.hasClass(this.attr.classConversationList) && this.openDialog()
        }, this.handleReadStateFailure = function(a, b) {
            b && b.sourceEventData && b.sourceEventData.recipient_name && (conversationCache[b.sourceEventData.recipient_name].forceRetryMarkAsRead=!0)
        }, this.refreshDMList = function(a, b) {
            b && b.d && b.d.status === "ok" && b.d.response != null && (this.unreadCount != b.d.response && this.$dialogContainer.is(":visible") && (this.$dialogContainer.hasClass(this.attr.classConversationList) ? this.trigger("dataRefreshDMs") : this.$dialogContainer.hasClass(this.attr.classConversation) && this.$dialogContainer.find(".dm-convo").length && (this.lastMessageIdInConversation =- 1, this.renderConversationView(null, {
                screen_name: this.$dialogContainer.find(".dm-convo").attr("data-thread-id"),
                name: this.$dialogContainer.find(".dm_dialog_real_name").html(),
                preventDMBoxReset: !0
            }))), this.unreadCount = b.d.response)
        }, this.showMarkReadConfirm = function(a, b) {
            a && a.preventDefault();
            if (this.select("markAllReadSelector").hasClass("disabled"))
                return;
            this.select("viewConversationList").addClass("show-mark-read"), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.markReadCancel = function(a, b) {
            this.select("viewConversationList").removeClass("show-mark-read"), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.showError = function(a, b) {
            this.select("errorTextSelector").html(b.message || b.error), this.select("errorContainerSelector").show(), this.trigger(this.select("toFieldSelector"), "uiTypeaheadIgnoreNextFocus"), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.hideError = function(a, b) {
            this.select("errorContainerSelector").hide()
        }, this.resetDMBox = function() {
            this.select("tweetBoxSelector").trigger("uiComposerResetAndFocus")
        }, this.setRecipientForNewConversationRender = function(a) {
            this.triggerEventOnUserSelector("uiSelectUser", {
                item: a
            })
        }, this.resetUserSelection = function() {
            this.triggerEventOnUserSelector("uiResetSelectedUserComposer"), this.triggerEventOnUserSelector("uiSetFocusInSelectedUserInput")
        }, this.triggerEventOnUserSelector = function(a, b) {
            this.trigger(this.attr.autocompleteParticipantSelector, a, b || {})
        }, this.isProfileLink = function(a) {
            return !!a.closest("a.js-user-profile-link").length
        }, this.previewMedia = function(a, b) {
            a.preventDefault();
            var c = $(a.target);
            this.trigger("uiDMDialogMediaPreview", {
                imgUrl: c.attr("data-full-img"),
                dmId: c.closest(".dm").attr("data-message-id") || c.closest(".DMInboxItem").attr("data-last-message-id")
            }), this.$dialogContainer.hide(), this.openState=!1
        }, this.showDMDialog = function(a, b) {
            this.$dialogContainer.show(), this.openState=!0
        }, this.restoreDialog = function() {
            this.isDialogNavigation && (this.isDialogNavigation=!1, this.open())
        }, this.dismissErrors = function() {
            this.trigger("uiHideMessage")
        }, this.refreshPageHandler = function() {
            this.refreshPage()
        }, this.refreshPage = function() {
            window.location.reload()
        }, this.deleteConversation = function(a, b) {
            this.deletedConversations = this.deletedConversations || {};
            var c = ".DMInboxItem[data-thread-id='" + b.conversation_id + "'] {display: none !important;}";
            this.deletedConversations[b.conversation_id] = this.insertCSSRule(this.stylesheet, c)
        }, this.deleteConversationFailure = function(a, b) {
            this.deletedConversations = this.deletedConversations || {};
            var c = this.deletedConversations[b.conversation_id];
            c != null && this.removeCSSRule(this.stylesheet, c)
        }, this.deleteMessage = function(a, b) {
            this.deletedMessages = this.deletedMessages || {};
            var c = ".dm-convo [data-message-id='" + b.id + "'] {display: none !important;}";
            this.deletedMessages[b.id] = this.insertCSSRule(this.stylesheet, c)
        }, this.deleteMessageFailure = function(a, b) {
            this.deletedMessages = this.deletedMessages || {};
            var c = this.deletedMessages[b.id];
            c != null && this.removeCSSRule(this.stylesheet, c)
        }, this.updateDMRecipientStatus = function(a, b) {
            a.type == "uiUpdatedSelectedUsers" ? this.dmRecipientSelected = b && b.items.length != 0 : this.dmRecipientTyped = b && b.text != "", this.triggerUiDMRecipientChanged()
        }, this.triggerUiDMRecipientChanged = function() {
            var a = Boolean(this.dmRecipientSelected || this.dmRecipientTyped);
            this.trigger("uiDMRecipientChanged", {
                hasValidRecipient: a
            })
        }, this.after("initialize", function() {
            this.$dialogContainer = this.select("dialogSelector"), this.viewClasses = [this.attr.classConversationList, this.attr.classConversation, this.attr.classNew, this.attr.classViewParticipants].join(" "), this.on(document, "uiShowProfilePopup", this.handleProfileLinkClick), this.on(document, "uiCloseProfilePopup", this.restoreDialog), this.on(document, "uiNeedsDMDialog", this.openDialog), this.on(document, "uiCloseDMDialog", this.closeDialog), this.on(document, "uiOpenNewDM", this.renderNewView), this.on(document, "uiViewParticipants", this.renderViewParticipants), this.on(document, "uiNavigate", "uiCloseDMDialog"), this.on("uiUpdatedSelectedUsers", this.possiblyRenderConversation), this.on(document, "dataDMConversationListResult", this.updateConversationList), this.on(document, "dataDMConversationEmptyState", this.clearConversationList), this.on(document, "dataDMSendSuccess dataDMDeleteSuccess", "dataDMSuccess"), this.on(document, "dataDMConversationResult dataDMSendSuccess dataDMDeleteSuccess", this.updateConversation), this.on(document, "dataConversationParticipantsResult", this.updateViewParticipants), this.on(document, "uiSendDM dataDMMediaUploadSuccess", this.sendMessage), this.on(document, "dataDMError", this.showError), this.on(document, "dataDMReadSuccess", this.updateReadState), this.on(document, "dataDMReadError", this.handleReadStateFailure), this.on(document, "dataNotificationsReceived", this.refreshDMList), this.on(document, "uiDMDialogMediaPreviewClosed", this.showDMDialog), this.on(this.attr.autocompleteParticipantSelector, "uiTextChanged", this.updateDMRecipientStatus), this.on("uiUpdatedSelectedUsers", this.updateDMRecipientStatus), this.stylesheet = this.createStyleSheet("deleted-direct-messages"), this.on(document, "uiDeleteConversation", "uiNeedsDMDialog"), this.on(document, "uiDeleteConversation", this.deleteConversation), this.on(document, "dataConversationDeleteFailed", this.deleteConversationFailure), this.on(document, "uiDMDialogDeleteMessage", this.deleteMessage), this.on(document, "dataDMDeleteFailed", this.deleteMessageFailure), this.on(document, "uiDMInboxWantsRefreshed", this.inboxNeedsRefresh), this.on("uiDialogClosed", this.dismissErrors), this.on("click", {
                linksForConversationListView: this.renderConversationListView,
                linksForConversationView: this.renderConversationView,
                linksForNewView: this.renderNewView,
                errorCloseSelector: this.hideError,
                errorReloadSelector: this.refreshPageHandler,
                markAllReadSelector: this.showMarkReadConfirm,
                markReadConfirmSelector: this.markAllMessages,
                markReadCancelSelector: this.markReadCancel,
                mediaPreviewSelector: this.previewMedia
            }), this.unreadCount = 0
        })
    }
    var defineComponent = require("core/component"), StringUtils = require("app/utils/string"), withDialog = require("app/ui/with_dialog"), withDynamicStylesheet = require("app/ui/with_dynamic_stylesheet"), withItemActions = require("app/ui/with_item_actions"), withTimestampUpdating = require("app/ui/with_timestamp_updating"), withTweetboxInitialization = require("app/ui/with_tweetbox_initialization"), withUserActions = require("app/ui/with_user_actions"), DirectMessageDialog = defineComponent(directMessageDialog, withDialog, withTimestampUpdating, withItemActions, withTweetboxInitialization, withDynamicStylesheet, withUserActions), conversationCache = {};
    module.exports = DirectMessageDialog
});
define("app/ui/direct_message_link_handler", ["module", "require", "exports", "core/component", "app/utils/user_dom_data"], function(module, require, exports) {
    function directMessageLinkHandler() {
        this.defaultAttrs({
            dmLinkSelector: ".js-dm-dialog, .dm-button, .global-dm-nav"
        }), this.openDialog = function(a, b) {
            a.preventDefault();
            var c = $(b.el);
            b = {
                id: userDomDataUtil.getId(c),
                screen_name: userDomDataUtil.getScreenName(c),
                name: userDomDataUtil.getName(c)
            }, this.trigger("uiNeedsDMDialog", b)
        }, this.after("initialize", function() {
            this.on(document, "click", {
                dmLinkSelector: this.openDialog
            })
        })
    }
    var defineComponent = require("core/component"), userDomDataUtil = require("app/utils/user_dom_data");
    module.exports = defineComponent(directMessageLinkHandler)
});
define("app/ui/direct_message_empty_state", ["module", "require", "exports", "core/component", "app/ui/direct_message_link_handler"], function(module, require, exports) {
    function directMessageEmptyState() {
        this.defaultAttrs({
            emptyStateClass: "dm-empty-state",
            noMessagesSelector: ".dm-no-messages",
            dmLinkSelector: ".DMEmptyState-suggestions .js-account-group"
        }), this.showEmptyState = function(a, b) {
            this.$node.addClass(this.attr.emptyStateClass), this.select("noMessagesSelector").html(b.html).show()
        }, this.hideEmptyState = function() {
            this.$node.removeClass(this.attr.emptyStateClass), this.select("noMessagesSelector").hide()
        }, this.after("initialize", function() {
            DirectMessageLinkHandler.attachTo(this.$node, {
                dmLinkSelector: this.attr.dmLinkSelector
            }), this.on(document, "dataDMConversationEmptyState", this.showEmptyState), this.on("uiDialogClosed", this.hideEmptyState)
        })
    }
    var defineComponent = require("core/component"), DirectMessageLinkHandler = require("app/ui/direct_message_link_handler");
    module.exports = defineComponent(directMessageEmptyState)
});
define("app/utils/image/image_resizer", ["module", "require", "exports"], function(module, require, exports) {
    var imageResizer = function() {
        var a = 300, b = 520, c = 50;
        return {
            galleryWidth: function(a, b) {
                var d = $(window).width() - 2 * c, e = Math.min(d, a.width());
                return Math.max(b.width(), e)
            },
            resetMinSize: function(c) {
                this.galW = b, this.galH = a, c.width(this.galW), c.css("min-height", this.galH)
            },
            resizeMedia: function(a, b, d, e, f) {
                var g = b.find(e).add(b.find(".modal-header")).map(function() {
                    return $(this).outerHeight()
                }).get().reduce(function(a, b) {
                    return a + b
                }), h = $(window).height() - 2 * c, i = $(window).width() - 2 * c, j = this.galH, k = this.galW, l = h - g, m = i, n = a.height(), o = a.width();
                d && (o += 130, n += 100), n > l && (o = Math.ceil(o * (l / n)), n = l, a.width(o).height(n)), o > m && (o = m, n = Math.ceil(n * (m / o)), a.width(o).height(n));
                var p = b.find(".GalleryNav").hasClass("enabled");
                if (!p || o > this.galW)
                    this.galW = o, b.width(this.galW);
                n + g > this.galH ? (this.galH = n + g, b.css("min-height", this.galH), a.css("margin-top", 0)) : f ? b.css("min-height", 0) : (a.css("margin-top", (this.galH - g - n) / 2), a.css("margin-bottom", (this.galH - g - n) / 2))
            }
        }
    }();
    module.exports = imageResizer
});
define("app/ui/direct_message_media_preview", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/ui/with_scrollbar_width", "app/utils/image/image_resizer"], function(module, require, exports) {
    function directMessageMediaPreview() {
        this.defaultAttrs({
            dmImagePreviewSelector: ".dm-media-preview",
            dmPreviewSelector: ".dm-media"
        }), this.loadImage = function(a, b) {
            var c = $('<img class="dm-media-image"/>'), d = this.select("dmImagePreviewSelector");
            d.empty(), c.on("load", function(a) {
                d.empty().append(c), c.attr({
                    "data-height": c[0].height,
                    "data-width": c[0].width
                }), imageResizer.resizeMedia(c, this.select("dmPreviewSelector")), this.position();
                var e = parseFloat(c.css("margin-top"));
                e && c.css("margin-bottom", e), this.trigger("uiDMMediaLoaded", {
                    url: c.attr("src"),
                    id: b
                })
            }.bind(this)), c.on("error", function(a) {
                this.trigger("uiDMMediaFailed", {
                    url: c.attr("src"),
                    id: b
                })
            }.bind(this)), c.attr("src", a)
        }, this.openDialog = function(a, b) {
            this.calculateScrollbarWidth(), imageResizer.resetMinSize(this.select("dmPreviewSelector")), this.loadImage(b.imgUrl, b.dmId), this.open()
        }, this.renderConversation = function(a, b) {
            this.trigger("uiDMDialogMediaPreviewClosed")
        }, this.after("initialize", function() {
            this.on(document, "uiDMDialogMediaPreview", this.openDialog), this.on("uiDialogCloseRequested", this.renderConversation)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withScrollbarWidth = require("app/ui/with_scrollbar_width"), imageResizer = require("app/utils/image/image_resizer");
    module.exports = defineComponent(directMessageMediaPreview, withDialog, withScrollbarWidth)
});
define("app/data/direct_messages_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function directMessagesScribe() {
        this.after("initialize", function() {
            this.scribeOnEvent("uiComposeNewDMWithTweet", "share_via_dm"), this.scribeOnEvent("uiDMDialogOpenedNewConversation", "open"), this.scribeOnEvent("uiDMDialogOpenedConversation", "open"), this.scribeOnEvent("uiDMDialogOpenedConversationList", "open"), this.scribeOnEvent("uiDMDialogSendMessage", "send_dm"), this.scribeOnEvent("uiDMDialogSendMessageWithTweet", "send_tweet_dm"), this.scribeOnEvent("uiDMDialogDeleteMessage", "delete_dm"), this.scribeOnEvent("uiDMDialogMarkSpam", "report_as_spam"), this.scribeOnEvent("uiDMDialogMarkMessage", {
                element: "mark_all_as_read",
                action: "click"
            }), this.scribeOnEvent("uiDMMediaLoaded", "media_preview"), this.scribeOnEvent("uiDMMediaFailed", "media_preview")
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), DirectMessagesScribe = defineComponent(directMessagesScribe, withScribe);
    module.exports = DirectMessagesScribe
});
define("app/data/dm/update_name", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function updateName() {
        this.updateName = function(a, b) {
            this.post({
                url: "/i/direct_messages/update_name",
                data: {
                    id: b.conversation_id,
                    name: b.name
                },
                success: this.success.bind(this),
                error: "dataUpdateConversationNameFailure"
            })
        }, this.success = function(a) {
            this.trigger("dataDMConversationResult", a), this.trigger("dataUpdateConversationNameSuccess", a)
        }, this.after("initialize", function() {
            this.on("uiUpdateConversationName", this.updateName)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data");
    module.exports = defineComponent(updateName, withData)
});
define("app/ui/dm/direct_message_update_name", ["module", "require", "exports", "core/component", "lib/twitter-text"], function(module, require, exports) {
    function updateName() {
        this.attributes({
            confirmButtonSelector: ".confirm-name-pencil",
            conversationIdSelector: "[data-conversation-id]",
            displayedNameSelector: ".dm_dialog_real_name",
            editButtonSelector: ".edit-name-pencil",
            beginEditModeSelector: ".edit-name-pencil, .dm_dialog_real_name.edit-allowed",
            inputGroupSelector: ".input-group",
            maxCharacters: 30,
            spinnerSelector: ".spinner",
            textSelector: ".update-name-input",
            updateNameContainerSelector: ".dm-update-name-ui"
        }), this.getConversationId = function() {
            return this.select("conversationIdSelector").attr("data-conversation-id")
        }, this.getCurrentName = function() {
            return this.select("displayedNameSelector").data("raw-name")
        }, this.handleCommandInputs = function(a, b) {
            a.keyCode == 27 ? (a.stopPropagation(), this.hideEditMode()) : a.keyCode == 13 && (a.stopPropagation(), this.requestUpdateConversationName())
        }, this.showEditMode = function() {
            this.select("editButtonSelector").addClass("u-hidden"), this.select("displayedNameSelector").addClass("u-hidden"), this.select("inputGroupSelector").removeClass("u-hidden"), this.select("textSelector").focus().val(this.getCurrentName())
        }, this.hideEditMode = function(a, b) {
            this.select("editButtonSelector").removeClass("u-hidden"), this.select("displayedNameSelector").removeClass("u-hidden"), this.select("inputGroupSelector").addClass("u-hidden")
        }, this.showSpinner = function() {
            this.select("textSelector").unbind("blur"), this.hideEditMode(), this.select("editButtonSelector").addClass("u-hidden"), this.select("displayedNameSelector").addClass("u-hidden"), this.select("spinnerSelector").removeClass("u-hidden")
        }, this.hideSpinner = function() {
            this.on(this.attr.textSelector, "blur", this.hideEditMode), this.hideEditMode(), this.select("spinnerSelector").addClass("u-hidden"), this.select("editButtonSelector").removeClass("u-hidden"), this.select("displayedNameSelector").removeClass("u-hidden")
        }, this.requestUpdateConversationName = function() {
            if (!this.select("inputGroupSelector").hasClass("text-too-long")) {
                var a = this.select("textSelector").val();
                this.trigger("uiUpdateConversationName", {
                    conversation_id: this.getConversationId(),
                    name: a
                }), this.showSpinner()
            }
        }, this.updateDisplayedName = function(a, b) {
            this.select("displayedNameSelector").html(b.recipient.name), this.hideSpinner()
        }, this.prepareUpdateNameUI = function(a, b) {
            this.hideSpinner(), typeof b.is_oto == "undefined" || b.is_oto ? (this.select("updateNameContainerSelector").addClass("u-hidden"), this.select("displayedNameSelector").removeClass("edit-allowed")) : (this.select("updateNameContainerSelector").removeClass("u-hidden"), this.select("displayedNameSelector").addClass("edit-allowed"))
        }, this.handleInputChange = function(a, b) {
            var c = twitterText.getTweetLength(b.text);
            c > this.attr.maxCharacters ? this.select("inputGroupSelector").addClass("text-too-long") : this.select("inputGroupSelector").removeClass("text-too-long")
        }, this.detectInputChange = function() {
            var a = this.select("textSelector").val();
            this.previousText != a && (this.trigger("uiTextChanged", {
                text: a
            }), this.previousText = a)
        }, this.startPolling = function() {
            this.pollingId === undefined && (this.pollingId = setInterval(this.detectInputChange.bind(this), 100))
        }, this.stopPolling = function() {
            this.pollingId !== undefined && (clearInterval(this.pollingId), delete this.pollingId)
        }, this.after("initialize", function() {
            this.on("click", {
                beginEditModeSelector: this.showEditMode,
                confirmButtonSelector: this.requestUpdateConversationName
            }), this.on(this.attr.textSelector, "blur", this.hideEditMode), this.on(this.attr.textSelector, "keydown", this.handleCommandInputs), this.on(document, "uiPrepareUpdateConversationName", this.prepareUpdateNameUI), this.on(document, "dataUpdateConversationNameSuccess", this.updateDisplayedName), this.on(document, "dataUpdateConversationNameFailure", this.hideSpinner), this.on("uiTextChanged", this.handleInputChange), this.on(this.attr.textSelector, "focus", this.startPolling), this.on(this.attr.textSelector, "blur", this.stopPolling)
        })
    }
    var defineComponent = require("core/component"), twitterText = require("lib/twitter-text"), UpdateName = defineComponent(updateName);
    module.exports = UpdateName
});
define("app/ui/dm/suspicious_content", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function suspiciousContent() {
        this.defaultAttrs({
            warningSelector: ".suspicious-content-warning",
            messageSelector: ".dm",
            conversationSelector: ".js-dm-conversation",
            reportButtonsSelector: ".suspicious-content-response",
            reportAsSpamSelector: ".js-message-is-spam",
            reportAsSafeSelector: ".js-message-is-safe"
        }), this.showMessage = function(a, b) {
            a.preventDefault();
            var c = $(a.target), d = c.next(this.attr.messageSelector);
            c.hide(), d.show()
        }, this.reportMessage = function(a, b, c) {
            var d = $(b.target), e = d.closest(this.attr.messageSelector).attr("data-message-id"), f = d.closest(this.attr.conversationSelector).attr("data-thread-id");
            this.trigger("uiDMDialogReportDM", {
                id: e,
                screen_name: f,
                report_as: a
            }), this.select("reportButtonsSelector").hide(), this.trigger("uiDMInboxWantsRefreshed")
        }, this.after("initialize", function() {
            this.on("uiRevealSuspiciousDM", this.showMessage), this.on("uiReportDMAsSpam", this.reportMessage.bind(this, "spam")), this.on("uiReportDMAsSafe", this.reportMessage.bind(this, "not_spam")), this.on("click", {
                warningSelector: "uiRevealSuspiciousDM",
                reportAsSpamSelector: "uiReportDMAsSpam",
                reportAsSafeSelector: "uiReportDMAsSafe"
            })
        })
    }
    var defineComponent = require("core/component"), SuspiciousContent = defineComponent(suspiciousContent);
    module.exports = SuspiciousContent
});
define("app/data/dm/suspicious_content_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function suspiciousContentScribe() {
        this.after("initialize", function() {
            this.scribeOnEvent("uiRevealSuspiciousDM", {
                component: "rtf_message",
                action: "open"
            }), this.scribeOnEvent("uiReportDMAsSpam", {
                component: "rtf_message",
                action: "report_as_spam"
            }), this.scribeOnEvent("uiReportDMAsSafe", {
                component: "rtf_message",
                action: "report_as_ok"
            })
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), SuspiciousContentScribe = defineComponent(suspiciousContentScribe, withScribe);
    module.exports = SuspiciousContentScribe
});
define("app/data/dm/toggle_notifications", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function toggleNotifications() {
        this.toggleNotifications = function(a, b) {
            this.post({
                url: "/i/direct_messages/notifications",
                data: {
                    id: b.conversation_id,
                    enable_notifications: b.enable_notifications
                },
                eventData: b,
                success: this.success.bind(this, b.enable_notifications),
                error: b.enable_notifications ? "dataNotificationsEnableFailed": "dataNotificationsDisableFailed"
            })
        }, this.success = function(a, b) {
            var c = a ? "dataNotificationsEnabled": "dataNotificationsDisabled";
            this.trigger(c, b), this.trigger("dataDMConversationResult", b)
        }, this.after("initialize", function() {
            this.on("uiToggleNotifications", this.toggleNotifications)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), ToggleNotifications = defineComponent(toggleNotifications, withData);
    module.exports = ToggleNotifications
});
deferred('$lib/mediaelement.js', function() {
    function onYouTubePlayerAPIReady() {
        mejs.YouTubeApi.iFrameReady()
    }
    function onYouTubePlayerReady(a) {
        mejs.YouTubeApi.flashReady(a)
    }
    /*!
    * MediaElement.js
    * HTML5 <video> and <audio> shim and player
    * http://mediaelementjs.com/
    *
    * Creates a JavaScript object that mimics HTML5 MediaElement API
    * for browsers that don't understand HTML5 or can't play the provided codec
    * Can play MP4 (H.264), Ogg, WebM, FLV, WMV, WMA, ACC, and MP3
    *
    * Copyright 2010-2014, John Dyer (http://j.hn)
    * License: MIT
    *
    */
    var mejs = mejs || {};
    mejs.version = "2.14.2", mejs.meIndex = 0, mejs.plugins = {
        silverlight: [{
            version: [3, 0],
            types: ["video/mp4", "video/m4v", "video/mov", "video/wmv", "audio/wma", "audio/m4a", "audio/mp3", "audio/wav", "audio/mpeg"]
        }
        ],
        flash: [{
            version: [9, 0, 124],
            types: ["video/mp4", "video/m4v", "video/mov", "video/flv", "video/rtmp", "video/x-flv", "audio/flv", "audio/x-flv", "audio/mp3", "audio/m4a", "audio/mpeg", "video/youtube", "video/x-youtube"]
        }
        ],
        youtube: [{
            version: null,
            types: ["video/youtube", "video/x-youtube", "audio/youtube", "audio/x-youtube"]
        }
        ],
        vimeo: [{
            version: null,
            types: ["video/vimeo", "video/x-vimeo"]
        }
        ]
    }, mejs.Utility = {
        encodeUrl: function(a) {
            return encodeURIComponent(a)
        },
        escapeHTML: function(a) {
            return a.toString().split("&").join("&amp;").split("<").join("&lt;").split('"').join("&quot;")
        },
        absolutizeUrl: function(a) {
            var b = document.createElement("div");
            return b.innerHTML = '<a href="' + this.escapeHTML(a) + '">x</a>', b.firstChild.href
        },
        getScriptPath: function(a) {
            var b = 0, c, d = "", e = "", f, g, h, i, j, k = document.getElementsByTagName("script"), l = k.length, m = a.length;
            for (; b < l; b++) {
                h = k[b].src, f = h.lastIndexOf("/"), f>-1 ? (j = h.substring(f + 1), i = h.substring(0, f + 1)) : (j = h, i = "");
                for (c = 0; c < m; c++) {
                    e = a[c], g = j.indexOf(e);
                    if (g>-1) {
                        d = i;
                        break
                    }
                }
                if (d !== "")
                    break
            }
            return d
        },
        secondsToTimeCode: function(a, b, c, d) {
            typeof c == "undefined" ? c=!1 : typeof d == "undefined" && (d = 25);
            var e = Math.floor(a / 3600)%24, f = Math.floor(a / 60)%60, g = Math.floor(a%60), h = Math.floor((a%1 * d).toFixed(3)), i = (b || e > 0 ? (e < 10 ? "0" + e : e) + ":" : "") + (f < 10 ? "0" + f : f) + ":" + (g < 10 ? "0" + g : g) + (c ? ":" + (h < 10 ? "0" + h : h) : "");
            return i
        },
        timeCodeToSeconds: function(a, b, c, d) {
            typeof c == "undefined" ? c=!1 : typeof d == "undefined" && (d = 25);
            var e = a.split(":"), f = parseInt(e[0], 10), g = parseInt(e[1], 10), h = parseInt(e[2], 10), i = 0, j = 0;
            return c && (i = parseInt(e[3]) / d), j = f * 3600 + g * 60 + h + i, j
        },
        convertSMPTEtoSeconds: function(a) {
            if (typeof a != "string")
                return !1;
            a = a.replace(",", ".");
            var b = 0, c = a.indexOf(".")!=-1 ? a.split(".")[1].length: 0, d = 1;
            a = a.split(":").reverse();
            for (var e = 0; e < a.length; e++)
                d = 1, e > 0 && (d = Math.pow(60, e)), b += Number(a[e]) * d;
            return Number(b.toFixed(c))
        },
        removeSwf: function(a) {
            var b = document.getElementById(a);
            b && /object|embed/i.test(b.nodeName) && (mejs.MediaFeatures.isIE ? (b.style.display = "none", function() {
                b.readyState == 4 ? mejs.Utility.removeObjectInIE(a) : setTimeout(arguments.callee, 10)
            }()) : b.parentNode.removeChild(b))
        },
        removeObjectInIE: function(a) {
            var b = document.getElementById(a);
            if (b) {
                for (var c in b)
                    typeof b[c] == "function" && (b[c] = null);
                b.parentNode.removeChild(b)
            }
        }
    }, mejs.PluginDetector = {
        hasPluginVersion: function(a, b) {
            var c = this.plugins[a];
            return b[1] = b[1] || 0, b[2] = b[2] || 0, c[0] > b[0] || c[0] == b[0] && c[1] > b[1] || c[0] == b[0] && c[1] == b[1] && c[2] >= b[2]?!0 : !1
        },
        nav: window.navigator,
        ua: window.navigator.userAgent.toLowerCase(),
        plugins: [],
        addPlugin: function(a, b, c, d, e) {
            this.plugins[a] = this.detectPlugin(b, c, d, e)
        },
        detectPlugin: function(a, b, c, d) {
            var e = [0, 0, 0], f, g, h;
            if (typeof this.nav.plugins != "undefined" && typeof this.nav.plugins[a] == "object") {
                f = this.nav.plugins[a].description;
                if (f && (typeof this.nav.mimeTypes == "undefined" ||!this.nav.mimeTypes[b]||!!this.nav.mimeTypes[b].enabledPlugin)) {
                    e = f.replace(a, "").replace(/^\s+/, "").replace(/\sr/gi, ".").split(".");
                    for (g = 0; g < e.length; g++)
                        e[g] = parseInt(e[g].match(/\d+/), 10)
                    }
            } else if (typeof window.ActiveXObject != "undefined")
                try {
                    h = new ActiveXObject(c), h && (e = d(h))
            } catch (i) {}
            return e
        }
    }, mejs.PluginDetector.addPlugin("flash", "Shockwave Flash", "application/x-shockwave-flash", "ShockwaveFlash.ShockwaveFlash", function(a) {
        var b = [], c = a.GetVariable("$version");
        return c && (c = c.split(" ")[1].split(","), b = [parseInt(c[0], 10), parseInt(c[1], 10), parseInt(c[2], 10)]), b
    }), mejs.PluginDetector.addPlugin("silverlight", "Silverlight Plug-In", "application/x-silverlight-2", "AgControl.AgControl", function(a) {
        var b = [0, 0, 0, 0], c = function(a, b, c, d) {
            while (a.isVersionSupported(b[0] + "." + b[1] + "." + b[2] + "." + b[3]))
                b[c] += d;
            b[c] -= d
        };
        return c(a, b, 0, 1), c(a, b, 1, 1), c(a, b, 2, 1e4), c(a, b, 2, 1e3), c(a, b, 2, 100), c(a, b, 2, 10), c(a, b, 2, 1), c(a, b, 3, 1), b
    }), mejs.MediaFeatures = {
        init: function() {
            var a = this, b = document, c = mejs.PluginDetector.nav, d = mejs.PluginDetector.ua.toLowerCase(), e, f, g = ["source", "track", "audio", "video"];
            a.isiPad = d.match(/ipad/i) !== null, a.isiPhone = d.match(/iphone/i) !== null, a.isiOS = a.isiPhone || a.isiPad, a.isAndroid = d.match(/android/i) !== null, a.isBustedAndroid = d.match(/android 2\.[12]/) !== null, a.isBustedNativeHTTPS = location.protocol === "https:" && (d.match(/android [12]\./) !== null || d.match(/macintosh.* version.* safari/) !== null), a.isIE = c.appName.toLowerCase().indexOf("microsoft")!=-1 || c.appName.toLowerCase().match(/trident/gi) !== null, a.isChrome = d.match(/chrome/gi) !== null, a.isFirefox = d.match(/firefox/gi) !== null, a.isWebkit = d.match(/webkit/gi) !== null, a.isGecko = d.match(/gecko/gi) !== null&&!a.isWebkit&&!a.isIE, a.isOpera = d.match(/opera/gi) !== null, a.hasTouch = "ontouchstart"in window, a.svg=!!document.createElementNS&&!!document.createElementNS("http://www.w3.org/2000/svg", "svg").createSVGRect;
            for (e = 0; e < g.length; e++)
                f = document.createElement(g[e]);
            a.supportsMediaTag = typeof f.canPlayType != "undefined" || a.isBustedAndroid;
            try {
                f.canPlayType("video/mp4")
            } catch (h) {
                a.supportsMediaTag=!1
            }
            a.hasSemiNativeFullScreen = typeof f.webkitEnterFullscreen != "undefined", a.hasNativeFullscreen = typeof f.requestFullscreen != "undefined", a.hasWebkitNativeFullScreen = typeof f.webkitRequestFullScreen != "undefined", a.hasMozNativeFullScreen = typeof f.mozRequestFullScreen != "undefined", a.hasMsNativeFullScreen = typeof f.msRequestFullscreen != "undefined", a.hasTrueNativeFullScreen = a.hasWebkitNativeFullScreen || a.hasMozNativeFullScreen || a.hasMsNativeFullScreen, a.nativeFullScreenEnabled = a.hasTrueNativeFullScreen, a.hasMozNativeFullScreen ? a.nativeFullScreenEnabled = document.mozFullScreenEnabled : a.hasMsNativeFullScreen && (a.nativeFullScreenEnabled = document.msFullscreenEnabled), a.isChrome && (a.hasSemiNativeFullScreen=!1), a.hasTrueNativeFullScreen && (a.fullScreenEventName = "", a.hasWebkitNativeFullScreen ? a.fullScreenEventName = "webkitfullscreenchange" : a.hasMozNativeFullScreen ? a.fullScreenEventName = "mozfullscreenchange" : a.hasMsNativeFullScreen && (a.fullScreenEventName = "MSFullscreenChange"), a.isFullScreen = function() {
                if (f.mozRequestFullScreen)
                    return b.mozFullScreen;
                if (f.webkitRequestFullScreen)
                    return b.webkitIsFullScreen;
                if (f.hasMsNativeFullScreen)
                    return b.msFullscreenElement !== null
            }, a.requestFullScreen = function(b) {
                a.hasWebkitNativeFullScreen ? b.webkitRequestFullScreen() : a.hasMozNativeFullScreen ? b.mozRequestFullScreen() : a.hasMsNativeFullScreen && b.msRequestFullscreen()
            }, a.cancelFullScreen = function() {
                a.hasWebkitNativeFullScreen ? document.webkitCancelFullScreen() : a.hasMozNativeFullScreen ? document.mozCancelFullScreen() : a.hasMsNativeFullScreen && document.msExitFullscreen()
            }), a.hasSemiNativeFullScreen && d.match(/mac os x 10_5/i) && (a.hasNativeFullScreen=!1, a.hasSemiNativeFullScreen=!1)
        }
    }, mejs.MediaFeatures.init(), mejs.HtmlMediaElement = {
        pluginType: "native",
        isFullScreen: !1,
        setCurrentTime: function(a) {
            this.currentTime = a
        },
        setMuted: function(a) {
            this.muted = a
        },
        setVolume: function(a) {
            this.volume = a
        },
        stop: function() {
            this.pause()
        },
        setSrc: function(a) {
            var b = this.getElementsByTagName("source");
            while (b.length > 0)
                this.removeChild(b[0]);
            if (typeof a == "string")
                this.src = a;
            else {
                var c, d;
                for (c = 0; c < a.length; c++) {
                    d = a[c];
                    if (this.canPlayType(d.type)) {
                        this.src = d.src;
                        break
                    }
                }
            }
        },
        setVideoSize: function(a, b) {
            this.width = a, this.height = b
        }
    }, mejs.PluginMediaElement = function(a, b, c) {
        this.id = a, this.pluginType = b, this.src = c, this.events = {}, this.attributes = {}
    }, mejs.PluginMediaElement.prototype = {
        pluginElement: null,
        pluginType: "",
        isFullScreen: !1,
        playbackRate: - 1,
        defaultPlaybackRate: - 1,
        seekable: [],
        played: [],
        paused: !0,
        ended: !1,
        seeking: !1,
        duration: 0,
        error: null,
        tagName: "",
        muted: !1,
        volume: 1,
        currentTime: 0,
        play: function() {
            this.pluginApi != null && (this.pluginType == "youtube" || this.pluginType == "vimeo" ? this.pluginApi.playVideo() : this.pluginApi.playMedia(), this.paused=!1)
        },
        load: function() {
            this.pluginApi != null && (this.pluginType != "youtube" && this.pluginType != "vimeo" && this.pluginApi.loadMedia(), this.paused=!1)
        },
        pause: function() {
            this.pluginApi != null && (this.pluginType == "youtube" || this.pluginType == "vimeo" ? this.pluginApi.pauseVideo() : this.pluginApi.pauseMedia(), this.paused=!0)
        },
        stop: function() {
            this.pluginApi != null && (this.pluginType == "youtube" || this.pluginType == "vimeo" ? this.pluginApi.stopVideo() : this.pluginApi.stopMedia(), this.paused=!0)
        },
        canPlayType: function(a) {
            var b, c, d, e = mejs.plugins[this.pluginType];
            for (b = 0; b < e.length; b++) {
                d = e[b];
                if (mejs.PluginDetector.hasPluginVersion(this.pluginType, d.version))
                    for (c = 0; c < d.types.length; c++)
                        if (a == d.types[c])
                            return "probably"
            }
            return ""
        },
        positionFullscreenButton: function(a, b, c) {
            this.pluginApi != null && this.pluginApi.positionFullscreenButton && this.pluginApi.positionFullscreenButton(Math.floor(a), Math.floor(b), c)
        },
        hideFullscreenButton: function() {
            this.pluginApi != null && this.pluginApi.hideFullscreenButton && this.pluginApi.hideFullscreenButton()
        },
        setSrc: function(a) {
            if (typeof a == "string")
                this.pluginApi.setSrc(mejs.Utility.absolutizeUrl(a)), this.src = mejs.Utility.absolutizeUrl(a);
            else {
                var b, c;
                for (b = 0; b < a.length; b++) {
                    c = a[b];
                    if (this.canPlayType(c.type)) {
                        this.pluginApi.setSrc(mejs.Utility.absolutizeUrl(c.src)), this.src = mejs.Utility.absolutizeUrl(a);
                        break
                    }
                }
            }
        },
        setCurrentTime: function(a) {
            this.pluginApi != null && (this.pluginType == "youtube" || this.pluginType == "vimeo" ? this.pluginApi.seekTo(a) : this.pluginApi.setCurrentTime(a), this.currentTime = a)
        },
        setVolume: function(a) {
            this.pluginApi != null && (this.pluginType == "youtube" || this.pluginType == "vimeo" ? this.pluginApi.setVolume(a * 100) : this.pluginApi.setVolume(a), this.volume = a)
        },
        setMuted: function(a) {
            this.pluginApi != null && (this.pluginType == "youtube" || this.pluginType == "vimeo" ? (a ? this.pluginApi.mute() : this.pluginApi.unMute(), this.muted = a, this.dispatchEvent("volumechange")) : this.pluginApi.setMuted(a), this.muted = a)
        },
        setVideoSize: function(a, b) {
            this.pluginElement.style && (this.pluginElement.style.width = a + "px", this.pluginElement.style.height = b + "px"), this.pluginApi != null && this.pluginApi.setVideoSize && this.pluginApi.setVideoSize(a, b)
        },
        setFullscreen: function(a) {
            this.pluginApi != null && this.pluginApi.setFullscreen && this.pluginApi.setFullscreen(a)
        },
        enterFullScreen: function() {
            this.pluginApi != null && this.pluginApi.setFullscreen && this.setFullscreen(!0)
        },
        exitFullScreen: function() {
            this.pluginApi != null && this.pluginApi.setFullscreen && this.setFullscreen(!1)
        },
        addEventListener: function(a, b, c) {
            this.events[a] = this.events[a] || [], this.events[a].push(b)
        },
        removeEventListener: function(a, b) {
            if (!a)
                return this.events = {}, !0;
            var c = this.events[a];
            if (!c)
                return !0;
            if (!b)
                return this.events[a] = [], !0;
            for (var d = 0; d < c.length; d++)
                if (c[d] === b)
                    return this.events[a].splice(d, 1), !0;
            return !1
        },
        dispatchEvent: function(a) {
            var b, c, d = this.events[a];
            if (d) {
                c = Array.prototype.slice.call(arguments, 1);
                for (b = 0; b < d.length; b++)
                    d[b].apply(null, c)
            }
        },
        hasAttribute: function(a) {
            return a in this.attributes
        },
        removeAttribute: function(a) {
            delete this.attributes[a]
        },
        getAttribute: function(a) {
            return this.hasAttribute(a) ? this.attributes[a] : ""
        },
        setAttribute: function(a, b) {
            this.attributes[a] = b
        },
        remove: function() {
            mejs.Utility.removeSwf(this.pluginElement.id), mejs.MediaPluginBridge.unregisterPluginElement(this.pluginElement.id)
        }
    }, mejs.MediaPluginBridge = {
        pluginMediaElements: {},
        htmlMediaElements: {},
        registerPluginElement: function(a, b, c) {
            this.pluginMediaElements[a] = b, this.htmlMediaElements[a] = c
        },
        unregisterPluginElement: function(a) {
            delete this.pluginMediaElements[a], delete this.htmlMediaElements[a]
        },
        initPlugin: function(a) {
            var b = this.pluginMediaElements[a], c = this.htmlMediaElements[a];
            if (b) {
                switch (b.pluginType) {
                case"flash":
                    b.pluginElement = b.pluginApi = document.getElementById(a);
                    break;
                case"silverlight":
                    b.pluginElement = document.getElementById(b.id), b.pluginApi = b.pluginElement.Content.MediaElementJS
                }
                b.pluginApi != null && b.success && b.success(b, c)
            }
        },
        fireEvent: function(a, b, c) {
            var d, e, f, g = this.pluginMediaElements[a];
            if (!g)
                return;
            d = {
                type: b,
                target: g
            };
            for (e in c)
                g[e] = c[e], d[e] = c[e];
            f = c.bufferedTime || 0, d.target.buffered = d.buffered = {
                start: function(a) {
                    return 0
                },
                end: function(a) {
                    return f
                },
                length: 1
            }, g.dispatchEvent(d.type, d)
        }
    }, mejs.MediaElementDefaults = {
        mode: "auto",
        plugins: ["flash", "silverlight", "youtube", "vimeo"],
        enablePluginDebug: !1,
        httpsBasicAuthSite: !1,
        type: "",
        pluginPath: mejs.Utility.getScriptPath(["mediaelement.js", "mediaelement.min.js", "mediaelement-and-player.js", "mediaelement-and-player.min.js"]),
        flashName: "flashmediaelement.swf",
        flashStreamer: "",
        enablePluginSmoothing: !1,
        enablePseudoStreaming: !1,
        pseudoStreamingStartQueryParam: "start",
        silverlightName: "silverlightmediaelement.xap",
        defaultVideoWidth: 480,
        defaultVideoHeight: 270,
        pluginWidth: - 1,
        pluginHeight: - 1,
        pluginVars: [],
        timerRate: 250,
        startVolume: .8,
        success: function() {},
        error: function() {}
    }, mejs.MediaElement = function(a, b) {
        return mejs.HtmlMediaElementShim.create(a, b)
    }, mejs.HtmlMediaElementShim = {
        create: function(a, b) {
            var c = mejs.MediaElementDefaults, d = typeof a == "string" ? document.getElementById(a): a, e = d.tagName.toLowerCase(), f = e === "audio" || e === "video", g = f ? d.getAttribute("src"): d.getAttribute("href"), h = d.getAttribute("poster"), i = d.getAttribute("autoplay"), j = d.getAttribute("preload"), k = d.getAttribute("controls"), l, m;
            for (m in b)
                c[m] = b[m];
            return g = typeof g == "undefined" || g === null || g == "" ? null : g, h = typeof h == "undefined" || h === null ? "" : h, j = typeof j == "undefined" || j === null || j === "false" ? "none" : j, i = typeof i != "undefined" && i !== null && i !== "false", k = typeof k != "undefined" && k !== null && k !== "false", l = this.determinePlayback(d, c, mejs.MediaFeatures.supportsMediaTag, f, g), l.url = l.url !== null ? mejs.Utility.absolutizeUrl(l.url) : "", l.method == "native" ? (mejs.MediaFeatures.isBustedAndroid && (d.src = l.url, d.addEventListener("click", function() {
                d.play()
            }, !1)), this.updateNative(l, c, i, j)) : l.method !== "" ? this.createPlugin(l, c, h, i, j, k) : (this.createErrorMessage(l, c, h), this)
        },
        determinePlayback: function(a, b, c, d, e) {
            var f = [], g, h, i, j, k, l, m = {
                method: "",
                url: "",
                htmlMediaElement: a,
                isVideo: a.tagName.toLowerCase() != "audio"
            }, n, o, p, q, r;
            if (typeof b.type != "undefined" && b.type !== "")
                if (typeof b.type == "string")
                    f.push({
                        type: b.type,
                        url: e
                    });
                else 
                    for (g = 0; g < b.type.length; g++)
                        f.push({
                            type: b.type[g],
                            url: e
                        });
            else if (e !== null)
                l = this.formatType(e, a.getAttribute("type")), f.push({
                    type: l,
                    url: e
                });
            else 
                for (g = 0; g < a.childNodes.length; g++)
                    k = a.childNodes[g], k.nodeType == 1 && k.tagName.toLowerCase() == "source" && (e = k.getAttribute("src"), l = this.formatType(e, k.getAttribute("type")), r = k.getAttribute("media"), (!r ||!window.matchMedia || window.matchMedia && window.matchMedia(r).matches) && f.push({
                        type: l,
                        url: e
                    }));
            !d && f.length > 0 && f[0].url !== null && this.getTypeFromFile(f[0].url).indexOf("audio")>-1 && (m.isVideo=!1), mejs.MediaFeatures.isBustedAndroid && (a.canPlayType = function(a) {
                return a.match(/video\/(mp4|m4v)/gi) !== null ? "maybe" : ""
            });
            if (c && (b.mode === "auto" || b.mode === "auto_plugin" || b.mode === "native") && (!mejs.MediaFeatures.isBustedNativeHTTPS || b.httpsBasicAuthSite!==!0)) {
                d || (q = document.createElement(m.isVideo ? "video" : "audio"), a.parentNode.insertBefore(q, a), a.style.display = "none", m.htmlMediaElement = a = q);
                for (g = 0; g < f.length; g++)
                    if (a.canPlayType(f[g].type).replace(/no/, "") !== "" || a.canPlayType(f[g].type.replace(/mp3/, "mpeg")).replace(/no/, "") !== "" || a.canPlayType(f[g].type.replace(/m4a/, "mp4")).replace(/no/, "") !== "") {
                        m.method = "native", m.url = f[g].url;
                        break
                    }
                if (m.method === "native") {
                    m.url !== null && (a.src = m.url);
                    if (b.mode !== "auto_plugin")
                        return m
                }
            }
            if (b.mode === "auto" || b.mode === "auto_plugin" || b.mode === "shim")
                for (g = 0; g < f.length; g++) {
                    l = f[g].type;
                    for (h = 0; h < b.plugins.length; h++) {
                        n = b.plugins[h], o = mejs.plugins[n];
                        for (i = 0; i < o.length; i++) {
                            p = o[i];
                            if (p.version == null || mejs.PluginDetector.hasPluginVersion(n, p.version))
                                for (j = 0; j < p.types.length; j++)
                                    if (l == p.types[j])
                                        return m.method = n, m.url = f[g].url, m
                        }
                    }
                }
            return b.mode === "auto_plugin" && m.method === "native" ? m : (m.method === "" && f.length > 0 && (m.url = f[0].url), m)
        },
        formatType: function(a, b) {
            var c;
            return a&&!b ? this.getTypeFromFile(a) : b&&~b.indexOf(";") ? b.substr(0, b.indexOf(";")) : b
        },
        getTypeFromFile: function(a) {
            a = a.split("?")[0];
            var b = a.substring(a.lastIndexOf(".") + 1).toLowerCase();
            return (/(mp4|m4v|ogg|ogv|webm|webmv|flv|wmv|mpeg|mov)/gi.test(b) ? "video" : "audio") + "/" + this.getTypeFromExtension(b)
        },
        getTypeFromExtension: function(a) {
            switch (a) {
            case"mp4":
            case"m4v":
            case"m4a":
                return "mp4";
            case"webm":
            case"webma":
            case"webmv":
                return "webm";
            case"ogg":
            case"oga":
            case"ogv":
                return "ogg";
            default:
                return a
            }
        },
        createErrorMessage: function(a, b, c) {
            var d = a.htmlMediaElement, e = document.createElement("div");
            e.className = "me-cannotplay";
            try {
                e.style.width = d.width + "px", e.style.height = d.height + "px"
            } catch (f) {}
            b.customError ? e.innerHTML = b.customError : e.innerHTML = c !== "" ? '<a href="' + a.url + '"><img src="' + c + '" width="100%" height="100%" /></a>' : '<a href="' + a.url + '"><span>' + mejs.i18n.t("Download File") + "</span></a>", d.parentNode.insertBefore(e, d), d.style.display = "none", b.error(d)
        },
        createPlugin: function(a, b, c, d, e, f) {
            var g = a.htmlMediaElement, h = 1, i = 1, j = "me_" + a.method + "_" + mejs.meIndex++, k = new mejs.PluginMediaElement(j, a.method, a.url), l = document.createElement("div"), m, n, o;
            k.tagName = g.tagName;
            for (var p = 0; p < g.attributes.length; p++) {
                var q = g.attributes[p];
                q.specified == 1 && k.setAttribute(q.name, q.value)
            }
            n = g.parentNode;
            while (n !== null && n.tagName.toLowerCase() !== "body" && n.parentNode != null) {
                if (n.parentNode.tagName.toLowerCase() === "p") {
                    n.parentNode.parentNode.insertBefore(n, n.parentNode);
                    break
                }
                n = n.parentNode
            }
            a.isVideo ? (h = b.pluginWidth > 0 ? b.pluginWidth : b.videoWidth > 0 ? b.videoWidth : g.getAttribute("width") !== null ? g.getAttribute("width") : b.defaultVideoWidth, i = b.pluginHeight > 0 ? b.pluginHeight : b.videoHeight > 0 ? b.videoHeight : g.getAttribute("height") !== null ? g.getAttribute("height") : b.defaultVideoHeight, h = mejs.Utility.encodeUrl(h), i = mejs.Utility.encodeUrl(i)) : b.enablePluginDebug && (h = 320, i = 240), k.success = b.success, mejs.MediaPluginBridge.registerPluginElement(j, k, g), l.className = "me-plugin", l.id = j + "_container", a.isVideo ? g.parentNode.insertBefore(l, g) : document.body.insertBefore(l, document.body.childNodes[0]), o = ["id=" + j, "isvideo=" + (a.isVideo ? "true" : "false"), "autoplay=" + (d ? "true" : "false"), "preload=" + e, "width=" + h, "startvolume=" + b.startVolume, "timerrate=" + b.timerRate, "flashstreamer=" + b.flashStreamer, "height=" + i, "pseudostreamstart=" + b.pseudoStreamingStartQueryParam], a.url !== null && (a.method == "flash" ? o.push("file=" + mejs.Utility.encodeUrl(a.url)) : o.push("file=" + a.url)), b.enablePluginDebug && o.push("debug=true"), b.enablePluginSmoothing && o.push("smoothing=true"), b.enablePseudoStreaming && o.push("pseudostreaming=true"), f && o.push("controls=true"), b.pluginVars && (o = o.concat(b.pluginVars));
            switch (a.method) {
            case"silverlight":
                l.innerHTML = '<object data="data:application/x-silverlight-2," type="application/x-silverlight-2" id="' + j + '" name="' + j + '" width="' + h + '" height="' + i + '" class="mejs-shim">' + '<param name="initParams" value="' + o.join(",") + '" />' + '<param name="windowless" value="true" />' + '<param name="background" value="black" />' + '<param name="minRuntimeVersion" value="3.0.0.0" />' + '<param name="autoUpgrade" value="true" />' + '<param name="source" value="' + b.pluginPath + b.silverlightName + '" />' + "</object>";
                break;
            case"flash":
                mejs.MediaFeatures.isIE ? (m = document.createElement("div"), l.appendChild(m), m.outerHTML = '<object classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" codebase="//download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab" id="' + j + '" width="' + h + '" height="' + i + '" class="mejs-shim">' + '<param name="movie" value="' + b.pluginPath + b.flashName + "?x=" + new Date + '" />' + '<param name="flashvars" value="' + o.join("&amp;") + '" />' + '<param name="quality" value="high" />' + '<param name="bgcolor" value="#000000" />' + '<param name="wmode" value="transparent" />' + '<param name="allowScriptAccess" value="always" />' + '<param name="allowFullScreen" value="true" />' + '<param name="scale" value="default" />' + "</object>") : l.innerHTML = '<embed id="' + j + '" name="' + j + '" ' + 'play="true" ' + 'loop="false" ' + 'quality="high" ' + 'bgcolor="#000000" ' + 'wmode="transparent" ' + 'allowScriptAccess="always" ' + 'allowFullScreen="true" ' + 'type="application/x-shockwave-flash" pluginspage="//www.macromedia.com/go/getflashplayer" ' + 'src="' + b.pluginPath + b.flashName + '" ' + 'flashvars="' + o.join("&") + '" ' + 'width="' + h + '" ' + 'height="' + i + '" ' + 'scale="default"' + 'class="mejs-shim"></embed>';
                break;
            case"youtube":
                var r;
                a.url.lastIndexOf("youtu.be")!=-1 ? (r = a.url.substr(a.url.lastIndexOf("/") + 1), r.indexOf("?")!=-1 && (r = r.substr(0, r.indexOf("?")))) : r = a.url.substr(a.url.lastIndexOf("=") + 1), youtubeSettings = {
                    container: l,
                    containerId: l.id,
                    pluginMediaElement: k,
                    pluginId: j,
                    videoId: r,
                    height: i,
                    width: h
                }, mejs.PluginDetector.hasPluginVersion("flash", [10, 0, 0]) ? mejs.YouTubeApi.createFlash(youtubeSettings) : mejs.YouTubeApi.enqueueIframe(youtubeSettings);
                break;
            case"vimeo":
                var s = j + "_player";
                k.vimeoid = a.url.substr(a.url.lastIndexOf("/") + 1), l.innerHTML = '<iframe src="//player.vimeo.com/video/' + k.vimeoid + "?api=1&portrait=0&byline=0&title=0&player_id=" + s + '" width="' + h + '" height="' + i + '" frameborder="0" class="mejs-shim" id="' + s + '"></iframe>';
                if (typeof $f == "function") {
                    var t = $f(l.childNodes[0]);
                    t.addEvent("ready", function() {
                        function a(a, b, c, d) {
                            var e = {
                                type: c,
                                target: b
                            };
                            c == "timeupdate" && (b.currentTime = e.currentTime = d.seconds, b.duration = e.duration = d.duration), b.dispatchEvent(e.type, e)
                        }
                        t.playVideo = function() {
                            t.api("play")
                        }, t.pauseVideo = function() {
                            t.api("pause")
                        }, t.seekTo = function(a) {
                            t.api("seekTo", a)
                        }, t.addEvent("play", function() {
                            a(t, k, "play"), a(t, k, "playing")
                        }), t.addEvent("pause", function() {
                            a(t, k, "pause")
                        }), t.addEvent("finish", function() {
                            a(t, k, "ended")
                        }), t.addEvent("playProgress", function(b) {
                            a(t, k, "timeupdate", b)
                        }), k.pluginApi = t, mejs.MediaPluginBridge.initPlugin(j)
                    })
                } else 
                    console.warn("You need to include froogaloop for vimeo to work")
            }
            return g.style.display = "none", g.removeAttribute("autoplay"), k
        },
        updateNative: function(a, b, c, d) {
            var e = a.htmlMediaElement, f;
            for (f in mejs.HtmlMediaElement)
                e[f] = mejs.HtmlMediaElement[f];
            return b.success(e, e), e
        }
    }, mejs.YouTubeApi = {
        isIframeStarted: !1,
        isIframeLoaded: !1,
        loadIframeApi: function() {
            if (!this.isIframeStarted) {
                var a = document.createElement("script");
                a.src = "//www.youtube.com/player_api";
                var b = document.getElementsByTagName("script")[0];
                b.parentNode.insertBefore(a, b), this.isIframeStarted=!0
            }
        },
        iframeQueue: [],
        enqueueIframe: function(a) {
            this.isLoaded ? this.createIframe(a) : (this.loadIframeApi(), this.iframeQueue.push(a))
        },
        createIframe: function(a) {
            var b = a.pluginMediaElement, c = new YT.Player(a.containerId, {
                height: a.height,
                width: a.width,
                videoId: a.videoId,
                playerVars: {
                    controls: 0
                },
                events: {
                    onReady: function() {
                        a.pluginMediaElement.pluginApi = c, mejs.MediaPluginBridge.initPlugin(a.pluginId), setInterval(function() {
                            mejs.YouTubeApi.createEvent(c, b, "timeupdate")
                        }, 250)
                    },
                    onStateChange: function(a) {
                        mejs.YouTubeApi.handleStateChange(a.data, c, b)
                    }
                }
            })
        },
        createEvent: function(a, b, c) {
            var d = {
                type: c,
                target: b
            };
            if (a && a.getDuration) {
                b.currentTime = d.currentTime = a.getCurrentTime(), b.duration = d.duration = a.getDuration(), d.paused = b.paused, d.ended = b.ended, d.muted = a.isMuted(), d.volume = a.getVolume() / 100, d.bytesTotal = a.getVideoBytesTotal(), d.bufferedBytes = a.getVideoBytesLoaded();
                var e = d.bufferedBytes / d.bytesTotal * d.duration;
                d.target.buffered = d.buffered = {
                    start: function(a) {
                        return 0
                    },
                    end: function(a) {
                        return e
                    },
                    length: 1
                }
            }
            b.dispatchEvent(d.type, d)
        },
        iFrameReady: function() {
            this.isLoaded=!0, this.isIframeLoaded=!0;
            while (this.iframeQueue.length > 0) {
                var a = this.iframeQueue.pop();
                this.createIframe(a)
            }
        },
        flashPlayers: {},
        createFlash: function(a) {
            this.flashPlayers[a.pluginId] = a;
            var b, c = "//www.youtube.com/apiplayer?enablejsapi=1&amp;playerapiid=" + a.pluginId + "&amp;version=3&amp;autoplay=0&amp;controls=0&amp;modestbranding=1&loop=0";
            mejs.MediaFeatures.isIE ? (b = document.createElement("div"), a.container.appendChild(b), b.outerHTML = '<object classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" codebase="//download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab" id="' + a.pluginId + '" width="' + a.width + '" height="' + a.height + '" class="mejs-shim">' + '<param name="movie" value="' + c + '" />' + '<param name="wmode" value="transparent" />' + '<param name="allowScriptAccess" value="always" />' + '<param name="allowFullScreen" value="true" />' + "</object>") : a.container.innerHTML = '<object type="application/x-shockwave-flash" id="' + a.pluginId + '" data="' + c + '" ' + 'width="' + a.width + '" height="' + a.height + '" style="visibility: visible; " class="mejs-shim">' + '<param name="allowScriptAccess" value="always">' + '<param name="wmode" value="transparent">' + "</object>"
        },
        flashReady: function(a) {
            var b = this.flashPlayers[a], c = document.getElementById(a), d = b.pluginMediaElement;
            d.pluginApi = d.pluginElement = c, mejs.MediaPluginBridge.initPlugin(a), c.cueVideoById(b.videoId);
            var e = b.containerId + "_callback";
            window[e] = function(a) {
                mejs.YouTubeApi.handleStateChange(a, c, d)
            }, c.addEventListener("onStateChange", e), setInterval(function() {
                mejs.YouTubeApi.createEvent(c, d, "timeupdate")
            }, 250)
        },
        handleStateChange: function(a, b, c) {
            switch (a) {
            case - 1:
                c.paused=!0, c.ended=!0, mejs.YouTubeApi.createEvent(b, c, "loadedmetadata");
                break;
            case 0:
                c.paused=!1, c.ended=!0, mejs.YouTubeApi.createEvent(b, c, "ended");
                break;
            case 1:
                c.paused=!1, c.ended=!1, mejs.YouTubeApi.createEvent(b, c, "play"), mejs.YouTubeApi.createEvent(b, c, "playing");
                break;
            case 2:
                c.paused=!0, c.ended=!1, mejs.YouTubeApi.createEvent(b, c, "pause");
                break;
            case 3:
                mejs.YouTubeApi.createEvent(b, c, "progress");
                break;
            case 5:
            }
        }
    }, window.mejs = mejs, window.MediaElement = mejs.MediaElement; /*!
     * Adds Internationalization and localization to mediaelement.
     *
     * This file does not contain translations, you have to add the manually.
     * The schema is always the same: me-i18n-locale-[ISO_639-1 Code].js
     *
     * Examples are provided both for german and chinese translation.
     *
     *
     * What is the concept beyond i18n?
     *   http://en.wikipedia.org/wiki/Internationalization_and_localization
     *
     * What langcode should i use?
     *   http://en.wikipedia.org/wiki/ISO_639-1
     *
     *
     * License?
     *
     *   The i18n file uses methods from the Drupal project (drupal.js):
     *     - i18n.methods.t() (modified)
     *     - i18n.methods.checkPlain() (full copy)
     *
     *   The Drupal project is (like mediaelementjs) licensed under GPLv2.
     *    - http://drupal.org/licensing/faq/#q1
     *    - https://github.com/johndyer/mediaelement
     *    - http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
     *
     *
     * @author
     *   Tim Latz (latz.tim@gmail.com)
     *
     *
     * @params
     *  - context - document, iframe ..
     *  - exports - CommonJS, window ..
     *
     */
    (function(a, b, c) {
        "use strict";
        var d = {
            locale: {
                language: "",
                strings: {}
            },
            methods: {}
        };
        d.getLanguage = function() {
            var a = d.locale.language || window.navigator.userLanguage || window.navigator.language;
            return a.substr(0, 2).toLowerCase()
        }, typeof mejsL10n != "undefined" && (d.locale.language = mejsL10n.language), d.methods.checkPlain = function(a) {
            var b, c, d = {
                "&": "&amp;",
                '"': "&quot;",
                "<": "&lt;",
                ">": "&gt;"
            };
            a = String(a);
            for (b in d)
                d.hasOwnProperty(b) && (c = new RegExp(b, "g"), a = a.replace(c, d[b]));
            return a
        }, d.methods.t = function(a, b) {
            return d.locale.strings && d.locale.strings[b.context] && d.locale.strings[b.context][a] && (a = d.locale.strings[b.context][a]), d.methods.checkPlain(a)
        }, d.t = function(a, b) {
            if (typeof a == "string" && a.length > 0) {
                var c = d.getLanguage();
                return b = b || {
                    context: c
                }, d.methods.t(a, b)
            }
            throw {
                name: "InvalidArgumentException",
                message: "First argument is either not a string or empty."
            }
        }, b.i18n = d
    })(document, mejs), function(a, b) {
        "use strict", typeof mejsL10n != "undefined" && (a[mejsL10n.language] = mejsL10n.strings)
    }(mejs.i18n.locale.strings); /*!
     * This is a i18n.locale language object.
     *
     * German translation by Tim Latz, latz.tim@gmail.com
     *
     * @author
     *   Tim Latz (latz.tim@gmail.com)
     *
     * @see
     *   me-i18n.js
     *
     * @params
     *  - exports - CommonJS, window ..
     */
    (function(a, b) {
        "use strict", typeof a.de == "undefined" && (a.de = {
            Fullscreen: "Vollbild",
            "Go Fullscreen": "Vollbild an",
            "Turn off Fullscreen": "Vollbild aus",
            Close: "Schließen"
        })
    })(mejs.i18n.locale.strings); /*!
     * This is a i18n.locale language object.
     *
     * Traditional chinese translation by Tim Latz, latz.tim@gmail.com
     *
     * @author
     *   Tim Latz (latz.tim@gmail.com)
     *
     * @see
     *   me-i18n.js
     *
     * @params
     *  - exports - CommonJS, window ..
     */
    (function(a, b) {
        "use strict", typeof a.zh == "undefined" && (a.zh = {
            Fullscreen: "全螢幕",
            "Go Fullscreen": "全屏模式",
            "Turn off Fullscreen": "退出全屏模式",
            Close: "關閉"
        })
    })(mejs.i18n.locale.strings); /*!
     * MediaElementPlayer
     * http://mediaelementjs.com/
     *
     * Creates a controller bar for HTML5 <video> add <audio> tags
     * using jQuery and MediaElement.js (HTML5 Flash/Silverlight wrapper)
     *
     * Copyright 2010-2013, John Dyer (http://j.hn/)
     * License: MIT
     *
     */
    typeof jQuery != "undefined" ? mejs.$ = jQuery : typeof ender != "undefined" && (mejs.$ = ender), function($) {
        mejs.MepDefaults = {
            poster: "",
            showPosterWhenEnded: !1,
            defaultVideoWidth: 480,
            defaultVideoHeight: 270,
            videoWidth: - 1,
            videoHeight: - 1,
            defaultAudioWidth: 400,
            defaultAudioHeight: 30,
            defaultSeekBackwardInterval: function(a) {
                return a.duration * .05
            },
            defaultSeekForwardInterval: function(a) {
                return a.duration * .05
            },
            audioWidth: - 1,
            audioHeight: - 1,
            startVolume: .8,
            loop: !1,
            autoRewind: !0,
            enableAutosize: !0,
            alwaysShowHours: !1,
            showTimecodeFrameCount: !1,
            framesPerSecond: 25,
            autosizeProgress: !0,
            alwaysShowControls: !1,
            hideVideoControlsOnLoad: !1,
            clickToPlayPause: !0,
            iPadUseNativeControls: !1,
            iPhoneUseNativeControls: !1,
            AndroidUseNativeControls: !1,
            features: ["playpause", "current", "progress", "duration", "tracks", "volume", "fullscreen"],
            isVideo: !0,
            enableKeyboard: !0,
            pauseOtherPlayers: !0,
            keyActions: [{
                keys: [32, 179],
                action: function(a, b) {
                    b.paused || b.ended ? a.play() : a.pause()
                }
            }, {
                keys: [38],
                action: function(a, b) {
                    var c = Math.min(b.volume + .1, 1);
                    b.setVolume(c)
                }
            }, {
                keys: [40],
                action: function(a, b) {
                    var c = Math.max(b.volume - .1, 0);
                    b.setVolume(c)
                }
            }, {
                keys: [37, 227],
                action: function(a, b) {
                    if (!isNaN(b.duration) && b.duration > 0) {
                        a.isVideo && (a.showControls(), a.startControlsTimer());
                        var c = Math.max(b.currentTime - a.options.defaultSeekBackwardInterval(b), 0);
                        b.setCurrentTime(c)
                    }
                }
            }, {
                keys: [39, 228],
                action: function(a, b) {
                    if (!isNaN(b.duration) && b.duration > 0) {
                        a.isVideo && (a.showControls(), a.startControlsTimer());
                        var c = Math.min(b.currentTime + a.options.defaultSeekForwardInterval(b), b.duration);
                        b.setCurrentTime(c)
                    }
                }
            }, {
                keys: [70],
                action: function(a, b) {
                    typeof a.enterFullScreen != "undefined" && (a.isFullScreen ? a.exitFullScreen() : a.enterFullScreen())
                }
            }
            ]
        }, mejs.mepIndex = 0, mejs.players = {}, mejs.MediaElementPlayer = function(a, b) {
            if (this instanceof mejs.MediaElementPlayer) {
                var c = this;
                return c.$media = c.$node = $(a), c.node = c.media = c.$media[0], typeof c.node.player != "undefined" ? c.node.player : (c.node.player = c, typeof b == "undefined" && (b = c.$node.data("mejsoptions")), c.options = $.extend({}, mejs.MepDefaults, b), c.id = "mep_" + mejs.mepIndex++, mejs.players[c.id] = c, c.init(), c)
            }
            return new mejs.MediaElementPlayer(a, b)
        }, mejs.MediaElementPlayer.prototype = {
            hasFocus: !1,
            controlsAreVisible: !0,
            init: function() {
                var a = this, b = mejs.MediaFeatures, c = $.extend(!0, {}, a.options, {
                    success: function(b, c) {
                        a.meReady(b, c)
                    },
                    error: function(b) {
                        a.handleError(b)
                    }
                }), d = a.media.tagName.toLowerCase();
                a.isDynamic = d !== "audio" && d !== "video", a.isDynamic ? a.isVideo = a.options.isVideo : a.isVideo = d !== "audio" && a.options.isVideo;
                if (b.isiPad && a.options.iPadUseNativeControls || b.isiPhone && a.options.iPhoneUseNativeControls)
                    a.$media.attr("controls", "controls"), b.isiPad && a.media.getAttribute("autoplay") !== null && a.play();
                else if (!b.isAndroid ||!a.options.AndroidUseNativeControls) {
                    a.$media.removeAttr("controls"), a.container = $('<div id="' + a.id + '" class="mejs-container ' + (mejs.MediaFeatures.svg ? "svg" : "no-svg") + '">' + '<div class="mejs-inner">' + '<div class="mejs-mediaelement"></div>' + '<div class="mejs-layers"></div>' + '<div class="mejs-controls"></div>' + '<div class="mejs-clear"></div>' + "</div>" + "</div>").addClass(a.$media[0].className).insertBefore(a.$media), a.container.addClass((b.isAndroid ? "mejs-android " : "") + (b.isiOS ? "mejs-ios " : "") + (b.isiPad ? "mejs-ipad " : "") + (b.isiPhone ? "mejs-iphone " : "") + (a.isVideo ? "mejs-video " : "mejs-audio "));
                    if (b.isiOS) {
                        var e = a.$media.clone();
                        a.container.find(".mejs-mediaelement").append(e), a.$media.remove(), a.$node = a.$media = e, a.node = a.media = e[0]
                    } else 
                        a.container.find(".mejs-mediaelement").append(a.$media);
                    a.controls = a.container.find(".mejs-controls"), a.layers = a.container.find(".mejs-layers");
                    var f = a.isVideo ? "video": "audio", g = f.substring(0, 1).toUpperCase() + f.substring(1);
                    a.options[f + "Width"] > 0 || a.options[f + "Width"].toString().indexOf("%")>-1 ? a.width = a.options[f + "Width"] : a.media.style.width !== "" && a.media.style.width !== null ? a.width = a.media.style.width : a.media.getAttribute("width") !== null ? a.width = a.$media.attr("width") : a.width = a.options["default" + g + "Width"], a.options[f + "Height"] > 0 || a.options[f + "Height"].toString().indexOf("%")>-1 ? a.height = a.options[f + "Height"] : a.media.style.height !== "" && a.media.style.height !== null ? a.height = a.media.style.height : a.$media[0].getAttribute("height") !== null ? a.height = a.$media.attr("height") : a.height = a.options["default" + g + "Height"], a.setPlayerSize(a.width, a.height), c.pluginWidth = a.width, c.pluginHeight = a.height
                }
                mejs.MediaElement(a.$media[0], c), typeof a.container != "undefined" && a.controlsAreVisible && a.container.trigger("controlsshown")
            },
            showControls: function(a) {
                var b = this;
                a = typeof a == "undefined" || a;
                if (b.controlsAreVisible)
                    return;
                a ? (b.controls.css("visibility", "visible").stop(!0, !0).fadeIn(200, function() {
                    b.controlsAreVisible=!0, b.container.trigger("controlsshown")
                }), b.container.find(".mejs-control").css("visibility", "visible").stop(!0, !0).fadeIn(200, function() {
                    b.controlsAreVisible=!0
                })) : (b.controls.css("visibility", "visible").css("display", "block"), b.container.find(".mejs-control").css("visibility", "visible").css("display", "block"), b.controlsAreVisible=!0, b.container.trigger("controlsshown")), b.setControlsSize()
            },
            hideControls: function(a) {
                var b = this;
                a = typeof a == "undefined" || a;
                if (!b.controlsAreVisible || b.options.alwaysShowControls)
                    return;
                a ? (b.controls.stop(!0, !0).fadeOut(200, function() {
                    $(this).css("visibility", "hidden").css("display", "block"), b.controlsAreVisible=!1, b.container.trigger("controlshidden")
                }), b.container.find(".mejs-control").stop(!0, !0).fadeOut(200, function() {
                    $(this).css("visibility", "hidden").css("display", "block")
                })) : (b.controls.css("visibility", "hidden").css("display", "block"), b.container.find(".mejs-control").css("visibility", "hidden").css("display", "block"), b.controlsAreVisible=!1, b.container.trigger("controlshidden"))
            },
            controlsTimer: null,
            startControlsTimer: function(a) {
                var b = this;
                a = typeof a != "undefined" ? a : 1500, b.killControlsTimer("start"), b.controlsTimer = setTimeout(function() {
                    b.hideControls(), b.killControlsTimer("hide")
                }, a)
            },
            killControlsTimer: function(a) {
                var b = this;
                b.controlsTimer !== null && (clearTimeout(b.controlsTimer), delete b.controlsTimer, b.controlsTimer = null)
            },
            controlsEnabled: !0,
            disableControls: function() {
                var a = this;
                a.killControlsTimer(), a.hideControls(!1), this.controlsEnabled=!1
            },
            enableControls: function() {
                var a = this;
                a.showControls(!1), a.controlsEnabled=!0
            },
            meReady: function(a, b) {
                var c = this, d = mejs.MediaFeatures, e = b.getAttribute("autoplay"), f = typeof e != "undefined" && e !== null && e !== "false", g, h;
                if (c.created)
                    return;
                c.created=!0, c.media = a, c.domNode = b;
                if ((!d.isAndroid ||!c.options.AndroidUseNativeControls) && (!d.isiPad ||!c.options.iPadUseNativeControls) && (!d.isiPhone ||!c.options.iPhoneUseNativeControls)) {
                    c.buildposter(c, c.controls, c.layers, c.media), c.buildkeyboard(c, c.controls, c.layers, c.media), c.buildoverlays(c, c.controls, c.layers, c.media), c.findTracks();
                    for (g in c.options.features) {
                        h = c.options.features[g];
                        if (c["build" + h])
                            try {
                                c["build" + h](c, c.controls, c.layers, c.media)
                            } catch (i) {}
                    }
                    c.container.trigger("controlsready"), c.setPlayerSize(c.width, c.height), c.setControlsSize(), c.isVideo && (mejs.MediaFeatures.hasTouch ? c.$media.bind("touchstart", function() {
                        c.controlsAreVisible ? c.hideControls(!1) : c.controlsEnabled && c.showControls(!1)
                    }) : (c.clickToPlayPauseCallback = function() {
                        c.options.clickToPlayPause && (c.media.paused ? c.play() : c.pause())
                    }, c.media.addEventListener("click", c.clickToPlayPauseCallback, !1), c.container.bind("mouseenter mouseover", function() {
                        c.controlsEnabled && (c.options.alwaysShowControls || (c.killControlsTimer("enter"), c.showControls(), c.startControlsTimer(2500)))
                    }).bind("mousemove", function() {
                        c.controlsEnabled && (c.controlsAreVisible || c.showControls(), c.options.alwaysShowControls || c.startControlsTimer(2500))
                    }).bind("mouseleave", function() {
                        c.controlsEnabled&&!c.media.paused&&!c.options.alwaysShowControls && c.startControlsTimer(1e3)
                    })), c.options.hideVideoControlsOnLoad && c.hideControls(!1), f&&!c.options.alwaysShowControls && c.hideControls(), c.options.enableAutosize && c.media.addEventListener("loadedmetadata", function(a) {
                        c.options.videoHeight <= 0 && c.domNode.getAttribute("height") === null&&!isNaN(a.target.videoHeight) && (c.setPlayerSize(a.target.videoWidth, a.target.videoHeight), c.setControlsSize(), c.media.setVideoSize(a.target.videoWidth, a.target.videoHeight))
                    }, !1)), a.addEventListener("play", function() {
                        var a;
                        for (a in mejs.players) {
                            var b = mejs.players[a];
                            b.id != c.id && c.options.pauseOtherPlayers&&!b.paused&&!b.ended && b.pause(), b.hasFocus=!1
                        }
                        c.hasFocus=!0
                    }, !1), c.media.addEventListener("ended", function(a) {
                        if (c.options.autoRewind)
                            try {
                                c.media.setCurrentTime(0)
                        } catch (b) {}
                        c.media.pause(), c.setProgressRail && c.setProgressRail(), c.setCurrentRail && c.setCurrentRail(), c.options.loop ? c.play() : !c.options.alwaysShowControls && c.controlsEnabled && c.showControls()
                    }, !1), c.media.addEventListener("loadedmetadata", function(a) {
                        c.updateDuration && c.updateDuration(), c.updateCurrent && c.updateCurrent(), c.isFullScreen || (c.setPlayerSize(c.width, c.height), c.setControlsSize())
                    }, !1), setTimeout(function() {
                        c.setPlayerSize(c.width, c.height), c.setControlsSize()
                    }, 50), c.globalBind("resize", function() {
                        c.isFullScreen || mejs.MediaFeatures.hasTrueNativeFullScreen && document.webkitIsFullScreen || c.setPlayerSize(c.width, c.height), c.setControlsSize()
                    }), c.media.pluginType == "youtube" && c.container.find(".mejs-overlay-play").hide()
                }
                f && a.pluginType == "native" && c.play(), c.options.success && (typeof c.options.success == "string" ? window[c.options.success](c.media, c.domNode, c) : c.options.success(c.media, c.domNode, c))
            },
            handleError: function(a) {
                var b = this;
                b.controls.hide(), b.options.error && b.options.error(a)
            },
            setPlayerSize: function(a, b) {
                var c = this;
                typeof a != "undefined" && (c.width = a), typeof b != "undefined" && (c.height = b);
                if (c.height.toString().indexOf("%") > 0 || c.$node.css("max-width") === "100%" || parseInt(c.$node.css("max-width").replace(/px/, ""), 10) / c.$node.offsetParent().width() === 1 || c.$node[0].currentStyle && c.$node[0].currentStyle.maxWidth === "100%") {
                    var d = c.isVideo ? c.media.videoWidth && c.media.videoWidth > 0 ? c.media.videoWidth: c.options.defaultVideoWidth: c.options.defaultAudioWidth, e = c.isVideo ? c.media.videoHeight && c.media.videoHeight > 0 ? c.media.videoHeight: c.options.defaultVideoHeight: c.options.defaultAudioHeight, f = c.container.parent().closest(":visible").width(), g = c.isVideo ||!c.options.autosizeProgress ? parseInt(f * e / d, 10): e;
                    isNaN(g) && (g = c.container.parent().closest(":visible").height()), c.container.parent()[0].tagName.toLowerCase() === "body" && (f = $(window).width(), g = $(window).height()), g != 0 && f != 0 && (c.container.width(f).height(g), c.$media.add(c.container.find(".mejs-shim")).width("100%").height("100%"), c.isVideo && c.media.setVideoSize && c.media.setVideoSize(f, g), c.layers.children(".mejs-layer").width("100%").height("100%"))
                } else 
                    c.container.width(c.width).height(c.height), c.layers.children(".mejs-layer").width(c.width).height(c.height);
                var h = c.layers.find(".mejs-overlay-play"), i = h.find(".mejs-overlay-button");
                h.height(c.container.height() - c.controls.height()), i.css("margin-top", "-" + (i.height() / 2 - c.controls.height() / 2).toString() + "px")
            },
            setControlsSize: function() {
                var a = this, b = 0, c = 0, d = a.controls.find(".mejs-time-rail"), e = a.controls.find(".mejs-time-total"), f = a.controls.find(".mejs-time-current"), g = a.controls.find(".mejs-time-loaded"), h = d.siblings(), i = h.last(), j = null;
                if (!a.container.is(":visible") ||!d.length ||!d.is(":visible"))
                    return;
                a.options&&!a.options.autosizeProgress && (c = parseInt(d.css("width")));
                if (c === 0 ||!c)
                    h.each(function() {
                        var a = $(this);
                        a.css("position") != "absolute" && a.is(":visible") && (b += $(this).outerWidth(!0))
                    }), c = a.controls.width() - b - (d.outerWidth(!0) - d.width());
                do 
                    d.width(c), e.width(c - (e.outerWidth(!0) - e.width())), i.css("position") != "absolute" && (j = i.position(), c--);
                while (j != null && j.top > 0 && c > 0);
                a.setProgressRail && a.setProgressRail(), a.setCurrentRail && a.setCurrentRail()
            },
            buildposter: function(a, b, c, d) {
                var e = this, f = $('<div class="mejs-poster mejs-layer"></div>').appendTo(c), g = a.$media.attr("poster");
                a.options.poster !== "" && (g = a.options.poster), g !== "" && g != null ? e.setPoster(g) : f.hide(), d.addEventListener("play", function() {
                    f.hide()
                }, !1), a.options.showPosterWhenEnded && a.options.autoRewind && d.addEventListener("ended", function() {
                    f.show()
                }, !1)
            },
            setPoster: function(a) {
                var b = this, c = b.container.find(".mejs-poster"), d = c.find("img");
                d.length == 0 && (d = $('<img width="100%" height="100%" />').appendTo(c)), d.attr("src", a), c.css({
                    "background-image": "url(" + a + ")"
                })
            },
            buildoverlays: function(a, b, c, d) {
                var e = this;
                if (!a.isVideo)
                    return;
                var f = $('<div class="mejs-overlay mejs-layer"><div class="mejs-overlay-loading"><span></span></div></div>').hide().appendTo(c), g = $('<div class="mejs-overlay mejs-layer"><div class="mejs-overlay-error"></div></div>').hide().appendTo(c), h = $('<div class="mejs-overlay mejs-layer mejs-overlay-play"><div class="mejs-overlay-button"></div></div>').appendTo(c).bind("click touchstart", function() {
                    e.options.clickToPlayPause && d.paused && d.play()
                });
                d.addEventListener("play", function() {
                    h.hide(), f.hide(), b.find(".mejs-time-buffering").hide(), g.hide()
                }, !1), d.addEventListener("playing", function() {
                    h.hide(), f.hide(), b.find(".mejs-time-buffering").hide(), g.hide()
                }, !1), d.addEventListener("seeking", function() {
                    f.show(), b.find(".mejs-time-buffering").show()
                }, !1), d.addEventListener("seeked", function() {
                    f.hide(), b.find(".mejs-time-buffering").hide()
                }, !1), d.addEventListener("pause", function() {
                    mejs.MediaFeatures.isiPhone || h.show()
                }, !1), d.addEventListener("waiting", function() {
                    f.show(), b.find(".mejs-time-buffering").show()
                }, !1), d.addEventListener("loadeddata", function() {
                    f.show(), b.find(".mejs-time-buffering").show()
                }, !1), d.addEventListener("canplay", function() {
                    f.hide(), b.find(".mejs-time-buffering").hide()
                }, !1), d.addEventListener("error", function() {
                    f.hide(), b.find(".mejs-time-buffering").hide(), g.show(), g.find("mejs-overlay-error").html("Error loading this resource")
                }, !1)
            },
            buildkeyboard: function(a, b, c, d) {
                var e = this;
                e.globalBind("keydown", function(b) {
                    if (a.hasFocus && a.options.enableKeyboard)
                        for (var c = 0, e = a.options.keyActions.length; c < e; c++) {
                            var f = a.options.keyActions[c];
                            for (var g = 0, h = f.keys.length; g < h; g++)
                                if (b.keyCode == f.keys[g])
                                    return b.preventDefault(), f.action(a, d, b.keyCode), !1
                        }
                    return !0
                }), e.globalBind("click", function(b) {
                    a.hasFocus = $(b.target).closest(".mejs-container").length != 0
                })
            },
            findTracks: function() {
                var a = this, b = a.$media.find("track");
                a.tracks = [], b.each(function(b, c) {
                    c = $(c), a.tracks.push({
                        srclang: c.attr("srclang") ? c.attr("srclang").toLowerCase(): "",
                        src: c.attr("src"),
                        kind: c.attr("kind"),
                        label: c.attr("label") || "",
                        entries: [],
                        isLoaded: !1
                    })
                })
            },
            changeSkin: function(a) {
                this.container[0].className = "mejs-container " + a, this.setPlayerSize(this.width, this.height), this.setControlsSize()
            },
            play: function() {
                this.load(), this.media.play()
            },
            pause: function() {
                try {
                    this.media.pause()
                } catch (a) {}
            },
            load: function() {
                this.isLoaded || this.media.load(), this.isLoaded=!0
            },
            setMuted: function(a) {
                this.media.setMuted(a)
            },
            setCurrentTime: function(a) {
                this.media.setCurrentTime(a)
            },
            getCurrentTime: function() {
                return this.media.currentTime
            },
            setVolume: function(a) {
                this.media.setVolume(a)
            },
            getVolume: function() {
                return this.media.volume
            },
            setSrc: function(a) {
                this.media.setSrc(a)
            },
            remove: function() {
                var a = this, b, c;
                for (b in a.options.features) {
                    c = a.options.features[b];
                    if (a["clean" + c])
                        try {
                            a["clean" + c](a)
                        } catch (d) {}
                }
                a.isDynamic ? a.$node.insertBefore(a.container) : (a.$media.prop("controls", !0), a.$node.clone().show().insertBefore(a.container), a.$node.remove()), a.media.pluginType !== "native" && a.media.remove(), delete mejs.players[a.id], typeof a.container == "object" && a.container.remove(), a.globalUnbind(), delete a.node.player
            }
        }, function() {
            function b(b, c) {
                var d = {
                    d: [],
                    w: []
                };
                return $.each((b || "").split(" "), function(b, e) {
                    var f = e + "." + c;
                    f.indexOf(".") === 0 ? (d.d.push(f), d.w.push(f)) : d[a.test(e) ? "w": "d"].push(f)
                }), d.d = d.d.join(" "), d.w = d.w.join(" "), d
            }
            var a = /^((after|before)print|(before)?unload|hashchange|message|o(ff|n)line|page(hide|show)|popstate|resize|storage)\b/;
            mejs.MediaElementPlayer.prototype.globalBind = function(a, c, d) {
                var e = this;
                a = b(a, e.id), a.d && $(document).bind(a.d, c, d), a.w && $(window).bind(a.w, c, d)
            }, mejs.MediaElementPlayer.prototype.globalUnbind = function(a, c) {
                var d = this;
                a = b(a, d.id), a.d && $(document).unbind(a.d, c), a.w && $(window).unbind(a.w, c)
            }
        }(), typeof jQuery != "undefined" && (jQuery.fn.mediaelementplayer = function(a) {
            return a===!1 ? this.each(function() {
                var a = jQuery(this).data("mediaelementplayer");
                a && a.remove(), jQuery(this).removeData("mediaelementplayer")
            }) : this.each(function() {
                jQuery(this).data("mediaelementplayer", new mejs.MediaElementPlayer(this, a))
            }), this
        }), $(document).ready(function() {
            $(".mejs-player").mediaelementplayer()
        }), window.MediaElementPlayer = mejs.MediaElementPlayer
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            playpauseText: mejs.i18n.t("Play/Pause")
        }), $.extend(MediaElementPlayer.prototype, {
            buildplaypause: function(a, b, c, d) {
                var e = this, f = $('<div class="mejs-button mejs-playpause-button mejs-play" ><button type="button" aria-controls="' + e.id + '" title="' + e.options.playpauseText + '" aria-label="' + e.options.playpauseText + '"></button>' + "</div>").appendTo(b).click(function(a) {
                    return a.preventDefault(), d.paused ? d.play() : d.pause(), !1
                });
                d.addEventListener("play", function() {
                    f.removeClass("mejs-play").addClass("mejs-pause")
                }, !1), d.addEventListener("playing", function() {
                    f.removeClass("mejs-play").addClass("mejs-pause")
                }, !1), d.addEventListener("pause", function() {
                    f.removeClass("mejs-pause").addClass("mejs-play")
                }, !1), d.addEventListener("paused", function() {
                    f.removeClass("mejs-pause").addClass("mejs-play")
                }, !1)
            }
        })
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            stopText: "Stop"
        }), $.extend(MediaElementPlayer.prototype, {
            buildstop: function(a, b, c, d) {
                var e = this, f = $('<div class="mejs-button mejs-stop-button mejs-stop"><button type="button" aria-controls="' + e.id + '" title="' + e.options.stopText + '" aria-label="' + e.options.stopText + '"></button>' + "</div>").appendTo(b).click(function() {
                    d.paused || d.pause(), d.currentTime > 0 && (d.setCurrentTime(0), d.pause(), b.find(".mejs-time-current").width("0px"), b.find(".mejs-time-handle").css("left", "0px"), b.find(".mejs-time-float-current").html(mejs.Utility.secondsToTimeCode(0)), b.find(".mejs-currenttime").html(mejs.Utility.secondsToTimeCode(0)), c.find(".mejs-poster").show())
                })
            }
        })
    }(mejs.$), function($) {
        $.extend(MediaElementPlayer.prototype, {
            buildprogress: function(a, b, c, d) {
                $('<div class="mejs-time-rail"><span class="mejs-time-total"><span class="mejs-time-buffering"></span><span class="mejs-time-loaded"></span><span class="mejs-time-current"></span><span class="mejs-time-handle"></span><span class="mejs-time-float"><span class="mejs-time-float-current">00:00</span><span class="mejs-time-float-corner"></span></span></span></div>').appendTo(b), b.find(".mejs-time-buffering").hide();
                var e = this, f = b.find(".mejs-time-total"), g = b.find(".mejs-time-loaded"), h = b.find(".mejs-time-current"), i = b.find(".mejs-time-handle"), j = b.find(".mejs-time-float"), k = b.find(".mejs-time-float-current"), l = function(a) {
                    if (a.originalEvent.changedTouches)
                        var b = a.originalEvent.changedTouches[0].pageX;
                    else 
                        var b = a.pageX;
                    var c = f.offset(), e = f.outerWidth(!0), g = 0, h = 0, i = 0;
                    d.duration && (b < c.left ? b = c.left : b > e + c.left && (b = e + c.left), i = b - c.left, g = i / e, h = g <= .02 ? 0 : g * d.duration, m && h !== d.currentTime && d.setCurrentTime(h), mejs.MediaFeatures.hasTouch || (j.css("left", i), k.html(mejs.Utility.secondsToTimeCode(h)), j.show()))
                }, m=!1, n=!1;
                f.bind("mousedown touchstart", function(a) {
                    if (a.which === 1 || a.which === 0)
                        return m=!0, l(a), e.globalBind("mousemove.dur touchmove.dur", function(a) {
                            l(a)
                        }), e.globalBind("mouseup.dur touchend.dur", function(a) {
                            m=!1, j.hide(), e.globalUnbind(".dur")
                        }), !1
                }).bind("mouseenter", function(a) {
                    n=!0, e.globalBind("mousemove.dur", function(a) {
                        l(a)
                    }), mejs.MediaFeatures.hasTouch || j.show()
                }).bind("mouseleave", function(a) {
                    n=!1, m || (e.globalUnbind(".dur"), j.hide())
                }), d.addEventListener("progress", function(b) {
                    a.setProgressRail(b), a.setCurrentRail(b)
                }, !1), d.addEventListener("timeupdate", function(b) {
                    a.setProgressRail(b), a.setCurrentRail(b)
                }, !1), e.loaded = g, e.total = f, e.current = h, e.handle = i
            },
            setProgressRail: function(a) {
                var b = this, c = a != undefined ? a.target: b.media, d = null;
                c && c.buffered && c.buffered.length > 0 && c.buffered.end && c.duration ? d = c.buffered.end(0) / c.duration : c && c.bytesTotal != undefined && c.bytesTotal > 0 && c.bufferedBytes != undefined ? d = c.bufferedBytes / c.bytesTotal : a && a.lengthComputable && a.total != 0 && (d = a.loaded / a.total), d !== null && (d = Math.min(1, Math.max(0, d)), b.loaded && b.total && b.loaded.width(b.total.width() * d))
            },
            setCurrentRail: function() {
                var a = this;
                if (a.media.currentTime != undefined && a.media.duration && a.total && a.handle) {
                    var b = Math.round(a.total.width() * a.media.currentTime / a.media.duration), c = b - Math.round(a.handle.outerWidth(!0) / 2);
                    a.current.width(b), a.handle.css("left", c)
                }
            }
        })
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            duration: - 1,
            timeAndDurationSeparator: "<span> | </span>"
        }), $.extend(MediaElementPlayer.prototype, {
            buildcurrent: function(a, b, c, d) {
                var e = this;
                $('<div class="mejs-time"><span class="mejs-currenttime">' + (a.options.alwaysShowHours ? "00:" : "") + (a.options.showTimecodeFrameCount ? "00:00:00" : "00:00") + "</span>" + "</div>").appendTo(b), e.currenttime = e.controls.find(".mejs-currenttime"), d.addEventListener("timeupdate", function() {
                    a.updateCurrent()
                }, !1)
            },
            buildduration: function(a, b, c, d) {
                var e = this;
                b.children().last().find(".mejs-currenttime").length > 0 ? $(e.options.timeAndDurationSeparator + '<span class="mejs-duration">' + (e.options.duration > 0 ? mejs.Utility.secondsToTimeCode(e.options.duration, e.options.alwaysShowHours || e.media.duration > 3600, e.options.showTimecodeFrameCount, e.options.framesPerSecond || 25) : (a.options.alwaysShowHours ? "00:" : "") + (a.options.showTimecodeFrameCount ? "00:00:00" : "00:00")) + "</span>").appendTo(b.find(".mejs-time")) : (b.find(".mejs-currenttime").parent().addClass("mejs-currenttime-container"), $('<div class="mejs-time mejs-duration-container"><span class="mejs-duration">' + (e.options.duration > 0 ? mejs.Utility.secondsToTimeCode(e.options.duration, e.options.alwaysShowHours || e.media.duration > 3600, e.options.showTimecodeFrameCount, e.options.framesPerSecond || 25) : (a.options.alwaysShowHours ? "00:" : "") + (a.options.showTimecodeFrameCount ? "00:00:00" : "00:00")) + "</span>" + "</div>").appendTo(b)), e.durationD = e.controls.find(".mejs-duration"), d.addEventListener("timeupdate", function() {
                    a.updateDuration()
                }, !1)
            },
            updateCurrent: function() {
                var a = this;
                a.currenttime && a.currenttime.html(mejs.Utility.secondsToTimeCode(a.media.currentTime, a.options.alwaysShowHours || a.media.duration > 3600, a.options.showTimecodeFrameCount, a.options.framesPerSecond || 25))
            },
            updateDuration: function() {
                var a = this;
                a.container.toggleClass("mejs-long-video", a.media.duration > 3600), a.durationD && (a.options.duration > 0 || a.media.duration) && a.durationD.html(mejs.Utility.secondsToTimeCode(a.options.duration > 0 ? a.options.duration : a.media.duration, a.options.alwaysShowHours, a.options.showTimecodeFrameCount, a.options.framesPerSecond || 25))
            }
        })
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            muteText: mejs.i18n.t("Mute Toggle"),
            hideVolumeOnTouchDevices: !0,
            audioVolume: "horizontal",
            videoVolume: "vertical"
        }), $.extend(MediaElementPlayer.prototype, {
            buildvolume: function(a, b, c, d) {
                if ((mejs.MediaFeatures.isAndroid || mejs.MediaFeatures.isiOS) && this.options.hideVolumeOnTouchDevices)
                    return;
                var e = this, f = e.isVideo ? e.options.videoVolume: e.options.audioVolume, g = f == "horizontal" ? $('<div class="mejs-button mejs-volume-button mejs-mute"><button type="button" aria-controls="' + e.id + '" title="' + e.options.muteText + '" aria-label="' + e.options.muteText + '"></button>' + "</div>" + '<div class="mejs-horizontal-volume-slider">' + '<div class="mejs-horizontal-volume-total"></div>' + '<div class="mejs-horizontal-volume-current"></div>' + '<div class="mejs-horizontal-volume-handle"></div>' + "</div>").appendTo(b): $('<div class="mejs-button mejs-volume-button mejs-mute"><button type="button" aria-controls="' + e.id + '" title="' + e.options.muteText + '" aria-label="' + e.options.muteText + '"></button>' + '<div class="mejs-volume-slider">' + '<div class="mejs-volume-total"></div>' + '<div class="mejs-volume-current"></div>' + '<div class="mejs-volume-handle"></div>' + "</div>" + "</div>").appendTo(b), h = e.container.find(".mejs-volume-slider, .mejs-horizontal-volume-slider"), i = e.container.find(".mejs-volume-total, .mejs-horizontal-volume-total"), j = e.container.find(".mejs-volume-current, .mejs-horizontal-volume-current"), k = e.container.find(".mejs-volume-handle, .mejs-horizontal-volume-handle"), l = function(a, b) {
                    if (!h.is(":visible") && typeof b == "undefined") {
                        h.show(), l(a, !0), h.hide();
                        return 
                    }
                    a = Math.max(0, a), a = Math.min(a, 1), a == 0 ? g.removeClass("mejs-mute").addClass("mejs-unmute") : g.removeClass("mejs-unmute").addClass("mejs-mute");
                    if (f == "vertical") {
                        var c = i.height(), d = i.position(), e = c - c * a;
                        k.css("top", Math.round(d.top + e - k.height() / 2)), j.height(c - e), j.css("top", d.top + e)
                    } else {
                        var m = i.width(), d = i.position(), n = m * a;
                        k.css("left", Math.round(d.left + n - k.width() / 2)), j.width(Math.round(n))
                    }
                }, m = function(a) {
                    var b = null, c = i.offset();
                    if (f == "vertical") {
                        var e = i.height(), g = parseInt(i.css("top").replace(/px/, ""), 10), h = a.pageY - c.top;
                        b = (e - h) / e;
                        if (c.top == 0 || c.left == 0)
                            return 
                    } else {
                        var j = i.width(), k = a.pageX - c.left;
                        b = k / j
                    }
                    b = Math.max(0, b), b = Math.min(b, 1), l(b), b == 0 ? d.setMuted(!0) : d.setMuted(!1), d.setVolume(b)
                }, n=!1, o=!1;
                g.hover(function() {
                    h.show(), o=!0
                }, function() {
                    o=!1, !n && f == "vertical" && h.hide()
                }), h.bind("mouseover", function() {
                    o=!0
                }).bind("mousedown", function(a) {
                    return m(a), e.globalBind("mousemove.vol", function(a) {
                        m(a)
                    }), e.globalBind("mouseup.vol", function() {
                        n=!1, e.globalUnbind(".vol"), !o && f == "vertical" && h.hide()
                    }), n=!0, !1
                }), g.find("button").click(function() {
                    d.setMuted(!d.muted)
                }), d.addEventListener("volumechange", function(a) {
                    n || (d.muted ? (l(0), g.removeClass("mejs-mute").addClass("mejs-unmute")) : (l(d.volume), g.removeClass("mejs-unmute").addClass("mejs-mute")))
                }, !1), e.container.is(":visible") && (l(a.options.startVolume), a.options.startVolume === 0 && d.setMuted(!0), d.pluginType === "native" && d.setVolume(a.options.startVolume))
            }
        })
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            usePluginFullScreen: !0,
            newWindowCallback: function() {
                return ""
            },
            fullscreenText: mejs.i18n.t("Fullscreen")
        }), $.extend(MediaElementPlayer.prototype, {
            isFullScreen: !1,
            isNativeFullScreen: !1,
            isInIframe: !1,
            buildfullscreen: function(a, b, c, d) {
                if (!a.isVideo)
                    return;
                a.isInIframe = window.location != window.parent.location;
                if (mejs.MediaFeatures.hasTrueNativeFullScreen) {
                    var e = function(b) {
                        a.isFullScreen && (mejs.MediaFeatures.isFullScreen() ? (a.isNativeFullScreen=!0, a.setControlsSize()) : (a.isNativeFullScreen=!1, a.exitFullScreen()))
                    };
                    mejs.MediaFeatures.hasMozNativeFullScreen ? a.globalBind(mejs.MediaFeatures.fullScreenEventName, e) : a.container.bind(mejs.MediaFeatures.fullScreenEventName, e)
                }
                var f = this, g = 0, h = 0, i = a.container, j = $('<div class="mejs-button mejs-fullscreen-button"><button type="button" aria-controls="' + f.id + '" title="' + f.options.fullscreenText + '" aria-label="' + f.options.fullscreenText + '"></button>' + "</div>").appendTo(b);
                if (f.media.pluginType === "native" ||!f.options.usePluginFullScreen&&!mejs.MediaFeatures.isFirefox)
                    j.click(function() {
                        var b = mejs.MediaFeatures.hasTrueNativeFullScreen && mejs.MediaFeatures.isFullScreen() || a.isFullScreen;
                        b ? a.exitFullScreen() : a.enterFullScreen()
                    });
                else {
                    var k = null, l = function() {
                        var a = document.createElement("x"), b = document.documentElement, c = window.getComputedStyle, d;
                        return "pointerEvents"in a.style ? (a.style.pointerEvents = "auto", a.style.pointerEvents = "x", b.appendChild(a), d = c && c(a, "").pointerEvents === "auto", b.removeChild(a), !!d) : !1
                    }();
                    if (l&&!mejs.MediaFeatures.isOpera) {
                        var m=!1, n = function() {
                            if (m) {
                                for (var a in o)
                                    o[a].hide();
                                j.css("pointer-events", ""), f.controls.css("pointer-events", ""), f.media.removeEventListener("click", f.clickToPlayPauseCallback), m=!1
                            }
                        }, o = {}, p = ["top", "left", "right", "bottom"], q, r, s = function() {
                            var a = j.offset().left - f.container.offset().left, b = j.offset().top - f.container.offset().top, c = j.outerWidth(!0), d = j.outerHeight(!0), e = f.container.width(), g = f.container.height();
                            for (q in o)
                                o[q].css({
                                    position: "absolute",
                                    top: 0,
                                    left: 0
                                });
                            o.top.width(e).height(b), o.left.width(a).height(d).css({
                                top: b
                            }), o.right.width(e - a - c).height(d).css({
                                top: b,
                                left: a + c
                            }), o.bottom.width(e).height(g - d - b).css({
                                top: b + d
                            })
                        };
                        f.globalBind("resize", function() {
                            s()
                        });
                        for (q = 0, r = p.length; q < r; q++)
                            o[p[q]] = $('<div class="mejs-fullscreen-hover" />').appendTo(f.container).mouseover(n).hide();
                        j.on("mouseover", function() {
                            if (!f.isFullScreen) {
                                var b = j.offset(), c = a.container.offset();
                                d.positionFullscreenButton(b.left - c.left, b.top - c.top, !1), j.css("pointer-events", "none"), f.controls.css("pointer-events", "none"), f.media.addEventListener("click", f.clickToPlayPauseCallback);
                                for (q in o)
                                    o[q].show();
                                s(), m=!0
                            }
                        }), d.addEventListener("fullscreenchange", function(a) {
                            f.isFullScreen=!f.isFullScreen, f.isFullScreen ? f.media.removeEventListener("click", f.clickToPlayPauseCallback) : f.media.addEventListener("click", f.clickToPlayPauseCallback), n()
                        }), f.globalBind("mousemove", function(a) {
                            if (m) {
                                var b = j.offset();
                                if (a.pageY < b.top || a.pageY > b.top + j.outerHeight(!0) || a.pageX < b.left || a.pageX > b.left + j.outerWidth(!0))
                                    j.css("pointer-events", ""), f.controls.css("pointer-events", ""), m=!1
                            }
                        })
                    } else 
                        j.on("mouseover", function() {
                            k !== null && (clearTimeout(k), delete k);
                            var b = j.offset(), c = a.container.offset();
                            d.positionFullscreenButton(b.left - c.left, b.top - c.top, !0)
                        }).on("mouseout", function() {
                            k !== null && (clearTimeout(k), delete k), k = setTimeout(function() {
                                d.hideFullscreenButton()
                            }, 1500)
                        })
                }
                a.fullscreenBtn = j, f.globalBind("keydown", function(b) {
                    (mejs.MediaFeatures.hasTrueNativeFullScreen && mejs.MediaFeatures.isFullScreen() || f.isFullScreen) && b.keyCode == 27 && a.exitFullScreen()
                })
            },
            cleanfullscreen: function(a) {
                a.exitFullScreen()
            },
            containerSizeTimeout: null,
            enterFullScreen: function() {
                var a = this;
                if (a.media.pluginType !== "native" && (mejs.MediaFeatures.isFirefox || a.options.usePluginFullScreen))
                    return;
                $(document.documentElement).addClass("mejs-fullscreen"), normalHeight = a.container.height(), normalWidth = a.container.width();
                if (a.media.pluginType === "native")
                    if (mejs.MediaFeatures.hasTrueNativeFullScreen)
                        mejs.MediaFeatures.requestFullScreen(a.container[0]), a.isInIframe && setTimeout(function c() {
                            if (a.isNativeFullScreen) {
                                var b = window.devicePixelRatio || 1, d = .002, e = b * $(window).width(), f = screen.width, g = Math.abs(f - e), h = f * d;
                                g > h ? a.exitFullScreen() : setTimeout(c, 500)
                            }
                        }, 500);
                    else if (mejs.MediaFeatures.hasSemiNativeFullScreen) {
                        a.media.webkitEnterFullscreen();
                        return 
                    }
                if (a.isInIframe) {
                    var b = a.options.newWindowCallback(this);
                    if (b !== "") {
                        if (!mejs.MediaFeatures.hasTrueNativeFullScreen) {
                            a.pause(), window.open(b, a.id, "top=0,left=0,width=" + screen.availWidth + ",height=" + screen.availHeight + ",resizable=yes,scrollbars=no,status=no,toolbar=no");
                            return 
                        }
                        setTimeout(function() {
                            a.isNativeFullScreen || (a.pause(), window.open(b, a.id, "top=0,left=0,width=" + screen.availWidth + ",height=" + screen.availHeight + ",resizable=yes,scrollbars=no,status=no,toolbar=no"))
                        }, 250)
                    }
                }
                a.container.addClass("mejs-container-fullscreen").width("100%").height("100%"), a.containerSizeTimeout = setTimeout(function() {
                    a.container.css({
                        width: "100%",
                        height: "100%"
                    }), a.setControlsSize()
                }, 500), a.media.pluginType === "native" ? a.$media.width("100%").height("100%") : (a.container.find(".mejs-shim").width("100%").height("100%"), a.media.setVideoSize($(window).width(), $(window).height())), a.layers.children("div").width("100%").height("100%"), a.fullscreenBtn && a.fullscreenBtn.removeClass("mejs-fullscreen").addClass("mejs-unfullscreen"), a.setControlsSize(), a.isFullScreen=!0
            },
            exitFullScreen: function() {
                var a = this;
                clearTimeout(a.containerSizeTimeout);
                if (a.media.pluginType !== "native" && mejs.MediaFeatures.isFirefox) {
                    a.media.setFullscreen(!1);
                    return 
                }
                mejs.MediaFeatures.hasTrueNativeFullScreen && (mejs.MediaFeatures.isFullScreen() || a.isFullScreen) && mejs.MediaFeatures.cancelFullScreen(), $(document.documentElement).removeClass("mejs-fullscreen"), a.container.removeClass("mejs-container-fullscreen").width(normalWidth).height(normalHeight), a.media.pluginType === "native" ? a.$media.width(normalWidth).height(normalHeight) : (a.container.find(".mejs-shim").width(normalWidth).height(normalHeight), a.media.setVideoSize(normalWidth, normalHeight)), a.layers.children("div").width(normalWidth).height(normalHeight), a.fullscreenBtn.removeClass("mejs-unfullscreen").addClass("mejs-fullscreen"), a.setControlsSize(), a.isFullScreen=!1
            }
        })
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            startLanguage: "",
            tracksText: mejs.i18n.t("Captions/Subtitles"),
            hideCaptionsButtonWhenEmpty: !0,
            toggleCaptionsButtonWhenOnlyOne: !1,
            slidesSelector: ""
        }), $.extend(MediaElementPlayer.prototype, {
            hasChapters: !1,
            buildtracks: function(a, b, c, d) {
                if (a.tracks.length == 0)
                    return;
                var e = this, f, g = "";
                if (e.domNode.textTracks)
                    for (var f = e.domNode.textTracks.length - 1; f >= 0; f--)
                        e.domNode.textTracks[f].mode = "hidden";
                a.chapters = $('<div class="mejs-chapters mejs-layer"></div>').prependTo(c).hide(), a.captions = $('<div class="mejs-captions-layer mejs-layer"><div class="mejs-captions-position mejs-captions-position-hover"><span class="mejs-captions-text"></span></div></div>').prependTo(c).hide(), a.captionsText = a.captions.find(".mejs-captions-text"), a.captionsButton = $('<div class="mejs-button mejs-captions-button"><button type="button" aria-controls="' + e.id + '" title="' + e.options.tracksText + '" aria-label="' + e.options.tracksText + '"></button>' + '<div class="mejs-captions-selector">' + "<ul>" + "<li>" + '<input type="radio" name="' + a.id + '_captions" id="' + a.id + '_captions_none" value="none" checked="checked" />' + '<label for="' + a.id + '_captions_none">' + mejs.i18n.t("None") + "</label>" + "</li>" + "</ul>" + "</div>" + "</div>").appendTo(b);
                var h = 0;
                for (f = 0; f < a.tracks.length; f++)
                    a.tracks[f].kind == "subtitles" && h++;
                e.options.toggleCaptionsButtonWhenOnlyOne && h == 1 ? a.captionsButton.on("click", function() {
                    if (a.selectedTrack == null)
                        var b = a.tracks[0].srclang;
                    else 
                        var b = "none";
                    a.setTrack(b)
                }) : a.captionsButton.hover(function() {
                    $(this).find(".mejs-captions-selector").css("visibility", "visible")
                }, function() {
                    $(this).find(".mejs-captions-selector").css("visibility", "hidden")
                }).on("click", "input[type=radio]", function() {
                    lang = this.value, a.setTrack(lang)
                }), a.options.alwaysShowControls ? a.container.find(".mejs-captions-position").addClass("mejs-captions-position-hover") : a.container.bind("controlsshown", function() {
                    a.container.find(".mejs-captions-position").addClass("mejs-captions-position-hover")
                }).bind("controlshidden", function() {
                    d.paused || a.container.find(".mejs-captions-position").removeClass("mejs-captions-position-hover")
                }), a.trackToLoad =- 1, a.selectedTrack = null, a.isLoadingTrack=!1;
                for (f = 0; f < a.tracks.length; f++)
                    a.tracks[f].kind == "subtitles" && a.addTrackButton(a.tracks[f].srclang, a.tracks[f].label);
                a.loadNextTrack(), d.addEventListener("timeupdate", function(b) {
                    a.displayCaptions()
                }, !1), a.options.slidesSelector != "" && (a.slidesContainer = $(a.options.slidesSelector), d.addEventListener("timeupdate", function(b) {
                    a.displaySlides()
                }, !1)), d.addEventListener("loadedmetadata", function(b) {
                    a.displayChapters()
                }, !1), a.container.hover(function() {
                    a.hasChapters && (a.chapters.css("visibility", "visible"), a.chapters.fadeIn(200).height(a.chapters.find(".mejs-chapter").outerHeight()))
                }, function() {
                    a.hasChapters&&!d.paused && a.chapters.fadeOut(200, function() {
                        $(this).css("visibility", "hidden"), $(this).css("display", "block")
                    })
                }), a.node.getAttribute("autoplay") !== null && a.chapters.css("visibility", "hidden")
            },
            setTrack: function(a) {
                var b = this, c;
                if (a == "none")
                    b.selectedTrack = null, b.captionsButton.removeClass("mejs-captions-enabled");
                else 
                    for (c = 0; c < b.tracks.length; c++)
                        if (b.tracks[c].srclang == a) {
                            b.selectedTrack == null && b.captionsButton.addClass("mejs-captions-enabled"), b.selectedTrack = b.tracks[c], b.captions.attr("lang", b.selectedTrack.srclang), b.displayCaptions();
                            break
                        }
            },
            loadNextTrack: function() {
                var a = this;
                a.trackToLoad++, a.trackToLoad < a.tracks.length ? (a.isLoadingTrack=!0, a.loadTrack(a.trackToLoad)) : (a.isLoadingTrack=!1, a.checkForTracks())
            },
            loadTrack: function(a) {
                var b = this, c = b.tracks[a], d = function() {
                    c.isLoaded=!0, b.enableTrackButton(c.srclang, c.label), b.loadNextTrack()
                };
                $.ajax({
                    url: c.src,
                    dataType: "text",
                    success: function(a) {
                        typeof a == "string" && /<tt\s+xml/gi.exec(a) ? c.entries = mejs.TrackFormatParser.dfxp.parse(a) : c.entries = mejs.TrackFormatParser.webvvt.parse(a), d(), c.kind == "chapters" && b.media.addEventListener("play", function(a) {
                            b.media.duration > 0 && b.displayChapters(c)
                        }, !1), c.kind == "slides" && b.setupSlides(c)
                    },
                    error: function() {
                        b.loadNextTrack()
                    }
                })
            },
            enableTrackButton: function(a, b) {
                var c = this;
                b === "" && (b = mejs.language.codes[a] || a), c.captionsButton.find("input[value=" + a + "]").prop("disabled", !1).siblings("label").html(b), c.options.startLanguage == a && $("#" + c.id + "_captions_" + a).click(), c.adjustLanguageBox()
            },
            addTrackButton: function(a, b) {
                var c = this;
                b === "" && (b = mejs.language.codes[a] || a), c.captionsButton.find("ul").append($('<li><input type="radio" name="' + c.id + '_captions" id="' + c.id + "_captions_" + a + '" value="' + a + '" disabled="disabled" />' + '<label for="' + c.id + "_captions_" + a + '">' + b + " (loading)" + "</label>" + "</li>")), c.adjustLanguageBox(), c.container.find(".mejs-captions-translations option[value=" + a + "]").remove()
            },
            adjustLanguageBox: function() {
                var a = this;
                a.captionsButton.find(".mejs-captions-selector").height(a.captionsButton.find(".mejs-captions-selector ul").outerHeight(!0) + a.captionsButton.find(".mejs-captions-translations").outerHeight(!0))
            },
            checkForTracks: function() {
                var a = this, b=!1;
                if (a.options.hideCaptionsButtonWhenEmpty) {
                    for (i = 0; i < a.tracks.length; i++)
                        if (a.tracks[i].kind == "subtitles") {
                            b=!0;
                            break
                        }
                    b || (a.captionsButton.hide(), a.setControlsSize())
                }
            },
            displayCaptions: function() {
                if (typeof this.tracks == "undefined")
                    return;
                var a = this, b, c = a.selectedTrack;
                if (c != null && c.isLoaded) {
                    for (b = 0; b < c.entries.times.length; b++)
                        if (a.media.currentTime >= c.entries.times[b].start && a.media.currentTime <= c.entries.times[b].stop) {
                            a.captionsText.html(c.entries.text[b]), a.captions.show().height(0);
                            return 
                        }
                    a.captions.hide()
                } else 
                    a.captions.hide()
            },
            setupSlides: function(a) {
                var b = this;
                b.slides = a, b.slides.entries.imgs = [b.slides.entries.text.length], b.showSlide(0)
            },
            showSlide: function(a) {
                if (typeof this.tracks == "undefined" || typeof this.slidesContainer == "undefined")
                    return;
                var b = this, c = b.slides.entries.text[a], d = b.slides.entries.imgs[a];
                typeof d == "undefined" || typeof d.fadeIn == "undefined" ? b.slides.entries.imgs[a] = d = $('<img src="' + c + '">').on("load", function() {
                    d.appendTo(b.slidesContainer).hide().fadeIn().siblings(":visible").fadeOut()
                }) : !d.is(":visible")&&!d.is(":animated") && d.fadeIn().siblings(":visible").fadeOut()
            },
            displaySlides: function() {
                if (typeof this.slides == "undefined")
                    return;
                var a = this, b = a.slides, c;
                for (c = 0; c < b.entries.times.length; c++)
                    if (a.media.currentTime >= b.entries.times[c].start && a.media.currentTime <= b.entries.times[c].stop) {
                        a.showSlide(c);
                        return 
                    }
            },
            displayChapters: function() {
                var a = this, b;
                for (b = 0; b < a.tracks.length; b++)
                    if (a.tracks[b].kind == "chapters" && a.tracks[b].isLoaded) {
                        a.drawChapters(a.tracks[b]), a.hasChapters=!0;
                        break
                    }
            },
            drawChapters: function(a) {
                var b = this, c, d, e = 0, f = 0;
                b.chapters.empty();
                for (c = 0; c < a.entries.times.length; c++) {
                    d = a.entries.times[c].stop - a.entries.times[c].start, e = Math.floor(d / b.media.duration * 100);
                    if (e + f > 100 || c == a.entries.times.length - 1 && e + f < 100)
                        e = 100 - f;
                    b.chapters.append($('<div class="mejs-chapter" rel="' + a.entries.times[c].start + '" style="left: ' + f.toString() + "%;width: " + e.toString() + '%;">' + '<div class="mejs-chapter-block' + (c == a.entries.times.length - 1 ? " mejs-chapter-block-last" : "") + '">' + '<span class="ch-title">' + a.entries.text[c] + "</span>" + '<span class="ch-time">' + mejs.Utility.secondsToTimeCode(a.entries.times[c].start) + "&ndash;" + mejs.Utility.secondsToTimeCode(a.entries.times[c].stop) + "</span>" + "</div>" + "</div>")), f += e
                }
                b.chapters.find("div.mejs-chapter").click(function() {
                    b.media.setCurrentTime(parseFloat($(this).attr("rel"))), b.media.paused && b.media.play()
                }), b.chapters.show()
            }
        }), mejs.language = {
            codes: {
                af: "Afrikaans",
                sq: "Albanian",
                ar: "Arabic",
                be: "Belarusian",
                bg: "Bulgarian",
                ca: "Catalan",
                zh: "Chinese",
                "zh-cn": "Chinese Simplified",
                "zh-tw": "Chinese Traditional",
                hr: "Croatian",
                cs: "Czech",
                da: "Danish",
                nl: "Dutch",
                en: "English",
                et: "Estonian",
                tl: "Filipino",
                fi: "Finnish",
                fr: "French",
                gl: "Galician",
                de: "German",
                el: "Greek",
                ht: "Haitian Creole",
                iw: "Hebrew",
                hi: "Hindi",
                hu: "Hungarian",
                is: "Icelandic",
                id: "Indonesian",
                ga: "Irish",
                it: "Italian",
                ja: "Japanese",
                ko: "Korean",
                lv: "Latvian",
                lt: "Lithuanian",
                mk: "Macedonian",
                ms: "Malay",
                mt: "Maltese",
                no: "Norwegian",
                fa: "Persian",
                pl: "Polish",
                pt: "Portuguese",
                ro: "Romanian",
                ru: "Russian",
                sr: "Serbian",
                sk: "Slovak",
                sl: "Slovenian",
                es: "Spanish",
                sw: "Swahili",
                sv: "Swedish",
                tl: "Tagalog",
                th: "Thai",
                tr: "Turkish",
                uk: "Ukrainian",
                vi: "Vietnamese",
                cy: "Welsh",
                yi: "Yiddish"
            }
        }, mejs.TrackFormatParser = {
            webvvt: {
                pattern_identifier: /^([a-zA-z]+-)?[0-9]+$/,
                pattern_timecode: /^([0-9]{2}:[0-9]{2}:[0-9]{2}([,.][0-9]{1,3})?) --\> ([0-9]{2}:[0-9]{2}:[0-9]{2}([,.][0-9]{3})?)(.*)$/,
                parse: function(a) {
                    var b = 0, c = mejs.TrackFormatParser.split2(a, /\r?\n/), d = {
                        text: [],
                        times: []
                    }, e, f;
                    for (; b < c.length; b++)
                        if (this.pattern_identifier.exec(c[b])) {
                            b++, e = this.pattern_timecode.exec(c[b]);
                            if (e && b < c.length) {
                                b++, f = c[b], b++;
                                while (c[b] !== "" && b < c.length)
                                    f = f + "\n" + c[b], b++;
                                    f = $.trim(f).replace(/(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/gi, "<a href='$1' target='_blank'>$1</a>"), d.text.push(f), d.times.push({
                                        start: mejs.Utility.convertSMPTEtoSeconds(e[1]) == 0 ? .2: mejs.Utility.convertSMPTEtoSeconds(e[1]),
                                        stop: mejs.Utility.convertSMPTEtoSeconds(e[3]),
                                        settings: e[5]
                                    })
                                }
                        }
                    return d
                }
            },
            dfxp: {
                parse: function(a) {
                    a = $(a).filter("tt");
                    var b = 0, c = a.children("div").eq(0), d = c.find("p"), e = a.find("#" + c.attr("style")), f, g, h, i, j = {
                        text: [],
                        times: []
                    };
                    if (e.length) {
                        var k = e.removeAttr("id").get(0).attributes;
                        if (k.length) {
                            f = {};
                            for (b = 0; b < k.length; b++)
                                f[k[b].name.split(":")[1]] = k[b].value
                        }
                    }
                    for (b = 0; b < d.length; b++) {
                        var l, m = {
                            start: null,
                            stop: null,
                            style: null
                        };
                        d.eq(b).attr("begin") && (m.start = mejs.Utility.convertSMPTEtoSeconds(d.eq(b).attr("begin"))), !m.start && d.eq(b - 1).attr("end") && (m.start = mejs.Utility.convertSMPTEtoSeconds(d.eq(b - 1).attr("end"))), d.eq(b).attr("end") && (m.stop = mejs.Utility.convertSMPTEtoSeconds(d.eq(b).attr("end"))), !m.stop && d.eq(b + 1).attr("begin") && (m.stop = mejs.Utility.convertSMPTEtoSeconds(d.eq(b + 1).attr("begin")));
                        if (f) {
                            l = "";
                            for (var n in f)
                                l += n + ":" + f[n] + ";"
                        }
                        l && (m.style = l), m.start == 0 && (m.start = .2), j.times.push(m), i = $.trim(d.eq(b).html()).replace(/(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/gi, "<a href='$1' target='_blank'>$1</a>"), j.text.push(i), j.times.start == 0 && (j.times.start = 2)
                    }
                    return j
                }
            },
            split2: function(a, b) {
                return a.split(b)
            }
        }, "x\n\ny".split(/\n/gi).length != 3 && (mejs.TrackFormatParser.split2 = function(a, b) {
            var c = [], d = "", e;
            for (e = 0; e < a.length; e++)
                d += a.substring(e, e + 1), b.test(d) && (c.push(d.replace(b, "")), d = "");
            return c.push(d), c
        })
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            contextMenuItems: [{
                render: function(a) {
                    return typeof a.enterFullScreen == "undefined" ? null : a.isFullScreen ? mejs.i18n.t("Turn off Fullscreen") : mejs.i18n.t("Go Fullscreen")
                },
                click: function(a) {
                    a.isFullScreen ? a.exitFullScreen() : a.enterFullScreen()
                }
            }, {
                render: function(a) {
                    return a.media.muted ? mejs.i18n.t("Unmute") : mejs.i18n.t("Mute")
                },
                click: function(a) {
                    a.media.muted ? a.setMuted(!1) : a.setMuted(!0)
                }
            }, {
                isSeparator: !0
            }, {
                render: function(a) {
                    return mejs.i18n.t("Download Video")
                },
                click: function(a) {
                    window.location.href = a.media.currentSrc
                }
            }
            ]
        }), $.extend(MediaElementPlayer.prototype, {
            buildcontextmenu: function(a, b, c, d) {
                a.contextMenu = $('<div class="mejs-contextmenu"></div>').appendTo($("body")).hide(), a.container.bind("contextmenu", function(b) {
                    if (a.isContextMenuEnabled)
                        return b.preventDefault(), a.renderContextMenu(b.clientX - 1, b.clientY - 1), !1
                }), a.container.bind("click", function() {
                    a.contextMenu.hide()
                }), a.contextMenu.bind("mouseleave", function() {
                    a.startContextMenuTimer()
                })
            },
            cleancontextmenu: function(a) {
                a.contextMenu.remove()
            },
            isContextMenuEnabled: !0,
            enableContextMenu: function() {
                this.isContextMenuEnabled=!0
            },
            disableContextMenu: function() {
                this.isContextMenuEnabled=!1
            },
            contextMenuTimeout: null,
            startContextMenuTimer: function() {
                var a = this;
                a.killContextMenuTimer(), a.contextMenuTimer = setTimeout(function() {
                    a.hideContextMenu(), a.killContextMenuTimer()
                }, 750)
            },
            killContextMenuTimer: function() {
                var a = this.contextMenuTimer;
                a != null && (clearTimeout(a), delete a, a = null)
            },
            hideContextMenu: function() {
                this.contextMenu.hide()
            },
            renderContextMenu: function(a, b) {
                var c = this, d = "", e = c.options.contextMenuItems;
                for (var f = 0, g = e.length; f < g; f++)
                    if (e[f].isSeparator)
                        d += '<div class="mejs-contextmenu-separator"></div>';
                    else {
                        var h = e[f].render(c);
                        h != null && (d += '<div class="mejs-contextmenu-item" data-itemindex="' + f + '" id="element-' + Math.random() * 1e6 + '">' + h + "</div>")
                    }
                c.contextMenu.empty().append($(d)).css({
                    top: b,
                    left: a
                }).show(), c.contextMenu.find(".mejs-contextmenu-item").each(function() {
                    var a = $(this), b = parseInt(a.data("itemindex"), 10), d = c.options.contextMenuItems[b];
                    typeof d.show != "undefined" && d.show(a, c), a.click(function() {
                        typeof d.click != "undefined" && d.click(c), c.contextMenu.hide()
                    })
                }), setTimeout(function() {
                    c.killControlsTimer("rev3")
                }, 100)
            }
        })
    }(mejs.$), function($) {
        $.extend(mejs.MepDefaults, {
            postrollCloseText: mejs.i18n.t("Close")
        }), $.extend(MediaElementPlayer.prototype, {
            buildpostroll: function(a, b, c, d) {
                var e = this, f = e.container.find('link[rel="postroll"]').attr("href");
                typeof f != "undefined" && (a.postroll = $('<div class="mejs-postroll-layer mejs-layer"><a class="mejs-postroll-close" onclick="$(this).parent().hide();return false;">' + e.options.postrollCloseText + '</a><div class="mejs-postroll-layer-content"></div></div>').prependTo(c).hide(), e.media.addEventListener("ended", function(b) {
                    $.ajax({
                        dataType: "html",
                        url: f,
                        success: function(a, b) {
                            c.find(".mejs-postroll-layer-content").html(a)
                        }
                    }), a.postroll.show()
                }, !1))
            }
        })
    }(mejs.$)
});
define("app/ui/with_tweet_click_handler", ["module", "require", "exports", "$lib/mediaelement.js", "app/utils/caret", "app/utils/push_state_enabled"], function(module, require, exports) {
    require("$lib/mediaelement.js");
    var caret = require("app/utils/caret"), pushStateEnabled = require("app/utils/push_state_enabled");
    module.exports = function() {
        this.defaultAttrs({
            jsMediaPreviewSelector: ".js-media-preview",
            jsTweetSelector: ".js-stream-tweet",
            jsLinkSelector: ".js-link",
            jsExpandoDisabledSelector: ".js-expando-disabled",
            jsPermalinkSelector: ".js-permalink",
            jsDetailsLinkSelector: ".js-details",
            quoteTweetSelector: ".QuoteTweet-container",
            quoteTweetClass: "QuoteTweet-container",
            pushState: !0,
            profileTweetSelector: ".ProfileTweet",
            profileElementsForHover: ".ProfileTweet-header .ProfileTweet-timestamp",
            expandoHandleSelector: ".js-open-close-tweet span",
            dropdownSelector: ".ProfileTweet-action .dropdown-menu",
            withheldTweetClass: "withheld-tweet",
            profileTweetClass: "ProfileTweet",
            suitActionClass: "is-actionable",
            textUserColorClass: "u-textUserColor",
            profileTweetActionButtonSelector: ".js-actionButton",
            profileTweetActionCountSelector: ".js-actionCount",
            mediaGifSelector: ".TwitterPhoto-gif",
            streamItemSelector: ".StreamItem",
            streamItemClass: "StreamItem"
        }), this.shouldNavigateToPermalink = function(a) {
            return this.attr.isMobile || this.isProfileTweet(a) || this.isQuoteTweet(a)
        }, this.isProfileTweet = function(a) {
            return a.hasClass(this.attr.profileTweetClass)
        }, this.isQuoteTweet = function(a) {
            return a.hasClass(this.attr.quoteTweetClass)
        }, this.isAnimatedGif = function(a) {
            return !!a.find(this.attr.mediaGifSelector).length
        }, this.focusProfileTweet = function(a) {
            a.addClass(this.attr.suitActionClass).focus().find(this.attr.profileElementsForHover).addClass(this.attr.textUserColorClass)
        }, this.blurProfileTweet = function(a) {
            a.removeClass(this.attr.suitActionClass).blur().find(this.attr.profileElementsForHover).removeClass(this.attr.textUserColorClass)
        }, this.handleFocusingProfileTweet = function(a, b) {
            this.focusProfileTweet($(b.el))
        }, this.handleBlurringProfileTweet = function(a, b) {
            this.blurProfileTweet($(b.el))
        }, this.handleBlurringAllProfileTweets = function(a, b) {
            var c = $(this.attr.profileTweetSelector);
            c && c.length && this.blurProfileTweet($(this.attr.profileTweetSelector))
        }, this.handleMediaPreviewClick = function(a, b) {
            var c = $(b.el), d = c.closest(this.attr.jsTweetSelector);
            this.shouldExpandWhenTargetIs($(a.target), d) && (a.preventDefault(), a.stopPropagation(), this.isProfileTweet(d) || this.handleClick(a, d, !0), this.isAnimatedGif(d) && this.handleGifClick(a, d))
        }, this.handleGifClick = function(a, b) {
            var c = b.find(this.attr.mediaGifSelector);
            c.hasClass("is-playing") ? (this.trigger(b, "uiStopGif"), c.removeClass("is-playing")) : (this.trigger(b, "uiStartGif"), c.addClass("is-playing"))
        }, this.handleTweetClick = function(a, b) {
            var c = $(b.el), d = c.hasClass(this.attr.streamItemClass) ? c.find(this.attr.profileTweetSelector): c;
            this.shouldExpandWhenTargetIs($(a.target), d) && (a.preventDefault(), a.stopPropagation(), this.handleClick(a, d, !1))
        }, this.handleClick = function(a, b, c) {
            if (!c && this.shouldNavigateToPermalink(b)) {
                b.trigger("uiHasClickedTweet", {
                    organicExpansion: !0,
                    impressionId: b.closest("[data-impression-id]").attr("data-impression-id"),
                    disclosureType: b.closest("[data-impression-id]").attr("data-disclosure-type")
                });
                var d = b.find(this.attr.jsPermalinkSelector), e = d.attr("href"), f = d.attr("target");
                this.modifierKey(a) || f === "_blank" ? window.open(e, "_blank") : this.trigger("uiNavigate", {
                    href: e
                })
            } else 
                this.trigger(b.parent(), "uiShouldToggleExpandedState", {
                    autoplay: c
                })
        }, this.shouldExpandWhenTargetIs = function(a, b) {
            return this.selectedText() || b.hasClass(this.attr.withheldTweetClass)?!1 : this.targetIsWhitelistedToExpandTweet(a, b) ||!this.targetIsBlacklistedToExpandTweet(a, b)
        }, this.targetIsWhitelistedToExpandTweet = function(a, b) {
            var c = [this.attr.expandoHandleSelector, this.attr.jsDetailsLinkSelector, this.attr.jsMediaPreviewSelector, this.attr.mediaGifSelector, this.attr.quoteTweetSelector];
            return a.closest(c.join(), b).length > 0
        }, this.targetIsBlacklistedToExpandTweet = function(a, b) {
            var c = ["a", "button", this.attr.jsLinkSelector, this.attr.dropdownSelector, this.attr.jsExpandoDisabledSelector];
            return a.closest(c.join(), b).length
        }, this.selectedText = function() {
            return caret.getSelection()
        }, this.modifierKey = function(a) {
            return a.shiftKey || a.ctrlKey || a.metaKey || a.which > 1
        }, this.preventFocusOnClick = function(a) {
            a.preventDefault()
        }, this.after("initialize", function() {
            this.on("click uiExpandTweet", {
                jsMediaPreviewSelector: this.handleMediaPreviewClick,
                quoteTweetSelector: this.handleTweetClick,
                jsTweetSelector: this.handleTweetClick
            }), this.on("uiShortcutEnter", {
                quoteTweetSelector: this.handleTweetClick,
                streamItemSelector: this.handleTweetClick
            }), this.on("mouseover onfocus", {
                profileTweetSelector: this.handleFocusingProfileTweet
            }), this.on("mouseout onblur", {
                profileTweetSelector: this.handleBlurringProfileTweet
            }), this.on("mousedown", {
                profileTweetActionButtonSelector: this.preventFocusOnClick,
                profileTweetActionCountSelector: this.preventFocusOnClick,
                quoteTweetSelector: this.preventFocusOnClick
            }), this.before("teardown", this.handleBlurringAllProfileTweets)
        })
    }
});
define("app/ui/dm/tweet_attachment", ["module", "require", "exports", "core/component", "app/ui/with_tweet_click_handler"], function(module, require, exports) {
    function tweetAttachment() {
        this.defaultAttrs({
            sharedTweetSelector: ".dm-attached-tweet"
        }), this.openSharedTweet = function(a, b) {
            this.trigger("uiDMDialogSharedTweetClick")
        }, this.after("initialize", function() {
            this.on("click", {
                sharedTweetSelector: this.openSharedTweet
            })
        })
    }
    var defineComponent = require("core/component"), withTweetClickHandler = require("app/ui/with_tweet_click_handler"), TweetAttachment = defineComponent(tweetAttachment, withTweetClickHandler);
    module.exports = TweetAttachment
});
define("app/data/dm/tweet_attachment", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function tweetAttachmentScribe() {
        this.after("initialize", function() {
            this.scribeOnEvent("uiDMDialogSharedTweetClick", {
                element: "shared_tweet_dm",
                action: "click"
            })
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), TweetAttachmentScribe = defineComponent(tweetAttachmentScribe, withScribe);
    module.exports = TweetAttachmentScribe
});
define("app/ui/with_condensing", ["module", "require", "exports"], function(module, require, exports) {
    function withCondensing() {
        this.defaultAttrs({
            condensable: !1,
            condensedClass: "condensed"
        }), this.isCondensed = function() {
            return this.attr.condensable && this.$node.hasClass(this.attr.condensedClass)
        }, this.expand = function(a) {
            this.isCondensed() && (this.$node.removeClass(this.attr.condensedClass), this.trigger("uiTweetBoxExpanded"), a && this.trigger("uiComposerResetAndFocus"))
        }, this.getCondensedText = function() {
            return this.condensedText === undefined && (this.condensedText = this.attr.condensedText !== undefined ? this.attr.condensedText : this.$node.attr("data-condensed-text") || ""), this.condensedText
        }, this.condense = function() {
            this.isCondensed() || (this.$node.addClass(this.attr.condensedClass), this.$text.blur(), this.resetTweetText(), this.trigger("uiTweetBoxCondensed"))
        }, this.condenseEmptyTweet = function() {
            this.detectUpdatedText(), this.trigger("uiHideAutocomplete"), this.isEmpty() && this.condense()
        }, this.condenseOnMouseDown = function(a) {
            this.isCondensed() || ($.contains(this.$condenseContainer[0], a.target) ? this.blockCondense=!0 : this.shouldCondenseOnMouseUp=!0)
        }, this.condenseOnMouseUp = function() {
            this.shouldCondenseOnMouseUp && (this.shouldCondenseOnMouseUp=!1, this.condenseEmptyTweet())
        }, this.condenseOnBlur = function() {
            this.blockCondense || this.shouldCondenseOnMouseUp ? this.blockCondense=!1 : this.condenseEmptyTweet()
        }, this.expandOnFocus = function() {
            this.expand(!0)
        }, this.around("resetTweetText", function(a) {
            this.isCondensed() ? this.setVisibleText(this.getCondensedText()) : a()
        }), this.around("isEmpty", function(a) {
            return this.isCondensed() && this.getVisibleText().trim() === this.getCondensedText().trim() || a()
        }), this.after("updateTweetStateOnTextChanged", function() {
            this.attr.preexpandTweetbox && this.$text.toggleClass("is-showPlaceholder", !this.getVisibleText().trim())
        }), this.after("clear", function() {
            this.attr.condensable && this.condense()
        }), this.after("initialize", function() {
            this.attr.condenseContainer ? this.$condenseContainer = $(this.attr.condenseContainer) : this.$condenseContainer = this.$node, this.attr.preexpandTweetbox && this.$text.attr("data-placeholder", this.getCondensedText()), this.attr.condensable && (this.on(this.$text, "focusin", this.ignoreDuringFakeFocus(this.expandOnFocus)), this.on(this.$text, "focusout", this.ignoreDuringFakeFocus(this.condenseOnBlur)), this.on(document, "mousedown", this.condenseOnMouseDown), this.on(document, "mouseup", this.condenseOnMouseUp), (this.hasFocus() ||!this.isEmpty()) && this.expand(!1))
        })
    }
    module.exports = withCondensing
});
define("app/ui/with_allow_teardown", ["module", "require", "exports"], function(module, require, exports) {
    function withAllowTeardown() {
        this.allowTeardown = function(a, b) {
            this.attr.noTeardown=!1
        }, this.after("initialize", function() {
            this.on("uiAllowTeardown", this.allowTeardown)
        })
    }
    module.exports = withAllowTeardown
});
define("app/ui/with_draft_tweets", ["module", "require", "exports", "app/utils/storage/custom"], function(module, require, exports) {
    var customStorage = require("app/utils/storage/custom");
    module.exports = function() {
        this.defaultAttrs({
            draftTweetTTL: 864e5
        }), this.getDraftTweet = function() {
            return this.attr.draftTweetId && this.draftTweets().getItem(this.attr.draftTweetId)
        }, this.hasDraftTweet = function() {
            return !!this.getDraftTweet()
        }, this.saveDraftTweet = function(a, b) {
            if (this.attr.draftTweetId && this.hasFocus()) {
                var c = b.text.trim();
                c&&!this.isEmpty(!0) ? this.draftTweets().setItem(this.attr.draftTweetId, c, this.attr.draftTweetTTL) : this.draftTweets().removeItem(this.attr.draftTweetId)
            }
        }, this.before("clear", function() {
            this.attr.draftTweetId && this.draftTweets().removeItem(this.attr.draftTweetId)
        }), this.overrideDraftTweetId = function(a, b) {
            this.attr.draftTweetId = b.draftTweetId, this.resetTweetText()
        }, this.draftTweets = function() {
            if (!this.draftTweetsStore) {
                var a = customStorage({
                    withExpiry: !0
                });
                this.draftTweetsStore = new a("draft_tweets")
            }
            return this.draftTweetsStore
        }, this.after("resetTweetText", function() {
            var a = this.getDraftTweet();
            a && (this.setVisibleText(a), this.trigger("uiSaveUndoState"))
        }), this.initDraftTweets = function() {
            this.on("uiTextChanged", this.saveDraftTweet), this.attr.modal && this.on(document, "uiOverrideTweetBoxOptions", this.overrideDraftTweetId)
        }
    }
});
define("app/utils/flash_version", ["module", "require", "exports"], function(module, require, exports) {
    module.exports = {
        path: "/flash/1/"
    }
});
deferred('$lib/jquery.swfobject.js', function() {
    /*! jQuery SWFObject v1.1.1 MIT/GPL @jon_neal http://jquery.thewikies.com/swfobject */
    (function(a, b, c) {
        function d(a, b) {
            var c = (a[0] || 0) - (b[0] || 0);
            return c > 0 ||!c && a.length > 0 && d(a.slice(1), b.slice(1))
        }
        function e(a) {
            if (typeof a != h)
                return a;
            var b = [], c = "";
            for (var d in a)
                c = typeof a[d] == h ? e(a[d]) : [d, i ? encodeURI(a[d]): a[d]].join("="), b.push(c);
            return b.join("&")
        }
        function f(a) {
            var b = [];
            for (var c in a)
                a[c] && b.push([c, '="', a[c], '"'].join(""));
            return b.join(" ")
        }
        function g(a) {
            var b = [];
            for (var c in a)
                b.push(['<param name="', c, '" value="', e(a[c]), '" />'].join(""));
            return b.join("")
        }
        var h = "object", i=!0;
        try {
            var j = c.description || function() {
                return (new c("ShockwaveFlash.ShockwaveFlash")).GetVariable("$version")
            }()
        } catch (k) {
            j = "Unavailable"
        }
        var l = j.match(/\d+/g) || [0];
        a[b] = {
            available: l[0] > 0,
            activeX: c&&!c.name,
            version: {
                original: j,
                array: l,
                string: l.join("."),
                major: parseInt(l[0], 10) || 0,
                minor: parseInt(l[1], 10) || 0,
                release: parseInt(l[2], 10) || 0
            },
            hasVersion: function(a) {
                return a = /string|number/.test(typeof a) ? a.toString().split(".") : /object/.test(typeof a) ? [a.major, a.minor] : a || [0, 0], d(l, a)
            },
            encodeParams: !0,
            expressInstall: "expressInstall.swf",
            expressInstallIsActive: !1,
            create: function(a) {
                if (!a.swf || this.expressInstallIsActive ||!this.available&&!a.hasVersionFail)
                    return !1;
                if (!this.hasVersion(a.hasVersion || 1)) {
                    this.expressInstallIsActive=!0;
                    if (typeof a.hasVersionFail == "function"&&!a.hasVersionFail.apply(a))
                        return !1;
                    a = {
                        swf: a.expressInstall || this.expressInstall,
                        height: 137,
                        width: 214,
                        flashvars: {
                            MMredirectURL: location.href,
                            MMplayerType: this.activeX ? "ActiveX": "PlugIn",
                            MMdoctitle: document.title.slice(0, 47) + " - Flash Player Installation"
                        }
                    }
                }
                attrs = {
                    data: a.swf,
                    type: "application/x-shockwave-flash",
                    id: a.id || "flash_" + Math.floor(Math.random() * 999999999),
                    width: a.width || 320,
                    height: a.height || 180,
                    style: a.style || ""
                }, i = typeof a.useEncode != "undefined" ? a.useEncode : this.encodeParams, a.movie = a.swf, a.wmode = a.wmode || "opaque", delete a.fallback, delete a.hasVersion, delete a.hasVersionFail, delete a.height, delete a.id, delete a.swf, delete a.useEncode, delete a.width;
                var b = document.createElement("div");
                return b.innerHTML = ["<object ", f(attrs), ">", g(a), "</object>"].join(""), b.firstChild
            }
        }, a.fn[b] = function(c) {
            var d = this.find(h).andSelf().filter(h);
            return /string|object/.test(typeof c) && this.each(function() {
                var d = a(this), e;
                c = typeof c == h ? c : {
                    swf: c
                }, c.fallback = this;
                if (e = a[b].create(c))
                    d.children().remove(), d.html(e)
            }), typeof c == "function" && d.each(function() {
                var d = this;
                d.jsInteractionTimeoutMs = d.jsInteractionTimeoutMs || 0, d.jsInteractionTimeoutMs < 660 && (d.clientWidth || d.clientHeight ? c.call(d) : setTimeout(function() {
                    a(d)[b](c)
                }, d.jsInteractionTimeoutMs + 66))
            }), d
        }
    })(jQuery, "flash", navigator.plugins["Shockwave Flash"] || window.ActiveXObject)
});
define("app/utils/image", ["module", "require", "exports", "app/utils/flash_version", "$lib/jquery.swfobject.js"], function(module, require, exports) {
    var flashVersion = require("app/utils/flash_version");
    require("$lib/jquery.swfobject.js");
    var image = {
        photoHelperSwfPath: flashVersion.path + "PhotoHelper.swf",
        photoSelectorSwfPath: flashVersion.path + "PhotoSelector.swf",
        MAX_FILE_SIZE: 5242880,
        validateFileName: function(a) {
            return /(.*)\.(jpg|jpeg|png|gif)$/i.test(a)
        },
        validateImageSize: function(a, b) {
            var c = a && a.size || a.fileSize, b = b || this.MAX_FILE_SIZE;
            return !c || c <= b
        },
        getFileName: function(a) {
            if (a.indexOf("/")==-1 && a.indexOf("\\")==-1)
                return a;
            var b = a.match(/(?:.*)[\/\\]([^\/\\]+(?:\.\w+)?)$/);
            return b[1]
        },
        isAnimatedGif: function(a) {
            return /(.*)\.gif$/i.test(a)
        },
        loadPhotoHelperSwf: function(a, b, c, d, e) {
            return a.flash({
                swf: this.photoHelperSwfPath,
                height: d,
                width: e,
                wmode: "transparent",
                AllowScriptAccess: "sameDomain",
                flashvars: {
                    callbackName: b,
                    errorCallbackName: c
                }
            }), a.find("object")
        },
        loadPhotoSelectorSwf: function(a, b, c, d, e, f) {
            return a.flash({
                swf: this.photoSelectorSwfPath,
                height: d,
                width: e,
                wmode: "transparent",
                AllowScriptAccess: "sameDomain",
                flashvars: {
                    callbackName: b,
                    errorCallbackName: c,
                    buttonWidth: e,
                    buttonHeight: d,
                    maxSizeInBytes: f
                }
            }), a.find("object")
        },
        hasFlash: function() {
            try {
                return $.flash.available && $.flash.hasVersion(10)
            } catch (a) {
                return !1
            }
        },
        getUserMedia: function() {
            var a = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
            return a ? a.bind(navigator) : !1
        }(),
        hasGetUserMedia: function() {
            var a = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia, b = $.browser.mozilla && parseInt($.browser.version) < 22;
            return !!a&&!b
        },
        hasFileReader: function() {
            return typeof FileReader == "function" || typeof FileReader == "object"?!0 : !1
        },
        hasCanvas: function() {
            var a = document.createElement("canvas");
            return !!a.getContext&&!!a.getContext("2d")
        },
        supportsCropper: function() {
            return this.hasCanvas() && (this.hasFileReader() || this.hasFlash())
        },
        getFileHandle: function(a) {
            return a.files && a.files[0] ? a.files[0] : !1
        },
        shouldUseFlash: function() {
            return !this.hasFileReader() && this.hasFlash()
        },
        mode: function() {
            return this.hasFileReader() ? "html5" : this.hasFlash() ? "flash" : "html4"
        },
        getDataUri: function(a, b) {
            var c = "data:image/jpeg;base64," + a;
            return b && (c = "url(" + c + ")"), c
        },
        getFileData: function(a, b, c) {
            var d = new FileReader;
            d.onload = function(b) {
                var d = b.target.result;
                c(a, d)
            }, d.readAsDataURL(b)
        }
    };
    module.exports = image
});
define("app/utils/drag_drop_helper", ["module", "require", "exports"], function(module, require, exports) {
    var dragDropHelper = {
        hasFiles: function(a, b) {
            var c = (a.originalEvent || a).dataTransfer, d = c && (c.types.contains ? c.types.contains("Files") : c.types.indexOf("Files") >= 0);
            if (!d ||!b ||!c.items&&!c.files)
                return d;
            var e = c.items || c.files;
            for (var f = 0; f < e.length; ++f)
                d = d&&!!e[f].type.match(b);
            return d
        },
        onlyHandleEventsWithFiles: function(a, b) {
            return function(c, d) {
                if (dragDropHelper.hasFiles(c, b))
                    return a.call(this, c, d)
            }
        }
    };
    module.exports = dragDropHelper
});
define("app/ui/with_drop_events", ["module", "require", "exports", "app/utils/image", "app/utils/drag_drop_helper"], function(module, require, exports) {
    function withDropEvents() {
        this.drop = function(a) {
            a.preventDefault(), a.stopImmediatePropagation();
            var b = image.getFileHandle(a.originalEvent.dataTransfer);
            this.trigger("uiDrop", {
                file: b
            }), this.trigger("uiDragEnd")
        }, this.after("initialize", function() {
            this.on("drop", dragDropHelper.onlyHandleEventsWithFiles(this.drop))
        })
    }
    var image = require("app/utils/image"), dragDropHelper = require("app/utils/drag_drop_helper");
    module.exports = withDropEvents
});
define("app/ui/with_droppable_image", ["module", "require", "exports", "core/compose", "app/ui/with_drop_events", "app/utils/image"], function(module, require, exports) {
    function withDroppableImage() {
        compose.mixin(this, [withDropEvents]), this.triggerGotImageData = function(a, b) {
            this.trigger("uiGotImageData", {
                name: a,
                contents: b
            })
        }, this.captureImageData = function(a, b) {
            var c = b.file;
            image.getFileData(c.name, c, this.triggerGotImageData.bind(this))
        }, this.after("initialize", function() {
            this.on("uiDrop", this.captureImageData)
        })
    }
    var compose = require("core/compose"), withDropEvents = require("app/ui/with_drop_events"), image = require("app/utils/image");
    module.exports = withDroppableImage
});
define("app/ui/with_image_upload_button", ["module", "require", "exports"], function(module, require, exports) {
    function withImageUploadButton() {
        this.defaultAttrs({
            focusClass: "focus",
            activeClass: "active",
            fileInputSelector: ".file-input",
            uploadPhotoSelector: ".photo-selector",
            imageBtnSelector: ".photo-selector .btn"
        }), this.hideAndDisableImageButton = function() {
            this.select("uploadPhotoSelector").hide()
        }, this.showAndEnableImageButton = function() {
            this.select("uploadPhotoSelector").show()
        }, this.applyFocusStyleImageButton = function(a) {
            this.select("imageBtnSelector").addClass(this.attr.focusClass)
        }, this.removeFocusStyleImageButton = function(a) {
            this.select("imageBtnSelector").removeClass(this.attr.focusClass)
        }, this.applyActiveStyleImageButton = function(a) {
            this.select("imageBtnSelector").addClass(this.attr.activeClass)
        }, this.removeActiveStyleImageButton = function(a) {
            this.select("imageBtnSelector").removeClass(this.attr.activeClass)
        }, this.after("initialize", function() {
            this.on("focusin", {
                fileInputSelector: this.applyFocusStyleImageButton
            }), this.on("focusout", {
                fileInputSelector: this.removeFocusStyleImageButton
            }), this.on("mousedown", {
                fileInputSelector: this.applyActiveStyleImageButton
            }), this.on("mouseup mouseleave", {
                fileInputSelector: this.removeActiveStyleImageButton
            }), this.on("uiEnableImageSelection", this.showAndEnableImageButton), this.on("uiDisableImageSelection", this.hideAndDisableImageButton)
        })
    }
    module.exports = withImageUploadButton
});
define("app/ui/with_lifeline", ["module", "require", "exports", "core/i18n"], function(module, require, exports) {
    function withLifeline() {
        this.defaultAttrs({
            lifelineAlertSelector: ".lifeline-alert-status",
            lifelineAlertBtnSelector: ".lifeline-alert-status-btn"
        }), this.sendLifelineAlertTweet = function() {
            this.sendTweet(), this.trigger("uiLifelineAlertsTweet")
        }, this.cancelLifelineAlertTweet = function() {
            this.trigger("uiLifelineAlertsCancel")
        }, this.around("confirmAndSendTweet", function(a) {
            this.lifelineAlertMode ? (this.trigger("uiLifelineAlertsImpression"), this.trigger("uiOpenConfirmDialog", {
                titleText: _('Send Twitter Alert'),
                bodyText: _('Are you sure you want to send a Twitter Alert to your followers?'),
                cancelText: _('Cancel'),
                submitText: _('Send'),
                action: "SendLifelineAlert"
            })) : a()
        }), this.setLifelineAlertMode = function(a) {
            a ? (this.select("lifelineAlertBtnSelector").addClass("active"), this.trigger("uiLifelineAlertsSelect")) : this.select("lifelineAlertBtnSelector").removeClass("active"), this.lifelineAlertMode = a, this.detectUpdatedText(!0)
        }, this.selectLifelineAlertAction = function() {
            this.setLifelineAlertMode(!this.lifelineAlertMode)
        }, this.around("addHiddenText", function(a, b) {
            var b = a(b);
            return this.lifelineAlertMode && (b += this.lifelineAlertText), b
        }), this.before("clear", function() {
            this.setLifelineAlertMode(!1)
        }), this.after("initialize", function() {
            this.lifelineAlertMode=!1, this.lifelineAlertText = " #alert", this.on(this.attr.lifelineAlertSelector, "click", this.selectLifelineAlertAction), this.on("uiSendLifelineAlertConfirm", this.sendLifelineAlertTweet), this.on("uiSendLifelineAlertCancel", this.cancelLifelineAlertTweet)
        })
    }
    var _ = require("core/i18n");
    module.exports = withLifeline
});
define("app/ui/with_rtl_tweet_box", ["module", "require", "exports", "lib/twitter-text", "app/utils/caret"], function(module, require, exports) {
    function replaceIndices(a, b, c) {
        var d = 0, e = "";
        return b(a).forEach(function(b) {
            e += a.slice(d, b.indices[0]) + c(a.slice(b.indices[0], b.indices[1])), d = b.indices[1]
        }), e + a.slice(d)
    }
    function withRTL() {
        this.defaultAttrs({
            isRTL: $("body").attr("dir") === "rtl",
            rtlCharRegex: /[\u0600-\u06FF]|[\u0750-\u077F]|[\u0590-\u05FF]|[\uFE70-\uFEFF]/gm,
            dirMarkRegex: /\u200e|\u200f/gm,
            rtlThreshold: .3
        }), this.shouldBeRTL = function(a, b, c) {
            c === undefined && (c = a.match(this.attr.rtlCharRegex));
            var d = a.trim();
            if (!d)
                return this.attr.isRTL;
            if (!c)
                return !1;
            var e = d.length - b;
            return e > 0 && c.length / e > this.attr.rtlThreshold
        }, this.removeMarkers = function(a) {
            return a.replace(this.attr.dirMarkRegex, "")
        }, this.setMarkersAndRTL = function(a, b) {
            var c = b.match(this.attr.rtlCharRegex), d = 0;
            if (c) {
                a = b, a = replaceIndices(a, txt.extractMentionsWithIndices, function(a) {
                    return d += a.length + 1, "‎" + a + "‏"
                });
                var e = this.attr.rtlCharRegex;
                a = replaceIndices(a, txt.extractHashtagsWithIndices, function(a) {
                    return a[1].match(e) ? a : "‎" + a
                }), a = replaceIndices(a, txt.extractUrlsWithIndices, function(a) {
                    return d += a.length + 2, a + "‎"
                })
            }
            var f = this.shouldBeRTL(b, d, c);
            return this.$text.attr("dir", f ? "rtl" : "ltr"), a
        }, this.erasePastMarkers = function(a) {
            if (a.which === 8)
                var b =- 1;
            else {
                if (a.which !== 46)
                    return;
                var b = 0
            }
            var c = caret.getPosition(this.$text[0]), d = this.$text.val(), e = 0;
            do {
                var f = d[c + b] || "";
                f && (c += b, e++, d = d.slice(0, c) + d.slice(c + 1))
            }
            while (f.match(this.attr.dirMarkRegex));
            e > 1 && (this.$text.val(d), caret.setPosition(this.$text[0], c), a.preventDefault(), this.detectUpdatedText())
        }, this.around("cleanText", function(a, b) {
            var c = this.removeMarkers(a(b)), d = this.setMarkersAndRTL(b, c);
            if (d !== b) {
                var e = this.$text[0], f = caret.getPosition(e);
                this.$text.val(d), this.prevText = d, caret.setPosition(e, f + d.length - b.length)
            }
            return c
        }), this.after("initTextNode", function() {
            this.on(this.$text, "keydown", this.erasePastMarkers)
        })
    }
    var txt = require("lib/twitter-text"), caret = require("app/utils/caret");
    module.exports = withRTL
});
define("app/ui/typeahead/accounts_renderer", ["module", "require", "exports", "core/i18n", "core/component"], function(module, require, exports) {
    function accountsRenderer() {
        this.defaultAttrs({
            accountsListSelector: ".js-typeahead-accounts",
            accountsItemSelector: ".typeahead-account-item",
            accountsShortcutSelector: ".typeahead-accounts-shortcut",
            accountsShortcutShow: !1,
            datasources: ["accounts"],
            socialContextMapping: {
                FOLLOWING: 1,
                FOLLOWS: 8
            }
        }), this.renderAccounts = function(a, b) {
            this.$accountsList.find(this.attr.accountsItemSelector).remove();
            var c = [];
            this.attr.datasources.forEach(function(a) {
                c = c.concat(b.suggestions[a] || [])
            });
            if (!c.length) {
                this.clearAccounts();
                return 
            }
            this.updateShortcut(b.queryData.query), c.forEach(function(a) {
                var b = this.$accountItemTemplate.clone(!1);
                b.attr("data-user-id", a.id), b.attr("data-user-screenname", a.screen_name), b.data("item", a);
                var c = b.find("a");
                c.attr("href", "/" + a.screen_name), c.attr("data-search-query", a.id), c.find(".avatar").attr("src", this.getAvatar(a)), c.find(".fullname").text(a.name), c.find(".username b").text(a.screen_name), a.verified && c.find(".js-verified").removeClass("hidden");
                if (this.attr.deciders.showDebugInfo) {
                    var d = a.prefetched;
                    c.attr("title", (d ? "local" : "remote") + " user, score: " + a.rounded_score)
                }
                if (a.social_proof !== 0 && this.attr.deciders.showSocialContext) {
                    var e = c.find(".typeahead-social-context"), f = this.getSocialContext(a);
                    f && (e.text(f), c.addClass("has-social-context"))
                }
                b.insertBefore(this.$accountsShortcut)
            }, this), this.$accountsList.addClass("has-results"), this.$accountsList.show()
        }, this.getAvatar = function(a) {
            var b = a.profile_image_url_https, c = this.attr.deciders.showSocialContext;
            return b && (b = b.replace(/^https?:/, ""), b = c ? b : b.replace(/_normal(\..*)?$/i, "_mini$1")), b
        }, this.isMutualFollow = function(a) {
            return this.currentUserFollowsAccount(a) && this.accountFollowsCurrentUser(a)
        }, this.currentUserFollowsAccount = function(a) {
            var b = this.attr.socialContextMapping.FOLLOWING;
            return !!(a & b)
        }, this.accountFollowsCurrentUser = function(a) {
            var b = this.attr.socialContextMapping.FOLLOWS;
            return !!(a & b)
        }, this.getSocialContext = function(a) {
            var b = a.social_proof;
            return this.isMutualFollow(b) ? _('You follow each other') : this.currentUserFollowsAccount(b) ? _('Following') : this.accountFollowsCurrentUser(b) ? _('Follows you') : a.first_connecting_user_name ? a.connecting_user_count > 1 ? _('Followed by {{user}} and {{number}} others', {
                user: a.first_connecting_user_name,
                number: a.connecting_user_count
            }) : _('Followed by {{user}}', {
                user: a.first_connecting_user_name
            }) : !1
        }, this.updateShortcut = function(a) {
            this.$accountsShortcut.toggle(this.attr.accountsShortcutShow);
            var b = this.$accountsShortcut.find("a");
            b.attr("href", "/search/users?q=" + encodeURIComponent(a)), b.attr("data-search-query", a), a = $("<div/>").text(a).html(), b.html(_('Search all people for <strong>{{query}}</strong>', {
                query: a
            }))
        }, this.clearAccounts = function() {
            this.$accountsList.removeClass("has-results"), this.$accountsList.hide()
        }, this.after("initialize", function() {
            this.$accountsList = this.select("accountsListSelector").first(), this.$accountsShortcut = this.select("accountsShortcutSelector").first(), this.$accountItemTemplate = this.select("accountsItemSelector").first().clone(!1), this.$accountsList.hide(), this.on("uiTypeaheadRenderResults", this.renderAccounts)
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component");
    module.exports = defineComponent(accountsRenderer)
});
define("app/ui/typeahead/contacts_renderer", ["module", "require", "exports", "core/i18n", "core/component"], function(module, require, exports) {
    function contactsRenderer() {
        this.defaultAttrs({
            contactsListSelector: ".js-typeahead-contacts",
            contactsItemSelector: ".typeahead-contact-item",
            emailSelector: ".email",
            fullnameSelector: ".fullname",
            datasources: ["contacts"]
        }), this.renderContacts = function(a, b) {
            this.$contactsList.find(this.attr.contactsItemSelector).remove();
            var c = [];
            this.attr.datasources.forEach(function(a) {
                c = c.concat(b.suggestions[a] || [])
            });
            if (!c.length) {
                this.clearContacts();
                return 
            }
            c.forEach(function(a) {
                var b = this.$contactItemTemplate.clone(!1), c = b.find("a");
                c.attr("data-search-query", a.email), c.find(this.attr.fullnameSelector).text(a.name), c.find(this.attr.emailSelector).text(a.email), this.$contactsList.append(b)
            }, this), this.$contactsList.addClass("has-results"), this.$contactsList.show()
        }, this.clearContacts = function() {
            this.$contactsList.removeClass("has-results"), this.$contactsList.html(""), this.$contactsList.hide()
        }, this.after("initialize", function() {
            this.$contactsList = this.select("contactsListSelector").first(), this.$contactItemTemplate = this.select("contactsItemSelector").first().clone(!1), this.$contactsList.hide(), this.on("uiTypeaheadRenderResults", this.renderContacts)
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component");
    module.exports = defineComponent(contactsRenderer)
});
define("app/ui/typeahead/saved_searches_renderer", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function savedSearchesRenderer() {
        this.defaultAttrs({
            savedSearchesListSelector: ".saved-searches-list",
            savedSearchesSelector: ".saved-searches-list",
            savedSearchesItemSelector: ".typeahead-saved-search-item",
            savedSearchesItemAnchorSelector: "a.js-nav",
            savedSearchesTitleSelector: ".saved-searches-title",
            savedSearchesBlockSelector: ".typeahead-saved-searches",
            savedSearchesRemoveSelector: ".typeahead-saved-search-item .close",
            datasources: ["savedSearches"]
        }), this.renderSavedSearches = function(a, b) {
            this.$savedSearchesList.empty();
            var c = [];
            this.attr.datasources.forEach(function(a) {
                c = c.concat(b.suggestions[a] || [])
            });
            if (c.length === 0) {
                this.$savedSearchesTitle.hide(), this.$savedSearchesBlock.removeClass("has-items");
                return 
            }
            var d = b.queryData.query === "", e = this.$savedSearchesTitle.attr("id");
            c.reverse(), c.forEach(function(a) {
                var b = this.$savedSearchItemTemplate.clone(!1);
                b.data("item", a);
                var c = b.find("a");
                c.attr("href", a.saved_search_path), c.attr("data-search-query", a.query), c.attr("data-query-source", a.search_query_source), c.attr("data-saved-search-id", a.id), c.append($("<span>").text(a.name)), d && c.attr("aria-describedby", e), this.$savedSearchesList.append(b)
            }, this), this.$savedSearchesTitle.toggle(d), this.$savedSearchesBlock.addClass("has-items")
        }, this.removeSavedSearch = function(a, b) {
            var c = $(a.target).closest(this.attr.savedSearchesItemSelector), d = c.find(this.attr.savedSearchesItemAnchorSelector), e = d.data("saved-search-id");
            c.siblings().length == 0 && (this.$savedSearchesTitle.hide(), this.$savedSearchesBlock.removeClass("has-items")), c.remove(), this.trigger("uiRemoveSavedSearch", {
                id: e,
                component: "top_bar_searchbox"
            })
        }, this.after("initialize", function() {
            this.$savedSearchItemTemplate = this.select("savedSearchesItemSelector").clone(!1), this.$savedSearchesList = this.select("savedSearchesSelector"), this.$savedSearchesTitle = this.select("savedSearchesTitleSelector"), this.$savedSearchesBlock = this.select("savedSearchesBlockSelector"), this.on("uiTypeaheadRenderResults", this.renderSavedSearches), this.on("click", {
                savedSearchesRemoveSelector: this.removeSavedSearch
            })
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(savedSearchesRenderer)
});
define("app/ui/typeahead/recent_searches_renderer", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function recentSearchesRenderer() {
        this.defaultAttrs({
            recentSearchesSelector: ".recent-searches-list",
            recentSearchesItemSelector: ".typeahead-recent-search-item",
            recentSearchesDismissSelector: ".typeahead-recent-search-item .close",
            recentSearchesBlockSelector: ".typeahead-recent-searches",
            recentSearchesTitleSelector: ".recent-searches-title",
            recentSearchesClearAllSelector: ".clear-recent-searches",
            datasources: ["recentSearches"]
        }), this.deleteRecentSearch = function(a, b) {
            var c = $(a.target).closest(this.attr.recentSearchesItemSelector), d = c.find("a.js-nav"), e = d.data("search-query");
            this.$recentSearchesList.children().length == 1 && (this.$recentSearchesTitle.hide(), this.$recentSearchesBlock.removeClass("has-results"), this.$recentSearchesBlock.removeClass("has-items"), this.$recentSearchesClearAll.hide()), c.remove(), this.trigger("uiTypeaheadDeleteRecentSearch", {
                query: e
            })
        }, this.deleteAllRecentSearches = function(a, b) {
            this.$recentSearchesList.empty(), this.$recentSearchesTitle.hide(), this.$recentSearchesBlock.removeClass("has-results"), this.$recentSearchesBlock.removeClass("has-items"), this.$recentSearchesClearAll.hide(), this.trigger("uiTypeaheadDeleteRecentSearch", {
                deleteAll: !0
            })
        }, this.renderRecentSearches = function(a, b) {
            this.$recentSearchesList.empty();
            var c = this.attr.datasources.map(function(a) {
                return b.suggestions[a] || []
            }).reduce(function(a, b) {
                return a.concat(b)
            });
            c.forEach(function(a) {
                var b = this.$recentSearchItemTemplate.clone(!1);
                b.data("item", a);
                var c = b.find("a");
                c.attr("href", a.recent_search_path), c.attr("data-search-query", a.name), c.attr("data-query-source", a.search_query_source), c.append($("<span>").text(a.name)), this.$recentSearchesList.append(b)
            }, this);
            var d = c.length !== 0, e = b.queryData.query === "", f = e && d;
            this.$recentSearchesBlock.toggleClass("has-results", !e && d), this.$recentSearchesBlock.toggleClass("has-items", d), this.$recentSearchesTitle.toggle(f), this.$recentSearchesClearAll.toggle(f)
        }, this.after("initialize", function() {
            this.$recentSearchItemTemplate = this.select("recentSearchesItemSelector").clone(!1), this.$recentSearchesList = this.select("recentSearchesSelector"), this.$recentSearchesBlock = this.select("recentSearchesBlockSelector"), this.$recentSearchesTitle = this.select("recentSearchesTitleSelector"), this.$recentSearchesClearAll = this.select("recentSearchesClearAllSelector"), this.on("click", {
                recentSearchesDismissSelector: this.deleteRecentSearch,
                recentSearchesClearAllSelector: this.deleteAllRecentSearches
            }), this.on("uiTypeaheadRenderResults", this.renderRecentSearches), this.on("uiTypeaheadDeleteAllRecentSearches", this.deleteAllRecentSearches)
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(recentSearchesRenderer)
});
define("app/ui/typeahead/topics_renderer", ["module", "require", "exports", "core/i18n", "core/component"], function(module, require, exports) {
    function topicsRenderer() {
        this.defaultAttrs({
            includeSearchGlass: !0,
            parseHashtags: !1,
            topicsListSelector: ".topics-list",
            topicsItemSelector: ".typeahead-topic-item",
            datasources: ["topics"],
            emptySocialContextClass: "empty-topics-social-context"
        }), this.isHashOrCashtag = function(a) {
            return a === "#" || a === "$"
        }, this.renderTopics = function(a, b) {
            this.$topicsList.empty();
            var c = [], d = b.queryData && b.queryData.query;
            this.attr.datasources.forEach(function(a) {
                c = c.concat(b.suggestions[a] || [])
            });
            if (!c.length) {
                this.clearTopics();
                return 
            }
            c.forEach(function(a) {
                var b = this.$topicsItemTemplate.clone(!1);
                b.data("item", a);
                var c = b.find("a"), e = a.topic || a.hashtag;
                c.attr("href", a.searchPath), c.attr("data-search-query", e);
                var f = e.charAt(0), g = this.attr.parseHashtags && this.isHashOrCashtag(f), h = a.location && this.attr.deciders.showTypeaheadTopicSocialContext;
                if (g) {
                    var i = $("<span>").text(f);
                    i.append($("<strong>").text(e.slice(1))), c.append(i)
                } else if (h) {
                    var j = c.find(".typeahead-social-context");
                    j.text(this.getSocialContext(a)), j.show(), c.children().last().before($("<span>").text(e))
                } else if (d && this.attr.deciders.reverseBoldingEnabled) {
                    var k = this.isHashOrCashtag(d.charAt(0)), l = this.isHashOrCashtag(e.charAt(0)), m=!k && l ? 1 : 0, i = $("<span>").text(e.substr(0, d.length + m));
                    i.append($("<strong>").text(e.slice(d.length + m))), c.append(i)
                } else 
                    a.name ? c.html(a.name) : c.append($("<span>").text(e)), this.attr.deciders.showTypeaheadTopicSocialContext && c.addClass(this.attr.emptySocialContextClass);
                b.appendTo(this.$topicsList)
            }, this), this.$topicsList.addClass("has-results"), this.$topicsList.show()
        }, this.getSocialContext = function(a) {
            return _('Trending in {{location}}', {
                location: a.location
            })
        }, this.clearTopics = function(a) {
            this.$topicsList.removeClass("has-results"), this.$topicsList.hide()
        }, this.after("initialize", function() {
            this.$topicsItemTemplate = this.select("topicsItemSelector").clone(!1), this.attr.includeSearchGlass || this.$topicsItemTemplate.find(".icon.generic-search").remove(), this.$topicsList = this.select("topicsListSelector"), this.$topicsList.hide(), this.on("uiTypeaheadRenderResults", this.renderTopics)
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component");
    module.exports = defineComponent(topicsRenderer)
});
define("app/ui/typeahead/trend_locations_renderer", ["module", "require", "exports", "core/component", "core/i18n"], function(module, require, exports) {
    function trendLocationsRenderer() {
        this.defaultAttrs({
            typeaheadItemClass: "typeahead-item",
            trendLocationsListSelector: ".typeahead-trend-locations-list",
            trendLocationsItemSelector: ".typeahead-trend-locations-item",
            datasources: ["trendLocations"]
        }), this.renderTrendLocations = function(a, b) {
            this.$trendLocationsList.empty();
            var c = [];
            this.attr.datasources.forEach(function(a) {
                c = c.concat(b.suggestions[a] || [])
            }), c.forEach(function(a) {
                var b = this.$trendLocationItemTemplate.clone(!1), c = b.find("a");
                b.data("item", a), c.attr("data-search-query", a.name), c.attr("href", "#"), c.append(this.getLocationHtml(a)), a.woeid==-1 && (b.removeClass(this.attr.typeaheadItemClass), c.attr("data-search-query", "")), b.appendTo(this.$trendLocationsList)
            }, this)
        }, this.getLocationHtml = function(a) {
            var b = $("<span>");
            switch (a.placeTypeCode) {
            case placeTypeMapping.WORLDWIDE:
            case placeTypeMapping.NOT_FOUND:
                b.text(a.name);
                break;
            case placeTypeMapping.COUNTRY:
                b.html(a.name + "  " + _('(All cities)'));
                break;
            default:
                b.text([a.name, a.countryName].join(", "))
            }
            return b
        }, this.after("initialize", function() {
            this.$trendLocationItemTemplate = this.select("trendLocationsItemSelector").clone(!1), this.$trendLocationsList = this.select("trendLocationsListSelector"), this.on("uiTypeaheadRenderResults", this.renderTrendLocations)
        })
    }
    var defineComponent = require("core/component"), _ = require("core/i18n");
    module.exports = defineComponent(trendLocationsRenderer);
    var placeTypeMapping = {
        WORLDWIDE: 19,
        COUNTRY: 12,
        CITY: 7,
        NOT_FOUND: - 1
    }
});
define("app/ui/typeahead/concierge_renderer", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function conciergeRenderer() {
        this.defaultAttrs({
            conciergeBlockSelector: ".typeahead-concierge",
            conciergeListSelector: ".typeahead-concierge-list",
            conciergeItemSelector: ".typeahead-concierge-item",
            datasource: "concierge"
        }), this.renderConcierge = function(a, b) {
            this.$conciergeList.empty();
            var c = b.suggestions && b.suggestions[this.attr.datasource] || [], d=!!c.length;
            c.forEach(function(a) {
                var b = this.$conciergeItemTemplate.clone(!1), c = b.find("a");
                b.data("item", a), c.attr("href", a.searchPath), c.attr("data-search-query", a.topic), c.attr("data-query-source", a.querySource), c.append($("<span>").text(a.name)), this.$conciergeList.append(b)
            }, this), this.$conciergeBlock.toggleClass("has-items", d), this.$conciergeBlock.toggle(d)
        }, this.after("initialize", function() {
            this.$conciergeItemTemplate = this.select("conciergeItemSelector").clone(!1), this.$conciergeBlock = this.select("conciergeBlockSelector"), this.$conciergeList = this.select("conciergeListSelector"), this.on("uiTypeaheadRenderResults", this.renderConcierge)
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(conciergeRenderer)
});
define("app/ui/typeahead/selected_users_renderer", ["module", "require", "exports", "core/i18n", "core/component"], function(module, require, exports) {
    function mediaTaggingRenderer() {
        function a(a, b) {
            if (!b)
                return !1;
            for (var c = 0; c < b.length; c++)
                if (a.id == b[c].id)
                    return !0;
            return !1
        }
        this.defaultAttrs({
            rootSelector: ".typeahead-user-select",
            accountsListSelector: ".js-typeahead-accounts",
            accountsItemSelector: ".typeahead-account-item",
            selectedItemSelector: ".typeahead-selected-item",
            accountsEndPlaceholderSelector: ".typeahead-accounts-end",
            selectedEndPlaceholderSelector: ".typeahead-selected-end",
            emptyTextSelector: ".typeahead-empty-suggestions",
            selectedListSelector: ".js-typeahead-selected",
            mediaTaggingSelector: ".tagging-dropdown",
            photoTaggingContainerClass: "photo-tagging-container",
            datasources: ["accounts", "selectedUsers"],
            socialContextMapping: {
                FOLLOWING: 1,
                FOLLOWS: 8
            }
        }), this.renderAccounts = function(a, b) {
            this.$accountsList.find(this.attr.accountsItemSelector).remove(), this.$selectedList.find(this.attr.selectedItemSelector).remove();
            var c = b.suggestions.selectedUsers || [], d;
            b.queryData ? d = b.queryData.query || "" : d = "";
            var e = [];
            this.attr.datasources.forEach(function(a) {
                if (a == "accounts" || a == "mediaTagAccounts" || a == "dmAccounts" || a == "prefillUsers" && d.length == 0 && c.length == 0)
                    e = e.concat(b.suggestions[a] || [])
            }), e && e.length > 0 ? (e.forEach(function(a) {
                var b = this.$accountItemTemplate.clone(!1);
                this.addUser(b, this.$accountsEnd, a, c)
            }, this), this.$accountsList.addClass("has-results"), this.$accountsList.show()) : this.clearAccounts();
            var f = e && e.length > 0, g = d.length > 0, h = c && c.length > 0;
            !f&&!g && h ? (c.forEach(function(a) {
                var b = this.$selectedItemTemplate.clone(!1);
                this.addUser(b, this.$selectedEnd, a, c)
            }, this), this.$selectedList.addClass("has-results"), this.$selectedList.show()) : this.clearSelected();
            var i = this.select("emptyTextSelector").first(), j = b.suggestions.prefillUsers && e.length - b.suggestions.prefillUsers.length == 0;
            j && c.length == 0 && e.length > 0 ? i.show() : i.hide(), a.stopPropagation()
        }, this.addUser = function(b, c, d, e) {
            b.attr("data-user-id", d.id), b.attr("data-user-screenname", d.screen_name), b.data("item", d);
            var f = b.find("a");
            f.attr("href", "/" + d.screen_name), f.attr("data-search-query", d.id), f.find(".avatar").attr("src", this.getAvatar(d)), f.find(".fullname").text(d.name), f.find(".username b").text(d.screen_name), d.verified && f.find(".js-verified").removeClass("hidden"), a(d, e) ? (b.addClass("selected-user"), f.attr("aria-selected", "true")) : f.attr("aria-selected", "false");
            var g = this.$node.hasClass(this.attr.photoTaggingContainerClass);
            g && d.can_media_tag===!1 && b.addClass("cannot-select");
            if (this.attr.deciders.showDebugInfo) {
                var h = d.prefetched;
                f.attr("title", (h ? "local" : "remote") + " user, score: " + d.rounded_score)
            }
            if (d.social_proof !== 0 && this.attr.deciders.showSocialContext) {
                var i = f.find(".typeahead-social-context"), j = this.getSocialContext(d);
                j && (i.text(j), f.addClass("has-social-context"))
            }
            b.insertBefore(c)
        }, this.getAvatar = function(a) {
            var b = a.profile_image_url_https, c = this.attr.deciders.showSocialContext;
            return b && (b = b.replace(/^https?:/, ""), b = c ? b : b.replace(/_normal(\..*)?$/i, "_mini$1")), b
        }, this.isMutualFollow = function(a) {
            return this.currentUserFollowsAccount(a) && this.accountFollowsCurrentUser(a)
        }, this.currentUserFollowsAccount = function(a) {
            var b = this.attr.socialContextMapping.FOLLOWING;
            return !!(a & b)
        }, this.accountFollowsCurrentUser = function(a) {
            var b = this.attr.socialContextMapping.FOLLOWS;
            return !!(a & b)
        }, this.getSocialContext = function(a) {
            var b = a.social_proof;
            return this.isMutualFollow(b) ? _('You follow each other') : this.currentUserFollowsAccount(b) ? _('Following') : this.accountFollowsCurrentUser(b) ? _('Follows you') : a.first_connecting_user_name ? a.connecting_user_count > 1 ? _('Followed by {{user}} and {{number}} others', {
                user: a.first_connecting_user_name,
                number: a.connecting_user_count
            }) : _('Followed by {{user}}', {
                user: a.first_connecting_user_name
            }) : !1
        }, this.clearAccounts = function() {
            this.$accountsList.removeClass("has-results"), this.$accountsList.hide()
        }, this.clearSelected = function() {
            this.$selectedList.removeClass("has-results"), this.$selectedList.hide()
        }, this.after("initialize", function() {
            this.$accountsList = this.select("accountsListSelector"), this.$selectedList = this.select("selectedListSelector"), this.$accountsEnd = this.select("accountsEndPlaceholderSelector"), this.$accountItemTemplate = this.select("accountsItemSelector").clone(!1), this.$selectedItemTemplate = this.select("selectedItemSelector").clone(!1), this.$selectedEnd = this.select("selectedEndPlaceholderSelector"), this.$accountsList.hide(), this.on("uiTypeaheadRenderResults", this.renderAccounts)
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component");
    module.exports = defineComponent(mediaTaggingRenderer)
});
define("app/utils/rtl_text", ["module", "require", "exports", "lib/twitter-text"], function(module, require, exports) {
    var TwitterText = require("lib/twitter-text"), RTLText = function() {
        function q(a) {
            try {
                return document.activeElement === a
            } catch (b) {
                return !1
            }
        }
        function r(a) {
            if (!q(a))
                return 0;
            var b;
            if (typeof a.selectionStart == "number")
                return a.selectionStart;
            if (document.selection) {
                a.focus(), b = document.selection.createRange(), b.moveStart("character", - a.value.length);
                var c = b.text.length;
                return c
            }
        }
        function s(a, b) {
            if (!q(a))
                return;
            if (typeof a.selectionStart == "number")
                a.selectionStart = b, a.selectionEnd = b;
            else if (document.selection) {
                var c = a.createTextRange();
                c.collapse(!0), c.moveEnd("character", b), c.moveStart("character", b), c.select()
            }
        }
        function t(a, b, c) {
            var d = 0, e = "", f = b(a);
            for (var g = 0; g < f.length; g++) {
                var h = f[g], i = "";
                h.screenName && (i = "screenName"), h.hashtag && (i = "hashtag"), h.url && (i = "url"), h.cashtag && (i = "cashtag");
                var j = {
                    entityText: a.slice(h.indices[0], h.indices[1]),
                    entityType: i
                };
                e += a.slice(d, h.indices[0]) + c(j), d = h.indices[1]
            }
            return e + a.slice(d, a.length)
        }
        function u(a) {
            var b = a.match(c), d = a;
            if (b || l === "rtl")
                d = t(d, TwitterText.extractEntitiesWithIndices, function(a) {
                    if (a.entityType === "screenName")
                        return e + a.entityText + f;
                        if (a.entityType === "hashtag")
                            return a.entityText.charAt(1).match(c) ? a.entityText : e + a.entityText;
                            if (a.entityType === "url")
                                return a.entityText + e;
                                if (a.entityType === "cashtag")
                                    return e + a.entityText
                                });
            return d
        }
        function v(a) {
            var b, c = a.target ? a.target: a.srcElement, e = a.which ? a.which: a.keyCode;
            if (e === g.BACKSPACE)
                b =- 1;
            else {
                if (e !== g.DELETE)
                    return;
                b = 0
            }
            var f = r(c), h = c.value, i = 0, j;
            do 
                j = h.charAt(f + b) || "", j && (f += b, i++, h = h.slice(0, f) + h.slice(f + 1, h.length));
            while (j.match(d));
            i > 1 && (c.value = h, s(c, f), a.preventDefault ? a.preventDefault() : a.returnValue=!1)
        }
        function w(a) {
            return a.replace(d, "")
        }
        function x(a) {
            var d = a.match(c);
            a = a.replace(k, "");
            var e = 0, f = a.replace(m, ""), g = l;
            if (!f ||!f.replace(/#/g, ""))
                return g === "rtl"?!0 : !1;
            if (!d)
                return !1;
            if (a) {
                var h = TwitterText.extractMentionsWithIndices(a), i = h.length, j;
                for (j = 0; j < i; j++)
                    e += h[j].screenName.length + 1;
                var n = TwitterText.extractUrlsWithIndices(a), o = n.length;
                for (j = 0; j < o; j++)
                    e += n[j].url.length + 2
            }
            var p = f.length - e;
            return p > 0 && d.length / p > b
        }
        function y(a) {
            var b = a.target || a.srcElement;
            a.type !== "keydown" || a.keyCode !== 91 && a.keyCode !== 16 && a.keyCode !== 88 && a.keyCode !== 17 ? a.type === "keyup" && (a.keyCode === 91 || a.keyCode === 16 || a.keyCode === 88 || a.keyCode === 17) && (o[String(a.keyCode)]=!1) : o[String(a.keyCode)]=!0, (!p && o[91] || p && o[17]) && o[16] && o[88] && (n=!0, b.dir === "rtl" ? z("ltr", b) : z("rtl", b), o = {
                91: !1,
                16: !1,
                88: !1,
                17: !1
            })
        }
        function z(a, b) {
            b.setAttribute("dir", a), b.style.direction = a, b.style.textAlign = a === "rtl" ? "right" : "left"
        }
        "use strict";
        var a = {}, b = .3, c = /[\u0590-\u083F]|[\u08A0-\u08FF]|[\uFB1D-\uFDFF]|[\uFE70-\uFEFF]/gm, d = /\u200e|\u200f/gm, e = "‎", f = "‏", g = {
            BACKSPACE: 8,
            DELETE: 46
        }, h = 0, i = 20, j=!1, k = "", l = "", m = /^\s+|\s+$/g, n=!1, o = {
            91: !1,
            16: !1,
            88: !1,
            17: !1
        }, p = navigator.userAgent.indexOf("Mac")===-1;
        return a.onTextChange = function(b) {
            var c = b || window.event;
            y(b), c.type === "keydown" && v(c), a.setText(c.target || c.srcElement)
        }, a.setText = function(a) {
            l || (a.style.direction ? l = a.style.direction : a.dir ? l = a.dir : document.body.style.direction ? l = document.body.style.direction : l = document.body.dir), arguments.length === 2 && (l = a.ownerDocument.documentElement.className, k = arguments[1]);
            var b = a.value;
            if (!b)
                return;
            var c = w(b);
            j = x(c);
            var d = u(c), e = j ? "rtl": "ltr";
            d !== b && (a.value = d, s(a, r(a) + d.length - c.length)), n || z(e, a)
        }, a.textLength = function(a) {
            var b = w(a), c = TwitterText.extractUrls(b), d = b.length - c.join("").length, e = c.length;
            for (var f = 0; f < e; f++)
                d += i, /^https:/.test(c[f]) && (d += 1);
            return h = d
        }, a.cleanText = function(a) {
            return w(a)
        }, a.addRTLMarkers = function(a) {
            return u(a)
        }, a.shouldBeRTL = function(a) {
            return x(a)
        }, a
    }();
    typeof module != "undefined" && module.exports && (module.exports = RTLText)
});
define("app/ui/typeahead/typeahead_dropdown", ["module", "require", "exports", "core/component", "core/utils", "app/ui/typeahead/accounts_renderer", "app/ui/typeahead/contacts_renderer", "app/ui/typeahead/saved_searches_renderer", "app/ui/typeahead/recent_searches_renderer", "app/ui/typeahead/topics_renderer", "app/ui/typeahead/trend_locations_renderer", "app/ui/typeahead/concierge_renderer", "app/ui/typeahead/selected_users_renderer", "app/utils/rtl_text"], function(module, require, exports) {
    function typeaheadDropdown() {
        this.defaultAttrs({
            inputSelector: "#search-query",
            dropdownSelector: ".dropdown-menu.typeahead",
            itemsContainerSelector: ".typeahead-items",
            itemsSelector: ".typeahead-items li",
            itemAnchorsSelector: ".typeahead-items li a",
            itemSelector: ".typeahead-item",
            categoryTitle: ".typeahead-category-title",
            selectedItemSelector: ".typeahead-items li.selected",
            selectedClass: "selected",
            blockLinkActions: !1,
            alwaysOpen: !1,
            deciders: {
                showDebugInfo: !1,
                showSocialContext: !1,
                showTypeaheadTopicSocialContext: !1
            },
            autocompleteAccounts: !0,
            datasourceRenders: [["concierge", ["concierge"]], ["savedSearches", ["savedSearches"]], ["recentSearches", ["recentSearches"]], ["topics", ["topics"]], ["accounts", ["accounts"]]],
            datasourceOptions: {},
            typeaheadSrc: "UNKNOWN",
            templateContainerSelector: ".dropdown-inner",
            recentSearchesListSelector: ".typeahead-recent-searches",
            savedSearchesListSelector: ".typeahead-saved-searches",
            topicsListSelector: ".typeahead-topics",
            accountsListSelector: ".js-typeahead-accounts",
            contactsListSelector: ".js-typeahead-contacts",
            trendLocationsListSelector: ".typeahead-trend-locations-list",
            conciergeSelector: ".typeahead-concierge",
            userSelectSelector: ".typeahead-user-select",
            selectedListSelector: ".typeahead-selected",
            renderLimit: undefined
        }), this.mouseOver = function(a, b) {
            this.clearSelected(), $(b.el).addClass(this.attr.selectedClass)
        }, this.clearSelected = function() {
            this.select("itemsSelector").removeClass(this.attr.selectedClass)
        }, this.moveSelection = function(a) {
            var b = this.select("itemsSelector").filter(":visible"), c = b.filter(".selected");
            c.removeClass(this.attr.selectedClass), c.find("a").removeAttr("aria-selected"), this.$input.removeAttr("aria-activedescendant");
            var d = b.index(c) + a;
            d = (d + 1)%(b.length + 1) - 1;
            if (d===-1) {
                this.trigger("uiTypeaheadSelectionCleared");
                return 
            }
            d<-1 && (d = b.length - 1);
            var e = b.eq(d), f = e.find("a");
            e.addClass(this.attr.selectedClass), f.attr("aria-selected", !0), this.$input.attr("aria-activedescendant", f.attr("id"))
        }, this.moveSelectionUp = function(a) {
            this.moveSelection( - 1)
        }, this.moveSelectionDown = function(a) {
            this.moveSelection(1)
        }, this.dropdownIsOpen = function() {
            if (this.attr.alwaysOpen)
                return !0;
            if (window.DEBUG && window.DEBUG.enabled && this.openState !== this.$dropdown.is(":visible"))
                throw new Error("Dropdown markup and internal openState variable are out of sync.");
            return this.openState
        }, this.show = function() {
            this.$dropdown.show(), this.$input.attr("aria-expanded", !0), this.openState=!0
        }, this.hide = function(a) {
            if (this.mouseIsOverDropdown)
                return;
            if (!this.dropdownIsOpen() || this.attr.alwaysOpen)
                return;
            this.$dropdown.find(".selected a").removeAttr("aria-selected"), this.$dropdown.hide(), this.$input.attr("aria-expanded", !1), this.$input.removeAttr("aria-activedescendant"), this.openState=!1
        }, this.hideAndManageEsc = function(a) {
            if (!this.dropdownIsOpen() || this.attr.alwaysOpen)
                return;
            this.forceHide(), a.preventDefault(), a.stopPropagation()
        }, this.forceHide = function() {
            this.clearMouseTracking(), this.hide()
        }, this.inputValueUpdated = function(a, b) {
            this.lastQuery = b.value;
            var c = utils.merge(this.attr.datasourceOptions, {
                query: b.value,
                tweetContext: b.tweetContext,
                typeaheadSrc: this.attr.typeaheadSrc
            });
            this.trigger("uiNeedsTypeaheadSuggestions", {
                datasources: this.datasources,
                queryData: c,
                id: this.getDropdownId()
            })
        }, this.getDropdownId = function() {
            return this.dropdownId || (this.dropdownId = "swift_typeahead_dropdown_" + Math.floor(Math.random() * 1e6)), this.dropdownId
        }, this.checkIfSelectionFromSearchInput = function(a) {
            return a.closest("form").find("input").hasClass("search-input")
        }, this.triggerSelectionEvent = function(a, b) {
            this.attr.blockLinkActions && a.preventDefault();
            var c = this.select("itemsSelector"), d = c.filter(".selected").first();
            if (d.length == 0)
                return;
            var e = d.find("a"), f = d.index(), g = this.lastQuery, h = e.attr("data-search-query"), i = e.attr("data-query-source");
            d.removeClass(this.attr.selectedClass), this.$input.removeAttr("aria-activedescendant");
            if (!g&&!h && i !== "typeahead_oneclick")
                return;
            var j = this.getItemData(d);
            this.trigger("uiTypeaheadItemSelected", {
                isSearchInput: this.checkIfSelectionFromSearchInput(e),
                index: f,
                source: e.data("ds"),
                query: h,
                input: g,
                display: d.data("user-screenname") || h,
                href: e.attr("href"),
                isClick: a.originalEvent ? a.originalEvent.type === "click": !1,
                item: j
            }), this.forceHide()
        }, this.getItemData = function(a) {
            return a.data("item")
        }, this.submitQuery = function(a, b) {
            var c = this.select("itemsSelector").filter(".selected").first();
            if (c.length) {
                this.triggerSelectionEvent(a, b);
                return 
            }
            var d = this.$input.val();
            if (d.trim() === "")
                return;
            this.trigger("uiTypeaheadSubmitQuery", {
                query: RTLText.cleanText(d)
            }), this.forceHide()
        }, this.getSelectedCompletion = function() {
            var a = this.select("itemsSelector").filter(".selected").first();
            !a.length && this.dropdownIsOpen() && (a = this.select("itemsSelector").filter(".typeahead-item").first());
            if (!a.length || this.$dropdown.attr("id") != a.closest(this.attr.dropdownSelector).attr("id"))
                return;
            var b = a.find("a"), c = b.data("search-query"), d = this.select("itemsSelector"), e = d.index(a), f = this.lastQuery;
            if (b.data("ds") == "account"&&!this.attr.autocompleteAccounts)
                return;
            var g = this.getItemData(a);
            this.trigger("uiTypeaheadItemPossiblyComplete", {
                value: c,
                source: b.data("ds"),
                index: e,
                query: c,
                item: g,
                display: a.data("user-screenname") || c,
                input: f,
                href: b.attr("href") || ""
            })
        }, this.renderResults = function(a) {
            if (!this.attr.renderLimit) {
                this.trigger("uiTypeaheadRenderResults", a);
                return 
            }
            var b = 0, c = 0, d = 0, e = {}, f = a.suggestions, g = this.attr.renderLimit;
            this.datasources.forEach(function(f) {
                a.suggestions[f] && d < g && (b = g - d, c = a.suggestions[f].length < b ? a.suggestions[f].length : b, e[f] = a.suggestions[f].slice(0, c), d += c)
            }), a.suggestions = e, this.trigger("uiTypeaheadRenderResults", a), a.suggestions = f
        }, this.updateDropdown = function(a, b) {
            var c = this.$input.is(document.activeElement);
            if (b.id !== this.getDropdownId() ||!b.queryData.atSignRemoved && b.queryData.query !== this.lastQuery || b.queryData.atSignRemoved && b.queryData.query !== this.lastQuery.substring(1) ||!c&&!this.attr.alwaysOpen)
                return;
            var d = this.select("itemsSelector").filter(".selected").first(), e = d.find("a").data("ds"), f = d.find("a").data("search-query");
            this.renderResults(b);
            if (e && f) {
                var g = this.select("itemsSelector").find("[data-ds='" + e + "'][data-search-query='" + f + "']");
                g.closest("li").addClass(this.attr.selectedClass)
            }
            var h = "typeahead-item-";
            this.select("itemAnchorsSelector").each(function(a, b) {
                $(b).attr("id", h + a)
            });
            var i = this.datasources.map(function(a) {
                return b.suggestions[a] ? b.suggestions[a].length : 0
            }), j = i.reduce(function(a, b) {
                return a + b
            }), k=!!b.queryData.query;
            j > 0 && (c || this.attr.alwaysOpen) ? (this.show(), this.trigger("uiTypeaheadSetPreventDefault", {
                preventDefault: k,
                key: 9
            }), this.trigger("uiTypeaheadResultsShown", {
                numResults: j,
                query: b.queryData.query
            })) : (this.forceHide(), this.trigger("uiTypeaheadSetPreventDefault", {
                preventDefault: !1,
                key: 9
            }), this.trigger("uiTypeaheadResultsHidden"))
        }, this.updateNumResults = function(a, b) {
            this.select("itemSelector").length === 0&&!this.attr.alwaysOpen && this.forceHide()
        }, this.trackMouse = function(a, b) {
            this.mouseIsOverDropdown=!0
        }, this.clearMouseTracking = function(a, b) {
            this.mouseIsOverDropdown=!1, this.clearSelected()
        }, this.resetTemplates = function() {
            this.$templateContainer.empty(), this.$templateContainer.append(this.$conciergeTemplate), this.$templateContainer.append(this.$savedSearchesTemplate), this.$templateContainer.append(this.$recentSearchesTemplate), this.$templateContainer.append(this.$topicsTemplate), this.$templateContainer.append(this.$accountsTemplate), this.$templateContainer.append(this.$contactsTemplate), this.$templateContainer.append(this.$trendLocationsTemplate), this.$templateContainer.append(this.$userSelectTemplate)
        }, this.addRenderer = function(a, b, c) {
            c = utils.merge(c, {
                datasources: b
            });
            var d = "block" + this.blockCount++;
            a === "accounts" ? (this.$accountsTemplate.clone().addClass(d).appendTo(this.$templateContainer), AccountsRenderer.attachTo(this.$node, utils.merge(c, {
                accountsListSelector: this.attr.accountsListSelector + "." + d
            }))) : a === "contacts" ? (this.$contactsTemplate.clone().addClass(d).appendTo(this.$templateContainer), ContactsRenderer.attachTo(this.$node, utils.merge(c, {
                contactsListSelector: this.attr.contactsListSelector + "." + d
            }))) : a === "topics" ? (this.$topicsTemplate.clone().addClass(d).appendTo(this.$templateContainer), TopicsRenderer.attachTo(this.$node, utils.merge(c, {
                topicsListSelector: this.attr.topicsListSelector + "." + d
            }))) : a === "savedSearches" ? (this.$savedSearchesTemplate.clone().addClass(d).appendTo(this.$templateContainer), SavedSearchesRenderer.attachTo(this.$node, utils.merge(c, {
                savedSearchesListSelector: this.attr.savedSearchesListSelector + "." + d
            }))) : a === "recentSearches" ? (this.$recentSearchesTemplate.clone().addClass(d).appendTo(this.$templateContainer), RecentSearchesRenderer.attachTo(this.$node, utils.merge(c, {
                recentSearchesListSelector: this.attr.recentSearchesListSelector + "." + d
            }))) : a === "trendLocations" ? (this.$trendLocationsTemplate.clone().addClass(d).appendTo(this.$templateContainer), TrendLocationsRenderer.attachTo(this.$node, utils.merge(c, {
                trendLocationsListSelector: this.attr.trendLocationsListSelector + "." + d
            }))) : a === "concierge" ? (this.$conciergeTemplate.clone().addClass(d).appendTo(this.$templateContainer), ConciergeRenderer.attachTo(this.$node, utils.merge(c, {
                conciergeSelector: this.attr.conciergeSelector + "." + d
            }))) : a === "selectedUsers" && (this.$userSelectTemplate.clone().addClass(d).appendTo(this.$templateContainer), SelectedUsersRenderer.attachTo(this.$node, utils.merge(c, {
                rootSelector: this.attr.userSelectSelector + "." + d,
                accountsListSelector: this.attr.accountsListSelector,
                selectedListSelector: this.attr.selectedListSelector
            })), this.$dropdown.attr("aria-multiselectable", "true"))
        }, this.applyARIAToInput = function() {
            instances += 1;
            var a = "typeahead-dropdown-" + instances;
            this.$dropdown.attr("id", a), this.$input.attr({
                "aria-autocomplete": "list",
                "aria-expanded": !1,
                "aria-owns": a
            })
        }, this.before("teardown", this.resetTemplates), this.after("initialize", function(a, b) {
            this.openState=!1, this.$input = this.select("inputSelector").first(), this.$dropdown = this.select("dropdownSelector").first(), this.applyARIAToInput(), this.$templateContainer = this.select("templateContainerSelector").first(), this.$accountsTemplate = this.select("accountsListSelector").first().clone(!1), this.$contactsTemplate = this.select("contactsListSelector").first().clone(!1), this.$savedSearchesTemplate = this.select("savedSearchesListSelector").first().clone(!1), this.$recentSearchesTemplate = this.select("recentSearchesListSelector").first().clone(!1), this.$topicsTemplate = this.select("topicsListSelector").first().clone(!1), this.$trendLocationsTemplate = this.select("trendLocationsListSelector").first().clone(!1), this.$conciergeTemplate = this.select("conciergeSelector").first().clone(!1), this.$userSelectTemplate = this.select("userSelectSelector").first().clone(!1), this.$templateContainer.empty(), this.datasources = [], this.attr.datasourceRenders.forEach(function(a) {
                this.datasources = this.datasources.concat(a[1])
            }, this), this.datasources = utils.uniqueArray(this.datasources), this.blockCount = 0, this.attr.datasourceRenders.forEach(function(a) {
                this.addRenderer(a[0], a[1], b)
            }, this), this.on(this.$input, "blur", this.hide), this.on(this.$input, "uiTypeaheadInputSubmit", this.submitQuery), this.on(this.$input, "uiTypeaheadInputChanged", this.inputValueUpdated), this.on(this.$input, "uiTypeaheadInputMoveUp", this.moveSelectionUp), this.on(this.$input, "uiTypeaheadInputMoveDown", this.moveSelectionDown), this.on(this.$input, "uiTypeaheadInputAutocomplete", this.getSelectedCompletion), this.on(this.$input, "uiTypeaheadInputTab", this.clearMouseTracking), this.on(this.$input, "uiShortcutEsc", this.hideAndManageEsc), this.on(this.$dropdown, "mouseenter", this.trackMouse), this.on(this.$dropdown, "mouseleave", this.clearMouseTracking), this.on(document, "dataTypeaheadSuggestionsResults", this.updateDropdown), this.on(document, "uiBeforePageChanged", this.forceHide), this.on("mouseover", {
                itemsSelector: this.mouseOver
            }), this.on("click", {
                itemsSelector: this.triggerSelectionEvent
            }), this.on("uiTypeaheadDeleteRecentSearch uiRemoveSavedSearch", this.updateNumResults), this.attr.alwaysOpen && (this.$input.attr("aria-expanded", !0), this.$dropdown.show().attr("aria-hidden", !1), this.openState=!0)
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), AccountsRenderer = require("app/ui/typeahead/accounts_renderer"), ContactsRenderer = require("app/ui/typeahead/contacts_renderer"), SavedSearchesRenderer = require("app/ui/typeahead/saved_searches_renderer"), RecentSearchesRenderer = require("app/ui/typeahead/recent_searches_renderer"), TopicsRenderer = require("app/ui/typeahead/topics_renderer"), TrendLocationsRenderer = require("app/ui/typeahead/trend_locations_renderer"), ConciergeRenderer = require("app/ui/typeahead/concierge_renderer"), SelectedUsersRenderer = require("app/ui/typeahead/selected_users_renderer"), RTLText = require("app/utils/rtl_text"), instances = 0;
    module.exports = defineComponent(typeaheadDropdown)
});
define("app/utils/event_support", ["module", "require", "exports"], function(module, require, exports) {
    var supportedEvents = {}, checkEventsSupport = function(a, b) {
        return a.forEach(function(a) {
            checkEventSupported(a, b[a])
        }, this), supportedEvents
    }, checkEventSupported = function(a, b) {
        var c = document.createElement(b || "div"), d = "on" + a, e = d in c;
        return e || (c.setAttribute(d, "return;"), e = typeof c[d] == "function"), c = null, supportedEvents[a] = e, e
    }, eventSupport = {
        checkEvents: function(a, b) {
            checkEventsSupport(a, b || {})
        },
        browserSupports: function(a, b) {
            return supportedEvents[a] === undefined && checkEventSupported(a, b), supportedEvents[a]
        }
    };
    module.exports = eventSupport
});
define("app/utils/typeahead_helpers", ["module", "require", "exports"], function(module, require, exports) {
    function tokenizeText(a) {
        return a.trim().toLowerCase().split(/[\s_,.-]+/)
    }
    function getFirstChar(a) {
        var b;
        return multiByteRegex.test(a.substr(0, 1)) ? b = a.substr(0, 2) : b = a.charAt(0), b
    }
    var multiByteRegex = /[\uD800-\uDFFF]/;
    module.exports = {
        tokenizeText: tokenizeText,
        getFirstChar: getFirstChar
    }
});
define("app/ui/typeahead/typeahead_input", ["module", "require", "exports", "core/component", "app/utils/caret", "app/utils/event_support", "app/utils/typeahead_helpers", "app/utils/string", "lib/twitter-text", "app/utils/rtl_text"], function(module, require, exports) {
    function typeaheadInput() {
        this.defaultAttrs({
            inputSelector: "#search-query",
            buttonSelector: ".nav-search",
            dropdownSelectedItemSelector: ".typeahead-items li.selected",
            completeAllEntities: !1,
            includeTweetContext: !1,
            tweetContextEnabled: !1,
            allowAccountsWithoutAtSign: !1,
            inputIsTweetbox: !1,
            triggerChangeTextAndPosition: !1
        }), this.getDefaultKeycodes = function() {
            var a = {
                13: {
                    name: "ENTER",
                    event: "uiTypeaheadInputSubmit",
                    on: "keypress",
                    preventDefault: !0,
                    enabled: !0
                },
                9: {
                    name: "TAB",
                    event: "uiTypeaheadInputTab",
                    on: "keydown",
                    preventDefault: !0,
                    canCauseComplete: !0,
                    enabled: !0
                },
                37: {
                    name: "LEFT",
                    event: "uiTypeaheadInputLeft",
                    on: "keydown",
                    canCauseComplete: !0,
                    enabled: !0
                },
                39: {
                    name: "RIGHT",
                    event: "uiTypeaheadInputRight",
                    on: "keydown",
                    canCauseComplete: !0,
                    enabled: !0
                },
                38: {
                    name: "UP",
                    event: "uiTypeaheadInputMoveUp",
                    on: "keydown",
                    preventDefault: !0,
                    enabled: !0
                },
                40: {
                    name: "DOWN",
                    event: "uiTypeaheadInputMoveDown",
                    on: "keydown",
                    preventDefault: !0,
                    enabled: !0
                }
            };
            return a
        }, this.setPreventKeyDefault = function(a, b) {
            this.KEY_CODE_MAP[b.key].preventDefault = b.preventDefault
        }, this.toggleTextareaActions = function(a) {
            this.KEY_CODE_MAP[13].enabled = a, this.KEY_CODE_MAP[38].enabled = a, this.KEY_CODE_MAP[40].enabled = a
        }, this.enableTextareaActionWatching = function() {
            this.toggleTextareaActions(!0)
        }, this.disableTextareaActionWatching = function() {
            this.toggleTextareaActions(!1)
        }, this.clearCurrentQuery = function(a) {
            this.currentQuery = null
        }, this.focusInput = function(a) {
            this.$input.focus()
        }, this.click = function(a) {
            this.updateCaretPosition()
        }, this.ignoreNextFocus = function(a) {
            this.ignoreFocus=!0
        }, this.updateCaretPosition = function() {
            if (this.ignoreFocus) {
                this.ignoreFocus=!1;
                return 
            }
            this.richTextareaMode || this.trigger(this.$input, "uiTextChanged", {
                text: this.$input.val(),
                position: caret.getPosition(this.$input[0])
            })
        }, this.modifierKeyPressed = function(a) {
            var b = this.KEY_CODE_MAP[a.which || a.keyCode], c = a.type == "keydown" && a.which == 16 || a.type == "keyup" && a.which == 16;
            if (b && b.enabled) {
                if (a.type !== b.on)
                    return;
                if (b.name == "TAB" && a.shiftKey)
                    return;
                if (this.releaseTabKey && b.name == "TAB")
                    return;
                if (b.preventDefault)
                    if (this.attr.inputIsTweetbox && b.name === "ENTER") {
                        var d = this.select("dropdownSelectedItemSelector");
                        d.length && a.preventDefault()
                    } else 
                        a.preventDefault();
                b.canCauseComplete && this.isValidCompletionAction(b.event) && (this.textareaMode || (this.releaseTabKey=!0), this.trigger(this.$input, "uiTypeaheadInputAutocomplete")), this.trigger(this.$input, b.event), this.updateCaretPosition()
            } else {
                if (a.keyCode == 27)
                    return;
                c || (this.releaseTabKey=!1), this.supportsInputEvent || this.handleInputChange(a)
            }
        }, this.handleInputChange = function(a) {
            this.richTextareaMode || (RTLText.onTextChange(a), this.trigger(this.$input, "uiTextChanged", {
                text: this.$input.val(),
                position: caret.getPosition(this.$input[0])
            }))
        }, this.getCurrentWord = function() {
            var a;
            if (this.textareaMode) {
                var b = twitterText.extractEntitiesWithIndices(this.text);
                b.forEach(function(b) {
                    var c = b.screenName&&!b.listSlug, d = this.attr.completeAllEntities && (b.cashtag || b.hashtag), e = this.position > b.indices[0] && this.position <= b.indices[1];
                    (c || d) && e && (a = this.text.slice(b.indices[0], b.indices[1]))
                }, this), this.attr.allowAccountsWithoutAtSign&&!a && (a = this.getCurrentWordUsingRegex(REAL_NAME_REGEXP_ONE_WORD) || this.getCurrentWordUsingRegex(REAL_NAME_REGEXP_TWO_WORD))
            } else 
                a = this.text.trim() == "" ? "" : this.text;
            return a
        }, this.getCurrentWordUsingRegex = function(a) {
            var b = stringUtils.wordAtPosition(this.text, this.position, a, 1);
            if (b) {
                var c = b.split(/\s+/), d = c.every(function(a) {
                    var b = helpers.getFirstChar(a);
                    return b == b.toUpperCase()
                });
                if (d)
                    return b
            }
        }, this.completeInput = function(a, b) {
            a.stopPropagation();
            var c = b.value || b.query, d = c !== this.currentQuery && (b.source != "account" || b.item.screen_name !== this.currentQuery);
            if (!d)
                return;
            var e = c;
            b.source == "account" && (e = (this.textareaMode ? "@" : "") + b.item.screen_name, this.currentQuery = b.item.screen_name);
            if (this.textareaMode || this.attr.triggerChangeTextAndPosition) {
                var f = this.replaceWordAtPosition(this.text, this.position, b.input, e + " ");
                this.$input.trigger("uiChangeTextAndPosition", f), (!this.richTextareaMode || a.type == "uiTypeaheadItemSelected") && this.$input.focus()
            } else 
                b.source != "account" && this.$input.val(e), a.type != "uiTypeaheadItemSelected" && (this.$input.focus(), this.setQuery(e));
            b.fromSelectionEvent = a.type == "uiTypeaheadItemSelected", this.trigger(this.$input, "uiTypeaheadItemComplete", b)
        }, this.replaceWordAtPosition = function(a, b, c, d) {
            var e = null;
            c = c.replace(UNSAFE_REGEX_CHARS, function(a) {
                return "\\" + a
            });
            var a = a.replace(new RegExp(c + "\\s?", "g"), function() {
                var a = arguments[0], c = arguments[arguments.length - 2];
                return c <= b && c + a.length >= b ? (e = c + d.length, d) : a
            });
            return {
                text: a,
                position: e
            }
        }, this.isValidCompletionAction = function(a) {
            var b = this.$input.attr("dir") === "rtl";
            return !this.textareaMode || a !== "uiTypeaheadInputRight" && a !== "uiTypeaheadInputLeft" ? b && a === "uiTypeaheadInputRight"?!1 : !b && a === "uiTypeaheadInputLeft"?!1 : !!this.text && this.position != this.text.length && (a === "uiTypeaheadInputRight" || b && a === "uiTypeaheadInputLeft")?!1 : !0 : !1
        }, this.setQuery = function(a) {
            var b;
            a = a ? RTLText.cleanText(a) : "";
            if (this.currentQuery == null || this.currentQuery !== a) {
                this.currentQuery = a, b = a.length > 0 ? 0 : - 1, this.$button.attr("tabIndex", b);
                var c = this.attr.tweetContextEnabled && this.attr.includeTweetContext ? this.text: undefined;
                this.trigger(this.$input, "uiTypeaheadInputChanged", {
                    value: this.currentQuery,
                    tweetContext: c
                })
            }
        }, this.setRTLMarkers = function() {
            RTLText.setText(this.$input.get(0))
        }, this.clearInput = function() {
            this.$input.val(""), this.clearCurrentQuery(), this.$button.attr("tabIndex", - 1), this.releaseTabKey=!1
        }, this.saveTextAndPosition = function(a, b) {
            if (b.position == Number.MAX_VALUE)
                return;
            this.text = b.text, this.position = b.position;
            var c = this.getCurrentWord();
            this.setQuery(c)
        }, this.after("initialize", function() {
            this.$input = this.select("inputSelector"), this.textareaMode=!this.$input.is("input"), this.richTextareaMode = this.$input.is(".rich-editor"), this.$button = this.select("buttonSelector"), this.KEY_CODE_MAP = this.getDefaultKeycodes(), this.richTextareaMode && this.disableTextareaActionWatching(), this.supportsInputEvent = eventSupport.browserSupports("input", "input"), this.$button.attr("tabIndex", - 1), this.on(this.$input, "keyup keydown keypress paste", this.modifierKeyPressed), this.on(this.$input, "input", this.handleInputChange), this.on("dataTypeaheadRecentSearchDeleted", this.focusInput), this.on("uiRemoveSavedSearch", this.focusInput), this.on(this.$input, "focus", this.updateCaretPosition), this.on("uiTypeaheadSelectionCleared", this.updateCaretPosition), this.on(this.$input, "uiTypeaheadIgnoreNextFocus", this.ignoreNextFocus), this.$input.is(":focus") && this.updateCaretPosition(), this.on(this.$input, "blur", this.clearCurrentQuery), this.textareaMode && (this.on(this.$input, "click", this.click), this.on("uiTypeaheadResultsShown", this.enableTextareaActionWatching), this.on("uiTypeaheadResultsHidden", this.disableTextareaActionWatching)), this.on("uiTextChanged", this.saveTextAndPosition), this.on(document, "uiBeforePageChanged", this.clearInput), this.on("uiTypeaheadSetPreventDefault", this.setPreventKeyDefault), this.on(document, "uiSwiftLoaded uiPageChanged", this.setRTLMarkers), this.on("uiTypeaheadItemPossiblyComplete uiTypeaheadItemSelected", this.completeInput)
        })
    }
    var defineComponent = require("core/component"), caret = require("app/utils/caret"), eventSupport = require("app/utils/event_support"), helpers = require("app/utils/typeahead_helpers"), stringUtils = require("app/utils/string"), twitterText = require("lib/twitter-text"), RTLText = require("app/utils/rtl_text");
    module.exports = defineComponent(typeaheadInput);
    var UNSAFE_REGEX_CHARS = /[[\]\\*?(){}.+$^]/g, REAL_NAME_REGEXP_ONE_WORD = twitterText.regexSupplant(/([a-z#{latinAccentChars}#{nonLatinHashtagChars}]{4,})/gi), REAL_NAME_REGEXP_TWO_WORD = twitterText.regexSupplant(/(?=((^|\b)[A-Z#{nonLatinHashtagChars}][A-Za-z#{latinAccentChars}#{nonLatinHashtagChars}]*\s[A-Z#{nonLatinHashtagChars}][A-Za-z#{latinAccentChars}#{nonLatinHashtagChars}]*))/g)
});
define("app/ui/select_users", ["module", "require", "exports", "app/utils/caret", "core/component", "app/ui/typeahead/typeahead_dropdown", "app/ui/typeahead/typeahead_input"], function(module, require, exports) {
    function selectUsers() {
        this.defaultAttrs({
            selectUsersContainerSelector: ".user-select-container",
            typeaheadContainerSelector: ".typeahead-container",
            dropdownSelector: ".dropdown-menu.typeahead",
            inputSelector: ".user-select-controls input",
            itemSelector: ".typeahead-item",
            maxSelection: 10,
            requestType: "",
            alwaysOpen: !1,
            includePreviouslySelectedUsers: !1,
            controlsSelector: ".user-select-controls",
            userBorderColorClass: "u-borderUserColorLight",
            datasourceForTagging: "accounts",
            typeaheadSrc: "",
            decidersForTypeahead: {},
            cannotSelectClass: "cannot-select",
            eventData: {
                scribeContext: {
                    element: "tweet_box"
                }
            }
        }), this.selectUser = function(a, b) {
            var c = this.shouldUpdateUserList(a, b), d = this.selectedUserIndex(b.item.id);
            d==-1&&!c && a.stopImmediatePropagation(), d==-1 && c ? this.selectedUsers.push(b.item) : d!=-1 && (this.selectedUsers.splice(d, 1), c=!0), c ? (this.trigger("uiUpdatedSelectedUsers", {
                items: this.selectedUsers
            }), this.$input.val(""), this.trigger(this.$input, "uiTextChanged", {
                text: this.$input.val(),
                position: caret.getPosition(this.$input[0])
            })) : this.trigger("uiCouldNotSelectUser"), this.$input.focus()
        }, this.shouldUpdateUserList = function(a, b) {
            return !$(a.target).find("[data-user-id=" + b.item.id + "]").hasClass(this.attr.cannotSelectClass) && this.selectedUsers.length < this.attr.maxSelection
        }, this.selectedUserIndex = function(a) {
            for (var b = 0; b < this.selectedUsers.length; b++)
                if (this.selectedUsers[b].id == a)
                    return b;
            return - 1
        }, this.resetSelectedUsers = function() {
            this.selectedUsers = []
        }, this.resetComposer = function() {
            this.resetSelectedUsers(), this.$input.val(""), this.trigger("uiUpdatedSelectedUsers", {
                items: this.selectedUsers
            }), this.trigger(this.$input, "uiTextChanged", {
                text: this.$input.val(),
                position: caret.getPosition(this.$input[0])
            })
        }, this.focusInputWithResultsShown = function(a, b) {
            this.$input.focus(), this.forceShow(), this.trigger("uiTypeaheadResultsShown")
        }, this.focusInput = function() {
            this.select("controlsSelector").addClass(this.attr.userBorderColorClass)
        }, this.blurInput = function() {
            this.select("controlsSelector").removeClass(this.attr.userBorderColorClass)
        }, this.forceShow = function() {
            this.select("dropdownSelector").show()
        }, this.forceHide = function() {
            this.select("dropdownSelector").hide(), this.trigger("uiTypeaheadSetPreventDefault", {
                preventDefault: !1,
                key: 9
            }), this.trigger("uiTypeaheadResultsHidden")
        }, this.hideAndManageEsc = function(a) {
            a.preventDefault(), a.stopPropagation(), this.$input.blur(), this.forceHide()
        }, this.possiblyHide = function(a) {
            a.target != this.$node.get(0)&&!$.contains(this.$node.get(0), a.target) && this.forceHide()
        }, this.after("initialize", function() {
            TypeaheadInput.attachTo(this.$node, {
                inputSelector: this.attr.inputSelector,
                eventData: this.attr.eventData
            });
            var a = [this.attr.datasourceForTagging, "selectedUsers"];
            this.attr.includePreviouslySelectedUsers && a.push("prefillUsers"), TypeaheadDropdown.attachTo(this.$node, {
                alwaysOpen: this.attr.alwaysOpen,
                blockLinkActions: !0,
                typeaheadSrc: this.attr.typeaheadSrc,
                datasourceRenders: [["selectedUsers", a]],
                datasourceOptions: {
                    requestType: this.attr.requestType
                },
                deciders: this.attr.decidersForTypeahead,
                inputSelector: this.attr.inputSelector,
                eventData: this.attr.eventData
            }), this.on("uiTypeaheadRenderResults uiShowSelectedUsers", this.forceShow), this.on("uiTypeaheadItemSelected uiSelectUser", this.selectUser), this.on("uiResetSelectedUsers", this.resetSelectedUsers), this.on("uiResetSelectedUserComposer", this.resetComposer), this.on("uiShortcutEsc", this.hideAndManageEsc), this.selectedUsers = [], this.$input = this.select("inputSelector"), this.on("uiSetFocusInSelectedUserInput", this.focusInputWithResultsShown), this.on(this.$input, "focusin", this.focusInput), this.on(this.$input, "focusout", this.blurInput), this.on(document, "mousedown : focusin", this.possiblyHide)
        })
    }
    var caret = require("app/utils/caret"), defineComponent = require("core/component"), TypeaheadDropdown = require("app/ui/typeahead/typeahead_dropdown"), TypeaheadInput = require("app/ui/typeahead/typeahead_input"), SelectUsers = defineComponent(selectUsers);
    module.exports = SelectUsers
});
define("app/ui/select_users_with_popover", ["module", "require", "exports", "core/component", "app/ui/select_users"], function(module, require, exports) {
    function selectUsersWithPopover() {
        this.defaultAttrs({
            dropdownSelector: ".user-select-container.dropdown-menu",
            openPopoverButtonSelector: ".js-open-user-select",
            includePreviouslySelectedUsers: !0,
            typeaheadSrc: "COMPOSE_MEDIA_TAGGING",
            datasourceForTagging: "mediaTagAccounts",
            requestType: "compose_media_tagging"
        }), this.showSelectPopover = function(a) {
            this.previouslyFocused = document.activeElement, a.stopPropagation(), a.preventDefault(), this.$node.show(), this.trigger("uiSetFocusInSelectedUserInput")
        }, this.hideSelectPopover = function(a) {
            if (a.target != this.$node.get(0)&&!$.contains(this.$node.get(0), a.target) && $(a.target).closest(this.attr.openPopoverButtonSelector).length == 0) {
                if (this.previouslyFocused && a.type == "focusin") {
                    var b = this.previouslyFocused;
                    this.previouslyFocused = null, b.focus()
                } else 
                    this.previouslyFocused = null;
                this.forceHide()
            }
        }, this.forceHide = function(a) {
            this.$node.hide(), this.trigger("uiTypeaheadSetPreventDefault", {
                preventDefault: !1,
                key: 9
            }), this.trigger("uiTypeaheadResultsHidden")
        }, this.hideAndManageEsc = function(a) {
            a.preventDefault(), a.stopPropagation(), this.forceHide()
        }, this.after("initialize", function() {
            SelectUsers.attachTo(this.$node, {
                alwaysOpen: !0,
                datasourceForTagging: this.attr.datasourceForTagging,
                includePreviouslySelectedUsers: this.attr.includePreviouslySelectedUsers,
                requestType: this.attr.requestType,
                typeaheadSrc: this.attr.typeaheadSrc
            }), this.on("uiShowSelectPopover", this.showSelectPopover), this.on("uiShortcutEsc", this.hideAndManageEsc), this.on(document, "mousedown : focusin", this.hideSelectPopover)
        })
    }
    var defineComponent = require("core/component"), SelectUsers = require("app/ui/select_users"), SelectUsersWithPopover = defineComponent(selectUsersWithPopover);
    module.exports = SelectUsersWithPopover
});
define("app/data/media_tags_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function mediaTagsScribe() {
        this.scribeClickOnButton = function(a, b) {
            this.scribe({
                page: "composition",
                element: "media_tag_prompt",
                action: "click"
            }, b)
        }, this.scribeUserSelected = function(a, b) {
            this.scribe({
                page: "composition",
                component: "media_tag",
                element: "taggable_user",
                action: "click"
            }, b)
        }, this.scribeUserNotSelected = function(a, b) {
            this.scribe({
                page: "composition",
                component: "media_tag",
                element: "nontaggable_user",
                action: "click"
            }, b)
        }, this.scribeTweet = function(a, b) {
            b.tweetData.tagged_users && b.tweetData.tagged_users.length && b.tweetData.in_reply_to_status_id && b.tweetData.in_reply_to_status_id.length ? this.scribe({
                page: "composition",
                action: "reply_with_tags"
            }, b) : b.tweetData.tagged_users && b.tweetData.tagged_users.length && this.scribe({
                page: "composition",
                action: "tweet_with_tags"
            }, b)
        }, this.after("initialize", function() {
            this.on("uiShowSelectPopover", this.scribeClickOnButton), this.on("uiUpdatedSelectedUsers", this.scribeUserSelected), this.on("uiSendTweetWithMedia", this.scribeTweet), this.on("uiCouldNotSelectUser", this.scribeUserNotSelected)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(mediaTagsScribe, withScribe)
});
define("app/ui/with_thumbnail_preview", ["module", "require", "exports", "app/ui/select_users_with_popover", "app/data/media_tags_scribe", "core/i18n"], function(module, require, exports) {
    function withThumbnailPreview() {
        this.defaultAttrs({
            thumbContainerSelector: ".thumbnail-container",
            taggingSelector: ".photo-tagging-container",
            taggingButtonSelector: ".js-open-user-select",
            taggedUsersSelector: ".tagged-users",
            thumbSelector: ".previews .preview",
            multiPhoto: !1
        }), this.hasMedia=!1, this.taggedUsers = [], this.defaultTagText = _('Who\'s in this photo?'), this.showPreview = function(a, b) {
            this.$node.addClass("has-preview has-thumbnail"), this.hasMedia=!0, this.detectUpdatedText(!0), b.isGif ? this.select("taggingButtonSelector").first().hide() : (this.select("taggingButtonSelector").first().show(), this.select("taggingButtonSelector").first().prop("disabled", !1))
        }, this.updateThumbnailCount = function(a) {
            var b = this.select("thumbSelector").length;
            b > 1 ? this.defaultTagText = _('Who\'s in these photos?') : this.defaultTagText = _('Who\'s in this photo?'), this.taggedUsers.length == 0 && this.select("taggedUsersSelector").html(this.defaultTagText)
        }, this.hidePreview = function(a, b) {
            this.$node.removeClass("has-preview has-thumbnail"), this.hasMedia=!1, this.detectUpdatedText(!0), this.select("taggingButtonSelector").first().prop("disabled", !0)
        }, this.updateTaggedUsers = function(a, b) {
            var c = this.select("taggedUsersSelector");
            if (b.items.length == 0) {
                c.text(this.defaultTagText), this.select("taggingButtonSelector").addClass("no-users");
                return 
            }
            this.select("taggingButtonSelector").removeClass("no-users"), this.taggedUsers = b.items, c.html(this.getTaggedUsersString(b.items))
        }, this.getTaggedUsersString = function(a) {
            if (a.length == 1)
                return _('{{name1}}', {
                    name1: a[0].name
                });
            if (a.length == 2)
                return _('<span class=\'user-name\'>{{name1}}</span> and 1 other', {
                    name1: a[0].name
                });
            if (a.length > 2)
                return _('<span class=\'user-name\'>{{name1}}</span> and {{count}} others', {
                    name1: a[0].name,
                    count: a.length - 1
                })
        }, this.after("reset", function() {
            this.trigger("uiPreviewThumbnailHide"), this.trigger("uiPreviewThumbnailRemoveAll"), this.trigger(this.attr.taggingSelector, "uiResetSelectedUserComposer")
        }), this.around("createDataForSending", function(a) {
            var b = a();
            return b.tagged_users = this.taggedUsers.map(function(a) {
                return a.id
            }).join(","), this.trigger("uiUpdateRecentTags", {
                items: this.taggedUsers
            }), b
        }), this.around("hasAttachment", function(a) {
            return this.hasMedia || a()
        }), this.after("initialize", function() {
            this.on("uiPreviewThumbnailShow", this.showPreview), this.on("uiPreviewThumbnailHide", this.hidePreview), this.on("uiPreviewThumbnailRemove", this.focus), taggingComposer.attachTo(this.select("taggingSelector"), {
                includePreviouslySelectedUsers: !0
            }), taggingScribe.attachTo(this.select("taggingSelector"), {}), this.on(this.select("taggingButtonSelector"), "click", function() {
                this.trigger(this.select("taggingSelector"), "uiShowSelectPopover")
            }), this.on("uiUpdatedSelectedUsers", this.updateTaggedUsers), this.attr.multiPhoto && this.select("thumbContainerSelector").addClass("multi-photo"), this.on("uiUpdatedThumbnails", this.updateThumbnailCount)
        })
    }
    var taggingComposer = require("app/ui/select_users_with_popover"), taggingScribe = require("app/data/media_tags_scribe"), _ = require("core/i18n");
    module.exports = withThumbnailPreview
});
define("app/ui/toolbar", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function toolbar() {
        this.defaultAttrs({
            buttonsSelector: ".btn:not([disabled])"
        }), this.current =- 1, this.focusNext = function(a) {
            var b = this.select("buttonsSelector");
            this.current==-1 && (this.current = $.inArray(document.activeElement, b));
            var c, d = this.current;
            switch (a.which) {
            case 37:
                d--;
                break;
            case 39:
                d++
            }
            c = b[d], c && (c.focus(), this.current = d)
        }, this.clearCurrent = function() {
            this.current =- 1
        }, this.after("initialize", function() {
            this.$node.attr("role", "toolbar"), this.on("keydown", {
                buttonsSelector: this.focusNext
            }), this.on("focusout", this.clearCurrent)
        })
    }
    var defineComponent = require("core/component"), Toolbar = defineComponent(toolbar);
    module.exports = Toolbar
});
define("app/ui/with_toolbar", ["module", "require", "exports", "app/ui/toolbar"], function(module, require, exports) {
    function withToolbar() {
        this.defaultAttrs({
            toolbarSelector: ".js-toolbar",
            tooltipSelector: ".js-tooltip",
            geoStatusSelector: ".geo-status",
            tooltipClass: "js-tooltip",
            visuallyHiddenClass: "u-hiddenVisually"
        }), this.prepareToolbarButtons = function() {
            var a = this.select("toolbarSelector");
            a.outerWidth() >= 490 ? a.find(this.attr.tooltipSelector).removeClass(this.attr.tooltipClass).removeAttr("title") : this.select("geoStatusSelector").addClass(this.attr.visuallyHiddenClass)
        }, this.after("initialize", function() {
            var a = this.select("toolbarSelector");
            Toolbar.attachTo(a, {
                buttonsSelector: ".file-input, .lifeline-alert-status-btn, .geo-picker-btn, .tweet-action"
            }), a.is(":visible") ? this.prepareToolbarButtons() : this.on("uiTweetBoxExpanded", this.prepareToolbarButtons)
        })
    }
    var Toolbar = require("app/ui/toolbar");
    module.exports = withToolbar
});
define("app/ui/tweet_box", ["module", "require", "exports", "core/i18n", "core/component", "app/utils/scribe_item_types", "core/utils", "app/ui/with_condensing", "app/ui/with_allow_teardown", "app/ui/with_character_counter", "app/ui/with_draft_tweets", "app/ui/with_droppable_image", "app/ui/with_image_upload_button", "app/ui/with_lifeline", "app/ui/with_rtl_tweet_box", "app/ui/with_text_editor", "app/ui/with_text_polling", "app/ui/with_thumbnail_preview", "app/ui/with_toolbar"], function(module, require, exports) {
    function tweetBox() {
        var a = "";
        this.defaultAttrs({
            shadowTextSelector: ".tweet-box-shadow",
            messagingClass: "messaging",
            tweetActionSelector: ".tweet-action",
            placeIdSelector: "input[name=place_id]",
            cursorPosition: undefined,
            inReplyToTweetData: {},
            inReplyToStatusId: undefined,
            impressionId: undefined,
            disclosureType: undefined,
            modal: !1,
            condensable: !1,
            suppressFlashMessage: !1,
            suppressSuccessMessage: !1,
            customData: {},
            position: undefined,
            itemType: "tweet",
            component: undefined
        }), this.dmRegex = /^\s*(?:d|m|dm)\s+[@＠]?(\S+)\s*(.*)/i, this.validUserRegex = /^(\w{1,20})$/, this.dmMode=!1, this.defaultTagText = _('Who\'s in this photo?'), this.confirmAndSendTweet = function() {
            this.sendTweet()
        }, this.geoPlaceId = function() {
            return this.select("placeIdSelector").val()
        }, this.createDataForSending = function() {
            var a = {
                status: this.addHiddenText(this.getVisibleText()),
                place_id: this.geoPlaceId(),
                in_reply_to_status_id: this.attr.inReplyToStatusId,
                impression_id: this.attr.impressionId,
                earned: this.attr.disclosureType ? this.attr.disclosureType == "earned": undefined
            };
            return this.attr.inReplyToTweetData && (a.tweet_stat_count = this.attr.inReplyToTweetData.tweetStatCount), a
        }, this.isEmpty = function() {
            var a = this.getVisibleText().trim();
            return !(a && (this.attr.canTweetDefaultText || a !== this.attr.defaultText.trim()) || this.hasAttachment())
        }, this.hasAttachment = function() {
            return !1
        }, this.sendTweet = function() {
            this.detectUpdatedText(), this.$node.attr("id", this.getTweetboxId());
            var a = this.hasMedia ? "uiSendTweetWithMedia": "uiSendTweet";
            this.trigger(a, {
                tweetboxId: this.getTweetboxId(),
                tweetData: this.createDataForSending(),
                customData: this.attr.customData
            }), this.$node.addClass("tweeting"), this.disable()
        }, this.getTweetboxId = function() {
            return this.tweetboxId || (this.tweetboxId = "swift_tweetbox_" + + (new Date)), this.tweetboxId
        }, this.overrideTweetBoxOptions = function(a, b) {
            this.attr.inReplyToTweetData = b, b.id && (this.attr.inReplyToStatusId = b.id), b.impressionId && (this.attr.impressionId = b.impressionId), b.disclosureType && (this.attr.disclosureType = b.disclosureType), b.defaultText && (this.attr.defaultText = b.defaultText), b.customData && (this.attr.customData = b.customData), b.itemType && (this.attr.itemType = b.itemType), b.scribeContext && b.scribeContext.component && (this.attr.component = b.scribeContext.component), b.position !== undefined && (this.attr.position = b.position), b.cursorPosition !== undefined && (this.attr.cursorPosition = b.cursorPosition)
        }, this.resetOverriddenOptions = function(a, b) {
            delete this.attr.defaultText, this.attr.inReplyToTweetData = this.defaults.inReplyToTweetData, this.attr.inReplyToStatusId = this.defaults.inReplyToStatusId, this.attr.impressionId = this.defaults.impressionId, this.attr.disclosureType = this.defaults.disclosureType, this.attr.defaultText = this.getDefaultText(), this.attr.cursorPosition = this.defaults.cursorPosition, this.attr.customData = this.defaults.customData, this.attr.position = this.defaults.position, this.attr.itemType = this.defaults.itemType, this.attr.component = this.attr.component
        }, this.updateTweetStateOnTextChanged = function(a, b) {
            this.select("shadowTextSelector").val(b.text), this.updateTitle(), this.updateTweetButton()
        }, this.updateTweetButton = function() {
            this.isEmpty() || this.maxReached() || this.$node.hasClass("tweeting") || this.dmMode && (!this.dmText ||!this.dmUsername.match(this.validUserRegex)) ? this.disable() : this.enable(), this.select("tweetActionSelector").toggleClass(this.attr.messagingClass, this.dmMode)
        }, this.updateTitle = function() {
            var a = this.getVisibleText().match(this.dmRegex), b = a && a[1];
            this.dmText = a && a[2], a && (!this.dmMode || this.dmMode && this.dmUsername != b) ? (this.dmMode=!0, this.dmUsername = b, this.trigger("uiDialogUpdateTitle", {
                title: _('Message @{{username}}', {
                    username: b
                })
            })) : this.dmMode&&!a && (this.dmMode=!1, this.dmUsername = undefined, this.trigger("uiDialogUpdateTitle", {
                title: _('Compose new Tweet')
            }))
        }, this.tweetSent = function(a, b) {
            var c = b.tweetboxId || b.sourceEventData.tweetboxId;
            if (c != this.getTweetboxId())
                return;
            b.customData = this.attr.customData, b.message && (!this.attr.suppressSuccessMessage || b.unusual) && this.trigger(b.unusual ? "uiShowError" : "uiShowMessage", {
                message: b.message
            });
            var d = "uiTweetboxTweetSuccess", e = {
                id: b.tweet_id,
                item_type: scribeItemTypes.tweet
            };
            if (this.attr.inReplyToStatusId || this.getVisibleText().indexOf("@") == 0) {
                if ((this.attr.inReplyToTweetData || {}).replyLinkClick) {
                    var f = utils.merge({}, this.attr.inReplyToTweetData);
                    f.scribeContext && (f.scribeContext.element = ""), this.trigger("uiReplyButtonTweetSuccess", f)
                }
                d = "uiTweetboxReplySuccess"
            } else 
                this.getVisibleText().match(this.dmRegex) && (d = "uiTweetboxDMSuccess");
            this.trigger(d, {
                scribeData: {
                    items: [e]
                }
            }), this.hasMedia && this.trigger("uiTweetboxPhotoTweetSuccess", {
                scribeData: {
                    items: [e]
                }
            }), this.geoPlaceId() && this.trigger("uiTweetboxGeotaggedTweetSuccess", {
                scribeData: {
                    items: [e]
                }
            }), this.$node.removeClass("tweeting"), this.trigger("uiTweetSent", b), this.clear()
        }, this.scribeDataForReply = function() {
            var a = {
                id: this.attr.inReplyToStatusId,
                item_type: scribeItemTypes.tweet
            }, b = {};
            this.attr.impressionId && (a.token = this.attr.impressionId, b.promoted=!0);
            if (this.attr.position == 0 || this.attr.position)
                a.position = this.attr.position;
            return b.items = [a], {
                scribeData: b,
                scribeContext: {
                    component: this.attr.component,
                    element: ""
                }
            }
        }, this.tweetError = function(a, b) {
            var c = b.tweetboxId || b.sourceEventData.tweetboxId;
            if (c != this.getTweetboxId())
                return;
            !this.attr.suppressFlashMessage&&!b.spamChallenge && this.trigger("uiShowError", {
                message: b.error || b.message
            }), this.$node.removeClass("tweeting"), this.enable(), this.trigger("uiTweetboxTweetError")
        }, this.enable = function() {
            this.select("tweetActionSelector").removeClass("disabled").attr("disabled", !1), this.$node.trigger("uiEnableTweetAction")
        }, this.disable = function() {
            this.select("tweetActionSelector").addClass("disabled").attr("disabled", !0), this.$node.trigger("uiDisableTweetAction")
        }, this.clear = function() {
            this.reset()
        }, this.reset = function() {
            this.resetTweetText(), this.$node.find("input[type=hidden]").val("")
        }, this.onTweetBoxExpand = function(a, b) {
            b && this.overrideTweetBoxOptions(a, b), this.trigger("uiComposerResetAndFocus")
        }, this.resetAndFocus = function() {
            this.reset(), this.focus(), this.setCursorPosition(), this.composerActive()
        }, this.composerActive = function() {
            this.trigger("uiComposerActive")
        }, this.resetTweetText = function() {
            this.trigger("uiClearUndoState"), this.setVisibleText(this.attr.defaultText)
        }, this.getDefaultText = function() {
            return typeof this.attr.defaultText != "undefined" ? this.attr.defaultText : this.getAttrOrElse("data-default-text", a)
        }, this.getAttrOrElse = function(a, b) {
            return typeof this.$node.attr(a) == "undefined" ? b : this.$node.attr(a)
        }, this.handleCmdEnterSubmit = function() {
            this.select("tweetActionSelector").hasClass("disabled") || this.confirmAndSendTweet()
        }, this.after("initialize", function() {
            this.attr.defaultText = this.getDefaultText(), utils.push(this.attr, {
                eventData: {
                    scribeContext: {
                        element: "tweet_box"
                    }
                }
            }, !1), this.on("uiTextChanged", this.updateTweetStateOnTextChanged), this.initDraftTweets(), this.initTextNode(), this.on(this.select("tweetActionSelector"), "click", this.confirmAndSendTweet), this.on("uiShortcutCmdEnter", this.handleCmdEnterSubmit), this.on(document, "dataTweetSuccess", this.tweetSent), this.on(document, "dataTweetError", this.tweetError), this.on(this.$text, "dragover", this.focus), this.on("uiComposerResetAndFocus", this.resetAndFocus), this.on("uiExpandFocus", this.focus), this.on("uiTweetBoxExpand", this.onTweetBoxExpand), this.attr.modal && (this.on(document, "uiOverrideTweetBoxOptions", this.overrideTweetBoxOptions), this.on(document, "uiDialogClosed", this.resetOverriddenOptions)), this.on(this.$text, "focusin", this.ignoreDuringFakeFocus(this.composerActive)), this.hasFocus() ? this.composerActive() : this.reset()
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component"), scribeItemTypes = require("app/utils/scribe_item_types"), utils = require("core/utils"), withCondensing = require("app/ui/with_condensing"), withAllowTeardown = require("app/ui/with_allow_teardown"), withCounter = require("app/ui/with_character_counter"), withDraftTweets = require("app/ui/with_draft_tweets"), withDroppableImage = require("app/ui/with_droppable_image"), withImageUploadButton = require("app/ui/with_image_upload_button"), withLifeline = require("app/ui/with_lifeline"), withRTL = require("app/ui/with_rtl_tweet_box"), withTextEditor = require("app/ui/with_text_editor"), withTextPolling = require("app/ui/with_text_polling"), withThumbnailPreview = require("app/ui/with_thumbnail_preview"), withToolbar = require("app/ui/with_toolbar"), TweetBox = defineComponent(tweetBox, withCondensing, withTextEditor, withCounter, withTextPolling, withRTL, withDraftTweets, withImageUploadButton, withThumbnailPreview, withDroppableImage, withLifeline, withAllowTeardown, withToolbar);
    module.exports = TweetBox
});
define("app/ui/with_tweet_attachment", ["module", "require", "exports", "core/i18n"], function(module, require, exports) {
    function tweetAttachment() {
        this.defaultAttrs({
            externalContainerSelector: "#dm_dialog_new .twttr-dialog-body, #dm_dialog_conversation .twttr-dialog-body",
            classForTweetAttachmentPresence: "has-tweet-attachment"
        }), this.around("createDataForSending", function(a) {
            var b = a();
            return this.hasTweetAttachment && (b.tweet_id = this.tweetAttachmentId), b
        }), this.around("hasAttachment", function(a) {
            return this.hasTweetAttachment || a()
        }), this.possiblyInsertTweetPreview = function(a, b) {
            var c = b && b.id;
            c && this.addTweetAttachment(c)
        }, this.modifyComposerForTweetAttachment = function() {
            this.trigger("uiAttachmentChanged"), this.trigger("uiDisableImageSelection"), this.trigger("uiUpdateComposerPlaceholder", {
                placeholderText: _('Add a comment...')
            }), $(this.attr.externalContainerSelector).addClass(this.attr.classForTweetAttachmentPresence)
        }, this.revertComposer = function() {
            this.trigger("uiAttachmentChanged"), this.trigger("uiEnableImageSelection"), this.trigger("uiUpdateComposerPlaceholder"), $(this.attr.externalContainerSelector).removeClass(this.attr.classForTweetAttachmentPresence)
        }, this.addTweetAttachment = function(a) {
            this.hasTweetAttachment=!0, this.tweetAttachmentId = a, this.modifyComposerForTweetAttachment(), this.trigger("uiWantsTweet", {
                tweetId: this.tweetAttachmentId,
                tweetOptions: {
                    modal: "dm",
                    disableLinks: !0
                }
            })
        }, this.removeTweetAttachment = function() {
            this.hasTweetAttachment=!1, this.tweetAttachmentId = null, this.revertComposer()
        }, this.after("initialize", function() {
            this.tweetAttachmentId = null, this.on(document, "uiComposeWithTweet", this.possiblyInsertTweetPreview), this.on(document, "uiDMDialogOpenedConversationList uiCloseDMDialog dataDMSendSuccess dataFailedToGetTweet", this.removeTweetAttachment)
        })
    }
    var _ = require("core/i18n");
    module.exports = tweetAttachment
});
define("app/ui/dm_composer", ["module", "require", "exports", "core/i18n", "core/component", "core/utils", "app/ui/with_allow_teardown", "app/ui/with_character_counter", "app/ui/with_droppable_image", "app/ui/with_image_upload_button", "app/ui/dialogs/with_modal_tweet", "app/ui/with_rtl_tweet_box", "app/ui/with_text_editor", "app/ui/with_text_polling", "app/ui/with_thumbnail_preview", "app/ui/with_toolbar", "app/ui/with_tweet_attachment"], function(module, require, exports) {
    function dmComposer() {
        this.defaultAttrs({
            tweetBoxSelector: ".tweet-box",
            tweetActionSelector: ".tweet-action"
        }), this.isEmpty = function() {
            return !this.hasAttachment()&&!this.getVisibleText().trim()
        }, this.hasAttachment = function() {
            return !1
        }, this.createDataForSending = function() {
            return {
                status: this.addHiddenText(this.getVisibleText())
            }
        }, this.sendDM = function() {
            this.detectUpdatedText(), this.$node.attr("id", this.getTweetboxId());
            var a = this.hasMedia ? "uiSendDMWithMedia": "uiSendDM";
            this.trigger(a, {
                tweetboxId: this.getTweetboxId(),
                tweetData: this.createDataForSending()
            }), this.$node.addClass("tweeting"), this.disable()
        }, this.getTweetboxId = function() {
            return this.tweetboxId || (this.tweetboxId = "swift_tweetbox_" + + (new Date)), this.tweetboxId
        }, this.updateSendButton = function() {
            var a=!this.isEmpty()&&!this.maxReached()&&!this.$node.hasClass("tweeting");
            this.hasValidRecipient && a ? this.enable() : this.disable()
        }, this.updateDMRecipientState = function(a, b) {
            this.hasValidRecipient = b.hasValidRecipient, this.updateSendButton()
        }, this.determinePlaceholderVisibility = function() {
            this.showPlaceholder() && this.select("tweetBoxSelector").toggleClass("is-showPlaceholder", !this.getVisibleText().trim())
        }, this.updatePlaceholderText = function(a, b) {
            var c = b && b.placeholderText, d = this.select("tweetBoxSelector");
            c ? d.attr("data-placeholder", c) : d.removeAttr("data-placeholder"), this.determinePlaceholderVisibility()
        }, this.dmSent = function(a, b) {
            var c = b.tweetboxId || b.sourceEventData.tweetboxId;
            if (c != this.getTweetboxId())
                return;
            b.customData = this.attr.customData, b.message && this.trigger("uiShowMessage", {
                message: b.message
            }), this.$node.removeClass("tweeting"), this.trigger("uiDMSent", b), this.reset()
        }, this.dmError = function(a, b) {
            var c = b.tweetboxId || b.sourceEventData.tweetboxId;
            if (c != this.getTweetboxId())
                return;
            this.$node.removeClass("tweeting"), this.enable()
        }, this.showPlaceholder = function() {
            return this.hasTweetAttachment
        }, this.enable = function() {
            this.select("tweetActionSelector").removeClass("disabled").attr("disabled", !1)
        }, this.disable = function() {
            this.select("tweetActionSelector").addClass("disabled").attr("disabled", !0)
        }, this.resetAndFocus = function() {
            this.reset(), this.focus()
        }, this.reset = function() {
            this.setVisibleText("")
        }, this.handleCmdEnterSubmit = function() {
            this.select("tweetActionSelector").hasClass("disabled") || this.sendDM()
        }, this.after("initialize", function() {
            this.initTextNode(), this.on(this.select("tweetActionSelector"), "click", this.sendDM), this.on("uiShortcutCmdEnter", this.handleCmdEnterSubmit), this.on(document, "dataDMSuccess", this.dmSent), this.on(document, "dataDMError", this.dmError), this.on("uiTextChanged uiAttachmentChanged", this.updateSendButton), this.on("uiTextChanged", this.determinePlaceholderVisibility), this.on("uiUpdateComposerPlaceholder", this.updatePlaceholderText), this.on("uiComposerResetAndFocus", this.resetAndFocus), this.on(document, "uiDMRecipientChanged", this.updateDMRecipientState)
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component"), utils = require("core/utils"), withAllowTeardown = require("app/ui/with_allow_teardown"), withCounter = require("app/ui/with_character_counter"), withDroppableImage = require("app/ui/with_droppable_image"), withImageUploadButton = require("app/ui/with_image_upload_button"), withModalTweet = require("app/ui/dialogs/with_modal_tweet"), withRTL = require("app/ui/with_rtl_tweet_box"), withTextEditor = require("app/ui/with_text_editor"), withTextPolling = require("app/ui/with_text_polling"), withThumbnailPreview = require("app/ui/with_thumbnail_preview"), withToolbar = require("app/ui/with_toolbar"), withTweetAttachment = require("app/ui/with_tweet_attachment"), DmComposer = defineComponent(dmComposer, withTextEditor, withCounter, withTextPolling, withRTL, withImageUploadButton, withThumbnailPreview, withDroppableImage, withModalTweet, withTweetAttachment, withAllowTeardown, withToolbar);
    module.exports = DmComposer
});
define("app/utils/image_thumbnail", ["module", "require", "exports", "app/utils/image"], function(module, require, exports) {
    var image = require("app/utils/image"), imageThumbnail = {
        createThumbnail: function(a, b, c, d) {
            var e = new Image;
            e.onload = function() {
                c(a, e, e.height, e.width, d)
            }, e.src = image.getDataUri(b)
        },
        getThumbnailOffset: function(a, b, c) {
            var d;
            if (b == a && b >= c && a >= c)
                return {
                    position: "absolute",
                    height: c,
                    width: c,
                    left: 0,
                    top: 0
                };
            if (a < c || b < c)
                d = {
                    position: "absolute",
                    height: a,
                    width: b,
                    top: (c - a) / 2,
                    left: (c - b) / 2
                };
            else if (b > a) {
                var e = c / a * b;
                d = {
                    position: "absolute",
                    height: c,
                    width: e,
                    left: - (e - c) / 2,
                    top: 0
                }
            } else if (a > b) {
                var f = c / b * a;
                d = {
                    position: "absolute",
                    height: f,
                    width: c,
                    top: - (f - c) / 2,
                    left: 0
                }
            }
            return d
        }
    };
    module.exports = imageThumbnail
});
define("app/ui/tweet_box_thumbnails", ["module", "require", "exports", "core/component", "core/utils", "app/utils/image_thumbnail"], function(module, require, exports) {
    function tweetBoxThumbnails() {
        this.defaults = {
            thumbSelector: ".previews .preview",
            thumbTemplateSelector: ".preview.template",
            thumbsSelector: ".previews",
            thumbImageSelector: ".preview img",
            filenameSelector: ".preview .filename",
            dismissSelector: ".dismiss",
            tweetBoxSelector: ".tweet-form",
            photoLimit: 1,
            photoCount: 0
        }, this.addThumbnail = function(a, b) {
            b.imageData && imageThumbnail.createThumbnail(b.fileName, b.imageData, this.gotThumbnail.bind(this), b.uploadId), this.select("filenameSelector").text(b.fileName)
        }, this.removeAllThumbnails = function(a, b) {
            this.select("thumbsSelector").empty(), this.select("filenameSelector").empty()
        }, this.gotThumbnail = function(a, b, c, d, e) {
            var f = imageThumbnail.getThumbnailOffset(c, d, 48);
            $(b).css(f);
            var g = $("<a>", {
                href: $(b).attr("src"),
                target: "_blank"
            }).append($(b)), h = this.select("thumbTemplateSelector").clone();
            h.removeClass("template"), h.append(g), e && h.attr("data-upload-id", e), this.on(h.find(".dismiss"), "click", this.removeImage), this.select("thumbsSelector").append(h), this.trigger("uiUpdatedThumbnails")
        }, this.removeImage = function(a) {
            var b = $(a.target).closest(this.attr.thumbSelector), c = b.attr("data-upload-id");
            this.trigger("uiPreviewThumbnailRemove", {
                uploadId: c
            }), b.remove(), this.trigger("uiImagePickerRemove"), this.trigger("uiUpdatedThumbnails")
        }, this.after("initialize", function() {
            utils.push(this.attr, {
                eventData: {
                    scribeContext: {
                        element: "image_picker"
                    }
                }
            }, !1);
            var a = this.$node.closest(this.attr.tweetBoxSelector);
            this.on(a, "uiPreviewThumbnailAdd", this.addThumbnail), this.on(a, "uiPreviewThumbnailRemoveAll", this.removeAllThumbnails)
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), imageThumbnail = require("app/utils/image_thumbnail"), TweetBoxThumbnails = defineComponent(tweetBoxThumbnails);
    module.exports = TweetBoxThumbnails
});
define("app/utils/image_resize", ["module", "require", "exports", "app/utils/image"], function(module, require, exports) {
    var image = require("app/utils/image"), imageResize = {
        resize: function(a, b, c, d) {
            if (!image.hasCanvas())
                return d(a, b.split(";base64,")[1]);
            var e = new Image, f = document.createElement("canvas"), g = f.getContext("2d");
            e.onload = function() {
                if (e.width == 0 || e.height == 0) {
                    d(a, !1);
                    return 
                }
                if (e.width < c && e.height < c) {
                    d(a, b.split(";base64,")[1]);
                    return 
                }
                var h, i;
                e.width > e.height ? (h = c, i = e.height / e.width * c) : (i = c, h = e.width / e.height * c), f.width = h, f.height = i, g.drawImage(e, 0, 0, h, i);
                var j = f.toDataURL("image/jpeg");
                d(a, j.split("data:image/jpeg;base64,")[1])
            }, e.onerror = function() {
                d(a, !1)
            }, e.src = b
        }
    };
    module.exports = imageResize
});
define("app/data/with_iframe_async_image_upload", ["module", "require", "exports", "core/i18n", "core/compose", "template", "app/data/with_auth_token"], function(module, require, exports) {
    function withIframeAsyncImageUpload() {
        compose.mixin(this, [withAuthToken]), this.UPLOAD_TIMEOUT = 3e4, this.UPLOAD_TYPE_TO_URL = {
            header: "/settings/profile/upload_profile_header",
            avatar: "/settings/profile/profile_image_update"
        }, this.saveImageWithIframeMethod = function(a) {
            this.trigger("uiEditProfileHideOptions"), this.trigger(document, "dataSaveImageWithAsyncIframeStart", {
                uploadType: this.attr.uploadType
            }), $.when(this.promiseSaveImage(a)).then(function(a) {
                this.trigger("dataSaveImageSuccess", a)
            }, function(a) {
                this.trigger("dataSaveImageFailure", a)
            })
        }, this.promiseSaveImage = function(a) {
            var b = a.upload_id = this.generateUploadId(), c = this.createIframeImageUploader(a), d = $.Deferred(), e = setTimeout(function() {
                d.rejectWith(this, [{
                    uploadType: this.attr.uploadType
                }
                ])
            }.bind(this), this.UPLOAD_TIMEOUT), f = this.handleIframeMessage(d, b), g = function() {
                c.find("form").remove(), c.find("iframe").remove(), clearTimeout(e), this.off(window, "message", f), c.remove(), this.trigger(document, "dataSaveImageWithAsyncIframeFinish", {
                    uploadType: this.attr.uploadType
                })
            }.bind(this);
            return d.always(g), this.on(window, "message", f), c.find("form").submit(), d.promise()
        }, this.handleIframeMessage = function(a, b) {
            return function(c) {
                var d = this.getMessageData(b, c);
                if (!d)
                    return;
                d.uploadType = this.attr.uploadType, d.status ? a.resolveWith(this, [d]) : a.rejectWith(this, [d])
            }.bind(this)
        }, this.getMessageData = function(a, b) {
            var c;
            b.originalEvent && (b = b.originalEvent);
            try {
                c = JSON.parse(b.data)
            } catch (d) {
                return !1
            }
            return c && (c.status || c.message) && c.upload_id == a ? c : !1
        }, this.createIframeImageUploader = function(a) {
            var b = location.protocol + "//" + location.hostname + (location.port ? ":" + location.port : ""), c = $(a.fileInput), d = {
                form_action: this.UPLOAD_TYPE_TO_URL[this.attr.uploadType],
                auth_token: this.getAuthToken(),
                origin: b,
                upload_id: a.upload_id,
                section_context: a.section,
                page_context: a.page,
                file_name: a.fileName,
                file_data: a.fileData
            }, e = template["highline/components/iframe_async_image_upload"].render(d, template), f = $(e), g = f.find(".IframeAsyncImageUpload-form"), h = f.find(".IframeAsyncImageUpload-iframe");
            if (c&&!a.fileData) {
                var i = c.parent();
                c.attr("name", "fileData"), g.append(c), this.reinsertEmptyFileInput(i)
            }
            return g.attr("target", h.attr("name")), this.$node.append(f), f
        }, this.reinsertEmptyFileInput = function(a) {
            var b = $('<input type="file" class="file-input js-tooltip" tabindex="-1">').attr({
                name: this.attr.fileInputString,
                "data-original-title": _('Add Photo')
            });
            a.append(b)
        }, this.generateUploadId = function() {
            return Date.now()
        }
    }
    var _ = require("core/i18n"), compose = require("core/compose"), template = require("template"), withAuthToken = require("app/data/with_auth_token");
    module.exports = withIframeAsyncImageUpload
});
define("app/ui/with_image_selection", ["module", "require", "exports", "core/compose", "app/utils/image", "app/utils/image_resize", "app/data/with_iframe_async_image_upload"], function(module, require, exports) {
    function withImageSelection() {
        compose.mixin(this, [withIframeAsyncImageUpload]), this.getFileData = image.getFileData.bind(image), this.getFileHandle = image.getFileHandle.bind(image), this.getFileName = image.getFileName.bind(image), this.resize = imageResize.resize.bind(image), this.validateFileName = image.validateFileName.bind(image), this.validateImageSize = image.validateImageSize.bind(image), this.defaultAttrs({
            buttonSelector: ".btn",
            fileDataSelector: "input.file-data",
            fileDataString: "media_data[]",
            fileInputString: "media[]",
            fileNameSelector: "input.file-name",
            fileNameString: "media_file_name",
            fileSelector: "input.file-input",
            maxGifSizeInBytes: 3145728,
            maxSizeInBytes: 5242880,
            swfSelector: ".swf-container",
            uploadType: "",
            supportsIframeUploadMethod: ["avatar", "header"]
        }), this.validateImage = function(a, b) {
            if (!this.validateFileName(a))
                return this.addFileError("notImage"), !1;
            if (b) {
                var c = image.isAnimatedGif(a), d = c ? this.validateImageSize(b, this.attr.maxGifSizeInBytes): this.validateImageSize(b, this.attr.maxSizeInBytes);
                if (!d)
                    return c ? this.addFileError("gifTooLarge") : this.addFileError("tooLarge"), !1
            }
            return !0
        }, this.imageSelected = function(a) {
            var b = this.select("fileSelector").get(0), c = this.getFileName(b.value);
            if (this.supportsCropper() || this.attr.supportsIframeUploadMethod.indexOf(this.attr.uploadType) < 0) {
                var d = this.getFileHandle(b);
                if (!this.validateImage(c, d))
                    return;
                this.gotFileHandle(c, d)
            } else {
                if (!this.validateImage(c))
                    return;
                var e = {
                    page: this.pageContext(),
                    section: this.sectionContext(),
                    fileInput: b,
                    uploadType: this.attr.uploadType,
                    fileName: c
                };
                this.saveImageWithIframeMethod(e)
            }
        }, this.pageContext = function() {
            return this.attr.eventData && this.attr.eventData.scribeContext && this.attr.eventData.scribeContext.page
        }, this.sectionContext = function() {
            return this.attr.eventData && this.attr.eventData.scribeContext && this.attr.eventData.scribeContext.section
        }, this.supportsCropper = function() {
            return image.supportsCropper()
        }, this.gotFileHandle = function(a, b) {
            this.mode() == "html5" ? this.getFileData(a, b, this.gotImageData.bind(this)) : this.gotFileInput(a)
        }, this.reset = function() {
            this.resetInput(), this.select("fileDataSelector").replaceWith('<input type="hidden" name="media_data_empty" class="file-data">'), this.trigger("uiPreviewThumbnailHide")
        }, this.getUploadId = function() {
            return + (new Date)
        }, this.gotFlashImageData = function(a, b, c) {
            if (!this.validateFileName(a)) {
                this.addFileError("notImage");
                return 
            }
            var d = this.getUploadId();
            this.addThumbnail({
                imageData: c,
                fileName: a,
                uploadId: d
            }), this.trigger("uiImagePickerAdd", {
                message: "flash"
            }), this.readyFileData(b, d), this.trigger("uiImagePickerFileReady", {
                uploadType: this.attr.uploadType
            })
        }, this.gotFlashImageError = function(a, b) {
            this.addFileError(a)
        }, this.gotResizedImageData = function(a, b, c) {
            if (!b) {
                this.addFileError("notImage");
                return 
            }
            if (this.photoCount && c) {
                this.addFileError("gifLimit");
                return 
            }
            var d = this.getUploadId();
            c && (this.gifId = d), this.addThumbnail({
                imageData: b,
                fileName: a,
                uploadId: d
            }), this.select("fileNameSelector").attr("value", a), this.trigger("uiImagePickerAdd", {
                message: "html5"
            });
            var e = b.split(",");
            e.length > 1 && (b = e[1]), this.readyFileData(b, d, c), this.trigger("uiImagePickerFileReady", {
                uploadType: this.attr.uploadType
            })
        }, this.gotFileInput = function(a) {
            var b = this.getUploadId();
            this.addThumbnail({
                fileName: a,
                uploadId: b
            }), this.trigger("uiImagePickerAdd", {
                message: "html4"
            }), this.readyFileInput(), this.trigger("uiImagePickerFileReady", {
                uploadType: this.attr.uploadType
            })
        }, this.readyFileInput = function() {
            this.select("fileSelector").attr("name", this.attr.fileInputString)
        }, this.readyFileData = function(a, b, c) {
            if (this.attr.photoLimit == 4) {
                var d = $('<input type="hidden" class="multi-photo-data">');
                d.attr("value", a), d.attr("data-upload-id", b), this.select("multiPhotoDataContainerSelector").append(d)
            } else 
                this.select("fileDataSelector").attr("value", a), this.select("fileDataSelector").attr("name", this.attr.fileDataString);
            this.resetInput()
        }, this.resetInput = function() {
            this.select("fileSelector").replaceWith('<input type="file" name="media_empty" class="file-input" tabindex="-1">')
        }, this.addThumbnail = function(a) {
            this.trigger("uiPreviewThumbnailAdd", a)
        }, this.setupFlash = function() {
            var a = "swift_tweetbox_callback_" + + (new Date), b = "swift_tweetbox_error_callback_" + + (new Date);
            window[a] = this.gotFlashImageData.bind(this), window[b] = this.gotFlashImageError.bind(this), setTimeout(function() {
                this.loadSwf(a, b)
            }.bind(this), 500)
        }, this.mode = function() {
            return this.attr.forceHTML5FileUploader && (this._mode = "html5"), this._mode = this._mode || image.mode(), this._mode
        }, this.setup = function() {
            this.mode() == "flash" && this.setupFlash(), this.select("fileNameSelector").attr("name", this.attr.fileNameString), this.select("fileDataSelector").attr("name", this.attr.fileDataString), this.select("fileSelector").attr("name", this.attr.fileInputString)
        }, this.after("initialize", function() {
            this.setup(), this.on("change", this.imageSelected)
        })
    }
    var compose = require("core/compose"), image = require("app/utils/image"), imageResize = require("app/utils/image_resize"), withIframeAsyncImageUpload = require("app/data/with_iframe_async_image_upload");
    module.exports = withImageSelection
});
define("app/ui/image_selector", ["module", "require", "exports", "core/component", "app/utils/image", "app/ui/with_image_selection", "app/data/user_info", "core/i18n"], function(module, require, exports) {
    function imageSelector() {
        this.defaults = {
            swfHeight: 30,
            swfWidth: 42,
            tweetBoxSelector: ".tweet-form",
            photoButtonSelector: ".file-input",
            addPhotoLabelSelector: ".add-photo-label",
            multiPhotoDataSelector: ".multi-photo-data",
            multiPhotoDataContainerSelector: ".multi-photo-data-container",
            interceptAvatarImageData: !1,
            photoLimit: 1
        }, this.disable = function() {
            this.$node.addClass("disabled"), this.select("buttonSelector").attr("disabled", !0).addClass("enabled")
        }, this.enable = function() {
            this.$node.removeClass("disabled"), this.select("buttonSelector").attr("disabled", !1).removeClass("enabled")
        }, this.thumbAdded = function(a, b) {
            this.photoCount++, (this.photoCount >= this.attr.photoLimit || this.gifId) && this.disable(), this.updateAddPhotoLabel();
            var c = b ? {
                isGif: image.isAnimatedGif(b.fileName)
            }
            : {};
            this.trigger("uiPreviewThumbnailShow", c)
        }, this.thumbRemoved = function(a, b) {
            var c = b.uploadId;
            this.select("multiPhotoDataContainerSelector").find("[data-upload-id=" + c + "]").remove(), this.photoCount = Math.max(--this.photoCount, 0), this.gifId == c && (this.gifId = 0), this.photoCount < this.attr.photoLimit&&!this.gifId && this.enable(), this.updateAddPhotoLabel(), this.photoCount == 0 && this.trigger("uiPreviewThumbnailHide")
        }, this.removeAllThumbs = function() {
            this.photoCount = 0, this.gifId = 0, this.updateAddPhotoLabel(), this.reset(), this.enable(), this.select("multiPhotoDataContainerSelector").empty()
        }, this.gotImageData = function(a, b) {
            image.isAnimatedGif(a) ? this.gotResizedImageData(a, b.split(";base64,")[1], !0) : this.resize(a, b, 2048, this.gotResizedImageData.bind(this))
        }, this.interceptGotImageData = function(a, b) {
            this.$node.hasClass("disabled") || this.gotImageData(b.name, b.contents)
        }, this.interceptAvatarImageData = function(a, b) {
            b.uploadType === "avatar" && this.gotResizedImageData(b.fileName, b.fileData)
        }, this.addFileError = function(a) {
            a == "tooLarge" ? this.trigger("uiShowError", {
                message: _('Sorry! The file is too big. Maximum file size is 5MB.')
            }) : a == "gifTooLarge" ? this.trigger("uiShowError", {
                message: _('Sorry! The file is too big. Maximum file size is 3MB.')
            }) : a == "notImage" || a == "ioError" ? this.trigger("uiShowError", {
                message: _('You did not select an image.')
            }) : a == "gifLimit" && this.trigger("uiShowError", {
                message: _('You can not upload multiple GIFs.')
            }), this.trigger("uiImagePickerError", {
                message: a
            })
        }, this.loadSwf = function(a, b) {
            image.loadPhotoHelperSwf(this.select("swfSelector"), a, b, this.attr.swfHeight, this.attr.swfWidth)
        }, this.imageSelectorClicked = function(a, b) {
            this.supportsImageUpload() ? this.trigger("uiImagePickerClick") : (this.trigger("uiShowError", {
                message: _('You need Adobe Flash to upload photos. <a href="http://get.adobe.com/flashplayer">Install Adobe Flash</a>')
            }), a.preventDefault(), a.stopPropagation())
        }, this.supportsImageUpload = function() {
            return image.hasFileReader() || image.hasFlash()
        }, this.updateAddPhotoLabel = function() {
            var a = "";
            this.attr.photoLimit == 1 || this.attr.photoLimit == 4 && this.photoCount == 0 ? a = _('Add photo') : this.gifId ? a = _('1 added') : this.attr.photoLimit == 4 && this.photoCount == 4 ? a = _('4 added') : a = _('Add more'), this.select("addPhotoLabelSelector").text(a)
        }, this.after("initialize", function() {
            this.on("click", {
                photoButtonSelector: this.imageSelectorClicked
            });
            var a = this.$node.closest(this.attr.tweetBoxSelector);
            this.on(a, "uiPreviewThumbnailRemove", this.thumbRemoved), this.on(a, "uiPreviewThumbnailAdd", this.thumbAdded), this.on(a, "uiPreviewThumbnailRemoveAll", this.removeAllThumbs), this.on(a, "uiGotImageData", this.interceptGotImageData), this.on(a, "uiEnableImageSelection", this.enable), this.on(a, "uiDisableImageSelection", this.disable), this.attr.interceptAvatarImageData && this.on(document, "uiImageSave", this.interceptAvatarImageData), this.gifId = 0, this.photoCount = 0
        })
    }
    var defineComponent = require("core/component"), image = require("app/utils/image"), withImageSelection = require("app/ui/with_image_selection"), userInfo = require("app/data/user_info"), _ = require("core/i18n"), ImageSelector = defineComponent(imageSelector, withImageSelection);
    module.exports = ImageSelector
});
define("app/utils/unique_id", ["module", "require", "exports"], function(module, require, exports) {
    function uniqueId(a, b) {
        b = b || defaultPrefix;
        var c = a.id;
        if (c)
            return c;
        do 
            count++, c = b + count;
        while (document.getElementById(c));
        return a.id = c, c
    }
    var defaultPrefix = "uuid-", count = 0;
    module.exports = uniqueId
});
define("app/ui/with_click_outside", ["module", "require", "exports"], function(module, require, exports) {
    function withClickOutside() {
        this.onClickOutside = function(a, b) {
            b = b.bind(this), this.clickOutsideHandler = function(c, d) {
                var e=!0;
                a.each(function() {
                    if ($(c.target).closest(this).length)
                        return e=!1, !1
                }), e && b(c, d)
            }, $(document).on("click", this.clickOutsideHandler)
        }, this.offClickOutside = function() {
            this.clickOutsideHandler && ($(document).off("click", this.clickOutsideHandler), this.clickOutsideHandler = null)
        }, this.before("teardown", function() {
            this.offClickOutside()
        })
    }
    module.exports = withClickOutside
});
define("app/ui/with_geo_search_dropdown", ["module", "require", "exports", "core/compose", "app/utils/unique_id", "app/ui/with_click_outside", "core/advice"], function(module, require, exports) {
    function withGeoSearchDropdown() {
        compose.mixin(this, [withClickOutside, advice.withAdvice]), this.defaultAttrs({
            querySelector: ".GeoSearch-queryInput",
            dropdownTriggerSelector: ".js-geo-search-trigger",
            dropdownSelector: ".GeoSearch-dropdownMenu",
            resultSelector: ".GeoSearch-result",
            resultClass: "GeoSearch-result",
            focusableSelector: ".GeoSearch-focusable",
            firstFocusableSelector: ".GeoSearch-focusable:first",
            focusedSelector: ".GeoSearch-focusable.is-focused",
            focusedClass: "is-focused"
        }), this.getQuery = function() {
            return this.select("querySelector").val() || ""
        }, this.captureActiveEl = function() {
            this.activeEl = document.activeElement
        }, this.applyAriaAttrsToDropdown = function() {
            var a = this.select("dropdownSelector").attr("id");
            if (!a) {
                if (window.DEBUG && window.DEBUG.enabled)
                    throw new Error("Application of ARIA attributes to geo search dropdown failed.");
                return 
            }
            this.$queryInput.attr({
                "aria-owns": a,
                "aria-expanded": !1,
                "aria-autocomplete": "list"
            }), this.select("dropdownSelector").attr("role", "listbox"), this.ariaApplied=!0
        }, this.applyAriaAttrsToResultsAndAnnounceNumResults = function() {
            if (!this.ariaApplied)
                return;
            var a = this.select("resultSelector").length;
            a && (this.select("resultSelector").each(function(a, b) {
                $(b).attr("role", "option"), uniqueId(b)
            }), this.trigger("uiGeoResultsShown", {
                numResults: a
            }))
        }, this.openDropdown = function() {
            if (this.dropdownIsOpen)
                return;
            this.captureActiveEl();
            var a = this.select("dropdownSelector");
            this.onClickOutside(a.add(this.select("dropdownTriggerSelector")), this.hideDropdown), a.show(), this.dropdownIsOpen=!0, this.ariaApplied && this.$queryInput.attr("aria-expanded", !0)
        }, this.hideDropdown = function() {
            if (!this.dropdownIsOpen)
                return;
            this.offClickOutside(), this.select("dropdownSelector").hide(), this.dropdownIsOpen=!1, this.ariaApplied && (this.select("focusedSelector").removeAttr("aria-selected"), this.select("focusedSelector").removeAttr("tabindex"), this.$queryInput.attr("aria-expanded", !1), this.$queryInput.removeAttr("aria-activedescendant"))
        }, this.hideDropdownAndRestoreFocus = function() {
            this.hideDropdown(), this.activeEl && (this.activeEl.focus(), this.activeEl = null)
        }, this.onEsc = function(a) {
            if (!this.dropdownIsOpen)
                return;
            a.preventDefault(), a.stopPropagation(), this.hideDropdownAndRestoreFocus()
        }, this.setFocus = function(a) {
            var b = $(a.target);
            this.select("focusedSelector").not(b).removeClass(this.attr.focusedClass), b.addClass(this.attr.focusedClass)
        }, this.clearFocus = function(a) {
            $(a.target).removeClass(this.attr.focusedClass)
        }, this.moveFocus = function(a) {
            var b = this.select("focusedSelector"), c = this.select("focusableSelector"), d = c.index(b) + a, e = c.length - 1;
            d < 0 ? d = e : d > e && (d = 0);
            var f = c.eq(d);
            b.removeClass(this.attr.focusedClass), f.addClass(this.attr.focusedClass), this.ariaApplied && (b.removeAttr("aria-selected"), b.removeAttr("tabindex"), f.attr({
                "aria-selected": !0,
                tabindex: "-1"
            }), this.$queryInput.attr("aria-activedescendant", f.attr("id")))
        }, this.queryKeyDown = function(a) {
            switch (a.which) {
            case 38:
                a.preventDefault(), this.moveFocus( - 1);
                return;
            case 40:
                a.preventDefault(), this.moveFocus(1);
                return;
            case 37:
            case 39:
                return;
            case 13:
                a.preventDefault();
                var b = this.select("focusedSelector");
                if (b.length) {
                    a.stopPropagation(), b.trigger("uiGeoSearchSelect");
                    return 
                }
                this.trigger("uiGeoSearchEnter");
                return;
            case 9:
                this.trigger("uiGeoSearchTab");
                return 
            }
            this.trigger("uiGeoSearchAutocomplete")
        }, this.selectResult = function(a) {
            var b = $(a.target), c = b.data("place-id"), d = b.text(), e = b.data("result-type"), f = b.data("result-type") ? '[data-result-type="' + b.data("result-type") + '"]': this.attr.resultSelector, g = this.$node.find(f).index(b);
            this.trigger("uiGeoSearchChangeLocation", {
                placeId: c,
                placeName: d,
                resultType: e,
                placeOffset: g
            })
        }, this.changeQuery = function(a) {
            this.trigger("uiGeoSearchAutocomplete")
        }, this.after("initialize", function() {
            this.$queryInput = this.select("querySelector"), this.on("click", {
                resultSelector: this.selectResult
            }), this.on("uiGeoSearchSelect", {
                resultSelector: this.selectResult
            }), this.on("mouseover", {
                focusableSelector: this.setFocus
            }), this.on("mouseout", {
                focusableSelector: this.clearFocus
            }), this.on("change paste cut input", {
                querySelector: this.changeQuery
            }), this.on("keydown", {
                querySelector: this.queryKeyDown
            }), this.on("uiShortcutEsc", this.onEsc)
        }), this.before("teardown", this.hideDropdown)
    }
    var compose = require("core/compose"), uniqueId = require("app/utils/unique_id"), withClickOutside = require("app/ui/with_click_outside"), advice = require("core/advice");
    module.exports = withGeoSearchDropdown
});
define("app/ui/geo_picker", ["module", "require", "exports", "core/component", "core/i18n", "app/ui/with_geo_search_dropdown", "core/utils", "app/utils/scribe_item_types"], function(module, require, exports) {
    function geoPicker() {
        this.defaultAttrs({
            buttonSelector: "button.geo-picker-btn",
            placeIdSelector: "input[name=place_id]",
            statusSelector: "span.geo-status",
            dropdownContainerSelector: "span.dropdown-container",
            dropdownDisabledSelector: "#geo-disabled-dropdown",
            enableButtonSelector: "button.geo-turn-on",
            notNowButtonSelector: "button.geo-not-now",
            dropdownEnabledSelector: "#geo-enabled-dropdown",
            geoSearchSelector: "li.geo-query-location .icon",
            dropdownStatusSelector: "li.geo-dropdown-status",
            searchResultsSelector: ".GeoSearch-result--search",
            placeResultsSelector: ".GeoSearch-result--place",
            turnOffButtonSelector: "li.geo-turn-off-item",
            disabledClass: "disabled"
        }), this.selectGeoAction = function(a) {
            if (this.dropdownIsOpen) {
                this.hideDropdownAndRestoreFocus();
                return 
            }
            switch (this.geoState.state) {
            case"disabled":
                this.disable(), this.trigger("uiGeoPickerOffer");
                break;
            case"enableIsUnavailable":
                this.trigger("uiGeoPickerOffer");
                break;
            case"locateIsUnavailable":
            case"enabledTurnedOn":
                this.requestGeoState();
                break;
            case"enabledTurnedOff":
                this.turnOn();
                break;
            case"changing":
            case"locating":
            case"located":
            case"locationUnknown":
                this.trigger("uiGeoPickerOpen")
            }
        }, this.prepareAndOpenDropdown = function(a, b) {
            var c, d;
            a.type == "uiGeoPickerOpen" ? (c = this.attr.dropdownEnabledSelector, d = 1, this.geoDetails&&!this.geoDetails.interacted_poi_list && (this.geoDetails.interacted_poi_list=!0)) : (c = this.attr.dropdownDisabledSelector, d = 0), d != this.dropdownState && (this.select("dropdownContainerSelector").html($(c).html()), this.dropdownState = d), this.showGeoState(), this.lastQuery = "", this.geoQueryFieldChanged=!1, this.openDropdown()
        }, this.after("openDropdown", function() {
            var a = this.select("enableButtonSelector");
            a.length || (a = this.select("querySelector")), a.focus()
        }), this.after("hideDropdown", function() {
            this.select("querySelector").val("")
        }), this.enable = function() {
            this.hideDropdownAndRestoreFocus(), this.trigger("uiGeoPickerEnable")
        }, this.turnOn = function() {
            this.trigger("uiGeoPickerTurnOn")
        }, this.turnOff = function(a) {
            this.hideDropdownAndRestoreFocus(), this.trigger("uiGeoPickerTurnOff")
        }, this.changeLocation = function(a, b) {
            var c = this.select("querySelector").val(), d = b.placeId, e = b.placeName, f = b.resultType === "search", g = b.placeOffset;
            this.hideDropdownAndRestoreFocus();
            if (!d || d === this.lastPlaceId)
                return;
            this.lastGeoPlace && (this.discardLastGeoPlace(), this.lastGeoPlace = {
                place_name: e,
                place_id: d,
                offset: g,
                is_autotag: !1,
                query: c !== "" ? c: null,
                source: f ? "search": "default",
                last_interaction_time: (new Date).getTime()
            });
            var h = {
                placeId: d,
                scribeData: {
                    items: [{
                        name: d
                    }
                    ]
                }
            };
            this.lastPlaceId && h.scribeData.items.push({
                name: this.lastPlaceId
            }), f && this.lastQueryData && (h.scribeData.query = this.lastQueryData.query), this.trigger("uiGeoPickerChange", h)
        }, this.updateState = function(a, b) {
            this.geoState = b, this.showGeoState()
        }, this.showGeoState = function() {
            var a = "", b = "", c=!1, d = "", e = this.geoState;
            switch (e.state) {
            case"enabling":
                this.select("buttonSelector").removeClass(this.attr.disabledClass);
            case"locating":
                a = _('Getting location...');
                break;
            case"enableIsUnavailable":
            case"locateIsUnavailable":
                a = _('Location service unavailable');
                break;
            case"changing":
                a = _('Changing location...');
                break;
            case"locationUnknown":
                a = _('Unknown location');
                break;
            case"located":
                a = e.place_name, b = e.place_id, d = e.places_html, c=!0, this.initializeGeoScribeData(e);
                break;
            case"enabledTurnedOff":
                a = _('Add location'), this.discardLastGeoPlace();
                break;
            case"disabled":
                this.disable(), a = _('Location disabled')
            }
            this.$node.toggleClass("active", c), c ? this.select("buttonSelector").addClass("enabled") : this.select("buttonSelector").removeClass("enabled"), this.select("statusSelector").text(a), this.select("buttonSelector").attr("title", a || _('Add location')), this.select("placeResultsSelector").add(this.select("searchResultsSelector")).remove(), this.select("dropdownStatusSelector").text(a).toggle(!c).after(d), this.select("placeIdSelector").val(b), this.lastPlaceId = b
        }, this.initializeGeoScribeData = function(a) {
            this.geoDetails || this.resetGeoDetails(), (!this.lastGeoPlace ||!this.lastGeoPlace.place_id) && a.place_id && (this.lastGeoPlace = {
                place_name: a.place_name,
                place_id: a.place_id,
                source: "default",
                is_autotag: !0,
                last_interaction_time: (new Date).getTime()
            })
        }, this.resetGeoDetails = function() {
            this.geoDetails = {
                interacted_poi_list: !1,
                interacted_server_search: !1,
                geo_place_details: []
            }
        }, this.discardLastGeoPlace = function() {
            this.lastGeoPlace && this.lastGeoPlace.place_id && (this.lastGeoPlace.state = "discarded", this.geoDetails.geo_place_details.push(this.lastGeoPlace), this.lastGeoPlace = {})
        }, this.sendTweetScribeData = function(a) {
            if (!this.geoDetails)
                return;
            this.lastGeoPlace && this.lastGeoPlace.place_id && (this.lastGeoPlace.state = "geotag", this.geoDetails.geo_place_details.push(this.lastGeoPlace));
            var b = {
                scribeData: {
                    items: [{
                        item_type: scribeItemTypes.geoDetails,
                        geo_details: this.geoDetails
                    }
                    ]
                }
            };
            this.trigger("uiGeoPickerSendTweet", b), this.resetGeoDetails(), this.lastGeoPlace = {}
        }, this.requestGeoState = function() {
            this.trigger("uiRequestGeoState")
        }, this.searchIfQueryChanged = function(a) {
            var b = this.select("querySelector").val() || "";
            if (a && this.lastQuery === b)
                return;
            this.geoDetails.interacted_server_search || (this.geoDetails.interacted_server_search=!0), this.lastIsPrefix = a, this.lastQuery = b, this.select("dropdownStatusSelector").text(_('Searching places...')).show(), this.geoQueryFieldChanged || (this.geoQueryFieldChanged=!0, this.trigger("uiGeoPickerInteraction")), this.trigger("uiGeoPickerSearch", {
                placeId: this.lastPlaceId,
                query: b,
                isPrefix: a
            })
        }, this.searchExactMatch = function() {
            this.searchIfQueryChanged(!1)
        }, this.searchAutocomplete = function() {
            setTimeout(function() {
                this.searchIfQueryChanged(!0)
            }.bind(this), 0)
        }, this.searchResults = function(a, b) {
            var c = b.sourceEventData;
            if (!c || c.placeId !== this.lastPlaceId || c.query !== this.select("querySelector").val() || c.isPrefix&&!this.lastIsPrefix)
                return;
            this.lastQueryData = c, this.select("searchResultsSelector").remove(), this.select("dropdownStatusSelector").hide().after(b.html)
        }, this.searchUnavailable = function(a, b) {
            this.select("dropdownStatusSelector").text(_('Location service unavailable')).show()
        }, this.preventFocusLoss = function(a) {
            var b;
            $.browser.msie && parseInt($.browser.version, 10) < 9 ? (b = $(document.activeElement), b.is(this.select("buttonSelector")) ? this.captureActiveEl() : b.one("beforedeactivate", function(a) {
                a.preventDefault()
            })) : a.preventDefault()
        }, this.disable = function() {
            this.select("buttonSelector").addClass(this.attr.disabledClass), this.select("statusSelector").text(_('Location disabled')), this.select("buttonSelector").attr("title", _('Location disabled'))
        }, this.after("initialize", function() {
            utils.push(this.attr, {
                eventData: {
                    scribeContext: {
                        element: "geo_picker"
                    }
                }
            }, !1), this.geoState = {}, this.on(this.attr.parent, "uiComposerActive uiPrepareGeoPicker", this.requestGeoState), this.on(this.attr.parent, "uiTweetSent", this.sendTweetScribeData), this.on(document, "dataGeoState", this.updateState), this.on(document, "dataGeoSearchResults", this.searchResults), this.on(document, "dataGeoSearchResultsUnavailable", this.searchUnavailable), this.on("mousedown", {
                buttonSelector: this.preventFocusLoss
            }), this.on("click", {
                buttonSelector: this.selectGeoAction,
                enableButtonSelector: this.enable,
                notNowButtonSelector: this.hideDropdownAndRestoreFocus,
                turnOffButtonSelector: this.turnOff,
                geoSearchSelector: this.searchExactMatch
            }), this.on("uiGeoSearchSelect", {
                turnOffButtonSelector: this.turnOff
            }), this.on("uiGeoPickerOpen uiGeoPickerOffer", this.prepareAndOpenDropdown), this.on("uiGeoSearchEnter", this.searchExactMatch), this.on("uiGeoSearchChangeLocation", this.changeLocation), this.on("uiGeoSearchAutocomplete", this.searchAutocomplete)
        })
    }
    var defineComponent = require("core/component"), _ = require("core/i18n"), withGeoSearchDropdown = require("app/ui/with_geo_search_dropdown"), utils = require("core/utils"), scribeItemTypes = require("app/utils/scribe_item_types");
    module.exports = defineComponent(withGeoSearchDropdown, geoPicker)
});
define("app/utils/tweet_helper", ["module", "require", "exports", "lib/twitter-text", "core/utils", "app/data/user_info"], function(module, require, exports) {
    var twitterText = require("lib/twitter-text"), utils = require("core/utils"), userInfo = require("app/data/user_info"), VALID_PROTOCOL_PREFIX_REGEX = /^https?:\/\//i, tweetHelper = {
        extractMentionsForReply: function(a, b) {
            var c = a.attr("data-screen-name"), d = a.attr("data-retweeter"), e = a.attr("data-mentions") ? a.attr("data-mentions").split(" "): [], f = [c, b, d];
            return e = e.filter(function(a) {
                return f.indexOf(a) < 0
            }), d && d != c && d != b && e.unshift(d), (!e.length || c != b) && e.unshift(c), e
        },
        linkify: function(a, b) {
            return b = utils.merge({
                hashtagClass: "twitter-hashtag pretty-link",
                hashtagUrlBase: "/search?q=%23",
                symbolTag: "s",
                textWithSymbolTag: "b",
                cashtagClass: "twitter-cashtag pretty-link",
                cashtagUrlBase: "/search?q=%24",
                usernameClass: "twitter-atreply pretty-link",
                usernameUrlBase: "/",
                usernameIncludeSymbol: !0,
                listClass: "twitter-listname pretty-link",
                urlClass: "twitter-timeline-link",
                urlTarget: "_blank",
                suppressNoFollow: !0,
                htmlEscapeNonEntities: !0
            }, b || {}), twitterText.autoLinkEntities(a, twitterText.extractEntitiesWithIndices(a), b)
        }
    };
    module.exports = tweetHelper
});
define("app/utils/html_text", ["module", "require", "exports"], function(module, require, exports) {
    function isTextNode(a) {
        return a.nodeType == 3 || a.nodeType == 4
    }
    function isElementNode(a) {
        return a.nodeType == 1
    }
    function isBrNode(a) {
        return isElementNode(a) && a.nodeName.toLowerCase() == "br"
    }
    function isOutsideContainer(a, b) {
        while (a !== b) {
            if (!a)
                return !0;
            a = a.parentNode
        }
    }
    var useW3CRange = window.getSelection, useMsftTextRange=!useW3CRange && document.selection, useIeHtmlFix = navigator.appName == "Microsoft Internet Explorer", NBSP_REGEX = /[\xa0\n\t]/g, CRLF_REGEX = /\r\n/g, LINES_REGEX = /(.*?)\n/g, SP_LEADING_OR_FOLLOWING_CLOSE_TAG_OR_PRECEDING_A_SP_REGEX = /^ |(<\/[^>]+>) | (?= )/g, SP_LEADING_OR_TRAILING_OR_FOLLOWING_A_SP_REGEX = /^ | $|( ) /g, MAX_OFFSET = Number.MAX_VALUE, htmlText = function(a, b) {
        function c(a, c) {
            function h(a) {
                var i = d.length;
                if (isTextNode(a)) {
                    var j = a.nodeValue.replace(NBSP_REGEX, " "), k = j.length;
                    k && (d += j, e=!0), c(a, !0, 0, i, i + k)
                } else if (isElementNode(a)) {
                    c(a, !1, 0, i, i);
                    if (isBrNode(a))
                        a == f ? g=!0 : (d += "\n", e=!1);
                    else {
                        var l = a.currentStyle || window.getComputedStyle(a, ""), m = l.display == "block";
                        m && b.msie && (e=!0);
                        for (var n = a.firstChild, o = 1; n; n = n.nextSibling, o++) {
                            h(n);
                            if (g)
                                return;
                            i = d.length, c(a, !1, o, i, i)
                        }
                        g || a == f ? g=!0 : m && e && (d += "\n", e=!1)
                    }
                }
            }
            var d = "", e, f, g;
            for (var i = a; i && isElementNode(i); i = i.lastChild)
                f = i;
            return h(a), d
        }
        function d(a, b) {
            var d = null, e = b.length - 1;
            if (useW3CRange) {
                var f = b.map(function() {
                    return {}
                }), g;
                c(a, function(a, c, d, h, i) {
                    g || f.forEach(function(f, j) {
                        var k = b[j];
                        h <= k&&!isBrNode(a) && (f.node = a, f.offset = c ? Math.min(k, i) - h : d, g = c && j == e && i >= k)
                    })
                }), f[0].node && f[e].node && (d = document.createRange(), d.setStart(f[0].node, f[0].offset), d.setEnd(f[e].node, f[e].offset))
            } else if (useMsftTextRange) {
                var h = document.body.createTextRange();
                h.moveToElementText(a), d = h.duplicate();
                if (b[0] == MAX_OFFSET)
                    d.setEndPoint("StartToEnd", h);
                else {
                    d.move("character", b[0]);
                    var i = e && b[1] - b[0];
                    i > 0 && d.moveEnd("character", i), h.inRange(d) || d.setEndPoint("EndToEnd", h)
                }
            }
            return d
        }
        function e() {
            return a.offsetWidth && a.offsetHeight
        }
        function f(b) {
            a.innerHTML = b;
            if (useIeHtmlFix)
                for (var c = a.firstChild; c; c = c.nextSibling)
                    isElementNode(c) && c.nodeName.toLowerCase() == "p" && c.innerHTML == "" && (c.innerText = "")
        }
        function g(a, b) {
            return a.map(function(a) {
                return Math.min(a, b.length)
            })
        }
        function h() {
            var b = getSelection();
            if (b.rangeCount !== 1)
                return null;
            var d = b.getRangeAt(0);
            if (isOutsideContainer(d.commonAncestorContainer, a))
                return null;
            var e = [{
                node: d.startContainer,
                offset: d.startOffset
            }
            ];
            d.collapsed || e.push({
                node: d.endContainer,
                offset: d.endOffset
            });
            var f = e.map(function() {
                return MAX_OFFSET
            }), h = c(a, function(a, b, c, d) {
                e.forEach(function(e, g) {
                    f[g] == MAX_OFFSET && a == e.node && (b || c == e.offset) && (f[g] = d + (b ? e.offset : 0))
                })
            });
            return g(f, h)
        }
        function i() {
            var b = document.selection.createRange();
            if (isOutsideContainer(b.parentElement(), a))
                return null;
            var d = ["Start"];
            b.compareEndPoints("StartToEnd", b) && d.push("End");
            var e = d.map(function() {
                return MAX_OFFSET
            }), f = document.body.createTextRange(), h = c(a, function(c, g, h, i) {
                function j(a, c) {
                    if (e[c] < MAX_OFFSET)
                        return;
                    var d = f.compareEndPoints("StartTo" + a, b);
                    if (d > 0)
                        return;
                    var g = f.compareEndPoints("EndTo" + a, b);
                    if (g < 0)
                        return;
                    var h = f.duplicate();
                    h.setEndPoint("EndTo" + a, b), e[c] = i + h.text.length, c&&!g && e[c]++
                }
                !g&&!h && c != a && (f.moveToElementText(c), d.forEach(j))
            });
            return g(e, h)
        }
        return {
            getHtml: function() {
                if (useIeHtmlFix) {
                    var b = "", c = document.createElement("div");
                    for (var d = a.firstChild; d; d = d.nextSibling)
                        isTextNode(d) ? (c.innerText = d.nodeValue, b += c.innerHTML) : b += d.outerHTML.replace(CRLF_REGEX, "");
                    return b
                }
                return a.innerHTML
            },
            setHtml: function(a) {
                f(a)
            },
            getText: function() {
                return c(a, function() {})
            },
            setTextWithMarkup: function(a) {
                f((a + "\n").replace(LINES_REGEX, function(a, c) {
                    return b.mozilla || b.msie ? (c = c.replace(SP_LEADING_OR_FOLLOWING_CLOSE_TAG_OR_PRECEDING_A_SP_REGEX, "$1&nbsp;"), b.mozilla ? c + "<BR>" : "<P>" + c + "</P>") : (c = (c || "<br>").replace(SP_LEADING_OR_TRAILING_OR_FOLLOWING_A_SP_REGEX, "$1&nbsp;"), b.opera ? "<p>" + c + "</p>" : "<div>" + c + "</div>")
                }))
            },
            getSelectionOffsets: function() {
                var a = null;
                return e() && (useW3CRange ? a = h() : useMsftTextRange && (a = i())), a
            },
            setSelectionOffsets: function(b) {
                if (b && e()) {
                    var c = d(a, b);
                    if (c)
                        if (useW3CRange) {
                            var f = window.getSelection();
                            f.removeAllRanges(), f.addRange(c)
                        } else 
                            useMsftTextRange && c.select()
                }
            },
            emphasizeText: function(b) {
                var f = [];
                b && b.length > 1 && e() && (c(a, function(a, c, d, e, g) {
                    if (c) {
                        var h = Math.max(e, b[0]), i = Math.min(g, b[1]);
                        i > h && f.push([h, i])
                    }
                }), f.forEach(function(b) {
                    var c = d(a, b);
                    c && (useW3CRange ? c.surroundContents(document.createElement("em")) : useMsftTextRange && c.execCommand("italic", !1, null))
                }))
            }
        }
    };
    module.exports = htmlText
});
define("app/ui/with_rich_editor", ["module", "require", "exports", "app/utils/tweet_helper", "lib/twitter-text", "app/utils/html_text"], function(module, require, exports) {
    function withRichEditor() {
        this.defaultAttrs({
            richSelector: "div.rich-editor",
            linksSelector: "a",
            normalizerSelector: "div.rich-normalizer"
        }), this.linkify = function(a) {
            var b = {
                urlTarget: null,
                textWithSymbolTag: RENDER_URLS_AS_PRETTY_LINKS ? "b": "",
                linkAttributeBlock: function(a, b) {
                    var c = a.screenName || a.url;
                    c && (this.urlAndMentionsCharCount += c.length + 2), delete b.title, delete b["data-screen-name"], b.dir = a.hashtag && this.shouldBeRTL(a.hashtag, 0) ? "rtl" : "ltr", b.role = "presentation"
                }.bind(this)
            };
            return this.urlAndMentionsCharCount = 0, tweetHelper.linkify(a, b)
        }, this.around("setSelection", function(a, b) {
            b && this.setSelectionIfFocused(b)
        }), this.around("setCursorPosition", function(a, b) {
            b === undefined && (b = this.attr.cursorPosition), b === undefined && (b = MAX_OFFSET), this.setSelectionIfFocused([b])
        }), this.around("detectUpdatedText", function(a, b, c) {
            var d = this.htmlRich.getHtml(), e = this.htmlRich.getSelectionOffsets() || [MAX_OFFSET], f = c !== undefined;
            if (d === this.prevHtml && e[0] === this.prevSelectionOffset&&!b&&!f)
                return;
            var g = f ? c: this.htmlRich.getText(), h = g.replace(INVALID_CHARS_REGEX, "");
            (f ||!(!d&&!this.hasFocus() || this.$text.attr("data-in-composition"))) && this.reformatHtml(h, d, e, f);
            if (b || this.cleanedText != h || this.prevSelectionOffset != e[0])
                this.prevSelectionOffset = e[0], this.updateCleanedTextAndOffset(h, e[0])
        }), this.reformatHtml = function(a, b, c, d) {
            this.htmlNormalizer.setTextWithMarkup(this.linkify(a));
            var e = this.shouldBeRTL(a, this.urlAndMentionsCharCount);
            this.$text.attr("dir", e ? "rtl" : "ltr"), this.$normalizer.find(e ? "[dir=rtl]" : "[dir=ltr]").removeAttr("dir"), RENDER_URLS_AS_PRETTY_LINKS && this.$normalizer.find(".twitter-timeline-link").wrapInner("<b>").addClass("pretty-link");
            var f = this.getMaxLengthOffset(a);
            f >= 0 && (this.htmlNormalizer.emphasizeText([f, MAX_OFFSET]), this.$normalizer.find("em").each(function() {
                this.innerHTML = this.innerHTML.replace(TRAILING_SINGLE_SPACE_REGEX, " ")
            }));
            var g = this.htmlNormalizer.getHtml();
            if (g !== b) {
                var h = d&&!this.isFocusing && this.hasFocus();
                h && this.$text.addClass("fake-focus").blur(), this.htmlRich.setHtml(g), h && this.$text.focus().removeClass("fake-focus"), this.setSelectionIfFocused(c)
            }
            this.prevHtml = g
        }, this.getMaxLengthOffset = function(a) {
            var b = this.getLength(a), c = this.attr.maxLength;
            if (b <= c)
                return - 1;
            c += twitterText.getUnicodeTextLength(a) - b;
            var d = [{
                indices: [c, c]
            }
            ];
            return twitterText.modifyIndicesFromUnicodeToUTF16(a, d), d[0].indices[0]
        }, this.setSelectionIfFocused = function(a) {
            this.hasFocus() ? this.htmlRich.setSelectionOffsets(a) : this.previousSelection = a
        }, this.selectPrevCharOnBackspace = function(a) {
            if (a.which == 8&&!a.ctrlKey) {
                var b = this.htmlRich.getSelectionOffsets();
                b && b[0] != MAX_OFFSET && b.length == 1 && (b[0] ? this.setSelectionIfFocused([b[0] - 1, b[0]]) : this.stopEvent(a))
            }
        }, this.emulateCommandArrow = function(a) {
            if (a.metaKey&&!a.shiftKey && (a.which == 37 || a.which == 39)) {
                var b = a.which == 37;
                this.htmlRich.setSelectionOffsets([b ? 0: MAX_OFFSET]), this.$text.scrollTop(b ? 0 : this.$text[0].scrollHeight), this.stopEvent(a)
            }
        }, this.stopEvent = function(a) {
            a.preventDefault(), a.stopPropagation()
        }, this.saveUndoStateDeferred = function(a) {
            a.type != "focusin" && this.saveUndoState(), setTimeout(function() {
                this.detectUpdatedText(), this.saveUndoState()
            }.bind(this), 0)
        }, this.clearUndoState = function() {
            this.undoHistory = [], this.undoIndex =- 1
        }, this.saveUndoState = function() {
            var a = this.htmlRich.getText(), b = this.htmlRich.getSelectionOffsets() || [a.length], c = this.undoHistory, d = c[this.undoIndex];
            (!d || d[0] !== a) && c.splice(++this.undoIndex, c.length, [a, b])
        }, this.isUndoKey = function(a) {
            return this.isMac ? a.which == 90 && a.metaKey&&!a.shiftKey&&!a.ctrlKey&&!a.altKey : a.which == 90 && a.ctrlKey&&!a.shiftKey&&!a.altKey
        }, this.emulateUndo = function(a) {
            this.isUndoKey(a) && (this.stopEvent(a), this.saveUndoState(), this.undoIndex > 0 && this.setUndoState(this.undoHistory[--this.undoIndex]))
        }, this.isRedoKey = function(a) {
            return this.isMac ? a.which == 90 && a.metaKey && a.shiftKey&&!a.ctrlKey&&!a.altKey : this.isWin ? a.which == 89 && a.ctrlKey&&!a.shiftKey&&!a.altKey : a.which == 90 && a.shiftKey && a.ctrlKey&&!a.altKey
        }, this.emulateRedo = function(a) {
            var b = this.undoHistory, c = this.undoIndex;
            c < b.length - 1 && this.htmlRich.getText() !== b[c][0] && b.splice(c + 1, b.length), this.isRedoKey(a) && (this.stopEvent(a), c < b.length - 1 && this.setUndoState(b[++this.undoIndex]))
        }, this.setUndoState = function(a) {
            this.detectUpdatedText(!1, a[0]), this.htmlRich.setSelectionOffsets(a[1]), this.trigger("uiHideAutocomplete")
        }, this.handleKeyDown = function(a) {
            this.isIE && this.selectPrevCharOnBackspace(a), $.browser.mozilla && this.emulateCommandArrow(a), this.emulateUndo(a), this.emulateRedo(a)
        }, this.interceptPaste = function(a) {
            if (a.originalEvent && a.originalEvent.clipboardData) {
                var b = a.originalEvent.clipboardData;
                (this.interceptImagePaste(b) || this.interceptTextPaste(b)) && a.preventDefault()
            }
        }, this.interceptImagePaste = function(a) {
            if (a.items && a.items.length === 1 && a.items[0].kind === "file" && a.items[0].type.indexOf("image/") === 0) {
                var b = new FileReader;
                b.onload = function(a) {
                    var b = a.target.result;
                    this.trigger("uiGotImageData", {
                        name: "pasted",
                        contents: b
                    })
                }.bind(this);
                var c = a.items[0].getAsFile();
                return b.readAsDataURL(c), !0
            }
            return !1
        }, this.interceptTextPaste = function(a) {
            var b = a.getData("text");
            return b && document.execCommand("insertHTML", !1, $("<div>").text(b).html().replace(LINE_FEEDS_REGEX, "<br>"))?!0 : !1
        }, this.clearSelectionOnBlur = function() {
            window.getSelection && (this.previousSelection = this.htmlRich.getSelectionOffsets(), this.previousSelection && getSelection().removeAllRanges())
        }, this.restoreSelectionOnFocus = function() {
            this.previousSelection ? setTimeout(function() {
                this.htmlRich.setSelectionOffsets(this.previousSelection), this.previousSelection = null
            }.bind(this), 0) : this.previousSelection = null
        }, this.setFocusingState = function() {
            this.isFocusing=!0, setTimeout(function() {
                this.isFocusing=!1
            }.bind(this), 0)
        }, this.around("initTextNode", function(a) {
            this.isIE = $.browser.msie || navigator.userAgent.indexOf("Trident")>-1, this.$text = this.select("richSelector"), this.undoHistory = [["", [0]]], this.undoIndex = 0, this.htmlRich = htmlText(this.$text[0], $.browser), this.$text.toggleClass("notie", !this.isIE), this.$normalizer = this.select("normalizerSelector"), this.htmlNormalizer = htmlText(this.$normalizer[0], $.browser);
            var b = navigator.platform;
            this.isMac = b.indexOf("Mac")!=-1, this.isWin = b.indexOf("Win")!=-1, this.on(this.$text, "click", {
                linksSelector: this.stopEvent
            }), this.on(this.$text, "focusin", this.setFocusingState), this.on(this.$text, "keydown", this.handleKeyDown), this.on(this.$text, "cut paste drop focusin", this.ignoreDuringFakeFocus(this.saveUndoStateDeferred)), this.on(this.$text, "paste", this.interceptPaste), this.on(this.$text, "focusout", this.ignoreDuringFakeFocus(this.clearSelectionOnBlur)), this.on(this.$text, "focusin", this.ignoreDuringFakeFocus(this.restoreSelectionOnFocus)), this.on("uiClearUndoState", this.clearUndoState), this.on("uiSaveUndoState", this.saveUndoState), this.detectUpdatedText()
        })
    }
    var tweetHelper = require("app/utils/tweet_helper"), twitterText = require("lib/twitter-text"), htmlText = require("app/utils/html_text");
    module.exports = withRichEditor;
    var INVALID_CHARS_REGEX = /[\uFFFE\uFEFF\uFFFF\u200E\u200F\u202A-\u202E\x00-\x09\x0B\x0C\x0E-\x1F]/g, RENDER_URLS_AS_PRETTY_LINKS = $.browser.mozilla && parseInt($.browser.version, 10) < 2, TRAILING_SINGLE_SPACE_REGEX = / $/, LINE_FEEDS_REGEX = /\r\n|\n\r|\n/g, MAX_OFFSET = Number.MAX_VALUE
});
define("app/ui/tweet_box_manager", ["module", "require", "exports", "core/utils", "app/ui/tweet_box", "app/ui/dm_composer", "app/ui/tweet_box_thumbnails", "app/ui/image_selector", "app/ui/typeahead/typeahead_dropdown", "app/ui/typeahead/typeahead_input", "app/ui/geo_picker", "core/utils", "app/utils/image", "app/data/user_info", "core/component", "app/ui/with_rich_editor"], function(module, require, exports) {
    function tweetBoxManager() {
        this.createTweetBoxAtTarget = function(a, b) {
            this.createTweetBox(a.target, b)
        }, this.createTweetBox = function(a, b) {
            var c = $(a);
            if (!((b.eventData || {}).scribeContext || {}).component)
                throw new Error("Please specify scribing component for tweet box.");
            c.find(".geo-picker").length > 0 && GeoPicker.attachTo(c.find(".geo-picker"), utils.merge(b, {
                parent: c
            }, !0));
            var d = c.find("div.rich-editor").length > 0 ? [withRichEditor]: [], e = (b.dmOnly ? DmComposer : TweetBox).mixin.apply(this, d), f = utils.merge({
                multiPhoto: this.getPhotoLimit() > 1
            }, b);
            e.attachTo(c, f);
            if (c.find(".photo-selector").length > 0) {
                TweetBoxThumbnails.attachTo(c.find(".thumbnail-container"), utils.merge(b, !0));
                var g = this.getPhotoLimit();
                ImageSelector.attachTo(c.find(".photo-selector"), utils.merge({
                    photoLimit: c.hasClass("dm-tweetbox") ? 1: g
                }, b))
            }
            if (b.hasTypeahead || b.hasTypeahead === undefined)
                TypeaheadInput.attachTo(c, utils.merge(b, {
                    inputSelector: "div.rich-editor, textarea.tweet-box",
                    completeAllEntities: this.attr.typeaheadData.hashtags && this.attr.typeaheadData.hashtags.enabled,
                    includeTweetContext: !0,
                    allowAccountsWithoutAtSign: this.attr.typeaheadData.fullNameMatchingInCompose,
                    inputIsTweetbox: !0
                })), TypeaheadDropdown.attachTo(c, utils.merge(b, {
                    inputSelector: "div.rich-editor, textarea.tweet-box",
                    blockLinkActions: !0,
                    includeSearchGlass: !1,
                    parseHashtags: !0,
                    typeaheadSrc: "COMPOSE",
                    datasourceRenders: [["accounts", ["accounts"]], ["topics", ["hashtags"]]],
                    datasourceOptions: {
                        accountsWithoutAtSignLocalOnly: !0,
                        topicsMustStartWithHashtag: !0
                    },
                    deciders: this.attr.typeaheadData
                }))
        }, this.getPhotoLimit = function() {
            return userInfo.getDecider("multi_compose") && image.hasFileReader() && image.hasCanvas() ? 4 : 1
        }, this.after("initialize", function() {
            this.on("uiInitTweetbox", this.createTweetBoxAtTarget)
        })
    }
    var utils = require("core/utils"), TweetBox = require("app/ui/tweet_box"), DmComposer = require("app/ui/dm_composer"), TweetBoxThumbnails = require("app/ui/tweet_box_thumbnails"), ImageSelector = require("app/ui/image_selector"), TypeaheadDropdown = require("app/ui/typeahead/typeahead_dropdown"), TypeaheadInput = require("app/ui/typeahead/typeahead_input"), GeoPicker = require("app/ui/geo_picker"), utils = require("core/utils"), image = require("app/utils/image"), userInfo = require("app/data/user_info"), defineComponent = require("core/component"), withRichEditor = require("app/ui/with_rich_editor"), TweetBoxManager = defineComponent(tweetBoxManager);
    module.exports = TweetBoxManager
});
define("app/data/dm/view_participants", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function viewParticipants() {
        this.fetchParticipants = function(a, b) {
            this.get({
                url: "/i/direct_messages/participants",
                data: {
                    conversation_id: b.conversation_id
                },
                eventData: b,
                success: "dataConversationParticipantsResult",
                error: "dataDMError"
            })
        }, this.after("initialize", function() {
            this.on("uiNeedsConversationParticipants", this.fetchParticipants)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data");
    module.exports = defineComponent(viewParticipants, withData)
});
define("app/boot/direct_messages", ["module", "require", "exports", "app/ui/direct_message_conversation_actions", "app/data/dm/delete_conversation", "app/ui/direct_message_actions", "app/ui/direct_message_compose_with_intent", "app/data/direct_messages", "app/ui/direct_message_dialog", "app/ui/direct_message_empty_state", "app/ui/direct_message_link_handler", "app/ui/direct_message_media_preview", "app/data/direct_messages_scribe", "app/data/dm/dm_settings", "app/data/dm/update_name", "app/ui/dm/direct_message_update_name", "app/ui/dm/suspicious_content", "app/data/dm/suspicious_content_scribe", "app/data/dm/toggle_notifications", "app/ui/dm/tweet_attachment", "app/data/dm/tweet_attachment", "app/ui/tweet_box_manager", "app/ui/select_users", "app/data/dm/view_participants"], function(module, require, exports) {
    function initialize(a) {
        DirectMessageSettings.set({
            maintain_behavior: a.deciders.dm_maintain
        });
        var b = "#dm_dialog", c = {
            scribeContext: {
                component: "dm_existing_conversation_dialog"
            }
        }, d = {
            scribeContext: {
                component: "tweet_box_dm"
            }
        };
        DirectMessageData.attachTo(document, a), DirectMessageScribe.attachTo(document, a), DirectMessageLinkHandler.attachTo(document, a), DirectMessageComposeWithIntent.attachTo(document), DirectMessageDialog.attachTo(b, a, {
            timestampSelector: ".time"
        }), DirectMessageEmptyState.attachTo(b), TweetAttachmentData.attachTo(document), DirectMessageActions.attachTo(b, {
            eventData: c
        }), TweetAttachment.attachTo(b, {
            eventData: c
        }), DirectMessageMediaPreview.attachTo(".dm-media-container"), SuspiciousContentScribe.attachTo("#dm_dialog_conversation"), SuspiciousContent.attachTo("#dm_dialog_conversation"), DeleteConversationData.attachTo(document), DirectMessageUpdateNameData.attachTo(document), ToggleNotificationsData.attachTo(document), ViewParticipantsData.attachTo(document), ConversationActions.attachTo(".dm-conversation-actions"), DirectMessageUpdateNameUI.attachTo("#dm_dialog_conversation .twttr-dialog-header.modal-header"), TweetBoxManager.attachTo(document, a), $("#dm_dialog form.tweet-form").trigger("uiInitTweetbox", {
            defaultText: "",
            condensedText: "",
            hasTypeahead: !1,
            suppressFlashMessage: !0,
            eventData: d,
            dmOnly: !0,
            fileDataString: "media",
            fileInputString: "media"
        }), UserSelection.attachTo("#dm_dialog_new .user-select-container", {
            dropdownSelector: ".dm-participant-container",
            datasourceForTagging: "dmAccounts",
            decidersForTypeahead: a.typeaheadData,
            eventData: d,
            alwaysOpen: !0,
            inputSelector: "input.twttr-directmessage-input",
            maxSelection: 1,
            typeaheadSrc: "COMPOSE_MESSAGE"
        })
    }
    var ConversationActions = require("app/ui/direct_message_conversation_actions"), DeleteConversationData = require("app/data/dm/delete_conversation"), DirectMessageActions = require("app/ui/direct_message_actions"), DirectMessageComposeWithIntent = require("app/ui/direct_message_compose_with_intent"), DirectMessageData = require("app/data/direct_messages"), DirectMessageDialog = require("app/ui/direct_message_dialog"), DirectMessageEmptyState = require("app/ui/direct_message_empty_state"), DirectMessageLinkHandler = require("app/ui/direct_message_link_handler"), DirectMessageMediaPreview = require("app/ui/direct_message_media_preview"), DirectMessageScribe = require("app/data/direct_messages_scribe"), DirectMessageSettings = require("app/data/dm/dm_settings"), DirectMessageUpdateNameData = require("app/data/dm/update_name"), DirectMessageUpdateNameUI = require("app/ui/dm/direct_message_update_name"), SuspiciousContent = require("app/ui/dm/suspicious_content"), SuspiciousContentScribe = require("app/data/dm/suspicious_content_scribe"), ToggleNotificationsData = require("app/data/dm/toggle_notifications"), TweetAttachment = require("app/ui/dm/tweet_attachment"), TweetAttachmentData = require("app/data/dm/tweet_attachment"), TweetBoxManager = require("app/ui/tweet_box_manager"), UserSelection = require("app/ui/select_users"), ViewParticipantsData = require("app/data/dm/view_participants"), hasDialog=!!$("#dm_dialog").length;
    module.exports = hasDialog ? initialize : $.noop
});
define("app/data/dm_poll", ["module", "require", "exports", "core/component", "app/data/with_data", "app/data/with_auth_token", "app/data/notifications"], function(module, require, exports) {
    function dmPoll() {
        this.defaultAttrs({
            noShowError: !0
        }), this.dispatch = function(a) {
            a && (notifications.updateNotificationState(a.note), this.trigger("dataNotificationsReceived", a.note))
        }, this.makeRequest = function(a, b, c) {
            this.get({
                url: "/i/toast_poll",
                data: c,
                eventData: b,
                success: this.dispatch.bind(this)
            })
        }, this.requestConversationList = function(a, b) {
            var c = {};
            notifications.addDMData(c), this.makeRequest(a, b, c)
        }, this.after("initialize", function() {
            this.on("uiDMPoll", this.requestConversationList)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), withAuthToken = require("app/data/with_auth_token"), notifications = require("app/data/notifications"), DMPoll = defineComponent(dmPoll, withData, withAuthToken);
    module.exports = DMPoll
});
define("app/ui/with_drag_events", ["module", "require", "exports", "app/utils/drag_drop_helper"], function(module, require, exports) {
    function withDragEvents() {
        this.defaultAttrs({
            fileTypeRegExp: !1
        }), this.childHover = function(a) {
            $.contains(this.$node.get(0), a.target) && (a.stopImmediatePropagation(), this.inChild = a.type === "dragenter")
        }, this.hover = function(a) {
            a.preventDefault();
            if (this.inChild)
                return !1;
            var b = this.prepareData(a);
            this.trigger(a.type === "dragenter" ? "uiDragEnter" : "uiDragLeave", b)
        }, this.finish = function(a) {
            a.stopImmediatePropagation(), this.inChild=!1, this.trigger("uiDragLeave")
        }, this.preventDefault = function(a) {
            return a.preventDefault(), !1
        }, this.detectDragEnd = function(a) {
            this.detectingEnd || (this.detectingEnd=!0, $(document.body).one("mousemove", this.dragEnd.bind(this)))
        }, this.dragEnd = function() {
            this.detectingEnd=!1, this.trigger("uiDragEnd")
        }, this.outOfBounds = function(a) {
            var b = a.originalEvent.pageX, c = a.originalEvent.pageY, d = document.body.clientWidth, e = document.body.clientHeight;
            (b <= 0 || c <= 0 || c >= e || b >= d) && this.dragEnd()
        }, this.prepareData = function(a) {
            var b = (a.originalEvent || a).dataTransfer, c = {};
            c.types = Array.prototype.slice.call(b.types, 0);
            var d = b.items || b.files;
            if (!d)
                c.items = null;
            else {
                c.items = [];
                for (var e = 0; e < d.length; ++e) {
                    var f = d[e];
                    c.items.push({
                        type: f.type
                    })
                }
            }
            return {
                dataTransfer: c
            }
        }, this.after("initialize", function() {
            this.inChild=!1, this.on(document.body, "dragenter dragover", this.detectDragEnd), this.on(document.body, "dragleave", this.outOfBounds), this.on("dragenter dragleave", dragDropHelper.onlyHandleEventsWithFiles(this.hover, this.attr.fileTypeRegExp)), this.on("dragover drop", dragDropHelper.onlyHandleEventsWithFiles(this.preventDefault, this.attr.fileTypeRegExp)), this.on("dragenter dragleave", dragDropHelper.onlyHandleEventsWithFiles(this.childHover, this.attr.fileTypeRegExp)), this.on(document, "uiDragEnd drop", this.finish)
        })
    }
    var dragDropHelper = require("app/utils/drag_drop_helper");
    module.exports = withDragEvents
});
define("app/ui/drag_state", ["module", "require", "exports", "core/component", "app/ui/with_drag_events"], function(module, require, exports) {
    function dragState() {
        this.defaultAttrs({
            draggingClass: "currently-dragging",
            supportsDraggingClass: "supports-drag-and-drop"
        }), this.dragEnter = function() {
            this.$node.addClass(this.attr.draggingClass)
        }, this.dragLeave = function() {
            this.$node.removeClass(this.attr.draggingClass)
        }, this.addSupportsDraggingClass = function() {
            this.$node.addClass(this.attr.supportsDraggingClass)
        }, this.hasSupport = function() {
            return "draggable"in document.createElement("span")&&!$.browser.msie
        }, this.after("initialize", function() {
            this.hasSupport() && (this.addSupportsDraggingClass(), this.on("uiDragEnter", this.dragEnter), this.on("uiDragLeave uiDrop", this.dragLeave))
        })
    }
    var defineComponent = require("core/component"), withDragEvents = require("app/ui/with_drag_events");
    module.exports = defineComponent(dragState, withDragEvents)
});
define("app/ui/banners/email_banner", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function emailBanner() {
        this.defaultAttrs({
            emailSettingsLinkSelector: ".email-settings-link",
            phoneSettingsLinkSelector: ".phone-settings-link",
            resendConfirmationEmailLinkSelector: ".resend-confirmation-email-link",
            resetBounceLinkSelector: ".reset-bounce-link"
        }), this.emailSettingsLink = function() {
            this.trigger("uiClickEmailSettingsLink")
        }, this.phoneSettingsLink = function() {
            this.trigger("uiClickPhoneSettingsLink")
        }, this.resendConfirmationEmail = function(a, b) {
            this.trigger("uiResendConfirmationEmail", {
                unsuspend: $(a.target).hasClass("unsuspend-confirmation")
            })
        }, this.resetBounceLink = function() {
            this.trigger("uiResetBounceLink")
        }, this.after("initialize", function() {
            this.on("click", {
                emailSettingsLinkSelector: this.emailSettingsLink,
                phoneSettingsLinkSelector: this.phoneSettingsLink,
                resendConfirmationEmailLinkSelector: this.resendConfirmationEmail,
                resetBounceLinkSelector: this.resetBounceLink
            })
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(emailBanner)
});
define("app/data/email_banner", ["module", "require", "exports", "core/component", "app/data/with_data", "core/i18n"], function(module, require, exports) {
    function emailBannerData() {
        this.resendConfirmationEmail = function(a, b) {
            var c = function(a) {
                this.trigger("uiShowMessage", {
                    message: a.messageForFlash
                })
            }, d = function() {
                this.trigger("uiShowMessage", {
                    message: _('Oops!  There was an error sending the confirmation email.')
                })
            };
            this.post({
                url: "/account/resend_confirmation_email",
                eventData: null,
                data: b && b.unsuspend ? {
                    unsuspend: !0
                }
                : {},
                success: c.bind(this),
                error: d.bind(this)
            })
        }, this.resetBounceScore = function() {
            var a = function() {
                this.trigger("uiShowMessage", {
                    message: _('Your email notifications should resume shortly.')
                })
            }, b = function() {
                this.trigger("uiShowMessage", {
                    message: _('Oops! There was an error sending email notifications.')
                })
            };
            this.post({
                url: "/bouncers/reset",
                eventData: null,
                data: null,
                success: a.bind(this),
                error: b.bind(this)
            })
        }, this.after("initialize", function() {
            this.on("uiResendConfirmationEmail", this.resendConfirmationEmail), this.on("uiResetBounceLink", this.resetBounceScore)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), _ = require("core/i18n");
    module.exports = defineComponent(emailBannerData, withData)
});
define("app/data/email_banner_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe", "core/utils"], function(module, require, exports) {
    function emailBannerScribe() {
        this.scribeEmailSettingsClick = function(a, b) {
            this.scribe({
                page: "self_unsuspend",
                section: "banner",
                component: "email_verification",
                element: "settings",
                action: "click"
            })
        }, this.scribePhoneSettingsClick = function(a, b) {
            this.scribe({
                page: "self_unsuspend",
                section: "banner",
                component: "phone_verification",
                action: "click"
            })
        }, this.scribeResendConfirmationEmailClick = function(a, b) {
            this.scribe({
                page: "self_unsuspend",
                section: "banner",
                component: "email_verification",
                element: "resend_confirmation_email",
                action: "click"
            })
        }, this.after("initialize", function() {
            this.on("uiClickEmailSettingsLink", this.scribeEmailSettingsClick), this.on("uiClickPhoneSettingsLink", this.scribePhoneSettingsClick), this.on("uiResendConfirmationEmail", this.scribeResendConfirmationEmailClick)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), utils = require("core/utils");
    module.exports = defineComponent(emailBannerScribe, withScribe)
});
define("app/data/embed_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe", "app/data/with_interaction_data_scribe", "core/utils"], function(module, require, exports) {
    function embedScribe() {
        this.scribeOpen = function(a, b) {
            this.scribeEmbedAction("open", b)
        }, this.scribeOembedError = function(a, b) {
            this.scribeEmbedAction("request_failed", b)
        }, this.scribeEmbedCopy = function(a, b) {
            this.scribeEmbedAction("copy", b)
        }, this.scribeEmbedError = function(a, b) {
            this.scribeEmbedAction("embed_request_failed")
        }, this.scribeEmbedAction = function(a, b) {
            this.scribeInteraction(a, utils.merge(b, {
                scribeContext: {
                    component: "embed_tweet_dialog"
                }
            }))
        }, this.after("initialize", function() {
            this.on("uiEmbedRequestFailed", this.scribeEmbedError), this.on("uiNeedsEmbedTweetDialog", this.scribeOpen), this.on("uiOembedError", this.scribeOembedError), this.on("uiUserCopiedEmbedCode", this.scribeEmbedCopy)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), withInteractionScribe = require("app/data/with_interaction_data_scribe"), utils = require("core/utils");
    module.exports = defineComponent(embedScribe, withScribe, withInteractionScribe)
});
define("app/data/with_widgets", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function withWidgets() {
        this.widgetsAreLoaded = function() {
            return !!this.widgets&&!!this.widgets.init
        }, this.getWidgets = function() {
            window.twttr || this.asyncWidgetsLoader(), this.widgets = window.twttr, window.twttr.ready(this._widgetsReady.bind(this))
        }, this._widgetsReady = function(_) {
            this.widgetsReady && this.widgetsReady()
        }, this.asyncWidgetsLoader = function() {
            window.twttr = function(a, b, c) {
                var d, e, f = a.getElementsByTagName(b)[0];
                if (a.getElementById(c))
                    return;
                return e = a.createElement(b), e.id = c, e.src = "//platform.twitter.com/widgets.js", f.parentNode.insertBefore(e, f), window.twttr || (d = {
                    _e: [],
                    ready: function(a) {
                        d._e.push(a)
                    }
                })
            }(document, "script", "twitter-wjs")
        }
    }
    var defineComponent = require("core/component"), WithWidgets = defineComponent(withWidgets);
    module.exports = withWidgets
});
define("app/ui/dialogs/embed_tweet", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/data/with_card_metadata", "app/data/with_widgets"], function(module, require, exports) {
    function embedTweetDialog() {
        this.defaultAttrs({
            dialogSelector: "#embed-tweet-dialog",
            dialogContentSelector: "#embed-tweet-dialog .modal-content",
            previewContainerSelector: ".embed-preview",
            embedFrameSelector: ".embed-preview iframe",
            visibleEmbedFrameSelector: ".embed-preview iframe:visible",
            embedCodeDestinationSelector: ".embed-destination",
            triggerSelector: ".js-embed-tweet",
            overlaySelector: ".embed-overlay",
            spinnerOverlaySelector: ".embed-overlay-spinner",
            errorOverlaySelector: ".embed-overlay-error",
            tryAgainSelector: ".embed-overlay-error .retry-embed",
            includeParentTweetContainerSelector: ".embed-include-parent-tweet",
            includeParentTweetSelector: ".include-parent-tweet",
            includeCardContainerSelector: ".embed-include-card",
            includeCardSelector: ".include-card",
            embedWidth: "469px",
            top: "90px"
        }), this.cacheKeyForOptions = function(a) {
            return JSON.stringify(a.data) + a.tweetId
        }, this.cacheKeyChanged = function(a) {
            var b = this.cacheKeyForOptions(a);
            return b != this.cacheKeyForOptions(this.getOptions())
        }, this.didReceiveEmbedCode = function(a, b) {
            if (this.cacheKeyChanged(b.options))
                return;
            this.select("overlaySelector").hide(), this.$embedCodeDestination.val(b.data.html), this.trigger(this.$dialog, "uiDialogContentChanged"), this.selectEmbedCode()
        }, this.retryEmbedCode = function(a, b) {
            if (this.cacheKeyChanged(b))
                return;
            this.select("overlaySelector").hide(), this.select("spinnerOverlaySelector").show(), this.trigger("uiOembedError", this.tweetData)
        }, this.failedToReceiveEmbedCode = function(a, b) {
            if (this.cacheKeyChanged(b))
                return;
            this.select("overlaySelector").hide(), this.select("embedCodeDestinationSelector").hide(), this.select("errorOverlaySelector").show(), this.clearOembed(), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.updateEmbedCode = function() {
            this.select("embedCodeDestinationSelector").show(), this.select("overlaySelector").hide(), this.trigger("uiNeedsOembed", this.getOptions()), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.requestTweetEmbed = function() {
            if (!this.widgetsAreLoaded())
                return;
            var a = this.getOptions(), b = this.cacheKeyForOptions(a);
            if (this.cachedTweetEmbeds[b]) {
                this.displayCachedTweetEmbed(b);
                return 
            }
            this.clearTweetEmbed(), this.widgets.widgets.createTweet(this.tweetId(), this.select("previewContainerSelector")[0], this.receivedTweetEmbed.bind(this, b), {
                width: this.attr.embedWidth,
                conversation: a.data.hide_thread ? "none": "all",
                cards: a.data.hide_media ? "hidden": "shown"
            })
        }, this.clearTweetEmbed = function() {
            var a = this.select("visibleEmbedFrameSelector");
            this.stopPlayer(), a.hide()
        }, this.clearOembed = function() {
            this.$embedCodeDestination.val("")
        }, this.tearDown = function() {
            this.stopPlayer(), this.clearTweetEmbed(), this.clearOembed()
        }, this.stopPlayer = function() {
            var a = this.select("embedFrameSelector");
            a.each(function(a, b) {
                var c = $(b.contentWindow.document), d = c.find("div.media iframe")[0], e;
                if (!d ||!d.src || d.src == document.location.href)
                    return;
                e = d.src, d.setAttribute("src", ""), d.setAttribute("src", e)
            })
        }, this.displayCachedTweetEmbed = function(a) {
            this.clearTweetEmbed(), $(this.cachedTweetEmbeds[a]).show()
        }, this.receivedTweetEmbed = function(a, b) {
            b ? this.cachedTweetEmbeds[a] = b : this.trigger("uiEmbedRequestFailed")
        }, this.embedCodeCopied = function(a) {
            this.trigger("uiUserCopiedEmbedCode")
        }, this.includeParentTweet = function() {
            return this.$includeParentTweet.attr("checked") == "checked"
        }, this.showCard = function() {
            return this.$includeCard.attr("checked") == "checked"
        }, this.getOptions = function() {
            return {
                data: {
                    lang: this.lang,
                    hide_thread: !this.includeParentTweet(),
                    hide_media: !this.showCard()
                },
                retry: !0,
                tweetId: this.tweetId(),
                screenName: this.screenName()
            }
        }, this.selectEmbedCode = function() {
            this.$embedCode.select()
        }, this.setUpDialog = function(a, b) {
            this.position(), this.eventData = a, this.tweetData = b, this.toggleIncludeParent(), this.toggleShowCard(), this.resetIncludeParent(), this.resetShowCard(), this.updateEmbedCode(), this.requestTweetEmbed(), this.open(), this.fixPosition(), this.widgetsAreLoaded() || this.getWidgets()
        }, this.fixPosition = function() {
            this.$dialog.css({
                position: "relative",
                top: this.attr.top
            })
        }, this.resetIncludeParent = function() {
            var a = this.cacheKeyForOptions(this.getOptions());
            if (this.cachedTweetEmbeds[a])
                return;
            this.$includeParentTweet.attr("checked", "CHECKED")
        }, this.resetShowCard = function() {
            var a = this.cacheKeyForOptions(this.getOptions());
            if (this.cachedTweetEmbeds[a])
                return;
            this.$includeCard.attr("checked", "CHECKED")
        }, this.toggleIncludeParent = function() {
            this.tweetHasParent() ? this.$includeParentCheckboxContainer.show() : this.$includeParentCheckboxContainer.hide()
        }, this.toggleShowCard = function() {
            this.tweetHasCard() ? this.$includeCardCheckboxContainer.show() : this.$includeCardCheckboxContainer.hide()
        }, this.tweetId = function() {
            return this.tweetData.tweetId
        }, this.tweetHasParent = function() {
            return this.tweetData.hasParentTweet
        }, this.tweetHasCard = function() {
            return this.getCardDataFromTweet($(this.eventData.target)).tweetHasCard
        }, this.screenName = function() {
            return this.tweetData.screenName
        }, this.widgetsReady = function() {
            this.$dialogContainer && this.isOpen() && this.requestTweetEmbed()
        }, this.onOptionChange = function() {
            this.trigger("uiNeedsOembed", this.getOptions()), this.requestTweetEmbed()
        }, this.restoreFocusToTweet = function(a) {
            $(a.target).is(this.$dialog) && this.activeEl && this.trigger($(this.activeEl).closest(".tweet"), "uiShouldAddFocusStyle")
        }, this.after("initialize", function() {
            this.$includeParentTweet = this.select("includeParentTweetSelector"), this.$embedCodeDestination = this.select("embedCodeDestinationSelector"), this.$includeParentCheckboxContainer = this.select("includeParentTweetContainerSelector"), this.$includeCard = this.select("includeCardSelector"), this.$includeCardCheckboxContainer = this.select("includeCardContainerSelector"), this.$embedCode = this.select("embedCodeDestinationSelector"), this.on(document, "uiNeedsEmbedTweetDialog", this.setUpDialog), this.on("uiDialogCloseRequested", this.tearDown), this.on(document, "uiBeforePageChanged", this.tearDown), this.on(document, "uiDialogRestorePreviousFocus", this.restoreFocusToTweet), this.on(document, "dataOembedSuccess", this.didReceiveEmbedCode), this.on(document, "dataOembedError", this.failedToReceiveEmbedCode), this.on(document, "dataOembedRetry", this.retryEmbedCode), this.on(this.$embedCodeDestination, "copy cut", this.embedCodeCopied), this.on(this.$embedCode, "click", this.selectEmbedCode), this.on("click", {
                tryAgainSelector: this.updateEmbedCode
            }), this.on("change", {
                includeParentTweetSelector: this.onOptionChange,
                includeCardSelector: this.onOptionChange
            }), this.lang = document.documentElement.getAttribute("lang"), this.cachedTweetEmbeds = {}
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withCardMetadata = require("app/data/with_card_metadata"), withWidgets = require("app/data/with_widgets"), EmbedTweetDialog = defineComponent(embedTweetDialog, withDialog, withCardMetadata, withWidgets);
    module.exports = EmbedTweetDialog
});
define("app/data/feedback/feedback", ["module", "require", "exports", "core/component", "app/utils/cookie", "app/data/with_data"], function(module, require, exports) {
    function feedbackData() {
        this.defaultAttrs({
            feedbackCookie: "debug_data",
            data: {}
        }), this.getFeedbackData = function(a, b) {
            this.trigger("dataFeedback", this.attr.data)
        }, this.toggleFeedbackCookie = function(a, b) {
            var c = b.enabled?!0 : null;
            cookie(this.attr.feedbackCookie, c), this.refreshPage(), this.checkDebugEnabled()
        }, this.refreshPage = function() {
            document.location.reload(!0)
        }, this.checkDebugEnabled = function() {
            this.trigger("dataDebugFeedbackChanged", {
                enabled: !!cookie(this.attr.feedbackCookie)
            })
        }, this.addFeedbackData = function(a, b) {
            var c = this.attr.data;
            for (var d in b)
                c[d] ? c[d] = [].concat.apply(c[d], b[d]) : c[d] = b[d]
        }, this.logNavigation = function(a, b) {
            this.addFeedbackData(a, {
                pushState: [{
                    data: {
                        href: b.href,
                        module: b.module,
                        title: b.title
                    }
                }
                ]
            }), b.init_data && b.init_data.debugData && this.addFeedbackData(a, b.init_data.debugData.data || {})
        }, this.after("initialize", function() {
            this.data = this.attr.data || {}, this.on("uiNeedsFeedbackData", this.getFeedbackData), this.on("uiToggleDebugFeedback", this.toggleFeedbackCookie), this.on("dataSetDebugData", this.addFeedbackData), this.on("uiPageChanged", this.logNavigation), this.checkDebugEnabled()
        })
    }
    var defineComponent = require("core/component"), cookie = require("app/utils/cookie"), withData = require("app/data/with_data");
    module.exports = defineComponent(feedbackData, withData)
});
define("app/ui/feedback/feedback_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog"], function(module, require, exports) {
    function feedbackDialog() {
        this.defaultAttrs({
            pastedContentSelector: ".feedback-json-output",
            debugKeySelectorSelector: "select[name=debug-key]",
            debugDataText: "",
            dialogToggleSelector: ".feedback-dialog .feedback-data-toggle"
        }), this.toggleDebugEnabled = function(a, b) {
            this.trigger("uiToggleDebugFeedback", {
                enabled: $(b.el).is(".off")
            })
        }, this.prepareDialog = function(a, b) {
            this.trigger("uiNeedsFeedbackData")
        }, this.openDialog = function(a, b) {
            this.debugData = b, this.refreshAvailableDataKeys(), this.open()
        }, this.refreshAvailableDataKeys = function() {
            var a = this.select("debugKeySelectorSelector");
            a.empty(), Object.keys(this.debugData).forEach(function(b) {
                var c = $("<option>" + b + "</option>");
                a.append(c)
            }), this.refreshDebugJSON()
        }, this.refreshDebugJSON = function(a) {
            var b = this.select("debugKeySelectorSelector").val();
            if (!b ||!this.debugData[b])
                return;
            var c = this.debugData[b].map(function(a) {
                return a.data
            });
            this.select("pastedContentSelector").html(JSON.stringify(c, undefined, 2))
        }, this.after("initialize", function() {
            this.on(document, "dataFeedback", this.openDialog), this.on(document, "uiPrepareFeedbackDialog", this.prepareDialog), this.on("change", {
                debugKeySelectorSelector: this.refreshDebugJSON
            }), this.on("click", {
                dialogToggleSelector: this.toggleDebugEnabled
            })
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog");
    module.exports = defineComponent(feedbackDialog, withDialog)
});
define("app/utils/image/image_loader", ["module", "require", "exports"], function(module, require, exports) {
    var imageLoader = {
        load: function(a, b, c) {
            function f(a) {
                e=!0, d.unbind("load"), c()
            }
            var d = $("<img/>"), e=!1;
            d.on("load", function(a) {
                !e && a.type != "error" && b(d)
            }), d.on("error", f), d.on("abort", f), d.attr("src", a)
        }
    };
    module.exports = imageLoader
});
define("app/ui/media/with_flag_action", ["module", "require", "exports"], function(module, require, exports) {
    module.exports = function() {
        this.defaultAttrs({
            flagContainerSelector: ".flag-container",
            flaggableSelector: ".flaggable",
            flaggedSelector: ".flagged",
            tweetWithIdSelector: ".tweet[data-tweet-id]"
        }), this.flagMedia = function(a) {
            var b = $(a.target).closest(this.attr.flagContainerSelector), c = b.find(this.attr.flaggableSelector);
            if (!c.hasClass("hidden")) {
                var d = b.closest(this.attr.tweetWithIdSelector);
                d.attr("data-possibly-sensitive") ? this.trigger("uiFlagConfirmation", {
                    id: d.attr("data-tweet-id")
                }) : (this.trigger("uiFlagMedia", {
                    id: d.attr("data-tweet-id")
                }), b.find(this.attr.flaggableSelector).addClass("hidden"), b.find(this.attr.flaggedSelector).removeClass("hidden"))
            }
        }, this.after("initialize", function() {
            this.on("click", {
                flagContainerSelector: this.flagMedia
            })
        })
    }
});
define("app/ui/with_tweet_translation", ["module", "require", "exports", "core/compose", "app/utils/tweet_helper", "app/ui/with_interaction_data", "app/utils/rtl_text", "core/i18n"], function(module, require, exports) {
    function withTweetTranslation() {
        compose.mixin(this, [withInteractionData]), this.defaultAttrs({
            tweetSelector: "div.tweet",
            tweetTranslationSelector: ".tweet-translation",
            tweetTranslationTextSelector: ".tweet-translation-text",
            translateTweetSelector: ".js-translate-tweet",
            translateAndExpandTweetSelector: ".js-translate-tweet.expand-stream-item",
            autoTranslatedTweetSelector: "div.auto-translatable-tweet",
            needsTranslationClass: "needs-translation",
            translateLabelSelector: ".translate-label",
            permalinkContainerSelector: ".permalink-tweet-container",
            permalinkClass: ".permalink-tweet",
            languageSelector: ".tweet-language",
            galleryTranslateButtonSelector: ".Gallery .GalleryTweet .tweet .translate-button"
        }), this.translateTweet = function(a) {
            var b = this.tweetFromEvent(a);
            if (b.find(this.attr.tweetTranslationSelector).hasClass(this.attr.needsTranslationClass)) {
                var c = this.interactionData(b);
                c.dest = b.find(this.attr.tweetTranslationSelector).data("destLang"), this.trigger(b, "uiNeedsTweetTranslation", c)
            }
        }, this.expandTweet = function(a) {
            this.trigger(this.tweetFromEvent(a), "uiExpandTweet")
        }, this.autoTranslationEnabled = function() {
            return !!$(this.attr.autoTranslatedTweetSelector).length
        }, this.tweetFromEvent = function(a) {
            return $(a.target).closest(this.attr.tweetSelector)
        }, this.showTweetTranslation = function(a, b) {
            var c, d;
            if (b.item_html) {
                d = this.findTweetById($(a.target), b.id_str), c = d.find(this.attr.tweetTranslationSelector);
                if (!b.translated_lang) {
                    d.find(this.attr.translateTweetSelector).hide();
                    return 
                }
                c.find(this.attr.languageSelector).html(b.translated_lang), c.find(this.attr.tweetTranslationTextSelector).html(b.item_html), c.removeClass(this.attr.needsTranslationClass), d.find(this.attr.translateTweetSelector).toArray().forEach(function(a) {
                    var b = $(a);
                    (!b.is(this.attr.translateAndExpandTweetSelector) || b.is(this.attr.galleryTranslateButtonSelector)) && b.hide()
                }, this)
            }
        }, this.findTweetById = function(a, b) {
            var c = this.tweetSelectorById(b);
            return a.is(c) ? a : a.find(c)
        }, this.tweetSelectorById = function(a) {
            return this.attr.tweetSelector + "[data-tweet-id=" + a.replace(/\D/g, "") + "]"
        }, this.showError = function(a, b) {
            this.trigger("uiShowMessage", {
                message: _('Unable to translate this Tweet. Please try again later.')
            })
        }, this.after("initialize", function() {
            this.on("dataTweetTranslationSuccess", this.showTweetTranslation), this.on("dataTweetTranslationError", this.showError);
            var a = {
                translateAndExpandTweetSelector: this.expandTweet
            };
            this.autoTranslationEnabled() ? this.on("uiTimelineNeedsTranslation", this.translateTweet) : a.translateTweetSelector = this.translateTweet, this.on("click", a)
        })
    }
    var compose = require("core/compose"), tweetHelper = require("app/utils/tweet_helper"), withInteractionData = require("app/ui/with_interaction_data"), RTLText = require("app/utils/rtl_text"), _ = require("core/i18n");
    module.exports = withTweetTranslation
});
define("app/ui/gallery/gallery", ["module", "require", "exports", "core/component", "core/utils", "core/i18n", "app/utils/image/image_loader", "app/utils/image/image_resizer", "app/ui/media/with_flag_action", "app/ui/with_item_actions", "app/ui/with_tweet_translation", "app/ui/with_scrollbar_width"], function(module, require, exports) {
    function gallery() {
        this.tweetHtml = {}, this.defaultAttrs({
            profileUser: !1,
            defaultGalleryTitle: _('Media Gallery'),
            mediaSelector: ".media-thumbnail",
            galleryParentSelector: ".Gallery",
            galleryMediaSelector: ".Gallery-media",
            galleryTweetSelector: ".GalleryTweet",
            closeSelector: ".js-close, .Gallery-closeTarget",
            gridSelector: ".grid-action",
            gallerySelector: ".Gallery-content",
            galleryTitleSelector: ".modal-title",
            imageSelector: ".media-image",
            navSelector: ".GalleryNav",
            prevSelector: ".GalleryNav--prev",
            nextSelector: ".GalleryNav--next",
            itemType: "tweet",
            modalHeaderSelector: ".modal-header",
            tweetlessGalleryClass: "is-tweetless",
            inOverlayGalleryClass: "Gallery--inOverlay"
        }), this.isOpen = function() {
            return this.$node.is(":visible")
        }, this.open = function(a, b) {
            this.calculateScrollbarWidth(), this.fromGrid = b&&!!b.fromGrid, this.title = b && b.title ? b.title : this.attr.defaultGalleryTitle, this.select("galleryTitleSelector").text(this.title), b && b.inOverlay ? this.$node.addClass(this.attr.inOverlayGalleryClass) : this.$node.removeClass(this.attr.inOverlayGalleryClass), b && b.showGrid && b.profileUser ? (this.select("gallerySelector").removeClass("no-grid"), this.select("gridSelector").attr("href", "/" + b.profileUser.screen_name + "/media"), this.select("gridSelector").find(".visuallyhidden").text(b.profileUser.name), this.select("gridSelector").addClass("js-nav")) : (this.select("gallerySelector").addClass("no-grid"), this.select("gridSelector").removeClass("js-nav"));
            var c = $(a.target).closest(this.attr.mediaSelector);
            if (this.isOpen() || c.length == 0)
                return;
            b && b.timelineSelector ? this.$timeline = c.closest(b.timelineSelector) : this.$timeline = c.parent(), imageResizer.resetMinSize(this.select("gallerySelector")), this.render(c), $("body").addClass("gallery-enabled"), this.select("gallerySelector").addClass("show-controls"), this.on("mousemove", function() {
                this.select("gallerySelector").removeClass("show-controls")
            }.bind(this)), this.trigger("uiGalleryOpened")
        }, this.handleClose = function(a) {
            if (!this.isOpen())
                return;
            this.fromGrid ? this.returnToGrid(!0) : this.closeGallery()
        }, this.returnToGrid = function(a) {
            this.trigger(this.$current, "uiOpenGrid", {
                title: this.title,
                fromGallery: a
            }), this.closeGallery()
        }, this.closeGallery = function() {
            $("body").hasClass("gallery-enabled") && ($("body").removeClass("gallery-enabled"), this.select("galleryMediaSelector").empty(), this.hideNav(), this.enableNav(!1, !1), this.off(window, "resize", this.debouncedResize), this.off("mousemove"), delete this.$timeline, delete this.$current, this.trigger("uiGalleryClosed"), this.trigger("uiDialogClosed"))
        }, this.render = function(a) {
            this.clearTweet(), this.$current = a, this.renderNav(), this.trigger(a, "uiGalleryMediaLoad"), this.renderMedia(!0, a)
        }, this.renderNav = function() {
            if (!this.$current)
                return;
            var a, b, c, d;
            a = this.$timeline.find(this.attr.mediaSelector), d = a.index(this.$current), b = a.slice(0, d), c = a.slice(d + 1);
            var e = c.length > 0, f = b.length > 0;
            this.enableNav(e, f), e || f ? this.showNav() : this.hideNav()
        }, this.preloadNeighbors = function(a) {
            this.preloadRecursive(a, "next", 2), this.preloadRecursive(a, "prev", 2)
        }, this.clearTweet = function() {
            this.select("galleryTweetSelector").empty()
        }, this.getTweet = function(a) {
            if (!a)
                return;
            this.tweetHtml[a] ? this.renderTweet(a, this.tweetHtml[a]) : (this.$current && this.getTweetId(this.$current) == a && this.select("galleryMediaSelector").find(".media-image,iframe").hide(), this.trigger("uiGetTweet", {
                id: a,
                modal: "gallery"
            }))
        }, this.gotTweet = function(a, b) {
            b.id && b.tweet_html && (this.tweetHtml[b.id] = b.tweet_html, this.renderTweet(b.id, b.tweet_html))
        }, this.renderTweet = function(a, b) {
            if (this.$current && this.getTweetId(this.$current) == a) {
                var c = this.select("galleryTweetSelector"), d = this.select("gallerySelector"), e = this.select("galleryMediaSelector").find(".media-image,iframe");
                c.html(b), this.trigger(c.find(".tweet"), "uiTimelineNeedsTranslation"), e.length && (e.show(), imageResizer.resizeMedia(e, d, !1, this.attr.galleryTweetSelector))
            }
        }, this.getTweetId = function(a) {
            return a.attr("data-status-id") ? a.attr("data-status-id") : a.closest("[data-tweet-id]").attr("data-tweet-id")
        }, this.getAllInDir = function(a, b) {
            var c, d = b == "prev"?-1 : 1, e = this.$timeline.find(this.attr.mediaSelector), f = e.index(a) + d;
            return f>-1 && (c = e.slice(f)), c || $()
        }, this.getInDir = function(a, b) {
            return this.getAllInDir(a, b).first()
        }, this.preloadRecursive = function(a, b, c) {
            if (c == 0)
                return;
            if (this.$timeline.find(this.attr.mediaSelector).length == 1)
                return;
            if (b == "next" && this.getAllInDir(a, b).length < c&&!this.$timeline.hasClass("dashboard")) {
                this.trigger("uiNearTheBottom");
                return 
            }
            var d = this.getInDir(a, b);
            if (!d ||!d.length)
                return;
            d.attr("data-preloading", !0);
            var e = function(a) {
                d.attr("data-preloaded", !0), this.getTweet(this.getTweetId(d)), this.preloadRecursive(d, b, --c)
            }.bind(this), f = function() {
                d.remove(), this.preloadRecursive(d, b, c)
            }.bind(this);
            imageLoader.load(d.attr("data-resolved-url-large"), e, f)
        }, this.renderMedia = function(a, b) {
            a ? (b.attr("data-source-url") ? this.loadVideo(b) : this.loadImage(b), this.preloadNeighbors(b)) : (b.remove(), this.next())
        }, this.loadImage = function(a) {
            var b = $('<img class="media-image"/>');
            b.on("load", function(c) {
                var d = this.getTweetId(a), e = this.select("gallerySelector");
                a.attr("loaded", !0), this.select("galleryMediaSelector").empty().append(b), b[0].height == 0 ? b.height(a.data("img-height")).width(a.data("img-width")) : (a.data("img-height", b[0].height), a.data("img-width", b[0].width)), b.attr({
                    "data-height": b[0].height,
                    "data-width": b[0].width
                }), imageResizer.resizeMedia(b, e, !1, this.attr.galleryTweetSelector, !0), this.$current = a, d ? (this.getTweet(d), e.closest(this.attr.galleryParentSelector).removeClass(this.attr.tweetlessGalleryClass)) : e.closest(this.attr.galleryParentSelector).addClass(this.attr.tweetlessGalleryClass), this.trigger("uiGalleryMediaLoaded", {
                    url: b.attr("src"),
                    id: a.attr("data-status-id")
                })
            }.bind(this)), b.on("error", function(c) {
                this.trigger("uiGalleryMediaFailed", {
                    url: b.attr("src"),
                    id: a.attr("data-status-id")
                }), a.remove(), this.next()
            }.bind(this)), b.attr("src", a.attr("data-resolved-url-large")), this.select("gallerySelector").removeClass("video")
        }, this.fixSnappyUrl = function(a) {
            var b = a.attr("data-source-url"), c = /https:\/\/www.snappytv.com\/tc\/(\d+)\?autoplay=true&h=(\d+)&w=(\d+)/;
            if (!c.test(b))
                return;
            var d = a.attr("data-height") * 2, e = a.attr("data-width") * 2;
            b = b.replace(c, "https://www.snappytv.com/tc/$1?autoplay=true&h=" + d + "&w=" + e), a.attr("data-source-url", b)
        }, this.loadVideo = function(a) {
            var b = $("<iframe>");
            this.fixSnappyUrl(a), b.height(a.attr("data-height") * 2).width(a.attr("data-width") * 2).attr("data-height", a.attr("data-height") * 2).attr("data-width", a.attr("data-width") * 2).attr("src", a.attr("data-source-url")), a.attr("loaded", !0), this.select("galleryMediaSelector").empty().append(b), this.$current = a, this.getTweet(a.attr("data-status-id")), this.select("gallerySelector").addClass("video")
        }, this.prev = function() {
            this.gotoMedia("prev"), this.trigger("uiGalleryNavigatePrev")
        }, this.next = function() {
            this.gotoMedia("next"), this.trigger("uiGalleryNavigateNext")
        }, this.gotoMedia = function(a) {
            if (this.$current) {
                var b = this.getInDir(this.$current, a);
                b.length && this.render(b)
            }
        }, this.showNav = function() {
            this.select("navSelector").show()
        }, this.hideNav = function() {
            this.select("navSelector").hide()
        }, this.enableNav = function(a, b) {
            a ? this.select("nextSelector").addClass("enabled") : this.select("nextSelector").removeClass("enabled"), b ? this.select("prevSelector").addClass("enabled") : this.select("prevSelector").removeClass("enabled")
        }, this.throttle = function(a, b, c) {
            var d=!1;
            return function() {
                d || (a.apply(c, arguments), d=!0, setTimeout(function() {
                    d=!1
                }, b))
            }
        }, this.after("initialize", function() {
            this.on("uiReplyToTweet", "uiOpenReplyDialog"), this.on(document, "dataDidDeleteTweet", this.closeGallery), this.on(document, "dataGotMoreMediaTimelineItems", this.renderNav), this.on(document, "uiOpenGallery", this.open), this.on(document, "uiCloseGallery", this.closeGallery), this.on(document, "uiShortcutEsc", this.handleClose), this.on(window, "popstate", this.closeGallery), this.on(document, "uiShortcutLeft", this.throttle(this.prev, 200, this)), this.on(document, "uiShortcutRight", this.throttle(this.next, 200, this)), this.on(document, "dataGotTweet", this.gotTweet), this.on("click", {
                prevSelector: this.prev,
                nextSelector: this.next,
                closeSelector: this.handleClose,
                gridSelector: this.closeGallery
            })
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), _ = require("core/i18n"), imageLoader = require("app/utils/image/image_loader"), imageResizer = require("app/utils/image/image_resizer"), withFlagAction = require("app/ui/media/with_flag_action"), withItemActions = require("app/ui/with_item_actions"), withTweetTranslation = require("app/ui/with_tweet_translation"), withScrollbarWidth = require("app/ui/with_scrollbar_width"), Gallery = defineComponent(gallery, withItemActions, withFlagAction, withScrollbarWidth, withTweetTranslation);
    module.exports = Gallery
});
define("app/data/gallery_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function galleryScribe() {
        this.scribeGalleryOpened = function(a, b) {
            this.scribe({
                element: "gallery",
                action: "open"
            }, b)
        }, this.scribeGalleryClosed = function(a, b) {
            this.scribe({
                element: "gallery",
                action: "close"
            }, b)
        }, this.scribeGalleryMediaLoaded = function(a, b) {
            var c = {
                url: b.url,
                item_ids: [b.id]
            };
            this.scribe({
                element: "photo",
                action: "impression"
            }, b, c)
        }, this.scribeGalleryMediaFailed = function(a, b) {
            var c = {
                url: b.url,
                item_ids: [b.id]
            };
            this.scribe({
                element: "photo",
                action: "error"
            }, b, c)
        }, this.scribeGalleryNavigateNext = function(a, b) {
            this.scribe({
                element: "next",
                action: "click"
            }, b)
        }, this.scribeGalleryNavigatePrev = function(a, b) {
            this.scribe({
                element: "prev",
                action: "click"
            }, b)
        }, this.scribeGridPaged = function(a, b) {
            this.scribe({
                element: "grid",
                action: "page"
            }, b)
        }, this.scribeGridOpened = function(a, b) {
            this.scribe({
                element: "grid",
                action: "impression"
            }, b)
        }, this.after("initialize", function() {
            this.on(document, "uiGalleryOpened", this.scribeGalleryOpened), this.on(document, "uiGalleryClosed", this.scribeGalleryClosed), this.on(document, "uiGalleryMediaLoaded", this.scribeGalleryMediaLoaded), this.on(document, "uiGalleryMediaFailed", this.scribeGalleryMediaFailed), this.on(document, "uiGalleryNavigateNext", this.scribeGalleryNavigateNext), this.on(document, "uiGalleryNavigatePrev", this.scribeGalleryNavigatePrev), this.on(document, "uiGridPaged", this.scribeGridPaged), this.on(document, "uiGridOpened", this.scribeGridOpened)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(galleryScribe, withScribe)
});
define("app/ui/global_loading_indicator", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function globalLoadingIndicator() {
        this.defaultAttrs({
            birdSelector: "h1.bird-topbar-etched",
            spinnerContainer: "body",
            globalHeadingSelector: ".global-nav h1",
            spinnerClass: "pushing-state",
            spinnerSelector: ".pushstate-spinner"
        }), this.showSpinner = function(a, b) {
            this.select("birdSelector").removeAttr("style"), this.select("spinnerContainer").addClass(this.attr.spinnerClass)
        }, this.hideSpinner = function(a, b) {
            this.select("spinnerContainer").removeClass(this.attr.spinnerClass)
        }, this.addSpinner = function() {
            this.select("spinnerSelector").length || $('<div class="pushstate-spinner"></div>').insertAfter(this.select("globalHeadingSelector"))
        }, this.after("initialize", function() {
            this.on("uiSwiftLoaded", this.addSpinner), this.on("uiShowGlobalLoadingIndicator", this.showSpinner), this.on("uiHideGlobalLoadingIndicator", this.hideSpinner)
        })
    }
    var defineComponent = require("core/component"), GlobalLoadingIndicator = defineComponent(globalLoadingIndicator);
    module.exports = GlobalLoadingIndicator
});
define("app/ui/dialogs/goto_user_dialog", ["module", "require", "exports", "core/component", "app/ui/typeahead/typeahead_dropdown", "app/ui/typeahead/typeahead_input", "app/ui/with_dialog"], function(module, require, exports) {
    function gotoUserDialog() {
        this.defaultAttrs({
            dropdownId: "swift_autocomplete_dropdown_goto_user",
            inputSelector: "input.username-input",
            usernameFormSelector: "form.goto-user-form"
        }), this.focus = function() {
            this.select("inputSelector").focus()
        }, this.gotoUser = function(a, b) {
            if (b && b.dropdownId && b.dropdownId != this.attr.dropdownId)
                return;
            a.preventDefault();
            if (b && b.item) {
                this.select("inputSelector").val(b.item.screen_name), this.trigger("uiNavigate", {
                    href: b.href
                });
                return 
            }
            var c = this.select("inputSelector").val().trim();
            c.substr(0, 1) == "@" && (c = c.substr(1)), this.trigger("uiNavigate", {
                href: "/" + c
            })
        }, this.reset = function() {
            this.select("inputSelector").val(""), this.select("inputSelector").blur(), this.trigger(this.select("inputSelector"), "uiHideAutocomplete")
        }, this.after("initialize", function() {
            this.on(document, "uiShortcutShowGotoUser", this.open), this.on("uiDialogOpened", this.focus), this.on("uiDialogClosed", this.reset), this.on(this.select("usernameFormSelector"), "submit", this.gotoUser), this.on("uiTypeaheadItemSelected uiTypeaheadSubmitQuery", this.gotoUser), TypeaheadInput.attachTo(this.$node, {
                inputSelector: this.attr.inputSelector
            }), TypeaheadDropdown.attachTo(this.$node, {
                inputSelector: this.attr.inputSelector,
                autocompleteAccounts: !1,
                datasourceRenders: [["accounts", ["accounts"]]],
                deciders: this.attr.typeaheadData,
                eventData: {
                    scribeContext: {
                        component: "goto_user_dialog"
                    }
                }
            })
        })
    }
    var defineComponent = require("core/component"), TypeaheadDropdown = require("app/ui/typeahead/typeahead_dropdown"), TypeaheadInput = require("app/ui/typeahead/typeahead_input"), withDialog = require("app/ui/with_dialog"), GotoUserDialog = defineComponent(gotoUserDialog, withDialog);
    module.exports = GotoUserDialog
});
define("app/ui/dialogs/keyboard_shortcuts_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "core/i18n", "app/utils/cookie", "template"], function(module, require, exports) {
    function keyboardShortcutsDialog() {
        this.defaultAttrs({
            showShortcutsBtnSelector: "#show-shortcuts-btn",
            dismissShortcutsBtnSelector: "#dismiss-shortcuts-btn",
            suppressSRMsgCookie: "suppress_sr_shortcuts_msg",
            enableActivity: !1,
            loggedIn: !1
        }), this.renderSRMessage = function() {
            if ($("#kb-shortcuts-msg").length)
                return;
            var a = template.screen_reader_kb_shortcuts_message.render({}, template);
            $(a).insertBefore(document.body.firstChild)
        }, this.addMessage = function() {
            if (cookie(this.attr.suppressSRMsgCookie))
                return;
            this.renderSRMessage()
        }, this.removeMessage = function() {
            $("#kb-shortcuts-msg").remove(), cookie(this.attr.suppressSRMsgCookie, "1")
        }, this.renderShortcutsDialog = function() {
            if ($("#keyboard-shortcut-dialog").length)
                return;
            var a = {
                is_mac: isMac,
                enable_activity: this.attr.enableActivity,
                logged_in: this.attr.loggedIn
            };
            $("body").append(template["dialogs/keyboard_shortcuts_dialog"].render(a, template)), this.$dialogContainer = $("#keyboard-shortcut-dialog"), this.$dialog = this.$dialogContainer.find(this.attr.modalSelector)
        }, this.showShortcutsDialog = function() {
            this.renderShortcutsDialog(), this.open()
        }, this.after("initialize", function() {
            this.on(document, "click", {
                showShortcutsBtnSelector: this.showShortcutsDialog,
                dismissShortcutsBtnSelector: this.removeMessage
            }), this.on(document, "uiSwiftLoaded", this.addMessage), this.on(document, "uiOpenKeyboardShortcutsDialog", this.showShortcutsDialog)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), _ = require("core/i18n"), cookie = require("app/utils/cookie"), template = require("template"), KeyboardShortcutsDialog = defineComponent(keyboardShortcutsDialog, withDialog), isMac = navigator.platform.toLowerCase().indexOf("mac")!=-1;
    module.exports = KeyboardShortcutsDialog
});
define("app/ui/dialogs/leadgen_confirm_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog"], function(module, require, exports) {
    function leadGenConfirmDialog() {
        this.defaultAttrs({
            promoCTASelector: ".stream-items .leadgen-card-forward .promotion-cta",
            leadgenDialogIframeSelector: ".cards2-promotion-iframe",
            tweetSelector: ".leadgen-card-forward"
        });
        var a = {
            OPEN_DIALOG: "uiOpenLeadGenConfirmDialog",
            DIALOG_CLOSED: "uiDialogClosed",
            EXPANDED_TWEET: "uiHasExpandedTweet",
            COLLAPSED_TWEET: "uiHasCollapsedTweet"
        };
        this.setUpDialogWithElement = function(a, b) {
            this.$tweet = $(b.el).closest(this.attr.tweetSelector);
            var c = this.$tweet.closest("[data-impression-id]").attr("data-impression-id"), d = this.$tweet.closest("[data-impression-id]").attr("data-disclosure-type"), e = $(b.el).data("card-url");
            this.setUpDialog(a, this.$tweet, c, d, e)
        }, this.setUpDialogWithData = function(a, b) {
            this.$tweet = $(this.attr.tweetSelector + "[data-tweet-id=" + b.tweetId + "]"), this.setUpDialog(a, this.$tweet, b.impressionId, b.disclosureType, b.cardUrl)
        }, this.scribeClose = function(b, c) {
            this.trigger(this.$tweet, a.COLLAPSED_TWEET)
        }, this.after("initialize", function() {
            this.leadgenDialogIframe = this.select("leadgenDialogIframeSelector"), this.on(document, "click", {
                promoCTASelector: this.setUpDialogWithElement
            }), this.on(document, a.OPEN_DIALOG, this.setUpDialogWithData), this.on(a.DIALOG_CLOSED, this.scribeClose)
        }), this.setUpDialog = function(b, c, d, e, f) {
            this.trigger(c, a.EXPANDED_TWEET, {
                organicExpansion: !0,
                impressionId: d,
                disclosureType: e
            }), this.leadgenDialogIframe.attr("src", f), b.preventDefault(), this.open()
        }
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), LeadGenConfirmDialog = defineComponent(leadGenConfirmDialog, withDialog);
    module.exports = LeadGenConfirmDialog
});
define("app/ui/dialogs/list_membership_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog"], function(module, require, exports) {
    function listMembershipDialog() {
        this.defaultAttrs({
            top: 90,
            contentSelector: ".list-membership-content",
            createListSelector: ".create-a-list",
            membershipSelector: ".list-membership-container li"
        }), this.openListMembershipDialog = function(a, b) {
            this.userId = b.userId, this.userId && this.trigger("uiNeedsListMembershipContent", {
                userId: this.userId
            }), this.$content.empty(), this.$node.removeClass("has-content"), this.open()
        }, this.addListMembershipContent = function(a, b) {
            this.$node.addClass("has-content"), this.$content.html(b.html)
        }, this.handleNoListMembershipContent = function(a, b) {
            this.close(), this.trigger("uiShowError", b)
        }, this.toggleListMembership = function(a, b) {
            var c = $(a.target), d = {
                userId: c.closest("[data-user-id]").attr("data-user-id"),
                listId: c.closest("[data-list-id]").attr("data-list-id")
            }, e = $("#list_" + d.listId);
            if (!e.is(":visible"))
                return;
            e.closest(this.attr.membershipSelector).addClass("pending"), e.data("is-checked") ? this.trigger("uiRemoveUserFromList", d) : this.trigger("uiAddUserToList", d)
        }, this.updateMembershipState = function(a) {
            return function(b, c) {
                var d = $("#list_" + c.sourceEventData.listId);
                d.closest(this.attr.membershipSelector).removeClass("pending"), d.attr("checked", a ? "checked" : null), d.data("is-checked", a), d.attr("data-is-checked", a)
            }.bind(this)
        }, this.openListCreateDialog = function() {
            this.close(), this.trigger("uiOpenCreateListDialog", {
                userId: this.userId
            })
        }, this.after("initialize", function() {
            this.$content = this.select("contentSelector"), this.on("click", {
                createListSelector: this.openListCreateDialog,
                membershipSelector: this.toggleListMembership
            }), this.on(document, "uiListAction uiOpenListMembershipDialog", this.openListMembershipDialog), this.on(document, "dataGotListMembershipContent", this.addListMembershipContent), this.on(document, "dataFailedToGetListMembershipContent", this.handleNoListMembershipContent), this.on(document, "dataDidAddUserToList dataFailedToRemoveUserFromList", this.updateMembershipState(!0)), this.on(document, "dataDidRemoveUserFromList dataFailedToAddUserToList", this.updateMembershipState(!1))
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), ListMembershipDialog = defineComponent(listMembershipDialog, withDialog);
    module.exports = ListMembershipDialog
});
define("app/ui/dialogs/list_operations_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "core/i18n", "core/utils"], function(module, require, exports) {
    function listOperationsDialog() {
        this.defaultAttrs({
            top: 90,
            win: window,
            saveListSelector: ".update-list-button",
            editorSelector: ".list-editor",
            currentNameSelector: ".js-list-details .js-list-name",
            currentDescSelector: ".js-list-details .js-list-desc",
            nameInputSelector: ".list-editor input[name='name']",
            descriptionSelector: ".list-editor textarea[name='description']",
            privacySelector: ".list-editor input[name='mode']"
        }), this.openListOperationsDialog = function(a, b) {
            b = b || {}, this.userId = b.userId, a.type == "uiOpenUpdateListDialog" && this.modifyDialog(), this.open(), this.$nameInput.focus()
        }, this.modifyDialog = function() {
            this.$modalTitle = this.select("modalTitleSelector"), this.originalTitle = this.originalTitle || this.$modalTitle.text(), this.$modalTitle.text(_('Edit list details')), this.$nameInput.val($(this.attr.currentNameSelector).text().trim()), this.$descriptionInput.val($(this.attr.currentDescSelector).text().trim()), this.isPublic || (this.$privacyInput[1].checked=!0), this.$saveButton.attr("data-list-id", this.listId).attr("data-operation", "update"), this.toggleSaveButtonDisabled(), this.modified=!0, this.$descriptionInput.on("keyup", this.toggleSaveButtonDisabled.bind(this)), this.$privacyInput.on("change", this.toggleSaveButtonDisabled.bind(this))
        }, this.revertModifications = function() {
            this.modified && (this.revertDialog(), this.$editor.find("input,textarea").val(""), this.$descriptionInput.off("keyup"), this.$privacyInput.off("change"), this.modified=!1)
        }, this.revertDialog = function() {
            this.$modalTitle.text(this.originalTitle), this.$saveButton.removeAttr("data-list-id").removeAttr("data-operation"), this.isPublic || (this.$privacyInput[0].checked=!0)
        }, this.saveList = function(a, b) {
            if (this.requestInProgress)
                return;
            this.requestInProgress=!0;
            var c = $(b.el), d = c.attr("data-operation") == "update" ? "uiUpdateList": "uiCreateList", e = {
                name: this.formValue("name"),
                description: this.formValue("description", {
                    type: "textarea"
                }),
                mode: this.formValue("mode", {
                    conditions: ":checked"
                })
            };
            c.attr("data-operation") == "update" && (e = utils.merge(e, {
                list_id: c.attr("data-list-id")
            })), this.trigger(d, e), this.$saveButton.attr("disabled", !0)
        }, this.saveListSuccess = function(a, b) {
            this.close();
            var c = _('List saved!');
            a.type == "dataDidCreateList" ? (c = _('List created!'), this.userId ? this.trigger("uiOpenListMembershipDialog", {
                userId: this.userId
            }) : b && b.slug && (this.attr.win.location = "/" + this.attr.screenName + "/lists/" + b.slug)) : this.revertDialog(), this.$editor.find("input,textarea").val(""), this.trigger("uiShowMessage", {
                message: c
            })
        }, this.saveListComplete = function(a, b) {
            this.requestInProgress=!1, this.toggleSaveButtonDisabled()
        }, this.toggleSaveButtonDisabled = function(a, b) {
            this.$saveButton.attr("disabled", this.$nameInput.val() == "")
        }, this.formValue = function(a, b) {
            return b = b || {}, b.type = b.type || "input", b.conditions = b.conditions || "", this.$editor.find(b.type + "[name='" + a + "']" + b.conditions).val()
        }, this.disableSaveButton = function() {
            this.$saveButton.attr("disabled", !0)
        }, this.updateState = function(a, b) {
            this.listId = b.init_data.list_id, this.isPublic = b.init_data.is_public
        }, this.after("initialize", function() {
            this.listId = this.attr.list_id, this.isPublic = this.attr.is_public, this.$editor = this.select("editorSelector"), this.$nameInput = this.select("nameInputSelector"), this.$descriptionInput = this.select("descriptionSelector"), this.$privacyInput = this.select("privacySelector"), this.$saveButton = this.select("saveListSelector"), this.on("click", {
                saveListSelector: this.saveList
            }), this.on("focus blur keyup", {
                nameInputSelector: this.toggleSaveButtonDisabled
            }), this.on("uiDialogOpened", this.disableSaveButton), this.on("uiDialogClosed", this.revertModifications), this.on(document, "uiOpenCreateListDialog uiOpenUpdateListDialog", this.openListOperationsDialog), this.on(document, "dataDidCreateList dataDidUpdateList", this.saveListSuccess), this.on(document, "dataDidCreateList dataDidUpdateList dataFailedToCreateList dataFailedToUpdateList", this.saveListComplete), this.on(document, "uiPageChanged", this.updateState)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), _ = require("core/i18n"), utils = require("core/utils"), ListOperationsDialog = defineComponent(listOperationsDialog, withDialog);
    module.exports = ListOperationsDialog
});
define("app/data/sms/sms_download_scribe", ["module", "require", "exports", "core/component", "core/utils", "app/data/with_scribe"], function(module, require, exports) {
    function smsDownloadScribe() {
        this.defaultAttrs({
            scribeContext: {
                component: "sms_download"
            }
        }), this.dismissScribe = function(a, b) {
            this.scribe(utils.merge(this.attr.scribeContext, {
                action: "dismiss"
            }))
        }, this.scribeImpression = function(a, b) {
            this.scribe(utils.merge(this.attr.scribeContext, {
                action: "impression"
            }))
        }, this.submitVerificationScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "device_verification",
                action: "submit"
            });
            this.scribe(c)
        }, this.submitDownloadLinkScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "download_link",
                action: "submit"
            });
            this.scribe(c)
        }, this.resendDownloadLinkScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "download_link_resend",
                action: "submit"
            });
            this.scribe(c)
        }, this.failedToVerifyDeviceScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "device_verification",
                action: "failure"
            });
            this.scribe(c)
        }, this.failedToSubmitDownloadLinkScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "download_link",
                action: "failure"
            });
            this.scribe(c)
        }, this.failedToResendDownloadLinkScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "download_link_resend",
                action: "failure"
            });
            this.scribe(c)
        }, this.succeededToVerifyDeviceScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "device_verification",
                action: "success"
            });
            this.scribe(c)
        }, this.succeededToSubmitDownloadLinkScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "download_link",
                action: "success"
            });
            this.scribe(c)
        }, this.succeededToResendDownloadLinkScribe = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "download_link_resend",
                action: "success"
            });
            this.scribe(c)
        }, this.after("initialize", function() {
            this.on("uiDismissMobileDownloadPrompt", this.dismissScribe), this.on("uiMobileDownloadPromptLoaded", this.scribeImpression), this.on("uiDidSubmitDeviceVerification", this.submitVerificationScribe), this.on("uiDidSubmitDownloadLinkRequest", this.submitDownloadLinkScribe), this.on("uiDidResendDownloadLinkRequest", this.resendDownloadLinkScribe), this.on("dataFailedToVerifyDevice", this.failedToVerifyDeviceScribe), this.on("dataFailedToSubmitDownloadLinkRequest", this.failedToSubmitDownloadLinkScribe), this.on("dataFailedToResendDownloadLinkRequest", this.failedToResendDownloadLinkScribe), this.on("dataDidVerifyDevice", this.succeededToVerifyDeviceScribe), this.on("dataDidSubmitDownloadLinkRequest", this.succeededToSubmitDownloadLinkScribe), this.on("dataDidResendDownloadLinkRequest", this.succeededToResendDownloadLinkScribe)
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(smsDownloadScribe, withScribe)
});
define("app/utils/find_tweet", ["module", "require", "exports"], function(module, require, exports) {
    function findTweet(a, b, c) {
        var d = b.split(",").map(function(a) {
            return a + "[data-tweet-id=" + c + "]"
        }).join();
        return a.find(d)
    }
    module.exports = findTweet
});
define("app/ui/with_navigation_links_scribing", ["module", "require", "exports"], function(module, require, exports) {
    module.exports = function() {
        this.defaultAttrs({
            navSelector: "a[data-nav]:not(.js-nonNavigable)"
        }), this.navEvent = function(a, b) {
            var c = $(b.el);
            this.trigger("uiNavigationLinkClick", {
                scribeContext: {
                    element: c.attr("data-nav")
                },
                url: c.attr("href")
            })
        }, this.after("initialize", function(a) {
            this.on("click", {
                navSelector: this.navEvent
            })
        })
    }
});
define("app/ui/with_tweet_actions_helper", ["module", "require", "exports", "core/utils"], function(module, require, exports) {
    function withTweetActionsHelper() {
        this.preventDefault = function(a, b, c) {
            a.preventDefault()
        }, this.composeHandler = function() {
            var a = arguments, b = {};
            return function(c, d) {
                for (var e = 0, f = a.length; e < f; e++)
                    if (a[e].call(this, c, d, b) == 0)
                        break
            }
        }, this.getTweetForEvent = function(a, b) {
            var c = $(a.target), d;
            return c.is(this.attr.tweetItemSelector) && (d = c), d || (d = c[b](this.attr.tweetItemSelector)), d
        }, this.getClosestTweet = function(a, b, c) {
            var d = this.getTweetForEvent(a, "closest"), e=!!d.length;
            return e && (c.$tweet = d), e
        }, this.getDataForTweet = function(a) {
            return {
                id: a.attr("data-tweet-id"),
                screenName: a.attr("data-screen-name"),
                isTweetProof: a.attr("data-is-tweet-proof") === "true",
                viewerFollowsAuthor: a.attr("data-you-follow") === "true",
                viewerBlocksAuthor: a.attr("data-you-block") === "true",
                featureContext: a.attr("data-feature-context"),
                mediaType: this.getMediaType(a)
            }
        }, this.getMediaType = function(a) {
            var b = "";
            if (a.attr("data-card-type") == "photo")
                b = "image";
            else if (a.attr("data-card2-type") == "player")
                b = "video";
            else if (a.attr("data-has-native-media") || a.attr("data-has-cards"))
                b = "media";
            return b
        }, this.mkTweetDataCollector = function() {
            var a = arguments;
            return function(b) {
                var c = {};
                for (var d = 0, e = a.length; d < e; d++)
                    c = utils.merge(c, a[d].call(this, b));
                return c = utils.merge(c, this.getDataForTweet(b)), c
            }.bind(this)
        }, this.mkTweetDataCollectorForAction = function() {
            var a = this.mkTweetDataCollector.apply(this, arguments);
            return function(b, c, d) {
                return d.tweetData = a(d.$tweet), !0
            }.bind(this)
        }, this.triggerTweetAction = function(a) {
            return function(b, c, d) {
                var e = typeof a == "function" ? a.call(this, b, c, d): a;
                this.trigger(d.$tweet, e, d.tweetData)
            }
        }
    }
    var utils = require("core/utils");
    module.exports = withTweetActionsHelper
});
define("app/ui/more_tweet_actions_dropdown", ["module", "require", "exports", "core/i18n", "core/component", "app/utils/find_tweet", "app/data/user_info", "app/ui/with_dropdownmenu", "app/ui/with_interaction_data", "app/ui/with_navigation_links_scribing", "app/ui/with_tweet_actions_helper"], function(module, require, exports) {
    function moreTweetActionsDropdown() {
        this.defaultAttrs({
            tweetItemSelector: ".tweet",
            deleteSelector: ".js-actionDelete",
            shareViaDMSelector: ".js-actionShareViaDM",
            shareViaEmailSelector: ".js-actionShareViaEmail",
            embedTweetSelector: ".js-actionEmbedTweet",
            blockOrReportTweetSelector: ".js-actionBlockOrReport",
            userTogglePinTweetSelector: ".js-actionPinTweet",
            curateTweetSelector: ".js-actionCurateTweet",
            curateToOtherSelector: ".js-actionCurateTweetOther",
            curateToNewTimelineSelector: ".js-actionCurateTweetNewTimeline",
            removeMediaTagSelector: ".js-actionRemoveTag",
            tweetContainerSelector: ".js-stream-item",
            mediaTagContainerSelector: ".media-tags-container",
            pinnedTweetClass: "user-pinned",
            hasTimelinesClass: "has-timelines"
        }), this.renderCustomTimelinesList = function(a, b) {
            var c = this.select("curateToOtherSelector");
            this.select("curateTweetSelector").remove(), c.before(b.html), this.select("curateTweetSelector").length > 0 && c.addClass(this.attr.hasTimelinesClass)
        }, this.curateTweet = function(a, b, c) {
            var d = $(b.el), e = d.attr("data-timeline-id"), f = d.data("is-member"), g = c.tweetData;
            g.timelineId = e, f ? this.trigger("uiCurateRemoveTweet", g) : this.trigger("uiCurateAddTweet", g), this.trigger("uiCloseDropdowns")
        }, this.getCustomTimelines = function(a, b, c) {
            var d = c.$tweet.attr("data-tweet-id");
            this.trigger("uiNeedsCustomTimelines", {
                tweetId: d
            })
        }, this.confirmRemoveMediaTag = function(a) {
            this.trigger(a.target, "uiOpenConfirmDialog", {
                titleText: _('Remove photo tag from Tweet?'),
                bodyText: _('Are you sure you want to remove your photo tag from this Tweet?'),
                cancelText: _('Cancel'),
                submitText: _('OK'),
                action: "MediaTagRemove"
            })
        }, this.removeMediaTag = function(a, b, c) {
            this.trigger(document, "uiDidRemoveMediaTag", c.tweetData)
        }, this.handleRemoveMediaTagSuccess = function(a, b) {
            var c = findTweet(this.$node, this.attr.tweetItemSelector, b.id_str);
            c.find(this.attr.mediaTagContainerSelector).html(b.item_html), this.trigger("uiShowMessage", {
                message: _('Photo tag removed')
            })
        }, this.handleRemoveMediaTagFailure = function(a) {
            this.trigger("uiShowMessage", {
                message: _('Cannot delete the tag. Please try again later.')
            })
        }, this.showConfirmPinDialog = function(a) {
            a.stopImmediatePropagation(), this.trigger(a.target, "uiOpenConfirmDialog", {
                titleText: _('Pin this Tweet to the top of your profile'),
                bodyText: _('This will replace any previously pinned Tweet. Are you sure?'),
                cancelText: _('Cancel'),
                submitText: _('Pin'),
                action: "UserPinTweetToggle"
            })
        }, this.showConfirmUnPinDialog = function(a) {
            a.stopImmediatePropagation(), this.trigger(a.target, "uiOpenConfirmDialog", {
                titleText: _('Unpin from your profile'),
                bodyText: _('Are you sure?'),
                cancelText: _('Cancel'),
                submitText: _('Unpin'),
                action: "UserPinTweetToggle"
            })
        }, this.showUserPinToggledMessage = function(a, b) {
            this.trigger("uiShowStickyMessage", b)
        }, this.unpinOtherTweets = function(a, b) {
            this.select("tweetItemSelector").removeClass(this.attr.pinnedTweetClass), this.handleTransition("addClass", this.attr.pinnedTweetClass).call(this, a, b)
        }, this.handleTransition = function(a, b) {
            return function(c, d) {
                var e = d.id || d.sourceEventData.id, f = findTweet(this.$node, this.attr.tweetItemSelector, e);
                f[a](b)
            }
        }, this.confirmPinToggle = function(a, b, c) {
            var d = c.$tweet.hasClass(this.attr.pinnedTweetClass) ? "uiNeedUserUnpinTweetConfirm": "uiNeedUserPinTweetConfirm";
            this.trigger(c.$tweet, d)
        }, this.togglePinnedTweet = function(a, b, c) {
            return c.$tweet.hasClass(this.attr.pinnedTweetClass) ? "uiDidUserUnpinTweet" : "uiDidUserPinTweet"
        }, this.after("initialize", function() {
            var a = this.mkTweetDataCollectorForAction(this.interactionDataWithCard);
            this.on("click", {
                embedTweetSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiNeedsEmbedTweetDialog"))
            });
            if (!this.attr.loggedIn)
                return;
            this.on("click", {
                deleteSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiOpenDeleteDialog")),
                userTogglePinTweetSelector: this.composeHandler(this.getClosestTweet, this.confirmPinToggle),
                shareViaDMSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiComposeNewDMWithTweet")),
                shareViaEmailSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiNeedsShareViaEmailDialog")),
                curateTweetSelector: this.composeHandler(this.getClosestTweet, a, this.curateTweet),
                curateToOtherSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiOpenCurateDialog")),
                curateToNewTimelineSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiOpenCreateCustomTimelineDialog")),
                removeMediaTagSelector: this.confirmRemoveMediaTag,
                blockOrReportTweetSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiNeedsBlockOrReportDialog"))
            }), this.on("uiDropdownOpened", this.composeHandler(this.getClosestTweet, this.getCustomTimelines)), this.on("uiUserPinTweetToggleConfirm", this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction(this.togglePinnedTweet))), this.on("uiMediaTagRemoveConfirm", this.composeHandler(this.getClosestTweet, a, this.removeMediaTag)), this.on(document, "uiHasInjectedNewTimeline uiHasInjectedOldTimelineItems uiHasInjectedRangeTimelineItems", this.applyARIAAttrs), this.on(document, "dataGotCustomTimelines", this.renderCustomTimelinesList), this.on(document, "uiNeedUserPinTweetConfirm", this.showConfirmPinDialog), this.on(document, "uiNeedUserUnpinTweetConfirm", this.showConfirmUnPinDialog), this.on(document, "dataDidUserPinTweet dataDidUserUnpinTweet", this.showUserPinToggledMessage), this.on(document, "dataDidUserPinTweet", this.unpinOtherTweets), this.on(document, "uiDidUserPinTweet dataFailedToUserUnpinTweet", this.handleTransition("addClass", this.attr.pinnedTweetClass)), this.on(document, "uiDidUserUnpinTweet dataFailedToUserPinTweet", this.handleTransition("removeClass", this.attr.pinnedTweetClass)), this.on(document, "dataMediaTagRemoveSuccess", this.handleRemoveMediaTagSuccess), this.on(document, "dataMediaTagRemoveFailure", this.handleRemoveMediaTagFailure)
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component"), findTweet = require("app/utils/find_tweet"), userInfo = require("app/data/user_info"), withDropdownMenu = require("app/ui/with_dropdownmenu"), withInteractionData = require("app/ui/with_interaction_data"), withNavigationLinks = require("app/ui/with_navigation_links_scribing"), withTweetActionsHelper = require("app/ui/with_tweet_actions_helper"), MoreTweetActionsDropdown = defineComponent(moreTweetActionsDropdown, withDropdownMenu, withTweetActionsHelper, withInteractionData, withNavigationLinks);
    module.exports = MoreTweetActionsDropdown
});
define("app/utils/multiline_text_range", ["module", "require", "exports"], function(module, require, exports) {
    function MultilineTextRange(a) {
        function c() {
            var b = document.createRange(), c = a.contents(), d = a.contents().last()[0];
            return c.length > 0 && (b.setStart(c[0], 0), b.setEnd(d, g(d))), b
        }
        function d(a) {
            return a.nodeType == 3 || a.nodeType == 4
        }
        function e() {
            var a = f(), b = null, c = null, d = null, e = null;
            for (var g = 0; g < a.length; g++) {
                var h = a[g];
                if (c === null || h.bottom > c)
                    c = h.bottom;
                if (b === null || h.top < b)
                    b = h.top;
                if (d === null || h.left < d)
                    d = h.left;
                if (e === null || h.right > e)
                    e = h.right
            }
            return {
                top: b,
                bottom: c,
                left: d,
                right: e,
                width: Math.abs(e - d),
                height: Math.abs(c - b)
            }
        }
        function f() {
            var a = b.getClientRects(), c = [];
            for (var d = 0; d < a.length; d++)
                c.push(a[d]);
            return c.filter(function(a) {
                return a.width > 0
            })
        }
        function g(a) {
            return d(a) ? a.nodeValue.length : a.childNodes.length
        }
        function h() {
            return a.width() - i()
        }
        function i() {
            var a = e(), b = f(), c = b[b.length - 1];
            return b.length == 0 ? 0 : c.right - a.left
        }
        function j(a) {
            var b = a.match(WORD_BREAKS_REGEX);
            if (b !== null) {
                var c = b[b.length - 1], d = a.lastIndexOf(c);
                return {
                    startIndex: d,
                    endIndex: d + c.length
                }
            }
            return null
        }
        function k(a) {
            return !!a.match(WORD_BREAK_REGEX)
        }
        function l(a) {
            return (a.match(CHARS_REGEX) || []).length
        }
        var b = c();
        this.shortenToVisibleContent = function() {
            var c = a.contents();
            for (var e = c.length - 1; e >= 0; e--) {
                var f = c[e];
                if (d(f))
                    for (var g = f.nodeValue.length; g >= 0; g--) {
                        b.setEnd(f, g);
                        if (!this.hasOverflow())
                            return 
                    } else {
                    b.setEnd(f, f.childNodes.length);
                    if (!this.hasOverflow())
                        return 
                    }
            }
        }, this.shortenToNearestWordBreak = function(a) {
            var c = b.endContainer, e = a.maxCharsToRemove;
            if (!d(c) ||!c.nodeValue.match(NEW_LINE_ONLY_REGEX) && (b.endOffset === c.nodeValue.length || k(c.nodeValue[b.endOffset])))
                return;
            var f = c.nodeValue.slice(0, b.endOffset), g = j(f);
            if (g === null)
                return;
            if (e !== undefined && l(f.substring(g.endIndex)) > e)
                return;
            b.setEnd(c, g.startIndex)
        }, this.ensureLastLineAvailableWidth = function(a) {
            while (h(b) <= a) {
                var c = b.endContainer;
                if (d(c) && b.endOffset > 1)
                    b.setEnd(c, b.endOffset - 1);
                else {
                    var e = c.previousSibling;
                    if (e === null)
                        break;
                    b.setEnd(e, g(e))
                }
            }
        }, this.hasOverflow = function() {
            var b = e(), c = parseFloat(a.css("line-height"));
            return Math.round(b.height / c) > Math.round(a.height() / c)
        }, this.toDocumentFragment = function() {
            return b.cloneContents()
        }, this.setEnd = function(a, c) {
            b.setEnd(a, c)
        }
    }
    var CHARS_REGEX = /([\uD800-\uDBFF][\uDC00-\uDFFF]|[\S\s])/g, WORD_BREAK_REGEX = /\s/, WORD_BREAKS_REGEX = /\s+/g, NEW_LINE_ONLY_REGEX = /^\n$/;
    module.exports = MultilineTextRange
});
define("app/ui/multiline_ellipses", ["module", "require", "exports", "core/component", "app/utils/multiline_text_range"], function(module, require, exports) {
    function multilineEllipses() {
        this.defaultAttrs({
            unEllipsifiedTextClass: "js-ellipsis",
            unEllipsifiedTextSelector: ".js-ellipsis",
            maxCharsRemoveEnsureEndOnWordBreak: 5
        }), this.after("initialize", function() {
            if (typeof document.createRange == "undefined")
                return;
            this.on(document, "uiSwiftLoaded uiPageChanged uiHasInjectedNewTimeline uiHasInjectedOldTimelineItems uiHasInjectedRangeTimelineItems dataTweetConversationResult uiOverlayUpdate uiTrendsDisplayed uiShowMoreTrends uiCommerceTabToggled", this.addEllipses)
        }), this.addEllipses = function() {
            var a = this.select("unEllipsifiedTextSelector").filter(":visible");
            a.removeClass(this.attr.unEllipsifiedTextClass), a.each(this.addEllipsis.bind(this))
        }, this.addEllipsis = function(a, b) {
            var c = $(b), d = this.createRange(c);
            if (!d.hasOverflow())
                return;
            c.data("full-text", c.text()), d.shortenToVisibleContent(), d.ensureLastLineAvailableWidth(this.maxEllipsisWidth(c)), d.shortenToNearestWordBreak({
                maxCharsToRemove: this.attr.maxCharsRemoveEnsureEndOnWordBreak
            });
            var e = d.toDocumentFragment();
            e.appendChild(document.createTextNode("…")), c.html(e), this.trigger(c, "uiEllipsisAdded")
        }, this.maxEllipsisWidth = function(a) {
            var b = parseInt(a.css("font-size"));
            return b * 2
        }, this.createRange = function(a) {
            return new MultilineTextRange(a)
        }
    }
    var defineComponent = require("core/component"), MultilineTextRange = require("app/utils/multiline_text_range");
    module.exports = defineComponent(multilineEllipses)
});
define("app/ui/navigation_links", ["module", "require", "exports", "core/component", "app/ui/with_navigation_links_scribing"], function(module, require, exports) {
    var defineComponent = require("core/component"), withNavigationLinks = require("app/ui/with_navigation_links_scribing");
    module.exports = defineComponent(withNavigationLinks)
});
define("app/utils/setup_polling_with_backoff", ["module", "require", "exports", "core/clock", "core/utils"], function(module, require, exports) {
    function setupPollingWithBackoff(a, b, c) {
        var d = {
            focusedInterval: 3e4,
            blurredInterval: 9e4,
            backoffFactor: 2
        };
        c = utils.merge(d, c);
        var e = clock.setIntervalEvent(a, c.focusedInterval, c.eventData);
        return $(document).on("uiPageHidden", e.backoff.bind(e, c.blurredInterval, c.backoffFactor)), $(document).on("uiPageVisible", e.cancelBackoff.bind(e)), e
    }
    var clock = require("core/clock"), utils = require("core/utils");
    module.exports = setupPollingWithBackoff
});
define("app/data/notification_listener", ["module", "require", "exports", "core/component", "app/utils/setup_polling_with_backoff", "app/data/notifications"], function(module, require, exports) {
    function notificationListener() {
        this.pollForNotifications = function(a, b) {
            notifications.shouldPoll() && this.trigger("uiDMPoll")
        }, this.resetDMs = function(a, b) {
            notifications.resetDMState(a, b)
        }, this.notifications = notifications, this.after("initialize", function() {
            notifications.init(this.attr), this.on(document, "uiResetDMPoll", this.resetDMs), this.on(document, "uiPollForNotifications", this.pollForNotifications), this.timer = setupPollingWithBackoff("uiPollForNotifications")
        })
    }
    var defineComponent = require("core/component"), setupPollingWithBackoff = require("app/utils/setup_polling_with_backoff"), notifications = require("app/data/notifications"), NotificationListener = defineComponent(notificationListener);
    module.exports = NotificationListener
});
define("app/data/oembed", ["module", "require", "exports", "core/component", "app/data/with_data", "app/data/user_info", "core/utils"], function(module, require, exports) {
    function OembedData() {
        this.requestEmbedCode = function(a, b) {
            var c = this.cacheKeyForOptions(b), d = this.cachedEmbedCodes[c], e = this.receivedEmbedCode.bind(this, b), f = this.failedToReceiveEmbedCode.bind(this, b);
            if (d) {
                this.receivedEmbedCode(b, d);
                return 
            }
            this.get({
                dataType: "jsonp",
                url: this.embedCodeUrl(),
                data: utils.merge({
                    id: b.tweetId
                }, b.data),
                eventData: {},
                success: e,
                error: f
            })
        }, this.embedCodeUrl = function() {
            return "//api.twitter.com/1/statuses/oembed.json"
        }, this.receivedEmbedCode = function(a, b) {
            var c = this.cacheKeyForOptions(a);
            this.cachedEmbedCodes[c] = b, this.trigger("dataOembedSuccess", {
                options: a,
                data: b
            })
        }, this.failedToReceiveEmbedCode = function(a) {
            a.retry ? (a.retry=!1, this.trigger("dataOembedRetry", a), this.requestEmbedCode({}, a)) : this.trigger("dataOembedError", a)
        }, this.cacheKeyForOptions = function(a) {
            return JSON.stringify(a.data) + a.tweetId
        }, this.after("initialize", function() {
            this.on("uiNeedsOembed", this.requestEmbedCode), this.cachedEmbedCodes = {}
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), userInfo = require("app/data/user_info"), utils = require("core/utils");
    module.exports = defineComponent(OembedData, withData)
});
define("app/data/oembed_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function oembedScribe() {
        this.scribeError = function(a, b) {
            this.scribe({
                component: "oembed",
                action: "request_failed"
            })
        }, this.scribeRetry = function(a, b) {
            this.scribe({
                component: "oembed",
                action: "retry"
            })
        }, this.scribeRequest = function(a, b) {
            this.scribe({
                component: "oembed",
                action: "request"
            }, b)
        }, this.scribeSuccess = function(a, b) {
            this.scribe({
                component: "oembed",
                action: "success"
            }, b)
        }, this.after("initialize", function() {
            this.on("dataOembedError", this.scribeError), this.on("dataOembedRetry", this.scribeRetry), this.on("dataOembedRequest", this.scribeRequest), this.on("dataOembedSuccess", this.scribeSuccess)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(oembedScribe, withScribe)
});
define("app/ui/dialogs/offers_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/ui/with_position", "app/ui/commerce/with_email_confirmation", "app/utils/with_iframe_height_adjuster"], function(module, require, exports) {
    function offersConfirmDialog() {
        this.defaultAttrs({
            offersDialogIframeSelector: ".offers-card-iframe",
            offersActionSelector: ".OffersCard-offersAction",
            tweetSelector: ".tweet, .ProfileTweet",
            commerceCardSelector: ".OffersCard",
            top: 47,
            defaultHeight: "600px"
        }), this.openDialog = function(a, b) {
            var c;
            b && b.tweetId && (c = "/i/redeem/status/" + b.tweetId, this.select("offersDialogIframeSelector").attr("src", c), this.open())
        }, this.closeDialog = function(a, b) {
            this.select("offersDialogIframeSelector").attr("src", "about:blank"), this.select("offersDialogIframeSelector").css("height", this.attr.defaultHeight)
        }, this.adjustIframeHeight = function(a) {
            var b = {
                iframeSelector: this.attr.offersDialogIframeSelector,
                isQualified: function(a) {
                    return a.name && a.name === "offers" && a.height
                }
            };
            this.fitIframeHeight(a, b)
        }, this.offersActionClicked = function(a, b) {
            if (this.isEmailConfirmedOrHandle(b.el)) {
                var c = $(b.el).closest(this.attr.tweetSelector).data("tweet-id");
                this.trigger("uiNeedsOffersDialog", {
                    tweetId: c
                })
            }
        }, this.showSuccessMessage = function(a, b) {
            this.trigger("uiCloseDialog"), this.trigger("uiShowMessage", {
                message: b.message
            })
        }, this.after("initialize", function() {
            this.on(window, "message", this.adjustIframeHeight), this.on("uiDialogClosed", this.closeDialog), this.on("uiCommerceShowSuccessMessage", this.showSuccessMessage), this.on("uiNeedsOffersDialog", this.openDialog), this.on(document, "click", {
                offersActionSelector: this.offersActionClicked
            })
        }), this.before("teardown", function() {
            this.off(window, "message", this.adjustIframeHeight)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withPosition = require("app/ui/with_position"), withEmailConfirmation = require("app/ui/commerce/with_email_confirmation"), withIframeHeightAdjuster = require("app/utils/with_iframe_height_adjuster"), OffersConfirmDialog = defineComponent(offersConfirmDialog, withDialog, withPosition, withIframeHeightAdjuster, withEmailConfirmation);
    module.exports = OffersConfirmDialog
});
define("app/ui/page_title", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function pageTitle() {
        this.addCount = function(a, b) {
            this.trigger(document, {
                type: "uiUpdatePageCount",
                defaultBehavior: this.updateCount.bind(this, b)
            })
        }, this.updateCount = function(a) {
            a.count && (document.title = "(" + a.count + ") " + this.title)
        }, this.removeCount = function(a, b) {
            document.title = this.title
        }, this.setTitle = function(a, b) {
            var c = b || a.originalEvent.state;
            c && c.title && (document.title = this.title = c.title)
        }, this.after("initialize", function() {
            this.title = document.title, this.on("uiAddPageCount", this.addCount), this.on("uiHasInjectedNewTimeline", this.removeCount), this.on(document, "uiBeforePageChanged uiUpdatePageTitle", this.setTitle)
        })
    }
    var defineComponent = require("core/component"), PageTitle = defineComponent(pageTitle);
    module.exports = PageTitle
});
define("app/ui/page_visibility", ["module", "require", "exports", "core/clock", "core/component"], function(module, require, exports) {
    function PageVisibility() {
        this.defaultAttrs({
            heartBeat: 0,
            doc: document,
            isIe: $.browser.msie
        });
        var a, b=!1;
        this.visibilityChange = function(c) {
            var d;
            switch (c.type) {
            case"focus":
            case"focusin":
                d=!1;
                break;
            case"blur":
            case"focusout":
                d=!0;
                break;
            case"beforeunload":
                this.trigger("uiPageUnload");
                break;
            default:
                a && (d = this.attr.doc[a])
            }
            d===!0 ? (b=!0, this.trigger("uiPageHidden")) : d===!1 && (b=!1, this.trigger("uiPageVisible"))
        }, this.focusTick = function() {
            b || this.trigger("uiPageVisible")
        }, this.after("initialize", function() {
            var b = {
                hidden: "visibilitychange",
                mozHidden: "mozvisibilitychange",
                webkitHidden: "webkitvisibilitychange",
                msHidden: "msvisibilitychange"
            }, c=!1;
            for (var d in b)
                d in this.attr.doc && (a = d, this.on(this.attr.doc, b[a], this.visibilityChange), c=!0);
            c || (this.attr.isIe ? (this.on(this.attr.doc, "focusin", this.visibilityChange), this.on(this.attr.doc, "focusout", this.visibilityChange)) : (this.on(window, "focus", this.visibilityChange), this.on(window, "blur", this.visibilityChange))), this.on(window, "beforeunload", this.visibilityChange), this.attr.heartBeat && (this.timer = clock.setIntervalEvent("uiPageFocusTick", this.attr.heartBeat), this.on("uiPageFocusTick", this.focusTick))
        })
    }
    var clock = require("core/clock"), defineComponent = require("core/component");
    module.exports = defineComponent(PageVisibility)
});
define("app/ui/dialogs/payment_order_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog"], function(module, require, exports) {
    function paymentsOrderDialog() {
        this.defaultAttrs({
            isLoadingClass: "PaymentOrderDialog-isLoading",
            isLoadedClass: "PaymentOrderDialog-isLoaded",
            isFailedClass: "PaymentOrderDialog-isFailed",
            containerSelector: ".PaymentOrderDialog-orderDetails",
            spinnerSelector: ".spinner-bigger",
            errorAlertSelector: ".alert"
        }), this.setLoading = function() {
            this.select("modalBodySelector").addClass(this.attr.isLoadingClass).removeClass(this.attr.isLoadedClass).removeClass(this.attr.isFailedClass)
        }, this.setLoaded = function() {
            this.select("modalBodySelector").removeClass(this.attr.isLoadingClass).addClass(this.attr.isLoadedClass).removeClass(this.attr.isFailedClass)
        }, this.setFailed = function() {
            this.select("modalBodySelector").removeClass(this.attr.isLoadingClass).removeClass(this.attr.isLoadedClass).addClass(this.attr.isFailedClass)
        }, this.openDialog = function(a, b) {
            this.setLoading(), this.open(), this.trigger("uiShowPaymentOrderDetail", {
                orderId: b.orderId
            })
        }, this.populateDialog = function(a, b) {
            this.setLoaded(), this.select("modalBodySelector").addClass(this.attr.isLoadedClass), this.select("containerSelector").html(b.html), this.position()
        }, this.after("initialize", function() {
            this.on(document, "uiRequestPaymentOrderDetail", this.openDialog), this.on(document, "dataCommercePaymentOrderDetailFetched", this.populateDialog), this.on(document, "dataCommercePaymentOrderDetailFetchFailed", this.setFailed)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), PaymentsOrderDialog = defineComponent(paymentsOrderDialog, withDialog);
    module.exports = PaymentsOrderDialog
});
define("app/data/with_perftown_scribe", ["module", "require", "exports", "app/data/scribe_transport", "app/utils/time"], function(module, require, exports) {
    function withPerftownScribe() {
        this.scribeTransport = scribeTransport, this.scribePerftown = function(a, b, c) {
            var d = {
                product: "webclient",
                description: a,
                duration_ms: b,
                start_time_ms: this.startTime,
                metadata: JSON.stringify(c)
            };
            this.scribeTransport.send(d, "perftown")
        }, this.after("initialize", function() {
            this.startTime = time.now()
        })
    }
    var scribeTransport = require("app/data/scribe_transport"), time = require("app/utils/time");
    module.exports = withPerftownScribe
});
define("app/data/performance_stats_scribe", ["module", "require", "exports", "core/component", "app/data/with_perftown_scribe", "app/data/user_info", "app/utils/browser"], function(module, require, exports) {
    function PerformanceStatsScribe() {
        this.scribeAjaxDuration = function(a, b) {
            if (!b.statsName)
                throw new Error("statsName is required for stats logging.");
            var c = ["stats", "ajax", "perf", b.statsName], d = b.requestData || {};
            d.interval ? c.push("polling") : d.min_position || d.since_id ? c.push("newer") : (d.max_position || d.max_id) && c.push("older"), b.status && c.push(b.status);
            var e = c.join(":"), f = {
                url: b.url,
                browser: browser($.browser),
                response_size: (b.responseText || {}).length
            };
            this.scribePerftown(e, b.duration, f)
        }, this.after("initialize", function() {
            userInfo.getDecider("web_perftown_stats") && this.on("dataAjaxDuration", this.scribeAjaxDuration)
        })
    }
    var defineComponent = require("core/component"), withPerftownScribe = require("app/data/with_perftown_scribe"), userInfo = require("app/data/user_info"), browser = require("app/utils/browser");
    module.exports = defineComponent(PerformanceStatsScribe, withPerftownScribe)
});
define("app/data/lists", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function listsData() {
        this.listMembershipContent = function(a, b) {
            this.get({
                url: "/i/" + b.userId + "/lists",
                dataType: "json",
                data: {},
                eventData: b,
                success: "dataGotListMembershipContent",
                error: "dataFailedToGetListMembershipContent"
            })
        }, this.addUserToList = function(a, b) {
            this.post({
                url: "/i/" + b.userId + "/lists/" + b.listId + "/members",
                dataType: "json",
                data: {},
                eventData: b,
                success: "dataDidAddUserToList",
                error: "dataFailedToAddUserToList"
            })
        }, this.removeUserFromList = function(a, b) {
            this.destroy({
                url: "/i/" + b.userId + "/lists/" + b.listId + "/members",
                dataType: "json",
                data: {},
                eventData: b,
                success: "dataDidRemoveUserFromList",
                error: "dataFailedToRemoveUserFromList"
            })
        }, this.createList = function(a, b) {
            this.post({
                url: "/i/lists/create",
                dataType: "json",
                data: b,
                eventData: b,
                success: "dataDidCreateList",
                error: "dataFailedToCreateList"
            })
        }, this.after("initialize", function() {
            this.on("uiNeedsListMembershipContent", this.listMembershipContent), this.on("uiAddUserToList", this.addUserToList), this.on("uiRemoveUserFromList", this.removeUserFromList), this.on("uiCreateList", this.createList)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data");
    module.exports = defineComponent(listsData, withData)
});
define("app/data/profile_edit_btn_scribe", ["module", "require", "exports", "core/component", "core/utils", "app/data/with_scribe"], function(module, require, exports) {
    function profileEditBtnScribe() {
        this.defaultAttrs({
            editButtonSelector: ".edit-profile-btn",
            scribeContext: {}
        }), this.scribeAction = function(a) {
            var b = utils.merge(this.attr.scribeContext, {
                action: a
            });
            return function(a, c) {
                b.element = $(a.target).attr("data-scribe-element"), this.scribe(b)
            }
        }, this.after("initialize", function() {
            this.on("click", {
                editButtonSelector: this.scribeAction("click")
            })
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(profileEditBtnScribe, withScribe)
});
define("app/ui/with_hover_card", ["module", "require", "exports", "core/utils", "core/compose", "app/ui/with_position"], function(module, require, exports) {
    function withHoverCard() {
        compose.mixin(this, [withPosition]), this.defaultAttrs({
            hoverContentSelector: ".hovercard",
            hoverContainerClasses: "hovercard",
            showDelay: 400,
            hideDelay: 50
        }), this.prepareHover = function(a) {
            this.currentKey !== a && this.forceHide(), this.openTimer = setTimeout(this.timerSaysOK.bind(this), this.attr.showDelay)
        }, this.timerSaysOK = function() {
            this.okToOpen=!0, this.queuedOpen && this.queuedOpen()
        }, this.showHTML = function(a, b, c, d) {
            d = d || {}, d.id = a;
            if (this.currentKey == a && this.timeout) {
                clearTimeout(this.timeout), this.timeout = null;
                return 
            }
            if (!this.okToOpen) {
                this.queuedOpen = function() {
                    this.showHTML(a, b, c, d)
                };
                return 
            }
            this.queuedOpen = null, this.forceHide(), this.currentKey = a, this.$hoverContent = $(b), this.$hoverContent.addClass(this.attr.hoverContainerClasses), this.$node.html(this.$hoverContent), this.pos = this.adjacent(this.$node, $(c), {
                gravity: "vertical-north"
            }), this.$hoverContent.addClass("gravity-" + this.pos.gravity), this.pos.gravity == "south" && (this.pos.top -= 5), this.$node.css({
                top: this.pos.top,
                left: this.pos.left
            });
            var e = this.$hoverContent.find("img.banner");
            e.length > 0 ? e.load(function() {
                this.$node.show(), this.trigger("uiHoverShown", d)
            }.bind(this)) : (this.$node.show(), this.trigger("uiHoverShown", d))
        }, this.forceHide = function() {
            this.timeout && (clearTimeout(this.timeout), this.timeout = null), this.openTimer && (clearTimeout(this.openTimer), this.openTimer = null), this.currentKey = null, this.$hoverContent && (this.$node.hide(), this.$hoverContent.remove(), this.$hoverContent = null)
        }, this.hide = function() {
            this.okToOpen=!1, this.queuedOpen = null, clearTimeout(this.openTimer), this.openTimer = null, this.timeout = setTimeout(this.forceHide.bind(this), this.attr.hideDelay)
        }, this.mouseOverContent = function() {
            if (this.timeout) {
                clearTimeout(this.timeout), this.timeout = null;
                return 
            }
        }, this.after("initialize", function() {
            this.on(document, "uiBeforePageChanged uiShortcutEsc", this.forceHide), this.on(document, "mouseover", {
                hoverContentSelector: this.mouseOverContent
            }), this.on(document, "mouseout", {
                hoverContentSelector: this.hide
            })
        })
    }
    var utils = require("core/utils"), compose = require("core/compose"), withPosition = require("app/ui/with_position");
    module.exports = withHoverCard
});
define("app/ui/profile_hovercard", ["module", "require", "exports", "core/component", "app/ui/with_hover_card", "app/ui/with_user_actions", "app/ui/navigation_links"], function(module, require, exports) {
    function profileHovercard() {
        this.defaultAttrs({
            modalContainerSelector: ".modal-container",
            profileLinkSelector: "a.profile-picture, .fullname a",
            itemType: "user"
        }), this.openProfileHover = function(a, b) {
            b.screenName && delete b.userId;
            if (this.skipScreenName && b.screenName && b.screenName.toLowerCase() == this.skipScreenName.toLowerCase() || this.skipUserId && b.userId == this.skipUserId)
                return;
            b.isHoverRequest=!0;
            if (a.target && $(a.target).closest(this.attr.modalContainerSelector).length > 0)
                return;
            this.refEventData = b, this.refEl = a.target, this.lastRequested = b.screenName ? b.screenName.toLowerCase() : b.userId, this.prepareHover(this.lastRequested), this.$node.attr("data-associated-tweet-id", b.tweetId || null), this.$node.attr("data-impression-id", b.impressionId || null), this.$node.attr("data-disclosure-type", b.disclosureType || null), this.$node.attr("data-impression-cookie", b.impressionCookie || null), this.trigger("uiWantsProfilePopup", b)
        }, this.dataAvailable = function(a, b) {
            if ((!b.screen_name || this.lastRequested != b.screen_name.toLowerCase()) && this.lastRequested != b.user_id)
                return;
            this.$node.attr("data-screen-name", b.screen_name || null), this.$node.attr("data-user-id", b.user_id || null), this.showHTML(b.user_id, b.html, $(this.refEl), this.refEventData)
        }, this.after("initialize", function(a, b) {
            NavigationLinks.attachTo(this.$node, {
                eventData: {
                    scribeContext: {
                        component: "profile_hover"
                    }
                }
            }), b.profile_user && (this.skipScreenName = b.profile_user.screen_name, this.skipUserId = b.profile_user.id_str), this.on(document, "dataProfilePopupSuccess", this.dataAvailable), this.on(document, "uiShowProfileHover", this.openProfileHover), this.on(document, "uiHideProfileHover", this.hide)
        })
    }
    var defineComponent = require("core/component"), withHoverCard = require("app/ui/with_hover_card"), withUserActions = require("app/ui/with_user_actions"), NavigationLinks = require("app/ui/navigation_links"), ProfileHovercard = defineComponent(profileHovercard, withHoverCard, withUserActions);
    module.exports = ProfileHovercard
});
define("app/ui/with_profile_stats", ["module", "require", "exports"], function(module, require, exports) {
    function withProfileStats() {
        this.defaultAttrs({}), this.updateProfileStats = function(a, b) {
            if (!b.stats ||!b.stats.length)
                return;
            $.each(b.stats, function(a, b) {
                var c = this.statNode(b.user_id, b.stat);
                this.isCompact(c) || c.html(b.html)
            }.bind(this))
        }, this.statSelector = function(a, b) {
            return '.stats[data-user-id="' + a + '"] a[data-element-term="' + b + '_stats"]'
        }, this.statNode = function(a, b) {
            return this.$node.find(this.statSelector(a, b))
        }, this.isCompact = function(a) {
            return a.data("is-compact")===!0
        }, this.after("initialize", function() {
            this.on(document, "dataGotProfileStats", this.updateProfileStats)
        })
    }
    module.exports = withProfileStats
});
define("app/ui/with_handle_overflow", ["module", "require", "exports"], function(module, require, exports) {
    function withHandleOverflow() {
        this.defaultAttrs({
            heightOverflowClassName: "height-overflow"
        }), this.checkForOverflow = function(a) {
            a = a || this.$node;
            if (!a ||!a.length)
                return;
            a[0].scrollHeight > a.height() ? a.addClass(this.attr.heightOverflowClassName) : a.removeClass(this.attr.heightOverflowClassName)
        }
    }
    module.exports = withHandleOverflow
});
define("app/ui/profile_popup", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/ui/with_user_actions", "app/ui/with_item_actions", "app/ui/with_profile_stats", "app/ui/with_handle_overflow"], function(module, require, exports) {
    function profilePopup() {
        this.defaultAttrs({
            dialogContentSelector: ".profile-modal",
            profileHeaderInnerSelector: ".profile-header-inner",
            socialProofSelector: ".social-proof",
            tweetSelector: ".simple-tweet",
            locationSelector: ".location a",
            slideDuration: 100,
            top: 47,
            bottom: 10,
            tweetMinimum: 2,
            itemType: "user"
        }), this.slideInContent = function(a) {
            var b = this.$dialog.height(), c = $(a);
            this.addHeaderImage(c), this.$contentContainer.html(c), this.$node.addClass("has-content"), this.removeTweets();
            var d = this.$dialog.height();
            this.$dialog.height(b), this.$dialog.animate({
                height: d
            }, this.attr.slideDuration, this.slideInComplete.bind(this))
        }, this.removeTweets = function() {
            var a = this.select("tweetSelector");
            for (var b = a.length - 1; b > this.attr.tweetMinimum - 1; b--) {
                if (!this.isTooTall())
                    return;
                a.eq(b).remove()
            }
        }, this.getWindowHeight = function() {
            return $(window).height()
        }, this.isTooTall = function() {
            return this.$dialog.height() + this.attr.top + this.attr.bottom > this.getWindowHeight()
        }, this.addHeaderImage = function(a) {
            var b = a.find(this.attr.profileHeaderInnerSelector);
            b.hasClass("no-image") ? b.css("background-color", b.attr("data-background-color")) : b.css("background-image", b.attr("data-background-image"))
        }, this.slideInComplete = function() {
            this.checkForOverflow(this.select("profileHeaderInnerSelector")), this.trigger(this.$dialog, "uiDialogContentChanged")
        }, this.clearPopup = function() {
            this.$dialog.height("auto"), this.$node.removeClass("muting"), this.$contentContainer.empty()
        }, this.openProfilePopup = function(a, b) {
            b.screenName && delete b.userId;
            if (b.userId && b.userId === this.currentUserId() || b.screenName && b.screenName === this.currentScreenName())
                return;
            this.open(), this.clearPopup(), this.$node.removeClass("has-content"), this.$node.attr("data-associated-tweet-id", b.tweetId || null), this.$node.attr("data-impression-id", b.impressionId || null), this.$node.attr("data-disclosure-type", b.disclosureType || null), this.$node.attr("data-impression-cookie", b.impressionCookie || null), this.trigger("uiWantsProfilePopup", b)
        }, this.closeProfilePopup = function(a) {
            this.clearPopup(), this.trigger("uiCloseProfilePopup", {
                userId: this.currentUserId(),
                screenName: this.currentScreenName()
            })
        }, this.fillProfile = function(a, b) {
            this.$node.attr("data-screen-name", b.screen_name || null), this.$node.attr("data-user-id", b.user_id || null), this.slideInContent(b.html)
        }, this.removeSocialProof = function(a, b) {
            this.select("socialProofSelector").remove()
        }, this.addSocialProof = function(a, b) {
            b.html ? (this.select("socialProofSelector").html(b.html), this.trigger("uiHasPopupSocialProof")) : this.removeSocialProof()
        }, this.showError = function(a, b) {
            var c = ['<div class="profile-modal-header error"><p>', b.message, "</p></div>"].join("");
            this.slideInContent(c)
        }, this.getPopupData = function(a) {
            return !this.isOpen() ||!this.$node.hasClass("has-content") ? null : this.$node.attr(a)
        }, this.currentScreenName = function() {
            return this.getPopupData("data-screen-name")
        }, this.currentUserId = function() {
            return this.getPopupData("data-user-id")
        }, this.locationClicked = function(a, b) {
            var c = $(a.target), d = c.data("place-id"), e = {
                scribeData: {
                    message: d
                }
            };
            this.trigger("uiPopupLocationClicked", e)
        }, this.after("initialize", function() {
            this.$contentContainer = this.select("dialogContentSelector"), this.on(document, "uiShowProfilePopup", this.openProfilePopup), this.on(document, "dataProfilePopupSuccess", this.fillProfile), this.on(document, "dataProfilePopupFailure", this.showError), this.on(document, "dataSocialProofSuccess", this.addSocialProof), this.on(document, "dataSocialProofFailure", this.removeSocialProof), this.on(document, "uiOpenConfirmDialog uiNeedsBlockOrReportDialog uiOpenTweetDialog uiNeedsDMDialog uiListAction uiOpenSigninOrSignupDialog uiEmbedProfileAction", this.close), this.on("uiDialogClosed", this.closeProfilePopup), this.on("click", {
                locationSelector: this.locationClicked
            })
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withUserActions = require("app/ui/with_user_actions"), withItemActions = require("app/ui/with_item_actions"), withProfileStats = require("app/ui/with_profile_stats"), withHandleOverflow = require("app/ui/with_handle_overflow"), ProfilePopup = defineComponent(profilePopup, withDialog, withUserActions, withItemActions, withProfileStats, withHandleOverflow);
    module.exports = ProfilePopup
});
define("app/data/profile_popup", ["module", "require", "exports", "core/component", "core/utils", "app/data/with_data"], function(module, require, exports) {
    function profilePopupData() {
        this.defaultAttrs({
            profileHoversEnabled: !1,
            hoverFetchDelay: 100,
            noShowError: !0,
            statsName: "swift_profile_popup"
        }), this.userCache = {
            screenNames: Object.create(null),
            ids: Object.create(null)
        }, this.socialProofCache = {
            screenNames: Object.create(null),
            ids: Object.create(null)
        }, this.saveToCache = function(a, b) {
            a.ids[b.user_id] = b, a.screenNames[b.screen_name] = b
        }, this.retrieveFromCache = function(a, b) {
            var c;
            return b.userId ? c = a.ids[b.userId] : b.user_id ? c = a.ids[b.user_id] : b.screenName ? c = a.screenNames[b.screenName] : b.screen_name && (c = a.screenNames[b.screen_name]), c
        }, this.invalidateCaches = function(a, b) {
            var c, d, e;
            b.userId ? (c = b.userId, e = this.userCache.ids[c], d = e && e.screen_name) : (d = b.screenName, e = this.userCache.screenNames[d], c = e && e.user_id), c && delete this.userCache.ids[c], c && delete this.socialProofCache.ids[c], d && delete this.userCache.screenNames[d], d && delete this.socialProofCache.screenNames[d]
        }, this.getProfilePopupMain = function(a, b) {
            var c = function(a) {
                this.saveToCache(this.userCache, a), this.trigger("dataProfilePopupSuccess", a);
                var b = this.retrieveFromCache(this.socialProofCache, a);
                b && this.trigger("dataSocialProofSuccess", b)
            }.bind(this), d = function(a) {
                this.trigger("dataProfilePopupFailure", a)
            }.bind(this), e = this.retrieveFromCache(this.userCache, a);
            if (e) {
                e.sourceEventData = a, c(e);
                return 
            }
            var f = function() {
                this.pendingRemoteRequest = null, this.get({
                    url: "/i/profiles/popup",
                    data: b,
                    eventData: a,
                    cache: !1,
                    success: c,
                    error: d
                })
            }.bind(this);
            b.wants_hovercard ? this.pendingRemoteRequest = setTimeout(f, this.attr.hoverFetchDelay) : f()
        }, this.getProfilePopup = function(a, b) {
            var c = {};
            b.screenName ? c.screen_name = b.screenName : b.userId && (c.user_id = b.userId), this.attr.profileHoversEnabled && b.isHoverRequest && (c.wants_hovercard=!0), this.getProfilePopupMain(b, c)
        }, this.hideProfileHover = function() {
            this.pendingRemoteRequest && (clearTimeout(this.pendingRemoteRequest), this.pendingRemoteRequest = null)
        }, this.after("initialize", function() {
            this.on("uiWantsProfilePopup", this.getProfilePopup), this.on("uiHideProfileHover", this.hideProfileHover), this.on(document, "dataFollowStateChange dataUserActionSuccess dataDidMuteUser dataDidUnmuteUser", this.invalidateCaches)
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), withData = require("app/data/with_data"), ProfilePopupData = defineComponent(profilePopupData, withData);
    module.exports = ProfilePopupData
});
define("app/data/profile_popup_scribe", ["module", "require", "exports", "core/component", "core/utils", "app/data/with_interaction_data_scribe", "app/data/client_event"], function(module, require, exports) {
    function profilePopupScribe() {
        this.defaultAttrs({
            scribeContext: {
                component: "profile_dialog"
            },
            profileHoversEnabled: !1
        }), this.scribeProfilePopupOpen = function(a, b) {
            if (this.attr.profileHoversEnabled) {
                if (a.target.id != "profile-hover-container")
                    return;
                b.user_id = $(a.target).attr("data-user-id")
            }
            if (this.clientEvent.scribeContext.page != "profile" ||!this.clientEvent.scribeData.profile_id)
                this.clientEvent.scribeData.profile_id = b.user_id;
            var c = utils.merge(this.attr.scribeContext, {
                action: "open"
            });
            this.scribe(c, b)
        }, this.cleanupProfilePopupScribing = function(a, b) {
            this.clientEvent.scribeData.profile_id && this.clientEvent.scribeContext.page != "profile" && delete this.clientEvent.scribeData.profile_id
        }, this.scribePopupSocialProof = function(a, b) {
            var c = utils.merge(this.attr.scribeContext, {
                element: "social_proof",
                action: "impression"
            });
            this.scribe(c, b)
        }, this.after("initialize", function() {
            this.clientEvent = clientEvent, this.attr.profileHoversEnabled ? (this.attr.scribeContext = {
                component: "profile_hover"
            }, this.on(document, "uiHoverShown", this.scribeProfilePopupOpen)) : (this.on(document, "uiCloseProfilePopup", this.cleanupProfilePopupScribing), this.on(document, "uiHasPopupSocialProof", this.scribePopupSocialProof), this.on(document, "dataProfilePopupSuccess", this.scribeProfilePopupOpen), this.scribeOnEvent("uiPopupLocationClicked", {
                element: "place_tag",
                action: "click"
            }))
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), withInteractionDataScribe = require("app/data/with_interaction_data_scribe"), clientEvent = require("app/data/client_event");
    module.exports = defineComponent(profilePopupScribe, withInteractionDataScribe)
});
define("app/data/user", ["module", "require", "exports", "core/component", "core/utils", "app/utils/cookie", "app/data/with_data"], function(module, require, exports) {
    function userData() {
        this.defaultAttrs({
            millisecInDay: 864e5,
            ageGateingErrorCode: 250,
            ageGateCookieKey: "age-gated",
            injectTweetOnFollow: !1
        }), this.hasBeenAgeGatedWithin24Hr = function() {
            var a = cookie(this.attr.ageGateCookieKey);
            return a ? (new Date).getTime() - parseInt(a, 10) < this.attr.millisecInDay : !1
        }, this.updateFollowStatus = function(a, b) {
            function c(c) {
                c.message && this.trigger("uiShowMessage", c), this.trigger("dataFollowStateChange", utils.merge(c, a, {
                    userId: a.userId,
                    newState: c.new_state,
                    requestUrl: b,
                    isFollowBack: c.is_follow_back
                })), this.trigger(document, "dataGotProfileStats", {
                    stats: c.profile_stats
                })
            }
            function d(b) {
                var c = a.userId, d = a.originalFollowState, e = b && b.data && b.data.code == this.attr.ageGateingErrorCode;
                b.new_state && (d = b.new_state), this.trigger("dataFollowStateChange", utils.merge(b, {
                    userId: c,
                    newState: d
                })), e && this.trigger("uiNeedsAgeGateDialog", {
                    originalData: a,
                    newState: d,
                    ageGated: this.hasBeenAgeGatedWithin24Hr(),
                    minBirthDate: parseInt(b.data.timestamp, 10)
                })
            }
            var e = a.disclosureType ? a.disclosureType == "earned": undefined, f = {
                user_id: a.userId,
                impression_id: a.impressionId,
                earned: e,
                fromShortcut: a.fromShortcut,
                handles_challenges: 1,
                inject_tweet: this.attr.injectTweetOnFollow && a.inject_tweet
            };
            a.lifeline && (f.lifeline = 1), f.challenges_passed=!!a.passedAgeGating, this.post({
                url: b,
                data: f,
                eventData: a,
                success: c.bind(this),
                error: d.bind(this),
                retryIfUnavailable: !0
            })
        }, this.reversibleAjaxCall = function(a, b, c) {
            function d(c) {
                this.trigger("dataUserActionSuccess", $.extend({}, c, {
                    userId: a.userId,
                    requestUrl: b
                })), c.message && this.trigger("uiShowMessage", c)
            }
            function e(b) {
                this.trigger(c, a)
            }
            this.post({
                url: b,
                data: {
                    user_id: a.userId,
                    impression_id: a.impressionId
                },
                eventData: a,
                success: d.bind(this),
                error: e.bind(this)
            })
        }, this.normalAjaxCall = function(a, b, c) {
            function d(c) {
                this.trigger("dataUserActionSuccess", $.extend({}, c, {
                    userId: a.userId,
                    requestUrl: b
                })), c.message && this.trigger("uiShowMessage", c)
            }
            var e = $.extend({
                user_id: a.userId,
                token: a.feedbackToken,
                impression_id: a.impressionId
            }, c);
            this.post({
                url: b,
                data: e,
                eventData: a,
                success: d.bind(this),
                error: "dataUserActionError"
            })
        }, this.followAction = function(a, b) {
            var c = "/i/user/follow";
            this.updateFollowStatus(b, c)
        }, this.unfollowAction = function(a, b) {
            var c = "/i/user/unfollow";
            this.updateFollowStatus(b, c)
        }, this.cancelAction = function(a, b) {
            var c = "/i/user/cancel";
            this.updateFollowStatus(b, c)
        }, this.blockAction = function(a, b) {
            var c = "/i/user/block";
            this.updateFollowStatus(b, c)
        }, this.unblockAction = function(a, b) {
            var c = "/i/user/unblock";
            this.updateFollowStatus(b, c)
        }, this.reportSpamAction = function(a, b) {
            this.normalAjaxCall(b, "/i/user/report_spam", {
                screen_name: b.screenName,
                report_type: b.reportType,
                block_user: b.blockUser,
                impression_id: b.impressionId
            })
        }, this.hideSuggestionAction = function(a, b) {
            this.normalAjaxCall(b, "/i/user/hide")
        }, this.retweetOnAction = function(a, b) {
            this.reversibleAjaxCall(b, "/i/user/retweets_on", "dataRetweetOffAction")
        }, this.retweetOffAction = function(a, b) {
            this.reversibleAjaxCall(b, "/i/user/retweets_off", "dataRetweetOnAction")
        }, this.deviceNotificationsOnAction = function(a, b) {
            this.reversibleAjaxCall(b, "/i/user/device_notifications_on", "dataDeviceNotificationsOffAction")
        }, this.deviceNotificationsOffAction = function(a, b) {
            this.reversibleAjaxCall(b, "/i/user/device_notifications_off", "dataDeviceNotificationsOnAction")
        }, this.after("initialize", function() {
            this.on(document, "uiFollowAction", this.followAction), this.on(document, "uiUnfollowAction", this.unfollowAction), this.on(document, "uiCancelFollowRequestAction", this.cancelAction), this.on(document, "uiBlockAction", this.blockAction), this.on(document, "uiUnblockAction", this.unblockAction), this.on(document, "uiReportSpamAction uiReportUserAction", this.reportSpamAction), this.on(document, "uiHideSuggestionAction", this.hideSuggestionAction), this.on(document, "uiRetweetOnAction", this.retweetOnAction), this.on(document, "uiRetweetOffAction", this.retweetOffAction), this.on(document, "uiDeviceNotificationsOnAction", this.deviceNotificationsOnAction), this.on(document, "uiDeviceNotificationsOffAction", this.deviceNotificationsOffAction), this.on(document, "uiAgeGatePassed", this.followAction)
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), cookie = require("app/utils/cookie"), withData = require("app/data/with_data"), UserData = defineComponent(userData, withData);
    module.exports = UserData
});
define("app/boot/profile_popup", ["module", "require", "exports", "app/data/lists", "app/data/profile_edit_btn_scribe", "app/ui/profile_hovercard", "app/ui/profile_popup", "app/data/profile_popup", "app/data/profile_popup_scribe", "app/data/user", "core/utils"], function(module, require, exports) {
    function initialize(a) {
        ProfileHovercard.attachTo("#profile-hover-container", utils.merge(a, {
            eventData: {
                scribeContext: {
                    component: "profile_hover"
                }
            }
        })), ProfilePopupData.attachTo(document, utils.merge(a, {
            eventData: {
                scribeContext: {
                    component: "profile_dialog"
                }
            }
        })), UserData.attachTo(document, a), Lists.attachTo(document, a), ProfilePopup.attachTo("#profile_popup", utils.merge(a, {
            eventData: {
                scribeContext: {
                    component: "profile_dialog"
                }
            }
        })), ProfileEditBtnScribe.attachTo("#profile_popup", {
            scribeContext: {
                component: "profile_dialog"
            }
        }), ProfilePopupScribe.attachTo(document, a)
    }
    var Lists = require("app/data/lists"), ProfileEditBtnScribe = require("app/data/profile_edit_btn_scribe"), ProfileHovercard = require("app/ui/profile_hovercard"), ProfilePopup = require("app/ui/profile_popup"), ProfilePopupData = require("app/data/profile_popup"), ProfilePopupScribe = require("app/data/profile_popup_scribe"), UserData = require("app/data/user"), utils = require("core/utils"), hasPopup = $("#profile_popup").length > 0;
    module.exports = hasPopup ? initialize : $.noop
});
define("app/ui/responsive_dashboard_width", ["module", "require", "exports", "core/component", "core/utils", "core/clock"], function(module, require, exports) {
    function responsiveDashboardWidth() {
        this.defaultAttrs({
            leftDashboardSelector: ".dashboard-left",
            rightDashboardSelector: ".dashboard-right",
            roamingSelector: ".roaming-module",
            threeColWidth: 1236
        }), this.setWindowWidth = function(a) {
            this.windowWidth = this.$window.width();
            if (this.windowWidth < this.attr.threeColWidth) {
                if (this.threeCol || a)
                    this.$dashboardLeft.append(this.$dashboardRight.find(this.attr.roamingSelector)), this.$body.removeClass("three-col"), this.threeCol=!1
            } else if (!this.threeCol || a)
                this.$dashboardRight.append(this.$dashboardLeft.find(this.attr.roamingSelector)), this.$body.addClass("three-col"), this.threeCol=!0
        }, this.after("initialize", function() {
            this.$dashboardRight = this.select("rightDashboardSelector"), this.$dashboardRight.length && (this.$window = $(window), this.$body = $(document.body), this.$dashboardLeft = this.select("leftDashboardSelector"), this.threeCol=!1, this.setWindowWidth(!0), this.on(window, "resize", utils.debounce(this.setWindowWidth.bind(this), 100)))
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), clock = require("core/clock");
    module.exports = defineComponent(responsiveDashboardWidth)
});
define("app/ui/dialogs/retweet_dialog", ["module", "require", "exports", "core/component", "app/ui/with_dialog", "app/ui/dialogs/with_modal_tweet"], function(module, require, exports) {
    function retweetDialog() {
        this.defaults = {
            cancelSelector: ".cancel-action",
            retweetSelector: ".retweet-action"
        }, this.openRetweet = function(a, b) {
            this.sourceEventData = b, this.displayTweet(b.tweetId, {
                modal: "retweet"
            }), this.open()
        }, this.retweet = function() {
            this.trigger("uiDidRetweet", this.sourceEventData)
        }, this.retweetSuccess = function(a, b) {
            this.sourceEventData != undefined && this.trigger("uiDidRetweetSuccess", this.sourceEventData), this.close(), this.sourceEventData = undefined
        }, this.restoreFocusToTweet = function(a) {
            $(a.target).is(this.$dialog) && this.activeEl && (a.preventDefault(), this.trigger($(this.activeEl).closest(".tweet"), "uiShouldAddFocusStyle"), this.activeEl.focus())
        }, this.after("initialize", function() {
            this.on("click", {
                cancelSelector: this.close,
                retweetSelector: this.retweet
            }), this.on(document, "uiOpenRetweetDialog", this.openRetweet), this.on(document, "dataDidRetweet", this.retweetSuccess), this.on(document, "uiDialogRestorePreviousFocus", this.restoreFocusToTweet)
        })
    }
    var defineComponent = require("core/component"), withDialog = require("app/ui/with_dialog"), withModalTweet = require("app/ui/dialogs/with_modal_tweet"), RetweetDialog = defineComponent(retweetDialog, withDialog, withModalTweet);
    module.exports = RetweetDialog
});
define("app/data/saved_searches", ["module", "require", "exports", "core/component", "app/data/with_data", "app/data/with_auth_token"], function(module, require, exports) {
    function savedSearches() {
        this.saveSearch = function(a, b) {
            this.post({
                url: "/i/saved_searches/create.json",
                data: b,
                headers: {
                    "X-PHX": !0
                },
                eventData: "",
                success: "dataAddedSavedSearch",
                error: "dataSavedSearchError"
            })
        }, this.removeSavedSearch = function(a, b) {
            this.post({
                url: "/i/saved_searches/destroy/" + encodeURIComponent(b.id) + ".json",
                data: "",
                headers: {
                    "X-PHX": !0
                },
                eventData: {
                    component: b.component
                },
                success: "dataRemovedSavedSearch",
                error: "dataSavedSearchError"
            })
        }, this.after("initialize", function() {
            this.on("uiAddSavedSearch", this.saveSearch), this.on("uiRemoveSavedSearch", this.removeSavedSearch)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), withAuthToken = require("app/data/with_auth_token");
    module.exports = defineComponent(savedSearches, withData, withAuthToken)
});
define("app/data/saved_searches_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function savedSearchesScribe() {
        this.scribeRemovedSavedSearch = function(a, b) {
            var c = "stream";
            b && b.sourceEventData && b.sourceEventData.component && (c = b.sourceEventData.component), this.scribe({
                action: "remove",
                component: c,
                element: "saved_search"
            })
        }, this.after("initialize", function() {
            this.scribeOnEvent("dataAddedSavedSearch", {
                action: "add",
                component: "stream",
                element: "saved_search"
            }), this.on("dataRemovedSavedSearch", this.scribeRemovedSavedSearch)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(savedSearchesScribe, withScribe)
});
define("app/data/sms_confirmation_data", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function smsConfirmationData() {
        this.smsConfirmationAjaxCall = function(a, b, c) {
            function d(a) {
                a.error ? this.trigger("uiAlertBanner", a) : this.trigger(c, {
                    message: a.message
                })
            }
            var e = $.extend({
                user_id: a.userId
            }, a);
            this.post({
                url: b,
                data: e,
                success: d.bind(this),
                error: "dataUserActionError"
            })
        }, this.sendPinAction = function(a, b) {
            var c = {
                phone_number: b.phone_number,
                country_code: b.country_code
            };
            this.smsConfirmationAjaxCall(c, "/i/devices/sms_confirmation_begin", "dataSendPinSuccess")
        }, this.resendPinAction = function(a, b) {
            var c = {
                phone_number: b.phone_number,
                country_code: b.country_code,
                resend: !0
            };
            this.smsConfirmationAjaxCall(c, "/i/devices/sms_confirmation_begin", "dataResendPinSuccess")
        }, this.verifyPinAction = function(a, b) {
            var c = {
                numeric_pin: b.numeric_pin,
                phone_number: b.phone_number,
                country_code: b.country_code
            };
            this.smsConfirmationAjaxCall(c, "/i/devices/sms_confirmation_complete", "dataVerifyPinSuccess")
        }, this.after("initialize", function() {
            this.on(document, "uiSendPinAction", this.sendPinAction), this.on(document, "uiResendPinAction", this.resendPinAction), this.on(document, "uiVerifyPinAction", this.verifyPinAction)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), SmsConfirmationData = defineComponent(smsConfirmationData, withData);
    module.exports = SmsConfirmationData
});
define("app/ui/forms/select_box", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function selectBox() {
        this.clickItems = function(a, b) {
            this.trigger("uiItemClick")
        }, this.itemSelect = function() {
            var a, b = {
                index: this.node.selectedIndex
            }, c, d;
            if (this.node.selectedIndex >= 0) {
                d = this.node.options[this.node.selectedIndex];
                for (c = 0; c < d.attributes.length; c++)
                    a = d.attributes[c].name, b[a] = d.attributes[c].value;
                b.text = d.innerHTML
            }
            this.trigger("uiItemSelect", b)
        }, this.setItems = function(a, b) {
            this.$node.empty(), b.items && b.items.forEach(function(a) {
                var b = $("<option />"), c;
                for (c in a)
                    c == "text" ? b.html(a[c]) : b.attr(c, a[c]);
                b.appendTo(this.$node)
            }.bind(this)), this.itemSelect()
        }, this.after("initialize", function() {
            this.node.jquery && (this.node = this.node[0]), this.attr.items && this.node.trigger("setItems", {
                items: this.attr.items
            }), this.on("setItems", this.setItems), this.on("change", this.itemSelect), this.on("mousedown", this.clickItems)
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(selectBox)
});
define("app/ui/with_phone_pin_verification", ["module", "require", "exports", "core/i18n", "app/ui/forms/select_box"], function(module, require, exports) {
    function withPhonePinVerification() {
        this.defaultAttrs({
            countryCodeDisplaySelector: "#country_code_display",
            countryCodeSelector: "#country_code",
            countryNameSelector: "#device_country_code",
            phoneNumberSelector: "#phone_number",
            pinCodeSelector: "#numeric_pin",
            pinCodeSelectorRaw: "#numeric_pin_raw",
            submitRegisterButtonSelector: "#send_verification_pin",
            submitVerifyButtonSelector: "#device_verify"
        }), this.updateCountryCode = function(a, b) {
            countryCode = b.value, this.select("countryCodeDisplaySelector").html("+" + countryCode), this.select("countryCodeSelector").val(countryCode), this.select("phoneNumberSelector").val() != "" && this.callbackIfValidPhoneNumber(this.clearInvalidPhoneMessage)
        }, this.callbackIfValidPhoneNumber = function(a) {
            var b = this.select("countryCodeSelector").val(), c = this.select("phoneNumberSelector").val(), d = this.validatePhoneNumber(b, c);
            d ? a.call(this) : this.showInvalidPhoneMessage(b)
        }, this.showInvalidPhoneMessage = function(a) {
            var b = _('US and Canadian phone numbers must have 11 digits.'), c = _('This appears to be an invalid phone number.'), d = a === "1" ? b: c;
            d += _(' Please enter a valid phone number.'), this.trigger("uiAlertBanner", {
                message: d
            })
        }, this.clearInvalidPhoneMessage = function() {
            this.trigger("uiAlertBannerClose")
        }, this.validatePhoneNumber = function(a, b) {
            var c = /\(*\)*\+*-*#* */g, d = b.replace(c, ""), e = /^\d{11}$/;
            return a != "1" && (e = /^\d{10,15}$/), e.test(a + d)
        }, this.enableIfValue = function(a) {
            return function(b, c) {
                var d = b.target || b.srcElement;
                this.select(a).attr("disabled", d.value === "")
            }
        }, this.stripPin = function(a) {
            return a.replace(/[^a-zA-Z0-9]/g, "")
        }, this.after("initialize", function() {
            SelectBox.attachTo(this.attr.countryNameSelector), this.on(this.attr.countryNameSelector, "uiItemSelect", this.updateCountryCode), this.on("input", {
                phoneNumberSelector: this.enableIfValue("submitRegisterButtonSelector"),
                pinCodeSelectorRaw: this.enableIfValue("submitVerifyButtonSelector")
            })
        })
    }
    var _ = require("core/i18n"), SelectBox = require("app/ui/forms/select_box");
    module.exports = withPhonePinVerification
});
define("app/ui/dialogs/sms_confirmation_dialog", ["module", "require", "exports", "core/component", "core/i18n", "app/ui/with_dialog", "app/ui/with_phone_pin_verification"], function(module, require, exports) {
    function smsConfirmationDialog() {
        this.defaultAttrs({
            phoneNumberInputModalSelector: "#sms-confirmation-begin-modal",
            pinVerificationModalSelector: "#sms-confirmation-complete-modal",
            alertBoxSelector: "#sms-alert-box",
            alertMessageSelector: "#sms-alert-message",
            alertBoxCloseSelector: "#sms-alert-close",
            authenticityTokenSelector: "#authenticity_token",
            resendCodeSelector: "#resend_code"
        }), this.setUpDialog = function(a, b) {
            this.reset(), this.open()
        }, this.reset = function() {
            this.select("pinVerificationModalSelector").hide(), this.select("phoneNumberSelector").val(""), this.savedPhoneNumber = "", this.savedCountryCode = ""
        }, this.afterOpen = function() {
            this.position(), this.select("alertBoxSelector").hide()
        }, this.alertBannerHandler = function(a, b) {
            this.showAlert(b.message)
        }, this.showAlert = function(a) {
            this.select("alertBoxSelector").show(), this.select("alertMessageSelector").text(a)
        }, this.hideAlert = function() {
            this.select("alertBoxSelector").hide(), this.select("alertMessageSelector").text("")
        }, this.sendPinCode = function(a, b) {
            a.preventDefault(), this.callbackIfValidPhoneNumber(function() {
                var a = this.select("phoneNumberSelector").val(), b = this.select("countryCodeSelector").val();
                this.savedPhoneNumber = a, this.savedCountryCode = b, this.trigger("uiSendPinAction", {
                    phone_number: a,
                    country_code: b
                })
            })
        }, this.resendPinCode = function() {
            this.trigger("uiResendPinAction", {
                phone_number: this.savedPhoneNumber,
                country_code: this.savedCountryCode
            })
        }, this.verifyPinCode = function(a, b) {
            a.preventDefault();
            var c = this.stripPin(this.select("pinCodeSelectorRaw").val());
            if (!c) {
                this.showAlert(_('Incorrect code. Please try again or request a new one.'));
                return 
            }
            this.trigger("uiVerifyPinAction", {
                numeric_pin: c,
                phone_number: this.savedPhoneNumber,
                country_code: this.savedCountryCode
            })
        }, this.sendPinSuccess = function(a, b) {
            this.select("phoneNumberInputModalSelector").remove(), this.hideAlert();
            var c = this.trimSavedPhoneNumber();
            this.select("pinVerificationModalSelector").find("#phone_number_to_verify").html(c), this.select("pinVerificationModalSelector").show(), this.trigger("uiVerifyPinFormImpression")
        }, this.trimSavedPhoneNumber = function() {
            var a = /\(*\)*\+*-*#* */g, b = this.savedPhoneNumber.replace(a, "");
            return this.savedCountryCode + b
        }, this.verifyPinSuccess = function(a, b) {
            this.isCompleted=!0, this.close(), this.trigger("uiShowMessage", {
                message: _('Your account is unlocked.')
            })
        }, this.resendPinSuccess = function(a, b) {
            this.showAlert(b.message)
        }, this.updateSmsConfirmationCountryCode = function(a, b) {
            this.trigger("uiSmsConfirmationCountryChange")
        }, this.phoneNumberInput = function(a, b) {
            this.startedPhoneInput || (this.trigger("uiSmsConfirmationPhoneNumberInput"), this.startedPhoneInput=!0)
        }, this.pinInput = function(a, b) {
            this.startedPinInput || (this.trigger("uiSmsConfirmationPinCodeInput"), this.startedPinInput=!0)
        }, this.around("close", function(a) {
            this.isOpen()&&!this.isCompleted && this.trigger("uiCancelSmsConfirmationDialog"), a()
        }), this.after("initialize", function() {
            this.on(document, "uiNeedsSmsConfirmationDialog", this.setUpDialog), this.on("uiDialogOpened", this.afterOpen), this.on(document, "dataSendPinSuccess", this.sendPinSuccess), this.on(document, "dataResendPinSuccess", this.resendPinSuccess), this.on(document, "dataVerifyPinSuccess", this.verifyPinSuccess);
            var a = this.attr.show_sms_confirmation_dialog;
            a && this.trigger("uiNeedsSmsConfirmationDialog"), this.on("click", {
                submitRegisterButtonSelector: this.sendPinCode,
                submitVerifyButtonSelector: this.verifyPinCode,
                alertBoxCloseSelector: this.clearInvalidPhoneMessage,
                resendCodeSelector: this.resendPinCode
            }), this.on("uiAlertBanner", this.alertBannerHandler), this.on("uiAlertBannerClose", this.hideAlert), this.on("submit", {
                phoneNumberInputModalSelector: this.sendPinCode,
                pinVerificationModalSelector: this.verifyPinCode
            }), this.on("change", {
                countryNameSelector: this.updateSmsConfirmationCountryCode
            }), this.on("input", {
                phoneNumberSelector: this.phoneNumberInput,
                pinCodeSelectorRaw: this.pinInput
            }), this.startedPhoneInput=!1, this.startedPinInput=!1, this.isCompleted=!1
        })
    }
    var defineComponent = require("core/component"), _ = require("core/i18n"), withDialog = require("app/ui/with_dialog"), withPhonePinVerification = require("app/ui/with_phone_pin_verification"), SmsConfirmationDialog = defineComponent(withDialog, withPhonePinVerification, smsConfirmationDialog);
    module.exports = SmsConfirmationDialog
});
define("app/data/sms_confirmation_scribe", ["module", "require", "exports", "core/component", "app/data/client_event", "app/data/with_scribe"], function(module, require, exports) {
    function smsConfirmationScribe() {
        this.scribeDialogImpression = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                action: "impression"
            }, b)
        }, this.scribeCountryChange = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "mobile_country_code",
                action: "change"
            }, b)
        }, this.scribePhoneNumberInput = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "phone_number",
                action: "change"
            }, b)
        }, this.scribeSendPinClick = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "activate_mobile",
                action: "click"
            }, b)
        }, this.scribeVerifyPinFormImpression = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "verify_pin",
                action: "impression"
            }, b)
        }, this.scribePinCodeInput = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "verify_pin",
                action: "input"
            }, b)
        }, this.scribeVerifyButtonClick = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "verify_pin",
                action: "click"
            }, b)
        }, this.scribeResendPinCodeButtonClick = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "resend_pin",
                action: "click"
            }, b)
        }, this.scribeDialogCancelClick = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                action: "cancel"
            }, b)
        }, this.scribeUnlockSuccessImpression = function(a, b) {
            this.scribe({
                component: "sms_confirmation_dialog",
                element: "verify_pin",
                action: "success"
            }, b)
        }, this.after("initialize", function() {
            this.on("dataVerifyPinSuccess", this.scribeUnlockSuccessImpression), this.on("uiCancelSmsConfirmationDialog", this.scribeDialogCancelClick), this.on("uiNeedsSmsConfirmationDialog", this.scribeDialogImpression), this.on("uiResendPinAction", this.scribeResendPinCodeButtonClick), this.on("uiSendPinAction", this.scribeSendPinClick), this.on("uiSmsConfirmationCountryChange", this.scribeCountryChange), this.on("uiSmsConfirmationPhoneNumberInput", this.scribePhoneNumberInput), this.on("uiSmsConfirmationPinCodeInput", this.scribePinCodeInput), this.on("uiVerifyPinAction", this.scribeVerifyButtonClick), this.on("uiVerifyPinFormImpression", this.scribeVerifyPinFormImpression)
        })
    }
    var defineComponent = require("core/component"), clientEvent = require("app/data/client_event"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(smsConfirmationScribe, withScribe)
});
define("app/ui/search_query_source", ["module", "require", "exports", "core/component", "app/utils/storage/custom"], function(module, require, exports) {
    function searchQuerySource() {
        this.defaultAttrs({
            querySourceLinkSelector: "a[data-query-source]",
            querySourceDataAttr: "data-query-source",
            storageExpiration: 6e4
        }), this.saveQuerySource = function(a) {
            this.storage.setItem("source", {
                source: {
                    value: a,
                    expire: Date.now() + this.attr.storageExpiration
                }
            }, this.attr.storageExpiration)
        }, this.catchLinkClick = function(a, b) {
            var c = $(b.el).attr(this.attr.querySourceDataAttr);
            c && this.saveQuerySource(c)
        }, this.saveTypedQuery = function(a, b) {
            if (b.source !== "search")
                return;
            this.saveQuerySource("typed_query")
        }, this.after("initialize", function() {
            var a = customStorage({
                withExpiry: !0
            });
            this.storage = new a("searchQuerySource"), this.on("click", {
                querySourceLinkSelector: this.catchLinkClick
            }), this.on("uiSearchQuery", this.saveTypedQuery)
        })
    }
    var defineComponent = require("core/component"), customStorage = require("app/utils/storage/custom");
    module.exports = defineComponent(searchQuerySource)
});
define("app/data/share_via_email_dialog_data", ["module", "require", "exports", "core/component", "app/data/with_data"], function(module, require, exports) {
    function shareViaEmailDialogData() {
        this.getDialogData = function(a, b) {
            var c = function(a) {
                this.trigger("dataShareViaEmailDialogSuccess", {
                    tweets: a.items_html,
                    name: b.name
                })
            }, d = function() {
                this.trigger("dataShareViaEmailDialogError")
            }, e = b.screenName, f = b.tweetId;
            this.get({
                url: "/i/" + e + "/conversation/" + f,
                success: c.bind(this),
                error: d.bind(this)
            })
        }, this.after("initialize", function() {
            this.on("uiNeedsShareViaDialogData", this.getDialogData)
        })
    }
    var defineComponent = require("core/component"), withData = require("app/data/with_data"), ShareViaEmailDialogData = defineComponent(shareViaEmailDialogData, withData);
    module.exports = ShareViaEmailDialogData
});
define("app/ui/dialogs/share_via_email_box", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function shareViaEmailBox() {
        var a = 8, b = 9, c = 13, d = 44;
        this.defaultAttrs({
            inputSelector: "input.js-share-tweet-emails",
            commentSelector: "textarea.js-share-comment",
            emailsListSelector: "div.emails-list",
            emailItemSelector: "span.email-item",
            insertedEmailItemSelector: "span.email-item.inserted-item",
            itemTextSelector: "span.email-item-text",
            itemRemoveButtonSelector: "a.email-item-remove",
            shareButtonSelector: "button.primary-btn"
        }), this.alreadyHasEmailItem = function(a) {
            var b = this.select("emailsListSelector").find("span.email-item"), c = b.filter(function() {
                return $.trim($(this).text()) === a
            });
            return c.length > 0
        }, this.appendEmailItem = function(a) {
            a = $.trim(a);
            if (!a.length)
                return;
            this.$input.val("");
            if (this.alreadyHasEmailItem(a))
                return;
            var b = this.$emailItemTemplate.clone(!1).css("display", "inline-block").addClass("inserted-item");
            b.find(this.attr.itemTextSelector).text(a), this.$emailsList.append(b), this.updatePlaceholderStatus(), this.trigger("uiUpdateShareViaEmailInputOverflow")
        }, this.removeLastEmailItem = function() {
            var a = this.$emailsList.find(this.attr.insertedEmailItemSelector);
            if (a.length > 0) {
                var b = a.last();
                b.hasClass("to-be-removed") ? (b.remove(), this.updatePlaceholderStatus(), this.trigger("uiUpdateShareViaEmailInputOverflow")) : b.addClass("to-be-removed")
            }
        }, this.removeEmailItem = function(a) {
            $(a.target).parent().parent().remove(), this.trigger("uiUpdateShareViaEmailInputOverflow"), this.$input.focus(), this.updatePlaceholderStatus()
        }, this.resetRemoveSelection = function() {
            var a = this.$emailsList.find(this.attr.insertedEmailItemSelector);
            a.last().removeClass("to-be-removed")
        }, this.inputFocusOut = function() {
            this.appendEmailItem(this.$input.val())
        }, this.changeText = function(a, b) {
            this.appendEmailItem(b.text)
        }, this.updatePlaceholderStatus = function() {
            this.$emailsList.find(this.attr.insertedEmailItemSelector).length ? this.removePlaceholder() : this.addPlaceholder()
        }, this.addPlaceholder = function() {
            this.$input.attr("placeholder", this.$input.attr("data-placeholder"))
        }, this.removePlaceholder = function() {
            this.$input.removeAttr("placeholder")
        }, this.clearEmailsList = function() {
            this.$emailsList.find(".inserted-item").remove(), this.updatePlaceholderStatus()
        }, this.inputKeypress = function(b) {
            (b.keyCode || b.which) == c ? (this.appendEmailItem(this.$input.val()), b.preventDefault()) : (b.keyCode || b.which) == d ? (this.appendEmailItem(this.$input.val()), b.preventDefault()) : (b.keyCode || b.which) != a && this.resetRemoveSelection()
        }, this.inputKeydown = function(c) {
            (c.keyCode || c.which) == a ? this.$input.val().length == 0 && this.removeLastEmailItem() : (c.keyCode || c.which) == b && this.$input.val().length > 0 && (this.appendEmailItem(this.$input.val()), c.preventDefault())
        }, this.after("initialize", function() {
            this.$emailsList = this.select("emailsListSelector"), this.$emailItemTemplate = this.select("emailItemSelector").first(), this.$shareButton = this.select("shareButtonSelector"), this.$comment = this.select("commentSelector"), this.$input = this.select("inputSelector"), this.updatePlaceholderStatus(), this.on("click", {
                shareButtonSelector: this.inputFocusOut,
                itemRemoveButtonSelector: this.removeEmailItem
            }), this.on("keypress", {
                inputSelector: this.inputKeypress
            }), this.on("keydown", {
                inputSelector: this.inputKeydown
            }), this.on("focusout", {
                inputSelector: this.resetRemoveSelection
            }), this.on("focusin", {
                commentSelector: this.inputFocusOut
            }), this.on("uiChangeTextAndPosition", this.changeText), this.on(document, "uiShareViaEmailBoxClearEmails", this.clearEmailsList)
        })
    }
    var defineComponent = require("core/component"), ShareViaEmailBox = defineComponent(shareViaEmailBox);
    module.exports = ShareViaEmailBox
});
define("app/ui/dialogs/share_via_email_dialog", ["module", "require", "exports", "core/i18n", "core/component", "app/data/share_via_email_dialog_data", "app/ui/dialogs/share_via_email_box", "app/ui/typeahead/typeahead_dropdown", "app/ui/typeahead/typeahead_input", "app/ui/updating_text_counter", "app/data/user_info", "core/utils", "app/data/with_data", "app/ui/with_dialog", "app/ui/dialogs/with_modal_tweet"], function(module, require, exports) {
    function shareViaEmailDialog() {
        this.defaultAttrs({
            formSelector: ".share-via-email-form",
            contentSelector: ".share-via-email-form .js-share-tweet-container",
            buttonSelector: ".share-via-email-form .primary-btn",
            inputSelector: ".share-via-email-form .js-share-tweet-emails",
            emailBoxSelector: ".share-via-email-form .emails-box",
            emailsListSelector: ".share-via-email-form .emails-list",
            inputWrapperSelector: ".share-via-email-form div.text-input-wrapper",
            commentSelector: ".share-via-email-form .js-share-comment",
            replyToUserSelector: ".share-via-email-form .js-reply-to-user",
            tweetSelector: ".share-via-email-form .tweet",
            socialProofSelector: ".share-via-email-form .comment-box .social-proof"
        }), this.openDialog = function(a, b) {
            userInfo.getDecider("swift_contacts_typeahead") && (TypeaheadInput.attachTo(this.attr.emailBoxSelector, {
                inputSelector: "#share-tweet-emails",
                triggerChangeTextAndPosition: !0
            }), TypeaheadDropdown.attachTo(this.attr.emailBoxSelector, {
                accountsShortcutShow: !0,
                autocompleteAccounts: !1,
                datasourceRenders: [["contacts", ["contacts"]]],
                deciders: this.attr.typeaheadData,
                inputSelector: "#share-tweet-emails",
                typeaheadSrc: "SHARE_VIA_EMAIL"
            }), ShareViaEmailBox.attachTo(this.attr.formSelector), this.trigger("uiShareViaEmailBoxClearEmails"), UpdatingTextCounter.attachTo(this.attr.formSelector, {
                inputSelector: ".js-share-comment",
                counterSelector: ".tweet-counter",
                maxLength: 140
            }));
            var c = $(a.target).clone().removeClass("retweeted favorited"), d = c.attr("data-tweet-id");
            this.sourceEventData = b, this.displayTweet(d, {
                modal: "share"
            }), this.$input.val(""), this.select("commentSelector").val(""), this.select("socialProofSelector").html(""), this.open(), this.trigger("uiNeedsShareViaDialogData", {
                tweetId: d,
                screenName: $(a.target).attr("data-screen-name"),
                name: $(a.target).attr("data-name")
            }), this.trigger("uiShareViaEmailDialogOpened", utils.merge(b, {
                scribeContext: {
                    component: "share_via_email_dialog"
                }
            }))
        }, this.fillDialog = function(a, b) {
            var c = $(b.tweets).find(".tweet"), d = b.name;
            this.select("socialProofSelector").html("");
            var e = [];
            $.each(c, function(a, b) {
                e.push($(b).attr("data-name"))
            }), e = jQuery.unique(e), e = jQuery.grep(e, function(a) {
                return a != d
            });
            var f = $(c[0]).attr("data-name"), g = $(c[0]).attr("data-protected"), h = $(c[c.length - 1]).attr("data-name"), i = $(c[c.length - 1]).attr("data-protected"), j = "", k = e.length - 2, l = {
                inReplyToTweetAuthor: h,
                rootTweetAuthor: f,
                othersLeft: k
            };
            f != d&&!g ? f != h&&!i ? e.length > 3 ? j = _('In reply to {{rootTweetAuthor}}, {{inReplyToTweetAuthor}} and {{othersLeft}} others', l) : e.length > 0 && (j = _('In reply to {{rootTweetAuthor}} and {{inReplyToTweetAuthor}}', l)) : i || (e.length > 3 ? j = _('In reply to  {{rootTweetAuthor}} and {{othersLeft}} others)', l) : e.length > 0 && (j = _('In reply to {{rootTweetAuthor}}', l))) : h != d&&!i && (e.length > 3 ? j = _('In reply to {{inReplyToTweetAuthor}} and {{othersLeft}} others', l) : e.length > 0 && (j = _('In reply to {{inReplyToTweetAuthor}}', l))), this.select("socialProofSelector").html(j)
        }, this.getCommaSeparatedEmails = function() {
            if (userInfo.getDecider("swift_contacts_typeahead")) {
                var a = this.select("emailsListSelector").find("span.email-item"), b = "";
                return a.each(function() {
                    var a = $.trim($(this).find(".email-item-text").text());
                    a.length > 0 && (b += a + ",")
                }), b
            }
            return this.select("inputSelector").val()
        }, this.updateOverflowStatus = function() {
            this.getCommaSeparatedEmails().length > this.inputMaxlength ? (this.$inputWrapper.addClass("with-overflow-error"), this.$shareButton.attr("disabled", "disabled"), this.$input.attr("maxlength", 0)) : (this.$inputWrapper.removeClass("with-overflow-error"), this.$shareButton.removeAttr("disabled"), this.$input.attr("maxlength", this.inputMaxlength))
        }, this.submitForm = function(a) {
            var b = {
                id: this.select("tweetSelector").attr("data-tweet-id"),
                emails: this.getCommaSeparatedEmails(),
                comment: this.select("commentSelector").val(),
                reply_to_user: this.select("replyToUserSelector").is(":checked")
            };
            this.post({
                url: "/i/tweet/share_via_email",
                data: b,
                eventData: null,
                success: "dataShareViaEmailSuccess",
                error: "dataShareViaEmailError"
            }), this.trigger("uiCloseDialog"), a.preventDefault()
        }, this.shareSuccess = function(a, b) {
            this.trigger("uiShowMessage", {
                message: b.message
            }), this.trigger("uiDidShareViaEmailSuccess", this.sourceEventData)
        }, this.restoreFocusToTweet = function(a) {
            $(a.target).is(this.$dialog) && this.activeEl && this.trigger($(this.activeEl).closest(".tweet"), "uiShouldAddFocusStyle")
        }, this.after("initialize", function() {
            this.$inputWrapper = this.select("inputWrapperSelector"), this.$shareButton = this.select("buttonSelector"), this.$input = this.select("inputSelector"), this.inputMaxlength = parseInt(this.$input.attr("maxlength")), this.on(document, "uiNeedsShareViaEmailDialog", this.openDialog), this.on(document, "dataShareViaEmailDialogSuccess", this.fillDialog), this.on(document, "dataShareViaEmailSuccess", this.shareSuccess), this.on(document, "uiDialogRestorePreviousFocus", this.restoreFocusToTweet), this.on(document, "uiUpdateShareViaEmailInputOverflow", this.updateOverflowStatus), this.on("click", {
                buttonSelector: this.submitForm
            }), DialogData.attachTo(this.$node, {
                noTeardown: !0
            })
        })
    }
    var _ = require("core/i18n"), defineComponent = require("core/component"), DialogData = require("app/data/share_via_email_dialog_data"), ShareViaEmailBox = require("app/ui/dialogs/share_via_email_box"), TypeaheadDropdown = require("app/ui/typeahead/typeahead_dropdown"), TypeaheadInput = require("app/ui/typeahead/typeahead_input"), UpdatingTextCounter = require("app/ui/updating_text_counter"), userInfo = require("app/data/user_info"), utils = require("core/utils"), withData = require("app/data/with_data"), withDialog = require("app/ui/with_dialog"), withModalTweet = require("app/ui/dialogs/with_modal_tweet"), ShareViaEmailDialog = defineComponent(shareViaEmailDialog, withDialog, withData, withModalTweet);
    module.exports = ShareViaEmailDialog
});
define("app/data/spam_challenge", ["module", "require", "exports", "core/component", "core/i18n", "app/data/with_data"], function(module, require, exports) {
    function spamChallenge() {
        this.defaultAttrs({
            supportedChallengeTypes: ["Captcha"],
            captchaName: "Captcha",
            defaultCompletionMessage: _('Challenge completed.'),
            recaptchaLoadTime: 100,
            maxCaptchaLoadRetries: 3
        }), this.hitChallengeEndpoint = function(a) {
            this.get({
                url: "/account/challenge",
                data: {
                    challenge: a.challenge,
                    response: a.response,
                    challenge_name: a.challengeName
                },
                success: this.handleSuccessResponse.bind(this),
                error: this.handleErrorResponse.bind(this)
            })
        }, this.handleErrorResponse = function(a) {
            this.closeCurrentChallenge(), this.resetChallengeParams()
        }, this.closeCurrentChallenge = function() {
            this.currentChallenge && this.trigger("uiCloseSpam" + this.currentChallenge + "Challenge")
        }, this.allChallengesCompleted = function(a) {
            return !a.challengeName
        }, this.isValidChallengeType = function(a) {
            return this.attr.supportedChallengeTypes.indexOf(a.challengeName)!==-1
        }, this.needToRefreshCaptchaChallenge = function(a) {
            return this.currentChallenge === this.attr.captchaName && a.challengeName === this.attr.captchaName
        }, this.handleSuccessResponse = function(a) {
            this.allChallengesCompleted(a) ? (this.closeCurrentChallenge(), this.sendCompletionMessage(), this.resetChallengeParams()) : this.needToRefreshCaptchaChallenge(a) ? this.trigger("uiRefreshSpamCaptchaChallenge") : this.isValidChallengeType(a) ? (this.closeCurrentChallenge(), this.setAndOpenNewChallenge(a)) : (this.closeCurrentChallenge(), this.resetChallengeParams())
        }, this.setAndOpenNewChallenge = function(a) {
            this.setChallengeParams(a.challengeName, a.completionMessage), this.trigger("uiOpenSpam" + this.currentChallenge + "Challenge", a)
        }, this.sendCompletionMessage = function() {
            this.trigger("uiShowMessage", {
                message: this.getCompletionMessage()
            })
        }, this.getCompletionMessage = function() {
            return this.completionMessage || this.attr.defaultCompletionMessage
        }, this.createInitialChallenge = function(a, b) {
            this.needToCreateRecaptchaWidget(b) ? this.fetchRecaptchaScript(a, b) : this.setAndOpenNewChallenge(b)
        }, this.needToCreateRecaptchaWidget = function(a) {
            return a.challengeName === this.attr.captchaName&&!this.isRecaptchaObjectDefined()
        }, this.isRecaptchaObjectDefined = function() {
            return typeof Recaptcha != "undefined"
        }, this.fetchScriptDone = function(a, b, c) {
            c == "success" ? this.ensureRecaptchaCreatedAndLoadChallenge(a) : this.trigger(document, "uiShowError", {
                message: "Internal server error."
            })
        }, this.fetchScriptFailure = function(a, b, c) {
            this.trigger(document, "uiShowError", {
                message: "Internal server error."
            })
        }, this.fetchRecaptchaScript = function(a, b) {
            $.getScript(this.attr.recaptchaApiUrl).done(this.fetchScriptDone.bind(this, b)).fail(this.fetchScriptFailure.bind(this))
        }, this.ensureRecaptchaCreatedAndLoadChallenge = function(a) {
            if (this.isRecaptchaObjectDefined()) {
                this.setAndOpenNewChallenge(a);
                return 
            }
            this.loadAttempts = this.attr.maxCaptchaLoadRetries, this.recaptchaLoadInterval = setInterval(this.checkIfRecaptchaCreated.bind(this, a), this.attr.recaptchaLoadTime)
        }, this.clearRecaptchLoadInterval = function() {
            clearInterval(this.recaptchaLoadInterval)
        }, this.clearLoadAttempts = function() {
            this.loadAttempts = undefined
        }, this.isLoadAttemptsDefined = function() {
            return typeof this.loadAttempts != "undefined"
        }, this.checkIfRecaptchaCreated = function(a) {
            this.isLoadAttemptsDefined() && this.loadAttempts--, this.isRecaptchaObjectDefined() && (this.clearRecaptchLoadInterval(), this.setAndOpenNewChallenge(a), this.clearLoadAttempts());
            if (!this.isLoadAttemptsDefined() || this.loadAttempts <= 0)
                this.clearRecaptchLoadInterval(), this.clearLoadAttempts(), this.trigger(document, "uiShowError", {
                    message: "Internal server error."
                })
        }, this.submitChallenge = function(a, b) {
            this.hitChallengeEndpoint(b)
        }, this.resetChallengeParams = function() {
            this.setChallengeParams()
        }, this.setChallengeParams = function(a, b) {
            this.currentChallenge = a, this.completionMessage = b
        }, this.after("initialize", function() {
            this.on("uiSubmitSpamChallenge", this.submitChallenge), this.on("dataInitialChallengeNeeded", this.createInitialChallenge)
        })
    }
    var defineComponent = require("core/component"), _ = require("core/i18n"), withData = require("app/data/with_data");
    module.exports = defineComponent(spamChallenge, withData)
});
define("app/ui/spoonbill", ["module", "require", "exports", "core/component", "template", "app/ui/with_user_actions", "app/ui/with_item_actions", "app/ui/with_interaction_data", "app/data/with_interaction_data_scribe"], function(module, require, exports) {
    function Spoonbill() {
        this.defaultAttrs({
            eventData: {
                scribeContext: {
                    component: "spoonbill"
                }
            },
            toastClass: "WebToast",
            toastSelector: ".WebToast",
            closeButtonSelector: ".WebToast-closeButton",
            headerSelector: ".WebToast-header",
            toastfollowButtonSelector: ".WebToast-followButton",
            fullnameSelector: ".WebToast-fullname",
            usernameSelector: ".WebToast-username",
            avatarSelector: ".WebToast-avatar",
            excerptSelector: ".WebToast-tweetExcerpt",
            actionBarSelector: ".WebToast-actionBar",
            tweetTextSelector: ".tweet-text",
            inlineReplySelector: ".js-inlineReply",
            tweetFormSelector: ".WebToast-box .tweet-form",
            redirectToNotificationsSelector: '.WebToast[data-redirect-to="notifications"]'
        }), this.elementIsClickableTarget = function(a) {
            var b = ["a", "b", "button", "strong"];
            return a && a.tagName && b.indexOf(a.tagName.toLowerCase())>-1
        }, this.hasAncestorWithClickableTarget = function(a, b) {
            var c=!1;
            while (a) {
                this.elementIsClickableTarget(a) && (c=!0);
                if (c || a == b)
                    break;
                a = a.parentElement
            }
            return c
        }, this.redirectToNotifications = function(a, b) {
            if (this.hasAncestorWithClickableTarget(a.target, a.currentTarget))
                return;
            this.trigger("uiNavigate", {
                href: "/i/notifications"
            })
        }, this.renderSpoonbillNotification = function(a, b) {
            var c = "spoonbill_" + (new Date).getTime(), d = $(template["spoonbill/" + b.type].render(b, template));
            this.$node.empty(), this.$node.append(d), d.fadeIn(), this.setupUIElementsForSpoonbill(d, b), d.attr("id", c), b.elementId = c, this.trigger("uiSpoonbillDidShow", b), this.scribeInteraction({
                action: "impression"
            }, this.interactionData(d))
        }, this.spoonbillDismiss = function(a, b) {
            this.markDMRead(), this.scribeInteraction({
                action: "dismiss"
            }, this.interactionData(a)), this.trigger("uiNotificationDismissed", {}), a.stopPropagation()
        }, this.replyClick = function() {
            this.select("tweetFormSelector").trigger("uiTweetBoxExpand")
        }, this.setupMentionUIForSpoonbill = function(a, b) {
            var c = b.mentions;
            c.unshift(b.user_screen_name), b.screen_names_for_tweet_box = c.map(function(a) {
                return "@" + a
            }).join(" "), b.tweet_box_id = "spoonbill_tweet_box", b.include_image_picker=!0, b.upload_domain = this.attr.uploadDomain, $(template.swift_tweet_box.render(b, template)).appendTo(this.select("inlineReplySelector")), this.select("tweetFormSelector").trigger("uiInitTweetbox", {
                noTeardown: !0,
                condensable: !0,
                condenseContainer: this.attr.toastSelector,
                preventTweetingDefaultText: !1,
                inReplyToStatusId: b.tweet_id,
                impressionId: b.impression_id,
                accountsShortcutShow: !1,
                renderLimit: 2,
                eventData: {
                    scribeContext: {
                        component: "spoonbill"
                    }
                }
            }), this.attr.itemType = "tweet"
        }, this.markDMRead = function() {
            this.dmData && this.dmData.tweet_id && this.trigger("dataMarkDMsAsRead", {
                last_message_id: this.dmData.tweet_id,
                recipient_id: this.dmData.user_id,
                recipient_name: this.dmData.user_screen_name,
                from_notification: !0
            }), this.dmData = null
        }, this.setupDMUIForSpoonbill = function(a, b) {
            this.dmData = b, b.media_upload_domain = this.attr.uploadDomain, $(template.swift_dm_box.render(b, template)).appendTo(this.select("inlineReplySelector"));
            var c = this.select("tweetFormSelector");
            c.trigger("uiInitTweetbox", {
                noTeardown: !0,
                suppressFlashMessage: !0,
                hasTypeahead: !1,
                eventData: {
                    scribeContext: {
                        component: "spoonbill"
                    }
                },
                dmOnly: !0,
                fileDataString: "media",
                fileInputString: "media"
            }), c.trigger("uiDMRecipientChanged", {
                hasValidRecipient: !0
            }), this.attr.itemType = "tweet"
        }, this.setupFollowUIForSpoonbill = function(a, b) {
            this.attr.itemType = "user"
        }, this.setupFavoriteUIForSpoonbill = function(a, b) {
            this.attr.itemType = "activity"
        }, this.setupRetweetUIForSpoonbill = function(a, b) {
            this.attr.itemType = "activity"
        }, this.setupUIElementsForSpoonbill = function(a, b) {
            b.type == "mention" ? this.setupMentionUIForSpoonbill(a, b) : b.type == "dm" ? this.setupDMUIForSpoonbill(a, b) : b.type == "follow" ? this.setupFollowUIForSpoonbill(a, b) : b.type == "favorite" ? this.setupFavoriteUIForSpoonbill(a, b) : b.type == "retweet" && this.setupRetweetUIForSpoonbill(a, b)
        }, this.dmScreenName = function() {
            return this.select("toastSelector").data("screen-name")
        }, this.sendDM = function(a, b) {
            this.markDMRead(), this.attr.sentTweetboxId = b.tweetboxId, a.stopPropagation(), a.preventDefault(), this.trigger("uiDMDialogSendMessage", {
                tweetboxId: b.tweetboxId,
                screen_name: this.dmScreenName(),
                text: b.tweetData.status,
                source: "spoonbill"
            }), this.trigger("uiNotificationComposeActive", b)
        }, this.sendDMWithMedia = function(a, b) {
            this.markDMRead(), b.recipient = this.dmScreenName(), this.attr.sentTweetboxId = b.tweetboxId, this.trigger("uiNotificationComposeActive", b)
        }, this.sendTweet = function(a, b) {
            this.attr.sentTweetboxId = b.tweetboxId, this.trigger("uiNotificationComposeActive", b)
        }, this.resetOnSend = function() {
            this.attr.sentTweetboxId = undefined, this.trigger("uiNotificationComposeComplete")
        }, this.sendDMSuccess = function(a, b) {
            b.sourceEventData.tweetboxId == this.attr.sentTweetboxId && this.resetOnSend()
        }, this.sendTweetSuccess = function(a, b) {
            b.tweetboxId == this.attr.sentTweetboxId && this.resetOnSend()
        }, this.resumeRotation = function(a, b) {
            this.trigger("uiResumeSpoonbillRotation")
        }, this.after("initialize", function() {
            this.on(document, "uiRequestSpoonbillRender", this.renderSpoonbillNotification), this.on("uiReplyToTweet", this.replyClick), this.on("uiSendDM", this.sendDM), this.on("uiSendDMWithMedia", this.sendDMWithMedia), this.on("uiSendTweet", this.sendTweet), this.on(document, "dataDMSuccess", this.sendDMSuccess), this.on(document, "dataTweetSuccess", this.sendTweetSuccess), this.on(document, "dataTweetError dataDMError", this.resumeRotation), this.on("click", {
                closeButtonSelector: this.spoonbillDismiss,
                redirectToNotificationsSelector: this.redirectToNotifications
            })
        })
    }
    var defineComponent = require("core/component"), template = require("template"), withUserActions = require("app/ui/with_user_actions"), withItemActions = require("app/ui/with_item_actions"), withInteractionData = require("app/ui/with_interaction_data"), withInteractionDataScribe = require("app/data/with_interaction_data_scribe");
    module.exports = defineComponent(Spoonbill, withUserActions, withItemActions, withInteractionData, withInteractionDataScribe)
});
define("app/ui/spoonbill_producer", ["module", "require", "exports", "core/component", "app/utils/storage/core", "core/clock"], function(module, require, exports) {
    function SpoonbillProducer() {
        this.defaultAttrs({
            spoonbillDuration: {
                mention: 8e3,
                dm: 8e3,
                follow: 8e3,
                normal: 5e3
            },
            betweenInterval: 2e3,
            eventTypeMapper: {
                direct_message: "dm",
                favorited: "favorite",
                favorited_mention: "favorite",
                favorited_retweet: "favorite",
                followed: "follow",
                followed_request: "follow",
                retweeted: "retweet",
                retweeted_mention: "retweet",
                retweeted_retweet: "retweet"
            },
            acceptEventTypes: ["dm", "favorite", "follow", "retweet", "mention"],
            queue: [],
            toastClass: "WebToast",
            toastSelector: ".WebToast",
            tweetFormSelector: ".WebToast-box .tweet-form",
            closeButtonSelector: ".WebToast-close",
            displayState: undefined
        }), this.incomingSpoonbills = function(a, b) {
            if (!b || typeof b != "object" ||!Array.isArray(b.toasts))
                return;
            b.force && (this.latestNotificationTimestamp = 0), this.normalizeSpoonbills(b.toasts).forEach(function(a) {
                this.attr.eventTypeMapper.hasOwnProperty(a.type) && (a.type = this.attr.eventTypeMapper[a.type]);
                if (this.attr.acceptEventTypes.indexOf(a.type) < 0)
                    return;
                this.addSpoonbill(a)
            }, this), this.latestNotificationTimestamp=++b.latest, storage.setItem("latestNotificationTimestamp", this.latestNotificationTimestamp)
        }, this.addSpoonbill = function(a) {
            if (!a.timestamp || a.timestamp < this.latestNotificationTimestamp)
                return;
            this.attr.queue.push(a), this.displayState == DisplayState.None && this.showSpoonbill()
        }, this.showSpoonbill = function() {
            var a = this.getNext();
            if (!a) {
                this.displayState = DisplayState.None;
                return 
            }
            this.displayState = DisplayState.Visible, this.removeMouseBindings(), this.trigger("uiRequestSpoonbillRender", a)
        }, this.getPath = function() {
            return location.pathname
        }, this.getNext = function() {
            var a = this.attr.queue.shift();
            if (this.getPath() == "/i/notifications")
                while (a && (a.type == "retweet" || a.type == "favorite"))
                    a = this.attr.queue.shift();
            return a && a.type == "follow" && (a.with_block_username=!0), a
        }, this.normalizeSpoonbills = function(a) {
            return a.filter(function(a) {
                return a.timestamp
            }).sort(function(a, b) {
                return a.timestamp - b.timestamp
            })
        }, this.removeMouseBindings = function() {
            this.select("toastSelector").unbind("mouseenter", this.notificationMouseEnterHandler), this.select("toastSelector").unbind("mouseleave", this.notificationMouseLeaveHandler)
        }, this.clearTimers = function(a) {
            a.forEach(function(a) {
                this[a] && (clearTimeout(this[a]), delete this[a])
            }.bind(this))
        }, this.purgeNotifications = function(a) {
            this.removeMouseBindings(), this.clearTimers(["rotateTimer", "updateTimer"]), this.attr.queue = [], this.fadeOut()
        }, this.fadeOut = function() {
            this.select("toastSelector").animate({
                opacity: 0
            }, function() {
                this.select("tweetFormSelector").trigger("uiAllowTeardown", {}), this.select("toastSelector").remove(), this.displayState = DisplayState.Between, this.clearTimers(["rotateTimer"]), this.rotateTimer = setTimeout(this.showSpoonbill.bind(this), this.attr.betweenInterval)
            }.bind(this))
        }, this.updateNotificationDisplay = function(a) {
            this.checkForUserData(), this.displayState > DisplayState.Visible ? this.monitorVisibleSpoonbill(a) : this.displayState == DisplayState.Visible && this.fadeOut()
        }, this.monitorVisibleSpoonbill = function(a) {
            this.clearTimers(["updateTimer"]), this.updateTimer = setTimeout(this.updateNotificationDisplay.bind(this, a), this.getRotateInterval(a.type))
        }, this.getRotateInterval = function(a) {
            return this.attr.spoonbillDuration.hasOwnProperty(a) ? this.attr.spoonbillDuration[a] : this.attr.spoonbillDuration.normal
        }, this.checkForUserData = function() {
            if (this.displayState > DisplayState.Composing)
                return;
            var a = this.$node.find(".tweet-button"), b = a.is(":visible") && a.find("button.primary-btn").attr("disabled") != "disabled" && a.find("button.primary-btn.disabled").length == 0, c = $.contains(this.$node[0], document.activeElement);
            b || c ? this.displayState = DisplayState.Composing : this.displayState > DisplayState.Visible && (this.displayState = DisplayState.Visible)
        }, this.spoonbillDidShow = function(a, b) {
            var c = $("#" + b.elementId);
            this.notificationMouseEnterHandler = this.handleMouseEnterForNotifications.bind(this), this.notificationMouseLeaveHandler = this.handleMouseLeaveForNotifications.bind(this), c.bind("mouseenter", this.notificationMouseEnterHandler), c.bind("mouseleave", this.notificationMouseLeaveHandler), this.monitorVisibleSpoonbill(b)
        }, this.processNewNotifications = function(a, b) {
            b && b.n && b.n.status == "ok" && b.n.response && typeof b.n.response == "object" && this.trigger("uiNewSpoonbillsArrived", b.n.response)
        }, this.composeActive = function(a, b) {
            this.displayState = DisplayState.Composing
        }, this.composeComplete = function(a, b) {
            this.fadeOut()
        }, this.handleMouseEnterForNotifications = function(a, b) {
            this.displayState = DisplayState.Focused
        }, this.handleMouseLeaveForNotifications = function(a, b) {
            this.displayState = DisplayState.Visible
        }, this.handleImagePickerClick = function(a, b) {
            this.displayState = DisplayState.Attaching
        }, this.after("initialize", function() {
            this.displayState = DisplayState.None, this.on(document, "uiNotificationComposeActive", this.composeActive), this.on(document, "uiNotificationComposeComplete", this.composeComplete), this.on(document, "uiNewSpoonbillsArrived", this.incomingSpoonbills), this.on(document, "uiSpoonbillDidShow", this.spoonbillDidShow), this.on(document, "uiNotificationRotate", this.showSpoonbill), this.on(document, "dataNotificationsReceived", this.processNewNotifications), this.on("uiImagePickerClick", this.handleImagePickerClick), this.on("uiNotificationDismissed", this.purgeNotifications), this.on("mouseenter", this.handleMouseEnterForNotifications), this.on("mouseleave", this.handleMouseLeaveForNotifications)
        })
    }
    var defineComponent = require("core/component"), Storage = require("app/utils/storage/core"), clock = require("core/clock");
    module.exports = defineComponent(SpoonbillProducer);
    var storage = new Storage("DM"), DisplayState = {
        None: 1,
        Between: 2,
        Visible: 3,
        Composing: 4,
        Focused: 5,
        Attaching: 6
    }
});
define("app/data/geo", ["module", "require", "exports", "core/component", "core/utils", "app/utils/cookie", "app/data/with_data"], function(module, require, exports) {
    function geo() {
        this.defaultAttrs({
            profileSearchThrottle: 35
        }), this.isInState = function(a) {
            return a.split(" ").indexOf(this.state) >= 0
        }, this.isEnabled = function(a) {
            return !this.isInState("disabled enabling enableIsUnavailable")
        }, this.setInitialState = function() {
            this.attr.geoEnabled ? this.state = Geo.webclientCookie() === "1" ? "enabledTurnedOn" : "enabledTurnedOff" : this.state = "disabled"
        }, this.requestState = function(a, b) {
            this.shouldLocate() ? this.locate() : this.sendState(this.state)
        }, this.shouldLocate = function() {
            return this.isInState("enabledTurnedOn locateIsUnavailable") || this.isInState("located locationUnknown") && (!this.lastLocationTime || this.now() - this.lastLocationTime > MIN_TIME_BETWEEN_LOCATES_IN_MS)
        }, this.locate = function(a) {
            this.sendState(a ? "changing" : "locating");
            var b = function(a) {
                this.sendState(this.locateOrEnableState(a) || "locateIsUnavailable")
            }.bind(this), c = function() {
                this.sendState("locateIsUnavailable")
            }.bind(this), d = {
                override_place_id: a
            };
            this.post({
                url: "/account/geo_locate",
                data: d,
                echoParams: {
                    spoof_ip: !0
                },
                eventData: d,
                success: b,
                error: c
            })
        }, this.locateOrEnableState = function(a) {
            switch (a.status) {
            case"ok":
                return this.place_id = a.place_id, this.place_name = a.place_name, this.places_html = a.html, this.lastLocationTime = this.now(), "located";
            case"unknown":
                return this.lastLocationTime = this.now(), "locationUnknown"
            }
        }, this.now = function() {
            return (new Date).getTime()
        }, this.sendState = function(a) {
            a && (this.state = a);
            var b = {
                state: this.state
            };
            this.state === "located" && (b.place_id = this.place_id, b.place_name = this.place_name, b.places_html = this.places_html), this.trigger("dataGeoState", b)
        }, this.turnOn = function() {
            this.isEnabled() && (Geo.webclientCookie("1"), this.locate())
        }, this.turnOff = function() {
            this.isEnabled() && (Geo.webclientCookie(null), this.sendState("enabledTurnedOff"))
        }, this.enable = function(a, b) {
            if (!this.isInState("disabled enableIsUnavailable"))
                return;
            this.sendState("enabling");
            var c = function(a) {
                Geo.webclientCookie("1"), this.sendState(this.locateOrEnableState(a) || "enableIsUnavailable")
            }.bind(this), d = function() {
                this.sendState("enableIsUnavailable")
            }.bind(this);
            this.post({
                url: "/account/geo_locate",
                data: {
                    enable: "1"
                },
                echoParams: {
                    spoof_ip: !0
                },
                eventData: b,
                success: c,
                error: d
            })
        }, this.change = function(a, b) {
            this.isEnabled() && this.locate(b.placeId)
        }, this.search = function(a, b) {
            if (this.searching) {
                this.pendingSearchData = b;
                return 
            }
            this.pendingSearchData = null;
            var c = function() {
                this.searching=!1;
                if (this.pendingSearchData)
                    return this.search(a, this.pendingSearchData), !0
            }.bind(this), d = b.query.trim(), e = [d, b.placeId, b.isPrefix].join(","), f = function(a) {
                this.searchCache[e] = a, a = $.extend({}, a, {
                    sourceEventData: b
                }), c() || this.trigger("dataGeoSearchResults", a)
            }.bind(this), g = function() {
                c() || this.trigger("dataGeoSearchResultsUnavailable")
            }.bind(this);
            if (!d) {
                f({
                    html: ""
                });
                return 
            }
            var h = this.searchCache[e];
            if (h) {
                f(h);
                return 
            }
            this.searching=!0, this.get({
                url: "/account/geo_search",
                data: {
                    query: d,
                    place_id: b.placeId,
                    is_prefix: b.isPrefix ? "1": "0"
                },
                eventData: b,
                success: f,
                error: g
            })
        }, this.profileSearch = function(a, b) {
            var c = b.query.trim(), d = c, e = function(a) {
                this.profileSearchCache[d] = a, a = utils.merge(a, {
                    sourceEventData: b
                }), this.trigger("dataProfileGeoSearchResults", a)
            }.bind(this), f = this.profileSearchCache[d];
            if (f) {
                e(f);
                return 
            }
            this.get({
                url: "/account/geo_profile",
                data: {
                    query: c
                },
                eventData: b,
                success: e,
                error: "dataProfileGeoSearchResultsUnavailable"
            })
        }, this.requestDefaultProfileLocations = function(a, b) {
            this.get({
                url: "/account/geo_profile",
                eventData: b,
                success: "dataDefaultProfileLocations",
                error: "dataProfileGeoSearchResultsUnavailable"
            })
        }, this.after("initialize", function() {
            this.searchCache = {}, this.profileSearchCache = {}, this.setInitialState(), this.on("uiRequestGeoState", this.requestState), this.on("uiGeoPickerEnable", this.enable), this.on("uiGeoPickerTurnOn", this.turnOn), this.on("uiGeoPickerTurnOff", this.turnOff), this.on("uiGeoPickerChange", this.change), this.on("uiGeoPickerSearch", this.search), this.on("uiRequestDefaultProfileLocations", this.requestDefaultProfileLocations), this.on("uiGeoProfileSearch", utils.throttle(this.profileSearch.bind(this), this.attr.profileSearchThrottle))
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), cookie = require("app/utils/cookie"), withData = require("app/data/with_data"), Geo = defineComponent(geo, withData), MIN_TIME_BETWEEN_LOCATES_IN_MS = 9e5;
    module.exports = Geo, Geo.webclientCookie = function(a) {
        return a === undefined ? cookie("geo_webclient") : cookie("geo_webclient", a, {
            expires: 3650,
            path: "/"
        })
    }
});
define("app/data/with_media_upload", ["module", "require", "exports", "core/i18n"], function(module, require, exports) {
    function withMediaUpload() {
        function b(a, b) {
            a.originalEvent && (a = a.originalEvent);
            try {
                var c = JSON.parse(a.data);
                if (c.media_id_string && c.upload_id == b)
                    return c.media_id_string
            } catch (d) {}
            return !1
        }
        var a = "https://upload.twitter.com/i/media/upload.iframe";
        this.UPLOAD_TIMEOUT = 12e4, this.$hiddenContent = $("#hidden-content"), this.$formTemplate = $(".media-upload-form"), this.getOnMessage = function(a, c) {
            return function(d) {
                var e = b(d, c);
                e && a.resolveWith(this, [e, c])
            }.bind(this)
        }, this.uploadMedia = function(a) {
            var b = a.mediaData, c = a.uploadId, d = this.createMediaUploadForm(this.$hiddenContent, this.$formTemplate, c, b), e = $.Deferred(), f = setTimeout(function() {
                e.reject(_('Uploading photo failed! Please try again later.'))
            }, this.UPLOAD_TIMEOUT), g = this.getOnMessage(e, c), h = function() {
                d.form.remove(), d.iframe.remove(), this.off(window, "message", g), clearTimeout(f)
            }.bind(this);
            return this.on(window, "message", g), e.always(h), d.form.submit(), e
        }, this.createMediaUploadForm = function(b, c, d, e) {
            var f = $("<iframe>");
            f.attr("name", "media-upload-iframe-" + d), b.append(f);
            var g = c.clone(), h = location.protocol + "//" + location.hostname + (location.port ? ":" + location.port : "");
            return g.attr("target", "media-upload-iframe-" + d), g.attr("action", a + "?origin=" + encodeURIComponent(h)), g.find(".auth-token").val(this.getAuthToken()), g.find(".media").val(e), g.find(".upload-id").val(d), g.find(".origin").val(h), b.append(g), {
                form: g,
                iframe: f
            }
        }
    }
    var _ = require("core/i18n");
    module.exports = withMediaUpload
});
define("app/data/tweet", ["module", "require", "exports", "core/component", "app/data/with_auth_token", "core/i18n", "app/data/with_data", "app/data/with_media_upload", "app/data/user_info"], function(module, require, exports) {
    function tweet() {
        this.IFRAME_TIMEOUT = 12e4, this.defaultAttrs({
            includePageContext: !0
        }), this.sendTweet = function(a, b) {
            var c = b.tweetboxId, d = function(a) {
                a.tweetboxId = c, this.trigger("dataTweetSuccess", a), this.trigger("dataGotProfileStats", {
                    stats: a.profile_stats
                }), a.tweet_stats && this.trigger("dataGotTweetStats", {
                    stats: a.tweet_stats
                })
            }, e = function(a) {
                var b;
                try {
                    b = {
                        message: a.message,
                        spamChallenge: a.challengeName
                    }
                } catch (d) {
                    b = {
                        error: _('Sorry! We did something wrong.')
                    }
                }
                b.tweetboxId = c, this.trigger("dataTweetError", b)
            };
            this.post({
                url: "/i/tweet/create",
                isMutation: !1,
                data: b.tweetData,
                success: d.bind(this),
                error: e.bind(this),
                retryIfUnavailable: !0
            })
        }, this.getMediaData = function(a) {
            var b = {}, c = $("#" + a), d = c.find(".multi-photo-data");
            if (d.length)
                $.each(d, function(a, c) {
                    var d = $(c).attr("data-upload-id");
                    b[d] = $(c).val()
                });
            else {
                var e =+ (new Date);
                b[e] = c.find(".file-data").val()
            }
            return b
        }, this.sendTweetWithMedia = function(a, b) {
            var c = b.tweetboxId, d = b.tweetData, e = this.getMediaData(c), f = $.map(e, function(a, b) {
                var d = this.uploadMedia({
                    mediaData: a,
                    tweetboxId: c,
                    uploadId: b
                });
                return d.done(function(a, b) {
                    e[b] = a
                }.bind(this)), d
            }.bind(this));
            $.when.apply($, f).then(function() {
                d.media_ids = $.map(e, function(a) {
                    return a
                }).join(","), this.sendTweet(a, {
                    tweetboxId: c,
                    tweetData: d
                })
            }.bind(this), function(a) {
                this.trigger("dataTweetError", {
                    tweetboxId: c,
                    message: a
                })
            }.bind(this))
        }, this.sendDirectMessage = function(a, b) {
            this.trigger("dataDirectMessageSuccess", b)
        }, this.after("initialize", function() {
            this.on("uiSendTweet", this.sendTweet), this.on("uiSendTweetWithMedia", this.sendTweetWithMedia), this.on("uiSendDirectMessage", this.sendDirectMessage)
        })
    }
    var defineComponent = require("core/component"), withAuthToken = require("app/data/with_auth_token"), _ = require("core/i18n"), withData = require("app/data/with_data"), withMediaUpload = require("app/data/with_media_upload"), userInfo = require("app/data/user_info"), Tweet = defineComponent(tweet, withAuthToken, withData, withMediaUpload);
    module.exports = Tweet
});
define("app/ui/tweet_dialog", ["module", "require", "exports", "core/component", "core/utils", "app/ui/with_dialog", "app/ui/dialogs/with_modal_tweet", "app/data/user_info", "core/i18n"], function(module, require, exports) {
    function tweetDialog() {
        this.defaultAttrs({
            tweetBoxSelector: "form.tweet-form",
            globalTweetBoxSelector: "#global-tweet-dialog form.tweet-form"
        }), this.openReply = function(a, b) {
            this.displayTweet(b.id, {
                modal: "reply"
            });
            var c = b.screenNames[0];
            this.openTweetDialog(a, utils.merge(b, {
                title: _('Reply to {{screenName}}', {
                    screenName: "@" + c
                })
            }))
        }, this.openGlobalTweetDialog = function(a, b) {
            if (this.isModalOpen())
                return;
            this.openTweetDialog(a, utils.merge(b, {
                draftTweetId: "global"
            }))
        }, this.openTweetDialog = function(a, b) {
            var c = $(this.attr.globalTweetBoxSelector);
            this.setTitle(b && b.title || _('Compose new Tweet')), this.tweetBoxReady || (this.trigger(c, "uiInitTweetbox", {
                eventData: {
                    scribeContext: {
                        component: "tweet_box_dialog"
                    }
                },
                modal: !0,
                canTweetDefaultText: b && b.canTweetDefaultText
            }), this.tweetBoxReady=!0);
            if (b) {
                var d = null;
                b.screenNames ? d = b.screenNames : b.screenName && (d = [b.screenName]), d && (b.defaultText = "@" + d.join(" @") + " ", b.condensedText = _('Reply to {{screenNames}}', {
                    screenNames: b.defaultText
                })), b.text && (b.defaultText = " " + b.text), b.setCursorToBeginning && (b.cursorPosition = 0), b.imageUrl && this.trigger(c, "uiGotImageData", {
                    contents: b.imageUrl,
                    name: b.imageName
                }), this.trigger(document, "uiOverrideTweetBoxOptions", b)
            }
            this.open()
        }, this.setTitle = function(a) {
            this.select("modalTitleSelector").text(a)
        }, this.updateTitle = function(a, b) {
            b && b.title && this.setTitle(b.title)
        }, this.prepareTweetBox = function() {
            this.select("tweetBoxSelector").trigger("uiComposerResetAndFocus")
        }, this.preventDefaultDialogFocus = function(a) {
            this.isOpen() && this.$dialog.is(a.target) && a.preventDefault()
        }, this.triggerTweetDialogClosedEvent = function() {
            this.trigger(document, "uiTweetDialogClosed")
        }, this.after("initialize", function() {
            this.on(document, "uiDialogInitialFocus", this.preventDefaultDialogFocus), this.on(document, "uiShortcutShowTweetbox", this.openGlobalTweetDialog), this.on(document, "uiOpenTweetDialog", this.openTweetDialog), this.on(document, "uiOpenReplyDialog", this.openReply), this.on("uiTweetSent", this.close), this.on("uiDialogOpened", this.prepareTweetBox), this.on("uiDialogClosed", this.removeTweet), this.on("uiDialogClosed", this.triggerTweetDialogClosedEvent), this.on("uiDialogUpdateTitle", this.updateTitle)
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), withDialog = require("app/ui/with_dialog"), withModalTweet = require("app/ui/dialogs/with_modal_tweet"), userInfo = require("app/data/user_info"), _ = require("core/i18n"), TweetDialog = defineComponent(tweetDialog, withDialog, withModalTweet);
    module.exports = TweetDialog
});
define("app/ui/new_tweet_button", ["module", "require", "exports", "core/component", "app/data/with_scribe", "app/utils/user_dom_data"], function(module, require, exports) {
    function newTweetButton() {
        this.defaultAttrs({
            draftTweetIdAttr: "draft-tweet-id",
            dialogTitleAttr: "dialog-title",
            defaultTextAttr: "default-text",
            setCursorToBeginningAttr: "cursor-beginning"
        }), this.openTweetDialog = function() {
            this.scribe({
                element: "tweet_button",
                action: "click"
            }, this.attr.eventData), this.trigger("uiOpenTweetDialog", this.tweetDialogData())
        }, this.tweetDialogData = function() {
            return {
                draftTweetId: this.$node.data(this.attr.draftTweetIdAttr) || "global",
                screenName: userDomDataUtil.getScreenName(this.$node),
                title: this.$node.data(this.attr.dialogTitleAttr),
                text: this.$node.data(this.attr.defaultTextAttr),
                setCursorToBeginning: this.$node.data(this.attr.setCursorToBeginningAttr)
            }
        }, this.after("initialize", function() {
            this.on("click", this.openTweetDialog)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), userDomDataUtil = require("app/utils/user_dom_data");
    module.exports = defineComponent(newTweetButton, withScribe)
});
define("app/data/tweet_box_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe"], function(module, require, exports) {
    function tweetBoxScribe() {
        var a = {
            tweetBox: {
                uiTweetboxTweetError: "failure",
                uiTweetboxTweetSuccess: "send_tweet",
                uiTweetboxGeotaggedTweetSuccess: "send_geotagged_tweet",
                uiTweetboxPhotoTweetSuccess: "send_photo_tweet",
                uiTweetboxReplySuccess: "send_reply",
                uiTweetboxDMSuccess: "send_dm",
                uiOpenTweetDialog: "compose_tweet"
            },
            imagePicker: {
                uiImagePickerClick: "click",
                uiImagePickerAdd: "add",
                uiImagePickerRemove: "remove",
                uiImagePickerError: "error",
                uiDrop: "drag_and_drop"
            },
            geoPicker: {
                uiGeoPickerOffer: "offer",
                uiGeoPickerEnable: "enable",
                uiGeoPickerOpen: "open",
                uiGeoPickerTurnOn: "on",
                uiGeoPickerTurnOff: "off",
                uiGeoPickerChange: "select",
                uiGeoPickerInteraction: "focus_field",
                uiGeoPickerSendTweet: "send_tweet_geo_interaction"
            },
            lifelineAlerts: {
                uiLifelineAlertsSelect: "select",
                uiLifelineAlertsImpression: "impression",
                uiLifelineAlertsTweet: "tweet",
                uiLifelineAlertsCancel: "cancel"
            }
        };
        this.after("initialize", function() {
            Object.keys(a.tweetBox).forEach(function(b) {
                this.scribeOnEvent(b, {
                    element: "tweet_box",
                    action: a.tweetBox[b]
                })
            }, this), Object.keys(a.imagePicker).forEach(function(b) {
                this.scribeOnEvent(b, {
                    element: "image_picker",
                    action: a.imagePicker[b]
                })
            }, this), Object.keys(a.geoPicker).forEach(function(b) {
                this.scribeOnEvent(b, {
                    element: "geo_picker",
                    action: a.geoPicker[b]
                })
            }, this), Object.keys(a.lifelineAlerts).forEach(function(b) {
                this.scribeOnEvent(b, {
                    element: "lifeline_alerts",
                    action: a.lifelineAlerts[b]
                })
            }, this)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(tweetBoxScribe, withScribe)
});
define("app/boot/tweet_boxes", ["module", "require", "exports", "app/data/geo", "app/data/tweet", "app/ui/tweet_dialog", "app/ui/new_tweet_button", "app/data/tweet_box_scribe", "app/ui/tweet_box_manager"], function(module, require, exports) {
    function initialize(a) {
        GeoData.attachTo(document, a), TweetData.attachTo(document, a), TweetDialog.attachTo("#global-tweet-dialog"), NewTweetButton.attachTo("#global-new-tweet-button", {
            eventData: {
                scribeContext: {
                    component: "top_bar",
                    element: "tweet_button"
                }
            }
        }), TweetBoxScribe.attachTo(document, a), TweetBoxManager.attachTo(document, a)
    }
    var GeoData = require("app/data/geo"), TweetData = require("app/data/tweet"), TweetDialog = require("app/ui/tweet_dialog"), NewTweetButton = require("app/ui/new_tweet_button"), TweetBoxScribe = require("app/data/tweet_box_scribe"), TweetBoxManager = require("app/ui/tweet_box_manager");
    module.exports = initialize
});
define("app/ui/user_dropdown", ["module", "require", "exports", "core/component", "app/ui/with_dropdownmenu", "app/ui/with_navigation_links_scribing", "app/utils/storage/core"], function(module, require, exports) {
    function userDropdown() {
        this.defaultAttrs({
            feedbackLinkSelector: ".feedback-callout-link"
        }), this.signout = function() {
            storage.clearAll(), this.clearIndexedDBs(function() {
                this.$signoutForm.submit()
            }.bind(this))
        }, this.clearIndexedDBHandle = function(a, b) {
            a.clearAllObjectStores(b)
        }, this.clearIndexedDBs = function(a) {
            function e() {
                c++, c == b.length && a()
            }
            if (!window.globalIndexedDBs || $.isEmptyObject(window.globalIndexedDBs)) {
                a();
                return 
            }
            var b = [], c = 0;
            for (var d in window.globalIndexedDBs)
                window.globalIndexedDBs.hasOwnProperty(d) && b.push(window.globalIndexedDBs[d]);
            b.forEach(function(a) {
                var b = {
                    onSuccess: function() {
                        e()
                    },
                    onError: function() {
                        e()
                    }
                };
                this.clearIndexedDBHandle(a, b)
            }, this)
        }, this.showKeyboardShortcutsDialog = function(a, b) {
            this.trigger(document, "uiOpenKeyboardShortcutsDialog"), a.preventDefault()
        }, this.showConversationNotification = function(a, b) {
            this.unreadThreads = b.threads, this.$node.addClass(this.attr.glowClass), this.$dmCount.addClass(this.attr.glowClass).text(b.threads.length)
        }, this.openFeedbackDialog = function(a, b) {
            this.closeDropdown(), this.trigger("uiPrepareFeedbackDialog", {})
        }, this.updateConversationNotication = function(a, b) {
            var c = $.inArray(b.recipient, this.unreadThreads);
            if (c===-1)
                return;
            this.unreadThreads.splice(c, 1);
            var d = parseInt(this.$dmCount.text(), 10) - 1;
            d ? this.$dmCount.text(d) : (this.$node.removeClass(this.attr.glowClass), this.$dmCount.removeClass(this.attr.glowClass).text(""))
        }, this.after("initialize", function() {
            this.unreadThreads = [], this.$signoutForm = this.select("signoutForm"), this.on(this.attr.keyboardShortcuts, "click", this.showKeyboardShortcutsDialog), this.on(this.attr.feedbackLinkSelector, "click", this.openFeedbackDialog), this.$dmCount = this.select("dmCount"), this.on(this.attr.signout, "click", this.signout), this.on(document, "uiDMDialogOpenedConversation", this.updateConversationNotication), this.on(document, "uiDMDialogHasNewConversations", this.showConversationNotification), this.on(document, "click", this.close), this.on(document, "uiNavigate", this.close)
        })
    }
    var defineComponent = require("core/component"), withDropdownMenu = require("app/ui/with_dropdownmenu"), withNavigationLinks = require("app/ui/with_navigation_links_scribing"), storage = require("app/utils/storage/core"), UserDropdown = defineComponent(userDropdown, withDropdownMenu);
    module.exports = UserDropdown
});
define("app/ui/signin_dropdown", ["module", "require", "exports", "core/component", "app/ui/with_dropdown"], function(module, require, exports) {
    function signinDropdown() {
        this.defaultAttrs({
            usernameSelector: ".email-input",
            dialogContainerSelector: ".signin-dialog-body"
        }), this.focusUsername = function() {
            this.applyARIADialogAttrs(), this.select("usernameSelector").focus()
        }, this.applyARIADialogAttrs = function() {
            this.select("dropDownMenuSelector").attr({
                role: "dialog",
                "aria-hidden": !1,
                "aria-labelledby": "signin-form-legend"
            })
        }, this.toggleARIAHiddenRole = function() {
            this.select("dropDownMenuSelector").attr("aria-hidden", !0)
        }, this.after("initialize", function() {
            this.on("uiDropdownOpened", this.focusUsername), this.on("uiDropdownClosed", this.toggleARIAHiddenRole)
        })
    }
    var defineComponent = require("core/component"), withDropdown = require("app/ui/with_dropdown"), SigninDropdown = defineComponent(signinDropdown, withDropdown);
    module.exports = SigninDropdown
});
define("app/utils/animate_window_scrolltop", ["module", "require", "exports"], function(module, require, exports) {
    function getScrollEl() {
        return scrollEl ? scrollEl : ([document.body, document.documentElement].forEach(function(a) {
            var b = a.scrollTop, c = b > 0 ? b - 1: b + 1;
            a.scrollTop = c, a.scrollTop == c && (scrollEl = a.tagName.toLowerCase(), a.scrollTop = b)
        }), scrollEl)
    }
    var scrollEl;
    module.exports = function(a, b, c) {
        $(getScrollEl()).animate({
            scrollTop: a
        }, b, c)
    }
});
define("app/ui/global_nav", ["module", "require", "exports", "core/component", "app/utils/full_path", "app/utils/animate_window_scrolltop"], function(module, require, exports) {
    function globalNav() {
        this.defaultAttrs({
            activeClass: "active",
            newClass: "new",
            newCountClass: "new-count",
            maxCountClass: "max",
            nav: "li",
            notificationsNav: "li.notifications",
            meNav: "li.profile",
            navLinkSelector: "li > a",
            linkSelector: "a",
            notificationsCountSelector: "li.people .count",
            notificationsCountInnerSelector: "li.people .count .count-inner",
            maxCountValue: 99,
            dynamicTooltipSelector: ".js-dynamic-tooltip"
        }), this.updateActive = function(a, b) {
            b && (this.select("nav").removeClass(this.attr.activeClass), this.select("nav").filter("[data-global-action=" + b.section + "]").addClass(this.attr.activeClass), this.removeGlowFromActive())
        }, this.isNotificationsActive = function() {
            return this.select("notificationsNav").hasClass(this.attr.activeClass)
        }, this.updateNotificationsCount = function(a, b) {
            b && typeof b == "number" && b > 0&&!this.isNotificationsActive() ? this.setNotificationsCount(b) : this.clearBadgeAndNotificationsCount()
        }, this.setNotificationsCount = function(a) {
            var b = this.select("notificationsCountSelector"), c = this.select("notificationsCountInnerSelector"), d = a > this.attr.maxCountValue;
            d && (a = this.attr.maxCountValue + "+"), b.toggleClass(this.attr.maxCountClass, d), c.text(a), b.addClass(this.attr.newCountClass)
        }, this.clearBadgeAndNotificationsCount = function() {
            var a = this.select("notificationsCountSelector");
            a.removeClass(this.attr.newCountClass), a.removeClass(this.attr.maxCountClass)
        }, this.addGlowToActive = function() {
            this.$node.find("." + this.attr.activeClass).addClass(this.attr.newClass)
        }, this.addGlowToMe = function() {
            this.select("meNav").addClass(this.attr.newClass)
        }, this.removeGlowFromActive = function() {
            this.$node.find("." + this.attr.activeClass).not(this.attr.meNav).removeClass(this.attr.newClass)
        }, this.removeGlowFromMe = function() {
            this.select("meNav").removeClass(this.attr.newClass)
        }, this.scrollToTopLink = function(a) {
            var b = $(a.target).closest(this.attr.linkSelector);
            b.attr("href") == fullPath() && (a.preventDefault(), b.blur(), this.scrollToTop())
        }, this.topBarBirdClick = function() {
            this.scrollToTop({
                wasTopBarBirdClick: !0
            })
        }, this.scrollToTop = function(a) {
            animateWinScrollTop(0, "fast"), this.trigger(document, "uiGotoTopOfScreen", a || {})
        }, this.addTooltip = function(a, b) {
            var c = $(b.el), d = c.find(".text");
            d.width() == 1&&!c.attr("data-original-title") && (d.addClass("u-hiddenVisually"), c.attr({
                "data-original-title": d.text().trim()
            }))
        }, this.removeTooltip = function(a, b) {
            var c = $(b.el), d = c.find(".text");
            c.attr("data-original-title") && (d.removeClass("u-hiddenVisually"), c.removeAttr("data-original-title"))
        }, this.after("initialize", function() {
            this.on(document, "uiAddPageCount", this.addGlowToActive), this.on(document, "uiHasInjectedNewTimeline", this.removeGlowFromActive), this.on(document, "uiUpdateConnectBadge", this.updateNotificationsCount), this.on(document, "dataPageRefresh", this.updateActive), this.on(".bird-topbar-etched", "click", this.topBarBirdClick), this.on("click", {
                navLinkSelector: this.scrollToTopLink
            }), this.on("mouseover focusin", {
                dynamicTooltipSelector: this.addTooltip
            }), this.on("mouseout focusout", {
                dynamicTooltipSelector: this.removeTooltip
            })
        })
    }
    var defineComponent = require("core/component"), fullPath = require("app/utils/full_path"), animateWinScrollTop = require("app/utils/animate_window_scrolltop"), GlobalNav = defineComponent(globalNav);
    module.exports = GlobalNav
});
define("app/ui/dm_envelope", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function dmEnvelope() {
        this.defaultAttrs({
            directMessagesSelector: ".global-dm-nav",
            msgCounterSelector: ".dm-new .count-inner",
            newMsgClass: "new",
            singleCount: "with-count",
            doubleCount: "with-count-2",
            tripleCount: "with-count-3"
        }), this.addGlowToEnvelope = function(a, b) {
            this.select("directMessagesSelector").addClass(this.attr.newMsgClass)
        }, this.removeGlowFromEnvelope = function(a, b) {
            this.select("directMessagesSelector").removeClass(this.attr.newMsgClass)
        }, this.addCountToEnvelope = function(a, b) {
            var c = parseInt(b.msgCount, 10);
            if (isNaN(c))
                return;
            var d = [this.attr.singleCount];
            c > 9 && d.push(this.attr.doubleCount), c > 99 && d.push(this.attr.tripleCount), this.removeCountFromEnvelope(a, c), this.select("directMessagesSelector").addClass(d.join(" ")), this.select("directMessagesSelector").find(this.attr.msgCounterSelector).text(c)
        }, this.removeCountFromEnvelope = function(a, b) {
            this.select("directMessagesSelector").removeClass(this.attr.singleCount + " " + this.attr.doubleCount + " " + this.attr.tripleCount)
        }, this.after("initialize", function() {
            this.on(document, "dataUserHasUnreadDMs dataUserHasUnreadDMsWithCount", this.addGlowToEnvelope), this.on(document, "dataUserHasNoUnreadDMs dataUserHasNoUnreadDMsWithCount", this.removeGlowFromEnvelope), this.on(document, "dataUserHasUnreadDMsWithCount", this.addCountToEnvelope), this.on(document, "dataUserHasNoUnreadDMsWithCount", this.removeCountFromEnvelope)
        })
    }
    var defineComponent = require("core/component"), DMEnvelope = defineComponent(dmEnvelope);
    module.exports = DMEnvelope
});
define("app/ui/search_input", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function searchInput() {
        this.defaultAttrs({
            magnifyingGlassSelector: ".js-search-action",
            inputFieldSelector: "#search-query",
            query: "",
            searchPathWithQuery: "/search?q=query&src=typd",
            focusClass: "focus"
        }), this.addFocusStyles = function(a) {
            this.$node.addClass(this.attr.focusClass), this.$input.addClass(this.attr.focusClass), this.trigger("uiSearchInputFocused")
        }, this.removeFocusStyles = function(a) {
            this.$node.removeClass(this.attr.focusClass), this.$input.removeClass(this.attr.focusClass)
        }, this.executeTypeaheadSelection = function(a, b) {
            b.source != "account" && this.$input.val(b.display);
            if (b.isClick)
                return;
            this.trigger("uiNavigate", {
                href: b.href
            })
        }, this.submitQuery = function(a, b) {
            this.trigger("uiSearchQuery", {
                query: b.query,
                source: "search"
            }), this.trigger("uiNavigate", {
                href: this.attr.searchPathWithQuery.replace("query", encodeURIComponent(b.query))
            })
        }, this.searchFormSubmit = function(a, b) {
            a.preventDefault(), this.trigger(this.$input, "uiTypeaheadInputSubmit")
        }, this.after("initialize", function() {
            this.$input = this.select("inputFieldSelector"), this.$input.val(this.attr.query), this.on("uiTypeaheadItemSelected", this.executeTypeaheadSelection), this.on("uiTypeaheadSubmitQuery", this.submitQuery), this.on(this.$input, "focus", this.addFocusStyles), this.on(this.$input, "blur", this.removeFocusStyles), this.on("submit", this.searchFormSubmit), this.on(this.select("magnifyingGlassSelector"), "click", this.searchFormSubmit)
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(searchInput)
});
define("app/data/search_input_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe", "app/utils/scribe_item_types"], function(module, require, exports) {
    function searchInputScribe() {
        var a = {
            account: function(a) {
                var b = {
                    message: a.input,
                    items: [{
                        id: a.query,
                        item_type: itemTypes.user,
                        position: a.index
                    }
                    ],
                    format_version: 2,
                    event_info: a.item ? a.item.origin: undefined
                };
                this.scribe("profile_click", a, b)
            },
            search: function(b) {
                if (this.lastCompletion && b.query === this.lastCompletion.query)
                    a.topics.call(this, this.lastCompletion);
                else {
                    var c = {
                        items: [{
                            item_query: b.query,
                            item_type: itemTypes.search
                        }
                        ],
                        format_version: 2
                    };
                    this.scribe("search", b, c)
                }
            },
            topics: function(a) {
                var b = {
                    message: a.input,
                    items: [{
                        item_query: a.query,
                        item_type: itemTypes.search,
                        position: a.index
                    }
                    ],
                    format_version: 2
                };
                this.scribe("search", a, b)
            },
            account_search: function(a) {
                this.scribe("people_search", a, {
                    query: a.input
                })
            },
            saved_search: function(a) {
                var b = {
                    message: a.input,
                    items: [{
                        item_query: a.query,
                        item_type: itemTypes.savedSearch,
                        position: a.index
                    }
                    ],
                    format_version: 2
                };
                this.scribe({
                    element: "saved_search",
                    action: "search"
                }, a, b)
            },
            recent_search: function(a) {
                var b = {
                    message: a.input,
                    items: [{
                        item_query: a.query,
                        item_type: itemTypes.search,
                        position: a.index
                    }
                    ],
                    format_version: 2
                };
                this.scribe({
                    element: "recent_search",
                    action: "search"
                }, a, b)
            },
            concierge: function(a) {
                var b = {
                    message: a.input,
                    items: [{
                        item_query: a.query,
                        item_type: itemTypes.search,
                        position: a.index
                    }
                    ],
                    format_version: 2
                };
                this.scribe({
                    element: "typeahead_concierge",
                    action: "search"
                }, a, b)
            }
        };
        this.storeCompletionData = function(a, b) {
            a.type == "uiTypeaheadItemSelected" || a.type == "uiSearchQuery" ? this.scribeSelection(a, b) : b.fromSelectionEvent || (this.lastCompletion = b)
        }, this.scribeSelection = function(b, c) {
            a[c.source] && a[c.source].call(this, c)
        }, this.after("initialize", function() {
            this.scribeOnEvent("uiSearchInputFocused", "focus_field"), this.on("uiTypeaheadItemComplete uiTypeaheadItemSelected uiSearchQuery", this.storeCompletionData)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), itemTypes = require("app/utils/scribe_item_types");
    module.exports = defineComponent(searchInputScribe, withScribe)
});
define("app/ui/login_form", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function loginForm() {
        this.defaultAttrs({
            forgotPasswordLinkSelector: ".forgot",
            accountIdentifierInputSelector: '[name="session[username_or_email]"]'
        }), this.navigateToPasswordReset = function(a) {
            a.stopPropagation(), a.preventDefault();
            var b = this.$accountIdentifierInput.val(), c = b ? "?account_identifier=" + encodeURIComponent(b): "";
            this.trigger("uiNavigate", {
                href: this.forgotPasswordHref + c
            })
        }, this.after("initialize", function() {
            this.$accountIdentifierInput = this.select("accountIdentifierInputSelector"), this.forgotPasswordHref = this.select("forgotPasswordLinkSelector").attr("href"), this.on("click", {
                forgotPasswordLinkSelector: this.navigateToPasswordReset
            })
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(loginForm)
});
define("app/ui/track_ga_events", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function trackGAEvent() {
        this.defaultAttrs({
            signinSelector: ".signin-wrapper, .js-front-signin, .signin-dialog-body",
            incompleteNUXSelector: ".StartPreviewLinks .js-skip-step"
        }), this.trackIncompleteNUX = function() {
            this.trigger("GATrackEvent", {
                gaEvent: "NUX-not-complete"
            })
        }, this.trackExistingUserSignin = function() {
            this.trigger("GATrackEvent", {
                gaEvent: "existing-user-signin"
            })
        }, this.after("initialize", function() {
            this.on("submit", {
                signinSelector: this.trackExistingUserSignin
            }), this.on("click", {
                incompleteNUXSelector: this.trackIncompleteNUX
            })
        })
    }
    var defineComponent = require("core/component"), TrackGAEvent = defineComponent(trackGAEvent);
    module.exports = TrackGAEvent
});
define("app/ui/macaw_nymizer_signin_conversion", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function macawNymizerSigninConversion() {
        this.defaultAttrs({
            signinSelector: ".signin-wrapper, .js-front-signin, .signin-dialog-body"
        }), this.signinConversion = function() {
            this.trigger("MNConversion", {
                mnType: "signin"
            })
        }, this.after("initialize", function() {
            this.on("submit", {
                signinSelector: this.signinConversion
            })
        })
    }
    var defineComponent = require("core/component"), MacawNymizerSigninConversion = defineComponent(macawNymizerSigninConversion);
    module.exports = MacawNymizerSigninConversion
});
define("app/boot/top_bar", ["module", "require", "exports", "app/boot/tweet_boxes", "app/ui/user_dropdown", "app/ui/signin_dropdown", "app/ui/global_nav", "app/ui/dm_envelope", "app/ui/navigation_links", "app/ui/typeahead/typeahead_dropdown", "app/ui/typeahead/typeahead_input", "app/ui/search_input", "app/data/search_input_scribe", "app/ui/login_form", "core/utils", "app/ui/track_ga_events", "app/ui/macaw_nymizer_signin_conversion"], function(module, require, exports) {
    function initialize(a) {
        GlobalNav.attachTo(".global-nav", {
            noTeardown: !0
        }), DMEnvelope.attachTo(".dm-nav", {
            noTeardown: !0
        }), SearchInput.attachTo("#global-nav-search", utils.merge(a, {
            eventData: {
                scribeContext: {
                    component: "top_bar_searchbox",
                    element: ""
                }
            }
        })), SearchInputScribe.attachTo("#global-nav-search", {
            noTeardown: !0
        });
        var b = [["recentSearches", ["recentSearches"]], ["savedSearches", ["savedSearches"]], ["topics", ["topics"]], ["accounts", ["accounts"]]];
        a.typeaheadData.accountsOnTop && (b = [["recentSearches", ["recentSearches"]], ["savedSearches", ["savedSearches"]], ["accounts", ["accounts"]], ["topics", ["topics"]]]), TypeaheadInput.attachTo("#global-nav-search"), TypeaheadDropdown.attachTo("#global-nav-search", {
            datasourceRenders: b,
            accountsShortcutShow: !0,
            autocompleteAccounts: !1,
            typeaheadSrc: "SEARCH_BOX",
            deciders: utils.merge(a.typeaheadData, {
                showSocialContext: a.typeaheadData.showSearchAccountSocialContext
            }),
            eventData: {
                scribeContext: {
                    component: "top_bar_searchbox",
                    element: "typeahead"
                }
            }
        }), a.loggedIn ? (tweetBoxes(a), UserDropdown.attachTo("#user-dropdown", {
            noTeardown: !0,
            signout: "#signout-button",
            signoutForm: "#signout-form",
            keyboardShortcuts: ".js-keyboard-shortcut-trigger",
            dmCount: ".js-direct-message-count",
            glowClass: "new"
        })) : (a.passwordResetAdvancedLoginForm && LoginForm.attachTo(".signin-dialog-body"), SigninDropdown.attachTo(".js-session")), NavigationLinks.attachTo(".global-nav", {
            noTeardown: !0,
            eventData: {
                scribeContext: {
                    component: "top_bar"
                }
            }
        }), GATrackEvent.attachTo(".signin-dialog-body"), MacawNymizerSigninConversion.attachTo(".signin-dialog-body")
    }
    var tweetBoxes = require("app/boot/tweet_boxes"), UserDropdown = require("app/ui/user_dropdown"), SigninDropdown = require("app/ui/signin_dropdown"), GlobalNav = require("app/ui/global_nav"), DMEnvelope = require("app/ui/dm_envelope"), NavigationLinks = require("app/ui/navigation_links"), TypeaheadDropdown = require("app/ui/typeahead/typeahead_dropdown"), TypeaheadInput = require("app/ui/typeahead/typeahead_input"), SearchInput = require("app/ui/search_input"), SearchInputScribe = require("app/data/search_input_scribe"), LoginForm = require("app/ui/login_form"), utils = require("core/utils"), GATrackEvent = require("app/ui/track_ga_events"), MacawNymizerSigninConversion = require("app/ui/macaw_nymizer_signin_conversion");
    module.exports = initialize
});
define("app/data/typeahead/with_cache", ["module", "require", "exports"], function(module, require, exports) {
    function WithCache() {
        this.defaultAttrs({
            cache_limit: 10
        }), this.getCachedSuggestions = function(a) {
            return this.cache[a] ? this.cache[a].value : null
        }, this.clearCache = function() {
            this.cache = {
                NEWEST: null,
                OLDEST: null,
                COUNT: 0,
                LIMIT: this.attr.cache_limit
            }
        }, this.deleteCachedSuggestions = function(a) {
            return this.cache[a] ? (this.cache.COUNT > 1 && (a == this.cache.NEWEST.query ? (this.cache.NEWEST = this.cache.NEWEST.before, this.cache.NEWEST.after = null) : a == this.cache.OLDEST.query ? (this.cache.OLDEST = this.cache.OLDEST.after, this.cache.OLDEST.before = null) : (this.cache[a].after.before = this.cache[a].before, this.cache[a].before.after = this.cache[a].after)), delete this.cache[a], this.cache.COUNT -= 1, !0) : !1
        }, this.setCachedSuggestions = function(a, b) {
            if (this.cache.LIMIT === 0)
                return;
            this.deleteCachedSuggestions(a), this.cache.COUNT >= this.cache.LIMIT && this.deleteCachedSuggestions(this.cache.OLDEST.query), this.cache.COUNT == 0 ? (this.cache[a] = {
                query: a,
                value: b,
                before: null,
                after: null
            }, this.cache.NEWEST = this.cache[a], this.cache.OLDEST = this.cache[a]) : (this.cache[a] = {
                query: a,
                value: b,
                before: this.cache.NEWEST,
                after: null
            }, this.cache.NEWEST.after = this.cache[a], this.cache.NEWEST = this.cache[a]), this.cache.COUNT += 1
        }, this.aroundGetSuggestions = function(a, b, c) {
            var d = c.id + ":" + c.query, e = this.getCachedSuggestions(d);
            if (e) {
                this.triggerSuggestionsEvent(c.id, c.query, e);
                return 
            }
            a(b, c)
        }, this.afterTriggerSuggestionsEvent = function(a, b, c, d) {
            if (d)
                return;
            var e = a + ":" + b;
            this.setCachedSuggestions(e, c)
        }, this.after("triggerSuggestionsEvent", this.afterTriggerSuggestionsEvent), this.around("getSuggestions", this.aroundGetSuggestions), this.after("initialize", function() {
            this.clearCache(), this.on("uiTypeaheadDeleteRecentSearch", this.clearCache)
        })
    }
    module.exports = WithCache
});
define("app/data/with_datasource_helpers", ["module", "require", "exports", "app/data/user_info"], function(module, require, exports) {
    function withDatasourceHelpers() {
        this.prefetch = function(a, b, c, d) {
            var e = {
                prefetch: !0,
                result_type: b,
                count: this.getPrefetchCount(),
                media_tagging_in_prefetch: this.shouldIncludeCanMediaTag(),
                experiments: !!this.attr.typeaheadExperiments && this.attr.typeaheadExperiments.join(",")
            };
            this.storage && (e[b + "_cache_age"] = this.storage.getCacheAge(this.attr.storageHash, this.attr.ttl_ms)), this.get({
                url: a,
                headers: {
                    "X-Phx": !0
                },
                data: e,
                eventData: {},
                success: c,
                error: d
            })
        }, this.useStaleData = function() {
            this.storage && (this.extendTTL(), this.getDataFromLocalStorage())
        }, this.extendTTL = function() {
            var a = this.getStorageKeys();
            for (var b = 0; b < a.length; b++)
                this.storage.updateTTL(a[b], this.attr.ttl_ms)
        }, this.loadData = function(a, b) {
            var c = this.getStorageKeys().some(function(a) {
                return this.storage.isExpired(a)
            }, this);
            c || this.isMetadataExpired() ? this.prefetch(a, b, this.processResults.bind(this), this.useStaleData.bind(this)) : this.getDataFromLocalStorage()
        }, this.isIE8 = function() {
            return $.browser.msie && parseInt($.browser.version, 10) === 8
        }, this.getProtocol = function() {
            return window.location.protocol
        }, this.shouldIncludeCanMediaTag = function() {
            return userInfo.getDecider("enable_media_tag_prefetch")
        }
    }
    var userInfo = require("app/data/user_info");
    module.exports = withDatasourceHelpers
});
define("app/data/typeahead/accounts_datasource", ["module", "require", "exports", "app/utils/typeahead_helpers", "app/utils/storage/custom", "app/data/with_data", "core/compose", "core/utils", "app/data/with_datasource_helpers"], function(module, require, exports) {
    function accountsDatasource(a) {
        this.attr = {
            id: null,
            ttl_ms: 1728e5,
            localStorageCount: 1200,
            ie8LocalStorageCount: 1e3,
            limit: 6,
            version: 4,
            localQueriesEnabled: !1,
            remoteQueriesEnabled: !1,
            onlyDMable: !1,
            onlyShowUsersWithCanMediaTag: !1,
            currentUserId: null,
            storageAdjacencyList: "userAdjacencyList",
            storageHash: "userHash",
            storageProtocol: "protocol",
            storageVersion: "userVersion",
            remotePath: "/i/search/typeahead.json",
            remoteType: "users",
            prefetchPath: "/i/search/typeahead.json",
            prefetchType: "users",
            storageBlackList: "userBlackList",
            maxLengthBlacklist: 100,
            typeaheadExperiments: []
        }, this.attr = util.merge(this.attr, a), this.after = $.noop, this.defaultAttrs = $.noop, compose.mixin(this, [withData, withDatasourceHelpers]), this.getPrefetchCount = function() {
            return this.isIE8() && this.attr.localStorageCount > this.attr.ie8LocalStorageCount ? this.attr.ie8LocalStorageCount : this.attr.localStorageCount
        }, this.isMetadataExpired = function() {
            var a = this.storage.getItem(this.attr.storageVersion), b = this.storage.getItem(this.attr.storageProtocol);
            return a == this.attr.version && b == this.getProtocol()?!1 : !0
        }, this.getStorageKeys = function() {
            return [this.attr.storageVersion, this.attr.storageHash, this.attr.storageAdjacencyList, this.attr.storageProtocol]
        }, this.getDataFromLocalStorage = function() {
            this.userHash = this.storage.getItem(this.attr.storageHash) || this.userHash, this.adjacencyList = this.storage.getItem(this.attr.storageAdjacencyList) || this.adjacencyList
        }, this.processResults = function(a) {
            if (!a ||!a[this.attr.prefetchType]) {
                this.useStaleData();
                return 
            }
            a[this.attr.prefetchType].forEach(function(a) {
                a.tokens = a.tokens.map(function(a) {
                    return typeof a == "string" ? a : a.token
                }), a.prefetched=!0, this.userHash[a.id] = a, a.tokens.forEach(function(b) {
                    var c = helpers.getFirstChar(b);
                    c.trim().length && (this.adjacencyList[c] === undefined && (this.adjacencyList[c] = []), this.adjacencyList[c].indexOf(a.id)===-1 && this.adjacencyList[c].push(a.id))
                }, this)
            }, this), this.storage.setItem(this.attr.storageHash, this.userHash, this.attr.ttl_ms), this.storage.setItem(this.attr.storageAdjacencyList, this.adjacencyList, this.attr.ttl_ms), this.storage.setItem(this.attr.storageVersion, this.attr.version, this.attr.ttl_ms), this.storage.setItem(this.attr.storageProtocol, this.getProtocol(), this.attr.ttl_ms)
        }, this.getLocalSuggestions = function(a, b) {
            if (!this.attr.localQueriesEnabled) {
                b([]);
                return 
            }
            var c = helpers.tokenizeText(a.query.replace("@", "")), d = this.getPotentiallyMatchingIds(c), e = this.getAccountsFromIds(d), f = e.filter(this.matcher(c));
            f.sort(this.sorter), f = f.slice(0, this.attr.limit), b(f)
        }, this.getPotentiallyMatchingIds = function(a) {
            var b = [];
            return a.map(function(a) {
                var c = this.adjacencyList[helpers.getFirstChar(a)];
                c && (b = b.concat(c))
            }, this), b = util.uniqueArray(b), b
        }, this.getAccountsFromIds = function(a) {
            var b = [];
            return a.forEach(function(a) {
                var c = this.userHash[a];
                c && b.push(c)
            }, this), b
        }, this.matcher = function(a) {
            return function(b) {
                var c = b.tokens, d = [];
                if (this.attr.onlyDMable&&!b.is_dm_able)
                    return !1;
                if (!this.isEligibleForMediaTagSuggestion(b))
                    return !1;
                var e = a.every(function(a) {
                    var b = c.filter(function(b) {
                        return b.indexOf(a) === 0
                    });
                    return b.length
                });
                if (e)
                    return b
            }.bind(this)
        }, this.sorter = function(a, b) {
            function e(a, b) {
                var c = a.score_boost ? a.score_boost: 0, d = b.score_boost ? b.score_boost: 0, e = a.rounded_score ? a.rounded_score: 0, f = b.rounded_score ? b.rounded_score: 0;
                return f + d - (e + c)
            }
            var c = a.prefetched, d = b.prefetched;
            return c&&!d?-1 : d&&!c ? 1 : e(a, b)
        }, this.processRemoteSuggestions = function(a, b, c) {
            var d = c[this.attr.id] || [], e = {};
            return d.forEach(function(a) {
                e[a.id]=!0
            }, this), this.requiresRemoteSuggestions(a.queryData) && b[this.attr.remoteType] && b[this.attr.remoteType].forEach(function(a) {
                !e[a.id] && (!this.attr.onlyDMable || a.is_dm_able) && this.isEligibleForMediaTagSuggestion(a) && d.push(a)
            }, this), c[this.attr.id] = d.slice(0, this.attr.limit), c[this.attr.id].forEach(function(a) {
                this.removeBlacklistSocialContext(a)
            }, this), c
        }, this.removeBlacklistSocialContext = function(a) {
            a.first_connecting_user_id in this.socialContextBlackList && (a.first_connecting_user_name = undefined, a.first_connecting_user_id = undefined)
        }, this.requiresRemoteSuggestions = function(a) {
            return !a ||!a.query?!1 : a.accountsWithoutAtSignLocalOnly ? this.attr.remoteQueriesEnabled && (helpers.getFirstChar(a.query) == "@" || a.atSignRemoved) : this.attr.remoteQueriesEnabled
        }, this.isEligibleForMediaTagSuggestion = function(a) {
            return !this.attr.onlyShowUsersWithCanMediaTag || a.can_media_tag || a.id == this.attr.currentUserId
        }, this.initialize = function() {
            var a = customStorage({
                withExpiry: !0
            });
            this.storage = new a("typeahead"), this.adjacencyList = {}, this.userHash = {}, this.attr.localQueriesEnabled && this.loadData(this.attr.prefetchPath, this.attr.prefetchType), this.socialContextBlackList = this.storage.getItem(this.attr.storageBlackList) || {}
        }, this.initialize()
    }
    var helpers = require("app/utils/typeahead_helpers"), customStorage = require("app/utils/storage/custom"), withData = require("app/data/with_data"), compose = require("core/compose"), util = require("core/utils"), withDatasourceHelpers = require("app/data/with_datasource_helpers");
    module.exports = accountsDatasource
});
define("app/data/typeahead/accounts_datasource_async", ["module", "require", "exports", "app/utils/typeahead_helpers", "app/data/with_data", "core/compose", "core/utils", "app/data/with_datasource_helpers", "app/utils/storage/custom"], function(module, require, exports) {
    function accountsDatasourceAsync(a, b) {
        this.attr = {
            id: null,
            ttlMilliseconds: 1728e5,
            itemCount: 1200,
            limit: 6,
            version: 1,
            localQueriesEnabled: !1,
            remoteQueriesEnabled: !1,
            onlyDMable: !1,
            onlyShowUsersWithCanMediaTag: !1,
            currentUserId: null,
            remotePath: "/i/search/typeahead.json",
            remoteType: "users",
            prefetchPath: "/i/search/typeahead.json",
            prefetchType: "users",
            typeaheadExperiments: [],
            ttlKeyName: "ttl",
            metadataObjectstoreName: "MetadataStore",
            idsToAccountsObjectstoreName: "IdsToAccounts",
            accountTokensToIdsObjectstoreName: "AccountTokensToIds",
            ttlObjectstoreKeyPath: "dataType",
            tokenKeyPath: "token",
            idKeyPath: "id",
            storageBlackList: "userBlackList",
            maxLengthBlacklist: 100
        }, this.attr = util.merge(this.attr, a), this.after = $.noop, this.defaultAttrs = $.noop, compose.mixin(this, [withData, withDatasourceHelpers]), this.getPrefetchCount = function() {
            return this.attr.itemCount
        }, this.isMetadataExpired = function(a) {
            if (!a)
                return !0;
            this.objectStoresArePopulated=!0;
            var b = a.version !== this.attr.version, c = this.hasTTLExpired(a.expirationTime), d = a.protocol !== this.getProtocol();
            return b || c || d
        }, this.getCandidateIdsFoundInAllTermGroups = function(a, b) {
            return a.filter(function(a) {
                return b.every(function(b) {
                    return b.indexOf(a)!==-1
                })
            })
        }, this.createIdGroups = function(a) {
            var b = [], c = [];
            return a.forEach(function(a) {
                var d = [];
                a.forEach(function(a) {
                    b = b.concat(a.ids), d = d.concat(a.ids)
                }), c.push(d)
            }), {
                candidateIds: b,
                allIdGroupsByTerm: c
            }
        }, this.processQueryResults = function(a, b) {
            if (a.length == 0) {
                b.onSuccess([]);
                return 
            }
            var c = this.createIdGroups(a), d = util.uniqueArray(c.candidateIds), e = this.getCandidateIdsFoundInAllTermGroups(d, c.allIdGroupsByTerm);
            if (e.length == 0) {
                b.onSuccess([]);
                return 
            }
            this.getAccountObjects(e, b)
        }, this.filterAndSortAccounts = function(a) {
            var b = a.filter(function(a) {
                return this.attr.onlyDMable&&!a.is_dm_able?!1 : this.isEligibleForMediaTagSuggestion(a)?!0 : !1
            }, this);
            return b.sort(this.sorter), b
        }, this.getAccountObjects = function(a, b) {
            var c = [], d = 0;
            a.forEach(function(e) {
                var f = {
                    key: e,
                    objectStoreName: this.attr.idsToAccountsObjectstoreName,
                    onSuccess: function(e) {
                        d++, c.push(e.account);
                        if (d == a.length) {
                            var f = this.filterAndSortAccounts(c);
                            b.onSuccess(f.slice(0, this.attr.limit))
                        }
                    }.bind(this),
                    onError: function() {
                        d++, d == a.length && b.onSuccess(c)
                    }.bind(this)
                };
                this.client.getItemByKey(f)
            }, this)
        }, this.getResults = function(a) {
            var b = [], c = [], d = 0, e = a.terms;
            if (e.length == 0) {
                a.onSuccess([]);
                return 
            }
            e.forEach(function(c) {
                var f = {
                    prefix: c,
                    objectStoreName: this.attr.accountTokensToIdsObjectstoreName,
                    onSuccess: function(c) {
                        d++, b.push(c), d == e.length && this.processQueryResults(b, a)
                    }.bind(this),
                    onError: function() {
                        d++, d == e.length && this.processQueryResults(b, a)
                    }.bind(this)
                };
                this.client.getItemsByPrefix(f)
            }, this)
        }, this.handleInitializationError = function() {
            this.client && (this.client.closeConnection(), this.client = undefined), b.callback()
        }, this.createIdsToAccountsObjectstore = function() {
            this.createObjectStore(this.createAccountTokensToIdsObjectstore.bind(this), this.attr.idsToAccountsObjectstoreName, this.attr.idKeyPath)
        }, this.createAccountTokensToIdsObjectstore = function() {
            this.createObjectStore(this.createMetadataObjectstore.bind(this), this.attr.accountTokensToIdsObjectstoreName, this.attr.tokenKeyPath)
        }, this.createMetadataObjectstore = function() {
            this.createObjectStore(this.possiblyPrefetch.bind(this), this.attr.metadataObjectstoreName, this.attr.ttlObjectstoreKeyPath)
        }, this.clearMetadataObjectstore = function(a) {
            this.clearItems(this.clearAccountTokensToIdsObjectstore.bind(this, a), this.attr.metadataObjectstoreName)
        }, this.clearAccountTokensToIdsObjectstore = function(a) {
            this.clearItems(this.clearIdsToAccountsObjectstore.bind(this, a), this.attr.accountTokensToIdsObjectstoreName)
        }, this.clearIdsToAccountsObjectstore = function(a) {
            this.clearItems(this.populateIdsToAccountObjectstore.bind(this, a), this.attr.idsToAccountsObjectstoreName)
        }, this.populateIdsToAccountObjectstore = function(a) {
            var b = this.createItemsForIdsToAccountsObjectstore(a);
            this.addItems(this.populateAccountTokensToIdsObjectstore.bind(this, a), this.attr.idsToAccountsObjectstoreName, b)
        }, this.populateAccountTokensToIdsObjectstore = function(a) {
            var b = this.createItemsForAccountTokensToIdsObjectstore(a);
            this.addItems(this.populateAccountMetadataObjectstore.bind(this, a), this.attr.accountTokensToIdsObjectstoreName, b)
        }, this.populateAccountMetadataObjectstore = function(a) {
            var c = this.createItemsForMetadataObjectstore(a);
            this.addItems(function() {
                b.callback(this.client)
            }.bind(this), this.attr.metadataObjectstoreName, c)
        }, this.addItems = function(a, b, c) {
            var d = {
                objectStoreName: b,
                onSuccess: function() {
                    a()
                }.bind(this),
                onError: function() {
                    this.handleInitializationError()
                }.bind(this),
                items: c
            };
            this.client.addItems(d)
        }, this.clearItems = function(a, b) {
            var c = {
                objectStoreName: b,
                onSuccess: function() {
                    a()
                }.bind(this),
                onError: function() {
                    this.handleInitializationError()
                }.bind(this)
            };
            this.client.clearObjectStore(c)
        }, this.createObjectStore = function(a, b, c) {
            var d = {
                objectStoreName: b,
                onSuccess: function() {
                    a()
                }.bind(this),
                onError: function() {
                    this.handleInitializationError()
                }.bind(this),
                onBlocked: function() {
                    this.handleInitializationError()
                }.bind(this),
                keyPathName: c
            };
            this.client.createObjectStore(d)
        }, this.createAndPopulateObjectstoresIfNeeded = function() {
            this.client.checkIfObjectStoreExists(this.attr.idsToAccountsObjectstoreName) ? this.client.checkIfObjectStoreExists(this.attr.accountTokensToIdsObjectstoreName) ? this.client.checkIfObjectStoreExists(this.attr.metadataObjectstoreName) ? this.possiblyPrefetch() : this.createMetadataObjectstore() : this.createAccountTokensToIdsObjectstore() : this.createIdsToAccountsObjectstore()
        }, this.processPrefetchedResults = function(a) {
            this.clearMetadataObjectstore(a)
        }, this.useStaleDataIfPossible = function() {
            this.objectStoresArePopulated ? b.callback(this.client) : this.handleInitializationError()
        }, this.prefetchErrorHandler = function() {
            this.useStaleDataIfPossible()
        }, this.possiblyPrefetch = function() {
            var a = {
                objectStoreName: this.attr.metadataObjectstoreName,
                key: this.attr.ttlKeyName,
                onSuccess: function(a) {
                    this.isMetadataExpired(a) ? this.prefetch(this.attr.prefetchPath, this.attr.prefetchType, this.processPrefetchedResults.bind(this), this.prefetchErrorHandler.bind(this)) : b.callback(this.client)
                }.bind(this),
                onError: function() {
                    this.handleInitializationError()
                }.bind(this)
            };
            this.client.getItemByKey(a)
        }, this.hasTTLExpired = function(a) {
            return a - this.now() <= 0
        }, this.now = function() {
            return (new Date).getTime()
        }, this.addAccount = function(a) {
            var b = {
                account: a,
                onSuccess: function(a, b) {
                    this.addTokenIds(a, b)
                }.bind(this),
                onError: function() {}
            };
            this.getTokenIdsForAccount(b)
        }, this.getAccount = function(a) {
            var b = {
                objectStoreName: this.attr.idsToAccountsObjectstoreName,
                key: a.id,
                onSuccess: function(b) {
                    a.onSuccess(b)
                }.bind(this),
                onError: function() {}
            };
            this.client.getItemByKey(b)
        }, this.removeAccount = function(a) {
            var b = {
                id: a,
                onSuccess: function(a) {
                    a && this.removeAccountData(a)
                }.bind(this),
                onError: function() {}
            };
            this.getAccount(b)
        }, this.removeAccountData = function(a) {
            var b = {
                account: a.account,
                onSuccess: function(a, b) {
                    this.removeTokenIds(a, b)
                }.bind(this),
                onError: function() {}
            };
            this.getTokenIdsForAccount(b)
        }, this.getTokenIdsForAccount = function(a) {
            var b = [], c = 0, d=!1, e = a.account.tokens;
            e.forEach(function(f) {
                var g = typeof f == "string" ? f: f.token;
                if (!g.trim().length)
                    return;
                var h = {
                    objectStoreName: this.attr.accountTokensToIdsObjectstoreName,
                    key: g,
                    onSuccess: function(f) {
                        var h;
                        f ? h = f : (h = {}, h[this.attr.tokenKeyPath] = g, h.ids = []), b.push(h), c++, c == e.length && (d ? a.onError() : a.onSuccess(b, a.account))
                    }.bind(this),
                    onError: function() {
                        c++, d=!0, c == e.length && a.onError()
                    }.bind(this)
                };
                this.client.getItemByKey(h)
            }, this)
        }, this.addTokenIds = function(a, b) {
            a.forEach(function(a) {
                a.ids.indexOf(b.id)==-1 && a.ids.push(b.id)
            }, this);
            var c = {
                objectStoreName: this.attr.accountTokensToIdsObjectstoreName,
                items: a,
                onSuccess: function() {
                    this.addToIdsToAccounts(b)
                }.bind(this),
                onError: function() {}
            };
            this.client.upsertItems(c)
        }, this.removeTokenIds = function(a, b) {
            var c = a.map(function(a) {
                return a.ids = a.ids.filter(function(a) {
                    return a != b.id
                }), a
            }, this), d = {
                objectStoreName: this.attr.accountTokensToIdsObjectstoreName,
                items: c,
                onSuccess: function() {
                    this.removeFromIdsToAccounts(b)
                }.bind(this),
                onError: function() {}
            };
            this.client.upsertItems(d)
        }, this.addToIdsToAccounts = function(a) {
            var b = {};
            b[this.attr.idKeyPath] = a.id, b.account = a;
            var c = {
                objectStoreName: this.attr.idsToAccountsObjectstoreName,
                items: [b],
                onSuccess: function() {},
                onError: function() {}
            };
            this.client.upsertItems(c)
        }, this.removeFromIdsToAccounts = function(a) {
            var b = {
                objectStoreName: this.attr.idsToAccountsObjectstoreName,
                key: a.id,
                onSuccess: function() {},
                onError: function() {}
            };
            this.client.deleteItem(b)
        }, this.getLocalSuggestions = function(a, b) {
            if (!this.attr.localQueriesEnabled || a.query === "") {
                b([]);
                return 
            }
            var c = helpers.tokenizeText(a.query.replace("@", ""));
            if (this.client) {
                var d = {
                    onSuccess: function(a) {
                        b(a)
                    },
                    onError: function() {
                        b([])
                    },
                    terms: c
                };
                this.getResults(d)
            } else 
                b([])
        }, this.sorter = function(a, b) {
            function e(a, b) {
                var c = a.score_boost ? a.score_boost: 0, d = b.score_boost ? b.score_boost: 0, e = a.rounded_score ? a.rounded_score: 0, f = b.rounded_score ? b.rounded_score: 0;
                return f + d - (e + c)
            }
            var c = a.prefetched, d = b.prefetched;
            return c&&!d?-1 : d&&!c ? 1 : e(a, b)
        }, this.processRemoteSuggestions = function(a, b, c) {
            var d = c[this.attr.id] || [], e = {};
            return d.forEach(function(a) {
                e[a.id]=!0
            }, this), this.requiresRemoteSuggestions(a.queryData) && b[this.attr.remoteType] && b[this.attr.remoteType].forEach(function(a) {
                !e[a.id] && (!this.attr.onlyDMable || a.is_dm_able) && this.isEligibleForMediaTagSuggestion(a) && d.push(a)
            }, this), c[this.attr.id] = d.slice(0, this.attr.limit), c[this.attr.id].forEach(function(a) {
                this.removeBlacklistSocialContext(a)
            }, this), c
        }, this.removeBlacklistSocialContext = function(a) {
            a.first_connecting_user_id in this.socialContextBlackList && (a.first_connecting_user_name = undefined, a.first_connecting_user_id = undefined)
        }, this.requiresRemoteSuggestions = function(a) {
            return !a ||!a.query?!1 : a.accountsWithoutAtSignLocalOnly ? this.attr.remoteQueriesEnabled && (helpers.getFirstChar(a.query) == "@" || a.atSignRemoved) : this.attr.remoteQueriesEnabled
        }, this.isEligibleForMediaTagSuggestion = function(a) {
            return !this.attr.onlyShowUsersWithCanMediaTag || a.can_media_tag || a.id == this.attr.currentUserId
        }, this.createItemsForIdsToAccountsObjectstore = function(a) {
            var b = a[this.attr.prefetchType].map(function(a) {
                a.prefetched=!0;
                var b = {};
                return b[this.attr.idKeyPath] = a.id, b.account = a, b
            }, this);
            return b
        }, this.createItemsForAccountTokensToIdsObjectstore = function(a) {
            var b = {}, c = [];
            a[this.attr.prefetchType].forEach(function(a) {
                a.tokens.forEach(function(c) {
                    var d = c.token;
                    d != "" && (d in b ? b[d].push(a[this.attr.idKeyPath]) : b[d] = [a[this.attr.idKeyPath]])
                }, this)
            }, this);
            for (token in b)
                if (b.hasOwnProperty(token)) {
                    var d = {};
                    d.ids = b[token], d[this.attr.tokenKeyPath] = token, c.push(d)
                }
            return c
        }, this.createItemsForMetadataObjectstore = function(a) {
            var b = {};
            return b[this.attr.ttlObjectstoreKeyPath] = this.attr.ttlKeyName, b.expirationTime = this.attr.ttlMilliseconds + (new Date).getTime(), b.version = this.attr.version, b.protocol = this.getProtocol(), [b]
        }, this.initializeSocialContextBlacklist = function() {
            var a = customStorage({
                withExpiry: !0
            });
            this.storage = new a("typeahead"), this.socialContextBlackList = this.storage.getItem(this.attr.storageBlackList) || {}
        }, this.initialize = function() {
            this.isUsingIndexedDB=!0, this.initializeSocialContextBlacklist();
            var a = {
                onSuccess: function(a) {
                    this.client = a, this.attr.localQueriesEnabled ? this.createAndPopulateObjectstoresIfNeeded() : b.callback(a)
                }.bind(this),
                onError: function() {
                    this.handleInitializationError()
                }.bind(this),
                dbName: b.dbName
            };
            b.indexedDBWrapper(a)
        }
    }
    var helpers = require("app/utils/typeahead_helpers"), withData = require("app/data/with_data"), compose = require("core/compose"), util = require("core/utils"), withDatasourceHelpers = require("app/data/with_datasource_helpers"), customStorage = require("app/utils/storage/custom");
    module.exports = accountsDatasourceAsync
});
define("app/data/typeahead/contacts_datasource", ["module", "require", "exports", "app/utils/typeahead_helpers", "app/utils/storage/custom", "app/data/with_data", "core/compose", "core/utils", "app/data/with_datasource_helpers"], function(module, require, exports) {
    function contactsDatasource(a) {
        this.attr = {
            id: null,
            limit: 4,
            prefetchCount: 2500,
            prefetchPath: "/i/contacts/address_book_contacts",
            prefetchType: "contacts"
        }, this.attr = util.merge(this.attr, a), this.after = $.noop, this.defaultAttrs = $.noop, compose.mixin(this, [withData, withDatasourceHelpers]), this.getPrefetchCount = function() {
            return this.attr.prefetchCount
        }, this.processTokens = function(a, b, c) {
            a.forEach(function(a) {
                var d = helpers.getFirstChar(a);
                d.trim().length && (this.adjacencyList[d] === undefined && (this.adjacencyList[d] = {}), c in this.adjacencyList[d] ? b > this.adjacencyList[d][c] && (this.adjacencyList[d][c] = b) : this.adjacencyList[d][c] = b)
            }, this)
        }, this.processResults = function(a) {
            if (!a ||!a[this.attr.prefetchType])
                return;
            a[this.attr.prefetchType].forEach(function(a) {
                a.emails.forEach(function(b) {
                    var c = helpers.tokenizeText(a.name);
                    this.processTokens(c, 2, b);
                    var d = b.match(/.*?(?=@)/), e = helpers.tokenizeText(d ? d[0] : "");
                    this.processTokens(e, 1, b);
                    var f = helpers.tokenizeText(b);
                    this.processTokens(f, 0, b);
                    var g = {
                        name: a.name,
                        email: b
                    };
                    g.tokens = c.concat(f), this.contactHash[b] = g
                }, this)
            }, this)
        }, this.getLocalSuggestions = function(a, b) {
            var c = helpers.tokenizeText(a.query), d = this.getPotentiallyMatchingIds(c), e = this.getContactsFromIds(d), f = e.filter(this.matcher(c));
            f = f.slice(0, this.attr.limit), b(f)
        }, this.getPotentiallyMatchingIds = function(a) {
            var b = [[], [], []];
            a.forEach(function(a) {
                var c = this.adjacencyList[helpers.getFirstChar(a)];
                for (token in c) {
                    var d = c[token];
                    b[d] = b[d].concat(token)
                }
            }, this);
            var c = b[2].concat(b[1]).concat(b[0]);
            return c = util.uniqueArray(c), c
        }, this.getContactsFromIds = function(a) {
            var b = [];
            return a.forEach(function(a) {
                var c = this.contactHash[a];
                c && b.push(c)
            }, this), b
        }, this.matcher = function(a) {
            return function(b) {
                var c = b.tokens, d = [], e = a.every(function(a) {
                    var b = c.filter(function(b) {
                        return b.indexOf(a) === 0
                    });
                    return b.length
                });
                if (e)
                    return b
            }.bind(this)
        }, this.requiresRemoteSuggestions = function(a) {
            return !1
        }, this.prefetchContactsData = function() {
            this.prefetch(this.attr.prefetchPath, this.attr.prefetchType, this.processResults.bind(this), this.useStaleData.bind(this))
        }, this.initialize = function() {
            this.adjacencyList = {}, this.contactHash = {}
        }, this.initialize()
    }
    var helpers = require("app/utils/typeahead_helpers"), customStorage = require("app/utils/storage/custom"), withData = require("app/data/with_data"), compose = require("core/compose"), util = require("core/utils"), withDatasourceHelpers = require("app/data/with_datasource_helpers");
    module.exports = contactsDatasource
});
define("app/data/typeahead/saved_searches_datasource", ["module", "require", "exports", "core/utils"], function(module, require, exports) {
    function savedSearchesDatasource(a) {
        this.attr = {
            id: null,
            items: [],
            limit: 0,
            searchPathWithQuery: "/search?src=savs&q=",
            querySource: "saved_search_click"
        }, this.attr = util.merge(this.attr, a), this.getRemoteSuggestions = function(a, b, c) {
            return c
        }, this.requiresRemoteSuggestions = function(a) {
            return !1
        }, this.getLocalSuggestions = function(a, b) {
            if (!a ||!a.query)
                b(this.attr.items);
            else {
                var c = this.attr.items.filter(function(b) {
                    return b.name.indexOf(a.query) == 0
                }).slice(0, this.attr.limit);
                b(c)
            }
        }, this.addSavedSearch = function(a) {
            if (!a ||!a.query)
                return;
            this.attr.items.push({
                id: a.id,
                id_str: a.id_str,
                name: a.name,
                query: a.query,
                saved_search_path: this.attr.searchPathWithQuery + encodeURIComponent(a.query),
                search_query_source: this.attr.querySource
            })
        }, this.removeSavedSearch = function(a) {
            if (!a ||!a.query)
                return;
            var b = this.attr.items;
            for (var c = 0; c < b.length; c++)
                if (b[c].query === a.query || b[c].name === a.query) {
                    b.splice(c, 1);
                    return 
                }
        }
    }
    var util = require("core/utils");
    module.exports = savedSearchesDatasource
});
define("app/data/typeahead/recent_searches_datasource", ["module", "require", "exports", "core/utils", "app/utils/storage/custom", "app/data/with_datasource_helpers"], function(module, require, exports) {
    function recentSearchesDatasource(a) {
        this.attr = {
            id: null,
            limit: 4,
            storageList: "recentSearchesList",
            maxLength: 100,
            ttl_ms: 12096e5,
            searchPathWithQuery: "/search?src=typd&q=",
            querySource: "typed_query"
        }, this.attr = util.merge(this.attr, a), this.getRemoteSuggestions = function(a, b, c) {
            return c
        }, this.requiresRemoteSuggestions = function() {
            return !1
        }, this.removeAllRecentSearches = function() {
            this.items = [], this.updateStorage()
        }, this.getLocalSuggestions = function(a, b) {
            if (!a || a.query === "")
                b(this.items.slice(0, this.attr.limit));
            else {
                var c = this.items.filter(function(b) {
                    return b.name.indexOf(a.query) == 0
                }).slice(0, this.attr.limit);
                b(c)
            }
        }, this.updateStorage = function() {
            this.storage.setItem(this.attr.storageList, this.items, this.attr.ttl_ms)
        }, this.removeOneRecentSearch = function(a) {
            this.removeRecentSearchFromList(a.query), this.updateStorage()
        }, this.addRecentSearch = function(a) {
            if (!a ||!a.query)
                return;
            a.query = a.query.trim(), this.updateRecentSearchList(a), this.updateStorage()
        }, this.updateRecentSearchList = function(a) {
            var b = this.items, c = {
                normalized_name: a.query.toLowerCase(),
                name: a.query,
                recent_search_path: this.attr.searchPathWithQuery + encodeURIComponent(a.query),
                search_query_source: this.attr.querySource
            };
            this.removeRecentSearchFromList(a.query), b.unshift(c), b.length > this.attr.maxLength && b.pop()
        }, this.removeRecentSearchFromList = function(a) {
            var b = this.items, c =- 1, d = a.toLowerCase();
            for (var e = 0; e < b.length; e++)
                if (b[e].normalized_name === d) {
                    c = e;
                    break
                }
            c!==-1 && b.splice(c, 1)
        }, this.initialize = function() {
            var a = customStorage({
                withExpiry: !0
            });
            this.storage = new a("typeahead"), this.items = this.storage.getItem(this.attr.storageList) || []
        }, this.initialize()
    }
    var util = require("core/utils"), customStorage = require("app/utils/storage/custom"), withDatasourceHelpers = require("app/data/with_datasource_helpers");
    module.exports = recentSearchesDatasource
});
define("app/data/typeahead/with_external_event_listeners", ["module", "require", "exports", "app/utils/typeahead_helpers"], function(module, require, exports) {
    function WithExternalEventListeners() {
        this.defaultAttrs({
            weights: {
                CACHED_PROFILE_VISIT: 10,
                UNCACHED_PROFILE_VISIT: 75,
                FOLLOW: 100
            }
        }), this.cleanupUserData = function(a) {
            this.shouldUseIndexedDB() ? this.removeAccountAsync(parseInt(a, 10)) : this.removeAccount(a), this.addToUserBlacklist(a)
        }, this.onFollowStateChange = function(a, b) {
            if (!b.userId)
                return;
            switch (b.newState) {
            case"blocked":
                this.cleanupUserData(b.userId);
                break;
            case"not-following":
                this.cleanupUserData(b.userId);
                break;
            case"following":
                b.user && (this.adjustScoreBoost(b.user, this.attr.weights.FOLLOW), this.shouldUseIndexedDB() ? this.addAccountForFollowAsync(b.user) : this.addAccount(b.user, "following"), this.removeUserFromBlacklist(b.userId))
            }
            this.shouldUseIndexedDB() || this.updateLocalStorage()
        }, this.handleAccountVisit = function(a, b) {
            return b ? this.adjustScoreBoost(b, this.attr.weights.CACHED_PROFILE_VISIT) : this.adjustScoreBoost(a, this.attr.weights.UNCACHED_PROFILE_VISIT), b || a
        }, this.handleProfileVisitAsync = function(a) {
            var b = {
                id: a.id,
                onSuccess: function(b) {
                    var c = this.handleAccountVisit(a, b);
                    this.datasources.accounts.addAccount(c)
                }.bind(this),
                onError: function() {}
            };
            this.datasources.accounts.getAccount(b)
        }, this.onProfileVisit = function(a, b) {
            if (this.shouldUseIndexedDB())
                this.handleProfileVisitAsync(b);
            else {
                var c = this.datasources.accounts.userHash[b.id];
                c ? this.adjustScoreBoost(c, this.attr.weights.CACHED_PROFILE_VISIT) : (this.adjustScoreBoost(b, this.attr.weights.UNCACHED_PROFILE_VISIT), this.addAccount(b, "visit")), this.updateLocalStorage()
            }
        }, this.updateLocalStorage = function() {
            this.datasources.accounts.storage.setItem("userHash", this.datasources.accounts.userHash, this.datasources.accounts.attr.ttl), this.datasources.accounts.storage.setItem("adjacencyList", this.datasources.accounts.adjacencyList, this.datasources.accounts.attr.ttl), this.datasources.accounts.storage.setItem("version", this.datasources.accounts.attr.version, this.datasources.accounts.attr.ttl)
        }, this.removeAccount = function(a) {
            if (!this.datasources.accounts.userHash[a])
                return;
            var b = this.datasources.accounts.userHash[a].tokens;
            b.forEach(function(b) {
                var c = this.datasources.accounts.adjacencyList[b.charAt(0)];
                if (!c)
                    return;
                var d = c.indexOf(a);
                if (d===-1)
                    return;
                c.splice(d, 1)
            }, this), delete this.datasources.accounts.userHash[a]
        }, this.removeAccountAsync = function(a) {
            this.datasources.accounts.removeAccount(a)
        }, this.adjustScoreBoost = function(a, b) {
            a.score_boost ? a.score_boost += b : a.score_boost = b
        }, this.addAccount = function(a, b) {
            this.datasources.accounts.userHash[a.id] = a, a.tokens = ["@" + a.screen_name, a.screen_name].concat(helpers.tokenizeText(a.name)), a.social_proof = b === "following" ? 1 : 0, a.tokens.forEach(function(b) {
                var c = b.charAt(0);
                if (!b.trim().length)
                    return;
                if (!this.datasources.accounts.adjacencyList[c]) {
                    this.datasources.accounts.adjacencyList[c] = [a.id];
                    return 
                }
                this.datasources.accounts.adjacencyList[c].indexOf(a.id)===-1 && this.datasources.accounts.adjacencyList[c].push(a.id)
            }, this)
        }, this.addAccountForFollowAsync = function(a) {
            a.tokens = ["@" + a.screen_name, a.screen_name].concat(helpers.tokenizeText(a.name)), a.social_proof = 1, this.datasources.accounts.addAccount(a)
        }, this.removeOldAccountsInBlackList = function() {
            var a, b, c = 0, d = this.datasources.accounts.attr.maxLengthBlacklist || 100, e = this.datasources.accounts.socialContextBlackList, f = (new Date).getTime(), g = this.datasources.accounts.attr.ttl_ms;
            for (b in e) {
                var h = e[b] + g;
                h < f ? delete e[b] : (a = a || b, a = e[a] > e[b] ? b : a, c += 1)
            }
            d < c && delete e[a]
        }, this.updateBlacklistLocalStorage = function(a) {
            this.datasources.accounts.storage.setItem("userBlackList", a, this.attr.ttl)
        }, this.addToUserBlacklist = function(a) {
            var b = this.datasources.accounts.socialContextBlackList;
            b[a] = (new Date).getTime(), this.removeOldAccountsInBlackList(), this.updateBlacklistLocalStorage(b)
        }, this.removeUserFromBlacklist = function(a) {
            var b = this.datasources.accounts.socialContextBlackList;
            this.removeOldAccountsInBlackList(), b[a] && (delete b[a], this.updateBlacklistLocalStorage(b))
        }, this.checkItemTypeForRecentSearch = function(a) {
            return a === "saved_search" || a === "topics" || a === "recent_search"?!0 : !1
        }, this.addSavedSearch = function(a, b) {
            this.datasources.savedSearches.addSavedSearch(b)
        }, this.removeSavedSearch = function(a, b) {
            this.datasources.savedSearches.removeSavedSearch(b)
        }, this.addRecentSearch = function(a, b) {
            b.source === "search" ? this.datasources.recentSearches.addRecentSearch({
                query: b.query
            }) : this.checkItemTypeForRecentSearch(b.source) && b.isSearchInput && this.datasources.recentSearches.addRecentSearch({
                query: b.query
            })
        }, this.removeRecentSearch = function(a, b) {
            b.deleteAll ? this.datasources.recentSearches.removeAllRecentSearches() : this.datasources.recentSearches.removeOneRecentSearch(b), $(a.target).trigger("dataTypeaheadRecentSearchDeleted")
        }, this.updateSelectedUsers = function(a, b) {
            this.datasources.selectedUsers.updateSelectedUsers(b)
        }, this.setTrendLocations = function(a, b) {
            this.datasources.trendLocations.setTrendLocations(b)
        }, this.updatePrefill = function(a, b) {
            this.datasources.prefillUsers.updateRecentlySelectedUsers(b)
        }, this.setupEventListeners = function(a) {
            switch (a) {
            case"accounts":
                this.on("dataFollowStateChange", this.onFollowStateChange), this.on("profileVisit", this.onProfileVisit);
                break;
            case"savedSearches":
                this.on(document, "dataAddedSavedSearch", this.addSavedSearch), this.on(document, "dataRemovedSavedSearch", this.removeSavedSearch);
                break;
            case"recentSearches":
                this.on("uiSearchQuery uiTypeaheadItemSelected", this.addRecentSearch), this.on("uiTypeaheadDeleteRecentSearch", this.removeRecentSearch);
                break;
            case"trendLocations":
                this.on(document, "dataLoadedTrendLocations", this.setTrendLocations);
                break;
            case"selectedUsers":
                this.on("uiUpdatedSelectedUsers", this.updateSelectedUsers);
                break;
            case"prefillUsers":
                this.on("uiUpdateRecentTags", this.updatePrefill)
            }
        }
    }
    var helpers = require("app/utils/typeahead_helpers");
    module.exports = WithExternalEventListeners
});
define("app/data/typeahead/topics_datasource", ["module", "require", "exports", "app/utils/typeahead_helpers", "app/utils/storage/custom", "app/data/with_data", "core/compose", "core/utils", "app/data/with_datasource_helpers", "core/i18n"], function(module, require, exports) {
    function topicsDatasource(a) {
        this.attr = {
            id: null,
            ttl_ms: 216e5,
            limit: 4,
            version: 4,
            storageAdjacencyList: "topicsAdjacencyList",
            storageHash: "topicsHash",
            storageVersion: "topicsVersion",
            prefetchLimit: 500,
            localQueriesEnabled: !1,
            remoteQueriesEnabled: !1,
            remotePath: "/i/search/typeahead.json",
            remoteType: "topics",
            prefetchPath: "/i/search/typeahead.json",
            prefetchType: "topics",
            basePath: "/search?",
            modePathMap: {
                images: "mode=photos",
                videos: "mode=videos",
                news: "mode=news",
                users: "mode=users"
            },
            querySourcePathTypeahead: "src=tyah"
        }, this.attr = util.merge(this.attr, a), this.after = $.noop, this.defaultAttrs = $.noop, compose.mixin(this, [withData, withDatasourceHelpers]), this.getLocalSuggestions = function(a, b) {
            b([])
        }, this.getTopicObjectsFromStrings = function(a) {
            var b = [];
            return a.forEach(function(a) {
                var c = this.topicsHash[a];
                c && b.push(c)
            }, this), b
        }, this.requiresRemoteSuggestions = function(a) {
            var b = helpers.getFirstChar(a.query), c = a.topicsMustStartWithHashtag && HASHTAG_CHARS.indexOf(b)===-1;
            return !a ||!a.query || c?!1 : this.attr.remoteQueriesEnabled
        }, this.processRemoteSuggestions = function(a, b, c) {
            var d = c[this.attr.id] || [], e = {};
            return d.forEach(function(a) {
                var b = a.topic || a.hashtag;
                e[b.toLowerCase()]=!0
            }, this), b[this.attr.remoteType] && b[this.attr.remoteType].forEach(function(a) {
                var b = a.topic || a.hashtag, c = a.filter;
                c && (a.name = this.constructName(b, c)), a.searchPath = this.constructSearchPath(b, c), e[b.toLowerCase()] || d.push(a)
            }, this), c[this.attr.id] = d.slice(0, this.attr.limit), c
        }, this.constructName = function(a, b) {
            var c = {
                images: _('Photos of <strong>{{topic}}</strong>', {
                    topic: a
                }),
                videos: _('Videos of <strong>{{topic}}</strong>', {
                    topic: a
                }),
                news: _('News about <strong>{{topic}}</strong>', {
                    topic: a
                }),
                users: _('People related to <strong>{{topic}}</strong>', {
                    topic: a
                })
            };
            return c[b]
        }, this.constructSearchPath = function(a, b) {
            var c = ["q=" + encodeURIComponent(a)], d = b ? this.attr.modePathMap[b]: "";
            return d && c.push(d), c.push(this.attr.querySourcePathTypeahead), this.attr.basePath + c.join("&")
        }, this.initialize = function() {}, this.initialize()
    }
    var helpers = require("app/utils/typeahead_helpers"), customStorage = require("app/utils/storage/custom"), withData = require("app/data/with_data"), compose = require("core/compose"), util = require("core/utils"), withDatasourceHelpers = require("app/data/with_datasource_helpers"), _ = require("core/i18n"), HASHTAG_CHARS = ["$", "#", "＃"];
    module.exports = topicsDatasource
});
define("app/data/typeahead/trend_locations_datasource", ["module", "require", "exports", "core/utils", "core/i18n"], function(module, require, exports) {
    function trendLocationsDatasource(a) {
        var b = {
            woeid: - 1,
            placeTypeCode: - 1,
            name: _('No results were found. Try selecting from a list.')
        };
        this.attr = {
            id: null,
            items: [],
            limit: 10
        }, this.attr = util.merge(this.attr, a), this.getRemoteSuggestions = function(a, b, c) {
            return c
        }, this.requiresRemoteSuggestions = function(a) {
            return !1
        }, this.getLocalSuggestions = function(a, c) {
            var d = this.attr.items, e = function(a) {
                return a.replace(/\s+/g, "").toLowerCase()
            };
            if (a && a.query) {
                var f = e(a.query);
                d = this.attr.items.filter(function(a) {
                    return e(a.name).indexOf(f) == 0
                })
            }
            d.length ? c(d.slice(0, this.attr.limit)) : c([b])
        }, this.setTrendLocations = function(a) {
            this.attr.items = a.trendLocations
        }
    }
    var util = require("core/utils"), _ = require("core/i18n");
    module.exports = trendLocationsDatasource
});
define("app/data/typeahead/concierge_datasource", ["module", "require", "exports", "core/compose", "core/utils", "core/i18n", "app/utils/storage/custom", "app/data/with_data", "app/data/with_datasource_helpers"], function(module, require, exports) {
    function conciergeDatasource(a) {
        this.attr = {
            id: null,
            ttl_ms: 216e5,
            limit: 3,
            version: 1,
            storageList: "conciergeList",
            storageVersion: "conciergeVersion",
            prefetchLimit: 3,
            localQueriesEnabled: !1,
            remoteQueriesEnabled: !1,
            prefetchPath: "/i/search/typeahead.json",
            prefetchType: "oneclick",
            remotePath: "/i/search/typeahead.json",
            remoteType: "oneclick",
            basePath: "/search?",
            scribeMap: {
                images: "photos",
                videos: "video",
                news: "news",
                users: "people"
            },
            nameMap: {
                "images near": _('Photos near you'),
                "images follow": _('Photos from people you follow'),
                "videos near": _('Videos near you'),
                "videos follow": _('Videos from people you follow'),
                "news near": _('News near you'),
                "news follow": _('News shared by people you follow'),
                "users near": _('People near you')
            },
            modePathMap: {
                images: "mode=photos",
                videos: "mode=videos",
                news: "mode=news",
                users: "mode=users"
            },
            locationFilterPath: "near=me",
            followFilterPath: "f=follows",
            querySourcePath: "src=taoc",
            querySource: "typeahead_oneclick",
            typeaheadExperiments: []
        }, this.attr = util.merge(this.attr, a), this.after = $.noop, this.defaultAttrs = $.noop, compose.mixin(this, [withData, withDatasourceHelpers]), this.getStorageKeys = function() {
            return [this.attr.storageVersion, this.attr.storageList]
        }, this.getPrefetchCount = function() {
            return this.attr.prefetchLimit
        }, this.isMetadataExpired = function() {
            var a = this.storage.getItem(this.attr.storageVersion);
            return a !== this.attr.version
        }, this.getDataFromLocalStorage = function() {
            this.conciergeList = this.storage.getItem(this.attr.storageList) || this.conciergeList
        }, this.processResults = function(a) {
            if (!a ||!a[this.attr.prefetchType]) {
                this.useStaleData();
                return 
            }
            a[this.attr.prefetchType].forEach(function(a) {
                var b = this.constructConciergeSuggestion(a);
                b && this.conciergeList.push(b)
            }, this), this.storage.setItem(this.attr.storageList, this.conciergeList, this.attr.ttl_ms), this.storage.setItem(this.attr.storageVersion, this.attr.version, this.attr.ttl_ms)
        }, this.getLocalSuggestions = function(a, b) {
            !a || a.query === "" ? b(this.conciergeList.slice(0, this.attr.limit)) : b([])
        }, this.requiresRemoteSuggestions = function(a) {
            return a && a.query === "" ? this.attr.remoteQueriesEnabled : !1
        }, this.processRemoteSuggestions = function(a, b, c) {
            var d = c[this.attr.id] || [];
            if (b[this.attr.remoteType]) {
                var e = b[this.attr.remoteType];
                for (var f = 0; f < e.length; f++) {
                    var g = this.constructConciergeSuggestion(e[f]);
                    if (g) {
                        var h = d.push(g);
                        if (h >= this.attr.limit)
                            break
                    }
                }
            }
            return c[this.attr.id] = d.slice(0, this.attr.limit), c
        }, this.constructConciergeSuggestion = function(a) {
            var b = a.filter, c=!!a.location, d=!!a.follow, e = a.topic, f = this.constructName(b, c, d, e);
            return f ? {
                topic: e,
                searchPath: this.constructSearchPath(b, c, d, e),
                name: f,
                querySource: this.attr.querySource,
                searchDetails: this.constructScribeDetails(b, c, d, e)
            } : null
        }, this.constructName = function(a, b, c, d) {
            var e = b ? " near": "", f = c ? " follow": "", g = d || this.attr.nameMap[a + e + f];
            return g
        }, this.constructSearchPath = function(a, b, c, d) {
            var e = [this.attr.querySourcePath], f = a ? this.attr.modePathMap[a]: "";
            return f && e.push(f), b && e.push(this.attr.locationFilterPath), c && e.push(this.attr.followFilterPath), d && e.push("q=" + encodeURIComponent(d)), this.attr.basePath + e.join("&")
        }, this.constructScribeDetails = function(a, b, c, d) {
            var e = {
                result_type: this.attr.scribeMap[a] || "everything",
                social_filter: c ? "following": "all"
            };
            return d && (e.query = d), b && (e.near = "me"), e
        }, this.initialize = function() {
            var a = customStorage({
                withExpiry: !0
            });
            this.storage = new a("typeahead"), this.conciergeList = [], this.attr.localQueriesEnabled && this.loadData(this.attr.prefetchPath, this.attr.prefetchType)
        }, this.initialize()
    }
    var compose = require("core/compose"), util = require("core/utils"), _ = require("core/i18n"), customStorage = require("app/utils/storage/custom"), withData = require("app/data/with_data"), withDatasourceHelpers = require("app/data/with_datasource_helpers");
    module.exports = conciergeDatasource
});
define("app/data/typeahead/selected_users_datasource", ["module", "require", "exports", "core/utils"], function(module, require, exports) {
    function selectedUsersDatasource(a) {
        this.attr = {
            id: null,
            maxLength: 100,
            querySource: "typed_query",
            items: []
        }, this.attr = util.merge(this.attr, a), this.getRemoteSuggestions = function(a, b, c) {
            return c
        }, this.requiresRemoteSuggestions = function() {
            return !1
        }, this.removeAllSelectedUsers = function() {
            this.attr.items = []
        }, this.getLocalSuggestions = function(a, b) {
            b(this.attr.items.slice(0, this.attr.maxLength))
        }, this.updateSelectedUsers = function(a) {
            this.attr.items = a.items
        }
    }
    var util = require("core/utils");
    module.exports = selectedUsersDatasource
});
define("app/data/typeahead/prefill_users_datasource", ["module", "require", "exports", "core/utils", "app/utils/storage/custom", "app/data/with_datasource_helpers"], function(module, require, exports) {
    function prefillUsersDatasource(a) {
        this.attr = {
            id: null,
            maxLength: 10,
            querySource: "typed_query",
            customItems: [],
            ttl_ms: 12096e5,
            storageList: "recentlySelectedList"
        }, this.attr = util.merge(this.attr, a), this.getRemoteSuggestions = function(a, b, c) {
            return c
        }, this.requiresRemoteSuggestions = function() {
            return !1
        }, this.updateStorage = function() {
            this.storage.setItem(this.attr.storageList, this.storageItems, this.attr.ttl_ms)
        }, this.updateRecentlySelectedUsers = function(a) {
            var b = this.storageItems;
            a.items.forEach(function(a) {
                this.removeRecentUserFromList(a), b.unshift(a), b.length > this.attr.maxLength && (this.storageItems = b.slice(0, this.attr.maxLength))
            }, this), this.updateStorage()
        }, this.removeRecentUserFromList = function(a) {
            var b = this.storageItems, c = this.getSuggestionIndexById(a, b);
            c!=-1 && b.splice(c, 1)
        }, this.getSuggestionIndexById = function(a, b) {
            for (var c = 0; c < b.length; c++)
                if (b[c].id == a.id)
                    return c;
            return - 1
        }, this.getLocalSuggestions = function(a, b) {
            var c = this.storageItems.slice(0, this.attr.maxLength);
            for (var d = 0; c.length < this.attr.maxLength && d < this.attr.customItems.length; d++) {
                var e = this.attr.customItems[d];
                this.getSuggestionIndexById(c, e)==-1 && c.push(e)
            }
            b(c)
        }, this.updateCustomItems = function(a) {
            this.attr.customItems = a.items
        }, this.initialize = function() {
            var a = customStorage({
                withExpiry: !0
            });
            this.storage = new a("typeahead"), this.storageItems = this.storage.getItem(this.attr.storageList) || []
        }, this.initialize()
    }
    var util = require("core/utils"), customStorage = require("app/utils/storage/custom"), withDatasourceHelpers = require("app/data/with_datasource_helpers");
    module.exports = prefillUsersDatasource
});
define("app/utils/storage/indexed_db", ["module", "require", "exports"], function(module, require, exports) {
    function IndexedDBClient(a) {
        this.database = a, this.readOnly = "readonly", this.readWrite = "readwrite", this.objectStoreDNEMessage = "Objectstore does not exist", this.database.onerror = function(a) {}
    }
    function getDBClient(a) {
        if (typeof window.indexedDB == "undefined") {
            a.onError();
            return 
        }
        var b = indexedDB.open(a.dbName);
        b.onerror = function(b) {
            a.onError(b)
        }, b.onsuccess = function(c) {
            var d = b.result, e = new IndexedDBClient(d);
            a.onSuccess(e)
        }
    }
    IndexedDBClient.prototype.closeConnection = function() {
        this.database.close()
    }, IndexedDBClient.prototype.getObjectStore = function(a, b) {
        var c = this.database.transaction([a.objectStoreName], b), d = c.objectStore(a.objectStoreName);
        return d
    }, IndexedDBClient.prototype.getObjectStoreNames = function() {
        var a = this.database.objectStoreNames, b = [];
        for (var c = 0; c < a.length; c++)
            b.push(a[c]);
        return b
    }, IndexedDBClient.prototype.getItemByKey = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.getObjectStore(a, this.readOnly), c = b.get(a.key);
        c.onsuccess = function(b) {
            a.onSuccess(c.result)
        }, c.onerror = function(b) {
            a.onError(b)
        }
    }, IndexedDBClient.prototype.getItemsByPrefix = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.getObjectStore(a, this.readOnly), c = IDBKeyRange.bound(a.prefix, a.prefix + "￿", !1, !1), d = [], e = b.openCursor(c);
        e.onsuccess = function(b) {
            var c = b.target.result;
            c ? (d.push(c.value), c["continue"]()) : a.onSuccess(d)
        }, e.onerror = function(b) {
            a.onError(b)
        }
    }, IndexedDBClient.prototype.addItems = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.database.transaction([a.objectStoreName], this.readWrite);
        b.oncomplete = function(b) {
            a.onSuccess()
        }, b.onerror = function(b) {
            a.onError(b)
        };
        var c = b.objectStore(a.objectStoreName);
        a.items.forEach(function(a) {
            c.add(a)
        })
    }, IndexedDBClient.prototype.upsertItems = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.database.transaction([a.objectStoreName], this.readWrite);
        b.oncomplete = function(b) {
            a.onSuccess()
        }, b.onerror = function(b) {
            a.onError(b)
        };
        var c = b.objectStore(a.objectStoreName);
        a.items.forEach(function(a) {
            c.put(a)
        })
    }, IndexedDBClient.prototype.createObjectStore = function(a) {
        function i() {
            c && d && a.onSuccess(b)
        }
        if (this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b, c=!1, d=!1, e = this.database.version, f = this.database.name;
        this.closeConnection();
        var g = indexedDB.open(f, ++e), h = this;
        g.onupgradeneeded = function(d) {
            h.database = d.target.result, b = h.database.createObjectStore(a.objectStoreName, {
                keyPath: a.keyPathName
            }), b.transaction.oncomplete = function(a) {
                c=!0, i()
            }, b.transaction.onerror = function(b) {
                a.onError(b)
            }
        }, g.onsuccess = function(a) {
            d=!0, i()
        }, g.onerror = function(b) {
            a.onError(b)
        }, g.onblocked = function(b) {
            typeof a.onBlocked == "function" && a.onBlocked()
        }
    }, IndexedDBClient.prototype.deleteObjectStore = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.database.version, c = this.database.name;
        this.closeConnection();
        var d = indexedDB.open(c, ++b), e = this;
        d.onupgradeneeded = function(b) {
            e.database = b.target.result, e.database.deleteObjectStore(a.objectStoreName)
        }, d.onerror = function(b) {
            a.onError(b)
        }, d.onsuccess = function(b) {
            a.onSuccess()
        }, d.onblocked = function(b) {
            typeof a.onBlocked == "function" && a.onBlocked()
        }
    }, IndexedDBClient.prototype.deleteItem = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.database.transaction([a.objectStoreName], this.readWrite), c = b.objectStore(a.objectStoreName);
        c["delete"](a.key), b.oncomplete = function() {
            a.onSuccess()
        }, b.onerror = function(b) {
            a.onError(b)
        }
    }, IndexedDBClient.prototype.clearObjectStore = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.database.transaction([a.objectStoreName], this.readWrite), c = b.objectStore(a.objectStoreName);
        c.clear(), b.oncomplete = function() {
            a.onSuccess()
        }, b.onerror = function(b) {
            a.onError(b)
        }
    }, IndexedDBClient.prototype.clearAllObjectStores = function(a) {
        var b = this.getObjectStoreNames(), c = 0, d = 0;
        b.forEach(function(e) {
            function h() {
                c++, c === b.length && (d > 0 ? a.onError() : a.onSuccess())
            }
            function i() {
                c++, d++, c === b.length && a.onError()
            }
            var f = this.database.transaction([e], this.readWrite), g = f.objectStore(e);
            g.clear(), f.oncomplete = function() {
                h()
            }, f.onerror = function(a) {
                i()
            }, f.onabort = function(a) {
                i()
            }
        }, this)
    }, IndexedDBClient.prototype.updateItem = function(a) {
        if (!this.checkIfObjectStoreExists(a.objectStoreName)) {
            a.onError(this.objectStoreDNEMessage);
            return 
        }
        var b = this.database.transaction([a.objectStoreName], this.readWrite), c = b.objectStore(a.objectStoreName);
        c.put(a.item), b.oncomplete = function() {
            a.onSuccess()
        }, b.onerror = function(b) {
            a.onError(b)
        }
    }, IndexedDBClient.prototype.checkIfObjectStoreExists = function(a) {
        return this.database.objectStoreNames.contains(a)
    }, IndexedDBClient.prototype.deleteDatabase = function(a) {
        var b = indexedDB.deleteDatabase(a.dbName);
        b.onsuccess = function(b) {
            a.onSuccess()
        }, b.onerror = function(b) {
            a.onError(b)
        }
    }, module.exports = getDBClient
});
define("app/data/typeahead/typeahead", ["module", "require", "exports", "core/component", "core/utils", "app/data/with_data", "app/data/typeahead/with_cache", "app/data/typeahead/accounts_datasource", "app/data/typeahead/accounts_datasource_async", "app/data/typeahead/contacts_datasource", "app/data/typeahead/saved_searches_datasource", "app/data/typeahead/recent_searches_datasource", "app/data/typeahead/with_external_event_listeners", "app/data/typeahead/topics_datasource", "app/data/typeahead/trend_locations_datasource", "app/data/typeahead/concierge_datasource", "app/data/typeahead/selected_users_datasource", "app/data/typeahead/prefill_users_datasource", "app/utils/storage/indexed_db", "app/data/with_scribe"], function(module, require, exports) {
    function typeahead() {
        this.defaultAttrs({
            limit: 10,
            remoteDebounceInterval: 300,
            remoteThrottleInterval: 300,
            outstandingRemoteRequestCount: 0,
            queuedRequestData: !1,
            outstandingRemoteRequestMax: 6,
            useThrottle: !1,
            tweetContextEnabled: !1,
            topicsWithFiltersEnabled: !1,
            typeaheadExperiments: []
        }), this.triggerSuggestionsEvent = function(a, b, c) {
            this.trigger("dataTypeaheadSuggestionsResults", {
                suggestions: c,
                queryData: b,
                id: a
            })
        }, this.hasAnySuggestions = function(a, b) {
            return b.some(function(b) {
                return a[b] && a[b].length
            })
        }, this.needsRemoteRequest = function(a, b) {
            return b.some(function(b) {
                return this.datasources[b] && this.datasources[b].requiresRemoteSuggestions && this.datasources[b].requiresRemoteSuggestions(a)
            }, this)
        }, this.getLocalSuggestions = function(a, b, c) {
            function f() {
                e++, e == b.length && c(d)
            }
            var d = {}, e = 0;
            b.forEach(function(b) {
                this.datasources[b] ? this.datasources[b].getLocalSuggestions(a, function(a) {
                    a.length && (a.forEach(function(a) {
                        a.origin = "prefetched"
                    }), d[b] = a), f()
                }) : f()
            }, this)
        }, this.processRemoteSuggestions = function(a) {
            this.adjustRequestCount( - 1);
            var b = a.sourceEventData, c = b.suggestions || {};
            b.datasources.forEach(function(d) {
                var e = d.processRemoteSuggestions(b, a, c);
                if (e[d.attr.id] && e[d.attr.id].length) {
                    for (var f in e)
                        e[f].forEach(function(a) {
                            a.origin = a.origin || "remote"
                        });
                    b.suggestions[d.attr.id] = e[d.attr.id]
                }
            }, this), this.processSuggestionsByDataset(c), this.triggerSuggestionsEvent(b.id, b.queryData, c), this.makeQueuedRemoteRequestIfPossible()
        }, this.getRemoteSuggestions = function(a, b, c, d) {
            if (!b ||!this.needsRemoteRequest(b, d))
                return;
            this.request[a] || (this.attr.useThrottle ? this.request[a] = utils.throttle(this.splitRemoteRequests.bind(this), this.attr.remoteThrottleInterval) : this.request[a] = utils.debounce(this.splitRemoteRequests.bind(this), this.attr.remoteDebounceInterval)), b.query.indexOf("@") === 0 && b.typeaheadSrc === "COMPOSE" && (b.query = b.query.substring(1), b.atSignRemoved=!0), this.request[a](a, b, c, d)
        }, this.makeQueuedRemoteRequestIfPossible = function() {
            if (this.attr.outstandingRemoteRequestCount === this.attr.outstandingRemoteRequestMax - 1 && this.attr.queuedRequestData) {
                var a = this.attr.queuedRequestData;
                this.getRemoteSuggestions(a.id, a.queryData, a.suggestions, a.datasources), this.attr.queuedRequestData=!1
            }
        }, this.adjustRequestCount = function(a) {
            this.attr.outstandingRemoteRequestCount += a
        }, this.canMakeRemoteRequest = function(a) {
            return this.attr.outstandingRemoteRequestCount < this.attr.outstandingRemoteRequestMax?!0 : (this.attr.queuedRequestData = a, !1)
        }, this.processRemoteRequestError = function() {
            this.adjustRequestCount( - 1), this.makeQueuedRemoteRequestIfPossible()
        }, this.splitRemoteRequests = function(a, b, c, d) {
            var e = {};
            d.forEach(function(a) {
                if (this[a] && this[a].requiresRemoteSuggestions && this[a].requiresRemoteSuggestions(b)) {
                    var c = this[a];
                    e[c.attr.remotePath] ? e[c.attr.remotePath].push(c) : e[c.attr.remotePath] = [c]
                }
            }, this.datasources);
            for (var f in e) {
                this.executeRemoteRequest(a, b, c, e[f]);
                break
            }
        }, this.executeRemoteRequest = function(a, b, c, d) {
            var e = {
                id: a,
                queryData: b,
                suggestions: c,
                datasources: d
            };
            if (!this.canMakeRemoteRequest(e))
                return;
            if (!this.attr.tweetContextEnabled || b.tweetContext && b.tweetContext.length > 1e3)
                b.tweetContext = undefined;
            this.adjustRequestCount(1), this.get({
                url: d[0].attr.remotePath,
                headers: {
                    "X-Phx": !0
                },
                data: {
                    q: b.query,
                    tweet_context: b.tweetContext,
                    count: this.attr.limit,
                    result_type: d.map(function(a) {
                        return a.attr.remoteType
                    }).join(","),
                    filters: this.attr.topicsWithFiltersEnabled,
                    src: b.typeaheadSrc,
                    experiments: this.attr.typeaheadExperiments.join(",")
                },
                eventData: e,
                success: this.processRemoteSuggestions.bind(this),
                error: this.processRemoteRequestError.bind(this)
            })
        }, this.truncateTopicsWithRecentSearches = function(a) {
            if (!a.topics ||!a.recentSearches)
                return;
            var b = a.recentSearches.length, c = 4 - b;
            a.topics = a.topics.slice(0, c)
        }, this.dedupSuggestions = function(a, b, c) {
            function e() {
                return b.some(function(a) {
                    return a in c
                })
            }
            function f(a) {
                var b = a.topic || a.name;
                return !d[b.toLowerCase()]
            }
            if (!c[a] ||!e())
                return;
            var d = {};
            c[a].forEach(function(a) {
                var b = a.name || a.topic;
                d[b.toLowerCase()]=!0
            }), b.forEach(function(a) {
                a in c && (c[a] = c[a].filter(f))
            })
        }, this.processSuggestionsByDataset = function(a) {
            this.dedupSuggestions("recentSearches", ["topics"], a), this.truncateTopicsWithRecentSearches(a)
        }, this.getSuggestions = function(a, b) {
            if (typeof b == "undefined")
                throw "No parameters specified";
            if (!b.datasources)
                throw "No datasources specified";
            if (typeof b.queryData == "undefined")
                throw "No query data specified";
            if (typeof b.queryData.query == "undefined")
                throw "No query specified";
            if (!b.id)
                throw "No id specified";
            this.getLocalSuggestions(b.queryData, b.datasources, function(a) {
                this.processSuggestionsByDataset(a);
                var c = b.queryData.query !== "@" && this.needsRemoteRequest(b.queryData, b.datasources);
                (this.hasAnySuggestions(a, b.datasources) ||!c) && this.triggerSuggestionsEvent(b.id, b.queryData, a), c && this.getRemoteSuggestions(b.id, b.queryData, a, b.datasources)
            }.bind(this))
        }, this.prefetchContactsData = function() {
            this.datasources.contacts && this.datasources.contacts.prefetchContactsData()
        }, this.getGlobalDatasources = function() {
            return globalDataSources
        }, this.setGlobalDatasources = function(a) {
            globalDataSources = a
        }, this.addSynchronousDatasource = function(a, b, c) {
            var d = c[b] || {};
            if (!this.isDatasourceEnabled(b))
                return;
            globalDataSources.hasOwnProperty(b) ? this.datasources[b] = globalDataSources[b] : globalDataSources[b] = this.datasources[b] = new a(utils.merge(d, {
                id: b
            })), this.setupEventListeners(b)
        }, this.addAsynchronousDatasource = function(a) {
            var b = a.dsName, c = this.componentOptions[b] || {}, d = globalDataSources[b];
            if (d && d.isUsingIndexedDB)
                this.datasources[b] = globalDataSources[b], a.callback();
            else {
                var e = new a.datasource(utils.merge(c, {
                    id: b
                }), a);
                e.initialize(), globalDataSources[b] = this.datasources[b] = e
            }
        }, this.initializeLocalData = function(a) {
            this.addDatasourcesNotUsingIndexedDB(), this.shouldUseIndexedDB() ? this.addAsynchronousDatasources(a) : this.addSynchronousDatasources(a)
        }, this.isDatasourceEnabled = function(a) {
            var b = this.componentOptions[a] || {};
            return !!b.enabled
        }, this.isBrowserSafari = function() {
            return $.browser.safari
        }, this.shouldUseIndexedDB = function() {
            return this.attr.useIndexedDB&&!this.isBrowserSafari()
        }, this.addAccountsDatasource = function(a) {
            var b = "accounts", c = "accounts", d = {
                dbName: c,
                indexedDBWrapper: IndexedDBWrapper,
                dsName: b,
                datasource: accountsDatasourceAsync,
                callback: function(d) {
                    d ? (this.setupEventListeners(b), this.addDBHandleToGlobals(c, d)) : this.indexedDBInitializationFailed=!0, this.addDMAccountsDatasource(a)
                }.bind(this)
            };
            this.isDatasourceEnabled(b) ? this.addAsynchronousDatasource(d) : this.addDMAccountsDatasource(a)
        }, this.addDMAccountsDatasource = function(a) {
            var b = "dmAccounts", c = "accounts", d = {
                dbName: c,
                indexedDBWrapper: IndexedDBWrapper,
                dsName: b,
                datasource: accountsDatasourceAsync,
                callback: function(d) {
                    d ? (this.setupEventListeners(b), this.addDBHandleToGlobals(c, d)) : this.indexedDBInitializationFailed=!0, this.addMediaTagAccountsDatasource(a)
                }.bind(this)
            };
            this.isDatasourceEnabled(b) ? this.addAsynchronousDatasource(d) : this.addMediaTagAccountsDatasource(a)
        }, this.addMediaTagAccountsDatasource = function(a) {
            var b = "mediaTagAccounts", c = "accounts", d = {
                dbName: c,
                indexedDBWrapper: IndexedDBWrapper,
                dsName: b,
                datasource: accountsDatasourceAsync,
                callback: function(d) {
                    d ? (this.setupEventListeners(b), this.addDBHandleToGlobals(c, d)) : this.indexedDBInitializationFailed=!0, a()
                }.bind(this)
            };
            this.isDatasourceEnabled(b) ? this.addAsynchronousDatasource(d) : a()
        }, this.addDBHandleToGlobals = function(a, b) {
            b && (window.globalIndexedDBs[a] = b)
        }, this.addAsynchronousDatasources = function(a) {
            this.addAccountsDatasource(a)
        }, this.addDatasourcesNotUsingIndexedDB = function() {
            this.addSynchronousDatasource(savedSearchesDatasource, "savedSearches", this.componentOptions), this.addSynchronousDatasource(topicsDatasource, "topics", this.componentOptions), this.addSynchronousDatasource(topicsDatasource, "hashtags", utils.merge(this.componentOptions, {
                hashtags: {
                    remoteType: "hashtags"
                }
            }, !0)), this.addSynchronousDatasource(trendLocationsDatasource, "trendLocations", this.componentOptions), this.addSynchronousDatasource(contactsDatasource, "contacts", this.componentOptions), this.addSynchronousDatasource(selectedUsersDatasource, "selectedUsers", this.componentOptions), this.addSynchronousDatasource(prefillUsersDatasource, "prefillUsers", this.componentOptions), this.addSynchronousDatasource(recentSearchesDatasource, "recentSearches", this.componentOptions), this.addSynchronousDatasource(conciergeDatasource, "concierge", this.componentOptions)
        }, this.addSynchronousDatasources = function(a) {
            this.addSynchronousDatasource(accountsDatasource, "accounts", this.componentOptions), this.addSynchronousDatasource(accountsDatasource, "dmAccounts", this.componentOptions), this.addSynchronousDatasource(accountsDatasource, "mediaTagAccounts", this.componentOptions), a()
        }, this.scribeIndexedDB = function() {
            var a = this.indexedDBInitializationFailed ? "failure": "success";
            this.scribe({
                component: "indexeddb",
                action: a
            }, {
                browserData: JSON.stringify($.browser)
            })
        }, this.after("initialize", function(a, b) {
            this.datasources = {}, this.request = {}, this.componentOptions = b, this.initializeLocalData(function() {
                this.on("uiNeedsTypeaheadSuggestions", this.getSuggestions), this.on("uiShareViaEmailDialogOpened", this.prefetchContactsData), this.scribeIndexedDB()
            }.bind(this))
        })
    }
    var defineComponent = require("core/component"), utils = require("core/utils"), withData = require("app/data/with_data"), withCache = require("app/data/typeahead/with_cache"), accountsDatasource = require("app/data/typeahead/accounts_datasource"), accountsDatasourceAsync = require("app/data/typeahead/accounts_datasource_async"), contactsDatasource = require("app/data/typeahead/contacts_datasource"), savedSearchesDatasource = require("app/data/typeahead/saved_searches_datasource"), recentSearchesDatasource = require("app/data/typeahead/recent_searches_datasource"), withExternalEventListeners = require("app/data/typeahead/with_external_event_listeners"), topicsDatasource = require("app/data/typeahead/topics_datasource"), trendLocationsDatasource = require("app/data/typeahead/trend_locations_datasource"), conciergeDatasource = require("app/data/typeahead/concierge_datasource"), selectedUsersDatasource = require("app/data/typeahead/selected_users_datasource"), prefillUsersDatasource = require("app/data/typeahead/prefill_users_datasource"), IndexedDBWrapper = require("app/utils/storage/indexed_db"), withScribe = require("app/data/with_scribe");
    module.exports = defineComponent(typeahead, withData, withCache, withExternalEventListeners, withScribe), window.globalIndexedDBs = {};
    var globalDataSources = {}
});
define("app/data/typeahead/typeahead_scribe", ["module", "require", "exports", "core/component", "app/data/with_scribe", "lib/twitter-text", "app/utils/scribe_item_types"], function(module, require, exports) {
    function typeaheadScribe() {
        this.defaultAttrs({
            tweetboxSelector: ".tweet-box-shadow"
        }), this.storeCompletionData = function(a, b) {
            if (b.scribeContext && b.scribeContext.component == "tweet_box") {
                var c = b.source == "account" ? "@" + b.display: b.display;
                this.completions.push(c)
            }
        }, this.handleTweetCompletion = function(a, b) {
            if (b.scribeContext.component != "tweet_box")
                return;
            var c = $(a.target).find(this.attr.tweetboxSelector).val(), d = twitterText.extractEntitiesWithIndices(c).filter(function(a) {
                return a.screenName || a.cashtag || a.hashtag
            });
            d = d.map(function(a) {
                return c.slice(a.indices[0], a.indices[1])
            });
            var e = d.filter(function(a) {
                return this.completions.indexOf(a) >= 0
            }, this);
            this.completions = [];
            if (d.length == 0)
                return;
            var f = {
                query: b.query,
                message: d.length,
                event_info: e.length,
                format_version: 2
            };
            this.scribe("entities", b, f)
        }, this.scribeSelection = function(a, b) {
            if (b.fromSelectionEvent && a.type == "uiTypeaheadItemComplete")
                return;
            var c = {
                element: "typeahead_" + b.source,
                action: "select"
            }, d;
            a.type == "uiTypeaheadItemComplete" ? d = "autocomplete" : b.isClick ? d = "click" : a.type == "uiTypeaheadItemSelected" && (d = "key_select");
            var e = {
                position: b.index,
                description: d
            };
            b.source == "account" ? (e.item_type = itemTypes.user, e.id = b.query) : b.source == "topics" ? (e.item_type = itemTypes.search, e.item_query = b.query) : b.source == "saved_search" ? (e.item_type = itemTypes.savedSearch, e.item_query = b.query) : b.source == "recent_search" ? (e.item_type = itemTypes.search, e.item_query = b.query) : b.source == "trend_location" ? (e.item_type = itemTypes.trend, e.item_query = b.item.woeid, b.item.woeid==-1 && (c.action = "no_results")) : b.source == "concierge" && (e.item_type = itemTypes.search, e.item_query = b.query || b.item.name);
            var f = {
                message: b.input,
                items: [e],
                format_version: 2,
                event_info: b.item ? b.item.origin: null,
                search_details: b.item ? b.item.searchDetails: null
            };
            this.scribe(c, b, f)
        }, this.after("initialize", function() {
            this.completions = [], this.on("uiTypeaheadItemComplete uiTypeaheadItemSelected", this.storeCompletionData), this.on("uiTypeaheadItemComplete uiTypeaheadItemSelected", this.scribeSelection), this.on("uiTweetboxTweetSuccess uiTweetboxReplySuccess", this.handleTweetCompletion)
        })
    }
    var defineComponent = require("core/component"), withScribe = require("app/data/with_scribe"), twitterText = require("lib/twitter-text"), itemTypes = require("app/utils/scribe_item_types");
    module.exports = defineComponent(typeaheadScribe, withScribe)
});
define("app/ui/with_profile_tweet_stat_count_data", ["module", "require", "exports"], function(module, require, exports) {
    function withProfileTweetStatCountData() {
        this.before("toggleFavorite", function(a, b, c) {
            c.tweetData.tweetStatCount = c.$tweet.find(".ProfileTweet-action--favorite .ProfileTweet-actionCount").attr("data-tweet-stat-count")
        }), this.before("toggleRetweet", function(a, b, c) {
            c.tweetData.tweetStatCount = c.$tweet.find(".ProfileTweet-action--retweet .ProfileTweet-actionCount").attr("data-tweet-stat-count")
        })
    }
    module.exports = withProfileTweetStatCountData
});
define("app/ui/with_tweet_action_animation", ["module", "require", "exports"], function(module, require, exports) {
    function withTweetActionAnimation() {
        this.defaultAttrs({
            tweetActionContainerSelector: ".ProfileTweet-action",
            animationStateClass: "is-animating"
        }), this.animateAction = function(a, b) {
            var c = $(a.target).find(this.attr.tweetActionContainerSelector);
            c.addClass(this.attr.animationStateClass).one("webkitAnimationEnd oanimationend oAnimationEnd msAnimationEnd animationend", function(a) {
                c.removeClass(this.attr.animationStateClass), this.trigger("uiAnimationEnd")
            }.bind(this))
        }, this.after("initialize", function() {
            this.on(document, "uiDidFavoriteTweet", this.animateAction)
        })
    }
    module.exports = withTweetActionAnimation
});
define("app/ui/with_hover_state_removal", ["module", "require", "exports"], function(module, require, exports) {
    function withHoverStateRemoval() {
        this.addCancelHoverStyleClass = function(a, b, c) {
            if (a.type != "click")
                return;
            var d = $(a.target).closest(this.attr.toggleContainerSelector);
            d.addClass("is-hoverStateCancelled"), d.one("mouseleave", function() {
                d.removeClass("is-hoverStateCancelled")
            }.bind(this))
        }, this.after("toggleFavorite", this.addCancelHoverStyleClass), this.after("toggleRetweet", this.addCancelHoverStyleClass)
    }
    module.exports = withHoverStateRemoval
});
define("app/ui/tweet_actions", ["module", "require", "exports", "core/component", "app/utils/find_tweet", "app/utils/tweet_helper", "app/ui/with_tweet_actions_helper", "app/ui/with_interaction_data", "app/ui/with_profile_tweet_stat_count_data", "app/ui/with_tweet_action_animation", "app/ui/with_hover_state_removal"], function(module, require, exports) {
    function tweetActions() {
        this.defaultAttrs({
            favoriteSelector: ".js-toggle-fav a, .js-actionFavorite",
            replySelector: ".js-actionReply",
            retweetSelector: ".js-toggleRt a:not([aria-disabled=true]), .js-actionRetweet:not(:disabled)",
            permalinkSelector: ".tweet .js-permalink",
            tweetItemSelector: ".tweet",
            retweetInlineSelector: ".js-actionInlineRetweet",
            togglableTweetActionsSelector: "a:visible",
            toggleContainerSelector: ".js-toggleState",
            anyLoggedInActionSelector: ".js-toggleRt a:not([aria-disabled=true]), .js-toggle-fav a, .js-actionReply, .js-actionRetweet, .js-actionFavorite, .js-actionCount",
            favoritedTweetClass: "favorited",
            retweetedTweetClass: "retweeted"
        }), this.handleTransition = function(a, b) {
            return function(c, d) {
                var e = d.id || d.sourceEventData.id, f = findTweet(this.$node, this.attr.tweetItemSelector, e), g = f[0], h = document.activeElement, i;
                g && $.contains(g, h) && (i = $(h).closest(this.attr.toggleContainerSelector)), f[a](b), i && i.find(this.attr.togglableTweetActionsSelector).focus()
            }
        }, this.getDataForReply = function(a) {
            return {
                screenNames: tweetHelper.extractMentionsForReply(a, this.attr.screenName),
                replyLinkClick: !0
            }
        }, this.toggleFavorite = function(a, b, c) {
            var d = c.$tweet.hasClass(this.attr.favoritedTweetClass) ? "uiDidUnfavoriteTweet": "uiDidFavoriteTweet";
            this.trigger(c.$tweet, d, c.tweetData)
        }, this.toggleRetweet = function(a, b, c) {
            var d = c.$tweet.hasClass(this.attr.retweetedTweetClass) ? "uiDidUnretweet": "uiOpenRetweetDialog";
            this.trigger(c.$tweet, d, c.tweetData)
        }, this.openSigninOrSignupDialog = function(a, b, c) {
            this.trigger("uiOpenSigninOrSignupDialog")
        }, this.handleLoggedOutFavoriteClick = function(a, b) {
            this.trigger("uiLoggedOutActionAttempt", {
                action: "favorite_attempt"
            })
        }, this.handleLoggedOutRetweetClick = function(a, b) {
            this.trigger("uiLoggedOutActionAttempt", {
                action: "retweet_attempt"
            })
        }, this.handleLoggedOutReplyClick = function(a, b) {
            this.trigger("uiLoggedOutActionAttempt", {
                action: "reply_attempt"
            })
        }, this.after("initialize", function() {
            if (!this.attr.loggedIn) {
                this.on("click", {
                    anyLoggedInActionSelector: this.composeHandler(this.preventDefault, this.openSigninOrSignupDialog),
                    favoriteSelector: this.handleLoggedOutFavoriteClick,
                    retweetSelector: this.handleLoggedOutRetweetClick,
                    replySelector: this.handleLoggedOutReplyClick
                });
                return 
            }
            var a = this.mkTweetDataCollectorForAction(this.interactionDataWithCard), b = this.mkTweetDataCollectorForAction(this.interactionDataWithCard, this.getDataForReply);
            this.on("click", {
                favoriteSelector: this.composeHandler(this.preventDefault, this.getClosestTweet, a, this.toggleFavorite),
                retweetSelector: this.composeHandler(this.preventDefault, this.getClosestTweet, a, this.toggleRetweet),
                replySelector: this.composeHandler(this.preventDefault, this.getClosestTweet, b, this.triggerTweetAction("uiReplyToTweet")),
                permalinkSelector: this.composeHandler(this.getClosestTweet, a, this.triggerTweetAction("uiPermalinkClick"))
            }), this.on(document, "uiDidFavoriteTweet dataFailedToUnfavoriteTweet", this.handleTransition("addClass", this.attr.favoritedTweetClass)), this.on(document, "uiDidUnfavoriteTweet dataFailedToFavoriteTweet", this.handleTransition("removeClass", this.attr.favoritedTweetClass)), this.on(document, "uiDidRetweet dataFailedToUnretweet", this.handleTransition("addClass", this.attr.retweetedTweetClass)), this.on(document, "uiDidUnretweet dataFailedToRetweet", this.handleTransition("removeClass", this.attr.retweetedTweetClass))
        })
    }
    var defineComponent = require("core/component"), findTweet = require("app/utils/find_tweet"), tweetHelper = require("app/utils/tweet_helper"), withTweetActionsHelper = require("app/ui/with_tweet_actions_helper"), withInteractionData = require("app/ui/with_interaction_data"), withTweetStatCountData = require("app/ui/with_profile_tweet_stat_count_data"), withTweetActionAnimation = require("app/ui/with_tweet_action_animation"), withHoverStateRemoval = require("app/ui/with_hover_state_removal"), TweetActions = defineComponent(tweetActions, withTweetActionsHelper, withInteractionData, withTweetStatCountData, withTweetActionAnimation, withHoverStateRemoval);
    module.exports = TweetActions
});
define("app/ui/tweet_state_updater", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function tweetStateUpdater() {
        this.updateRetweetState = function(a, b) {
            var c = this.$node.find('[data-tweet-id="' + b.tweet_id + '"]');
            c.each(function() {
                $(this).attr("data-my-retweet-id", b.retweet_id)
            })
        }, this.updateUnRetweetState = function(a, b) {
            var c = this.$node.find('[data-tweet-id="' + b.tweet_id + '"]');
            c.each(function() {
                $(this).removeAttr("data-my-retweet-id")
            })
        }, this.after("initialize", function() {
            if (!this.attr.loggedIn)
                return;
            this.on(document, "dataDidRetweet", this.updateRetweetState), this.on(document, "dataDidUnretweet", this.updateUnRetweetState)
        })
    }
    var defineComponent = require("core/component"), TweetStateUpdater = defineComponent(tweetStateUpdater);
    module.exports = TweetStateUpdater
});
define("app/ui/dialogs/uz_survey", ["module", "require", "exports", "core/component"], function(module, require, exports) {
    function uzSurvey() {
        this.defaultAttrs({
            backgroundSelector: "#uz_bg",
            containerSelector: "#uz_popup_container",
            backgroundParentSelector: "body",
            blockUserSurveyClass: "uz_block_user_survey"
        }), this.fadeoutSurvey = function(a) {
            $(this.attr.containerSelector).fadeOut(500)
        }, this.openSurvey = function() {
            window._uzactions = window._uzactions || [];
            var a = this.overrideCookie ? this.overrideCookie: "twittersurvey" + this.surveyId;
            _uzactions.push(["_setCookie", a]);
            var b = document.getElementsByTagName("script")[0];
            b.parentNode.insertBefore($.extend(document.createElement("script"), {
                type: "text/javascript",
                async: !0,
                id: "injected_uz_survey_script",
                charset: "utf-8",
                src: "//cdn4.userzoom.com/files/js/" + this.surveyId + ".js?t=uz_til&cuid=2BA733EB49F6DF1188490022196C4538"
            }), b)
        }, this.onBlockAction = function() {
            this.$node.hasClass(this.attr.blockUserSurveyClass) && this.openSurvey()
        }, this.after("initialize", function() {
            this.surveyId = this.$node.attr("data-survey-id"), this.overrideCookie = this.$node.attr("data-override-cookie"), this.$node.attr("data-show") && this.openSurvey(), this.on(document, "uiOpenSurvey", this.openSurvey), this.on(this.attr.backgroundParentSelector, "click", {
                backgroundSelector: this.fadeoutSurvey
            }), this.on(document, "uiBlockAction", this.onBlockAction)
        })
    }
    var defineComponent = require("core/component");
    module.exports = defineComponent(uzSurvey)
});
define("app/data/with_scribe_data_from_dom", ["module", "require", "exports"], function(module, require, exports) {
    function withScribeDataFromDom() {
        this.defaultAttrs({
            itemType: "",
            eventData: {
                scribeContext: {
                    component: ""
                }
            }
        }), this.setItemType = function(a) {
            if (this.attr.itemType)
                return;
            var b = a.closest("[data-item-type]"), c = b.closest("[data-component-term]");
            this.attr.itemType = b.data("item-type") || this.attr.itemType, this.attr.eventData.scribeContext.component = c.data("component-term") || this.attr.eventData.scribeContext.component, this.attr.eventData.scribeContext.component || console.warn('You must configure your component with a "data-item-type" attribute in the DOM, with the value of activity, story, user, tweet, list, or trend in order for it to scribe properly.')
        }, this.getItemPosition = function(a) {
            var b = $(this.attr.genericInteractionItemSelector).index(a.closest(this.attr.genericInteractionItemSelector));
            if (b!=-1)
                return b
        }
    }
    module.exports = withScribeDataFromDom
});
define("app/ui/with_inline_retweet", ["module", "require", "exports", "core/i18n"], function(module, require, exports) {
    function withInlineRetweet() {
        this.handleSuccessfulRetweet = function(a, b) {
            this.retweetEventInfo !== undefined && this.trigger("uiDidRetweetSuccess", this.retweetEventInfo), delete this.retweetEventInfo
        }, this.around("toggleRetweet", function(a, b, c, d) {
            var e = d.$tweet.hasClass(this.attr.retweetedTweetClass), f = e ? "uiDidUnretweet": "uiDidRetweet", g = e ? _('Retweet'): _('Retweeted');
            e || (this.retweetEventInfo = d.tweetData), this.trigger(d.$tweet, f, d.tweetData), this.select("retweetSelector").find(".u-hiddenVisually").text(g)
        }), this.after("initialize", function() {
            if (!this.attr.loggedIn)
                return;
            this.on(document, "dataDidRetweet", this.handleSuccessfulRetweet)
        })
    }
    var _ = require("core/i18n");
    module.exports = withInlineRetweet
});
define("app/boot/app", ["module", "require", "exports", "app/ui/dialogs/age_gate_dialog", "app/ui/dialogs/block_or_report", "app/ui/dialogs/block_user_dialog", "app/boot/common", "app/ui/dialogs/buy_now_dialog", "app/data/commerce/buy_now_dialog_scribe", "app/ui/dialogs/captcha_challenge_dialog", "app/ui/dialogs/confirm_dialog", "app/ui/connect_badge", "app/ui/dialogs/create_custom_timeline_dialog", "app/ui/dialogs/curate_dialog", "app/ui/dashboard", "app/ui/dialogs/delete_tweet_dialog", "app/data/sms/device_verification", "app/ui/sms/device_verification_banner", "app/boot/direct_messages", "app/data/dm_poll", "app/ui/drag_state", "app/ui/banners/email_banner", "app/data/email_banner", "app/data/email_banner_scribe", "app/data/embed_scribe", "app/ui/dialogs/embed_tweet", "app/data/feedback/feedback", "app/ui/feedback/feedback_dialog", "app/ui/gallery/gallery", "app/data/gallery_scribe", "app/ui/global_loading_indicator", "app/ui/dialogs/goto_user_dialog", "app/ui/dialogs/keyboard_shortcuts_dialog", "app/ui/dialogs/leadgen_confirm_dialog", "app/ui/dialogs/list_membership_dialog", "app/ui/dialogs/list_operations_dialog", "app/data/sms/sms_download_scribe", "app/ui/more_tweet_actions_dropdown", "app/ui/multiline_ellipses", "app/ui/navigation_links", "app/data/notification_listener", "app/data/oembed", "app/data/oembed_scribe", "app/ui/dialogs/offers_dialog", "app/ui/page_title", "app/ui/page_visibility", "app/ui/dialogs/payment_order_dialog", "app/data/performance_stats_scribe", "app/boot/profile_popup", "app/ui/responsive_dashboard_width", "app/ui/dialogs/retweet_dialog", "app/data/saved_searches", "app/data/saved_searches_scribe", "app/data/sms_confirmation_data", "app/ui/dialogs/sms_confirmation_dialog", "app/data/sms_confirmation_scribe", "app/ui/search_query_source", "app/utils/setup_polling_with_backoff", "app/ui/dialogs/share_via_email_dialog", "app/data/spam_challenge", "app/ui/spoonbill", "app/ui/spoonbill_producer", "app/boot/top_bar", "app/data/typeahead/typeahead", "app/data/typeahead/typeahead_scribe", "app/ui/tweet_actions", "app/ui/tweet_state_updater", "app/ui/dialogs/uz_survey", "app/data/with_scribe_data_from_dom", "app/ui/with_inline_retweet"], function(module, require, exports) {
    var AgeGateDialog = require("app/ui/dialogs/age_gate_dialog"), BlockOrReportDialog = require("app/ui/dialogs/block_or_report"), BlockUserDialog = require("app/ui/dialogs/block_user_dialog"), bootCommon = require("app/boot/common"), BuyNowConfirmDialog = require("app/ui/dialogs/buy_now_dialog"), BuyNowDialogScribe = require("app/data/commerce/buy_now_dialog_scribe"), CaptchaChallengeDialog = require("app/ui/dialogs/captcha_challenge_dialog"), ConfirmDialog = require("app/ui/dialogs/confirm_dialog"), ConnectBadge = require("app/ui/connect_badge"), CreateCustomTimelineDialog = require("app/ui/dialogs/create_custom_timeline_dialog"), CurateDialog = require("app/ui/dialogs/curate_dialog"), Dashboard = require("app/ui/dashboard"), DeleteTweetDialog = require("app/ui/dialogs/delete_tweet_dialog"), DeviceVerification = require("app/data/sms/device_verification"), DeviceVerificationBanner = require("app/ui/sms/device_verification_banner"), directMessages = require("app/boot/direct_messages"), DMPoll = require("app/data/dm_poll"), DragState = require("app/ui/drag_state"), EmailBanner = require("app/ui/banners/email_banner"), EmailBannerData = require("app/data/email_banner"), EmailBannerScribe = require("app/data/email_banner_scribe"), EmbedScribe = require("app/data/embed_scribe"), EmbedTweetDialog = require("app/ui/dialogs/embed_tweet"), Feedback = require("app/data/feedback/feedback"), FeedbackDialog = require("app/ui/feedback/feedback_dialog"), Gallery = require("app/ui/gallery/gallery"), GalleryScribe = require("app/data/gallery_scribe"), GlobalLoadingIndicator = require("app/ui/global_loading_indicator"), GotoUserDialog = require("app/ui/dialogs/goto_user_dialog"), KeyboardShortcutsDialog = require("app/ui/dialogs/keyboard_shortcuts_dialog"), LeadGenConfirmDialog = require("app/ui/dialogs/leadgen_confirm_dialog"), ListMembershipDialog = require("app/ui/dialogs/list_membership_dialog"), ListOperationsDialog = require("app/ui/dialogs/list_operations_dialog"), MobileDownloadScribe = require("app/data/sms/sms_download_scribe"), MoreTweetActionsDropdown = require("app/ui/more_tweet_actions_dropdown"), MultilineEllipses = require("app/ui/multiline_ellipses"), NavigationLinks = require("app/ui/navigation_links"), NotificationListener = require("app/data/notification_listener"), OembedData = require("app/data/oembed"), OembedScribe = require("app/data/oembed_scribe"), OffersConfirmDialog = require("app/ui/dialogs/offers_dialog"), PageTitle = require("app/ui/page_title"), PageVisibility = require("app/ui/page_visibility"), PaymentOrderDialog = require("app/ui/dialogs/payment_order_dialog"), PerformanceStatsScribe = require("app/data/performance_stats_scribe"), profilePopup = require("app/boot/profile_popup"), ResponsiveDashboardWidth = require("app/ui/responsive_dashboard_width"), RetweetDialog = require("app/ui/dialogs/retweet_dialog"), SavedSearchesData = require("app/data/saved_searches"), SavedSearchesScribe = require("app/data/saved_searches_scribe"), SmsConfirmationData = require("app/data/sms_confirmation_data"), SmsConfirmationDialog = require("app/ui/dialogs/sms_confirmation_dialog"), SmsConfirmationScribe = require("app/data/sms_confirmation_scribe"), SearchQuerySource = require("app/ui/search_query_source"), setupPollingWithBackoff = require("app/utils/setup_polling_with_backoff"), ShareViaEmailDialog = require("app/ui/dialogs/share_via_email_dialog"), SpamChallengeData = require("app/data/spam_challenge"), Spoonbill = require("app/ui/spoonbill"), SpoonbillProducer = require("app/ui/spoonbill_producer"), topBar = require("app/boot/top_bar"), TypeaheadData = require("app/data/typeahead/typeahead"), TypeaheadScribe = require("app/data/typeahead/typeahead_scribe"), TweetActionsUI = require("app/ui/tweet_actions"), TweetStateUpdater = require("app/ui/tweet_state_updater"), uzSurvey = require("app/ui/dialogs/uz_survey"), withScribeDataFromDom = require("app/data/with_scribe_data_from_dom"), withInlineRetweet = require("app/ui/with_inline_retweet");
    module.exports = function(b) {
        bootCommon(b), topBar(b), PageTitle.attachTo(document, {
            noTeardown: !0
        }), Dashboard.attachTo(".dashboard", b), ResponsiveDashboardWidth.attachTo(document), b.dragAndDropPhotoUpload && DragState.attachTo("body"), SearchQuerySource.attachTo("body", {
            noTeardown: !0
        }), ConfirmDialog.attachTo(document, {
            noTeardown: !0
        }), CaptchaChallengeDialog.attachTo("#spam_challenge_dialog", {
            noTeardown: !0
        }), SpamChallengeData.attachTo(document, b, {
            noTeardown: !0
        }), b.loggedIn && (NotificationListener.attachTo(document, b, {
            noTeardown: !0
        }), DMPoll.attachTo(document, b, {
            noTeardown: !0
        }), ListMembershipDialog.attachTo("#list-membership-dialog", b, {
            noTeardown: !0
        }), ListOperationsDialog.attachTo("#list-operations-dialog", b, {
            noTeardown: !0
        }), CreateCustomTimelineDialog.attachTo("#create-custom-timeline-dialog", b), CurateDialog.attachTo("#curate-dialog", b), directMessages(b), EmailBanner.attachTo(document, {
            noTeardown: !0
        }), EmailBannerScribe.attachTo(document, {
            noTeardown: !0
        }), EmailBannerData.attachTo(document, {
            noTeardown: !0
        }), b.skipDeviceVerificationBanner || DeviceVerificationBanner.attachTo("#page-container"), DeviceVerification.attachTo(document), MobileDownloadScribe.attachTo(document)), TypeaheadScribe.attachTo(document, {
            noTeardown: !0
        }), TypeaheadData.attachTo(document, b.typeaheadData, {
            noTeardown: !0
        }), SavedSearchesScribe.attachTo(document, {
            noTeardown: !0
        }), SavedSearchesData.attachTo(document, {
            noTeardown: !0
        }), GotoUserDialog.attachTo("#goto-user-dialog", b), profilePopup({
            profileHoversEnabled: b.profileHoversEnabled,
            profile_user: b.profile_user,
            deviceEnabled: b.deviceEnabled,
            deviceVerified: b.deviceVerified,
            formAuthenticityToken: b.formAuthenticityToken,
            loggedIn: b.loggedIn,
            injectTweetOnFollow: b.injectTweetOnFollow
        }), GalleryScribe.attachTo(document, {
            noTeardown: !0
        });
        var c = {
            itemType: "tweet",
            noTeardown: !0,
            loggedIn: b.loggedIn,
            eventData: {
                scribeContext: {
                    component: "gallery"
                }
            }
        };
        Gallery.attachTo(".Gallery", b, c), TweetActionsUI.attachTo(".Gallery", b, c), MoreTweetActionsDropdown.attachTo(".Gallery", b, c), OembedScribe.attachTo(document, {
            noTeardown: !0
        }), OembedData.attachTo(document, b), KeyboardShortcutsDialog.attachTo(document, {
            noTeardown: !0,
            loggedIn: b.loggedIn,
            enableActivity: b.enableActivity
        }), RetweetDialog.attachTo("#retweet-tweet-dialog", b, {
            noTeardown: !0
        }), DeleteTweetDialog.attachTo("#delete-tweet-dialog", b, {
            noTeardown: !0
        }), BlockUserDialog.attachTo("#block-user-dialog", b, {
            noTeardown: !0
        }), ShareViaEmailDialog.attachTo("#share-via-email-dialog", b, {
            noTeardown: !0
        }), EmbedScribe.attachTo(document, {
            noTeardown: !0
        }), EmbedTweetDialog.attachTo("#embed-tweet-dialog", b, {
            noTeardown: !0
        }), BlockOrReportDialog.attachTo("#block-or-report-dialog", b, {
            noTeardown: !0
        }), AgeGateDialog.attachTo("#age-gate-dialog", b, {
            noTeardown: !0
        }), LeadGenConfirmDialog.attachTo("#leadgen-confirm-dialog", b), setupPollingWithBackoff("uiWantsToRefreshTimestamps"), PageVisibility.attachTo(document), PerformanceStatsScribe.attachTo(document, {
            noTeardown: !0
        }), NavigationLinks.attachTo(".dashboard", {
            eventData: {
                scribeContext: {
                    component: "dashboard_nav"
                }
            }
        }), NavigationLinks.attachTo(".analytics-footer-link"), FeedbackDialog.attachTo("#feedback_dialog", b.debugData, {
            noTeardown: !0
        }), Feedback.attachTo(document, b.debugData, {
            noTeardown: !0
        }), $(".uz_block_user_survey")[0] && uzSurvey.attachTo(".uz_block_user_survey"), Spoonbill.attachTo("#spoonbill-outer", b, {
            noTeardown: !0
        }), TweetActionsUI.mixin(withScribeDataFromDom, withInlineRetweet).attachTo("#spoonbill-outer", b, {
            eventData: {
                scribeContext: {
                    component: "spoonbill"
                }
            },
            tweetItemSelector: ".WebToast",
            favoriteSelector: ".js-actionFavorite",
            replySelector: ".js-actionReply",
            retweetSelector: ".js-actionInlineRetweet",
            noTeardown: !0
        }), SpoonbillProducer.attachTo("#spoonbill-outer", b, {
            noTeardown: !0
        }), GlobalLoadingIndicator.attachTo(document, b, {
            noTeardown: !0
        }), ConnectBadge.attachTo(document, b, {
            noTeardown: !0
        }), BuyNowConfirmDialog.attachTo("#buy-now-dialog", b, {
            noTeardown: !0
        }), BuyNowDialogScribe.attachTo(document, {
            noTeardown: !0
        }), OffersConfirmDialog.attachTo("#offers-dialog", b, {
            noTeardown: !0
        }), PaymentOrderDialog.attachTo("#payment-order-detail-dialog", b, {
            noTeardown: !0
        }), MultilineEllipses.attachTo(document), TweetStateUpdater.attachTo(document, b, {
            noTeardown: !0
        }), SmsConfirmationData.attachTo("#sms-confirmation-dialog"), SmsConfirmationDialog.attachTo("#sms-confirmation-dialog", b, {
            noTeardown: !0
        }), SmsConfirmationScribe.attachTo(document)
    }
});
define("lib/twitter_cldr", ["module", "require", "exports"], function(a, b, c) {
    (function() {
        var a, b, d, e;
        a = {};
        a.is_rtl=!1;
        a.Utilities = function() {
            function a() {}
            a.from_char_code = function(a) {
                if (a > 65535) {
                    a -= 65536;
                    return String.fromCharCode(55296 + (a>>10), 56320 + (a & 1023)
                    )
                }
                return String.fromCharCode(a)
            };
            a.char_code_at = function(a, b) {
                var c, d, e, f, g, i;
                a += "";
                d = a.length;
                i = /[\uD800-\uDBFF][\uDC00-\uDFFF]/g;
                while (i.exec(a) !== null) {
                    f = i.lastIndex;
                    if (!(f - 2 < b))
                        break;
                    b += 1
                }
                if (b >= d || b < 0)
                    return NaN;
                c = a.charCodeAt(b);
                if (55296 <= c && c <= 56319)
                {
                    e = c;
                    g = a.charCodeAt(b + 1);
                    return (e - 55296) * 1024 + (g - 56320) + 65536
                }
                return c
            };
            a.unpack_string = function(a) {
                var b, c, d, e, f;
                d = [];
                for (c = e = 0, f = a.length; 0 <= f ? e < f : e > f; c = 0 <= f?++e : --e) {
                    b = this.char_code_at(a, c);
                    if (!b)
                        break;
                    d.push(b)
                }
                return d
            };
            a.pack_array = function(a) {
                var b
                ;
                return function() {
                    var c, d, e;
                    e = [];
                    for (c = 0, d = a.length; c < d; c++) {
                        b = a[c];
                        e.push(this.from_char_code(b))
                    }
                    return e
                }.call(this).join("")
            };
            a.arraycopy = function(a, b, c, d, e) {
                var f, g, i, j, k;
                k = a.slice(b, b + e);
                for (f = i = 0, j = k.length; i < j; f=++i) {
                    g = k[f];
                    c[d + f] = g
                }
            };
            a.max = function(
            a) {
                var b, c, d, e, f, g, i, j;
                d = null;
                for (e = f = 0, i = a.length; f < i; e=++f) {
                    b = a[e];
                    if (b != null) {
                        d = b;
                        break
                    }
                }
                for (c = g = e, j = a.length; e <= j ? g <= j : g >= j; c = e <= j?++g : --g)
                    a[c] > d && (d = a[c]);
                return d
            };
            a.min = function(a) {
                var b, c, d, e, f, g, i, j;
                d = null;
                for (e = f = 0, i = a.length; f < i; e=++f) {
                    b = a[e];
                    if (
                    b != null) {
                        d = b;
                        break
                    }
                }
                for (c = g = e, j = a.length; e <= j ? g <= j : g >= j; c = e <= j?++g : --g)
                    a[c] < d && (d = a[c]);
                return d
            };
            a.is_even = function(a) {
                return a%2 === 0
            };
            a.is_odd = function(a) {
                return a%2 === 1
            };
            return a
        }();
        a.PluralRules = function() {
            function a() {}
            a.rules = {
                keys: ["one", "other"],
                rule
                : function(a) {
                    return function() {
                        return a == 1 ? "one" : "other"
                    }()
                }
            };
            a.all = function() {
                return this.rules.keys
            };
            a.rule_for = function(a) {
                var b;
                try {
                    return this.rules.rule(a)
                } catch (c) {
                    b = c;
                    return "other"
                }
            };
            return a
        }();
        a.TimespanFormatter = function() {
            function b() {
                this.approximate_multiplier = .75
                ;
                this.default_type = "default";
                this.tokens = {
                    ago: {
                        second: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " second ago",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " seconds ago",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        minute: {
                            "default": {
                                one: [{
                                    value
                                    : "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " minute ago",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " minutes ago",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        hour: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hour ago",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value
                                    : "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hours ago",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        day: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " day ago",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " days ago",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        week: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " week ago",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " weeks ago",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        month: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " month ago",
                                    type: "plaintext"
                                }
                                ],
                                other
                                : [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " months ago",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        year: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " year ago",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " years ago",
                                    type: "plaintext"
                                }
                                ]
                            }
                        }
                    },
                    until
                    : {
                        second: {
                            "default": {
                                one: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " second",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " seconds",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        minute: {
                            "default": {
                                one
                                : [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " minute",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " minutes",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        hour: {
                            "default": {
                                one: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hour",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hours",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        day: {
                            "default": {
                                one: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " day",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " days",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        week: {
                            "default": {
                                one: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " week",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " weeks",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        month: {
                            "default": {
                                one: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " month",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "In ",
                                    type
                                    : "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " months",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        year: {
                            "default": {
                                one: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " year",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "In ",
                                    type: "plaintext"
                                }, {
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " years",
                                    type: "plaintext"
                                }
                                ]
                            }
                        }
                    },
                    none: {
                        second: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " second",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " seconds",
                                    type: "plaintext"
                                }
                                ]
                            },
                            "short": {
                                one: [{
                                    value: "{0}"
                                    ,
                                    type: "placeholder"
                                }, {
                                    value: " sec",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " secs",
                                    type: "plaintext"
                                }
                                ]
                            },
                            abbreviated: {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: "s",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value
                                    : "s",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        minute: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " minute",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " minutes",
                                    type: "plaintext"
                                }
                                ]
                            },
                            "short": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " min"
                                    ,
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " mins",
                                    type: "plaintext"
                                }
                                ]
                            },
                            abbreviated: {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: "m",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: "m",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        hour: {
                            "default"
                            : {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hour",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hours",
                                    type: "plaintext"
                                }
                                ]
                            },
                            "short": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hr",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " hrs",
                                    type: "plaintext"
                                }
                                ]
                            },
                            abbreviated: {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: "h",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: "h",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        day: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " day"
                                    ,
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " days",
                                    type: "plaintext"
                                }
                                ]
                            },
                            "short": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " day",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " days",
                                    type: "plaintext"
                                }
                                ]
                            },
                            abbreviated
                            : {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: "d",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: "d",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        week: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " week",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " weeks",
                                    type: "plaintext"
                                }
                                ]
                            },
                            "short": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " wk",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " wks",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        month: {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " month"
                                    ,
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " months",
                                    type: "plaintext"
                                }
                                ]
                            },
                            "short": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " mth",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " mths",
                                    type: "plaintext"
                                }
                                ]
                            }
                        },
                        year
                        : {
                            "default": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " year",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " years",
                                    type: "plaintext"
                                }
                                ]
                            },
                            "short": {
                                one: [{
                                    value: "{0}",
                                    type: "placeholder"
                                }, {
                                    value: " yr",
                                    type: "plaintext"
                                }
                                ],
                                other: [{
                                    value: "{0}"
                                    ,
                                    type: "placeholder"
                                }, {
                                    value: " yrs",
                                    type: "plaintext"
                                }
                                ]
                            }
                        }
                    }
                };
                this.time_in_seconds = {
                    second: 1,
                    minute: 60,
                    hour: 3600,
                    day: 86400,
                    week: 604800,
                    month: 2629743.83,
                    year: 31556926
                }
            }
            b.prototype.format = function(b, c) {
                var d, e, f, g, i, j;
                c == null && (c = {});
                g = {};
                for (d in c) {
                    f = c[d];
                    g[d] =
                    f
                }
                g.direction || (g.direction = b < 0 ? "ago" : "until");
                if (g.unit === null || g.unit === void 0)
                    g.unit = this.calculate_unit(Math.abs(b), g);
                g.type || (g.type = this.default_type);
                g.number = this.calculate_time(Math.abs(b), g.unit);
                e = this.calculate_time(Math.abs(b), g.unit);
                g.rule =
                a.PluralRules.rule_for(e);
                i = function() {
                    var a, b, c, d;
                    c = this.tokens[g.direction][g.unit][g.type][g.rule];
                    d = [];
                    for (a = 0, b = c.length; a < b; a++) {
                        j = c[a];
                        d.push(j.value)
                    }
                    return d
                }.call(this);
                return i.join("").replace(/\{[0-9]\}/, e.toString())
            };
            b.prototype.calculate_unit =
            function(a, b) {
                var c, d, e, f;
                b == null && (b = {});
                f = {};
                for (c in b) {
                    e = b[c];
                    f[c] = e
                }
                f.approximate == null && (f.approximate=!1);
                d = f.approximate ? this.approximate_multiplier : 1;
                return a < this.time_in_seconds.minute * d ? "second" : a < this.time_in_seconds.hour * d ? "minute" : a < this.time_in_seconds
                .day * d ? "hour" : a < this.time_in_seconds.week * d ? "day" : a < this.time_in_seconds.month * d ? "week" : a < this.time_in_seconds.year * d ? "month" : "year"
            };
            b.prototype.calculate_time = function(a, b) {
                return Math.round(a / this.time_in_seconds[b])
            };
            return b
        }();
        e = typeof c != "undefined" &&
        c !== null ? c : (this.TwitterCldr = {}, this.TwitterCldr);
        for (b in a) {
            d = a[b];
            e[b] = d
        }
    }).call(this)
})
