define("site/account/Account", ["backbone"], function(a) {
    var b = a.Model.extend({
        defaults: {
            id: null,
            total_sites: 1,
            logoutURL: "/s/logout"
        },
        initialize: function() {
            this.id && this.fetch()
        },
        dataForView: function() {
            var a = this.toJSON();
            return a.displayName = this.get("email") || this.get("username"), a.isWebsId=!!this.get("email"), a
        },
        url: function() {
            return "/api/accounts/" + this.id
        }
    });
    return b
}), define("instance/site/account", ["site/account/Account"], function(a) {
    return webs.account ? new a(webs.account) : null
}), define("site/site/Site", ["backbone"], function(a) {
    var b = a.Model.extend({
        initialize: function() {
            this.siteBuilderVersion = 3
        },
        url: function() {
            return "/api/sites/" + this.id
        },
        headerEditable: function() {
            return this.siteBuilderVersion != 3
        }
    });
    return b
}), define("instance/site/site", ["underscore", "site/site/Site"], function(a, b) {
    return new b({
        id: webs.site.id,
        url: webs.site.url,
        username: webs.site.username,
        preferredProvidedUrl: a.find(webs.site.providedRootURLs, function(a) {
            return !a.match(/^http:\/\/www.freeweb[sz].com\/.*$/)
        })
    })
}), define("site/accountSettings/Sites", ["underscore", "backbone"], function(a, b) {
    var c = b.Model.extend({
        dataForView: function() {
            return this.toJSON()
        }
    }), d = b.Collection.extend({
        model: c,
        initialize: function(a) {
            this.memberId = a.memberId, this.includeSubscriptions = a.includeSubscriptions
        },
        parse: function(b) {
            return this.page_size = b.page_size, this.last_page = b.last_page, this.total_size = b.total_size, this.page = b.page, a.filter(b.elements, function(a) {
                return a !== null
            })
        },
        url: function() {
            return this.includeSubscriptions ? "/api/accounts/" + this.memberId + "/sites?pagesize=8&includeSubscriptions=true" : "/api/accounts/" + this.memberId + "/sites?pagesize=10"
        },
        dataForView: function() {
            return this.toJSON()
        }
    });
    return d
}), define("site/accountSettings/NotificationView", ["backbone"], function(a) {
    var b = a.View.extend({
        el: "#websid_popover_notification",
        className: "notification banner hidden",
        initialize: function(a) {
            this.messageContainer = this.$(".message"), this.popover = a.popover
        },
        render: function(a, b) {
            var c = this;
            this.$el.attr("class", "notification banner " + a), this.messageContainer.text(b), this.popover.reposition()
        },
        hide: function() {
            this.$el.attr("class", this.className), this.messageContainer.text(""), this.popover.reposition()
        }
    });
    return b
}), define("site/accountSettings/StatusView", ["backbone"], function(a) {
    var b = a.View.extend({
        tagName: "span",
        className: "status hidden",
        currentClass: "",
        timeout: null,
        initialize: function() {
            this.messageContainer = $('<span class="message"></span>'), this.$el.append(this.messageContainer)
        },
        render: function(a, b) {
            var c = this;
            this.currentClass = a, this.$el.stop().removeClass("hidden").parents(".control").addClass(a), this.messageContainer.text(b), clearTimeout(this.timeout), this.timeout = setTimeout(function() {
                c.hide()
            }, 3e3)
        },
        hide: function(a) {
            var b = this, c = a ? 1: 400;
            clearTimeout(this.timeout), this.$el.fadeOut(c, function() {
                b.$el.addClass("hidden").removeAttr("style").parents(".control").removeClass(b.currentClass), b.currentClass = "", b.messageContainer.text("")
            })
        }
    });
    return b
}), define("site/ErrorLocalizer", ["underscore"], function(a) {
    var b = {
        bad_gateway: 502,
        bad_request: 400,
        conflict: 409,
        destination_locked: 421,
        expectation_failed: 417,
        failed_dependency: 424,
        forbidden: 403,
        gateway_timeout: 504,
        gone: 410,
        http_version_not_supported: 505,
        insufficient_space_on_resource: 419,
        insufficient_storage: 507,
        internal_server_error: 500,
        length_required: 411,
        locked: 423,
        loop_detected: 508,
        method_failure: 420,
        method_not_allowed: 405,
        not_acceptable: 406,
        not_extended: 510,
        not_found: 404,
        not_implemented: 501,
        payment_required: 402,
        precondition_failed: 412,
        proxy_authentication_required: 407,
        request_entity_too_large: 413,
        request_timeout: 408,
        request_uri_too_long: 414,
        requested_range_not_satisfiable: 416,
        service_unavailable: 503,
        unauthorized: 401,
        unprocessable_entity: 422,
        unsupported_media_type: 415,
        upgrade_required: 426,
        variant_also_negotiates: 506
    }, c = function(c) {
        var d, e;
        try {
            e = JSON.parse(c.responseText);
            if (e.message && e.message.length > 0)
                return e.message;
            throw "No Message"
        } catch (f) {
            return a.each(b, function(a, b) {
                c.status === a && (d = b)
            }), "status." + d
        }
    };
    return {
        parse: c
    }
});
var templates = [];
define("dust", {
    load: function(a, b, c, d) {
        if (d.isBuild) {
            var e = b.toUrl(a), f = require.nodeRequire("fs"), g = require.nodeRequire("dustjs-linkedin"), h = f.readFileSync(e);
            templates[a] = g.compile(h.toString(), a), c(h)
        } else 
            b([a, "dust-core", "dust-helpers"])
    },
    write: function(a, b, c) {
        c("define('" + a + "!" + b + "', ['dust-helpers', 'dust-core'], function(helpers){" + templates[b] + "return function(data, callback){ dust.render('" + b + "', helpers.push(data), callback); }" + "});")
    }
}), define("dust!site/accountSettings/dropdown/templates/dropdownView.dust", ["dust-helpers", "dust-core"], function(a) {
    return function() {
        function a(a, d) {
            return a.exists(d.get(["account"], !1), d, {
                "else": b,
                block: c
            }, null).write('<div id="websid_dropdown_links"><ul>').exists(d.get(["createSiteLink"], !1), d, {
                block: h
            }, null).write('<li><a href="').reference(d.get(["logoutUrl"], !1), d, "h").write('" id="websid_dropdown_logout">').section(d.get(["l"], !1), d, {
                block: j
            }, null).write("</a></li></ul></div>")
        }
        function b(a, b) {
            return a.write('<div id="websid_dropdown_user"><h5 class="fullheight">').reference(b.getPath(!1, ["site", "username"]), b, "h").write("</h5></div>")
        }
        function c(a, b) {
            return a.write('<div id="websid_dropdown_user"><h5 id="fwMember_email">').reference(b.getPath(!1, ["account", "displayName"]), b, "h").write('</h5><a class="settings" href="#">').section(b.get(["l"], !1), b, {
                block: d
            }, null).write('</a></div><div id="websid_dropdown_others"><ul class="notranslate"><li class="loading">').section(b.get(["l"], !1), b, {
                block: e
            }, null).write("</li>").exists(b.get(["showAllSitesLink"], !1), b, {
                block: f
            }, null).write("</ul></div>")
        }
        function d(a, b) {
            return a.write("webs.websid.accountSettings")
        }
        function e(a, b) {
            return a.write("webs.websid.loadingSites")
        }
        function f(a, b) {
            return a.write('<li><a href="/sites">').section(b.get(["l"], !1), b, {
                block: g
            }, null).write("</a></li>")
        }
        function g(a, b) {
            return a.write("webs.websid.viewAllSites")
        }
        function h(a, b) {
            return a.write('<li><a href="').reference(b.get(["createSiteLink"], !1), b, "h").write('" id="websid_dropdown_create">').section(b.get(["l"], !1), b, {
                block: i
            }, null).write("</a></li>")
        }
        function i(a, b) {
            return a.write("webs.websid.addSite")
        }
        function j(a, b) {
            return a.write("webs.util.logout")
        }
        return dust.register("site/accountSettings/dropdown/templates/dropdownView.dust", a), a
    }(), function(b, c) {
        dust.render("site/accountSettings/dropdown/templates/dropdownView.dust", a.push(b), c)
    }
}), define("dust!site/accountSettings/dropdown/templates/dropdownSiteView.dust", ["dust-helpers", "dust-core"], function(a) {
    return function() {
        function a(a, c) {
            return a.exists(c.get(["currentSite"], !1), c, {
                "else": b,
                block: d
            }, null).write('<a class="view-site" href="').reference(c.getPath(!1, ["site", "url"]), c, "h").write('" data-tooltip="').section(c.get(["l"], !1), c, {
                block: f
            }, null).write(": ").reference(c.getPath(!1, ["site", "title"]), c, "h").write('" target="_blank"><span></span></a>')
        }
        function b(a, b) {
            return a.write('<a href="/s/webs/switchAccount?userID=').reference(b.getPath(!1, ["site", "id"]), b, "h").write("&next=/sites/").reference(b.getPath(!1, ["site", "id"]), b, "h").write('/dashboard/" data-tooltip="').section(b.get(["l"], !1), b, {
                block: c
            }, null).write(": ").reference(b.getPath(!1, ["site", "title"]), b, "h").write('">').reference(b.getPath(!1, ["site", "title"]), b, "h").write("</a>")
        }
        function c(a, b) {
            return a.write("webs.util.manage")
        }
        function d(a, b) {
            return a.write('<a class="active" href="/s/webs/switchAccount?userID=').reference(b.getPath(!1, ["site", "id"]), b, "h").write("&next=/sites/").reference(b.getPath(!1, ["site", "id"]), b, "h").write('/dashboard/" data-tooltip="').section(b.get(["l"], !1), b, {
                block: e
            }, null).write(": ").reference(b.getPath(!1, ["site", "title"]), b, "h").write('">').reference(b.getPath(!1, ["site", "title"]), b, "h").write("</a>")
        }
        function e(a, b) {
            return a.write("webs.util.manage")
        }
        function f(a, b) {
            return a.write("webs.util.view")
        }
        return dust.register("site/accountSettings/dropdown/templates/dropdownSiteView.dust", a), a
    }(), function(b, c) {
        dust.render("site/accountSettings/dropdown/templates/dropdownSiteView.dust", a.push(b), c)
    }
}), define("site/accountSettings/dropdown/DropdownSiteListView", ["underscore", "backbone", "dust!site/accountSettings/dropdown/templates/dropdownSiteView.dust"], function(a, b, c) {
    var d = b.View.extend({
        el: "#websid_dropdown_others ul.notranslate",
        initialize: function() {
            var a = this;
            this.collection.on("reset", this.render, this), this.collection.fetch(), webs.app.on("sites.change", this.collection.fetch, this.collection)
        },
        render: function() {
            var b = this;
            return this.$(".loading").removeClass("hidden"), this.$(".dd_site").remove(), a.each(this.collection.models.reverse(), function(a) {
                var c = new e({
                    model: a,
                    tagName: "li",
                    id: "dropdown_site_" + a.get("id"),
                    className: "dd_site"
                });
                b.$(".loading").addClass("hidden"), b.$el.prepend(c.el), c.render()
            }), this
        }
    }), e = b.View.extend({
        tagName: "li",
        events: {},
        initialize: function() {
            a.bindAll(this, "_rendered"), this.id = this.model.get("id")
        },
        render: function() {
            var a = this.model.dataForView();
            return a.title.length === 0 && (a.title = a.url), c({
                site: a,
                props: webs.props,
                currentSite: webs.site.id === this.id
            }, this._rendered), this
        },
        _rendered: function(a, b) {
            this.$el.html(b)
        }
    });
    return d
}), define("dust!site/accountSettings/dialog/templates/popoverView.dust", ["dust-helpers", "dust-core"], function(a) {
    return function() {
        function a(a, k) {
            return a.write('<div id="websid_popover"><div id="websid_popover_notification" class="hidden"><p class="message"></p></div><div id="account_settings"><div class="w-form"><div class="w-form-items"></div></div></div><div id="site_list"><ul><li class="loading"><div class="loadingSpinner"></div>').section(k.get(["l"], !1), k, {
                block: b
            }, null).write('</li><li class="hidden" id="delete_account_container"><a class="delete_account" href="#">').section(k.get(["l"], !1), k, {
                block: c
            }, null).write('</a></li><li class="hidden" id="delete_account_confirmation"><h4>').section(k.get(["l"], !1), k, {
                block: d
            }, null).write("</h4><h5>").section(k.get(["l"], !1), k, {
                block: e
            }, null).write(' <a href="#">').section(k.get(["l"], !1), k, {
                block: f
            }, null).write('</a>.</h5><a class="w-button delete" data-tooltip="').section(k.get(["l"], !1), k, {
                block: g
            }, null).write('" href="/s/webs/delete">').section(k.get(["l"], !1), k, {
                block: h
            }, null).write('</a><a class="w-button mid-gray delete_account_cancel" data-tooltip="').section(k.get(["l"], !1), k, {
                block: i
            }, null).write('">').section(k.get(["l"], !1), k, {
                block: j
            }, null).write("</a></li></ul></div></div>")
        }
        function b(a, b) {
            return a.write("webs.websid.loadingSites")
        }
        function c(a, b) {
            return a.write("webs.websid.deleteAccount")
        }
        function d(a, b) {
            return a.write("webs.websid.delete.confirmation")
        }
        function e(a, b) {
            return a.write("webs.websid.somethingNotLiked")
        }
        function f(a, b) {
            return a.write("webs.websid.tellUs")
        }
        function g(a, b) {
            return a.write("webs.websid.deleteAccount")
        }
        function h(a, b) {
            return a.write("webs.websid.deleteAccount")
        }
        function i(a, b) {
            return a.write("webs.websid.cancel")
        }
        function j(a, b) {
            return a.write("webs.websid.cancel")
        }
        return dust.register("site/accountSettings/dialog/templates/popoverView.dust", a), a
    }(), function(b, c) {
        dust.render("site/accountSettings/dialog/templates/popoverView.dust", a.push(b), c)
    }
}), define("dust!site/accountSettings/dialog/templates/credentialView.dust", ["dust-helpers", "dust-core"], function(a) {
    return function() {
        function a(a, g) {
            return a.write('<div class="title"><label for="email">').reference(g.get(["label"], !1), g, "h").write(':</label></div><div class="control text"><span class="value_container">').exists(g.get(["showValue"], !1), g, {
                "else": b,
                block: c
            }, null).write('<a class="edit" data-tooltip="').section(g.get(["l"], !1), g, {
                block: d
            }, null).write('">').section(g.get(["l"], !1), g, {
                block: e
            }, null).write('</a></span><div class="field_container hidden">').exists(g.get(["showCancel"], !1), g, {
                block: f
            }, null).write('<a class="settings_button submit" data-tooltip="').section(g.get(["l"], !1), g, {
                block: i
            }, null).write('">').section(g.get(["l"], !1), g, {
                block: j
            }, null).write("</a>").section(g.get(["inputs"], !1), g, {
                block: k
            }, {
                showValue: g.get(["showValue"], !1)
            }).write("</div></div>")
        }
        function b(a, b) {
            return a.write('<span class="value">&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;</span>')
        }
        function c(a, b) {
            return a.write('<span class="value">').reference(b.get(["value"], !1), b, "h").write("</span>")
        }
        function d(a, b) {
            return a.write("webs.util.edit")
        }
        function e(a, b) {
            return a.write("webs.util.edit")
        }
        function f(a, b) {
            return a.write('<a class="settings_button cancel" data-tooltip="').section(b.get(["l"], !1), b, {
                block: g
            }, null).write('">').section(b.get(["l"], !1), b, {
                block: h
            }, null).write("</a>")
        }
        function g(a, b) {
            return a.write("webs.util.cancel")
        }
        function h(a, b) {
            return a.write("webs.util.cancel")
        }
        function i(a, b) {
            return a.write("webs.util.submit")
        }
        function j(a, b) {
            return a.write("webs.util.submit")
        }
        function k(a, b) {
            return a.write('<input type="').reference(b.get(["type"], !1), b, "h").write('" name="').reference(b.get(["name"], !1), b, "h").write('" ').exists(b.get(["placeholder"], !1), b, {
                block: l
            }, null).write(" ").exists(b.get(["showValue"], !1), b, {
                block: m
            }, null).write(">")
        }
        function l(a, b) {
            return a.write('placeholder="').reference(b.get(["placeholder"], !1), b, "h").write('"')
        }
        function m(a, b) {
            return a.write('value="').reference(b.get(["value"], !1), b, "h").write('"')
        }
        return dust.register("site/accountSettings/dialog/templates/credentialView.dust", a), a
    }(), function(b, c) {
        dust.render("site/accountSettings/dialog/templates/credentialView.dust", a.push(b), c)
    }
}), define("dust!site/accountSettings/dialog/templates/popoverSiteView.dust", ["dust-helpers", "dust-core"], function(a) {
    return function() {
        function a(a, c) {
            return a.write('<div class="site_info"><img class="thumb" src="http://custom.pagepeeker.com/thumbs.php?size=l&url=').reference(c.getPath(!1, ["site", "url"]), c, "h").write('" width="22" height="22" alt="website preview"><span class="site_name"><a href="').reference(c.getPath(!1, ["site", "url"]), c, "h").write('" target="_blank">').reference(c.getPath(!1, ["site", "title"]), c, "h").write("<span></span></a></span>").exists(c.get(["canDelete"], !1), c, {
                "else": b,
                block: e
            }, null).write("</div>").exists(c.get(["canDelete"], !1), c, {
                "else": h,
                block: n
            }, null).write('<div class="removal_notice hidden">').exists(c.get(["canDelete"], !1), c, {
                "else": t,
                block: v
            }, null).write("</div>")
        }
        function b(a, b) {
            return a.write('<a class="action confirm_user" data-tooltip="').section(b.get(["l"], !1), b, {
                block: c
            }, null).write('" href="#">').section(b.get(["l"], !1), b, {
                block: d
            }, null).write("</a>")
        }
        function c(a, b) {
            return a.write("webs.websid.website.unsubscribe")
        }
        function d(a, b) {
            return a.write("webs.websid.website.unsubscribe")
        }
        function e(a, b) {
            return a.write('<a class="action confirm_user" data-tooltip="').section(b.get(["l"], !1), b, {
                block: f
            }, null).write('" href="#">').section(b.get(["l"], !1), b, {
                block: g
            }, null).write("</a>")
        }
        function f(a, b) {
            return a.write("webs.websid.website.delete")
        }
        function g(a, b) {
            return a.write("webs.websid.website.delete")
        }
        function h(a, b) {
            return a.write('<div class="confirm hidden"><p>').section(b.get(["l"], !1), b, {
                block: i
            }, null).write('</p><div class="loadingSpinner hidden"></div><a class="settings_button cancel" data-tooltip="').section(b.get(["l"], !1), b, {
                block: j
            }, null).write('">').section(b.get(["l"], !1), b, {
                block: k
            }, null).write('</a><a class="settings_button submit submit_unsubscribe" data-tooltip="').section(b.get(["l"], !1), b, {
                block: l
            }, null).write('">').section(b.get(["l"], !1), b, {
                block: m
            }, null).write("</a></div>")
        }
        function i(a, b) {
            return a.write("webs.websid.unsubscribe.confirmation")
        }
        function j(a, b) {
            return a.write("webs.util.cancel")
        }
        function k(a, b) {
            return a.write("webs.util.cancel")
        }
        function l(a, b) {
            return a.write("webs.util.submit")
        }
        function m(a, b) {
            return a.write("webs.util.submit")
        }
        function n(a, b) {
            return a.write('<div class="confirm w-form hidden"><dl class="w-form-items"><dt class="title"><label for="').reference(b.getPath(!1, ["site", "id"]), b, "h").write('_password">').section(b.get(["l"], !1), b, {
                block: o
            }, null).write('</label></dt><dd class="control text"><div class="loadingSpinner hidden"></div><input type="password" name="').reference(b.getPath(!1, ["site", "id"]), b, "h").write('_password"><a class="settings_button cancel" data-tooltip="').section(b.get(["l"], !1), b, {
                block: p
            }, null).write('">').section(b.get(["l"], !1), b, {
                block: q
            }, null).write('</a><a class="settings_button submit submit_delete" data-tooltip="').section(b.get(["l"], !1), b, {
                block: r
            }, null).write('">').section(b.get(["l"], !1), b, {
                block: s
            }, null).write("</a></dd></dl></div>")
        }
        function o(a, b) {
            return a.write("webs.websid.enterPasswordToDelete")
        }
        function p(a, b) {
            return a.write("webs.util.cancel")
        }
        function q(a, b) {
            return a.write("webs.util.cancel")
        }
        function r(a, b) {
            return a.write("webs.util.submit")
        }
        function s(a, b) {
            return a.write("webs.util.submit")
        }
        function t(a, b) {
            return a.section(b.get(["l"], !1), b, {
                block: u
            }, null).write(" <em>").reference(b.getPath(!1, ["site", "title"]), b, "h").write("</em>")
        }
        function u(a, b) {
            return a.write("webs.websid.unsubscribed")
        }
        function v(a, b) {
            return a.write("<em>").reference(b.getPath(!1, ["site", "title"]), b, "h").write("</em> ").section(b.get(["l"], !1), b, {
                block: w
            }, null).write(".")
        }
        function w(a, b) {
            return a.write("webs.websid.website.deleted")
        }
        return dust.register("site/accountSettings/dialog/templates/popoverSiteView.dust", a), a
    }(), function(b, c) {
        dust.render("site/accountSettings/dialog/templates/popoverSiteView.dust", a.push(b), c)
    }
}), function() {
    function a(a, b) {
        window.Popover = function(a, b) {
            return this.url = a, this.options = b, this.viewportwidth = "", this.viewportheight = "", this
        }, window.Popover.prototype = {
            reposition: function() {
                websover.reposition()
            },
            resize: function(a) {
                websover.resize(a)
            },
            show: function() {
                var c = this;
                websover.removeBadElements();
                var d = this.options, e = typeof d.scrolling != "undefined" && d.scrolling ? "auto": "no";
                websover.options = d, websover.width = d.width ? d.width : 100, websover.height = d.height ? d.height : 100, websover.height === "auto" ? websover.top = 0 : websover.top =- (parseInt(websover.extraHeight, 10) + parseInt(websover.height, 10)) / 2, typeof d.closeButton == "undefined" && (d.closeButton=!0), typeof d.closeOnEsc == "undefined" && (d.closeOnEsc=!0), websover.feedback = websover.feedback ? websover.feedback : a("<div/>").attr("id", "websover_feedback").appendTo(document.body), websover.feedback.hide();
                if (!d.hasOwnProperty("showCover") || d.showCover)
                    websover.currentShadow || (websover.currentShadow = document.createElement("div"), websover.currentShadow.setAttribute("id", "websover_shadow"), document.body.appendChild(websover.currentShadow), a(websover.currentShadow).bind("drop dragover", function(a) {
                        a.preventDefault()
                    })), websover.currentShadow.style.display = "block";
                websover.garbageBin || (websover.garbageBin = document.createElement("div"), websover.garbageBin.cssText = "display: none;", document.body.appendChild(websover.garbageBin)), websover.currentWrap || (websover.currentWrap = document.createElement("div"), websover.currentWrap.setAttribute("id", "websover_wrap"), document.body.appendChild(websover.currentWrap)), d.className ? websover.currentWrap.setAttribute("class", d.className) : websover.currentWrap.removeAttribute("class"), websover.currentBorder || (websover.currentBorder = document.createElement("div"), websover.currentBorder.setAttribute("id", "websover_border"), websover.currentWrap.appendChild(websover.currentBorder)), websover.currentPopover || (websover.currentPopover = document.createElement("div"), websover.currentPopover.setAttribute("id", "inner_websover"), websover.currentBorder.appendChild(websover.currentPopover));
                var f = typeof jQuery != "undefined" && this.options.feedback;
                websover.$fb = null;
                if (f) {
                    var g = this.options.feedback, h = jQuery(g.ref);
                    if (h.length > 0) {
                        var i = h.width(), j = h.height(), k = h.offset(), l = jQuery(h[0].ownerDocument).scrollTop(), m = g.offset ? g.offset.left: 0, n = g.offset ? g.offset.top: 0;
                        websover.$fb = jQuery(websover.feedback).css({
                            left: k.left + m,
                            top: k.top + n - l,
                            width: i,
                            height: j
                        })
                    }
                }
                if (typeof this.url == "string") {
                    websover.currentContent = document.createElement("iframe"), websover.currentContent.setAttribute("id", "websover_iframe"), websover.currentContent.setAttribute("rel", this.url), websover.currentContent.setAttribute("name", "popoverIframe"), websover.currentContent.setAttribute("frameborder", "0");
                    try {
                        websover.currentContent.frameBorder = 0
                    } catch (o) {}
                    websover.currentContent.setAttribute("scrolling", e), websover.currentContent.src = this.url, a(websover.currentContent).load(function() {
                        websover.currentContent.style.visibility = "visible", a(p).removeClass("loading")
                    }), websover.currentContent.style.cssText = "visibility: hidden; " + (e === "no" ? "overflow: hidden;" : "overflow: auto;")
                } else 
                    this.url.style.display = "block", websover.currentContent = this.url, websover.currentContent.className = "websover_content";
                websover.currentPopover.innerHTML = '<div id="websover_header"><span id="websover_title">' + d.heading + "</span>" + (d.closeButton ? '<a href="#" title="' + b("webs.util.close") + '" id="websover_close">' + b("webs.util.close") + "</a>" : "") + "</div>" + '<div id="websover_iframe_container" class="loading"></div>', websover.currentWrap.style.display = "block", websover.currentPopover.style.width = websover.width + "px", websover.currentBorder.style.visibility = "hidden", websover.currentBorder.style.width = websover.width + "px", websover.currentBorder.style.left =- (parseInt(websover.extraWidth, 10) + parseInt(websover.width, 10)) / 2 + "px", websover.currentBorder.style.top = websover.top + "px";
                var p = document.getElementById("websover_iframe_container");
                return typeof this.url == "string" ? p.style.cssText = "height: " + (websover.height - 26) + "px" : a(p).removeClass("loading"), websover.currentContent.tagName === "IFRAME" ? c.afterShow() : (websover.resize({
                    height: websover.currentContent.offsetHeight,
                    width: websover.width
                }), websover.currentBorder.style.height = "auto", c.afterShow()), document.getElementById("websover_close") && (document.getElementById("websover_close").onclick = function() {
                    return websover.hide(), !1
                }), websover.reposition(), window.currentPopover = this, websover.active=!0, this
            },
            complete: function(a, b) {
                var c = this, d = JSON.stringify(b);
                window.setTimeout(function() {
                    c.hide(function() {
                        c.trigger(a, window.JSON.parse(d))
                    })
                }, 1)
            },
            hide: function(b, c) {
                typeof this.options != "undefined" && this.options.onClose && this.options.onClose();
                if (websover.$fb) {
                    var d = a(this.options.feedback.ref), e = d.width(), f = d.height(), g = d.offset(), h = a(d[0].ownerDocument).scrollTop(), i = this.options.feedback, j = i.offset ? i.offset.left: 0, k = i.offset ? i.offset.top: 0, l = websover.animate ? 500: 1, m = {
                        width: e,
                        height: f,
                        left: g.left + j,
                        top: g.top + k - h
                    };
                    websover.displayHide(), websover.$fb.show().animate(m, l, function() {
                        websover.$fb.hide(), websover.hide({
                            fromInst: !0
                        })
                    })
                } else 
                    websover.hide({
                        fromInst: !0
                    });
                return typeof b == "function" && b.call(this, c), this
            },
            afterShow: function() {
                var b = document.getElementById("websover_iframe_container");
                if (websover.$fb) {
                    var c = a(websover.currentBorder), d = c.offset(), e = websover.animate ? 300: 1, f = {
                        width: c.width(),
                        height: c.height(),
                        left: d.left,
                        top: d.top
                    };
                    websover.$fb.show().animate(f, e, function() {
                        websover.currentContent.style.visibility = "visible", websover.currentBorder.style.visibility = "visible", websover.$fb.hide(), b.appendChild(websover.currentContent)
                    })
                } else 
                    websover.currentBorder.style.visibility = "visible", websover.currentContent.style.visibility = "visible", b.appendChild(websover.currentContent);
                a(b).find("#websover_iframe").load(function() {
                    a(this).contents().find("body").on("click", ".w-submit:not(.invalid)", function() {
                        a(this).removeClass("green w-btn1-green").addClass("disabled"), a(this).click(function(a) {
                            a.stopPropagation()
                        })
                    })
                })
            },
            trigger: function(a, b) {
                return this.suspendListeners ? this : (this.eventListeners || (this.eventListeners = {
                    defaultListener: function() {}
                }), typeof this.eventListeners[a] == "function" ? this.eventListeners[a](b, a) : this.eventListeners.defaultListener(b, a), this)
            },
            bind: function(b, c) {
                var d = this;
                return this.eventListeners || (this.eventListeners = {
                    defaultListener: function() {}
                }), typeof b == "object" ? a.each(b, function(a, b) {
                    d.eventListeners[a] = b
                }) : typeof b == "string" ? this.eventListeners[b] = c : typeof b == "function" && (this.eventListeners.defaultListener = b), this
            },
            unbind: function() {
                return this.eventListeners = null, this
            }
        }, window.websover = {
            animate: !0,
            extraWidth: 22,
            extraHeight: 68,
            currentShadow: null,
            currentWrap: null,
            currentBorder: null,
            currentPopover: null,
            currentContent: null,
            garbageBin: null,
            imageToLoad: null,
            loadSrcTo: null,
            removeBadElements: function() {
                return 
            },
            getTotalDocSize: function() {
                return Math.max(document.documentElement.scrollHeight || document.body.scrollHeight, document.documentElement.offsetHeight || document.body.offsetHeight)
            },
            displayHide: function() {
                websover.currentWrap.style.display = "none", websover.currentShadow.style.display = "none"
            },
            hide: function(a) {
                var b = a && a.fromInst;
                if (!b && currentPopover && currentPopover.options && currentPopover.options.onClose && currentPopover.options.onClose(!0)===!1)
                    return !1;
                if (!b && currentPopover && currentPopover.options && currentPopover.options.reloadOnClose)
                    return window.location.reload();
                websover.active=!1, websover.removeBadElements(), websover.currentWrap.style.display = "none", websover.currentShadow && (websover.currentShadow.style.display = "none"), websover.currentWrap.removeAttribute("class"), websover.currentContent.tagName === "IFRAME" ? (websover.currentContent.src = "javascript:false;", websover.garbageBin.appendChild(websover.currentContent), websover.garbageBin.innerHTML = "") : websover.currentContent.parentNode && websover.currentContent.parentNode.removeChild(websover.currentContent), websover.top = null, websover.width = null, websover.height = null
            },
            changeIframeDest: function(a) {
                var b;
                if (websover.currentContent.tagName === "IFRAME")
                    b = websover.currentContent;
                else {
                    b = document.createElement("iframe"), b.setAttribute("id", "websover_iframe"), b.setAttribute("rel", this.url), b.setAttribute("name", "popoverIframe"), b.setAttribute("frameborder", "0");
                    var c = websover.currentContent.parentNode;
                    c.removeChild(websover.currentContent), c.appendChild(b), websover.currentContent = b
                }
                websover.changeHeading(a.title), b.src = a.url
            },
            checkScrolling: function(a, b) {
                typeof a.scrolling != "undefined" && (a.scrolling===!0 ? (b.setAttribute("scrolling", "auto"), b.style.overflow = "auto") : a.scrolling===!1 && (b.setAttribute("scrolling", "no"), b.style.overflow = "hidden"))
            },
            changeHeading: function(a) {
                var b = document.getElementById("websover_title");
                b && (b.innerHTML = a)
            },
            resize: function(a) {
                var b = document.getElementById("websover_iframe_container") || websover.currentContent, c = a.width ? a.width: websover.width, d = a.height ? a.height: websover.height, e =- parseInt((c + websover.extraWidth) / 2, 10), f =- parseInt((d + websover.extraHeight) / 2, 10), g = websover.animate ? 500 : 1;
                typeof a.heading == "string" && websover.changeHeading(a.heading), typeof a.before != "undefined" && typeof a.before == "function" && a.before();
                var h = {
                    left: e,
                    width: c
                };
                websover.height < d && websover.top > f && (h.top = f, websover.top = h.top), jQuery(websover.currentBorder).animate(h, g), jQuery(websover.currentPopover).animate({
                    width: c
                }, g, function() {
                    typeof a.after != "undefined" && typeof a.after == "function" && a.after(), websover.checkScrolling(a, b), jQuery(b).fadeIn("fast"), websover.height !== "auto" && websover.reposition()
                }), b.tagName !== "IMG" && jQuery(b).animate({
                    height: d
                }, g), websover.width = c, websover.height = d
            },
            reposition: function() {
                var a = typeof window.innerHeight != "undefined" ? window.innerHeight: document.documentElement.clientHeight, b = typeof window.pageYOffset != "undefined" ? window.pageYOffset: document.documentElement.scrollTop;
                a < websover.height + websover.extraHeight ? (websover.currentWrap.style.position = "absolute", websover.currentWrap.style.top = "0", websover.currentBorder.style.top = b + 20 + "px") : (websover.currentWrap.style.position = "fixed", websover.options.height == "auto" && (websover.currentBorder.style.top =- (websover.currentBorder.offsetHeight / 2) + "px"), websover.currentWrap.style.top = this.options.position && this.options.position.top || "50%")
            }
        }, typeof a != "undefined" && a(document.body).keydown(function(a) {
            websover && websover.options && websover.options.closeOnEsc && a.keyCode === 27 && websover.hide()
        })
    }
    typeof define == "function" && define.amd ? define("internal/common/websover", ["jquery", "translate!webs.util"], a) : a(window.jQuery, function() {
        return "Close"
    })
}(), define("ui/dialog/PopoverView", ["jquery", "underscore", "backbone", "internal/common/websover"], function(a, b) {
    var c = Backbone.View.extend({
        initialize: function() {
            b.extend(this, this.options)
        },
        open: function() {
            if (this._popover)
                this._popover.show();
            else if (this.opts.url)
                this._popover = new Popover(this.opts.url, this.opts), this._popover.show(), this.afterRender();
            else if (this.template) {
                var b = this;
                this.template(this.data(), function(c, d) {
                    b.setElement(a("<div />")), b._popover = new Popover(b.$el.html(d)[0], b.opts), b._popover.show(), b.afterRender()
                })
            }
        },
        close: function() {
            this._popover.hide()
        },
        reposition: function() {
            this._popover.reposition()
        },
        resize: function(a) {
            this._popover.resize(a)
        },
        data: function() {
            return {}
        },
        afterRender: function() {}
    });
    return c
}), define("site/accountSettings/dialog/CredentialView", ["jquery", "underscore", "backbone", "site/accountSettings/StatusView", "dust!site/accountSettings/dialog/templates/credentialView.dust", "translate!webs.util"], function(a, b, c, d, e, f) {
    var g = c.View.extend({
        mode: "view",
        events: {
            "click .edit": "toggle",
            "click .submit": "updateCredential",
            "click .cancel": "cancelUpdate",
            "keypress input": "checkKeyPress"
        },
        initialize: function(a) {
            var c = this;
            b.bindAll(this, "_rendered"), this.notification = a.notification, this.popover = a.popover, this.status = new d, this.data.showValue && (this.data.value = this.model.get(this.data.name)), this.render(), webs.app.on("credential.open", function(a) {
                a.id !== c.id && c.mode === "edit" && (c.cancelUpdate(), c.status.render("working", f("webs.util.update.canceled")))
            }), webs.app.on("confirmation.open", function() {
                c.mode === "edit" && (c.cancelUpdate(), c.status.render("working", f("webs.util.update.canceled")))
            }), webs.app.on("popover.close", function() {
                c.cancelUpdate()
            })
        },
        render: function() {
            e(this.data, this._rendered)
        },
        _rendered: function(b, c) {
            this.$el.html(c), a("#account_settings .w-form-items").append(this.$el), this.$(".control").append(this.status.$el), this.valueContainer = this.$(".value_container"), this.valueSpan = this.valueContainer.find(".value"), this.fieldContainer = this.$(".field_container"), this.input = this.fieldContainer.find("input")
        },
        checkKeyPress: function(a) {
            a.which === 13 && this.updateCredential()
        },
        toggle: function() {
            this.mode === "view" ? (this.mode = "edit", this.status.hide(!0), this.notification.hide(), webs.app.trigger("credential.open", this)) : this.mode = "view", this.valueContainer.toggleClass("hidden"), this.fieldContainer.toggleClass("hidden"), this.input.removeClass("error").eq(0).focus(), this.popover.reposition()
        },
        updateCredential: function() {},
        cancelUpdate: function() {
            this.mode === "edit" && (this.toggle(), this.notification.hide(), this.data.showValue ? this.input.removeClass("error").val(this.valueSpan.text()) : this.input.each(function() {
                a(this).removeClass("error").val("")
            }))
        }
    });
    return g
}), define("site/accountSettings/dialog/EmailView", ["jquery", "underscore", "backbone", "site/accountSettings/dialog/CredentialView", "translate!webs.websid", "site/ErrorLocalizer"], function(a, b, c, d, e, f) {
    var g = d.extend({
        id: "email_container",
        data: {
            name: "email_address",
            label: "Email Address",
            showValue: !0,
            showCancel: !0,
            inputs: [{
                type: "text",
                name: "email"
            }
            ]
        },
        updateCredential: function() {
            var b = this, c = /^([0-9a-zA-Z]+([_.-]?[0-9a-zA-Z]+)*@[0-9a-zA-Z]+[0-9,a-z,A-Z,.,-]*(\.){1}[a-zA-Z]{2,4})+$/;
            this.input.removeClass("error"), this.input.val().length === 0 || this.valueSpan.text() === this.input.val() ? this.cancelUpdate() : c.test(this.input.val()) ? a.ajax({
                type: "PUT",
                url: "/api/accounts/" + this.model.id,
                contentType: "application/json",
                data: JSON.stringify({
                    email_address: this.input.val()
                }),
                success: function() {
                    b.toggle(), b.notification.render("success", e("webs.websid.email.change.confirm"))
                },
                error: function(a) {
                    var c = f.parse(a);
                    b.notification.render("error", e("webs.error." + c)), b.input.addClass("error")
                }
            }) : (this.notification.render("error", e("webs.error.email.invalid")), this.input.addClass("error"))
        }
    });
    return g
}), define("site/accountSettings/dialog/PasswordView", ["jquery", "underscore", "backbone", "site/accountSettings/dialog/CredentialView", "translate!webs.websid", "site/ErrorLocalizer"], function(a, b, c, d, e, f) {
    var g = d.extend({
        id: "password_container",
        data: {
            name: "password",
            label: "Password",
            showValue: !1,
            showCancel: !0,
            inputs: [{
                type: "password",
                name: "currentpassword",
                placeholder: e("webs.util.password.current")
            }, {
                type: "password",
                name: "newpassword",
                placeholder: e("webs.util.password.new")
            }, {
                type: "password",
                name: "confirmpassword",
                placeholder: e("webs.util.password.confirm")
            }
            ]
        },
        updateCredential: function() {
            var b = this;
            this.input.removeClass("error"), this.input.eq(0).val() === "" || this.input.eq(1).val() === "" || this.input.eq(2).val() === "" ? (b.notification.render("error", e("webs.error.fillAllFields")), this.input.each(function() {
                a(this).val() === "" && a(this).addClass("error")
            }).filter(function() {
                return a(this).val().length === 0
            }).eq(0).focus()) : this.input.eq(1).val() !== this.input.eq(2).val() ? (b.notification.render("error", e("webs.error.confirmPassword")), this.input.eq(1).addClass("error").focus(), this.input.eq(2).addClass("error")) : a.ajax({
                type: "PUT",
                url: "/api/accounts/" + this.model.id,
                contentType: "application/json",
                data: JSON.stringify({
                    current_password: this.input.eq(0).val(),
                    password: this.input.eq(1).val(),
                    confirm_password: this.input.eq(2).val()
                }),
                success: function() {
                    b.toggle(), b.notification.hide(), b.status.render("success", e("webs.websid.passwordChanged")), b.input.each(function() {
                        a(this).val("")
                    }), b.timeout = setTimeout(function() {
                        b.status.hide()
                    }, 3e3)
                },
                error: function(a) {
                    var c = f.parse(a);
                    b.notification.render("error", e("webs.error." + c)), b.input.eq(0).focus()
                }
            })
        }
    });
    return g
}), define("site/accountSettings/dialog/PopoverSiteListView", ["jquery", "underscore", "backbone", "instance/site/account", "dust!site/accountSettings/dialog/templates/popoverSiteView.dust", "translate!webs.error", "site/ErrorLocalizer"], function(a, b, c, d, e, f, g) {
    var h = c.View.extend({
        el: "#site_list ul",
        events: {
            "click .delete_account": "confirmAccountDelete",
            "click .delete_account_cancel": "cancelAccountDelete"
        },
        initialize: function(a) {
            var b = this;
            this.popover = a.popover, this.notification = a.notification, this.collection.on("reset", this.render, this), this.collection.on("add", this.renderSite, this), this.$el.on("mousewheel DOMMouseScroll", function(a) {
                var c = 0;
                a.originalEvent.wheelDelta ? (c = a.originalEvent.wheelDelta / 120, window.opera && (c =- c)) : a.originalEvent.detail && (c =- a.originalEvent.detail / 3), (c > 0 && this.scrollTop === 0 || c < 0 && this.scrollTop + b.$el.height() === b.el.scrollHeight) && a.preventDefault()
            }), this.collection.fetch({
                success: function() {
                    b.collection.last_page > b.collection.page && b.$el.on("scroll", function() {
                        b.checkScroll()
                    })
                }
            }), webs.app.on("lastSite.delete", function() {
                b.showAccountDelete()
            })
        },
        render: function() {
            var a = this, c = this.$(".loading");
            return b.each(this.collection.models, function(b) {
                a.renderSite(b, a.collection)
            }), this.deleteAccount = this.$("#delete_account_container"), this.confirmAccountDelete = this.$("#delete_account_confirmation"), this
        },
        showAccountDelete: function() {
            this.deleteAccount.removeClass("hidden")
        },
        confirmAccountDelete: function() {
            this.confirmAccountDelete.removeClass("hidden"), this.deleteAccount.addClass("hidden")
        },
        cancelAccountDelete: function() {
            this.confirmAccountDelete.addClass("hidden"), this.deleteAccount.removeClass("hidden")
        },
        checkScroll: b.debounce(function() {
            var a = this.$el.scrollTop(), b = this.$el.outerHeight();
            a >= b / 3 && this.getMoreSites()
        }, 50),
        renderSite: function(a, b) {
            var c = this, d = this.$(".loading"), e=!1;
            a.poSiteView = new i({
                model: a,
                tagName: "li",
                id: "dropdown_site_" + a.get("id"),
                notification: c.notification,
                currentSite: webs.site.id === a.get("id")
            }), d.before(a.poSiteView.el), a.poSiteView.render(), this.collection.each(function(a) {
                a.poSiteView || (e=!0)
            }), e || d.addClass("hidden"), c.popover.reposition()
        },
        getMoreSites: function() {
            var a = this, b = this.$(".loading");
            this.collection.page < this.collection.last_page ? (b.removeClass("hidden"), this.collection.fetch({
                add: !0,
                data: {
                    page: this.collection.page + 1
                }
            })) : this.$el.off("scroll")
        }
    }), i = c.View.extend({
        tagName: "li",
        events: {
            "click .confirm_user": "confirmChange",
            "click .submit_delete": "deleteSite",
            "click .submit_unsubscribe": "unsubscribeSite",
            "click .cancel": "cancelChange"
        },
        initialize: function(a) {
            var c = this;
            b.bindAll(this, "_rendered"), this.id = this.model.get("id"), this.notification = a.notification, this.currentSite = a.currentSite, webs.app.on("confirmation.open", function(a) {
                a.id !== c.id && c.cancelChange()
            }), webs.app.on("credential.open popover.close", function() {
                c.cancelChange()
            })
        },
        render: function() {
            return e({
                site: this.model.dataForView(),
                props: webs.props,
                canDelete: this.model.get("permission") === "owner"
            }, this._rendered), this
        },
        _rendered: function(a, b) {
            this.$el.html(b), this.confirmContainer = this.$(".confirm"), this.siteInfo = this.$(".site_info"), this.removalNotice = this.$(".removal_notice"), this.input = this.confirmContainer.find("input"), webs.site.id === this.id && this.$el.addClass("current")
        },
        confirmChange: function(a) {
            a.preventDefault(), this.confirmContainer.removeClass("hidden"), this.input.focus(), webs.app.trigger("confirmation.open", this)
        },
        unsubscribeSite: function() {
            var b = this;
            this.$(".loadingSpinner").removeClass("hidden"), this.notification.hide(), a.ajax({
                type: "DELETE",
                url: "/api/websid/" + d.get("id") + "/sites/" + this.model.get("id"),
                success: function() {
                    b.removeSite(), b.notification.hide()
                },
                error: function(a) {
                    var c = g.parse(a);
                    b.notification.render("error", f("webs.error." + c) + " " + f("webs.util.tryAgain"))
                },
                complete: function() {
                    b.$(".loadingSpinner").addClass("hidden")
                }
            })
        },
        removeSite: function() {
            var a = this, b = this.model, c = b.collection, d = c.length;
            c.remove(b), this.confirmContainer.addClass("hidden"), this.$el.addClass("removing"), this.removalNotice.removeClass("hidden"), this.siteInfo.animate({
                left: "+=500px"
            }, 700, function() {
                setTimeout(function() {
                    d === 1 && (a.$el.css("position", "absolute"), webs.app.trigger("lastSite.delete")), a.$el.slideUp(300, function() {
                        a.unbind(), a.remove()
                    })
                }, 3e3)
            })
        },
        deleteSite: function() {
            var b = this, c = {
                siteId: this.model.get("id"),
                password: this.input.val()
            };
            this.$(".loadingSpinner").removeClass("hidden"), this.input.removeClass("error"), this.notification.hide(), a.ajax({
                type: "POST",
                url: "/s/deleteSite",
                data: c,
                success: function() {
                    b.removeSite(), b.notification.hide(), b.currentSite && (webs.currentSiteDeleted=!0), webs.app.trigger("sites.change")
                },
                error: function(a) {
                    var c = g.parse(a);
                    b.notification.render("error", f("webs.error." + c)), b.input.addClass("error").focus()
                },
                complete: function() {
                    b.$(".loadingSpinner").addClass("hidden")
                }
            })
        },
        cancelChange: function() {
            this.confirmContainer.addClass("hidden"), this.input.removeClass("error").val(""), this.notification.hide()
        }
    });
    return h
}), define("site/accountSettings/dialog/PopoverView", ["instance/site/account", "ui/dialog/PopoverView", "site/accountSettings/Sites", "site/accountSettings/NotificationView", "site/accountSettings/dialog/EmailView", "site/accountSettings/dialog/PasswordView", "site/accountSettings/dialog/PopoverSiteListView", "dust!site/accountSettings/dialog/templates/popoverView.dust", "translate!webs.websid"], function(a, b, c, d, e, f, g, h, i) {
    var j = b.extend({
        template: h,
        opts: {
            heading: i("webs.websid.accountSettings"),
            width: 500,
            height: "auto",
            position: {
                top: "40%"
            },
            onClose: function() {
                webs.currentSiteDeleted && (window.location.href = "/sites"), webs.app.trigger("popover.close")
            }
        },
        afterRender: function() {
            this.notification = new d({
                popover: this
            });
            var b = new c({
                memberId: a.get("id"),
                includeSubscriptions: !0
            }), h = new e({
                model: a,
                notification: this.notification,
                popover: this
            }), i = new f({
                model: a,
                notification: this.notification,
                popover: this
            }), j = new g({
                collection: b,
                popover: this,
                notification: this.notification
            })
        }
    });
    return j
}), define("site/accountSettings/dropdown/DropdownView", ["jquery", "underscore", "backbone", "apps/App", "instance/site/account", "instance/site/site", "site/accountSettings/Sites", "site/accountSettings/dropdown/DropdownSiteListView", "site/accountSettings/dialog/PopoverView", "dust!site/accountSettings/dropdown/templates/dropdownView.dust", "translate!webs.websid", "translate!webs.util"], function(a, b, c, d, e, f, g, h, i, j, k) {
    var l = c.View.extend({
        el: "#websid_dropdown_container",
        events: {
            "click a.settings": "showPopover"
        },
        initialize: function() {
            var a = this;
            this.account = e, b.bindAll(this, "render")
        },
        render: function() {
            var a = this;
            a.renderView()
        },
        dataForView: function() {
            var a = {
                props: webs.props,
                site: f.toJSON()
            };
            return this.account ? (a.account = this.account.dataForView(), a.showAllSitesLink = a.account.total_sites > 10, a.logoutUrl = this.account.get("logoutURL") || "/s/logout", a.createSiteLink = "/s/signup") : a.logoutUrl = "/s/logout", a
        },
        renderView: function() {
            var a = this;
            return j(this.dataForView(), function(b, c) {
                a.$el.removeClass("loading"), a.$el.html(c);
                if (a.account)
                    var d = new g({
                        memberId: a.account.get("id"),
                        includeSubscriptions: !1
                    }), e = new h({
                        collection: d
                    })
            }), this
        },
        showPopover: function(a) {
            a.preventDefault(), this.popoverView || (this.popoverView = new i), this.popoverView.open()
        }
    });
    return l
}), define("apps/Dropdown", ["jquery", "site/accountSettings/dropdown/DropdownView", "apps/App"], function(a, b, c) {
    a(function() {
        var a = new b;
        webs = webs || {}, webs.app = webs.app || new c, webs.currentSiteDeleted=!1, a.render()
    })
})
