if (typeof SF === "undefined") {
    SF = {}
}
SF.Popover = function(e) {
    var b = e.attr("id").replace(/-tooltip/, "");
    var a = e.parents("header");
    var d = false;
    function c(g) {
        g.preventDefault();
        g.stopPropagation();
        $(".tooltip:not(." + b + "):visible", a).trigger("popover:close");
        $(".tooltip." + b, e).show();
        $("body").on("click.popover", function(h) {
            e.trigger("popover:close")
        });
        $(document).on("keydown.popover", function(h) {
            if ((h.which || h.keyCode) === 27) {
                h.preventDefault();
                e.trigger("popover:close")
            }
        });
        d = true
    }
    function f(g) {
        if (!$(g.target.parentNode).closest("div").hasClass("tooltip")) {
            g.preventDefault();
            g.stopPropagation()
        }
        a.find(".tooltip:visible").hide();
        $("body").off("click.popover");
        $(document).off("keydown.popover");
        d = false
    }
    e.on("click", "a", function(g) {
        if (d) {
            f(g)
        } else {
            if (!$(g.target).hasClass("not-available")) {
                c(g)
            }
        }
    });
    e.on("popover:close", f)
};
SF.SimplifiedCookieNotice = function() {
    return {
        overlay: null,
        banner: null,
        body: null,
        msg: "By using the SourceForge site, you agree to our use of cookies.",
        win: null,
        cookieKey: "tsn",
        init: function() {
            this.win = $(window);
            this.body = $("body");
            var a = $.cookie(this.cookieKey);
            if (Number(a) !== 1) {
                this._setupBanner();
                this._setupListeners();
                this.show()
            }
            return this
        },
        _setupListeners: function() {
            var a = this;
            this.body.on("click", ".truste-cookie-accept", function(b) {
                b.preventDefault();
                $.cookie(a.cookieKey, 1, {
                    path: "/",
                    expires: 390
                });
                a.hide()
            });
            this.body.on("click", ".truste-cookie-denied", function(b) {
                b.preventDefault();
                if (truste.eu && truste.eu.clickListener) {
                    truste.eu.clickListener()
                }
            });
            this.win.resize(function(b) {
                if (a.win.width() > 1075) {
                    a.banner.width(a.win.width() - 80)
                }
            })
        },
        _setupBanner: function() {
            this._createBanner();
            this._populateBanner()
        },
        _createBanner: function() {
            this.banner = $('<div class="truste-sf-banner" />')
        },
        _populateBanner: function() {
            this.body.prepend(this.banner);
            var a = '<div class="truste-sf-inner"><h2>' + this.msg + '</h2><div id="truste-consent-buttons" class="button-container"><button id="truste-consent-button" class="truste-cookie-accept">I consent to cookies</button><button class="truste-cookie-denied">I want more information</button</div></div>';
            this.banner.html(a)
        },
        show: function() {
            this.banner.width(this.win.width() - 80)
        },
        hide: function() {
            this.banner.animate({
                height: "toggle",
                opacity: "toggle"
            }, 1000, "linear", function() {
                $(this).remove()
            })
        }
    }
};
$(function() {
    var b = $("#updater-tooltip");
    if (b.length) {
        if (b.hasClass("fetch")) {
            $.ajax({
                url: "/user/updates/find",
                global: false,
                success: function(c) {
                    if (c.length) {
                        b.hide().html(c).show();
                        SF.Popover(b)
                    }
                }
            })
        } else {
            SF.Popover(b)
        }
    }
    var a = $("#account-tooltip");
    if (a.length) {
        SF.Popover(a)
    }
});
(function($) {
    $.extend({
        metadata: {
            defaults: {
                type: "class",
                name: "metadata",
                cre: /({.*})/,
                single: "metadata"
            },
            setType: function(type, name) {
                this.defaults.type = type;
                this.defaults.name = name
            },
            get: function(elem, opts) {
                var settings = $.extend({}, this.defaults, opts);
                if (!settings.single.length) {
                    settings.single = "metadata"
                }
                var data = $.data(elem, settings.single);
                if (data) {
                    return data
                }
                data = "{}";
                var getData = function(data) {
                    if (typeof data != "string") {
                        return data
                    }
                    if (data.indexOf("{") < 0) {
                        data = eval("(" + data + ")")
                    }
                };
                var getObject = function(data) {
                    if (typeof data != "string") {
                        return data
                    }
                    data = eval("(" + data + ")");
                    return data
                };
                if (settings.type == "html5") {
                    var object = {};
                    $(elem.attributes).each(function() {
                        var name = this.nodeName;
                        if (name.match(/^data-/)) {
                            name = name.replace(/^data-/, "")
                        } else {
                            return true
                        }
                        object[name] = getObject(this.nodeValue)
                    })
                } else {
                    if (settings.type == "class") {
                        var m = settings.cre.exec(elem.className);
                        if (m) {
                            data = m[1]
                        }
                    } else {
                        if (settings.type == "elem") {
                            if (!elem.getElementsByTagName) {
                                return 
                            }
                            var e = elem.getElementsByTagName(settings.name);
                            if (e.length) {
                                data = $.trim(e[0].innerHTML)
                            }
                        } else {
                            if (elem.getAttribute != undefined) {
                                var attr = elem.getAttribute(settings.name);
                                if (attr) {
                                    data = attr
                                }
                            }
                        }
                    }
                    object = getObject(data.indexOf("{") < 0 ? "{" + data + "}" : data)
                }
                $.data(elem, settings.single, object);
                return object
            }
        }
    });
    $.fn.metadata = function(opts) {
        return $.metadata.get(this[0], opts)
    }
})(jQuery); /*! jQuery UI - v1.11.1 - 2014-08-24
* http://jqueryui.com
* Includes: core.js, widget.js, mouse.js, position.js, draggable.js, droppable.js, resizable.js, selectable.js, sortable.js, accordion.js, autocomplete.js, button.js, datepicker.js, dialog.js, menu.js, progressbar.js, selectmenu.js, slider.js, spinner.js, tabs.js, tooltip.js, effect.js, effect-blind.js, effect-bounce.js, effect-clip.js, effect-drop.js, effect-explode.js, effect-fade.js, effect-fold.js, effect-highlight.js, effect-puff.js, effect-pulsate.js, effect-scale.js, effect-shake.js, effect-size.js, effect-slide.js, effect-transfer.js
* Copyright 2014 jQuery Foundation and other contributors; Licensed MIT */

(function(e) {
    "function" == typeof define && define.amd ? define(["jquery"], e) : e(jQuery)
})(function(e) {
    function t(t, s) {
        var a, n, r, o = t.nodeName.toLowerCase();
        return "area" === o ? (a = t.parentNode, n = a.name, t.href && n && "map" === a.nodeName.toLowerCase() ? (r = e("img[usemap='#" + n + "']")[0], !!r && i(r)) : !1) : (/input|select|textarea|button|object/.test(o)?!t.disabled : "a" === o ? t.href || s : s) && i(t)
    }
    function i(t) {
        return e.expr.filters.visible(t)&&!e(t).parents().addBack().filter(function() {
            return "hidden" === e.css(this, "visibility")
        }).length
    }
    function s(e) {
        for (var t, i; e.length && e[0] !== document;) {
            if (t = e.css("position"), ("absolute" === t || "relative" === t || "fixed" === t) && (i = parseInt(e.css("zIndex"), 10), !isNaN(i) && 0 !== i))
                return i;
            e = e.parent()
        }
        return 0
    }
    function a() {
        this._curInst = null, this._keyEvent=!1, this._disabledInputs = [], this._datepickerShowing=!1, this._inDialog=!1, this._mainDivId = "ui-datepicker-div", this._inlineClass = "ui-datepicker-inline", this._appendClass = "ui-datepicker-append", this._triggerClass = "ui-datepicker-trigger", this._dialogClass = "ui-datepicker-dialog", this._disableClass = "ui-datepicker-disabled", this._unselectableClass = "ui-datepicker-unselectable", this._currentClass = "ui-datepicker-current-day", this._dayOverClass = "ui-datepicker-days-cell-over", this.regional = [], this.regional[""] = {
            closeText: "Done",
            prevText: "Prev",
            nextText: "Next",
            currentText: "Today",
            monthNames: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            monthNamesShort: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            dayNames: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            dayNamesShort: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
            dayNamesMin: ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"],
            weekHeader: "Wk",
            dateFormat: "mm/dd/yy",
            firstDay: 0,
            isRTL: !1,
            showMonthAfterYear: !1,
            yearSuffix: ""
        }, this._defaults = {
            showOn: "focus",
            showAnim: "fadeIn",
            showOptions: {},
            defaultDate: null,
            appendText: "",
            buttonText: "...",
            buttonImage: "",
            buttonImageOnly: !1,
            hideIfNoPrevNext: !1,
            navigationAsDateFormat: !1,
            gotoCurrent: !1,
            changeMonth: !1,
            changeYear: !1,
            yearRange: "c-10:c+10",
            showOtherMonths: !1,
            selectOtherMonths: !1,
            showWeek: !1,
            calculateWeek: this.iso8601Week,
            shortYearCutoff: "+10",
            minDate: null,
            maxDate: null,
            duration: "fast",
            beforeShowDay: null,
            beforeShow: null,
            onSelect: null,
            onChangeMonthYear: null,
            onClose: null,
            numberOfMonths: 1,
            showCurrentAtPos: 0,
            stepMonths: 1,
            stepBigMonths: 12,
            altField: "",
            altFormat: "",
            constrainInput: !0,
            showButtonPanel: !1,
            autoSize: !1,
            disabled: !1
        }, e.extend(this._defaults, this.regional[""]), this.regional.en = e.extend(!0, {}, this.regional[""]), this.regional["en-US"] = e.extend(!0, {}, this.regional.en), this.dpDiv = n(e("<div id='" + this._mainDivId + "' class='ui-datepicker ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>"))
    }
    function n(t) {
        var i = "button, .ui-datepicker-prev, .ui-datepicker-next, .ui-datepicker-calendar td a";
        return t.delegate(i, "mouseout", function() {
            e(this).removeClass("ui-state-hover"), - 1 !== this.className.indexOf("ui-datepicker-prev") && e(this).removeClass("ui-datepicker-prev-hover"), - 1 !== this.className.indexOf("ui-datepicker-next") && e(this).removeClass("ui-datepicker-next-hover")
        }).delegate(i, "mouseover", r)
    }
    function r() {
        e.datepicker._isDisabledDatepicker(v.inline ? v.dpDiv.parent()[0] : v.input[0]) || (e(this).parents(".ui-datepicker-calendar").find("a").removeClass("ui-state-hover"), e(this).addClass("ui-state-hover"), - 1 !== this.className.indexOf("ui-datepicker-prev") && e(this).addClass("ui-datepicker-prev-hover"), - 1 !== this.className.indexOf("ui-datepicker-next") && e(this).addClass("ui-datepicker-next-hover"))
    }
    function o(t, i) {
        e.extend(t, i);
        for (var s in i)
            null == i[s] && (t[s] = i[s]);
        return t
    }
    function h(e) {
        return function() {
            var t = this.element.val();
            e.apply(this, arguments), this._refresh(), t !== this.element.val() && this._trigger("change")
        }
    }
    e.ui = e.ui || {}, e.extend(e.ui, {
        version: "1.11.1",
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
            PAGE_DOWN: 34,
            PAGE_UP: 33,
            PERIOD: 190,
            RIGHT: 39,
            SPACE: 32,
            TAB: 9,
            UP: 38
        }
    }), e.fn.extend({
        scrollParent: function(t) {
            var i = this.css("position"), s = "absolute" === i, a = t ? /(auto|scroll|hidden)/: /(auto|scroll)/, n = this.parents().filter(function() {
                var t = e(this);
                return s && "static" === t.css("position")?!1 : a.test(t.css("overflow") + t.css("overflow-y") + t.css("overflow-x"))
            }).eq(0);
            return "fixed" !== i && n.length ? n : e(this[0].ownerDocument || document)
        },
        uniqueId: function() {
            var e = 0;
            return function() {
                return this.each(function() {
                    this.id || (this.id = "ui-id-" + ++e)
                })
            }
        }(),
        removeUniqueId: function() {
            return this.each(function() {
                /^ui-id-\d+$/.test(this.id) && e(this).removeAttr("id")
            })
        }
    }), e.extend(e.expr[":"], {
        data: e.expr.createPseudo ? e.expr.createPseudo(function(t) {
            return function(i) {
                return !!e.data(i, t)
            }
        }): function(t, i, s) {
            return !!e.data(t, s[3])
        },
        focusable: function(i) {
            return t(i, !isNaN(e.attr(i, "tabindex")))
        },
        tabbable: function(i) {
            var s = e.attr(i, "tabindex"), a = isNaN(s);
            return (a || s >= 0) && t(i, !a)
        }
    }), e("<a>").outerWidth(1).jquery || e.each(["Width", "Height"], function(t, i) {
        function s(t, i, s, n) {
            return e.each(a, function() {
                i -= parseFloat(e.css(t, "padding" + this)) || 0, s && (i -= parseFloat(e.css(t, "border" + this + "Width")) || 0), n && (i -= parseFloat(e.css(t, "margin" + this)) || 0)
            }), i
        }
        var a = "Width" === i ? ["Left", "Right"]: ["Top", "Bottom"], n = i.toLowerCase(), r = {
            innerWidth: e.fn.innerWidth,
            innerHeight: e.fn.innerHeight,
            outerWidth: e.fn.outerWidth,
            outerHeight: e.fn.outerHeight
        };
        e.fn["inner" + i] = function(t) {
            return void 0 === t ? r["inner" + i].call(this) : this.each(function() {
                e(this).css(n, s(this, t) + "px")
            })
        }, e.fn["outer" + i] = function(t, a) {
            return "number" != typeof t ? r["outer" + i].call(this, t) : this.each(function() {
                e(this).css(n, s(this, t, !0, a) + "px")
            })
        }
    }), e.fn.addBack || (e.fn.addBack = function(e) {
        return this.add(null == e ? this.prevObject : this.prevObject.filter(e))
    }), e("<a>").data("a-b", "a").removeData("a-b").data("a-b") && (e.fn.removeData = function(t) {
        return function(i) {
            return arguments.length ? t.call(this, e.camelCase(i)) : t.call(this)
        }
    }(e.fn.removeData)), e.ui.ie=!!/msie [\w.]+/.exec(navigator.userAgent.toLowerCase()), e.fn.extend({
        focus: function(t) {
            return function(i, s) {
                return "number" == typeof i ? this.each(function() {
                    var t = this;
                    setTimeout(function() {
                        e(t).focus(), s && s.call(t)
                    }, i)
                }) : t.apply(this, arguments)
            }
        }(e.fn.focus),
        disableSelection: function() {
            var e = "onselectstart"in document.createElement("div") ? "selectstart": "mousedown";
            return function() {
                return this.bind(e + ".ui-disableSelection", function(e) {
                    e.preventDefault()
                })
            }
        }(),
        enableSelection: function() {
            return this.unbind(".ui-disableSelection")
        },
        zIndex: function(t) {
            if (void 0 !== t)
                return this.css("zIndex", t);
            if (this.length)
                for (var i, s, a = e(this[0]); a.length && a[0] !== document;) {
                    if (i = a.css("position"), ("absolute" === i || "relative" === i || "fixed" === i) && (s = parseInt(a.css("zIndex"), 10), !isNaN(s) && 0 !== s))
                        return s;
                        a = a.parent()
                }
            return 0
        }
    }), e.ui.plugin = {
        add: function(t, i, s) {
            var a, n = e.ui[t].prototype;
            for (a in s)
                n.plugins[a] = n.plugins[a] || [], n.plugins[a].push([i, s[a]])
        },
        call: function(e, t, i, s) {
            var a, n = e.plugins[t];
            if (n && (s || e.element[0].parentNode && 11 !== e.element[0].parentNode.nodeType))
                for (a = 0; n.length > a; a++)
                    e.options[n[a][0]] && n[a][1].apply(e.element, i)
        }
    };
    var l = 0, u = Array.prototype.slice;
    e.cleanData = function(t) {
        return function(i) {
            var s, a, n;
            for (n = 0; null != (a = i[n]); n++)
                try {
                    s = e._data(a, "events"), s && s.remove && e(a).triggerHandler("remove")
            } catch (r) {}
            t(i)
        }
    }(e.cleanData), e.widget = function(t, i, s) {
        var a, n, r, o, h = {}, l = t.split(".")[0];
        return t = t.split(".")[1], a = l + "-" + t, s || (s = i, i = e.Widget), e.expr[":"][a.toLowerCase()] = function(t) {
            return !!e.data(t, a)
        }, e[l] = e[l] || {}, n = e[l][t], r = e[l][t] = function(e, t) {
            return this._createWidget ? (arguments.length && this._createWidget(e, t), void 0) : new r(e, t)
        }, e.extend(r, n, {
            version: s.version,
            _proto: e.extend({}, s),
            _childConstructors: []
        }), o = new i, o.options = e.widget.extend({}, o.options), e.each(s, function(t, s) {
            return e.isFunction(s) ? (h[t] = function() {
                var e = function() {
                    return i.prototype[t].apply(this, arguments)
                }, a = function(e) {
                    return i.prototype[t].apply(this, e)
                };
                return function() {
                    var t, i = this._super, n = this._superApply;
                    return this._super = e, this._superApply = a, t = s.apply(this, arguments), this._super = i, this._superApply = n, t
                }
            }(), void 0) : (h[t] = s, void 0)
        }), r.prototype = e.widget.extend(o, {
            widgetEventPrefix: n ? o.widgetEventPrefix || t: t
        }, h, {
            constructor: r,
            namespace: l,
            widgetName: t,
            widgetFullName: a
        }), n ? (e.each(n._childConstructors, function(t, i) {
            var s = i.prototype;
            e.widget(s.namespace + "." + s.widgetName, r, i._proto)
        }), delete n._childConstructors) : i._childConstructors.push(r), e.widget.bridge(t, r), r
    }, e.widget.extend = function(t) {
        for (var i, s, a = u.call(arguments, 1), n = 0, r = a.length; r > n; n++)
            for (i in a[n])
                s = a[n][i], a[n].hasOwnProperty(i) && void 0 !== s && (t[i] = e.isPlainObject(s) ? e.isPlainObject(t[i]) ? e.widget.extend({}, t[i], s) : e.widget.extend({}, s) : s);
        return t
    }, e.widget.bridge = function(t, i) {
        var s = i.prototype.widgetFullName || t;
        e.fn[t] = function(a) {
            var n = "string" == typeof a, r = u.call(arguments, 1), o = this;
            return a=!n && r.length ? e.widget.extend.apply(null, [a].concat(r)) : a, n ? this.each(function() {
                var i, n = e.data(this, s);
                return "instance" === a ? (o = n, !1) : n ? e.isFunction(n[a]) && "_" !== a.charAt(0) ? (i = n[a].apply(n, r), i !== n && void 0 !== i ? (o = i && i.jquery ? o.pushStack(i.get()) : i, !1) : void 0) : e.error("no such method '" + a + "' for " + t + " widget instance") : e.error("cannot call methods on " + t + " prior to initialization; " + "attempted to call method '" + a + "'")
            }) : this.each(function() {
                var t = e.data(this, s);
                t ? (t.option(a || {}), t._init && t._init()) : e.data(this, s, new i(a, this))
            }), o
        }
    }, e.Widget = function() {}, e.Widget._childConstructors = [], e.Widget.prototype = {
        widgetName: "widget",
        widgetEventPrefix: "",
        defaultElement: "<div>",
        options: {
            disabled: !1,
            create: null
        },
        _createWidget: function(t, i) {
            i = e(i || this.defaultElement || this)[0], this.element = e(i), this.uuid = l++, this.eventNamespace = "." + this.widgetName + this.uuid, this.options = e.widget.extend({}, this.options, this._getCreateOptions(), t), this.bindings = e(), this.hoverable = e(), this.focusable = e(), i !== this && (e.data(i, this.widgetFullName, this), this._on(!0, this.element, {
                remove: function(e) {
                    e.target === i && this.destroy()
                }
            }), this.document = e(i.style ? i.ownerDocument : i.document || i), this.window = e(this.document[0].defaultView || this.document[0].parentWindow)), this._create(), this._trigger("create", null, this._getCreateEventData()), this._init()
        },
        _getCreateOptions: e.noop,
        _getCreateEventData: e.noop,
        _create: e.noop,
        _init: e.noop,
        destroy: function() {
            this._destroy(), this.element.unbind(this.eventNamespace).removeData(this.widgetFullName).removeData(e.camelCase(this.widgetFullName)), this.widget().unbind(this.eventNamespace).removeAttr("aria-disabled").removeClass(this.widgetFullName + "-disabled " + "ui-state-disabled"), this.bindings.unbind(this.eventNamespace), this.hoverable.removeClass("ui-state-hover"), this.focusable.removeClass("ui-state-focus")
        },
        _destroy: e.noop,
        widget: function() {
            return this.element
        },
        option: function(t, i) {
            var s, a, n, r = t;
            if (0 === arguments.length)
                return e.widget.extend({}, this.options);
            if ("string" == typeof t)
                if (r = {}, s = t.split("."), t = s.shift(), s.length) {
                    for (a = r[t] = e.widget.extend({}, this.options[t]), n = 0; s.length - 1 > n; n++)
                        a[s[n]] = a[s[n]] || {}, a = a[s[n]];
                        if (t = s.pop(), 1 === arguments.length)
                            return void 0 === a[t] ? null : a[t];
                            a[t] = i
                } else {
                    if (1 === arguments.length)
                        return void 0 === this.options[t] ? null : this.options[t];
                        r[t] = i
                }
            return this._setOptions(r), this
        },
        _setOptions: function(e) {
            var t;
            for (t in e)
                this._setOption(t, e[t]);
            return this
        },
        _setOption: function(e, t) {
            return this.options[e] = t, "disabled" === e && (this.widget().toggleClass(this.widgetFullName + "-disabled", !!t), t && (this.hoverable.removeClass("ui-state-hover"), this.focusable.removeClass("ui-state-focus"))), this
        },
        enable: function() {
            return this._setOptions({
                disabled: !1
            })
        },
        disable: function() {
            return this._setOptions({
                disabled: !0
            })
        },
        _on: function(t, i, s) {
            var a, n = this;
            "boolean" != typeof t && (s = i, i = t, t=!1), s ? (i = a = e(i), this.bindings = this.bindings.add(i)) : (s = i, i = this.element, a = this.widget()), e.each(s, function(s, r) {
                function o() {
                    return t || n.options.disabled!==!0&&!e(this).hasClass("ui-state-disabled") ? ("string" == typeof r ? n[r] : r).apply(n, arguments) : void 0
                }
                "string" != typeof r && (o.guid = r.guid = r.guid || o.guid || e.guid++);
                var h = s.match(/^([\w:-]*)\s*(.*)$/), l = h[1] + n.eventNamespace, u = h[2];
                u ? a.delegate(u, l, o) : i.bind(l, o)
            })
        },
        _off: function(e, t) {
            t = (t || "").split(" ").join(this.eventNamespace + " ") + this.eventNamespace, e.unbind(t).undelegate(t)
        },
        _delay: function(e, t) {
            function i() {
                return ("string" == typeof e ? s[e] : e).apply(s, arguments)
            }
            var s = this;
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
        _trigger: function(t, i, s) {
            var a, n, r = this.options[t];
            if (s = s || {}, i = e.Event(i), i.type = (t === this.widgetEventPrefix ? t : this.widgetEventPrefix + t).toLowerCase(), i.target = this.element[0], n = i.originalEvent)
                for (a in n)
                    a in i || (i[a] = n[a]);
            return this.element.trigger(i, s), !(e.isFunction(r) && r.apply(this.element[0], [i].concat(s))===!1 || i.isDefaultPrevented())
        }
    }, e.each({
        show: "fadeIn",
        hide: "fadeOut"
    }, function(t, i) {
        e.Widget.prototype["_" + t] = function(s, a, n) {
            "string" == typeof a && (a = {
                effect: a
            });
            var r, o = a ? a===!0 || "number" == typeof a ? i: a.effect || i: t;
            a = a || {}, "number" == typeof a && (a = {
                duration: a
            }), r=!e.isEmptyObject(a), a.complete = n, a.delay && s.delay(a.delay), r && e.effects && e.effects.effect[o] ? s[t](a) : o !== t && s[o] ? s[o](a.duration, a.easing, n) : s.queue(function(i) {
                e(this)[t](), n && n.call(s[0]), i()
            })
        }
    }), e.widget;
    var d=!1;
    e(document).mouseup(function() {
        d=!1
    }), e.widget("ui.mouse", {
        version: "1.11.1",
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
            this.element.unbind("." + this.widgetName), this._mouseMoveDelegate && this.document.unbind("mousemove." + this.widgetName, this._mouseMoveDelegate).unbind("mouseup." + this.widgetName, this._mouseUpDelegate)
        },
        _mouseDown: function(t) {
            if (!d) {
                this._mouseStarted && this._mouseUp(t), this._mouseDownEvent = t;
                var i = this, s = 1 === t.which, a = "string" == typeof this.options.cancel && t.target.nodeName ? e(t.target).closest(this.options.cancel).length: !1;
                return s&&!a && this._mouseCapture(t) ? (this.mouseDelayMet=!this.options.delay, this.mouseDelayMet || (this._mouseDelayTimer = setTimeout(function() {
                    i.mouseDelayMet=!0
                }, this.options.delay)), this._mouseDistanceMet(t) && this._mouseDelayMet(t) && (this._mouseStarted = this._mouseStart(t)!==!1, !this._mouseStarted) ? (t.preventDefault(), !0) : (!0 === e.data(t.target, this.widgetName + ".preventClickEvent") && e.removeData(t.target, this.widgetName + ".preventClickEvent"), this._mouseMoveDelegate = function(e) {
                    return i._mouseMove(e)
                }, this._mouseUpDelegate = function(e) {
                    return i._mouseUp(e)
                }, this.document.bind("mousemove." + this.widgetName, this._mouseMoveDelegate).bind("mouseup." + this.widgetName, this._mouseUpDelegate), t.preventDefault(), d=!0, !0)) : !0
            }
        },
        _mouseMove: function(t) {
            return e.ui.ie && (!document.documentMode || 9 > document.documentMode)&&!t.button ? this._mouseUp(t) : t.which ? this._mouseStarted ? (this._mouseDrag(t), t.preventDefault()) : (this._mouseDistanceMet(t) && this._mouseDelayMet(t) && (this._mouseStarted = this._mouseStart(this._mouseDownEvent, t)!==!1, this._mouseStarted ? this._mouseDrag(t) : this._mouseUp(t)), !this._mouseStarted) : this._mouseUp(t)
        },
        _mouseUp: function(t) {
            return this.document.unbind("mousemove." + this.widgetName, this._mouseMoveDelegate).unbind("mouseup." + this.widgetName, this._mouseUpDelegate), this._mouseStarted && (this._mouseStarted=!1, t.target === this._mouseDownEvent.target && e.data(t.target, this.widgetName + ".preventClickEvent", !0), this._mouseStop(t)), d=!1, !1
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
    }), function() {
        function t(e, t, i) {
            return [parseFloat(e[0]) * (p.test(e[0]) ? t / 100 : 1), parseFloat(e[1]) * (p.test(e[1]) ? i / 100 : 1)]
        }
        function i(t, i) {
            return parseInt(e.css(t, i), 10) || 0
        }
        function s(t) {
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
        var a, n, r = Math.max, o = Math.abs, h = Math.round, l = /left|center|right/, u = /top|center|bottom/, d = /[\+\-]\d+(\.[\d]+)?%?/, c = /^\w+/, p = /%$/, f = e.fn.position;
        e.position = {
            scrollbarWidth: function() {
                if (void 0 !== a)
                    return a;
                var t, i, s = e("<div style='display:block;position:absolute;width:50px;height:50px;overflow:hidden;'><div style='height:100px;width:auto;'></div></div>"), n = s.children()[0];
                return e("body").append(s), t = n.offsetWidth, s.css("overflow", "scroll"), i = n.offsetWidth, t === i && (i = s[0].clientWidth), s.remove(), a = t - i
            },
            getScrollInfo: function(t) {
                var i = t.isWindow || t.isDocument ? "": t.element.css("overflow-x"), s = t.isWindow || t.isDocument ? "": t.element.css("overflow-y"), a = "scroll" === i || "auto" === i && t.width < t.element[0].scrollWidth, n = "scroll" === s || "auto" === s && t.height < t.element[0].scrollHeight;
                return {
                    width: n ? e.position.scrollbarWidth(): 0,
                    height: a ? e.position.scrollbarWidth(): 0
                }
            },
            getWithinInfo: function(t) {
                var i = e(t || window), s = e.isWindow(i[0]), a=!!i[0] && 9 === i[0].nodeType;
                return {
                    element: i,
                    isWindow: s,
                    isDocument: a,
                    offset: i.offset() || {
                        left: 0,
                        top: 0
                    },
                    scrollLeft: i.scrollLeft(),
                    scrollTop: i.scrollTop(),
                    width: s || a ? i.width(): i.outerWidth(),
                    height: s || a ? i.height(): i.outerHeight()
                }
            }
        }, e.fn.position = function(a) {
            if (!a ||!a.of)
                return f.apply(this, arguments);
            a = e.extend({}, a);
            var p, m, g, v, y, b, _ = e(a.of), x = e.position.getWithinInfo(a.within), k = e.position.getScrollInfo(x), w = (a.collision || "flip").split(" "), T = {};
            return b = s(_), _[0].preventDefault && (a.at = "left top"), m = b.width, g = b.height, v = b.offset, y = e.extend({}, v), e.each(["my", "at"], function() {
                var e, t, i = (a[this] || "").split(" ");
                1 === i.length && (i = l.test(i[0]) ? i.concat(["center"]) : u.test(i[0]) ? ["center"].concat(i) : ["center", "center"]), i[0] = l.test(i[0]) ? i[0] : "center", i[1] = u.test(i[1]) ? i[1] : "center", e = d.exec(i[0]), t = d.exec(i[1]), T[this] = [e ? e[0]: 0, t ? t[0]: 0], a[this] = [c.exec(i[0])[0], c.exec(i[1])[0]]
            }), 1 === w.length && (w[1] = w[0]), "right" === a.at[0] ? y.left += m : "center" === a.at[0] && (y.left += m / 2), "bottom" === a.at[1] ? y.top += g : "center" === a.at[1] && (y.top += g / 2), p = t(T.at, m, g), y.left += p[0], y.top += p[1], this.each(function() {
                var s, l, u = e(this), d = u.outerWidth(), c = u.outerHeight(), f = i(this, "marginLeft"), b = i(this, "marginTop"), S = d + f + i(this, "marginRight") + k.width, D = c + b + i(this, "marginBottom") + k.height, M = e.extend({}, y), N = t(T.my, u.outerWidth(), u.outerHeight());
                "right" === a.my[0] ? M.left -= d : "center" === a.my[0] && (M.left -= d / 2), "bottom" === a.my[1] ? M.top -= c : "center" === a.my[1] && (M.top -= c / 2), M.left += N[0], M.top += N[1], n || (M.left = h(M.left), M.top = h(M.top)), s = {
                    marginLeft: f,
                    marginTop: b
                }, e.each(["left", "top"], function(t, i) {
                    e.ui.position[w[t]] && e.ui.position[w[t]][i](M, {
                        targetWidth: m,
                        targetHeight: g,
                        elemWidth: d,
                        elemHeight: c,
                        collisionPosition: s,
                        collisionWidth: S,
                        collisionHeight: D,
                        offset: [p[0] + N[0], p[1] + N[1]],
                        my: a.my,
                        at: a.at,
                        within: x,
                        elem: u
                    })
                }), a.using && (l = function(e) {
                    var t = v.left - M.left, i = t + m - d, s = v.top - M.top, n = s + g - c, h = {
                        target: {
                            element: _,
                            left: v.left,
                            top: v.top,
                            width: m,
                            height: g
                        },
                        element: {
                            element: u,
                            left: M.left,
                            top: M.top,
                            width: d,
                            height: c
                        },
                        horizontal: 0 > i ? "left": t > 0 ? "right": "center",
                        vertical: 0 > n ? "top": s > 0 ? "bottom": "middle"
                    };
                    d > m && m > o(t + i) && (h.horizontal = "center"), c > g && g > o(s + n) && (h.vertical = "middle"), h.important = r(o(t), o(i)) > r(o(s), o(n)) ? "horizontal" : "vertical", a.using.call(this, e, h)
                }), u.offset(e.extend(M, {
                    using: l
                }))
            })
        }, e.ui.position = {
            fit: {
                left: function(e, t) {
                    var i, s = t.within, a = s.isWindow ? s.scrollLeft: s.offset.left, n = s.width, o = e.left - t.collisionPosition.marginLeft, h = a - o, l = o + t.collisionWidth - n - a;
                    t.collisionWidth > n ? h > 0 && 0 >= l ? (i = e.left + h + t.collisionWidth - n - a, e.left += h - i) : e.left = l > 0 && 0 >= h ? a : h > l ? a + n - t.collisionWidth : a : h > 0 ? e.left += h : l > 0 ? e.left -= l : e.left = r(e.left - o, e.left)
                },
                top: function(e, t) {
                    var i, s = t.within, a = s.isWindow ? s.scrollTop: s.offset.top, n = t.within.height, o = e.top - t.collisionPosition.marginTop, h = a - o, l = o + t.collisionHeight - n - a;
                    t.collisionHeight > n ? h > 0 && 0 >= l ? (i = e.top + h + t.collisionHeight - n - a, e.top += h - i) : e.top = l > 0 && 0 >= h ? a : h > l ? a + n - t.collisionHeight : a : h > 0 ? e.top += h : l > 0 ? e.top -= l : e.top = r(e.top - o, e.top)
                }
            },
            flip: {
                left: function(e, t) {
                    var i, s, a = t.within, n = a.offset.left + a.scrollLeft, r = a.width, h = a.isWindow ? a.scrollLeft: a.offset.left, l = e.left - t.collisionPosition.marginLeft, u = l - h, d = l + t.collisionWidth - r - h, c = "left" === t.my[0]?-t.elemWidth : "right" === t.my[0] ? t.elemWidth : 0, p = "left" === t.at[0] ? t.targetWidth : "right" === t.at[0]?-t.targetWidth : 0, f =- 2 * t.offset[0];
                    0 > u ? (i = e.left + c + p + f + t.collisionWidth - r - n, (0 > i || o(u) > i) && (e.left += c + p + f)) : d > 0 && (s = e.left - t.collisionPosition.marginLeft + c + p + f - h, (s > 0 || d > o(s)) && (e.left += c + p + f))
                },
                top: function(e, t) {
                    var i, s, a = t.within, n = a.offset.top + a.scrollTop, r = a.height, h = a.isWindow ? a.scrollTop: a.offset.top, l = e.top - t.collisionPosition.marginTop, u = l - h, d = l + t.collisionHeight - r - h, c = "top" === t.my[1], p = c?-t.elemHeight : "bottom" === t.my[1] ? t.elemHeight : 0, f = "top" === t.at[1] ? t.targetHeight : "bottom" === t.at[1]?-t.targetHeight : 0, m =- 2 * t.offset[1];
                    0 > u ? (s = e.top + p + f + m + t.collisionHeight - r - n, e.top + p + f + m > u && (0 > s || o(u) > s) && (e.top += p + f + m)) : d > 0 && (i = e.top - t.collisionPosition.marginTop + p + f + m - h, e.top + p + f + m > d && (i > 0 || d > o(i)) && (e.top += p + f + m))
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
            var t, i, s, a, r, o = document.getElementsByTagName("body")[0], h = document.createElement("div");
            t = document.createElement(o ? "div" : "body"), s = {
                visibility: "hidden",
                width: 0,
                height: 0,
                border: 0,
                margin: 0,
                background: "none"
            }, o && e.extend(s, {
                position: "absolute",
                left: "-1000px",
                top: "-1000px"
            });
            for (r in s)
                t.style[r] = s[r];
            t.appendChild(h), i = o || document.documentElement, i.insertBefore(t, i.firstChild), h.style.cssText = "position: absolute; left: 10.7432222px;", a = e(h).offset().left, n = a > 10 && 11 > a, t.innerHTML = "", i.removeChild(t)
        }()
    }(), e.ui.position, e.widget("ui.draggable", e.ui.mouse, {
        version: "1.11.1",
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
            "original" !== this.options.helper || /^(?:r|a|f)/.test(this.element.css("position")) || (this.element[0].style.position = "relative"), this.options.addClasses && this.element.addClass("ui-draggable"), this.options.disabled && this.element.addClass("ui-draggable-disabled"), this._setHandleClassName(), this._mouseInit()
        },
        _setOption: function(e, t) {
            this._super(e, t), "handle" === e && (this._removeHandleClassName(), this._setHandleClassName())
        },
        _destroy: function() {
            return (this.helper || this.element).is(".ui-draggable-dragging") ? (this.destroyOnClear=!0, void 0) : (this.element.removeClass("ui-draggable ui-draggable-dragging ui-draggable-disabled"), this._removeHandleClassName(), this._mouseDestroy(), void 0)
        },
        _mouseCapture: function(t) {
            var i = this.document[0], s = this.options;
            try {
                i.activeElement && "body" !== i.activeElement.nodeName.toLowerCase() && e(i.activeElement).blur()
            } catch (a) {}
            return this.helper || s.disabled || e(t.target).closest(".ui-resizable-handle").length > 0?!1 : (this.handle = this._getHandle(t), this.handle ? (e(s.iframeFix===!0 ? "iframe" : s.iframeFix).each(function() {
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
            return this.helper = this._createHelper(t), this.helper.addClass("ui-draggable-dragging"), this._cacheHelperProportions(), e.ui.ddmanager && (e.ui.ddmanager.current = this), this._cacheMargins(), this.cssPosition = this.helper.css("position"), this.scrollParent = this.helper.scrollParent(!0), this.offsetParent = this.helper.offsetParent(), this.offsetParentCssPosition = this.offsetParent.css("position"), this.offset = this.positionAbs = this.element.offset(), this.offset = {
                top: this.offset.top - this.margins.top,
                left: this.offset.left - this.margins.left
            }, this.offset.scroll=!1, e.extend(this.offset, {
                click: {
                    left: t.pageX - this.offset.left,
                    top: t.pageY - this.offset.top
                },
                parent: this._getParentOffset(),
                relative: this._getRelativeOffset()
            }), this.originalPosition = this.position = this._generatePosition(t, !1), this.originalPageX = t.pageX, this.originalPageY = t.pageY, i.cursorAt && this._adjustOffsetFromHelper(i.cursorAt), this._setContainment(), this._trigger("start", t)===!1 ? (this._clear(), !1) : (this._cacheHelperProportions(), e.ui.ddmanager&&!i.dropBehaviour && e.ui.ddmanager.prepareOffsets(this, t), this._mouseDrag(t, !0), e.ui.ddmanager && e.ui.ddmanager.dragStart(this, t), !0)
        },
        _mouseDrag: function(t, i) {
            if ("fixed" === this.offsetParentCssPosition && (this.offset.parent = this._getParentOffset()), this.position = this._generatePosition(t, !0), this.positionAbs = this._convertPositionTo("absolute"), !i) {
                var s = this._uiHash();
                if (this._trigger("drag", t, s)===!1)
                    return this._mouseUp({}), !1;
                this.position = s.position
            }
            return this.helper[0].style.left = this.position.left + "px", this.helper[0].style.top = this.position.top + "px", e.ui.ddmanager && e.ui.ddmanager.drag(this, t), !1
        },
        _mouseStop: function(t) {
            var i = this, s=!1;
            return e.ui.ddmanager&&!this.options.dropBehaviour && (s = e.ui.ddmanager.drop(this, t)), this.dropped && (s = this.dropped, this.dropped=!1), "invalid" === this.options.revert&&!s || "valid" === this.options.revert && s || this.options.revert===!0 || e.isFunction(this.options.revert) && this.options.revert.call(this.element, s) ? e(this.helper).animate(this.originalPosition, parseInt(this.options.revertDuration, 10), function() {
                i._trigger("stop", t)!==!1 && i._clear()
            }) : this._trigger("stop", t)!==!1 && this._clear(), !1
        },
        _mouseUp: function(t) {
            return e("div.ui-draggable-iframeFix").each(function() {
                this.parentNode.removeChild(this)
            }), e.ui.ddmanager && e.ui.ddmanager.dragStop(this, t), this.element.focus(), e.ui.mouse.prototype._mouseUp.call(this, t)
        },
        cancel: function() {
            return this.helper.is(".ui-draggable-dragging") ? this._mouseUp({}) : this._clear(), this
        },
        _getHandle: function(t) {
            return this.options.handle?!!e(t.target).closest(this.element.find(this.options.handle)).length : !0
        },
        _setHandleClassName: function() {
            this.handleElement = this.options.handle ? this.element.find(this.options.handle) : this.element, this.handleElement.addClass("ui-draggable-handle")
        },
        _removeHandleClassName: function() {
            this.handleElement.removeClass("ui-draggable-handle")
        },
        _createHelper: function(t) {
            var i = this.options, s = e.isFunction(i.helper) ? e(i.helper.apply(this.element[0], [t])): "clone" === i.helper ? this.element.clone().removeAttr("id"): this.element;
            return s.parents("body").length || s.appendTo("parent" === i.appendTo ? this.element[0].parentNode : i.appendTo), s[0] === this.element[0] || /(fixed|absolute)/.test(s.css("position")) || s.css("position", "absolute"), s
        },
        _adjustOffsetFromHelper: function(t) {
            "string" == typeof t && (t = t.split(" ")), e.isArray(t) && (t = {
                left: + t[0],
                top: + t[1] || 0
            }), "left"in t && (this.offset.click.left = t.left + this.margins.left), "right"in t && (this.offset.click.left = this.helperProportions.width - t.right + this.margins.left), "top"in t && (this.offset.click.top = t.top + this.margins.top), "bottom"in t && (this.offset.click.top = this.helperProportions.height - t.bottom + this.margins.top)
        },
        _isRootNode: function(e) {
            return /(html|body)/i.test(e.tagName) || e === this.document[0]
        },
        _getParentOffset: function() {
            var t = this.offsetParent.offset(), i = this.document[0];
            return "absolute" === this.cssPosition && this.scrollParent[0] !== i && e.contains(this.scrollParent[0], this.offsetParent[0]) && (t.left += this.scrollParent.scrollLeft(), t.top += this.scrollParent.scrollTop()), this._isRootNode(this.offsetParent[0]) && (t = {
                top: 0,
                left: 0
            }), {
                top: t.top + (parseInt(this.offsetParent.css("borderTopWidth"), 10) || 0),
                left: t.left + (parseInt(this.offsetParent.css("borderLeftWidth"), 10) || 0)
            }
        },
        _getRelativeOffset: function() {
            if ("relative" !== this.cssPosition)
                return {
                    top: 0,
                    left: 0
                };
            var e = this.element.position(), t = this._isRootNode(this.scrollParent[0]);
            return {
                top: e.top - (parseInt(this.helper.css("top"), 10) || 0) + (t ? 0 : this.scrollParent.scrollTop()),
                left: e.left - (parseInt(this.helper.css("left"), 10) || 0) + (t ? 0 : this.scrollParent.scrollLeft())
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
            var t, i, s, a = this.options, n = this.document[0];
            return this.relativeContainer = null, a.containment ? "window" === a.containment ? (this.containment = [e(window).scrollLeft() - this.offset.relative.left - this.offset.parent.left, e(window).scrollTop() - this.offset.relative.top - this.offset.parent.top, e(window).scrollLeft() + e(window).width() - this.helperProportions.width - this.margins.left, e(window).scrollTop() + (e(window).height() || n.body.parentNode.scrollHeight) - this.helperProportions.height - this.margins.top], void 0) : "document" === a.containment ? (this.containment = [0, 0, e(n).width() - this.helperProportions.width - this.margins.left, (e(n).height() || n.body.parentNode.scrollHeight) - this.helperProportions.height - this.margins.top], void 0) : a.containment.constructor === Array ? (this.containment = a.containment, void 0) : ("parent" === a.containment && (a.containment = this.helper[0].parentNode), i = e(a.containment), s = i[0], s && (t = "hidden" !== i.css("overflow"), this.containment = [(parseInt(i.css("borderLeftWidth"), 10) || 0) + (parseInt(i.css("paddingLeft"), 10) || 0), (parseInt(i.css("borderTopWidth"), 10) || 0) + (parseInt(i.css("paddingTop"), 10) || 0), (t ? Math.max(s.scrollWidth, s.offsetWidth) : s.offsetWidth) - (parseInt(i.css("borderRightWidth"), 10) || 0) - (parseInt(i.css("paddingRight"), 10) || 0) - this.helperProportions.width - this.margins.left - this.margins.right, (t ? Math.max(s.scrollHeight, s.offsetHeight) : s.offsetHeight) - (parseInt(i.css("borderBottomWidth"), 10) || 0) - (parseInt(i.css("paddingBottom"), 10) || 0) - this.helperProportions.height - this.margins.top - this.margins.bottom], this.relativeContainer = i), void 0) : (this.containment = null, void 0)
        },
        _convertPositionTo: function(e, t) {
            t || (t = this.position);
            var i = "absolute" === e ? 1: - 1, s = this._isRootNode(this.scrollParent[0]);
            return {
                top: t.top + this.offset.relative.top * i + this.offset.parent.top * i - ("fixed" === this.cssPosition?-this.offset.scroll.top : s ? 0 : this.offset.scroll.top) * i,
                left: t.left + this.offset.relative.left * i + this.offset.parent.left * i - ("fixed" === this.cssPosition?-this.offset.scroll.left : s ? 0 : this.offset.scroll.left) * i
            }
        },
        _generatePosition: function(e, t) {
            var i, s, a, n, r = this.options, o = this._isRootNode(this.scrollParent[0]), h = e.pageX, l = e.pageY;
            return o && this.offset.scroll || (this.offset.scroll = {
                top: this.scrollParent.scrollTop(),
                left: this.scrollParent.scrollLeft()
            }), t && (this.containment && (this.relativeContainer ? (s = this.relativeContainer.offset(), i = [this.containment[0] + s.left, this.containment[1] + s.top, this.containment[2] + s.left, this.containment[3] + s.top]) : i = this.containment, e.pageX - this.offset.click.left < i[0] && (h = i[0] + this.offset.click.left), e.pageY - this.offset.click.top < i[1] && (l = i[1] + this.offset.click.top), e.pageX - this.offset.click.left > i[2] && (h = i[2] + this.offset.click.left), e.pageY - this.offset.click.top > i[3] && (l = i[3] + this.offset.click.top)), r.grid && (a = r.grid[1] ? this.originalPageY + Math.round((l - this.originalPageY) / r.grid[1]) * r.grid[1] : this.originalPageY, l = i ? a - this.offset.click.top >= i[1] || a - this.offset.click.top > i[3] ? a : a - this.offset.click.top >= i[1] ? a - r.grid[1] : a + r.grid[1] : a, n = r.grid[0] ? this.originalPageX + Math.round((h - this.originalPageX) / r.grid[0]) * r.grid[0] : this.originalPageX, h = i ? n - this.offset.click.left >= i[0] || n - this.offset.click.left > i[2] ? n : n - this.offset.click.left >= i[0] ? n - r.grid[0] : n + r.grid[0] : n), "y" === r.axis && (h = this.originalPageX), "x" === r.axis && (l = this.originalPageY)), {
                top: l - this.offset.click.top - this.offset.relative.top - this.offset.parent.top + ("fixed" === this.cssPosition?-this.offset.scroll.top : o ? 0 : this.offset.scroll.top),
                left: h - this.offset.click.left - this.offset.relative.left - this.offset.parent.left + ("fixed" === this.cssPosition?-this.offset.scroll.left : o ? 0 : this.offset.scroll.left)
            }
        },
        _clear: function() {
            this.helper.removeClass("ui-draggable-dragging"), this.helper[0] === this.element[0] || this.cancelHelperRemoval || this.helper.remove(), this.helper = null, this.cancelHelperRemoval=!1, this.destroyOnClear && this.destroy()
        },
        _trigger: function(t, i, s) {
            return s = s || this._uiHash(), e.ui.plugin.call(this, t, [i, s, this], !0), "drag" === t && (this.positionAbs = this._convertPositionTo("absolute")), e.Widget.prototype._trigger.call(this, t, i, s)
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
        start: function(t, i, s) {
            var a = s.options, n = e.extend({}, i, {
                item: s.element
            });
            s.sortables = [], e(a.connectToSortable).each(function() {
                var i = e(this).sortable("instance");
                i&&!i.options.disabled && (s.sortables.push({
                    instance: i,
                    shouldRevert: i.options.revert
                }), i.refreshPositions(), i._trigger("activate", t, n))
            })
        },
        stop: function(t, i, s) {
            var a = e.extend({}, i, {
                item: s.element
            });
            e.each(s.sortables, function() {
                this.instance.isOver ? (this.instance.isOver = 0, s.cancelHelperRemoval=!0, this.instance.cancelHelperRemoval=!1, this.shouldRevert && (this.instance.options.revert = this.shouldRevert), this.instance._mouseStop(t), this.instance.options.helper = this.instance.options._helper, "original" === s.options.helper && this.instance.currentItem.css({
                    top: "auto",
                    left: "auto"
                })) : (this.instance.cancelHelperRemoval=!1, this.instance._trigger("deactivate", t, a))
            })
        },
        drag: function(t, i, s) {
            var a = this;
            e.each(s.sortables, function() {
                var n=!1, r = this;
                this.instance.positionAbs = s.positionAbs, this.instance.helperProportions = s.helperProportions, this.instance.offset.click = s.offset.click, this.instance._intersectsWith(this.instance.containerCache) && (n=!0, e.each(s.sortables, function() {
                    return this.instance.positionAbs = s.positionAbs, this.instance.helperProportions = s.helperProportions, this.instance.offset.click = s.offset.click, this !== r && this.instance._intersectsWith(this.instance.containerCache) && e.contains(r.instance.element[0], this.instance.element[0]) && (n=!1), n
                })), n ? (this.instance.isOver || (this.instance.isOver = 1, this.instance.currentItem = e(a).clone().removeAttr("id").appendTo(this.instance.element).data("ui-sortable-item", !0), this.instance.options._helper = this.instance.options.helper, this.instance.options.helper = function() {
                    return i.helper[0]
                }, t.target = this.instance.currentItem[0], this.instance._mouseCapture(t, !0), this.instance._mouseStart(t, !0, !0), this.instance.offset.click.top = s.offset.click.top, this.instance.offset.click.left = s.offset.click.left, this.instance.offset.parent.left -= s.offset.parent.left - this.instance.offset.parent.left, this.instance.offset.parent.top -= s.offset.parent.top - this.instance.offset.parent.top, s._trigger("toSortable", t), s.dropped = this.instance.element, s.currentItem = s.element, this.instance.fromOutside = s), this.instance.currentItem && this.instance._mouseDrag(t)) : this.instance.isOver && (this.instance.isOver = 0, this.instance.cancelHelperRemoval=!0, this.instance.options.revert=!1, this.instance._trigger("out", t, this.instance._uiHash(this.instance)), this.instance._mouseStop(t, !0), this.instance.options.helper = this.instance.options._helper, this.instance.currentItem.remove(), this.instance.placeholder && this.instance.placeholder.remove(), s._trigger("fromSortable", t), s.dropped=!1)
            })
        }
    }), e.ui.plugin.add("draggable", "cursor", {
        start: function(t, i, s) {
            var a = e("body"), n = s.options;
            a.css("cursor") && (n._cursor = a.css("cursor")), a.css("cursor", n.cursor)
        },
        stop: function(t, i, s) {
            var a = s.options;
            a._cursor && e("body").css("cursor", a._cursor)
        }
    }), e.ui.plugin.add("draggable", "opacity", {
        start: function(t, i, s) {
            var a = e(i.helper), n = s.options;
            a.css("opacity") && (n._opacity = a.css("opacity")), a.css("opacity", n.opacity)
        },
        stop: function(t, i, s) {
            var a = s.options;
            a._opacity && e(i.helper).css("opacity", a._opacity)
        }
    }), e.ui.plugin.add("draggable", "scroll", {
        start: function(e, t, i) {
            i.scrollParentNotHidden || (i.scrollParentNotHidden = i.helper.scrollParent(!1)), i.scrollParentNotHidden[0] !== i.document[0] && "HTML" !== i.scrollParentNotHidden[0].tagName && (i.overflowOffset = i.scrollParentNotHidden.offset())
        },
        drag: function(t, i, s) {
            var a = s.options, n=!1, r = s.scrollParentNotHidden[0], o = s.document[0];
            r !== o && "HTML" !== r.tagName ? (a.axis && "x" === a.axis || (s.overflowOffset.top + r.offsetHeight - t.pageY < a.scrollSensitivity ? r.scrollTop = n = r.scrollTop + a.scrollSpeed : t.pageY - s.overflowOffset.top < a.scrollSensitivity && (r.scrollTop = n = r.scrollTop - a.scrollSpeed)), a.axis && "y" === a.axis || (s.overflowOffset.left + r.offsetWidth - t.pageX < a.scrollSensitivity ? r.scrollLeft = n = r.scrollLeft + a.scrollSpeed : t.pageX - s.overflowOffset.left < a.scrollSensitivity && (r.scrollLeft = n = r.scrollLeft - a.scrollSpeed))) : (a.axis && "x" === a.axis || (t.pageY - e(o).scrollTop() < a.scrollSensitivity ? n = e(o).scrollTop(e(o).scrollTop() - a.scrollSpeed) : e(window).height() - (t.pageY - e(o).scrollTop()) < a.scrollSensitivity && (n = e(o).scrollTop(e(o).scrollTop() + a.scrollSpeed))), a.axis && "y" === a.axis || (t.pageX - e(o).scrollLeft() < a.scrollSensitivity ? n = e(o).scrollLeft(e(o).scrollLeft() - a.scrollSpeed) : e(window).width() - (t.pageX - e(o).scrollLeft()) < a.scrollSensitivity && (n = e(o).scrollLeft(e(o).scrollLeft() + a.scrollSpeed)))), n!==!1 && e.ui.ddmanager&&!a.dropBehaviour && e.ui.ddmanager.prepareOffsets(s, t)
        }
    }), e.ui.plugin.add("draggable", "snap", {
        start: function(t, i, s) {
            var a = s.options;
            s.snapElements = [], e(a.snap.constructor !== String ? a.snap.items || ":data(ui-draggable)" : a.snap).each(function() {
                var t = e(this), i = t.offset();
                this !== s.element[0] && s.snapElements.push({
                    item: this,
                    width: t.outerWidth(),
                    height: t.outerHeight(),
                    top: i.top,
                    left: i.left
                })
            })
        },
        drag: function(t, i, s) {
            var a, n, r, o, h, l, u, d, c, p, f = s.options, m = f.snapTolerance, g = i.offset.left, v = g + s.helperProportions.width, y = i.offset.top, b = y + s.helperProportions.height;
            for (c = s.snapElements.length - 1; c >= 0; c--)
                h = s.snapElements[c].left, l = h + s.snapElements[c].width, u = s.snapElements[c].top, d = u + s.snapElements[c].height, h - m > v || g > l + m || u - m > b || y > d + m ||!e.contains(s.snapElements[c].item.ownerDocument, s.snapElements[c].item) ? (s.snapElements[c].snapping && s.options.snap.release && s.options.snap.release.call(s.element, t, e.extend(s._uiHash(), {
                    snapItem: s.snapElements[c].item
                })), s.snapElements[c].snapping=!1) : ("inner" !== f.snapMode && (a = m >= Math.abs(u - b), n = m >= Math.abs(d - y), r = m >= Math.abs(h - v), o = m >= Math.abs(l - g), a && (i.position.top = s._convertPositionTo("relative", {
                    top: u - s.helperProportions.height,
                    left: 0
                }).top - s.margins.top), n && (i.position.top = s._convertPositionTo("relative", {
                    top: d,
                    left: 0
                }).top - s.margins.top), r && (i.position.left = s._convertPositionTo("relative", {
                    top: 0,
                    left: h - s.helperProportions.width
                }).left - s.margins.left), o && (i.position.left = s._convertPositionTo("relative", {
                    top: 0,
                    left: l
                }).left - s.margins.left)), p = a || n || r || o, "outer" !== f.snapMode && (a = m >= Math.abs(u - y), n = m >= Math.abs(d - b), r = m >= Math.abs(h - g), o = m >= Math.abs(l - v), a && (i.position.top = s._convertPositionTo("relative", {
                    top: u,
                    left: 0
                }).top - s.margins.top), n && (i.position.top = s._convertPositionTo("relative", {
                    top: d - s.helperProportions.height,
                    left: 0
                }).top - s.margins.top), r && (i.position.left = s._convertPositionTo("relative", {
                    top: 0,
                    left: h
                }).left - s.margins.left), o && (i.position.left = s._convertPositionTo("relative", {
                    top: 0,
                    left: l - s.helperProportions.width
                }).left - s.margins.left)), !s.snapElements[c].snapping && (a || n || r || o || p) && s.options.snap.snap && s.options.snap.snap.call(s.element, t, e.extend(s._uiHash(), {
                    snapItem: s.snapElements[c].item
                })), s.snapElements[c].snapping = a || n || r || o || p)
        }
    }), e.ui.plugin.add("draggable", "stack", {
        start: function(t, i, s) {
            var a, n = s.options, r = e.makeArray(e(n.stack)).sort(function(t, i) {
                return (parseInt(e(t).css("zIndex"), 10) || 0) - (parseInt(e(i).css("zIndex"), 10) || 0)
            });
            r.length && (a = parseInt(e(r[0]).css("zIndex"), 10) || 0, e(r).each(function(t) {
                e(this).css("zIndex", a + t)
            }), this.css("zIndex", a + r.length))
        }
    }), e.ui.plugin.add("draggable", "zIndex", {
        start: function(t, i, s) {
            var a = e(i.helper), n = s.options;
            a.css("zIndex") && (n._zIndex = a.css("zIndex")), a.css("zIndex", n.zIndex)
        },
        stop: function(t, i, s) {
            var a = s.options;
            a._zIndex && e(i.helper).css("zIndex", a._zIndex)
        }
    }), e.ui.draggable, e.widget("ui.droppable", {
        version: "1.11.1",
        widgetEventPrefix: "drop",
        options: {
            accept: "*",
            activeClass: !1,
            addClasses: !0,
            greedy: !1,
            hoverClass: !1,
            scope: "default",
            tolerance: "intersect",
            activate: null,
            deactivate: null,
            drop: null,
            out: null,
            over: null
        },
        _create: function() {
            var t, i = this.options, s = i.accept;
            this.isover=!1, this.isout=!0, this.accept = e.isFunction(s) ? s : function(e) {
                return e.is(s)
            }, this.proportions = function() {
                return arguments.length ? (t = arguments[0], void 0) : t ? t : t = {
                    width: this.element[0].offsetWidth,
                    height: this.element[0].offsetHeight
                }
            }, this._addToManager(i.scope), i.addClasses && this.element.addClass("ui-droppable")
        },
        _addToManager: function(t) {
            e.ui.ddmanager.droppables[t] = e.ui.ddmanager.droppables[t] || [], e.ui.ddmanager.droppables[t].push(this)
        },
        _splice: function(e) {
            for (var t = 0; e.length > t; t++)
                e[t] === this && e.splice(t, 1)
        },
        _destroy: function() {
            var t = e.ui.ddmanager.droppables[this.options.scope];
            this._splice(t), this.element.removeClass("ui-droppable ui-droppable-disabled")
        },
        _setOption: function(t, i) {
            if ("accept" === t)
                this.accept = e.isFunction(i) ? i : function(e) {
                    return e.is(i)
                };
            else if ("scope" === t) {
                var s = e.ui.ddmanager.droppables[this.options.scope];
                this._splice(s), this._addToManager(i)
            }
            this._super(t, i)
        },
        _activate: function(t) {
            var i = e.ui.ddmanager.current;
            this.options.activeClass && this.element.addClass(this.options.activeClass), i && this._trigger("activate", t, this.ui(i))
        },
        _deactivate: function(t) {
            var i = e.ui.ddmanager.current;
            this.options.activeClass && this.element.removeClass(this.options.activeClass), i && this._trigger("deactivate", t, this.ui(i))
        },
        _over: function(t) {
            var i = e.ui.ddmanager.current;
            i && (i.currentItem || i.element)[0] !== this.element[0] && this.accept.call(this.element[0], i.currentItem || i.element) && (this.options.hoverClass && this.element.addClass(this.options.hoverClass), this._trigger("over", t, this.ui(i)))
        },
        _out: function(t) {
            var i = e.ui.ddmanager.current;
            i && (i.currentItem || i.element)[0] !== this.element[0] && this.accept.call(this.element[0], i.currentItem || i.element) && (this.options.hoverClass && this.element.removeClass(this.options.hoverClass), this._trigger("out", t, this.ui(i)))
        },
        _drop: function(t, i) {
            var s = i || e.ui.ddmanager.current, a=!1;
            return s && (s.currentItem || s.element)[0] !== this.element[0] ? (this.element.find(":data(ui-droppable)").not(".ui-draggable-dragging").each(function() {
                var i = e(this).droppable("instance");
                return i.options.greedy&&!i.options.disabled && i.options.scope === s.options.scope && i.accept.call(i.element[0], s.currentItem || s.element) && e.ui.intersect(s, e.extend(i, {
                    offset: i.element.offset()
                }), i.options.tolerance, t) ? (a=!0, !1) : void 0
            }), a?!1 : this.accept.call(this.element[0], s.currentItem || s.element) ? (this.options.activeClass && this.element.removeClass(this.options.activeClass), this.options.hoverClass && this.element.removeClass(this.options.hoverClass), this._trigger("drop", t, this.ui(s)), this.element) : !1) : !1
        },
        ui: function(e) {
            return {
                draggable: e.currentItem || e.element,
                helper: e.helper,
                position: e.position,
                offset: e.positionAbs
            }
        }
    }), e.ui.intersect = function() {
        function e(e, t, i) {
            return e >= t && t + i > e
        }
        return function(t, i, s, a) {
            if (!i.offset)
                return !1;
            var n = (t.positionAbs || t.position.absolute).left, r = (t.positionAbs || t.position.absolute).top, o = n + t.helperProportions.width, h = r + t.helperProportions.height, l = i.offset.left, u = i.offset.top, d = l + i.proportions().width, c = u + i.proportions().height;
            switch (s) {
            case"fit":
                return n >= l && d >= o && r >= u && c >= h;
            case"intersect":
                return n + t.helperProportions.width / 2 > l && d > o - t.helperProportions.width / 2 && r + t.helperProportions.height / 2 > u && c > h - t.helperProportions.height / 2;
            case"pointer":
                return e(a.pageY, u, i.proportions().height) && e(a.pageX, l, i.proportions().width);
            case"touch":
                return (r >= u && c >= r || h >= u && c >= h || u > r && h > c) && (n >= l && d >= n || o >= l && d >= o || l > n && o > d);
            default:
                return !1
            }
        }
    }(), e.ui.ddmanager = {
        current: null,
        droppables: {
            "default": []
        },
        prepareOffsets: function(t, i) {
            var s, a, n = e.ui.ddmanager.droppables[t.options.scope] || [], r = i ? i.type: null, o = (t.currentItem || t.element).find(":data(ui-droppable)").addBack();
            e: for (s = 0; n.length > s; s++)
                if (!(n[s].options.disabled || t&&!n[s].accept.call(n[s].element[0], t.currentItem || t.element))) {
                    for (a = 0; o.length > a; a++)
                        if (o[a] === n[s].element[0]) {
                            n[s].proportions().height = 0;
                            continue e
                        }
                        n[s].visible = "none" !== n[s].element.css("display"), n[s].visible && ("mousedown" === r && n[s]._activate.call(n[s], i), n[s].offset = n[s].element.offset(), n[s].proportions({
                            width: n[s].element[0].offsetWidth,
                            height: n[s].element[0].offsetHeight
                        }))
                }
        },
        drop: function(t, i) {
            var s=!1;
            return e.each((e.ui.ddmanager.droppables[t.options.scope] || []).slice(), function() {
                this.options && (!this.options.disabled && this.visible && e.ui.intersect(t, this, this.options.tolerance, i) && (s = this._drop.call(this, i) || s), !this.options.disabled && this.visible && this.accept.call(this.element[0], t.currentItem || t.element) && (this.isout=!0, this.isover=!1, this._deactivate.call(this, i)))
            }), s
        },
        dragStart: function(t, i) {
            t.element.parentsUntil("body").bind("scroll.droppable", function() {
                t.options.refreshPositions || e.ui.ddmanager.prepareOffsets(t, i)
            })
        },
        drag: function(t, i) {
            t.options.refreshPositions && e.ui.ddmanager.prepareOffsets(t, i), e.each(e.ui.ddmanager.droppables[t.options.scope] || [], function() {
                if (!this.options.disabled&&!this.greedyChild && this.visible) {
                    var s, a, n, r = e.ui.intersect(t, this, this.options.tolerance, i), o=!r && this.isover ? "isout" : r&&!this.isover ? "isover" : null;
                    o && (this.options.greedy && (a = this.options.scope, n = this.element.parents(":data(ui-droppable)").filter(function() {
                        return e(this).droppable("instance").options.scope === a
                    }), n.length && (s = e(n[0]).droppable("instance"), s.greedyChild = "isover" === o)), s && "isover" === o && (s.isover=!1, s.isout=!0, s._out.call(s, i)), this[o]=!0, this["isout" === o ? "isover": "isout"]=!1, this["isover" === o ? "_over": "_out"].call(this, i), s && "isout" === o && (s.isout=!1, s.isover=!0, s._over.call(s, i)))
                }
            })
        },
        dragStop: function(t, i) {
            t.element.parentsUntil("body").unbind("scroll.droppable"), t.options.refreshPositions || e.ui.ddmanager.prepareOffsets(t, i)
        }
    }, e.ui.droppable, e.widget("ui.resizable", e.ui.mouse, {
        version: "1.11.1",
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
        _num: function(e) {
            return parseInt(e, 10) || 0
        },
        _isNumber: function(e) {
            return !isNaN(parseInt(e, 10))
        },
        _hasScroll: function(t, i) {
            if ("hidden" === e(t).css("overflow"))
                return !1;
            var s = i && "left" === i ? "scrollLeft": "scrollTop", a=!1;
            return t[s] > 0?!0 : (t[s] = 1, a = t[s] > 0, t[s] = 0, a)
        },
        _create: function() {
            var t, i, s, a, n, r = this, o = this.options;
            if (this.element.addClass("ui-resizable"), e.extend(this, {
                _aspectRatio: !!o.aspectRatio,
                aspectRatio: o.aspectRatio,
                originalElement: this.element,
                _proportionallyResizeElements: [],
                _helper: o.helper || o.ghost || o.animate ? o.helper || "ui-resizable-helper": null
            }), this.element[0].nodeName.match(/canvas|textarea|input|select|button|img/i) && (this.element.wrap(e("<div class='ui-wrapper' style='overflow: hidden;'></div>").css({
                position: this.element.css("position"),
                width: this.element.outerWidth(),
                height: this.element.outerHeight(),
                top: this.element.css("top"),
                left: this.element.css("left")
            })), this.element = this.element.parent().data("ui-resizable", this.element.resizable("instance")), this.elementIsWrapper=!0, this.element.css({
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
            }), this._proportionallyResize()), this.handles = o.handles || (e(".ui-resizable-handle", this.element).length ? {
                n: ".ui-resizable-n",
                e: ".ui-resizable-e",
                s: ".ui-resizable-s",
                w: ".ui-resizable-w",
                se: ".ui-resizable-se",
                sw: ".ui-resizable-sw",
                ne: ".ui-resizable-ne",
                nw: ".ui-resizable-nw"
            } : "e,s,se"), this.handles.constructor === String)
                for ("all" === this.handles && (this.handles = "n,e,s,w,se,sw,ne,nw"), t = this.handles.split(","), this.handles = {}, i = 0; t.length > i; i++)
                    s = e.trim(t[i]), n = "ui-resizable-" + s, a = e("<div class='ui-resizable-handle " + n + "'></div>"), a.css({
                        zIndex: o.zIndex
                    }), "se" === s && a.addClass("ui-icon ui-icon-gripsmall-diagonal-se"), this.handles[s] = ".ui-resizable-" + s, this.element.append(a);
            this._renderAxis = function(t) {
                var i, s, a, n;
                t = t || this.element;
                for (i in this.handles)
                    this.handles[i].constructor === String && (this.handles[i] = this.element.children(this.handles[i]).first().show()), this.elementIsWrapper && this.originalElement[0].nodeName.match(/textarea|input|select|button/i) && (s = e(this.handles[i], this.element), n = /sw|ne|nw|se|n|s/.test(i) ? s.outerHeight() : s.outerWidth(), a = ["padding", /ne|nw|n/.test(i) ? "Top": /se|sw|s/.test(i) ? "Bottom": /^e$/.test(i) ? "Right": "Left"].join(""), t.css(a, n), this._proportionallyResize()), e(this.handles[i]).length
            }, this._renderAxis(this.element), this._handles = e(".ui-resizable-handle", this.element).disableSelection(), this._handles.mouseover(function() {
                r.resizing || (this.className && (a = this.className.match(/ui-resizable-(se|sw|ne|nw|n|e|s|w)/i)), r.axis = a && a[1] ? a[1] : "se")
            }), o.autoHide && (this._handles.hide(), e(this.element).addClass("ui-resizable-autohide").mouseenter(function() {
                o.disabled || (e(this).removeClass("ui-resizable-autohide"), r._handles.show())
            }).mouseleave(function() {
                o.disabled || r.resizing || (e(this).addClass("ui-resizable-autohide"), r._handles.hide())
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
            var i, s, a=!1;
            for (i in this.handles)
                s = e(this.handles[i])[0], (s === t.target || e.contains(s, t.target)) && (a=!0);
            return !this.options.disabled && a
        },
        _mouseStart: function(t) {
            var i, s, a, n = this.options, r = this.element;
            return this.resizing=!0, this._renderProxy(), i = this._num(this.helper.css("left")), s = this._num(this.helper.css("top")), n.containment && (i += e(n.containment).scrollLeft() || 0, s += e(n.containment).scrollTop() || 0), this.offset = this.helper.offset(), this.position = {
                left: i,
                top: s
            }, this.size = this._helper ? {
                width: this.helper.width(),
                height: this.helper.height()
            } : {
                width: r.width(),
                height: r.height()
            }, this.originalSize = this._helper ? {
                width: r.outerWidth(),
                height: r.outerHeight()
            } : {
                width: r.width(),
                height: r.height()
            }, this.sizeDiff = {
                width: r.outerWidth() - r.width(),
                height: r.outerHeight() - r.height()
            }, this.originalPosition = {
                left: i,
                top: s
            }, this.originalMousePosition = {
                left: t.pageX,
                top: t.pageY
            }, this.aspectRatio = "number" == typeof n.aspectRatio ? n.aspectRatio : this.originalSize.width / this.originalSize.height || 1, a = e(".ui-resizable-" + this.axis).css("cursor"), e("body").css("cursor", "auto" === a ? this.axis + "-resize" : a), r.addClass("ui-resizable-resizing"), this._propagate("start", t), !0
        },
        _mouseDrag: function(t) {
            var i, s, a = this.originalMousePosition, n = this.axis, r = t.pageX - a.left || 0, o = t.pageY - a.top || 0, h = this._change[n];
            return this._updatePrevProperties(), h ? (i = h.apply(this, [t, r, o]), this._updateVirtualBoundaries(t.shiftKey), (this._aspectRatio || t.shiftKey) && (i = this._updateRatio(i, t)), i = this._respectSize(i, t), this._updateCache(i), this._propagate("resize", t), s = this._applyChanges(), !this._helper && this._proportionallyResizeElements.length && this._proportionallyResize(), e.isEmptyObject(s) || (this._updatePrevProperties(), this._trigger("resize", t, this.ui()), this._applyChanges()), !1) : !1
        },
        _mouseStop: function(t) {
            this.resizing=!1;
            var i, s, a, n, r, o, h, l = this.options, u = this;
            return this._helper && (i = this._proportionallyResizeElements, s = i.length && /textarea/i.test(i[0].nodeName), a = s && this._hasScroll(i[0], "left") ? 0 : u.sizeDiff.height, n = s ? 0 : u.sizeDiff.width, r = {
                width: u.helper.width() - n,
                height: u.helper.height() - a
            }, o = parseInt(u.element.css("left"), 10) + (u.position.left - u.originalPosition.left) || null, h = parseInt(u.element.css("top"), 10) + (u.position.top - u.originalPosition.top) || null, l.animate || this.element.css(e.extend(r, {
                top: h,
                left: o
            })), u.helper.height(u.size.height), u.helper.width(u.size.width), this._helper&&!l.animate && this._proportionallyResize()), e("body").css("cursor", "auto"), this.element.removeClass("ui-resizable-resizing"), this._propagate("stop", t), this._helper && this.helper.remove(), !1
        },
        _updatePrevProperties: function() {
            this.prevPosition = {
                top: this.position.top,
                left: this.position.left
            }, this.prevSize = {
                width: this.size.width,
                height: this.size.height
            }
        },
        _applyChanges: function() {
            var e = {};
            return this.position.top !== this.prevPosition.top && (e.top = this.position.top + "px"), this.position.left !== this.prevPosition.left && (e.left = this.position.left + "px"), this.size.width !== this.prevSize.width && (e.width = this.size.width + "px"), this.size.height !== this.prevSize.height && (e.height = this.size.height + "px"), this.helper.css(e), e
        },
        _updateVirtualBoundaries: function(e) {
            var t, i, s, a, n, r = this.options;
            n = {
                minWidth: this._isNumber(r.minWidth) ? r.minWidth: 0,
                maxWidth: this._isNumber(r.maxWidth) ? r.maxWidth: 1 / 0,
                minHeight: this._isNumber(r.minHeight) ? r.minHeight: 0,
                maxHeight: this._isNumber(r.maxHeight) ? r.maxHeight: 1 / 0
            }, (this._aspectRatio || e) && (t = n.minHeight * this.aspectRatio, s = n.minWidth / this.aspectRatio, i = n.maxHeight * this.aspectRatio, a = n.maxWidth / this.aspectRatio, t > n.minWidth && (n.minWidth = t), s > n.minHeight && (n.minHeight = s), n.maxWidth > i && (n.maxWidth = i), n.maxHeight > a && (n.maxHeight = a)), this._vBoundaries = n
        },
        _updateCache: function(e) {
            this.offset = this.helper.offset(), this._isNumber(e.left) && (this.position.left = e.left), this._isNumber(e.top) && (this.position.top = e.top), this._isNumber(e.height) && (this.size.height = e.height), this._isNumber(e.width) && (this.size.width = e.width)
        },
        _updateRatio: function(e) {
            var t = this.position, i = this.size, s = this.axis;
            return this._isNumber(e.height) ? e.width = e.height * this.aspectRatio : this._isNumber(e.width) && (e.height = e.width / this.aspectRatio), "sw" === s && (e.left = t.left + (i.width - e.width), e.top = null), "nw" === s && (e.top = t.top + (i.height - e.height), e.left = t.left + (i.width - e.width)), e
        },
        _respectSize: function(e) {
            var t = this._vBoundaries, i = this.axis, s = this._isNumber(e.width) && t.maxWidth && t.maxWidth < e.width, a = this._isNumber(e.height) && t.maxHeight && t.maxHeight < e.height, n = this._isNumber(e.width) && t.minWidth && t.minWidth > e.width, r = this._isNumber(e.height) && t.minHeight && t.minHeight > e.height, o = this.originalPosition.left + this.originalSize.width, h = this.position.top + this.size.height, l = /sw|nw|w/.test(i), u = /nw|ne|n/.test(i);
            return n && (e.width = t.minWidth), r && (e.height = t.minHeight), s && (e.width = t.maxWidth), a && (e.height = t.maxHeight), n && l && (e.left = o - t.minWidth), s && l && (e.left = o - t.maxWidth), r && u && (e.top = h - t.minHeight), a && u && (e.top = h - t.maxHeight), e.width || e.height || e.left ||!e.top ? e.width || e.height || e.top ||!e.left || (e.left = null) : e.top = null, e
        },
        _getPaddingPlusBorderDimensions: function(e) {
            for (var t = 0, i = [], s = [e.css("borderTopWidth"), e.css("borderRightWidth"), e.css("borderBottomWidth"), e.css("borderLeftWidth")], a = [e.css("paddingTop"), e.css("paddingRight"), e.css("paddingBottom"), e.css("paddingLeft")]; 4 > t; t++)
                i[t] = parseInt(s[t], 10) || 0, i[t] += parseInt(a[t], 10) || 0;
            return {
                height: i[0] + i[2],
                width: i[1] + i[3]
            }
        },
        _proportionallyResize: function() {
            if (this._proportionallyResizeElements.length)
                for (var e, t = 0, i = this.helper || this.element; this._proportionallyResizeElements.length > t; t++)
                    e = this._proportionallyResizeElements[t], this.outerDimensions || (this.outerDimensions = this._getPaddingPlusBorderDimensions(e)), e.css({
                        height: i.height() - this.outerDimensions.height || 0,
                        width: i.width() - this.outerDimensions.width || 0
                    })
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
                var i = this.originalSize, s = this.originalPosition;
                return {
                    left: s.left + t,
                    width: i.width - t
                }
            },
            n: function(e, t, i) {
                var s = this.originalSize, a = this.originalPosition;
                return {
                    top: a.top + i,
                    height: s.height - i
                }
            },
            s: function(e, t, i) {
                return {
                    height: this.originalSize.height + i
                }
            },
            se: function(t, i, s) {
                return e.extend(this._change.s.apply(this, arguments), this._change.e.apply(this, [t, i, s]))
            },
            sw: function(t, i, s) {
                return e.extend(this._change.s.apply(this, arguments), this._change.w.apply(this, [t, i, s]))
            },
            ne: function(t, i, s) {
                return e.extend(this._change.n.apply(this, arguments), this._change.e.apply(this, [t, i, s]))
            },
            nw: function(t, i, s) {
                return e.extend(this._change.n.apply(this, arguments), this._change.w.apply(this, [t, i, s]))
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
            var i = e(this).resizable("instance"), s = i.options, a = i._proportionallyResizeElements, n = a.length && /textarea/i.test(a[0].nodeName), r = n && i._hasScroll(a[0], "left") ? 0: i.sizeDiff.height, o = n ? 0: i.sizeDiff.width, h = {
                width: i.size.width - o,
                height: i.size.height - r
            }, l = parseInt(i.element.css("left"), 10) + (i.position.left - i.originalPosition.left) || null, u = parseInt(i.element.css("top"), 10) + (i.position.top - i.originalPosition.top) || null;
            i.element.animate(e.extend(h, u && l ? {
                top: u,
                left: l
            } : {}), {
                duration: s.animateDuration,
                easing: s.animateEasing,
                step: function() {
                    var s = {
                        width: parseInt(i.element.css("width"), 10),
                        height: parseInt(i.element.css("height"), 10),
                        top: parseInt(i.element.css("top"), 10),
                        left: parseInt(i.element.css("left"), 10)
                    };
                    a && a.length && e(a[0]).css({
                        width: s.width,
                        height: s.height
                    }), i._updateCache(s), i._propagate("resize", t)
                }
            })
        }
    }), e.ui.plugin.add("resizable", "containment", {
        start: function() {
            var t, i, s, a, n, r, o, h = e(this).resizable("instance"), l = h.options, u = h.element, d = l.containment, c = d instanceof e ? d.get(0): /parent/.test(d) ? u.parent().get(0): d;
            c && (h.containerElement = e(c), /document/.test(d) || d === document ? (h.containerOffset = {
                left: 0,
                top: 0
            }, h.containerPosition = {
                left: 0,
                top: 0
            }, h.parentData = {
                element: e(document),
                left: 0,
                top: 0,
                width: e(document).width(),
                height: e(document).height() || document.body.parentNode.scrollHeight
            }) : (t = e(c), i = [], e(["Top", "Right", "Left", "Bottom"]).each(function(e, s) {
                i[e] = h._num(t.css("padding" + s))
            }), h.containerOffset = t.offset(), h.containerPosition = t.position(), h.containerSize = {
                height: t.innerHeight() - i[3],
                width: t.innerWidth() - i[1]
            }, s = h.containerOffset, a = h.containerSize.height, n = h.containerSize.width, r = h._hasScroll(c, "left") ? c.scrollWidth : n, o = h._hasScroll(c) ? c.scrollHeight : a, h.parentData = {
                element: c,
                left: s.left,
                top: s.top,
                width: r,
                height: o
            }))
        },
        resize: function(t) {
            var i, s, a, n, r = e(this).resizable("instance"), o = r.options, h = r.containerOffset, l = r.position, u = r._aspectRatio || t.shiftKey, d = {
                top: 0,
                left: 0
            }, c = r.containerElement, p=!0;
            c[0] !== document && /static/.test(c.css("position")) && (d = h), l.left < (r._helper ? h.left : 0) && (r.size.width = r.size.width + (r._helper ? r.position.left - h.left : r.position.left - d.left), u && (r.size.height = r.size.width / r.aspectRatio, p=!1), r.position.left = o.helper ? h.left : 0), l.top < (r._helper ? h.top : 0) && (r.size.height = r.size.height + (r._helper ? r.position.top - h.top : r.position.top), u && (r.size.width = r.size.height * r.aspectRatio, p=!1), r.position.top = r._helper ? h.top : 0), a = r.containerElement.get(0) === r.element.parent().get(0), n = /relative|absolute/.test(r.containerElement.css("position")), a && n ? (r.offset.left = r.parentData.left + r.position.left, r.offset.top = r.parentData.top + r.position.top) : (r.offset.left = r.element.offset().left, r.offset.top = r.element.offset().top), i = Math.abs(r.sizeDiff.width + (r._helper ? r.offset.left - d.left : r.offset.left - h.left)), s = Math.abs(r.sizeDiff.height + (r._helper ? r.offset.top - d.top : r.offset.top - h.top)), i + r.size.width >= r.parentData.width && (r.size.width = r.parentData.width - i, u && (r.size.height = r.size.width / r.aspectRatio, p=!1)), s + r.size.height >= r.parentData.height && (r.size.height = r.parentData.height - s, u && (r.size.width = r.size.height * r.aspectRatio, p=!1)), p || (r.position.left = r.prevPosition.left, r.position.top = r.prevPosition.top, r.size.width = r.prevSize.width, r.size.height = r.prevSize.height)
        },
        stop: function() {
            var t = e(this).resizable("instance"), i = t.options, s = t.containerOffset, a = t.containerPosition, n = t.containerElement, r = e(t.helper), o = r.offset(), h = r.outerWidth() - t.sizeDiff.width, l = r.outerHeight() - t.sizeDiff.height;
            t._helper&&!i.animate && /relative/.test(n.css("position")) && e(this).css({
                left: o.left - a.left - s.left,
                width: h,
                height: l
            }), t._helper&&!i.animate && /static/.test(n.css("position")) && e(this).css({
                left: o.left - a.left - s.left,
                width: h,
                height: l
            })
        }
    }), e.ui.plugin.add("resizable", "alsoResize", {
        start: function() {
            var t = e(this).resizable("instance"), i = t.options, s = function(t) {
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
            "object" != typeof i.alsoResize || i.alsoResize.parentNode ? s(i.alsoResize) : i.alsoResize.length ? (i.alsoResize = i.alsoResize[0], s(i.alsoResize)) : e.each(i.alsoResize, function(e) {
                s(e)
            })
        },
        resize: function(t, i) {
            var s = e(this).resizable("instance"), a = s.options, n = s.originalSize, r = s.originalPosition, o = {
                height: s.size.height - n.height || 0,
                width: s.size.width - n.width || 0,
                top: s.position.top - r.top || 0,
                left: s.position.left - r.left || 0
            }, h = function(t, s) {
                e(t).each(function() {
                    var t = e(this), a = e(this).data("ui-resizable-alsoresize"), n = {}, r = s && s.length ? s: t.parents(i.originalElement[0]).length ? ["width", "height"]: ["width", "height", "top", "left"];
                    e.each(r, function(e, t) {
                        var i = (a[t] || 0) + (o[t] || 0);
                        i && i >= 0 && (n[t] = i || null)
                    }), t.css(n)
                })
            };
            "object" != typeof a.alsoResize || a.alsoResize.nodeType ? h(a.alsoResize) : e.each(a.alsoResize, function(e, t) {
                h(e, t)
            })
        },
        stop: function() {
            e(this).removeData("resizable-alsoresize")
        }
    }), e.ui.plugin.add("resizable", "ghost", {
        start: function() {
            var t = e(this).resizable("instance"), i = t.options, s = t.size;
            t.ghost = t.originalElement.clone(), t.ghost.css({
                opacity: .25,
                display: "block",
                position: "relative",
                height: s.height,
                width: s.width,
                margin: 0,
                left: 0,
                top: 0
            }).addClass("ui-resizable-ghost").addClass("string" == typeof i.ghost ? i.ghost : ""), t.ghost.appendTo(t.helper)
        },
        resize: function() {
            var t = e(this).resizable("instance");
            t.ghost && t.ghost.css({
                position: "relative",
                height: t.size.height,
                width: t.size.width
            })
        },
        stop: function() {
            var t = e(this).resizable("instance");
            t.ghost && t.helper && t.helper.get(0).removeChild(t.ghost.get(0))
        }
    }), e.ui.plugin.add("resizable", "grid", {
        resize: function() {
            var t, i = e(this).resizable("instance"), s = i.options, a = i.size, n = i.originalSize, r = i.originalPosition, o = i.axis, h = "number" == typeof s.grid ? [s.grid, s.grid]: s.grid, l = h[0] || 1, u = h[1] || 1, d = Math.round((a.width - n.width) / l) * l, c = Math.round((a.height - n.height) / u) * u, p = n.width + d, f = n.height + c, m = s.maxWidth && p > s.maxWidth, g = s.maxHeight && f > s.maxHeight, v = s.minWidth && s.minWidth > p, y = s.minHeight && s.minHeight > f;
            s.grid = h, v && (p += l), y && (f += u), m && (p -= l), g && (f -= u), /^(se|s|e)$/.test(o) ? (i.size.width = p, i.size.height = f) : /^(ne)$/.test(o) ? (i.size.width = p, i.size.height = f, i.position.top = r.top - c) : /^(sw)$/.test(o) ? (i.size.width = p, i.size.height = f, i.position.left = r.left - d) : ((0 >= f - u || 0 >= p - l) && (t = i._getPaddingPlusBorderDimensions(this)), f - u > 0 ? (i.size.height = f, i.position.top = r.top - c) : (f = u - t.height, i.size.height = f, i.position.top = r.top + n.height - f), p - l > 0 ? (i.size.width = p, i.position.left = r.left - d) : (p = u - t.height, i.size.width = p, i.position.left = r.left + n.width - p))
        }
    }), e.ui.resizable, e.widget("ui.selectable", e.ui.mouse, {
        version: "1.11.1",
        options: {
            appendTo: "body",
            autoRefresh: !0,
            distance: 0,
            filter: "*",
            tolerance: "touch",
            selected: null,
            selecting: null,
            start: null,
            stop: null,
            unselected: null,
            unselecting: null
        },
        _create: function() {
            var t, i = this;
            this.element.addClass("ui-selectable"), this.dragged=!1, this.refresh = function() {
                t = e(i.options.filter, i.element[0]), t.addClass("ui-selectee"), t.each(function() {
                    var t = e(this), i = t.offset();
                    e.data(this, "selectable-item", {
                        element: this,
                        $element: t,
                        left: i.left,
                        top: i.top,
                        right: i.left + t.outerWidth(),
                        bottom: i.top + t.outerHeight(),
                        startselected: !1,
                        selected: t.hasClass("ui-selected"),
                        selecting: t.hasClass("ui-selecting"),
                        unselecting: t.hasClass("ui-unselecting")
                    })
                })
            }, this.refresh(), this.selectees = t.addClass("ui-selectee"), this._mouseInit(), this.helper = e("<div class='ui-selectable-helper'></div>")
        },
        _destroy: function() {
            this.selectees.removeClass("ui-selectee").removeData("selectable-item"), this.element.removeClass("ui-selectable ui-selectable-disabled"), this._mouseDestroy()
        },
        _mouseStart: function(t) {
            var i = this, s = this.options;
            this.opos = [t.pageX, t.pageY], this.options.disabled || (this.selectees = e(s.filter, this.element[0]), this._trigger("start", t), e(s.appendTo).append(this.helper), this.helper.css({
                left: t.pageX,
                top: t.pageY,
                width: 0,
                height: 0
            }), s.autoRefresh && this.refresh(), this.selectees.filter(".ui-selected").each(function() {
                var s = e.data(this, "selectable-item");
                s.startselected=!0, t.metaKey || t.ctrlKey || (s.$element.removeClass("ui-selected"), s.selected=!1, s.$element.addClass("ui-unselecting"), s.unselecting=!0, i._trigger("unselecting", t, {
                    unselecting: s.element
                }))
            }), e(t.target).parents().addBack().each(function() {
                var s, a = e.data(this, "selectable-item");
                return a ? (s=!t.metaKey&&!t.ctrlKey ||!a.$element.hasClass("ui-selected"), a.$element.removeClass(s ? "ui-unselecting" : "ui-selected").addClass(s ? "ui-selecting" : "ui-unselecting"), a.unselecting=!s, a.selecting = s, a.selected = s, s ? i._trigger("selecting", t, {
                    selecting: a.element
                }) : i._trigger("unselecting", t, {
                    unselecting: a.element
                }), !1) : void 0
            }))
        },
        _mouseDrag: function(t) {
            if (this.dragged=!0, !this.options.disabled) {
                var i, s = this, a = this.options, n = this.opos[0], r = this.opos[1], o = t.pageX, h = t.pageY;
                return n > o && (i = o, o = n, n = i), r > h && (i = h, h = r, r = i), this.helper.css({
                    left: n,
                    top: r,
                    width: o - n,
                    height: h - r
                }), this.selectees.each(function() {
                    var i = e.data(this, "selectable-item"), l=!1;
                    i && i.element !== s.element[0] && ("touch" === a.tolerance ? l=!(i.left > o || n > i.right || i.top > h || r > i.bottom) : "fit" === a.tolerance && (l = i.left > n && o > i.right && i.top > r && h > i.bottom), l ? (i.selected && (i.$element.removeClass("ui-selected"), i.selected=!1), i.unselecting && (i.$element.removeClass("ui-unselecting"), i.unselecting=!1), i.selecting || (i.$element.addClass("ui-selecting"), i.selecting=!0, s._trigger("selecting", t, {
                        selecting: i.element
                    }))) : (i.selecting && ((t.metaKey || t.ctrlKey) && i.startselected ? (i.$element.removeClass("ui-selecting"), i.selecting=!1, i.$element.addClass("ui-selected"), i.selected=!0) : (i.$element.removeClass("ui-selecting"), i.selecting=!1, i.startselected && (i.$element.addClass("ui-unselecting"), i.unselecting=!0), s._trigger("unselecting", t, {
                        unselecting: i.element
                    }))), i.selected && (t.metaKey || t.ctrlKey || i.startselected || (i.$element.removeClass("ui-selected"), i.selected=!1, i.$element.addClass("ui-unselecting"), i.unselecting=!0, s._trigger("unselecting", t, {
                        unselecting: i.element
                    })))))
                }), !1
            }
        },
        _mouseStop: function(t) {
            var i = this;
            return this.dragged=!1, e(".ui-unselecting", this.element[0]).each(function() {
                var s = e.data(this, "selectable-item");
                s.$element.removeClass("ui-unselecting"), s.unselecting=!1, s.startselected=!1, i._trigger("unselected", t, {
                    unselected: s.element
                })
            }), e(".ui-selecting", this.element[0]).each(function() {
                var s = e.data(this, "selectable-item");
                s.$element.removeClass("ui-selecting").addClass("ui-selected"), s.selecting=!1, s.selected=!0, s.startselected=!0, i._trigger("selected", t, {
                    selected: s.element
                })
            }), this._trigger("stop", t), this.helper.remove(), !1
        }
    }), e.widget("ui.sortable", e.ui.mouse, {
        version: "1.11.1",
        widgetEventPrefix: "sort",
        ready: !1,
        options: {
            appendTo: "parent",
            axis: !1,
            connectWith: !1,
            containment: !1,
            cursor: "auto",
            cursorAt: !1,
            dropOnEmpty: !0,
            forcePlaceholderSize: !1,
            forceHelperSize: !1,
            grid: !1,
            handle: !1,
            helper: "original",
            items: "> *",
            opacity: !1,
            placeholder: !1,
            revert: !1,
            scroll: !0,
            scrollSensitivity: 20,
            scrollSpeed: 20,
            scope: "default",
            tolerance: "intersect",
            zIndex: 1e3,
            activate: null,
            beforeStop: null,
            change: null,
            deactivate: null,
            out: null,
            over: null,
            receive: null,
            remove: null,
            sort: null,
            start: null,
            stop: null,
            update: null
        },
        _isOverAxis: function(e, t, i) {
            return e >= t && t + i > e
        },
        _isFloating: function(e) {
            return /left|right/.test(e.css("float")) || /inline|table-cell/.test(e.css("display"))
        },
        _create: function() {
            var e = this.options;
            this.containerCache = {}, this.element.addClass("ui-sortable"), this.refresh(), this.floating = this.items.length ? "x" === e.axis || this._isFloating(this.items[0].item) : !1, this.offset = this.element.offset(), this._mouseInit(), this._setHandleClassName(), this.ready=!0
        },
        _setOption: function(e, t) {
            this._super(e, t), "handle" === e && this._setHandleClassName()
        },
        _setHandleClassName: function() {
            this.element.find(".ui-sortable-handle").removeClass("ui-sortable-handle"), e.each(this.items, function() {
                (this.instance.options.handle ? this.item.find(this.instance.options.handle) : this.item).addClass("ui-sortable-handle")
            })
        },
        _destroy: function() {
            this.element.removeClass("ui-sortable ui-sortable-disabled").find(".ui-sortable-handle").removeClass("ui-sortable-handle"), this._mouseDestroy();
            for (var e = this.items.length - 1; e >= 0; e--)
                this.items[e].item.removeData(this.widgetName + "-item");
            return this
        },
        _mouseCapture: function(t, i) {
            var s = null, a=!1, n = this;
            return this.reverting?!1 : this.options.disabled || "static" === this.options.type?!1 : (this._refreshItems(t), e(t.target).parents().each(function() {
                return e.data(this, n.widgetName + "-item") === n ? (s = e(this), !1) : void 0
            }), e.data(t.target, n.widgetName + "-item") === n && (s = e(t.target)), s?!this.options.handle || i || (e(this.options.handle, s).find("*").addBack().each(function() {
                this === t.target && (a=!0)
            }), a) ? (this.currentItem = s, this._removeCurrentsFromItems(), !0) : !1 : !1)
        },
        _mouseStart: function(t, i, s) {
            var a, n, r = this.options;
            if (this.currentContainer = this, this.refreshPositions(), this.helper = this._createHelper(t), this._cacheHelperProportions(), this._cacheMargins(), this.scrollParent = this.helper.scrollParent(), this.offset = this.currentItem.offset(), this.offset = {
                top: this.offset.top - this.margins.top,
                left: this.offset.left - this.margins.left
            }, e.extend(this.offset, {
                click: {
                    left: t.pageX - this.offset.left,
                    top: t.pageY - this.offset.top
                },
                parent: this._getParentOffset(),
                relative: this._getRelativeOffset()
            }), this.helper.css("position", "absolute"), this.cssPosition = this.helper.css("position"), this.originalPosition = this._generatePosition(t), this.originalPageX = t.pageX, this.originalPageY = t.pageY, r.cursorAt && this._adjustOffsetFromHelper(r.cursorAt), this.domPosition = {
                prev: this.currentItem.prev()[0],
                parent: this.currentItem.parent()[0]
            }, this.helper[0] !== this.currentItem[0] && this.currentItem.hide(), this._createPlaceholder(), r.containment && this._setContainment(), r.cursor && "auto" !== r.cursor && (n = this.document.find("body"), this.storedCursor = n.css("cursor"), n.css("cursor", r.cursor), this.storedStylesheet = e("<style>*{ cursor: " + r.cursor + " !important; }</style>").appendTo(n)), r.opacity && (this.helper.css("opacity") && (this._storedOpacity = this.helper.css("opacity")), this.helper.css("opacity", r.opacity)), r.zIndex && (this.helper.css("zIndex") && (this._storedZIndex = this.helper.css("zIndex")), this.helper.css("zIndex", r.zIndex)), this.scrollParent[0] !== document && "HTML" !== this.scrollParent[0].tagName && (this.overflowOffset = this.scrollParent.offset()), this._trigger("start", t, this._uiHash()), this._preserveHelperProportions || this._cacheHelperProportions(), !s)
                for (a = this.containers.length - 1; a >= 0; a--)
                    this.containers[a]._trigger("activate", t, this._uiHash(this));
            return e.ui.ddmanager && (e.ui.ddmanager.current = this), e.ui.ddmanager&&!r.dropBehaviour && e.ui.ddmanager.prepareOffsets(this, t), this.dragging=!0, this.helper.addClass("ui-sortable-helper"), this._mouseDrag(t), !0
        },
        _mouseDrag: function(t) {
            var i, s, a, n, r = this.options, o=!1;
            for (this.position = this._generatePosition(t), this.positionAbs = this._convertPositionTo("absolute"), this.lastPositionAbs || (this.lastPositionAbs = this.positionAbs), this.options.scroll && (this.scrollParent[0] !== document && "HTML" !== this.scrollParent[0].tagName ? (this.overflowOffset.top + this.scrollParent[0].offsetHeight - t.pageY < r.scrollSensitivity ? this.scrollParent[0].scrollTop = o = this.scrollParent[0].scrollTop + r.scrollSpeed : t.pageY - this.overflowOffset.top < r.scrollSensitivity && (this.scrollParent[0].scrollTop = o = this.scrollParent[0].scrollTop - r.scrollSpeed), this.overflowOffset.left + this.scrollParent[0].offsetWidth - t.pageX < r.scrollSensitivity ? this.scrollParent[0].scrollLeft = o = this.scrollParent[0].scrollLeft + r.scrollSpeed : t.pageX - this.overflowOffset.left < r.scrollSensitivity && (this.scrollParent[0].scrollLeft = o = this.scrollParent[0].scrollLeft - r.scrollSpeed)) : (t.pageY - e(document).scrollTop() < r.scrollSensitivity ? o = e(document).scrollTop(e(document).scrollTop() - r.scrollSpeed) : e(window).height() - (t.pageY - e(document).scrollTop()) < r.scrollSensitivity && (o = e(document).scrollTop(e(document).scrollTop() + r.scrollSpeed)), t.pageX - e(document).scrollLeft() < r.scrollSensitivity ? o = e(document).scrollLeft(e(document).scrollLeft() - r.scrollSpeed) : e(window).width() - (t.pageX - e(document).scrollLeft()) < r.scrollSensitivity && (o = e(document).scrollLeft(e(document).scrollLeft() + r.scrollSpeed))), o!==!1 && e.ui.ddmanager&&!r.dropBehaviour && e.ui.ddmanager.prepareOffsets(this, t)), this.positionAbs = this._convertPositionTo("absolute"), this.options.axis && "y" === this.options.axis || (this.helper[0].style.left = this.position.left + "px"), this.options.axis && "x" === this.options.axis || (this.helper[0].style.top = this.position.top + "px"), i = this.items.length - 1; i >= 0; i--)
                if (s = this.items[i], a = s.item[0], n = this._intersectsWithPointer(s), n && s.instance === this.currentContainer && a !== this.currentItem[0] && this.placeholder[1 === n ? "next": "prev"]()[0] !== a&&!e.contains(this.placeholder[0], a) && ("semi-dynamic" === this.options.type?!e.contains(this.element[0], a) : !0)) {
                    if (this.direction = 1 === n ? "down" : "up", "pointer" !== this.options.tolerance&&!this._intersectsWithSides(s))
                        break;
                        this._rearrange(t, s), this._trigger("change", t, this._uiHash());
                        break
                }
            return this._contactContainers(t), e.ui.ddmanager && e.ui.ddmanager.drag(this, t), this._trigger("sort", t, this._uiHash()), this.lastPositionAbs = this.positionAbs, !1
        },
        _mouseStop: function(t, i) {
            if (t) {
                if (e.ui.ddmanager&&!this.options.dropBehaviour && e.ui.ddmanager.drop(this, t), this.options.revert) {
                    var s = this, a = this.placeholder.offset(), n = this.options.axis, r = {};
                    n && "x" !== n || (r.left = a.left - this.offset.parent.left - this.margins.left + (this.offsetParent[0] === document.body ? 0 : this.offsetParent[0].scrollLeft)), n && "y" !== n || (r.top = a.top - this.offset.parent.top - this.margins.top + (this.offsetParent[0] === document.body ? 0 : this.offsetParent[0].scrollTop)), this.reverting=!0, e(this.helper).animate(r, parseInt(this.options.revert, 10) || 500, function() {
                        s._clear(t)
                    })
                } else 
                    this._clear(t, i);
                return !1
            }
        },
        cancel: function() {
            if (this.dragging) {
                this._mouseUp({
                    target: null
                }), "original" === this.options.helper ? this.currentItem.css(this._storedCSS).removeClass("ui-sortable-helper") : this.currentItem.show();
                for (var t = this.containers.length - 1; t >= 0; t--)
                    this.containers[t]._trigger("deactivate", null, this._uiHash(this)), this.containers[t].containerCache.over && (this.containers[t]._trigger("out", null, this._uiHash(this)), this.containers[t].containerCache.over = 0)
            }
            return this.placeholder && (this.placeholder[0].parentNode && this.placeholder[0].parentNode.removeChild(this.placeholder[0]), "original" !== this.options.helper && this.helper && this.helper[0].parentNode && this.helper.remove(), e.extend(this, {
                helper: null,
                dragging: !1,
                reverting: !1,
                _noFinalSort: null
            }), this.domPosition.prev ? e(this.domPosition.prev).after(this.currentItem) : e(this.domPosition.parent).prepend(this.currentItem)), this
        },
        serialize: function(t) {
            var i = this._getItemsAsjQuery(t && t.connected), s = [];
            return t = t || {}, e(i).each(function() {
                var i = (e(t.item || this).attr(t.attribute || "id") || "").match(t.expression || /(.+)[\-=_](.+)/);
                i && s.push((t.key || i[1] + "[]") + "=" + (t.key && t.expression ? i[1] : i[2]))
            }), !s.length && t.key && s.push(t.key + "="), s.join("&")
        },
        toArray: function(t) {
            var i = this._getItemsAsjQuery(t && t.connected), s = [];
            return t = t || {}, i.each(function() {
                s.push(e(t.item || this).attr(t.attribute || "id") || "")
            }), s
        },
        _intersectsWith: function(e) {
            var t = this.positionAbs.left, i = t + this.helperProportions.width, s = this.positionAbs.top, a = s + this.helperProportions.height, n = e.left, r = n + e.width, o = e.top, h = o + e.height, l = this.offset.click.top, u = this.offset.click.left, d = "x" === this.options.axis || s + l > o && h > s + l, c = "y" === this.options.axis || t + u > n && r > t + u, p = d && c;
            return "pointer" === this.options.tolerance || this.options.forcePointerForContainers || "pointer" !== this.options.tolerance && this.helperProportions[this.floating ? "width": "height"] > e[this.floating ? "width": "height"] ? p : t + this.helperProportions.width / 2 > n && r > i - this.helperProportions.width / 2 && s + this.helperProportions.height / 2 > o && h > a - this.helperProportions.height / 2
        },
        _intersectsWithPointer: function(e) {
            var t = "x" === this.options.axis || this._isOverAxis(this.positionAbs.top + this.offset.click.top, e.top, e.height), i = "y" === this.options.axis || this._isOverAxis(this.positionAbs.left + this.offset.click.left, e.left, e.width), s = t && i, a = this._getDragVerticalDirection(), n = this._getDragHorizontalDirection();
            return s ? this.floating ? n && "right" === n || "down" === a ? 2 : 1 : a && ("down" === a ? 2 : 1) : !1
        },
        _intersectsWithSides: function(e) {
            var t = this._isOverAxis(this.positionAbs.top + this.offset.click.top, e.top + e.height / 2, e.height), i = this._isOverAxis(this.positionAbs.left + this.offset.click.left, e.left + e.width / 2, e.width), s = this._getDragVerticalDirection(), a = this._getDragHorizontalDirection();
            return this.floating && a ? "right" === a && i || "left" === a&&!i : s && ("down" === s && t || "up" === s&&!t)
        },
        _getDragVerticalDirection: function() {
            var e = this.positionAbs.top - this.lastPositionAbs.top;
            return 0 !== e && (e > 0 ? "down" : "up")
        },
        _getDragHorizontalDirection: function() {
            var e = this.positionAbs.left - this.lastPositionAbs.left;
            return 0 !== e && (e > 0 ? "right" : "left")
        },
        refresh: function(e) {
            return this._refreshItems(e), this._setHandleClassName(), this.refreshPositions(), this
        },
        _connectWith: function() {
            var e = this.options;
            return e.connectWith.constructor === String ? [e.connectWith] : e.connectWith
        },
        _getItemsAsjQuery: function(t) {
            function i() {
                o.push(this)
            }
            var s, a, n, r, o = [], h = [], l = this._connectWith();
            if (l && t)
                for (s = l.length - 1; s >= 0; s--)
                    for (n = e(l[s]), a = n.length - 1; a >= 0; a--)
                        r = e.data(n[a], this.widgetFullName), r && r !== this&&!r.options.disabled && h.push([e.isFunction(r.options.items) ? r.options.items.call(r.element): e(r.options.items, r.element).not(".ui-sortable-helper").not(".ui-sortable-placeholder"), r]);
            for (h.push([e.isFunction(this.options.items) ? this.options.items.call(this.element, null, {
                options: this.options,
                item: this.currentItem
            }): e(this.options.items, this.element).not(".ui-sortable-helper").not(".ui-sortable-placeholder"), this]), s = h.length - 1; s >= 0; s--)
                h[s][0].each(i);
            return e(o)
        },
        _removeCurrentsFromItems: function() {
            var t = this.currentItem.find(":data(" + this.widgetName + "-item)");
            this.items = e.grep(this.items, function(e) {
                for (var i = 0; t.length > i; i++)
                    if (t[i] === e.item[0])
                        return !1;
                return !0
            })
        },
        _refreshItems: function(t) {
            this.items = [], this.containers = [this];
            var i, s, a, n, r, o, h, l, u = this.items, d = [[e.isFunction(this.options.items) ? this.options.items.call(this.element[0], t, {
                item: this.currentItem
            }): e(this.options.items, this.element), this]], c = this._connectWith();
            if (c && this.ready)
                for (i = c.length - 1; i >= 0; i--)
                    for (a = e(c[i]), s = a.length - 1; s >= 0; s--)
                        n = e.data(a[s], this.widgetFullName), n && n !== this&&!n.options.disabled && (d.push([e.isFunction(n.options.items) ? n.options.items.call(n.element[0], t, {
                            item: this.currentItem
                        }): e(n.options.items, n.element), n]), this.containers.push(n));
            for (i = d.length - 1; i >= 0; i--)
                for (r = d[i][1], o = d[i][0], s = 0, l = o.length; l > s; s++)
                    h = e(o[s]), h.data(this.widgetName + "-item", r), u.push({
                        item: h,
                        instance: r,
                        width: 0,
                        height: 0,
                        left: 0,
                        top: 0
                    })
        },
        refreshPositions: function(t) {
            this.offsetParent && this.helper && (this.offset.parent = this._getParentOffset());
            var i, s, a, n;
            for (i = this.items.length - 1; i >= 0; i--)
                s = this.items[i], s.instance !== this.currentContainer && this.currentContainer && s.item[0] !== this.currentItem[0] || (a = this.options.toleranceElement ? e(this.options.toleranceElement, s.item) : s.item, t || (s.width = a.outerWidth(), s.height = a.outerHeight()), n = a.offset(), s.left = n.left, s.top = n.top);
            if (this.options.custom && this.options.custom.refreshContainers)
                this.options.custom.refreshContainers.call(this);
            else 
                for (i = this.containers.length - 1; i >= 0; i--)
                    n = this.containers[i].element.offset(), this.containers[i].containerCache.left = n.left, this.containers[i].containerCache.top = n.top, this.containers[i].containerCache.width = this.containers[i].element.outerWidth(), this.containers[i].containerCache.height = this.containers[i].element.outerHeight();
            return this
        },
        _createPlaceholder: function(t) {
            t = t || this;
            var i, s = t.options;
            s.placeholder && s.placeholder.constructor !== String || (i = s.placeholder, s.placeholder = {
                element: function() {
                    var s = t.currentItem[0].nodeName.toLowerCase(), a = e("<" + s + ">", t.document[0]).addClass(i || t.currentItem[0].className + " ui-sortable-placeholder").removeClass("ui-sortable-helper");
                    return "tr" === s ? t.currentItem.children().each(function() {
                        e("<td>&#160;</td>", t.document[0]).attr("colspan", e(this).attr("colspan") || 1).appendTo(a)
                    }) : "img" === s && a.attr("src", t.currentItem.attr("src")), i || a.css("visibility", "hidden"), a
                },
                update: function(e, a) {
                    (!i || s.forcePlaceholderSize) && (a.height() || a.height(t.currentItem.innerHeight() - parseInt(t.currentItem.css("paddingTop") || 0, 10) - parseInt(t.currentItem.css("paddingBottom") || 0, 10)), a.width() || a.width(t.currentItem.innerWidth() - parseInt(t.currentItem.css("paddingLeft") || 0, 10) - parseInt(t.currentItem.css("paddingRight") || 0, 10)))
                }
            }), t.placeholder = e(s.placeholder.element.call(t.element, t.currentItem)), t.currentItem.after(t.placeholder), s.placeholder.update(t, t.placeholder)
        },
        _contactContainers: function(t) {
            var i, s, a, n, r, o, h, l, u, d, c = null, p = null;
            for (i = this.containers.length - 1; i >= 0; i--)
                if (!e.contains(this.currentItem[0], this.containers[i].element[0]))
                    if (this._intersectsWith(this.containers[i].containerCache)) {
                        if (c && e.contains(this.containers[i].element[0], c.element[0]))
                            continue;
                            c = this.containers[i], p = i
                    } else 
                        this.containers[i].containerCache.over && (this.containers[i]._trigger("out", t, this._uiHash(this)), this.containers[i].containerCache.over = 0);
            if (c)
                if (1 === this.containers.length)
                    this.containers[p].containerCache.over || (this.containers[p]._trigger("over", t, this._uiHash(this)), this.containers[p].containerCache.over = 1);
                else {
                    for (a = 1e4, n = null, u = c.floating || this._isFloating(this.currentItem), r = u ? "left" : "top", o = u ? "width" : "height", d = u ? "clientX" : "clientY", s = this.items.length - 1; s >= 0; s--)
                        e.contains(this.containers[p].element[0], this.items[s].item[0]) && this.items[s].item[0] !== this.currentItem[0] && (h = this.items[s].item.offset()[r], l=!1, t[d] - h > this.items[s][o] / 2 && (l=!0), a > Math.abs(t[d] - h) && (a = Math.abs(t[d] - h), n = this.items[s], this.direction = l ? "up" : "down"));
                        if (!n&&!this.options.dropOnEmpty)
                            return;
                            if (this.currentContainer === this.containers[p])
                                return;
                                n ? this._rearrange(t, n, null, !0) : this._rearrange(t, null, this.containers[p].element, !0), this._trigger("change", t, this._uiHash()), this.containers[p]._trigger("change", t, this._uiHash(this)), this.currentContainer = this.containers[p], this.options.placeholder.update(this.currentContainer, this.placeholder), this.containers[p]._trigger("over", t, this._uiHash(this)), this.containers[p].containerCache.over = 1
                }
        },
        _createHelper: function(t) {
            var i = this.options, s = e.isFunction(i.helper) ? e(i.helper.apply(this.element[0], [t, this.currentItem])): "clone" === i.helper ? this.currentItem.clone(): this.currentItem;
            return s.parents("body").length || e("parent" !== i.appendTo ? i.appendTo : this.currentItem[0].parentNode)[0].appendChild(s[0]), s[0] === this.currentItem[0] && (this._storedCSS = {
                width: this.currentItem[0].style.width,
                height: this.currentItem[0].style.height,
                position: this.currentItem.css("position"),
                top: this.currentItem.css("top"),
                left: this.currentItem.css("left")
            }), (!s[0].style.width || i.forceHelperSize) && s.width(this.currentItem.width()), (!s[0].style.height || i.forceHelperSize) && s.height(this.currentItem.height()), s
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
                var e = this.currentItem.position();
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
                left: parseInt(this.currentItem.css("marginLeft"), 10) || 0,
                top: parseInt(this.currentItem.css("marginTop"), 10) || 0
            }
        },
        _cacheHelperProportions: function() {
            this.helperProportions = {
                width: this.helper.outerWidth(),
                height: this.helper.outerHeight()
            }
        },
        _setContainment: function() {
            var t, i, s, a = this.options;
            "parent" === a.containment && (a.containment = this.helper[0].parentNode), ("document" === a.containment || "window" === a.containment) && (this.containment = [0 - this.offset.relative.left - this.offset.parent.left, 0 - this.offset.relative.top - this.offset.parent.top, e("document" === a.containment ? document : window).width() - this.helperProportions.width - this.margins.left, (e("document" === a.containment ? document : window).height() || document.body.parentNode.scrollHeight) - this.helperProportions.height - this.margins.top]), /^(document|window|parent)$/.test(a.containment) || (t = e(a.containment)[0], i = e(a.containment).offset(), s = "hidden" !== e(t).css("overflow"), this.containment = [i.left + (parseInt(e(t).css("borderLeftWidth"), 10) || 0) + (parseInt(e(t).css("paddingLeft"), 10) || 0) - this.margins.left, i.top + (parseInt(e(t).css("borderTopWidth"), 10) || 0) + (parseInt(e(t).css("paddingTop"), 10) || 0) - this.margins.top, i.left + (s ? Math.max(t.scrollWidth, t.offsetWidth) : t.offsetWidth) - (parseInt(e(t).css("borderLeftWidth"), 10) || 0) - (parseInt(e(t).css("paddingRight"), 10) || 0) - this.helperProportions.width - this.margins.left, i.top + (s ? Math.max(t.scrollHeight, t.offsetHeight) : t.offsetHeight) - (parseInt(e(t).css("borderTopWidth"), 10) || 0) - (parseInt(e(t).css("paddingBottom"), 10) || 0) - this.helperProportions.height - this.margins.top])
        },
        _convertPositionTo: function(t, i) {
            i || (i = this.position);
            var s = "absolute" === t ? 1: - 1, a = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && e.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent, n = /(html|body)/i.test(a[0].tagName);
            return {
                top: i.top + this.offset.relative.top * s + this.offset.parent.top * s - ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : n ? 0 : a.scrollTop()) * s,
                left: i.left + this.offset.relative.left * s + this.offset.parent.left * s - ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft(): n ? 0: a.scrollLeft()) * s
            }
        },
        _generatePosition: function(t) {
            var i, s, a = this.options, n = t.pageX, r = t.pageY, o = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && e.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent, h = /(html|body)/i.test(o[0].tagName);
            return "relative" !== this.cssPosition || this.scrollParent[0] !== document && this.scrollParent[0] !== this.offsetParent[0] || (this.offset.relative = this._getRelativeOffset()), this.originalPosition && (this.containment && (t.pageX - this.offset.click.left < this.containment[0] && (n = this.containment[0] + this.offset.click.left), t.pageY - this.offset.click.top < this.containment[1] && (r = this.containment[1] + this.offset.click.top), t.pageX - this.offset.click.left > this.containment[2] && (n = this.containment[2] + this.offset.click.left), t.pageY - this.offset.click.top > this.containment[3] && (r = this.containment[3] + this.offset.click.top)), a.grid && (i = this.originalPageY + Math.round((r - this.originalPageY) / a.grid[1]) * a.grid[1], r = this.containment ? i - this.offset.click.top >= this.containment[1] && i - this.offset.click.top <= this.containment[3] ? i : i - this.offset.click.top >= this.containment[1] ? i - a.grid[1] : i + a.grid[1] : i, s = this.originalPageX + Math.round((n - this.originalPageX) / a.grid[0]) * a.grid[0], n = this.containment ? s - this.offset.click.left >= this.containment[0] && s - this.offset.click.left <= this.containment[2] ? s : s - this.offset.click.left >= this.containment[0] ? s - a.grid[0] : s + a.grid[0] : s)), {
                top: r - this.offset.click.top - this.offset.relative.top - this.offset.parent.top + ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : h ? 0 : o.scrollTop()),
                left: n - this.offset.click.left - this.offset.relative.left - this.offset.parent.left + ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft(): h ? 0: o.scrollLeft())
            }
        },
        _rearrange: function(e, t, i, s) {
            i ? i[0].appendChild(this.placeholder[0]) : t.item[0].parentNode.insertBefore(this.placeholder[0], "down" === this.direction ? t.item[0] : t.item[0].nextSibling), this.counter = this.counter?++this.counter : 1;
            var a = this.counter;
            this._delay(function() {
                a === this.counter && this.refreshPositions(!s)
            })
        },
        _clear: function(e, t) {
            function i(e, t, i) {
                return function(s) {
                    i._trigger(e, s, t._uiHash(t))
                }
            }
            this.reverting=!1;
            var s, a = [];
            if (!this._noFinalSort && this.currentItem.parent().length && this.placeholder.before(this.currentItem), this._noFinalSort = null, this.helper[0] === this.currentItem[0]) {
                for (s in this._storedCSS)("auto" === this._storedCSS[s] || "static" === this._storedCSS[s]) 
                    && (this._storedCSS[s] = "");
                this.currentItem.css(this._storedCSS).removeClass("ui-sortable-helper")
            } else 
                this.currentItem.show();
            for (this.fromOutside&&!t && a.push(function(e) {
                this._trigger("receive", e, this._uiHash(this.fromOutside))
            }), !this.fromOutside && this.domPosition.prev === this.currentItem.prev().not(".ui-sortable-helper")[0] && this.domPosition.parent === this.currentItem.parent()[0] || t || a.push(function(e) {
                this._trigger("update", e, this._uiHash())
            }), this !== this.currentContainer && (t || (a.push(function(e) {
                this._trigger("remove", e, this._uiHash())
            }), a.push(function(e) {
                return function(t) {
                    e._trigger("receive", t, this._uiHash(this))
                }
            }.call(this, this.currentContainer)), a.push(function(e) {
                return function(t) {
                    e._trigger("update", t, this._uiHash(this))
                }
            }.call(this, this.currentContainer)))), s = this.containers.length - 1; s >= 0; s--)
                t || a.push(i("deactivate", this, this.containers[s])), this.containers[s].containerCache.over && (a.push(i("out", this, this.containers[s])), this.containers[s].containerCache.over = 0);
            if (this.storedCursor && (this.document.find("body").css("cursor", this.storedCursor), this.storedStylesheet.remove()), this._storedOpacity && this.helper.css("opacity", this._storedOpacity), this._storedZIndex && this.helper.css("zIndex", "auto" === this._storedZIndex ? "" : this._storedZIndex), this.dragging=!1, this.cancelHelperRemoval) {
                if (!t) {
                    for (this._trigger("beforeStop", e, this._uiHash()), s = 0; a.length > s; s++)
                        a[s].call(this, e);
                    this._trigger("stop", e, this._uiHash())
                }
                return this.fromOutside=!1, !1
            }
            if (t || this._trigger("beforeStop", e, this._uiHash()), this.placeholder[0].parentNode.removeChild(this.placeholder[0]), this.helper[0] !== this.currentItem[0] && this.helper.remove(), this.helper = null, !t) {
                for (s = 0; a.length > s; s++)
                    a[s].call(this, e);
                this._trigger("stop", e, this._uiHash())
            }
            return this.fromOutside=!1, !0
        },
        _trigger: function() {
            e.Widget.prototype._trigger.apply(this, arguments)===!1 && this.cancel()
        },
        _uiHash: function(t) {
            var i = t || this;
            return {
                helper: i.helper,
                placeholder: i.placeholder || e([]),
                position: i.position,
                originalPosition: i.originalPosition,
                offset: i.positionAbs,
                item: i.currentItem,
                sender: t ? t.element: null
            }
        }
    }), e.widget("ui.accordion", {
        version: "1.11.1",
        options: {
            active: 0,
            animate: {},
            collapsible: !1,
            event: "click",
            header: "> li > :first-child,> :not(li):even",
            heightStyle: "auto",
            icons: {
                activeHeader: "ui-icon-triangle-1-s",
                header: "ui-icon-triangle-1-e"
            },
            activate: null,
            beforeActivate: null
        },
        hideProps: {
            borderTopWidth: "hide",
            borderBottomWidth: "hide",
            paddingTop: "hide",
            paddingBottom: "hide",
            height: "hide"
        },
        showProps: {
            borderTopWidth: "show",
            borderBottomWidth: "show",
            paddingTop: "show",
            paddingBottom: "show",
            height: "show"
        },
        _create: function() {
            var t = this.options;
            this.prevShow = this.prevHide = e(), this.element.addClass("ui-accordion ui-widget ui-helper-reset").attr("role", "tablist"), t.collapsible || t.active!==!1 && null != t.active || (t.active = 0), this._processPanels(), 0 > t.active && (t.active += this.headers.length), this._refresh()
        },
        _getCreateEventData: function() {
            return {
                header: this.active,
                panel: this.active.length ? this.active.next(): e()
            }
        },
        _createIcons: function() {
            var t = this.options.icons;
            t && (e("<span>").addClass("ui-accordion-header-icon ui-icon " + t.header).prependTo(this.headers), this.active.children(".ui-accordion-header-icon").removeClass(t.header).addClass(t.activeHeader), this.headers.addClass("ui-accordion-icons"))
        },
        _destroyIcons: function() {
            this.headers.removeClass("ui-accordion-icons").children(".ui-accordion-header-icon").remove()
        },
        _destroy: function() {
            var e;
            this.element.removeClass("ui-accordion ui-widget ui-helper-reset").removeAttr("role"), this.headers.removeClass("ui-accordion-header ui-accordion-header-active ui-state-default ui-corner-all ui-state-active ui-state-disabled ui-corner-top").removeAttr("role").removeAttr("aria-expanded").removeAttr("aria-selected").removeAttr("aria-controls").removeAttr("tabIndex").removeUniqueId(), this._destroyIcons(), e = this.headers.next().removeClass("ui-helper-reset ui-widget-content ui-corner-bottom ui-accordion-content ui-accordion-content-active ui-state-disabled").css("display", "").removeAttr("role").removeAttr("aria-hidden").removeAttr("aria-labelledby").removeUniqueId(), "content" !== this.options.heightStyle && e.css("height", "")
        },
        _setOption: function(e, t) {
            return "active" === e ? (this._activate(t), void 0) : ("event" === e && (this.options.event && this._off(this.headers, this.options.event), this._setupEvents(t)), this._super(e, t), "collapsible" !== e || t || this.options.active!==!1 || this._activate(0), "icons" === e && (this._destroyIcons(), t && this._createIcons()), "disabled" === e && (this.element.toggleClass("ui-state-disabled", !!t).attr("aria-disabled", t), this.headers.add(this.headers.next()).toggleClass("ui-state-disabled", !!t)), void 0)
        },
        _keydown: function(t) {
            if (!t.altKey&&!t.ctrlKey) {
                var i = e.ui.keyCode, s = this.headers.length, a = this.headers.index(t.target), n=!1;
                switch (t.keyCode) {
                case i.RIGHT:
                case i.DOWN:
                    n = this.headers[(a + 1)%s];
                    break;
                case i.LEFT:
                case i.UP:
                    n = this.headers[(a - 1 + s)%s];
                    break;
                case i.SPACE:
                case i.ENTER:
                    this._eventHandler(t);
                    break;
                case i.HOME:
                    n = this.headers[0];
                    break;
                case i.END:
                    n = this.headers[s - 1]
                }
                n && (e(t.target).attr("tabIndex", - 1), e(n).attr("tabIndex", 0), n.focus(), t.preventDefault())
            }
        },
        _panelKeyDown: function(t) {
            t.keyCode === e.ui.keyCode.UP && t.ctrlKey && e(t.currentTarget).prev().focus()
        },
        refresh: function() {
            var t = this.options;
            this._processPanels(), t.active===!1 && t.collapsible===!0 ||!this.headers.length ? (t.active=!1, this.active = e()) : t.active===!1 ? this._activate(0) : this.active.length&&!e.contains(this.element[0], this.active[0]) ? this.headers.length === this.headers.find(".ui-state-disabled").length ? (t.active=!1, this.active = e()) : this._activate(Math.max(0, t.active - 1)) : t.active = this.headers.index(this.active), this._destroyIcons(), this._refresh()
        },
        _processPanels: function() {
            this.headers = this.element.find(this.options.header).addClass("ui-accordion-header ui-state-default ui-corner-all"), this.headers.next().addClass("ui-accordion-content ui-helper-reset ui-widget-content ui-corner-bottom").filter(":not(.ui-accordion-content-active)").hide()
        },
        _refresh: function() {
            var t, i = this.options, s = i.heightStyle, a = this.element.parent();
            this.active = this._findActive(i.active).addClass("ui-accordion-header-active ui-state-active ui-corner-top").removeClass("ui-corner-all"), this.active.next().addClass("ui-accordion-content-active").show(), this.headers.attr("role", "tab").each(function() {
                var t = e(this), i = t.uniqueId().attr("id"), s = t.next(), a = s.uniqueId().attr("id");
                t.attr("aria-controls", a), s.attr("aria-labelledby", i)
            }).next().attr("role", "tabpanel"), this.headers.not(this.active).attr({
                "aria-selected": "false",
                "aria-expanded": "false",
                tabIndex: - 1
            }).next().attr({
                "aria-hidden": "true"
            }).hide(), this.active.length ? this.active.attr({
                "aria-selected": "true",
                "aria-expanded": "true",
                tabIndex: 0
            }).next().attr({
                "aria-hidden": "false"
            }) : this.headers.eq(0).attr("tabIndex", 0), this._createIcons(), this._setupEvents(i.event), "fill" === s ? (t = a.height(), this.element.siblings(":visible").each(function() {
                var i = e(this), s = i.css("position");
                "absolute" !== s && "fixed" !== s && (t -= i.outerHeight(!0))
            }), this.headers.each(function() {
                t -= e(this).outerHeight(!0)
            }), this.headers.next().each(function() {
                e(this).height(Math.max(0, t - e(this).innerHeight() + e(this).height()))
            }).css("overflow", "auto")) : "auto" === s && (t = 0, this.headers.next().each(function() {
                t = Math.max(t, e(this).css("height", "").height())
            }).height(t))
        },
        _activate: function(t) {
            var i = this._findActive(t)[0];
            i !== this.active[0] && (i = i || this.active[0], this._eventHandler({
                target: i,
                currentTarget: i,
                preventDefault: e.noop
            }))
        },
        _findActive: function(t) {
            return "number" == typeof t ? this.headers.eq(t) : e()
        },
        _setupEvents: function(t) {
            var i = {
                keydown: "_keydown"
            };
            t && e.each(t.split(" "), function(e, t) {
                i[t] = "_eventHandler"
            }), this._off(this.headers.add(this.headers.next())), this._on(this.headers, i), this._on(this.headers.next(), {
                keydown: "_panelKeyDown"
            }), this._hoverable(this.headers), this._focusable(this.headers)
        },
        _eventHandler: function(t) {
            var i = this.options, s = this.active, a = e(t.currentTarget), n = a[0] === s[0], r = n && i.collapsible, o = r ? e(): a.next(), h = s.next(), l = {
                oldHeader: s,
                oldPanel: h,
                newHeader: r ? e(): a,
                newPanel: o
            };
            t.preventDefault(), n&&!i.collapsible || this._trigger("beforeActivate", t, l)===!1 || (i.active = r?!1 : this.headers.index(a), this.active = n ? e() : a, this._toggle(l), s.removeClass("ui-accordion-header-active ui-state-active"), i.icons && s.children(".ui-accordion-header-icon").removeClass(i.icons.activeHeader).addClass(i.icons.header), n || (a.removeClass("ui-corner-all").addClass("ui-accordion-header-active ui-state-active ui-corner-top"), i.icons && a.children(".ui-accordion-header-icon").removeClass(i.icons.header).addClass(i.icons.activeHeader), a.next().addClass("ui-accordion-content-active")))
        },
        _toggle: function(t) {
            var i = t.newPanel, s = this.prevShow.length ? this.prevShow: t.oldPanel;
            this.prevShow.add(this.prevHide).stop(!0, !0), this.prevShow = i, this.prevHide = s, this.options.animate ? this._animate(i, s, t) : (s.hide(), i.show(), this._toggleComplete(t)), s.attr({
                "aria-hidden": "true"
            }), s.prev().attr("aria-selected", "false"), i.length && s.length ? s.prev().attr({
                tabIndex: - 1,
                "aria-expanded": "false"
            }) : i.length && this.headers.filter(function() {
                return 0 === e(this).attr("tabIndex")
            }).attr("tabIndex", - 1), i.attr("aria-hidden", "false").prev().attr({
                "aria-selected": "true",
                tabIndex: 0,
                "aria-expanded": "true"
            })
        },
        _animate: function(e, t, i) {
            var s, a, n, r = this, o = 0, h = e.length && (!t.length || e.index() < t.index()), l = this.options.animate || {}, u = h && l.down || l, d = function() {
                r._toggleComplete(i)
            };
            return "number" == typeof u && (n = u), "string" == typeof u && (a = u), a = a || u.easing || l.easing, n = n || u.duration || l.duration, t.length ? e.length ? (s = e.show().outerHeight(), t.animate(this.hideProps, {
                duration: n,
                easing: a,
                step: function(e, t) {
                    t.now = Math.round(e)
                }
            }), e.hide().animate(this.showProps, {
                duration: n,
                easing: a,
                complete: d,
                step: function(e, i) {
                    i.now = Math.round(e), "height" !== i.prop ? o += i.now : "content" !== r.options.heightStyle && (i.now = Math.round(s - t.outerHeight() - o), o = 0)
                }
            }), void 0) : t.animate(this.hideProps, n, a, d) : e.animate(this.showProps, n, a, d)
        },
        _toggleComplete: function(e) {
            var t = e.oldPanel;
            t.removeClass("ui-accordion-content-active").prev().removeClass("ui-corner-top").addClass("ui-corner-all"), t.length && (t.parent()[0].className = t.parent()[0].className), this._trigger("activate", null, e)
        }
    }), e.widget("ui.menu", {
        version: "1.11.1",
        defaultElement: "<ul>",
        delay: 300,
        options: {
            icons: {
                submenu: "ui-icon-carat-1-e"
            },
            items: "> *",
            menus: "ul",
            position: {
                my: "left-1 top",
                at: "right top"
            },
            role: "menu",
            blur: null,
            focus: null,
            select: null
        },
        _create: function() {
            this.activeMenu = this.element, this.mouseHandled=!1, this.element.uniqueId().addClass("ui-menu ui-widget ui-widget-content").toggleClass("ui-menu-icons", !!this.element.find(".ui-icon").length).attr({
                role: this.options.role,
                tabIndex: 0
            }), this.options.disabled && this.element.addClass("ui-state-disabled").attr("aria-disabled", "true"), this._on({
                "mousedown .ui-menu-item": function(e) {
                    e.preventDefault()
                },
                "click .ui-menu-item": function(t) {
                    var i = e(t.target);
                    !this.mouseHandled && i.not(".ui-state-disabled").length && (this.select(t), t.isPropagationStopped() || (this.mouseHandled=!0), i.has(".ui-menu").length ? this.expand(t) : !this.element.is(":focus") && e(this.document[0].activeElement).closest(".ui-menu").length && (this.element.trigger("focus", [!0]), this.active && 1 === this.active.parents(".ui-menu").length && clearTimeout(this.timer)))
                },
                "mouseenter .ui-menu-item": function(t) {
                    var i = e(t.currentTarget);
                    i.siblings(".ui-state-active").removeClass("ui-state-active"), this.focus(t, i)
                },
                mouseleave: "collapseAll",
                "mouseleave .ui-menu": "collapseAll",
                focus: function(e, t) {
                    var i = this.active || this.element.find(this.options.items).eq(0);
                    t || this.focus(e, i)
                },
                blur: function(t) {
                    this._delay(function() {
                        e.contains(this.element[0], this.document[0].activeElement) || this.collapseAll(t)
                    })
                },
                keydown: "_keydown"
            }), this.refresh(), this._on(this.document, {
                click: function(e) {
                    this._closeOnDocumentClick(e) && this.collapseAll(e), this.mouseHandled=!1
                }
            })
        },
        _destroy: function() {
            this.element.removeAttr("aria-activedescendant").find(".ui-menu").addBack().removeClass("ui-menu ui-widget ui-widget-content ui-menu-icons ui-front").removeAttr("role").removeAttr("tabIndex").removeAttr("aria-labelledby").removeAttr("aria-expanded").removeAttr("aria-hidden").removeAttr("aria-disabled").removeUniqueId().show(), this.element.find(".ui-menu-item").removeClass("ui-menu-item").removeAttr("role").removeAttr("aria-disabled").removeUniqueId().removeClass("ui-state-hover").removeAttr("tabIndex").removeAttr("role").removeAttr("aria-haspopup").children().each(function() {
                var t = e(this);
                t.data("ui-menu-submenu-carat") && t.remove()
            }), this.element.find(".ui-menu-divider").removeClass("ui-menu-divider ui-widget-content")
        },
        _keydown: function(t) {
            function i(e) {
                return e.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, "\\$&")
            }
            var s, a, n, r, o, h=!0;
            switch (t.keyCode) {
            case e.ui.keyCode.PAGE_UP:
                this.previousPage(t);
                break;
            case e.ui.keyCode.PAGE_DOWN:
                this.nextPage(t);
                break;
            case e.ui.keyCode.HOME:
                this._move("first", "first", t);
                break;
            case e.ui.keyCode.END:
                this._move("last", "last", t);
                break;
            case e.ui.keyCode.UP:
                this.previous(t);
                break;
            case e.ui.keyCode.DOWN:
                this.next(t);
                break;
            case e.ui.keyCode.LEFT:
                this.collapse(t);
                break;
            case e.ui.keyCode.RIGHT:
                this.active&&!this.active.is(".ui-state-disabled") && this.expand(t);
                break;
            case e.ui.keyCode.ENTER:
            case e.ui.keyCode.SPACE:
                this._activate(t);
                break;
            case e.ui.keyCode.ESCAPE:
                this.collapse(t);
                break;
            default:
                h=!1, a = this.previousFilter || "", n = String.fromCharCode(t.keyCode), r=!1, clearTimeout(this.filterTimer), n === a ? r=!0 : n = a + n, o = RegExp("^" + i(n), "i"), s = this.activeMenu.find(this.options.items).filter(function() {
                    return o.test(e(this).text())
                }), s = r&&-1 !== s.index(this.active.next()) ? this.active.nextAll(".ui-menu-item") : s, s.length || (n = String.fromCharCode(t.keyCode), o = RegExp("^" + i(n), "i"), s = this.activeMenu.find(this.options.items).filter(function() {
                    return o.test(e(this).text())
                })), s.length ? (this.focus(t, s), s.length > 1 ? (this.previousFilter = n, this.filterTimer = this._delay(function() {
                    delete this.previousFilter
                }, 1e3)) : delete this.previousFilter) : delete this.previousFilter
            }
            h && t.preventDefault()
        },
        _activate: function(e) {
            this.active.is(".ui-state-disabled") || (this.active.is("[aria-haspopup='true']") ? this.expand(e) : this.select(e))
        },
        refresh: function() {
            var t, i, s = this, a = this.options.icons.submenu, n = this.element.find(this.options.menus);
            this.element.toggleClass("ui-menu-icons", !!this.element.find(".ui-icon").length), n.filter(":not(.ui-menu)").addClass("ui-menu ui-widget ui-widget-content ui-front").hide().attr({
                role: this.options.role,
                "aria-hidden": "true",
                "aria-expanded": "false"
            }).each(function() {
                var t = e(this), i = t.parent(), s = e("<span>").addClass("ui-menu-icon ui-icon " + a).data("ui-menu-submenu-carat", !0);
                i.attr("aria-haspopup", "true").prepend(s), t.attr("aria-labelledby", i.attr("id"))
            }), t = n.add(this.element), i = t.find(this.options.items), i.not(".ui-menu-item").each(function() {
                var t = e(this);
                s._isDivider(t) && t.addClass("ui-widget-content ui-menu-divider")
            }), i.not(".ui-menu-item, .ui-menu-divider").addClass("ui-menu-item").uniqueId().attr({
                tabIndex: - 1,
                role: this._itemRole()
            }), i.filter(".ui-state-disabled").attr("aria-disabled", "true"), this.active&&!e.contains(this.element[0], this.active[0]) && this.blur()
        },
        _itemRole: function() {
            return {
                menu: "menuitem",
                listbox: "option"
            }
            [this.options.role]
        },
        _setOption: function(e, t) {
            "icons" === e && this.element.find(".ui-menu-icon").removeClass(this.options.icons.submenu).addClass(t.submenu), "disabled" === e && this.element.toggleClass("ui-state-disabled", !!t).attr("aria-disabled", t), this._super(e, t)
        },
        focus: function(e, t) {
            var i, s;
            this.blur(e, e && "focus" === e.type), this._scrollIntoView(t), this.active = t.first(), s = this.active.addClass("ui-state-focus").removeClass("ui-state-active"), this.options.role && this.element.attr("aria-activedescendant", s.attr("id")), this.active.parent().closest(".ui-menu-item").addClass("ui-state-active"), e && "keydown" === e.type ? this._close() : this.timer = this._delay(function() {
                this._close()
            }, this.delay), i = t.children(".ui-menu"), i.length && e && /^mouse/.test(e.type) && this._startOpening(i), this.activeMenu = t.parent(), this._trigger("focus", e, {
                item: t
            })
        },
        _scrollIntoView: function(t) {
            var i, s, a, n, r, o;
            this._hasScroll() && (i = parseFloat(e.css(this.activeMenu[0], "borderTopWidth")) || 0, s = parseFloat(e.css(this.activeMenu[0], "paddingTop")) || 0, a = t.offset().top - this.activeMenu.offset().top - i - s, n = this.activeMenu.scrollTop(), r = this.activeMenu.height(), o = t.outerHeight(), 0 > a ? this.activeMenu.scrollTop(n + a) : a + o > r && this.activeMenu.scrollTop(n + a - r + o))
        },
        blur: function(e, t) {
            t || clearTimeout(this.timer), this.active && (this.active.removeClass("ui-state-focus"), this.active = null, this._trigger("blur", e, {
                item: this.active
            }))
        },
        _startOpening: function(e) {
            clearTimeout(this.timer), "true" === e.attr("aria-hidden") && (this.timer = this._delay(function() {
                this._close(), this._open(e)
            }, this.delay))
        },
        _open: function(t) {
            var i = e.extend({
                of: this.active
            }, this.options.position);
            clearTimeout(this.timer), this.element.find(".ui-menu").not(t.parents(".ui-menu")).hide().attr("aria-hidden", "true"), t.show().removeAttr("aria-hidden").attr("aria-expanded", "true").position(i)
        },
        collapseAll: function(t, i) {
            clearTimeout(this.timer), this.timer = this._delay(function() {
                var s = i ? this.element: e(t && t.target).closest(this.element.find(".ui-menu"));
                s.length || (s = this.element), this._close(s), this.blur(t), this.activeMenu = s
            }, this.delay)
        },
        _close: function(e) {
            e || (e = this.active ? this.active.parent() : this.element), e.find(".ui-menu").hide().attr("aria-hidden", "true").attr("aria-expanded", "false").end().find(".ui-state-active").not(".ui-state-focus").removeClass("ui-state-active")
        },
        _closeOnDocumentClick: function(t) {
            return !e(t.target).closest(".ui-menu").length
        },
        _isDivider: function(e) {
            return !/[^\-\u2014\u2013\s]/.test(e.text())
        },
        collapse: function(e) {
            var t = this.active && this.active.parent().closest(".ui-menu-item", this.element);
            t && t.length && (this._close(), this.focus(e, t))
        },
        expand: function(e) {
            var t = this.active && this.active.children(".ui-menu ").find(this.options.items).first();
            t && t.length && (this._open(t.parent()), this._delay(function() {
                this.focus(e, t)
            }))
        },
        next: function(e) {
            this._move("next", "first", e)
        },
        previous: function(e) {
            this._move("prev", "last", e)
        },
        isFirstItem: function() {
            return this.active&&!this.active.prevAll(".ui-menu-item").length
        },
        isLastItem: function() {
            return this.active&&!this.active.nextAll(".ui-menu-item").length
        },
        _move: function(e, t, i) {
            var s;
            this.active && (s = "first" === e || "last" === e ? this.active["first" === e ? "prevAll": "nextAll"](".ui-menu-item").eq( - 1) : this.active[e + "All"](".ui-menu-item").eq(0)), s && s.length && this.active || (s = this.activeMenu.find(this.options.items)[t]()), this.focus(i, s)
        },
        nextPage: function(t) {
            var i, s, a;
            return this.active ? (this.isLastItem() || (this._hasScroll() ? (s = this.active.offset().top, a = this.element.height(), this.active.nextAll(".ui-menu-item").each(function() {
                return i = e(this), 0 > i.offset().top - s - a
            }), this.focus(t, i)) : this.focus(t, this.activeMenu.find(this.options.items)[this.active ? "last": "first"]())), void 0) : (this.next(t), void 0)
        },
        previousPage: function(t) {
            var i, s, a;
            return this.active ? (this.isFirstItem() || (this._hasScroll() ? (s = this.active.offset().top, a = this.element.height(), this.active.prevAll(".ui-menu-item").each(function() {
                return i = e(this), i.offset().top - s + a > 0
            }), this.focus(t, i)) : this.focus(t, this.activeMenu.find(this.options.items).first())), void 0) : (this.next(t), void 0)
        },
        _hasScroll: function() {
            return this.element.outerHeight() < this.element.prop("scrollHeight")
        },
        select: function(t) {
            this.active = this.active || e(t.target).closest(".ui-menu-item");
            var i = {
                item: this.active
            };
            this.active.has(".ui-menu").length || this.collapseAll(t, !0), this._trigger("select", t, i)
        }
    }), e.widget("ui.autocomplete", {
        version: "1.11.1",
        defaultElement: "<input>",
        options: {
            appendTo: null,
            autoFocus: !1,
            delay: 300,
            minLength: 1,
            position: {
                my: "left top",
                at: "left bottom",
                collision: "none"
            },
            source: null,
            change: null,
            close: null,
            focus: null,
            open: null,
            response: null,
            search: null,
            select: null
        },
        requestIndex: 0,
        pending: 0,
        _create: function() {
            var t, i, s, a = this.element[0].nodeName.toLowerCase(), n = "textarea" === a, r = "input" === a;
            this.isMultiLine = n?!0 : r?!1 : this.element.prop("isContentEditable"), this.valueMethod = this.element[n || r ? "val": "text"], this.isNewMenu=!0, this.element.addClass("ui-autocomplete-input").attr("autocomplete", "off"), this._on(this.element, {
                keydown: function(a) {
                    if (this.element.prop("readOnly"))
                        return t=!0, s=!0, i=!0, void 0;
                    t=!1, s=!1, i=!1;
                    var n = e.ui.keyCode;
                    switch (a.keyCode) {
                    case n.PAGE_UP:
                        t=!0, this._move("previousPage", a);
                        break;
                    case n.PAGE_DOWN:
                        t=!0, this._move("nextPage", a);
                        break;
                    case n.UP:
                        t=!0, this._keyEvent("previous", a);
                        break;
                    case n.DOWN:
                        t=!0, this._keyEvent("next", a);
                        break;
                    case n.ENTER:
                        this.menu.active && (t=!0, a.preventDefault(), this.menu.select(a));
                        break;
                    case n.TAB:
                        this.menu.active && this.menu.select(a);
                        break;
                    case n.ESCAPE:
                        this.menu.element.is(":visible") && (this.isMultiLine || this._value(this.term), this.close(a), a.preventDefault());
                        break;
                    default:
                        i=!0, this._searchTimeout(a)
                    }
                },
                keypress: function(s) {
                    if (t)
                        return t=!1, (!this.isMultiLine || this.menu.element.is(":visible")) && s.preventDefault(), void 0;
                    if (!i) {
                        var a = e.ui.keyCode;
                        switch (s.keyCode) {
                        case a.PAGE_UP:
                            this._move("previousPage", s);
                            break;
                        case a.PAGE_DOWN:
                            this._move("nextPage", s);
                            break;
                        case a.UP:
                            this._keyEvent("previous", s);
                            break;
                        case a.DOWN:
                            this._keyEvent("next", s)
                        }
                    }
                },
                input: function(e) {
                    return s ? (s=!1, e.preventDefault(), void 0) : (this._searchTimeout(e), void 0)
                },
                focus: function() {
                    this.selectedItem = null, this.previous = this._value()
                },
                blur: function(e) {
                    return this.cancelBlur ? (delete this.cancelBlur, void 0) : (clearTimeout(this.searching), this.close(e), this._change(e), void 0)
                }
            }), this._initSource(), this.menu = e("<ul>").addClass("ui-autocomplete ui-front").appendTo(this._appendTo()).menu({
                role: null
            }).hide().menu("instance"), this._on(this.menu.element, {
                mousedown: function(t) {
                    t.preventDefault(), this.cancelBlur=!0, this._delay(function() {
                        delete this.cancelBlur
                    });
                    var i = this.menu.element[0];
                    e(t.target).closest(".ui-menu-item").length || this._delay(function() {
                        var t = this;
                        this.document.one("mousedown", function(s) {
                            s.target === t.element[0] || s.target === i || e.contains(i, s.target) || t.close()
                        })
                    })
                },
                menufocus: function(t, i) {
                    var s, a;
                    return this.isNewMenu && (this.isNewMenu=!1, t.originalEvent && /^mouse/.test(t.originalEvent.type)) ? (this.menu.blur(), this.document.one("mousemove", function() {
                        e(t.target).trigger(t.originalEvent)
                    }), void 0) : (a = i.item.data("ui-autocomplete-item"), !1 !== this._trigger("focus", t, {
                        item: a
                    }) && t.originalEvent && /^key/.test(t.originalEvent.type) && this._value(a.value), s = i.item.attr("aria-label") || a.value, s && e.trim(s).length && (this.liveRegion.children().hide(), e("<div>").text(s).appendTo(this.liveRegion)), void 0)
                },
                menuselect: function(e, t) {
                    var i = t.item.data("ui-autocomplete-item"), s = this.previous;
                    this.element[0] !== this.document[0].activeElement && (this.element.focus(), this.previous = s, this._delay(function() {
                        this.previous = s, this.selectedItem = i
                    })), !1 !== this._trigger("select", e, {
                        item: i
                    }) && this._value(i.value), this.term = this._value(), this.close(e), this.selectedItem = i
                }
            }), this.liveRegion = e("<span>", {
                role: "status",
                "aria-live": "assertive",
                "aria-relevant": "additions"
            }).addClass("ui-helper-hidden-accessible").appendTo(this.document[0].body), this._on(this.window, {
                beforeunload: function() {
                    this.element.removeAttr("autocomplete")
                }
            })
        },
        _destroy: function() {
            clearTimeout(this.searching), this.element.removeClass("ui-autocomplete-input").removeAttr("autocomplete"), this.menu.element.remove(), this.liveRegion.remove()
        },
        _setOption: function(e, t) {
            this._super(e, t), "source" === e && this._initSource(), "appendTo" === e && this.menu.element.appendTo(this._appendTo()), "disabled" === e && t && this.xhr && this.xhr.abort()
        },
        _appendTo: function() {
            var t = this.options.appendTo;
            return t && (t = t.jquery || t.nodeType ? e(t) : this.document.find(t).eq(0)), t && t[0] || (t = this.element.closest(".ui-front")), t.length || (t = this.document[0].body), t
        },
        _initSource: function() {
            var t, i, s = this;
            e.isArray(this.options.source) ? (t = this.options.source, this.source = function(i, s) {
                s(e.ui.autocomplete.filter(t, i.term))
            }) : "string" == typeof this.options.source ? (i = this.options.source, this.source = function(t, a) {
                s.xhr && s.xhr.abort(), s.xhr = e.ajax({
                    url: i,
                    data: t,
                    dataType: "json",
                    success: function(e) {
                        a(e)
                    },
                    error: function() {
                        a([])
                    }
                })
            }) : this.source = this.options.source
        },
        _searchTimeout: function(e) {
            clearTimeout(this.searching), this.searching = this._delay(function() {
                var t = this.term === this._value(), i = this.menu.element.is(":visible"), s = e.altKey || e.ctrlKey || e.metaKey || e.shiftKey;
                (!t || t&&!i&&!s) && (this.selectedItem = null, this.search(null, e))
            }, this.options.delay)
        },
        search: function(e, t) {
            return e = null != e ? e : this._value(), this.term = this._value(), e.length < this.options.minLength ? this.close(t) : this._trigger("search", t)!==!1 ? this._search(e) : void 0
        },
        _search: function(e) {
            this.pending++, this.element.addClass("ui-autocomplete-loading"), this.cancelSearch=!1, this.source({
                term: e
            }, this._response())
        },
        _response: function() {
            var t=++this.requestIndex;
            return e.proxy(function(e) {
                t === this.requestIndex && this.__response(e), this.pending--, this.pending || this.element.removeClass("ui-autocomplete-loading")
            }, this)
        },
        __response: function(e) {
            e && (e = this._normalize(e)), this._trigger("response", null, {
                content: e
            }), !this.options.disabled && e && e.length&&!this.cancelSearch ? (this._suggest(e), this._trigger("open")) : this._close()
        },
        close: function(e) {
            this.cancelSearch=!0, this._close(e)
        },
        _close: function(e) {
            this.menu.element.is(":visible") && (this.menu.element.hide(), this.menu.blur(), this.isNewMenu=!0, this._trigger("close", e))
        },
        _change: function(e) {
            this.previous !== this._value() && this._trigger("change", e, {
                item: this.selectedItem
            })
        },
        _normalize: function(t) {
            return t.length && t[0].label && t[0].value ? t : e.map(t, function(t) {
                return "string" == typeof t ? {
                    label: t,
                    value: t
                } : e.extend({}, t, {
                    label: t.label || t.value,
                    value: t.value || t.label
                })
            })
        },
        _suggest: function(t) {
            var i = this.menu.element.empty();
            this._renderMenu(i, t), this.isNewMenu=!0, this.menu.refresh(), i.show(), this._resizeMenu(), i.position(e.extend({
                of: this.element
            }, this.options.position)), this.options.autoFocus && this.menu.next()
        },
        _resizeMenu: function() {
            var e = this.menu.element;
            e.outerWidth(Math.max(e.width("").outerWidth() + 1, this.element.outerWidth()))
        },
        _renderMenu: function(t, i) {
            var s = this;
            e.each(i, function(e, i) {
                s._renderItemData(t, i)
            })
        },
        _renderItemData: function(e, t) {
            return this._renderItem(e, t).data("ui-autocomplete-item", t)
        },
        _renderItem: function(t, i) {
            return e("<li>").text(i.label).appendTo(t)
        },
        _move: function(e, t) {
            return this.menu.element.is(":visible") ? this.menu.isFirstItem() && /^previous/.test(e) || this.menu.isLastItem() && /^next/.test(e) ? (this.isMultiLine || this._value(this.term), this.menu.blur(), void 0) : (this.menu[e](t), void 0) : (this.search(null, t), void 0)
        },
        widget: function() {
            return this.menu.element
        },
        _value: function() {
            return this.valueMethod.apply(this.element, arguments)
        },
        _keyEvent: function(e, t) {
            (!this.isMultiLine || this.menu.element.is(":visible")) && (this._move(e, t), t.preventDefault())
        }
    }), e.extend(e.ui.autocomplete, {
        escapeRegex: function(e) {
            return e.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, "\\$&")
        },
        filter: function(t, i) {
            var s = RegExp(e.ui.autocomplete.escapeRegex(i), "i");
            return e.grep(t, function(e) {
                return s.test(e.label || e.value || e)
            })
        }
    }), e.widget("ui.autocomplete", e.ui.autocomplete, {
        options: {
            messages: {
                noResults: "No search results.",
                results: function(e) {
                    return e + (e > 1 ? " results are" : " result is") + " available, use up and down arrow keys to navigate."
                }
            }
        },
        __response: function(t) {
            var i;
            this._superApply(arguments), this.options.disabled || this.cancelSearch || (i = t && t.length ? this.options.messages.results(t.length) : this.options.messages.noResults, this.liveRegion.children().hide(), e("<div>").text(i).appendTo(this.liveRegion))
        }
    }), e.ui.autocomplete;
    var c, p = "ui-button ui-widget ui-state-default ui-corner-all", f = "ui-button-icons-only ui-button-icon-only ui-button-text-icons ui-button-text-icon-primary ui-button-text-icon-secondary ui-button-text-only", m = function() {
        var t = e(this);
        setTimeout(function() {
            t.find(":ui-button").button("refresh")
        }, 1)
    }, g = function(t) {
        var i = t.name, s = t.form, a = e([]);
        return i && (i = i.replace(/'/g, "\\'"), a = s ? e(s).find("[name='" + i + "'][type=radio]") : e("[name='" + i + "'][type=radio]", t.ownerDocument).filter(function() {
            return !this.form
        })), a
    };
    e.widget("ui.button", {
        version: "1.11.1",
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
            this.element.closest("form").unbind("reset" + this.eventNamespace).bind("reset" + this.eventNamespace, m), "boolean" != typeof this.options.disabled ? this.options.disabled=!!this.element.prop("disabled") : this.element.prop("disabled", this.options.disabled), this._determineButtonType(), this.hasTitle=!!this.buttonElement.attr("title");
            var t = this, i = this.options, s = "checkbox" === this.type || "radio" === this.type, a = s ? "": "ui-state-active";
            null === i.label && (i.label = "input" === this.type ? this.buttonElement.val() : this.buttonElement.html()), this._hoverable(this.buttonElement), this.buttonElement.addClass(p).attr("role", "button").bind("mouseenter" + this.eventNamespace, function() {
                i.disabled || this === c && e(this).addClass("ui-state-active")
            }).bind("mouseleave" + this.eventNamespace, function() {
                i.disabled || e(this).removeClass(a)
            }).bind("click" + this.eventNamespace, function(e) {
                i.disabled && (e.preventDefault(), e.stopImmediatePropagation())
            }), this._on({
                focus: function() {
                    this.buttonElement.addClass("ui-state-focus")
                },
                blur: function() {
                    this.buttonElement.removeClass("ui-state-focus")
                }
            }), s && this.element.bind("change" + this.eventNamespace, function() {
                t.refresh()
            }), "checkbox" === this.type ? this.buttonElement.bind("click" + this.eventNamespace, function() {
                return i.disabled?!1 : void 0
            }) : "radio" === this.type ? this.buttonElement.bind("click" + this.eventNamespace, function() {
                if (i.disabled)
                    return !1;
                e(this).addClass("ui-state-active"), t.buttonElement.attr("aria-pressed", "true");
                var s = t.element[0];
                g(s).not(s).map(function() {
                    return e(this).button("widget")[0]
                }).removeClass("ui-state-active").attr("aria-pressed", "false")
            }) : (this.buttonElement.bind("mousedown" + this.eventNamespace, function() {
                return i.disabled?!1 : (e(this).addClass("ui-state-active"), c = this, t.document.one("mouseup", function() {
                    c = null
                }), void 0)
            }).bind("mouseup" + this.eventNamespace, function() {
                return i.disabled?!1 : (e(this).removeClass("ui-state-active"), void 0)
            }).bind("keydown" + this.eventNamespace, function(t) {
                return i.disabled?!1 : ((t.keyCode === e.ui.keyCode.SPACE || t.keyCode === e.ui.keyCode.ENTER) && e(this).addClass("ui-state-active"), void 0)
            }).bind("keyup" + this.eventNamespace + " blur" + this.eventNamespace, function() {
                e(this).removeClass("ui-state-active")
            }), this.buttonElement.is("a") && this.buttonElement.keyup(function(t) {
                t.keyCode === e.ui.keyCode.SPACE && e(this).click()
            })), this._setOption("disabled", i.disabled), this._resetButton()
        },
        _determineButtonType: function() {
            var e, t, i;
            this.type = this.element.is("[type=checkbox]") ? "checkbox" : this.element.is("[type=radio]") ? "radio" : this.element.is("input") ? "input" : "button", "checkbox" === this.type || "radio" === this.type ? (e = this.element.parents().last(), t = "label[for='" + this.element.attr("id") + "']", this.buttonElement = e.find(t), this.buttonElement.length || (e = e.length ? e.siblings() : this.element.siblings(), this.buttonElement = e.filter(t), this.buttonElement.length || (this.buttonElement = e.find(t))), this.element.addClass("ui-helper-hidden-accessible"), i = this.element.is(":checked"), i && this.buttonElement.addClass("ui-state-active"), this.buttonElement.prop("aria-pressed", i)) : this.buttonElement = this.element
        },
        widget: function() {
            return this.buttonElement
        },
        _destroy: function() {
            this.element.removeClass("ui-helper-hidden-accessible"), this.buttonElement.removeClass(p + " ui-state-active " + f).removeAttr("role").removeAttr("aria-pressed").html(this.buttonElement.find(".ui-button-text").html()), this.hasTitle || this.buttonElement.removeAttr("title")
        },
        _setOption: function(e, t) {
            return this._super(e, t), "disabled" === e ? (this.widget().toggleClass("ui-state-disabled", !!t), this.element.prop("disabled", !!t), t && ("checkbox" === this.type || "radio" === this.type ? this.buttonElement.removeClass("ui-state-focus") : this.buttonElement.removeClass("ui-state-focus ui-state-active")), void 0) : (this._resetButton(), void 0)
        },
        refresh: function() {
            var t = this.element.is("input, button") ? this.element.is(":disabled"): this.element.hasClass("ui-button-disabled");
            t !== this.options.disabled && this._setOption("disabled", t), "radio" === this.type ? g(this.element[0]).each(function() {
                e(this).is(":checked") ? e(this).button("widget").addClass("ui-state-active").attr("aria-pressed", "true") : e(this).button("widget").removeClass("ui-state-active").attr("aria-pressed", "false")
            }) : "checkbox" === this.type && (this.element.is(":checked") ? this.buttonElement.addClass("ui-state-active").attr("aria-pressed", "true") : this.buttonElement.removeClass("ui-state-active").attr("aria-pressed", "false"))
        },
        _resetButton: function() {
            if ("input" === this.type)
                return this.options.label && this.element.val(this.options.label), void 0;
            var t = this.buttonElement.removeClass(f), i = e("<span></span>", this.document[0]).addClass("ui-button-text").html(this.options.label).appendTo(t.empty()).text(), s = this.options.icons, a = s.primary && s.secondary, n = [];
            s.primary || s.secondary ? (this.options.text && n.push("ui-button-text-icon" + (a ? "s" : s.primary ? "-primary" : "-secondary")), s.primary && t.prepend("<span class='ui-button-icon-primary ui-icon " + s.primary + "'></span>"), s.secondary && t.append("<span class='ui-button-icon-secondary ui-icon " + s.secondary + "'></span>"), this.options.text || (n.push(a ? "ui-button-icons-only" : "ui-button-icon-only"), this.hasTitle || t.attr("title", e.trim(i)))) : n.push("ui-button-text-only"), t.addClass(n.join(" "))
        }
    }), e.widget("ui.buttonset", {
        version: "1.11.1",
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
            var t = "rtl" === this.element.css("direction"), i = this.element.find(this.options.items), s = i.filter(":ui-button");
            i.not(":ui-button").button(), s.button("refresh"), this.buttons = i.map(function() {
                return e(this).button("widget")[0]
            }).removeClass("ui-corner-all ui-corner-left ui-corner-right").filter(":first").addClass(t ? "ui-corner-right" : "ui-corner-left").end().filter(":last").addClass(t ? "ui-corner-left" : "ui-corner-right").end().end()
        },
        _destroy: function() {
            this.element.removeClass("ui-buttonset"), this.buttons.map(function() {
                return e(this).button("widget")[0]
            }).removeClass("ui-corner-left ui-corner-right").end().button("destroy")
        }
    }), e.ui.button, e.extend(e.ui, {
        datepicker: {
            version: "1.11.1"
        }
    });
    var v;
    e.extend(a.prototype, {
        markerClassName: "hasDatepicker",
        maxRows: 4,
        _widgetDatepicker: function() {
            return this.dpDiv
        },
        setDefaults: function(e) {
            return o(this._defaults, e || {}), this
        },
        _attachDatepicker: function(t, i) {
            var s, a, n;
            s = t.nodeName.toLowerCase(), a = "div" === s || "span" === s, t.id || (this.uuid += 1, t.id = "dp" + this.uuid), n = this._newInst(e(t), a), n.settings = e.extend({}, i || {}), "input" === s ? this._connectDatepicker(t, n) : a && this._inlineDatepicker(t, n)
        },
        _newInst: function(t, i) {
            var s = t[0].id.replace(/([^A-Za-z0-9_\-])/g, "\\\\$1");
            return {
                id: s,
                input: t,
                selectedDay: 0,
                selectedMonth: 0,
                selectedYear: 0,
                drawMonth: 0,
                drawYear: 0,
                inline: i,
                dpDiv: i ? n(e("<div class='" + this._inlineClass + " ui-datepicker ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>")): this.dpDiv
            }
        },
        _connectDatepicker: function(t, i) {
            var s = e(t);
            i.append = e([]), i.trigger = e([]), s.hasClass(this.markerClassName) || (this._attachments(s, i), s.addClass(this.markerClassName).keydown(this._doKeyDown).keypress(this._doKeyPress).keyup(this._doKeyUp), this._autoSize(i), e.data(t, "datepicker", i), i.settings.disabled && this._disableDatepicker(t))
        },
        _attachments: function(t, i) {
            var s, a, n, r = this._get(i, "appendText"), o = this._get(i, "isRTL");
            i.append && i.append.remove(), r && (i.append = e("<span class='" + this._appendClass + "'>" + r + "</span>"), t[o ? "before": "after"](i.append)), t.unbind("focus", this._showDatepicker), i.trigger && i.trigger.remove(), s = this._get(i, "showOn"), ("focus" === s || "both" === s) && t.focus(this._showDatepicker), ("button" === s || "both" === s) && (a = this._get(i, "buttonText"), n = this._get(i, "buttonImage"), i.trigger = e(this._get(i, "buttonImageOnly") ? e("<img/>").addClass(this._triggerClass).attr({
                src: n,
                alt: a,
                title: a
            }) : e("<button type='button'></button>").addClass(this._triggerClass).html(n ? e("<img/>").attr({
                src: n,
                alt: a,
                title: a
            }) : a)), t[o ? "before": "after"](i.trigger), i.trigger.click(function() {
                return e.datepicker._datepickerShowing && e.datepicker._lastInput === t[0] ? e.datepicker._hideDatepicker() : e.datepicker._datepickerShowing && e.datepicker._lastInput !== t[0] ? (e.datepicker._hideDatepicker(), e.datepicker._showDatepicker(t[0])) : e.datepicker._showDatepicker(t[0]), !1
            }))
        },
        _autoSize: function(e) {
            if (this._get(e, "autoSize")&&!e.inline) {
                var t, i, s, a, n = new Date(2009, 11, 20), r = this._get(e, "dateFormat");
                r.match(/[DM]/) && (t = function(e) {
                    for (i = 0, s = 0, a = 0; e.length > a; a++)
                        e[a].length > i && (i = e[a].length, s = a);
                    return s
                }, n.setMonth(t(this._get(e, r.match(/MM/) ? "monthNames" : "monthNamesShort"))), n.setDate(t(this._get(e, r.match(/DD/) ? "dayNames" : "dayNamesShort")) + 20 - n.getDay())), e.input.attr("size", this._formatDate(e, n).length)
            }
        },
        _inlineDatepicker: function(t, i) {
            var s = e(t);
            s.hasClass(this.markerClassName) || (s.addClass(this.markerClassName).append(i.dpDiv), e.data(t, "datepicker", i), this._setDate(i, this._getDefaultDate(i), !0), this._updateDatepicker(i), this._updateAlternate(i), i.settings.disabled && this._disableDatepicker(t), i.dpDiv.css("display", "block"))
        },
        _dialogDatepicker: function(t, i, s, a, n) {
            var r, h, l, u, d, c = this._dialogInst;
            return c || (this.uuid += 1, r = "dp" + this.uuid, this._dialogInput = e("<input type='text' id='" + r + "' style='position: absolute; top: -100px; width: 0px;'/>"), this._dialogInput.keydown(this._doKeyDown), e("body").append(this._dialogInput), c = this._dialogInst = this._newInst(this._dialogInput, !1), c.settings = {}, e.data(this._dialogInput[0], "datepicker", c)), o(c.settings, a || {}), i = i && i.constructor === Date ? this._formatDate(c, i) : i, this._dialogInput.val(i), this._pos = n ? n.length ? n : [n.pageX, n.pageY] : null, this._pos || (h = document.documentElement.clientWidth, l = document.documentElement.clientHeight, u = document.documentElement.scrollLeft || document.body.scrollLeft, d = document.documentElement.scrollTop || document.body.scrollTop, this._pos = [h / 2 - 100 + u, l / 2 - 150 + d]), this._dialogInput.css("left", this._pos[0] + 20 + "px").css("top", this._pos[1] + "px"), c.settings.onSelect = s, this._inDialog=!0, this.dpDiv.addClass(this._dialogClass), this._showDatepicker(this._dialogInput[0]), e.blockUI && e.blockUI(this.dpDiv), e.data(this._dialogInput[0], "datepicker", c), this
        },
        _destroyDatepicker: function(t) {
            var i, s = e(t), a = e.data(t, "datepicker");
            s.hasClass(this.markerClassName) && (i = t.nodeName.toLowerCase(), e.removeData(t, "datepicker"), "input" === i ? (a.append.remove(), a.trigger.remove(), s.removeClass(this.markerClassName).unbind("focus", this._showDatepicker).unbind("keydown", this._doKeyDown).unbind("keypress", this._doKeyPress).unbind("keyup", this._doKeyUp)) : ("div" === i || "span" === i) && s.removeClass(this.markerClassName).empty())
        },
        _enableDatepicker: function(t) {
            var i, s, a = e(t), n = e.data(t, "datepicker");
            a.hasClass(this.markerClassName) && (i = t.nodeName.toLowerCase(), "input" === i ? (t.disabled=!1, n.trigger.filter("button").each(function() {
                this.disabled=!1
            }).end().filter("img").css({
                opacity: "1.0",
                cursor: ""
            })) : ("div" === i || "span" === i) && (s = a.children("." + this._inlineClass), s.children().removeClass("ui-state-disabled"), s.find("select.ui-datepicker-month, select.ui-datepicker-year").prop("disabled", !1)), this._disabledInputs = e.map(this._disabledInputs, function(e) {
                return e === t ? null : e
            }))
        },
        _disableDatepicker: function(t) {
            var i, s, a = e(t), n = e.data(t, "datepicker");
            a.hasClass(this.markerClassName) && (i = t.nodeName.toLowerCase(), "input" === i ? (t.disabled=!0, n.trigger.filter("button").each(function() {
                this.disabled=!0
            }).end().filter("img").css({
                opacity: "0.5",
                cursor: "default"
            })) : ("div" === i || "span" === i) && (s = a.children("." + this._inlineClass), s.children().addClass("ui-state-disabled"), s.find("select.ui-datepicker-month, select.ui-datepicker-year").prop("disabled", !0)), this._disabledInputs = e.map(this._disabledInputs, function(e) {
                return e === t ? null : e
            }), this._disabledInputs[this._disabledInputs.length] = t)
        },
        _isDisabledDatepicker: function(e) {
            if (!e)
                return !1;
            for (var t = 0; this._disabledInputs.length > t; t++)
                if (this._disabledInputs[t] === e)
                    return !0;
            return !1
        },
        _getInst: function(t) {
            try {
                return e.data(t, "datepicker")
            } catch (i) {
                throw "Missing instance data for this datepicker"
            }
        },
        _optionDatepicker: function(t, i, s) {
            var a, n, r, h, l = this._getInst(t);
            return 2 === arguments.length && "string" == typeof i ? "defaults" === i ? e.extend({}, e.datepicker._defaults) : l ? "all" === i ? e.extend({}, l.settings) : this._get(l, i) : null : (a = i || {}, "string" == typeof i && (a = {}, a[i] = s), l && (this._curInst === l && this._hideDatepicker(), n = this._getDateDatepicker(t, !0), r = this._getMinMaxDate(l, "min"), h = this._getMinMaxDate(l, "max"), o(l.settings, a), null !== r && void 0 !== a.dateFormat && void 0 === a.minDate && (l.settings.minDate = this._formatDate(l, r)), null !== h && void 0 !== a.dateFormat && void 0 === a.maxDate && (l.settings.maxDate = this._formatDate(l, h)), "disabled"in a && (a.disabled ? this._disableDatepicker(t) : this._enableDatepicker(t)), this._attachments(e(t), l), this._autoSize(l), this._setDate(l, n), this._updateAlternate(l), this._updateDatepicker(l)), void 0)
        },
        _changeDatepicker: function(e, t, i) {
            this._optionDatepicker(e, t, i)
        },
        _refreshDatepicker: function(e) {
            var t = this._getInst(e);
            t && this._updateDatepicker(t)
        },
        _setDateDatepicker: function(e, t) {
            var i = this._getInst(e);
            i && (this._setDate(i, t), this._updateDatepicker(i), this._updateAlternate(i))
        },
        _getDateDatepicker: function(e, t) {
            var i = this._getInst(e);
            return i&&!i.inline && this._setDateFromField(i, t), i ? this._getDate(i) : null
        },
        _doKeyDown: function(t) {
            var i, s, a, n = e.datepicker._getInst(t.target), r=!0, o = n.dpDiv.is(".ui-datepicker-rtl");
            if (n._keyEvent=!0, e.datepicker._datepickerShowing)
                switch (t.keyCode) {
                case 9:
                    e.datepicker._hideDatepicker(), r=!1;
                    break;
                case 13:
                    return a = e("td." + e.datepicker._dayOverClass + ":not(." + e.datepicker._currentClass + ")", n.dpDiv), a[0] && e.datepicker._selectDay(t.target, n.selectedMonth, n.selectedYear, a[0]), i = e.datepicker._get(n, "onSelect"), i ? (s = e.datepicker._formatDate(n), i.apply(n.input ? n.input[0] : null, [s, n])) : e.datepicker._hideDatepicker(), !1;
                case 27:
                    e.datepicker._hideDatepicker();
                    break;
                case 33:
                    e.datepicker._adjustDate(t.target, t.ctrlKey?-e.datepicker._get(n, "stepBigMonths") : - e.datepicker._get(n, "stepMonths"), "M");
                    break;
                case 34:
                    e.datepicker._adjustDate(t.target, t.ctrlKey?+e.datepicker._get(n, "stepBigMonths") : + e.datepicker._get(n, "stepMonths"), "M");
                    break;
                case 35:
                    (t.ctrlKey || t.metaKey) && e.datepicker._clearDate(t.target), r = t.ctrlKey || t.metaKey;
                    break;
                case 36:
                    (t.ctrlKey || t.metaKey) && e.datepicker._gotoToday(t.target), r = t.ctrlKey || t.metaKey;
                    break;
                case 37:
                    (t.ctrlKey || t.metaKey) && e.datepicker._adjustDate(t.target, o ? 1 : - 1, "D"), r = t.ctrlKey || t.metaKey, t.originalEvent.altKey && e.datepicker._adjustDate(t.target, t.ctrlKey?-e.datepicker._get(n, "stepBigMonths") : - e.datepicker._get(n, "stepMonths"), "M");
                    break;
                case 38:
                    (t.ctrlKey || t.metaKey) && e.datepicker._adjustDate(t.target, - 7, "D"), r = t.ctrlKey || t.metaKey;
                    break;
                case 39:
                    (t.ctrlKey || t.metaKey) && e.datepicker._adjustDate(t.target, o?-1 : 1, "D"), r = t.ctrlKey || t.metaKey, t.originalEvent.altKey && e.datepicker._adjustDate(t.target, t.ctrlKey?+e.datepicker._get(n, "stepBigMonths") : + e.datepicker._get(n, "stepMonths"), "M");
                    break;
                case 40:
                    (t.ctrlKey || t.metaKey) && e.datepicker._adjustDate(t.target, 7, "D"), r = t.ctrlKey || t.metaKey;
                    break;
                default:
                    r=!1
                } else 
                    36 === t.keyCode && t.ctrlKey ? e.datepicker._showDatepicker(this) : r=!1;
            r && (t.preventDefault(), t.stopPropagation())
        },
        _doKeyPress: function(t) {
            var i, s, a = e.datepicker._getInst(t.target);
            return e.datepicker._get(a, "constrainInput") ? (i = e.datepicker._possibleChars(e.datepicker._get(a, "dateFormat")), s = String.fromCharCode(null == t.charCode ? t.keyCode : t.charCode), t.ctrlKey || t.metaKey || " " > s ||!i || i.indexOf(s)>-1) : void 0
        },
        _doKeyUp: function(t) {
            var i, s = e.datepicker._getInst(t.target);
            if (s.input.val() !== s.lastVal)
                try {
                    i = e.datepicker.parseDate(e.datepicker._get(s, "dateFormat"), s.input ? s.input.val() : null, e.datepicker._getFormatConfig(s)), i && (e.datepicker._setDateFromField(s), e.datepicker._updateAlternate(s), e.datepicker._updateDatepicker(s))
            } catch (a) {}
            return !0
        },
        _showDatepicker: function(t) {
            if (t = t.target || t, "input" !== t.nodeName.toLowerCase() && (t = e("input", t.parentNode)[0]), !e.datepicker._isDisabledDatepicker(t) && e.datepicker._lastInput !== t) {
                var i, a, n, r, h, l, u;
                i = e.datepicker._getInst(t), e.datepicker._curInst && e.datepicker._curInst !== i && (e.datepicker._curInst.dpDiv.stop(!0, !0), i && e.datepicker._datepickerShowing && e.datepicker._hideDatepicker(e.datepicker._curInst.input[0])), a = e.datepicker._get(i, "beforeShow"), n = a ? a.apply(t, [t, i]) : {}, n!==!1 && (o(i.settings, n), i.lastVal = null, e.datepicker._lastInput = t, e.datepicker._setDateFromField(i), e.datepicker._inDialog && (t.value = ""), e.datepicker._pos || (e.datepicker._pos = e.datepicker._findPos(t), e.datepicker._pos[1] += t.offsetHeight), r=!1, e(t).parents().each(function() {
                    return r|="fixed" === e(this).css("position"), !r
                }), h = {
                    left: e.datepicker._pos[0],
                    top: e.datepicker._pos[1]
                }, e.datepicker._pos = null, i.dpDiv.empty(), i.dpDiv.css({
                    position: "absolute",
                    display: "block",
                    top: "-1000px"
                }), e.datepicker._updateDatepicker(i), h = e.datepicker._checkOffset(i, h, r), i.dpDiv.css({
                    position: e.datepicker._inDialog && e.blockUI ? "static": r ? "fixed": "absolute",
                    display: "none",
                    left: h.left + "px",
                    top: h.top + "px"
                }), i.inline || (l = e.datepicker._get(i, "showAnim"), u = e.datepicker._get(i, "duration"), i.dpDiv.css("z-index", s(e(t)) + 1), e.datepicker._datepickerShowing=!0, e.effects && e.effects.effect[l] ? i.dpDiv.show(l, e.datepicker._get(i, "showOptions"), u) : i.dpDiv[l || "show"](l ? u : null), e.datepicker._shouldFocusInput(i) && i.input.focus(), e.datepicker._curInst = i))
            }
        },
        _updateDatepicker: function(t) {
            this.maxRows = 4, v = t, t.dpDiv.empty().append(this._generateHTML(t)), this._attachHandlers(t);
            var i, s = this._getNumberOfMonths(t), a = s[1], n = 17, o = t.dpDiv.find("." + this._dayOverClass + " a");
            o.length > 0 && r.apply(o.get(0)), t.dpDiv.removeClass("ui-datepicker-multi-2 ui-datepicker-multi-3 ui-datepicker-multi-4").width(""), a > 1 && t.dpDiv.addClass("ui-datepicker-multi-" + a).css("width", n * a + "em"), t.dpDiv[(1 !== s[0] || 1 !== s[1] ? "add" : "remove") + "Class"]("ui-datepicker-multi"), t.dpDiv[(this._get(t, "isRTL") ? "add" : "remove") + "Class"]("ui-datepicker-rtl"), t === e.datepicker._curInst && e.datepicker._datepickerShowing && e.datepicker._shouldFocusInput(t) && t.input.focus(), t.yearshtml && (i = t.yearshtml, setTimeout(function() {
                i === t.yearshtml && t.yearshtml && t.dpDiv.find("select.ui-datepicker-year:first").replaceWith(t.yearshtml), i = t.yearshtml = null
            }, 0))
        },
        _shouldFocusInput: function(e) {
            return e.input && e.input.is(":visible")&&!e.input.is(":disabled")&&!e.input.is(":focus")
        },
        _checkOffset: function(t, i, s) {
            var a = t.dpDiv.outerWidth(), n = t.dpDiv.outerHeight(), r = t.input ? t.input.outerWidth(): 0, o = t.input ? t.input.outerHeight(): 0, h = document.documentElement.clientWidth + (s ? 0 : e(document).scrollLeft()), l = document.documentElement.clientHeight + (s ? 0 : e(document).scrollTop());
            return i.left -= this._get(t, "isRTL") ? a - r : 0, i.left -= s && i.left === t.input.offset().left ? e(document).scrollLeft() : 0, i.top -= s && i.top === t.input.offset().top + o ? e(document).scrollTop() : 0, i.left -= Math.min(i.left, i.left + a > h && h > a ? Math.abs(i.left + a - h) : 0), i.top -= Math.min(i.top, i.top + n > l && l > n ? Math.abs(n + o) : 0), i
        },
        _findPos: function(t) {
            for (var i, s = this._getInst(t), a = this._get(s, "isRTL"); t && ("hidden" === t.type || 1 !== t.nodeType || e.expr.filters.hidden(t));)
                t = t[a ? "previousSibling": "nextSibling"];
            return i = e(t).offset(), [i.left, i.top]
        },
        _hideDatepicker: function(t) {
            var i, s, a, n, r = this._curInst;
            !r || t && r !== e.data(t, "datepicker") || this._datepickerShowing && (i = this._get(r, "showAnim"), s = this._get(r, "duration"), a = function() {
                e.datepicker._tidyDialog(r)
            }, e.effects && (e.effects.effect[i] || e.effects[i]) ? r.dpDiv.hide(i, e.datepicker._get(r, "showOptions"), s, a) : r.dpDiv["slideDown" === i ? "slideUp": "fadeIn" === i ? "fadeOut": "hide"](i ? s : null, a), i || a(), this._datepickerShowing=!1, n = this._get(r, "onClose"), n && n.apply(r.input ? r.input[0] : null, [r.input ? r.input.val(): "", r]), this._lastInput = null, this._inDialog && (this._dialogInput.css({
                position: "absolute",
                left: "0",
                top: "-100px"
            }), e.blockUI && (e.unblockUI(), e("body").append(this.dpDiv))), this._inDialog=!1)
        },
        _tidyDialog: function(e) {
            e.dpDiv.removeClass(this._dialogClass).unbind(".ui-datepicker-calendar")
        },
        _checkExternalClick: function(t) {
            if (e.datepicker._curInst) {
                var i = e(t.target), s = e.datepicker._getInst(i[0]);
                (i[0].id !== e.datepicker._mainDivId && 0 === i.parents("#" + e.datepicker._mainDivId).length&&!i.hasClass(e.datepicker.markerClassName)&&!i.closest("." + e.datepicker._triggerClass).length && e.datepicker._datepickerShowing && (!e.datepicker._inDialog ||!e.blockUI) || i.hasClass(e.datepicker.markerClassName) && e.datepicker._curInst !== s) && e.datepicker._hideDatepicker()
            }
        },
        _adjustDate: function(t, i, s) {
            var a = e(t), n = this._getInst(a[0]);
            this._isDisabledDatepicker(a[0]) || (this._adjustInstDate(n, i + ("M" === s ? this._get(n, "showCurrentAtPos") : 0), s), this._updateDatepicker(n))
        },
        _gotoToday: function(t) {
            var i, s = e(t), a = this._getInst(s[0]);
            this._get(a, "gotoCurrent") && a.currentDay ? (a.selectedDay = a.currentDay, a.drawMonth = a.selectedMonth = a.currentMonth, a.drawYear = a.selectedYear = a.currentYear) : (i = new Date, a.selectedDay = i.getDate(), a.drawMonth = a.selectedMonth = i.getMonth(), a.drawYear = a.selectedYear = i.getFullYear()), this._notifyChange(a), this._adjustDate(s)
        },
        _selectMonthYear: function(t, i, s) {
            var a = e(t), n = this._getInst(a[0]);
            n["selected" + ("M" === s ? "Month" : "Year")] = n["draw" + ("M" === s ? "Month" : "Year")] = parseInt(i.options[i.selectedIndex].value, 10), this._notifyChange(n), this._adjustDate(a)
        },
        _selectDay: function(t, i, s, a) {
            var n, r = e(t);
            e(a).hasClass(this._unselectableClass) || this._isDisabledDatepicker(r[0]) || (n = this._getInst(r[0]), n.selectedDay = n.currentDay = e("a", a).html(), n.selectedMonth = n.currentMonth = i, n.selectedYear = n.currentYear = s, this._selectDate(t, this._formatDate(n, n.currentDay, n.currentMonth, n.currentYear)))
        },
        _clearDate: function(t) {
            var i = e(t);
            this._selectDate(i, "")
        },
        _selectDate: function(t, i) {
            var s, a = e(t), n = this._getInst(a[0]);
            i = null != i ? i : this._formatDate(n), n.input && n.input.val(i), this._updateAlternate(n), s = this._get(n, "onSelect"), s ? s.apply(n.input ? n.input[0] : null, [i, n]) : n.input && n.input.trigger("change"), n.inline ? this._updateDatepicker(n) : (this._hideDatepicker(), this._lastInput = n.input[0], "object" != typeof n.input[0] && n.input.focus(), this._lastInput = null)
        },
        _updateAlternate: function(t) {
            var i, s, a, n = this._get(t, "altField");
            n && (i = this._get(t, "altFormat") || this._get(t, "dateFormat"), s = this._getDate(t), a = this.formatDate(i, s, this._getFormatConfig(t)), e(n).each(function() {
                e(this).val(a)
            }))
        },
        noWeekends: function(e) {
            var t = e.getDay();
            return [t > 0 && 6 > t, ""]
        },
        iso8601Week: function(e) {
            var t, i = new Date(e.getTime());
            return i.setDate(i.getDate() + 4 - (i.getDay() || 7)), t = i.getTime(), i.setMonth(0), i.setDate(1), Math.floor(Math.round((t - i) / 864e5) / 7) + 1
        },
        parseDate: function(t, i, s) {
            if (null == t || null == i)
                throw "Invalid arguments";
            if (i = "object" == typeof i ? "" + i : i + "", "" === i)
                return null;
            var a, n, r, o, h = 0, l = (s ? s.shortYearCutoff : null) || this._defaults.shortYearCutoff, u = "string" != typeof l ? l: (new Date).getFullYear()%100 + parseInt(l, 10), d = (s ? s.dayNamesShort : null) || this._defaults.dayNamesShort, c = (s ? s.dayNames : null) || this._defaults.dayNames, p = (s ? s.monthNamesShort : null) || this._defaults.monthNamesShort, f = (s ? s.monthNames : null) || this._defaults.monthNames, m =- 1, g =- 1, v =- 1, y =- 1, b=!1, _ = function(e) {
                var i = t.length > a + 1 && t.charAt(a + 1) === e;
                return i && a++, i
            }, x = function(e) {
                var t = _(e), s = "@" === e ? 14: "!" === e ? 20: "y" === e && t ? 4: "o" === e ? 3: 2, a = "y" === e ? s: 1, n = RegExp("^\\d{" + a + "," + s + "}"), r = i.substring(h).match(n);
                if (!r)
                    throw "Missing number at position " + h;
                return h += r[0].length, parseInt(r[0], 10)
            }, k = function(t, s, a) {
                var n =- 1, r = e.map(_(t) ? a : s, function(e, t) {
                    return [[t, e]]
                }).sort(function(e, t) {
                    return - (e[1].length - t[1].length)
                });
                if (e.each(r, function(e, t) {
                    var s = t[1];
                    return i.substr(h, s.length).toLowerCase() === s.toLowerCase() ? (n = t[0], h += s.length, !1) : void 0
                }), - 1 !== n)
                    return n + 1;
                throw "Unknown name at position " + h
            }, w = function() {
                if (i.charAt(h) !== t.charAt(a))
                    throw "Unexpected literal at position " + h;
                h++
            };
            for (a = 0; t.length > a; a++)
                if (b)
                    "'" !== t.charAt(a) || _("'") ? w() : b=!1;
                else 
                    switch (t.charAt(a)) {
                    case"d":
                        v = x("d");
                        break;
                    case"D":
                        k("D", d, c);
                        break;
                    case"o":
                        y = x("o");
                        break;
                    case"m":
                        g = x("m");
                        break;
                    case"M":
                        g = k("M", p, f);
                        break;
                    case"y":
                        m = x("y");
                        break;
                    case"@":
                        o = new Date(x("@")), m = o.getFullYear(), g = o.getMonth() + 1, v = o.getDate();
                        break;
                    case"!":
                        o = new Date((x("!") - this._ticksTo1970) / 1e4), m = o.getFullYear(), g = o.getMonth() + 1, v = o.getDate();
                        break;
                    case"'":
                        _("'") ? w() : b=!0;
                        break;
                    default:
                        w()
                    }
            if (i.length > h && (r = i.substr(h), !/^\s+/.test(r)))
                throw "Extra/unparsed characters found in date: " + r;
            if ( - 1 === m ? m = (new Date).getFullYear() : 100 > m && (m += (new Date).getFullYear() - (new Date).getFullYear()%100 + (u >= m ? 0 : - 100)), y>-1)
                for (g = 1, v = y; ;) {
                    if (n = this._getDaysInMonth(m, g - 1), n >= v)
                        break;
                        g++, v -= n
                }
            if (o = this._daylightSavingAdjust(new Date(m, g - 1, v)), o.getFullYear() !== m || o.getMonth() + 1 !== g || o.getDate() !== v)
                throw "Invalid date";
            return o
        },
        ATOM: "yy-mm-dd",
        COOKIE: "D, dd M yy",
        ISO_8601: "yy-mm-dd",
        RFC_822: "D, d M y",
        RFC_850: "DD, dd-M-y",
        RFC_1036: "D, d M y",
        RFC_1123: "D, d M yy",
        RFC_2822: "D, d M yy",
        RSS: "D, d M y",
        TICKS: "!",
        TIMESTAMP: "@",
        W3C: "yy-mm-dd",
        _ticksTo1970: 1e7 * 60 * 60 * 24 * (718685 + Math.floor(492.5) - Math.floor(19.7) + Math.floor(4.925)),
        formatDate: function(e, t, i) {
            if (!t)
                return "";
            var s, a = (i ? i.dayNamesShort : null) || this._defaults.dayNamesShort, n = (i ? i.dayNames : null) || this._defaults.dayNames, r = (i ? i.monthNamesShort : null) || this._defaults.monthNamesShort, o = (i ? i.monthNames : null) || this._defaults.monthNames, h = function(t) {
                var i = e.length > s + 1 && e.charAt(s + 1) === t;
                return i && s++, i
            }, l = function(e, t, i) {
                var s = "" + t;
                if (h(e))
                    for (; i > s.length;)
                        s = "0" + s;
                return s
            }, u = function(e, t, i, s) {
                return h(e) ? s[t] : i[t]
            }, d = "", c=!1;
            if (t)
                for (s = 0; e.length > s; s++)
                    if (c)
                        "'" !== e.charAt(s) || h("'") ? d += e.charAt(s) : c=!1;
                    else 
                        switch (e.charAt(s)) {
                        case"d":
                            d += l("d", t.getDate(), 2);
                            break;
                        case"D":
                            d += u("D", t.getDay(), a, n);
                            break;
                        case"o":
                            d += l("o", Math.round((new Date(t.getFullYear(), t.getMonth(), t.getDate()).getTime() - new Date(t.getFullYear(), 0, 0).getTime()) / 864e5), 3);
                            break;
                        case"m":
                            d += l("m", t.getMonth() + 1, 2);
                            break;
                        case"M":
                            d += u("M", t.getMonth(), r, o);
                            break;
                        case"y":
                            d += h("y") ? t.getFullYear() : (10 > t.getYear()%100 ? "0" : "") + t.getYear()%100;
                            break;
                        case"@":
                            d += t.getTime();
                            break;
                        case"!":
                            d += 1e4 * t.getTime() + this._ticksTo1970;
                            break;
                        case"'":
                            h("'") ? d += "'" : c=!0;
                            break;
                        default:
                            d += e.charAt(s)
                        }
            return d
        },
        _possibleChars: function(e) {
            var t, i = "", s=!1, a = function(i) {
                var s = e.length > t + 1 && e.charAt(t + 1) === i;
                return s && t++, s
            };
            for (t = 0; e.length > t; t++)
                if (s)
                    "'" !== e.charAt(t) || a("'") ? i += e.charAt(t) : s=!1;
                else 
                    switch (e.charAt(t)) {
                    case"d":
                    case"m":
                    case"y":
                    case"@":
                        i += "0123456789";
                        break;
                    case"D":
                    case"M":
                        return null;
                    case"'":
                        a("'") ? i += "'" : s=!0;
                        break;
                    default:
                        i += e.charAt(t)
                    }
            return i
        },
        _get: function(e, t) {
            return void 0 !== e.settings[t] ? e.settings[t] : this._defaults[t]
        },
        _setDateFromField: function(e, t) {
            if (e.input.val() !== e.lastVal) {
                var i = this._get(e, "dateFormat"), s = e.lastVal = e.input ? e.input.val(): null, a = this._getDefaultDate(e), n = a, r = this._getFormatConfig(e);
                try {
                    n = this.parseDate(i, s, r) || a
                } catch (o) {
                    s = t ? "" : s
                }
                e.selectedDay = n.getDate(), e.drawMonth = e.selectedMonth = n.getMonth(), e.drawYear = e.selectedYear = n.getFullYear(), e.currentDay = s ? n.getDate() : 0, e.currentMonth = s ? n.getMonth() : 0, e.currentYear = s ? n.getFullYear() : 0, this._adjustInstDate(e)
            }
        },
        _getDefaultDate: function(e) {
            return this._restrictMinMax(e, this._determineDate(e, this._get(e, "defaultDate"), new Date))
        },
        _determineDate: function(t, i, s) {
            var a = function(e) {
                var t = new Date;
                return t.setDate(t.getDate() + e), t
            }, n = function(i) {
                try {
                    return e.datepicker.parseDate(e.datepicker._get(t, "dateFormat"), i, e.datepicker._getFormatConfig(t))
                } catch (s) {}
                for (var a = (i.toLowerCase().match(/^c/) ? e.datepicker._getDate(t) : null) || new Date, n = a.getFullYear(), r = a.getMonth(), o = a.getDate(), h = /([+\-]?[0-9]+)\s*(d|D|w|W|m|M|y|Y)?/g, l = h.exec(i); l;) {
                    switch (l[2] || "d") {
                    case"d":
                    case"D":
                        o += parseInt(l[1], 10);
                        break;
                    case"w":
                    case"W":
                        o += 7 * parseInt(l[1], 10);
                        break;
                    case"m":
                    case"M":
                        r += parseInt(l[1], 10), o = Math.min(o, e.datepicker._getDaysInMonth(n, r));
                        break;
                    case"y":
                    case"Y":
                        n += parseInt(l[1], 10), o = Math.min(o, e.datepicker._getDaysInMonth(n, r))
                    }
                    l = h.exec(i)
                }
                return new Date(n, r, o)
            }, r = null == i || "" === i ? s: "string" == typeof i ? n(i): "number" == typeof i ? isNaN(i) ? s: a(i): new Date(i.getTime());
            return r = r && "Invalid Date" == "" + r ? s : r, r && (r.setHours(0), r.setMinutes(0), r.setSeconds(0), r.setMilliseconds(0)), this._daylightSavingAdjust(r)
        },
        _daylightSavingAdjust: function(e) {
            return e ? (e.setHours(e.getHours() > 12 ? e.getHours() + 2 : 0), e) : null
        },
        _setDate: function(e, t, i) {
            var s=!t, a = e.selectedMonth, n = e.selectedYear, r = this._restrictMinMax(e, this._determineDate(e, t, new Date));
            e.selectedDay = e.currentDay = r.getDate(), e.drawMonth = e.selectedMonth = e.currentMonth = r.getMonth(), e.drawYear = e.selectedYear = e.currentYear = r.getFullYear(), a === e.selectedMonth && n === e.selectedYear || i || this._notifyChange(e), this._adjustInstDate(e), e.input && e.input.val(s ? "" : this._formatDate(e))
        },
        _getDate: function(e) {
            var t=!e.currentYear || e.input && "" === e.input.val() ? null : this._daylightSavingAdjust(new Date(e.currentYear, e.currentMonth, e.currentDay));
            return t
        },
        _attachHandlers: function(t) {
            var i = this._get(t, "stepMonths"), s = "#" + t.id.replace(/\\\\/g, "\\");
            t.dpDiv.find("[data-handler]").map(function() {
                var t = {
                    prev: function() {
                        e.datepicker._adjustDate(s, - i, "M")
                    },
                    next: function() {
                        e.datepicker._adjustDate(s, + i, "M")
                    },
                    hide: function() {
                        e.datepicker._hideDatepicker()
                    },
                    today: function() {
                        e.datepicker._gotoToday(s)
                    },
                    selectDay: function() {
                        return e.datepicker._selectDay(s, + this.getAttribute("data-month"), + this.getAttribute("data-year"), this), !1
                    },
                    selectMonth: function() {
                        return e.datepicker._selectMonthYear(s, this, "M"), !1
                    },
                    selectYear: function() {
                        return e.datepicker._selectMonthYear(s, this, "Y"), !1
                    }
                };
                e(this).bind(this.getAttribute("data-event"), t[this.getAttribute("data-handler")])
            })
        },
        _generateHTML: function(e) {
            var t, i, s, a, n, r, o, h, l, u, d, c, p, f, m, g, v, y, b, _, x, k, w, T, S, D, M, N, C, A, P, I, z, H, F, j, E, O, L, W = new Date, R = this._daylightSavingAdjust(new Date(W.getFullYear(), W.getMonth(), W.getDate())), Y = this._get(e, "isRTL"), J = this._get(e, "showButtonPanel"), B = this._get(e, "hideIfNoPrevNext"), Q = this._get(e, "navigationAsDateFormat"), V = this._getNumberOfMonths(e), K = this._get(e, "showCurrentAtPos"), $ = this._get(e, "stepMonths"), q = 1 !== V[0] || 1 !== V[1], U = this._daylightSavingAdjust(e.currentDay ? new Date(e.currentYear, e.currentMonth, e.currentDay) : new Date(9999, 9, 9)), G = this._getMinMaxDate(e, "min"), X = this._getMinMaxDate(e, "max"), Z = e.drawMonth - K, et = e.drawYear;
            if (0 > Z && (Z += 12, et--), X)
                for (t = this._daylightSavingAdjust(new Date(X.getFullYear(), X.getMonth() - V[0] * V[1] + 1, X.getDate())), t = G && G > t ? G : t; this._daylightSavingAdjust(new Date(et, Z, 1)) > t;)
                    Z--, 0 > Z && (Z = 11, et--);
            for (e.drawMonth = Z, e.drawYear = et, i = this._get(e, "prevText"), i = Q ? this.formatDate(i, this._daylightSavingAdjust(new Date(et, Z - $, 1)), this._getFormatConfig(e)) : i, s = this._canAdjustMonth(e, - 1, et, Z) ? "<a class='ui-datepicker-prev ui-corner-all' data-handler='prev' data-event='click' title='" + i + "'><span class='ui-icon ui-icon-circle-triangle-" + (Y ? "e" : "w") + "'>" + i + "</span></a>" : B ? "" : "<a class='ui-datepicker-prev ui-corner-all ui-state-disabled' title='" + i + "'><span class='ui-icon ui-icon-circle-triangle-" + (Y ? "e" : "w") + "'>" + i + "</span></a>", a = this._get(e, "nextText"), a = Q ? this.formatDate(a, this._daylightSavingAdjust(new Date(et, Z + $, 1)), this._getFormatConfig(e)) : a, n = this._canAdjustMonth(e, 1, et, Z) ? "<a class='ui-datepicker-next ui-corner-all' data-handler='next' data-event='click' title='" + a + "'><span class='ui-icon ui-icon-circle-triangle-" + (Y ? "w" : "e") + "'>" + a + "</span></a>" : B ? "" : "<a class='ui-datepicker-next ui-corner-all ui-state-disabled' title='" + a + "'><span class='ui-icon ui-icon-circle-triangle-" + (Y ? "w" : "e") + "'>" + a + "</span></a>", r = this._get(e, "currentText"), o = this._get(e, "gotoCurrent") && e.currentDay ? U : R, r = Q ? this.formatDate(r, o, this._getFormatConfig(e)) : r, h = e.inline ? "" : "<button type='button' class='ui-datepicker-close ui-state-default ui-priority-primary ui-corner-all' data-handler='hide' data-event='click'>" + this._get(e, "closeText") + "</button>", l = J ? "<div class='ui-datepicker-buttonpane ui-widget-content'>" + (Y ? h : "") + (this._isInRange(e, o) ? "<button type='button' class='ui-datepicker-current ui-state-default ui-priority-secondary ui-corner-all' data-handler='today' data-event='click'>" + r + "</button>" : "") + (Y ? "" : h) + "</div>" : "", u = parseInt(this._get(e, "firstDay"), 10), u = isNaN(u) ? 0 : u, d = this._get(e, "showWeek"), c = this._get(e, "dayNames"), p = this._get(e, "dayNamesMin"), f = this._get(e, "monthNames"), m = this._get(e, "monthNamesShort"), g = this._get(e, "beforeShowDay"), v = this._get(e, "showOtherMonths"), y = this._get(e, "selectOtherMonths"), b = this._getDefaultDate(e), _ = "", k = 0; V[0] > k; k++) {
                for (w = "", this.maxRows = 4, T = 0; V[1] > T; T++) {
                    if (S = this._daylightSavingAdjust(new Date(et, Z, e.selectedDay)), D = " ui-corner-all", M = "", q) {
                        if (M += "<div class='ui-datepicker-group", V[1] > 1)
                            switch (T) {
                            case 0:
                                M += " ui-datepicker-group-first", D = " ui-corner-" + (Y ? "right" : "left");
                                break;
                            case V[1] - 1:
                                M += " ui-datepicker-group-last", D = " ui-corner-" + (Y ? "left" : "right");
                                break;
                            default:
                                M += " ui-datepicker-group-middle", D = ""
                            }
                        M += "'>"
                    }
                    for (M += "<div class='ui-datepicker-header ui-widget-header ui-helper-clearfix" + D + "'>" + (/all|left/.test(D) && 0 === k ? Y ? n : s : "") + (/all|right/.test(D) && 0 === k ? Y ? s : n : "") + this._generateMonthYearHeader(e, Z, et, G, X, k > 0 || T > 0, f, m) + "</div><table class='ui-datepicker-calendar'><thead>" + "<tr>", N = d ? "<th class='ui-datepicker-week-col'>" + this._get(e, "weekHeader") + "</th>" : "", x = 0; 7 > x; x++)
                        C = (x + u)%7, N += "<th scope='col'" + ((x + u + 6)%7 >= 5 ? " class='ui-datepicker-week-end'" : "") + ">" + "<span title='" + c[C] + "'>" + p[C] + "</span></th>";
                    for (M += N + "</tr></thead><tbody>", A = this._getDaysInMonth(et, Z), et === e.selectedYear && Z === e.selectedMonth && (e.selectedDay = Math.min(e.selectedDay, A)), P = (this._getFirstDayOfMonth(et, Z) - u + 7)%7, I = Math.ceil((P + A) / 7), z = q ? this.maxRows > I ? this.maxRows : I : I, this.maxRows = z, H = this._daylightSavingAdjust(new Date(et, Z, 1 - P)), F = 0; z > F; F++) {
                        for (M += "<tr>", j = d ? "<td class='ui-datepicker-week-col'>" + this._get(e, "calculateWeek")(H) + "</td>" : "", x = 0; 7 > x; x++)
                            E = g ? g.apply(e.input ? e.input[0] : null, [H]) : [!0, ""], O = H.getMonth() !== Z, L = O&&!y ||!E[0] || G && G > H || X && H > X, j += "<td class='" + ((x + u + 6)%7 >= 5 ? " ui-datepicker-week-end" : "") + (O ? " ui-datepicker-other-month" : "") + (H.getTime() === S.getTime() && Z === e.selectedMonth && e._keyEvent || b.getTime() === H.getTime() && b.getTime() === S.getTime() ? " " + this._dayOverClass : "") + (L ? " " + this._unselectableClass + " ui-state-disabled" : "") + (O&&!v ? "" : " " + E[1] + (H.getTime() === U.getTime() ? " " + this._currentClass : "") + (H.getTime() === R.getTime() ? " ui-datepicker-today" : "")) + "'" + (O&&!v ||!E[2] ? "" : " title='" + E[2].replace(/'/g, "&#39;") + "'") + (L ? "" : " data-handler='selectDay' data-event='click' data-month='" + H.getMonth() + "' data-year='" + H.getFullYear() + "'") + ">" + (O&&!v ? "&#xa0;" : L ? "<span class='ui-state-default'>" + H.getDate() + "</span>" : "<a class='ui-state-default" + (H.getTime() === R.getTime() ? " ui-state-highlight" : "") + (H.getTime() === U.getTime() ? " ui-state-active" : "") + (O ? " ui-priority-secondary" : "") + "' href='#'>" + H.getDate() + "</a>") + "</td>", H.setDate(H.getDate() + 1), H = this._daylightSavingAdjust(H);
                        M += j + "</tr>"
                    }
                    Z++, Z > 11 && (Z = 0, et++), M += "</tbody></table>" + (q ? "</div>" + (V[0] > 0 && T === V[1] - 1 ? "<div class='ui-datepicker-row-break'></div>" : "") : ""), w += M
                }
                _ += w
            }
            return _ += l, e._keyEvent=!1, _
        },
        _generateMonthYearHeader: function(e, t, i, s, a, n, r, o) {
            var h, l, u, d, c, p, f, m, g = this._get(e, "changeMonth"), v = this._get(e, "changeYear"), y = this._get(e, "showMonthAfterYear"), b = "<div class='ui-datepicker-title'>", _ = "";
            if (n ||!g)
                _ += "<span class='ui-datepicker-month'>" + r[t] + "</span>";
            else {
                for (h = s && s.getFullYear() === i, l = a && a.getFullYear() === i, _ += "<select class='ui-datepicker-month' data-handler='selectMonth' data-event='change'>", u = 0; 12 > u; u++)(!h || u >= s.getMonth()
                    ) && (!l || a.getMonth() >= u) && (_ += "<option value='" + u + "'" + (u === t ? " selected='selected'" : "") + ">" + o[u] + "</option>");
                _ += "</select>"
            }
            if (y || (b += _ + (!n && g && v ? "" : "&#xa0;")), !e.yearshtml)
                if (e.yearshtml = "", n ||!v)
                    b += "<span class='ui-datepicker-year'>" + i + "</span>";
                else {
                    for (d = this._get(e, "yearRange").split(":"), c = (new Date).getFullYear(), p = function(e) {
                        var t = e.match(/c[+\-].*/) ? i + parseInt(e.substring(1), 10): e.match(/[+\-].*/) ? c + parseInt(e, 10): parseInt(e, 10);
                        return isNaN(t) ? c : t
                    }, f = p(d[0]), m = Math.max(f, p(d[1] || "")), f = s ? Math.max(f, s.getFullYear()) : f, m = a ? Math.min(m, a.getFullYear()) : m, e.yearshtml += "<select class='ui-datepicker-year' data-handler='selectYear' data-event='change'>"; m >= f; f++)
                        e.yearshtml += "<option value='" + f + "'" + (f === i ? " selected='selected'" : "") + ">" + f + "</option>";
                        e.yearshtml += "</select>", b += e.yearshtml, e.yearshtml = null
                }
            return b += this._get(e, "yearSuffix"), y && (b += (!n && g && v ? "" : "&#xa0;") + _), b += "</div>"
        },
        _adjustInstDate: function(e, t, i) {
            var s = e.drawYear + ("Y" === i ? t : 0), a = e.drawMonth + ("M" === i ? t : 0), n = Math.min(e.selectedDay, this._getDaysInMonth(s, a)) + ("D" === i ? t : 0), r = this._restrictMinMax(e, this._daylightSavingAdjust(new Date(s, a, n)));
            e.selectedDay = r.getDate(), e.drawMonth = e.selectedMonth = r.getMonth(), e.drawYear = e.selectedYear = r.getFullYear(), ("M" === i || "Y" === i) && this._notifyChange(e)
        },
        _restrictMinMax: function(e, t) {
            var i = this._getMinMaxDate(e, "min"), s = this._getMinMaxDate(e, "max"), a = i && i > t ? i: t;
            return s && a > s ? s : a
        },
        _notifyChange: function(e) {
            var t = this._get(e, "onChangeMonthYear");
            t && t.apply(e.input ? e.input[0] : null, [e.selectedYear, e.selectedMonth + 1, e])
        },
        _getNumberOfMonths: function(e) {
            var t = this._get(e, "numberOfMonths");
            return null == t ? [1, 1] : "number" == typeof t ? [1, t] : t
        },
        _getMinMaxDate: function(e, t) {
            return this._determineDate(e, this._get(e, t + "Date"), null)
        },
        _getDaysInMonth: function(e, t) {
            return 32 - this._daylightSavingAdjust(new Date(e, t, 32)).getDate()
        },
        _getFirstDayOfMonth: function(e, t) {
            return new Date(e, t, 1).getDay()
        },
        _canAdjustMonth: function(e, t, i, s) {
            var a = this._getNumberOfMonths(e), n = this._daylightSavingAdjust(new Date(i, s + (0 > t ? t : a[0] * a[1]), 1));
            return 0 > t && n.setDate(this._getDaysInMonth(n.getFullYear(), n.getMonth())), this._isInRange(e, n)
        },
        _isInRange: function(e, t) {
            var i, s, a = this._getMinMaxDate(e, "min"), n = this._getMinMaxDate(e, "max"), r = null, o = null, h = this._get(e, "yearRange");
            return h && (i = h.split(":"), s = (new Date).getFullYear(), r = parseInt(i[0], 10), o = parseInt(i[1], 10), i[0].match(/[+\-].*/) && (r += s), i[1].match(/[+\-].*/) && (o += s)), (!a || t.getTime() >= a.getTime()) && (!n || t.getTime() <= n.getTime()) && (!r || t.getFullYear() >= r) && (!o || o >= t.getFullYear())
        },
        _getFormatConfig: function(e) {
            var t = this._get(e, "shortYearCutoff");
            return t = "string" != typeof t ? t : (new Date).getFullYear()%100 + parseInt(t, 10), {
                shortYearCutoff: t,
                dayNamesShort: this._get(e, "dayNamesShort"),
                dayNames: this._get(e, "dayNames"),
                monthNamesShort: this._get(e, "monthNamesShort"),
                monthNames: this._get(e, "monthNames")
            }
        },
        _formatDate: function(e, t, i, s) {
            t || (e.currentDay = e.selectedDay, e.currentMonth = e.selectedMonth, e.currentYear = e.selectedYear);
            var a = t ? "object" == typeof t ? t: this._daylightSavingAdjust(new Date(s, i, t)): this._daylightSavingAdjust(new Date(e.currentYear, e.currentMonth, e.currentDay));
            return this.formatDate(this._get(e, "dateFormat"), a, this._getFormatConfig(e))
        }
    }), e.fn.datepicker = function(t) {
        if (!this.length)
            return this;
        e.datepicker.initialized || (e(document).mousedown(e.datepicker._checkExternalClick), e.datepicker.initialized=!0), 0 === e("#" + e.datepicker._mainDivId).length && e("body").append(e.datepicker.dpDiv);
        var i = Array.prototype.slice.call(arguments, 1);
        return "string" != typeof t || "isDisabled" !== t && "getDate" !== t && "widget" !== t ? "option" === t && 2 === arguments.length && "string" == typeof arguments[1] ? e.datepicker["_" + t + "Datepicker"].apply(e.datepicker, [this[0]].concat(i)) : this.each(function() {
            "string" == typeof t ? e.datepicker["_" + t + "Datepicker"].apply(e.datepicker, [this].concat(i)) : e.datepicker._attachDatepicker(this, t)
        }) : e.datepicker["_" + t + "Datepicker"].apply(e.datepicker, [this[0]].concat(i))
    }, e.datepicker = new a, e.datepicker.initialized=!1, e.datepicker.uuid = (new Date).getTime(), e.datepicker.version = "1.11.1", e.datepicker, e.widget("ui.dialog", {
        version: "1.11.1",
        options: {
            appendTo: "body",
            autoOpen: !0,
            buttons: [],
            closeOnEscape: !0,
            closeText: "Close",
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
        sizeRelatedOptions: {
            buttons: !0,
            height: !0,
            maxHeight: !0,
            maxWidth: !0,
            minHeight: !0,
            minWidth: !0,
            width: !0
        },
        resizableRelatedOptions: {
            maxHeight: !0,
            maxWidth: !0,
            minHeight: !0,
            minWidth: !0
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
            }, this.originalTitle = this.element.attr("title"), this.options.title = this.options.title || this.originalTitle, this._createWrapper(), this.element.show().removeAttr("title").addClass("ui-dialog-content ui-widget-content").appendTo(this.uiDialog), this._createTitlebar(), this._createButtonPane(), this.options.draggable && e.fn.draggable && this._makeDraggable(), this.options.resizable && e.fn.resizable && this._makeResizable(), this._isOpen=!1, this._trackFocus()
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
            var i, s = this;
            if (this._isOpen && this._trigger("beforeClose", t)!==!1) {
                if (this._isOpen=!1, this._focusedElement = null, this._destroyOverlay(), this._untrackInstance(), !this.opener.filter(":focusable").focus().length)
                    try {
                        i = this.document[0].activeElement, i && "body" !== i.nodeName.toLowerCase() && e(i).blur()
                    } catch (a) {}
                this._hide(this.uiDialog, this.options.hide, function() {
                    s._trigger("close", t)
                })
            }
        },
        isOpen: function() {
            return this._isOpen
        },
        moveToTop: function() {
            this._moveToTop()
        },
        _moveToTop: function(t, i) {
            var s=!1, a = this.uiDialog.siblings(".ui-front:visible").map(function() {
                return + e(this).css("z-index")
            }).get(), n = Math.max.apply(null, a);
            return n>=+this.uiDialog.css("z-index") && (this.uiDialog.css("z-index", n + 1), s=!0), s&&!i && this._trigger("focus", t), s
        },
        open: function() {
            var t = this;
            return this._isOpen ? (this._moveToTop() && this._focusTabbable(), void 0) : (this._isOpen=!0, this.opener = e(this.document[0].activeElement), this._size(), this._position(), this._createOverlay(), this._moveToTop(null, !0), this.overlay && this.overlay.css("z-index", this.uiDialog.css("z-index") - 1), this._show(this.uiDialog, this.options.show, function() {
                t._focusTabbable(), t._trigger("focus")
            }), this._makeFocusTarget(), this._trigger("open"), void 0)
        },
        _focusTabbable: function() {
            var e = this._focusedElement;
            e || (e = this.element.find("[autofocus]")), e.length || (e = this.element.find(":tabbable")), e.length || (e = this.uiDialogButtonPane.find(":tabbable")), e.length || (e = this.uiDialogTitlebarClose.filter(":tabbable")), e.length || (e = this.uiDialog), e.eq(0).focus()
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
                    if (t.keyCode === e.ui.keyCode.TAB&&!t.isDefaultPrevented()) {
                        var i = this.uiDialog.find(":tabbable"), s = i.filter(":first"), a = i.filter(":last");
                        t.target !== a[0] && t.target !== this.uiDialog[0] || t.shiftKey ? t.target !== s[0] && t.target !== this.uiDialog[0] ||!t.shiftKey || (this._delay(function() {
                            a.focus()
                        }), t.preventDefault()) : (this._delay(function() {
                            s.focus()
                        }), t.preventDefault())
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
            }), this.uiDialogTitlebarClose = e("<button type='button'></button>").button({
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
            return this.uiDialogButtonPane.remove(), this.uiButtonSet.empty(), e.isEmptyObject(i) || e.isArray(i)&&!i.length ? (this.uiDialog.removeClass("ui-dialog-buttons"), void 0) : (e.each(i, function(i, s) {
                var a, n;
                s = e.isFunction(s) ? {
                    click: s,
                    text: i
                } : s, s = e.extend({
                    type: "button"
                }, s), a = s.click, s.click = function() {
                    a.apply(t.element[0], arguments)
                }, n = {
                    icons: s.icons,
                    text: s.showText
                }, delete s.icons, delete s.showText, e("<button></button>", s).button(n).appendTo(t.uiButtonSet)
            }), this.uiDialog.addClass("ui-dialog-buttons"), this.uiDialogButtonPane.appendTo(this.uiDialog), void 0)
        },
        _makeDraggable: function() {
            function t(e) {
                return {
                    position: e.position,
                    offset: e.offset
                }
            }
            var i = this, s = this.options;
            this.uiDialog.draggable({
                cancel: ".ui-dialog-content, .ui-dialog-titlebar-close",
                handle: ".ui-dialog-titlebar",
                containment: "document",
                start: function(s, a) {
                    e(this).addClass("ui-dialog-dragging"), i._blockFrames(), i._trigger("dragStart", s, t(a))
                },
                drag: function(e, s) {
                    i._trigger("drag", e, t(s))
                },
                stop: function(a, n) {
                    var r = n.offset.left - i.document.scrollLeft(), o = n.offset.top - i.document.scrollTop();
                    s.position = {
                        my: "left top",
                        at: "left" + (r >= 0 ? "+" : "") + r + " " + "top" + (o >= 0 ? "+" : "") + o,
                        of: i.window
                    }, e(this).removeClass("ui-dialog-dragging"), i._unblockFrames(), i._trigger("dragStop", a, t(n))
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
            var i = this, s = this.options, a = s.resizable, n = this.uiDialog.css("position"), r = "string" == typeof a ? a: "n,e,s,w,se,sw,ne,nw";
            this.uiDialog.resizable({
                cancel: ".ui-dialog-content",
                containment: "document",
                alsoResize: this.element,
                maxWidth: s.maxWidth,
                maxHeight: s.maxHeight,
                minWidth: s.minWidth,
                minHeight: this._minHeight(),
                handles: r,
                start: function(s, a) {
                    e(this).addClass("ui-dialog-resizing"), i._blockFrames(), i._trigger("resizeStart", s, t(a))
                },
                resize: function(e, s) {
                    i._trigger("resize", e, t(s))
                },
                stop: function(a, n) {
                    var r = i.uiDialog.offset(), o = r.left - i.document.scrollLeft(), h = r.top - i.document.scrollTop();
                    s.height = i.uiDialog.height(), s.width = i.uiDialog.width(), s.position = {
                        my: "left top",
                        at: "left" + (o >= 0 ? "+" : "") + o + " " + "top" + (h >= 0 ? "+" : "") + h,
                        of: i.window
                    }, e(this).removeClass("ui-dialog-resizing"), i._unblockFrames(), i._trigger("resizeStop", a, t(n))
                }
            }).css("position", n)
        },
        _trackFocus: function() {
            this._on(this.widget(), {
                focusin: function(t) {
                    this._makeFocusTarget(), this._focusedElement = e(t.target)
                }
            })
        },
        _makeFocusTarget: function() {
            this._untrackInstance(), this._trackingInstances().unshift(this)
        },
        _untrackInstance: function() {
            var t = this._trackingInstances(), i = e.inArray(this, t);
            - 1 !== i && t.splice(i, 1)
        },
        _trackingInstances: function() {
            var e = this.document.data("ui-dialog-instances");
            return e || (e = [], this.document.data("ui-dialog-instances", e)), e
        },
        _minHeight: function() {
            var e = this.options;
            return "auto" === e.height ? e.minHeight : Math.min(e.minHeight, e.height)
        },
        _position: function() {
            var e = this.uiDialog.is(":visible");
            e || this.uiDialog.show(), this.uiDialog.position(this.options.position), e || this.uiDialog.hide()
        },
        _setOptions: function(t) {
            var i = this, s=!1, a = {};
            e.each(t, function(e, t) {
                i._setOption(e, t), e in i.sizeRelatedOptions && (s=!0), e in i.resizableRelatedOptions && (a[e] = t)
            }), s && (this._size(), this._position()), this.uiDialog.is(":data(ui-resizable)") && this.uiDialog.resizable("option", a)
        },
        _setOption: function(e, t) {
            var i, s, a = this.uiDialog;
            "dialogClass" === e && a.removeClass(this.options.dialogClass).addClass(t), "disabled" !== e && (this._super(e, t), "appendTo" === e && this.uiDialog.appendTo(this._appendTo()), "buttons" === e && this._createButtons(), "closeText" === e && this.uiDialogTitlebarClose.button({
                label: "" + t
            }), "draggable" === e && (i = a.is(":data(ui-draggable)"), i&&!t && a.draggable("destroy"), !i && t && this._makeDraggable()), "position" === e && this._position(), "resizable" === e && (s = a.is(":data(ui-resizable)"), s&&!t && a.resizable("destroy"), s && "string" == typeof t && a.resizable("option", "handles", t), s || t===!1 || this._makeResizable()), "title" === e && this._title(this.uiDialogTitlebar.find(".ui-dialog-title")))
        },
        _size: function() {
            var e, t, i, s = this.options;
            this.element.show().css({
                width: "auto",
                minHeight: 0,
                maxHeight: "none",
                height: 0
            }), s.minWidth > s.width && (s.width = s.minWidth), e = this.uiDialog.css({
                height: "auto",
                width: s.width
            }).outerHeight(), t = Math.max(0, s.minHeight - e), i = "number" == typeof s.maxHeight ? Math.max(0, s.maxHeight - e) : "none", "auto" === s.height ? this.element.css({
                minHeight: t,
                maxHeight: i,
                height: "auto"
            }) : this.element.height(Math.max(0, s.height - e)), this.uiDialog.is(":data(ui-resizable)") && this.uiDialog.resizable("option", "minHeight", this._minHeight())
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
                var t=!0;
                this._delay(function() {
                    t=!1
                }), this.document.data("ui-dialog-overlays") || this._on(this.document, {
                    focusin: function(e) {
                        t || this._allowInteraction(e) || (e.preventDefault(), this._trackingInstances()[0]._focusTabbable())
                    }
                }), this.overlay = e("<div>").addClass("ui-widget-overlay ui-front").appendTo(this._appendTo()), this._on(this.overlay, {
                    mousedown: "_keepFocus"
                }), this.document.data("ui-dialog-overlays", (this.document.data("ui-dialog-overlays") || 0) + 1)
            }
        },
        _destroyOverlay: function() {
            if (this.options.modal && this.overlay) {
                var e = this.document.data("ui-dialog-overlays") - 1;
                e ? this.document.data("ui-dialog-overlays", e) : this.document.unbind("focusin").removeData("ui-dialog-overlays"), this.overlay.remove(), this.overlay = null
            }
        }
    }), e.widget("ui.progressbar", {
        version: "1.11.1",
        options: {
            max: 100,
            value: 0,
            change: null,
            complete: null
        },
        min: 0,
        _create: function() {
            this.oldValue = this.options.value = this._constrainedValue(), this.element.addClass("ui-progressbar ui-widget ui-widget-content ui-corner-all").attr({
                role: "progressbar",
                "aria-valuemin": this.min
            }), this.valueDiv = e("<div class='ui-progressbar-value ui-widget-header ui-corner-left'></div>").appendTo(this.element), this._refreshValue()
        },
        _destroy: function() {
            this.element.removeClass("ui-progressbar ui-widget ui-widget-content ui-corner-all").removeAttr("role").removeAttr("aria-valuemin").removeAttr("aria-valuemax").removeAttr("aria-valuenow"), this.valueDiv.remove()
        },
        value: function(e) {
            return void 0 === e ? this.options.value : (this.options.value = this._constrainedValue(e), this._refreshValue(), void 0)
        },
        _constrainedValue: function(e) {
            return void 0 === e && (e = this.options.value), this.indeterminate = e===!1, "number" != typeof e && (e = 0), this.indeterminate?!1 : Math.min(this.options.max, Math.max(this.min, e))
        },
        _setOptions: function(e) {
            var t = e.value;
            delete e.value, this._super(e), this.options.value = this._constrainedValue(t), this._refreshValue()
        },
        _setOption: function(e, t) {
            "max" === e && (t = Math.max(this.min, t)), "disabled" === e && this.element.toggleClass("ui-state-disabled", !!t).attr("aria-disabled", t), this._super(e, t)
        },
        _percentage: function() {
            return this.indeterminate ? 100 : 100 * (this.options.value - this.min) / (this.options.max - this.min)
        },
        _refreshValue: function() {
            var t = this.options.value, i = this._percentage();
            this.valueDiv.toggle(this.indeterminate || t > this.min).toggleClass("ui-corner-right", t === this.options.max).width(i.toFixed(0) + "%"), this.element.toggleClass("ui-progressbar-indeterminate", this.indeterminate), this.indeterminate ? (this.element.removeAttr("aria-valuenow"), this.overlayDiv || (this.overlayDiv = e("<div class='ui-progressbar-overlay'></div>").appendTo(this.valueDiv))) : (this.element.attr({
                "aria-valuemax": this.options.max,
                "aria-valuenow": t
            }), this.overlayDiv && (this.overlayDiv.remove(), this.overlayDiv = null)), this.oldValue !== t && (this.oldValue = t, this._trigger("change")), t === this.options.max && this._trigger("complete")
        }
    }), e.widget("ui.selectmenu", {
        version: "1.11.1",
        defaultElement: "<select>",
        options: {
            appendTo: null,
            disabled: null,
            icons: {
                button: "ui-icon-triangle-1-s"
            },
            position: {
                my: "left top",
                at: "left bottom",
                collision: "none"
            },
            width: null,
            change: null,
            close: null,
            focus: null,
            open: null,
            select: null
        },
        _create: function() {
            var e = this.element.uniqueId().attr("id");
            this.ids = {
                element: e,
                button: e + "-button",
                menu: e + "-menu"
            }, this._drawButton(), this._drawMenu(), this.options.disabled && this.disable()
        },
        _drawButton: function() {
            var t = this, i = this.element.attr("tabindex");
            this.label = e("label[for='" + this.ids.element + "']").attr("for", this.ids.button), this._on(this.label, {
                click: function(e) {
                    this.button.focus(), e.preventDefault()
                }
            }), this.element.hide(), this.button = e("<span>", {
                "class": "ui-selectmenu-button ui-widget ui-state-default ui-corner-all",
                tabindex: i || this.options.disabled?-1: 0,
                id: this.ids.button,
                role: "combobox",
                "aria-expanded": "false",
                "aria-autocomplete": "list",
                "aria-owns": this.ids.menu,
                "aria-haspopup": "true"
            }).insertAfter(this.element), e("<span>", {
                "class": "ui-icon " + this.options.icons.button
            }).prependTo(this.button), this.buttonText = e("<span>", {
                "class": "ui-selectmenu-text"
            }).appendTo(this.button), this._setText(this.buttonText, this.element.find("option:selected").text()), this._resizeButton(), this._on(this.button, this._buttonEvents), this.button.one("focusin", function() {
                t.menuItems || t._refreshMenu()
            }), this._hoverable(this.button), this._focusable(this.button)
        },
        _drawMenu: function() {
            var t = this;
            this.menu = e("<ul>", {
                "aria-hidden": "true",
                "aria-labelledby": this.ids.button,
                id: this.ids.menu
            }), this.menuWrap = e("<div>", {
                "class": "ui-selectmenu-menu ui-front"
            }).append(this.menu).appendTo(this._appendTo()), this.menuInstance = this.menu.menu({
                role: "listbox",
                select: function(e, i) {
                    e.preventDefault(), t._select(i.item.data("ui-selectmenu-item"), e)
                },
                focus: function(e, i) {
                    var s = i.item.data("ui-selectmenu-item");
                    null != t.focusIndex && s.index !== t.focusIndex && (t._trigger("focus", e, {
                        item: s
                    }), t.isOpen || t._select(s, e)), t.focusIndex = s.index, t.button.attr("aria-activedescendant", t.menuItems.eq(s.index).attr("id"))
                }
            }).menu("instance"), this.menu.addClass("ui-corner-bottom").removeClass("ui-corner-all"), this.menuInstance._off(this.menu, "mouseleave"), this.menuInstance._closeOnDocumentClick = function() {
                return !1
            }, this.menuInstance._isDivider = function() {
                return !1
            }
        },
        refresh: function() {
            this._refreshMenu(), this._setText(this.buttonText, this._getSelectedItem().text()), this.options.width || this._resizeButton()
        },
        _refreshMenu: function() {
            this.menu.empty();
            var e, t = this.element.find("option");
            t.length && (this._parseOptions(t), this._renderMenu(this.menu, this.items), this.menuInstance.refresh(), this.menuItems = this.menu.find("li").not(".ui-selectmenu-optgroup"), e = this._getSelectedItem(), this.menuInstance.focus(null, e), this._setAria(e.data("ui-selectmenu-item")), this._setOption("disabled", this.element.prop("disabled")))
        },
        open: function(e) {
            this.options.disabled || (this.menuItems ? (this.menu.find(".ui-state-focus").removeClass("ui-state-focus"), this.menuInstance.focus(null, this._getSelectedItem())) : this._refreshMenu(), this.isOpen=!0, this._toggleAttr(), this._resizeMenu(), this._position(), this._on(this.document, this._documentClick), this._trigger("open", e))
        },
        _position: function() {
            this.menuWrap.position(e.extend({
                of: this.button
            }, this.options.position))
        },
        close: function(e) {
            this.isOpen && (this.isOpen=!1, this._toggleAttr(), this._off(this.document), this._trigger("close", e))
        },
        widget: function() {
            return this.button
        },
        menuWidget: function() {
            return this.menu
        },
        _renderMenu: function(t, i) {
            var s = this, a = "";
            e.each(i, function(i, n) {
                n.optgroup !== a && (e("<li>", {
                    "class": "ui-selectmenu-optgroup ui-menu-divider" + (n.element.parent("optgroup").prop("disabled") ? " ui-state-disabled" : ""),
                    text: n.optgroup
                }).appendTo(t), a = n.optgroup), s._renderItemData(t, n)
            })
        },
        _renderItemData: function(e, t) {
            return this._renderItem(e, t).data("ui-selectmenu-item", t)
        },
        _renderItem: function(t, i) {
            var s = e("<li>");
            return i.disabled && s.addClass("ui-state-disabled"), this._setText(s, i.label), s.appendTo(t)
        },
        _setText: function(e, t) {
            t ? e.text(t) : e.html("&#160;")
        },
        _move: function(e, t) {
            var i, s, a = ".ui-menu-item";
            this.isOpen ? i = this.menuItems.eq(this.focusIndex) : (i = this.menuItems.eq(this.element[0].selectedIndex), a += ":not(.ui-state-disabled)"), s = "first" === e || "last" === e ? i["first" === e ? "prevAll": "nextAll"](a).eq( - 1) : i[e + "All"](a).eq(0), s.length && this.menuInstance.focus(t, s)
        },
        _getSelectedItem: function() {
            return this.menuItems.eq(this.element[0].selectedIndex)
        },
        _toggle: function(e) {
            this[this.isOpen ? "close": "open"](e)
        },
        _documentClick: {
            mousedown: function(t) {
                this.isOpen && (e(t.target).closest(".ui-selectmenu-menu, #" + this.ids.button).length || this.close(t))
            }
        },
        _buttonEvents: {
            mousedown: function(e) {
                e.preventDefault()
            },
            click: "_toggle",
            keydown: function(t) {
                var i=!0;
                switch (t.keyCode) {
                case e.ui.keyCode.TAB:
                case e.ui.keyCode.ESCAPE:
                    this.close(t), i=!1;
                    break;
                case e.ui.keyCode.ENTER:
                    this.isOpen && this._selectFocusedItem(t);
                    break;
                case e.ui.keyCode.UP:
                    t.altKey ? this._toggle(t) : this._move("prev", t);
                    break;
                case e.ui.keyCode.DOWN:
                    t.altKey ? this._toggle(t) : this._move("next", t);
                    break;
                case e.ui.keyCode.SPACE:
                    this.isOpen ? this._selectFocusedItem(t) : this._toggle(t);
                    break;
                case e.ui.keyCode.LEFT:
                    this._move("prev", t);
                    break;
                case e.ui.keyCode.RIGHT:
                    this._move("next", t);
                    break;
                case e.ui.keyCode.HOME:
                case e.ui.keyCode.PAGE_UP:
                    this._move("first", t);
                    break;
                case e.ui.keyCode.END:
                case e.ui.keyCode.PAGE_DOWN:
                    this._move("last", t);
                    break;
                default:
                    this.menu.trigger(t), i=!1
                }
                i && t.preventDefault()
            }
        },
        _selectFocusedItem: function(e) {
            var t = this.menuItems.eq(this.focusIndex);
            t.hasClass("ui-state-disabled") || this._select(t.data("ui-selectmenu-item"), e)
        },
        _select: function(e, t) {
            var i = this.element[0].selectedIndex;
            this.element[0].selectedIndex = e.index, this._setText(this.buttonText, e.label), this._setAria(e), this._trigger("select", t, {
                item: e
            }), e.index !== i && this._trigger("change", t, {
                item: e
            }), this.close(t)
        },
        _setAria: function(e) {
            var t = this.menuItems.eq(e.index).attr("id");
            this.button.attr({
                "aria-labelledby": t,
                "aria-activedescendant": t
            }), this.menu.attr("aria-activedescendant", t)
        },
        _setOption: function(e, t) {
            "icons" === e && this.button.find("span.ui-icon").removeClass(this.options.icons.button).addClass(t.button), this._super(e, t), "appendTo" === e && this.menuWrap.appendTo(this._appendTo()), "disabled" === e && (this.menuInstance.option("disabled", t), this.button.toggleClass("ui-state-disabled", t).attr("aria-disabled", t), this.element.prop("disabled", t), t ? (this.button.attr("tabindex", - 1), this.close()) : this.button.attr("tabindex", 0)), "width" === e && this._resizeButton()
        },
        _appendTo: function() {
            var t = this.options.appendTo;
            return t && (t = t.jquery || t.nodeType ? e(t) : this.document.find(t).eq(0)), t && t[0] || (t = this.element.closest(".ui-front")), t.length || (t = this.document[0].body), t
        },
        _toggleAttr: function() {
            this.button.toggleClass("ui-corner-top", this.isOpen).toggleClass("ui-corner-all", !this.isOpen).attr("aria-expanded", this.isOpen), this.menuWrap.toggleClass("ui-selectmenu-open", this.isOpen), this.menu.attr("aria-hidden", !this.isOpen)
        },
        _resizeButton: function() {
            var e = this.options.width;
            e || (e = this.element.show().outerWidth(), this.element.hide()), this.button.outerWidth(e)
        },
        _resizeMenu: function() {
            this.menu.outerWidth(Math.max(this.button.outerWidth(), this.menu.width("").outerWidth() + 1))
        },
        _getCreateOptions: function() {
            return {
                disabled: this.element.prop("disabled")
            }
        },
        _parseOptions: function(t) {
            var i = [];
            t.each(function(t, s) {
                var a = e(s), n = a.parent("optgroup");
                i.push({
                    element: a,
                    index: t,
                    value: a.attr("value"),
                    label: a.text(),
                    optgroup: n.attr("label") || "",
                    disabled: n.prop("disabled") || a.prop("disabled")
                })
            }), this.items = i
        },
        _destroy: function() {
            this.menuWrap.remove(), this.button.remove(), this.element.show(), this.element.removeUniqueId(), this.label.attr("for", this.ids.element)
        }
    }), e.widget("ui.slider", e.ui.mouse, {
        version: "1.11.1",
        widgetEventPrefix: "slide",
        options: {
            animate: !1,
            distance: 0,
            max: 100,
            min: 0,
            orientation: "horizontal",
            range: !1,
            step: 1,
            value: 0,
            values: null,
            change: null,
            slide: null,
            start: null,
            stop: null
        },
        numPages: 5,
        _create: function() {
            this._keySliding=!1, this._mouseSliding=!1, this._animateOff=!0, this._handleIndex = null, this._detectOrientation(), this._mouseInit(), this.element.addClass("ui-slider ui-slider-" + this.orientation + " ui-widget" + " ui-widget-content" + " ui-corner-all"), this._refresh(), this._setOption("disabled", this.options.disabled), this._animateOff=!1
        },
        _refresh: function() {
            this._createRange(), this._createHandles(), this._setupEvents(), this._refreshValue()
        },
        _createHandles: function() {
            var t, i, s = this.options, a = this.element.find(".ui-slider-handle").addClass("ui-state-default ui-corner-all"), n = "<span class='ui-slider-handle ui-state-default ui-corner-all' tabindex='0'></span>", r = [];
            for (i = s.values && s.values.length || 1, a.length > i && (a.slice(i).remove(), a = a.slice(0, i)), t = a.length; i > t; t++)
                r.push(n);
            this.handles = a.add(e(r.join("")).appendTo(this.element)), this.handle = this.handles.eq(0), this.handles.each(function(t) {
                e(this).data("ui-slider-handle-index", t)
            })
        },
        _createRange: function() {
            var t = this.options, i = "";
            t.range ? (t.range===!0 && (t.values ? t.values.length && 2 !== t.values.length ? t.values = [t.values[0], t.values[0]] : e.isArray(t.values) && (t.values = t.values.slice(0)) : t.values = [this._valueMin(), this._valueMin()]), this.range && this.range.length ? this.range.removeClass("ui-slider-range-min ui-slider-range-max").css({
                left: "",
                bottom: ""
            }) : (this.range = e("<div></div>").appendTo(this.element), i = "ui-slider-range ui-widget-header ui-corner-all"), this.range.addClass(i + ("min" === t.range || "max" === t.range ? " ui-slider-range-" + t.range : ""))) : (this.range && this.range.remove(), this.range = null)
        },
        _setupEvents: function() {
            this._off(this.handles), this._on(this.handles, this._handleEvents), this._hoverable(this.handles), this._focusable(this.handles)
        },
        _destroy: function() {
            this.handles.remove(), this.range && this.range.remove(), this.element.removeClass("ui-slider ui-slider-horizontal ui-slider-vertical ui-widget ui-widget-content ui-corner-all"), this._mouseDestroy()
        },
        _mouseCapture: function(t) {
            var i, s, a, n, r, o, h, l, u = this, d = this.options;
            return d.disabled?!1 : (this.elementSize = {
                width: this.element.outerWidth(),
                height: this.element.outerHeight()
            }, this.elementOffset = this.element.offset(), i = {
                x: t.pageX,
                y: t.pageY
            }, s = this._normValueFromMouse(i), a = this._valueMax() - this._valueMin() + 1, this.handles.each(function(t) {
                var i = Math.abs(s - u.values(t));
                (a > i || a === i && (t === u._lastChangedValue || u.values(t) === d.min)) && (a = i, n = e(this), r = t)
            }), o = this._start(t, r), o===!1?!1 : (this._mouseSliding=!0, this._handleIndex = r, n.addClass("ui-state-active").focus(), h = n.offset(), l=!e(t.target).parents().addBack().is(".ui-slider-handle"), this._clickOffset = l ? {
                left: 0,
                top: 0
            } : {
                left: t.pageX - h.left - n.width() / 2,
                top: t.pageY - h.top - n.height() / 2 - (parseInt(n.css("borderTopWidth"), 10) || 0) - (parseInt(n.css("borderBottomWidth"), 10) || 0) + (parseInt(n.css("marginTop"), 10) || 0)
            }, this.handles.hasClass("ui-state-hover") || this._slide(t, r, s), this._animateOff=!0, !0))
        },
        _mouseStart: function() {
            return !0
        },
        _mouseDrag: function(e) {
            var t = {
                x: e.pageX,
                y: e.pageY
            }, i = this._normValueFromMouse(t);
            return this._slide(e, this._handleIndex, i), !1
        },
        _mouseStop: function(e) {
            return this.handles.removeClass("ui-state-active"), this._mouseSliding=!1, this._stop(e, this._handleIndex), this._change(e, this._handleIndex), this._handleIndex = null, this._clickOffset = null, this._animateOff=!1, !1
        },
        _detectOrientation: function() {
            this.orientation = "vertical" === this.options.orientation ? "vertical" : "horizontal"
        },
        _normValueFromMouse: function(e) {
            var t, i, s, a, n;
            return "horizontal" === this.orientation ? (t = this.elementSize.width, i = e.x - this.elementOffset.left - (this._clickOffset ? this._clickOffset.left : 0)) : (t = this.elementSize.height, i = e.y - this.elementOffset.top - (this._clickOffset ? this._clickOffset.top : 0)), s = i / t, s > 1 && (s = 1), 0 > s && (s = 0), "vertical" === this.orientation && (s = 1 - s), a = this._valueMax() - this._valueMin(), n = this._valueMin() + s * a, this._trimAlignValue(n)
        },
        _start: function(e, t) {
            var i = {
                handle: this.handles[t],
                value: this.value()
            };
            return this.options.values && this.options.values.length && (i.value = this.values(t), i.values = this.values()), this._trigger("start", e, i)
        },
        _slide: function(e, t, i) {
            var s, a, n;
            this.options.values && this.options.values.length ? (s = this.values(t ? 0 : 1), 2 === this.options.values.length && this.options.range===!0 && (0 === t && i > s || 1 === t && s > i) && (i = s), i !== this.values(t) && (a = this.values(), a[t] = i, n = this._trigger("slide", e, {
                handle: this.handles[t],
                value: i,
                values: a
            }), s = this.values(t ? 0 : 1), n!==!1 && this.values(t, i))) : i !== this.value() && (n = this._trigger("slide", e, {
                handle: this.handles[t],
                value: i
            }), n!==!1 && this.value(i))
        },
        _stop: function(e, t) {
            var i = {
                handle: this.handles[t],
                value: this.value()
            };
            this.options.values && this.options.values.length && (i.value = this.values(t), i.values = this.values()), this._trigger("stop", e, i)
        },
        _change: function(e, t) {
            if (!this._keySliding&&!this._mouseSliding) {
                var i = {
                    handle: this.handles[t],
                    value: this.value()
                };
                this.options.values && this.options.values.length && (i.value = this.values(t), i.values = this.values()), this._lastChangedValue = t, this._trigger("change", e, i)
            }
        },
        value: function(e) {
            return arguments.length ? (this.options.value = this._trimAlignValue(e), this._refreshValue(), this._change(null, 0), void 0) : this._value()
        },
        values: function(t, i) {
            var s, a, n;
            if (arguments.length > 1)
                return this.options.values[t] = this._trimAlignValue(i), this._refreshValue(), this._change(null, t), void 0;
            if (!arguments.length)
                return this._values();
            if (!e.isArray(arguments[0]))
                return this.options.values && this.options.values.length ? this._values(t) : this.value();
            for (s = this.options.values, a = arguments[0], n = 0; s.length > n; n += 1)
                s[n] = this._trimAlignValue(a[n]), this._change(null, n);
            this._refreshValue()
        },
        _setOption: function(t, i) {
            var s, a = 0;
            switch ("range" === t && this.options.range===!0 && ("min" === i ? (this.options.value = this._values(0), this.options.values = null) : "max" === i && (this.options.value = this._values(this.options.values.length - 1), this.options.values = null)), e.isArray(this.options.values) && (a = this.options.values.length), "disabled" === t && this.element.toggleClass("ui-state-disabled", !!i), this._super(t, i), t) {
            case"orientation":
                this._detectOrientation(), this.element.removeClass("ui-slider-horizontal ui-slider-vertical").addClass("ui-slider-" + this.orientation), this._refreshValue(), this.handles.css("horizontal" === i ? "bottom" : "left", "");
                break;
            case"value":
                this._animateOff=!0, this._refreshValue(), this._change(null, 0), this._animateOff=!1;
                break;
            case"values":
                for (this._animateOff=!0, this._refreshValue(), s = 0; a > s; s += 1)
                    this._change(null, s);
                this._animateOff=!1;
                break;
            case"min":
            case"max":
                this._animateOff=!0, this._refreshValue(), this._animateOff=!1;
                break;
            case"range":
                this._animateOff=!0, this._refresh(), this._animateOff=!1
            }
        },
        _value: function() {
            var e = this.options.value;
            return e = this._trimAlignValue(e)
        },
        _values: function(e) {
            var t, i, s;
            if (arguments.length)
                return t = this.options.values[e], t = this._trimAlignValue(t);
            if (this.options.values && this.options.values.length) {
                for (i = this.options.values.slice(), s = 0; i.length > s; s += 1)
                    i[s] = this._trimAlignValue(i[s]);
                return i
            }
            return []
        },
        _trimAlignValue: function(e) {
            if (this._valueMin() >= e)
                return this._valueMin();
            if (e >= this._valueMax())
                return this._valueMax();
            var t = this.options.step > 0 ? this.options.step: 1, i = (e - this._valueMin())%t, s = e - i;
            return 2 * Math.abs(i) >= t && (s += i > 0 ? t : - t), parseFloat(s.toFixed(5))
        },
        _valueMin: function() {
            return this.options.min
        },
        _valueMax: function() {
            return this.options.max
        },
        _refreshValue: function() {
            var t, i, s, a, n, r = this.options.range, o = this.options, h = this, l = this._animateOff?!1 : o.animate, u = {};
            this.options.values && this.options.values.length ? this.handles.each(function(s) {
                i = 100 * ((h.values(s) - h._valueMin()) / (h._valueMax() - h._valueMin())), u["horizontal" === h.orientation ? "left": "bottom"] = i + "%", e(this).stop(1, 1)[l ? "animate": "css"](u, o.animate), h.options.range===!0 && ("horizontal" === h.orientation ? (0 === s && h.range.stop(1, 1)[l ? "animate": "css"]({
                    left: i + "%"
                }, o.animate), 1 === s && h.range[l ? "animate": "css"]({
                    width: i - t + "%"
                }, {
                    queue: !1,
                    duration: o.animate
                })) : (0 === s && h.range.stop(1, 1)[l ? "animate": "css"]({
                    bottom: i + "%"
                }, o.animate), 1 === s && h.range[l ? "animate": "css"]({
                    height: i - t + "%"
                }, {
                    queue: !1,
                    duration: o.animate
                }))), t = i
            }) : (s = this.value(), a = this._valueMin(), n = this._valueMax(), i = n !== a ? 100 * ((s - a) / (n - a)) : 0, u["horizontal" === this.orientation ? "left": "bottom"] = i + "%", this.handle.stop(1, 1)[l ? "animate": "css"](u, o.animate), "min" === r && "horizontal" === this.orientation && this.range.stop(1, 1)[l ? "animate": "css"]({
                width: i + "%"
            }, o.animate), "max" === r && "horizontal" === this.orientation && this.range[l ? "animate": "css"]({
                width: 100 - i + "%"
            }, {
                queue: !1,
                duration: o.animate
            }), "min" === r && "vertical" === this.orientation && this.range.stop(1, 1)[l ? "animate": "css"]({
                height: i + "%"
            }, o.animate), "max" === r && "vertical" === this.orientation && this.range[l ? "animate": "css"]({
                height: 100 - i + "%"
            }, {
                queue: !1,
                duration: o.animate
            }))
        },
        _handleEvents: {
            keydown: function(t) {
                var i, s, a, n, r = e(t.target).data("ui-slider-handle-index");
                switch (t.keyCode) {
                case e.ui.keyCode.HOME:
                case e.ui.keyCode.END:
                case e.ui.keyCode.PAGE_UP:
                case e.ui.keyCode.PAGE_DOWN:
                case e.ui.keyCode.UP:
                case e.ui.keyCode.RIGHT:
                case e.ui.keyCode.DOWN:
                case e.ui.keyCode.LEFT:
                    if (t.preventDefault(), !this._keySliding && (this._keySliding=!0, e(t.target).addClass("ui-state-active"), i = this._start(t, r), i===!1))
                        return 
                }
                switch (n = this.options.step, s = a = this.options.values && this.options.values.length ? this.values(r) : this.value(), t.keyCode) {
                case e.ui.keyCode.HOME:
                    a = this._valueMin();
                    break;
                case e.ui.keyCode.END:
                    a = this._valueMax();
                    break;
                case e.ui.keyCode.PAGE_UP:
                    a = this._trimAlignValue(s + (this._valueMax() - this._valueMin()) / this.numPages);
                    break;
                case e.ui.keyCode.PAGE_DOWN:
                    a = this._trimAlignValue(s - (this._valueMax() - this._valueMin()) / this.numPages);
                    break;
                case e.ui.keyCode.UP:
                case e.ui.keyCode.RIGHT:
                    if (s === this._valueMax())
                        return;
                    a = this._trimAlignValue(s + n);
                    break;
                case e.ui.keyCode.DOWN:
                case e.ui.keyCode.LEFT:
                    if (s === this._valueMin())
                        return;
                    a = this._trimAlignValue(s - n)
                }
                this._slide(t, r, a)
            },
            keyup: function(t) {
                var i = e(t.target).data("ui-slider-handle-index");
                this._keySliding && (this._keySliding=!1, this._stop(t, i), this._change(t, i), e(t.target).removeClass("ui-state-active"))
            }
        }
    }), e.widget("ui.spinner", {
        version: "1.11.1",
        defaultElement: "<input>",
        widgetEventPrefix: "spin",
        options: {
            culture: null,
            icons: {
                down: "ui-icon-triangle-1-s",
                up: "ui-icon-triangle-1-n"
            },
            incremental: !0,
            max: null,
            min: null,
            numberFormat: null,
            page: 10,
            step: 1,
            change: null,
            spin: null,
            start: null,
            stop: null
        },
        _create: function() {
            this._setOption("max", this.options.max), this._setOption("min", this.options.min), this._setOption("step", this.options.step), "" !== this.value() && this._value(this.element.val(), !0), this._draw(), this._on(this._events), this._refresh(), this._on(this.window, {
                beforeunload: function() {
                    this.element.removeAttr("autocomplete")
                }
            })
        },
        _getCreateOptions: function() {
            var t = {}, i = this.element;
            return e.each(["min", "max", "step"], function(e, s) {
                var a = i.attr(s);
                void 0 !== a && a.length && (t[s] = a)
            }), t
        },
        _events: {
            keydown: function(e) {
                this._start(e) && this._keydown(e) && e.preventDefault()
            },
            keyup: "_stop",
            focus: function() {
                this.previous = this.element.val()
            },
            blur: function(e) {
                return this.cancelBlur ? (delete this.cancelBlur, void 0) : (this._stop(), this._refresh(), this.previous !== this.element.val() && this._trigger("change", e), void 0)
            },
            mousewheel: function(e, t) {
                if (t) {
                    if (!this.spinning&&!this._start(e))
                        return !1;
                    this._spin((t > 0 ? 1 : - 1) * this.options.step, e), clearTimeout(this.mousewheelTimer), this.mousewheelTimer = this._delay(function() {
                        this.spinning && this._stop(e)
                    }, 100), e.preventDefault()
                }
            },
            "mousedown .ui-spinner-button": function(t) {
                function i() {
                    var e = this.element[0] === this.document[0].activeElement;
                    e || (this.element.focus(), this.previous = s, this._delay(function() {
                        this.previous = s
                    }))
                }
                var s;
                s = this.element[0] === this.document[0].activeElement ? this.previous : this.element.val(), t.preventDefault(), i.call(this), this.cancelBlur=!0, this._delay(function() {
                    delete this.cancelBlur, i.call(this)
                }), this._start(t)!==!1 && this._repeat(null, e(t.currentTarget).hasClass("ui-spinner-up") ? 1 : - 1, t)
            },
            "mouseup .ui-spinner-button": "_stop",
            "mouseenter .ui-spinner-button": function(t) {
                return e(t.currentTarget).hasClass("ui-state-active") ? this._start(t)===!1?!1 : (this._repeat(null, e(t.currentTarget).hasClass("ui-spinner-up") ? 1 : - 1, t), void 0) : void 0
            },
            "mouseleave .ui-spinner-button": "_stop"
        },
        _draw: function() {
            var e = this.uiSpinner = this.element.addClass("ui-spinner-input").attr("autocomplete", "off").wrap(this._uiSpinnerHtml()).parent().append(this._buttonHtml());
            this.element.attr("role", "spinbutton"), this.buttons = e.find(".ui-spinner-button").attr("tabIndex", - 1).button().removeClass("ui-corner-all"), this.buttons.height() > Math.ceil(.5 * e.height()) && e.height() > 0 && e.height(e.height()), this.options.disabled && this.disable()
        },
        _keydown: function(t) {
            var i = this.options, s = e.ui.keyCode;
            switch (t.keyCode) {
            case s.UP:
                return this._repeat(null, 1, t), !0;
            case s.DOWN:
                return this._repeat(null, - 1, t), !0;
            case s.PAGE_UP:
                return this._repeat(null, i.page, t), !0;
            case s.PAGE_DOWN:
                return this._repeat(null, - i.page, t), !0
            }
            return !1
        },
        _uiSpinnerHtml: function() {
            return "<span class='ui-spinner ui-widget ui-widget-content ui-corner-all'></span>"
        },
        _buttonHtml: function() {
            return "<a class='ui-spinner-button ui-spinner-up ui-corner-tr'><span class='ui-icon " + this.options.icons.up + "'>&#9650;</span>" + "</a>" + "<a class='ui-spinner-button ui-spinner-down ui-corner-br'>" + "<span class='ui-icon " + this.options.icons.down + "'>&#9660;</span>" + "</a>"
        },
        _start: function(e) {
            return this.spinning || this._trigger("start", e)!==!1 ? (this.counter || (this.counter = 1), this.spinning=!0, !0) : !1
        },
        _repeat: function(e, t, i) {
            e = e || 500, clearTimeout(this.timer), this.timer = this._delay(function() {
                this._repeat(40, t, i)
            }, e), this._spin(t * this.options.step, i)
        },
        _spin: function(e, t) {
            var i = this.value() || 0;
            this.counter || (this.counter = 1), i = this._adjustValue(i + e * this._increment(this.counter)), this.spinning && this._trigger("spin", t, {
                value: i
            })===!1 || (this._value(i), this.counter++)
        },
        _increment: function(t) {
            var i = this.options.incremental;
            return i ? e.isFunction(i) ? i(t) : Math.floor(t * t * t / 5e4 - t * t / 500 + 17 * t / 200 + 1) : 1
        },
        _precision: function() {
            var e = this._precisionOf(this.options.step);
            return null !== this.options.min && (e = Math.max(e, this._precisionOf(this.options.min))), e
        },
        _precisionOf: function(e) {
            var t = "" + e, i = t.indexOf(".");
            return - 1 === i ? 0 : t.length - i - 1
        },
        _adjustValue: function(e) {
            var t, i, s = this.options;
            return t = null !== s.min ? s.min : 0, i = e - t, i = Math.round(i / s.step) * s.step, e = t + i, e = parseFloat(e.toFixed(this._precision())), null !== s.max && e > s.max ? s.max : null !== s.min && s.min > e ? s.min : e
        },
        _stop: function(e) {
            this.spinning && (clearTimeout(this.timer), clearTimeout(this.mousewheelTimer), this.counter = 0, this.spinning=!1, this._trigger("stop", e))
        },
        _setOption: function(e, t) {
            if ("culture" === e || "numberFormat" === e) {
                var i = this._parse(this.element.val());
                return this.options[e] = t, this.element.val(this._format(i)), void 0
            }("max" === e || "min" === e || "step" === e) && "string" == typeof t && (t = this._parse(t)), "icons" === e && (this.buttons.first().find(".ui-icon").removeClass(this.options.icons.up).addClass(t.up), this.buttons.last().find(".ui-icon").removeClass(this.options.icons.down).addClass(t.down)), this._super(e, t), "disabled" === e && (this.widget().toggleClass("ui-state-disabled", !!t), this.element.prop("disabled", !!t), this.buttons.button(t ? "disable" : "enable"))
        },
        _setOptions: h(function(e) {
            this._super(e)
        }),
        _parse: function(e) {
            return "string" == typeof e && "" !== e && (e = window.Globalize && this.options.numberFormat ? Globalize.parseFloat(e, 10, this.options.culture) : + e), "" === e || isNaN(e) ? null : e
        },
        _format: function(e) {
            return "" === e ? "" : window.Globalize && this.options.numberFormat ? Globalize.format(e, this.options.numberFormat, this.options.culture) : e
        },
        _refresh: function() {
            this.element.attr({
                "aria-valuemin": this.options.min,
                "aria-valuemax": this.options.max,
                "aria-valuenow": this._parse(this.element.val())
            })
        },
        isValid: function() {
            var e = this.value();
            return null === e?!1 : e === this._adjustValue(e)
        },
        _value: function(e, t) {
            var i;
            "" !== e && (i = this._parse(e), null !== i && (t || (i = this._adjustValue(i)), e = this._format(i))), this.element.val(e), this._refresh()
        },
        _destroy: function() {
            this.element.removeClass("ui-spinner-input").prop("disabled", !1).removeAttr("autocomplete").removeAttr("role").removeAttr("aria-valuemin").removeAttr("aria-valuemax").removeAttr("aria-valuenow"), this.uiSpinner.replaceWith(this.element)
        },
        stepUp: h(function(e) {
            this._stepUp(e)
        }),
        _stepUp: function(e) {
            this._start() && (this._spin((e || 1) * this.options.step), this._stop())
        },
        stepDown: h(function(e) {
            this._stepDown(e)
        }),
        _stepDown: function(e) {
            this._start() && (this._spin((e || 1)*-this.options.step), this._stop())
        },
        pageUp: h(function(e) {
            this._stepUp((e || 1) * this.options.page)
        }),
        pageDown: h(function(e) {
            this._stepDown((e || 1) * this.options.page)
        }),
        value: function(e) {
            return arguments.length ? (h(this._value).call(this, e), void 0) : this._parse(this.element.val())
        },
        widget: function() {
            return this.uiSpinner
        }
    }), e.widget("ui.tabs", {
        version: "1.11.1",
        delay: 300,
        options: {
            active: null,
            collapsible: !1,
            event: "click",
            heightStyle: "content",
            hide: null,
            show: null,
            activate: null,
            beforeActivate: null,
            beforeLoad: null,
            load: null
        },
        _isLocal: function() {
            var e = /#.*$/;
            return function(t) {
                var i, s;
                t = t.cloneNode(!1), i = t.href.replace(e, ""), s = location.href.replace(e, "");
                try {
                    i = decodeURIComponent(i)
                } catch (a) {}
                try {
                    s = decodeURIComponent(s)
                } catch (a) {}
                return t.hash.length > 1 && i === s
            }
        }(),
        _create: function() {
            var t = this, i = this.options;
            this.running=!1, this.element.addClass("ui-tabs ui-widget ui-widget-content ui-corner-all").toggleClass("ui-tabs-collapsible", i.collapsible), this._processTabs(), i.active = this._initialActive(), e.isArray(i.disabled) && (i.disabled = e.unique(i.disabled.concat(e.map(this.tabs.filter(".ui-state-disabled"), function(e) {
                return t.tabs.index(e)
            }))).sort()), this.active = this.options.active!==!1 && this.anchors.length ? this._findActive(i.active) : e(), this._refresh(), this.active.length && this.load(i.active)
        },
        _initialActive: function() {
            var t = this.options.active, i = this.options.collapsible, s = location.hash.substring(1);
            return null === t && (s && this.tabs.each(function(i, a) {
                return e(a).attr("aria-controls") === s ? (t = i, !1) : void 0
            }), null === t && (t = this.tabs.index(this.tabs.filter(".ui-tabs-active"))), (null === t||-1 === t) && (t = this.tabs.length ? 0 : !1)), t!==!1 && (t = this.tabs.index(this.tabs.eq(t)), - 1 === t && (t = i?!1 : 0)), !i && t===!1 && this.anchors.length && (t = 0), t
        },
        _getCreateEventData: function() {
            return {
                tab: this.active,
                panel: this.active.length ? this._getPanelForTab(this.active): e()
            }
        },
        _tabKeydown: function(t) {
            var i = e(this.document[0].activeElement).closest("li"), s = this.tabs.index(i), a=!0;
            if (!this._handlePageNav(t)) {
                switch (t.keyCode) {
                case e.ui.keyCode.RIGHT:
                case e.ui.keyCode.DOWN:
                    s++;
                    break;
                case e.ui.keyCode.UP:
                case e.ui.keyCode.LEFT:
                    a=!1, s--;
                    break;
                case e.ui.keyCode.END:
                    s = this.anchors.length - 1;
                    break;
                case e.ui.keyCode.HOME:
                    s = 0;
                    break;
                case e.ui.keyCode.SPACE:
                    return t.preventDefault(), clearTimeout(this.activating), this._activate(s), void 0;
                case e.ui.keyCode.ENTER:
                    return t.preventDefault(), clearTimeout(this.activating), this._activate(s === this.options.active?!1 : s), void 0;
                default:
                    return 
                }
                t.preventDefault(), clearTimeout(this.activating), s = this._focusNextTab(s, a), t.ctrlKey || (i.attr("aria-selected", "false"), this.tabs.eq(s).attr("aria-selected", "true"), this.activating = this._delay(function() {
                    this.option("active", s)
                }, this.delay))
            }
        },
        _panelKeydown: function(t) {
            this._handlePageNav(t) || t.ctrlKey && t.keyCode === e.ui.keyCode.UP && (t.preventDefault(), this.active.focus())
        },
        _handlePageNav: function(t) {
            return t.altKey && t.keyCode === e.ui.keyCode.PAGE_UP ? (this._activate(this._focusNextTab(this.options.active - 1, !1)), !0) : t.altKey && t.keyCode === e.ui.keyCode.PAGE_DOWN ? (this._activate(this._focusNextTab(this.options.active + 1, !0)), !0) : void 0
        },
        _findNextTab: function(t, i) {
            function s() {
                return t > a && (t = 0), 0 > t && (t = a), t
            }
            for (var a = this.tabs.length - 1; - 1 !== e.inArray(s(), this.options.disabled);)
                t = i ? t + 1 : t - 1;
            return t
        },
        _focusNextTab: function(e, t) {
            return e = this._findNextTab(e, t), this.tabs.eq(e).focus(), e
        },
        _setOption: function(e, t) {
            return "active" === e ? (this._activate(t), void 0) : "disabled" === e ? (this._setupDisabled(t), void 0) : (this._super(e, t), "collapsible" === e && (this.element.toggleClass("ui-tabs-collapsible", t), t || this.options.active!==!1 || this._activate(0)), "event" === e && this._setupEvents(t), "heightStyle" === e && this._setupHeightStyle(t), void 0)
        },
        _sanitizeSelector: function(e) {
            return e ? e.replace(/[!"$%&'()*+,.\/:;<=>?@\[\]\^`{|}~]/g, "\\$&") : ""
        },
        refresh: function() {
            var t = this.options, i = this.tablist.children(":has(a[href])");
            t.disabled = e.map(i.filter(".ui-state-disabled"), function(e) {
                return i.index(e)
            }), this._processTabs(), t.active!==!1 && this.anchors.length ? this.active.length&&!e.contains(this.tablist[0], this.active[0]) ? this.tabs.length === t.disabled.length ? (t.active=!1, this.active = e()) : this._activate(this._findNextTab(Math.max(0, t.active - 1), !1)) : t.active = this.tabs.index(this.active) : (t.active=!1, this.active = e()), this._refresh()
        },
        _refresh: function() {
            this._setupDisabled(this.options.disabled), this._setupEvents(this.options.event), this._setupHeightStyle(this.options.heightStyle), this.tabs.not(this.active).attr({
                "aria-selected": "false",
                "aria-expanded": "false",
                tabIndex: - 1
            }), this.panels.not(this._getPanelForTab(this.active)).hide().attr({
                "aria-hidden": "true"
            }), this.active.length ? (this.active.addClass("ui-tabs-active ui-state-active").attr({
                "aria-selected": "true",
                "aria-expanded": "true",
                tabIndex: 0
            }), this._getPanelForTab(this.active).show().attr({
                "aria-hidden": "false"
            })) : this.tabs.eq(0).attr("tabIndex", 0)
        },
        _processTabs: function() {
            var t = this;
            this.tablist = this._getList().addClass("ui-tabs-nav ui-helper-reset ui-helper-clearfix ui-widget-header ui-corner-all").attr("role", "tablist").delegate("> li", "mousedown" + this.eventNamespace, function(t) {
                e(this).is(".ui-state-disabled") && t.preventDefault()
            }).delegate(".ui-tabs-anchor", "focus" + this.eventNamespace, function() {
                e(this).closest("li").is(".ui-state-disabled") && this.blur()
            }), this.tabs = this.tablist.find("> li:has(a[href])").addClass("ui-state-default ui-corner-top").attr({
                role: "tab",
                tabIndex: - 1
            }), this.anchors = this.tabs.map(function() {
                return e("a", this)[0]
            }).addClass("ui-tabs-anchor").attr({
                role: "presentation",
                tabIndex: - 1
            }), this.panels = e(), this.anchors.each(function(i, s) {
                var a, n, r, o = e(s).uniqueId().attr("id"), h = e(s).closest("li"), l = h.attr("aria-controls");
                t._isLocal(s) ? (a = s.hash, r = a.substring(1), n = t.element.find(t._sanitizeSelector(a))) : (r = h.attr("aria-controls") || e({}).uniqueId()[0].id, a = "#" + r, n = t.element.find(a), n.length || (n = t._createPanel(r), n.insertAfter(t.panels[i - 1] || t.tablist)), n.attr("aria-live", "polite")), n.length && (t.panels = t.panels.add(n)), l && h.data("ui-tabs-aria-controls", l), h.attr({
                    "aria-controls": r,
                    "aria-labelledby": o
                }), n.attr("aria-labelledby", o)
            }), this.panels.addClass("ui-tabs-panel ui-widget-content ui-corner-bottom").attr("role", "tabpanel")
        },
        _getList: function() {
            return this.tablist || this.element.find("ol,ul").eq(0)
        },
        _createPanel: function(t) {
            return e("<div>").attr("id", t).addClass("ui-tabs-panel ui-widget-content ui-corner-bottom").data("ui-tabs-destroy", !0)
        },
        _setupDisabled: function(t) {
            e.isArray(t) && (t.length ? t.length === this.anchors.length && (t=!0) : t=!1);
            for (var i, s = 0; i = this.tabs[s]; s++)
                t===!0||-1 !== e.inArray(s, t) ? e(i).addClass("ui-state-disabled").attr("aria-disabled", "true") : e(i).removeClass("ui-state-disabled").removeAttr("aria-disabled");
            this.options.disabled = t
        },
        _setupEvents: function(t) {
            var i = {};
            t && e.each(t.split(" "), function(e, t) {
                i[t] = "_eventHandler"
            }), this._off(this.anchors.add(this.tabs).add(this.panels)), this._on(!0, this.anchors, {
                click: function(e) {
                    e.preventDefault()
                }
            }), this._on(this.anchors, i), this._on(this.tabs, {
                keydown: "_tabKeydown"
            }), this._on(this.panels, {
                keydown: "_panelKeydown"
            }), this._focusable(this.tabs), this._hoverable(this.tabs)
        },
        _setupHeightStyle: function(t) {
            var i, s = this.element.parent();
            "fill" === t ? (i = s.height(), i -= this.element.outerHeight() - this.element.height(), this.element.siblings(":visible").each(function() {
                var t = e(this), s = t.css("position");
                "absolute" !== s && "fixed" !== s && (i -= t.outerHeight(!0))
            }), this.element.children().not(this.panels).each(function() {
                i -= e(this).outerHeight(!0)
            }), this.panels.each(function() {
                e(this).height(Math.max(0, i - e(this).innerHeight() + e(this).height()))
            }).css("overflow", "auto")) : "auto" === t && (i = 0, this.panels.each(function() {
                i = Math.max(i, e(this).height("").height())
            }).height(i))
        },
        _eventHandler: function(t) {
            var i = this.options, s = this.active, a = e(t.currentTarget), n = a.closest("li"), r = n[0] === s[0], o = r && i.collapsible, h = o ? e(): this._getPanelForTab(n), l = s.length ? this._getPanelForTab(s): e(), u = {
                oldTab: s,
                oldPanel: l,
                newTab: o ? e(): n,
                newPanel: h
            };
            t.preventDefault(), n.hasClass("ui-state-disabled") || n.hasClass("ui-tabs-loading") || this.running || r&&!i.collapsible || this._trigger("beforeActivate", t, u)===!1 || (i.active = o?!1 : this.tabs.index(n), this.active = r ? e() : n, this.xhr && this.xhr.abort(), l.length || h.length || e.error("jQuery UI Tabs: Mismatching fragment identifier."), h.length && this.load(this.tabs.index(n), t), this._toggle(t, u))
        },
        _toggle: function(t, i) {
            function s() {
                n.running=!1, n._trigger("activate", t, i)
            }
            function a() {
                i.newTab.closest("li").addClass("ui-tabs-active ui-state-active"), r.length && n.options.show ? n._show(r, n.options.show, s) : (r.show(), s())
            }
            var n = this, r = i.newPanel, o = i.oldPanel;
            this.running=!0, o.length && this.options.hide ? this._hide(o, this.options.hide, function() {
                i.oldTab.closest("li").removeClass("ui-tabs-active ui-state-active"), a()
            }) : (i.oldTab.closest("li").removeClass("ui-tabs-active ui-state-active"), o.hide(), a()), o.attr("aria-hidden", "true"), i.oldTab.attr({
                "aria-selected": "false",
                "aria-expanded": "false"
            }), r.length && o.length ? i.oldTab.attr("tabIndex", - 1) : r.length && this.tabs.filter(function() {
                return 0 === e(this).attr("tabIndex")
            }).attr("tabIndex", - 1), r.attr("aria-hidden", "false"), i.newTab.attr({
                "aria-selected": "true",
                "aria-expanded": "true",
                tabIndex: 0
            })
        },
        _activate: function(t) {
            var i, s = this._findActive(t);
            s[0] !== this.active[0] && (s.length || (s = this.active), i = s.find(".ui-tabs-anchor")[0], this._eventHandler({
                target: i,
                currentTarget: i,
                preventDefault: e.noop
            }))
        },
        _findActive: function(t) {
            return t===!1 ? e() : this.tabs.eq(t)
        },
        _getIndex: function(e) {
            return "string" == typeof e && (e = this.anchors.index(this.anchors.filter("[href$='" + e + "']"))), e
        },
        _destroy: function() {
            this.xhr && this.xhr.abort(), this.element.removeClass("ui-tabs ui-widget ui-widget-content ui-corner-all ui-tabs-collapsible"), this.tablist.removeClass("ui-tabs-nav ui-helper-reset ui-helper-clearfix ui-widget-header ui-corner-all").removeAttr("role"), this.anchors.removeClass("ui-tabs-anchor").removeAttr("role").removeAttr("tabIndex").removeUniqueId(), this.tablist.unbind(this.eventNamespace), this.tabs.add(this.panels).each(function() {
                e.data(this, "ui-tabs-destroy") ? e(this).remove() : e(this).removeClass("ui-state-default ui-state-active ui-state-disabled ui-corner-top ui-corner-bottom ui-widget-content ui-tabs-active ui-tabs-panel").removeAttr("tabIndex").removeAttr("aria-live").removeAttr("aria-busy").removeAttr("aria-selected").removeAttr("aria-labelledby").removeAttr("aria-hidden").removeAttr("aria-expanded").removeAttr("role")
            }), this.tabs.each(function() {
                var t = e(this), i = t.data("ui-tabs-aria-controls");
                i ? t.attr("aria-controls", i).removeData("ui-tabs-aria-controls") : t.removeAttr("aria-controls")
            }), this.panels.show(), "content" !== this.options.heightStyle && this.panels.css("height", "")
        },
        enable: function(t) {
            var i = this.options.disabled;
            i!==!1 && (void 0 === t ? i=!1 : (t = this._getIndex(t), i = e.isArray(i) ? e.map(i, function(e) {
                return e !== t ? e : null
            }) : e.map(this.tabs, function(e, i) {
                return i !== t ? i : null
            })), this._setupDisabled(i))
        },
        disable: function(t) {
            var i = this.options.disabled;
            if (i!==!0) {
                if (void 0 === t)
                    i=!0;
                else {
                    if (t = this._getIndex(t), - 1 !== e.inArray(t, i))
                        return;
                    i = e.isArray(i) ? e.merge([t], i).sort() : [t]
                }
                this._setupDisabled(i)
            }
        },
        load: function(t, i) {
            t = this._getIndex(t);
            var s = this, a = this.tabs.eq(t), n = a.find(".ui-tabs-anchor"), r = this._getPanelForTab(a), o = {
                tab: a,
                panel: r
            };
            this._isLocal(n[0]) || (this.xhr = e.ajax(this._ajaxSettings(n, i, o)), this.xhr && "canceled" !== this.xhr.statusText && (a.addClass("ui-tabs-loading"), r.attr("aria-busy", "true"), this.xhr.success(function(e) {
                setTimeout(function() {
                    r.html(e), s._trigger("load", i, o)
                }, 1)
            }).complete(function(e, t) {
                setTimeout(function() {
                    "abort" === t && s.panels.stop(!1, !0), a.removeClass("ui-tabs-loading"), r.removeAttr("aria-busy"), e === s.xhr && delete s.xhr
                }, 1)
            })))
        },
        _ajaxSettings: function(t, i, s) {
            var a = this;
            return {
                url: t.attr("href"),
                beforeSend: function(t, n) {
                    return a._trigger("beforeLoad", i, e.extend({
                        jqXHR: t,
                        ajaxSettings: n
                    }, s))
                }
            }
        },
        _getPanelForTab: function(t) {
            var i = e(t).attr("aria-controls");
            return this.element.find(this._sanitizeSelector("#" + i))
        }
    }), e.widget("ui.tooltip", {
        version: "1.11.1",
        options: {
            content: function() {
                var t = e(this).attr("title") || "";
                return e("<a>").text(t).html()
            },
            hide: !0,
            items: "[title]:not([disabled])",
            position: {
                my: "left top+15",
                at: "left bottom",
                collision: "flipfit flip"
            },
            show: !0,
            tooltipClass: null,
            track: !1,
            close: null,
            open: null
        },
        _addDescribedBy: function(t, i) {
            var s = (t.attr("aria-describedby") || "").split(/\s+/);
            s.push(i), t.data("ui-tooltip-id", i).attr("aria-describedby", e.trim(s.join(" ")))
        },
        _removeDescribedBy: function(t) {
            var i = t.data("ui-tooltip-id"), s = (t.attr("aria-describedby") || "").split(/\s+/), a = e.inArray(i, s);
            - 1 !== a && s.splice(a, 1), t.removeData("ui-tooltip-id"), s = e.trim(s.join(" ")), s ? t.attr("aria-describedby", s) : t.removeAttr("aria-describedby")
        },
        _create: function() {
            this._on({
                mouseover: "open",
                focusin: "open"
            }), this.tooltips = {}, this.parents = {}, this.options.disabled && this._disable(), this.liveRegion = e("<div>").attr({
                role: "log",
                "aria-live": "assertive",
                "aria-relevant": "additions"
            }).addClass("ui-helper-hidden-accessible").appendTo(this.document[0].body)
        },
        _setOption: function(t, i) {
            var s = this;
            return "disabled" === t ? (this[i ? "_disable": "_enable"](), this.options[t] = i, void 0) : (this._super(t, i), "content" === t && e.each(this.tooltips, function(e, t) {
                s._updateContent(t)
            }), void 0)
        },
        _disable: function() {
            var t = this;
            e.each(this.tooltips, function(i, s) {
                var a = e.Event("blur");
                a.target = a.currentTarget = s[0], t.close(a, !0)
            }), this.element.find(this.options.items).addBack().each(function() {
                var t = e(this);
                t.is("[title]") && t.data("ui-tooltip-title", t.attr("title")).removeAttr("title")
            })
        },
        _enable: function() {
            this.element.find(this.options.items).addBack().each(function() {
                var t = e(this);
                t.data("ui-tooltip-title") && t.attr("title", t.data("ui-tooltip-title"))
            })
        },
        open: function(t) {
            var i = this, s = e(t ? t.target : this.element).closest(this.options.items);
            s.length&&!s.data("ui-tooltip-id") && (s.attr("title") && s.data("ui-tooltip-title", s.attr("title")), s.data("ui-tooltip-open", !0), t && "mouseover" === t.type && s.parents().each(function() {
                var t, s = e(this);
                s.data("ui-tooltip-open") && (t = e.Event("blur"), t.target = t.currentTarget = this, i.close(t, !0)), s.attr("title") && (s.uniqueId(), i.parents[this.id] = {
                    element: this,
                    title: s.attr("title")
                }, s.attr("title", ""))
            }), this._updateContent(s, t))
        },
        _updateContent: function(e, t) {
            var i, s = this.options.content, a = this, n = t ? t.type: null;
            return "string" == typeof s ? this._open(t, e, s) : (i = s.call(e[0], function(i) {
                e.data("ui-tooltip-open") && a._delay(function() {
                    t && (t.type = n), this._open(t, e, i)
                })
            }), i && this._open(t, e, i), void 0)
        },
        _open: function(t, i, s) {
            function a(e) {
                l.of = e, n.is(":hidden") || n.position(l)
            }
            var n, r, o, h, l = e.extend({}, this.options.position);
            if (s) {
                if (n = this._find(i), n.length)
                    return n.find(".ui-tooltip-content").html(s), void 0;
                i.is("[title]") && (t && "mouseover" === t.type ? i.attr("title", "") : i.removeAttr("title")), n = this._tooltip(i), this._addDescribedBy(i, n.attr("id")), n.find(".ui-tooltip-content").html(s), this.liveRegion.children().hide(), s.clone ? (h = s.clone(), h.removeAttr("id").find("[id]").removeAttr("id")) : h = s, e("<div>").html(h).appendTo(this.liveRegion), this.options.track && t && /^mouse/.test(t.type) ? (this._on(this.document, {
                    mousemove: a
                }), a(t)) : n.position(e.extend({
                    of: i
                }, this.options.position)), this.hiding=!1, this.closing=!1, n.hide(), this._show(n, this.options.show), this.options.show && this.options.show.delay && (o = this.delayedShow = setInterval(function() {
                    n.is(":visible") && (a(l.of), clearInterval(o))
                }, e.fx.interval)), this._trigger("open", t, {
                    tooltip: n
                }), r = {
                    keyup: function(t) {
                        if (t.keyCode === e.ui.keyCode.ESCAPE) {
                            var s = e.Event(t);
                            s.currentTarget = i[0], this.close(s, !0)
                        }
                    }
                }, i[0] !== this.element[0] && (r.remove = function() {
                    this._removeTooltip(n)
                }), t && "mouseover" !== t.type || (r.mouseleave = "close"), t && "focusin" !== t.type || (r.focusout = "close"), this._on(!0, i, r)
            }
        },
        close: function(t) {
            var i = this, s = e(t ? t.currentTarget : this.element), a = this._find(s);
            this.closing || (clearInterval(this.delayedShow), s.data("ui-tooltip-title")&&!s.attr("title") && s.attr("title", s.data("ui-tooltip-title")), this._removeDescribedBy(s), this.hiding=!0, a.stop(!0), this._hide(a, this.options.hide, function() {
                i._removeTooltip(e(this)), this.hiding=!1, this.closing=!1
            }), s.removeData("ui-tooltip-open"), this._off(s, "mouseleave focusout keyup"), s[0] !== this.element[0] && this._off(s, "remove"), this._off(this.document, "mousemove"), t && "mouseleave" === t.type && e.each(this.parents, function(t, s) {
                e(s.element).attr("title", s.title), delete i.parents[t]
            }), this.closing=!0, this._trigger("close", t, {
                tooltip: a
            }), this.hiding || (this.closing=!1))
        },
        _tooltip: function(t) {
            var i = e("<div>").attr("role", "tooltip").addClass("ui-tooltip ui-widget ui-corner-all ui-widget-content " + (this.options.tooltipClass || "")), s = i.uniqueId().attr("id");
            return e("<div>").addClass("ui-tooltip-content").appendTo(i), i.appendTo(this.document[0].body), this.tooltips[s] = t, i
        },
        _find: function(t) {
            var i = t.data("ui-tooltip-id");
            return i ? e("#" + i) : e()
        },
        _removeTooltip: function(e) {
            e.remove(), delete this.tooltips[e.attr("id")]
        },
        _destroy: function() {
            var t = this;
            e.each(this.tooltips, function(i, s) {
                var a = e.Event("blur");
                a.target = a.currentTarget = s[0], t.close(a, !0), e("#" + i).remove(), s.data("ui-tooltip-title") && (s.attr("title") || s.attr("title", s.data("ui-tooltip-title")), s.removeData("ui-tooltip-title"))
            }), this.liveRegion.remove()
        }
    });
    var y = "ui-effects-", b = e;
    e.effects = {
        effect: {}
    }, function(e, t) {
        function i(e, t, i) {
            var s = d[t.type] || {};
            return null == e ? i ||!t.def ? null : t.def : (e = s.floor?~~e : parseFloat(e), isNaN(e) ? t.def : s.mod ? (e + s.mod)%s.mod : 0 > e ? 0 : e > s.max ? s.max : e)
        }
        function s(i) {
            var s = l(), a = s._rgba = [];
            return i = i.toLowerCase(), f(h, function(e, n) {
                var r, o = n.re.exec(i), h = o && n.parse(o), l = n.space || "rgba";
                return h ? (r = s[l](h), s[u[l].cache] = r[u[l].cache], a = s._rgba = r._rgba, !1) : t
            }), a.length ? ("0,0,0,0" === a.join() && e.extend(a, n.transparent), s) : n[i]
        }
        function a(e, t, i) {
            return i = (i + 1)%1, 1 > 6 * i ? e + 6 * (t - e) * i : 1 > 2 * i ? t : 2 > 3 * i ? e + 6 * (t - e) * (2 / 3 - i) : e
        }
        var n, r = "backgroundColor borderBottomColor borderLeftColor borderRightColor borderTopColor color columnRuleColor outlineColor textDecorationColor textEmphasisColor", o = /^([\-+])=\s*(\d+\.?\d*)/, h = [{
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
        ], l = e.Color = function(t, i, s, a) {
            return new e.Color.fn.parse(t, i, s, a)
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
        }, c = l.support = {}, p = e("<p>")[0], f = e.each;
        p.style.cssText = "background-color:rgba(1,1,1,.5)", c.rgba = p.style.backgroundColor.indexOf("rgba")>-1, f(u, function(e, t) {
            t.cache = "_" + e, t.props.alpha = {
                idx: 3,
                type: "percent",
                def: 1
            }
        }), l.fn = e.extend(l.prototype, {
            parse: function(a, r, o, h) {
                if (a === t)
                    return this._rgba = [null, null, null, null], this;
                (a.jquery || a.nodeType) && (a = e(a).css(r), r = t);
                var d = this, c = e.type(a), p = this._rgba = [];
                return r !== t && (a = [a, r, o, h], c = "array"), "string" === c ? this.parse(s(a) || n._default) : "array" === c ? (f(u.rgba.props, function(e, t) {
                    p[t.idx] = i(a[t.idx], t)
                }), this) : "object" === c ? (a instanceof l ? f(u, function(e, t) {
                    a[t.cache] && (d[t.cache] = a[t.cache].slice())
                }) : f(u, function(t, s) {
                    var n = s.cache;
                    f(s.props, function(e, t) {
                        if (!d[n] && s.to) {
                            if ("alpha" === e || null == a[e])
                                return;
                            d[n] = s.to(d._rgba)
                        }
                        d[n][t.idx] = i(a[e], t, !0)
                    }), d[n] && 0 > e.inArray(null, d[n].slice(0, 3)) && (d[n][3] = 1, s.from && (d._rgba = s.from(d[n])))
                }), this) : t
            },
            is: function(e) {
                var i = l(e), s=!0, a = this;
                return f(u, function(e, n) {
                    var r, o = i[n.cache];
                    return o && (r = a[n.cache] || n.to && n.to(a._rgba) || [], f(n.props, function(e, i) {
                        return null != o[i.idx] ? s = o[i.idx] === r[i.idx] : t
                    })), s
                }), s
            },
            _space: function() {
                var e = [], t = this;
                return f(u, function(i, s) {
                    t[s.cache] && e.push(i)
                }), e.pop()
            },
            transition: function(e, t) {
                var s = l(e), a = s._space(), n = u[a], r = 0 === this.alpha() ? l("transparent"): this, o = r[n.cache] || n.to(r._rgba), h = o.slice();
                return s = s[n.cache], f(n.props, function(e, a) {
                    var n = a.idx, r = o[n], l = s[n], u = d[a.type] || {};
                    null !== l && (null === r ? h[n] = l : (u.mod && (l - r > u.mod / 2 ? r += u.mod : r - l > u.mod / 2 && (r -= u.mod)), h[n] = i((l - r) * t + r, a)))
                }), this[a](h)
            },
            blend: function(t) {
                if (1 === this._rgba[3])
                    return this;
                var i = this._rgba.slice(), s = i.pop(), a = l(t)._rgba;
                return l(e.map(i, function(e, t) {
                    return (1 - s) * a[t] + s * e
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
                var i = this._rgba.slice(), s = i.pop();
                return t && i.push(~~(255 * s)), "#" + e.map(i, function(e) {
                    return e = (e || 0).toString(16), 1 === e.length ? "0" + e : e
                }).join("")
            },
            toString: function() {
                return 0 === this._rgba[3] ? "transparent" : this.toRgbaString()
            }
        }), l.fn.parse.prototype = l.fn, u.hsla.to = function(e) {
            if (null == e[0] || null == e[1] || null == e[2])
                return [null, null, null, e[3]];
            var t, i, s = e[0] / 255, a = e[1] / 255, n = e[2] / 255, r = e[3], o = Math.max(s, a, n), h = Math.min(s, a, n), l = o - h, u = o + h, d = .5 * u;
            return t = h === o ? 0 : s === o ? 60 * (a - n) / l + 360 : a === o ? 60 * (n - s) / l + 120 : 60 * (s - a) / l + 240, i = 0 === l ? 0 : .5 >= d ? l / u : l / (2 - u), [Math.round(t)%360, i, d, null == r ? 1: r]
        }, u.hsla.from = function(e) {
            if (null == e[0] || null == e[1] || null == e[2])
                return [null, null, null, e[3]];
            var t = e[0] / 360, i = e[1], s = e[2], n = e[3], r = .5 >= s ? s * (1 + i): s + i - s * i, o = 2 * s - r;
            return [Math.round(255 * a(o, r, t + 1 / 3)), Math.round(255 * a(o, r, t)), Math.round(255 * a(o, r, t - 1 / 3)), n]
        }, f(u, function(s, a) {
            var n = a.props, r = a.cache, h = a.to, u = a.from;
            l.fn[s] = function(s) {
                if (h&&!this[r] && (this[r] = h(this._rgba)), s === t)
                    return this[r].slice();
                var a, o = e.type(s), d = "array" === o || "object" === o ? s: arguments, c = this[r].slice();
                return f(n, function(e, t) {
                    var s = d["object" === o ? e: t.idx];
                    null == s && (s = c[t.idx]), c[t.idx] = i(s, t)
                }), u ? (a = l(u(c)), a[r] = c, a) : l(c)
            }, f(n, function(t, i) {
                l.fn[t] || (l.fn[t] = function(a) {
                    var n, r = e.type(a), h = "alpha" === t ? this._hsla ? "hsla": "rgba": s, l = this[h](), u = l[i.idx];
                    return "undefined" === r ? u : ("function" === r && (a = a.call(this, u), r = e.type(a)), null == a && i.empty ? this : ("string" === r && (n = o.exec(a), n && (a = u + parseFloat(n[2]) * ("+" === n[1] ? 1 : - 1))), l[i.idx] = a, this[h](l)))
                })
            })
        }), l.hook = function(t) {
            var i = t.split(" ");
            f(i, function(t, i) {
                e.cssHooks[i] = {
                    set: function(t, a) {
                        var n, r, o = "";
                        if ("transparent" !== a && ("string" !== e.type(a) || (n = s(a)))) {
                            if (a = l(n || a), !c.rgba && 1 !== a._rgba[3]) {
                                for (r = "backgroundColor" === i ? t.parentNode : t; ("" === o || "transparent" === o) && r && r.style;)
                                    try {
                                        o = e.css(r, "backgroundColor"), r = r.parentNode
                                } catch (h) {}
                                a = a.blend(o && "transparent" !== o ? o : "_default")
                            }
                            a = a.toRgbaString()
                        }
                        try {
                            t.style[i] = a
                        } catch (h) {}
                    }
                }, e.fx.step[i] = function(t) {
                    t.colorInit || (t.start = l(t.elem, i), t.end = l(t.end), t.colorInit=!0), e.cssHooks[i].set(t.elem, t.start.transition(t.end, t.pos))
                }
            })
        }, l.hook(r), e.cssHooks.borderColor = {
            expand: function(e) {
                var t = {};
                return f(["Top", "Right", "Bottom", "Left"], function(i, s) {
                    t["border" + s + "Color"] = e
                }), t
            }
        }, n = e.Color.names = {
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
    }(b), function() {
        function t(t) {
            var i, s, a = t.ownerDocument.defaultView ? t.ownerDocument.defaultView.getComputedStyle(t, null): t.currentStyle, n = {};
            if (a && a.length && a[0] && a[a[0]])
                for (s = a.length; s--;)
                    i = a[s], "string" == typeof a[i] && (n[e.camelCase(i)] = a[i]);
            else 
                for (i in a)
                    "string" == typeof a[i] && (n[i] = a[i]);
            return n
        }
        function i(t, i) {
            var s, n, r = {};
            for (s in i)
                n = i[s], t[s] !== n && (a[s] || (e.fx.step[s] ||!isNaN(parseFloat(n))) && (r[s] = n));
            return r
        }
        var s = ["add", "remove", "toggle"], a = {
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
        e.each(["borderLeftStyle", "borderRightStyle", "borderBottomStyle", "borderTopStyle"], function(t, i) {
            e.fx.step[i] = function(e) {
                ("none" !== e.end&&!e.setAttr || 1 === e.pos&&!e.setAttr) && (b.style(e.elem, i, e.end), e.setAttr=!0)
            }
        }), e.fn.addBack || (e.fn.addBack = function(e) {
            return this.add(null == e ? this.prevObject : this.prevObject.filter(e))
        }), e.effects.animateClass = function(a, n, r, o) {
            var h = e.speed(n, r, o);
            return this.queue(function() {
                var n, r = e(this), o = r.attr("class") || "", l = h.children ? r.find("*").addBack(): r;
                l = l.map(function() {
                    var i = e(this);
                    return {
                        el: i,
                        start: t(this)
                    }
                }), n = function() {
                    e.each(s, function(e, t) {
                        a[t] && r[t + "Class"](a[t])
                    })
                }, n(), l = l.map(function() {
                    return this.end = t(this.el[0]), this.diff = i(this.start, this.end), this
                }), r.attr("class", o), l = l.map(function() {
                    var t = this, i = e.Deferred(), s = e.extend({}, h, {
                        queue: !1,
                        complete: function() {
                            i.resolve(t)
                        }
                    });
                    return this.el.animate(this.diff, s), i.promise()
                }), e.when.apply(e, l.get()).done(function() {
                    n(), e.each(arguments, function() {
                        var t = this.el;
                        e.each(this.diff, function(e) {
                            t.css(e, "")
                        })
                    }), h.complete.call(r[0])
                })
            })
        }, e.fn.extend({
            addClass: function(t) {
                return function(i, s, a, n) {
                    return s ? e.effects.animateClass.call(this, {
                        add: i
                    }, s, a, n) : t.apply(this, arguments)
                }
            }(e.fn.addClass),
            removeClass: function(t) {
                return function(i, s, a, n) {
                    return arguments.length > 1 ? e.effects.animateClass.call(this, {
                        remove: i
                    }, s, a, n) : t.apply(this, arguments)
                }
            }(e.fn.removeClass),
            toggleClass: function(t) {
                return function(i, s, a, n, r) {
                    return "boolean" == typeof s || void 0 === s ? a ? e.effects.animateClass.call(this, s ? {
                        add: i
                    } : {
                        remove: i
                    }, a, n, r) : t.apply(this, arguments) : e.effects.animateClass.call(this, {
                        toggle: i
                    }, s, a, n)
                }
            }(e.fn.toggleClass),
            switchClass: function(t, i, s, a, n) {
                return e.effects.animateClass.call(this, {
                    add: i,
                    remove: t
                }, s, a, n)
            }
        })
    }(), function() {
        function t(t, i, s, a) {
            return e.isPlainObject(t) && (i = t, t = t.effect), t = {
                effect: t
            }, null == i && (i = {}), e.isFunction(i) && (a = i, s = null, i = {}), ("number" == typeof i || e.fx.speeds[i]) && (a = s, s = i, i = {}), e.isFunction(s) && (a = s, s = null), i && e.extend(t, i), s = s || i.duration, t.duration = e.fx.off ? 0 : "number" == typeof s ? s : s in e.fx.speeds ? e.fx.speeds[s] : e.fx.speeds._default, t.complete = a || i.complete, t
        }
        function i(t) {
            return !t || "number" == typeof t || e.fx.speeds[t]?!0 : "string" != typeof t || e.effects.effect[t] ? e.isFunction(t)?!0 : "object" != typeof t || t.effect?!1 : !0 : !0
        }
        e.extend(e.effects, {
            version: "1.11.1",
            save: function(e, t) {
                for (var i = 0; t.length > i; i++)
                    null !== t[i] && e.data(y + t[i], e[0].style[t[i]])
            },
            restore: function(e, t) {
                var i, s;
                for (s = 0; t.length > s; s++)
                    null !== t[s] && (i = e.data(y + t[s]), void 0 === i && (i = ""), e.css(t[s], i))
            },
            setMode: function(e, t) {
                return "toggle" === t && (t = e.is(":hidden") ? "show" : "hide"), t
            },
            getBaseline: function(e, t) {
                var i, s;
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
                    s = 0;
                    break;
                case"center":
                    s = .5;
                    break;
                case"right":
                    s = 1;
                    break;
                default:
                    s = e[1] / t.width
                }
                return {
                    x: s,
                    y: i
                }
            },
            createWrapper: function(t) {
                if (t.parent().is(".ui-effects-wrapper"))
                    return t.parent();
                var i = {
                    width: t.outerWidth(!0),
                    height: t.outerHeight(!0),
                    "float": t.css("float")
                }, s = e("<div></div>").addClass("ui-effects-wrapper").css({
                    fontSize: "100%",
                    background: "transparent",
                    border: "none",
                    margin: 0,
                    padding: 0
                }), a = {
                    width: t.width(),
                    height: t.height()
                }, n = document.activeElement;
                try {
                    n.id
                } catch (r) {
                    n = document.body
                }
                return t.wrap(s), (t[0] === n || e.contains(t[0], n)) && e(n).focus(), s = t.parent(), "static" === t.css("position") ? (s.css({
                    position: "relative"
                }), t.css({
                    position: "relative"
                })) : (e.extend(i, {
                    position: t.css("position"),
                    zIndex: t.css("z-index")
                }), e.each(["top", "left", "bottom", "right"], function(e, s) {
                    i[s] = t.css(s), isNaN(parseInt(i[s], 10)) && (i[s] = "auto")
                }), t.css({
                    position: "relative",
                    top: 0,
                    left: 0,
                    right: "auto",
                    bottom: "auto"
                })), t.css(a), s.css(i).show()
            },
            removeWrapper: function(t) {
                var i = document.activeElement;
                return t.parent().is(".ui-effects-wrapper") && (t.parent().replaceWith(t), (t[0] === i || e.contains(t[0], i)) && e(i).focus()), t
            },
            setTransition: function(t, i, s, a) {
                return a = a || {}, e.each(i, function(e, i) {
                    var n = t.cssUnit(i);
                    n[0] > 0 && (a[i] = n[0] * s + n[1])
                }), a
            }
        }), e.fn.extend({
            effect: function() {
                function i(t) {
                    function i() {
                        e.isFunction(n) && n.call(a[0]), e.isFunction(t) && t()
                    }
                    var a = e(this), n = s.complete, o = s.mode;
                    (a.is(":hidden") ? "hide" === o : "show" === o) ? (a[o](), i()) : r.call(a[0], s, i)
                }
                var s = t.apply(this, arguments), a = s.mode, n = s.queue, r = e.effects.effect[s.effect];
                return e.fx.off ||!r ? a ? this[a](s.duration, s.complete) : this.each(function() {
                    s.complete && s.complete.call(this)
                }) : n===!1 ? this.each(i) : this.queue(n || "fx", i)
            },
            show: function(e) {
                return function(s) {
                    if (i(s))
                        return e.apply(this, arguments);
                    var a = t.apply(this, arguments);
                    return a.mode = "show", this.effect.call(this, a)
                }
            }(e.fn.show),
            hide: function(e) {
                return function(s) {
                    if (i(s))
                        return e.apply(this, arguments);
                    var a = t.apply(this, arguments);
                    return a.mode = "hide", this.effect.call(this, a)
                }
            }(e.fn.hide),
            toggle: function(e) {
                return function(s) {
                    if (i(s) || "boolean" == typeof s)
                        return e.apply(this, arguments);
                    var a = t.apply(this, arguments);
                    return a.mode = "toggle", this.effect.call(this, a)
                }
            }(e.fn.toggle),
            cssUnit: function(t) {
                var i = this.css(t), s = [];
                return e.each(["em", "px", "%", "pt"], function(e, t) {
                    i.indexOf(t) > 0 && (s = [parseFloat(i), t])
                }), s
            }
        })
    }(), function() {
        var t = {};
        e.each(["Quad", "Cubic", "Quart", "Quint", "Expo"], function(e, i) {
            t[i] = function(t) {
                return Math.pow(t, e + 2)
            }
        }), e.extend(t, {
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
                for (var t, i = 4; ((t = Math.pow(2, --i)) - 1) / 11 > e;);
                return 1 / Math.pow(4, 3 - i) - 7.5625 * Math.pow((3 * t - 2) / 22 - e, 2)
            }
        }), e.each(t, function(t, i) {
            e.easing["easeIn" + t] = i, e.easing["easeOut" + t] = function(e) {
                return 1 - i(1 - e)
            }, e.easing["easeInOut" + t] = function(e) {
                return .5 > e ? i(2 * e) / 2 : 1 - i( - 2 * e + 2) / 2
            }
        })
    }(), e.effects, e.effects.effect.blind = function(t, i) {
        var s, a, n, r = e(this), o = /up|down|vertical/, h = /up|left|vertical|horizontal/, l = ["position", "top", "bottom", "left", "right", "height", "width"], u = e.effects.setMode(r, t.mode || "hide"), d = t.direction || "up", c = o.test(d), p = c ? "height": "width", f = c ? "top": "left", m = h.test(d), g = {}, v = "show" === u;
        r.parent().is(".ui-effects-wrapper") ? e.effects.save(r.parent(), l) : e.effects.save(r, l), r.show(), s = e.effects.createWrapper(r).css({
            overflow: "hidden"
        }), a = s[p](), n = parseFloat(s.css(f)) || 0, g[p] = v ? a : 0, m || (r.css(c ? "bottom" : "right", 0).css(c ? "top" : "left", "auto").css({
            position: "absolute"
        }), g[f] = v ? n : a + n), v && (s.css(p, 0), m || s.css(f, n + a)), s.animate(g, {
            duration: t.duration,
            easing: t.easing,
            queue: !1,
            complete: function() {
                "hide" === u && r.hide(), e.effects.restore(r, l), e.effects.removeWrapper(r), i()
            }
        })
    }, e.effects.effect.bounce = function(t, i) {
        var s, a, n, r = e(this), o = ["position", "top", "bottom", "left", "right", "height", "width"], h = e.effects.setMode(r, t.mode || "effect"), l = "hide" === h, u = "show" === h, d = t.direction || "up", c = t.distance, p = t.times || 5, f = 2 * p + (u || l ? 1 : 0), m = t.duration / f, g = t.easing, v = "up" === d || "down" === d ? "top": "left", y = "up" === d || "left" === d, b = r.queue(), _ = b.length;
        for ((u || l) && o.push("opacity"), e.effects.save(r, o), r.show(), e.effects.createWrapper(r), c || (c = r["top" === v ? "outerHeight": "outerWidth"]() / 3), u && (n = {
            opacity: 1
        }, n[v] = 0, r.css("opacity", 0).css(v, y ? 2*-c : 2 * c).animate(n, m, g)), l && (c/=Math.pow(2, p - 1)), n = {}, n[v] = 0, s = 0; p > s; s++)
            a = {}, a[v] = (y ? "-=" : "+=") + c, r.animate(a, m, g).animate(n, m, g), c = l ? 2 * c : c / 2;
        l && (a = {
            opacity: 0
        }, a[v] = (y ? "-=" : "+=") + c, r.animate(a, m, g)), r.queue(function() {
            l && r.hide(), e.effects.restore(r, o), e.effects.removeWrapper(r), i()
        }), _ > 1 && b.splice.apply(b, [1, 0].concat(b.splice(_, f + 1))), r.dequeue()
    }, e.effects.effect.clip = function(t, i) {
        var s, a, n, r = e(this), o = ["position", "top", "bottom", "left", "right", "height", "width"], h = e.effects.setMode(r, t.mode || "hide"), l = "show" === h, u = t.direction || "vertical", d = "vertical" === u, c = d ? "height": "width", p = d ? "top": "left", f = {};
        e.effects.save(r, o), r.show(), s = e.effects.createWrapper(r).css({
            overflow: "hidden"
        }), a = "IMG" === r[0].tagName ? s : r, n = a[c](), l && (a.css(c, 0), a.css(p, n / 2)), f[c] = l ? n : 0, f[p] = l ? 0 : n / 2, a.animate(f, {
            queue: !1,
            duration: t.duration,
            easing: t.easing,
            complete: function() {
                l || r.hide(), e.effects.restore(r, o), e.effects.removeWrapper(r), i()
            }
        })
    }, e.effects.effect.drop = function(t, i) {
        var s, a = e(this), n = ["position", "top", "bottom", "left", "right", "opacity", "height", "width"], r = e.effects.setMode(a, t.mode || "hide"), o = "show" === r, h = t.direction || "left", l = "up" === h || "down" === h ? "top": "left", u = "up" === h || "left" === h ? "pos": "neg", d = {
            opacity: o ? 1: 0
        };
        e.effects.save(a, n), a.show(), e.effects.createWrapper(a), s = t.distance || a["top" === l ? "outerHeight": "outerWidth"](!0) / 2, o && a.css("opacity", 0).css(l, "pos" === u?-s : s), d[l] = (o ? "pos" === u ? "+=" : "-=" : "pos" === u ? "-=" : "+=") + s, a.animate(d, {
            queue: !1,
            duration: t.duration,
            easing: t.easing,
            complete: function() {
                "hide" === r && a.hide(), e.effects.restore(a, n), e.effects.removeWrapper(a), i()
            }
        })
    }, e.effects.effect.explode = function(t, i) {
        function s() {
            b.push(this), b.length === d * c && a()
        }
        function a() {
            p.css({
                visibility: "visible"
            }), e(b).remove(), m || p.hide(), i()
        }
        var n, r, o, h, l, u, d = t.pieces ? Math.round(Math.sqrt(t.pieces)): 3, c = d, p = e(this), f = e.effects.setMode(p, t.mode || "hide"), m = "show" === f, g = p.show().css("visibility", "hidden").offset(), v = Math.ceil(p.outerWidth() / c), y = Math.ceil(p.outerHeight() / d), b = [];
        for (n = 0; d > n; n++)
            for (h = g.top + n * y, u = n - (d - 1) / 2, r = 0; c > r; r++)
                o = g.left + r * v, l = r - (c - 1) / 2, p.clone().appendTo("body").wrap("<div></div>").css({
                    position: "absolute",
                    visibility: "visible",
                    left: - r * v,
                    top: - n * y
                }).parent().addClass("ui-effects-explode").css({
                    position: "absolute",
                    overflow: "hidden",
                    width: v,
                    height: y,
                    left: o + (m ? l * v : 0),
                    top: h + (m ? u * y : 0),
                    opacity: m ? 0: 1
                }).animate({
                    left: o + (m ? 0 : l * v),
                    top: h + (m ? 0 : u * y),
                    opacity: m ? 1: 0
                }, t.duration || 500, t.easing, s)
    }, e.effects.effect.fade = function(t, i) {
        var s = e(this), a = e.effects.setMode(s, t.mode || "toggle");
        s.animate({
            opacity: a
        }, {
            queue: !1,
            duration: t.duration,
            easing: t.easing,
            complete: i
        })
    }, e.effects.effect.fold = function(t, i) {
        var s, a, n = e(this), r = ["position", "top", "bottom", "left", "right", "height", "width"], o = e.effects.setMode(n, t.mode || "hide"), h = "show" === o, l = "hide" === o, u = t.size || 15, d = /([0-9]+)%/.exec(u), c=!!t.horizFirst, p = h !== c, f = p ? ["width", "height"] : ["height", "width"], m = t.duration / 2, g = {}, v = {};
        e.effects.save(n, r), n.show(), s = e.effects.createWrapper(n).css({
            overflow: "hidden"
        }), a = p ? [s.width(), s.height()] : [s.height(), s.width()], d && (u = parseInt(d[1], 10) / 100 * a[l ? 0: 1]), h && s.css(c ? {
            height: 0,
            width: u
        } : {
            height: u,
            width: 0
        }), g[f[0]] = h ? a[0] : u, v[f[1]] = h ? a[1] : 0, s.animate(g, m, t.easing).animate(v, m, t.easing, function() {
            l && n.hide(), e.effects.restore(n, r), e.effects.removeWrapper(n), i()
        })
    }, e.effects.effect.highlight = function(t, i) {
        var s = e(this), a = ["backgroundImage", "backgroundColor", "opacity"], n = e.effects.setMode(s, t.mode || "show"), r = {
            backgroundColor: s.css("backgroundColor")
        };
        "hide" === n && (r.opacity = 0), e.effects.save(s, a), s.show().css({
            backgroundImage: "none",
            backgroundColor: t.color || "#ffff99"
        }).animate(r, {
            queue: !1,
            duration: t.duration,
            easing: t.easing,
            complete: function() {
                "hide" === n && s.hide(), e.effects.restore(s, a), i()
            }
        })
    }, e.effects.effect.size = function(t, i) {
        var s, a, n, r = e(this), o = ["position", "top", "bottom", "left", "right", "width", "height", "overflow", "opacity"], h = ["position", "top", "bottom", "left", "right", "overflow", "opacity"], l = ["width", "height", "overflow"], u = ["fontSize"], d = ["borderTopWidth", "borderBottomWidth", "paddingTop", "paddingBottom"], c = ["borderLeftWidth", "borderRightWidth", "paddingLeft", "paddingRight"], p = e.effects.setMode(r, t.mode || "effect"), f = t.restore || "effect" !== p, m = t.scale || "both", g = t.origin || ["middle", "center"], v = r.css("position"), y = f ? o: h, b = {
            height: 0,
            width: 0,
            outerHeight: 0,
            outerWidth: 0
        };
        "show" === p && r.show(), s = {
            height: r.height(),
            width: r.width(),
            outerHeight: r.outerHeight(),
            outerWidth: r.outerWidth()
        }, "toggle" === t.mode && "show" === p ? (r.from = t.to || b, r.to = t.from || s) : (r.from = t.from || ("show" === p ? b : s), r.to = t.to || ("hide" === p ? b : s)), n = {
            from: {
                y: r.from.height / s.height,
                x: r.from.width / s.width
            },
            to: {
                y: r.to.height / s.height,
                x: r.to.width / s.width
            }
        }, ("box" === m || "both" === m) && (n.from.y !== n.to.y && (y = y.concat(d), r.from = e.effects.setTransition(r, d, n.from.y, r.from), r.to = e.effects.setTransition(r, d, n.to.y, r.to)), n.from.x !== n.to.x && (y = y.concat(c), r.from = e.effects.setTransition(r, c, n.from.x, r.from), r.to = e.effects.setTransition(r, c, n.to.x, r.to))), ("content" === m || "both" === m) && n.from.y !== n.to.y && (y = y.concat(u).concat(l), r.from = e.effects.setTransition(r, u, n.from.y, r.from), r.to = e.effects.setTransition(r, u, n.to.y, r.to)), e.effects.save(r, y), r.show(), e.effects.createWrapper(r), r.css("overflow", "hidden").css(r.from), g && (a = e.effects.getBaseline(g, s), r.from.top = (s.outerHeight - r.outerHeight()) * a.y, r.from.left = (s.outerWidth - r.outerWidth()) * a.x, r.to.top = (s.outerHeight - r.to.outerHeight) * a.y, r.to.left = (s.outerWidth - r.to.outerWidth) * a.x), r.css(r.from), ("content" === m || "both" === m) && (d = d.concat(["marginTop", "marginBottom"]).concat(u), c = c.concat(["marginLeft", "marginRight"]), l = o.concat(d).concat(c), r.find("*[width]").each(function() {
            var i = e(this), s = {
                height: i.height(),
                width: i.width(),
                outerHeight: i.outerHeight(),
                outerWidth: i.outerWidth()
            };
            f && e.effects.save(i, l), i.from = {
                height: s.height * n.from.y,
                width: s.width * n.from.x,
                outerHeight: s.outerHeight * n.from.y,
                outerWidth: s.outerWidth * n.from.x
            }, i.to = {
                height: s.height * n.to.y,
                width: s.width * n.to.x,
                outerHeight: s.height * n.to.y,
                outerWidth: s.width * n.to.x
            }, n.from.y !== n.to.y && (i.from = e.effects.setTransition(i, d, n.from.y, i.from), i.to = e.effects.setTransition(i, d, n.to.y, i.to)), n.from.x !== n.to.x && (i.from = e.effects.setTransition(i, c, n.from.x, i.from), i.to = e.effects.setTransition(i, c, n.to.x, i.to)), i.css(i.from), i.animate(i.to, t.duration, t.easing, function() {
                f && e.effects.restore(i, l)
            })
        })), r.animate(r.to, {
            queue: !1,
            duration: t.duration,
            easing: t.easing,
            complete: function() {
                0 === r.to.opacity && r.css("opacity", r.from.opacity), "hide" === p && r.hide(), e.effects.restore(r, y), f || ("static" === v ? r.css({
                    position: "relative",
                    top: r.to.top,
                    left: r.to.left
                }) : e.each(["top", "left"], function(e, t) {
                    r.css(t, function(t, i) {
                        var s = parseInt(i, 10), a = e ? r.to.left: r.to.top;
                        return "auto" === i ? a + "px" : s + a + "px"
                    })
                })), e.effects.removeWrapper(r), i()
            }
        })
    }, e.effects.effect.scale = function(t, i) {
        var s = e(this), a = e.extend(!0, {}, t), n = e.effects.setMode(s, t.mode || "effect"), r = parseInt(t.percent, 10) || (0 === parseInt(t.percent, 10) ? 0 : "hide" === n ? 0 : 100), o = t.direction || "both", h = t.origin, l = {
            height: s.height(),
            width: s.width(),
            outerHeight: s.outerHeight(),
            outerWidth: s.outerWidth()
        }, u = {
            y: "horizontal" !== o ? r / 100: 1,
            x: "vertical" !== o ? r / 100: 1
        };
        a.effect = "size", a.queue=!1, a.complete = i, "effect" !== n && (a.origin = h || ["middle", "center"], a.restore=!0), a.from = t.from || ("show" === n ? {
            height: 0,
            width: 0,
            outerHeight: 0,
            outerWidth: 0
        } : l), a.to = {
            height: l.height * u.y,
            width: l.width * u.x,
            outerHeight: l.outerHeight * u.y,
            outerWidth: l.outerWidth * u.x
        }, a.fade && ("show" === n && (a.from.opacity = 0, a.to.opacity = 1), "hide" === n && (a.from.opacity = 1, a.to.opacity = 0)), s.effect(a)
    }, e.effects.effect.puff = function(t, i) {
        var s = e(this), a = e.effects.setMode(s, t.mode || "hide"), n = "hide" === a, r = parseInt(t.percent, 10) || 150, o = r / 100, h = {
            height: s.height(),
            width: s.width(),
            outerHeight: s.outerHeight(),
            outerWidth: s.outerWidth()
        };
        e.extend(t, {
            effect: "scale",
            queue: !1,
            fade: !0,
            mode: a,
            complete: i,
            percent: n ? r: 100,
            from: n ? h: {
                height: h.height * o,
                width: h.width * o,
                outerHeight: h.outerHeight * o,
                outerWidth: h.outerWidth * o
            }
        }), s.effect(t)
    }, e.effects.effect.pulsate = function(t, i) {
        var s, a = e(this), n = e.effects.setMode(a, t.mode || "show"), r = "show" === n, o = "hide" === n, h = r || "hide" === n, l = 2 * (t.times || 5) + (h ? 1 : 0), u = t.duration / l, d = 0, c = a.queue(), p = c.length;
        for ((r ||!a.is(":visible")) && (a.css("opacity", 0).show(), d = 1), s = 1; l > s; s++)
            a.animate({
                opacity: d
            }, u, t.easing), d = 1 - d;
        a.animate({
            opacity: d
        }, u, t.easing), a.queue(function() {
            o && a.hide(), i()
        }), p > 1 && c.splice.apply(c, [1, 0].concat(c.splice(p, l + 1))), a.dequeue()
    }, e.effects.effect.shake = function(t, i) {
        var s, a = e(this), n = ["position", "top", "bottom", "left", "right", "height", "width"], r = e.effects.setMode(a, t.mode || "effect"), o = t.direction || "left", h = t.distance || 20, l = t.times || 3, u = 2 * l + 1, d = Math.round(t.duration / u), c = "up" === o || "down" === o ? "top": "left", p = "up" === o || "left" === o, f = {}, m = {}, g = {}, v = a.queue(), y = v.length;
        for (e.effects.save(a, n), a.show(), e.effects.createWrapper(a), f[c] = (p ? "-=" : "+=") + h, m[c] = (p ? "+=" : "-=") + 2 * h, g[c] = (p ? "-=" : "+=") + 2 * h, a.animate(f, d, t.easing), s = 1; l > s; s++)
            a.animate(m, d, t.easing).animate(g, d, t.easing);
        a.animate(m, d, t.easing).animate(f, d / 2, t.easing).queue(function() {
            "hide" === r && a.hide(), e.effects.restore(a, n), e.effects.removeWrapper(a), i()
        }), y > 1 && v.splice.apply(v, [1, 0].concat(v.splice(y, u + 1))), a.dequeue()
    }, e.effects.effect.slide = function(t, i) {
        var s, a = e(this), n = ["position", "top", "bottom", "left", "right", "width", "height"], r = e.effects.setMode(a, t.mode || "show"), o = "show" === r, h = t.direction || "left", l = "up" === h || "down" === h ? "top": "left", u = "up" === h || "left" === h, d = {};
        e.effects.save(a, n), a.show(), s = t.distance || a["top" === l ? "outerHeight": "outerWidth"](!0), e.effects.createWrapper(a).css({
            overflow: "hidden"
        }), o && a.css(l, u ? isNaN(s) ? "-" + s : - s : s), d[l] = (o ? u ? "+=" : "-=" : u ? "-=" : "+=") + s, a.animate(d, {
            queue: !1,
            duration: t.duration,
            easing: t.easing,
            complete: function() {
                "hide" === r && a.hide(), e.effects.restore(a, n), e.effects.removeWrapper(a), i()
            }
        })
    }, e.effects.effect.transfer = function(t, i) {
        var s = e(this), a = e(t.to), n = "fixed" === a.css("position"), r = e("body"), o = n ? r.scrollTop(): 0, h = n ? r.scrollLeft(): 0, l = a.offset(), u = {
            top: l.top - o,
            left: l.left - h,
            height: a.innerHeight(),
            width: a.innerWidth()
        }, d = s.offset(), c = e("<div class='ui-effects-transfer'></div>").appendTo(document.body).addClass(t.className).css({
            top: d.top - o,
            left: d.left - h,
            height: s.innerHeight(),
            width: s.innerWidth(),
            position: n ? "fixed": "absolute"
        }).animate(u, t.duration, t.easing, function() {
            c.remove(), i()
        })
    }
});
(function($) {
    $.fn.jCarouselLite = function(o) {
        o = $.extend({
            btnPrev: null,
            btnNext: null,
            btnGo: null,
            mouseWheel: false,
            auto: null,
            speed: 200,
            easing: null,
            vertical: false,
            circular: true,
            visible: 3,
            start: 0,
            scroll: 1,
            beforeStart: null,
            afterEnd: null
        }, o || {});
        return this.each(function() {
            var b = false, animCss = o.vertical ? "top": "left", sizeCss = o.vertical ? "height": "width";
            var c = $(this), ul = $("ul", c), tLi = $("li", ul), tl = tLi.length, v = o.visible;
            if (o.circular) {
                ul.prepend(tLi.slice(tl - v - 1 + 1).clone()).append(tLi.slice(0, v).clone());
                o.start += v
            }
            var f = $("li", ul), itemLength = f.length, curr = o.start;
            c.css("visibility", "visible");
            f.css({
                overflow: "hidden",
                float: o.vertical ? "none": "left"
            });
            ul.css({
                margin: "0",
                padding: "0",
                position: "relative",
                "list-style-type": "none",
                "z-index": "1"
            });
            c.css({
                overflow: "hidden",
                position: "relative",
                "z-index": "2",
                left: "0px"
            });
            var g = o.vertical ? height(f): width(f);
            var h = g * itemLength;
            var j = g * v;
            f.css({
                width: f.width(),
                height: f.height()
            });
            ul.css(sizeCss, h + "px").css(animCss, - (curr * g));
            c.css(sizeCss, j + "px");
            if (o.btnPrev)
                $(o.btnPrev).click(function() {
                    return go(curr - o.scroll)
                });
            if (o.btnNext)
                $(o.btnNext).click(function() {
                    return go(curr + o.scroll)
                });
            if (o.btnGo)
                $.each(o.btnGo, function(i, a) {
                    $(a).click(function() {
                        return go(o.circular ? o.visible + i : i)
                    })
                });
            if (o.mouseWheel && c.mousewheel)
                c.mousewheel(function(e, d) {
                    return d > 0 ? go(curr - o.scroll) : go(curr + o.scroll)
                });
            if (o.auto)
                setInterval(function() {
                    go(curr + o.scroll)
                }, o.auto + o.speed);
            function vis() {
                return f.slice(curr).slice(0, v)
            };
            function go(a) {
                if (!b) {
                    if (o.beforeStart)
                        o.beforeStart.call(this, vis());
                    if (o.circular) {
                        if (a <= o.start - v - 1) {
                            ul.css(animCss, - ((itemLength - (v * 2)) * g) + "px");
                            curr = a == o.start - v - 1 ? itemLength - (v * 2) - 1 : itemLength - (v * 2) - o.scroll
                        } else if (a >= itemLength - v + 1) {
                            ul.css(animCss, - ((v) * g) + "px");
                            curr = a == itemLength - v + 1 ? v + 1 : v + o.scroll
                        } else 
                            curr = a
                    } else {
                        if (a < 0 || a > itemLength - v)
                            return;
                        else 
                            curr = a
                    }
                    b = true;
                    ul.animate(animCss == "left" ? {
                        left: - (curr * g)
                    } : {
                        top: - (curr * g)
                    }, o.speed, o.easing, function() {
                        if (o.afterEnd)
                            o.afterEnd.call(this, vis());
                        b = false
                    });
                    if (!o.circular) {
                        $(o.btnPrev + "," + o.btnNext).removeClass("disabled");
                        $((curr - o.scroll < 0 && o.btnPrev) || (curr + o.scroll > itemLength - v && o.btnNext) || []).addClass("disabled")
                    }
                }
                return false
            }
            c[0].go = go
        })
    };
    function css(a, b) {
        return parseInt($.css(a[0], b)) || 0
    };
    function width(a) {
        return a[0].offsetWidth + css(a, 'marginLeft') + css(a, 'marginRight')
    };
    function height(a) {
        return a[0].offsetHeight + css(a, 'marginTop') + css(a, 'marginBottom')
    }
})(jQuery);
(function(a) {
    a.extend(a.fn, {
        clearingInput: function(b) {
            var c = {
                blurClass: "blur"
            };
            b = a.extend(c, b);
            return this.each(function() {
                var e = a(this).addClass(b.blurClass);
                var g = e.parents("form:first");
                var f, h;
                h = b.text || d() || e.val();
                if (h) {
                    e.val(h);
                    e.blur(function() {
                        if (e.val() === "") {
                            e.addClass(b.blurClass).val(h)
                        }
                    }).focus(function() {
                        if (e.val() === h) {
                            e.val("")
                        }
                        e.removeClass(b.blurClass)
                    });
                    g.submit(function() {
                        if (e.hasClass(b.blurClass)) {
                            e.val("")
                        }
                    });
                    e.blur()
                }
                function d() {
                    f = g.find("label[for=" + e.attr("id") + "]");
                    return f ? f.css({
                        position: "absolute",
                        left: "-9999px"
                    }).text() : ""
                }
            })
        }
    })
})(jQuery);
jQuery.cookie = function(b, j, m) {
    if (typeof j != "undefined") {
        m = m || {};
        if (j === null) {
            j = "";
            m.expires =- 1
        }
        var e = "";
        if (m.expires && (typeof m.expires == "number" || m.expires.toUTCString)) {
            var f;
            if (typeof m.expires == "number") {
                f = new Date();
                f.setTime(f.getTime() + (m.expires * 24 * 60 * 60 * 1000))
            } else {
                f = m.expires
            }
            e = "; expires=" + f.toUTCString()
        }
        var l = m.path ? "; path=" + (m.path): "";
        var g = m.domain ? "; domain=" + (m.domain): "";
        var a = m.secure ? "; secure": "";
        document.cookie = [b, "=", encodeURIComponent(j), e, l, g, a].join("")
    } else {
        var d = null;
        if (document.cookie && document.cookie != "") {
            var k = document.cookie.split(";");
            for (var h = 0; h < k.length; h++) {
                var c = jQuery.trim(k[h]);
                if (c.substring(0, b.length + 1) == (b + "=")) {
                    d = decodeURIComponent(c.substring(b.length + 1));
                    break
                }
            }
        }
        return d
    }
};
jQuery.fn.truncate = function(a, d) {
    d = jQuery.extend({
        chars: /\s/,
        trail: ["...", ""]
    }, d);
    var b = {};
    var e = $.browser.msie;
    function c(f) {
        if (e) {
            f.style.removeAttribute("filter")
        }
    }
    return this.each(function() {
        var k = jQuery(this);
        var g = k.html().replace(/\r\n/gim, "");
        var i = g;
        var m = /<\/?[^<>]*\/?>/gim;
        var h;
        var n = {};
        var l = $("*").index(this);
        while ((h = m.exec(i)) != null) {
            n[h.index] = h[0]
        }
        i = jQuery.trim(i.split(m).join(""));
        if (i.length > a) {
            var j;
            while (a < i.length) {
                j = i.charAt(a);
                if (j.match(d.chars)) {
                    i = i.substring(0, a);
                    break
                }
                a--;
                if (a < 0) {
                    return 
                }
            }
            if (g.search(m)!=-1) {
                var f = 0;
                for (eachEl in n) {
                    i = [i.substring(0, eachEl), n[eachEl], i.substring(eachEl, i.length)].join("");
                    if (eachEl < i.length) {
                        f = i.length
                    }
                }
                k.html([i.substring(0, f), i.substring(f, i.length).replace(/<(\w+)[^>]*>.*<\/\1>/gim, "").replace(/<(br|hr|img|input)[^<>]*\/?>/gim, "")].join(""))
            } else {
                k.html(i)
            }
            b[l] = g;
            k.html(["<div class='truncate_less'>", k.html(), d.trail[0], "</div>"].join("")).find(".truncate_show", this).click(function() {
                if (k.find(".truncate_more").length == 0) {
                    k.append(["<div class='truncate_more' style='display: none;'>", b[l], d.trail[1], "</div>"].join("")).find(".truncate_hide").click(function() {
                        k.find(".truncate_more").css("background", "#fff").fadeOut("normal", function() {
                            k.find(".truncate_less").css("background", "#fff").fadeIn("normal", function() {
                                c(this);
                                $(this).css("background", "none")
                            });
                            c(this)
                        });
                        return false
                    })
                }
                k.find(".truncate_less").fadeOut("normal", function() {
                    k.find(".truncate_more").fadeIn("normal", function() {
                        c(this)
                    });
                    c(this)
                });
                jQuery(".truncate_show", k).click(function() {
                    k.find(".truncate_less").css("background", "#fff").fadeOut("normal", function() {
                        k.find(".truncate_more").css("background", "#fff").fadeIn("normal", function() {
                            c(this);
                            $(this).css("background", "none")
                        });
                        c(this)
                    });
                    return false
                });
                return false
            })
        }
    })
};
(function(a) {
    jQuery.fn.extend({
        elastic: function() {
            var b = ["paddingTop", "paddingRight", "paddingBottom", "paddingLeft", "fontSize", "lineHeight", "fontFamily", "width", "fontWeight", "border-top-width", "border-right-width", "border-bottom-width", "border-left-width", "borderTopStyle", "borderTopColor", "borderRightStyle", "borderRightColor", "borderBottomStyle", "borderBottomColor", "borderLeftStyle", "borderLeftColor"];
            return this.each(function() {
                if (this.type !== "textarea") {
                    return false
                }
                var g = jQuery(this), c = jQuery("<div />").css({
                    position: "absolute",
                    display: "none",
                    "word-wrap": "break-word",
                    "white-space": "pre-wrap"
                }), j = parseInt(g.css("line-height"), 10) || parseInt(g.css("font-size"), "10"), l = parseInt(g.css("height"), 10) || j * 3, k = parseInt(g.css("max-height"), 10) || Number.MAX_VALUE, d = 0;
                if (k < 0) {
                    k = Number.MAX_VALUE
                }
                c.appendTo(g.parent());
                var f = b.length;
                while (f--) {
                    c.css(b[f].toString(), g.css(b[f].toString()))
                }
                function h() {
                    var i = Math.floor(parseInt(g.width(), 10));
                    if (c.width() !== i) {
                        c.css({
                            width: i + "px"
                        });
                        e(true)
                    }
                }
                function m(i, o) {
                    var n = Math.floor(parseInt(i, 10));
                    if (g.height() !== n) {
                        g.css({
                            height: n + "px",
                            overflow: o
                        })
                    }
                }
                function e(p) {
                    var o = g.val().replace(/&/g, "&amp;").replace(/ {2}/g, "&nbsp;").replace(/<|>/g, "&gt;").replace(/\n/g, "<br />");
                    var i = c.html().replace(/<br>/ig, "<br />");
                    if (p || o + "&nbsp;" !== i) {
                        c.html(o + "&nbsp;");
                        if (Math.abs(c.height() + j - g.height()) > 3) {
                            var n = c.height() + j;
                            if (n >= k) {
                                m(k, "auto")
                            } else {
                                if (n <= l) {
                                    m(l, "hidden")
                                } else {
                                    m(n, "hidden")
                                }
                            }
                        }
                    }
                }
                g.css({
                    overflow: "hidden"
                });
                g.bind("keyup change cut paste", function() {
                    e()
                });
                a(window).bind("resize", h);
                g.bind("resize", h);
                g.bind("update", e);
                g.bind("blur", function() {
                    if (c.height() < k) {
                        if (c.height() > l) {
                            g.height(c.height())
                        } else {
                            g.height(l)
                        }
                    }
                });
                g.bind("input paste", function(i) {
                    setTimeout(e, 250)
                });
                e()
            })
        }
    })
})(jQuery);
(function(b) {
    b.fn.daterangepicker = function(y) {
        var N = b(this);
        var M = b.extend({
            presetRanges: [{
                text: "Today",
                dateStart: "today",
                dateEnd: "today"
            }, {
                text: "Last 7 days",
                dateStart: "today-7days",
                dateEnd: "today"
            }, {
                text: "Month to date",
                dateStart: function() {
                    return Date.parse("today").moveToFirstDayOfMonth()
                },
                dateEnd: "today"
            }, {
                text: "Year to date",
                dateStart: function() {
                    var c = Date.parse("today");
                    c.setMonth(0);
                    c.setDate(1);
                    return c
                },
                dateEnd: "today"
            }, {
                text: "The previous Month",
                dateStart: function() {
                    return Date.parse("1 month ago").moveToFirstDayOfMonth()
                },
                dateEnd: function() {
                    return Date.parse("1 month ago").moveToLastDayOfMonth()
                }
            }
            ],
            presets: {
                specificDate: "Specific Date",
                allDatesBefore: "All Dates Before",
                allDatesAfter: "All Dates After",
                dateRange: "Date Range"
            },
            rangeStartTitle: "Start date",
            rangeEndTitle: "End date",
            nextLinkText: "Next",
            prevLinkText: "Prev",
            target: N,
            doneButtonText: "Done",
            earliestDate: Date.parse("-15years"),
            latestDate: Date.parse("+15years"),
            constrainDates: false,
            rangeSplitter: "-",
            dateFormat: "m/d/yy",
            closeOnSelect: true,
            arrows: false,
            appendTo: "body",
            onClose: function() {},
            onOpen: function() {},
            onChange: function() {},
            datepickerOptions: null
        }, y);
        var K = {
            onSelect: function(d, f) {
                var h = H.find(".range-start");
                var g = H.find(".range-end");
                if (H.find(".ui-daterangepicker-specificDate").is(".ui-state-active")) {
                    g.datepicker("setDate", h.datepicker("getDate"))
                }
                b(this).trigger("constrainOtherPicker");
                var c = P(h.datepicker("getDate"));
                var e = P(g.datepicker("getDate"));
                if (N.length == 2) {
                    N.eq(0).val(c);
                    N.eq(1).val(e)
                } else {
                    N.val((c != e) ? c + " " + M.rangeSplitter + " " + e : c)
                }
                if (M.closeOnSelect) {
                    if (!H.find("li.ui-state-active").is(".ui-daterangepicker-dateRange")&&!H.is(":animated")) {
                        G()
                    }
                    b(this).trigger("constrainOtherPicker");
                    M.onChange()
                }
            },
            defaultDate: + 0
        };
        N.bind("change", M.onChange);
        M.datepickerOptions = (y) ? b.extend(K, y.datepickerOptions) : K;
        var E, F = Date.parse("today");
        var C, I;
        if (N.size() == 2) {
            C = Date.parse(N.eq(0).val());
            I = Date.parse(N.eq(1).val());
            if (C == null) {
                C = I
            }
            if (I == null) {
                I = C
            }
        } else {
            C = Date.parse(N.val().split(M.rangeSplitter)[0]);
            I = Date.parse(N.val().split(M.rangeSplitter)[1]);
            if (I == null) {
                I = C
            }
        }
        if (C != null) {
            E = C
        }
        if (I != null) {
            F = I
        }
        var H = b('<div class="ui-daterangepicker ui-widget ui-helper-clearfix ui-widget-content ui-corner-all"></div>');
        var w = (function() {
            var d = b('<ul class="ui-widget-content"></ul>').appendTo(H);
            b.each(M.presetRanges, function() {
                b('<li class="ui-daterangepicker-' + this.text.replace(/ /g, "") + ' ui-corner-all"><a href="#">' + this.text + "</a></li>").data("dateStart", this.dateStart).data("dateEnd", this.dateEnd).appendTo(d)
            });
            var c = 0;
            b.each(M.presets, function(e, f) {
                b('<li class="ui-daterangepicker-' + e + " preset_" + c + ' ui-helper-clearfix ui-corner-all"><span class="ui-icon ui-icon-triangle-1-e"></span><a href="#">' + f + "</a></li>").appendTo(d);
                c++
            });
            d.find("li").hover(function() {
                b(this).addClass("ui-state-hover")
            }, function() {
                b(this).removeClass("ui-state-hover")
            }).click(function() {
                H.find(".ui-state-active").removeClass("ui-state-active");
                b(this).addClass("ui-state-active");
                A(b(this), H, D, L);
                return false
            });
            return d
        })();
        function P(g) {
            if (!g.getDate()) {
                return ""
            }
            var d = g.getDate();
            var c = g.getMonth();
            var f = g.getFullYear();
            c++;
            var e = M.dateFormat;
            return b.datepicker.formatDate(e, g)
        }
        b.fn.restoreDateFromData = function() {
            if (b(this).data("saveDate")) {
                b(this).datepicker("setDate", b(this).data("saveDate")).removeData("saveDate")
            }
            return this
        };
        b.fn.saveDateToData = function() {
            if (!b(this).data("saveDate")) {
                b(this).data("saveDate", b(this).datepicker("getDate"))
            }
            return this
        };
        function x() {
            if (H.data("state") == "closed") {
                a();
                H.fadeIn(300).data("state", "open");
                M.onOpen()
            }
        }
        function G() {
            if (H.data("state") == "open") {
                H.fadeOut(300).data("state", "closed");
                M.onClose()
            }
        }
        function O() {
            if (H.data("state") == "open") {
                G()
            } else {
                x()
            }
        }
        function a() {
            var e = B || N;
            var d = e.offset(), g = "left", f = d.left, c = b(window).width() - f - e.outerWidth();
            if (f > c) {
                g = "right", f = c
            }
            H.parent().css(g, f).css("top", d.top + e.outerHeight())
        }
        function A(f, h, d, e) {
            if (f.is(".ui-daterangepicker-specificDate")) {
                e.hide();
                d.show();
                h.find(".title-start").text(M.presets.specificDate);
                h.find(".range-start").restoreDateFromData().css("opacity", 1).show(400);
                h.find(".range-end").restoreDateFromData().css("opacity", 0).hide(400);
                setTimeout(function() {
                    e.fadeIn()
                }, 400)
            } else {
                if (f.is(".ui-daterangepicker-allDatesBefore")) {
                    e.hide();
                    d.show();
                    h.find(".title-end").text(M.presets.allDatesBefore);
                    h.find(".range-start").saveDateToData().datepicker("setDate", M.earliestDate).css("opacity", 0).hide(400);
                    h.find(".range-end").restoreDateFromData().css("opacity", 1).show(400);
                    setTimeout(function() {
                        e.fadeIn()
                    }, 400)
                } else {
                    if (f.is(".ui-daterangepicker-allDatesAfter")) {
                        e.hide();
                        d.show();
                        h.find(".title-start").text(M.presets.allDatesAfter);
                        h.find(".range-start").restoreDateFromData().css("opacity", 1).show(400);
                        h.find(".range-end").saveDateToData().datepicker("setDate", M.latestDate).css("opacity", 0).hide(400);
                        setTimeout(function() {
                            e.fadeIn()
                        }, 400)
                    } else {
                        if (f.is(".ui-daterangepicker-dateRange")) {
                            e.hide();
                            d.show();
                            h.find(".title-start").text(M.rangeStartTitle);
                            h.find(".title-end").text(M.rangeEndTitle);
                            h.find(".range-start").restoreDateFromData().css("opacity", 1).show(400);
                            h.find(".range-end").restoreDateFromData().css("opacity", 1).show(400);
                            setTimeout(function() {
                                e.fadeIn()
                            }, 400)
                        } else {
                            e.hide();
                            h.find(".range-start, .range-end").css("opacity", 0).hide(400, function() {
                                d.hide()
                            });
                            var g = (typeof f.data("dateStart") == "string") ? Date.parse(f.data("dateStart")): f.data("dateStart")();
                            var c = (typeof f.data("dateEnd") == "string") ? Date.parse(f.data("dateEnd")): f.data("dateEnd")();
                            h.find(".range-start").datepicker("setDate", g).find(".ui-datepicker-current-day").trigger("click");
                            h.find(".range-end").datepicker("setDate", c).find(".ui-datepicker-current-day").trigger("click")
                        }
                    }
                }
            }
            return false
        }
        var D = b('<div class="ranges ui-widget-header ui-corner-all ui-helper-clearfix"><div class="range-start"><span class="title-start">Start Date</span></div><div class="range-end"><span class="title-end">End Date</span></div></div>').appendTo(H);
        D.find(".range-start, .range-end").datepicker(M.datepickerOptions);
        D.find(".range-start").datepicker("setDate", E);
        D.find(".range-end").datepicker("setDate", F);
        D.find(".range-start, .range-end").bind("constrainOtherPicker", function() {
            if (M.constrainDates) {
                if (b(this).is(".range-start")) {
                    H.find(".range-end").datepicker("option", "minDate", b(this).datepicker("getDate"))
                } else {
                    H.find(".range-start").datepicker("option", "maxDate", b(this).datepicker("getDate"))
                }
            }
        }).trigger("constrainOtherPicker");
        var L = b('<button class="btnDone ui-state-default ui-corner-all">' + M.doneButtonText + "</button>").click(function() {
            H.find(".ui-datepicker-current-day").trigger("click");
            G()
        }).hover(function() {
            b(this).addClass("ui-state-hover")
        }, function() {
            b(this).removeClass("ui-state-hover")
        }).appendTo(D);
        b(this).click(function() {
            O();
            return false
        });
        D.hide().find(".range-start, .range-end, .btnDone").hide();
        H.data("state", "closed");
        D.find(".ui-datepicker").css("display", "block");
        b(M.appendTo).append(H);
        H.wrap('<div class="ui-daterangepickercontain"></div>');
        if (M.arrows && N.size() == 1) {
            var J = b('<a href="#" class="ui-daterangepicker-prev ui-corner-all" title="' + M.prevLinkText + '"><span class="ui-icon ui-icon-circle-triangle-w">' + M.prevLinkText + "</span></a>");
            var z = b('<a href="#" class="ui-daterangepicker-next ui-corner-all" title="' + M.nextLinkText + '"><span class="ui-icon ui-icon-circle-triangle-e">' + M.nextLinkText + "</span></a>");
            b(this).addClass("ui-rangepicker-input ui-widget-content").wrap('<div class="ui-daterangepicker-arrows ui-widget ui-widget-header ui-helper-clearfix ui-corner-all"></div>').before(J).before(z).parent().find("a").click(function() {
                var c = D.find(".range-start").datepicker("getDate");
                var d = D.find(".range-end").datepicker("getDate");
                var e = Math.abs(new TimeSpan(c - d).getTotalMilliseconds()) + 86400000;
                if (b(this).is(".ui-daterangepicker-prev")) {
                    e =- e
                }
                D.find(".range-start, .range-end ").each(function() {
                    var f = b(this).datepicker("getDate");
                    if (f == null) {
                        return false
                    }
                    b(this).datepicker("setDate", f.add({
                        milliseconds: e
                    })).find(".ui-datepicker-current-day").trigger("click")
                });
                return false
            }).hover(function() {
                b(this).addClass("ui-state-hover")
            }, function() {
                b(this).removeClass("ui-state-hover")
            });
            var B = N.parent()
        }
        b(document).click(function() {
            if (H.is(":visible")) {
                G()
            }
        });
        H.click(function() {
            return false
        }).hide();
        return this
    }
})(jQuery);
Date.CultureInfo = {
    name: "en-US",
    englishName: "English (United States)",
    nativeName: "English (United States)",
    dayNames: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    abbreviatedDayNames: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    shortestDayNames: ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"],
    firstLetterDayNames: ["S", "M", "T", "W", "T", "F", "S"],
    monthNames: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    abbreviatedMonthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    amDesignator: "AM",
    pmDesignator: "PM",
    firstDayOfWeek: 0,
    twoDigitYearMax: 2029,
    dateElementOrder: "mdy",
    formatPatterns: {
        shortDate: "M/d/yyyy",
        longDate: "dddd, MMMM dd, yyyy",
        shortTime: "h:mm tt",
        longTime: "h:mm:ss tt",
        fullDateTime: "dddd, MMMM dd, yyyy h:mm:ss tt",
        sortableDateTime: "yyyy-MM-ddTHH:mm:ss",
        universalSortableDateTime: "yyyy-MM-dd HH:mm:ssZ",
        rfc1123: "ddd, dd MMM yyyy HH:mm:ss GMT",
        monthDay: "MMMM dd",
        yearMonth: "MMMM, yyyy"
    },
    regexPatterns: {
        jan: /^jan(uary)?/i,
        feb: /^feb(ruary)?/i,
        mar: /^mar(ch)?/i,
        apr: /^apr(il)?/i,
        may: /^may/i,
        jun: /^jun(e)?/i,
        jul: /^jul(y)?/i,
        aug: /^aug(ust)?/i,
        sep: /^sep(t(ember)?)?/i,
        oct: /^oct(ober)?/i,
        nov: /^nov(ember)?/i,
        dec: /^dec(ember)?/i,
        sun: /^su(n(day)?)?/i,
        mon: /^mo(n(day)?)?/i,
        tue: /^tu(e(s(day)?)?)?/i,
        wed: /^we(d(nesday)?)?/i,
        thu: /^th(u(r(s(day)?)?)?)?/i,
        fri: /^fr(i(day)?)?/i,
        sat: /^sa(t(urday)?)?/i,
        future: /^next/i,
        past: /^last|past|prev(ious)?/i,
        add: /^(\+|after|from)/i,
        subtract: /^(\-|before|ago)/i,
        yesterday: /^yesterday/i,
        today: /^t(oday)?/i,
        tomorrow: /^tomorrow/i,
        now: /^n(ow)?/i,
        millisecond: /^ms|milli(second)?s?/i,
        second: /^sec(ond)?s?/i,
        minute: /^min(ute)?s?/i,
        hour: /^h(ou)?rs?/i,
        week: /^w(ee)?k/i,
        month: /^m(o(nth)?s?)?/i,
        day: /^d(ays?)?/i,
        year: /^y((ea)?rs?)?/i,
        shortMeridian: /^(a|p)/i,
        longMeridian: /^(a\.?m?\.?|p\.?m?\.?)/i,
        timezone: /^((e(s|d)t|c(s|d)t|m(s|d)t|p(s|d)t)|((gmt)?\s*(\+|\-)\s*\d\d\d\d?)|gmt)/i,
        ordinalSuffix: /^\s*(st|nd|rd|th)/i,
        timeContext: /^\s*(\:|a|p)/i
    },
    abbreviatedTimeZoneStandard: {
        GMT: "-000",
        EST: "-0400",
        CST: "-0500",
        MST: "-0600",
        PST: "-0700"
    },
    abbreviatedTimeZoneDST: {
        GMT: "-000",
        EDT: "-0500",
        CDT: "-0600",
        MDT: "-0700",
        PDT: "-0800"
    }
};
Date.getMonthNumberFromName = function(b) {
    var e = Date.CultureInfo.monthNames, a = Date.CultureInfo.abbreviatedMonthNames, d = b.toLowerCase();
    for (var c = 0; c < e.length; c++) {
        if (e[c].toLowerCase() == d || a[c].toLowerCase() == d) {
            return c
        }
    }
    return - 1
};
Date.getDayNumberFromName = function(b) {
    var f = Date.CultureInfo.dayNames, a = Date.CultureInfo.abbreviatedDayNames, e = Date.CultureInfo.shortestDayNames, d = b.toLowerCase();
    for (var c = 0; c < f.length; c++) {
        if (f[c].toLowerCase() == d || a[c].toLowerCase() == d) {
            return c
        }
    }
    return - 1
};
Date.isLeapYear = function(a) {
    return (((a%4 === 0) && (a%100 !== 0)) || (a%400 === 0))
};
Date.getDaysInMonth = function(a, b) {
    return [31, (Date.isLeapYear(a) ? 29 : 28), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][b]
};
Date.getTimezoneOffset = function(a, b) {
    return (b || false) ? Date.CultureInfo.abbreviatedTimeZoneDST[a.toUpperCase()] : Date.CultureInfo.abbreviatedTimeZoneStandard[a.toUpperCase()]
};
Date.getTimezoneAbbreviation = function(b, d) {
    var c = (d || false) ? Date.CultureInfo.abbreviatedTimeZoneDST: Date.CultureInfo.abbreviatedTimeZoneStandard, a;
    for (a in c) {
        if (c[a] === b) {
            return a
        }
    }
    return null
};
Date.prototype.clone = function() {
    return new Date(this.getTime())
};
Date.prototype.compareTo = function(a) {
    if (isNaN(this)) {
        throw new Error(this)
    }
    if (a instanceof Date&&!isNaN(a)) {
        return (this > a) ? 1 : (this < a)?-1 : 0
    } else {
        throw new TypeError(a)
    }
};
Date.prototype.equals = function(a) {
    return (this.compareTo(a) === 0)
};
Date.prototype.between = function(c, a) {
    var b = this.getTime();
    return b >= c.getTime() && b <= a.getTime()
};
Date.prototype.addMilliseconds = function(a) {
    this.setMilliseconds(this.getMilliseconds() + a);
    return this
};
Date.prototype.addSeconds = function(a) {
    return this.addMilliseconds(a * 1000)
};
Date.prototype.addMinutes = function(a) {
    return this.addMilliseconds(a * 60000)
};
Date.prototype.addHours = function(a) {
    return this.addMilliseconds(a * 3600000)
};
Date.prototype.addDays = function(a) {
    return this.addMilliseconds(a * 86400000)
};
Date.prototype.addWeeks = function(a) {
    return this.addMilliseconds(a * 604800000)
};
Date.prototype.addMonths = function(a) {
    var b = this.getDate();
    this.setDate(1);
    this.setMonth(this.getMonth() + a);
    this.setDate(Math.min(b, this.getDaysInMonth()));
    return this
};
Date.prototype.addYears = function(a) {
    return this.addMonths(a * 12)
};
Date.prototype.add = function(b) {
    if (typeof b == "number") {
        this._orient = b;
        return this
    }
    var a = b;
    if (a.millisecond || a.milliseconds) {
        this.addMilliseconds(a.millisecond || a.milliseconds)
    }
    if (a.second || a.seconds) {
        this.addSeconds(a.second || a.seconds)
    }
    if (a.minute || a.minutes) {
        this.addMinutes(a.minute || a.minutes)
    }
    if (a.hour || a.hours) {
        this.addHours(a.hour || a.hours)
    }
    if (a.month || a.months) {
        this.addMonths(a.month || a.months)
    }
    if (a.year || a.years) {
        this.addYears(a.year || a.years)
    }
    if (a.day || a.days) {
        this.addDays(a.day || a.days)
    }
    return this
};
Date._validate = function(d, c, a, b) {
    if (typeof d != "number") {
        throw new TypeError(d + " is not a Number.")
    } else {
        if (d < c || d > a) {
            throw new RangeError(d + " is not a valid value for " + b + ".")
        }
    }
    return true
};
Date.validateMillisecond = function(a) {
    return Date._validate(a, 0, 999, "milliseconds")
};
Date.validateSecond = function(a) {
    return Date._validate(a, 0, 59, "seconds")
};
Date.validateMinute = function(a) {
    return Date._validate(a, 0, 59, "minutes")
};
Date.validateHour = function(a) {
    return Date._validate(a, 0, 23, "hours")
};
Date.validateDay = function(c, a, b) {
    return Date._validate(c, 1, Date.getDaysInMonth(a, b), "days")
};
Date.validateMonth = function(a) {
    return Date._validate(a, 0, 11, "months")
};
Date.validateYear = function(a) {
    return Date._validate(a, 1, 9999, "seconds")
};
Date.prototype.set = function(b) {
    var a = b;
    if (!a.millisecond && a.millisecond !== 0) {
        a.millisecond =- 1
    }
    if (!a.second && a.second !== 0) {
        a.second =- 1
    }
    if (!a.minute && a.minute !== 0) {
        a.minute =- 1
    }
    if (!a.hour && a.hour !== 0) {
        a.hour =- 1
    }
    if (!a.day && a.day !== 0) {
        a.day =- 1
    }
    if (!a.month && a.month !== 0) {
        a.month =- 1
    }
    if (!a.year && a.year !== 0) {
        a.year =- 1
    }
    if (a.millisecond!=-1 && Date.validateMillisecond(a.millisecond)) {
        this.addMilliseconds(a.millisecond - this.getMilliseconds())
    }
    if (a.second!=-1 && Date.validateSecond(a.second)) {
        this.addSeconds(a.second - this.getSeconds())
    }
    if (a.minute!=-1 && Date.validateMinute(a.minute)) {
        this.addMinutes(a.minute - this.getMinutes())
    }
    if (a.hour!=-1 && Date.validateHour(a.hour)) {
        this.addHours(a.hour - this.getHours())
    }
    if (a.month!==-1 && Date.validateMonth(a.month)) {
        this.addMonths(a.month - this.getMonth())
    }
    if (a.year!=-1 && Date.validateYear(a.year)) {
        this.addYears(a.year - this.getFullYear())
    }
    if (a.day!=-1 && Date.validateDay(a.day, this.getFullYear(), this.getMonth())) {
        this.addDays(a.day - this.getDate())
    }
    if (a.timezone) {
        this.setTimezone(a.timezone)
    }
    if (a.timezoneOffset) {
        this.setTimezoneOffset(a.timezoneOffset)
    }
    return this
};
Date.prototype.clearTime = function() {
    this.setHours(0);
    this.setMinutes(0);
    this.setSeconds(0);
    this.setMilliseconds(0);
    return this
};
Date.prototype.isLeapYear = function() {
    var a = this.getFullYear();
    return (((a%4 === 0) && (a%100 !== 0)) || (a%400 === 0))
};
Date.prototype.isWeekday = function() {
    return !(this.is().sat() || this.is().sun())
};
Date.prototype.getDaysInMonth = function() {
    return Date.getDaysInMonth(this.getFullYear(), this.getMonth())
};
Date.prototype.moveToFirstDayOfMonth = function() {
    return this.set({
        day: 1
    })
};
Date.prototype.moveToLastDayOfMonth = function() {
    return this.set({
        day: this.getDaysInMonth()
    })
};
Date.prototype.moveToDayOfWeek = function(a, b) {
    var c = (a - this.getDay() + 7 * (b||+1))%7;
    return this.addDays((c === 0) ? c += 7 * (b||+1) : c)
};
Date.prototype.moveToMonth = function(c, a) {
    var b = (c - this.getMonth() + 12 * (a||+1))%12;
    return this.addMonths((b === 0) ? b += 12 * (a||+1) : b)
};
Date.prototype.getDayOfYear = function() {
    return Math.floor((this - new Date(this.getFullYear(), 0, 1)) / 86400000)
};
Date.prototype.getWeekOfYear = function(a) {
    var h = this.getFullYear(), c = this.getMonth(), f = this.getDate();
    var j = a || Date.CultureInfo.firstDayOfWeek;
    var e = 7 + 1 - new Date(h, 0, 1).getDay();
    if (e == 8) {
        e = 1
    }
    var b = ((Date.UTC(h, c, f, 0, 0, 0) - Date.UTC(h, 0, 1, 0, 0, 0)) / 86400000) + 1;
    var i = Math.floor((b - e + 7) / 7);
    if (i === j) {
        h--;
        var g = 7 + 1 - new Date(h, 0, 1).getDay();
        if (g == 2 || g == 8) {
            i = 53
        } else {
            i = 52
        }
    }
    return i
};
Date.prototype.isDST = function() {
    return this.toString().match(/(E|C|M|P)(S|D)T/)[2] == "D"
};
Date.prototype.getTimezone = function() {
    return Date.getTimezoneAbbreviation(this.getUTCOffset, this.isDST())
};
Date.prototype.setTimezoneOffset = function(b) {
    var a = this.getTimezoneOffset(), c = Number(b)*-6 / 10;
    this.addMinutes(c - a);
    return this
};
Date.prototype.setTimezone = function(a) {
    return this.setTimezoneOffset(Date.getTimezoneOffset(a))
};
Date.prototype.getUTCOffset = function() {
    var b = this.getTimezoneOffset()*-10 / 6, a;
    if (b < 0) {
        a = (b - 10000).toString();
        return a[0] + a.substr(2)
    } else {
        a = (b + 10000).toString();
        return "+" + a.substr(1)
    }
};
Date.prototype.getDayName = function(a) {
    return a ? Date.CultureInfo.abbreviatedDayNames[this.getDay()] : Date.CultureInfo.dayNames[this.getDay()]
};
Date.prototype.getMonthName = function(a) {
    return a ? Date.CultureInfo.abbreviatedMonthNames[this.getMonth()] : Date.CultureInfo.monthNames[this.getMonth()]
};
Date.prototype._toString = Date.prototype.toString;
Date.prototype.toString = function(c) {
    var a = this;
    var b = function b(d) {
        return (d.toString().length == 1) ? "0" + d : d
    };
    return c ? c.replace(/dd?d?d?|MM?M?M?|yy?y?y?|hh?|HH?|mm?|ss?|tt?|zz?z?/g, function(d) {
        switch (d) {
        case"hh":
            return b(a.getHours() < 13 ? a.getHours() : (a.getHours() - 12));
        case"h":
            return a.getHours() < 13 ? a.getHours() : (a.getHours() - 12);
        case"HH":
            return b(a.getHours());
        case"H":
            return a.getHours();
        case"mm":
            return b(a.getMinutes());
        case"m":
            return a.getMinutes();
        case"ss":
            return b(a.getSeconds());
        case"s":
            return a.getSeconds();
        case"yyyy":
            return a.getFullYear();
        case"yy":
            return a.getFullYear().toString().substring(2, 4);
        case"dddd":
            return a.getDayName();
        case"ddd":
            return a.getDayName(true);
        case"dd":
            return b(a.getDate());
        case"d":
            return a.getDate().toString();
        case"MMMM":
            return a.getMonthName();
        case"MMM":
            return a.getMonthName(true);
        case"MM":
            return b((a.getMonth() + 1));
        case"M":
            return a.getMonth() + 1;
        case"t":
            return a.getHours() < 12 ? Date.CultureInfo.amDesignator.substring(0, 1) : Date.CultureInfo.pmDesignator.substring(0, 1);
        case"tt":
            return a.getHours() < 12 ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator;
        case"zzz":
        case"zz":
        case"z":
            return ""
        }
    }) : this._toString()
};
Date.now = function() {
    return new Date()
};
Date.today = function() {
    return Date.now().clearTime()
};
Date.prototype._orient =+ 1;
Date.prototype.next = function() {
    this._orient =+ 1;
    return this
};
Date.prototype.last = Date.prototype.prev = Date.prototype.previous = function() {
    this._orient =- 1;
    return this
};
Date.prototype._is = false;
Date.prototype.is = function() {
    this._is = true;
    return this
};
Number.prototype._dateElement = "day";
Number.prototype.fromNow = function() {
    var a = {};
    a[this._dateElement] = this;
    return Date.now().add(a)
};
Number.prototype.ago = function() {
    var a = {};
    a[this._dateElement] = this*-1;
    return Date.now().add(a)
};
(function() {
    var g = Date.prototype, a = Number.prototype;
    var p = ("sunday monday tuesday wednesday thursday friday saturday").split(/\s/), o = ("january february march april may june july august september october november december").split(/\s/), n = ("Millisecond Second Minute Hour Day Week Month Year").split(/\s/), m;
    var l = function(i) {
        return function() {
            if (this._is) {
                this._is = false;
                return this.getDay() == i
            }
            return this.moveToDayOfWeek(i, this._orient)
        }
    };
    for (var f = 0; f < p.length; f++) {
        g[p[f]] = g[p[f].substring(0, 3)] = l(f)
    }
    var h = function(i) {
        return function() {
            if (this._is) {
                this._is = false;
                return this.getMonth() === i
            }
            return this.moveToMonth(i, this._orient)
        }
    };
    for (var d = 0; d < o.length; d++) {
        g[o[d]] = g[o[d].substring(0, 3)] = h(d)
    }
    var e = function(i) {
        return function() {
            if (i.substring(i.length - 1) != "s") {
                i += "s"
            }
            return this["add" + i](this._orient)
        }
    };
    var b = function(i) {
        return function() {
            this._dateElement = i;
            return this
        }
    };
    for (var c = 0; c < n.length; c++) {
        m = n[c].toLowerCase();
        g[m] = g[m + "s"] = e(n[c]);
        a[m] = a[m + "s"] = b(m)
    }
}());
Date.prototype.toJSONString = function() {
    return this.toString("yyyy-MM-ddThh:mm:ssZ")
};
Date.prototype.toShortDateString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.shortDatePattern)
};
Date.prototype.toLongDateString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.longDatePattern)
};
Date.prototype.toShortTimeString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.shortTimePattern)
};
Date.prototype.toLongTimeString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.longTimePattern)
};
Date.prototype.getOrdinal = function() {
    switch (this.getDate()) {
    case 1:
    case 21:
    case 31:
        return "st";
    case 2:
    case 22:
        return "nd";
    case 3:
    case 23:
        return "rd";
    default:
        return "th"
    }
};
(function() {
    Date.Parsing = {
        Exception: function(i) {
            this.message = "Parse error at '" + i.substring(0, 10) + " ...'"
        }
    };
    var a = Date.Parsing;
    var c = a.Operators = {
        rtoken: function(i) {
            return function(j) {
                var k = j.match(i);
                if (k) {
                    return ([k[0], j.substring(k[0].length)])
                } else {
                    throw new a.Exception(j)
                }
            }
        },
        token: function(i) {
            return function(j) {
                return c.rtoken(new RegExp("^s*" + j + "s*"))(j)
            }
        },
        stoken: function(i) {
            return c.rtoken(new RegExp("^" + i))
        },
        until: function(i) {
            return function(j) {
                var k = [], m = null;
                while (j.length) {
                    try {
                        m = i.call(this, j)
                    } catch (l) {
                        k.push(m[0]);
                        j = m[1];
                        continue
                    }
                    break
                }
                return [k, j]
            }
        },
        many: function(i) {
            return function(j) {
                var m = [], k = null;
                while (j.length) {
                    try {
                        k = i.call(this, j)
                    } catch (l) {
                        return [m, j]
                    }
                    m.push(k[0]);
                    j = k[1]
                }
                return [m, j]
            }
        },
        optional: function(i) {
            return function(j) {
                var k = null;
                try {
                    k = i.call(this, j)
                } catch (l) {
                    return [null, j]
                }
                return [k[0], k[1]]
            }
        },
        not: function(i) {
            return function(j) {
                try {
                    i.call(this, j)
                } catch (k) {
                    return [null, j]
                }
                throw new a.Exception(j)
            }
        },
        ignore: function(i) {
            return i ? function(j) {
                var k = null;
                k = i.call(this, j);
                return [null, k[1]]
            } : null
        },
        product: function() {
            var k = arguments[0], l = Array.prototype.slice.call(arguments, 1), m = [];
            for (var j = 0; j < k.length; j++) {
                m.push(c.each(k[j], l))
            }
            return m
        },
        cache: function(k) {
            var i = {}, j = null;
            return function(l) {
                try {
                    j = i[l] = (i[l] || k.call(this, l))
                } catch (m) {
                    j = i[l] = m
                }
                if (j instanceof a.Exception) {
                    throw j
                } else {
                    return j
                }
            }
        },
        any: function() {
            var i = arguments;
            return function(k) {
                var l = null;
                for (var j = 0; j < i.length; j++) {
                    if (i[j] == null) {
                        continue
                    }
                    try {
                        l = (i[j].call(this, k))
                    } catch (m) {
                        l = null
                    }
                    if (l) {
                        return l
                    }
                }
                throw new a.Exception(k)
            }
        },
        each: function() {
            var i = arguments;
            return function(k) {
                var n = [], l = null;
                for (var j = 0; j < i.length; j++) {
                    if (i[j] == null) {
                        continue
                    }
                    try {
                        l = (i[j].call(this, k))
                    } catch (m) {
                        throw new a.Exception(k)
                    }
                    n.push(l[0]);
                    k = l[1]
                }
                return [n, k]
            }
        },
        all: function() {
            var j = arguments, i = i;
            return i.each(i.optional(j))
        },
        sequence: function(i, j, k) {
            j = j || c.rtoken(/^\s*/);
            k = k || null;
            if (i.length == 1) {
                return i[0]
            }
            return function(o) {
                var p = null, t = null;
                var v = [];
                for (var n = 0; n < i.length; n++) {
                    try {
                        p = i[n].call(this, o)
                    } catch (u) {
                        break
                    }
                    v.push(p[0]);
                    try {
                        t = j.call(this, p[1])
                    } catch (m) {
                        t = null;
                        break
                    }
                    o = t[1]
                }
                if (!p) {
                    throw new a.Exception(o)
                }
                if (t) {
                    throw new a.Exception(t[1])
                }
                if (k) {
                    try {
                        p = k.call(this, p[1])
                    } catch (l) {
                        throw new a.Exception(p[1])
                    }
                }
                return [v, (p ? p[1] : o)]
            }
        },
        between: function(j, k, i) {
            i = i || j;
            var l = c.each(c.ignore(j), k, c.ignore(i));
            return function(m) {
                var n = l.call(this, m);
                return [[n[0][0], r[0][2]], n[1]]
            }
        },
        list: function(i, j, k) {
            j = j || c.rtoken(/^\s*/);
            k = k || null;
            return (i instanceof Array ? c.each(c.product(i.slice(0, - 1), c.ignore(j)), i.slice( - 1), c.ignore(k)) : c.each(c.many(c.each(i, c.ignore(j))), px, c.ignore(k)))
        },
        set: function(i, j, k) {
            j = j || c.rtoken(/^\s*/);
            k = k || null;
            return function(B) {
                var l = null, n = null, m = null, o = null, t = [[], B], A = false;
                for (var v = 0; v < i.length; v++) {
                    m = null;
                    n = null;
                    l = null;
                    A = (i.length == 1);
                    try {
                        l = i[v].call(this, B)
                    } catch (y) {
                        continue
                    }
                    o = [[l[0]], l[1]];
                    if (l[1].length > 0&&!A) {
                        try {
                            m = j.call(this, l[1])
                        } catch (z) {
                            A = true
                        }
                    } else {
                        A = true
                    }
                    if (!A && m[1].length === 0) {
                        A = true
                    }
                    if (!A) {
                        var w = [];
                        for (var u = 0; u < i.length; u++) {
                            if (v != u) {
                                w.push(i[u])
                            }
                        }
                        n = c.set(w, j).call(this, m[1]);
                        if (n[0].length > 0) {
                            o[0] = o[0].concat(n[0]);
                            o[1] = n[1]
                        }
                    }
                    if (o[1].length < t[1].length) {
                        t = o
                    }
                    if (t[1].length === 0) {
                        break
                    }
                }
                if (t[0].length === 0) {
                    return t
                }
                if (k) {
                    try {
                        m = k.call(this, t[1])
                    } catch (x) {
                        throw new a.Exception(t[1])
                    }
                    t[1] = m[1]
                }
                return t
            }
        },
        forward: function(i, j) {
            return function(k) {
                return i[j].call(this, k)
            }
        },
        replace: function(j, i) {
            return function(k) {
                var l = j.call(this, k);
                return [i, l[1]]
            }
        },
        process: function(j, i) {
            return function(k) {
                var l = j.call(this, k);
                return [i.call(this, l[0]), l[1]]
            }
        },
        min: function(i, j) {
            return function(k) {
                var l = j.call(this, k);
                if (l[0].length < i) {
                    throw new a.Exception(k)
                }
                return l
            }
        }
    };
    var h = function(i) {
        return function() {
            var j = null, m = [];
            if (arguments.length > 1) {
                j = Array.prototype.slice.call(arguments)
            } else {
                if (arguments[0] instanceof Array) {
                    j = arguments[0]
                }
            }
            if (j) {
                for (var l = 0, k = j.shift(); l < k.length; l++) {
                    j.unshift(k[l]);
                    m.push(i.apply(null, j));
                    j.shift();
                    return m
                }
            } else {
                return i.apply(null, arguments)
            }
        }
    };
    var g = "optional not ignore cache".split(/\s/);
    for (var d = 0; d < g.length; d++) {
        c[g[d]] = h(c[g[d]])
    }
    var f = function(i) {
        return function() {
            if (arguments[0] instanceof Array) {
                return i.apply(null, arguments[0])
            } else {
                return i.apply(null, arguments)
            }
        }
    };
    var e = "each any all".split(/\s/);
    for (var b = 0; b < e.length; b++) {
        c[e[b]] = f(c[e[b]])
    }
}());
(function() {
    var f = function(j) {
        var k = [];
        for (var g = 0; g < j.length; g++) {
            if (j[g] instanceof Array) {
                k = k.concat(f(j[g]))
            } else {
                if (j[g]) {
                    k.push(j[g])
                }
            }
        }
        return k
    };
    Date.Grammar = {};
    Date.Translator = {
        hour: function(g) {
            return function() {
                this.hour = Number(g)
            }
        },
        minute: function(g) {
            return function() {
                this.minute = Number(g)
            }
        },
        second: function(g) {
            return function() {
                this.second = Number(g)
            }
        },
        meridian: function(g) {
            return function() {
                this.meridian = g.slice(0, 1).toLowerCase()
            }
        },
        timezone: function(g) {
            return function() {
                var j = g.replace(/[^\d\+\-]/g, "");
                if (j.length) {
                    this.timezoneOffset = Number(j)
                } else {
                    this.timezone = g.toLowerCase()
                }
            }
        },
        day: function(g) {
            var j = g[0];
            return function() {
                this.day = Number(j.match(/\d+/)[0])
            }
        },
        month: function(g) {
            return function() {
                this.month = ((g.length == 3) ? Date.getMonthNumberFromName(g) : (Number(g) - 1))
            }
        },
        year: function(g) {
            return function() {
                var j = Number(g);
                this.year = ((g.length > 2) ? j : (j + (((j + 2000) < Date.CultureInfo.twoDigitYearMax) ? 2000 : 1900)))
            }
        },
        rday: function(g) {
            return function() {
                switch (g) {
                case"yesterday":
                    this.days =- 1;
                    break;
                case"tomorrow":
                    this.days = 1;
                    break;
                case"today":
                    this.days = 0;
                    break;
                case"now":
                    this.days = 0;
                    this.now = true;
                    break
                }
            }
        },
        finishExact: function(g) {
            g = (g instanceof Array) ? g : [g];
            var j = new Date();
            this.year = j.getFullYear();
            this.month = j.getMonth();
            this.day = 1;
            this.hour = 0;
            this.minute = 0;
            this.second = 0;
            for (var k = 0; k < g.length; k++) {
                if (g[k]) {
                    g[k].call(this)
                }
            }
            this.hour = (this.meridian == "p" && this.hour < 13) ? this.hour + 12 : this.hour;
            if (this.day > Date.getDaysInMonth(this.year, this.month)) {
                throw new RangeError(this.day + " is not a valid value for days.")
            }
            var l = new Date(this.year, this.month, this.day, this.hour, this.minute, this.second);
            if (this.timezone) {
                l.set({
                    timezone: this.timezone
                })
            } else {
                if (this.timezoneOffset) {
                    l.set({
                        timezoneOffset: this.timezoneOffset
                    })
                }
            }
            return l
        },
        finish: function(g) {
            g = (g instanceof Array) ? f(g) : [g];
            if (g.length === 0) {
                return null
            }
            for (var m = 0; m < g.length; m++) {
                if (typeof g[m] == "function") {
                    g[m].call(this)
                }
            }
            if (this.now) {
                return new Date()
            }
            var j = Date.today();
            var p = null;
            var n=!!(this.days != null || this.orient || this.operator);
            if (n) {
                var o, l, k;
                k = ((this.orient == "past" || this.operator == "subtract")?-1 : 1);
                if (this.weekday) {
                    this.unit = "day";
                    o = (Date.getDayNumberFromName(this.weekday) - j.getDay());
                    l = 7;
                    this.days = o ? ((o + (k * l))%l) : (k * l)
                }
                if (this.month) {
                    this.unit = "month";
                    o = (this.month - j.getMonth());
                    l = 12;
                    this.months = o ? ((o + (k * l))%l) : (k * l);
                    this.month = null
                }
                if (!this.unit) {
                    this.unit = "day"
                }
                if (this[this.unit + "s"] == null || this.operator != null) {
                    if (!this.value) {
                        this.value = 1
                    }
                    if (this.unit == "week") {
                        this.unit = "day";
                        this.value = this.value * 7
                    }
                    this[this.unit + "s"] = this.value * k
                }
                return j.add(this)
            } else {
                if (this.meridian && this.hour) {
                    this.hour = (this.hour < 13 && this.meridian == "p") ? this.hour + 12 : this.hour
                }
                if (this.weekday&&!this.day) {
                    this.day = (j.addDays((Date.getDayNumberFromName(this.weekday) - j.getDay()))).getDate()
                }
                if (this.month&&!this.day) {
                    this.day = 1
                }
                return j.set(this)
            }
        }
    };
    var b = Date.Parsing.Operators, e = Date.Grammar, d = Date.Translator, i;
    e.datePartDelimiter = b.rtoken(/^([\s\-\.\,\/\x27]+)/);
    e.timePartDelimiter = b.stoken(":");
    e.whiteSpace = b.rtoken(/^\s*/);
    e.generalDelimiter = b.rtoken(/^(([\s\,]|at|on)+)/);
    var a = {};
    e.ctoken = function(m) {
        var l = a[m];
        if (!l) {
            var n = Date.CultureInfo.regexPatterns;
            var k = m.split(/\s+/), j = [];
            for (var g = 0; g < k.length; g++) {
                j.push(b.replace(b.rtoken(n[k[g]]), k[g]))
            }
            l = a[m] = b.any.apply(null, j)
        }
        return l
    };
    e.ctoken2 = function(g) {
        return b.rtoken(Date.CultureInfo.regexPatterns[g])
    };
    e.h = b.cache(b.process(b.rtoken(/^(0[0-9]|1[0-2]|[1-9])/), d.hour));
    e.hh = b.cache(b.process(b.rtoken(/^(0[0-9]|1[0-2])/), d.hour));
    e.H = b.cache(b.process(b.rtoken(/^([0-1][0-9]|2[0-3]|[0-9])/), d.hour));
    e.HH = b.cache(b.process(b.rtoken(/^([0-1][0-9]|2[0-3])/), d.hour));
    e.m = b.cache(b.process(b.rtoken(/^([0-5][0-9]|[0-9])/), d.minute));
    e.mm = b.cache(b.process(b.rtoken(/^[0-5][0-9]/), d.minute));
    e.s = b.cache(b.process(b.rtoken(/^([0-5][0-9]|[0-9])/), d.second));
    e.ss = b.cache(b.process(b.rtoken(/^[0-5][0-9]/), d.second));
    e.hms = b.cache(b.sequence([e.H, e.mm, e.ss], e.timePartDelimiter));
    e.t = b.cache(b.process(e.ctoken2("shortMeridian"), d.meridian));
    e.tt = b.cache(b.process(e.ctoken2("longMeridian"), d.meridian));
    e.z = b.cache(b.process(b.rtoken(/^(\+|\-)?\s*\d\d\d\d?/), d.timezone));
    e.zz = b.cache(b.process(b.rtoken(/^(\+|\-)\s*\d\d\d\d/), d.timezone));
    e.zzz = b.cache(b.process(e.ctoken2("timezone"), d.timezone));
    e.timeSuffix = b.each(b.ignore(e.whiteSpace), b.set([e.tt, e.zzz]));
    e.time = b.each(b.optional(b.ignore(b.stoken("T"))), e.hms, e.timeSuffix);
    e.d = b.cache(b.process(b.each(b.rtoken(/^([0-2]\d|3[0-1]|\d)/), b.optional(e.ctoken2("ordinalSuffix"))), d.day));
    e.dd = b.cache(b.process(b.each(b.rtoken(/^([0-2]\d|3[0-1])/), b.optional(e.ctoken2("ordinalSuffix"))), d.day));
    e.ddd = e.dddd = b.cache(b.process(e.ctoken("sun mon tue wed thu fri sat"), function(g) {
        return function() {
            this.weekday = g
        }
    }));
    e.M = b.cache(b.process(b.rtoken(/^(1[0-2]|0\d|\d)/), d.month));
    e.MM = b.cache(b.process(b.rtoken(/^(1[0-2]|0\d)/), d.month));
    e.MMM = e.MMMM = b.cache(b.process(e.ctoken("jan feb mar apr may jun jul aug sep oct nov dec"), d.month));
    e.y = b.cache(b.process(b.rtoken(/^(\d\d?)/), d.year));
    e.yy = b.cache(b.process(b.rtoken(/^(\d\d)/), d.year));
    e.yyy = b.cache(b.process(b.rtoken(/^(\d\d?\d?\d?)/), d.year));
    e.yyyy = b.cache(b.process(b.rtoken(/^(\d\d\d\d)/), d.year));
    i = function() {
        return b.each(b.any.apply(null, arguments), b.not(e.ctoken2("timeContext")))
    };
    e.day = i(e.d, e.dd);
    e.month = i(e.M, e.MMM);
    e.year = i(e.yyyy, e.yy);
    e.orientation = b.process(e.ctoken("past future"), function(g) {
        return function() {
            this.orient = g
        }
    });
    e.operator = b.process(e.ctoken("add subtract"), function(g) {
        return function() {
            this.operator = g
        }
    });
    e.rday = b.process(e.ctoken("yesterday tomorrow today now"), d.rday);
    e.unit = b.process(e.ctoken("minute hour day week month year"), function(g) {
        return function() {
            this.unit = g
        }
    });
    e.value = b.process(b.rtoken(/^\d\d?(st|nd|rd|th)?/), function(g) {
        return function() {
            this.value = g.replace(/\D/g, "")
        }
    });
    e.expression = b.set([e.rday, e.operator, e.value, e.unit, e.orientation, e.ddd, e.MMM]);
    i = function() {
        return b.set(arguments, e.datePartDelimiter)
    };
    e.mdy = i(e.ddd, e.month, e.day, e.year);
    e.ymd = i(e.ddd, e.year, e.month, e.day);
    e.dmy = i(e.ddd, e.day, e.month, e.year);
    e.date = function(g) {
        return ((e[Date.CultureInfo.dateElementOrder] || e.mdy).call(this, g))
    };
    e.format = b.process(b.many(b.any(b.process(b.rtoken(/^(dd?d?d?|MM?M?M?|yy?y?y?|hh?|HH?|mm?|ss?|tt?|zz?z?)/), function(g) {
        if (e[g]) {
            return e[g]
        } else {
            throw Date.Parsing.Exception(g)
        }
    }), b.process(b.rtoken(/^[^dMyhHmstz]+/), function(g) {
        return b.ignore(b.stoken(g))
    }))), function(g) {
        return b.process(b.each.apply(null, g), d.finishExact)
    });
    var h = {};
    var c = function(g) {
        return h[g] = (h[g] || e.format(g)[0])
    };
    e.formats = function(j) {
        if (j instanceof Array) {
            var k = [];
            for (var g = 0; g < j.length; g++) {
                k.push(c(j[g]))
            }
            return b.any.apply(null, k)
        } else {
            return c(j)
        }
    };
    e._formats = e.formats(["yyyy-MM-ddTHH:mm:ss", "ddd, MMM dd, yyyy H:mm:ss tt", "ddd MMM d yyyy HH:mm:ss zzz", "d"]);
    e._start = b.process(b.set([e.date, e.time, e.expression], e.generalDelimiter, e.whiteSpace), d.finish);
    e.start = function(g) {
        try {
            var j = e._formats.call({}, g);
            if (j[1].length === 0) {
                return j
            }
        } catch (k) {}
        return e._start.call({}, g)
    }
}());
Date._parse = Date.parse;
Date.parse = function(a) {
    var b = null;
    if (!a) {
        return null
    }
    try {
        b = Date.Grammar.start.call({}, a)
    } catch (c) {
        return null
    }
    return ((b[1].length === 0) ? b[0] : null)
};
Date.getParseFunction = function(b) {
    var a = Date.Grammar.formats(b);
    return function(c) {
        var d = null;
        try {
            d = a.call({}, c)
        } catch (f) {
            return null
        }
        return ((d[1].length === 0) ? d[0] : null)
    }
};
Date.parseExact = function(a, b) {
    return Date.getParseFunction(b)(a)
};
var TimeSpan = function(m, h, e, j, d) {
    var l = "days hours minutes seconds milliseconds".split(/\s+/);
    var c = function(i) {
        return function() {
            return this[i]
        }
    };
    var k = function(i) {
        return function(n) {
            this[i] = n;
            return this
        }
    };
    for (var g = 0; g < l.length; g++) {
        var b = l[g], a = b.slice(0, 1).toUpperCase() + b.slice(1);
        TimeSpan.prototype[b] = 0;
        TimeSpan.prototype["get" + a] = c(b);
        TimeSpan.prototype["set" + a] = k(b)
    }
    if (arguments.length == 4) {
        this.setDays(m);
        this.setHours(h);
        this.setMinutes(e);
        this.setSeconds(j)
    } else {
        if (arguments.length == 5) {
            this.setDays(m);
            this.setHours(h);
            this.setMinutes(e);
            this.setSeconds(j);
            this.setMilliseconds(d)
        } else {
            if (arguments.length == 1 && typeof m == "number") {
                var f = (m < 0)?-1 : + 1;
                this.setMilliseconds(Math.abs(m));
                this.setDays(Math.floor(this.getMilliseconds() / 86400000) * f);
                this.setMilliseconds(this.getMilliseconds()%86400000);
                this.setHours(Math.floor(this.getMilliseconds() / 3600000) * f);
                this.setMilliseconds(this.getMilliseconds()%3600000);
                this.setMinutes(Math.floor(this.getMilliseconds() / 60000) * f);
                this.setMilliseconds(this.getMilliseconds()%60000);
                this.setSeconds(Math.floor(this.getMilliseconds() / 1000) * f);
                this.setMilliseconds(this.getMilliseconds()%1000);
                this.setMilliseconds(this.getMilliseconds() * f)
            }
        }
    }
    this.getTotalMilliseconds = function() {
        return (this.getDays() * 86400000) + (this.getHours() * 3600000) + (this.getMinutes() * 60000) + (this.getSeconds() * 1000)
    };
    this.compareTo = function(o) {
        var n = new Date(1970, 1, 1, this.getHours(), this.getMinutes(), this.getSeconds()), i;
        if (o === null) {
            i = new Date(1970, 1, 1, 0, 0, 0)
        } else {
            i = new Date(1970, 1, 1, o.getHours(), o.getMinutes(), o.getSeconds())
        }
        return (n < i)?-1 : (n > i) ? 1 : 0
    };
    this.equals = function(i) {
        return (this.compareTo(i) === 0)
    };
    this.add = function(i) {
        return (i === null) ? this : this.addSeconds(i.getTotalMilliseconds() / 1000)
    };
    this.subtract = function(i) {
        return (i === null) ? this : this.addSeconds( - i.getTotalMilliseconds() / 1000)
    };
    this.addDays = function(i) {
        return new TimeSpan(this.getTotalMilliseconds() + (i * 86400000))
    };
    this.addHours = function(i) {
        return new TimeSpan(this.getTotalMilliseconds() + (i * 3600000))
    };
    this.addMinutes = function(i) {
        return new TimeSpan(this.getTotalMilliseconds() + (i * 60000))
    };
    this.addSeconds = function(i) {
        return new TimeSpan(this.getTotalMilliseconds() + (i * 1000))
    };
    this.addMilliseconds = function(i) {
        return new TimeSpan(this.getTotalMilliseconds() + i)
    };
    this.get12HourHour = function() {
        return (this.getHours() > 12) ? this.getHours() - 12 : (this.getHours() === 0) ? 12 : this.getHours()
    };
    this.getDesignator = function() {
        return (this.getHours() < 12) ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator
    };
    this.toString = function(n) {
        this._toString = function() {
            if (this.getDays() !== null && this.getDays() > 0) {
                return this.getDays() + "." + this.getHours() + ":" + this.p(this.getMinutes()) + ":" + this.p(this.getSeconds())
            } else {
                return this.getHours() + ":" + this.p(this.getMinutes()) + ":" + this.p(this.getSeconds())
            }
        };
        this.p = function(o) {
            return (o.toString().length < 2) ? "0" + o : o
        };
        var i = this;
        return n ? n.replace(/dd?|HH?|hh?|mm?|ss?|tt?/g, function(o) {
            switch (o) {
            case"d":
                return i.getDays();
            case"dd":
                return i.p(i.getDays());
            case"H":
                return i.getHours();
            case"HH":
                return i.p(i.getHours());
            case"h":
                return i.get12HourHour();
            case"hh":
                return i.p(i.get12HourHour());
            case"m":
                return i.getMinutes();
            case"mm":
                return i.p(i.getMinutes());
            case"s":
                return i.getSeconds();
            case"ss":
                return i.p(i.getSeconds());
            case"t":
                return ((i.getHours() < 12) ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator).substring(0, 1);
            case"tt":
                return (i.getHours() < 12) ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator
            }
        }) : this._toString()
    };
    return this
};
Date.prototype.getTimeOfDay = function() {
    return new TimeSpan(0, this.getHours(), this.getMinutes(), this.getSeconds(), this.getMilliseconds())
};
var TimePeriod = function(n, f, e, q, o, k, g) {
    var m = "years months days hours minutes seconds milliseconds".split(/\s+/);
    var j = function(i) {
        return function() {
            return this[i]
        }
    };
    var b = function(i) {
        return function(v) {
            this[i] = v;
            return this
        }
    };
    for (var p = 0; p < m.length; p++) {
        var d = m[p], c = d.slice(0, 1).toUpperCase() + d.slice(1);
        TimePeriod.prototype[d] = 0;
        TimePeriod.prototype["get" + c] = j(d);
        TimePeriod.prototype["set" + c] = b(d)
    }
    if (arguments.length == 7) {
        this.years = n;
        this.months = f;
        this.setDays(e);
        this.setHours(q);
        this.setMinutes(o);
        this.setSeconds(k);
        this.setMilliseconds(g)
    } else {
        if (arguments.length == 2 && arguments[0] instanceof Date && arguments[1] instanceof Date) {
            var u = n.clone();
            var t = f.clone();
            var s = u.clone();
            var h = (u > t)?-1 : + 1;
            this.years = t.getFullYear() - u.getFullYear();
            s.addYears(this.years);
            if (h ==+ 1) {
                if (s > t) {
                    if (this.years !== 0) {
                        this.years--
                    }
                }
            } else {
                if (s < t) {
                    if (this.years !== 0) {
                        this.years++
                    }
                }
            }
            u.addYears(this.years);
            if (h ==+ 1) {
                while (u < t && u.clone().addDays(Date.getDaysInMonth(u.getYear(), u.getMonth())) < t) {
                    u.addMonths(1);
                    this.months++
                }
            } else {
                while (u > t && u.clone().addDays( - u.getDaysInMonth()) > t) {
                    u.addMonths( - 1);
                    this.months--
                }
            }
            var l = t - u;
            if (l !== 0) {
                var a = new TimeSpan(l);
                this.setDays(a.getDays());
                this.setHours(a.getHours());
                this.setMinutes(a.getMinutes());
                this.setSeconds(a.getSeconds());
                this.setMilliseconds(a.getMilliseconds())
            }
        }
    }
    return this
};
(function($) {
    $.extend({
        tablesorter: new function() {
            var parsers = [], widgets = [];
            this.defaults = {
                cssHeader: "header",
                cssAsc: "headerSortUp",
                cssDesc: "headerSortDown",
                cssChildRow: "expand-child",
                sortInitialOrder: "asc",
                sortMultiSortKey: "shiftKey",
                sortForce: null,
                sortAppend: null,
                sortLocaleCompare: true,
                textExtraction: "simple",
                parsers: {},
                widgets: [],
                widgetZebra: {
                    css: ["even", "odd"]
                },
                headers: {},
                widthFixed: false,
                cancelSelection: true,
                sortList: [],
                headerList: [],
                dateFormat: "us",
                decimal: "/.|,/g",
                onRenderHeader: null,
                selectorHeaders: "thead th",
                debug: false
            };
            function benchmark(s, d) {
                log(s + "," + (new Date().getTime() - d.getTime()) + "ms")
            }
            this.benchmark = benchmark;
            function log(s) {
                if (typeof console != "undefined" && typeof console.debug != "undefined") {
                    console.log(s)
                } else {
                    alert(s)
                }
            }
            function buildParserCache(table, $headers) {
                if (table.config.debug) {
                    var parsersDebug = ""
                }
                if (table.tBodies.length == 0) {
                    return 
                }
                var rows = table.tBodies[0].rows;
                if (rows[0]) {
                    var list = [], cells = rows[0].cells, l = cells.length;
                    for (var i = 0; i < l; i++) {
                        var p = false;
                        if ($.metadata && ($($headers[i]).metadata() && $($headers[i]).metadata().sorter)) {
                            p = getParserById($($headers[i]).metadata().sorter)
                        } else {
                            if ((table.config.headers[i] && table.config.headers[i].sorter)) {
                                p = getParserById(table.config.headers[i].sorter)
                            }
                        }
                        if (!p) {
                            p = detectParserForColumn(table, rows, - 1, i)
                        }
                        if (table.config.debug) {
                            parsersDebug += "column:" + i + " parser:" + p.id + "\n"
                        }
                        list.push(p)
                    }
                }
                if (table.config.debug) {
                    log(parsersDebug)
                }
                return list
            }
            function detectParserForColumn(table, rows, rowIndex, cellIndex) {
                var l = parsers.length, node = false, nodeValue = false, keepLooking = true;
                while (nodeValue == "" && keepLooking) {
                    rowIndex++;
                    if (rows[rowIndex]) {
                        node = getNodeFromRowAndCellIndex(rows, rowIndex, cellIndex);
                        nodeValue = trimAndGetNodeText(table.config, node);
                        if (table.config.debug) {
                            log("Checking if value was empty on row:" + rowIndex)
                        }
                    } else {
                        keepLooking = false
                    }
                }
                for (var i = 1; i < l; i++) {
                    if (parsers[i].is(nodeValue, table, node)) {
                        return parsers[i]
                    }
                }
                return parsers[0]
            }
            function getNodeFromRowAndCellIndex(rows, rowIndex, cellIndex) {
                return rows[rowIndex].cells[cellIndex]
            }
            function trimAndGetNodeText(config, node) {
                return $.trim(getElementText(config, node))
            }
            function getParserById(name) {
                var l = parsers.length;
                for (var i = 0; i < l; i++) {
                    if (parsers[i].id.toLowerCase() == name.toLowerCase()) {
                        return parsers[i]
                    }
                }
                return false
            }
            function buildCache(table) {
                if (table.config.debug) {
                    var cacheTime = new Date()
                }
                var totalRows = (table.tBodies[0] && table.tBodies[0].rows.length) || 0, totalCells = (table.tBodies[0].rows[0] && table.tBodies[0].rows[0].cells.length) || 0, parsers = table.config.parsers, cache = {
                    row: [],
                    normalized: []
                };
                for (var i = 0; i < totalRows; ++i) {
                    var c = $(table.tBodies[0].rows[i]), cols = [];
                    if (c.hasClass(table.config.cssChildRow)) {
                        cache.row[cache.row.length - 1] = cache.row[cache.row.length - 1].add(c);
                        continue
                    }
                    cache.row.push(c);
                    for (var j = 0; j < totalCells; ++j) {
                        cols.push(parsers[j].format(getElementText(table.config, c[0].cells[j]), table, c[0].cells[j]))
                    }
                    cols.push(cache.normalized.length);
                    cache.normalized.push(cols);
                    cols = null
                }
                if (table.config.debug) {
                    benchmark("Building cache for " + totalRows + " rows:", cacheTime)
                }
                return cache
            }
            function getElementText(config, node) {
                var text = "";
                if (!node) {
                    return ""
                }
                if (!config.supportsTextContent) {
                    config.supportsTextContent = node.textContent || false
                }
                if (config.textExtraction == "simple") {
                    if (config.supportsTextContent) {
                        text = node.textContent
                    } else {
                        if (node.childNodes[0] && node.childNodes[0].hasChildNodes()) {
                            text = node.childNodes[0].innerHTML
                        } else {
                            text = node.innerHTML
                        }
                    }
                } else {
                    if (typeof(config.textExtraction) == "function") {
                        text = config.textExtraction(node)
                    } else {
                        text = $(node).text()
                    }
                }
                return text
            }
            function appendToTable(table, cache) {
                if (table.config.debug) {
                    var appendTime = new Date()
                }
                var c = cache, r = c.row, n = c.normalized, totalRows = n.length, checkCell = (n[0].length - 1), tableBody = $(table.tBodies[0]), rows = [];
                for (var i = 0; i < totalRows; i++) {
                    var pos = n[i][checkCell];
                    rows.push(r[pos]);
                    if (!table.config.appender) {
                        var l = r[pos].length;
                        for (var j = 0; j < l; j++) {
                            tableBody[0].appendChild(r[pos][j])
                        }
                    }
                }
                if (table.config.appender) {
                    table.config.appender(table, rows)
                }
                rows = null;
                if (table.config.debug) {
                    benchmark("Rebuilt table:", appendTime)
                }
                applyWidget(table);
                setTimeout(function() {
                    $(table).trigger("sortEnd")
                }, 0)
            }
            function buildHeaders(table) {
                if (table.config.debug) {
                    var time = new Date()
                }
                var meta = ($.metadata) ? true: false;
                var header_index = computeTableHeaderCellIndexes(table);
                $tableHeaders = $(table.config.selectorHeaders, table).each(function(index) {
                    this.column = header_index[this.parentNode.rowIndex + "-" + this.cellIndex];
                    this.order = formatSortingOrder(table.config.sortInitialOrder);
                    this.count = this.order;
                    if (checkHeaderMetadata(this) || checkHeaderOptions(table, index)) {
                        this.sortDisabled = true
                    }
                    if (checkHeaderOptionsSortingLocked(table, index)) {
                        this.order = this.lockedOrder = checkHeaderOptionsSortingLocked(table, index)
                    }
                    if (!this.sortDisabled) {
                        var $th = $(this).addClass(table.config.cssHeader);
                        if (table.config.onRenderHeader) {
                            table.config.onRenderHeader.apply($th)
                        }
                    }
                    table.config.headerList[index] = this
                });
                if (table.config.debug) {
                    benchmark("Built headers:", time);
                    log($tableHeaders)
                }
                return $tableHeaders
            }
            function computeTableHeaderCellIndexes(t) {
                var matrix = [];
                var lookup = {};
                var thead = t.getElementsByTagName("THEAD")[0];
                var trs = thead.getElementsByTagName("TR");
                for (var i = 0; i < trs.length; i++) {
                    var cells = trs[i].cells;
                    for (var j = 0; j < cells.length; j++) {
                        var c = cells[j];
                        var rowIndex = c.parentNode.rowIndex;
                        var cellId = rowIndex + "-" + c.cellIndex;
                        var rowSpan = c.rowSpan || 1;
                        var colSpan = c.colSpan || 1;
                        var firstAvailCol;
                        if (typeof(matrix[rowIndex]) == "undefined") {
                            matrix[rowIndex] = []
                        }
                        for (var k = 0; k < matrix[rowIndex].length + 1; k++) {
                            if (typeof(matrix[rowIndex][k]) == "undefined") {
                                firstAvailCol = k;
                                break
                            }
                        }
                        lookup[cellId] = firstAvailCol;
                        for (var k = rowIndex; k < rowIndex + rowSpan; k++) {
                            if (typeof(matrix[k]) == "undefined") {
                                matrix[k] = []
                            }
                            var matrixrow = matrix[k];
                            for (var l = firstAvailCol; l < firstAvailCol + colSpan; l++) {
                                matrixrow[l] = "x"
                            }
                        }
                    }
                }
                return lookup
            }
            function checkCellColSpan(table, rows, row) {
                var arr = [], r = table.tHead.rows, c = r[row].cells;
                for (var i = 0; i < c.length; i++) {
                    var cell = c[i];
                    if (cell.colSpan > 1) {
                        arr = arr.concat(checkCellColSpan(table, headerArr, row++))
                    } else {
                        if (table.tHead.length == 1 || (cell.rowSpan > 1 ||!r[row + 1])) {
                            arr.push(cell)
                        }
                    }
                }
                return arr
            }
            function checkHeaderMetadata(cell) {
                if (($.metadata) && ($(cell).metadata().sorter === false)) {
                    return true
                }
                return false
            }
            function checkHeaderOptions(table, i) {
                if ((table.config.headers[i]) && (table.config.headers[i].sorter === false)) {
                    return true
                }
                return false
            }
            function checkHeaderOptionsSortingLocked(table, i) {
                if ((table.config.headers[i]) && (table.config.headers[i].lockedOrder)) {
                    return table.config.headers[i].lockedOrder
                }
                return false
            }
            function applyWidget(table) {
                var c = table.config.widgets;
                var l = c.length;
                for (var i = 0; i < l; i++) {
                    getWidgetById(c[i]).format(table)
                }
            }
            function getWidgetById(name) {
                var l = widgets.length;
                for (var i = 0; i < l; i++) {
                    if (widgets[i].id.toLowerCase() == name.toLowerCase()) {
                        return widgets[i]
                    }
                }
            }
            function formatSortingOrder(v) {
                if (typeof(v) != "Number") {
                    return (v.toLowerCase() == "desc") ? 1 : 0
                } else {
                    return (v == 1) ? 1 : 0
                }
            }
            function isValueInArray(v, a) {
                var l = a.length;
                for (var i = 0; i < l; i++) {
                    if (a[i][0] == v) {
                        return true
                    }
                }
                return false
            }
            function setHeadersCss(table, $headers, list, css) {
                $headers.removeClass(css[0]).removeClass(css[1]);
                var h = [];
                $headers.each(function(offset) {
                    if (!this.sortDisabled) {
                        h[this.column] = $(this)
                    }
                });
                var l = list.length;
                for (var i = 0; i < l; i++) {
                    h[list[i][0]].addClass(css[list[i][1]])
                }
            }
            function fixColumnWidth(table, $headers) {
                var c = table.config;
                if (c.widthFixed) {
                    var colgroup = $("<colgroup>");
                    $("tr:first td", table.tBodies[0]).each(function() {
                        colgroup.append($("<col>").css("width", $(this).width()))
                    });
                    $(table).prepend(colgroup)
                }
            }
            function updateHeaderSortCount(table, sortList) {
                var c = table.config, l = sortList.length;
                for (var i = 0; i < l; i++) {
                    var s = sortList[i], o = c.headerList[s[0]];
                    o.count = s[1];
                    o.count++
                }
            }
            function multisort(table, sortList, cache) {
                if (table.config.debug) {
                    var sortTime = new Date()
                }
                var dynamicExp = "var sortWrapper = function(a,b) {", l = sortList.length;
                for (var i = 0; i < l; i++) {
                    var c = sortList[i][0];
                    var order = sortList[i][1];
                    var s = (table.config.parsers[c].type == "text") ? ((order == 0) ? makeSortFunction("text", "asc", c) : makeSortFunction("text", "desc", c)): ((order == 0) ? makeSortFunction("numeric", "asc", c) : makeSortFunction("numeric", "desc", c));
                    var e = "e" + i;
                    dynamicExp += "var " + e + " = " + s;
                    dynamicExp += "if(" + e + ") { return " + e + "; } ";
                    dynamicExp += "else { "
                }
                var orgOrderCol = cache.normalized[0].length - 1;
                dynamicExp += "return a[" + orgOrderCol + "]-b[" + orgOrderCol + "];";
                for (var i = 0; i < l; i++) {
                    dynamicExp += "}; "
                }
                dynamicExp += "return 0; ";
                dynamicExp += "}; ";
                if (table.config.debug) {
                    benchmark("Evaling expression:" + dynamicExp, new Date())
                }
                eval(dynamicExp);
                cache.normalized.sort(sortWrapper);
                if (table.config.debug) {
                    benchmark("Sorting on " + sortList.toString() + " and dir " + order + " time:", sortTime)
                }
                return cache
            }
            function makeSortFunction(type, direction, index) {
                var a = "a[" + index + "]", b = "b[" + index + "]";
                if (type == "text" && direction == "asc") {
                    return "(" + a + " == " + b + " ? 0 : (" + a + " === null ? Number.POSITIVE_INFINITY : (" + b + " === null ? Number.NEGATIVE_INFINITY : (" + a + " < " + b + ") ? -1 : 1 )));"
                } else {
                    if (type == "text" && direction == "desc") {
                        return "(" + a + " == " + b + " ? 0 : (" + a + " === null ? Number.POSITIVE_INFINITY : (" + b + " === null ? Number.NEGATIVE_INFINITY : (" + b + " < " + a + ") ? -1 : 1 )));"
                    } else {
                        if (type == "numeric" && direction == "asc") {
                            return "(" + a + " === null && " + b + " === null) ? 0 :(" + a + " === null ? Number.POSITIVE_INFINITY : (" + b + " === null ? Number.NEGATIVE_INFINITY : " + a + " - " + b + "));"
                        } else {
                            if (type == "numeric" && direction == "desc") {
                                return "(" + a + " === null && " + b + " === null) ? 0 :(" + a + " === null ? Number.POSITIVE_INFINITY : (" + b + " === null ? Number.NEGATIVE_INFINITY : " + b + " - " + a + "));"
                            }
                        }
                    }
                }
            }
            function makeSortText(i) {
                return "((a[" + i + "] < b[" + i + "]) ? -1 : ((a[" + i + "] > b[" + i + "]) ? 1 : 0));"
            }
            function makeSortTextDesc(i) {
                return "((b[" + i + "] < a[" + i + "]) ? -1 : ((b[" + i + "] > a[" + i + "]) ? 1 : 0));"
            }
            function makeSortNumeric(i) {
                return "a[" + i + "]-b[" + i + "];"
            }
            function makeSortNumericDesc(i) {
                return "b[" + i + "]-a[" + i + "];"
            }
            function sortText(a, b) {
                if (table.config.sortLocaleCompare) {
                    return a.localeCompare(b)
                }
                return ((a < b)?-1 : ((a > b) ? 1 : 0))
            }
            function sortTextDesc(a, b) {
                if (table.config.sortLocaleCompare) {
                    return b.localeCompare(a)
                }
                return ((b < a)?-1 : ((b > a) ? 1 : 0))
            }
            function sortNumeric(a, b) {
                return a - b
            }
            function sortNumericDesc(a, b) {
                return b - a
            }
            function getCachedSortType(parsers, i) {
                return parsers[i].type
            }
            this.construct = function(settings) {
                return this.each(function() {
                    if (!this.tHead ||!this.tBodies) {
                        return 
                    }
                    var $this, $document, $headers, cache, config, shiftDown = 0, sortOrder;
                    this.config = {};
                    config = $.extend(this.config, $.tablesorter.defaults, settings);
                    $this = $(this);
                    $.data(this, "tablesorter", config);
                    $headers = buildHeaders(this);
                    this.config.parsers = buildParserCache(this, $headers);
                    cache = buildCache(this);
                    var sortCSS = [config.cssDesc, config.cssAsc];
                    fixColumnWidth(this);
                    $headers.click(function(e) {
                        var totalRows = ($this[0].tBodies[0] && $this[0].tBodies[0].rows.length) || 0;
                        if (!this.sortDisabled && totalRows > 0) {
                            $this.trigger("sortStart");
                            var $cell = $(this);
                            var i = this.column;
                            this.order = this.count++%2;
                            if (this.lockedOrder) {
                                this.order = this.lockedOrder
                            }
                            if (!e[config.sortMultiSortKey]) {
                                config.sortList = [];
                                if (config.sortForce != null) {
                                    var a = config.sortForce;
                                    for (var j = 0; j < a.length; j++) {
                                        if (a[j][0] != i) {
                                            config.sortList.push(a[j])
                                        }
                                    }
                                }
                                config.sortList.push([i, this.order])
                            } else {
                                if (isValueInArray(i, config.sortList)) {
                                    for (var j = 0; j < config.sortList.length; j++) {
                                        var s = config.sortList[j], o = config.headerList[s[0]];
                                        if (s[0] == i) {
                                            o.count = s[1];
                                            o.count++;
                                            s[1] = o.count%2
                                        }
                                    }
                                } else {
                                    config.sortList.push([i, this.order])
                                }
                            }
                            setTimeout(function() {
                                setHeadersCss($this[0], $headers, config.sortList, sortCSS);
                                appendToTable($this[0], multisort($this[0], config.sortList, cache))
                            }, 1);
                            return false
                        }
                    }).mousedown(function() {
                        if (config.cancelSelection) {
                            this.onselectstart = function() {
                                return false
                            };
                            return false
                        }
                    });
                    $this.bind("update", function() {
                        var me = this;
                        setTimeout(function() {
                            me.config.parsers = buildParserCache(me, $headers);
                            cache = buildCache(me)
                        }, 1)
                    }).bind("updateCell", function(e, cell) {
                        var config = this.config;
                        var pos = [(cell.parentNode.rowIndex - 1), cell.cellIndex];
                        cache.normalized[pos[0]][pos[1]] = config.parsers[pos[1]].format(getElementText(config, cell), cell)
                    }).bind("sorton", function(e, list) {
                        $(this).trigger("sortStart");
                        config.sortList = list;
                        var sortList = config.sortList;
                        updateHeaderSortCount(this, sortList);
                        setHeadersCss(this, $headers, sortList, sortCSS);
                        appendToTable(this, multisort(this, sortList, cache))
                    }).bind("appendCache", function() {
                        appendToTable(this, cache)
                    }).bind("applyWidgetId", function(e, id) {
                        getWidgetById(id).format(this)
                    }).bind("applyWidgets", function() {
                        applyWidget(this)
                    });
                    if ($.metadata && ($(this).metadata() && $(this).metadata().sortlist)) {
                        config.sortList = $(this).metadata().sortlist
                    }
                    if (config.sortList.length > 0) {
                        $this.trigger("sorton", [config.sortList])
                    }
                    applyWidget(this)
                })
            };
            this.addParser = function(parser) {
                var l = parsers.length, a = true;
                for (var i = 0; i < l; i++) {
                    if (parsers[i].id.toLowerCase() == parser.id.toLowerCase()) {
                        a = false
                    }
                }
                if (a) {
                    parsers.push(parser)
                }
            };
            this.addWidget = function(widget) {
                widgets.push(widget)
            };
            this.formatFloat = function(s) {
                var i = parseFloat(s);
                return (isNaN(i)) ? 0 : i
            };
            this.formatInt = function(s) {
                var i = parseInt(s);
                return (isNaN(i)) ? 0 : i
            };
            this.isDigit = function(s, config) {
                return /^[-+]?\d*$/.test($.trim(s.replace(/[,.']/g, "")))
            };
            this.clearTableBody = function(table) {
                if ($.browser.msie) {
                    function empty() {
                        while (this.firstChild) {
                            this.removeChild(this.firstChild)
                        }
                    }
                    empty.apply(table.tBodies[0])
                } else {
                    table.tBodies[0].innerHTML = ""
                }
            }
        }
    });
    $.fn.extend({
        tablesorter: $.tablesorter.construct
    });
    var ts = $.tablesorter;
    ts.addParser({
        id: "text",
        is: function(s) {
            return true
        },
        format: function(s) {
            return $.trim(s.toLocaleLowerCase())
        },
        type: "text"
    });
    ts.addParser({
        id: "digit",
        is: function(s, table) {
            var c = table.config;
            return $.tablesorter.isDigit(s, c)
        },
        format: function(s) {
            return $.tablesorter.formatFloat(s)
        },
        type: "numeric"
    });
    ts.addParser({
        id: "currency",
        is: function(s) {
            return /^[£$€?.]/.test(s)
        },
        format: function(s) {
            return $.tablesorter.formatFloat(s.replace(new RegExp(/[£$€]/g), ""))
        },
        type: "numeric"
    });
    ts.addParser({
        id: "ipAddress",
        is: function(s) {
            return /^\d{2,3}[\.]\d{2,3}[\.]\d{2,3}[\.]\d{2,3}$/.test(s)
        },
        format: function(s) {
            var a = s.split("."), r = "", l = a.length;
            for (var i = 0; i < l; i++) {
                var item = a[i];
                if (item.length == 2) {
                    r += "0" + item
                } else {
                    r += item
                }
            }
            return $.tablesorter.formatFloat(r)
        },
        type: "numeric"
    });
    ts.addParser({
        id: "url",
        is: function(s) {
            return /^(https?|ftp|file):\/\/$/.test(s)
        },
        format: function(s) {
            return jQuery.trim(s.replace(new RegExp(/(https?|ftp|file):\/\//), ""))
        },
        type: "text"
    });
    ts.addParser({
        id: "isoDate",
        is: function(s) {
            return /^\d{4}[\/-]\d{1,2}[\/-]\d{1,2}$/.test(s)
        },
        format: function(s) {
            return $.tablesorter.formatFloat((s != "") ? new Date(s.replace(new RegExp(/-/g), "/")).getTime() : "0")
        },
        type: "numeric"
    });
    ts.addParser({
        id: "percent",
        is: function(s) {
            return /\%$/.test($.trim(s))
        },
        format: function(s) {
            return $.tablesorter.formatFloat(s.replace(new RegExp(/%/g), ""))
        },
        type: "numeric"
    });
    ts.addParser({
        id: "usLongDate",
        is: function(s) {
            return s.match(new RegExp(/^[A-Za-z]{3,10}\.? [0-9]{1,2}, ([0-9]{4}|'?[0-9]{2}) (([0-2]?[0-9]:[0-5][0-9])|([0-1]?[0-9]:[0-5][0-9]\s(AM|PM)))$/))
        },
        format: function(s) {
            return $.tablesorter.formatFloat(new Date(s).getTime())
        },
        type: "numeric"
    });
    ts.addParser({
        id: "shortDate",
        is: function(s) {
            return /\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}/.test(s)
        },
        format: function(s, table) {
            var c = table.config;
            s = s.replace(/\-/g, "/");
            if (c.dateFormat == "us") {
                s = s.replace(/(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})/, "$3/$1/$2")
            } else {
                if (c.dateFormat == "uk") {
                    s = s.replace(/(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})/, "$3/$2/$1")
                } else {
                    if (c.dateFormat == "dd/mm/yy" || c.dateFormat == "dd-mm-yy") {
                        s = s.replace(/(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})/, "$1/$2/$3")
                    }
                }
            }
            return $.tablesorter.formatFloat(new Date(s).getTime())
        },
        type: "numeric"
    });
    ts.addParser({
        id: "time",
        is: function(s) {
            return /^(([0-2]?[0-9]:[0-5][0-9])|([0-1]?[0-9]:[0-5][0-9]\s(am|pm)))$/.test(s)
        },
        format: function(s) {
            return $.tablesorter.formatFloat(new Date("2000/01/01 " + s).getTime())
        },
        type: "numeric"
    });
    ts.addParser({
        id: "metadata",
        is: function(s) {
            return false
        },
        format: function(s, table, cell) {
            var c = table.config, p = (!c.parserMetadataName) ? "sortValue": c.parserMetadataName;
            return $(cell).metadata()[p]
        },
        type: "numeric"
    });
    ts.addWidget({
        id: "zebra",
        format: function(table) {
            if (table.config.debug) {
                var time = new Date()
            }
            var $tr, row =- 1, odd;
            $("tr:visible", table.tBodies[0]).each(function(i) {
                $tr = $(this);
                if (!$tr.hasClass(table.config.cssChildRow)) {
                    row++
                }
                odd = (row%2 == 0);
                $tr.removeClass(table.config.widgetZebra.css[odd ? 0: 1]).addClass(table.config.widgetZebra.css[odd ? 1: 0])
            });
            if (table.config.debug) {
                $.tablesorter.benchmark("Applying Zebra widget", time)
            }
        }
    })
})(jQuery);
(function(a) {
    a.fn.droppy = function(b) {
        b = a.extend({
            speed: 250,
            delay: 500,
            className: "droppy"
        }, b || {});
        this.each(function() {
            var d = this, h = 1000;
            a(d).addClass(b.className);
            function g(j) {
                if (j.nodeName.toLowerCase() == "li") {
                    var i = a("> ul", j);
                    return i.length ? i[0] : null
                } else {
                    return j
                }
            }
            function e(i) {
                if (i.nodeName.toLowerCase() == "ul") {
                    return a(i).parents("li")[0]
                } else {
                    return i
                }
            }
            function f() {
                var i = g(this);
                if (!i) {
                    return 
                }
                a.data(i, "cancelHide", false);
                if (b.delay) {
                    setTimeout(function() {
                        if (!a.data(i, "cancelHide")) {
                            if (b.speed) {
                                a(i).slideUp(b.speed)
                            } else {
                                a(i).hide()
                            }
                        }
                    }, b.delay)
                } else {
                    if (b.speed) {
                        a(i).slideUp(b.speed)
                    } else {
                        a(i).hide()
                    }
                }
            }
            function c() {
                var j = g(this);
                if (!j) {
                    return 
                }
                a.data(j, "cancelHide", true);
                a(j).css({
                    zIndex: h++
                });
                if (b.speed) {
                    a(j).slideDown(b.speed)
                } else {
                    a(j).show()
                }
                if (this.nodeName.toLowerCase() == "ul") {
                    var i = e(this);
                    a(i).addClass("hover");
                    a("> a", i).addClass("hover")
                }
            }
            a("ul, li", this).hover(c, f);
            a("li:has(ul)").addClass("subnav");
            a("ul.b-hornav > li").mouseover(c);
            a("li", this).hover(function() {
                a(this).addClass("hover");
                a("> a", this).addClass("hover");
                a(this).css("z-index", "10000")
            }, function() {
                a(this).removeClass("hover");
                a("> a", this).removeClass("hover");
                a(this).css("z-index", "1000")
            })
        })
    }
})(jQuery);
(function(g) {
    function k(D, E, F) {
        var G = this;
        return this.on("click.pjax", D, function(I) {
            var H = g.extend({}, s(E, F));
            if (!H.container) {
                H.container = g(this).attr("data-pjax") || G
            }
            l(I, H)
        })
    }
    function l(I, E, F) {
        F = s(E, F);
        var H = I.currentTarget;
        if (H.tagName.toUpperCase() !== "A") {
            throw "$.fn.pjax or $.pjax.click requires an anchor element"
        }
        if (I.which > 1 || I.metaKey || I.ctrlKey || I.shiftKey || I.altKey) {
            return 
        }
        if (location.protocol !== H.protocol || location.hostname !== H.hostname) {
            return 
        }
        if (H.hash && H.href.replace(H.hash, "") === location.href.replace(location.hash, "")) {
            return 
        }
        if (H.href === location.href + "#") {
            return 
        }
        if (I.isDefaultPrevented()) {
            return 
        }
        var J = {
            url: H.href,
            container: g(H).attr("data-pjax"),
            target: H
        };
        var G = g.extend({}, J, F);
        var D = g.Event("pjax:click");
        g(H).trigger(D, [G]);
        if (!D.isDefaultPrevented()) {
            z(G);
            I.preventDefault();
            g(H).trigger("pjax:clicked", [G])
        }
    }
    function q(G, D, E) {
        E = s(D, E);
        var F = G.currentTarget;
        if (F.tagName.toUpperCase() !== "FORM") {
            throw "$.pjax.submit requires a form element"
        }
        var H = {
            type: F.method.toUpperCase(),
            url: F.action,
            data: g(F).serializeArray(),
            container: g(F).attr("data-pjax"),
            target: F
        };
        z(g.extend({}, H, E));
        G.preventDefault()
    }
    function z(D) {
        D = g.extend(true, {}, g.ajaxSettings, z.defaults, D);
        if (g.isFunction(D.url)) {
            D.url = D.url()
        }
        var I = D.target;
        var H = p(D.url).hash;
        var E = D.context = r(D.container);
        if (!D.data) {
            D.data = {}
        }
        D.data._pjax = E.selector;
        function G(M, K, L) {
            if (!L) {
                L = {}
            }
            L.relatedTarget = I;
            var N = g.Event(M, L);
            E.trigger(N, K);
            return !N.isDefaultPrevented()
        }
        var F;
        D.beforeSend = function(L, K) {
            if (K.type !== "GET") {
                K.timeout = 0
            }
            L.setRequestHeader("X-PJAX", "true");
            L.setRequestHeader("X-PJAX-Container", E.selector);
            if (!G("pjax:beforeSend", [L, K])) {
                return false
            }
            if (K.timeout > 0) {
                F = setTimeout(function() {
                    if (G("pjax:timeout", [L, D])) {
                        L.abort("timeout")
                    }
                }, K.timeout);
                K.timeout = 0
            }
            D.requestUrl = p(K.url).href
        };
        D.complete = function(K, L) {
            if (F) {
                clearTimeout(F)
            }
            G("pjax:complete", [K, L, D]);
            G("pjax:end", [K, D])
        };
        D.error = function(N, O, L) {
            var K = w("", N, D);
            var M = G("pjax:error", [N, O, L, D]);
            if (D.type == "GET" && O !== "abort" && M) {
                x(K.url)
            }
        };
        D.success = function(N, M, T) {
            var P = z.state;
            var S = (typeof g.pjax.defaults.version === "function") ? g.pjax.defaults.version(): g.pjax.defaults.version;
            var U = T.getResponseHeader("X-PJAX-Version");
            var L = w(N, T, D);
            if (S && U && S !== U) {
                x(L.url);
                return 
            }
            if (!L.contents) {
                x(L.url);
                return 
            }
            z.state = {
                id: D.id || m(),
                url: L.url,
                title: L.title,
                container: E.selector,
                fragment: D.fragment,
                timeout: D.timeout
            };
            if (D.push || D.replace) {
                window.history.replaceState(z.state, L.title, L.url)
            }
            try {
                document.activeElement.blur()
            } catch (R) {}
            if (L.title) {
                document.title = L.title
            }
            G("pjax:beforeReplace", [L.contents, D], {
                state: z.state,
                previousState: P
            });
            E.html(L.contents);
            var O = E.find("input[autofocus], textarea[autofocus]").last()[0];
            if (O && document.activeElement !== O) {
                O.focus()
            }
            a(L.scripts);
            if (typeof D.scrollTo === "number") {
                g(window).scrollTop(D.scrollTo)
            }
            if (H !== "") {
                var K = p(L.url);
                K.hash = H;
                z.state.url = K.href;
                window.history.replaceState(z.state, L.title, K.href);
                var Q = g(K.hash);
                if (Q.length) {
                    g(window).scrollTop(Q.offset().top)
                }
            }
            G("pjax:success", [N, M, T, D])
        };
        if (!z.state) {
            z.state = {
                id: m(),
                url: window.location.href,
                title: document.title,
                container: E.selector,
                fragment: D.fragment,
                timeout: D.timeout
            };
            window.history.replaceState(z.state, document.title)
        }
        var J = z.xhr;
        if (J && J.readyState < 4) {
            J.onreadystatechange = g.noop;
            J.abort()
        }
        z.options = D;
        var J = z.xhr = g.ajax(D);
        if (J.readyState > 0) {
            if (D.push&&!D.replace) {
                j(z.state.id, E.clone().contents());
                window.history.pushState(null, "", B(D.requestUrl))
            }
            G("pjax:start", [J, D]);
            G("pjax:send", [J, D])
        }
        return z.xhr
    }
    function v(D, E) {
        var F = {
            url: window.location.href,
            push: false,
            replace: true,
            scrollTo: false
        };
        return z(g.extend(F, s(D, E)))
    }
    function x(D) {
        window.history.replaceState(null, "", "#");
        window.location.replace(D)
    }
    var i = true;
    var C = window.location.href;
    var A = window.history.state;
    if (A && A.container) {
        z.state = A
    }
    if ("state" in window.history) {
        i = false
    }
    function b(E) {
        var J = z.state;
        var D = E.state;
        if (D && D.container) {
            if (i && C == D.url) {
                return 
            }
            if (z.state && z.state.id === D.id) {
                return 
            }
            var F = g(D.container);
            if (F.length) {
                var K, H = e[D.id];
                if (z.state) {
                    K = z.state.id < D.id ? "forward" : "back";
                    t(K, z.state.id, F.clone().contents())
                }
                var I = g.Event("pjax:popstate", {
                    state: D,
                    direction: K
                });
                F.trigger(I);
                var L = {
                    id: D.id,
                    url: D.url,
                    container: F,
                    push: false,
                    fragment: D.fragment,
                    timeout: D.timeout,
                    scrollTo: false
                };
                if (H) {
                    F.trigger("pjax:start", [null, L]);
                    z.state = D;
                    if (D.title) {
                        document.title = D.title
                    }
                    var G = g.Event("pjax:beforeReplace", {
                        state: D,
                        previousState: J
                    });
                    F.trigger(G, [H, L]);
                    F.html(H);
                    F.trigger("pjax:end", [null, L])
                } else {
                    z(L)
                }
                F[0].offsetHeight
            } else {
                x(location.href)
            }
        }
        i = false
    }
    function d(E) {
        var D = g.isFunction(E.url) ? E.url(): E.url, H = E.type ? E.type.toUpperCase(): "GET";
        var F = g("<form>", {
            method: H === "GET" ? "GET": "POST",
            action: D,
            style: "display:none"
        });
        if (H !== "GET" && H !== "POST") {
            F.append(g("<input>", {
                type: "hidden",
                name: "_method",
                value: H.toLowerCase()
            }))
        }
        var G = E.data;
        if (typeof G === "string") {
            g.each(G.split("&"), function(I, J) {
                var K = J.split("=");
                F.append(g("<input>", {
                    type: "hidden",
                    name: K[0],
                    value: K[1]
                }))
            })
        } else {
            if (typeof G === "object") {
                for (key in G) {
                    F.append(g("<input>", {
                        type: "hidden",
                        name: key,
                        value: G[key]
                    }))
                }
            }
        }
        g(document.body).append(F);
        F.submit()
    }
    function m() {
        return (new Date).getTime()
    }
    function B(D) {
        return D.replace(/\?_pjax=[^&]+&?/, "?").replace(/_pjax=[^&]+&?/, "").replace(/[\?&]$/, "")
    }
    function p(E) {
        var D = document.createElement("a");
        D.href = E;
        return D
    }
    function s(D, E) {
        if (D && E) {
            E.container = D
        } else {
            if (g.isPlainObject(D)) {
                E = D
            } else {
                E = {
                    container: D
                }
            }
        }
        if (E.container) {
            E.container = r(E.container)
        }
        return E
    }
    function r(D) {
        D = g(D);
        if (!D.length) {
            throw "no pjax container for " + D.selector
        } else {
            if (D.selector !== "" && D.context === document) {
                return D
            } else {
                if (D.attr("id")) {
                    return g("#" + D.attr("id"))
                } else {
                    throw "cant get selector for pjax container!"
                }
            }
        }
    }
    function n(E, D) {
        return E.filter(D).add(E.find(D))
    }
    function u(D) {
        return g.parseHTML(D, document, true)
    }
    function w(G, I, E) {
        var H = {};
        H.url = B(I.getResponseHeader("X-PJAX-URL") || E.requestUrl);
        if (/<html/i.test(G)) {
            var D = g(u(G.match(/<head[^>]*>([\s\S.]*)<\/head>/i)[0]));
            var F = g(u(G.match(/<body[^>]*>([\s\S.]*)<\/body>/i)[0]))
        } else {
            var D = F = g(u(G))
        }
        if (F.length === 0) {
            return H
        }
        H.title = n(D, "title").last().text();
        if (E.fragment) {
            if (E.fragment === "body") {
                var J = F
            } else {
                var J = n(F, E.fragment).first()
            }
            if (J.length) {
                H.contents = J.contents();
                if (!H.title) {
                    H.title = J.attr("title") || J.data("title")
                }
            }
        } else {
            if (!/<html/i.test(G)) {
                H.contents = F
            }
        }
        if (H.contents) {
            H.contents = H.contents.not(function() {
                return g(this).is("title")
            });
            H.contents.find("title").remove();
            H.scripts = n(H.contents, "script[src]").remove();
            H.contents = H.contents.not(H.scripts)
        }
        if (H.title) {
            H.title = g.trim(H.title)
        }
        return H
    }
    function a(D) {
        if (!D) {
            return 
        }
        var E = g("script[src]");
        D.each(function() {
            var G = this.src;
            var H = E.filter(function() {
                return this.src === G
            });
            if (H.length) {
                return 
            }
            var F = document.createElement("script");
            F.type = g(this).attr("type");
            F.src = g(this).attr("src");
            document.head.appendChild(F)
        })
    }
    var e = {};
    var f = [];
    var h = [];
    function j(E, D) {
        e[E] = D;
        h.push(E);
        while (f.length) {
            delete e[f.shift()]
        }
        while (h.length > z.defaults.maxCacheLength) {
            delete e[h.shift()]
        }
    }
    function t(F, H, E) {
        var G, D;
        e[H] = E;
        if (F === "forward") {
            G = h;
            D = f
        } else {
            G = f;
            D = h
        }
        G.push(H);
        if (H = D.pop()) {
            delete e[H]
        }
    }
    function y() {
        return g("meta").filter(function() {
            var D = g(this).attr("http-equiv");
            return D && D.toUpperCase() === "X-PJAX-VERSION"
        }).attr("content")
    }
    function o() {
        g.fn.pjax = k;
        g.pjax = z;
        g.pjax.enable = g.noop;
        g.pjax.disable = c;
        g.pjax.click = l;
        g.pjax.submit = q;
        g.pjax.reload = v;
        g.pjax.defaults = {
            timeout: 650,
            push: true,
            replace: false,
            type: "GET",
            dataType: "html",
            scrollTo: 0,
            maxCacheLength: 20,
            version: y
        };
        g(window).on("popstate.pjax", b)
    }
    function c() {
        g.fn.pjax = function() {
            return this
        };
        g.pjax = d;
        g.pjax.enable = o;
        g.pjax.disable = g.noop;
        g.pjax.click = g.noop;
        g.pjax.submit = g.noop;
        g.pjax.reload = function() {
            window.location.reload()
        };
        g(window).off("popstate.pjax", b)
    }
    if (g.inArray("state", g.event.props) < 0) {
        g.event.props.push("state")
    }
    g.support.pjax = window.history && window.history.pushState && window.history.replaceState&&!navigator.userAgent.match(/((iPod|iPhone|iPad).+\bOS\s+[1-4]|WebApps\/.+CFNetwork)/);
    g.support.pjax ? o() : c()
})(jQuery);
(function(c) {
    function b(d) {
        this.input = d;
        if (d.attr("type") == "password") {
            this.handlePassword()
        }
        c(d[0].form).submit(function() {
            if (d.hasClass("placeholder")) {
                d[0].value = ""
            }
        })
    }
    b.prototype = {
        show: function(f) {
            if (this.input[0].value === "" || (f && this.valueIsPlaceholder())) {
                if (this.isPassword) {
                    try {
                        this.input[0].setAttribute("type", "text")
                    } catch (d) {
                        this.input.before(this.fakePassword.show()).hide()
                    }
                }
                this.input[0].value = this.input.attr("placeholder");
                this.input.addClass("placeholder")
            }
        },
        hide: function() {
            if (this.valueIsPlaceholder() && this.input.hasClass("placeholder")) {
                if (this.isPassword) {
                    try {
                        this.input[0].setAttribute("type", "password")
                    } catch (d) {}
                    this.input.show();
                    this.input[0].focus()
                }
                this.input[0].value = "";
                this.input.removeClass("placeholder")
            }
        },
        valueIsPlaceholder: function() {
            return this.input[0].value == this.input.attr("placeholder")
        },
        handlePassword: function() {
            var d = this.input;
            d.attr("realType", "password");
            this.isPassword = true;
            if (c.browser.msie && d[0].outerHTML) {
                var e = d[0].outerHTML.replace(/type=(['"])?password\1/gi, "type=$1text$1");
                this.fakePassword = c(e).val(d.attr("placeholder")).addClass("placeholder").focus(function() {
                    d.trigger("focus");
                    c(this).hide()
                })
            }
        }
    };
    var a=!!("placeholder" in document.createElement("input"));
    c.fn.placeholder = function() {
        return a ? this : this.each(function() {
            var d = c(this);
            var e = new b(d);
            e.show(true);
            d.focus(function() {
                e.hide()
            });
            d.blur(function() {
                e.show(false)
            });
            if (c.browser.msie) {
                c(window).load(function() {
                    if (d.val()) {
                        d.removeClass("placeholder")
                    }
                    e.show(true)
                })
            }
        })
    }
})(jQuery);
(function(a) {
    a.fn.autofocus = function() {
        return (this.first().autofocus !== true) ? this.focus() : this
    }
})(jQuery);
(function(c) {
    var a = c.scrollTo = function(f, e, d) {
        c(window).scrollTo(f, e, d)
    };
    a.defaults = {
        axis: "xy",
        duration: parseFloat(c.fn.jquery) >= 1.3 ? 0: 1
    };
    a.window = function(d) {
        return c(window)._scrollable()
    };
    c.fn._scrollable = function() {
        return this.map(function() {
            var e = this, d=!e.nodeName || c.inArray(e.nodeName.toLowerCase(), ["iframe", "#document", "html", "body"])!=-1;
            if (!d) {
                return e
            }
            var f = (e.contentWindow || e).document || e.ownerDocument || e;
            return c.browser.safari || f.compatMode == "BackCompat" ? f.body : f.documentElement
        })
    };
    c.fn.scrollTo = function(f, e, d) {
        if (typeof e == "object") {
            d = e;
            e = 0
        }
        if (typeof d == "function") {
            d = {
                onAfter: d
            }
        }
        if (f == "max") {
            f = 9000000000
        }
        d = c.extend({}, a.defaults, d);
        e = e || d.speed || d.duration;
        d.queue = d.queue && d.axis.length > 1;
        if (d.queue) {
            e/=2
        }
        d.offset = b(d.offset);
        d.over = b(d.over);
        return this._scrollable().each(function() {
            var l = this, j = c(l), k = f, i, g = {}, m = j.is("html,body");
            switch (typeof k) {
            case"number":
            case"string":
                if (/^([+-]=)?\d+(\.\d+)?(px|%)?$/.test(k)) {
                    k = b(k);
                    break
                }
                k = c(k, this);
            case"object":
                if (k.is || k.style) {
                    i = (k = c(k)).offset()
                }
            }
            c.each(d.axis.split(""), function(q, r) {
                var s = r == "x" ? "Left": "Top", u = s.toLowerCase(), p = "scroll" + s, o = l[p], n = a.max(l, r);
                if (i) {
                    g[p] = i[u] + (m ? 0 : o - j.offset()[u]);
                    if (d.margin) {
                        g[p] -= parseInt(k.css("margin" + s)) || 0;
                        g[p] -= parseInt(k.css("border" + s + "Width")) || 0
                    }
                    g[p] += d.offset[u] || 0;
                    if (d.over[u]) {
                        g[p] += k[r == "x" ? "width": "height"]() * d.over[u]
                    }
                } else {
                    var t = k[u];
                    g[p] = t.slice && t.slice( - 1) == "%" ? parseFloat(t) / 100 * n : t
                }
                if (/^\d+$/.test(g[p])) {
                    g[p] = g[p] <= 0 ? 0 : Math.min(g[p], n)
                }
                if (!q && d.queue) {
                    if (o != g[p]) {
                        h(d.onAfterFirst)
                    }
                    delete g[p]
                }
            });
            h(d.onAfter);
            function h(n) {
                j.animate(g, e, d.easing, n && function() {
                    n.call(this, f, d)
                })
            }
        }).end()
    };
    a.max = function(j, i) {
        var h = i == "x" ? "Width": "Height", e = "scroll" + h;
        if (!c(j).is("html,body")) {
            return j[e] - c(j)[h.toLowerCase()]()
        }
        var g = "client" + h, f = j.ownerDocument.documentElement, d = j.ownerDocument.body;
        return Math.max(f[e], d[e]) - Math.min(f[g], d[g])
    };
    function b(d) {
        return typeof d == "object" ? d : {
            top: d,
            left: d
        }
    }
})(jQuery);
strings = {
    strConversion: {
        __repr: function(a) {
            switch (this.__getType(a)) {
            case"array":
            case"date":
            case"number":
                return a.toString();
            case"object":
                var b = [];
                for (x = 0; x < a.length; a++) {
                    b.push(a + ": " + this.__repr(a[x]))
                }
                return b.join(", ");
            case"string":
                return a;
            default:
                return a
            }
        },
        __getType: function(b) {
            if (!b ||!b.constructor) {
                return typeof(b)
            }
            var a = b.constructor.toString().match(/Array|Number|String|Object|Date/);
            return a && a[0].toLowerCase() || typeof(b)
        },
        __pad: function(f, a, c, b) {
            var d = c || " ";
            var e = f;
            if (a - f.length > 0) {
                e = new Array(Math.ceil(a / d.length)).join(d).substr(0, b=!b ? a : b == 1 ? 0 : Math.ceil(a / 2)) + f + d.substr(0, a - b)
            }
            return e
        },
        __getInput: function(a, c) {
            var d = a.getKey();
            switch (this.__getType(c)) {
            case"object":
                var e = d.split(".");
                var f = c;
                for (var b = 0; b < e.length; b++) {
                    f = f[e[b]]
                }
                if (typeof(f) != "undefined") {
                    if (strings.strConversion.__getType(f) == "array") {
                        return a.getFormat().match(/\.\*/) && f[1] || f
                    }
                    return f
                } else {}
                break;
            case"array":
                d = parseInt(d, 10);
                if (a.getFormat().match(/\.\*/) && typeof c[d + 1] != "undefined") {
                    return c[d + 1]
                } else {
                    if (typeof c[d] != "undefined") {
                        return c[d]
                    } else {
                        return d
                    }
                }
                break
            }
            return "{" + d + "}"
        },
        __formatToken: function(c, b) {
            var a = new Argument(c, b);
            return strings.strConversion[a.getFormat().slice( - 1)](this.__getInput(a, b), a)
        },
        d: function(b, a) {
            var d = parseInt(b, 10);
            var c = a.getPaddingLength();
            if (c) {
                return this.__pad(d.toString(), c, a.getPaddingString(), 0)
            } else {
                return d
            }
        },
        i: function(a, b) {
            return this.d(a, b)
        },
        o: function(b, a) {
            var c = b.toString(8);
            if (a.isAlternate()) {
                c = this.__pad(c, c.length + 1, "0", 0)
            }
            return this.__pad(c, a.getPaddingLength(), a.getPaddingString(), 0)
        },
        u: function(a, b) {
            return Math.abs(this.d(a, b))
        },
        x: function(b, a) {
            var c = parseInt(b, 10).toString(16);
            c = this.__pad(c, a.getPaddingLength(), a.getPaddingString(), 0);
            return a.isAlternate() ? "0x" + c : c
        },
        X: function(b, a) {
            return this.x(b, a).toUpperCase()
        },
        e: function(b, a) {
            return parseFloat(b, 10).toExponential(a.getPrecision())
        },
        E: function(b, a) {
            return this.e(b, a).toUpperCase()
        },
        f: function(b, a) {
            return this.__pad(parseFloat(b, 10).toFixed(a.getPrecision()), a.getPaddingLength(), a.getPaddingString(), 0)
        },
        F: function(a, b) {
            return this.f(a, b)
        },
        g: function(b, a) {
            var c = parseFloat(b, 10);
            return (c.toString().length > 6) ? Math.round(c.toExponential(a.getPrecision())) : c
        },
        G: function(a, b) {
            return this.g(a, b)
        },
        c: function(a, c) {
            var b = a.match(/\w|\d/);
            return b && b[0] || ""
        },
        r: function(a, b) {
            return this.__repr(a)
        },
        s: function(a, b) {
            return a.toString && a.toString() || "" + a
        }
    },
    format: function(h, g) {
        var c = 0;
        var a = 0;
        var f = false;
        var d = [];
        var b = "";
        var e = (h || "").split("");
        for (a = 0; a < e.length; a++) {
            if (e[a] == "{" && e[a + 1] != "{") {
                c = h.indexOf("}", a);
                b = e.slice(a + 1, c).join("");
                if (e[a - 1] != "{" && e[c + 1] != "}") {
                    var i = (typeof arguments[1] != "object") ? arguments2Array(arguments, 2): g || [];
                    d.push(strings.strConversion.__formatToken(b, i))
                } else {
                    d.push(b)
                }
            } else {
                if (a > c || d.length < 1) {
                    d.push(e[a])
                }
            }
        }
        return (d.length > 1) ? d.join("") : d[0]
    }
};
var Argument = function(a, b) {
    this.__arg = a;
    this.__args = b;
    this.__max_precision = parseFloat("1." + (new Array(32)).join("1"), 10).toString().length - 3;
    this.__def_precision = 6;
    this.getString = function() {
        return this.__arg
    };
    this.getKey = function() {
        return this.__arg.split(":")[0]
    };
    this.getFormat = function() {
        var c = this.getString().split(":");
        return (c && c[1]) ? c[1] : "s"
    };
    this.getPrecision = function() {
        var c = this.getFormat().match(/\.(\d+|\*)/g);
        if (!c) {
            return this.__def_precision
        } else {
            c = c[0].slice(1);
            if (c != "*") {
                return parseInt(c, 10)
            } else {
                if (strings.strConversion.__getType(this.__args) == "array") {
                    return this.__args[1] && this.__args[0] || this.__def_precision
                } else {
                    if (strings.strConversion.__getType(this.__args) == "object") {
                        return this.__args[this.getKey()] && this.__args[this.getKey()][0] || this.__def_precision
                    } else {
                        return this.__def_precision
                    }
                }
            }
        }
    };
    this.getPaddingLength = function() {
        var c = false;
        if (this.isAlternate()) {
            c = this.getString().match(/0?#0?(\d+)/);
            if (c && c[1]) {
                return parseInt(c[1], 10)
            }
        }
        c = this.getString().match(/(0|\.)(\d+|\*)/g);
        return c && parseInt(c[0].slice(1), 10) || 0
    };
    this.getPaddingString = function() {
        var c = "";
        if (this.isAlternate()) {
            c = " "
        }
        if (this.getFormat().match(/#0|0#|^0|\.\d+/)) {
            c = "0"
        }
        return c
    };
    this.getFlags = function() {
        var c = this.getString().matc(/^(0|\#|\-|\+|\s)+/);
        return c && c[0].split("") || []
    };
    this.isAlternate = function() {
        return !!this.getFormat().match(/^0?#/)
    }
};
var arguments2Array = function(b, a) {
    var c = [];
    for (l = b.length, x = (a || 0) - 1; x < l; x++) {
        c.push(b[x])
    }
    return c
}; /* fancyBox v2.1.5 fancyapps.com | fancyapps.com/fancybox/#license */
(function(A, o, h, n) {
    var i = h("html"), d = h(A), e = h(o), j = h.fancybox = function() {
        j.open.apply(this, arguments)
    }, l = navigator.userAgent.match(/msie/i), c = null, z = o.createTouch !== n, v = function(b) {
        return b && b.hasOwnProperty && b instanceof h
    }, a = function(b) {
        return b && "string" === h.type(b)
    }, y = function(b) {
        return a(b) && 0 < b.indexOf("%")
    }, g = function(b, m) {
        var f = parseInt(b, 10) || 0;
        m && y(b) && (f*=j.getViewport()[m] / 100);
        return Math.ceil(f)
    }, k = function(m, f) {
        return g(m, f) + "px"
    };
    h.extend(j, {
        version: "2.1.5",
        defaults: {
            padding: 15,
            margin: 20,
            width: 800,
            height: 600,
            minWidth: 100,
            minHeight: 100,
            maxWidth: 9999,
            maxHeight: 9999,
            pixelRatio: 1,
            autoSize: !0,
            autoHeight: !1,
            autoWidth: !1,
            autoResize: !0,
            autoCenter: !z,
            fitToView: !0,
            aspectRatio: !1,
            topRatio: 0.5,
            leftRatio: 0.5,
            scrolling: "auto",
            wrapCSS: "",
            arrows: !0,
            closeBtn: !0,
            closeClick: !1,
            nextClick: !1,
            mouseWheel: !0,
            autoPlay: !1,
            playSpeed: 3000,
            preload: 3,
            modal: !1,
            loop: !0,
            ajax: {
                dataType: "html",
                headers: {
                    "X-fancyBox": !0
                }
            },
            iframe: {
                scrolling: "auto",
                preload: !0
            },
            swf: {
                wmode: "transparent",
                allowfullscreen: "true",
                allowscriptaccess: "always"
            },
            keys: {
                next: {
                    13: "left",
                    34: "up",
                    39: "left",
                    40: "up"
                },
                prev: {
                    8: "right",
                    33: "down",
                    37: "right",
                    38: "down"
                },
                close: [27],
                play: [32],
                toggle: [70]
            },
            direction: {
                next: "left",
                prev: "right"
            },
            scrollOutside: !0,
            index: 0,
            type: null,
            href: null,
            content: null,
            title: null,
            tpl: {
                wrap: '<div class="fancybox-wrap" tabIndex="-1"><div class="fancybox-skin"><div class="fancybox-outer"><div class="fancybox-inner"></div></div></div></div>',
                image: '<img class="fancybox-image" src="{href}" alt="" />',
                iframe: '<iframe id="fancybox-frame{rnd}" name="fancybox-frame{rnd}" class="fancybox-iframe" frameborder="0" vspace="0" hspace="0" webkitAllowFullScreen mozallowfullscreen allowFullScreen' + (l ? ' allowtransparency="true"' : "") + "></iframe>",
                error: '<p class="fancybox-error">The requested content cannot be loaded.<br/>Please try again later.</p>',
                closeBtn: '<a title="Close" class="fancybox-item fancybox-close" href="javascript:;"></a>',
                next: '<a title="Next" class="fancybox-nav fancybox-next" href="javascript:;"><span></span></a>',
                prev: '<a title="Previous" class="fancybox-nav fancybox-prev" href="javascript:;"><span></span></a>'
            },
            openEffect: "fade",
            openSpeed: 250,
            openEasing: "swing",
            openOpacity: !0,
            openMethod: "zoomIn",
            closeEffect: "fade",
            closeSpeed: 250,
            closeEasing: "swing",
            closeOpacity: !0,
            closeMethod: "zoomOut",
            nextEffect: "elastic",
            nextSpeed: 250,
            nextEasing: "swing",
            nextMethod: "changeIn",
            prevEffect: "elastic",
            prevSpeed: 250,
            prevEasing: "swing",
            prevMethod: "changeOut",
            helpers: {
                overlay: !0,
                title: !0
            },
            onCancel: h.noop,
            beforeLoad: h.noop,
            afterLoad: h.noop,
            beforeShow: h.noop,
            afterShow: h.noop,
            beforeChange: h.noop,
            beforeClose: h.noop,
            afterClose: h.noop
        },
        group: {},
        opts: {},
        previous: null,
        coming: null,
        current: null,
        isActive: !1,
        isOpen: !1,
        isOpened: !1,
        wrap: null,
        skin: null,
        outer: null,
        inner: null,
        player: {
            timer: null,
            isActive: !1
        },
        ajaxLoad: null,
        imgPreload: null,
        transitions: {},
        helpers: {},
        open: function(b, f) {
            if (b && (h.isPlainObject(f) || (f = {}), !1 !== j.close(!0))) {
                return h.isArray(b) || (b = v(b) ? h(b).get() : [b]), h.each(b, function(u, x) {
                    var q = {}, t, s, r, w, p;
                    "object" === h.type(x) && (x.nodeType && (x = h(x)), v(x) ? (q = {
                        href: x.data("fancybox-href") || x.attr("href"),
                        title: h("<div/>").text(x.data("fancybox-title") || x.attr("title")).html(),
                        isDom: !0,
                        element: x
                    }, h.metadata && h.extend(!0, q, x.metadata())) : q = x);
                    t = f.href || q.href || (a(x) ? x : null);
                    s = f.title !== n ? f.title : q.title || "";
                    w = (r = f.content || q.content) ? "html" : f.type || q.type;
                    !w && q.isDom && (w = x.data("fancybox-type"), w || (w = (w = x.prop("class").match(/fancybox\.(\w+)/)) ? w[1] : null));
                    a(t) && (w || (j.isImage(t) ? w = "image" : j.isSWF(t) ? w = "swf" : "#" === t.charAt(0) ? w = "inline" : a(x) && (w = "html", r = x)), "ajax" === w && (p = t.split(/\s+/, 2), t = p.shift(), p = p.shift()));
                    r || ("inline" === w ? t ? r = h(a(t) ? t.replace(/.*(?=#[^\s]+$)/, "") : t) : q.isDom && (r = x) : "html" === w ? r = t : w || t ||!q.isDom || (w = "inline", r = x));
                    h.extend(q, {
                        href: t,
                        type: w,
                        content: r,
                        title: s,
                        selector: p
                    });
                    b[u] = q
                }), j.opts = h.extend(!0, {}, j.defaults, f), f.keys !== n && (j.opts.keys = f.keys ? h.extend({}, j.defaults.keys, f.keys) : !1), j.group = b, j._start(j.opts.index)
            }
        },
        cancel: function() {
            var b = j.coming;
            b&&!1 === j.trigger("onCancel") || (j.hideLoading(), b && (j.ajaxLoad && j.ajaxLoad.abort(), j.ajaxLoad = null, j.imgPreload && (j.imgPreload.onload = j.imgPreload.onerror = null), b.wrap && b.wrap.stop(!0, !0).trigger("onReset").remove(), j.coming = null, j.current || j._afterZoomOut(b)))
        },
        close: function(b) {
            j.cancel();
            !1 !== j.trigger("beforeClose") && (j.unbindEvents(), j.isActive && (j.isOpen&&!0 !== b ? (j.isOpen = j.isOpened=!1, j.isClosing=!0, h(".fancybox-item, .fancybox-nav").remove(), j.wrap.stop(!0, !0).removeClass("fancybox-opened"), j.transitions[j.current.closeMethod]()) : (h(".fancybox-wrap").stop(!0).trigger("onReset").remove(), j._afterZoomOut())))
        },
        play: function(b) {
            var m = function() {
                clearTimeout(j.player.timer)
            }, f = function() {
                m();
                j.current && j.player.isActive && (j.player.timer = setTimeout(j.next, j.current.playSpeed))
            }, p = function() {
                m();
                e.unbind(".player");
                j.player.isActive=!1;
                j.trigger("onPlayEnd")
            };
            !0 === b ||!j.player.isActive&&!1 !== b ? j.current && (j.current.loop || j.current.index < j.group.length - 1) && (j.player.isActive=!0, e.bind({
                "onCancel.player beforeClose.player": p,
                "onUpdate.player": f,
                "beforeLoad.player": m
            }), f(), j.trigger("onPlayStart")) : p()
        },
        next: function(b) {
            var f = j.current;
            f && (a(b) || (b = f.direction.next), j.jumpto(f.index + 1, b, "next"))
        },
        prev: function(b) {
            var f = j.current;
            f && (a(b) || (b = f.direction.prev), j.jumpto(f.index - 1, b, "prev"))
        },
        jumpto: function(b, m, f) {
            var p = j.current;
            p && (b = g(b), j.direction = m || p.direction[b >= p.index ? "next": "prev"], j.router = f || "jumpto", p.loop && (0 > b && (b = p.group.length + b%p.group.length), b%=p.group.length), p.group[b] !== n && (j.cancel(), j._start(b)))
        },
        reposition: function(f, p) {
            var m = j.current, q = m ? m.wrap: null, b;
            q && (b = j._getPosition(p), f && "scroll" === f.type ? (delete b.position, q.stop(!0, !0).animate(b, 200)) : (q.css(b), m.pos = h.extend({}, m.dim, b)))
        },
        update: function(b) {
            var m = b && b.originalEvent && b.originalEvent.type, f=!m || "orientationchange" === m;
            f && (clearTimeout(c), c = null);
            j.isOpen&&!c && (c = setTimeout(function() {
                var p = j.current;
                p&&!j.isClosing && (j.wrap.removeClass("fancybox-tmp"), (f || "load" === m || "resize" === m && p.autoResize) && j._setDimension(), "scroll" === m && p.canShrink || j.reposition(b), j.trigger("onUpdate"), c = null)
            }, f&&!z ? 0 : 300))
        },
        toggle: function(b) {
            j.isOpen && (j.current.fitToView = "boolean" === h.type(b) ? b : !j.current.fitToView, z && (j.wrap.removeAttr("style").addClass("fancybox-tmp"), j.trigger("onUpdate")), j.update())
        },
        hideLoading: function() {
            e.unbind(".loading");
            h("#fancybox-loading").remove()
        },
        showLoading: function() {
            var b, f;
            j.hideLoading();
            b = h('<div id="fancybox-loading"><div></div></div>').click(j.cancel).appendTo("body");
            e.bind("keydown.loading", function(m) {
                27 === (m.which || m.keyCode) && (m.preventDefault(), j.cancel())
            });
            j.defaults.fixed || (f = j.getViewport(), b.css({
                position: "absolute",
                top: 0.5 * f.h + f.y,
                left: 0.5 * f.w + f.x
            }));
            j.trigger("onLoading")
        },
        getViewport: function() {
            var b = j.current && j.current.locked ||!1, f = {
                x: d.scrollLeft(),
                y: d.scrollTop()
            };
            b && b.length ? (f.w = b[0].clientWidth, f.h = b[0].clientHeight) : (f.w = z && A.innerWidth ? A.innerWidth : d.width(), f.h = z && A.innerHeight ? A.innerHeight : d.height());
            return f
        },
        unbindEvents: function() {
            j.wrap && v(j.wrap) && j.wrap.unbind(".fb");
            e.unbind(".fb");
            d.unbind(".fb")
        },
        bindEvents: function() {
            var b = j.current, f;
            b && (d.bind("orientationchange.fb" + (z ? "" : " resize.fb") + (b.autoCenter&&!b.locked ? " scroll.fb" : ""), j.update), (f = b.keys) && e.bind("keydown.fb", function(p) {
                var q = p.which || p.keyCode, m = p.target || p.srcElement;
                if (27 === q && j.coming) {
                    return !1
                }
                p.ctrlKey || p.altKey || p.shiftKey || p.metaKey || m && (m.type || h(m).is("[contenteditable]")) || h.each(f, function(s, r) {
                    if (1 < b.group.length && r[q] !== n) {
                        return j[s](r[q]), p.preventDefault(), !1
                    }
                    if ( - 1 < h.inArray(q, r)) {
                        return j[s](), p.preventDefault(), !1
                    }
                })
            }), h.fn.mousewheel && b.mouseWheel && j.wrap.bind("mousewheel.fb", function(s, t, m, r) {
                for (var q = h(s.target || null), p=!1; q.length&&!(p || q.is(".fancybox-skin") || q.is(".fancybox-wrap"));) {
                    p = q[0]&&!(q[0].style.overflow && "hidden" === q[0].style.overflow) && (q[0].clientWidth && q[0].scrollWidth > q[0].clientWidth || q[0].clientHeight && q[0].scrollHeight > q[0].clientHeight), q = h(q).parent()
                }
                0 !== t&&!p && 1 < j.group.length&&!b.canShrink && (0 < r || 0 < m ? j.prev(0 < r ? "down" : "left") : (0 > r || 0 > m) && j.next(0 > r ? "up" : "right"), s.preventDefault())
            }))
        },
        trigger: function(b, m) {
            var f, p = m || j.coming || j.current;
            if (p) {
                h.isFunction(p[b]) && (f = p[b].apply(p, Array.prototype.slice.call(arguments, 1)));
                if (!1 === f) {
                    return !1
                }
                p.helpers && h.each(p.helpers, function(r, q) {
                    if (q && j.helpers[r] && h.isFunction(j.helpers[r][b])) {
                        j.helpers[r][b](h.extend(!0, {}, j.helpers[r].defaults, q), p)
                    }
                })
            }
            e.trigger(b)
        },
        isImage: function(b) {
            return a(b) && b.match(/(^data:image\/.*,)|(\.(jp(e|g|eg)|gif|png|bmp|webp|svg)((\?|#).*)?$)/i)
        },
        isSWF: function(b) {
            return a(b) && b.match(/\.(swf)((\?|#).*)?$/i)
        },
        _start: function(b) {
            var m = {}, f, p;
            b = g(b);
            f = j.group[b] || null;
            if (!f) {
                return !1
            }
            m = h.extend(!0, {}, j.opts, f);
            f = m.margin;
            p = m.padding;
            "number" === h.type(f) && (m.margin = [f, f, f, f]);
            "number" === h.type(p) && (m.padding = [p, p, p, p]);
            m.modal && h.extend(!0, m, {
                closeBtn: !1,
                closeClick: !1,
                nextClick: !1,
                arrows: !1,
                mouseWheel: !1,
                keys: null,
                helpers: {
                    overlay: {
                        closeClick: !1
                    }
                }
            });
            m.autoSize && (m.autoWidth = m.autoHeight=!0);
            "auto" === m.width && (m.autoWidth=!0);
            "auto" === m.height && (m.autoHeight=!0);
            m.group = j.group;
            m.index = b;
            j.coming = m;
            if (!1 === j.trigger("beforeLoad")) {
                j.coming = null
            } else {
                p = m.type;
                f = m.href;
                if (!p) {
                    return j.coming = null, j.current && j.router && "jumpto" !== j.router ? (j.current.index = b, j[j.router](j.direction)) : !1
                }
                j.isActive=!0;
                if ("image" === p || "swf" === p) {
                    m.autoHeight = m.autoWidth=!1, m.scrolling = "visible"
                }
                "image" === p && (m.aspectRatio=!0);
                "iframe" === p && z && (m.scrolling = "scroll");
                m.wrap = h(m.tpl.wrap).addClass("fancybox-" + (z ? "mobile" : "desktop") + " fancybox-type-" + p + " fancybox-tmp " + m.wrapCSS).appendTo(m.parent || "body");
                h.extend(m, {
                    skin: h(".fancybox-skin", m.wrap),
                    outer: h(".fancybox-outer", m.wrap),
                    inner: h(".fancybox-inner", m.wrap)
                });
                h.each(["Top", "Right", "Bottom", "Left"], function(r, q) {
                    m.skin.css("padding" + q, k(m.padding[r]))
                });
                j.trigger("onReady");
                if ("inline" === p || "html" === p) {
                    if (!m.content ||!m.content.length) {
                        return j._error("content")
                    }
                } else {
                    if (!f) {
                        return j._error("href")
                    }
                }
                "image" === p ? j._loadImage() : "ajax" === p ? j._loadAjax() : "iframe" === p ? j._loadIframe() : j._afterLoad()
            }
        },
        _error: function(b) {
            h.extend(j.coming, {
                type: "html",
                autoWidth: !0,
                autoHeight: !0,
                minWidth: 0,
                minHeight: 0,
                scrolling: "no",
                hasError: b,
                content: j.coming.tpl.error
            });
            j._afterLoad()
        },
        _loadImage: function() {
            var b = j.imgPreload = new Image;
            b.onload = function() {
                this.onload = this.onerror = null;
                j.coming.width = this.width / j.opts.pixelRatio;
                j.coming.height = this.height / j.opts.pixelRatio;
                j._afterLoad()
            };
            b.onerror = function() {
                this.onload = this.onerror = null;
                j._error("image")
            };
            b.src = j.coming.href;
            !0 !== b.complete && j.showLoading()
        },
        _loadAjax: function() {
            var b = j.coming;
            j.showLoading();
            j.ajaxLoad = h.ajax(h.extend({}, b.ajax, {
                url: b.href,
                error: function(f, m) {
                    j.coming && "abort" !== m ? j._error("ajax", f) : j.hideLoading()
                },
                success: function(m, f) {
                    "success" === f && (b.content = m, j._afterLoad())
                }
            }))
        },
        _loadIframe: function() {
            var b = j.coming, f = h(b.tpl.iframe.replace(/\{rnd\}/g, (new Date).getTime())).attr("scrolling", z ? "auto" : b.iframe.scrolling).attr("src", b.href);
            h(b.wrap).bind("onReset", function() {
                try {
                    h(this).find("iframe").hide().attr("src", "//about:blank").end().empty()
                } catch (m) {}
            });
            b.iframe.preload && (j.showLoading(), f.one("load", function() {
                h(this).data("ready", 1);
                z || h(this).bind("load.fb", j.update);
                h(this).parents(".fancybox-wrap").width("100%").removeClass("fancybox-tmp").show();
                j._afterLoad()
            }));
            b.content = f.appendTo(b.inner);
            b.iframe.preload || j._afterLoad()
        },
        _preloadImages: function() {
            var b = j.group, r = j.current, q = b.length, s = r.preload ? Math.min(r.preload, q - 1): 0, p, m;
            for (m = 1; m <= s; m += 1) {
                p = b[(r.index + m)%q], "image" === p.type && p.href && ((new Image).src = p.href)
            }
        },
        _afterLoad: function() {
            var f = j.coming, r = j.current, q, s, b, p, m;
            j.hideLoading();
            if (f&&!1 !== j.isActive) {
                if (!1 === j.trigger("afterLoad", f, r)) {
                    f.wrap.stop(!0).trigger("onReset").remove(), j.coming = null
                } else {
                    r && (j.trigger("beforeChange", r), r.wrap.stop(!0).removeClass("fancybox-opened").find(".fancybox-item, .fancybox-nav").remove());
                    j.unbindEvents();
                    q = f.content;
                    s = f.type;
                    b = f.scrolling;
                    h.extend(j, {
                        wrap: f.wrap,
                        skin: f.skin,
                        outer: f.outer,
                        inner: f.inner,
                        current: f,
                        previous: r
                    });
                    p = f.href;
                    switch (s) {
                    case"inline":
                    case"ajax":
                    case"html":
                        f.selector ? q = h("<div>").html(q).find(f.selector) : v(q) && (q.data("fancybox-placeholder") || q.data("fancybox-placeholder", h('<div class="fancybox-placeholder"></div>').insertAfter(q).hide()), q = q.show().detach(), f.wrap.bind("onReset", function() {
                            h(this).find(q).length && q.hide().replaceAll(q.data("fancybox-placeholder")).data("fancybox-placeholder", !1)
                        }));
                        break;
                    case"image":
                        q = f.tpl.image.replace(/\{href\}/g, p);
                        break;
                    case"swf":
                        q = '<object id="fancybox-swf" classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" width="100%" height="100%"><param name="movie" value="' + p + '"></param>', m = "", h.each(f.swf, function(u, t) {
                            q += '<param name="' + u + '" value="' + t + '"></param>';
                            m += " " + u + '="' + t + '"'
                        }), q += '<embed src="' + p + '" type="application/x-shockwave-flash" width="100%" height="100%"' + m + "></embed></object>"
                    }
                    v(q) && q.parent().is(f.inner) || f.inner.append(q);
                    j.trigger("beforeShow");
                    f.inner.css("overflow", "yes" === b ? "scroll" : "no" === b ? "hidden" : b);
                    j._setDimension();
                    j.reposition();
                    j.isOpen=!1;
                    j.coming = null;
                    j.bindEvents();
                    if (!j.isOpened) {
                        h(".fancybox-wrap").not(f.wrap).stop(!0).trigger("onReset").remove()
                    } else {
                        if (r.prevMethod) {
                            j.transitions[r.prevMethod]()
                        }
                    }
                    j.transitions[j.isOpened ? f.nextMethod: f.openMethod]();
                    j._preloadImages()
                }
            }
        },
        _setDimension: function() {
            var ae = j.getViewport(), ac = 0, ab=!1, ad=!1, ab = j.wrap, X = j.skin, aa = j.inner, Z = j.current, ad = Z.width, Y = Z.height, W = Z.minWidth, K = Z.minHeight, V = Z.maxWidth, T = Z.maxHeight, N = Z.scrolling, R = Z.scrollOutside ? Z.scrollbarWidth : 0, f = Z.margin, b = g(f[1] + f[3]), P = g(f[0] + f[2]), F, U, L, O, S, J, Q, M, m;
            ab.add(X).add(aa).width("auto").height("auto").removeClass("fancybox-tmp");
            f = g(X.outerWidth(!0) - X.width());
            F = g(X.outerHeight(!0) - X.height());
            U = b + f;
            L = P + F;
            O = y(ad) ? (ae.w - U) * g(ad) / 100 : ad;
            S = y(Y) ? (ae.h - L) * g(Y) / 100 : Y;
            if ("iframe" === Z.type) {
                if (m = Z.content, Z.autoHeight && 1 === m.data("ready")) {
                    try {
                        m[0].contentWindow.document.location && (aa.width(O).height(9999), J = m.contents().find("body"), R && J.css("overflow-x", "hidden"), S = J.outerHeight(!0))
                    } catch (x) {}
                }
            } else {
                if (Z.autoWidth || Z.autoHeight) {
                    aa.addClass("fancybox-tmp"), Z.autoWidth || aa.width(O), Z.autoHeight || aa.height(S), Z.autoWidth && (O = aa.width()), Z.autoHeight && (S = aa.height()), aa.removeClass("fancybox-tmp")
                }
            }
            ad = g(O);
            Y = g(S);
            M = O / S;
            W = g(y(W) ? g(W, "w") - U : W);
            V = g(y(V) ? g(V, "w") - U : V);
            K = g(y(K) ? g(K, "h") - L : K);
            T = g(y(T) ? g(T, "h") - L : T);
            J = V;
            Q = T;
            Z.fitToView && (V = Math.min(ae.w - U, V), T = Math.min(ae.h - L, T));
            U = ae.w - b;
            P = ae.h - P;
            Z.aspectRatio ? (ad > V && (ad = V, Y = g(ad / M)), Y > T && (Y = T, ad = g(Y * M)), ad < W && (ad = W, Y = g(ad / M)), Y < K && (Y = K, ad = g(Y * M))) : (ad = Math.max(W, Math.min(ad, V)), Z.autoHeight && "iframe" !== Z.type && (aa.width(ad), Y = aa.height()), Y = Math.max(K, Math.min(Y, T)));
            if (Z.fitToView) {
                if (aa.width(ad).height(Y), ab.width(ad + f), ae = ab.width(), b = ab.height(), Z.aspectRatio) {
                    for (; (ae > U || b > P) && ad > W && Y > K&&!(19 < ac++);) {
                        Y = Math.max(K, Math.min(T, Y - 10)), ad = g(Y * M), ad < W && (ad = W, Y = g(ad / M)), ad > V && (ad = V, Y = g(ad / M)), aa.width(ad).height(Y), ab.width(ad + f), ae = ab.width(), b = ab.height()
                    }
                } else {
                    ad = Math.max(W, Math.min(ad, ad - (ae - U))), Y = Math.max(K, Math.min(Y, Y - (b - P)))
                }
            }
            R && "auto" === N && Y < S && ad + f + R < U && (ad += R);
            aa.width(ad).height(Y);
            ab.width(ad + f);
            ae = ab.width();
            b = ab.height();
            ab = (ae > U || b > P) && ad > W && Y > K;
            ad = Z.aspectRatio ? ad < J && Y < Q && ad < O && Y < S : (ad < J || Y < Q) && (ad < O || Y < S);
            h.extend(Z, {
                dim: {
                    width: k(ae),
                    height: k(b)
                },
                origWidth: O,
                origHeight: S,
                canShrink: ab,
                canExpand: ad,
                wPadding: f,
                hPadding: F,
                wrapSpace: b - X.outerHeight(!0),
                skinSpace: X.height() - Y
            });
            !m && Z.autoHeight && Y > K && Y < T&&!ad && aa.height("auto")
        },
        _getPosition: function(b) {
            var r = j.current, q = j.getViewport(), s = r.margin, p = j.wrap.width() + s[1] + s[3], m = j.wrap.height() + s[0] + s[2], s = {
                position: "absolute",
                top: s[0],
                left: s[3]
            };
            r.autoCenter && r.fixed&&!b && m <= q.h && p <= q.w ? s.position = "fixed" : r.locked || (s.top += q.y, s.left += q.x);
            s.top = k(Math.max(s.top, s.top + (q.h - m) * r.topRatio));
            s.left = k(Math.max(s.left, s.left + (q.w - p) * r.leftRatio));
            return s
        },
        _afterZoomIn: function() {
            var b = j.current;
            b && ((j.isOpen = j.isOpened=!0, j.wrap.css("overflow", "visible").addClass("fancybox-opened"), j.update(), (b.closeClick || b.nextClick && 1 < j.group.length) && j.inner.css("cursor", "pointer").bind("click.fb", function(f) {
                h(f.target).is("a") || h(f.target).parent().is("a") || (f.preventDefault(), j[b.closeClick ? "close": "next"]())
            }), b.closeBtn && h(b.tpl.closeBtn).appendTo(j.skin).bind("click.fb", function(f) {
                f.preventDefault();
                j.close()
            }), b.arrows && 1 < j.group.length && ((b.loop || 0 < b.index) && h(b.tpl.prev).appendTo(j.outer).bind("click.fb", j.prev), (b.loop || b.index < j.group.length - 1) && h(b.tpl.next).appendTo(j.outer).bind("click.fb", j.next)), j.trigger("afterShow"), b.loop || b.index !== b.group.length - 1) ? j.opts.autoPlay&&!j.player.isActive && (j.opts.autoPlay=!1, j.play(!0)) : j.play(!1))
        },
        _afterZoomOut: function(b) {
            b = b || j.current;
            h(".fancybox-wrap").trigger("onReset").remove();
            h.extend(j, {
                group: {},
                opts: {},
                router: !1,
                current: null,
                isActive: !1,
                isOpened: !1,
                isOpen: !1,
                isClosing: !1,
                wrap: null,
                skin: null,
                outer: null,
                inner: null
            });
            j.trigger("afterClose", b)
        }
    });
    j.transitions = {
        getOrigPosition: function() {
            var w = j.current, t = w.element, s = w.orig, u = {}, r = 50, q = 50, p = w.hPadding, m = w.wPadding, b = j.getViewport();
            !s && w.isDom && t.is(":visible") && (s = t.find("img:first"), s.length || (s = t));
            v(s) ? (u = s.offset(), s.is("img") && (r = s.outerWidth(), q = s.outerHeight())) : (u.top = b.y + (b.h - q) * w.topRatio, u.left = b.x + (b.w - r) * w.leftRatio);
            if ("fixed" === j.wrap.css("position") || w.locked) {
                u.top -= b.y, u.left -= b.x
            }
            return u = {
                top: k(u.top - p * w.topRatio),
                left: k(u.left - m * w.leftRatio),
                width: k(r + m),
                height: k(q + p)
            }
        },
        step: function(b, s) {
            var r, t, q = s.prop;
            t = j.current;
            var p = t.wrapSpace, m = t.skinSpace;
            if ("width" === q || "height" === q) {
                r = s.end === s.start ? 1 : (b - s.start) / (s.end - s.start), j.isClosing && (r = 1 - r), t = "width" === q ? t.wPadding : t.hPadding, t = b - t, j.skin[q](g("width" === q ? t : t - p * r)), j.inner[q](g("width" === q ? t : t - p * r - m * r))
            }
        },
        zoomIn: function() {
            var f = j.current, p = f.pos, m = f.openEffect, q = "elastic" === m, b = h.extend({
                opacity: 1
            }, p);
            delete b.position;
            q ? (p = this.getOrigPosition(), f.openOpacity && (p.opacity = 0.1)) : "fade" === m && (p.opacity = 0.1);
            j.wrap.css(p).animate(b, {
                duration: "none" === m ? 0: f.openSpeed,
                easing: f.openEasing,
                step: q ? this.step: null,
                complete: j._afterZoomIn
            })
        },
        zoomOut: function() {
            var b = j.current, m = b.closeEffect, f = "elastic" === m, p = {
                opacity: 0.1
            };
            f && (p = this.getOrigPosition(), b.closeOpacity && (p.opacity = 0.1));
            j.wrap.animate(p, {
                duration: "none" === m ? 0: b.closeSpeed,
                easing: b.closeEasing,
                step: f ? this.step: null,
                complete: j._afterZoomOut
            })
        },
        changeIn: function() {
            var b = j.current, r = b.nextEffect, q = b.pos, s = {
                opacity: 1
            }, p = j.direction, m;
            q.opacity = 0.1;
            "elastic" === r && (m = "down" === p || "up" === p ? "top" : "left", "down" === p || "right" === p ? (q[m] = k(g(q[m]) - 200), s[m] = "+=200px") : (q[m] = k(g(q[m]) + 200), s[m] = "-=200px"));
            "none" === r ? j._afterZoomIn() : j.wrap.css(q).animate(s, {
                duration: b.nextSpeed,
                easing: b.nextEasing,
                complete: j._afterZoomIn
            })
        },
        changeOut: function() {
            var b = j.previous, m = b.prevEffect, f = {
                opacity: 0.1
            }, p = j.direction;
            "elastic" === m && (f["down" === p || "up" === p ? "top": "left"] = ("up" === p || "left" === p ? "-" : "+") + "=200px");
            b.wrap.animate(f, {
                duration: "none" === m ? 0: b.prevSpeed,
                easing: b.prevEasing,
                complete: function() {
                    h(this).trigger("onReset").remove()
                }
            })
        }
    };
    j.helpers.overlay = {
        defaults: {
            closeClick: !0,
            speedOut: 200,
            showEarly: !0,
            css: {},
            locked: !z,
            fixed: !0
        },
        overlay: null,
        fixed: !1,
        el: h("html"),
        create: function(b) {
            var f;
            b = h.extend({}, this.defaults, b);
            this.overlay && this.close();
            f = j.coming ? j.coming.parent : b.parent;
            this.overlay = h('<div class="fancybox-overlay"></div>').appendTo(f && f.lenth ? f : "body");
            this.fixed=!1;
            b.fixed && j.defaults.fixed && (this.overlay.addClass("fancybox-overlay-fixed"), this.fixed=!0)
        },
        open: function(b) {
            var f = this;
            b = h.extend({}, this.defaults, b);
            this.overlay ? this.overlay.unbind(".overlay").width("auto").height("auto") : this.create(b);
            this.fixed || (d.bind("resize.overlay", h.proxy(this.update, this)), this.update());
            b.closeClick && this.overlay.bind("click.overlay", function(m) {
                if (h(m.target).hasClass("fancybox-overlay")) {
                    return j.isActive ? j.close() : f.close(), !1
                }
            });
            this.overlay.css(b.css).show()
        },
        close: function() {
            d.unbind("resize.overlay");
            this.el.hasClass("fancybox-lock") && (h(".fancybox-margin").removeClass("fancybox-margin"), this.el.removeClass("fancybox-lock"), d.scrollTop(this.scrollV).scrollLeft(this.scrollH));
            h(".fancybox-overlay").remove().hide();
            h.extend(this, {
                overlay: null,
                fixed: !1
            })
        },
        update: function() {
            var m = "100%", f;
            this.overlay.width(m).height("100%");
            l ? (f = Math.max(o.documentElement.offsetWidth, o.body.offsetWidth), e.width() > f && (m = e.width())) : e.width() > d.width() && (m = e.width());
            this.overlay.width(m).height(e.height())
        },
        onReady: function(m, f) {
            var p = this.overlay;
            h(".fancybox-overlay").stop(!0, !0);
            p || this.create(m);
            m.locked && this.fixed && f.fixed && (f.locked = this.overlay.append(f.wrap), f.fixed=!1);
            !0 === m.showEarly && this.beforeShow.apply(this, arguments)
        },
        beforeShow: function(m, f) {
            f.locked&&!this.el.hasClass("fancybox-lock") && (!1 !== this.fixPosition && h("*").filter(function() {
                return "fixed" === h(this).css("position")&&!h(this).hasClass("fancybox-overlay")&&!h(this).hasClass("fancybox-wrap")
            }).addClass("fancybox-margin"), this.el.addClass("fancybox-margin"), this.scrollV = d.scrollTop(), this.scrollH = d.scrollLeft(), this.el.addClass("fancybox-lock"), d.scrollTop(this.scrollV).scrollLeft(this.scrollH));
            this.open(m)
        },
        onUpdate: function() {
            this.fixed || this.update()
        },
        afterClose: function(b) {
            this.overlay&&!j.coming && this.overlay.fadeOut(b.speedOut, h.proxy(this.close, this))
        }
    };
    j.helpers.title = {
        defaults: {
            type: "float",
            position: "bottom"
        },
        beforeShow: function(b) {
            var m = j.current, f = m.title, p = b.type;
            h.isFunction(f) && (f = f.call(m.element, m));
            if (a(f) && "" !== h.trim(f)) {
                m = h('<div class="fancybox-title fancybox-title-' + p + '-wrap">' + f + "</div>");
                switch (p) {
                case"inside":
                    p = j.skin;
                    break;
                case"outside":
                    p = j.wrap;
                    break;
                case"over":
                    p = j.inner;
                    break;
                default:
                    p = j.skin, m.appendTo("body"), l && m.width(m.width()), m.wrapInner('<span class="child"></span>'), j.current.margin[2] += Math.abs(g(m.css("margin-bottom")))
                }
                m["top" === b.position ? "prependTo": "appendTo"](p)
            }
        }
    };
    h.fn.fancybox = function(f) {
        var p, m = h(this), q = this.selector || "", b = function(w) {
            var u = h(this).blur(), t = p, s, r;
            w.ctrlKey || w.altKey || w.shiftKey || w.metaKey || u.is(".fancybox-wrap") || (s = f.groupAttr || "data-fancybox-group", r = u.attr(s), r || (s = "rel", r = u.get(0)[s]), r && "" !== r && "nofollow" !== r && (u = q.length ? h(q) : m, u = u.filter("[" + s + '="' + r + '"]'), t = u.index(this)), f.index = t, !1 !== j.open(u, f) && w.preventDefault())
        };
        f = f || {};
        p = f.index || 0;
        q&&!1 !== f.live ? e.undelegate(q, "click.fb-start").delegate(q + ":not('.fancybox-item, .fancybox-nav')", "click.fb-start", b) : m.unbind("click.fb-start").bind("click.fb-start", b);
        this.filter("[data-fancybox-start=1]").trigger("click");
        return this
    };
    e.ready(function() {
        var b, f;
        h.scrollbarWidth === n && (h.scrollbarWidth = function() {
            var p = h('<div style="width:50px;height:50px;overflow:auto"><div/></div>').appendTo("body"), m = p.children(), m = m.innerWidth() - m.height(99).innerWidth();
            p.remove();
            return m
        });
        h.support.fixedPosition === n && (h.support.fixedPosition = function() {
            var p = h('<div style="position:fixed;top:20px;"></div>').appendTo("body"), m = 20 === p[0].offsetTop || 15 === p[0].offsetTop;
            p.remove();
            return m
        }());
        h.extend(j.defaults, {
            scrollbarWidth: h.scrollbarWidth(),
            fixed: h.support.fixedPosition,
            parent: h("body")
        });
        b = h(A).width();
        i.addClass("fancybox-lock-test");
        f = h(A).width();
        i.removeClass("fancybox-lock-test");
        h("<style type='text/css'>.fancybox-margin{margin-right:" + (f - b) + "px !important;}</style>").appendTo("head")
    })
})(window, document, jQuery); /*
 * hoverIntent v1.8.0 // 2014.06.29 // jQuery v1.9.1+
 * http://cherne.net/brian/resources/jquery.hoverIntent.html
 *
 * You may use hoverIntent under the terms of the MIT license. Basically that
 * means you are free to use hoverIntent as long as this header is left intact.
 * Copyright 2007, 2014 Brian Cherne
 */
(function(a) {
    a.fn.hoverIntent = function(m, d, h) {
        var j = {
            interval: 100,
            sensitivity: 6,
            timeout: 0
        };
        if (typeof m === "object") {
            j = a.extend(j, m)
        } else {
            if (a.isFunction(d)) {
                j = a.extend(j, {
                    over: m,
                    out: d,
                    selector: h
                })
            } else {
                j = a.extend(j, {
                    over: m,
                    out: m,
                    selector: d
                })
            }
        }
        var l, k, g, f;
        var e = function(n) {
            l = n.pageX;
            k = n.pageY
        };
        var c = function(o, n) {
            n.hoverIntent_t = clearTimeout(n.hoverIntent_t);
            if (Math.sqrt((g - l) * (g - l) + (f - k) * (f - k)) < j.sensitivity) {
                a(n).off("mousemove.hoverIntent", e);
                n.hoverIntent_s = true;
                return j.over.apply(n, [o])
            } else {
                g = l;
                f = k;
                n.hoverIntent_t = setTimeout(function() {
                    c(o, n)
                }, j.interval)
            }
        };
        var i = function(o, n) {
            n.hoverIntent_t = clearTimeout(n.hoverIntent_t);
            n.hoverIntent_s = false;
            return j.out.apply(n, [o])
        };
        var b = function(p) {
            var o = a.extend({}, p);
            var n = this;
            if (n.hoverIntent_t) {
                n.hoverIntent_t = clearTimeout(n.hoverIntent_t)
            }
            if (p.type === "mouseenter") {
                g = o.pageX;
                f = o.pageY;
                a(n).on("mousemove.hoverIntent", e);
                if (!n.hoverIntent_s) {
                    n.hoverIntent_t = setTimeout(function() {
                        c(o, n)
                    }, j.interval)
                }
            } else {
                a(n).off("mousemove.hoverIntent", e);
                if (n.hoverIntent_s) {
                    n.hoverIntent_t = setTimeout(function() {
                        i(o, n)
                    }, j.timeout)
                }
            }
        };
        return this.on({
            "mouseenter.hoverIntent": b,
            "mouseleave.hoverIntent": b
        }, j.selector)
    }
})(jQuery);
(function(b) {
    var a = function(m, s) {
        var j = b.extend({}, b.fn.nivoSlider.defaults, s);
        var o = {
            currentSlide: 0,
            currentImage: "",
            totalSlides: 0,
            running: false,
            paused: false,
            stop: false,
            controlNavEl: false
        };
        var d = b(m);
        d.data("nivo:vars", o).addClass("nivoSlider");
        var e = d.children();
        e.each(function() {
            var v = b(this);
            var u = "";
            if (!v.is("img")) {
                if (v.is("a")) {
                    v.addClass("nivo-imageLink");
                    u = v
                }
                v = v.find("img:first")
            }
            var t = (t === 0) ? v.attr("width"): v.width(), i = (i === 0) ? v.attr("height"): v.height();
            if (u !== "") {
                u.css("display", "none")
            }
            v.css("display", "none");
            o.totalSlides++
        });
        if (j.randomStart) {
            j.startSlide = Math.floor(Math.random() * o.totalSlides)
        }
        if (j.startSlide > 0) {
            if (j.startSlide >= o.totalSlides) {
                j.startSlide = o.totalSlides - 1
            }
            o.currentSlide = j.startSlide
        }
        if (b(e[o.currentSlide]).is("img")) {
            o.currentImage = b(e[o.currentSlide])
        } else {
            o.currentImage = b(e[o.currentSlide]).find("img:first")
        }
        if (b(e[o.currentSlide]).is("a")) {
            b(e[o.currentSlide]).css("display", "block")
        }
        var r = b("<img/>").addClass("nivo-main-image");
        r.attr("src", o.currentImage.attr("src")).show();
        d.append(r);
        b(window).resize(function() {
            d.children("img").width(d.width());
            r.attr("src", o.currentImage.attr("src"));
            r.stop().height("auto");
            b(".nivo-slice").remove();
            b(".nivo-box").remove()
        });
        d.append(b('<div class="nivo-caption"></div>'));
        var p = function(i) {
            var u = b(".nivo-caption", d);
            if (o.currentImage.attr("title") != "" && o.currentImage.attr("title") != undefined) {
                var t = o.currentImage.attr("title");
                if (t.substr(0, 1) == "#") {
                    t = b(t).html()
                }
                if (u.css("display") == "block") {
                    setTimeout(function() {
                        u.html(t)
                    }, i.animSpeed)
                } else {
                    u.html(t);
                    u.stop().fadeIn(i.animSpeed)
                }
            } else {
                u.stop().fadeOut(i.animSpeed)
            }
        };
        p(j);
        var c = 0;
        if (!j.manualAdvance && e.length > 1) {
            c = setInterval(function() {
                q(d, e, j, false)
            }, j.pauseTime)
        }
        if (j.directionNav) {
            d.append('<div class="nivo-directionNav"><a class="nivo-prevNav">' + j.prevText + '</a><a class="nivo-nextNav">' + j.nextText + "</a></div>");
            b(d).on("click", "a.nivo-prevNav", function() {
                if (o.running) {
                    return false
                }
                clearInterval(c);
                c = "";
                o.currentSlide -= 2;
                q(d, e, j, "prev")
            });
            b(d).on("click", "a.nivo-nextNav", function() {
                if (o.running) {
                    return false
                }
                clearInterval(c);
                c = "";
                q(d, e, j, "next")
            })
        }
        if (j.controlNav) {
            o.controlNavEl = b('<div class="nivo-controlNav"></div>');
            d.after(o.controlNavEl);
            for (var l = 0; l < e.length; l++) {
                if (j.controlNavThumbs) {
                    o.controlNavEl.addClass("nivo-thumbs-enabled");
                    var f = e.eq(l);
                    if (!f.is("img")) {
                        f = f.find("img:first")
                    }
                    if (f.attr("data-thumb")) {
                        o.controlNavEl.append('<a class="nivo-control" rel="' + l + '"><img src="' + f.attr("data-thumb") + '" alt="" /></a>')
                    }
                } else {
                    o.controlNavEl.append('<a class="nivo-control" rel="' + l + '">' + (l + 1) + "</a>")
                }
            }
            b("a:eq(" + o.currentSlide + ")", o.controlNavEl).addClass("active");
            b("a", o.controlNavEl).bind("click", function() {
                if (o.running) {
                    return false
                }
                if (b(this).hasClass("active")) {
                    return false
                }
                clearInterval(c);
                c = "";
                r.attr("src", o.currentImage.attr("src"));
                o.currentSlide = b(this).attr("rel") - 1;
                q(d, e, j, "control")
            })
        }
        if (j.pauseOnHover) {
            d.hover(function() {
                o.paused = true;
                clearInterval(c);
                c = ""
            }, function() {
                o.paused = false;
                if (c === ""&&!j.manualAdvance) {
                    c = setInterval(function() {
                        q(d, e, j, false)
                    }, j.pauseTime)
                }
            })
        }
        d.bind("nivo:animFinished", function() {
            r.attr("src", o.currentImage.attr("src"));
            o.running = false;
            b(e).each(function() {
                if (b(this).is("a")) {
                    b(this).css("display", "none")
                }
            });
            if (b(e[o.currentSlide]).is("a")) {
                b(e[o.currentSlide]).css("display", "block")
            }
            if (c === ""&&!o.paused&&!j.manualAdvance) {
                c = setInterval(function() {
                    q(d, e, j, false)
                }, j.pauseTime)
            }
            j.afterChange.call(this)
        });
        var g = function(v, u, y) {
            if (b(y.currentImage).parent().is("a")) {
                b(y.currentImage).parent().css("display", "block")
            }
            b('img[src="' + y.currentImage.attr("src") + '"]', v).not(".nivo-main-image,.nivo-control img").width(v.width()).css("visibility", "hidden").show();
            var w = (b('img[src="' + y.currentImage.attr("src") + '"]', v).not(".nivo-main-image,.nivo-control img").parent().is("a")) ? b('img[src="' + y.currentImage.attr("src") + '"]', v).not(".nivo-main-image,.nivo-control img").parent().height(): b('img[src="' + y.currentImage.attr("src") + '"]', v).not(".nivo-main-image,.nivo-control img").height();
            for (var t = 0; t < u.slices; t++) {
                var x = Math.round(v.width() / u.slices);
                if (t === u.slices - 1) {
                    v.append(b('<div class="nivo-slice" name="' + t + '"><img src="' + y.currentImage.attr("src") + '" style="position:absolute; width:' + v.width() + "px; height:auto; display:block !important; top:0; left:-" + ((x + (t * x)) - x) + 'px;" /></div>').css({
                        left: (x * t) + "px",
                        width: (v.width() - (x * t)) + "px",
                        height: w + "px",
                        opacity: "0",
                        overflow: "hidden"
                    }))
                } else {
                    v.append(b('<div class="nivo-slice" name="' + t + '"><img src="' + y.currentImage.attr("src") + '" style="position:absolute; width:' + v.width() + "px; height:auto; display:block !important; top:0; left:-" + ((x + (t * x)) - x) + 'px;" /></div>').css({
                        left: (x * t) + "px",
                        width: x + "px",
                        height: w + "px",
                        opacity: "0",
                        overflow: "hidden"
                    }))
                }
            }
            b(".nivo-slice", v).height(w);
            r.stop().animate({
                height: b(y.currentImage).height()
            }, u.animSpeed)
        };
        var h = function(u, i, x) {
            if (b(x.currentImage).parent().is("a")) {
                b(x.currentImage).parent().css("display", "block")
            }
            b('img[src="' + x.currentImage.attr("src") + '"]', u).not(".nivo-main-image,.nivo-control img").width(u.width()).css("visibility", "hidden").show();
            var t = Math.round(u.width() / i.boxCols), y = Math.round(b('img[src="' + x.currentImage.attr("src") + '"]', u).not(".nivo-main-image,.nivo-control img").height() / i.boxRows);
            for (var v = 0; v < i.boxRows; v++) {
                for (var w = 0; w < i.boxCols; w++) {
                    if (w === i.boxCols - 1) {
                        u.append(b('<div class="nivo-box" name="' + w + '" rel="' + v + '"><img src="' + x.currentImage.attr("src") + '" style="position:absolute; width:' + u.width() + "px; height:auto; display:block; top:-" + (y * v) + "px; left:-" + (t * w) + 'px;" /></div>').css({
                            opacity: 0,
                            left: (t * w) + "px",
                            top: (y * v) + "px",
                            width: (u.width() - (t * w)) + "px"
                        }));
                        b('.nivo-box[name="' + w + '"]', u).height(b('.nivo-box[name="' + w + '"] img', u).height() + "px")
                    } else {
                        u.append(b('<div class="nivo-box" name="' + w + '" rel="' + v + '"><img src="' + x.currentImage.attr("src") + '" style="position:absolute; width:' + u.width() + "px; height:auto; display:block; top:-" + (y * v) + "px; left:-" + (t * w) + 'px;" /></div>').css({
                            opacity: 0,
                            left: (t * w) + "px",
                            top: (y * v) + "px",
                            width: t + "px"
                        }));
                        b('.nivo-box[name="' + w + '"]', u).height(b('.nivo-box[name="' + w + '"] img', u).height() + "px")
                    }
                }
            }
            r.stop().animate({
                height: b(x.currentImage).height()
            }, i.animSpeed)
        };
        var q = function(I, H, L, E) {
            var G = I.data("nivo:vars");
            if (G && (G.currentSlide === G.totalSlides - 1)) {
                L.lastSlide.call(this)
            }
            if ((!G || G.stop)&&!E) {
                return false
            }
            L.beforeChange.call(this);
            if (!E) {
                r.attr("src", G.currentImage.attr("src"))
            } else {
                if (E === "prev") {
                    r.attr("src", G.currentImage.attr("src"))
                }
                if (E === "next") {
                    r.attr("src", G.currentImage.attr("src"))
                }
            }
            G.currentSlide++;
            if (G.currentSlide === G.totalSlides) {
                G.currentSlide = 0;
                L.slideshowEnd.call(this)
            }
            if (G.currentSlide < 0) {
                G.currentSlide = (G.totalSlides - 1)
            }
            if (b(H[G.currentSlide]).is("img")) {
                G.currentImage = b(H[G.currentSlide])
            } else {
                G.currentImage = b(H[G.currentSlide]).find("img:first")
            }
            if (L.controlNav) {
                b("a", G.controlNavEl).removeClass("active");
                b("a:eq(" + G.currentSlide + ")", G.controlNavEl).addClass("active")
            }
            p(L);
            b(".nivo-slice", I).remove();
            b(".nivo-box", I).remove();
            var u = L.effect, N = "";
            if (L.effect === "random") {
                N = new Array("sliceDownRight", "sliceDownLeft", "sliceUpRight", "sliceUpLeft", "sliceUpDown", "sliceUpDownLeft", "fold", "fade", "boxRandom", "boxRain", "boxRainReverse", "boxRainGrow", "boxRainGrowReverse");
                u = N[Math.floor(Math.random() * (N.length + 1))];
                if (u === undefined) {
                    u = "fade"
                }
            }
            if (L.effect.indexOf(",")!==-1) {
                N = L.effect.split(",");
                u = N[Math.floor(Math.random() * (N.length))];
                if (u === undefined) {
                    u = "fade"
                }
            }
            if (G.currentImage.attr("data-transition")) {
                u = G.currentImage.attr("data-transition")
            }
            G.running = true;
            var M = 0, J = 0, t = "", A = "", K = "", x = "";
            if (u === "sliceDown" || u === "sliceDownRight" || u === "sliceDownLeft") {
                g(I, L, G);
                M = 0;
                J = 0;
                t = b(".nivo-slice", I);
                if (u === "sliceDownLeft") {
                    t = b(".nivo-slice", I)._reverse()
                }
                t.each(function() {
                    var i = b(this);
                    i.css({
                        top: "0px"
                    });
                    if (J === L.slices - 1) {
                        setTimeout(function() {
                            i.animate({
                                opacity: "1.0"
                            }, L.animSpeed, "", function() {
                                I.trigger("nivo:animFinished")
                            })
                        }, (100 + M))
                    } else {
                        setTimeout(function() {
                            i.animate({
                                opacity: "1.0"
                            }, L.animSpeed)
                        }, (100 + M))
                    }
                    M += 50;
                    J++
                })
            } else {
                if (u === "sliceUp" || u === "sliceUpRight" || u === "sliceUpLeft") {
                    g(I, L, G);
                    M = 0;
                    J = 0;
                    t = b(".nivo-slice", I);
                    if (u === "sliceUpLeft") {
                        t = b(".nivo-slice", I)._reverse()
                    }
                    t.each(function() {
                        var i = b(this);
                        i.css({
                            bottom: "0px"
                        });
                        if (J === L.slices - 1) {
                            setTimeout(function() {
                                i.animate({
                                    opacity: "1.0"
                                }, L.animSpeed, "", function() {
                                    I.trigger("nivo:animFinished")
                                })
                            }, (100 + M))
                        } else {
                            setTimeout(function() {
                                i.animate({
                                    opacity: "1.0"
                                }, L.animSpeed)
                            }, (100 + M))
                        }
                        M += 50;
                        J++
                    })
                } else {
                    if (u === "sliceUpDown" || u === "sliceUpDownRight" || u === "sliceUpDownLeft") {
                        g(I, L, G);
                        M = 0;
                        J = 0;
                        var C = 0;
                        t = b(".nivo-slice", I);
                        if (u === "sliceUpDownLeft") {
                            t = b(".nivo-slice", I)._reverse()
                        }
                        t.each(function() {
                            var i = b(this);
                            if (J === 0) {
                                i.css("top", "0px");
                                J++
                            } else {
                                i.css("bottom", "0px");
                                J = 0
                            }
                            if (C === L.slices - 1) {
                                setTimeout(function() {
                                    i.animate({
                                        opacity: "1.0"
                                    }, L.animSpeed, "", function() {
                                        I.trigger("nivo:animFinished")
                                    })
                                }, (100 + M))
                            } else {
                                setTimeout(function() {
                                    i.animate({
                                        opacity: "1.0"
                                    }, L.animSpeed)
                                }, (100 + M))
                            }
                            M += 50;
                            C++
                        })
                    } else {
                        if (u === "fold") {
                            g(I, L, G);
                            M = 0;
                            J = 0;
                            b(".nivo-slice", I).each(function() {
                                var i = b(this);
                                var v = i.width();
                                i.css({
                                    top: "0px",
                                    width: "0px"
                                });
                                if (J === L.slices - 1) {
                                    setTimeout(function() {
                                        i.animate({
                                            width: v,
                                            opacity: "1.0"
                                        }, L.animSpeed, "", function() {
                                            I.trigger("nivo:animFinished")
                                        })
                                    }, (100 + M))
                                } else {
                                    setTimeout(function() {
                                        i.animate({
                                            width: v,
                                            opacity: "1.0"
                                        }, L.animSpeed)
                                    }, (100 + M))
                                }
                                M += 50;
                                J++
                            })
                        } else {
                            if (u === "fade") {
                                g(I, L, G);
                                A = b(".nivo-slice:first", I);
                                A.css({
                                    width: I.width() + "px"
                                });
                                A.animate({
                                    opacity: "1.0"
                                }, (L.animSpeed * 2), "", function() {
                                    I.trigger("nivo:animFinished")
                                })
                            } else {
                                if (u === "slideInRight") {
                                    g(I, L, G);
                                    A = b(".nivo-slice:first", I);
                                    A.css({
                                        width: "0px",
                                        opacity: "1"
                                    });
                                    A.animate({
                                        width: I.width() + "px"
                                    }, (L.animSpeed * 2), "", function() {
                                        I.trigger("nivo:animFinished")
                                    })
                                } else {
                                    if (u === "slideInLeft") {
                                        g(I, L, G);
                                        A = b(".nivo-slice:first", I);
                                        A.css({
                                            width: "0px",
                                            opacity: "1",
                                            left: "",
                                            right: "0px"
                                        });
                                        A.animate({
                                            width: I.width() + "px"
                                        }, (L.animSpeed * 2), "", function() {
                                            A.css({
                                                left: "0px",
                                                right: ""
                                            });
                                            I.trigger("nivo:animFinished")
                                        })
                                    } else {
                                        if (u === "boxRandom") {
                                            h(I, L, G);
                                            K = L.boxCols * L.boxRows;
                                            J = 0;
                                            M = 0;
                                            x = n(b(".nivo-box", I));
                                            x.each(function() {
                                                var i = b(this);
                                                if (J === K - 1) {
                                                    setTimeout(function() {
                                                        i.animate({
                                                            opacity: "1"
                                                        }, L.animSpeed, "", function() {
                                                            I.trigger("nivo:animFinished")
                                                        })
                                                    }, (100 + M))
                                                } else {
                                                    setTimeout(function() {
                                                        i.animate({
                                                            opacity: "1"
                                                        }, L.animSpeed)
                                                    }, (100 + M))
                                                }
                                                M += 20;
                                                J++
                                            })
                                        } else {
                                            if (u === "boxRain" || u === "boxRainReverse" || u === "boxRainGrow" || u === "boxRainGrowReverse") {
                                                h(I, L, G);
                                                K = L.boxCols * L.boxRows;
                                                J = 0;
                                                M = 0;
                                                var z = 0;
                                                var F = 0;
                                                var D = [];
                                                D[z] = [];
                                                x = b(".nivo-box", I);
                                                if (u === "boxRainReverse" || u === "boxRainGrowReverse") {
                                                    x = b(".nivo-box", I)._reverse()
                                                }
                                                x.each(function() {
                                                    D[z][F] = b(this);
                                                    F++;
                                                    if (F === L.boxCols) {
                                                        z++;
                                                        F = 0;
                                                        D[z] = []
                                                    }
                                                });
                                                for (var B = 0; B < (L.boxCols * 2); B++) {
                                                    var w = B;
                                                    for (var y = 0; y < L.boxRows; y++) {
                                                        if (w >= 0 && w < L.boxCols) {
                                                            (function(T, O, S, P, U) {
                                                                var R = b(D[T][O]);
                                                                var v = R.width();
                                                                var Q = R.height();
                                                                if (u === "boxRainGrow" || u === "boxRainGrowReverse") {
                                                                    R.width(0).height(0)
                                                                }
                                                                if (P === U - 1) {
                                                                    setTimeout(function() {
                                                                        R.animate({
                                                                            opacity: "1",
                                                                            width: v,
                                                                            height: Q
                                                                        }, L.animSpeed / 1.3, "", function() {
                                                                            I.trigger("nivo:animFinished")
                                                                        })
                                                                    }, (100 + S))
                                                                } else {
                                                                    setTimeout(function() {
                                                                        R.animate({
                                                                            opacity: "1",
                                                                            width: v,
                                                                            height: Q
                                                                        }, L.animSpeed / 1.3)
                                                                    }, (100 + S))
                                                                }
                                                            })(y, w, M, J, K);
                                                            J++
                                                        }
                                                        w--
                                                    }
                                                    M += 100
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        var n = function(u) {
            for (var v, t, w = u.length; w; v = parseInt(Math.random() * w, 10), t = u[--w], u[w] = u[v], u[v] = t) {}
            return u
        };
        var k = function(i) {
            if (this.console && typeof console.log !== "undefined") {
                console.log(i)
            }
        };
        this.stop = function() {
            if (!b(m).data("nivo:vars").stop) {
                b(m).data("nivo:vars").stop = true;
                k("Stop Slider")
            }
        };
        this.start = function() {
            if (b(m).data("nivo:vars").stop) {
                b(m).data("nivo:vars").stop = false;
                k("Start Slider")
            }
        };
        j.afterLoad.call(this);
        return this
    };
    b.fn.nivoSlider = function(c) {
        return this.each(function(e, g) {
            var d = b(this);
            if (d.data("nivoslider")) {
                return d.data("nivoslider")
            }
            var f = new a(this, c);
            d.data("nivoslider", f)
        })
    };
    b.fn.nivoSlider.defaults = {
        effect: "random",
        slices: 15,
        boxCols: 8,
        boxRows: 4,
        animSpeed: 500,
        pauseTime: 3000,
        startSlide: 0,
        directionNav: true,
        controlNav: true,
        controlNavThumbs: false,
        pauseOnHover: true,
        manualAdvance: false,
        prevText: "Prev",
        nextText: "Next",
        randomStart: false,
        beforeChange: function() {},
        afterChange: function() {},
        slideshowEnd: function() {},
        lastSlide: function() {},
        afterLoad: function() {}
    };
    b.fn._reverse = [].reverse
})(jQuery);
if (window.jQuery) {
    (function(a) {
        if ((!a.support.opacity&&!a.support.style)) {
            try {
                document.execCommand("BackgroundImageCache", false, true)
            } catch (b) {}
        }
        a.fn.rating = function(e) {
            if (this.length === 0) {
                return this
            }
            if (typeof arguments[0] == "string") {
                if (this.length > 1) {
                    var d = arguments;
                    return this.each(function() {
                        a.fn.rating.apply(a(this), d)
                    })
                }
                a.fn.rating[arguments[0]].apply(this, a.makeArray(arguments).slice(1) || []);
                return this
            }
            var c = a.extend({}, a.fn.rating.options, e || {});
            a.fn.rating.calls++;
            this.not(".star-rating-applied").addClass("star-rating-applied").each(function() {
                var h, m = a(this);
                var f = (this.name || "unnamed-rating").replace(/\[|\]/g, "_").replace(/^\_+|\_+$/g, "");
                var g = a(this.form || document.body);
                var l = g.data("rating");
                if (!l || l.call != a.fn.rating.calls) {
                    l = {
                        count: 0,
                        call: a.fn.rating.calls
                    }
                }
                var o = l[f] || g.data("rating" + f);
                if (o) {
                    h = o.data("rating")
                }
                if (o && h) {
                    h.count++
                } else {
                    h = a.extend({}, c || {}, (a.metadata ? m.metadata() : (a.meta ? m.data() : null)) || {}, {
                        count: 0,
                        stars: [],
                        inputs: []
                    });
                    h.serial = l.count++;
                    o = a('<span class="star-rating-control"/>');
                    m.before(o);
                    o.addClass("rating-to-be-drawn");
                    if (m.attr("disabled") || m.hasClass("disabled")) {
                        h.readOnly = true
                    }
                    if (m.hasClass("required")) {
                        h.required = true
                    }
                    o.append(h.cancel = a('<div class="rating-cancel"><a title="' + h.cancel + '">' + h.cancelValue + "</a></div>").on("mouseover", function() {
                        a(this).rating("drain");
                        a(this).addClass("star-rating-hover")
                    }).on("mouseout", function() {
                        a(this).rating("draw");
                        a(this).removeClass("star-rating-hover")
                    }).on("click", function() {
                        a(this).rating("select")
                    }).data("rating", h))
                }
                var k = a('<div role="text" aria-label="' + (this.title || this.value) + '" class="star-rating rater-' + h.serial + '"><a title="' + (this.title || this.value) + '">' + this.value + "</a></div>");
                o.append(k);
                if (this.id) {
                    k.attr("id", this.id)
                }
                if (this.className) {
                    k.addClass(this.className)
                }
                if (h.half) {
                    h.split = 2
                }
                if (typeof h.split == "number" && h.split > 0) {
                    var j = (a.fn.width ? k.width() : 0) || h.starWidth;
                    var i = (h.count%h.split), n = Math.floor(j / h.split);
                    k.width(n).find("a").css({
                        "margin-left": "-" + (i * n) + "px"
                    })
                }
                if (h.readOnly) {
                    k.addClass("star-rating-readonly")
                } else {
                    k.addClass("star-rating-live").on("mouseover", function() {
                        a(this).rating("fill");
                        a(this).rating("focus")
                    }).on("mouseout", function() {
                        a(this).rating("draw");
                        a(this).rating("blur")
                    }).on("click", function() {
                        a(this).rating("select")
                    })
                }
                if (this.checked) {
                    h.current = k
                }
                if (this.nodeName == "A") {
                    if (a(this).hasClass("selected")) {
                        h.current = k
                    }
                }
                m.hide();
                m.on("change.rating", function(p) {
                    if (p.selfTriggered) {
                        return false
                    }
                    a(this).rating("select")
                });
                k.data("rating.input", m.data("rating.star", k));
                h.stars[h.stars.length] = k[0];
                h.inputs[h.inputs.length] = m[0];
                h.rater = l[f] = o;
                h.context = g;
                m.data("rating", h);
                o.data("rating", h);
                k.data("rating", h);
                g.data("rating", l);
                g.data("rating" + f, o)
            });
            a(".rating-to-be-drawn").rating("draw").removeClass("rating-to-be-drawn");
            return this
        };
        a.extend(a.fn.rating, {
            calls: 0,
            focus: function() {
                var d = this.data("rating");
                if (!d) {
                    return this
                }
                if (!d.focus) {
                    return this
                }
                var c = a(this).data("rating.input") || a(this.tagName == "INPUT" ? this : null);
                if (d.focus) {
                    d.focus.apply(c[0], [c.val(), a("a", c.data("rating.star"))[0]])
                }
            },
            blur: function() {
                var d = this.data("rating");
                if (!d) {
                    return this
                }
                if (!d.blur) {
                    return this
                }
                var c = a(this).data("rating.input") || a(this.tagName == "INPUT" ? this : null);
                if (d.blur) {
                    d.blur.apply(c[0], [c.val(), a("a", c.data("rating.star"))[0]])
                }
            },
            fill: function() {
                var c = this.data("rating");
                if (!c) {
                    return this
                }
                if (c.readOnly) {
                    return 
                }
                this.rating("drain");
                this.prevAll().addBack().filter(".rater-" + c.serial).addClass("star-rating-hover")
            },
            drain: function() {
                var c = this.data("rating");
                if (!c) {
                    return this
                }
                if (c.readOnly) {
                    return 
                }
                c.rater.children().filter(".rater-" + c.serial).removeClass("star-rating-on").removeClass("star-rating-hover")
            },
            draw: function() {
                var e = this.data("rating");
                if (!e) {
                    return this
                }
                this.rating("drain");
                var d = a(e.current);
                var c = d.length ? d.prevAll().addBack().filter(".rater-" + e.serial): null;
                if (c) {
                    c.addClass("star-rating-on")
                }
                e.cancel[e.readOnly || e.required ? "hide": "show"]();
                this.siblings()[e.readOnly ? "addClass": "removeClass"]("star-rating-readonly")
            },
            select: function(d, g) {
                var f = this.data("rating");
                if (!f) {
                    return this
                }
                if (f.readOnly) {
                    return 
                }
                f.current = null;
                if (typeof d != "undefined" || this.length > 1) {
                    if (typeof d == "number") {
                        return a(f.stars[d]).rating("select", undefined, g)
                    }
                    if (typeof d == "string") {
                        a.each(f.stars, function() {
                            if (a(this).data("rating.input").val() == d) {
                                a(this).rating("select", undefined, g)
                            }
                        });
                        return this
                    }
                } else {
                    f.current = this[0].tagName == "INPUT" ? this.data("rating.star") : (this.is(".rater-" + f.serial) ? this : null)
                }
                this.data("rating", f);
                this.rating("draw");
                var e = a(f.current ? f.current.data("rating.input") : null);
                var c = a(f.inputs).filter(":checked");
                var h = a(f.inputs).not(e);
                h.prop("checked", false);
                e.prop("checked", true);
                a(e.length ? e : c).trigger({
                    type: "change",
                    selfTriggered: true
                });
                if ((g || g === undefined) && f.callback) {
                    f.callback.apply(e[0], [e.val(), a("a", f.current)[0]])
                }
                return this
            },
            readOnly: function(c, d) {
                var e = this.data("rating");
                if (!e) {
                    return this
                }
                e.readOnly = c || c === undefined ? true : false;
                if (d) {
                    a(e.inputs).attr("disabled", "disabled")
                } else {
                    a(e.inputs).removeAttr("disabled")
                }
                this.data("rating", e);
                this.rating("draw")
            },
            disable: function() {
                this.rating("readOnly", true, true)
            },
            enable: function() {
                this.rating("readOnly", false, false)
            }
        });
        a.fn.rating.options = {
            cancel: "Cancel Rating",
            cancelValue: "",
            split: 0,
            starWidth: 16
        };
        a(function() {
            a("input[type=radio].star").rating()
        })
    })(jQuery)
};
(function(a) {
    a.fn.pajinate = function(d) {
        var j = "current_page";
        var q = "items_per_page";
        var m;
        var l = {
            item_container_id: ".content",
            items_per_page: 10,
            nav_panel_id: ".page_navigation",
            nav_info_id: ".info_text",
            num_page_links_to_display: 20,
            start_page: 0,
            wrap_around: false,
            nav_label_first: "First",
            nav_label_prev: "Prev",
            nav_label_next: "Next",
            nav_label_last: "Last",
            nav_order: ["first", "prev", "num", "next", "last"],
            nav_label_info: "Showing {0}-{1} of {2} results",
            show_first_last: true,
            abort_on_small_lists: false,
            jquery_ui: false,
            jquery_ui_active: "ui-state-highlight",
            jquery_ui_default: "ui-state-default",
            jquery_ui_disabled: "ui-state-disabled"
        };
        var d = a.extend(l, d);
        var n;
        var o;
        var e;
        var g;
        var k;
        var s = d.jquery_ui ? d.jquery_ui_default: "";
        var h = d.jquery_ui ? d.jquery_ui_active: "";
        var u = d.jquery_ui ? d.jquery_ui_disabled: "";
        return this.each(function() {
            o = a(this);
            n = a(this).find(d.item_container_id);
            e = o.find(d.item_container_id).children();
            if (d.abort_on_small_lists && d.items_per_page >= e.length) {
                return o
            }
            m = o;
            m.data(j, 0);
            m.data(q, d.items_per_page);
            var v = n.children().length;
            var D = Math.ceil(v / d.items_per_page);
            var y = '<span class="ellipse more">...</span>';
            var A = '<span class="ellipse less">...</span>';
            var x=!d.show_first_last ? "" : '<a class="first_link ' + s + '" href="">' + d.nav_label_first + "</a>";
            var C=!d.show_first_last ? "" : '<a class="last_link ' + s + '" href="">' + d.nav_label_last + "</a>";
            var B = "";
            for (var w = 0; w < d.nav_order.length; w++) {
                switch (d.nav_order[w]) {
                case"first":
                    B += x;
                    break;
                case"last":
                    B += C;
                    break;
                case"next":
                    B += '<a class="next_link ' + s + '" href="">' + d.nav_label_next + "</a>";
                    break;
                case"prev":
                    B += '<a class="previous_link ' + s + '" href="">' + d.nav_label_prev + "</a>";
                    break;
                case"num":
                    B += A;
                    var z = 0;
                    while (D > z) {
                        B += '<a class="page_link ' + s + '" href="" longdesc="' + z + '">' + (z + 1) + "</a>";
                        z++
                    }
                    B += y;
                    break;
                default:
                    break
                }
            }
            g = o.find(d.nav_panel_id);
            g.html(B).each(function() {
                a(this).find(".page_link:first").addClass("first");
                a(this).find(".page_link:last").addClass("last")
            });
            g.children(".ellipse").hide();
            g.find(".previous_link").next().next().addClass("active_page " + h);
            e.hide();
            e.slice(0, m.data(q)).show();
            k = o.children(d.nav_panel_id + ":first").children(".page_link").length;
            d.num_page_links_to_display = Math.min(d.num_page_links_to_display, k);
            g.children(".page_link").hide();
            g.each(function() {
                a(this).children(".page_link").slice(0, d.num_page_links_to_display).show()
            });
            o.find(".first_link").click(function(E) {
                E.preventDefault();
                i(a(this), 0);
                r(0)
            });
            o.find(".last_link").click(function(F) {
                F.preventDefault();
                var E = k - 1;
                t(a(this), E);
                r(E)
            });
            o.find(".previous_link").click(function(E) {
                E.preventDefault();
                p(a(this))
            });
            o.find(".next_link").click(function(E) {
                E.preventDefault();
                c(a(this))
            });
            o.find(".page_link").click(function(E) {
                E.preventDefault();
                r(a(this).attr("longdesc"))
            });
            r(parseInt(d.start_page));
            f();
            if (!d.wrap_around) {
                b()
            }
        });
        function p(v) {
            new_page = parseInt(m.data(j)) - 1;
            if (a(v).siblings(".active_page").prev(".page_link").length == true) {
                i(v, new_page);
                r(new_page)
            } else {
                if (d.wrap_around) {
                    r(k - 1)
                }
            }
        }
        function c(v) {
            new_page = parseInt(m.data(j)) + 1;
            if (a(v).siblings(".active_page").next(".page_link").length == true) {
                t(v, new_page);
                r(new_page)
            } else {
                if (d.wrap_around) {
                    r(0)
                }
            }
        }
        function r(w) {
            var x = parseInt(m.data(q));
            start_from = w * x;
            end_on = start_from + x;
            var v = e.hide().slice(start_from, end_on);
            v.show();
            o.find(d.nav_panel_id).children(".page_link[longdesc=" + w + "]").addClass("active_page " + h).siblings(".active_page").removeClass("active_page " + h);
            m.data(j, w);
            o.find(d.nav_info_id).html(d.nav_label_info.replace("{0}", start_from + 1).replace("{1}", start_from + v.length).replace("{2}", e.length));
            f();
            b()
        }
        function t(y, x) {
            var v = x;
            var w = a(y).siblings(".active_page");
            if (w.siblings(".page_link[longdesc=" + v + "]").css("display") == "none") {
                g.each(function() {
                    a(this).children(".page_link").hide().slice(parseInt(v - d.num_page_links_to_display + 1), v + 1).show()
                })
            }
        }
        function i(y, x) {
            var v = x;
            var w = a(y).siblings(".active_page");
            if (w.siblings(".page_link[longdesc=" + v + "]").css("display") == "none") {
                g.each(function() {
                    a(this).children(".page_link").hide().slice(v, v + parseInt(d.num_page_links_to_display)).show()
                })
            }
        }
        function f() {
            if (!g.children(".page_link:visible").hasClass("last")) {
                g.children(".more").show()
            } else {
                g.children(".more").hide()
            }
            if (!g.children(".page_link:visible").hasClass("first")) {
                g.children(".less").show()
            } else {
                g.children(".less").hide()
            }
        }
        function b() {
            if (g.children(".last").hasClass("active_page")) {
                g.children(".next_link").add(".last_link").addClass("no_more " + u)
            } else {
                g.children(".next_link").add(".last_link").removeClass("no_more " + u)
            }
            if (g.children(".first").hasClass("active_page")) {
                g.children(".previous_link").add(".first_link").addClass("no_more " + u)
            } else {
                g.children(".previous_link").add(".first_link").removeClass("no_more " + u)
            }
        }
    }
})(jQuery);
(function(a) {
    if (typeof define === "function" && define.amd) {
        define(["jquery"], a)
    } else {
        a(jQuery)
    }
}(function(a) {
    a.fn.tweet = function(d) {
        var r = a.extend({
            username: null,
            list: null,
            favorites: false,
            query: null,
            avatar_size: null,
            count: 3,
            fetch: null,
            page: 1,
            retweets: true,
            intro_text: null,
            outro_text: null,
            join_text: null,
            auto_join_text_default: " I said, ",
            auto_join_text_ed: " I ",
            auto_join_text_ing: " I am ",
            auto_join_text_reply: " I replied to ",
            auto_join_text_url: " I was looking at ",
            loading_text: null,
            refresh_interval: null,
            twitter_url: "twitter.com",
            twitter_api_url: "api.twitter.com",
            twitter_search_url: "search.twitter.com",
            template: "{avatar}{time}{join} {text}",
            comparator: function(s, o) {
                return o.tweet_time - s.tweet_time
            },
            filter: function(o) {
                return true
            }
        }, d);
        var b = /\b((?:https?:\/\/|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))/gi;
        function n(t, u) {
            if (typeof t === "string") {
                var o = t;
                for (var s in u) {
                    var v = u[s];
                    o = o.split("{" + s + "}").join(v === null ? "" : v)
                }
                return o
            } else {
                return t(u)
            }
        }
        a.extend({
            tweet: {
                t: n
            }
        });
        function g(s, o) {
            return function() {
                var t = [];
                this.each(function() {
                    t.push(this.replace(s, o))
                });
                return a(t)
            }
        }
        function m(o) {
            return o.replace(/</g, "&lt;").replace(/>/g, "^&gt;")
        }
        a.fn.extend({
            linkUser: g(/(^|[\W])@(\w+)/gi, '$1<span class="at">@</span><a href="http://' + r.twitter_url + '/$2">$2</a>'),
            linkHash: g(/(?:^| )[\#]+([\w\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0600-\u06ff]+)/gi, ' <a href="http://' + r.twitter_search_url + "/search?q=&tag=$1&lang=all" + ((r.username && r.username.length == 1&&!r.list) ? "&from=" + r.username.join("%2BOR%2B") : "") + '" class="tweet_hashtag">#$1</a>'),
            makeHeart: g(/(&lt;)+[3]/gi, "<tt class='heart'>&#x2665;</tt>")
        });
        function f(s, o) {
            return s.replace(b, function(v) {
                var u = (/^[a-z]+:/i).test(v) ? v: "http://" + v;
                var x = v;
                for (var w = 0; w < o.length; ++w) {
                    var t = o[w];
                    if (t.url == u && t.expanded_url) {
                        u = t.expanded_url;
                        x = t.display_url;
                        break
                    }
                }
                return '<a href="' + m(u) + '">' + m(x) + "</a>"
            })
        }
        function l(o) {
            return Date.parse(o.replace(/^([a-z]{3})( [a-z]{3} \d\d?)(.*)( \d{4})$/i, "$1,$2$4$3"))
        }
        function q(o) {
            var t = function(v) {
                return parseInt(v, 10)
            };
            var s = new Date();
            var u = t((s.getTime() - o) / 1000);
            if (u < 1) {
                u = 0
            }
            return {
                days: t(u / 86400),
                hours: t(u / 3600),
                minutes: t(u / 60),
                seconds: t(u)
            }
        }
        function c(o) {
            if (o.days > 2) {
                return "about " + o.days + " days ago"
            }
            if (o.hours > 24) {
                return "about a day ago"
            }
            if (o.hours > 2) {
                return "about " + o.hours + " hours ago"
            }
            if (o.minutes > 45) {
                return "about an hour ago"
            }
            if (o.minutes > 2) {
                return "about " + o.minutes + " minutes ago"
            }
            if (o.seconds > 1) {
                return "about " + o.seconds + " seconds ago"
            }
            return "just now"
        }
        function i(o) {
            if (o.match(/^(@([A-Za-z0-9-_]+)) .*/i)) {
                return r.auto_join_text_reply
            } else {
                if (o.match(b)) {
                    return r.auto_join_text_url
                } else {
                    if (o.match(/^((\w+ed)|just) .*/im)) {
                        return r.auto_join_text_ed
                    } else {
                        if (o.match(/^(\w*ing) .*/i)) {
                            return r.auto_join_text_ing
                        } else {
                            return r.auto_join_text_default
                        }
                    }
                }
            }
        }
        function e() {
            var s = ("https:" == document.location.protocol ? "https:" : "http:");
            var o = (r.fetch === null) ? r.count: r.fetch;
            var u = "&callback=?";
            if (r.list) {
                return s + "//" + r.twitter_api_url + "/1/" + r.username[0] + "/lists/" + r.list + "/statuses.json?page=" + r.page + "&per_page=" + o + u
            } else {
                if (r.favorites) {
                    return s + "//" + r.twitter_api_url + "/1/favorites.json?screen_name=" + r.username[0] + "&page=" + r.page + "&count=" + o + u
                } else {
                    if (r.query === null && r.username.length == 1) {
                        return s + "//" + r.twitter_api_url + "/1/statuses/user_timeline.json?screen_name=" + r.username[0] + "&count=" + o + (r.retweets ? "&include_rts=1" : "") + "&page=" + r.page + u
                    } else {
                        var t = (r.query || "from:" + r.username.join(" OR from:"));
                        return s + "//" + r.twitter_search_url + "/search.json?&q=" + encodeURIComponent(t) + "&rpp=" + o + "&page=" + r.page + u
                    }
                }
            }
        }
        function p(o, s) {
            if (s) {
                return ("user" in o) ? o.user.profile_image_url_https : p(o, false).replace(/^http:\/\/[a-z0-9]{1,3}\.twimg\.com\//, "https://s3.amazonaws.com/twitter_production/")
            } else {
                return o.profile_image_url || o.user.profile_image_url
            }
        }
        function k(s) {
            var t = {};
            t.item = s;
            t.source = s.source;
            t.screen_name = s.from_user || s.user.screen_name;
            t.name = s.from_user_name || s.user.name;
            t.retweet = typeof(s.retweeted_status) != "undefined";
            t.tweet_time = l(s.created_at);
            t.join_text = r.join_text == "auto" ? i(s.text) : r.join_text;
            t.tweet_id = s.id_str;
            t.twitter_base = "http://" + r.twitter_url + "/";
            t.user_url = t.twitter_base + t.screen_name;
            t.tweet_url = t.user_url + "/status/" + t.tweet_id;
            t.reply_url = t.twitter_base + "intent/tweet?in_reply_to=" + t.tweet_id;
            t.retweet_url = t.twitter_base + "intent/retweet?tweet_id=" + t.tweet_id;
            t.favorite_url = t.twitter_base + "intent/favorite?tweet_id=" + t.tweet_id;
            t.retweeted_screen_name = t.retweet && s.retweeted_status.user.screen_name;
            t.tweet_relative_time = c(q(t.tweet_time));
            t.entities = s.entities ? (s.entities.urls || []).concat(s.entities.media || []) : [];
            t.tweet_raw_text = t.retweet ? ("RT @" + t.retweeted_screen_name + " " + s.retweeted_status.text) : s.text;
            t.tweet_text = a([f(t.tweet_raw_text, t.entities)]).linkUser().linkHash()[0];
            t.retweeted_tweet_text = a([f(s.text, t.entities)]).linkUser().linkHash()[0];
            t.tweet_text_fancy = a([t.tweet_text]).makeHeart()[0];
            t.avatar_size = r.avatar_size;
            t.avatar_url = p(t.retweet ? s.retweeted_status : s, (document.location.protocol === "https:"));
            t.avatar_screen_name = t.retweet ? t.retweeted_screen_name : t.screen_name;
            t.avatar_profile_url = t.twitter_base + t.avatar_screen_name;
            t.user = n('<a class="tweet_user" href="{user_url}">{screen_name}</a>', t);
            t.join = r.join_text ? n('<span class="tweet_join">{join_text}</span>', t) : "";
            t.avatar = t.avatar_size ? n('<a class="tweet_avatar" href="{avatar_profile_url}"><img src="{avatar_url}" height="{avatar_size}" width="{avatar_size}" alt="{avatar_screen_name}\'s avatar" title="{avatar_screen_name}\'s avatar" border="0"/></a>', t) : "";
            t.time = n('<span class="tweet_time"><a href="{tweet_url}" title="view tweet on twitter">{tweet_relative_time}</a></span>', t);
            t.text = n('<span class="tweet_text">{tweet_text_fancy}</span>', t);
            t.retweeted_text = n('<span class="tweet_text">{retweeted_tweet_text}</span>', t);
            t.reply_action = n('<a class="tweet_action tweet_reply" href="{reply_url}">reply</a>', t);
            t.retweet_action = n('<a class="tweet_action tweet_retweet" href="{retweet_url}">retweet</a>', t);
            t.favorite_action = n('<a class="tweet_action tweet_favorite" href="{favorite_url}">favorite</a>', t);
            return t
        }
        function h(s, t) {
            var o = a('<ul class="tweet_list">');
            o.append(a.map(t, function(u) {
                return "<li>" + n(r.template, u) + "</li>"
            }).join("")).children("li:first").addClass("tweet_first").end().children("li:odd").addClass("tweet_even").end().children("li:even").addClass("tweet_odd");
            a(s).empty().append(o);
            if (r.intro_text) {
                o.before('<p class="tweet_intro">' + r.intro_text + "</p>")
            }
            if (r.outro_text) {
                o.after('<p class="tweet_outro">' + r.outro_text + "</p>")
            }
            a(s).trigger("loaded").trigger((t.length === 0 ? "empty" : "full"));
            if (r.refresh_interval) {
                window.setTimeout(function() {
                    a(s).trigger("tweet:load")
                }, 1000 * r.refresh_interval)
            }
        }
        function j(o) {
            var s = a('<p class="loading">' + r.loading_text + "</p>");
            if (r.loading_text) {
                a(o).not(":has(.tweet_list)").empty().append(s)
            }
            a.getJSON(e(), function(t) {
                var u = a.map(t.results || t, k);
                u = a.grep(u, r.filter).sort(r.comparator).slice(0, r.count);
                a(o).trigger("tweet:retrieved", [u])
            })
        }
        return this.each(function(o, s) {
            if (r.username && typeof(r.username) == "string") {
                r.username = [r.username]
            }
            a(s).unbind("tweet:render").unbind("tweet:retrieved").unbind("tweet:load").bind({
                "tweet:load": function() {
                    j(s)
                },
                "tweet:retrieved": function(t, u) {
                    a(s).trigger("tweet:render", [u])
                },
                "tweet:render": function(t, u) {
                    h(a(s), u)
                }
            }).trigger("tweet:load")
        })
    }
}));
$.extend($.fn, {
    validate: function(a) {
        if (!this.length) {
            if (a && a.debug && window.console) {
                console.warn("Nothing selected, can't validate, returning nothing.")
            }
            return 
        }
        var b = $.data(this[0], "validator");
        if (b) {
            return b
        }
        this.attr("novalidate", "novalidate");
        b = new $.validator(a, this[0]);
        $.data(this[0], "validator", b);
        if (b.settings.onsubmit) {
            this.validateDelegate(":submit", "click", function(c) {
                if (b.settings.submitHandler) {
                    b.submitButton = c.target
                }
                if ($(c.target).hasClass("cancel")) {
                    b.cancelSubmit = true
                }
                if ($(c.target).attr("formnovalidate") !== undefined) {
                    b.cancelSubmit = true
                }
            });
            this.submit(function(c) {
                if (b.settings.debug) {
                    c.preventDefault()
                }
                function d() {
                    var e;
                    if (b.settings.submitHandler) {
                        if (b.submitButton) {
                            e = $("<input type='hidden'/>").attr("name", b.submitButton.name).val($(b.submitButton).val()).appendTo(b.currentForm)
                        }
                        b.settings.submitHandler.call(b, b.currentForm, c);
                        if (b.submitButton) {
                            e.remove()
                        }
                        return false
                    }
                    return true
                }
                if (b.cancelSubmit) {
                    b.cancelSubmit = false;
                    return d()
                }
                if (b.form()) {
                    if (b.pendingRequest) {
                        b.formSubmitted = true;
                        return false
                    }
                    return d()
                } else {
                    b.focusInvalid();
                    return false
                }
            })
        }
        return b
    },
    valid: function() {
        var b, a;
        if ($(this[0]).is("form")) {
            b = this.validate().form()
        } else {
            b = true;
            a = $(this[0].form).validate();
            this.each(function() {
                b = a.element(this) && b
            })
        }
        return b
    },
    removeAttrs: function(c) {
        var a = {}, b = this;
        $.each(c.split(/\s/), function(d, e) {
            a[e] = b.attr(e);
            b.removeAttr(e)
        });
        return a
    },
    rules: function(d, a) {
        var f = this[0], c, h, i, e, b, g;
        if (d) {
            c = $.data(f.form, "validator").settings;
            h = c.rules;
            i = $.validator.staticRules(f);
            switch (d) {
            case"add":
                $.extend(i, $.validator.normalizeRule(a));
                delete i.messages;
                h[f.name] = i;
                if (a.messages) {
                    c.messages[f.name] = $.extend(c.messages[f.name], a.messages)
                }
                break;
            case"remove":
                if (!a) {
                    delete h[f.name];
                    return i
                }
                g = {};
                $.each(a.split(/\s/), function(j, k) {
                    g[k] = i[k];
                    delete i[k];
                    if (k === "required") {
                        $(f).removeAttr("aria-required")
                    }
                });
                return g
            }
        }
        e = $.validator.normalizeRules($.extend({}, $.validator.classRules(f), $.validator.attributeRules(f), $.validator.dataRules(f), $.validator.staticRules(f)), f);
        if (e.required) {
            b = e.required;
            delete e.required;
            e = $.extend({
                required: b
            }, e);
            $(f).attr("aria-required", "true")
        }
        if (e.remote) {
            b = e.remote;
            delete e.remote;
            e = $.extend(e, {
                remote: b
            })
        }
        return e
    }
});
$.extend($.expr[":"], {
    blank: function(b) {
        return !$.trim("" + $(b).val())
    },
    filled: function(b) {
        return !!$.trim("" + $(b).val())
    },
    unchecked: function(b) {
        return !$(b).prop("checked")
    }
});
$.validator = function(a, b) {
    this.settings = $.extend(true, {}, $.validator.defaults, a);
    this.currentForm = b;
    this.init()
};
$.validator.format = function(a, b) {
    if (arguments.length === 1) {
        return function() {
            var c = $.makeArray(arguments);
            c.unshift(a);
            return $.validator.format.apply(this, c)
        }
    }
    if (arguments.length > 2 && b.constructor !== Array) {
        b = $.makeArray(arguments).slice(1)
    }
    if (b.constructor !== Array) {
        b = [b]
    }
    $.each(b, function(c, d) {
        a = a.replace(new RegExp("\\{" + c + "\\}", "g"), function() {
            return d
        })
    });
    return a
};
$.extend($.validator, {
    defaults: {
        messages: {},
        groups: {},
        rules: {},
        errorClass: "error",
        validClass: "valid",
        errorElement: "label",
        focusInvalid: true,
        errorContainer: $([]),
        errorLabelContainer: $([]),
        onsubmit: true,
        ignore: ":hidden",
        ignoreTitle: false,
        onfocusin: function(a) {
            this.lastActive = a;
            if (this.settings.focusCleanup&&!this.blockFocusCleanup) {
                if (this.settings.unhighlight) {
                    this.settings.unhighlight.call(this, a, this.settings.errorClass, this.settings.validClass)
                }
                this.hideThese(this.errorsFor(a))
            }
        },
        onfocusout: function(a) {
            if (!this.checkable(a) && (a.name in this.submitted ||!this.optional(a))) {
                this.element(a)
            }
        },
        onkeyup: function(a, b) {
            if (b.which === 9 && this.elementValue(a) === "") {
                return 
            } else {
                if (a.name in this.submitted || a === this.lastElement) {
                    this.element(a)
                }
            }
        },
        onclick: function(a) {
            if (a.name in this.submitted) {
                this.element(a)
            } else {
                if (a.parentNode.name in this.submitted) {
                    this.element(a.parentNode)
                }
            }
        },
        highlight: function(c, a, b) {
            if (c.type === "radio") {
                this.findByName(c.name).addClass(a).removeClass(b)
            } else {
                $(c).addClass(a).removeClass(b)
            }
        },
        unhighlight: function(c, a, b) {
            if (c.type === "radio") {
                this.findByName(c.name).removeClass(a).addClass(b)
            } else {
                $(c).removeClass(a).addClass(b)
            }
        }
    },
    setDefaults: function(a) {
        $.extend($.validator.defaults, a)
    },
    messages: {
        required: "This field is required.",
        remote: "Please fix this field.",
        email: "Please enter a valid email address.",
        url: "Please enter a valid URL.",
        date: "Please enter a valid date.",
        dateISO: "Please enter a valid date ( ISO ).",
        number: "Please enter a valid number.",
        digits: "Please enter only digits.",
        creditcard: "Please enter a valid credit card number.",
        equalTo: "Please enter the same value again.",
        maxlength: $.validator.format("Please enter no more than {0} characters."),
        minlength: $.validator.format("Please enter at least {0} characters."),
        rangelength: $.validator.format("Please enter a value between {0} and {1} characters long."),
        range: $.validator.format("Please enter a value between {0} and {1}."),
        max: $.validator.format("Please enter a value less than or equal to {0}."),
        min: $.validator.format("Please enter a value greater than or equal to {0}.")
    },
    autoCreateRanges: false,
    prototype: {
        init: function() {
            this.labelContainer = $(this.settings.errorLabelContainer);
            this.errorContext = this.labelContainer.length && this.labelContainer || $(this.currentForm);
            this.containers = $(this.settings.errorContainer).add(this.settings.errorLabelContainer);
            this.submitted = {};
            this.valueCache = {};
            this.pendingRequest = 0;
            this.pending = {};
            this.invalid = {};
            this.reset();
            var a = (this.groups = {}), c;
            $.each(this.settings.groups, function(d, e) {
                if (typeof e === "string") {
                    e = e.split(/\s/)
                }
                $.each(e, function(g, f) {
                    a[f] = d
                })
            });
            c = this.settings.rules;
            $.each(c, function(d, e) {
                c[d] = $.validator.normalizeRule(e)
            });
            function b(g) {
                var e = $.data(this[0].form, "validator"), d = "on" + g.type.replace(/^validate/, ""), f = e.settings;
                if (f[d]&&!this.is(f.ignore)) {
                    f[d].call(e, this[0], g)
                }
            }
            $(this.currentForm).validateDelegate(":text, [type='password'], [type='file'], select, textarea, [type='number'], [type='search'] ,[type='tel'], [type='url'], [type='email'], [type='datetime'], [type='date'], [type='month'], [type='week'], [type='time'], [type='datetime-local'], [type='range'], [type='color'], [type='radio'], [type='checkbox']", "focusin focusout keyup", b).validateDelegate("select, option, [type='radio'], [type='checkbox']", "click", b);
            if (this.settings.invalidHandler) {
                $(this.currentForm).bind("invalid-form.validate", this.settings.invalidHandler)
            }
            $(this.currentForm).find("[required], [data-rule-required], .required").attr("aria-required", "true")
        },
        form: function() {
            this.checkForm();
            $.extend(this.submitted, this.errorMap);
            this.invalid = $.extend({}, this.errorMap);
            if (!this.valid()) {
                $(this.currentForm).triggerHandler("invalid-form", [this])
            }
            this.showErrors();
            return this.valid()
        },
        checkForm: function() {
            this.prepareForm();
            for (var a = 0, b = (this.currentElements = this.elements()); b[a]; a++) {
                this.check(b[a])
            }
            return this.valid()
        },
        element: function(c) {
            var d = this.clean(c), b = this.validationTargetFor(d), a = true;
            this.lastElement = b;
            if (b === undefined) {
                delete this.invalid[d.name]
            } else {
                this.prepareElement(b);
                this.currentElements = $(b);
                a = this.check(b) !== false;
                if (a) {
                    delete this.invalid[b.name]
                } else {
                    this.invalid[b.name] = true
                }
            }
            $(c).attr("aria-invalid", !a);
            if (!this.numberOfInvalids()) {
                this.toHide = this.toHide.add(this.containers)
            }
            this.showErrors();
            return a
        },
        showErrors: function(b) {
            if (b) {
                $.extend(this.errorMap, b);
                this.errorList = [];
                for (var a in b) {
                    this.errorList.push({
                        message: b[a],
                        element: this.findByName(a)[0]
                    })
                }
                this.successList = $.grep(this.successList, function(c) {
                    return !(c.name in b)
                })
            }
            if (this.settings.showErrors) {
                this.settings.showErrors.call(this, this.errorMap, this.errorList)
            } else {
                this.defaultShowErrors()
            }
        },
        resetForm: function() {
            if ($.fn.resetForm) {
                $(this.currentForm).resetForm()
            }
            this.submitted = {};
            this.lastElement = null;
            this.prepareForm();
            this.hideErrors();
            this.elements().removeClass(this.settings.errorClass).removeData("previousValue").removeAttr("aria-invalid")
        },
        numberOfInvalids: function() {
            return this.objectLength(this.invalid)
        },
        objectLength: function(c) {
            var b = 0, a;
            for (a in c) {
                b++
            }
            return b
        },
        hideErrors: function() {
            this.hideThese(this.toHide)
        },
        hideThese: function(a) {
            a.not(this.containers).text("");
            this.addWrapper(a).hide()
        },
        valid: function() {
            return this.size() === 0
        },
        size: function() {
            return this.errorList.length
        },
        focusInvalid: function() {
            if (this.settings.focusInvalid) {
                try {
                    $(this.findLastActive() || this.errorList.length && this.errorList[0].element || []).filter(":visible").focus().trigger("focusin")
                } catch (a) {}
            }
        },
        findLastActive: function() {
            var a = this.lastActive;
            return a && $.grep(this.errorList, function(b) {
                return b.element.name === a.name
            }).length === 1 && a
        },
        elements: function() {
            var b = this, a = {};
            return $(this.currentForm).find("input, select, textarea").not(":submit, :reset, :image, [disabled]").not(this.settings.ignore).filter(function() {
                if (!this.name && b.settings.debug && window.console) {
                    console.error("%o has no name assigned", this)
                }
                if (this.name in a ||!b.objectLength($(this).rules())) {
                    return false
                }
                a[this.name] = true;
                return true
            })
        },
        clean: function(a) {
            return $(a)[0]
        },
        errors: function() {
            var a = this.settings.errorClass.split(" ").join(".");
            return $(this.settings.errorElement + "." + a, this.errorContext)
        },
        reset: function() {
            this.successList = [];
            this.errorList = [];
            this.errorMap = {};
            this.toShow = $([]);
            this.toHide = $([]);
            this.currentElements = $([])
        },
        prepareForm: function() {
            this.reset();
            this.toHide = this.errors().add(this.containers)
        },
        prepareElement: function(a) {
            this.reset();
            this.toHide = this.errorsFor(a)
        },
        elementValue: function(b) {
            var d, a = $(b), c = b.type;
            if (c === "radio" || c === "checkbox") {
                return $("input[name='" + b.name + "']:checked").val()
            } else {
                if (c === "number" && typeof b.validity !== "undefined") {
                    return b.validity.badInput ? false : a.val()
                }
            }
            d = a.val();
            if (typeof d === "string") {
                return d.replace(/\r/g, "")
            }
            return d
        },
        check: function(c) {
            c = this.validationTargetFor(this.clean(c));
            var h = $(c).rules(), f = $.map(h, function(k, e) {
                return e
            }).length, i = false, b = this.elementValue(c), j, a, g;
            for (a in h) {
                g = {
                    method: a,
                    parameters: h[a]
                };
                try {
                    j = $.validator.methods[a].call(this, b, c, g.parameters);
                    if (j === "dependency-mismatch" && f === 1) {
                        i = true;
                        continue
                    }
                    i = false;
                    if (j === "pending") {
                        this.toHide = this.toHide.not(this.errorsFor(c));
                        return 
                    }
                    if (!j) {
                        this.formatAndAdd(c, g);
                        return false
                    }
                } catch (d) {
                    if (this.settings.debug && window.console) {
                        console.log("Exception occurred when checking element " + c.id + ", check the '" + g.method + "' method.", d)
                    }
                    throw d
                }
            }
            if (i) {
                return 
            }
            if (this.objectLength(h)) {
                this.successList.push(c)
            }
            return true
        },
        customDataMessage: function(a, b) {
            return $(a).data("msg" + b.charAt(0).toUpperCase() + b.substring(1).toLowerCase()) || $(a).data("msg")
        },
        customMessage: function(b, c) {
            var a = this.settings.messages[b];
            return a && (a.constructor === String ? a : a[c])
        },
        findDefined: function() {
            for (var a = 0; a < arguments.length; a++) {
                if (arguments[a] !== undefined) {
                    return arguments[a]
                }
            }
            return undefined
        },
        defaultMessage: function(a, b) {
            return this.findDefined(this.customMessage(a.name, b), this.customDataMessage(a, b), !this.settings.ignoreTitle && a.title || undefined, $.validator.messages[b], "<strong>Warning: No message defined for " + a.name + "</strong>")
        },
        formatAndAdd: function(b, d) {
            var c = this.defaultMessage(b, d.method), a = /\$?\{(\d+)\}/g;
            if (typeof c === "function") {
                c = c.call(this, d.parameters, b)
            } else {
                if (a.test(c)) {
                    c = $.validator.format(c.replace(a, "{$1}"), d.parameters)
                }
            }
            this.errorList.push({
                message: c,
                element: b,
                method: d.method
            });
            this.errorMap[b.name] = c;
            this.submitted[b.name] = c
        },
        addWrapper: function(a) {
            if (this.settings.wrapper) {
                a = a.add(a.parent(this.settings.wrapper))
            }
            return a
        },
        defaultShowErrors: function() {
            var b, c, a;
            for (b = 0; this.errorList[b]; b++) {
                a = this.errorList[b];
                if (this.settings.highlight) {
                    this.settings.highlight.call(this, a.element, this.settings.errorClass, this.settings.validClass)
                }
                this.showLabel(a.element, a.message)
            }
            if (this.errorList.length) {
                this.toShow = this.toShow.add(this.containers)
            }
            if (this.settings.success) {
                for (b = 0; this.successList[b]; b++) {
                    this.showLabel(this.successList[b])
                }
            }
            if (this.settings.unhighlight) {
                for (b = 0, c = this.validElements(); c[b]; b++) {
                    this.settings.unhighlight.call(this, c[b], this.settings.errorClass, this.settings.validClass)
                }
            }
            this.toHide = this.toHide.not(this.toShow);
            this.hideErrors();
            this.addWrapper(this.toShow).show()
        },
        validElements: function() {
            return this.currentElements.not(this.invalidElements())
        },
        invalidElements: function() {
            return $(this.errorList).map(function() {
                return this.element
            })
        },
        showLabel: function(e, f) {
            var b, h, d, c = this.errorsFor(e), a = this.idOrName(e), g = $(e).attr("aria-describedby");
            if (c.length) {
                c.removeClass(this.settings.validClass).addClass(this.settings.errorClass);
                c.html(f)
            } else {
                c = $("<" + this.settings.errorElement + ">").attr("id", a + "-error").addClass(this.settings.errorClass).html(f || "");
                b = c;
                if (this.settings.wrapper) {
                    b = c.hide().show().wrap("<" + this.settings.wrapper + "/>").parent()
                }
                if (this.labelContainer.length) {
                    this.labelContainer.append(b)
                } else {
                    if (this.settings.errorPlacement) {
                        this.settings.errorPlacement(b, $(e))
                    } else {
                        b.insertAfter(e)
                    }
                }
                if (c.is("label")) {
                    c.attr("for", a)
                } else {
                    if (c.parents("label[for='" + a + "']").length === 0) {
                        d = c.attr("id");
                        if (!g) {
                            g = d
                        } else {
                            if (!g.match(new RegExp("\\b" + d + "\\b"))) {
                                g += " " + d
                            }
                        }
                        $(e).attr("aria-describedby", g);
                        h = this.groups[e.name];
                        if (h) {
                            $.each(this.groups, function(j, i) {
                                if (i === h) {
                                    $("[name='" + j + "']", this.currentForm).attr("aria-describedby", c.attr("id"))
                                }
                            })
                        }
                    }
                }
            }
            if (!f && this.settings.success) {
                c.text("");
                if (typeof this.settings.success === "string") {
                    c.addClass(this.settings.success)
                } else {
                    this.settings.success(c, e)
                }
            }
            this.toShow = this.toShow.add(c)
        },
        errorsFor: function(c) {
            var b = this.idOrName(c), d = $(c).attr("aria-describedby"), a = "label[for='" + b + "'], label[for='" + b + "'] *";
            if (d) {
                a = a + ", #" + d.replace(/\s+/g, ", #")
            }
            return this.errors().filter(a)
        },
        idOrName: function(a) {
            return this.groups[a.name] || (this.checkable(a) ? a.name : a.id || a.name)
        },
        validationTargetFor: function(a) {
            if (this.checkable(a)) {
                a = this.findByName(a.name).not(this.settings.ignore)[0]
            }
            return a
        },
        checkable: function(a) {
            return (/radio|checkbox/i).test(a.type)
        },
        findByName: function(a) {
            return $(this.currentForm).find("[name='" + a + "']")
        },
        getLength: function(b, a) {
            switch (a.nodeName.toLowerCase()) {
            case"select":
                return $("option:selected", a).length;
            case"input":
                if (this.checkable(a)) {
                    return this.findByName(a.name).filter(":checked").length
                }
            }
            return b.length
        },
        depend: function(b, a) {
            return this.dependTypes[typeof b] ? this.dependTypes[typeof b](b, a) : true
        },
        dependTypes: {
            "boolean": function(a) {
                return a
            },
            string: function(b, a) {
                return !!$(b, a.form).length
            },
            "function": function(b, a) {
                return b(a)
            }
        },
        optional: function(a) {
            var b = this.elementValue(a);
            return !$.validator.methods.required.call(this, b, a) && "dependency-mismatch"
        },
        startRequest: function(a) {
            if (!this.pending[a.name]) {
                this.pendingRequest++;
                this.pending[a.name] = true
            }
        },
        stopRequest: function(a, b) {
            this.pendingRequest--;
            if (this.pendingRequest < 0) {
                this.pendingRequest = 0
            }
            delete this.pending[a.name];
            if (b && this.pendingRequest === 0 && this.formSubmitted && this.form()) {
                $(this.currentForm).submit();
                this.formSubmitted = false
            } else {
                if (!b && this.pendingRequest === 0 && this.formSubmitted) {
                    $(this.currentForm).triggerHandler("invalid-form", [this]);
                    this.formSubmitted = false
                }
            }
        },
        previousValue: function(a) {
            return $.data(a, "previousValue") || $.data(a, "previousValue", {
                old: null,
                valid: true,
                message: this.defaultMessage(a, "remote")
            })
        }
    },
    classRuleSettings: {
        required: {
            required: true
        },
        email: {
            email: true
        },
        url: {
            url: true
        },
        date: {
            date: true
        },
        dateISO: {
            dateISO: true
        },
        number: {
            number: true
        },
        digits: {
            digits: true
        },
        creditcard: {
            creditcard: true
        }
    },
    addClassRules: function(a, b) {
        if (a.constructor === String) {
            this.classRuleSettings[a] = b
        } else {
            $.extend(this.classRuleSettings, a)
        }
    },
    classRules: function(b) {
        var c = {}, a = $(b).attr("class");
        if (a) {
            $.each(a.split(" "), function() {
                if (this in $.validator.classRuleSettings) {
                    $.extend(c, $.validator.classRuleSettings[this])
                }
            })
        }
        return c
    },
    attributeRules: function(b) {
        var e = {}, a = $(b), c = b.getAttribute("type"), f, d;
        for (f in $.validator.methods) {
            if (f === "required") {
                d = b.getAttribute(f);
                if (d === "") {
                    d = true
                }
                d=!!d
            } else {
                d = a.attr(f)
            }
            if (/min|max/.test(f) && (c === null || /number|range|text/.test(c))) {
                d = Number(d)
            }
            if (d || d === 0) {
                e[f] = d
            } else {
                if (c === f && c !== "range") {
                    e[f] = true
                }
            }
        }
        if (e.maxlength && /-1|2147483647|524288/.test(e.maxlength)) {
            delete e.maxlength
        }
        return e
    },
    dataRules: function(b) {
        var e, c, d = {}, a = $(b);
        for (e in $.validator.methods) {
            c = a.data("rule" + e.charAt(0).toUpperCase() + e.substring(1).toLowerCase());
            if (c !== undefined) {
                d[e] = c
            }
        }
        return d
    },
    staticRules: function(b) {
        var c = {}, a = $.data(b.form, "validator");
        if (a.settings.rules) {
            c = $.validator.normalizeRule(a.settings.rules[b.name]) || {}
        }
        return c
    },
    normalizeRules: function(b, a) {
        $.each(b, function(e, d) {
            if (d === false) {
                delete b[e];
                return 
            }
            if (d.param || d.depends) {
                var c = true;
                switch (typeof d.depends) {
                case"string":
                    c=!!$(d.depends, a.form).length;
                    break;
                case"function":
                    c = d.depends.call(a, a);
                    break
                }
                if (c) {
                    b[e] = d.param !== undefined ? d.param : true
                } else {
                    delete b[e]
                }
            }
        });
        $.each(b, function(c, d) {
            b[c] = $.isFunction(d) ? d(a) : d
        });
        $.each(["minlength", "maxlength"], function() {
            if (b[this]) {
                b[this] = Number(b[this])
            }
        });
        $.each(["rangelength", "range"], function() {
            var c;
            if (b[this]) {
                if ($.isArray(b[this])) {
                    b[this] = [Number(b[this][0]), Number(b[this][1])]
                } else {
                    if (typeof b[this] === "string") {
                        c = b[this].replace(/[\[\]]/g, "").split(/[\s,]+/);
                        b[this] = [Number(c[0]), Number(c[1])]
                    }
                }
            }
        });
        if ($.validator.autoCreateRanges) {
            if (b.min && b.max) {
                b.range = [b.min, b.max];
                delete b.min;
                delete b.max
            }
            if (b.minlength && b.maxlength) {
                b.rangelength = [b.minlength, b.maxlength];
                delete b.minlength;
                delete b.maxlength
            }
        }
        return b
    },
    normalizeRule: function(b) {
        if (typeof b === "string") {
            var a = {};
            $.each(b.split(/\s/), function() {
                a[this] = true
            });
            b = a
        }
        return b
    },
    addMethod: function(a, c, b) {
        $.validator.methods[a] = c;
        $.validator.messages[a] = b !== undefined ? b : $.validator.messages[a];
        if (c.length < 3) {
            $.validator.addClassRules(a, $.validator.normalizeRule(a))
        }
    },
    methods: {
        required: function(b, a, d) {
            if (!this.depend(d, a)) {
                return "dependency-mismatch"
            }
            if (a.nodeName.toLowerCase() === "select") {
                var c = $(a).val();
                return c && c.length > 0
            }
            if (this.checkable(a)) {
                return this.getLength(b, a) > 0
            }
            return $.trim(b).length > 0
        },
        email: function(b, a) {
            return this.optional(a) || /^[a-zA-Z0-9.!#$%&'*+\/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/.test(b)
        },
        url: function(b, a) {
            return this.optional(a) || /^(https?|s?ftp):\/\/(((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:)*@)?(((\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]))|((([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?)(:\d*)?)(\/((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)+(\/(([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)*)*)?)?(\?((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|[\uE000-\uF8FF]|\/|\?)*)?(#((([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|\/|\?)*)?$/i.test(b)
        },
        date: function(b, a) {
            return this.optional(a) ||!/Invalid|NaN/.test(new Date(b).toString())
        },
        dateISO: function(b, a) {
            return this.optional(a) || /^\d{4}[\/\-](0?[1-9]|1[012])[\/\-](0?[1-9]|[12][0-9]|3[01])$/.test(b)
        },
        number: function(b, a) {
            return this.optional(a) || /^-?(?:\d+|\d{1,3}(?:,\d{3})+)?(?:\.\d+)?$/.test(b)
        },
        digits: function(b, a) {
            return this.optional(a) || /^\d+$/.test(b)
        },
        creditcard: function(e, b) {
            if (this.optional(b)) {
                return "dependency-mismatch"
            }
            if (/[^0-9 \-]+/.test(e)) {
                return false
            }
            var f = 0, d = 0, a = false, g, c;
            e = e.replace(/\D/g, "");
            if (e.length < 13 || e.length > 19) {
                return false
            }
            for (g = e.length - 1; g >= 0; g--) {
                c = e.charAt(g);
                d = parseInt(c, 10);
                if (a) {
                    if ((d*=2) > 9) {
                        d -= 9
                    }
                }
                f += d;
                a=!a
            }
            return (f%10) === 0
        },
        minlength: function(c, a, d) {
            var b = $.isArray(c) ? c.length: this.getLength($.trim(c), a);
            return this.optional(a) || b >= d
        },
        maxlength: function(c, a, d) {
            var b = $.isArray(c) ? c.length: this.getLength($.trim(c), a);
            return this.optional(a) || b <= d
        },
        rangelength: function(c, a, d) {
            var b = $.isArray(c) ? c.length: this.getLength($.trim(c), a);
            return this.optional(a) || (b >= d[0] && b <= d[1])
        },
        min: function(b, a, c) {
            return this.optional(a) || b >= c
        },
        max: function(b, a, c) {
            return this.optional(a) || b <= c
        },
        range: function(b, a, c) {
            return this.optional(a) || (b >= c[0] && b <= c[1])
        },
        equalTo: function(b, a, d) {
            var c = $(d);
            if (this.settings.onfocusout) {
                c.unbind(".validate-equalTo").bind("blur.validate-equalTo", function() {
                    $(a).valid()
                })
            }
            return b === c.val()
        },
        remote: function(e, b, f) {
            if (this.optional(b)) {
                return "dependency-mismatch"
            }
            var c = this.previousValue(b), a, d;
            if (!this.settings.messages[b.name]) {
                this.settings.messages[b.name] = {}
            }
            c.originalMessage = this.settings.messages[b.name].remote;
            this.settings.messages[b.name].remote = c.message;
            f = typeof f === "string" && {
                url: f
            }
            || f;
            if (c.old === e) {
                return c.valid
            }
            c.old = e;
            a = this;
            this.startRequest(b);
            d = {};
            d[b.name] = e;
            $.ajax($.extend(true, {
                url: f,
                mode: "abort",
                port: "validate" + b.name,
                dataType: "json",
                data: d,
                context: a.currentForm,
                success: function(h) {
                    var j = h === true || h === "true", k, i, g;
                    a.settings.messages[b.name].remote = c.originalMessage;
                    if (j) {
                        g = a.formSubmitted;
                        a.prepareElement(b);
                        a.formSubmitted = g;
                        a.successList.push(b);
                        delete a.invalid[b.name];
                        a.showErrors()
                    } else {
                        k = {};
                        i = h || a.defaultMessage(b, "remote");
                        k[b.name] = c.message = $.isFunction(i) ? i(e) : i;
                        a.invalid[b.name] = true;
                        a.showErrors(k)
                    }
                    c.valid = j;
                    a.stopRequest(b, j)
                }
            }, f));
            return "pending"
        }
    }
});
$.format = function deprecated() {
    throw "$.format has been deprecated. Please use $.validator.format instead."
};
var pendingRequests = {}, ajax;
if ($.ajaxPrefilter) {
    $.ajaxPrefilter(function(c, b, d) {
        var a = c.port;
        if (c.mode === "abort") {
            if (pendingRequests[a]) {
                pendingRequests[a].abort()
            }
            pendingRequests[a] = d
        }
    })
} else {
    ajax = $.ajax;
    $.ajax = function(b) {
        var c = ("mode" in b ? b : $.ajaxSettings).mode, a = ("port" in b ? b : $.ajaxSettings).port;
        if (c === "abort") {
            if (pendingRequests[a]) {
                pendingRequests[a].abort()
            }
            pendingRequests[a] = ajax.apply(this, arguments);
            return pendingRequests[a]
        }
        return ajax.apply(this, arguments)
    }
}
$.extend($.fn, {
    validateDelegate: function(c, b, a) {
        return this.bind(b, function(d) {
            var e = $(d.target);
            if (e.is(c)) {
                return a.apply(e, arguments)
            }
        })
    }
});
(function(b, a, c) {
    (function(d) {
        if (typeof define === "function" && define.amd) {
            define(["jquery"], d)
        } else {
            if (jQuery&&!jQuery.fn.qtip) {
                d(jQuery)
            }
        }
    }(function(K) {
        var B = true, ah = false, F = null, e = "x", d = "y", i = "width", J = "height", L = "top", z = "left", I = "bottom", ai = "right", H = "center", q = "flip", U = "flipinvert", O = "shift", Z, S, n, f, D = {}, j = "qtip", R = "data-hasqtip", N = "data-qtip-id", p = ["ui-widget", "ui-tooltip"], l = "." + j, x = "click dblclick mousedown mouseup mousemove mouseleave mouseenter".split(" "), aa = j + "-fixed", h = j + "-default", v = j + "-focus", ab = j + "-hover", P = j + "-disabled", u = "_replacedByqTip", o = "oldtitle", ae, G = {
            ie: (function() {
                var C = 3, X = a.createElement("div");
                while ((X.innerHTML = "<!--[if gt IE " + (++C) + "]><i></i><![endif]-->")) {
                    if (!X.getElementsByTagName("i")[0]) {
                        break
                    }
                }
                return C > 4 ? C : NaN
            }()),
            iOS: parseFloat(("" + (/CPU.*OS ([0-9_]{1,5})|(CPU like).*AppleWebKit.*Mobile/i.exec(navigator.userAgent) || [0, ""])[1]).replace("undefined", "3_2").replace("_", ".").replace("_", "")) || ah
        };
        function g(Y, X, aj, C) {
            this.id = aj;
            this.target = Y;
            this.tooltip = F;
            this.elements = {
                target: Y
            };
            this._id = j + "-" + aj;
            this.timers = {
                img: {}
            };
            this.options = X;
            this.plugins = {};
            this.cache = {
                event: {},
                target: K(),
                disabled: ah,
                attr: C,
                onTooltip: ah,
                lastClass: ""
            };
            this.rendered = this.destroyed = this.disabled = this.waiting = this.hiddenDuringWait = this.positioning = this.triggering = ah
        }
        S = g.prototype;
        S._when = function(C) {
            return K.when.apply(K, C)
        };
        S.render = function(al) {
            if (this.rendered || this.destroyed) {
                return this
            }
            var ao = this, ar = this.options, X = this.cache, C = this.elements, an = ar.content.text, ak = ar.content.title, aj = ar.content.button, am = ar.position, Y = "." + this._id + " ", aq = [], ap;
            K.attr(this.target[0], "aria-describedby", this._id);
            this.tooltip = C.tooltip = ap = K("<div/>", {
                id: this._id,
                "class": [j, h, ar.style.classes, j + "-pos-" + ar.position.my.abbrev()].join(" "),
                width: ar.style.width || "",
                height: ar.style.height || "",
                tracking: am.target === "mouse" && am.adjust.mouse,
                role: "alert",
                "aria-live": "polite",
                "aria-atomic": ah,
                "aria-describedby": this._id + "-content",
                "aria-hidden": B
            }).toggleClass(P, this.disabled).attr(N, this.id).data(j, this).appendTo(am.container).append(C.content = K("<div />", {
                "class": j + "-content",
                id: this._id + "-content",
                "aria-atomic": B
            }));
            this.rendered =- 1;
            this.positioning = B;
            if (ak) {
                this._createTitle();
                if (!K.isFunction(ak)) {
                    aq.push(this._updateTitle(ak, ah))
                }
            }
            if (aj) {
                this._createButton()
            }
            if (!K.isFunction(an)) {
                aq.push(this._updateContent(an, ah))
            }
            this.rendered = B;
            this._setWidget();
            K.each(D, function(au) {
                var at;
                if (this.initialize === "render" && (at = this (ao))) {
                    ao.plugins[au] = at
                }
            });
            this._unassignEvents();
            this._assignEvents();
            this._when(aq).then(function() {
                ao._trigger("render");
                ao.positioning = ah;
                if (!ao.hiddenDuringWait && (ar.show.ready || al)) {
                    ao.toggle(B, X.event, ah)
                }
                ao.hiddenDuringWait = ah
            });
            Z.api[this.id] = this;
            return this
        };
        S.destroy = function(C) {
            if (this.destroyed) {
                return this.target
            }
            function X() {
                if (this.destroyed) {
                    return 
                }
                this.destroyed = B;
                var Y = this.target, aj = Y.attr(o);
                if (this.rendered) {
                    this.tooltip.stop(1, 0).find("*").remove().end().remove()
                }
                K.each(this.plugins, function(ak) {
                    this.destroy && this.destroy()
                });
                clearTimeout(this.timers.show);
                clearTimeout(this.timers.hide);
                this._unassignEvents();
                Y.removeData(j).removeAttr(N).removeAttr(R).removeAttr("aria-describedby");
                if (this.options.suppress && aj) {
                    Y.attr("title", aj).removeAttr(o)
                }
                this._unbind(Y);
                this.options = this.elements = this.cache = this.timers = this.plugins = this.mouse = F;
                delete Z.api[this.id]
            }
            if ((C !== B || this.triggering === "hide") && this.rendered) {
                this.tooltip.one("tooltiphidden", K.proxy(X, this));
                !this.triggering && this.hide()
            } else {
                X.call(this)
            }
            return this.target
        };
        function s(C) {
            return C === F || K.type(C) !== "object"
        }
        function Q(C) {
            return !(K.isFunction(C) || (C && C.attr) || C.length || (K.type(C) === "object" && (C.jquery || C.then)))
        }
        function A(Y) {
            var X, ak, aj, C;
            if (s(Y)) {
                return ah
            }
            if (s(Y.metadata)) {
                Y.metadata = {
                    type: Y.metadata
                }
            }
            if ("content" in Y) {
                X = Y.content;
                if (s(X) || X.jquery || X.done) {
                    X = Y.content = {
                        text: (ak = Q(X) ? ah : X)
                    }
                } else {
                    ak = X.text
                }
                if ("ajax" in X) {
                    aj = X.ajax;
                    C = aj && aj.once !== ah;
                    delete X.ajax;
                    X.text = function(an, am) {
                        var ao = ak || K(this).attr(am.options.content.attr) || "Loading...", al = K.ajax(K.extend({}, aj, {
                            context: am
                        })).then(aj.success, F, aj.error).then(function(ap) {
                            if (ap && C) {
                                am.set("content.text", ap)
                            }
                            return ap
                        }, function(ar, ap, aq) {
                            if (am.destroyed || ar.status === 0) {
                                return 
                            }
                            am.set("content.text", ap + ": " + aq)
                        });
                        return !C ? (am.set("content.text", ao), al) : ao
                    }
                }
                if ("title" in X) {
                    if (!s(X.title)) {
                        X.button = X.title.button;
                        X.title = X.title.text
                    }
                    if (Q(X.title || ah)) {
                        X.title = ah
                    }
                }
            }
            if ("position" in Y && s(Y.position)) {
                Y.position = {
                    my: Y.position,
                    at: Y.position
                }
            }
            if ("show" in Y && s(Y.show)) {
                Y.show = Y.show.jquery ? {
                    target: Y.show
                } : Y.show === B ? {
                    ready: B
                } : {
                    event: Y.show
                }
            }
            if ("hide" in Y && s(Y.hide)) {
                Y.hide = Y.hide.jquery ? {
                    target: Y.hide
                } : {
                    event: Y.hide
                }
            }
            if ("style" in Y && s(Y.style)) {
                Y.style = {
                    classes: Y.style
                }
            }
            K.each(D, function() {
                this.sanitize && this.sanitize(Y)
            });
            return Y
        }
        f = S.checks = {
            builtin: {
                "^id$": function(aj, ak, X, Y) {
                    var al = X === B ? Z.nextid: X, C = j + "-" + al;
                    if (al !== ah && al.length > 0&&!K("#" + C).length) {
                        this._id = C;
                        if (this.rendered) {
                            this.tooltip[0].id = this._id;
                            this.elements.content[0].id = this._id + "-content";
                            this.elements.title[0].id = this._id + "-title"
                        }
                    } else {
                        aj[ak] = Y
                    }
                },
                "^prerender": function(X, Y, C) {
                    C&&!this.rendered && this.render(this.options.show.ready)
                },
                "^content.text$": function(X, Y, C) {
                    this._updateContent(C)
                },
                "^content.attr$": function(Y, aj, C, X) {
                    if (this.options.content.text === this.target.attr(X)) {
                        this._updateContent(this.target.attr(C))
                    }
                },
                "^content.title$": function(X, Y, C) {
                    if (!C) {
                        return this._removeTitle()
                    }
                    C&&!this.elements.title && this._createTitle();
                    this._updateTitle(C)
                },
                "^content.button$": function(X, Y, C) {
                    this._updateButton(C)
                },
                "^content.title.(text|button)$": function(X, Y, C) {
                    this.set("content." + Y, C)
                },
                "^position.(my|at)$": function(X, Y, C) {
                    "string" === typeof C && (X[Y] = new n(C, Y === "at"))
                },
                "^position.container$": function(X, Y, C) {
                    this.rendered && this.tooltip.appendTo(C)
                },
                "^show.ready$": function(X, Y, C) {
                    C && (!this.rendered && this.render(B) || this.toggle(B))
                },
                "^style.classes$": function(Y, aj, C, X) {
                    this.rendered && this.tooltip.removeClass(X).addClass(C)
                },
                "^style.(width|height)": function(X, Y, C) {
                    this.rendered && this.tooltip.css(Y, C)
                },
                "^style.widget|content.title": function() {
                    this.rendered && this._setWidget()
                },
                "^style.def": function(X, Y, C) {
                    this.rendered && this.tooltip.toggleClass(h, !!C)
                },
                "^events.(render|show|move|hide|focus|blur)$": function(X, Y, C) {
                    this.rendered && this.tooltip[(K.isFunction(C) ? "" : "un") + "bind"]("tooltip" + Y, C)
                },
                "^(show|hide|position).(event|target|fixed|inactive|leave|distance|viewport|adjust)": function() {
                    if (!this.rendered) {
                        return 
                    }
                    var C = this.options.position;
                    this.tooltip.attr("tracking", C.target === "mouse" && C.adjust.mouse);
                    this._unassignEvents();
                    this._assignEvents()
                }
            }
        };
        function m(C, aj) {
            var X = 0, al, Y = C, ak = aj.split(".");
            while (Y = Y[ak[X++]]) {
                if (X < ak.length) {
                    al = Y
                }
            }
            return [al || C, ak.pop()]
        }
        S.get = function(X) {
            if (this.destroyed) {
                return this
            }
            var Y = m(this.options, X.toLowerCase()), C = Y[0][Y[1]];
            return C.precedance ? C.string() : C
        };
        function W(aj, X) {
            var Y, ak, C;
            for (Y in this.checks) {
                for (ak in this.checks[Y]) {
                    if (C = (new RegExp(ak, "i")).exec(aj)) {
                        X.push(C);
                        if (Y === "builtin" || this.plugins[Y]) {
                            this.checks[Y][ak].apply(this.plugins[Y] || this, X)
                        }
                    }
                }
            }
        }
        var k = /^position\.(my|at|adjust|target|container|viewport)|style|content|show\.ready/i, ag = /^prerender|show\.ready/i;
        S.set = function(ak, al) {
            if (this.destroyed) {
                return this
            }
            var am = this.rendered, C = ah, aj = this.options, Y = this.checks, X;
            if ("string" === typeof ak) {
                X = ak;
                ak = {};
                ak[X] = al
            } else {
                ak = K.extend({}, ak)
            }
            K.each(ak, function(ao, ap) {
                if (am && ag.test(ao)) {
                    delete ak[ao];
                    return 
                }
                var aq = m(aj, ao.toLowerCase()), an;
                an = aq[0][aq[1]];
                aq[0][aq[1]] = ap && ap.nodeType ? K(ap) : ap;
                C = k.test(ao) || C;
                ak[ao] = [aq[0], aq[1], ap, an]
            });
            A(aj);
            this.positioning = B;
            K.each(ak, K.proxy(W, this));
            this.positioning = ah;
            if (this.rendered && this.tooltip[0].offsetWidth > 0 && C) {
                this.reposition(aj.position.target === "mouse" ? F : this.cache.event)
            }
            return this
        };
        S._update = function(ak, aj, C) {
            var Y = this, X = this.cache;
            if (!this.rendered ||!ak) {
                return ah
            }
            if (K.isFunction(ak)) {
                ak = ak.call(this.elements.target, X.event, this) || ""
            }
            if (K.isFunction(ak.then)) {
                X.waiting = B;
                return ak.then(function(al) {
                    X.waiting = ah;
                    return Y._update(al, aj)
                }, F, function(al) {
                    return Y._update(al, aj)
                })
            }
            if (ak === ah || (!ak && ak !== "")) {
                return ah
            }
            if (ak.jquery && ak.length > 0) {
                aj.empty().append(ak.css({
                    display: "block",
                    visibility: "visible"
                }))
            } else {
                aj.html(ak)
            }
            return this._waitForContent(aj).then(function(al) {
                if (al.images && al.images.length && Y.rendered && Y.tooltip[0].offsetWidth > 0) {
                    Y.reposition(X.event, !al.length)
                }
            })
        };
        S._waitForContent = function(X) {
            var C = this.cache;
            C.waiting = B;
            return (K.fn.imagesLoaded ? X.imagesLoaded() : K.Deferred().resolve([])).done(function() {
                C.waiting = ah
            }).promise()
        };
        S._updateContent = function(X, C) {
            this._update(X, this.elements.content, C)
        };
        S._updateTitle = function(X, C) {
            if (this._update(X, this.elements.title, C) === ah) {
                this._removeTitle(ah)
            }
        };
        S._createTitle = function() {
            var C = this.elements, X = this._id + "-title";
            if (C.titlebar) {
                this._removeTitle()
            }
            C.titlebar = K("<div />", {
                "class": j + "-titlebar " + (this.options.style.widget ? y("header") : "")
            }).append(C.title = K("<div />", {
                id: X,
                "class": j + "-title",
                "aria-atomic": B
            })).insertBefore(C.content).delegate(".qtip-close", "mousedown keydown mouseup keyup mouseout", function(Y) {
                K(this).toggleClass("ui-state-active ui-state-focus", Y.type.substr( - 4) === "down")
            }).delegate(".qtip-close", "mouseover mouseout", function(Y) {
                K(this).toggleClass("ui-state-hover", Y.type === "mouseover")
            });
            if (this.options.content.button) {
                this._createButton()
            }
        };
        S._removeTitle = function(C) {
            var X = this.elements;
            if (X.title) {
                X.titlebar.remove();
                X.titlebar = X.title = X.button = F;
                if (C !== ah) {
                    this.reposition()
                }
            }
        };
        S.reposition = function(aA, ax) {
            if (!this.rendered || this.positioning || this.destroyed) {
                return this
            }
            this.positioning = B;
            var aw = this.cache, ak = this.tooltip, aD = this.options.position, aF = aD.target, au = aD.my, av = aD.at, aC = aD.viewport, aq = aD.container, az = aD.adjust, X = az.method.split(" "), aB = ak.outerWidth(ah), ay = ak.outerHeight(ah), am = 0, an = 0, Y = ak.css("position"), aE = {
                left: 0,
                top: 0
            }, C = ak[0].offsetWidth > 0, ar = aA && aA.type === "scroll", aj = K(b), aG = aq[0].ownerDocument, ap = this.mouse, ao, al;
            if (K.isArray(aF) && aF.length === 2) {
                av = {
                    x: z,
                    y: L
                };
                aE = {
                    left: aF[0],
                    top: aF[1]
                }
            } else {
                if (aF === "mouse") {
                    av = {
                        x: z,
                        y: L
                    };
                    if (ap && ap.pageX && (az.mouse ||!aA ||!aA.pageX)) {
                        aA = ap
                    } else {
                        if (!aA ||!aA.pageX) {
                            if ((!az.mouse || this.options.show.distance) && aw.origin && aw.origin.pageX) {
                                aA = aw.origin
                            } else {
                                if (!aA || (aA && (aA.type === "resize" || aA.type === "scroll"))) {
                                    aA = aw.event
                                }
                            }
                        }
                    }
                    if (Y !== "static") {
                        aE = aq.offset()
                    }
                    if (aG.body.offsetWidth !== (b.innerWidth || aG.documentElement.clientWidth)) {
                        al = K(a.body).offset()
                    }
                    aE = {
                        left: aA.pageX - aE.left + (al && al.left || 0),
                        top: aA.pageY - aE.top + (al && al.top || 0)
                    };
                    if (az.mouse && ar && ap) {
                        aE.left -= (ap.scrollX || 0) - aj.scrollLeft();
                        aE.top -= (ap.scrollY || 0) - aj.scrollTop()
                    }
                } else {
                    if (aF === "event") {
                        if (aA && aA.target && aA.type !== "scroll" && aA.type !== "resize") {
                            aw.target = K(aA.target)
                        } else {
                            if (!aA.target) {
                                aw.target = this.elements.target
                            }
                        }
                    } else {
                        if (aF !== "event") {
                            aw.target = K(aF.jquery ? aF : this.elements.target)
                        }
                    }
                    aF = aw.target;
                    aF = K(aF).eq(0);
                    if (aF.length === 0) {
                        return this
                    } else {
                        if (aF[0] === a || aF[0] === b) {
                            am = G.iOS ? b.innerWidth : aF.width();
                            an = G.iOS ? b.innerHeight : aF.height();
                            if (aF[0] === b) {
                                aE = {
                                    top: (aC || aF).scrollTop(),
                                    left: (aC || aF).scrollLeft()
                                }
                            }
                        } else {
                            if (D.imagemap && aF.is("area")) {
                                ao = D.imagemap(this, aF, av, D.viewport ? X : ah)
                            } else {
                                if (D.svg && aF && aF[0].ownerSVGElement) {
                                    ao = D.svg(this, aF, av, D.viewport ? X : ah)
                                } else {
                                    am = aF.outerWidth(ah);
                                    an = aF.outerHeight(ah);
                                    aE = aF.offset()
                                }
                            }
                        }
                    }
                    if (ao) {
                        am = ao.width;
                        an = ao.height;
                        al = ao.offset;
                        aE = ao.position
                    }
                    aE = this.reposition.offset(aF, aE, aq);
                    if ((G.iOS > 3.1 && G.iOS < 4.1) || (G.iOS >= 4.3 && G.iOS < 4.33) || (!G.iOS && Y === "fixed")) {
                        aE.left -= aj.scrollLeft();
                        aE.top -= aj.scrollTop()
                    }
                    if (!ao || (ao && ao.adjustable !== ah)) {
                        aE.left += av.x === ai ? am : av.x === H ? am / 2 : 0;
                        aE.top += av.y === I ? an : av.y === H ? an / 2 : 0
                    }
                }
            }
            aE.left += az.x + (au.x === ai?-aB : au.x === H?-aB / 2 : 0);
            aE.top += az.y + (au.y === I?-ay : au.y === H?-ay / 2 : 0);
            if (D.viewport) {
                aE.adjusted = D.viewport(this, aE, aD, am, an, aB, ay);
                if (al && aE.adjusted.left) {
                    aE.left += al.left
                }
                if (al && aE.adjusted.top) {
                    aE.top += al.top
                }
            } else {
                aE.adjusted = {
                    left: 0,
                    top: 0
                }
            }
            if (!this._trigger("move", [aE, aC.elem || aC], aA)) {
                return this
            }
            delete aE.adjusted;
            if (ax === ah ||!C || isNaN(aE.left) || isNaN(aE.top) || aF === "mouse" ||!K.isFunction(aD.effect)) {
                ak.css(aE)
            } else {
                if (K.isFunction(aD.effect)) {
                    aD.effect.call(ak, this, K.extend({}, aE));
                    ak.queue(function(at) {
                        K(this).css({
                            opacity: "",
                            height: ""
                        });
                        if (G.ie) {
                            this.style.removeAttribute("filter")
                        }
                        at()
                    })
                }
            }
            this.positioning = ah;
            return this
        };
        S.reposition.offset = function(aj, an, X) {
            if (!X[0]) {
                return an
            }
            var aq = K(aj[0].ownerDocument), am=!!G.ie && a.compatMode !== "CSS1Compat", ap = X[0], Y, al, C, ak;
            function ao(at, ar) {
                an.left += ar * at.scrollLeft();
                an.top += ar * at.scrollTop()
            }
            do {
                if ((al = K.css(ap, "position")) !== "static") {
                    if (al === "fixed") {
                        C = ap.getBoundingClientRect();
                        ao(aq, - 1)
                    } else {
                        C = K(ap).position();
                        C.left += (parseFloat(K.css(ap, "borderLeftWidth")) || 0);
                        C.top += (parseFloat(K.css(ap, "borderTopWidth")) || 0)
                    }
                    an.left -= C.left + (parseFloat(K.css(ap, "marginLeft")) || 0);
                    an.top -= C.top + (parseFloat(K.css(ap, "marginTop")) || 0);
                    if (!Y && (ak = K.css(ap, "overflow")) !== "hidden" && ak !== "visible") {
                        Y = K(ap)
                    }
                }
            }
            while ((ap = ap.offsetParent));
            if (Y && (Y[0] !== aq[0] || am)) {
                ao(Y, 1)
            }
            return an
        };
        var r = (n = S.reposition.Corner = function(X, C) {
            X = ("" + X).replace(/([A-Z])/, " $1").replace(/middle/gi, H).toLowerCase();
            this.x = (X.match(/left|right/i) || X.match(/center/) || ["inherit"])[0].toLowerCase();
            this.y = (X.match(/top|bottom|center/i) || ["inherit"])[0].toLowerCase();
            this.forceY=!!C;
            var Y = X.charAt(0);
            this.precedance = (Y === "t" || Y === "b" ? d : e)
        }).prototype;
        r.invert = function(X, C) {
            this[X] = this[X] === z ? ai : this[X] === ai ? z : C || this[X]
        };
        r.string = function() {
            var C = this.x, X = this.y;
            return C === X ? C : this.precedance === d || (this.forceY && X !== "center") ? X + " " + C : C + " " + X
        };
        r.abbrev = function() {
            var C = this.string().split(" ");
            return C[0].charAt(0) + (C[1] && C[1].charAt(0) || "")
        };
        r.clone = function() {
            return new n(this.string(), this.forceY)
        };
        S.toggle = function(am, av) {
            var at = this.cache, Y = this.options, al = this.tooltip;
            if (av) {
                if ((/over|enter/).test(av.type) && (/out|leave/).test(at.event.type) && Y.show.target.add(av.target).length === Y.show.target.length && al.has(av.relatedTarget).length) {
                    return this
                }
                at.event = w(av)
            }
            this.waiting&&!am && (this.hiddenDuringWait = B);
            if (!this.rendered) {
                return am ? this.render(1) : this
            } else {
                if (this.destroyed || this.disabled) {
                    return this
                }
            }
            var ak = am ? "show": "hide", ar = this.options[ak], aq = this.options[!am ? "show": "hide"], ax = this.options.position, ao = this.options.content, au = this.tooltip.css("width"), C = this.tooltip.is(":visible"), an = am || ar.target.length === 1, ap=!av || ar.target.length < 2 || at.target[0] === av.target, aw, az, aj, ay, X;
            if ((typeof am).search("boolean|number")) {
                am=!C
            }
            aw=!al.is(":animated") && C === am && ap;
            az=!aw?!!this._trigger(ak, [90]) : F;
            if (this.destroyed) {
                return this
            }
            if (az !== ah && am) {
                this.focus(av)
            }
            if (!az || aw) {
                return this
            }
            K.attr(al[0], "aria-hidden", !!!am);
            if (am) {
                at.origin = w(this.mouse);
                if (K.isFunction(ao.text)) {
                    this._updateContent(ao.text, ah)
                }
                if (K.isFunction(ao.title)) {
                    this._updateTitle(ao.title, ah)
                }
                if (!ae && ax.target === "mouse" && ax.adjust.mouse) {
                    K(a).bind("mousemove." + j, this._storeMouse);
                    ae = B
                }
                if (!au) {
                    al.css("width", al.outerWidth(ah))
                }
                this.reposition(av, arguments[2]);
                if (!au) {
                    al.css("width", "")
                }
                if (!!ar.solo) {
                    (typeof ar.solo === "string" ? K(ar.solo) : K(l, ar.solo)).not(al).not(ar.target).qtip("hide", K.Event("tooltipsolo"))
                }
            } else {
                clearTimeout(this.timers.show);
                delete at.origin;
                if (ae&&!K(l + '[tracking="true"]:visible', ar.solo).not(al).length) {
                    K(a).unbind("mousemove." + j);
                    ae = ah
                }
                this.blur(av)
            }
            X = K.proxy(function() {
                if (am) {
                    if (G.ie) {
                        al[0].style.removeAttribute("filter")
                    }
                    al.css("overflow", "");
                    if ("string" === typeof ar.autofocus) {
                        K(this.options.show.autofocus, al).focus()
                    }
                    this.options.show.target.trigger("qtip-" + this.id + "-inactive")
                } else {
                    al.css({
                        display: "",
                        visibility: "",
                        opacity: "",
                        left: "",
                        top: ""
                    })
                }
                this._trigger(am ? "visible" : "hidden")
            }, this);
            if (ar.effect === ah || an === ah) {
                al[ak]();
                X()
            } else {
                if (K.isFunction(ar.effect)) {
                    al.stop(1, 1);
                    ar.effect.call(al, this);
                    al.queue("fx", function(aA) {
                        X();
                        aA()
                    })
                } else {
                    al.fadeTo(90, am ? 1 : 0, X)
                }
            }
            if (am) {
                ar.target.trigger("qtip-" + this.id + "-inactive")
            }
            return this
        };
        S.show = function(C) {
            return this.toggle(B, C)
        };
        S.hide = function(C) {
            return this.toggle(ah, C)
        };
        S.focus = function(aj) {
            if (!this.rendered || this.destroyed) {
                return this
            }
            var al = K(l), ak = this.tooltip, Y = parseInt(ak[0].style.zIndex, 10), X = Z.zindex + al.length, C;
            if (!ak.hasClass(v)) {
                if (this._trigger("focus", [X], aj)) {
                    if (Y !== X) {
                        al.each(function() {
                            if (this.style.zIndex > Y) {
                                this.style.zIndex = this.style.zIndex - 1
                            }
                        });
                        al.filter("." + v).qtip("blur", aj)
                    }
                    ak.addClass(v)[0].style.zIndex = X
                }
            }
            return this
        };
        S.blur = function(C) {
            if (!this.rendered || this.destroyed) {
                return this
            }
            this.tooltip.removeClass(v);
            this._trigger("blur", [this.tooltip.css("zIndex")], C);
            return this
        };
        S.disable = function(C) {
            if (this.destroyed) {
                return this
            }
            if (C === "toggle") {
                C=!(this.rendered ? this.tooltip.hasClass(P) : this.disabled)
            } else {
                if ("boolean" !== typeof C) {
                    C = B
                }
            }
            if (this.rendered) {
                this.tooltip.toggleClass(P, C).attr("aria-disabled", C)
            }
            this.disabled=!!C;
            return this
        };
        S.enable = function() {
            return this.disable(ah)
        };
        S._createButton = function() {
            var X = this, ak = this.elements, aj = ak.tooltip, Y = this.options.content.button, C = typeof Y === "string", al = C ? Y: "Close tooltip";
            if (ak.button) {
                ak.button.remove()
            }
            if (Y.jquery) {
                ak.button = Y
            } else {
                ak.button = K("<a />", {
                    "class": "qtip-close " + (this.options.style.widget ? "" : j + "-icon"),
                    title: al,
                    "aria-label": al
                }).prepend(K("<span />", {
                    "class": "ui-icon ui-icon-close",
                    html: "&times;"
                }))
            }
            ak.button.appendTo(ak.titlebar || aj).attr("role", "button").click(function(am) {
                if (!aj.hasClass(P)) {
                    X.hide(am)
                }
                return ah
            })
        };
        S._updateButton = function(C) {
            if (!this.rendered) {
                return ah
            }
            var X = this.elements.button;
            if (C) {
                this._createButton()
            } else {
                X.remove()
            }
        };
        function y(C) {
            return p.concat("").join(C ? "-" + C + " " : " ")
        }
        S._setWidget = function() {
            var C = this.options.style.widget, aj = this.elements, Y = aj.tooltip, X = Y.hasClass(P);
            Y.removeClass(P);
            P = C ? "ui-state-disabled" : "qtip-disabled";
            Y.toggleClass(P, X);
            Y.toggleClass("ui-helper-reset " + y(), C).toggleClass(h, this.options.style.def&&!C);
            if (aj.content) {
                aj.content.toggleClass(y("content"), C)
            }
            if (aj.titlebar) {
                aj.titlebar.toggleClass(y("header"), C)
            }
            if (aj.button) {
                aj.button.toggleClass(j + "-icon", !C)
            }
        };
        function w(C) {
            return C && {
                type: C.type,
                pageX: C.pageX,
                pageY: C.pageY,
                target: C.target,
                relatedTarget: C.relatedTarget,
                scrollX: C.scrollX || b.pageXOffset || a.body.scrollLeft || a.documentElement.scrollLeft,
                scrollY: C.scrollY || b.pageYOffset || a.body.scrollTop || a.documentElement.scrollTop
            }
            || {}
        }
        function M(X, C) {
            if (C > 0) {
                return setTimeout(K.proxy(X, this), C)
            } else {
                X.call(this)
            }
        }
        function E(C) {
            if (this.tooltip.hasClass(P)) {
                return ah
            }
            clearTimeout(this.timers.show);
            clearTimeout(this.timers.hide);
            this.timers.show = M.call(this, function() {
                this.toggle(B, C)
            }, this.options.show.delay)
        }
        function V(aj) {
            if (this.tooltip.hasClass(P)) {
                return ah
            }
            var X = K(aj.relatedTarget), C = X.closest(l)[0] === this.tooltip[0], Y = X[0] === this.options.show.target[0];
            clearTimeout(this.timers.show);
            clearTimeout(this.timers.hide);
            if (this !== X[0] && (this.options.position.target === "mouse" && C) || (this.options.hide.fixed && ((/mouse(out|leave|move)/).test(aj.type) && (C || Y)))) {
                try {
                    aj.preventDefault();
                    aj.stopImmediatePropagation()
                } catch (ak) {}
                return 
            }
            this.timers.hide = M.call(this, function() {
                this.toggle(ah, aj)
            }, this.options.hide.delay, this)
        }
        function ad(C) {
            if (this.tooltip.hasClass(P) ||!this.options.hide.inactive) {
                return ah
            }
            clearTimeout(this.timers.inactive);
            this.timers.inactive = M.call(this, function() {
                this.hide(C)
            }, this.options.hide.inactive)
        }
        function T(C) {
            if (this.rendered && this.tooltip[0].offsetWidth > 0) {
                this.reposition(C)
            }
        }
        S._storeMouse = function(C) {
            (this.mouse = w(C)).type = "mousemove"
        };
        S._bind = function(C, Y, al, ak, X) {
            var aj = "." + this._id + (ak ? "-" + ak : "");
            Y.length && K(C).bind((Y.split ? Y : Y.join(aj + " ")) + aj, K.proxy(al, X || this))
        };
        S._unbind = function(C, X) {
            K(C).unbind("." + this._id + (X ? "-" + X : ""))
        };
        var af = "." + j;
        function t(C, X, Y) {
            K(a.body).delegate(C, (X.split ? X : X.join(af + " ")) + af, function() {
                var aj = Z.api[K.attr(this, N)];
                aj&&!aj.disabled && Y.apply(aj, arguments)
            })
        }
        K(function() {
            t(l, ["mouseenter", "mouseleave"], function(X) {
                var aj = X.type === "mouseenter", Y = K(X.currentTarget), ak = K(X.relatedTarget || X.target), C = this.options;
                if (aj) {
                    this.focus(X);
                    Y.hasClass(aa)&&!Y.hasClass(P) && clearTimeout(this.timers.hide)
                } else {
                    if (C.position.target === "mouse" && C.hide.event && C.show.target&&!ak.closest(C.show.target[0]).length) {
                        this.hide(X)
                    }
                }
                Y.toggleClass(ab, aj)
            });
            t("[" + N + "]", x, ad)
        });
        S._trigger = function(X, C, Y) {
            var aj = K.Event("tooltip" + X);
            aj.originalEvent = (Y && K.extend({}, Y)) || this.cache.event || F;
            this.triggering = X;
            this.tooltip.trigger(aj, [this].concat(C || []));
            this.triggering = ah;
            return !aj.isDefaultPrevented()
        };
        S._bindEvents = function(C, am, Y, X, ak, aj) {
            if (X.add(Y).length === X.length) {
                var al = [];
                am = K.map(am, function(ao) {
                    var an = K.inArray(ao, C);
                    if (an>-1) {
                        al.push(C.splice(an, 1)[0]);
                        return 
                    }
                    return ao
                });
                al.length && this._bind(Y, al, function(an) {
                    var ao = this.rendered ? this.tooltip[0].offsetWidth > 0: false;
                    (ao ? aj : ak).call(this, an)
                })
            }
            this._bind(Y, C, ak);
            this._bind(X, am, aj)
        };
        S._assignInitialEvents = function(ak) {
            var X = this.options, aj = X.show.target, Y = X.hide.target, C = X.show.event ? K.trim("" + X.show.event).split(" "): [], am = X.hide.event ? K.trim("" + X.hide.event).split(" "): [];
            if (/mouse(over|enter)/i.test(X.show.event)&&!/mouse(out|leave)/i.test(X.hide.event)) {
                am.push("mouseleave")
            }
            this._bind(aj, "mousemove", function(an) {
                this._storeMouse(an);
                this.cache.onTarget = B
            });
            function al(an) {
                if (this.disabled || this.destroyed) {
                    return ah
                }
                this.cache.event = w(an);
                this.cache.target = an ? K(an.target) : [c];
                clearTimeout(this.timers.show);
                this.timers.show = M.call(this, function() {
                    this.render(typeof an === "object" || X.show.ready)
                }, X.show.delay)
            }
            this._bindEvents(C, am, aj, Y, al, function() {
                clearTimeout(this.timers.show)
            });
            if (X.show.ready || X.prerender) {
                al.call(this, ak)
            }
        };
        S._assignEvents = function() {
            var aq = this, at = this.options, ao = at.position, ar = this.tooltip, ak = at.show.target, an = at.hide.target, al = ao.container, X = ao.viewport, aj = K(a), Y = K(a.body), am = K(b), C = at.show.event ? K.trim("" + at.show.event).split(" "): [], ap = at.hide.event ? K.trim("" + at.hide.event).split(" "): [];
            K.each(at.events, function(au, av) {
                aq._bind(ar, au === "toggle" ? ["tooltipshow", "tooltiphide"] : ["tooltip" + au], av, null, ar)
            });
            if (/mouse(out|leave)/i.test(at.hide.event) && at.hide.leave === "window") {
                this._bind(aj, ["mouseout", "blur"], function(au) {
                    if (!/select|option/.test(au.target.nodeName)&&!au.relatedTarget) {
                        this.hide(au)
                    }
                })
            }
            if (at.hide.fixed) {
                an = an.add(ar.addClass(aa))
            } else {
                if (/mouse(over|enter)/i.test(at.show.event)) {
                    this._bind(an, "mouseleave", function() {
                        clearTimeout(this.timers.show)
                    })
                }
            }
            if (("" + at.hide.event).indexOf("unfocus")>-1) {
                this._bind(al.closest("html"), ["mousedown", "touchstart"], function(ax) {
                    var aw = K(ax.target), av = this.rendered&&!this.tooltip.hasClass(P) && this.tooltip[0].offsetWidth > 0, au = aw.parents(l).filter(this.tooltip[0]).length > 0;
                    if (aw[0] !== this.target[0] && aw[0] !== this.tooltip[0]&&!au&&!this.target.has(aw[0]).length && av) {
                        this.hide(ax)
                    }
                })
            }
            if ("number" === typeof at.hide.inactive) {
                this._bind(ak, "qtip-" + this.id + "-inactive", ad);
                this._bind(an.add(ar), Z.inactiveEvents, ad, "-inactive")
            }
            this._bindEvents(C, ap, ak, an, E, V);
            this._bind(ak.add(ar), "mousemove", function(ax) {
                if ("number" === typeof at.hide.distance) {
                    var aw = this.cache.origin || {}, av = this.options.hide.distance, au = Math.abs;
                    if (au(ax.pageX - aw.pageX) >= av || au(ax.pageY - aw.pageY) >= av) {
                        this.hide(ax)
                    }
                }
                this._storeMouse(ax)
            });
            if (ao.target === "mouse") {
                if (ao.adjust.mouse) {
                    if (at.hide.event) {
                        this._bind(ak, ["mouseenter", "mouseleave"], function(au) {
                            this.cache.onTarget = au.type === "mouseenter"
                        })
                    }
                    this._bind(aj, "mousemove", function(au) {
                        if (this.rendered && this.cache.onTarget&&!this.tooltip.hasClass(P) && this.tooltip[0].offsetWidth > 0) {
                            this.reposition(au)
                        }
                    })
                }
            }
            if (ao.adjust.resize || X.length) {
                this._bind(K.event.special.resize ? X : am, "resize", T)
            }
            if (ao.adjust.scroll) {
                this._bind(am.add(ao.container), "scroll", T)
            }
        };
        S._unassignEvents = function() {
            var C = [this.options.show.target[0], this.options.hide.target[0], this.rendered && this.tooltip[0], this.options.position.container[0], this.options.position.viewport[0], this.options.position.container.closest("html")[0], b, a];
            this._unbind(K([]).pushStack(K.grep(C, function(X) {
                return typeof X === "object"
            })))
        };
        function ac(aj, X, C) {
            var ak, au, ao, Y, ar, al = K(a.body), aq = aj[0] === a ? al: aj, ap = (aj.metadata) ? aj.metadata(C.metadata): F, at = C.metadata.type === "html5" && ap ? ap[C.metadata.name]: F, am = aj.data(C.metadata.name || "qtipopts");
            try {
                am = typeof am === "string" ? K.parseJSON(am) : am
            } catch (an) {}
            Y = K.extend(B, {}, Z.defaults, C, typeof am === "object" ? A(am) : F, A(at || ap));
            au = Y.position;
            Y.id = X;
            if ("boolean" === typeof Y.content.text) {
                ao = aj.attr(Y.content.attr);
                if (Y.content.attr !== ah && ao) {
                    Y.content.text = ao
                } else {
                    return ah
                }
            }
            if (!au.container.length) {
                au.container = al
            }
            if (au.target === ah) {
                au.target = aq
            }
            if (Y.show.target === ah) {
                Y.show.target = aq
            }
            if (Y.show.solo === B) {
                Y.show.solo = au.container.closest("body")
            }
            if (Y.hide.target === ah) {
                Y.hide.target = aq
            }
            if (Y.position.viewport === B) {
                Y.position.viewport = au.container
            }
            au.container = au.container.eq(0);
            au.at = new n(au.at, B);
            au.my = new n(au.my);
            if (aj.data(j)) {
                if (Y.overwrite) {
                    aj.qtip("destroy", true)
                } else {
                    if (Y.overwrite === ah) {
                        return ah
                    }
                }
            }
            aj.attr(R, X);
            if (Y.suppress && (ar = aj.attr("title"))) {
                aj.removeAttr("title").attr(o, ar).attr("title", "")
            }
            ak = new g(aj, Y, X, !!ao);
            aj.data(j, ak);
            aj.one("remove.qtip-" + X + " removeqtip.qtip-" + X, function() {
                var av;
                if ((av = K(this).data(j))) {
                    av.destroy(true)
                }
            });
            return ak
        }
        Z = K.fn.qtip = function(X, al, am) {
            var an = ("" + X).toLowerCase(), ak = F, C = K.makeArray(arguments).slice(1), aj = C[C.length - 1], Y = this[0] ? K.data(this[0], j): F;
            if ((!arguments.length && Y) || an === "api") {
                return Y
            } else {
                if ("string" === typeof X) {
                    this.each(function() {
                        var ao = K.data(this, j);
                        if (!ao) {
                            return B
                        }
                        if (aj && aj.timeStamp) {
                            ao.cache.event = aj
                        }
                        if (al && (an === "option" || an === "options")) {
                            if (am !== c || K.isPlainObject(al)) {
                                ao.set(al, am)
                            } else {
                                ak = ao.get(al);
                                return ah
                            }
                        } else {
                            if (ao[an]) {
                                ao[an].apply(ao, C)
                            }
                        }
                    });
                    return ak !== F ? ak : this
                } else {
                    if ("object" === typeof X ||!arguments.length) {
                        Y = A(K.extend(B, {}, X));
                        return this.each(function(ao) {
                            var ap, aq;
                            aq = K.isArray(Y.id) ? Y.id[ao] : Y.id;
                            aq=!aq || aq === ah || aq.length < 1 || Z.api[aq] ? Z.nextid++ : aq;
                            ap = ac(K(this), aq, Y);
                            if (ap === ah) {
                                return B
                            } else {
                                Z.api[aq] = ap
                            }
                            K.each(D, function() {
                                if (this.initialize === "initialize") {
                                    this (ap)
                                }
                            });
                            ap._assignInitialEvents(aj)
                        })
                    }
                }
            }
        };
        K.qtip = g;
        Z.api = {};
        K.each({
            attr: function(C, ak) {
                if (this.length) {
                    var X = this[0], aj = "title", Y = K.data(X, "qtip");
                    if (C === aj && Y && "object" === typeof Y && Y.options.suppress) {
                        if (arguments.length < 2) {
                            return K.attr(X, o)
                        }
                        if (Y && Y.options.content.attr === aj && Y.cache.attr) {
                            Y.set("content.text", ak)
                        }
                        return this.attr(o, ak)
                    }
                }
                return K.fn["attr" + u].apply(this, arguments)
            },
            clone: function(X) {
                var aj = K([]), Y = "title", C = K.fn["clone" + u].apply(this, arguments);
                if (!X) {
                    C.filter("[" + o + "]").attr("title", function() {
                        return K.attr(this, o)
                    }).removeAttr(o)
                }
                return C
            }
        }, function(X, Y) {
            if (!Y || K.fn[X + u]) {
                return B
            }
            var C = K.fn[X + u] = K.fn[X];
            K.fn[X] = function() {
                return Y.apply(this, arguments) || C.apply(this, arguments)
            }
        });
        if (!K.ui) {
            K["cleanData" + u] = K.cleanData;
            K.cleanData = function(C) {
                for (var X = 0, Y; (Y = K(C[X])).length; X++) {
                    if (Y.attr(R)) {
                        try {
                            Y.triggerHandler("removeqtip")
                        } catch (aj) {}
                    }
                }
                K["cleanData" + u].apply(this, arguments)
            }
        }
        Z.version = "2.2.0";
        Z.nextid = 0;
        Z.inactiveEvents = x;
        Z.zindex = 15000;
        Z.defaults = {
            prerender: ah,
            id: ah,
            overwrite: B,
            suppress: B,
            content: {
                text: B,
                attr: "title",
                title: ah,
                button: ah
            },
            position: {
                my: "top left",
                at: "bottom right",
                target: ah,
                container: ah,
                viewport: ah,
                adjust: {
                    x: 0,
                    y: 0,
                    mouse: B,
                    scroll: B,
                    resize: B,
                    method: "flipinvert flipinvert"
                },
                effect: function(X, Y, C) {
                    K(this).animate(Y, {
                        duration: 200,
                        queue: ah
                    })
                }
            },
            show: {
                target: ah,
                event: "mouseenter",
                effect: B,
                delay: 90,
                solo: ah,
                ready: ah,
                autofocus: ah
            },
            hide: {
                target: ah,
                event: "mouseleave",
                effect: B,
                delay: 0,
                fixed: ah,
                inactive: ah,
                leave: "window",
                distance: ah
            },
            style: {
                classes: "",
                widget: ah,
                width: ah,
                height: ah,
                def: B
            },
            events: {
                render: F,
                move: F,
                show: F,
                hide: F,
                toggle: F,
                visible: F,
                hidden: F,
                focus: F,
                blur: F
            }
        }
    }))
}(window, document));
eval(function(h, b, i, d, g, f) {
    g = function(a) {
        return (a < b ? "" : g(parseInt(a / b))) + ((a = a%b) > 35 ? String.fromCharCode(a + 29) : a.toString(36))
    };
    if (!"".replace(/^/, String)) {
        while (i--) {
            f[g(i)] = d[i] || g(i)
        }
        d = [function(a) {
            return f[a]
        }
        ];
        g = function() {
            return "\\w+"
        };
        i = 1
    }
    while (i--) {
        if (d[i]) {
            h = h.replace(new RegExp("\\b" + g(i) + "\\b", "g"), d[i])
        }
    }
    return h
}("9 17={3i:'0.1.3',16:1e-6};l v(){}v.23={e:l(i){8(i<1||i>7.4.q)?w:7.4[i-1]},2R:l(){8 7.4.q},1u:l(){8 F.1x(7.2u(7))},24:l(a){9 n=7.4.q;9 V=a.4||a;o(n!=V.q){8 1L}J{o(F.13(7.4[n-1]-V[n-1])>17.16){8 1L}}H(--n);8 2x},1q:l(){8 v.u(7.4)},1b:l(a){9 b=[];7.28(l(x,i){b.19(a(x,i))});8 v.u(b)},28:l(a){9 n=7.4.q,k=n,i;J{i=k-n;a(7.4[i],i+1)}H(--n)},2q:l(){9 r=7.1u();o(r===0){8 7.1q()}8 7.1b(l(x){8 x/r})},1C:l(a){9 V=a.4||a;9 n=7.4.q,k=n,i;o(n!=V.q){8 w}9 b=0,1D=0,1F=0;7.28(l(x,i){b+=x*V[i-1];1D+=x*x;1F+=V[i-1]*V[i-1]});1D=F.1x(1D);1F=F.1x(1F);o(1D*1F===0){8 w}9 c=b/(1D*1F);o(c<-1){c=-1}o(c>1){c=1}8 F.37(c)},1m:l(a){9 b=7.1C(a);8(b===w)?w:(b<=17.16)},34:l(a){9 b=7.1C(a);8(b===w)?w:(F.13(b-F.1A)<=17.16)},2k:l(a){9 b=7.2u(a);8(b===w)?w:(F.13(b)<=17.16)},2j:l(a){9 V=a.4||a;o(7.4.q!=V.q){8 w}8 7.1b(l(x,i){8 x+V[i-1]})},2C:l(a){9 V=a.4||a;o(7.4.q!=V.q){8 w}8 7.1b(l(x,i){8 x-V[i-1]})},22:l(k){8 7.1b(l(x){8 x*k})},x:l(k){8 7.22(k)},2u:l(a){9 V=a.4||a;9 i,2g=0,n=7.4.q;o(n!=V.q){8 w}J{2g+=7.4[n-1]*V[n-1]}H(--n);8 2g},2f:l(a){9 B=a.4||a;o(7.4.q!=3||B.q!=3){8 w}9 A=7.4;8 v.u([(A[1]*B[2])-(A[2]*B[1]),(A[2]*B[0])-(A[0]*B[2]),(A[0]*B[1])-(A[1]*B[0])])},2A:l(){9 m=0,n=7.4.q,k=n,i;J{i=k-n;o(F.13(7.4[i])>F.13(m)){m=7.4[i]}}H(--n);8 m},2Z:l(x){9 a=w,n=7.4.q,k=n,i;J{i=k-n;o(a===w&&7.4[i]==x){a=i+1}}H(--n);8 a},3g:l(){8 S.2X(7.4)},2d:l(){8 7.1b(l(x){8 F.2d(x)})},2V:l(x){8 7.1b(l(y){8(F.13(y-x)<=17.16)?x:y})},1o:l(a){o(a.K){8 a.1o(7)}9 V=a.4||a;o(V.q!=7.4.q){8 w}9 b=0,2b;7.28(l(x,i){2b=x-V[i-1];b+=2b*2b});8 F.1x(b)},3a:l(a){8 a.1h(7)},2T:l(a){8 a.1h(7)},1V:l(t,a){9 V,R,x,y,z;2S(7.4.q){27 2:V=a.4||a;o(V.q!=2){8 w}R=S.1R(t).4;x=7.4[0]-V[0];y=7.4[1]-V[1];8 v.u([V[0]+R[0][0]*x+R[0][1]*y,V[1]+R[1][0]*x+R[1][1]*y]);1I;27 3:o(!a.U){8 w}9 C=a.1r(7).4;R=S.1R(t,a.U).4;x=7.4[0]-C[0];y=7.4[1]-C[1];z=7.4[2]-C[2];8 v.u([C[0]+R[0][0]*x+R[0][1]*y+R[0][2]*z,C[1]+R[1][0]*x+R[1][1]*y+R[1][2]*z,C[2]+R[2][0]*x+R[2][1]*y+R[2][2]*z]);1I;2P:8 w}},1t:l(a){o(a.K){9 P=7.4.2O();9 C=a.1r(P).4;8 v.u([C[0]+(C[0]-P[0]),C[1]+(C[1]-P[1]),C[2]+(C[2]-(P[2]||0))])}1d{9 Q=a.4||a;o(7.4.q!=Q.q){8 w}8 7.1b(l(x,i){8 Q[i-1]+(Q[i-1]-x)})}},1N:l(){9 V=7.1q();2S(V.4.q){27 3:1I;27 2:V.4.19(0);1I;2P:8 w}8 V},2n:l(){8'['+7.4.2K(', ')+']'},26:l(a){7.4=(a.4||a).2O();8 7}};v.u=l(a){9 V=25 v();8 V.26(a)};v.i=v.u([1,0,0]);v.j=v.u([0,1,0]);v.k=v.u([0,0,1]);v.2J=l(n){9 a=[];J{a.19(F.2F())}H(--n);8 v.u(a)};v.1j=l(n){9 a=[];J{a.19(0)}H(--n);8 v.u(a)};l S(){}S.23={e:l(i,j){o(i<1||i>7.4.q||j<1||j>7.4[0].q){8 w}8 7.4[i-1][j-1]},33:l(i){o(i>7.4.q){8 w}8 v.u(7.4[i-1])},2E:l(j){o(j>7.4[0].q){8 w}9 a=[],n=7.4.q,k=n,i;J{i=k-n;a.19(7.4[i][j-1])}H(--n);8 v.u(a)},2R:l(){8{2D:7.4.q,1p:7.4[0].q}},2D:l(){8 7.4.q},1p:l(){8 7.4[0].q},24:l(a){9 M=a.4||a;o(1g(M[0][0])=='1f'){M=S.u(M).4}o(7.4.q!=M.q||7.4[0].q!=M[0].q){8 1L}9 b=7.4.q,15=b,i,G,10=7.4[0].q,j;J{i=15-b;G=10;J{j=10-G;o(F.13(7.4[i][j]-M[i][j])>17.16){8 1L}}H(--G)}H(--b);8 2x},1q:l(){8 S.u(7.4)},1b:l(a){9 b=[],12=7.4.q,15=12,i,G,10=7.4[0].q,j;J{i=15-12;G=10;b[i]=[];J{j=10-G;b[i][j]=a(7.4[i][j],i+1,j+1)}H(--G)}H(--12);8 S.u(b)},2i:l(a){9 M=a.4||a;o(1g(M[0][0])=='1f'){M=S.u(M).4}8(7.4.q==M.q&&7.4[0].q==M[0].q)},2j:l(a){9 M=a.4||a;o(1g(M[0][0])=='1f'){M=S.u(M).4}o(!7.2i(M)){8 w}8 7.1b(l(x,i,j){8 x+M[i-1][j-1]})},2C:l(a){9 M=a.4||a;o(1g(M[0][0])=='1f'){M=S.u(M).4}o(!7.2i(M)){8 w}8 7.1b(l(x,i,j){8 x-M[i-1][j-1]})},2B:l(a){9 M=a.4||a;o(1g(M[0][0])=='1f'){M=S.u(M).4}8(7.4[0].q==M.q)},22:l(a){o(!a.4){8 7.1b(l(x){8 x*a})}9 b=a.1u?2x:1L;9 M=a.4||a;o(1g(M[0][0])=='1f'){M=S.u(M).4}o(!7.2B(M)){8 w}9 d=7.4.q,15=d,i,G,10=M[0].q,j;9 e=7.4[0].q,4=[],21,20,c;J{i=15-d;4[i]=[];G=10;J{j=10-G;21=0;20=e;J{c=e-20;21+=7.4[i][c]*M[c][j]}H(--20);4[i][j]=21}H(--G)}H(--d);9 M=S.u(4);8 b?M.2E(1):M},x:l(a){8 7.22(a)},32:l(a,b,c,d){9 e=[],12=c,i,G,j;9 f=7.4.q,1p=7.4[0].q;J{i=c-12;e[i]=[];G=d;J{j=d-G;e[i][j]=7.4[(a+i-1)%f][(b+j-1)%1p]}H(--G)}H(--12);8 S.u(e)},31:l(){9 a=7.4.q,1p=7.4[0].q;9 b=[],12=1p,i,G,j;J{i=1p-12;b[i]=[];G=a;J{j=a-G;b[i][j]=7.4[j][i]}H(--G)}H(--12);8 S.u(b)},1y:l(){8(7.4.q==7.4[0].q)},2A:l(){9 m=0,12=7.4.q,15=12,i,G,10=7.4[0].q,j;J{i=15-12;G=10;J{j=10-G;o(F.13(7.4[i][j])>F.13(m)){m=7.4[i][j]}}H(--G)}H(--12);8 m},2Z:l(x){9 a=w,12=7.4.q,15=12,i,G,10=7.4[0].q,j;J{i=15-12;G=10;J{j=10-G;o(7.4[i][j]==x){8{i:i+1,j:j+1}}}H(--G)}H(--12);8 w},30:l(){o(!7.1y){8 w}9 a=[],n=7.4.q,k=n,i;J{i=k-n;a.19(7.4[i][i])}H(--n);8 v.u(a)},1K:l(){9 M=7.1q(),1c;9 n=7.4.q,k=n,i,1s,1n=7.4[0].q,p;J{i=k-n;o(M.4[i][i]==0){2e(j=i+1;j<k;j++){o(M.4[j][i]!=0){1c=[];1s=1n;J{p=1n-1s;1c.19(M.4[i][p]+M.4[j][p])}H(--1s);M.4[i]=1c;1I}}}o(M.4[i][i]!=0){2e(j=i+1;j<k;j++){9 a=M.4[j][i]/M.4[i][i];1c=[];1s=1n;J{p=1n-1s;1c.19(p<=i?0:M.4[j][p]-M.4[i][p]*a)}H(--1s);M.4[j]=1c}}}H(--n);8 M},3h:l(){8 7.1K()},2z:l(){o(!7.1y()){8 w}9 M=7.1K();9 a=M.4[0][0],n=M.4.q-1,k=n,i;J{i=k-n+1;a=a*M.4[i][i]}H(--n);8 a},3f:l(){8 7.2z()},2y:l(){8(7.1y()&&7.2z()===0)},2Y:l(){o(!7.1y()){8 w}9 a=7.4[0][0],n=7.4.q-1,k=n,i;J{i=k-n+1;a+=7.4[i][i]}H(--n);8 a},3e:l(){8 7.2Y()},1Y:l(){9 M=7.1K(),1Y=0;9 a=7.4.q,15=a,i,G,10=7.4[0].q,j;J{i=15-a;G=10;J{j=10-G;o(F.13(M.4[i][j])>17.16){1Y++;1I}}H(--G)}H(--a);8 1Y},3d:l(){8 7.1Y()},2W:l(a){9 M=a.4||a;o(1g(M[0][0])=='1f'){M=S.u(M).4}9 T=7.1q(),1p=T.4[0].q;9 b=T.4.q,15=b,i,G,10=M[0].q,j;o(b!=M.q){8 w}J{i=15-b;G=10;J{j=10-G;T.4[i][1p+j]=M[i][j]}H(--G)}H(--b);8 T},2w:l(){o(!7.1y()||7.2y()){8 w}9 a=7.4.q,15=a,i,j;9 M=7.2W(S.I(a)).1K();9 b,1n=M.4[0].q,p,1c,2v;9 c=[],2c;J{i=a-1;1c=[];b=1n;c[i]=[];2v=M.4[i][i];J{p=1n-b;2c=M.4[i][p]/2v;1c.19(2c);o(p>=15){c[i].19(2c)}}H(--b);M.4[i]=1c;2e(j=0;j<i;j++){1c=[];b=1n;J{p=1n-b;1c.19(M.4[j][p]-M.4[i][p]*M.4[j][i])}H(--b);M.4[j]=1c}}H(--a);8 S.u(c)},3c:l(){8 7.2w()},2d:l(){8 7.1b(l(x){8 F.2d(x)})},2V:l(x){8 7.1b(l(p){8(F.13(p-x)<=17.16)?x:p})},2n:l(){9 a=[];9 n=7.4.q,k=n,i;J{i=k-n;a.19(v.u(7.4[i]).2n())}H(--n);8 a.2K('\\n')},26:l(a){9 i,4=a.4||a;o(1g(4[0][0])!='1f'){9 b=4.q,15=b,G,10,j;7.4=[];J{i=15-b;G=4[i].q;10=G;7.4[i]=[];J{j=10-G;7.4[i][j]=4[i][j]}H(--G)}H(--b);8 7}9 n=4.q,k=n;7.4=[];J{i=k-n;7.4.19([4[i]])}H(--n);8 7}};S.u=l(a){9 M=25 S();8 M.26(a)};S.I=l(n){9 a=[],k=n,i,G,j;J{i=k-n;a[i]=[];G=k;J{j=k-G;a[i][j]=(i==j)?1:0}H(--G)}H(--n);8 S.u(a)};S.2X=l(a){9 n=a.q,k=n,i;9 M=S.I(n);J{i=k-n;M.4[i][i]=a[i]}H(--n);8 M};S.1R=l(b,a){o(!a){8 S.u([[F.1H(b),-F.1G(b)],[F.1G(b),F.1H(b)]])}9 d=a.1q();o(d.4.q!=3){8 w}9 e=d.1u();9 x=d.4[0]/e,y=d.4[1]/e,z=d.4[2]/e;9 s=F.1G(b),c=F.1H(b),t=1-c;8 S.u([[t*x*x+c,t*x*y-s*z,t*x*z+s*y],[t*x*y+s*z,t*y*y+c,t*y*z-s*x],[t*x*z-s*y,t*y*z+s*x,t*z*z+c]])};S.3b=l(t){9 c=F.1H(t),s=F.1G(t);8 S.u([[1,0,0],[0,c,-s],[0,s,c]])};S.39=l(t){9 c=F.1H(t),s=F.1G(t);8 S.u([[c,0,s],[0,1,0],[-s,0,c]])};S.38=l(t){9 c=F.1H(t),s=F.1G(t);8 S.u([[c,-s,0],[s,c,0],[0,0,1]])};S.2J=l(n,m){8 S.1j(n,m).1b(l(){8 F.2F()})};S.1j=l(n,m){9 a=[],12=n,i,G,j;J{i=n-12;a[i]=[];G=m;J{j=m-G;a[i][j]=0}H(--G)}H(--12);8 S.u(a)};l 14(){}14.23={24:l(a){8(7.1m(a)&&7.1h(a.K))},1q:l(){8 14.u(7.K,7.U)},2U:l(a){9 V=a.4||a;8 14.u([7.K.4[0]+V[0],7.K.4[1]+V[1],7.K.4[2]+(V[2]||0)],7.U)},1m:l(a){o(a.W){8 a.1m(7)}9 b=7.U.1C(a.U);8(F.13(b)<=17.16||F.13(b-F.1A)<=17.16)},1o:l(a){o(a.W){8 a.1o(7)}o(a.U){o(7.1m(a)){8 7.1o(a.K)}9 N=7.U.2f(a.U).2q().4;9 A=7.K.4,B=a.K.4;8 F.13((A[0]-B[0])*N[0]+(A[1]-B[1])*N[1]+(A[2]-B[2])*N[2])}1d{9 P=a.4||a;9 A=7.K.4,D=7.U.4;9 b=P[0]-A[0],2a=P[1]-A[1],29=(P[2]||0)-A[2];9 c=F.1x(b*b+2a*2a+29*29);o(c===0)8 0;9 d=(b*D[0]+2a*D[1]+29*D[2])/c;9 e=1-d*d;8 F.13(c*F.1x(e<0?0:e))}},1h:l(a){9 b=7.1o(a);8(b!==w&&b<=17.16)},2T:l(a){8 a.1h(7)},1v:l(a){o(a.W){8 a.1v(7)}8(!7.1m(a)&&7.1o(a)<=17.16)},1U:l(a){o(a.W){8 a.1U(7)}o(!7.1v(a)){8 w}9 P=7.K.4,X=7.U.4,Q=a.K.4,Y=a.U.4;9 b=X[0],1z=X[1],1B=X[2],1T=Y[0],1S=Y[1],1M=Y[2];9 c=P[0]-Q[0],2s=P[1]-Q[1],2r=P[2]-Q[2];9 d=-b*c-1z*2s-1B*2r;9 e=1T*c+1S*2s+1M*2r;9 f=b*b+1z*1z+1B*1B;9 g=1T*1T+1S*1S+1M*1M;9 h=b*1T+1z*1S+1B*1M;9 k=(d*g/f+h*e)/(g-h*h);8 v.u([P[0]+k*b,P[1]+k*1z,P[2]+k*1B])},1r:l(a){o(a.U){o(7.1v(a)){8 7.1U(a)}o(7.1m(a)){8 w}9 D=7.U.4,E=a.U.4;9 b=D[0],1l=D[1],1k=D[2],1P=E[0],1O=E[1],1Q=E[2];9 x=(1k*1P-b*1Q),y=(b*1O-1l*1P),z=(1l*1Q-1k*1O);9 N=v.u([x*1Q-y*1O,y*1P-z*1Q,z*1O-x*1P]);9 P=11.u(a.K,N);8 P.1U(7)}1d{9 P=a.4||a;o(7.1h(P)){8 v.u(P)}9 A=7.K.4,D=7.U.4;9 b=D[0],1l=D[1],1k=D[2],1w=A[0],18=A[1],1a=A[2];9 x=b*(P[1]-18)-1l*(P[0]-1w),y=1l*((P[2]||0)-1a)-1k*(P[1]-18),z=1k*(P[0]-1w)-b*((P[2]||0)-1a);9 V=v.u([1l*x-1k*z,1k*y-b*x,b*z-1l*y]);9 k=7.1o(P)/V.1u();8 v.u([P[0]+V.4[0]*k,P[1]+V.4[1]*k,(P[2]||0)+V.4[2]*k])}},1V:l(t,a){o(1g(a.U)=='1f'){a=14.u(a.1N(),v.k)}9 R=S.1R(t,a.U).4;9 C=a.1r(7.K).4;9 A=7.K.4,D=7.U.4;9 b=C[0],1E=C[1],1J=C[2],1w=A[0],18=A[1],1a=A[2];9 x=1w-b,y=18-1E,z=1a-1J;8 14.u([b+R[0][0]*x+R[0][1]*y+R[0][2]*z,1E+R[1][0]*x+R[1][1]*y+R[1][2]*z,1J+R[2][0]*x+R[2][1]*y+R[2][2]*z],[R[0][0]*D[0]+R[0][1]*D[1]+R[0][2]*D[2],R[1][0]*D[0]+R[1][1]*D[1]+R[1][2]*D[2],R[2][0]*D[0]+R[2][1]*D[1]+R[2][2]*D[2]])},1t:l(a){o(a.W){9 A=7.K.4,D=7.U.4;9 b=A[0],18=A[1],1a=A[2],2N=D[0],1l=D[1],1k=D[2];9 c=7.K.1t(a).4;9 d=b+2N,2h=18+1l,2o=1a+1k;9 Q=a.1r([d,2h,2o]).4;9 e=[Q[0]+(Q[0]-d)-c[0],Q[1]+(Q[1]-2h)-c[1],Q[2]+(Q[2]-2o)-c[2]];8 14.u(c,e)}1d o(a.U){8 7.1V(F.1A,a)}1d{9 P=a.4||a;8 14.u(7.K.1t([P[0],P[1],(P[2]||0)]),7.U)}},1Z:l(a,b){a=v.u(a);b=v.u(b);o(a.4.q==2){a.4.19(0)}o(b.4.q==2){b.4.19(0)}o(a.4.q>3||b.4.q>3){8 w}9 c=b.1u();o(c===0){8 w}7.K=a;7.U=v.u([b.4[0]/c,b.4[1]/c,b.4[2]/c]);8 7}};14.u=l(a,b){9 L=25 14();8 L.1Z(a,b)};14.X=14.u(v.1j(3),v.i);14.Y=14.u(v.1j(3),v.j);14.Z=14.u(v.1j(3),v.k);l 11(){}11.23={24:l(a){8(7.1h(a.K)&&7.1m(a))},1q:l(){8 11.u(7.K,7.W)},2U:l(a){9 V=a.4||a;8 11.u([7.K.4[0]+V[0],7.K.4[1]+V[1],7.K.4[2]+(V[2]||0)],7.W)},1m:l(a){9 b;o(a.W){b=7.W.1C(a.W);8(F.13(b)<=17.16||F.13(F.1A-b)<=17.16)}1d o(a.U){8 7.W.2k(a.U)}8 w},2k:l(a){9 b=7.W.1C(a.W);8(F.13(F.1A/2-b)<=17.16)},1o:l(a){o(7.1v(a)||7.1h(a)){8 0}o(a.K){9 A=7.K.4,B=a.K.4,N=7.W.4;8 F.13((A[0]-B[0])*N[0]+(A[1]-B[1])*N[1]+(A[2]-B[2])*N[2])}1d{9 P=a.4||a;9 A=7.K.4,N=7.W.4;8 F.13((A[0]-P[0])*N[0]+(A[1]-P[1])*N[1]+(A[2]-(P[2]||0))*N[2])}},1h:l(a){o(a.W){8 w}o(a.U){8(7.1h(a.K)&&7.1h(a.K.2j(a.U)))}1d{9 P=a.4||a;9 A=7.K.4,N=7.W.4;9 b=F.13(N[0]*(A[0]-P[0])+N[1]*(A[1]-P[1])+N[2]*(A[2]-(P[2]||0)));8(b<=17.16)}},1v:l(a){o(1g(a.U)=='1f'&&1g(a.W)=='1f'){8 w}8!7.1m(a)},1U:l(a){o(!7.1v(a)){8 w}o(a.U){9 A=a.K.4,D=a.U.4,P=7.K.4,N=7.W.4;9 b=(N[0]*(P[0]-A[0])+N[1]*(P[1]-A[1])+N[2]*(P[2]-A[2]))/(N[0]*D[0]+N[1]*D[1]+N[2]*D[2]);8 v.u([A[0]+D[0]*b,A[1]+D[1]*b,A[2]+D[2]*b])}1d o(a.W){9 c=7.W.2f(a.W).2q();9 N=7.W.4,A=7.K.4,O=a.W.4,B=a.K.4;9 d=S.1j(2,2),i=0;H(d.2y()){i++;d=S.u([[N[i%3],N[(i+1)%3]],[O[i%3],O[(i+1)%3]]])}9 e=d.2w().4;9 x=N[0]*A[0]+N[1]*A[1]+N[2]*A[2];9 y=O[0]*B[0]+O[1]*B[1]+O[2]*B[2];9 f=[e[0][0]*x+e[0][1]*y,e[1][0]*x+e[1][1]*y];9 g=[];2e(9 j=1;j<=3;j++){g.19((i==j)?0:f[(j+(5-i)%3)%3])}8 14.u(g,c)}},1r:l(a){9 P=a.4||a;9 A=7.K.4,N=7.W.4;9 b=(A[0]-P[0])*N[0]+(A[1]-P[1])*N[1]+(A[2]-(P[2]||0))*N[2];8 v.u([P[0]+N[0]*b,P[1]+N[1]*b,(P[2]||0)+N[2]*b])},1V:l(t,a){9 R=S.1R(t,a.U).4;9 C=a.1r(7.K).4;9 A=7.K.4,N=7.W.4;9 b=C[0],1E=C[1],1J=C[2],1w=A[0],18=A[1],1a=A[2];9 x=1w-b,y=18-1E,z=1a-1J;8 11.u([b+R[0][0]*x+R[0][1]*y+R[0][2]*z,1E+R[1][0]*x+R[1][1]*y+R[1][2]*z,1J+R[2][0]*x+R[2][1]*y+R[2][2]*z],[R[0][0]*N[0]+R[0][1]*N[1]+R[0][2]*N[2],R[1][0]*N[0]+R[1][1]*N[1]+R[1][2]*N[2],R[2][0]*N[0]+R[2][1]*N[1]+R[2][2]*N[2]])},1t:l(a){o(a.W){9 A=7.K.4,N=7.W.4;9 b=A[0],18=A[1],1a=A[2],2M=N[0],2L=N[1],2Q=N[2];9 c=7.K.1t(a).4;9 d=b+2M,2p=18+2L,2m=1a+2Q;9 Q=a.1r([d,2p,2m]).4;9 e=[Q[0]+(Q[0]-d)-c[0],Q[1]+(Q[1]-2p)-c[1],Q[2]+(Q[2]-2m)-c[2]];8 11.u(c,e)}1d o(a.U){8 7.1V(F.1A,a)}1d{9 P=a.4||a;8 11.u(7.K.1t([P[0],P[1],(P[2]||0)]),7.W)}},1Z:l(a,b,c){a=v.u(a);a=a.1N();o(a===w){8 w}b=v.u(b);b=b.1N();o(b===w){8 w}o(1g(c)=='1f'){c=w}1d{c=v.u(c);c=c.1N();o(c===w){8 w}}9 d=a.4[0],18=a.4[1],1a=a.4[2];9 e=b.4[0],1W=b.4[1],1X=b.4[2];9 f,1i;o(c!==w){9 g=c.4[0],2l=c.4[1],2t=c.4[2];f=v.u([(1W-18)*(2t-1a)-(1X-1a)*(2l-18),(1X-1a)*(g-d)-(e-d)*(2t-1a),(e-d)*(2l-18)-(1W-18)*(g-d)]);1i=f.1u();o(1i===0){8 w}f=v.u([f.4[0]/1i,f.4[1]/1i,f.4[2]/1i])}1d{1i=F.1x(e*e+1W*1W+1X*1X);o(1i===0){8 w}f=v.u([b.4[0]/1i,b.4[1]/1i,b.4[2]/1i])}7.K=a;7.W=f;8 7}};11.u=l(a,b,c){9 P=25 11();8 P.1Z(a,b,c)};11.2I=11.u(v.1j(3),v.k);11.2H=11.u(v.1j(3),v.i);11.2G=11.u(v.1j(3),v.j);11.36=11.2I;11.35=11.2H;11.3j=11.2G;9 $V=v.u;9 $M=S.u;9 $L=14.u;9 $P=11.u;", 62, 206, "||||elements|||this|return|var||||||||||||function|||if||length||||create|Vector|null|||||||||Math|nj|while||do|anchor||||||||Matrix||direction||normal||||kj|Plane|ni|abs|Line|ki|precision|Sylvester|A2|push|A3|map|els|else||undefined|typeof|contains|mod|Zero|D3|D2|isParallelTo|kp|distanceFrom|cols|dup|pointClosestTo|np|reflectionIn|modulus|intersects|A1|sqrt|isSquare|X2|PI|X3|angleFrom|mod1|C2|mod2|sin|cos|break|C3|toRightTriangular|false|Y3|to3D|E2|E1|E3|Rotation|Y2|Y1|intersectionWith|rotate|v12|v13|rank|setVectors|nc|sum|multiply|prototype|eql|new|setElements|case|each|PA3|PA2|part|new_element|round|for|cross|product|AD2|isSameSizeAs|add|isPerpendicularTo|v22|AN3|inspect|AD3|AN2|toUnitVector|PsubQ3|PsubQ2|v23|dot|divisor|inverse|true|isSingular|determinant|max|canMultiplyFromLeft|subtract|rows|col|random|ZX|YZ|XY|Random|join|N2|N1|D1|slice|default|N3|dimensions|switch|liesIn|translate|snapTo|augment|Diagonal|trace|indexOf|diagonal|transpose|minor|row|isAntiparallelTo|ZY|YX|acos|RotationZ|RotationY|liesOn|RotationX|inv|rk|tr|det|toDiagonalMatrix|toUpperTriangular|version|XZ".split("|"), 0, {}));
var dateFormat = function() {
    var a = /d{1,4}|m{1,4}|yy(?:yy)?|([HhMsTt])\1?|[LloSZ]|"[^"]*"|'[^']*'/g, b = /\b(?:[PMCEA][SDP]T|(?:Pacific|Mountain|Central|Eastern|Atlantic) (?:Standard|Daylight|Prevailing) Time|(?:GMT|UTC)(?:[-+]\d{4})?)\b/g, d = /[^-+\dA-Z]/g, c = function(f, e) {
        f = String(f);
        e = e || 2;
        while (f.length < e) {
            f = "0" + f
        }
        return f
    };
    return function(i, v, q) {
        var g = dateFormat;
        if (arguments.length == 1 && Object.prototype.toString.call(i) == "[object String]"&&!/\d/.test(i)) {
            v = i;
            i = undefined
        }
        i = i ? new Date(i) : new Date;
        if (isNaN(i)) {
            throw SyntaxError("invalid date")
        }
        v = String(g.masks[v] || v || g.masks["default"]);
        if (v.slice(0, 4) == "UTC:") {
            v = v.slice(4);
            q = true
        }
        var t = q ? "getUTC": "get", l = i[t + "Date"](), e = i[t + "Day"](), j = i[t + "Month"](), p = i[t + "FullYear"](), r = i[t + "Hours"](), k = i[t + "Minutes"](), u = i[t + "Seconds"](), n = i[t + "Milliseconds"](), f = q ? 0: i.getTimezoneOffset(), h = {
            d: l,
            dd: c(l),
            ddd: g.i18n.dayNames[e],
            dddd: g.i18n.dayNames[e + 7],
            m: j + 1,
            mm: c(j + 1),
            mmm: g.i18n.monthNames[j],
            mmmm: g.i18n.monthNames[j + 12],
            yy: String(p).slice(2),
            yyyy: p,
            h: r%12 || 12,
            hh: c(r%12 || 12),
            H: r,
            HH: c(r),
            M: k,
            MM: c(k),
            s: u,
            ss: c(u),
            l: c(n, 3),
            L: c(n > 99 ? Math.round(n / 10) : n),
            t: r < 12 ? "a": "p",
            tt: r < 12 ? "am": "pm",
            T: r < 12 ? "A": "P",
            TT: r < 12 ? "AM": "PM",
            Z: q ? "UTC": (String(i).match(b) || [""]).pop().replace(d, ""),
            o: (f > 0 ? "-" : "+") + c(Math.floor(Math.abs(f) / 60) * 100 + Math.abs(f)%60, 4),
            S: ["th", "st", "nd", "rd"][l%10 > 3 ? 0: (l%100 - l%10 != 10) * l%10]
        };
        return v.replace(a, function(m) {
            return m in h ? h[m] : m.slice(1, m.length - 1)
        })
    }
}();
dateFormat.masks = {
    "default": "ddd mmm dd yyyy HH:MM:ss",
    shortDate: "m/d/yy",
    mediumDate: "mmm d, yyyy",
    longDate: "mmmm d, yyyy",
    fullDate: "dddd, mmmm d, yyyy",
    shortTime: "h:MM TT",
    mediumTime: "h:MM:ss TT",
    longTime: "h:MM:ss TT Z",
    isoDate: "yyyy-mm-dd",
    isoTime: "HH:MM:ss",
    isoDateTime: "yyyy-mm-dd'T'HH:MM:ss",
    isoUtcDateTime: "UTC:yyyy-mm-dd'T'HH:MM:ss'Z'"
};
dateFormat.i18n = {
    dayNames: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    monthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
};
Date.prototype.format = function(a, b) {
    return dateFormat(this, a, b)
};
Inflector = {
    Inflections: {
        plural: [[/(quiz)$/i, "$1zes"], [/^(ox)$/i, "$1en"], [/([m|l])ouse$/i, "$1ice"], [/(matr|vert|ind)ix|ex$/i, "$1ices"], [/(x|ch|ss|sh)$/i, "$1es"], [/([^aeiouy]|qu)y$/i, "$1ies"], [/(hive)$/i, "$1s"], [/(?:([^f])fe|([lr])f)$/i, "$1$2ves"], [/sis$/i, "ses"], [/([ti])um$/i, "$1a"], [/(buffal|tomat)o$/i, "$1oes"], [/(bu)s$/i, "$1ses"], [/(alias|status)$/i, "$1es"], [/(octop|vir)us$/i, "$1i"], [/(ax|test)is$/i, "$1es"], [/s$/i, "s"], [/$/, "s"]],
        singular: [[/(quiz)zes$/i, "$1"], [/(matr)ices$/i, "$1ix"], [/(vert|ind)ices$/i, "$1ex"], [/^(ox)en/i, "$1"], [/(alias|status)es$/i, "$1"], [/(octop|vir)i$/i, "$1us"], [/(cris|ax|test)es$/i, "$1is"], [/(shoe)s$/i, "$1"], [/(o)es$/i, "$1"], [/(bus)es$/i, "$1"], [/([m|l])ice$/i, "$1ouse"], [/(x|ch|ss|sh)es$/i, "$1"], [/(m)ovies$/i, "$1ovie"], [/(s)eries$/i, "$1eries"], [/([^aeiouy]|qu)ies$/i, "$1y"], [/([lr])ves$/i, "$1f"], [/(tive)s$/i, "$1"], [/(hive)s$/i, "$1"], [/([^f])ves$/i, "$1fe"], [/(^analy)ses$/i, "$1sis"], [/((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$/i, "$1$2sis"], [/([ti])a$/i, "$1um"], [/(n)ews$/i, "$1ews"], [/s$/i, ""]],
        irregular: [["move", "moves"], ["sex", "sexes"], ["child", "children"], ["man", "men"], ["person", "people"]],
        uncountable: ["sheep", "fish", "series", "species", "money", "rice", "information", "equipment"]
    },
    ordinalize: function(a) {
        if (11 <= parseInt(a)%100 && parseInt(a)%100 <= 13) {
            return a + "th"
        } else {
            switch (parseInt(a)%10) {
            case 1:
                return a + "st";
            case 2:
                return a + "nd";
            case 3:
                return a + "rd";
            default:
                return a + "th"
            }
        }
    },
    pluralize: function(f) {
        for (var c = 0; c < Inflector.Inflections.uncountable.length; c++) {
            var b = Inflector.Inflections.uncountable[c];
            if (f.toLowerCase() == b) {
                return b
            }
        }
        for (var c = 0; c < Inflector.Inflections.irregular.length; c++) {
            var d = Inflector.Inflections.irregular[c][0], a = Inflector.Inflections.irregular[c][1];
            if ((f.toLowerCase() == d) || (f == a)) {
                return a
            }
        }
        for (var c = 0; c < Inflector.Inflections.plural.length; c++) {
            var e = Inflector.Inflections.plural[c][0], g = Inflector.Inflections.plural[c][1];
            if (e.test(f)) {
                return f.replace(e, g)
            }
        }
    }
};
String.prototype.pluralize = function(b, a) {
    return b + " " + (1 == parseInt(b) ? this : a || Inflector.pluralize(this))
};
String.prototype.ordinalize = function() {
    return Inflector.ordinalize(this)
};
String.prototype.rsplit = function(e, b) {
    var a = this.split(e);
    if (!b || b >= a.length) {
        return a
    }
    var d = a.slice(0, a.length - b).join(e), c = a.slice(0 - b);
    return [d].concat(c)
};
String.prototype.format = function() {
    var a = [this];
    for (var b = 0; b < arguments.length; b += 1) {
        a.push(arguments[b])
    }
    return strings.format.apply(null, a)
};
!function(b) {
    function a(f, e) {
        var g = b.proxy(this.process, this), c = b(f).is("body") ? b(window): b(f), d;
        this.options = b.extend({}, b.fn.scrollspy.defaults, e);
        this.$scrollElement = c.on("scroll.scroll-spy.data-api", g);
        this.selector = (this.options.target || ((d = b(f).attr("href")) && d.replace(/.*(?=#[^\s]+$)/, "")) || "") + "ul li > a";
        this.$body = b("body");
        this.refresh();
        this.process()
    }
    a.prototype = {
        constructor: a,
        refresh: function() {
            var c = this, d;
            this.offsets = b([]);
            this.targets = b([]);
            d = this.$body.find(this.selector).map(function() {
                var f = b(this), e = f.data("target") || f.attr("href"), g = /^#\w/.test(e) && b(e);
                return (g && g.length && [[g.position().top, e]]) || null
            }).sort(function(f, e) {
                return f[0] - e[0]
            }).each(function() {
                c.offsets.push(this[0]);
                c.targets.push(this[1])
            })
        },
        process: function() {
            var h = this.$scrollElement.scrollTop() + this.options.offset, e = this.$scrollElement[0].scrollHeight || this.$body[0].scrollHeight, g = e - this.$scrollElement.height(), f = this.offsets, c = this.targets, j = this.activeTarget, d;
            if (h >= g) {
                return j != (d = c.last()[0]) && this.activate(d)
            }
            for (d = f.length; d--;) {
                j != c[d] && h >= f[d] && (!f[d + 1] || h <= f[d + 1]) && this.activate(c[d])
            }
        },
        activate: function(e) {
            var d, c;
            this.activeTarget = e;
            b(this.selector).parent(".active").removeClass("active");
            c = this.selector + '[data-target="' + e + '"],' + this.selector + '[href="' + e + '"]';
            d = b(c).parent("li").addClass("active");
            if (d.parent(".dropdown-menu").length) {
                d = d.closest("li.dropdown").addClass("active")
            }
            d.trigger("activate")
        }
    };
    b.fn.scrollspy = function(c) {
        return this.each(function() {
            var f = b(this), e = f.data("scrollspy"), d = typeof c == "object" && c;
            if (!e) {
                f.data("scrollspy", (e = new a(this, d)))
            }
            if (typeof c == "string") {
                e[c]()
            }
        })
    };
    b.fn.scrollspy.Constructor = a;
    b.fn.scrollspy.defaults = {
        offset: 10
    };
    b(window).on("load", function() {
        b('[data-spy="scroll"]').each(function() {
            var c = b(this);
            c.scrollspy(c.data())
        })
    })
}(window.jQuery);
!function(c) {
    var b = function(d) {
        this.element = c(d)
    };
    b.prototype = {
        constructor: b,
        show: function() {
            var j = this.element, g = j.closest("ul:not(.dropdown-menu)"), f = j.attr("data-target"), h, d, i;
            if (!f) {
                f = j.attr("href");
                f = f && f.replace(/.*(?=#[^\s]*$)/, "")
            }
            if (j.parent("li").hasClass("active")) {
                return 
            }
            h = g.find(".active:last a")[0];
            i = c.Event("show", {
                relatedTarget: h
            });
            j.trigger(i);
            if (i.isDefaultPrevented()) {
                return 
            }
            d = c(f);
            this.activate(j.parent("li"), g);
            this.activate(d, d.parent(), function() {
                j.trigger({
                    type: "shown",
                    relatedTarget: h
                })
            })
        },
        activate: function(f, e, i) {
            var d = e.find("> .active"), h = i && c.support.transition && d.hasClass("fade");
            function g() {
                d.removeClass("active").find("> .dropdown-menu > .active").removeClass("active");
                f.addClass("active");
                if (h) {
                    f[0].offsetWidth;
                    f.addClass("in")
                } else {
                    f.removeClass("fade")
                }
                if (f.parent(".dropdown-menu")) {
                    f.closest("li.dropdown").addClass("active")
                }
                i && i()
            }
            h ? d.one(c.support.transition.end, g) : g();
            d.removeClass("in")
        }
    };
    var a = c.fn.tab;
    c.fn.tab = function(d) {
        return this.each(function() {
            var f = c(this), e = f.data("tab");
            if (!e) {
                f.data("tab", (e = new b(this)))
            }
            if (typeof d == "string") {
                e[d]()
            }
        })
    };
    c.fn.tab.Constructor = b;
    c.fn.tab.noConflict = function() {
        c.fn.tab = a;
        return this
    };
    c(document).on("click.tab.data-api", '[data-toggle="tab"], [data-toggle="pill"]', function(d) {
        d.preventDefault();
        c(this).tab("show")
    })
}(window.$);
this.Handlebars = {};
(function(c) {
    c.VERSION = "1.0.rc.1";
    c.helpers = {};
    c.partials = {};
    c.registerHelper = function(e, f, d) {
        if (d) {
            f.not = d
        }
        this.helpers[e] = f
    };
    c.registerPartial = function(d, e) {
        this.partials[d] = e
    };
    c.registerHelper("helperMissing", function(d) {
        if (arguments.length === 2) {
            return undefined
        } else {
            throw new Error("Could not find property '" + d + "'")
        }
    });
    var a = Object.prototype.toString, b = "[object Function]";
    c.registerHelper("blockHelperMissing", function(g, f) {
        var d = f.inverse || function() {}, i = f.fn;
        var e = "";
        var h = a.call(g);
        if (h === b) {
            g = g.call(this)
        }
        if (g === true) {
            return i(this)
        } else {
            if (g === false || g == null) {
                return d(this)
            } else {
                if (h === "[object Array]") {
                    if (g.length > 0) {
                        return c.helpers.each(g, f)
                    } else {
                        return d(this)
                    }
                } else {
                    return i(g)
                }
            }
        }
    });
    c.K = function() {};
    c.createFrame = Object.create || function(d) {
        c.K.prototype = d;
        var e = new c.K();
        c.K.prototype = null;
        return e
    };
    c.registerHelper("each", function(k, g) {
        var l = g.fn, d = g.inverse;
        var f = "", m;
        if (g.data) {
            m = c.createFrame(g.data)
        }
        if (k && k.length > 0) {
            for (var h = 0, e = k.length; h < e; h++) {
                if (m) {
                    m.index = h
                }
                f = f + l(k[h], {
                    data: m
                })
            }
        } else {
            f = d(this)
        }
        return f
    });
    c.registerHelper("if", function(e, d) {
        var f = a.call(e);
        if (f === b) {
            e = e.call(this)
        }
        if (!e || c.Utils.isEmpty(e)) {
            return d.inverse(this)
        } else {
            return d.fn(this)
        }
    });
    c.registerHelper("unless", function(f, e) {
        var g = e.fn, d = e.inverse;
        e.fn = d;
        e.inverse = g;
        return c.helpers["if"].call(this, f, e)
    });
    c.registerHelper("with", function(e, d) {
        return d.fn(e)
    });
    c.registerHelper("log", function(d) {
        c.log(d)
    })
}(this.Handlebars));
var handlebars = (function() {
    var g = {
        trace: function c() {},
        yy: {},
        symbols_: {
            error: 2,
            root: 3,
            program: 4,
            EOF: 5,
            statements: 6,
            simpleInverse: 7,
            statement: 8,
            openInverse: 9,
            closeBlock: 10,
            openBlock: 11,
            mustache: 12,
            partial: 13,
            CONTENT: 14,
            COMMENT: 15,
            OPEN_BLOCK: 16,
            inMustache: 17,
            CLOSE: 18,
            OPEN_INVERSE: 19,
            OPEN_ENDBLOCK: 20,
            path: 21,
            OPEN: 22,
            OPEN_UNESCAPED: 23,
            OPEN_PARTIAL: 24,
            params: 25,
            hash: 26,
            DATA: 27,
            param: 28,
            STRING: 29,
            INTEGER: 30,
            BOOLEAN: 31,
            hashSegments: 32,
            hashSegment: 33,
            ID: 34,
            EQUALS: 35,
            pathSegments: 36,
            SEP: 37,
            "$accept": 0,
            "$end": 1
        },
        terminals_: {
            2: "error",
            5: "EOF",
            14: "CONTENT",
            15: "COMMENT",
            16: "OPEN_BLOCK",
            18: "CLOSE",
            19: "OPEN_INVERSE",
            20: "OPEN_ENDBLOCK",
            22: "OPEN",
            23: "OPEN_UNESCAPED",
            24: "OPEN_PARTIAL",
            27: "DATA",
            29: "STRING",
            30: "INTEGER",
            31: "BOOLEAN",
            34: "ID",
            35: "EQUALS",
            37: "SEP"
        },
        productions_: [0, [3, 2], [4, 3], [4, 1], [4, 0], [6, 1], [6, 2], [8, 3], [8, 3], [8, 1], [8, 1], [8, 1], [8, 1], [11, 3], [9, 3], [10, 3], [12, 3], [12, 3], [13, 3], [13, 4], [7, 2], [17, 3], [17, 2], [17, 2], [17, 1], [17, 1], [25, 2], [25, 1], [28, 1], [28, 1], [28, 1], [28, 1], [28, 1], [26, 1], [32, 2], [32, 1], [33, 3], [33, 3], [33, 3], [33, 3], [33, 3], [21, 1], [36, 3], [36, 1]],
        performAction: function b(h, k, l, o, n, j, m) {
            var i = j.length - 1;
            switch (n) {
            case 1:
                return j[i - 1];
                break;
            case 2:
                this.$ = new o.ProgramNode(j[i - 2], j[i]);
                break;
            case 3:
                this.$ = new o.ProgramNode(j[i]);
                break;
            case 4:
                this.$ = new o.ProgramNode([]);
                break;
            case 5:
                this.$ = [j[i]];
                break;
            case 6:
                j[i - 1].push(j[i]);
                this.$ = j[i - 1];
                break;
            case 7:
                this.$ = new o.BlockNode(j[i - 2], j[i - 1].inverse, j[i - 1], j[i]);
                break;
            case 8:
                this.$ = new o.BlockNode(j[i - 2], j[i - 1], j[i - 1].inverse, j[i]);
                break;
            case 9:
                this.$ = j[i];
                break;
            case 10:
                this.$ = j[i];
                break;
            case 11:
                this.$ = new o.ContentNode(j[i]);
                break;
            case 12:
                this.$ = new o.CommentNode(j[i]);
                break;
            case 13:
                this.$ = new o.MustacheNode(j[i - 1][0], j[i - 1][1]);
                break;
            case 14:
                this.$ = new o.MustacheNode(j[i - 1][0], j[i - 1][1]);
                break;
            case 15:
                this.$ = j[i - 1];
                break;
            case 16:
                this.$ = new o.MustacheNode(j[i - 1][0], j[i - 1][1]);
                break;
            case 17:
                this.$ = new o.MustacheNode(j[i - 1][0], j[i - 1][1], true);
                break;
            case 18:
                this.$ = new o.PartialNode(j[i - 1]);
                break;
            case 19:
                this.$ = new o.PartialNode(j[i - 2], j[i - 1]);
                break;
            case 20:
                break;
            case 21:
                this.$ = [[j[i - 2]].concat(j[i - 1]), j[i]];
                break;
            case 22:
                this.$ = [[j[i - 1]].concat(j[i]), null];
                break;
            case 23:
                this.$ = [[j[i - 1]], j[i]];
                break;
            case 24:
                this.$ = [[j[i]], null];
                break;
            case 25:
                this.$ = [[new o.DataNode(j[i])], null];
                break;
            case 26:
                j[i - 1].push(j[i]);
                this.$ = j[i - 1];
                break;
            case 27:
                this.$ = [j[i]];
                break;
            case 28:
                this.$ = j[i];
                break;
            case 29:
                this.$ = new o.StringNode(j[i]);
                break;
            case 30:
                this.$ = new o.IntegerNode(j[i]);
                break;
            case 31:
                this.$ = new o.BooleanNode(j[i]);
                break;
            case 32:
                this.$ = new o.DataNode(j[i]);
                break;
            case 33:
                this.$ = new o.HashNode(j[i]);
                break;
            case 34:
                j[i - 1].push(j[i]);
                this.$ = j[i - 1];
                break;
            case 35:
                this.$ = [j[i]];
                break;
            case 36:
                this.$ = [j[i - 2], j[i]];
                break;
            case 37:
                this.$ = [j[i - 2], new o.StringNode(j[i])];
                break;
            case 38:
                this.$ = [j[i - 2], new o.IntegerNode(j[i])];
                break;
            case 39:
                this.$ = [j[i - 2], new o.BooleanNode(j[i])];
                break;
            case 40:
                this.$ = [j[i - 2], new o.DataNode(j[i])];
                break;
            case 41:
                this.$ = new o.IdNode(j[i]);
                break;
            case 42:
                j[i - 2].push(j[i]);
                this.$ = j[i - 2];
                break;
            case 43:
                this.$ = [j[i]];
                break
            }
        },
        table: [{
            3: 1,
            4: 2,
            5: [2, 4],
            6: 3,
            8: 4,
            9: 5,
            11: 6,
            12: 7,
            13: 8,
            14: [1, 9],
            15: [1, 10],
            16: [1, 12],
            19: [1, 11],
            22: [1, 13],
            23: [1, 14],
            24: [1, 15]
        }, {
            1: [3]
        }, {
            5: [1, 16]
        }, {
            5: [2, 3],
            7: 17,
            8: 18,
            9: 5,
            11: 6,
            12: 7,
            13: 8,
            14: [1, 9],
            15: [1, 10],
            16: [1, 12],
            19: [1, 19],
            20: [2, 3],
            22: [1, 13],
            23: [1, 14],
            24: [1, 15]
        }, {
            5: [2, 5],
            14: [2, 5],
            15: [2, 5],
            16: [2, 5],
            19: [2, 5],
            20: [2, 5],
            22: [2, 5],
            23: [2, 5],
            24: [2, 5]
        }, {
            4: 20,
            6: 3,
            8: 4,
            9: 5,
            11: 6,
            12: 7,
            13: 8,
            14: [1, 9],
            15: [1, 10],
            16: [1, 12],
            19: [1, 11],
            20: [2, 4],
            22: [1, 13],
            23: [1, 14],
            24: [1, 15]
        }, {
            4: 21,
            6: 3,
            8: 4,
            9: 5,
            11: 6,
            12: 7,
            13: 8,
            14: [1, 9],
            15: [1, 10],
            16: [1, 12],
            19: [1, 11],
            20: [2, 4],
            22: [1, 13],
            23: [1, 14],
            24: [1, 15]
        }, {
            5: [2, 9],
            14: [2, 9],
            15: [2, 9],
            16: [2, 9],
            19: [2, 9],
            20: [2, 9],
            22: [2, 9],
            23: [2, 9],
            24: [2, 9]
        }, {
            5: [2, 10],
            14: [2, 10],
            15: [2, 10],
            16: [2, 10],
            19: [2, 10],
            20: [2, 10],
            22: [2, 10],
            23: [2, 10],
            24: [2, 10]
        }, {
            5: [2, 11],
            14: [2, 11],
            15: [2, 11],
            16: [2, 11],
            19: [2, 11],
            20: [2, 11],
            22: [2, 11],
            23: [2, 11],
            24: [2, 11]
        }, {
            5: [2, 12],
            14: [2, 12],
            15: [2, 12],
            16: [2, 12],
            19: [2, 12],
            20: [2, 12],
            22: [2, 12],
            23: [2, 12],
            24: [2, 12]
        }, {
            17: 22,
            21: 23,
            27: [1, 24],
            34: [1, 26],
            36: 25
        }, {
            17: 27,
            21: 23,
            27: [1, 24],
            34: [1, 26],
            36: 25
        }, {
            17: 28,
            21: 23,
            27: [1, 24],
            34: [1, 26],
            36: 25
        }, {
            17: 29,
            21: 23,
            27: [1, 24],
            34: [1, 26],
            36: 25
        }, {
            21: 30,
            34: [1, 26],
            36: 25
        }, {
            1: [2, 1]
        }, {
            6: 31,
            8: 4,
            9: 5,
            11: 6,
            12: 7,
            13: 8,
            14: [1, 9],
            15: [1, 10],
            16: [1, 12],
            19: [1, 11],
            22: [1, 13],
            23: [1, 14],
            24: [1, 15]
        }, {
            5: [2, 6],
            14: [2, 6],
            15: [2, 6],
            16: [2, 6],
            19: [2, 6],
            20: [2, 6],
            22: [2, 6],
            23: [2, 6],
            24: [2, 6]
        }, {
            17: 22,
            18: [1, 32],
            21: 23,
            27: [1, 24],
            34: [1, 26],
            36: 25
        }, {
            10: 33,
            20: [1, 34]
        }, {
            10: 35,
            20: [1, 34]
        }, {
            18: [1, 36]
        }, {
            18: [2, 24],
            21: 41,
            25: 37,
            26: 38,
            27: [1, 45],
            28: 39,
            29: [1, 42],
            30: [1, 43],
            31: [1, 44],
            32: 40,
            33: 46,
            34: [1, 47],
            36: 25
        }, {
            18: [2, 25]
        }, {
            18: [2, 41],
            27: [2, 41],
            29: [2, 41],
            30: [2, 41],
            31: [2, 41],
            34: [2, 41],
            37: [1, 48]
        }, {
            18: [2, 43],
            27: [2, 43],
            29: [2, 43],
            30: [2, 43],
            31: [2, 43],
            34: [2, 43],
            37: [2, 43]
        }, {
            18: [1, 49]
        }, {
            18: [1, 50]
        }, {
            18: [1, 51]
        }, {
            18: [1, 52],
            21: 53,
            34: [1, 26],
            36: 25
        }, {
            5: [2, 2],
            8: 18,
            9: 5,
            11: 6,
            12: 7,
            13: 8,
            14: [1, 9],
            15: [1, 10],
            16: [1, 12],
            19: [1, 11],
            20: [2, 2],
            22: [1, 13],
            23: [1, 14],
            24: [1, 15]
        }, {
            14: [2, 20],
            15: [2, 20],
            16: [2, 20],
            19: [2, 20],
            22: [2, 20],
            23: [2, 20],
            24: [2, 20]
        }, {
            5: [2, 7],
            14: [2, 7],
            15: [2, 7],
            16: [2, 7],
            19: [2, 7],
            20: [2, 7],
            22: [2, 7],
            23: [2, 7],
            24: [2, 7]
        }, {
            21: 54,
            34: [1, 26],
            36: 25
        }, {
            5: [2, 8],
            14: [2, 8],
            15: [2, 8],
            16: [2, 8],
            19: [2, 8],
            20: [2, 8],
            22: [2, 8],
            23: [2, 8],
            24: [2, 8]
        }, {
            14: [2, 14],
            15: [2, 14],
            16: [2, 14],
            19: [2, 14],
            20: [2, 14],
            22: [2, 14],
            23: [2, 14],
            24: [2, 14]
        }, {
            18: [2, 22],
            21: 41,
            26: 55,
            27: [1, 45],
            28: 56,
            29: [1, 42],
            30: [1, 43],
            31: [1, 44],
            32: 40,
            33: 46,
            34: [1, 47],
            36: 25
        }, {
            18: [2, 23]
        }, {
            18: [2, 27],
            27: [2, 27],
            29: [2, 27],
            30: [2, 27],
            31: [2, 27],
            34: [2, 27]
        }, {
            18: [2, 33],
            33: 57,
            34: [1, 58]
        }, {
            18: [2, 28],
            27: [2, 28],
            29: [2, 28],
            30: [2, 28],
            31: [2, 28],
            34: [2, 28]
        }, {
            18: [2, 29],
            27: [2, 29],
            29: [2, 29],
            30: [2, 29],
            31: [2, 29],
            34: [2, 29]
        }, {
            18: [2, 30],
            27: [2, 30],
            29: [2, 30],
            30: [2, 30],
            31: [2, 30],
            34: [2, 30]
        }, {
            18: [2, 31],
            27: [2, 31],
            29: [2, 31],
            30: [2, 31],
            31: [2, 31],
            34: [2, 31]
        }, {
            18: [2, 32],
            27: [2, 32],
            29: [2, 32],
            30: [2, 32],
            31: [2, 32],
            34: [2, 32]
        }, {
            18: [2, 35],
            34: [2, 35]
        }, {
            18: [2, 43],
            27: [2, 43],
            29: [2, 43],
            30: [2, 43],
            31: [2, 43],
            34: [2, 43],
            35: [1, 59],
            37: [2, 43]
        }, {
            34: [1, 60]
        }, {
            14: [2, 13],
            15: [2, 13],
            16: [2, 13],
            19: [2, 13],
            20: [2, 13],
            22: [2, 13],
            23: [2, 13],
            24: [2, 13]
        }, {
            5: [2, 16],
            14: [2, 16],
            15: [2, 16],
            16: [2, 16],
            19: [2, 16],
            20: [2, 16],
            22: [2, 16],
            23: [2, 16],
            24: [2, 16]
        }, {
            5: [2, 17],
            14: [2, 17],
            15: [2, 17],
            16: [2, 17],
            19: [2, 17],
            20: [2, 17],
            22: [2, 17],
            23: [2, 17],
            24: [2, 17]
        }, {
            5: [2, 18],
            14: [2, 18],
            15: [2, 18],
            16: [2, 18],
            19: [2, 18],
            20: [2, 18],
            22: [2, 18],
            23: [2, 18],
            24: [2, 18]
        }, {
            18: [1, 61]
        }, {
            18: [1, 62]
        }, {
            18: [2, 21]
        }, {
            18: [2, 26],
            27: [2, 26],
            29: [2, 26],
            30: [2, 26],
            31: [2, 26],
            34: [2, 26]
        }, {
            18: [2, 34],
            34: [2, 34]
        }, {
            35: [1, 59]
        }, {
            21: 63,
            27: [1, 67],
            29: [1, 64],
            30: [1, 65],
            31: [1, 66],
            34: [1, 26],
            36: 25
        }, {
            18: [2, 42],
            27: [2, 42],
            29: [2, 42],
            30: [2, 42],
            31: [2, 42],
            34: [2, 42],
            37: [2, 42]
        }, {
            5: [2, 19],
            14: [2, 19],
            15: [2, 19],
            16: [2, 19],
            19: [2, 19],
            20: [2, 19],
            22: [2, 19],
            23: [2, 19],
            24: [2, 19]
        }, {
            5: [2, 15],
            14: [2, 15],
            15: [2, 15],
            16: [2, 15],
            19: [2, 15],
            20: [2, 15],
            22: [2, 15],
            23: [2, 15],
            24: [2, 15]
        }, {
            18: [2, 36],
            34: [2, 36]
        }, {
            18: [2, 37],
            34: [2, 37]
        }, {
            18: [2, 38],
            34: [2, 38]
        }, {
            18: [2, 39],
            34: [2, 39]
        }, {
            18: [2, 40],
            34: [2, 40]
        }
        ],
        defaultActions: {
            16: [2, 1],
            24: [2, 25],
            38: [2, 23],
            55: [2, 21]
        },
        parseError: function d(i, h) {
            throw new Error(i)
        },
        parse: function f(s) {
            var z = this, n = [0], I = [null], u = [], J = this.table, i = "", t = 0, G = 0, k = 0, q = 2, w = 1;
            this.lexer.setInput(s);
            this.lexer.yy = this.yy;
            this.yy.lexer = this.lexer;
            this.yy.parser = this;
            if (typeof this.lexer.yylloc == "undefined") {
                this.lexer.yylloc = {}
            }
            var j = this.lexer.yylloc;
            u.push(j);
            var l = this.lexer.options && this.lexer.options.ranges;
            if (typeof this.yy.parseError === "function") {
                this.parseError = this.yy.parseError
            }
            function y(p) {
                n.length = n.length - 2 * p;
                I.length = I.length - p;
                u.length = u.length - p
            }
            function x() {
                var p;
                p = z.lexer.lex() || 1;
                if (typeof p !== "number") {
                    p = z.symbols_[p] || p
                }
                return p
            }
            var F, B, m, E, K, v, D = {}, A, H, h, o;
            while (true) {
                m = n[n.length - 1];
                if (this.defaultActions[m]) {
                    E = this.defaultActions[m]
                } else {
                    if (F === null || typeof F == "undefined") {
                        F = x()
                    }
                    E = J[m] && J[m][F]
                }
                if (typeof E === "undefined" ||!E.length ||!E[0]) {
                    var C = "";
                    if (!k) {
                        o = [];
                        for (A in J[m]) {
                            if (this.terminals_[A] && A > 2) {
                                o.push("'" + this.terminals_[A] + "'")
                            }
                        }
                        if (this.lexer.showPosition) {
                            C = "Parse error on line " + (t + 1) + ":\n" + this.lexer.showPosition() + "\nExpecting " + o.join(", ") + ", got '" + (this.terminals_[F] || F) + "'"
                        } else {
                            C = "Parse error on line " + (t + 1) + ": Unexpected " + (F == 1 ? "end of input" : "'" + (this.terminals_[F] || F) + "'")
                        }
                        this.parseError(C, {
                            text: this.lexer.match,
                            token: this.terminals_[F] || F,
                            line: this.lexer.yylineno,
                            loc: j,
                            expected: o
                        })
                    }
                }
                if (E[0] instanceof Array && E.length > 1) {
                    throw new Error("Parse Error: multiple actions possible at state: " + m + ", token: " + F)
                }
                switch (E[0]) {
                case 1:
                    n.push(F);
                    I.push(this.lexer.yytext);
                    u.push(this.lexer.yylloc);
                    n.push(E[1]);
                    F = null;
                    if (!B) {
                        G = this.lexer.yyleng;
                        i = this.lexer.yytext;
                        t = this.lexer.yylineno;
                        j = this.lexer.yylloc;
                        if (k > 0) {
                            k--
                        }
                    } else {
                        F = B;
                        B = null
                    }
                    break;
                case 2:
                    H = this.productions_[E[1]][1];
                    D.$ = I[I.length - H];
                    D._$ = {
                        first_line: u[u.length - (H || 1)].first_line,
                        last_line: u[u.length - 1].last_line,
                        first_column: u[u.length - (H || 1)].first_column,
                        last_column: u[u.length - 1].last_column
                    };
                    if (l) {
                        D._$.range = [u[u.length - (H || 1)].range[0], u[u.length - 1].range[1]]
                    }
                    v = this.performAction.call(D, i, G, t, this.yy, E[1], I, u);
                    if (typeof v !== "undefined") {
                        return v
                    }
                    if (H) {
                        n = n.slice(0, - 1 * H * 2);
                        I = I.slice(0, - 1 * H);
                        u = u.slice(0, - 1 * H)
                    }
                    n.push(this.productions_[E[1]][0]);
                    I.push(D.$);
                    u.push(D._$);
                    h = J[n[n.length - 2]][n[n.length - 1]];
                    n.push(h);
                    break;
                case 3:
                    return true
                }
            }
            return true
        }
    };
    var a = (function() {
        var k = ({
            EOF: 1,
            parseError: function m(p, o) {
                if (this.yy.parser) {
                    this.yy.parser.parseError(p, o)
                } else {
                    throw new Error(p)
                }
            },
            setInput: function(o) {
                this._input = o;
                this._more = this._less = this.done = false;
                this.yylineno = this.yyleng = 0;
                this.yytext = this.matched = this.match = "";
                this.conditionStack = ["INITIAL"];
                this.yylloc = {
                    first_line: 1,
                    first_column: 0,
                    last_line: 1,
                    last_column: 0
                };
                if (this.options.ranges) {
                    this.yylloc.range = [0, 0]
                }
                this.offset = 0;
                return this
            },
            input: function() {
                var p = this._input[0];
                this.yytext += p;
                this.yyleng++;
                this.offset++;
                this.match += p;
                this.matched += p;
                var o = p.match(/(?:\r\n?|\n).*/g);
                if (o) {
                    this.yylineno++;
                    this.yylloc.last_line++
                } else {
                    this.yylloc.last_column++
                }
                if (this.options.ranges) {
                    this.yylloc.range[1]++
                }
                this._input = this._input.slice(1);
                return p
            },
            unput: function(q) {
                var o = q.length;
                var p = q.split(/(?:\r\n?|\n)/g);
                this._input = q + this._input;
                this.yytext = this.yytext.substr(0, this.yytext.length - o - 1);
                this.offset -= o;
                var t = this.match.split(/(?:\r\n?|\n)/g);
                this.match = this.match.substr(0, this.match.length - 1);
                this.matched = this.matched.substr(0, this.matched.length - 1);
                if (p.length - 1) {
                    this.yylineno -= p.length - 1
                }
                var s = this.yylloc.range;
                this.yylloc = {
                    first_line: this.yylloc.first_line,
                    last_line: this.yylineno + 1,
                    first_column: this.yylloc.first_column,
                    last_column: p ? (p.length === t.length ? this.yylloc.first_column : 0) + t[t.length - p.length].length - p[0].length: this.yylloc.first_column - o
                };
                if (this.options.ranges) {
                    this.yylloc.range = [s[0], s[0] + this.yyleng - o]
                }
                return this
            },
            more: function() {
                this._more = true;
                return this
            },
            less: function(o) {
                this.unput(this.match.slice(o))
            },
            pastInput: function() {
                var o = this.matched.substr(0, this.matched.length - this.match.length);
                return (o.length > 20 ? "..." : "") + o.substr( - 20).replace(/\n/g, "")
            },
            upcomingInput: function() {
                var o = this.match;
                if (o.length < 20) {
                    o += this._input.substr(0, 20 - o.length)
                }
                return (o.substr(0, 20) + (o.length > 20 ? "..." : "")).replace(/\n/g, "")
            },
            showPosition: function() {
                var o = this.pastInput();
                var p = new Array(o.length + 1).join("-");
                return o + this.upcomingInput() + "\n" + p + "^"
            },
            next: function() {
                if (this.done) {
                    return this.EOF
                }
                if (!this._input) {
                    this.done = true
                }
                var u, s, p, r, q, o;
                if (!this._more) {
                    this.yytext = "";
                    this.match = ""
                }
                var v = this._currentRules();
                for (var t = 0; t < v.length; t++) {
                    p = this._input.match(this.rules[v[t]]);
                    if (p && (!s || p[0].length > s[0].length)) {
                        s = p;
                        r = t;
                        if (!this.options.flex) {
                            break
                        }
                    }
                }
                if (s) {
                    o = s[0].match(/(?:\r\n?|\n).*/g);
                    if (o) {
                        this.yylineno += o.length
                    }
                    this.yylloc = {
                        first_line: this.yylloc.last_line,
                        last_line: this.yylineno + 1,
                        first_column: this.yylloc.last_column,
                        last_column: o ? o[o.length - 1].length - o[o.length - 1].match(/\r?\n?/)[0].length: this.yylloc.last_column + s[0].length
                    };
                    this.yytext += s[0];
                    this.match += s[0];
                    this.matches = s;
                    this.yyleng = this.yytext.length;
                    if (this.options.ranges) {
                        this.yylloc.range = [this.offset, this.offset += this.yyleng]
                    }
                    this._more = false;
                    this._input = this._input.slice(s[0].length);
                    this.matched += s[0];
                    u = this.performAction.call(this, this.yy, this, v[r], this.conditionStack[this.conditionStack.length - 1]);
                    if (this.done && this._input) {
                        this.done = false
                    }
                    if (u) {
                        return u
                    } else {
                        return 
                    }
                }
                if (this._input === "") {
                    return this.EOF
                } else {
                    return this.parseError("Lexical error on line " + (this.yylineno + 1) + ". Unrecognized text.\n" + this.showPosition(), {
                        text: "",
                        token: null,
                        line: this.yylineno
                    })
                }
            },
            lex: function h() {
                var o = this.next();
                if (typeof o !== "undefined") {
                    return o
                } else {
                    return this.lex()
                }
            },
            begin: function i(o) {
                this.conditionStack.push(o)
            },
            popState: function n() {
                return this.conditionStack.pop()
            },
            _currentRules: function l() {
                return this.conditions[this.conditionStack[this.conditionStack.length - 1]].rules
            },
            topState: function() {
                return this.conditionStack[this.conditionStack.length - 2]
            },
            pushState: function i(o) {
                this.begin(o)
            }
        });
        k.options = {};
        k.performAction = function j(s, p, r, o) {
            var q = o;
            switch (r) {
            case 0:
                if (p.yytext.slice( - 1) !== "\\") {
                    this.begin("mu")
                }
                if (p.yytext.slice( - 1) === "\\") {
                    p.yytext = p.yytext.substr(0, p.yyleng - 1), this.begin("emu")
                }
                if (p.yytext) {
                    return 14
                }
                break;
            case 1:
                return 14;
                break;
            case 2:
                if (p.yytext.slice( - 1) !== "\\") {
                    this.popState()
                }
                if (p.yytext.slice( - 1) === "\\") {
                    p.yytext = p.yytext.substr(0, p.yyleng - 1)
                }
                return 14;
                break;
            case 3:
                return 24;
                break;
            case 4:
                return 16;
                break;
            case 5:
                return 20;
                break;
            case 6:
                return 19;
                break;
            case 7:
                return 19;
                break;
            case 8:
                return 23;
                break;
            case 9:
                return 23;
                break;
            case 10:
                p.yytext = p.yytext.substr(3, p.yyleng - 5);
                this.popState();
                return 15;
                break;
            case 11:
                return 22;
                break;
            case 12:
                return 35;
                break;
            case 13:
                return 34;
                break;
            case 14:
                return 34;
                break;
            case 15:
                return 37;
                break;
            case 16:
                break;
            case 17:
                this.popState();
                return 18;
                break;
            case 18:
                this.popState();
                return 18;
                break;
            case 19:
                p.yytext = p.yytext.substr(1, p.yyleng - 2).replace(/\\"/g, '"');
                return 29;
                break;
            case 20:
                p.yytext = p.yytext.substr(1, p.yyleng - 2).replace(/\\"/g, '"');
                return 29;
                break;
            case 21:
                p.yytext = p.yytext.substr(1);
                return 27;
                break;
            case 22:
                return 31;
                break;
            case 23:
                return 31;
                break;
            case 24:
                return 30;
                break;
            case 25:
                return 34;
                break;
            case 26:
                p.yytext = p.yytext.substr(1, p.yyleng - 2);
                return 34;
                break;
            case 27:
                return "INVALID";
                break;
            case 28:
                return 5;
                break
            }
        };
        k.rules = [/^(?:[^\x00]*?(?=(\{\{)))/, /^(?:[^\x00]+)/, /^(?:[^\x00]{2,}?(?=(\{\{|$)))/, /^(?:\{\{>)/, /^(?:\{\{#)/, /^(?:\{\{\/)/, /^(?:\{\{\^)/, /^(?:\{\{\s*else\b)/, /^(?:\{\{\{)/, /^(?:\{\{&)/, /^(?:\{\{![\s\S]*?\}\})/, /^(?:\{\{)/, /^(?:=)/, /^(?:\.(?=[} ]))/, /^(?:\.\.)/, /^(?:[\/.])/, /^(?:\s+)/, /^(?:\}\}\})/, /^(?:\}\})/, /^(?:"(\\["]|[^"])*")/, /^(?:'(\\[']|[^'])*')/, /^(?:@[a-zA-Z]+)/, /^(?:true(?=[}\s]))/, /^(?:false(?=[}\s]))/, /^(?:[0-9]+(?=[}\s]))/, /^(?:[a-zA-Z0-9_$-]+(?=[=}\s\/.]))/, /^(?:\[[^\]]*\])/, /^(?:.)/, /^(?:$)/];
        k.conditions = {
            mu: {
                rules: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                inclusive: false
            },
            emu: {
                rules: [2],
                inclusive: false
            },
            INITIAL: {
                rules: [0, 1, 28],
                inclusive: true
            }
        };
        return k
    })();
    g.lexer = a;
    function e() {
        this.yy = {}
    }
    e.prototype = g;
    g.Parser = e;
    return new e
})();
if (typeof require !== "undefined" && typeof exports !== "undefined") {
    exports.parser = handlebars;
    exports.Parser = handlebars.Parser;
    exports.parse = function() {
        return handlebars.parse.apply(handlebars, arguments)
    };
    exports.main = function commonjsMain(a) {
        if (!a[1]) {
            throw new Error("Usage: " + a[0] + " FILE")
        }
        var c, b;
        if (typeof process !== "undefined") {
            c = require("fs").readFileSync(require("path").resolve(a[1]), "utf8")
        } else {
            c = require("file").path(require("file").cwd()).join(a[1]).read({
                charset: "utf-8"
            })
        }
        return exports.parser.parse(c)
    };
    if (typeof module !== "undefined" && require.main === module) {
        exports.main(typeof process !== "undefined" ? process.argv.slice(1) : require("system").args)
    }
}
Handlebars.Parser = handlebars;
Handlebars.parse = function(a) {
    Handlebars.Parser.yy = Handlebars.AST;
    return Handlebars.Parser.parse(a)
};
Handlebars.print = function(a) {
    return new Handlebars.PrintVisitor().accept(a)
};
Handlebars.logger = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
    level: 3,
    log: function(b, a) {}
};
Handlebars.log = function(b, a) {
    Handlebars.logger.log(b, a)
};
(function() {
    Handlebars.AST = {};
    Handlebars.AST.ProgramNode = function(c, b) {
        this.type = "program";
        this.statements = c;
        if (b) {
            this.inverse = new Handlebars.AST.ProgramNode(b)
        }
    };
    Handlebars.AST.MustacheNode = function(g, d, c) {
        this.type = "mustache";
        this.escaped=!c;
        this.hash = d;
        var f = this.id = g[0];
        var e = this.params = g.slice(1);
        var b = this.eligibleHelper = f.isSimple;
        this.isHelper = b && (e.length || d)
    };
    Handlebars.AST.PartialNode = function(c, b) {
        this.type = "partial";
        this.id = c;
        this.context = b
    };
    var a = function(b, c) {
        if (b.original !== c.original) {
            throw new Handlebars.Exception(b.original + " doesn't match " + c.original)
        }
    };
    Handlebars.AST.BlockNode = function(d, c, b, e) {
        a(d.id, e);
        this.type = "block";
        this.mustache = d;
        this.program = c;
        this.inverse = b;
        if (this.inverse&&!this.program) {
            this.isInverse = true
        }
    };
    Handlebars.AST.ContentNode = function(b) {
        this.type = "content";
        this.string = b
    };
    Handlebars.AST.HashNode = function(b) {
        this.type = "hash";
        this.pairs = b
    };
    Handlebars.AST.IdNode = function(f) {
        this.type = "ID";
        this.original = f.join(".");
        var d = [], g = 0;
        for (var e = 0, b = f.length; e < b; e++) {
            var c = f[e];
            if (c === "..") {
                g++
            } else {
                if (c === "." || c === "this") {
                    this.isScoped = true
                } else {
                    d.push(c)
                }
            }
        }
        this.parts = d;
        this.string = d.join(".");
        this.depth = g;
        this.isSimple = f.length === 1&&!this.isScoped && g === 0
    };
    Handlebars.AST.DataNode = function(b) {
        this.type = "DATA";
        this.id = b
    };
    Handlebars.AST.StringNode = function(b) {
        this.type = "STRING";
        this.string = b
    };
    Handlebars.AST.IntegerNode = function(b) {
        this.type = "INTEGER";
        this.integer = b
    };
    Handlebars.AST.BooleanNode = function(b) {
        this.type = "BOOLEAN";
        this.bool = b
    };
    Handlebars.AST.CommentNode = function(b) {
        this.type = "comment";
        this.comment = b
    }
})();
Handlebars.Exception = function(b) {
    var a = Error.prototype.constructor.apply(this, arguments);
    for (var c in a) {
        if (a.hasOwnProperty(c)) {
            this[c] = a[c]
        }
    }
    this.message = a.message
};
Handlebars.Exception.prototype = new Error();
Handlebars.SafeString = function(a) {
    this.string = a
};
Handlebars.SafeString.prototype.toString = function() {
    return this.string.toString()
};
(function() {
    var c = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "`": "&#x60;"
    };
    var d = /[&<>"'`]/g;
    var b = /[&<>"'`]/;
    var a = function(e) {
        return c[e] || "&amp;"
    };
    Handlebars.Utils = {
        escapeExpression: function(e) {
            if (e instanceof Handlebars.SafeString) {
                return e.toString()
            } else {
                if (e == null || e === false) {
                    return ""
                }
            }
            if (!b.test(e)) {
                return e
            }
            return e.replace(d, a)
        },
        isEmpty: function(e) {
            if (typeof e === "undefined") {
                return true
            } else {
                if (e === null) {
                    return true
                } else {
                    if (e === false) {
                        return true
                    } else {
                        if (Object.prototype.toString.call(e) === "[object Array]" && e.length === 0) {
                            return true
                        } else {
                            return false
                        }
                    }
                }
            }
        }
    }
})();
Handlebars.Compiler = function() {};
Handlebars.JavaScriptCompiler = function() {};
(function(g, f) {
    g.prototype = {
        compiler: g,
        disassemble: function() {
            var p = this.opcodes, o, m = [], r, q;
            for (var n = 0, h = p.length; n < h; n++) {
                o = p[n];
                if (o.opcode === "DECLARE") {
                    m.push("DECLARE " + o.name + "=" + o.value)
                } else {
                    r = [];
                    for (var k = 0; k < o.args.length; k++) {
                        q = o.args[k];
                        if (typeof q === "string") {
                            q = '"' + q.replace("\n", "\\n") + '"'
                        }
                        r.push(q)
                    }
                    m.push(o.opcode + " " + r.join(" "))
                }
            }
            return m.join("\n")
        },
        guid: 0,
        compile: function(h, j) {
            this.children = [];
            this.depths = {
                list: []
            };
            this.options = j;
            var k = this.options.knownHelpers;
            this.options.knownHelpers = {
                helperMissing: true,
                blockHelperMissing: true,
                each: true,
                "if": true,
                unless: true,
                "with": true,
                log: true
            };
            if (k) {
                for (var i in k) {
                    this.options.knownHelpers[i] = k[i]
                }
            }
            return this.program(h)
        },
        accept: function(h) {
            return this[h.type](h)
        },
        program: function(k) {
            var j = k.statements, n;
            this.opcodes = [];
            for (var m = 0, h = j.length; m < h; m++) {
                n = j[m];
                this[n.type](n)
            }
            this.isSimple = h === 1;
            this.depths.list = this.depths.list.sort(function(l, i) {
                return l - i
            });
            return this
        },
        compileProgram: function(k) {
            var h = new this.compiler().compile(k, this.options);
            var m = this.guid++, o;
            this.usePartial = this.usePartial || h.usePartial;
            this.children[m] = h;
            for (var n = 0, j = h.depths.list.length; n < j; n++) {
                o = h.depths.list[n];
                if (o < 2) {
                    continue
                } else {
                    this.addDepth(o - 1)
                }
            }
            return m
        },
        block: function(l) {
            var k = l.mustache, i = l.program, h = l.inverse;
            if (i) {
                i = this.compileProgram(i)
            }
            if (h) {
                h = this.compileProgram(h)
            }
            var j = this.classifyMustache(k);
            if (j === "helper") {
                this.helperMustache(k, i, h)
            } else {
                if (j === "simple") {
                    this.simpleMustache(k);
                    this.opcode("pushProgram", i);
                    this.opcode("pushProgram", h);
                    this.opcode("pushLiteral", "{}");
                    this.opcode("blockValue")
                } else {
                    this.ambiguousMustache(k, i, h);
                    this.opcode("pushProgram", i);
                    this.opcode("pushProgram", h);
                    this.opcode("pushLiteral", "{}");
                    this.opcode("ambiguousBlockValue")
                }
            }
            this.opcode("append")
        },
        hash: function(m) {
            var k = m.pairs, o, n;
            this.opcode("push", "{}");
            for (var j = 0, h = k.length; j < h; j++) {
                o = k[j];
                n = o[1];
                this.accept(n);
                this.opcode("assignToHash", o[0])
            }
        },
        partial: function(h) {
            var i = h.id;
            this.usePartial = true;
            if (h.context) {
                this.ID(h.context)
            } else {
                this.opcode("push", "depth0")
            }
            this.opcode("invokePartial", i.original);
            this.opcode("append")
        },
        content: function(h) {
            this.opcode("appendContent", h.string)
        },
        mustache: function(j) {
            var h = this.options;
            var i = this.classifyMustache(j);
            if (i === "simple") {
                this.simpleMustache(j)
            } else {
                if (i === "helper") {
                    this.helperMustache(j)
                } else {
                    this.ambiguousMustache(j)
                }
            }
            if (j.escaped&&!h.noEscape) {
                this.opcode("appendEscaped")
            } else {
                this.opcode("append")
            }
        },
        ambiguousMustache: function(k, i, h) {
            var l = k.id, j = l.parts[0];
            this.opcode("getContext", l.depth);
            this.opcode("pushProgram", i);
            this.opcode("pushProgram", h);
            this.opcode("invokeAmbiguous", j)
        },
        simpleMustache: function(j, i, h) {
            var k = j.id;
            if (k.type === "DATA") {
                this.DATA(k)
            } else {
                if (k.parts.length) {
                    this.ID(k)
                } else {
                    this.addDepth(k.depth);
                    this.opcode("getContext", k.depth);
                    this.opcode("pushContext")
                }
            }
            this.opcode("resolvePossibleLambda")
        },
        helperMustache: function(k, i, h) {
            var l = this.setupFullMustacheParams(k, i, h), j = k.id.parts[0];
            if (this.options.knownHelpers[j]) {
                this.opcode("invokeKnownHelper", l.length, j)
            } else {
                if (this.knownHelpersOnly) {
                    throw new Error("You specified knownHelpersOnly, but used the unknown helper " + j)
                } else {
                    this.opcode("invokeHelper", l.length, j)
                }
            }
        },
        ID: function(m) {
            this.addDepth(m.depth);
            this.opcode("getContext", m.depth);
            var j = m.parts[0];
            if (!j) {
                this.opcode("pushContext")
            } else {
                this.opcode("lookupOnContext", m.parts[0])
            }
            for (var k = 1, h = m.parts.length; k < h; k++) {
                this.opcode("lookup", m.parts[k])
            }
        },
        DATA: function(h) {
            this.options.data = true;
            this.opcode("lookupData", h.id)
        },
        STRING: function(h) {
            this.opcode("pushString", h.string)
        },
        INTEGER: function(h) {
            this.opcode("pushLiteral", h.integer)
        },
        BOOLEAN: function(h) {
            this.opcode("pushLiteral", h.bool)
        },
        comment: function() {},
        opcode: function(h) {
            this.opcodes.push({
                opcode: h,
                args: [].slice.call(arguments, 1)
            })
        },
        declare: function(h, i) {
            this.opcodes.push({
                opcode: "DECLARE",
                name: h,
                value: i
            })
        },
        addDepth: function(h) {
            if (isNaN(h)) {
                throw new Error("EWOT")
            }
            if (h === 0) {
                return 
            }
            if (!this.depths[h]) {
                this.depths[h] = true;
                this.depths.list.push(h)
            }
        },
        classifyMustache: function(k) {
            var j = k.isHelper;
            var l = k.eligibleHelper;
            var i = this.options;
            if (l&&!j) {
                var h = k.id.parts[0];
                if (i.knownHelpers[h]) {
                    j = true
                } else {
                    if (i.knownHelpersOnly) {
                        l = false
                    }
                }
            }
            if (j) {
                return "helper"
            } else {
                if (l) {
                    return "ambiguous"
                } else {
                    return "simple"
                }
            }
        },
        pushParams: function(k) {
            var h = k.length, j;
            while (h--) {
                j = k[h];
                if (this.options.stringParams) {
                    if (j.depth) {
                        this.addDepth(j.depth)
                    }
                    this.opcode("getContext", j.depth || 0);
                    this.opcode("pushStringParam", j.string)
                } else {
                    this[j.type](j)
                }
            }
        },
        setupMustacheParams: function(h) {
            var i = h.params;
            this.pushParams(i);
            if (h.hash) {
                this.hash(h.hash)
            } else {
                this.opcode("pushLiteral", "{}")
            }
            return i
        },
        setupFullMustacheParams: function(j, i, h) {
            var k = j.params;
            this.pushParams(k);
            this.opcode("pushProgram", i);
            this.opcode("pushProgram", h);
            if (j.hash) {
                this.hash(j.hash)
            } else {
                this.opcode("pushLiteral", "{}")
            }
            return k
        }
    };
    var c = function(h) {
        this.value = h
    };
    f.prototype = {
        nameLookup: function(j, h, i) {
            if (/^[0-9]+$/.test(h)) {
                return j + "[" + h + "]"
            } else {
                if (f.isValidJavaScriptVariableName(h)) {
                    return j + "." + h
                } else {
                    return j + "['" + h + "']"
                }
            }
        },
        appendToBuffer: function(h) {
            if (this.environment.isSimple) {
                return "return " + h + ";"
            } else {
                return "buffer += " + h + ";"
            }
        },
        initializeBuffer: function() {
            return this.quotedString("")
        },
        namespace: "Handlebars",
        compile: function(h, i, k, j) {
            this.environment = h;
            this.options = i || {};
            Handlebars.log(Handlebars.logger.DEBUG, this.environment.disassemble() + "\n\n");
            this.name = this.environment.name;
            this.isChild=!!k;
            this.context = k || {
                programs: [],
                aliases: {}
            };
            this.preamble();
            this.stackSlot = 0;
            this.stackVars = [];
            this.registers = {
                list: []
            };
            this.compileStack = [];
            this.compileChildren(h, i);
            var m = h.opcodes, l;
            this.i = 0;
            for (b = m.length; this.i < b; this.i++) {
                l = m[this.i];
                if (l.opcode === "DECLARE") {
                    this[l.name] = l.value
                } else {
                    this[l.opcode].apply(this, l.args)
                }
            }
            return this.createFunctionContext(j)
        },
        nextOpcode: function() {
            var i = this.environment.opcodes, h = i[this.i + 1];
            return i[this.i + 1]
        },
        eat: function(h) {
            this.i = this.i + 1
        },
        preamble: function() {
            var h = [];
            if (!this.isChild) {
                var i = this.namespace;
                var j = "helpers = helpers || " + i + ".helpers;";
                if (this.environment.usePartial) {
                    j = j + " partials = partials || " + i + ".partials;"
                }
                if (this.options.data) {
                    j = j + " data = data || {};"
                }
                h.push(j)
            } else {
                h.push("")
            }
            if (!this.environment.isSimple) {
                h.push(", buffer = " + this.initializeBuffer())
            } else {
                h.push("")
            }
            this.lastContext = 0;
            this.source = h
        },
        createFunctionContext: function(o) {
            var p = this.stackVars.concat(this.registers.list);
            if (p.length > 0) {
                this.source[1] = this.source[1] + ", " + p.join(", ")
            }
            if (!this.isChild) {
                var j = [];
                for (var n in this.context.aliases) {
                    this.source[1] = this.source[1] + ", " + n + "=" + this.context.aliases[n]
                }
            }
            if (this.source[1]) {
                this.source[1] = "var " + this.source[1].substring(2) + ";"
            }
            if (!this.isChild) {
                this.source[1] += "\n" + this.context.programs.join("\n") + "\n"
            }
            if (!this.environment.isSimple) {
                this.source.push("return buffer;")
            }
            var q = this.isChild ? ["depth0", "data"]: ["Handlebars", "depth0", "helpers", "partials", "data"];
            for (var m = 0, h = this.environment.depths.list.length; m < h; m++) {
                q.push("depth" + this.environment.depths.list[m])
            }
            if (o) {
                q.push(this.source.join("\n  "));
                return Function.apply(this, q)
            } else {
                var k = "function " + (this.name || "") + "(" + q.join(",") + ") {\n  " + this.source.join("\n  ") + "}";
                Handlebars.log(Handlebars.logger.DEBUG, k + "\n\n");
                return k
            }
        },
        blockValue: function() {
            this.context.aliases.blockHelperMissing = "helpers.blockHelperMissing";
            var h = ["depth0"];
            this.setupParams(0, h);
            this.replaceStack(function(i) {
                h.splice(1, 0, i);
                return i + " = blockHelperMissing.call(" + h.join(", ") + ")"
            })
        },
        ambiguousBlockValue: function() {
            this.context.aliases.blockHelperMissing = "helpers.blockHelperMissing";
            var i = ["depth0"];
            this.setupParams(0, i);
            var h = this.topStack();
            i.splice(1, 0, h);
            this.source.push("if (!" + this.lastHelper + ") { " + h + " = blockHelperMissing.call(" + i.join(", ") + "); }")
        },
        appendContent: function(h) {
            this.source.push(this.appendToBuffer(this.quotedString(h)))
        },
        append: function() {
            var h = this.popStack();
            this.source.push("if(" + h + " || " + h + " === 0) { " + this.appendToBuffer(h) + " }");
            if (this.environment.isSimple) {
                this.source.push("else { " + this.appendToBuffer("''") + " }")
            }
        },
        appendEscaped: function() {
            var i = this.nextOpcode(), h = "";
            this.context.aliases.escapeExpression = "this.escapeExpression";
            if (i && i.opcode === "appendContent") {
                h = " + " + this.quotedString(i.args[0]);
                this.eat(i)
            }
            this.source.push(this.appendToBuffer("escapeExpression(" + this.popStack() + ")" + h))
        },
        getContext: function(h) {
            if (this.lastContext !== h) {
                this.lastContext = h
            }
        },
        lookupOnContext: function(h) {
            this.pushStack(this.nameLookup("depth" + this.lastContext, h, "context"))
        },
        pushContext: function() {
            this.pushStackLiteral("depth" + this.lastContext)
        },
        resolvePossibleLambda: function() {
            this.context.aliases.functionType = '"function"';
            this.replaceStack(function(h) {
                return "typeof " + h + " === functionType ? " + h + "() : " + h
            })
        },
        lookup: function(h) {
            this.replaceStack(function(i) {
                return i + " == null || " + i + " === false ? " + i + " : " + this.nameLookup(i, h, "context")
            })
        },
        lookupData: function(h) {
            this.pushStack(this.nameLookup("data", h, "data"))
        },
        pushStringParam: function(h) {
            this.pushStackLiteral("depth" + this.lastContext);
            this.pushString(h)
        },
        pushString: function(h) {
            this.pushStackLiteral(this.quotedString(h))
        },
        push: function(h) {
            this.pushStack(h)
        },
        pushLiteral: function(h) {
            this.pushStackLiteral(h)
        },
        pushProgram: function(h) {
            if (h != null) {
                this.pushStackLiteral(this.programExpression(h))
            } else {
                this.pushStackLiteral(null)
            }
        },
        invokeHelper: function(j, h) {
            this.context.aliases.helperMissing = "helpers.helperMissing";
            var i = this.lastHelper = this.setupHelper(j, h);
            this.register("foundHelper", i.name);
            this.pushStack("foundHelper ? foundHelper.call(" + i.callParams + ") : helperMissing.call(" + i.helperMissingParams + ")")
        },
        invokeKnownHelper: function(j, h) {
            var i = this.setupHelper(j, h);
            this.pushStack(i.name + ".call(" + i.callParams + ")")
        },
        invokeAmbiguous: function(h) {
            this.context.aliases.functionType = '"function"';
            this.pushStackLiteral("{}");
            var i = this.setupHelper(0, h);
            var j = this.lastHelper = this.nameLookup("helpers", h, "helper");
            this.register("foundHelper", j);
            var l = this.nameLookup("depth" + this.lastContext, h, "context");
            var k = this.nextStack();
            this.source.push("if (foundHelper) { " + k + " = foundHelper.call(" + i.callParams + "); }");
            this.source.push("else { " + k + " = " + l + "; " + k + " = typeof " + k + " === functionType ? " + k + "() : " + k + "; }")
        },
        invokePartial: function(h) {
            var i = [this.nameLookup("partials", h, "partial"), "'" + h + "'", this.popStack(), "helpers", "partials"];
            if (this.options.data) {
                i.push("data")
            }
            this.context.aliases.self = "this";
            this.pushStack("self.invokePartial(" + i.join(", ") + ");")
        },
        assignToHash: function(h) {
            var i = this.popStack();
            var j = this.topStack();
            this.source.push(j + "['" + h + "'] = " + i + ";")
        },
        compiler: f,
        compileChildren: function(h, m) {
            var o = h.children, q, p;
            for (var n = 0, j = o.length; n < j; n++) {
                q = o[n];
                p = new this.compiler();
                this.context.programs.push("");
                var k = this.context.programs.length;
                q.index = k;
                q.name = "program" + k;
                this.context.programs[k] = p.compile(q, m, this.context)
            }
        },
        programExpression: function(j) {
            this.context.aliases.self = "this";
            if (j == null) {
                return "self.noop"
            }
            var p = this.environment.children[j], o = p.depths.list, n;
            var m = [p.index, p.name, "data"];
            for (var k = 0, h = o.length; k < h; k++) {
                n = o[k];
                if (n === 1) {
                    m.push("depth0")
                } else {
                    m.push("depth" + (n - 1))
                }
            }
            if (o.length === 0) {
                return "self.program(" + m.join(", ") + ")"
            } else {
                m.shift();
                return "self.programWithDepth(" + m.join(", ") + ")"
            }
        },
        register: function(h, i) {
            this.useRegister(h);
            this.source.push(h + " = " + i + ";")
        },
        useRegister: function(h) {
            if (!this.registers[h]) {
                this.registers[h] = true;
                this.registers.list.push(h)
            }
        },
        pushStackLiteral: function(h) {
            this.compileStack.push(new c(h));
            return h
        },
        pushStack: function(h) {
            this.source.push(this.incrStack() + " = " + h + ";");
            this.compileStack.push("stack" + this.stackSlot);
            return "stack" + this.stackSlot
        },
        replaceStack: function(i) {
            var h = i.call(this, this.topStack());
            this.source.push(this.topStack() + " = " + h + ";");
            return "stack" + this.stackSlot
        },
        nextStack: function(h) {
            var i = this.incrStack();
            this.compileStack.push("stack" + this.stackSlot);
            return i
        },
        incrStack: function() {
            this.stackSlot++;
            if (this.stackSlot > this.stackVars.length) {
                this.stackVars.push("stack" + this.stackSlot)
            }
            return "stack" + this.stackSlot
        },
        popStack: function() {
            var h = this.compileStack.pop();
            if (h instanceof c) {
                return h.value
            } else {
                this.stackSlot--;
                return h
            }
        },
        topStack: function() {
            var h = this.compileStack[this.compileStack.length - 1];
            if (h instanceof c) {
                return h.value
            } else {
                return h
            }
        },
        quotedString: function(h) {
            return '"' + h.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r") + '"'
        },
        setupHelper: function(k, i) {
            var j = [];
            this.setupParams(k, j);
            var h = this.nameLookup("helpers", i, "helper");
            return {
                params: j,
                name: h,
                callParams: ["depth0"].concat(j).join(", "),
                helperMissingParams: ["depth0", this.quotedString(i)].concat(j).join(", ")
            }
        },
        setupParams: function(o, n) {
            var k = [], p = [], m, h, j;
            k.push("hash:" + this.popStack());
            h = this.popStack();
            j = this.popStack();
            if (j || h) {
                if (!j) {
                    this.context.aliases.self = "this";
                    j = "self.noop"
                }
                if (!h) {
                    this.context.aliases.self = "this";
                    h = "self.noop"
                }
                k.push("inverse:" + h);
                k.push("fn:" + j)
            }
            for (var l = 0; l < o; l++) {
                m = this.popStack();
                n.push(m);
                if (this.options.stringParams) {
                    p.push(this.popStack())
                }
            }
            if (this.options.stringParams) {
                k.push("contexts:[" + p.join(",") + "]")
            }
            if (this.options.data) {
                k.push("data:data")
            }
            n.push("{" + k.join(",") + "}");
            return n.join(", ")
        }
    };
    var a = ("break else new var case finally return void catch for switch while continue function this with default if throw delete in try do instanceof typeof abstract enum int short boolean export interface static byte extends long super char final native synchronized class float package throws const goto private transient debugger implements protected volatile double import public let yield").split(" ");
    var e = f.RESERVED_WORDS = {};
    for (var d = 0, b = a.length; d < b; d++) {
        e[a[d]] = true
    }
    f.isValidJavaScriptVariableName = function(h) {
        if (!f.RESERVED_WORDS[h] && /^[a-zA-Z_$][0-9a-zA-Z_$]+$/.test(h)) {
            return true
        }
        return false
    }
})(Handlebars.Compiler, Handlebars.JavaScriptCompiler);
Handlebars.precompile = function(d, c) {
    c = c || {};
    var b = Handlebars.parse(d);
    var a = new Handlebars.Compiler().compile(b, c);
    return new Handlebars.JavaScriptCompiler().compile(a, c)
};
Handlebars.compile = function(b, a) {
    a = a || {};
    var d;
    function c() {
        var g = Handlebars.parse(b);
        var f = new Handlebars.Compiler().compile(g, a);
        var e = new Handlebars.JavaScriptCompiler().compile(f, a, undefined, true);
        return Handlebars.template(e)
    }
    return function(f, e) {
        if (!d) {
            d = c()
        }
        return d.call(this, f, e)
    }
};
Handlebars.VM = {
    template: function(a) {
        var b = {
            escapeExpression: Handlebars.Utils.escapeExpression,
            invokePartial: Handlebars.VM.invokePartial,
            programs: [],
            program: function(d, e, f) {
                var c = this.programs[d];
                if (f) {
                    return Handlebars.VM.program(e, f)
                } else {
                    if (c) {
                        return c
                    } else {
                        c = this.programs[d] = Handlebars.VM.program(e);
                        return c
                    }
                }
            },
            programWithDepth: Handlebars.VM.programWithDepth,
            noop: Handlebars.VM.noop
        };
        return function(d, c) {
            c = c || {};
            return a.call(b, Handlebars, d, c.helpers, c.partials, c.data)
        }
    },
    programWithDepth: function(b, d, c) {
        var a = Array.prototype.slice.call(arguments, 2);
        return function(f, e) {
            e = e || {};
            return b.apply(this, [f, e.data || d].concat(a))
        }
    },
    program: function(a, b) {
        return function(d, c) {
            c = c || {};
            return a(d, c.data || b)
        }
    },
    noop: function() {
        return ""
    },
    invokePartial: function(a, c, e, f, d, g) {
        var b = {
            helpers: f,
            partials: d,
            data: g
        };
        if (a === undefined) {
            throw new Handlebars.Exception("The partial " + c + " could not be found")
        } else {
            if (a instanceof Function) {
                return a(e, b)
            } else {
                if (!Handlebars.compile) {
                    throw new Handlebars.Exception("The partial " + c + " could not be compiled when running in runtime-only mode")
                } else {
                    d[c] = Handlebars.compile(a, {
                        data: g !== undefined
                    });
                    return d[c](e, b)
                }
            }
        }
    }
};
Handlebars.template = Handlebars.VM.template; /*
 *      jQuery dotdotdot 1.6.16
 *
 *      Copyright (c) Fred Heusschen
 *      www.frebsite.nl
 *
 *      Plugin website:
 *      dotdotdot.frebsite.nl
 *
 *      Dual licensed under the MIT and GPL licenses.
 *      http://en.wikipedia.org/wiki/MIT_License
 *      http://en.wikipedia.org/wiki/GNU_General_Public_License
 */
!function(t, e) {
    function n(t, e, n) {
        var r = t.children(), o=!1;
        t.empty();
        for (var i = 0, d = r.length; d > i; i++) {
            var l = r.eq(i);
            if (t.append(l), n && t.append(n), a(t, e)) {
                l.remove(), o=!0;
                break
            }
            n && n.detach()
        }
        return o
    }
    function r(e, n, i, d, l) {
        var s=!1, c = "table, thead, tbody, tfoot, tr, col, colgroup, object, embed, param, ol, ul, dl, blockquote, select, optgroup, option, textarea, script, style", u = "script, .dotdotdot-keep";
        return e.contents().detach().each(function() {
            var f = this, h = t(f);
            if ("undefined" == typeof f || 3 == f.nodeType && 0 == t.trim(f.data).length)
                return !0;
            if (h.is(u))
                e.append(h);
            else {
                if (s)
                    return !0;
                e.append(h), l && e[e.is(c) ? "after": "append"](l), a(i, d) && (s = 3 == f.nodeType ? o(h, n, i, d, l) : r(h, n, i, d, l), s || (h.detach(), s=!0)), s || l && l.detach()
            }
        }), s
    }
    function o(e, n, r, o, d) {
        var c = e[0];
        if (!c)
            return !1;
        var f = s(c), h =- 1 !== f.indexOf(" ") ? " " : "　", p = "letter" == o.wrap ? "" : h, g = f.split(p), v =- 1, w =- 1, b = 0, y = g.length - 1;
        for (o.fallbackToLetter && 0 == b && 0 == y && (p = "", g = f.split(p), y = g.length - 1); y >= b && (0 != b || 0 != y);) {
            var m = Math.floor((b + y) / 2);
            if (m == w)
                break;
            w = m, l(c, g.slice(0, w + 1).join(p) + o.ellipsis), a(r, o) ? (y = w, o.fallbackToLetter && 0 == b && 0 == y && (p = "", g = g[0].split(p), v =- 1, w =- 1, b = 0, y = g.length - 1)) : (v = w, b = w)
        }
        if ( - 1 == v || 1 == g.length && 0 == g[0].length) {
            var x = e.parent();
            e.detach();
            var T = d && d.closest(x).length ? d.length: 0;
            x.contents().length > T ? c = u(x.contents().eq( - 1 - T), n) : (c = u(x, n, !0), T || x.detach()), c && (f = i(s(c), o), l(c, f), T && d && t(c).parent().append(d))
        } else 
            f = i(g.slice(0, v + 1).join(p), o), l(c, f);
        return !0
    }
    function a(t, e) {
        return t.innerHeight() > e.maxHeight
    }
    function i(e, n) {
        for (; t.inArray(e.slice( - 1), n.lastCharacter.remove)>-1;)
            e = e.slice(0, - 1);
        return t.inArray(e.slice( - 1), n.lastCharacter.noEllipsis) < 0 && (e += n.ellipsis), e
    }
    function d(t) {
        return {
            width: t.innerWidth(),
            height: t.innerHeight()
        }
    }
    function l(t, e) {
        t.innerText ? t.innerText = e : t.nodeValue ? t.nodeValue = e : t.textContent && (t.textContent = e)
    }
    function s(t) {
        return t.innerText ? t.innerText : t.nodeValue ? t.nodeValue : t.textContent ? t.textContent : ""
    }
    function c(t) {
        do 
            t = t.previousSibling;
        while (t && 1 !== t.nodeType && 3 !== t.nodeType);
        return t
    }
    function u(e, n, r) {
        var o, a = e && e[0];
        if (a) {
            if (!r) {
                if (3 === a.nodeType)
                    return a;
                if (t.trim(e.text()))
                    return u(e.contents().last(), n)
                }
            for (o = c(a); !o;) {
                if (e = e.parent(), e.is(n) ||!e.length)
                    return !1;
                o = c(e[0])
            }
            if (o)
                return u(t(o), n)
        }
        return !1
    }
    function f(e, n) {
        return e ? "string" == typeof e ? (e = t(e, n), e.length ? e : !1) : e.jquery ? e : !1 : !1
    }
    function h(t) {
        for (var e = t.innerHeight(), n = ["paddingTop", "paddingBottom"], r = 0, o = n.length; o > r; r++) {
            var a = parseInt(t.css(n[r]), 10);
            isNaN(a) && (a = 0), e -= a
        }
        return e
    }
    if (!t.fn.dotdotdot) {
        t.fn.dotdotdot = function(e) {
            if (0 == this.length)
                return t.fn.dotdotdot.debug('No element found for "' + this.selector + '".'), this;
            if (this.length > 1)
                return this.each(function() {
                    t(this).dotdotdot(e)
                });
            var o = this;
            o.data("dotdotdot") && o.trigger("destroy.dot"), o.data("dotdotdot-style", o.attr("style") || ""), o.css("word-wrap", "break-word"), "nowrap" === o.css("white-space") && o.css("white-space", "normal"), o.bind_events = function() {
                return o.bind("update.dot", function(e, d) {
                    e.preventDefault(), e.stopPropagation(), l.maxHeight = "number" == typeof l.height ? l.height : h(o), l.maxHeight += l.tolerance, "undefined" != typeof d && (("string" == typeof d || d instanceof HTMLElement) && (d = t("<div />").append(d).contents()), d instanceof t && (i = d)), g = o.wrapInner('<div class="dotdotdot" />').children(), g.contents().detach().end().append(i.clone(!0)).find("br").replaceWith("  <br />  ").end().css({
                        height: "auto",
                        width: "auto",
                        border: "none",
                        padding: 0,
                        margin: 0
                    });
                    var c=!1, u=!1;
                    return s.afterElement && (c = s.afterElement.clone(!0), c.show(), s.afterElement.detach()), a(g, l) && (u = "children" == l.wrap ? n(g, l, c) : r(g, o, g, l, c)), g.replaceWith(g.contents()), g = null, t.isFunction(l.callback) && l.callback.call(o[0], u, i), s.isTruncated = u, u
                }).bind("isTruncated.dot", function(t, e) {
                    return t.preventDefault(), t.stopPropagation(), "function" == typeof e && e.call(o[0], s.isTruncated), s.isTruncated
                }).bind("originalContent.dot", function(t, e) {
                    return t.preventDefault(), t.stopPropagation(), "function" == typeof e && e.call(o[0], i), i
                }).bind("destroy.dot", function(t) {
                    t.preventDefault(), t.stopPropagation(), o.unwatch().unbind_events().contents().detach().end().append(i).attr("style", o.data("dotdotdot-style") || "").data("dotdotdot", !1)
                }), o
            }, o.unbind_events = function() {
                return o.unbind(".dot"), o
            }, o.watch = function() {
                if (o.unwatch(), "window" == l.watch) {
                    var e = t(window), n = e.width(), r = e.height();
                    e.bind("resize.dot" + s.dotId, function() {
                        n == e.width() && r == e.height() && l.windowResizeFix || (n = e.width(), r = e.height(), u && clearInterval(u), u = setTimeout(function() {
                            o.trigger("update.dot")
                        }, 100))
                    })
                } else 
                    c = d(o), u = setInterval(function() {
                        if (o.is(":visible")) {
                            var t = d(o);
                            (c.width != t.width || c.height != t.height) && (o.trigger("update.dot"), c = t)
                        }
                    }, 500);
                return o
            }, o.unwatch = function() {
                return t(window).unbind("resize.dot" + s.dotId), u && clearInterval(u), o
            };
            var i = o.contents(), l = t.extend(!0, {}, t.fn.dotdotdot.defaults, e), s = {}, c = {}, u = null, g = null;
            return l.lastCharacter.remove instanceof Array || (l.lastCharacter.remove = t.fn.dotdotdot.defaultArrays.lastCharacter.remove), l.lastCharacter.noEllipsis instanceof Array || (l.lastCharacter.noEllipsis = t.fn.dotdotdot.defaultArrays.lastCharacter.noEllipsis), s.afterElement = f(l.after, o), s.isTruncated=!1, s.dotId = p++, o.data("dotdotdot", !0).bind_events().trigger("update.dot"), l.watch && o.watch(), o
        }, t.fn.dotdotdot.defaults = {
            ellipsis: "... ",
            wrap: "word",
            fallbackToLetter: !0,
            lastCharacter: {},
            tolerance: 0,
            callback: null,
            after: null,
            height: null,
            watch: !1,
            windowResizeFix: !0
        }, t.fn.dotdotdot.defaultArrays = {
            lastCharacter: {
                remove: [" ", "　", ",", ";", ".", "!", "?"],
                noEllipsis: []
            }
        }, t.fn.dotdotdot.debug = function() {};
        var p = 1, g = t.fn.html;
        t.fn.html = function(n) {
            return n != e&&!t.isFunction(n) && this.data("dotdotdot") ? this.trigger("update", [n]) : g.apply(this, arguments)
        };
        var v = t.fn.text;
        t.fn.text = function(n) {
            return n != e&&!t.isFunction(n) && this.data("dotdotdot") ? (n = t("<div />").text(n).html(), this.trigger("update", [n])) : v.apply(this, arguments)
        }
    }
}(jQuery);
(function(e) {
    var c = {};
    function b(i, h) {
        var g=!/\W/.test(i) ? c[i] = c[i] || b(document.getElementById(i).innerHTML) : new Function("obj", "var p=[],print=function(){p.push.apply(p,arguments);};with(obj){p.push('" + i.replace(/[\r\t\n]/g, " ").split("<%").join("\t").replace(/((^|%>)[^\t]*)'/g, "$1\r").replace(/\t=(.*?)%>/g, "',$1,'").split("\t").join("');").split("%>").join("p.push('").split("\r").join("\\'") + "');}return p.join('');");
        return h ? g(h) : g
    }
    function f(g, h) {
        function j() {
            e(this).remove()
        }
        function i() {
            e(this).slideUp(100, j)
        }
        e(g).animate({
            opacity: 0
        }, {
            duration: 250,
            queue: false,
            complete: i
        })
    }
    function a(g, i) {
        function h() {
            var k = "." + i.newClass + "." + i.messageClass, j;
            if (e.cookie && e.cookie(i.persistentCookie)) {
                j = e(k, g).not(i.persistentClass)
            } else {
                j = e(k, g)
            }
            if (j.length) {
                j.prepend(i.closeIcon);
                j.click(function(l) {
                    if (e.cookie && j.hasClass(i.persistentClass)) {
                        e.cookie(i.persistentCookie, 1, {
                            path: "/"
                        })
                    }
                    f(this, i)
                });
                j.removeClass(i.newClass).addClass(i.activeClass);
                j.each(function() {
                    var l = this;
                    if (!e(l).hasClass(i.stickyClass) ||!e(l).hasClass(i.persistentClass)) {
                        var m = e(l).attr("data-timer") || i.timer;
                        setTimeout(function() {
                            f(e(l), i)
                        }, m)
                    }
                    e(l).fadeIn(500)
                })
            }
            setTimeout(h, i.interval)
        }
        h()
    }
    function d(g) {
        return g.replace(/</g, "&lt;").replace(/>/g, "&gt;")
    }
    e.fn.notifier = function(g) {
        var h = e.extend({}, e.fn.notify.defaults, g);
        return e(this).each(function() {
            var i = this, j = e.metadata ? e.extend(h, e(this).metadata()): h;
            if (j.scrollcss) {
                e(window).scroll(function() {
                    e(i).css(j.scrollcss)
                })
            }
            e("." + j.messageClass, i).addClass(j.newClass);
            a(i, j)
        })
    };
    e.fn.notify = function(i, g) {
        var h;
        if (typeof i === "string") {
            h = e.extend({
                message: i
            }, e.fn.notify.defaults, g)
        } else {
            h = e.extend({}, e.fn.notify.defaults, i)
        }
        if (h.status === "success") {
            h.status = "confirm"
        }
        return e(this).each(function() {
            if (h.message) {
                var k = e.metadata ? e.extend(h, e(this).metadata()): h;
                if (k.sanitize) {
                    k.message = d(k.message);
                    k.title = d(k.title)
                }
                var j = b(k.tmpl, k);
                e(this).append(j)
            } else {
                if (window.console) {
                    window.console.warn("No message was set in notify's config: ", k)
                }
            }
        })
    };
    e.fn.notify.defaults = {
        status: "info",
        interval: 500,
        timer: 15000,
        sticky: false,
        title: "",
        sanitize: true,
        tmpl: '<div class="message <%=newClass%> <%=status%> <% if (sticky) { %><%=stickyClass %><% } %>" data-timer="<%=timer%>"><% if (title) { %><h6><%=title%></h6><% } %><div class="content"><%=message%></div></div>',
        scrollcss: {
            position: "fixed",
            top: "20px"
        },
        stickyClass: "notify-sticky",
        persistentClass: "notify-persistent",
        persistentCookie: "notify-persistent-closed",
        newClass: "notify-new",
        activeClass: "notify-active",
        inactiveClass: "notify-inactive",
        messageClass: "message",
        closeIcon: '<b title="Close" class="ico ico-close" data-icon="D"></b>'
    }
}(jQuery));
(function(d) {
    var a = false;
    function f(l) {
        l.preventDefault();
        if (d(":radio:checked", d(this).closest("form")).length === 0) {
            d("#rating-yes").attr("checked", "checked")
        }
        var k = this, g = d(k).closest("form"), i = g.serialize(), j = g.metadata(), h = d("textarea", g).val() ? "submit": "rate";
        if (d(k).attr("name")) {
            i += "&" + d(k).attr("name") + "=" + d(k).val()
        }
        if (d(k).attr("type") === "submit") {
            h = d(k).val()
        }
        d.ajax({
            type: "POST",
            url: g.attr("action"),
            data: i,
            dataType: "json",
            success: function(m, o, n) {
                d("#messages").notify(m.message, {
                    status: m.status
                });
                if (h === "delete") {
                    g.trigger("deleteSuccess")
                } else {
                    if (h === "submit") {
                        g.trigger("postSuccess")
                    }
                }
            },
            error: function(o, q, n) {
                var m = d("textarea", g).val() ? "review": "rating";
                var p = d.parseJSON(o.responseText);
                if (h === "delete") {
                    d("#messages").notify("Whoops\u2014something broke while deleting your " + m + ".", {
                        status: "error"
                    })
                } else {
                    if (p && p.error) {
                        d("#messages").notify(p.error, {
                            status: "error"
                        })
                    } else {
                        d("#messages").notify("Whoops\u2014something broke while posting your " + m + ".", {
                            status: "error"
                        })
                    }
                }
            }
        })
    }
    function e(j) {
        var g = d(this), h = d("button[value=submit]", g), i = ['<button value="delete" type="submit" name="delete" class="btn link">', '<b title="Trash" class="ico ico-trash" data-icon="#"></b> Delete', "</button>"].join("");
        if (h.text().toLowerCase() === "post") {
            h.text("Update").after(i)
        }
    }
    function c(h) {
        var g = d(this);
        d("textarea", g).val("");
        d("#rating-yes").removeAttr("checked");
        d("#rating-no").removeAttr("checked");
        d("button[value=delete]", g).remove();
        d("button[value=submit]", g).text("Post");
        d(".rate", g).removeClass("active")
    }
    function b(j) {
        j.preventDefault();
        var i = j.data.comments, g = j.data.form, h = d(this).closest(".rate");
        if (!h.hasClass("active")) {
            h.addClass("active");
            h.siblings(".rate").removeClass("active")
        }
        if (h.hasClass("up")) {
            d("#rating-yes").attr("checked", "checked");
            d("#rating-no").removeAttr("checked")
        } else {
            d("#rating-no").attr("checked", "checked");
            d("#rating-yes").removeAttr("checked")
        }
        d("textarea", i).focus()
    }
    d.extend({
        reviewform: {
            init: function(j, i) {
                var g = d(j), l = d("label[for=rating-comments]", g).parent(), k = {
                    comments: l,
                    form: g
                }, n = d('<strong class="rate up btn"><img src="' + SF.cdn + "/img/thumbs_up_large.png?" + SF.deploy_time + '" alt="Thumbs Up"></strong>'), h = d('<strong class="rate down btn"><img src="' + SF.cdn + "/img/thumbs_down_large.png?" + SF.deploy_time + '" alt="Thumbs Down"></strong>'), m = d.metadata ? d.extend(i, d(j).metadata()): i;
                m.existing = g.attr("data-existing") === "true";
                if (d(":radio:checked", g).length) {
                    if (parseInt(d(":radio:checked", g).val(), 10)) {
                        n.addClass("active")
                    } else {
                        h.addClass("active")
                    }
                }
                d("label[for=rating-yes]", g).hide();
                d("label[for=rating-no]", g).hide();
                d("label[for=rating-no]", g).after(n);
                d(".up", g).bind("click", k, b).click(f);
                n.after(h);
                d(".down", g).bind("click", k, b).click(f);
                d("button", g).click(f);
                g.bind({
                    deleteSuccess: c,
                    postSuccess: e
                })
            }
        }
    });
    d.fn.reviewform = function(g) {
        var h = d.extend({}, d.fn.reviewform.defaults, g);
        if (this[0]) {
            return d.reviewform.init(this[0], h)
        }
    };
    d.fn.reviewform.defaults = {
        existing: "false"
    }
}(jQuery));
(function() {
    String.prototype.times = function(b) {
        return Array.prototype.join.call({
            length: b + 1
        }, this)
    };
    function a(c, d) {
        $.fn.listedit.oldlist = c.cloneNode(true);
        var b = d.add_html.join("").replace(/placeholder/, d.placeholder).times(d.times);
        $(c).append(b);
        $("li[class!=listedit-add]", c).each(function() {
            var e = $(this).text();
            e = e.replace(/"/g, "&#34;");
            $(this).empty().html('<span class="ui-icon ui-icon-arrowthick-2-n-s"></span><input class="text" name="features" type="text" value="' + e + '"/>')
        });
        return $(c)
    }
    $.fn.listedit = function(b) {
        var c = $.extend({}, $.fn.listedit.defaults, b);
        return $(this).each(function() {
            var h = $.metadata ? $.extend(c, $(this).metadata()): c, e = this, g = a(e, h), d = g.children("li");
            $(e).trigger("listedit-begin");
            g.wrap('<form class="listedit-form" action="' + h.action + '" method="post"><div class="item"></div></form>');
            var f = $(".listedit-form");
            if (h.html) {
                f.append(h.html.join(""))
            }
            f.append('<input type="submit" value="Save Changes"/>');
            f.append('<a class="btn link cancel" href="#cancel">Cancel</a>');
            $(".cancel", f).click(function(j) {
                j.preventDefault();
                var i = f.parent();
                f.remove();
                i.append($.fn.listedit.oldlist);
                $($.fn.listedit.oldlist).trigger("listedit-end")
            });
            $(".listedit-form").on("keydown", "li:last", function(j) {
                var i = h.add_html.join("").replace(/placeholder/, h.placeholder);
                $(i).appendTo($(this).parent()).children("input").clearingInput()
            });
            f.submit(function(i) {
                $(e).trigger("listedit-end")
            });
            g.sortable({
                placeholder: "ui-state-highlight"
            });
            $(".listedit-add input").clearingInput()
        })
    };
    $.fn.listedit.defaults = {
        action: "#listedit-update",
        placeholder: "Add a new list item",
        times: 3,
        add_html: ['<li class="listedit-add">', '<span class="ui-icon ui-icon-arrowthick-2-n-s"></span>', '<input name="features" type="text" value="placeholder"/>', "</li>"]
    }
}());
jQuery.fn.seltext = function(e) {
    e = e || window;
    var d = e.document, b = this[0], c, a;
    if (e.getSelection && d.createRange) {
        c = e.getSelection();
        a = d.createRange();
        a.selectNodeContents(b);
        c.removeAllRanges();
        c.addRange(a)
    } else {
        if (d.body.createTextRange) {
            a = d.body.createTextRange();
            a.moveToElementText(b);
            a.select()
        }
    }
    return this
};
(function(a) {
    Handlebars.registerHelper("stage_message", function(b, c) {
        if (b.staged&&!b.explicitly_staged) {
            return "Staging inherited from parent folder"
        } else {
            return "Stage this folder for " + c + " days"
        }
    });
    Handlebars.registerHelper("stage_onclick", function(b) {
        if (b.staged&&!b.explicitly_staged) {
            return 'onclick="return false;"'
        } else {
            return ""
        }
    });
    Handlebars.registerHelper("checked", function(b) {
        return b ? 'checked="checked"' : ""
    });
    Handlebars.registerHelper("file_title_url", function(b) {
        return b.type === "f" ? b.download_url : b.url
    });
    Handlebars.registerHelper("stage_date", function(c) {
        if (c.staged) {
            var b = new Date(c.stage * 1000).toUTCString();
            return "This file is staged until " + b
        } else {
            return ""
        }
    });
    Handlebars.registerHelper("download_display", function(b) {
        if (b === null) {
            return "Not counted"
        } else {
            return addCommas(b)
        }
    });
    Handlebars.registerHelper("should_exclude_reports", function(b) {
        return b === true ? "checked" : ""
    });
    a.fn.drawer = function(b) {
        var c = a.extend({}, a.fn.drawer.defaults, b);
        return a(this).each(function() {
            var g = a.metadata ? a.extend(c, a(this).metadata()): c, d = this, f = a(g.selector), e = (g.anchor) ? a(d).closest(g.anchor): a(d);
            a(d).bind(g.evt, g.data, function(i) {
                if (g.filter && g.filter.search(new RegExp(i.target.tagName, "i")) >= 0) {
                    return true
                }
                a.fn.drawer.hide.call(f, e);
                a.fn.drawer.show.call(f, e);
                var h = net.sf.files[e.attr("title")];
                if (a(".delete-button", e).length) {
                    if (h.type === "f") {
                        a.ajax({
                            url: h.url + "metadata",
                            global: false,
                            dataType: "json",
                            success: function(j) {
                                a("#mirror_count").text("mirror".pluralize(j.mirrors));
                                a("a[href=#select_all]").toggle(function(k) {
                                    k.preventDefault();
                                    a(".default input:checkbox").each(function() {
                                        this.checked = true
                                    });
                                    a(this).text("Deselect all");
                                    a.fn.drawer.data_changed = true
                                }, function(k) {
                                    k.preventDefault();
                                    a(".default input:checkbox").each(function() {
                                        this.checked = false
                                    });
                                    a(this).text("Select all");
                                    a.fn.drawer.data_changed = true
                                })
                            }
                        })
                    }
                }
                a(":input:not(:hidden,:submit,:password,:button)", a(".fs-drawer")[0]).on("change paste", function() {
                    a.fn.drawer.data_changed = true
                });
                a(document).keyup(function(j) {
                    if (j.keyCode === 27) {
                        a(document).trigger("click")
                    }
                });
                setTimeout(function() {
                    a(document).click(function(j) {
                        if (!a(j.target).parents(g.selector).length) {
                            a(document).unbind("click").unbind("keypress");
                            a.fn.drawer.hide.call(f, d)
                        }
                    })
                }, 0);
                return false
            });
            return true
        })
    };
    a.fn.drawer.defaults = {
        evt: "click",
        selector: "#drawer",
        anchor: false
    };
    a.fn.drawer.data_changed = false;
    a.fn.drawer.show = function(k) {
        var u = a(k).offset().top, g = a(k).offset().left - 1, p = a(k).outerWidth(), f = net.sf.files[a(k).attr("title")];
        var h = [{
            value: "windows",
            label: "Windows"
        }, {
            value: "mac",
            label: "Mac OS X"
        }, {
            value: "linux",
            label: "Linux"
        }, {
            value: "bsd",
            label: "BSD"
        }, {
            value: "solaris",
            label: "Solaris"
        }, {
            value: "others",
            label: "Others"
        }
        ];
        f.exclude_reports = f.exclude_reports === "True" ? true : false;
        f.d_type = f.type === "d" ? true : false;
        f.f_type = f.type === "f" ? true : false;
        f.not_downloadable = f.downloadable ? false : true;
        f.show_platforms = false;
        if (f.download_label) {
            f.download_label = f.download_label.replace('"', "&#34;")
        }
        var m = [];
        for (var j = 0, n = h.length; j < n; j++) {
            var o = h[j];
            var e = {
                value: o.value,
                title: o.label,
                authorized: f.authorized,
                skip: true,
                _checked: ""
            };
            if (f["default"].indexOf(o.value)>-1) {
                f.show_platforms = true;
                e.skip = false;
                e._checked = "checked"
            }
            m.push(e)
        }
        f.platforms = m;
        var b = a("#file-drawer-template").html();
        var r = Handlebars.compile(b);
        Handlebars.registerPartial("platform", a("#platform-partial").html());
        a(this).html(r(f));
        a.fn.drawer.data_changed = false;
        a(".files-date", this).html(a("td[headers=files_date_h]", k).html());
        a(".files-size", this).html(a("td[headers=files_size_h]", k).html());
        a(".files-downloads", this).html(a("td[headers=files_downloads_h]", k).html());
        a(".files-status", this).html(a("td[headers=files_status_h]", k).html());
        var s = a("input[name=name]", this);
        if (s && s.length) {
            var c = s[0].value;
            s.keyup(function() {
                var d = a.fn.files.valid_filename(this.value);
                if (d !== true || (this.value !== c && f.downloads > 0)) {
                    if (d !== true) {
                        a("#name_message").removeClass("hide").text(d)
                    } else {
                        if (this.value !== c && f.type === "d") {
                            a("#name_message").removeClass("hide").text("If you rename this folder, links to its files will break, and the download counts and reports will be reset.")
                        } else {
                            if (this.value !== c) {
                                a("#name_message").removeClass("hide").text("If you rename this file, links to it will break, and the download count and reports will be reset.")
                            } else {
                                a("#name_message").addClass("hide").text("")
                            }
                        }
                    }
                } else {
                    a("#name_message").addClass("hide").text("")
                }
            })
        }
        a(".fg-button", this).bind("mouseover mouseout mousedown mouseup", function(d) {
            if (d.type === "mouseover") {
                a(this).addClass("ui-state-hover")
            } else {
                if (d.type === "mouseout") {
                    a(this).removeClass("ui-state-hover")
                } else {
                    if (d.type === "mousedown") {
                        a(this).addClass("ui-state-active")
                    } else {
                        if (d.type === "mouseup") {
                            a(this).removeClass("ui-state-active")
                        }
                    }
                }
            }
        });
        a(".selectable", this).bind("click", function(d) {
            a(this).seltext()
        }).wrapInner("<span></span>").hover(function() {
            var i = a(this).contents().width();
            var d = a(this).width();
            var l = a(this).offsetParent().width() - a(this).position().left - 15;
            if (i > d) {
                a(this).stop().animate({
                    width: Math.min(i, l)
                }, 250)
            }
        }, function() {
            a(this).stop().animate({
                width: "97%"
            }, 250)
        });
        a(".ico-info, #cancel", this).bind("click", function(d) {
            d.preventDefault();
            a.fn.drawer.hide.call(a(".fs-drawer"), a(k))
        });
        a(".delete-button", this).click(function(d) {
            a.fn.drawer.hide.call(a(".fs-drawer"), k);
            a.fn.files.del.call(this, k)
        });
        a("form#file_properties_content").unbind("submit").bind("submit", function(d) {
            d.preventDefault();
            if (a.fn.drawer.data_changed) {
                if (a("#download_label").val() === a("#download_label").attr("placeholder")) {
                    a("#download_label").val("")
                }
                a.fn.files.saveHandler(a(this).serialize(), a(k).get(0))
            }
            a.fn.drawer.hide.call(a(".fs-drawer"), k)
        });
        a("input[name=name]", this).focus();
        if ((u + a(this).height()) > (a(window).scrollTop() + a(window).height())) {
            var q = (u + a(this).height()) - (a(window).scrollTop() + a(window).height());
            q += 12;
            a(window).scrollTo("+=" + q + "px")
        }
        a(this).css({
            top: u,
            left: g,
            width: p
        }).show();
        a(k).trigger("drawerShow")
    };
    a.fn.drawer.hide = function(b) {
        a(this).hide();
        a(b).trigger("drawerHide")
    }
}(jQuery));
(function() {
    $.fn.files = function(a) {
        var c = $.extend({}, $.fn.files.defaults, a);
        var b = $($(this)[0]);
        b.addClass("fs-filemanager");
        if (c.admin) {
            $.fn.files.initToolbar.call(this);
            $.fn.files.initCreateDialog.call(this, b);
            $.fn.files.initAdminActions.call(this, b)
        }
        $.fn.files.init.call(this, b);
        return $(this)
    };
    $.fn.files.initToolbar = function() {
        var a = $(".files-toolbar").fadeOut(0, function() {
            $(this).css("visibility", "visible").fadeIn()
        });
        $(".folder-create").click(function(b) {
            b.preventDefault();
            $("#folder-create-dialog").dialog("open");
            $("#folder-create-dialog input:text").focus().keyup(function() {
                var c = $.fn.files.valid_filename(this.value);
                if (c === true) {
                    $("#folder-create-dialog .invalid").text("");
                    $("#folder-create-dialog input[type=submit]").removeClass("ui-state-disabled").addClass("ui-state-default").removeAttr("disabled")
                } else {
                    $("#folder-create-dialog .invalid").text(c);
                    $("#folder-create-dialog input[type=submit]").addClass("ui-state-disabled").removeClass("ui-state-default").attr("disabled", "disabled")
                }
            })
        })
    };
    $.fn.files.initCreateDialog = function(a) {
        $("#folder-create-dialog").dialog({
            autoOpen: false,
            modal: true,
            title: "Create Folder",
            width: 450
        });
        $(".ui-dialog").css({
            zIndex: 1000
        });
        $("#folder-create-dialog form").submit(function(b) {
            b.preventDefault();
            if ($("input:text", this).val()) {
                $("label.error", this).removeClass("error");
                $("span.error", this).remove();
                $.fn.files.create.call(a, this)
            } else {
                $("input:text", this).focus().select();
                $("label", this).addClass("error");
                $(this).prepend('<span class="error">The folder name is invalid, please try again.</span>')
            }
        })
    };
    $.fn.files.initAdminActions = function(c) {
        $("#delete-confirm-dialog").dialog({
            autoOpen: false,
            modal: true,
            title: "Confirm Delete",
            width: 450
        });
        var b = $('<span title="Delete file" data-icon="#" class="ico ico-trash delete-button"></span>');
        $("tbody .status", c).append(b);
        var a = $('<span class="info ico-info" title="View details">i</span>');
        $("tbody .folder .status", c).prepend(a);
        $("#files_list .delete-button", c).on("click", function(g) {
            var f = $(g.target);
            var d = this;
            $("#delete-confirm-error").hide();
            var h = f.closest("tr").get(0);
            $("#delete-confirm-filename").html($(h).attr("title"));
            $("#delete-confirm-dialog").dialog("open");
            $("#delete-confirm-form").bind("submit", function(i) {
                i.preventDefault();
                if ($("input:text", this).val() === "DELETE") {
                    $.fn.files.del.call(d, h);
                    $("#delete-confirm-dialog").dialog("close");
                    return true
                } else {
                    $("#delete-confirm-error").html("You must confirm with the word DELETE");
                    $("#delete-confirm-error").fadeIn("slow");
                    return false
                }
            });
            $("#delete-confirm-dialog").bind("dialogclose", function() {
                $("#delete-confirm-form").unbind("submit")
            })
        })
    };
    $.fn.files.drawerConfig = {
        selector: "#files-drawer",
        anchor: "tr"
    };
    $.fn.files.init = function(b) {
        var a = $('<span class="info ico-info" title="View details">i</span>');
        $("tbody .file .status", b).prepend(a);
        $("tbody .symlink .status", b).prepend('<b class="ico ico-linkout" data-icon="R" title="Symlink"></b>');
        $("tbody .folder .status", b).not(".status:has(.ico-info)").prepend('<span class="ico spacer">&nbsp;</span>');
        $("tbody .ico-info", b).drawer($.fn.files.drawerConfig);
        $(b).bind("sortStart", function() {
            $.fn.drawer.hide.call($("#files-drawer"), "tr")
        });
        $("#files_list tbody tr", b).on("drawerShow", function(c) {
            $(this).addClass("ui-state-active")
        });
        $("#files_list tbody tr", b).on("drawerHide", function(c) {
            $("#files_list tbody tr").removeClass("ui-state-active")
        })
    };
    $.fn.files.saveHandler = function(c, b) {
        var a = net.sf.files[b.title];
        $.ajax({
            url: $.fn.files.absurl.call(this, [a.name]),
            type: "PUT",
            data: c,
            beforeSend: function(d) {
                d.setRequestHeader("Accept", "application/json; charset=utf-8")
            },
            success: function(i, e) {
                var l = $(b).find(".name"), h = $(b).find(".staged"), j = i.result, f = a.name, g = j.name, d = encodeURIComponent(f), k = encodeURIComponent(g);
                if (g.length > 49) {
                    l.html(g.substr(0, 49) + "...")
                } else {
                    l.html(g)
                }
                $("a[href].name", b).each(function() {
                    var n = $(this).attr("href").rsplit(d, 1).join(k), m = $(this).attr("title").replace(f, g);
                    $(this).attr("href", n);
                    $(this).attr("title", m)
                });
                $(b).attr("title", $(b).attr("title").replace(f, g));
                if (f !== g) {
                    delete net.sf.files[f]
                }
                a.download_url = a.download_url.rsplit(d, 1).join(k);
                a.url = a.url.rsplit(d, 1).join(k);
                a.full_path = a.full_path.rsplit(f, 1).join(g);
                a.name = g;
                a.sha1 = j.sha1;
                a.md5 = j.md5;
                if (j.x_sf) {
                    a["default"] = j.x_sf["default"] || "";
                    a.download_label = j.x_sf.download_label || "";
                    a.exclude_reports = j.x_sf.exclude_reports || false
                }
                a.downloadable = j.downloadable;
                a.staged = j.staged;
                a.stage = j.stage;
                a.explicitly_staged = j.explicitly_staged;
                net.sf.files[a.name] = a;
                if (a.staged&&!h.length) {
                    l.after('<span class="staged">(staged)</span>')
                } else {
                    if (!a.staged && h.length) {
                        h.remove()
                    }
                }
                if (a.exclude_reports === false) {
                    $(".exclude-reports span", b).removeClass("ico ico-noentry")
                } else {
                    $(".fs-stats", b).replaceWith('<a class="fs-stats ui-corner-all no-recent-downloads exclude-reports file" title="Statistics are not counted for this file"><span data-icon="d" class="ico ico-noentry"></span></a>')
                }
                $("#messages").notify({
                    status: "success",
                    message: "File properties have been updated."
                })
            },
            error: function(f, h) {
                var d = {};
                try {
                    d = $.parseJSON(f.responseText)
                } catch (g) {
                    if (window.console) {
                        console.error("Invalid JSON: " + f.responseText)
                    }
                }
                if (!("error" in d) || typeof(d.error) !== "string") {
                    d.error = "An error occurred while updating file properties"
                }
                $("#messages").notify({
                    status: "error",
                    message: d.error
                })
            }
        })
    };
    $.fn.files.updateMetadata = function(a, c) {
        var b = net.sf.files[a.attr("title")];
        var d = function() {
            if (a.find(".mirror-status").length) {
                if ($(".mirror-status .mirrors", a).length) {
                    $(".mirror-status .mirrors", a).html(b.mirrors);
                    if (b.mirrors > 1) {
                        $(".mirror-status .mirrorsw", a).html("mirrors")
                    }
                } else {
                    if (b.mirrors > 0) {
                        var e = ['<a class="', "mirror-status " + (b.type === "f") ? "folder": "file", '" href="#mirror_status" title="Check on the file status">', 'On <span class="mirrors">', b.mirrors, "</span> ", '<span class="mirrorsw">', (b.mirrors > 1) ? "mirrors": "mirror", "</span>", "&#8230;", "</a>"].join("");
                        $(".mirror-status", a).html(e)
                    }
                }
                if (b.downloadable) {
                    $(".downloadable", a).remove()
                }
            }
            net.sf.files[a.attr("title")] = b;
            $(".default input:checkbox").each(function() {
                if (b["default"].indexOf(this.value) >= 0) {
                    this.checked = true
                } else {
                    this.checked = false
                }
            });
            if (b.downloadable === true && $(".info:visible", "#file_properties_content").length) {
                $(".info:visible", "#file_properties_content").slideUp()
            }
        };
        $.getJSON(b.url + "metadata", function(e) {
            b.mirrors = e.mirrors;
            b.downloadable = e.downloadable;
            if (!c) {
                b["default"] = e.x_sf["default"] || ""
            }
            d()
        })
    };
    $.fn.files.create = function(b) {
        var a = $(b);
        $.ajax({
            url: $.fn.files.absurl.call(this, []),
            type: "POST",
            data: a.serialize(),
            beforeSend: function(c) {
                c.setRequestHeader("Accept", "application/json; charset=utf-8")
            },
            success: function(i, j) {
                var h = i.result, f = h.name || "", d = h.staged ? '<span class="staged">(staged)</span>': "";
                var e = new Date().format("isoDate"), g = ['<tr class="folder" title="' + f + '">', '<th headers="files_name_h" scope="row">', '<span data-icon="o" class="ico ico-folder"></span>', '<a class="name" title="Click to enter ' + f + '" href="', encodeURIComponent(f), '">' + f + "</a>", d, "</th>", '<td class="opt" headers="files_date_h">' + e + "</td>", '<td class="opt" headers="files_size_h"></td>', '<td class="opt" headers="files_downloads_h">', '<a class="fs-stats ui-corner-all no-recent-downloads folder" href="TEST%202/stats/timeline"></a>', "</td>", '<td class="status" headers="files_status_h">', '<span class="info ico-info" title="View details">i</span>', '<span data-icon="#" class="ico ico-trash delete-button" title="Delete file"></span>', "</td>", "</tr>"], c = $(g.join(""));
                a.trigger("filesCreate");
                $("#folder-create-dialog").dialog("close");
                $("#folder-create-dialog input:text").val("");
                $("#files_list tbody").prepend(c);
                c.effect("highlight");
                if (!net.sf.files) {
                    net.sf.files = {}
                }
                net.sf.files[f] = {
                    name: f,
                    download_url: "",
                    url: "",
                    full_path: "",
                    type: "d",
                    staged: h.staged,
                    stage: h.stage,
                    explicitly_staged: h.explicitly_staged
                };
                $(".ico-info", c).drawer($.fn.files.drawerConfig);
                $("#messages").notify({
                    status: "success",
                    message: 'The folder "' + f + '" has been created.'
                })
            },
            error: function(d, g) {
                $("#folder-create-dialog").dialog("close");
                var c = {};
                try {
                    c = $.parseJSON(d.responseText)
                } catch (f) {
                    if (window.console) {
                        console.error("Invalid JSON: " + d.responseText)
                    }
                }
                if (!("error" in c) || typeof(c.error) !== "string") {
                    c.error = "An error occurred while creating the folder."
                }
                $("#messages").notify({
                    status: "error",
                    message: c.error
                })
            }
        })
    };
    $.fn.files.del = function(d, f) {
        var b = $(d), e = b.attr("title"), h = 0, a = 0, g = false;
        function c() {
            f = typeof(f) !== "undefined" ? f : true;
            $.ajax({
                url: $.fn.files.absurl.call(this, [e]),
                type: "DELETE",
                beforeSend: function(i) {
                    i.setRequestHeader("Accept", "application/json; charset=utf-8")
                },
                success: function(j, k) {
                    b.closest("table").trigger("update");
                    b.trigger("filesDelete").remove();
                    a += 1;
                    if (f) {
                        var i = j.result.folder ? "folder": "file";
                        $("#messages").notify({
                            status: "success",
                            message: 'The "' + j.result.name + '" ' + i + " has been deleted."
                        })
                    }
                },
                error: function(j, l) {
                    h += 1;
                    var i = {};
                    try {
                        i = $.parseJSON(j.responseText)
                    } catch (k) {
                        if (window.console) {
                            console.error("Invalid JSON: " + j.responseText)
                        }
                    }
                    if (!("error" in i) || typeof(i.error) !== "string") {
                        i.error = 'An error occurred while attempting to delete "' + e + '".'
                    }
                    $("#messages").notify({
                        status: "error",
                        message: i.error
                    })
                }
            })
        }
        if (b.hasClass("warn")) {
            g = true
        }
        if (g) {
            if (confirm('Links to "' + e + '" will no longer work. Are you sure you want to delete it?')) {
                c()
            }
        } else {
            c()
        }
    };
    $.fn.files.absurl = function(c) {
        var a = window.location.pathname;
        for (var b = 0; b < c.length; b += 1) {
            c[b] = encodeURIComponent(c[b])
        }
        return (a.lastIndexOf("/") + 1 === a.length) ? a + c.join("/") : a + "/" + c.join("/")
    };
    $.fn.files.valid_filename = function(b) {
        var c = new RegExp("^[a-zA-Z0-9_ +.,=#~@!()\\[\\]-]+$");
        var a;
        if (b.length === 0) {
            a = "The name is too short."
        } else {
            if (b.slice(0, 1) === ".") {
                a = 'The name cannot start with ".".'
            } else {
                if (b.slice(0, 1) === " ") {
                    a = "The name cannot start with a space."
                } else {
                    if (b.slice( - 1) === " ") {
                        a = "The name cannot end with a space."
                    } else {
                        if (!c.test(b)) {
                            a = 'May only contain a-zA-Z0-9_+-.,=#~@!()[] (including " " - space).'
                        } else {
                            a = true
                        }
                    }
                }
            }
        }
        return a
    };
    $.fn.files.defaults = {
        admin: false
    }
}());
if (typeof(jQuery) !== "undefined") {
    var FileUploadRegistry = {
        wrappers: [],
        forms: {}
    };
    (function(a) {
        a.fn.fileUpload = function(c) {
            var f = document.location.href.split("#")[0].split("?")[0];
            var e = a.extend({
                replace_existing_form: false,
                add_submit: true,
                submit_label: "Send files",
                hidden_submit_name: "submit_button",
                max_size: 0,
                max_size_error_label: "Size limit exceeded",
                field_name: "file",
                submit_empty_forms: true,
                use_iframes: true,
                stat_url: f + "/gp.fileupload.stat/",
                stat_delay: 1500,
                stat_timeout: 7000,
                success: function() {},
                error: function() {},
                action: f
            }, c);
            var d = {
                info: function(h) {
                    if (e.debug) {
                        if (window.console) {
                            window.console.info(h)
                        }
                    }
                },
                warn: function(h) {
                    if (e.debug) {
                        if (window.console) {
                            window.console.warn(h)
                        }
                    }
                }
            };
            if (e.hidden_submit_name === "submit") {
                d.warn("Don't set the hidden_submit_name options to 'submit'")
            }
            var g = function(h) {
                this.element = h;
                this.settings = e;
                this.forms = [];
                this.forms_wrapper = a('<div class="forms-wrap"></div>').appendTo(h);
                this.submitted = []
            };
            a.extend(g.prototype, {
                initialize: function() {
                    var h = this;
                    var j = true;
                    if (h.settings.replace_existing_form === true) {
                        this.forms_wrapper.prev().remove()
                    }
                    a("form", this.element).each(function() {
                        if (!a(this).hasClass("fuForm")) {
                            h.forms.push(new b(this, h))
                        } else {
                            j = false
                        }
                    });
                    if (j && h.forms.length === 0 || h.settings.replace_existing_form) {
                        h.showNext();
                        var i = ['<div class="submit-wrap">', '<input id="upload_submit" class="fuButton" type="submit" value="Upload" disabled="disabled" name="', e.hidden_submit_name, '"/><br>', ("multiple" in document.createElement("input")) ? '<span class="new">New:</span> thanks to the magic of HTML5, you can upload multiple files at the same time.<br>': "", 'You may upload files up to 1GB in size using the web form. For larger files, <a href="https://sourceforge.net/p/forge/documentation/Release%20Files%20for%20Download#scp">use FTP, SCP, or rsync</a>.<br>', 'Only <a href="http://slashdotmedia.com/terms-of-use/">Open Source-licensed</a> materials may be uploaded to SourceForge.<br>', 'When uploading files, you agree to abide by our <a href="http://slashdotmedia.com/terms-of-use/">Terms of Use</a> and <a href="http://slashdotmedia.com/privacy-statement/">Privacy Policy</a>.<br><br>', "You also agree to not upload or share any content that you do not have the legal right to share.", "</div>"].join("");
                        a(i).appendTo(h.element).val(h.settings.submit_label).find("input").click(function() {
                            h.submit()
                        })
                    }
                },
                showNext: function() {
                    var h = this;
                    var i = a('<form class="fuForm fuLastForm" method="POST" enctype="multipart/form-data" action="' + this.settings.action + '"> <input type="file" class="title" value="" name="' + this.settings.field_name + '" multiple /> <input type="hidden" value="' + this.settings.submit_label + '" name="' + this.settings.hidden_submit_name + '" /><hr class="div"></form>').appendTo(this.forms_wrapper);
                    a("input:file", i).change(function() {
                        if (i.hasClass("fuLastForm")) {
                            i.removeClass("fuLastForm");
                            i.find("hr.div").show().prev().show();
                            h.showNext()
                        }
                    });
                    a("body").on("click", ".fuShowNext", function() {
                        a(this).prop("disabled", true);
                        a(".fuButton").prop("disabled", true);
                        h.showNext();
                        a(".submit-wrap", h.element).show();
                        return false
                    });
                    this.forms.push(new b(i, this))
                },
                submit: function() {
                    var h = this;
                    a(".submit-wrap", h.element).hide();
                    a(".fuShowNext").prop("disabled", false);
                    a(h.forms).each(function(n, s) {
                        var m = 0;
                        var l = s.form;
                        if (a(l).hasClass("fuLastForm")) {
                            d.info("Don't submit last form " + s.id);
                            s.submit = false;
                            a(l).remove();
                            return 
                        }
                        a(l).css("display", "none");
                        var o = [];
                        a("input:file", l).each(function(t, w) {
                            var x = (typeof(w.files) !== "undefined");
                            var u = a(this).val();
                            if (x) {
                                for (var v = 0; v < w.files.length; v += 1) {
                                    if (typeof(w.files[v].name) !== "undefined") {
                                        o.push(w.files[v].name);
                                        m += w.files[v].size
                                    } else {
                                        o.push(w.files[v].fileName);
                                        m += w.files[v].fileSize
                                    }
                                }
                            } else {
                                if (u) {
                                    if (u.match("/")) {
                                        u = u.split("/").pop()
                                    }
                                    if (u.match("\\\\")) {
                                        u = u.split("\\").pop()
                                    }
                                    o.push(u)
                                }
                            }
                        });
                        var r = "";
                        if (o.length > 3) {
                            r = ", etc. (" + o.length + " total)";
                            o = o.slice(0, 3)
                        }
                        o = o.join(", ") + r;
                        if (o) {
                            a(".btn-done").attr("disabled", true);
                            var j = '<div class="fuWrapper"><div class="fuFilename">' + o + '</div><div class="ui-progressbar ui-widget ui-widget-content ui-corner-all" ><div class="fuProgress ui-progressbar-value ui-widget-header ui-corner-all">&nbsp;</div></div></div> <hr class="div">';
                            var k = window.location.toString().replace("upload/", "files/");
                            var q = '<div class="fuShowNext-container"><button class="fuShowNext">Select more files</button></div><form class="btn-done-container" action="' + k + '"><button class="btn-done" disabled="disabled">Done</button></form>';
                            j = a(j).prependTo(s.wrapper.element);
                            if (!a(".fuShowNext").length) {
                                q = a(q).appendTo(a("div#upload_forms"))
                            }
                            s.progress = a(".fuProgress", j);
                            a("body").addClass("busyUploading");
                            a("#upload_forms").find("hr.div:last").hide();
                            if (h.settings.use_iframes) {
                                var p = '<iframe style="display:none" src="about:blank" name="iframe_' + s.id + '"></iframe>';
                                a(p).appendTo(s.wrapper.element);
                                a(l).attr("target", "iframe_" + s.id);
                                d.info("Submiting form " + s.id + " to iframe")
                            }
                            if (h.settings.max_size && m > h.settings.max_size) {
                                a("#messages").notify({
                                    status: "error",
                                    message: "File size limit exceeded: " + o + ", not uploading"
                                });
                                s.progress.css("width", "100%").addClass("fuProgressFailure").html(h.settings.max_size_error_label);
                                s.submit = false
                            }
                        } else {
                            if (!h.settings.submit_empty_forms) {
                                s.submit = false
                            }
                        }
                    });
                    h.submitNext()
                },
                submitNext: function() {
                    if (this.forms.length > 0) {
                        var h = this.forms.shift();
                        if (h.submit) {
                            h.form.submit();
                            h.setTimeout();
                            this.submitted.push(h.id);
                            d.info("Form " + h.id + " submited")
                        } else {
                            d.info("Skipping form " + h.id);
                            this.submitNext()
                        }
                    } else {
                        this.settings.success(this.submitted);
                        this.submitted = [];
                        a("body").removeClass("busyUploading")
                    }
                }
            });
            var b = function(h, k) {
                var j = "" + Math.floor(Math.random() * 10000000000000000000);
                this.id = j;
                this.submit = true;
                this.retries = 0;
                this.form = h;
                this.wrapper = k;
                this.progress = null;
                this.max_retries = 10;
                FileUploadRegistry.forms[j] = this;
                var i = a(h).attr("action");
                if (i.match("\\?")) {
                    i += "&gp.fileupload.id=" + j
                } else {
                    i += "?gp.fileupload.id=" + j
                }
                d.info("Form " + j + " " + i + " registered");
                a(h).addClass("fuForm").attr("id", j).attr("action", i).attr("method", "POST").attr("enctype", "multipart/form-data").wrap("<div></div>");
                a('input[type^="submit"]', h).addClass("fuButton").click(function() {
                    k.submit()
                })
            };
            a.extend(b.prototype, {
                setTimeout: function(h) {
                    if (!h) {
                        h = this.wrapper.settings.stat_delay
                    }
                    setTimeout('FileUploadRegistry.forms["' + this.id + '"].stat()', h)
                },
                stat: function() {
                    var h = this;
                    var i = "?q=" + Math.random() * 10000000000000000000;
                    var j = h.wrapper.settings.max_size_error_label;
                    a(".fuShowNext").prop("disabled", true);
                    a.ajax({
                        type: "GET",
                        dataType: "jsonp",
                        timeout: h.wrapper.settings.stat_timeout,
                        url: h.wrapper.settings.stat_url + h.id + i,
                        success: function(k) {
                            if (k["!error"]) {
                                var l = k["!error"].message || "An error occurred.";
                                h.progress.css("width", "100%").addClass("fuProgressFailure").html(l);
                                h.wrapper.submitNext();
                                return 
                            }
                            if (k.state===-1) {
                                h.progress.css("width", "100%").addClass("fuProgressFailure").html(j);
                                h.wrapper.submitNext();
                                return 
                            }
                            if (k.state === 0) {
                                h.retries += 1;
                                if (h.retries > h.max_retries) {
                                    h.progress.css("width", "100%").addClass("fuProgressFailure").html("Sorry, that upload didn't work. Please try again.");
                                    h.wrapper.submitNext()
                                } else {
                                    h.setTimeout(300)
                                }
                                return 
                            } else {
                                h.retries = 0
                            }
                            if (k.percent >= 0 && k.percent < 100) {
                                h.progress.css("width", k.percent + "%").html(k.percent + "%");
                                h.setTimeout()
                            }
                            if (k.percent === 100) {
                                h.progress.css("width", "100%").addClass("fuProgressSuccess").html(k.percent + "%").removeClass("fuProgress");
                                h.wrapper.submitNext()
                            }
                            a(".fuShowNext").prop("disabled", false)
                        },
                        error: function(k) {
                            h.retries += 1;
                            if (h.retries > 3) {
                                h.progress.css("width", "100%").addClass("fuProgressFailure").html("&nbsp;");
                                h.wrapper.submitNext()
                            } else {
                                h.setTimeout(500)
                            }
                        }
                    })
                }
            });
            return this.each(function(h, j) {
                if (a(j).attr("enctype") === "multipart/form-data") {
                    j = a(j).wrap("<div></div>").parent()
                }
                var k = new g(j);
                k.initialize();
                FileUploadRegistry.wrappers.push(k)
            })
        }
    }(jQuery))
};
SF.Humanize = {
    numberFormat: function(e, b, m, l) {
        var a = e, k = isNaN(b = Math.abs(b)) ? 2: b;
        var h = m === undefined ? ".": m;
        var o = l === undefined ? ",": l, p = a < 0 ? "-": "";
        var g = parseInt(a = Math.abs( + a || 0).toFixed(k), 10) + "", f = (f = g.length) > 3 ? f%3: 0;
        return p + (f ? g.substr(0, f) + o : "") + g.substr(f).replace(/(\d{3})(?=\d)/g, "$1" + o) + (k ? h + Math.abs(a - g).toFixed(k).slice(2) : "")
    },
    intComma: function(b, a) {
        a = a === undefined ? 0 : a;
        return this.numberFormat(b, a, ".", ",")
    },
    intWord: function(a) {
        a = parseInt(a, 10);
        if (a < 1000) {
            return this.numberFormat(a, 0)
        } else {
            if (a < Math.pow(10, 6)) {
                return this.intComma(a / 1000, 1) + "K"
            } else {
                if (a < Math.pow(10, 9)) {
                    return this.intComma(a / 1000000, 1) + "M"
                } else {
                    if (a < Math.pow(10, 12)) {
                        return this.intComma(a / 1000000000, 1) + "B"
                    }
                }
            }
        }
        return this.numberFormat(a, 0)
    }
};
net.sf.ua = navigator.userAgent.toLowerCase();
net.sf.Browser = {
    safari: /webkit/.test(net.sf.ua)&&!/chrome/.test(net.sf.ua),
    chrome: /chrome/.test(net.sf.ua),
    opera: /opera/.test(net.sf.ua),
    msie: /msie/.test(net.sf.ua)&&!/opera/.test(net.sf.ua),
    mozilla: /mozilla/.test(net.sf.ua)&&!/(compatible|webkit)/.test(net.sf.ua)
};
$(function() {
    if ($(".bluesteel .b-hornav").length) {
        $(".b-hornav").droppy({
            speed: 0,
            delay: 0
        })
    }
    if ($("#pg_files.enterprise .b-hornav").length) {
        $(".b-hornav").droppy({
            speed: 0,
            delay: 0
        })
    }
    if ($("#pg_project.enterprise .b-hornav").length) {
        $("#menu_consume_summary, #menu_consume_files, #menu_consume_reviews").remove()
    }
    $(".ie7 .ico[data-icon]").each(function() {
        var c = $(this);
        c.text(c.attr("data-icon"));
        c.css("display", "inline");
        c.css("padding", "0 10px " + Math.max(16, c.height()) + "px 0");
        c.css("height", "0px")
    });
    $("input[placeholder]", this).not("*[type=password]").placeholder();
    $("#messages").notifier();
    if ($("textarea").length) {
        $("textarea").elastic()
    }
    $(".ui-dialog-titlebar-close").click(function(c) {
        $(this).closest(".ui-widget").hide();
        c.preventDefault()
    });
    $(".ui-widget-overlay").on("click", function() {
        $(".ui-dialog").dialog("close")
    });
    var b = $("#busy-spinner");
    b.hide();
    b.ajaxStart(function() {
        if (!window.location.toString().match("upload")) {
            b.show()
        }
    });
    b.ajaxStop(function() {
        b.hide()
    });
    $("#nav-site a").click(function(c) {
        c.preventDefault();
        SF.trackClick(this.href, "GlobalNavbarClick", $(this).text())
    });
    $("#nav-hubs a").click(function(c) {
        c.preventDefault();
        SF.trackClick(this.href, "HubsClick", $(this).text())
    });
    $("#nav-collateral a").click(function(c) {
        c.preventDefault();
        SF.trackClick(this.href, "CollateralClick", $(this).text())
    });
    if (_gaq) {
        $("*[data-event-category]").each(function() {
            var d = $(this).attr("data-event-category"), e = $(this).attr("data-event-action"), c = {};
            if ($(this).attr("data-event-label")) {
                c.label = $(this).attr("data-event-label")
            }
            if ($(this).attr("data-event-value")) {
                c.value = $(this).attr("data-event-value")
            }
            if ($(this).attr("data-event-noninteraction")) {
                c.noninteraction = $(this).attr("data-event-noninteraction")
            }
            if (this.tagName.toLowerCase() === "a") {
                $(this).click(function(f) {
                    f.preventDefault();
                    SF.trackClick(this.href, d, e, c)
                })
            } else {
                _gaq.push(["_trackEvent", d, e, c.label, c.value, c.noninteraction])
            }
        });
        $("[data-project-name]").click(function(c) {
            c.preventDefault();
            var d = this;
            var f = $(this).attr("data-project-name");
            var e = $(this).attr("data-location");
            if ($(d).prop("href")) {
                _gaq.push(["_set", "hitCallback", function() {
                    window.location = $(d).prop("href")
                }
                ])
            }
            _gaq.push(["_trackEvent", "EditorsChoiceCustomLink" + e, "Click", f])
        })
    }
    $("#site-footer nav a").click(function() {
        _gaq.push(["_trackEvent", "FooterClick", $(this).text()])
    });
    var a = $(".jobs-widget");
    if (a.length) {
        a.on("click", "li a", function(d) {
            d.preventDefault();
            var c = $("body").attr("id");
            SF.trackClick(this.href, "JobWidget", "JobClick", {
                label: c
            })
        });
        $("form", a).bind("submit.gaq", function(i) {
            if (!window._gaq) {
                return 
            }
            try {
                var c = $("body").attr("id");
                var g = $(i.target);
                var h = g.find("input[type=text]").val();
                var d = ["_trackEvent", "JobWidget", "Search", c];
                _gaq.push(d)
            } catch (f) {}
        })
    }
});
if ($ && $.tablesorter) {
    $.tablesorter.addParser({
        id: "localnum",
        is: function(a) {
            return false
        },
        format: function(a) {
            return parseInt(a.split(",").join(""), 10)
        },
        type: "numeric"
    })
}
function _(a) {
    return (net.sf.i18n && net.sf.i18n[a]) || a
}
function addCommas(a) {
    return String(a).replace(new RegExp("(\\d)(?=(\\d\\d\\d)+(?!\\d))", "g"), "$1,")
}
function getQuerystring(b, d) {
    if (d === null) {
        d = ""
    }
    b = b.replace(/[\[]/, "\\[").replace(/[\]]/, "\\]");
    var c = new RegExp("[\\?&]" + b + "=([^&#]*)");
    var a = c.exec(window.location.href);
    if (a === null) {
        return d
    } else {
        return a[1]
    }
}
if (SF.seo_debug) {
    $(function() {
        $("a").each(function() {
            var a = $(this).attr("href");
            var d = String(a);
            var b = new RegExp("^(http|https)://(?!(?:sourceforge.net|(.*).sb.sf.net)).*", "gi");
            var c = d.search(b);
            if (c >= 0) {
                $(this).css({
                    background: "#900000",
                    color: "#fff"
                })
            }
        })
    })
};
SF.Widgets = SF.Widgets || {};
SF.Widgets.submitSubscription = function(a) {
    var b = a;
    $(".subscriber-submit", b).attr("disabled", "disabled").fadeTo("fast", 0.25);
    $.ajax({
        url: b.attr("action"),
        type: "POST",
        data: b.serialize(),
        success: function(e, g, f) {
            $("#messages").notify(e.message, {
                status: e.status,
                sticky: true
            });
            if ($(".subscriber-email[type=email]").length > 0) {
                b.find("input[type=email]").clone().attr("type", "hidden").appendTo(b);
                b.find("input[type=email]").remove();
                $(".subscriber-existing-email").children("code").text(b.find(".subscriber-email").val()).end().show()
            }
            var d = b.attr("action");
            var c;
            if (d.search(/unsubscribe$/)===-1) {
                c = "UpdaterSubscribe";
                if (_gaq) {
                    _gaq.push(["_trackEvent", "Updater", c, "{{ p.shortname }}", 1, true])
                }
                if (SF.trackEvent) {
                    SF.trackEvent(c)
                }
                $(".subscriber-submit>span").text("Unsubscribe from Updates");
                b.attr("action", d.replace(/subscribe$/, "unsubscribe"))
            } else {
                c = "UpdaterUnsubscribe";
                if (_gaq) {
                    _gaq.push(["_trackEvent", "Updater", c, "{{ p.shortname }}", 1, true])
                }
                if (SF.trackEvent) {
                    SF.trackEvent(c)
                }
                $(".subscriber-submit>span").text("Subscribe to Updates");
                b.attr("action", d.replace(/unsubscribe$/, "subscribe"))
            }
        },
        error: function(e, f, c) {
            var d = $.parseJSON(e.responseText);
            $("#messages").notify(d.error, {
                status: "error"
            })
        },
        complete: function(c, d) {
            $(".subscriber-submit", b).removeAttr("disabled").fadeTo("fast", 1)
        }
    })
};
SF.Widgets.TopDataWidget = (function(e, d) {
    var a, c;
    var f = {
        randomize: true,
        maxCount: 4,
        data: null,
        fetchImmediately: true,
        rowTemplate: '<li class="item">                     <a href="/projects/project3/?source=recommended_dlp_t4" title="{{shortname}}"><img alt="Icon" src="//ktaylor-4216.sb.sf.net/cdn/allura/icons/pr/project3%40sf.net/chat_48.png"></a>                     <a class="project-name" href="/projects/project3/?source=recommended_dlp_t4" title="Learn more about {{shortname}} ">{{shortname}}</a>                 </li>',
        resultsKey: "download_alltime"
    };
    function b(g, h) {
        a = this;
        a.settings = c = e.extend(f, h);
        a.$ = e(g);
        if (a.$.length !== 1) {
            throw "Element not found: " + g
        }
        a.$.on("top_downloads_widget:data_failed", a.dataFailHandler);
        a.$.on("top_downloads_widget:data_fetched", a.dataSuccessHandler);
        a.template = d.compile(c.rowTemplate);
        if (c.fetchImmediately) {
            a.fetch()
        }
    }
    b.prototype._fetchPromise = function() {
        return e.get("/top/top_data" + document.location.search)
    };
    b.prototype.fetch = function() {
        a._fetchPromise().done(function(g) {
            a.$.trigger("top_downloads_widget:data_fetched", g)
        }).fail(function(g) {
            a.$.trigger("top_downloads_widget:data_failed", g)
        })
    };
    b.prototype.dataSuccessHandler = function(h, g) {
        a.fetchStatus = 1;
        a.data = g;
        var m = g[c.resultsKey];
        var k = "";
        for (var j = 0; j < m.length; j++) {
            var l = m[j];
            k += a.template(l)
        }
        a.rendered = k
    };
    b.prototype.dataFailHandler = function(h, g) {
        a.fetchStatus =- 1;
        a.fetchError = g
    };
    return b
})($, Handlebars);
function initSorting(a) {
    var b = {
        widgets: ["zebra"],
        sortInitialOrder: "desc"
    };
    if ($(a).length) {
        $(a).tablesorter(b);
        $("tr th.header", a).addClass("usersortable").append('<span class="ui-icon fs-toggle"></span>')
    }
}
$(function(d) {
    var b = function(e) {
        d(".stats-picker-table-wrapper").load(e + " #files_list, .files-breadcrumb", function() {
            if (d("#parent_folder").length > 0) {
                var f = e.split("/");
                f.splice( - 2, 2);
                f = f.join("/");
                d("#parent_folder a").prop("href", f)
            }
            d(".stats-picker-table-wrapper #files_list .file a.name").each(function(g, h) {
                var i = d(this).prop("href").split(":")[1].split("download")[0] + "stats/timeline";
                d(this).attr("href", i)
            })
        })
    };
    d("#stat-file-picker").on("click", ".folder a.name, .static a, .stats-picker-table-wrapper .files-breadcrumb a, #parent_folder a", function(g) {
        var f = d(this).attr("href");
        b(f);
        g.preventDefault()
    });
    d("#stat-file-picker-trigger").click(function(l) {
        var m = "";
        var k = 540, n = d(this).children("span").filter(":first").offset().top, i = d(this).children("span").filter(":first").offset().left - k / 2, g = d("#page-body").offset().left;
        if (g > i) {
            var f = g - i + 5;
            i = i + f;
            d(".ui-widget-overlay-top-ext").css("left", k / 2 - f - 16 + "px")
        }
        d("#stat-file-picker").toggle().css({
            top: n + 34 + "px",
            left: i + "px"
        });
        d("#busy-spinner").css({
            top: n + 64 + "px",
            left: i + k / 2 + "px"
        });
        if (d("#files_list").length === 0) {
            if (d(this).attr("data-type") === "file") {
                var j = d(this).attr("href").split("/"), h = j[j.length - 2] + "/";
                m = d(this).attr("href").replace(h, "")
            } else {
                m = d(this).attr("href")
            }
            b(m)
        }
        l.preventDefault()
    });
    if (d(".picker input").length) {
        var c = function() {
            d(".picker input").daterangepicker({
                onOpen: function() {
                    d(".picker input")[0].prev_value = d(".picker input").val()
                },
                onClose: function() {
                    if (d(".picker input")[0].prev_value !== d(".picker input").val()) {
                        d(".picker input").parents("form").submit()
                    }
                },
                rangeSplitter: "to",
                dateFormat: "yy-mm-dd",
                earliestDate: new Date(d(".picker input").attr("data-start-date")),
                latestDate: new Date()
            })
        };
        if (d(".picker input").is(":visible")) {
            c()
        } else {
            var a = false;
            d(".picker input").mouseover(function() {
                if (!a) {
                    c();
                    a = true
                }
            })
        }
    }
    d("#osbc_percent_toggle").change(function() {
        d(".download_count, .download_percent").toggle()
    });
    initSorting("#data_percent, #data_count");
    d("#data_percent, #data_count").bind("sortEnd", function() {
        var e = d(this);
        e.find("tr").each(function(f, g) {
            d(g).find("td:first:not([colspan])").text(f + ".")
        })
    })
});
function initMap() {
    var c = new google.visualization.DataTable({
        cols: [{
            type: "string",
            label: "Country"
        }, {
            type: "number",
            label: "Downloads"
        }
        ]
    });
    $.each(MAP_DATA, function(d, e) {
        c.addRow([e[0], {
            v: e[1],
            f: e[2]
        }
        ])
    });
    var a = document.getElementById("stats-viz");
    var b = new google.visualization.GeoMap(a);
    b.draw(c, {
        colors: [15334113, 2253059],
        width: 698,
        height: 250,
        showLegend: false,
        showZoomOut: false
    })
}
function initOS(d, b) {
    var c = new Raphael("stats-viz", 707, 250), a;
    c.g.txtattr.font = "13px Helvetica, sans-serif";
    c.g.txtattr.color = "#484848";
    a = c.g.piechart(340, 125, 100, d, {
        colors: ["#0685c6", "#87c706", "#c7c706", "#c76606"],
        legend: b,
        legendpos: "east",
        href: []
    });
    a.hover(function() {
        var e = this;
        e.sector.stop();
        (e.blob = e.blob || c.g.blob(e.mx, e.my, e.value)).show();
        if (e.label) {
            e.label[0].stop();
            e.label[0].scale(1.5);
            e.label[1].attr({
                "font-weight": 800
            })
        }
    }, function() {
        var e = this;
        (e.blob = e.blob || c.g.blob(e.mx, e.my, e.value)).hide();
        if (e.label) {
            e.label[0].animate({
                scale: 1
            }, 500, "bounce");
            e.label[1].attr({
                "font-weight": 400
            })
        }
    })
}
function initTimeline(e, g) {
    var b = function(i) {
        return i < 10 ? "0" + i : i
    }, h = function(i) {
        var l = i.getUTCFullYear() + "-" + b(i.getUTCMonth() + 1);
        if (g === "monthly") {
            return l
        } else {
            return l + "-" + b(i.getUTCDate())
        }
    }, k = function(l, i) {
        return h(new Date(l))
    }, a = {
        colors: ["#0685c6", "#2ea8e5", "#5cb8e5", "#99ddff"],
        series: {
            lines: {
                show: true,
                lineWidth: 3
            },
            points: {
                show: true,
                radius: 2,
                fill: true,
                fillColor: "#0685c6"
            },
            label: "Downloads",
            shadowSize: 0
        },
        grid: {
            hoverable: true,
            clickable: true,
            color: "#aaa"
        },
        xaxis: {
            mode: "time",
            color: "#484848",
            minTickSize: [1, "day"],
            tickFormatter: k
        },
        yaxis: {
            min: 0,
            tickDecimals: 0,
            minTickSize: 1
        }
    }, c = null;
    if (g === "monthly") {
        a.xaxis.minTickSize = [1, "month"]
    } else {
        if (g === "weekly") {
            a.xaxis.ticks = [];
            var f = Math.ceil(e.length / 10);
            for (var d = 0; d < e.length; d += f) {
                a.xaxis.ticks.push(e[d][0])
            }
        }
    }
    $.plot($("#grid"), [e], a);
    function j(i, m, l) {
        $('<div class="tooltip ui-corner-all ">' + l + "</div>").css({
            position: "absolute",
            display: "none",
            top: m + 5,
            left: i + 5
        }).appendTo("body").show()
    }
    $("#grid").bind("plothover", function(m, o, l) {
        $("#x").text(o.x.toFixed(2));
        $("#y").text(o.y.toFixed(2));
        if (l) {
            if (c !== l.datapoint) {
                c = l.datapoint;
                $("#tooltip").remove();
                var n = l.datapoint[1].toFixed(2);
                var i = new Date(l.datapoint[0]);
                j(l.pageX, l.pageY, parseInt(n, 10) + " on " + h(i))
            }
        } else {
            $(".tooltip").remove();
            c = null
        }
    })
}
function createDataTable(e) {
    var h = [];
    var d = 0;
    var a = $('<table id="data" class="ui-corner-all"></table>');
    var c = $("<tbody/>");
    var l = $('<thead><tr><th class="usersortable">Date<span class="ui-icon fs-toggle"></span></th></tr></thead>');
    var m = $("<tfoot><tr><td></td></tr></tfoot>");
    for (var f = 0; f < e.length; f++) {
        l.find("tr").append('<th class="usersortable" style="text-align:right">' + e[f].label + '<span class="ui-icon fs-toggle"></span></th>');
        h[f] = 0
    }
    for (f = 0; f < e[0].data.length; f++) {
        var g = $("<tr/>");
        if (f%2 === 0) {
            g.addClass("even")
        }
        for (var b = 0; b < e.length; b++) {
            if (b === 0) {
                g.append("<td>" + new Date(e[b].data[f][0]).format("UTC:yyyy-mm-dd") + "</td>")
            }
            g.append('<td style="text-align:right">' + addCommas(e[b].data[f][1]) + "</td>");
            h[b] += e[b].data[f][1];
            if (f === e[0].data.length - 1) {
                m.find("tr").append('<td style="text-align:right">' + addCommas(h[b]) + "</td>");
                d += h[b]
            }
        }
        c.append(g)
    }
    a.append(l);
    a.append(c);
    a.append(m);
    a.tablesorter({
        widgets: ["zebra"],
        sortList: [[0, 1]]
    });
    var k = $(".sidetab .statpac .title");
    if (k) {
        k.after("<strong>" + addCommas(d) + "</strong>")
    }
    return a
}
function chartProjectStats(b, h, i, f, l, j, c) {
    var k = $("#stats-viz");
    j = j || "day";
    c = c || "%y-%0m-%0d";
    var a = $("#dates").val().split(" to ");
    var d = Date.parse(a[0]).setTimezoneOffset(0);
    var g = Date.parse(a.slice( - 1)[0]).setTimezoneOffset(0);
    h.begin = a[0];
    h.end = a.slice( - 1)[0];
    if (j === "month") {
        d.setUTCDate(1);
        g.setUTCDate(1)
    }
    if (g >= d) {
        $.get(b, h, function(o) {
            if (f(o.data)) {
                k.hide();
                if (!$("#no_results_msg").length) {
                    k.parent().after('<p id="no_results_msg">No results found for the parameters you have selected.</p>')
                } else {
                    $("#no_results_msg").show()
                }
            } else {
                k.show();
                $("#no_results_msg").hide();
                var m = function(q, p) {
                    if (q >= 1000000) {
                        return addCommas(q / 1000000) + "mil"
                    } else {
                        if (q >= 1000) {
                            return addCommas(q / 1000) + "k"
                        }
                    }
                    return q
                }, n = $.plot($("#project_stats_holder"), i(o.data), {
                    colors: ["#0685c6", "#87c706", "#c7c706", "#c76606"],
                    xaxis: {
                        mode: "time",
                        timeformat: c,
                        minTickSize: [1, j],
                        min: d,
                        max: g,
                        color: "#484848"
                    },
                    yaxis: {
                        tickDecimals: 0,
                        min: 0,
                        tickFormatter: m
                    },
                    grid: {
                        hoverable: true,
                        color: "#aaa"
                    },
                    legend: {
                        show: true,
                        margin: 10,
                        backgroundOpacity: 0.5
                    }
                });
                n.draw();
                if ($("#stats-table")) {
                    $("#stats-table").empty().append(createDataTable(i(o.data)))
                }
            }
        })
    } else {
        k.html("<p>The date range you have chosen is not valid.</p>")
    }
    $(".busy").hide();
    var e = null;
    k.bind("plothover", function(o, q, n) {
        if (n) {
            if (e !== n.dataIndex) {
                e = n.dataIndex;
                $("#tooltip").remove();
                var m = n.datapoint[0].toFixed(0), p = n.datapoint[1].toFixed(0);
                $('<div id="tooltip" class="tooltip">' + l(m, p, n) + "</div>").css({
                    position: "absolute",
                    display: "none",
                    top: n.pageY + 5,
                    left: n.pageX + 5
                }).appendTo("body").fadeIn(200)
            }
        } else {
            $("#tooltip").remove();
            e = null
        }
    })
};
var _project;
var get_project = function() {
    if (!_project) {
        _project = $.parseJSON($("meta#webtracker").attr("content"))
    }
    return _project
};
var project = get_project();
var project_name = project ? project.project: "";
var truncation = (function() {
    var c = $("body");
    if (c.attr("id") !== "pg_project") {
        return 
    }
    var e = window.location.pathname;
    function a() {
        var f = {
            after: ".read-more-editorial",
            watch: true
        };
        $("#editorial-review-contnt").append('<a class="read-more-editorial"><b>Read More &#187;</b></a>');
        $("#editorial-review").height("125").dotdotdot(f)
    }
    function d(g) {
        g.preventDefault();
        var f = this;
        if (_gaq) {
            var h = get_project().project;
            if ($(f).attr("href")) {
                _gaq.push(["_set", "hitCallback", function() {
                    window.location = $(f).attr("href")
                }
                ])
            } else {
                $("#editorial-review").trigger("destroy").css({
                    height: "auto"
                })
            }
            _gaq.push(["_trackEvent", "Summary", "EditorialRedirect", h, 1, true])
        }
    }
    a();
    if (!c.hasClass("enterprise")) {
        var b = "/projects/" + project_name + "/editorial/?source=psp";
        $(".read-more-editorial").attr("href", b)
    }
    $("#editorial-review").delegate(".read-more-editorial", "click", d);
    $("#editorial-review").delegate(".read-less-editorial", "click", a);
    return {
        truncateReview: a,
        unTruncateReview: d
    }
})();
$(function() {
    var n = $("#pg_reviews");
    var e = $("#pg_editorial");
    function c() {
        $(".review-toggle").click(function(q) {
            q.preventDefault();
            var t = $(this).prop("href");
            var s = "";
            if (e.length > 0) {
                s = "Editorials"
            } else {
                s = "Reviews"
            }
            if (_gaq) {
                _gaq.push(["_set", "hitCallback", function() {
                    window.location = t
                }
                ]);
                var r = s + "Redirect";
                _gaq.push(["_trackEvent", "Reviews", r, project_name, 1, true])
            }
        })
    }
    if (n.length > 0 || e.length > 0) {
        c()
    }
    var g = $("#pg_project");
    if (!g.length) {
        return 
    }
    function m(q) {
        q.preventDefault();
        $(this).hide();
        $(".collapse").show();
        $("#additional-details > .content").slideDown()
    }
    function j(q) {
        q.preventDefault();
        $(this).hide();
        $(".expand").show();
        $("#additional-details > .content").slideUp()
    }
    function l() {
        var q = $(this);
        if (q.hasClass("editable")) {
            q.addClass("editable-active")
        } else {
            q.parents(".editable").addClass("editable-active")
        }
    }
    function f() {
        var q = $(this);
        if (q.hasClass("editable")) {
            q.removeClass("editable-active")
        } else {
            q.parents(".editable").removeClass("editable-active")
        }
    }
    function d(q) {
        q.preventDefault();
        $("#icon-dialog").dialog("open")
    }
    function i(q) {
        q.preventDefault();
        if (!$("ul.features").length) {
            $(this).parent().append('<ul class="features"></ul>')
        }
        $("ul.features").listedit({
            action: "update",
            placeholder: "Add a new feature",
            html: [$("#antixsrf").html()],
            times: 1
        });
        $(this).hide()
    }
    function k() {
        $(".edit-button").hover(l, f);
        $("#icon-dialog").dialog({
            bgiframe: true,
            autoOpen: false,
            modal: true,
            width: 300,
            buttons: {
                "Save icon": function() {
                    $("form", this).submit()
                },
                Cancel: function() {
                    $(this).dialog("close")
                }
            }
        });
        g.delegate("#project-icon .edit-button", "click", d);
        g.delegate("#edit-features", "click", i);
        g.delegate(".features", "listedit-end", function() {
            $("#edit-features").show()
        });
        $("#edit-features").show()
    }
    function p() {
        if (!$("#additional-details").hasClass("expanded")) {
            if ($.cookie("show_details") === "expand") {
                $("#additional-details > .content").show();
                $(".collapse").show()
            } else {
                $("#additional-details > .content").hide();
                $(".expand").show()
            }
            $("#additional-details header a").on("click", function() {
                $.cookie("show_details", this.className)
            });
            $(".expand").on("click", m);
            $(".collapse").on("click", j)
        }
        if ($("#reviews").length > 0 && $(".helpful-reviews").length === 0) {
            $("#reviews-header").addClass("no-helpful")
        }
        $("#partners, #events, #related-books, #add-ons-plugins .content-group").each(function() {
            var q = $(this);
            q.pajinate({
                items_per_page: q.attr("data-per-page"),
                show_first_last: false,
                nav_label_prev: "Previous",
                abort_on_small_lists: true,
                num_page_links_to_display: 0
            });
            q.find(".page_navigation .ellipse").remove()
        })
    }
    function o() {
        $("#main-content").css("min-height", $("#sidebar").height());
        var q = $("#subscribe-form");
        q.submit(function(r) {
            r.preventDefault();
            $.ajax({
                url: q.attr("action"),
                type: "POST",
                data: q.serialize(),
                success: function(s, u, t) {
                    $("#messages").notify(s.message, {
                        status: s.status
                    });
                    if ($("#subscriber-email[type=email]").length > 0) {
                        $("#subscriber-email").clone().attr("type", "hidden").appendTo(q);
                        q.find("input[type=email]").remove();
                        $("#subscriber-existing-email").children("code").text($("#subscriber-email").val()).end().show()
                    }
                    if (q.attr("action") === "subscribe") {
                        $("#subscriber-submit>span").text("Unsubscribe from Updates");
                        q.attr("action", "unsubscribe")
                    } else {
                        $("#subscriber-submit>span").text("Subscribe to Updates");
                        q.attr("action", "subscribe")
                    }
                },
                error: function(u, v, s) {
                    var t = $.parseJSON(u.responseText);
                    $("#messages").notify(t.error, {
                        status: "error"
                    })
                }
            })
        });
        if ($("#overview .screenshot img").length > 0) {
            $("#overview section.screenshot").nivoSlider({
                effect: "fade",
                animSpeed: 250,
                manualAdvance: true
            })
        }
        $(".enterprise #toggle-description").click(function() {
            $("#toggle-description, #project-description, #project-summary").toggle();
            return false
        })
    }
    if (g.hasClass("admin")) {
        k()
    }
    if (g.hasClass("enterprise")) {
        p()
    }
    o();
    var a = $(".sidebar-widget.metadata");
    if (a.length) {
        a.find("li a").click(function() {
            try {
                var q = $(this).parents("aside").find("h3").text();
                var s = get_project().project;
                _gaq.push(["_trackEvent", "MetadataSidebarClick", s + ": " + q, $(this).attr("href")])
            } catch (r) {}
        })
    }
    var h = $("#project-nav");
    if (h.length) {
        var b = h.offset().top;
        $(window).scroll(function() {
            if ($(window).scrollTop() > b + h.height()) {
                h.addClass("fixed");
                h.slideDown(800)
            } else {
                h.removeClass("fixed")
            }
        });
        $("body").scrollspy({
            offset: 50
        })
    }
});
$(function() {
    var d = $("body").attr("id");
    if ($.inArray(d, ["pg_project", "pg_reviews"])===-1) {
        return 
    }
    if ($("form.rate-this-project").length > 0) {
        var b = $("form.rate-this-project");
        b.find(".star-rating").click(function() {
            $(this).parents("form").submit()
        });
        b.find("input.star").change(function() {
            $(this).parents("form").submit()
        })
    }
    var a = $("#download_countdown");
    var c = $(".data", a);
    var f = new CountdownTimer(new Date(a.attr("data-target-date") * 1000), new Date(a.attr("data-start-date") * 1000), parseInt(a.attr("data-start-count")));
    var e = function(g) {
        c.html(g)
    };
    f.start(null, e)
});
$(window).on("load", function() {
    function a(c, e) {
        if (!c) {
            return 
        }
        var d = decodeURI(c).split("&");
        if (d.length === 1) {
            return 
        }
        e += "=";
        for (var b = 0; b < d.length; ++b) {
            var f = d[b];
            if (f.indexOf(e) === 0) {
                return unescape(f.split("=")[1])
            }
        }
    }
    if (window.FB) {
        FB.Event.subscribe("edge.create", function(b) {
            _gaq.push(["_trackSocial", "facebook", "like", b])
        })
    }
    if (window.twttr) {
        twttr.events.bind("tweet", function(b) {
            if (b) {
                var c;
                if (b.target && b.target.nodeName === "IFRAME") {
                    c = a(b.target.src, "url")
                }
                _gaq.push(["_trackSocial", "twitter", "tweet", c])
            }
        })
    }
});
var CountdownTimer = (function() {
    var b;
    function a(g, d, e, f) {
        b = this;
        this.targetCount = f || 100000000;
        this.startCount = e;
        this.targetDate = g || new Date(2014, 3, 17, 12, 0, 0, 0);
        this.startDate = d;
        this.currentDate = new Date();
        this.currentCount = e;
        this.currentCount = e + this.initialOffset()
    }
    a.prototype.calculateRate = function() {
        return b.countRemaining() / b.msRemaining()
    };
    a.prototype.countRemaining = function() {
        return b.targetCount - b.currentCount
    };
    a.prototype.msRemaining = function() {
        return b.targetDate - b.startDate
    };
    a.prototype.initialOffset = function() {
        return (b.currentDate - b.startDate) * this.calculateRate()
    };
    a.prototype.prettyCount = function() {
        var d = Math.floor(b.currentCount);
        return d.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")
    };
    a.prototype.start = function(d, e) {
        d = d || 1000;
        b.callback = e;
        b.interval = setInterval(function() {
            c(d, e)
        }, d)
    };
    var c = function(d, g) {
        var f = b.calculateRate();
        var e = f * d;
        b.currentCount += e;
        if (g) {
            g(b.prettyCount())
        }
    };
    a.prototype.stop = function() {
        b.interval = clearInterval(b.interval)
    };
    return a
})();
net.sf.Registration = function() {};
net.sf.Registration.prototype = {
    init: function() {
        var a = this;
        $(document).ready(function(b) {
            a.fielderrorHandler(b);
            a.questionHandler(b);
            if (!$(".fielderror").text()) {
                a.passwordHandler(b)
            }
        });
        $("#security_question").change(function(b) {
            a.questionHandler(b)
        });
        $("#username").change(function(b) {
            a.usernameHandler(b)
        });
        $("#password, #password_confirm").change(function(b) {
            a.passwordHandler(b)
        });
        a.optinAllmailingsHandler();
        $("#country").change(function(c) {
            var b = $('#optin input[type="checkbox"]');
            if ($(this).val() === "CA") {
                b.prop("checked", false)
            }
        })
    },
    questionHandler: function(a) {
        if ($("#security_question option:selected").val() === "custom") {
            $("#custom_question").show();
            if (!$(".fielderror").text()) {
                $("#custom_question").focus()
            }
        } else {
            $("#custom_question").hide()
        }
    },
    fielderrorHandler: function(b) {
        var a = $(".fielderror").filter(function() {
            return $(this).text()
        }).first();
        a.parent("p").find("input, select").focus()
    },
    usernameHandler: function(a) {
        var b = $("#username").val();
        if (b) {
            $.ajax({
                url: "username/" + encodeURIComponent(b),
                dataType: "json",
                global: false,
                success: function(c) {
                    $("#username-error").empty()
                },
                error: function(c) {
                    $("#username-error").html("Sorry, that username is invalid or taken")
                }
            })
        }
    },
    passwordHandler: function(a) {
        if ($("#password, #password_confirm").val()) {
            if ($("#password").val().length < 10) {
                $("#password-error").html("Enter a value 10 characters long or more")
            } else {
                if ($("#password").val().length > 128) {
                    $("#password-error").html("Enter a value 128 characters or less. Too long by " + ($("#password").val().length - 128))
                } else {
                    $("#password-error, #password-confirm-error").empty()
                }
            }
            if ($("#password").val() !== $("#password_confirm").val()) {
                $("#password-confirm-error").html("Fields do not match")
            } else {
                $("#password-confirm-error").empty()
            }
        } else {
            $("#password-error, #password-confirm-error").empty()
        }
    },
    optinAllmailingsHandler: function() {
        if (net.sf.country !== "CA") {
            $('#optin input[type="checkbox"]').prop("checked", true)
        }
    }
};
$(function() {
    var b = $("#pg_project,#pg_reviews"), e = 0, a = 0;
    if (!b.length) {
        return 
    }
    function d(j) {
        j.preventDefault();
        var f = $(this), g = f.parent(), h = g.closest(".review"), i = g.serialize();
        i += "&" + f.attr("name") + "=" + f.val();
        $.ajax({
            type: "POST",
            url: g.attr("action"),
            data: i,
            dataType: "json",
            success: function(k, n) {
                var l = parseInt(h.attr("data-helpful-count"), 10), m = l + k.rating;
                $(".count", f).html("(" + m + ")");
                f.attr("disabled", "disabled");
                $("#messages").notify({
                    status: "success",
                    message: "Thank you for rating this review."
                })
            },
            error: function(l, m, k) {
                $("#messages").notify({
                    status: "error",
                    message: "An error occurred while posting your rating of this review."
                })
            }
        })
    }
    function c(j) {
        j.preventDefault();
        var f = $(this), g = f.parent(), h = g.serialize(), i = g.metadata();
        h += "&" + f.attr("name") + "=" + f.val();
        $.ajax({
            type: "POST",
            url: g.attr("action"),
            data: h,
            dataType: "json",
            success: function(k, p) {
                var n = (k.action === "show") ? "hide": "show", m = (n === "show") ? "Not Spam": "Flag as Spam", o = f.parents(".reviewed"), l = $(".review-txt", o);
                l.toggleClass("spam");
                f.attr("name", n).attr("value", m);
                o.effect("highlight", {}, 3000)
            },
            error: function(l, m, k) {
                $("#messages").notify({
                    status: "error",
                    message: "An error occurred while moderating this review."
                })
            }
        })
    }
    $("#reviews-n-ratings .review-txt").truncate(250, {
        trail: [" (<a href='#' class='truncate_show'>more</a>&#8230;)", " (&#8230;<a href='#' class='truncate_hide'>less</a>)"]
    });
    $(".helpful-reviews article p").truncate(350, {
        trail: [" (<a href='#' class='truncate_show'>more</a>&#8230;)", " (&#8230;<a href='#' class='truncate_hide'>less</a>)"]
    });
    $("#featured-reviews .review-txt").truncate(200, {
        trail: [" (<a href='#' class='truncate_show'>more</a>&#8230;)", " (&#8230;<a href='#' class='truncate_hide'>less</a>)"]
    });
    $("#review-form form").reviewform();
    $("form.rate-review input[type=submit], form.rate-review button[type=submit]").on("click", d);
    $("form.moderate-review input[type=submit]").on("click", c)
});
$(function() {
    var b = $("#pg_reviews_new");
    if (!b.length) {
        return 
    }
    var a = $(".fielderror").filter(function() {
        return $(this).text()
    }).first();
    a.parent("div.input").find("input, select").focus();
    $("#delete-review").click(function() {
        if (window.confirm("Are you sure you want to delete your review?")) {
            var e = $(this), d = $('<form method="post" action="' + e.attr("href") + '"></form>'), f = $("meta[name=_visit_cookie]").attr("content"), c = '<input name="_visit_cookie" value="' + f + '" type="hidden" />';
            d.hide().append(c).appendTo("body");
            d.submit()
        }
        return false
    })
});
$(function() {
    if (!$("#project-screenshots, #guide-screenshots").length) {
        return 
    }
    var j = $("#project-screenshots, #guide-screenshots"), h = $(".slider", j), k = $(".thumbnail", j), i = {}, f = 0, b = 3, g = k.length, a = k.first().outerWidth(true), c = b * a;
    j.addClass("enhanced");
    h.prepend('<div title="Previous" class="previous">&#8249;</div>').append('<div title="Next" class="next">&#8250;</div>');
    function d() {
        if (g - f <= b) {
            return 
        }
        f += b;
        $(this).siblings(".strip").animate({
            left: "-=" + c
        });
        j.trigger("screenshots:change")
    }
    function e() {
        if (f < b) {
            return 
        }
        f -= b;
        $(this).siblings(".strip").animate({
            left: "+=" + c
        });
        j.trigger("screenshots:change")
    }
    function l(o) {
        var m = $(".next", j);
        if (g - f <= b) {
            m.addClass("disabled")
        } else {
            m.removeClass("disabled")
        }
        var n = $(".previous", j);
        if (f < b) {
            n.addClass("disabled")
        } else {
            n.removeClass("disabled")
        }
    }
    j.on("click", ".next", d);
    j.on("click", ".previous", e);
    j.on("screenshots:change", l);
    j.trigger("screenshots:change");
    $(".fancybox").fancybox({
        type: "image",
        transitionIn: "none",
        transitionOut: "none",
        titlePosition: "inside",
        titleFormat: function(p, o, m, n) {
            return "<span>Screenshot " + (m + 1) + " / " + o.length + (p.length ? " &nbsp; " + p : "") + "</span>"
        }
    });
    $(".strip", h).css({
        visibility: "visible",
        opacity: "0"
    }).animate({
        opacity: 1
    });
    if (window.location.hash === "#screenshots") {
        j.find("a:first").trigger("click")
    }
});
$(function() {
    if (!$("#pg_directory,#pg_search").length) {
        return 
    }
    var g = $("#advanced_options");
    $("a.advanced").click(function(h) {
        g.toggle();
        h.preventDefault()
    });
    var f = "#result_data";
    $(".paging li a").pjax(f);
    function c() {
        $(".ie6 .browse, .ie7 .browse, .ie8 .browse").toggleClass("iefix")
    }
    function e(i) {
        if ($(".more-info .busy", i).length) {
            var h = $("a", i).eq(0).attr("href").replace(new RegExp("\\?.+$"), "") + "more";
            $.ajax({
                url: h,
                global: false,
                success: function(j) {
                    $(".more-info", i).empty().append(j)
                },
                error: function() {
                    $(".more-info", i).empty().append("Whoops, something broke. We will fix it shortly.")
                },
                complete: function() {
                    c()
                }
            })
        }
    }
    function a() {
        var h = $(this);
        $(".more-info", this).show();
        $(".ui-icon", this).removeClass("ui-icon-circle-triangle-s").addClass("ui-icon-circle-close");
        e(this);
        if (h.next().length) {
            e(h.next())
        }
        c()
    }
    function b() {
        $(".more-info", this).hide();
        $(".ui-icon", this).removeClass("ui-icon-circle-close").addClass("ui-icon-circle-triangle-s");
        c()
    }
    $(f).delegate("li", "result:expand", a).delegate("li", "result:collapse", b).delegate(".ui-icon-circle-close", "click", function() {
        $(this).parent("li").trigger("result:collapse")
    }).delegate(".ui-icon-circle-triangle-s", "click", function() {
        $(this).parent("li").trigger("result:expand")
    });
    $("li", f).hoverIntent({
        over: function(h) {
            $(this).trigger("result:expand")
        },
        out: function() {},
        interval: 250
    });
    if ($(".projects > li").length >= 2) {
        $(".projects > li").slice(0, 2).each(function() {
            e(this)
        })
    }
    $(".projects > li.featured").trigger("result:expand");
    function d() {
        var h = $("#featured").data("nivo:vars").currentSlide;
        var i = $("#featured a.item")[h];
        if (_gaq) {
            _gaq.push(["_trackEvent", "CarouselView", $(i).attr("title")])
        }
    }
    if ($("#featured").length > 0) {
        $("#featured").nivoSlider({
            effect: "fade",
            pauseTime: 5000,
            afterChange: function() {
                d()
            },
            afterLoad: function() {
                d()
            }
        });
        $("#featured a.item").click(function() {
            _gaq.push(["_trackEvent", "CarouselClick", $(this).attr("title")])
        })
    }
});
$(function() {
    if (!$("#pg_enterprise").length) {
        return 
    }
    function a() {
        var b = $("#featured").data("nivo:vars").currentSlide;
        var c = $("#featured a.item")[b];
        if (_gaq) {
            _gaq.push(["_trackEvent", "CarouselView", $(c).attr("title")])
        }
    }
    if ($("#featured").length > 0) {
        $("#featured").nivoSlider({
            effect: "fade",
            pauseTime: 5000,
            afterChange: function() {
                a()
            },
            afterLoad: function() {
                a()
            }
        });
        $("#featured a.item").click(function() {
            _gaq.push(["_trackEvent", "CarouselClick", $(this).attr("title")])
        })
    }
});
(function() {
    function munch(n) {
        var c = document.cookie;
        if (c) {
            var i = c.indexOf(n + "=");
            if (i>-1) {
                var j = c.indexOf(";", i);
                return c.substring(i + n.length + 1, j < 0 ? c.length : j)
            }
        }
    }
    var getElementsByClassName = function(b, a, c) {
        if (document.getElementsByClassName) {
            getElementsByClassName = function(j, m, h) {
                h = h || document;
                var d = h.getElementsByClassName(j), l = (m) ? new RegExp("\\b" + m + "\\b", "i"): null, e = [], g;
                for (var f = 0, k = d.length; f < k; f += 1) {
                    g = d[f];
                    if (!l || l.test(g.nodeName)) {
                        e.push(g)
                    }
                }
                return e
            }
        } else {
            if (document.evaluate) {
                getElementsByClassName = function(o, r, n) {
                    r = r || "*";
                    n = n || document;
                    var g = o.split(" "), p = "", l = "http://www.w3.org/1999/xhtml", q = (document.documentElement.namespaceURI === l) ? l: null, h = [], d, f;
                    for (var i = 0, k = g.length; i < k; i += 1) {
                        p += "[contains(concat(' ', @class, ' '), ' " + g[i] + " ')]"
                    }
                    try {
                        d = document.evaluate(".//" + r + p, n, q, 0, null)
                    } catch (m) {
                        d = document.evaluate(".//" + r + p, n, null, 0, null)
                    }
                    while ((f = d.iterateNext())) {
                        h.push(f)
                    }
                    return h
                }
            } else {
                getElementsByClassName = function(r, u, q) {
                    u = u || "*";
                    q = q || document;
                    var h = r.split(" "), t = [], d = (u === "*" && q.all) ? q.all: q.getElementsByTagName(u), p, j = [], o;
                    for (var i = 0, e = h.length; i < e; i += 1) {
                        t.push(new RegExp("(^|\\s)" + h[i] + "(\\s|$)"))
                    }
                    for (var g = 0, s = d.length; g < s; g += 1) {
                        p = d[g];
                        o = false;
                        for (var f = 0, n = t.length; f < n; f += 1) {
                            o = t[f].test(p.className);
                            if (!o) {
                                break
                            }
                        }
                        if (o) {
                            j.push(p)
                        }
                    }
                    return j
                }
            }
        }
        return getElementsByClassName(b, a, c)
    };
    function parseQueryString() {
        var result = {}, queryString = location.search.substring(1), re = /([^&=]+)=([^&]*)/g, m;
        while ((m = re.exec(queryString))) {
            result[decodeURIComponent(m[1])] = decodeURIComponent(m[2])
        }
        return result
    }
    function hash(d) {
        var a = 1, c = 0, h, o;
        if (d) {
            a = 0;
            for (h = d.length - 1; h >= 0; h--) {
                o = d.charCodeAt(h);
                a = (a<<6 & 268435455) + o + (o<<14);
                c = a & 266338304;
                a = c !== 0 ? a^c>>21 : a
            }
        }
        return a
    }
    var doc = document, body = doc.body, ads = getElementsByClassName("ad"), qs = parseQueryString(), url = "/log/webtracker/", testString = munch("__utmx") || munch("switchboard.test"), domainHash = munch("__utmc") || hash(window.location.hostname), data;
    if (body.id === "error-content") {
        return 
    }
    if (window.JSON && JSON.parse) {
        data = JSON.parse(doc.getElementById("webtracker").content)
    } else {
        data = eval("(" + doc.getElementById("webtracker").content + ")")
    }
    data.url = location.href;
    if (body.id&&!data.action_type) {
        data.action_type = body.id
    }
    if (body.getAttribute("data-template")) {
        data.download_ad_template = body.getAttribute("data-template")
    }
    if (ads.length) {
        var zones = [];
        for (var i = 0, l = ads.length; i < l; i++) {
            var zone = ads[i].id;
            if (zone) {
                zones.push(zone)
            }
        }
        if (zones.length) {
            data.ad_zones = zones
        }
    }
    if (qs.accel_key) {
        data.ticket = qs.accel_key
    }
    if (qs.click_id) {
        data.click_id = qs.click_id
    }
    if (testString && domainHash && data.active_tests && data.active_tests.length) {
        data.tests = testString;
        data.domain_hash = domainHash
    }
    if (qs.accel_key) {
        url += "accel/"
    }
    data.referer = data.referer || doc.referrer;
    $.ajax({
        url: url,
        data: data,
        traditional: true,
        global: false
    })
})();
$(function(a) {
    if (!a("#pg_jobs").length) {
        return 
    }
    a(".page_navigation a").click(function(b) {
        b.preventDefault();
        SF.trackClick(this.href, "JobBoard", "PaginationClick", {
            label: a(this).text()
        })
    });
    a("a#dice_house_ad").click(function(b) {
        b.preventDefault();
        SF.trackClick(this.href, "JobBoard", "HouseAdClick")
    });
    a("#job-search").bind("submit.gaq", function(g) {
        if (!window._gaq) {
            return 
        }
        try {
            var d = a(g.target);
            var f = d.find("input[type=text]").val();
            var b = ["_trackEvent", "JobBoard", "Search", f];
            _gaq.push(b)
        } catch (c) {}
    });
    a("#job-age").bind("submit.gaq", function(g) {
        if (!window._gaq) {
            return 
        }
        try {
            var f = a(g.target);
            var b = f.find("select").val();
            var c = ["_trackEvent", "JobBoard", "AgeSelect", b];
            _gaq.push(c)
        } catch (d) {}
    });
    a("#job-age select").change(function() {
        a(this).parent().submit()
    })
});
$(function() {
    $(".feature .card p").dotdotdot({
        after: "a.read-more",
        callback: function(c, d) {
            if (!c) {
                d.prevObject.addClass("not-truncated")
            }
        }
    });
    var b = $(".faq");
    function a() {
        var e = $(this).parent(), c = "active", d = $(".item." + c, b);
        if (!e.hasClass("active")) {
            e.addClass(c);
            d.removeClass(c);
            $(".answer", d).hide();
            $(".answer", e).show();
            $(".ui-icon", e).removeClass("ui-icon-triangle-1-e").addClass("ui-icon-triangle-1-s");
            $(".ui-icon", d).removeClass("ui-icon-triangle-1-s").addClass("ui-icon-triangle-1-e")
        }
    }
    if (b.length) {
        $(".question", b).click(a);
        $(".question", b).append('<span class="ui-icon ui-icon-triangle-1-e"></span>');
        b.fadeIn("fast")
    }
});
$(function() {
    if (!$("#project-activity").length) {
        return 
    }
    Handlebars.registerHelper("ago", function(f) {
        f = new Date(f);
        var k = Math.max(Math.floor((new Date() - f) / 1000), 0);
        var l = [["year", Math.floor(k / 31536000)], ["month", Math.floor(k / 2592000)], ["week", Math.floor(k / 604800)], ["day", Math.floor(k / 86400)], ["hour", Math.floor(k / 3600)], ["minute", Math.floor(k / 60)]];
        for (var g = 0; g < l.length; g++) {
            var j = l[g];
            var h = j[1] > 1 ? "s": "";
            if (j[1] > 0) {
                return j[1] + " " + j[0] + h + " ago"
            }
        }
        return "just now"
    });
    Handlebars.registerHelper("ifTagged", function(f, g) {
        if (!this.tags || $.inArray(f, this.tags)===-1) {
            return g.inverse(this)
        }
        return g.fn(this)
    });
    Handlebars.registerHelper("cap", function(f) {
        return f.charAt(0).toUpperCase() + f.slice(1)
    });
    Handlebars.registerHelper("activity_obj", function(g) {
        var f = g.activity_name;
        if (g.activity_url) {
            f = '<a href="' + g.activity_url + '">' + f + "</a>"
        }
        return new Handlebars.SafeString(f)
    });
    Handlebars.registerHelper("icon", function(f) {
        if (f) {
            return new Handlebars.SafeString('<img src="' + f + '" class="x16 emboss avatar" width="16" height="16"/>')
        } else {
            return new Handlebars.SafeString('<b data-icon="U" class="ico ico-user x16 emboss avatar"></b>')
        }
    });
    function c(f) {
        var g = $("#follow-action");
        if (f.following) {
            g.addClass("active");
            g.attr("title", g.attr("title").replace("Follow ", "Stop following "))
        } else {
            g.removeClass("active");
            g.attr("title", g.attr("title").replace("Stop following ", "Follow "))
        }
    }
    function b(h) {
        var g = $(this);
        var f = {
            follow: !g.hasClass("active"),
            _session_id: $.cookie("_session_id")
        };
        $.post(g.attr("href"), f, c, "json");
        h.preventDefault();
        return false
    }
    function a(f) {
        if (!f.timeline.length) {
            return 
        }
        $("#project-activity").html(d(f));
        var g = $("#follow-action");
        g.attr("title", (f.following ? "Stop following " : "Follow ") + g.attr("title"));
        g.click(b)
    }
    var e = location.pathname.split("/")[2];
    var d = Handlebars.compile($("#project-activity-template").html());
    $.getJSON("/rest/p/" + e + "/activity/?limit=5", a)
});
